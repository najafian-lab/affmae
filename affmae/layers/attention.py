import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function
from torch.nn.init import xavier_uniform_, constant_

from affmae.ops.deform_reduce_triton import MSDETRPCFunction
from affmae.ops.deform_attn_triton import deform_point_attn

from affmae.layers.pos_embed import pre_table_fp16, pre_table_fp32
from affmae.ops.deform_attn_torch import (
    deform_point_attn_torch,
    msdetrpc_reduce,
    resolve_deform_backend,
)
from affmae.ops.nbhd_attn import nbhd_attn, resolve_backend
from affmae.ops.dispatch import can_use_triton
from affmae.ops.knn import CachedKNN
from affmae.ops.knn_keops import knn_keops

class _TorchMSDETRPC:
    """Adapter giving :func:`msdetrpc_reduce` the ``.apply`` shape of the
    autograd Functions it stands in for."""

    @staticmethod
    def apply(nn_idx, nn_weight, attn, val):
        return msdetrpc_reduce(nn_idx, nn_weight, attn, val)


class ClusterAttention(nn.Module):
    """
    Performs local attention on nearest clusters using Triton kernels.
    """

    def __init__(self, dim, num_heads, proj_drop=0., backend="auto"):
        """
        Args:
            dim: int, embedding width; must divide by ``num_heads``.
            num_heads: int, attention heads.
            proj_drop: float, dropout on the output projection.
            backend: str, one of ``"flash_nbhd_attn"`` (fused Triton, default),
                ``"flash_nbhd_attn_v1"`` (older Triton kernel), ``"cuda"``
                (compiled CLUSTEN extension), or ``"torch"`` to force the
                device-agnostic path. Any Triton choice falls back to ``"torch"``
                automatically when no GPU backend is live. See
                :data:`affmae.ops.nbhd_attn.BACKENDS` for the canonical names.
        Raises:
            ValueError: on an unknown backend name.
        """
        super().__init__()
        # Validate here rather than in forward, so a typo in a config fails at
        # construction instead of thousands of steps into a run.
        resolve_backend(backend)
        self.dim = dim
        self.pos_dim = 2
        self.num_heads = num_heads

        head_dim = dim // num_heads
        assert head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.scale = head_dim ** -0.5  # softmax normalization factor

        # Cloned into buffers so `model.to(device)` moves them and two models
        # never share storage. Read from the module-level tables built at import;
        # nothing rebinds those names any more.
        self.register_buffer('pre_table_fp32', pre_table_fp32.clone())
        self.register_buffer('pre_table_fp16', pre_table_fp16.clone())

        # match names of the CUDA module for weight copying convenience
        self.q = nn.Linear(dim, dim)
        self.q.inp = 'attn'
        self.kv = nn.Linear(dim, 2 * dim)
        self.kv.inp = 'attn'

        self.blank_k = nn.Parameter(torch.randn(dim) * 0.2)
        self.blank_v = nn.Parameter(torch.randn(dim) * 0.2)

        # pos_embed takes 5-dim features -> per-head bias
        self.pos_embed = nn.Linear(self.pos_dim+3, num_heads)
        self.pos_embed.inp = 'norm'

        # projection layer
        self.proj = nn.Linear(dim, dim)
        self.proj.inp = 'norm'
        self.proj_drop = nn.Dropout(proj_drop)

        self.backend = backend

    def _heads(self, x: torch.Tensor, H: int):
        """[B,N,C] -> [B,H,N,C//H] contiguous"""
        B, N, C = x.shape
        C_h = C // H
        return x.view(B, N, H, C_h).permute(0, 2, 1, 3).contiguous()

    def _make_pos_bias(self, pe_idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Args:
            pe_idx: [B, N, M] long indices into the relative-position table.
            dtype: torch.dtype, selects the fp16 or fp32 table buffer.
        Returns:
            pos_bias: [B, H, N, M] additive per-head bias.
        Raises:
            ValueError: on a dtype with no precomputed table.
        """
        B, N, M = pe_idx.shape
        device = pe_idx.device

        # Ensure data types; move to correct device
        if dtype == torch.float16:
            pre_table = self.pre_table_fp16
        elif dtype == torch.float32:
            pre_table = self.pre_table_fp32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        # [T*T, 5] -> [T*T, H]
        pe_table = self.pos_embed(pre_table)              # [T*T, H]
        H = pe_table.shape[-1]

        # Efficient gather along dim=0 with a 1D index
        flat_idx = pe_idx.reshape(-1).to(dtype=torch.long, device=device)  # [BNM]
        pe = torch.index_select(pe_table, dim=0, index=flat_idx)           # [BNM, H]

        # Reshape and permute to [B,H,N,M]
        pos_bias = pe.view(B, N, M, H).permute(0, 3, 1, 2).contiguous()
        return pos_bias

    def forward(self, feat, member_idx, cluster_mask, pe_idx, global_attn: bool):
        """
        Args:
            feat         : [B,N,C], token features
            member_idx   : [B,N,M], neighbor indices per (b,n)
            cluster_mask : [B,N,M] (0/1 or bool) or None
            pe_idx       : [B,N,M], integer indices into pre_table grid
            global_attn  : bool, if True do dense attention (reference path)
        Returns:
            feat_out     : [B,N,C]
        """
        B, N, C = feat.shape
        H = self.num_heads
        C_h = C // H
        dtype = feat.dtype

        # QKV
        q = self.q(feat)                                  # [B,N,C]
        kv = self.kv(feat)                                # [B,N,2C]
        qh = self._heads(q, H)                            # [B,H,N,C_h]
        kvh = self._heads(kv, H)                          # [B,H,N,2*C_h]
        k, v = kvh.split([C_h, C_h], dim=-1)              # each [B,H,N,C_h]

        # create relative position bias
        pos_bias = self._make_pos_bias(pe_idx, dtype)     # [B,H,N,M] or [B,H,N,N] if global

        if global_attn:
            # sanity check just in case
            if N != pos_bias.shape[-1]:
                raise ValueError(f"Global attention requires M == N, but got M={pos_bias.shape[-1]} and N={N}")

            # pad it for blank k and v tokens since we can't do conditionals in score function
            pos_bias = F.pad(pos_bias, (0, 1), "constant", 0.0)


            # get k and v blank tokens, expand to batch dim
            blank_k_h = self.blank_k.view(1, H, 1, C_h).expand(B, -1, -1, -1) # [B,H,1,C_h]
            blank_v_h = self.blank_v.view(1, H, 1, C_h).expand(B, -1, -1, -1) # [B,H,1,C_h]

            # concatenate to k and v, acting as an "extra token"
            k_full = torch.cat([k, blank_k_h], dim=2) # [B,H, N+1, C_h]
            v_full = torch.cat([v, blank_v_h], dim=2) # [B,H, N+1, C_h]

            # essentially do it like cross attention with the k and v having an extra token
            out = F.scaled_dot_product_attention(
                qh,                 # unscaked
                k_full,             # full keys
                v_full,             # full values
                attn_mask=pos_bias # Additive float mask
            )
        else:
            # create neighborhood mask (if provided)
            mask = None
            if cluster_mask is not None:
                # Keep the head dim at 1 and let the consumer broadcast: every backend
                # below handles [B,1,N,M] (the Triton wrappers expand internally, the
                # CUDA path relies on masked_fill broadcasting). Materializing
                # [B,H,N,M] here would just be thrown away and re-cast downstream.
                mask_bool = cluster_mask if cluster_mask.dtype == torch.bool else (cluster_mask > 0)
                mask = mask_bool.unsqueeze(1)  # [B,1,N,M]

            # split blank tokens into heads
            blank_k_hd = self.blank_k.view(H, C_h).contiguous()
            blank_v_hd = self.blank_v.view(H, C_h).contiguous()

            # One operator, four backends; see affmae/ops/nbhd_attn.py. It
            # falls back to torch when Triton cannot run on these tensors.
            out = nbhd_attn(
                qh, k, v, member_idx, float(self.scale),
                bias=pos_bias,               # [B,H,N,M]
                mask=mask,                   # None, [B,1,N,M] or [B,H,N,M]
                blank_k=blank_k_hd, blank_v=blank_v_hd,   # [H,D]
                backend=self.backend,
            )

        # [B,H,N,C_h] -> [B,N,C]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class _DeformablePointAttention(nn.Module):
    """
    Shared deformable point-attention implementation for cross/self-attention.

    Args:
        d_model: int, embedding width.
        n_heads: int, number of attention heads.
        n_points: int, sampling points per head.
        grid_h: int, dense KNN lookup grid height. Must equal the patch-grid
            height (``img_size // patch_size``): the KNN operator clamps KV coordinates
            to the grid extent, so an undersized grid silently collapses every
            coordinate beyond it onto the last row/column. Required by design --
            a defaulted grid size is what made that failure invisible.
        grid_w: int, dense KNN lookup grid width. Same constraint.
        shepard_power: float, distance scale for Shepard interpolation.
        shepard_power_learnable: bool, whether shepard power is learnable.
        deform_backend: str, one of {"csr_knn_cached", "atomic", "unfused",
            "cuda"}. "csr_cached" is a legacy alias of "csr_knn_cached".
    """
    def __init__(
        self,
        d_model,
        n_heads,
        n_points,
        grid_h,
        grid_w,
        shepard_power=3.0,
        shepard_power_learnable=True,
        deform_backend="auto",
    ):
        super().__init__()
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"grid_h/grid_w must be positive, got {grid_h}x{grid_w}.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.c_ = d_model // n_heads
        self.grid_h = grid_h
        self.grid_w = grid_w
        self._deform_backend = resolve_deform_backend(deform_backend)
        # The neighbour table is a reusable operator, not something this module
        # should re-implement. CachedKNN picks Triton on CUDA and the torch
        # reference elsewhere, which is what lets the fused backends run on CPU.
        self.knn = CachedKNN(grid_h=grid_h, grid_w=grid_w)

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        if shepard_power_learnable:
            self.shepard_power = nn.Parameter(torch.tensor(float(shepard_power)))
        else:
            self.register_buffer("shepard_power", torch.tensor(float(shepard_power)))

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize offsets on radial directions per head/point.
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 2).repeat(1, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        # Standard projection initialization for attention/value/output layers.
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def _get_nn4(self, kv_pos: torch.Tensor, cache_key=None):
        """Build or fetch the dense top-4 neighbour table for KV positions.

        Args:
            kv_pos: [B, Nk, 2] KV token coordinates.
            cache_key: hashable identity for these positions within the active
                cache scope, or None to always recompute. Deliberately supplied
                by the caller: keying on ``kv_pos.data_ptr()`` is unsafe because
                CUDA reuses addresses after a free.
        Returns:
            [B, grid_h * grid_w, 4] int table of KV indices.
        """
        return self.knn(kv_pos, cache_key=cache_key)

    def _forward_unfused(self, query, query_pos, key_value, kv_pos, use_cuda_kernel: bool = False):
        """
        Reference unfused deformable attention path.

        Args:
            query: [B, Nq, C], query tokens.
            query_pos: [B, Nq, 2], query coordinates.
            key_value: [B, Nk, C], KV tokens.
            kv_pos: [B, Nk, 2], KV coordinates.
        Returns:
            output tensor [B, Nq, C].
        """
        b, n_q, c = query.shape
        _, n_kv, _ = key_value.shape
        h = self.n_heads
        k = self.n_points
        c_ = self.c_

        # Project values and build per-head sampling/attention tensors.
        values = self.value_proj(key_value).reshape(b, n_kv, h, c_).permute(0, 2, 1, 3).reshape(b * h, n_kv, c_)
        sampling_offsets = self.sampling_offsets(query).view(b, n_q, h, k, 2)
        attention_weights = F.softmax(self.attention_weights(query).view(b, n_q, h, k), dim=-1)
        sampling_locations = query_pos.unsqueeze(2).unsqueeze(3) + sampling_offsets
        sampling_locations = sampling_locations.permute(0, 2, 1, 3, 4).reshape(b * h, n_q * k, 2).contiguous()
        kv_pos_locations = kv_pos.unsqueeze(1).expand(-1, h, -1, -1).reshape(b * h, n_kv, 2).contiguous()
        nb_idx_real = knn_keops(sampling_locations, kv_pos_locations, k=4)

        # Gather neighbor coordinates and compute Shepard interpolation weights.
        nb_idx_real_expanded = nb_idx_real.unsqueeze(-1).expand(-1, -1, -1, 2)
        kv_pos_expanded = kv_pos_locations.unsqueeze(1).expand(-1, n_q * k, -1, -1)
        nb_kv_pos = torch.gather(kv_pos_expanded, 2, nb_idx_real_expanded)
        nb_token_rel_pos = nb_kv_pos - sampling_locations.unsqueeze(2)

        dist = torch.norm(nb_token_rel_pos, dim=-1, p=2) + 1e-6
        power = F.relu(self.shepard_power) + 1e-6
        logits = -power * dist.unsqueeze(-1)
        nn_weights = F.softmax(logits, dim=-2)
        nn_idx_reshaped = nb_idx_real.reshape(b * h, n_q, k, 4)
        nn_weights_reshaped = nn_weights.reshape(b * h, n_q, k, 4)
        attention_weights_reshaped = attention_weights.permute(0, 2, 1, 3).reshape(b * h, n_q, k)
        # Fused reduce over neighbors and sampling points.
        if use_cuda_kernel:
            try:
                from affmae.ops.cuda_ext.clusten import MSDETRPCFunction as ClustenMSDETRPCFunction
            except Exception as exc:
                raise RuntimeError(
                    "Deformable attention backend 'cuda' needs the CLUSTEN CUDA "
                    "extension, which is not built. Either use the default "
                    "backend='csr_knn_cached' (Triton, faster), or build it:\n"
                    "    cd affmae/ops/cuda_ext/src && python setup.py build_ext --inplace\n"
                    "See affmae/ops/cuda_ext/README.md."
                ) from exc
            msdetrpc_impl = ClustenMSDETRPCFunction
        elif not can_use_triton(values):
            # msdetrpc is itself a Triton kernel, so "unfused" was never
            # Triton-free. Same reduce in plain torch, exact to ~5e-7.
            msdetrpc_impl = _TorchMSDETRPC
        else:
            msdetrpc_impl = MSDETRPCFunction
        if use_cuda_kernel:
            output = msdetrpc_impl.apply(
                nn_idx_reshaped,
                nn_weights_reshaped.to(torch.float32),
                attention_weights_reshaped.to(torch.float32),
                values.to(torch.float32),
            ).reshape(b, h, n_q, c_).permute(0, 2, 1, 3).reshape(b, n_q, c)
            output = output.to(query.dtype)
        else:
            output = msdetrpc_impl.apply(
                nn_idx_reshaped,
                nn_weights_reshaped,
                attention_weights_reshaped,
                values,
            ).reshape(b, h, n_q, c_).permute(0, 2, 1, 3).reshape(b, n_q, c)
        return self.output_proj(output)

    def _forward_impl(self, query, query_pos, key_value, kv_pos, cache_key=None):
        """
        Backend-dispatched deformable point-attention forward pass.

        Args:
            query: [B, Nq, C], query tokens.
            query_pos: [B, Nq, 2], query coordinates.
            key_value: [B, Nk, C], KV tokens.
            kv_pos: [B, Nk, 2], KV coordinates.
            use_decoder_self_cache: bool, enable stage-scoped KNN cache.
        Returns:
            output tensor [B, Nq, C].
        """
        b, n_q, c = query.shape
        _, n_kv, _ = key_value.shape
        if n_q <= 0:
            raise ValueError(f"Deformable attention requires Nq > 0, got {n_q}.")
        if n_kv < 4:
            raise ValueError(f"Deformable attention requires Nk >= 4 for dense_top4_knn uint16 path, got {n_kv}.")
        h = self.n_heads
        k = self.n_points

        with record_function("aff.decoder.deform.proj"):
            sampling_offsets = self.sampling_offsets(query).view(b, n_q, h, k, 2).contiguous()
            attn_logits = self.attention_weights(query).view(b, n_q, h, k).contiguous()
            values = self.value_proj(key_value).view(b, n_kv, h, self.c_).contiguous()
        with record_function("aff.decoder.deform.knn_table"):
            nn4 = self._get_nn4(kv_pos, cache_key=cache_key)
        with record_function("aff.decoder.deform.kernel"):
            if not can_use_triton(values, nn4):
                # No GPU backend: pure-torch path, exact to ~1e-5 of the kernel.
                out = deform_point_attn_torch(
                    query_pos.contiguous(), kv_pos.contiguous(),
                    sampling_offsets, attn_logits, values,
                    self.shepard_power, nn4, self.grid_h, self.grid_w)
            else:
                out = deform_point_attn(
                    query_pos=query_pos.contiguous(),
                    kv_pos=kv_pos.contiguous(),
                    sampling_offsets=sampling_offsets,
                    attn_logits=attn_logits,
                    values=values,
                    tau=self.shepard_power,
                    nn4_idx=nn4,
                    grid_h=self.grid_h,
                    grid_w=self.grid_w,
                    backend=self._deform_backend,
                )
        with record_function("aff.decoder.deform.output_proj"):
            out = out.reshape(b, n_q, c).to(query.dtype)
            out = self.output_proj(out)
        return out


class DeformableCrossAttention(_DeformablePointAttention):
    """Deformable attention from query tokens to encoder features.

    Args:
        cache_key: hashable identity of ``kv_pos`` within the active cache
            scope. The KV positions differ between stages but are fixed across
            the blocks of one stage, so passing a key lets those blocks share a
            single KNN table. None recomputes per block.
    """

    def forward(self, query, query_pos, key_value, kv_pos, cache_key=None):
        if self._deform_backend == "unfused":
            return self._forward_unfused(query, query_pos, key_value, kv_pos, use_cuda_kernel=False)
        if self._deform_backend == "cuda":
            return self._forward_unfused(query, query_pos, key_value, kv_pos, use_cuda_kernel=True)
        return self._forward_impl(query, query_pos, key_value, kv_pos,
                                  cache_key=cache_key)


class DeformableSelfAttention(_DeformablePointAttention):
    """Deformable self-attention over query tokens.

    Args:
        cache_key: hashable identity of ``pos`` within the active cache scope.
            Every block in a decoder stage sees the same query positions, so
            passing a key lets them share one KNN table. None recomputes.
    """

    def forward(self, x, pos, cache_key=None):
        if self._deform_backend == "unfused":
            return self._forward_unfused(x, pos, x, pos, use_cuda_kernel=False)
        if self._deform_backend == "cuda":
            return self._forward_unfused(x, pos, x, pos, use_cuda_kernel=True)
        return self._forward_impl(x, pos, x, pos, cache_key=cache_key)
