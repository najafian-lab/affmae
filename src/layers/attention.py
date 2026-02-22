import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.amp import custom_fwd, custom_bwd
from torch.testing import assert_close
from torch.nn.init import xavier_uniform_, constant_

from .kernels.local_attn import launch_flash_nbhood_attn_fwd, launch_flash_nbhood_attn_bwd
from .kernels.local_attn import FlashLocalAttentionFunction
from .kernels.msdetrpc import MSDETRPCFunction
from .kernels.fused_deform_attn import fused_deformable_attention
from .kernels.fused_deform_attn import fused_knn_deformable_self_attention, fused_knn_deformable_cross_attention

from src.utils.pos_embed import pre_table_fp16, pre_table_fp32
from src.layers.kernels.knn import knn as triton_knn

class ClusterAttention(nn.Module):
    """
    Performs local attention on nearest clusters using Triton kernels.
    """

    def __init__(self, dim, num_heads, proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.pos_dim = 2
        self.num_heads = num_heads

        head_dim = dim // num_heads
        assert head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.scale = head_dim ** -0.5  # softmax normalization factor

        global pre_table_fp32, pre_table_fp16
        
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

        # Cache for inference optimization: store pe_table per (device, dtype)
        # Only used when model is in eval mode (not training)
        self._pe_table_cache = {}

    def clear_pe_table_cache(self):
        """Clear the cached pe_table. Useful when switching between training/eval modes."""
        self._pe_table_cache.clear()

    def _heads(self, x: torch.Tensor, H: int):
        """[B,N,C] -> [B,H,N,C//H] contiguous"""
        B, N, C = x.shape
        C_h = C // H
        return x.view(B, N, H, C_h).permute(0, 2, 1, 3).contiguous()

    def _make_pos_bias(self, pe_idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:  # , pre_table: torch.Tensor) -> torch.Tensor:
        """
        pe_idx : [B,N,M] (long)
        dtype: torch.dtype
        pre_table: [T*T, 5]  -> pos_embed -> [T*T, H]
        returns pos_bias: [B,H,N,M]
        """
        global pre_table_fp16, pre_table_fp32

        B, N, M = pe_idx.shape
        device = pe_idx.device

        # Ensure data types; move to correct device
        if dtype == torch.float16:
            pre_table_fp16 = self.pre_table_fp16
            pre_table = pre_table_fp16
        elif dtype == torch.float32:
            pre_table_fp32 = self.pre_table_fp32
            pre_table = pre_table_fp32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        # Move pre_table to correct device if needed
        if pre_table.device != device:
            pre_table = pre_table.to(device)

        # Cache pe_table during inference (when not training)
        # Key by (device, dtype) to handle different devices/dtypes
        cache_key = (device, dtype)
        
        if not self.training and cache_key in self._pe_table_cache:
            # Use cached pe_table during inference
            pe_table = self._pe_table_cache[cache_key]
            # Ensure cached tensor is on the correct device (should already be, but verify)
            if pe_table.device != device:
                pe_table = pe_table.to(device)
                self._pe_table_cache[cache_key] = pe_table
        else:
            # Compute pe_table: [T*T, 5] -> [T*T, H]
            pe_table = self.pos_embed(pre_table)              # [T*T, H]
            
            # Cache for inference mode only (when weights are frozen)
            if not self.training:
                self._pe_table_cache[cache_key] = pe_table
        
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
            pre_table    : [T*T,5], precomputed relative-pos features (global)
            global_attn  : bool, if True do dense attention (reference path)
        Returns:
            feat_out     : [B,N,C]
        """
        B, N, C = feat.shape
        H = self.num_heads
        C_h = C // H
        device = feat.device
        dtype = feat.dtype
        # get member idx
        M = member_idx.shape[-1]

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
                # expand across heads
                mask = (cluster_mask > 0) if cluster_mask.dtype != torch.bool else cluster_mask
                mask = mask.unsqueeze(1).expand(-1, H, -1, -1).contiguous()

            # split blank tokens into heads
            blank_k_hd = self.blank_k.view(H, C_h).contiguous()
            blank_v_hd = self.blank_v.view(H, C_h).contiguous()
            
            # compute local attention using custom triton kernel
            out = FlashLocalAttentionFunction.apply(
                qh, k, v,
                member_idx,                  # [B,N,M] int32
                pos_bias,                    # [B,H,N,M]
                mask,                        # None or [B,H,N,M]
                blank_k_hd, blank_v_hd,      # [H,D]
                float(self.scale)
            )

        # [B,H,N,C_h] -> [B,N,C]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class DeformableCrossAttention(nn.Module):
    """
    Single-level deformable cross attention block.
    
    This block performs deformable attention between query and key/value features
    without the multi-level hierarchy complexity.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_points: Number of sampling points per head
        shepard_power: Power for Shepard interpolation (default: 3.0)
        shepard_power_learnable: Whether to make shepard_power learnable
        use_fused: Whether to use fused Triton kernel (faster, default=True)
        use_fused_knn: Whether to use fully fused KNN+attention kernel (fastest, default=True)
    """
    
    def __init__(self, d_model, n_heads, n_points, shepard_power=3.0, shepard_power_learnable=True, use_fused=False, use_fused_knn=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.c_ = d_model // n_heads
        self.use_fused = use_fused
        self.use_fused_knn = use_fused_knn
        
        # learnable projections
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        initial_power = 1.0 
        # define it as a learnable parameter
        self.shepard_power = nn.Parameter(torch.tensor(initial_power))
            
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters with proper initialization schemes."""
        # initialize sampling offsets to zero
        constant_(self.sampling_offsets.weight.data, 0.)
        
        # initialize sampling offsets bias with circular patterns
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 2).repeat(1, self.n_points, 1)
        
        # scale offsets by point index
        for i in range(self.n_points):
            grid_init[:, i, :] *= i + 1
            
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        # initialize attention weights
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        
        # initialize value and output projections
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
    
    def forward(self, query, query_pos, key_value, kv_pos):
        """
        Forward pass for single-level deformable cross attention.
        
        Args:
            query: Query features [b, n_q, c]
            query_pos: Query positions [b, n_q, 2]
            key_value: Key/value features [b, n_kv, c]
            kv_pos: Key/value positions [b, n_kv, 2]
            
        Returns:
            output: Attended features [b, n_q, c]
        """
        b, n_q, c = query.shape
        _, n_kv, _ = key_value.shape
        h = self.n_heads
        k = self.n_points
        c_ = self.c_
        
        # generate sampling offsets and attention weights
        sampling_offsets = self.sampling_offsets(query).view(b, n_q, h, k, 2)  # [b, n_q, h, k, 2]
        attention_weights = self.attention_weights(query).view(b, n_q, h, k)   # [b, n_q, h, k]
        attention_weights = F.softmax(attention_weights, dim=-1)  # [b, n_q, h, k]
        
        if self.use_fused_knn and k == 4:
            # Project values: [b, n_kv, c] -> [b, n_kv, h, c_]
            values = self.value_proj(key_value).view(b, n_kv, h, c_)
            
            # Ensure contiguous tensors for optimal kernel performance
            # Non-contiguous tensors cause stride issues and potential slowdowns
            query_pos = query_pos.contiguous()
            kv_pos = kv_pos.contiguous()
            sampling_offsets = sampling_offsets.contiguous()
            attention_weights = attention_weights.contiguous()
            values = values.contiguous()
            
            # Fully fused kernel: KNN + Shepard weighting + gather in one pass
            # Positions are NOT expanded across heads
            output = fused_knn_deformable_cross_attention(
                query_pos,          # [b, n_q, 2] - NOT expanded
                kv_pos,             # [b, n_kv, 2] - shared across heads
                sampling_offsets,   # [b, n_q, h, k, 2]
                attention_weights,  # [b, n_q, h, k]
                values,             # [b, n_kv, h, c_]
                self.shepard_power,
                D=2
            )  # Returns [b, n_q, h, c_]
            
            # Reshape to [b, n_q, c]
            output = output.reshape(b, n_q, c)
        else:
            # project values with old layout
            values = self.value_proj(key_value).reshape(b, n_kv, h, c_).permute(0, 2, 1, 3).reshape(b*h, n_kv, c_)
            
            # compute sampling locations: query_pos + sampling_offsets
            sampling_locations = query_pos.unsqueeze(2).unsqueeze(3) + sampling_offsets  # [b, n_q, h, k, 2]
            sampling_locations = sampling_locations.permute(0, 2, 1, 3, 4).reshape(b*h, n_q*k, 2).contiguous()  # [b*h, n_q*k, 2]
            
            # for single-level attention, compute KNN directly instead of spatial lookup
            # this is simpler and avoids the complex spatial indexing
            # [b, h, n_kv, 2].reshape(b*h, n_kv, 2)
            kv_pos_locations = kv_pos.unsqueeze(1).expand(-1, h, -1, -1).reshape(b*h, n_kv, 2).contiguous()
            nb_idx_real = triton_knn(sampling_locations, kv_pos_locations, k=4) # [b*h, n_q*k, 4]

            if self.use_fused:
                # Use fused Triton kernel (faster, lower memory bandwidth)
                # Reshape for fused kernel: expects [B, N, K] indices, [B, N] attention weights
                nb_idx_fused = nb_idx_real.reshape(b*h, n_q*k, 4)  # [b*h, n_q*k, 4]
                attn_weights_fused = attention_weights.permute(0, 2, 1, 3).reshape(b*h, n_q*k)  # [b*h, n_q*k]
                
                # Fused kernel computes: distance -> softmax -> weighted gather in one pass
                output = fused_deformable_attention(
                    sampling_locations,      # [b*h, n_q*k, 2]
                    kv_pos_locations,         # [b*h, n_kv, 2]
                    nb_idx_fused,             # [b*h, n_q*k, 4]
                    self.shepard_power,       # scalar
                    attn_weights_fused,       # [b*h, n_q*k]
                    values,                   # [b*h, n_kv, c_]
                    D=2
                )  # [b*h, n_q*k, c_]
                
                # Reshape output: [b*h, n_q*k, c_] -> [b*h, n_q, k, c_] -> sum over k -> [b, n_q, c]
                output = output.reshape(b*h, n_q, k, c_).sum(dim=2)  # [b*h, n_q, c_]
                output = output.reshape(b, h, n_q, c_).permute(0, 2, 1, 3).reshape(b, n_q, c)
            else:
                # PyTorch reference implementation (slower, more memory)
                # expand for gather
                # [b*h, n_q*k, 4] -> [b*h, n_q*k, 4, 2]
                nb_idx_real_expanded = nb_idx_real.unsqueeze(-1).expand(-1, -1, -1, 2)
                # [b*h, n_kv, 2] -> [b*h, n_q*k, n_kv, 2]
                kv_pos_expanded = kv_pos_locations.unsqueeze(1).expand(-1, n_q*k, -1, -1)
                # gather the absolute positions of the k nearest tokens to each virtual token
                # [b*h, n_q*k, 4, 2]
                nb_kv_pos = torch.gather(kv_pos_expanded, 2, nb_idx_real_expanded)
                # relative position of k nearest tokens to each virtual token
                # [b*h, n_q*k, 4, 2] - [b*h, n_q*k, 1, 2] = [b*h, n_q*k, 4, 2]
                nb_token_rel_pos = nb_kv_pos - sampling_locations.unsqueeze(2)

                # add 1e-6 inside the sqrt for numerical stability
                dist = torch.norm(nb_token_rel_pos, dim=-1, p=2) + 1e-6 
                # dist shape: [b*h, n_q*k, 4]

                # ensure power is positive
                power = F.relu(self.shepard_power) + 1e-6

                # add a .unsqueeze(-1) to 'dist' to match the 
                # [..., 1] shape of the MLP output
                logits = -power * dist.unsqueeze(-1)
                # logits shape: [b*h, n_q*k, 4, 1]

                nn_weights = F.softmax(logits, dim=-2)
                
                # reshape for MSDETRPC function
                # original: [b*h, -1, k*l, 4] where l=1 for single level
                nn_idx_reshaped = nb_idx_real.reshape(b*h, n_q, k, 4)  # [b*h, n_q, k, 4]
                nn_weights_reshaped = nn_weights.reshape(b*h, n_q, k, 4)  # [b*h, n_q, k, 4]
                attention_weights_reshaped = attention_weights.permute(0, 2, 1, 3).reshape(b*h, n_q, k)  # [b*h, n_q, k]
                
                # apply deformable attention using CUDA kernel
                output = MSDETRPCFunction.apply(
                    nn_idx_reshaped,
                    nn_weights_reshaped, 
                    attention_weights_reshaped,
                    values
                ).reshape(b, h, n_q, c_).permute(0, 2, 1, 3).reshape(b, n_q, c)
        
        # final output projection
        output = self.output_proj(output)
        
        return output


class DeformableSelfAttention(nn.Module):
    """
    Single-level deformable self attention block.
    
    This block performs deformable self attention where queries, keys, and values
    all come from the same input features.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_points: Number of sampling points per head
        shepard_power: Power for Shepard interpolation (default: 3.0)
        shepard_power_learnable: Whether to make shepard_power learnable
        use_fused: Whether to use fused Triton kernel (faster, default=True)
        use_fused_knn: Whether to use fully fused KNN+attention kernel (fastest, default=True)
    """
    
    def __init__(self, d_model, n_heads, n_points, shepard_power=3.0, shepard_power_learnable=True, use_fused=False, use_fused_knn=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.c_ = d_model // n_heads
        self.use_fused = use_fused
        self.use_fused_knn = use_fused_knn
        
        # learnable projections
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        initial_power = 1.0 
        # define it as a learnable parameter
        self.shepard_power = nn.Parameter(torch.tensor(initial_power))
            
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters with proper initialization schemes."""
        # initialize sampling offsets to zero
        constant_(self.sampling_offsets.weight.data, 0.)
        
        # initialize sampling offsets bias with circular patterns
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 2).repeat(1, self.n_points, 1)
        
        # scale offsets by point index
        for i in range(self.n_points):
            grid_init[:, i, :] *= i + 1
            
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        # initialize attention weights
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        
        # initialize value and output projections
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
    
    def forward(self, x, pos):
        """
        Forward pass for single-level deformable self attention.
        
        Args:
            x: Input features [b, n, c]
            pos: Input positions [b, n, 2]
            
        Returns:
            output: Self-attended features [b, n, c]
        """
        b, n, c = x.shape
        h = self.n_heads
        k = self.n_points
        c_ = self.c_
        
        # generate sampling offsets and attention weights
        sampling_offsets = self.sampling_offsets(x).view(b, n, h, k, 2)  # [b, n, h, k, 2]
        attention_weights = self.attention_weights(x).view(b, n, h, k)   # [b, n, h, k]
        attention_weights = F.softmax(attention_weights, dim=-1)  # [b, n, h, k]
        
        if self.use_fused_knn and k == 4:
            # Project values: [b, n, c] -> [b, n, h, c_]
            values = self.value_proj(x).view(b, n, h, c_)
            
            # Ensure contiguous tensors for optimal kernel performance
            # Non-contiguous tensors cause stride issues and potential slowdowns
            pos = pos.contiguous()
            sampling_offsets = sampling_offsets.contiguous()
            attention_weights = attention_weights.contiguous()
            values = values.contiguous()
            
            # Fully fused kernel: KNN + Shepard weighting + gather in one pass
            # pos is shared across heads (not expanded!)
            output = fused_knn_deformable_self_attention(
                pos,                # [b, n, 2] - shared, NOT expanded
                sampling_offsets,   # [b, n, h, k, 2]
                attention_weights,  # [b, n, h, k]
                values,             # [b, n, h, c_]
                self.shepard_power,
                D=2
            )  # Returns [b, n, h, c_]
            
            # Reshape to [b, n, c]
            output = output.reshape(b, n, c)
        else:
            # Project values with old layout
            values = self.value_proj(x).reshape(b, n, h, c_).permute(0, 2, 1, 3).reshape(b*h, n, c_)
            
            # compute sampling locations: pos + sampling_offsets
            sampling_locations = pos.unsqueeze(2).unsqueeze(3) + sampling_offsets  # [b, n, h, k, 2]
            sampling_locations = sampling_locations.permute(0, 2, 1, 3, 4).reshape(b*h, n*k, 2).contiguous()  # [b*h, n*k, 2]
            
            # for single-level attention, compute KNN directly instead of spatial lookup
            # this is simpler and avoids the complex spatial indexing
            pos_locations = pos.unsqueeze(1).expand(-1, h, -1, -1).reshape(b*h, n, 2).contiguous()
            nb_idx_real = triton_knn(sampling_locations, pos_locations, k=4)

            if self.use_fused:
                # Use fused Triton kernel (faster, lower memory bandwidth)
                # Reshape for fused kernel: expects [B, N, K] indices, [B, N] attention weights
                nb_idx_fused = nb_idx_real.reshape(b*h, n*k, 4)  # [b*h, n*k, 4]
                attn_weights_fused = attention_weights.permute(0, 2, 1, 3).reshape(b*h, n*k)  # [b*h, n*k]
                
                # Fused kernel computes: distance -> softmax -> weighted gather in one pass
                output = fused_deformable_attention(
                    sampling_locations,      # [b*h, n*k, 2]
                    pos_locations,           # [b*h, n, 2]
                    nb_idx_fused,            # [b*h, n*k, 4]
                    self.shepard_power,      # scalar
                    attn_weights_fused,      # [b*h, n*k]
                    values,                  # [b*h, n, c_]
                    D=2
                )  # [b*h, n*k, c_]
                
                # Reshape output: [b*h, n*k, c_] -> [b*h, n, k, c_] -> sum over k -> [b, n, c]
                output = output.reshape(b*h, n, k, c_).sum(dim=2)  # [b*h, n, c_]
                output = output.reshape(b, h, n, c_).permute(0, 2, 1, 3).reshape(b, n, c)
            else:
                # PyTorch reference implementation (slower, more memory)
                # [b*h, n_q*k, 4] -> [b*h, n_q*k, 4, 2]
                nb_idx_real_expanded = nb_idx_real.unsqueeze(-1).expand(-1, -1, -1, 2)
                # [b*h, n_kv, 2] -> [b*h, n_q*k, n_kv, 2]
                pos_expanded = pos_locations.unsqueeze(1).expand(-1, n*k, -1, -1)
                # gather the absolute positions of the k nearest tokens to each virtual token
                # [b*h, n_q*k, 4, 2]
                nb_pos = torch.gather(pos_expanded, 2, nb_idx_real_expanded)
                # relative position of k nearest tokens to each virtual token
                # [b*h, n_q*k, 4, 2] - [b*h, n_q*k, 1, 2] = [b*h, n_q*k, 4, 2]
                nb_token_rel_pos = nb_pos - sampling_locations.unsqueeze(2)

                # add 1e-6 inside the sqrt for numerical stability
                dist = torch.norm(nb_token_rel_pos, dim=-1, p=2) + 1e-6 
                # dist shape: [b*h, n_q*k, 4]

                # ensure power is positive
                power = F.relu(self.shepard_power) + 1e-6

                # add a .unsqueeze(-1) to 'dist' to match the 
                # [..., 1] shape of the MLP output
                logits = -power * dist.unsqueeze(-1)
                # logits shape: [b*h, n_q*k, 4, 1]

                nn_weights = F.softmax(logits, dim=-2)
                # reshape for MSDETRPC function
                # original: [b*h, -1, k*l, 4] where l=1 for single level
                nn_idx_reshaped = nb_idx_real.reshape(b*h, n, k, 4)  # [b*h, n, k, 4]
                nn_weights_reshaped = nn_weights.reshape(b*h, n, k, 4)  # [b*h, n, k, 4]
                attention_weights_reshaped = attention_weights.permute(0, 2, 1, 3).reshape(b*h, n, k)  # [b*h, n, k]
                
                # apply deformable attention using CUDA kernel
                output = MSDETRPCFunction.apply(
                    nn_idx_reshaped,
                    nn_weights_reshaped,
                    attention_weights_reshaped,
                    values
                ).reshape(b, h, n, c_).permute(0, 2, 1, 3).reshape(b, n, c)
        
        # final output projection
        output = self.output_proj(output)
        
        return output


class GlobalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout_p = dropout

    def forward(self, x, pos=None):
        """
        x: [batch_size, num_tokens, d_model]
        pos: [batch_size, num_tokens, d_model] (Optional positional encoding)
        """
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(b, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        is_training = self.training
        
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=0.0,
            is_causal=False
        )

        out = out.transpose(1, 2).reshape(b, n, c)
        
        return self.out_proj(out)