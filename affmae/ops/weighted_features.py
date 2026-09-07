# Imported through util so a torch-only install can still import this
# module; the kernels raise only when actually launched.
from .dispatch import triton, tl
from .dispatch import custom_fwd, custom_bwd  # torch-version shim
import torch


# @triton.heuristics({
#     'BLOCK_M': lambda args: 16,
#     'BLOCK_N': lambda args: max(8, min(32, args['M'])),
#     'BLOCK_D': lambda args: max(32, min(128, args['C'])),
#     'BLOCK_IC': lambda args: max(8, min(32, args['IC'])),
#     'num_warps': lambda args: 4,
#     'num_stages': lambda args: 3,
# })

# configs = [
#     triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_D': BD, 'BLOCK_IC': IC, 'num_warps': W, 'num_stages': S}) \
#     for BM in [8, 16, 32] \
#     for BN in [4, 8, 16] \
#     for BD in [8, 16, 32] \
#     for IC in [8, 16, 32] \
#     for W in [2, 4] \
#     for S in [2, 3] \
# ]

# @triton.autotune(
#     configs=configs,
#     key=['B', 'N', 'M', 'C', 'IC'],
# )
@triton.heuristics({
    'BLOCK_M': lambda args: 2,
    'BLOCK_N': lambda args: 4,
    'BLOCK_D': lambda args: 32,
    'BLOCK_IC': lambda args: 4,
    'num_warps': lambda args: 1,
    'num_stages': lambda args: 5,
})
@triton.jit
def _weighted_features_fwd_kernel(
    WEIGHTS,     # [B, N_out, M, IC]
    FEAT,        # [B, N, C]
    NBHD_IDX,    # [B, N_out, M]
    OUT,         # [B, N_out, IC, C]
    B, N, N_OUT, M, C, IC,
    # strides
    s_w_b, s_w_n, s_w_m, s_w_ic,
    s_f_b, s_f_n, s_f_c,
    s_idx_b, s_idx_n, s_idx_m,
    s_o_b, s_o_n, s_o_ic, s_o_c,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_IC: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tiles over output tokens i
    pid_b = tl.program_id(1)  # batch
    pid_k = tl.program_id(2)  # packs (ic-tiles, c-tiles)

    # decode packed (IC, C) tiling
    n_d_tiles = (C + BLOCK_D - 1) // BLOCK_D
    ic_tile = pid_k // n_d_tiles
    d_tile  = pid_k %  n_d_tiles

    offs_m  = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)          # i
    offs_ic = ic_tile * BLOCK_IC + tl.arange(0, BLOCK_IC)       # ic
    offs_d  = d_tile  * BLOCK_D  + tl.arange(0, BLOCK_D)        # c

    mask_m  = offs_m < N_OUT
    mask_ic = offs_ic < IC
    mask_d  = offs_d < C

    # base ptrs
    idx_ptr = NBHD_IDX + pid_b * s_idx_b + offs_m[:, None] * s_idx_n
    w_base  = WEIGHTS  + pid_b * s_w_b    + offs_m[:, None, None] * s_w_n
    f_base  = FEAT     + pid_b * s_f_b
    out_ptr = OUT      + pid_b * s_o_b    + offs_m[:, None, None] * s_o_n + offs_ic[None, :, None] * s_o_ic + offs_d[None, None, :] * s_o_c

    # accumulator [BM, BIC, BD]
    acc = tl.zeros((BLOCK_M, BLOCK_IC, BLOCK_D), dtype=tl.float32)

    for nb in range(0, M, BLOCK_N):
        nb_offs = nb + tl.arange(0, BLOCK_N)
        mask_n  = nb_offs < M
        mask_n_m = mask_m[:, None] & mask_n[None, :]

        # indices [BM, BN]
        idx_block = tl.load(
            idx_ptr + nb_offs[None, :] * s_idx_m,
            mask=mask_n_m,
            other=0
        ).to(tl.int32)

        # weights [BM, BN, BIC]
        w_block = tl.load(
            w_base + nb_offs[None, :, None] * s_w_m + offs_ic[None, None, :] * s_w_ic,
            mask=mask_n_m[:, :, None] & mask_ic[None, None, :],
            other=0.0
        ).to(tl.float32)

        # feat gather [BM, BN, BD] (broadcast to BIC later)
        feat_block = tl.load(
            f_base + idx_block[:, :, None] * s_f_n + offs_d[None, None, :] * s_f_c,
            mask=mask_n_m[:, :, None] & mask_d[None, None, :],
            other=0.0
        ).to(tl.float32)

        # accumulate: OUT[b,i,ic,c] = sum_ni weights[b,i,ni,ic] * feat[b, idx, c]
        acc += tl.sum(w_block[:, :, :, None] * feat_block[:, :, None, :], axis=1)  # sum over BN

    tl.store(
        out_ptr,
        acc,
        mask=mask_m[:, None, None] & mask_ic[None, :, None] & mask_d[None, None, :]
    )


def weighted_features_triton(weights, feat, nbhd_idx):
    """
    weights:  [B, N_out, M, IC]
    feat:     [B, N, C]
    nbhd_idx: [B, N_out, M]
    returns:  [B, N_out, IC, C]
    """
    B, N_out, M, IC = weights.shape
    _, N, C = feat.shape
    out = torch.zeros((B, N_out, IC, C), device=feat.device, dtype=feat.dtype)

    grid = lambda META: (
        triton.cdiv(N_out, META['BLOCK_M']),
        B,
        triton.cdiv(IC, META['BLOCK_IC']) * triton.cdiv(C, META['BLOCK_D']),
    )

    _weighted_features_fwd_kernel[grid](
        weights, feat, nbhd_idx, out,
        B, N, N_out, M, C, IC,
        # strides
        weights.stride(0), weights.stride(1), weights.stride(2), weights.stride(3),
        feat.stride(0),    feat.stride(1),    feat.stride(2),
        nbhd_idx.stride(0), nbhd_idx.stride(1), nbhd_idx.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )

    # convert out to dtype
    out = out.to(feat.dtype)

    # print('best fwd', _weighted_features_fwd_kernel.best_config)
    return out




# configs = [
#     triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_D': BD, 'BLOCK_IC': IC, 'num_warps': W, 'num_stages': S}) \
#     for BM in [8, 16, 32] \
#     for BN in [4, 8, 16] \
#     for BD in [8, 16, 32] \
#     for IC in [8, 16, 32] \
#     for W in [2, 4] \
#     for S in [2, 3] \
# ]

# @triton.autotune(
#     configs=configs,
#     key=['B', 'N', 'M', 'C', 'IC'],
# )
@triton.heuristics({
    'BLOCK_M': lambda args: 2,
    'BLOCK_N': lambda args: 4,
    'BLOCK_D': lambda args: 32,
    'BLOCK_IC': lambda args: 4,
    'num_warps': lambda args: 1,
    'num_stages': lambda args: 5,
})
@triton.jit
def _weighted_features_bwd_dfeat_kernel(
    DOUT,        # [B, N_out, IC, C]
    WEIGHTS,     # [B, N_out, M, IC]
    NBHD_IDX,    # [B, N_out, M]
    DFEAT,       # [B, N, C]  (atomic-add)
    B, N, N_OUT, M, C, IC,
    s_do_b, s_do_n, s_do_ic, s_do_c,
    s_w_b,  s_w_n,  s_w_m,   s_w_ic,
    s_idx_b, s_idx_n, s_idx_m,
    s_df_b, s_df_n, s_df_c,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_IC: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tiles over i
    pid_b = tl.program_id(1)  # batch
    pid_d = tl.program_id(2)  # tiles over c

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_m = offs_m < N_OUT
    mask_d = offs_d < C

    # Reduce over IC INSIDE the kernel to minimize atomics
    # accum_contrib[bm, bn, bd] = sum_ic (weights * dY)
    for nb in range(0, M, BLOCK_N):
        nb_offs = nb + tl.arange(0, BLOCK_N)
        mask_n  = nb_offs < M
        mask_nm = mask_m[:, None] & mask_n[None, :]

        # load neighbor indices
        idx_block = tl.load(
            NBHD_IDX + pid_b * s_idx_b + offs_m[:, None] * s_idx_n + nb_offs[None, :] * s_idx_m,
            mask=mask_nm,
            other=0
        ).to(tl.int32)

        # accumulator for the IC reduction
        contrib = tl.zeros((BLOCK_M, BLOCK_N, BLOCK_D), dtype=tl.float32)

        for ic0 in range(0, IC, BLOCK_IC):
            offs_ic = ic0 + tl.arange(0, BLOCK_IC)
            mask_ic = offs_ic < IC

            # dY tile: [BM, BIC, BD]
            dY_tile = tl.load(
                DOUT + pid_b * s_do_b + offs_m[:, None, None] * s_do_n + offs_ic[None, :, None] * s_do_ic + offs_d[None, None, :] * s_do_c,
                mask=mask_m[:, None, None] & mask_ic[None, :, None] & mask_d[None, None, :],
                other=0.0
            ).to(tl.float32)

            # weights: [BM, BN, BIC]
            w_tile = tl.load(
                WEIGHTS + pid_b * s_w_b + offs_m[:, None, None] * s_w_n + nb_offs[None, :, None] * s_w_m + offs_ic[None, None, :] * s_w_ic,
                mask=mask_nm[:, :, None] & mask_ic[None, None, :],
                other=0.0
            ).to(tl.float32)

            # reduce over IC
            contrib += tl.sum(w_tile[:, :, :, None] * dY_tile[:, None, :, :], axis=2)

        # atomic add into DFEAT at gathered indices
        dfeat_ptrs = DFEAT + pid_b * s_df_b + idx_block[:, :, None] * s_df_n + offs_d[None, None, :] * s_df_c
        tl.atomic_add(
            dfeat_ptrs,
            contrib,
            mask=mask_nm[:, :, None] & mask_d[None, None, :]
        )


# @triton.heuristics({
#     'BLOCK_M': lambda args: 16,
#     'BLOCK_N': lambda args: 4,
#     'BLOCK_D': lambda args: 16,
#     'BLOCK_IC': lambda args: 16,
#     'num_warps': lambda args: 4,
#     'num_stages': lambda args: 3,
# })
# configs = [
#     triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_D': BD, 'BLOCK_IC': IC, 'num_warps': W, 'num_stages': S}) \
#     for BM in [8, 16, 32] \
#     for BN in [4, 8, 16] \
#     for BD in [8, 16, 32] \
#     for IC in [8, 16, 32] \
#     for W in [2, 4] \
#     for S in [2, 3] \
# ]

# @triton.autotune(
#     configs=configs,
#     key=['B', 'N', 'M', 'C', 'IC'],
# )
@triton.heuristics({
    'BLOCK_M': lambda args: 2,
    'BLOCK_N': lambda args: 4,
    'BLOCK_D': lambda args: 32,
    'BLOCK_IC': lambda args: 4,
    'num_warps': lambda args: 1,
    'num_stages': lambda args: 5,
})
@triton.jit
def _weighted_features_bwd_dweights_kernel(
    DOUT,        # [B, N_out, IC, C]
    FEAT,        # [B, N, C]
    NBHD_IDX,    # [B, N_out, M]
    DWEIGHTS,    # [B, N_out, M, IC]
    B, N, N_OUT, M, C, IC,
    s_do_b, s_do_n, s_do_ic, s_do_c,
    s_f_b,  s_f_n,  s_f_c,
    s_idx_b, s_idx_n, s_idx_m,
    s_dw_b, s_dw_n, s_dw_m, s_dw_ic,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_IC: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tiles over i
    pid_b = tl.program_id(1)  # batch

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N_OUT

    idx_row_ptr = NBHD_IDX + pid_b * s_idx_b + offs_m[:, None] * s_idx_n
    dweights_row_ptr = DWEIGHTS + pid_b * s_dw_b + offs_m[:, None, None] * s_dw_n

    for nb in range(0, M, BLOCK_N):
        nb_offs = nb + tl.arange(0, BLOCK_N)
        mask_n  = nb_offs < M
        mask_nm = mask_m[:, None] & mask_n[None, :]

        # gather indices [BM, BN]
        idx_block = tl.load(
            idx_row_ptr + nb_offs[None, :] * s_idx_m,
            mask=mask_nm,
            other=0
        ).to(tl.int32)

        # accumulator [BM, BN, BIC]
        dW_acc = tl.zeros((BLOCK_M, BLOCK_N, BLOCK_IC), dtype=tl.float32)

        # loop over IC × C reduction inside to avoid atomics
        for ic0 in range(0, IC, BLOCK_IC):
            offs_ic = ic0 + tl.arange(0, BLOCK_IC)
            mask_ic = offs_ic < IC

            # local accumulator for this IC tile
            dW_ic = tl.zeros((BLOCK_M, BLOCK_N, BLOCK_IC), dtype=tl.float32)

            for d0 in range(0, C, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                mask_d = offs_d < C

                # dY: [BM, BIC, BD]
                dY_tile = tl.load(
                    DOUT + pid_b * s_do_b + offs_m[:, None, None] * s_do_n + offs_ic[None, :, None] * s_do_ic + offs_d[None, None, :] * s_do_c,
                    mask=mask_m[:, None, None] & mask_ic[None, :, None] & mask_d[None, None, :],
                    other=0.0
                ).to(tl.float32)

                # FEAT gather: [BM, BN, BD]
                feat_block = tl.load(
                    FEAT + pid_b * s_f_b + idx_block[:, :, None] * s_f_n + offs_d[None, None, :] * s_f_c,
                    mask=mask_nm[:, :, None] & mask_d[None, None, :],
                    other=0.0
                ).to(tl.float32)

                # accumulate over BD
                dW_ic += tl.sum(feat_block[:, :, None, :] * dY_tile[:, None, :, :], axis=3)  # sum over BD

            dW_acc += dW_ic  # add this IC tile

        # store [BM, BN, BIC] into DWEIGHTS row chunk
        tl.store(
            dweights_row_ptr + nb_offs[None, :, None] * s_dw_m + tl.arange(0, BLOCK_IC)[None, None, :] * s_dw_ic,
            dW_acc,
            mask=mask_nm[:, :, None] & (tl.arange(0, BLOCK_IC) < IC)[None, None, :]
        )


def weighted_features_triton_backward(dout, weights, feat, nbhd_idx):
    """
    dout:     [B, N_out, IC, C]
    weights:  [B, N_out, M, IC]
    feat:     [B, N, C]
    nbhd_idx: [B, N_out, M]
    returns: d_weights, d_feat
    """
    B, N_out, IC, C = dout.shape
    _, _, M, _ = weights.shape
    _, N, _ = feat.shape

    d_weights = torch.zeros_like(weights, dtype=torch.float32)
    d_feat = torch.zeros_like(feat, dtype=torch.float32)

    # d_feat (atomics; reduce over IC inside)
    grid_df = lambda META: (
        triton.cdiv(N_out, META['BLOCK_M']),
        B,
        triton.cdiv(C, META['BLOCK_D']),
    )
    _weighted_features_bwd_dfeat_kernel[grid_df](
        dout, weights, nbhd_idx, d_feat,
        B, N, N_out, M, C, IC,
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        weights.stride(0), weights.stride(1), weights.stride(2), weights.stride(3),
        nbhd_idx.stride(0), nbhd_idx.stride(1), nbhd_idx.stride(2),
        d_feat.stride(0), d_feat.stride(1), d_feat.stride(2),
    )
    # print('best bwd dfeat', _weighted_features_bwd_dfeat_kernel.best_config)

    # d_weights (no atomics; full channel reduction inside)
    grid_dw = lambda META: (
        triton.cdiv(N_out, META['BLOCK_M']),
        B,
    )
    _weighted_features_bwd_dweights_kernel[grid_dw](
        dout, feat, nbhd_idx, d_weights,
        B, N, N_out, M, C, IC,
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        feat.stride(0),  feat.stride(1),  feat.stride(2),
        nbhd_idx.stride(0), nbhd_idx.stride(1), nbhd_idx.stride(2),
        d_weights.stride(0), d_weights.stride(1), d_weights.stride(2), d_weights.stride(3),
    )
    # print('best bwd dweights', _weighted_features_bwd_dweights_kernel.best_config)

    # convert d_weights and d_feat to dtype
    d_weights = d_weights.to(weights.dtype)
    d_feat = d_feat.to(feat.dtype)

    return d_weights, d_feat


class WeightedFeaturesFunction(torch.autograd.Function):
    """Autograd wrapper over the weighted gather-reduce kernels.

    Prefer :func:`weighted_features`, which falls back to
    :func:`weighted_features_ref` when Triton cannot run on the given tensors.
    """
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float16)
    def forward(ctx, weights: torch.Tensor, feat: torch.Tensor, nbhd_idx: torch.Tensor):
        """
        weights:  [B, N_out, M, IC]
        feat:     [B, N, C]
        nbhd_idx: [B, N_out, M]
        returns:  [B, N_out, IC, C]
        """
        assert weights.is_cuda and feat.is_cuda and nbhd_idx.is_cuda
        weights = weights.contiguous()
        feat = feat.contiguous()
        nbhd_idx = nbhd_idx.to(torch.int32).contiguous()
        out = weighted_features_triton(weights, feat, nbhd_idx)
        ctx.save_for_backward(weights, feat, nbhd_idx)
        return out

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dout: torch.Tensor):
        """
        dout: [B, N_out, IC, C]
        returns: (d_weights, d_feat, None)
        """
        weights, feat, nbhd_idx = ctx.saved_tensors
        dout = dout.contiguous()
        d_weights, d_feat = weighted_features_triton_backward(dout, weights, feat, nbhd_idx)
        return d_weights, d_feat, None


# Convenience API
def weighted_features(weights: torch.Tensor, feat: torch.Tensor,
                      nbhd_idx: torch.Tensor) -> torch.Tensor:
    """Weighted gather-and-reduce over each token's neighbourhood.

    Args:
        weights: [B, N_out, M, IC] per-neighbour weights.
        feat: [B, N, C] source features.
        nbhd_idx: [B, N_out, M] indices into the N axis of ``feat``.
    Returns:
        [B, N_out, IC, C]

    Note:
        Dispatches to the Triton kernel on a live GPU backend and to
        ``weighted_features_ref`` otherwise. The reference is plain gather and
        reduce, so autograd covers its backward pass.
    """
    from .dispatch import can_use_triton

    if not can_use_triton(weights, feat, nbhd_idx):
        return weighted_features_ref(weights, feat, nbhd_idx.long())
    return WeightedFeaturesFunction.apply(weights, feat, nbhd_idx)


# --- PyTorch reference implementation (also the CPU/MPS path) ---
def weighted_features_ref(weights, feat, nbhd_idx):
    """
    weights:  [B, N_out, M, IC]
    feat:     [B, N, C]
    nbhd_idx: [B, N_out, M]
    returns:  [B, N_out, IC, C]
    """
    B, N_out, M, IC = weights.shape
    _, N, C = feat.shape
    # gather neighbors: (B, N_out*M, C) -> reshape to (B, N_out, M, C)
    idx_flat = nbhd_idx.reshape(B, N_out * M)
    gathered = feat.gather(
        dim=1,
        index=idx_flat[..., None].expand(-1, -1, C)
    ).reshape(B, N_out, M, C)
    # (B, N_out, M, IC) * (B, N_out, M, 1, C) -> sum over M
    out = (weights[..., None] * gathered[:, :, :, None, :]).sum(dim=2)
    return out
