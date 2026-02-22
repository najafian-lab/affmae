import triton
import triton.language as tl
import torch
import sys
import os
import time
from torch.autograd import Function
from torch.amp import custom_fwd, custom_bwd

# configs = [
#     triton.Config({'BLOCK_M': BM, 'BLOCK_NB': BN, 'BLOCK_K': K, 'BLOCK_D': BD, 'num_warps': W, 'num_stages': S}) \
#     for BM in [16, 32] \
#     for BN in [16, 32] \
#     for K in [4, 8] \
#     for BD in [16, 32] \
#     for W in [2, 4, 8] \
#     for S in [2, 3, 4] \
# ]

# @triton.autotune(
#     configs=configs,
#     key=['B', 'N', 'M', 'K', 'C'],
# )

@triton.heuristics({
    "BLOCK_M": lambda a: 16,
    "BLOCK_NB": lambda a: 16,
    "BLOCK_K": lambda a: 4,
    "BLOCK_D": lambda a: 32,
    "num_warps": lambda a: 4,
    "num_stages": lambda a: 3,
})
@triton.jit
def _msdetrpc_fwd_kernel(
    NN_IDX,        # [B, N, M, K] int
    NN_WEIGHT,     # [B, N, M, K] fp
    ATTN,          # [B, N, M]    fp
    VAL,           # [B, N_val, C] fp
    OUT,           # [B, N, C]    fp
    B, N, M, K, C, N_VAL,
    s_i_b, s_i_n, s_i_m, s_i_k,    # strides for nn_idx
    s_w_b, s_w_n, s_w_m, s_w_k,    # strides for nn_weight
    s_a_b, s_a_n, s_a_m,           # strides for attn
    s_v_b, s_v_n, s_v_c,           # strides for val
    s_o_b, s_o_n, s_o_c,           # strides for out
    BLOCK_M: tl.constexpr, BLOCK_NB: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)                            # tile over N (tokens)
    pid_b = tl.program_id(1)                            # batch
    pid_d = tl.program_id(2)                            # tile over C (channels)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)    # [BM]
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)    # [BD]

    mask_m = offs_m < N
    mask_d = offs_d < C

    # Base pointers
    idx_ptr = NN_IDX + pid_b * s_i_b + offs_m[:, None, None] * s_i_n
    wgt_ptr = NN_WEIGHT + pid_b * s_w_b + offs_m[:, None, None] * s_w_n
    att_ptr = ATTN + pid_b * s_a_b + offs_m[:, None] * s_a_n
    val_b   = VAL + pid_b * s_v_b
    out_ptr = OUT + pid_b * s_o_b + offs_m[:, None] * s_o_n + offs_d[None, :] * s_o_c

    # Accumulator for output features: Σ_ni Σ_ki attn * nn_weight * val[nbi, :]
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Stream neighborhoods (ni) in tiles
    for nb0 in range(0, M, BLOCK_NB):
        ni = nb0 + tl.arange(0, BLOCK_NB)               # [BN]
        m_nb = ni < M
        mask_mnb = mask_m[:, None] & m_nb[None, :]      # [BM, BN]

        # attn: [BM, BN]
        att_blk = tl.load(att_ptr + ni[None, :] * s_a_m, mask=mask_mnb, other=0.0).to(tl.float32)

        # Stream K interpolation points in small tiles
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + tl.arange(0, BLOCK_K)             # [BK]
            m_kk = kk < K
            mask_mnbk = mask_mnb[:, :, None] & m_kk[None, None, :]  # [BM, BN, BK]

            # nn_idx, nn_weight
            idx_blk = tl.load(
                idx_ptr + ni[None, :, None] * s_i_m + kk[None, None, :] * s_i_k,
                mask=mask_mnbk, other=0
            ).to(tl.int32)  # [BM, BN, BK]

            w_blk = tl.load(
                wgt_ptr + ni[None, :, None] * s_w_m + kk[None, None, :] * s_w_k,
                mask=mask_mnbk, other=0.0
            ).to(tl.float32)  # [BM, BN, BK]

            # (attn * weight): [BM, BN, BK]
            aw = (att_blk[:, :, None] * w_blk).to(tl.float32)

            # gather VAL and accumulate across K then across BN
            # val[nbi, d]
            val_blk = tl.load(
                val_b + idx_blk[:, :, :, None] * s_v_n + offs_d[None, None, None, :] * s_v_c,
                mask=mask_mnbk[:, :, :, None] & mask_d[None, None, None, :],
                other=0.0
            ).to(tl.float32)  # [BM, BN, BK, BD]

            # sum over K then over BN
            acc += tl.sum(tl.sum(aw[:, :, :, None] * val_blk, axis=2), axis=1)  # [BM, BD]

    # Store
    tl.store(out_ptr, acc, mask=mask_m[:, None] & mask_d[None, :])

# configs = [
#     triton.Config({'BLOCK_M': BM, 'BLOCK_NB': BN, 'BLOCK_K': K, 'BLOCK_D': BD, 'num_warps': W, 'num_stages': S}) \
#     for BM in [16, 32] \
#     for BN in [16, 32] \
#     for K in [4, 8] \
#     for BD in [16, 32] \
#     for W in [2, 4, 8] \
#     for S in [2, 3, 4] \
# ]
# @triton.autotune(
#     configs=configs,
#     key=['B', 'N', 'M', 'K', 'C'],
# )
@triton.heuristics({
    "BLOCK_M": lambda a: 16,
    "BLOCK_NB": lambda a: 16,
    "BLOCK_K": lambda a: 4,
    "BLOCK_D": lambda a: 16,
    "num_warps": lambda a: 4,
    "num_stages": lambda a: 2,
})
@triton.jit
def _msdetrpc_bwd_dval_kernel(
    DFEAT, NN_IDX, NN_WEIGHT, ATTN, DVAL,
    B, N, M, K, C, N_VAL,
    s_df_b, s_df_n, s_df_c,
    s_i_b, s_i_n, s_i_m, s_i_k,
    s_w_b, s_w_n, s_w_m, s_w_k,
    s_a_b, s_a_n, s_a_m,
    s_dv_b, s_dv_n, s_dv_c,
    BLOCK_M: tl.constexpr, BLOCK_NB: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_m = offs_m < N
    mask_d = offs_d < C

    dfeat_ptr = DFEAT + pid_b * s_df_b + offs_m[:, None] * s_df_n + offs_d[None, :] * s_df_c
    idx_ptr   = NN_IDX + pid_b * s_i_b + offs_m[:, None, None] * s_i_n
    wgt_ptr   = NN_WEIGHT + pid_b * s_w_b + offs_m[:, None, None] * s_w_n
    att_ptr   = ATTN + pid_b * s_a_b + offs_m[:, None] * s_a_n
    dval_b    = DVAL + pid_b * s_dv_b

    # dO tile
    dO = tl.load(dfeat_ptr, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)  # [BM, BD]

    for nb0 in range(0, M, BLOCK_NB):
        ni = nb0 + tl.arange(0, BLOCK_NB)
        m_nb = ni < M
        mask_mnb = mask_m[:, None] & m_nb[None, :]

        att_blk = tl.load(att_ptr + ni[None, :] * s_a_m, mask=mask_mnb, other=0.0).to(tl.float32)  # [BM, BN]

        for k0 in range(0, K, BLOCK_K):
            kk = k0 + tl.arange(0, BLOCK_K)
            m_kk = kk < K
            mask_mnbk = mask_mnb[:, :, None] & m_kk[None, None, :]

            idx_blk = tl.load(
                idx_ptr + ni[None, :, None] * s_i_m + kk[None, None, :] * s_i_k,
                mask=mask_mnbk, other=0
            ).to(tl.int32)  # [BM, BN, BK]

            w_blk = tl.load(
                wgt_ptr + ni[None, :, None] * s_w_m + kk[None, None, :] * s_w_k,
                mask=mask_mnbk, other=0.0
            ).to(tl.float32)  # [BM, BN, BK]

            scale = (att_blk[:, :, None] * w_blk).to(tl.float32)  # [BM, BN, BK]
            contrib = scale[:, :, :, None] * dO[:, None, None, :]  # [BM, BN, BK, BD]

            # Atomic add into DVAL[b, nbi, d]
            dval_ptrs = dval_b + idx_blk[:, :, :, None] * s_dv_n + offs_d[None, None, None, :] * s_dv_c
            tl.atomic_add(dval_ptrs, contrib, mask=mask_mnbk[:, :, :, None] & mask_d[None, None, None, :])

# configs = [
#     triton.Config({'BLOCK_M': BM, 'BLOCK_NB': BN, 'BLOCK_K': K, 'BLOCK_D': BD, 'num_warps': W, 'num_stages': S}) \
#     for BM in [16, 32] \
#     for BN in [16, 32] \
#     for K in [4, 8] \
#     for BD in [16, 32] \
#     for W in [2, 4, 8] \
#     for S in [2, 3, 4] \
# ]

# @triton.autotune(
#     configs=configs,
#     key=['B', 'N', 'M', 'K', 'C'],
# )
@triton.heuristics({
    "BLOCK_M": lambda a: 16,
    "BLOCK_NB": lambda a: 16,
    "BLOCK_K": lambda a: 4,
    "BLOCK_D": lambda a: 32,
    "num_warps": lambda a: 4,
    "num_stages": lambda a: 3,
})
@triton.jit
def _msdetrpc_bwd_dwk_kernel(
    DFEAT, NN_IDX, NN_WEIGHT, ATTN, VAL, DNW, DATTN,
    B, N, M, K, C, N_VAL,
    s_df_b, s_df_n, s_df_c,
    s_i_b, s_i_n, s_i_m, s_i_k,
    s_w_b, s_w_n, s_w_m, s_w_k,
    s_a_b, s_a_n, s_a_m,
    s_v_b, s_v_n, s_v_c,
    s_dnw_b, s_dnw_n, s_dnw_m, s_dnw_k,
    s_da_b, s_da_n, s_da_m,
    BLOCK_M: tl.constexpr, BLOCK_NB: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N

    dfeat_row = DFEAT + pid_b * s_df_b + offs_m[:, None] * s_df_n
    idx_ptr   = NN_IDX + pid_b * s_i_b + offs_m[:, None, None] * s_i_n
    wgt_ptr   = NN_WEIGHT + pid_b * s_w_b + offs_m[:, None, None] * s_w_n
    att_ptr   = ATTN + pid_b * s_a_b + offs_m[:, None] * s_a_n
    val_b     = VAL + pid_b * s_v_b
    dnw_ptr   = DNW + pid_b * s_dnw_b + offs_m[:, None, None] * s_dnw_n
    dattn_ptr = DATTN + pid_b * s_da_b + offs_m[:, None] * s_da_n

    for nb0 in range(0, M, BLOCK_NB):
        ni = nb0 + tl.arange(0, BLOCK_NB)
        m_nb = ni < M
        mask_mnb = mask_m[:, None] & m_nb[None, :]

        att_blk = tl.load(att_ptr + ni[None, :] * s_a_m, mask=mask_mnb, other=0.0).to(tl.float32)  # [BM, BN]

        # accumulators for this (BM, BN) tile
        d_att_acc = tl.zeros((BLOCK_M, BLOCK_NB), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            kk = k0 + tl.arange(0, BLOCK_K)
            m_kk = kk < K
            mask_mnbk = mask_mnb[:, :, None] & m_kk[None, None, :]

            idx_blk = tl.load(
                idx_ptr + ni[None, :, None] * s_i_m + kk[None, None, :] * s_i_k,
                mask=mask_mnbk, other=0
            ).to(tl.int32)  # [BM, BN, BK]

            w_blk = tl.load(
                wgt_ptr + ni[None, :, None] * s_w_m + kk[None, None, :] * s_w_k,
                mask=mask_mnbk, other=0.0
            ).to(tl.float32)  # [BM, BN, BK]

            # Accumulator for d_nn_weight over channel tiles
            d_w_acc = tl.zeros((BLOCK_M, BLOCK_NB, BLOCK_K), dtype=tl.float32)

            # Reduce over channels inside the kernel
            for d0 in range(0, C, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                mask_d = offs_d < C

                dO = tl.load(dfeat_row + offs_d[None, :] * s_df_c,
                             mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)  # [BM, BD]

                v_blk = tl.load(
                    val_b + idx_blk[:, :, :, None] * s_v_n + offs_d[None, None, None, :] * s_v_c,
                    mask=mask_mnbk[:, :, :, None] & mask_d[None, None, None, :],
                    other=0.0
                ).to(tl.float32)  # [BM, BN, BK, BD]

                # tmp = <val[nbi], dO> over channels
                tmp = tl.sum(v_blk * dO[:, None, None, :], axis=3)  # [BM, BN, BK]

                # d_nn_weight += tmp * attn
                d_w_acc += tmp * att_blk[:, :, None]

                # d_attn accumulates across K: += tmp * weight
                d_att_acc += tl.sum(tmp * w_blk, axis=2)

            # store d_nn_weight for this k-tile
            tl.store(
                dnw_ptr + ni[None, :, None] * s_dnw_m + kk[None, None, :] * s_dnw_k,
                d_w_acc,
                mask=mask_mnbk
            )

        # store d_attn for this nb tile
        tl.store(dattn_ptr + ni[None, :] * s_da_m, d_att_acc, mask=mask_mnb)


# -----------------------
# Python launchers + autograd
# -----------------------
def launch_msdetrpc_forward(nn_idx, nn_weight, attn, val):
    B, N, M, K = nn_idx.shape
    _, N_val, C = val.shape
    out = torch.empty((B, N, C), device=val.device, dtype=val.dtype)

    def grid_fwd(META):
        assert META['BLOCK_M'] <= N, f"BLOCK_M ({META['BLOCK_M']}) must be less than or equal to N ({N})"
        assert META['BLOCK_D'] <= C, f"BLOCK_D ({META['BLOCK_D']}) must be less than or equal to C ({C})"
        return (triton.cdiv(N, META['BLOCK_M']), B, triton.cdiv(C, META['BLOCK_D']))
    
    _msdetrpc_fwd_kernel[grid_fwd](
        nn_idx, nn_weight, attn, val, out,
        B, N, M, K, C, N_val,
        nn_idx.stride(0), nn_idx.stride(1), nn_idx.stride(2), nn_idx.stride(3),
        nn_weight.stride(0), nn_weight.stride(1), nn_weight.stride(2), nn_weight.stride(3),
        attn.stride(0), attn.stride(1), attn.stride(2),
        val.stride(0), val.stride(1), val.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
    )
    # print('best fwd kernel: ', _msdetrpc_fwd_kernel.best_config)
    return out


def launch_msdetrpc_backward(d_feat, nn_idx, nn_weight, attn, val):
    B, N, M, K = nn_idx.shape
    _, N_val, C = val.shape

    d_val = torch.zeros_like(val)
    d_nw  = torch.zeros_like(nn_weight)
    d_att = torch.zeros_like(attn)

    def grid_dval(META):
        assert META['BLOCK_M'] <= N, f"BLOCK_M ({META['BLOCK_M']}) must be less than or equal to N ({N})"
        assert META['BLOCK_D'] <= C, f"BLOCK_D ({META['BLOCK_D']}) must be less than or equal to C ({C})"
        return (triton.cdiv(N, META['BLOCK_M']), B, triton.cdiv(C, META['BLOCK_D']))

    _msdetrpc_bwd_dval_kernel[grid_dval](
        d_feat, nn_idx, nn_weight, attn, d_val,
        B, N, M, K, C, N_val,
        d_feat.stride(0), d_feat.stride(1), d_feat.stride(2),
        nn_idx.stride(0), nn_idx.stride(1), nn_idx.stride(2), nn_idx.stride(3),
        nn_weight.stride(0), nn_weight.stride(1), nn_weight.stride(2), nn_weight.stride(3),
        attn.stride(0), attn.stride(1), attn.stride(2),
        d_val.stride(0), d_val.stride(1), d_val.stride(2),
    )

    # print('best dval kernel: ', _msdetrpc_bwd_dval_kernel.best_config)

    def grid_dwk(META):
        assert META['BLOCK_M'] <= N, f"BLOCK_M ({META['BLOCK_M']}) must be less than or equal to N ({N})"
        return (triton.cdiv(N, META['BLOCK_M']), B)

    _msdetrpc_bwd_dwk_kernel[grid_dwk](
        d_feat, nn_idx, nn_weight, attn, val, d_nw, d_att,
        B, N, M, K, C, N_val,
        d_feat.stride(0), d_feat.stride(1), d_feat.stride(2),
        nn_idx.stride(0), nn_idx.stride(1), nn_idx.stride(2), nn_idx.stride(3),
        nn_weight.stride(0), nn_weight.stride(1), nn_weight.stride(2), nn_weight.stride(3),
        attn.stride(0), attn.stride(1), attn.stride(2),
        val.stride(0), val.stride(1), val.stride(2),
        d_nw.stride(0), d_nw.stride(1), d_nw.stride(2), d_nw.stride(3),
        d_att.stride(0), d_att.stride(1), d_att.stride(2),
    )
    # print('best dwk kernel: ', _msdetrpc_bwd_dwk_kernel.best_config)
    return d_nw, d_att, d_val


class MSDETRPCFunction(Function):
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float16)
    def forward(ctx, nn_idx, nn_weight, attn, val):
        nn_idx = nn_idx.to(torch.int32).contiguous()
        nn_weight = nn_weight.contiguous()
        attn = attn.contiguous()
        val = val.contiguous()
        out = launch_msdetrpc_forward(nn_idx, nn_weight, attn, val)
        ctx.save_for_backward(nn_idx, nn_weight, attn, val)
        return out

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, d_out):
        nn_idx, nn_weight, attn, val = ctx.saved_tensors
        d_out = d_out.contiguous()
        d_nw, d_att, d_val = launch_msdetrpc_backward(d_out, nn_idx, nn_weight, attn, val)
        return None, d_nw, d_att, d_val


def test_msdetrpc_speed(
    B=64, N=1024, N_val=2048, M=8, K=4, C=256, half=True, iters=10, warmup=3, seed=0
):
    """
    Compare Triton MSDETRPCFunctionTriton vs CUDA MSDETRPCFunction.
    Shapes:
      nn_idx      : [B, N, M, K] (int32 for Triton; int64 for CUDA)
      nn_weight   : [B, N, M, K]
      attn        : [B, N, M]
      val         : [B, N_val, C]
      out/feat    : [B, N, C]
    """
    torch.manual_seed(seed)
    device = "cuda"

    # --- Inputs ---
    nn_idx32   = torch.randint(N_val, (B, N, M, K), device=device, dtype=torch.int32).contiguous()
    nn_idx64   = nn_idx32.long().contiguous()  # CUDA kernel expects int64
    nn_weight  = torch.rand(B, N, M, K, device=device)            # unnormalized ok; CUDA just multiplies
    attn       = torch.randn(B, N, M, device=device)
    attn       = torch.softmax(attn, dim=-1).contiguous()
    val        = torch.randn(B, N_val, C, device=device)

    if half:
        nn_weight = nn_weight.to(torch.float16).contiguous()
        attn      = attn.to(torch.float16).contiguous()
        val       = val.to(torch.float16).contiguous()

    # grads enabled (match your CUDA kernels’ backward signatures)
    for t in (nn_weight, attn, val):
        t.requires_grad_(True)
        t.retain_grad()

    for _ in range(warmup):
        nn_weight.grad = attn.grad = val.grad = None
        out_t = MSDETRPCFunction.apply(nn_idx32, nn_weight, attn, val)
        out_t.mean().backward()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        torch.cuda.synchronize()
        nn_weight.grad = attn.grad = val.grad = None
        out_t = MSDETRPCFunction.apply(nn_idx32, nn_weight, attn, val)
        out_t.mean().backward()
        torch.cuda.synchronize()
    triton_time = (time.time() - t0) / iters

    # Save Triton results for correctness comparisons
    out_t_det   = out_t.detach()
    dnw_t_det   = nn_weight.grad.detach().clone()
    datt_t_det  = attn.grad.detach().clone()
    dval_t_det  = val.grad.detach().clone()

    # --- CUDA forward/backward (warmup) ---
    try:
        from archive.clusten import MSDETRPCFunction as MSDETRPCFunctionCUDA
    except Exception as e:
        print("CUDA MSDETRPCFunction not available:", e)
        print(f"Triton time only: {triton_time*1e3:.2f} ms")
        return

    for _ in range(warmup):
        nn_weight.grad = attn.grad = val.grad = None
        out_c = MSDETRPCFunctionCUDA.apply(nn_idx64, nn_weight, attn, val)
        out_c.mean().backward()

    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(iters):
        torch.cuda.synchronize()
        nn_weight.grad = attn.grad = val.grad = None
        out_c = MSDETRPCFunctionCUDA.apply(nn_idx64, nn_weight, attn, val)
        out_c.mean().backward()
        torch.cuda.synchronize()
    cuda_time = (time.time() - t1) / iters

    # --- Correctness ---
    out_c_det   = out_c.detach()
    dnw_c_det   = nn_weight.grad.detach().clone()
    datt_c_det  = attn.grad.detach().clone()
    dval_c_det  = val.grad.detach().clone()

    f_l2   = torch.linalg.norm(out_c_det - out_t_det).item()
    f_max  = torch.max(torch.abs(out_c_det - out_t_det)).item()
    dnw_l2 = torch.linalg.norm(dnw_c_det - dnw_t_det).item()
    datt_l2= torch.linalg.norm(datt_c_det - datt_t_det).item()
    dval_l2= torch.linalg.norm(dval_c_det - dval_t_det).item()

    print("=== MSDETRPC Triton vs CUDA ===")
    print(f"Shapes match: {tuple(out_t_det.shape) == tuple(out_c_det.shape)}")
    print(f"Fwd  L2 diff:   {f_l2:.6e}   | max abs: {f_max:.6e}")
    print(f"dW   L2 diff:   {dnw_l2:.6e}")
    print(f"dAtt L2 diff:   {datt_l2:.6e}")
    print(f"dVal L2 diff:   {dval_l2:.6e}")
    print(f"CUDA  time:     {cuda_time*1e3:.2f} ms")
    print(f"Triton time:    {triton_time*1e3:.2f} ms")
    print(f"Speedup (CUDA/Triton): {cuda_time / triton_time:.2f}×")


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    b = 16
    n = 801
    n_ = 901
    m = 16
    k = 4
    c = 256

    # dummy data
    nn_idx = torch.randint(n_, (b, n, m, k)).cuda()
    nn_weights = torch.rand(b, n, m, k).cuda()
    attn = torch.rand(b, n, m).cuda()
    val = torch.rand(b, n_, c).cuda()

    nn_weights.requires_grad_(True)
    nn_weights.retain_grad()
    attn.requires_grad_(True)
    attn.retain_grad()
    val.requires_grad_(True)
    val.retain_grad()

    # use the custom kernel
    feat = MSDETRPCFunction.apply(nn_idx, nn_weights, attn, val)
    feat.mean().backward()
    grad_weights = nn_weights.grad.clone().detach()
    grad_attn = attn.grad.clone().detach()
    grad_val = val.grad.clone().detach()
    nn_weights.grad.data.zero_()
    attn.grad.data.zero_()
    val.grad.data.zero_()

    # use the pytorch equivalent
    # nn_val = val.gather(index=nn_idx.view(b, -1).unsqueeze(2).expand(-1, -1, c), dim=1).reshape(b, n, m, k, c)
    # feat2 = ((nn_val * nn_weights.unsqueeze(4)).sum(3) * attn.unsqueeze(3)).sum(2)  # b x n x c
    from archive.clusten import MSDETRPCFunction as MSDETRPCFunctionCUDA
    feat2 = MSDETRPCFunctionCUDA.apply(nn_idx, nn_weights, attn, val)
    feat2.mean().backward()
    grad_weights2 = nn_weights.grad.clone().detach()
    grad_attn2 = attn.grad.clone().detach()
    grad_val2 = val.grad.clone().detach()
    nn_weights.grad.data.zero_()
    attn.grad.data.zero_()
    val.grad.data.zero_()

    print('diff of forward: ', torch.linalg.norm(feat2 - feat))
    print('diff of grad weights: ', torch.linalg.norm(grad_weights2 - grad_weights))
    print('diff of grad attn: ', torch.linalg.norm(grad_attn2 - grad_attn))
    print('diff of grad val: ', torch.linalg.norm(grad_val2 - grad_val))

    # Example quick run (tweak sizes for your GPU)
    test_msdetrpc_speed(
        B=128, N=1024, N_val=2048, M=8, K=4, C=256,
        half=True, iters=10, warmup=3, seed=0
    )
