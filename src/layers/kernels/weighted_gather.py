import torch
from torch.autograd import Function
from torch.amp import custom_fwd, custom_bwd
import triton
import triton.language as tl

# configs = [
#     triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_D': BD, 'num_warps': W, 'num_stages': S}) \
#     for BM in [16, 32, 64, 128] \
#     for BN in [4, 16, 32] \
#     for BD in [16, 32, 64] \
#     for W in [1, 2, 4, 8] \
#     for S in [1, 2, 3, 4] \
# ]

# @triton.autotune(
#     configs=configs,
#     key=['B', 'N', 'M', 'C', 'N_OLD'],
# )

@triton.heuristics({
    'BLOCK_M': lambda args: 16,  # number of tokens per block
    'BLOCK_N': lambda args: max(8, min(32, args['M'])),  # neighborhood block size
    'BLOCK_D': lambda args: max(16, min(64, args['C'])),  # feature dim tile
    'num_warps': lambda args: 4,
    'num_stages': lambda args: 3,
})
@triton.jit
def _weighted_gather_fwd_kernel(
    NBHD_IDX,   # [b, n, m] (int32 or int64)
    WEIGHTS,    # [b, n, m]
    FEAT,       # [b, n_old, c]
    FEAT_NEW,   # [b, n, c]
    B, N, M, C, N_OLD,
    stride_idx_b, stride_idx_n, stride_idx_m,
    stride_w_b, stride_w_n, stride_w_m,
    stride_f_b, stride_f_n, stride_f_c,
    stride_fn_b, stride_fn_n, stride_fn_c,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):

    # assert multiples of 8 (nah, just use arange)
    # tl.static_assert(BLOCK_M % 8 == 0)
    # tl.static_assert(BLOCK_N % 8 == 0)
    # tl.static_assert(BLOCK_D % 8 == 0)

    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    # Mask for valid elements
    mask_m = offs_m < N
    mask_d = offs_d < C

    # pointers
    nbhd_idx_ptr = NBHD_IDX + pid_b * stride_idx_b + offs_m[:, None] * stride_idx_n
    weights_ptr  = WEIGHTS + pid_b * stride_w_b + offs_m[:, None] * stride_w_n
    feat_ptr     = FEAT + pid_b * stride_f_b
    out_ptr      = FEAT_NEW  + pid_b * stride_fn_b + offs_m[:, None] * stride_fn_n + offs_d[None, :]

    # accumulator for output
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # loop over neighborhood in chunks of BLOCK_N
    for nb in range(0, M, BLOCK_N):
        nb_offs = nb + tl.arange(0, BLOCK_N)
        mask_n = nb_offs < M
        mask_nm = mask_m[:, None] & mask_n[None, :]

        # load neighbor indices and weights
        idx_block = tl.load(
            nbhd_idx_ptr + nb_offs[None, :] * stride_idx_m,
            mask=mask_nm,
            other=0
        ).to(tl.int32)

        w_block = tl.load(
            weights_ptr + nb_offs[None, :] * stride_w_m,
            mask=mask_nm,
            other=0.0
        )

        # gather features from FEAT
        feat_block = tl.load(
             feat_ptr + idx_block[:, :, None] * stride_f_n + offs_d[None, None, :] * stride_f_c,
             mask=mask_nm[:, :, None] & mask_d[None, None, :],
             other=0.0
         )

        # instead, perform weighted sum manually across neighborhood
        acc += tl.sum(w_block[:, :, None].to(tl.float32) * feat_block.to(tl.float32), axis=1)


    # Store back results
    tl.store(
        out_ptr,
        acc,
        mask=mask_m[:, None] & mask_d[None, :]
    )


def launch_weighted_gather_fwd_kernel(nbhd_idx, weights, feat):
    B, N, M = weights.shape
    _, N_old, C = feat.shape
    feat_new = torch.zeros((B, N, C), device=feat.device, dtype=feat.dtype)
    nbhd_idx = nbhd_idx.to(torch.int32).contiguous()

    def grid(META):
        assert META['BLOCK_M'] <= N, f"BLOCK_M ({META['BLOCK_M']}) must be less than or equal to N ({N})"
        assert META['BLOCK_D'] <= C, f"BLOCK_D ({META['BLOCK_D']}) must be less than or equal to C ({C})"
        return (triton.cdiv(N, META['BLOCK_M']), B, triton.cdiv(C, META['BLOCK_D']))

    _weighted_gather_fwd_kernel[grid](
        nbhd_idx, weights, feat, feat_new,
        B, N, M, C, N_old,
        nbhd_idx.stride(0), nbhd_idx.stride(1), nbhd_idx.stride(2),
        weights.stride(0), weights.stride(1), weights.stride(2),
        feat.stride(0), feat.stride(1), feat.stride(2),
        feat_new.stride(0), feat_new.stride(1), feat_new.stride(2),
    )
    return feat_new


@triton.heuristics({
    'BLOCK_M': lambda args: 16,                  # tokens per program
    'BLOCK_N': lambda args: max(8, min(32, args['M'])),
    'BLOCK_D': lambda args: max(32, min(128, args['C'])),
    'num_warps': lambda args: 4,
    'num_stages': lambda args: 3,
})
@triton.jit
def _weighted_gather_bwd_dfeat_kernel(
    NBHD_IDX,     # [B, N, M]  (int32/int64)
    WEIGHTS,      # [B, N, M]
    DOUT,         # [B, N, C]
    DFEAT,        # [B, N_old, C]  (to be atomic-added)
    B, N, M, C, N_OLD,
    s_idx_b, s_idx_n, s_idx_m,
    s_w_b, s_w_n, s_w_m,
    s_do_b, s_do_n, s_do_c,
    s_df_b, s_df_n, s_df_c,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tile over tokens i
    pid_b = tl.program_id(1)  # batch
    pid_d = tl.program_id(2)  # tile over channels

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_m = offs_m < N
    mask_d = offs_d < C

    # base ptrs
    nbhd_idx_ptr = NBHD_IDX + pid_b * s_idx_b + offs_m[:, None] * s_idx_n
    weights_ptr  = WEIGHTS   + pid_b * s_w_b   + offs_m[:, None] * s_w_n
    dout_ptr     = DOUT      + pid_b * s_do_b  + offs_m[:, None] * s_do_n + offs_d[None, :] * s_do_c
    dfeat_base   = DFEAT     + pid_b * s_df_b

    # load dO tile once
    dO_tile = tl.load(
        dout_ptr,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0
    ).to(tl.float32)  # [BM, BD]

    # loop neighborhood
    for nb in range(0, M, BLOCK_N):
        nb_offs = nb + tl.arange(0, BLOCK_N)
        mask_n  = nb_offs < M
        mask_nm = mask_m[:, None] & mask_n[None, :]

        idx_block = tl.load(
            nbhd_idx_ptr + nb_offs[None, :] * s_idx_m,
            mask=mask_nm,
            other=0
        ).to(tl.int32)  # [BM, BN]

        w_block = tl.load(
            weights_ptr + nb_offs[None, :] * s_w_m,
            mask=mask_nm,
            other=0.0
        ).to(tl.float32)  # [BM, BN]

        # contribution: [BM, BN, BD]
        contrib = w_block[:, :, None] * dO_tile[:, None, :]

        # atomic add into dfeat
        dfeat_ptrs = dfeat_base + idx_block[:, :, None] * s_df_n + offs_d[None, None, :] * s_df_c
        tl.atomic_add(
            dfeat_ptrs,
            contrib,
            mask=mask_nm[:, :, None] & mask_d[None, None, :]
        )


@triton.heuristics({
    'BLOCK_M': lambda args: 16,
    'BLOCK_N': lambda args: max(8, min(32, args['M'])),
    'BLOCK_D': lambda args: max(32, min(128, args['C'])),
    'num_warps': lambda args: 4,
    'num_stages': lambda args: 3,
})
@triton.jit
def _weighted_gather_bwd_dweights_kernel(
    NBHD_IDX,     # [B, N, M]
    FEAT,         # [B, N_old, C]
    DOUT,         # [B, N, C]
    DWEIGHTS,     # [B, N, M]
    B, N, M, C, N_OLD,
    s_idx_b, s_idx_n, s_idx_m,
    s_f_b, s_f_n, s_f_c,
    s_do_b, s_do_n, s_do_c,
    s_dw_b, s_dw_n, s_dw_m,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tile over tokens i
    pid_b = tl.program_id(1)  # batch

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N

    nbhd_idx_ptr = NBHD_IDX + pid_b * s_idx_b + offs_m[:, None] * s_idx_n
    dout_row_ptr = DOUT     + pid_b * s_do_b  + offs_m[:, None] * s_do_n  # we'll add channel offset inside loop
    dweights_ptr = DWEIGHTS + pid_b * s_dw_b  + offs_m[:, None] * s_dw_n

    # accumulator for dW over channels: [BM, BN]
    # (fp32 to reduce reduction error)
    # initialize to zero for the NB chunk; we’ll store per chunk
    for nb in range(0, M, BLOCK_N):
        nb_offs = nb + tl.arange(0, BLOCK_N)
        mask_n  = nb_offs < M
        mask_nm = mask_m[:, None] & mask_n[None, :]

        idx_block = tl.load(
            nbhd_idx_ptr + nb_offs[None, :] * s_idx_m,
            mask=mask_nm,
            other=0
        ).to(tl.int32)  # [BM, BN]

        # local fp32 accumulator
        dW_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # loop over channels in tiles so this kernel owns the whole reduction
        for d0 in range(0, C, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < C

            # dO: [BM, BD]
            dO_tile = tl.load(
                dout_row_ptr + offs_d[None, :] * s_do_c,
                mask=mask_m[:, None] & mask_d[None, :],
                other=0.0
            ).to(tl.float32)

            # FEAT gather: [BM, BN, BD]
            feat_base = FEAT + pid_b * s_f_b
            feat_block = tl.load(
                feat_base + idx_block[:, :, None] * s_f_n + offs_d[None, None, :] * s_f_c,
                mask=mask_nm[:, :, None] & mask_d[None, None, :],
                other=0.0
            ).to(tl.float32)

            # accumulate dot over BD
            # dW += sum_d (feat * dO)
            dW_acc += tl.sum(feat_block * dO_tile[:, None, :], axis=2)

        # store this NB chunk
        tl.store(
            dweights_ptr + nb_offs[None, :] * s_dw_m,
            dW_acc,
            mask=mask_nm
        )


def launch_weighted_gather_bwd_kernel(dout, nbhd_idx, weights, feat):
    """
    dout:  [B, N, C]
    nbhd_idx: [B, N, M]
    weights:  [B, N, M]
    feat:     [B, N_old, C]
    returns: d_weights, d_feat
    """
    B, N, C = dout.shape
    _, _, M = weights.shape
    _, N_old, _ = feat.shape

    d_weights = torch.zeros_like(weights, dtype=torch.float32)
    d_feat = torch.zeros_like(feat, dtype=torch.float32)

    # --- d_feat ---
    def grid_df(META):
        assert META['BLOCK_M'] <= N, f"BLOCK_M ({META['BLOCK_M']}) must be less than or equal to N ({N})"
        assert META['BLOCK_D'] <= C, f"BLOCK_D ({META['BLOCK_D']}) must be less than or equal to C ({C})"
        return (triton.cdiv(N, META['BLOCK_M']), B, triton.cdiv(C, META['BLOCK_D']))
    _weighted_gather_bwd_dfeat_kernel[grid_df](
        nbhd_idx, weights, dout, d_feat,
        B, N, M, C, N_old,
        nbhd_idx.stride(0), nbhd_idx.stride(1), nbhd_idx.stride(2),
        weights.stride(0), weights.stride(1), weights.stride(2),
        dout.stride(0), dout.stride(1), dout.stride(2),
        d_feat.stride(0), d_feat.stride(1), d_feat.stride(2),
    )

    # --- d_weights ---
    # (note: no pid_d; reduction across channels happens inside the kernel)
    def grid_dw(META):
        assert META['BLOCK_M'] <= N, f"BLOCK_M ({META['BLOCK_M']}) must be less than or equal to N ({N})"
        return (triton.cdiv(N, META['BLOCK_M']), B)
    _weighted_gather_bwd_dweights_kernel[grid_dw](
        nbhd_idx, feat, dout, d_weights,
        B, N, M, C, N_old,
        nbhd_idx.stride(0), nbhd_idx.stride(1), nbhd_idx.stride(2),
        feat.stride(0), feat.stride(1), feat.stride(2),
        dout.stride(0), dout.stride(1), dout.stride(2),
        d_weights.stride(0), d_weights.stride(1), d_weights.stride(2),
    )

    # convert d_weights and d_feat to dtype
    d_weights = d_weights.to(weights.dtype)
    d_feat = d_feat.to(feat.dtype)

    return d_weights, d_feat


class WeightedGatherFunction(Function):
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float16)
    def forward(ctx, nbhd_idx: torch.Tensor, weights: torch.Tensor, feat: torch.Tensor):
        """
        nbhd_idx: [B, N, M] (int32/int64, cuda)
        weights:  [B, N, M] (fp16/bf16/fp32, cuda)
        feat:     [B, N_old, C] (same dtype as weights recommended, cuda)
        returns:  [B, N, C]
        """
        assert nbhd_idx.is_cuda and weights.is_cuda and feat.is_cuda, "All inputs must be CUDA tensors."
        nbhd_idx = nbhd_idx.to(torch.int32).contiguous()
        weights = weights.contiguous()
        feat = feat.contiguous()
        out = launch_weighted_gather_fwd_kernel(nbhd_idx, weights, feat)
        ctx.save_for_backward(nbhd_idx, weights, feat)
        return out

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dout: torch.Tensor):
        """
        dout: [B, N, C]
        returns: (None, d_weights, d_feat)
        """
        (nbhd_idx, weights, feat) = ctx.saved_tensors
        assert dout.is_cuda, "dout must be CUDA."
        dout = dout.contiguous()
        d_weights, d_feat = launch_weighted_gather_bwd_kernel(dout, nbhd_idx, weights, feat)
        # nbhd_idx has no gradient
        return None, d_weights, d_feat


# Convenience API
def weighted_gather(nbhd_idx: torch.Tensor, weights: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
    return WeightedGatherFunction.apply(nbhd_idx, weights, feat)



if __name__ == "__main__":
    import torch
    import time
    from clusten import WEIGHTEDGATHERMIXEDFunction as WEIGHTEDGATHERFunction

    """
    Test the correctness of WeightedGather custom kernel
    """
    b = 100
    n = 500
    n_ = 10_000
    k = 4
    c = 128
    half = False

    # dummy data
    nn_idx = torch.randint(n_, (b, n, k)).cuda()
    nn_weights = (torch.rand(b, n, k) + 1.0).cuda()
    feature = torch.rand(b, n_, c).cuda() * 5

    if half:
        nn_weights = nn_weights.to(torch.float16)
        feature = feature.to(torch.float16)

    # nn_weights = nn_weights
    # feature = feature.to(torch.float16)

    nn_weights.requires_grad_(True)
    nn_weights.retain_grad()
    feature.requires_grad_(True)
    feature.retain_grad()

    # use the custom kernel
    # up_features = WEIGHTEDGATHERFunction.apply(nn_idx, nn_weights.to(torch.float16), feature.to(torch.float16))
    up_features = weighted_gather(nn_idx, nn_weights, feature)
    (up_features.mean() * 100).backward()
    grad_weights = nn_weights.grad.clone().detach()
    grad_feat = feature.grad.clone().detach()
    nn_weights.grad.data.zero_()
    feature.grad.data.zero_()

    # use the pytorch equivalent
    nn_features = feature.gather(index=nn_idx.view(b, -1).unsqueeze(2).expand(-1, -1, c), dim=1).reshape(b, n, k, c)
    up_features2 = nn_features.mul(nn_weights.unsqueeze(3).expand(-1, -1, -1, c)).sum(dim=2)  # b x n x c
    (up_features2.mean() * 100).backward()
    grad_weights2 = nn_weights.grad.clone().detach()
    grad_feat2 = feature.grad.clone().detach()
    nn_weights.grad.data.zero_()
    feature.grad.data.zero_()

    print(up_features[0, 0, :10].tolist())
    print(up_features2[0, 0, :10].tolist())
    print('diff of forward: ', torch.linalg.norm(up_features2 - up_features))
    print('diff of grad weights: ', torch.linalg.norm(grad_weights2 - grad_weights))
    print('diff of grad feat: ', torch.linalg.norm(grad_feat2 - grad_feat))
    print('max abs diff: ', torch.max(torch.abs(up_features2 - up_features)))
    # print(up_features.shape)
    # print(torch.argmax(torch.abs(up_features2 - up_features), dim=0))


    def test_weighted_gather_speed():
        print("Testing between triton and cuda weighted gather")
        torch.manual_seed(0)

        b = 128
        n = 1_000
        n_ = 2_000
        k = 4
        c = 256
        half = True

        nn_idx = torch.randint(n_, (b, n, k), device='cuda', dtype=torch.int32)
        nn_weights = (torch.rand(b, n, k, device='cuda') + 1.0)
        feature = torch.rand(b, n_, c, device='cuda') * 5

        if half:
            nn_weights = nn_weights.to(torch.float16)
            feature = feature.to(torch.float16)

        # Warmup
        for _ in range(3):
            nn_idx.grad = None
            nn_weights.grad = None
            feature.grad = None
            nn_weights.requires_grad_(True)
            nn_weights.retain_grad()
            feature.requires_grad_(True)
            feature.retain_grad()
            up = weighted_gather(nn_idx, nn_weights, feature)
            up.mean().backward()

        # print(weighted_gather_fwd_kernel.best_config)

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            torch.cuda.synchronize()
            nn_idx.grad = None
            nn_weights.grad = None
            feature.grad = None
            nn_weights.requires_grad_(True)
            nn_weights.retain_grad()
            feature.requires_grad_(True)
            feature.retain_grad()
            up_features = weighted_gather(nn_idx, nn_weights, feature)
            up_features.mean().backward()
            torch.cuda.synchronize()
        triton_time = (time.time() - t0) / 10
        
        # Warmup CUDA
        for _ in range(3):
            nn_idx.grad = None
            nn_weights.grad = None
            feature.grad = None
            nn_weights.requires_grad_(True)
            nn_weights.retain_grad()
            feature.requires_grad_(True)
            feature.retain_grad()
            up = WEIGHTEDGATHERFunction.apply(nn_idx.long(), nn_weights.to(torch.float16), feature.to(torch.float16))
            up.mean().backward()

        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(10):
            torch.cuda.synchronize()
            nn_idx.grad = None
            nn_weights.grad = None
            feature.grad = None
            nn_weights.requires_grad_(True)
            nn_weights.retain_grad()
            feature.requires_grad_(True)
            feature.retain_grad()
            up_features2 = WEIGHTEDGATHERFunction.apply(nn_idx.long(), nn_weights.to(torch.float16), feature.to(torch.float16))
            up_features2.mean().backward()
            torch.cuda.synchronize()
        torch_time = (time.time() - t1) / 10
        
        # correctness
        diff = torch.linalg.norm(up_features2 - up_features).item()
        max_diff = torch.max(torch.abs(up_features2 - up_features)).item()

        print(f"Shapes match: {up_features.shape == up_features2.shape}")
        print(f"L2 diff: {diff:.6f}")
        print(f"Max abs diff: {max_diff:.6e}")
        print(f"PyTorch time: {torch_time*1000:.2f} ms")
        print(f"Triton time:  {triton_time*1000:.2f} ms")
        print(f"Speedup: {torch_time / triton_time:.2f}×")

    # Run test
    test_weighted_gather_speed()
