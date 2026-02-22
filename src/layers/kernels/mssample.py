import torch
from torch.autograd import Function
from torch.amp import custom_fwd, custom_bwd
import triton
import triton.language as tl


# Autotune configurations for forward kernel
# Note: BLOCK_N must equal K since we process all neighbors at once
configs_fwd = [
    triton.Config({'BLOCK_M': BM, 'num_warps': W, 'num_stages': S})
    for BM in [32, 64, 128, 256]
    for W in [2, 4, 8]
    for S in [2, 3, 4]
]

@triton.autotune(
    configs=configs_fwd,
    key=['B', 'N'],
)
@triton.jit
def _mssample_fwd_kernel(
    SAMPLING_LOCS,   # [B, N, D] - sampling locations
    KV_POS,          # [B, N_kv, D] - key/value positions
    NB_IDX,          # [B, N, K] - neighbor indices (int32)
    POWER,           # scalar (float32)
    NN_WEIGHTS,      # [B, N, K, 1] - output softmax weights
    B, N, N_KV,
    stride_sl_b, stride_sl_n, stride_sl_d,
    stride_kv_b, stride_kv_n, stride_kv_d,
    stride_idx_b, stride_idx_n, stride_idx_k,
    stride_out_b, stride_out_n, stride_out_k, stride_out_1,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # BLOCK_N is set to K (number of neighbors) - we process all neighbors at once
    BLOCK_N: tl.constexpr = K
    """
    Forward kernel for multi-scale sampling.
    Computes: nn_weights = softmax(-power * dist(sampling_locs, neighbor_positions))
    
    For each sampling location, we:
    1. Load K neighbor indices
    2. Gather K neighbor positions from KV_POS
    3. Compute distances between sampling location and neighbors
    4. Apply learned power: logits = -power * dist
    5. Compute softmax over K neighbors
    """
    pid_m = tl.program_id(0)  # tile over N
    pid_b = tl.program_id(1)  # batch
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N
    
    # Process K neighbors at once (typically K=4)
    offs_k = tl.arange(0, BLOCK_N)
    mask_k = offs_k < K
    
    # Load neighbor indices [BLOCK_M, BLOCK_N]
    idx_ptr = NB_IDX + pid_b * stride_idx_b + offs_m[:, None] * stride_idx_n + offs_k[None, :] * stride_idx_k
    nb_indices = tl.load(
        idx_ptr,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0
    ).to(tl.int32)  # [BLOCK_M, BLOCK_N]
    
    # Initialize distance accumulator [BLOCK_M, BLOCK_N]
    dist_sq = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Accumulate squared distance across dimensions
    # We iterate over each dimension and compute the contribution to distance
    kv_base = KV_POS + pid_b * stride_kv_b
    sl_base = SAMPLING_LOCS + pid_b * stride_sl_b
    
    for d in range(D):
        # Load sampling locations for dimension d: [BLOCK_M]
        sampling_d = tl.load(
            sl_base + offs_m * stride_sl_n + d * stride_sl_d,
            mask=mask_m,
            other=0.0
        ).to(tl.float32)
        
        # Load neighbor positions for dimension d: [BLOCK_M, BLOCK_N]
        nb_pos_d = tl.load(
            kv_base + nb_indices * stride_kv_n + d * stride_kv_d,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        ).to(tl.float32)
        
        # Compute (sampling_loc[d] - nb_pos[d])^2
        # Broadcast sampling_d from [BLOCK_M] to [BLOCK_M, BLOCK_N]
        diff = sampling_d[:, None] - nb_pos_d  # [BLOCK_M, BLOCK_N]
        dist_sq += diff * diff
    
    # Compute distance: sqrt(sum(diff^2)) + eps
    dist = tl.sqrt(dist_sq + 1e-6)  # [BLOCK_M, BLOCK_N]
    
    # Load power (scalar)
    power_val = tl.load(POWER)
    
    # Compute logits: -power * dist
    logits = -power_val * dist  # [BLOCK_M, BLOCK_N]
    
    # Softmax over K dimension
    # First, find max for numerical stability
    logits_max = tl.max(logits, axis=1)[:, None]  # [BLOCK_M, 1]
    logits_shifted = logits - logits_max  # [BLOCK_M, BLOCK_N]
    
    # Compute exp
    exp_logits = tl.exp(logits_shifted)  # [BLOCK_M, BLOCK_N]
    
    # Compute sum for normalization
    exp_sum = tl.sum(exp_logits, axis=1)[:, None]  # [BLOCK_M, 1]
    
    # Normalize to get softmax weights
    nn_weights = exp_logits / (exp_sum + 1e-8)  # [BLOCK_M, BLOCK_N]
    
    # Store results [B, N, K, 1]
    out_ptr = NN_WEIGHTS + pid_b * stride_out_b + offs_m[:, None] * stride_out_n + offs_k[None, :] * stride_out_k
    tl.store(
        out_ptr,
        nn_weights,
        mask=mask_m[:, None] & mask_k[None, :]
    )


# Autotune configurations for backward kernel
# Note: BLOCK_N must equal K since we process all neighbors at once
configs_bwd = [
    triton.Config({'BLOCK_M': BM, 'num_warps': W, 'num_stages': S})
    for BM in [16, 32, 64, 128]
    for W in [2, 4, 8]
    for S in [2, 3, 4]
]

@triton.autotune(
    configs=configs_bwd,
    key=['B', 'N'],
)
@triton.jit
def _mssample_bwd_kernel(
    DOUT,            # [B, N, K, 1] - gradient of output
    SAMPLING_LOCS,   # [B, N, D]
    KV_POS,          # [B, N_kv, D]
    NB_IDX,          # [B, N, K]
    POWER,           # scalar
    NN_WEIGHTS,      # [B, N, K, 1] - forward softmax output
    DSAMPLING_LOCS,  # [B, N, D] - gradient output
    DKV_POS,         # [B, N_kv, D] - gradient output (atomic add)
    DPOWER,          # [1] - gradient output (atomic add)
    B, N, N_KV,
    stride_do_b, stride_do_n, stride_do_k, stride_do_1,
    stride_sl_b, stride_sl_n, stride_sl_d,
    stride_kv_b, stride_kv_n, stride_kv_d,
    stride_idx_b, stride_idx_n, stride_idx_k,
    stride_w_b, stride_w_n, stride_w_k, stride_w_1,
    stride_dsl_b, stride_dsl_n, stride_dsl_d,
    stride_dkv_b, stride_dkv_n, stride_dkv_d,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # BLOCK_N is set to K (number of neighbors) - we process all neighbors at once
    BLOCK_N: tl.constexpr = K
    """
    Backward kernel for multi-scale sampling.
    Computes gradients w.r.t. sampling_locs, kv_pos, and power.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N
    
    offs_k = tl.arange(0, BLOCK_N)
    mask_k = offs_k < K
    
    # Load neighbor indices [BLOCK_M, BLOCK_N]
    idx_ptr = NB_IDX + pid_b * stride_idx_b + offs_m[:, None] * stride_idx_n + offs_k[None, :] * stride_idx_k
    nb_indices = tl.load(
        idx_ptr,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0
    ).to(tl.int32)
    
    # Load forward outputs: nn_weights and dout [BLOCK_M, BLOCK_N, 1]
    # Note: both have shape [B, N, K, 1] but we only need [B, N, K] part
    w_ptr = NN_WEIGHTS + pid_b * stride_w_b + offs_m[:, None] * stride_w_n + offs_k[None, :] * stride_w_k + 0 * stride_w_1
    nn_weights_tile = tl.load(
        w_ptr,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0
    ).to(tl.float32)
    
    dout_ptr = DOUT + pid_b * stride_do_b + offs_m[:, None] * stride_do_n + offs_k[None, :] * stride_do_k + 0 * stride_do_1
    dout_tile = tl.load(
        dout_ptr,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0
    ).to(tl.float32)
    
    # Load power
    power_val = tl.load(POWER)
    
    # Recompute distances
    kv_base = KV_POS + pid_b * stride_kv_b
    sl_base = SAMPLING_LOCS + pid_b * stride_sl_b
    
    dist_sq = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # First pass: compute distances
    for d in range(D):
        # Load sampling locations for dimension d: [BLOCK_M]
        sampling_d = tl.load(
            sl_base + offs_m * stride_sl_n + d * stride_sl_d,
            mask=mask_m,
            other=0.0
        ).to(tl.float32)
        
        # Load neighbor positions for dimension d: [BLOCK_M, BLOCK_N]
        nb_pos_d = tl.load(
            kv_base + nb_indices * stride_kv_n + d * stride_kv_d,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        ).to(tl.float32)
        
        diff = sampling_d[:, None] - nb_pos_d
        dist_sq += diff * diff
    
    dist = tl.sqrt(dist_sq + 1e-6)
    
    # Compute softmax gradient: d_logits
    # For softmax: d_softmax/d_logits[i] = softmax[i] * (delta[i,j] - softmax[j])
    # So: d_loss/d_logits[i] = sum_j (d_loss/d_softmax[j] * softmax[j] * (delta[i,j] - softmax[i]))
    #                         = d_loss/d_softmax[i] * softmax[i] - softmax[i] * sum_j(d_loss/d_softmax[j] * softmax[j])
    weighted_dout = dout_tile * nn_weights_tile  # [BLOCK_M, BLOCK_N]
    sum_weighted = tl.sum(weighted_dout, axis=1)[:, None]  # [BLOCK_M, 1]
    d_logits = nn_weights_tile * (dout_tile - sum_weighted)  # [BLOCK_M, BLOCK_N]
    
    # d_logits/d_dist = -power  (since logits = -power * dist)
    # So d_loss/d_dist = d_loss/d_logits * d_logits/d_dist = d_logits * (-power)
    d_dist = d_logits * (-power_val)  # [BLOCK_M, BLOCK_N]
    
    # Gradient w.r.t. power: d_loss/d_power = sum(d_loss/d_logits * d_logits/d_power)
    # d_logits/d_power = -dist (since logits = -power * dist)
    # Only accumulate for valid elements
    valid_mask = mask_m[:, None] & mask_k[None, :]
    d_power_contrib = tl.where(valid_mask, d_logits * (-dist), 0.0)
    d_power_local = tl.sum(d_power_contrib)  # scalar per block
    # Use atomic_add to accumulate across blocks - explicitly index position 0
    tl.atomic_add(DPOWER + 0, d_power_local)
    
    # Second pass: compute gradients w.r.t. positions
    for d in range(D):
        # Reload sampling locations for dimension d: [BLOCK_M]
        sampling_d = tl.load(
            sl_base + offs_m * stride_sl_n + d * stride_sl_d,
            mask=mask_m,
            other=0.0
        ).to(tl.float32)
        
        # Reload neighbor positions for this dimension
        nb_pos_d = tl.load(
            kv_base + nb_indices * stride_kv_n + d * stride_kv_d,
            mask=valid_mask,
            other=0.0
        ).to(tl.float32)
        
        # diff = sampling - nb_pos
        diff = sampling_d[:, None] - nb_pos_d  # [BLOCK_M, BLOCK_N]
        
        # d_dist/d_diff_d = diff_d / dist
        d_diff = d_dist * diff / (dist + 1e-8)  # [BLOCK_M, BLOCK_N]
        
        # d_sampling_locs += d_diff (since diff = sampling - nb_pos)
        # Only accumulate valid elements
        valid_mask = mask_m[:, None] & mask_k[None, :]
        d_diff_masked = tl.where(valid_mask, d_diff, 0.0)
        d_sampling_d = tl.sum(d_diff_masked, axis=1)  # [BLOCK_M]
        
        # Store gradient for this dimension
        dsl_ptr = DSAMPLING_LOCS + pid_b * stride_dsl_b + offs_m * stride_dsl_n + d * stride_dsl_d
        tl.store(
            dsl_ptr,
            d_sampling_d,
            mask=mask_m
        )
        
        # d_kv_pos -= d_diff (negative because diff = sampling - nb_pos)
        # Atomic add to global memory - only for valid elements
        dkv_ptr = DKV_POS + pid_b * stride_dkv_b + nb_indices * stride_dkv_n + d * stride_dkv_d
        tl.atomic_add(
            dkv_ptr,
            -d_diff,
            mask=valid_mask
        )


def mssample_forward(sampling_locs, kv_pos, nb_idx, power, D=2):
    """
    Forward pass for multi-scale sampling.
    
    Args:
        sampling_locs: [B, N, D] - sampling locations
        kv_pos: [B, N_kv, D] - key/value positions  
        nb_idx: [B, N, K] - neighbor indices (int32)
        power: scalar tensor - learned temperature parameter
        D: spatial dimension (2, 3, or 4)
        
    Returns:
        nn_weights: [B, N, K, 1] - softmax weights over neighbors
    """
    B, N, _ = sampling_locs.shape
    _, N_kv, _ = kv_pos.shape
    _, _, K = nb_idx.shape
    
    # Ensure power is positive
    power_val = torch.clamp(power, min=1e-6)
    
    # Output tensor
    nn_weights = torch.zeros((B, N, K, 1), device=sampling_locs.device, dtype=sampling_locs.dtype)
    
    # Allocate power buffer
    power_buf = torch.tensor([power_val.item()], device=sampling_locs.device, dtype=torch.float32)
    
    grid = lambda META: (
        triton.cdiv(N, META['BLOCK_M']),
        B,
    )
    
    _mssample_fwd_kernel[grid](
        sampling_locs, kv_pos, nb_idx, power_buf, nn_weights,
        B, N, N_kv,
        sampling_locs.stride(0), sampling_locs.stride(1), sampling_locs.stride(2),
        kv_pos.stride(0), kv_pos.stride(1), kv_pos.stride(2),
        nb_idx.stride(0), nb_idx.stride(1), nb_idx.stride(2),
        nn_weights.stride(0), nn_weights.stride(1), nn_weights.stride(2), nn_weights.stride(3),
        K, D,
    )
    
    return nn_weights


def mssample_backward(dout, sampling_locs, kv_pos, nb_idx, power, nn_weights, D=2):
    """
    Backward pass for multi-scale sampling.
    
    Args:
        dout: [B, N, K, 1] - gradient of output
        sampling_locs: [B, N, D] - sampling locations (from forward)
        kv_pos: [B, N_kv, D] - key/value positions (from forward)
        nb_idx: [B, N, K] - neighbor indices (from forward)
        power: scalar tensor - learned temperature (from forward)
        nn_weights: [B, N, K, 1] - softmax weights (from forward)
        D: spatial dimension
        
    Returns:
        d_sampling_locs: [B, N, D]
        d_kv_pos: [B, N_kv, D]
        d_power: scalar tensor
    """
    B, N, _ = sampling_locs.shape
    _, N_kv, _ = kv_pos.shape
    _, _, K = nb_idx.shape
    
    # Ensure power is positive
    power_val = torch.clamp(power, min=1e-6)
    power_buf = torch.tensor([power_val.item()], device=sampling_locs.device, dtype=torch.float32)
    
    # Allocate gradients
    d_sampling_locs = torch.zeros_like(sampling_locs, dtype=torch.float32)
    d_kv_pos = torch.zeros_like(kv_pos, dtype=torch.float32)
    d_power = torch.zeros((1,), device=sampling_locs.device, dtype=torch.float32)
    
    grid = lambda META: (
        triton.cdiv(N, META['BLOCK_M']),
        B,
    )
    
    _mssample_bwd_kernel[grid](
        dout, sampling_locs, kv_pos, nb_idx, power_buf, nn_weights,
        d_sampling_locs, d_kv_pos, d_power,
        B, N, N_kv,
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        sampling_locs.stride(0), sampling_locs.stride(1), sampling_locs.stride(2),
        kv_pos.stride(0), kv_pos.stride(1), kv_pos.stride(2),
        nb_idx.stride(0), nb_idx.stride(1), nb_idx.stride(2),
        nn_weights.stride(0), nn_weights.stride(1), nn_weights.stride(2), nn_weights.stride(3),
        d_sampling_locs.stride(0), d_sampling_locs.stride(1), d_sampling_locs.stride(2),
        d_kv_pos.stride(0), d_kv_pos.stride(1), d_kv_pos.stride(2),
        K, D,
    )
    
    # Convert back to original dtype
    d_sampling_locs = d_sampling_locs.to(sampling_locs.dtype)
    d_kv_pos = d_kv_pos.to(kv_pos.dtype)
    
    return d_sampling_locs, d_kv_pos, d_power[0]


class MSSampleFunction(Function):
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float16)
    def forward(ctx, sampling_locs, kv_pos, nb_idx, power, D=2):
        """
        Args:
            sampling_locs: [B, N, D] - sampling locations
            kv_pos: [B, N_kv, D] - key/value positions
            nb_idx: [B, N, K] - neighbor indices (int32)
            power: scalar tensor - learned temperature
            D: spatial dimension (2, 3, or 4)
            
        Returns:
            nn_weights: [B, N, K, 1] - softmax weights
        """
        assert sampling_locs.is_cuda and kv_pos.is_cuda and nb_idx.is_cuda
        sampling_locs = sampling_locs.contiguous()
        kv_pos = kv_pos.contiguous()
        nb_idx = nb_idx.to(torch.int32).contiguous()
        
        nn_weights = mssample_forward(sampling_locs, kv_pos, nb_idx, power, D)
        
        ctx.save_for_backward(sampling_locs, kv_pos, nb_idx, power, nn_weights)
        ctx.D = D
        
        return nn_weights
    
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dout):
        """
        Args:
            dout: [B, N, K, 1] - gradient of output
            
        Returns:
            d_sampling_locs, d_kv_pos, None (nb_idx), d_power, None (D)
        """
        sampling_locs, kv_pos, nb_idx, power, nn_weights = ctx.saved_tensors
        D = ctx.D
        
        dout = dout.contiguous()
        
        d_sampling_locs, d_kv_pos, d_power = mssample_backward(
            dout, sampling_locs, kv_pos, nb_idx, power, nn_weights, D
        )
        
        return d_sampling_locs, d_kv_pos, None, d_power, None


# Convenience API
def mssample(sampling_locs, kv_pos, nb_idx, power, D=2):
    """
    Multi-scale sampling with learned temperature and softmax.
    
    Computes softmax weights over K nearest neighbors for each sampling location,
    with learned temperature parameter.
    
    Args:
        sampling_locs: [B, N, D] - sampling locations in D-dimensional space
        kv_pos: [B, N_kv, D] - key/value positions in D-dimensional space
        nb_idx: [B, N, K] - indices of K nearest neighbors for each sampling location
        power: scalar tensor - learned temperature parameter (applied as -power * distance)
        D: spatial dimension (2, 3, or 4)
        
    Returns:
        nn_weights: [B, N, K, 1] - softmax weights over K neighbors for each location
        
    Example:
        >>> sampling_locs = torch.randn(2, 100, 2, device='cuda')  # 2 batches, 100 queries, 2D
        >>> kv_pos = torch.randn(2, 200, 2, device='cuda')  # 200 key/value locations
        >>> nb_idx = torch.randint(0, 200, (2, 100, 4), device='cuda')  # 4 neighbors per query
        >>> power = torch.tensor(3.0, device='cuda')  # temperature parameter
        >>> weights = mssample(sampling_locs, kv_pos, nb_idx, power, D=2)
        >>> weights.shape
        torch.Size([2, 100, 4, 1])
    """
    return MSSampleFunction.apply(sampling_locs, kv_pos, nb_idx, power, D)

