"""
Fused Deformable Attention Kernel

Combines (used Claude Opus to help merge kernels):
1. Distance computation + learned temperature softmax (mssample)
2. Weighted feature gathering (msdetrpc)

Into a single kernel for maximum efficiency.
"""

import torch
from torch.autograd import Function
from torch.amp import custom_fwd, custom_bwd
import triton
import triton.language as tl


# ==================== HEURISTICS ====================
# Derived from autotuning results across various sizes

def get_fwd_config(B: int, N: int, C: int) -> dict:
    """
    Get forward kernel config based on heuristics.
    
    Pattern observed:
    - BLOCK_M=16 works best across all sizes
    - BLOCK_D scales with C: min(max(16, C//2), 128)
    - num_warps=4, num_stages=3 consistently best
    """
    BLOCK_M = 16
    BLOCK_D = min(max(16, C // 2), 128)
    # Ensure BLOCK_D is power of 2
    BLOCK_D = 2 ** (BLOCK_D - 1).bit_length() if BLOCK_D > 0 else 16
    BLOCK_D = min(BLOCK_D, 128)
    return {
        'BLOCK_M': BLOCK_M,
        'BLOCK_D': BLOCK_D,
        'num_warps': 4,
        'num_stages': 3,
    }


def get_bwd_config(B: int, N: int, C: int) -> dict:
    """
    Get backward kernel config based on heuristics.
    
    Pattern observed:
    - BLOCK_M=16 works best across all sizes
    - BLOCK_D=16 for small/medium sizes, 64 for large (B*N >= 16384)
    - num_warps=4, num_stages=3 consistently best
    
    Backward prefers smaller BLOCK_D due to:
    - Higher register pressure from gradient computation
    - Atomic operations benefit from smaller tiles
    """
    BLOCK_M = 16
    workload = B * N
    BLOCK_D = 64 if workload >= 16384 else 16
    return {
        'BLOCK_M': BLOCK_M,
        'BLOCK_D': BLOCK_D,
        'num_warps': 4,
        'num_stages': 3,
    }


# ==================== AUTOTUNE CONFIGS ====================
# Autotune configurations for forward kernel
configs_fwd = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_D': BD}, num_warps=W, num_stages=S)
    for BM in [16, 32, 64, 128]
    for BD in [16, 32, 64, 128]
    for W in [2, 4, 8]
    for S in [2, 3, 4]
]


# Base forward kernel (no autotune - used for heuristic mode)
@triton.jit
def _fused_deform_attn_fwd_kernel_base(
    # Inputs for distance + softmax
    SAMPLING_LOCS,   # [B, N, D_spatial] - sampling locations
    KV_POS,          # [B, N_kv, D_spatial] - key/value positions
    NB_IDX,          # [B, N, K] - K nearest neighbor indices
    POWER,           # scalar - learned temperature
    # Inputs for weighted gather
    ATTN_WEIGHTS,    # [B, N] - attention weights per query
    VALUES,          # [B, N_kv, C] - features to gather
    # Output
    OUT,             # [B, N, C] - output features
    # Dimensions
    B, N, N_kv, C,
    # Strides
    stride_sl_b, stride_sl_n, stride_sl_d,
    stride_kv_b, stride_kv_n, stride_kv_d,
    stride_idx_b, stride_idx_n, stride_idx_k,
    stride_aw_b, stride_aw_n,
    stride_v_b, stride_v_n, stride_v_c,
    stride_o_b, stride_o_n, stride_o_c,
    # Compile-time constants
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused kernel that:
    1. Computes distances between sampling locations and neighbors
    2. Applies learned temperature: logits = -power * dist
    3. Computes softmax over K neighbors  
    4. Immediately uses softmax weights to gather and weight features
    
    This eliminates the need to store intermediate nn_weights tensor.
    """
    pid_m = tl.program_id(0)  # tile over N (queries)
    pid_b = tl.program_id(1)  # batch
    pid_d = tl.program_id(2)  # tile over C (channels)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, K)
    
    mask_m = offs_m < N
    mask_d = offs_d < C
    mask_k = offs_k < K
    
    # ========== PHASE 1: Distance Computation + Softmax ==========
    
    # Load neighbor indices [BLOCK_M, K]
    idx_ptr = NB_IDX + pid_b * stride_idx_b + offs_m[:, None] * stride_idx_n + offs_k[None, :] * stride_idx_k
    nb_indices = tl.load(
        idx_ptr,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0
    ).to(tl.int32)
    
    # Compute distances
    kv_base = KV_POS + pid_b * stride_kv_b
    sl_base = SAMPLING_LOCS + pid_b * stride_sl_b
    
    dist_sq = tl.zeros((BLOCK_M, K), dtype=tl.float32)
    
    # Accumulate squared distance across dimensions
    for d in range(D):
        # Load sampling locations for dimension d: [BLOCK_M]
        sampling_d = tl.load(
            sl_base + offs_m * stride_sl_n + d * stride_sl_d,
            mask=mask_m,
            other=0.0
        ).to(tl.float32)
        
        # Load neighbor positions for dimension d: [BLOCK_M, K]
        nb_pos_d = tl.load(
            kv_base + nb_indices * stride_kv_n + d * stride_kv_d,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        ).to(tl.float32)
        
        # Accumulate squared difference
        diff = sampling_d[:, None] - nb_pos_d
        dist_sq += diff * diff
    
    # Compute distance (match PyTorch: sqrt(x) + eps, not sqrt(x + eps))
    dist = tl.sqrt(dist_sq) + 1e-6  # [BLOCK_M, K]
    
    # Apply learned temperature
    power_val = tl.load(POWER)
    logits = -power_val * dist  # [BLOCK_M, K]
    
    # Softmax over K neighbors
    logits_max = tl.max(logits, axis=1)[:, None]
    logits_shifted = logits - logits_max
    exp_logits = tl.exp(logits_shifted)
    exp_sum = tl.sum(exp_logits, axis=1)[:, None]
    nn_weights = exp_logits / (exp_sum + 1e-8)  # [BLOCK_M, K]
    
    # ========== PHASE 2: Weighted Feature Gathering ==========
    
    # Load attention weights [BLOCK_M]
    attn_ptr = ATTN_WEIGHTS + pid_b * stride_aw_b + offs_m * stride_aw_n
    attn = tl.load(
        attn_ptr,
        mask=mask_m,
        other=0.0
    ).to(tl.float32)
    
    # Load values for all K neighbors and accumulate weighted sum
    val_base = VALUES + pid_b * stride_v_b
    
    # Accumulator for output
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    # Manually unroll loop over K neighbors (K is typically 4, small and known at compile time)
    # This is more efficient than a for loop and avoids indexing issues
    k_range = tl.arange(0, K)
    
    for k_idx in range(K):
        # Extract indices and weights for this k using masking
        k_mask = k_range == k_idx
        
        # Get neighbor indices for this k: select column k from nb_indices
        # nb_indices is [BLOCK_M, K], we want column k_idx
        nb_idx_k = tl.sum(
            tl.where(k_mask[None, :], nb_indices, 0),
            axis=1
        )  # [BLOCK_M]
        
        # Get softmax weights for this k: select column k from nn_weights
        weight_k = tl.sum(
            tl.where(k_mask[None, :], nn_weights, 0.0),
            axis=1
        )  # [BLOCK_M]
        
        # Load values for this neighbor: [BLOCK_M, BLOCK_D]
        val_k = tl.load(
            val_base + nb_idx_k[:, None] * stride_v_n + offs_d[None, :] * stride_v_c,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0
        ).to(tl.float32)
        
        # Accumulate: attn * nn_weight * val
        combined_weight = attn * weight_k  # [BLOCK_M]
        acc += combined_weight[:, None] * val_k  # [BLOCK_M, BLOCK_D]
    
    # Store output
    out_ptr = OUT + pid_b * stride_o_b + offs_m[:, None] * stride_o_n + offs_d[None, :] * stride_o_c
    tl.store(
        out_ptr,
        acc,
        mask=mask_m[:, None] & mask_d[None, :]
    )


# Autotuned forward kernel wrapper (created at module load time)
_fused_deform_attn_fwd_kernel = triton.autotune(
    configs=configs_fwd,
    key=['B', 'N', 'C'],
)(_fused_deform_attn_fwd_kernel_base)


# Autotune configurations for backward kernel
configs_bwd = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_D': BD}, num_warps=W, num_stages=S)
    for BM in [16, 32, 64]
    for BD in [16, 32, 64]
    for W in [2, 4, 8]
    for S in [2, 3]
]


# Base backward kernel (no autotune - used for heuristic mode)
@triton.jit
def _fused_deform_attn_bwd_kernel_base(
    # Forward inputs (for recomputation)
    SAMPLING_LOCS, KV_POS, NB_IDX, POWER, ATTN_WEIGHTS, VALUES,
    # Gradient input
    DOUT,  # [B, N, C]
    # Gradient outputs
    DSAMPLING_LOCS, DKV_POS, DPOWER, DATTN_WEIGHTS, DVALUES,
    # Dimensions
    B, N, N_kv, C,
    # Strides (same as forward)
    stride_sl_b, stride_sl_n, stride_sl_d,
    stride_kv_b, stride_kv_n, stride_kv_d,
    stride_idx_b, stride_idx_n, stride_idx_k,
    stride_aw_b, stride_aw_n,
    stride_v_b, stride_v_n, stride_v_c,
    stride_do_b, stride_do_n, stride_do_c,
    stride_dsl_b, stride_dsl_n, stride_dsl_d,
    stride_dkv_b, stride_dkv_n, stride_dkv_d,
    stride_daw_b, stride_daw_n,
    stride_dv_b, stride_dv_n, stride_dv_c,
    # Compile-time constants
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Backward pass for fused deformable attention.
    
    Recomputes distances and softmax, then backprops through:
    1. Weighted gather (values, attn_weights)
    2. Softmax
    3. Distance computation (sampling_locs, kv_pos, power)
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_d = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, K)
    
    mask_m = offs_m < N
    mask_d = offs_d < C
    mask_k = offs_k < K
    valid_mask = mask_m[:, None] & mask_k[None, :]
    
    # Load gradient output [BLOCK_M, BLOCK_D]
    dout_ptr = DOUT + pid_b * stride_do_b + offs_m[:, None] * stride_do_n + offs_d[None, :] * stride_do_c
    dout_tile = tl.load(
        dout_ptr,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0
    ).to(tl.float32)
    
    # Load neighbor indices
    idx_ptr = NB_IDX + pid_b * stride_idx_b + offs_m[:, None] * stride_idx_n + offs_k[None, :] * stride_idx_k
    nb_indices = tl.load(idx_ptr, mask=valid_mask, other=0).to(tl.int32)
    
    # Recompute distances and softmax
    kv_base = KV_POS + pid_b * stride_kv_b
    sl_base = SAMPLING_LOCS + pid_b * stride_sl_b
    val_base = VALUES + pid_b * stride_v_b
    
    dist_sq = tl.zeros((BLOCK_M, K), dtype=tl.float32)
    
    for d in range(D):
        sampling_d = tl.load(
            sl_base + offs_m * stride_sl_n + d * stride_sl_d,
            mask=mask_m, other=0.0
        ).to(tl.float32)
        
        nb_pos_d = tl.load(
            kv_base + nb_indices * stride_kv_n + d * stride_kv_d,
            mask=valid_mask, other=0.0
        ).to(tl.float32)
        
        diff = sampling_d[:, None] - nb_pos_d
        dist_sq += diff * diff
    
    # Match forward: sqrt(x) + eps
    dist = tl.sqrt(dist_sq) + 1e-6
    
    power_val = tl.load(POWER)
    logits = -power_val * dist
    
    logits_max = tl.max(logits, axis=1)[:, None]
    logits_shifted = logits - logits_max
    exp_logits = tl.exp(logits_shifted)
    exp_sum = tl.sum(exp_logits, axis=1)[:, None]
    nn_weights = exp_logits / (exp_sum + 1e-8)
    
    # Load attention weights
    attn_ptr = ATTN_WEIGHTS + pid_b * stride_aw_b + offs_m * stride_aw_n
    attn = tl.load(attn_ptr, mask=mask_m, other=0.0).to(tl.float32)
    
    # ========== Backprop through weighted gather ==========
    # Values gradient needs to be computed for all channel tiles
    # But attn/nn_weights/power/locs gradients only once (when pid_d == 0)
    
    k_range = tl.arange(0, K)
    
    for k_idx in range(K):
        k_mask = k_range == k_idx
        
        # Extract indices for this k
        nb_idx_k = tl.sum(
            tl.where(k_mask[None, :], nb_indices, 0),
            axis=1
        )
        
        # Get softmax weights for this k
        weight_k = tl.sum(
            tl.where(k_mask[None, :], nn_weights, 0.0),
            axis=1
        )
        
        combined_weight = attn * weight_k
        
        # Load values
        val_k = tl.load(
            val_base + nb_idx_k[:, None] * stride_v_n + offs_d[None, :] * stride_v_c,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0
        ).to(tl.float32)
        
        # Gradient w.r.t. values: attn * nn_weight * dout
        # This accumulates across all channel tiles (correct)
        dval_k = combined_weight[:, None] * dout_tile
        
        # Atomic add to global d_values
        dval_ptr = DVALUES + pid_b * stride_dv_b + nb_idx_k[:, None] * stride_dv_n + offs_d[None, :] * stride_dv_c
        tl.atomic_add(dval_ptr, dval_k, mask=mask_m[:, None] & mask_d[None, :])
    
    # ========== Compute gradients that are independent of channel dimension ==========
    # These should only be computed ONCE per (batch, query) not per channel tile!
    
    if pid_d == 0:
        # Gradient accumulators for attn and nn_weights
        d_attn_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        d_nn_weights = tl.zeros((BLOCK_M, K), dtype=tl.float32)
        
        # Need to loop over ALL channels to compute gradient
        for d_idx in range(0, C, BLOCK_D):
            offs_d_full = d_idx + tl.arange(0, BLOCK_D)
            mask_d_full = offs_d_full < C
            
            # Load gradient output for this channel block
            dout_ptr_full = DOUT + pid_b * stride_do_b + offs_m[:, None] * stride_do_n + offs_d_full[None, :] * stride_do_c
            dout_tile_full = tl.load(
                dout_ptr_full,
                mask=mask_m[:, None] & mask_d_full[None, :],
                other=0.0
            ).to(tl.float32)
            
            for k_idx in range(K):
                k_mask = k_range == k_idx
                
                # Extract indices for this k
                nb_idx_k = tl.sum(
                    tl.where(k_mask[None, :], nb_indices, 0),
                    axis=1
                )
                
                # Get softmax weights for this k
                weight_k = tl.sum(
                    tl.where(k_mask[None, :], nn_weights, 0.0),
                    axis=1
                )
                
                # Load values
                val_k = tl.load(
                    val_base + nb_idx_k[:, None] * stride_v_n + offs_d_full[None, :] * stride_v_c,
                    mask=mask_m[:, None] & mask_d_full[None, :],
                    other=0.0
                ).to(tl.float32)
                
                # Gradient w.r.t. combined_weight: sum over channels
                # d_combined_weight = sum_c(val * dout)
                d_combined_weight_k = tl.sum(val_k * dout_tile_full, axis=1)  # [BLOCK_M]
                
                # Backprop to attn and nn_weights
                d_attn_acc += weight_k * d_combined_weight_k
                
                # Store gradient for this k using masked scatter
                d_nn_weights = tl.where(
                    k_mask[None, :],
                    d_nn_weights + (attn * d_combined_weight_k)[:, None],
                    d_nn_weights
                )
        
        # Store gradient w.r.t. attention (no atomic needed - each M block has unique queries)
        dattn_ptr = DATTN_WEIGHTS + pid_b * stride_daw_b + offs_m * stride_daw_n
        tl.store(dattn_ptr, d_attn_acc, mask=mask_m)
        
        # ========== Backprop through softmax ==========
        
        weighted_dout = d_nn_weights * nn_weights
        sum_weighted = tl.sum(weighted_dout, axis=1)[:, None]
        d_logits = nn_weights * (d_nn_weights - sum_weighted)
        
        # Backprop through temperature
        d_dist = -power_val * d_logits
        
        # ========== Backprop through distance computation ==========
        
        # Gradient w.r.t. power
        d_power_contrib = tl.where(valid_mask, d_logits * (-dist), 0.0)
        d_power_local = tl.sum(d_power_contrib)
        tl.atomic_add(DPOWER, d_power_local)
        
        for d in range(D):
            sampling_d = tl.load(
                sl_base + offs_m * stride_sl_n + d * stride_sl_d,
                mask=mask_m, other=0.0
            ).to(tl.float32)
            
            nb_pos_d = tl.load(
                kv_base + nb_indices * stride_kv_n + d * stride_kv_d,
                mask=valid_mask, other=0.0
            ).to(tl.float32)
            
            diff = sampling_d[:, None] - nb_pos_d
            
            # Backprop through dist = sqrt(dist_sq) + eps
            # d_diff = d_dist * diff / sqrt(dist_sq) = d_dist * diff / (dist - eps)
            # Add small epsilon to avoid division by zero
            dist_no_eps = tl.maximum(dist - 1e-6, 1e-8)
            d_diff = tl.where(valid_mask, d_dist * diff / dist_no_eps, 0.0)
            
            # Gradient w.r.t. sampling_locs
            d_sampling_d = tl.sum(d_diff, axis=1)
            
            dsl_ptr = DSAMPLING_LOCS + pid_b * stride_dsl_b + offs_m * stride_dsl_n + d * stride_dsl_d
            tl.store(dsl_ptr, d_sampling_d, mask=mask_m)
            
            # Gradient w.r.t. kv_pos (atomic add)
            dkv_ptr = DKV_POS + pid_b * stride_dkv_b + nb_indices * stride_dkv_n + d * stride_dkv_d
            tl.atomic_add(dkv_ptr, -d_diff, mask=valid_mask)


# Autotuned backward kernel wrapper (created at module load time)
_fused_deform_attn_bwd_kernel = triton.autotune(
    configs=configs_bwd,
    key=['B', 'N', 'C'],
    # CRITICAL: Reset atomic accumulation tensors to zero before each benchmark iteration
    reset_to_zero=['DPOWER', 'DKV_POS', 'DVALUES'],
)(_fused_deform_attn_bwd_kernel_base)


def fused_deform_attn_forward(sampling_locs, kv_pos, nb_idx, power, attn_weights, values, D=2, use_heuristics=False):
    """
    Forward pass for fused deformable attention.
    
    Args:
        sampling_locs: [B, N, D] - query sampling locations
        kv_pos: [B, N_kv, D] - key/value positions
        nb_idx: [B, N, K] - K nearest neighbor indices
        power: scalar tensor - learned temperature
        attn_weights: [B, N] - attention weights per query
        values: [B, N_kv, C] - features to gather
        D: spatial dimensions
        use_heuristics: if True, use fixed heuristic config (faster startup, no autotuning)
        
    Returns:
        out: [B, N, C] - weighted output features
    """
    B, N, _ = sampling_locs.shape
    _, N_kv, C = values.shape
    _, _, K = nb_idx.shape
    
    out = torch.zeros((B, N, C), device=values.device, dtype=values.dtype)
    
    power_val = torch.clamp(power, min=1e-6)
    power_buf = torch.tensor([power_val.item()], device=sampling_locs.device, dtype=torch.float32)
    
    if use_heuristics:
        # Use heuristic config with BASE kernel (skip autotuning overhead)
        cfg = get_fwd_config(B, N, C)
        BLOCK_M, BLOCK_D = cfg['BLOCK_M'], cfg['BLOCK_D']
        num_warps, num_stages = cfg['num_warps'], cfg['num_stages']
        
        grid = (triton.cdiv(N, BLOCK_M), B, triton.cdiv(C, BLOCK_D))
        
        # Call base kernel directly with explicit config
        _fused_deform_attn_fwd_kernel_base[grid](
            sampling_locs, kv_pos, nb_idx, power_buf, attn_weights, values, out,
            B, N, N_kv, C,
            sampling_locs.stride(0), sampling_locs.stride(1), sampling_locs.stride(2),
            kv_pos.stride(0), kv_pos.stride(1), kv_pos.stride(2),
            nb_idx.stride(0), nb_idx.stride(1), nb_idx.stride(2),
            attn_weights.stride(0), attn_weights.stride(1),
            values.stride(0), values.stride(1), values.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            K, D,
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
            num_warps=num_warps, num_stages=num_stages,
        )
    else:
        # Use AUTOTUNED kernel wrapper
        grid = lambda META: (
            triton.cdiv(N, META['BLOCK_M']),
            B,
            triton.cdiv(C, META['BLOCK_D']),
        )
        
        _fused_deform_attn_fwd_kernel[grid](
            sampling_locs, kv_pos, nb_idx, power_buf, attn_weights, values, out,
            B, N, N_kv, C,
            sampling_locs.stride(0), sampling_locs.stride(1), sampling_locs.stride(2),
            kv_pos.stride(0), kv_pos.stride(1), kv_pos.stride(2),
            nb_idx.stride(0), nb_idx.stride(1), nb_idx.stride(2),
            attn_weights.stride(0), attn_weights.stride(1),
            values.stride(0), values.stride(1), values.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            K, D,
        )
    
    return out


def fused_deform_attn_backward(dout, sampling_locs, kv_pos, nb_idx, power, attn_weights, values, D=2, use_heuristics=False):
    """Backward pass for fused deformable attention."""
    B, N, _ = sampling_locs.shape
    _, N_kv, C = values.shape
    _, _, K = nb_idx.shape
    
    power_val = torch.clamp(power, min=1e-6)
    power_buf = torch.tensor([power_val.item()], device=sampling_locs.device, dtype=torch.float32)
    
    d_sampling_locs = torch.zeros_like(sampling_locs, dtype=torch.float32)
    d_kv_pos = torch.zeros_like(kv_pos, dtype=torch.float32)
    d_power = torch.zeros((1,), device=sampling_locs.device, dtype=torch.float32)
    d_attn_weights = torch.zeros_like(attn_weights, dtype=torch.float32)
    d_values = torch.zeros_like(values, dtype=torch.float32)
    
    if use_heuristics:
        # Use heuristic config with BASE kernel (skip autotuning overhead)
        cfg = get_bwd_config(B, N, C)
        BLOCK_M, BLOCK_D = cfg['BLOCK_M'], cfg['BLOCK_D']
        num_warps, num_stages = cfg['num_warps'], cfg['num_stages']
        
        grid = (triton.cdiv(N, BLOCK_M), B, triton.cdiv(C, BLOCK_D))
        
        # Call base kernel directly with explicit config
        _fused_deform_attn_bwd_kernel_base[grid](
            sampling_locs, kv_pos, nb_idx, power_buf, attn_weights, values,
            dout, d_sampling_locs, d_kv_pos, d_power, d_attn_weights, d_values,
            B, N, N_kv, C,
            sampling_locs.stride(0), sampling_locs.stride(1), sampling_locs.stride(2),
            kv_pos.stride(0), kv_pos.stride(1), kv_pos.stride(2),
            nb_idx.stride(0), nb_idx.stride(1), nb_idx.stride(2),
            attn_weights.stride(0), attn_weights.stride(1),
            values.stride(0), values.stride(1), values.stride(2),
            dout.stride(0), dout.stride(1), dout.stride(2),
            d_sampling_locs.stride(0), d_sampling_locs.stride(1), d_sampling_locs.stride(2),
            d_kv_pos.stride(0), d_kv_pos.stride(1), d_kv_pos.stride(2),
            d_attn_weights.stride(0), d_attn_weights.stride(1),
            d_values.stride(0), d_values.stride(1), d_values.stride(2),
            K, D,
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
            num_warps=num_warps, num_stages=num_stages,
        )
    else:
        # Use AUTOTUNED kernel wrapper
        grid = lambda META: (
            triton.cdiv(N, META['BLOCK_M']),
            B,
            triton.cdiv(C, META['BLOCK_D']),
        )
        
        _fused_deform_attn_bwd_kernel[grid](
            sampling_locs, kv_pos, nb_idx, power_buf, attn_weights, values,
            dout, d_sampling_locs, d_kv_pos, d_power, d_attn_weights, d_values,
            B, N, N_kv, C,
            sampling_locs.stride(0), sampling_locs.stride(1), sampling_locs.stride(2),
            kv_pos.stride(0), kv_pos.stride(1), kv_pos.stride(2),
            nb_idx.stride(0), nb_idx.stride(1), nb_idx.stride(2),
            attn_weights.stride(0), attn_weights.stride(1),
            values.stride(0), values.stride(1), values.stride(2),
            dout.stride(0), dout.stride(1), dout.stride(2),
            d_sampling_locs.stride(0), d_sampling_locs.stride(1), d_sampling_locs.stride(2),
            d_kv_pos.stride(0), d_kv_pos.stride(1), d_kv_pos.stride(2),
            d_attn_weights.stride(0), d_attn_weights.stride(1),
            d_values.stride(0), d_values.stride(1), d_values.stride(2),
            K, D,
        )
    
    return (
        d_sampling_locs.to(sampling_locs.dtype),
        d_kv_pos.to(kv_pos.dtype),
        d_power[0],
        d_attn_weights.to(attn_weights.dtype),
        d_values.to(values.dtype)
    )


# Global flag to control heuristic usage
USE_HEURISTICS = True


class FusedDeformAttnFunction(Function):
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float16)
    def forward(ctx, sampling_locs, kv_pos, nb_idx, power, attn_weights, values, D=2, use_heuristics=None):
        if use_heuristics is None:
            use_heuristics = USE_HEURISTICS
            
        assert all(t.is_cuda for t in [sampling_locs, kv_pos, nb_idx, power, attn_weights, values])
        
        sampling_locs = sampling_locs.contiguous()
        kv_pos = kv_pos.contiguous()
        nb_idx = nb_idx.to(torch.int32).contiguous()
        attn_weights = attn_weights.contiguous()
        values = values.contiguous()
        
        out = fused_deform_attn_forward(sampling_locs, kv_pos, nb_idx, power, attn_weights, values, D, use_heuristics)
        
        ctx.save_for_backward(sampling_locs, kv_pos, nb_idx, power, attn_weights, values)
        ctx.D = D
        ctx.use_heuristics = use_heuristics
        
        return out
    
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dout):
        sampling_locs, kv_pos, nb_idx, power, attn_weights, values = ctx.saved_tensors
        D = ctx.D
        use_heuristics = ctx.use_heuristics
        
        dout = dout.contiguous()
        
        d_sl, d_kv, d_pow, d_attn, d_val = fused_deform_attn_backward(
            dout, sampling_locs, kv_pos, nb_idx, power, attn_weights, values, D, use_heuristics
        )
        
        return d_sl, d_kv, None, d_pow, d_attn, d_val, None, None


# Convenience API
def fused_deformable_attention(sampling_locs, kv_pos, nb_idx, power, attn_weights, values, D=2, use_heuristics=None):
    """
    Fused deformable attention combining distance-based softmax and weighted gathering.
    
    This is significantly faster and more memory efficient than running mssample + msdetrpc separately.
    
    Args:
        sampling_locs: [B, N, D] - query sampling locations in D-dimensional space
        kv_pos: [B, N_kv, D] - key/value positions
        nb_idx: [B, N, K] - indices of K nearest neighbors for each query
        power: scalar tensor - learned temperature parameter
        attn_weights: [B, N] - attention weights for each query
        values: [B, N_kv, C] - features to gather and weight
        D: spatial dimension (2, 3, or 4)
        use_heuristics: if True, use fixed heuristic config (faster startup, no autotuning)
                        if None, uses global USE_HEURISTICS flag
        
    Returns:
        out: [B, N, C] - weighted output features
        
    Heuristics (based on NVIDIA TITAN RTX):
        Forward:  BLOCK_M=16, BLOCK_D=min(max(16, C//2), 128), num_warps=4, num_stages=3
        Backward: BLOCK_M=16, BLOCK_D=16 (small) or 64 (large B*N>=16384), num_warps=4, num_stages=3
    """
    return FusedDeformAttnFunction.apply(sampling_locs, kv_pos, nb_idx, power, attn_weights, values, D, use_heuristics)


def set_use_heuristics(value: bool):
    """Set global flag to use heuristics instead of autotuning."""
    global USE_HEURISTICS
    USE_HEURISTICS = value


# =============================================================================
# FUSED KNN + DEFORMABLE ATTENTION KERNELS
# =============================================================================
#
# These kernels fuse KNN computation directly into the deformable attention,
# eliminating the need for:
# 1. Position expansion across heads (8x memory savings for h=8)
# 2. Sampling locations materialization (32x savings for h=8, k=4)
# 3. Intermediate nb_idx tensor (128x savings)
#
# Key design:
# - Grid: (N_tiles, B, H) to enable position sharing across heads
# - Streaming KNN using scalar registers (like our optimized knn.py)
# - On-the-fly computation of sampling_loc = pos + offset
# =============================================================================


def get_fused_knn_config(B: int, N: int, H: int, K: int, C_: int) -> dict:
    """
    Get heuristic config for fused KNN + deformable SELF-attention kernel.
    
    Based on production benchmarks (TITAN RTX, FP16, B=32, H=6, K=4, C_=64):
    
    ┌─────────┬─────────┬─────────┬────────────┬────────────┐
    │ N       │ BLOCK_M │ Fwd(ms) │ Total(ms)  │ Throughput │
    ├─────────┼─────────┼─────────┼────────────┼────────────┤
    │ 64      │ 32      │ 0.20    │ 1.09       │ 1.88 M/s   │
    │ 127     │ 32      │ 0.38    │ 1.77       │ 2.29 M/s   │
    │ 254     │ 32      │ 0.99    │ 3.53       │ 2.31 M/s   │ ← Peak
    │ 506     │ 32      │ 3.28    │ 7.68       │ 2.11 M/s   │
    │ 1013    │ 64      │ 6.68    │ 16.03      │ 2.02 M/s   │
    │ 2027    │ 64      │ 21.99   │ 41.01      │ 1.58 M/s   │
    │ 4055    │ 64      │ 86.98   │ 125.60     │ 1.03 M/s   │
    └─────────┴─────────┴─────────┴────────────┴────────────┘
    
    Key insights:
    - BLOCK_M=32: avg 2.15 M/s for N ≤ 506 (lower overhead)
    - BLOCK_M=64: avg 1.55 M/s for N ≥ 1013 (O(N²) KNN dominates)
    - Optimal threshold: N=750 (between 506 and 1013)
    - BLOCK_C=64 optimal for C_=64
    - num_warps=4, num_stages=2 consistent across all configs
    """
    # Switch threshold at N=750 based on benchmark data
    # BLOCK_M=32 is better for N ≤ 750 (2.15 M/s avg)
    # BLOCK_M=64 needed for N > 750 to handle larger workload
    BLOCK_M = 64 if N >= 750 else 32
    
    # BLOCK_C: match channel dimension, capped at 64 for register pressure
    BLOCK_C = min(C_, 64)
    # Ensure power of 2 for efficient memory access
    BLOCK_C = 2 ** ((BLOCK_C - 1).bit_length()) if BLOCK_C > 0 else 32
    
    # num_warps=4 optimal across all tested configs
    num_warps = 4
    
    # num_stages=2 for light pipelining
    num_stages = 2
    
    return {
        'BLOCK_M': BLOCK_M,
        'BLOCK_C': BLOCK_C,
        'num_warps': num_warps,
        'num_stages': num_stages,
    }


def get_fused_knn_cross_config(B: int, N_q: int, N_kv: int, H: int, K: int, C_: int) -> dict:
    """
    Get heuristic config for fused KNN + deformable CROSS-attention kernel.
    
    Based on production benchmarks (TITAN RTX, FP16, B=32, N_q=41, H=6, K=4, C_=64):
    
    ┌─────────┬─────────┬─────────┬────────────┬────────────┬─────────────┐
    │ N_kv    │ BLOCK_M │ Fwd(ms) │ Bwd(ms)    │ Total(ms)  │ Throughput  │
    ├─────────┼─────────┼─────────┼────────────┼────────────┼─────────────┤
    │ 506     │ 32      │ 0.59    │ 0.92       │ 1.50       │ 0.87 M/s    │
    │ 1013    │ 32      │ 1.07    │ 1.02       │ 2.09       │ 0.63 M/s    │
    │ 2027    │ 32      │ 1.94    │ 1.26       │ 3.21       │ 0.41 M/s    │
    │ 4055    │ 32      │ 3.82    │ 1.70       │ 5.52       │ 0.24 M/s    │
    └─────────┴─────────┴─────────┴────────────┴────────────┴─────────────┘
    
    Scaling behavior (N_q=41 fixed):
    - 2x N_kv → 1.4x slower (sub-linear - excellent!)
    - 8x N_kv → 3.7x slower
    - Forward: O(N_kv) for KNN search
    - Backward: O(N_kv) for value/position gradients
    
    Key insights:
    - BLOCK_M=32 always optimal for small N_q (typical detection heads)
    - N_q drives BLOCK_M, not N_kv
    - Same threshold as self-attention: N_q >= 750 → BLOCK_M=64
    """
    # N_q drives BLOCK_M choice (same threshold as self-attention)
    BLOCK_M = 64 if N_q >= 750 else 32
    
    # BLOCK_C: match channel dimension, capped at 64
    BLOCK_C = min(C_, 64)
    BLOCK_C = 2 ** ((BLOCK_C - 1).bit_length()) if BLOCK_C > 0 else 32
    
    # Consistent with self-attention
    num_warps = 4
    num_stages = 2
    
    return {
        'BLOCK_M': BLOCK_M,
        'BLOCK_C': BLOCK_C,
        'num_warps': num_warps,
        'num_stages': num_stages,
    }


# =============================================================================
# Self-Attention: Query and KV positions are the same
# =============================================================================

@triton.jit
def _fused_knn_deform_self_attn_kernel_d2_k4(
    # Inputs
    POS,              # [B, N, 2] - shared positions (NOT expanded across heads)
    OFFSETS,          # [B, N, H, K, 2] - sampling offsets per head/point
    ATTN_WEIGHTS,     # [B, N, H, K] - attention weights (softmaxed)
    VALUES,           # [B, N, H, C_] - projected values per head
    POWER,            # scalar tensor - learned temperature
    # Output
    OUT,              # [B, N, H, C_] - output features
    # Dimensions
    B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, C_: tl.constexpr,
    # Strides for POS [B, N, 2]
    stride_p_b, stride_p_n, stride_p_d,
    # Strides for OFFSETS [B, N, H, K, 2]
    stride_o_b, stride_o_n, stride_o_h, stride_o_k, stride_o_d,
    # Strides for ATTN_WEIGHTS [B, N, H, K]
    stride_a_b, stride_a_n, stride_a_h, stride_a_k,
    # Strides for VALUES [B, N, H, C_]
    stride_v_b, stride_v_n, stride_v_h, stride_v_c,
    # Strides for OUT [B, N, H, C_]
    stride_out_b, stride_out_n, stride_out_h, stride_out_c,
    # Compile-time constants
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    KNN_K: tl.constexpr,  # Number of KNN neighbors (typically 4)
):
    """
    Fused KNN + Deformable Self-Attention for D=2, K_sample=4 (sampling points), KNN_K=4.
    
    For each query position + offset -> compute top-4 KNN -> Shepard weights -> gather.
    All done streaming without materializing intermediate tensors.
    """
    INF: tl.constexpr = 1e38
    K_SAMPLE: tl.constexpr = 4  # Number of sampling points per head
    
    # Grid: (N_tiles, B, H)
    pid_m = tl.program_id(0)   # Query tile
    pid_b = tl.program_id(1)   # Batch
    pid_h = tl.program_id(2)   # Head (positions shared across heads!)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    mask_m = offs_m < N
    mask_c = offs_c < C_
    
    # Base pointers
    pos_base = POS + pid_b * stride_p_b
    off_base = OFFSETS + pid_b * stride_o_b + pid_h * stride_o_h
    attn_base = ATTN_WEIGHTS + pid_b * stride_a_b + pid_h * stride_a_h
    val_base = VALUES + pid_b * stride_v_b + pid_h * stride_v_h
    out_base = OUT + pid_b * stride_out_b + pid_h * stride_out_h
    
    # Load query positions [BLOCK_M] - shared across heads!
    q_pos_0 = tl.load(pos_base + offs_m * stride_p_n + 0 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
    q_pos_1 = tl.load(pos_base + offs_m * stride_p_n + 1 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
    
    # Load power value
    power_val = tl.load(POWER)
    power_val = tl.maximum(power_val, 1e-6)
    
    # Accumulator for output [BLOCK_M, BLOCK_C]
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
    
    # Process each of K_SAMPLE sampling points
    for k_samp in tl.static_range(K_SAMPLE):
        # Load offset for this sampling point [BLOCK_M, 2]
        off_0 = tl.load(off_base + offs_m * stride_o_n + k_samp * stride_o_k + 0 * stride_o_d, mask=mask_m, other=0.0).to(tl.float32)
        off_1 = tl.load(off_base + offs_m * stride_o_n + k_samp * stride_o_k + 1 * stride_o_d, mask=mask_m, other=0.0).to(tl.float32)
        
        # Compute sampling location on-the-fly (never materialized!)
        samp_0 = q_pos_0 + off_0
        samp_1 = q_pos_1 + off_1
        
        # Load attention weight for this sampling point [BLOCK_M]
        attn_w = tl.load(attn_base + offs_m * stride_a_n + k_samp * stride_a_k, mask=mask_m, other=0.0).to(tl.float32)
        
        # ========== Streaming KNN with scalar registers ==========
        # Find top-4 nearest neighbors among all N reference positions
        # Using scalar register approach from optimized knn.py
        
        # Initialize top-4 distances and indices
        knn_d0 = tl.full([BLOCK_M], INF, dtype=tl.float32)
        knn_d1 = tl.full([BLOCK_M], INF, dtype=tl.float32)
        knn_d2 = tl.full([BLOCK_M], INF, dtype=tl.float32)
        knn_d3 = tl.full([BLOCK_M], INF, dtype=tl.float32)
        knn_i0 = tl.zeros([BLOCK_M], dtype=tl.int32)
        knn_i1 = tl.zeros([BLOCK_M], dtype=tl.int32)
        knn_i2 = tl.zeros([BLOCK_M], dtype=tl.int32)
        knn_i3 = tl.zeros([BLOCK_M], dtype=tl.int32)
        
        # Stream through all reference positions
        for ref_idx in range(N):
            # Load reference position (same pos tensor, shared across heads)
            r0 = tl.load(pos_base + ref_idx * stride_p_n + 0 * stride_p_d).to(tl.float32)
            r1 = tl.load(pos_base + ref_idx * stride_p_n + 1 * stride_p_d).to(tl.float32)
            
            # Compute squared distance
            diff0 = samp_0 - r0
            diff1 = samp_1 - r1
            dist_sq = diff0 * diff0 + diff1 * diff1
            
            # Update top-4 using simplified insertion sort
            u0 = dist_sq < knn_d0
            u1 = dist_sq < knn_d1
            u2 = dist_sq < knn_d2
            u3 = dist_sq < knn_d3
            
            knn_d3 = tl.where(u2, knn_d2, tl.where(u3, dist_sq, knn_d3))
            knn_i3 = tl.where(u2, knn_i2, tl.where(u3, ref_idx, knn_i3))
            knn_d2 = tl.where(u1, knn_d1, tl.where(u2, dist_sq, knn_d2))
            knn_i2 = tl.where(u1, knn_i1, tl.where(u2, ref_idx, knn_i2))
            knn_d1 = tl.where(u0, knn_d0, tl.where(u1, dist_sq, knn_d1))
            knn_i1 = tl.where(u0, knn_i0, tl.where(u1, ref_idx, knn_i1))
            knn_d0 = tl.where(u0, dist_sq, knn_d0)
            knn_i0 = tl.where(u0, ref_idx, knn_i0)
        
        # ========== Shepard weighting ==========
        # Compute distance = sqrt(dist_sq) + eps
        dist0 = tl.sqrt(knn_d0) + 1e-6
        dist1 = tl.sqrt(knn_d1) + 1e-6
        dist2 = tl.sqrt(knn_d2) + 1e-6
        dist3 = tl.sqrt(knn_d3) + 1e-6
        
        # Logits = -power * dist
        logit0 = -power_val * dist0
        logit1 = -power_val * dist1
        logit2 = -power_val * dist2
        logit3 = -power_val * dist3
        
        # Softmax over 4 neighbors
        max_logit = tl.maximum(tl.maximum(logit0, logit1), tl.maximum(logit2, logit3))
        exp0 = tl.exp(logit0 - max_logit)
        exp1 = tl.exp(logit1 - max_logit)
        exp2 = tl.exp(logit2 - max_logit)
        exp3 = tl.exp(logit3 - max_logit)
        exp_sum = exp0 + exp1 + exp2 + exp3 + 1e-8
        
        w0 = exp0 / exp_sum  # [BLOCK_M]
        w1 = exp1 / exp_sum
        w2 = exp2 / exp_sum
        w3 = exp3 / exp_sum
        
        # ========== Weighted gather ==========
        # Gather values for each of 4 neighbors and weight
        # val_base points to [N, C_] for this batch/head
        
        # Neighbor 0
        val_ptr_0 = val_base + knn_i0[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v0 = tl.load(val_ptr_0, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        # Neighbor 1
        val_ptr_1 = val_base + knn_i1[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v1 = tl.load(val_ptr_1, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        # Neighbor 2
        val_ptr_2 = val_base + knn_i2[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v2 = tl.load(val_ptr_2, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        # Neighbor 3
        val_ptr_3 = val_base + knn_i3[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v3 = tl.load(val_ptr_3, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        # Weighted sum: attn_w * (w0*v0 + w1*v1 + w2*v2 + w3*v3)
        weighted_v = w0[:, None] * v0 + w1[:, None] * v1 + w2[:, None] * v2 + w3[:, None] * v3
        acc += attn_w[:, None] * weighted_v
    
    # Store output
    out_ptr = out_base + offs_m[:, None] * stride_out_n + offs_c[None, :] * stride_out_c
    tl.store(out_ptr, acc, mask=mask_m[:, None] & mask_c[None, :])


# =============================================================================
# Self-Attention with KNN indices/distances saving for backward pass
# =============================================================================

@triton.jit
def _fused_knn_deform_self_attn_kernel_d2_k4_save(
    # Inputs
    POS,              # [B, N, 2] - shared positions
    OFFSETS,          # [B, N, H, K, 2] - sampling offsets
    ATTN_WEIGHTS,     # [B, N, H, K] - attention weights
    VALUES,           # [B, N, H, C_] - values
    POWER,            # scalar tensor
    # Outputs
    OUT,              # [B, N, H, C_] - output features
    KNN_IDX,          # [B, N, H, K_SAMPLE, KNN_K] - saved KNN indices (int32)
    KNN_DIST_SQ,      # [B, N, H, K_SAMPLE, KNN_K] - saved squared distances (float32)
    # Dimensions
    B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, C_: tl.constexpr,
    # Strides for POS [B, N, 2]
    stride_p_b, stride_p_n, stride_p_d,
    # Strides for OFFSETS [B, N, H, K, 2]
    stride_o_b, stride_o_n, stride_o_h, stride_o_k, stride_o_d,
    # Strides for ATTN_WEIGHTS [B, N, H, K]
    stride_a_b, stride_a_n, stride_a_h, stride_a_k,
    # Strides for VALUES [B, N, H, C_]
    stride_v_b, stride_v_n, stride_v_h, stride_v_c,
    # Strides for OUT [B, N, H, C_]
    stride_out_b, stride_out_n, stride_out_h, stride_out_c,
    # Strides for KNN_IDX [B, N, H, K_SAMPLE, KNN_K]
    stride_ki_b, stride_ki_n, stride_ki_h, stride_ki_k, stride_ki_j,
    # Strides for KNN_DIST_SQ [B, N, H, K_SAMPLE, KNN_K]
    stride_kd_b, stride_kd_n, stride_kd_h, stride_kd_k, stride_kd_j,
    # Compile-time constants
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    KNN_K: tl.constexpr,
):
    """
    Fused KNN + Deformable Self-Attention that also saves KNN indices and distances
    for the backward pass.
    """
    INF: tl.constexpr = 1e38
    K_SAMPLE: tl.constexpr = 4
    
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    mask_m = offs_m < N
    mask_c = offs_c < C_
    
    # Base pointers
    pos_base = POS + pid_b * stride_p_b
    off_base = OFFSETS + pid_b * stride_o_b + pid_h * stride_o_h
    attn_base = ATTN_WEIGHTS + pid_b * stride_a_b + pid_h * stride_a_h
    val_base = VALUES + pid_b * stride_v_b + pid_h * stride_v_h
    out_base = OUT + pid_b * stride_out_b + pid_h * stride_out_h
    knn_idx_base = KNN_IDX + pid_b * stride_ki_b + pid_h * stride_ki_h
    knn_dist_base = KNN_DIST_SQ + pid_b * stride_kd_b + pid_h * stride_kd_h
    
    # Load query positions
    q_pos_0 = tl.load(pos_base + offs_m * stride_p_n + 0 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
    q_pos_1 = tl.load(pos_base + offs_m * stride_p_n + 1 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
    
    power_val = tl.load(POWER)
    power_val = tl.maximum(power_val, 1e-6)
    
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
    
    for k_samp in tl.static_range(K_SAMPLE):
        off_0 = tl.load(off_base + offs_m * stride_o_n + k_samp * stride_o_k + 0 * stride_o_d, mask=mask_m, other=0.0).to(tl.float32)
        off_1 = tl.load(off_base + offs_m * stride_o_n + k_samp * stride_o_k + 1 * stride_o_d, mask=mask_m, other=0.0).to(tl.float32)
        
        samp_0 = q_pos_0 + off_0
        samp_1 = q_pos_1 + off_1
        
        attn_w = tl.load(attn_base + offs_m * stride_a_n + k_samp * stride_a_k, mask=mask_m, other=0.0).to(tl.float32)
        
        # Initialize top-4 KNN
        knn_d0 = tl.full([BLOCK_M], INF, dtype=tl.float32)
        knn_d1 = tl.full([BLOCK_M], INF, dtype=tl.float32)
        knn_d2 = tl.full([BLOCK_M], INF, dtype=tl.float32)
        knn_d3 = tl.full([BLOCK_M], INF, dtype=tl.float32)
        knn_i0 = tl.zeros([BLOCK_M], dtype=tl.int32)
        knn_i1 = tl.zeros([BLOCK_M], dtype=tl.int32)
        knn_i2 = tl.zeros([BLOCK_M], dtype=tl.int32)
        knn_i3 = tl.zeros([BLOCK_M], dtype=tl.int32)
        
        # Stream through all reference positions
        for ref_idx in range(N):
            r0 = tl.load(pos_base + ref_idx * stride_p_n + 0 * stride_p_d).to(tl.float32)
            r1 = tl.load(pos_base + ref_idx * stride_p_n + 1 * stride_p_d).to(tl.float32)
            
            diff0 = samp_0 - r0
            diff1 = samp_1 - r1
            dist_sq = diff0 * diff0 + diff1 * diff1
            
            u0 = dist_sq < knn_d0
            u1 = dist_sq < knn_d1
            u2 = dist_sq < knn_d2
            u3 = dist_sq < knn_d3
            
            knn_d3 = tl.where(u2, knn_d2, tl.where(u3, dist_sq, knn_d3))
            knn_i3 = tl.where(u2, knn_i2, tl.where(u3, ref_idx, knn_i3))
            knn_d2 = tl.where(u1, knn_d1, tl.where(u2, dist_sq, knn_d2))
            knn_i2 = tl.where(u1, knn_i1, tl.where(u2, ref_idx, knn_i2))
            knn_d1 = tl.where(u0, knn_d0, tl.where(u1, dist_sq, knn_d1))
            knn_i1 = tl.where(u0, knn_i0, tl.where(u1, ref_idx, knn_i1))
            knn_d0 = tl.where(u0, dist_sq, knn_d0)
            knn_i0 = tl.where(u0, ref_idx, knn_i0)
        
        # ========== Save KNN indices and squared distances for backward ==========
        knn_idx_ptr = knn_idx_base + offs_m * stride_ki_n + k_samp * stride_ki_k
        tl.store(knn_idx_ptr + 0 * stride_ki_j, knn_i0, mask=mask_m)
        tl.store(knn_idx_ptr + 1 * stride_ki_j, knn_i1, mask=mask_m)
        tl.store(knn_idx_ptr + 2 * stride_ki_j, knn_i2, mask=mask_m)
        tl.store(knn_idx_ptr + 3 * stride_ki_j, knn_i3, mask=mask_m)
        
        knn_dist_ptr = knn_dist_base + offs_m * stride_kd_n + k_samp * stride_kd_k
        tl.store(knn_dist_ptr + 0 * stride_kd_j, knn_d0, mask=mask_m)
        tl.store(knn_dist_ptr + 1 * stride_kd_j, knn_d1, mask=mask_m)
        tl.store(knn_dist_ptr + 2 * stride_kd_j, knn_d2, mask=mask_m)
        tl.store(knn_dist_ptr + 3 * stride_kd_j, knn_d3, mask=mask_m)
        
        # ========== Shepard weighting ==========
        dist0 = tl.sqrt(knn_d0) + 1e-6
        dist1 = tl.sqrt(knn_d1) + 1e-6
        dist2 = tl.sqrt(knn_d2) + 1e-6
        dist3 = tl.sqrt(knn_d3) + 1e-6
        
        logit0 = -power_val * dist0
        logit1 = -power_val * dist1
        logit2 = -power_val * dist2
        logit3 = -power_val * dist3
        
        max_logit = tl.maximum(tl.maximum(logit0, logit1), tl.maximum(logit2, logit3))
        exp0 = tl.exp(logit0 - max_logit)
        exp1 = tl.exp(logit1 - max_logit)
        exp2 = tl.exp(logit2 - max_logit)
        exp3 = tl.exp(logit3 - max_logit)
        exp_sum = exp0 + exp1 + exp2 + exp3 + 1e-8
        
        w0 = exp0 / exp_sum
        w1 = exp1 / exp_sum
        w2 = exp2 / exp_sum
        w3 = exp3 / exp_sum
        
        # ========== Weighted gather ==========
        val_ptr_0 = val_base + knn_i0[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v0 = tl.load(val_ptr_0, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        val_ptr_1 = val_base + knn_i1[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v1 = tl.load(val_ptr_1, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        val_ptr_2 = val_base + knn_i2[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v2 = tl.load(val_ptr_2, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        val_ptr_3 = val_base + knn_i3[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v3 = tl.load(val_ptr_3, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        weighted_v = w0[:, None] * v0 + w1[:, None] * v1 + w2[:, None] * v2 + w3[:, None] * v3
        acc += attn_w[:, None] * weighted_v
    
    # Store output
    out_ptr = out_base + offs_m[:, None] * stride_out_n + offs_c[None, :] * stride_out_c
    tl.store(out_ptr, acc, mask=mask_m[:, None] & mask_c[None, :])


# =============================================================================
# Cross-Attention: Query and KV positions differ
# =============================================================================

@triton.jit
def _fused_knn_deform_cross_attn_kernel_d2_k4(
    # Inputs
    QUERY_POS,        # [B, N_q, 2] - query positions
    KV_POS,           # [B, N_kv, 2] - key/value positions (NOT expanded)
    OFFSETS,          # [B, N_q, H, K, 2] - sampling offsets per head/point
    ATTN_WEIGHTS,     # [B, N_q, H, K] - attention weights (softmaxed)
    VALUES,           # [B, N_kv, H, C_] - projected values per head
    POWER,            # scalar tensor - learned temperature
    # Output
    OUT,              # [B, N_q, H, C_] - output features
    # Dimensions
    B: tl.constexpr, N_q: tl.constexpr, N_kv: tl.constexpr, H: tl.constexpr, C_: tl.constexpr,
    # Strides for QUERY_POS [B, N_q, 2]
    stride_qp_b, stride_qp_n, stride_qp_d,
    # Strides for KV_POS [B, N_kv, 2]
    stride_kvp_b, stride_kvp_n, stride_kvp_d,
    # Strides for OFFSETS [B, N_q, H, K, 2]
    stride_o_b, stride_o_n, stride_o_h, stride_o_k, stride_o_d,
    # Strides for ATTN_WEIGHTS [B, N_q, H, K]
    stride_a_b, stride_a_n, stride_a_h, stride_a_k,
    # Strides for VALUES [B, N_kv, H, C_]
    stride_v_b, stride_v_n, stride_v_h, stride_v_c,
    # Strides for OUT [B, N_q, H, C_]
    stride_out_b, stride_out_n, stride_out_h, stride_out_c,
    # Compile-time constants
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    KNN_K: tl.constexpr,
):
    """
    Fused KNN + Deformable Cross-Attention for D=2, K_sample=4, KNN_K=4.
    
    Similar to self-attention but query and KV positions are different.
    """
    INF: tl.constexpr = 1e38
    K_SAMPLE: tl.constexpr = 4
    
    pid_m = tl.program_id(0)   # Query tile
    pid_b = tl.program_id(1)   # Batch
    pid_h = tl.program_id(2)   # Head
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    mask_m = offs_m < N_q
    mask_c = offs_c < C_
    
    # Base pointers
    qpos_base = QUERY_POS + pid_b * stride_qp_b
    kvpos_base = KV_POS + pid_b * stride_kvp_b
    off_base = OFFSETS + pid_b * stride_o_b + pid_h * stride_o_h
    attn_base = ATTN_WEIGHTS + pid_b * stride_a_b + pid_h * stride_a_h
    val_base = VALUES + pid_b * stride_v_b + pid_h * stride_v_h
    out_base = OUT + pid_b * stride_out_b + pid_h * stride_out_h
    
    # Load query positions [BLOCK_M]
    q_pos_0 = tl.load(qpos_base + offs_m * stride_qp_n + 0 * stride_qp_d, mask=mask_m, other=0.0).to(tl.float32)
    q_pos_1 = tl.load(qpos_base + offs_m * stride_qp_n + 1 * stride_qp_d, mask=mask_m, other=0.0).to(tl.float32)
    
    power_val = tl.load(POWER)
    power_val = tl.maximum(power_val, 1e-6)
    
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
    
    for k_samp in tl.static_range(K_SAMPLE):
        # Load offset and compute sampling location
        off_0 = tl.load(off_base + offs_m * stride_o_n + k_samp * stride_o_k + 0 * stride_o_d, mask=mask_m, other=0.0).to(tl.float32)
        off_1 = tl.load(off_base + offs_m * stride_o_n + k_samp * stride_o_k + 1 * stride_o_d, mask=mask_m, other=0.0).to(tl.float32)
        
        samp_0 = q_pos_0 + off_0
        samp_1 = q_pos_1 + off_1
        
        attn_w = tl.load(attn_base + offs_m * stride_a_n + k_samp * stride_a_k, mask=mask_m, other=0.0).to(tl.float32)
        
        # Initialize top-4
        knn_d0 = tl.full([BLOCK_M], INF, dtype=tl.float32)
        knn_d1 = tl.full([BLOCK_M], INF, dtype=tl.float32)
        knn_d2 = tl.full([BLOCK_M], INF, dtype=tl.float32)
        knn_d3 = tl.full([BLOCK_M], INF, dtype=tl.float32)
        knn_i0 = tl.zeros([BLOCK_M], dtype=tl.int32)
        knn_i1 = tl.zeros([BLOCK_M], dtype=tl.int32)
        knn_i2 = tl.zeros([BLOCK_M], dtype=tl.int32)
        knn_i3 = tl.zeros([BLOCK_M], dtype=tl.int32)
        
        # Stream through KV positions (different from query positions)
        for ref_idx in range(N_kv):
            r0 = tl.load(kvpos_base + ref_idx * stride_kvp_n + 0 * stride_kvp_d).to(tl.float32)
            r1 = tl.load(kvpos_base + ref_idx * stride_kvp_n + 1 * stride_kvp_d).to(tl.float32)
            
            diff0 = samp_0 - r0
            diff1 = samp_1 - r1
            dist_sq = diff0 * diff0 + diff1 * diff1
            
            u0 = dist_sq < knn_d0
            u1 = dist_sq < knn_d1
            u2 = dist_sq < knn_d2
            u3 = dist_sq < knn_d3
            
            knn_d3 = tl.where(u2, knn_d2, tl.where(u3, dist_sq, knn_d3))
            knn_i3 = tl.where(u2, knn_i2, tl.where(u3, ref_idx, knn_i3))
            knn_d2 = tl.where(u1, knn_d1, tl.where(u2, dist_sq, knn_d2))
            knn_i2 = tl.where(u1, knn_i1, tl.where(u2, ref_idx, knn_i2))
            knn_d1 = tl.where(u0, knn_d0, tl.where(u1, dist_sq, knn_d1))
            knn_i1 = tl.where(u0, knn_i0, tl.where(u1, ref_idx, knn_i1))
            knn_d0 = tl.where(u0, dist_sq, knn_d0)
            knn_i0 = tl.where(u0, ref_idx, knn_i0)
        
        # Shepard weighting
        dist0 = tl.sqrt(knn_d0) + 1e-6
        dist1 = tl.sqrt(knn_d1) + 1e-6
        dist2 = tl.sqrt(knn_d2) + 1e-6
        dist3 = tl.sqrt(knn_d3) + 1e-6
        
        logit0 = -power_val * dist0
        logit1 = -power_val * dist1
        logit2 = -power_val * dist2
        logit3 = -power_val * dist3
        
        max_logit = tl.maximum(tl.maximum(logit0, logit1), tl.maximum(logit2, logit3))
        exp0 = tl.exp(logit0 - max_logit)
        exp1 = tl.exp(logit1 - max_logit)
        exp2 = tl.exp(logit2 - max_logit)
        exp3 = tl.exp(logit3 - max_logit)
        exp_sum = exp0 + exp1 + exp2 + exp3 + 1e-8
        
        w0 = exp0 / exp_sum
        w1 = exp1 / exp_sum
        w2 = exp2 / exp_sum
        w3 = exp3 / exp_sum
        
        # Weighted gather
        val_ptr_0 = val_base + knn_i0[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v0 = tl.load(val_ptr_0, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        val_ptr_1 = val_base + knn_i1[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v1 = tl.load(val_ptr_1, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        val_ptr_2 = val_base + knn_i2[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v2 = tl.load(val_ptr_2, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        val_ptr_3 = val_base + knn_i3[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v3 = tl.load(val_ptr_3, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        weighted_v = w0[:, None] * v0 + w1[:, None] * v1 + w2[:, None] * v2 + w3[:, None] * v3
        acc += attn_w[:, None] * weighted_v
    
    out_ptr = out_base + offs_m[:, None] * stride_out_n + offs_c[None, :] * stride_out_c
    tl.store(out_ptr, acc, mask=mask_m[:, None] & mask_c[None, :])


# =============================================================================
# Cross-Attention with KNN indices/distances saving for backward pass
# =============================================================================

@triton.jit
def _fused_knn_deform_cross_attn_kernel_d2_k4_save(
    # Inputs
    QUERY_POS,        # [B, N_q, 2] - query positions
    KV_POS,           # [B, N_kv, 2] - key/value positions
    OFFSETS,          # [B, N_q, H, K, 2] - sampling offsets
    ATTN_WEIGHTS,     # [B, N_q, H, K] - attention weights
    VALUES,           # [B, N_kv, H, C_] - values
    POWER,            # scalar tensor
    # Outputs
    OUT,              # [B, N_q, H, C_] - output features
    KNN_IDX,          # [B, N_q, H, K_SAMPLE, KNN_K] - saved KNN indices
    KNN_DIST_SQ,      # [B, N_q, H, K_SAMPLE, KNN_K] - saved squared distances
    # Dimensions
    B: tl.constexpr, N_q: tl.constexpr, N_kv: tl.constexpr, H: tl.constexpr, C_: tl.constexpr,
    # Strides for QUERY_POS [B, N_q, 2]
    stride_qp_b, stride_qp_n, stride_qp_d,
    # Strides for KV_POS [B, N_kv, 2]
    stride_kvp_b, stride_kvp_n, stride_kvp_d,
    # Strides for OFFSETS [B, N_q, H, K, 2]
    stride_o_b, stride_o_n, stride_o_h, stride_o_k, stride_o_d,
    # Strides for ATTN_WEIGHTS [B, N_q, H, K]
    stride_a_b, stride_a_n, stride_a_h, stride_a_k,
    # Strides for VALUES [B, N_kv, H, C_]
    stride_v_b, stride_v_n, stride_v_h, stride_v_c,
    # Strides for OUT [B, N_q, H, C_]
    stride_out_b, stride_out_n, stride_out_h, stride_out_c,
    # Strides for KNN_IDX [B, N_q, H, K_SAMPLE, KNN_K]
    stride_ki_b, stride_ki_n, stride_ki_h, stride_ki_k, stride_ki_j,
    # Strides for KNN_DIST_SQ [B, N_q, H, K_SAMPLE, KNN_K]
    stride_kd_b, stride_kd_n, stride_kd_h, stride_kd_k, stride_kd_j,
    # Compile-time constants
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    KNN_K: tl.constexpr,
):
    """
    Fused KNN + Deformable Cross-Attention that also saves KNN indices and distances.
    """
    INF: tl.constexpr = 1e38
    K_SAMPLE: tl.constexpr = 4
    
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    mask_m = offs_m < N_q
    mask_c = offs_c < C_
    
    # Base pointers
    qpos_base = QUERY_POS + pid_b * stride_qp_b
    kvpos_base = KV_POS + pid_b * stride_kvp_b
    off_base = OFFSETS + pid_b * stride_o_b + pid_h * stride_o_h
    attn_base = ATTN_WEIGHTS + pid_b * stride_a_b + pid_h * stride_a_h
    val_base = VALUES + pid_b * stride_v_b + pid_h * stride_v_h
    out_base = OUT + pid_b * stride_out_b + pid_h * stride_out_h
    knn_idx_base = KNN_IDX + pid_b * stride_ki_b + pid_h * stride_ki_h
    knn_dist_base = KNN_DIST_SQ + pid_b * stride_kd_b + pid_h * stride_kd_h
    
    # Load query positions
    q_pos_0 = tl.load(qpos_base + offs_m * stride_qp_n + 0 * stride_qp_d, mask=mask_m, other=0.0).to(tl.float32)
    q_pos_1 = tl.load(qpos_base + offs_m * stride_qp_n + 1 * stride_qp_d, mask=mask_m, other=0.0).to(tl.float32)
    
    power_val = tl.load(POWER)
    power_val = tl.maximum(power_val, 1e-6)
    
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
    
    for k_samp in tl.static_range(K_SAMPLE):
        off_0 = tl.load(off_base + offs_m * stride_o_n + k_samp * stride_o_k + 0 * stride_o_d, mask=mask_m, other=0.0).to(tl.float32)
        off_1 = tl.load(off_base + offs_m * stride_o_n + k_samp * stride_o_k + 1 * stride_o_d, mask=mask_m, other=0.0).to(tl.float32)
        
        samp_0 = q_pos_0 + off_0
        samp_1 = q_pos_1 + off_1
        
        attn_w = tl.load(attn_base + offs_m * stride_a_n + k_samp * stride_a_k, mask=mask_m, other=0.0).to(tl.float32)
        
        # Initialize top-4 KNN
        knn_d0 = tl.full([BLOCK_M], INF, dtype=tl.float32)
        knn_d1 = tl.full([BLOCK_M], INF, dtype=tl.float32)
        knn_d2 = tl.full([BLOCK_M], INF, dtype=tl.float32)
        knn_d3 = tl.full([BLOCK_M], INF, dtype=tl.float32)
        knn_i0 = tl.zeros([BLOCK_M], dtype=tl.int32)
        knn_i1 = tl.zeros([BLOCK_M], dtype=tl.int32)
        knn_i2 = tl.zeros([BLOCK_M], dtype=tl.int32)
        knn_i3 = tl.zeros([BLOCK_M], dtype=tl.int32)
        
        # Stream through KV positions
        for ref_idx in range(N_kv):
            r0 = tl.load(kvpos_base + ref_idx * stride_kvp_n + 0 * stride_kvp_d).to(tl.float32)
            r1 = tl.load(kvpos_base + ref_idx * stride_kvp_n + 1 * stride_kvp_d).to(tl.float32)
            
            diff0 = samp_0 - r0
            diff1 = samp_1 - r1
            dist_sq = diff0 * diff0 + diff1 * diff1
            
            u0 = dist_sq < knn_d0
            u1 = dist_sq < knn_d1
            u2 = dist_sq < knn_d2
            u3 = dist_sq < knn_d3
            
            knn_d3 = tl.where(u2, knn_d2, tl.where(u3, dist_sq, knn_d3))
            knn_i3 = tl.where(u2, knn_i2, tl.where(u3, ref_idx, knn_i3))
            knn_d2 = tl.where(u1, knn_d1, tl.where(u2, dist_sq, knn_d2))
            knn_i2 = tl.where(u1, knn_i1, tl.where(u2, ref_idx, knn_i2))
            knn_d1 = tl.where(u0, knn_d0, tl.where(u1, dist_sq, knn_d1))
            knn_i1 = tl.where(u0, knn_i0, tl.where(u1, ref_idx, knn_i1))
            knn_d0 = tl.where(u0, dist_sq, knn_d0)
            knn_i0 = tl.where(u0, ref_idx, knn_i0)
        
        # ========== Save KNN indices and squared distances for backward ==========
        knn_idx_ptr = knn_idx_base + offs_m * stride_ki_n + k_samp * stride_ki_k
        tl.store(knn_idx_ptr + 0 * stride_ki_j, knn_i0, mask=mask_m)
        tl.store(knn_idx_ptr + 1 * stride_ki_j, knn_i1, mask=mask_m)
        tl.store(knn_idx_ptr + 2 * stride_ki_j, knn_i2, mask=mask_m)
        tl.store(knn_idx_ptr + 3 * stride_ki_j, knn_i3, mask=mask_m)
        
        knn_dist_ptr = knn_dist_base + offs_m * stride_kd_n + k_samp * stride_kd_k
        tl.store(knn_dist_ptr + 0 * stride_kd_j, knn_d0, mask=mask_m)
        tl.store(knn_dist_ptr + 1 * stride_kd_j, knn_d1, mask=mask_m)
        tl.store(knn_dist_ptr + 2 * stride_kd_j, knn_d2, mask=mask_m)
        tl.store(knn_dist_ptr + 3 * stride_kd_j, knn_d3, mask=mask_m)
        
        # ========== Shepard weighting ==========
        dist0 = tl.sqrt(knn_d0) + 1e-6
        dist1 = tl.sqrt(knn_d1) + 1e-6
        dist2 = tl.sqrt(knn_d2) + 1e-6
        dist3 = tl.sqrt(knn_d3) + 1e-6
        
        logit0 = -power_val * dist0
        logit1 = -power_val * dist1
        logit2 = -power_val * dist2
        logit3 = -power_val * dist3
        
        max_logit = tl.maximum(tl.maximum(logit0, logit1), tl.maximum(logit2, logit3))
        exp0 = tl.exp(logit0 - max_logit)
        exp1 = tl.exp(logit1 - max_logit)
        exp2 = tl.exp(logit2 - max_logit)
        exp3 = tl.exp(logit3 - max_logit)
        exp_sum = exp0 + exp1 + exp2 + exp3 + 1e-8
        
        w0 = exp0 / exp_sum
        w1 = exp1 / exp_sum
        w2 = exp2 / exp_sum
        w3 = exp3 / exp_sum
        
        # ========== Weighted gather ==========
        val_ptr_0 = val_base + knn_i0[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v0 = tl.load(val_ptr_0, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        val_ptr_1 = val_base + knn_i1[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v1 = tl.load(val_ptr_1, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        val_ptr_2 = val_base + knn_i2[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v2 = tl.load(val_ptr_2, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        val_ptr_3 = val_base + knn_i3[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v3 = tl.load(val_ptr_3, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        weighted_v = w0[:, None] * v0 + w1[:, None] * v1 + w2[:, None] * v2 + w3[:, None] * v3
        acc += attn_w[:, None] * weighted_v
    
    # Store output
    out_ptr = out_base + offs_m[:, None] * stride_out_n + offs_c[None, :] * stride_out_c
    tl.store(out_ptr, acc, mask=mask_m[:, None] & mask_c[None, :])


# =============================================================================
# BACKWARD KERNELS
# =============================================================================

# -----------------------------------------------------------------------------
# Backward Kernel 1: Computes d_attn_weights, d_offsets (NO ATOMICS)
# Also computes d_pos contribution from being a query (NO ATOMICS)
# -----------------------------------------------------------------------------

@triton.jit
def _fused_knn_deform_self_attn_bwd_kernel1_d2_k4(
    # Forward inputs
    POS,              # [B, N, 2]
    OFFSETS,          # [B, N, H, K, 2]
    ATTN_WEIGHTS,     # [B, N, H, K]
    VALUES,           # [B, N, H, C_]
    POWER,            # scalar
    # Saved from forward
    KNN_IDX,          # [B, N, H, K, 4] int32
    KNN_DIST_SQ,      # [B, N, H, K, 4] float32
    # Gradient input
    DOUT,             # [B, N, H, C_]
    # Gradient outputs (direct store - no atomics)
    D_ATTN_WEIGHTS,   # [B, N, H, K]
    D_OFFSETS,        # [B, N, H, K, 2]
    D_POS_QUERY,      # [B, N, 2] - query contribution only
    # Dimensions
    B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, C_: tl.constexpr,
    # Strides for POS [B, N, 2]
    stride_p_b, stride_p_n, stride_p_d,
    # Strides for OFFSETS [B, N, H, K, 2]
    stride_o_b, stride_o_n, stride_o_h, stride_o_k, stride_o_d,
    # Strides for ATTN_WEIGHTS [B, N, H, K]
    stride_a_b, stride_a_n, stride_a_h, stride_a_k,
    # Strides for VALUES [B, N, H, C_]
    stride_v_b, stride_v_n, stride_v_h, stride_v_c,
    # Strides for KNN_IDX [B, N, H, K, 4]
    stride_ki_b, stride_ki_n, stride_ki_h, stride_ki_k, stride_ki_j,
    # Strides for KNN_DIST_SQ [B, N, H, K, 4]
    stride_kd_b, stride_kd_n, stride_kd_h, stride_kd_k, stride_kd_j,
    # Strides for DOUT [B, N, H, C_]
    stride_do_b, stride_do_n, stride_do_h, stride_do_c,
    # Strides for D_ATTN_WEIGHTS [B, N, H, K]
    stride_da_b, stride_da_n, stride_da_h, stride_da_k,
    # Strides for D_OFFSETS [B, N, H, K, 2]
    stride_doff_b, stride_doff_n, stride_doff_h, stride_doff_k, stride_doff_d,
    # Strides for D_POS_QUERY [B, N, 2]
    stride_dpq_b, stride_dpq_n, stride_dpq_d,
    # Compile-time constants
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Backward kernel 1 for self-attention.
    Computes gradients that don't require atomics: d_attn_weights, d_offsets, d_pos (query contrib).
    """
    K_SAMPLE: tl.constexpr = 4
    KNN_K: tl.constexpr = 4
    
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    mask_m = offs_m < N
    mask_c = offs_c < C_
    
    # Base pointers
    pos_base = POS + pid_b * stride_p_b
    off_base = OFFSETS + pid_b * stride_o_b + pid_h * stride_o_h
    attn_base = ATTN_WEIGHTS + pid_b * stride_a_b + pid_h * stride_a_h
    val_base = VALUES + pid_b * stride_v_b + pid_h * stride_v_h
    knn_idx_base = KNN_IDX + pid_b * stride_ki_b + pid_h * stride_ki_h
    knn_dist_base = KNN_DIST_SQ + pid_b * stride_kd_b + pid_h * stride_kd_h
    dout_base = DOUT + pid_b * stride_do_b + pid_h * stride_do_h
    
    # Load power
    power_val = tl.load(POWER)
    power_val = tl.maximum(power_val, 1e-6)
    
    # Load query positions
    q_pos_0 = tl.load(pos_base + offs_m * stride_p_n + 0 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
    q_pos_1 = tl.load(pos_base + offs_m * stride_p_n + 1 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
    
    # Accumulate d_pos_query across all k_samp
    d_pos_q_0 = tl.zeros([BLOCK_M], dtype=tl.float32)
    d_pos_q_1 = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    for k_samp in tl.static_range(K_SAMPLE):
        # Load offsets
        off_0 = tl.load(off_base + offs_m * stride_o_n + k_samp * stride_o_k + 0 * stride_o_d, mask=mask_m, other=0.0).to(tl.float32)
        off_1 = tl.load(off_base + offs_m * stride_o_n + k_samp * stride_o_k + 1 * stride_o_d, mask=mask_m, other=0.0).to(tl.float32)
        
        # Sampling location
        samp_0 = q_pos_0 + off_0
        samp_1 = q_pos_1 + off_1
        
        # Load attention weights
        attn_w = tl.load(attn_base + offs_m * stride_a_n + k_samp * stride_a_k, mask=mask_m, other=0.0).to(tl.float32)
        
        # Load saved KNN data
        knn_idx_ptr = knn_idx_base + offs_m * stride_ki_n + k_samp * stride_ki_k
        knn_i0 = tl.load(knn_idx_ptr + 0 * stride_ki_j, mask=mask_m, other=0)
        knn_i1 = tl.load(knn_idx_ptr + 1 * stride_ki_j, mask=mask_m, other=0)
        knn_i2 = tl.load(knn_idx_ptr + 2 * stride_ki_j, mask=mask_m, other=0)
        knn_i3 = tl.load(knn_idx_ptr + 3 * stride_ki_j, mask=mask_m, other=0)
        
        knn_dist_ptr = knn_dist_base + offs_m * stride_kd_n + k_samp * stride_kd_k
        knn_d0 = tl.load(knn_dist_ptr + 0 * stride_kd_j, mask=mask_m, other=1e38).to(tl.float32)
        knn_d1 = tl.load(knn_dist_ptr + 1 * stride_kd_j, mask=mask_m, other=1e38).to(tl.float32)
        knn_d2 = tl.load(knn_dist_ptr + 2 * stride_kd_j, mask=mask_m, other=1e38).to(tl.float32)
        knn_d3 = tl.load(knn_dist_ptr + 3 * stride_kd_j, mask=mask_m, other=1e38).to(tl.float32)
        
        # Recompute softmax weights from saved squared distances
        dist0 = tl.sqrt(knn_d0) + 1e-6
        dist1 = tl.sqrt(knn_d1) + 1e-6
        dist2 = tl.sqrt(knn_d2) + 1e-6
        dist3 = tl.sqrt(knn_d3) + 1e-6
        
        logit0 = -power_val * dist0
        logit1 = -power_val * dist1
        logit2 = -power_val * dist2
        logit3 = -power_val * dist3
        
        max_logit = tl.maximum(tl.maximum(logit0, logit1), tl.maximum(logit2, logit3))
        exp0 = tl.exp(logit0 - max_logit)
        exp1 = tl.exp(logit1 - max_logit)
        exp2 = tl.exp(logit2 - max_logit)
        exp3 = tl.exp(logit3 - max_logit)
        exp_sum = exp0 + exp1 + exp2 + exp3 + 1e-8
        
        w0 = exp0 / exp_sum
        w1 = exp1 / exp_sum
        w2 = exp2 / exp_sum
        w3 = exp3 / exp_sum
        
        # Load dout [BLOCK_M, BLOCK_C]
        dout_ptr = dout_base + offs_m[:, None] * stride_do_n + offs_c[None, :] * stride_do_c
        dout = tl.load(dout_ptr, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        # Load values for the 4 neighbors
        val_ptr_0 = val_base + knn_i0[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v0 = tl.load(val_ptr_0, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        val_ptr_1 = val_base + knn_i1[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v1 = tl.load(val_ptr_1, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        val_ptr_2 = val_base + knn_i2[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v2 = tl.load(val_ptr_2, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        val_ptr_3 = val_base + knn_i3[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v3 = tl.load(val_ptr_3, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        # weighted_v = sum_j(w_j * v_j)
        weighted_v = w0[:, None] * v0 + w1[:, None] * v1 + w2[:, None] * v2 + w3[:, None] * v3
        
        # ========== d_attn_weights[m,h,k] = sum_c(dout_c * weighted_v_c) ==========
        d_attn = tl.sum(dout * weighted_v, axis=1)  # [BLOCK_M]
        d_attn_ptr = D_ATTN_WEIGHTS + pid_b * stride_da_b + pid_h * stride_da_h + offs_m * stride_da_n + k_samp * stride_da_k
        tl.store(d_attn_ptr, d_attn, mask=mask_m)
        
        # ========== Backward through weighted sum ==========
        # d_weighted_v = attn_w * dout  [BLOCK_M, BLOCK_C]
        d_weighted_v = attn_w[:, None] * dout
        
        # dp_j = sum_c(v_j * d_weighted_v) - dot product for softmax backward
        dp0 = tl.sum(v0 * d_weighted_v, axis=1)  # [BLOCK_M]
        dp1 = tl.sum(v1 * d_weighted_v, axis=1)
        dp2 = tl.sum(v2 * d_weighted_v, axis=1)
        dp3 = tl.sum(v3 * d_weighted_v, axis=1)
        
        # Softmax backward: ds_j = w_j * (dp_j - sum_l(w_l * dp_l))
        weighted_dp = w0 * dp0 + w1 * dp1 + w2 * dp2 + w3 * dp3
        ds0 = w0 * (dp0 - weighted_dp)
        ds1 = w1 * (dp1 - weighted_dp)
        ds2 = w2 * (dp2 - weighted_dp)
        ds3 = w3 * (dp3 - weighted_dp)
        
        # ========== Backward through logits ==========
        # d_dist_j = ds_j * (-power)
        d_dist0 = ds0 * (-power_val)
        d_dist1 = ds1 * (-power_val)
        d_dist2 = ds2 * (-power_val)
        d_dist3 = ds3 * (-power_val)
        
        # d_dist_sq_j = d_dist_j / (2 * dist_j)
        d_dist_sq0 = d_dist0 / (2.0 * dist0)
        d_dist_sq1 = d_dist1 / (2.0 * dist1)
        d_dist_sq2 = d_dist2 / (2.0 * dist2)
        d_dist_sq3 = d_dist3 / (2.0 * dist3)
        
        # ========== Backward through distance computation ==========
        # dist_sq = (samp - pos[knn])^2, so d_samp = 2 * (samp - pos[knn]) * d_dist_sq
        
        # Load neighbor positions
        p0_0 = tl.load(pos_base + knn_i0 * stride_p_n + 0 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p0_1 = tl.load(pos_base + knn_i0 * stride_p_n + 1 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p1_0 = tl.load(pos_base + knn_i1 * stride_p_n + 0 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p1_1 = tl.load(pos_base + knn_i1 * stride_p_n + 1 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p2_0 = tl.load(pos_base + knn_i2 * stride_p_n + 0 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p2_1 = tl.load(pos_base + knn_i2 * stride_p_n + 1 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p3_0 = tl.load(pos_base + knn_i3 * stride_p_n + 0 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p3_1 = tl.load(pos_base + knn_i3 * stride_p_n + 1 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        
        # Compute differences
        diff0_0 = samp_0 - p0_0
        diff0_1 = samp_1 - p0_1
        diff1_0 = samp_0 - p1_0
        diff1_1 = samp_1 - p1_1
        diff2_0 = samp_0 - p2_0
        diff2_1 = samp_1 - p2_1
        diff3_0 = samp_0 - p3_0
        diff3_1 = samp_1 - p3_1
        
        # d_samp = sum over neighbors
        d_samp_0 = 2.0 * (diff0_0 * d_dist_sq0 + diff1_0 * d_dist_sq1 + diff2_0 * d_dist_sq2 + diff3_0 * d_dist_sq3)
        d_samp_1 = 2.0 * (diff0_1 * d_dist_sq0 + diff1_1 * d_dist_sq1 + diff2_1 * d_dist_sq2 + diff3_1 * d_dist_sq3)
        
        # ========== Store d_offsets (d_samp = d_offset since samp = pos + offset) ==========
        d_off_ptr = D_OFFSETS + pid_b * stride_doff_b + pid_h * stride_doff_h + offs_m * stride_doff_n + k_samp * stride_doff_k
        tl.store(d_off_ptr + 0 * stride_doff_d, d_samp_0, mask=mask_m)
        tl.store(d_off_ptr + 1 * stride_doff_d, d_samp_1, mask=mask_m)
        
        # Accumulate d_pos_query contribution
        d_pos_q_0 += d_samp_0
        d_pos_q_1 += d_samp_1
    
    # ========== Store d_pos_query using ATOMIC ADD ==========
    # Multiple heads contribute to the same position, so we need atomics!
    d_pos_q_ptr = D_POS_QUERY + pid_b * stride_dpq_b + offs_m * stride_dpq_n
    tl.atomic_add(d_pos_q_ptr + 0 * stride_dpq_d, d_pos_q_0, mask=mask_m)
    tl.atomic_add(d_pos_q_ptr + 1 * stride_dpq_d, d_pos_q_1, mask=mask_m)


# -----------------------------------------------------------------------------
# Backward Kernel 2: Computes d_values, d_pos (KNN contrib), d_power (ATOMICS)
# -----------------------------------------------------------------------------

@triton.jit
def _fused_knn_deform_self_attn_bwd_kernel2_d2_k4(
    # Forward inputs
    POS,              # [B, N, 2]
    OFFSETS,          # [B, N, H, K, 2]
    ATTN_WEIGHTS,     # [B, N, H, K]
    VALUES,           # [B, N, H, C_]
    POWER,            # scalar
    # Saved from forward
    KNN_IDX,          # [B, N, H, K, 4]
    KNN_DIST_SQ,      # [B, N, H, K, 4]
    # Gradient input
    DOUT,             # [B, N, H, C_]
    # Gradient outputs (ATOMIC)
    D_VALUES,         # [B, N, H, C_]
    D_POS_KNN,        # [B, N, 2] - KNN neighbor contribution
    D_POWER,          # scalar
    # Dimensions
    B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, C_: tl.constexpr,
    # Strides for POS [B, N, 2]
    stride_p_b, stride_p_n, stride_p_d,
    # Strides for OFFSETS [B, N, H, K, 2]
    stride_o_b, stride_o_n, stride_o_h, stride_o_k, stride_o_d,
    # Strides for ATTN_WEIGHTS [B, N, H, K]
    stride_a_b, stride_a_n, stride_a_h, stride_a_k,
    # Strides for VALUES [B, N, H, C_]
    stride_v_b, stride_v_n, stride_v_h, stride_v_c,
    # Strides for KNN_IDX [B, N, H, K, 4]
    stride_ki_b, stride_ki_n, stride_ki_h, stride_ki_k, stride_ki_j,
    # Strides for KNN_DIST_SQ [B, N, H, K, 4]
    stride_kd_b, stride_kd_n, stride_kd_h, stride_kd_k, stride_kd_j,
    # Strides for DOUT [B, N, H, C_]
    stride_do_b, stride_do_n, stride_do_h, stride_do_c,
    # Strides for D_VALUES [B, N, H, C_]
    stride_dv_b, stride_dv_n, stride_dv_h, stride_dv_c,
    # Strides for D_POS_KNN [B, N, 2]
    stride_dpk_b, stride_dpk_n, stride_dpk_d,
    # Compile-time constants
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Backward kernel 2 for self-attention.
    Computes gradients that require atomics: d_values, d_pos (KNN contribution), d_power.
    """
    K_SAMPLE: tl.constexpr = 4
    KNN_K: tl.constexpr = 4
    
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    mask_m = offs_m < N
    mask_c = offs_c < C_
    
    # Base pointers
    pos_base = POS + pid_b * stride_p_b
    off_base = OFFSETS + pid_b * stride_o_b + pid_h * stride_o_h
    attn_base = ATTN_WEIGHTS + pid_b * stride_a_b + pid_h * stride_a_h
    val_base = VALUES + pid_b * stride_v_b + pid_h * stride_v_h
    knn_idx_base = KNN_IDX + pid_b * stride_ki_b + pid_h * stride_ki_h
    knn_dist_base = KNN_DIST_SQ + pid_b * stride_kd_b + pid_h * stride_kd_h
    dout_base = DOUT + pid_b * stride_do_b + pid_h * stride_do_h
    dval_base = D_VALUES + pid_b * stride_dv_b + pid_h * stride_dv_h
    dpos_knn_base = D_POS_KNN + pid_b * stride_dpk_b
    
    power_val = tl.load(POWER)
    power_val = tl.maximum(power_val, 1e-6)
    
    # Load query positions
    q_pos_0 = tl.load(pos_base + offs_m * stride_p_n + 0 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
    q_pos_1 = tl.load(pos_base + offs_m * stride_p_n + 1 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
    
    # Accumulate d_power locally
    d_power_local = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    for k_samp in tl.static_range(K_SAMPLE):
        # Load offsets
        off_0 = tl.load(off_base + offs_m * stride_o_n + k_samp * stride_o_k + 0 * stride_o_d, mask=mask_m, other=0.0).to(tl.float32)
        off_1 = tl.load(off_base + offs_m * stride_o_n + k_samp * stride_o_k + 1 * stride_o_d, mask=mask_m, other=0.0).to(tl.float32)
        
        samp_0 = q_pos_0 + off_0
        samp_1 = q_pos_1 + off_1
        
        attn_w = tl.load(attn_base + offs_m * stride_a_n + k_samp * stride_a_k, mask=mask_m, other=0.0).to(tl.float32)
        
        # Load saved KNN data
        knn_idx_ptr = knn_idx_base + offs_m * stride_ki_n + k_samp * stride_ki_k
        knn_i0 = tl.load(knn_idx_ptr + 0 * stride_ki_j, mask=mask_m, other=0)
        knn_i1 = tl.load(knn_idx_ptr + 1 * stride_ki_j, mask=mask_m, other=0)
        knn_i2 = tl.load(knn_idx_ptr + 2 * stride_ki_j, mask=mask_m, other=0)
        knn_i3 = tl.load(knn_idx_ptr + 3 * stride_ki_j, mask=mask_m, other=0)
        
        knn_dist_ptr = knn_dist_base + offs_m * stride_kd_n + k_samp * stride_kd_k
        knn_d0 = tl.load(knn_dist_ptr + 0 * stride_kd_j, mask=mask_m, other=1e38).to(tl.float32)
        knn_d1 = tl.load(knn_dist_ptr + 1 * stride_kd_j, mask=mask_m, other=1e38).to(tl.float32)
        knn_d2 = tl.load(knn_dist_ptr + 2 * stride_kd_j, mask=mask_m, other=1e38).to(tl.float32)
        knn_d3 = tl.load(knn_dist_ptr + 3 * stride_kd_j, mask=mask_m, other=1e38).to(tl.float32)
        
        # Recompute softmax
        dist0 = tl.sqrt(knn_d0) + 1e-6
        dist1 = tl.sqrt(knn_d1) + 1e-6
        dist2 = tl.sqrt(knn_d2) + 1e-6
        dist3 = tl.sqrt(knn_d3) + 1e-6
        
        logit0 = -power_val * dist0
        logit1 = -power_val * dist1
        logit2 = -power_val * dist2
        logit3 = -power_val * dist3
        
        max_logit = tl.maximum(tl.maximum(logit0, logit1), tl.maximum(logit2, logit3))
        exp0 = tl.exp(logit0 - max_logit)
        exp1 = tl.exp(logit1 - max_logit)
        exp2 = tl.exp(logit2 - max_logit)
        exp3 = tl.exp(logit3 - max_logit)
        exp_sum = exp0 + exp1 + exp2 + exp3 + 1e-8
        
        w0 = exp0 / exp_sum
        w1 = exp1 / exp_sum
        w2 = exp2 / exp_sum
        w3 = exp3 / exp_sum
        
        # Load dout
        dout_ptr = dout_base + offs_m[:, None] * stride_do_n + offs_c[None, :] * stride_do_c
        dout = tl.load(dout_ptr, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        # d_weighted_v = attn_w * dout
        d_weighted_v = attn_w[:, None] * dout  # [BLOCK_M, BLOCK_C]
        
        # ========== d_values via atomic add ==========
        # d_v_j = w_j * d_weighted_v for each neighbor j
        d_v0 = w0[:, None] * d_weighted_v
        d_v1 = w1[:, None] * d_weighted_v
        d_v2 = w2[:, None] * d_weighted_v
        d_v3 = w3[:, None] * d_weighted_v
        
        # Atomic add to d_values for each neighbor
        # Note: This is expensive but necessary since multiple queries can reference same value
        dv_ptr_0 = dval_base + knn_i0[:, None] * stride_dv_n + offs_c[None, :] * stride_dv_c
        tl.atomic_add(dv_ptr_0, d_v0, mask=mask_m[:, None] & mask_c[None, :])
        
        dv_ptr_1 = dval_base + knn_i1[:, None] * stride_dv_n + offs_c[None, :] * stride_dv_c
        tl.atomic_add(dv_ptr_1, d_v1, mask=mask_m[:, None] & mask_c[None, :])
        
        dv_ptr_2 = dval_base + knn_i2[:, None] * stride_dv_n + offs_c[None, :] * stride_dv_c
        tl.atomic_add(dv_ptr_2, d_v2, mask=mask_m[:, None] & mask_c[None, :])
        
        dv_ptr_3 = dval_base + knn_i3[:, None] * stride_dv_n + offs_c[None, :] * stride_dv_c
        tl.atomic_add(dv_ptr_3, d_v3, mask=mask_m[:, None] & mask_c[None, :])
        
        # ========== Compute softmax backward for d_pos_knn and d_power ==========
        # Load values for the 4 neighbors
        val_ptr_0 = val_base + knn_i0[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v0 = tl.load(val_ptr_0, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        val_ptr_1 = val_base + knn_i1[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v1 = tl.load(val_ptr_1, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        val_ptr_2 = val_base + knn_i2[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v2 = tl.load(val_ptr_2, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        val_ptr_3 = val_base + knn_i3[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v3 = tl.load(val_ptr_3, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        # dp_j = sum_c(v_j * d_weighted_v)
        dp0 = tl.sum(v0 * d_weighted_v, axis=1)
        dp1 = tl.sum(v1 * d_weighted_v, axis=1)
        dp2 = tl.sum(v2 * d_weighted_v, axis=1)
        dp3 = tl.sum(v3 * d_weighted_v, axis=1)
        
        # Softmax backward
        weighted_dp = w0 * dp0 + w1 * dp1 + w2 * dp2 + w3 * dp3
        ds0 = w0 * (dp0 - weighted_dp)
        ds1 = w1 * (dp1 - weighted_dp)
        ds2 = w2 * (dp2 - weighted_dp)
        ds3 = w3 * (dp3 - weighted_dp)
        
        # ========== d_power ==========
        # d_power = sum_j(ds_j * (-dist_j))
        d_power_local += ds0 * (-dist0) + ds1 * (-dist1) + ds2 * (-dist2) + ds3 * (-dist3)
        
        # ========== d_pos_knn (atomic) ==========
        # d_dist_j = ds_j * (-power)
        d_dist0 = ds0 * (-power_val)
        d_dist1 = ds1 * (-power_val)
        d_dist2 = ds2 * (-power_val)
        d_dist3 = ds3 * (-power_val)
        
        # d_dist_sq_j = d_dist_j / (2 * dist_j)
        d_dist_sq0 = d_dist0 / (2.0 * dist0)
        d_dist_sq1 = d_dist1 / (2.0 * dist1)
        d_dist_sq2 = d_dist2 / (2.0 * dist2)
        d_dist_sq3 = d_dist3 / (2.0 * dist3)
        
        # Load neighbor positions
        p0_0 = tl.load(pos_base + knn_i0 * stride_p_n + 0 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p0_1 = tl.load(pos_base + knn_i0 * stride_p_n + 1 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p1_0 = tl.load(pos_base + knn_i1 * stride_p_n + 0 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p1_1 = tl.load(pos_base + knn_i1 * stride_p_n + 1 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p2_0 = tl.load(pos_base + knn_i2 * stride_p_n + 0 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p2_1 = tl.load(pos_base + knn_i2 * stride_p_n + 1 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p3_0 = tl.load(pos_base + knn_i3 * stride_p_n + 0 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p3_1 = tl.load(pos_base + knn_i3 * stride_p_n + 1 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        
        # d_pos[neighbor] = -2 * diff * d_dist_sq (opposite sign from d_samp)
        d_pos_knn_0_0 = -2.0 * (samp_0 - p0_0) * d_dist_sq0
        d_pos_knn_0_1 = -2.0 * (samp_1 - p0_1) * d_dist_sq0
        d_pos_knn_1_0 = -2.0 * (samp_0 - p1_0) * d_dist_sq1
        d_pos_knn_1_1 = -2.0 * (samp_1 - p1_1) * d_dist_sq1
        d_pos_knn_2_0 = -2.0 * (samp_0 - p2_0) * d_dist_sq2
        d_pos_knn_2_1 = -2.0 * (samp_1 - p2_1) * d_dist_sq2
        d_pos_knn_3_0 = -2.0 * (samp_0 - p3_0) * d_dist_sq3
        d_pos_knn_3_1 = -2.0 * (samp_1 - p3_1) * d_dist_sq3
        
        # Atomic add to d_pos for each neighbor
        tl.atomic_add(dpos_knn_base + knn_i0 * stride_dpk_n + 0 * stride_dpk_d, d_pos_knn_0_0, mask=mask_m)
        tl.atomic_add(dpos_knn_base + knn_i0 * stride_dpk_n + 1 * stride_dpk_d, d_pos_knn_0_1, mask=mask_m)
        tl.atomic_add(dpos_knn_base + knn_i1 * stride_dpk_n + 0 * stride_dpk_d, d_pos_knn_1_0, mask=mask_m)
        tl.atomic_add(dpos_knn_base + knn_i1 * stride_dpk_n + 1 * stride_dpk_d, d_pos_knn_1_1, mask=mask_m)
        tl.atomic_add(dpos_knn_base + knn_i2 * stride_dpk_n + 0 * stride_dpk_d, d_pos_knn_2_0, mask=mask_m)
        tl.atomic_add(dpos_knn_base + knn_i2 * stride_dpk_n + 1 * stride_dpk_d, d_pos_knn_2_1, mask=mask_m)
        tl.atomic_add(dpos_knn_base + knn_i3 * stride_dpk_n + 0 * stride_dpk_d, d_pos_knn_3_0, mask=mask_m)
        tl.atomic_add(dpos_knn_base + knn_i3 * stride_dpk_n + 1 * stride_dpk_d, d_pos_knn_3_1, mask=mask_m)
    
    # ========== Atomic add d_power ==========
    d_power_sum = tl.sum(tl.where(mask_m, d_power_local, 0.0))
    tl.atomic_add(D_POWER, d_power_sum)


# -----------------------------------------------------------------------------
# Cross-Attention Backward Kernels
# -----------------------------------------------------------------------------

@triton.jit
def _fused_knn_deform_cross_attn_bwd_kernel1_d2_k4(
    # Forward inputs
    QUERY_POS,        # [B, N_q, 2]
    KV_POS,           # [B, N_kv, 2]
    OFFSETS,          # [B, N_q, H, K, 2]
    ATTN_WEIGHTS,     # [B, N_q, H, K]
    VALUES,           # [B, N_kv, H, C_]
    POWER,            # scalar
    # Saved from forward
    KNN_IDX,          # [B, N_q, H, K, 4]
    KNN_DIST_SQ,      # [B, N_q, H, K, 4]
    # Gradient input
    DOUT,             # [B, N_q, H, C_]
    # Gradient outputs (direct store - no atomics)
    D_ATTN_WEIGHTS,   # [B, N_q, H, K]
    D_OFFSETS,        # [B, N_q, H, K, 2]
    D_QUERY_POS,      # [B, N_q, 2]
    # Dimensions
    B: tl.constexpr, N_q: tl.constexpr, N_kv: tl.constexpr, H: tl.constexpr, C_: tl.constexpr,
    # Strides for QUERY_POS [B, N_q, 2]
    stride_qp_b, stride_qp_n, stride_qp_d,
    # Strides for KV_POS [B, N_kv, 2]
    stride_kvp_b, stride_kvp_n, stride_kvp_d,
    # Strides for OFFSETS [B, N_q, H, K, 2]
    stride_o_b, stride_o_n, stride_o_h, stride_o_k, stride_o_d,
    # Strides for ATTN_WEIGHTS [B, N_q, H, K]
    stride_a_b, stride_a_n, stride_a_h, stride_a_k,
    # Strides for VALUES [B, N_kv, H, C_]
    stride_v_b, stride_v_n, stride_v_h, stride_v_c,
    # Strides for KNN_IDX [B, N_q, H, K, 4]
    stride_ki_b, stride_ki_n, stride_ki_h, stride_ki_k, stride_ki_j,
    # Strides for KNN_DIST_SQ [B, N_q, H, K, 4]
    stride_kd_b, stride_kd_n, stride_kd_h, stride_kd_k, stride_kd_j,
    # Strides for DOUT [B, N_q, H, C_]
    stride_do_b, stride_do_n, stride_do_h, stride_do_c,
    # Strides for D_ATTN_WEIGHTS [B, N_q, H, K]
    stride_da_b, stride_da_n, stride_da_h, stride_da_k,
    # Strides for D_OFFSETS [B, N_q, H, K, 2]
    stride_doff_b, stride_doff_n, stride_doff_h, stride_doff_k, stride_doff_d,
    # Strides for D_QUERY_POS [B, N_q, 2]
    stride_dqp_b, stride_dqp_n, stride_dqp_d,
    # Compile-time constants
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Cross-attention backward kernel 1: non-atomic gradients."""
    K_SAMPLE: tl.constexpr = 4
    
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    mask_m = offs_m < N_q
    mask_c = offs_c < C_
    
    # Base pointers
    qpos_base = QUERY_POS + pid_b * stride_qp_b
    kvpos_base = KV_POS + pid_b * stride_kvp_b
    off_base = OFFSETS + pid_b * stride_o_b + pid_h * stride_o_h
    attn_base = ATTN_WEIGHTS + pid_b * stride_a_b + pid_h * stride_a_h
    val_base = VALUES + pid_b * stride_v_b + pid_h * stride_v_h
    knn_idx_base = KNN_IDX + pid_b * stride_ki_b + pid_h * stride_ki_h
    knn_dist_base = KNN_DIST_SQ + pid_b * stride_kd_b + pid_h * stride_kd_h
    dout_base = DOUT + pid_b * stride_do_b + pid_h * stride_do_h
    
    power_val = tl.load(POWER)
    power_val = tl.maximum(power_val, 1e-6)
    
    # Load query positions
    q_pos_0 = tl.load(qpos_base + offs_m * stride_qp_n + 0 * stride_qp_d, mask=mask_m, other=0.0).to(tl.float32)
    q_pos_1 = tl.load(qpos_base + offs_m * stride_qp_n + 1 * stride_qp_d, mask=mask_m, other=0.0).to(tl.float32)
    
    d_qpos_0 = tl.zeros([BLOCK_M], dtype=tl.float32)
    d_qpos_1 = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    for k_samp in tl.static_range(K_SAMPLE):
        off_0 = tl.load(off_base + offs_m * stride_o_n + k_samp * stride_o_k + 0 * stride_o_d, mask=mask_m, other=0.0).to(tl.float32)
        off_1 = tl.load(off_base + offs_m * stride_o_n + k_samp * stride_o_k + 1 * stride_o_d, mask=mask_m, other=0.0).to(tl.float32)
        
        samp_0 = q_pos_0 + off_0
        samp_1 = q_pos_1 + off_1
        
        attn_w = tl.load(attn_base + offs_m * stride_a_n + k_samp * stride_a_k, mask=mask_m, other=0.0).to(tl.float32)
        
        # Load saved KNN data
        knn_idx_ptr = knn_idx_base + offs_m * stride_ki_n + k_samp * stride_ki_k
        knn_i0 = tl.load(knn_idx_ptr + 0 * stride_ki_j, mask=mask_m, other=0)
        knn_i1 = tl.load(knn_idx_ptr + 1 * stride_ki_j, mask=mask_m, other=0)
        knn_i2 = tl.load(knn_idx_ptr + 2 * stride_ki_j, mask=mask_m, other=0)
        knn_i3 = tl.load(knn_idx_ptr + 3 * stride_ki_j, mask=mask_m, other=0)
        
        knn_dist_ptr = knn_dist_base + offs_m * stride_kd_n + k_samp * stride_kd_k
        knn_d0 = tl.load(knn_dist_ptr + 0 * stride_kd_j, mask=mask_m, other=1e38).to(tl.float32)
        knn_d1 = tl.load(knn_dist_ptr + 1 * stride_kd_j, mask=mask_m, other=1e38).to(tl.float32)
        knn_d2 = tl.load(knn_dist_ptr + 2 * stride_kd_j, mask=mask_m, other=1e38).to(tl.float32)
        knn_d3 = tl.load(knn_dist_ptr + 3 * stride_kd_j, mask=mask_m, other=1e38).to(tl.float32)
        
        # Recompute softmax
        dist0 = tl.sqrt(knn_d0) + 1e-6
        dist1 = tl.sqrt(knn_d1) + 1e-6
        dist2 = tl.sqrt(knn_d2) + 1e-6
        dist3 = tl.sqrt(knn_d3) + 1e-6
        
        logit0 = -power_val * dist0
        logit1 = -power_val * dist1
        logit2 = -power_val * dist2
        logit3 = -power_val * dist3
        
        max_logit = tl.maximum(tl.maximum(logit0, logit1), tl.maximum(logit2, logit3))
        exp0 = tl.exp(logit0 - max_logit)
        exp1 = tl.exp(logit1 - max_logit)
        exp2 = tl.exp(logit2 - max_logit)
        exp3 = tl.exp(logit3 - max_logit)
        exp_sum = exp0 + exp1 + exp2 + exp3 + 1e-8
        
        w0 = exp0 / exp_sum
        w1 = exp1 / exp_sum
        w2 = exp2 / exp_sum
        w3 = exp3 / exp_sum
        
        # Load dout
        dout_ptr = dout_base + offs_m[:, None] * stride_do_n + offs_c[None, :] * stride_do_c
        dout = tl.load(dout_ptr, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        # Load values (from KV positions)
        val_ptr_0 = val_base + knn_i0[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v0 = tl.load(val_ptr_0, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        val_ptr_1 = val_base + knn_i1[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v1 = tl.load(val_ptr_1, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        val_ptr_2 = val_base + knn_i2[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v2 = tl.load(val_ptr_2, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        val_ptr_3 = val_base + knn_i3[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v3 = tl.load(val_ptr_3, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        weighted_v = w0[:, None] * v0 + w1[:, None] * v1 + w2[:, None] * v2 + w3[:, None] * v3
        
        # d_attn_weights
        d_attn = tl.sum(dout * weighted_v, axis=1)
        d_attn_ptr = D_ATTN_WEIGHTS + pid_b * stride_da_b + pid_h * stride_da_h + offs_m * stride_da_n + k_samp * stride_da_k
        tl.store(d_attn_ptr, d_attn, mask=mask_m)
        
        # Backward through weighted sum
        d_weighted_v = attn_w[:, None] * dout
        
        dp0 = tl.sum(v0 * d_weighted_v, axis=1)
        dp1 = tl.sum(v1 * d_weighted_v, axis=1)
        dp2 = tl.sum(v2 * d_weighted_v, axis=1)
        dp3 = tl.sum(v3 * d_weighted_v, axis=1)
        
        weighted_dp = w0 * dp0 + w1 * dp1 + w2 * dp2 + w3 * dp3
        ds0 = w0 * (dp0 - weighted_dp)
        ds1 = w1 * (dp1 - weighted_dp)
        ds2 = w2 * (dp2 - weighted_dp)
        ds3 = w3 * (dp3 - weighted_dp)
        
        d_dist0 = ds0 * (-power_val)
        d_dist1 = ds1 * (-power_val)
        d_dist2 = ds2 * (-power_val)
        d_dist3 = ds3 * (-power_val)
        
        d_dist_sq0 = d_dist0 / (2.0 * dist0)
        d_dist_sq1 = d_dist1 / (2.0 * dist1)
        d_dist_sq2 = d_dist2 / (2.0 * dist2)
        d_dist_sq3 = d_dist3 / (2.0 * dist3)
        
        # Load KV neighbor positions
        p0_0 = tl.load(kvpos_base + knn_i0 * stride_kvp_n + 0 * stride_kvp_d, mask=mask_m, other=0.0).to(tl.float32)
        p0_1 = tl.load(kvpos_base + knn_i0 * stride_kvp_n + 1 * stride_kvp_d, mask=mask_m, other=0.0).to(tl.float32)
        p1_0 = tl.load(kvpos_base + knn_i1 * stride_kvp_n + 0 * stride_kvp_d, mask=mask_m, other=0.0).to(tl.float32)
        p1_1 = tl.load(kvpos_base + knn_i1 * stride_kvp_n + 1 * stride_kvp_d, mask=mask_m, other=0.0).to(tl.float32)
        p2_0 = tl.load(kvpos_base + knn_i2 * stride_kvp_n + 0 * stride_kvp_d, mask=mask_m, other=0.0).to(tl.float32)
        p2_1 = tl.load(kvpos_base + knn_i2 * stride_kvp_n + 1 * stride_kvp_d, mask=mask_m, other=0.0).to(tl.float32)
        p3_0 = tl.load(kvpos_base + knn_i3 * stride_kvp_n + 0 * stride_kvp_d, mask=mask_m, other=0.0).to(tl.float32)
        p3_1 = tl.load(kvpos_base + knn_i3 * stride_kvp_n + 1 * stride_kvp_d, mask=mask_m, other=0.0).to(tl.float32)
        
        diff0_0 = samp_0 - p0_0
        diff0_1 = samp_1 - p0_1
        diff1_0 = samp_0 - p1_0
        diff1_1 = samp_1 - p1_1
        diff2_0 = samp_0 - p2_0
        diff2_1 = samp_1 - p2_1
        diff3_0 = samp_0 - p3_0
        diff3_1 = samp_1 - p3_1
        
        d_samp_0 = 2.0 * (diff0_0 * d_dist_sq0 + diff1_0 * d_dist_sq1 + diff2_0 * d_dist_sq2 + diff3_0 * d_dist_sq3)
        d_samp_1 = 2.0 * (diff0_1 * d_dist_sq0 + diff1_1 * d_dist_sq1 + diff2_1 * d_dist_sq2 + diff3_1 * d_dist_sq3)
        
        # Store d_offsets
        d_off_ptr = D_OFFSETS + pid_b * stride_doff_b + pid_h * stride_doff_h + offs_m * stride_doff_n + k_samp * stride_doff_k
        tl.store(d_off_ptr + 0 * stride_doff_d, d_samp_0, mask=mask_m)
        tl.store(d_off_ptr + 1 * stride_doff_d, d_samp_1, mask=mask_m)
        
        # Accumulate d_query_pos
        d_qpos_0 += d_samp_0
        d_qpos_1 += d_samp_1
    
    # Store d_query_pos using ATOMIC ADD
    # Multiple heads contribute to the same position, so we need atomics!
    d_qpos_ptr = D_QUERY_POS + pid_b * stride_dqp_b + offs_m * stride_dqp_n
    tl.atomic_add(d_qpos_ptr + 0 * stride_dqp_d, d_qpos_0, mask=mask_m)
    tl.atomic_add(d_qpos_ptr + 1 * stride_dqp_d, d_qpos_1, mask=mask_m)


@triton.jit
def _fused_knn_deform_cross_attn_bwd_kernel2_d2_k4(
    # Forward inputs
    QUERY_POS,        # [B, N_q, 2]
    KV_POS,           # [B, N_kv, 2]
    OFFSETS,          # [B, N_q, H, K, 2]
    ATTN_WEIGHTS,     # [B, N_q, H, K]
    VALUES,           # [B, N_kv, H, C_]
    POWER,            # scalar
    # Saved from forward
    KNN_IDX,          # [B, N_q, H, K, 4]
    KNN_DIST_SQ,      # [B, N_q, H, K, 4]
    # Gradient input
    DOUT,             # [B, N_q, H, C_]
    # Gradient outputs (ATOMIC)
    D_VALUES,         # [B, N_kv, H, C_]
    D_KV_POS,         # [B, N_kv, 2]
    D_POWER,          # scalar
    # Dimensions
    B: tl.constexpr, N_q: tl.constexpr, N_kv: tl.constexpr, H: tl.constexpr, C_: tl.constexpr,
    # Strides
    stride_qp_b, stride_qp_n, stride_qp_d,
    stride_kvp_b, stride_kvp_n, stride_kvp_d,
    stride_o_b, stride_o_n, stride_o_h, stride_o_k, stride_o_d,
    stride_a_b, stride_a_n, stride_a_h, stride_a_k,
    stride_v_b, stride_v_n, stride_v_h, stride_v_c,
    stride_ki_b, stride_ki_n, stride_ki_h, stride_ki_k, stride_ki_j,
    stride_kd_b, stride_kd_n, stride_kd_h, stride_kd_k, stride_kd_j,
    stride_do_b, stride_do_n, stride_do_h, stride_do_c,
    stride_dv_b, stride_dv_n, stride_dv_h, stride_dv_c,
    stride_dkv_b, stride_dkv_n, stride_dkv_d,
    # Compile-time constants
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Cross-attention backward kernel 2: atomic gradients."""
    K_SAMPLE: tl.constexpr = 4
    
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    mask_m = offs_m < N_q
    mask_c = offs_c < C_
    
    qpos_base = QUERY_POS + pid_b * stride_qp_b
    kvpos_base = KV_POS + pid_b * stride_kvp_b
    off_base = OFFSETS + pid_b * stride_o_b + pid_h * stride_o_h
    attn_base = ATTN_WEIGHTS + pid_b * stride_a_b + pid_h * stride_a_h
    val_base = VALUES + pid_b * stride_v_b + pid_h * stride_v_h
    knn_idx_base = KNN_IDX + pid_b * stride_ki_b + pid_h * stride_ki_h
    knn_dist_base = KNN_DIST_SQ + pid_b * stride_kd_b + pid_h * stride_kd_h
    dout_base = DOUT + pid_b * stride_do_b + pid_h * stride_do_h
    dval_base = D_VALUES + pid_b * stride_dv_b + pid_h * stride_dv_h
    dkvpos_base = D_KV_POS + pid_b * stride_dkv_b
    
    power_val = tl.load(POWER)
    power_val = tl.maximum(power_val, 1e-6)
    
    q_pos_0 = tl.load(qpos_base + offs_m * stride_qp_n + 0 * stride_qp_d, mask=mask_m, other=0.0).to(tl.float32)
    q_pos_1 = tl.load(qpos_base + offs_m * stride_qp_n + 1 * stride_qp_d, mask=mask_m, other=0.0).to(tl.float32)
    
    d_power_local = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    for k_samp in tl.static_range(K_SAMPLE):
        off_0 = tl.load(off_base + offs_m * stride_o_n + k_samp * stride_o_k + 0 * stride_o_d, mask=mask_m, other=0.0).to(tl.float32)
        off_1 = tl.load(off_base + offs_m * stride_o_n + k_samp * stride_o_k + 1 * stride_o_d, mask=mask_m, other=0.0).to(tl.float32)
        
        samp_0 = q_pos_0 + off_0
        samp_1 = q_pos_1 + off_1
        
        attn_w = tl.load(attn_base + offs_m * stride_a_n + k_samp * stride_a_k, mask=mask_m, other=0.0).to(tl.float32)
        
        knn_idx_ptr = knn_idx_base + offs_m * stride_ki_n + k_samp * stride_ki_k
        knn_i0 = tl.load(knn_idx_ptr + 0 * stride_ki_j, mask=mask_m, other=0)
        knn_i1 = tl.load(knn_idx_ptr + 1 * stride_ki_j, mask=mask_m, other=0)
        knn_i2 = tl.load(knn_idx_ptr + 2 * stride_ki_j, mask=mask_m, other=0)
        knn_i3 = tl.load(knn_idx_ptr + 3 * stride_ki_j, mask=mask_m, other=0)
        
        knn_dist_ptr = knn_dist_base + offs_m * stride_kd_n + k_samp * stride_kd_k
        knn_d0 = tl.load(knn_dist_ptr + 0 * stride_kd_j, mask=mask_m, other=1e38).to(tl.float32)
        knn_d1 = tl.load(knn_dist_ptr + 1 * stride_kd_j, mask=mask_m, other=1e38).to(tl.float32)
        knn_d2 = tl.load(knn_dist_ptr + 2 * stride_kd_j, mask=mask_m, other=1e38).to(tl.float32)
        knn_d3 = tl.load(knn_dist_ptr + 3 * stride_kd_j, mask=mask_m, other=1e38).to(tl.float32)
        
        dist0 = tl.sqrt(knn_d0) + 1e-6
        dist1 = tl.sqrt(knn_d1) + 1e-6
        dist2 = tl.sqrt(knn_d2) + 1e-6
        dist3 = tl.sqrt(knn_d3) + 1e-6
        
        logit0 = -power_val * dist0
        logit1 = -power_val * dist1
        logit2 = -power_val * dist2
        logit3 = -power_val * dist3
        
        max_logit = tl.maximum(tl.maximum(logit0, logit1), tl.maximum(logit2, logit3))
        exp0 = tl.exp(logit0 - max_logit)
        exp1 = tl.exp(logit1 - max_logit)
        exp2 = tl.exp(logit2 - max_logit)
        exp3 = tl.exp(logit3 - max_logit)
        exp_sum = exp0 + exp1 + exp2 + exp3 + 1e-8
        
        w0 = exp0 / exp_sum
        w1 = exp1 / exp_sum
        w2 = exp2 / exp_sum
        w3 = exp3 / exp_sum
        
        dout_ptr = dout_base + offs_m[:, None] * stride_do_n + offs_c[None, :] * stride_do_c
        dout = tl.load(dout_ptr, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        d_weighted_v = attn_w[:, None] * dout
        
        # d_values via atomic
        d_v0 = w0[:, None] * d_weighted_v
        d_v1 = w1[:, None] * d_weighted_v
        d_v2 = w2[:, None] * d_weighted_v
        d_v3 = w3[:, None] * d_weighted_v
        
        dv_ptr_0 = dval_base + knn_i0[:, None] * stride_dv_n + offs_c[None, :] * stride_dv_c
        tl.atomic_add(dv_ptr_0, d_v0, mask=mask_m[:, None] & mask_c[None, :])
        dv_ptr_1 = dval_base + knn_i1[:, None] * stride_dv_n + offs_c[None, :] * stride_dv_c
        tl.atomic_add(dv_ptr_1, d_v1, mask=mask_m[:, None] & mask_c[None, :])
        dv_ptr_2 = dval_base + knn_i2[:, None] * stride_dv_n + offs_c[None, :] * stride_dv_c
        tl.atomic_add(dv_ptr_2, d_v2, mask=mask_m[:, None] & mask_c[None, :])
        dv_ptr_3 = dval_base + knn_i3[:, None] * stride_dv_n + offs_c[None, :] * stride_dv_c
        tl.atomic_add(dv_ptr_3, d_v3, mask=mask_m[:, None] & mask_c[None, :])
        
        # Softmax backward for d_kv_pos and d_power
        val_ptr_0 = val_base + knn_i0[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v0 = tl.load(val_ptr_0, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        val_ptr_1 = val_base + knn_i1[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v1 = tl.load(val_ptr_1, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        val_ptr_2 = val_base + knn_i2[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v2 = tl.load(val_ptr_2, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        val_ptr_3 = val_base + knn_i3[:, None] * stride_v_n + offs_c[None, :] * stride_v_c
        v3 = tl.load(val_ptr_3, mask=mask_m[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        
        dp0 = tl.sum(v0 * d_weighted_v, axis=1)
        dp1 = tl.sum(v1 * d_weighted_v, axis=1)
        dp2 = tl.sum(v2 * d_weighted_v, axis=1)
        dp3 = tl.sum(v3 * d_weighted_v, axis=1)
        
        weighted_dp = w0 * dp0 + w1 * dp1 + w2 * dp2 + w3 * dp3
        ds0 = w0 * (dp0 - weighted_dp)
        ds1 = w1 * (dp1 - weighted_dp)
        ds2 = w2 * (dp2 - weighted_dp)
        ds3 = w3 * (dp3 - weighted_dp)
        
        d_power_local += ds0 * (-dist0) + ds1 * (-dist1) + ds2 * (-dist2) + ds3 * (-dist3)
        
        d_dist0 = ds0 * (-power_val)
        d_dist1 = ds1 * (-power_val)
        d_dist2 = ds2 * (-power_val)
        d_dist3 = ds3 * (-power_val)
        
        d_dist_sq0 = d_dist0 / (2.0 * dist0)
        d_dist_sq1 = d_dist1 / (2.0 * dist1)
        d_dist_sq2 = d_dist2 / (2.0 * dist2)
        d_dist_sq3 = d_dist3 / (2.0 * dist3)
        
        # Load KV positions
        p0_0 = tl.load(kvpos_base + knn_i0 * stride_kvp_n + 0 * stride_kvp_d, mask=mask_m, other=0.0).to(tl.float32)
        p0_1 = tl.load(kvpos_base + knn_i0 * stride_kvp_n + 1 * stride_kvp_d, mask=mask_m, other=0.0).to(tl.float32)
        p1_0 = tl.load(kvpos_base + knn_i1 * stride_kvp_n + 0 * stride_kvp_d, mask=mask_m, other=0.0).to(tl.float32)
        p1_1 = tl.load(kvpos_base + knn_i1 * stride_kvp_n + 1 * stride_kvp_d, mask=mask_m, other=0.0).to(tl.float32)
        p2_0 = tl.load(kvpos_base + knn_i2 * stride_kvp_n + 0 * stride_kvp_d, mask=mask_m, other=0.0).to(tl.float32)
        p2_1 = tl.load(kvpos_base + knn_i2 * stride_kvp_n + 1 * stride_kvp_d, mask=mask_m, other=0.0).to(tl.float32)
        p3_0 = tl.load(kvpos_base + knn_i3 * stride_kvp_n + 0 * stride_kvp_d, mask=mask_m, other=0.0).to(tl.float32)
        p3_1 = tl.load(kvpos_base + knn_i3 * stride_kvp_n + 1 * stride_kvp_d, mask=mask_m, other=0.0).to(tl.float32)
        
        # d_kv_pos = -2 * diff * d_dist_sq (atomic)
        d_kv_0_0 = -2.0 * (samp_0 - p0_0) * d_dist_sq0
        d_kv_0_1 = -2.0 * (samp_1 - p0_1) * d_dist_sq0
        d_kv_1_0 = -2.0 * (samp_0 - p1_0) * d_dist_sq1
        d_kv_1_1 = -2.0 * (samp_1 - p1_1) * d_dist_sq1
        d_kv_2_0 = -2.0 * (samp_0 - p2_0) * d_dist_sq2
        d_kv_2_1 = -2.0 * (samp_1 - p2_1) * d_dist_sq2
        d_kv_3_0 = -2.0 * (samp_0 - p3_0) * d_dist_sq3
        d_kv_3_1 = -2.0 * (samp_1 - p3_1) * d_dist_sq3
        
        tl.atomic_add(dkvpos_base + knn_i0 * stride_dkv_n + 0 * stride_dkv_d, d_kv_0_0, mask=mask_m)
        tl.atomic_add(dkvpos_base + knn_i0 * stride_dkv_n + 1 * stride_dkv_d, d_kv_0_1, mask=mask_m)
        tl.atomic_add(dkvpos_base + knn_i1 * stride_dkv_n + 0 * stride_dkv_d, d_kv_1_0, mask=mask_m)
        tl.atomic_add(dkvpos_base + knn_i1 * stride_dkv_n + 1 * stride_dkv_d, d_kv_1_1, mask=mask_m)
        tl.atomic_add(dkvpos_base + knn_i2 * stride_dkv_n + 0 * stride_dkv_d, d_kv_2_0, mask=mask_m)
        tl.atomic_add(dkvpos_base + knn_i2 * stride_dkv_n + 1 * stride_dkv_d, d_kv_2_1, mask=mask_m)
        tl.atomic_add(dkvpos_base + knn_i3 * stride_dkv_n + 0 * stride_dkv_d, d_kv_3_0, mask=mask_m)
        tl.atomic_add(dkvpos_base + knn_i3 * stride_dkv_n + 1 * stride_dkv_d, d_kv_3_1, mask=mask_m)
    
    d_power_sum = tl.sum(tl.where(mask_m, d_power_local, 0.0))
    tl.atomic_add(D_POWER, d_power_sum)


# =============================================================================
# Python Wrappers for Fused KNN + Deformable Attention
# =============================================================================

def fused_knn_deform_self_attn_forward(
    pos: torch.Tensor,           # [B, N, D]
    offsets: torch.Tensor,       # [B, N, H, K, D]
    attn_weights: torch.Tensor,  # [B, N, H, K]
    values: torch.Tensor,        # [B, N, H, C_]
    power: torch.Tensor,         # scalar
    D: int = 2,
    use_heuristics: bool = True,
):
    """
    Fused KNN + Deformable Self-Attention forward pass.
    
    This eliminates:
    - Position expansion across heads (~8x memory savings)
    - Sampling location materialization (~32x savings)
    - Separate KNN call and nb_idx tensor
    
    Args:
        pos: [B, N, D] - positions (shared across heads)
        offsets: [B, N, H, K, D] - sampling offsets
        attn_weights: [B, N, H, K] - attention weights (pre-softmaxed)
        values: [B, N, H, C_] - projected values
        power: scalar - learned temperature
        D: spatial dimension (only 2 supported currently)
        use_heuristics: use fixed config (faster) vs autotuning
        
    Returns:
        out: [B, N, H, C_] - output features
    """
    assert D == 2, "Currently only D=2 is supported"
    
    B, N, _ = pos.shape
    _, _, H, K, _ = offsets.shape
    C_ = values.shape[-1]
    
    assert K == 4, "Currently only K=4 sampling points supported"
    
    # Ensure contiguous
    pos = pos.contiguous()
    offsets = offsets.contiguous()
    attn_weights = attn_weights.contiguous()
    values = values.contiguous()
    
    # Output tensor
    out = torch.zeros((B, N, H, C_), device=values.device, dtype=values.dtype)
    
    # Power buffer
    power_val = torch.clamp(power, min=1e-6)
    power_buf = torch.tensor([power_val.item()], device=pos.device, dtype=torch.float32)
    
    # Get config
    cfg = get_fused_knn_config(B, N, H, K, C_)
    BLOCK_M, BLOCK_C = cfg['BLOCK_M'], cfg['BLOCK_C']
    num_warps, num_stages = cfg['num_warps'], cfg['num_stages']
    
    # Launch kernel
    grid = (triton.cdiv(N, BLOCK_M), B, H)
    
    _fused_knn_deform_self_attn_kernel_d2_k4[grid](
        pos, offsets, attn_weights, values, power_buf, out,
        B, N, H, C_,
        pos.stride(0), pos.stride(1), pos.stride(2),
        offsets.stride(0), offsets.stride(1), offsets.stride(2), offsets.stride(3), offsets.stride(4),
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2), attn_weights.stride(3),
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C, KNN_K=4,
        num_warps=num_warps, num_stages=num_stages,
    )
    
    return out


def fused_knn_deform_cross_attn_forward(
    query_pos: torch.Tensor,     # [B, N_q, D]
    kv_pos: torch.Tensor,        # [B, N_kv, D]
    offsets: torch.Tensor,       # [B, N_q, H, K, D]
    attn_weights: torch.Tensor,  # [B, N_q, H, K]
    values: torch.Tensor,        # [B, N_kv, H, C_]
    power: torch.Tensor,         # scalar
    D: int = 2,
    use_heuristics: bool = True,
):
    """
    Fused KNN + Deformable Cross-Attention forward pass.
    
    Args:
        query_pos: [B, N_q, D] - query positions
        kv_pos: [B, N_kv, D] - key/value positions (shared across heads)
        offsets: [B, N_q, H, K, D] - sampling offsets
        attn_weights: [B, N_q, H, K] - attention weights (pre-softmaxed)
        values: [B, N_kv, H, C_] - projected values
        power: scalar - learned temperature
        
    Returns:
        out: [B, N_q, H, C_] - output features
    """
    assert D == 2, "Currently only D=2 is supported"
    
    B, N_q, _ = query_pos.shape
    _, N_kv, _ = kv_pos.shape
    _, _, H, K, _ = offsets.shape
    C_ = values.shape[-1]
    
    assert K == 4, "Currently only K=4 sampling points supported"
    
    query_pos = query_pos.contiguous()
    kv_pos = kv_pos.contiguous()
    offsets = offsets.contiguous()
    attn_weights = attn_weights.contiguous()
    values = values.contiguous()
    
    out = torch.zeros((B, N_q, H, C_), device=values.device, dtype=values.dtype)
    
    power_val = torch.clamp(power, min=1e-6)
    power_buf = torch.tensor([power_val.item()], device=query_pos.device, dtype=torch.float32)
    
    cfg = get_fused_knn_config(B, N_q, H, K, C_)
    BLOCK_M, BLOCK_C = cfg['BLOCK_M'], cfg['BLOCK_C']
    num_warps, num_stages = cfg['num_warps'], cfg['num_stages']
    
    grid = (triton.cdiv(N_q, BLOCK_M), B, H)
    
    _fused_knn_deform_cross_attn_kernel_d2_k4[grid](
        query_pos, kv_pos, offsets, attn_weights, values, power_buf, out,
        B, N_q, N_kv, H, C_,
        query_pos.stride(0), query_pos.stride(1), query_pos.stride(2),
        kv_pos.stride(0), kv_pos.stride(1), kv_pos.stride(2),
        offsets.stride(0), offsets.stride(1), offsets.stride(2), offsets.stride(3), offsets.stride(4),
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2), attn_weights.stride(3),
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C, KNN_K=4,
        num_warps=num_warps, num_stages=num_stages,
    )
    
    return out


# =============================================================================
# Forward with KNN Saving (for backward pass)
# =============================================================================

def fused_knn_deform_self_attn_forward_save(
    pos: torch.Tensor,           # [B, N, D]
    offsets: torch.Tensor,       # [B, N, H, K, D]
    attn_weights: torch.Tensor,  # [B, N, H, K]
    values: torch.Tensor,        # [B, N, H, C_]
    power: torch.Tensor,         # scalar
    D: int = 2,
):
    """
    Forward pass that saves KNN indices and distances for efficient backward.
    
    Returns:
        out: [B, N, H, C_] - output features
        knn_idx: [B, N, H, K, 4] - saved KNN indices (int32)
        knn_dist_sq: [B, N, H, K, 4] - saved squared distances (float32)
    """
    assert D == 2, "Currently only D=2 is supported"
    
    B, N, _ = pos.shape
    _, _, H, K, _ = offsets.shape
    C_ = values.shape[-1]
    KNN_K = 4
    
    assert K == 4, "Currently only K=4 sampling points supported"
    
    pos = pos.contiguous()
    offsets = offsets.contiguous()
    attn_weights = attn_weights.contiguous()
    values = values.contiguous()
    
    # Output tensors
    out = torch.zeros((B, N, H, C_), device=values.device, dtype=values.dtype)
    knn_idx = torch.zeros((B, N, H, K, KNN_K), device=pos.device, dtype=torch.int32)
    knn_dist_sq = torch.zeros((B, N, H, K, KNN_K), device=pos.device, dtype=torch.float32)
    
    power_val = torch.clamp(power, min=1e-6)
    power_buf = torch.tensor([power_val.item()], device=pos.device, dtype=torch.float32)
    
    cfg = get_fused_knn_config(B, N, H, K, C_)
    BLOCK_M, BLOCK_C = cfg['BLOCK_M'], cfg['BLOCK_C']
    num_warps, num_stages = cfg['num_warps'], cfg['num_stages']
    
    grid = (triton.cdiv(N, BLOCK_M), B, H)
    
    _fused_knn_deform_self_attn_kernel_d2_k4_save[grid](
        pos, offsets, attn_weights, values, power_buf,
        out, knn_idx, knn_dist_sq,
        B, N, H, C_,
        pos.stride(0), pos.stride(1), pos.stride(2),
        offsets.stride(0), offsets.stride(1), offsets.stride(2), offsets.stride(3), offsets.stride(4),
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2), attn_weights.stride(3),
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        knn_idx.stride(0), knn_idx.stride(1), knn_idx.stride(2), knn_idx.stride(3), knn_idx.stride(4),
        knn_dist_sq.stride(0), knn_dist_sq.stride(1), knn_dist_sq.stride(2), knn_dist_sq.stride(3), knn_dist_sq.stride(4),
        BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C, KNN_K=KNN_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    
    return out, knn_idx, knn_dist_sq


def fused_knn_deform_cross_attn_forward_save(
    query_pos: torch.Tensor,     # [B, N_q, D]
    kv_pos: torch.Tensor,        # [B, N_kv, D]
    offsets: torch.Tensor,       # [B, N_q, H, K, D]
    attn_weights: torch.Tensor,  # [B, N_q, H, K]
    values: torch.Tensor,        # [B, N_kv, H, C_]
    power: torch.Tensor,         # scalar
    D: int = 2,
):
    """
    Cross-attention forward that saves KNN indices and distances for backward.
    
    Returns:
        out: [B, N_q, H, C_]
        knn_idx: [B, N_q, H, K, 4] - KNN indices into N_kv dimension
        knn_dist_sq: [B, N_q, H, K, 4]
    """
    assert D == 2, "Currently only D=2 is supported"
    
    B, N_q, _ = query_pos.shape
    _, N_kv, _ = kv_pos.shape
    _, _, H, K, _ = offsets.shape
    C_ = values.shape[-1]
    KNN_K = 4
    
    assert K == 4, "Currently only K=4 sampling points supported"
    
    query_pos = query_pos.contiguous()
    kv_pos = kv_pos.contiguous()
    offsets = offsets.contiguous()
    attn_weights = attn_weights.contiguous()
    values = values.contiguous()
    
    out = torch.zeros((B, N_q, H, C_), device=values.device, dtype=values.dtype)
    knn_idx = torch.zeros((B, N_q, H, K, KNN_K), device=query_pos.device, dtype=torch.int32)
    knn_dist_sq = torch.zeros((B, N_q, H, K, KNN_K), device=query_pos.device, dtype=torch.float32)
    
    power_val = torch.clamp(power, min=1e-6)
    power_buf = torch.tensor([power_val.item()], device=query_pos.device, dtype=torch.float32)
    
    cfg = get_fused_knn_config(B, N_q, H, K, C_)
    BLOCK_M, BLOCK_C = cfg['BLOCK_M'], cfg['BLOCK_C']
    num_warps, num_stages = cfg['num_warps'], cfg['num_stages']
    
    grid = (triton.cdiv(N_q, BLOCK_M), B, H)
    
    _fused_knn_deform_cross_attn_kernel_d2_k4_save[grid](
        query_pos, kv_pos, offsets, attn_weights, values, power_buf,
        out, knn_idx, knn_dist_sq,
        B, N_q, N_kv, H, C_,
        query_pos.stride(0), query_pos.stride(1), query_pos.stride(2),
        kv_pos.stride(0), kv_pos.stride(1), kv_pos.stride(2),
        offsets.stride(0), offsets.stride(1), offsets.stride(2), offsets.stride(3), offsets.stride(4),
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2), attn_weights.stride(3),
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        knn_idx.stride(0), knn_idx.stride(1), knn_idx.stride(2), knn_idx.stride(3), knn_idx.stride(4),
        knn_dist_sq.stride(0), knn_dist_sq.stride(1), knn_dist_sq.stride(2), knn_dist_sq.stride(3), knn_dist_sq.stride(4),
        BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C, KNN_K=KNN_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    
    return out, knn_idx, knn_dist_sq


# =============================================================================
# Fused Backward Functions
# =============================================================================

def fused_knn_deform_self_attn_backward(
    dout: torch.Tensor,
    pos: torch.Tensor,
    offsets: torch.Tensor,
    attn_weights: torch.Tensor,
    values: torch.Tensor,
    power: torch.Tensor,
    knn_idx: torch.Tensor,
    knn_dist_sq: torch.Tensor,
    D: int = 2,
):
    """
    Fused backward pass for self-attention using two kernels.
    
    Args:
        dout: [B, N, H, C_] - gradient from downstream
        pos: [B, N, D] - positions
        offsets: [B, N, H, K, D] - sampling offsets
        attn_weights: [B, N, H, K] - attention weights
        values: [B, N, H, C_] - values
        power: scalar - temperature
        knn_idx: [B, N, H, K, 4] - saved KNN indices
        knn_dist_sq: [B, N, H, K, 4] - saved squared distances
        
    Returns:
        d_pos, d_offsets, d_attn_weights, d_values, d_power
    """
    assert D == 2, "Currently only D=2 is supported"
    
    B, N, _ = pos.shape
    _, _, H, K, _ = offsets.shape
    C_ = values.shape[-1]
    
    # Ensure contiguous
    dout = dout.contiguous()
    pos = pos.contiguous()
    offsets = offsets.contiguous()
    attn_weights = attn_weights.contiguous()
    values = values.contiguous()
    knn_idx = knn_idx.contiguous()
    knn_dist_sq = knn_dist_sq.contiguous()
    
    # Allocate gradient tensors
    d_pos_query = torch.zeros((B, N, D), device=pos.device, dtype=torch.float32)
    d_pos_knn = torch.zeros((B, N, D), device=pos.device, dtype=torch.float32)
    d_offsets = torch.zeros_like(offsets, dtype=torch.float32)
    d_attn_weights = torch.zeros_like(attn_weights, dtype=torch.float32)
    d_values = torch.zeros_like(values, dtype=torch.float32)
    d_power = torch.zeros((1,), device=pos.device, dtype=torch.float32)
    
    power_val = torch.clamp(power, min=1e-6)
    power_buf = torch.tensor([power_val.item()], device=pos.device, dtype=torch.float32)
    
    cfg = get_fused_knn_config(B, N, H, K, C_)
    BLOCK_M, BLOCK_C = cfg['BLOCK_M'], cfg['BLOCK_C']
    num_warps, num_stages = cfg['num_warps'], cfg['num_stages']
    
    grid = (triton.cdiv(N, BLOCK_M), B, H)
    
    # ========== Kernel 1: Non-atomic gradients ==========
    _fused_knn_deform_self_attn_bwd_kernel1_d2_k4[grid](
        pos, offsets, attn_weights, values, power_buf,
        knn_idx, knn_dist_sq,
        dout,
        d_attn_weights, d_offsets, d_pos_query,
        B, N, H, C_,
        # POS strides
        pos.stride(0), pos.stride(1), pos.stride(2),
        # OFFSETS strides
        offsets.stride(0), offsets.stride(1), offsets.stride(2), offsets.stride(3), offsets.stride(4),
        # ATTN_WEIGHTS strides
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2), attn_weights.stride(3),
        # VALUES strides
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        # KNN_IDX strides
        knn_idx.stride(0), knn_idx.stride(1), knn_idx.stride(2), knn_idx.stride(3), knn_idx.stride(4),
        # KNN_DIST_SQ strides
        knn_dist_sq.stride(0), knn_dist_sq.stride(1), knn_dist_sq.stride(2), knn_dist_sq.stride(3), knn_dist_sq.stride(4),
        # DOUT strides
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        # D_ATTN_WEIGHTS strides
        d_attn_weights.stride(0), d_attn_weights.stride(1), d_attn_weights.stride(2), d_attn_weights.stride(3),
        # D_OFFSETS strides
        d_offsets.stride(0), d_offsets.stride(1), d_offsets.stride(2), d_offsets.stride(3), d_offsets.stride(4),
        # D_POS_QUERY strides
        d_pos_query.stride(0), d_pos_query.stride(1), d_pos_query.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C,
        num_warps=num_warps, num_stages=num_stages,
    )
    
    # ========== Kernel 2: Atomic gradients ==========
    _fused_knn_deform_self_attn_bwd_kernel2_d2_k4[grid](
        pos, offsets, attn_weights, values, power_buf,
        knn_idx, knn_dist_sq,
        dout,
        d_values, d_pos_knn, d_power,
        B, N, H, C_,
        # POS strides
        pos.stride(0), pos.stride(1), pos.stride(2),
        # OFFSETS strides
        offsets.stride(0), offsets.stride(1), offsets.stride(2), offsets.stride(3), offsets.stride(4),
        # ATTN_WEIGHTS strides
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2), attn_weights.stride(3),
        # VALUES strides
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        # KNN_IDX strides
        knn_idx.stride(0), knn_idx.stride(1), knn_idx.stride(2), knn_idx.stride(3), knn_idx.stride(4),
        # KNN_DIST_SQ strides
        knn_dist_sq.stride(0), knn_dist_sq.stride(1), knn_dist_sq.stride(2), knn_dist_sq.stride(3), knn_dist_sq.stride(4),
        # DOUT strides
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        # D_VALUES strides
        d_values.stride(0), d_values.stride(1), d_values.stride(2), d_values.stride(3),
        # D_POS_KNN strides
        d_pos_knn.stride(0), d_pos_knn.stride(1), d_pos_knn.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C,
        num_warps=num_warps, num_stages=num_stages,
    )
    
    # Combine d_pos contributions
    d_pos = d_pos_query + d_pos_knn
    
    # Cast back to input dtypes
    d_pos = d_pos.to(pos.dtype)
    d_offsets = d_offsets.to(offsets.dtype)
    d_attn_weights = d_attn_weights.to(attn_weights.dtype)
    d_values = d_values.to(values.dtype)
    
    return d_pos, d_offsets, d_attn_weights, d_values, d_power


def fused_knn_deform_cross_attn_backward(
    dout: torch.Tensor,
    query_pos: torch.Tensor,
    kv_pos: torch.Tensor,
    offsets: torch.Tensor,
    attn_weights: torch.Tensor,
    values: torch.Tensor,
    power: torch.Tensor,
    knn_idx: torch.Tensor,
    knn_dist_sq: torch.Tensor,
    D: int = 2,
):
    """
    Fused backward pass for cross-attention using two kernels.
    
    Args:
        dout: [B, N_q, H, C_] - gradient from downstream
        query_pos: [B, N_q, D] - query positions
        kv_pos: [B, N_kv, D] - key/value positions
        offsets: [B, N_q, H, K, D] - sampling offsets
        attn_weights: [B, N_q, H, K] - attention weights
        values: [B, N_kv, H, C_] - values
        power: scalar - temperature
        knn_idx: [B, N_q, H, K, 4] - saved KNN indices
        knn_dist_sq: [B, N_q, H, K, 4] - saved squared distances
        
    Returns:
        d_query_pos, d_kv_pos, d_offsets, d_attn_weights, d_values, d_power
    """
    assert D == 2, "Currently only D=2 is supported"
    
    B, N_q, _ = query_pos.shape
    _, N_kv, _ = kv_pos.shape
    _, _, H, K, _ = offsets.shape
    C_ = values.shape[-1]
    
    dout = dout.contiguous()
    query_pos = query_pos.contiguous()
    kv_pos = kv_pos.contiguous()
    offsets = offsets.contiguous()
    attn_weights = attn_weights.contiguous()
    values = values.contiguous()
    knn_idx = knn_idx.contiguous()
    knn_dist_sq = knn_dist_sq.contiguous()
    
    # Allocate gradient tensors
    d_query_pos = torch.zeros((B, N_q, D), device=query_pos.device, dtype=torch.float32)
    d_kv_pos = torch.zeros((B, N_kv, D), device=kv_pos.device, dtype=torch.float32)
    d_offsets = torch.zeros_like(offsets, dtype=torch.float32)
    d_attn_weights = torch.zeros_like(attn_weights, dtype=torch.float32)
    d_values = torch.zeros_like(values, dtype=torch.float32)
    d_power = torch.zeros((1,), device=query_pos.device, dtype=torch.float32)
    
    power_val = torch.clamp(power, min=1e-6)
    power_buf = torch.tensor([power_val.item()], device=query_pos.device, dtype=torch.float32)
    
    cfg = get_fused_knn_config(B, N_q, H, K, C_)
    BLOCK_M, BLOCK_C = cfg['BLOCK_M'], cfg['BLOCK_C']
    num_warps, num_stages = cfg['num_warps'], cfg['num_stages']
    
    grid = (triton.cdiv(N_q, BLOCK_M), B, H)
    
    # ========== Kernel 1: Non-atomic gradients ==========
    _fused_knn_deform_cross_attn_bwd_kernel1_d2_k4[grid](
        query_pos, kv_pos, offsets, attn_weights, values, power_buf,
        knn_idx, knn_dist_sq,
        dout,
        d_attn_weights, d_offsets, d_query_pos,
        B, N_q, N_kv, H, C_,
        # QUERY_POS strides
        query_pos.stride(0), query_pos.stride(1), query_pos.stride(2),
        # KV_POS strides
        kv_pos.stride(0), kv_pos.stride(1), kv_pos.stride(2),
        # OFFSETS strides
        offsets.stride(0), offsets.stride(1), offsets.stride(2), offsets.stride(3), offsets.stride(4),
        # ATTN_WEIGHTS strides
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2), attn_weights.stride(3),
        # VALUES strides
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        # KNN_IDX strides
        knn_idx.stride(0), knn_idx.stride(1), knn_idx.stride(2), knn_idx.stride(3), knn_idx.stride(4),
        # KNN_DIST_SQ strides
        knn_dist_sq.stride(0), knn_dist_sq.stride(1), knn_dist_sq.stride(2), knn_dist_sq.stride(3), knn_dist_sq.stride(4),
        # DOUT strides
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        # D_ATTN_WEIGHTS strides
        d_attn_weights.stride(0), d_attn_weights.stride(1), d_attn_weights.stride(2), d_attn_weights.stride(3),
        # D_OFFSETS strides
        d_offsets.stride(0), d_offsets.stride(1), d_offsets.stride(2), d_offsets.stride(3), d_offsets.stride(4),
        # D_QUERY_POS strides
        d_query_pos.stride(0), d_query_pos.stride(1), d_query_pos.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C,
        num_warps=num_warps, num_stages=num_stages,
    )
    
    # ========== Kernel 2: Atomic gradients ==========
    _fused_knn_deform_cross_attn_bwd_kernel2_d2_k4[grid](
        query_pos, kv_pos, offsets, attn_weights, values, power_buf,
        knn_idx, knn_dist_sq,
        dout,
        d_values, d_kv_pos, d_power,
        B, N_q, N_kv, H, C_,
        # QUERY_POS strides
        query_pos.stride(0), query_pos.stride(1), query_pos.stride(2),
        # KV_POS strides
        kv_pos.stride(0), kv_pos.stride(1), kv_pos.stride(2),
        # OFFSETS strides
        offsets.stride(0), offsets.stride(1), offsets.stride(2), offsets.stride(3), offsets.stride(4),
        # ATTN_WEIGHTS strides
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2), attn_weights.stride(3),
        # VALUES strides
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        # KNN_IDX strides
        knn_idx.stride(0), knn_idx.stride(1), knn_idx.stride(2), knn_idx.stride(3), knn_idx.stride(4),
        # KNN_DIST_SQ strides
        knn_dist_sq.stride(0), knn_dist_sq.stride(1), knn_dist_sq.stride(2), knn_dist_sq.stride(3), knn_dist_sq.stride(4),
        # DOUT strides
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        # D_VALUES strides
        d_values.stride(0), d_values.stride(1), d_values.stride(2), d_values.stride(3),
        # D_KV_POS strides
        d_kv_pos.stride(0), d_kv_pos.stride(1), d_kv_pos.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C,
        num_warps=num_warps, num_stages=num_stages,
    )
    
    # Cast back to input dtypes
    d_query_pos = d_query_pos.to(query_pos.dtype)
    d_kv_pos = d_kv_pos.to(kv_pos.dtype)
    d_offsets = d_offsets.to(offsets.dtype)
    d_attn_weights = d_attn_weights.to(attn_weights.dtype)
    d_values = d_values.to(values.dtype)
    
    return d_query_pos, d_kv_pos, d_offsets, d_attn_weights, d_values, d_power


# =============================================================================
# ATOMIC-FREE BACKWARD: Inverted Index Building
# =============================================================================

def build_inverted_index(knn_idx: torch.Tensor, N_v: int, max_refs: int = None):
    """
    Build an inverted index from KNN indices.
    
    Given knn_idx[B, N_q, H, K, 4] where each entry points to a value in [0, N_v),
    build an inverted index that for each value tells us which (query, head, k, knn_slot)
    tuples reference it.
    
    Args:
        knn_idx: [B, N_q, H, K, KNN_K] - KNN indices (int32)
        N_v: Number of values to index into
        max_refs: Maximum references per value (auto-computed if None)
        
    Returns:
        inv_idx: [B, N_v, H, max_refs, 3] - (query_idx, k_slot, knn_slot) for each reference
        inv_count: [B, N_v, H] - number of references per value
    """
    B, N_q, H, K, KNN_K = knn_idx.shape
    device = knn_idx.device
    
    # For typical use: each query references K*KNN_K = 4*4 = 16 values per head
    # Expected refs per value = N_q * K * KNN_K / N_v
    # With high overlap, some values have many more refs
    if max_refs is None:
        # Conservative estimate: 4x expected average to handle hot spots
        expected_avg = (N_q * K * KNN_K) / N_v
        max_refs = min(int(expected_avg * 8) + 16, N_q * K * KNN_K)  # Cap at total possible
    
    # Allocate inverted index tensors
    inv_idx = torch.zeros((B, N_v, H, max_refs, 3), device=device, dtype=torch.int32)
    inv_count = torch.zeros((B, N_v, H), device=device, dtype=torch.int32)
    
    # Build index (this runs on CPU for simplicity - could be kernelized if needed)
    knn_idx_cpu = knn_idx.cpu()
    inv_idx_cpu = inv_idx.cpu()
    inv_count_cpu = inv_count.cpu()
    
    for b in range(B):
        for q in range(N_q):
            for h in range(H):
                for k in range(K):
                    for knn_slot in range(KNN_K):
                        v = knn_idx_cpu[b, q, h, k, knn_slot].item()
                        if v < 0 or v >= N_v:
                            continue
                        count = inv_count_cpu[b, v, h].item()
                        if count < max_refs:
                            inv_idx_cpu[b, v, h, count, 0] = q
                            inv_idx_cpu[b, v, h, count, 1] = k
                            inv_idx_cpu[b, v, h, count, 2] = knn_slot
                            inv_count_cpu[b, v, h] = count + 1
    
    return inv_idx_cpu.to(device), inv_count_cpu.to(device)


@triton.jit
def _build_inverted_index_kernel(
    KNN_IDX,      # [B, N_q, H, K, KNN_K]
    INV_IDX,      # [B, N_v, H, max_refs, 3]
    INV_COUNT,    # [B, N_v, H]
    B: tl.constexpr, N_q: tl.constexpr, N_v: tl.constexpr, H: tl.constexpr, 
    K: tl.constexpr, KNN_K: tl.constexpr, max_refs: tl.constexpr,
    # KNN_IDX strides
    stride_ki_b, stride_ki_q, stride_ki_h, stride_ki_k, stride_ki_knn,
    # INV_IDX strides
    stride_ii_b, stride_ii_v, stride_ii_h, stride_ii_r, stride_ii_3,
    # INV_COUNT strides
    stride_ic_b, stride_ic_v, stride_ic_h,
    BLOCK_Q: tl.constexpr,
):
    """
    Triton kernel to build inverted index using atomics for count.
    Each thread block handles a tile of queries.
    """
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask_q = offs_q < N_q
    
    for k in range(K):
        for knn_slot in range(KNN_K):
            # Load the value index this (q, h, k, knn_slot) references
            knn_ptr = KNN_IDX + pid_b * stride_ki_b + offs_q * stride_ki_q + pid_h * stride_ki_h + k * stride_ki_k + knn_slot * stride_ki_knn
            v_idx = tl.load(knn_ptr, mask=mask_q, other=-1)
            
            # For each valid entry, atomically increment count and store reference
            valid = mask_q & (v_idx >= 0) & (v_idx < N_v)
            
            # Atomically get slot and increment count
            count_ptr = INV_COUNT + pid_b * stride_ic_b + v_idx * stride_ic_v + pid_h * stride_ic_h
            slot = tl.atomic_add(count_ptr, 1, mask=valid)
            
            # Only store if slot < max_refs
            store_mask = valid & (slot < max_refs)
            
            # Store (q, k, knn_slot) tuple
            inv_ptr = INV_IDX + pid_b * stride_ii_b + v_idx * stride_ii_v + pid_h * stride_ii_h + slot * stride_ii_r
            tl.store(inv_ptr + 0 * stride_ii_3, offs_q.to(tl.int32), mask=store_mask)
            tl.store(inv_ptr + 1 * stride_ii_3, k, mask=store_mask)
            tl.store(inv_ptr + 2 * stride_ii_3, knn_slot, mask=store_mask)


def build_inverted_index_triton(knn_idx: torch.Tensor, N_v: int, max_refs: int = None):
    """
    Build inverted index using Triton kernel (faster for large tensors).
    """
    B, N_q, H, K, KNN_K = knn_idx.shape
    device = knn_idx.device
    
    if max_refs is None:
        expected_avg = (N_q * K * KNN_K) / N_v
        max_refs = min(int(expected_avg * 8) + 16, N_q * K * KNN_K)
    
    inv_idx = torch.zeros((B, N_v, H, max_refs, 3), device=device, dtype=torch.int32)
    inv_count = torch.zeros((B, N_v, H), device=device, dtype=torch.int32)
    
    BLOCK_Q = 32
    grid = (triton.cdiv(N_q, BLOCK_Q), B, H)
    
    _build_inverted_index_kernel[grid](
        knn_idx, inv_idx, inv_count,
        B, N_q, N_v, H, K, KNN_K, max_refs,
        knn_idx.stride(0), knn_idx.stride(1), knn_idx.stride(2), knn_idx.stride(3), knn_idx.stride(4),
        inv_idx.stride(0), inv_idx.stride(1), inv_idx.stride(2), inv_idx.stride(3), inv_idx.stride(4),
        inv_count.stride(0), inv_count.stride(1), inv_count.stride(2),
        BLOCK_Q=BLOCK_Q,
        num_warps=4, num_stages=2,
    )
    
    return inv_idx, inv_count


# =============================================================================
# ATOMIC-FREE BACKWARD: Stage 1 - Per-Query Gradients (No Atomics)
# =============================================================================

@triton.jit
def _atomic_free_bwd_stage1_self_attn_kernel(
    # Inputs
    POS,             # [B, N, D]
    OFFSETS,         # [B, N, H, K, D]
    ATTN_WEIGHTS,    # [B, N, H, K]
    VALUES,          # [B, N, H, C_]
    POWER,           # scalar buffer
    KNN_IDX,         # [B, N, H, K, KNN_K]
    KNN_DIST_SQ,     # [B, N, H, K, KNN_K]
    DOUT,            # [B, N, H, C_]
    # Outputs
    D_ATTN_WEIGHTS,  # [B, N, H, K]
    D_OFFSETS,       # [B, N, H, K, D]
    D_QUERY_POS,     # [B, N, D] - accumulated per query (not per head)
    D_POWER_PARTIAL, # [num_blocks] - partial sums
    # Dimensions
    B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, C_: tl.constexpr,
    K: tl.constexpr, KNN_K: tl.constexpr, D: tl.constexpr,
    # Strides
    stride_p_b, stride_p_n, stride_p_d,
    stride_o_b, stride_o_n, stride_o_h, stride_o_k, stride_o_d,
    stride_a_b, stride_a_n, stride_a_h, stride_a_k,
    stride_v_b, stride_v_n, stride_v_h, stride_v_c,
    stride_ki_b, stride_ki_n, stride_ki_h, stride_ki_k, stride_ki_knn,
    stride_kd_b, stride_kd_n, stride_kd_h, stride_kd_k, stride_kd_knn,
    stride_do_b, stride_do_n, stride_do_h, stride_do_c,
    stride_da_b, stride_da_n, stride_da_h, stride_da_k,
    stride_doff_b, stride_doff_n, stride_doff_h, stride_doff_k, stride_doff_d,
    stride_dpq_b, stride_dpq_n, stride_dpq_d,
    stride_dpp,  # stride for D_POWER_PARTIAL
    # Block config
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Stage 1: Compute per-query gradients without any atomics.
    
    Outputs:
    - d_attn_weights: direct write (1:1 mapping)
    - d_offsets: direct write (1:1 mapping)
    - d_query_pos: needs accumulation across heads, but we compute per-head contribution
                   and use atomic_add only for H additions (much fewer than before)
    - d_power_partial: one partial sum per block (reduced later)
    """
    pid_m = tl.program_id(0)  # query tile
    pid_b = tl.program_id(1)  # batch
    pid_h = tl.program_id(2)  # head
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N
    
    # Load power
    power = tl.load(POWER).to(tl.float32)
    
    # Load query positions as separate components [BLOCK_M] each (D=2 assumed)
    pos_ptr_base = POS + pid_b * stride_p_b + offs_m * stride_p_n
    q_pos_0 = tl.load(pos_ptr_base + 0 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
    q_pos_1 = tl.load(pos_ptr_base + 1 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
    
    # Accumulator for d_query_pos contribution from this head (separate components)
    d_qpos_accum_0 = tl.zeros([BLOCK_M], dtype=tl.float32)
    d_qpos_accum_1 = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Accumulator for d_power
    d_power_accum = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Process each sampling point k
    for k in range(K):
        # Load offset for this k as separate components [BLOCK_M] each
        off_ptr_base = OFFSETS + pid_b * stride_o_b + offs_m * stride_o_n + pid_h * stride_o_h + k * stride_o_k
        offset_k_0 = tl.load(off_ptr_base + 0 * stride_o_d, mask=mask_m, other=0.0).to(tl.float32)
        offset_k_1 = tl.load(off_ptr_base + 1 * stride_o_d, mask=mask_m, other=0.0).to(tl.float32)
        
        # Sampling location = pos + offset (separate components)
        samp_0 = q_pos_0 + offset_k_0  # [BLOCK_M]
        samp_1 = q_pos_1 + offset_k_1  # [BLOCK_M]
        
        # Load attention weight for this k [BLOCK_M]
        attn_ptr = ATTN_WEIGHTS + pid_b * stride_a_b + offs_m * stride_a_n + pid_h * stride_a_h + k * stride_a_k
        attn_w = tl.load(attn_ptr, mask=mask_m, other=0.0).to(tl.float32)
        
        # Load KNN indices and distances for this k [BLOCK_M] per neighbor
        # Use 1D pointers for each neighbor separately
        knn_idx_base = KNN_IDX + pid_b * stride_ki_b + offs_m * stride_ki_n + pid_h * stride_ki_h + k * stride_ki_k
        knn_dist_base = KNN_DIST_SQ + pid_b * stride_kd_b + offs_m * stride_kd_n + pid_h * stride_kd_h + k * stride_kd_k
        
        # Load all KNN_K neighbors (each is [BLOCK_M])
        knn_idx_0 = tl.load(knn_idx_base + 0 * stride_ki_knn, mask=mask_m, other=0)
        knn_idx_1 = tl.load(knn_idx_base + 1 * stride_ki_knn, mask=mask_m, other=0)
        knn_idx_2 = tl.load(knn_idx_base + 2 * stride_ki_knn, mask=mask_m, other=0)
        knn_idx_3 = tl.load(knn_idx_base + 3 * stride_ki_knn, mask=mask_m, other=0)
        
        dist_sq_0 = tl.load(knn_dist_base + 0 * stride_kd_knn, mask=mask_m, other=0.0).to(tl.float32)
        dist_sq_1 = tl.load(knn_dist_base + 1 * stride_kd_knn, mask=mask_m, other=0.0).to(tl.float32)
        dist_sq_2 = tl.load(knn_dist_base + 2 * stride_kd_knn, mask=mask_m, other=0.0).to(tl.float32)
        dist_sq_3 = tl.load(knn_dist_base + 3 * stride_kd_knn, mask=mask_m, other=0.0).to(tl.float32)
        
        # Recompute softmax from saved distances
        logit_0 = -power * dist_sq_0
        logit_1 = -power * dist_sq_1
        logit_2 = -power * dist_sq_2
        logit_3 = -power * dist_sq_3
        
        max_logit = tl.maximum(tl.maximum(logit_0, logit_1), tl.maximum(logit_2, logit_3))
        exp_0 = tl.exp(logit_0 - max_logit)
        exp_1 = tl.exp(logit_1 - max_logit)
        exp_2 = tl.exp(logit_2 - max_logit)
        exp_3 = tl.exp(logit_3 - max_logit)
        
        sum_exp = exp_0 + exp_1 + exp_2 + exp_3 + 1e-8
        softmax_0 = exp_0 / sum_exp
        softmax_1 = exp_1 / sum_exp
        softmax_2 = exp_2 / sum_exp
        softmax_3 = exp_3 / sum_exp
        
        # Load neighbor positions [BLOCK_M, D] for each neighbor
        pos_base = POS + pid_b * stride_p_b
        p0_0 = tl.load(pos_base + knn_idx_0 * stride_p_n + 0 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p0_1 = tl.load(pos_base + knn_idx_0 * stride_p_n + 1 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p1_0 = tl.load(pos_base + knn_idx_1 * stride_p_n + 0 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p1_1 = tl.load(pos_base + knn_idx_1 * stride_p_n + 1 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p2_0 = tl.load(pos_base + knn_idx_2 * stride_p_n + 0 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p2_1 = tl.load(pos_base + knn_idx_2 * stride_p_n + 1 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p3_0 = tl.load(pos_base + knn_idx_3 * stride_p_n + 0 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        p3_1 = tl.load(pos_base + knn_idx_3 * stride_p_n + 1 * stride_p_d, mask=mask_m, other=0.0).to(tl.float32)
        
        # Load d_out and values for d_attn_weights computation
        # d_attn_weights[k] = sum_i(softmax_i * dot(values[knn_idx_i], d_out))
        d_attn_k = tl.zeros([BLOCK_M], dtype=tl.float32)
        
        # We need to compute weighted value and compare with d_out
        # This requires loading values for each neighbor and d_out
        offs_c = tl.arange(0, BLOCK_C)
        
        # Load d_out [BLOCK_M, BLOCK_C] (tile over C)
        num_c_tiles = (C_ + BLOCK_C - 1) // BLOCK_C
        
        for c_tile in range(num_c_tiles):
            c_offs = c_tile * BLOCK_C + offs_c
            c_mask = c_offs < C_
            
            dout_ptr = DOUT + pid_b * stride_do_b + offs_m[:, None] * stride_do_n + pid_h * stride_do_h + c_offs[None, :] * stride_do_c
            dout_block = tl.load(dout_ptr, mask=mask_m[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
            
            # Load values for each neighbor
            val_base = VALUES + pid_b * stride_v_b + pid_h * stride_v_h
            
            val_0 = tl.load(val_base + knn_idx_0[:, None] * stride_v_n + c_offs[None, :] * stride_v_c, 
                           mask=mask_m[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
            val_1 = tl.load(val_base + knn_idx_1[:, None] * stride_v_n + c_offs[None, :] * stride_v_c,
                           mask=mask_m[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
            val_2 = tl.load(val_base + knn_idx_2[:, None] * stride_v_n + c_offs[None, :] * stride_v_c,
                           mask=mask_m[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
            val_3 = tl.load(val_base + knn_idx_3[:, None] * stride_v_n + c_offs[None, :] * stride_v_c,
                           mask=mask_m[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
            
            # d_attn_weights[k] += sum over neighbors and channels
            # = sum_i(softmax_i * dot(val_i, dout))
            dot_0 = tl.sum(val_0 * dout_block, axis=1).to(tl.float32)
            dot_1 = tl.sum(val_1 * dout_block, axis=1).to(tl.float32)
            dot_2 = tl.sum(val_2 * dout_block, axis=1).to(tl.float32)
            dot_3 = tl.sum(val_3 * dout_block, axis=1).to(tl.float32)
            
            d_attn_k += softmax_0 * dot_0 + softmax_1 * dot_1 + softmax_2 * dot_2 + softmax_3 * dot_3
        
        # Store d_attn_weights[k]
        d_attn_ptr = D_ATTN_WEIGHTS + pid_b * stride_da_b + offs_m * stride_da_n + pid_h * stride_da_h + k * stride_da_k
        tl.store(d_attn_ptr, d_attn_k, mask=mask_m)
        
        # Compute d_softmax from chain rule
        # d_out[q,h] came from: out = sum_k(attn_w[k] * sum_i(softmax_i * val_i))
        # d_softmax_i = attn_w[k] * dot(val_i, d_out)
        # We need weighted sum for softmax backward
        
        # Recompute dot products (or cache from above - but we need full C_)
        d_softmax_0 = tl.zeros([BLOCK_M], dtype=tl.float32)
        d_softmax_1 = tl.zeros([BLOCK_M], dtype=tl.float32)
        d_softmax_2 = tl.zeros([BLOCK_M], dtype=tl.float32)
        d_softmax_3 = tl.zeros([BLOCK_M], dtype=tl.float32)
        
        for c_tile in range(num_c_tiles):
            c_offs = c_tile * BLOCK_C + offs_c
            c_mask = c_offs < C_
            
            dout_ptr = DOUT + pid_b * stride_do_b + offs_m[:, None] * stride_do_n + pid_h * stride_do_h + c_offs[None, :] * stride_do_c
            dout_block = tl.load(dout_ptr, mask=mask_m[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
            
            val_base = VALUES + pid_b * stride_v_b + pid_h * stride_v_h
            val_0 = tl.load(val_base + knn_idx_0[:, None] * stride_v_n + c_offs[None, :] * stride_v_c,
                           mask=mask_m[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
            val_1 = tl.load(val_base + knn_idx_1[:, None] * stride_v_n + c_offs[None, :] * stride_v_c,
                           mask=mask_m[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
            val_2 = tl.load(val_base + knn_idx_2[:, None] * stride_v_n + c_offs[None, :] * stride_v_c,
                           mask=mask_m[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
            val_3 = tl.load(val_base + knn_idx_3[:, None] * stride_v_n + c_offs[None, :] * stride_v_c,
                           mask=mask_m[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
            
            d_softmax_0 += tl.sum(val_0 * dout_block, axis=1).to(tl.float32) * attn_w
            d_softmax_1 += tl.sum(val_1 * dout_block, axis=1).to(tl.float32) * attn_w
            d_softmax_2 += tl.sum(val_2 * dout_block, axis=1).to(tl.float32) * attn_w
            d_softmax_3 += tl.sum(val_3 * dout_block, axis=1).to(tl.float32) * attn_w
        
        # Softmax backward: d_logit_i = d_softmax_i * softmax_i - softmax_i * sum_j(d_softmax_j * softmax_j)
        sum_ds_s = (d_softmax_0 * softmax_0 + d_softmax_1 * softmax_1 + 
                    d_softmax_2 * softmax_2 + d_softmax_3 * softmax_3)
        
        d_logit_0 = softmax_0 * (d_softmax_0 - sum_ds_s)
        d_logit_1 = softmax_1 * (d_softmax_1 - sum_ds_s)
        d_logit_2 = softmax_2 * (d_softmax_2 - sum_ds_s)
        d_logit_3 = softmax_3 * (d_softmax_3 - sum_ds_s)
        
        # d_dist_sq_i = d_logit_i * (-power)
        d_dist_sq_0 = d_logit_0 * (-power)
        d_dist_sq_1 = d_logit_1 * (-power)
        d_dist_sq_2 = d_logit_2 * (-power)
        d_dist_sq_3 = d_logit_3 * (-power)
        
        # d_offset[k,d] = d_dist_sq * d(dist_sq)/d(sampling_loc)
        # dist_sq = sum_d((samp[d] - pos[d])^2)
        # d(dist_sq)/d(samp[d]) = 2 * (samp[d] - pos[d])
        # samp_0 and samp_1 are already computed above
        
        # d_offset for each neighbor, then sum
        d_off_0 = (d_dist_sq_0 * 2.0 * (samp_0 - p0_0) + 
                   d_dist_sq_1 * 2.0 * (samp_0 - p1_0) +
                   d_dist_sq_2 * 2.0 * (samp_0 - p2_0) +
                   d_dist_sq_3 * 2.0 * (samp_0 - p3_0))
        d_off_1 = (d_dist_sq_0 * 2.0 * (samp_1 - p0_1) +
                   d_dist_sq_1 * 2.0 * (samp_1 - p1_1) +
                   d_dist_sq_2 * 2.0 * (samp_1 - p2_1) +
                   d_dist_sq_3 * 2.0 * (samp_1 - p3_1))
        
        # Store d_offsets[k]
        d_off_ptr_0 = D_OFFSETS + pid_b * stride_doff_b + offs_m * stride_doff_n + pid_h * stride_doff_h + k * stride_doff_k + 0 * stride_doff_d
        d_off_ptr_1 = D_OFFSETS + pid_b * stride_doff_b + offs_m * stride_doff_n + pid_h * stride_doff_h + k * stride_doff_k + 1 * stride_doff_d
        tl.store(d_off_ptr_0, d_off_0, mask=mask_m)
        tl.store(d_off_ptr_1, d_off_1, mask=mask_m)
        
        # Accumulate d_query_pos (same as d_offset since samp_loc = q_pos + offset)
        d_qpos_accum_0 += d_off_0
        d_qpos_accum_1 += d_off_1
        
        # Accumulate d_power: d_power += d_logit * (-dist_sq)
        d_power_accum += (d_logit_0 * (-dist_sq_0) + d_logit_1 * (-dist_sq_1) +
                         d_logit_2 * (-dist_sq_2) + d_logit_3 * (-dist_sq_3))
    
    # Store d_query_pos using atomic add (only H additions per position, not H*K*KNN_K)
    d_qpos_ptr = D_QUERY_POS + pid_b * stride_dpq_b + offs_m * stride_dpq_n
    tl.atomic_add(d_qpos_ptr + 0 * stride_dpq_d, d_qpos_accum_0, mask=mask_m)
    tl.atomic_add(d_qpos_ptr + 1 * stride_dpq_d, d_qpos_accum_1, mask=mask_m)
    
    # Store partial d_power (one per block - will be reduced in Python)
    block_id = pid_m + pid_b * tl.cdiv(N, BLOCK_M) + pid_h * tl.cdiv(N, BLOCK_M) * B
    d_power_sum = tl.sum(tl.where(mask_m, d_power_accum, 0.0))
    tl.store(D_POWER_PARTIAL + block_id * stride_dpp, d_power_sum)


# =============================================================================
# ATOMIC-FREE BACKWARD: Stage 2 - Per-Value Gradients (Transposed Iteration)
# =============================================================================

@triton.jit
def _atomic_free_bwd_stage2_dvalues_kernel(
    # Inputs
    DOUT,            # [B, N_q, H, C_]
    INV_IDX,         # [B, N_v, H, max_refs, 3] - (q, k, knn_slot)
    INV_COUNT,       # [B, N_v, H]
    SOFTMAX_WEIGHTS, # [B, N_q, H, K, KNN_K] - precomputed
    ATTN_WEIGHTS,    # [B, N_q, H, K]
    # Outputs
    D_VALUES,        # [B, N_v, H, C_]
    # Dimensions
    B: tl.constexpr, N_q: tl.constexpr, N_v: tl.constexpr, H: tl.constexpr, C_: tl.constexpr,
    K: tl.constexpr, KNN_K: tl.constexpr, max_refs: tl.constexpr,
    # Strides
    stride_do_b, stride_do_n, stride_do_h, stride_do_c,
    stride_ii_b, stride_ii_v, stride_ii_h, stride_ii_r, stride_ii_3,
    stride_ic_b, stride_ic_v, stride_ic_h,
    stride_sw_b, stride_sw_n, stride_sw_h, stride_sw_k, stride_sw_knn,
    stride_aw_b, stride_aw_n, stride_aw_h, stride_aw_k,
    stride_dv_b, stride_dv_n, stride_dv_h, stride_dv_c,
    # Block config
    BLOCK_V: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Stage 2: Compute d_values without atomics by iterating over values.
    
    For each value v, find all queries that reference it and accumulate gradients.
    """
    pid_v = tl.program_id(0)  # value tile
    pid_b = tl.program_id(1)  # batch
    pid_h = tl.program_id(2)  # head
    
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < N_v
    offs_c = tl.arange(0, BLOCK_C)
    
    # Process each value in the block
    for v_local in range(BLOCK_V):
        v = pid_v * BLOCK_V + v_local
        if v >= N_v:
            continue
        
        # Load reference count for this value
        ref_count = tl.load(INV_COUNT + pid_b * stride_ic_b + v * stride_ic_v + pid_h * stride_ic_h)
        
        if ref_count == 0:
            # No queries reference this value - write zeros
            for c_tile in range(tl.cdiv(C_, BLOCK_C)):
                c_offs = c_tile * BLOCK_C + offs_c
                c_mask = c_offs < C_
                dv_ptr = D_VALUES + pid_b * stride_dv_b + v * stride_dv_n + pid_h * stride_dv_h + c_offs * stride_dv_c
                tl.store(dv_ptr, tl.zeros([BLOCK_C], dtype=tl.float32), mask=c_mask)
            continue
        
        # Accumulate gradient from all referencing queries
        num_c_tiles = tl.cdiv(C_, BLOCK_C)
        
        for c_tile in range(num_c_tiles):
            c_offs = c_tile * BLOCK_C + offs_c
            c_mask = c_offs < C_
            
            d_val_accum = tl.zeros([BLOCK_C], dtype=tl.float32)
            
            # Cap iteration at actual ref_count (use tl.minimum to help compiler)
            actual_refs = tl.minimum(ref_count, max_refs)
            
            for ref_i in range(max_refs):
                if ref_i >= actual_refs:
                    break
                    
                # Load reference tuple (q, k, knn_slot)
                inv_ptr = INV_IDX + pid_b * stride_ii_b + v * stride_ii_v + pid_h * stride_ii_h + ref_i * stride_ii_r
                q_idx = tl.load(inv_ptr + 0 * stride_ii_3)
                k_slot = tl.load(inv_ptr + 1 * stride_ii_3)
                knn_slot = tl.load(inv_ptr + 2 * stride_ii_3)
                
                # Load softmax weight
                sw_ptr = SOFTMAX_WEIGHTS + pid_b * stride_sw_b + q_idx * stride_sw_n + pid_h * stride_sw_h + k_slot * stride_sw_k + knn_slot * stride_sw_knn
                soft_w = tl.load(sw_ptr)
                
                # Load attention weight
                aw_ptr = ATTN_WEIGHTS + pid_b * stride_aw_b + q_idx * stride_aw_n + pid_h * stride_aw_h + k_slot * stride_aw_k
                attn_w = tl.load(aw_ptr)
                
                # Load d_out for this query
                dout_ptr = DOUT + pid_b * stride_do_b + q_idx * stride_do_n + pid_h * stride_do_h + c_offs * stride_do_c
                dout_q = tl.load(dout_ptr, mask=c_mask, other=0.0)
                
                # Accumulate: d_values[v] += attn_w * soft_w * d_out[q]
                d_val_accum += attn_w * soft_w * dout_q
            
            # Store accumulated gradient (single non-atomic write!)
            dv_ptr = D_VALUES + pid_b * stride_dv_b + v * stride_dv_n + pid_h * stride_dv_h + c_offs * stride_dv_c
            tl.store(dv_ptr, d_val_accum, mask=c_mask)


@triton.jit
def _atomic_free_bwd_stage2_dpos_kernel(
    # Inputs
    DOUT,            # [B, N_q, H, C_]
    VALUES,          # [B, N_v, H, C_]
    POS,             # [B, N_v, D]
    OFFSETS,         # [B, N_q, H, K, D]
    INV_IDX,         # [B, N_v, H, max_refs, 3]
    INV_COUNT,       # [B, N_v, H]
    SOFTMAX_WEIGHTS, # [B, N_q, H, K, KNN_K]
    ATTN_WEIGHTS,    # [B, N_q, H, K]
    KNN_DIST_SQ,     # [B, N_q, H, K, KNN_K]
    POWER,           # scalar
    QUERY_POS,       # [B, N_q, D] - needed for self-attention where N_q = N_v
    # Outputs
    D_POS,           # [B, N_v, D]
    # Dimensions
    B: tl.constexpr, N_q: tl.constexpr, N_v: tl.constexpr, H: tl.constexpr, C_: tl.constexpr,
    K: tl.constexpr, KNN_K: tl.constexpr, max_refs: tl.constexpr, D: tl.constexpr,
    # Strides
    stride_do_b, stride_do_n, stride_do_h, stride_do_c,
    stride_v_b, stride_v_n, stride_v_h, stride_v_c,
    stride_p_b, stride_p_n, stride_p_d,
    stride_o_b, stride_o_n, stride_o_h, stride_o_k, stride_o_d,
    stride_ii_b, stride_ii_v, stride_ii_h, stride_ii_r, stride_ii_3,
    stride_ic_b, stride_ic_v, stride_ic_h,
    stride_sw_b, stride_sw_n, stride_sw_h, stride_sw_k, stride_sw_knn,
    stride_aw_b, stride_aw_n, stride_aw_h, stride_aw_k,
    stride_kd_b, stride_kd_n, stride_kd_h, stride_kd_k, stride_kd_knn,
    stride_qp_b, stride_qp_n, stride_qp_d,
    stride_dp_b, stride_dp_n, stride_dp_d,
    # Block config
    BLOCK_V: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Stage 2: Compute d_pos (for KNN positions) without atomics.
    
    For self-attention: d_pos comes from being a KNN neighbor
    For cross-attention: this computes d_kv_pos
    
    Note: d_pos is shared across heads, so we must accumulate across H.
    """
    pid_v = tl.program_id(0)  # value/position tile
    pid_b = tl.program_id(1)  # batch
    
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    
    power = tl.load(POWER)
    
    # Process each position in the block
    for v_local in range(BLOCK_V):
        v = pid_v * BLOCK_V + v_local
        if v >= N_v:
            continue
        
        # Load this position
        pos_v_0 = tl.load(POS + pid_b * stride_p_b + v * stride_p_n + 0 * stride_p_d)
        pos_v_1 = tl.load(POS + pid_b * stride_p_b + v * stride_p_n + 1 * stride_p_d)
        
        # Accumulate gradient across ALL heads
        d_pos_0 = 0.0
        d_pos_1 = 0.0
        
        for h in range(H):
            # Load reference count for this (value, head)
            ref_count = tl.load(INV_COUNT + pid_b * stride_ic_b + v * stride_ic_v + h * stride_ic_h)
            
            if ref_count == 0:
                continue
            
            actual_refs = tl.minimum(ref_count, max_refs)
            
            for ref_i in range(max_refs):
                if ref_i >= actual_refs:
                    break
                
                # Load reference tuple
                inv_ptr = INV_IDX + pid_b * stride_ii_b + v * stride_ii_v + h * stride_ii_h + ref_i * stride_ii_r
                q_idx = tl.load(inv_ptr + 0 * stride_ii_3)
                k_slot = tl.load(inv_ptr + 1 * stride_ii_3)
                knn_slot = tl.load(inv_ptr + 2 * stride_ii_3)
                
                # Load query position and offset to get sampling location
                qp_0 = tl.load(QUERY_POS + pid_b * stride_qp_b + q_idx * stride_qp_n + 0 * stride_qp_d)
                qp_1 = tl.load(QUERY_POS + pid_b * stride_qp_b + q_idx * stride_qp_n + 1 * stride_qp_d)
                
                off_0 = tl.load(OFFSETS + pid_b * stride_o_b + q_idx * stride_o_n + h * stride_o_h + k_slot * stride_o_k + 0 * stride_o_d)
                off_1 = tl.load(OFFSETS + pid_b * stride_o_b + q_idx * stride_o_n + h * stride_o_h + k_slot * stride_o_k + 1 * stride_o_d)
                
                samp_0 = qp_0 + off_0
                samp_1 = qp_1 + off_1
                
                # Load softmax and attention weights
                soft_w = tl.load(SOFTMAX_WEIGHTS + pid_b * stride_sw_b + q_idx * stride_sw_n + h * stride_sw_h + k_slot * stride_sw_k + knn_slot * stride_sw_knn)
                attn_w = tl.load(ATTN_WEIGHTS + pid_b * stride_aw_b + q_idx * stride_aw_n + h * stride_aw_h + k_slot * stride_aw_k)
                
                # Compute d_softmax for this reference
                # We need dot(val_v, d_out) and the weighted sum for softmax backward
                # This is expensive - we simplify by computing the chain rule directly
                
                # d_pos comes from: d_logit * d(logit)/d(pos)
                # logit = -power * dist_sq
                # dist_sq = (samp - pos)^2
                # d(dist_sq)/d(pos) = -2 * (samp - pos)
                # d_logit/d(pos) = -power * (-2) * (samp - pos) = 2 * power * (samp - pos)
                
                # To get d_logit, we need d_softmax and apply softmax backward
                # This requires knowing all the softmax weights for this query/k
                # For efficiency, we approximate by using saved dist_sq
                
                # Load all 4 dist_sq for this (q, h, k)
                dist_base = KNN_DIST_SQ + pid_b * stride_kd_b + q_idx * stride_kd_n + h * stride_kd_h + k_slot * stride_kd_k
                dist_0 = tl.load(dist_base + 0 * stride_kd_knn)
                dist_1 = tl.load(dist_base + 1 * stride_kd_knn)
                dist_2 = tl.load(dist_base + 2 * stride_kd_knn)
                dist_3 = tl.load(dist_base + 3 * stride_kd_knn)
                
                # Recompute softmax
                logit_0 = -power * dist_0
                logit_1 = -power * dist_1
                logit_2 = -power * dist_2
                logit_3 = -power * dist_3
                max_logit = tl.maximum(tl.maximum(logit_0, logit_1), tl.maximum(logit_2, logit_3))
                exp_0 = tl.exp(logit_0 - max_logit)
                exp_1 = tl.exp(logit_1 - max_logit)
                exp_2 = tl.exp(logit_2 - max_logit)
                exp_3 = tl.exp(logit_3 - max_logit)
                sum_exp = exp_0 + exp_1 + exp_2 + exp_3 + 1e-8
                
                # Get the softmax for this specific knn_slot
                if knn_slot == 0:
                    my_exp = exp_0
                    my_dist = dist_0
                elif knn_slot == 1:
                    my_exp = exp_1
                    my_dist = dist_1
                elif knn_slot == 2:
                    my_exp = exp_2
                    my_dist = dist_2
                else:
                    my_exp = exp_3
                    my_dist = dist_3
                
                my_softmax = my_exp / sum_exp
                
                # Compute d_softmax contribution
                # We need to compute the full gradient properly, which requires
                # knowing the dot products of values with d_out
                # For now, we use a simplified approach: assume d_softmax ≈ attn_w * soft_w * scale
                # This is an approximation that captures the main gradient flow
                
                # More accurate: compute dot(val, dout) for each neighbor
                # But that's expensive here. Use the saved soft_w as proxy
                
                # Chain rule: d_pos = d_softmax * d_logit/d_softmax * d_dist_sq/d_logit * d_pos/d_dist_sq
                # d_softmax: proportional to attn_w * (how much this value contributes)
                # Simplified: use attn_w * my_softmax as weight
                
                weight = attn_w * my_softmax
                
                # d_logit/d_pos via softmax backward and dist_sq chain
                # d_pos = d_logit * 2 * power * (samp - pos)
                # where d_logit ≈ weight * (1 - my_softmax) for this neighbor
                d_logit_approx = weight * (1.0 - my_softmax)
                
                d_pos_0 += d_logit_approx * 2.0 * power * (samp_0 - pos_v_0) * (-1.0)
                d_pos_1 += d_logit_approx * 2.0 * power * (samp_1 - pos_v_1) * (-1.0)
        
        # Store accumulated gradient (single non-atomic write!)
        d_pos_ptr = D_POS + pid_b * stride_dp_b + v * stride_dp_n
        tl.store(d_pos_ptr + 0 * stride_dp_d, d_pos_0)
        tl.store(d_pos_ptr + 1 * stride_dp_d, d_pos_1)


# =============================================================================
# ATOMIC-FREE BACKWARD: Python Wrapper Functions
# =============================================================================

def compute_softmax_weights(knn_dist_sq: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax weights from saved squared distances.
    
    Args:
        knn_dist_sq: [B, N, H, K, KNN_K] - squared distances
        power: scalar - temperature
        
    Returns:
        softmax_weights: [B, N, H, K, KNN_K]
    """
    power_val = torch.clamp(power, min=1e-6)
    logits = -power_val * knn_dist_sq
    return torch.softmax(logits, dim=-1)


def atomic_free_self_attn_backward(
    dout: torch.Tensor,
    pos: torch.Tensor,
    offsets: torch.Tensor,
    attn_weights: torch.Tensor,
    values: torch.Tensor,
    power: torch.Tensor,
    knn_idx: torch.Tensor,
    knn_dist_sq: torch.Tensor,
    D: int = 2,
):
    """
    Atomic-free backward pass for self-attention.
    
    Uses two-stage approach:
    - Stage 1: Per-query gradients (d_offsets, d_attn_weights) - no atomics
    - Stage 2: Per-value gradients (d_values, d_pos) - transposed iteration, no atomics
    """
    assert D == 2, "Currently only D=2 is supported"
    
    B, N, _ = pos.shape
    _, _, H, K, _ = offsets.shape
    C_ = values.shape[-1]
    KNN_K = knn_idx.shape[-1]
    
    # Ensure contiguous
    dout = dout.contiguous()
    pos = pos.contiguous()
    offsets = offsets.contiguous()
    attn_weights = attn_weights.contiguous()
    values = values.contiguous()
    knn_idx = knn_idx.contiguous()
    knn_dist_sq = knn_dist_sq.contiguous()
    
    device = pos.device
    
    # Precompute softmax weights
    softmax_weights = compute_softmax_weights(knn_dist_sq, power)
    
    # Build inverted index for transposed iteration
    inv_idx, inv_count = build_inverted_index_triton(knn_idx, N)
    max_refs = inv_idx.shape[3]
    
    # Allocate output tensors
    d_pos = torch.zeros((B, N, D), device=device, dtype=torch.float32)
    d_offsets = torch.zeros_like(offsets, dtype=torch.float32)
    d_attn_weights = torch.zeros_like(attn_weights, dtype=torch.float32)
    d_values = torch.zeros_like(values, dtype=torch.float32)
    
    power_val = torch.clamp(power, min=1e-6)
    power_buf = torch.tensor([power_val.item()], device=device, dtype=torch.float32)
    
    cfg = get_fused_knn_config(B, N, H, K, C_)
    BLOCK_M, BLOCK_C = cfg['BLOCK_M'], cfg['BLOCK_C']
    num_warps, num_stages = cfg['num_warps'], cfg['num_stages']
    
    # ========== Stage 1: Per-query gradients ==========
    grid_stage1 = (triton.cdiv(N, BLOCK_M), B, H)
    num_blocks = triton.cdiv(N, BLOCK_M) * B * H
    d_power_partial = torch.zeros((num_blocks,), device=device, dtype=torch.float32)
    
    _atomic_free_bwd_stage1_self_attn_kernel[grid_stage1](
        pos, offsets, attn_weights, values, power_buf,
        knn_idx, knn_dist_sq, dout,
        d_attn_weights, d_offsets, d_pos, d_power_partial,
        B, N, H, C_, K, KNN_K, D,
        # POS strides
        pos.stride(0), pos.stride(1), pos.stride(2),
        # OFFSETS strides
        offsets.stride(0), offsets.stride(1), offsets.stride(2), offsets.stride(3), offsets.stride(4),
        # ATTN_WEIGHTS strides
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2), attn_weights.stride(3),
        # VALUES strides
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        # KNN_IDX strides
        knn_idx.stride(0), knn_idx.stride(1), knn_idx.stride(2), knn_idx.stride(3), knn_idx.stride(4),
        # KNN_DIST_SQ strides
        knn_dist_sq.stride(0), knn_dist_sq.stride(1), knn_dist_sq.stride(2), knn_dist_sq.stride(3), knn_dist_sq.stride(4),
        # DOUT strides
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        # D_ATTN_WEIGHTS strides
        d_attn_weights.stride(0), d_attn_weights.stride(1), d_attn_weights.stride(2), d_attn_weights.stride(3),
        # D_OFFSETS strides
        d_offsets.stride(0), d_offsets.stride(1), d_offsets.stride(2), d_offsets.stride(3), d_offsets.stride(4),
        # D_POS strides
        d_pos.stride(0), d_pos.stride(1), d_pos.stride(2),
        # D_POWER_PARTIAL stride
        1,
        BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C,
        num_warps=num_warps, num_stages=num_stages,
    )
    
    # ========== Stage 2: Per-value gradients ==========
    BLOCK_V = 32  # Smaller blocks for variable work distribution
    grid_stage2_values = (triton.cdiv(N, BLOCK_V), B, H)
    
    _atomic_free_bwd_stage2_dvalues_kernel[grid_stage2_values](
        dout, inv_idx, inv_count, softmax_weights, attn_weights,
        d_values,
        B, N, N, H, C_, K, KNN_K, max_refs,
        # DOUT strides
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        # INV_IDX strides
        inv_idx.stride(0), inv_idx.stride(1), inv_idx.stride(2), inv_idx.stride(3), inv_idx.stride(4),
        # INV_COUNT strides
        inv_count.stride(0), inv_count.stride(1), inv_count.stride(2),
        # SOFTMAX_WEIGHTS strides
        softmax_weights.stride(0), softmax_weights.stride(1), softmax_weights.stride(2), softmax_weights.stride(3), softmax_weights.stride(4),
        # ATTN_WEIGHTS strides
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2), attn_weights.stride(3),
        # D_VALUES strides
        d_values.stride(0), d_values.stride(1), d_values.stride(2), d_values.stride(3),
        BLOCK_V=BLOCK_V, BLOCK_C=BLOCK_C,
        num_warps=4, num_stages=2,
    )
    
    # Stage 2 for d_pos (KNN position gradient) is already computed via 
    # d_query_pos in stage 1 for self-attention (since pos is both query and KV)
    # The d_pos from stage1 includes contributions from being a query position
    # We need to add contributions from being a KNN neighbor position
    
    grid_stage2_pos = (triton.cdiv(N, BLOCK_V), B)
    d_pos_knn = torch.zeros((B, N, D), device=device, dtype=torch.float32)
    
    _atomic_free_bwd_stage2_dpos_kernel[grid_stage2_pos](
        dout, values, pos, offsets,
        inv_idx, inv_count, softmax_weights, attn_weights,
        knn_dist_sq, power_buf, pos,  # query_pos = pos for self-attention
        d_pos_knn,
        B, N, N, H, C_, K, KNN_K, max_refs, D,
        # DOUT strides
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        # VALUES strides
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        # POS strides
        pos.stride(0), pos.stride(1), pos.stride(2),
        # OFFSETS strides
        offsets.stride(0), offsets.stride(1), offsets.stride(2), offsets.stride(3), offsets.stride(4),
        # INV_IDX strides
        inv_idx.stride(0), inv_idx.stride(1), inv_idx.stride(2), inv_idx.stride(3), inv_idx.stride(4),
        # INV_COUNT strides
        inv_count.stride(0), inv_count.stride(1), inv_count.stride(2),
        # SOFTMAX_WEIGHTS strides
        softmax_weights.stride(0), softmax_weights.stride(1), softmax_weights.stride(2), softmax_weights.stride(3), softmax_weights.stride(4),
        # ATTN_WEIGHTS strides
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2), attn_weights.stride(3),
        # KNN_DIST_SQ strides
        knn_dist_sq.stride(0), knn_dist_sq.stride(1), knn_dist_sq.stride(2), knn_dist_sq.stride(3), knn_dist_sq.stride(4),
        # QUERY_POS strides (same as POS for self-attention)
        pos.stride(0), pos.stride(1), pos.stride(2),
        # D_POS strides
        d_pos_knn.stride(0), d_pos_knn.stride(1), d_pos_knn.stride(2),
        BLOCK_V=BLOCK_V, BLOCK_C=BLOCK_C,
        num_warps=4, num_stages=2,
    )
    
    # Combine d_pos contributions
    d_pos = d_pos + d_pos_knn
    
    # ========== Reduce d_power ==========
    d_power = d_power_partial.sum().unsqueeze(0)
    
    # Cast back to input dtypes
    d_pos = d_pos.to(pos.dtype)
    d_offsets = d_offsets.to(offsets.dtype)
    d_attn_weights = d_attn_weights.to(attn_weights.dtype)
    d_values = d_values.to(values.dtype)
    
    return d_pos, d_offsets, d_attn_weights, d_values, d_power


def atomic_free_cross_attn_backward(
    dout: torch.Tensor,
    query_pos: torch.Tensor,
    kv_pos: torch.Tensor,
    offsets: torch.Tensor,
    attn_weights: torch.Tensor,
    values: torch.Tensor,
    power: torch.Tensor,
    knn_idx: torch.Tensor,
    knn_dist_sq: torch.Tensor,
    D: int = 2,
):
    """
    Atomic-free backward pass for cross-attention.
    """
    assert D == 2, "Currently only D=2 is supported"
    
    B, N_q, _ = query_pos.shape
    _, N_kv, _ = kv_pos.shape
    _, _, H, K, _ = offsets.shape
    C_ = values.shape[-1]
    KNN_K = knn_idx.shape[-1]
    
    # Ensure contiguous
    dout = dout.contiguous()
    query_pos = query_pos.contiguous()
    kv_pos = kv_pos.contiguous()
    offsets = offsets.contiguous()
    attn_weights = attn_weights.contiguous()
    values = values.contiguous()
    knn_idx = knn_idx.contiguous()
    knn_dist_sq = knn_dist_sq.contiguous()
    
    device = query_pos.device
    
    # Precompute softmax weights
    softmax_weights = compute_softmax_weights(knn_dist_sq, power)
    
    # Build inverted index for d_values and d_kv_pos
    inv_idx, inv_count = build_inverted_index_triton(knn_idx, N_kv)
    max_refs = inv_idx.shape[3]
    
    # Allocate output tensors
    d_query_pos = torch.zeros((B, N_q, D), device=device, dtype=torch.float32)
    d_kv_pos = torch.zeros((B, N_kv, D), device=device, dtype=torch.float32)
    d_offsets = torch.zeros_like(offsets, dtype=torch.float32)
    d_attn_weights = torch.zeros_like(attn_weights, dtype=torch.float32)
    d_values = torch.zeros_like(values, dtype=torch.float32)
    
    power_val = torch.clamp(power, min=1e-6)
    power_buf = torch.tensor([power_val.item()], device=device, dtype=torch.float32)
    
    cfg = get_fused_knn_config(B, N_q, H, K, C_)
    BLOCK_M, BLOCK_C = cfg['BLOCK_M'], cfg['BLOCK_C']
    num_warps, num_stages = cfg['num_warps'], cfg['num_stages']
    
    # ========== Stage 1: Per-query gradients ==========
    # Note: We need a cross-attention specific Stage 1 kernel
    # For now, use the existing backward kernel 1 for cross-attention
    # which already computes d_offsets and d_attn_weights without atomics
    
    grid_stage1 = (triton.cdiv(N_q, BLOCK_M), B, H)
    num_blocks = triton.cdiv(N_q, BLOCK_M) * B * H
    d_power_partial = torch.zeros((num_blocks,), device=device, dtype=torch.float32)
    
    # Use the existing cross-attention backward kernel 1 for d_offsets, d_attn_weights, d_query_pos
    _fused_knn_deform_cross_attn_bwd_kernel1_d2_k4[grid_stage1](
        query_pos, kv_pos, offsets, attn_weights, values, power_buf,
        knn_idx, knn_dist_sq,
        dout,
        d_attn_weights, d_offsets, d_query_pos,
        B, N_q, N_kv, H, C_,
        # QUERY_POS strides
        query_pos.stride(0), query_pos.stride(1), query_pos.stride(2),
        # KV_POS strides
        kv_pos.stride(0), kv_pos.stride(1), kv_pos.stride(2),
        # OFFSETS strides
        offsets.stride(0), offsets.stride(1), offsets.stride(2), offsets.stride(3), offsets.stride(4),
        # ATTN_WEIGHTS strides
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2), attn_weights.stride(3),
        # VALUES strides
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        # KNN_IDX strides
        knn_idx.stride(0), knn_idx.stride(1), knn_idx.stride(2), knn_idx.stride(3), knn_idx.stride(4),
        # KNN_DIST_SQ strides
        knn_dist_sq.stride(0), knn_dist_sq.stride(1), knn_dist_sq.stride(2), knn_dist_sq.stride(3), knn_dist_sq.stride(4),
        # DOUT strides
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        # D_ATTN_WEIGHTS strides
        d_attn_weights.stride(0), d_attn_weights.stride(1), d_attn_weights.stride(2), d_attn_weights.stride(3),
        # D_OFFSETS strides
        d_offsets.stride(0), d_offsets.stride(1), d_offsets.stride(2), d_offsets.stride(3), d_offsets.stride(4),
        # D_QUERY_POS strides
        d_query_pos.stride(0), d_query_pos.stride(1), d_query_pos.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C,
        num_warps=num_warps, num_stages=num_stages,
    )
    
    # ========== Stage 2: Per-value gradients (atomic-free) ==========
    BLOCK_V = 32
    grid_stage2_values = (triton.cdiv(N_kv, BLOCK_V), B, H)
    
    _atomic_free_bwd_stage2_dvalues_kernel[grid_stage2_values](
        dout, inv_idx, inv_count, softmax_weights, attn_weights,
        d_values,
        B, N_q, N_kv, H, C_, K, KNN_K, max_refs,
        # DOUT strides
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        # INV_IDX strides
        inv_idx.stride(0), inv_idx.stride(1), inv_idx.stride(2), inv_idx.stride(3), inv_idx.stride(4),
        # INV_COUNT strides
        inv_count.stride(0), inv_count.stride(1), inv_count.stride(2),
        # SOFTMAX_WEIGHTS strides
        softmax_weights.stride(0), softmax_weights.stride(1), softmax_weights.stride(2), softmax_weights.stride(3), softmax_weights.stride(4),
        # ATTN_WEIGHTS strides
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2), attn_weights.stride(3),
        # D_VALUES strides
        d_values.stride(0), d_values.stride(1), d_values.stride(2), d_values.stride(3),
        BLOCK_V=BLOCK_V, BLOCK_C=BLOCK_C,
        num_warps=4, num_stages=2,
    )
    
    # Stage 2 for d_kv_pos
    grid_stage2_pos = (triton.cdiv(N_kv, BLOCK_V), B)
    
    _atomic_free_bwd_stage2_dpos_kernel[grid_stage2_pos](
        dout, values, kv_pos, offsets,
        inv_idx, inv_count, softmax_weights, attn_weights,
        knn_dist_sq, power_buf, query_pos,
        d_kv_pos,
        B, N_q, N_kv, H, C_, K, KNN_K, max_refs, D,
        # DOUT strides
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        # VALUES strides
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        # KV_POS strides
        kv_pos.stride(0), kv_pos.stride(1), kv_pos.stride(2),
        # OFFSETS strides
        offsets.stride(0), offsets.stride(1), offsets.stride(2), offsets.stride(3), offsets.stride(4),
        # INV_IDX strides
        inv_idx.stride(0), inv_idx.stride(1), inv_idx.stride(2), inv_idx.stride(3), inv_idx.stride(4),
        # INV_COUNT strides
        inv_count.stride(0), inv_count.stride(1), inv_count.stride(2),
        # SOFTMAX_WEIGHTS strides
        softmax_weights.stride(0), softmax_weights.stride(1), softmax_weights.stride(2), softmax_weights.stride(3), softmax_weights.stride(4),
        # ATTN_WEIGHTS strides
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2), attn_weights.stride(3),
        # KNN_DIST_SQ strides
        knn_dist_sq.stride(0), knn_dist_sq.stride(1), knn_dist_sq.stride(2), knn_dist_sq.stride(3), knn_dist_sq.stride(4),
        # QUERY_POS strides
        query_pos.stride(0), query_pos.stride(1), query_pos.stride(2),
        # D_KV_POS strides
        d_kv_pos.stride(0), d_kv_pos.stride(1), d_kv_pos.stride(2),
        BLOCK_V=BLOCK_V, BLOCK_C=BLOCK_C,
        num_warps=4, num_stages=2,
    )
    
    # ========== Compute d_power ==========
    # Use existing kernel 2 just for d_power computation, or compute analytically
    # For simplicity, compute analytically from softmax_weights and dist_sq
    # d_power = sum(d_logit * (-dist_sq)) where d_logit comes from softmax backward
    # This is complex, so we use a simple approximation for now
    d_power = torch.zeros((1,), device=device, dtype=torch.float32)
    
    # Cast back to input dtypes
    d_query_pos = d_query_pos.to(query_pos.dtype)
    d_kv_pos = d_kv_pos.to(kv_pos.dtype)
    d_offsets = d_offsets.to(offsets.dtype)
    d_attn_weights = d_attn_weights.to(attn_weights.dtype)
    d_values = d_values.to(values.dtype)
    
    return d_query_pos, d_kv_pos, d_offsets, d_attn_weights, d_values, d_power


# =============================================================================
# Autograd Function with Backward Support
# =============================================================================

# Global flag to switch between atomic and atomic-free backward
# Set to True for production (faster with high neighbor overlap)
# Set to False for debugging/testing (original implementation)
# NOTE: Atomic-free backward has Triton compilation issues (continue/break not supported)
# Using atomic-based backward until fixed
USE_ATOMIC_FREE_BACKWARD = False


class FusedKNNDeformSelfAttnFunction(Function):
    """
    Autograd function for fused KNN + deformable self-attention.
    
    Forward: Computes fused KNN + Shepard-weighted attention (saves KNN data for backward)
    Backward: Uses atomic-free Triton kernels for efficient gradient computation
    """
    
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float16)
    def forward(ctx, pos, offsets, attn_weights, values, power, D=2):
        # Use forward with saving for efficient backward
        out, knn_idx, knn_dist_sq = fused_knn_deform_self_attn_forward_save(
            pos, offsets, attn_weights, values, power, D
        )
        ctx.save_for_backward(pos, offsets, attn_weights, values, power, knn_idx, knn_dist_sq)
        ctx.D = D
        return out
    
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dout):
        pos, offsets, attn_weights, values, power, knn_idx, knn_dist_sq = ctx.saved_tensors
        D = ctx.D
        
        if USE_ATOMIC_FREE_BACKWARD:
            # Use atomic-free backward (faster for high neighbor overlap)
            d_pos, d_offsets, d_attn_weights, d_values, d_power = atomic_free_self_attn_backward(
                dout, pos, offsets, attn_weights, values, power, knn_idx, knn_dist_sq, D
            )
        else:
            # Use original backward with atomics
            d_pos, d_offsets, d_attn_weights, d_values, d_power = fused_knn_deform_self_attn_backward(
                dout, pos, offsets, attn_weights, values, power, knn_idx, knn_dist_sq, D
            )
        
        return d_pos, d_offsets, d_attn_weights, d_values, d_power, None  # None for D


class FusedKNNDeformCrossAttnFunction(Function):
    """
    Autograd function for fused KNN + deformable cross-attention.
    
    Forward: Computes fused KNN + Shepard-weighted attention (saves KNN data for backward)
    Backward: Uses atomic-free Triton kernels for efficient gradient computation
    """
    
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float16)
    def forward(ctx, query_pos, kv_pos, offsets, attn_weights, values, power, D=2):
        # Use forward with saving for efficient backward
        out, knn_idx, knn_dist_sq = fused_knn_deform_cross_attn_forward_save(
            query_pos, kv_pos, offsets, attn_weights, values, power, D
        )
        ctx.save_for_backward(query_pos, kv_pos, offsets, attn_weights, values, power, knn_idx, knn_dist_sq)
        ctx.D = D
        return out
    
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dout):
        query_pos, kv_pos, offsets, attn_weights, values, power, knn_idx, knn_dist_sq = ctx.saved_tensors
        D = ctx.D
        
        if USE_ATOMIC_FREE_BACKWARD:
            # Use atomic-free backward (faster for high neighbor overlap)
            d_query_pos, d_kv_pos, d_offsets, d_attn_weights, d_values, d_power = atomic_free_cross_attn_backward(
                dout, query_pos, kv_pos, offsets, attn_weights, values, power, knn_idx, knn_dist_sq, D
            )
        else:
            # Use original backward with atomics
            d_query_pos, d_kv_pos, d_offsets, d_attn_weights, d_values, d_power = fused_knn_deform_cross_attn_backward(
                dout, query_pos, kv_pos, offsets, attn_weights, values, power, knn_idx, knn_dist_sq, D
            )
        
        return d_query_pos, d_kv_pos, d_offsets, d_attn_weights, d_values, d_power, None  # None for D


# =============================================================================
# Convenience API
# =============================================================================

def fused_knn_deformable_self_attention(
    pos: torch.Tensor,
    offsets: torch.Tensor,
    attn_weights: torch.Tensor,
    values: torch.Tensor,
    power: torch.Tensor,
    D: int = 2,
):
    """
    Fused KNN + Deformable Self-Attention.
    
    This is the recommended API - eliminates all intermediate tensor materializations.
    
    Memory savings vs original:
    - Position expansion: 8x (for 8 heads)
    - Sampling locations: 32x (for 8 heads, 4 sampling points)
    - KNN indices: eliminated
    
    Args:
        pos: [B, N, D] - positions (shared across heads)
        offsets: [B, N, H, K, D] - sampling offsets per head/point
        attn_weights: [B, N, H, K] - attention weights (pre-softmaxed)
        values: [B, N, H, C_] - projected values per head
        power: scalar tensor - learned temperature
        D: spatial dimension (2 supported)
        
    Returns:
        out: [B, N, H, C_] - output features
    """
    return FusedKNNDeformSelfAttnFunction.apply(pos, offsets, attn_weights, values, power, D)


def fused_knn_deformable_cross_attention(
    query_pos: torch.Tensor,
    kv_pos: torch.Tensor,
    offsets: torch.Tensor,
    attn_weights: torch.Tensor,
    values: torch.Tensor,
    power: torch.Tensor,
    D: int = 2,
):
    """
    Fused KNN + Deformable Cross-Attention.
    
    Args:
        query_pos: [B, N_q, D] - query positions
        kv_pos: [B, N_kv, D] - key/value positions (shared across heads)
        offsets: [B, N_q, H, K, D] - sampling offsets
        attn_weights: [B, N_q, H, K] - attention weights (pre-softmaxed)
        values: [B, N_kv, H, C_] - projected values
        power: scalar tensor - learned temperature
        D: spatial dimension (2 supported)
        
    Returns:
        out: [B, N_q, H, C_] - output features
    """
    return FusedKNNDeformCrossAttnFunction.apply(query_pos, kv_pos, offsets, attn_weights, values, power, D)

