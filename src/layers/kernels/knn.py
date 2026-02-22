"""
Streaming FP16 KNN Kernel using KeOps-style symbolic/lazy evaluation. Used Claude Opus and planned most, impressed by the bugs it caught...

This kernel computes K-nearest neighbors without materializing the full O(M*N) distance matrix.
Instead, it uses a tiled map-reduce scheme that keeps all intermediate state in CUDA registers.

Architecture follows principles from:
- KeOps: Kernel Operations on GPU (https://github.com/getkeops/keops)
- Fast Geometric Learning with Symbolic Matrices (NeurIPS 2020)
"""

import torch
from torch.autograd import Function
from torch.amp import custom_fwd
import triton
import triton.language as tl

try:
    from .util import NUM_STAGES_OPTIONS, is_hip
except ImportError:
    from util import NUM_STAGES_OPTIONS, is_hip


# =============================================================================
# Helper: Compute L2 squared norm inline (for D=2,3,4)
# =============================================================================

@triton.jit
def compute_l2_norm_sq_d2(
    ptr, offset, stride_n, stride_d, mask
):
    """Compute ||x||² for D=2, fully unrolled."""
    x0 = tl.load(ptr + offset * stride_n + 0 * stride_d, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(ptr + offset * stride_n + 1 * stride_d, mask=mask, other=0.0).to(tl.float32)
    return x0 * x0 + x1 * x1


@triton.jit
def compute_l2_norm_sq_d3(
    ptr, offset, stride_n, stride_d, mask
):
    """Compute ||x||² for D=3, fully unrolled."""
    x0 = tl.load(ptr + offset * stride_n + 0 * stride_d, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(ptr + offset * stride_n + 1 * stride_d, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(ptr + offset * stride_n + 2 * stride_d, mask=mask, other=0.0).to(tl.float32)
    return x0 * x0 + x1 * x1 + x2 * x2


@triton.jit
def compute_l2_norm_sq_d4(
    ptr, offset, stride_n, stride_d, mask
):
    """Compute ||x||² for D=4, fully unrolled."""
    x0 = tl.load(ptr + offset * stride_n + 0 * stride_d, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(ptr + offset * stride_n + 1 * stride_d, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(ptr + offset * stride_n + 2 * stride_d, mask=mask, other=0.0).to(tl.float32)
    x3 = tl.load(ptr + offset * stride_n + 3 * stride_d, mask=mask, other=0.0).to(tl.float32)
    return x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3


# =============================================================================
# Helper: Load coordinates for a tile
# =============================================================================

@triton.jit
def load_coords_d2(ptr, offsets, stride_n, stride_d, mask):
    """Load 2D coordinates [BLOCK, 2] as separate arrays."""
    x0 = tl.load(ptr + offsets * stride_n + 0 * stride_d, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(ptr + offsets * stride_n + 1 * stride_d, mask=mask, other=0.0).to(tl.float32)
    return x0, x1


@triton.jit
def load_coords_d3(ptr, offsets, stride_n, stride_d, mask):
    """Load 3D coordinates [BLOCK, 3] as separate arrays."""
    x0 = tl.load(ptr + offsets * stride_n + 0 * stride_d, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(ptr + offsets * stride_n + 1 * stride_d, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(ptr + offsets * stride_n + 2 * stride_d, mask=mask, other=0.0).to(tl.float32)
    return x0, x1, x2


@triton.jit
def load_coords_d4(ptr, offsets, stride_n, stride_d, mask):
    """Load 4D coordinates [BLOCK, 4] as separate arrays."""
    x0 = tl.load(ptr + offsets * stride_n + 0 * stride_d, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(ptr + offsets * stride_n + 1 * stride_d, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(ptr + offsets * stride_n + 2 * stride_d, mask=mask, other=0.0).to(tl.float32)
    x3 = tl.load(ptr + offsets * stride_n + 3 * stride_d, mask=mask, other=0.0).to(tl.float32)
    return x0, x1, x2, x3


# =============================================================================
# Helper: Compute pairwise squared distances
# =============================================================================

@triton.jit
def compute_pairwise_dist_sq_d2(
    q0, q1,  # query coords [BLOCK_M]
    r0, r1,  # ref coords [BLOCK_N]
):
    """Compute ||q - r||² for D=2, returns [BLOCK_M, BLOCK_N]."""
    d0 = q0[:, None] - r0[None, :]
    d1 = q1[:, None] - r1[None, :]
    return d0 * d0 + d1 * d1


@triton.jit
def compute_pairwise_dist_sq_d3(
    q0, q1, q2,  # query coords [BLOCK_M]
    r0, r1, r2,  # ref coords [BLOCK_N]
):
    """Compute ||q - r||² for D=3, returns [BLOCK_M, BLOCK_N]."""
    d0 = q0[:, None] - r0[None, :]
    d1 = q1[:, None] - r1[None, :]
    d2 = q2[:, None] - r2[None, :]
    return d0 * d0 + d1 * d1 + d2 * d2


@triton.jit
def compute_pairwise_dist_sq_d4(
    q0, q1, q2, q3,  # query coords [BLOCK_M]
    r0, r1, r2, r3,  # ref coords [BLOCK_N]
):
    """Compute ||q - r||² for D=4, returns [BLOCK_M, BLOCK_N]."""
    d0 = q0[:, None] - r0[None, :]
    d1 = q1[:, None] - r1[None, :]
    d2 = q2[:, None] - r2[None, :]
    d3 = q3[:, None] - r3[None, :]
    return d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3


# =============================================================================
# Online K-min: Process single candidate and update top-K (Legacy - for fallback)
# =============================================================================

@triton.jit
def update_topk_single(
    top_k_dist,   # [BLOCK_M, K] - current top-K distances (sorted ascending)
    top_k_idx,    # [BLOCK_M, K] - current top-K indices
    new_dist,     # [BLOCK_M] - new candidate distance
    new_idx,      # int - new candidate index
    valid,        # [BLOCK_M] - validity mask
    K: tl.constexpr,
):
    """
    Update top-K with a single new candidate per row.
    Simple replace-max-and-bubble approach.
    """
    offs_k = tl.arange(0, K)
    
    # Get current max (last element, since sorted ascending)
    max_dist = tl.sum(tl.where(offs_k[None, :] == K - 1, top_k_dist, 0.0), axis=1)  # [BLOCK_M]
    
    # Should we update? new_dist < max_dist AND valid
    should_update = (new_dist < max_dist) & valid
    
    # Replace max element with new candidate
    top_k_dist = tl.where(
        should_update[:, None] & (offs_k[None, :] == K - 1),
        new_dist[:, None],
        top_k_dist
    )
    top_k_idx = tl.where(
        should_update[:, None] & (offs_k[None, :] == K - 1),
        new_idx,
        top_k_idx
    )
    
    # Bubble sort: move the new element to its correct position
    # Unrolled for small K
    for i in tl.static_range(K - 1, 0, -1):
        left_mask = offs_k[None, :] == i - 1
        right_mask = offs_k[None, :] == i
        
        left_dist = tl.sum(tl.where(left_mask, top_k_dist, 0.0), axis=1)
        right_dist = tl.sum(tl.where(right_mask, top_k_dist, 0.0), axis=1)
        left_idx = tl.sum(tl.where(left_mask, top_k_idx, 0), axis=1)
        right_idx = tl.sum(tl.where(right_mask, top_k_idx, 0), axis=1)
        
        should_swap = right_dist < left_dist
        
        new_left_dist = tl.where(should_swap, right_dist, left_dist)
        new_right_dist = tl.where(should_swap, left_dist, right_dist)
        new_left_idx = tl.where(should_swap, right_idx, left_idx)
        new_right_idx = tl.where(should_swap, left_idx, right_idx)
        
        top_k_dist = tl.where(left_mask, new_left_dist[:, None], top_k_dist)
        top_k_dist = tl.where(right_mask, new_right_dist[:, None], top_k_dist)
        top_k_idx = tl.where(left_mask, new_left_idx[:, None], top_k_idx)
        top_k_idx = tl.where(right_mask, new_right_idx[:, None], top_k_idx)
    
    return top_k_dist, top_k_idx


# =============================================================================
# Simplified Scalar-Register Top-K (matches KeOps approach)
# =============================================================================
# 
# Key insight: For small K (4 or 8), use scalar variables instead of tensor ops.
# Each query maintains K scalar distances/indices that get simple conditional updates.
# This matches how KeOps internally handles top-K selection.
#
# Complexity: O(N × K) simple scalar comparisons per query
# vs Plan C: O(N × K × BLOCK_N) tensor operations per query tile


# =============================================================================
# Simplified Scalar-Register Top-K Kernels
# =============================================================================
#
# These kernels use a simple streaming approach with scalar variables for top-K:
# - Load reference tiles for coalesced memory access
# - Compute distances for all queries to current ref tile
# - Update top-K using simple conditional insertion (like KeOps)
#
# This is much simpler than the bitonic merge approach and compiles faster.

# =============================================================================
# Heuristic Config Selection (avoids autotuning overhead)
# =============================================================================

def get_knn_heuristic_config(M, N, K, D):
    """
    Select optimal BLOCK_M and BLOCK_N based on problem size.
    
    Based on extensive autotuning results on NVIDIA TITAN RTX:
    - BLOCK_M=128, BLOCK_N=32 wins in almost all cases
    - Only exception: very small M (<= 64) benefits from BLOCK_M=64
    
    Returns:
        (BLOCK_M, BLOCK_N, num_warps, num_stages)
    """
    # Autotuning consistently shows BLOCK_M=128, BLOCK_N=32 is optimal
    BLOCK_M = 128
    BLOCK_N = 32
    num_warps = 4
    num_stages = 2
    
    # Only reduce BLOCK_M for very small M to avoid wasted threads
    if M <= 64:
        BLOCK_M = 64
    
    return BLOCK_M, BLOCK_N, num_warps, num_stages


# Autotune configurations (used when use_heuristic=False)
configs_knn_simple = [
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
]


# =============================================================================
# Non-Autotuned Kernels (for heuristic mode)
# =============================================================================

@triton.jit
def _knn_kernel_simple_k4_d2_noautotune(
    QUERY, REF, OUT_IDX,
    B, M, N,
    stride_q_b, stride_q_m, stride_q_d,
    stride_r_b, stride_r_n, stride_r_d,
    stride_o_b, stride_o_m, stride_o_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Simplified KNN kernel for D=2, K=4 using streaming scalar top-K.
    
    Each query maintains 4 scalar distance/index values that get updated
    via simple conditional insertion - matches KeOps's approach.
    """
    INF: tl.constexpr = 1e38
    
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    
    query_base = QUERY + pid_b * stride_q_b
    ref_base = REF + pid_b * stride_r_b
    
    # Load query coordinates [BLOCK_M]
    q0 = tl.load(query_base + offs_m * stride_q_m + 0 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q1 = tl.load(query_base + offs_m * stride_q_m + 1 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    
    # Initialize top-4 as separate scalar arrays [BLOCK_M] each
    # d0 <= d1 <= d2 <= d3 (sorted ascending)
    d0 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d1 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d2 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d3 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    i0 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i1 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i2 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i3 = tl.zeros([BLOCK_M], dtype=tl.int32)
    
    # Stream through all reference points
    for ref_idx in range(N):
        # Load single reference point (broadcast to all queries)
        r0 = tl.load(ref_base + ref_idx * stride_r_n + 0 * stride_r_d).to(tl.float32)
        r1 = tl.load(ref_base + ref_idx * stride_r_n + 1 * stride_r_d).to(tl.float32)
        
        # Compute squared distance to all queries [BLOCK_M]
        diff0 = q0 - r0
        diff1 = q1 - r1
        dist = diff0 * diff0 + diff1 * diff1
        
        # Optimized insertion: compute all comparisons, then do single-pass update
        # Only update if dist < d3 (current worst)
        u0 = dist < d0
        u1 = dist < d1
        u2 = dist < d2
        u3 = dist < d3
        
        # Simplified cascading shift-and-insert (6 tl.where ops instead of 14)
        # Position 3: gets d2 if shifting, else dist if inserting at 3, else unchanged
        d3 = tl.where(u2, d2, tl.where(u3, dist, d3))
        i3 = tl.where(u2, i2, tl.where(u3, ref_idx, i3))
        # Position 2: gets d1 if shifting from 0/1, else dist if inserting at 2, else unchanged
        d2 = tl.where(u1, d1, tl.where(u2, dist, d2))
        i2 = tl.where(u1, i1, tl.where(u2, ref_idx, i2))
        # Position 1: gets d0 if inserting at 0, else dist if inserting at 1, else unchanged
        d1 = tl.where(u0, d0, tl.where(u1, dist, d1))
        i1 = tl.where(u0, i0, tl.where(u1, ref_idx, i1))
        # Position 0: gets dist if inserting at 0, else unchanged
        d0 = tl.where(u0, dist, d0)
        i0 = tl.where(u0, ref_idx, i0)
    
    # Store results [BLOCK_M, 4]
    out_base = OUT_IDX + pid_b * stride_o_b
    tl.store(out_base + offs_m * stride_o_m + 0 * stride_o_k, i0, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 1 * stride_o_k, i1, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 2 * stride_o_k, i2, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 3 * stride_o_k, i3, mask=mask_m)


@triton.jit
def _knn_kernel_simple_k4_d3_noautotune(
    QUERY, REF, OUT_IDX, B, M, N,
    stride_q_b, stride_q_m, stride_q_d,
    stride_r_b, stride_r_n, stride_r_d,
    stride_o_b, stride_o_m, stride_o_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Non-autotuned K=4 D=3 kernel for heuristic mode."""
    INF: tl.constexpr = 1e38
    pid_m, pid_b = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    query_base = QUERY + pid_b * stride_q_b
    ref_base = REF + pid_b * stride_r_b
    q0 = tl.load(query_base + offs_m * stride_q_m + 0 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q1 = tl.load(query_base + offs_m * stride_q_m + 1 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q2 = tl.load(query_base + offs_m * stride_q_m + 2 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    d0 = tl.full([BLOCK_M], INF, dtype=tl.float32); d1 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d2 = tl.full([BLOCK_M], INF, dtype=tl.float32); d3 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    i0 = tl.zeros([BLOCK_M], dtype=tl.int32); i1 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i2 = tl.zeros([BLOCK_M], dtype=tl.int32); i3 = tl.zeros([BLOCK_M], dtype=tl.int32)
    for ref_idx in range(N):
        r0 = tl.load(ref_base + ref_idx * stride_r_n + 0 * stride_r_d).to(tl.float32)
        r1 = tl.load(ref_base + ref_idx * stride_r_n + 1 * stride_r_d).to(tl.float32)
        r2 = tl.load(ref_base + ref_idx * stride_r_n + 2 * stride_r_d).to(tl.float32)
        diff0 = q0 - r0; diff1 = q1 - r1; diff2 = q2 - r2
        dist = diff0*diff0 + diff1*diff1 + diff2*diff2
        u0 = dist < d0; u1 = dist < d1; u2 = dist < d2; u3 = dist < d3
        d3 = tl.where(u2, d2, tl.where(u3, dist, d3)); i3 = tl.where(u2, i2, tl.where(u3, ref_idx, i3))
        d2 = tl.where(u1, d1, tl.where(u2, dist, d2)); i2 = tl.where(u1, i1, tl.where(u2, ref_idx, i2))
        d1 = tl.where(u0, d0, tl.where(u1, dist, d1)); i1 = tl.where(u0, i0, tl.where(u1, ref_idx, i1))
        d0 = tl.where(u0, dist, d0); i0 = tl.where(u0, ref_idx, i0)
    out_base = OUT_IDX + pid_b * stride_o_b
    tl.store(out_base + offs_m * stride_o_m + 0 * stride_o_k, i0, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 1 * stride_o_k, i1, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 2 * stride_o_k, i2, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 3 * stride_o_k, i3, mask=mask_m)


@triton.jit
def _knn_kernel_simple_k4_d4_noautotune(
    QUERY, REF, OUT_IDX, B, M, N,
    stride_q_b, stride_q_m, stride_q_d,
    stride_r_b, stride_r_n, stride_r_d,
    stride_o_b, stride_o_m, stride_o_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Non-autotuned K=4 D=4 kernel for heuristic mode."""
    INF: tl.constexpr = 1e38
    pid_m, pid_b = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    query_base = QUERY + pid_b * stride_q_b
    ref_base = REF + pid_b * stride_r_b
    q0 = tl.load(query_base + offs_m * stride_q_m + 0 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q1 = tl.load(query_base + offs_m * stride_q_m + 1 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q2 = tl.load(query_base + offs_m * stride_q_m + 2 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q3 = tl.load(query_base + offs_m * stride_q_m + 3 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    d0 = tl.full([BLOCK_M], INF, dtype=tl.float32); d1 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d2 = tl.full([BLOCK_M], INF, dtype=tl.float32); d3 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    i0 = tl.zeros([BLOCK_M], dtype=tl.int32); i1 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i2 = tl.zeros([BLOCK_M], dtype=tl.int32); i3 = tl.zeros([BLOCK_M], dtype=tl.int32)
    for ref_idx in range(N):
        r0 = tl.load(ref_base + ref_idx * stride_r_n + 0 * stride_r_d).to(tl.float32)
        r1 = tl.load(ref_base + ref_idx * stride_r_n + 1 * stride_r_d).to(tl.float32)
        r2 = tl.load(ref_base + ref_idx * stride_r_n + 2 * stride_r_d).to(tl.float32)
        r3 = tl.load(ref_base + ref_idx * stride_r_n + 3 * stride_r_d).to(tl.float32)
        diff0 = q0 - r0; diff1 = q1 - r1; diff2 = q2 - r2; diff3 = q3 - r3
        dist = diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
        u0 = dist < d0; u1 = dist < d1; u2 = dist < d2; u3 = dist < d3
        d3 = tl.where(u2, d2, tl.where(u3, dist, d3)); i3 = tl.where(u2, i2, tl.where(u3, ref_idx, i3))
        d2 = tl.where(u1, d1, tl.where(u2, dist, d2)); i2 = tl.where(u1, i1, tl.where(u2, ref_idx, i2))
        d1 = tl.where(u0, d0, tl.where(u1, dist, d1)); i1 = tl.where(u0, i0, tl.where(u1, ref_idx, i1))
        d0 = tl.where(u0, dist, d0); i0 = tl.where(u0, ref_idx, i0)
    out_base = OUT_IDX + pid_b * stride_o_b
    tl.store(out_base + offs_m * stride_o_m + 0 * stride_o_k, i0, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 1 * stride_o_k, i1, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 2 * stride_o_k, i2, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 3 * stride_o_k, i3, mask=mask_m)


@triton.jit
def _knn_kernel_simple_k8_d2_noautotune(
    QUERY, REF, OUT_IDX, B, M, N,
    stride_q_b, stride_q_m, stride_q_d,
    stride_r_b, stride_r_n, stride_r_d,
    stride_o_b, stride_o_m, stride_o_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Non-autotuned K=8 D=2 kernel for heuristic mode."""
    INF: tl.constexpr = 1e38
    pid_m, pid_b = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    query_base = QUERY + pid_b * stride_q_b
    ref_base = REF + pid_b * stride_r_b
    q0 = tl.load(query_base + offs_m * stride_q_m + 0 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q1 = tl.load(query_base + offs_m * stride_q_m + 1 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    d0 = tl.full([BLOCK_M], INF, dtype=tl.float32); d1 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d2 = tl.full([BLOCK_M], INF, dtype=tl.float32); d3 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d4 = tl.full([BLOCK_M], INF, dtype=tl.float32); d5 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d6 = tl.full([BLOCK_M], INF, dtype=tl.float32); d7 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    i0 = tl.zeros([BLOCK_M], dtype=tl.int32); i1 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i2 = tl.zeros([BLOCK_M], dtype=tl.int32); i3 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i4 = tl.zeros([BLOCK_M], dtype=tl.int32); i5 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i6 = tl.zeros([BLOCK_M], dtype=tl.int32); i7 = tl.zeros([BLOCK_M], dtype=tl.int32)
    for ref_idx in range(N):
        r0 = tl.load(ref_base + ref_idx * stride_r_n + 0 * stride_r_d).to(tl.float32)
        r1 = tl.load(ref_base + ref_idx * stride_r_n + 1 * stride_r_d).to(tl.float32)
        diff0 = q0 - r0; diff1 = q1 - r1
        dist = diff0*diff0 + diff1*diff1
        u0 = dist < d0; u1 = dist < d1; u2 = dist < d2; u3 = dist < d3
        u4 = dist < d4; u5 = dist < d5; u6 = dist < d6; u7 = dist < d7
        # Simplified 16-op insertion (down from 30+)
        d7 = tl.where(u6, d6, tl.where(u7, dist, d7)); i7 = tl.where(u6, i6, tl.where(u7, ref_idx, i7))
        d6 = tl.where(u5, d5, tl.where(u6, dist, d6)); i6 = tl.where(u5, i5, tl.where(u6, ref_idx, i6))
        d5 = tl.where(u4, d4, tl.where(u5, dist, d5)); i5 = tl.where(u4, i4, tl.where(u5, ref_idx, i5))
        d4 = tl.where(u3, d3, tl.where(u4, dist, d4)); i4 = tl.where(u3, i3, tl.where(u4, ref_idx, i4))
        d3 = tl.where(u2, d2, tl.where(u3, dist, d3)); i3 = tl.where(u2, i2, tl.where(u3, ref_idx, i3))
        d2 = tl.where(u1, d1, tl.where(u2, dist, d2)); i2 = tl.where(u1, i1, tl.where(u2, ref_idx, i2))
        d1 = tl.where(u0, d0, tl.where(u1, dist, d1)); i1 = tl.where(u0, i0, tl.where(u1, ref_idx, i1))
        d0 = tl.where(u0, dist, d0); i0 = tl.where(u0, ref_idx, i0)
    out_base = OUT_IDX + pid_b * stride_o_b
    tl.store(out_base + offs_m * stride_o_m + 0 * stride_o_k, i0, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 1 * stride_o_k, i1, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 2 * stride_o_k, i2, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 3 * stride_o_k, i3, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 4 * stride_o_k, i4, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 5 * stride_o_k, i5, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 6 * stride_o_k, i6, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 7 * stride_o_k, i7, mask=mask_m)


@triton.jit
def _knn_kernel_simple_k8_d3_noautotune(
    QUERY, REF, OUT_IDX, B, M, N,
    stride_q_b, stride_q_m, stride_q_d,
    stride_r_b, stride_r_n, stride_r_d,
    stride_o_b, stride_o_m, stride_o_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Non-autotuned K=8 D=3 kernel for heuristic mode."""
    INF: tl.constexpr = 1e38
    pid_m, pid_b = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    query_base = QUERY + pid_b * stride_q_b
    ref_base = REF + pid_b * stride_r_b
    q0 = tl.load(query_base + offs_m * stride_q_m + 0 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q1 = tl.load(query_base + offs_m * stride_q_m + 1 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q2 = tl.load(query_base + offs_m * stride_q_m + 2 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    d0 = tl.full([BLOCK_M], INF, dtype=tl.float32); d1 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d2 = tl.full([BLOCK_M], INF, dtype=tl.float32); d3 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d4 = tl.full([BLOCK_M], INF, dtype=tl.float32); d5 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d6 = tl.full([BLOCK_M], INF, dtype=tl.float32); d7 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    i0 = tl.zeros([BLOCK_M], dtype=tl.int32); i1 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i2 = tl.zeros([BLOCK_M], dtype=tl.int32); i3 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i4 = tl.zeros([BLOCK_M], dtype=tl.int32); i5 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i6 = tl.zeros([BLOCK_M], dtype=tl.int32); i7 = tl.zeros([BLOCK_M], dtype=tl.int32)
    for ref_idx in range(N):
        r0 = tl.load(ref_base + ref_idx * stride_r_n + 0 * stride_r_d).to(tl.float32)
        r1 = tl.load(ref_base + ref_idx * stride_r_n + 1 * stride_r_d).to(tl.float32)
        r2 = tl.load(ref_base + ref_idx * stride_r_n + 2 * stride_r_d).to(tl.float32)
        diff0 = q0 - r0; diff1 = q1 - r1; diff2 = q2 - r2
        dist = diff0*diff0 + diff1*diff1 + diff2*diff2
        u0 = dist < d0; u1 = dist < d1; u2 = dist < d2; u3 = dist < d3
        u4 = dist < d4; u5 = dist < d5; u6 = dist < d6; u7 = dist < d7
        # Simplified 16-op insertion (down from 30+)
        d7 = tl.where(u6, d6, tl.where(u7, dist, d7)); i7 = tl.where(u6, i6, tl.where(u7, ref_idx, i7))
        d6 = tl.where(u5, d5, tl.where(u6, dist, d6)); i6 = tl.where(u5, i5, tl.where(u6, ref_idx, i6))
        d5 = tl.where(u4, d4, tl.where(u5, dist, d5)); i5 = tl.where(u4, i4, tl.where(u5, ref_idx, i5))
        d4 = tl.where(u3, d3, tl.where(u4, dist, d4)); i4 = tl.where(u3, i3, tl.where(u4, ref_idx, i4))
        d3 = tl.where(u2, d2, tl.where(u3, dist, d3)); i3 = tl.where(u2, i2, tl.where(u3, ref_idx, i3))
        d2 = tl.where(u1, d1, tl.where(u2, dist, d2)); i2 = tl.where(u1, i1, tl.where(u2, ref_idx, i2))
        d1 = tl.where(u0, d0, tl.where(u1, dist, d1)); i1 = tl.where(u0, i0, tl.where(u1, ref_idx, i1))
        d0 = tl.where(u0, dist, d0); i0 = tl.where(u0, ref_idx, i0)
    out_base = OUT_IDX + pid_b * stride_o_b
    tl.store(out_base + offs_m * stride_o_m + 0 * stride_o_k, i0, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 1 * stride_o_k, i1, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 2 * stride_o_k, i2, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 3 * stride_o_k, i3, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 4 * stride_o_k, i4, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 5 * stride_o_k, i5, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 6 * stride_o_k, i6, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 7 * stride_o_k, i7, mask=mask_m)


@triton.jit
def _knn_kernel_simple_k8_d4_noautotune(
    QUERY, REF, OUT_IDX, B, M, N,
    stride_q_b, stride_q_m, stride_q_d,
    stride_r_b, stride_r_n, stride_r_d,
    stride_o_b, stride_o_m, stride_o_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Non-autotuned K=8 D=4 kernel for heuristic mode."""
    INF: tl.constexpr = 1e38
    pid_m, pid_b = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    query_base = QUERY + pid_b * stride_q_b
    ref_base = REF + pid_b * stride_r_b
    q0 = tl.load(query_base + offs_m * stride_q_m + 0 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q1 = tl.load(query_base + offs_m * stride_q_m + 1 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q2 = tl.load(query_base + offs_m * stride_q_m + 2 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q3 = tl.load(query_base + offs_m * stride_q_m + 3 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    d0 = tl.full([BLOCK_M], INF, dtype=tl.float32); d1 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d2 = tl.full([BLOCK_M], INF, dtype=tl.float32); d3 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d4 = tl.full([BLOCK_M], INF, dtype=tl.float32); d5 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d6 = tl.full([BLOCK_M], INF, dtype=tl.float32); d7 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    i0 = tl.zeros([BLOCK_M], dtype=tl.int32); i1 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i2 = tl.zeros([BLOCK_M], dtype=tl.int32); i3 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i4 = tl.zeros([BLOCK_M], dtype=tl.int32); i5 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i6 = tl.zeros([BLOCK_M], dtype=tl.int32); i7 = tl.zeros([BLOCK_M], dtype=tl.int32)
    for ref_idx in range(N):
        r0 = tl.load(ref_base + ref_idx * stride_r_n + 0 * stride_r_d).to(tl.float32)
        r1 = tl.load(ref_base + ref_idx * stride_r_n + 1 * stride_r_d).to(tl.float32)
        r2 = tl.load(ref_base + ref_idx * stride_r_n + 2 * stride_r_d).to(tl.float32)
        r3 = tl.load(ref_base + ref_idx * stride_r_n + 3 * stride_r_d).to(tl.float32)
        diff0 = q0 - r0; diff1 = q1 - r1; diff2 = q2 - r2; diff3 = q3 - r3
        dist = diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
        u0 = dist < d0; u1 = dist < d1; u2 = dist < d2; u3 = dist < d3
        u4 = dist < d4; u5 = dist < d5; u6 = dist < d6; u7 = dist < d7
        # Simplified 16-op insertion (down from 30+)
        d7 = tl.where(u6, d6, tl.where(u7, dist, d7)); i7 = tl.where(u6, i6, tl.where(u7, ref_idx, i7))
        d6 = tl.where(u5, d5, tl.where(u6, dist, d6)); i6 = tl.where(u5, i5, tl.where(u6, ref_idx, i6))
        d5 = tl.where(u4, d4, tl.where(u5, dist, d5)); i5 = tl.where(u4, i4, tl.where(u5, ref_idx, i5))
        d4 = tl.where(u3, d3, tl.where(u4, dist, d4)); i4 = tl.where(u3, i3, tl.where(u4, ref_idx, i4))
        d3 = tl.where(u2, d2, tl.where(u3, dist, d3)); i3 = tl.where(u2, i2, tl.where(u3, ref_idx, i3))
        d2 = tl.where(u1, d1, tl.where(u2, dist, d2)); i2 = tl.where(u1, i1, tl.where(u2, ref_idx, i2))
        d1 = tl.where(u0, d0, tl.where(u1, dist, d1)); i1 = tl.where(u0, i0, tl.where(u1, ref_idx, i1))
        d0 = tl.where(u0, dist, d0); i0 = tl.where(u0, ref_idx, i0)
    out_base = OUT_IDX + pid_b * stride_o_b
    tl.store(out_base + offs_m * stride_o_m + 0 * stride_o_k, i0, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 1 * stride_o_k, i1, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 2 * stride_o_k, i2, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 3 * stride_o_k, i3, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 4 * stride_o_k, i4, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 5 * stride_o_k, i5, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 6 * stride_o_k, i6, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 7 * stride_o_k, i7, mask=mask_m)


# =============================================================================
# Autotuned Kernels (used when use_heuristic=False)
# =============================================================================

@triton.autotune(configs=configs_knn_simple, key=['M', 'N'])
@triton.jit
def _knn_kernel_simple_k4_d2(
    QUERY, REF, OUT_IDX, B, M, N,
    stride_q_b, stride_q_m, stride_q_d,
    stride_r_b, stride_r_n, stride_r_d,
    stride_o_b, stride_o_m, stride_o_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Autotuned K=4 D=2 kernel."""
    INF: tl.constexpr = 1e38
    pid_m, pid_b = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    query_base = QUERY + pid_b * stride_q_b
    ref_base = REF + pid_b * stride_r_b
    q0 = tl.load(query_base + offs_m * stride_q_m + 0 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q1 = tl.load(query_base + offs_m * stride_q_m + 1 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    d0 = tl.full([BLOCK_M], INF, dtype=tl.float32); d1 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d2 = tl.full([BLOCK_M], INF, dtype=tl.float32); d3 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    i0 = tl.zeros([BLOCK_M], dtype=tl.int32); i1 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i2 = tl.zeros([BLOCK_M], dtype=tl.int32); i3 = tl.zeros([BLOCK_M], dtype=tl.int32)
    for ref_idx in range(N):
        r0 = tl.load(ref_base + ref_idx * stride_r_n + 0 * stride_r_d).to(tl.float32)
        r1 = tl.load(ref_base + ref_idx * stride_r_n + 1 * stride_r_d).to(tl.float32)
        diff0 = q0 - r0; diff1 = q1 - r1
        dist = diff0*diff0 + diff1*diff1
        u0 = dist < d0; u1 = dist < d1; u2 = dist < d2; u3 = dist < d3
        # Simplified 8-op insertion (down from 14)
        d3 = tl.where(u2, d2, tl.where(u3, dist, d3)); i3 = tl.where(u2, i2, tl.where(u3, ref_idx, i3))
        d2 = tl.where(u1, d1, tl.where(u2, dist, d2)); i2 = tl.where(u1, i1, tl.where(u2, ref_idx, i2))
        d1 = tl.where(u0, d0, tl.where(u1, dist, d1)); i1 = tl.where(u0, i0, tl.where(u1, ref_idx, i1))
        d0 = tl.where(u0, dist, d0); i0 = tl.where(u0, ref_idx, i0)
    out_base = OUT_IDX + pid_b * stride_o_b
    tl.store(out_base + offs_m * stride_o_m + 0 * stride_o_k, i0, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 1 * stride_o_k, i1, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 2 * stride_o_k, i2, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 3 * stride_o_k, i3, mask=mask_m)


@triton.autotune(
    configs=configs_knn_simple,
    key=['M', 'N'],
)
@triton.jit
def _knn_kernel_simple_k4_d3(
    QUERY, REF, OUT_IDX,
    B, M, N,
    stride_q_b, stride_q_m, stride_q_d,
    stride_r_b, stride_r_n, stride_r_d,
    stride_o_b, stride_o_m, stride_o_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Simplified KNN kernel for D=3, K=4."""
    INF: tl.constexpr = 1e38
    
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    
    query_base = QUERY + pid_b * stride_q_b
    ref_base = REF + pid_b * stride_r_b
    
    q0 = tl.load(query_base + offs_m * stride_q_m + 0 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q1 = tl.load(query_base + offs_m * stride_q_m + 1 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q2 = tl.load(query_base + offs_m * stride_q_m + 2 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    
    d0 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d1 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d2 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d3 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    i0 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i1 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i2 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i3 = tl.zeros([BLOCK_M], dtype=tl.int32)
    
    for ref_idx in range(N):
        r0 = tl.load(ref_base + ref_idx * stride_r_n + 0 * stride_r_d).to(tl.float32)
        r1 = tl.load(ref_base + ref_idx * stride_r_n + 1 * stride_r_d).to(tl.float32)
        r2 = tl.load(ref_base + ref_idx * stride_r_n + 2 * stride_r_d).to(tl.float32)
        
        diff0 = q0 - r0
        diff1 = q1 - r1
        diff2 = q2 - r2
        dist = diff0 * diff0 + diff1 * diff1 + diff2 * diff2
        
        update3 = dist < d3
        update2 = dist < d2
        update1 = dist < d1
        update0 = dist < d0
        
        new_d3 = tl.where(update3, tl.where(update2, tl.where(update1, tl.where(update0, d2, d2), d2), d2), d3)
        new_i3 = tl.where(update3, tl.where(update2, tl.where(update1, tl.where(update0, i2, i2), i2), i2), i3)
        new_d2 = tl.where(update2, tl.where(update1, tl.where(update0, d1, d1), d1), d2)
        new_i2 = tl.where(update2, tl.where(update1, tl.where(update0, i1, i1), i1), i2)
        new_d1 = tl.where(update1, tl.where(update0, d0, d0), d1)
        new_i1 = tl.where(update1, tl.where(update0, i0, i0), i1)
        new_d0 = tl.where(update0, dist, d0)
        new_i0 = tl.where(update0, ref_idx, i0)
        
        new_d3 = tl.where(update3 & ~update2, dist, new_d3)
        new_i3 = tl.where(update3 & ~update2, ref_idx, new_i3)
        new_d2 = tl.where(update2 & ~update1, dist, new_d2)
        new_i2 = tl.where(update2 & ~update1, ref_idx, new_i2)
        new_d1 = tl.where(update1 & ~update0, dist, new_d1)
        new_i1 = tl.where(update1 & ~update0, ref_idx, new_i1)
        
        d0, d1, d2, d3 = new_d0, new_d1, new_d2, new_d3
        i0, i1, i2, i3 = new_i0, new_i1, new_i2, new_i3
    
    out_base = OUT_IDX + pid_b * stride_o_b
    tl.store(out_base + offs_m * stride_o_m + 0 * stride_o_k, i0, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 1 * stride_o_k, i1, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 2 * stride_o_k, i2, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 3 * stride_o_k, i3, mask=mask_m)


@triton.autotune(
    configs=configs_knn_simple,
    key=['M', 'N'],
)
@triton.jit
def _knn_kernel_simple_k4_d4(
    QUERY, REF, OUT_IDX,
    B, M, N,
    stride_q_b, stride_q_m, stride_q_d,
    stride_r_b, stride_r_n, stride_r_d,
    stride_o_b, stride_o_m, stride_o_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Simplified KNN kernel for D=4, K=4."""
    INF: tl.constexpr = 1e38
    
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    
    query_base = QUERY + pid_b * stride_q_b
    ref_base = REF + pid_b * stride_r_b
    
    q0 = tl.load(query_base + offs_m * stride_q_m + 0 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q1 = tl.load(query_base + offs_m * stride_q_m + 1 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q2 = tl.load(query_base + offs_m * stride_q_m + 2 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q3 = tl.load(query_base + offs_m * stride_q_m + 3 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    
    d0 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d1 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d2 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d3 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    i0 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i1 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i2 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i3 = tl.zeros([BLOCK_M], dtype=tl.int32)
    
    for ref_idx in range(N):
        r0 = tl.load(ref_base + ref_idx * stride_r_n + 0 * stride_r_d).to(tl.float32)
        r1 = tl.load(ref_base + ref_idx * stride_r_n + 1 * stride_r_d).to(tl.float32)
        r2 = tl.load(ref_base + ref_idx * stride_r_n + 2 * stride_r_d).to(tl.float32)
        r3 = tl.load(ref_base + ref_idx * stride_r_n + 3 * stride_r_d).to(tl.float32)
        
        diff0 = q0 - r0
        diff1 = q1 - r1
        diff2 = q2 - r2
        diff3 = q3 - r3
        dist = diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3
        
        update3 = dist < d3
        update2 = dist < d2
        update1 = dist < d1
        update0 = dist < d0
        
        new_d3 = tl.where(update3, tl.where(update2, tl.where(update1, tl.where(update0, d2, d2), d2), d2), d3)
        new_i3 = tl.where(update3, tl.where(update2, tl.where(update1, tl.where(update0, i2, i2), i2), i2), i3)
        new_d2 = tl.where(update2, tl.where(update1, tl.where(update0, d1, d1), d1), d2)
        new_i2 = tl.where(update2, tl.where(update1, tl.where(update0, i1, i1), i1), i2)
        new_d1 = tl.where(update1, tl.where(update0, d0, d0), d1)
        new_i1 = tl.where(update1, tl.where(update0, i0, i0), i1)
        new_d0 = tl.where(update0, dist, d0)
        new_i0 = tl.where(update0, ref_idx, i0)
        
        new_d3 = tl.where(update3 & ~update2, dist, new_d3)
        new_i3 = tl.where(update3 & ~update2, ref_idx, new_i3)
        new_d2 = tl.where(update2 & ~update1, dist, new_d2)
        new_i2 = tl.where(update2 & ~update1, ref_idx, new_i2)
        new_d1 = tl.where(update1 & ~update0, dist, new_d1)
        new_i1 = tl.where(update1 & ~update0, ref_idx, new_i1)
        
        d0, d1, d2, d3 = new_d0, new_d1, new_d2, new_d3
        i0, i1, i2, i3 = new_i0, new_i1, new_i2, new_i3
    
    out_base = OUT_IDX + pid_b * stride_o_b
    tl.store(out_base + offs_m * stride_o_m + 0 * stride_o_k, i0, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 1 * stride_o_k, i1, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 2 * stride_o_k, i2, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 3 * stride_o_k, i3, mask=mask_m)


# =============================================================================
# Simplified K=8 Kernels (Scalar-Register Top-K)
# =============================================================================

@triton.autotune(
    configs=configs_knn_simple,
    key=['M', 'N'],
)
@triton.jit
def _knn_kernel_simple_k8_d2(
    QUERY, REF, OUT_IDX,
    B, M, N,
    stride_q_b, stride_q_m, stride_q_d,
    stride_r_b, stride_r_n, stride_r_d,
    stride_o_b, stride_o_m, stride_o_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Simplified KNN kernel for D=2, K=8 using streaming scalar top-K."""
    INF: tl.constexpr = 1e38
    
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    
    query_base = QUERY + pid_b * stride_q_b
    ref_base = REF + pid_b * stride_r_b
    
    q0 = tl.load(query_base + offs_m * stride_q_m + 0 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q1 = tl.load(query_base + offs_m * stride_q_m + 1 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    
    # Initialize top-8 as separate scalar arrays [BLOCK_M] each (sorted ascending)
    d0 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d1 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d2 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d3 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d4 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d5 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d6 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d7 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    i0 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i1 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i2 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i3 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i4 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i5 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i6 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i7 = tl.zeros([BLOCK_M], dtype=tl.int32)
    
    for ref_idx in range(N):
        r0 = tl.load(ref_base + ref_idx * stride_r_n + 0 * stride_r_d).to(tl.float32)
        r1 = tl.load(ref_base + ref_idx * stride_r_n + 1 * stride_r_d).to(tl.float32)
        
        diff0 = q0 - r0
        diff1 = q1 - r1
        dist = diff0 * diff0 + diff1 * diff1
        
        # Simplified 16-op insertion (down from 30+)
        u0 = dist < d0; u1 = dist < d1; u2 = dist < d2; u3 = dist < d3
        u4 = dist < d4; u5 = dist < d5; u6 = dist < d6; u7 = dist < d7
        d7 = tl.where(u6, d6, tl.where(u7, dist, d7)); i7 = tl.where(u6, i6, tl.where(u7, ref_idx, i7))
        d6 = tl.where(u5, d5, tl.where(u6, dist, d6)); i6 = tl.where(u5, i5, tl.where(u6, ref_idx, i6))
        d5 = tl.where(u4, d4, tl.where(u5, dist, d5)); i5 = tl.where(u4, i4, tl.where(u5, ref_idx, i5))
        d4 = tl.where(u3, d3, tl.where(u4, dist, d4)); i4 = tl.where(u3, i3, tl.where(u4, ref_idx, i4))
        d3 = tl.where(u2, d2, tl.where(u3, dist, d3)); i3 = tl.where(u2, i2, tl.where(u3, ref_idx, i3))
        d2 = tl.where(u1, d1, tl.where(u2, dist, d2)); i2 = tl.where(u1, i1, tl.where(u2, ref_idx, i2))
        d1 = tl.where(u0, d0, tl.where(u1, dist, d1)); i1 = tl.where(u0, i0, tl.where(u1, ref_idx, i1))
        d0 = tl.where(u0, dist, d0); i0 = tl.where(u0, ref_idx, i0)
    
    out_base = OUT_IDX + pid_b * stride_o_b
    tl.store(out_base + offs_m * stride_o_m + 0 * stride_o_k, i0, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 1 * stride_o_k, i1, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 2 * stride_o_k, i2, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 3 * stride_o_k, i3, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 4 * stride_o_k, i4, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 5 * stride_o_k, i5, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 6 * stride_o_k, i6, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 7 * stride_o_k, i7, mask=mask_m)


@triton.autotune(
    configs=configs_knn_simple,
    key=['M', 'N'],
)
@triton.jit
def _knn_kernel_simple_k8_d3(
    QUERY, REF, OUT_IDX,
    B, M, N,
    stride_q_b, stride_q_m, stride_q_d,
    stride_r_b, stride_r_n, stride_r_d,
    stride_o_b, stride_o_m, stride_o_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Simplified KNN kernel for D=3, K=8."""
    INF: tl.constexpr = 1e38
    
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    
    query_base = QUERY + pid_b * stride_q_b
    ref_base = REF + pid_b * stride_r_b
    
    q0 = tl.load(query_base + offs_m * stride_q_m + 0 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q1 = tl.load(query_base + offs_m * stride_q_m + 1 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q2 = tl.load(query_base + offs_m * stride_q_m + 2 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    
    d0 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d1 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d2 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d3 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d4 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d5 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d6 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d7 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    i0 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i1 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i2 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i3 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i4 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i5 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i6 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i7 = tl.zeros([BLOCK_M], dtype=tl.int32)
    
    for ref_idx in range(N):
        r0 = tl.load(ref_base + ref_idx * stride_r_n + 0 * stride_r_d).to(tl.float32)
        r1 = tl.load(ref_base + ref_idx * stride_r_n + 1 * stride_r_d).to(tl.float32)
        r2 = tl.load(ref_base + ref_idx * stride_r_n + 2 * stride_r_d).to(tl.float32)
        
        diff0 = q0 - r0
        diff1 = q1 - r1
        diff2 = q2 - r2
        dist = diff0 * diff0 + diff1 * diff1 + diff2 * diff2
        
        # Simplified 16-op insertion
        u0 = dist < d0; u1 = dist < d1; u2 = dist < d2; u3 = dist < d3
        u4 = dist < d4; u5 = dist < d5; u6 = dist < d6; u7 = dist < d7
        d7 = tl.where(u6, d6, tl.where(u7, dist, d7)); i7 = tl.where(u6, i6, tl.where(u7, ref_idx, i7))
        d6 = tl.where(u5, d5, tl.where(u6, dist, d6)); i6 = tl.where(u5, i5, tl.where(u6, ref_idx, i6))
        d5 = tl.where(u4, d4, tl.where(u5, dist, d5)); i5 = tl.where(u4, i4, tl.where(u5, ref_idx, i5))
        d4 = tl.where(u3, d3, tl.where(u4, dist, d4)); i4 = tl.where(u3, i3, tl.where(u4, ref_idx, i4))
        d3 = tl.where(u2, d2, tl.where(u3, dist, d3)); i3 = tl.where(u2, i2, tl.where(u3, ref_idx, i3))
        d2 = tl.where(u1, d1, tl.where(u2, dist, d2)); i2 = tl.where(u1, i1, tl.where(u2, ref_idx, i2))
        d1 = tl.where(u0, d0, tl.where(u1, dist, d1)); i1 = tl.where(u0, i0, tl.where(u1, ref_idx, i1))
        d0 = tl.where(u0, dist, d0); i0 = tl.where(u0, ref_idx, i0)
    
    out_base = OUT_IDX + pid_b * stride_o_b
    tl.store(out_base + offs_m * stride_o_m + 0 * stride_o_k, i0, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 1 * stride_o_k, i1, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 2 * stride_o_k, i2, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 3 * stride_o_k, i3, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 4 * stride_o_k, i4, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 5 * stride_o_k, i5, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 6 * stride_o_k, i6, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 7 * stride_o_k, i7, mask=mask_m)


@triton.autotune(
    configs=configs_knn_simple,
    key=['M', 'N'],
)
@triton.jit
def _knn_kernel_simple_k8_d4(
    QUERY, REF, OUT_IDX,
    B, M, N,
    stride_q_b, stride_q_m, stride_q_d,
    stride_r_b, stride_r_n, stride_r_d,
    stride_o_b, stride_o_m, stride_o_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Simplified KNN kernel for D=4, K=8."""
    INF: tl.constexpr = 1e38
    
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    
    query_base = QUERY + pid_b * stride_q_b
    ref_base = REF + pid_b * stride_r_b
    
    q0 = tl.load(query_base + offs_m * stride_q_m + 0 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q1 = tl.load(query_base + offs_m * stride_q_m + 1 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q2 = tl.load(query_base + offs_m * stride_q_m + 2 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q3 = tl.load(query_base + offs_m * stride_q_m + 3 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    
    d0 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d1 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d2 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d3 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d4 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d5 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d6 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    d7 = tl.full([BLOCK_M], INF, dtype=tl.float32)
    i0 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i1 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i2 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i3 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i4 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i5 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i6 = tl.zeros([BLOCK_M], dtype=tl.int32)
    i7 = tl.zeros([BLOCK_M], dtype=tl.int32)
    
    for ref_idx in range(N):
        r0 = tl.load(ref_base + ref_idx * stride_r_n + 0 * stride_r_d).to(tl.float32)
        r1 = tl.load(ref_base + ref_idx * stride_r_n + 1 * stride_r_d).to(tl.float32)
        r2 = tl.load(ref_base + ref_idx * stride_r_n + 2 * stride_r_d).to(tl.float32)
        r3 = tl.load(ref_base + ref_idx * stride_r_n + 3 * stride_r_d).to(tl.float32)
        
        diff0 = q0 - r0
        diff1 = q1 - r1
        diff2 = q2 - r2
        diff3 = q3 - r3
        dist = diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3
        
        # Simplified 16-op insertion
        u0 = dist < d0; u1 = dist < d1; u2 = dist < d2; u3 = dist < d3
        u4 = dist < d4; u5 = dist < d5; u6 = dist < d6; u7 = dist < d7
        d7 = tl.where(u6, d6, tl.where(u7, dist, d7)); i7 = tl.where(u6, i6, tl.where(u7, ref_idx, i7))
        d6 = tl.where(u5, d5, tl.where(u6, dist, d6)); i6 = tl.where(u5, i5, tl.where(u6, ref_idx, i6))
        d5 = tl.where(u4, d4, tl.where(u5, dist, d5)); i5 = tl.where(u4, i4, tl.where(u5, ref_idx, i5))
        d4 = tl.where(u3, d3, tl.where(u4, dist, d4)); i4 = tl.where(u3, i3, tl.where(u4, ref_idx, i4))
        d3 = tl.where(u2, d2, tl.where(u3, dist, d3)); i3 = tl.where(u2, i2, tl.where(u3, ref_idx, i3))
        d2 = tl.where(u1, d1, tl.where(u2, dist, d2)); i2 = tl.where(u1, i1, tl.where(u2, ref_idx, i2))
        d1 = tl.where(u0, d0, tl.where(u1, dist, d1)); i1 = tl.where(u0, i0, tl.where(u1, ref_idx, i1))
        d0 = tl.where(u0, dist, d0); i0 = tl.where(u0, ref_idx, i0)
    
    out_base = OUT_IDX + pid_b * stride_o_b
    tl.store(out_base + offs_m * stride_o_m + 0 * stride_o_k, i0, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 1 * stride_o_k, i1, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 2 * stride_o_k, i2, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 3 * stride_o_k, i3, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 4 * stride_o_k, i4, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 5 * stride_o_k, i5, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 6 * stride_o_k, i6, mask=mask_m)
    tl.store(out_base + offs_m * stride_o_m + 7 * stride_o_k, i7, mask=mask_m)


# =============================================================================
# Legacy Streaming KNN Kernel (D=2) - Fallback for other K values
# =============================================================================

# Autotune configurations - optimized for memory coalescing
configs_knn = [
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
    # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_stages=2, num_warps=4),
    # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
]


@triton.autotune(
    configs=configs_knn,
    key=['M', 'N', 'K'],
)
@triton.jit
def _knn_kernel_d2(
    # Input tensors
    QUERY,      # [B, M, D] - query points
    REF,        # [B, N, D] - reference points
    # Output tensor
    OUT_IDX,    # [B, M, K] - output indices
    # Dimensions
    B, M, N,
    # Strides
    stride_q_b, stride_q_m, stride_q_d,
    stride_r_b, stride_r_n, stride_r_d,
    stride_o_b, stride_o_m, stride_o_k,
    # Compile-time constants
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Streaming KNN kernel for D=2 with tiled memory access.
    Loads BLOCK_N refs at a time for better memory coalescing.
    """
    pid_m = tl.program_id(0)  # Query tile
    pid_b = tl.program_id(1)  # Batch
    
    # Query offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    
    # Base pointers for this batch
    query_base = QUERY + pid_b * stride_q_b
    ref_base = REF + pid_b * stride_r_b
    
    # Load query coordinates [BLOCK_M] - these stay in registers
    q0 = tl.load(query_base + offs_m * stride_q_m + 0 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q1 = tl.load(query_base + offs_m * stride_q_m + 1 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    
    # Initialize top-K state in registers
    INF = 1e38
    top_k_dist = tl.full([BLOCK_M, K], INF, dtype=tl.float32)
    top_k_idx = tl.zeros([BLOCK_M, K], dtype=tl.int32)
    
    # Stream through reference points in TILES of BLOCK_N
    # This gives coalesced memory access
    for tile_start in range(0, N, BLOCK_N):
        # Reference offsets for this tile
        offs_n = tile_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        # Load BLOCK_N reference coordinates at once (coalesced!) [BLOCK_N]
        r0 = tl.load(ref_base + offs_n * stride_r_n + 0 * stride_r_d, mask=mask_n, other=0.0).to(tl.float32)
        r1 = tl.load(ref_base + offs_n * stride_r_n + 1 * stride_r_d, mask=mask_n, other=0.0).to(tl.float32)
        
        # Compute pairwise squared distances [BLOCK_M, BLOCK_N]
        # Broadcasting: q[BLOCK_M, 1] - r[1, BLOCK_N]
        d0 = q0[:, None] - r0[None, :]  # [BLOCK_M, BLOCK_N]
        d1 = q1[:, None] - r1[None, :]  # [BLOCK_M, BLOCK_N]
        dist_sq_tile = d0 * d0 + d1 * d1  # [BLOCK_M, BLOCK_N]
        
        # Mask invalid entries
        valid_mask = mask_m[:, None] & mask_n[None, :]
        dist_sq_tile = tl.where(valid_mask, dist_sq_tile, INF)
        
        # Process each ref in the tile for top-K update
        # Use Python range (not static_range) to avoid code explosion
        for n in range(BLOCK_N):
            ref_idx = tile_start + n
            if ref_idx < N:  # Bounds check
                # Extract column n from distance tile
                # Using masked reduction to get the column
                col_mask = tl.arange(0, BLOCK_N) == n
                dist_col = tl.sum(tl.where(col_mask[None, :], dist_sq_tile, 0.0), axis=1)
                col_valid = mask_m & (ref_idx < N)
                
                # Update top-K
                top_k_dist, top_k_idx = update_topk_single(
                    top_k_dist, top_k_idx, dist_col, ref_idx, col_valid, K
                )
    
    # Store output indices [BLOCK_M, K]
    out_base = OUT_IDX + pid_b * stride_o_b
    offs_k = tl.arange(0, K)
    
    out_ptrs = out_base + offs_m[:, None] * stride_o_m + offs_k[None, :] * stride_o_k
    tl.store(out_ptrs, top_k_idx, mask=mask_m[:, None])


@triton.autotune(
    configs=configs_knn,
    key=['M', 'N', 'K'],
)
@triton.jit
def _knn_kernel_d3(
    # Input tensors
    QUERY,      # [B, M, D] - query points
    REF,        # [B, N, D] - reference points
    # Output tensor
    OUT_IDX,    # [B, M, K] - output indices
    # Dimensions
    B, M, N,
    # Strides
    stride_q_b, stride_q_m, stride_q_d,
    stride_r_b, stride_r_n, stride_r_d,
    stride_o_b, stride_o_m, stride_o_k,
    # Compile-time constants
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Streaming KNN kernel for D=3 with tiled memory access."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    
    query_base = QUERY + pid_b * stride_q_b
    ref_base = REF + pid_b * stride_r_b
    
    # Load query coords [BLOCK_M]
    q0 = tl.load(query_base + offs_m * stride_q_m + 0 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q1 = tl.load(query_base + offs_m * stride_q_m + 1 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q2 = tl.load(query_base + offs_m * stride_q_m + 2 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    
    INF = 1e38
    top_k_dist = tl.full([BLOCK_M, K], INF, dtype=tl.float32)
    top_k_idx = tl.zeros([BLOCK_M, K], dtype=tl.int32)
    
    # Stream through refs in tiles
    for tile_start in range(0, N, BLOCK_N):
        offs_n = tile_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        # Load BLOCK_N ref coords (coalesced)
        r0 = tl.load(ref_base + offs_n * stride_r_n + 0 * stride_r_d, mask=mask_n, other=0.0).to(tl.float32)
        r1 = tl.load(ref_base + offs_n * stride_r_n + 1 * stride_r_d, mask=mask_n, other=0.0).to(tl.float32)
        r2 = tl.load(ref_base + offs_n * stride_r_n + 2 * stride_r_d, mask=mask_n, other=0.0).to(tl.float32)
        
        # Pairwise distances [BLOCK_M, BLOCK_N]
        d0 = q0[:, None] - r0[None, :]
        d1 = q1[:, None] - r1[None, :]
        d2 = q2[:, None] - r2[None, :]
        dist_sq_tile = d0 * d0 + d1 * d1 + d2 * d2
        
        valid_mask = mask_m[:, None] & mask_n[None, :]
        dist_sq_tile = tl.where(valid_mask, dist_sq_tile, INF)
        
        for n in range(BLOCK_N):
            ref_idx = tile_start + n
            if ref_idx < N:
                col_mask = tl.arange(0, BLOCK_N) == n
                dist_col = tl.sum(tl.where(col_mask[None, :], dist_sq_tile, 0.0), axis=1)
                col_valid = mask_m & (ref_idx < N)
                top_k_dist, top_k_idx = update_topk_single(
                    top_k_dist, top_k_idx, dist_col, ref_idx, col_valid, K
                )
    
    out_base = OUT_IDX + pid_b * stride_o_b
    offs_k = tl.arange(0, K)
    out_ptrs = out_base + offs_m[:, None] * stride_o_m + offs_k[None, :] * stride_o_k
    tl.store(out_ptrs, top_k_idx, mask=mask_m[:, None])


@triton.autotune(
    configs=configs_knn,
    key=['M', 'N', 'K'],
)
@triton.jit
def _knn_kernel_d4(
    # Input tensors
    QUERY,      # [B, M, D] - query points
    REF,        # [B, N, D] - reference points
    # Output tensor
    OUT_IDX,    # [B, M, K] - output indices
    # Dimensions
    B, M, N,
    # Strides
    stride_q_b, stride_q_m, stride_q_d,
    stride_r_b, stride_r_n, stride_r_d,
    stride_o_b, stride_o_m, stride_o_k,
    # Compile-time constants
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Streaming KNN kernel for D=4 with tiled memory access."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    
    query_base = QUERY + pid_b * stride_q_b
    ref_base = REF + pid_b * stride_r_b
    
    # Load query coords [BLOCK_M]
    q0 = tl.load(query_base + offs_m * stride_q_m + 0 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q1 = tl.load(query_base + offs_m * stride_q_m + 1 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q2 = tl.load(query_base + offs_m * stride_q_m + 2 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    q3 = tl.load(query_base + offs_m * stride_q_m + 3 * stride_q_d, mask=mask_m, other=0.0).to(tl.float32)
    
    INF = 1e38
    top_k_dist = tl.full([BLOCK_M, K], INF, dtype=tl.float32)
    top_k_idx = tl.zeros([BLOCK_M, K], dtype=tl.int32)
    
    # Stream through refs in tiles
    for tile_start in range(0, N, BLOCK_N):
        offs_n = tile_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        # Load BLOCK_N ref coords (coalesced)
        r0 = tl.load(ref_base + offs_n * stride_r_n + 0 * stride_r_d, mask=mask_n, other=0.0).to(tl.float32)
        r1 = tl.load(ref_base + offs_n * stride_r_n + 1 * stride_r_d, mask=mask_n, other=0.0).to(tl.float32)
        r2 = tl.load(ref_base + offs_n * stride_r_n + 2 * stride_r_d, mask=mask_n, other=0.0).to(tl.float32)
        r3 = tl.load(ref_base + offs_n * stride_r_n + 3 * stride_r_d, mask=mask_n, other=0.0).to(tl.float32)
        
        # Pairwise distances [BLOCK_M, BLOCK_N]
        d0 = q0[:, None] - r0[None, :]
        d1 = q1[:, None] - r1[None, :]
        d2 = q2[:, None] - r2[None, :]
        d3 = q3[:, None] - r3[None, :]
        dist_sq_tile = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3
        
        valid_mask = mask_m[:, None] & mask_n[None, :]
        dist_sq_tile = tl.where(valid_mask, dist_sq_tile, INF)
        
        for n in range(BLOCK_N):
            ref_idx = tile_start + n
            if ref_idx < N:
                col_mask = tl.arange(0, BLOCK_N) == n
                dist_col = tl.sum(tl.where(col_mask[None, :], dist_sq_tile, 0.0), axis=1)
                col_valid = mask_m & (ref_idx < N)
                top_k_dist, top_k_idx = update_topk_single(
                    top_k_dist, top_k_idx, dist_col, ref_idx, col_valid, K
                )
    
    out_base = OUT_IDX + pid_b * stride_o_b
    offs_k = tl.arange(0, K)
    out_ptrs = out_base + offs_m[:, None] * stride_o_m + offs_k[None, :] * stride_o_k
    tl.store(out_ptrs, top_k_idx, mask=mask_m[:, None])


# =============================================================================
# Python Interface
# =============================================================================

def _next_power_of_2(x):
    """Return the smallest power of 2 >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


# =============================================================================
# Large K Fallback Functions (for K > 8)
# =============================================================================

# Cache for PyKeOps availability check
_KEOPS_AVAILABLE = None


def knn_pytorch_large(query, ref, k):
    """
    KNN using PyTorch cdist + topk (fallback when PyKeOps unavailable).
    
    Warning: This materializes O(B*M*N) distance matrix in memory.
    
    Args:
        query: [B, M, D] - query points
        ref: [B, N, D] - reference points
        k: int - number of nearest neighbors
        
    Returns:
        indices: [B, M, K] - indices of K nearest neighbors (int64)
    """
    # Compute pairwise squared distances: [B, M, N]
    dist = torch.cdist(query.float(), ref.float(), p=2.0).pow(2)
    
    # Get top-k smallest distances
    _, indices = torch.topk(dist, k, dim=-1, largest=False)
    
    return indices.to(torch.int64)


def knn_large_k_fallback(query, ref, k):
    """
    Dispatch KNN for large K values to PyKeOps or PyTorch.
    
    Uses PyKeOps if available and inputs are on CUDA, otherwise falls back to PyTorch.
    
    Args:
        query: [B, M, D] - query points
        ref: [B, N, D] - reference points
        k: int - number of nearest neighbors
        
    Returns:
        indices: [B, M, K] - indices of K nearest neighbors (int64)
    """
    global _KEOPS_AVAILABLE
    
    # Check PyKeOps availability (cached)
    if _KEOPS_AVAILABLE is None:
        try:
            from pykeops.torch import LazyTensor
            _KEOPS_AVAILABLE = True
        except ImportError:
            _KEOPS_AVAILABLE = False
    
    # Use PyKeOps if available and on CUDA
    if _KEOPS_AVAILABLE and query.is_cuda:
        # Use the verified knn_keops from geometry.py
        from src.utils.geometry import knn_keops
        return knn_keops(query, ref, k)
    else:
        return knn_pytorch_large(query, ref, k)


def knn_forward(query, ref, k, D=2, use_v2=True, use_heuristic=True):
    """
    Forward pass for streaming KNN.
    
    Args:
        query: [B, M, D] - query points (FP16 or FP32)
        ref: [B, N, D] - reference points (FP16 or FP32)
        k: int - number of nearest neighbors
        D: int - spatial dimension (2, 3, or 4)
        use_v2: bool - if True and K=4/8, use simplified scalar-register kernels
        use_heuristic: bool - if True, use pre-computed config heuristics (faster startup)
                              if False, use Triton autotuning (slower startup, potentially better perf)
        
    Returns:
        indices: [B, M, K] - indices of K nearest neighbors (int32 for K<=8, int64 for K>8)
    """
    assert query.is_cuda and ref.is_cuda, "Inputs must be on CUDA"
    assert query.dim() == 3 and ref.dim() == 3, "Expected 3D tensors [B, N, D]"
    assert query.shape[0] == ref.shape[0], "Batch sizes must match"
    assert query.shape[2] == ref.shape[2] == D, f"Expected D={D}, got query D={query.shape[2]}, ref D={ref.shape[2]}"
    assert D in [2, 3, 4], f"D must be 2, 3, or 4, got {D}"
    assert k <= ref.shape[1], f"k={k} exceeds number of reference points N={ref.shape[1]}"
    
    # For K > 8, use PyKeOps or PyTorch fallback (Triton kernel only supports K <= 8)
    if k > 8:
        return knn_large_k_fallback(query, ref, k).to(torch.int32)
    
    # Make contiguous
    query = query.contiguous()
    ref = ref.contiguous()
    
    B, M, _ = query.shape
    _, N, _ = ref.shape
    K = k
    
    # Determine effective K for kernel
    # For K <= 4: use K=4 kernel, slice result
    # For 4 < K <= 8: use K=8 kernel, slice result
    if K <= 4:
        K_KERNEL = 4
    else:  # 4 < K <= 8 (asserted above)
        K_KERNEL = 8
    
    need_slice = (K_KERNEL != K)
    
    # Output tensor (may be padded)
    out_idx = torch.zeros((B, M, K_KERNEL), device=query.device, dtype=torch.int32)
    
    # Use simplified scalar-register kernels for K_KERNEL=4 or K_KERNEL=8
    if use_v2:
        if use_heuristic:
            # Use heuristic-based config selection (no autotuning overhead)
            BLOCK_M, BLOCK_N, num_warps, num_stages = get_knn_heuristic_config(M, N, K_KERNEL, D)
            grid = (triton.cdiv(M, BLOCK_M), B)
            
            # Select non-autotuned kernel based on K_KERNEL and D
            if K_KERNEL == 4:
                if D == 2:
                    _knn_kernel_simple_k4_d2_noautotune[grid](
                        query, ref, out_idx, B, M, N,
                        query.stride(0), query.stride(1), query.stride(2),
                        ref.stride(0), ref.stride(1), ref.stride(2),
                        out_idx.stride(0), out_idx.stride(1), out_idx.stride(2),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                        num_warps=num_warps, num_stages=num_stages,
                    )
                elif D == 3:
                    _knn_kernel_simple_k4_d3_noautotune[grid](
                        query, ref, out_idx, B, M, N,
                        query.stride(0), query.stride(1), query.stride(2),
                        ref.stride(0), ref.stride(1), ref.stride(2),
                        out_idx.stride(0), out_idx.stride(1), out_idx.stride(2),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                        num_warps=num_warps, num_stages=num_stages,
                    )
                elif D == 4:
                    _knn_kernel_simple_k4_d4_noautotune[grid](
                        query, ref, out_idx, B, M, N,
                        query.stride(0), query.stride(1), query.stride(2),
                        ref.stride(0), ref.stride(1), ref.stride(2),
                        out_idx.stride(0), out_idx.stride(1), out_idx.stride(2),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                        num_warps=num_warps, num_stages=num_stages,
                    )
            else:  # K_KERNEL == 8
                if D == 2:
                    _knn_kernel_simple_k8_d2_noautotune[grid](
                        query, ref, out_idx, B, M, N,
                        query.stride(0), query.stride(1), query.stride(2),
                        ref.stride(0), ref.stride(1), ref.stride(2),
                        out_idx.stride(0), out_idx.stride(1), out_idx.stride(2),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                        num_warps=num_warps, num_stages=num_stages,
                    )
                elif D == 3:
                    _knn_kernel_simple_k8_d3_noautotune[grid](
                        query, ref, out_idx, B, M, N,
                        query.stride(0), query.stride(1), query.stride(2),
                        ref.stride(0), ref.stride(1), ref.stride(2),
                        out_idx.stride(0), out_idx.stride(1), out_idx.stride(2),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                        num_warps=num_warps, num_stages=num_stages,
                    )
                elif D == 4:
                    _knn_kernel_simple_k8_d4_noautotune[grid](
                        query, ref, out_idx, B, M, N,
                        query.stride(0), query.stride(1), query.stride(2),
                        ref.stride(0), ref.stride(1), ref.stride(2),
                        out_idx.stride(0), out_idx.stride(1), out_idx.stride(2),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                        num_warps=num_warps, num_stages=num_stages,
                    )
        else:
            # Use autotuned kernels (slower startup, but may find better config)
            def grid(META):
                return (triton.cdiv(M, META['BLOCK_M']), B)
            
            if K_KERNEL == 4:
                if D == 2:
                    _knn_kernel_simple_k4_d2[grid](
                        query, ref, out_idx, B, M, N,
                        query.stride(0), query.stride(1), query.stride(2),
                        ref.stride(0), ref.stride(1), ref.stride(2),
                        out_idx.stride(0), out_idx.stride(1), out_idx.stride(2),
                    )
                elif D == 3:
                    _knn_kernel_simple_k4_d3[grid](
                        query, ref, out_idx, B, M, N,
                        query.stride(0), query.stride(1), query.stride(2),
                        ref.stride(0), ref.stride(1), ref.stride(2),
                        out_idx.stride(0), out_idx.stride(1), out_idx.stride(2),
                    )
                elif D == 4:
                    _knn_kernel_simple_k4_d4[grid](
                        query, ref, out_idx, B, M, N,
                        query.stride(0), query.stride(1), query.stride(2),
                        ref.stride(0), ref.stride(1), ref.stride(2),
                        out_idx.stride(0), out_idx.stride(1), out_idx.stride(2),
                    )
            else:  # K_KERNEL == 8
                if D == 2:
                    _knn_kernel_simple_k8_d2[grid](
                        query, ref, out_idx, B, M, N,
                        query.stride(0), query.stride(1), query.stride(2),
                        ref.stride(0), ref.stride(1), ref.stride(2),
                        out_idx.stride(0), out_idx.stride(1), out_idx.stride(2),
                    )
                elif D == 3:
                    _knn_kernel_simple_k8_d3[grid](
                        query, ref, out_idx, B, M, N,
                        query.stride(0), query.stride(1), query.stride(2),
                        ref.stride(0), ref.stride(1), ref.stride(2),
                        out_idx.stride(0), out_idx.stride(1), out_idx.stride(2),
                    )
                elif D == 4:
                    _knn_kernel_simple_k8_d4[grid](
                        query, ref, out_idx, B, M, N,
                        query.stride(0), query.stride(1), query.stride(2),
                        ref.stride(0), ref.stride(1), ref.stride(2),
                        out_idx.stride(0), out_idx.stride(1), out_idx.stride(2),
                    )
    else:
        # Legacy path disabled - use_v2=False is not recommended
        raise ValueError(
            f"use_v2=False is deprecated. Only optimized K=4/K=8 kernels are supported. "
            f"Got K={K}, use_v2={use_v2}"
        )
    
    # Slice to requested K if we used a padded kernel
    if need_slice:
        out_idx = out_idx[:, :, :K]
    
    return out_idx


class KNNFunction(Function):
    """
    Autograd Function for KNN (forward-only, no gradients).
    KNN indices are non-differentiable.
    """
    
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, query, ref, k, D=2):
        """
        Args:
            query: [B, M, D] - query points
            ref: [B, N, D] - reference points
            k: int - number of nearest neighbors
            D: int - spatial dimension (2, 3, or 4)
            
        Returns:
            indices: [B, M, K] - indices of K nearest neighbors (int64 for PyTorch compatibility)
        """
        # Detach from graph - KNN indices have no gradient
        with torch.no_grad():
            query_detached = query.detach()
            ref_detached = ref.detach()
            
            # Convert to FP32 for computation if needed
            # (kernel computes in FP32 internally anyway)
            if query_detached.dtype == torch.float16:
                query_detached = query_detached.to(torch.float32)
            if ref_detached.dtype == torch.float16:
                ref_detached = ref_detached.to(torch.float32)
            
            indices = knn_forward(query_detached, ref_detached, k, D)
            
            # Convert to int64 for compatibility with gather operations
            return indices.to(torch.int64)
    
    @staticmethod
    def backward(ctx, grad_output):
        # KNN has no gradient
        return None, None, None, None


def knn_triton(query, ref, k, D=2):
    """
    Compute K-nearest neighbors using streaming Triton kernel.
    
    This is a drop-in replacement for knn_keops() that:
    - Supports FP16 input natively
    - Uses O(M*K) memory instead of O(M*N)
    - Keeps intermediate state in registers (KeOps-style)
    
    Args:
        query: [B, M, D] - query points in D-dimensional space
        ref: [B, N, D] - reference points in D-dimensional space
        k: int - number of nearest neighbors to find
        D: int - spatial dimension (2, 3, or 4), default 2
        
    Returns:
        indices: [B, M, K] - indices of K nearest neighbors for each query point
                 (int64 tensor, compatible with torch.gather)
        
    Example:
        >>> query = torch.randn(2, 1000, 2, device='cuda')
        >>> ref = torch.randn(2, 5000, 2, device='cuda')
        >>> indices = knn_triton(query, ref, k=4, D=2)
        >>> indices.shape
        torch.Size([2, 1000, 4])
    """
    return KNNFunction.apply(query, ref, k, D)


# Convenience alias matching knn_keops signature
def knn(query, database, k, return_dist=False, D=None):
    """
    Compute K-nearest neighbors (compatible with knn_keops signature).
    
    Args:
        query: [B, M, D] - query points
        database: [B, N, D] - reference/database points
        k: int - number of nearest neighbors
        return_dist: bool - if True, also return distances
        D: int - spatial dimension (auto-detected if None)
        
    Returns:
        indices: [B, M, K] - indices of K nearest neighbors
        distances: [B, M, K] - squared distances (only if return_dist=True)
    """
    if D is None:
        D = query.shape[-1]
    
    indices = knn_triton(query, database, k, D=D)
    
    # HACKY/NOT PERFORMANT FOR NOW, BUT WHATEVER WILL REMOVE IT LATER
    if return_dist:
        # Compute distances: gather neighbor positions and compute squared L2 distance
        B, M, K = indices.shape
        # Gather neighbor positions: [B, M, K, D]
        neighbor_pos = database.gather(
            dim=1,
            index=indices.view(B, M * K, 1).expand(-1, -1, D)
        ).view(B, M, K, D)
        # Compute squared distances: [B, M, K]
        distances = ((query.unsqueeze(2) - neighbor_pos) ** 2).sum(dim=-1)
        return indices, distances
    
    return indices
