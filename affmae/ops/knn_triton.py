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
from .dispatch import custom_fwd  # torch-version shim
# Imported through util so a torch-only install can still import this
# module; the kernels raise only when actually launched.
from .dispatch import clamp_num_stages, triton, tl


# online k-min: process single candidate and update top-K
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
# Simplified Scalar-Register Top-K Kernels
# =============================================================================
#
# These kernels use a simple streaming approach with scalar variables for top-K:
# - Load reference tiles for coalesced memory access
# - Compute distances for all queries to current ref tile
# - Update top-K using simple conditional insertion (like KeOps)
#
# This is much simpler than the bitonic merge approach and compiles faster.


# Autotune configurations (used when use_heuristic=False)
configs_knn_simple = (
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=clamp_num_stages(2), num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=clamp_num_stages(2), num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=clamp_num_stages(2), num_warps=4),
)


@triton.autotune(configs=list(configs_knn_simple), key=['M', 'N'])
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
    configs=list(configs_knn_simple),
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


# =============================================================================
# Large K Fallback Functions (for K > 8)
# =============================================================================



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

    Uses PyKeOps on a supported CPU/CUDA runtime and PyTorch otherwise.

    Args:
        query: [B, M, D] - query points
        ref: [B, N, D] - reference points
        k: int - number of nearest neighbors

    Returns:
        indices: [B, M, K] - indices of K nearest neighbors (int64)
    """
    from .dispatch import can_use_keops

    # can_use_keops, not is_cuda: KeOps has no ROCm and no MPS backend, and a
    # ROCm tensor reports is_cuda True -- so is_cuda would route AMD into a
    # backend that does not exist. knn_pytorch_large is exact and runs anywhere.
    if can_use_keops(query, ref):
        from affmae.ops.knn_keops import knn_keops

        return knn_keops(query, ref, k)
    return knn_pytorch_large(query, ref, k)


def knn_forward(query, ref, k, D=2, use_v2=True, use_heuristic=True):
    """
    Forward pass for streaming KNN.

    Args:
        query: [B, M, D] - query points (FP16 or FP32)
        ref: [B, N, D] - reference points (FP16 or FP32)
        k: int - number of nearest neighbors
        D: int - spatial dimension (currently only 2 is supported)
    Returns:
        indices: [B, M, K] - indices of K nearest neighbors (int32)
    """
    assert query.dim() == 3 and ref.dim() == 3, "Expected 3D tensors [B, N, D]"
    assert query.shape[0] == ref.shape[0], "Batch sizes must match"
    assert query.shape[2] == ref.shape[2] == D, f"Expected D={D}, got query D={query.shape[2]}, ref D={ref.shape[2]}"
    assert D == 2, "D must be 2"
    assert k <= ref.shape[1], f"k={k} exceeds number of reference points N={ref.shape[1]}"

    # Off the Triton path: no accelerator, or k above what the kernel handles.
    # knn_pytorch_large is exact and device-agnostic, so this is a speed
    # trade-off, not an accuracy one. Note the CUDA assert used to sit *above*
    # this dispatch, so the fallback was unreachable on CPU even though it
    # existed.
    from .dispatch import can_use_triton

    if k > 8 or not can_use_triton(query, ref):
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
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']), B)

    if K_KERNEL == 4:
        _knn_kernel_simple_k4_d2[grid](
            query, ref, out_idx, B, M, N,
            query.stride(0), query.stride(1), query.stride(2),
            ref.stride(0), ref.stride(1), ref.stride(2),
            out_idx.stride(0), out_idx.stride(1), out_idx.stride(2),
        )
    elif K_KERNEL == 8:
        _knn_kernel_simple_k8_d2[grid](
            query, ref, out_idx, B, M, N,
            query.stride(0), query.stride(1), query.stride(2),
            ref.stride(0), ref.stride(1), ref.stride(2),
            out_idx.stride(0), out_idx.stride(1), out_idx.stride(2),
        )
    else:
        raise ValueError(f"Invalid K_KERNEL: {K_KERNEL}")

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
            D: int - spatial dimension (currently only 2 is supported)

        Returns:
            indices: [B, M, K] - indices of K nearest neighbors (int32)
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
