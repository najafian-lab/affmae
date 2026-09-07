"""
Test data utilities and pytest fixtures for cluster attention modules.

This module provides utilities for creating test data, comparing implementations,
and running comprehensive tests for cluster attention modules.
"""

import math
import pytest
import torch
import torch.nn as nn
from typing import Dict
import warnings

# pykeops is a hard dependency, but it compiles on first use, so a machine with
# no C++ toolchain can still lack it; knn_keops() below falls back
# when it is missing so the suite runs on a bare install.
try:
    from pykeops.torch import LazyTensor
    HAS_KEOPS = True
except ImportError:  # pragma: no cover - environment dependent
    LazyTensor = None
    HAS_KEOPS = False

# CLUSTEN is an optional compiled extension. Tests that diff the Triton kernels
# against the reference CUDA kernels skip when it has not been built; see
# affmae/ops/cuda_ext/README.md.
try:
    from affmae.ops.cuda_ext.clusten import CLUSTENQKFunction  # noqa: F401
    HAS_CLUSTEN = True
except Exception:  # pragma: no cover - environment dependent
    HAS_CLUSTEN = False

# Clustering helpers come from the production module, not a copy. conftest used
# to carry its own space_filling_cluster / peano / hilbert implementations, and
# they had drifted from src (peano was 16 lines here vs 74 there), so fixtures
# were clustering tokens differently from training.
from affmae.utils.geometry import space_filling_cluster  # noqa: E402

# Relative-position table geometry MUST come from production. conftest used to
# hardcode `rel_pos_width = 2048 // 4 - 1` (a 1023-wide table, 1,046,529 rows) —
# the commented-out variant in affmae/utils/pos_embed.py. The live table is
# 255-wide (65,025 rows), so fixture pe_idx values indexed ~16x past the end of
# the buffer ClusterAttention actually holds, tripping a device-side assert that
# poisoned the CUDA context for the rest of the run.
from affmae.layers.pos_embed import rel_pos_width as PROD_REL_POS_WIDTH  # noqa: E402
from affmae.layers.pos_embed import table_width as PROD_TABLE_WIDTH  # noqa: E402


def knn_keops(query, database, k, return_dist=False):
    """KNN via PyKeOps when available, else an exact torch fallback.

    Test fixtures only need correct neighbours, not KeOps' memory behaviour.
    """
    if HAS_KEOPS:
        from affmae.utils.geometry import knn_keops as _keops_knn
        return _keops_knn(query, database, k, return_dist=return_dist)
    return knn_torch_fallback(query, database, k, return_dist=return_dist)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device")
requires_clusten = pytest.mark.skipif(
    not HAS_CLUSTEN,
    reason="CLUSTEN CUDA extension not built; see affmae/ops/cuda_ext/README.md")
requires_keops = pytest.mark.skipif(
    not HAS_KEOPS, reason="pykeops is not importable (needs a C++ toolchain)")

# Suppress pydantic warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
# Suppress torch.meshgrid warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")
# Suppress timm warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")


def compare_tensor(a: torch.Tensor, b: torch.Tensor, tol: float = 6e-3, name: str = "tensor") -> None:
    """
    Compare two tensors with detailed diff reporting.
    
    Args:
        a: First tensor
        b: Second tensor  
        tol: Tolerance for comparison
        name: Name for error reporting
        
    Raises:
        AssertionError: If tensors differ beyond tolerance
    """
    if a.shape != b.shape:
        raise AssertionError(f"{name} shapes differ: {a.shape} vs {b.shape}")
    
    # Compute differences
    diff = torch.abs(a - b)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"{name} comparison:")
    print(f"  Max |diff|: {max_diff:.2e}")
    print(f"  Mean |diff|: {mean_diff:.2e}")
    print(f"  Tolerance: {tol:.2e}")
    print(f"  dtype: {a.dtype},{b.dtype}")
    print(f"  shape: {a.shape},{b.shape}")
    print(f"  a-max: {a.max().item()}, b-max: {b.max().item()}")
    print(f"  a-min: {a.min().item()}, b-min: {b.min().item()}")
    print(f"  a-mean: {a.mean().item()}, b-mean: {b.mean().item()}")
    
    # bump up tol by 100x for float16 (or within 5e-2)
    if a.dtype == torch.float16 or b.dtype == torch.float16:
        tol = min(tol * 100, 5e-2)

    if max_diff > tol:
        raise AssertionError(
            f"{name} differs beyond tolerance. Max |diff|: {max_diff:.2e}, "
            f"Mean |diff|: {mean_diff:.2e}, Tolerance: {tol:.2e}"
        )


def validate_test_parameters(B: int, N: int, C: int, H: int, M: int) -> None:
    """
    Validate test parameters to ensure they meet the required constraints.
    
    Args:
        B: Batch size (must be >= 1)
        N: Sequence length (must be >= 32)
        C: Hidden dimension (must be >= H and C % H == 0)
        H: Number of attention heads
        M: Neighborhood size (must be < N)
        
    Raises:
        ValueError: If any constraint is violated
    """
    if B < 1:
        raise ValueError(f"Batch size B must be >= 1, got B={B}")
    if N < 32:
        raise ValueError(f"Sequence length N must be >= 32, got N={N}")
    if C < H:
        raise ValueError(f"Hidden dimension C must be >= H, got C={C}, H={H}")
    if C % H != 0:
        raise ValueError(f"C must be divisible by H, got C={C}, H={H}, C%H={C%H}")
    if M >= N:
        raise ValueError(f"Neighborhood size M must be < N, got M={M}, N={N}")


def create_test_data(
    B: int, N: int, C: int, H: int, M: int, 
    device: str = "cuda", dtype: torch.dtype = torch.float32,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    Create comprehensive test data for cluster attention testing.
    
    Args:
        B: Batch size
        N: Sequence length (number of tokens)
        C: Hidden dimension
        H: Number of attention heads
        M: Neighborhood size
        device: Device to create tensors on
        dtype: Data type for tensors
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing all test data tensors
    """
    # Validate parameters before proceeding
    validate_test_parameters(B, N, C, H, M)
    
    # torch.manual_seed(seed)
    
    # Position table geometry, taken from production so pe_idx stays in range
    # for the pre_table buffer ClusterAttention registers.
    rel_pos_width = PROD_REL_POS_WIDTH
    table_width = PROD_TABLE_WIDTH
    
    pre_hs = torch.arange(table_width).float() - rel_pos_width
    pre_ws = torch.arange(table_width).float() - rel_pos_width
    pre_ys, pre_xs = torch.meshgrid(pre_hs, pre_ws, indexing="ij")
    dis_table = (pre_ys**2 + pre_xs**2).sqrt()
    sin_table = torch.nan_to_num(pre_ys / dis_table)
    cos_table = torch.nan_to_num(pre_xs / dis_table)
    pre_table = torch.stack([pre_xs, pre_ys, dis_table, sin_table, cos_table], dim=2).reshape(-1, 5).to(device)
    
    # Input features
    feat = torch.randn(B, N, C, device=device, dtype=dtype, requires_grad=True)
    
    # Create a fake grid for spatial clustering
    h, w = 512, 512
    hs = torch.arange(0, h, device=device)
    ws = torch.arange(0, w, device=device)
    ys, xs = torch.meshgrid(hs, ws, indexing="ij")
    pos_init = torch.stack([xs, ys], dim=2).unsqueeze(0).expand(B, -1, -1, -1).reshape(B, -1, 2).to(dtype)
    
    # Sample N points from the grid
    pos = torch.zeros((B, N, 2), device=device, dtype=dtype)
    for b in range(B):
        sampled_indices = torch.randperm(h*w, device=device)[:N]
        sampled_pos = pos_init[b].reshape(h*w, 2)[sampled_indices].to(dtype)
        pos[b] = sampled_pos.reshape(N, 2)
    
    # Create clustering data
    cluster_size = 8
    nbhd_size = M
    k = int(math.ceil(N / float(cluster_size)))  # number of clusters
    nnc = min(int(round(nbhd_size / float(cluster_size))), k)  # number of nearest clusters
    nbhd_size = cluster_size * nnc
    
    if k == N:
        cluster_mean_pos = pos
        member_idx = torch.arange(N, device=device).long().reshape(1, N, 1).expand(B, -1, -1)
        cluster_mask = None
    else:
        pos, cluster_mean_pos, member_idx, cluster_mask, reorder = space_filling_cluster(
            pos, cluster_size, h, w, no_reorder=False
        )
        feat = feat[torch.arange(B).to(feat.device).repeat_interleave(N), reorder.view(-1)].reshape(B, N, C)
    
    # Find nearest clusters
    nearest_cluster = knn_keops(pos, cluster_mean_pos, nnc)  # b x n x nnc
    
    # Gather member indices for nearest clusters
    # member_idx has shape [B, k, cluster_size], nearest_cluster has shape [B, N, nnc]
    member_idx = member_idx.gather(
        index=nearest_cluster.view(B, -1, 1).expand(-1, -1, cluster_size), dim=1
    ).reshape(B, N, nbhd_size)
    
    if cluster_mask is not None:
        cluster_mask = cluster_mask.gather(
            index=nearest_cluster.view(B, -1, 1).expand(-1, -1, cluster_size), dim=1
        ).reshape(B, N, nbhd_size)
    
    # Compute position embedding indices
    pos_ = pos.gather(
        index=member_idx.view(B, -1, 1).expand(-1, -1, 2), dim=1
    ).reshape(B, N, nbhd_size, 2)
    rel_pos = pos_ - (pos.unsqueeze(2) - rel_pos_width)
    rel_pos = rel_pos.clamp(0, table_width - 1)
    pe_idx = (rel_pos[..., 1] * table_width + rel_pos[..., 0]).long()
    
    return {
        'feat': feat,
        'pos_xy': pos,
        'member_idx': member_idx,
        'cluster_mask': cluster_mask,
        'pre_table': pre_table,
        'pe_idx': pe_idx,
        'table_width': table_width,
        'rel_pos_width': rel_pos_width
    }


def knn_torch_fallback(query, database, k, return_dist=False):
    """Fallback KNN implementation using PyTorch."""
    b, n, c = database.shape
    
    with torch.no_grad():
        # Compute pairwise distances
        query_expanded = query.unsqueeze(2)  # [b, n_, 1, c]
        database_expanded = database.unsqueeze(1)  # [b, 1, n, c]
        dist = torch.norm(query_expanded - database_expanded, dim=-1)  # [b, n_, n]
        
        # Find k nearest neighbors
        if return_dist:
            nn_dist, nn_idx = torch.topk(dist, k, dim=-1, largest=False)
            return nn_idx, nn_dist
        else:
            nn_idx = torch.topk(dist, k, dim=-1, largest=False)[1]
            return nn_idx


TEST_CONFIGS = {
    'small': {
        'B': 1, 'N': 128, 'C': 32, 'H': 1, 'M': 64,
        'name': 'small',
        'description': 'Small configuration: B=1, N=64, C=32, H=1, M=32'
    },
    'medium': {
        'B': 16, 'N': 256, 'C': 64, 'H': 2, 'M': 64,
        'name': 'medium',
        'description': 'Medium configuration: B=16, N=128, C=64, H=2, M=48'
    },
    'large': {
        'B': 128, 'N': 1024, 'C': 256, 'H': 8, 'M': 64,
        'name': 'large',
        'description': 'Large configuration: B=128, N=1024, C=192, H=6, M=48'
    }
}


@pytest.fixture(params=list(TEST_CONFIGS.keys()))
def test_config(request):
    """Fixture providing test configurations."""
    return TEST_CONFIGS[request.param]


@pytest.fixture
def device():
    """Fixture providing the device for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def dtype():
    """Fixture providing the data type for tests."""
    return torch.float32


def copy_weights(old_module: nn.Module, new_module: nn.Module):
    """Copy weights from old module to new module for fair comparison."""
    with torch.no_grad():
        # Copy Q projection weights
        new_module.q.weight.copy_(old_module.q.weight)
        new_module.q.bias.copy_(old_module.q.bias)
        
        # Copy KV projection weights
        new_module.kv.weight.copy_(old_module.kv.weight)
        new_module.kv.bias.copy_(old_module.kv.bias)
        
        # Copy output projection weights
        new_module.proj.weight.copy_(old_module.proj.weight)
        new_module.proj.bias.copy_(old_module.proj.bias)
        
        # Copy position embedding weights
        new_module.pos_embed.weight.copy_(old_module.pos_embed.weight)
        new_module.pos_embed.bias.copy_(old_module.pos_embed.bias)
        
        # Copy blank token weights
        new_module.blank_k.copy_(old_module.blank_k)
        new_module.blank_v.copy_(old_module.blank_v)


@pytest.fixture
def tolerance():
    """Fixture providing test tolerances."""
    return {
        'forward_pass': 1e-3,
        'backward_pass': 2e-3,  # Increased slightly for gradient comparisons
        'loss': 1e-4
    }


def make_cluster_attention_pair(dim: int, num_heads: int, device: str,
                                proj_drop: float = 0.0,
                                reference_backend: str = "cuda"):
    """Build two ClusterAttention modules that differ only in kernel backend.

    Before the ``affmae/`` refactor these were two separate classes
    (``cluster.archive.aff`` vs ``cluster.cluster_attn``). They are now one
    class parameterized by ``backend``, so a fair comparison is two instances
    with identical weights.

    Args:
        dim: int, embedding width.
        num_heads: int, attention heads.
        device: str, device to place both modules on.
        proj_drop: float, output projection dropout.
        reference_backend: str, backend for the reference module. "cuda" uses
            the compiled CLUSTEN kernels; guard such tests with
            ``requires_clusten``.
    Returns:
        (reference_module, triton_module), both in eval mode with identical
        parameters.
    """
    from affmae.layers.attention import ClusterAttention

    reference = ClusterAttention(
        dim=dim, num_heads=num_heads, proj_drop=proj_drop,
        backend=reference_backend).to(device).eval()
    triton = ClusterAttention(
        dim=dim, num_heads=num_heads, proj_drop=proj_drop,
        backend="flash_nbhd_attn").to(device).eval()

    # Identical architectures, so a plain state-dict copy keeps them in sync.
    triton.load_state_dict(reference.state_dict())
    return reference, triton
