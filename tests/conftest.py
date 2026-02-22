"""
Test data utilities and pytest fixtures for cluster attention modules.

This module provides utilities for creating test data, comparing implementations,
and running comprehensive tests for cluster attention modules.
"""

import math
import pytest
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
import pykeops
from pykeops.torch import LazyTensor
import warnings

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
    
    # Position table setup (matching the original implementation)
    rel_pos_width = 2048 // 4 - 1
    table_width = 2 * rel_pos_width + 1
    
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


def knn_keops(query, database, k, return_dist=False):
    """
    Compute k-nearest neighbors using the Keops library.
    
    Args:
        query: b x n_ x c, the position of tokens looking for knn
        database: b x n x c, the candidate tokens for knn
        k: int, the number of neighbors to be found
        return_dist: bool, whether to return distance to the neighbors
        
    Returns:
        nn_idx: b x n x k, the indices of the knn
        nn_dist: b x n x k, if return_dist, the distance to the knn
    """
    b, n, c = database.shape
    
    # Disable amp/not supported for knn keops computation
    with torch.amp.autocast(device_type='cuda', enabled=False):
        with torch.no_grad():
            # Detach from graph
            query = query.detach()
            database = database.detach()
            
            # Keops does not support half precision
            if query.dtype != torch.float32:
                query = query.to(torch.float32)
            if database.dtype != torch.float32:
                database = database.to(torch.float32)
            
            n_ = query.shape[1]
            query_ = LazyTensor(query[:, None, :, :])
            database_ = LazyTensor(database[:, :, None, :])
            dist = ((query_ - database_) ** 2).sum(-1) ** 0.5  # b x n x n_
        
        try:
            if return_dist:
                nn_dist, nn_idx = dist.Kmin_argKmin(k, dim=1)  # b x n_ x k
                return nn_idx, nn_dist
            else:
                nn_idx = dist.argKmin(k, dim=1)  # b x n_ x k
                return nn_idx
        except Exception as e:
            # Fallback to PyTorch implementation if KeOps fails
            print(f"KeOps failed, falling back to PyTorch: {e}")
            return knn_torch_fallback(query, database, k, return_dist)


def knn_torch_fallback(query, database, k, return_dist=False):
    """Fallback KNN implementation using PyTorch."""
    b, n, c = database.shape
    n_ = query.shape[1]
    
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


def space_filling_cluster(pos, m, h, w, no_reorder=False, sf_type='', use_anchor=True):
    """
    The balanced clustering algorithm based on space-filling curves.
    
    Args:
        pos: b x n x 2, positions of tokens
        m: int, target size of the clusters
        h, w: int, height and width
        no_reorder: bool, if True, return the clustering based on the original order of tokens
        sf_type: str, can be 'peano' or 'hilbert', or otherwise, horizontal scanlines
        use_anchor: bool, whether to use space-filling anchors or not
        
    Returns:
        pos: b x n x 2, returned only if no_reorder is False
        cluster_mean_pos: b x k x 2, the clustering centers
        member_idx: b x k x m, the indices of tokens in each cluster
        cluster_mask: b x k x m, the binary mask indicating the paddings in last cluster
        pos_ranking: b x n x 1, returned only if no_reorder is False
    """
    with torch.no_grad():
        pos = pos.detach()
        
        if pos.dtype != torch.float:
            pos = pos.to(torch.float)
        b, n, d = pos.shape
        if not isinstance(b, int):
            b, n, d = b.item(), n.item(), d.item()
        
        k = int(math.ceil(n / m))
        
        if use_anchor:
            patch_len = (h * w / k) ** 0.5
            num_patch_h = int(round(h / patch_len))
            num_patch_w = int(round(w / patch_len))
            patch_len_h, patch_len_w = h / num_patch_h, w / num_patch_w
            
            if sf_type == 'peano':
                num_patch_h = max(3, int(3 ** round(math.log(num_patch_h, 3))))
                patch_len_h = h / num_patch_h
                num_patch_w = int(round(w / h * 3) * (num_patch_h / 3))
                patch_len_w = w / num_patch_w
            elif sf_type == 'hilbert':
                num_patch_h = max(2, int(2 ** round(math.log(num_patch_h, 2))))
                patch_len_h = h / num_patch_h
                num_patch_w = int(round(w / h * 2) * (num_patch_h / 2))
                patch_len_w = w / num_patch_w
            
            hs = torch.arange(0, num_patch_h, device=pos.device)
            ws = torch.arange(0, num_patch_w, device=pos.device)
            ys, xs = torch.meshgrid(hs, ws, indexing="ij")
            grid_pos = torch.stack([xs, ys], dim=2)  # h x w x 2
            grid_pos = grid_pos.reshape(-1, 2)
            
            # Sort the grid centers to one line
            if sf_type == 'peano':
                order_grid_idx, order_idx = calculate_peano_order(num_patch_h, num_patch_w, grid_pos.unsqueeze(0))
                order_grid_idx = order_grid_idx[0]
                order_idx = order_idx[0]
            elif sf_type == 'hilbert':
                order_grid_idx, order_idx = calculate_hilbert_order(num_patch_h, num_patch_w, grid_pos.unsqueeze(0))
                order_grid_idx = order_grid_idx[0]
                order_idx = order_idx[0]
            else:
                order_mask = torch.ones_like(ys)  # h x w
                order_mask[1::2] = -1
                order_mask = order_mask * xs
                order_mask = order_mask + ys * w
                order_mask[1::2] += (w - 1)
                order_mask = order_mask.reshape(-1)
                order_idx = order_mask.sort()[1]
                order_idx_src = torch.arange(len(order_idx)).to(pos.device)
                order_grid_idx = torch.zeros_like(order_idx_src)
                order_grid_idx.scatter_(index=order_idx, dim=0, src=order_idx_src)
            
            ordered_grid = grid_pos[order_idx]
            patch_len_hw = torch.Tensor([patch_len_w, patch_len_h]).to(pos.device)
            
            init_pos_means = ordered_grid * patch_len_hw + patch_len_hw / 2 - 0.5
            nump = ordered_grid.shape[0]
            
            prev_means = torch.zeros_like(init_pos_means)
            prev_means[1:] = init_pos_means[:nump - 1].clone()
            prev_means[0] = prev_means[1] - (prev_means[2] - prev_means[1])
            next_means = torch.zeros_like(init_pos_means)
            next_means[:nump - 1] = init_pos_means[1:].clone()
            next_means[-1] = next_means[-2] + (next_means[-2] - next_means[-3])
            
            mean_assignment = (pos / patch_len_hw).floor()
            mean_assignment = mean_assignment[..., 0] + mean_assignment[..., 1] * num_patch_w
            mean_assignment = order_grid_idx.unsqueeze(0).expand(b, -1).gather(
                index=mean_assignment.long(), dim=1
            ).unsqueeze(2)  # b x n x 1
            
            prev_mean_assign = prev_means.unsqueeze(0).expand(b, -1, -1).gather(
                index=mean_assignment.expand(-1, -1, d), dim=1
            )  # b x n x d
            next_mean_assign = next_means.unsqueeze(0).expand(b, -1, -1).gather(
                index=mean_assignment.expand(-1, -1, d), dim=1
            )  # b x n x d
            dist_prev = (pos - prev_mean_assign).pow(2).sum(-1)  # b x n
            dist_next = (pos - next_mean_assign).pow(2).sum(-1)
            dist_ratio = dist_prev / (dist_next + 1e-5)
            
            pos_ranking = mean_assignment * (dist_ratio.max() + 1) + dist_ratio.unsqueeze(2)
            pos_ranking = pos_ranking.sort(dim=1)[1]  # b x n x 1
        
        else:
            if sf_type == 'peano':
                _, pos_ranking = calculate_peano_order(h, w, pos)
            elif sf_type == 'hilbert':
                _, pos_ranking = calculate_hilbert_order(h, w, pos)
            else:
                hs = torch.arange(0, h, device=pos.device)
                ws = torch.arange(0, w, device=pos.device)
                ys, xs = torch.meshgrid(hs, ws, indexing="ij")
                order_mask = torch.ones_like(ys)  # h x w
                order_mask[1::2] = -1
                order_mask = order_mask * xs
                order_mask = order_mask + ys * w
                order_mask[1::2] += (w - 1)
                order_mask = order_mask.reshape(-1)
                pos_idx = pos[..., 0] + pos[..., 1] * w
                order_mask = order_mask.gather(index=pos_idx.long().reshape(-1), dim=0).reshape(b, n)
                pos_ranking = order_mask.sort()[1]
            pos_ranking = pos_ranking.unsqueeze(2)
        
        pos = pos.gather(index=pos_ranking.expand(-1, -1, d), dim=1)  # b x n x d
        
        if k * m == n:
            cluster_mask = None
            cluster_mean_pos = pos.reshape(b, k, -1, d).mean(2)
        else:
            pos_pad = torch.zeros(b, k * m, d, dtype=pos.dtype, device=pos.device)
            pos_pad[:, :n] = pos.clone()
            cluster_mask = torch.zeros(b, k * m, device=pos.device).long()
            cluster_mask[:, :n] = 1
            cluster_mask = cluster_mask.reshape(b, k, m)
            cluster_mean_pos = pos_pad.reshape(b, k, -1, d).sum(2) / cluster_mask.sum(2, keepdim=True)
        
        if no_reorder:
            if k * m == n:
                member_idx = pos_ranking.reshape(b, k, m)
            else:
                member_idx = torch.zeros(b, k * m, device=pos.device, dtype=torch.int64)
                member_idx[:, :n] = pos_ranking.squeeze(2)
                member_idx = member_idx.reshape(b, k, m)
            return cluster_mean_pos, member_idx, cluster_mask
        else:
            member_idx = torch.arange(k * m, device=pos.device)
            member_idx[n:] = 0
            member_idx = member_idx.unsqueeze(0).expand(b, -1)  # b x k*m
            member_idx = member_idx.reshape(b, k, m)
            
            return pos, cluster_mean_pos, member_idx, cluster_mask, pos_ranking


def calculate_peano_order(h, w, pos):
    """Calculate Peano curve ordering."""
    # Simplified implementation - in practice you'd want a full Peano curve implementation
    order_mask = torch.ones(h, w, device=pos.device)
    order_mask[1::2] = -1
    order_mask = order_mask * torch.arange(w, device=pos.device)
    order_mask = order_mask + torch.arange(h, device=pos.device).unsqueeze(1) * w
    order_mask[1::2] += (w - 1)
    order_mask = order_mask.reshape(-1)
    order_idx = order_mask.sort()[1]
    order_idx_src = torch.arange(len(order_idx), device=pos.device)
    order_grid_idx = torch.zeros_like(order_idx_src)
    order_grid_idx.scatter_(index=order_idx, dim=0, src=order_idx_src)
    return order_grid_idx, order_idx


def calculate_hilbert_order(h, w, pos):
    """Calculate Hilbert curve ordering."""
    # Simplified implementation - in practice you'd want a full Hilbert curve implementation
    order_mask = torch.ones(h, w, device=pos.device)
    order_mask[1::2] = -1
    order_mask = order_mask * torch.arange(w, device=pos.device)
    order_mask = order_mask + torch.arange(h, device=pos.device).unsqueeze(1) * w
    order_mask[1::2] += (w - 1)
    order_mask = order_mask.reshape(-1)
    order_idx = order_mask.sort()[1]
    order_idx_src = torch.arange(len(order_idx), device=pos.device)
    order_grid_idx = torch.zeros_like(order_idx_src)
    order_grid_idx.scatter_(index=order_idx, dim=0, src=order_idx_src)
    return order_grid_idx, order_idx


# Test configurations
# Constraints: B >= 1, N >= 32, C >= H, C % H == 0, M < N
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
