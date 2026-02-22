"""
Integration tests for cluster attention modules.

These tests verify that the modules work correctly in realistic scenarios
and handle edge cases properly.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
import warnings

# Suppress pydantic warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import create_test_data, compare_tensor, copy_weights
from cluster.cluster_attn import ClusterAttention as TritonClusterAttention
from cluster.archive.aff import ClusterAttention as OldClusterAttention


class TestIntegration:
    """Integration tests for cluster attention modules."""
    
    def test_module_initialization(self, device):
        """Test that modules can be initialized correctly."""
        dim, num_heads = 64, 4
        
        # Test old module
        old_module = OldClusterAttention(dim=dim, num_heads=num_heads).to(device)
        assert old_module.dim == dim
        assert old_module.num_heads == num_heads
        
        # Test new module
        new_module = TritonClusterAttention(dim=dim, num_heads=num_heads).to(device)
        assert new_module.dim == dim
        assert new_module.num_heads == num_heads
        
        # Test that both modules have the same number of parameters
        old_params = sum(p.numel() for p in old_module.parameters())
        new_params = sum(p.numel() for p in new_module.parameters())
        assert old_params == new_params, f"Parameter count mismatch: {old_params} vs {new_params}"
    
    def test_parameter_shapes(self, device):
        """Test that parameter shapes are correct."""
        dim, num_heads = 128, 8
        
        old_module = OldClusterAttention(dim=dim, num_heads=num_heads).to(device)
        new_module = TritonClusterAttention(dim=dim, num_heads=num_heads).to(device)
        
        # Check Q projection
        assert old_module.q.weight.shape == (dim, dim)
        assert new_module.q.weight.shape == (dim, dim)
        
        # Check KV projection
        assert old_module.kv.weight.shape == (2 * dim, dim)
        assert new_module.kv.weight.shape == (2 * dim, dim)
        
        # Check output projection
        assert old_module.proj.weight.shape == (dim, dim)
        assert new_module.proj.weight.shape == (dim, dim)
        
        # Check position embedding
        assert old_module.pos_embed.weight.shape == (num_heads, 5)
        assert new_module.pos_embed.weight.shape == (num_heads, 5)
        
        # Check blank tokens
        assert old_module.blank_k.shape == (dim,)
        assert new_module.blank_k.shape == (dim,)
        assert old_module.blank_v.shape == (dim,)
        assert new_module.blank_v.shape == (dim,)
    
    def test_edge_case_small_batch(self, device, dtype):
        """Test behavior with minimal batch size."""
        B, N, C, H, M = 1, 64, 32, 1, 32
        
        # Create test data with minimal batch size
        test_data = create_test_data(B, N, C, H, M, device, dtype)
        
        # Create modules
        old_module = OldClusterAttention(dim=C, num_heads=H).to(device).eval()
        new_module = TritonClusterAttention(dim=C, num_heads=H).to(device).eval()
        
        # Copy weights for fair comparison
        copy_weights(old_module, new_module)
        
        # Run forward passes
        with torch.amp.autocast('cuda', dtype=dtype):
            old_output = old_module(
                feat=test_data['feat'],
                member_idx=test_data['member_idx'],
                cluster_mask=test_data['cluster_mask'],
                pe_idx=test_data['pe_idx'],
                global_attn=False
            )
            
            new_output = new_module(
                feat=test_data['feat'],
                member_idx=test_data['member_idx'],
                cluster_mask=test_data['cluster_mask'],
                pe_idx=test_data['pe_idx'],
                global_attn=False
            )
        
        # Check that outputs have correct shape
        assert old_output.shape == (B, N, C)
        assert new_output.shape == (B, N, C)
        
        # Check that outputs are equal using compare_tensor for detailed diff reporting
        compare_tensor(old_output, new_output, tol=1e-3, name="Small batch outputs")
    
    def test_gradient_flow(self, test_config, device, dtype):
        """Test that gradients flow correctly through the modules."""
        config = test_config
        
        # Create test data
        test_data = create_test_data(
            config['B'], config['N'], config['C'], config['H'], config['M'], 
            device, dtype
        )
        
        # Create modules
        old_module = OldClusterAttention(dim=config['C'], num_heads=config['H']).to(device)
        new_module = TritonClusterAttention(dim=config['C'], num_heads=config['H']).to(device)
        
        # Copy weights for fair comparison
        copy_weights(old_module, new_module)
        
        # Create input with requires_grad
        old_feat = test_data['feat'].clone().detach().requires_grad_(True)
        new_feat = test_data['feat'].clone().detach().requires_grad_(True)
        
        # Run forward passes
        with torch.amp.autocast('cuda', dtype=dtype):
            old_output = old_module(
                feat=old_feat,
                member_idx=test_data['member_idx'],
                cluster_mask=test_data['cluster_mask'],
                pe_idx=test_data['pe_idx'],
                global_attn=False
            )
            
            new_output = new_module(
                feat=new_feat,
                member_idx=test_data['member_idx'],
                cluster_mask=test_data['cluster_mask'],
                pe_idx=test_data['pe_idx'],
                global_attn=False
            )
        
        # Backward passes
        old_output.float().mean().backward()
        new_output.float().mean().backward()
        
        # Check that gradients exist and are not zero
        assert old_feat.grad is not None, "Old module input gradient is None"
        assert new_feat.grad is not None, "New module input gradient is None"
        assert old_feat.grad.abs().sum() > 0, "Old module input gradient is zero"
        assert new_feat.grad.abs().sum() > 0, "New module input gradient is zero"
        
        # Check that parameter gradients exist
        for name, param in old_module.named_parameters():
            assert param.grad is not None, f"Old module parameter {name} gradient is None"
            assert param.grad.abs().sum() > 0, f"Old module parameter {name} gradient is zero"
        
        for name, param in new_module.named_parameters():
            assert param.grad is not None, f"New module parameter {name} gradient is None"
            assert param.grad.abs().sum() > 0, f"New module parameter {name} gradient is zero"
