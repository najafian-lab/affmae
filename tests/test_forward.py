"""
Forward pass tests for cluster attention modules using pytest and torch.testing.
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


def create_modules(dim: int, num_heads: int, device: str, proj_drop: float = 0.0):
    """Create both old and new cluster attention modules."""
    old_module = OldClusterAttention(
        dim=dim, num_heads=num_heads, proj_drop=proj_drop
    ).to(device).eval()
    
    new_module = TritonClusterAttention(
        dim=dim, num_heads=num_heads, proj_drop=proj_drop
    ).to(device).eval()
    
    return old_module, new_module


class TestForwardPass:
    """Test class for forward pass comparisons."""
    
    def test_local_attention_forward_pass(self, test_config, device, dtype, tolerance):
        """Test forward pass for local attention."""
        config = test_config
        
        # Create test data
        test_data = create_test_data(
            config['B'], config['N'], config['C'], config['H'], config['M'], 
            device, dtype
        )
        
        # Create modules
        old_module, new_module = create_modules(config['C'], config['H'], device)
        
        # Copy weights for fair comparison
        copy_weights(old_module, new_module)
        
        # Run forward passes
        with torch.amp.autocast('cuda', dtype=dtype):
            # Old implementation forward pass
            old_output = old_module(
                feat=test_data['feat'],
                member_idx=test_data['member_idx'],
                cluster_mask=test_data['cluster_mask'],
                pe_idx=test_data['pe_idx'],
                global_attn=False
            )
            
            # New implementation forward pass
            new_output = new_module(
                feat=test_data['feat'],
                member_idx=test_data['member_idx'],
                cluster_mask=test_data['cluster_mask'],
                pe_idx=test_data['pe_idx'],
                global_attn=False
            )
        
        # Compare outputs using compare_tensor for detailed diff reporting
        compare_tensor(
            old_output, 
            new_output, 
            tol=tolerance['forward_pass'],
            name=f"Forward pass outputs for {config['name']}"
        )
    
    # Commented out global attention tests for now
    # def test_global_attention_forward_pass(self, test_config, device, dtype, tolerance):
    #     """Test forward pass for global attention."""
    #     config = test_config
    #     
    #     # Create test data
    #     test_data = create_test_data(
    #         config['B'], config['N'], config['C'], config['H'], config['M'], 
    #         device, dtype
    #     )
    #     
    #     # Create modules
    #     old_module, new_module = create_modules(config['C'], config['H'], device)
    #     
    #     # Copy weights for fair comparison
    #     copy_weights(old_module, new_module)
    #     
    #     # Run forward passes with global attention
    #     with torch.amp.autocast('cuda', dtype=dtype):
    #         # Old implementation global attention
    #         old_output = old_module(
    #             feat=test_data['feat'],
    #             member_idx=test_data['member_idx'],
    #                         cluster_mask=test_data['cluster_mask'],
    #             pe_idx=test_data['pe_idx'],
    #             global_attn=True
    #         )
    #         
    #         # New implementation global attention
    #         new_output = new_module(
    #             feat=test_data['feat'],
    #             member_idx=test_data['member_idx'],
    #             cluster_mask=test_data['cluster_mask'],
    #             pe_idx=test_data['pe_idx'],
    #             pre_table=test_data['pre_table'],
    #             global_attn=True
    #         )
    #     
    #     # Compare outputs using torch.testing.assert_close
    #     torch.testing.assert_close(
    #         old_output.float(), 
    #         new_output.float(), 
    #         rtol=tolerance['forward_pass'], 
    #         atol=tolerance['forward_pass'],
    #         msg=f"Global attention forward pass outputs differ for {config['name']}"
    #     )
    
    def test_output_shapes(self, test_config, device, dtype):
        """Test that output shapes match between implementations."""
        config = test_config
        
        # Create test data
        test_data = create_test_data(
            config['B'], config['N'], config['C'], config['H'], config['M'], 
            device, dtype
        )
        
        # Create modules
        old_module, new_module = create_modules(config['C'], config['H'], device)
        
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
        
        # Check shapes match
        assert old_output.shape == new_output.shape, f"Output shapes don't match: {old_output.shape} vs {new_output.shape}"
        assert old_output.shape == (config['B'], config['N'], config['C']), f"Expected shape {(config['B'], config['N'], config['C'])}, got {old_output.shape}"
    
    def test_different_dtypes(self, test_config, device, tolerance):
        """Test that implementations work with different dtypes."""
        config = test_config
        
        for dtype in [torch.float32, torch.float16]:
            if dtype == torch.float16 and device == "cpu":
                pytest.skip("float16 not supported on CPU")
            
            # Create test data
            test_data = create_test_data(
                config['B'], config['N'], config['C'], config['H'], config['M'], 
                device, dtype
            )
            
            # Create modules
            old_module, new_module = create_modules(config['C'], config['H'], device)
            
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
            
            # Compare outputs (more lenient tolerance for float16)
            compare_tensor(
                old_output, 
                new_output, 
                tol=tolerance['forward_pass'],
                name=f"Forward pass outputs for {config['name']}"
            )
