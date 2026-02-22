"""
Backward pass tests for cluster attention modules using pytest and torch.testing.
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
    ).to(device)
    
    new_module = TritonClusterAttention(
        dim=dim, num_heads=num_heads, proj_drop=proj_drop
    ).to(device)
    
    return old_module, new_module


def get_parameter_gradients(module: nn.Module):
    """Extract gradients from module parameters."""
    gradients = {}
    for name, param in module.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone()
    return gradients


class TestBackwardPass:
    """Test class for backward pass comparisons."""
    
    def test_local_attention_backward_pass(self, test_config, device, dtype, tolerance):
        """Test backward pass for local attention."""
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
        
        # Create separate input tensors for each module to avoid gradient interference
        old_feat = test_data['feat'].clone().detach().requires_grad_(True)
        new_feat = test_data['feat'].clone().detach().requires_grad_(True)
        
        # Run forward passes
        with torch.amp.autocast('cuda', dtype=dtype):
            # Old implementation forward pass
            old_output = old_module(
                feat=old_feat,
                member_idx=test_data['member_idx'],
                cluster_mask=test_data['cluster_mask'],
                pe_idx=test_data['pe_idx'],
                global_attn=False
            )
            
            # New implementation forward pass
            new_output = new_module(
                feat=new_feat,
                member_idx=test_data['member_idx'],
                cluster_mask=test_data['cluster_mask'],
                pe_idx=test_data['pe_idx'],
                global_attn=False
            )
        
        # Compute losses
        old_loss = old_output.float().mean()
        new_loss = new_output.float().mean()
        
        # Compare losses
        compare_tensor(
            old_loss, 
            new_loss, 
            tol=tolerance['loss'], 
            name=f"Loss values for {config['name']}"
        )
        
        # Backward passes
        old_loss.backward(inputs=[old_feat, old_module.q.weight, old_module.q.bias, 
                                  old_module.kv.weight, old_module.kv.bias, 
                                  old_module.proj.weight, old_module.proj.bias,
                                  old_module.pos_embed.weight, old_module.pos_embed.bias,
                                  old_module.blank_k, old_module.blank_v])
        
        new_loss.backward(inputs=[new_feat, new_module.q.weight, new_module.q.bias,
                                  new_module.kv.weight, new_module.kv.bias,
                                  new_module.proj.weight, new_module.proj.bias,
                                  new_module.pos_embed.weight, new_module.pos_embed.bias,
                                  new_module.blank_k, new_module.blank_v])
        
        # Compare input gradients using compare_tensor for detailed diff reporting
        compare_tensor(
            old_feat.grad.float(), 
            new_feat.grad.float(), 
            tol=tolerance['backward_pass'],
            name=f"Input gradients for {config['name']}"
        )
        
        # Get parameter gradients
        old_grads = get_parameter_gradients(old_module)
        new_grads = get_parameter_gradients(new_module)
        
        # Compare parameter gradients
        param_names = ['q.weight', 'q.bias', 'kv.weight', 'kv.bias', 'proj.weight', 'proj.bias', 
                      'pos_embed.weight', 'pos_embed.bias', 'blank_k', 'blank_v']
        
        for name in param_names:
            if name in old_grads and name in new_grads:
                compare_tensor(
                    old_grads[name].float(), 
                    new_grads[name].float(), 
                    tol=tolerance['backward_pass'],
                    name=f"Parameter gradient {name} for {config['name']}"
                )
    
    # Commented out global attention tests for now
    # def test_global_attention_backward_pass(self, test_config, device, dtype, tolerance):
    #     """Test backward pass for global attention."""
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
    #     # Create separate input tensors for each module
    #     old_feat = test_data['feat'].clone().detach().requires_grad_(True)
    #     new_feat = test_data['feat'].clone().detach().requires_grad_(True)
    #     
    #     # Run forward passes with global attention
    #     with torch.amp.autocast('cuda', dtype=dtype):
    #         # Old implementation global attention
    #         old_output = old_module(
    #             feat=old_feat,
    #             member_idx=test_data['member_idx'],
    #             cluster_mask=test_data['cluster_mask'],
    #             pe_idx=test_data['pe_idx'],
    #             global_attn=True
    #         )
    #         
    #         # New implementation global attention
    #         new_output = new_module(
    #             feat=new_feat,
    #             member_idx=test_data['member_idx'],
    #             cluster_mask=test_data['cluster_mask'],
    #             pe_idx=test_data['pe_idx'],
    #             pre_table=test_data['pre_table'],
    #             global_attn=True
    #         )
    #     
    #     # Compute losses
    #     old_loss = old_output.float().mean()
    #     new_loss = new_output.float().mean()
    #     
    #     # Compare losses
    #     torch.testing.assert_close(
    #         old_loss, 
    #         new_loss, 
    #         rtol=tolerance['loss'], 
    #         atol=tolerance['loss'],
    #         msg=f"Global attention loss values differ for {config['name']}"
    #     )
    #     
    #     # Backward passes
    #     old_loss.backward(inputs=[old_feat, old_module.q.weight, old_module.q.bias, 
    #                               old_module.kv.weight, old_module.kv.bias, 
    #                               old_module.proj.weight, old_module.proj.bias,
    #                               old_module.pos_embed.weight, old_module.pos_embed.bias,
    #                               old_module.blank_k, old_module.blank_v])
    #     
    #     new_loss.backward(inputs=[new_feat, new_module.q.weight, new_module.q.bias,
    #                               new_module.kv.weight, new_module.kv.bias,
    #                               new_module.proj.weight, new_module.proj.bias,
    #                               new_module.pos_embed.weight, new_module.pos_embed.bias,
    #                               new_module.blank_k, new_module.blank_v])
    #     
    #     # Compare input gradients
    #     torch.testing.assert_close(
    #         old_feat.grad.float(), 
    #         new_feat.grad.float(), 
    #         rtol=tolerance['backward_pass'], 
    #         atol=tolerance['backward_pass'],
    #         msg=f"Global attention input gradients differ for {config['name']}"
    #     )
    #     
    #     # Get parameter gradients
    #     old_grads = get_parameter_gradients(old_module)
    #     new_grads = get_parameter_gradients(new_module)
    #     
    #     # Compare parameter gradients
    #     param_names = ['q.weight', 'q.bias', 'kv.weight', 'kv.bias', 'proj.weight', 'proj.bias', 
    #                   'pos_embed.weight', 'pos_embed.bias', 'blank_k', 'blank_v']
    #     
    #     for name in param_names:
    #         if name in old_grads and name in new_grads:
    #             torch.testing.assert_close(
    #                 old_grads[name].float(), 
    #                 new_grads[name].float(), 
    #                 rtol=tolerance['backward_pass'], 
    #                 atol=tolerance['backward_pass'],
    #                 msg=f"Global attention parameter gradient {name} differs for {config['name']}"
    #             )
    
    def test_gradient_norms(self, test_config, device, dtype):
        """Test that gradient norms are reasonable."""
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
        
        # Create separate input tensors
        old_feat = test_data['feat'].clone().detach().requires_grad_(True)
        new_feat = test_data['feat'].clone().detach().requires_grad_(True)
        
        # Run forward and backward passes
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
        old_output.float().mean().backward(inputs=[old_feat, old_module.q.weight, old_module.q.bias, 
                                                    old_module.kv.weight, old_module.kv.bias, 
                                                    old_module.proj.weight, old_module.proj.bias,
                                                    old_module.pos_embed.weight, old_module.pos_embed.bias,
                                                    old_module.blank_k, old_module.blank_v])
        
        new_output.float().mean().backward(inputs=[new_feat, new_module.q.weight, new_module.q.bias,
                                                    new_module.kv.weight, new_module.kv.bias,
                                                    new_module.proj.weight, new_module.proj.bias,
                                                    new_module.pos_embed.weight, new_module.pos_embed.bias,
                                                    new_module.blank_k, new_module.blank_v])
        
        # Check that gradients are not NaN or infinite
        assert not torch.isnan(old_feat.grad).any(), "Old implementation has NaN gradients"
        assert not torch.isnan(new_feat.grad).any(), "New implementation has NaN gradients"
        assert not torch.isinf(old_feat.grad).any(), "Old implementation has infinite gradients"
        assert not torch.isinf(new_feat.grad).any(), "New implementation has infinite gradients"
        
        # Check that gradient norms are reasonable (not too large or too small)
        old_grad_norm = old_feat.grad.norm().item()
        new_grad_norm = new_feat.grad.norm().item()
        
        assert old_grad_norm > 1e-8, f"Old implementation gradient norm too small: {old_grad_norm}"
        assert new_grad_norm > 1e-8, f"New implementation gradient norm too small: {new_grad_norm}"
        assert old_grad_norm < 1e3, f"Old implementation gradient norm too large: {old_grad_norm}"
        assert new_grad_norm < 1e3, f"New implementation gradient norm too large: {new_grad_norm}"
    
    def test_different_dtypes_backward(self, test_config, device):
        """Test that backward pass works with different dtypes."""
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
            
            # Create separate input tensors
            old_feat = test_data['feat'].clone().detach().requires_grad_(True)
            new_feat = test_data['feat'].clone().detach().requires_grad_(True)
            
            # Run forward and backward passes
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
            old_output.float().mean().backward(inputs=[old_feat, old_module.q.weight, old_module.q.bias, 
                                                        old_module.kv.weight, old_module.kv.bias, 
                                                        old_module.proj.weight, old_module.proj.bias,
                                                        old_module.pos_embed.weight, old_module.pos_embed.bias,
                                                        old_module.blank_k, old_module.blank_v])
            
            new_output.float().mean().backward(inputs=[new_feat, new_module.q.weight, new_module.q.bias,
                                                        new_module.kv.weight, new_module.kv.bias,
                                                        new_module.proj.weight, new_module.proj.bias,
                                                        new_module.pos_embed.weight, new_module.pos_embed.bias,
                                                        new_module.blank_k, new_module.blank_v])
            
            # Compare gradients (more lenient tolerance for float16)
            tolerance = 1e-2 if dtype == torch.float16 else 1e-3
            torch.testing.assert_close(
                old_feat.grad.float(), 
                new_feat.grad.float(), 
                rtol=tolerance, 
                atol=tolerance,
                msg=f"Input gradients differ for {config['name']} with dtype {dtype}"
            )
