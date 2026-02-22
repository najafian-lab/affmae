#!/usr/bin/env python3
"""
Performance comparison script for Cluster Attention implementations.

Compares:
- Old CUDA implementation (from archive.aff)
- New Triton implementation (from cluster_attn.py)

Usage:
    PYTHONPATH=. python perf.py                    # Run all test configurations
    PYTHONPATH=. python perf.py --config config_1_small  # Run specific configuration
    PYTHONPATH=. python perf.py --help             # Show help
"""

import argparse
import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import sys
import os
import traceback

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test utilities
from conftest import create_test_data, TEST_CONFIGS, copy_weights

# Import implementations
from src.layers.attention import ClusterAttention as TritonClusterAttention
from cluster.archive.aff import ClusterAttention as CUDAClusterAttention


def warmup_module(module, test_data, device, dtype, num_warmup=10):
    """Warm up a module with forward and backward passes."""
    feat = test_data['feat'].clone().detach().requires_grad_(True)
    
    for _ in range(num_warmup):
        with torch.amp.autocast('cuda', dtype=dtype):
            # Forward pass
            if isinstance(module, TritonClusterAttention):
                output = module(
                    feat=feat,
                    member_idx=test_data['member_idx'],
                    cluster_mask=test_data['cluster_mask'],
                    pe_idx=test_data['pe_idx'],
                    global_attn=False
                )
            else:  # CUDAClusterAttention
                output = module(
                    feat=feat,
                    member_idx=test_data['member_idx'],
                    cluster_mask=test_data['cluster_mask'],
                    pe_idx=test_data['pe_idx'],
                    global_attn=False
                )
            
            # Backward pass
            loss = output.float().mean()
            loss.backward(inputs=[feat])
            
            # Clear gradients
            feat.grad = None


def benchmark_forward_backward(module, test_data, device, dtype, num_runs=50):
    """Benchmark forward + backward pass."""
    feat = test_data['feat'].clone().detach().requires_grad_(True)
    
    # Warm up
    warmup_module(module, test_data, device, dtype)
    
    # Clear gradients
    feat.grad = None
    
    # Time forward + backward
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        with torch.amp.autocast('cuda', dtype=dtype):
            if isinstance(module, TritonClusterAttention):
                output = module(
                    feat=feat,
                    member_idx=test_data['member_idx'],
                    cluster_mask=test_data['cluster_mask'],
                    pe_idx=test_data['pe_idx'],
                    global_attn=False
                )
            else:  # CUDAClusterAttention
                output = module(
                    feat=feat,
                    member_idx=test_data['member_idx'],
                    cluster_mask=test_data['cluster_mask'],
                    pe_idx=test_data['pe_idx'],
                    global_attn=False
                )
            
            loss = output.float().mean()
            loss.backward(inputs=[feat])
            
            # Clear gradients for next iteration
            feat.grad = None
    
    end_event.record()
    torch.cuda.synchronize()
    
    total_time = start_event.elapsed_time(end_event)
    avg_time = total_time / num_runs
    
    return avg_time


def benchmark_forward_only(module, test_data, device, dtype, num_runs=100):
    """Benchmark forward pass only."""
    feat = test_data['feat'].clone().detach()
    
    # Warm up
    warmup_module(module, test_data, device, dtype)
    
    # Time forward only
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        with torch.amp.autocast('cuda', dtype=dtype):
            if isinstance(module, TritonClusterAttention):
                output = module(
                    feat=feat,
                    member_idx=test_data['member_idx'],
                    cluster_mask=test_data['cluster_mask'],
                    pe_idx=test_data['pe_idx'],
                    global_attn=False
                )
            else:  # CUDAClusterAttention
                output = module(
                    feat=feat,
                    member_idx=test_data['member_idx'],
                    cluster_mask=test_data['cluster_mask'],
                    pe_idx=test_data['pe_idx'],
                    global_attn=False
                )
    
    end_event.record()
    torch.cuda.synchronize()
    
    total_time = start_event.elapsed_time(end_event)
    avg_time = total_time / num_runs
    
    return avg_time


def benchmark_backward_only(module, test_data, device, dtype, num_runs=50):
    """Benchmark backward pass only."""
    feat = test_data['feat'].clone().detach().requires_grad_(True)
    
    # Warm up
    warmup_module(module, test_data, device, dtype)
    
    # Time backward only
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        with torch.amp.autocast('cuda', dtype=dtype):
            if isinstance(module, TritonClusterAttention):
                output = module(
                    feat=feat,
                    member_idx=test_data['member_idx'],
                    cluster_mask=test_data['cluster_mask'],
                    pe_idx=test_data['pe_idx'],
                    global_attn=False
                )
            else:  # CUDAClusterAttention
                output = module(
                    feat=feat,
                    member_idx=test_data['member_idx'],
                    cluster_mask=test_data['cluster_mask'],
                    pe_idx=test_data['pe_idx'],
                    global_attn=False
                )
            
            loss = output.float().mean()
            loss.backward(inputs=[feat])
            
            # Clear gradients for next iteration
            feat.grad = None
    
    end_event.record()
    torch.cuda.synchronize()
    
    total_time = start_event.elapsed_time(end_event)
    avg_time = total_time / num_runs
    
    return avg_time


def run_performance_test(config_name: str, config: Dict, device: str = "cuda"):
    """Run performance test for a specific configuration with both fp32 and fp16."""
    print(f"\n{'='*80}")
    print(f"PERFORMANCE TEST: {config_name}")
    print(f"{'='*80}")
    print(f"Configuration: {config['description']}")
    print(f"B={config['B']}, N={config['N']}, C={config['C']}, H={config['H']}, M={config['M']}")
    print(f"Device: {device}")
    print()
    
    results = {}
    
    # Test both fp32 and fp16
    for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
        if dtype == torch.float16 and device == "cpu":
            print(f"Skipping {dtype_name} on CPU (not supported)")
            continue
            
        print(f"\n--- {dtype_name.upper()} BENCHMARKS ---")
        
        # Create test data
        test_data = create_test_data(
            config['B'], config['N'], config['C'], config['H'], config['M'],
            device, dtype
        )
        
        # Create modules
        cuda_module = CUDAClusterAttention(
            dim=config['C'], 
            num_heads=config['H'], 
            proj_drop=0.0
        ).to(device).eval()
        
        triton_module = TritonClusterAttention(
            dim=config['C'], 
            num_heads=config['H'], 
            proj_drop=0.0
        ).to(device).eval()
        
        # Copy weights for fair comparison
        with torch.no_grad():
            copy_weights(cuda_module, triton_module)
        
        print("Running benchmarks...")
        
        # Benchmark forward pass
        print("  Forward pass only:")
        cuda_fwd_time = benchmark_forward_only(cuda_module, test_data, device, dtype)
        triton_fwd_time = benchmark_forward_only(triton_module, test_data, device, dtype)
        
        print(f"    CUDA:     {cuda_fwd_time:.3f} ms")
        print(f"    Triton:   {triton_fwd_time:.3f} ms")
        print(f"    Speedup:  {cuda_fwd_time/triton_fwd_time:.2f}x")
        
        # Benchmark backward pass
        print("  Backward pass only:")
        cuda_bwd_time = benchmark_backward_only(cuda_module, test_data, device, dtype)
        triton_bwd_time = benchmark_backward_only(triton_module, test_data, device, dtype)
        
        print(f"    CUDA:     {cuda_bwd_time:.3f} ms")
        print(f"    Triton:   {triton_bwd_time:.3f} ms")
        print(f"    Speedup:  {cuda_bwd_time/triton_bwd_time:.2f}x")
        
        # Benchmark forward + backward
        print("  Forward + Backward:")
        cuda_total_time = benchmark_forward_backward(cuda_module, test_data, device, dtype)
        triton_total_time = benchmark_forward_backward(triton_module, test_data, device, dtype)
        
        print(f"    CUDA:     {cuda_total_time:.3f} ms")
        print(f"    Triton:   {triton_total_time:.3f} ms")
        print(f"    Speedup:  {cuda_total_time/triton_total_time:.2f}x")
        
        # Summary for this dtype
        print(f"\nSummary for {config_name} ({dtype_name}):")
        print(f"  Forward speedup:  {cuda_fwd_time/triton_fwd_time:.2f}x")
        print(f"  Backward speedup: {cuda_bwd_time/triton_bwd_time:.2f}x")
        print(f"  Total speedup:    {cuda_total_time/triton_total_time:.2f}x")
        
        results[dtype_name] = {
            'config_name': config_name,
            'config': config,
            'dtype': dtype_name,
            'cuda_fwd_time': cuda_fwd_time,
            'triton_fwd_time': triton_fwd_time,
            'cuda_bwd_time': cuda_bwd_time,
            'triton_bwd_time': triton_bwd_time,
            'cuda_total_time': cuda_total_time,
            'triton_total_time': triton_total_time,
            'fwd_speedup': cuda_fwd_time/triton_fwd_time,
            'bwd_speedup': cuda_bwd_time/triton_bwd_time,
            'total_speedup': cuda_total_time/triton_total_time
        }
    
    return results


def print_summary_table(results: List[Dict]):
    """Print a summary table of all results."""
    print(f"\n{'='*140}")
    print("PERFORMANCE SUMMARY TABLE")
    print(f"{'='*140}")
    
    # Header
    print(f"{'Config':<20} {'Dtype':<6} {'B':<4} {'N':<6} {'C':<6} {'H':<4} {'M':<4} {'Fwd Speedup':<12} {'Bwd Speedup':<12} {'Total Speedup':<14}")
    print(f"{'-'*140}")
    
    # Data rows
    for config_results in results:
        for dtype_name, result in config_results.items():
            config = result['config']
            print(f"{result['config_name']:<20} {dtype_name:<6} {config['B']:<4} {config['N']:<6} {config['C']:<6} {config['H']:<4} {config['M']:<4} "
                  f"{result['fwd_speedup']:<12.2f} {result['bwd_speedup']:<12.2f} {result['total_speedup']:<14.2f}")
    
    # Calculate averages by dtype
    fp32_results = []
    fp16_results = []
    
    for config_results in results:
        if 'fp32' in config_results:
            fp32_results.append(config_results['fp32'])
        if 'fp16' in config_results:
            fp16_results.append(config_results['fp16'])
    
    print(f"{'-'*140}")
    
    if fp32_results:
        avg_fwd_speedup = sum(r['fwd_speedup'] for r in fp32_results) / len(fp32_results)
        avg_bwd_speedup = sum(r['bwd_speedup'] for r in fp32_results) / len(fp32_results)
        avg_total_speedup = sum(r['total_speedup'] for r in fp32_results) / len(fp32_results)
        print(f"{'AVERAGE (FP32)':<20} {'fp32':<6} {'':<4} {'':<6} {'':<6} {'':<4} {'':<4} {avg_fwd_speedup:<12.2f} {avg_bwd_speedup:<12.2f} {avg_total_speedup:<14.2f}")
    
    if fp16_results:
        avg_fwd_speedup = sum(r['fwd_speedup'] for r in fp16_results) / len(fp16_results)
        avg_bwd_speedup = sum(r['bwd_speedup'] for r in fp16_results) / len(fp16_results)
        avg_total_speedup = sum(r['total_speedup'] for r in fp16_results) / len(fp16_results)
        print(f"{'AVERAGE (FP16)':<20} {'fp16':<6} {'':<4} {'':<6} {'':<6} {'':<4} {'':<4} {avg_fwd_speedup:<12.2f} {avg_bwd_speedup:<12.2f} {avg_total_speedup:<14.2f}")


def main():
    parser = argparse.ArgumentParser(description='Performance comparison for Cluster Attention implementations')
    parser.add_argument('--config', type=str, help='Specific configuration to test (e.g., config_1_small)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (default: cuda)')
    parser.add_argument('--num-runs', type=int, default=50, help='Number of runs for timing (default: 50)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Please use CPU or install CUDA.")
        return
    
    print("Cluster Attention Performance Comparison")
    print("=" * 50)
    print(f"Device: {args.device}")
    print(f"Number of runs: {args.num_runs}")
    
    # Determine which configurations to test
    if args.config:
        if args.config not in TEST_CONFIGS:
            print(f"Error: Configuration '{args.config}' not found.")
            print(f"Available configurations: {list(TEST_CONFIGS.keys())}")
            return
        configs_to_test = {args.config: TEST_CONFIGS[args.config]}
    else:
        configs_to_test = TEST_CONFIGS
    
    # Run performance tests
    results = []
    for config_name, config in configs_to_test.items():
        try:
            result = run_performance_test(config_name, config, args.device)
            results.append(result)
        except Exception as e:
            traceback.print_exc()
            print(f"Error running {config_name}: {e}")
            continue
    
    # Print summary table if multiple configurations were tested
    if len(results) > 1:
        print_summary_table(results)
    
    # Final summary
    if results:
        print(f"\n{'='*50}")
        print("FINAL SUMMARY")
        print(f"{'='*50}")
        
        # Calculate averages by dtype
        fp32_results = []
        fp16_results = []
        
        for config_results in results:
            if 'fp32' in config_results:
                fp32_results.append(config_results['fp32'])
            if 'fp16' in config_results:
                fp16_results.append(config_results['fp16'])
        
        if fp32_results:
            avg_fwd_speedup = sum(r['fwd_speedup'] for r in fp32_results) / len(fp32_results)
            avg_bwd_speedup = sum(r['bwd_speedup'] for r in fp32_results) / len(fp32_results)
            avg_total_speedup = sum(r['total_speedup'] for r in fp32_results) / len(fp32_results)
            
            print(f"FP32 Results:")
            print(f"  Average forward speedup:  {avg_fwd_speedup:.2f}x")
            print(f"  Average backward speedup: {avg_bwd_speedup:.2f}x")
            print(f"  Average total speedup:    {avg_total_speedup:.2f}x")
            
            if avg_total_speedup > 1.0:
                print(f"  Triton implementation is {avg_total_speedup:.2f}x faster on average")
            else:
                print(f"  CUDA implementation is {1/avg_total_speedup:.2f}x faster on average")
        
        if fp16_results:
            avg_fwd_speedup = sum(r['fwd_speedup'] for r in fp16_results) / len(fp16_results)
            avg_bwd_speedup = sum(r['bwd_speedup'] for r in fp16_results) / len(fp16_results)
            avg_total_speedup = sum(r['total_speedup'] for r in fp16_results) / len(fp16_results)
            
            print(f"\nFP16 Results:")
            print(f"  Average forward speedup:  {avg_fwd_speedup:.2f}x")
            print(f"  Average backward speedup: {avg_bwd_speedup:.2f}x")
            print(f"  Average total speedup:    {avg_total_speedup:.2f}x")
            
            if avg_total_speedup > 1.0:
                print(f"  Triton implementation is {avg_total_speedup:.2f}x faster on average")
            else:
                print(f"  CUDA implementation is {1/avg_total_speedup:.2f}x faster on average")


if __name__ == "__main__":
    main()
