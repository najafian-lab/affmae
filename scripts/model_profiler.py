import os
import torch
import argparse
import logging
from torch.profiler import profile, record_function, ProfilerActivity

# Import your local modules
from src.config import load_config
from src.models.aff_mae import AFFMaskedAutoEncoder

def main():
    parser = argparse.ArgumentParser(description='Profile AFF Model with AMP')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on device: {device} using AMP")

    model = AFFMaskedAutoEncoder(
        patch_size=config.patch_size,
        img_size=config.img_size,
        in_channels=config.in_channels,
        mask_ratio=config.mask_ratio,
        encoder_embed_dim=config.aff_embed_dims,
        encoder_depth=config.aff_depths,
        encoder_num_heads=config.aff_num_heads,
        encoder_nbhd_size=config.aff_nbhd_sizes,
        ds_rate=config.aff_ds_rates,
        cluster_size=config.aff_cluster_size,
        mlp_ratio=config.aff_mlp_ratio,
        alpha=config.aff_alpha,
        global_attention=config.aff_global_attention,
        merging_method=config.aff_merging_method,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_num_heads=config.decoder_num_heads,
    ).to(device)

    model.train()

    loss_scaler = torch.amp.GradScaler()

    batch_size = getattr(config, 'batch_size', 2)
    
    samples = torch.randn(
        batch_size, 
        config.in_channels, 
        config.img_size, 
        config.img_size
    ).to(device)

    print(f"Input Shape: {samples.shape}")

    print("Warming up CUDA with AMP...")
    for _ in range(3):
        model.zero_grad()
        with torch.autocast(device_type='cuda'):
            loss, _ = model(samples)
        
        loss_scaler.scale(loss).backward()

    print("Starting Profiler...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
        record_shapes=True, 
        profile_memory=True, 
        with_stack=True
    ) as prof:
        
        with record_function("AFF_Training_Step_AMP"):
            model.zero_grad()
            
            with torch.autocast(device_type='cuda'):
                loss, aux_losses_list = model(samples)
            
            loss_scaler.scale(loss).backward()

    print("\n" + "="*80)
    print(" TOP 20 OPERATIONS BY GPU MEMORY ALLOCATION (MB)")
    print("="*80)
    # converting display to MB for easier reading
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

    print("\n" + "="*80)
    print(" TOP 20 OPERATIONS BY GPU TIME")
    print("="*80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    trace_path = "aff_amp_profile.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nTrace saved to {trace_path}")

if __name__ == "__main__":
    main()