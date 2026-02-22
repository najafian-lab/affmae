import os
import time
import argparse
import logging
import datetime
from typing import Any, Union, Tuple
from tqdm import tqdm
import subprocess
import warnings
# specific warning ignore from pydantic and timm
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")

import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from src.config import load_config, create_experiment_dir
from src.data.pretrain_dataset import build_pretrain_dataloader, get_stable_visualization_batch
from src.models.aff_mae import AFFMaskedAutoEncoder
from src.models.vit_mae import VanillaViTMAE
from src.utils.misc import set_seed, save_checkpoint, load_checkpoint, cosine_lr_schedule, AverageMeter, run_pca_visualization, setup_logging

Config = Any
AMP = True 

def get_flops(model, inp: Union[torch.Tensor, Tuple], with_backward=False):
    istrain = model.training
    model.eval()
    
    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)
    inp = inp.to("cuda")

    flop_counter = FlopCounterMode(display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inp)[0].sum().backward()
        else:
            model(inp)
    total_flops =  flop_counter.get_total_flops()
    if istrain:
        model.train()
    return total_flops

def measure_throughput(model, input_shape, device, batch_size=24, warmup_steps=10, measure_steps=50):
    """
    Measures inference throughput (samples/second) with proper synchronization.
    """
    model.eval()
    
    # Create synthetic input
    dummy_input = torch.randn(batch_size, *input_shape).to(device)
    
    # 1. Warmup: Run a few passes to initialize CUDA kernels and stabilize clock speeds
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True):
            for _ in range(warmup_steps):
                model(dummy_input)
    
    # Synchronize before starting the timer (wait for warmup to finish)
    torch.cuda.synchronize()
    
    start_time = time.time()
    
    # 2. Measurement Loop
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True):
            for _ in range(measure_steps):
                model(dummy_input)
    
    # Synchronize after loop (wait for all kernels to finish)
    torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    total_samples = batch_size * measure_steps
    throughput = total_samples / total_time
    
    return throughput

def measure_throughput_training(model, optimizer, input_shape, device, batch_size=24, warmup_steps=10, measure_steps=50):
    """
    Measures inference throughput (samples/second) with proper synchronization.
    """
    loss_scalar = torch.amp.GradScaler()
    model.train()    
    # Create synthetic input
    dummy_input = torch.randn(batch_size, *input_shape).to(device)
    
    # 1. Warmup: Run a few passes to initialize CUDA kernels and stabilize clock speeds
    with torch.amp.autocast("cuda", enabled=True):
        for _ in range(warmup_steps):
            optimizer.zero_grad()
            loss, _ = model(dummy_input)
            loss_scalar.scale(loss).backward()
    
    # Synchronize before starting the timer (wait for warmup to finish)
    torch.cuda.synchronize()
    
    start_time = time.time()
    
    # 2. Measurement Loop
    with torch.amp.autocast("cuda", enabled=True):
        for _ in range(measure_steps):
            optimizer.zero_grad()
            loss, _ = model(dummy_input)
            loss_scalar.scale(loss).backward()

    # Synchronize after loop (wait for all kernels to finish)
    torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    total_samples = batch_size * measure_steps
    throughput = total_samples / total_time
    
    return throughput

def main():
    parser = argparse.ArgumentParser(description='Train MAE with AFF or ViT backbone')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    # config is flattened, i.e any value can be directly accessed by config.param
    config = load_config(args.config)

    set_seed(config.seed)
    device = torch.device(config.device)
    config.device = device 

    logging.info(f"Experiment started: {config.experiment_name}")
    logging.info(f"Using device: {device}")

    logging.info(f"Initializing model of type: '{config.model_type}'")

    img_sizes = [1024]
    TEST_BATCH_SIZE = 7    # Standardized batch size for memory and throughput tests

    for sizes in img_sizes:
        if config.model_type == 'aff':
            model = AFFMaskedAutoEncoder(
                # general model configs
                patch_size=config.patch_size,
                img_size=sizes,
                in_channels=config.in_channels,
                mask_ratio=config.mask_ratio,
                # aff encoder configs
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
                # decoder configs
                decoder_embed_dim=config.decoder_embed_dim,
                decoder_num_heads=config.decoder_num_heads,
            ).to(device)
        elif config.model_type == 'vit':
            model = VanillaViTMAE(
                patch_size=config.patch_size,
                img_size=sizes,
                in_chans=config.in_channels,
                mask_ratio=config.mask_ratio,
                embed_dim=config.vit_embed_dim,
                depth=config.vit_depth,
                num_heads=config.vit_num_heads,
                decoder_embed_dim=config.decoder_embed_dim,
                decoder_depth=config.decoder_depth,
                decoder_num_heads=config.decoder_num_heads,
                mlp_ratio=4.0,
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

        # set decay and no decay params
        decay_params = []
        no_decay_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if len(p.shape) == 1 or n.endswith('.bias'):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
                
        optims = [
            {'params': decay_params, 'weight_decay': config.weight_decay}, 
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = torch.optim.AdamW(
            optims, 
            lr=config.base_lr, 
            betas=(config.beta1, config.beta2)
        )
        
        loss_scaler = torch.amp.GradScaler()
        max_mem = lambda: torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        
        print(f"\n===== RESULTS FOR IMAGE SIZE {sizes}x{sizes} =====")
        
        with torch.amp.autocast("cuda", enabled=True):
            # 1. FLOPs Check
            fwd_flops = get_flops(model, (1, 1, sizes, sizes), with_backward=False) / 1e9
            bwd_flops = get_flops(model, (1, 1, sizes, sizes), with_backward=True) / 1e9
            print(f"FLOPs (BS=1)       | Fwd: {fwd_flops:.1f} G | Fwd+Bwd: {bwd_flops:.1f} G")

            # 2. Memory Check: Forward Only
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                model(torch.randn(TEST_BATCH_SIZE, 1, sizes, sizes).to(device))
            print(f"VRAM (BS={TEST_BATCH_SIZE})    | Inference: {max_mem(): .2f} GB")

            # 3. Memory Check: Training
            optimizer.zero_grad()
            torch.cuda.reset_peak_memory_stats(device)
            loss, aux_losses_list = model(torch.randn(TEST_BATCH_SIZE, 1, sizes, sizes).to(device))
            torch.amp.GradScaler().scale(loss).backward()
            print(f"VRAM (BS={TEST_BATCH_SIZE})    | Training:  {max_mem(): .2f} GB")
        
        # 4. Throughput Test
        # Clear cache/memory to ensure fair throughput testing
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        
        # Note: Using 1 channel as per your memory test above, modify config.in_channels if needed
        input_shape = (1, sizes, sizes) 
        
        throughput = measure_throughput(
            model=model,
            input_shape=input_shape,
            device=device,
            batch_size=TEST_BATCH_SIZE,
            warmup_steps=100,
            measure_steps=300
        )
        print(f"Throughput INFERENCE (BS={TEST_BATCH_SIZE}) | {throughput:.1f} samples/sec")

        throughput = measure_throughput_training(
            model=model,
            optimizer=optimizer,
            input_shape=input_shape,
            device=device,
            batch_size=TEST_BATCH_SIZE,
            warmup_steps=50,
            measure_steps=150
        )

        print(f"Throughput TRAINING (BS={TEST_BATCH_SIZE}) | {throughput:.1f} samples/sec")

    
        del model
        torch.cuda.empty_cache()

    # if config.wandb_enabled:
    #     wandb.watch(model, log='gradients', log_freq=1000)


if __name__ == '__main__':
    main()