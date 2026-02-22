import os
import time
import argparse
import logging
import datetime
from typing import Any
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

from src.config import load_config, create_experiment_dir
from src.data.pretrain_dataset import build_pretrain_dataloader, get_stable_visualization_batch
from src.models.aff_mae import AFFMaskedAutoEncoder
from src.models.vit_mae import VanillaViTMAE
from src.utils.misc import set_seed, save_checkpoint, load_checkpoint, cosine_lr_schedule, AverageMeter, run_pca_visualization, setup_logging

Config = Any
AMP = True 


def calculate_dataset_size(dataloader: DataLoader) -> int:
    """
    iterates through the dataloader once to calculate the exact number of batches.
    this is needed as webdataset loaders don't have __len__
    """
    log_prefix = "dataset"
    logging.info(f"[{log_prefix}] Dry run started to calculate total batches/steps...")
    start = time.time()
    count = 0
    for _ in tqdm(dataloader, desc=f"Counting {log_prefix} batches"):
        count += 1
    
    duration = time.time() - start
    logging.info(f"[{log_prefix}] Count finished in {duration:.2f}s. Total batches: {count}")
    return count


def train_one_epoch(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                    loss_scaler: torch.amp.GradScaler, config: 'Config', epoch: int, global_step: int, 
                    total_max_steps: int, params=None) -> tuple[float, int]:
    model.train()

    meters = {'loss': AverageMeter()} 
    if isinstance(model, AFFMaskedAutoEncoder):
        aux_names = ['res5', 'res4', 'res2'] 
        for name in aux_names:
            meters[name] = AverageMeter()
    else:
        aux_names = []  # vit has not aux losses

    log_start_time = time.time()
    num_accum = config.num_accum
    current_step = global_step

    optimizer.zero_grad()
    
    for batch_idx, (samples, _) in enumerate(dataloader):
        samples = samples.to(config.device)

        if AMP:
            with torch.autocast(device_type='cuda'):
                loss, aux_losses_list = model(samples)
                loss = loss / num_accum
            
            loss_scaler.scale(loss).backward()
        else:
            loss, aux_losses_list = model(samples)
            loss = loss / num_accum
            loss.backward()

        batch_size = samples.size(0)
        meters['loss'].update(loss.item() * num_accum, batch_size) 
        
        # update aux meters
        for i, val in enumerate(aux_losses_list):
            meters[aux_names[i]].update(val.item(), batch_size)
        
        # grad accum
        if (batch_idx + 1) % num_accum == 0:
            GRAD_SCALE = 5.0
            if AMP:
                loss_scaler.unscale_(optimizer)
            
            # Clip grads
            torch.nn.utils.clip_grad_norm_(params, GRAD_SCALE, error_if_nonfinite=False)

            if AMP:
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            
            current_step += 1
            
            # apply lr sched based on the training steps
            cosine_lr_schedule(optimizer, current_step, total_max_steps, config.base_lr, 1e-6, config.warmup_steps)

        # logging
        if (batch_idx + 1) % config.log_freq == 0:
            elapsed_time = time.time() - log_start_time
            samples_per_sec = (config.log_freq * samples.size(0)) / elapsed_time

            aux_str = " ".join([f"{k}:{m.avg:.4f}" for k, m in meters.items() if k != 'loss'])
            
            curr_lr = optimizer.param_groups[0]['lr']
            steps_per_epoch = total_max_steps // config.epochs

            logging.info(
                f"Epoch: {epoch} [{batch_idx+1:>4}/{steps_per_epoch*num_accum}] | "
                f"Loss: {meters['loss'].avg:.4f} ({aux_str}) | "
                f"LR: {curr_lr:.8f} | "
                f"Spd: {samples_per_sec:.1f}/s"
            )

            if config.wandb_enabled:
                log_dict = {
                    "batch_loss": meters['loss'].avg,
                    "lr": curr_lr,
                    "global_step": current_step,
                    "samples_per_sec": samples_per_sec
                }
                for name in aux_names:
                    log_dict[f"loss_{name}"] = meters[name].avg
                
                wandb.log(log_dict)

            log_start_time = time.time()

    return meters['loss'].avg, current_step


def run_evaluation(model: torch.nn.Module, vis_images: torch.Tensor, 
                    epoch: int, output_dir: str, config: Config):
    model.eval()
    logging.info(f"Running evaluation for epoch {epoch}...")

    eval_epoch_dir = os.path.join(output_dir, f'epoch_{epoch}')
    os.makedirs(eval_epoch_dir, exist_ok=True)

    recon_path = os.path.join(eval_epoch_dir, 'recon.png')
    num_vis = min(5, vis_images.shape[0])
    model.visualize(vis_images, num_vis, recon_path, seed=config.seed)

    pca_path = os.path.join(eval_epoch_dir, 'pca_feats.png')

    # only aff has token visualization and per stage pca
    if isinstance(model, AFFMaskedAutoEncoder):
        token_path = None
        token_path = os.path.join(eval_epoch_dir, 'token_loc.png')
        model.visualize_tokens(vis_images, num_vis, token_path, seed=config.seed)
        run_pca_visualization(model, vis_images[:num_vis], pca_path, config.device)
    else:
        token_path = None

    model.train()


def main():
    parser = argparse.ArgumentParser(description='Train MAE with AFF or ViT backbone')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--resume', type=str, required=False, help='Path to resume from.')
    args = parser.parse_args()

    # config is flattened, i.e any value can be directly accessed by config.param
    config = load_config(args.config)

    if resume_path := args.resume:
        config.resume_path = resume_path

    exp_dir = create_experiment_dir(config, args.config)
    setup_logging(exp_dir)

    if config.wandb_enabled:
        try:
            wandb.init(
                project=config.project,
                entity=config.entity,
                config=config.__dict__, 
                name=config.experiment_name,
                resume="allow" if config.resume_path else None
            )
            logging.info("W&B run initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize W&B: {e}")
            return

    set_seed(config.seed)
    device = torch.device(config.device)
    config.device = device 

    logging.info(f"Experiment started: {config.experiment_name}")
    logging.info(f"Using device: {device}")


    # we create a temporary loader just to count batches.
    logging.info("Pre-calculating dataset length...")
    count_loader = build_pretrain_dataloader(config)
    total_batches_per_epoch = calculate_dataset_size(count_loader)
    
    # calculate total training steps (batches / accumulation)
    steps_per_epoch = total_batches_per_epoch // config.num_accum
    total_max_steps = config.epochs * steps_per_epoch
    
    logging.info(f"Batches per epoch: {total_batches_per_epoch}")
    logging.info(f"Steps per epoch: {steps_per_epoch}")
    logging.info(f"Total steps (Training Length): {total_max_steps}")

    # main dataloader
    dataloader = build_pretrain_dataloader(config) 
    
    vis_images = get_stable_visualization_batch(config, device)

    logging.info(f"Initializing model of type: '{config.model_type}'")
    
    if config.model_type == 'aff':
        model = AFFMaskedAutoEncoder(
            # general model configs
            patch_size=config.patch_size,
            img_size=config.img_size,
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
            img_size=config.img_size,
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
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logging.info(f'Model initialized with {param_count:.2f}M trainable parameters.')

    # if config.wandb_enabled:
    #     wandb.watch(model, log='gradients', log_freq=1000)


    start_epoch = 0
    global_step = 0
    
    resume_path = config.resume_path
    
    if resume_path and os.path.exists(resume_path):
        logging.info(f"Resuming from checkpoint: {resume_path}")
        start_epoch, _ = load_checkpoint(model, optimizer, resume_path)
        start_epoch += 1
        
        # we just start at the beginning of next epoch for the lr sched
        global_step = start_epoch * steps_per_epoch
        
        logging.info(f'Resumed successfully. Starting Epoch: {start_epoch}, Global Step: {global_step}')
    

    logging.info(f"Starting training loop from epoch {start_epoch} to {config.epochs}.")
    total_start_time = time.time()

    try:
        for epoch in range(start_epoch, config.epochs):
            epoch_start_time = time.time()
            
            train_loss, global_step = train_one_epoch(
                model=model, 
                dataloader=dataloader, 
                optimizer=optimizer, 
                loss_scaler=loss_scaler, 
                config=config, 
                epoch=epoch, 
                global_step=global_step, 
                total_max_steps=total_max_steps, 
                params=(decay_params + no_decay_params)
            )
            
            epoch_time = time.time() - epoch_start_time
            lr = optimizer.param_groups[0]['lr']

            logging.info("-" * 80)
            logging.info(f"Epoch {epoch} Done | Avg Loss: {train_loss:.4f} | Time: {epoch_time:.2f}s | LR: {lr:.8f}")
            logging.info("-" * 80)

            if config.wandb_enabled:
                wandb.log({
                    'epoch': epoch, 
                    'epoch_avg_loss': train_loss, 
                    'epoch_time_s': epoch_time
                })

            # Checkpointing, save on first epoch
            if (epoch + 1) % config.save_freq == 0 or epoch == config.epochs - 1 or epoch == 0:
                checkpoint_path = os.path.join(exp_dir, 'checkpoints', f'ckpt_epoch_{epoch}.pth') 
                save_checkpoint(model, optimizer, epoch, global_step, train_loss, checkpoint_path)
                
                # Visualization
                run_evaluation(model, vis_images, epoch, os.path.join(exp_dir, 'evaluations'), config)

    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise e
    finally:
        total_time_taken = time.time() - total_start_time
        logging.info(f"Training finished in {str(datetime.timedelta(seconds=int(total_time_taken)))}.")

        if config.wandb_enabled:
            wandb.finish()

if __name__ == '__main__':
    main()
