import os
import time
import random
import logging
import copy
import datetime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from src.config import load_config
from src.data.finetune_dataset import build_finetune_dataloader, build_test_dataloader
from src.models.aff_segmentation import AFFSegmentation
from src.models.vit_segmentation import ViTSegmentation
from src.losses import FocalLoss, DiceLoss, ComboLoss, compute_iou
from src.utils.misc import AverageMeter, cosine_lr_schedule, visualize_results_multiclass, setup_logging, set_seed
from src.utils.build_finetune_optimizer import build_optimizer_with_llrd


def train_epoch(model, dataloader, optimizer, loss_fn, epoch, loss_scaler, cfg, global_step):
    model.train()
    meters = {'loss': AverageMeter()} 
    if isinstance(model, AFFSegmentation):
        aux_names = ['res5', 'res4', 'res2'] 
    else:
        aux_names = ['res2']
    for name in aux_names:
        meters[name] = AverageMeter()
    
    total_batches = len(dataloader)
    max_steps = cfg.epochs * total_batches
    min_lr = cfg.min_lr
    warmup_steps = cfg.warmup_epochs * total_batches
    GRAD_SCALE = 5.0

    for i, (images, targets, _) in enumerate(dataloader):
        images, targets = images.to(cfg.device), targets.to(cfg.device).long()
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            pred_logits = model(images)
            if len(pred_logits) > 1:
                aux_losses_list = [loss_fn(pred_logits[0], targets), loss_fn(pred_logits[1], targets), loss_fn(pred_logits[2], targets)]   
                loss = aux_losses_list[0] * 0.05 + aux_losses_list[1] * 0.12 + aux_losses_list[2]
            else:
                aux_losses_list = [loss_fn(pred_logits[0], targets)]
                loss = aux_losses_list[-1]
        
        for i, val in enumerate(aux_losses_list):
            meters[aux_names[i]].update(val.item(), images.size(0)) 


        loss_scaler.scale(loss).backward()
        loss_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_SCALE)
        loss_scaler.step(optimizer)
        loss_scaler.update()  

        global_step += 1
        cosine_lr_schedule(optimizer, global_step, max_steps, cfg.learning_rate, min_lr, warmup_steps)

    return meters['res2'].avg, optimizer.param_groups[0]['lr'], global_step

def validate(model, dataloader, cfg):
    model.eval()
    
    # Trackers
    meters = {'loss': AverageMeter()} 
    if isinstance(model, AFFSegmentation):
        aux_names = ['res5', 'res4', 'res2'] 
    else:
        aux_names = ['res2']

    for name in aux_names:
        meters[name] = AverageMeter()

    class_iou_sums = torch.zeros(cfg.num_classes - 1, device=cfg.device)
    total_images = 0
    
    sample_results = []

    with torch.no_grad():
        for images, targets, paths in dataloader:
            images, targets = images.to(cfg.device), targets.to(cfg.device).long()
            
            pred_logits = model(images)
            if len(pred_logits) > 1:
                aux_losses_list = [compute_iou(pred_logits[0], targets), compute_iou(pred_logits[1], targets), compute_iou(pred_logits[2], targets)]   
            else:
                aux_losses_list = [compute_iou(pred_logits[0], targets)]

            for i, val in enumerate(aux_losses_list):
                meters[aux_names[i]].update(val.mean().item(), images.size(0)) 
            
            # Calculate IoU per image per class [B, C-1]
            batch_ious = compute_iou(pred_logits[-1], targets)
                        
            class_iou_sums += batch_ious.sum(dim=0)
            total_images += images.size(0)

            sample_mean_ious = batch_ious.mean(dim=1)

            img_cpu = images.cpu()
            tgt_cpu = targets.cpu()
            pred_cpu = pred_logits[-1].cpu()
            
            batch_paths = paths[0] if isinstance(paths, (tuple, list)) else paths

            for b in range(images.size(0)):
                sample_results.append({
                    'img': img_cpu[b],
                    'tgt': tgt_cpu[b],
                    'pred': pred_cpu[b],
                    'path': batch_paths[b],
                    'iou': sample_mean_ious[b].item()
                })

    global_iou_per_class = class_iou_sums / (total_images + 1e-6)
    
    sample_results.sort(key=lambda x: x['iou'])
    worst_k = sample_results[:20]

    val_xs = torch.stack([x['img'] for x in worst_k], dim=0)
    val_ys = torch.stack([x['tgt'] for x in worst_k], dim=0)
    val_preds = torch.stack([x['pred'] for x in worst_k], dim=0)
    val_paths = [x['path'] for x in worst_k]

    return meters, global_iou_per_class, val_xs, val_ys, val_preds, val_paths


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    original_exp_name = base_cfg.name
    
    seeds = [42, 123, 2026, 999, 67]

    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg.name = f"{original_exp_name}_seed{seed}"
        
        print(f"\n{'='*40}")
        print(f"STARTING RUN WITH SEED: {seed}")
        print(f"OUTPUT DIR: {os.path.join(cfg.output_dir, cfg.name)}")
        print(f"{'='*40}\n")

        exp_dir = os.path.join(cfg.output_dir, cfg.name)
        os.makedirs(exp_dir, exist_ok=True)
        
        set_seed(seed)
        
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        setup_logging(exp_dir)
        
        cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if cfg.wandb_enabled:
            wandb.init(
                entity="najafian-lab-2025", 
                project="aff-mae-finetune", 
                name=cfg.name, 
                config=vars(cfg),
                reinit=True
            )

        train_dl = build_finetune_dataloader(cfg, is_train=True)

        if cfg.model_type == "aff":
            model = AFFSegmentation(
                patch_size=cfg.patch_size,
                img_size=cfg.img_size,
                in_channels=cfg.in_channels,
                encoder_embed_dim=cfg.aff_embed_dims,
                encoder_depth=cfg.aff_depths,
                encoder_num_heads=cfg.aff_num_heads,
                encoder_nbhd_size=cfg.aff_nbhd_sizes,
                ds_rate=cfg.aff_ds_rates,
                cluster_size=cfg.aff_cluster_size,
                global_attention=cfg.aff_global_attention,
                decoder_embed_dim=cfg.decoder_embed_dim,
                decoder_num_heads=cfg.decoder_num_heads,
                merging_method=cfg.aff_merging_method,
                mlp_ratio=cfg.aff_mlp_ratio,
                alpha=cfg.aff_alpha,
                num_classes=cfg.num_classes
            )
        elif cfg.model_type == 'vit':
            model = ViTSegmentation(
                patch_size=cfg.patch_size,
                img_size=cfg.img_size,
                in_chans=cfg.in_channels,
                embed_dim=cfg.vit_embed_dim,
                depth=cfg.vit_depth,
                num_heads=cfg.vit_num_heads,
                decoder_embed_dim=cfg.decoder_embed_dim,
                decoder_depth=cfg.decoder_depth,
                decoder_num_heads=cfg.decoder_num_heads,
                mlp_ratio=4.0,
                num_classes=cfg.num_classes
            )

        logging.info(f"Loading pretrained weights from: {cfg.pretrained_ckpt_path}")
        checkpoint = torch.load(cfg.pretrained_ckpt_path, map_location='cpu', weights_only=False)
        
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        weight_key = 'decoder_pred_head.weight'
        bias_key = 'decoder_pred_head.bias'

        if weight_key in state_dict:
            old_weight = state_dict[weight_key]
            old_bias = state_dict[bias_key]

            logging.info(f"Expanding decoder head from {old_weight.shape[0]} to {model.decoder_pred_head.out_features} output units.")
            
            new_weight = old_weight.repeat(cfg.num_classes, 1)
            new_bias = old_bias.repeat(cfg.num_classes)

            weight_noise = torch.randn_like(new_weight) * 1e-4
            bias_noise = torch.randn_like(new_bias) * 1e-4
            
            state_dict[weight_key] = new_weight + weight_noise
            state_dict[bias_key] = new_bias + bias_noise
            
            logging.info("Successfully resized and added noise to decoder_pred_head.")

        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logging.warning(f"Missing keys: {incompatible.missing_keys}")
        
        model.to(cfg.device)
        
        optimizer = build_optimizer_with_llrd(model, cfg, 0.6)
        scaler = torch.amp.GradScaler()
        
        weights = torch.tensor(cfg.class_weighting, device=cfg.device)
        if cfg.loss_fn == 'focal':
            loss_fn = FocalLoss(alpha=weights)
        elif cfg.loss_fn == 'combo':
            loss_fn = ComboLoss(alpha=weights, gamma=2.0)
        elif cfg.loss_fn == 'dice':
            loss_fn = DiceLoss()
        else:
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        
        if not os.path.isdir(os.path.join(cfg.base_path, "test")):
            logging.error("Test folder not found.")
            exit(0)
        
        test_dl = build_test_dataloader(cfg)

        start_epoch = 0
        global_step = 0
        
        logging.info(f"Starting training for {cfg.epochs} epochs (Seed {seed}).")

        for epoch in range(start_epoch, 600):
            train_loss, current_lr, global_step = train_epoch(
                model, train_dl, optimizer, loss_fn, epoch, scaler, cfg, global_step
            )
            
            logging.info(f"Epoch {epoch}: Train Loss {train_loss:.4f} | LR {current_lr:.8f}")
            
            
            if cfg.wandb_enabled:
                wandb.log({
                    'train_loss': train_loss, 
                    'epoch': epoch, 
                    'lr': current_lr
                })


        logging.info("Testing Last Model (End of Training)")
        test_miou, test_class_iou, test_xs, test_ys, test_preds, test_paths = validate(
            model, test_dl, cfg
        )

        logging.info("Saving last model...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'test_miou': test_miou,
        }, os.path.join(exp_dir, "last_model.pth"))

        if cfg.model_type == "aff":
             logging.info(f"Last Model mIoU: {test_miou['res2'].avg:.4f}")
        else:
             logging.info(f"Last Model mIoU: {test_miou['res2'].avg:.4f}")

        class_iou_str = " | ".join([f"C{i+1}: {iou:.4f}" for i, iou in enumerate(test_class_iou)])
        logging.info(f"Last Model Class-wise IoU: [{class_iou_str}]")

        visualize_results_multiclass(
            test_xs, test_ys, test_preds, test_paths, 
            cfg.num_classes, 0, 12, exp_dir, "TEST_LAST"
        )

        token_path = os.path.join(exp_dir, 'TEST_LAST_token_loc.png')
        # model.visualize_tokens(test_xs.to(cfg.device), 6, token_path, seed=seed)

        if cfg.wandb_enabled:
            wandb.finish()

if __name__ == "__main__":
    main()