import os
import sys
import argparse
import logging
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.finetune_dataset import build_finetune_dataloader
from src.models.aff_segmentation import AFFSegmentation
from src.models.vit_mae import VanillaViTMAE
from src.utils.misc import set_seed

CONFIG_PATH = "/homes/iws/ziawang/Documents/lab/affmae_root/affmae/configs/aff_small_finetune_l2norm.yaml"
CKPT_PATH = "/homes/iws/ziawang/Documents/lab/affmae_root/output/PTC_AFF_FT_l2norm/best_model.pth"
OUTPUT_DIR = "/homes/iws/ziawang/Documents/lab/affmae_root/output/PTC_AFF_FT_l2norm"

# Visualization Settings
NUM_GRID_ROWS = 10  # 10 Worst samples
NUM_GRID_COLS = 1  
NUM_PCA_IMAGES = 6
NUM_TOKEN_IMAGES = 6

def one_minus_iou_batch(logits, targets, num_classes, smooth=1e-6):
    """
    Computes 1-IoU per sample in a batch.
    Returns: (B,) tensor of scores, plus raw intersection/union for global stats.
    """
    # pred labels
    pred_labels = torch.argmax(logits, dim=1)  # (B, H, W)
    
    # One-hot encode (B, C, H, W)
    pred_oh = F.one_hot(pred_labels, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_oh = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Exclude BG (Class 0)
    pred_no_bg = pred_oh[:, 1:, :, :]
    target_no_bg = target_oh[:, 1:, :, :]
    
    # Sum over H, W (dims 2, 3)
    intersection = (pred_no_bg * target_no_bg).sum(dim=(2, 3)) # (B, C-1)
    total = pred_no_bg.sum(dim=(2, 3)) + target_no_bg.sum(dim=(2, 3))
    union = total - intersection # (B, C-1)
    
    # Calculate per-sample metric for sorting: Average 1-IoU across channels
    iou_per_channel = (intersection + smooth) / (union + smooth)
    
    # Mean across channels for the single metric
    sample_scores = 1.0 - iou_per_channel.mean(dim=1) # (B,)
    
    return sample_scores, intersection, union

def denormalize(img_tensor):
    mean = torch.tensor([0.6266], device=img_tensor.device).view(1, 1, 1, 1)
    std = torch.tensor([0.2259], device=img_tensor.device).view(1, 1, 1, 1)
    img = img_tensor * std + mean
    return img

def get_class_colors(num_classes):
    cmap = [
        [0, 0, 0],       
        [1, 0, 0],       
        [0, 1, 0],     
        [0, 0, 1],     
        [1, 1, 0],   
        [0, 1, 1],   
    ]
    return np.array(cmap[:num_classes])

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Loading config from {CONFIG_PATH}")
    cfg = load_config(CONFIG_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(77)

    logger.info(f"Initializing model type: {cfg.model_type}")
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
            mlp_ratio=cfg.aff_mlp_ratio,
            alpha=cfg.aff_alpha,
            num_classes=cfg.num_classes
        )
    elif cfg.model_type == 'vit':
        model = VanillaViTMAE(
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
        )

    model.to(device)

    logger.info(f"Loading weights from {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    val_loader = build_finetune_dataloader(cfg, is_train=False)

    logger.info("Running validation loop...")
    
    stored_samples = []
    
    global_inter = torch.zeros(cfg.num_classes - 1, device=device)
    global_union = torch.zeros(cfg.num_classes - 1, device=device)
    
    avg_sample_score_sum = 0.0

    with torch.no_grad():
        for images, targets, paths in tqdm(val_loader):
            images = images.to(device)
            targets = targets.to(device).long()
            
            logits = model(images)
            
            # Calculate Metrics
            sample_scores, inter, union = one_minus_iou_batch(logits, targets, cfg.num_classes)
            
            # Accumulate
            avg_sample_score_sum += sample_scores.sum().item()
            global_inter += inter.sum(dim=0)
            global_union += union.sum(dim=0)
            
            # Store data for visualization (CPU)
            img_cpu = denormalize(images).cpu()
            
            for i in range(images.shape[0]):
                stored_samples.append({
                    'score': sample_scores[i].item(),
                    'image': img_cpu[i],
                    'target': targets[i].cpu(),
                    'logits': logits[i].cpu(),
                    'path': paths[0][i]
                })
    
    smooth = 1e-6
    global_ious = (global_inter + smooth) / (global_union + smooth)
    global_1_ious = 1.0 - global_ious

    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    print(f"Sample-Averaged 1-IoU (Matches Training Log): {global_1_ious.mean():.4f}")
    print("-" * 60)
    print("Global 1-IoU (Dataset-wide Aggregation):")
    for idx, val in enumerate(global_1_ious):
        print(f"  Class {idx+1}: {val.item():.4f}")
    print("="*60 + "\n")

    logger.info("Generating sorted prediction grid...")
    
    stored_samples.sort(key=lambda x: x['score'], reverse=True)
    
    viz_samples = stored_samples[:10]
    
    fig, axes = plt.subplots(10, 4, figsize=(16, 40))
    if len(viz_samples) == 1: axes = np.expand_dims(axes, 0)
    
    colors = get_class_colors(cfg.num_classes)
    
    for i, ax_row in enumerate(axes):
        if i >= len(viz_samples):
            for ax in ax_row: ax.axis('off')
            continue
            
        sample = viz_samples[i]
        
        img = sample['image'].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        if img.shape[2] == 1: img = np.concatenate([img]*3, axis=2)

        pred_labels = torch.argmax(sample['logits'], dim=0).numpy()
        target_labels = sample['target'].numpy()
        
        ax_row[0].imshow(img)
        ax_row[0].set_title(f"Original\n1-IoU: {sample['score']:.3f}")
        ax_row[0].axis('off')

        def make_overlay(base, labels, alpha=0.6):
            ov = base.copy()
            for c in range(1, cfg.num_classes):
                mask = (labels == c)
                if mask.any():
                    # Blend: color * alpha + base * (1-alpha)
                    colored_mask = np.zeros_like(base)
                    colored_mask[mask] = colors[c]
                    ov[mask] = colored_mask[mask] * alpha + base[mask] * (1-alpha)
            return np.clip(ov, 0, 1)

        gt_viz = make_overlay(img, target_labels)
        ax_row[1].imshow(gt_viz)
        ax_row[1].set_title("Ground Truth")
        ax_row[1].axis('off')

        pred_viz = make_overlay(img, pred_labels)
        ax_row[2].imshow(pred_viz)
        ax_row[2].set_title("Prediction")
        ax_row[2].axis('off')

        incorrect_mask = (pred_labels != target_labels)
        
        err_viz = img.copy()
        grey_color = np.array([0.8, 0.8, 0.8])
        
        if incorrect_mask.any():
            err_viz[incorrect_mask] = grey_color
        
        ax_row[3].imshow(err_viz)
        ax_row[3].set_title("Incorrect (Grey)")
        ax_row[3].axis('off')

    plt.tight_layout()
    grid_path = os.path.join(OUTPUT_DIR, "validation_worst_10.png")
    plt.savefig(grid_path, dpi=150)
    plt.close()
    logger.info(f"Saved prediction grid to {grid_path}")
    logger.info("Generating Token Visualization (Encoder)...")
    
    random.seed(77)
    viz_indices = random.sample(range(len(stored_samples)), min(len(stored_samples), NUM_TOKEN_IMAGES))
    viz_batch = torch.stack([stored_samples[idx]['image'] for idx in viz_indices]).to(device)
    
    with torch.no_grad():
        # Encoder Forward to get positions
        pos, feat, h, w = model.encoder.patch_embed(viz_batch, ids_masked=None)
        features_dict = model.encoder(feat, pos, h, w)
    
    stages = [k for k in features_dict.keys() if k.endswith("_pos")]
    stages.sort()
    
    fig, axes = plt.subplots(len(viz_indices), len(stages), figsize=(4*len(stages), 4*len(viz_indices)))
    if len(viz_indices) == 1: axes = np.array([axes])
    
    patch_size = cfg.patch_size
    
    for i in range(len(viz_indices)):
        base = viz_batch[i].permute(1, 2, 0).cpu().numpy()
        base = (base - base.min()) / (base.max() - base.min() + 1e-6)
        base = (base * 255).astype(np.uint8)
        if base.shape[2] == 1: base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
            
        for j, stage_key in enumerate(stages):
            ax = axes[i, j]
            canvas = base.copy()
            
            pos_tensor = features_dict[stage_key][i].cpu().numpy()
            for (x, y) in pos_tensor:
                cx = int(x * patch_size) + patch_size // 2
                cy = int(y * patch_size) + patch_size // 2
                if 0 <= cx < canvas.shape[1] and 0 <= cy < canvas.shape[0]:
                    cv2.circle(canvas, (cx, cy), 2, (255, 0, 0), -1)
            
            ax.imshow(canvas)
            ax.set_title(f"{stage_key.replace('_pos', '')}: {len(pos_tensor)}")
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tokens.png"), dpi=100)
    plt.close()

    logger.info("Generating PCA Visualization (Decoder Stages)...")
    
    activations = {}
    hooks = []
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    decoder = model.cross_attention_decoder
    for i, stage_blocks in enumerate(decoder.decoder_blocks):
        hooks.append(stage_blocks[-1].register_forward_hook(get_activation(f'Decoder_Stage_{i}')))

    with torch.no_grad():
        _ = model(viz_batch)

    for h in hooks: h.remove()
    
    sorted_keys = sorted(activations.keys())
    
    h_grid = w_grid = model.img_size // patch_size
    
    fig, axes = plt.subplots(len(viz_indices), 1 + len(sorted_keys), figsize=(4*(1+len(sorted_keys)), 4*len(viz_indices)))
    if len(viz_indices) == 1: axes = np.array([axes])

    for i in range(len(viz_indices)):
        base = viz_batch[i].permute(1, 2, 0).cpu().numpy()
        base = (base - base.min()) / (base.max() - base.min() + 1e-6)
        axes[i, 0].imshow(base, cmap='gray')
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
        
        for k_idx, key in enumerate(sorted_keys):
            feats = activations[key][i].cpu().numpy()
            
            N_tokens = feats.shape[0]
            side = int(np.sqrt(N_tokens))
            
            # PCA
            if feats.shape[0] > 3:
                f_mean = feats.mean(0)
                f_std = feats.std(0) + 1e-6
                feats_norm = (feats - f_mean) / f_std
                
                pca = PCA(n_components=3)
                pca_proj = pca.fit_transform(feats_norm)
                
                pca_rgb = np.zeros_like(pca_proj)
                for c in range(3):
                    c_min, c_max = pca_proj[:,c].min(), pca_proj[:,c].max()
                    if c_max - c_min > 1e-8:
                        pca_rgb[:,c] = (pca_proj[:,c] - c_min) / (c_max - c_min)
                    else:
                        pca_rgb[:,c] = 0.5
            else:
                pca_rgb = np.zeros((feats.shape[0], 3))

            pca_grid = pca_rgb.reshape(side, side, 3)
            
            pca_big = cv2.resize(pca_grid, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_NEAREST)
            
            axes[i, k_idx+1].imshow(pca_big)
            axes[i, k_idx+1].set_title(key)
            axes[i, k_idx+1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_decoder.png"), dpi=100)
    plt.close()
    
    logger.info("Analysis Complete.")

if __name__ == "__main__":
    main()