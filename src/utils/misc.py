import random
from typing import Union
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
import numpy as np
from PIL import Image


class AverageMeter:
    """Compute and store the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, step: int, loss: float, path: str):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model: nn.Module, optimizer: Union[None, torch.optim.Optimizer], path: str) -> tuple[int, int]:
    """Load model checkpoint and return (epoch, step)."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint.get('step', 0)


def cosine_lr_schedule(optimizer: torch.optim.Optimizer, step: int,
                       max_steps: int, lr: float, min_lr: float = 0.,
                       warmup_steps: int = 0):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        lr_scale = step / warmup_steps
    else:
        # handle case where step might exceed max_steps 
        progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
        lr_scale = 0.5 * (1 + np.cos(np.pi * progress))

    current_lr = min_lr + (lr - min_lr) * lr_scale

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return current_lr

def denormalize(img_tensor):
    mean = torch.tensor([0.5562, 0.5562, 0.5562], device=img_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.2396, 0.2396, 0.2396], device=img_tensor.device).view(1, 3, 1, 1)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img

def compute_pca_rgb(feats: torch.Tensor) -> np.ndarray:
    x = feats.detach().cpu().numpy()
    if x.shape[0] < 3:
        return np.zeros((x.shape[0], 3))
        
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0) + 1e-8
    x = (x - x_mean) / x_std

    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(x)
    
    for i in range(3):
        _min = x_pca[:, i].min()
        _max = x_pca[:, i].max()
        if _max - _min > 1e-8:
            x_pca[:, i] = (x_pca[:, i] - _min) / (_max - _min)
        else:
            x_pca[:, i] = 0.5
            
    return x_pca

def generate_distinct_colors(n):
    # Professional muted palette (Okabe-Ito inspired)
    predefined_colors = [
        (0.000, 0.000, 0.000), # 0: Background 
        (0.835, 0.368, 0.000), # 1: Vermillion / Orange
        (0.337, 0.705, 0.913), # 2: Sky Blue
        (0.000, 0.619, 0.450), # 3: Bluish Green
        (0.941, 0.894, 0.258), # 4: Yellow
        (0.000, 0.447, 0.698), # 5: Blue
        (0.800, 0.474, 0.654), # 6: Reddish Purple
        (0.901, 0.623, 0.000), # 7: Orange
        (0.345, 0.239, 0.443), # 8: Dark Purple
        (0.850, 0.325, 0.098)  # 9: Rust
    ]

    if n <= len(predefined_colors):
        return predefined_colors[:n]

    # Fallback to HSV for very high class counts
    colors = []
    for i in range(n):
        h = i / n
        s, v = 0.7, 0.85 
        colors.append(colorsys.hsv_to_rgb(h, s, v))
    return colors

def setup_logging(exp_dir: str):
    """
    sets up logging to both the console and a file in the experiment directory.
    """
    log_file = os.path.join(exp_dir, 'training.log')
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def run_pca_visualization(model, images, output_path, device):
    model.eval()
    batch_size = images.shape[0]
    
    p = model.encoder_patch_size
    h_grid = w_grid = model.img_size // p
    num_total_patches = h_grid * w_grid
    
    activations = {}
    hooks = []

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # hooks
    if hasattr(model, 'cross_attention_decoder'):
        decoder_stages = model.cross_attention_decoder.decoder_blocks
        for i, stage_blocks in enumerate(decoder_stages):
            h = stage_blocks[-1].register_forward_hook(get_activation(f'Stage_{i}'))
            hooks.append(h)
    else:
        print("Error: Model does not have 'cross_attention_decoder'.")
        return

    with torch.no_grad():
        # run Model Inference
        pred_patches = model.forward_without_masking(images)
        
        # if pred_patches has size == num_total_patches, we are in full recon mode.
        imgs_masked = images 
        
        # Since preds are ordered 0..N, we just reshape directly
        imgs_recon = model.unpatchify(pred_patches)
        
        pixel_masks = torch.ones_like(images)
        
        # We need these to be [B, N]
        vis_indices = torch.arange(num_total_patches, device=device).unsqueeze(0).expand(batch_size, -1)

    for h in hooks: h.remove()

    # Processing and Plotting
    sorted_stages = sorted(activations.keys())
    cols = 2 + len(sorted_stages)
    
    # Prepare filename parts for individual saves
    base_path, ext = os.path.splitext(output_path)

    fig, axes = plt.subplots(batch_size, cols, figsize=(4 * cols, 4 * batch_size))
    if batch_size == 1: axes = axes.reshape(1, -1)


    for b in range(batch_size):
        # 1. Original
        img_orig_vis = denormalize(images[b].unsqueeze(0)).squeeze(0).permute(1, 2, 0).cpu().numpy()
        # Clip to ensure valid range for saving
        img_orig_vis = np.clip(img_orig_vis, 0, 1)
        
        axes[b, 0].imshow(img_orig_vis)
        if b == 0: axes[b, 0].set_title("Original")
        axes[b, 0].axis('off')
        
        # Save Individual Cell
        plt.imsave(f"{base_path}_row{b}_col0{ext}", img_orig_vis)

        # 2. Reconstruction
        img_recon_vis = denormalize(imgs_recon[b].unsqueeze(0)).squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_recon_vis = np.clip(img_recon_vis, 0, 1)
        
        axes[b, 1].imshow(img_recon_vis)
        if b == 0: axes[b, 1].set_title("Reconstruction")
        axes[b, 1].axis('off')

        # Save Individual Cell
        plt.imsave(f"{base_path}_row{b}_col1{ext}", img_recon_vis)

        # Mask logic for overlay
        mask_b = pixel_masks[b].permute(1, 2, 0).cpu().numpy()
        mask_b = (mask_b > 0.5).astype(np.float32)
        base_image = img_orig_vis * (1 - mask_b)

        # 3. PCA Stages
        for i, stage_name in enumerate(sorted_stages):
            tokens = activations[stage_name][b] 
            
            rgb_tokens = compute_pca_rgb(tokens)
            rgb_tokens = torch.tensor(rgb_tokens, device=device).float()

            canvas_grid = torch.zeros((1, num_total_patches, 3), device=device)
            
            indices = vis_indices[b]
            scatter_indices = indices.view(1, -1, 1).expand(1, -1, 3)
            
            canvas_grid.scatter_(dim=1, index=scatter_indices, src=rgb_tokens.unsqueeze(0))
            
            canvas_grid = canvas_grid.reshape(1, h_grid, w_grid, 3).permute(0, 3, 1, 2)
            
            pca_upscaled = F.interpolate(
                canvas_grid, 
                size=(model.img_size, model.img_size), 
                mode='nearest'
            ) 
            pca_upscaled = pca_upscaled.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            composite_img = base_image + (pca_upscaled * mask_b)
            composite_img = np.clip(composite_img, 0, 1)
            
            # Plot
            ax = axes[b, 2 + i]
            ax.imshow(composite_img)
            if b == 0: ax.set_title(f"{stage_name}")
            ax.axis('off')

            # Save Individual Cell
            # Col index is offset by 2 (Original + Recon)
            col_idx = 2 + i
            plt.imsave(f"{base_path}_row{b}_col{col_idx}{ext}", composite_img)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_results_multiclass(x, y_true, pred_logits, paths, num_classes, start_idx, num_samples, folder, ep=None):
    """
    Visualize multi-class segmentation results.
    - Column 2: Ground Truth
    - Column 3: Predictions
    - Column 4: Correct predictions colored, incorrect pixels in gray.
    """
    x = x[start_idx : start_idx + num_samples, 0, :, :].cpu()
    y_true = y_true[start_idx : start_idx + num_samples].cpu()
    pred_logits = pred_logits[start_idx : start_idx + num_samples].cpu()
    paths = paths[start_idx : start_idx + num_samples]

    # Get predicted class labels by finding the max logit
    pred_labels = torch.argmax(pred_logits, dim=1) # Shape: [B, H, W]

    # Generate distinct colors for each class (including background)
    base_colors = generate_distinct_colors(num_classes)

    fig, axs = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axs = np.expand_dims(axs, 0)

    for i in range(num_samples):
        axs[i, 0].imshow(x[i], cmap="gray")
        axs[i, 0].axis("off")
        axs[i, 0].set_title(os.path.basename(paths[i]))

        # Ground Truth Overlay
        axs[i, 1].imshow(x[i], cmap="gray")
        gt_overlay = np.zeros((*x[i].shape, 4))

        # if there's only one class we need to actually show it
        num_classes = max(2, num_classes)
        for c in range(1, num_classes): 
            mask = y_true[i] == c
            gt_overlay[mask] = (*base_colors[c], 0.6) # Add color and alpha
        axs[i, 1].imshow(gt_overlay)
        axs[i, 1].set_title("Ground Truth")
        axs[i, 1].axis("off")

        # Prediction Overlay
        axs[i, 2].imshow(x[i], cmap="gray")
        pred_overlay = np.zeros((*x[i].shape, 4))
        for c in range(1, num_classes):
            mask = pred_labels[i] == c
            pred_overlay[mask] = (*base_colors[c], 0.6) # Add color and alpha
        axs[i, 2].imshow(pred_overlay)
        axs[i, 2].set_title("Prediction")
        axs[i, 2].axis("off")

        # Error Map
        axs[i, 3].imshow(x[i], cmap="gray")
        error_overlay = np.zeros((*x[i].shape, 4))

        # Find correct and incorrect pixels
        correct_mask = (pred_labels[i] == y_true[i])
        incorrect_mask = ~correct_mask

        # Color the correctly predicted pixels
        for c in range(1, num_classes):
            correct_class_mask = correct_mask & (pred_labels[i] == c)
            error_overlay[correct_class_mask] = (*base_colors[c], 0.7)

        # Color all incorrect pixels with a semi-transparent gray
        error_overlay[incorrect_mask] = (0.5, 0.5, 0.5, 0.7)

        axs[i, 3].imshow(error_overlay)
        axs[i, 3].set_title("Correct (Color) vs Incorrect (Gray)")
        axs[i, 3].axis("off")

    plt.tight_layout()
    if ep is not None:
        plt.savefig(f"{folder}/val_results_multiclass_{ep}.png", dpi=150)
    else:
        plt.savefig(f"{folder}/val_results_multiclass.png", dpi=150)
    plt.close(fig)


def visualize_model_comparison_zoom(x, y_true, pred_logits_1, pred_logits_2, model_1_name, model_2_name, num_classes, indices, folder, zoom_boxes=None, filename="model_comparison.png"):
    """
    4-Column layout: [M1 Pred] | [M1 Zoom] | [M2 Pred] | [M2 Zoom]
    Errors are highlighted in bright pink.
    """
    x = x[indices, 0, :, :].cpu()
    y_true = y_true[indices].cpu()
    pred_1 = pred_logits_1[indices].cpu()
    pred_2 = pred_logits_2[indices].cpu()
    
    num_samples = len(indices)
    labels_1 = torch.argmax(pred_1, dim=1)
    labels_2 = torch.argmax(pred_2, dim=1)
    base_colors = generate_distinct_colors(num_classes)
    
    _, H, W = x.shape
    if zoom_boxes is None:
        zoom_boxes = [(W//2 - 25, H//2 - 25, 50, 50)] * num_samples

    # Exactly 4 columns
    fig, axs = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axs = np.expand_dims(axs, 0)

    for i in range(num_samples):
        # reduce contrast
        img_np = x[i].numpy()
        img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_bg = img_norm * 0.5 + 0.3 

        overlay_1 = np.zeros((*x[i].shape, 4))
        overlay_2 = np.zeros((*x[i].shape, 4))
        
        correct_1 = (labels_1[i] == y_true[i])
        correct_2 = (labels_2[i] == y_true[i])

        for c in range(1, max(2, num_classes)):
            c_mask_1 = correct_1 & (labels_1[i] == c)
            overlay_1[c_mask_1] = (*base_colors[c], 0.65)
            
            c_mask_2 = correct_2 & (labels_2[i] == c)
            overlay_2[c_mask_2] = (*base_colors[c], 0.65)

        pink_color = (1.0, 0.0, 1.0, 0.85) 
        overlay_1[~correct_1] = pink_color
        overlay_2[~correct_2] = pink_color

        plot_configs = [
            (0, overlay_1, False, model_1_name),
            (1, overlay_1, True,  model_1_name + " Zoom"),
            (2, overlay_2, False, model_2_name),
            (3, overlay_2, True,  model_2_name + " Zoom")
        ]

        zx, zy, zw, zh = zoom_boxes[i]
        box_color = '#181A18' 

        for ax_idx, overlay, is_zoom, title in plot_configs:
            ax = axs[i, ax_idx]
            ax.imshow(img_bg, cmap="gray", vmin=0, vmax=1)
            ax.imshow(overlay)

            if is_zoom:
                # apply zoom boundaries
                ax.set_xlim(zx, zx + zw)
                ax.set_ylim(zy + zh, zy)
            else:
                # add bounding box
                rect = patches.Rectangle((zx, zy), zw, zh, linewidth=2.5, edgecolor=box_color, facecolor='none')
                ax.add_patch(rect)
                
                # connection lines linking this plot to the zoomed plot
                con_top = ConnectionPatch(xyA=(zx+zw, zy), coordsA="data", xyB=(0, 1), coordsB="axes fraction", axesA=ax, axesB=axs[i, ax_idx+1], color=box_color, linewidth=1.5, linestyle=':')
                con_bot = ConnectionPatch(xyA=(zx+zw, zy+zh), coordsA="data", xyB=(0, 0), coordsB="axes fraction", axesA=ax, axesB=axs[i, ax_idx+1], color=box_color, linewidth=1.5, linestyle=':')
                ax.add_artist(con_top)
                ax.add_artist(con_bot)

        for c, ax in enumerate(axs[i]):
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if i == 0:
                titles = [model_1_name, "Zoom Detail", model_2_name, "Zoom Detail"]
                # ax.set_title(titles[c], fontsize=16, pad=10)

    plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.02, right=0.98, top=0.95, bottom=0.02)
    plt.savefig(os.path.join(folder, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
