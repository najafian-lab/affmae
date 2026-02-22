import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
from torchvision import transforms

from src.config import load_config
from src.models.aff_mae import AFFMaskedAutoEncoder
from src.utils.misc import set_seed

from src.data import create_dataloader


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

def denormalize(img_tensor):
    mean = torch.tensor([0.5562, 0.5562, 0.5562], device=img_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.2396, 0.2396, 0.2396], device=img_tensor.device).view(1, 3, 1, 1)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img


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

    try:
        with torch.no_grad():
            # run Model Inference
            outputs = model._forward_internal(images)
            
            pred_patches = outputs['pred_masked']
            ids_keep = outputs['ids_keep']
            ids_masked = outputs['ids_masked']
            ids_restore = outputs['ids_restore']

            # if pred_patches has size == num_total_patches, we are in full recon mode.
            # in this mode, the predictions are ordered      
            is_full_recon = (pred_patches.shape[1] == num_total_patches)

            if is_full_recon:
                print(f"Full Reconstruction Mode detected ({pred_patches.shape[1]} tokens). Ignoring IDs.")
                
                imgs_masked = images 
                
                # Since preds are ordered 0..N, we just reshape directly
                imgs_recon = model.unpatchify(pred_patches)
                
                pixel_masks = torch.ones_like(images)
                
                # We need these to be [B, N]
                vis_indices = torch.arange(num_total_patches, device=device).unsqueeze(0).expand(batch_size, -1)
                
            else:
                # Standard MAE Logic
                patch_size = model.encoder_patch_size
                original_patches = model.patchify(images)
                visible_patches = torch.gather(original_patches, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, original_patches.shape[-1]))

                masked_input_patches = torch.cat([visible_patches, torch.zeros_like(pred_patches)], dim=1)
                masked_input_restored = torch.gather(masked_input_patches, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, masked_input_patches.shape[-1]))
                imgs_masked = model.unpatchify(masked_input_restored)

                all_patches = torch.cat([visible_patches, pred_patches], dim=1)
                all_patches_restored = torch.gather(all_patches, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, all_patches.shape[-1]))
                imgs_recon = model.unpatchify(all_patches_restored)
                
                ones_masked = torch.ones_like(pred_patches)
                zeros_keep = torch.zeros_like(visible_patches)
                combined_binary = torch.cat([zeros_keep, ones_masked], dim=1)
                combined_binary_restored = torch.gather(combined_binary, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, combined_binary.shape[-1]))
                pixel_masks = model.unpatchify(combined_binary_restored)
                
                vis_indices = ids_masked.long()
            
    finally:
        for h in hooks: h.remove()

    # Processing and Plotting
    sorted_stages = sorted(activations.keys())
    cols = 3 + len(sorted_stages)
    fig, axes = plt.subplots(batch_size, cols, figsize=(4 * cols, 4 * batch_size))
    if batch_size == 1: axes = axes.reshape(1, -1)

    print(f"Generating visualization for {batch_size} images...")

    for b in range(batch_size):
        # Original
        img_orig_vis = denormalize(images[b].unsqueeze(0)).squeeze(0).permute(1, 2, 0).cpu().numpy()
        axes[b, 0].imshow(img_orig_vis)
        if b == 0: axes[b, 0].set_title("Original")
        axes[b, 0].axis('off')

        # Masked Input
        img_masked_vis = denormalize(imgs_masked[b].unsqueeze(0)).squeeze(0).permute(1, 2, 0).cpu().numpy()
        axes[b, 1].imshow(img_masked_vis)
        if b == 0: axes[b, 1].set_title("Masked Input")
        axes[b, 1].axis('off')

        # Reconstruction
        img_recon_vis = denormalize(imgs_recon[b].unsqueeze(0)).squeeze(0).permute(1, 2, 0).cpu().numpy()
        axes[b, 2].imshow(img_recon_vis)
        if b == 0: axes[b, 2].set_title("Reconstruction")
        axes[b, 2].axis('off')

        # Mask logic for overlay
        mask_b = pixel_masks[b].permute(1, 2, 0).cpu().numpy()
        mask_b = (mask_b > 0.5).astype(np.float32)
        base_image = img_orig_vis * (1 - mask_b)

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
            
            ax = axes[b, 3 + i]
            ax.imshow(np.clip(composite_img, 0, 1))
            if b == 0: ax.set_title(f"{stage_name}")
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization successfully saved to: {output_path}")


def main():
    CONFIG_PATH = "/homes/iws/ziawang/Documents/lab/affmae_root/affmae_pretraining/configs/aff_smallv2.yaml"
    OUTPUT_PATH = "/homes/iws/ziawang/Documents/lab/affmae_root/output/AFF_NEW_CODE_TEST_LOW_CLAHE/pca_visualization_batch0.png"
    MODEL_PATH = "/homes/iws/ziawang/Documents/lab/affmae_root/output/AFF_NEW_CODE_TEST_LOW_CLAHE/checkpoints/ckpt_epoch_last.pth"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    BATCH_SKIP = 6       
    IMAGES_TO_VIS = 6
    
    # FOR FULL RECON AFFMAE.PY GOTTA BE MODIFIED
    OVERRIDE_MASK_RATIO = 0.0  # Set to 0.0 for full reconstruction

    print(f"Loading config from: {CONFIG_PATH}")
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config file not found")
        return

    config = load_config(CONFIG_PATH)
    config.device = DEVICE
    
    print(f"Initializing model type: {config.model_type}")
    device = torch.device(DEVICE)
    
    if config.model_type == 'aff':
        model = AFFMaskedAutoEncoder(
            patch_size=config.patch_size,
            img_size=config.img_size,
            in_channels=config.in_channels,
            mask_ratio=config.mask_ratio,
            encoder_embed_dim=config.aff_embed_dims,
            encoder_depth=config.aff_depths,
            encoder_num_heads=config.aff_num_heads,
            encoder_nbhd_size=config.aff_nbhd_sizes,
            global_attention=config.aff_global_attention,
            cluster_size=config.aff_cluster_size,
            ds_rate=config.aff_ds_rates,
            decoder_embed_dim=config.decoder_embed_dim,
            decoder_depth=config.decoder_depth,
            decoder_num_heads=config.decoder_num_heads,
            decoder_use_cls_token=config.decoder_use_cls_token,
            mlp_ratio=config.aff_mlp_ratio,
            alpha=config.aff_alpha,
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
            mlp_ratio=config.mlp_ratio,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type {config.model_type}")

    if OVERRIDE_MASK_RATIO is not None:
        print(f"Overriding config mask_ratio {model.mask_ratio} -> {OVERRIDE_MASK_RATIO}")
        model.mask_ratio = OVERRIDE_MASK_RATIO

    print(f"Loading checkpoint from: {MODEL_PATH}")
    if os.path.isfile(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print(f"Checkpoint file not found at {MODEL_PATH}")
        return

    print("Initializing Dataloader...")
    dataloader = create_dataloader(config)

    print(f"Skipping {BATCH_SKIP} batches...")
    data_iter = iter(dataloader)
    try:
        for _ in range(BATCH_SKIP):
            _ = next(data_iter)
        batch_images, _ = next(data_iter)
    except StopIteration:
        print("Error: Dataloader ran out of data.")
        return

    actual_bs = batch_images.shape[0]
    num_to_vis = min(actual_bs, IMAGES_TO_VIS)
    print(f"Visualizing {num_to_vis} images...")
    images = batch_images[:num_to_vis].to(device)

    run_pca_visualization(model, images, OUTPUT_PATH, device)

if __name__ == "__main__":
    main()