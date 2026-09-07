import os

import cv2
import matplotlib
matplotlib.use("Agg")  # never require a display; must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from affmae.data.stats import (  # noqa: E402
    PRETRAIN_IMAGE_MEAN,
    PRETRAIN_IMAGE_STD,
)
from affmae.models.masking import perlin_masking  # noqa: E402
from .primitives import compute_pca_rgb as _compute_pca_rgb  # noqa: E402
from .primitives import denormalize as _denormalize  # noqa: E402

__all__ = ["run_pca_visualization", "render_mae_reconstruction",
           "render_tokens", "render_vit_reconstruction"]


def denormalize(img_tensor):
    """Undo the pretraining normalization and return a 3-channel image.

    The micrographs are single-channel, but ``imshow`` and ``imsave`` accept
    only (H, W), (H, W, 3) or (H, W, 4), so grey is expanded to three identical
    channels.

    Args:
        img_tensor: [B, C, H, W] normalized images.
    Returns:
        [B, 3, H, W] in [0, 1].
    """
    out = _denormalize(img_tensor, mean=PRETRAIN_IMAGE_MEAN,
                       std=PRETRAIN_IMAGE_STD)
    if out.shape[1] == 1:
        out = out.expand(-1, 3, -1, -1)
    return out


def compute_pca_rgb(feats: torch.Tensor) -> np.ndarray:
    """Project decoder features to RGB, z-scoring the channels first.

    Args:
        feats: [N, C] feature vectors.
    Returns:
        [N, 3] float array in [0, 1].
    """
    return _compute_pca_rgb(feats, standardize=True)


def run_pca_visualization(model, images, output_path, device):
    """Render per-decoder-stage features as RGB via PCA, beside the reconstruction.

    Registers a forward hook on the last block of each decoder stage, runs the model
    without masking, projects each stage's token features to three channels and
    paints them back onto the patch grid. Individual cells are also written next to
    ``output_path`` for figure assembly.

    Args:
        model: a pretraining model exposing ``cross_attention_decoder``,
            ``forward_without_masking`` and ``unpatchify``.
        images: [B, C, H, W] normalized inputs; one figure row per image.
        output_path: file to write; per-cell images go beside it.
        device: device the model is on.
    Returns:
        None. Writes ``output_path``.
    """
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


def render_mae_reconstruction(model, images: torch.Tensor, num_vis_images: int, save_path: str, seed: int = 42):
    """Render MAE reconstructions: original, masked input, per-stage predictions.

    Args:
        model: MAE model exposing ``_forward_internal``, ``patchify`` and
            ``unpatchify``. Left in whatever train/eval mode it arrived in.
        images: [B, C, H, W] input batch; the first ``num_vis_images`` are used.
        num_vis_images: int, rows to render.
        save_path: str, grid destination. Individual cells are also written
            alongside it as ``{stem}_row{i}_col{j}{ext}``.
        seed: int, fixes the masking pattern so runs are comparable.
    """
    model.eval()

    # use fixed seed for consistent evaluation masking
    current_rng_state = torch.get_rng_state()
    torch.manual_seed(seed)

    with torch.no_grad():
        vis_images = images[:num_vis_images]
        outputs = model._forward_internal(vis_images)

        ids_restore = outputs['ids_restore']
        ids_keep = outputs['ids_keep']

        # get list of predictions: [pred_res5, pred_res4, pred_res3, pred_res2]
        all_preds_patches = outputs['all_preds']

        original_patches = model.patchify(vis_images)
        visible_patches = torch.gather(original_patches, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, original_patches.shape[-1]))

        # create "Masked Input" image (Visible + Zeros)
        ref_shape = all_preds_patches[0]
        masked_input_patches = torch.cat([visible_patches, torch.zeros_like(ref_shape)], dim=1)
        masked_input_patches_restored = torch.gather(masked_input_patches, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, masked_input_patches.shape[-1]))
        masked_imgs = model.unpatchify(masked_input_patches_restored)

        # create Reconstructions for every stage
        recon_imgs_all = []
        for pred_patches in all_preds_patches:
            all_patches = torch.cat([visible_patches, pred_patches], dim=1)
            all_patches_restored = torch.gather(all_patches, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, all_patches.shape[-1]))
            recon_img = model.unpatchify(all_patches_restored)
            recon_imgs_all.append(recon_img)

    # restore RNG state
    torch.set_rng_state(current_rng_state)

    stage_names = ["Res5", "Res4", "Res2"]
    ncols = 2 + len(recon_imgs_all)

    base_path, ext = os.path.splitext(save_path)

    fig, axes = plt.subplots(nrows=num_vis_images, ncols=ncols, figsize=(4 * ncols, 4 * num_vis_images))
    if num_vis_images == 1: axes = axes.reshape(1, -1)

    def prep_img(tensor_img):
        img = tensor_img.detach().cpu().permute(1, 2, 0)
        return img.numpy()

    for i in range(num_vis_images):
        # original
        img_original = prep_img(vis_images[i])
        axes[i][0].imshow(img_original, cmap="gray")
        axes[i][0].set_title('Original')
        axes[i][0].axis('off')
        # Save individual cell
        cell_name = f"{base_path}_row{i}_col0{ext}"
        plt.imsave(cell_name, img_original.squeeze(), cmap="gray")

        # masked input
        img_masked = prep_img(masked_imgs[i])
        axes[i][1].imshow(img_masked, cmap="gray")
        axes[i][1].set_title('Masked Input')
        axes[i][1].axis('off')
        # save individual images
        cell_name = f"{base_path}_row{i}_col1{ext}"
        plt.imsave(cell_name, img_masked.squeeze(), cmap="gray")

        # stages
        for stage_idx, recon_img_batch in enumerate(recon_imgs_all):
            col_idx = 2 + stage_idx
            name = stage_names[stage_idx] if stage_idx < len(stage_names) else f"Stage {stage_idx}"

            img_recon = prep_img(recon_img_batch[i])
            axes[i][col_idx].imshow(img_recon, cmap="gray")
            axes[i][col_idx].set_title(name)
            axes[i][col_idx].axis('off')

            # save individual images
            cell_name = f"{base_path}_row{i}_col{col_idx}{ext}"
            plt.imsave(cell_name, img_recon.squeeze(), cmap="gray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=75)
    plt.close(fig)


def _patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Flatten an image batch into patch vectors.

    Deliberately not ``model.patchify``: AFFSegmentation has no such method, and
    its ``unpatchify`` reshapes with ``num_classes`` because it exists to turn
    logits into a map. Round-tripping an *image* through that gives the wrong
    channel count, so the pair here derives channels from the tensor and works
    for any model.

    Args:
        imgs: [B, C, H, W].
        patch_size: side of one square patch; must divide H and W.
    Returns:
        [B, (H//p)*(W//p), p*p*C].
    """
    p = patch_size
    b, c, h_dim, w_dim = imgs.shape
    h, w = h_dim // p, w_dim // p
    x = imgs.reshape(b, c, h, p, w, p)
    x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(b, h * w, p * p * c)


def _unpatchify(x: torch.Tensor, patch_size: int, channels: int) -> torch.Tensor:
    """Inverse of :func:`_patchify`.

    Args:
        x: [B, L, p*p*C] patch vectors.
        patch_size: side of one square patch.
        channels: image channels, since L alone cannot recover them.
    Returns:
        [B, C, H, W].
    """
    p = patch_size
    h = w = int(x.shape[1] ** 0.5)
    x = x.reshape(x.shape[0], h, w, p, p, channels)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(x.shape[0], channels, h * p, w * p)


def render_tokens(model, images: torch.Tensor, num_vis_images: int, save_path: str, seed: int = 42):
    """Render adaptive token centres per encoder stage, over the input image.

    Args:
        model: model exposing ``forward_with_pos``/``patchify`` as appropriate.
            Left in whatever train/eval mode it arrived in.
        images: [B, C, H, W] input batch.
        num_vis_images: int, rows to render.
        save_path: str, grid destination.
        seed: int, fixes any masking so runs are comparable.
    """
    model.eval()

    current_rng_state = torch.get_rng_state()
    torch.manual_seed(seed)

    with torch.no_grad():
        vis_images = images[:num_vis_images]
        batch_size = vis_images.shape[0]
        all_pos_items = []
        all_images = []
        for i in range(batch_size):
            img_to_process = vis_images[i:i+1]
            img_patches = _patchify(img_to_process, model.encoder_patch_size)

            # we use 0.0 mask ratio to keep all tokens for visualization
            ids_keep, ids_masked, ids_restore = perlin_masking(img_patches, model.img_size, model.encoder_patch_size, mask_ratio=0.0)

            N, L, D = img_patches.shape
            mask = torch.ones(N, L, 1, device=img_to_process.device)
            if ids_masked.numel() > 0:
                 mask.scatter_(dim=1, index=ids_masked.unsqueeze(-1), value=0.0)

            x_masked_patches = img_patches * mask
            masked_imgs = _unpatchify(x_masked_patches,
                                      model.encoder_patch_size,
                                      img_to_process.shape[1])

            pos, feat, h, w = model.encoder.patch_embed(masked_imgs, ids_masked)

            visible_tokens = torch.gather(feat, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, feat.shape[-1]))
            visible_pos = torch.gather(pos, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 2))

            encoder_stage_outputs = model.encoder.forward_with_pos(visible_tokens, visible_pos, h, w)
            all_pos_items.append([item.cpu().numpy() for item in encoder_stage_outputs])
            all_images.append(masked_imgs)

    torch.set_rng_state(current_rng_state)

    # plotting
    base_path, ext = os.path.splitext(save_path)

    num_stages = len(all_pos_items[0])
    fig, axes = plt.subplots(num_vis_images, num_stages, figsize=(5 * num_stages, 5 * num_vis_images))
    if num_vis_images == 1: axes = np.array([axes])

    for img_idx in range(num_vis_images):
        masked_imgs_batch = all_images[img_idx]
        for stage_idx, pos_tensor in enumerate(all_pos_items[img_idx]):
            img_to_draw_on = masked_imgs_batch[0][0].cpu().numpy().copy()
            img_to_draw_on_scaled = (img_to_draw_on - img_to_draw_on.min()) / (img_to_draw_on.max() - img_to_draw_on.min()) * 255
            img_to_draw_on_uint8 = np.uint8(img_to_draw_on_scaled)
            img_to_draw_on_bgr = cv2.cvtColor(img_to_draw_on_uint8, cv2.COLOR_GRAY2BGR)

            positions = pos_tensor.squeeze(0)
            num_tokens = positions.shape[0]

            for x, y in positions:
                center_x = int(x * model.encoder_patch_size) + model.encoder_patch_size // 2
                center_y = int(y * model.encoder_patch_size) + model.encoder_patch_size // 2
                if 0 <= center_x < img_to_draw_on_bgr.shape[1] and 0 <= center_y < img_to_draw_on_bgr.shape[0]:
                    cv2.circle(img_to_draw_on_bgr, (center_x, center_y), 2, (255, 0, 0), -1)

            # save each individual image
            cell_name = f"{base_path}_row{img_idx}_col{stage_idx}{ext}"
            cv2.imwrite(cell_name, img_to_draw_on_bgr)

            # continue with grid plotting
            img_to_show_rgb = cv2.cvtColor(img_to_draw_on_bgr, cv2.COLOR_BGR2RGB)
            axes[img_idx, stage_idx].imshow(img_to_show_rgb)
            axes[img_idx, stage_idx].set_title(f'Stage {stage_idx+1}: {num_tokens} tokens')
            axes[img_idx, stage_idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=75)
    plt.close(fig)


def render_vit_reconstruction(model, images: torch.Tensor, num_vis_images: int, save_path: str, seed: int=42):
    """Render MAE reconstructions: original, masked input, per-stage predictions.

    Args:
        model: MAE model exposing ``_forward_internal``, ``patchify`` and
            ``unpatchify``. Left in whatever train/eval mode it arrived in.
        images: [B, C, H, W] input batch; the first ``num_vis_images`` are used.
        num_vis_images: int, rows to render.
        save_path: str, grid destination. Individual cells are also written
            alongside it as ``{stem}_row{i}_col{j}{ext}``.
        seed: int, fixes the masking pattern so runs are comparable.
    """
    # use fixed seed for consistent evaluation masking
    current_rng_state = torch.get_rng_state()
    torch.manual_seed(seed)

    import matplotlib.pyplot as plt
    model.eval()
    with torch.no_grad():
        # get a batch of images to visualize
        vis_images = images[:num_vis_images]

        # run the forward pass to get predictions and the mask
        latent, mask, ids_restore = model.forward_encoder(vis_images)
        predicted_patches = model.forward_decoder(latent, ids_restore)

        # the mask is a binary tensor (1 for masked, 0 for visible)
        # we expand it to the patch dimension and use it to zero-out the masked patches
        mask_expanded = mask.unsqueeze(-1).repeat(1, 1, model.patch_size**2 * model.in_chans)
        original_patches = model.patchify(vis_images)
        masked_patches = original_patches.clone()
        masked_patches[mask_expanded.bool()] = 0 # zero-out the masked patches
        masked_imgs = model.unpatchify(masked_patches)

        # we use the mask to combine the original visible patches with the predicted masked patches
        recon_patches = original_patches.clone()
        recon_patches[mask_expanded.bool()] = predicted_patches[mask_expanded.bool()] # Fill in predictions
        recon_imgs = model.unpatchify(recon_patches)

    fig, axes = plt.subplots(nrows=num_vis_images, ncols=3, figsize=(12, num_vis_images * 4))
    if num_vis_images == 1:
        axes = [axes]

    # restore RNG state
    torch.set_rng_state(current_rng_state)

    for i in range(num_vis_images):
        prep_img = lambda x: x.detach().cpu().permute(1, 2, 0).numpy()

        # original
        axes[i][0].imshow(prep_img(vis_images[i]), cmap="gray")
        axes[i][0].set_title('Original')
        axes[i][0].axis('off')

        # masked
        axes[i][1].imshow(prep_img(masked_imgs[i]), cmap="gray")
        axes[i][1].set_title('Masked')
        axes[i][1].axis('off')

        # recon
        axes[i][2].imshow(prep_img(recon_imgs[i]), cmap="gray")
        axes[i][2].set_title('Reconstruction')
        axes[i][2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Visualization saved to {save_path}")


