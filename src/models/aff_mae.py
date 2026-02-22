import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import os

from src.layers.aff import AFFEncoder
from src.layers.decoder import CrossAttentionPixelDecoder
from src.utils.masking import perlin_masking, random_masking

from dataclasses import dataclass
from typing import Optional


@dataclass
class ShapeSpec:
    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None


class AFFMaskedAutoEncoder(nn.Module):
    """
    Main Model Class.
    """
    def __init__(self, patch_size=8, img_size=384, in_channels=1, mask_ratio=0.5,
                 encoder_embed_dim=[64, 128, 256, 448],
                 encoder_depth=[3, 3, 7, 4],
                 encoder_num_heads=[2, 4, 8, 14],
                 encoder_nbhd_size=[64, 64, 64, 64],
                 global_attention=False, cluster_size=8, ds_rate=[0.4, 0.4, 0.4, 0.4],
                 decoder_embed_dim=384, decoder_num_heads=6,
                 mlp_ratio=2, alpha=10.0, merging_method="l2norm"):
        super().__init__()
        
        self.img_size = img_size
        self.mask_ratio = mask_ratio
        self.encoder_patch_size = patch_size
        self.in_channels = in_channels

        self.encoder = AFFEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=in_channels,
            embed_dims=encoder_embed_dim, depths=encoder_depth,
            num_heads=encoder_num_heads, nbhd_sizes=encoder_nbhd_size,
            cluster_size=cluster_size, ds_rates=ds_rate,
            global_attention=global_attention, alpha=alpha, mlp_ratio=mlp_ratio,
            merging_method=merging_method
        )

        # define shapes for the decoder to know what to expect from encoder
        decoder_input_shape = {
            "res2": ShapeSpec(channels=encoder_embed_dim[0], stride=8),
            "res3": ShapeSpec(channels=encoder_embed_dim[1], stride=16),
            "res4": ShapeSpec(channels=encoder_embed_dim[2], stride=32),
            "res5": ShapeSpec(channels=encoder_embed_dim[3], stride=64),
        }

        self.cross_attention_decoder = CrossAttentionPixelDecoder(
            input_shape=decoder_input_shape,
            transformer_dropout=0.1,
            transformer_nheads=decoder_num_heads,
            transformer_dim_feedforward=decoder_embed_dim * 2,
            transformer_dec_layers=1,
            conv_dim=decoder_embed_dim,
            mask_dim=decoder_embed_dim,
            transformer_in_features=["res2", "res3", "res4", "res5"],
            common_stride=8,
            shepard_power=2.0,
            shepard_power_learnable=True
        )

        self.masked_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.masked_token, std=0.02)

        self.decoder_pred_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred_head = nn.Linear(decoder_embed_dim, (patch_size**2) * in_channels)
        
        # aux heads for deep supervision
        self.aux_head_res5 = self._make_aux_head(decoder_embed_dim, patch_size, in_channels)
        self.aux_head_res4 = self._make_aux_head(decoder_embed_dim, patch_size, in_channels)

    def _make_aux_head(self, dim, patch_size, in_chans):
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, (patch_size**2) * in_chans)
        )

    def patchify(self, imgs):
        """Standard patchify logic"""
        p = self.encoder_patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], self.in_channels, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(imgs.shape[0], h * w, p**2 * self.in_channels)

    def unpatchify(self, x):
        """Standard unpatchify logic"""
        p = self.encoder_patch_size
        h = w = int(x.shape[1] ** 0.5)
        c = self.in_channels
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward_encoder(self, x):
        img_patches = self.patchify(x)
        
        with torch.amp.autocast("cuda"):
            # ids_keep, ids_masked, ids_restore = random_masking(img_patches, self.mask_ratio)
            ids_keep, ids_masked, ids_restore = perlin_masking(img_patches, self.img_size, self.encoder_patch_size, self.mask_ratio)
        
        # we are masking at the image level, not after patch embedding 
        # the patch embedding has a stride size > patch_size, like convmae
        # if masking was not done before patch embed, information from visible patches would leak
        N, L, D = img_patches.shape
        mask = torch.ones(N, L, 1, device=x.device)
        mask.scatter_(dim=1, index=ids_masked.unsqueeze(-1), value=0.0)
        x_masked_patches = img_patches * mask
        x_with_mask = self.unpatchify(x_masked_patches)

        pos, feat, h, w = self.encoder.patch_embed(x_with_mask, ids_masked)
        
        # gather visible tokens
        visible_tokens = torch.gather(feat, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, feat.shape[-1]))
        visible_pos = torch.gather(pos, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 2))
        
        # get features
        encoder_features = self.encoder(visible_tokens, visible_pos, h, w)

        return encoder_features

    def _forward_internal(self, x):
        img_patches = self.patchify(x)
        
        with torch.amp.autocast(device_type="cuda", enabled=False):
            # ids_keep, ids_masked, ids_restore = random_masking(img_patches, self.mask_ratio)
            ids_keep, ids_masked, ids_restore = perlin_masking(img_patches, self.img_size, self.encoder_patch_size, self.mask_ratio)
        
        # we are masking at the image level, not after patch embedding 
        # the patch embedding has a stride size > patch_size, like convmae
        # if masking was not done before patch embed, information from visible patches would leak
        N, L, D = img_patches.shape
        mask = torch.ones(N, L, 1, device=x.device)
        mask.scatter_(dim=1, index=ids_masked.unsqueeze(-1), value=0.0)
        x_masked_patches = img_patches * mask
        x_with_mask = self.unpatchify(x_masked_patches)

        pos, feat, h, w = self.encoder.patch_embed(x_with_mask, ids_masked)
        
        # gather visible tokens
        visible_tokens = torch.gather(feat, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, feat.shape[-1]))
        visible_pos = torch.gather(pos, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 2))
        
        # get features
        encoder_features = self.encoder(visible_tokens, visible_pos, h, w)

        masked_pos = torch.gather(pos, dim=1, index=ids_masked.unsqueeze(-1).repeat(1, 1, 2))
        batch_size, num_masked = masked_pos.shape[:2]
        masked_tokens = self.masked_token.expand(batch_size, num_masked, -1)

        predicted_tokens = self.cross_attention_decoder(encoder_features, masked_tokens, masked_pos)

        pred_res5 = self.aux_head_res5(predicted_tokens[0])
        pred_res4 = self.aux_head_res4(predicted_tokens[1])
        pred_res2 = self.decoder_pred_head(self.decoder_pred_norm(predicted_tokens[3]))

        masked_patches_gt = torch.gather(img_patches, dim=1, index=ids_masked.unsqueeze(-1).repeat(1, 1, img_patches.shape[2]))
        
        loss_res5 = F.mse_loss(pred_res5, masked_patches_gt)
        loss_res4 = F.mse_loss(pred_res4, masked_patches_gt)
        loss_res2 = F.mse_loss(pred_res2, masked_patches_gt)
        
        loss = loss_res5*0.05 + loss_res4*0.12 + loss_res2

        return {
            'loss': loss, 
            'all_losses': [loss_res5, loss_res4, loss_res2],
            'pred_masked': pred_res2, 
            'gt_all': img_patches, 
            'ids_keep': ids_keep, 
            'ids_masked': ids_masked, 
            'ids_restore': ids_restore,
            'all_preds': [pred_res5, pred_res4, pred_res2] 
        }
    
    # debug method used for pca visualization
    def forward_without_masking(self, x: torch.Tensor) -> torch.Tensor:
        pos, feat, h, w = self.encoder.patch_embed(x, torch.empty(x.shape[0], 0, device=x.device))
        
        # get features
        encoder_features = self.encoder(feat, pos, h, w)

        batch_size, num_masked = pos.shape[:2]
        all_context_tokens = self.masked_token.expand(batch_size, num_masked, -1)

        predicted_tokens = self.cross_attention_decoder(encoder_features, all_context_tokens, pos)
        pred_res2 = self.decoder_pred_head(self.decoder_pred_norm(predicted_tokens[3]))

        return pred_res2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self._forward_internal(x)
        return outputs['loss'], outputs['all_losses']

    def visualize(self, images: torch.Tensor, num_vis_images: int, save_path: str, seed: int = 42):
        self.eval()

        # use fixed seed for consistent evaluation masking
        current_rng_state = torch.get_rng_state()
        torch.manual_seed(seed)

        with torch.no_grad():
            vis_images = images[:num_vis_images]
            outputs = self._forward_internal(vis_images)
            
            ids_restore = outputs['ids_restore']
            ids_keep = outputs['ids_keep']
            
            # get list of predictions: [pred_res5, pred_res4, pred_res3, pred_res2]
            all_preds_patches = outputs['all_preds']
            
            original_patches = self.patchify(vis_images)
            visible_patches = torch.gather(original_patches, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, original_patches.shape[-1]))
            
            # create "Masked Input" image (Visible + Zeros)
            ref_shape = all_preds_patches[0]
            masked_input_patches = torch.cat([visible_patches, torch.zeros_like(ref_shape)], dim=1)
            masked_input_patches_restored = torch.gather(masked_input_patches, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, masked_input_patches.shape[-1]))
            masked_imgs = self.unpatchify(masked_input_patches_restored)

            # create Reconstructions for every stage
            recon_imgs_all = []
            for pred_patches in all_preds_patches:
                all_patches = torch.cat([visible_patches, pred_patches], dim=1)
                all_patches_restored = torch.gather(all_patches, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, all_patches.shape[-1]))
                recon_img = self.unpatchify(all_patches_restored)
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
        self.train()

    def visualize_tokens(self, images: torch.Tensor, num_vis_images: int, save_path: str, seed: int = 42):
        self.eval()

        current_rng_state = torch.get_rng_state()
        torch.manual_seed(seed)

        with torch.no_grad():
            vis_images = images[:num_vis_images]
            batch_size = vis_images.shape[0]
            all_pos_items = []
            all_images = []
            for i in range(batch_size):
                img_to_process = vis_images[i:i+1]
                img_patches = self.patchify(img_to_process)
                
                # we use 0.0 mask ratio to keep all tokens for visualization
                ids_keep, ids_masked, ids_restore = perlin_masking(img_patches, self.img_size, self.encoder_patch_size, mask_ratio=0.0)
                
                N, L, D = img_patches.shape
                mask = torch.ones(N, L, 1, device=img_to_process.device)
                if ids_masked.numel() > 0:
                     mask.scatter_(dim=1, index=ids_masked.unsqueeze(-1), value=0.0)
                
                x_masked_patches = img_patches * mask
                masked_imgs = self.unpatchify(x_masked_patches)

                pos, feat, h, w = self.encoder.patch_embed(masked_imgs, ids_masked)
                
                visible_tokens = torch.gather(feat, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, feat.shape[-1]))
                visible_pos = torch.gather(pos, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 2))
                
                encoder_stage_outputs = self.encoder.forward_with_pos(visible_tokens, visible_pos, h, w)
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
                    center_x = int(x * self.encoder_patch_size) + self.encoder_patch_size // 2
                    center_y = int(y * self.encoder_patch_size) + self.encoder_patch_size // 2
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
        self.train()
        