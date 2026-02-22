# vanilla_vit_mae.py
from functools import partial
import torch
import torch.nn as nn
import numpy as np
from src.utils.pos_embed import get_2d_sincos_pos_embed
from src.utils.masking import perlin_masking, random_masking

from timm.models.vision_transformer import PatchEmbed, Block


class VanillaViTMAE(nn.Module):
    """
    Classic Masked Autoencoder with VisionTransformer backbone as described in the MAE paper.
    """
    def __init__(self, img_size=224, mask_ratio=0.65, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.img_size = img_size

        # MAE Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.encoder_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.encoder_norm = norm_layer(embed_dim)

        # MAE Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])

        self.decoder_pred_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred_head = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # positional embeddings
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # patch embedding
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # CLS and mask tokens
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # other layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """imgs: (N, C, H, W) -> x: (N, L, patch_size**2 * C)"""
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """x: (N, L, patch_size**2 * C) -> imgs: (N, C, H, W)"""
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, w * p))
        return imgs

    def forward_encoder(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        N, L, D = x.shape

        # ids_keep, ids_masked, ids_restore = random_masking(x, self.mask_ratio)
        ids_keep, ids_masked, ids_restore  = perlin_masking(x, self.img_size, self.patch_size, self.mask_ratio)
        
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :ids_keep.shape[1]] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no CLS token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append CLS token

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_pred_norm(x)

        x = self.decoder_pred_head(x)
        x = x[:, 1:, :] # remove CLS token
        return x

    def forward_loss(self, imgs, pred, mask):
        """Calculate loss on the masked patches."""
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) # loss per patch

        loss = (loss * mask).sum() / mask.sum() # mean loss on masked patches
        return loss

    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, [] 

    def visualize(self, images: torch.Tensor, num_vis_images: int, save_path: str, seed: int=42):
        # use fixed seed for consistent evaluation masking
        current_rng_state = torch.get_rng_state()
        torch.manual_seed(seed)

        import matplotlib.pyplot as plt
        self.eval()
        with torch.no_grad():
            # get a batch of images to visualize
            vis_images = images[:num_vis_images]

            # run the forward pass to get predictions and the mask
            latent, mask, ids_restore = self.forward_encoder(vis_images)
            predicted_patches = self.forward_decoder(latent, ids_restore)

            # the mask is a binary tensor (1 for masked, 0 for visible)
            # we expand it to the patch dimension and use it to zero-out the masked patches
            mask_expanded = mask.unsqueeze(-1).repeat(1, 1, self.patch_size**2 * self.in_chans)
            original_patches = self.patchify(vis_images)
            masked_patches = original_patches.clone()
            masked_patches[mask_expanded.bool()] = 0 # zero-out the masked patches
            masked_imgs = self.unpatchify(masked_patches)

            # we use the mask to combine the original visible patches with the predicted masked patches
            recon_patches = original_patches.clone()
            recon_patches[mask_expanded.bool()] = predicted_patches[mask_expanded.bool()] # Fill in predictions
            recon_imgs = self.unpatchify(recon_patches)

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
            axes[i][1].set_title(f'Masked')
            axes[i][1].axis('off')

            # recon
            axes[i][2].imshow(prep_img(recon_imgs[i]), cmap="gray")
            axes[i][2].set_title('Reconstruction')
            axes[i][2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig) 
        print(f"Visualization saved to {save_path}")

        self.train() 