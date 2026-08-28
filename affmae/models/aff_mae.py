import torch
import torch.nn as nn
import torch.nn.functional as F

from affmae.layers.aff import AFFEncoder
from affmae.layers.decoder import CrossAttentionPixelDecoder
from affmae.models.masking import perlin_masking, random_masking  # noqa: F401  (random_masking is the documented alternative below)

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
                 mlp_ratio=2, alpha=10.0, merging_method="l2norm",
                 decoder_deform_backend="auto", decoder_knn_cache=True,
                 cluster_attention_backend="auto"):
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
            merging_method=merging_method,
            cluster_attention_backend=cluster_attention_backend
        )

        # define shapes for the decoder to know what to expect from encoder
        decoder_input_shape = {
            "res2": ShapeSpec(channels=encoder_embed_dim[0], stride=8),
            "res3": ShapeSpec(channels=encoder_embed_dim[1], stride=16),
            "res4": ShapeSpec(channels=encoder_embed_dim[2], stride=32),
            "res5": ShapeSpec(channels=encoder_embed_dim[3], stride=64),
        }

        # KNN grid must cover the full patch coordinate range
        knn_grid_size = img_size // patch_size

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
            shepard_power_learnable=True,
            deform_backend=decoder_deform_backend,
            use_knn_cache=decoder_knn_cache,
            knn_grid_h=knn_grid_size,
            knn_grid_w=knn_grid_size,
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
        """Mask, embed, and encode the visible tokens.

        Note:
            **Nothing calls this.** Pretraining goes ``forward`` ->
            ``_forward_internal``, which does its own masking via
            :meth:`mask_and_embed`. Kept because it is the readable statement of
            the encoder path, but it is not the code that trains -- reading it as
            such is what made the masking strategy look inconsistent.
        """
        img_patches = self.patchify(x)

        # Device-derived: a hardcoded "cuda" warns and self-disables on CPU.
        with torch.amp.autocast(device_type=x.device.type):
            # Perlin, matching mask_and_embed. These two diverged: training
            # masked at random here while every reconstruction figure came from
            # the Perlin path, so the published figures showed the model handling
            # a mask distribution it had never trained on. Swap in
            # random_masking on both sides if you want the old behaviour.
            ids_keep, ids_masked, ids_restore = perlin_masking(
                img_patches, self.img_size, self.encoder_patch_size,
                self.mask_ratio)

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

    def mask_and_embed(self, x):
        """Perlin-mask the image, embed it, and gather the visible tokens.

        Shared by :meth:`_forward_internal` and by token-layout visualization, so
        a figure showing "where the tokens landed" uses the same mask the
        reconstruction did rather than a second copy of this logic.

        Note:
            Masking happens at the *image* level, before the patch embed, because
            the embed's stride exceeds the patch size (as in ConvMAE); masking
            afterwards would leak information from visible patches.

        Args:
            x: [B, C, H, W] input.
        Returns:
            dict with ``img_patches``, ``ids_keep``, ``ids_masked``,
            ``ids_restore``, ``visible_tokens``, ``visible_pos``, ``pos``, ``h``,
            ``w``.
        """
        img_patches = self.patchify(x)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            ids_keep, ids_masked, ids_restore = perlin_masking(
                img_patches, self.img_size, self.encoder_patch_size,
                self.mask_ratio)

        N, L, D = img_patches.shape
        mask = torch.ones(N, L, 1, device=x.device)
        mask.scatter_(dim=1, index=ids_masked.unsqueeze(-1), value=0.0)
        x_with_mask = self.unpatchify(img_patches * mask)

        pos, feat, h, w = self.encoder.patch_embed(x_with_mask, ids_masked)

        visible_tokens = torch.gather(
            feat, dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, feat.shape[-1]))
        visible_pos = torch.gather(
            pos, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 2))

        return {
            "img_patches": img_patches,
            "ids_keep": ids_keep,
            "ids_masked": ids_masked,
            "ids_restore": ids_restore,
            "visible_tokens": visible_tokens,
            "visible_pos": visible_pos,
            "pos": pos,
            "h": h,
            "w": w,
        }

    def _forward_internal(self, x):
        embedded = self.mask_and_embed(x)
        img_patches = embedded["img_patches"]
        ids_keep = embedded["ids_keep"]
        ids_masked = embedded["ids_masked"]
        ids_restore = embedded["ids_restore"]
        pos, h, w = embedded["pos"], embedded["h"], embedded["w"]

        # get features
        encoder_features = self.encoder(embedded["visible_tokens"],
                                        embedded["visible_pos"], h, w)

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
