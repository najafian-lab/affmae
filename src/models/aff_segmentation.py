import torch
import torch.nn as nn
from typing import List
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Import shared layers
from src.layers.aff import AFFEncoder
from src.layers.decoder import CrossAttentionPixelDecoder
from dataclasses import dataclass
from typing import Optional


@dataclass
class ShapeSpec:
    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None


class AFFSegmentation(nn.Module):
    """
    aff model for segmentation, model arch is the same as pre-training but without masking
    """
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 8,
                 in_channels: int = 3,
                 encoder_embed_dim: List[int] = [64, 128, 256, 512],
                 encoder_depth: List[int] = [2, 2, 6, 2],
                 encoder_num_heads: List[int] = [4, 8, 16, 32],
                 encoder_nbhd_size: List[int] = [48, 48, 48, 48],
                 ds_rate: List[float] = [0.25, 0.75, 0.75, 0.75],
                 decoder_embed_dim: int = 384,
                 decoder_num_heads: int = 16,
                 num_classes: int = 4,
                 global_attention: bool = True,
                 mlp_ratio: float = 2.0,
                 alpha: float = 10.0,
                 **kwargs):
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.encoder_patch_size = patch_size

        self.encoder = AFFEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dims=encoder_embed_dim,
            depths=encoder_depth,
            num_heads=encoder_num_heads,
            ds_rates=ds_rate,
            nbhd_sizes=encoder_nbhd_size,
            global_attention=global_attention,
            alpha=alpha,
            mlp_ratio=mlp_ratio,
            **kwargs
        )

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
            transformer_dim_feedforward=768,
            transformer_dec_layers=1,
            conv_dim=decoder_embed_dim,
            mask_dim=decoder_embed_dim,
            norm="GN",
            transformer_in_features=["res2", "res3", "res4", "res5"],
            common_stride=8,
            shepard_power=1.0,
            shepard_power_learnable=True
        )
        
        # query full grid
        # keeping the name "masked token" in order to facilitate loading pre-trained weights 
        # more appropriately called "query_tokens" or "segmentation_tokens"
        self.masked_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.masked_token, std=0.02)

        self.decoder_pred_norm = nn.LayerNorm(decoder_embed_dim)
        # output channels = patch_size^2 * num_classes
        self.decoder_pred_head = nn.Linear(decoder_embed_dim, (patch_size**2) * num_classes)

        self.aux_head_res4_norm = nn.LayerNorm(decoder_embed_dim)
        self.aux_head_res4_head =  nn.Linear(decoder_embed_dim, (patch_size**2) * num_classes)

        self.aux_head_res5_norm = nn.LayerNorm(decoder_embed_dim)
        self.aux_head_res5_head =  nn.Linear(decoder_embed_dim, (patch_size**2) * num_classes)

        # init final proj
        nn.init.ones_(self.decoder_pred_norm.weight)
        nn.init.zeros_(self.decoder_pred_norm.bias)
        nn.init.xavier_uniform_(self.decoder_pred_head.weight)
        nn.init.zeros_(self.decoder_pred_head.bias)
        
        nn.init.ones_(self.aux_head_res4_norm.weight)
        nn.init.zeros_(self.aux_head_res4_norm.bias)
        nn.init.xavier_uniform_(self.aux_head_res4_head.weight)
        nn.init.zeros_(self.aux_head_res4_head.bias)

        nn.init.ones_(self.aux_head_res5_norm.weight)
        nn.init.zeros_(self.aux_head_res5_norm.bias)
        nn.init.xavier_uniform_(self.aux_head_res5_head.weight)
        nn.init.zeros_(self.aux_head_res5_head.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape flattened patches back to spatial map [B, num_classes, H, W]"""
        p = self.encoder_patch_size
        h = w = int(x.shape[1] ** 0.5)
        c = self.num_classes
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x: torch.Tensor):
        b = x.shape[0]

        # we pass ids_masked=None so PatchEmbedWithMasking acts like a standard conv stem
        pos, feat, h, w = self.encoder.patch_embed(x, ids_masked=None)
        
        encoder_features_dict = self.encoder(feat, pos, h, w)
        
        query_tokens = self.masked_token.expand(b, encoder_features_dict["res2"].shape[1], -1)
        
        predicted_tokens = self.cross_attention_decoder.forward_features(
            encoder_features_dict, query_tokens, pos
        )
        
        aux_res5_pred = self.aux_head_res5_head(self.aux_head_res5_norm(predicted_tokens[0]))
        aux_res4_pred = self.aux_head_res4_head(self.aux_head_res4_norm(predicted_tokens[1]))
        # -1 here since cross attention decoder returns a list of intermediate features
        # we want the last one
        pred_head = self.decoder_pred_norm(predicted_tokens[-1])
        pred_patches = self.decoder_pred_head(pred_head)

        return [self.unpatchify(aux_res5_pred), self.unpatchify(aux_res4_pred), self.unpatchify(pred_patches)]
    
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

                pos, feat, h, w = self.encoder.patch_embed(img_to_process, ids_masked=None)
                
                encoder_stage_outputs = self.encoder.forward_with_pos(feat, pos, h, w)
                all_pos_items.append([item.cpu().numpy() for item in encoder_stage_outputs])
                all_images.append(img_to_process)

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
      
