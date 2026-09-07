import torch
import torch.nn as nn
from typing import List

# Import shared layers
from affmae.layers.aff import AFFEncoder
from affmae.layers.decoder import CrossAttentionPixelDecoder
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
                 decoder_deform_backend: str = "auto",
                 decoder_knn_cache: bool = True,
                 cluster_attention_backend: str = "auto",
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
            cluster_attention_backend=cluster_attention_backend,
            **kwargs
        )

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
            transformer_dim_feedforward=768,
            transformer_dec_layers=1,
            conv_dim=decoder_embed_dim,
            mask_dim=decoder_embed_dim,
            norm="GN",
            transformer_in_features=["res2", "res3", "res4", "res5"],
            common_stride=8,
            shepard_power=1.0,
            shepard_power_learnable=True,
            deform_backend=decoder_deform_backend,
            use_knn_cache=decoder_knn_cache,
            knn_grid_h=knn_grid_size,
            knn_grid_w=knn_grid_size,
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
