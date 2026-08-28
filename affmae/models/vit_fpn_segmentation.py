import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block
from affmae.layers.pos_embed import get_2d_sincos_pos_embed

class ViTMultiScaleGenerator(nn.Module):
    """
    Takes 4 Stride-16 feature maps from the ViT and turns them into
    Strides 4, 8, 16, and 32 to feed into an FPN/UperNet.
    """
    def __init__(self, embed_dim, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

        if patch_size == 16:
            self.stage1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
                nn.BatchNorm2d(embed_dim // 2),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim // 2, embed_dim, kernel_size=2, stride=2),
            ) # 4x up (Stride 4)
            self.stage2 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2) # 2x up (Stride 8)
            self.stage3 = nn.Identity() # 1x (Stride 16)
            self.stage4 = nn.MaxPool2d(kernel_size=2, stride=2) # 2x down (Stride 32)

        elif patch_size == 8:
            self.stage1 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2) # 2x up (Stride 4)
            self.stage2 = nn.Identity() # 1x (Stride 8)
            self.stage3 = nn.MaxPool2d(kernel_size=2, stride=2) # 2x down (Stride 16)
            self.stage4 = nn.MaxPool2d(kernel_size=4, stride=4) # 4x down (Stride 32)
        else:
            raise ValueError(f"Patch size {patch_size} not supported!")

    def forward(self, features):
        return [
            self.stage1(features[0]),
            self.stage2(features[1]),
            self.stage3(features[2]),
            self.stage4(features[3])
        ]

class PPM(nn.Module):
    """Pyramid Pooling Module used in UperNet"""
    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.features = nn.ModuleList()
        for pool_scale in pool_scales:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_scale),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_scales) * out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        ppm_outs = [x]
        for ppm_layer in self.features:
            ppm_out = ppm_layer(x)
            ppm_out = F.interpolate(ppm_out, size=(h, w), mode='bilinear', align_corners=False)
            ppm_outs.append(ppm_out)
        ppm_outs = torch.cat(ppm_outs, dim=1)
        return self.bottleneck(ppm_outs)

class UperNetDecoder(nn.Module):
    def __init__(self, in_channels_list, out_channels=256, num_classes=3):
        super().__init__()
        self.ppm = PPM(in_channels_list[-1], out_channels)

        self.lateral_convs = nn.ModuleList()
        for in_c in in_channels_list[:-1]:
            self.lateral_convs.append(nn.Conv2d(in_c, out_channels, 1))

        self.fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        self.seg_head = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, num_classes, kernel_size=1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        ppm_out = self.ppm(features[-1])
        laterals = [lateral_conv(f) for f, lateral_conv in zip(features[:-1], self.lateral_convs)]
        laterals.append(ppm_out)

        for i in range(len(laterals) - 1, 0, -1):
            target_shape = laterals[i - 1].shape[-2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=target_shape, mode='bilinear', align_corners=False)

        fpn_outs = [fpn_conv(lat) for lat, fpn_conv in zip(laterals[:-1], self.fpn_convs)]
        fpn_outs.append(laterals[-1])

        target_shape = fpn_outs[0].shape[-2:]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=target_shape, mode='bilinear', align_corners=False)

        fpn_fused = torch.cat(fpn_outs, dim=1)
        logits = self.seg_head(fpn_fused)

        # Upsample to original resolution (assumes highest FPN resolution is stride 4)
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        return logits

class ViTSegmentationUperNet(nn.Module):
    """
    Vision Transformer Backbone with UperNet Decoder for multi-class segmentation.
    """
    def __init__(self, img_size=512, patch_size=16, in_chans=1,
                 embed_dim=768, depth=12, num_heads=12, decoder_conv_dim=256,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, num_classes=4):
        super().__init__()
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.img_size = img_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.grid_size = img_size // patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.encoder_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.encoder_norm = norm_layer(embed_dim)

        # evenly spaced layers
        self.out_indices = [int(depth * i / 4) - 1 for i in range(1, 5)]

        self.multi_scale_generator = ViTMultiScaleGenerator(embed_dim, patch_size)
        self.decoder = UperNetDecoder(
            in_channels_list=[embed_dim] * 4,
            out_channels=256,
            num_classes=num_classes
        )

        self.initialize_weights()

    def initialize_weights(self):
        # 2D Positional embeddings matching your MAE setup
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Patch embedding
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # CLS token
        torch.nn.init.normal_(self.cls_token, std=.02)

        # Other layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        outs = []
        for i, blk in enumerate(self.encoder_blocks):
            x = blk(x)
            if i in self.out_indices:
                # Remove CLS token: [B, N+1, C] -> [B, N, C]
                tokens = x[:, 1:, :]
                # Reshape 1D tokens to 2D spatial grid: [B, C, H/p, W/p]
                spatial_grid = tokens.reshape(B, self.grid_size, self.grid_size, self.embed_dim)
                spatial_grid = spatial_grid.permute(0, 3, 1, 2).contiguous()
                outs.append(spatial_grid)

        return outs # Returns 4 stages of Stride 16 (or Stride 8 if patch_size=8)

    def forward(self, imgs):
        vit_features = self.forward_encoder(imgs)

        multi_scale_features = self.multi_scale_generator(vit_features)

        logits = self.decoder(multi_scale_features)

        return [logits]