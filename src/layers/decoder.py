"""
Cross-attention based pixel decoder using single-level deformable attention.
Processes encoder features from res5 down to res2 with multiple decoder blocks per stage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Callable
import numpy as np
import math

import fvcore.nn.weight_init as weight_init
from .attention import DeformableSelfAttention, DeformableCrossAttention, GlobalSelfAttention


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, pos):
        '''
        pos - b x n x d
        '''
        b, n, d = pos.shape
        y_embed = pos[:, :, 1]  # b x n
        x_embed = pos[:, :, 0]
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed.max() + eps) * self.scale
            x_embed = x_embed / (x_embed.max() + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=pos.device)  # npf
        dim_t = self.temperature ** (2 * (dim_t.div(2, rounding_mode='floor')) / self.num_pos_feats)  # npf

        pos_x = x_embed[:, :, None] / dim_t  # b x n x npf
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.cat(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=2
        )
        pos_y = torch.cat(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=2
        )
        pos = torch.cat((pos_x, pos_y), dim=2)  # b x n x d'
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class MSDecoderBlock(nn.Module):
    """
    Multi-Scale Decoder Block for deformable attention.
    
    This block performs self-attention on query tokens followed by cross-attention
    to encoder features. Designed for masked token prediction tasks.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_points: Number of sampling points per head
        d_ffn: Feed-forward network dimension
        dropout: Dropout probability
        activation: Activation function
        shepard_power: Power for Shepard interpolation
        shepard_power_learnable: Whether to make shepard_power learnable
    """
    
    def __init__(self, d_model=256, n_heads=8, n_points=4, d_ffn=1024,
                 dropout=0.1, activation="relu", shepard_power=3.0, 
                 shepard_power_learnable=True):
        super().__init__()
        
        # self attention for query tokens
        self.self_attn = DeformableSelfAttention(
            d_model, n_heads, n_points, shepard_power, shepard_power_learnable
        )
        # self.self_attn = GlobalSelfAttention(
        #     d_model, n_heads
        # )

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # cross attention to encoder features
        self.cross_attn = DeformableCrossAttention(
            d_model, n_heads, n_points, shepard_power, shepard_power_learnable
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # feed-forward network
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
    
    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        """Add positional embeddings to tensor."""
        if pos_embed is None:
            return tensor
        else:
            return tensor + pos_embed
    
    def forward(self, query_tokens, query_pos, src, pos, pos_embed):
        """
        Forward pass for MSDecoderBlock.
        
        Args:
            query_tokens: Query tokens (e.g., masked tokens) [b, n_q, c]
            query_pos: Query token positions [b, n_q, 2]
            src: Encoder source features [b, n_src, c]
            pos: Encoder source positions [b, n_src, 2]
            pos_embed: Positional embeddings for src [b, n_src, c]
            
        Returns:
            output: Processed query tokens [b, n_q, c]
        """
        # self attention on query tokens
        self_attn_out = self.self_attn(query_tokens, query_pos)
        # self_attn_out = self.self_attn(query_tokens)
        
        # residual connection and layer norm
        query_tokens = query_tokens + self.dropout1(self_attn_out)
        query_tokens = self.norm1(query_tokens)
        
        # cross attention to encoder features
        src_with_pos = self.with_pos_embed(src, pos_embed)
        cross_attn_out = self.cross_attn(query_tokens, query_pos, src_with_pos, pos)
        
        # residual connection and layer norm
        query_tokens = query_tokens + self.dropout2(cross_attn_out)
        query_tokens = self.norm2(query_tokens)
        
        # feed-forward network
        ff_out = self.linear2(self.dropout3(self.activation(self.linear1(query_tokens))))
        query_tokens = query_tokens + self.dropout4(ff_out)
        query_tokens = self.norm3(query_tokens)
        
        return query_tokens


class CrossAttentionPixelDecoder(nn.Module):
    """
    Cross-attention based pixel decoder for masked token prediction.
    
    This decoder processes encoder features from res5 down to res2, using multiple
    MSDecoderBlocks at each stage to cross-attend to encoder features.
    
    Args:
        input_shape: Dictionary mapping feature names to their shapes (channels, stride)
        transformer_dropout: Dropout probability in transformer
        transformer_nheads: Number of attention heads
        transformer_dim_feedforward: Dimension of feedforward network
        transformer_dec_layers: Number of decoder blocks per stage
        conv_dim: Number of output channels for intermediate layers
        mask_dim: Number of output channels for final mask prediction
        norm: Normalization for conv layers
        transformer_in_features: List of encoder feature names to use
        common_stride: Target stride for output features
        shepard_power: Power for Shepard interpolation
        shepard_power_learnable: Whether to make shepard_power learnable
    """
    
    def __init__(
        self,
        input_shape: Dict[str, str],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_dec_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        transformer_in_features: List[str],
        common_stride: int,
        shepard_power: float,
        shepard_power_learnable: bool
    ):
        super().__init__()
        
        # process input shapes
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }
        
        # sort features by stride (res2 to res5)
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]
        
        # transformer input features (sorted by stride)
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]
        
        self.transformer_num_feature_levels = len(self.transformer_in_features)
        self.transformer_dec_layers = transformer_dec_layers
        self.conv_dim = conv_dim
        self.mask_dim = mask_dim
        self.common_stride = common_stride
        self.shepard_power = shepard_power
        self.shepard_power_learnable = shepard_power_learnable
        
        # input projection layers for encoder features
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # From low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Linear(in_channels, conv_dim, bias=True),
                    nn.LayerNorm(conv_dim)
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(transformer_in_channels[-1], conv_dim, bias=True),
                    nn.LayerNorm(conv_dim)
                )])
        
        # initialize input projections
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        
        # positional embedding
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # decoder blocks for each stage (res5 -> res2)
        self.decoder_blocks = nn.ModuleList()
        for i in range(self.transformer_num_feature_levels):
            stage_blocks = nn.ModuleList()
            for j in range(transformer_dec_layers):
                decoder_block = MSDecoderBlock(
                    d_model=conv_dim,
                    n_heads=transformer_nheads,
                    n_points=4,  # Fixed number of sampling points
                    d_ffn=transformer_dim_feedforward,
                    dropout=transformer_dropout,
                    activation="relu",
                    shepard_power=shepard_power,
                    shepard_power_learnable=shepard_power_learnable
                )
                stage_blocks.append(decoder_block)
            self.decoder_blocks.append(stage_blocks)        


    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, str]):
        """Create decoder from configuration."""
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["transformer_dim_feedforward"] = 1024
        ret["transformer_dec_layers"] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_DEC_LAYERS  # New config
        ret["transformer_in_features"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
        ret["common_stride"] = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        ret['shepard_power'] = cfg.MODEL.AFF.SHEPARD_POWER / 2.0
        ret['shepard_power_learnable'] = cfg.MODEL.AFF.SHEPARD_POWER_LEARNABLE
        return ret
    
    def forward_features(self, features, masked_tokens, masked_pos):
        """
        Forward pass for cross-attention decoder.
        
        Args:
            features: Dictionary of encoder features with keys like 'res2', 'res3', etc.
                     Each feature should have corresponding '_pos' and '_spatial_shape' keys
            masked_tokens: Query tokens (masked tokens) [b, n_masked, c]
            masked_pos: Positions of masked tokens [b, n_masked, 2]
            
        Returns:
            query_tokens: Updated query tokens [b, n_masked, conv_dim]
        """
        # process encoder features
        srcs = []
        poss = []
        pos_embeds = []
        spatial_shapes = []
        
        # get finest feature for grid reference
        finest_feat = self.in_features[0]
        grid_hw = features[finest_feat + "_spatial_shape"]
        b = features[finest_feat].shape[0]
        
        # process encoder features from res5 to res2
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f]
            pos = features[f + "_pos"]
            spatial_shape = features[f + "_spatial_shape"]

            # Project features to common dimension
            src = self.input_proj[idx](x)       
            pos_embed = self.pe_layer(pos)
            
            srcs.append(src)
            poss.append(pos)
            pos_embeds.append(pos_embed)
            spatial_shapes.append(spatial_shape)
        
        # process through decoder stages (res5 -> res2)
        query_tokens = masked_tokens
        query_pos = masked_pos
        query_pos_embed = self.pe_layer(masked_pos)

        # add the positional embeddings to the query tokens
        query_tokens = query_tokens + query_pos_embed
        
        
        intermediates = []

        # from res5 to res2
        for stage_idx in range(self.transformer_num_feature_levels):
            # get encoder features for this stage
            src = srcs[stage_idx]
            pos = poss[stage_idx]
            pos_embed = pos_embeds[stage_idx]
            
            # apply decoder blocks for this stage
            stage_blocks = self.decoder_blocks[stage_idx]
            for block in stage_blocks:
                query_tokens = block(
                    query_tokens=query_tokens,
                    query_pos=query_pos,
                    src=src,
                    pos=pos,
                    pos_embed=pos_embed
                )
                intermediates.append(query_tokens)
        
        # intermediates should be [res5, res4, res3, res2]
        return intermediates
    
    def forward(self, features, masked_tokens, masked_pos):
        """Main forward pass."""
        return self.forward_features(features, masked_tokens, masked_pos)