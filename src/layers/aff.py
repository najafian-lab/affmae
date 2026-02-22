#
# Modified from original AFF repo
#
from typing import Tuple, Union
from itertools import repeat
import logging

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from src.utils.pos_embed import pre_table, rel_pos_width, table_width
from src.utils.geometry import space_filling_cluster
from src.layers.kernels.knn import knn as triton_knn
from src.layers.kernels.weighted_features import weighted_features
from src.layers.attention import ClusterAttention


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1.inp = 'norm'
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2.inp = 'act'
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ClusterTransformerBlock(nn.Module):
    r""" Cluster Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads,
                 mlp_ratio=2., drop=0., attn_drop=0., drop_path=0., layer_scale=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = ClusterAttention(
            dim, num_heads=num_heads,
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # layer_scale code copied from https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/a2cfef599fffd36d058a5a4cfdbd81c008e1c349/classification/nat.py
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float] and layer_scale > 0:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, feat, member_idx, cluster_mask, pe_idx, global_attn):
        """
        Args:
            feat - b x n x c, token features
            member_idx - b x n x nbhd, token idx in each local nbhd
            cluster_mask - b x n x nbhd, binary mask for valid tokens (1 if valid)
            pe_idx - b x n x nbhd, idx for the pre-computed position embedding lookup table
            global_attn - bool, whether to perform global attention
        """

        b, n, c = feat.shape
        assert c == self.dim, "dim does not accord to input"

        shortcut = feat
        feat = self.norm1(feat)

        # cluster attention
        feat = self.attn(feat=feat,
                        member_idx=member_idx,
                        cluster_mask=cluster_mask,
                        pe_idx=pe_idx,
                        global_attn=global_attn)

        # FFN
        if not self.layer_scale:
            feat = shortcut + self.drop_path(feat)
            feat_mlp = self.mlp(self.norm2(feat))
            feat = feat + self.drop_path(feat_mlp)
        else:
            feat = shortcut + self.drop_path(self.gamma1 * feat)
            feat_mlp = self.mlp(self.norm2(feat))
            feat = feat + self.drop_path(self.gamma2 * feat_mlp)

        return feat

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"mlp_ratio={self.mlp_ratio}"


class ClusterMerging(nn.Module):
    r""" Adaptive Downsampling.

    Args:
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        alpha (float, optional): the weight to be multiplied with importance scores. Default: 4.0
        ds_rate (float, optional): downsampling rate, to be multiplied with the number of tokens. Default: 0.25
        reserve_on (bool, optional): whether to turn on reserve tokens in downsampling. Default: True
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm, alpha=4.0, ds_rate=0.25, reserve_on=True):
        super().__init__()
        self.dim = dim
        self.pos_dim = 2
        self.alpha = alpha
        self.ds_rate = ds_rate
        self.reserve_on = reserve_on

        # pointconv
        inner_ch = 4
        self.weight_net = nn.Sequential(
            nn.Linear(self.pos_dim+3, inner_ch, bias=True),
            nn.LayerNorm(inner_ch),
            nn.GELU()
        )
        self.weight_net[0].inp = 'norm'

        self.norm = norm_layer(inner_ch*dim)
        self.linear = nn.Linear(dim*inner_ch, out_dim)
        self.linear.inp = 'norm'

        # Add a separate norm and linear layer for the global fallback path
        self.global_norm = norm_layer(dim)
        self.global_linear = nn.Linear(dim, out_dim)
        self.global_linear.inp = 'norm'

        # Cache for inference optimization: store weights_table per (device, dtype)
        # Only used when model is in eval mode (not training)
        self._weights_table_cache = {}

    def clear_weights_cache(self):
        """Clear the cached weights_table. Useful when switching between training/eval modes."""
        self._weights_table_cache.clear()

    def forward(self, pos, feat, member_idx, cluster_mask, learned_prob, stride, pe_idx, reserve_num):
        """
        Args:
            pos - b x n x 2, token positions
            feat - b x n x c, token features
            member_idx - b x n x nbhd, token idx in each local nbhd
            cluster_mask - b x n x nbhd, binary mask for valid tokens (1 if valid)
            learned_prob - b x n x 1, learned importance scores
            stride - int, "stride" of the current feature map, 2,4,8 for the 3 stages respectively
            pe_idx - b x n x nbhd, idx for the pre-computed position embedding lookup table
            reserve_num - int, number of tokens to be reserved
        """

        b, n, c = feat.shape
        d = pos.shape[2]

        keep_num = int(n*self.ds_rate)
        pos_long = pos.long()

        # track amp state
        if torch.is_autocast_enabled():
            cast_dtype = torch.get_autocast_dtype('cuda') or torch.float16
        else:
            cast_dtype = torch.float32

        # grid prior
        if False:  # TESTING (please see note below if True)
            if stride == 2:  # no ada ds yet, no need ada grid
                grid_prob = ((pos_long % stride) == 0).all(-1).to(cast_dtype)  # b x n
            else:
                _, min_dist = triton_knn(pos, pos, 2, return_dist=True)  # b x n x 2
                min_dist = min_dist[:, :, 1]  # b x n
                ada_stride = (2.0**(min_dist.float().log2().ceil()+1)).to(pos_long.dtype)  # b x n
                grid_prob = ((pos_long % ada_stride.unsqueeze(2).long()) == 0).all(-1).to(cast_dtype)  # b x n

            final_prob = grid_prob

        # add importance score
        if learned_prob is not None:
            lp = learned_prob.detach().view(b, n)
            
            # NOTE: if above TESTING is TRUE use this
            # lp = lp * self.alpha
            # final_prob = final_prob + lp
            final_prob = lp

        # reserve points on a coarse grid
        if self.reserve_on:
            reserve_mask = ((pos_long % (stride*2)) == 0).all(dim=-1).to(cast_dtype)  # b x n
            final_prob = final_prob + (reserve_mask*(-100))
            sample_num = keep_num - reserve_num
        else:
            sample_num = keep_num

        # probabilistically sample tokens


        # deterministic sampling
        sample_idx = final_prob.topk(sample_num, dim=1, sorted=False)[1]  # b x n_

        if self.reserve_on:
            reserve_idx = reserve_mask.nonzero(as_tuple=True)[1].reshape(b, reserve_num)
            idx = torch.cat([sample_idx, reserve_idx], dim=-1).unsqueeze(2)  # b x n_ x 1
        else:
            idx = sample_idx.unsqueeze(2)

        n_down = idx.shape[1]
        assert n_down == keep_num, "n not equal to keep num!"

        pos_down = pos.gather(index=idx.expand(-1, -1, d), dim=1)

        # fall back, code should never run this path anymore since member index cannot be none regardless of global attention
        if member_idx is None:
            # Gather the features of the sampled tokens
            feat_down = feat.gather(index=idx.expand(-1, -1, c), dim=1)
            # Apply the dedicated global norm and linear layers
            feat_down = self.global_norm(feat_down)
            feat_down = self.global_linear(feat_down)
        else:
            # LOCAL PATH
            nbhd_size = member_idx.shape[-1]
            member_idx = member_idx.gather(index=idx.expand(-1, -1, nbhd_size), dim=1)
            pe_idx = pe_idx.gather(index=idx.expand(-1, -1, nbhd_size), dim=1)
            if cluster_mask is not None:
                cluster_mask = cluster_mask.gather(index=idx.expand(-1, -1, nbhd_size), dim=1)
            if learned_prob is not None:
                lp = learned_prob.gather(index=member_idx.view(b, -1, 1), dim=1).reshape(b, n_down, nbhd_size, 1)

            global pre_table
            device = pe_idx.device
            dtype = feat.dtype
            
            # Move pre_table to correct device if needed
            if not pre_table.is_cuda or pre_table.device != device:
                pre_table = pre_table.to(device)
            
            # Cache weights_table during inference (when not training)
            # Key by (device, dtype) to handle different devices/dtypes
            cache_key = (device, dtype)
            
            if not self.training and cache_key in self._weights_table_cache:
                # Use cached weights_table during inference
                weights_table = self._weights_table_cache[cache_key]
                # Ensure cached tensor is on the correct device (should already be, but verify)
                if weights_table.device != device:
                    weights_table = weights_table.to(device)
                    self._weights_table_cache[cache_key] = weights_table
            else:
                # Compute weights_table
                weights_table = self.weight_net(pre_table.to(dtype))
                
                # Cache for inference mode only (when weights are frozen)
                if not self.training:
                    self._weights_table_cache[cache_key] = weights_table

            weight_shape = pe_idx.shape
            inner_ch = weights_table.shape[-1]
            weights = weights_table.gather(index=pe_idx.view(-1, 1).expand(-1, inner_ch), dim=0).reshape(*(weight_shape), inner_ch)

            if learned_prob is not None:
                if cluster_mask is not None:
                    lp = lp * cluster_mask.unsqueeze(3)
                weights = weights * lp
            else:
                if cluster_mask is not None:
                    weights = weights * cluster_mask.unsqueeze(3)

            feat_down = weighted_features(weights, feat, member_idx.view(b, n_down, -1)).reshape(b, n_down, -1)
            feat_down = self.norm(feat_down)
            feat_down = self.linear(feat_down)

        return pos_down, feat_down
        

class L2NormMerging(nn.Module):
    """
    Parameter free downsampling by marking tokens with the highest L2 norms before the 
    last transformer layer in a stage. the last transformer block will then act implicitly
    as a merging layer. 

    Args:
        dim (int): Number of input channels.
        ds_rate (float, optional): downsampling rate, to be multiplied with the number of tokens. Default: 0.4
    """
    def __init__(self, dim, ds_rate=0.4):
        super().__init__()
        self.ds_rate = ds_rate
        
        self.token_selection_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.token_selection_embedding, std=0.02)

    def forward(self, feat): 
        b, n, c = feat.shape
        k = int(self.ds_rate * n)

        # [b, n]
        scores = torch.norm(feat, p=2, dim=-1)

        _, indices = scores.topk(k, dim=1)

        # [b, n]
        mask = torch.zeros_like(scores).scatter_(1, indices, 1.0)
        
        # mark tokens
        feat_out = feat + (mask.unsqueeze(-1) * self.token_selection_embedding)
        
        return feat_out, indices


class BasicLayer(nn.Module):
    """ AutoFocusFormer layer for one stage.

    Args:
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        cluster_size (int): Cluster size.
        nbhd_size (int): Neighbor size. If larger than or equal to number of tokens, perform global attention;
                            otherwise, rounded to the nearest multiples of cluster_size.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        alpha (float, optional): the weight to be multiplied with importance scores. Default: 4.0
        ds_rate (float, optional): downsampling rate, to be multiplied with the number of tokens. Default: 0.25
        reserve_on (bool, optional): whether to turn on reserve tokens in downsampling. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        layer_scale (float, optional): Layer scale initial parameter. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, out_dim, cluster_size, nbhd_size,
                 depth, num_heads, mlp_ratio,
                 alpha=4.0, ds_rate=0.25, reserve_on=False,
                 global_attention=False, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 layer_scale=0.0, downsample=None):

        super().__init__()
        self.dim = dim
        self.nbhd_size = nbhd_size
        self.cluster_size = cluster_size
        self.depth = depth
        self.global_attention = global_attention 

        # build blocks
        self.blocks = nn.ModuleList([
            ClusterTransformerBlock(dim=dim,
                                    num_heads=num_heads,
                                    mlp_ratio=mlp_ratio,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    layer_scale=layer_scale,
                                    norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            if issubclass(downsample, ClusterMerging):
                self.downsample = downsample(dim=dim, out_dim=out_dim, norm_layer=norm_layer, alpha=alpha, ds_rate=ds_rate, reserve_on=reserve_on)
                self.prob_net = nn.Linear(dim, 1)
                self.prob_net.inp = 'norm'
            elif issubclass(downsample, L2NormMerging):
                self.downsample = downsample(dim=dim, ds_rate=ds_rate)
                self.downsample_norm = norm_layer(dim)
                self.downsample_proj = nn.Linear(dim, out_dim)
                self.downsample_proj.inp = "norm"
            else:
                raise ValueError(f"{downsample} method not supported!")
        else:
            self.downsample = None

        self.ds_rate = ds_rate

        # cache the clustering result for the first feature map since it is on grid
        self.pos, self.cluster_mean_pos, self.member_idx, self.cluster_mask, self.reorder = None, None, None, None, None

    def forward(self, pos, feat, h, w, on_grid, stride):
        """
        Args:
            pos - b x n x 2, token positions
            feat - b x n x c, token features
            h,w - max height and width of token positions
            on_grid - bool, whether the tokens are still on grid; True for the first feature map
            stride - int, "stride" of the current token set; starts with 2, then doubles in each stage
        """
        b, n, d = pos.shape
        if not isinstance(b, int):
            b, n, d = b.item(), n.item(), d.item()  # make the flop analyzer happy
        c = feat.shape[2]
        assert self.cluster_size > 0, 'self.cluster_size must be positive'

        global_attn = self.global_attention

        if self.nbhd_size >= n:
            global_attn = True

        k = int(math.ceil(n / float(self.cluster_size)))  # number of clusters
        nnc = min(int(round(self.nbhd_size / float(self.cluster_size))), k)  # number of nearest clusters
        nbhd_size = self.cluster_size * nnc 
        self.nbhd_size = nbhd_size  # if not global attention, then nbhd size is rounded to nearest multiples of cluster

        # do member index stuff regardless, needed for cluster merging anyways
        if k == n:
            cluster_mean_pos = pos
            member_idx = torch.arange(n, device=feat.device).long().reshape(1, n, 1).expand(b, -1, -1)  # b x n x 1
            cluster_mask = None
        else:
            if on_grid and self.training:
                if self.cluster_mean_pos is None:
                    self.pos, self.cluster_mean_pos, self.member_idx, self.cluster_mask, self.reorder = space_filling_cluster(pos, self.cluster_size, h, w, no_reorder=False)
                pos, cluster_mean_pos, member_idx, cluster_mask = self.pos[:b], self.cluster_mean_pos[:b], self.member_idx[:b], self.cluster_mask
                feat = feat[torch.arange(b).to(feat.device).repeat_interleave(n), self.reorder[:b].view(-1)].reshape(b, n, c)
                if cluster_mask is not None:
                    cluster_mask = cluster_mask[:b]
            else:
                pos, cluster_mean_pos, member_idx, cluster_mask, reorder = space_filling_cluster(pos, self.cluster_size, h, w, no_reorder=False)
                feat = feat[torch.arange(b).to(feat.device).repeat_interleave(n), reorder.view(-1)].reshape(b, n, c)

        assert member_idx.shape[1] == k and member_idx.shape[2] == self.cluster_size, "member_idx shape incorrect!"

        nearest_cluster = triton_knn(pos, cluster_mean_pos, nnc)  # b x n x nnc

        m = self.cluster_size
        member_idx = member_idx.gather(index=nearest_cluster.view(b, -1, 1).expand(-1, -1, m), dim=1).reshape(b, n, nbhd_size)  # b x n x nnc*m

        if cluster_mask is not None:
            cluster_mask = cluster_mask.gather(index=nearest_cluster.view(b, -1, 1).expand(-1, -1, m), dim=1).reshape(b, n, nbhd_size)

        if global_attn:
            rel_pos = (pos[:, None, :, :]+rel_pos_width) - pos[:, :, None, :]  # b x n x n x d
        else:
            pos_ = pos.gather(index=member_idx.view(b, -1, 1).expand(-1, -1, d), dim=1).reshape(b, n, nbhd_size, d)
            rel_pos = pos_ - (pos.unsqueeze(2)-rel_pos_width)  # b x n x nbhd_size x d

        rel_pos = rel_pos.clamp(0, table_width-1)
        pe_idx = (rel_pos[..., 1] * table_width + rel_pos[..., 0]).long()

        selection_indices = None

        for idx, i_blk in enumerate(range(len(self.blocks))):
            # right before the last block
            if idx == len(self.blocks) - 1 and self.downsample is not None and isinstance(self.downsample, L2NormMerging):
                feat, selection_indices = self.downsample(feat)

            blk = self.blocks[i_blk]
            feat = blk(feat=feat,
                       member_idx=member_idx,
                       cluster_mask=cluster_mask,
                       pe_idx=pe_idx,
                       global_attn=global_attn)

        if self.downsample is not None:
            if isinstance(self.downsample, L2NormMerging) and selection_indices is not None:
                # (B, k) -> (B, k, C)
                C = feat.shape[-1]
                idx_feat = selection_indices.unsqueeze(-1).expand(-1, -1, C)
                feat_down = torch.gather(feat, dim=1, index=idx_feat)
                
                # (B, k) -> (B, k, 2)
                D_pos = pos.shape[-1]
                idx_pos = selection_indices.unsqueeze(-1).expand(-1, -1, D_pos)
                pos_down = torch.gather(pos, dim=1, index=idx_pos)

                # feat_down (B, k, out_dim) and pos_down (B, k, 2)
                feat_down = self.downsample_norm(feat_down)
                feat_down = self.downsample_proj(feat_down)
            elif isinstance(self.downsample, ClusterMerging):
                learned_prob = self.prob_net(feat).sigmoid()  # b x n x 1
                reserve_num = math.ceil(h/(stride*2)) * math.ceil(w/(stride*2))

                pos_down, feat_down = self.downsample(pos=pos, feat=feat,
                                                        member_idx=member_idx, cluster_mask=cluster_mask,
                                                        learned_prob=learned_prob, stride=stride,
                                                        pe_idx=pe_idx, reserve_num=reserve_num)

        if self.downsample is not None:
            return pos, feat, pos_down, feat_down
        else:
            return pos, feat, pos, feat

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"


class LayerNorm2dImages(nn.Module):
    """A LayerNorm module for 4D tensors (images)."""
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (B, C, H, W)
        # Permute to (B, H, W, C) for LayerNorm
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        # Permute back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x


class AFFPatchEmbed(nn.Module):
    """
    image to patch embedding with dynamic downsampling layers and masking.
    """
    def __init__(self, img_size=None, patch_size=8, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        # get the number of patches
        _, _, self.num_patches = self._init_img_size(img_size)

        # ensure patch_size is a power of 2
        if (patch_size & (patch_size - 1) != 0) or patch_size == 0:
            raise ValueError(f"patch_size must be a power of 2, but got {patch_size}")

        # calculate the number of downsampling stages
        num_layers = int(math.log2(patch_size))

        self.projs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

        # create layers dynamically in a loop
        current_chans = in_chans
        for i in range(num_layers):
            # the last layer outputs embed_dim, intermediate layers use embed_dim // 2
            out_chans = embed_dim if i == num_layers - 1 else embed_dim // 2

            self.projs.append(nn.Conv2d(current_chans, out_chans, kernel_size=3, stride=2, padding=1))

            # norm and activation are not applied after the final projection layer
            if i < num_layers - 1:
                self.lns.append(LayerNorm2dImages(out_chans))
                self.acts.append(nn.GELU())

            current_chans = out_chans

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = tuple(repeat(img_size, 2))
        grid_size = tuple([s // p for s, p in zip(img_size, tuple(repeat(self.patch_size, 2)))])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def patchify(self, imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
        p = patch_size; _, c, h_dim, w_dim = imgs.shape
        h = h_dim // p; w = w_dim // p
        x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(imgs.shape[0], h * w, p**2 * c)

    def unpatchify(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        p = patch_size; h = w = int(x.shape[1] ** 0.5)
        c = x.shape[2] // (p * p)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, w * p)


    def forward(self, x, ids_masked):
        # padding
        ps = self.patch_size
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps))
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps))

        # create mask based on ids_masked
        if ids_masked is not None:
            B = x.shape[0]
            L = (H // ps) * (W // ps)
            token_mask = torch.ones(B, L, 1, device=x.device, dtype=x.dtype)
            token_mask.scatter_(dim=1, index=ids_masked.unsqueeze(-1), value=0.0)
        else:
            # setting it to 1 will keep everything, essentially no masking
            token_mask = 1.0 

        for idx, proj_i in enumerate(self.projs):
            x = proj_i(x)

            # the effective patch size for the current feature map resolution
            current_patch_scale = ps // (2**(idx + 1))

            # for the very last layer, the feature map is 1x1 per token, so the scale is 1
            if current_patch_scale == 0: current_patch_scale = 1

            x_tokens = self.patchify(x, patch_size=current_patch_scale)
            x_tokens = x_tokens * token_mask
            x = self.unpatchify(x_tokens, patch_size=current_patch_scale)

            # apply norm and act if they exist for this layer (all but the last)
            if idx < len(self.lns):
                x = self.lns[idx](x)
                x = self.acts[idx](x)

        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # b x n x c

        if self.norm is not None:
            x = self.norm(x)

        hs = torch.arange(0, h, device=x.device)
        ws = torch.arange(0, w, device=x.device)
        ys, xs = torch.meshgrid(hs, ws, indexing='ij')
        pos = torch.stack([xs, ys], dim=2).unsqueeze(0).expand(b, -1, -1, -1).reshape(b, -1, 2).to(x.dtype)

        return pos, x, h, w


class AFFEncoder(nn.Module):
    def __init__(self, img_size=384, patch_size=8, in_chans=1, 
                 embed_dims=[64, 128, 256, 448], 
                 depths=[3, 3, 7, 4], 
                 num_heads=[2, 4, 8, 14],
                 nbhd_sizes=[64, 64, 64, 64],
                 cluster_size=8, 
                 ds_rates=[0.4, 0.4, 0.4, 0.4],
                 global_attention=False, drop_path_rate=0.1,
                 alpha=10.0, mlp_ratio=2, drop_rate=0,
                 merging_method="l2norm"):
        super().__init__()
        
        self.num_layers = len(depths)
        self.patch_embed = AFFPatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=embed_dims[0], norm_layer=nn.LayerNorm
        )
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # determine merging method
        if merging_method == "l2norm":
            merging_method = L2NormMerging
        elif merging_method == "cluster":
            merging_method = ClusterMerging
        else:
            raise ValueError(f"merging_method must be from one of the two options: 'l2norm' or 'cluster'! \
                              current method: {merging_method} is not compatible")

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer+1] if (i_layer < self.num_layers - 1) else None,
                cluster_size=cluster_size,
                nbhd_size=nbhd_sizes[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                mlp_ratio=mlp_ratio,
                alpha=alpha,
                ds_rate=ds_rates[i_layer],
                global_attention=global_attention,
                drop=drop_rate,
                downsample=merging_method if (i_layer < self.num_layers - 1) else None,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
            )
            self.layers.append(layer)

        self.norms = nn.ModuleList([nn.LayerNorm(dim) for dim in embed_dims])

        self.init_weights()

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if hasattr(m, 'inp'):
                    if m.inp == 'special':
                        nn.init.normal_(m.weight, mean=0.0, std=0.3)
                        if m.bias is not None: nn.init.constant_(m.bias, 0)
                    elif m.inp == 'norm':
                        nn.init.xavier_uniform_(m.weight)
                    elif m.inp == 'attn':
                        nn.init.trunc_normal_(m.weight, std=0.02)
                    elif m.inp == 'act':
                        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                else:
                    if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                if m.bias is not None: nn.init.constant_(m.bias, 0)
                if m.weight is not None: nn.init.constant_(m.weight, 1.0)

    def forward(self, feat, pos, h, w):
        """
        Returns dictionary of features for the decoder
        """
        outputs = {}
        on_grid = False 

        for i_layer in range(self.num_layers):
            layer = self.layers[i_layer]
            # forward pass through block
            pos_out, x_out, pos, feat = layer(pos, feat, h=h, w=w, on_grid=on_grid, stride=(2**(i_layer+1)))
            
            # apply Norm
            x_norm = self.norms[i_layer](x_out)

            # store for decoder (res2, res3, res4, res5)
            stage_name = f"res{i_layer + 2}" 
            outputs[stage_name] = x_norm
            outputs[f"{stage_name}_pos"] = pos_out
            outputs[f"{stage_name}_spatial_shape"] = (h, w) # canvas size stays constant
            
            on_grid = False
            
        return outputs

    # needed for token visualization
    def forward_with_pos(self, feat, pos, h, w):
        stage_outputs = []
        on_grid = False
        for i_layer in range(self.num_layers):
            layer = self.layers[i_layer]
            pos_out, x_out, pos, feat = layer(pos, feat, h=h, w=w, on_grid=on_grid, stride=2**(i_layer+1))
            stage_outputs.append(pos_out)
            on_grid = False
        return stage_outputs
