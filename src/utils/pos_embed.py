import numpy as np
import torch
import torch.nn.functional as F
import math


# assumes largest input resolution is 2048 x 2048
# rel_pos_width = 2048 // 4 - 1
rel_pos_width = 1024 // 8 - 1
table_width = 2 * rel_pos_width + 1

pre_hs = torch.arange(table_width).float()-rel_pos_width
pre_ws = torch.arange(table_width).float()-rel_pos_width
pre_ys, pre_xs = torch.meshgrid(pre_hs, pre_ws)  # table_width x table_width

# expanded relative position lookup table
dis_table = (pre_ys**2 + pre_xs**2) ** 0.5
sin_table = pre_ys / dis_table
cos_table = pre_xs / dis_table
pre_table = torch.stack([pre_xs, pre_ys, dis_table, sin_table, cos_table], dim=2)  # table_width x table_width x 5
pre_table[torch.bitwise_or(pre_table.isnan(), pre_table.isinf()).nonzero(as_tuple=True)] = 0
pre_table = pre_table.reshape(-1, 5)
pre_table_fp32 = pre_table.to(torch.float32)
pre_table_fp16 = pre_table.to(torch.float16)


def assert_strides(name, t, expect_shape=None):
    if expect_shape is not None:
        assert tuple(t.shape) == tuple(expect_shape), f"{name} shape {t.shape} != {expect_shape}"
    assert t.is_contiguous(), f"{name} not contiguous: stride={t.stride()}"


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """Generate 2D sin-cos position embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Generate position embeddings from grid."""
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D position embeddings."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    return np.concatenate([emb_sin, emb_cos], axis=1)

# standard vit interpolate pos embed
def interpolate_pos_embed(state_dict, new_model, key='pos_embed'):
    """
    Interpolate the positional embeddings in the state_dict to match the new model's size.
    
    Args:
        state_dict (dict): The checkpoint state dictionary to load.
        new_model (nn.Module): The new model instantiated with larger img_size.
        key (str): The key of the positional embedding parameter (e.g., 'pos_embed').
    """
    if key not in state_dict:
        return

    # Get the source pos_embed from checkpoint
    pos_embed_checkpoint = state_dict[key]
    embedding_size = pos_embed_checkpoint.shape[-1]
    
    # Get the number of patches in the checkpoint (excluding CLS token)
    num_patches_checkpoint = pos_embed_checkpoint.shape[1] - 1
    
    # Get the number of patches in the new model (excluding CLS token)
    num_patches_new = new_model.patch_embed.num_patches
    
    # If shapes match, no need to interpolate
    if num_patches_checkpoint == num_patches_new:
        return

    # Determine grid sizes
    # We assume square images: grid_size = sqrt(num_patches)
    src_grid_size = int(math.sqrt(num_patches_checkpoint))
    dst_grid_size = int(math.sqrt(num_patches_new))
    
    print(f"Interpolating {key} from {src_grid_size}x{src_grid_size} to {dst_grid_size}x{dst_grid_size}")

    # Extract CLS token and Grid tokens
    # shape: (1, N+1, Dim)
    cls_token = pos_embed_checkpoint[:, :1, :]
    grid_tokens = pos_embed_checkpoint[:, 1:, :]
    
    # Reshape grid tokens to (1, Dim, H, W) for interpolation
    grid_tokens = grid_tokens.reshape(1, src_grid_size, src_grid_size, embedding_size).permute(0, 3, 1, 2)
    
    # Perform bicubic interpolation
    new_grid_tokens = F.interpolate(
        grid_tokens, 
        size=(dst_grid_size, dst_grid_size), 
        mode='bicubic', 
        align_corners=False
    )
    
    # Reshape back to (1, N_new, Dim)
    new_grid_tokens = new_grid_tokens.permute(0, 2, 3, 1).reshape(1, dst_grid_size * dst_grid_size, embedding_size)
    
    # Concatenate CLS token back
    new_pos_embed = torch.cat((cls_token, new_grid_tokens), dim=1)
    
    # Update state_dict
    state_dict[key] = new_pos_embed