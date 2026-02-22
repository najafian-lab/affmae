from typing import Tuple
import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from perlin_noise import perlin_noise

# masking implementations

def random_masking(x, mask_ratio):
    """Perform per-sample random masking by shuffling."""
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    ids_masked = ids_shuffle[:, len_keep:]

    return ids_keep, ids_masked, ids_restore

def random_masking_grouped(x: torch.Tensor, img_size: int, encoder_patch_size, mask_ratio=None, masking_block_size: int = 8):
    """Perform per-sample random grouped masking by shuffling."""
    masking_block_size = encoder_patch_size # disable group masking for now
    assert masking_block_size >= encoder_patch_size, f"masking block size ({masking_block_size}) must be equal \
    or larger than encoder patch size ({encoder_patch_size})"

    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    coarse_grid_size = img_size // masking_block_size
    block_downsample_ratio = masking_block_size // encoder_patch_size
    noise_coarse = torch.rand(N, coarse_grid_size, coarse_grid_size, device=x.device)
    noise_fine = noise_coarse.repeat_interleave(block_downsample_ratio, dim=1).repeat_interleave(block_downsample_ratio, dim=2)
    ids_shuffle = torch.argsort(noise_fine.view(N, -1), dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    ids_masked = ids_shuffle[:, len_keep:]
    return ids_keep, ids_masked, ids_restore

# uses perlin noise
def perlin_masking(
    x: torch.Tensor, 
    img_size: int,
    encoder_patch_size: int,
    mask_ratio: float,
    perlin_grid_shape: Tuple[int, int] = (8, 8),
    use_fbm: bool = False,
    fbm_octaves: int = 3,
    fbm_persistence: float = 0.5,
    fbm_lacunarity: float = 2.0,
    generator: torch.Generator = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs MAE-style masking using Perlin or fBM noise with a fixed mask ratio.
    
    The noise map is used to sort tokens, ensuring blocky/continuous
    regions are masked together while maintaining a fixed mask_ratio.
    """
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    grid_hw = img_size // encoder_patch_size
    patch_grid_shape = (grid_hw, grid_hw)

    # generate noise map (either single Perlin or multi-octave fBM)
    if not use_fbm:
        if N < 4:
            noise = perlin_noise(
                grid_shape=perlin_grid_shape,
                out_shape=patch_grid_shape,
                batch_size=N,
                generator=generator,
                device=x.device, 
                dtype=x.dtype
            )  # Shape: [N, H, W]
        else:        
            group_size = math.ceil(N / 4)
            noise_grids = []
            for ind, grid_size in enumerate([8, 16, 32, 64][:N]):  # grid sizes (need up to N)
                perlin_grid_shape = (grid_size, grid_size)
                noise = perlin_noise(
                    grid_shape=perlin_grid_shape,
                    out_shape=patch_grid_shape,
                    batch_size=group_size,
                    generator=generator,
                    device=x.device,
                    dtype=x.dtype
                )  # Shape: [N, H, W]
                noise_grids.append(noise)
            noise = torch.cat(noise_grids, dim=0)[:N]
    else:
        # fmb noise
        total_noise = torch.zeros(N, *patch_grid_shape, device=x.device, dtype=x.dtype)
        amplitude = 1.0
        max_amplitude = 0.0
        current_h, current_w = perlin_grid_shape

        for _ in range(fbm_octaves):
            current_grid_shape_octave = (int(current_h), int(current_w))
            
            octave_noise = perlin_noise(
                grid_shape=current_grid_shape_octave,
                out_shape=patch_grid_shape,
                batch_size=N,
                generator=generator, 
                device=x.device,
                dtype=x.dtype
            )
            
            total_noise += amplitude * octave_noise
            max_amplitude += amplitude
            
            # update amplitude and frequency for next octave
            amplitude *= fbm_persistence
            current_h *= fbm_lacunarity
            current_w *= fbm_lacunarity
            
        # normalize fBM noise to approx [0, 1] range
        noise = total_noise / max_amplitude  # Shape: [N, H, W]

    # flatten the 2D noise maps to 1D token lists
    noise_flat = noise.reshape(N, -1)  # Shape: [N, L]
    assert noise_flat.shape[1] == L, f"Noise map length ({noise_flat.shape[1]}) does not match token length L ({L})."

    # sort tokens based on the noise values
    ids_shuffle = torch.argsort(noise_flat, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # separate into keep and mask indices
    ids_keep = ids_shuffle[:, :len_keep]
    ids_masked = ids_shuffle[:, len_keep:]

    return ids_keep, ids_masked, ids_restore