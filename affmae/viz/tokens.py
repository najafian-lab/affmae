"""Token-layout rendering: where the adaptive tokens land, per encoder stage. """

import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import PAPER, VizConfig
from .primitives import to_display_image

__all__ = ["draw_token_positions", "render_token_layout"]


def draw_token_positions(image: torch.Tensor, positions: torch.Tensor,
                         patch_size: int, config: VizConfig = PAPER) -> np.ndarray:
    """Draw token centres on one image.

    Args:
        image: [C, H, W] or [H, W] background.
        positions: [N, 2] token positions in *patch* units, (x, y).
        patch_size: pixels per patch, to convert to pixel coordinates.
        config: VizConfig; supplies radius, colour and supersampling.
    Returns:
        [H', W', 3] uint8 RGB array, where H' = H * token_render_scale.
    """
    import cv2

    background = to_display_image(image, config)
    if background.ndim == 2:
        background = np.stack([background] * 3, axis=-1)
    canvas = (background * 255.0).astype(np.uint8)

    scale = float(config.token_render_scale)
    if scale != 1.0:
        canvas = cv2.resize(
            canvas, (int(canvas.shape[1] * scale), int(canvas.shape[0] * scale)),
            interpolation=cv2.INTER_NEAREST)

    radius = max(1, int(round(config.resolve_token_radius(image.shape[-1]) * scale)))
    # No channel reversal: cv2.circle does not interpret colour, it writes these
    # values into the array's channels, and this canvas is RGB (built from
    # to_display_image) and documented as RGB. Reversing here returned BGR, so a
    # red token came out blue for any direct caller.
    colour = tuple(int(round(c * 255)) for c in config.token_color)

    canvas = np.ascontiguousarray(canvas)
    coords = positions.detach().float().cpu().numpy()
    for x, y in coords:
        px = int((x * patch_size + patch_size / 2) * scale)
        py = int((y * patch_size + patch_size / 2) * scale)
        cv2.circle(canvas, (px, py), radius, colour, -1)
    return canvas


def render_token_layout(images: torch.Tensor,
                        positions_per_stage: Sequence[torch.Tensor],
                        patch_size: int, save_path: str,
                        stage_names: Optional[Sequence[str]] = None,
                        config: VizConfig = PAPER) -> str:
    """Render a grid of images x encoder stages showing token positions.

    Args:
        images: [B, C, H, W] inputs.
        positions_per_stage: one [B, N_s, 2] tensor per stage, in patch units.
        patch_size: pixels per patch at the input resolution.
        save_path: output image path.
        stage_names: column labels; defaults to ``Stage 1..k`` with token counts.
        config: VizConfig.
    Returns:
        ``save_path``.

    Note:
        Does not change the model's train/eval mode — it never touches the
        model at all, only tensors the caller already produced.
    """
    count = min(images.shape[0], config.max_samples)
    stages = len(positions_per_stage)
    if stages == 0:
        raise ValueError("positions_per_stage is empty; nothing to render.")

    size = config.figsize_per_cell
    fig, axes = plt.subplots(count, stages,
                             figsize=(size * stages, size * count),
                             squeeze=False)

    for row in range(count):
        for column, positions in enumerate(positions_per_stage):
            rendered = draw_token_positions(images[row], positions[row],
                                            patch_size, config)
            ax = axes[row][column]
            ax.imshow(rendered)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0 and config.show_titles:
                if stage_names is not None and column < len(stage_names):
                    label = stage_names[column]
                else:
                    label = f"Stage {column + 1}: {positions[row].shape[0]} tokens"
                ax.set_title(label, fontsize=config.font_size)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path
