"""MAE reconstruction rendering: input, masked input, per-stage predictions."""

import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import torch

from .config import PAPER, VizConfig
from .primitives import to_display_image

__all__ = ["render_reconstruction"]


def render_reconstruction(images: torch.Tensor,
                          masked: Optional[torch.Tensor],
                          reconstructions: Sequence[torch.Tensor],
                          save_path: str,
                          stage_names: Optional[Sequence[str]] = None,
                          config: VizConfig = PAPER,
                          show_residual: bool = False) -> str:
    """Render a reconstruction grid.

    Columns are: original, masked input (if given), one per reconstruction, and
    optionally the residual of the last one.

    Args:
        images: [B, C, H, W] originals.
        masked: [B, C, H, W] masked inputs, or None to omit that column.
        reconstructions: per-stage [B, C, H, W] predictions, coarse to fine.
        save_path: output image path.
        stage_names: labels for the reconstruction columns.
        config: VizConfig.
        show_residual: append ``|original - finest|``.
    Returns:
        ``save_path``.
    """
    if not reconstructions:
        raise ValueError("reconstructions is empty; nothing to render.")

    count = min(images.shape[0], config.max_samples)
    columns = 1 + (1 if masked is not None else 0) + len(reconstructions) \
        + (1 if show_residual else 0)

    headings = ["Original"]
    if masked is not None:
        headings.append("Masked")
    for index in range(len(reconstructions)):
        if stage_names is not None and index < len(stage_names):
            headings.append(stage_names[index])
        else:
            headings.append(f"Recon {index + 1}")
    if show_residual:
        headings.append("Residual")

    size = config.figsize_per_cell
    fig, axes = plt.subplots(count, columns,
                             figsize=(size * columns, size * count),
                             squeeze=False)

    for row in range(count):
        panels = [images[row]]
        if masked is not None:
            panels.append(masked[row])
        panels.extend(r[row] for r in reconstructions)
        if show_residual:
            panels.append((images[row] - reconstructions[-1][row]).abs())

        for column, panel in enumerate(panels):
            shown = to_display_image(panel, config)
            ax = axes[row][column]
            ax.imshow(shown, cmap=config.cmap if shown.ndim == 2 else None,
                      vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0 and config.show_titles:
                ax.set_title(headings[column], fontsize=config.font_size)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path
