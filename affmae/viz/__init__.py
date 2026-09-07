"""Rendering. Imports matplotlib; nothing in ops/, layers/ or models/ may import this.

Sets the Agg backend on import so a renderer never requires a display — the old
renderers relied on whatever the ambient default happened to be.

    from affmae.viz import VizConfig, render_segmentation

    render_segmentation(images, predictions, num_classes=4,
                        save_path="out/pred.png",
                        config=VizConfig(dpi=300, token_radius=3))
"""

import matplotlib

matplotlib.use("Agg")

from .config import DEBUG, OKABE_ITO, PAPER, PRESENTATION, VizConfig  # noqa: E402
from .primitives import (  # noqa: E402
    class_overlay,
    compute_pca_rgb,
    denormalize,
    draw_polylines,
    error_overlay,
    logits_to_labels,
    to_display_image,
)
from .reconstruction import render_reconstruction  # noqa: E402
from .segmentation import (  # noqa: E402
    render_comparison,
    render_segmentation,
    save_segment_overlay,
)
from .tokens import draw_token_positions, render_token_layout  # noqa: E402

__all__ = [
    # Configuration
    "VizConfig", "PAPER", "PRESENTATION", "DEBUG", "OKABE_ITO",
    # Renderers
    "render_segmentation", "render_comparison", "render_reconstruction",
    "render_token_layout", "draw_token_positions", "save_segment_overlay",
    # Primitives
    "denormalize", "to_display_image", "compute_pca_rgb",
    "class_overlay", "error_overlay", "logits_to_labels", "draw_polylines",
]
