"""Shared rendering primitives. """
from typing import Optional

import numpy as np
import torch

from .config import VizConfig

__all__ = [
    "denormalize",
    "to_display_image",
    "compute_pca_rgb",
    "class_overlay",
    "error_overlay",
    "logits_to_labels",
    "draw_polylines",
]


def denormalize(images: torch.Tensor, mean: Optional[float] = None,
                std: Optional[float] = None) -> torch.Tensor:
    """Undo dataset normalization.

    Args:
        images: [..., C, H, W] normalized images.
        mean: scalar mean, or None to use the finetune dataset's.
        std: scalar std, or None likewise.
    Returns:
        Tensor in [0, 1], same shape.

    Note:
        Channel-count agnostic: the statistics are scalars, so this works on
        single-channel micrographs and on RGB alike.
    """
    if mean is None or std is None:
        # Imported lazily: affmae.data pulls OpenCV, and a renderer should not
        # require it just to be imported.
        from affmae.data.stats import IMAGE_MEAN, IMAGE_STD

        mean = IMAGE_MEAN if mean is None else mean
        std = IMAGE_STD if std is None else std
    return (images.float() * float(std) + float(mean)).clamp(0.0, 1.0)


def to_display_image(image: torch.Tensor, config: VizConfig) -> np.ndarray:
    """Prepare one image for use as a figure background.

    Min-max normalizes, then compresses contrast so class overlays stay legible.

    Args:
        image: [H, W] or [1, H, W] or [C, H, W] tensor.
        config: VizConfig supplying ``background_gain`` / ``background_lift``.
    Returns:
        [H, W] (greyscale) or [H, W, C] float array, clipped to [0, 1].
    """
    array = image.detach().float().cpu()
    if array.ndim == 3:
        array = array.squeeze(0) if array.shape[0] == 1 else array.permute(1, 2, 0)
    array = array.numpy()

    lo, hi = float(array.min()), float(array.max())
    array = (array - lo) / (hi - lo + 1e-8)
    return np.clip(array * config.background_gain + config.background_lift, 0.0, 1.0)


def compute_pca_rgb(features: torch.Tensor, eps: float = 1e-8,
                    standardize: bool = False) -> np.ndarray:
    """Project features to 3 channels and scale each to [0, 1] for display.

    Args:
        features: [N, C] feature vectors.
        eps: guard for the per-channel range.
        standardize: z-score each input channel first, so the projection is over
            the correlation matrix rather than the covariance. The paper's
            per-stage decoder figures use this; it stops one high-variance
            channel dominating the three components.
    Returns:
        [N, 3] float array in [0, 1]. All 0.5 for a channel with no range, and
        all zeros when there are fewer than 3 samples to fit.
    Raises:
        ValueError: if ``features`` is not 2-D.
    """
    from sklearn.decomposition import PCA

    if features.ndim != 2:
        raise ValueError(f"features must be [N, C], got {tuple(features.shape)}.")
    flat = features.detach().float().cpu().numpy()
    if flat.shape[0] < 3:
        return np.zeros((flat.shape[0], 3))

    if standardize:
        flat = (flat - flat.mean(axis=0)) / (flat.std(axis=0) + eps)

    reduced = PCA(n_components=3).fit_transform(flat)
    lo = reduced.min(axis=0, keepdims=True)
    hi = reduced.max(axis=0, keepdims=True)
    span = hi - lo
    # A degenerate component would otherwise divide by ~eps and blow up to a
    # saturated channel; mid-grey reads as "no information here".
    return np.where(span > eps, (reduced - lo) / (span + eps), 0.5)


def logits_to_labels(predictions: torch.Tensor) -> torch.Tensor:
    """Reduce model output to a label map.

    Args:
        predictions: [C, H, W] logits or [H, W] labels. A model's ``forward``
            returns a *list* of heads; pass the one you want (conventionally
            ``out[-1]``, the finest).
    Returns:
        [H, W] long tensor of class indices.
    """
    if predictions.ndim == 3:
        return predictions.argmax(dim=0)
    if predictions.ndim == 2:
        return predictions.long()
    raise ValueError(
        f"expected [C, H, W] logits or [H, W] labels, got "
        f"{tuple(predictions.shape)}.")


def class_overlay(labels: torch.Tensor, num_classes: int, config: VizConfig,
                  valid: Optional[torch.Tensor] = None) -> np.ndarray:
    """Build an RGBA overlay colouring each foreground class.

    Args:
        labels: [H, W] class indices.
        num_classes: total classes including background.
        config: VizConfig supplying the palette and alpha.
        valid: optional [H, W] bool; where False the pixel is left transparent
            (used to punch out the error map).
    Returns:
        [H, W, 4] float array.
    """
    labels = labels.detach().cpu()
    overlay = np.zeros((*labels.shape, 4), dtype=np.float32)
    for class_id in range(1, max(2, num_classes)):     # 0 is background
        selected = labels == class_id
        if valid is not None:
            selected = selected & valid.detach().cpu()
        if not selected.any():
            continue
        overlay[selected.numpy()] = (*config.class_color(class_id),
                                     config.overlay_alpha)
    return overlay


def error_overlay(labels: torch.Tensor, target: torch.Tensor,
                  config: VizConfig) -> np.ndarray:
    """Build an RGBA overlay marking mispredicted pixels.

    Args:
        labels: [H, W] predicted class indices.
        target: [H, W] ground-truth indices.
        config: VizConfig supplying ``error_color`` / ``error_alpha``.
    Returns:
        [H, W, 4] float array, transparent where the prediction is correct.
    """
    wrong = (labels.detach().cpu() != target.detach().cpu()).numpy()
    overlay = np.zeros((*wrong.shape, 4), dtype=np.float32)
    overlay[wrong] = (*config.error_color, config.error_alpha)
    return overlay


def draw_polylines(ax, segments, color: str, label: Optional[str] = None,
                   linewidth: float = 1.0, alpha: float = 0.65,
                   marker_size: float = 12.0) -> None:
    """Draw segment skeletons as lines and their slit points as markers.

    Args:
        ax: matplotlib axis to draw on.
        segments: iterable of SegmentPolyline, each with ``skeleton_points_xy``
            and ``slit_points_xy`` as [N, 2] arrays.
        color: matplotlib colour for both the lines and the markers.
        label: legend entry, attached to the first marker set only so the
            legend gets one entry per group rather than one per segment.
        linewidth: skeleton line width.
        alpha: skeleton line opacity. Markers are drawn opaque.
        marker_size: slit marker area, in points squared.
    """
    for segment in segments:
        if segment.skeleton_points_xy.size > 0:
            ax.plot(segment.skeleton_points_xy[:, 0],
                    segment.skeleton_points_xy[:, 1],
                    color=color, linewidth=linewidth, alpha=alpha)
        if segment.slit_points_xy.size > 0:
            ax.scatter(segment.slit_points_xy[:, 0],
                       segment.slit_points_xy[:, 1],
                       c=color, s=marker_size, marker="o", label=label)
            label = None
