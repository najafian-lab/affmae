"""Segmentation renderers. """

import os
from typing import Optional, Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import torch

from .config import PAPER, VizConfig
from .primitives import (
    class_overlay,
    denormalize,
    draw_polylines,
    error_overlay,
    logits_to_labels,
    to_display_image,
)

__all__ = ["render_segmentation", "render_comparison",
           "save_segment_overlay"]


def _save(fig, path, config):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


def _style_axis(ax, title, config):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)
    if title and config.show_titles:
        ax.set_title(title, fontsize=config.font_size)


def render_segmentation(images: torch.Tensor, predictions: torch.Tensor,
                        num_classes: int, save_path: str,
                        targets: Optional[torch.Tensor] = None,
                        titles: Optional[Sequence[str]] = None,
                        config: VizConfig = PAPER) -> str:
    """Render predicted masks over their inputs.

    Args:
        images: [B, C, H, W] inputs.
        predictions: [B, K, H, W] logits or [B, H, W] labels.
        num_classes: classes including background.
        save_path: output image path.
        targets: [B, H, W] ground truth. When given, a GT column and an error
            column are added; when None only input and prediction are drawn.
        titles: per-row captions, e.g. filenames.
        config: VizConfig.
    Returns:
        ``save_path``.
    """
    count = min(images.shape[0], config.max_samples)
    with_gt = targets is not None
    columns = 4 if with_gt else 2
    names = (["Input", "Ground Truth", "Prediction", "Errors"] if with_gt
             else ["Input", "Prediction"])

    size = config.figsize_per_cell
    fig, axes = plt.subplots(count, columns,
                             figsize=(size * columns, size * count),
                             squeeze=False)

    for row in range(count):
        background = to_display_image(images[row], config)
        labels = logits_to_labels(predictions[row])
        cmap = config.cmap if background.ndim == 2 else None

        panels = [(background, None)]
        if with_gt:
            panels.append((background, class_overlay(targets[row], num_classes, config)))
        panels.append((background, class_overlay(labels, num_classes, config)))
        if with_gt:
            panels.append((background, error_overlay(labels, targets[row], config)))

        for column, (base, overlay) in enumerate(panels):
            ax = axes[row][column]
            ax.imshow(base, cmap=cmap, vmin=0, vmax=1)
            if overlay is not None:
                ax.imshow(overlay)
            heading = names[column] if row == 0 else None
            if column == 0 and titles is not None and row < len(titles):
                heading = f"{names[0]}\n{titles[row]}" if row == 0 else str(titles[row])
            _style_axis(ax, heading, config)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    _save(fig, save_path, config)
    return save_path


def render_comparison(images: torch.Tensor,
                      predictions_per_model: Sequence[torch.Tensor],
                      model_names: Sequence[str], num_classes: int,
                      save_path: str,
                      targets: Optional[torch.Tensor] = None,
                      indices: Optional[Sequence[int]] = None,
                      zoom_boxes: Optional[Sequence[Sequence[int]]] = None,
                      config: VizConfig = PAPER) -> str:
    """Render several models' predictions side by side.

    Takes any number of models. With ``targets`` each model's panel is shaded
    correct-vs-error; without it the panels are plain class overlays.

    Args:
        images: [B, C, H, W] inputs.
        predictions_per_model: one [B, K, H, W] (or [B, H, W]) tensor per model.
        model_names: column labels, parallel to ``predictions_per_model``.
        num_classes: classes including background.
        save_path: output image path.
        targets: optional [B, H, W] ground truth; adds a GT column and switches
            each model's panel to correct-vs-error shading.
        indices: which samples to draw; defaults to the first ``max_samples``.
        zoom_boxes: optional (x, y, w, h) per row -- or one box for every row --
            which crops the model columns to that region and marks it on the
            input column with leader lines. Defaults to no zoom.
        config: VizConfig.
    Returns:
        ``save_path``.
    Raises:
        ValueError: if names and predictions disagree in length.
    """
    if len(predictions_per_model) != len(model_names):
        raise ValueError(
            f"got {len(predictions_per_model)} prediction sets but "
            f"{len(model_names)} names.")

    rows = list(indices) if indices is not None else list(
        range(min(images.shape[0], config.max_samples)))

    if zoom_boxes is not None:
        boxes = list(zoom_boxes)
        if len(boxes) == 4 and all(isinstance(v, (int, float)) for v in boxes):
            boxes = [tuple(boxes)] * len(rows)   # one box, applied to every row
        if len(boxes) != len(rows):
            raise ValueError(
                f"got {len(boxes)} zoom box(es) for {len(rows)} row(s); pass "
                f"one box per row or a single (x, y, w, h).")
    else:
        boxes = None
    with_gt = targets is not None
    columns = len(model_names) + (2 if with_gt else 1)

    size = config.figsize_per_cell
    fig, axes = plt.subplots(len(rows), columns,
                             figsize=(size * columns, size * len(rows)),
                             squeeze=False)

    headings = ["Input"] + (["Ground Truth"] if with_gt else []) + list(model_names)

    for row_idx, sample in enumerate(rows):
        background = to_display_image(images[sample], config)
        cmap = config.cmap if background.ndim == 2 else None

        overlays = [None]
        if with_gt:
            overlays.append(class_overlay(targets[sample], num_classes, config))
        for predictions in predictions_per_model:
            labels = logits_to_labels(predictions[sample])
            if with_gt:
                correct = labels.cpu() == targets[sample].detach().cpu()
                layer = class_overlay(labels, num_classes, config, valid=correct)
                wrong = error_overlay(labels, targets[sample], config)
                layer = np.where(wrong[..., 3:] > 0, wrong, layer)
            else:
                layer = class_overlay(labels, num_classes, config)
            overlays.append(layer)

        for column, overlay in enumerate(overlays):
            ax = axes[row_idx][column]
            ax.imshow(background, cmap=cmap, vmin=0, vmax=1)
            if overlay is not None:
                ax.imshow(overlay)
            _style_axis(ax, headings[column] if row_idx == 0 else None, config)

        if boxes is not None:
            _apply_zoom(axes[row_idx], boxes[row_idx], len(overlays), config)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    _save(fig, save_path, config)
    return save_path


def save_segment_overlay(image: torch.Tensor, gt_segments, pred_segments,
                         save_path: str, config: VizConfig = PAPER,
                         gt_color: str = "lime",
                         pred_color: str = "red") -> str:
    """Draw ground-truth and predicted segment geometry over one image.

    Args:
        image: [C, H, W] or [H, W] normalized image tensor.
        gt_segments: iterable of SegmentPolyline for the ground truth.
        pred_segments: iterable of SegmentPolyline for the prediction.
        save_path: file to write.
        config: VizConfig controlling dpi and normalization.
        gt_color: colour for the ground-truth geometry.
        pred_color: colour for the predicted geometry.
    Returns:
        ``save_path``.
    """
    # denormalize rather than to_display_image: the latter compresses contrast
    # so class overlays stay legible, which is the wrong trade-off when the
    # marks are thin lines and points over a greyscale micrograph.
    background = denormalize(image).detach().cpu()
    if background.ndim == 3:
        background = background[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(background.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    draw_polylines(ax, gt_segments, color=gt_color, label="GT slits")
    draw_polylines(ax, pred_segments, color=pred_color, label="Pred slits")
    ax.set_axis_off()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right")
    fig.tight_layout(pad=0)
    _save(fig, save_path, config)
    return save_path


def _apply_zoom(row_axes, box, num_columns, config):
    """Crop the model columns to ``box`` and mark it on the input column.

    Args:
        row_axes: the axes of one figure row.
        box: (x, y, w, h) in pixel coordinates.
        num_columns: how many columns this row actually filled.
        config: VizConfig supplying the outline colour and width.
    """
    left, top, width, height = (int(v) for v in box)
    first_zoom = 1 if num_columns > 1 else 0

    for column in range(first_zoom, num_columns):
        ax = row_axes[column]
        ax.set_xlim(left, left + width)
        ax.set_ylim(top + height, top)   # inverted: image rows run downward

    if first_zoom == 0:
        return

    source = row_axes[0]
    source.add_patch(patches.Rectangle(
        (left, top), width, height, linewidth=config.zoom_box_width,
        edgecolor=config.zoom_box_color, facecolor="none"))
    # Leader lines from the box's right edge to the first zoomed panel, so the
    # reader can see which region the zoom came from.
    for corner, target in (((left + width, top), (0, 1)),
                           ((left + width, top + height), (0, 0))):
        source.add_artist(ConnectionPatch(
            xyA=corner, coordsA="data", xyB=target, coordsB="axes fraction",
            axesA=source, axesB=row_axes[first_zoom],
            color=config.zoom_box_color, linewidth=config.zoom_box_width,
            linestyle=":"))
