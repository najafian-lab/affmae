"""Foot-process-width geometry evaluation from PGBMI and slit segmentations.

The metrics themselves live in :mod:`affmae.eval.fpw_geometry`; this module is
the driver that runs a model over a split and aggregates per-segment results.

Tunables travel in :class:`FpwParams` rather than an ``argparse.Namespace``, so
the functions are callable from a notebook or another project without building a
parser first.
"""

import copy
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from affmae.data.finetune_dataset import build_test_dataloader
from affmae.eval.fpw_geometry import (
    extract_segment_polylines,
    match_segments_iou,
    segment_metrics,
    summarize_values,
)
from affmae.eval.loader import load_for_eval, resolve_checkpoint
from affmae.eval.segmentation import iterate_predictions
from affmae.utils.misc import set_seed

__all__ = ["FpwParams", "evaluate_fpw", "evaluate_fpw_across_seeds",
           "evaluate_image", "aggregate_results", "summarize_across_seeds",
           "format_seed_summary", "json_safe", "parse_grid_size"]

# Pooled as a {count, mean, std} block across seeds. Foot-process width is the
# reported metric; segment_match_rate rides along because an FPW error over three
# matched segments and one over three hundred are not comparable numbers.
_SEED_POOLED_KEYS = ("fpw_mean_abs_error", "segment_match_rate")


@dataclass(frozen=True)
class FpwParams:
    """Thresholds and scales for one FPW evaluation.

    Args:
        pgbmi_class: label index of the basement membrane class.
        slit_class: label index of the slit class.
        slit_min_area: smallest accepted slit blob, in pixels.
        slit_max_area: largest accepted slit blob, in pixels.
        slit_circularity: minimum circularity for a slit blob.
        pgbmi_dilate: dilation radius applied to the PGBMI mask, in pixels.
        min_pgbmi_area: smallest accepted PGBMI component, in pixels.
        min_segment_iou: IoU above which a predicted segment matches a GT one.
        pixel_size_nm: nanometres per pixel, for physical widths.
        eval_grid_size: (width, height) reference grid that pixel distances are
            rescaled to, so runs at different input resolutions compare.
    """

    pgbmi_class: int = 1
    slit_class: int = 2
    slit_min_area: float = 4.0
    slit_max_area: float = 400.0
    slit_circularity: float = 0.4
    pgbmi_dilate: int = 3
    min_pgbmi_area: float = 16.0
    min_segment_iou: float = 0.1
    pixel_size_nm: float = 1.0
    eval_grid_size: tuple[int, int] = (1024, 1024)

    def extract(self, mask: np.ndarray, rng: np.random.Generator) -> list:
        """Extract PGBMI segments with slits from a label mask.

        Args:
            mask: [H, W] integer label mask.
            rng: generator used for tie-breaking inside the extractor.
        Returns:
            List of SegmentPolyline.
        """
        return extract_segment_polylines(
            mask,
            pgbmi_class=self.pgbmi_class,
            slit_class=self.slit_class,
            pgbmi_dilate=self.pgbmi_dilate,
            slit_min_area=self.slit_min_area,
            slit_max_area=self.slit_max_area,
            slit_min_circularity=self.slit_circularity,
            min_pgbmi_area=self.min_pgbmi_area,
            rng=rng,
        )


def parse_grid_size(value: str) -> tuple[int, int]:
    """Parse ``"N"`` or ``"W,H"`` into a (width, height) pair.

    Args:
        value: str, one or two comma-separated integers.
    Returns:
        (width, height).
    Raises:
        ValueError: on any other shape.
    """
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) == 1:
        return int(parts[0]), int(parts[0])
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise ValueError(f"expected a grid size as N or W,H; got {value!r}")


def evaluate_image(gt_mask: np.ndarray, pred_mask: np.ndarray,
                   params: FpwParams, rng: np.random.Generator) -> dict:
    """Match predicted segments to ground truth and measure each pair.

    Args:
        gt_mask: [H, W] ground-truth label mask.
        pred_mask: [H, W] predicted label mask, same shape.
        params: thresholds and scales.
        rng: generator for tie-breaking inside the extractor.
    Returns:
        dict of per-segment results plus the segment counts for this image.
    """
    gt_segments = params.extract(gt_mask, rng)
    pred_segments = params.extract(pred_mask, rng)
    matches, unmatched_gt, unmatched_pred = match_segments_iou(
        gt_segments, pred_segments, min_iou=params.min_segment_iou)

    mask_h, mask_w = gt_mask.shape
    target_w, target_h = params.eval_grid_size
    grid_scale_xy = (target_w / mask_w, target_h / mask_h)

    segment_results = []
    insufficient = 0
    for gt_idx, pred_idx, iou in matches:
        metrics = segment_metrics(
            gt_segments[gt_idx], pred_segments[pred_idx],
            pixel_size=params.pixel_size_nm, grid_scale_xy=grid_scale_xy)
        if metrics["fpw_arc_count_gt"] == 0 or metrics["fpw_arc_count_pred"] == 0:
            insufficient += 1
        segment_results.append({
            "gt_segment_id": gt_segments[gt_idx].segment_id,
            "pred_segment_id": pred_segments[pred_idx].segment_id,
            "segment_iou": iou,
            **metrics,
        })

    return {
        "gt_segments": gt_segments,
        "pred_segments": pred_segments,
        "segment_results": segment_results,
        "num_gt_segments": len(gt_segments),
        "num_pred_segments": len(pred_segments),
        "image_grid_size": {"width": int(mask_w), "height": int(mask_h)},
        "num_matched_segments": len(matches),
        "num_unmatched_gt_segments": len(unmatched_gt),
        "num_unmatched_pred_segments": len(unmatched_pred),
        "num_insufficient_slit_pairs": insufficient,
    }


def aggregate_results(images: list[dict]) -> dict:
    """Pool per-image results into one summary.

    Args:
        images: per-image records from :func:`evaluate_image`.
    Returns:
        dict of segment counts plus one metric block, ``fpw_mean_abs_error``,
        measured on the reference evaluation grid (1024x1024 by default) so runs
        at different input resolutions stay comparable.
    """
    segments = [seg for image in images for seg in image["segments"]]
    total_gt = sum(image["num_gt_segments"] for image in images)
    total_pred = sum(image["num_pred_segments"] for image in images)
    total_matched = sum(image["num_matched_segments"] for image in images)

    def values(key: str) -> list[float]:
        return [float(seg[key]) for seg in segments if key in seg]

    return {
        "num_images": len(images),
        "num_gt_segments": total_gt,
        "num_pred_segments": total_pred,
        "num_matched_segments": total_matched,
        "num_unmatched_gt_segments": sum(
            image["num_unmatched_gt_segments"] for image in images),
        "num_unmatched_pred_segments": sum(
            image["num_unmatched_pred_segments"] for image in images),
        "segment_match_rate": (float(total_matched / total_gt) if total_gt
                               else float("nan")),
        "num_insufficient_slit_pairs": sum(
            image["num_insufficient_slit_pairs"] for image in images),
        "fpw_mean_abs_error": summarize_values(values("fpw_mean_abs_error")),
    }


def _summary_mean(summary: dict, key: str) -> float:
    """Read a metric's mean whether it is a scalar or a {mean: ...} block."""
    value = summary.get(key)
    if isinstance(value, dict):
        return float(value.get("mean", float("nan")))
    return float("nan") if value is None else float(value)


def summarize_across_seeds(seed_outputs: list[dict]) -> dict:
    """Pool per-seed summaries into mean and std per metric.

    Args:
        seed_outputs: one entry per seed, each with a ``summary``.
    Returns:
        dict with ``num_seeds``, ``seeds``, and a {count, mean, std} block per
        metric. Non-finite seeds are dropped before averaging.
    """
    summary: dict[str, Any] = {
        "num_seeds": len(seed_outputs),
        "seeds": [run["seed"] for run in seed_outputs],
    }
    for key in _SEED_POOLED_KEYS:
        values = np.asarray([_summary_mean(run["summary"], key)
                             for run in seed_outputs], dtype=np.float64)
        values = values[np.isfinite(values)]
        summary[key] = {
            "count": int(values.size),
            "mean": float(values.mean()) if values.size else float("nan"),
            "std": float(values.std(ddof=0)) if values.size else float("nan"),
        }
    return summary


def format_seed_summary(cross_seed_summary: dict) -> str:
    """Render a cross-seed summary as ``metric: mean +/- std`` lines."""
    lines = ["", "Cross-seed summary (mean +/- std)"]
    for key, value in cross_seed_summary.items():
        if not isinstance(value, dict) or "mean" not in value:
            continue
        mean, std = value["mean"], value["std"]
        lines.append(f"{key}: n/a" if mean is None or std is None
                     else f"{key}: {mean:.6f} +/- {std:.6f}")
    return "\n".join(lines)


def json_safe(value: Any) -> Any:
    """Convert numpy scalars to Python and non-finite floats to None.

    Args:
        value: any nested structure of dicts, lists, tuples and scalars.
    Returns:
        The same structure, JSON-serialisable.
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def evaluate_fpw(cfg: Any, params: Optional[FpwParams] = None, *, seed: int = 42,
                 checkpoint: Optional[str] = None,
                 vis_dir: Optional[str] = None,
                 max_vis: Optional[int] = None) -> dict:
    """Run one FPW evaluation over the test split.

    Args:
        cfg: a loaded Config.
        params: thresholds and scales; defaults to :class:`FpwParams`.
        seed: seed for both torch and the extractor's tie-breaking.
        checkpoint: explicit checkpoint path, or None for the run's default.
        vis_dir: directory to write per-image overlays into, or None to skip.
        max_vis: stop writing overlays after this many images.
    Returns:
        dict with ``summary`` and per-image ``images`` records.
    Raises:
        FileNotFoundError: if the checkpoint does not exist.
    """
    params = params or FpwParams()
    set_seed(seed)
    ckpt_path = resolve_checkpoint(cfg, checkpoint)
    model = load_for_eval(cfg, ckpt_path)
    loader = build_test_dataloader(cfg)
    rng = np.random.default_rng(seed)

    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    image_results = []
    for index, (images, targets, logits, paths) in enumerate(
            iterate_predictions(model, loader, cfg)):
        pred_mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)
        gt_mask = targets[0].cpu().numpy().astype(np.int64)
        evaluated = evaluate_image(gt_mask, pred_mask, params, rng)

        path = _first_path(paths)
        image_results.append({
            "image_index": index,
            "path": path,
            **{k: evaluated[k] for k in (
                "num_gt_segments", "num_pred_segments", "image_grid_size",
                "num_matched_segments", "num_unmatched_gt_segments",
                "num_unmatched_pred_segments", "num_insufficient_slit_pairs")},
            "segments": evaluated["segment_results"],
        })

        if vis_dir and (max_vis is None or index < max_vis):
            from affmae.viz.segmentation import save_segment_overlay
            stem = os.path.splitext(os.path.basename(path))[0] or f"image_{index:04d}"
            save_segment_overlay(
                images[0], evaluated["gt_segments"], evaluated["pred_segments"],
                os.path.join(vis_dir, f"{index:04d}_{stem}.png"))

    summary = aggregate_results(image_results)
    logging.info("FPW summary: %s", summary)
    return {
        "config": getattr(cfg, "_source_path", cfg.name),
        "seed": seed,
        "checkpoint": ckpt_path,
        "model_type": cfg.model_type,
        "classes": {"pgbmi": params.pgbmi_class, "slit": params.slit_class},
        "pixel_size_nm": params.pixel_size_nm,
        "eval_grid_size": {"width": params.eval_grid_size[0],
                           "height": params.eval_grid_size[1]},
        "summary": summary,
        "images": image_results,
    }


def evaluate_fpw_across_seeds(cfg: Any, seeds: list[int],
                              params: Optional[FpwParams] = None, *,
                              checkpoint: Optional[str] = None,
                              vis_dir: Optional[str] = None,
                              max_vis: Optional[int] = None) -> dict:
    """Run :func:`evaluate_fpw` once per seed and pool the results.

    Each seed evaluates ``<output_dir>/<name>_seed<seed>/last_model.pth`` unless
    ``checkpoint`` is given, in which case ``{seed}`` in it is formatted.

    Args:
        cfg: a loaded Config; ``name`` is suffixed per seed.
        seeds: seeds to evaluate.
        params: thresholds and scales.
        checkpoint: explicit checkpoint template, or None.
        vis_dir: parent directory for per-seed overlay subdirectories.
        max_vis: overlay cap per seed.
    Returns:
        dict with ``cross_seed_summary`` and one entry per seed.
    """
    params = params or FpwParams()
    outputs = []
    for seed in seeds:
        seed_cfg = copy.deepcopy(cfg)
        seed_cfg.name = f"{cfg.name}_seed{seed}"
        seed_cfg.device = cfg.device
        outputs.append(evaluate_fpw(
            seed_cfg, params, seed=seed,
            checkpoint=resolve_checkpoint(cfg, checkpoint, seed),
            vis_dir=os.path.join(vis_dir, f"seed{seed}") if vis_dir else None,
            max_vis=max_vis))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "config": getattr(cfg, "_source_path", cfg.name),
        "model_type": cfg.model_type,
        "seeds": seeds,
        "eval_grid_size": {"width": params.eval_grid_size[0],
                           "height": params.eval_grid_size[1]},
        "cross_seed_summary": summarize_across_seeds(outputs),
        "seed_results": [{"seed": run["seed"], "checkpoint": run["checkpoint"],
                          "summary": run["summary"]} for run in outputs],
    }


def _first_path(paths: Any) -> str:
    """Unwrap the collated path field down to a single string."""
    if isinstance(paths, (list, tuple)):
        first = paths[0]
        return str(first[0]) if isinstance(first, (list, tuple)) else str(first)
    return str(paths)
