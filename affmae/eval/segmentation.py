"""Segmentation metrics over a labelled test split. """

import logging
from typing import Any, Iterator, Optional

import torch

from affmae.data.finetune_dataset import build_test_dataloader
from affmae.eval.loader import load_for_eval
from affmae.eval.metrics import compute_dice, compute_iou
from affmae.models.registry import get_model_spec
from affmae.utils.dist import autocast_context, synchronize
from affmae.utils.misc import AverageMeter

__all__ = ["iterate_predictions", "evaluate_segmentation",
           "collect_predictions", "compare_models", "compare_backends"]


def _final_logits(outputs):
    """Deep supervision returns a list; the last entry is the full-resolution head."""
    return outputs[-1] if isinstance(outputs, (list, tuple)) else outputs


def _batch_paths(paths):
    """Unwrap the collated path field, which nests one level for tuple datasets."""
    return paths[0] if isinstance(paths, (tuple, list)) else paths


def iterate_predictions(model: torch.nn.Module, loader, cfg: Any
                        ) -> Iterator[tuple[torch.Tensor, torch.Tensor,
                                            torch.Tensor, Any]]:
    """Run the model over ``loader``, yielding one batch at a time.

    Args:
        model: a model in eval mode.
        loader: dataloader yielding ``(images, targets, paths)``.
        cfg: a loaded Config; supplies ``device``.
    Yields:
        ``(images, targets, logits, paths)``. Images and targets are on
        ``cfg.device``; logits are the final head, in float32.
    """
    with torch.no_grad():
        for images, targets, paths in loader:
            images = images.to(cfg.device, non_blocking=True)
            targets = targets.to(cfg.device, non_blocking=True).long()
            with autocast_context(cfg.device):
                outputs = model(images)
            yield images, targets, _final_logits(outputs).float(), paths


def evaluate_segmentation(cfg: Any, checkpoint: Optional[str] = None) -> dict:
    """Compute per-class IoU and Dice over the test split.

    Args:
        cfg: a loaded Config with a ``test`` folder under ``base_path``.
        checkpoint: explicit checkpoint path, or None for the run's
            ``last_model.pth``.
    Returns:
        dict with ``model``, ``images``, ``miou``, ``dice``, ``per_class_iou``
        and ``per_class_dice``.
    """
    model = load_for_eval(cfg, checkpoint)
    loader = build_test_dataloader(cfg)

    iou_meter, dice_meter = AverageMeter(), AverageMeter()
    class_iou = torch.zeros(cfg.num_classes - 1, device=cfg.device)
    class_dice = torch.zeros(cfg.num_classes - 1, device=cfg.device)
    seen = 0

    for images, targets, logits, _paths in iterate_predictions(model, loader, cfg):
        ious = compute_iou(logits, targets)
        dices = compute_dice(logits, targets)
        iou_meter.update(ious.mean().item(), images.size(0))
        dice_meter.update(dices.mean().item(), images.size(0))
        class_iou += ious.sum(dim=0)
        class_dice += dices.sum(dim=0)
        seen += images.size(0)

    per_class_iou = (class_iou / max(seen, 1)).tolist()
    per_class_dice = (class_dice / max(seen, 1)).tolist()
    name = get_model_spec(cfg.model_type).name

    logging.info("=" * 56)
    logging.info("%-10s mIoU %.4f    Dice %.4f", name,
                 iou_meter.avg, dice_meter.avg)
    for index, (iou, dice) in enumerate(zip(per_class_iou, per_class_dice), 1):
        logging.info("  class %d: IoU %.4f  Dice %.4f", index, iou, dice)
    logging.info("=" * 56)

    return {"model": name, "images": seen, "miou": iou_meter.avg,
            "dice": dice_meter.avg, "per_class_iou": per_class_iou,
            "per_class_dice": per_class_dice}


def collect_predictions(cfg: Any, checkpoint: Optional[str] = None,
                        loader=None) -> list[dict]:
    """Gather per-image tensors for rendering, sorted by path.

    Sorting matters: the comparison figure indexes several models' results by
    position, so they have to agree on the order.

    Args:
        cfg: a loaded Config.
        checkpoint: explicit checkpoint path, or None for the run's default.
        loader: dataloader to reuse. Several models being compared must see the
            same samples, so the caller supplies one loader for all of them.
    Returns:
        List of dicts with ``image``, ``target``, ``logits`` and ``path``, all
        tensors on CPU.
    """
    model = load_for_eval(cfg, checkpoint)
    loader = build_test_dataloader(cfg) if loader is None else loader

    results = []
    for images, targets, logits, paths in iterate_predictions(model, loader, cfg):
        images, targets, logits = images.cpu(), targets.cpu(), logits.cpu()
        names = _batch_paths(paths)
        for index in range(images.size(0)):
            results.append({"image": images[index], "target": targets[index],
                            "logits": logits[index], "path": names[index]})
    results.sort(key=lambda record: record["path"])
    logging.info("collected %d prediction(s) for %s", len(results), cfg.name)
    return results


def compare_models(cfgs: list, labels: list[str], save_path: str,
                   indices: Optional[list[int]] = None,
                   zoom_boxes=None, checkpoints: Optional[list] = None) -> str:
    """Render one figure comparing several trained models on the same samples.

    Models are loaded and released one at a time; these are full-resolution
    backbones and holding them all at once does not fit on one GPU.

    Args:
        cfgs: loaded Configs, one per model. The first supplies the dataloader,
            so every model sees identical samples.
        labels: display name per model, parallel to ``cfgs``.
        save_path: output figure path.
        indices: which test samples to draw, one row each.
        zoom_boxes: optional (x, y, w, h) per row for the inset.
        checkpoints: explicit checkpoint per model, or None for the defaults.
    Returns:
        ``save_path``.
    Raises:
        ValueError: if labels or checkpoints disagree in length with ``cfgs``.
    """
    from affmae.viz.segmentation import render_comparison

    if len(labels) != len(cfgs):
        raise ValueError(f"got {len(cfgs)} config(s) but {len(labels)} label(s).")
    checkpoints = checkpoints or [None] * len(cfgs)
    if len(checkpoints) != len(cfgs):
        raise ValueError(
            f"got {len(cfgs)} config(s) but {len(checkpoints)} checkpoint(s).")

    loader = build_test_dataloader(cfgs[0])
    per_model = []
    for cfg, label, checkpoint in zip(cfgs, labels, checkpoints):
        logging.info("evaluating %s (%s)", label, cfg.model_type)
        per_model.append(collect_predictions(cfg, checkpoint, loader=loader))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    counts = {len(records) for records in per_model}
    if len(counts) != 1:
        raise ValueError(f"models disagree on sample count: {counts}")

    images = torch.stack([r["image"] for r in per_model[0]])
    targets = torch.stack([r["target"] for r in per_model[0]])
    predictions = [torch.stack([r["logits"] for r in records])
                   for records in per_model]

    return render_comparison(
        images=images, predictions_per_model=predictions, model_names=labels,
        num_classes=cfgs[0].num_classes, save_path=save_path,
        targets=targets, indices=indices, zoom_boxes=zoom_boxes)


#: Kernel settings that make the decoder take the lattice-snapped fused path.
FUSED_BACKENDS = {
    "decoder_deform_backend": "csr_knn_cached",
    "decoder_knn_cache": True,
    "cluster_attention_backend": "flash_nbhd_attn",
}


def compare_backends(cfg, checkpoint: Optional[str] = None,
                     variants: Optional[dict] = None,
                     time_it: bool = True) -> dict:
    """Score one checkpoint through several kernel backends on the same split.

    The fused decoder (``csr_knn_cached``) snaps each sampled location to its
    lattice cell and reuses that cell's precomputed top-4 neighbours, while
    ``unfused`` takes the exact 4 nearest neighbours of the continuous location.
    They therefore compute different functions -- so switching a trained model
    between them is only safe if the metric barely moves, and that has to be
    measured on real data rather than assumed.

    Every variant is loaded with identical weights (the backend is not a
    parameter) and sees the same images, so any difference is attributable to
    the kernel path alone.

    Args:
        cfg: a loaded Config. The first variant always uses it unchanged.
        checkpoint: explicit checkpoint path, or None for the run's default.
        variants: label -> dict of config overrides. Defaults to comparing the
            config as shipped against :data:`FUSED_BACKENDS`.
        time_it: also measure per-image latency for each variant. Kernels are
            warmed first; this is still a coarse in-loop timer, so prefer
            a dedicated benchmark for a careful latency comparison.
    Returns:
        dict with a ``variants`` list -- each carrying ``miou``, ``dice``,
        per-class values, ``pixel_agreement`` with the baseline and, when
        ``time_it``, ``latency_ms`` -- plus a ``deltas`` summary.
    """
    import copy
    import time

    variants = variants or {"as shipped (config)": {},
                            "fused (lattice-snapped)": dict(FUSED_BACKENDS)}
    loader = build_test_dataloader(cfg)

    results, baseline_labels = [], None
    for label, overrides in variants.items():
        variant_cfg = copy.deepcopy(cfg)
        for key, value in overrides.items():
            setattr(variant_cfg, key, value)
        variant_cfg.device = cfg.device

        model = load_for_eval(variant_cfg, checkpoint)

        if time_it:
            # Warm the kernels before timing anything. The fused path compiles
            # more Triton kernels than the unfused one, and without this its
            # first batches carry that compilation -- which made the fused
            # variant look ~2x slower than a careful standalone measurement.
            warm = next(iter(loader))[0].to(cfg.device, non_blocking=True)
            with torch.no_grad():
                for _ in range(3):
                    model(warm)
            synchronize(warm)
            del warm

        iou_meter, dice_meter = AverageMeter(), AverageMeter()
        class_iou = torch.zeros(cfg.num_classes - 1, device=cfg.device)
        class_dice = torch.zeros(cfg.num_classes - 1, device=cfg.device)
        seen, agree, pixels, elapsed = 0, 0, 0, 0.0
        labels_here = []

        for index, (images, targets, logits, _paths) in enumerate(
                iterate_predictions(model, loader, cfg)):
            if time_it:
                synchronize(images)
                start = time.perf_counter()
                with torch.no_grad():
                    model(images)
                synchronize(images)
                elapsed += (time.perf_counter() - start) / images.size(0) * 1000

            ious = compute_iou(logits, targets)
            dices = compute_dice(logits, targets)
            iou_meter.update(ious.mean().item(), images.size(0))
            dice_meter.update(dices.mean().item(), images.size(0))
            class_iou += ious.sum(dim=0)
            class_dice += dices.sum(dim=0)
            seen += images.size(0)

            predicted = logits.argmax(dim=1).cpu()
            labels_here.append(predicted)
            if baseline_labels is not None and index < len(baseline_labels):
                reference = baseline_labels[index]
                agree += (predicted == reference).sum().item()
                pixels += reference.numel()

        entry = {
            "label": label,
            "overrides": overrides,
            "images": seen,
            "miou": iou_meter.avg,
            "dice": dice_meter.avg,
            "per_class_iou": (class_iou / max(seen, 1)).tolist(),
            "per_class_dice": (class_dice / max(seen, 1)).tolist(),
            "pixel_agreement": (agree / pixels) if pixels else 1.0,
        }
        if time_it:
            entry["latency_ms"] = elapsed / max(1, len(labels_here))
        results.append(entry)
        if baseline_labels is None:
            baseline_labels = labels_here
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    base = results[0]
    for entry in results[1:]:
        entry["delta_miou"] = entry["miou"] - base["miou"]
        entry["delta_dice"] = entry["dice"] - base["dice"]
        if time_it and base.get("latency_ms"):
            entry["speedup"] = base["latency_ms"] / entry["latency_ms"]

    _log_backend_table(results, base)
    return {"baseline": base["label"], "variants": results}


def _log_backend_table(results, base):
    """Log the comparison as one table, so the trade-off is visible at a glance."""
    logging.info("=" * 84)
    logging.info("%-26s %8s %8s %9s %9s %10s %8s", "backend", "mIoU", "Dice",
                 "d mIoU", "d Dice", "agreement", "speedup")
    logging.info("-" * 84)
    for entry in results:
        logging.info(
            "%-26s %8.4f %8.4f %9s %9s %9.2f%% %8s",
            entry["label"][:26], entry["miou"], entry["dice"],
            f"{entry.get('delta_miou', 0.0):+.4f}" if entry is not base else "-",
            f"{entry.get('delta_dice', 0.0):+.4f}" if entry is not base else "-",
            entry["pixel_agreement"] * 100,
            f"{entry['speedup']:.2f}x" if entry.get("speedup") else "-")
    logging.info("=" * 84)
