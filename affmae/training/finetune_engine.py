"""Shared finetuning loop. """

import copy
import logging
import os
import time

import torch
import torch.nn as nn

from affmae.data.finetune_dataset import build_finetune_dataloader, build_test_dataloader
from affmae.eval.metrics import compute_iou
from affmae.training.losses import ComboLoss, DiceLoss, FocalLoss
from affmae.models.registry import get_model_spec
from affmae.training.optimizer import build_optimizer_with_llrd
from affmae.utils.dist import resolve_device, unwrap_model
from affmae.utils.misc import (
    AverageMeter,
    cosine_lr_schedule,
    set_seed,
    setup_logging,
    strip_module_prefix,
)

__all__ = ["get_amp_dtype", "train_epoch", "validate", "run_finetune", "build_loss_fn"]

# Gradient-norm clip threshold.
GRAD_CLIP_NORM = 5.0

# Deep-supervision weights for the res5 / res4 auxiliary heads. The primary head
# (res2) carries weight 1.0.
AUX_LOSS_WEIGHTS = (0.05, 0.12)

# Not every shipped config defines these, so they are read with getattr rather
# than as attributes. Values match what the pre-consolidation scripts hardcoded.
DEFAULT_SEED = 123           # finetune.py called set_seed(123)
DEFAULT_TEST_EVAL_FREQ = 25  # `if epoch % 25 == 0`
DEFAULT_NUM_ACCUM = 1


def get_amp_dtype(cfg):
    """Resolve ``cfg.amp_dtype`` to a torch dtype.

    Args:
        cfg: Config, may define ``amp_dtype`` as one of fp16/float16/bf16/bfloat16.
    Returns:
        torch.dtype, defaulting to float16.
    Raises:
        ValueError: on an unrecognized value.
    """
    amp_dtype = getattr(cfg, "amp_dtype", "float16")
    if amp_dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if amp_dtype in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")


def _scaler_scale(loss_scaler):
    """GradScaler scale for diagnostics, tolerating a disabled scaler."""
    try:
        return loss_scaler.get_scale()
    except Exception:
        return "unknown"


def build_loss_fn(cfg):
    """Construct the segmentation loss named by ``cfg.loss_fn``."""
    weights = torch.tensor(cfg.class_weighting, device=cfg.device)
    if cfg.loss_fn == "focal":
        return FocalLoss(alpha=weights)
    if cfg.loss_fn == "combo":
        return ComboLoss(alpha=weights, gamma=2.0)
    if cfg.loss_fn == "dice":
        return DiceLoss()
    return nn.CrossEntropyLoss(weight=weights)


def _combine_losses(pred_logits, targets, loss_fn):
    """Weighted deep-supervision loss.

    Args:
        pred_logits: list of logit tensors, coarse-to-fine, primary head last.
        targets: [B,H,W] label tensor.
        loss_fn: callable, per-head loss.
    Returns:
        (per_head_losses, total_loss).
    """
    if len(pred_logits) > 1:
        per_head = [loss_fn(logits, targets) for logits in pred_logits]
        total = per_head[-1]
        for weight, aux in zip(AUX_LOSS_WEIGHTS, per_head[:-1]):
            total = total + weight * aux
        return per_head, total
    per_head = [loss_fn(pred_logits[0], targets)]
    return per_head, per_head[-1]


def train_epoch(model, dataloader, optimizer, loss_fn, epoch, loss_scaler, cfg,
                global_step, aux_names, throttle_s=0.0):
    """Train one epoch.

    Args:
        model: nn.Module in train mode.
        dataloader: iterable of (images, targets, paths).
        optimizer: torch optimizer.
        loss_fn: callable, per-head loss.
        epoch: int, current epoch.
        loss_scaler: torch.amp.GradScaler.
        cfg: Config.
        global_step: int, optimizer steps so far.
        aux_names: tuple of str, from ``spec.aux_names``.
        throttle_s: float, optional per-batch sleep. The original scripts slept
            5 ms every batch; kept configurable and defaulted off.
    Returns:
        (primary_head_loss, current_lr, global_step).
    Raises:
        FloatingPointError: on non-finite logits or loss, rather than letting
            NaNs propagate silently into the weights.
    """
    model.train()
    meters = {name: AverageMeter() for name in aux_names}

    total_batches = len(dataloader)
    amp_dtype = get_amp_dtype(cfg)
    num_accum = getattr(cfg, "num_accum", DEFAULT_NUM_ACCUM)
    # In optimizer steps, not batches: global_step only advances at an
    # accumulation boundary, so counting batches made num_accum=2 stretch warmup
    # to 2x the configured epochs and leave the cosine at its midpoint after the
    # last epoch. A no-op at num_accum=1, which is what every shipped config
    # uses -- so this only ever bit someone trading batch size for accumulation
    # to fit a smaller card.
    steps_per_epoch = -(-total_batches // num_accum)
    max_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    for batch_idx, (images, targets, _) in enumerate(dataloader):
        images = images.to(cfg.device)
        targets = targets.to(cfg.device).long()

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            pred_logits = model(images)
            finite = [torch.isfinite(logits).all().item() for logits in pred_logits]
            if not all(finite):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Non-finite logits at epoch {epoch} batch {batch_idx}: "
                    f"finite_per_head={finite} "
                    f"lr={optimizer.param_groups[0]['lr']:.8f} "
                    f"scaler={_scaler_scale(loss_scaler)}")

            per_head, loss = _combine_losses(pred_logits, targets, loss_fn)
            loss = loss / num_accum

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch} batch {batch_idx}: "
                f"loss={loss.item()} lr={optimizer.param_groups[0]['lr']:.8f} "
                f"scaler={_scaler_scale(loss_scaler)}")

        for name, value in zip(aux_names, per_head):
            meters[name].update(value.item(), images.size(0))

        loss_scaler.scale(loss).backward()

        at_boundary = ((batch_idx + 1) % num_accum == 0
                       or (batch_idx + 1) == total_batches)
        if at_boundary:
            loss_scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                        GRAD_CLIP_NORM)
            if not torch.isfinite(total_norm):
                # Skip the step rather than abort: a single overflowing batch is
                # recoverable, and GradScaler will back the scale off.
                logging.warning(
                    "Epoch %d batch %d: non-finite grad norm (%s); skipping step.",
                    epoch, batch_idx, total_norm.item())
                optimizer.zero_grad(set_to_none=True)
                loss_scaler.update()
                continue

            loss_scaler.step(optimizer)
            loss_scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            cosine_lr_schedule(optimizer, global_step, max_steps,
                               cfg.learning_rate, cfg.min_lr, warmup_steps)

        if throttle_s:
            torch.cuda.synchronize()
            time.sleep(throttle_s)

    return (meters[aux_names[-1]].avg,
            optimizer.param_groups[0]["lr"],
            global_step)


def validate(model, dataloader, cfg, aux_names, worst_k=20):
    """Evaluate mIoU over a dataloader.

    Args:
        model: nn.Module.
        dataloader: iterable of (images, targets, paths).
        cfg: Config.
        aux_names: tuple of str, from ``spec.aux_names``.
        worst_k: int, how many lowest-IoU samples to return for visualization.
    Returns:
        (meters, per_class_iou, images, targets, predictions, paths) where
        ``meters[aux_names[-1]].avg`` is the primary-head mIoU.
    """
    model.eval()
    meters = {name: AverageMeter() for name in aux_names}

    class_iou_sums = torch.zeros(cfg.num_classes - 1, device=cfg.device)
    total_images = 0
    samples = []
    amp_dtype = get_amp_dtype(cfg)

    with torch.no_grad():
        for images, targets, paths in dataloader:
            images = images.to(cfg.device)
            targets = targets.to(cfg.device).long()

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                pred_logits = model(images)
                per_head = [compute_iou(logits, targets) for logits in pred_logits]
                for name, value in zip(aux_names, per_head):
                    meters[name].update(value.mean().item(), images.size(0))

            batch_ious = compute_iou(pred_logits[-1], targets)
            class_iou_sums += batch_ious.sum(dim=0)
            total_images += images.size(0)

            per_sample_iou = batch_ious.mean(dim=1)
            img_cpu, tgt_cpu = images.cpu(), targets.cpu()
            pred_cpu = pred_logits[-1].cpu()
            batch_paths = paths[0] if isinstance(paths, (tuple, list)) else paths

            for b in range(images.size(0)):
                samples.append({
                    "img": img_cpu[b], "tgt": tgt_cpu[b], "pred": pred_cpu[b],
                    "path": batch_paths[b], "iou": per_sample_iou[b].item(),
                })

    per_class_iou = class_iou_sums / (total_images + 1e-6)
    samples.sort(key=lambda s: s["iou"])
    worst = samples[:worst_k]

    return (meters, per_class_iou,
            torch.stack([s["img"] for s in worst], dim=0),
            torch.stack([s["tgt"] for s in worst], dim=0),
            torch.stack([s["pred"] for s in worst], dim=0),
            [s["path"] for s in worst])


def load_pretrained(model, spec, cfg):
    """Load and adapt the pretrained checkpoint named by ``cfg``."""
    logging.info("Loading pretrained weights from: %s", cfg.pretrained_ckpt_path)
    checkpoint = torch.load(cfg.pretrained_ckpt_path, map_location="cpu",
                            weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    state_dict = strip_module_prefix(state_dict)
    state_dict = spec.adapt_state_dict(state_dict, unwrap_model(model), cfg)

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        logging.warning("Missing keys: %s", incompatible.missing_keys)
    return model


def _split_has_images(base_path: str, split: str) -> bool:
    """True if ``base_path/split`` holds at least one image and one mask.

    A directory that exists but is empty is not a usable split: the dataset
    builds its file list with ``np.vectorize``, which raises on a size-0 input
    rather than yielding an empty dataset.

    Args:
        base_path: dataset root holding train/val/test.
        split: subdirectory name.
    Returns:
        True only if both images/ and masks/ are non-empty.
    """
    root = os.path.join(base_path, split)
    try:
        return all(bool(os.listdir(os.path.join(root, kind)))
                   for kind in ("images", "masks"))
    except OSError:
        return False


def run_finetune(base_cfg, seed=None, train_loader_fn=None, wandb_run=None,
                 throttle_s=0.0):
    """Run one finetuning job end to end.

    Args:
        base_cfg: Config. Deep-copied, so callers can reuse it across seeds.
        seed: int or None. When given, the seed is set and ``_seed{seed}`` is
            appended to the experiment name so runs land in separate directories.
        train_loader_fn: callable or None, ``(cfg) -> DataLoader`` overriding the
            training loader. Used by the percent-data sweep.
        wandb_run: callable or None, ``(cfg) -> None`` to start experiment
            tracking. None disables tracking regardless of ``cfg.wandb_enabled``.
        throttle_s: float, per-batch sleep passed to :func:`train_epoch`.
    Returns:
        dict with ``name``, ``seed``, ``best_val_miou``, ``test_miou`` and
        ``exp_dir``; ``test_miou`` is None when the dataset has no test split.
    """
    # Imported here, not at module scope: the training engine should not pull
    # matplotlib/cv2 just to be imported. See tests/test_import_hygiene.py.
    # Model-coupled figures live in affmae.viz.model_figures; everything that
    # renders plain tensors is in the other affmae.viz modules.
    from affmae.viz import PAPER, render_segmentation
    from affmae.viz.model_figures import render_tokens

    cfg = copy.deepcopy(base_cfg)
    if seed is not None:
        cfg.name = f"{cfg.name}_seed{seed}"

    exp_dir = os.path.join(cfg.output_dir, cfg.name)
    os.makedirs(exp_dir, exist_ok=True)
    set_seed(getattr(cfg, "seed", DEFAULT_SEED) if seed is None else seed)

    # Re-point logging at this run's directory; without clearing handlers a
    # multi-seed sweep keeps writing into the first run's log file.
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    setup_logging(exp_dir)

    cfg.device = resolve_device(getattr(cfg, 'device', None))
    logging.info("===== Starting run '%s'%s =====", cfg.name,
                 f" (seed {seed})" if seed is not None else "")

    if wandb_run is not None:
        wandb_run(cfg)

    train_dl = (train_loader_fn(cfg) if train_loader_fn is not None
                else build_finetune_dataloader(cfg, is_train=True))
    # Built only when the split exists, mirroring the test split below. The
    # released fpwdata ships an empty val/, and this call is unconditional and
    # ahead of the epoch loop -- so every shipped finetune config crashed here
    # before training a single step, whatever start_eval_epoch said.
    has_val_split = _split_has_images(cfg.base_path, "val")
    val_dl = build_finetune_dataloader(cfg, is_train=False) if has_val_split else None
    if not has_val_split:
        logging.warning(
            "No images under %s/val; validation and best_model.pth are "
            "skipped. Test metrics are unaffected.", cfg.base_path)

    spec = get_model_spec(cfg.model_type)
    logging.info("Building segmentation model '%s'", spec.name)
    model = load_pretrained(spec.build_segmentation(cfg), spec, cfg).to(cfg.device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logging.info("Model initialized with %.2fM trainable parameters.", trainable)

    layer_decay = getattr(cfg, "layer_decay", spec.default_llrd)
    optimizer = build_optimizer_with_llrd(model, cfg, layer_decay,
                                          spec.layer_decay_plan)
    scaler = torch.amp.GradScaler(enabled=get_amp_dtype(cfg) == torch.float16)
    loss_fn = build_loss_fn(cfg)

    # Same check as val: isdir() alone passes for an empty directory, which
    # then raises inside the dataset rather than skipping evaluation.
    has_test_split = _split_has_images(cfg.base_path, "test")
    test_dl = build_test_dataloader(cfg) if has_test_split else None
    if not has_test_split:
        logging.warning("No 'test' split under %s; skipping test evaluation.",
                        cfg.base_path)

    best_val_miou = -1.0
    global_step = 0
    val_miou = None
    test_eval_freq = getattr(cfg, "test_eval_freq", DEFAULT_TEST_EVAL_FREQ)

    for epoch in range(cfg.epochs):
        train_loss, current_lr, global_step = train_epoch(
            model, train_dl, optimizer, loss_fn, epoch, scaler, cfg,
            global_step, spec.aux_names, throttle_s=throttle_s)
        logging.info("Epoch %d: Train Loss %.4f | LR %.8f",
                     epoch, train_loss, current_lr)

        if test_dl is not None and epoch % test_eval_freq == 0:
            test_meters, test_class_iou, *_ = validate(
                model, test_dl, cfg, spec.aux_names)
            miou_str = " ".join(f"{n.upper()}: {test_meters[n].avg:.4f}"
                                for n in spec.aux_names)
            logging.info("TEST mIoU: %s", miou_str)
            logging.info("TEST class-wise IoU: [%s]", " | ".join(
                f"C{i + 1}: {iou:.4f}" for i, iou in enumerate(test_class_iou)))

        if (val_dl is not None and epoch >= cfg.start_eval_epoch
                and epoch % cfg.log_freq == 0):
            val_meters, val_class_iou, val_xs, val_ys, val_preds, val_paths = validate(
                model, val_dl, cfg, spec.aux_names)
            # validate() returns a dict of meters. The pre-consolidation drivers
            # assigned it straight to `val_miou` and then did `1.0 - val_miou`
            # and `val_miou > best`, which raises TypeError — so this whole
            # branch, and best_model.pth with it, never ran.
            val_miou = val_meters[spec.aux_names[-1]].avg
            logging.info("Validation mIoU: %.4f", val_miou)

            if val_miou > best_val_miou:
                best_val_miou = val_miou
                logging.info("New best mIoU %.4f; writing best_model.pth", val_miou)
                logging.info("Class-wise IoU: [%s]", " | ".join(
                    f"C{i + 1}: {iou:.4f}" for i, iou in enumerate(val_class_iou)))
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_miou": best_val_miou,
                }, os.path.join(exp_dir, "best_model.pth"))
                render_segmentation(
                    val_xs, val_preds, cfg.num_classes,
                    os.path.join(exp_dir, f"val_results_epoch{epoch}.png"),
                    targets=val_ys, titles=val_paths, config=PAPER)

        if wandb_run is not None:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": current_lr,
                **({"val_mIoU": val_miou, "val_loss": 1.0 - val_miou}
                   if val_miou is not None else {}),
            })

    test_miou = None
    if test_dl is not None:
        logging.info("Evaluating last model on the test set")
        test_meters, test_class_iou, test_xs, test_ys, test_preds, test_paths = validate(
            model, test_dl, cfg, spec.aux_names)
        test_miou = test_meters[spec.aux_names[-1]].avg
        logging.info("Last model test mIoU: %.4f", test_miou)
        logging.info("Last model class-wise IoU: [%s]", " | ".join(
            f"C{i + 1}: {iou:.4f}" for i, iou in enumerate(test_class_iou)))

        torch.save({
            "epoch": cfg.epochs - 1,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "test_miou": test_miou,
        }, os.path.join(exp_dir, "last_model.pth"))

        render_segmentation(
            test_xs, test_preds, cfg.num_classes,
            os.path.join(exp_dir, "TEST_LAST_results.png"),
            targets=test_ys, titles=test_paths, config=PAPER)
        if spec.supports_token_viz:
            render_tokens(unwrap_model(model), test_xs.to(cfg.device), 6,
                          os.path.join(exp_dir, "TEST_LAST_token_loc.png"), seed=7)

    if wandb_run is not None:
        import wandb
        wandb.finish()

    return {"name": cfg.name, "seed": seed, "best_val_miou": best_val_miou,
            "test_miou": test_miou, "exp_dir": exp_dir}
