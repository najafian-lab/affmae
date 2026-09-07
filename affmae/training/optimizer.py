"""Layer-wise learning-rate decay (LLRD) optimizer construction. """

import logging

import torch

from affmae.utils.dist import unwrap_model

logger = logging.getLogger(__name__)

# Parameters matching any of these train at the full base learning rate: they
# are newly initialized for the downstream task, not part of the pretrained
# trunk.
_FULL_LR_KEYS = (
    "decoder", "mask_token", "head", "masked_token", "multi_scale_generator",
)

# Never weight-decay these, on top of the usual 1D/bias rule. The Swin
# relative_position_bias_table is 2D but documented as no-decay upstream.
_NO_DECAY_KEYS = (
    "pos_embed", "cls_token", "masked_token", "relative_position_bias_table",
)


def _no_weight_decay(name, param):
    """True if this parameter should be excluded from weight decay."""
    return (
        param.ndim == 1
        or name.endswith(".bias")
        or any(k in name for k in _NO_DECAY_KEYS)
    )


def build_optimizer_with_llrd(model, cfg, layer_decay, plan_fn=None):
    """Build an AdamW optimizer with layer-wise LR decay.

    Args:
        model: nn.Module, possibly wrapped; unwrapped internally before any
            attribute access, so this is safe to call with a DDP-wrapped model.
        cfg: Config, needs ``learning_rate`` and ``weight_decay``.
        layer_decay: float, per-layer LR multiplier. Typically
            ``spec.default_llrd``.
        plan_fn: callable or None, ``(model, cfg) -> LayerDecayPlan``. Typically
            ``spec.layer_decay_plan``. When None, every trunk parameter trains
            at the base rate (no decay).
    Returns:
        torch.optim.AdamW over the grouped parameters.
    """
    model = unwrap_model(model)
    base_lr = cfg.learning_rate
    weight_decay = cfg.weight_decay

    if plan_fn is None:
        logger.warning(
            "No layer-decay plan for model_type '%s'; training the whole trunk "
            "at the base learning rate.", getattr(cfg, "model_type", "?")
        )
        plan = None
        num_layers = 1
    else:
        plan = plan_fn(model, cfg)
        num_layers = max(1, plan.num_layers)

    param_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        this_wd = 0.0 if _no_weight_decay(name, param) else weight_decay

        if any(k in name for k in _FULL_LR_KEYS):
            scale = 1.0
        elif plan is None:
            scale = 1.0
        else:
            layer_id = plan.layer_id(name)
            scale = layer_decay ** (num_layers - 1 - layer_id)

        param_groups.setdefault((scale, this_wd), []).append(param)

    grouped = [
        {
            "params": params,
            "weight_decay": wd,
            "lr": base_lr * scale,
            "initial_lr": base_lr * scale,
        }
        for (scale, wd), params in param_groups.items()
    ]

    _log_llrd_table(grouped, base_lr, layer_decay, num_layers)
    return torch.optim.AdamW(grouped)


def _log_llrd_table(grouped, base_lr, layer_decay, num_layers):
    """Log one row per distinct LR scale, highest first."""
    logger.info("=" * 50)
    logger.info(" LLRD CONFIGURATION CHECK (decay=%s, layers=%d)", layer_decay, num_layers)
    logger.info("=" * 50)
    logger.info("%-25s | %-15s | %-15s", "Layer / Group", "LR Multiplier", "Actual LR")
    logger.info("-" * 60)

    seen = set()
    for group in sorted(grouped, key=lambda g: g["lr"], reverse=True):
        scale = group["lr"] / base_lr
        if abs(scale - 1.0) < 1e-6:
            label = "Decoder / Head"
        elif abs(scale - layer_decay) < 1e-6:
            label = "Backbone TOP Layer"
        elif abs(scale - layer_decay ** num_layers) < 1e-6:
            label = "Backbone BOTTOM Layer"
        else:
            label = "Backbone Intermediate"

        key = round(scale, 5)
        if key not in seen:
            logger.info("%-25s | %-15.4f | %-15.2e", label, scale, group["lr"])
            seen.add(key)

    logger.info("=" * 50)
