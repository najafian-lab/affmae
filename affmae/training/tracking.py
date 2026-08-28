"""W&B run setup, shared by both training stages. 

Every function is a no-op off rank 0, so callers do not need to guard. """

import logging
import os

from affmae.utils.dist import is_main_process
from affmae.utils.env import wandb_available

__all__ = ["start_run", "log", "finish", "is_active"]

_ACTIVE = False


def is_active() -> bool:
    """True if a run was started and is being logged to."""
    return _ACTIVE


def start_run(config, *, name=None, project=None, entity=None,
              resume=None, reinit=False) -> bool:
    """Start a W&B run, or degrade to no tracking.

    A missing key is not an error: training should not abort because logging is
    unavailable. ``config.wandb_enabled`` is set to False when tracking cannot
    start, so downstream ``if config.wandb_enabled`` checks stay correct.

    Args:
        config: Config; read for ``wandb_enabled`` and the destination defaults,
            and mutated to False if tracking cannot start.
        name: run name; defaults to ``experiment_name`` or ``name`` on the config.
        project: overrides ``config.project``; WANDB_PROJECT wins over both.
        entity: overrides ``config.entity``; WANDB_ENTITY wins over both.
        resume: passed through to ``wandb.init``.
        reinit: passed through, for multi-run drivers like ``--seeds``.
    Returns:
        True if tracking is active on this rank.
    """
    global _ACTIVE
    _ACTIVE = False

    if not getattr(config, "wandb_enabled", False):
        return False
    if not is_main_process():
        # Every rank calling wandb.init creates duplicate runs.
        return False
    if not wandb_available():
        logging.warning(
            "wandb_enabled is set but no credentials found. Put WANDB_API_KEY "
            "in .env (see .env.example), export it, or run `wandb login`. "
            "Continuing without tracking.")
        config.wandb_enabled = False
        return False

    import wandb

    try:
        wandb.init(
            project=os.environ.get(
                "WANDB_PROJECT", project or getattr(config, "project", None)),
            entity=os.environ.get(
                "WANDB_ENTITY", entity or getattr(config, "entity", None)),
            name=name or getattr(config, "experiment_name", None)
                 or getattr(config, "name", None),
            config=vars(config),
            resume=resume,
            reinit=reinit,
        )
    except Exception as error:
        logging.error("Failed to initialize W&B: %s; continuing without it.",
                      error)
        config.wandb_enabled = False
        return False

    logging.info("W&B run initialized successfully.")
    _ACTIVE = True
    return True


def log(metrics: dict, step=None) -> None:
    """Log a metrics dict if a run is active. No-op otherwise."""
    if not _ACTIVE:
        return
    import wandb

    wandb.log(metrics, step=step)


def finish() -> None:
    """Close the run if one is active. No-op otherwise."""
    global _ACTIVE
    if not _ACTIVE:
        return
    import wandb

    wandb.finish()
    _ACTIVE = False
