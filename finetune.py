"""Finetune a pretrained backbone for segmentation.

Replaces the previous ``finetune.py`` / ``finetune_multi_seed.py`` /
``finetune_seed_runs.py`` trio, which were the same loop with different
hardcoded seed lists. Repeat a run over seeds with ``--seeds``:

    python finetune.py --config configs/aff_base_finetune_512_fpw.yaml
    python finetune.py --config configs/aff_base_finetune_512_fpw.yaml --seeds 42 77 2026

With ``--seeds``, each run gets its own ``<name>_seed<N>`` output directory and a
mean/std summary is printed at the end.
"""

import logging
import statistics
from argparse import ArgumentParser

from affmae.config import load_config
from affmae.training import tracking
from affmae.training.finetune_engine import run_finetune
from affmae.utils.env import load_dotenv

# Defaults for W&B destination; override with WANDB_ENTITY / WANDB_PROJECT.
DEFAULT_WANDB_ENTITY = "najafian-lab-2025"
DEFAULT_WANDB_PROJECT = "aff-mae-finetune"


def make_wandb_starter(enabled):
    """Return a ``(cfg) -> None`` W&B initializer, or None to skip tracking.

    Delegates to :mod:`affmae.training.tracking`, which both stages share; this
    used to be a second copy of pretraining's setup with different defaults.
    """
    if not enabled:
        return None

    def start(cfg):
        tracking.start_run(cfg, name=getattr(cfg, "name", None),
                           project=DEFAULT_WANDB_PROJECT,
                           entity=DEFAULT_WANDB_ENTITY, reinit=True)

    return start


def summarize(results):
    """Log a mean/std table over multi-seed results."""
    scored = [r for r in results if r["test_miou"] is not None]
    logging.info("=" * 62)
    logging.info("%-32s %10s %10s", "run", "best val", "test")
    for r in results:
        test = f"{r['test_miou']:.4f}" if r["test_miou"] is not None else "n/a"
        logging.info("%-32s %10.4f %10s", r["name"], r["best_val_miou"], test)

    if len(scored) > 1:
        values = [r["test_miou"] for r in scored]
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        logging.info("-" * 62)
        logging.info("test mIoU over %d seeds: %.4f +/- %.4f", len(values), mean, std)
    logging.info("=" * 62)


def main():
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=str, required=True,
                        help="Path to a YAML config.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        metavar="SEED",
                        help="Repeat the run once per seed, each in its own "
                             "output directory. Omit for a single run.")
    parser.add_argument("--throttle-ms", type=float, default=0.0,
                        help="Sleep this long after each batch. The original "
                             "scripts slept 5 ms unconditionally; default off.")
    args = parser.parse_args()

    load_dotenv()
    base_cfg = load_config(args.config)
    wandb_starter = make_wandb_starter(getattr(base_cfg, "wandb_enabled", False))
    throttle_s = args.throttle_ms / 1000.0

    if args.seeds is None:
        run_finetune(base_cfg, wandb_run=wandb_starter, throttle_s=throttle_s)
        return

    results = []
    for seed in args.seeds:
        results.append(run_finetune(base_cfg, seed=seed, wandb_run=wandb_starter,
                                    throttle_s=throttle_s))
    summarize(results)


if __name__ == "__main__":
    main()
