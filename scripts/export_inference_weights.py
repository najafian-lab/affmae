#!/usr/bin/env python
"""Strip a training checkpoint down to what inference needs.

A training checkpoint carries the optimizer and AMP-scaler state so a run can
resume: 795 MB, of which 297 MB is the model. Inference never reads the rest,
and for a hosted demo the difference is the whole cold start.

    python scripts/export_inference_weights.py \
      --checkpoint weights/segmentation/fpw_aff_base_ft_512_slits_pgbmi.pth \
      --output dist/weights/fpw_aff_base_ft_512_slits_pgbmi.pth

Keeps ``model_state_dict`` plus the small scalar metadata (epoch, test_miou),
so the result still loads through the same path as the original.
"""

import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Everything else in a checkpoint is resume state.
KEEP = ("model_state_dict", "epoch", "test_miou")

logger = logging.getLogger(__name__)


def strip_checkpoint(payload: dict) -> dict:
    """Return a copy of ``payload`` with only the inference-relevant keys.

    Args:
        payload: a loaded checkpoint dict.
    Returns:
        A new dict holding the keys in :data:`KEEP` that were present.
    Raises:
        KeyError: if there is no model state to keep, which would make the
            output silently useless.
    """
    if "model_state_dict" not in payload:
        raise KeyError(
            f"no model_state_dict in this checkpoint; keys are "
            f"{sorted(payload)}. Nothing would be left to load.")
    return {key: payload[key] for key in KEEP if key in payload}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="Training checkpoint to read.")
    parser.add_argument("--output", required=True,
                        help="Where to write the stripped copy.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from affmae.eval.loader import legacy_checkpoint_compat

    with legacy_checkpoint_compat():
        payload = torch.load(args.checkpoint, map_location="cpu",
                             weights_only=False)
    if not isinstance(payload, dict):
        raise SystemExit(
            f"{args.checkpoint} holds a {type(payload).__name__}, not a dict; "
            f"it is already a bare state dict and needs no stripping.")

    kept = strip_checkpoint(payload)
    dropped = sorted(set(payload) - set(kept))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    torch.save(kept, args.output)

    before = os.path.getsize(args.checkpoint) / 1048576
    after = os.path.getsize(args.output) / 1048576
    logger.info("%s -> %s", args.checkpoint, args.output)
    logger.info("  %.1f MB -> %.1f MB (%.1fx smaller), dropped %s",
                before, after, before / after if after else 0, dropped)


if __name__ == "__main__":
    main()
