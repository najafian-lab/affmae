#!/usr/bin/env python
"""Render segmentation and token-layout figures for every image in a test split.

    python scripts/render_test_set.py \
        --config configs/aff_base_finetune_512_fpw.yaml \
        --checkpoint weights/segmentation/fpw_aff_base_ft_512_slits_pgbmi.pth --tag ft512 --img-size 512

Writes ``<out>/<tag>/segmentation/NNNN_<name>.png`` and
``<out>/<tag>/tokens/NNNN_<name>.png``.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from affmae.config import load_config  # noqa: E402
from affmae.data.finetune_dataset import build_test_dataloader  # noqa: E402
from affmae.eval.loader import load_for_eval  # noqa: E402
from affmae.utils.dist import resolve_device, unwrap_model  # noqa: E402


def stage_positions(model, images):
    """Per-stage token positions for a preprocessed batch.

    Mirrors ``AFFMAE.token_layout`` but takes an already-normalized batch, so
    the dataloader's own preprocessing is not applied twice.

    Args:
        model: an AFF segmentation model.
        images: [B, C, H, W] normalized batch on the model's device.
    Returns:
        List of [B, N_s, 2] position tensors, or None if this encoder keeps a
        fixed grid.
    """
    bare = unwrap_model(model)
    encoder = getattr(bare, "encoder", None)
    if encoder is None or not hasattr(encoder, "forward_with_pos"):
        return None
    # ids_masked=None makes the patch embed behave as a plain conv stem.
    pos, feat, height, width = encoder.patch_embed(images, ids_masked=None)
    return encoder.forward_with_pos(feat, pos, height, width)


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", required=True, help="Subdirectory name.")
    parser.add_argument("--img-size", type=int, default=None,
                        help="Override the config's resolution to match the "
                             "checkpoint it was trained at.")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Override the config's class count. The dataset has "
                             "3 (background, PGBMI, slit), which is what every "
                             "config declares, so this is only for a checkpoint "
                             "trained against a different labelling.")
    parser.add_argument("--out", default="output/test_set_renders")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many images (for a smoke test).")
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from affmae.viz import PAPER, render_segmentation, render_token_layout

    cfg = load_config(args.config)
    if args.img_size:
        cfg.img_size = args.img_size
    if args.num_classes:
        cfg.num_classes = args.num_classes
    cfg.device = resolve_device(args.device or getattr(cfg, "device", None))

    seg_dir = os.path.join(args.out, args.tag, "segmentation")
    tok_dir = os.path.join(args.out, args.tag, "tokens")
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)

    model = load_for_eval(cfg, args.checkpoint)
    loader = build_test_dataloader(cfg)
    print(f"[{args.tag}] {cfg.img_size}px, {cfg.num_classes} classes, "
          f"{len(loader)} images -> {args.out}/{args.tag}/")

    written, skipped_tokens = 0, 0
    with torch.no_grad():
        for index, (images, targets, paths) in enumerate(loader):
            if args.limit and index >= args.limit:
                break
            images = images.to(cfg.device)
            targets = targets.to(cfg.device).long()
            outputs = model(images)
            logits = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs

            raw = paths[0] if isinstance(paths, (list, tuple)) else paths
            name = os.path.splitext(os.path.basename(str(
                raw[0] if isinstance(raw, (list, tuple)) else raw)))[0]
            # Filenames carry '++' and spaces; keep them filesystem-friendly.
            safe = "".join(c if c.isalnum() or c in "-_." else "_"
                           for c in name)[:80]

            render_segmentation(
                images.cpu(), logits.float().cpu(), cfg.num_classes,
                os.path.join(seg_dir, f"{index:04d}_{safe}.png"),
                targets=targets.cpu(), config=PAPER)

            positions = stage_positions(model, images)
            if positions is None:
                skipped_tokens += 1
            else:
                render_token_layout(
                    images.cpu(), [p.float().cpu() for p in positions],
                    cfg.patch_size,
                    os.path.join(tok_dir, f"{index:04d}_{safe}.png"),
                    config=PAPER)
            written += 1
            if written % 25 == 0:
                print(f"  {written}/{len(loader)}")

    print(f"[{args.tag}] wrote {written} segmentation figure(s)"
          + (f", token layout skipped for {skipped_tokens} (fixed-grid encoder)"
             if skipped_tokens else f" and {written} token figure(s)"))


if __name__ == "__main__":
    main()
