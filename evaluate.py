#!/usr/bin/env python
"""Evaluate a finetuned model on a labelled test set.

Three modes, one entry point:

    # segmentation metrics (mIoU and Dice, per class)
    python evaluate.py --config configs/aff_base_finetune_512_fpw.yaml

    # foot-process-width geometry metrics from the paper
    python evaluate.py --config <cfg> --mode fpw --out-json output/fpw.json
    python evaluate.py --config <cfg> --mode fpw --seeds 42,77,2026

    # side-by-side figure comparing several trained models
    python evaluate.py --mode compare \
        --config configs/aff_base_finetune_512_fpw.yaml \
        --config configs/vit_base_finetune_fpn_512.yaml \
        --label AFF-MAE --label MAE
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from affmae.config import load_config  # noqa: E402
from affmae.eval.fpw import (  # noqa: E402
    FpwParams,
    evaluate_fpw,
    evaluate_fpw_across_seeds,
    format_seed_summary,
    json_safe,
    parse_grid_size,
)
from affmae.eval.segmentation import (  # noqa: E402
    compare_backends,
    compare_models,
    evaluate_segmentation,
)
from affmae.utils.dist import resolve_device  # noqa: E402
from affmae.utils.env import load_dotenv  # noqa: E402
from affmae.utils.misc import set_seed, setup_logging  # noqa: E402
from affmae.utils.paths import output_path  # noqa: E402


def _parse_boxes(raw):
    """Parse repeated ``x,y,w,h`` strings into tuples of int.

    Args:
        raw: sequence of comma-separated strings.
    Returns:
        List of (x, y, w, h) tuples.
    Raises:
        ValueError: if any entry does not have four parts.
    """
    boxes = []
    for item in raw:
        parts = [int(value) for value in item.split(",")]
        if len(parts) != 4:
            raise ValueError(f"zoom box {item!r} must be x,y,w,h")
        boxes.append(tuple(parts))
    return boxes


def _fpw_params(args) -> FpwParams:
    """Build FpwParams from the parsed arguments."""
    return FpwParams(
        pgbmi_class=args.pgbmi_class,
        slit_class=args.slit_class,
        slit_min_area=args.slit_min_area,
        slit_max_area=args.slit_max_area,
        slit_circularity=args.slit_circularity,
        pgbmi_dilate=args.pgbmi_dilate,
        min_pgbmi_area=args.min_pgbmi_area,
        min_segment_iou=args.min_segment_iou,
        pixel_size_nm=args.pixel_size_nm,
        eval_grid_size=parse_grid_size(args.eval_grid_size),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--config", action="append", required=True,
                        help="Training YAML. Repeat for --mode compare.")
    parser.add_argument("--mode",
                        choices=("metrics", "fpw", "compare", "backends"),
                        default="metrics",
                        help="metrics: mIoU/Dice. fpw: paper geometry metrics. "
                             "compare: side-by-side figure. backends: score one "
                             "checkpoint through each kernel path, to see what "
                             "the faster fused decoder costs in accuracy.")
    parser.add_argument("--checkpoint", action="append", default=None,
                        help="Explicit checkpoint; defaults to "
                             "<output_dir>/<name>/last_model.pth. Repeat for "
                             "--mode compare. May contain {seed}.")
    parser.add_argument("--device", default=None, help="cuda | cpu | mps.")
    parser.add_argument("--out-json", default=None,
                        help="Write the results here as JSON.")
    parser.add_argument("--seed", type=int, default=42)

    fpw = parser.add_argument_group("fpw mode")
    fpw.add_argument("--seeds", default=None,
                     help="Comma-separated seeds, e.g. 42,77,2026. Each reads "
                          "<output_dir>/<name>_seed<seed>/last_model.pth.")
    fpw.add_argument("--pgbmi-class", type=int, default=1)
    fpw.add_argument("--slit-class", type=int, default=2)
    fpw.add_argument("--slit-min-area", type=float, default=4.0)
    fpw.add_argument("--slit-max-area", type=float, default=400.0)
    fpw.add_argument("--slit-circularity", type=float, default=0.4)
    fpw.add_argument("--pgbmi-dilate", type=int, default=3)
    fpw.add_argument("--min-pgbmi-area", type=float, default=16.0)
    fpw.add_argument("--min-segment-iou", type=float, default=0.1)
    fpw.add_argument("--pixel-size-nm", type=float, default=1.0)
    fpw.add_argument("--eval-grid-size", default="1024",
                     help="Reference grid for pixel distances: N or W,H.")
    fpw.add_argument("--vis-dir", default=None,
                     help="Write per-image geometry overlays here.")
    fpw.add_argument("--max-vis", type=int, default=None)

    compare = parser.add_argument_group("compare mode")
    compare.add_argument("--label", action="append", default=None,
                         help="Display name per config; defaults to model_type.")
    compare.add_argument("--indices", type=int, nargs="+", default=[19, 91, 76],
                         help="Test-set sample indices to plot, one row each.")
    compare.add_argument("--zoom-box", action="append", default=None,
                         metavar="X,Y,W,H",
                         help="Zoom window per plotted row.")
    compare.add_argument("--out", default=None,
                         help="Figure path; defaults to "
                              "output/model_comparison.pdf.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()

    cfgs = [load_config(path) for path in args.config]
    device = resolve_device(args.device or getattr(cfgs[0], "device", None))
    for cfg in cfgs:
        cfg.device = device

    if args.mode != "compare" and len(cfgs) > 1:
        parser.error(f"--mode {args.mode} takes one --config; use --mode compare")

    checkpoints = args.checkpoint or [None] * len(cfgs)
    if len(checkpoints) != len(cfgs):
        parser.error(f"got {len(cfgs)} config(s) but {len(checkpoints)} "
                     f"--checkpoint value(s).")

    if args.mode == "compare":
        labels = args.label or [cfg.model_type for cfg in cfgs]
        if len(labels) != len(cfgs):
            parser.error(f"got {len(cfgs)} config(s) but {len(labels)} label(s).")
        boxes = _parse_boxes(args.zoom_box) if args.zoom_box else None
        if boxes and len(boxes) != len(args.indices):
            parser.error(f"got {len(boxes)} zoom box(es) for "
                         f"{len(args.indices)} index/indices.")

        set_seed(args.seed)
        out = args.out or output_path("model_comparison.pdf", create_parent=True)
        setup_logging(os.path.dirname(os.path.abspath(out)))
        path = compare_models(cfgs, labels, out, indices=args.indices,
                              zoom_boxes=boxes, checkpoints=checkpoints)
        logging.info("wrote %s", path)
        return

    cfg = cfgs[0]
    exp_dir = os.path.join(cfg.output_dir, cfg.name)
    os.makedirs(exp_dir, exist_ok=True)
    setup_logging(exp_dir)

    if args.mode == "backends":
        results = compare_backends(cfg, checkpoints[0])
        default_json = os.path.join(exp_dir, "backend_comparison.json")
    elif args.mode == "fpw":
        params = _fpw_params(args)
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()] \
            if args.seeds else None
        if seeds:
            results = evaluate_fpw_across_seeds(
                cfg, seeds, params, checkpoint=checkpoints[0],
                vis_dir=args.vis_dir, max_vis=args.max_vis)
            print(format_seed_summary(results["cross_seed_summary"]))
        else:
            results = evaluate_fpw(
                cfg, params, seed=args.seed, checkpoint=checkpoints[0],
                vis_dir=args.vis_dir, max_vis=args.max_vis)
            print(json.dumps(json_safe(results["summary"]), indent=2))
        results = json_safe(results)
        default_json = os.path.join(exp_dir, "fpw_metrics.json")
    else:
        results = evaluate_segmentation(cfg, checkpoints[0])
        default_json = None

    out_json = args.out_json or default_json
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".",
                    exist_ok=True)
        with open(out_json, "w") as handle:
            json.dump(results, handle, indent=2)
        logging.info("wrote %s", out_json)


if __name__ == "__main__":
    main()
