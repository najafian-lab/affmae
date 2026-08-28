#!/usr/bin/env python
"""Run a trained AFF-MAE model on images.

    # one image, using a released checkpoint (downloaded on first use)
    python inference.py --checkpoint AFFMAE_BASE_FT_512 --image docs/assets/sample1.png

    # a folder, writing masks and overlays
    python inference.py --checkpoint AFFMAE_BASE_FT_512 --input-dir images/ \
        --output-dir output/predictions

    # interactive browser demo
    python inference.py --checkpoint AFFMAE_BASE_FT_512 --gradio

    # your own run: a path, plus the YAML it was trained with
    python inference.py --checkpoint output/best_model.pth \
        --config configs/aff_base_finetune_512_fpw.yaml --image image.png
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from affmae.inference import AFFMAE  # noqa: E402
from affmae.utils.env import load_dotenv  # noqa: E402

IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def collect_images(input_dir: str) -> list:
    """List readable images in a directory, sorted.

    Args:
        input_dir: directory to scan (non-recursive).
    Returns:
        Sorted list of paths.
    Raises:
        SystemExit: if the directory is missing or holds no images.
    """
    root = Path(input_dir)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {input_dir}")
    found = sorted(p for p in root.iterdir()
                   if p.suffix.lower() in IMAGE_SUFFIXES)
    if not found:
        raise SystemExit(
            f"no images in {input_dir} (looked for {', '.join(IMAGE_SUFFIXES)})")
    return found


def save_mask(result, path: str) -> str:
    """Write the predicted label map as a PNG.

    Args:
        result: a SegmentationResult.
        path: destination path.
    Returns:
        ``path``.
    """
    from PIL import Image

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    Image.fromarray(result.labels.numpy().astype("uint8")).save(path)
    return path


def run_cli(args) -> None:
    """Predict on one image or a directory and write the outputs."""
    predictor = AFFMAE.from_checkpoint(
        args.checkpoint, config=args.config, device=args.device)
    logging.info("model ready on %s at %dpx", predictor.device, predictor.img_size)

    sources = [args.image] if args.image else collect_images(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from affmae.viz import VizConfig

    config = VizConfig(dpi=args.dpi)

    for source in sources:
        stem = Path(str(source)).stem
        result = predictor.segment(source)
        mask_path = save_mask(result, str(output_dir / f"{stem}_mask.png"))
        overlay_path = result.save_overlay(
            str(output_dir / f"{stem}_overlay.png"), config=config)
        counts = result.class_pixel_counts
        total = sum(counts.values())
        share = " ".join(f"c{k}:{100 * v / total:.1f}%"
                         for k, v in sorted(counts.items()))
        logging.info("%s -> %s, %s  [%s]", stem, Path(mask_path).name,
                     Path(overlay_path).name, share)

    logging.info("wrote %d prediction(s) to %s", len(sources), output_dir)


def list_weights() -> None:
    """Print the released checkpoint names, with what each one is."""
    from affmae.data.weights import EMWeights, WEIGHTS_FOLDER_URL

    print("Released checkpoints -- pass a name to --checkpoint:\n")
    for member in EMWeights:
        spec = member.spec
        classes = "" if spec.num_classes is None else f", {spec.num_classes} classes"
        cached = " [cached]" if os.path.isfile(member.download_path) else ""
        print(f"  {member.name:26s} {spec.backbone} {spec.task}, "
              f"{spec.img_size}px{classes}{cached}")
        print(f"  {'':26s} {spec.description}")
    print(f"\nAll of them are also in {WEIGHTS_FOLDER_URL}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--checkpoint",
                        help="A released checkpoint name (AFFMAE_BASE_FT_512; "
                             "see --list-weights), a path to a .pth, or a URL.")
    parser.add_argument("--list-weights", action="store_true",
                        help="List the released checkpoint names and exit.")
    parser.add_argument("--config", default=None,
                        help="Training YAML. Not needed for a released "
                             "checkpoint, which carries its own; otherwise "
                             "defaults to config.yaml beside the checkpoint.")
    parser.add_argument("--device", default=None,
                        help="cuda | cpu | mps. Defaults to the best available; "
                             "an unavailable choice is downgraded with a warning.")

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", help="Predict on a single image.")
    source.add_argument("--input-dir", help="Predict on every image in a folder.")

    parser.add_argument("--output-dir", default="output/predictions",
                        help="Where masks and overlays are written.")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Overlay resolution.")
    parser.add_argument("--gradio", action="store_true",
                        help="Launch the interactive browser demo instead of "
                             "writing files.")
    parser.add_argument("--share", action="store_true",
                        help="With --gradio, create a public link.")
    parser.add_argument("--port", type=int, default=7860,
                        help="With --gradio, the port to serve on.")
    parser.add_argument("--pretrain-checkpoint", default=None,
                        help="With --gradio, MAE checkpoint for the "
                             "reconstruction tab. Defaults to the released one "
                             "if already cached locally.")
    parser.add_argument("--pretrain-config", default=None,
                        help="Config for --pretrain-checkpoint.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()

    if args.list_weights:
        list_weights()
        return

    # Not argparse's required=True: --list-weights is a valid invocation with no
    # checkpoint at all, and required=True would reject it before we get here.
    if not args.checkpoint:
        parser.error("give --checkpoint, or --list-weights to see the names")

    if args.gradio:
        from affmae.demo import launch

        launch(checkpoint=args.checkpoint, config=args.config,
               device=args.device, share=args.share, port=args.port,
               pretrain_checkpoint=args.pretrain_checkpoint,
               pretrain_config=args.pretrain_config)
        return

    if not args.image and not args.input_dir:
        parser.error("give --image, --input-dir, or --gradio")
    run_cli(args)


if __name__ == "__main__":
    main()
