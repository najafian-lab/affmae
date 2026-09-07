#!/usr/bin/env python
"""Render per-decoder-stage PCA features from a pretraining checkpoint.

    python scripts/visualize_ckpt.py \
      --config configs/aff_base_pretrain_0.4ds_0.5mask_last_local.yaml \
      --checkpoint weights/pretrain/ckpt_epoch_399_affmae_fpw.pth \
      --samples docs/assets/sample{1,2,3,4}.png \
      --output output/plots/pca_decoder_stages.png

Without ``--samples`` it pulls a batch from the config's pretraining shards
instead, which is what the training loop does; naming files is usually what you
want for a figure, since the same images then appear in every run.

The model runs **unmasked**: ``forward_without_masking`` queries the decoder at
every patch position, so each stage's PCA covers the whole image with no holes.

This used to carry its own copy of the renderer, along with copies of
``compute_pca_rgb`` and ``denormalize``, and that copy ran ``_forward_internal``
instead -- so it masked the input and painted PCA only inside the masked region,
leaving the visible image showing through everywhere else. Two divergent
renderers for one figure, and the masked one is not what you want to look at:
the holes are where the model had no input, not where its features are
interesting. It now calls the same function the training loop uses.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from affmae.config import load_config  # noqa: E402
from affmae.models.registry import get_model_spec  # noqa: E402
from affmae.utils.misc import strip_module_prefix  # noqa: E402
from affmae.viz.model_figures import run_pca_visualization  # noqa: E402



def load_samples(paths, config):
    """Preprocess image files exactly as the pretraining loader would.

    Reuses ``apply_custom_processing`` and ``create_transforms`` rather than
    ``preprocess_image``: the latter normalizes with the *finetuning* statistics
    and skips the microscope info-bar crop, and both matter here. The bar is a
    bright strip the encoder never saw during pretraining, because that crop
    removed it, so leaving it in puts out-of-distribution input at the bottom of
    every PCA panel.

    Args:
        paths: image files, one row of the figure each.
        config: Config, read for ``img_size`` and ``in_channels``.
    Returns:
        [len(paths), C, img_size, img_size] normalized tensor.
    """
    from PIL import Image

    from affmae.data.pretrain_dataset import (
        apply_custom_processing,
        create_transforms,
    )

    transform = create_transforms(config.img_size, config.in_channels)
    tensors = []
    for path in paths:
        if not os.path.isfile(path):
            raise SystemExit(f"sample not found: {path}")
        # apply_custom_processing reads the third key positionally, the way a
        # decoded WebDataset sample arrives, so the order here is load-bearing.
        sample = {"__key__": path, "__url__": "", "png": Image.open(path)}
        tensors.append(transform(apply_custom_processing(sample)["png"]))
    return torch.stack(tensors)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PCA of each decoder stage's features, unmasked.")
    parser.add_argument("--config", required=True,
                        help="Pretraining config: model architecture, img_size, "
                             "and data.path when --samples is not given.")
    parser.add_argument("--samples", nargs="+", default=None,
                        help="Image files to use instead of the shards.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint (.pth).")
    parser.add_argument("--output", required=True, help="Output image (.png/.pdf).")
    parser.add_argument("--batch-skip", type=int, default=6,
                        help="Batches to discard before visualizing.")
    parser.add_argument("--num-images", type=int, default=6,
                        help="How many images to visualize.")
    parser.add_argument("--device", default=None,
                        help="Defaults to cuda when available.")
    return parser.parse_args()


def main():
    args = parse_args()

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.config):
        raise SystemExit(f"Config file not found: {args.config}")

    config = load_config(args.config)
    config.device = device_name
    if not args.samples and not getattr(config, "path", None):
        raise SystemExit(
            "without --samples this reads the pretraining shards, so it needs a "
            "pretraining config with data.path set. A finetuning config has "
            "data.base_path instead and will not work here.")

    device = torch.device(device_name)
    spec = get_model_spec(config.model_type)
    if spec.build_pretrain is None:
        raise SystemExit(f"model '{spec.name}' has no pretraining variant.")
    print(f"Initializing model type: {config.model_type}")
    model = spec.build_pretrain(config).to(device)

    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f"Checkpoint file not found: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = strip_module_prefix(
        checkpoint.get("model_state_dict", checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    if args.samples:
        batch_images = load_samples(args.samples, config)
    else:
        from affmae.data.pretrain_dataset import build_pretrain_dataloader

        dataloader, _ = build_pretrain_dataloader(config)
        data_iter = iter(dataloader)
        try:
            for _ in range(args.batch_skip):
                next(data_iter)
            batch_images, _ = next(data_iter)
        except StopIteration:
            raise SystemExit(
                f"the loader ran out of data before batch {args.batch_skip}; "
                f"lower --batch-skip.")

    count = min(batch_images.shape[0], args.num_images)
    print(f"Visualizing {count} images, unmasked...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    run_pca_visualization(model, batch_images[:count].to(device), args.output,
                          device)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
