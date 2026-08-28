import argparse
import os
import sys

import torch

# Allow running as `python scripts/<name>.py` from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# NOTE: this used to import `create_dataloader` from affmae.data, which has never
# existed on any branch, and to load a hardcoded 'config/aff_small.yaml'.
from affmae.config import load_config
from affmae.data.pretrain_dataset import build_pretrain_dataloader
from affmae.data.stats import PRETRAIN_IMAGE_MEAN, PRETRAIN_IMAGE_STD

@torch.no_grad()
def estimate_mean_std(dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Per-channel mean and std over the whole loader.

    Accumulates sums and sums of squares rather than averaging per-batch
    variances. Averaging the per-batch variances discards the spread *between*
    batch means, which underestimates the true variance whenever batches differ
    in brightness -- exactly the case for EM tiles from different sessions.
    """
    n_channels = None
    count = 0
    total = None
    total_sq = None

    for images, *_ in dataloader:
        images = images.to(device, dtype=torch.float32)
        if n_channels is None:
            n_channels = images.shape[1]
            total = torch.zeros(n_channels, device=device, dtype=torch.float64)
            total_sq = torch.zeros(n_channels, device=device, dtype=torch.float64)

        pixels = images.view(images.size(0), n_channels, -1).double()
        total += pixels.sum(dim=(0, 2))
        total_sq += pixels.pow(2).sum(dim=(0, 2))
        count += pixels.size(0) * pixels.size(2)

    if not count:
        raise SystemExit(
            "the loader yielded no batches, so there is nothing to measure. "
            "With WebDataset this usually means fewer shards than "
            "num_workers -- see docs/custom_data.md.")

    mean = total / count
    std = (total_sq / count - mean.pow(2)).clamp_min(0).sqrt()
    return mean.float().cpu(), std.float().cpu()

def main():
    parser = argparse.ArgumentParser(
        description="Estimate per-channel dataset mean/std for normalization.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--device", default=None,
                        help="Defaults to cuda when available.")
    args = parser.parse_args()

    config = load_config(args.config)
    if not getattr(config, "path", None):
        raise SystemExit(
            "this script reads the pretraining WebDataset shards, so it needs a "
            "pretraining config with data.path set. A finetuning config has "
            "data.base_path instead and will not work here.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # normalize=False matters: the training pipeline normalizes with the constants
    # in affmae/data/stats.py, so measuring through it reports the residual
    # (mean ~0, std ~1) rather than the statistics you came here to find.
    dataloader, _ = build_pretrain_dataloader(config, normalize=False)
    mean, std = estimate_mean_std(dataloader, device=device)
    print(f"mean: {mean.tolist()}")
    print(f"std:  {std.tolist()}")
    print(f"\ncurrently configured, for comparison:")
    print(f"  PRETRAIN_IMAGE_MEAN = {PRETRAIN_IMAGE_MEAN}")
    print(f"  PRETRAIN_IMAGE_STD  = {PRETRAIN_IMAGE_STD}")


if __name__ == "__main__":
    main()
