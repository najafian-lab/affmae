#!/usr/bin/env python
"""Montage of qualitative examples across the pretraining and finetuned models.

Regenerates docs/assets/affmae_512_examples.png:

    python scripts/render_examples.py \
      --config configs/aff_base_finetune_512_fpw.yaml \
      --checkpoint weights/segmentation/fpw_aff_base_ft_512_slits_pgbmi.pth \
      --pretrain-config configs/aff_base_pretrain_0.4ds_0.5mask_last_local.yaml \
      --pretrain-checkpoint weights/pretrain/ckpt_epoch_399_affmae_fpw.pth \
      --samples docs/assets/sample{1,2,3,4}.png \
      --output docs/assets/affmae_512_examples.png

One row per sample, one column per panel in ``--columns``. Seeded per sample, so
the command above reproduces the committed asset byte-for-byte.

Note:
    Reconstruction panels use Perlin masking, matching pretraining
    (``forward`` -> ``_forward_internal`` -> ``mask_and_embed``). The token
    positions come from the reconstruction result, not a second
    ``token_layout`` call: that would draw a fresh mask, and only ~50% of the
    tokens would land on a patch the masked image shows as visible.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from affmae.config import load_config  # noqa: E402
from affmae.inference import AFFMAE  # noqa: E402
from affmae.utils.paths import default_plot_path  # noqa: E402
from affmae.viz import (  # noqa: E402
    PAPER,
    class_overlay,
    draw_token_positions,
    to_display_image,
)

#: Stage banners. The two training stages the figure contrasts.
STAGE_1 = "Stage 1: MAE pretraining"
STAGE_2 = "Stage 2: Supervised finetune"

#: Panel name -> (column title, needs the pretraining model, stage banner).
#: The banner groups columns under a training stage; "" means no banner, and
#: consecutive columns sharing one get a single banner spanning them with dashed
#: rules at the boundaries.
PANELS = {
    "input":          ("Input", False, ""),
    "masked":         ("Perlin-masked input", True, STAGE_1),
    "tokens_sparse":  ("Sparse tokens, encoder stage {sparse_stage}", True, STAGE_1),
    "reconstruction": ("Reconstruction", True, STAGE_1),
    "tokens":         ("Final stage\ntoken locations", False, STAGE_2),
    "truth":          ("Ground truth", False, STAGE_2),
    "prediction":     ("Prediction", False, STAGE_2),
}
#: The sparse-token panel is available but not shown by default: the masked input
#: and the reconstruction already carry the pretraining story, and a token panel
#: on both sides of the figure invites a comparison between a sparse stage-2
#: layout and a dense final-stage one that is not like-for-like.
DEFAULT_COLUMNS = ("input", "masked", "reconstruction", "tokens", "prediction")


def column_groups(columns):
    """Contiguous runs of columns sharing a banner.

    Returns:
        list of ``(banner, first_index, last_index)``, in column order. A run is
        broken by any change of banner, so reordering ``--columns`` cannot
        produce a banner that spans a column belonging to another stage.
    """
    groups = []
    for index, name in enumerate(columns):
        banner = PANELS[name][2]
        if groups and groups[-1][0] == banner:
            groups[-1][2] = index
        else:
            groups.append([banner, index, index])
    return [tuple(g) for g in groups]


def draw_stage_banners(fig, axes, columns, groups, viz, rows):
    """Write the stage banners and the dashed rules between stages.

    Called after ``tight_layout``, because it reads final axes positions. Works
    in figure coordinates, so ``bbox_inches="tight"`` crops around the banners
    instead of clipping them.
    """
    from matplotlib.lines import Line2D

    top = max(axes[0][c].get_position().y1 for c in range(len(columns)))
    bottom = min(axes[rows - 1][c].get_position().y0 for c in range(len(columns)))

    # Place the banner above the *rendered* column titles rather than above the
    # axes plus a guess. Estimating from font points broke as soon as a title
    # wrapped -- "Final stage\ntoken locations" is two lines, and a pad sized for
    # one put the banner on top of it. Ask matplotlib where the text actually
    # ended up instead.
    fig.canvas.draw()
    inverse = fig.transFigure.inverted()
    title_top = top
    for column in range(len(columns)):
        title = axes[0][column].title
        if not title.get_text():
            continue
        box = inverse.transform(title.get_window_extent(fig.canvas.get_renderer()))
        title_top = max(title_top, box[1][1])
    # One line of the banner's own font, as breathing room above the titles.
    pad = (title_top - top) + (viz.font_size + 6) * 1.1 / (72.0 * fig.get_size_inches()[1])

    for banner, first, last in groups:
        if not banner:
            continue
        left = axes[0][first].get_position().x0
        right = axes[0][last].get_position().x1
        fig.text((left + right) / 2.0, top + pad, banner,
                 ha="center", va="bottom",
                 fontsize=viz.font_size + 6, fontweight="bold")

    for previous, following in zip(groups, groups[1:]):
        gap_left = axes[0][previous[2]].get_position().x1
        gap_right = axes[0][following[1]].get_position().x0
        x = (gap_left + gap_right) / 2.0
        fig.add_artist(Line2D([x, x], [bottom, top + pad * 1.9],
                              transform=fig.transFigure,
                              color="0.35", linestyle="--", linewidth=2.0))


def load_truth(mask_dir, stem, num_classes):
    """Load a multi-channel mask and flatten it the way training does.

    Mirrors ``EMDatasetMultiClass.__getitem__``: each channel is binarized at 10
    and a pixel takes ``channel_position + 1``, so later channels win on overlap.
    """
    import glob

    import tifffile

    matches = glob.glob(os.path.join(mask_dir, stem + ".*"))
    if not matches:
        raise FileNotFoundError(
            f"no mask for {stem!r} in {mask_dir}. Ground truth needs a mask "
            f"whose filename stem matches the image.")
    mask = tifffile.imread(matches[0])
    if mask.ndim == 4:
        mask = mask[:, :, :, 0]
    if mask.ndim == 2:
        return torch.from_numpy(mask.astype(np.int64))
    labels = np.zeros(mask.shape[1:], dtype=np.int64)
    for position in range(min(mask.shape[0], num_classes - 1)):
        labels[mask[position] > 10] = position + 1
    return torch.from_numpy(labels)


def resize_labels(labels, shape):
    """Nearest-neighbour resize of a label map to ``shape``.

    Masks are stored at the microscope's native resolution while predictions come
    back at ``img_size``; overlaying the two unresized makes matplotlib size the
    axes to whichever imshow came last, shrinking the background into a corner.
    Nearest, not bilinear: interpolating class indices invents classes that were
    never annotated.
    """
    if tuple(labels.shape) == tuple(shape):
        return labels
    resized = torch.nn.functional.interpolate(
        labels[None, None].float(), size=tuple(shape), mode="nearest")
    return resized[0, 0].long()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="Finetuning config.")
    parser.add_argument("--checkpoint", required=True,
                        help="Finetuned segmentation checkpoint.")
    parser.add_argument("--pretrain-config", default=None,
                        help="Pretraining config; needed by the MAE panels.")
    parser.add_argument("--pretrain-checkpoint", default=None,
                        help="Pretraining checkpoint; needed by the MAE panels.")
    parser.add_argument("--samples", nargs="+", required=True,
                        help="Image paths, one row each.")
    parser.add_argument("--columns", default=",".join(DEFAULT_COLUMNS),
                        help=f"comma-separated, from {sorted(PANELS)}")
    parser.add_argument("--sparse-stage", type=int, default=2,
                        help="1-based encoder stage for the optional "
                             "tokens_sparse panel, which is off by default.")
    parser.add_argument("--mask-ratio", type=float, default=0.5,
                        help="Fraction of patches hidden from the MAE.")
    parser.add_argument("--masks", default=None,
                        help="Mask directory, required by the truth column.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seeds the Perlin mask, so the figure is "
                             "reproducible. Pass -1 to leave the RNG alone.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    return parser


def main():
    args = build_parser().parse_args()

    columns = [name.strip() for name in args.columns.split(",") if name.strip()]
    unknown = [name for name in columns if name not in PANELS]
    if unknown:
        raise SystemExit(f"unknown column(s) {unknown}; choose from {sorted(PANELS)}")
    if "truth" in columns and not args.masks:
        raise SystemExit("--columns includes 'truth', so --masks is required.")

    needs_mae = any(PANELS[name][1] for name in columns)
    if needs_mae and not (args.pretrain_config and args.pretrain_checkpoint):
        wanted = [n for n in columns if PANELS[n][1]]
        raise SystemExit(
            f"column(s) {wanted} come from the MAE, so --pretrain-config and "
            f"--pretrain-checkpoint are both required.")
    if not 0.0 < args.mask_ratio < 1.0:
        raise SystemExit(
            f"--mask-ratio must be in (0, 1), got {args.mask_ratio}: nothing "
            f"masked leaves the decoder no queries, everything masked leaves "
            f"the encoder no input.")

    config = load_config(args.config)
    segmenter = AFFMAE.from_checkpoint(args.checkpoint, config=config,
                                       device=args.device)
    print(f"segmentation: {args.checkpoint} on {segmenter.device} "
          f"({sorted(segmenter.capabilities)})")

    mae = None
    if needs_mae:
        mae = AFFMAE.from_checkpoint(args.pretrain_checkpoint,
                                     config=args.pretrain_config,
                                     device=args.device)
        print(f"pretraining:  {args.pretrain_checkpoint} on {mae.device} "
              f"({sorted(mae.capabilities)})")
        if not mae.can_reconstruct:
            raise SystemExit(
                f"{args.pretrain_checkpoint} has no MAE head, so it cannot "
                f"produce the masked/reconstruction panels.")

    viz = PAPER
    stage_index = args.sparse_stage - 1

    rows = []
    for index, path in enumerate(args.samples):
        # Re-seed per sample so a row's mask does not depend on how many rows
        # precede it -- otherwise dropping one sample changes all the others.
        if args.seed >= 0:
            torch.manual_seed(args.seed + index)
        row = {"stem": os.path.splitext(os.path.basename(path))[0]}
        row["result"] = segmenter.segment(path)
        _, dense = segmenter.token_layout(path)
        row["dense_tokens"] = dense[-1]
        if mae is not None:
            # One pass. Taking the tokens from a second call to
            # token_layout(mask_ratio=...) drew a *fresh* Perlin mask, so the
            # figure showed one mask's tokens over another mask's image -- only
            # ~50% of them landed on a visible patch, which is chance. The
            # reconstruction result carries the tokens from its own pass, so
            # they and the masked image agree by construction (measured: 100%).
            row["recon"] = mae.reconstruct(path, mask_ratio=args.mask_ratio)
            if "tokens_sparse" in columns:
                sparse = row["recon"].locations
                if not sparse:
                    raise SystemExit(
                        f"{args.pretrain_checkpoint} exposes no per-stage token "
                        f"positions, so the sparse-token panel cannot be drawn.")
                if not -len(sparse) <= stage_index < len(sparse):
                    raise SystemExit(
                        f"--sparse-stage {args.sparse_stage} is out of range; "
                        f"the encoder has {len(sparse)} stages.")
                row["sparse_tokens"] = sparse[stage_index]
        rows.append(row)
        counts = row["result"].class_pixel_counts
        print(f"  {row['stem'][:44]}: classes {sorted(counts)}, "
              f"dense {row['dense_tokens'].shape[0]} tokens"
              + (f", sparse {row['sparse_tokens'].shape[0]}"
                 if "sparse_tokens" in row else ""))

    n_rows, n_cols = len(rows), len(columns)
    size = viz.figsize_per_cell
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(size * n_cols, size * n_rows),
                             squeeze=False)

    for r, row in enumerate(rows):
        result = row["result"]
        dense_bg = to_display_image(result.image, viz)
        for c, column in enumerate(columns):
            ax = axes[r][c]
            if column == "input":
                ax.imshow(dense_bg, cmap=viz.cmap, vmin=0, vmax=1)
            elif column == "masked":
                ax.imshow(to_display_image(row["recon"].masked, viz),
                          cmap=viz.cmap, vmin=0, vmax=1)
            elif column == "reconstruction":
                # reconstructions[-1] is the finest head, and AFFMAE.reconstruct
                # already stitches the visible original patches back in, so only
                # the masked region is model output.
                ax.imshow(to_display_image(row["recon"].reconstructions[-1], viz),
                          cmap=viz.cmap, vmin=0, vmax=1)
            elif column == "tokens_sparse":
                ax.imshow(draw_token_positions(
                    row["recon"].masked, row["sparse_tokens"],
                    config.patch_size, viz))
            elif column == "tokens":
                ax.imshow(draw_token_positions(
                    result.image, row["dense_tokens"], config.patch_size, viz))
            else:
                ax.imshow(dense_bg, cmap=viz.cmap, vmin=0, vmax=1)
                if column == "prediction":
                    ax.imshow(class_overlay(result.labels, result.num_classes, viz))
                elif column == "truth":
                    truth = load_truth(args.masks, row["stem"], result.num_classes)
                    ax.imshow(class_overlay(
                        resize_labels(truth, result.labels.shape),
                        result.num_classes, viz))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(
                    PANELS[column][0].format(sparse_stage=args.sparse_stage),
                    fontsize=viz.font_size + 2)
            if c == 0:
                label = row["stem"]
                if len(label) > 24:      # microscope filenames run past 60 chars
                    label = label[:11] + "..." + label[-10:]
                ax.set_ylabel(label, fontsize=viz.font_size)

    fig.tight_layout()
    draw_stage_banners(fig, axes, columns, column_groups(columns), viz, n_rows)

    output = args.output or default_plot_path("affmae_examples.png")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=viz.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}  ({n_rows} rows x {n_cols} cols: {', '.join(columns)})")


if __name__ == "__main__":
    main()
