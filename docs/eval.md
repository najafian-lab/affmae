# Evaluation

`evaluate.py` evaluates a fine-tuned checkpoint against labelled test images.
Use `inference.py` instead for images without ground truth.

## Segmentation

```bash
python evaluate.py \
  --config configs/aff_base_finetune_512_fpw.yaml \
  --checkpoint AFFMAE_BASE_FT_512
```

The paper reports mean Intersection over Union (mIoU) and the filtration-slits
class IoU. Results are summarized as mean ± standard deviation across four
random seeds.

## FPW geometry

```bash
python evaluate.py \
  --config configs/aff_base_finetune_512_fpw.yaml \
  --checkpoint AFFMAE_BASE_FT_512 \
  --mode fpw --eval-grid-size 1024 \
  --out-json output/fpw.json
```

The geometry pass recovers connected PGBMI segments from the segmentation,
forms a one-pixel-wide ordered centerline for each segment, detects filtration
slits, and projects the slit locations onto that centerline. Arc length between
successive slit locations represents foot-process width. Predicted and
ground-truth segments are paired geometrically before their mean widths are
compared. FPW MAE: the mean per-image absolute pixel error
`|FPW_pred - FPW_GT|`, with distances scaled to a 1024×1024 reference grid.

```bash
python evaluate.py --config configs/aff_base_finetune_512_fpw.yaml \
  --mode fpw --seeds 42,77,2026,31415 \
  --checkpoint 'output/fpw_seed{seed}/last_model.pth'
```

Use `--vis-dir output/fpw_geometry` to render matched PGBMI centerlines and slit
locations.