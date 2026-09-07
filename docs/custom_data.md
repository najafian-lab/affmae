# Training on your own data

Pretraining reads **WebDataset `.tar` shards**; finetuning is supervised and reads **image/mask
directories**.

```
data/
  pretrain/                  # stage 1: WebDataset shards
    customdata-000.tar
    customdata-001.tar
    ...
  finetune/                  # stage 2: supervised splits
    train/  images/  masks/
    val/    images/  masks/
    test/   images/  masks/
```

`data/` does not have to be local. Configs support env vars and default reads
`${DATA_ROOT:-data}/...`, so you can either symlink it

```bash
ln -s /mnt/big-disk/affmae-data data
```

or point the variable elsewhere and leave the repository alone:

```bash
export DATA_ROOT=/mnt/big-disk/affmae-data
```

Relative paths resolve against the **repository root**, not your shell's working
directory, so training scripts work from anywhere. The same applies to
`CHECKPOINT_ROOT` (default `weights`) and `AFFMAE_OUTPUT_DIR` (default `output`).

---

## Stage 1 — pretraining data (WebDataset)

### Shard layout

Pretraining is masked autoencoding, so just images. Each
shard is an uncompressed `.tar` of images, and `webdataset` groups files by
basename, so one image per basename is all that is required:

```
customdata-000.tar
├── 000000.png
├── 000001.png
├── 000002.png
└── ...
```

Three things to consider when building shards:
1. **The extension is the key.** The loader calls `.decode("pil")` and then reads
   `sample["png"]`, so files must be `.png`. Rename before packing, or change the
   `map_dict(png=...)` key in `affmae/data/pretrain_dataset.py` to match your
   extension.
2. **Shards should be roughly equal in size**, ideally 100–1000 images each.
   Under multi-GPU training each rank gets a disjoint set of shards, so one giant
   shard leaves other ranks idle.
3. **Have at least as many shards as `num_workers` × ranks.** Shards are split
   across DataLoader workers *and* across ranks, so a worker that gets none makes
   WebDataset raise *"No samples found in dataset; perhaps you have fewer shards
   than workers."* Three shards with `num_workers: 8` fails, and it fails at the
   first batch rather than at startup.

### Building shards

```python
import webdataset as wds
from pathlib import Path

images = sorted(Path("/my/raw/images").glob("*.png"))

# maxcount caps images per shard; %03d must match the brace range in the config
with wds.ShardWriter("data/pretrain/customdata-%03d.tar", maxcount=1000) as sink:
    for i, path in enumerate(images):
        sink.write({"__key__": f"{i:08d}", "png": path.read_bytes()})
```

`ShardWriter` prints the shard count when it finishes. If it wrote
`customdata-000.tar` through `customdata-031.tar`, the config pattern is
`customdata-{000..031}.tar` — the brace range is **inclusive** and the digit
count must match the filenames exactly.

### Counting your samples

A WebDataset is an `IterableDataset` with no length, so the epoch boundary comes
from `data.total_samples` in the config rather than from the data. Getting it
wrong does not crash: too high repeats images, too low silently truncates the
epoch. Count once:

```bash
python -c "
import tarfile, glob
n = sum(sum(1 for m in tarfile.open(f) if m.name.endswith('.png'))
        for f in glob.glob('data/pretrain/*.tar'))
print(n)
"
```

### Running it

```bash
cp configs/template_pretrain_512.yaml configs/my_pretrain.yaml
# edit data.path, data.total_samples, data.in_channels
python pretrain.py --config configs/my_pretrain.yaml
```

Multi-GPU is mostly untested (use at your own rick) with `torchrun`; `data.batch_size` is the **global** batch and is divided
across ranks, so the effective batch does not change with GPU count:

```bash
torchrun --nproc_per_node=4 pretrain.py --config configs/my_pretrain.yaml
```

---

## Stage 2 — finetuning data (images and masks)

### Directory layout

```
data/finetune/
  train/
    images/   img_0001.tif   img_0002.tif   ...
    masks/    img_0001.tiff  img_0002.tiff  ...
  val/        images/  masks/
  test/       images/  masks/
```

Things to consider for finetuning on your data:
- **Images and masks pair by filename stem.** `img_0001.tif` needs
  `img_0001.tiff`. The loader takes the *intersection* of the two directories, so
  an unpaired file is skipped **without warning**.
- **Image extension is configurable** via `data.input_ext` (default `.tif`).
  **The mask extension is fixed at `.tiff`** in
  `affmae/data/finetune_dataset.py`.
- `val/` is the split evaluated during training. `test/` is read only by
  `evaluate.py`. All three are required by the templates; point `val` and `test`
  at the same files if you only have two splits.

### Mask format
Masks are multi-channel (ie multi-label) TIFFs, one binary channel per annotated structure — **not** a single-channel label map
with integer class IDs.

`data.indices` selects which channels to train on, and background is added
implicitly:

```yaml
data:
  indices: [0, 1]      # train on mask channels 0 and 1
  num_classes: 3       # len(indices) + 1 for background
```

So `num_classes` is typically `len(indices) + 1`, depending on loss selected, and `train.class_weighting` must
have exactly `num_classes` entries, background first. A wrong length fails at
loss construction with *"weight tensor should be defined either for all or no classes"*.

The selected channels are flattened into a single label map in `__getitem__`:
each channel is binarized at `THRESHOLD = 10`, and a pixel is assigned
`position_in_indices + 1`. Two consequences:

- **Class IDs follow the order of `indices`, not the channel numbers.**
  `indices: [2, 3]` makes channel 2 class 1 and channel 3 class 2.
- **Channels are applied in order, so on overlapping annotations the later channel wins.** Put the class you care most about last.

A single-channel integer label map is **not** supported — `__getitem__` unpacks
`C, H, W` and a 2-D mask raises a `ValueError`. One-hot it into channels when preparing the data.

Raise a class's weight if it is rare — the shipped `[0.2, 2.0, 3.0]` deliberately down-weights background.

### Running it

```bash
cp configs/template_finetune_512.yaml configs/my_finetune.yaml
# edit data.base_path, data.indices, data.num_classes,
#      train.class_weighting, model.pretrained_ckpt_path
python finetune.py --config configs/my_finetune.yaml
```

You do not have to pretrain first. To start from our released backbone, fetch it
once and point the config at the result:

```python
from affmae.data.weights import EMWeights
print(EMWeights.AFFMAE_BASE_PRETRAIN_512.fetch())
# -> weights/pretrain/ckpt_epoch_399_affmae_fpw.pth
```

```yaml
model:
  pretrained_ckpt_path: "${CHECKPOINT_ROOT:-weights}/pretrain/ckpt_epoch_399_affmae_fpw.pth"
```

The encoder architecture keys in your config must match that checkpoint, so keep
the `aff_*` block from the template unchanged unless you are pretraining yourself.
See the weights table in the README for every released checkpoint.

---

## Checklist before a long run

Cheap to check, expensive to discover at hour six.

| Check | Symptom if wrong |
|---|---|
| `data.total_samples` matches your shard count | epoch silently truncated or images repeated |
| Brace range digits match filenames | `FileNotFoundError` on the first batch |
| `num_classes == len(indices) + 1` | crash in the loss, or a dead output channel |
| `len(class_weighting) == num_classes` | *"weight tensor should be defined either for all or no classes"* |
| `in_channels` matches your images (1 grey, 3 RGB) | shape mismatch in the patch embedding |
| Encoder architecture keys identical to pretraining | checkpoint loads with missing/unexpected keys |
| Dataset size is what you expect | unpaired image/mask stems dropped silently |
| Shards >= `num_workers` x ranks | *"No samples found in dataset"* at the first batch |
| `stats.py` matches your data | washed-out or clipped figures; slower convergence |
| CLAHE removed if not wanted | real intensity differences flattened |
| Info-bar crop removed for non-EM images | bottom of bright images silently truncated |

Sanity-check the pipeline before committing to a full run:

```bash
python -c "
from affmae.config import load_config
from affmae.data.finetune_dataset import build_finetune_dataloader
cfg = load_config('configs/my_finetune.yaml')
loader = build_finetune_dataloader(cfg, is_train=True)   # False reads val/
img, mask = next(iter(loader))[:2]
print('image', img.shape, img.dtype)
print('mask ', mask.shape, 'classes present', mask.unique().tolist())
"
```

`mask.unique()` should be class indices in `[0, num_classes)`. If you see values
above that, `indices` and `num_classes` disagree.

---

## Adapting the preprocessing

Everything in this section is tuned for greyscale electron micrographs. None of it
is wrong for other data, but none of it is neutral either — these are the four
places where "trained on EM" is baked in.

### 1. Images are single-channel

We trained on greyscale. Two consequences:

- `data.in_channels: 1` in the config. Set it to 3 for RGB; the patch embedding
  and the normalization tuple both follow it.
- The **pretraining loader converts to greyscale regardless**:
  `apply_custom_processing` in `affmae/data/pretrain_dataset.py` calls
  `pil_image.convert("L")`. For RGB pretraining, remove that call — otherwise
  `in_channels: 3` silently receives a replicated grey channel.

### 2. Normalization statistics

`affmae/data/stats.py` holds four numbers measured on our data:

```python
IMAGE_MEAN = 0.6266           # labelled FPW split: finetuning and inference
IMAGE_STD  = 0.2259
PRETRAIN_IMAGE_MEAN = 0.5562  # unlabelled pretraining corpus
PRETRAIN_IMAGE_STD  = 0.2396
```

**Change these for your own data.** They are used by the loaders *and* by the
renderers, which denormalize with the same numbers — so a mismatch shows up as
washed-out or clipped figures long before you suspect the statistics.

Measure yours with:

```bash
python scripts/calc_norm.py --config configs/my_pretrain.yaml
```

It reports mean/std over your shards and prints the currently configured values
next to them. It measures the *raw* pipeline deliberately: an earlier version read
through the normalized pipeline and reported the residual (~0, ~1), which looks
like a passing result and is useless.

For multi-channel data, note these are scalars broadcast across channels
(`(mean,) * n_channels`). Per-channel statistics need `create_transforms` in
`affmae/data/pretrain_dataset.py` changed to take tuples.

### 3. CLAHE

Contrast-limited adaptive histogram equalization is applied in **four** places,
because EM micrographs have low local contrast. If your images are already
well-exposed, this will flatten real intensity differences and you should remove
it:

| Where | What | Flag? |
|---|---|---|
| `affmae/data/preprocess.py` | `apply_clahe`, `clipLimit=4.25`, `8x8` tiles | yes — `preprocess_image(..., use_clahe=False)` |
| `affmae/data/finetune_dataset.py:116, 214` | same parameters, inline | **no flag, edit the call** |
| `affmae/data/pretrain_dataset.py:80` | `clipLimit=4.0`, inside `apply_custom_processing` | **no flag, edit the call** |
| `affmae/data/pretrain_dataset.py:58` | random CLAHE, an augmentation | part of `random_transform` |

Only the inference path takes a flag. The dataset classes hardcode it, so removing
CLAHE from training means editing those call sites. If you do, **re-measure the
normalization statistics afterwards** — CLAHE changes the intensity distribution,
so the old mean/std no longer describe your inputs.

### 4. The microscope info bar

`apply_custom_processing` in `affmae/data/pretrain_dataset.py` crops the bottom of
the image when it finds a row that is more than 95% pure white below the 60%
height mark. That removes the scale-bar and metadata strip our microscope burns
into every frame.

This is the most EM-specific step in the pipeline and the most likely to damage
other data: any image with a genuinely bright bottom edge — a light background, a
white border — gets silently truncated, and because the crop is variable-height
the resize then changes the aspect ratio per image. **Remove it unless your images
have the same burned-in strip.**

### 5. Augmentation

`ElasticTransform` in `affmae/data/transforms.py` plus affine and photometric
transforms are tuned for tissue sections, which deform plausibly. For rigid
subjects (documents, industrial parts) elastic warping invents geometry that
cannot occur; see `scripts/visualize_aug.py` to look at what the current pipeline
does to your images before training on it:

```bash
python scripts/visualize_aug.py --base-path data/finetune --output output/aug.png
```

## Adapting the architecture

Defaults are AFF-Base at 512px with patch size 8. What is worth changing:

| Goal | Change |
|---|---|
| Larger images | `data.img_size` 768 or 1024, and lower `batch_size` |
| Smaller/faster model | shorten `aff_depths`, narrow `aff_embed_dims` |
| More tokens kept per stage | raise `aff_ds_rates` (0.4 keeps 40% per stage) |
| Different masking difficulty | `model.mask_ratio`, pretraining only |
| Out of memory | lower `batch_size`, raise `train.num_accum` to compensate |

Two big constraints:

- **`decoder_depth` must equal the number of encoder stages** (4 for AFF-Base),
  one decoder block per stage.
- **Encoder keys must match between pretraining and finetuning.** The finetune
  config re-declares `aff_embed_dims`, `aff_depths`, `aff_num_heads`,
  `aff_nbhd_sizes`, `aff_cluster_size`, `aff_ds_rates`, `aff_mlp_ratio` and
  `aff_merging_method`; any disagreement means the checkpoint does not load into
  the encoder.

`img_size` is the exception — it may differ between stages. Cost is driven by the
token count, which scales with `img_size / patch_size` rather than with pixel
count, so moving from 512 to 1024 at patch size 8 doubles the grid on each axis.

Leave `cluster_attention_backend` and `decoder_deform_backend` at `"auto"` unless
you have profiled the alternative. `"auto"` selects the fused Triton kernels on
CUDA and the PyTorch fallback elsewhere, so the same config runs on CPU, CUDA and
Apple silicon; see *Hardware and Kernel Coverage* in the README.
