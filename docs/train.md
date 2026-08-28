# Training

AFFMAE uses YAML files from `configs/` for both self-supervised pretraining and
supervised segmentation fine-tuning. Config sections are flattened when loaded,
so `model.aff_nbhd_sizes` in YAML is available as `cfg.aff_nbhd_sizes`.

## Configuration reference

| Field | Meaning |
|---|---|
| `patch_size` | input pixels dim for each token |
| `aff_embed_dims` | feature dimension for each of the four encoder stages |
| `aff_depths` | transformer block count per stage |
| `aff_num_heads` | attention heads per stage |
| `aff_nbhd_sizes` | number of nearby tokens considered at each stage (needs to be a multiple of cluster size) |
| `aff_cluster_size` | tokens grouped into each cluster-attention block from space-filling curve |
| `aff_ds_rates` | fraction (0-1) of tokens retained by each adaptive merge. 0.5 means keep half of tokens, 1 means keep all |
| `aff_mlp_ratio` | hidden expansion ratio in encoder MLPs |
| `decoder_embed_dim` | decoder feature dimension |
| `decoder_depth` | number of decoder stages/blocks |
| `decoder_num_heads` | decoder attention heads |

The list-valued fields are in stage order and must agree with the four-stage AFF
encoder. The paper architecture uses neighbourhood size 64, cluster size 8,
encoder dimensions `[128, 256, 512, 768]`, depths `[3, 4, 16, 2]`, and a
384-dimensional decoder.

Training fields control `epochs`, batch size, gradient accumulation, optimizer
learning rate, minimum learning rate, warmup, weight decay, `layer_decay`, and
logging/checkpoint frequency. Set `cluster_attention_backend` and
`decoder_deform_backend` to `auto` for fused GPU execution with portable PyTorch
fallbacks. When `layer_decay` is omitted, the registered model default is used.

## Pretraining

Pretraining reads unlabelled images from WebDataset shards, masks input patches,
and learns reconstruction targets. Its config additionally controls
`mask_ratio`, masking strategy, deep supervision, `base_lr`, AdamW betas,
`warmup_steps`, and `num_accum`.

The paper configuration uses 400 epochs at 512×512, a 0.5 masking ratio, AdamW,
a base learning rate of `3.5e-4`, minimum learning rate `1e-6`, 10,000 warmup
steps, and weight decay `0.05`. It applies CLAHE followed by dataset
normalization without additional pretraining augmentation.

```bash
export DATA_ROOT=/datasets          # shards under $DATA_ROOT/pretrain/
python pretrain.py \
  --config configs/aff_base_pretrain_0.4ds_0.5mask_last_local.yaml

torchrun --nproc_per_node=4 pretrain.py \
  --config configs/aff_base_pretrain_0.4ds_0.5mask_last_local.yaml
```

Checkpoints and the resolved config are written beneath the configured output
directory. Pass `--resume` to continue from a checkpoint.

## Fine-tuning

Fine-tuning reads labelled image/mask pairs, replaces the reconstruction output
with a segmentation head, and initializes the encoder from
`model.pretrained_ckpt_path`. Its config adds `num_classes`, label indices,
class weights, segmentation loss, augmentation, layer-wise learning-rate decay,
and validation settings.

For FPW, the paper fine-tunes end to end for 400 epochs with a base learning rate
of `1e-4`, minimum learning rate `1e-6`, 25 warmup epochs, layer decay `0.6`, and
class-weighted BCE plus Dice. The three weights `[0.2, 2.0, 3.0]` correspond to
background, PGBMI, and filtration slits. Augmentation includes affine,
photometric, and elastic transformations.

```bash
export DATA_ROOT=/datasets
export CHECKPOINT_ROOT=$PWD/weights

python pretrain.py \
  --config configs/aff_base_pretrain_0.4ds_0.5mask_last_local.yaml

# Copy the chosen pretraining checkpoint to model.pretrained_ckpt_path, or edit
# that config field to point directly at the produced checkpoint.
python finetune.py --config configs/aff_base_finetune_512_fpw.yaml

python finetune.py --config configs/aff_base_finetune_512_fpw.yaml \
  --seeds 42 77 2026
```

Expected paths are `DATA_ROOT/fpwdata/{train,val,test}/{images,masks}` for FPW,
`DATA_ROOT/pretrain/*.tar` for pretraining shards, `CHECKPOINT_ROOT` for weights,
and the configured output directory for new runs. Relative values resolve against
the repository root, not your shell's working directory. These variables may be
placed in `.env`; existing shell values take precedence.

To train on your own dataset, start from `configs/template_pretrain_512.yaml` and
`configs/template_finetune_512.yaml`; [custom_data.md](custom_data.md) covers the
shard format, the mask layout, and what to change.

Evaluation is documented separately in [eval.md](eval.md).
