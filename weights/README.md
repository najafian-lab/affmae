# Weights

Expected layout. Override the root with `CHECKPOINT_ROOT`; a relative value is
resolved against the repository root.

```
weights/pretrain/       backbones that finetuning initializes from
                        (config key: pretrained_ckpt_path, resume_path)
weights/segmentation/   finetuned segmentation checkpoints
```

The checkpoint files themselves are not in the repository. The names the shipped
configs expect:

| file | used by |
|---|---|
| `pretrain/ckpt_epoch_399_affmae_fpw.pth` | `aff_base_finetune_512_fpw`, `aff_base_finetune_1024_fpw` |
| `pretrain/ckpt_epoch_499_aff_base.pth` | `aff_base_finetune_768` |
| `pretrain/ckpt_epoch_99_aff_base_0.4ds.pth` | `aff_base_pretrain_0.4ds_0.5mask_last_local` (resume) |
| `pretrain/ckpt_epoch_399_vit_base.pth` | the ViT baselines |
| `pretrain/ckpt_epoch_274_vit_base.pth` | `vit_base_finetune_imgnet` |
| `pretrain/mae_pretrain_vit_base_imgnet.pth` | `vit_base_imgnet_ft` (official MAE ImageNet weights) |
