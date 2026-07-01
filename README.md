ECCV 2026 repository (under progress)

Repository for the preprint version of the paper is available, but some improvments to the codebase are still being made and documentation is in-progress: be weary that kernels, in current repo, are not full proof yet, and we will build more testing to verify all cases. It works for the current configs, but if you have any issues with other resolutions/custom model sizes please leave a comment. Expect updates to the more stable kernel code shortly.

This repository contains the code to train both ViT and AFF using masked auto-encoding

### Relevant folders/files
`configs/`: all pretraining and finetuning configs for ViT/AFF on our EM dataset

`src/layers/attention.py`: contains the point-based ClusterAttention, Deformable{Cross,Self}Attention

`src/layers/kernels`: Triton kernels used in the attention/downsampling modules

`src/layers/decoder.py`: Stage-based dense pixel decoder for AFF to produce grid-based outputs. Although, one can optionally sample off-grid

`src/models`: contains structure for MAE pretraining (files ending in )mae.py) and segmentation downstream (files ending in _segmentation.py)

`pretrain.py/finetune.py`: self-explanatory main files to run you must provide a `--config configs/<config filename>.yaml` to train the model. Ensure that you update the paths/batch size for your data in the example config file.
