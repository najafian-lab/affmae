"""Self-supervised masked-autoencoder pretraining.

A thin CLI: the loop lives in :mod:`affmae.training.pretrain_engine`, matching
``finetune.py`` over :mod:`affmae.training.finetune_engine`.

    python pretrain.py --config configs/aff_base_pretrain_0.4ds_0.5mask_last_local.yaml
    torchrun --nproc_per_node=4 pretrain.py --config <same>

``torchrun`` needs no separate launcher: RANK/WORLD_SIZE/LOCAL_RANK are picked up
automatically, and absent them this runs single-process.
"""

import warnings

# Before the torch/timm imports the engine triggers, or the filters miss them.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")

from argparse import ArgumentParser  # noqa: E402

from affmae.config import load_config  # noqa: E402
from affmae.training.pretrain_engine import run_pretrain  # noqa: E402
from affmae.utils.env import load_dotenv  # noqa: E402


def main():
    parser = ArgumentParser(description="Pretrain an MAE with an AFF or ViT backbone")
    parser.add_argument("--config", required=True,
                        help="Path to the configuration file.")
    parser.add_argument("--resume", required=False,
                        help="Checkpoint to resume from; overrides resume_path.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.resume:
        config.resume_path = args.resume

    # Seeds os.environ from .env when present; real env vars still win.
    load_dotenv()
    run_pretrain(config, args.config)


if __name__ == "__main__":
    main()
