import random
from typing import Union
import os
import logging

import numpy as np
import torch
import torch.nn as nn


from affmae.utils.dist import unwrap_model


class AverageMeter:
    """Compute and store the average and current value.

    Args:
        reduce_fn: callable or None. When set, ``avg`` is passed through it
            before being returned, which is the seam for cross-rank averaging
            under distributed training (pass
            :func:`affmae.utils.dist.reduce_metric`). Accumulation stays
            process-local; only the reported mean is reduced. None keeps the
            single-process behaviour exactly.
    """

    def __init__(self, reduce_fn=None):
        self.reduce_fn = reduce_fn
        self.reset()

    def reset(self):
        self.val = 0
        self._avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self._avg = self.sum / self.count

    @property
    def avg(self):
        """Mean value, reduced across ranks when ``reduce_fn`` is set."""
        if self.reduce_fn is None:
            return self._avg
        return self.reduce_fn(self._avg)

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, step: int, loss: float, path: str):
    """Save a model checkpoint.

    The model is unwrapped first so the keys are always bare. Saving a wrapped
    model would emit ``module.``-prefixed keys that no single-GPU run can load.
    """
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': unwrap_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)


def strip_module_prefix(state_dict: dict) -> dict:
    """Drop a leading ``module.`` from every key, if uniformly present.

    Lets checkpoints written by an older wrapped run load into a bare model.

    Args:
        state_dict: dict, checkpoint weights.
    Returns:
        A dict with the prefix removed, or the input unchanged if not every key
        carried it.
    """
    if state_dict and all(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model: nn.Module, optimizer: Union[None, torch.optim.Optimizer], path: str) -> tuple[int, int]:
    """Load a model checkpoint and return (epoch, step)."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = strip_module_prefix(checkpoint['model_state_dict'])
    unwrap_model(model).load_state_dict(state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint.get('step', 0)

def adapt_in_pretrained_vit(old_state_dict: dict):
    new_state_dict = {}
    for key, weight in old_state_dict.items():

        if "patch_embed.proj.weight" in key:
            print(f"Converting {key} from 3-channel to 1-channel.")
            new_state_dict[key] = weight.sum(dim=1, keepdim=True)
            continue

        if "decoder_norm" in key:
            new_key = key.replace("decoder_norm", "decoder_pred_norm")
            print(f"Renaming: {key} -> {new_key}")
            new_state_dict[new_key] = weight
            continue

        if "decoder_pred" in key:
            print(f"Dropping {key} because output dimensions changed.")
            continue

        if "blocks" in key and "decoder" not in key:
            new_key = key.replace("blocks", "encoder_blocks")
            new_state_dict[new_key] = weight
            continue

        if "norm" in key and "decoder" not in key:
            new_key = key.replace("norm", "encoder_norm")
            new_state_dict[new_key] = weight
            continue


        new_state_dict[key] = weight

    return new_state_dict

def cosine_lr_schedule(optimizer: torch.optim.Optimizer, step: int,
                       max_steps: int, lr: float, min_lr: float = 0.,
                       warmup_steps: int = 0):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        lr_scale = step / warmup_steps
    else:
        # handle case where step might exceed max_steps
        progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
        lr_scale = 0.5 * (1 + np.cos(np.pi * progress))

    current_lr = min_lr + (lr - min_lr) * lr_scale

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return current_lr

def setup_logging(exp_dir: str):
    """Log to both the console and ``training.log`` in the experiment directory.

    Note:
        ``force=True`` is load-bearing. ``basicConfig`` silently does nothing
        when the root logger already has a handler, and by the time any training
        script calls this it does: ``load_config`` emits a ``logging.info`` for a
        legacy ``model_type`` alias, and the module-level ``logging.info``
        function installs a StreamHandler when none exists. So every run whose
        config used an alias -- which is most of them, ``aff`` among them -- got
        no FileHandler and left a zero-byte ``training.log``.
    """
    log_file = os.path.join(exp_dir, 'training.log')
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True,
    )
