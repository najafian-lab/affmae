"""Stochastic depth (DropPath), vendored from timm.

``timm`` was a hard dependency of the AFF encoder for this one parameter-free
class, and importing it drags in PIL (and more). Vendoring the ~15 lines keeps
``affmae/layers`` importable with nothing but torch, which is what makes the layers
reusable in another project.

Semantics match ``timm.layers.DropPath`` exactly, including ``scale_by_keep``;
verified numerically against timm in ``tests/test_import_hygiene.py``. DropPath
holds no parameters, so existing checkpoints are unaffected.
"""

import torch
import torch.nn as nn

__all__ = ["drop_path", "DropPath"]


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False,
              scale_by_keep: bool = True) -> torch.Tensor:
    """Drop whole residual branches per sample (stochastic depth).

    Args:
        x: input tensor; the batch is dim 0.
        drop_prob: float, probability of dropping a sample's branch.
        training: bool, no-op when False.
        scale_by_keep: bool, rescale survivors by ``1 / keep_prob`` so the
            expected activation is unchanged.
    Returns:
        Tensor of the same shape as ``x``.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # Broadcast over every non-batch dim so the whole branch drops per sample.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Module wrapper around :func:`drop_path`.

    Args:
        drop_prob: float, probability of dropping a sample's branch.
        scale_by_keep: bool, rescale survivors by ``1 / keep_prob``.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:0.3f}"
