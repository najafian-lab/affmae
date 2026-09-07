"""Shared training loops.

One engine per stage -- :mod:`affmae.training.pretrain_engine` and
:mod:`affmae.training.finetune_engine` -- so ``pretrain.py`` and ``finetune.py``
are both thin CLIs, and the W&B setup they share lives in
:mod:`affmae.training.tracking` rather than being written twice.
"""
