"""Building a trained model for evaluation, and finding its checkpoint. """

import contextlib
import importlib
import importlib.abc
import importlib.machinery
import logging
import os
import sys
from typing import Any, Optional

import torch
import torch.nn as nn

from affmae.models.registry import get_model_spec
from affmae.utils.misc import strip_module_prefix

__all__ = ["load_for_eval", "load_state_dict_into", "resolve_checkpoint",
           "amp_dtype_for", "legacy_checkpoint_compat"]

# Rebuilt from the config at construction time, so a stale copy in a checkpoint
# is not just redundant but can disagree about width.
_SKIP_KEY_SUBSTRINGS = ("pre_table",)

#: The package was called ``src`` before the rename to ``affmae``.
_LEGACY_PACKAGE = "src"


class _LegacyPackageAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve ``src.*`` imports to ``affmae.*`` while unpickling.

    Checkpoints written before the package was renamed carry pickled references
    to ``src.utils.misc.AverageMeter`` and friends -- training bookkeeping saved
    beside the weights. Without this, ``torch.load`` on the released weights
    fails with ``ModuleNotFoundError: No module named 'src'``, which would make
    the published checkpoints unloadable by the published code.

    Installed only for the duration of a load, so it cannot shadow a real
    ``src`` package in someone else's project.
    """

    def find_spec(self, name, path=None, target=None):
        if name != _LEGACY_PACKAGE and not name.startswith(_LEGACY_PACKAGE + "."):
            return None
        return importlib.machinery.ModuleSpec(name, self)

    def create_module(self, spec):
        return importlib.import_module("affmae" + spec.name[len(_LEGACY_PACKAGE):])

    def exec_module(self, module):
        """Nothing to execute: the aliased module is already imported."""


@contextlib.contextmanager
def legacy_checkpoint_compat():
    """Make checkpoints pickled against the old ``src`` package loadable.

    Yields:
        None. The alias is removed on exit, along with any ``src.*`` entries it
        added to ``sys.modules``.
    """
    finder = _LegacyPackageAlias()
    sys.meta_path.insert(0, finder)
    before = set(sys.modules)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(finder)
        for name in set(sys.modules) - before:
            if name == _LEGACY_PACKAGE or name.startswith(_LEGACY_PACKAGE + "."):
                del sys.modules[name]


def _extract_state_dict(payload: Any) -> dict:
    """Pull the weights out of whatever a training run happened to save.

    Args:
        payload: the object returned by ``torch.load``.
    Returns:
        A state dict, module prefixes stripped and rebuilt buffers dropped.
    """
    if isinstance(payload, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in payload:
                payload = payload[key]
                break
    return {k: v for k, v in strip_module_prefix(payload).items()
            if not any(s in k for s in _SKIP_KEY_SUBSTRINGS)}


def load_state_dict_into(model: nn.Module, path: str,
                         map_location: Any = "cpu") -> nn.Module:
    """Load a checkpoint into ``model`` in place, reporting what did not match.

    Args:
        model: the module to populate.
        path: checkpoint file.
        map_location: passed to ``torch.load``.
    Returns:
        ``model``.
    Raises:
        FileNotFoundError: if ``path`` does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no checkpoint at {path}. Pass an explicit --checkpoint.")

    with legacy_checkpoint_compat():
        payload = torch.load(path, map_location=map_location, weights_only=False)
    state = _extract_state_dict(payload)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        logging.warning("%d key(s) missing from %s: %s",
                        len(incompatible.missing_keys), path,
                        incompatible.missing_keys[:8])
    if incompatible.unexpected_keys:
        logging.warning("%d unexpected key(s) in %s: %s",
                        len(incompatible.unexpected_keys), path,
                        incompatible.unexpected_keys[:8])
    logging.info("loaded %s", path)
    return model


def resolve_checkpoint(cfg: Any, checkpoint: Optional[str] = None,
                       seed: Optional[int] = None) -> str:
    """Work out which checkpoint file to evaluate.

    Args:
        cfg: a loaded Config; supplies ``output_dir`` and ``name``.
        checkpoint: explicit path, a released checkpoint name such as
            ``AFFMAE_BASE_FT_512``, or a URL. May contain ``{seed}``, which is
            formatted when ``seed`` is given.
        seed: seed whose run directory to use, for multi-seed evaluation.
    Returns:
        A filesystem path, not necessarily one that exists. A name or URL is
        downloaded first, so that path does exist.
    """
    if checkpoint:
        if seed is not None:
            checkpoint = checkpoint.format(seed=seed)
        # Same resolver the inference entry point uses, so `--checkpoint
        # AFFMAE_BASE_FT_512` means the same thing to both. A plain path is
        # returned untouched; only names and URLs are fetched.
        from affmae.data.weights import resolve_source

        return resolve_source(checkpoint)[0]
    name = cfg.name if seed is None else f"{cfg.name}_seed{seed}"
    return os.path.join(cfg.output_dir, name, "last_model.pth")


def load_for_eval(cfg: Any, checkpoint: Optional[str] = None,
                  seed: Optional[int] = None) -> nn.Module:
    """Build the segmentation model for ``cfg`` and load its trained weights.

    Args:
        cfg: a loaded Config.
        checkpoint: explicit checkpoint path, or None to use
            ``<output_dir>/<name>/last_model.pth``.
        seed: seed whose run directory to use, when ``checkpoint`` is None.
    Returns:
        The model in eval mode on ``cfg.device``.
    Raises:
        FileNotFoundError: if no checkpoint is found.
    """
    model = get_model_spec(cfg.model_type).build_segmentation(cfg)
    path = resolve_checkpoint(cfg, checkpoint, seed)
    load_state_dict_into(model, path, map_location="cpu")
    return model.to(cfg.device).eval()


def amp_dtype_for(cfg: Any) -> torch.dtype:
    """Autocast dtype named by ``cfg.amp_dtype``.

    Args:
        cfg: a loaded Config; ``amp_dtype`` defaults to float16.
    Returns:
        torch.float16 or torch.bfloat16.
    Raises:
        ValueError: on an unrecognised name.
    """
    name = getattr(cfg, "amp_dtype", "float16")
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"unsupported amp_dtype {name!r}; use float16 or bfloat16.")
