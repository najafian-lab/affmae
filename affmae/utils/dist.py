"""Distributed training helpers. Thanks claude :)

Every function here is a no-op when there is no process group, so the same
driver runs unchanged under ``python pretrain.py`` and
``torchrun --nproc_per_node=N pretrain.py``.

Native ``torch.distributed`` rather than ``accelerate``: the release has no
other Accelerate usage (that lives on ``rebuttals``), and explicit DDP keeps
gradient behaviour directly testable — see ``tests/test_distributed.py``.

Why the helpers exist:

* ``unwrap_model`` — ``isinstance(model, AFFSegmentation)`` and
  ``model.encoder.layers`` both fail silently or loudly the moment a model is
  wrapped. Route every reach-through and type check through here.
* ``is_main_process`` — guards side effects (wandb, checkpoint writes,
  visualizations) that must happen exactly once.
* ``reduce_metric`` — a metric averaged per-rank is not the epoch metric.
* ``convert_sync_batchnorm`` — the ViT UperNet head uses ``BatchNorm2d``, whose
  running statistics diverge across ranks unless converted.
"""

import logging
import os

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "synchronize",
    "resolve_device",
    "autocast_context",
    "init_distributed",
    "cleanup_distributed",
    "wrap_for_distributed",
    "has_unused_parameters",
    "unwrap_model",
    "is_distributed",
    "is_main_process",
    "get_rank",
    "get_world_size",
    "reduce_metric",
    "convert_sync_batchnorm",
]


def unwrap_model(model):
    """Return the underlying module, peeling any wrapper layers.

    Handles ``DistributedDataParallel``, ``DataParallel``, and Accelerate's
    wrappers uniformly, and recurses in case of nesting (e.g. DDP over a
    compiled module).

    Args:
        model: nn.Module, possibly wrapped.
    Returns:
        The innermost nn.Module.
    """
    seen = 0
    while hasattr(model, "module") and isinstance(getattr(model, "module"), nn.Module):
        model = model.module
        seen += 1
        if seen > 8:  # pathological nesting; bail rather than spin
            break
    # torch.compile stores the original under _orig_mod
    if hasattr(model, "_orig_mod") and isinstance(model._orig_mod, nn.Module):
        model = model._orig_mod
    return model


def is_distributed() -> bool:
    """True if a torch.distributed process group is initialized."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    """Rank of this process, or 0 when not distributed."""
    return torch.distributed.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Number of processes, or 1 when not distributed."""
    return torch.distributed.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    """True on rank 0, and always true when not distributed.

    Guard every side effect that must happen exactly once — wandb logging,
    checkpoint writes, figure output — with this.
    """
    return get_rank() == 0


def reduce_metric(value, device=None):
    """Average a scalar metric across ranks.

    Args:
        value: float or torch.Tensor, this rank's value.
        device: torch.device or None, where to stage the all-reduce. Defaults to
            the current CUDA device when available.
    Returns:
        float, the cross-rank mean. Returns ``value`` unchanged when not
        distributed, so callers need no branch.
    """
    if not is_distributed():
        return float(value)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.as_tensor(value, dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor.item() / get_world_size())


def convert_sync_batchnorm(model):
    """Convert BatchNorm layers to SyncBatchNorm when running distributed.

    ``ViTSegmentationUperNet`` uses ``nn.BatchNorm2d`` in its FPN and PPM
    blocks. Under DDP each rank would otherwise keep its own running statistics,
    computed over a fraction of the effective batch.

    Call this on the **unwrapped** model, before wrapping it.

    Args:
        model: nn.Module to convert in place.
    Returns:
        The converted module (unchanged when not distributed).
    """
    if not is_distributed():
        return model
    n_bn = sum(1 for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
    if n_bn:
        logger.info("Converting %d BatchNorm layer(s) to SyncBatchNorm.", n_bn)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def init_distributed(backend=None):
    """Initialize a process group from torchrun's environment variables.

    Reads ``RANK``, ``WORLD_SIZE`` and ``LOCAL_RANK``, which ``torchrun`` sets.
    Does nothing when they are absent, so the same script runs unchanged under
    ``python pretrain.py``.

    Args:
        backend: str or None, torch.distributed backend. Defaults to "nccl" on
            CUDA and "gloo" otherwise.
    Returns:
        (is_distributed, local_rank). ``local_rank`` is 0 when not distributed.
    """
    if is_distributed():
        return True, int(os.environ.get("LOCAL_RANK", 0))
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    torch.distributed.init_process_group(backend=backend)
    logger.info(
        "distributed: rank %d/%d local_rank %d backend %s",
        get_rank(), get_world_size(), local_rank, backend,
    )
    return True, local_rank


def cleanup_distributed():
    """Tear down the process group if one is active."""
    if is_distributed():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def wrap_for_distributed(model, device=None, find_unused_parameters=False,
                         sync_batchnorm=True):
    """Convert BatchNorm and wrap in DistributedDataParallel.

    No-op when not distributed, so callers need no branch. Always call this on
    the **unwrapped** model, after moving it to the device.

    Args:
        model: nn.Module to wrap.
        device: torch.device or None, the local device; inferred when None.
        find_unused_parameters: bool, needed when some parameters get no
            gradient in a given forward (e.g. auxiliary heads that a config
            disables). Costs a graph traversal per step, so leave it False
            unless required.
        sync_batchnorm: bool, convert BatchNorm to SyncBatchNorm first. The ViT
            UperNet head has BatchNorm2d, whose running stats would otherwise
            diverge per rank.
    Returns:
        The wrapped module, or ``model`` unchanged when not distributed.
    """
    if not is_distributed():
        return model

    if sync_batchnorm:
        model = convert_sync_batchnorm(model)

    device_ids = None
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_ids = [local_rank if device is None else device.index or local_rank]

    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
    )


def has_unused_parameters(model, loss):
    """Report parameters that received no gradient from ``loss``.

    DDP raises unless ``find_unused_parameters=True`` when this is non-empty.
    Use it to decide that flag for a given config rather than guessing.

    Args:
        model: nn.Module already backward-ed through.
        loss: unused, present so call sites read as a post-backward assertion.
    Returns:
        list of str, names of trainable parameters with ``grad is None``.
    """
    return [n for n, p in unwrap_model(model).named_parameters()
            if p.requires_grad and p.grad is None]


def resolve_device(preferred=None):
    """Pick a usable torch device, honouring a preference when it is available.

    Device selection was duplicated across four drivers with three different
    behaviours: ``pretrain.py`` trusted the YAML with no availability check (so
    ``device: "cuda"`` hard-failed on a CPU-only box), while ``finetune.py`` and
    ``evaluate.py`` silently overrode it.

    Args:
        preferred: str, torch.device, or None. A named device is used only if
            it is actually available; otherwise it is downgraded with a warning.
            None picks the best available.
    Returns:
        torch.device: cuda > mps > cpu, or the honoured preference.
    """
    def _available(kind):
        if kind == "cuda":
            return torch.cuda.is_available()
        if kind == "mps":
            return (hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available())
        return kind == "cpu"

    best = "cuda" if _available("cuda") else ("mps" if _available("mps") else "cpu")

    if preferred is None:
        return torch.device(best)

    wanted = torch.device(preferred)
    if _available(wanted.type):
        return wanted

    logger.warning(
        "Requested device %r is not available; falling back to %r. "
        "Triton kernels need CUDA or ROCm; other devices use the pure-PyTorch "
        "path, which is exact but slower.", str(wanted), best)
    return torch.device(best)


def synchronize(device=None) -> None:
    """Block until queued work on ``device`` has finished.

    Needed before reading a wall clock: both CUDA and MPS queue work
    asynchronously. Guards written as ``if t.is_cuda: torch.cuda.synchronize()``
    silently skip MPS, so timings taken there measure only the launch.

    Args:
        device: torch device, a tensor, or None for the current default.
    """
    kind = None
    if device is not None:
        kind = getattr(getattr(device, "device", device), "type", None)
    if kind is None:
        kind = "cuda" if torch.cuda.is_available() else None
    if kind == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif kind == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def autocast_context(device, dtype=None):
    """Return an autocast context appropriate for ``device``.

    ``torch.amp.autocast('cuda')`` on a CPU tensor warns and disables itself,
    which is survivable but noisy and hides the fact that no mixed precision is
    happening. This picks the right device type and only enables autocast where
    it helps.

    Args:
        device: torch.device or str.
        dtype: torch.dtype or None. Defaults to float16 on CUDA and bfloat16
            elsewhere (CPU fp16 autocast is slow and MPS prefers bf16).
    Returns:
        A context manager.
    """
    device = torch.device(device)
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.amp.autocast(device_type=device.type,
                              dtype=dtype,
                              enabled=device.type in ("cuda", "cpu", "mps"))
