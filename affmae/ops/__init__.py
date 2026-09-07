"""Standalone operators, reusable outside AFF-MAE.

One module per operation, plus one sibling per backend named for the machinery
it uses -- ``<op>.py`` dispatches, ``<op>_torch.py`` is the device-agnostic
implementation, ``<op>_triton.py`` the fused kernel, ``cuda_ext/`` the vendored
CUDA extension. Picking a backend is therefore an argument, not an import path::

    from affmae.ops import nbhd_attn

    out = nbhd_attn(q, k, v, member_idx, scale, backend="auto")

Every module here imports torch and nothing else from this project (Triton is
resolved lazily through :mod:`affmae.ops.dispatch`, so a torch-only install
works). None of them holds global state, so two instances in one process cannot
interfere.

    from affmae.ops import SpaceFillingCluster, CachedKNN, cache_scope
"""

from .cache import (
    CachePolicy,
    MissingCacheContextError,
    TensorCache,
    active_cache,
    cache_scope,
)
from .clustering import (
    SpaceFillingCluster,
    calculate_hilbert_order,
    calculate_peano_order,
    space_filling_cluster,
)
from .knn import CachedKNN, clamp_to_grid, dense_top4_knn, reference_dense_top4_knn
from .nbhd_attn import BACKENDS, nbhd_attn, resolve_backend

__all__ = [
    # Neighbourhood attention
    "nbhd_attn",
    "BACKENDS",
    "resolve_backend",
    # Backend capability probing
    "HAS_TRITON",
    "has_triton_backend",
    "can_use_triton",
    # Clustering / downsampling
    "SpaceFillingCluster",
    "space_filling_cluster",
    "calculate_peano_order",
    "calculate_hilbert_order",
    # KNN
    "CachedKNN",
    "dense_top4_knn",
    "reference_dense_top4_knn",
    "clamp_to_grid",
    # Cache contract
    "TensorCache",
    "CachePolicy",
    "MissingCacheContextError",
    "cache_scope",
    "active_cache",
]

# Backend capability probing lives in .dispatch, which imports Triton. Resolved
# on first access so `import affmae.ops` stays torch-only.
_FROM_DISPATCH = frozenset(
    {"HAS_TRITON", "has_triton_backend", "can_use_triton"})


def __getattr__(name):
    """Resolve the Triton-dependent names on first access (PEP 562)."""
    if name not in _FROM_DISPATCH:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f"{__name__}.dispatch"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
