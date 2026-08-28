"""Cached dense top-4 nearest-neighbour lookup over a token grid.

The deformable decoder samples continuous locations and needs, for each cell of
the patch lattice, the four nearest KV tokens. That table depends only on the KV
positions and the grid size, so it is worth caching across the decoder blocks of
one forward pass — but the cache must be owned by the caller, not by the class.
See :mod:`affmae.ops.cache` for why.

    from affmae.ops import CachedKNN, cache_scope

    knn = CachedKNN(grid_h=128, grid_w=128)
    with cache_scope() as cache:
        table = knn(kv_positions, cache=cache)   # built once
        table = knn(kv_positions, cache=cache)   # reused

Used outside a scope it recomputes and warns once; the values are identical.
"""

import torch
import torch.nn as nn

from .cache import CachePolicy, MissingCacheContextError, resolve_cache

__all__ = ["CachedKNN", "dense_top4_knn", "reference_dense_top4_knn",
           "clamp_to_grid"]


def dense_top4_knn(*args, **kwargs):
    """Triton dense top-4 KNN. Imported lazily so torch-only installs work.

    See :func:`affmae.ops.deform_attn_triton.dense_top4_knn`.
    """
    from .deform_attn_triton import dense_top4_knn as _impl

    return _impl(*args, **kwargs)


def reference_dense_top4_knn(*args, **kwargs):
    """Pure-PyTorch reference for :func:`dense_top4_knn`; device-agnostic.

    See :func:`affmae.ops.knn_torch.reference_dense_top4_knn`.
    """
    from .knn_torch import reference_dense_top4_knn as _impl

    return _impl(*args, **kwargs)


def clamp_to_grid(pos: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """Round positions and clamp each axis to its own grid extent.

    Args:
        pos: [..., 2] coordinates as (x, y).
        grid_h: int, lattice height; bounds the y axis.
        grid_w: int, lattice width; bounds the x axis.
    Returns:
        int32 tensor of the same shape.

    Note:
        Per-axis on purpose. Clamping both axes to ``max(grid_h, grid_w) - 1``
        lets the shorter axis index outside the lattice on a non-square grid.
    """
    rounded = pos.round()
    return torch.stack(
        (rounded[..., 0].clamp(min=0, max=grid_w - 1),
         rounded[..., 1].clamp(min=0, max=grid_h - 1)),
        dim=-1,
    ).to(torch.int32).contiguous()


class CachedKNN(nn.Module):
    """Dense top-4 KNN table over a fixed token lattice, with optional caching.

    Holds no parameters. The grid size is required: it must equal the patch grid
    (``img_size // patch_size``), because coordinates are clamped to it and an
    undersized grid silently folds distant tokens onto the last row/column.

    Args:
        grid_h: int, lattice height.
        grid_w: int, lattice width.
        policy: CachePolicy applied when no cache is available. WARN (default)
            recomputes and warns once; STRICT raises; SILENT recomputes quietly.
        backend: str, ``"auto"`` uses Triton where it can actually run and the
            PyTorch reference otherwise; ``"triton"`` and ``"reference"`` force
            one. ``"triton"`` on a host without Triton raises at launch, by
            design -- an explicit choice should not be silently overridden.
    """

    def __init__(self, grid_h: int, grid_w: int,
                 policy: CachePolicy = CachePolicy.WARN,
                 backend: str = "auto"):
        super().__init__()
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"grid must be positive, got {grid_h}x{grid_w}.")
        if backend not in ("auto", "triton", "reference"):
            raise ValueError(f"backend must be auto|triton|reference, got {backend!r}.")
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.policy = policy
        self.backend = backend

    def _impl(self, kv_int):
        """Pick an implementation for these tensors.

        ``auto`` asks whether Triton can actually run, not whether the tensor is
        on a CUDA-like device. Those differ on a CUDA or ROCm host with no
        Triton installed and no live Triton backend, where ``is_cuda`` is True
        and the kernel would fail on launch.
        """
        from .dispatch import can_use_triton

        use_triton = self.backend == "triton" or (
            self.backend == "auto" and can_use_triton(kv_int))
        fn = dense_top4_knn if use_triton else reference_dense_top4_knn
        return fn(kv_int, H=self.grid_h, W=self.grid_w)

    def forward(self, kv_pos: torch.Tensor, cache=None,
                cache_key=None) -> torch.Tensor:
        """Build (or fetch) the neighbour table for ``kv_pos``.

        Args:
            kv_pos: [B, Nk, 2] KV coordinates.
            cache: TensorCache to use, or None to fall back to the active scope.
            cache_key: hashable identity for these positions. Required to cache;
                without it the table is recomputed, because deriving a key from
                the tensor's storage address is not safe (addresses are reused
                after a free).
        Returns:
            [B, grid_h * grid_w, 4] neighbour indices.
        """
        if kv_pos.shape[-1] != 2:
            raise ValueError(f"kv_pos must be [..., 2], got {tuple(kv_pos.shape)}.")
        if kv_pos.shape[-2] < 4:
            raise ValueError(
                f"dense top-4 KNN needs Nk >= 4, got {kv_pos.shape[-2]}.")

        kv_int = clamp_to_grid(kv_pos, self.grid_h, self.grid_w)

        if cache_key is None:
            # The caller opted out of caching, so recomputing is the requested
            # behaviour, not a surprise -- WARN would fire on every forward of a
            # config that deliberately disables the cache. STRICT still raises,
            # because under STRICT the caller promised to always cache.
            if self.policy is CachePolicy.STRICT:
                raise MissingCacheContextError(
                    "CachedKNN: cache_key is None, so the dense KNN table "
                    "cannot be cached, and CachePolicy.STRICT forbids "
                    "recomputing it. Pass a cache_key, or use CachePolicy.WARN.")
            return self._impl(kv_int)

        resolved = resolve_cache(cache, op="CachedKNN",
                                 reason="the dense KNN table",
                                 policy=self.policy)
        if resolved is None:
            return self._impl(kv_int)
        key = ("dense_top4_knn", cache_key, self.grid_h, self.grid_w,
               str(kv_int.device), self.backend)
        return resolved.get_or_compute(key, lambda: self._impl(kv_int))

    def extra_repr(self) -> str:
        return (f"grid_h={self.grid_h}, grid_w={self.grid_w}, "
                f"backend={self.backend!r}, policy={self.policy.value}")
