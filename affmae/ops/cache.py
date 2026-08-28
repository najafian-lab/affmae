"""Explicit, caller-owned caches with a documented recompute policy.

We cache expensive intermediates: dense top-4 KNN table over KV
positions, and the KV edge index (for backward) used by the neighbourhood-attention backward.

Policy, when an operator wants a cache and none is available:

* ``CachePolicy.WARN`` (default): recompute and warn once per reason. Right
  for anything recoverable, like the KNN grid: the result is identical, only
  slower.
* ``CachePolicy.STRICT``: for tests and benchmarks, where a silent
  recompute would invalidate the measurement. a decoder stage needing a table it
did not build -- raises :class:`MissingCacheContextError`
* ``CachePolicy.SILENT``: recompute quietly.

"""

import logging
import warnings
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Hashable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "CachePolicy",
    "MissingCacheContextError",
    "TensorCache",
    "cache_scope",
    "active_cache",
]


class CachePolicy(Enum):
    """What to do when a cache is wanted but absent."""

    WARN = "warn"
    STRICT = "strict"
    SILENT = "silent"


class MissingCacheContextError(RuntimeError):
    """Raised when an operator requires a cache scope it cannot reconstruct."""


class TensorCache:
    """A bounded, explicitly-keyed cache of derived tensors.

    Args:
        max_entries: int, LRU bound. Prevents the unbounded growth the previous
            class-level dicts allowed.
        policy: CachePolicy, what a miss-without-a-cache does downstream. Stored
            here so an operator can consult it without a second argument.
        name: str, used in warnings so the message says which op recomputed.
    """

    def __init__(self, max_entries: int = 32,
                 policy: CachePolicy = CachePolicy.WARN,
                 name: str = "cache"):
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}.")
        self.max_entries = int(max_entries)
        self.policy = policy
        self.name = name
        self._store: "dict[Hashable, Any]" = {}
        self._order: "list[Hashable]" = []
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        """Return the cached value for ``key``, computing it on a miss.

        Args:
            key: hashable identity of the value. Must not be derived from
                ``tensor.data_ptr()`` — see the module docstring.
            compute: zero-argument callable producing the value.
        Returns:
            The cached or freshly computed value.
        """
        if key in self._store:
            self.hits += 1
            self._touch(key)
            return self._store[key]

        self.misses += 1
        value = compute()
        self._store[key] = value
        self._order.append(key)
        self._evict()
        return value

    def _touch(self, key):
        # Cheap for the sizes involved (max_entries is tens, not thousands).
        self._order.remove(key)
        self._order.append(key)

    def _evict(self):
        while len(self._order) > self.max_entries:
            self._store.pop(self._order.pop(0), None)

    def clear(self):
        """Drop every entry and release the tensors."""
        self._store.clear()
        self._order.clear()

    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return (f"TensorCache(name={self.name!r}, entries={len(self._store)}/"
                f"{self.max_entries}, hits={self.hits}, misses={self.misses}, "
                f"policy={self.policy.value})")


# Warn once per reason rather than once per call; a training loop would otherwise
# emit the same line every step.
_WARNED: "set[str]" = set()

# The active cache, per scope. A module-level *stack* would reintroduce exactly
# the cross-instance interference this file exists to remove, so the value is
# owned by the context manager and passed explicitly to operators that want it;
# this holder only serves `active_cache()` for code that cannot thread it
# through. It is deliberately not a stack of caches keyed by model.
_ACTIVE: "list[TensorCache]" = []


@contextmanager
def cache_scope(cache: Optional[TensorCache] = None, **kwargs):
    """Make a cache available for the duration of the block.

    Entries are dropped on exit, so device memory is bounded by the scope rather
    than by the process.

    Args:
        cache: TensorCache to activate, or None to build one from ``kwargs``.
        **kwargs: forwarded to :class:`TensorCache` when ``cache`` is None.
    Yields:
        The active TensorCache.
    """
    cache = cache if cache is not None else TensorCache(**kwargs)
    _ACTIVE.append(cache)
    try:
        yield cache
    finally:
        popped = _ACTIVE.pop()
        popped.clear()


def active_cache() -> Optional[TensorCache]:
    """Return the innermost active cache, or None outside any scope."""
    return _ACTIVE[-1] if _ACTIVE else None


def resolve_cache(cache: Optional[TensorCache], *, op: str, reason: str,
                  policy: Optional[CachePolicy] = None,
                  required: bool = False) -> Optional[TensorCache]:
    """Pick the cache to use, applying the miss policy.

    Args:
        cache: an explicitly supplied cache, or None to fall back to
            :func:`active_cache`.
        op: str, operator name for the message.
        reason: str, what would be recomputed. Also the warn-once key.
        policy: CachePolicy override; defaults to the resolved cache's policy,
            or ``WARN`` when there is no cache at all.
        required: bool, True when the value cannot be recomputed from what the
            operator holds.
    Returns:
        The cache to use, or None to mean "recompute".
    Raises:
        MissingCacheContextError: if ``required`` and no cache is available, or
            if the policy is STRICT.
    """
    resolved = cache if cache is not None else active_cache()
    if resolved is not None:
        return resolved

    effective = policy if policy is not None else CachePolicy.WARN

    if required:
        raise MissingCacheContextError(
            f"{op} needs a cache scope it cannot rebuild ({reason}). Wrap the "
            f"call:\n\n    from affmae.ops.cache import cache_scope\n"
            f"    with cache_scope() as cache:\n        ...\n")

    if effective is CachePolicy.STRICT:
        raise MissingCacheContextError(
            f"{op} would recompute {reason} because no cache scope is active, "
            f"and CachePolicy.STRICT forbids it. Wrap the call in "
            f"affmae.ops.cache.cache_scope(), or use CachePolicy.WARN.")

    if effective is CachePolicy.WARN:
        warn_key = f"{op}:{reason}"
        if warn_key not in _WARNED:
            _WARNED.add(warn_key)
            warnings.warn(
                f"{op}: recomputing {reason} because no cache scope is active. "
                f"Results are identical, only slower. Wrap the call in "
                f"affmae.ops.cache.cache_scope() to cache it, or pass "
                f"CachePolicy.SILENT to silence this.",
                RuntimeWarning, stacklevel=3)
    return None


def reset_warnings():
    """Forget which warnings have fired. For tests."""
    _WARNED.clear()
