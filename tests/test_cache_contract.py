"""The cache contract: warn-and-recompute by default, error when unrecoverable.

The rule, chosen deliberately:

* Something that **can** be rebuilt from what the operator already holds — the
  dense KNN table, for instance — recomputes and warns once. Results are
  identical, only slower, so failing would be worse than continuing.
* Something that **cannot** — a decoder stage needing a table it did not build —
  raises :class:`MissingCacheContextError` and names the context manager to wrap
  with. A missing required context is a correctness problem, not a performance
  one.
* ``CachePolicy.STRICT`` turns the first case into an error too, for benchmarks
  where a silent recompute invalidates the measurement.
"""

import warnings

import pytest
import torch

from affmae.ops.cache import (
    CachePolicy,
    MissingCacheContextError,
    TensorCache,
    active_cache,
    cache_scope,
    reset_warnings,
    resolve_cache,
)


@pytest.fixture(autouse=True)
def _fresh_warn_state():
    reset_warnings()
    yield
    reset_warnings()


class TestScope:
    def test_no_cache_outside_a_scope(self):
        assert active_cache() is None

    def test_scope_provides_and_then_clears(self):
        with cache_scope(name="t") as cache:
            assert active_cache() is cache
            cache.get_or_compute(("k",), lambda: torch.zeros(2))
            assert len(cache) == 1
        assert active_cache() is None
        # Entries released on exit, so device memory is bounded by the scope.
        assert len(cache) == 0

    def test_scopes_nest_innermost_first(self):
        with cache_scope(name="outer") as outer:
            with cache_scope(name="inner") as inner:
                assert active_cache() is inner
            assert active_cache() is outer

    def test_scope_clears_even_on_exception(self):
        try:
            with cache_scope(name="t") as cache:
                cache.get_or_compute(("k",), lambda: 1)
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert active_cache() is None
        assert len(cache) == 0


class TestCaching:
    def test_second_lookup_does_not_recompute(self):
        calls = []
        with cache_scope() as cache:
            for _ in range(3):
                cache.get_or_compute(("k",), lambda: calls.append(1) or 5)
        assert len(calls) == 1
        assert (cache.hits, cache.misses) == (2, 1)

    def test_distinct_keys_are_distinct_entries(self):
        with cache_scope() as cache:
            a = cache.get_or_compute(("a",), lambda: torch.tensor([1]))
            b = cache.get_or_compute(("b",), lambda: torch.tensor([2]))
        assert a.item() == 1 and b.item() == 2

    def test_lru_bound_is_enforced(self):
        """Unbounded growth is how the old class-level dict leaked memory."""
        with cache_scope(max_entries=2) as cache:
            for i in range(5):
                cache.get_or_compute((i,), lambda i=i: i)
            assert len(cache) == 2

    def test_rejects_a_nonsensical_bound(self):
        with pytest.raises(ValueError, match="max_entries"):
            TensorCache(max_entries=0)


class TestPolicy:
    def test_warn_recomputes_and_warns_once(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(4):
                got = resolve_cache(None, op="CachedKNN", reason="the KNN grid")
            assert got is None, "WARN must fall through to a recompute"
        assert len(caught) == 1, "a training loop must not warn every step"
        assert "cache_scope" in str(caught[0].message)

    def test_warning_names_the_operator_and_the_work(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_cache(None, op="CachedKNN", reason="the KNN grid")
        message = str(caught[0].message)
        assert "CachedKNN" in message and "the KNN grid" in message

    def test_strict_raises_instead_of_recomputing(self):
        with pytest.raises(MissingCacheContextError, match="STRICT"):
            resolve_cache(None, op="CachedKNN", reason="the KNN grid",
                          policy=CachePolicy.STRICT)

    def test_silent_recomputes_without_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = resolve_cache(None, op="X", reason="y",
                                policy=CachePolicy.SILENT)
        assert got is None and caught == []

    def test_required_raises_and_names_the_context_manager(self):
        """Unrecoverable: no policy should let this through silently."""
        for policy in CachePolicy:
            with pytest.raises(MissingCacheContextError) as exc:
                resolve_cache(None, op="DecoderStage", reason="the stage table",
                              policy=policy, required=True)
            assert "cache_scope" in str(exc.value)

    def test_inside_a_scope_no_policy_applies(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with cache_scope() as cache:
                got = resolve_cache(None, op="X", reason="y",
                                    policy=CachePolicy.STRICT)
        assert got is cache, "a present cache must satisfy even STRICT"
        assert caught == []

    def test_explicit_cache_beats_the_ambient_scope(self):
        explicit = TensorCache(name="explicit")
        with cache_scope(name="ambient"):
            assert resolve_cache(explicit, op="X", reason="y") is explicit


class TestCachedKNNContract:
    """The concrete operator the user singled out: the KNN grid."""

    @staticmethod
    def _positions(grid=8):
        ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid),
                                indexing="ij")
        return torch.stack([xs.reshape(-1), ys.reshape(-1)], -1).unsqueeze(0).float()

    def test_same_result_inside_and_outside_a_scope(self):
        """The whole justification for warn-and-continue."""
        from affmae.ops import CachedKNN

        pos = self._positions()
        knn = CachedKNN(grid_h=8, grid_w=8, backend="reference")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outside = knn(pos, cache_key=("stage", 0))
        with cache_scope() as cache:
            inside = knn(pos, cache_key=("stage", 0), cache=cache)

        torch.testing.assert_close(outside.to(torch.int64),
                                   inside.to(torch.int64), rtol=0, atol=0)

    def test_warns_once_outside_a_scope(self):
        from affmae.ops import CachedKNN

        pos = self._positions()
        knn = CachedKNN(grid_h=8, grid_w=8, backend="reference")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            knn(pos, cache_key=("s", 0))
            knn(pos, cache_key=("s", 0))
        assert len(caught) == 1

    def test_strict_policy_refuses_to_recompute(self):
        from affmae.ops import CachedKNN

        knn = CachedKNN(grid_h=8, grid_w=8, backend="reference",
                        policy=CachePolicy.STRICT)
        with pytest.raises(MissingCacheContextError):
            knn(self._positions(), cache_key=("s", 0))

    def test_reuses_the_table_within_a_scope(self):
        from affmae.ops import CachedKNN

        pos = self._positions()
        knn = CachedKNN(grid_h=8, grid_w=8, backend="reference")
        with cache_scope() as cache:
            for _ in range(3):
                knn(pos, cache_key=("stage", 0), cache=cache)
        assert (cache.hits, cache.misses) == (2, 1)

    def test_grid_is_required_and_validated(self):
        from affmae.ops import CachedKNN

        with pytest.raises(TypeError):
            CachedKNN()
        with pytest.raises(ValueError, match="positive"):
            CachedKNN(grid_h=0, grid_w=8)

    def test_rejects_too_few_kv_tokens(self):
        """dense_top4 needs at least four neighbours to pick from."""
        from affmae.ops import CachedKNN

        knn = CachedKNN(grid_h=8, grid_w=8, backend="reference")
        with pytest.raises(ValueError, match="Nk >= 4"):
            knn(torch.zeros(1, 3, 2))

    def test_no_cache_key_means_no_caching(self):
        """Without an explicit identity we must not invent one from data_ptr."""
        from affmae.ops import CachedKNN

        pos = self._positions()
        knn = CachedKNN(grid_h=8, grid_w=8, backend="reference")
        with cache_scope() as cache:
            knn(pos)
            knn(pos)
        assert len(cache) == 0

    def test_opting_out_of_caching_does_not_warn(self):
        """cache_key=None is a deliberate choice, not a missing cache scope.

        A config with ``decoder_knn_cache: false`` passes no key on every
        forward, so warning there would fire constantly for behaviour the user
        asked for. STRICT still refuses, because it promises to always cache.
        """
        from affmae.ops import CachedKNN

        knn = CachedKNN(grid_h=8, grid_w=8, backend="reference")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            knn(self._positions())
        assert [w for w in caught if w.category is RuntimeWarning] == []

        strict = CachedKNN(grid_h=8, grid_w=8, backend="reference",
                           policy=CachePolicy.STRICT)
        with pytest.raises(MissingCacheContextError, match="cache_key is None"):
            strict(self._positions())


class TestDecoderSharesTablesAcrossBlocks:
    """Every block in a decoder stage sees the same positions, so the KNN
    tables should be built once per stage, not once per block."""

    @staticmethod
    def _stage(n_blocks, backend="csr_knn_cached"):
        from affmae.layers.decoder import MSDecoderBlock

        torch.manual_seed(0)
        return torch.nn.ModuleList([
            MSDecoderBlock(d_model=64, n_heads=4, n_points=4, d_ffn=128,
                           grid_h=16, grid_w=16, deform_backend=backend)
            for _ in range(n_blocks)]).eval()

    def test_self_and_cross_attention_both_reuse(self):
        """Cross-attention used to pass cache_key=None, so it rebuilt the table
        in every block of every stage at inference."""
        blocks = self._stage(3)
        query_pos = torch.rand(1, 24, 2) * 15
        pos = torch.rand(1, 32, 2) * 15
        src = torch.randn(1, 32, 64)
        key = ("res5", 0, tuple(query_pos.shape), tuple(pos.shape))

        with torch.no_grad(), cache_scope() as cache:
            x = torch.randn(1, 24, 64)
            for block in blocks:
                x = block(query_tokens=x, query_pos=query_pos, src=src,
                          pos=pos, pos_embed=None, cache_key=key)
            entries = len(cache)
        # Two tables (self over query_pos, cross over pos), built once each;
        # the remaining 2 blocks x 2 tables are hits.
        assert (cache.misses, cache.hits) == (2, 4)
        assert entries == 2

    def test_self_and_cross_tables_do_not_collide(self):
        """Both use the same grid, so without a role in the key the cross table
        would be served the query-position table."""
        blocks = self._stage(1)
        query_pos = torch.rand(1, 32, 2) * 15
        pos = torch.rand(1, 32, 2) * 15   # same shape as query_pos on purpose
        key = ("res5", 0, tuple(query_pos.shape), tuple(pos.shape))

        with torch.no_grad(), cache_scope() as cache:
            blocks[0](query_tokens=torch.randn(1, 32, 64), query_pos=query_pos,
                      src=torch.randn(1, 32, 64), pos=pos, pos_embed=None,
                      cache_key=key)
            keys = list(cache._store)
        assert len(keys) == 2, keys
        roles = {k[1][-1] for k in keys}
        assert roles == {"self", "cross"}, keys
