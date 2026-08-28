"""Triton kernel correctness against CLUSTEN-free reference implementations.

The pre-existing forward/backward tests diff Triton against the compiled
CLUSTEN CUDA extension, which most installs will not have built, so they skip.
These tests use the pure-PyTorch references that ship in the kernel modules
instead, giving real kernel coverage on any CUDA device.
"""

import pytest
import torch

from conftest import requires_cuda
from affmae.ops.deform_attn_triton import dense_top4_knn
from affmae.ops.knn_torch import reference_dense_top4_knn
from affmae.ops.nbhd_attn_triton import (
    FlashNeighborhoodAttentionFunction,
    KVEdgeIndexCache,
    flash_nbhd_attn_reference_forward,
)

pytestmark = requires_cuda


def _neighbor_distances(coords, nn4, H, W):
    """Squared distances from each grid cell to the 4 neighbours chosen for it.

    Indices are not directly comparable between implementations: many
    candidates are equidistant, so Triton and the reference make different but
    equally valid tie-breaks. The distances are the real invariant.

    Note the output is indexed by *grid cell* ``p = y * W + x``, not by KV
    index, so the query position is derived from ``p`` rather than looked up in
    ``coords``.

    Args:
        coords: [1, Nk, 2] KV coordinates as (x, y).
        nn4: [1, H*W, 4] chosen neighbour indices into ``coords``.
        H: int, grid height.
        W: int, grid width.
    Returns:
        [H*W, 4] squared distances, sorted ascending per row.
    """
    pos = coords[0].float()
    p = torch.arange(H * W, device=coords.device)
    cells = torch.stack([(p % W).float(), (p // W).float()], dim=-1)
    neighbours = pos[nn4[0].long()]
    d2 = ((neighbours - cells[:, None, :]) ** 2).sum(-1)
    return d2.sort(dim=-1).values


def _grid_coords(grid, device):
    """All integer (x, y) positions of a ``grid x grid`` lattice, as [N, 2]."""
    ys, xs = torch.meshgrid(
        torch.arange(grid, device=device),
        torch.arange(grid, device=device),
        indexing="ij",
    )
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)


class TestDenseTop4KNN:
    """The dense KNN table that feeds deformable attention."""

    @pytest.mark.parametrize("grid", [8, 16, 64])
    def test_matches_reference_on_full_grid(self, grid):
        device = "cuda"
        coords = _grid_coords(grid, device).unsqueeze(0).to(torch.int32)

        got = dense_top4_knn(coords, H=grid, W=grid)
        want = reference_dense_top4_knn(coords, H=grid, W=grid)

        assert got.shape == want.shape, f"{got.shape} != {want.shape}"
        torch.testing.assert_close(_neighbor_distances(coords, got, grid, grid),
                                   _neighbor_distances(coords, want, grid, grid),
                                   rtol=0, atol=0)

    def test_matches_reference_on_sparse_positions(self):
        """A subsampled point set, i.e. Nk far smaller than H*W."""
        device, grid = "cuda", 32
        gen = torch.Generator(device=device).manual_seed(0)
        coords = torch.randint(0, grid, (2, 200, 2), device=device,
                               generator=gen, dtype=torch.int32)

        got = dense_top4_knn(coords, H=grid, W=grid)
        want = reference_dense_top4_knn(coords, H=grid, W=grid)
        for b in range(coords.shape[0]):
            torch.testing.assert_close(
                _neighbor_distances(coords[b:b + 1], got[b:b + 1], grid, grid),
                _neighbor_distances(coords[b:b + 1], want[b:b + 1], grid, grid),
                rtol=0, atol=0)

    def test_indices_are_in_range(self):
        """Every returned index must address a real KV token."""
        device, grid, n_kv = "cuda", 16, 50
        gen = torch.Generator(device=device).manual_seed(1)
        coords = torch.randint(0, grid, (1, n_kv, 2), device=device,
                               generator=gen, dtype=torch.int32)
        # uint16 does not support reductions on CUDA, so widen first.
        nn4 = dense_top4_knn(coords, H=grid, W=grid).to(torch.int64)
        assert nn4.min().item() >= 0
        assert nn4.max().item() < n_kv, (
            f"index {nn4.max().item()} out of range for Nk={n_kv}")


class TestFlashNeighborhoodAttention:
    """The fused neighborhood attention used by the AFF encoder."""

    @staticmethod
    def _inputs(b, h, n, d, m, device, seed=0):
        gen = torch.Generator(device=device).manual_seed(seed)
        q = torch.randn(b, h, n, d, device=device, generator=gen)
        k = torch.randn(b, h, n, d, device=device, generator=gen)
        v = torch.randn(b, h, n, d, device=device, generator=gen)
        member = torch.randint(0, n, (b, n, m), device=device, generator=gen,
                               dtype=torch.int32)
        bias = torch.randn(b, h, n, m, device=device, generator=gen)
        blank_k = torch.randn(h, d, device=device, generator=gen)
        blank_v = torch.randn(h, d, device=device, generator=gen)
        return q, k, v, member, bias, blank_k, blank_v

    @pytest.mark.parametrize("b,h,n,d,m", [(1, 2, 64, 32, 16), (2, 4, 128, 32, 32)])
    def test_forward_matches_reference(self, b, h, n, d, m):
        device = "cuda"
        q, k, v, member, bias, bk, bv = self._inputs(b, h, n, d, m, device)
        scale = d ** -0.5

        got = FlashNeighborhoodAttentionFunction.apply(
            q, k, v, member, bias, None, bk, bv, float(scale))
        want, _lse = flash_nbhd_attn_reference_forward(
            q, k, v, member, scale, bias=bias, mask=None,
            blank_k=bk, blank_v=bv)

        torch.testing.assert_close(got.to(torch.float64), want.to(torch.float64),
                                   rtol=2e-3, atol=2e-3)

    def test_broadcast_mask_matches_expanded_mask(self):
        """[B,1,N,M] and [B,H,N,M] masks must give identical results.

        ClusterAttention passes the size-1 form to avoid materializing the
        per-head copy; this pins that the kernels really do broadcast it.
        """
        device = "cuda"
        b, h, n, d, m = 2, 4, 64, 32, 16
        q, k, v, member, bias, bk, bv = self._inputs(b, h, n, d, m, device, seed=3)
        gen = torch.Generator(device=device).manual_seed(4)
        mask_1 = torch.rand(b, 1, n, m, device=device, generator=gen) > 0.3
        mask_h = mask_1.expand(b, h, n, m).contiguous()
        scale = float(d ** -0.5)

        out_1 = FlashNeighborhoodAttentionFunction.apply(
            q, k, v, member, bias, mask_1, bk, bv, scale)
        out_h = FlashNeighborhoodAttentionFunction.apply(
            q, k, v, member, bias, mask_h, bk, bv, scale)

        torch.testing.assert_close(out_1, out_h, rtol=0, atol=0)

    def test_backward_gradients_match_reference(self):
        device = "cuda"
        b, h, n, d, m = 1, 2, 64, 32, 16
        q, k, v, member, bias, bk, bv = self._inputs(b, h, n, d, m, device, seed=7)
        scale = float(d ** -0.5)

        def run(fn):
            qs, ks, vs = (t.clone().detach().requires_grad_(True) for t in (q, k, v))
            if fn is FlashNeighborhoodAttentionFunction:
                out = fn.apply(qs, ks, vs, member, bias, None, bk, bv, scale)
            else:
                out, _ = fn(qs, ks, vs, member, scale, bias=bias, mask=None,
                            blank_k=bk, blank_v=bv)
            out.sum().backward()
            return qs.grad, ks.grad, vs.grad

        got = run(FlashNeighborhoodAttentionFunction)
        want = run(flash_nbhd_attn_reference_forward)

        for name, g, w in zip(("dq", "dk", "dv"), got, want):
            torch.testing.assert_close(
                g.to(torch.float64), w.to(torch.float64),
                rtol=5e-3, atol=5e-3, msg=lambda s, n=name: f"{n}: {s}")


class TestDeformBackendParity:
    """Both deform backends must exist and agree.

    ``deform_point_attn`` mapped ``backend="atomic"`` onto
    ``dv_backend="backend3_kv_atomic"``, but the dispatch had been renamed to
    ``"kv_atomic"`` without updating the caller, so the atomic path raised
    ``ValueError: Unknown dv_backend`` in the backward pass. With both live they
    cross-validate each other.
    """

    @staticmethod
    def _inputs(device, grid=8, n_heads=2, n_points=4, channels=16, seed=7):
        from affmae.ops.deform_attn_triton import dense_top4_knn

        coords = _grid_coords(grid, device).float()
        n = coords.shape[0]
        qpos = coords.unsqueeze(0).contiguous()
        kvpos = coords.unsqueeze(0).contiguous()
        nn4 = dense_top4_knn(
            kvpos.round().clamp(0, grid - 1).to(torch.int32), H=grid, W=grid)
        gen = torch.Generator(device=device).manual_seed(seed)
        return dict(
            query_pos=qpos, kv_pos=kvpos, nn4_idx=nn4, grid=grid,
            offsets=torch.randn(1, n, n_heads, n_points, 2, device=device, generator=gen),
            logits=torch.randn(1, n, n_heads, n_points, device=device, generator=gen),
            values=torch.randn(1, n, n_heads, channels, device=device, generator=gen),
            tau=torch.tensor(3.0, device=device),
        )

    def _run(self, backend, data):
        from affmae.ops.deform_attn_triton import deform_point_attn

        offs, logits, vals = (t.clone().detach().requires_grad_(True)
                              for t in (data["offsets"], data["logits"], data["values"]))
        out = deform_point_attn(
            query_pos=data["query_pos"], kv_pos=data["kv_pos"],
            sampling_offsets=offs, attn_logits=logits, values=vals,
            tau=data["tau"], nn4_idx=data["nn4_idx"],
            grid_h=data["grid"], grid_w=data["grid"], backend=backend)
        out.sum().backward()
        return out.detach(), vals.grad.clone()

    @pytest.mark.parametrize("backend", ["csr_knn_cached", "atomic"])
    def test_backend_runs_forward_and_backward(self, backend):
        out, dv = self._run(backend, self._inputs("cuda"))
        assert torch.isfinite(out).all()
        assert torch.isfinite(dv).all()

    def test_backends_agree(self):
        data = self._inputs("cuda")
        out_csr, dv_csr = self._run("csr_knn_cached", data)
        out_atomic, dv_atomic = self._run("atomic", data)
        torch.testing.assert_close(out_csr, out_atomic, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(dv_csr, dv_atomic, rtol=1e-3, atol=1e-3)

    def test_unknown_backend_is_rejected(self):
        from affmae.ops.deform_attn_triton import deform_point_attn

        data = self._inputs("cuda")
        with pytest.raises(ValueError, match="Unknown deform backend"):
            deform_point_attn(
                query_pos=data["query_pos"], kv_pos=data["kv_pos"],
                sampling_offsets=data["offsets"], attn_logits=data["logits"],
                values=data["values"], tau=data["tau"], nn4_idx=data["nn4_idx"],
                grid_h=data["grid"], grid_w=data["grid"], backend="nope")


def test_autotune_disk_cache_is_off_by_default():
    """One shared cache file plus many ranks is a write race.

    Also keeps ranks from independently selecting different kernels.
    """
    from affmae.ops import nbhd_attn_triton

    assert nbhd_attn_triton._CACHE_TO_DISK is False, (
        "CACHE_TO_DISK must default off; opt in per-run via the env var")


class TestInferenceSkipsBackwardOnlyWork:
    """The KV-owner edge index is read only by the backward pass.

    ``affmae/layers/aff.py`` installs a cache unconditionally, so before this
    guard every block built the index under ``torch.no_grad()`` -- about 0.7 ms
    per block, which made the v2 kernel 1.12x slower than v1 at inference
    despite being the faster kernel when training.
    """

    @staticmethod
    def _inputs(requires_grad, device="cuda", b=1, h=2, n=64, d=32, m=16):
        torch.manual_seed(3)
        make = lambda *shape: torch.randn(
            *shape, device=device, dtype=torch.float16
        ).requires_grad_(requires_grad)
        return dict(
            q=make(b, h, n, d), k=make(b, h, n, d), v=make(b, h, n, d),
            member=torch.randint(0, n, (b, n, m), device=device,
                                 dtype=torch.int32),
            bias=torch.randn(b, h, n, m, device=device, dtype=torch.float16),
            bk=make(h, d), bv=make(h, d), scale=float(d ** -0.5))

    def _apply(self, args):
        return FlashNeighborhoodAttentionFunction.apply(
            args["q"], args["k"], args["v"], args["member"], args["bias"],
            None, args["bk"], args["bv"], args["scale"])

    def test_no_grad_builds_no_edge_index(self):
        cache = KVEdgeIndexCache(max_entries=4)
        FlashNeighborhoodAttentionFunction.set_kv_edge_cache(cache)
        try:
            args = self._inputs(requires_grad=False)
            with torch.no_grad():
                self._apply(args)
            assert len(cache._cache) == 0, (
                "an edge index was built under no_grad; nothing reads it")
        finally:
            FlashNeighborhoodAttentionFunction.set_kv_edge_cache(None)

    def test_training_builds_it_in_the_forward_pass(self):
        """Skipping it when grad *is* needed makes backward rebuild it.

        Asserting only after ``.backward()`` is not enough: the backward
        launcher populates the cache itself, so that check passes even when the
        forward pass wrongly skipped it. Inspect the cache *before* backward.
        """
        cache = KVEdgeIndexCache(max_entries=4)
        FlashNeighborhoodAttentionFunction.set_kv_edge_cache(cache)
        try:
            args = self._inputs(requires_grad=True)
            out = self._apply(args)
            assert len(cache._cache) == 1, (
                "the forward pass skipped the edge index while training, so "
                "backward has to rebuild it")
            out.sum().backward()
            assert args["q"].grad is not None
        finally:
            FlashNeighborhoodAttentionFunction.set_kv_edge_cache(None)

    def test_forward_is_bit_identical_either_way(self):
        cache = KVEdgeIndexCache(max_entries=4)
        FlashNeighborhoodAttentionFunction.set_kv_edge_cache(cache)
        try:
            args = self._inputs(requires_grad=False)
            with torch.no_grad():
                without = self._apply(args)
            grad_args = dict(args)
            for name in ("q", "k", "v", "bk", "bv"):
                grad_args[name] = args[name].detach().requires_grad_(True)
            with_grad = self._apply(grad_args)
            torch.testing.assert_close(without, with_grad.detach(),
                                       rtol=0, atol=0)
        finally:
            FlashNeighborhoodAttentionFunction.set_kv_edge_cache(None)

class TestGradModeIsReadOutsideTheKernel:
    """Whether backward state is needed cannot be detected inside forward.

    Two traps, both of which produced a guard that looked right and was not:

    * ``torch.is_grad_enabled()`` is always False inside an
      ``autograd.Function.forward``, so a guard using it skips the work in
      training too -- and the cost merely moves into backward, which rebuilds
      the index and warns.
    * ``ctx.needs_input_grad`` is True even under ``no_grad`` when an input
      arrives as an ``nn.Parameter``, because a Parameter's ``requires_grad``
      flag does not depend on grad mode. Neighbourhood attention passes
      ``blank_k``/``blank_v`` straight in, so this fired on every inference call.

    The caller therefore reads grad mode and passes it down.
    """

    @staticmethod
    def _count_builds(fn):
        """Run ``fn`` with a stage cache installed, counting index builds.

        The cache is what ``affmae/layers/aff.py`` installs per encoder stage;
        without it the forward pass has nowhere to put an index and skips it
        regardless of grad mode, which would make these assertions vacuous.
        """
        import affmae.ops.nbhd_attn_triton as kernels

        calls = {"n": 0}
        original = kernels._build_kv_edge_index

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        kernels._build_kv_edge_index = counting
        FlashNeighborhoodAttentionFunction.set_kv_edge_cache(
            KVEdgeIndexCache(max_entries=4))
        try:
            fn()
        finally:
            kernels._build_kv_edge_index = original
            FlashNeighborhoodAttentionFunction.set_kv_edge_cache(None)
        return calls["n"]

    @staticmethod
    def _module():
        from affmae.layers.attention import ClusterAttention

        torch.manual_seed(0)
        return ClusterAttention(dim=64, num_heads=4,
                                backend="flash_nbhd_attn").cuda().half()

    @staticmethod
    def _inputs(n=64, m=8):
        return (torch.randn(1, n, 64, device="cuda", dtype=torch.float16),
                torch.randint(0, n, (1, n, m), device="cuda", dtype=torch.int32),
                torch.randint(0, 200, (1, n, m), device="cuda", dtype=torch.int32))

    def test_inference_builds_none(self):
        module = self._module().eval()
        feat, member, pe = self._inputs()

        def run():
            with torch.no_grad():
                module(feat, member, None, pe, False)

        assert self._count_builds(run) == 0

    def test_training_builds_them_in_forward(self):
        module = self._module().train()
        feat, member, pe = self._inputs()

        def run():
            module(feat, member, None, pe, False)

        assert self._count_builds(run) > 0

    def test_output_does_not_depend_on_the_guard(self):
        module = self._module().eval()
        feat, member, pe = self._inputs()
        with torch.no_grad():
            without = module(feat, member, None, pe, False)
        with_grad = module(feat, member, None, pe, False)
        torch.testing.assert_close(without, with_grad.detach(), rtol=0, atol=0)


class TestDenseTop4LaunchParameters:
    """Warps are matched to the vector width, which must not change results."""

    def test_warps_cover_the_block_width(self):
        from affmae.ops.deform_attn_triton import _warps_for_width

        # Every value in the kernel is a [BLOCK_P] vector, so a program wider
        # than the data wastes lanes: 32 elements across 4 warps uses 8 of each
        # warp's 32 lanes.
        assert _warps_for_width(32) == 1
        assert _warps_for_width(64) == 2
        assert _warps_for_width(128) == 4

    def test_never_returns_zero_warps(self):
        """A block narrower than one warp still needs a warp."""
        from affmae.ops.deform_attn_triton import _warps_for_width

        assert _warps_for_width(16) == 1
        assert _warps_for_width(1) == 1

    @pytest.mark.parametrize("grid,n_kv", [(16, 64), (32, 512), (64, 4096)])
    def test_launch_parameters_do_not_change_the_table(self, grid, n_kv):
        """Bit-identical across warp counts, so this is purely a speed choice."""
        import affmae.ops.deform_attn_triton as kernels
        from affmae.ops.knn import clamp_to_grid

        torch.manual_seed(0)
        coords = clamp_to_grid(
            torch.rand(1, n_kv, 2, device="cuda") * (grid - 1), grid, grid)

        original = kernels._warps_for_width
        try:
            tables = []
            for warps in (1, 2, 4):
                kernels._warps_for_width = lambda _width, w=warps: w
                tables.append(kernels.dense_top4_knn(coords, H=grid, W=grid))
        finally:
            kernels._warps_for_width = original

        for other in tables[1:]:
            torch.testing.assert_close(tables[0], other, rtol=0, atol=0)


class TestTileDepthNeverUnderrunsHeadDim:
    """BLOCK_D >= HEAD_DIM is correctness, not tuning.

    The neighbourhood-attention kernel does not tile its depth loop, so a
    BLOCK_D smaller than HEAD_DIM never computes the remaining channels. The
    config pruner said as much in a comment and then violated it in its own
    fallback: it minimized over (BLOCK_Q, BLOCK_MN, BLOCK_D) together, so
    whenever the tile filters emptied the candidate list it happily returned the
    32-wide tile for a 64-wide head.

    No config offers BLOCK_MN below 8, so any neighbourhood smaller than 8
    emptied that list. head_dim=64 with a neighbourhood of 4 returned all-NaN;
    head_dim=32 was correct only because 32 happens to match. Reproduced on a
    GH200, so this was never ROCm-specific -- it surfaced during ROCm testing
    only because that sweep varied head_dim, which no test had.
    """

    @staticmethod
    def _configs():
        import affmae.ops.nbhd_attn_triton as kernels

        return next(v for name, v in vars(kernels).items()
                    if name.endswith("CONFIGS") and isinstance(v, list) and v
                    and "BLOCK_Q" in getattr(v[0], "kwargs", {}))

    @pytest.mark.parametrize("head_dim", [8, 16, 32, 64])
    @pytest.mark.parametrize("n_nb", [1, 2, 4, 8, 16, 32])
    def test_pruner_never_returns_a_shallow_tile(self, head_dim, n_nb):
        from affmae.ops.nbhd_attn_triton import _prune_tile_configs

        chosen = _prune_tile_configs(
            self._configs(),
            {"N_CTX": 64, "NEIGHBOR_SIZE": n_nb, "HEAD_DIM": head_dim})
        assert chosen, "the pruner must always return at least one config"
        for cfg in chosen:
            block_d = int(cfg.kwargs["BLOCK_D"])
            assert block_d >= head_dim, (
                f"BLOCK_D={block_d} < HEAD_DIM={head_dim} at neighbourhood "
                f"{n_nb}: channels {block_d}..{head_dim - 1} would never be "
                f"computed")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.parametrize("head_dim,n_nb", [(64, 4), (64, 2), (64, 1),
                                               (32, 4), (16, 4), (8, 4)])
    def test_output_is_finite_and_matches_the_reference(self, head_dim, n_nb):
        """The end-to-end symptom, at the shapes that used to produce NaN."""
        from affmae.ops.nbhd_attn import nbhd_attn
        from affmae.ops.nbhd_attn_torch import neighborhood_attention

        b, h, n, d, m = 1, 2, 64, head_dim, n_nb
        gen = torch.Generator(device="cuda").manual_seed(0)
        make = lambda *shape: torch.randn(*shape, device="cuda", generator=gen)
        q, k, v = make(b, h, n, d), make(b, h, n, d), make(b, h, n, d)
        member = torch.randint(0, n, (b, n, m), device="cuda", generator=gen,
                               dtype=torch.int32)
        bias, blank_k, blank_v = make(b, h, n, m), make(h, d), make(h, d)
        scale = float(d ** -0.5)

        triton_out = nbhd_attn(q, k, v, member, scale, bias=bias,
                               blank_k=blank_k, blank_v=blank_v,
                               backend="triton")
        reference = neighborhood_attention(q, k, v, member, scale, bias=bias,
                                           blank_k=blank_k, blank_v=blank_v)
        assert torch.isfinite(triton_out).all(), "non-finite kernel output"
        torch.testing.assert_close(triton_out, reference, rtol=2e-2, atol=5e-3)


class TestWarpCountFitsTheWavefront:
    """`num_warps` must not exceed what the tile can fill on a 64-lane device.

    The configs were templated for 32-lane NVIDIA warps. On an MI300X a
    wavefront is 64 lanes, so a config asking for 2 warps over an 8x8 tile
    reserves 128 lanes for 64 elements and the cross-warp reductions in the
    online softmax return garbage.

    Measured on that hardware, fp16, BLOCK_Q=BLOCK_MN=8: num_warps 1 finite,
    2 and 4 NaN in 60-70% of elements, identical at BLOCK_D 16 and 32 and at
    num_stages 1 and 2 -- so it is the warp count, not the tile depth or the
    pipeline depth. Clamping took a 270-cell ROCm sweep from 56 non-finite to 0.
    """

    def test_it_is_a_no_op_on_cuda(self):
        """The shipped configs are tuned at 32 lanes; do not silently retune."""
        from affmae.ops.dispatch import clamp_num_warps, is_hip

        if is_hip():
            pytest.skip("this asserts the CUDA branch")
        for requested, elements in ((1, 64), (2, 64), (4, 64), (8, 256), (4, 16)):
            assert clamp_num_warps(requested, elements) == requested

    def test_it_clamps_to_the_lane_budget_on_rocm(self, monkeypatch):
        from affmae.ops import dispatch

        monkeypatch.setattr(dispatch, "is_hip", lambda: True)
        # 64 elements is one 64-lane wavefront, so 2 or 4 warps cannot be filled.
        assert dispatch.clamp_num_warps(2, 64) == 1
        assert dispatch.clamp_num_warps(4, 64) == 1
        # 256 elements is four wavefronts, so up to 4 warps is honest.
        assert dispatch.clamp_num_warps(4, 256) == 4
        assert dispatch.clamp_num_warps(8, 256) == 4
        # Smaller than a single wavefront still needs one warp, never zero.
        assert dispatch.clamp_num_warps(4, 16) == 1

    def test_never_returns_zero(self, monkeypatch):
        from affmae.ops import dispatch

        monkeypatch.setattr(dispatch, "is_hip", lambda: True)
        for elements in (0, 1, 8, 63):
            assert dispatch.clamp_num_warps(4, elements) >= 1

    def test_every_shipped_config_fits_its_tile_on_rocm(self, monkeypatch):
        """Applied where templates become configs, so no config can violate it."""
        from affmae.ops import dispatch

        monkeypatch.setattr(dispatch, "is_hip", lambda: True)
        import importlib

        import affmae.ops.nbhd_attn_triton as kernels
        importlib.reload(kernels)
        try:
            configs = next(
                v for name, v in vars(kernels).items()
                if name.endswith("CONFIGS") and isinstance(v, list) and v
                and "BLOCK_Q" in getattr(v[0], "kwargs", {}))
            for cfg in configs:
                tile = int(cfg.kwargs["BLOCK_Q"]) * int(cfg.kwargs["BLOCK_MN"])
                assert cfg.num_warps * 64 <= max(tile, 64), (
                    f"{cfg.kwargs} asks for {cfg.num_warps} warps "
                    f"({cfg.num_warps * 64} lanes) over {tile} elements")
        finally:
            monkeypatch.undo()
            importlib.reload(kernels)


class TestPositionTableCoversTheInput:
    """An input larger than the table degrades silently, so it must be rejected.

    aff.py clamps relative offsets to the table width. Past that every distant
    token pair shares one positional-bias entry, which does not raise and does
    not look wrong -- it just removes long-range positional information. The
    table was sized for 1024px while the repo benchmarks the encoder at 1536
    and 2048.
    """

    @pytest.mark.parametrize("res", [512, 768, 1024, 1536, 2048])
    def test_supported_resolutions_build(self, res):
        from affmae.layers.pos_embed import assert_resolution_fits

        assert_resolution_fits(res, 8) is None

    def test_an_oversized_input_is_rejected(self):
        from affmae.layers.pos_embed import assert_resolution_fits

        with pytest.raises(ValueError, match="MAX_INPUT_RESOLUTION"):
            assert_resolution_fits(2560, 8)

    def test_the_table_spans_the_declared_maximum(self):
        """rel_pos_width must actually cover MAX_INPUT_RESOLUTION."""
        from affmae.layers import pos_embed as pe

        needed = pe.MAX_INPUT_RESOLUTION // pe.MIN_PATCH_SIZE - 1
        assert pe.rel_pos_width == needed
        assert pe.table_width == 2 * needed + 1
        assert pe.pre_table.shape == (pe.table_width ** 2, 5)

    def test_a_smaller_patch_needs_a_wider_table(self):
        """patch 4 at 2048 exceeds the table; the guard must say so."""
        from affmae.layers.pos_embed import assert_resolution_fits

        with pytest.raises(ValueError):
            assert_resolution_fits(2048, 4)
