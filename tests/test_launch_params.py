"""Launch parameters are derived from device properties, not hardcoded.

Static constants tuned on one card are wrong on the next, so the rules in
:mod:`affmae.ops.launch` compute them. Two constraints are load-bearing and
easy to break:

* a Triton block reaching ``tl.arange(0, BLOCK)`` must be a power of two -- 48
  fails to compile, so a rule may never return one;
* changing a block size changes which tiles are partial, so masking must hold
  for ``total_work % BLOCK != 0``.
"""

import pytest
import torch

from affmae.ops.launch import (
    DeviceProfile,
    block_for_saturation,
    clamp_pow2,
    largest_pow2_at_most,
    warps_for_width,
)


def _is_pow2(value):
    return value >= 1 and (value & (value - 1)) == 0


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class TestPowerOfTwoGuards:
    @pytest.mark.parametrize("value,expected", [
        (1, 1), (2, 2), (3, 2), (31, 16), (32, 32), (33, 32), (1000, 512)])
    def test_largest_pow2_at_most(self, value, expected):
        assert largest_pow2_at_most(value) == expected

    @pytest.mark.parametrize("value", [0, -1, -100])
    def test_never_returns_zero(self, value):
        """A zero block size would launch nothing and divide by zero."""
        assert largest_pow2_at_most(value) == 1

    @pytest.mark.parametrize("value", [3, 17, 48, 100, 300, 1 << 20])
    def test_clamp_always_yields_a_power_of_two_in_range(self, value):
        got = clamp_pow2(value, 16, 256)
        assert _is_pow2(got), got
        assert 16 <= got <= 256

    def test_clamp_rounds_the_low_bound_up(self):
        """A non-power-of-two low bound must not leak through."""
        assert clamp_pow2(1, low=17, high=256) == 32

    def test_clamp_rejects_an_empty_range(self):
        with pytest.raises(ValueError, match="no power of two"):
            clamp_pow2(64, low=100, high=127)


class TestWarpsForWidth:
    @staticmethod
    def _profile(warp_size=32, max_threads_per_block=1024):
        return DeviceProfile(
            name="test", warp_size=warp_size, sm_count=100,
            max_threads_per_sm=2048, max_threads_per_block=max_threads_per_block,
            shared_mem_per_block=49152, regs_per_sm=65536, capability=(9, 0),
            l2_bytes=0, sm_clock_mhz=None, memory_clock_mhz=None,
            bandwidth_gb_s=None, is_hip=False)

    def test_matches_the_vector_width_on_cuda(self):
        p = self._profile(warp_size=32)
        assert [warps_for_width(w, p) for w in (32, 64, 128, 256)] == [1, 2, 4, 8]

    def test_scales_with_the_wavefront_on_rocm(self):
        """A 64-lane wavefront needs half the warps for the same width."""
        p = self._profile(warp_size=64)
        assert [warps_for_width(w, p) for w in (32, 64, 128, 256)] == [1, 1, 2, 4]

    def test_never_returns_zero_warps(self):
        p = self._profile(warp_size=32)
        assert warps_for_width(1, p) == 1
        assert warps_for_width(16, p) == 1

    def test_respects_the_thread_ceiling(self):
        """A very wide block cannot ask for more warps than a block may hold."""
        p = self._profile(warp_size=32, max_threads_per_block=1024)
        assert warps_for_width(1 << 20, p) == p.max_warps_per_block == 32


class TestBlockForSaturation:
    @staticmethod
    def _profile(sm_count):
        return DeviceProfile(
            name="test", warp_size=32, sm_count=sm_count,
            max_threads_per_sm=2048, max_threads_per_block=1024,
            shared_mem_per_block=49152, regs_per_sm=65536, capability=(9, 0),
            l2_bytes=0, sm_clock_mhz=None, memory_clock_mhz=None,
            bandwidth_gb_s=None, is_hip=False)

    def test_keeps_the_requested_number_of_waves(self):
        p = self._profile(sm_count=128)
        work = 16384
        block = block_for_saturation(work, p, min_waves=2)
        waves = work / block / p.sm_count
        assert waves >= 2.0, (block, waves)

    def test_a_card_with_more_sms_gets_smaller_blocks(self):
        """The whole point: the answer depends on the card."""
        work = 16384
        small = block_for_saturation(work, self._profile(16), min_waves=2)
        large = block_for_saturation(work, self._profile(256), min_waves=2)
        assert small > large, (small, large)

    def test_output_is_always_a_legal_block(self):
        for sm in (1, 16, 132, 1024):
            for work in (16, 1024, 4096, 16384, 1 << 20):
                got = block_for_saturation(work, self._profile(sm))
                assert _is_pow2(got) and 16 <= got <= 256, (sm, work, got)

    def test_tiny_workloads_do_not_collapse_the_block(self):
        got = block_for_saturation(4, self._profile(132))
        assert got == 16


@CUDA
class TestRealDeviceProfile:
    def test_reads_the_live_device(self):
        p = DeviceProfile.current()
        assert p.sm_count >= 1
        assert p.warp_size in (32, 64)
        assert p.capability[0] >= 1

    def test_bandwidth_is_plausible(self):
        """Computed from bus width and clock; a wrong formula shows up here."""
        p = DeviceProfile.current()
        if p.bandwidth_gb_s is None:
            pytest.skip("device does not report bus width or memory clock")
        assert 50 < p.bandwidth_gb_s < 20000, p.bandwidth_gb_s

    def test_profile_is_cached(self):
        assert DeviceProfile.current() is DeviceProfile.current()

    def test_works_without_cuda_too(self, monkeypatch):
        """Callers should not need a special case for a CPU-only host."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        p = DeviceProfile.current()
        assert p.name == "cpu" and p.sm_count == 1


@CUDA
class TestPartialTileMasking:
    """A retuned block size changes which tiles are partial.

    Every grid the other tests use is an exact multiple of the shipped
    BLOCK_P=32, so the ``mask_p = p < p_total`` path went unexercised until this
    class existed.
    """

    @staticmethod
    def _distances(table, kv, H, W):
        ys, xs = torch.meshgrid(torch.arange(H, device="cuda"),
                                torch.arange(W, device="cuda"), indexing="ij")
        query = torch.stack([xs.flatten(), ys.flatten()], -1).float()
        neighbours = kv[0].float()[table[0].long()]
        return (neighbours - query[:, None, :]).pow(2).sum(-1).sort(-1).values

    @pytest.mark.parametrize("H,W", [(7, 5), (13, 13), (17, 19), (33, 31)])
    @pytest.mark.parametrize("block", [16, 32, 64, 128])
    def test_partial_tiles_match_the_reference(self, H, W, block):
        from affmae.ops.deform_attn_triton import dense_top4_knn
        from affmae.ops.knn import clamp_to_grid
        from affmae.ops.knn_torch import reference_dense_top4_knn

        assert (H * W) % block != 0, "this case is meant to be a partial tile"
        torch.manual_seed(0)
        n_kv = max(4, H * W // 2)
        kv = clamp_to_grid(
            torch.rand(1, n_kv, 2, device="cuda") * max(H, W), H, W)

        got = dense_top4_knn(kv, H=H, W=W, BLOCK_P=block)
        want = reference_dense_top4_knn(kv, H=H, W=W)
        # Ties may resolve to different indices; the distances may not differ.
        torch.testing.assert_close(self._distances(got, kv, H, W),
                                   self._distances(want, kv, H, W),
                                   rtol=0, atol=1e-4)

    @pytest.mark.parametrize("block", [16, 32, 64, 128, 256])
    def test_block_size_never_changes_the_result(self, block):
        """Block size is a speed knob, so a production grid must be invariant."""
        from affmae.ops.deform_attn_triton import dense_top4_knn
        from affmae.ops.knn import clamp_to_grid

        grid = 64
        torch.manual_seed(0)
        kv = clamp_to_grid(
            torch.rand(1, grid * grid, 2, device="cuda") * (grid - 1), grid, grid)
        baseline = dense_top4_knn(kv, H=grid, W=grid, BLOCK_P=32)
        got = dense_top4_knn(kv, H=grid, W=grid, BLOCK_P=block)
        torch.testing.assert_close(
            self._distances(got, kv, grid, grid),
            self._distances(baseline, kv, grid, grid), rtol=0, atol=1e-4)


class TestCapabilityProbesNotIsCuda:
    """`is_cuda` is the wrong predicate for anything backend-specific.

    A ROCm tensor reports ``is_cuda == True`` and ``device.type == "cuda"``, so
    ``is_cuda`` cannot tell NVIDIA from AMD; and it says nothing about whether
    Triton is installed or has a live backend. Three dispatch sites used it:
    CachedKNN, the generic KNN's Triton gate, and the KeOps gate -- the last of
    which would have routed ROCm into a backend KeOps does not have.
    """

    @CUDA
    def test_cached_knn_respects_the_triton_probe(self, monkeypatch):
        import affmae.ops.dispatch as dispatch
        import affmae.ops.knn as knn_module

        picked = []
        real = knn_module.reference_dense_top4_knn
        monkeypatch.setattr(knn_module, "reference_dense_top4_knn",
                            lambda *a, **k: picked.append("reference") or real(*a, **k))
        monkeypatch.setattr(dispatch, "can_use_triton", lambda *a: False)

        positions = torch.rand(1, 64, 2, device="cuda") * 15
        knn_module.CachedKNN(grid_h=8, grid_w=8)(positions)
        assert picked == ["reference"], (
            "a CUDA tensor with no usable Triton must take the torch path")

    def test_keops_probe_rejects_a_rocm_build(self, monkeypatch):
        """KeOps has no ROCm backend, and a ROCm tensor looks like CUDA."""
        import affmae.ops.dispatch as dispatch

        monkeypatch.setattr(dispatch, "is_rocm_build", lambda: True)
        assert dispatch.can_use_keops(torch.zeros(2, 2)) is False

    def test_keops_probe_rejects_unsupported_device_types(self):
        """Anything that is not CPU or NVIDIA CUDA must be refused."""
        import importlib.util

        import affmae.ops.dispatch as dispatch

        class FakeTensor:
            def __init__(self, kind):
                self.device = torch.device(kind)

        assert dispatch.can_use_keops(FakeTensor("meta")) is False
        expected = (importlib.util.find_spec("pykeops") is not None
                    and not dispatch.is_rocm_build())
        assert dispatch.can_use_keops(torch.zeros(2, 2)) is expected

    def test_keops_probe_tolerates_none(self):
        import importlib.util

        import affmae.ops.dispatch as dispatch

        expected = (importlib.util.find_spec("pykeops") is not None
                    and not dispatch.is_rocm_build())
        assert dispatch.can_use_keops(torch.zeros(2, 2), None) is expected

    @CUDA
    def test_knn_keops_falls_back_correctly_on_every_branch(self, monkeypatch):
        """Each rung of keops -> triton -> cdist must give the same answer."""
        import affmae.ops.dispatch as dispatch
        from affmae.utils.geometry import knn_keops

        torch.manual_seed(0)
        query = torch.rand(2, 48, 2, device="cuda")
        database = torch.rand(2, 192, 2, device="cuda")
        distances = torch.cdist(query, database)
        want_idx = distances.topk(4, dim=-1, largest=False).indices
        want_dist = distances.gather(-1, want_idx)

        def agrees():
            idx = knn_keops(query, database, 4)
            assert (idx.sort(-1).values == want_idx.sort(-1).values).all()
            _, dist = knn_keops(query, database, 4, return_dist=True)
            torch.testing.assert_close(dist.sort(-1).values,
                                       want_dist.sort(-1).values,
                                       rtol=0, atol=1e-4)

        agrees()                                            # keops
        monkeypatch.setattr(dispatch, "can_use_keops", lambda *a: False)
        agrees()                                            # triton, as on ROCm
        monkeypatch.setattr(dispatch, "can_use_triton", lambda *a: False)
        agrees()                                            # cdist, as on MPS


class TestRocmStageClamp:
    """The ROCm single-stage restriction has to apply to every kernel.

    NUM_STAGES_OPTIONS existed to encode it but was consumed by one of six
    Triton modules, while the others hardcoded 2-4 stages -- and even that one
    module's triton.heuristics bypassed it. Untestable on this hardware, so what
    is pinned is that every stage choice now routes through the clamp.
    """

    def test_clamp_is_a_no_op_on_cuda(self):
        from affmae.ops.dispatch import clamp_num_stages

        assert [clamp_num_stages(s) for s in (1, 2, 3, 4)] == [1, 2, 3, 4]

    def test_clamp_collapses_to_one_on_rocm(self, monkeypatch):
        import affmae.ops.dispatch as dispatch

        monkeypatch.setattr(dispatch, "NUM_STAGES_OPTIONS", [1])
        assert [dispatch.clamp_num_stages(s) for s in (1, 2, 4, 9)] == [1, 1, 1, 1]

    def test_clamp_never_returns_zero(self):
        from affmae.ops.dispatch import clamp_num_stages

        assert clamp_num_stages(0) == 1
        assert clamp_num_stages(-3) == 1

    def test_no_kernel_module_hardcodes_a_stage_count(self):
        """A new unclamped literal would silently reintroduce the ROCm hazard."""
        import re
        from pathlib import Path

        ops = Path(__file__).resolve().parents[1] / "affmae" / "ops"
        offenders = []
        for path in sorted(ops.glob("*triton*.py")) + [ops / "weighted_features.py"]:
            depth, in_signature = 0, False
            for number, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("def ") or stripped.startswith("async def "):
                    in_signature = True
                    depth = 0
                # A default inside a def signature is fine: the launcher clamps
                # it in the body. Only a literal in a *call* reaches a kernel.
                flag = (not stripped.startswith("#")
                        and "clamp_num_stages" not in line
                        and re.search(r"num_stages\s*=\s*\d", line)
                        and not in_signature)
                if flag:
                    offenders.append(f"{path.name}:{number}: {stripped[:70]}")
                depth += line.count("(") - line.count(")")
                if in_signature and depth <= 0 and "(" in line or (
                        in_signature and line.rstrip().endswith(":")):
                    in_signature = depth > 0
        assert offenders == [], (
            "unclamped num_stages reaching a kernel:\n  " + "\n  ".join(offenders))
