"""The pure-PyTorch path must agree with the Triton kernels, and run off CUDA.

Non-CUDA support is only worth having if the fallbacks are *right*. Each core
operator is checked twice:

1. **Parity** — Triton vs pure PyTorch on the same inputs, on CUDA, within
   tolerance. Skipped without a GPU.
2. **Portability** — the PyTorch path alone on CPU, which is where the fallbacks
   actually get used.

Plus an end-to-end model forward on CPU, run in a subprocess with CUDA hidden.
That last one is the real test: before this, ``import affmae.models`` itself
failed on a CPU-only host because ``kernels/util.py`` probed the Triton driver at
module scope.

Tolerances are ~1e-3 for neighborhood attention because the Triton kernel writes
fp16 output, and ~1e-5 elsewhere.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from affmae.ops.deform_attn_torch import deform_point_attn_torch, msdetrpc_reduce
from affmae.ops.nbhd_attn_torch import neighborhood_attention

REPO = Path(__file__).resolve().parents[1]
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="requires a CUDA device")


def _grid(n, device):
    ys, xs = torch.meshgrid(torch.arange(n, device=device),
                            torch.arange(n, device=device), indexing="ij")
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).float()


class TestNeighborhoodAttention:
    @staticmethod
    def _inputs(device, b=2, h=4, n=64, d=32, m=16, seed=0):
        g = torch.Generator(device=device).manual_seed(seed)
        return dict(
            q=torch.randn(b, h, n, d, device=device, generator=g),
            k=torch.randn(b, h, n, d, device=device, generator=g),
            v=torch.randn(b, h, n, d, device=device, generator=g),
            member=torch.randint(0, n, (b, n, m), device=device, generator=g,
                                 dtype=torch.int32),
            bias=torch.randn(b, h, n, m, device=device, generator=g),
            blank_k=torch.randn(h, d, device=device, generator=g),
            blank_v=torch.randn(h, d, device=device, generator=g),
            scale=float(d ** -0.5),
        )

    @requires_cuda
    @pytest.mark.parametrize("shape", [(1, 2, 64, 32, 16), (2, 4, 128, 32, 32)])
    def test_matches_triton(self, shape):
        from affmae.ops.nbhd_attn_triton import (
            FlashNeighborhoodAttentionFunction as Triton,
        )

        b, h, n, d, m = shape
        i = self._inputs("cuda", b, h, n, d, m)
        expected = Triton.apply(i["q"], i["k"], i["v"], i["member"], i["bias"],
                                None, i["blank_k"], i["blank_v"], i["scale"])
        got = neighborhood_attention(
            i["q"], i["k"], i["v"], i["member"], i["scale"], bias=i["bias"],
            mask=None, blank_k=i["blank_k"], blank_v=i["blank_v"])
        torch.testing.assert_close(got.float(), expected.float(),
                                   rtol=2e-2, atol=2e-3)

    @requires_cuda
    def test_matches_triton_with_a_mask(self):
        from affmae.ops.nbhd_attn_triton import (
            FlashNeighborhoodAttentionFunction as Triton,
        )

        i = self._inputs("cuda", seed=3)
        g = torch.Generator(device="cuda").manual_seed(4)
        mask = torch.rand(2, 1, 64, 16, device="cuda", generator=g) > 0.3
        expected = Triton.apply(i["q"], i["k"], i["v"], i["member"], i["bias"],
                                mask, i["blank_k"], i["blank_v"], i["scale"])
        got = neighborhood_attention(
            i["q"], i["k"], i["v"], i["member"], i["scale"], bias=i["bias"],
            mask=mask, blank_k=i["blank_k"], blank_v=i["blank_v"])
        torch.testing.assert_close(got.float(), expected.float(),
                                   rtol=2e-2, atol=2e-3)

    def test_runs_on_cpu(self):
        i = self._inputs("cpu")
        out = neighborhood_attention(
            i["q"], i["k"], i["v"], i["member"], i["scale"], bias=i["bias"],
            blank_k=i["blank_k"], blank_v=i["blank_v"])
        assert out.shape == i["q"].shape
        assert torch.isfinite(out).all()

    def test_chunking_does_not_change_the_result(self):
        """Chunking bounds memory; it must not be observable in the output."""
        i = self._inputs("cpu", n=64)
        kwargs = dict(bias=i["bias"], blank_k=i["blank_k"], blank_v=i["blank_v"])
        small = neighborhood_attention(i["q"], i["k"], i["v"], i["member"],
                                       i["scale"], chunk_size=7, **kwargs)
        whole = neighborhood_attention(i["q"], i["k"], i["v"], i["member"],
                                       i["scale"], chunk_size=10_000, **kwargs)
        torch.testing.assert_close(small, whole, rtol=0, atol=0)

    def test_gradients_flow(self):
        """Autograd is the whole reason there is no hand-written backward."""
        i = self._inputs("cpu")
        q, k, v = (t.clone().requires_grad_(True)
                   for t in (i["q"], i["k"], i["v"]))
        out = neighborhood_attention(q, k, v, i["member"], i["scale"],
                                     bias=i["bias"], blank_k=i["blank_k"],
                                     blank_v=i["blank_v"])
        out.sum().backward()
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            assert tensor.grad is not None, f"no gradient for {name}"
            assert torch.isfinite(tensor.grad).all()

    def test_rejects_mismatched_shapes(self):
        i = self._inputs("cpu")
        with pytest.raises(ValueError, match="does not match"):
            neighborhood_attention(i["q"], i["k"], i["v"],
                                   i["member"][:, :8, :], i["scale"])


class TestMsdetrpcReduce:
    @staticmethod
    def _inputs(device, seed=0):
        g = torch.Generator(device=device).manual_seed(seed)
        b, n, m, k, n_val, c = 2, 64, 4, 4, 128, 32
        return (
            torch.randint(0, n_val, (b, n, m, k), device=device, generator=g),
            torch.rand(b, n, m, k, device=device, generator=g),
            torch.softmax(torch.randn(b, n, m, device=device, generator=g), -1),
            torch.randn(b, n_val, c, device=device, generator=g),
        )

    @requires_cuda
    def test_matches_triton(self):
        from affmae.ops.deform_reduce_triton import MSDETRPCFunction

        idx, weight, attn, val = self._inputs("cuda")
        expected = MSDETRPCFunction.apply(idx, weight, attn, val)
        got = msdetrpc_reduce(idx, weight, attn, val)
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)

    def test_runs_on_cpu(self):
        idx, weight, attn, val = self._inputs("cpu")
        out = msdetrpc_reduce(idx, weight, attn, val)
        assert out.shape == (2, 64, 32)
        assert torch.isfinite(out).all()

    def test_rejects_mismatched_shapes(self):
        idx, weight, attn, val = self._inputs("cpu")
        with pytest.raises(ValueError, match="must match"):
            msdetrpc_reduce(idx, weight[..., :2], attn, val)


class TestDeformablePointAttention:
    @staticmethod
    def _inputs(device, grid=8, heads=2, points=4, channels=16, seed=5):
        from affmae.ops.knn_torch import reference_dense_top4_knn

        coords = _grid(grid, device)
        n = coords.shape[0]
        pos = coords.unsqueeze(0).contiguous()
        nn4 = reference_dense_top4_knn(
            pos.round().clamp(0, grid - 1).to(torch.int32), H=grid, W=grid)
        g = torch.Generator(device=device).manual_seed(seed)
        return dict(
            query_pos=pos, kv_pos=pos, nn4=nn4, grid=grid,
            offsets=torch.randn(1, n, heads, points, 2, device=device,
                                generator=g) * 0.5,
            logits=torch.randn(1, n, heads, points, device=device, generator=g),
            values=torch.randn(1, n, heads, channels, device=device, generator=g),
            tau=torch.tensor(3.0, device=device),
        )

    @requires_cuda
    @pytest.mark.parametrize("grid", [8, 16])
    def test_matches_triton(self, grid):
        from affmae.ops.deform_attn_triton import deform_point_attn

        i = self._inputs("cuda", grid=grid)
        expected = deform_point_attn(
            query_pos=i["query_pos"], kv_pos=i["kv_pos"],
            sampling_offsets=i["offsets"], attn_logits=i["logits"],
            values=i["values"], tau=i["tau"], nn4_idx=i["nn4"],
            grid_h=grid, grid_w=grid, backend="csr_knn_cached")
        got = deform_point_attn_torch(
            i["query_pos"], i["kv_pos"], i["offsets"], i["logits"],
            i["values"], i["tau"], i["nn4"], grid, grid)
        torch.testing.assert_close(got.float(), expected.float(),
                                   rtol=1e-3, atol=1e-4)

    def test_runs_on_cpu(self):
        i = self._inputs("cpu")
        out = deform_point_attn_torch(
            i["query_pos"], i["kv_pos"], i["offsets"], i["logits"],
            i["values"], i["tau"], i["nn4"], i["grid"], i["grid"])
        assert out.shape == i["values"].shape
        assert torch.isfinite(out).all()

    def test_gradients_flow(self):
        i = self._inputs("cpu")
        offs, logits, vals = (t.clone().requires_grad_(True) for t in
                              (i["offsets"], i["logits"], i["values"]))
        deform_point_attn_torch(i["query_pos"], i["kv_pos"], offs, logits,
                                vals, i["tau"], i["nn4"], i["grid"],
                                i["grid"]).sum().backward()
        for name, tensor in (("offsets", offs), ("logits", logits),
                             ("values", vals)):
            assert tensor.grad is not None, f"no gradient for {name}"


class TestDenseKnnReference:
    def test_reference_avoids_uint16(self):
        """MPS has no meaningful uint16 support, and CUDA cannot reduce over it."""
        from affmae.ops.knn_torch import reference_dense_top4_knn

        coords = _grid(8, "cpu").unsqueeze(0).to(torch.int32)
        out = reference_dense_top4_knn(coords, H=8, W=8)
        assert out.dtype == torch.int32

    @requires_cuda
    def test_reference_neighbour_distances_match_triton(self):
        """Indices may tie-break differently; the distances may not."""
        from affmae.ops.deform_attn_triton import dense_top4_knn
        from affmae.ops.knn_torch import reference_dense_top4_knn

        grid = 8
        coords = _grid(grid, "cuda").unsqueeze(0).to(torch.int32)
        triton = dense_top4_knn(coords, H=grid, W=grid)
        reference = reference_dense_top4_knn(coords, H=grid, W=grid)

        pos = coords[0].float()
        cell_ids = torch.arange(grid * grid, device="cuda")
        cells = torch.stack([(cell_ids % grid).float(),
                             (cell_ids // grid).float()], dim=-1)

        def distances(table):
            neighbours = pos[table[0].long()]
            return ((neighbours - cells[:, None, :]) ** 2).sum(-1).sort(-1).values

        torch.testing.assert_close(distances(triton), distances(reference),
                                   rtol=0, atol=0)


@pytest.mark.slow
class TestCpuOnlyHost:
    """Simulates a machine with no GPU, by hiding CUDA from a subprocess."""

    @staticmethod
    def _run(body):
        code = textwrap.dedent(f"""
            import sys; sys.path.insert(0, {str(REPO)!r})
            import logging; logging.disable(logging.INFO)
            import torch
            {textwrap.indent(textwrap.dedent(body), ' ' * 12).lstrip()}
        """)
        return subprocess.run(
            [sys.executable, "-c", code], cwd=REPO, capture_output=True,
            text=True, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"})

    def test_model_stack_imports_without_a_gpu(self):
        """kernels/util.py probed the Triton driver at module scope, so this
        used to fail before a single tensor existed."""
        result = self._run("""
            assert not torch.cuda.is_available()
            from affmae.ops.dispatch import has_triton_backend
            assert not has_triton_backend()
            import affmae.layers.attention          # noqa: F401
            import affmae.models.aff_segmentation   # noqa: F401
            print("OK")
        """)
        assert result.returncode == 0, result.stderr[-2500:]
        assert "OK" in result.stdout

    def test_fused_default_backend_forwards_on_cpu(self):
        """The default ``csr_knn_cached`` path used to be CUDA-only.

        ``_get_nn4`` built the neighbour table with the Triton kernel *before*
        the ``can_use_triton`` guard, so the advertised torch fallback was
        unreachable. ``test_full_forward_on_cpu`` did not catch it because the
        config it loads selects ``unfused``.
        """
        result = self._run("""
            from affmae.layers.attention import DeformableSelfAttention
            m = DeformableSelfAttention(
                d_model=64, n_heads=4, n_points=4, grid_h=16, grid_w=16,
                deform_backend="csr_knn_cached").eval()
            with torch.no_grad():
                out = m(torch.randn(2, 32, 64), torch.rand(2, 32, 2) * 15)
            assert out.shape == (2, 32, 64), out.shape
            assert torch.isfinite(out).all()
            print("OK")
        """)
        assert result.returncode == 0, result.stderr[-2500:]
        assert "OK" in result.stdout

    def test_imports_and_runs_without_triton_installed(self):
        """Triton is an optional extra, so importing must not require it.

        The kernel modules decorate with ``@triton.jit`` at module scope, so a
        bare ``import triton`` there made ``affmae.layers.attention``
        unimportable on a torch-only install.
        """
        result = self._run("""
            import importlib.abc
            class Blocker(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path=None, target=None):
                    if name == "triton" or name.startswith("triton."):
                        raise ImportError(f"No module named {name!r}")
            sys.meta_path.insert(0, Blocker())
            for m in [m for m in sys.modules if m.startswith("triton")]:
                del sys.modules[m]

            from affmae.ops.dispatch import HAS_TRITON
            assert HAS_TRITON is False
            from affmae.layers.attention import DeformableSelfAttention
            m = DeformableSelfAttention(
                d_model=64, n_heads=4, n_points=4, grid_h=16, grid_w=16,
                deform_backend="csr_knn_cached").eval()
            with torch.no_grad():
                out = m(torch.randn(1, 16, 64), torch.rand(1, 16, 2) * 15)
            assert torch.isfinite(out).all()
            print("OK")
        """)
        assert result.returncode == 0, result.stderr[-2500:]
        assert "OK" in result.stdout

    def test_full_forward_on_cpu(self):
        result = self._run("""
            from affmae.config import load_config
            from affmae.models.registry import get_model_spec
            cfg = load_config("configs/aff_base_finetune_512_fpw.yaml")
            cfg.img_size = 256   # valid geometry, bearable on CPU
            model = get_model_spec(cfg.model_type).build_segmentation(cfg).eval()
            x = torch.randn(1, cfg.in_channels, cfg.img_size, cfg.img_size)
            with torch.no_grad():
                out = model(x)
            assert len(out) == 3, out
            assert all(o.shape[-1] == cfg.img_size for o in out)
            assert all(torch.isfinite(o).all() for o in out)
            print("OK")
        """)
        assert result.returncode == 0, result.stderr[-2500:]
        assert "OK" in result.stdout
