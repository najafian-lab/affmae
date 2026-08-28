"""ClusterAttention module-level tests.

Replaces ``test_forward.py`` / ``test_backward.py`` / ``test_integration.py``,
which were 751 lines and 12 tests written when the Triton and CUDA
implementations were two separate classes in a ``cluster.*`` package. They are
now one class selected by ``backend``, which made four of those tests
tautological (comparing a class's parameter count to its own) and two were
already commented out.

Split by what each actually needs:

* :class:`TestClusterAttention` — invariants of the shipped Triton path. Needs
  only CUDA, so it runs everywhere.
* :class:`TestCudaReferenceParity` — the optional differential against the
  compiled CLUSTEN kernels. Skips unless the extension is built.

Kernel-level numerics live in ``test_kernels.py``, which checks Triton against
pure-PyTorch references and needs no optional dependency.
"""

import pytest
import torch

from conftest import (
    TEST_CONFIGS,
    compare_tensor,
    create_test_data,
    make_cluster_attention_pair,
    requires_clusten,
    requires_cuda,
)
from affmae.layers.attention import ClusterAttention

pytestmark = requires_cuda

# The 'large' entry in TEST_CONFIGS is B=128, N=1024 — minutes of runtime for no
# extra coverage over 'medium'. Kept available via TEST_CONFIGS for perf work.
FAST_CONFIGS = ["small", "medium"]


def _forward(module, data):
    """Run one local-attention forward pass from a create_test_data dict."""
    return module(
        feat=data["feat"],
        member_idx=data["member_idx"],
        cluster_mask=data["cluster_mask"],
        pe_idx=data["pe_idx"],
        global_attn=False,
    )


@pytest.fixture(params=FAST_CONFIGS)
def cfg(request):
    return TEST_CONFIGS[request.param]


class TestClusterAttention:
    """Invariants of the Triton path that ships by default."""

    def test_output_shape_matches_input(self, cfg):
        device = "cuda"
        data = create_test_data(cfg["B"], cfg["N"], cfg["C"], cfg["H"], cfg["M"],
                                device, torch.float32)
        module = ClusterAttention(dim=cfg["C"], num_heads=cfg["H"]).to(device).eval()
        out = _forward(module, data)
        assert out.shape == (cfg["B"], cfg["N"], cfg["C"])

    def test_output_is_finite(self, cfg):
        device = "cuda"
        data = create_test_data(cfg["B"], cfg["N"], cfg["C"], cfg["H"], cfg["M"],
                                device, torch.float32)
        module = ClusterAttention(dim=cfg["C"], num_heads=cfg["H"]).to(device).eval()
        assert torch.isfinite(_forward(module, data)).all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_runs_under_autocast(self, dtype):
        """Both precisions are used in training; fp16 is the default."""
        device, c = "cuda", TEST_CONFIGS["small"]
        data = create_test_data(c["B"], c["N"], c["C"], c["H"], c["M"], device, dtype)
        module = ClusterAttention(dim=c["C"], num_heads=c["H"]).to(device).eval()
        with torch.amp.autocast("cuda", dtype=dtype):
            out = _forward(module, data)
        assert torch.isfinite(out).all()

    def test_gradients_reach_every_parameter(self, cfg):
        """A parameter with no gradient would also break DDP without
        ``find_unused_parameters=True``."""
        device = "cuda"
        data = create_test_data(cfg["B"], cfg["N"], cfg["C"], cfg["H"], cfg["M"],
                                device, torch.float32)
        module = ClusterAttention(dim=cfg["C"], num_heads=cfg["H"]).to(device).train()

        _forward(module, data).sum().backward()

        missing = [n for n, p in module.named_parameters()
                   if p.requires_grad and p.grad is None]
        assert not missing, f"no gradient for: {missing}"

    def test_gradients_are_finite(self, cfg):
        device = "cuda"
        data = create_test_data(cfg["B"], cfg["N"], cfg["C"], cfg["H"], cfg["M"],
                                device, torch.float32)
        module = ClusterAttention(dim=cfg["C"], num_heads=cfg["H"]).to(device).train()
        _forward(module, data).sum().backward()
        for name, p in module.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"

    def test_rejects_indivisible_head_count(self):
        with pytest.raises(AssertionError):
            ClusterAttention(dim=30, num_heads=4)


@requires_clusten
class TestCudaReferenceParity:
    """Triton vs the compiled CLUSTEN CUDA kernels.

    This is the independent-oracle check; ``affmae/ops/cuda_ext/README.md``
    covers building the extension.
    """

    def test_forward_matches(self, cfg, tolerance):
        device = "cuda"
        data = create_test_data(cfg["B"], cfg["N"], cfg["C"], cfg["H"], cfg["M"],
                                device, torch.float32)
        reference, triton = make_cluster_attention_pair(cfg["C"], cfg["H"], device)

        with torch.amp.autocast("cuda", dtype=torch.float32):
            got = _forward(triton, data)
            want = _forward(reference, data)

        compare_tensor(want, got, tol=tolerance["forward_pass"],
                       name=f"forward parity [{cfg['name']}]")

    def test_backward_matches(self, cfg, tolerance):
        device = "cuda"
        data = create_test_data(cfg["B"], cfg["N"], cfg["C"], cfg["H"], cfg["M"],
                                device, torch.float32)
        reference, triton = make_cluster_attention_pair(cfg["C"], cfg["H"], device)
        reference.train()
        triton.train()

        for module in (reference, triton):
            _forward(module, data).sum().backward()

        for (name, ref_p), (_, tri_p) in zip(reference.named_parameters(),
                                             triton.named_parameters()):
            if ref_p.grad is None:
                continue
            compare_tensor(ref_p.grad, tri_p.grad,
                           tol=tolerance["backward_pass"],
                           name=f"grad parity {name} [{cfg['name']}]")
