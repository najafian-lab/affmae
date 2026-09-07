"""Two models in one process must not interfere.

Five places used to break this, none fixed on any earlier branch:

1. ``attention.pre_table_fp16/fp32`` — rebound to *this instance's* buffer on
   every forward via ``global``. After one forward on GPU the module-level name
   pointed at a CUDA tensor, so a model constructed *afterwards* cloned that
   tensor — on whatever device the first model happened to live on — before
   ``.to(device)`` was ever called. It also kept the first model's buffer alive
   forever.
2. ``aff.pre_table`` — guarded on ``is_cuda`` rather than *which* device, and
   not a registered buffer, so ``model.to(device)`` could never fix it. A second
   model on another device fed a ``cuda:0`` tensor into a ``cuda:1`` Linear.
3. ``_decoder_self_cache_scope_stack`` — a class attribute, so one model could
   read another's scope.
4. ``_decoder_self_nn4_cache`` — a class attribute keyed partly on
   ``data_ptr()``, cleaned by scope-key prefix match.
5. ``BasicLayer.nbhd_size`` — overwritten during forward, so one short sequence
   permanently shrank the layer.
"""

import ast
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]

# Files that carried the globals. Auditing the source is the only way to catch a
# reintroduction: a `global` re-binding is invisible until two instances collide.
AUDITED = [
    "affmae/layers/attention.py",
    "affmae/layers/aff.py",
    "affmae/layers/decoder.py",
]


class TestNoGlobalRebinding:
    @pytest.mark.parametrize("relpath", AUDITED)
    def test_no_global_statements(self, relpath):
        tree = ast.parse((REPO / relpath).read_text())
        offenders = [
            f"{relpath}:{node.lineno} global {', '.join(node.names)}"
            for node in ast.walk(tree) if isinstance(node, ast.Global)
        ]
        assert offenders == [], (
            f"`global` rebinding is how two models end up sharing state: {offenders}")

    @pytest.mark.parametrize("relpath", AUDITED)
    def test_no_class_level_mutable_attributes(self, relpath):
        """A class-level dict or list is shared by every instance."""
        tree = ast.parse((REPO / relpath).read_text())
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                if isinstance(stmt.value, (ast.Dict, ast.List, ast.Set)):
                    names = [getattr(t, "id", "?") for t in stmt.targets]
                    offenders.append(f"{node.name}.{names}")
        assert offenders == [], f"class-level mutable state: {offenders}"


class TestPositionTablesAreBuffers:
    """A buffer moves with `.to(device)` and is per-instance. A global is not."""

    def test_cluster_attention_tables_are_buffers(self):
        from affmae.layers.attention import ClusterAttention

        module = ClusterAttention(dim=32, num_heads=2)
        names = dict(module.named_buffers())
        assert "pre_table_fp32" in names
        assert "pre_table_fp16" in names

    def test_cluster_merging_table_is_a_buffer(self):
        from affmae.layers.aff import ClusterMerging

        names = dict(ClusterMerging(dim=32, out_dim=32).named_buffers())
        assert "pre_table" in names, (
            "pre_table must be a buffer so model.to(device) moves it")

    def test_table_is_not_persisted_so_old_checkpoints_load(self):
        """Adding a persistent buffer would make every prior checkpoint
        report a missing key."""
        from affmae.layers.aff import ClusterMerging

        module = ClusterMerging(dim=32, out_dim=32)
        assert "pre_table" not in module.state_dict()

    def test_two_instances_hold_independent_tables(self):
        from affmae.layers.attention import ClusterAttention

        a = ClusterAttention(dim=32, num_heads=2)
        b = ClusterAttention(dim=32, num_heads=2)
        assert a.pre_table_fp32 is not b.pre_table_fp32
        a.pre_table_fp32.add_(1.0)
        assert not torch.equal(a.pre_table_fp32, b.pre_table_fp32), (
            "mutating one instance's table changed the other's")

    def test_moving_one_instance_leaves_the_other_alone(self):
        """The exact failure mode of the `is_cuda` guard."""
        from affmae.layers.aff import ClusterMerging

        a = ClusterMerging(dim=32, out_dim=32)
        b = ClusterMerging(dim=32, out_dim=32)
        target = "cuda" if torch.cuda.is_available() else "meta"
        a.to(target)

        assert a.pre_table.device.type == target
        assert b.pre_table.device.type == "cpu", (
            "moving one model moved the other's table -- they share storage")

    def test_moving_the_model_moves_the_table(self):
        """The other half: a global could never be moved at all."""
        from affmae.layers.aff import ClusterMerging

        module = ClusterMerging(dim=32, out_dim=32)
        assert module.pre_table.device.type == "cpu"
        target = "cuda" if torch.cuda.is_available() else "meta"
        module.to(target)
        assert module.pre_table.device.type == target


class TestNeighborhoodSizeIsStable:
    def test_forward_does_not_shrink_nbhd_size(self):
        """`self.nbhd_size = ...` in forward made a short sequence permanent.

        Asserted on the source because triggering it needs a full encoder
        forward at two sequence lengths; the assignment is the bug.
        """
        source = (REPO / "affmae/layers/aff.py").read_text()
        tree = ast.parse(source)
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != "forward":
                continue
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Assign):
                    continue
                for target in sub.targets:
                    if (isinstance(target, ast.Attribute)
                            and target.attr == "nbhd_size"
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"):
                        offenders.append(sub.lineno)
        assert offenders == [], (
            f"forward assigns self.nbhd_size at line(s) {offenders}; that "
            f"permanently changes the layer for later calls")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestTwoModelsOnDevice:
    def test_two_models_forward_independently(self):
        """End-to-end: build, move, forward twice, interleaved."""
        from affmae.config import load_config
        from affmae.models.registry import get_model_spec

        cfg = load_config("configs/aff_base_finetune_512_fpw.yaml")
        spec = get_model_spec(cfg.model_type)

        first = spec.build_segmentation(cfg).to("cuda").eval()
        second = spec.build_segmentation(cfg).to("cuda").eval()

        x = torch.randn(1, cfg.in_channels, cfg.img_size, cfg.img_size,
                        device="cuda")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            a1 = first(x)[-1]
            b1 = second(x)[-1]
            a2 = first(x)[-1]   # must be unaffected by `second` having run

        torch.testing.assert_close(a1, a2, rtol=0, atol=0)
        assert a1.shape == b1.shape
