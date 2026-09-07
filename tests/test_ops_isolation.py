"""The operators in `affmae.ops` must be usable on their own.

"On their own" means three concrete things, each tested here:

1. **Import-light** — torch and nothing else. Triton is resolved lazily so a
   torch-only install works.
2. **Copyable** — the directory dropped into an unrelated project, with no
   `affmae` package present, still imports. This is the property people actually
   want when they say "I'd like to reuse your downsampling".
3. **Stateless** — no module-level or class-level mutable state, so two
   instances in one process cannot interfere.

Before this, lifting either component out was impossible: `affmae/utils/geometry.py`
imported `weighted_gather` at module scope (pulling Triton) solely for a dead
function, and the KNN cache lived in class attributes on a shared base class.
"""

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
OPS = REPO / "affmae" / "ops"


def _run(code, cwd, extra_path=None):
    env_setup = f"import sys; sys.path.insert(0, {str(extra_path)!r})\n" if extra_path else ""
    result = subprocess.run([sys.executable, "-c", env_setup + textwrap.dedent(code)],
                            cwd=cwd, capture_output=True, text=True)
    return result


class TestImportLight:
    @pytest.mark.parametrize("module", [
        "affmae.ops",
        "affmae.ops.clustering",
        "affmae.ops.cache",
        "affmae.ops.knn",
    ])
    def test_imports_nothing_heavy(self, module):
        heavy = ["matplotlib", "cv2", "sklearn", "PIL", "timm", "fvcore",
                 "triton", "scipy", "skimage"]
        result = _run(f"""
            import json, sys
            sys.path.insert(0, {str(REPO)!r})
            import {module}
            print(json.dumps([h for h in {heavy!r} if h in sys.modules]))
        """, cwd=REPO)
        assert result.returncode == 0, result.stderr[-2000:]
        leaked = json.loads(result.stdout.strip().splitlines()[-1])
        assert leaked == [], f"{module} pulls in {leaked}"

    def test_clustering_needs_only_math_and_torch(self):
        """The clustering algorithm is the one piece people ask to reuse.

        Parses the AST rather than grepping: the module docstring contains a
        usage example that mentions ``affmae``, which is documentation, not a
        dependency.
        """
        import ast

        tree = ast.parse((OPS / "clustering.py").read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported <= {"math", "torch"}, (
            f"clustering.py imports {sorted(imported)}; only math and torch are "
            f"allowed so the file stays standalone")


class TestCopyable:
    def test_ops_directory_works_when_copied_out(self, tmp_path):
        """Copy affmae/ops/ elsewhere, as `myops`, with no affmae installed."""
        target = tmp_path / "myops"
        shutil.copytree(OPS, target, ignore=shutil.ignore_patterns("__pycache__"))

        result = _run("""
            import torch
            from myops.clustering import SpaceFillingCluster, space_filling_cluster
            from myops.cache import TensorCache, cache_scope, CachePolicy

            pos = torch.stack(torch.meshgrid(
                torch.arange(8), torch.arange(8), indexing="ij"), -1
            ).reshape(1, -1, 2).float()

            cluster = SpaceFillingCluster(cluster_size=8)
            out = cluster(pos, h=8, w=8)
            print("clustered", len(out), tuple(out[2].shape))

            with cache_scope(name="x") as c:
                v = c.get_or_compute(("k",), lambda: 7)
                print("cached", v, c.hits, c.misses)
        """, cwd=tmp_path, extra_path=tmp_path)
        assert result.returncode == 0, (
            f"copied ops failed to work standalone:\n{result.stderr[-3000:]}")
        assert "clustered" in result.stdout
        assert "cached 7 0 1" in result.stdout


class TestStateless:
    def test_no_module_level_mutable_state(self):
        """A module-level dict/list is how two models end up sharing a cache.

        `cache.py` owns exactly two: the warn-once set and the scope holder,
        both documented. Nothing else may add one.
        """
        import ast

        # __all__ is a declaration, not state. The two in cache.py are the
        # documented exceptions: a warn-once set and the scope holder.
        allowed = {"__all__"} | {"_WARNED", "_ACTIVE"}
        offenders = []
        for path in sorted(OPS.glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                if not isinstance(node.value, (ast.Dict, ast.List, ast.Set)):
                    continue
                for target in node.targets:
                    name = getattr(target, "id", None)
                    if name and name not in allowed:
                        offenders.append(f"{path.name}:{name}")
        assert offenders == [], f"module-level mutable state: {offenders}"

    def test_two_cluster_instances_do_not_interfere(self):
        from affmae.ops import SpaceFillingCluster

        pos = torch.stack(torch.meshgrid(
            torch.arange(8), torch.arange(8), indexing="ij"), -1
        ).reshape(1, -1, 2).float()

        a = SpaceFillingCluster(cluster_size=8)
        b = SpaceFillingCluster(cluster_size=16)

        _, _, member_a, _, _ = a(pos, h=8, w=8)
        _, _, member_b, _, _ = b(pos, h=8, w=8)
        # Re-running a must be unaffected by b having run.
        _, _, member_a2, _, _ = a(pos, h=8, w=8)

        assert member_a.shape[-1] == 8
        assert member_b.shape[-1] == 16
        torch.testing.assert_close(member_a, member_a2, rtol=0, atol=0)

    def test_clustering_holds_no_parameters_or_buffers(self):
        from affmae.ops import SpaceFillingCluster

        module = SpaceFillingCluster(cluster_size=8)
        assert list(module.parameters()) == []
        assert list(module.buffers()) == []


class TestFunctionalEquivalence:
    def test_module_matches_the_function(self):
        """The nn.Module wrapper must not change the algorithm."""
        from affmae.ops import SpaceFillingCluster, space_filling_cluster

        pos = torch.stack(torch.meshgrid(
            torch.arange(16), torch.arange(16), indexing="ij"), -1
        ).reshape(1, -1, 2).float()

        via_module = SpaceFillingCluster(cluster_size=8, sf_type="hilbert")(pos, 16, 16)
        via_fn = space_filling_cluster(pos, 8, 16, 16, sf_type="hilbert")

        assert len(via_module) == len(via_fn)
        for lhs, rhs in zip(via_module, via_fn):
            if isinstance(lhs, torch.Tensor):
                torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)

    def test_geometry_still_re_exports_for_compatibility(self):
        """Existing imports from affmae.utils.geometry must keep working."""
        from affmae.ops.clustering import space_filling_cluster as from_ops
        from affmae.utils.geometry import space_filling_cluster as from_geometry

        assert from_ops is from_geometry

    @pytest.mark.parametrize("sf_type", ["", "peano", "hilbert"])
    def test_every_curve_produces_balanced_clusters(self, sf_type):
        from affmae.ops import SpaceFillingCluster

        grid, size = 16, 8
        pos = torch.stack(torch.meshgrid(
            torch.arange(grid), torch.arange(grid), indexing="ij"), -1
        ).reshape(1, -1, 2).float()

        _, centers, member_idx, mask, _ = SpaceFillingCluster(
            cluster_size=size, sf_type=sf_type)(pos, grid, grid)

        n_clusters = (grid * grid) // size
        assert member_idx.shape == (1, n_clusters, size)
        assert centers.shape == (1, n_clusters, 2)
        # Evenly divisible, so nothing is padded.
        assert mask is None or bool(mask.all())
        # Every token assigned exactly once.
        assert sorted(member_idx.flatten().tolist()) == list(range(grid * grid))
