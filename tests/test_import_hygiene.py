"""Importing a layer or a model must not pull in plotting or optional deps.

This is the test that makes the layering rules real rather than aspirational.
Before it existed:

* ``affmae/layers/decoder.py`` imported ``fvcore``, which ``pyproject.toml``
  declares as an **optional** extra — so the model could not be imported at all
  without an optional dependency installed.
* ``affmae/models/aff_mae.py`` and ``aff_segmentation.py`` imported ``matplotlib``
  and ``cv2`` at module scope, so ``from affmae.models... import ...`` pulled the
  whole plotting stack.
* ``affmae/utils/misc.py`` imported ``sklearn``, ``matplotlib`` and ``PIL``, and 11
  modules import ``misc`` — so getting an ``AverageMeter`` cost you sklearn.
* ``affmae/layers/aff.py`` imported ``timm`` for ``DropPath`` alone, which drags in
  PIL. ``DropPath`` is now vendored in ``affmae/layers/drop_path.py``.

Each check runs in a subprocess because ``sys.modules`` is process-global: once
any earlier test imports matplotlib, an in-process assertion is meaningless.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]

# Rendering and analysis deps: never reachable from any layer or model.
RENDER_DEPS = ["matplotlib", "cv2", "sklearn", "seaborn", "pandas", "gradio"]

# Optional extras that must not be import-time requirements.
OPTIONAL_DEPS = ["fvcore"]

# timm (and its transitive PIL / wandb) is a *legitimate* dependency of the ViT
# baseline models: they use timm's reference `PatchEmbed` and `Block`, ~200 lines
# of real architecture whose weights are in the trained ViT checkpoints. Using
# the reference implementation is safer for a baseline than a vendored copy that
# can silently drift. The AFF core is timm-free (DropPath is vendored in
# affmae/layers/drop_path.py), which is what matters for reuse.
ARCH_DEPS = ["timm", "PIL", "wandb"]

# Tier 1: the reusable core. Nothing heavy at all.
CORE_MODULES = [
    "affmae.layers.drop_path",
    "affmae.layers.attention",
    "affmae.layers.decoder",
    "affmae.layers.aff",
    "affmae.ops.deform_attn_triton",
    "affmae.ops.nbhd_attn_triton",
    "affmae.models.aff_mae",
    "affmae.models.aff_segmentation",
    "affmae.models.registry",
    "affmae.models.masking",
    "affmae.models.perlin",
    "affmae.layers.pos_embed",
    "affmae.ops.knn_keops",
    "affmae.utils.misc",
    "affmae.utils.geometry",
    "affmae.utils.dist",
]

# Tier 2: the ViT baselines. May use timm; still no renderers.
BASELINE_MODULES = [
    "affmae.models.vit_mae",
    "affmae.models.vit_fpn_segmentation",
]

FORBIDDEN = RENDER_DEPS + OPTIONAL_DEPS + ARCH_DEPS


def _imported_forbidden(module):
    """Import `module` in a fresh process; return which FORBIDDEN deps loaded."""
    code = textwrap.dedent(f"""
        import json, sys
        sys.path.insert(0, {str(REPO)!r})
        import {module}
        print(json.dumps([d for d in {FORBIDDEN!r} if d in sys.modules]))
    """)
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                            capture_output=True, text=True)
    assert result.returncode == 0, (
        f"importing {module} failed:\n{result.stderr[-2000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_module_pulls_no_heavy_deps(module):
    """The reusable core must import with nothing but torch (+triton)."""
    leaked = _imported_forbidden(module)
    assert leaked == [], (
        f"{module} pulls in {leaked}. Move the import inside the function that "
        f"needs it, or relocate the code to affmae/viz/.")


@pytest.mark.parametrize("module", BASELINE_MODULES)
def test_baseline_module_pulls_no_renderers(module):
    """ViT baselines may use timm, but never a renderer or an optional extra."""
    leaked = _imported_forbidden(module)
    disallowed = [d for d in leaked if d not in ARCH_DEPS]
    assert disallowed == [], f"{module} pulls in {disallowed}"


def test_aff_core_is_timm_free():
    """DropPath was the only reason the AFF encoder needed timm.

    Pinned separately because this is the property that makes affmae/layers
    liftable into another project.
    """
    assert "timm" not in _imported_forbidden("affmae.layers.aff")


def test_renderers_live_outside_models():
    """Models must not carry visualization methods.

    Renderers are free functions under `affmae/viz/`, selected per model by name
    via `ModelSpec.reconstruction_renderer`. A `visualize` method on a model is
    what made the model classes import matplotlib.
    """
    import affmae.models.aff_mae as aff_mae
    import affmae.models.aff_segmentation as aff_seg

    for module in (aff_mae, aff_seg):
        for cls_name in dir(module):
            cls = getattr(module, cls_name)
            if isinstance(cls, type) and issubclass(cls, torch.nn.Module):
                assert not hasattr(cls, "visualize"), f"{cls_name}.visualize"
                assert not hasattr(cls, "visualize_tokens"), f"{cls_name}.visualize_tokens"


def test_visualize_module_is_importable_and_sets_a_backend():
    """The renderer module must not require a display."""
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(REPO)!r})
        import matplotlib
        import affmae.viz.model_figures  # noqa: F401
        print(matplotlib.get_backend())
    """)
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().lower() == "agg"


class TestVendoredDropPath:
    """DropPath replaced timm; it must behave identically."""

    @pytest.mark.parametrize("drop_prob", [0.0, 0.1, 0.5, 0.9])
    def test_matches_timm(self, drop_prob):
        timm_layers = pytest.importorskip("timm.layers")
        from affmae.layers.drop_path import DropPath

        x = torch.randn(64, 7, 5)
        theirs = timm_layers.DropPath(drop_prob).train()
        ours = DropPath(drop_prob).train()

        torch.manual_seed(0)
        expected = theirs(x)
        torch.manual_seed(0)
        got = ours(x)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_eval_is_identity(self):
        from affmae.layers.drop_path import DropPath

        x = torch.randn(8, 4)
        module = DropPath(0.5).eval()
        torch.testing.assert_close(module(x), x, rtol=0, atol=0)

    def test_holds_no_parameters(self):
        """So swapping timm out cannot invalidate a checkpoint."""
        from affmae.layers.drop_path import DropPath

        assert list(DropPath(0.5).parameters()) == []


class TestStatisticsAreALeaf:
    """Reading a normalization constant must not import the training stack.

    ``IMAGE_MEAN``/``IMAGE_STD`` used to live in ``finetune_dataset`` and the
    pretrain pair in ``pretrain_dataset``, so ``affmae.data.preprocess`` -- the
    inference preprocessing chain -- pulled albumentations, scipy, torchvision
    and OpenCV to read two floats, and rendering a pretraining figure pulled
    webdataset. Caught by running inference on a bare PyTorch container.
    """

    HEAVY = ("scipy", "albumentations", "webdataset", "torchvision", "cv2")

    @staticmethod
    def _imported(module):
        code = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(REPO)!r})
            import {module}
            print(",".join(sorted(m for m in sys.modules if "." not in m)))
        """)
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr[-1500:]
        return set(result.stdout.strip().split(","))

    def test_stats_module_imports_nothing(self):
        assert not (self._imported("affmae.data.stats") & set(self.HEAVY))

    def test_preprocess_needs_no_training_dependency(self):
        """This is the inference path; it must work on an inference install."""
        leaked = self._imported("affmae.data.preprocess") & set(self.HEAVY)
        assert leaked == set(), f"affmae.data.preprocess pulls {sorted(leaked)}"

    def test_model_figures_needs_no_dataset_stack(self):
        """A renderer may use matplotlib and OpenCV, but not the data loaders."""
        leaked = self._imported("affmae.viz.model_figures") & {
            "scipy", "albumentations", "webdataset"}
        assert leaked == set(), f"affmae.viz.model_figures pulls {sorted(leaked)}"

    def test_the_two_splits_have_different_statistics(self):
        """Using one where the other belongs shifts every rendered image."""
        from affmae.data import stats

        assert stats.IMAGE_MEAN != stats.PRETRAIN_IMAGE_MEAN
        assert stats.IMAGE_STD != stats.PRETRAIN_IMAGE_STD

    def test_the_loaders_still_re_export_them(self):
        """Existing imports must keep working."""
        from affmae.data.finetune_dataset import IMAGE_MEAN, IMAGE_STD
        from affmae.data.pretrain_dataset import (
            PRETRAIN_IMAGE_MEAN,
            PRETRAIN_IMAGE_STD,
        )
        from affmae.data import stats

        assert (IMAGE_MEAN, IMAGE_STD) == (stats.IMAGE_MEAN, stats.IMAGE_STD)
        assert (PRETRAIN_IMAGE_MEAN, PRETRAIN_IMAGE_STD) == (
            stats.PRETRAIN_IMAGE_MEAN, stats.PRETRAIN_IMAGE_STD)


class TestPackageLayout:
    """Where a module lives should be predictable from what it does.

    The package had four competing organizing principles at once: by abstraction
    layer (ops -> layers -> models), by pipeline stage (data, training, eval), by
    "is it reusable" (utils), and by nothing at all (modules at the package root).
    The first two are observable from the name; the last two are not, and `utils`
    was measurably untrue -- `build_finetune_optimizer` had one importer and it
    was `training/`, `perlin` had one and it was `utils/masking`.
    """

    #: Genuine cross-cutting utilities: process, environment, filesystem. None of
    #: these may know anything about models, stages, or kernels.
    UTILS_ALLOWED = {"__init__", "dist", "env", "paths", "misc", "geometry"}

    #: Modules allowed to sit directly in `affmae/`, each a package-level entry
    #: point rather than an implementation detail.
    ROOT_ALLOWED = {"__init__", "config", "inference", "demo"}

    def test_utils_holds_only_utilities(self):
        found = {p.stem for p in (REPO / "affmae" / "utils").glob("*.py")}
        unexpected = found - self.UTILS_ALLOWED
        assert unexpected == set(), (
            f"{sorted(unexpected)} sits in affmae/utils/ but is not a "
            f"cross-cutting utility. Masking is a model concern, optimizers and "
            f"losses are training concerns, positional embeddings are layers.")

    def test_no_stray_modules_at_the_package_root(self):
        found = {p.stem for p in (REPO / "affmae").glob("*.py")}
        unexpected = found - self.ROOT_ALLOWED
        assert unexpected == set(), (
            f"{sorted(unexpected)} sits at the affmae/ root. Put it in the "
            f"subpackage that owns the concern: data/, layers/, models/, ops/, "
            f"training/, eval/ or viz/.")

    def test_ops_does_not_import_utils(self):
        """`affmae/ops/` must copy out standalone, so it cannot reach upward.

        `ops/knn_triton.py` used to import `affmae.utils.geometry` for the KeOps
        path. That was a lazy in-function import, so it never broke an
        import-time check -- only a copied-out `ops/` at runtime.
        """
        offenders = []
        for path in (REPO / "affmae" / "ops").rglob("*.py"):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#") or '"""' in stripped:
                    continue
                if "affmae.utils" in stripped and (
                        "import" in stripped):
                    offenders.append(
                        f"{path.relative_to(REPO)}:{number}: {stripped[:60]}")
        assert offenders == [], "\n  ".join([""] + offenders)

    def test_the_two_training_stages_are_symmetric(self):
        """Both stages should have an engine, or neither should.

        finetune.py was a 107-line CLI over a 435-line engine while pretrain.py
        carried all 442 lines inline, so `affmae/training/` held exactly one
        module and `wandb.init` was implemented twice.
        """
        training = REPO / "affmae" / "training"
        for stage in ("pretrain", "finetune"):
            assert (training / f"{stage}_engine.py").is_file(), (
                f"affmae/training/{stage}_engine.py is missing; the two stages "
                f"should be structured the same way.")
