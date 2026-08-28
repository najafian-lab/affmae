"""The package must actually be installable and importable.

Before the `src/` -> `affmae/` rename, `pyproject.toml` declared
`packages.find include = ["src*"]` with **no `__init__.py` anywhere**, so `src`
was a PEP-420 namespace package that `find_packages` could not see:
`pip install .` shipped **zero modules**. Only `pip install -e .` worked, and
only because `pythonpath = ["."]` put the source tree on the path. Reusing any
part of this repo from another project was impossible regardless of coupling.

The build itself is exercised by test_wheel_ships_modules, which is marked slow.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PACKAGE = "affmae"


@pytest.fixture(scope="module")
def pyproject():
    with open(REPO / "pyproject.toml", "rb") as handle:
        return _load(handle)


def _load(handle):
    """tomllib is 3.11+; tomli is the 3.10 backport."""
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            import pytest

            pytest.skip("needs tomllib (py>=3.11) or tomli", allow_module_level=True)
    return tomllib.load(handle)


class TestDiscoverability:
    def test_setuptools_finds_the_packages(self, pyproject):
        """The failure mode that shipped an empty wheel."""
        from setuptools import find_packages

        include = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
        found = find_packages(where=str(REPO), include=include)
        assert found, (
            f"find_packages(include={include}) is empty -- pip install . would "
            f"ship no modules. Every package directory needs an __init__.py.")
        assert PACKAGE in found

    def test_every_package_dir_has_an_init(self):
        """A missing __init__.py silently drops a subpackage from the wheel."""
        missing = []
        for path in sorted((REPO / PACKAGE).rglob("*")):
            if not path.is_dir() or "__pycache__" in path.parts:
                continue
            # The CUDA extension's own src/ holds .cu/.cpp, not importable code.
            if path.name == "src" and path.parent.name == "cuda_ext":
                continue
            if not (path / "__init__.py").exists():
                missing.append(path.relative_to(REPO).as_posix())
        assert missing == [], f"package dirs without __init__.py: {missing}"

    def test_package_is_not_named_src(self, pyproject):
        """`src` as a distributed package name collides with every other repo."""
        assert pyproject["project"]["name"] == PACKAGE
        assert not (REPO / "src").exists(), "the old src/ tree is still present"


class TestDependencyDeclaration:
    def test_triton_is_optional(self, pyproject):
        """Triton is CUDA-centric; a hard dep contradicts CPU/MPS support."""
        required = " ".join(pyproject["project"]["dependencies"])
        assert "triton" not in required, "triton must live in the [cuda] extra"
        extras = pyproject["project"]["optional-dependencies"]
        assert any("triton" in dep for dep in extras["cuda"])

    def test_renderers_are_optional(self, pyproject):
        """Someone who wants only the operators should not install matplotlib."""
        required = " ".join(pyproject["project"]["dependencies"])
        for dep in ("matplotlib", "scikit-learn", "opencv"):
            assert dep not in required, f"{dep} must be an extra, not required"

    def test_declared_extras_exist(self, pyproject):
        extras = pyproject["project"]["optional-dependencies"]
        for name in ("cuda", "baselines", "viz", "train", "demo"):
            assert name in extras, f"missing extra: {name}"

    def test_license_is_not_asserted_prematurely(self, pyproject):
        """Licensing is deferred pending lab guidance.

        Drop this test once the LICENSE file lands.
        """
        assert "license" not in pyproject["project"], (
            "add the LICENSE file and the pyproject license field together")


@pytest.mark.slow
def test_wheel_ships_modules(tmp_path):
    """Build and install into a clean venv, then import from elsewhere.

    Importing from a different cwd is the point: from inside the repo, `affmae`
    resolves to the source tree whether or not the install worked.
    """
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True,
                   capture_output=True)
    python = venv / "bin" / "python"

    build = subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", "--quiet", str(REPO)],
        capture_output=True, text=True)
    assert build.returncode == 0, build.stderr[-3000:]

    probe = subprocess.run(
        [str(python), "-c",
         "import affmae, pathlib;"
         "root = pathlib.Path(affmae.__file__).parent;"
         "print('site-packages' in str(root), len(list(root.rglob('*.py'))))"],
        cwd=tmp_path, capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr[-2000:]

    in_site, count = probe.stdout.split()
    assert in_site == "True", "import resolved to the source tree, not the install"
    assert int(count) > 20, f"wheel shipped only {count} modules"


class TestKeopsIsDefaultInstalledButCapabilityGated:
    """pykeops is a default dependency, and that is safe because of the probe.

    Two requirements that look contradictory and are not. It must be installed by
    default: the ``unfused`` decoder path calls ``knn_keops``, and without it that
    silently falls back to a slower implementation, which inflated every
    fused-vs-unfused ratio measured before it was installed -- the 1024px figure
    went from 2.66x to 1.82x once it was. A silent fallback that moves a headline
    number by 46% does not belong behind an optional extra.

    And it must not break a portable install: KeOps has no ROCm and no MPS
    backend. That is handled by capability rather than by absence --
    ``dispatch.can_use_keops`` refuses those platforms with the package present --
    so installing it everywhere is safe. This class pins both halves, because
    dropping either one reintroduces a real bug: without the dependency, silent
    slow paths; without the probe, a hard crash on ROCm and Apple silicon.
    """

    def test_declared_as_a_runtime_dependency(self, pyproject):
        deps = [d.split(">")[0].split("[")[0].strip().lower()
                for d in pyproject["project"]["dependencies"]]
        assert "pykeops" in deps, (
            f"pykeops must be default-installed, not an extra; got {deps}")

    def test_the_extra_still_resolves(self, pyproject):
        """`affmae[keops]` appears in existing scripts and docs; keep it valid."""
        extras = pyproject["project"]["optional-dependencies"]
        assert extras.get("keops") == ["pykeops"]

    def test_the_probe_refuses_rocm_and_mps(self, monkeypatch):
        """The flag that makes the default install safe.

        Verified on an MI300X with pykeops 2.3 actually installed: the refusal is
        real, not an import failure. `is_cuda` cannot be used here, because a
        ROCm tensor reports `is_cuda == True` -- measured on that same hardware.
        """
        import torch

        from affmae.ops import dispatch

        monkeypatch.setattr(dispatch, "is_rocm_build", lambda: True)
        assert dispatch.can_use_keops(torch.zeros(1)) is False, (
            "a ROCm build must refuse KeOps even for a CPU tensor")

        monkeypatch.setattr(dispatch, "is_rocm_build", lambda: False)
        mps = torch.zeros(1).to("meta")           # stands in for a non-CUDA accel
        assert dispatch.can_use_keops(mps) is False


class TestSpaceApp:
    """The HuggingFace Space entry point, kept honest against the library.

    The Space is a copy of this repository plus app.py, so the two cannot be
    allowed to drift: a rename in affmae/demo.py that app.py still calls the old
    way fails only once the Space rebuilds, where nobody is watching.
    """

    APP = REPO / "app.py"

    def test_it_exists_and_parses(self):
        import ast

        ast.parse(self.APP.read_text())

    def test_importing_it_has_no_side_effects(self, monkeypatch):
        """It must not set CHECKPOINT_ROOT at import; other tests read that."""
        import importlib
        import os

        monkeypatch.delenv("CHECKPOINT_ROOT", raising=False)
        module = importlib.import_module("app")
        importlib.reload(module)
        assert "CHECKPOINT_ROOT" not in os.environ

    def test_zerogpu_is_gated_on_the_hardware_not_the_package(self, monkeypatch):
        """`spaces` is in requirements.txt, so it is importable on a CPU Space.

        Gating on the import alone would decorate handlers with spaces.GPU and
        force device="cuda" on hardware that has neither.
        """
        import importlib

        monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
        module = importlib.reload(importlib.import_module("app"))
        assert module.HAS_ZEROGPU is False

    def test_it_calls_build_interface_with_the_gpu_hook(self):
        """The whole point of the Space; without gpu= it silently runs on CPU."""
        source = self.APP.read_text()
        assert "build_interface(" in source
        assert "gpu=gpu" in source

    def test_requirements_covers_what_the_app_needs(self):
        required = {"spaces", "torch", "gradio", "huggingface_hub", "triton"}
        text = (REPO / "requirements.txt").read_text()
        listed = {line.split(">=")[0].split("==")[0].strip()
                  for line in text.splitlines()
                  if line.strip() and not line.startswith("#")}
        assert required <= listed, required - listed

    def test_requirements_omits_pykeops(self):
        """KeOps compiles CUDA C++ on first use; a Space has no toolchain.

        dispatch.can_use_keops() gates it by capability, so leaving it out means
        the PyTorch KNN path, not a failure.
        """
        listed = {line.split(">=")[0].split("==")[0].strip()
                  for line in (REPO / "requirements.txt").read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith("#")}
        # Not a substring check: the file's own comment explains the omission.
        assert "pykeops" not in listed
