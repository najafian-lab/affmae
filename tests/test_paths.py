"""Shared output locations, config path portability, and the plotting scripts.

Two classes of magic path used to make a fresh clone unusable: hardcoded
`/homes/iws/...` paths in `scripts/`, and absolute `output_dir` / `base_path` /
`pretrained_ckpt_path` values in all 66 configs pointing at four different
machines. Both are pinned here.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from affmae.config import _expand_env, load_config
from affmae.utils.paths import (
    OUTPUT_ENV_VAR,
    default_plot_path,
    output_path,
    output_root,
    plots_dir,
)

REPO = Path(__file__).resolve().parents[1]


class TestOutputRoot:
    def test_defaults_beside_the_repo(self, monkeypatch):
        monkeypatch.delenv(OUTPUT_ENV_VAR, raising=False)
        assert output_root() == REPO / "output"

    def test_env_var_overrides(self, monkeypatch, tmp_path):
        """The hook for scratch or network storage without editing configs."""
        monkeypatch.setenv(OUTPUT_ENV_VAR, str(tmp_path / "elsewhere"))
        assert output_root() == tmp_path / "elsewhere"

    def test_expands_user(self, monkeypatch):
        monkeypatch.setenv(OUTPUT_ENV_VAR, "~/affmae-out")
        assert "~" not in str(output_root())

    def test_plots_live_under_the_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv(OUTPUT_ENV_VAR, str(tmp_path))
        assert plots_dir() == tmp_path / "plots"
        assert plots_dir().is_dir()
        assert default_plot_path("f.pdf") == str(tmp_path / "plots" / "f.pdf")

    def test_output_path_can_create_parents(self, monkeypatch, tmp_path):
        monkeypatch.setenv(OUTPUT_ENV_VAR, str(tmp_path))
        target = output_path("a", "b", "c.json", create_parent=True)
        assert target.parent.is_dir()

    def test_output_dir_is_gitignored(self):
        """Artifacts must never be committable.

        Checks a path *inside* output/ rather than the directory itself: the
        ignore rule is `output/`, which matches directories only, so
        `git check-ignore output` reports "not ignored" whenever the directory
        does not happen to exist yet.
        """
        result = subprocess.run(
            ["git", "check-ignore", "-q", "output/plots/example.pdf"], cwd=REPO)
        assert result.returncode == 0, "output/ is not gitignored"


class TestConfigEnvExpansion:
    def test_expands_set_variable(self, monkeypatch):
        monkeypatch.setenv("AFFMAE_TEST_ROOT", "/data/root")
        assert _expand_env("${AFFMAE_TEST_ROOT}/x") == "/data/root/x"

    def test_uses_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("AFFMAE_TEST_ROOT", raising=False)
        assert _expand_env("${AFFMAE_TEST_ROOT:-/fallback}/x") == "/fallback/x"

    def test_set_variable_beats_default(self, monkeypatch):
        monkeypatch.setenv("AFFMAE_TEST_ROOT", "/live")
        assert _expand_env("${AFFMAE_TEST_ROOT:-/fallback}/x") == "/live/x"

    def test_unset_without_default_names_the_variable(self, monkeypatch):
        monkeypatch.delenv("AFFMAE_NO_SUCH", raising=False)
        with pytest.raises(KeyError, match="AFFMAE_NO_SUCH"):
            _expand_env("${AFFMAE_NO_SUCH}/x")

    def test_non_strings_pass_through(self):
        for value in (5, 1.5, None, True, ["a"]):
            assert _expand_env(value) is value


class TestConfigsArePortable:
    """No config may hardcode a path from someone's home directory."""

    #: Shipped configs only. `configs/smoke_*.yaml` is gitignored scratch for
    #: local runs -- pointing one at an absolute dataset path is the normal way
    #: to use it, so scanning them makes anyone's local smoke config fail the
    #: suite for a rule that only governs what we publish.
    CONFIGS = sorted(path for path in (REPO / "configs").glob("*.yaml")
                     if not path.name.startswith("smoke_"))

    def test_there_are_configs_to_check(self):
        assert self.CONFIGS

    def test_no_bare_home_directory_paths(self):
        bad = re.compile(r':\s*"(/homes?/|/Users/|/var/tmp/|/nfs/|/bigdata/)')
        offenders = []
        for path in self.CONFIGS:
            for num, line in enumerate(path.read_text().splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                if bad.search(line):
                    offenders.append(f"{path.name}:{num}")
        assert not offenders, (
            f"configs with unparameterized absolute paths: {offenders[:8]}")

    def test_output_dir_is_relative(self):
        for path in self.CONFIGS:
            for line in path.read_text().splitlines():
                stripped = line.lstrip()
                if stripped.startswith("output_dir:"):
                    value = stripped.split(":", 1)[1].split("#")[0].strip().strip('"')
                    assert not value.startswith("/"), f"{path.name}: {value}"

    def test_configs_load_without_any_env_setup(self):
        """A fresh clone must be able to read every config."""
        for path in self.CONFIGS:
            load_config(str(path))


class TestPlottingRemoved:
    """`scripts/plotting/` was deleted deliberately, to keep the repo small.

    It held two paper-figure scripts and their committed JSON measurements. The
    tests that ran them are gone with it; this one just pins the absence, so the
    directory does not quietly reappear along with its data files.
    """

    def test_the_plotting_package_is_gone(self):
        assert not (REPO / "scripts" / "plotting").exists(), (
            "scripts/plotting/ was removed to keep the repository small. If it "
            "is being restored deliberately, delete this test too.")

    def test_no_module_imports_it(self):
        offenders = []
        for path in list((REPO / "affmae").rglob("*.py")) + \
                list((REPO / "scripts").glob("*.py")):
            text = path.read_text()
            if "scripts.plotting" in text or "scripts/plotting" in text:
                offenders.append(str(path.relative_to(REPO)))
        assert offenders == [], f"still reference scripts/plotting: {offenders}"


class TestRemovedIcmlPlot:
    def test_icml_named_plot_is_gone(self):
        """The paper is ECCV; the ICML-named figure script was removed."""
        assert not (REPO / "scripts" / "generate_finetune_data_plot.py").exists()
        assert not (REPO / "scripts" / "generate_eff_rank_plot.py").exists()

    def test_no_icml_references_remain(self):
        offenders = []
        for path in list(REPO.glob("*.py")) + list((REPO / "scripts").rglob("*.py")):
            if "icml" in path.read_text().lower():
                offenders.append(path.name)
        assert not offenders, f"stale ICML references in {offenders}"


class TestWeightPathsArePortable:
    """Weight and data paths must not name anyone's home directory.

    Every config used to carry an absolute default like
    `${AFFMAE_ROOT:-/homes/iws/.../affmae_weights}/ckpt_epoch_99_aff_base_0.4ds.pth`,
    and the dataset was reached through *two* different variables with two
    different authors' defaults -- so which one you had to set depended on which
    config you opened.
    """

    #: Shipped configs only. `configs/smoke_*.yaml` is gitignored scratch for
    #: local runs -- pointing one at an absolute dataset path is the normal way
    #: to use it, so scanning them makes anyone's local smoke config fail the
    #: suite for a rule that only governs what we publish.
    CONFIGS = sorted(path for path in (REPO / "configs").glob("*.yaml")
                     if not path.name.startswith("smoke_"))

    def test_there_are_configs_to_check(self):
        assert self.CONFIGS, "no configs found"

    @staticmethod
    def _string_values(path):
        """Yield every (key, string value) in a config, comments excluded.

        Scanning raw lines instead was the bug in the first two versions of
        these tests: an inline ``# ...`` comment was read as part of the value,
        and a ``/mnt/...`` example inside a comment counted as an offence.
        """
        def walk(node, key=None):
            if isinstance(node, dict):
                for k, v in node.items():
                    yield from walk(v, k)
            elif isinstance(node, list):
                for v in node:
                    yield from walk(v, key)
            elif isinstance(node, str):
                yield key, node

        with open(path) as handle:
            yield from walk(yaml.safe_load(handle) or {})

    def test_no_config_embeds_a_home_directory(self):
        # Any absolute path under a user- or site-specific root. Earlier versions
        # looked only for /home, /homes and /Users, so they missed both an
        # /nfs/stak/users/... default and a /var/tmp/<user> shard directory.
        pattern = re.compile(r"/(home|homes|Users|nfs|scratch|mnt|var|tmp|opt)/")
        offenders = []
        for path in self.CONFIGS:
            for key, value in self._string_values(path):
                if pattern.search(value):
                    offenders.append(f"{path.name}: {key} = {value[:70]!r}")
        assert offenders == [], "\n  ".join([""] + offenders)

    def test_checkpoint_paths_use_checkpoint_root(self):
        offenders = []
        for path in self.CONFIGS:
            for key, value in self._string_values(path):
                if key not in ("pretrained_ckpt_path", "resume_path"):
                    continue
                if not value:
                    continue            # empty means "start from scratch"
                if not value.startswith("${CHECKPOINT_ROOT:-weights}/"):
                    offenders.append(f"{path.name}: {key} = {value!r}")
        assert offenders == [], "\n  ".join([""] + offenders)

    def test_only_one_variable_locates_the_dataset(self):
        """Two variables for one thing is how the confusion started.

        There were three: AFFMAE_ROOT, AFFMAE_DATA_DIR for ``base_path``, and
        AFFMAE_SHARD_DIR for the pretraining ``path`` -- which is how a
        ``/var/tmp`` default survived two portability passes.
        """
        seen = set()
        for path in self.CONFIGS:
            for line in path.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if not re.match(r"(base_)?path:", stripped):
                    continue
                seen |= set(re.findall(r"\$\{(\w+):-", stripped))
        assert seen <= {"DATA_ROOT", "CHECKPOINT_ROOT"}, (
            f"dataset located via {sorted(seen)}")

    def test_every_dataset_path_lives_under_data_root(self):
        """Both dataset keys, not just the one I audited the first time."""
        offenders = []
        for path in self.CONFIGS:
            for line in path.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                match = re.match(r'(base_path|path): *"([^"]*)"', stripped)
                if not match or not match.group(2):
                    continue
                key, value = match.groups()
                if key == "path" and "CHECKPOINT_ROOT" in value:
                    continue        # `path` is overloaded: some are checkpoints
                if not value.startswith("${DATA_ROOT:-data}/"):
                    offenders.append(f"{path.name}: {key} is {value!r}")
        assert offenders == [], "\n  ".join([""] + offenders)

    def test_class_weighting_length_matches_num_classes(self):
        """A mismatch is not cosmetic: it raises inside the loss.

        Eight configs shipped five weights against ``num_classes: 3``, left over
        from an earlier label set. ComboLoss rejects that with "weight tensor
        should be defined either for all or no classes", so those runs died at
        loss construction rather than training with the wrong weights.
        """
        offenders = []
        for path in self.CONFIGS:
            text = path.read_text()
            classes = re.search(r"^\s*num_classes: *(\d+)", text, re.M)
            weights = re.search(r"^\s*class_weighting: *\[([^\]]*)\]", text, re.M)
            if not classes or not weights:
                continue
            count = len([w for w in weights.group(1).split(",") if w.strip()])
            if int(classes.group(1)) != count:
                offenders.append(
                    f"{path.name}: num_classes={classes.group(1)} "
                    f"but {count} weights")
        assert offenders == [], "\n  ".join([""] + offenders)

    def test_the_templates_load_and_are_self_consistent(self):
        """The copy-me configs are the first thing a new user runs."""
        templates = sorted(REPO.glob("configs/template_*.yaml"))
        assert len(templates) == 2, [t.name for t in templates]
        for path in templates:
            cfg = load_config(str(path))
            assert os.path.isabs(cfg.output_dir) or cfg.output_dir == "output"
            if hasattr(cfg, "num_classes"):
                assert cfg.num_classes == len(cfg.indices) + 1
                assert len(cfg.class_weighting) == cfg.num_classes
            if hasattr(cfg, "total_samples"):
                assert cfg.total_samples > 0

    def test_relative_paths_resolve_against_the_repo_not_the_cwd(self, monkeypatch):
        """`python evaluate.py` has to work from any directory."""
        monkeypatch.delenv("CHECKPOINT_ROOT", raising=False)
        monkeypatch.delenv("DATA_ROOT", raising=False)
        cfg = load_config(str(REPO / "configs" / "aff_base_finetune_512_fpw.yaml"))
        assert os.path.isabs(cfg.pretrained_ckpt_path)
        assert cfg.pretrained_ckpt_path.startswith(str(REPO))
        assert os.path.isabs(cfg.base_path)
        assert cfg.base_path.startswith(str(REPO))

    def test_checkpoint_root_relocates_every_weight(self, monkeypatch):
        monkeypatch.setenv("CHECKPOINT_ROOT", "/mnt/shared/w")
        cfg = load_config(str(REPO / "configs" / "aff_base_finetune_512_fpw.yaml"))
        assert cfg.pretrained_ckpt_path.startswith("/mnt/shared/w/pretrain/")

    def test_an_absolute_override_is_left_alone(self, monkeypatch):
        """An absolute CHECKPOINT_ROOT must be used verbatim.

        The filename is read from the config rather than written here: this test
        is about the override, and hardcoding it meant a legitimate change of
        backbone failed a path test for the wrong reason.
        """
        import re

        name = re.search(
            r"pretrained_ckpt_path:.*/pretrain/([^\"\']+)",
            (REPO / "configs" / "aff_base_finetune_768.yaml").read_text()).group(1)

        monkeypatch.setenv("CHECKPOINT_ROOT", "/abs/elsewhere")
        cfg = load_config(str(REPO / "configs" / "aff_base_finetune_768.yaml"))
        assert cfg.pretrained_ckpt_path == f"/abs/elsewhere/pretrain/{name}"

    def test_the_documented_layout_exists(self):
        """The README promises weights/pretrain and weights/segmentation."""
        assert (REPO / "weights" / "README.md").exists()


class TestEveryConfigsClassesAreConsistent:
    """num_classes must equal len(indices) + 1 in every config, not just AFF's.

    Seven ViT configs carried indices=[0,1,2,3] against num_classes=3. The
    mismatch only shows up eight seconds into training, as a CUDA device-side
    assert inside cross_entropy -- a long way from the config that caused it.
    """

    CONFIGS = sorted(path for path in (REPO / "configs").glob("*.yaml")
                     if not path.name.startswith(("smoke_", "template_")))

    @staticmethod
    def _flat(path):
        import yaml

        flat = {}
        for section in (yaml.safe_load(path.read_text()) or {}).values():
            if isinstance(section, dict):
                flat.update(section)
        return flat

    @pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
    def test_num_classes_counts_the_background(self, path):
        flat = self._flat(path)
        if "indices" not in flat or "num_classes" not in flat:
            pytest.skip("not a segmentation config")
        indices, num_classes = flat["indices"], flat["num_classes"]
        assert num_classes == len(indices) + 1, (
            f"{path.name}: indices={indices} selects {len(indices)} foreground "
            f"classes, so num_classes should be {len(indices) + 1}, not "
            f"{num_classes}. Trimming class_weighting instead hides this.")

    @pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
    def test_one_class_weight_per_class(self, path):
        flat = self._flat(path)
        if "class_weighting" not in flat or "num_classes" not in flat:
            pytest.skip("no class weighting")
        assert len(flat["class_weighting"]) == flat["num_classes"], (
            f"{path.name}: {len(flat['class_weighting'])} weights for "
            f"{flat['num_classes']} classes.")


class TestTheThreeAffConfigsShareOneArchitecture:
    """512/768/1024 are the same backbone at three resolutions.

    aff_base_finetune_768.yaml had aff_nbhd_sizes [128,128,128,128] where the
    other two use [64,64,64,64]. Neighbourhood size adds no parameters, so the
    wrong value cannot fail to load -- it silently builds a different model.
    Evaluating the released 768 checkpoint proved 64 is what it was trained
    with: mIoU 0.6267 at 64 against 0.5973 at 128.

    Only architecture is compared. img_size, batch_size and num_workers differ
    by design.
    """

    KEYS = ("aff_embed_dims", "aff_depths", "aff_num_heads", "aff_nbhd_sizes",
            "aff_cluster_size", "aff_ds_rates", "aff_mlp_ratio", "patch_size",
            "decoder_embed_dim", "decoder_depth", "decoder_num_heads",
            "in_channels", "num_classes")

    NAMES = ("aff_base_finetune_512_fpw.yaml", "aff_base_finetune_768.yaml",
             "aff_base_finetune_1024_fpw.yaml")

    @staticmethod
    def _flat(name):
        import yaml

        flat = {}
        for section in (yaml.safe_load(
                (REPO / "configs" / name).read_text()) or {}).values():
            if isinstance(section, dict):
                flat.update(section)
        return flat

    @pytest.mark.parametrize("key", KEYS)
    def test_the_key_is_the_same_in_all_three(self, key):
        values = {name: self._flat(name).get(key, "<absent>")
                  for name in self.NAMES}
        distinct = {repr(v) for v in values.values()}
        assert len(distinct) == 1, (
            f"{key} differs across the aff finetune configs: {values}. These "
            f"are one backbone at three resolutions; an architecture key that "
            f"differs is a silent model change, not a resolution setting.")


class TestFinetuneConfigsMatchThePaper:
    """The FPW finetune configs must state what the paper says they used.

    The 768 config had drifted on four counts: 600 epochs instead of 400, no
    `layer_decay` at all (so it silently used the affmae spec default of 0.8
    rather than 0.6), no `num_accum`, and a `pretrained_ckpt_path` naming a
    checkpoint that is neither on disk nor in the released registry -- so the
    config the registry advertises for AFFMAE_BASE_FT_768 could not run.
    """

    CONFIGS = ("aff_base_finetune_512_fpw.yaml",
               "aff_base_finetune_768.yaml",
               "aff_base_finetune_1024_fpw.yaml")

    #: From the paper's FPW segmentation setup.
    EXPECTED = {
        "epochs": 400,
        "learning_rate": 1.0e-4,
        "min_lr": 1.0e-6,
        "warmup_epochs": 25,
        "layer_decay": 0.6,
        "loss_fn": "combo",
        "class_weighting": [0.2, 2.0, 3.0],
        "num_classes": 3,
    }

    def _load(self, name):
        import yaml

        raw = yaml.safe_load((REPO / "configs" / name).read_text())
        flat = {}
        for section in raw.values():
            if isinstance(section, dict):
                flat.update(section)
        return flat

    @pytest.mark.parametrize("name", CONFIGS)
    def test_every_paper_hyperparameter_is_stated(self, name):
        flat = self._load(name)
        wrong = {key: (flat.get(key, "<absent>"), want)
                 for key, want in self.EXPECTED.items()
                 if flat.get(key, "<absent>") != want}
        assert not wrong, (
            f"{name} disagrees with the paper (got, want): {wrong}. "
            f"An absent key is not harmless: layer_decay falls back to the "
            f"affmae spec default of 0.8.")

    @pytest.mark.parametrize("name", CONFIGS)
    def test_indices_and_num_classes_agree(self, name):
        """num_classes must be len(indices) + 1, for the background class.

        The 768 config selected four mask channels against a three-way head, so
        targets carried label 4 and cross_entropy tripped a CUDA device-side
        assert eight seconds into training -- an opaque failure a long way from
        its cause. The config's own comment says "should be indices + bg class".
        """
        flat = self._load(name)
        indices, num_classes = flat["indices"], flat["num_classes"]
        assert num_classes == len(indices) + 1, (
            f"{name}: indices={indices} selects {len(indices)} foreground "
            f"classes, so num_classes should be {len(indices) + 1}, not "
            f"{num_classes}.")

    @pytest.mark.parametrize("name", CONFIGS)
    def test_class_weighting_has_one_entry_per_class(self, name):
        """A weight vector of the wrong length is a silent mis-weighting."""
        flat = self._load(name)
        assert len(flat["class_weighting"]) == flat["num_classes"], (
            f"{name}: {len(flat['class_weighting'])} weights for "
            f"{flat['num_classes']} classes.")

    @pytest.mark.parametrize("name", CONFIGS)
    def test_the_pretrained_backbone_is_a_released_checkpoint(self, name):
        """A config naming an unreleased backbone cannot be reproduced."""
        from affmae.data.weights import EMWeights

        flat = self._load(name)
        path = flat["pretrained_ckpt_path"]
        released = {entry.spec.filename for entry in EMWeights
                    if entry.spec.task == "pretrain"}
        assert any(name_ in path for name_ in released), (
            f"{name} starts from {path!r}, which is not one of the released "
            f"pretraining checkpoints {sorted(released)}.")
