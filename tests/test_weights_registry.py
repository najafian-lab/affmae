"""The released-checkpoint registry, and that the README table matches it.

A weights table drifts the moment a checkpoint is re-uploaded: the member stays,
the Drive id changes, and the link 404s for everyone but the author. These tests
pin the two representations to each other so the drift fails here instead.
"""

import re
from pathlib import Path

import pytest

from affmae.data.weights import (
    WEIGHTS_FOLDER_URL,
    CheckpointSpec,
    EMWeights,
    resolve_source,
)

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text()

def readme_table(first_column: str) -> list[str]:
    """Rows of the README table whose header starts with ``first_column``.

    Scoped on purpose: the weights table and the install-extras table have the
    same row shape, so a bare regex over the whole file reads each one's rows as
    the other's and reports phantom entries in both directions.
    """
    lines = README.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(f"| {first_column} |"):
            rows = []
            for row in lines[index + 2:]:        # skip the |---| separator
                if not row.startswith("|"):
                    break
                rows.append(row)
            return rows
    raise AssertionError(f"no README table starting with | {first_column} |")



class TestRegistry:
    def test_every_member_is_fully_described(self):
        for entry in EMWeights:
            assert isinstance(entry.spec, CheckpointSpec)
            assert entry.filename.endswith(".pth")
            assert entry.backbone in ("affmae", "vit")
            assert entry.task in ("pretrain", "segmentation")
            assert entry.img_size in (512, 768, 1024)
            assert entry.patch_size in (8, 16)
            assert entry.description.strip()

    def test_class_count_matches_the_task(self):
        """A pretraining checkpoint has no classes; a finetuned one must."""
        for entry in EMWeights:
            if entry.task == "pretrain":
                assert entry.num_classes is None, entry.name
            else:
                assert entry.num_classes == 3, (
                    f"{entry.name} claims {entry.num_classes} classes. The FPW "
                    f"release is 3: background, PGBMI, filtration slits.")

    def test_gdrive_ids_are_unique(self):
        ids = [entry.spec.gdrive_id for entry in EMWeights]
        assert len(ids) == len(set(ids)), "two members share one Drive file"

    def test_filenames_are_unique(self):
        """They share one cache directory per task, so a clash would overwrite."""
        seen = {}
        for entry in EMWeights:
            key = (entry.task, entry.filename)
            assert key not in seen, f"{entry.name} collides with {seen[key]}"
            seen[key] = entry.name

    def test_the_config_each_was_trained_with_exists(self):
        for entry in EMWeights:
            assert (REPO / entry.config).is_file(), (
                f"{entry.name} names {entry.config}, which is not in the repo. "
                f"from_checkpoint falls back to it when config= is omitted.")

    def test_names_encode_backbone_and_resolution(self):
        """The member name is the API; it should not disagree with the spec."""
        for entry in EMWeights:
            assert str(entry.img_size) in entry.name, entry.name
            prefix = "AFFMAE" if entry.backbone == "affmae" else "VIT"
            assert entry.name.startswith(prefix), entry.name
            expected = "PRETRAIN" if entry.task == "pretrain" else "FT"
            assert expected in entry.name, entry.name


class TestDownloadPath:
    def test_defaults_into_the_projects_weight_folders(self, monkeypatch):
        monkeypatch.delenv("CHECKPOINT_ROOT", raising=False)
        for entry in EMWeights:
            path = Path(entry.download_path)
            assert path.parent.name == entry.task
            assert path.parent.parent.name == "weights"
            assert path.name == entry.filename

    def test_checkpoint_root_relocates_the_cache(self, monkeypatch):
        monkeypatch.setenv("CHECKPOINT_ROOT", "/mnt/shared/w")
        assert EMWeights.AFFMAE_BASE_FT_512.download_path == (
            "/mnt/shared/w/segmentation/fpw_aff_base_ft_512_slits_pgbmi.pth")

    def test_an_existing_path_is_returned_untouched(self, tmp_path):
        """resolve_source must not treat a local file as something to fetch."""
        local = tmp_path / "mine.pth"
        local.write_bytes(b"not a real checkpoint")
        path, spec = resolve_source(str(local))
        assert path == str(local)
        assert spec is None


class TestNameResolution:
    """A registry name has to work from a command line, where all args are str.

    Without this the README's first command could only name a checkpoint file
    the reader does not have yet.
    """

    def test_a_member_name_resolves_to_that_member(self, monkeypatch):
        seen = {}

        def fake_fetch(self, progress=True):
            seen["name"] = self.name
            return f"/cache/{self.spec.filename}"

        monkeypatch.setattr(EMWeights, "fetch", fake_fetch)
        path, spec = resolve_source("AFFMAE_BASE_FT_512")
        assert seen["name"] == "AFFMAE_BASE_FT_512"
        assert spec is EMWeights.AFFMAE_BASE_FT_512.spec
        assert path.endswith(EMWeights.AFFMAE_BASE_FT_512.spec.filename)

    @pytest.mark.parametrize("entry", list(EMWeights), ids=lambda e: e.name)
    def test_every_member_is_reachable_by_name(self, entry, monkeypatch):
        monkeypatch.setattr(EMWeights, "fetch",
                            lambda self, progress=True: f"/cache/{self.name}")
        _, spec = resolve_source(entry.name)
        assert spec is entry.spec

    def test_a_near_miss_names_the_alternatives(self):
        """A typo must not fall through to a FileNotFoundError about a path."""
        with pytest.raises(KeyError) as caught:
            resolve_source("AFFMAE_BASE_FT_513")
        message = str(caught.value)
        assert "AFFMAE_BASE_FT_512" in message
        assert "not a released checkpoint" in message

    @pytest.mark.parametrize("source", [
        "weights/segmentation/mine.pth",     # has a separator and a suffix
        "mine.pth",                          # has a suffix
        "last_model.pth",
        "AFFMAE_BASE_FT_512.pth",            # a name plus a suffix is a file
    ])
    def test_paths_are_not_mistaken_for_names(self, source):
        path, spec = resolve_source(source)
        assert path == source
        assert spec is None

    def test_a_real_file_beats_a_member_of_the_same_name(self, tmp_path,
                                                        monkeypatch):
        """Someone with a file literally called AFFMAE_BASE_FT_512 must win.

        Otherwise the registry silently shadows their file and downloads
        something else.
        """
        local = tmp_path / "AFFMAE_BASE_FT_512"
        local.write_bytes(b"mine")
        monkeypatch.chdir(tmp_path)
        path, spec = resolve_source("AFFMAE_BASE_FT_512")
        assert path == "AFFMAE_BASE_FT_512"
        assert spec is None

    def test_lower_case_is_a_path_not_a_name(self):
        """Members are upper case; a lower-case string is somebody's filename."""
        path, spec = resolve_source("affmae_base_ft_512")
        assert path == "affmae_base_ft_512"
        assert spec is None

    def test_the_eval_loader_uses_the_same_resolver(self, monkeypatch):
        """`--checkpoint AFFMAE_BASE_FT_512` must mean one thing repo-wide."""
        from affmae.eval.loader import resolve_checkpoint

        monkeypatch.setattr(EMWeights, "fetch",
                            lambda self, progress=True: f"/cache/{self.name}")
        assert resolve_checkpoint(None, "AFFMAE_BASE_FT_512") == \
            "/cache/AFFMAE_BASE_FT_512"
        assert resolve_checkpoint(None, "out/last.pth") == "out/last.pth"
        assert resolve_checkpoint(None, "out/s{seed}/last.pth", seed=2) == \
            "out/s2/last.pth"


class TestDocsDoNotHardcodeReleasedFilenames:
    """A doc *command* must be runnable straight from a fresh clone.

    Every released .pth is gitignored, so a command naming one fails until the
    reader has separately worked out where to get it. The registry name is the
    runnable form, and both entry points accept it.

    Scoped to lines that load a checkpoint. Prose and YAML may legitimately name
    the file -- docs/custom_data.md shows `fetch()` printing the path and then
    the `pretrained_ckpt_path` to put it in, which is the point of the example.
    """

    FILENAMES = sorted(entry.spec.filename for entry in EMWeights)
    LOADERS = ("--checkpoint", "--pretrain-checkpoint", "from_checkpoint(")

    @pytest.mark.parametrize(
        "doc",
        sorted([REPO / "README.md"] + list((REPO / "docs").glob("*.md"))),
        ids=lambda p: p.name)
    def test_no_command_names_a_released_checkpoint_file(self, doc):
        offenders = []
        lines = doc.read_text().splitlines()
        for number, line in enumerate(lines, start=1):
            # A backslash-continued command puts the path on its own line, so
            # look at the previous line too rather than only this one.
            window = line if number == 1 else lines[number - 2] + line
            if not any(token in window for token in self.LOADERS):
                continue
            for name in self.FILENAMES:
                if name in line:
                    offenders.append(f"{doc.name}:{number}: {name}")
        assert not offenders, (
            f"these load a checkpoint a fresh clone does not have: {offenders}. "
            f"Use the registry name instead, e.g. AFFMAE_BASE_FT_512.")


class TestReadmeTable:
    """The table is what a user actually clicks."""

    @property
    def rows(self):
        return [re.match(r"\| `(\w+)` \|", row).group(1)
                for row in readme_table("Model")]

    def test_every_member_appears(self):
        listed = set(self.rows)
        missing = {entry.name for entry in EMWeights} - listed
        assert missing == set(), f"not in the README table: {sorted(missing)}"

    def test_no_phantom_members(self):
        known = {entry.name for entry in EMWeights}
        phantom = [name for name in self.rows if name not in known]
        assert phantom == [], f"table lists non-existent members: {phantom}"

    @pytest.mark.parametrize("entry", list(EMWeights), ids=lambda e: e.name)
    def test_each_row_links_the_right_drive_file(self, entry):
        """A stale id is invisible until someone clicks it."""
        assert entry.spec.gdrive_id in README, (
            f"{entry.name}'s Drive id is not in the README; the table link is "
            f"stale or missing.")

    @pytest.mark.parametrize("entry", list(EMWeights), ids=lambda e: e.name)
    def test_each_row_states_the_right_resolution(self, entry):
        """Class count and config were dropped from the table; resolution is the
        one spec value still shown, and a finetuned checkpoint is
        resolution-specific, so a wrong number sends someone to the wrong file."""
        row = next(line for line in README.splitlines()
                   if line.startswith(f"| `{entry.name}` |"))
        assert f"| {entry.img_size} |" in row, row

    @pytest.mark.parametrize("entry", list(EMWeights), ids=lambda e: e.name)
    def test_each_row_links_generically(self, entry):
        """The link text is `weights.pth`, not the real filename.

        Deliberate: the filenames are long and carry a class list that means
        nothing outside this project. The registry stays the single source of
        truth for the actual name, so the table cannot drift on it.
        """
        row = next(line for line in README.splitlines()
                   if line.startswith(f"| `{entry.name}` |"))
        assert f"[weights.pth]({entry.url})" in row, row
        assert entry.filename not in row, (
            f"{entry.name}: the row hardcodes the filename again")

    def test_the_folder_link_is_present(self):
        assert WEIGHTS_FOLDER_URL in README


class TestOnePackageSourceOfTruth:
    """pyproject.toml declares packages; nothing else may.

    The repo carried three overlapping lists: pyproject's dependencies and
    extras, a 406-line `conda env export` in environment.yml, and a
    requirements.txt. They disagreed, and there was no rule saying which won.
    """

    #: Files that would reintroduce a second list if someone added one.
    #: requirements.txt is deliberately absent: a HuggingFace Space can only be
    #: built from a file of exactly that name, so it is infrastructure rather
    #: than a rival spec. test_space_requirements_cannot_drift below is what
    #: keeps it honest -- it may not name a package pyproject does not.
    RIVALS = ("requirements-dev.txt", "constraints.txt",
              "Pipfile", "poetry.lock", "setup.py", "setup.cfg")

    @staticmethod
    def _environment():
        import yaml

        with open(REPO / "environment.yml") as handle:
            return yaml.safe_load(handle)

    @staticmethod
    def _pyproject():
        # tomllib is 3.11+; pyproject declares >=3.10, and the ROCm image
        # ships 3.10, where importing it made this module uncollectable.
        try:
            import tomllib
        except ModuleNotFoundError:
            try:
                import tomli as tomllib
            except ModuleNotFoundError:
                pytest.skip("needs tomllib (py>=3.11) or tomli")
        with open(REPO / "pyproject.toml", "rb") as handle:
            return tomllib.load(handle)

    def test_no_rival_package_list_exists(self):
        found = [name for name in self.RIVALS if (REPO / name).is_file()]
        assert found == [], (
            f"{found} would be a second package list. Declare packages in "
            f"pyproject.toml; environment.yml installs them via pip.")

    def test_space_requirements_cannot_drift_from_pyproject(self):
        """Every Space requirement must be declared in pyproject too.

        requirements.txt exists only because HuggingFace insists on the name.
        The rule that keeps it from becoming a second, disagreeing spec is that
        it may not introduce a package pyproject has never heard of -- so a new
        dependency has to land in pyproject first.
        """
        import re

        text = (REPO / "pyproject.toml").read_text()
        declared = set(re.findall(r'"([A-Za-z0-9_.\-]+)(?:[><=!\[]|")', text))
        declared = {name.lower().replace("_", "-") for name in declared}

        listed = {line.split(">=")[0].split("==")[0].strip().lower()
                  for line in (REPO / "requirements.txt").read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith("#")}

        # `spaces` is the ZeroGPU runtime, present only on a Space, and
        # affmae/demo.py deliberately never imports it -- so it has no business
        # in pyproject. opencv-python-headless is the same wheel as pyproject's
        # opencv-python without the GUI libs a container lacks.
        space_only = {"spaces", "opencv-python-headless"}
        undeclared = {name.replace("_", "-") for name in listed} - declared - space_only
        assert not undeclared, (
            f"requirements.txt names {sorted(undeclared)}, which pyproject.toml "
            f"does not declare. Add it to pyproject first, or to the "
            f"space_only exemption with a reason.")

    def test_environment_only_provides_the_interpreter(self):
        """Anything else in the conda list is a package pyproject should own."""
        allowed_prefixes = ("python", "pip")
        conda = [entry for entry in self._environment()["dependencies"]
                 if isinstance(entry, str)]
        unexpected = [entry for entry in conda
                      if not entry.split("=")[0].startswith(allowed_prefixes)]
        assert unexpected == [], (
            f"environment.yml installs {unexpected} through conda. Unless conda "
            f"is genuinely required, declare it in pyproject and let pip do it.")

    def test_environment_defers_to_pyproject(self):
        pip_entries = [entry["pip"] for entry in self._environment()["dependencies"]
                       if isinstance(entry, dict)]
        assert len(pip_entries) == 1, "expected exactly one pip block"
        entries = pip_entries[0]
        assert len(entries) == 1, (
            f"environment.yml pins packages directly: {entries}. It should hold "
            f"a single editable install of this project.")
        assert entries[0].startswith("-e ."), entries[0]

    def test_every_extra_it_requests_exists(self):
        """A typo here fails at install time with an unhelpful message."""
        import re

        entry = [e["pip"] for e in self._environment()["dependencies"]
                 if isinstance(e, dict)][0][0]
        requested = set(re.search(r"\[(.*)\]", entry).group(1).split(","))
        declared = set(self._pyproject()["project"]["optional-dependencies"])
        assert requested <= declared, (
            f"environment.yml asks for extras that pyproject does not define: "
            f"{sorted(requested - declared)}")

    def test_requests_is_a_core_dependency(self):
        """EMWeights.fetch is in the quick start, so it must not be an extra."""
        core = {name.split(">")[0].split("=")[0].strip()
                for name in self._pyproject()["project"]["dependencies"]}
        assert "requests" in core

    def test_every_extra_the_readme_installs_exists(self):
        """Every `pip install -e ".[a,b]"` in the docs must name real extras.

        pip does not fail on an unknown extra, it prints a warning and installs
        the base package, so a typo here leaves someone with a broken env and a
        line they scrolled past. This deliberately does not check *which* extras
        the README mentions or how many: how the install section is worded is an
        editorial choice, and pinning that just breaks the test when someone
        rewrites the prose.
        """
        import re

        declared = set(self._pyproject()["project"]["optional-dependencies"])
        commands = re.findall(r'pip install -e "\.\[([^\]]+)\]"', README)
        assert commands, "the README should show at least one extras install"

        named = {part.strip() for group in commands for part in group.split(",")}
        unknown = named - declared
        assert unknown == set(), (
            f"the README installs extras pyproject does not define: "
            f"{sorted(unknown)}")

    def test_the_environment_installs_the_gpu_stack(self):
        """GPU by default was a deliberate call: one command, working kernels.

        If `cuda` ever drops out of the recipe, a fresh env silently runs the
        PyTorch fallbacks and looks merely slow rather than misconfigured.
        """
        import re

        entry = [e["pip"] for e in self._environment()["dependencies"]
                 if isinstance(e, dict)][0][0]
        requested = {part.strip()
                     for part in re.search(r"\[(.*)\]", entry).group(1).split(",")}
        for needed in ("cuda", "viz", "dev"):
            assert needed in requested, (
                f"environment.yml no longer installs `{needed}`; the README "
                f"promises Triton kernels, figures and pytest from one command.")
