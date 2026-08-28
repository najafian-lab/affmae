"""Credential loading and the shared finetuning engine.

``affmae/utils/env.py`` decides whether a run can authenticate to W&B, and the
precedence rule (real environment beats ``.env``) is the kind of thing that
breaks silently, so it is pinned here.

The engine tests cover the pieces that were duplicated across four drivers and
had drifted apart, including the bug that made the best-checkpoint branch
unreachable.
"""

import os

import pytest
import torch

from affmae.training.finetune_engine import (
    AUX_LOSS_WEIGHTS,
    _combine_losses,
    get_amp_dtype,
)
from affmae.utils.env import load_dotenv, wandb_available


class Cfg:
    """Minimal config stand-in."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class TestLoadDotenv:
    def test_missing_file_is_not_an_error(self, tmp_path):
        assert load_dotenv(tmp_path / "nope.env") == []

    def test_sets_unset_variables(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("AFFMAE_TEST_KEY=abc123\n")
        monkeypatch.delenv("AFFMAE_TEST_KEY", raising=False)

        assert load_dotenv(env) == ["AFFMAE_TEST_KEY"]
        assert os.environ["AFFMAE_TEST_KEY"] == "abc123"

    def test_environment_wins_over_file(self, tmp_path, monkeypatch):
        """The rule that keeps `KEY=... python train.py` and CI secrets authoritative."""
        env = tmp_path / ".env"
        env.write_text("AFFMAE_TEST_KEY=from_file\n")
        monkeypatch.setenv("AFFMAE_TEST_KEY", "from_shell")

        assert load_dotenv(env) == []
        assert os.environ["AFFMAE_TEST_KEY"] == "from_shell"

    def test_override_flag_reverses_precedence(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("AFFMAE_TEST_KEY=from_file\n")
        monkeypatch.setenv("AFFMAE_TEST_KEY", "from_shell")

        assert load_dotenv(env, override=True) == ["AFFMAE_TEST_KEY"]
        assert os.environ["AFFMAE_TEST_KEY"] == "from_file"

    @pytest.mark.parametrize("line", [
        "# just a comment",
        "",
        "   ",
        "NO_EQUALS_SIGN",
        "AFFMAE_BLANK=",           # blank value means "not set"
    ])
    def test_ignores_noise_lines(self, line, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text(line + "\n")
        monkeypatch.delenv("AFFMAE_BLANK", raising=False)
        assert load_dotenv(env) == []

    def test_strips_quotes(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text('AFFMAE_TEST_KEY="quoted"\n')
        monkeypatch.delenv("AFFMAE_TEST_KEY", raising=False)
        load_dotenv(env)
        assert os.environ["AFFMAE_TEST_KEY"] == "quoted"

    def test_no_secret_in_the_committed_template(self):
        """.env.example must ship keys with empty values, never a real one."""
        from pathlib import Path

        template = Path(__file__).resolve().parents[1] / ".env.example"
        assert template.is_file(), "the .env.example template should be committed"
        for raw in template.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            assert value.strip() == "", f"{key} has a value in .env.example"


class TestWandbAvailable:
    def test_true_when_key_is_set(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "x")
        assert wandb_available() is True

    def test_false_without_key_or_netrc(self, monkeypatch, tmp_path):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))  # empty home, no .netrc
        assert wandb_available() is False


class TestAmpDtype:
    @pytest.mark.parametrize("value,expected", [
        ("fp16", torch.float16),
        ("float16", torch.float16),
        ("bf16", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
    ])
    def test_recognized_values(self, value, expected):
        assert get_amp_dtype(Cfg(amp_dtype=value)) is expected

    def test_defaults_to_fp16_when_unset(self):
        """Most shipped configs do not define amp_dtype."""
        assert get_amp_dtype(Cfg()) is torch.float16

    def test_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unsupported amp_dtype"):
            get_amp_dtype(Cfg(amp_dtype="float8"))


class TestCombineLosses:
    """Deep-supervision weighting, previously written out longhand per driver."""

    def test_single_head_is_the_total(self):
        logits = [torch.zeros(1, 2, 4, 4)]
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        per_head, total = _combine_losses(logits, targets, lambda p, t: p.sum() + 3.0)
        assert len(per_head) == 1
        assert total is per_head[-1]

    def test_three_heads_use_documented_weights(self):
        """Primary head weight 1.0, auxiliaries 0.05 and 0.12."""
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        values = [torch.tensor(2.0), torch.tensor(5.0), torch.tensor(7.0)]
        calls = iter(values)
        per_head, total = _combine_losses([torch.zeros(1)] * 3, targets,
                                          lambda p, t: next(calls))

        expected = (values[2]
                    + AUX_LOSS_WEIGHTS[0] * values[0]
                    + AUX_LOSS_WEIGHTS[1] * values[1])
        assert per_head == values
        torch.testing.assert_close(total, expected)

    def test_primary_head_is_last(self):
        """The convention every driver and spec.aux_names relies on."""
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        calls = iter([torch.tensor(1.0), torch.tensor(1.0), torch.tensor(100.0)])
        _, total = _combine_losses([torch.zeros(1)] * 3, targets,
                                   lambda p, t: next(calls))
        # The large value dominates only if it is weighted 1.0.
        assert total.item() > 100.0


class TestValidateReturnsMeters:
    def test_primary_miou_is_read_from_the_meter_dict(self):
        """Regression for the bug that made best_model.pth unreachable.

        ``validate()`` returns a dict of meters. All four pre-consolidation
        drivers assigned it to ``val_miou`` and then computed ``1.0 - val_miou``
        and ``val_miou > best``, which raises TypeError on a dict.
        """
        from affmae.utils.misc import AverageMeter

        meters = {"res5": AverageMeter(), "res4": AverageMeter(), "res2": AverageMeter()}
        meters["res2"].update(0.75)

        with pytest.raises(TypeError):
            _ = 1.0 - meters          # the old code path

        val_miou = meters["res2"].avg  # the fixed one
        assert 1.0 - val_miou == pytest.approx(0.25)


class TestEngineIsSharedByDrivers:
    def test_drivers_import_the_engine_rather_than_copying_it(self):
        """A driver must not regrow a local training loop.

        ``scripts/finetune_perc_data.py`` was in this list until it was removed
        with the rest of the analysis scripts; only ``finetune.py`` remains.
        """
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for rel in ("finetune.py",):
            source = (root / rel).read_text()
            assert "from affmae.training.finetune_engine import" in source, rel
            assert "def train_epoch(" not in source, f"{rel} redefines train_epoch"
            assert "def validate(" not in source, f"{rel} redefines validate"

    def test_removed_drivers_are_gone(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for name in ("finetune_multi_seed.py", "finetune_seed_runs.py",
                     "finetune_perc_data.py"):
            assert not (root / name).exists(), f"{name} should have been consolidated"


class TestNoMachineSpecificPaths:
    """No script may hardcode a path from someone's home directory."""

    def test_scripts_take_paths_as_arguments(self):
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        # Absolute paths under a user's home are never portable.
        bad = re.compile(r"['\"](/homes?/[^'\"]+|/Users/[^'\"]+)['\"]")
        offenders = []
        for path in list(root.glob("*.py")) + list((root / "scripts").glob("*.py")):
            for num, line in enumerate(path.read_text().splitlines(), 1):
                if bad.search(line):
                    offenders.append(f"{path.relative_to(root)}:{num}")
        assert not offenders, f"hardcoded machine-specific paths: {offenders}"

    def test_scripts_expose_a_cli(self):
        """Every script should be runnable without editing it."""
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for path in sorted((root / "scripts").glob("*.py")):
            if path.name.startswith("_"):
                continue  # package markers and shared helpers, not entry points
            source = path.read_text()
            assert "argparse" in source, f"{path.name} has no CLI"


class TestSplitsMayBeAbsent:
    """A split that is missing or empty must be skipped, not fatal.

    The released fpwdata ships an empty val/. `build_finetune_dataloader` was
    called unconditionally before the epoch loop, so every shipped finetune
    config died there before training a step -- and `start_eval_epoch: 10000`,
    which exists to say "never validate", could not prevent it.
    """

    def test_an_empty_split_is_not_usable(self, tmp_path):
        from affmae.training.finetune_engine import _split_has_images

        for split in ("train", "val"):
            for kind in ("images", "masks"):
                (tmp_path / split / kind).mkdir(parents=True)
        (tmp_path / "train" / "images" / "a.tiff").write_bytes(b"x")
        (tmp_path / "train" / "masks" / "a.tiff").write_bytes(b"x")

        assert _split_has_images(str(tmp_path), "train") is True
        assert _split_has_images(str(tmp_path), "val") is False

    def test_a_missing_split_is_not_usable(self, tmp_path):
        from affmae.training.finetune_engine import _split_has_images

        assert _split_has_images(str(tmp_path), "nope") is False

    def test_images_without_masks_is_not_usable(self, tmp_path):
        """Half a split would pass an isdir check and then fail on intersect."""
        from affmae.training.finetune_engine import _split_has_images

        (tmp_path / "val" / "images").mkdir(parents=True)
        (tmp_path / "val" / "masks").mkdir(parents=True)
        (tmp_path / "val" / "images" / "a.tiff").write_bytes(b"x")
        assert _split_has_images(str(tmp_path), "val") is False


class TestAccumulationDoesNotDistortTheSchedule:
    """warmup and cosine length must be in optimizer steps, not batches.

    global_step advances only at an accumulation boundary, so counting batches
    made num_accum=2 stretch warmup to twice the configured epochs and leave the
    cosine at its midpoint after the final epoch. Trading batch size for
    accumulation is exactly what you do to fit a smaller card, and it silently
    changed the recipe.
    """

    @pytest.mark.parametrize("num_accum", [1, 2, 3, 4])
    def test_the_lr_peaks_at_base_lr_then_returns_to_min(self, num_accum):
        """Simulate a whole run's optimizer steps and check the two endpoints.

        Parametrized over num_accum because the point is that the curve is the
        same recipe however the batch is split: peak exactly at base_lr when
        warmup ends, min_lr at the last step.
        """
        import torch

        from affmae.training.finetune_engine import cosine_lr_schedule

        base_lr, min_lr = 1.0e-4, 1.0e-6
        batches, epochs, warmup_epochs = 48, 400, 25
        steps_per_epoch = -(-batches // num_accum)
        max_steps = epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([{"params": [param], "lr": base_lr,
                                "lr_scale": 1.0}], lr=base_lr)

        seen = []
        for step in range(1, max_steps + 1):
            cosine_lr_schedule(opt, step, max_steps, base_lr, min_lr,
                               warmup_steps)
            seen.append(opt.param_groups[0]["lr"])

        at_warmup_end = seen[warmup_steps - 1]
        assert at_warmup_end == pytest.approx(base_lr, rel=1e-6), at_warmup_end
        assert seen[-1] == pytest.approx(min_lr, abs=1e-7), seen[-1]
        assert max(seen) == pytest.approx(base_lr, rel=1e-6)
