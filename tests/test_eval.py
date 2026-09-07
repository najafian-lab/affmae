"""Tests for the evaluation package.

These cover the pure functions that used to live in the root ``eval_*.py``
scripts, where they were exercised only by running a full evaluation against a
trained checkpoint -- so a wrong reduction or a dropped payload key went
unnoticed until it silently changed a reported number.
"""

import json
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from affmae.eval.fpw import (
    FpwParams,
    aggregate_results,
    json_safe,
    parse_grid_size,
    summarize_across_seeds,
)
from affmae.eval.loader import (
    _extract_state_dict,
    amp_dtype_for,
    load_state_dict_into,
    resolve_checkpoint,
)


class _Cfg:
    """Minimal stand-in for a loaded Config."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestParseGridSize:
    def test_square_from_one_number(self):
        assert parse_grid_size("512") == (512, 512)

    def test_width_and_height(self):
        assert parse_grid_size("1024,768") == (1024, 768)

    @pytest.mark.parametrize("bad", ["", "1,2,3", "abc"])
    def test_rejects_anything_else(self, bad):
        with pytest.raises(ValueError):
            parse_grid_size(bad)


class TestFpwParams:
    def test_is_frozen(self):
        """Thresholds are shared across seeds; mutating one must not leak."""
        params = FpwParams()
        with pytest.raises(Exception):
            params.pgbmi_class = 5

    def test_defaults_match_the_documented_cli(self):
        params = FpwParams()
        assert (params.pgbmi_class, params.slit_class) == (1, 2)
        assert params.eval_grid_size == (1024, 1024)


class TestJsonSafe:
    def test_non_finite_floats_become_null(self):
        """json.dump writes NaN, which is not valid JSON and breaks readers."""
        out = json_safe({"a": float("nan"), "b": float("inf"),
                         "c": float("-inf"), "d": 1.5})
        assert out == {"a": None, "b": None, "c": None, "d": 1.5}
        json.dumps(out)   # must not raise

    def test_numpy_scalars_become_python(self):
        out = json_safe({"i": np.int64(3), "f": np.float32(0.5)})
        assert out == {"i": 3, "f": 0.5}
        assert type(out["i"]) is int

    def test_recurses_through_lists_and_tuples(self):
        assert json_safe((1, [np.int64(2), {"x": np.float64(3.0)}])) == \
            [1, [2, {"x": 3.0}]]


class TestSummarizeAcrossSeeds:
    @staticmethod
    def _run(seed, match_rate):
        return {"seed": seed, "summary": {"segment_match_rate": match_rate}}

    def test_pools_mean_and_population_std(self):
        out = summarize_across_seeds(
            [self._run(1, 0.2), self._run(2, 0.4), self._run(3, 0.6)])
        assert out["num_seeds"] == 3
        assert out["seeds"] == [1, 2, 3]
        assert out["segment_match_rate"]["count"] == 3
        assert out["segment_match_rate"]["mean"] == pytest.approx(0.4)
        # ddof=0, so std is over the seeds observed, not an estimate of a wider
        # population.
        assert out["segment_match_rate"]["std"] == pytest.approx(
            np.std([0.2, 0.4, 0.6], ddof=0))

    def test_drops_non_finite_seeds_instead_of_poisoning_the_mean(self):
        out = summarize_across_seeds(
            [self._run(1, 0.5), self._run(2, float("nan"))])
        assert out["segment_match_rate"]["count"] == 1
        assert out["segment_match_rate"]["mean"] == pytest.approx(0.5)

    def test_all_non_finite_reports_nan_not_a_crash(self):
        out = summarize_across_seeds([self._run(1, float("nan"))])
        assert out["segment_match_rate"]["count"] == 0
        assert math.isnan(out["segment_match_rate"]["mean"])

    def test_reads_nested_metric_blocks(self):
        """The metric arrives as {count, mean, std} rather than a scalar."""
        runs = [{"seed": s, "summary": {"fpw_mean_abs_error": {"mean": v, "count": 1}}}
                for s, v in ((1, 2.0), (2, 4.0))]
        assert summarize_across_seeds(runs)["fpw_mean_abs_error"]["mean"] == (
            pytest.approx(3.0))


class TestAggregateResults:
    @staticmethod
    def _image(n_gt, n_pred, matched, segments=()):
        return {"num_gt_segments": n_gt, "num_pred_segments": n_pred,
                "num_matched_segments": matched,
                "num_unmatched_gt_segments": n_gt - matched,
                "num_unmatched_pred_segments": n_pred - matched,
                "num_insufficient_slit_pairs": 0,
                "segments": list(segments)}

    def test_match_rate_is_matched_over_ground_truth(self):
        out = aggregate_results([self._image(4, 3, 2), self._image(6, 6, 3)])
        assert out["num_gt_segments"] == 10
        assert out["segment_match_rate"] == pytest.approx(5 / 10)

    def test_no_ground_truth_gives_nan_not_a_zero_division(self):
        out = aggregate_results([self._image(0, 2, 0)])
        assert math.isnan(out["segment_match_rate"])

    def test_pools_per_segment_metrics_across_images(self):
        out = aggregate_results([
            self._image(1, 1, 1, [{"fpw_mean_abs_error": 1.0}]),
            self._image(1, 1, 1, [{"fpw_mean_abs_error": 3.0}]),
        ])
        assert out["fpw_mean_abs_error"]["count"] == 2
        assert out["fpw_mean_abs_error"]["mean"] == pytest.approx(2.0)

    def test_only_foot_process_width_is_reported(self):
        """Frechet, Chamfer and slit-count error were reported beside it.

        Four numbers per segment invited quoting whichever looked best, and only
        width is the clinical quantity. Their absence is the point of this test.
        """
        out = aggregate_results([self._image(1, 1, 1, [{"fpw_mean_abs_error": 1.0}])])
        blocks = {key for key, value in out.items()
                  if isinstance(value, dict) and "mean" in value}
        assert blocks == {"fpw_mean_abs_error"}, (
            f"unexpected metric blocks in the summary: "
            f"{sorted(blocks - {'fpw_mean_abs_error'})}")


class TestCheckpointResolution:
    def test_defaults_to_the_run_directory(self):
        cfg = _Cfg(output_dir="/out", name="run")
        assert resolve_checkpoint(cfg) == "/out/run/last_model.pth"

    def test_seed_suffixes_the_run_directory(self):
        cfg = _Cfg(output_dir="/out", name="run")
        assert resolve_checkpoint(cfg, seed=77) == "/out/run_seed77/last_model.pth"

    def test_explicit_path_wins(self):
        cfg = _Cfg(output_dir="/out", name="run")
        assert resolve_checkpoint(cfg, "/tmp/a.pth") == "/tmp/a.pth"

    def test_seed_is_formatted_into_an_explicit_template(self):
        cfg = _Cfg(output_dir="/out", name="run")
        assert resolve_checkpoint(cfg, "/tmp/s{seed}.pth", seed=3) == "/tmp/s3.pth"


class TestStateDictExtraction:
    """Training runs saved weights under four different keys over time."""

    @pytest.mark.parametrize("wrapper", ["model_state_dict", "model",
                                         "state_dict"])
    def test_unwraps_every_payload_key(self, wrapper):
        state = _extract_state_dict({wrapper: {"weight": torch.zeros(2)}})
        assert list(state) == ["weight"]

    def test_accepts_a_bare_state_dict(self):
        assert list(_extract_state_dict({"weight": torch.zeros(2)})) == ["weight"]

    def test_strips_the_ddp_module_prefix(self):
        state = _extract_state_dict({"module.weight": torch.zeros(2)})
        assert list(state) == ["weight"]

    def test_drops_rebuilt_position_tables(self):
        """pre_table is rebuilt from the config, and a stale copy can disagree
        about width, so loading one is worse than missing it."""
        state = _extract_state_dict(
            {"weight": torch.zeros(2), "blocks.0.pre_table": torch.zeros(9)})
        assert list(state) == ["weight"]

    def test_load_reports_a_missing_file_by_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no checkpoint at"):
            load_state_dict_into(nn.Linear(2, 2), str(tmp_path / "nope.pth"))

    def test_round_trips_through_a_file(self, tmp_path):
        source = nn.Linear(3, 2)
        path = tmp_path / "ckpt.pth"
        torch.save({"model_state_dict": {f"module.{k}": v for k, v
                                         in source.state_dict().items()}}, path)
        target = nn.Linear(3, 2)
        load_state_dict_into(target, str(path))
        assert torch.equal(target.weight, source.weight)
        assert torch.equal(target.bias, source.bias)


class TestAmpDtype:
    @pytest.mark.parametrize("name,expected", [
        ("float16", torch.float16), ("fp16", torch.float16),
        ("bfloat16", torch.bfloat16), ("bf16", torch.bfloat16)])
    def test_accepted_names(self, name, expected):
        assert amp_dtype_for(_Cfg(amp_dtype=name)) is expected

    def test_defaults_to_fp16(self):
        assert amp_dtype_for(_Cfg()) is torch.float16

    def test_rejects_an_unknown_name(self):
        with pytest.raises(ValueError, match="unsupported amp_dtype"):
            amp_dtype_for(_Cfg(amp_dtype="float8"))
