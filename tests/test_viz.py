"""Visualization: parameters must take effect, and ground truth must be optional.

The old renderers hardcoded everything — the same kind of artifact was written
at five different dpi values, the error colour differed in three files, and the
token-dot radius was a fixed 2 px regardless of image size (drawn into a cell
rasterized at ~375 px, so dots often vanished).

The load-bearing test here is
:meth:`TestParametersTakeEffect.test_changing_a_setting_changes_the_bytes`: a
config field that is accepted and then ignored is worse than no field at all,
because it looks configured.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from affmae.viz import (
    DEBUG,
    OKABE_ITO,
    PAPER,
    PRESENTATION,
    VizConfig,
    class_overlay,
    denormalize,
    error_overlay,
    logits_to_labels,
    draw_polylines,
    render_comparison,
    render_reconstruction,
    render_segmentation,
    render_token_layout,
    save_segment_overlay,
    to_display_image,
)

B, C, H, W, K = 3, 1, 32, 32, 4


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return dict(
        images=torch.randn(B, C, H, W),
        logits=torch.randn(B, K, H, W),
        targets=torch.randint(0, K, (B, H, W)),
    )


class TestConfig:
    def test_frozen_so_it_cannot_be_mutated_in_place(self):
        """A shared config must not be editable by one renderer."""
        with pytest.raises(Exception):
            PAPER.dpi = 999

    def test_with_returns_a_variant(self):
        variant = PAPER.with_(dpi=300)
        assert variant.dpi == 300 and PAPER.dpi != 300

    @pytest.mark.parametrize("kwargs,match", [
        ({"dpi": 0}, "dpi"),
        ({"overlay_alpha": 1.5}, "overlay_alpha"),
        ({"error_alpha": -1}, "error_alpha"),
        ({"token_render_scale": 0.5}, "token_render_scale"),
        ({"max_samples": 0}, "max_samples"),
        ({"palette": ()}, "palette"),
    ])
    def test_validates_its_fields(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            VizConfig(**kwargs)

    def test_palette_cycles_instead_of_indexerror(self):
        """A previous palette ignored its size argument and returned 3 colours,
        so any config with more than 3 classes raised IndexError."""
        config = VizConfig()
        assert config.class_color(len(OKABE_ITO) + 1) == config.class_color(1)

    def test_token_radius_scales_with_image_size(self):
        """A fixed radius is invisible at 1024 or covers the image at 256."""
        config = VizConfig(token_radius=None)
        assert config.resolve_token_radius(256) < config.resolve_token_radius(1024)

    def test_explicit_token_radius_wins(self):
        assert VizConfig(token_radius=7).resolve_token_radius(1024) == 7

    def test_presets_differ(self):
        assert PAPER.dpi != PRESENTATION.dpi
        assert DEBUG.max_samples < PAPER.max_samples


class TestPrimitives:
    @pytest.mark.parametrize("channels", [1, 3])
    def test_denormalize_is_channel_agnostic(self, channels):
        """The copies this replaces reshaped stats to (1,3,1,1), which broke on
        the single-channel data the configs actually use."""
        out = denormalize(torch.randn(2, channels, 8, 8))
        assert out.shape == (2, channels, 8, 8)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_denormalize_accepts_explicit_stats(self):
        got = denormalize(torch.zeros(1, 1, 4, 4), mean=0.5, std=2.0)
        assert torch.allclose(got, torch.full((1, 1, 4, 4), 0.5))

    @pytest.mark.parametrize("shape,expected_ndim", [
        ((8, 8), 2), ((1, 8, 8), 2), ((3, 8, 8), 3),
    ])
    def test_to_display_image_shapes(self, shape, expected_ndim):
        out = to_display_image(torch.randn(*shape), PAPER)
        assert out.ndim == expected_ndim
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_logits_to_labels_accepts_logits_or_labels(self):
        assert logits_to_labels(torch.randn(K, 8, 8)).shape == (8, 8)
        assert logits_to_labels(torch.zeros(8, 8, dtype=torch.long)).shape == (8, 8)

    def test_logits_to_labels_rejects_a_batch(self):
        """Callers must pick a head; a model's forward returns a list."""
        with pytest.raises(ValueError, match="expected"):
            logits_to_labels(torch.randn(2, K, 8, 8))

    def test_class_overlay_leaves_background_transparent(self):
        labels = torch.zeros(8, 8, dtype=torch.long)
        assert class_overlay(labels, K, PAPER)[..., 3].max() == 0.0

    def test_class_overlay_uses_the_configured_palette(self):
        labels = torch.ones(4, 4, dtype=torch.long)
        custom = VizConfig(palette=((0, 0, 0), (1.0, 0.0, 0.0)))
        overlay = class_overlay(labels, 2, custom)
        assert np.allclose(overlay[0, 0, :3], (1.0, 0.0, 0.0))

    def test_error_overlay_marks_only_mistakes(self):
        labels = torch.zeros(4, 4, dtype=torch.long)
        target = torch.zeros(4, 4, dtype=torch.long)
        target[0, 0] = 1
        alpha = error_overlay(labels, target, PAPER)[..., 3]
        assert alpha[0, 0] > 0 and alpha[1, 1] == 0


class TestSegmentationRenderer:
    def test_ground_truth_is_optional(self, batch, tmp_path):
        """The demo case: predict on an unlabelled image.

        The renderer this replaces required targets for both the GT column and
        the error map, so a prediction-only figure was impossible.
        """
        out = render_segmentation(batch["images"], batch["logits"], K,
                                  str(tmp_path / "pred.png"), targets=None)
        assert Path(out).stat().st_size > 0

    def test_with_ground_truth_adds_columns(self, batch, tmp_path):
        without = tmp_path / "a.png"
        with_gt = tmp_path / "b.png"
        render_segmentation(batch["images"], batch["logits"], K, str(without))
        render_segmentation(batch["images"], batch["logits"], K, str(with_gt),
                            targets=batch["targets"])
        # 4 columns vs 2: the file must be materially larger.
        assert with_gt.stat().st_size > without.stat().st_size

    def test_respects_max_samples(self, batch, tmp_path):
        few = tmp_path / "few.png"
        many = tmp_path / "many.png"
        render_segmentation(batch["images"], batch["logits"], K, str(few),
                            config=VizConfig(max_samples=1))
        render_segmentation(batch["images"], batch["logits"], K, str(many),
                            config=VizConfig(max_samples=B))
        assert many.stat().st_size > few.stat().st_size

    def test_accepts_label_maps_not_only_logits(self, batch, tmp_path):
        labels = batch["logits"].argmax(dim=1)
        out = render_segmentation(batch["images"], labels, K,
                                  str(tmp_path / "labels.png"))
        assert Path(out).stat().st_size > 0

    def test_creates_missing_directories(self, batch, tmp_path):
        nested = tmp_path / "deep" / "deeper" / "x.png"
        render_segmentation(batch["images"], batch["logits"], K, str(nested))
        assert nested.exists()


class TestComparisonRenderer:
    def test_renders_n_models(self, batch, tmp_path):
        preds = [batch["logits"], torch.randn(B, K, H, W), torch.randn(B, K, H, W)]
        out = render_comparison(batch["images"], preds, ["A", "B", "C"], K,
                                str(tmp_path / "cmp.png"),
                                targets=batch["targets"])
        assert Path(out).stat().st_size > 0

    def test_rejects_mismatched_names(self, batch, tmp_path):
        with pytest.raises(ValueError, match="names"):
            render_comparison(batch["images"], [batch["logits"]], ["A", "B"], K,
                              str(tmp_path / "x.png"))

    def test_works_without_ground_truth(self, batch, tmp_path):
        out = render_comparison(batch["images"], [batch["logits"]], ["A"], K,
                                str(tmp_path / "nogt.png"), targets=None)
        assert Path(out).stat().st_size > 0


class TestReconstructionRenderer:
    def test_renders_stages_and_residual(self, batch, tmp_path):
        recons = [torch.randn(B, C, H, W), torch.randn(B, C, H, W)]
        out = render_reconstruction(batch["images"], batch["images"] * 0.5,
                                    recons, str(tmp_path / "r.png"),
                                    stage_names=["Res5", "Res2"],
                                    show_residual=True)
        assert Path(out).stat().st_size > 0

    def test_masked_column_is_optional(self, batch, tmp_path):
        out = render_reconstruction(batch["images"], None,
                                    [torch.randn(B, C, H, W)],
                                    str(tmp_path / "r2.png"))
        assert Path(out).stat().st_size > 0

    def test_rejects_empty_reconstructions(self, batch, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            render_reconstruction(batch["images"], None, [],
                                  str(tmp_path / "x.png"))


class TestTokenRenderer:
    def test_renders_stages(self, batch, tmp_path):
        positions = [torch.rand(B, 32, 2) * 4, torch.rand(B, 8, 2) * 4]
        out = render_token_layout(batch["images"], positions, patch_size=8,
                                  save_path=str(tmp_path / "t.png"))
        assert Path(out).stat().st_size > 0

    def test_dots_are_the_configured_colour(self, batch):
        """Pin RGB, and mean it.

        The previous version of this test said "Pin RGB" in its docstring and
        then asserted channel 2 -- i.e. BGR -- so it passed while a configured
        red rendered blue for every direct caller. cv2.circle does not interpret
        colour; it writes the tuple into the array's channels, and this canvas is
        RGB.
        """
        from affmae.viz import draw_token_positions

        config = VizConfig(token_color=(1.0, 0.0, 0.0), token_radius=4)
        canvas = draw_token_positions(torch.zeros(1, 32, 32),
                                      torch.tensor([[2.0, 2.0]]),
                                      patch_size=8, config=config)
        centre = canvas[20, 20]
        assert centre[0] > 200 and centre[2] < 60, f"expected red, got {centre}"

    def test_a_colour_background_keeps_its_channels(self, batch):
        """render_token_layout used to flip the whole array to undo the token
        reversal, which also swapped the background's R and B. Invisible on the
        greyscale EM images, wrong on anything colour."""
        from affmae.viz import draw_token_positions

        red = torch.zeros(3, 32, 32)
        red[0] = 1.0
        canvas = draw_token_positions(red, torch.tensor([[0.0, 0.0]]), 8,
                                      VizConfig(token_radius=2))
        corner = canvas[30, 30]          # far from the token
        assert corner[0] > corner[2], (
            f"red background came back as {corner}")

    def test_both_token_paths_agree_on_colour(self, batch, tmp_path):
        """One source of truth: the montage path and the direct call had
        opposite channel order, so figures disagreed depending on entry point."""
        from affmae.viz import draw_token_positions

        config = VizConfig(token_color=(1.0, 0.0, 0.0), token_radius=4)
        direct = draw_token_positions(torch.zeros(1, 32, 32),
                                      torch.tensor([[2.0, 2.0]]), 8, config)
        # render_token_layout consumes draw_token_positions unmodified now, so
        # what it hands to imshow is exactly what a direct caller receives.
        assert direct[20, 20][0] > direct[20, 20][2]

    def test_supersampling_enlarges_the_canvas(self, batch):
        from affmae.viz import draw_token_positions

        plain = draw_token_positions(torch.zeros(1, 32, 32),
                                     torch.tensor([[1.0, 1.0]]), 8,
                                     VizConfig(token_render_scale=1.0))
        scaled = draw_token_positions(torch.zeros(1, 32, 32),
                                      torch.tensor([[1.0, 1.0]]), 8,
                                      VizConfig(token_render_scale=2.0))
        assert scaled.shape[0] == plain.shape[0] * 2

    def test_rejects_no_stages(self, batch, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            render_token_layout(batch["images"], [], 8, str(tmp_path / "x.png"))


class TestParametersTakeEffect:
    """A setting that is accepted and ignored is worse than no setting."""

    @pytest.mark.parametrize("overrides", [
        {"dpi": 72},
        {"palette": ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))},
        {"overlay_alpha": 0.2},
        {"error_color": (0.0, 1.0, 0.0)},
        {"background_gain": 0.9},
        {"show_titles": False},
        {"figsize_per_cell": 2.0},
    ])
    def test_changing_a_setting_changes_the_bytes(self, batch, tmp_path, overrides):
        baseline = tmp_path / "base.png"
        variant = tmp_path / "variant.png"
        render_segmentation(batch["images"], batch["logits"], K, str(baseline),
                            targets=batch["targets"], config=PAPER)
        render_segmentation(batch["images"], batch["logits"], K, str(variant),
                            targets=batch["targets"],
                            config=PAPER.with_(**overrides))
        assert baseline.read_bytes() != variant.read_bytes(), (
            f"{overrides} was accepted but had no effect on the output")


def test_viz_sets_a_headless_backend():
    """Renderers must not require a display; only two scripts set this before."""
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"


def _segment(segment_id, skeleton, slits):
    """Build a SegmentPolyline with only the fields the renderers read."""
    from affmae.eval.fpw_geometry import SegmentPolyline

    return SegmentPolyline(
        segment_id=segment_id,
        mask=np.zeros((4, 4), dtype=bool),
        skeleton_mask=np.zeros((4, 4), dtype=bool),
        skeleton_points_xy=np.asarray(skeleton, dtype=np.float64).reshape(-1, 2),
        skeleton_arc_lengths=np.zeros(len(skeleton), dtype=np.float64),
        slit_points_xy=np.asarray(slits, dtype=np.float64).reshape(-1, 2),
        slit_arc_lengths=np.zeros(len(slits), dtype=np.float64),
    )


class TestDrawPolylines:
    """One drawer for segment geometry, parameterized by colour and label."""

    @staticmethod
    def _axis():
        import matplotlib.pyplot as plt

        return plt.subplots()

    def test_draws_a_line_and_markers_per_segment(self):
        fig, ax = self._axis()
        draw_polylines(ax, [_segment(0, [(0, 0), (1, 1)], [(0, 0)]),
                            _segment(1, [(2, 2), (3, 3)], [(3, 3)])],
                       color="red")
        assert len(ax.lines) == 2
        assert len(ax.collections) == 2
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_labels_only_the_first_group_so_the_legend_has_one_entry(self):
        """Labelling every segment produced a legend with one row per segment."""
        fig, ax = self._axis()
        draw_polylines(ax, [_segment(i, [(i, i), (i + 1, i + 1)], [(i, i)])
                            for i in range(4)],
                       color="red", label="Pred slits")
        _, labels = ax.get_legend_handles_labels()
        assert labels == ["Pred slits"]
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_skips_empty_geometry_without_erroring(self):
        fig, ax = self._axis()
        draw_polylines(ax, [_segment(0, [], [])], color="red")
        assert not ax.lines and not ax.collections
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestSaveSegmentOverlay:
    def test_writes_a_figure(self, tmp_path):
        out = tmp_path / "fpw.png"
        returned = save_segment_overlay(
            torch.randn(1, H, W),
            [_segment(0, [(1, 1), (2, 2)], [(1, 1)])],
            [_segment(0, [(1, 2), (2, 3)], [(2, 3)])],
            str(out))
        assert returned == str(out)
        assert out.exists() and out.stat().st_size > 0

    def test_accepts_a_two_dimensional_image(self, tmp_path):
        out = tmp_path / "fpw2d.png"
        save_segment_overlay(torch.randn(H, W), [], [], str(out))
        assert out.exists()


class TestComparisonZoom:
    def test_zoom_crops_the_model_columns_only(self, batch, tmp_path):
        """The input column keeps full extent and carries the locator box."""
        import matplotlib.pyplot as plt

        out = tmp_path / "zoom.pdf"
        render_comparison(batch["images"], [batch["logits"]], ["m"], K,
                          str(out), targets=batch["targets"], indices=[0],
                          zoom_boxes=[(4, 6, 10, 12)])
        assert out.exists() and out.stat().st_size > 0
        plt.close("all")

    def test_one_box_applies_to_every_row(self, batch, tmp_path):
        out = tmp_path / "zoom_all.pdf"
        render_comparison(batch["images"], [batch["logits"]], ["m"], K,
                          str(out), targets=batch["targets"], indices=[0, 1],
                          zoom_boxes=(4, 6, 10, 12))
        assert out.exists()

    def test_rejects_a_box_count_that_does_not_match_the_rows(self, batch, tmp_path):
        with pytest.raises(ValueError, match="zoom box"):
            render_comparison(batch["images"], [batch["logits"]], ["m"], K,
                              str(tmp_path / "bad.pdf"),
                              targets=batch["targets"], indices=[0, 1],
                              zoom_boxes=[(0, 0, 4, 4)])

    def test_zoom_changes_the_output(self, batch, tmp_path):
        """A zoom that renders identically to no zoom would be a silent no-op."""
        plain, zoomed = tmp_path / "plain.pdf", tmp_path / "zoomed.pdf"
        for path, boxes in ((plain, None), (zoomed, [(4, 6, 10, 12)])):
            render_comparison(batch["images"], [batch["logits"]], ["m"], K,
                              str(path), targets=batch["targets"],
                              indices=[0], zoom_boxes=boxes)
        assert plain.read_bytes() != zoomed.read_bytes()


class TestRenderTokensWorksForBothModelKinds:
    """render_tokens must not depend on the model's patchify/unpatchify.

    AFFSegmentation has no `patchify` at all, and its `unpatchify` reshapes with
    num_classes because it exists to turn logits into a map -- so round-tripping
    an *image* through it gives the wrong channel count. render_tokens used
    both, which meant every AFF finetune run crashed with AttributeError after
    training finished and the checkpoint was already written.
    """

    @pytest.mark.parametrize("channels", [1, 3])
    @pytest.mark.parametrize("patch", [8, 16])
    def test_the_patch_round_trip_is_exact(self, channels, patch):
        import torch

        from affmae.viz.model_figures import _patchify, _unpatchify

        image = torch.randn(2, channels, patch * 4, patch * 4)
        assert torch.equal(
            _unpatchify(_patchify(image, patch), patch, channels), image)

    def test_it_does_not_call_the_models_patchify(self):
        """Pin the fix: reaching for model.patchify reintroduces the crash."""
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1]
                  / "affmae" / "viz" / "model_figures.py").read_text()
        start = source.index("def render_tokens(")
        # Bound to this function: render_reconstruction and
        # render_vit_reconstruction legitimately call model.patchify, because a
        # MAE model has it and reconstructs in patch space.
        after = source.find("\ndef ", start + 1)
        body = source[start:after if after != -1 else len(source)]
        assert "model.patchify" not in body
        assert "model.unpatchify" not in body
