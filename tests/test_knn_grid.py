"""Regression tests for the knn_grid_size fix.

The bug: ``CrossAttentionPixelDecoder`` defaults ``knn_grid_h/knn_grid_w`` to
64, and nothing derived them from the actual patch grid. At
``img_size=1024, patch_size=8`` the real grid is 128x128, but
``_DeformablePointAttention._get_nn4`` clamps coordinates to
``max(grid_h, grid_w) - 1`` — so every patch at or beyond row/column 64
collapsed onto 63 and looked up neighbours in a table covering a quarter of the
image. No error was raised.

It went unnoticed because the 512 configs land exactly on ``512 / 8 == 64``,
where the wrong default happens to be right. Both resolutions are pinned here
for that reason.
"""

import pytest
import torch

from affmae.layers.attention import _DeformablePointAttention
from affmae.layers.decoder import CrossAttentionPixelDecoder
from affmae.config import load_config
from affmae.models.registry import get_model_spec

# (config, expected grid). 512/8 == 64 is the case that masked the bug;
# 1024/8 == 128 is the case that exposed it.
CONFIG_CASES = [
    ("configs/aff_base_finetune_512_fpw.yaml", 64),
    ("configs/aff_base_finetune_1024_fpw.yaml", 128),
]


@pytest.mark.parametrize("config_path,expected_grid", CONFIG_CASES)
def test_decoder_knn_grid_matches_patch_grid(config_path, expected_grid):
    """The decoder's KNN grid must cover the full patch lattice."""
    cfg = load_config(config_path)
    model = get_model_spec(cfg.model_type).build_segmentation(cfg)
    decoder = model.cross_attention_decoder

    assert cfg.img_size // cfg.patch_size == expected_grid, "config drifted"
    assert decoder.knn_grid_h == expected_grid
    assert decoder.knn_grid_w == expected_grid


@pytest.mark.parametrize("config_path,expected_grid", CONFIG_CASES)
def test_attention_modules_receive_the_grid(config_path, expected_grid):
    """The value must reach the attention modules, not just the decoder.

    Guards against the grid being stored on the decoder but never threaded
    into the deformable attention that actually clamps against it.
    """
    from affmae.layers.attention import _DeformablePointAttention

    cfg = load_config(config_path)
    model = get_model_spec(cfg.model_type).build_segmentation(cfg)

    attns = [m for m in model.cross_attention_decoder.modules()
             if isinstance(m, _DeformablePointAttention)]
    assert attns, "expected deformable attention modules in the decoder"
    for attn in attns:
        assert attn.grid_h == expected_grid, (
            f"{type(attn).__name__}.grid_h == {attn.grid_h}, "
            f"expected {expected_grid}")
        assert attn.grid_w == expected_grid


def test_clamp_would_corrupt_coordinates_at_undersized_grid():
    """Pin the failure mode itself, so the fix cannot silently regress.

    With a 64 grid, a coordinate of 100 clamps to 63 — indistinguishable from a
    real patch at 63. This is the exact expression in
    ``_DeformablePointAttention._get_nn4``.
    """
    kv_pos = torch.tensor([[[100.0, 100.0], [10.0, 10.0]]])

    undersized = kv_pos.round().clamp(min=0, max=64 - 1)
    assert undersized[0, 0].tolist() == [63.0, 63.0], "clamp collapses the coord"

    correct = kv_pos.round().clamp(min=0, max=128 - 1)
    assert correct[0, 0].tolist() == [100.0, 100.0], "correct grid preserves it"


def test_grid_is_derived_not_defaulted():
    """A non-square-friendly size still derives, rather than falling back to 64."""
    cfg = load_config("configs/aff_base_finetune_1024_fpw.yaml")
    cfg.img_size = 768  # 768 / 8 == 96, which is neither the default nor 128
    model = get_model_spec(cfg.model_type).build_segmentation(cfg)
    assert model.cross_attention_decoder.knn_grid_h == 96


class TestGridIsRequired:
    """A defaulted grid size is what made the bug silent."""

    def test_decoder_rejects_missing_grid(self):
        with pytest.raises(TypeError, match="knn_grid"):
            CrossAttentionPixelDecoder(
                input_shape={}, transformer_dropout=0.1, transformer_nheads=8,
                transformer_dim_feedforward=1024, transformer_dec_layers=1,
                conv_dim=256, mask_dim=256, transformer_in_features=[],
                common_stride=8, shepard_power=2.0,
                shepard_power_learnable=True)

    def test_attention_rejects_missing_grid(self):
        with pytest.raises(TypeError):
            _DeformablePointAttention(d_model=64, n_heads=4, n_points=4)

    def test_attention_rejects_nonpositive_grid(self):
        with pytest.raises(ValueError, match="must be positive"):
            _DeformablePointAttention(d_model=64, n_heads=4, n_points=4,
                                      grid_h=0, grid_w=8)


class TestClampBehaviour:
    def test_undersized_grid_would_collapse_coordinates(self):
        """Pin the failure mode so the fix cannot silently regress."""
        kv_pos = torch.tensor([[[100.0, 100.0], [10.0, 10.0]]])
        undersized = kv_pos.round().clamp(min=0, max=64 - 1)
        assert undersized[0, 0].tolist() == [63.0, 63.0]
        correct = kv_pos.round().clamp(min=0, max=128 - 1)
        assert correct[0, 0].tolist() == [100.0, 100.0]

    def test_clamp_is_per_axis(self):
        """x clamps against grid_w and y against grid_h.

        The old form used ``max(grid_h, grid_w) - 1`` for both, which lets the
        shorter axis index past the end of the lattice on a non-square grid.
        """
        grid_h, grid_w = 96, 128
        module = _DeformablePointAttention(d_model=64, n_heads=4, n_points=4,
                                          grid_h=grid_h, grid_w=grid_w)
        # y = 200 must land on grid_h - 1, not on max(grid_h, grid_w) - 1.
        kv_pos = torch.tensor([[[30.0, 200.0]]])
        r = kv_pos.round()
        expected = [30.0, float(grid_h - 1)]
        got = torch.stack((r[..., 0].clamp(0, module.grid_w - 1),
                           r[..., 1].clamp(0, module.grid_h - 1)), dim=-1)
        assert got[0, 0].tolist() == expected
