"""Perlin noise, pinned to the implementation the checkpoints were trained with.

The mask is the *rank* of this field -- ``perlin_masking`` argsorts it -- so the
field's scale is irrelevant but its ordering is not. Any change to the
interpolation curve or the gradient draw changes which patches the MAE hides,
silently and without changing a single shape.

The reference is https://github.com/tasptz/pytorch-perlin-noise. It publishes no
license, so it is neither vendored nor depended on; ours was written to match it
and verified bit-for-bit at six grid/output/batch combinations. These tests pin
the numerics with golden values so that equivalence cannot quietly lapse.
"""

import math

import pytest
import torch

from affmae.models.perlin import perlin_noise, smooth_step


class TestNumericsArePinned:
    def test_golden_values(self):
        """Exact values for a fixed seed, from the verified reference match."""
        noise = perlin_noise(grid_shape=(4, 4), out_shape=(8, 8), batch_size=1,
                            generator=torch.Generator().manual_seed(0))
        expected_row = [-0.148796, -0.266999, -0.2656, -0.06445]
        expected_col = [-0.148796, 0.060078, -0.24581, -0.263103]
        assert noise[0, 0, :4].tolist() == pytest.approx(expected_row, abs=1e-6)
        assert noise[0, :4, 0].tolist() == pytest.approx(expected_col, abs=1e-6)
        assert float(noise.sum()) == pytest.approx(0.257286, abs=1e-5)

    def test_the_step_is_cubic_not_quintic(self):
        """A quintic ease looks smoother and reorders the field.

        Perlin's own later curve is 6t^5 - 15t^4 + 10t^3; the reference uses the
        cubic, and the pretrained masks follow from it. They agree at 0, 0.5 and
        1, so only an intermediate point separates them.
        """
        t = torch.tensor([0.25])
        cubic = float(smooth_step(t))
        quintic = float(t ** 3 * (t * (t * 6 - 15) + 10))
        assert cubic == pytest.approx(0.156250, abs=1e-6)
        assert quintic == pytest.approx(0.103516, abs=1e-6)
        assert cubic != pytest.approx(quintic, abs=1e-3)

    def test_the_field_is_not_normalized(self):
        """Min-max normalizing per sample would be a monotone map, so masks would
        not change -- but it would hide that this is a raw gradient field, and
        the previous implementation did exactly that while also using a different
        curve."""
        noise = perlin_noise((8, 8), (128, 128), batch_size=4,
                            generator=torch.Generator().manual_seed(3))
        assert float(noise.min()) < 0.0
        assert float(noise.max()) < 1.0
        assert abs(float(noise.mean())) < 0.1

    def test_a_shared_generator_reproduces_the_field(self):
        kwargs = dict(grid_shape=(8, 8), out_shape=(64, 64), batch_size=3)
        first = perlin_noise(**kwargs, generator=torch.Generator().manual_seed(11))
        again = perlin_noise(**kwargs, generator=torch.Generator().manual_seed(11))
        other = perlin_noise(**kwargs, generator=torch.Generator().manual_seed(12))
        assert torch.equal(first, again)
        assert not torch.equal(first, other)


class TestShapeContract:
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_batch_dimension_is_always_present(self, batch_size):
        """The reference squeezed it away for batch_size=1.

        That is a live bug at the one call site which concatenates:
        ``perlin_masking`` uses ``group_size = ceil(N / 4)``, which is 1 for
        N=4, so four squeezed [H, W] maps would hit ``cat(dim=0)`` and stack
        along height into [4H, W] instead of [4, H, W].
        """
        noise = perlin_noise((8, 8), (64, 64), batch_size=batch_size)
        assert noise.shape == (batch_size, 64, 64)

    def test_non_square_grids_and_outputs(self):
        noise = perlin_noise((2, 8), (64, 128), batch_size=2)
        assert noise.shape == (2, 64, 128)

    def test_dtype_and_device_are_honoured(self):
        noise = perlin_noise((4, 4), (32, 32), dtype=torch.float64)
        assert noise.dtype == torch.float64

    @pytest.mark.parametrize("grid,out", [((3, 3), (64, 64)), ((8, 8), (60, 64)),
                                          ((8, 8), (64, 60))])
    def test_indivisible_shapes_are_rejected(self, grid, out):
        """Silently flooring the block size would misalign the lattice."""
        with pytest.raises(ValueError, match="divisible"):
            perlin_noise(grid, out)

    @pytest.mark.parametrize("kwargs", [
        dict(grid_shape=(0, 4), out_shape=(64, 64)),
        dict(grid_shape=(4, 4), out_shape=(0, 64)),
        dict(grid_shape=(4, 4), out_shape=(64, 64), batch_size=0),
    ])
    def test_non_positive_dimensions_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            perlin_noise(**kwargs)


class TestMaskingConsumesIt:
    @pytest.mark.parametrize("batch", [1, 2, 4, 8])
    def test_masking_partitions_every_batch_size(self, batch):
        """N=4 is the case the squeeze broke."""
        from affmae.models.masking import perlin_masking

        tokens = torch.randn(batch, 4096, 8)
        keep, masked, restore = perlin_masking(
            tokens, img_size=512, encoder_patch_size=8, mask_ratio=0.5)
        assert keep.shape == (batch, 2048)
        assert masked.shape == (batch, 2048)
        for row in range(batch):
            union = set(keep[row].tolist()) | set(masked[row].tolist())
            assert len(union) == 4096, "keep and masked must partition the grid"

    def test_the_mask_ratio_is_exact(self):
        from affmae.models.masking import perlin_mix_mask

        mask = perlin_mix_mask(out_hw=(64, 64), mask_ratio=0.25,
                               device=torch.device("cpu"), dtype=torch.float32)
        assert mask.shape == (1, 1, 64, 64)
        assert float(mask.mean()) == pytest.approx(0.25, abs=1e-6)

    def test_masking_is_spatially_clustered_not_random(self):
        """The point of Perlin masking: hidden patches form blobs.

        A uniformly random mask at ratio 0.5 gives a hidden neighbour about half
        the time. Blobs push that well above chance, and this is what a
        regression to torch.rand would destroy while every shape stayed right.
        """
        from affmae.models.masking import perlin_mix_mask

        mask = perlin_mix_mask(out_hw=(64, 64), mask_ratio=0.5,
                               device=torch.device("cpu"), dtype=torch.float32,
                               generator=torch.Generator().manual_seed(5))[0, 0]
        same = ((mask[:, :-1] == mask[:, 1:]).float().mean()
                + (mask[:-1] == mask[1:]).float().mean()) / 2
        assert float(same) > 0.85, (
            f"neighbouring patches agree only {float(same):.1%} of the time; "
            f"the mask looks uncorrelated rather than blob-like")
