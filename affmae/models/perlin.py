""" Batched 2-D Perlin noise, used to build the MAE's masks. """

import math
import torch

__all__ = ["perlin_noise", "smooth_step"]


def smooth_step(t: torch.Tensor) -> torch.Tensor:
    """Cubic ease ``3t^2 - 2t^3`` on [0, 1].

    Note:
        Cubic, not Perlin's later quintic ``6t^5 - 15t^4 + 10t^3``. The quintic
        has vanishing second derivatives at the knots and so looks smoother, but
        the reference implementation uses the cubic and the pretrained masks
        follow from it. Swapping curves reorders the field and therefore changes
        which patches the MAE hides.
    """
    return t * t * (3.0 - 2.0 * t)


def perlin_noise(grid_shape, out_shape, batch_size: int = 1, generator=None,
                 device=None, dtype=torch.float32) -> torch.Tensor:
    """Generate a batch of 2-D Perlin noise fields.

    Conventions reproduced from the reference implementation, all of which affect
    the resulting mask:

    1. Gradients are unit vectors at uniformly random angles, drawn as one
       ``uniform_(to=2*pi)`` over a ``[B, gh + 1, gw + 1]`` tensor. Same
       generator and seed therefore give the same field.
    2. Cells are sampled at their pixel centres, ``(i + 0.5) / block``, not at
       the cell edges.
    3. Interpolation is :func:`smooth_step`, applied on x then y.
    4. The field is **not** normalized; values land roughly in [-0.7, 0.7].
       Callers that rank it do not care, and rescaling would only hide that.

    Args:
        grid_shape: (gh, gw) coarse lattice. Smaller means larger, smoother blobs.
        out_shape: (oh, ow) output resolution, each dimension divisible by the
            matching grid dimension.
        batch_size: number of independent fields.
        generator: torch.Generator for reproducibility; must live on ``device``.
        device: where to allocate.
        dtype: output dtype. Gradients are always built in float32, because the
            dot products lose too much in half precision, and cast at the end.
    Returns:
        Tensor ``[batch_size, oh, ow]``.
    Raises:
        ValueError: if a grid dimension does not divide its output dimension, or
            any dimension is not positive.
    """
    grid_h, grid_w = (int(v) for v in grid_shape)
    out_h, out_w = (int(v) for v in out_shape)
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"grid_shape must be positive, got {grid_shape}.")
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"out_shape must be positive, got {out_shape}.")
    if out_h % grid_h or out_w % grid_w:
        raise ValueError(
            f"out_shape {out_shape} must be divisible by grid_shape "
            f"{grid_shape}; got block sizes "
            f"{out_h / grid_h:.3f}x{out_w / grid_w:.3f}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    block_h, block_w = out_h // grid_h, out_w // grid_w

    # One draw over the whole gradient lattice, in this order, so a shared
    # generator reproduces the reference field exactly.
    angle = torch.empty((batch_size, grid_h + 1, grid_w + 1),
                        device=device, dtype=torch.float32)
    angle.uniform_(to=2.0 * math.pi, generator=generator)
    grad_x, grad_y = torch.cos(angle), torch.sin(angle)

    # Cell-local sample positions at pixel centres, shared by every cell.
    u = (torch.arange(block_w, device=device, dtype=torch.float32) + 0.5) / block_w
    v = (torch.arange(block_h, device=device, dtype=torch.float32) + 0.5) / block_h
    u = u.view(1, 1, 1, 1, block_w)          # [1, 1, 1, 1, bw]
    v = v.view(1, 1, 1, block_h, 1)          # [1, 1, 1, bh, 1]

    def corner(dy: int, dx: int):
        """Gradient at the cell corner offset by (dy, dx), as [B, gh, gw, 1, 1]."""
        gx = grad_x[:, dy:dy + grid_h, dx:dx + grid_w]
        gy = grad_y[:, dy:dy + grid_h, dx:dx + grid_w]
        return gx.unsqueeze(-1).unsqueeze(-1), gy.unsqueeze(-1).unsqueeze(-1)

    def ramp(dy: int, dx: int):
        """Gradient dotted with the offset from that corner to each sample."""
        gx, gy = corner(dy, dx)
        return gx * (u - dx) + gy * (v - dy)

    step_u, step_v = smooth_step(u), smooth_step(v)
    top = torch.lerp(ramp(0, 0), ramp(0, 1), step_u)
    bottom = torch.lerp(ramp(1, 0), ramp(1, 1), step_u)
    noise = torch.lerp(top, bottom, step_v)      # [B, gh, gw, bh, bw]

    # Interleave cell and within-cell axes back into a plain image.
    noise = noise.permute(0, 1, 3, 2, 4).reshape(batch_size, out_h, out_w)
    return noise.to(dtype) if dtype != torch.float32 else noise
