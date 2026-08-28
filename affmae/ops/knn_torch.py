"""PyTorch reference implementations of the KNN. """

import torch

__all__ = ["reference_dense_top4_knn"]


def reference_dense_top4_knn(database: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Exact reference implementation matching the Triton kernel semantics. Do NOT use this for production/performance.
    @TODO just use cdist + topk instead...

    database: [B, Nk, 2] int-like coordinates (x, y)
    returns:  [B, H*W, 4] int32

    Note:
        int32, not the kernel's uint16. uint16 saves two bytes per entry inside
        the Triton kernel, but MPS has no meaningful uint16 support and CUDA
        cannot even reduce over it, so the portable path pays the four bytes.
        Consumers call ``.long()`` or ``.to(torch.uint16)`` as needed.
    """
    assert database.ndim == 3 and database.shape[-1] == 2
    B, Nk, _ = database.shape
    device = database.device

    db = database.to(torch.int64)
    if Nk < 4:
        raise ValueError("reference_dense_top4_knn requires at least 4 database tokens (Nk >= 4).")
    if Nk >= (1 << 16):
        raise ValueError(
            "reference_dense_top4_knn: Nk must be < 65536 so the result stays "
            "interchangeable with the kernel's uint16 output.")
    out = torch.zeros((B, H * W, 4), dtype=torch.int32, device=device)

    # row-major grid: p = y * W + x
    ys = torch.arange(H, device=device, dtype=torch.int64)
    xs = torch.arange(W, device=device, dtype=torch.int64)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gx = gx.reshape(-1)  # [P]
    gy = gy.reshape(-1)  # [P]

    for b in range(B):
        kvx = db[b, :, 0]  # [Nk]
        kvy = db[b, :, 1]  # [Nk]

        # distances: [P, Nk]
        dx = kvx[None, :] - gx[:, None]
        dy = kvy[None, :] - gy[:, None]
        dist = dx * dx + dy * dy  # int64 safe reference

        # We want exact kernel tie behavior:
        # strict < means earlier indices win ties.
        # A stable way is to sort by (distance, index).
        idx = torch.arange(Nk, device=device, dtype=torch.int64)[None, :].expand(H * W, Nk)
        keys = dist * (Nk + 1) + idx  # lexicographic surrogate: distance first, then smaller idx
        top4 = torch.argsort(keys, dim=1)[:, :4]
        out[b, :, :] = top4.to(torch.int32)

    return out
