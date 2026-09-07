"""Compact backward reduction for contiguous eight-token AFF clusters."""

import torch

from .dispatch import triton, tl


def build(member: torch.Tensor):
    bsz, n_ctx, n_nb = member.shape
    if n_nb % 8:
        raise ValueError("Clustered reduction requires whole eight-token clusters")
    n_clusters = (n_ctx + 7) // 8
    ids = (member[:, :, ::8] // 8).reshape(bsz, -1).long()
    # Stable sorting preserves the generic reducer's FP32 accumulation order.
    order = torch.argsort(ids, dim=1, stable=True).to(torch.int32)
    counts = torch.zeros((bsz, n_clusters), device=member.device, dtype=torch.int32)
    counts.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.int32))
    row_ptr = torch.cat((
        torch.zeros((bsz, 1), device=member.device, dtype=torch.int32),
        counts.cumsum(1, dtype=torch.int32),
    ), dim=1)
    return dict(clustered=True, row_ptr=row_ptr, cluster_edges=order,
                n_ctx=n_ctx, n_nb=n_nb)


@triton.jit
def _reduce_cluster(Q, DO, DS, P, ROW, EDGE, DK, DV,
                    N: tl.constexpr, M: tl.constexpr, H: tl.constexpr,
                    D: tl.constexpr, NC: tl.constexpr, SCALE: tl.constexpr,
                    BD: tl.constexpr):
    cluster = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // H
    token = tl.arange(0, 8)
    dim = tl.arange(0, BD)
    dk = tl.full((8, BD), 0, tl.float32)
    dv = tl.full((8, BD), 0, tl.float32)
    start = tl.load(ROW + batch * (NC + 1) + cluster)
    end = tl.load(ROW + batch * (NC + 1) + cluster + 1)
    # Eight destinations share each reverse reference and query load.
    for edge in range(start, end):
        ref = tl.load(EDGE + batch * N * (M // 8) + edge)
        query = ref // (M // 8)
        slot = ref % (M // 8)
        flat = query * M + slot * 8 + token
        valid = cluster * 8 + token < N
        ds = tl.load(DS + bh * N * M + flat, mask=valid, other=0).to(tl.float32)
        p = tl.load(P + bh * N * M + flat, mask=valid, other=0).to(tl.float32)
        q = tl.load(Q + bh * N * D + query * D + dim, mask=dim < D, other=0).to(tl.float32)
        do = tl.load(DO + bh * N * D + query * D + dim, mask=dim < D, other=0).to(tl.float32)
        dk += ds[:, None] * q[None, :] * SCALE
        dv += p[:, None] * do[None, :]
    offset = bh * N * D + (cluster * 8 + token[:, None]) * D + dim[None, :]
    mask = (cluster * 8 + token[:, None] < N) & (dim[None, :] < D)
    tl.store(DK + offset, dk, mask)
    tl.store(DV + offset, dv, mask)


def reduce(q, do, ds, p, edge, scale, h):
    bh, n_ctx, head_dim = q.shape
    if not all(t.is_contiguous() for t in (q, do, ds, p)):
        raise ValueError("Clustered reduction requires contiguous inputs")
    dk = torch.empty_like(q, dtype=torch.float32)
    dv = torch.empty_like(q, dtype=torch.float32)
    n_clusters = triton.cdiv(n_ctx, 8)
    _reduce_cluster[(n_clusters, bh)](
        q, do, ds, p, edge["row_ptr"], edge["cluster_edges"], dk, dv,
        n_ctx, edge["n_nb"], h, head_dim, n_clusters, scale,
        triton.next_power_of_2(head_dim), num_warps=4,
    )
    return dk, dv
