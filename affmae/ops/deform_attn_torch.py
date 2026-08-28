"""Device-agnostic deformable point attention.

Pure PyTorch, will use a lot more memory than the Triton version.
"""

import torch

__all__ = ["msdetrpc_reduce", "deform_point_attn_torch",
           "resolve_deform_backend"]


def resolve_deform_backend(backend: str) -> str:
    """Normalize public decoder backend names to an implementation name."""
    aliases = {
        "auto": "csr_knn_cached",
        "fused": "csr_knn_cached",
        "csr_cached": "csr_knn_cached",
    }
    resolved = aliases.get(backend, backend)
    valid = {"csr_knn_cached", "atomic", "unfused", "cuda"}
    if resolved not in valid:
        expected = "auto, fused, csr_knn_cached, csr_cached, atomic, unfused, cuda"
        raise ValueError(
            f"unknown decoder deform backend {backend!r}; expected one of "
            f"{expected}.")
    return resolved


def msdetrpc_reduce(nn_idx: torch.Tensor, nn_weight: torch.Tensor,
                    attn: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    """Reduce sampled neighbours into one vector per query.

    Computes, for every batch-head lane ``b`` and query ``n``::

        out[b, n, :] = sum_m sum_k attn[b, n, m]
                       * nn_weight[b, n, m, k] * val[b, nn_idx[b, n, m, k], :]

    Args:
        nn_idx: [B, N, M, K] indices into the token axis of ``val``.
        nn_weight: [B, N, M, K] Shepard weights over the K neighbours.
        attn: [B, N, M] attention weights over the M sampling points.
        val: [B, N_val, C] value vectors.
    Returns:
        [B, N, C] in ``val``'s dtype.
    Raises:
        ValueError: on inconsistent shapes.
    """
    if nn_idx.shape != nn_weight.shape:
        raise ValueError(
            f"nn_idx {tuple(nn_idx.shape)} and nn_weight "
            f"{tuple(nn_weight.shape)} must match.")
    if nn_idx.shape[:3] != attn.shape:
        raise ValueError(
            f"attn {tuple(attn.shape)} must be nn_idx's leading [B, N, M].")

    b, n, m, k = nn_idx.shape
    c = val.shape[-1]
    work = torch.float32

    # Combined weight per (query, point, neighbour): attention x Shepard.
    weight = attn.to(work).unsqueeze(-1) * nn_weight.to(work)      # [B,N,M,K]

    # One gather over a flattened (M*K) axis rather than an [B,N,M,K,C] expand.
    flat_idx = nn_idx.reshape(b, n * m * k, 1).expand(-1, -1, c).long()
    gathered = torch.gather(val.to(work), 1, flat_idx).view(b, n, m, k, c)

    return (weight.unsqueeze(-1) * gathered).sum(dim=(2, 3)).to(val.dtype)


def deform_point_attn_torch(query_pos: torch.Tensor, kv_pos: torch.Tensor,
                            sampling_offsets: torch.Tensor,
                            attn_logits: torch.Tensor, values: torch.Tensor,
                            tau: torch.Tensor, nn4_idx: torch.Tensor,
                            grid_h: int, grid_w: int) -> torch.Tensor:
    """Fused deformable point attention, in plain PyTorch.

    Each head samples ``S`` points around its query, each sampled location is 
    snapped to the token lattice to read its 4 precomputed neighbours, 
    those are weighted by a softmax over negative distance (Shepard interpolation with learned ``tau``), 
    and the result is weighted by a softmax over the ``S`` attention logits.

    Args:
        query_pos: [B, Nq, 2] query coordinates.
        kv_pos: [B, Nk, 2] KV coordinates.
        sampling_offsets: [B, Nq, H, S, 2] offsets from the query position.
        attn_logits: [B, Nq, H, S] unnormalized attention over sampling points.
        values: [B, Nk, H, C] value vectors.
        tau: scalar or [1] Shepard distance scale.
        nn4_idx: [B, grid_h * grid_w, 4] neighbour table, shared across heads.
        grid_h: int, lattice height.
        grid_w: int, lattice width.
    Returns:
        [B, Nq, H, C] in ``values``' dtype.
    """
    b, n_q, heads, s, _ = sampling_offsets.shape
    n_kv, channels = values.shape[1], values.shape[-1]
    work = torch.float32

    # Absolute sample locations, then the lattice cell each one falls in.
    locs = query_pos.to(work).unsqueeze(2).unsqueeze(3) + sampling_offsets.to(work)
    cell_x = locs[..., 0].round().clamp(0, grid_w - 1).long()
    cell_y = locs[..., 1].round().clamp(0, grid_h - 1).long()
    cell = cell_y * grid_w + cell_x                              # [B,Nq,H,S]

    # nn4 is shared across heads: [B, grid_h*grid_w, 4] -> per (query, head, point).
    flat_cell = cell.reshape(b, -1, 1).expand(-1, -1, 4)
    neighbours = torch.gather(nn4_idx.long(), 1, flat_cell)
    neighbours = neighbours.view(b, n_q, heads, s, 4)             # [B,Nq,H,S,4]

    # Shepard weights from distance to each of the 4 neighbours.
    nb_pos = torch.gather(
        kv_pos.to(work).unsqueeze(1).expand(b, n_q * heads * s, n_kv, 2),
        2,
        neighbours.reshape(b, n_q * heads * s, 4, 1).expand(-1, -1, -1, 2),
    ).view(b, n_q, heads, s, 4, 2)
    dist = torch.linalg.vector_norm(nb_pos - locs.unsqueeze(-2), dim=-1) + 1e-6
    power = torch.relu(tau.to(work)) + 1e-6
    shepard = torch.softmax(-power * dist, dim=-1)                # [B,Nq,H,S,4]

    attn = torch.softmax(attn_logits.to(work), dim=-1)            # [B,Nq,H,S]
    weight = attn.unsqueeze(-1) * shepard                         # [B,Nq,H,S,4]

    # Gather values per head, then reduce over (S, 4).
    vals = values.to(work).permute(0, 2, 1, 3)                    # [B,H,Nk,C]
    idx = neighbours.permute(0, 2, 1, 3, 4).reshape(b, heads, n_q * s * 4, 1)
    gathered = torch.gather(vals, 2, idx.expand(-1, -1, -1, channels))
    gathered = gathered.view(b, heads, n_q, s, 4, channels).permute(0, 2, 1, 3, 4, 5)

    out = (weight.unsqueeze(-1) * gathered).sum(dim=(3, 4))       # [B,Nq,H,C]
    return out.to(values.dtype)
