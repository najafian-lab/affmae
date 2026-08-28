"""Device-agnostic neighborhood (cluster) attention.

Pure PyTorch (slow), so it runs on CPU, MPS and ROCm as well as CUDA.
"""

import torch
import torch.nn as nn

__all__ = ["neighborhood_attention", "NeighborhoodAttention"]

# Bounds peak memory at chunk * M * D per head rather than N * M * D. 1024
# queries x 48 neighbours x 64 channels x 4 bytes is ~12 MB per head.
DEFAULT_CHUNK = 1024


def neighborhood_attention(q, k, v, member_index, sm_scale, bias=None,
                           mask=None, blank_k=None, blank_v=None,
                           chunk_size: int = DEFAULT_CHUNK):
    """Attend each query to its gathered neighbours plus a blank token.

    Args:
        q: [B, H, N, D] queries, unscaled.
        k: [B, H, N, D] keys.
        v: [B, H, N, D] values.
        member_index: [B, N, M] indices into the N axis of ``k``/``v``. Shared
            across heads.
        sm_scale: float, multiplies the logits before softmax.
        bias: [B, H, N, M] additive relative-position bias, or None.
        mask: [B, 1, N, M] or [B, H, N, M] boolean; False entries are excluded.
            A size-1 head dim broadcasts.
        blank_k: [H, D] learned key for the extra always-available token, or
            None to omit it.
        blank_v: [H, D] matching value.
        chunk_size: int, queries processed per step. Lower it to cut peak
            memory; it does not change the result.
    Returns:
        [B, H, N, D] attention output in ``q``'s dtype.
    Raises:
        ValueError: on inconsistent shapes.
    """
    if q.dim() != 4:
        raise ValueError(f"q must be [B, H, N, D], got {tuple(q.shape)}.")
    if member_index.dim() != 3:
        raise ValueError(
            f"member_index must be [B, N, M], got {tuple(member_index.shape)}.")

    bsz, heads, n, d = q.shape
    m = member_index.shape[-1]
    if member_index.shape[0] != bsz or member_index.shape[1] != n:
        raise ValueError(
            f"member_index {tuple(member_index.shape)} does not match q "
            f"{tuple(q.shape)} in batch/sequence.")

    out_dtype = q.dtype
    # fp16 accumulation loses too much in the softmax; fp64 is unavailable on
    # MPS. fp32 is the portable middle.
    work = torch.float32
    q32, k32, v32 = q.to(work), k.to(work), v.to(work)
    idx = member_index.long()

    has_blank = (blank_k is not None and blank_v is not None
                 and blank_k.numel() > 0 and blank_v.numel() > 0)
    if has_blank:
        bk = blank_k.to(work).view(1, heads, 1, d)
        bv = blank_v.to(work).view(1, heads, 1, d)

    mask_bool = None
    if mask is not None and mask.numel() > 0:
        mask_bool = mask.to(torch.bool)
        if mask_bool.shape[1] == 1 and heads > 1:
            mask_bool = mask_bool.expand(bsz, heads, n, m)

    chunks = []
    for start in range(0, n, max(1, chunk_size)):
        stop = min(start + chunk_size, n)
        q_c = q32[:, :, start:stop, :]                       # [B,H,c,D]
        idx_c = idx[:, start:stop, :]                        # [B,c,M]

        # Gather neighbours per (batch, head). index_select on a flattened
        # batch-head axis avoids the [B,H,N,M,D] expand the reference builds.
        flat_idx = idx_c.unsqueeze(1).expand(bsz, heads, stop - start, m)
        gather_idx = flat_idx.reshape(bsz * heads, -1, 1).expand(-1, -1, d)
        k_flat = k32.reshape(bsz * heads, n, d)
        v_flat = v32.reshape(bsz * heads, n, d)
        k_nb = torch.gather(k_flat, 1, gather_idx).view(bsz, heads, stop - start, m, d)
        v_nb = torch.gather(v_flat, 1, gather_idx).view(bsz, heads, stop - start, m, d)

        logits = (q_c.unsqueeze(3) * k_nb).sum(-1) * float(sm_scale)   # [B,H,c,M]
        if bias is not None and bias.numel() > 0:
            logits = logits + bias[:, :, start:stop, :].to(work)

        if mask_bool is not None:
            invalid = ~mask_bool[:, :, start:stop, :]
            logits = logits.masked_fill(invalid, float("-inf"))

        if has_blank:
            blank_logits = (q_c * bk).sum(-1, keepdim=True) * float(sm_scale)
            full = torch.cat([logits, blank_logits], dim=-1)
            probs = torch.softmax(full, dim=-1)
            neighbour_p = probs[..., :m]
            out_c = ((neighbour_p.unsqueeze(-1) * v_nb).sum(dim=3)
                     + probs[..., -1:] * bv)
        else:
            # Without a blank token an all-masked row has nothing to attend to;
            # softmax of all -inf is NaN, so zero it explicitly.
            all_masked = torch.isinf(logits).all(dim=-1, keepdim=True)
            safe = logits.masked_fill(all_masked, 0.0)
            probs = torch.softmax(safe, dim=-1)
            probs = probs.masked_fill(all_masked, 0.0)
            out_c = (probs.unsqueeze(-1) * v_nb).sum(dim=3)

        chunks.append(out_c)

    return torch.cat(chunks, dim=2).to(out_dtype)


class NeighborhoodAttention(nn.Module):
    """Module wrapper around :func:`neighborhood_attention`.

    Holds no parameters; it exists so the chunk size can be configured once.

    Args:
        chunk_size: int, queries processed per step.
    """

    def __init__(self, chunk_size: int = DEFAULT_CHUNK):
        super().__init__()
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
        self.chunk_size = int(chunk_size)

    def forward(self, q, k, v, member_index, sm_scale, bias=None, mask=None,
                blank_k=None, blank_v=None):
        return neighborhood_attention(
            q, k, v, member_index, sm_scale, bias=bias, mask=mask,
            blank_k=blank_k, blank_v=blank_v, chunk_size=self.chunk_size)

    def extra_repr(self) -> str:
        return f"chunk_size={self.chunk_size}"
