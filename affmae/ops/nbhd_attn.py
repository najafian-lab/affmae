"""Neighbourhood attention over clustered tokens across various backends.

Each token attends to the ``M`` members of its cluster plus one learned blank token. ``nbhd_attn`` picks an implementation:
    nbhd_attn_torch.py: device-agnostic, autograd-capable, slow fallback
    nbhd_attn_triton.py: fused Triton, KV-owner backward
    nbhd_attn_triton_v1.py: the earlier fused Triton kernel, still selectable
    cuda_ext/clusten.py: the original AFF paper's CLUSTEN CUDA extension

All four compute the same function. The Triton kernels emit fp16, so they agree with the torch path to about 1e-3 rather than exactly.
"""

from types import MappingProxyType
from typing import Optional

import torch
import torch.nn.functional as F

__all__ = ["nbhd_attn", "BACKENDS", "resolve_backend"]

#: Canonical backend names. ``"auto"`` uses Triton when it can run on the given
#: tensors and the torch implementation otherwise.
BACKENDS = ("auto", "torch", "triton", "triton_v1", "cuda")

# Config files and checkpoints predate the canonical names. Read-only: ops/ is
# documented as holding no mutable module-level state.
_ALIASES = MappingProxyType({
    "flash_nbhd_attn": "triton",
    "flash_nbhd_attn_v1": "triton_v1",
})


def resolve_backend(backend: str) -> str:
    """Normalize a backend name, accepting the legacy config spellings.

    Args:
        backend: a name from :data:`BACKENDS`, or ``"flash_nbhd_attn"`` /
            ``"flash_nbhd_attn_v1"``.
    Returns:
        The canonical name.
    Raises:
        ValueError: on an unknown name.
    """
    canonical = _ALIASES.get(backend, backend)
    if canonical not in BACKENDS:
        raise ValueError(
            f"unknown neighbourhood-attention backend {backend!r}; expected one "
            f"of {', '.join(BACKENDS)} (or the legacy names "
            f"{', '.join(_ALIASES)}).")
    return canonical


def nbhd_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              member_idx: torch.Tensor, scale: float,
              bias: Optional[torch.Tensor] = None,
              mask: Optional[torch.Tensor] = None,
              blank_k: Optional[torch.Tensor] = None,
              blank_v: Optional[torch.Tensor] = None,
              backend: str = "auto") -> torch.Tensor:
    """Attend each token to its cluster members plus one blank token.

    Args:
        q: [B, H, N, C] queries, unscaled.
        k: [B, H, N, C] keys.
        v: [B, H, N, C] values.
        member_idx: [B, N, M] indices of each token's cluster members.
        scale: softmax temperature, normally ``C ** -0.5``.
        bias: [B, H, N, M] additive positional bias, or None.
        mask: [B, 1, N, M] or [B, H, N, M] bool, True where a member is real.
            Marks padding in the last cluster when N is not divisible by M.
        blank_k: [H, C] learned blank key, or None to omit the blank token.
        blank_v: [H, C] learned blank value.
        backend: see :data:`BACKENDS`.
    Returns:
        [B, H, N, C] in ``v``'s dtype.
    Raises:
        ValueError: on an unknown backend.
        RuntimeError: if ``backend="cuda"`` and the extension is not built.
    """
    chosen = resolve_backend(backend)
    if chosen == "auto" or chosen in ("triton", "triton_v1"):
        # Imported here, not at module scope: dispatch pulls in Triton, and
        # importing this operator must not require it.
        from .dispatch import can_use_triton

        if not can_use_triton(q, k, v):
            # No live Triton backend for these tensors -- CPU, MPS, or Triton
            # not installed. The torch path is exact to ~1e-3 of the kernels.
            chosen = "torch"
        elif chosen == "auto":
            chosen = "triton"

    if chosen == "torch":
        from .nbhd_attn_torch import neighborhood_attention

        out = neighborhood_attention(q, k, v, member_idx, float(scale),
                                     bias=bias, mask=mask,
                                     blank_k=blank_k, blank_v=blank_v)
        return out[0] if isinstance(out, tuple) else out

    member_i32 = member_idx.to(torch.int32).contiguous()

    if chosen == "triton":
        from .nbhd_attn_triton import FlashNeighborhoodAttentionFunction

        # Grad mode has to be read out here: inside a Function.forward it always
        # reads False, so the kernel cannot decide this for itself.
        return FlashNeighborhoodAttentionFunction.apply(
            q, k, v, member_i32, bias, mask, blank_k, blank_v, float(scale),
            torch.is_grad_enabled())

    if chosen == "triton_v1":
        from .nbhd_attn_triton_v1 import FlashLocalAttentionFunction

        return FlashLocalAttentionFunction.apply(
            q, k, v, member_i32, bias, mask, blank_k, blank_v, float(scale))

    return _cuda_ext_attn(q, k, v, member_idx, scale, bias, mask,
                          blank_k, blank_v)


def _cuda_ext_attn(q, k, v, member_idx, scale, bias, mask, blank_k, blank_v):
    """CLUSTEN CUDA path: QK, then bias/mask/blank softmax, then AV.

    The extension supplies only the two matmuls, so the softmax, the positional
    bias and the blank token are assembled here. Kept beside the other backends
    rather than inside the calling module, so all four share one definition of
    what this operator means.
    """
    try:
        from .cuda_ext.clusten import CLUSTENAVFunction, CLUSTENQKFunction
    except Exception as exc:
        raise RuntimeError(
            "neighbourhood attention backend 'cuda' needs the CLUSTEN CUDA "
            "extension, which is not built. Either use backend='triton' "
            "(faster), or build it:\n"
            "    cd affmae/ops/cuda_ext/src && python setup.py build_ext "
            "--inplace\n"
            "See affmae/ops/cuda_ext/README.md."
        ) from exc

    members = member_idx.to(torch.long).contiguous()
    qf = q.to(torch.float32).contiguous()
    kf = k.to(torch.float32).contiguous()
    vf = v.to(torch.float32).contiguous()

    attn = CLUSTENQKFunction.apply(qf, kf, members).to(torch.float32)
    attn = attn * float(scale)
    if bias is not None:
        attn = attn + bias.to(torch.float32)
    if mask is not None:
        attn = attn.masked_fill(~mask, -1.0e4)

    if blank_k is None:
        attn = F.softmax(attn, dim=-1)
        return CLUSTENAVFunction.apply(attn.contiguous(), vf, members).to(v.dtype)

    blank_logits = (qf * blank_k[None, :, None, :].to(torch.float32)).sum(
        -1, keepdim=True) * float(scale)
    attn = F.softmax(torch.cat([attn, blank_logits], dim=-1), dim=-1)
    out = CLUSTENAVFunction.apply(attn[..., :-1].contiguous(), vf, members)
    out = out + attn[..., -1:] * blank_v[None, :, None, :].to(torch.float32)
    return out.to(v.dtype)
