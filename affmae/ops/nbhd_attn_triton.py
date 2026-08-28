''' Improved on archived kernel, by introducing kv-owner style backward (fewer atomics)

See archive.local_attn.py for the original kernel/the longer part.
'''

from collections import OrderedDict

import torch
from .dispatch import autotune_extra_kwargs, custom_bwd, custom_fwd  # compat shims
from types import MappingProxyType

from torch.autograd import Function
# Imported through util so a torch-only install can still import this
# module; the kernels raise only when actually launched.
from .dispatch import triton, tl

from .dispatch import _env_flag, clamp_num_stages, clamp_num_warps

# Off by default: the Triton autotune disk cache is a single file, so every rank
# of a multi-process run races to write it. Opt in with CACHE_TO_DISK=1 on
# single-GPU runs where the compile-time saving is worth it.
_CACHE_TO_DISK = _env_flag("CACHE_TO_DISK", "0")
#: Only newer Triton accepts cache_results; see dispatch.autotune_extra_kwargs.
_AUTOTUNE_EXTRA = autotune_extra_kwargs(_CACHE_TO_DISK)


def _configs_from_templates(block_ds, templates):
    dedup = OrderedDict()
    for bd in block_ds:
        for block_q, block_mn, warps, stages in templates:
            # Triton tl.arange(0, BLOCK_*) requires power-of-two extents.
            if (int(block_q) & (int(block_q) - 1)) != 0:
                continue
            if (int(block_mn) & (int(block_mn) - 1)) != 0:
                continue
            # Dedup on the clamped values, which are what actually runs;
            # keying on the template values left ROCm benchmarking duplicates.
            effective_warps = clamp_num_warps(int(warps), int(block_q) * int(block_mn))
            effective_stages = clamp_num_stages(int(stages))
            key = (int(block_q), int(block_mn), int(bd),
                   effective_warps, effective_stages)
            if key in dedup:
                continue
            dedup[key] = triton.Config(
                {"BLOCK_Q": key[0], "BLOCK_MN": key[1], "BLOCK_D": key[2]},
                num_warps=effective_warps,
                num_stages=effective_stages,
            )
    return list(dedup.values())


def _build_tile_configs(block_ds, include_low_m: bool = False, include_base: bool = True):
    conservative = [
        (8, 8, 2, 1), (8, 8, 2, 2),
        (8, 16, 2, 2),
        (16, 8, 2, 1), (16, 8, 4, 2),
        (16, 16, 4, 2),
        (16, 32, 4, 2),
        (32, 8, 4, 2),
    ]
    balanced = [
        (16, 16, 4, 3),
        (16, 32, 4, 3),
        (32, 8, 4, 3),
        (32, 16, 4, 3),
        (32, 16, 4, 2), (32, 16, 8, 2), (32, 16, 8, 3),
        (32, 32, 8, 2),
        (64, 8, 4, 2), (64, 8, 8, 2), (64, 8, 8, 3),
        (64, 16, 8, 2), (64, 16, 8, 3),
        (64, 32, 8, 2),
    ]
    aggressive = [
        (32, 32, 8, 2), (32, 32, 8, 3),
        (64, 16, 8, 4),
        (64, 32, 8, 2), (64, 32, 8, 3),
        (64, 64, 8, 2), (64, 64, 8, 3),
        (64, 128, 8, 2),
        (128, 8, 8, 2), (128, 16, 8, 2),
        (128, 32, 8, 2), (128, 32, 8, 3), (128, 32, 8, 4),
        (128, 64, 8, 2), (128, 64, 8, 3),
        (128, 128, 8, 2),
    ]
    low_m = [
        (16, 4, 2, 1), (16, 4, 2, 2),
        (16, 8, 4, 2),
        (32, 4, 4, 2), (32, 4, 4, 3),
        (64, 4, 8, 2),
    ]

    templates = []
    if include_base:
        templates.extend(conservative)
        templates.extend(balanced)
        templates.extend(aggressive)
    if include_low_m:
        templates.extend(low_m)
    return _configs_from_templates(block_ds, templates)


_FWD_CONFIGS = _build_tile_configs([32, 64], include_low_m=False)
_BWD_TILE_TEMPLATES = (
    (8, 8, 2, 1), (8, 8, 2, 2),
    (8, 16, 2, 2),
    (16, 8, 2, 1), (16, 8, 4, 2),
    (16, 32, 4, 2),
    (16, 16, 4, 2), (16, 16, 4, 3),
    (32, 8, 4, 2), (32, 8, 4, 3),
    (32, 16, 4, 2), (32, 16, 8, 2), (32, 16, 8, 3),
    (32, 32, 8, 2),
    (64, 8, 4, 2), (64, 8, 8, 2), (64, 8, 8, 3),
    (64, 16, 8, 2), (64, 16, 8, 3), (64, 16, 8, 4),
    (64, 32, 8, 2), (64, 32, 8, 3),
    (64, 64, 8, 2), (64, 64, 8, 3),
    (64, 128, 8, 2),
    (128, 8, 8, 2), (128, 16, 8, 2),
    (128, 32, 8, 2), (128, 32, 8, 3), (128, 32, 8, 4),
    (128, 64, 8, 2), (128, 64, 8, 3),
    (128, 128, 8, 2),
)
_BWD_CONFIGS = _configs_from_templates([32, 64], _BWD_TILE_TEMPLATES)
_BWD_LOW_M_TEMPLATES = (
    (16, 4, 2, 1),
    (16, 4, 2, 2),
    (16, 8, 4, 2),
    (32, 4, 4, 2),
    (32, 4, 4, 3),
    (64, 4, 8, 2),
)
_BWD_LOW_M_CONFIGS = _configs_from_templates([32, 64], _BWD_LOW_M_TEMPLATES)
_BWD_PREPROCESS_CONFIGS = [
    triton.Config({"BLOCK_M": bm}, num_warps=w, num_stages=clamp_num_stages(s))
    for bm in [16, 32, 64]
    for w in [1, 2, 4, 8]
    for s in [1, 2, 3]
]
_KV_OWNER_REDUCE_CONFIGS = [
    triton.Config({"BLOCK_KV": bkv, "BLOCK_D": bd}, num_warps=w,
                  num_stages=clamp_num_stages(2))
    for (bkv, w) in [(1, 1), (2, 2), (4, 4), (8, 8), (16, 8), (32, 8)]
    for bd in [32, 64]
]


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _get_named_int(named_args, kwargs, key: str, default: int = 1) -> int:
    if named_args is not None and key in named_args:
        return int(named_args[key])
    if kwargs is not None and key in kwargs:
        return int(kwargs[key])
    return int(default)


# Benchmarked across L4, RTX 3090/4090/5090, A100 80GB and GH200.
# (BLOCK_Q, BLOCK_MN, num_warps, num_stages). These may need to be tuned.
# AFFMAE_AUTOTUNE_KERNELS=1 restores the full autotune sweep.
_STATIC_TILE = MappingProxyType({
    "cuda": (32, 16, 4, 3),
    "hip": (16, 8, 2, 1),   # untested on silicon; both clamps are no-ops here
})


def _static_tile_config(head_dim: int):
    """The config used in place of autotuning. BLOCK_D must be a power of two
    and at least ``head_dim``, since the kernel does not tile the depth loop."""
    from .dispatch import is_hip

    block_q, block_mn, warps, stages = _STATIC_TILE["hip" if is_hip() else "cuda"]
    block_d = 1
    while block_d < max(int(head_dim), 1):
        block_d *= 2
    return triton.Config(
        {"BLOCK_Q": block_q, "BLOCK_MN": block_mn, "BLOCK_D": block_d},
        num_warps=clamp_num_warps(warps, block_q * block_mn),
        num_stages=clamp_num_stages(stages),
    )


def _autotune_requested() -> bool:
    """True when the caller asked for the full autotune sweep."""
    return _env_flag("AFFMAE_AUTOTUNE_KERNELS")



def _prune_tile_configs(configs, named_args=None, **kwargs):
    head_dim_static = _get_named_int(named_args, kwargs, "HEAD_DIM")
    if not _autotune_requested():
        return [_static_tile_config(head_dim_static)]
    n_ctx = _get_named_int(named_args, kwargs, "N_CTX")
    n_nb = _get_named_int(named_args, kwargs, "NEIGHBOR_SIZE")
    head_dim = _get_named_int(named_args, kwargs, "HEAD_DIM")
    candidates = []
    for cfg in configs:
        block_q = int(cfg.kwargs.get("BLOCK_Q", 0))
        block_mn = int(cfg.kwargs.get("BLOCK_MN", 0))
        block_d = int(cfg.kwargs.get("BLOCK_D", head_dim))
        if block_q <= 0 or block_mn <= 0 or block_d <= 0:
            continue
        # Lock tile dimensions to actual input dimensions.
        if block_q > n_ctx or block_mn > n_nb:
            continue
        # This kernel family does not tile the D loop; BLOCK_D must match runtime HEAD_DIM.
        if block_d != head_dim:
            continue
        candidates.append(cfg)
    if not candidates:
        # BLOCK_D == HEAD_DIM is a correctness requirement, not a tuning knob:
        # this kernel family does not tile the D loop, so a smaller BLOCK_D
        # simply never computes the remaining channels. BLOCK_Q and BLOCK_MN are
        # masked, so exceeding N_CTX/NEIGHBOR_SIZE is only wasted lanes -- they
        # are pruned above for speed, and relaxing them here is safe.
        #
        # The previous fallback minimized over all three, which silently picked
        # BLOCK_D=32 for HEAD_DIM=64 whenever the tile filters emptied the list.
        # No config offers BLOCK_MN < 8, so any NEIGHBOR_SIZE < 8 emptied it:
        # head_dim=64 with a neighbourhood of 4 returned NaN, and head_dim=32
        # was correct only because 32 happened to match. Reproduced on a GH200,
        # so this was never ROCm-specific.
        # BLOCK_D >= HEAD_DIM, not ==: the configs only offer 32 and 64, so a
        # head_dim of 8 or 16 legitimately runs in a 32-wide tile and the
        # d_off mask discards the surplus lanes. Too small is the only unsafe
        # direction.
        deep_enough = [cfg for cfg in configs
                       if int(cfg.kwargs.get("BLOCK_D", 0)) >= head_dim]
        if deep_enough:
            return [min(deep_enough,
                        key=lambda c: (int(c.kwargs.get("BLOCK_D", 1)),
                                       int(c.kwargs.get("BLOCK_Q", 1)),
                                       int(c.kwargs.get("BLOCK_MN", 1))))]
        raise ValueError(
            f"no neighbourhood-attention tile config has BLOCK_D >= HEAD_DIM "
            f"({head_dim}); available BLOCK_D: "
            f"{sorted({int(c.kwargs.get('BLOCK_D', 0)) for c in configs})}. The "
            f"kernel does not tile the depth loop, so a smaller BLOCK_D would "
            f"silently leave the remaining channels uncomputed.")
    return candidates


def _prune_preprocess_configs(configs, named_args=None, **kwargs):
    n_ctx = _get_named_int(named_args, kwargs, "N_CTX")
    valid = [cfg for cfg in configs if int(cfg.kwargs.get("BLOCK_M", 0)) <= n_ctx]
    return valid if valid else [min(configs, key=lambda c: int(c.kwargs.get("BLOCK_M", 1)))]


def _prune_kv_owner_reduce_configs(configs, named_args=None, **kwargs):
    n_ctx = _get_named_int(named_args, kwargs, "N_CTX")
    head_dim = _get_named_int(named_args, kwargs, "HEAD_DIM")
    valid = []
    for cfg in configs:
        block_kv = int(cfg.kwargs.get("BLOCK_KV", 0))
        block_d = int(cfg.kwargs.get("BLOCK_D", 0))
        if block_kv <= 0 or block_d <= 0:
            continue
        if block_kv > n_ctx or block_d > head_dim:
            continue
        valid.append(cfg)
    if valid:
        return valid
    return [min(configs, key=lambda c: (int(c.kwargs.get("BLOCK_KV", 1)), int(c.kwargs.get("BLOCK_D", 1))))]


@triton.autotune(
    configs=_BWD_PREPROCESS_CONFIGS,
    key=["N_CTX", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _prune_preprocess_configs},
    reset_to_zero=["Delta"],
    **_AUTOTUNE_EXTRA,
)
@triton.jit
def _nbhood_attn_bwd_preprocess(O, DO, Delta, N_CTX, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr):
    start_m = tl.program_id(0) * BLOCK_M
    off_bh = tl.program_id(1)
    rows_local = tl.arange(0, BLOCK_M) + start_m
    row_mask = rows_local < N_CTX
    d = tl.arange(0, HEAD_DIM)
    d_mask = d < HEAD_DIM
    rows = off_bh * N_CTX + rows_local
    o_ptr = O + rows[:, None] * HEAD_DIM + d[None, :]
    do_ptr = DO + rows[:, None] * HEAD_DIM + d[None, :]
    o = tl.load(o_ptr, mask=row_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    do = tl.load(do_ptr, mask=row_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    tl.store(Delta + off_bh * N_CTX + rows_local, tl.sum(o * do, axis=1), mask=row_mask)


@triton.autotune(
    configs=_FWD_CONFIGS,
    key=["N_CTX", "NEIGHBOR_SIZE", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _prune_tile_configs},
    **_AUTOTUNE_EXTRA,
)
@triton.jit
def _nbhood_attn_fwd_kernel(
    Q, K, V, MEMBER_IDX, BIAS, MASK, BLANK_K, BLANK_V, OUT, LSE,
    N_CTX, NEIGHBOR_SIZE, HEAD_DIM, H, sm_scale,
    stride_q_bh, stride_q_n, stride_q_d,
    stride_k_bh, stride_k_n, stride_k_d,
    stride_v_bh, stride_v_n, stride_v_d,
    stride_mi_b, stride_mi_q, stride_mi_m,
    stride_bi_bh, stride_bi_q, stride_bi_m,
    stride_ma_bh, stride_ma_q, stride_ma_m,
    stride_bk_h, stride_bk_d,
    stride_bv_h, stride_bv_d,
    stride_o_bh, stride_o_q, stride_o_d,
    stride_lse_bh, stride_lse_q,
    BLOCK_Q: tl.constexpr,
    BLOCK_MN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_BLANK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_qt = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    q_local = tl.arange(0, BLOCK_Q)
    q_off = pid_qt * BLOCK_Q + q_local
    q_mask = q_off < N_CTX
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < HEAD_DIM
    q_ptr = Q + pid_bh * stride_q_bh + q_off[:, None] * stride_q_n + d_off[None, :] * stride_q_d
    q_tile = tl.load(q_ptr, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    m_i = tl.full([BLOCK_Q], -1.0e20, tl.float32)
    l_i = tl.zeros([BLOCK_Q], tl.float32)
    acc = tl.zeros([BLOCK_Q, BLOCK_D], tl.float32)
    mn_local = tl.arange(0, BLOCK_MN)
    n_off = 0
    while n_off < NEIGHBOR_SIZE:
        mn_abs = n_off + mn_local
        mn_mask = mn_abs < NEIGHBOR_SIZE
        idx_ptr = MEMBER_IDX + pid_b * stride_mi_b + q_off[:, None] * stride_mi_q + mn_abs[None, :] * stride_mi_m
        idx = tl.load(idx_ptr, mask=q_mask[:, None] & mn_mask[None, :], other=0).to(tl.int32)
        k_ptr = K + pid_bh * stride_k_bh + idx[:, :, None] * stride_k_n + d_off[None, None, :] * stride_k_d
        v_ptr = V + pid_bh * stride_v_bh + idx[:, :, None] * stride_v_n + d_off[None, None, :] * stride_v_d
        k_tile = tl.load(k_ptr, mask=q_mask[:, None, None] & mn_mask[None, :, None] & d_mask[None, None, :], other=0.0).to(tl.float32)
        v_tile = tl.load(v_ptr, mask=q_mask[:, None, None] & mn_mask[None, :, None] & d_mask[None, None, :], other=0.0).to(tl.float32)
        logits = tl.sum(q_tile[:, None, :] * k_tile, axis=2) * sm_scale
        logits = tl.where(q_mask[:, None] & mn_mask[None, :], logits, -1.0e20)
        if HAS_BIAS:
            b_ptr = BIAS + pid_bh * stride_bi_bh + q_off[:, None] * stride_bi_q + mn_abs[None, :] * stride_bi_m
            logits += tl.load(b_ptr, mask=q_mask[:, None] & mn_mask[None, :], other=0.0).to(tl.float32)
        if HAS_MASK:
            ma_ptr = MASK + pid_bh * stride_ma_bh + q_off[:, None] * stride_ma_q + mn_abs[None, :] * stride_ma_m
            ma = tl.load(ma_ptr, mask=q_mask[:, None] & mn_mask[None, :], other=0).to(tl.int32)
            logits = tl.where(ma != 0, logits, -1.0e20)
        m_chunk = tl.max(logits, axis=1)
        m_new = tl.maximum(m_i, m_chunk)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(logits - m_new[:, None])
        p = tl.where(q_mask[:, None] & mn_mask[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v_tile, axis=1)
        m_i = m_new
        n_off += BLOCK_MN
    if HAS_BLANK:
        bk_ptr = BLANK_K + pid_h * stride_bk_h + d_off * stride_bk_d
        bv_ptr = BLANK_V + pid_h * stride_bv_h + d_off * stride_bv_d
        bk = tl.load(bk_ptr, mask=d_mask, other=0.0).to(tl.float32)
        bv = tl.load(bv_ptr, mask=d_mask, other=0.0).to(tl.float32)
        blank_logits = tl.sum(q_tile * bk[None, :], axis=1) * sm_scale
        blank_logits = tl.where(q_mask, blank_logits, -1.0e20)
        m_new = tl.maximum(m_i, blank_logits)
        alpha = tl.exp(m_i - m_new)
        pb = tl.exp(blank_logits - m_new)
        pb = tl.where(q_mask, pb, 0.0)
        l_i = l_i * alpha + pb
        acc = acc * alpha[:, None] + pb[:, None] * bv[None, :]
        m_i = m_new
    l_i = tl.maximum(l_i, 1e-20)
    out_tile = acc / l_i[:, None]
    o_ptr = OUT + pid_bh * stride_o_bh + q_off[:, None] * stride_o_q + d_off[None, :] * stride_o_d
    tl.store(o_ptr, out_tile.to(tl.float16), mask=q_mask[:, None] & d_mask[None, :])
    tl.store(LSE + pid_bh * stride_lse_bh + q_off * stride_lse_q, m_i + tl.log(l_i), mask=q_mask)


@triton.autotune(
    configs=_BWD_CONFIGS,
    key=["N_CTX", "NEIGHBOR_SIZE", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _prune_tile_configs},
    reset_to_zero=["DQ", "DK", "DV", "DBIAS", "DBK", "DBV"],
    **_AUTOTUNE_EXTRA,
)
@triton.jit
def _nbhood_attn_bwd_kernel(
    Q, K, V, DO, LSE, DELTA, MEMBER_IDX, BIAS, MASK, BLANK_K, BLANK_V,
    DQ, DK, DV, DBIAS, DBK, DBV,
    N_CTX, NEIGHBOR_SIZE, HEAD_DIM, H, sm_scale,
    stride_q_bh, stride_q_n, stride_q_d,
    stride_k_bh, stride_k_n, stride_k_d,
    stride_v_bh, stride_v_n, stride_v_d,
    stride_do_bh, stride_do_n, stride_do_d,
    stride_lse_bh, stride_lse_n,
    stride_delta_bh, stride_delta_n,
    stride_mi_b, stride_mi_q, stride_mi_m,
    stride_bi_bh, stride_bi_q, stride_bi_m,
    stride_ma_bh, stride_ma_q, stride_ma_m,
    stride_bk_h, stride_bk_d,
    stride_bv_h, stride_bv_d,
    stride_dq_bh, stride_dq_n, stride_dq_d,
    stride_dk_bh, stride_dk_n, stride_dk_d,
    stride_dv_bh, stride_dv_n, stride_dv_d,
    stride_db_bh, stride_db_n, stride_db_m,
    BLOCK_Q: tl.constexpr,
    BLOCK_MN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_BLANK: tl.constexpr,
    ENABLE_ATOMICS: tl.constexpr,
):
    pid_qt = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    q_local = tl.arange(0, BLOCK_Q)
    q_off = pid_qt * BLOCK_Q + q_local
    q_mask = q_off < N_CTX
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < HEAD_DIM
    q_ptr = Q + pid_bh * stride_q_bh + q_off[:, None] * stride_q_n + d_off[None, :] * stride_q_d
    do_ptr = DO + pid_bh * stride_do_bh + q_off[:, None] * stride_do_n + d_off[None, :] * stride_do_d
    q_tile = tl.load(q_ptr, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    do_tile = tl.load(do_ptr, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    lse = tl.load(LSE + pid_bh * stride_lse_bh + q_off * stride_lse_n, mask=q_mask, other=-1.0e20).to(tl.float32)
    delta = tl.load(DELTA + pid_bh * stride_delta_bh + q_off * stride_delta_n, mask=q_mask, other=0.0).to(tl.float32)
    dq_acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)
    mn_local = tl.arange(0, BLOCK_MN)
    n_off = 0
    while n_off < NEIGHBOR_SIZE:
        mn_abs = n_off + mn_local
        mn_mask = mn_abs < NEIGHBOR_SIZE
        idx_ptr = MEMBER_IDX + pid_b * stride_mi_b + q_off[:, None] * stride_mi_q + mn_abs[None, :] * stride_mi_m
        idx = tl.load(idx_ptr, mask=q_mask[:, None] & mn_mask[None, :], other=0).to(tl.int32)
        k_ptr = K + pid_bh * stride_k_bh + idx[:, :, None] * stride_k_n + d_off[None, None, :] * stride_k_d
        v_ptr = V + pid_bh * stride_v_bh + idx[:, :, None] * stride_v_n + d_off[None, None, :] * stride_v_d
        k_tile = tl.load(k_ptr, mask=q_mask[:, None, None] & mn_mask[None, :, None] & d_mask[None, None, :], other=0.0).to(tl.float32)
        v_tile = tl.load(v_ptr, mask=q_mask[:, None, None] & mn_mask[None, :, None] & d_mask[None, None, :], other=0.0).to(tl.float32)
        logits = tl.sum(q_tile[:, None, :] * k_tile, axis=2) * sm_scale
        logits = tl.where(q_mask[:, None] & mn_mask[None, :], logits, -1.0e20)
        if HAS_BIAS:
            b_ptr = BIAS + pid_bh * stride_bi_bh + q_off[:, None] * stride_bi_q + mn_abs[None, :] * stride_bi_m
            logits += tl.load(b_ptr, mask=q_mask[:, None] & mn_mask[None, :], other=0.0).to(tl.float32)
        if HAS_MASK:
            ma_ptr = MASK + pid_bh * stride_ma_bh + q_off[:, None] * stride_ma_q + mn_abs[None, :] * stride_ma_m
            ma = tl.load(ma_ptr, mask=q_mask[:, None] & mn_mask[None, :], other=0).to(tl.int32)
            logits = tl.where(ma != 0, logits, -1.0e20)
        p = tl.exp(logits - lse[:, None])
        p = tl.where(q_mask[:, None] & mn_mask[None, :], p, 0.0)
        dp = tl.sum(do_tile[:, None, :] * v_tile, axis=2)
        ds = tl.where(q_mask[:, None] & mn_mask[None, :], p * (dp - delta[:, None]), 0.0)
        dq_acc += tl.sum(ds[:, :, None] * k_tile, axis=1)
        dk_add = ds[:, :, None] * q_tile[:, None, :] * sm_scale
        dv_add = p[:, :, None] * do_tile[:, None, :]
        scatter_mask = q_mask[:, None, None] & mn_mask[None, :, None] & d_mask[None, None, :]
        dk_ptr = DK + pid_bh * stride_dk_bh + idx[:, :, None] * stride_dk_n + d_off[None, None, :] * stride_dk_d
        dv_ptr = DV + pid_bh * stride_dv_bh + idx[:, :, None] * stride_dv_n + d_off[None, None, :] * stride_dv_d
        if ENABLE_ATOMICS:
            tl.atomic_add(dk_ptr, dk_add, mask=scatter_mask)
            tl.atomic_add(dv_ptr, dv_add, mask=scatter_mask)
        if HAS_BIAS:
            db_ptr = DBIAS + pid_bh * stride_db_bh + q_off[:, None] * stride_db_n + mn_abs[None, :] * stride_db_m
            tl.store(db_ptr, ds, mask=q_mask[:, None] & mn_mask[None, :])
        n_off += BLOCK_MN
    if HAS_BLANK:
        bk_ptr = BLANK_K + pid_h * stride_bk_h + d_off * stride_bk_d
        bv_ptr = BLANK_V + pid_h * stride_bv_h + d_off * stride_bv_d
        bk = tl.load(bk_ptr, mask=d_mask, other=0.0).to(tl.float32)
        bv = tl.load(bv_ptr, mask=d_mask, other=0.0).to(tl.float32)
        blank_logits = tl.sum(q_tile * bk[None, :], axis=1) * sm_scale
        blank_logits = tl.where(q_mask, blank_logits, -1.0e20)
        p_b = tl.where(q_mask, tl.exp(blank_logits - lse), 0.0)
        dp_b = tl.where(q_mask, tl.sum(do_tile * bv[None, :], axis=1), 0.0)
        ds_b = p_b * (dp_b - delta)
        dq_acc += ds_b[:, None] * bk[None, :]
        if ENABLE_ATOMICS:
            tl.atomic_add(DBK + pid_h * stride_bk_h + d_off * stride_bk_d, tl.sum((ds_b[:, None] * q_tile * sm_scale), axis=0), mask=d_mask)
            tl.atomic_add(DBV + pid_h * stride_bv_h + d_off * stride_bv_d, tl.sum((p_b[:, None] * do_tile), axis=0), mask=d_mask)
    dq_ptr = DQ + pid_bh * stride_dq_bh + q_off[:, None] * stride_dq_n + d_off[None, :] * stride_dq_d
    tl.store(dq_ptr, (dq_acc * sm_scale).to(tl.float32), mask=q_mask[:, None] & d_mask[None, :])


@triton.autotune(
    configs=_BWD_CONFIGS,
    key=["N_CTX", "NEIGHBOR_SIZE", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _prune_tile_configs},
    reset_to_zero=["DQ", "DBIAS", "DBK", "DBV", "DS_FLAT"],
    **_AUTOTUNE_EXTRA,
)
@triton.jit
def _nbhood_attn_bwd_core_kernel(
    Q, K, V, DO, LSE, DELTA, MEMBER_IDX, BIAS, MASK, BLANK_K, BLANK_V,
    DQ, DBIAS, DBK, DBV, DS_FLAT, P_FLAT,
    N_CTX, NEIGHBOR_SIZE, HEAD_DIM, H, sm_scale,
    stride_q_bh, stride_q_n, stride_q_d,
    stride_k_bh, stride_k_n, stride_k_d,
    stride_v_bh, stride_v_n, stride_v_d,
    stride_do_bh, stride_do_n, stride_do_d,
    stride_lse_bh, stride_lse_n,
    stride_delta_bh, stride_delta_n,
    stride_mi_b, stride_mi_q, stride_mi_m,
    stride_bi_bh, stride_bi_q, stride_bi_m,
    stride_ma_bh, stride_ma_q, stride_ma_m,
    stride_bk_h, stride_bk_d,
    stride_bv_h, stride_bv_d,
    stride_dq_bh, stride_dq_n, stride_dq_d,
    stride_db_bh, stride_db_n, stride_db_m,
    stride_ds_bh, stride_ds_nm,
    stride_p_bh, stride_p_nm,
    BLOCK_Q: tl.constexpr,
    BLOCK_MN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_BLANK: tl.constexpr,
    STORE_INTERMED_FP16: tl.constexpr,
):
    pid_qt = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    q_local = tl.arange(0, BLOCK_Q)
    q_off = pid_qt * BLOCK_Q + q_local
    q_mask = q_off < N_CTX
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < HEAD_DIM
    q_ptr = Q + pid_bh * stride_q_bh + q_off[:, None] * stride_q_n + d_off[None, :] * stride_q_d
    do_ptr = DO + pid_bh * stride_do_bh + q_off[:, None] * stride_do_n + d_off[None, :] * stride_do_d
    q_tile = tl.load(q_ptr, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    do_tile = tl.load(do_ptr, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    lse = tl.load(LSE + pid_bh * stride_lse_bh + q_off * stride_lse_n, mask=q_mask, other=-1.0e20).to(tl.float32)
    delta = tl.load(DELTA + pid_bh * stride_delta_bh + q_off * stride_delta_n, mask=q_mask, other=0.0).to(tl.float32)
    dq_acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)
    mn_local = tl.arange(0, BLOCK_MN)
    n_off = 0
    while n_off < NEIGHBOR_SIZE:
        mn_abs = n_off + mn_local
        mn_mask = mn_abs < NEIGHBOR_SIZE
        idx_ptr = MEMBER_IDX + pid_b * stride_mi_b + q_off[:, None] * stride_mi_q + mn_abs[None, :] * stride_mi_m
        idx = tl.load(idx_ptr, mask=q_mask[:, None] & mn_mask[None, :], other=0).to(tl.int32)
        k_ptr = K + pid_bh * stride_k_bh + idx[:, :, None] * stride_k_n + d_off[None, None, :] * stride_k_d
        v_ptr = V + pid_bh * stride_v_bh + idx[:, :, None] * stride_v_n + d_off[None, None, :] * stride_v_d
        k_tile = tl.load(k_ptr, mask=q_mask[:, None, None] & mn_mask[None, :, None] & d_mask[None, None, :], other=0.0).to(tl.float32)
        v_tile = tl.load(v_ptr, mask=q_mask[:, None, None] & mn_mask[None, :, None] & d_mask[None, None, :], other=0.0).to(tl.float32)
        logits = tl.sum(q_tile[:, None, :] * k_tile, axis=2) * sm_scale
        logits = tl.where(q_mask[:, None] & mn_mask[None, :], logits, -1.0e20)
        if HAS_BIAS:
            b_ptr = BIAS + pid_bh * stride_bi_bh + q_off[:, None] * stride_bi_q + mn_abs[None, :] * stride_bi_m
            logits += tl.load(b_ptr, mask=q_mask[:, None] & mn_mask[None, :], other=0.0).to(tl.float32)
        if HAS_MASK:
            ma_ptr = MASK + pid_bh * stride_ma_bh + q_off[:, None] * stride_ma_q + mn_abs[None, :] * stride_ma_m
            ma = tl.load(ma_ptr, mask=q_mask[:, None] & mn_mask[None, :], other=0).to(tl.int32)
            logits = tl.where(ma != 0, logits, -1.0e20)
        p = tl.where(q_mask[:, None] & mn_mask[None, :], tl.exp(logits - lse[:, None]), 0.0)
        ds = tl.where(q_mask[:, None] & mn_mask[None, :], p * (tl.sum(do_tile[:, None, :] * v_tile, axis=2) - delta[:, None]), 0.0)
        dq_acc += tl.sum(ds[:, :, None] * k_tile, axis=1)
        flat = q_off[:, None] * NEIGHBOR_SIZE + mn_abs[None, :]
        ds_ptr = DS_FLAT + pid_bh * stride_ds_bh + flat * stride_ds_nm
        p_ptr = P_FLAT + pid_bh * stride_p_bh + flat * stride_p_nm
        if STORE_INTERMED_FP16:
            tl.store(ds_ptr, ds.to(tl.float16), mask=q_mask[:, None] & mn_mask[None, :])
            tl.store(p_ptr, p.to(tl.float16), mask=q_mask[:, None] & mn_mask[None, :])
        else:
            tl.store(ds_ptr, ds, mask=q_mask[:, None] & mn_mask[None, :])
            tl.store(p_ptr, p, mask=q_mask[:, None] & mn_mask[None, :])
        if HAS_BIAS:
            db_ptr = DBIAS + pid_bh * stride_db_bh + q_off[:, None] * stride_db_n + mn_abs[None, :] * stride_db_m
            tl.store(db_ptr, ds, mask=q_mask[:, None] & mn_mask[None, :])
        n_off += BLOCK_MN
    if HAS_BLANK:
        bk_ptr = BLANK_K + pid_h * stride_bk_h + d_off * stride_bk_d
        bv_ptr = BLANK_V + pid_h * stride_bv_h + d_off * stride_bv_d
        bk = tl.load(bk_ptr, mask=d_mask, other=0.0).to(tl.float32)
        bv = tl.load(bv_ptr, mask=d_mask, other=0.0).to(tl.float32)
        blank_logits = tl.sum(q_tile * bk[None, :], axis=1) * sm_scale
        blank_logits = tl.where(q_mask, blank_logits, -1.0e20)
        p_b = tl.where(q_mask, tl.exp(blank_logits - lse), 0.0)
        ds_b = p_b * (tl.where(q_mask, tl.sum(do_tile * bv[None, :], axis=1), 0.0) - delta)
        dq_acc += ds_b[:, None] * bk[None, :]
        tl.atomic_add(DBK + pid_h * stride_bk_h + d_off * stride_bk_d, tl.sum((ds_b[:, None] * q_tile * sm_scale), axis=0), mask=d_mask)
        tl.atomic_add(DBV + pid_h * stride_bv_h + d_off * stride_bv_d, tl.sum((p_b[:, None] * do_tile), axis=0), mask=d_mask)
    dq_ptr = DQ + pid_bh * stride_dq_bh + q_off[:, None] * stride_dq_n + d_off[None, :] * stride_dq_d
    tl.store(dq_ptr, (dq_acc * sm_scale).to(tl.float32), mask=q_mask[:, None] & d_mask[None, :])


@triton.autotune(
    configs=_BWD_CONFIGS + _BWD_LOW_M_CONFIGS,
    key=["N_CTX", "NEIGHBOR_SIZE", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _prune_tile_configs},
    reset_to_zero=["DK", "DV"],
    **_AUTOTUNE_EXTRA,
)
@triton.jit
def _nbhood_attn_bwd_reduce_kernel(
    Q, DO, MEMBER_IDX, DS_FLAT, P_FLAT, DK, DV,
    N_CTX, NEIGHBOR_SIZE, HEAD_DIM, H, sm_scale,
    stride_q_bh, stride_q_n, stride_q_d,
    stride_do_bh, stride_do_n, stride_do_d,
    stride_mi_b, stride_mi_q, stride_mi_m,
    stride_ds_bh, stride_ds_nm,
    stride_p_bh, stride_p_nm,
    stride_dk_bh, stride_dk_n, stride_dk_d,
    stride_dv_bh, stride_dv_n, stride_dv_d,
    BLOCK_Q: tl.constexpr,
    BLOCK_MN: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_qt = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    q_local = tl.arange(0, BLOCK_Q)
    q_off = pid_qt * BLOCK_Q + q_local
    q_mask = q_off < N_CTX
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < HEAD_DIM
    q_ptr = Q + pid_bh * stride_q_bh + q_off[:, None] * stride_q_n + d_off[None, :] * stride_q_d
    do_ptr = DO + pid_bh * stride_do_bh + q_off[:, None] * stride_do_n + d_off[None, :] * stride_do_d
    q_tile = tl.load(q_ptr, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    do_tile = tl.load(do_ptr, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    mn_local = tl.arange(0, BLOCK_MN)
    n_off = 0
    while n_off < NEIGHBOR_SIZE:
        mn_abs = n_off + mn_local
        mn_mask = mn_abs < NEIGHBOR_SIZE
        idx_ptr = MEMBER_IDX + pid_b * stride_mi_b + q_off[:, None] * stride_mi_q + mn_abs[None, :] * stride_mi_m
        idx = tl.load(idx_ptr, mask=q_mask[:, None] & mn_mask[None, :], other=0).to(tl.int32)
        flat = q_off[:, None] * NEIGHBOR_SIZE + mn_abs[None, :]
        ds = tl.load(DS_FLAT + pid_bh * stride_ds_bh + flat * stride_ds_nm, mask=q_mask[:, None] & mn_mask[None, :], other=0.0).to(tl.float32)
        p = tl.load(P_FLAT + pid_bh * stride_p_bh + flat * stride_p_nm, mask=q_mask[:, None] & mn_mask[None, :], other=0.0).to(tl.float32)
        dk_add = ds[:, :, None] * q_tile[:, None, :] * sm_scale
        dv_add = p[:, :, None] * do_tile[:, None, :]
        mask3 = q_mask[:, None, None] & mn_mask[None, :, None] & d_mask[None, None, :]
        dk_ptr = DK + pid_bh * stride_dk_bh + idx[:, :, None] * stride_dk_n + d_off[None, None, :] * stride_dk_d
        dv_ptr = DV + pid_bh * stride_dv_bh + idx[:, :, None] * stride_dv_n + d_off[None, None, :] * stride_dv_d
        tl.atomic_add(dk_ptr, dk_add, mask=mask3)
        tl.atomic_add(dv_ptr, dv_add, mask=mask3)
        n_off += BLOCK_MN


@triton.autotune(
    configs=_KV_OWNER_REDUCE_CONFIGS,
    key=["N_CTX", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _prune_kv_owner_reduce_configs},
    **_AUTOTUNE_EXTRA,
)
@triton.jit
def _nbhood_attn_kv_owner_reduce_kernel(
    Q, DO, DS_FLAT, P_FLAT, ROW_PTR, EDGE_FLAT, KV_ORDER, DK, DV,
    N_CTX, NEIGHBOR_SIZE, HEAD_DIM, H,
    sm_scale,
    stride_q_bh, stride_q_n, stride_q_d,
    stride_do_bh, stride_do_n, stride_do_d,
    stride_ds_bh, stride_ds_nm,
    stride_p_bh, stride_p_nm,
    stride_rp_b, stride_rp_n,
    stride_ef_b, stride_ef_e,
    stride_ko_b, stride_ko_n,
    stride_dk_bh, stride_dk_n, stride_dk_d,
    stride_dv_bh, stride_dv_n, stride_dv_d,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_kv_tile = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_dt = tl.program_id(2)
    pid_b = pid_bh // H

    d_off = pid_dt * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < HEAD_DIM

    for ki in tl.static_range(0, BLOCK_KV):
        kv_lin = pid_kv_tile * BLOCK_KV + ki
        valid_kv = kv_lin < N_CTX
        kv_i = tl.load(KV_ORDER + pid_b * stride_ko_b + kv_lin * stride_ko_n, mask=valid_kv, other=0)
        dk_row = tl.zeros([BLOCK_D], dtype=tl.float32)
        dv_row = tl.zeros([BLOCK_D], dtype=tl.float32)
        start = tl.load(ROW_PTR + pid_b * stride_rp_b + kv_i * stride_rp_n, mask=valid_kv, other=0)
        end = tl.load(ROW_PTR + pid_b * stride_rp_b + (kv_i + 1) * stride_rp_n, mask=valid_kv, other=0)
        e = start
        while e < end:
            flat_idx = tl.load(EDGE_FLAT + pid_b * stride_ef_b + e * stride_ef_e)
            q_idx = flat_idx // NEIGHBOR_SIZE
            ds = tl.load(DS_FLAT + pid_bh * stride_ds_bh + flat_idx * stride_ds_nm).to(tl.float32)
            p = tl.load(P_FLAT + pid_bh * stride_p_bh + flat_idx * stride_p_nm).to(tl.float32)
            q_ptr = Q + pid_bh * stride_q_bh + q_idx * stride_q_n + d_off * stride_q_d
            do_ptr = DO + pid_bh * stride_do_bh + q_idx * stride_do_n + d_off * stride_do_d
            q_vec = tl.load(q_ptr, mask=d_mask, other=0.0).to(tl.float32)
            do_vec = tl.load(do_ptr, mask=d_mask, other=0.0).to(tl.float32)
            dk_row += ds * q_vec * sm_scale
            dv_row += p * do_vec
            e += 1

        dk_ptr = DK + pid_bh * stride_dk_bh + kv_i * stride_dk_n + d_off * stride_dk_d
        dv_ptr = DV + pid_bh * stride_dv_bh + kv_i * stride_dv_n + d_off * stride_dv_d
        tl.store(dk_ptr, dk_row, mask=d_mask & valid_kv)
        tl.store(dv_ptr, dv_row, mask=d_mask & valid_kv)


def _estimate_dup_ratio(member_index: torch.Tensor, max_rows: int = 1024) -> float:
    flat = member_index.reshape(-1, member_index.shape[-1])
    if flat.shape[0] > max_rows:
        step = max(1, flat.shape[0] // max_rows)
        flat = flat[::step][:max_rows]
    vals = flat.reshape(-1)
    return float(vals.numel()) / float(max(torch.unique(vals).numel(), 1))


def _compute_collision_stats(member_index: torch.Tensor, sample_rows: int = 1024, block_q: int = 16):
    bsz, n_ctx, n_nb = member_index.shape
    flat = member_index.reshape(-1, n_nb)
    if flat.shape[0] > sample_rows:
        step = max(1, flat.shape[0] // sample_rows)
        flat = flat[::step][:sample_rows]
    uniq_per_row = torch.tensor(
        [torch.unique(flat[i]).numel() for i in range(flat.shape[0])],
        device=flat.device,
        dtype=torch.float32,
    )
    dup_ratio = float((flat.numel()) / max(torch.unique(flat.reshape(-1)).numel(), 1))
    # Block-level unique count is a stronger proxy for atomic collision pressure.
    block_uniques = []
    for b in range(bsz):
        for q0 in range(0, n_ctx, block_q):
            q1 = min(n_ctx, q0 + block_q)
            block_uniques.append(torch.unique(member_index[b, q0:q1].reshape(-1)).numel())
    block_uniques_t = torch.tensor(block_uniques, device=member_index.device, dtype=torch.float32)
    return {
        "dup_ratio": dup_ratio,
        "row_unique_mean": float(uniq_per_row.mean().item()) if uniq_per_row.numel() > 0 else 0.0,
        "row_unique_p90": float(torch.quantile(uniq_per_row, 0.9).item()) if uniq_per_row.numel() > 0 else 0.0,
        "block_unique_mean": float(block_uniques_t.mean().item()) if block_uniques_t.numel() > 0 else 0.0,
        "block_unique_p90": float(torch.quantile(block_uniques_t, 0.9).item()) if block_uniques_t.numel() > 0 else 0.0,
    }


class KVEdgeIndexCache:
    """LRU cache of KV-owner edge indices, keyed by cluster membership.

    The backward pass needs, for each KV token, the list of queries that attended to
    it. Building that is a sort and a scatter over [B, N, M]; every block in an
    encoder stage shares one membership, so the index is built once per stage.
    ``affmae/layers/aff.py`` installs one of these per stage.

    Args:
        max_entries: LRU bound. Beyond it the least recently used entry is dropped.
    """
    def __init__(self, max_entries: int = 16):
        self.max_entries = int(max(1, max_entries))
        self._cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def clear(self):
        """Drop every entry.
        """
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def _make_key(self, member_index: torch.Tensor):
        mi = member_index
        return (
            str(mi.device),
            tuple(mi.shape),
            tuple(mi.stride()),
            int(mi.data_ptr()),
        )

    def get(self, member_index: torch.Tensor):
        """Look up a cached index.

        Args:
            member_index: [B, N, M] cluster membership.
        Returns:
            The cached edge index, or None on a miss.
        """
        key = self._make_key(member_index)
        value = self._cache.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self._cache.move_to_end(key)
        return value

    def put(self, member_index: torch.Tensor, edge_index: dict):
        """Store an edge index, evicting the least recently used if over bound.

        Args:
            member_index: [B, N, M] cluster membership.
            edge_index: the built index.
        """
        key = self._make_key(member_index)
        self._cache[key] = edge_index
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def get_or_build(self, member_index: torch.Tensor):
        """Return the cached index for ``member_index``, building it on a miss.

        Args:
            member_index: [B, N, M] cluster membership.
        Returns:
            The edge index.
        """
        cached = self.get(member_index)
        if cached is not None:
            return cached
        built = _build_kv_edge_index(member_index)
        self.put(member_index, built)
        return built

    def get_or_build_with_hit(self, member_index: torch.Tensor):
        """As :meth:`get_or_build`, also reporting whether it was a hit.

        Args:
            member_index: [B, N, M] cluster membership.
        Returns:
            ``(edge_index, was_hit)``.
        """
        cached = self.get(member_index)
        if cached is not None:
            return cached, True
        built = _build_kv_edge_index(member_index)
        self.put(member_index, built)
        return built, False


def _resolve_kv_edge_index(
    member_index: torch.Tensor,
    kv_edge_index: dict = None,
    kv_edge_cache: KVEdgeIndexCache = None,
    validate: bool = True,
):
    cache_hit = False
    if kv_edge_index is not None:
        kv_edge_index = _ensure_kv_edge_index_fields(kv_edge_index)
        if validate:
            _validate_kv_edge_index(kv_edge_index, member_index)
        return kv_edge_index, cache_hit
    if kv_edge_cache is not None:
        edge_index, cache_hit = kv_edge_cache.get_or_build_with_hit(member_index)
        edge_index = _ensure_kv_edge_index_fields(edge_index)
        if validate:
            _validate_kv_edge_index(edge_index, member_index)
        return edge_index, cache_hit
    edge_index = _build_kv_edge_index(member_index)
    edge_index = _ensure_kv_edge_index_fields(edge_index)
    if validate:
        _validate_kv_edge_index(edge_index, member_index)
    return edge_index, cache_hit


def _ensure_kv_edge_index_fields(edge_index: dict):
    if "edge_flat" not in edge_index:
        edge_index["edge_flat"] = (
            edge_index["edge_q"].to(torch.int32) * int(edge_index["n_nb"]) + edge_index["edge_m"].to(torch.int32)
        ).contiguous()
    if "kv_order" not in edge_index:
        row_ptr = edge_index["row_ptr"].to(torch.int32)
        row_len = row_ptr[:, 1:] - row_ptr[:, :-1]
        edge_index["kv_order"] = torch.argsort(row_len, dim=1, descending=True).to(torch.int32).contiguous()
    return edge_index


def _build_kv_edge_index(member_index: torch.Tensor):
    # Build per-batch inverted mapping: kv -> sorted list of (q, m) edges.
    # Shapes:
    #   row_ptr: [B, N+1]
    #   edge_q, edge_m, kv_sorted: [B, N*M]
    bsz, n_ctx, n_nb = member_index.shape
    device = member_index.device
    e = n_ctx * n_nb
    q_template = torch.arange(n_ctx, device=device, dtype=torch.int32).repeat_interleave(n_nb)
    m_template = torch.arange(n_nb, device=device, dtype=torch.int32).repeat(n_ctx)
    row_ptr = torch.zeros((bsz, n_ctx + 1), device=device, dtype=torch.int32)
    edge_q = torch.empty((bsz, e), device=device, dtype=torch.int32)
    edge_m = torch.empty((bsz, e), device=device, dtype=torch.int32)
    edge_flat = torch.empty((bsz, e), device=device, dtype=torch.int32)
    kv_sorted = torch.empty((bsz, e), device=device, dtype=torch.int32)
    kv_order = torch.empty((bsz, n_ctx), device=device, dtype=torch.int32)
    flat_template = (q_template * n_nb + m_template).to(torch.int32)
    for b in range(bsz):
        kv_flat = member_index[b].reshape(-1).to(torch.int64)
        kv_s, order = torch.sort(kv_flat, stable=True)
        counts = torch.bincount(kv_s, minlength=n_ctx)
        row_ptr[b, 1:] = torch.cumsum(counts.to(torch.int32), dim=0)
        edge_q[b] = q_template.index_select(0, order.to(torch.int64))
        edge_m[b] = m_template.index_select(0, order.to(torch.int64))
        edge_flat[b] = flat_template.index_select(0, order.to(torch.int64))
        kv_sorted[b] = kv_s.to(torch.int32)
        kv_order[b] = torch.argsort(counts.to(torch.int32), descending=True).to(torch.int32)
    return {
        "row_ptr": row_ptr.contiguous(),
        "edge_q": edge_q.contiguous(),
        "edge_m": edge_m.contiguous(),
        "edge_flat": edge_flat.contiguous(),
        "kv_sorted": kv_sorted.contiguous(),
        "kv_order": kv_order.contiguous(),
        "n_ctx": int(n_ctx),
        "n_nb": int(n_nb),
    }


def _validate_kv_edge_index(edge_index: dict, member_index: torch.Tensor):
    bsz, n_ctx, n_nb = member_index.shape
    e = n_ctx * n_nb
    if edge_index["n_ctx"] != n_ctx or edge_index["n_nb"] != n_nb:
        raise ValueError("kv_edge_index shape metadata mismatch.")
    if edge_index["row_ptr"].shape != (bsz, n_ctx + 1):
        raise ValueError("kv_edge_index row_ptr shape mismatch.")
    if edge_index["edge_q"].shape != (bsz, e) or edge_index["edge_m"].shape != (bsz, e) or edge_index["kv_sorted"].shape != (bsz, e):
        raise ValueError("kv_edge_index edge tensor shape mismatch.")
    if edge_index["edge_flat"].shape != (bsz, e):
        raise ValueError("kv_edge_index edge_flat shape mismatch.")
    if edge_index["kv_order"].shape != (bsz, n_ctx):
        raise ValueError("kv_edge_index kv_order shape mismatch.")


def _kv_owner_reduce_from_edges_python(
    q_: torch.Tensor,
    do_: torch.Tensor,
    ds_flat: torch.Tensor,
    p_flat: torch.Tensor,
    edge_index: dict,
    sm_scale: float,
    bsz: int,
    h: int,
):
    # KV-owner style reduction at tensor level:
    # accumulate dK/dV by grouping edges by destination kv index.
    bh, n_ctx, head_dim = q_.shape
    n_nb = edge_index["n_nb"]
    edge_q = edge_index["edge_q"]
    edge_m = edge_index["edge_m"]
    kv_sorted = edge_index["kv_sorted"]
    dk_ = torch.zeros((bh, n_ctx, head_dim), device=q_.device, dtype=torch.float32)
    dv_ = torch.zeros((bh, n_ctx, head_dim), device=q_.device, dtype=torch.float32)
    for bh_idx in range(bh):
        b = bh_idx // h
        q_idx = edge_q[b].to(torch.int64)
        m_idx = edge_m[b].to(torch.int64)
        kv_idx = kv_sorted[b].to(torch.int64)
        flat_idx = q_idx * n_nb + m_idx
        ds_edge = ds_flat[bh_idx].reshape(-1).index_select(0, flat_idx).to(torch.float32)
        p_edge = p_flat[bh_idx].reshape(-1).index_select(0, flat_idx).to(torch.float32)
        q_edge = q_[bh_idx].index_select(0, q_idx).to(torch.float32)
        do_edge = do_[bh_idx].index_select(0, q_idx).to(torch.float32)
        dk_[bh_idx].index_add_(0, kv_idx, ds_edge[:, None] * q_edge * float(sm_scale))
        dv_[bh_idx].index_add_(0, kv_idx, p_edge[:, None] * do_edge)
    return dk_, dv_


def _kv_owner_reduce_from_edges_triton(
    q_: torch.Tensor,
    do_: torch.Tensor,
    ds_flat: torch.Tensor,
    p_flat: torch.Tensor,
    edge_index: dict,
    sm_scale: float,
    h: int,
):
    bh, n_ctx, head_dim = q_.shape
    dk_ = torch.empty((bh, n_ctx, head_dim), device=q_.device, dtype=torch.float32)
    dv_ = torch.empty((bh, n_ctx, head_dim), device=q_.device, dtype=torch.float32)
    row_ptr = edge_index["row_ptr"]
    edge_flat = edge_index["edge_flat"]
    kv_order = edge_index["kv_order"]
    grid = lambda meta: (
        triton.cdiv(n_ctx, meta["BLOCK_KV"]),
        bh,
        triton.cdiv(head_dim, meta["BLOCK_D"]),
    )
    _nbhood_attn_kv_owner_reduce_kernel[grid](
        q_,
        do_,
        ds_flat,
        p_flat,
        row_ptr,
        edge_flat,
        kv_order,
        dk_,
        dv_,
        n_ctx,
        edge_index["n_nb"],
        head_dim,
        h,
        sm_scale,
        q_.stride(0), q_.stride(1), q_.stride(2),
        do_.stride(0), do_.stride(1), do_.stride(2),
        ds_flat.stride(0), ds_flat.stride(1),
        p_flat.stride(0), p_flat.stride(1),
        row_ptr.stride(0), row_ptr.stride(1),
        edge_flat.stride(0), edge_flat.stride(1),
        kv_order.stride(0), kv_order.stride(1),
        dk_.stride(0), dk_.stride(1), dk_.stride(2),
        dv_.stride(0), dv_.stride(1), dv_.stride(2),
    )
    return dk_, dv_


def launch_flash_nbhd_attn_fwd(
    q,
    k,
    v,
    member_index,
    sm_scale,
    bias=None,
    mask=None,
    blank_k=None,
    blank_v=None,
    kv_edge_cache: KVEdgeIndexCache = None,
):
    """Launch the fused forward kernel.

    Args:
        q: [B, H, N, D] queries, unscaled.
        k: [B, H, N, D] keys.
        v: [B, H, N, D] values.
        member_index: [B, N, M] cluster members per token.
        sm_scale: softmax temperature.
        bias: [B, H, N, M] additive positional bias, or None.
        mask: bool mask over members, or None.
        blank_k: [H, D] learned blank key, or None.
        blank_v: [H, D] learned blank value, or None.
        kv_edge_cache: cache to pre-warm for the backward, or None to skip.
    Returns:
        ``(out, lse)``: [B, H, N, D] output and [B, H, N] log-sum-exp.
    """
    assert q.shape == k.shape == v.shape, "q, k, v must have the same shape"

    bsz, h, n, d = q.shape
    d_orig = d
    assert d % 8 == 0, "head_dim must be a multiple of 8"
    m = member_index.shape[-1]
    q_ = q.contiguous().view(-1, n, d)
    k_ = k.contiguous().view(-1, n, d)
    v_ = v.contiguous().view(-1, n, d)
    mi = member_index.to(torch.int32).contiguous()
    has_bias = bias is not None and bias.numel() > 0
    has_mask = mask is not None and mask.numel() > 0
    has_blank = blank_k is not None and blank_v is not None and blank_k.numel() > 0 and blank_v.numel() > 0
    bias_ = bias.contiguous().view(-1, n, m) if has_bias else torch.tensor([], device=q.device, dtype=q.dtype)
    if has_mask:
        if mask.shape[1] == 1:
            mask = mask.expand(bsz, h, n, m)
        mask_ = (mask > 0).to(torch.int32).contiguous().view(-1, n, m)
    else:
        mask_ = torch.tensor([], device=q.device, dtype=torch.int32)
    if has_blank:
        bk = blank_k.contiguous().view(h, d)
        bv = blank_v.contiguous().view(h, d)
    else:
        bk = torch.tensor([], device=q.device, dtype=q.dtype)
        bv = torch.tensor([], device=q.device, dtype=q.dtype)

    # Pad to a multiple of 8 for head_dim if necessary (required for triton kernel)
    # NOTE: this currently does not do anything due to the current assert. @TODO: test this
    padded = False
    if (d % 8) != 0 or (d & (d - 1)) != 0:
        d_pad = max(8, _next_pow2(d))
        pad = (0, d_pad - d)
        q_ = torch.nn.functional.pad(q_, pad).contiguous()
        k_ = torch.nn.functional.pad(k_, pad).contiguous()
        v_ = torch.nn.functional.pad(v_, pad).contiguous()
        if has_blank:
            bk = torch.nn.functional.pad(bk, pad).contiguous()
            bv = torch.nn.functional.pad(bv, pad).contiguous()
        d = d_pad
        padded = True

    # allocate output tensor
    out = torch.empty_like(q_, dtype=torch.float16)

    # allocate LSE tensor for backward pass
    lse = torch.empty((bsz * h, n), device=q.device, dtype=torch.float32)

    # launch forward kernel
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_Q"]), bsz * h)
    _nbhood_attn_fwd_kernel[grid](
        q_, k_, v_, mi, bias_, mask_, bk, bv, out, lse,
        n, m, d, h, sm_scale,
        q_.stride(0), q_.stride(1), q_.stride(2),
        k_.stride(0), k_.stride(1), k_.stride(2),
        v_.stride(0), v_.stride(1), v_.stride(2),
        mi.stride(0), mi.stride(1), mi.stride(2),
        bias_.stride(0) if has_bias else 0, bias_.stride(1) if has_bias else 0, bias_.stride(2) if has_bias else 0,
        mask_.stride(0) if has_mask else 0, mask_.stride(1) if has_mask else 0, mask_.stride(2) if has_mask else 0,
        bk.stride(0) if has_blank else 0, bk.stride(1) if has_blank else 0,
        bv.stride(0) if has_blank else 0, bv.stride(1) if has_blank else 0,
        out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1),
        HAS_BIAS=has_bias, HAS_MASK=has_mask, HAS_BLANK=has_blank,
    )
    out_ret = out.view(bsz, h, n, d).to(q.dtype)

    # unpad output dimension if it was padded
    if padded:
        out_ret = out_ret[..., :d_orig].contiguous()

    # Cache the edge index if it is not None and the cache is enabled
    if kv_edge_cache is not None and _env_flag("CACHE_KV_EDGE_INDEX", "1"):
        kv_edge_cache.get_or_build(mi.view(bsz, n, m))

    return out_ret, lse.view(bsz, h, n)


def launch_flash_nbhd_attn_bwd(
    q,
    k,
    v,
    member_index,
    sm_scale,
    o,
    lse,
    do,
    bias=None,
    mask=None,
    blank_k=None,
    blank_v=None,
    backend="atomic_split",
    split_intermed_fp16=True,
    auto_dup_threshold=3.0,
    kv_edge_index=None,
    kv_edge_cache: KVEdgeIndexCache = None,
    profile: bool = False,
):
    """Launch the fused backward kernels.

    Args:
        q, k, v: [B, H, N, D] forward inputs.
        member_index: [B, N, M] cluster members per token.
        sm_scale: softmax temperature.
        o: [B, H, N, D] forward output.
        lse: [B, H, N] log-sum-exp from the forward.
        do: [B, H, N, D] gradient of the output.
        bias: [B, H, N, M] additive bias, or None.
        mask: bool mask over members, or None.
        blank_k: [H, D] learned blank key, or None.
        blank_v: [H, D] learned blank value, or None.
        backend: dV strategy; ``"kv_owner"`` is the default.
        split_intermed_fp16: keep intermediates in fp16 to halve traffic.
        auto_dup_threshold: switch to the duplicating strategy above this
            average KV fan-in.
        kv_edge_index: prebuilt edge index, or None to resolve from the cache.
        kv_edge_cache: cache to resolve or build the edge index from.
        profile: emit per-kernel timings.
    Returns:
        ``(dq, dk, dv, dbias, dblank_k, dblank_v)``.
    """
    assert q.shape == k.shape == v.shape == o.shape == do.shape, "q, k, v, o, do must have the same shape"

    profile = bool(profile or _env_flag("ATTN_PROFILE_BWD", "0"))
    manual_fast = _env_flag("KV_OWNER_FAST_MANUAL", "1")
    need_dup_ratio = (backend == "auto") or profile or (not manual_fast)
    dup_ratio = _estimate_dup_ratio(member_index) if need_dup_ratio else 0.0
    collision = _compute_collision_stats(member_index) if profile else {
        "dup_ratio": dup_ratio if need_dup_ratio else 0.0,
        "row_unique_mean": 0.0,
        "row_unique_p90": 0.0,
        "block_unique_mean": 0.0,
        "block_unique_p90": 0.0,
    }
    if backend == "auto":
        # Keep auto conservative: kv_owner currently helps only in narrow, high-collision regimes.
        n_ctx = int(member_index.shape[1])
        n_nb = int(member_index.shape[2])
        use_kv_owner = (dup_ratio >= auto_dup_threshold) and (n_ctx <= 256) and (n_nb <= 64)
        backend = "kv_owner" if use_kv_owner else "atomic_split"
    if backend == "atomic":
        backend = "atomic_full"
    if backend == "noatom":
        backend = "atomic_full"
        enable_atomics = False
    else:
        enable_atomics = True
    if backend not in {"atomic_full", "atomic_split", "kv_owner"}:
        raise ValueError(f"Unknown backward backend: {backend}")
    bsz, h, n, d = q.shape
    assert d % 8 == 0, "head_dim must be a multiple of 8"
    m = member_index.shape[-1]
    device, dtype = q.device, q.dtype
    has_bias = bias is not None and bias.numel() > 0
    has_mask = mask is not None and mask.numel() > 0
    has_blank = blank_k is not None and blank_v is not None and blank_k.numel() > 0 and blank_v.numel() > 0
    q_ = q.contiguous().view(-1, n, d)
    k_ = k.contiguous().view(-1, n, d)
    v_ = v.contiguous().view(-1, n, d)
    o_ = o.contiguous().view(-1, n, d)
    do_ = do.contiguous().view(-1, n, d)
    lse_ = lse.contiguous().view(-1, n)
    mi = member_index.to(torch.int32).contiguous()
    bias_ = bias.contiguous().view(-1, n, m) if has_bias else torch.tensor([], device=device, dtype=dtype)
    if has_mask:
        if mask.shape[1] == 1:
            mask = mask.expand(bsz, h, n, m)
        mask_ = (mask > 0).to(torch.int32).contiguous().view(-1, n, m)
    else:
        mask_ = torch.tensor([], device=device, dtype=torch.int32)
    if has_blank:
        bk = blank_k.contiguous().view(h, d)
        bv = blank_v.contiguous().view(h, d)
    else:
        bk = torch.tensor([], device=device, dtype=dtype)
        bv = torch.tensor([], device=device, dtype=dtype)

    # Pad to a multiple of 8 for head_dim if necessary (required for triton kernel)
    # NOTE: this currently does not do anything due to the current assert. @TODO: test this
    padded = False
    d_orig = d
    if (d % 8) != 0 or (d & (d - 1)) != 0:
        d_pad = max(8, _next_pow2(d))
        pad = (0, d_pad - d)
        q_ = torch.nn.functional.pad(q_, pad).contiguous()
        k_ = torch.nn.functional.pad(k_, pad).contiguous()
        v_ = torch.nn.functional.pad(v_, pad).contiguous()
        o_ = torch.nn.functional.pad(o_, pad).contiguous()
        do_ = torch.nn.functional.pad(do_, pad).contiguous()
        if has_blank:
            bk = torch.nn.functional.pad(bk, pad).contiguous()
            bv = torch.nn.functional.pad(bv, pad).contiguous()
        d = d_pad
        padded = True

    dq_ = torch.zeros_like(q_, dtype=torch.float32)
    dk_ = torch.zeros_like(k_, dtype=torch.float32)
    dv_ = torch.zeros_like(v_, dtype=torch.float32)
    dbias_ = torch.empty((bsz * h, n, m), device=device, dtype=torch.float32) if has_bias else torch.tensor([], device=device, dtype=torch.float32)
    dbk = torch.zeros((h, d), device=device, dtype=torch.float32) if has_blank else torch.tensor([], device=device, dtype=torch.float32)
    dbv = torch.zeros((h, d), device=device, dtype=torch.float32) if has_blank else torch.tensor([], device=device, dtype=torch.float32)
    delta = torch.zeros((bsz * h, n), device=device, dtype=torch.float32)
    timing = {"preprocess_ms": 0.0, "core_ms": 0.0, "reduce_ms": 0.0, "total_ms": 0.0}
    reduce_impl = "none"
    cache_hit = False

    def _timed_call(key: str, fn):
        if not profile:
            fn()
            return
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        timing[key] += float(e0.elapsed_time(e1))

    grid_pre = lambda meta: (triton.cdiv(n, meta["BLOCK_M"]), bsz * h)
    _timed_call("preprocess_ms", lambda: _nbhood_attn_bwd_preprocess[grid_pre](o_, do_, delta, n, HEAD_DIM=d))
    grid_q = lambda meta: (triton.cdiv(n, meta["BLOCK_Q"]), bsz * h)
    if backend in {"atomic_split", "kv_owner"}:
        intermed_dtype = torch.float16 if split_intermed_fp16 else torch.float32
        ds_flat = torch.empty((bsz * h, n * m), device=device, dtype=intermed_dtype)
        p_flat = torch.empty((bsz * h, n * m), device=device, dtype=intermed_dtype)
        _timed_call(
            "core_ms",
            lambda: _nbhood_attn_bwd_core_kernel[grid_q](
                q_, k_, v_, do_, lse_, delta, mi, bias_, mask_, bk, bv,
                dq_, dbias_, dbk, dbv, ds_flat, p_flat,
                n, m, d, h, sm_scale,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                do_.stride(0), do_.stride(1), do_.stride(2),
                lse_.stride(0), lse_.stride(1),
                delta.stride(0), delta.stride(1),
                mi.stride(0), mi.stride(1), mi.stride(2),
                bias_.stride(0) if has_bias else 0, bias_.stride(1) if has_bias else 0, bias_.stride(2) if has_bias else 0,
                mask_.stride(0) if has_mask else 0, mask_.stride(1) if has_mask else 0, mask_.stride(2) if has_mask else 0,
                bk.stride(0) if has_blank else 0, bk.stride(1) if has_blank else 0,
                bv.stride(0) if has_blank else 0, bv.stride(1) if has_blank else 0,
                dq_.stride(0), dq_.stride(1), dq_.stride(2),
                dbias_.stride(0) if has_bias else 0, dbias_.stride(1) if has_bias else 0, dbias_.stride(2) if has_bias else 0,
                ds_flat.stride(0), ds_flat.stride(1),
                p_flat.stride(0), p_flat.stride(1),
                HAS_BIAS=has_bias, HAS_MASK=has_mask, HAS_BLANK=has_blank, STORE_INTERMED_FP16=split_intermed_fp16,
            ),
        )
        if backend == "atomic_split":
            reduce_impl = "atomic_split_triton"
            _timed_call(
                "reduce_ms",
                lambda: _nbhood_attn_bwd_reduce_kernel[grid_q](
                    q_, do_, mi, ds_flat, p_flat, dk_, dv_,
                    n, m, d, h, sm_scale,
                    q_.stride(0), q_.stride(1), q_.stride(2),
                    do_.stride(0), do_.stride(1), do_.stride(2),
                    mi.stride(0), mi.stride(1), mi.stride(2),
                    ds_flat.stride(0), ds_flat.stride(1),
                    p_flat.stride(0), p_flat.stride(1),
                    dk_.stride(0), dk_.stride(1), dk_.stride(2),
                    dv_.stride(0), dv_.stride(1), dv_.stride(2),
                ),
            )
        else:
            reduce_impl = "kv_owner_triton"
            validate_cache_hit = _env_flag("KV_OWNER_VALIDATE_CACHE_HIT", "0")
            validate_provided = _env_flag("KV_OWNER_VALIDATE_PROVIDED", "1")
            should_validate = validate_provided if kv_edge_index is not None else validate_cache_hit
            kv_edge_index, cache_hit_resolved = _resolve_kv_edge_index(
                mi.view(bsz, n, m),
                kv_edge_index=kv_edge_index,
                kv_edge_cache=kv_edge_cache,
                validate=should_validate,
            )
            cache_hit = bool(cache_hit_resolved)
            use_python_reduce = _env_flag("KV_OWNER_PYTHON_REDUCE", "0")
            if use_python_reduce:
                reduce_impl = "kv_owner_python"
            reduce_fn = _kv_owner_reduce_from_edges_python if use_python_reduce else _kv_owner_reduce_from_edges_triton
            def _run_kv_reduce():
                if use_python_reduce:
                    dk_red, dv_red = reduce_fn(
                        q_, do_, ds_flat, p_flat, kv_edge_index, sm_scale, bsz=bsz, h=h
                    )
                else:
                    dk_red, dv_red = reduce_fn(
                        q_, do_, ds_flat, p_flat, kv_edge_index, sm_scale, h=h
                    )
                dk_.copy_(dk_red)
                dv_.copy_(dv_red)
            _timed_call(
                "reduce_ms",
                _run_kv_reduce,
            )
    else:
        _timed_call(
            "core_ms",
            lambda: _nbhood_attn_bwd_kernel[grid_q](
                q_, k_, v_, do_, lse_, delta, mi, bias_, mask_, bk, bv,
                dq_, dk_, dv_, dbias_, dbk, dbv,
                n, m, d, h, sm_scale,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                do_.stride(0), do_.stride(1), do_.stride(2),
                lse_.stride(0), lse_.stride(1),
                delta.stride(0), delta.stride(1),
                mi.stride(0), mi.stride(1), mi.stride(2),
                bias_.stride(0) if has_bias else 0, bias_.stride(1) if has_bias else 0, bias_.stride(2) if has_bias else 0,
                mask_.stride(0) if has_mask else 0, mask_.stride(1) if has_mask else 0, mask_.stride(2) if has_mask else 0,
                bk.stride(0) if has_blank else 0, bk.stride(1) if has_blank else 0,
                bv.stride(0) if has_blank else 0, bv.stride(1) if has_blank else 0,
                dq_.stride(0), dq_.stride(1), dq_.stride(2),
                dk_.stride(0), dk_.stride(1), dk_.stride(2),
                dv_.stride(0), dv_.stride(1), dv_.stride(2),
                dbias_.stride(0) if has_bias else 0, dbias_.stride(1) if has_bias else 0, dbias_.stride(2) if has_bias else 0,
                HAS_BIAS=has_bias, HAS_MASK=has_mask, HAS_BLANK=has_blank, ENABLE_ATOMICS=enable_atomics,
            ),
        )

    dq = dq_.view(bsz, h, n, d).to(dtype).contiguous()
    dk = dk_.view(bsz, h, n, d).to(dtype).contiguous()
    dv = dv_.view(bsz, h, n, d).to(dtype).contiguous()
    dbias = dbias_.view(bsz, h, n, m).to(dtype).contiguous() if has_bias else None
    dblank_k = dbk.to(dtype).contiguous() if has_blank else None
    dblank_v = dbv.to(dtype).contiguous() if has_blank else None
    if padded:
        dq = dq[..., :d_orig].contiguous()
        dk = dk[..., :d_orig].contiguous()
        dv = dv[..., :d_orig].contiguous()
        if has_blank:
            dblank_k = dblank_k[..., :d_orig].contiguous()
            dblank_v = dblank_v[..., :d_orig].contiguous()
    if profile:
        timing["total_ms"] = timing["preprocess_ms"] + timing["core_ms"] + timing["reduce_ms"]
    launch_flash_nbhd_attn_bwd.last_profile = {
        **timing,
        "backend": backend,
        "reduce_impl": reduce_impl,
        "kv_cache_hit": cache_hit,
        "dup_ratio": collision["dup_ratio"],
        "row_unique_mean": collision["row_unique_mean"],
        "row_unique_p90": collision["row_unique_p90"],
        "block_unique_mean": collision["block_unique_mean"],
        "block_unique_p90": collision["block_unique_p90"],
    }
    return dq, dk, dv, dbias, dblank_k, dblank_v


class FlashNeighborhoodAttentionFunction(Function):
    """Fused neighbourhood attention (v2), with a KV-owner backward.

    Prefer :func:`affmae.ops.nbhd_attn.nbhd_attn`, which picks between this, the v1
    kernel, the torch fallback and the CUDA extension.

    The backward accumulates into KV tokens rather than queries, which needs the
    edge index in :class:`KVEdgeIndexCache`. ``build_backward_state`` says whether
    to build it: the caller must decide, because ``torch.is_grad_enabled()`` reads
    False inside ``Function.forward`` and ``ctx.needs_input_grad`` is True even
    under ``no_grad`` when ``blank_k``/``blank_v`` arrive as ``nn.Parameter``.
    """
    kv_edge_cache = None

    @staticmethod
    def set_kv_edge_cache(cache):
        """Install the cache the next forward will use.

        Args:
            cache: a KVEdgeIndexCache, or None to disable caching.
        """
        FlashNeighborhoodAttentionFunction.kv_edge_cache = cache

    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.float16)
    def forward(ctx, q, k, v, member_idx, bias, mask, blank_k, blank_v,
                softmax_scale, build_backward_state=True):
        has_bias = bias is not None and bias.numel() > 0
        has_mask = mask is not None and mask.numel() > 0
        has_blank = blank_k is not None and blank_v is not None and blank_k.numel() > 0 and blank_v.numel() > 0
        out, lse = launch_flash_nbhd_attn_fwd(
            q,
            k,
            v,
            member_idx,
            softmax_scale,
            bias=bias if has_bias else None,
            mask=mask if has_mask else None,
            blank_k=blank_k if has_blank else None,
            blank_v=blank_v if has_blank else None,
        )
        # The KV-owner edge index is consumed only by the backward pass, and
        # aff.py installs a cache unconditionally, so inference used to pay
        # ~0.7 ms per block to build state nothing would read.
        # Decided by the caller, not detected here. Inside a Function.forward
        # torch.is_grad_enabled() is always False, and ctx.needs_input_grad is
        # True even under no_grad because blank_k/blank_v arrive as nn.Parameters
        # whose requires_grad flag does not depend on grad mode.
        if not (build_backward_state and any(ctx.needs_input_grad)):
            ctx.kv_edge_index = None
        else:
            mi32 = member_idx.to(torch.int32).contiguous()
            cache_obj = FlashNeighborhoodAttentionFunction.kv_edge_cache
            if cache_obj is not None:
                ctx.kv_edge_index = cache_obj.get_or_build(mi32)
            elif _env_flag("CACHE_KV_EDGE_INDEX", "0"):
                ctx.kv_edge_index = _build_kv_edge_index(mi32)
            else:
                ctx.kv_edge_index = None
        ctx.save_for_backward(
            q,
            k,
            v,
            member_idx,
            bias if has_bias else torch.tensor([], device=q.device, dtype=q.dtype),
            mask if has_mask else torch.tensor([], device=q.device, dtype=torch.int32),
            blank_k if has_blank else torch.tensor([], device=q.device, dtype=q.dtype),
            blank_v if has_blank else torch.tensor([], device=q.device, dtype=q.dtype),
            out,
            lse,
        )
        ctx.scale = softmax_scale
        ctx.flags = (has_bias, has_mask, has_blank)
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        q, k, v, member_idx, bias, mask, blank_k, blank_v, out, lse = ctx.saved_tensors
        has_bias, has_mask, has_blank = ctx.flags
        dq, dk, dv, dbias, dblank_k, dblank_v = launch_flash_nbhd_attn_bwd(
            q,
            k,
            v,
            member_idx,
            ctx.scale,
            out,
            lse,
            grad_out.contiguous(),
            bias=bias if has_bias else None,
            mask=mask if has_mask else None,
            blank_k=blank_k if has_blank else None,
            blank_v=blank_v if has_blank else None,
            backend="kv_owner",
            split_intermed_fp16=True,
            kv_edge_index=ctx.kv_edge_index,
            kv_edge_cache=FlashNeighborhoodAttentionFunction.kv_edge_cache,
        )
        return (dq, dk, dv, None, dbias if has_bias else None, None,
                dblank_k if has_blank else None,
                dblank_v if has_blank else None, None, None)


# pytorch reference implementation of neighborhood attention forward pass (for testing)
def flash_nbhd_attn_reference_forward(
    q,
    k,
    v,
    member_index,
    sm_scale,
    bias=None,
    mask=None,
    blank_k=None,
    blank_v=None,
):
    """Float64 reference forward, for differential-testing the kernel.

    Args:
        q: [B, H, N, D] queries, unscaled.
        k: [B, H, N, D] keys.
        v: [B, H, N, D] values.
        member_index: [B, N, M] cluster members per token.
        sm_scale: softmax temperature.
        bias: [B, H, N, M] additive positional bias, or None.
        mask: [B, 1, N, M] or [B, H, N, M] bool, True where a member is real.
        blank_k: [H, D] learned blank key, or None.
        blank_v: [H, D] learned blank value, or None.
    Returns:
        ``(out, lse)``: [B, H, N, D] output and [B, H, N] log-sum-exp.
    """
    qd = q.to(torch.float64)
    kd = k.to(torch.float64)
    vd = v.to(torch.float64)
    bsz, h, n, d = qd.shape
    m = member_index.shape[-1]
    idx = member_index.long().unsqueeze(1).expand(bsz, h, n, m)
    idx_d = idx.unsqueeze(-1).expand(bsz, h, n, m, d)
    k_nb = torch.gather(kd.unsqueeze(3).expand(bsz, h, n, m, d), 2, idx_d)
    v_nb = torch.gather(vd.unsqueeze(3).expand(bsz, h, n, m, d), 2, idx_d)
    logits = (qd.unsqueeze(3) * k_nb).sum(-1) * float(sm_scale)
    if bias is not None and bias.numel() > 0:
        logits = logits + bias.to(torch.float64)

    mask_h = None
    if mask is not None and mask.numel() > 0:
        if mask.shape[1] == 1:
            mask_h = mask.expand(bsz, h, n, m)
        else:
            mask_h = mask
        valid = mask_h.to(torch.bool)
        logits = torch.where(valid, logits, torch.full_like(logits, -1.0e20))
    else:
        valid = None

    if blank_k is not None and blank_v is not None and blank_k.numel() > 0 and blank_v.numel() > 0:
        bk = blank_k.to(torch.float64).view(1, h, 1, d)
        bv = blank_v.to(torch.float64).view(1, h, 1, d)
        blank_logits = (qd * bk).sum(-1, keepdim=True) * float(sm_scale)
        logits_full = torch.cat([logits, blank_logits], dim=-1)
        p_full = torch.softmax(logits_full, dim=-1)
        p = p_full[..., :m]
        out = (p.unsqueeze(-1) * v_nb).sum(dim=3) + p_full[..., -1:] * bv
        lse = torch.logsumexp(logits_full, dim=-1)
    else:
        p = torch.softmax(logits, dim=-1)
        if valid is not None:
            p = torch.where(valid, p, torch.zeros_like(p))
            denom = p.sum(-1, keepdim=True)
            p = torch.where(denom > 0, p / torch.clamp(denom, min=1e-20), torch.zeros_like(p))
        out = (p.unsqueeze(-1) * v_nb).sum(dim=3)
        # Keep lse finite even when all entries are masked.
        lse = torch.where(
            (p.sum(-1) > 0),
            torch.logsumexp(logits, dim=-1),
            torch.full((bsz, h, n), -1.0e20, device=qd.device, dtype=torch.float64),
        )
    return out, lse


# pytorch reference implementation of neighborhood attention backward pass/autograd (for testing)
def flash_nbhd_attn_reference_backward(
    q,
    k,
    v,
    member_index,
    sm_scale,
    do,
    bias=None,
    mask=None,
    blank_k=None,
    blank_v=None,
):
    """Float64 reference backward, for differential-testing the kernel.

    Args:
        q: [B, H, N, D] queries.
        k: [B, H, N, D] keys.
        v: [B, H, N, D] values.
        member_index: [B, N, M] cluster members per token.
        sm_scale: softmax temperature.
        do: [B, H, N, D] gradient of the output.
        bias: [B, H, N, M] additive bias, or None.
        mask: bool mask over members, or None.
        blank_k: [H, D] learned blank key, or None.
        blank_v: [H, D] learned blank value, or None.
    Returns:
        Gradients matching the forward's differentiable inputs.
    """
    qd = q.to(torch.float64).detach().requires_grad_(True)
    kd = k.to(torch.float64).detach().requires_grad_(True)
    vd = v.to(torch.float64).detach().requires_grad_(True)
    dod = do.to(torch.float64).detach()
    tensors = [qd, kd, vd]
    b64 = None
    bk64 = None
    bv64 = None
    if bias is not None and bias.numel() > 0:
        b64 = bias.to(torch.float64).detach().requires_grad_(True)
        tensors.append(b64)
    if blank_k is not None and blank_v is not None and blank_k.numel() > 0 and blank_v.numel() > 0:
        bk64 = blank_k.to(torch.float64).detach().requires_grad_(True)
        bv64 = blank_v.to(torch.float64).detach().requires_grad_(True)
        tensors.extend([bk64, bv64])

    out, _ = flash_nbhd_attn_reference_forward(
        qd,
        kd,
        vd,
        member_index,
        sm_scale,
        bias=b64,
        mask=mask,
        blank_k=bk64,
        blank_v=bv64,
    )
    loss = (out * dod).sum()
    grads = torch.autograd.grad(loss, tensors, allow_unused=True)

    it = iter(grads)
    dq = next(it)
    dk = next(it)
    dv = next(it)
    dbias = next(it) if b64 is not None else None
    dblank_k = next(it) if bk64 is not None else None
    dblank_v = next(it) if bv64 is not None else None
    return dq, dk, dv, dbias, dblank_k, dblank_v
