''' Optimized Triton kernel for Cluster Attention

Based loosely on FlashAttention implementation, fused neighborhood-aware QK -> softmax -> AV: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py
heavily based on triton implementation: https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py#L164

Please for those in the future be weary of messing with any .contiguous() or .reshape() calls. If you do, verify gradient/backward
calls with previous CUDA implementation. Triton implementation is extremely sensitive to this and you can end up breaking
strides and call tensor indices out of indented order!

I spent forever debugging this and you will too.
'''
import os
import math
import pytest
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.amp import custom_fwd, custom_bwd
from torch.testing import assert_close

import triton
import triton.language as tl

try:
    from .util import NUM_STAGES_OPTIONS, is_hip, is_cuda, supports_host_descriptor
except ImportError:
    from util import NUM_STAGES_OPTIONS, is_hip, is_cuda, supports_host_descriptor


configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [16, 32]\
    for BN in [16, 32, 64]\
    for s in NUM_STAGES_OPTIONS \
    for w in [1, 2, 4, 8]\
]

configs_no_neighbor = [
    triton.Config({'BLOCK_M': BM}, num_stages=s, num_warps=w) \
    for BM in [16, 32]\
    for s in NUM_STAGES_OPTIONS \
    for w in [1, 2, 4, 8]\
]

def smallest_valid_power_2(x):
    """ Selected for block m """
    if x >= 32:
        return 32
    elif x >= 16:
        return 16
    elif x >= 8:
        return 8
    elif x >= 4:
        return 4
    else:
        return 1


def prune_invalid_configs(configs, named_args, **kwargs):
    """ Strip block sizes that are greater than the sequence length or neighborhood size. """
    N_CTX = kwargs["N_CTX"]
    NEIGHBOR_SIZE = kwargs["NEIGHBOR_SIZE"]
    confs = []
    for conf in configs:
        if conf.kwargs.get("BLOCK_M", 0) <= N_CTX and conf.kwargs.get("BLOCK_N", 0) <= NEIGHBOR_SIZE and conf.kwargs.get("warp_specialize", 0) == 0:
            confs.append(conf)
    return confs


def prune_invalid_configs_no_neighbor(configs, named_args, **kwargs):
    """ Strip block sizes that are greater than the sequence length """
    N_CTX = kwargs["N_CTX"]
    confs = []
    for conf in configs:
        if conf.kwargs.get("BLOCK_M", 0) <= N_CTX:
            confs.append(conf)
    return confs


@triton.jit
def _update_online_softmax_block(
    acc, l_i, m_i,
    q_tile,                 # [BM, D] fp16
    k_block, v_block,       # [BM, D] fp16
    bias_col,               # [BM]    fp16 or fp32
    valid_mask,             # [BM]    bool
    qk_scale: tl.constexpr,
    USE_EXP2: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    # logits in fp32
    qk = tl.sum(q_tile * k_block, axis=1) * qk_scale
    qk = tl.where(valid_mask, qk, -1e+11)  # mask by valid_mask (note: -inf causes nan in exp when using nbhood based masking)

    # bias masked by mask (same as neighbors)
    if HAS_BIAS:
        b = bias_col
        # mask by mask
        b = tl.where(valid_mask, b, 0.0)
        qk = qk + b

    # online softmax (fp32)
    m_ij = tl.maximum(m_i, qk)
    if USE_EXP2:
        p     = tl.math.exp2(qk - m_ij)
        alpha = tl.math.exp2(m_i - m_ij)
    else:
        p     = tl.math.exp(qk - m_ij)
        alpha = tl.math.exp(m_i - m_ij)

    l_i = l_i * alpha + p
    m_i = m_ij

    # acc in fp32
    acc = acc * alpha[:, None] + (p[:, None] * v_block)
    return acc, l_i, m_i


@triton.jit
def _nbhood_attn_fwd_inner(
    acc, l_i, m_i,
    q_tile,                         # fp16 tile
    member_index, desc_k, desc_v,
    bias,
    mask,
    offset_y_kv,
    offset_y_mi,
    start_m,
    qk_scale, N_CTX,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    NEIGHBOR_SIZE: tl.constexpr, USE_EXP2: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_MASK: tl.constexpr
):
    # Vectorization hints
    tl.static_assert(HEAD_DIM % 8 == 0)
    mb = tl.arange(0, BLOCK_M)   # [BM]
    mb = tl.multiple_of(mb, 8)

    # Hoisted locals
    rows_local = start_m * BLOCK_M + mb # [BM]
    row_mask   = rows_local < N_CTX

    # Base (no head dim on member_index)
    base_mi = member_index + offset_y_mi * NEIGHBOR_SIZE
    d = tl.arange(0, HEAD_DIM)
    d = tl.multiple_of(d, 8)

    for nb in tl.range(0, NEIGHBOR_SIZE, BLOCK_N):
        nb = tl.multiple_of(nb, BLOCK_N)
        for j in tl.range(0, BLOCK_N):
            # Vectorization hint for neighbor index
            # j = tl.multiple_of(j, 8)
            # neighborhood index masking
            nmask = (nb + j) < NEIGHBOR_SIZE
            masking = row_mask & nmask

            # member_index column
            mi_col = tl.make_block_ptr(
                base=base_mi + (nb + j),
                shape=(N_CTX,), strides=(NEIGHBOR_SIZE,),
                offsets=(start_m * BLOCK_M,),
                block_shape=(BLOCK_M,), order=(0,),
            )
            idx_col = tl.load(mi_col, boundary_check=(0,)).to(tl.int32)            # [BM], int32 storage

            # rows for k/v (with head base)
            rows = (offset_y_kv + idx_col).to(tl.int32)

            # per-neighborhood mask loading
            if HAS_MASK:
                base_mask = mask + offset_y_kv * NEIGHBOR_SIZE
                mask_col_bp = tl.make_block_ptr(
                    base        = base_mask + (nb + j),
                    shape       = (N_CTX,),
                    strides     = (NEIGHBOR_SIZE,),
                    offsets     = (start_m * BLOCK_M,),
                    block_shape = (BLOCK_M,),
                    order       = (0,),
                )
                mask_col = tl.load(mask_col_bp, boundary_check=(0,))
                masking = (mask_col != 0) & masking

            # Gather K/V as fp16 from the neighborhood indices
            # k_block = desc_k.gather(rows, y0)   # [BM, HD], fp16
            # v_block = desc_v.gather(rows, y0)   # [BM, HD], fp16
            k_ptrs = desc_k + rows[:, None] * HEAD_DIM + d[None, :]
            v_ptrs = desc_v + rows[:, None] * HEAD_DIM + d[None, :]
            k_block = tl.load(k_ptrs, mask=masking[:, None], other=0.0).to(tl.float32)
            v_block = tl.load(v_ptrs, mask=masking[:, None], other=0.0).to(tl.float32)

            if HAS_BIAS:
                # bias flattened as [Z*H*N_CTX, M]
                base_bias = bias + offset_y_kv * NEIGHBOR_SIZE
                bias_col_bp = tl.make_block_ptr(
                    base        = base_bias + (nb + j),
                    shape       = (N_CTX,),
                    strides     = (NEIGHBOR_SIZE,),
                    offsets     = (start_m * BLOCK_M,),
                    block_shape = (BLOCK_M,),
                    order       = (0,),
                )
                bcol = tl.load(bias_col_bp, boundary_check=(0,))       # fp16
                bcol = tl.where(masking, bcol, 0.0).to(tl.float32)

            acc, l_i, m_i = _update_online_softmax_block(
                acc, l_i, m_i,
                q_tile,
                k_block, v_block,
                (bcol if HAS_BIAS else bias),  # give block ptr if bias, otherwise give empty tensor ptr at bias
                masking,  # update mask for only valid neighborhood indices
                qk_scale,
                USE_EXP2=USE_EXP2,
                HAS_BIAS=HAS_BIAS
            )

    return acc, l_i, m_i


# @triton.autotune(
#     configs=configs,
#     key=["N_CTX", "HEAD_DIM", "warp_specialize", "USE_EXP2", "HAS_BIAS", "HAS_BLANK", "NEIGHBOR_SIZE"],
#     prune_configs_by={'early_config_prune': prune_invalid_configs},
#     cache_results=True
# )
@triton.heuristics(values={
    'BLOCK_M': lambda args: 16,
    'BLOCK_N': lambda args: smallest_valid_power_2(args['NEIGHBOR_SIZE']),
    'num_warps': lambda args: 1,
    'num_stages': lambda args: 2
})
@triton.jit
def _nbhood_attn_fwd(sm_scale, Z, H, q, k, v, member_index, bias, mask, blank_k, blank_v, o, lse, N_CTX, NEIGHBOR_SIZE,
              HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              warp_specialize: tl.constexpr, USE_EXP2: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_BLANK: tl.constexpr, HAS_MASK: tl.constexpr):

    LN2: tl.constexpr = 0.6931471805599453
    tl.static_assert(HEAD_DIM % 8 == 0)
    # tl.multiple_of(HEAD_DIM, 16)

    # desc_q = tl.make_tensor_descriptor(q, shape=[Z*H*N_CTX, HEAD_DIM],
    #                                    strides=[HEAD_DIM, 1],
    #                                    block_shape=[BLOCK_M, HEAD_DIM])
    desc_k = k # tl.make_tensor_descriptor(k, shape=[Z*H*N_CTX, HEAD_DIM],
    #                                   strides=[HEAD_DIM, 1],
    #                                   block_shape=[1, HEAD_DIM])
    desc_v = v # tl.make_tensor_descriptor(v, shape=[Z*H*N_CTX, HEAD_DIM],
    #                                   strides=[HEAD_DIM, 1],
    #                                   block_shape=[1, HEAD_DIM])
    # desc_o = tl.make_tensor_descriptor(o, shape=[Z*H*N_CTX, HEAD_DIM],
    #                                    strides=[HEAD_DIM, 1],
    #                                    block_shape=[BLOCK_M, HEAD_DIM])

    assert BLOCK_N <= NEIGHBOR_SIZE
    start_m = tl.program_id(0)
    off_hz  = tl.program_id(1)
    off_z   = off_hz // H
    off_h   = off_hz % H

    # Bases
    offset_y_kv = off_z * (N_CTX * H) + off_h * N_CTX
    offset_y_mi = off_z * N_CTX

    # Hoisted tails and indices
    rows_local = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask   = rows_local < N_CTX
    rows_qov   = offset_y_kv + rows_local
    # rows_qov = tl.where(row_mask, rows_qov, 0)

    # Row-vector gather for Q (fp16)
    # desc_q_row = tl.make_tensor_descriptor(
    #    q, shape=[Z*H*N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[1, HEAD_DIM]
    #)
    #q_tile = desc_q_row.gather(rows_qov, y0)     # [BM, HD], fp16
    # q_tile = tl.where(row_mask[:, None], q_tile, 0.0)
    d = tl.arange(0, HEAD_DIM)
    d = tl.multiple_of(d, 8)
    q_ptrs = q + rows_qov[:, None] * HEAD_DIM + d[None, :]
    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0).to(tl.float32)
    # q_tile = tl.where(row_mask[:, None], q_tile, 0.0)

    # Softmax state in fp32
    m_i = tl.full([BLOCK_M], -1e+11, tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    if USE_EXP2:
        qk_scale = sm_scale * 1.44269504
    else:
        qk_scale = sm_scale

    acc, l_i, m_i = _nbhood_attn_fwd_inner(
        acc, l_i, m_i,
        q_tile,
        member_index, desc_k, desc_v,
        bias,
        mask,
        offset_y_kv,
        offset_y_mi,
        start_m, qk_scale,
        N_CTX,
        BLOCK_M, HEAD_DIM, BLOCK_N, NEIGHBOR_SIZE, USE_EXP2, HAS_BIAS, HAS_MASK
    )

    if HAS_BLANK:
        bk = tl.load(blank_k + off_h * HEAD_DIM + d).to(tl.float32)    # [D]
        bv = tl.load(blank_v + off_h * HEAD_DIM + d).to(tl.float32)    # [D]

        # logits for blank in fp32, mask by row_mask ONLY
        qk_blank = tl.sum(q_tile * bk[None, :], axis=1) * qk_scale
        qk_blank = tl.where(row_mask, qk_blank, -1e+11)

        # online-softmax update (identical to neighbors)
        m_ij = tl.maximum(m_i, qk_blank)
        if USE_EXP2:
            p_b  = tl.math.exp2(qk_blank - m_ij)
            alpha = tl.math.exp2(m_i - m_ij)
        else:
            p_b  = tl.math.exp(qk_blank - m_ij)
            alpha = tl.math.exp(m_i - m_ij)

        l_i = l_i * alpha + p_b
        m_i = m_ij
        acc = acc * alpha[:, None] + p_b[:, None] * bv[None, :]

    # LSE kept in fp32
    if USE_EXP2:
        lse_row = (m_i + tl.math.log2(l_i + 1e-12) * LN2)
    else:
        lse_row = (m_i + tl.math.log(l_i + 1e-12))
    lse_ptrs = lse + rows_qov
    tl.store(lse_ptrs, lse_row, mask=row_mask)

    # Epilogue: store O in fp16
    inv_l = tl.math.exp(m_i - lse_row)
    o_tile_f32 = acc * inv_l[:, None]
    o_tile_f32 = tl.where(row_mask[:, None], o_tile_f32, 0.0)

    d = tl.arange(0, HEAD_DIM)
    d = tl.multiple_of(d, 8)
    o_ptrs = o + rows_qov[:, None] * HEAD_DIM + d[None, :]
    tl.store(o_ptrs, o_tile_f32.to(tl.float16), mask=row_mask[:, None])


# @triton.autotune(
#     configs=configs_no_neighbor,
#     key=["N_CTX", "HEAD_DIM"],
#     prune_configs_by={'early_config_prune': prune_invalid_configs_no_neighbor},
#     reset_to_zero=['Delta'],
#     cache_results=True
# )
@triton.heuristics(values={
    'BLOCK_M': lambda args: smallest_valid_power_2(args['N_CTX']),
    'num_warps': lambda args: 8,
    'num_stages': lambda args: 4
})
@triton.jit
def _nbhood_attn_bwd_preprocess(O, DO, Delta, Z, H, N_CTX,
                                BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
    start_m = tl.program_id(0) * BLOCK_M
    off_hz  = tl.program_id(1)

    rows_local = tl.arange(0, BLOCK_M) + start_m
    rows_y     = (off_hz * N_CTX + rows_local).to(tl.int32)  # tail-free addressing

    tl.static_assert(HEAD_DIM % 8 == 0)
    d = tl.arange(0, HEAD_DIM)
    d = tl.multiple_of(d, 8)

    o_ptrs  = O  + rows_y[:, None] * HEAD_DIM + d[None, :]
    do_ptrs = DO + rows_y[:, None] * HEAD_DIM + d[None, :]

    # apply safety mask
    row_mask = rows_local < N_CTX
    o  = tl.load(o_ptrs,  mask=row_mask[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + rows_local, delta, mask=row_mask)



# @triton.autotune(
#     configs=configs,
#     key=["N_CTX", "HEAD_DIM", "USE_EXP2", "HAS_BIAS", "HAS_BLANK", "NEIGHBOR_SIZE"],
#     prune_configs_by={'early_config_prune': prune_invalid_configs},
#     reset_to_zero=['DQ', 'DK', 'DV', 'DBIAS', 'DBK', 'DBV'],
#     cache_results=True
# )
@triton.heuristics(values={
    'BLOCK_M': lambda args: 16,
    'BLOCK_N': lambda args: smallest_valid_power_2(args['NEIGHBOR_SIZE']),
    'num_warps': lambda args: 4,
    'num_stages': lambda args: 4
})
@triton.jit
def _nbhood_attn_bwd_internal(
    Q, K, V, sm_scale,
    DO,
    DQ, DK, DV,
    LSE, Delta,
    member_index, bias, mask, DBIAS,
    blank_k, blank_v, DBK, DBV,
    Z, H, N_CTX, NEIGHBOR_SIZE,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    USE_EXP2: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_BLANK: tl.constexpr,
    HAS_MASK: tl.constexpr
):

        # program coords
    start_m = tl.program_id(0)
    off_hz  = tl.program_id(1)
    off_z   = off_hz // H
    off_h   = off_hz % H

    offset_y_kv = off_z * (N_CTX * H) + off_h * N_CTX
    offset_y_mi = off_z * N_CTX

    rows_local = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows_y     = (offset_y_kv + rows_local).to(tl.int32)
    rows_mask = rows_local < N_CTX

    d = tl.arange(0, HEAD_DIM)
    d = tl.multiple_of(d, 8)

    # fp16 IO
    q  = tl.load(Q  + rows_y[:, None] * HEAD_DIM + d[None, :], mask=rows_mask[:, None], other=0.0).to(tl.float32)
    do = tl.load(DO + rows_y[:, None] * HEAD_DIM + d[None, :], mask=rows_mask[:, None], other=0.0).to(tl.float32)
    lse_row   = tl.load(LSE   + rows_y, mask=rows_mask, other=0.0)
    delta_row = tl.load(Delta + off_hz * N_CTX + rows_local, mask=rows_mask, other=0.0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    desc_k = K  # tl.make_tensor_descriptor(K, shape=[Z*H*N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[1, HEAD_DIM])
    desc_v = V  # tl.make_tensor_descriptor(V, shape=[Z*H*N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[1, HEAD_DIM])

    base_mi = member_index + offset_y_mi * NEIGHBOR_SIZE

    for nb in tl.range(0, NEIGHBOR_SIZE, BLOCK_N):
        nb = tl.multiple_of(nb, BLOCK_N)
        for j in tl.range(0, BLOCK_N):
            # Vectorization hint for neighbor index
            # j = tl.multiple_of(j, 8)
            neighbor_mask = (nb + j) < NEIGHBOR_SIZE
            masking = rows_mask & neighbor_mask

            if HAS_MASK:
                # per-neighborhood mask loading
                base_mask = mask + offset_y_kv * NEIGHBOR_SIZE
                mask_col_bp = tl.make_block_ptr(
                    base        = base_mask + (nb + j),
                    shape       = (N_CTX,),
                    strides     = (NEIGHBOR_SIZE,),
                    offsets     = (start_m * BLOCK_M,),
                    block_shape = (BLOCK_M,),
                    order       = (0,),
                )
                mask_col = tl.load(mask_col_bp, boundary_check=(0,))
                masking = masking & (mask_col != 0)

            mi_col = tl.make_block_ptr(
                base=base_mi + (nb + j),
                shape=(N_CTX,), strides=(NEIGHBOR_SIZE,),
                offsets=(start_m * BLOCK_M,),
                block_shape=(BLOCK_M,), order=(0,),
            )
            idx_col = tl.load(mi_col, boundary_check=(0,)).to(tl.int32)
            # idx_col = tl.where(masking, idx_col, 0)

            # gather k and v based off of neighbor indices
            rows_kv = (offset_y_kv + idx_col).to(tl.int32)
            # rows_kv = tl.where(masking, rows_kv, 0)

            # k_block = desc_k.gather(rows_kv, y0)
            # v_block = desc_v.gather(rows_kv, y0)
            k_ptrs = desc_k + rows_kv[:, None] * HEAD_DIM + d[None, :]
            v_ptrs = desc_v + rows_kv[:, None] * HEAD_DIM + d[None, :]
            k_block = tl.load(k_ptrs, mask=masking[:, None], other=0).to(tl.float32)
            v_block = tl.load(v_ptrs, mask=masking[:, None], other=0).to(tl.float32)

            # calculate logits
            qk = tl.sum(q * k_block, axis=1) * sm_scale
            qk = tl.where(masking, qk, -1e+11)

            # add bias to logits
            if HAS_BIAS:
                base_bias = bias + (offset_y_kv * NEIGHBOR_SIZE)
                bias_col_bp = tl.make_block_ptr(
                    base        = base_bias + (nb + j),
                    shape       = (N_CTX,),
                    strides     = (NEIGHBOR_SIZE,),
                    offsets     = (start_m * BLOCK_M,),
                    block_shape = (BLOCK_M,),
                    order       = (0,),
                )
                bias_col = tl.load(bias_col_bp, boundary_check=(0,)).to(tl.float32)
                bias_col = tl.where(masking, bias_col, 0.0)
                sm = qk + bias_col
            else:
                sm = qk

            # calculate probabilities from lse
            if USE_EXP2:
                p  = tl.math.exp2(sm - lse_row)
            else:
                p  = tl.math.exp(sm - lse_row)

            # calculate delta probabilities
            dp = tl.sum(do * v_block, axis=1)
            ds = p * (dp - delta_row)

            # accumulate gradients
            dq += ds[:, None] * k_block

            dk_add = (ds[:, None] * q) * sm_scale
            dv_add = (p[:, None]  * do)

            dk_ptrs = DK + rows_kv[:, None] * HEAD_DIM + d[None, :]
            dv_ptrs = DV + rows_kv[:, None] * HEAD_DIM + d[None, :]

            tl.atomic_add(dk_ptrs, dk_add, mask=masking[:, None])
            tl.atomic_add(dv_ptrs, dv_add,  mask=masking[:, None])

            if HAS_BIAS:
                db_rows = (offset_y_kv + rows_local).to(tl.int32)
                tl.atomic_add(DBIAS + db_rows * NEIGHBOR_SIZE + (nb + j), ds, mask=masking)

    # Blank grads (identical branching)
    if HAS_BLANK:
        bk = tl.load(blank_k + off_h * HEAD_DIM + d).to(tl.float32)
        bv = tl.load(blank_v + off_h * HEAD_DIM + d).to(tl.float32)

        qk_b = tl.sum(q * bk[None, :], axis=1) * sm_scale
        qk_b = tl.where(rows_mask, qk_b, -1e+11)

        if USE_EXP2:
            p_b = tl.math.exp2(qk_b - lse_row)
        else:
            p_b = tl.math.exp(qk_b - lse_row)

        dp_b = tl.sum(do * bv[None, :], axis=1)
        dp_b = tl.where(rows_mask, dp_b, 0.0)
        ds_b = p_b * (dp_b - delta_row)

        dq += ds_b[:, None] * bk[None, :]

        dbk_add = (ds_b[:, None] * q) * sm_scale
        dbk_add = tl.where(rows_mask[:, None], dbk_add, 0.0)
        dbv_add = (p_b[:, None]  * do)
        dbv_add = tl.where(rows_mask[:, None], dbv_add, 0.0)

        tl.atomic_add(DBK + off_h * HEAD_DIM + d, tl.sum(dbk_add, axis=0))
        tl.atomic_add(DBV + off_h * HEAD_DIM + d, tl.sum(dbv_add, axis=0))

    # dQ store
    tl.store(DQ + rows_y[:, None] * HEAD_DIM + d[None, :], dq * sm_scale, mask=rows_mask[:, None])


def _nbhood_attn_bwd(
    Q, K, V, O, sm_scale,
    DO,
    DQ, DK, DV,
    LSE, Delta,
    member_index, bias, mask, DBIAS,
    blank_k, blank_v, DBK, DBV,
    Z, H, N_CTX, NEIGHBOR_SIZE,
    HEAD_DIM: tl.constexpr, USE_EXP2: tl.constexpr,
    HAS_BIAS: tl.constexpr, HAS_BLANK: tl.constexpr,
    HAS_MASK: tl.constexpr
):
    assert not USE_EXP2, "EXP2 is not supported in backward pass... yet"

    # Descriptors expect plain pointers; we pass flattened Q/K/V/DO/O like in forward
    # Preprocess kernel grid
    def grid_pre(META):
        assert META['BLOCK_M'] <= N_CTX, f"BLOCK_M ({META['BLOCK_M']}) must be less than or equal to N_CTX ({N_CTX})"
        return (triton.cdiv(N_CTX, META['BLOCK_M']), Z * H)

    # NOTE: Preprocess works on row-major [B*H, N, D]
    _nbhood_attn_bwd_preprocess[grid_pre](
        O=O, DO=DO,
        Delta=Delta, Z=Z, H=H, N_CTX=N_CTX, HEAD_DIM=HEAD_DIM
    )
    # 128, 1024, 512, 32, 64: BLOCK_M: 32, num_warps: 8, num_ctas: 1, num_stages: 3, maxnreg: None
    # 128, 4096, 64, 32, 64: BLOCK_M: 32, num_warps: 8, num_ctas: 1, num_stages: 4, maxnreg: None
    # 128, 256, 512, 32, 64: BLOCK_M: 32, num_warps: 8, num_ctas: 1, num_stages: 4, maxnreg: None
    # print('best preprocess', _nbhood_attn_bwd_preprocess.best_config)
    # heuristic: BLOCK_M: 32, num_warps: 8, num_ctas: 1, num_stages: 4

    # Main backward grid
    def grid_bwd(META):
        assert META['BLOCK_M'] <= N_CTX, f"BLOCK_M ({META['BLOCK_M']}) must be less than or equal to N_CTX ({N_CTX})"
        return (triton.cdiv(N_CTX, META['BLOCK_M']), Z * H)

    _nbhood_attn_bwd_internal[grid_bwd](
        Q=Q, K=K, V=V, sm_scale=sm_scale,
        DO=DO,
        DQ=DQ, DK=DK, DV=DV,
        LSE=LSE, Delta=Delta,
        member_index=member_index, bias=bias, mask=mask, DBIAS=DBIAS,
        blank_k=blank_k, blank_v=blank_v, DBK=DBK, DBV=DBV,
        Z=Z, H=H, N_CTX=N_CTX, NEIGHBOR_SIZE=NEIGHBOR_SIZE,
        HEAD_DIM=HEAD_DIM, USE_EXP2=USE_EXP2, HAS_BIAS=HAS_BIAS, HAS_BLANK=HAS_BLANK, HAS_MASK=HAS_MASK
    )
    # 128, 4096, 64, 32, 64:  BLOCK_M: 16, BLOCK_N: 32, num_warps: 4, num_ctas: 1, num_stages: 4, maxnreg: None
    # 128, 1024, 512, 32, 64: BLOCK_M: 16, BLOCK_N: 64, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None
    # 128, 256, 512, 32, 64:  BLOCK_M: 16, BLOCK_N: 16, num_warps: 4, num_ctas: 1, num_stages: 4, maxnreg: None
    # print('best bwd', _nbhood_attn_bwd_internal.best_config)
    # heuristic: BLOCK_M: 16, BLOCK_N: 32, num_warps: 4, num_ctas: 1, num_stages: 4


def launch_flash_nbhood_attn_fwd(q, k, v, member_index, sm_scale, bias=None, mask=None,blank_k=None, blank_v=None, exp2=False):
    # check constraints
    assert q.shape == k.shape and k.shape == v.shape, "q, k, v must have the same shape"
    assert q.dim() == 4 and q.shape == k.shape and q.shape == v.shape, "q, k, v must have 4 dimensions"
    assert q.shape[-1] == k.shape[-1] and k.shape[-1] == v.shape[-1], "q, k, v must have the same head dimension"
    assert q.shape[-1] % 8 == 0, "head_dim must be a multiple of 8"
    batch_size, n_heads, seq_len, head_dim = q.shape
    neighbor_size = member_index.shape[-1]
    assert neighbor_size % 8 == 0, "neighbor_size must be a multiple of 8"

    # optionally add per head bias to the logits
    has_bias = True
    if bias is not None and bias.numel() > 0:
        assert bias.shape == (batch_size, n_heads, seq_len, neighbor_size), "bias must have shape [batch_size, n_heads, seq_len, neighbor_size]"
        bias_2d = bias.view(-1, seq_len, neighbor_size).contiguous()  # [B*H, N, M]
    else:
        bias_2d = torch.tensor([], device=q.device, dtype=q.dtype)  # empty tensor
        has_bias = False

    # add the per neighborhood blank token to the logits
    has_blank = True
    if blank_k is not None and blank_v is not None and blank_k.numel() > 0 and blank_v.numel() > 0:
        assert blank_k.shape == (n_heads, head_dim), "blank_k must have shape [n_heads, head_dim]"
        assert blank_v.shape == (n_heads, head_dim), "blank_v must have shape [n_heads, head_dim]"
        blank_k_2d = blank_k.view(-1, head_dim).contiguous()  # [B*H, D]
        blank_v_2d = blank_v.view(-1, head_dim).contiguous()  # [B*H, D]
    else:
        has_blank = False
        blank_k_2d = torch.tensor([], device=q.device, dtype=q.dtype)  # empty tensor
        blank_v_2d = torch.tensor([], device=q.device, dtype=q.dtype)  # empty tensor

    # add the per neighborhood mask
    has_mask = True
    if mask is not None and mask.numel() > 0:
        # allow [B,1,N,M] or [B,H,N,M]
        if mask.shape[1] == 1:
            mask = mask.expand(batch_size, n_heads, seq_len, neighbor_size)
        assert mask.shape == (batch_size, n_heads, seq_len, neighbor_size), f"mask must have shape [batch_size, n_heads, seq_len, neighbor_size] (mask shape: {mask.shape}) and (batch_size: {batch_size}, n_heads: {n_heads}, seq_len: {seq_len}, neighbor_size: {neighbor_size})"
        mask_2d = (mask > 0).to(torch.bool).contiguous().reshape(batch_size*n_heads, seq_len, neighbor_size)  # [B*H, N, M]
    else:
        has_mask = False
        mask_2d = torch.tensor([], device=q.device, dtype=q.dtype)  # empty tensor

    # kernel expects inputs as [batch_size * n_heads, seq_len, head_dim]
    q = q.view(-1, seq_len, head_dim).contiguous()
    k = k.view(-1, seq_len, head_dim).contiguous()
    v = v.view(-1, seq_len, head_dim).contiguous()
    
    # Pad to a multiple of 8 for head_dim if necessary (required for triton kernel)
    # NOTE: this currently does not do anything due to the current assert. @TODO: test this
    padded = False
    if head_dim % 8 != 0:
        original_head_dim = head_dim
        head_dim = (head_dim // 8 + 1) * 8
        q = torch.nn.functional.pad(q, (0, head_dim - original_head_dim))
        k = torch.nn.functional.pad(k, (0, head_dim - original_head_dim))
        v = torch.nn.functional.pad(v, (0, head_dim - original_head_dim))
        padded = True
    else:
        original_head_dim = head_dim

    # allocate output tensor
    o = torch.zeros_like(q, dtype=torch.float32)
    lse = torch.zeros((batch_size*n_heads, seq_len), device=q.device, dtype=torch.float32)  # keep LSE in fp32 for backward pass

    def grid(META):
        # check constraints before computing grid
        assert META['BLOCK_M'] <= seq_len, f"BLOCK_M ({META['BLOCK_M']}) must be less than or equal to seq_len ({seq_len})"
        assert META['BLOCK_N'] <= neighbor_size, f"BLOCK_N ({META['BLOCK_N']}) must be less than or equal to neighbor_size ({neighbor_size})"
        
        # compute grid
        return (triton.cdiv(seq_len, META['BLOCK_M']), batch_size * n_heads)
    
    # launch nbhood kernel
    _nbhood_attn_fwd[grid](
        sm_scale=sm_scale,
        Z=batch_size, H=n_heads, q=q,
        k=k, v=v, member_index=member_index, bias=bias_2d, mask=mask_2d, blank_k=blank_k_2d, blank_v=blank_v_2d, o=o, lse=lse,
        N_CTX=seq_len, NEIGHBOR_SIZE=neighbor_size,
        HEAD_DIM=head_dim,
        warp_specialize=False,
        USE_EXP2=exp2,
        HAS_BIAS=has_bias,
        HAS_BLANK=has_blank,
        HAS_MASK=has_mask
    )
    # 128, 1024, 512, 32, 64: best fwd BLOCK_M: 16, BLOCK_N: 32, num_warps: 1, num_ctas: 1, num_stages: 1, maxnreg: None
    # 128, 4096, 64, 32, 64: BLOCK_M: 16, BLOCK_N: 32, num_warps: 1, num_ctas: 1, num_stages: 3, maxnreg: None
    # 128, 256, 512, 32, 64: BLOCK_M: 16, BLOCK_N: 16, num_warps: 1, num_ctas: 1, num_stages: 2, maxnreg: None
    # print('best fwd', _nbhood_attn_fwd.best_config)
    # heuristic: BLOCK_M: 16, BLOCK_N: 32, num_warps: 1, num_ctas: 1, num_stages: 2
    
    # Unpad the output if head dimension was padded
    if padded:
        o = o[:, :, :original_head_dim]
    
    # convert o to dtype
    o = o.to(q.dtype)

    return o.reshape(batch_size, n_heads, seq_len, original_head_dim), lse.reshape(batch_size, n_heads, seq_len)


def launch_flash_nbhood_attn_bwd(
    q, k, v,                      # [B,H,N,D]
    member_index,                 # [B,N,M] (int32)
    sm_scale,
    o, lse,                       # forward outputs: o [B,H,N,D], lse [B,H,N]
    do,                           # upstream grad dO [B,H,N,D]
    bias=None,                    # [B,H,N,M] or None
    mask=None,                    # [B,H,N,M] or None
    blank_k=None, blank_v=None,   # [H,D] or None
    exp2=False,
):
    # ---- Sanity & shapes ----
    assert q.shape == k.shape == v.shape == o.shape == do.shape
    assert q.dim() == 4
    B, H, N, D = q.shape
    M = member_index.shape[-1]
    device = q.device
    dtype = q.dtype

    # Optional flags and 2D re-views for kernels
    has_bias = (bias is not None) and (bias.numel() > 0)
    if has_bias:
        assert bias.shape == (B, H, N, M)
        bias_1d = bias.view(-1, M).contiguous()        # [B*H*N, M]
    else:
        bias_1d = torch.tensor([], device=device, dtype=dtype)

    has_blank = (blank_k is not None) and (blank_v is not None)
    if has_blank:
        assert blank_k.shape == (H, D) and blank_v.shape == (H, D)
        # grads for blank_k/blank_v (per head)
    else:
        blank_k = torch.tensor([], device=device, dtype=dtype)
        blank_v = torch.tensor([], device=device, dtype=dtype)

    # add the per neighborhood mask
    has_mask = True
    if mask is not None and mask.numel() > 0:
        # allow [B,1,N,M] or [B,H,N,M]
        if mask.shape[1] == 1:
            mask = mask.expand(B, H, N, M)
        assert mask.shape == (B, H, N, M), "mask must have shape [B, H, N, M]"
        mask_2d = (mask > 0).to(torch.bool).contiguous().reshape(B*H, N, M)  # [B*H, N, M]
    else:
        has_mask = False
        mask_2d = torch.tensor([], device=q.device, dtype=torch.bool)  # empty tensor

    # Make contiguous like forward launcher
    q_  = q.contiguous().view(-1, N, D)     # [B*H, N, D]
    k_  = k.contiguous().view(-1, N, D)
    v_  = v.contiguous().view(-1, N, D)
    o_  = o.contiguous().view(-1, N, D)
    do_ = do.contiguous().view(-1, N, D)
    lse_ = lse.contiguous().view(-1, N)     # [B*H, N]
    mi  = member_index.contiguous()         # [B, N, M]

    # Head-dim padding (same policy as forward)
    padded = False
    D_orig = D
    if D & (D - 1):
        D_pad = int(2 ** (D - 1).bit_length())
        pad = (0, D_pad - D)
        q_  = torch.nn.functional.pad(q_,  pad).contiguous()
        k_  = torch.nn.functional.pad(k_,  pad).contiguous()
        v_  = torch.nn.functional.pad(v_,  pad).contiguous()
        o_  = torch.nn.functional.pad(o_,  pad).contiguous()
        do_ = torch.nn.functional.pad(do_, pad).contiguous()
        if has_blank:
            blank_k = torch.nn.functional.pad(blank_k, pad).contiguous()
            blank_v = torch.nn.functional.pad(blank_v, pad).contiguous()
        D = D_pad
        padded = True

    # Allocate grads (flattened)
    dq_ = torch.zeros_like(q_, dtype=torch.float32)             # [B*H, N, D]
    dk_ = torch.zeros_like(k_, dtype=torch.float32)
    dv_ = torch.zeros_like(v_, dtype=torch.float32)
    # Bias grad (optional)
    if has_bias:
        dbias_ = torch.zeros((B*H, N, M), device=device, dtype=torch.float32)
    else:
        dbias_ = torch.tensor([], device=device, dtype=torch.float32)
    # Blank grads (optional, per head)
    if has_blank:
        dbk = torch.zeros((H, D), device=device, dtype=torch.float32)
        dbv = torch.zeros((H, D), device=device, dtype=torch.float32)
    else:
        dbk = torch.tensor([], device=device, dtype=torch.float32)
        dbv = torch.tensor([], device=device, dtype=torch.float32)

    # Delta buffer (B*H, N) in float32
    Delta = torch.zeros((B*H, N), device=device, dtype=torch.float32)

    # call the adaptive launcher, which adjusts grid and masking depending on block dimensions
    _nbhood_attn_bwd(
        Q=q_, K=k_, V=v_, O=o_, sm_scale=sm_scale,
        DO=do_,
        DQ=dq_, DK=dk_, DV=dv_,
        LSE=lse_, Delta=Delta,
        member_index=mi, bias=bias_1d, mask=mask_2d, DBIAS=dbias_,
        blank_k=blank_k, blank_v=blank_v, DBK=dbk, DBV=dbv,
        Z=B, H=H, N_CTX=N, NEIGHBOR_SIZE=M,
        HEAD_DIM=D, USE_EXP2=exp2, HAS_BIAS=has_bias, HAS_BLANK=has_blank, HAS_MASK=has_mask
    )

    # Unpad grads if needed
    if padded:
        dq_ = dq_[:, :, :D_orig]
        dk_ = dk_[:, :, :D_orig]
        dv_ = dv_[:, :, :D_orig]
        if has_blank:
            dbk = dbk[:, :D_orig]
            dbv = dbv[:, :D_orig]

    # Reshape back
    dq = dq_.view(B, H, N, D_orig).to(dtype)
    dk = dk_.view(B, H, N, D_orig).to(dtype)
    dv = dv_.view(B, H, N, D_orig).to(dtype)
    if has_bias:
        dbias = dbias_.view(B, H, N, M).to(dtype)
    else:
        dbias = None
    if has_blank:
        dblank_k = dbk.to(dtype)
        dblank_v = dbv.to(dtype)
    else:
        dblank_k = None
        dblank_v = None

    return dq.contiguous(), dk.contiguous(), dv.contiguous(), dbias.contiguous() if has_bias else None, dblank_k.contiguous() if has_blank else None, dblank_v.contiguous() if has_blank else None


class FlashLocalAttentionFunction(Function):
    """
    Flash-attention style local/neighborhood attention with optional bias and blank tokens.
    """
    
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float16)
    def forward(ctx, Q, K, V, member_idx, bias, mask, blank_k, blank_v, softmax_scale):
        """
        Args:
            Q, K, V: [B, H, N, D] query, key, value tensors
            member_idx: [B, N, M] neighborhood indices (int32)
            bias: [B, H, N, M] optional positional bias
            mask: [B, 1/H, N, M] optional neighborhood mask
            blank_k, blank_v: [H, D] optional per-head blank token
            softmax_scale: scaling factor for attention scores
        """
        # Extract shapes
        B, H, N, D = Q.shape
        M = member_idx.shape[-1]
        
        # Check optional inputs
        has_bias = bias is not None and bias.numel() > 0
        has_blank = blank_k is not None and blank_v is not None
        has_mask = mask is not None and mask.numel() > 0

        # Ensure contiguous memory layout
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        member_idx = member_idx.to(torch.int32).contiguous()
        bias = bias.contiguous() if has_bias else None
        mask = mask.contiguous() if has_mask else None
        blank_k = blank_k.contiguous() if has_blank else None
        blank_v = blank_v.contiguous() if has_blank else None

        # Forward pass
        out, lse = launch_flash_nbhood_attn_fwd(
            Q, K, V, member_idx, softmax_scale, 
            bias=bias, mask=mask,
            blank_k=blank_k, 
            blank_v=blank_v, 
            exp2=False
        )

        # Save tensors and metadata for backward pass
        ctx.save_for_backward(Q, K, V, member_idx, bias, mask, blank_k, blank_v, out, lse)
        ctx.scale = softmax_scale
        ctx.shape_params = (B, H, N, M, D)
        ctx.optional_flags = (has_bias, has_blank, has_mask)
        
        return out

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        # Restore saved tensors
        Q, K, V, member_idx, bias, mask, blank_k, blank_v, out, lse = ctx.saved_tensors
        B, H, N, M, D = ctx.shape_params
        softmax_scale = ctx.scale
        has_bias, has_blank, has_mask = ctx.optional_flags

        # Ensure contiguous
        grad_output = grad_output.contiguous()
        lse = lse.contiguous()

        # Backward pass
        dq, dk, dv, dbias, dblank_k, dblank_v = launch_flash_nbhood_attn_bwd(
            Q, K, V, member_idx, softmax_scale,
            out, lse, grad_output,
            bias=bias if has_bias else None,
            mask=mask if has_mask else None,
            blank_k=blank_k if has_blank else None,
            blank_v=blank_v if has_blank else None,
            exp2=False
        )
        
        # Cast gradients back to input dtypes
        dq = dq.to(Q.dtype)
        dk = dk.to(K.dtype)
        dv = dv.to(V.dtype)
        
        if has_bias:
            dbias = dbias.to(bias.dtype)
        
        if has_blank:
            dblank_k = dblank_k.to(blank_k.dtype)
            dblank_v = dblank_v.to(blank_v.dtype)
        else:
            dblank_k = None
            dblank_v = None

        # Return gradients (None for non-differentiable inputs)
        return dq, dk, dv, None, dbias, None, dblank_k, dblank_v, None


def test_local_attention():
    # Test parameters
    batch_size = 5
    n_heads = 2
    n_tokens = 128
    head_dim = 64
    neighbor_size = 64
    sm_scale = 0.5
    exp2 = False
    dtype = torch.float16
    
    # Create random input tensors
    q = torch.randn(batch_size, n_heads, n_tokens, head_dim, device='cuda', dtype=dtype)
    k = torch.randn(batch_size, n_heads, n_tokens, head_dim, device='cuda', dtype=dtype)
    v = torch.randn(batch_size, n_heads, n_tokens, head_dim, device='cuda', dtype=dtype)
    with torch.no_grad():
        q[:, :, :, :10] = 1.0
        k[:, :, :, :5] = 0.1
        v[:, :, :, :5] = 1.0
    # create random per-head blank token
    blank_k = torch.randn(n_heads, head_dim, device='cuda', dtype=dtype) * 5.0
    blank_v = torch.randn(n_heads, head_dim, device='cuda', dtype=dtype) * 5.0

    # Random positional bias [B, H, N, M]
    bias = torch.randn(batch_size, n_heads, n_tokens, neighbor_size, device='cuda', dtype=dtype) * 5.0
    with torch.no_grad():
        bias[:, :, :, :5] = 1.0
    bias = bias.requires_grad_()
    
    # Create a random member_index tensor
    member_index = torch.randint(0, n_tokens, (batch_size, n_tokens, neighbor_size), 
                                 device='cuda', dtype=torch.int32)

    # Random neighborhood mask [B, 1/H, N, M]
    # mask = torch.randint(0, 2, (batch_size, 1, n_tokens, neighbor_size), device='cuda', dtype=torch.int32)
    mask = torch.ones(batch_size, 1, n_tokens, neighbor_size, device='cuda', dtype=torch.int32)

    # first 10 neighborhood tokens are masked
    # mask[:, :, :, :10] = 0

    # -------- PyTorch baseline (autograd) --------
    # Gather neighbors
    q_exp = q.unsqueeze(3)  # [B,H,N,1,D]
    mi_exp = member_index.unsqueeze(1).expand(-1, n_heads, -1, -1)
    k_gath = torch.gather(
        k.unsqueeze(2).expand(-1, -1, n_tokens, -1, -1),
        dim=3,
        index=mi_exp.unsqueeze(4).expand(-1, -1, -1, -1, head_dim)
    )
    v_gath = torch.gather(
        v.unsqueeze(2).expand(-1, -1, n_tokens, -1, -1),
        dim=3,
        index=mi_exp.unsqueeze(4).expand(-1, -1, -1, -1, head_dim)
    )
    # logits for neighbors + bias
    scores = torch.matmul(q_exp, k_gath.transpose(-2, -1)).squeeze(3) * sm_scale  # [B,H,N,M]
    scores = scores + bias  # [B,H,N,M]
    # blank logits
    blank_logits = (q * blank_k.view(1, n_heads, 1, head_dim)).sum(-1, keepdim=True) * sm_scale  # [B,H,N,1]
    # mask the scores
    scores = scores.masked_fill(mask.expand_as(scores) == 0, float('-inf'))
    scores_plus = torch.cat([scores.unsqueeze(3), blank_logits.unsqueeze(3)], dim=-1)  # [B,H,N,1,M+1]
    attn_plus = torch.softmax(scores_plus, dim=-1).squeeze(3)  # [B,H,N,M+1]
    attn_nb, attn_blank = attn_plus[..., :-1], attn_plus[..., -1:]  # [B,H,N,M], [B,H,N,1]

    out_nb = torch.matmul(attn_nb.unsqueeze(3), v_gath).squeeze(3)  # [B,H,N,D]
    out_blank = attn_blank * blank_v.view(1, n_heads, 1, head_dim)               # [B,H,N,D]
    pytorch_output = out_nb + out_blank                                    # [B,H,N,D]

    # --- Triton local attention version ---
    triton_output, triton_lse = launch_flash_nbhood_attn_fwd(q, k, v, member_index, sm_scale, bias=bias, mask=mask, blank_k=blank_k, blank_v=blank_v, exp2=exp2)

    # calculate the logsumexp
    pytorch_lse = torch.logsumexp(scores_plus.float(), dim=-1)
    print('pytorch_lse: ', pytorch_lse.flatten()[:10], pytorch_lse.shape)
    print('triton_lse: ', triton_lse.flatten()[:10], triton_lse.shape)
    assert_close(pytorch_lse.squeeze(-1), triton_lse, atol=1e-2, rtol=0)
    print("Test passed! Triton local attention logsumexp matches PyTorch gathered version.")

    # --- Comparison ---
    print('torch_output:  ', pytorch_output[0, 1, 2].flatten()[:10])
    print('triton_output: ', triton_output[0, 1, 2].flatten()[:10])
    assert_close(pytorch_output, triton_output, atol=1e-2, rtol=0)
    print("Test passed! Triton local attention output matches PyTorch gathered version.")


def test_local_attention_backward_gradnorms():
    # Small-ish case
    batch_size = 8
    n_heads = 6
    n_tokens = 215
    head_dim = 128
    neighbor_size = 8
    sm_scale = 0.5
    dtype = torch.float64
    device = "cuda"
    exp2 = False

    # Create random input tensors
    q = torch.randn(batch_size, n_heads, n_tokens, head_dim, device='cuda', dtype=dtype).requires_grad_()
    k = torch.randn(batch_size, n_heads, n_tokens, head_dim, device='cuda', dtype=dtype).requires_grad_()
    v = torch.randn(batch_size, n_heads, n_tokens, head_dim, device='cuda', dtype=dtype).requires_grad_()
    with torch.no_grad():
        q[:, :, :, :10] = 1.0
        k[:, :, :, :5] = 0.1
        v[:, :, :, :5] = 1.0
    # create random per-head blank token
    blank_k = (torch.randn(n_heads, head_dim, device='cuda', dtype=dtype) * 0.0).requires_grad_()
    blank_v = (torch.randn(n_heads, head_dim, device='cuda', dtype=dtype) * 0.0).requires_grad_()

    # Random positional bias [B, H, N, M]
    bias = (torch.randn(batch_size, n_heads, n_tokens, neighbor_size, device='cuda', dtype=dtype) * 0.0).requires_grad_()
    with torch.no_grad():
        bias[:, :, :, :5] = 1.0
    bias = bias.requires_grad_()
    
    # Create a random member_index tensor
    member_index = torch.randint(0, n_tokens, (batch_size, n_tokens, neighbor_size), 
                                 device='cuda', dtype=torch.int32)

    # Random neighborhood mask [B, 1/H, N, M]
    # mask = torch.randint(0, 2, (batch_size, 1, n_tokens, neighbor_size), device='cuda', dtype=torch.int32)
    mask = torch.ones(batch_size, 1, n_tokens, neighbor_size, device='cuda', dtype=torch.int32)

    # first 10 neighborhood tokens are masked
    # mask[:, :, :, :10] = 0

    # Upstream gradient dO
    do = torch.randn(batch_size, n_heads, n_tokens, head_dim, device=device, dtype=dtype)
    do[:, :, :, :5] = 1.0
    do[:, :, :, 5:] = -1.0

    # -------- PyTorch baseline (autograd) --------
    # Gather neighbors
    # -------- PyTorch baseline (autograd) --------
    # Gather neighbors
    q_exp = q.unsqueeze(3)  # [B,H,N,1,D]
    mi_exp = member_index.unsqueeze(1).expand(-1, n_heads, -1, -1)
    k_gath = torch.gather(
        k.unsqueeze(2).expand(-1, -1, n_tokens, -1, -1),
        dim=3,
        index=mi_exp.unsqueeze(4).expand(-1, -1, -1, -1, head_dim)
    )
    v_gath = torch.gather(
        v.unsqueeze(2).expand(-1, -1, n_tokens, -1, -1),
        dim=3,
        index=mi_exp.unsqueeze(4).expand(-1, -1, -1, -1, head_dim)
    )
    # logits for neighbors + bias
    scores = torch.matmul(q_exp, k_gath.transpose(-2, -1)).squeeze(3) * sm_scale  # [B,H,N,M]
    scores = scores + bias  # [B,H,N,M]
    # blank logits
    blank_logits = (q * blank_k.view(1, n_heads, 1, head_dim)).sum(-1, keepdim=True) * sm_scale  # [B,H,N,1]
    # mask the scores
    scores = scores.masked_fill(mask.expand_as(scores) == 0, float('-inf'))
    scores_plus = torch.cat([scores.unsqueeze(3), blank_logits.unsqueeze(3)], dim=-1)  # [B,H,N,1,M+1]
    attn_plus = torch.softmax(scores_plus, dim=-1).squeeze(3)  # [B,H,N,M+1]
    attn_nb, attn_blank = attn_plus[..., :-1], attn_plus[..., -1:]  # [B,H,N,M], [B,H,N,1]

    out_nb = torch.matmul(attn_nb.unsqueeze(3), v_gath).squeeze(3)  # [B,H,N,D]
    out_blank = attn_blank * blank_v.view(1, n_heads, 1, head_dim)               # [B,H,N,D]
    out_ref = out_nb + out_blank                             # [B,H,N,D]

    # backward pass
    grads = torch.autograd.grad(out_ref, grad_outputs=do, inputs=[q, k, v, bias, blank_k, blank_v], retain_graph=False, allow_unused=True)

    dq_ref, dk_ref, dv_ref, dbias_ref, dblankk_ref, dblankv_ref = grads
    # Some may be None if graph pruned; replace with zeros for norm compare
    dq_ref = dq_ref if dq_ref is not None else torch.zeros_like(q)
    dk_ref = dk_ref if dk_ref is not None else torch.zeros_like(k)
    dv_ref = dv_ref if dv_ref is not None else torch.zeros_like(v)
    dbias_ref = dbias_ref if dbias_ref is not None else torch.zeros_like(bias)
    dblankk_ref = dblankk_ref if dblankk_ref is not None else torch.zeros_like(blank_k)
    dblankv_ref = dblankv_ref if dblankv_ref is not None else torch.zeros_like(blank_v)

    # -------- Triton path (forward + backward) --------
    # Forward (for o, lse)
    q = q.float()
    k = k.float()
    v = v.float()
    bias = bias.float()
    mask = mask.float()
    blank_k = blank_k.float()
    blank_v = blank_v.float()
    member_index = member_index.to(torch.int32)
    with torch.no_grad():
        triton_out, triton_lse = launch_flash_nbhood_attn_fwd(q, k, v, member_index, sm_scale, bias=bias, mask=mask, blank_k=blank_k, blank_v=blank_v, exp2=exp2)

    # Backward launcher
    with torch.no_grad():
        dq, dk, dv, dbias, dblankk, dblankv = launch_flash_nbhood_attn_bwd(q, k, v, member_index, sm_scale, triton_out, triton_lse, do, bias=bias, mask=mask, blank_k=blank_k, blank_v=blank_v, exp2=exp2)

    # convert out ref to fp32
    out_ref = out_ref.float()
    dq_ref = dq_ref.float()
    dk_ref = dk_ref.float()
    dv_ref = dv_ref.float()
    dbias_ref = dbias_ref.float()
    dblankk_ref = dblankk_ref.float()
    dblankv_ref = dblankv_ref.float()

    # -------- Compare grad norms --------
    def norms(x): return (x.norm(p=2).item(), x.abs().max().item())
    print('forward ref: ', out_ref[0, 1, 2].flatten()[:10], out_ref.shape)
    print('forward triton: ', triton_out[0, 1, 2].flatten()[:10], triton_out.shape)
    print('lse ref: ', torch.logsumexp(scores_plus, dim=-1).flatten()[:10], scores_plus.shape)
    print('lse triton: ', triton_lse.flatten()[:10], triton_lse.shape)
    # exit(0)
    print("||dq||:", norms(dq_ref), "vs", norms(dq))
    print('dq_ref: ', dq_ref[0, 1, 2].flatten()[:10], dq_ref.shape)
    print('dq: ', dq[0, 1, 2].flatten()[:10], dq.shape)
    print("||dk||:", norms(dk_ref), "vs", norms(dk))
    print('dk_ref: ', dk_ref.flatten()[:10], dk_ref.shape)
    print('dk: ', dk.flatten()[:10], dk.shape)
    print("||dv||:", norms(dv_ref), "vs", norms(dv))
    print('dv_ref: ', dv_ref.flatten()[:10], dv_ref.shape)
    print('dv: ', dv.flatten()[:10], dv.shape)
    print("||dbias||:", norms(dbias_ref), "vs", norms(dbias if dbias is not None else torch.zeros_like(dbias_ref)))
    print('dbias_ref: ', dbias_ref.flatten()[:10], dbias_ref.shape)
    print('dbias: ', dbias.flatten()[:10], dbias.shape)
    print("||dblankk||:", norms(dblankk_ref), "vs", norms(dblankk if dblankk is not None else torch.zeros_like(dblankk_ref)))
    print('dblankk_ref: ', dblankk_ref.flatten()[:10], dblankk_ref.shape)
    print('dblankk: ', dblankk.flatten()[:10], dblankk.shape)
    print("||dblankv||:", norms(dblankv_ref), "vs", norms(dblankv if dblankv is not None else torch.zeros_like(dblankv_ref)))
    print('dblankv_ref: ', dblankv_ref.flatten()[:10], dblankv_ref.shape)
    print('dblankv: ', dblankv.flatten()[:10], dblankv.shape)

    # Tolerant norm check (kernels use atomics; tiny numeric diffs ok)
    atol, rtol = 5e-2, 5e-3
    def close(a, b): return torch.allclose(torch.tensor(a, device=device), torch.tensor(b, device=device), atol=atol, rtol=rtol)

    assert close(dq_ref.norm(), dq.norm())
    assert close(dk_ref.norm(), dk.norm())
    assert close(dv_ref.norm(), dv.norm())
    assert close(dbias_ref.norm(), (dbias if dbias is not None else torch.zeros_like(dbias_ref)).norm())
    assert close(dblankk_ref.norm(), (dblankk if dblankk is not None else torch.zeros_like(dblankk_ref)).norm())
    assert close(dblankv_ref.norm(), (dblankv if dblankv is not None else torch.zeros_like(dblankv_ref)).norm())

    print("Backward grad-norms match within tolerance.")


if __name__ == "__main__":
    # Run the test
    # for i in range(300):
    # test_local_attention()
    # exit(0)

    # Run the test a few times
    # import time
    test_local_attention_backward_gradnorms()  # warmup
    exit(0)
    # start_time = time.time()
    # for _ in range(30):
    #     test_local_attention_backward_gradnorms()
    # end_time = time.time()
    # print(f"Time taken: {(end_time - start_time) / 30} seconds per iteration")
