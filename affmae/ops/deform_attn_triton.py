"""Deformable point-attention Triton kernels and autograd wrappers.

This module provides:
- `csr_knn_cached` (default): fused forward + KNN cached + CSR dV backward.
- `atomic`: self-contained fused path with atomic dV backward.
"""

import torch
from torch.autograd import Function
from .dispatch import custom_fwd, custom_bwd  # torch-version shim
from torch.profiler import record_function
# Imported through util so a torch-only install can still import this
# module; the kernels raise only when actually launched.
from .dispatch import clamp_num_stages, triton, tl


@triton.jit
def _dense_top4_knn_kernel(
    kv_x_ptr, kv_y_ptr, nn4_ptr,
    n_kv,
    stride_kv_b,
    stride_nn4_b,
    stride_nn4_p,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_P: tl.constexpr
):
    p_total = H * W
    pid = tl.program_id(0)
    tiles_per_b = tl.cdiv(p_total, BLOCK_P)
    b = pid // tiles_per_b
    tile = pid % tiles_per_b

    # tile pixels: [BLOCK_P]
    p = tile * BLOCK_P + tl.arange(0, BLOCK_P)

    # mask for valid pixels when H*W is not a multiple of BLOCK_P
    mask_p = p < p_total

    # query pixel location (px, py) in grid coordinates
    py = p // W
    px = p % W

    # initial top-4 distances and indices (0 as default slot value)
    inf = 2147483647  # max int32 value
    d0 = tl.full([BLOCK_P], inf, dtype=tl.int32)
    d1 = tl.full([BLOCK_P], inf, dtype=tl.int32)
    d2 = tl.full([BLOCK_P], inf, dtype=tl.int32)
    d3 = tl.full([BLOCK_P], inf, dtype=tl.int32)

    # top-4 indices
    i0 = tl.zeros([BLOCK_P], dtype=tl.int32)
    i1 = tl.zeros([BLOCK_P], dtype=tl.int32)
    i2 = tl.zeros([BLOCK_P], dtype=tl.int32)
    i3 = tl.zeros([BLOCK_P], dtype=tl.int32)

    # base pointers for KV coordinates: [B, Nk]
    kv_x_base = kv_x_ptr + b * stride_kv_b
    kv_y_base = kv_y_ptr + b * stride_kv_b

    # iterate over all KV tokens and find the top-4 closest tokens
    for t_idx in range(n_kv):
        # load KV token coordinate t_idx
        tx = tl.load(kv_x_base + t_idx)
        ty = tl.load(kv_y_base + t_idx)

        # get distance
        dx = tx - px
        dy = ty - py
        cand_d = (dx * dx) + (dy * dy)  # squared distance for candidate t_idx
        cand_i = (tl.zeros([BLOCK_P], dtype=tl.int32) + t_idx).to(tl.int32)  # candidate index

        # insertion-style update into sorted (d0,d1,d2,d3) list
        # gist: swap closest distance to front of list then continue swapping with next closest (cheap for N=4 in registers)
        s = cand_d < d0
        cand_d, d0 = tl.where(s, d0, cand_d), tl.where(s, cand_d, d0)
        cand_i, i0 = tl.where(s, i0, cand_i), tl.where(s, cand_i, i0)
        s = cand_d < d1
        cand_d, d1 = tl.where(s, d1, cand_d), tl.where(s, cand_d, d1)
        cand_i, i1 = tl.where(s, i1, cand_i), tl.where(s, cand_i, i1)
        s = cand_d < d2
        cand_d, d2 = tl.where(s, d2, cand_d), tl.where(s, cand_d, d2)
        cand_i, i2 = tl.where(s, i2, cand_i), tl.where(s, cand_i, i2)
        s = cand_d < d3
        d3 = tl.where(s, cand_d, d3)
        i3 = tl.where(s, cand_i, i3)

    # store top-4 indices for this tile
    out = nn4_ptr + b * stride_nn4_b + p * stride_nn4_p
    tl.store(out    , i0.to(tl.uint16), mask=mask_p)
    tl.store(out + 1, i1.to(tl.uint16), mask=mask_p)
    tl.store(out + 2, i2.to(tl.uint16), mask=mask_p)
    tl.store(out + 3, i3.to(tl.uint16), mask=mask_p)


def _warps_for_width(width: int) -> int:
    """Warps needed to cover a ``width``-element vector, at least one.

    Every value in this kernel is a ``[BLOCK_P]`` vector, so a program wider
    than the data leaves lanes idle: ``BLOCK_P=32`` with Triton's default of 4
    warps uses 8 of each warp's 32 lanes.

    See :func:`affmae.ops.launch.warps_for_width` for the device-derived rule.
    
    Args:
        width: elements per program, i.e. BLOCK_P.
    Returns:
        A warp count.
    """
    from .dispatch import is_hip

    lanes = 64 if is_hip() else 32   # wavefront on ROCm, warp on CUDA
    return max(1, width // lanes)


def dense_top4_knn(database: torch.Tensor, H: int = 64, W: int = 64, BLOCK_P: int = 32) -> torch.Tensor:
    """
    Build dense top-4 nearest-neighbor lookup per grid pixel.

    Args:
        database: [B, Nk, 2], integer KV coordinates.
        H: int, grid height.
        W: int, grid width.
        BLOCK_P: int, pixels per program.
    Returns:
        nn4 table of shape [B, H*W, 4] with uint16 token indices.
    """
    with record_function("deform_attn.dense_top4_knn"):
        bsz, n_kv, _ = database.shape
        if n_kv < 4:
            raise ValueError("dense_top4_knn requires at least 4 database tokens (Nk >= 4).")
        if n_kv >= (1 << 16):
            raise ValueError("dense_top4_knn database tokens too large for uint16 indices (Nk must be < 65536).")
        p = H * W
        kv_x = database[:, :, 0].contiguous().to(torch.int32)
        kv_y = database[:, :, 1].contiguous().to(torch.int32)
        nn4 = torch.zeros((bsz, p, 4), dtype=torch.uint16, device=database.device)

        def grid(meta):
            return (bsz * triton.cdiv(p, meta["BLOCK_P"]),)

        _dense_top4_knn_kernel[grid](
            kv_x, kv_y, nn4,
            n_kv,
            kv_x.stride(0),
            nn4.stride(0),
            nn4.stride(1),
            H=H, W=W,
            BLOCK_P=BLOCK_P,
            num_warps=_warps_for_width(BLOCK_P),
        )
        return nn4


# @triton.autotune(
#     configs=_DEFORM_FWD_CONFIGS,
#     key=["Nq", "C", "S", "STORE_FP16"],
#     prune_configs_by={"early_config_prune": _prune_fwd_configs},
#     cache_results=_CACHE_TO_DISK,
# )
@triton.jit
def _deform_attn_fwd_kernel(
    QPOS_X, QPOS_Y,              # [BH, Nq]
    OFFS_X, OFFS_Y,              # [BH, Nq, S]
    ATTN_LOGITS,                 # [BH, Nq, S]
    TAU,                         # [BH]
    NN4_IDX,                     # [B, H*W, 4]
    KV_POS,                      # [BH, Nk, 2]
    V,                           # [BH, Nk, C]
    OUT,                         # [BH, Nq, C]
    EDGE_COEFF,                  # [BH, Nq*S*4] — only alphas; KV/Q derived during CSR build
    Nq, Nk, Hh,
    C: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_qpos_bh, stride_qpos_q,
    stride_oas_bh, stride_oas_q, stride_oas_s,
    stride_nn_bh, stride_nn_p, stride_nn_k,
    stride_kv_bh, stride_kv_n, stride_kv_d,
    stride_v_bh, stride_v_n, stride_v_c,
    stride_o_bh, stride_o_q, stride_o_c,
    stride_coeff_bh, stride_coeff_e,
    BLOCK_Q: tl.constexpr,
    BLOCK_C: tl.constexpr,
    EMIT_EDGES: tl.constexpr,
    EPS: tl.constexpr,
    STORE_FP16: tl.constexpr = True,
):
    # Program tiles: q-block and channel-block for one (batch, head) lane.
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_c = tl.program_id(2)


    q_off = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    c_off = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_q = q_off < Nq
    mask_c = c_off < C

    # Query coordinates and Shepard temperature: [BLOCK_Q]
    qx = tl.load(QPOS_X + pid_bh * stride_qpos_bh + q_off * stride_qpos_q, mask=mask_q, other=0.0).to(tl.float32)
    qy = tl.load(QPOS_Y + pid_bh * stride_qpos_bh + q_off * stride_qpos_q, mask=mask_q, other=0.0).to(tl.float32)
    tau = tl.maximum(tl.load(TAU + pid_bh).to(tl.float32), 0.0) + EPS

    # Softmax(attn_logits) over sampling points S for each query.
    a_max = tl.full([BLOCK_Q], value=-1e30, dtype=tl.float32)
    for s in tl.static_range(S):
        ls = tl.load(ATTN_LOGITS + pid_bh * stride_oas_bh + q_off * stride_oas_q + s * stride_oas_s, mask=mask_q, other=-1e30).to(tl.float32)
        a_max = tl.maximum(a_max, ls)

    a_denom = tl.zeros([BLOCK_Q], dtype=tl.float32)
    for s in tl.static_range(S):
        ls = tl.load(ATTN_LOGITS + pid_bh * stride_oas_bh + q_off * stride_oas_q + s * stride_oas_s, mask=mask_q, other=-1e30).to(tl.float32)
        a_denom += tl.exp(ls - a_max)
    a_inv = 1.0 / a_denom

    # Output accumulator tile: [BLOCK_Q, BLOCK_C]
    acc = tl.zeros([BLOCK_Q, BLOCK_C], dtype=tl.float32)
    for s in tl.static_range(S):
        ls = tl.load(ATTN_LOGITS + pid_bh * stride_oas_bh + q_off * stride_oas_q + s * stride_oas_s, mask=mask_q, other=-1e30).to(tl.float32)
        a_s = tl.exp(ls - a_max) * a_inv
        dx = tl.load(OFFS_X + pid_bh * stride_oas_bh + q_off * stride_oas_q + s * stride_oas_s, mask=mask_q, other=0.0).to(tl.float32)
        dy = tl.load(OFFS_Y + pid_bh * stride_oas_bh + q_off * stride_oas_q + s * stride_oas_s, mask=mask_q, other=0.0).to(tl.float32)
        px_f = qx + dx
        py_f = qy + dy

        # round coordinates to nearest grid point: [BLOCK_Q]
        px_i = tl.minimum(tl.maximum((px_f + 0.5).to(tl.int32), 0), W - 1)
        py_i = tl.minimum(tl.maximum((py_f + 0.5).to(tl.int32), 0), H - 1)
        pix = py_i * W + px_i  # flattened pixel ids: [BLOCK_Q]

        b_idx = pid_bh // Hh
        nn_base = b_idx * stride_nn_bh + pix * stride_nn_p
        # gather precomputed 4-NN token ids from dense table: [BLOCK_Q] each
        idx0 = tl.load(NN4_IDX + nn_base + 0 * stride_nn_k, mask=mask_q, other=0).to(tl.int32)
        idx1 = tl.load(NN4_IDX + nn_base + 1 * stride_nn_k, mask=mask_q, other=0).to(tl.int32)
        idx2 = tl.load(NN4_IDX + nn_base + 2 * stride_nn_k, mask=mask_q, other=0).to(tl.int32)
        idx3 = tl.load(NN4_IDX + nn_base + 3 * stride_nn_k, mask=mask_q, other=0).to(tl.int32)

        # Gather 4 neighbor coordinates from KV positions: [BLOCK_Q] per neighbor.
        kv_base = pid_bh * stride_kv_bh
        x0 = tl.load(KV_POS + kv_base + idx0 * stride_kv_n + 0 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        y0 = tl.load(KV_POS + kv_base + idx0 * stride_kv_n + 1 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        x1 = tl.load(KV_POS + kv_base + idx1 * stride_kv_n + 0 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        y1 = tl.load(KV_POS + kv_base + idx1 * stride_kv_n + 1 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        x2 = tl.load(KV_POS + kv_base + idx2 * stride_kv_n + 0 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        y2 = tl.load(KV_POS + kv_base + idx2 * stride_kv_n + 1 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        x3 = tl.load(KV_POS + kv_base + idx3 * stride_kv_n + 0 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        y3 = tl.load(KV_POS + kv_base + idx3 * stride_kv_n + 1 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)

        # Shepard weights over 4 neighbors: distances -> logits -> normalized weights.
        d0 = tl.sqrt((x0 - px_f) * (x0 - px_f) + (y0 - py_f) * (y0 - py_f) + EPS)
        d1 = tl.sqrt((x1 - px_f) * (x1 - px_f) + (y1 - py_f) * (y1 - py_f) + EPS)
        d2 = tl.sqrt((x2 - px_f) * (x2 - px_f) + (y2 - py_f) * (y2 - py_f) + EPS)
        d3 = tl.sqrt((x3 - px_f) * (x3 - px_f) + (y3 - py_f) * (y3 - py_f) + EPS)

        # learned temperature
        l0 = -tau * d0
        l1 = -tau * d1
        l2 = -tau * d2
        l3 = -tau * d3

        # stable softmax (sub max logit)
        lm = tl.maximum(tl.maximum(l0, l1), tl.maximum(l2, l3))
        e0 = tl.exp(l0 - lm)
        e1 = tl.exp(l1 - lm)
        e2 = tl.exp(l2 - lm)
        e3 = tl.exp(l3 - lm)
        w_inv = 1.0 / (e0 + e1 + e2 + e3)

        # Shepard interpolation weights
        alpha0 = a_s * e0 * w_inv
        alpha1 = a_s * e1 * w_inv
        alpha2 = a_s * e2 * w_inv
        alpha3 = a_s * e3 * w_inv

        if EMIT_EDGES and pid_c == 0:
            # Only store the alpha coefficients. KV indices and Q indices are
            # derived during CSR build: KV via NN4_IDX lookup, Q via edge position
            # arithmetic (q = flat_e // (S * 4)).
            coeff_base = pid_bh * stride_coeff_bh + (q_off * S + s) * 4 * stride_coeff_e
            tl.store(EDGE_COEFF + coeff_base + 0 * stride_coeff_e, alpha0, mask=mask_q)
            tl.store(EDGE_COEFF + coeff_base + 1 * stride_coeff_e, alpha1, mask=mask_q)
            tl.store(EDGE_COEFF + coeff_base + 2 * stride_coeff_e, alpha2, mask=mask_q)
            tl.store(EDGE_COEFF + coeff_base + 3 * stride_coeff_e, alpha3, mask=mask_q)

        # Gather value vectors and accumulate weighted sum into output tile.
        v_base = pid_bh * stride_v_bh
        v_ptrs0 = v_base + idx0[:, None] * stride_v_n + c_off[None, :] * stride_v_c
        v_ptrs1 = v_base + idx1[:, None] * stride_v_n + c_off[None, :] * stride_v_c
        v_ptrs2 = v_base + idx2[:, None] * stride_v_n + c_off[None, :] * stride_v_c
        v_ptrs3 = v_base + idx3[:, None] * stride_v_n + c_off[None, :] * stride_v_c
        qc_mask = mask_q[:, None] & mask_c[None, :]
        v0 = tl.load(V + v_ptrs0, mask=qc_mask, other=0.0).to(tl.float32)
        v1 = tl.load(V + v_ptrs1, mask=qc_mask, other=0.0).to(tl.float32)
        v2 = tl.load(V + v_ptrs2, mask=qc_mask, other=0.0).to(tl.float32)
        v3 = tl.load(V + v_ptrs3, mask=qc_mask, other=0.0).to(tl.float32)

        # accumulate weighted sum into the output tile
        acc += alpha0[:, None] * v0 + alpha1[:, None] * v1 + alpha2[:, None] * v2 + alpha3[:, None] * v3

    # Store final output tile: [BLOCK_Q, BLOCK_C]
    out_ptrs = pid_bh * stride_o_bh + q_off[:, None] * stride_o_q + c_off[None, :] * stride_o_c
    if STORE_FP16:
        tl.store(OUT + out_ptrs, acc.to(tl.float16), mask=mask_q[:, None] & mask_c[None, :])
    else:
        tl.store(OUT + out_ptrs, acc, mask=mask_q[:, None] & mask_c[None, :])


@triton.jit
def _deform_attn_bwd_core_kernel(
    QPOS_X, QPOS_Y, OFFS_X, OFFS_Y, ATTN_LOGITS, TAU, NN4_IDX, KV_POS, V, DOUT,
    D_QPOS_X, D_QPOS_Y, D_OFFS_X, D_OFFS_Y, D_ATTN_LOGITS, D_TAU_PARTIAL,
    Nq, Nk, Hh,
    C: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_qpos_bh, stride_qpos_q,
    stride_oas_bh, stride_oas_q, stride_oas_s,
    stride_nn_bh, stride_nn_p, stride_nn_k,
    stride_kv_bh, stride_kv_n, stride_kv_d,
    stride_v_bh, stride_v_n, stride_v_c,
    stride_do_bh, stride_do_q, stride_do_c,
    stride_dq_bh, stride_dq_q,
    stride_dos_bh, stride_dos_q, stride_dos_s,
    stride_da_bh, stride_da_q, stride_da_s,
    stride_dt_bh, stride_dt_tile,
    BLOCK_Q: tl.constexpr,
    BLOCK_C: tl.constexpr,
    EPS: tl.constexpr,
):
    # Backward for all parameters except dV (which is handled by a separate dV centric kernel to reduce atomics).
    # Two-pass algorithm:
    #   Pass 1: accumulate the softmax coupling term a_dot_raw = sum_s a_s * <dO, weighted_v_s>
    #   Pass 2: use a_dot_raw to compute d_attn, d_offsets, d_query_pos, d_tau
    # the gist is: d_attn uses softmax-jacobian (a_s * (raw_s - a_dot_raw)),
    #        d_offsets and d_query_pos come from chain rule through Shepard weights and distances.

    # one program per (batch*head, query-tile); all C channels in one BLOCK_C tile
    pid_bh = tl.program_id(0)
    pid_qt = tl.program_id(1)
    q_off = pid_qt * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    c_off = tl.arange(0, BLOCK_C)                       # [BLOCK_C]
    mask_q = q_off < Nq
    mask_c = c_off < C
    qc_mask = mask_q[:, None] & mask_c[None, :]

    # query coordinates: [BLOCK_Q], upstream gradient tile: [BLOCK_Q, BLOCK_C]
    qx = tl.load(QPOS_X + pid_bh * stride_qpos_bh + q_off * stride_qpos_q, mask=mask_q, other=0.0).to(tl.float32)
    qy = tl.load(QPOS_Y + pid_bh * stride_qpos_bh + q_off * stride_qpos_q, mask=mask_q, other=0.0).to(tl.float32)
    dout = tl.load(DOUT + pid_bh * stride_do_bh + q_off[:, None] * stride_do_q + c_off[None, :] * stride_do_c, mask=qc_mask, other=0.0).to(tl.float32)
    tau = tl.maximum(tl.load(TAU + pid_bh).to(tl.float32), 0.0) + EPS

    # recompute stable softmax over S sampling points (same as forward)
    a_max = tl.full([BLOCK_Q], -1e30, dtype=tl.float32)
    for s in tl.static_range(S):
        ls = tl.load(ATTN_LOGITS + pid_bh * stride_oas_bh + q_off * stride_oas_q + s * stride_oas_s, mask=mask_q, other=-1e30).to(tl.float32)
        a_max = tl.maximum(a_max, ls)
    a_denom = tl.zeros([BLOCK_Q], dtype=tl.float32)
    for s in tl.static_range(S):
        ls = tl.load(ATTN_LOGITS + pid_bh * stride_oas_bh + q_off * stride_oas_q + s * stride_oas_s, mask=mask_q, other=-1e30).to(tl.float32)
        a_denom += tl.exp(ls - a_max)
    a_inv = 1.0 / a_denom

    # First Pass: accumulate softmax coupling term a_dot_raw = sum_a a_s * <dO, sum_k w_k * v_k> ---
    a_dot_raw = tl.zeros([BLOCK_Q], dtype=tl.float32)
    for s in tl.static_range(S):
        ls = tl.load(ATTN_LOGITS + pid_bh * stride_oas_bh + q_off * stride_oas_q + s * stride_oas_s, mask=mask_q, other=-1e30).to(tl.float32)
        a_s = tl.exp(ls - a_max) * a_inv  # softmax weight for sampling point s: [BLOCK_Q]

        # sampling location = query_pos + offset: [BLOCK_Q]
        dx = tl.load(OFFS_X + pid_bh * stride_oas_bh + q_off * stride_oas_q + s * stride_oas_s, mask=mask_q, other=0.0).to(tl.float32)
        dy = tl.load(OFFS_Y + pid_bh * stride_oas_bh + q_off * stride_oas_q + s * stride_oas_s, mask=mask_q, other=0.0).to(tl.float32)
        px_f = qx + dx
        py_f = qy + dy

        # snap to grid and look up 4-NN from precomputed table
        px_i = tl.minimum(tl.maximum((px_f + 0.5).to(tl.int32), 0), W - 1)
        py_i = tl.minimum(tl.maximum((py_f + 0.5).to(tl.int32), 0), H - 1)
        pix = py_i * W + px_i

        b_idx = pid_bh // Hh
        nn_base = b_idx * stride_nn_bh + pix * stride_nn_p
        idx0 = tl.load(NN4_IDX + nn_base + 0 * stride_nn_k, mask=mask_q, other=0).to(tl.int32)
        idx1 = tl.load(NN4_IDX + nn_base + 1 * stride_nn_k, mask=mask_q, other=0).to(tl.int32)
        idx2 = tl.load(NN4_IDX + nn_base + 2 * stride_nn_k, mask=mask_q, other=0).to(tl.int32)
        idx3 = tl.load(NN4_IDX + nn_base + 3 * stride_nn_k, mask=mask_q, other=0).to(tl.int32)

        # gather real KV coordinates for Shepard distance computation
        kv_base = pid_bh * stride_kv_bh
        x0 = tl.load(KV_POS + kv_base + idx0 * stride_kv_n + 0 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        y0 = tl.load(KV_POS + kv_base + idx0 * stride_kv_n + 1 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        x1 = tl.load(KV_POS + kv_base + idx1 * stride_kv_n + 0 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        y1 = tl.load(KV_POS + kv_base + idx1 * stride_kv_n + 1 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        x2 = tl.load(KV_POS + kv_base + idx2 * stride_kv_n + 0 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        y2 = tl.load(KV_POS + kv_base + idx2 * stride_kv_n + 1 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        x3 = tl.load(KV_POS + kv_base + idx3 * stride_kv_n + 0 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        y3 = tl.load(KV_POS + kv_base + idx3 * stride_kv_n + 1 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)

        # Shepard distances and stable softmax over 4 neighbors (recomputed from forward)
        d0 = tl.sqrt((x0 - px_f) * (x0 - px_f) + (y0 - py_f) * (y0 - py_f) + EPS)
        d1 = tl.sqrt((x1 - px_f) * (x1 - px_f) + (y1 - py_f) * (y1 - py_f) + EPS)
        d2 = tl.sqrt((x2 - px_f) * (x2 - px_f) + (y2 - py_f) * (y2 - py_f) + EPS)
        d3 = tl.sqrt((x3 - px_f) * (x3 - px_f) + (y3 - py_f) * (y3 - py_f) + EPS)

        # Learned temp scaling
        l0 = -tau * d0
        l1 = -tau * d1
        l2 = -tau * d2
        l3 = -tau * d3

        # stable softmax (sub max-logit)
        lm = tl.maximum(tl.maximum(l0, l1), tl.maximum(l2, l3))
        e0 = tl.exp(l0 - lm)
        e1 = tl.exp(l1 - lm)
        e2 = tl.exp(l2 - lm)
        e3 = tl.exp(l3 - lm)
        invw = 1.0 / (e0 + e1 + e2 + e3)
        w0 = e0 * invw  # Shepard interpolation weight for neighbor 0: [BLOCK_Q]
        w1 = e1 * invw
        w2 = e2 * invw
        w3 = e3 * invw

        # gather value vectors: [BLOCK_Q, BLOCK_C]
        v_base = pid_bh * stride_v_bh
        v0 = tl.load(V + v_base + idx0[:, None] * stride_v_n + c_off[None, :] * stride_v_c, mask=qc_mask, other=0.0).to(tl.float32)
        v1 = tl.load(V + v_base + idx1[:, None] * stride_v_n + c_off[None, :] * stride_v_c, mask=qc_mask, other=0.0).to(tl.float32)
        v2 = tl.load(V + v_base + idx2[:, None] * stride_v_n + c_off[None, :] * stride_v_c, mask=qc_mask, other=0.0).to(tl.float32)
        v3 = tl.load(V + v_base + idx3[:, None] * stride_v_n + c_off[None, :] * stride_v_c, mask=qc_mask, other=0.0).to(tl.float32)
        weighted_v = w0[:, None] * v0 + w1[:, None] * v1 + w2[:, None] * v2 + w3[:, None] * v3

        # raw_s = <dO, weighted_v> per query: [BLOCK_Q]
        raw_s = tl.sum(dout * weighted_v, axis=1)
        a_dot_raw += a_s * raw_s

    # Second Pass: compute d_attn, d_offsets, d_query_pos, d_tau using a_dot_raw ---
    dqx = tl.zeros([BLOCK_Q], dtype=tl.float32)  # accumulates d_query_pos_x across S
    dqy = tl.zeros([BLOCK_Q], dtype=tl.float32)
    dtau_local = tl.zeros([BLOCK_Q], dtype=tl.float32)
    for s in tl.static_range(S):
        ls = tl.load(ATTN_LOGITS + pid_bh * stride_oas_bh + q_off * stride_oas_q + s * stride_oas_s, mask=mask_q, other=-1e30).to(tl.float32)
        a_s = tl.exp(ls - a_max) * a_inv
        dx = tl.load(OFFS_X + pid_bh * stride_oas_bh + q_off * stride_oas_q + s * stride_oas_s, mask=mask_q, other=0.0).to(tl.float32)
        dy = tl.load(OFFS_Y + pid_bh * stride_oas_bh + q_off * stride_oas_q + s * stride_oas_s, mask=mask_q, other=0.0).to(tl.float32)
        px_f = qx + dx
        py_f = qy + dy
        px_i = tl.minimum(tl.maximum((px_f + 0.5).to(tl.int32), 0), W - 1)
        py_i = tl.minimum(tl.maximum((py_f + 0.5).to(tl.int32), 0), H - 1)
        pix = py_i * W + px_i

        b_idx = pid_bh // Hh
        nn_base = b_idx * stride_nn_bh + pix * stride_nn_p
        idx0 = tl.load(NN4_IDX + nn_base + 0 * stride_nn_k, mask=mask_q, other=0).to(tl.int32)
        idx1 = tl.load(NN4_IDX + nn_base + 1 * stride_nn_k, mask=mask_q, other=0).to(tl.int32)
        idx2 = tl.load(NN4_IDX + nn_base + 2 * stride_nn_k, mask=mask_q, other=0).to(tl.int32)
        idx3 = tl.load(NN4_IDX + nn_base + 3 * stride_nn_k, mask=mask_q, other=0).to(tl.int32)
        kv_base = pid_bh * stride_kv_bh
        x0 = tl.load(KV_POS + kv_base + idx0 * stride_kv_n + 0 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        y0 = tl.load(KV_POS + kv_base + idx0 * stride_kv_n + 1 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        x1 = tl.load(KV_POS + kv_base + idx1 * stride_kv_n + 0 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        y1 = tl.load(KV_POS + kv_base + idx1 * stride_kv_n + 1 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        x2 = tl.load(KV_POS + kv_base + idx2 * stride_kv_n + 0 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        y2 = tl.load(KV_POS + kv_base + idx2 * stride_kv_n + 1 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        x3 = tl.load(KV_POS + kv_base + idx3 * stride_kv_n + 0 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)
        y3 = tl.load(KV_POS + kv_base + idx3 * stride_kv_n + 1 * stride_kv_d, mask=mask_q, other=0.0).to(tl.float32)

        # relative position vectors: kv neighbor -> sampling point
        ddx0 = x0 - px_f
        ddy0 = y0 - py_f
        ddx1 = x1 - px_f
        ddy1 = y1 - py_f
        ddx2 = x2 - px_f
        ddy2 = y2 - py_f
        ddx3 = x3 - px_f
        ddy3 = y3 - py_f

        # recompute Shepard distances and weights (same as pass 1)
        d0 = tl.sqrt(ddx0 * ddx0 + ddy0 * ddy0 + EPS)
        d1 = tl.sqrt(ddx1 * ddx1 + ddy1 * ddy1 + EPS)
        d2 = tl.sqrt(ddx2 * ddx2 + ddy2 * ddy2 + EPS)
        d3 = tl.sqrt(ddx3 * ddx3 + ddy3 * ddy3 + EPS)

        # learned temperature
        l0 = -tau * d0
        l1 = -tau * d1
        l2 = -tau * d2
        l3 = -tau * d3

        # stable softmax (sub max logit)
        lm = tl.maximum(tl.maximum(l0, l1), tl.maximum(l2, l3))
        e0 = tl.exp(l0 - lm)
        e1 = tl.exp(l1 - lm)
        e2 = tl.exp(l2 - lm)
        e3 = tl.exp(l3 - lm)
        invw = 1.0 / (e0 + e1 + e2 + e3)
        w0 = e0 * invw
        w1 = e1 * invw
        w2 = e2 * invw
        w3 = e3 * invw
        v_base = pid_bh * stride_v_bh
        v0 = tl.load(V + v_base + idx0[:, None] * stride_v_n + c_off[None, :] * stride_v_c, mask=qc_mask, other=0.0).to(tl.float32)
        v1 = tl.load(V + v_base + idx1[:, None] * stride_v_n + c_off[None, :] * stride_v_c, mask=qc_mask, other=0.0).to(tl.float32)
        v2 = tl.load(V + v_base + idx2[:, None] * stride_v_n + c_off[None, :] * stride_v_c, mask=qc_mask, other=0.0).to(tl.float32)
        v3 = tl.load(V + v_base + idx3[:, None] * stride_v_n + c_off[None, :] * stride_v_c, mask=qc_mask, other=0.0).to(tl.float32)
        weighted_v = w0[:, None] * v0 + w1[:, None] * v1 + w2[:, None] * v2 + w3[:, None] * v3
        raw_s = tl.sum(dout * weighted_v, axis=1)

        # d_attn_logits via softmax jacobian: a_s * (raw_s - a_dot_raw): [BLOCK_Q]
        d_attn_s = a_s * (raw_s - a_dot_raw)
        tl.store(D_ATTN_LOGITS + pid_bh * stride_da_bh + q_off * stride_da_q + s * stride_da_s, d_attn_s, mask=mask_q)

        # d_Shepard_weights via chain rule through the 4-neighbor softmax
        # dp_k = <v_k, a_s * dOut> is the per-neighbor dot product: [BLOCK_Q]
        d_wv = a_s[:, None] * dout
        dp0 = tl.sum(v0 * d_wv, axis=1)
        dp1 = tl.sum(v1 * d_wv, axis=1)
        dp2 = tl.sum(v2 * d_wv, axis=1)
        dp3 = tl.sum(v3 * d_wv, axis=1)

        # softmax jacobian for Shepard weights: ds_k = w_k * (dp_k - bar)
        bar = w0 * dp0 + w1 * dp1 + w2 * dp2 + w3 * dp3
        ds0 = w0 * (dp0 - bar)
        ds1 = w1 * (dp1 - bar)
        ds2 = w2 * (dp2 - bar)
        ds3 = w3 * (dp3 - bar)

        # chain rule: d_dist_k = ds_k * (-tau), then d_px/d_py via d_dist/d_pos
        d_dist0 = ds0 * (-tau)
        d_dist1 = ds1 * (-tau)
        d_dist2 = ds2 * (-tau)
        d_dist3 = ds3 * (-tau)
        dpx = d_dist0 * (px_f - x0) / d0 + d_dist1 * (px_f - x1) / d1 + d_dist2 * (px_f - x2) / d2 + d_dist3 * (px_f - x3) / d3
        dpy = d_dist0 * (py_f - y0) / d0 + d_dist1 * (py_f - y1) / d1 + d_dist2 * (py_f - y2) / d2 + d_dist3 * (py_f - y3) / d3

        # d_offset = d_sampling_pos (same gradient since offset is additive)
        tl.store(D_OFFS_X + pid_bh * stride_dos_bh + q_off * stride_dos_q + s * stride_dos_s, dpx, mask=mask_q)
        tl.store(D_OFFS_Y + pid_bh * stride_dos_bh + q_off * stride_dos_q + s * stride_dos_s, dpy, mask=mask_q)

        # d_query_pos accumulates from all S sampling points
        dqx += dpx
        dqy += dpy

        # d_tau: chain through -dist * ds_k
        dtau_local += ds0 * (-d0) + ds1 * (-d1) + ds2 * (-d2) + ds3 * (-d3)

    # store d_query_pos (summed across S): [BLOCK_Q]
    tl.store(D_QPOS_X + pid_bh * stride_dq_bh + q_off * stride_dq_q, dqx, mask=mask_q)
    tl.store(D_QPOS_Y + pid_bh * stride_dq_bh + q_off * stride_dq_q, dqy, mask=mask_q)

    # d_tau: reduce across queries in this tile, store one scalar per (bh, tile)
    dtau_sum = tl.sum(tl.where(mask_q, dtau_local, 0.0))
    tl.store(D_TAU_PARTIAL + pid_bh * stride_dt_bh + pid_qt * stride_dt_tile, dtau_sum)


@triton.jit
def _csr_count_kernel(
    QPOS_X, QPOS_Y, OFFS_X, OFFS_Y, NN4_IDX, COUNTS,
    TOTAL_EDGES, E_PER_BH, Nq, Nk, H, W, S, Hh,
    stride_qpos_bh, stride_qpos_q,
    stride_oas_bh, stride_oas_q, stride_oas_s,
    stride_nn_bh, stride_nn_p, stride_nn_k,
    stride_counts,
    BLOCK_E: tl.constexpr,
):
    """
    Count edges per CSR row (bh, kv).

    Args:
        QPOS_X/QPOS_Y: [BH, Nq], query coordinates.
        OFFS_X/OFFS_Y: [BH, Nq, S], sampling offsets.
        NN4_IDX: [B, H*W, 4], dense top-4 neighbor table shared across heads.
        COUNTS: [BH*Nk], output row counts.
    Returns:
        None, writes counts in-place via atomics.
    """
    # Count how many edges (q->kv contributions) land in each CSR row (one row per (bh, kv) pair)
    pid = tl.program_id(0)
    edge_off = pid * BLOCK_E + tl.arange(0, BLOCK_E)  # [BLOCK_E]
    mask = edge_off < TOTAL_EDGES
    bh = edge_off // E_PER_BH
    e  = edge_off - bh * E_PER_BH

    # Edge decoding: flat edge index -> (bh, q, s, k) via modular arithmetic
    k  = e % 4
    qs = e // 4
    q  = qs // S
    s  = qs % S

    # Grid re-derivation: same snap-to-grid + NN4 lookup as forward kernel to recover kv index
    qx = tl.load(QPOS_X + bh * stride_qpos_bh + q * stride_qpos_q, mask=mask, other=0.0).to(tl.float32)
    qy = tl.load(QPOS_Y + bh * stride_qpos_bh + q * stride_qpos_q, mask=mask, other=0.0).to(tl.float32)
    ox = tl.load(OFFS_X + bh * stride_oas_bh + q * stride_oas_q + s * stride_oas_s, mask=mask, other=0.0).to(tl.float32)
    oy = tl.load(OFFS_Y + bh * stride_oas_bh + q * stride_oas_q + s * stride_oas_s, mask=mask, other=0.0).to(tl.float32)

    # grid snapping (nearest token location)
    px_i = tl.minimum(tl.maximum(((qx + ox) + 0.5).to(tl.int32), 0), W - 1)
    py_i = tl.minimum(tl.maximum(((qy + oy) + 0.5).to(tl.int32), 0), H - 1)
    pix  = py_i * W + px_i
    b_idx = bh // Hh
    kv = tl.load(NN4_IDX + b_idx * stride_nn_bh + pix * stride_nn_p + k * stride_nn_k, mask=mask, other=0).to(tl.int32)
    kv = tl.minimum(tl.maximum(kv, 0), Nk - 1)
    row = bh * Nk + kv
    # Atomic count increment per (bh, kv) row
    tl.atomic_add(COUNTS + row * stride_counts, 1, mask=mask)


@triton.jit
def _csr_scatter_kernel(
    QPOS_X, QPOS_Y, OFFS_X, OFFS_Y, NN4_IDX, EDGE_COEFF,
    WRITE_PTR, Q_FLAT, COEFF_FLAT,
    TOTAL_EDGES, E_PER_BH, Nq, Nk, H, W, S, Hh,
    stride_qpos_bh, stride_qpos_q,
    stride_oas_bh, stride_oas_q, stride_oas_s,
    stride_nn_bh, stride_nn_p, stride_nn_k,
    stride_coeff_bh, stride_coeff_e,
    stride_wp, stride_qf, stride_cf,
    BLOCK_E: tl.constexpr,
):
    """
    Scatter edge (q, coeff) entries into CSR storage.

    Args:
        QPOS_X/QPOS_Y: [BH, Nq], query coordinates.
        OFFS_X/OFFS_Y: [BH, Nq, S], sampling offsets.
        NN4_IDX: [B, H*W, 4], dense top-4 neighbor table shared across heads.
        EDGE_COEFF: [BH, Nq*S*4], edge coefficients.
        WRITE_PTR/Q_FLAT/COEFF_FLAT: CSR write buffers.
    Returns:
        None, writes CSR arrays in-place.
    """
    # Scatter (query_id, coefficient) pairs into CSR storage using atomic write pointers
    pid = tl.program_id(0)
    edge_off = pid * BLOCK_E + tl.arange(0, BLOCK_E)  # [BLOCK_E]
    mask = edge_off < TOTAL_EDGES
    bh = edge_off // E_PER_BH
    e  = edge_off - bh * E_PER_BH

    # Same edge decoding and grid re-derivation as count kernel
    k  = e % 4
    qs = e // 4
    q  = qs // S
    s  = qs % S
    qx = tl.load(QPOS_X + bh * stride_qpos_bh + q * stride_qpos_q, mask=mask, other=0.0).to(tl.float32)
    qy = tl.load(QPOS_Y + bh * stride_qpos_bh + q * stride_qpos_q, mask=mask, other=0.0).to(tl.float32)
    ox = tl.load(OFFS_X + bh * stride_oas_bh + q * stride_oas_q + s * stride_oas_s, mask=mask, other=0.0).to(tl.float32)
    oy = tl.load(OFFS_Y + bh * stride_oas_bh + q * stride_oas_q + s * stride_oas_s, mask=mask, other=0.0).to(tl.float32)

    # grid snapping (nearest token location)
    px_i = tl.minimum(tl.maximum(((qx + ox) + 0.5).to(tl.int32), 0), W - 1)
    py_i = tl.minimum(tl.maximum(((qy + oy) + 0.5).to(tl.int32), 0), H - 1)
    pix  = py_i * W + px_i
    b_idx = bh // Hh
    kv = tl.load(NN4_IDX + b_idx * stride_nn_bh + pix * stride_nn_p + k * stride_nn_k, mask=mask, other=0).to(tl.int32)
    kv = tl.minimum(tl.maximum(kv, 0), Nk - 1)

    # Load alpha coefficient from forward pass edge_coeff buffer
    coeff = tl.load(EDGE_COEFF + bh * stride_coeff_bh + e * stride_coeff_e, mask=mask, other=0.0).to(tl.float32)
    row = bh * Nk + kv

    # Atomic increment of write_ptr to claim slot, then store q and coeff at that slot
    pos = tl.atomic_add(WRITE_PTR + row * stride_wp, 1, mask=mask).to(tl.int32)
    tl.store(Q_FLAT  + pos * stride_qf, q.to(tl.int32), mask=mask)
    tl.store(COEFF_FLAT + pos * stride_cf, coeff, mask=mask)


def _build_kv_csr_refs_from_edges_triton(
    edge_coeff, qpos_x, qpos_y, offs_x, offs_y, nn4_idx, Nk, Nq, H, W,
    BLOCK_E: int = 256, num_warps: int = 4, num_stages: int = 2,
):
    num_stages = clamp_num_stages(num_stages)
    """
    Build KV-centric CSR references from edge coefficients.

    Args:
        edge_coeff: [BH, E], flattened edge coefficients where E=Nq*S*4.
        qpos_x/qpos_y: [BH, Nq], query coordinates.
        offs_x/offs_y: [BH, Nq, S], sampling offsets.
        nn4_idx: [B, H*W, 4], dense top-4 neighbor table shared across heads.
        Nk: int, number of KV tokens.
        Nq: int, number of query tokens.
        H: int, grid height.
        W: int, grid width.
    Returns:
        row_ptr: [BH*Nk+1], CSR row pointer.
        q_flat: [BH*E], flattened query ids.
        coeff_flat: [BH*E], flattened coefficients.
    """
    BH, E = edge_coeff.shape
    Bnn = nn4_idx.shape[0]
    if Bnn <= 0 or (BH % Bnn) != 0:
        raise ValueError(f"NN4 batch dimension must divide BH (got BH={BH}, B={Bnn}).")
    Hh = BH // Bnn
    S = offs_x.shape[2]
    device = edge_coeff.device
    nrows = BH * Nk       # one CSR row per (batch*head, kv_token) pair
    total_edges = BH * E   # E = Nq * S * 4 edges per batch-head lane

    # step 1: count edges per CSR row via atomics
    counts = torch.zeros((nrows,), device=device, dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(total_edges, meta["BLOCK_E"]),)
    with record_function("deform_attn.csr.count"):
        _csr_count_kernel[grid](
            qpos_x, qpos_y, offs_x, offs_y, nn4_idx, counts,
            total_edges, E, Nq, Nk, H, W, S, Hh,
            qpos_x.stride(0), qpos_x.stride(1),
            offs_x.stride(0), offs_x.stride(1), offs_x.stride(2),
            nn4_idx.stride(0), nn4_idx.stride(1), nn4_idx.stride(2),
            counts.stride(0),
            BLOCK_E=BLOCK_E, num_warps=num_warps, num_stages=num_stages,
        )

    # step 2: exclusive prefix sum -> CSR row_ptr
    row_ptr = torch.empty((nrows + 1,), device=device, dtype=torch.int32)
    row_ptr[0] = 0
    row_ptr[1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)

    # step 3: scatter (q_id, coeff) pairs into CSR arrays
    write_ptr = row_ptr[:-1].clone()  # mutable copy for atomic slot claiming
    q_flat = torch.empty((total_edges,), device=device, dtype=torch.int32)
    coeff_flat = torch.empty((total_edges,), device=device, dtype=torch.float32)
    with record_function("deform_attn.csr.scatter"):
        _csr_scatter_kernel[grid](
            qpos_x, qpos_y, offs_x, offs_y, nn4_idx, edge_coeff,
            write_ptr, q_flat, coeff_flat,
            total_edges, E, Nq, Nk, H, W, S, Hh,
            qpos_x.stride(0), qpos_x.stride(1),
            offs_x.stride(0), offs_x.stride(1), offs_x.stride(2),
            nn4_idx.stride(0), nn4_idx.stride(1), nn4_idx.stride(2),
            edge_coeff.stride(0), edge_coeff.stride(1),
            write_ptr.stride(0), q_flat.stride(0), coeff_flat.stride(0),
            BLOCK_E=BLOCK_E, num_warps=num_warps, num_stages=num_stages,
        )
    return row_ptr, q_flat, coeff_flat


@triton.jit
def _deform_attn_dv_gather_kernel(
    ROW_IDS, ROW_PTR, Q_FLAT, COEFF_FLAT, DOUT, D_V,
    NROWS, Nk, C, ROW_BUCKET,
    stride_rid, stride_rp, stride_qf, stride_cf,
    stride_do_bh, stride_do_q, stride_do_c,
    stride_dv_bh, stride_dv_kv, stride_dv_c,
    BLOCK_C: tl.constexpr,
):
    # One kernel reduces one KV row (from CSR) to compute dV for that (bh, kv) token.
    pid_task = tl.program_id(0)
    pid_c = tl.program_id(1)
    if pid_task >= NROWS:
        return
    pid_row = tl.load(ROW_IDS + pid_task * stride_rid).to(tl.int32)

    # Channel tiling: pid_c tiles over the C dimension with BLOCK_C.
    c_off = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c_off < C
    pid_bh = pid_row // Nk
    pid_kv = pid_row - pid_bh * Nk
    do_base = DOUT + pid_bh * stride_do_bh
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    start = tl.load(ROW_PTR + pid_row * stride_rp).to(tl.int32)
    end = tl.load(ROW_PTR + (pid_row + 1) * stride_rp).to(tl.int32)

    # Iterate over CSR edge span: for each edge, load (q_id, coeff), gather dOut[q_id], accumulate coeff * dOut into dV channel tile [BLOCK_C].
    r = start
    while r < end:
        q = tl.load(Q_FLAT + r * stride_qf).to(tl.int32)
        coeff = tl.load(COEFF_FLAT + r * stride_cf).to(tl.float32)
        dout_vec = tl.load(do_base + q * stride_do_q + c_off * stride_do_c, mask=mask_c, other=0.0).to(tl.float32)
        acc += coeff * dout_vec
        r += 1
    tl.store(D_V + pid_bh * stride_dv_bh + pid_kv * stride_dv_kv + c_off * stride_dv_c, acc, mask=mask_c)


def _compute_dv_gather_triton_from_csr(
    dout, Nk, row_ptr, q_flat, coeff_flat,
    BLOCK_C: int | None = None, num_warps: int = 1, num_stages: int = 2,
):
    num_stages = clamp_num_stages(num_stages)
    BH = dout.shape[0]
    C = dout.shape[2]
    d_v = torch.empty((BH, Nk, C), device=dout.device, dtype=torch.float32)
    row_ids = torch.arange(row_ptr.numel() - 1, device=row_ptr.device, dtype=torch.int32)
    nrows = row_ids.shape[0]
    if BLOCK_C is None:
        block_c = 1  # next power-of-two >= C
        while block_c < C:
            block_c <<= 1
        BLOCK_C = block_c
    elif BLOCK_C < C:
        raise ValueError(f"BLOCK_C must be at least C: {BLOCK_C} < {C}")
    grid = (nrows, triton.cdiv(C, BLOCK_C))
    # We intentionally skip row-length bucketing because it regressed runtime.
    with record_function("deform_attn.gather_dv_triton"):
        _deform_attn_dv_gather_kernel[grid](
            row_ids, row_ptr, q_flat, coeff_flat, dout, d_v,
            nrows, Nk, C, 0,
            row_ids.stride(0), row_ptr.stride(0), q_flat.stride(0), coeff_flat.stride(0),
            dout.stride(0), dout.stride(1), dout.stride(2),
            d_v.stride(0), d_v.stride(1), d_v.stride(2),
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return d_v


@triton.jit
def _deform_attn_dv_atomic_kernel(
    QPOS_X, QPOS_Y, OFFS_X, OFFS_Y, ATTN_LOGITS, TAU, NN4_IDX, KV_POS, DOUT, D_V,
    Nq, Nk, C, S: tl.constexpr, H, W, Hh,
    stride_qpos_bh, stride_qpos_q,
    stride_oas_bh, stride_oas_q, stride_oas_s,
    stride_nn_bh, stride_nn_p, stride_nn_k,
    stride_kv_bh, stride_kv_n, stride_kv_d,
    stride_do_bh, stride_do_q, stride_do_c,
    stride_dv_bh, stride_dv_n, stride_dv_c,
    BLOCK_Q: tl.constexpr,
    BLOCK_C: tl.constexpr,
    MERGE_DUP: tl.constexpr,
    EPS: tl.constexpr,
):
    # Atomic dV fallback: recomputes alphas from scratch (no CSR cache needed).
    # Recomputes the same softmax-over-S and Shepard-over-4 as the forward kernel.
    pid_bh = tl.program_id(0)
    pid_qt = tl.program_id(1)
    pid_c = tl.program_id(2)

    q_off = pid_qt * BLOCK_Q + tl.arange(0, BLOCK_Q)
    c_off = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_q = q_off < Nq
    mask_c = c_off < C
    qc_mask = mask_q[:, None] & mask_c[None, :]

    qx = tl.load(QPOS_X + pid_bh * stride_qpos_bh + q_off * stride_qpos_q, mask=mask_q, other=0.0).to(tl.float32)
    qy = tl.load(QPOS_Y + pid_bh * stride_qpos_bh + q_off * stride_qpos_q, mask=mask_q, other=0.0).to(tl.float32)
    dout = tl.load(
        DOUT + pid_bh * stride_do_bh + q_off[:, None] * stride_do_q + c_off[None, :] * stride_do_c,
        mask=qc_mask,
        other=0.0,
    ).to(tl.float32)

    tau_raw = tl.load(TAU + pid_bh).to(tl.float32)
    tau = tl.maximum(tau_raw, 0.0) + EPS

    a_max = tl.full([BLOCK_Q], -1e30, dtype=tl.float32)
    for s in tl.static_range(0, S):
        ls = tl.load(ATTN_LOGITS + pid_bh * stride_oas_bh + q_off * stride_oas_q + s * stride_oas_s, mask=mask_q, other=-1e30).to(tl.float32)
        a_max = tl.maximum(a_max, ls)
    a_denom = tl.zeros([BLOCK_Q], dtype=tl.float32)
    for s in tl.static_range(0, S):
        ls = tl.load(ATTN_LOGITS + pid_bh * stride_oas_bh + q_off * stride_oas_q + s * stride_oas_s, mask=mask_q, other=-1e30).to(tl.float32)
        a_denom += tl.exp(ls - a_max)
    a_inv = 1.0 / a_denom

    # For each (q, s, 4-neighbor): alpha = attn_weight * shepard_weight, then atomically add alpha * dOut into dV[kv_idx].
    for s in tl.static_range(0, S):
        ls = tl.load(ATTN_LOGITS + pid_bh * stride_oas_bh + q_off * stride_oas_q + s * stride_oas_s, mask=mask_q, other=-1e30).to(tl.float32)
        a_s = tl.exp(ls - a_max) * a_inv

        dx = tl.load(OFFS_X + pid_bh * stride_oas_bh + q_off * stride_oas_q + s * stride_oas_s, mask=mask_q, other=0.0).to(tl.float32)
        dy = tl.load(OFFS_Y + pid_bh * stride_oas_bh + q_off * stride_oas_q + s * stride_oas_s, mask=mask_q, other=0.0).to(tl.float32)
        px_f = qx + dx
        py_f = qy + dy
        px_i = tl.minimum(tl.maximum((px_f + 0.5).to(tl.int32), 0), W - 1)
        py_i = tl.minimum(tl.maximum((py_f + 0.5).to(tl.int32), 0), H - 1)
        pix = py_i * W + px_i

        b_idx = pid_bh // Hh
        nn_base = b_idx * stride_nn_bh + pix * stride_nn_p
        idx0 = tl.load(NN4_IDX + nn_base + 0 * stride_nn_k, mask=mask_q, other=0).to(tl.int32)
        idx1 = tl.load(NN4_IDX + nn_base + 1 * stride_nn_k, mask=mask_q, other=0).to(tl.int32)
        idx2 = tl.load(NN4_IDX + nn_base + 2 * stride_nn_k, mask=mask_q, other=0).to(tl.int32)
        idx3 = tl.load(NN4_IDX + nn_base + 3 * stride_nn_k, mask=mask_q, other=0).to(tl.int32)

        mask_idx0 = mask_q & (idx0 >= 0) & (idx0 < Nk)
        mask_idx1 = mask_q & (idx1 >= 0) & (idx1 < Nk)
        mask_idx2 = mask_q & (idx2 >= 0) & (idx2 < Nk)
        mask_idx3 = mask_q & (idx3 >= 0) & (idx3 < Nk)

        kv_base = pid_bh * stride_kv_bh
        x0 = tl.load(KV_POS + kv_base + idx0 * stride_kv_n + 0 * stride_kv_d, mask=mask_idx0, other=0.0).to(tl.float32)
        y0 = tl.load(KV_POS + kv_base + idx0 * stride_kv_n + 1 * stride_kv_d, mask=mask_idx0, other=0.0).to(tl.float32)
        x1 = tl.load(KV_POS + kv_base + idx1 * stride_kv_n + 0 * stride_kv_d, mask=mask_idx1, other=0.0).to(tl.float32)
        y1 = tl.load(KV_POS + kv_base + idx1 * stride_kv_n + 1 * stride_kv_d, mask=mask_idx1, other=0.0).to(tl.float32)
        x2 = tl.load(KV_POS + kv_base + idx2 * stride_kv_n + 0 * stride_kv_d, mask=mask_idx2, other=0.0).to(tl.float32)
        y2 = tl.load(KV_POS + kv_base + idx2 * stride_kv_n + 1 * stride_kv_d, mask=mask_idx2, other=0.0).to(tl.float32)
        x3 = tl.load(KV_POS + kv_base + idx3 * stride_kv_n + 0 * stride_kv_d, mask=mask_idx3, other=0.0).to(tl.float32)
        y3 = tl.load(KV_POS + kv_base + idx3 * stride_kv_n + 1 * stride_kv_d, mask=mask_idx3, other=0.0).to(tl.float32)

        d0 = tl.sqrt((x0 - px_f) * (x0 - px_f) + (y0 - py_f) * (y0 - py_f) + EPS)
        d1 = tl.sqrt((x1 - px_f) * (x1 - px_f) + (y1 - py_f) * (y1 - py_f) + EPS)
        d2 = tl.sqrt((x2 - px_f) * (x2 - px_f) + (y2 - py_f) * (y2 - py_f) + EPS)
        d3 = tl.sqrt((x3 - px_f) * (x3 - px_f) + (y3 - py_f) * (y3 - py_f) + EPS)
        l0 = -tau * d0
        l1 = -tau * d1
        l2 = -tau * d2
        l3 = -tau * d3
        lm = tl.maximum(tl.maximum(l0, l1), tl.maximum(l2, l3))
        e0 = tl.exp(l0 - lm)
        e1 = tl.exp(l1 - lm)
        e2 = tl.exp(l2 - lm)
        e3 = tl.exp(l3 - lm)
        w_inv = 1.0 / (e0 + e1 + e2 + e3)
        w0 = e0 * w_inv
        w1 = e1 * w_inv
        w2 = e2 * w_inv
        w3 = e3 * w_inv

        coeff0 = a_s * w0
        coeff1 = a_s * w1
        coeff2 = a_s * w2
        coeff3 = a_s * w3
        # MERGE_DUP: merge duplicate kv indices within the same 4-NN set to reduce atomic contention.
        if MERGE_DUP:
            coeff0 = coeff0 + tl.where(idx1 == idx0, coeff1, 0.0) + tl.where(idx2 == idx0, coeff2, 0.0) + tl.where(idx3 == idx0, coeff3, 0.0)
            coeff1 = tl.where(idx1 != idx0, coeff1 + tl.where(idx2 == idx1, coeff2, 0.0) + tl.where(idx3 == idx1, coeff3, 0.0), 0.0)
            coeff2 = tl.where((idx2 != idx0) & (idx2 != idx1), coeff2 + tl.where(idx3 == idx2, coeff3, 0.0), 0.0)
            coeff3 = tl.where((idx3 != idx0) & (idx3 != idx1) & (idx3 != idx2), coeff3, 0.0)
            mask_idx1 = mask_idx1 & (idx1 != idx0)
            mask_idx2 = mask_idx2 & (idx2 != idx0) & (idx2 != idx1)
            mask_idx3 = mask_idx3 & (idx3 != idx0) & (idx3 != idx1) & (idx3 != idx2)

        tl.atomic_add(
            D_V + pid_bh * stride_dv_bh + idx0[:, None] * stride_dv_n + c_off[None, :] * stride_dv_c,
            coeff0[:, None] * dout,
            mask=mask_idx0[:, None] & mask_c[None, :],
        )
        tl.atomic_add(
            D_V + pid_bh * stride_dv_bh + idx1[:, None] * stride_dv_n + c_off[None, :] * stride_dv_c,
            coeff1[:, None] * dout,
            mask=mask_idx1[:, None] & mask_c[None, :],
        )
        tl.atomic_add(
            D_V + pid_bh * stride_dv_bh + idx2[:, None] * stride_dv_n + c_off[None, :] * stride_dv_c,
            coeff2[:, None] * dout,
            mask=mask_idx2[:, None] & mask_c[None, :],
        )
        tl.atomic_add(
            D_V + pid_bh * stride_dv_bh + idx3[:, None] * stride_dv_n + c_off[None, :] * stride_dv_c,
            coeff3[:, None] * dout,
            mask=mask_idx3[:, None] & mask_c[None, :],
        )


def _compute_dv_atomic_triton(
    dout, qpos_x, qpos_y, offs_x, offs_y, attn_logits, tau, nn4_idx, kv_pos, Nk, Hh, H=64, W=64, BLOCK_Q=32, BLOCK_C=None, num_warps=4, num_stages=4
):
    BH, _, _ = offs_x.shape
    C = dout.shape[2]

    # automatically determine BLOCK_C if not provided
    if BLOCK_C is None:
        block_c = 1
        while block_c < C:
            block_c <<= 1
        BLOCK_C = block_c
    elif BLOCK_C < C:
        raise ValueError(f"BLOCK_C must be at least C: {BLOCK_C} < {C}")

    d_v = torch.zeros((BH, Nk, C), device=dout.device, dtype=torch.float32)
    grid = lambda meta: (BH, triton.cdiv(offs_x.shape[1], meta["BLOCK_Q"]), triton.cdiv(C, meta["BLOCK_C"]))
    _deform_attn_dv_atomic_kernel[grid](
        qpos_x, qpos_y, offs_x, offs_y, attn_logits, tau, nn4_idx, kv_pos, dout, d_v,
        offs_x.shape[1], Nk, C, offs_x.shape[2], H, W, Hh,
        qpos_x.stride(0), qpos_x.stride(1),
        offs_x.stride(0), offs_x.stride(1), offs_x.stride(2),
        nn4_idx.stride(0), nn4_idx.stride(1), nn4_idx.stride(2),
        kv_pos.stride(0), kv_pos.stride(1), kv_pos.stride(2),
        dout.stride(0), dout.stride(1), dout.stride(2),
        d_v.stride(0), d_v.stride(1), d_v.stride(2),
        MERGE_DUP=True, EPS=1e-6,
        BLOCK_Q=BLOCK_Q, BLOCK_C=BLOCK_C,
        num_warps=num_warps, num_stages=num_stages,
    )
    return d_v


def deform_attn_forward(
    qpos_x, qpos_y, offs_x, offs_y, attn_logits, tau, nn4_idx, kv_pos, v,
    # Static launch heuristics. These replaced @triton.autotune: autotuning picked
    # per-rank under multi-process training and raced on the shared disk cache.
    # The values were hand-tuned on an A100 for the configs in configs/; the
    # benchmark script that produced them was not preserved, so re-tune with
    # Keep these conservative across supported Triton backends.
    H=64, W=64, BLOCK_Q=32, BLOCK_C=None, num_warps=4, num_stages=4,
    return_edge_cache: bool = False,
):
    """
    Forward deformable attention on flattened batch-head lanes.

    Args:
        qpos_x/qpos_y: [BH, Nq], query coordinates.
        offs_x/offs_y: [BH, Nq, S], sampling offsets.
        attn_logits: [BH, Nq, S], attention logits over S points.
        tau: [BH], Shepard distance scale.
        nn4_idx: [B, H*W, 4], dense KNN lookup table shared across heads.
        kv_pos: [BH, Nk, 2], KV coordinates.
        v: [BH, Nk, C], value vectors.
        H: int, lookup grid height.
        W: int, lookup grid width.
        BLOCK_Q: int, query tile size.
        BLOCK_C: int, channel tile size.
        num_warps: int, Triton launch warps.
        num_stages: int, Triton pipeline stages.
        return_edge_cache: bool, if True also return edge coefficients used
            by CSR dV path in backward.
    Returns:
        If return_edge_cache is False:
            out: [BH, Nq, C].
        If return_edge_cache is True:
            (out, edge_coeff) where edge_coeff is [BH, Nq*S*4].
    """
    num_stages = clamp_num_stages(num_stages)
    BH, Nq, S = offs_x.shape
    Nk = v.shape[1]
    C = v.shape[2]
    if BLOCK_Q <= 0:
        raise ValueError(f"BLOCK_Q must be > 0, got {BLOCK_Q}.")
    if Nq < 0:
        raise ValueError(f"Nq must be >= 0, got {Nq}.")
    if Nq == 0:
        out_dtype = torch.float16 if (v.dtype == torch.float16) else torch.float32
        out_empty = torch.empty((BH, 0, C), dtype=out_dtype, device=v.device)
        if return_edge_cache:
            edge_empty = torch.empty((BH, 0), dtype=torch.float32, device=v.device)
            return out_empty, edge_empty
        return out_empty
    Bnn = nn4_idx.shape[0]
    if Bnn <= 0 or (BH % Bnn) != 0:
        raise ValueError(f"NN4 batch dimension must divide BH (got BH={BH}, B={Bnn}).")
    Hh = BH // Bnn

    # must be valid
    if BLOCK_C is None:
        block_c = 1
        while block_c < C:
            block_c <<= 1
        BLOCK_C = block_c
    elif BLOCK_C < C:
        raise ValueError(f"BLOCK_C must be at least C: {BLOCK_C} < {C}")

    store_fp16 = (v.dtype == torch.float16)
    out_dtype = torch.float16 if store_fp16 else torch.float32
    out = torch.empty(BH, Nq, C, dtype=out_dtype, device=v.device)
    if return_edge_cache:
        edge_coeff = torch.empty((BH, Nq * S * 4), dtype=torch.float32, device=v.device)
    else:
        # Placeholder when edge emission is disabled.
        edge_coeff = torch.empty((1, 1), dtype=torch.float32, device=v.device)
    q_tiles = triton.cdiv(Nq, BLOCK_Q)
    # Safety check: with cdiv tiling, all query indices [0, Nq) are covered
    # and masked in-kernel for tail elements when Nq is not multiple of BLOCK_Q.
    if q_tiles * BLOCK_Q < Nq:
        raise RuntimeError("Forward launch would not cover all queries.")
    grid = (BH, q_tiles, triton.cdiv(C, BLOCK_C))
    with record_function("deform_attn.forward_core"):
        _deform_attn_fwd_kernel[grid](
            qpos_x, qpos_y, offs_x, offs_y, attn_logits, tau, nn4_idx, kv_pos, v, out,
            edge_coeff, Nq, Nk, Hh,
            C=C, S=S, H=H, W=W,
            stride_qpos_bh=qpos_x.stride(0), stride_qpos_q=qpos_x.stride(1),
            stride_oas_bh=offs_x.stride(0), stride_oas_q=offs_x.stride(1), stride_oas_s=offs_x.stride(2),
            stride_nn_bh=nn4_idx.stride(0), stride_nn_p=nn4_idx.stride(1), stride_nn_k=nn4_idx.stride(2),
            stride_kv_bh=kv_pos.stride(0), stride_kv_n=kv_pos.stride(1), stride_kv_d=kv_pos.stride(2),
            stride_v_bh=v.stride(0), stride_v_n=v.stride(1), stride_v_c=v.stride(2),
            stride_o_bh=out.stride(0), stride_o_q=out.stride(1), stride_o_c=out.stride(2),
            stride_coeff_bh=edge_coeff.stride(0), stride_coeff_e=edge_coeff.stride(1),
            EMIT_EDGES=return_edge_cache, EPS=1e-6, STORE_FP16=store_fp16,
            BLOCK_Q=BLOCK_Q, BLOCK_C=BLOCK_C,
            num_warps=num_warps, num_stages=num_stages,
        )
    if return_edge_cache:
        return out, edge_coeff
    return out


def deform_attn_backward(
    dout, qpos_x, qpos_y, offs_x, offs_y, attn_logits, tau, nn4_idx, kv_pos, v,
    H=64, W=64, BLOCK_Q=16, BLOCK_C=None, num_warps=4, num_stages=2,
    CSR_BLOCK_E=256, CSR_num_warps=2, CSR_num_stages=2,
    fwd_edge_cache: torch.Tensor | None = None,
    dv_backend: str = "csr_knn_cached",
):
    """
    Backward deformable attention on flattened batch-head lanes.

    Args:
        dout: [BH, Nq, C], upstream gradient of output.
        qpos_x/qpos_y: [BH, Nq], query coordinates.
        offs_x/offs_y: [BH, Nq, S], sampling offsets.
        attn_logits: [BH, Nq, S], attention logits.
        tau: [BH], Shepard distance scale.
        nn4_idx: [B, H*W, 4], dense KNN table shared across heads.
        kv_pos: [BH, Nk, 2], KV coordinates.
        v: [BH, Nk, C], value vectors.
        H: int, lookup grid height.
        W: int, lookup grid width.
        BLOCK_Q: int, query tile size.
        BLOCK_C: Optional[int], channel tile size for backward core kernel.
            If None, uses the next power-of-two >= C.
        num_warps: int, Triton launch warps for backward core kernel.
        num_stages: int, Triton pipeline stages for backward core kernel.
        CSR_BLOCK_E: int, edge tile size for CSR count/scatter build.
        CSR_num_warps: int, Triton launch warps for CSR build kernels.
        CSR_num_stages: int, Triton pipeline stages for CSR build kernels.
        fwd_edge_cache: optional [BH, Nq*S*4] edge coefficients emitted by
            forward pass when EMIT_EDGES=True.
        dv_backend: str, one of {"csr_knn_cached", "kv_atomic"}. "csr_cached"
            is accepted as a legacy alias of "csr_knn_cached"; shipped configs
            still use it.
    Returns:
        Tuple of gradients:
            d_qpos_x, d_qpos_y, d_offs_x, d_offs_y, d_attn_logits, d_tau, d_v.
    """
    num_stages = clamp_num_stages(num_stages)
    CSR_num_stages = clamp_num_stages(CSR_num_stages)
    BH, Nq, S = offs_x.shape
    Nk = v.shape[1]
    C = v.shape[2]
    if BLOCK_Q <= 0:
        raise ValueError(f"BLOCK_Q must be > 0, got {BLOCK_Q}.")
    if Nq < 0:
        raise ValueError(f"Nq must be >= 0, got {Nq}.")
    if Nq == 0:
        d_qpos_x = torch.empty((BH, 0), device=v.device, dtype=torch.float32)
        d_qpos_y = torch.empty((BH, 0), device=v.device, dtype=torch.float32)
        d_offs_x = torch.empty((BH, 0, S), device=v.device, dtype=torch.float32)
        d_offs_y = torch.empty((BH, 0, S), device=v.device, dtype=torch.float32)
        d_attn_logits = torch.empty((BH, 0, S), device=v.device, dtype=torch.float32)
        d_tau = torch.zeros((BH,), device=v.device, dtype=torch.float32)
        d_v = torch.zeros((BH, Nk, C), device=v.device, dtype=torch.float32)
        return d_qpos_x, d_qpos_y, d_offs_x, d_offs_y, d_attn_logits, d_tau, d_v
    Bnn = nn4_idx.shape[0]
    if Bnn <= 0 or (BH % Bnn) != 0:
        raise ValueError(f"NN4 batch dimension must divide BH (got BH={BH}, B={Bnn}).")
    Hh = BH // Bnn
    d_qpos_x = torch.empty((BH, Nq), device=v.device, dtype=torch.float32)
    d_qpos_y = torch.empty((BH, Nq), device=v.device, dtype=torch.float32)
    d_offs_x = torch.empty((BH, Nq, S), device=v.device, dtype=torch.float32)
    d_offs_y = torch.empty((BH, Nq, S), device=v.device, dtype=torch.float32)
    d_attn_logits = torch.empty((BH, Nq, S), device=v.device, dtype=torch.float32)
    d_tau_partial = torch.empty((BH, triton.cdiv(Nq, BLOCK_Q)), device=v.device, dtype=torch.float32)
    if BLOCK_C is None:
        block_c = 1
        while block_c < C:
            block_c <<= 1
    elif BLOCK_C < C:
        raise ValueError(f"BLOCK_C must be at least C: {BLOCK_C} < {C}")
    else:
        block_c = int(BLOCK_C)
    q_tiles = triton.cdiv(Nq, BLOCK_Q)
    if q_tiles * BLOCK_Q < Nq:
        raise RuntimeError("Backward launch would not cover all queries.")
    grid = lambda meta: (BH, q_tiles)
    with record_function("deform_attn.backward_core"):
        _deform_attn_bwd_core_kernel[grid](
            qpos_x, qpos_y, offs_x, offs_y, attn_logits, tau, nn4_idx, kv_pos, v, dout,
            d_qpos_x, d_qpos_y, d_offs_x, d_offs_y, d_attn_logits, d_tau_partial,
            Nq, Nk, Hh, C=C, S=S, H=H, W=W,
            stride_qpos_bh=qpos_x.stride(0), stride_qpos_q=qpos_x.stride(1),
            stride_oas_bh=offs_x.stride(0), stride_oas_q=offs_x.stride(1), stride_oas_s=offs_x.stride(2),
            stride_nn_bh=nn4_idx.stride(0), stride_nn_p=nn4_idx.stride(1), stride_nn_k=nn4_idx.stride(2),
            stride_kv_bh=kv_pos.stride(0), stride_kv_n=kv_pos.stride(1), stride_kv_d=kv_pos.stride(2),
            stride_v_bh=v.stride(0), stride_v_n=v.stride(1), stride_v_c=v.stride(2),
            stride_do_bh=dout.stride(0), stride_do_q=dout.stride(1), stride_do_c=dout.stride(2),
            stride_dq_bh=d_qpos_x.stride(0), stride_dq_q=d_qpos_x.stride(1),
            stride_dos_bh=d_offs_x.stride(0), stride_dos_q=d_offs_x.stride(1), stride_dos_s=d_offs_x.stride(2),
            stride_da_bh=d_attn_logits.stride(0), stride_da_q=d_attn_logits.stride(1), stride_da_s=d_attn_logits.stride(2),
            stride_dt_bh=d_tau_partial.stride(0), stride_dt_tile=d_tau_partial.stride(1),
            BLOCK_Q=BLOCK_Q, BLOCK_C=block_c, EPS=1e-6,
            num_warps=num_warps, num_stages=num_stages,
        )
    if dv_backend == "kv_atomic":
        with record_function("deform_attn.dv.kv_atomic"):
            d_v = _compute_dv_atomic_triton(
                dout, qpos_x, qpos_y, offs_x, offs_y, attn_logits, tau, nn4_idx, kv_pos, Nk=Nk, Hh=Hh, H=H, W=W, BLOCK_Q=BLOCK_Q, BLOCK_C=block_c, num_warps=num_warps, num_stages=num_stages
            )
    elif dv_backend in {"csr_cached", "csr_knn_cached"}:
        with record_function("deform_attn.gather_dv_triton.csr_build"):
            if fwd_edge_cache is not None:
                edge_coeff = fwd_edge_cache
            else:
                print("WARNING! No forward edge cache provided, building it now! See deform_attn.deform_attn_backward to remove this warning and you know what you are doing.")
                # Fallback for direct backward-only callers (e.g. benches) that
                # do not provide forward edge cache.
                edge_coeff = torch.empty((qpos_x.shape[0], qpos_x.shape[1] * offs_x.shape[2] * 4), device=v.device, dtype=torch.float32)
                out_dummy = torch.empty((qpos_x.shape[0], qpos_x.shape[1], v.shape[2]), device=v.device, dtype=v.dtype)
                grid_fwd = (qpos_x.shape[0], triton.cdiv(qpos_x.shape[1], BLOCK_Q), triton.cdiv(v.shape[2], block_c))
                _deform_attn_fwd_kernel[grid_fwd](
                    qpos_x, qpos_y, offs_x, offs_y, attn_logits, tau, nn4_idx, kv_pos, v, out_dummy,
                    edge_coeff, qpos_x.shape[1], Nk, Hh,
                    C=v.shape[2], S=offs_x.shape[2], H=H, W=W,
                    stride_qpos_bh=qpos_x.stride(0), stride_qpos_q=qpos_x.stride(1),
                    stride_oas_bh=offs_x.stride(0), stride_oas_q=offs_x.stride(1), stride_oas_s=offs_x.stride(2),
                    stride_nn_bh=nn4_idx.stride(0), stride_nn_p=nn4_idx.stride(1), stride_nn_k=nn4_idx.stride(2),
                    stride_kv_bh=kv_pos.stride(0), stride_kv_n=kv_pos.stride(1), stride_kv_d=kv_pos.stride(2),
                    stride_v_bh=v.stride(0), stride_v_n=v.stride(1), stride_v_c=v.stride(2),
                    stride_o_bh=out_dummy.stride(0), stride_o_q=out_dummy.stride(1), stride_o_c=out_dummy.stride(2),
                    stride_coeff_bh=edge_coeff.stride(0), stride_coeff_e=edge_coeff.stride(1),
                    EMIT_EDGES=True, EPS=1e-6, STORE_FP16=(v.dtype == torch.float16),
                    BLOCK_Q=BLOCK_Q, BLOCK_C=block_c, num_warps=num_warps, num_stages=num_stages,
                )
            row_ptr, q_flat, coeff_flat = _build_kv_csr_refs_from_edges_triton(
                edge_coeff, qpos_x, qpos_y, offs_x, offs_y, nn4_idx,
                Nk=Nk, Nq=qpos_x.shape[1], H=H, W=W,
                BLOCK_E=CSR_BLOCK_E, num_warps=CSR_num_warps, num_stages=CSR_num_stages,
            )
            d_v = _compute_dv_gather_triton_from_csr(
                dout, Nk=Nk, row_ptr=row_ptr, q_flat=q_flat, coeff_flat=coeff_flat,
                BLOCK_C=block_c, num_warps=1,
                num_stages=clamp_num_stages(2),
            )
    else:
        raise ValueError(f"Unknown dv_backend: {dv_backend}")
    d_tau = d_tau_partial.sum(dim=1)
    return d_qpos_x, d_qpos_y, d_offs_x, d_offs_y, d_attn_logits, d_tau, d_v


class DeformAttnFunction(Function):
    """
    Autograd wrapper for fused deformable attention kernels.

    Args:
        forward inputs follow `deform_attn(...)` with shapes:
            query_pos [B,Nq,2], kv_pos [B,Nk,2], sampling_offsets [B,Nq,Hh,S,2],
            attn_logits [B,Nq,Hh,S], values [B,Nk,Hh,C], tau scalar/[1],
            nn4_idx [B,H*W,4] (shared table across heads).
    Returns:
        output [B, Nq, Hh, C] and matching gradients in backward.
    """
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.float16)
    def forward(
        ctx,
        query_pos,          # [B, Nq, 2]
        kv_pos,             # [B, Nk, 2]
        sampling_offsets,   # [B, Nq, Hh, S, 2]
        attn_logits,        # [B, Nq, Hh, S]
        values,             # [B, Nk, Hh, C]
        tau,                # scalar or [1]
        nn4_idx,            # [B, H*W, 4]
        grid_h: int,
        grid_w: int,
        dv_backend: str,
        build_backward_state: bool = True,
    ):
        B, Nq, Hh, S, _ = sampling_offsets.shape
        _, Nk, Hv, C = values.shape
        if Hh != Hv:
            raise ValueError("head mismatch between sampling_offsets and values")

        BH = B * Hh
        qx = query_pos[:, None, :, 0].expand(B, Hh, Nq).reshape(BH, Nq).contiguous()
        qy = query_pos[:, None, :, 1].expand(B, Hh, Nq).reshape(BH, Nq).contiguous()
        ox = sampling_offsets[..., 0].permute(0, 2, 1, 3).reshape(BH, Nq, S).contiguous()
        oy = sampling_offsets[..., 1].permute(0, 2, 1, 3).reshape(BH, Nq, S).contiguous()
        a = attn_logits.permute(0, 2, 1, 3).reshape(BH, Nq, S).contiguous()
        kv = kv_pos[:, None, :, :].expand(B, Hh, Nk, 2).reshape(BH, Nk, 2).contiguous().to(values.dtype)
        v = values.permute(0, 2, 1, 3).reshape(BH, Nk, C).contiguous()
        tau_bh = tau.reshape(1).to(torch.float32).expand(BH).contiguous()
        if nn4_idx.ndim != 3 or nn4_idx.shape[0] != B or nn4_idx.shape[1] != grid_h * grid_w or nn4_idx.shape[2] != 4:
            raise ValueError("nn4_idx must be [B, H*W, 4] for shared-table deform attention.")
        nn4_shared = nn4_idx.contiguous().to(torch.uint16)

        # The edge-coefficient CSR structure is read only by the backward
        # pass, and building it costs ~1.6x the forward kernel at 1024. Skip it
        # when nothing will call backward.
        # Decided by the caller. Inside a Function.forward
        # torch.is_grad_enabled() always reads False, and ctx.needs_input_grad
        # can be True under no_grad when an input arrives as an nn.Parameter.
        needs_backward = build_backward_state and any(ctx.needs_input_grad)
        need_edge_cache = (dv_backend in {"csr_cached", "csr_knn_cached"}
                           and needs_backward)
        with record_function("deform_attn.function.forward"):
            if need_edge_cache:
                out, edge_coeff = deform_attn_forward(
                    qx, qy, ox, oy, a, tau_bh, nn4_shared, kv, v,
                    H=grid_h, W=grid_w, return_edge_cache=True,
                )
            else:
                out = deform_attn_forward(
                    qx, qy, ox, oy, a, tau_bh, nn4_shared, kv, v,
                    H=grid_h, W=grid_w, return_edge_cache=False,
                )
                edge_coeff = torch.empty((0,), device=out.device, dtype=torch.float32)
        out_bnhc = out.reshape(B, Hh, Nq, C).permute(0, 2, 1, 3).contiguous()

        ctx.save_for_backward(qx, qy, ox, oy, a, tau_bh, nn4_shared, kv, v, edge_coeff)
        ctx.grid_h = grid_h
        ctx.grid_w = grid_w
        ctx.B = B
        ctx.Hh = Hh
        ctx.Nq = Nq
        ctx.Nk = Nk
        ctx.S = S
        ctx.C = C
        ctx.tau_shape = tuple(tau.shape)
        ctx.dv_backend = dv_backend
        ctx.has_edge_cache = need_edge_cache
        return out_bnhc

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        qx, qy, ox, oy, a, tau_bh, nn4_shared, kv, v, edge_coeff = ctx.saved_tensors
        B = ctx.B
        Hh = ctx.Hh
        Nq = ctx.Nq
        Nk = ctx.Nk
        S = ctx.S
        C = ctx.C
        BH = B * Hh

        dout = grad_out.permute(0, 2, 1, 3).reshape(BH, Nq, C).contiguous()
        with record_function("deform_attn.function.backward"):
            d_qx, d_qy, d_ox, d_oy, d_a, d_tau, d_v = deform_attn_backward(
                dout, qx, qy, ox, oy, a, tau_bh, nn4_shared, kv, v,
                H=ctx.grid_h, W=ctx.grid_w,
                fwd_edge_cache=edge_coeff if ctx.has_edge_cache else None,
                dv_backend=ctx.dv_backend,
            )

        d_query_pos = torch.stack(
            [
                d_qx.reshape(B, Hh, Nq).sum(dim=1),
                d_qy.reshape(B, Hh, Nq).sum(dim=1),
            ],
            dim=-1,
        ).to(grad_out.dtype)
        d_sampling_offsets = torch.stack(
            [
                d_ox.reshape(B, Hh, Nq, S).permute(0, 2, 1, 3),
                d_oy.reshape(B, Hh, Nq, S).permute(0, 2, 1, 3),
            ],
            dim=-1,
        ).to(grad_out.dtype)
        d_attn_logits = d_a.reshape(B, Hh, Nq, S).permute(0, 2, 1, 3).to(grad_out.dtype)
        d_values = d_v.reshape(B, Hh, Nk, C).permute(0, 2, 1, 3).to(grad_out.dtype)

        d_tau_sum = d_tau.sum().to(grad_out.dtype)
        if len(ctx.tau_shape) == 0:
            d_tau_out = d_tau_sum
        elif len(ctx.tau_shape) == 1 and ctx.tau_shape[0] == 1:
            d_tau_out = d_tau_sum.reshape(1)
        else:
            d_tau_out = d_tau_sum.reshape(ctx.tau_shape)

        return (
            d_query_pos,
            None,               # kv_pos
            d_sampling_offsets,
            d_attn_logits,
            d_values,
            d_tau_out,
            None,               # nn4_idx
            None,               # grid_h
            None,               # grid_w
            None,               # dv_backend
            None,               # build_backward_state
        )


def deform_attn(
    query_pos: torch.Tensor,
    kv_pos: torch.Tensor,
    sampling_offsets: torch.Tensor,
    attn_logits: torch.Tensor,
    values: torch.Tensor,
    tau: torch.Tensor,
    nn4_idx: torch.Tensor = None,
    grid_h: int = 64,
    grid_w: int = 64,
    dv_backend: str = "csr_knn_cached",
    build_backward_state: bool = None,
) -> torch.Tensor:
    """
    Run fused deformable point attention (`csr_knn_cached` family).

    Args:
        query_pos: [B, Nq, 2], query coordinates.
        kv_pos: [B, Nk, 2], KV coordinates.
        sampling_offsets: [B, Nq, Hh, S, 2], learned offsets.
        attn_logits: [B, Nq, Hh, S], attention logits over sampling points.
        values: [B, Nk, Hh, C], value vectors.
        tau: scalar or [1], Shepard distance scale.
        nn4_idx: optional [B, H*W, 4] precomputed KNN table.
        grid_h: int, KNN grid height (must be >= img_size // patch_size).
        grid_w: int, KNN grid width (must be >= img_size // patch_size).
        dv_backend: str, one of {"csr_knn_cached", "kv_atomic"}. "csr_cached"
            is accepted as a legacy alias of "csr_knn_cached"; shipped configs
            still use it.
    Returns:
        output tensor [B, Nq, Hh, C].
    """
    if nn4_idx is None:
        kv_int = kv_pos.round().clamp(min=0, max=max(grid_h, grid_w) - 1).to(torch.int32)
        nn4_idx = dense_top4_knn(kv_int, H=grid_h, W=grid_w)  # [B, H*W, 4]
    else:
        nn4_idx = nn4_idx.to(torch.uint16).contiguous()
    if build_backward_state is None:
        # Read grad mode here, at the last point where it is still visible.
        build_backward_state = torch.is_grad_enabled()
    return DeformAttnFunction.apply(
        query_pos,
        kv_pos,
        sampling_offsets,
        attn_logits,
        values,
        tau,
        nn4_idx,
        grid_h,
        grid_w,
        dv_backend,
        build_backward_state,
    )


DeformAttnCSRFunction = DeformAttnFunction


def deform_point_attn(
    query_pos: torch.Tensor,
    kv_pos: torch.Tensor,
    sampling_offsets: torch.Tensor,
    attn_logits: torch.Tensor,
    values: torch.Tensor,
    tau: torch.Tensor,
    nn4_idx: torch.Tensor = None,
    grid_h: int = 64,
    grid_w: int = 64,
    backend: str = "csr_knn_cached",
) -> torch.Tensor:
    """
    Dispatch deformable attention backend.

    Args:
        query_pos: [B, Nq, 2], query coordinates.
        kv_pos: [B, Nk, 2], KV coordinates.
        sampling_offsets: [B, Nq, Hh, S, 2], learned offsets.
        attn_logits: [B, Nq, Hh, S], attention logits.
        values: [B, Nk, Hh, C], value vectors.
        tau: scalar or [1], Shepard distance scale.
        nn4_idx: optional [B, H*W, 4] KNN table.
        grid_h: int, grid height (must be >= img_size // patch_size).
        grid_w: int, grid width (must be >= img_size // patch_size).
        backend: str, one of:
            - "csr_knn_cached": fused Triton path with cached CSR dV option.
              Also accepted as "csr_cached"; shipped configs still use that name.
            - "atomic": fused Triton path with atomic dV updates.
    Returns:
        output tensor [B, Nq, Hh, C].
    """
    if backend in {"csr_cached", "csr_knn_cached"}:
        return deform_attn(
            query_pos=query_pos,
            kv_pos=kv_pos,
            sampling_offsets=sampling_offsets,
            attn_logits=attn_logits,
            values=values,
            tau=tau,
            nn4_idx=nn4_idx,
            grid_h=grid_h,
            grid_w=grid_w,
            dv_backend="csr_knn_cached",
        )
    if backend == "atomic":
        return deform_attn(
            query_pos=query_pos,
            kv_pos=kv_pos,
            sampling_offsets=sampling_offsets,
            attn_logits=attn_logits,
            values=values,
            tau=tau,
            nn4_idx=nn4_idx,
            grid_h=grid_h,
            grid_w=grid_w,
            dv_backend="kv_atomic",
        )
    raise ValueError(f"Unknown deform backend: {backend}")
