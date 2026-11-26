# ruff: noqa
"""PH1S/MP32 V32 sparse MLA prefill: V4 kimi schedule over the 576B KV cache.

Data model follows sparse_mla_v32_fwd_pipelined.py: Q and the unified KV cache
are [*, 576] (512 nope + 64 rope), and the first 512 cache dims dual-serve as
K-noPE and V.  The schedule is the V4 kimi production pipeline (producer robust
gather + split KV ready/free barriers + consumer V staging), extended with a
fifth QK panel for the 64-dim K-rope tail.  sm_scale defaults to 576**-0.5.
"""

from typing import Any

import torch
import tilelang
from tilelang import language as T

from ....utils import cosize
from ....mate_runtime import MP32_ARCH, get_musa_arch, resolve_num_mps
from ..sparse_mla_prefill_common import (
    SPARSE_PREFILL_COMPILE_FLAGS,
    SPARSE_PREFILL_PASS_CONFIGS,
    prepare_sparse_mla_strided_tensor,
    validate_prefill_attn_sink,
    validate_token_lengths,
)
from ..sparse_mla_index_type import jit_for_tensor_addressing
from ....execution_context import raise_complete_if_dry_run


# tilelang.disable_cache()

TARGET = {"kind": "musa", "arch": "mp_32"}

NOPE_HEAD_DIM = 512
KEY_ROPE_DIM = 64
QK_HEAD_DIM = NOPE_HEAD_DIM + KEY_ROPE_DIM
VALUE_HEAD_DIM = NOPE_HEAD_DIM
BLOCK_TOPK = 64
THREADS = 640


V4_SPARSE_PREFILL_PASS_CONFIGS = {
    **SPARSE_PREFILL_PASS_CONFIGS,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
}


@tilelang.jit(
    target=TARGET,
    out_idx=[5, 6, 7],
    pass_configs=V4_SPARSE_PREFILL_PASS_CONFIGS,
    compile_flags=SPARSE_PREFILL_COMPILE_FLAGS,
)
def sparse_mla_prefill_kernel(
    num_heads,
    topk=None,
    *,
    kv_group=1,
    is_causal=True,
    block_i=64,
    threads=640,
    has_attn_sink=False,
    has_topk_length=False,
    is_persistence=True,
    persistent_blocks=48,
    use_rs_qk=None,
    sm_scale=None,
    use_int64_cosize=False,
    use_wide_head_tile=None,
):
    # V32 geometry: 512 nope dims dual-serve as K-noPE and V; the 64-dim
    # rope tail only joins QK.
    dim = 512
    rope_dim = 64
    assert dim == tilelang.math.next_power_of_2(dim), (
        f"haven't check padding correctness yet, dim={dim}"
    )
    if sm_scale is None:
        sm_scale = (dim + rope_dim) ** -0.5
    sm_scale = float(sm_scale) * 1.4426950408889634
    topk = T.dynamic("topk") if topk is None else topk
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    q_stride_s = T.dynamic("q_stride_s")
    q_stride_h = T.dynamic("q_stride_h")
    kv_stride_s = T.dynamic("kv_stride_s")
    kv_stride_g = T.dynamic("kv_stride_g")
    indices_stride_s = T.dynamic("indices_stride_s")
    indices_stride_g = T.dynamic("indices_stride_g")

    head_kv = num_heads // kv_group
    q_shape = [seq_len, num_heads, dim + rope_dim]
    kv_shape = [seq_len_kv, kv_group, dim + rope_dim]
    o_shape = [seq_len, num_heads, dim]
    lse_shape = [seq_len, num_heads]
    max_logits_shape = [seq_len, num_heads]
    indices_shape = [seq_len, kv_group, topk]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"
    dtype_bytes = 2
    q_strides = (q_stride_s, q_stride_h, 1)
    kv_strides = (kv_stride_s, kv_stride_g, 1)
    indices_strides = (indices_stride_s, indices_stride_g, 1)
    q_type: Any = T.StridedTensor(q_shape, q_strides, dtype)
    kv_type: Any = T.StridedTensor(kv_shape, kv_strides, dtype)
    indices_type: Any = T.StridedTensor(indices_shape, indices_strides, indices_dtype)

    padded_head_kv = max(tilelang.math.next_power_of_2(head_kv), 64)
    if padded_head_kv != head_kv:
        assert kv_group == 1
    dim_qk = dim
    qk_panel = min(block_i * 2, 128)
    assert dim_qk % qk_panel == 0
    qk_panels = dim_qk // qk_panel
    assert block_i % 64 == 0
    v_chunks = block_i // 64
    lanes_per_vec = block_i // 8
    topk_blocks = (topk + block_i - 1) // block_i

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        head_repeats = head_kv // 64
    else:
        head_repeats = 1

    # H128 defaults to the narrow 64-head tile (head_repeats=2): with the
    # hoisted register-source QK path it beats the wide 128-head tile at
    # 64k topk (144.6T vs 138.6T) and matches it at 8k.
    if use_wide_head_tile is None:
        use_wide_head_tile = False
    if use_wide_head_tile:
        assert num_heads == 128 and padded_head_kv == 128
        head_repeats = 1
        heads_per_block = 128
    else:
        heads_per_block = padded_head_kv if head_repeats == 1 else 64

    # Case03's highest-performing ASM uses register-source QK.  Start with
    # the shape that has the same M64N128K128 QK tile as the current V4 path;
    # H128 needs a separate M64 split before it can safely use RS operands.
    if use_rs_qk is None:
        use_rs_qk = heads_per_block == 64
    use_rs_qk = bool(use_rs_qk and heads_per_block == 64)
    pv_shape_annotations = (
        {"sqmma_inst_shape": [64, 128, 64]}
        if heads_per_block == 128 and block_i == 64
        else {}
    )
    qk_shape_annotations = (
        {"sqmma_inst_shape": [64, 64, 128]}
        if heads_per_block == 128 and block_i == 64
        else {}
    )

    if is_persistence:
        assert kv_group == 1, "persistent V4 prefill currently supports MQA only"
    logical_blocks = seq_len * head_repeats
    producer_threads = 128
    consumer0_threads = 256
    consumer1_threads = 256
    use_index_vectorization = heads_per_block == 128 and block_i == 64

    def index_loop(extent, explicit=True):
        if use_index_vectorization:
            return T.vectorized(extent)
        return T.unroll(extent, explicit=explicit)

    @T.macro
    def load_q_panels(
        q,
        q_shared,
        s_i,
        h0,
        h1,
        q_robust_desc,
        producer_ldg_tx,
        producer_ldg_ty,
    ):
        # Case03-style producer gather: robust LDLMS copies instead of TMA so
        # Q joins the same ldlms_crossbb protocol as KV (commit/wait at the
        # call site drains the group before bar_q_ready is signalled).
        for r in T.unroll(heads_per_block // 16):
            for panel in T.unroll(qk_panels):
                for u in T.unroll(qk_panel // 64):
                    for v in T.vectorized(8):
                        T.copy(
                            q[
                                s_i,
                                h0 + r * 16 + producer_ldg_ty,
                                panel * qk_panel + 64 * u + producer_ldg_tx * 8 + v,
                            ],
                            q_shared[
                                panel,
                                r * 16 + producer_ldg_ty,
                                64 * u + producer_ldg_tx * 8 + v,
                            ],
                            force_async_copy=True,
                            src_robust_desc=q_robust_desc,
                        )

    @T.macro
    def load_kv_half(
        kv,
        kv_shared,
        kperm_indices_local,
        g_i,
        panel_offset,
        producer_ldg_tx,
        producer_ldg_ty,
        kv_robust_desc,
    ):
        for r in T.unroll(block_i // 16):
            for panel in T.unroll(qk_panels // 2):
                for u in T.unroll(qk_panel // 64):
                    for v in T.vectorized(8):
                        T.copy(
                            kv[
                                kperm_indices_local[r],
                                g_i,
                                (panel + panel_offset) * qk_panel
                                + 64 * u
                                + producer_ldg_tx * 8
                                + v,
                            ],
                            kv_shared[
                                panel + panel_offset,
                                r * 16 + producer_ldg_ty,
                                64 * u + producer_ldg_tx * 8 + v,
                            ],
                            force_async_copy=True,
                            src_robust_desc=kv_robust_desc,
                        )

    @T.macro
    def load_kv_rope(
        kv,
        k_rope_shared,
        kperm_indices_local,
        g_i,
        producer_ldg_tx,
        producer_ldg_ty,
        kv_robust_desc,
    ):
        # 64-dim K-rope tail lives at cache offset 512..575 of the same row.
        for r in T.unroll(block_i // 16):
            for v in T.vectorized(8):
                T.copy(
                    kv[
                        kperm_indices_local[r],
                        g_i,
                        dim + producer_ldg_tx * 8 + v,
                    ],
                    k_rope_shared[r * 16 + producer_ldg_ty, producer_ldg_tx * 8 + v],
                    force_async_copy=True,
                    src_robust_desc=kv_robust_desc,
                )

    @T.macro
    def mask_init_scores(acc_s, is_kv_valid):
        for h_i, bi_i in T.Parallel(heads_per_block, block_i):
            acc_s[h_i, bi_i] = T.if_then_else(
                T.Cast(
                    "bool",
                    is_kv_valid[
                        (bi_i // 64) * 64 + (bi_i % 64) % 8 * 8 + (bi_i % 64) // 8
                    ],
                ),
                0,
                -(2**30),
            )

    @T.macro
    def annotate_kv01(kv_shared):
        T.annotate_layout(
            {
                kv_shared[0, :, :]: tilelang.layout.make_sqmma_swizzled_layout(
                    kv_shared[0, :, :],
                    k_major=True,
                    swizzle_granularity=16,
                ),
                kv_shared[1, :, :]: tilelang.layout.make_sqmma_swizzled_layout(
                    kv_shared[1, :, :],
                    k_major=True,
                    swizzle_granularity=16,
                ),
            },
            allow_reannotation=True,
            allow_buffer_region=True,
        )

    @T.macro
    def qk01_gemms(kv_shared, q_reg_0, q_reg_1, q_shared, acc_s):
        if use_rs_qk:
            T.gemm(
                q_reg_0,
                kv_shared[0, :, :],
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
                wg_wait=-1,
            )
            T.gemm(
                q_reg_1,
                kv_shared[1, :, :],
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
                wg_wait=-1,
            )
        else:
            T.gemm(
                q_shared[0, :, :],
                kv_shared[0, :, :],
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
                wg_wait=-1,
            )
            T.gemm(
                q_shared[1, :, :],
                kv_shared[1, :, :],
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
                wg_wait=-1,
            )

    @T.macro
    def dsa_prefill_body(
        q,
        kv,
        indices,
        topk_length,
        attn_sink,
        output,
        max_logits_out,
        lse,
        sm_scale,
        persistent_blocks,
        q_cosize,
        kv_cosize,
    ):
        launch_blocks = (
            T.min(persistent_blocks, logical_blocks)
            if is_persistence
            else logical_blocks
        )
        with T.Kernel(launch_blocks, kv_group, threads=threads) as (bx, by):
            T.assume(q_stride_s % 8 == 0)
            T.assume(q_stride_h % 8 == 0)
            T.assume(kv_stride_s % 8 == 0)
            T.assume(kv_stride_g % 8 == 0)
            T.assume(indices_stride_s % 8 == 0)
            T.assume(indices_stride_g % 8 == 0)
            q_shared = (
                T.alloc_shared([qk_panels, heads_per_block, qk_panel], dtype)
                if not use_rs_qk
                else None
            )
            kv_shared = T.alloc_shared([qk_panels, block_i, qk_panel], dtype)
            # 64-dim K-rope stage; gathered by the producer with the kv1 half
            # and read by QK's fifth (register-source) gemm.
            k_rope_shared = T.alloc_shared([block_i, rope_dim], dtype)
            v_shared_0 = T.alloc_shared([64, dim_qk // 4], dtype)
            v_shared_1 = T.alloc_shared([64, dim_qk // 4], dtype)
            # block_i==64 path stages the full 512-dim V per tile in four
            # 128-dim chunks.  Each consumer group owns two chunks end to end
            # (stage -> own PV gemms), so the per-chunk ready/free handshakes
            # of the legacy two-buffer design collapse to one group sync per
            # group per tile.
            v_buf = (
                T.alloc_shared([4, 64, dim_qk // 4], dtype) if block_i == 64 else None
            )
            # Keep V shared layout inferred by the consumer.  The explicit
            # Case01 V layout causes PH1S reannotation conflicts in the fused
            # online-softmax path and regresses H128.
            case01_v0_layout = {}
            case01_v1_layout = {}

            if block_i == 64:
                scores_shared = T.alloc_shared([heads_per_block, block_i], dtype)
            else:
                scores_shared = T.alloc_shared([v_chunks, heads_per_block, 64], dtype)
            sum_exp_inv_shared = T.alloc_shared([heads_per_block], accum_dtype)
            alpha_shared = T.alloc_shared([heads_per_block], accum_dtype)
            is_kv_valid = T.alloc_shared([block_i], "int32", scope="shared")

            bar_q_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_q_free = T.alloc_barrier(arrive_count=consumer0_threads)

            bar_kv0_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_kv1_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_kv1_read_ready = (
                T.alloc_barrier(arrive_count=consumer0_threads)
                if block_i != 64
                else None
            )
            bar_kv0_free = T.alloc_barrier(arrive_count=consumer0_threads)
            # kv_shared right panels are consumed by both groups: consumer0's
            # async SQMMA and consumer1's LMA read.  Producer reuse must wait
            # for both completions (consumer0 + consumer1 arrivals).
            bar_kv1_free = T.alloc_barrier(
                arrive_count=consumer0_threads + consumer1_threads
            )

            bar_vl_pair_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_vr0_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_vr1_ready = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_vl0_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_vl1_free = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_vr0_free = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_vr1_free = T.alloc_barrier(arrive_count=consumer1_threads)

            bar_p_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_alpha_ready = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_final = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_final_free = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_producer_protect = T.alloc_barrier(arrive_count=producer_threads)

            q_robust_desc = T.make_robust_desc(
                T.address_of(q[0, 0, 0]),
                q_cosize * dtype_bytes,
            )
            kv_robust_desc = T.make_robust_desc(
                T.address_of(kv[0, 0, 0]),
                kv_cosize * dtype_bytes,
            )
            T.sync_threads()

            tid = T.get_thread_binding()
            logical_bx = T.alloc_var(T.int32)
            phase_count = T.alloc_local([1], T.int32)
            logical_phase = T.alloc_local([1], T.int32)
            T.fill(phase_count, 0)
            T.fill(logical_phase, 0)
            logical_bx = bx
            if tid >= 512 and logical_bx < seq_len * head_repeats:
                bootstrap_s_i = (
                    logical_bx if head_repeats == 1 else logical_bx // head_repeats
                )
                bootstrap_h0 = by * padded_head_kv + (
                    0 if head_repeats == 1 else (logical_bx % head_repeats) * 64
                )
                bootstrap_h1 = bootstrap_h0 + heads_per_block
                if not use_rs_qk:
                    load_q_panels(
                        q,
                        q_shared,
                        bootstrap_s_i,
                        bootstrap_h0,
                        bootstrap_h1,
                        q_robust_desc,
                        (tid - 512) % 8,
                        (tid - 512) // 8,
                    )
                    T.ldlms_crossbb_commit_0()
                    T.ldlms_crossbb_wait_0()
                T.barrier_arrive(bar_q_ready)

            if tid >= 256 and tid < 512:
                T.barrier_arrive(bar_vr0_free)
                T.barrier_arrive(bar_vr1_free)

            while logical_bx < seq_len * head_repeats:
                if is_persistence:
                    T.call_extern("void", "__musa_loop_transparent_outermost")
                g_i = by
                s_i = logical_bx if head_repeats == 1 else (logical_bx // head_repeats)
                h0 = g_i * padded_head_kv + (
                    0 if head_repeats == 1 else (logical_bx % head_repeats) * 64
                )
                h1 = h0 + heads_per_block
                dynamic_total_blocks = T.alloc_var(T.int32)
                if has_topk_length:
                    dynamic_total_blocks = T.ceildiv(topk_length[s_i], block_i)
                else:
                    dynamic_total_blocks = T.ceildiv(topk, block_i)

                if tid < 256:
                    sumexp = T.alloc_fragment([heads_per_block], accum_dtype)
                    sumexp_i = T.alloc_fragment([heads_per_block], accum_dtype)
                    sumexp_inv = T.alloc_fragment([heads_per_block], accum_dtype)
                    alpha_local = T.alloc_fragment([heads_per_block], accum_dtype)
                    m_i = T.alloc_fragment([heads_per_block], accum_dtype)
                    m_i_prev = T.alloc_fragment([heads_per_block], accum_dtype)
                    max_logits = T.alloc_fragment([heads_per_block], accum_dtype)
                    acc_s = T.alloc_fragment([heads_per_block, block_i], accum_dtype)
                    acc_s_cast = T.alloc_fragment([heads_per_block, block_i], dtype)
                    if heads_per_block == 128 and block_i == 64:
                        T.annotate_layout(
                            {
                                acc_s: T.Fragment(
                                    (heads_per_block, block_i),
                                    forward_thread_fn=lambda fr, fc: (fr // 64) * 128
                                    + (fr % 32) * 4
                                    + fc % 4,
                                    forward_index_fn=lambda fr, fc: (fr % 64 // 32)
                                    * (block_i // 4)
                                    + fc // 4,
                                ),
                                m_i: T.Fragment(
                                    (heads_per_block,),
                                    replicate=4,
                                    forward_thread_fn=lambda fr, rep: (fr // 64) * 128
                                    + (fr % 32) * 4
                                    + rep,
                                    forward_index_fn=lambda fr: fr % 64 // 32,
                                ),
                            }
                        )
                    if use_rs_qk:
                        q_reg_0 = T.alloc_fragment([heads_per_block, qk_panel], dtype)
                        q_reg_1 = T.alloc_fragment([heads_per_block, qk_panel], dtype)
                        q_reg_2 = T.alloc_fragment([heads_per_block, qk_panel], dtype)
                        q_reg_3 = T.alloc_fragment([heads_per_block, qk_panel], dtype)
                    # Rope Q is always register-source (avoids q_shared rope SMEM).
                    q_rope_reg = T.alloc_fragment([heads_per_block, rope_dim], dtype)
                    acc_o_l_0 = T.alloc_fragment(
                        [heads_per_block, dim_qk // 4], accum_dtype
                    )
                    acc_o_l_1 = T.alloc_fragment(
                        [heads_per_block, dim_qk // 4], accum_dtype
                    )
                    kv_reg_l = T.alloc_local([block_i], dtype)
                    score_pack = T.alloc_local([8], dtype)
                    v_phase = T.alloc_local([1], T.int32)
                    T.fill(v_phase, 0)
                    consumer0_ldg_tx = tid % 8
                    consumer0_ldg_ty = tid // 8
                    T.fill(sumexp, 0)
                    T.fill(m_i, -(2**30))
                    T.fill(acc_o_l_0, 0)
                    T.fill(acc_o_l_1, 0)
                    T.barrier_wait(bar_q_ready, logical_phase[0] & 1)
                    # Rope Q segment is loop-invariant: load once per block
                    # (a per-tile reload cost ~2x kernel time on H128).
                    T.copy(
                        q[s_i, h0:h1, dim : dim + rope_dim],
                        q_rope_reg,
                    )
                    if use_rs_qk:
                        # All four nope Q panels are loop-invariant too:
                        # hoist the global loads out of the tile loop.
                        T.copy(
                            q[s_i, h0:h1, 0:qk_panel],
                            q_reg_0,
                        )
                        T.copy(
                            q[s_i, h0:h1, qk_panel : 2 * qk_panel],
                            q_reg_1,
                        )
                        T.copy(
                            q[
                                s_i,
                                h0:h1,
                                (qk_panels // 2) * qk_panel : (qk_panels // 2 + 1)
                                * qk_panel,
                            ],
                            q_reg_2,
                        )
                        T.copy(
                            q[
                                s_i,
                                h0:h1,
                                (qk_panels // 2 + 1) * qk_panel : (qk_panels // 2 + 2)
                                * qk_panel,
                            ],
                            q_reg_3,
                        )
                    # K-rope stage only serves QK: annotate once (its layout
                    # is loop-invariant, unlike kv_shared which flips for PV).
                    T.annotate_layout(
                        {
                            k_rope_shared[
                                :, :
                            ]: tilelang.layout.make_sqmma_swizzled_layout(
                                k_rope_shared,
                                k_major=True,
                                swizzle_granularity=16,
                            ),
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True,
                    )
                    qk_ahead = block_i == 64 and head_repeats == 2
                    if qk_ahead and dynamic_total_blocks > 0:
                        # Software-pipeline prologue: tile-0 QK panels 0/1 are
                        # issued before the loop; each tile tail then issues the
                        # next tile's QK panels 0/1 right after its own PV gemms,
                        # keeping the SQMMA pipe fed across the tile boundary.
                        T.barrier_wait(bar_kv0_ready, phase_count[0] & 1)
                        mask_init_scores(acc_s, is_kv_valid)
                        annotate_kv01(kv_shared)
                        qk01_gemms(
                            kv_shared,
                            q_reg_0 if use_rs_qk else None,
                            q_reg_1 if use_rs_qk else None,
                            q_shared,
                            acc_s,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                    for i_i in range(dynamic_total_blocks):
                        if not qk_ahead:
                            T.barrier_wait(bar_kv0_ready, phase_count[0] & 1)
                            mask_init_scores(acc_s, is_kv_valid)
                            annotate_kv01(kv_shared)
                            if use_rs_qk:
                                T.gemm(
                                    q_reg_0,
                                    kv_shared[0, :, :],
                                    acc_s,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    wg_wait=-1,
                                )
                            else:
                                T.gemm(
                                    q_shared[0, :, :],
                                    kv_shared[0, :, :],
                                    acc_s,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    wg_wait=-1,
                                )
                        else:
                            # QK panels 0/1 for this tile were issued in the
                            # previous tail (or the prologue); only the register
                            # staging reads of panels 0/1 remain here.
                            pass
                        for r in T.unroll(block_i // 32):
                            for v in T.vectorized(8):
                                kv_reg_l[r * 32 + v] = kv_shared[
                                    0,
                                    ((consumer0_ldg_ty + r * 32) // 64) * 64
                                    + ((consumer0_ldg_ty + r * 32) % 64) % 8 * 8
                                    + ((consumer0_ldg_ty + r * 32) % 64) // 8,
                                    consumer0_ldg_tx * 8 + v,
                                ]
                        for r in T.unroll(block_i // 32):
                            for v in T.vectorized(8):
                                kv_reg_l[r * 32 + 8 + v] = kv_shared[
                                    0,
                                    ((consumer0_ldg_ty + r * 32) // 64) * 64
                                    + ((consumer0_ldg_ty + r * 32) % 64) % 8 * 8
                                    + ((consumer0_ldg_ty + r * 32) % 64) // 8,
                                    64 + consumer0_ldg_tx * 8 + v,
                                ]

                        if not qk_ahead:
                            if use_rs_qk:
                                T.gemm(
                                    q_reg_1,
                                    kv_shared[1, :, :],
                                    acc_s,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    wg_wait=-1,
                                )
                            else:
                                T.gemm(
                                    q_shared[1, :, :],
                                    kv_shared[1, :, :],
                                    acc_s,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    wg_wait=-1,
                                )
                        if block_i != 64:
                            T.barrier_wait(bar_vr0_free, v_phase[0] & 1)
                        if block_i == 64:
                            for r in T.unroll(2):
                                for v in T.vectorized(8):
                                    v_buf[
                                        0,
                                        r * 32 + consumer0_ldg_ty,
                                        consumer0_ldg_tx * 8 + v,
                                    ] = kv_reg_l[r * 32 + v]
                            for r in T.unroll(2):
                                for v in T.vectorized(8):
                                    v_buf[
                                        0,
                                        r * 32 + consumer0_ldg_ty,
                                        64 + consumer0_ldg_tx * 8 + v,
                                    ] = kv_reg_l[r * 32 + 8 + v]
                        else:
                            for r in T.unroll(2):
                                for v in T.vectorized(8):
                                    v_shared_0[
                                        r * 32 + consumer0_ldg_ty,
                                        consumer0_ldg_tx * 8 + v,
                                    ] = kv_reg_l[r * 32 + v]
                            for r in T.unroll(2):
                                for v in T.vectorized(8):
                                    v_shared_0[
                                        r * 32 + consumer0_ldg_ty,
                                        64 + consumer0_ldg_tx * 8 + v,
                                    ] = kv_reg_l[r * 32 + 8 + v]
                        for r in T.unroll(block_i // 32):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    kv_reg_l[r * 32 + (u + 2) * 8 + v] = kv_shared[
                                        1,
                                        ((consumer0_ldg_ty + r * 32) // 64) * 64
                                        + ((consumer0_ldg_ty + r * 32) % 64) % 8 * 8
                                        + ((consumer0_ldg_ty + r * 32) % 64) // 8,
                                        64 * u + consumer0_ldg_tx * 8 + v,
                                    ]
                        if qk_ahead:
                            # QK panels 0/1 already drained in the previous
                            # tail; only the staging reads gate kv0 reuse.
                            T.lma_wait()
                            T.barrier_arrive(bar_kv0_free)
                        else:
                            T.warpgroup_commit_batch()
                            T.warpgroup_wait(0)
                            T.lma_wait()
                            T.barrier_arrive(bar_kv0_free)

                        T.barrier_wait(bar_kv1_ready, phase_count[0] & 1)
                        if block_i != 64:
                            T.barrier_arrive(bar_kv1_read_ready)
                        T.annotate_layout(
                            {
                                kv_shared[
                                    qk_panels // 2, :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared[qk_panels // 2, :, :],
                                    k_major=True,
                                    swizzle_granularity=16,
                                ),
                                kv_shared[
                                    qk_panels // 2 + 1, :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared[qk_panels // 2 + 1, :, :],
                                    k_major=True,
                                    swizzle_granularity=16,
                                ),
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        if use_rs_qk:
                            T.gemm(
                                q_reg_2,
                                kv_shared[qk_panels // 2, :, :],
                                acc_s,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                wg_wait=-1,
                            )
                        else:
                            T.gemm(
                                q_shared[qk_panels // 2, :, :],
                                kv_shared[qk_panels // 2, :, :],
                                acc_s,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                wg_wait=-1,
                            )
                        if block_i != 64:
                            T.barrier_wait(bar_vr1_free, v_phase[0] & 1)
                        if block_i == 64:
                            for v in T.vectorized(8):
                                v_buf[
                                    1,
                                    consumer0_ldg_ty,
                                    consumer0_ldg_tx * 8 + v,
                                ] = kv_reg_l[16 + v]
                        else:
                            for v in T.vectorized(8):
                                v_shared_1[
                                    consumer0_ldg_ty,
                                    consumer0_ldg_tx * 8 + v,
                                ] = kv_reg_l[16 + v]
                        if use_rs_qk:
                            T.gemm(
                                q_reg_3,
                                kv_shared[qk_panels // 2 + 1, :, :],
                                acc_s,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                wg_wait=-1,
                            )
                        else:
                            T.gemm(
                                q_shared[qk_panels // 2 + 1, :, :],
                                kv_shared[qk_panels // 2 + 1, :, :],
                                acc_s,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                wg_wait=-1,
                            )
                        if block_i == 64:
                            for v in T.vectorized(8):
                                v_buf[
                                    1,
                                    consumer0_ldg_ty,
                                    64 + consumer0_ldg_tx * 8 + v,
                                ] = kv_reg_l[24 + v]
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    v_buf[
                                        1,
                                        32 + consumer0_ldg_ty,
                                        64 * u + consumer0_ldg_tx * 8 + v,
                                    ] = kv_reg_l[48 + u * 8 + v]
                        else:
                            for v in T.vectorized(8):
                                v_shared_1[
                                    consumer0_ldg_ty,
                                    64 + consumer0_ldg_tx * 8 + v,
                                ] = kv_reg_l[24 + v]
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    v_shared_1[
                                        32 + consumer0_ldg_ty,
                                        64 * u + consumer0_ldg_tx * 8 + v,
                                    ] = kv_reg_l[48 + u * 8 + v]
                        # Fifth QK panel: 64-dim rope tail (register-source Q,
                        # shared K_rope staged by the producer with kv1).
                        T.gemm(
                            q_rope_reg,
                            k_rope_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.lma_wait()
                        T.barrier_arrive(bar_vl_pair_ready)
                        T.copy(m_i, m_i_prev)
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_kv1_free)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(heads_per_block):
                            m_i[h_i] = T.max(m_i_prev[h_i], m_i[h_i])
                        for h_i in T.Parallel(heads_per_block):
                            alpha_local[h_i] = T.exp2(
                                (m_i_prev[h_i] - m_i[h_i]) * sm_scale
                            )
                        for h_i, bi_i in T.Parallel(heads_per_block, block_i):
                            acc_s[h_i, bi_i] = T.exp2(
                                acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                            )

                        # scores_shared/alpha_shared are single-buffered: gate
                        # their overwrite on consumer1's PV gemms of the
                        # previous tile (signalled by bar_vr0_free).  Only the
                        # block_i==64 path needs this; the chunked bi128 path
                        # keeps its own per-chunk free/wait protocol.
                        if block_i == 64:
                            T.barrier_wait(bar_vr0_free, phase_count[0] & 1)
                        T.copy(alpha_local, alpha_shared)
                        T.barrier_arrive(bar_alpha_ready)
                        T.copy(acc_s, acc_s_cast)
                        if block_i == 64:
                            if heads_per_block == 64:
                                for i, t in T.Parallel(heads_per_block, 8):
                                    base = t * lanes_per_vec
                                    for l in T.unroll(lanes_per_vec, explicit=True):
                                        score_pack[l] = acc_s_cast[i, l * 8 + t]
                                    for l in T.vectorized(lanes_per_vec):
                                        scores_shared[i, base + l] = score_pack[l]
                            else:
                                for i, t in T.Parallel(heads_per_block, 8):
                                    base = t * lanes_per_vec
                                    for l in T.vectorized(lanes_per_vec):
                                        scores_shared[i, base + l] = acc_s_cast[
                                            i, l * 8 + t
                                        ]

                            T.lma_wait()
                            T.barrier_arrive(bar_p_ready)
                            # Both acc_o rescales must complete before this
                            # tile's PV gemms accumulate into them.
                            for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                                acc_o_l_0[h_i, d_i] *= alpha_local[h_i]
                            for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                                acc_o_l_1[h_i, d_i] *= alpha_local[h_i]

                            T.reduce_sum(acc_s, sumexp_i, dim=1)
                            for h_i in T.Parallel(heads_per_block):
                                sumexp[h_i] = (
                                    sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                                )

                            # Group-local sync: v_buf[0, :, :]/v_buf[1, :, :] staging (from
                            # this group's warps) is visible to the SQMMA reads.
                            T.barrier_wait(bar_vl_pair_ready, phase_count[0] & 1)

                            T.gemm(
                                scores_shared,
                                v_buf[0, :, :],
                                acc_o_l_0,
                                policy=T.GemmWarpPolicy.FullRow,
                                wg_wait=-1,
                            )
                            T.gemm(
                                scores_shared,
                                v_buf[1, :, :],
                                acc_o_l_1,
                                policy=T.GemmWarpPolicy.FullRow,
                                wg_wait=-1,
                            )
                            T.warpgroup_commit_batch()
                            T.sched_boundary()
                            if qk_ahead and i_i + 1 < dynamic_total_blocks:
                                mask_init_scores(acc_s, is_kv_valid)
                                T.barrier_wait(bar_kv0_ready, (phase_count[0] & 1) ^ 1)
                                annotate_kv01(kv_shared)
                                qk01_gemms(
                                    kv_shared,
                                    q_reg_0 if use_rs_qk else None,
                                    q_reg_1 if use_rs_qk else None,
                                    q_shared,
                                    acc_s,
                                )
                                T.warpgroup_commit_batch()
                            T.warpgroup_wait(0)
                            phase_count[0] = phase_count[0] ^ 1
                        else:
                            for vc in T.unroll(v_chunks):
                                if heads_per_block == 64:
                                    for i, t in T.Parallel(heads_per_block, 8):
                                        base = t * 8
                                        for l in T.unroll(8, explicit=True):
                                            score_pack[l] = acc_s_cast[
                                                i, vc * 64 + l * 8 + t
                                            ]
                                        for l in T.vectorized(8):
                                            scores_shared[vc, i, base + l] = score_pack[
                                                l
                                            ]
                                else:
                                    for i, t in T.Parallel(heads_per_block, 8):
                                        base = t * 8
                                        for l in T.vectorized(8):
                                            scores_shared[vc, i, base + l] = acc_s_cast[
                                                i, vc * 64 + l * 8 + t
                                            ]

                            T.lma_wait()
                            for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                                acc_o_l_0[h_i, d_i] *= alpha_local[h_i]

                            for vc in T.unroll(v_chunks):
                                if vc > 0:
                                    T.barrier_wait(bar_vr0_free, v_phase[0] & 1)
                                    for r in T.unroll(2):
                                        for v in T.vectorized(8):
                                            v_shared_0[
                                                r * 32 + consumer0_ldg_ty,
                                                consumer0_ldg_tx * 8 + v,
                                            ] = kv_reg_l[vc * 64 + r * 32 + v]
                                    for r in T.unroll(2):
                                        for v in T.vectorized(8):
                                            v_shared_0[
                                                r * 32 + consumer0_ldg_ty,
                                                64 + consumer0_ldg_tx * 8 + v,
                                            ] = kv_reg_l[vc * 64 + r * 32 + 8 + v]
                                    T.barrier_wait(bar_vr1_free, v_phase[0] & 1)
                                    for v in T.vectorized(8):
                                        v_shared_1[
                                            consumer0_ldg_ty,
                                            consumer0_ldg_tx * 8 + v,
                                        ] = kv_reg_l[vc * 64 + 16 + v]
                                    for v in T.vectorized(8):
                                        v_shared_1[
                                            consumer0_ldg_ty,
                                            64 + consumer0_ldg_tx * 8 + v,
                                        ] = kv_reg_l[vc * 64 + 24 + v]
                                    for u in T.unroll(2):
                                        for v in T.vectorized(8):
                                            v_shared_1[
                                                32 + consumer0_ldg_ty,
                                                64 * u + consumer0_ldg_tx * 8 + v,
                                            ] = kv_reg_l[vc * 64 + 48 + u * 8 + v]
                                    T.lma_wait()
                                    T.barrier_arrive(bar_vl_pair_ready)
                                T.annotate_layout(
                                    case01_v0_layout,
                                    allow_reannotation=True,
                                    allow_buffer_region=True,
                                )
                                T.barrier_wait(bar_vl_pair_ready, v_phase[0] & 1)

                                T.gemm(
                                    scores_shared[vc, :, :],
                                    v_shared_0,
                                    acc_o_l_0,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    wg_wait=-1,
                                )
                                T.warpgroup_commit_batch()
                                if vc == 0:
                                    T.barrier_arrive(bar_p_ready)
                                    for h_i, d_i in T.Parallel(
                                        heads_per_block, dim_qk // 4
                                    ):
                                        acc_o_l_1[h_i, d_i] *= alpha_local[h_i]

                                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                                    for h_i in T.Parallel(heads_per_block):
                                        sumexp[h_i] = (
                                            sumexp[h_i] * alpha_local[h_i]
                                            + sumexp_i[h_i]
                                        )
                                T.annotate_layout(
                                    case01_v1_layout,
                                    allow_reannotation=True,
                                    allow_buffer_region=True,
                                )
                                T.warpgroup_wait(0)
                                T.barrier_arrive(bar_vl0_free)

                                T.gemm(
                                    scores_shared[vc, :, :],
                                    v_shared_1,
                                    acc_o_l_1,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    wg_wait=-1,
                                )
                                T.warpgroup_commit_batch()
                                T.warpgroup_wait(0)
                                T.barrier_arrive(bar_vl1_free)
                                v_phase[0] = v_phase[0] ^ 1
                            phase_count[0] = phase_count[0] ^ 1

                    T.barrier_arrive(bar_q_free)
                    T.barrier_wait(bar_final_free, (logical_phase[0] & 1) ^ 1)
                    for h_i in T.Parallel(heads_per_block):
                        if m_i[h_i] > -(2**29):
                            if has_attn_sink:
                                sumexp_inv[h_i] = 1 / (
                                    sumexp[h_i]
                                    + T.exp2(
                                        attn_sink[h0 + h_i] * 1.4426950408889634
                                        - m_i[h_i] * sm_scale
                                    )
                                )
                            else:
                                sumexp_inv[h_i] = 1 / sumexp[h_i]
                            max_logits[h_i] = m_i[h_i] * sm_scale * 0.6931471805599453
                            sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                        else:
                            sumexp_inv[h_i] = 0
                            max_logits[h_i] = -T.infinity(accum_dtype)
                            sumexp[h_i] = T.infinity(accum_dtype)
                        sum_exp_inv_shared[h_i] = sumexp_inv[h_i]
                    T.barrier_arrive(bar_final)
                    for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                        acc_o_l_0[h_i, d_i] *= sumexp_inv[h_i]
                        acc_o_l_1[h_i, d_i] *= sumexp_inv[h_i]

                    T.copy(acc_o_l_0, output[s_i, h0:h1, 0 : dim_qk // 4])
                    T.copy(acc_o_l_1, output[s_i, h0:h1, dim_qk // 4 : dim_qk // 2])
                    T.copy(max_logits, max_logits_out[s_i, h0:h1])
                    for h_i in T.Parallel(heads_per_block):
                        lse[s_i, h0 + h_i] = sumexp[h_i] * 0.6931471805599453
                elif tid < 512:
                    acc_o_r_0 = T.alloc_fragment(
                        [heads_per_block, dim_qk // 4], accum_dtype
                    )
                    acc_o_r_1 = T.alloc_fragment(
                        [heads_per_block, dim_qk // 4], accum_dtype
                    )
                    kv_reg_r = T.alloc_local([block_i], dtype)
                    alpha_r = T.alloc_fragment([heads_per_block], accum_dtype)
                    v_phase = T.alloc_local([1], T.int32)
                    T.fill(v_phase, 0)
                    T.fill(acc_o_r_0, 0)
                    T.fill(acc_o_r_1, 0)

                    consumer1_ldg_tx = (tid - 256) % 8
                    consumer1_ldg_ty = (tid - 256) // 8
                    T.barrier_wait(bar_q_ready, logical_phase[0] & 1)
                    for i_i in range(dynamic_total_blocks):
                        if block_i == 64:
                            T.barrier_wait(bar_kv1_ready, phase_count[0] & 1)
                        else:
                            T.barrier_wait(bar_kv1_read_ready, phase_count[0] & 1)
                        for r in T.unroll(block_i // 32):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    kv_reg_r[r * 32 + u * 8 + v] = kv_shared[
                                        qk_panels // 2,
                                        ((consumer1_ldg_ty + r * 32) // 64) * 64
                                        + ((consumer1_ldg_ty + r * 32) % 64) % 8 * 8
                                        + ((consumer1_ldg_ty + r * 32) % 64) // 8,
                                        64 * u + consumer1_ldg_tx * 8 + v,
                                    ]
                                    kv_reg_r[r * 32 + (u + 2) * 8 + v] = kv_shared[
                                        qk_panels // 2 + 1,
                                        ((consumer1_ldg_ty + r * 32) // 64) * 64
                                        + ((consumer1_ldg_ty + r * 32) % 64) % 8 * 8
                                        + ((consumer1_ldg_ty + r * 32) % 64) // 8,
                                        64 * u + consumer1_ldg_tx * 8 + v,
                                    ]

                        T.lma_wait()
                        T.barrier_arrive(bar_kv1_free)
                        if block_i == 64:
                            # Stage both owned V chunks immediately (v_buf[2, :, :]/[3]
                            # are exclusive to this group; previous-tile PV is
                            # complete by program order via wait_wgmma below).
                            for r in T.unroll(2):
                                for u in T.unroll(2):
                                    for v in T.vectorized(8):
                                        v_buf[
                                            2,
                                            r * 32 + consumer1_ldg_ty,
                                            64 * u + consumer1_ldg_tx * 8 + v,
                                        ] = kv_reg_r[r * 32 + u * 8 + v]
                            for r in T.unroll(2):
                                for u in T.unroll(2):
                                    for v in T.vectorized(8):
                                        v_buf[
                                            3,
                                            r * 32 + consumer1_ldg_ty,
                                            64 * u + consumer1_ldg_tx * 8 + v,
                                        ] = kv_reg_r[r * 32 + (u + 2) * 8 + v]
                            T.lma_wait()
                            T.barrier_arrive(bar_vr0_ready)
                            T.barrier_wait(bar_alpha_ready, phase_count[0] & 1)
                            for h_i in T.Parallel(heads_per_block):
                                alpha_r[h_i] = alpha_shared[h_i]
                            for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                                acc_o_r_0[h_i, d_i] *= alpha_r[h_i]
                                acc_o_r_1[h_i, d_i] *= alpha_r[h_i]
                            T.barrier_wait(bar_p_ready, phase_count[0] & 1)
                            T.barrier_wait(bar_vr0_ready, phase_count[0] & 1)

                            T.gemm(
                                scores_shared,
                                v_buf[2, :, :],
                                acc_o_r_0,
                                policy=T.GemmWarpPolicy.FullRow,
                                wg_wait=-1,
                            )
                            T.gemm(
                                scores_shared,
                                v_buf[3, :, :],
                                acc_o_r_1,
                                policy=T.GemmWarpPolicy.FullRow,
                                wg_wait=-1,
                            )
                            T.wait_wgmma(0)
                            T.barrier_arrive(bar_vr0_free)
                            phase_count[0] = phase_count[0] ^ 1
                        else:
                            T.barrier_wait(bar_alpha_ready, phase_count[0] & 1)
                            for h_i in T.Parallel(heads_per_block):
                                alpha_r[h_i] = alpha_shared[h_i]
                            for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                                acc_o_r_0[h_i, d_i] *= alpha_r[h_i]
                                acc_o_r_1[h_i, d_i] *= alpha_r[h_i]
                            T.barrier_wait(bar_p_ready, phase_count[0] & 1)
                            T.sched_boundary()
                            for vc in T.unroll(v_chunks):
                                if vc == 0:
                                    T.sched_boundary()
                                T.barrier_wait(bar_vl0_free, v_phase[0] & 1)
                                T.annotate_layout(
                                    case01_v0_layout,
                                    allow_reannotation=True,
                                    allow_buffer_region=True,
                                )
                                for r in T.unroll(2):
                                    for u in T.unroll(2):
                                        for v in T.vectorized(8):
                                            v_shared_0[
                                                r * 32 + consumer1_ldg_ty,
                                                64 * u + consumer1_ldg_tx * 8 + v,
                                            ] = kv_reg_r[vc * 64 + r * 32 + u * 8 + v]

                                T.lma_wait()
                                if vc == 0:
                                    T.sched_boundary()
                                T.barrier_arrive(bar_vr0_ready)
                                T.barrier_wait(bar_vr0_ready, v_phase[0] & 1)

                                T.gemm(
                                    scores_shared[vc, :, :],
                                    v_shared_0,
                                    acc_o_r_0,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    wg_wait=-1,
                                )
                                T.wait_wgmma(0)
                                T.barrier_arrive(bar_vr0_free)
                                if vc == 0:
                                    T.sched_boundary()

                                T.barrier_wait(bar_vl1_free, v_phase[0] & 1)
                                T.annotate_layout(
                                    case01_v1_layout,
                                    allow_reannotation=True,
                                    allow_buffer_region=True,
                                )
                                for r in T.unroll(2):
                                    for u in T.unroll(2):
                                        for v in T.vectorized(8):
                                            v_shared_1[
                                                r * 32 + consumer1_ldg_ty,
                                                64 * u + consumer1_ldg_tx * 8 + v,
                                            ] = kv_reg_r[
                                                vc * 64 + r * 32 + (u + 2) * 8 + v
                                            ]
                                T.lma_wait()
                                T.barrier_arrive(bar_vr1_ready)
                                T.barrier_wait(bar_vr1_ready, v_phase[0] & 1)

                                T.gemm(
                                    scores_shared[vc, :, :],
                                    v_shared_1,
                                    acc_o_r_1,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    wg_wait=-1,
                                )
                                T.wait_wgmma(0)
                                T.barrier_arrive(bar_vr1_free)
                                v_phase[0] = v_phase[0] ^ 1
                            phase_count[0] = phase_count[0] ^ 1

                    T.barrier_wait(bar_final, logical_phase[0] & 1)
                    for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                        acc_o_r_0[h_i, d_i] *= sum_exp_inv_shared[h_i]
                        acc_o_r_1[h_i, d_i] *= sum_exp_inv_shared[h_i]

                    T.barrier_arrive(bar_final_free)
                    T.copy(
                        acc_o_r_0,
                        output[s_i, h0:h1, dim_qk // 2 : dim_qk // 2 + dim_qk // 4],
                    )
                    T.copy(
                        acc_o_r_1,
                        output[s_i, h0:h1, dim_qk // 2 + dim_qk // 4 : dim_qk],
                    )
                else:
                    kperm_mask_local = T.alloc_local([block_i // 16], "bool")
                    kperm_indices_local = T.alloc_local([block_i // 16], indices_dtype)
                    topk_len_local = T.alloc_local([1], indices_dtype)
                    producer_ldg_tx = (tid - 512) % 8
                    producer_ldg_ty = (tid - 512) // 8
                    if has_topk_length:
                        topk_len_local[0] = topk_length[s_i]
                    else:
                        topk_len_local[0] = topk

                    if dynamic_total_blocks > 0:
                        T.barrier_arrive(bar_producer_protect)
                        for r in index_loop(block_i // 16, explicit=True):
                            token_pos = (
                                ((r * 16 + producer_ldg_ty) // 64) * 64
                                + ((r * 16 + producer_ldg_ty) % 64) % 8 * 8
                                + ((r * 16 + producer_ldg_ty) % 64) // 8
                            )
                            kperm_indices_local[r] = indices[s_i, g_i, token_pos]

                        for r in index_loop(block_i // 16, explicit=True):
                            token_pos = (
                                ((r * 16 + producer_ldg_ty) // 64) * 64
                                + ((r * 16 + producer_ldg_ty) % 64) % 8 * 8
                                + ((r * 16 + producer_ldg_ty) % 64) // 8
                            )
                            kperm_mask_local[r] = (
                                T.Cast("uint32", kperm_indices_local[r])
                                < T.Cast("uint32", seq_len_kv)
                            ) and (token_pos < topk_len_local[0])
                            kperm_indices_local[r] = T.if_then_else(
                                kperm_mask_local[r], kperm_indices_local[r], 0
                            )

                        T.barrier_wait(bar_kv0_free, (phase_count[0] & 1) ^ 1)
                        T.annotate_layout(
                            {
                                kv_shared[
                                    0, :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared[0, :, :],
                                    k_major=True,
                                    swizzle_granularity=16,
                                ),
                                kv_shared[
                                    1, :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared[1, :, :],
                                    k_major=True,
                                    swizzle_granularity=16,
                                ),
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        load_kv_half(
                            kv,
                            kv_shared,
                            kperm_indices_local,
                            g_i,
                            0,
                            producer_ldg_tx,
                            producer_ldg_ty,
                            kv_robust_desc,
                        )
                        T.ldlms_crossbb_commit_0()
                        for r in index_loop(block_i // 16):
                            is_kv_valid[
                                ((r * 16 + producer_ldg_ty) // 64) * 64
                                + ((r * 16 + producer_ldg_ty) % 64) % 8 * 8
                                + ((r * 16 + producer_ldg_ty) % 64) // 8
                            ] = kperm_mask_local[r]
                        T.lma_wait()

                        T.barrier_wait(bar_kv1_free, (phase_count[0] & 1) ^ 1)
                        T.annotate_layout(
                            {
                                kv_shared[
                                    qk_panels // 2, :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared[qk_panels // 2, :, :],
                                    k_major=True,
                                    swizzle_granularity=16,
                                ),
                                kv_shared[
                                    qk_panels // 2 + 1, :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared[qk_panels // 2 + 1, :, :],
                                    k_major=True,
                                    swizzle_granularity=16,
                                ),
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        load_kv_half(
                            kv,
                            kv_shared,
                            kperm_indices_local,
                            g_i,
                            qk_panels // 2,
                            producer_ldg_tx,
                            producer_ldg_ty,
                            kv_robust_desc,
                        )
                        load_kv_rope(
                            kv,
                            k_rope_shared,
                            kperm_indices_local,
                            g_i,
                            producer_ldg_tx,
                            producer_ldg_ty,
                            kv_robust_desc,
                        )
                        T.ldlms_crossbb_commit_1()

                        for i_i in T.serial(
                            1,
                            dynamic_total_blocks if has_topk_length else topk_blocks,
                        ):
                            T.ldlms_crossbb_wait_0()
                            T.barrier_arrive(bar_kv0_ready)

                            T.ldlms_crossbb_wait_1()
                            T.barrier_arrive(bar_kv1_ready)
                            T.barrier_wait(bar_producer_protect, phase_count[0] & 1)
                            phase_count[0] = phase_count[0] ^ 1
                            T.barrier_arrive(bar_producer_protect)

                            for r in index_loop(block_i // 16, explicit=True):
                                token_pos = (
                                    i_i * block_i
                                    + ((r * 16 + producer_ldg_ty) // 64) * 64
                                    + ((r * 16 + producer_ldg_ty) % 64) % 8 * 8
                                    + ((r * 16 + producer_ldg_ty) % 64) // 8
                                )
                                kperm_indices_local[r] = indices[s_i, g_i, token_pos]

                            for r in index_loop(block_i // 16, explicit=True):
                                token_pos = (
                                    i_i * block_i
                                    + ((r * 16 + producer_ldg_ty) // 64) * 64
                                    + ((r * 16 + producer_ldg_ty) % 64) % 8 * 8
                                    + ((r * 16 + producer_ldg_ty) % 64) // 8
                                )
                                kperm_mask_local[r] = (
                                    T.Cast("uint32", kperm_indices_local[r])
                                    < T.Cast("uint32", seq_len_kv)
                                ) and (token_pos < topk_len_local[0])
                                kperm_indices_local[r] = T.if_then_else(
                                    kperm_mask_local[r], kperm_indices_local[r], 0
                                )

                            T.barrier_wait(bar_kv0_free, (phase_count[0] & 1) ^ 1)
                            load_kv_half(
                                kv,
                                kv_shared,
                                kperm_indices_local,
                                g_i,
                                0,
                                producer_ldg_tx,
                                producer_ldg_ty,
                                kv_robust_desc,
                            )
                            T.ldlms_crossbb_commit_0()
                            for r in index_loop(block_i // 16):
                                is_kv_valid[
                                    ((r * 16 + producer_ldg_ty) // 64) * 64
                                    + ((r * 16 + producer_ldg_ty) % 64) % 8 * 8
                                    + ((r * 16 + producer_ldg_ty) % 64) // 8
                                ] = kperm_mask_local[r]
                            T.lma_wait()

                            T.barrier_wait(bar_kv1_free, (phase_count[0] & 1) ^ 1)
                            load_kv_half(
                                kv,
                                kv_shared,
                                kperm_indices_local,
                                g_i,
                                qk_panels // 2,
                                producer_ldg_tx,
                                producer_ldg_ty,
                                kv_robust_desc,
                            )
                            load_kv_rope(
                                kv,
                                k_rope_shared,
                                kperm_indices_local,
                                g_i,
                                producer_ldg_tx,
                                producer_ldg_ty,
                                kv_robust_desc,
                            )
                            T.ldlms_crossbb_commit_1()

                        T.ldlms_crossbb_wait_0()
                        T.barrier_arrive(bar_kv0_ready)
                        T.ldlms_crossbb_wait_1()
                        T.barrier_arrive(bar_kv1_ready)
                        T.barrier_wait(bar_producer_protect, phase_count[0] & 1)
                        phase_count[0] = phase_count[0] ^ 1
                    if is_persistence:
                        next_logical_bx = logical_bx + persistent_blocks
                        if next_logical_bx < seq_len * head_repeats:
                            T.barrier_wait(bar_q_free, logical_phase[0] & 1)
                            next_s_i = (
                                next_logical_bx
                                if head_repeats == 1
                                else next_logical_bx // head_repeats
                            )
                            next_h0 = by * padded_head_kv + (
                                0
                                if head_repeats == 1
                                else (next_logical_bx % head_repeats) * 64
                            )
                            next_h1 = next_h0 + heads_per_block
                            if not use_rs_qk:
                                load_q_panels(
                                    q,
                                    q_shared,
                                    next_s_i,
                                    next_h0,
                                    next_h1,
                                    q_robust_desc,
                                    (tid - 512) % 8,
                                    (tid - 512) // 8,
                                )
                                T.ldlms_crossbb_commit_0()
                                T.ldlms_crossbb_wait_0()
                            T.barrier_arrive(bar_q_ready)
                if is_persistence:
                    logical_phase[0] = logical_phase[0] ^ 1
                    logical_bx += persistent_blocks
                else:
                    logical_bx = seq_len * head_repeats

    @T.prim_func
    def dsa_prefill(
        q: q_type,
        kv: kv_type,
        indices: indices_type,
        topk_length: T.Tensor([seq_len], indices_dtype),
        attn_sink: T.Tensor([num_heads], accum_dtype),
        output: T.Tensor(o_shape, dtype),
        max_logits_out: T.Tensor(max_logits_shape, accum_dtype),
        lse: T.Tensor(lse_shape, accum_dtype),
        q_cosize: T.int64,
        kv_cosize: T.int64,
    ):
        dsa_prefill_body(
            q,
            kv,
            indices,
            topk_length,
            attn_sink,
            output,
            max_logits_out,
            lse,
            sm_scale,
            persistent_blocks,
            q_cosize,
            kv_cosize,
        )

    return dsa_prefill


def _select_block_i(heads: int, topk: int) -> int:
    # block_i==64 wins everywhere with the four-chunk V staging path:
    # H64 8k 174.1T / 64k 182.0T vs bi128 173.8T / 181.2T, and the
    # e2e overhead of the chunked path disappears (13.2ms -> 6.7ms at 8k).
    return 64


def sparse_mla_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run PH1S sparse MLA prefill using a unified V32 BF16 KV cache."""
    assert q.dtype == torch.bfloat16
    assert kv.dtype == torch.bfloat16
    assert indices.dtype == torch.int32
    assert q.ndim == 3 and q.shape[-1] == QK_HEAD_DIM
    assert kv.ndim == 3 and kv.shape[-1] == QK_HEAD_DIM

    q, q_shape = prepare_sparse_mla_strided_tensor("q", q, multiple=8)
    kv, kv_shape = prepare_sparse_mla_strided_tensor("kv", kv, multiple=8)
    indices, indices_shape = prepare_sparse_mla_strided_tensor(
        "indices", indices, multiple=8
    )
    seq_len, num_heads, _ = q_shape
    _, kv_group, _ = kv_shape
    assert indices_shape[:2] == (seq_len, kv_group)
    topk = indices_shape[-1]

    kernel_factory = jit_for_tensor_addressing(
        sparse_mla_prefill_kernel, q, kv, indices
    )
    kernel = kernel_factory(
        num_heads,
        topk,
        kv_group=kv_group,
        block_i=_select_block_i(num_heads, topk),
        sm_scale=sm_scale,
        has_topk_length=False,
        persistent_blocks=resolve_num_mps(q.device),
        use_int64_cosize=kernel_factory is not sparse_mla_prefill_kernel,
    )
    topk_length_arg = indices.as_strided((seq_len,), (1,))
    attn_sink_arg = kv.view(torch.float32).as_strided((num_heads,), (1,))
    # cosize is computed on the host and passed in as a scalar argument.
    q_cosize = cosize(q.shape, q.stride())
    kv_cosize = cosize(kv.shape, kv.stride())
    output, _, lse = kernel(
        q,
        kv,
        indices,
        topk_length_arg,
        attn_sink_arg,
        q_cosize,
        kv_cosize,
    )
    return output, lse


def tilelang_sparse_mla_prefill_fwd_interface(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float | None = None,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    return_p_sum: bool = False,
    d_v: int = VALUE_HEAD_DIM,
    threads: int = THREADS,
    verbose: bool = False,
    return_max_logits: bool = False,
    is_persistence: bool = True,
    persistent_blocks: int | None = None,
    block_i: int | None = None,
    use_wide_head_tile: bool | None = None,
) -> tuple[torch.Tensor, ...]:
    """Run the V4-schedule (kimi) PH1S V32 BF16 prefill kernel for Mate."""
    assert not return_p_sum, "This kernel file is for fwd only"
    assert d_v == VALUE_HEAD_DIM
    assert threads == THREADS
    assert q.dtype == torch.bfloat16 and q.ndim == 3
    assert kv.dtype == torch.bfloat16 and kv.ndim == 3
    assert indices.dtype == torch.int32 and indices.ndim == 3

    q, q_shape = prepare_sparse_mla_strided_tensor("q", q, multiple=8)
    kv, kv_shape = prepare_sparse_mla_strided_tensor("kv", kv, multiple=8)
    indices, indices_shape = prepare_sparse_mla_strided_tensor(
        "indices", indices, multiple=8
    )
    seq_len, num_heads, dim_qk = q_shape
    _, kv_group, dim_kv = kv_shape
    assert dim_qk == QK_HEAD_DIM and dim_kv == QK_HEAD_DIM
    assert kv_group == 1
    assert indices_shape[:2] == (seq_len, kv_group)
    topk = int(indices_shape[-1])
    assert topk > 0 and topk % BLOCK_TOPK == 0

    has_topk_length = topk_length is not None
    if topk_length is None:
        topk_length_arg = indices.as_strided((seq_len,), (1,))
    else:
        assert topk_length.dtype == torch.int32
        assert topk_length.shape == (seq_len,)
        assert topk_length.device == q.device
        topk_length_arg = topk_length.contiguous()

    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink_arg = kv.view(torch.float32).as_strided((num_heads,), (1,))
    else:
        assert attn_sink.dtype == torch.float32
        assert attn_sink.shape == (num_heads,)
        assert attn_sink.device == q.device
        attn_sink_arg = attn_sink.contiguous()

    if is_persistence:
        persistent_blocks = resolve_num_mps(q.device, persistent_blocks)
    else:
        persistent_blocks = 0
    if block_i is None:
        block_i = _select_block_i(num_heads, topk)

    kernel_factory = jit_for_tensor_addressing(
        sparse_mla_prefill_kernel, q, kv, indices
    )
    kernel = kernel_factory(
        num_heads,
        topk,
        kv_group=kv_group,
        block_i=block_i,
        threads=threads,
        has_attn_sink=has_attn_sink,
        has_topk_length=has_topk_length,
        is_persistence=is_persistence,
        persistent_blocks=persistent_blocks,
        sm_scale=sm_scale,
        use_int64_cosize=kernel_factory is not sparse_mla_prefill_kernel,
        use_wide_head_tile=use_wide_head_tile,
    )
    if verbose:
        kernel.show_source()
    raise_complete_if_dry_run()
    # cosize is computed on the host and passed in as a scalar argument.
    q_cosize = cosize(q.shape, q.stride())
    kv_cosize = cosize(kv.shape, kv.stride())
    output, max_logits, lse = kernel(
        q,
        kv,
        indices,
        topk_length_arg,
        attn_sink_arg,
        q_cosize,
        kv_cosize,
    )
    if return_max_logits:
        return output, max_logits, lse
    return output, lse
