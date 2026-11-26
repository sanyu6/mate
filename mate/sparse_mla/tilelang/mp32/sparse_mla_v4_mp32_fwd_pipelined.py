# ruff: noqa
"""MP32 V4 sparse MLA prefill with register-balanced PV ownership.

The default implementation keeps the generic pipeline's proven
Q-panel prefetch, split KV-ready/free barriers, double-buffer ownership, dynamic
stride ABI, and optional topk-length/attention-sink handling. The JIT target is
fixed to MP32 here so this is an explicit optimized MP32 artifact, while the
native MP32 implementation remains in sparse_mla_v4_mp32_fwd.py.

The H128/B64, D512 shared-Q path uses asymmetric PV ownership:
the 896-thread block is split into P=128, C0=256, and C1=512.  C0 owns D0:128
and C1 owns D128:512.  The C1 local K/V staging uses one 64x128 BF16 panel per
thread group (16 BF16 values per thread per panel), and the three PV panels are
consumed as V1, V0, V1.

The H64/B128 path uses a 768-thread block (C0=256, C1=256, P=256).
QK and PV share a three-slot packed KV ring, avoiding intermediate V repacking.
The first two Q panels reside in shared memory; the remaining panels use RS QK.
Copy groups 0/1 overlap the two KV halves, with explicit reader release before
slot reuse. Explicit 640-thread calls and other geometries keep the original
schedule.
"""

from typing import Any

import torch
import tilelang
from tilelang import language as T

from ....utils import cosize
from ....mate_runtime import MP32_ARCH, get_musa_arch, resolve_num_mps
from ..sparse_mla_prefill_common import (
    SPARSE_PREFILL_MP32_OPTIMIZED_COMPILE_FLAGS,
    SPARSE_PREFILL_PASS_CONFIGS,
    prepare_sparse_mla_strided_tensor,
    validate_prefill_attn_sink,
    validate_token_lengths,
)
from ..sparse_mla_index_type import jit_for_tensor_addressing
from ....execution_context import raise_complete_if_dry_run


# tilelang.disable_cache()

TARGET = {"kind": "musa", "arch": "mp_32"}


V4_SPARSE_PREFILL_PASS_CONFIGS = {
    **SPARSE_PREFILL_PASS_CONFIGS,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
}


@tilelang.jit(
    target=TARGET,
    out_idx=[3, 4, 5],
    pass_configs=V4_SPARSE_PREFILL_PASS_CONFIGS,
    compile_flags=SPARSE_PREFILL_MP32_OPTIMIZED_COMPILE_FLAGS,
)
def sparse_attention_fwd_kernel_v4_mp32_optimized(
    num_heads,
    dim,
    *,
    kv_group=1,
    is_causal=True,
    block_i=64,
    threads=640,
    has_attn_sink=False,
    has_topk_length=False,
    is_persistence=True,
    use_rs_qk=False,
    use_kv_source_precompute=False,
    asymmetric_pv=None,
):
    assert dim == tilelang.math.next_power_of_2(dim), (
        f"haven't check padding correctness yet, dim={dim}"
    )
    direct_kv_ring = threads == 768
    if direct_kv_ring:
        assert num_heads == 64 and dim == 512 and block_i == 128
        assert kv_group == 1 and asymmetric_pv is True and use_rs_qk
    if asymmetric_pv is None:
        asymmetric_pv = (
            threads == 896
            and num_heads == 128
            and dim == 512
            and block_i == 64
            and kv_group == 1
            and not use_rs_qk
        )
    asymmetric_pv = bool(asymmetric_pv)
    if asymmetric_pv:
        # C0 owns QK/softmax and D0:128; C1 owns D128:512. Other geometries
        # retain the 640-thread ownership schedule.
        assert num_heads in (64, 128) and dim == 512
        assert (num_heads, block_i, threads) in (
            (128, 64, 896),
            (64, 128, 640),
            (64, 128, 768),
        )
        assert kv_group == 1
    topk = T.dynamic("topk")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    q_stride_s = T.dynamic("q_stride_s")
    q_stride_h = T.dynamic("q_stride_h")
    kv_stride_s = T.dynamic("kv_stride_s")
    kv_stride_g = T.dynamic("kv_stride_g")
    indices_stride_s = T.dynamic("indices_stride_s")
    indices_stride_g = T.dynamic("indices_stride_g")

    head_kv = num_heads // kv_group
    q_shape = [seq_len, num_heads, dim]
    kv_shape = [seq_len_kv, kv_group, dim]
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

    # H128 can use one wider head tile; this removes the second persistent
    # head group and its duplicate Q/KV/softmax scheduling.
    if num_heads == 128 and padded_head_kv == 128:
        head_repeats = 1
        heads_per_block = 128
    else:
        heads_per_block = padded_head_kv if head_repeats == 1 else 64

    # Case03's highest-performing ASM uses register-source QK.  Start with
    # the shape that has the same M64N128K128 QK tile as the current V4 path;
    # H128 needs a separate M64 split before it can safely use RS operands.
    # Case03's resident-Q path is the 64-token score tile.  Keep the
    # block_i=128 probe available for history, but only block_i=64 has the
    # matching M64x64x128 QK contract.
    use_rs_qk = bool(use_rs_qk and heads_per_block == 64 and block_i in (64, 128))
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
    asym_b128 = asymmetric_pv and block_i == 128
    # Resident Q frees enough LM for a separate K3/PV3 operand. Keep K2
    # in registers but publish K3 directly, halving C1's KV live range.
    private_v2 = asym_b128 and use_rs_qk
    consumer1_threads = 512 if asymmetric_pv and heads_per_block == 128 else 256
    use_index_vectorization = heads_per_block == 128 and block_i == 64

    def index_loop(extent, explicit=True):
        if use_index_vectorization:
            return T.vectorized(extent)
        return T.unroll(extent, explicit=explicit)

    @T.macro
    def load_kv_half(
        kv,
        kv_shared,
        kperm_indices_local,
        kv_source_local,
        g_i,
        panel_offset,
        producer_ldg_tx,
        producer_ldg_ty,
        kv_robust_desc,
    ):
        for r in T.unroll(block_i // 16):
            for panel in T.unroll(qk_panels // 2):
                for u in T.unroll(qk_panel // 64):
                    if use_kv_source_precompute:
                        T.call_intrin(
                            "handle",
                            T.tirx.op.Op.get("tl.musa_cp_async_robust_2d"),
                            T.address_of(
                                kv_shared[
                                    panel + panel_offset,
                                    r * 16 + producer_ldg_ty,
                                    64 * u + producer_ldg_tx * 8,
                                ]
                            ),
                            T.address_of(
                                kv[
                                    kperm_indices_local[r],
                                    g_i,
                                    (panel + panel_offset) * qk_panel
                                    + 64 * u
                                    + producer_ldg_tx * 8,
                                ]
                            ),
                            16,
                            1,
                            16,
                            kv_robust_desc,
                            0,
                        )
                    else:
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
    def load_q_panels(
        q,
        q_shared,
        s_i,
        h0,
        producer_ldg_tx,
        producer_ldg_ty,
        q_robust_desc,
    ):
        # Case03 ownership probe: the producer gathers Q into the shared
        # panels, so consumers only wait on q_ready before QK.  This first
        # version is intentionally serialized with the KV producer; later
        # tuning can overlap it once correctness is established.
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
            q_shared_raw = T.alloc_shared(
                [qk_panels, heads_per_block, qk_panel], T.uint16
            )
            q_shared = T.Buffer(
                [qk_panels, heads_per_block, qk_panel],
                dtype,
                data=q_shared_raw.data,
            )
            kv_shared = T.alloc_shared([qk_panels, block_i, qk_panel], dtype)
            v_shared_0 = T.alloc_shared([64, dim_qk // 4], dtype)
            v_shared_1 = T.alloc_shared([64, dim_qk // 4], dtype)
            v_shared_2 = (
                T.alloc_shared([v_chunks, 64, dim_qk // 4], dtype)
                if private_v2
                else None
            )
            if private_v2:
                T.annotate_layout(
                    {
                        v_shared_2: tilelang.layout.make_sqmma_swizzled_layout(
                            v_shared_2, continuity=128, k_major=False
                        )
                    }
                )
            # Keep V shared layout inferred by the consumer.  The explicit
            # Case01 V layout causes MP32 reannotation conflicts in the fused
            # online-softmax path and regresses H128.
            case01_v0_layout = {}
            case01_v1_layout = {}
            c1_square_v0_layout = {}
            if asymmetric_pv:
                case01_v0_layout = {
                    v_shared_0: tilelang.layout.make_sqmma_swizzled_layout(
                        v_shared_0, continuity=128, k_major=False
                    )
                }
                c1_square_v0_layout = {
                    v_shared_0: tilelang.layout.make_sqmma_swizzled_layout(
                        v_shared_0, continuity=64, k_major=False
                    )
                }

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
            bar_kv1_read_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            # In the asymmetric candidate C1 reads panel 1 before C0 can
            # release KV0, so producer reuse waits for all C0+C1 lanes.
            bar_kv0_free = T.alloc_barrier(
                arrive_count=consumer0_threads
                + (consumer1_threads if asymmetric_pv else 0)
            )
            # KV1 is still consumed by both QK (C0) and the C1 panel cache.
            bar_kv1_free = T.alloc_barrier(
                arrive_count=consumer0_threads + consumer1_threads
            )

            bar_vl_pair_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_vr0_ready = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_vr1_ready = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_vl0_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_vl1_free = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_vr0_free = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_vr1_free = T.alloc_barrier(arrive_count=consumer1_threads)

            bar_p_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_v2_ready = (
                T.alloc_barrier(arrive_count=consumer1_threads) if private_v2 else None
            )
            bar_p_free = (
                T.alloc_barrier(arrive_count=consumer1_threads) if asym_b128 else None
            )
            bar_alpha_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_final = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_final_free = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_producer_protect = T.alloc_barrier(arrive_count=producer_threads)

            kv_robust_desc = T.make_robust_desc(
                T.address_of(kv[0, 0, 0]),
                kv_cosize * dtype_bytes,
            )
            q_robust_desc = T.make_robust_desc(
                T.address_of(q[0, 0, 0]),
                q_cosize * dtype_bytes,
            )
            T.sync_threads()

            tid = T.get_thread_binding()
            logical_bx = T.alloc_var(T.int32)
            phase_count = T.alloc_local([1], T.int32)
            logical_phase = T.alloc_local([1], T.int32)
            q_phase = T.alloc_local([1], T.int32)
            producer_q_first = T.alloc_local([1], T.int32)
            T.fill(phase_count, 0)
            T.fill(logical_phase, 0)
            T.fill(q_phase, 0)
            T.fill(producer_q_first, 1)
            logical_bx = bx
            if tid >= 256 and tid < 256 + consumer1_threads:
                if asym_b128:
                    T.barrier_arrive(bar_p_free)
                T.barrier_arrive(bar_vr0_free)
                T.barrier_arrive(bar_vr1_free)

            # Fixed roles retain independent persistent loops.  Keeping the
            # work-loop merge outside the role bodies shortens pointer and
            # row-state live ranges without changing shared-buffer ownership.
            if tid < 256:
                while logical_bx < seq_len * head_repeats:
                    if is_persistence:
                        T.call_extern("void", "__musa_loop_transparent_outermost")
                    g_i = by
                    s_i = (
                        logical_bx
                        if head_repeats == 1
                        else (logical_bx // head_repeats)
                    )
                    h0 = g_i * padded_head_kv + (
                        0 if head_repeats == 1 else (logical_bx % head_repeats) * 64
                    )
                    h1 = h0 + heads_per_block
                    dynamic_total_blocks = T.alloc_var(T.int32)
                    if has_topk_length:
                        dynamic_total_blocks = T.ceildiv(topk_length[s_i], block_i)
                    else:
                        dynamic_total_blocks = T.ceildiv(topk, block_i)

                    # Derive parity from the tile index and the row's prefix.
                    # No shape constants: dense rows all have the same tile count,
                    # and optional lengths carry just one prefix bit across rows.
                    tile_base_phase = T.alloc_var(T.int32)
                    tile_base_phase = (
                        phase_count[0]
                        if has_topk_length
                        else (logical_phase[0] & dynamic_total_blocks)
                    ) & 1

                    sumexp = T.alloc_fragment([heads_per_block], accum_dtype)
                    sumexp_i = T.alloc_fragment([heads_per_block], accum_dtype)
                    sumexp_inv = T.alloc_fragment([heads_per_block], accum_dtype)
                    alpha_local = T.alloc_fragment([heads_per_block], accum_dtype)
                    m_i = T.alloc_fragment([heads_per_block], accum_dtype)
                    m_i_prev = T.alloc_fragment([heads_per_block], accum_dtype)
                    max_logits = T.alloc_fragment([heads_per_block], accum_dtype)
                    acc_s = T.alloc_fragment([heads_per_block, block_i], accum_dtype)
                    row_max_pairs = (
                        T.alloc_fragment([heads_per_block, 4, 8], accum_dtype)
                        if heads_per_block == 128 and block_i == 64
                        else None
                    )
                    row_max_partial = (
                        T.alloc_fragment([heads_per_block, 4], accum_dtype)
                        if heads_per_block == 128 and block_i == 64
                        else None
                    )
                    direct_score_cast = heads_per_block == 128 and block_i == 64
                    acc_s_cast = (
                        None
                        if direct_score_cast
                        else T.alloc_fragment([heads_per_block, block_i], dtype)
                    )
                    if heads_per_block == 128 and block_i == 64:
                        T.annotate_layout(
                            {
                                acc_s: T.Fragment(
                                    (heads_per_block, block_i),
                                    forward_thread_fn=lambda fr, fc: (
                                        (fr // 64) * 128 + (fr % 32) * 4 + fc % 4
                                    ),
                                    forward_index_fn=lambda fr, fc: (
                                        (fr % 64 // 32) * (block_i // 4) + fc // 4
                                    ),
                                ),
                                m_i: T.Fragment(
                                    (heads_per_block,),
                                    replicate=4,
                                    forward_thread_fn=lambda fr, rep: (
                                        (fr // 64) * 128 + (fr % 32) * 4 + rep
                                    ),
                                    forward_index_fn=lambda fr: fr % 64 // 32,
                                ),
                                row_max_pairs: T.Fragment(
                                    (heads_per_block, 4, 8),
                                    forward_thread_fn=lambda fr, fc, fp: (
                                        (fr // 64) * 128 + (fr % 32) * 4 + fc
                                    ),
                                    forward_index_fn=lambda fr, fc, fp: (
                                        (fr % 64 // 32) * 8 + fp
                                    ),
                                ),
                                row_max_partial: T.Fragment(
                                    (heads_per_block, 4),
                                    forward_thread_fn=lambda fr, fc: (
                                        (fr // 64) * 128 + (fr % 32) * 4 + fc
                                    ),
                                    forward_index_fn=lambda fr, fc: fr % 64 // 32,
                                ),
                            }
                        )
                    if use_rs_qk:
                        q_reg_0 = T.alloc_fragment([heads_per_block, qk_panel], dtype)
                        q_reg_1 = T.alloc_fragment([heads_per_block, qk_panel], dtype)
                        q_reg_2 = T.alloc_fragment([heads_per_block, qk_panel], dtype)
                        q_reg_3 = T.alloc_fragment([heads_per_block, qk_panel], dtype)
                    acc_o_l_0 = T.alloc_fragment(
                        [heads_per_block, dim_qk // 4], accum_dtype
                    )
                    # Asymmetric C0 owns only D0:128.  Avoid allocating the
                    # second 128-column accumulator in the candidate.
                    acc_o_l_1 = (
                        None
                        if asymmetric_pv
                        else T.alloc_fragment(
                            [heads_per_block, dim_qk // 4], accum_dtype
                        )
                    )
                    kv_reg_l = T.alloc_local(
                        [32 if asymmetric_pv and block_i == 64 else block_i], dtype
                    )
                    score_pack = T.alloc_local([8], dtype)
                    consumer0_ldg_tx = tid % 8
                    consumer0_ldg_ty = tid // 8
                    T.fill(sumexp, 0)
                    T.fill(m_i, -(2**30))
                    T.fill(acc_o_l_0, 0)
                    if not asymmetric_pv:
                        T.fill(acc_o_l_1, 0)
                    if not use_rs_qk and dynamic_total_blocks > 0:
                        T.barrier_wait(
                            bar_q_ready,
                            (q_phase[0] if has_topk_length else logical_phase[0]) & 1,
                        )
                    if asym_b128 and use_rs_qk and dynamic_total_blocks > 0:
                        # Q is invariant across this query's KV tiles.
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
                    for i_i in range(dynamic_total_blocks):
                        T.barrier_wait(bar_kv0_ready, (tile_base_phase ^ i_i) & 1)

                        for h_i, bi_i in T.Parallel(heads_per_block, block_i):
                            acc_s[h_i, bi_i] = T.if_then_else(
                                T.Cast(
                                    "bool",
                                    is_kv_valid[
                                        (bi_i // 64) * 64
                                        + (bi_i % 64) % 8 * 8
                                        + (bi_i % 64) // 8
                                    ],
                                ),
                                0,
                                -(2**30),
                            )

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
                        if use_rs_qk:
                            if not asym_b128:
                                T.copy(
                                    q[s_i, h0:h1, 0:qk_panel],
                                    q_reg_0,
                                )
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
                        if asymmetric_pv and block_i == 64:
                            # C0 caches only K0 (panel 0) for its D0:128 PV.
                            # The row inverse permutation is the same one used
                            # by the baseline SQMMA layout.
                            for r in T.unroll(2):
                                row = consumer0_ldg_ty + r * 32
                                row_inv = (
                                    (row // 64) * 64
                                    + (row % 64) % 8 * 8
                                    + (row % 64) // 8
                                )
                                for u in T.unroll(2):
                                    for v in T.vectorized(8):
                                        kv_reg_l[r * 16 + u * 8 + v] = kv_shared[
                                            0,
                                            row_inv,
                                            64 * u + consumer0_ldg_tx * 8 + v,
                                        ]
                        else:
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

                        if use_rs_qk:
                            if not asym_b128:
                                T.copy(
                                    q[s_i, h0:h1, qk_panel : 2 * qk_panel],
                                    q_reg_1,
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
                                q_shared[1, :, :],
                                kv_shared[1, :, :],
                                acc_s,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                wg_wait=-1,
                            )

                        if block_i == 64:
                            T.barrier_wait(bar_vr0_free, (tile_base_phase ^ i_i) & 1)
                        elif not asymmetric_pv or asym_b128:
                            T.barrier_wait(bar_vr0_free, (i_i * v_chunks) & 1)
                        if asymmetric_pv and block_i == 64:
                            T.annotate_layout(
                                case01_v0_layout,
                                allow_reannotation=True,
                            )
                            for r in T.unroll(2):
                                for u in T.unroll(2):
                                    for v in T.vectorized(8):
                                        v_shared_0[
                                            r * 32 + consumer0_ldg_ty,
                                            64 * u + consumer0_ldg_tx * 8 + v,
                                        ] = kv_reg_l[r * 16 + u * 8 + v]
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
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        T.lma_wait()
                        T.barrier_arrive(bar_kv0_free)

                        T.barrier_wait(bar_kv1_ready, (tile_base_phase ^ i_i) & 1)
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
                            if not asym_b128:
                                T.copy(
                                    q[
                                        s_i,
                                        h0:h1,
                                        (qk_panels // 2) * qk_panel : (
                                            qk_panels // 2 + 1
                                        )
                                        * qk_panel,
                                    ],
                                    q_reg_2,
                                )
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
                        if block_i == 64:
                            # Retain this wait even when C0 no longer stages V1:
                            # it also fences C1's previous P/alpha reads.
                            T.barrier_wait(bar_vr1_free, (tile_base_phase ^ i_i) & 1)
                        elif not asymmetric_pv:
                            T.barrier_wait(bar_vr1_free, (i_i * v_chunks) & 1)
                        if not asymmetric_pv:
                            for v in T.vectorized(8):
                                v_shared_1[
                                    consumer0_ldg_ty,
                                    consumer0_ldg_tx * 8 + v,
                                ] = kv_reg_l[16 + v]
                        if use_rs_qk:
                            if not asym_b128:
                                T.copy(
                                    q[
                                        s_i,
                                        h0:h1,
                                        (qk_panels // 2 + 1) * qk_panel : (
                                            qk_panels // 2 + 2
                                        )
                                        * qk_panel,
                                    ],
                                    q_reg_3,
                                )
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
                        if not asymmetric_pv:
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
                        T.warpgroup_commit_batch()
                        T.lma_wait()
                        T.barrier_arrive(bar_vl_pair_ready)
                        T.copy(m_i, m_i_prev)
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_kv1_free)
                        if heads_per_block == 128 and block_i == 64:
                            # Separate projections keep one acc_s index map per
                            # ParallelOp while expressing the native max tree.
                            for h_i, lane, pair in T.Parallel(heads_per_block, 4, 8):
                                slot = (pair % 4) // 2 + (pair % 2) * 2
                                row_max_pairs[h_i, lane, pair] = acc_s[
                                    h_i, lane + 4 * ((pair // 4) * 8 + slot)
                                ]
                            for h_i, lane, pair in T.Parallel(heads_per_block, 4, 8):
                                slot = (pair % 4) // 2 + (pair % 2) * 2
                                other = T.if_then_else(slot == 3, 4, slot + 5)
                                row_max_pairs[h_i, lane, pair] = T.max(
                                    row_max_pairs[h_i, lane, pair],
                                    acc_s[h_i, lane + 4 * ((pair // 4) * 8 + other)],
                                )
                            T.reduce_max(
                                row_max_pairs, row_max_partial, dim=2, clear=True
                            )
                            T.reduce_max(row_max_partial, m_i, dim=1, clear=False)
                        else:
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

                        if asym_b128:
                            # KV1 is released before C1 PV now; it no longer
                            # implicitly protects P/alpha from the next QK.
                            T.barrier_wait(bar_p_free, (tile_base_phase ^ i_i) & 1)
                        T.copy(alpha_local, alpha_shared)
                        # Complete alpha stores before publishing availability.
                        T.lma_wait()
                        T.barrier_arrive(bar_alpha_ready)
                        if not direct_score_cast:
                            T.copy(acc_s, acc_s_cast)
                        if asymmetric_pv and block_i == 64:
                            # C0 owns only D0:128.  Publish P_READY after
                            # issuing PV0; C1 can consume V1 while C0 finishes
                            # this V0 operation.  V0-free is published only
                            # after the PV operation has completed.
                            for i, t in T.Parallel(heads_per_block, 8):
                                base = t * lanes_per_vec
                                for l in T.vectorized(lanes_per_vec):
                                    scores_shared[i, base + l] = T.Cast(
                                        dtype, acc_s[i, l * 8 + t]
                                    )
                            T.lma_wait()
                            for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                                acc_o_l_0[h_i, d_i] *= alpha_local[h_i]
                            T.annotate_layout(
                                case01_v0_layout,
                                allow_reannotation=True,
                                allow_buffer_region=True,
                            )
                            T.barrier_wait(
                                bar_vl_pair_ready, (tile_base_phase ^ i_i) & 1
                            )
                            T.gemm(
                                scores_shared,
                                v_shared_0,
                                acc_o_l_0,
                                policy=T.GemmWarpPolicy.FullRow,
                                wg_wait=-1,
                            )
                            T.warpgroup_commit_batch()
                            T.sched_boundary()
                            T.barrier_arrive(bar_p_ready)
                            T.reduce_sum(acc_s, sumexp_i, dim=1)
                            for h_i in T.Parallel(heads_per_block):
                                sumexp[h_i] = (
                                    sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                                )
                            T.warpgroup_wait(0)
                            T.barrier_arrive(bar_vl0_free)
                        elif block_i == 64:
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
                                        scores_shared[i, base + l] = T.Cast(
                                            dtype, acc_s[i, l * 8 + t]
                                        )

                            T.lma_wait()
                            for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                                acc_o_l_0[h_i, d_i] *= alpha_local[h_i]

                            T.annotate_layout(
                                case01_v0_layout,
                                allow_reannotation=True,
                                allow_buffer_region=True,
                            )
                            T.barrier_wait(
                                bar_vl_pair_ready, (tile_base_phase ^ i_i) & 1
                            )

                            T.gemm(
                                scores_shared,
                                v_shared_0,
                                acc_o_l_0,
                                policy=T.GemmWarpPolicy.FullRow,
                                wg_wait=-1,
                            )
                            T.warpgroup_commit_batch()
                            T.sched_boundary()
                            T.barrier_arrive(bar_p_ready)
                            if not asym_b128:
                                for h_i, d_i in T.Parallel(
                                    heads_per_block, dim_qk // 4
                                ):
                                    acc_o_l_1[h_i, d_i] *= alpha_local[h_i]

                            T.reduce_sum(acc_s, sumexp_i, dim=1)
                            for h_i in T.Parallel(heads_per_block):
                                sumexp[h_i] = (
                                    sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                                )

                            if not asym_b128:
                                T.annotate_layout(
                                    case01_v1_layout,
                                    allow_reannotation=True,
                                    allow_buffer_region=True,
                                )
                                T.warpgroup_wait(0)
                                T.barrier_arrive(bar_vl0_free)

                                T.gemm(
                                    scores_shared,
                                    v_shared_1,
                                    acc_o_l_1,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    wg_wait=-1,
                                )
                                T.warpgroup_commit_batch()
                                T.warpgroup_wait(0)
                                T.barrier_arrive(bar_vl1_free)
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
                                    T.barrier_wait(
                                        bar_vr0_free, (i_i * v_chunks + vc) & 1
                                    )
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
                                    if asym_b128:
                                        T.lma_wait()
                                    if not asym_b128:
                                        T.barrier_wait(
                                            bar_vr1_free, (i_i * v_chunks + vc) & 1
                                        )
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
                                if asym_b128 and vc > 0:
                                    # vc0 arrives in the common prefix; each
                                    # remaining V0 chunk needs both C0 squads.
                                    T.barrier_arrive(bar_vl_pair_ready)
                                T.barrier_wait(
                                    bar_vl_pair_ready, (i_i * v_chunks + vc) & 1
                                )

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
                                    if not asym_b128:
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
                                if asym_b128:
                                    # C0 owns only V0 in this branch.  Complete
                                    # PV0 before C1 is allowed to overwrite the
                                    # shared V0 tile for K2.
                                    T.warpgroup_wait(0)
                                    T.barrier_arrive(bar_vl0_free)
                                if not asym_b128:
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

                    if not use_rs_qk and dynamic_total_blocks > 0:
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
                    # Complete all shared inverse stores before publishing the
                    # final normalization state to the other consumer group.
                    T.lma_wait()
                    T.barrier_arrive(bar_final)
                    if asymmetric_pv:
                        # Publish scalar statistics before converting/storing O
                        # to shorten the H128 output epilogue's live ranges.
                        T.copy(max_logits, max_logits_out[s_i, h0:h1])
                        for h_i in T.Parallel(heads_per_block):
                            lse[s_i, h0 + h_i] = sumexp[h_i] * 0.6931471805599453
                    if asymmetric_pv:
                        for row_group in T.unroll(heads_per_block // 32):
                            for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                                if h_i // 32 == row_group:
                                    output[s_i, h0 + h_i, d_i] = (
                                        acc_o_l_0[h_i, d_i] * sumexp_inv[h_i]
                                    ).astype(dtype)
                    else:
                        for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                            acc_o_l_0[h_i, d_i] *= sumexp_inv[h_i]
                            if not asymmetric_pv:
                                acc_o_l_1[h_i, d_i] *= sumexp_inv[h_i]
                        T.copy(acc_o_l_0, output[s_i, h0:h1, 0 : dim_qk // 4])
                    if not asymmetric_pv:
                        T.copy(
                            acc_o_l_1,
                            output[s_i, h0:h1, dim_qk // 4 : dim_qk // 2],
                        )
                        T.copy(max_logits, max_logits_out[s_i, h0:h1])
                        for h_i in T.Parallel(heads_per_block):
                            lse[s_i, h0 + h_i] = sumexp[h_i] * 0.6931471805599453
                    if has_topk_length:
                        phase_count[0] = tile_base_phase ^ (dynamic_total_blocks & 1)
                        if not use_rs_qk and dynamic_total_blocks > 0:
                            q_phase[0] = q_phase[0] ^ 1
                    if is_persistence:
                        logical_phase[0] = logical_phase[0] ^ 1
                        logical_bx += persistent_blocks
                    else:
                        logical_bx = seq_len * head_repeats

            elif tid < 256 + consumer1_threads:
                while logical_bx < seq_len * head_repeats:
                    if is_persistence:
                        T.call_extern("void", "__musa_loop_transparent_outermost")
                    g_i = by
                    s_i = (
                        logical_bx
                        if head_repeats == 1
                        else (logical_bx // head_repeats)
                    )
                    h0 = g_i * padded_head_kv + (
                        0 if head_repeats == 1 else (logical_bx % head_repeats) * 64
                    )
                    h1 = h0 + heads_per_block
                    dynamic_total_blocks = T.alloc_var(T.int32)
                    if has_topk_length:
                        dynamic_total_blocks = T.ceildiv(topk_length[s_i], block_i)
                    else:
                        dynamic_total_blocks = T.ceildiv(topk, block_i)

                    # Derive parity from the tile index and the row's prefix.
                    # No shape constants: dense rows all have the same tile count,
                    # and optional lengths carry just one prefix bit across rows.
                    tile_base_phase = T.alloc_var(T.int32)
                    tile_base_phase = (
                        phase_count[0]
                        if has_topk_length
                        else (logical_phase[0] & dynamic_total_blocks)
                    ) & 1

                    acc_o_r_0 = T.alloc_fragment(
                        [heads_per_block, dim_qk // 4], accum_dtype
                    )
                    acc_o_r_1 = T.alloc_fragment(
                        [heads_per_block, dim_qk // 4], accum_dtype
                    )
                    acc_o_r_2 = (
                        T.alloc_fragment([heads_per_block, dim_qk // 4], accum_dtype)
                        if asymmetric_pv
                        else None
                    )
                    # C1 caches panel 1, then panels 2 and 3: three 16-value
                    # vectors per 64-row thread coordinate.
                    # H64/B128 C1 only needs one 64-row PV chunk at a time.
                    # Keep the two panel payloads in a 64-value window and
                    # consume each vchunk before loading the next one.  The
                    # generic path retains its full block_i staging window.
                    k1_reg = T.alloc_local([32], dtype) if asym_b128 else None
                    kv_reg_r = T.alloc_local(
                        [
                            48
                            if asymmetric_pv and block_i == 64
                            else (64 if private_v2 else block_i)
                        ],
                        dtype,
                    )
                    alpha_r = T.alloc_fragment([heads_per_block], accum_dtype)
                    v1_phase = T.alloc_local([1], T.int32)
                    T.fill(v1_phase, 0)
                    T.fill(acc_o_r_0, 0)
                    T.fill(acc_o_r_1, 0)
                    if asymmetric_pv:
                        T.fill(acc_o_r_2, 0)

                    consumer1_ldg_tx = (tid - 256) % 8
                    consumer1_ldg_ty = (tid - 256) // 8
                    if not asymmetric_pv and not use_rs_qk and dynamic_total_blocks > 0:
                        T.barrier_wait(
                            bar_q_ready,
                            (q_phase[0] if has_topk_length else logical_phase[0]) & 1,
                        )
                    if asymmetric_pv and block_i == 64:
                        # C1 reads K1 before releasing KV0, then K2/K3
                        # after KV1_READ_READY. Every lane owns one
                        # inverse-permuted row and 16 BF16 values per panel.
                        for i_i in range(dynamic_total_blocks):
                            phase = (tile_base_phase ^ i_i) & 1
                            T.barrier_wait(bar_kv0_ready, phase)
                            row = consumer1_ldg_ty
                            row_inv = (
                                (row // 64) * 64 + (row % 64) % 8 * 8 + (row % 64) // 8
                            )
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    kv_reg_r[u * 8 + v] = kv_shared[
                                        1,
                                        row_inv,
                                        64 * u + consumer1_ldg_tx * 8 + v,
                                    ]
                            T.lma_wait()
                            T.barrier_arrive(bar_kv0_free)

                            # V1 is private to C1 on this path. Stage K1 before
                            # waiting for K2/K3 or softmax; previous PV3 completion
                            # is still protected by the original free barrier.
                            T.barrier_wait(bar_vr1_free, phase)
                            v1_phase[0] = 0
                            T.annotate_layout(
                                case01_v1_layout,
                                allow_reannotation=True,
                                allow_buffer_region=True,
                            )
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    v_shared_1[
                                        row,
                                        64 * u + consumer1_ldg_tx * 8 + v,
                                    ] = kv_reg_r[u * 8 + v]
                            T.lma_wait()
                            T.barrier_arrive(bar_vr1_ready)

                            T.barrier_wait(bar_kv1_read_ready, phase)
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    kv_reg_r[16 + u * 8 + v] = kv_shared[
                                        qk_panels // 2,
                                        row_inv,
                                        64 * u + consumer1_ldg_tx * 8 + v,
                                    ]
                                    kv_reg_r[32 + u * 8 + v] = kv_shared[
                                        qk_panels // 2 + 1,
                                        row_inv,
                                        64 * u + consumer1_ldg_tx * 8 + v,
                                    ]
                            T.lma_wait()
                            T.barrier_arrive(bar_kv1_free)

                            T.barrier_wait(bar_alpha_ready, phase)
                            for h_i in T.Parallel(heads_per_block):
                                alpha_r[h_i] = alpha_shared[h_i]
                            for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                                acc_o_r_0[h_i, d_i] *= alpha_r[h_i]
                                acc_o_r_1[h_i, d_i] *= alpha_r[h_i]
                                acc_o_r_2[h_i, d_i] *= alpha_r[h_i]
                            T.barrier_wait(bar_p_ready, phase)

                            # D128:256 uses V1 from K1.
                            T.barrier_wait(bar_vr1_ready, v1_phase[0])
                            T.gemm(
                                scores_shared,
                                v_shared_1,
                                acc_o_r_0,
                                policy=T.GemmWarpPolicy.Square,
                                wg_wait=-1,
                            )
                            T.warpgroup_commit_batch()

                            # D256:384 waits for C0's V0 release, then
                            # reuses V0 for panel K2 while PV1 reads disjoint V1.
                            T.barrier_wait(bar_vl0_free, phase)
                            T.annotate_layout(
                                c1_square_v0_layout,
                                allow_reannotation=True,
                                allow_buffer_region=True,
                            )
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    v_shared_0[
                                        row,
                                        64 * u + consumer1_ldg_tx * 8 + v,
                                    ] = kv_reg_r[16 + u * 8 + v]
                            T.lma_wait()
                            # V0 publication does not overwrite V1. PV1 and
                            # PV2 may overlap; their common completion rendezvous
                            # is below, before any K3 overwrite of V1.
                            T.barrier_arrive(bar_vr0_ready)
                            T.barrier_wait(bar_vr0_ready, phase)
                            T.gemm(
                                scores_shared,
                                v_shared_0,
                                acc_o_r_1,
                                policy=T.GemmWarpPolicy.Square,
                                wg_wait=-1,
                            )
                            T.warpgroup_commit_batch()

                            # Finish both committed PV operations, then
                            # rendezvous across all C1 squads before V1 reuse.
                            # This free event is the NEXT tile's V0 credit.
                            T.warpgroup_wait(0)
                            T.barrier_arrive(bar_vr0_free)
                            T.barrier_wait(bar_vr0_free, phase ^ 1)
                            v1_phase[0] = v1_phase[0] ^ 1
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    v_shared_1[
                                        row,
                                        64 * u + consumer1_ldg_tx * 8 + v,
                                    ] = kv_reg_r[32 + u * 8 + v]
                            T.lma_wait()
                            T.barrier_arrive(bar_vr1_ready)
                            T.barrier_wait(bar_vr1_ready, v1_phase[0])
                            T.gemm(
                                scores_shared,
                                v_shared_1,
                                acc_o_r_2,
                                policy=T.GemmWarpPolicy.Square,
                                wg_wait=-1,
                            )
                            T.warpgroup_commit_batch()
                            T.warpgroup_wait(0)
                            T.barrier_arrive(bar_vr1_free)
                    else:
                        for i_i in range(dynamic_total_blocks):
                            if asym_b128:
                                # V1 is C1-private: park K1/vc0 directly in
                                # shared and retain only vc1 in registers.
                                T.barrier_wait(
                                    bar_kv0_ready, (tile_base_phase ^ i_i) & 1
                                )
                                T.barrier_wait(bar_vr1_free, v1_phase[0])
                                T.annotate_layout(
                                    case01_v1_layout,
                                    allow_reannotation=True,
                                    allow_buffer_region=True,
                                )
                                for r in T.unroll(2):
                                    for u in T.unroll(2):
                                        for v in T.vectorized(8):
                                            row = r * 32 + consumer1_ldg_ty
                                            row_inv = (
                                                (row // 64) * 64
                                                + (row % 64) % 8 * 8
                                                + (row % 64) // 8
                                            )
                                            v_shared_1[
                                                row, 64 * u + consumer1_ldg_tx * 8 + v
                                            ] = kv_shared[
                                                qk_panels // 2 - 1,
                                                row_inv,
                                                64 * u + consumer1_ldg_tx * 8 + v,
                                            ]
                                            k1_reg[r * 16 + u * 8 + v] = kv_shared[
                                                qk_panels // 2 - 1,
                                                row_inv + 64,
                                                64 * u + consumer1_ldg_tx * 8 + v,
                                            ]
                                T.lma_wait()
                                T.barrier_arrive(bar_kv0_free)
                            T.barrier_wait(
                                bar_kv1_read_ready, (tile_base_phase ^ i_i) & 1
                            )
                            if private_v2:
                                # K1/V1 has separate lifetime protection.
                                # P-free is required only before overwriting V2.
                                T.barrier_wait(bar_p_free, (tile_base_phase ^ i_i) & 1)
                            if asym_b128:
                                for r in T.unroll(4):
                                    for u in T.unroll(2):
                                        for v in T.vectorized(8):
                                            row = r * 32 + consumer1_ldg_ty
                                            row_inv = (
                                                (row // 64) * 64
                                                + (row % 64) % 8 * 8
                                                + (row % 64) // 8
                                            )
                                            kv_reg_r[
                                                r * (16 if private_v2 else 32)
                                                + u * 8
                                                + v
                                            ] = kv_shared[
                                                qk_panels // 2,
                                                row_inv,
                                                64 * u + consumer1_ldg_tx * 8 + v,
                                            ]
                                            if private_v2:
                                                v_shared_2[
                                                    r // 2,
                                                    (r % 2) * 32 + consumer1_ldg_ty,
                                                    64 * u + consumer1_ldg_tx * 8 + v,
                                                ] = kv_shared[
                                                    qk_panels // 2 + 1,
                                                    row_inv,
                                                    64 * u + consumer1_ldg_tx * 8 + v,
                                                ]
                                            else:
                                                kv_reg_r[r * 32 + (u + 2) * 8 + v] = (
                                                    kv_shared[
                                                        qk_panels // 2 + 1,
                                                        row_inv,
                                                        64 * u
                                                        + consumer1_ldg_tx * 8
                                                        + v,
                                                    ]
                                                )
                                T.lma_wait()
                                if private_v2:
                                    T.barrier_arrive(bar_v2_ready)
                                T.barrier_arrive(bar_kv1_free)
                            if not asym_b128:
                                for r in T.unroll(block_i // 32):
                                    for u in T.unroll(2):
                                        for v in T.vectorized(8):
                                            kv_reg_r[r * 32 + u * 8 + v] = kv_shared[
                                                qk_panels // 2,
                                                ((consumer1_ldg_ty + r * 32) // 64) * 64
                                                + ((consumer1_ldg_ty + r * 32) % 64)
                                                % 8
                                                * 8
                                                + ((consumer1_ldg_ty + r * 32) % 64)
                                                // 8,
                                                64 * u + consumer1_ldg_tx * 8 + v,
                                            ]
                                            kv_reg_r[r * 32 + (u + 2) * 8 + v] = (
                                                kv_shared[
                                                    qk_panels // 2 + 1,
                                                    ((consumer1_ldg_ty + r * 32) // 64)
                                                    * 64
                                                    + ((consumer1_ldg_ty + r * 32) % 64)
                                                    % 8
                                                    * 8
                                                    + ((consumer1_ldg_ty + r * 32) % 64)
                                                    // 8,
                                                    64 * u + consumer1_ldg_tx * 8 + v,
                                                ]
                                            )

                                T.lma_wait()
                                T.barrier_arrive(bar_kv1_free)
                            if block_i == 64:
                                T.barrier_wait(
                                    bar_alpha_ready, (tile_base_phase ^ i_i) & 1
                                )
                                for h_i in T.Parallel(heads_per_block):
                                    alpha_r[h_i] = alpha_shared[h_i]
                                for h_i, d_i in T.Parallel(
                                    heads_per_block, dim_qk // 4
                                ):
                                    acc_o_r_0[h_i, d_i] *= alpha_r[h_i]
                                    acc_o_r_1[h_i, d_i] *= alpha_r[h_i]
                                    if asymmetric_pv:
                                        acc_o_r_2[h_i, d_i] *= alpha_r[h_i]
                                T.barrier_wait(bar_p_ready, (tile_base_phase ^ i_i) & 1)
                                T.barrier_wait(
                                    bar_vl0_free, (tile_base_phase ^ i_i) & 1
                                )
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
                                            ] = kv_reg_r[r * 32 + u * 8 + v]

                                T.lma_wait()
                                T.sched_boundary()
                                T.barrier_arrive(bar_vr0_ready)
                                T.barrier_wait(
                                    bar_vr0_ready, (tile_base_phase ^ i_i) & 1
                                )

                                T.gemm(
                                    scores_shared,
                                    v_shared_0,
                                    acc_o_r_0,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    wg_wait=-1,
                                )
                                T.wait_wgmma(0)
                                T.barrier_arrive(bar_vr0_free)
                                T.sched_boundary()

                                T.barrier_wait(
                                    bar_vl1_free, (tile_base_phase ^ i_i) & 1
                                )
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
                                            ] = kv_reg_r[r * 32 + (u + 2) * 8 + v]
                                T.lma_wait()
                                T.barrier_arrive(bar_vr1_ready)
                                T.barrier_wait(
                                    bar_vr1_ready, (tile_base_phase ^ i_i) & 1
                                )

                                T.gemm(
                                    scores_shared,
                                    v_shared_1,
                                    acc_o_r_1,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    wg_wait=-1,
                                )
                                T.wait_wgmma(0)
                                T.barrier_arrive(bar_vr1_free)
                            else:
                                T.barrier_wait(
                                    bar_alpha_ready, (tile_base_phase ^ i_i) & 1
                                )
                                for h_i in T.Parallel(heads_per_block):
                                    alpha_r[h_i] = alpha_shared[h_i]
                                for h_i, d_i in T.Parallel(
                                    heads_per_block, dim_qk // 4
                                ):
                                    acc_o_r_0[h_i, d_i] *= alpha_r[h_i]
                                    acc_o_r_1[h_i, d_i] *= alpha_r[h_i]
                                    if asymmetric_pv:
                                        acc_o_r_2[h_i, d_i] *= alpha_r[h_i]
                                T.barrier_wait(bar_p_ready, (tile_base_phase ^ i_i) & 1)
                                T.sched_boundary()
                                if asym_b128:
                                    # All K1/K2/K3 data has been captured and
                                    # both producer banks are already free.
                                    if private_v2:
                                        T.barrier_wait(
                                            bar_v2_ready, (tile_base_phase ^ i_i) & 1
                                        )
                                    for vc in T.unroll(v_chunks):
                                        phase_vc = (i_i * v_chunks + vc) & 1
                                        # K1 -> V1 -> D128:256. K panels carry
                                        # the QK row permutation, whereas V
                                        # must be published in logical order.
                                        # V1 is private to C1 in asymmetric
                                        # H64/B128; protect reuse with the C1
                                        # ready/free phase, not C0's V1 credit.
                                        if vc == 1:
                                            T.barrier_wait(bar_vr1_free, v1_phase[0])
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
                                                            64 * u
                                                            + consumer1_ldg_tx * 8
                                                            + v,
                                                        ] = k1_reg[r * 16 + u * 8 + v]
                                        T.lma_wait()
                                        T.barrier_arrive(bar_vr1_ready)
                                        T.barrier_wait(bar_vr1_ready, v1_phase[0])
                                        T.gemm(
                                            scores_shared[vc, :, :],
                                            v_shared_1,
                                            acc_o_r_2,
                                            policy=T.GemmWarpPolicy.FullRow,
                                            wg_wait=-1,
                                        )
                                        T.warpgroup_commit_batch()
                                        # K2 -> V0 -> D256:384.
                                        # PV1 consumes V1 while C0 finishes V0.
                                        # Do not drain PV1 until PV2 is issued.
                                        T.barrier_wait(bar_vl0_free, phase_vc)
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
                                                        64 * u
                                                        + consumer1_ldg_tx * 8
                                                        + v,
                                                    ] = kv_reg_r[
                                                        vc * (32 if private_v2 else 64)
                                                        + r * (16 if private_v2 else 32)
                                                        + u * 8
                                                        + v
                                                    ]
                                        T.lma_wait()
                                        T.barrier_arrive(bar_vr0_ready)
                                        T.barrier_wait(bar_vr0_ready, phase_vc)
                                        T.gemm(
                                            scores_shared[vc, :, :],
                                            v_shared_0,
                                            acc_o_r_0,
                                            policy=T.GemmWarpPolicy.FullRow,
                                            wg_wait=-1,
                                        )
                                        T.warpgroup_commit_batch()
                                        T.warpgroup_wait(0)
                                        T.barrier_arrive(bar_vr0_free)
                                        T.barrier_arrive(bar_vr1_free)
                                        v1_phase[0] = v1_phase[0] ^ 1

                                        if private_v2:
                                            T.gemm(
                                                scores_shared[vc, :, :],
                                                v_shared_2[vc, :, :],
                                                acc_o_r_1,
                                                policy=T.GemmWarpPolicy.FullRow,
                                                wg_wait=-1,
                                            )
                                            T.wait_wgmma(0)
                                        else:
                                            # K3 -> V1 -> D384:512.  Wait for the
                                            # K1 WGMMA completion credit before
                                            # reusing C1's private V1 tile.
                                            T.barrier_wait(bar_vr1_free, v1_phase[0])
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
                                                            64 * u
                                                            + consumer1_ldg_tx * 8
                                                            + v,
                                                        ] = kv_reg_r[
                                                            vc * 64
                                                            + r * 32
                                                            + (u + 2) * 8
                                                            + v
                                                        ]
                                            T.lma_wait()
                                            T.barrier_arrive(bar_vr1_ready)
                                            T.barrier_wait(bar_vr1_ready, v1_phase[0])
                                            T.gemm(
                                                scores_shared[vc, :, :],
                                                v_shared_1,
                                                acc_o_r_1,
                                                policy=T.GemmWarpPolicy.FullRow,
                                                wg_wait=-1,
                                            )
                                            T.wait_wgmma(0)
                                            T.barrier_arrive(bar_vr1_free)
                                            v1_phase[0] = v1_phase[0] ^ 1

                                    # P and alpha may only be overwritten after
                                    # all C1 PV readers have completed.
                                    T.barrier_arrive(bar_p_free)
                                else:
                                    for vc in T.unroll(v_chunks):
                                        if vc == 0:
                                            T.sched_boundary()
                                        T.barrier_wait(
                                            bar_vl0_free, (i_i * v_chunks + vc) & 1
                                        )
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
                                                        64 * u
                                                        + consumer1_ldg_tx * 8
                                                        + v,
                                                    ] = kv_reg_r[
                                                        vc * 64 + r * 32 + u * 8 + v
                                                    ]
                                        T.lma_wait()
                                        if vc == 0:
                                            T.sched_boundary()
                                        T.barrier_arrive(bar_vr0_ready)
                                        T.barrier_wait(
                                            bar_vr0_ready, (i_i * v_chunks + vc) & 1
                                        )
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
                                        T.barrier_wait(
                                            bar_vl1_free, (i_i * v_chunks + vc) & 1
                                        )
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
                                                        64 * u
                                                        + consumer1_ldg_tx * 8
                                                        + v,
                                                    ] = kv_reg_r[
                                                        vc * 64
                                                        + r * 32
                                                        + (u + 2) * 8
                                                        + v
                                                    ]
                                        T.lma_wait()
                                        T.barrier_arrive(bar_vr1_ready)
                                        T.barrier_wait(
                                            bar_vr1_ready,
                                            (i_i * v_chunks + vc) & 1,
                                        )
                                        T.gemm(
                                            scores_shared[vc, :, :],
                                            v_shared_1,
                                            acc_o_r_1,
                                            policy=T.GemmWarpPolicy.FullRow,
                                            wg_wait=-1,
                                        )
                                        T.wait_wgmma(0)
                                        T.barrier_arrive(bar_vr1_free)

                    T.barrier_wait(bar_final, logical_phase[0] & 1)
                    for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                        acc_o_r_0[h_i, d_i] *= sum_exp_inv_shared[h_i]
                        acc_o_r_1[h_i, d_i] *= sum_exp_inv_shared[h_i]
                        if asymmetric_pv:
                            acc_o_r_2[h_i, d_i] *= sum_exp_inv_shared[h_i]

                    # Do not allow C0 to overwrite the shared inverse until
                    # this group's reads have completed.
                    T.lma_wait()
                    T.barrier_arrive(bar_final_free)
                    if asymmetric_pv and block_i == 64:
                        T.copy(
                            acc_o_r_0,
                            output[s_i, h0:h1, dim_qk // 4 : dim_qk // 2],
                        )
                        T.copy(
                            acc_o_r_1,
                            output[s_i, h0:h1, dim_qk // 2 : dim_qk // 2 + dim_qk // 4],
                        )
                        T.copy(
                            acc_o_r_2,
                            output[s_i, h0:h1, dim_qk // 2 + dim_qk // 4 : dim_qk],
                        )
                    elif asym_b128:
                        T.copy(
                            acc_o_r_2,
                            output[s_i, h0:h1, dim_qk // 4 : dim_qk // 2],
                        )
                        T.copy(
                            acc_o_r_0,
                            output[s_i, h0:h1, dim_qk // 2 : dim_qk // 2 + dim_qk // 4],
                        )
                        T.copy(
                            acc_o_r_1,
                            output[s_i, h0:h1, dim_qk // 2 + dim_qk // 4 : dim_qk],
                        )
                    else:
                        T.copy(
                            acc_o_r_0,
                            output[s_i, h0:h1, dim_qk // 2 : dim_qk // 2 + dim_qk // 4],
                        )
                        T.copy(
                            acc_o_r_1,
                            output[s_i, h0:h1, dim_qk // 2 + dim_qk // 4 : dim_qk],
                        )
                    if has_topk_length:
                        phase_count[0] = tile_base_phase ^ (dynamic_total_blocks & 1)
                        if not use_rs_qk and dynamic_total_blocks > 0:
                            q_phase[0] = q_phase[0] ^ 1
                    if is_persistence:
                        logical_phase[0] = logical_phase[0] ^ 1
                        logical_bx += persistent_blocks
                    else:
                        logical_bx = seq_len * head_repeats

            else:
                while logical_bx < seq_len * head_repeats:
                    if is_persistence:
                        T.call_extern("void", "__musa_loop_transparent_outermost")
                    g_i = by
                    s_i = (
                        logical_bx
                        if head_repeats == 1
                        else (logical_bx // head_repeats)
                    )
                    h0 = g_i * padded_head_kv + (
                        0 if head_repeats == 1 else (logical_bx % head_repeats) * 64
                    )
                    h1 = h0 + heads_per_block
                    dynamic_total_blocks = T.alloc_var(T.int32)
                    if has_topk_length:
                        dynamic_total_blocks = T.ceildiv(topk_length[s_i], block_i)
                    else:
                        dynamic_total_blocks = T.ceildiv(topk, block_i)

                    # Derive parity from the tile index and the row's prefix.
                    # No shape constants: dense rows all have the same tile count,
                    # and optional lengths carry just one prefix bit across rows.
                    tile_base_phase = T.alloc_var(T.int32)
                    tile_base_phase = (
                        phase_count[0]
                        if has_topk_length
                        else (logical_phase[0] & dynamic_total_blocks)
                    ) & 1

                    kperm_mask_local = T.alloc_local([block_i // 16], "bool")
                    kperm_indices_local = T.alloc_local([block_i // 16], indices_dtype)
                    kv_source_local = (
                        T.alloc_local([block_i // 16], T.int32)
                        if use_kv_source_precompute
                        else None
                    )
                    topk_len_local = T.alloc_local([1], indices_dtype)
                    producer_tid = tid - (256 + consumer1_threads)
                    producer_ldg_tx = producer_tid % 8
                    producer_ldg_ty = producer_tid // 8
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
                        if use_kv_source_precompute:
                            for r in index_loop(block_i // 16, explicit=True):
                                kv_source_local[r] = (
                                    T.Cast("int32", kperm_indices_local[r])
                                    * kv_stride_s
                                    + T.Cast("int32", g_i) * kv_stride_g
                                )

                        T.barrier_wait(bar_kv0_free, (tile_base_phase) ^ 1)
                        if not use_rs_qk:
                            if producer_q_first[0] == 0:
                                # Q is still owned by the previous nonempty
                                # query until its consumer releases it.
                                T.barrier_wait(
                                    bar_q_free,
                                    (
                                        q_phase[0]
                                        if has_topk_length
                                        else logical_phase[0]
                                    )
                                    & 1
                                    ^ 1,
                                )
                            load_q_panels(
                                q,
                                q_shared,
                                s_i,
                                h0,
                                producer_ldg_tx,
                                producer_ldg_ty,
                                q_robust_desc,
                            )
                            producer_q_first[0] = 0
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
                            kv_source_local,
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

                        T.barrier_wait(bar_kv1_free, (tile_base_phase) ^ 1)
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
                            kv_source_local,
                            g_i,
                            qk_panels // 2,
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
                            if not use_rs_qk and i_i == 1:
                                T.barrier_arrive(bar_q_ready)
                            T.barrier_arrive(bar_kv0_ready)

                            T.ldlms_crossbb_wait_1()
                            T.barrier_arrive(bar_kv1_ready)
                            T.barrier_wait(
                                bar_producer_protect, (tile_base_phase ^ (i_i - 1)) & 1
                            )
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
                            if use_kv_source_precompute:
                                for r in index_loop(block_i // 16, explicit=True):
                                    kv_source_local[r] = (
                                        T.Cast("int32", kperm_indices_local[r])
                                        * kv_stride_s
                                        + T.Cast("int32", g_i) * kv_stride_g
                                    )

                            T.barrier_wait(
                                bar_kv0_free, ((tile_base_phase ^ i_i) & 1) ^ 1
                            )
                            load_kv_half(
                                kv,
                                kv_shared,
                                kperm_indices_local,
                                kv_source_local,
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

                            T.barrier_wait(
                                bar_kv1_free, ((tile_base_phase ^ i_i) & 1) ^ 1
                            )
                            load_kv_half(
                                kv,
                                kv_shared,
                                kperm_indices_local,
                                kv_source_local,
                                g_i,
                                qk_panels // 2,
                                producer_ldg_tx,
                                producer_ldg_ty,
                                kv_robust_desc,
                            )
                            T.ldlms_crossbb_commit_1()

                        T.ldlms_crossbb_wait_0()
                        if not use_rs_qk and dynamic_total_blocks == 1:
                            T.barrier_arrive(bar_q_ready)
                        T.barrier_arrive(bar_kv0_ready)
                        T.ldlms_crossbb_wait_1()
                        T.barrier_arrive(bar_kv1_ready)
                        T.barrier_wait(
                            bar_producer_protect,
                            (tile_base_phase ^ (dynamic_total_blocks - 1)) & 1,
                        )
                    if has_topk_length:
                        phase_count[0] = tile_base_phase ^ (dynamic_total_blocks & 1)
                        if not use_rs_qk and dynamic_total_blocks > 0:
                            q_phase[0] = q_phase[0] ^ 1
                    if is_persistence:
                        logical_phase[0] = logical_phase[0] ^ 1
                        logical_bx += persistent_blocks
                    else:
                        logical_bx = seq_len * head_repeats

    @T.macro
    def direct_ring_body(
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
        launch_blocks = T.min(persistent_blocks, seq_len) if is_persistence else seq_len
        with T.Kernel(launch_blocks, 1, threads=threads) as (bx, by):
            T.assume(q_stride_s % 8 == 0)
            T.assume(q_stride_h % 8 == 0)
            T.assume(kv_stride_s % 8 == 0)
            T.assume(kv_stride_g % 8 == 0)
            T.assume(indices_stride_s % 8 == 0)
            kv_ring = T.alloc_shared([3, 2, 128, 128], dtype)
            kv_pv = T.Buffer(
                [3, 2, 2, 64, 128], dtype, data=kv_ring.data, scope="shared"
            )
            T.annotate_layout(
                {
                    kv_ring: tilelang.layout.make_sqmma_swizzled_layout(
                        kv_ring,
                        continuity=128,
                        k_major=True,
                        swizzle_granularity=16,
                    )
                }
            )
            p_shared = T.alloc_shared([2, 64, 64], dtype)
            q0_cache = T.alloc_shared([2, 64, 128], dtype)
            q_fence = T.alloc_barrier(256)
            q_loaded = T.alloc_barrier(256)
            alpha_shared = T.alloc_shared([64], accum_dtype)
            inv_shared = T.alloc_shared([64], accum_dtype)
            valid_ring = T.alloc_shared([3, 4, 32], dtype, scope="shared")
            # The widened producer has 256 participants (tid 512:768),
            # while the C0+C1 KV-free ownership remains 512 participants.
            kv_ready = T.alloc_barrier([256, 256, 256])
            kv_free = T.alloc_barrier([512, 512, 512])
            p_ready = T.alloc_barrier(256)
            p_free = T.alloc_barrier(256)
            final_ready = T.alloc_barrier(256)
            final_free = T.alloc_barrier(256)
            kv_desc = T.make_robust_desc(T.address_of(kv[0, 0, 0]), kv_cosize * 2)
            q_desc = T.make_robust_desc(T.address_of(q[0, 0, 0]), q_cosize * 2)
            T.sync_threads()
            tid = T.get_thread_binding()
            query = T.alloc_var("int32", bx)
            # Six half-generations encode slot | (phase << 2) as 3-bit fields.
            # [0, 1, 2, 4, 5, 6] packs to 0x35888. Rotate one tile (6 bits)
            # after both halves drain, preserving the three-slot/two-phase cycle.
            ring_bits = T.alloc_var("uint32", 0x35888)
            tile_base = T.alloc_var("int32", 0)
            query_phase = T.alloc_var("int32", 0)
            if tid < 256:
                while query < seq_len:
                    if is_persistence:
                        T.call_extern("void", "__musa_loop_transparent_outermost")
                    nt = T.alloc_var("int32")
                    nt = (
                        T.ceildiv(topk_length[query], 128)
                        if has_topk_length
                        else T.ceildiv(topk, 128)
                    )
                    q2 = T.alloc_fragment([64, 128], dtype)
                    q3 = T.alloc_fragment([64, 128], dtype)
                    score = T.alloc_fragment([64, 128], accum_dtype)
                    o0 = T.alloc_fragment([64, 128], accum_dtype)
                    maximum = T.alloc_fragment([64], accum_dtype)
                    previous = T.alloc_fragment([64], accum_dtype)
                    alpha = T.alloc_fragment([64], accum_dtype)
                    denominator = T.alloc_fragment([64], accum_dtype)
                    tile_sum = T.alloc_fragment([64], accum_dtype)
                    inverse = T.alloc_fragment([64], accum_dtype)
                    maximum_out = T.alloc_fragment([64], accum_dtype)
                    T.fill(maximum, -(2**30))
                    T.clear(denominator)
                    T.clear(o0)
                    T.barrier_arrive(q_fence)
                    T.barrier_wait(q_fence, query_phase)
                    if nt > 0:
                        T.copy(
                            q[query, 0:64, 0:128],
                            q0_cache[0, :, :],
                            force_async_copy=True,
                            src_robust_desc=q_desc,
                        )
                        T.copy(
                            q[query, 0:64, 128:256],
                            q0_cache[1, :, :],
                            force_async_copy=True,
                            src_robust_desc=q_desc,
                        )
                        T.async_copy_commit_group(1)
                        T.copy(q[query, 0:64, 256:384], q2)
                        T.copy(q[query, 0:64, 384:512], q3)
                        T.async_copy_wait_group(1)
                    T.barrier_arrive(q_loaded)
                    T.barrier_wait(q_loaded, query_phase)
                    for tile in T.serial(nt):
                        s0 = T.Cast("int32", ring_bits & 3)
                        s1 = T.Cast("int32", (ring_bits >> 3) & 3)
                        phase = (tile_base ^ tile) & 1
                        T.barrier_wait(
                            kv_ready[s0], T.Cast("int32", ((ring_bits >> 2) & 1))
                        )
                        for h, j in T.Parallel(64, 128):
                            score[h, j] = T.Cast(
                                accum_dtype, valid_ring[s0, j % 4, j // 4]
                            )
                        T.annotate_layout(
                            {
                                kv_ring: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_ring,
                                    continuity=128,
                                    k_major=True,
                                    swizzle_granularity=16,
                                )
                            },
                            allow_reannotation=True,
                        )
                        T.gemm(
                            q0_cache[0, :, :],
                            kv_ring[s0, 0, :, :],
                            score,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.gemm(
                            q0_cache[1, :, :],
                            kv_ring[s0, 1, :, :],
                            score,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.barrier_wait(
                            kv_ready[s1], T.Cast("int32", ((ring_bits >> 5) & 1))
                        )
                        T.gemm(
                            q2,
                            kv_ring[s1, 0, :, :],
                            score,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.gemm(
                            q3,
                            kv_ring[s1, 1, :, :],
                            score,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.copy(maximum, previous)
                        T.warpgroup_wait(0)
                        # C0 has no PV readers in this half; C1 still owns it.
                        T.barrier_arrive(kv_free[s1])
                        T.reduce_max(score, maximum, dim=1, clear=False)
                        for h in T.Parallel(64):
                            maximum[h] = T.max(previous[h], maximum[h])
                            alpha[h] = T.exp2((previous[h] - maximum[h]) * sm_scale)
                        for h, j in T.Parallel(64, 128):
                            score[h, j] = T.exp2(
                                score[h, j] * sm_scale - maximum[h] * sm_scale
                            )
                        # Previous C1 reads must retire before overwriting either P or alpha.
                        T.barrier_wait(p_free, phase ^ 1)
                        T.copy(alpha, alpha_shared)
                        for vc in T.unroll(2):
                            for h, j in T.Parallel(64, 64):
                                p_shared[vc, h, j] = T.Cast(
                                    dtype, score[h, vc * 64 + j]
                                )
                        T.lma_wait()
                        T.sched_boundary()
                        T.barrier_arrive(p_ready)
                        for h, d in T.Parallel(64, 128):
                            o0[h, d] *= alpha[h]
                        for vc in T.unroll(2):
                            T.annotate_layout(
                                {
                                    kv_pv[
                                        s0, 0, vc, :, :
                                    ]: tilelang.layout.make_sqmma_swizzled_layout(
                                        kv_pv[s0, 0, vc, :, :],
                                        continuity=128,
                                        k_major=False,
                                        swizzle_granularity=16,
                                    )
                                },
                                allow_buffer_region=True,
                                allow_reannotation=True,
                            )
                            T.gemm(
                                p_shared[vc, :, :],
                                kv_pv[s0, 0, vc, :, :],
                                o0,
                                policy=T.GemmWarpPolicy.FullRow,
                                wg_wait=-1,
                            )
                        T.warpgroup_commit_batch()
                        T.reduce_sum(score, tile_sum, dim=1)
                        for h in T.Parallel(64):
                            denominator[h] = denominator[h] * alpha[h] + tile_sum[h]
                        T.warpgroup_wait(0)
                        T.barrier_arrive(kv_free[s0])
                        ring_bits = (ring_bits >> 6) | ((ring_bits & 63) << 12)
                    T.barrier_wait(final_free, query_phase ^ 1)
                    for h in T.Parallel(64):
                        if maximum[h] > -(2**29):
                            if has_attn_sink:
                                inverse[h] = 1 / (
                                    denominator[h]
                                    + T.exp2(
                                        attn_sink[h] * 1.4426950408889634
                                        - maximum[h] * sm_scale
                                    )
                                )
                            else:
                                inverse[h] = 1 / denominator[h]
                            maximum_out[h] = maximum[h] * sm_scale * 0.6931471805599453
                            denominator[h] = (
                                T.log2(denominator[h]) + maximum[h] * sm_scale
                            )
                        else:
                            inverse[h] = 0
                            maximum_out[h] = -T.infinity(accum_dtype)
                            denominator[h] = T.infinity(accum_dtype)
                    T.copy(inverse, inv_shared)
                    T.lma_wait()
                    T.barrier_arrive(final_ready)
                    T.copy(maximum_out, max_logits_out[query, 0:64])
                    for h in T.Parallel(64):
                        lse[query, h] = denominator[h] * 0.6931471805599453
                    for row_group in T.unroll(2):
                        for h, d in T.Parallel(64, 128):
                            if h // 32 == row_group:
                                output[query, h, d] = T.Cast(
                                    dtype, o0[h, d] * inverse[h]
                                )
                    tile_base = (tile_base ^ nt) & 1
                    query_phase = query_phase ^ 1
                    query += persistent_blocks if is_persistence else seq_len
            elif tid < 512:
                while query < seq_len:
                    if is_persistence:
                        T.call_extern("void", "__musa_loop_transparent_outermost")
                    nt = T.alloc_var("int32")
                    nt = (
                        T.ceildiv(topk_length[query], 128)
                        if has_topk_length
                        else T.ceildiv(topk, 128)
                    )
                    o1 = T.alloc_fragment([64, 128], accum_dtype)
                    o2 = T.alloc_fragment([64, 128], accum_dtype)
                    o3 = T.alloc_fragment([64, 128], accum_dtype)
                    a1 = T.alloc_fragment([64], accum_dtype)
                    inv = T.alloc_fragment([64], accum_dtype)
                    T.clear(o1)
                    T.clear(o2)
                    T.clear(o3)
                    for tile in T.serial(nt):
                        s0 = T.Cast("int32", ring_bits & 3)
                        s1 = T.Cast("int32", (ring_bits >> 3) & 3)
                        phase = (tile_base ^ tile) & 1
                        # P-ready implies C0 has consumed both KV-ready events.
                        T.barrier_wait(p_ready, phase)
                        T.copy(alpha_shared, a1)
                        for h, d in T.Parallel(64, 128):
                            o1[h, d] *= a1[h]
                            o2[h, d] *= a1[h]
                            o3[h, d] *= a1[h]
                        for vc in T.unroll(2):
                            T.annotate_layout(
                                {
                                    kv_pv[
                                        s0, 1, vc, :, :
                                    ]: tilelang.layout.make_sqmma_swizzled_layout(
                                        kv_pv[s0, 1, vc, :, :],
                                        continuity=128,
                                        k_major=False,
                                        swizzle_granularity=16,
                                    )
                                },
                                allow_buffer_region=True,
                                allow_reannotation=True,
                            )
                            T.gemm(
                                p_shared[vc, :, :],
                                kv_pv[s0, 1, vc, :, :],
                                o1,
                                policy=T.GemmWarpPolicy.FullRow,
                                wg_wait=-1,
                            )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        T.barrier_arrive(kv_free[s0])
                        for vc in T.unroll(2):
                            T.annotate_layout(
                                {
                                    kv_pv[
                                        s1, 0, vc, :, :
                                    ]: tilelang.layout.make_sqmma_swizzled_layout(
                                        kv_pv[s1, 0, vc, :, :],
                                        continuity=128,
                                        k_major=False,
                                        swizzle_granularity=16,
                                    ),
                                    kv_pv[
                                        s1, 1, vc, :, :
                                    ]: tilelang.layout.make_sqmma_swizzled_layout(
                                        kv_pv[s1, 1, vc, :, :],
                                        continuity=128,
                                        k_major=False,
                                        swizzle_granularity=16,
                                    ),
                                },
                                allow_buffer_region=True,
                                allow_reannotation=True,
                            )
                            T.gemm(
                                p_shared[vc, :, :],
                                kv_pv[s1, 0, vc, :, :],
                                o2,
                                policy=T.GemmWarpPolicy.FullRow,
                                wg_wait=-1,
                            )
                            T.gemm(
                                p_shared[vc, :, :],
                                kv_pv[s1, 1, vc, :, :],
                                o3,
                                policy=T.GemmWarpPolicy.FullRow,
                                wg_wait=-1,
                            )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        T.barrier_arrive(kv_free[s1])
                        T.lma_wait()
                        T.barrier_arrive(p_free)
                        ring_bits = (ring_bits >> 6) | ((ring_bits & 63) << 12)
                    T.barrier_wait(final_ready, query_phase)
                    T.copy(inv_shared, inv)
                    T.lma_wait()
                    T.barrier_arrive(final_free)
                    for h, d in T.Parallel(64, 128):
                        o1[h, d] *= inv[h]
                        o2[h, d] *= inv[h]
                        o3[h, d] *= inv[h]
                    T.copy(o1, output[query, 0:64, 128:256])
                    T.copy(o2, output[query, 0:64, 256:384])
                    T.copy(o3, output[query, 0:64, 384:512])
                    tile_base = (tile_base ^ nt) & 1
                    query_phase = query_phase ^ 1
                    query += persistent_blocks if is_persistence else seq_len
            else:
                producer_tid = tid - 512
                tx = producer_tid % 8
                ty = producer_tid // 8
                producer_rows = 4
                producer_row_stride = 32
                while query < seq_len:
                    if is_persistence:
                        T.call_extern("void", "__musa_loop_transparent_outermost")
                    length = T.alloc_var("int32")
                    length = topk_length[query] if has_topk_length else topk
                    nt = T.alloc_var("int32")
                    nt = T.ceildiv(length, 128)
                    idx = T.alloc_local([producer_rows], "int32")
                    valid = T.alloc_local([producer_rows], "int32")
                    for tile in T.serial(nt):
                        s0 = T.Cast("int32", ring_bits & 3)
                        s1 = T.Cast("int32", (ring_bits >> 3) & 3)
                        for r in T.unroll(producer_rows):
                            row = r * producer_row_stride + ty
                            token_pos = (
                                tile * 128
                                + (row // 64) * 64
                                + (row % 8) * 8
                                + (row % 64) // 8
                            )
                            idx[r] = T.if_then_else(
                                token_pos < topk, indices[query, 0, token_pos], -1
                            )
                            valid[r] = T.Cast(
                                "int32",
                                (
                                    T.Cast("uint32", idx[r])
                                    < T.Cast("uint32", seq_len_kv)
                                )
                                & (token_pos < length),
                            )
                            idx[r] = T.if_then_else(valid[r] != 0, idx[r], 0)
                        for half in T.unroll(2):
                            slot = T.if_then_else(half == 0, s0, s1)
                            copy_phase = T.Cast(
                                "int32",
                                T.if_then_else(
                                    half == 0,
                                    ((ring_bits >> 2) & 1),
                                    ((ring_bits >> 5) & 1),
                                ),
                            )
                            T.barrier_wait(kv_free[slot], copy_phase ^ 1)
                            T.annotate_layout(
                                {
                                    kv_ring: tilelang.layout.make_sqmma_swizzled_layout(
                                        kv_ring,
                                        continuity=128,
                                        k_major=True,
                                        swizzle_granularity=16,
                                    )
                                },
                                allow_reannotation=True,
                            )
                            for r in T.unroll(producer_rows):
                                for panel in T.unroll(2):
                                    for u in T.unroll(2):
                                        for v in T.vectorized(8):
                                            T.copy(
                                                kv[
                                                    idx[r],
                                                    0,
                                                    (half * 2 + panel) * 128
                                                    + u * 64
                                                    + tx * 8
                                                    + v,
                                                ],
                                                kv_ring[
                                                    slot,
                                                    panel,
                                                    r * producer_row_stride + ty,
                                                    u * 64 + tx * 8 + v,
                                                ],
                                                force_async_copy=True,
                                                src_robust_desc=kv_desc,
                                            )
                            if half == 0:
                                T.async_copy_commit_group(0)
                            else:
                                T.async_copy_commit_group(1)
                            if half == 0:
                                for r in T.unroll(producer_rows):
                                    if tx == 0:
                                        valid_ring[
                                            slot,
                                            (r * producer_row_stride + ty) % 4,
                                            (r * producer_row_stride + ty) // 4,
                                        ] = T.Cast(
                                            dtype,
                                            T.if_then_else(valid[r] != 0, 0, -(2**30)),
                                        )
                                T.lma_wait()
                        # Both halves are in flight before either is drained.
                        # Keep both ready events and all reader-release counts.
                        for half in T.unroll(2):
                            if half == 0:
                                T.async_copy_wait_group(0)
                            else:
                                T.async_copy_wait_group(1)
                            slot = T.if_then_else(half == 0, s0, s1)
                            T.barrier_arrive(kv_ready[slot])
                        ring_bits = (ring_bits >> 6) | ((ring_bits & 63) << 12)
                    query += persistent_blocks if is_persistence else seq_len

    if direct_kv_ring:
        dsa_prefill_body = direct_ring_body

    if has_topk_length and has_attn_sink:

        @T.prim_func
        def dsa_prefill(
            q: q_type,
            kv: kv_type,
            indices: indices_type,
            output: T.Tensor(o_shape, dtype),
            max_logits_out: T.Tensor(max_logits_shape, accum_dtype),
            lse: T.Tensor(lse_shape, accum_dtype),
            topk_length: T.Tensor([seq_len], indices_dtype),
            attn_sink: T.Tensor([num_heads], accum_dtype),
            sm_scale: T.float32,
            persistent_blocks: T.int32,
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

    elif has_topk_length:

        @T.prim_func
        def dsa_prefill(
            q: q_type,
            kv: kv_type,
            indices: indices_type,
            output: T.Tensor(o_shape, dtype),
            max_logits_out: T.Tensor(max_logits_shape, accum_dtype),
            lse: T.Tensor(lse_shape, accum_dtype),
            topk_length: T.Tensor([seq_len], indices_dtype),
            sm_scale: T.float32,
            persistent_blocks: T.int32,
            q_cosize: T.int64,
            kv_cosize: T.int64,
        ):
            dsa_prefill_body(
                q,
                kv,
                indices,
                topk_length,
                None,
                output,
                max_logits_out,
                lse,
                sm_scale,
                persistent_blocks,
                q_cosize,
                kv_cosize,
            )

    elif has_attn_sink:

        @T.prim_func
        def dsa_prefill(
            q: q_type,
            kv: kv_type,
            indices: indices_type,
            output: T.Tensor(o_shape, dtype),
            max_logits_out: T.Tensor(max_logits_shape, accum_dtype),
            lse: T.Tensor(lse_shape, accum_dtype),
            attn_sink: T.Tensor([num_heads], accum_dtype),
            sm_scale: T.float32,
            persistent_blocks: T.int32,
            q_cosize: T.int64,
            kv_cosize: T.int64,
        ):
            dsa_prefill_body(
                q,
                kv,
                indices,
                None,
                attn_sink,
                output,
                max_logits_out,
                lse,
                sm_scale,
                persistent_blocks,
                q_cosize,
                kv_cosize,
            )

    else:

        @T.prim_func
        def dsa_prefill(
            q: q_type,
            kv: kv_type,
            indices: indices_type,
            output: T.Tensor(o_shape, dtype),
            max_logits_out: T.Tensor(max_logits_shape, accum_dtype),
            lse: T.Tensor(lse_shape, accum_dtype),
            sm_scale: T.float32,
            persistent_blocks: T.int32,
            q_cosize: T.int64,
            kv_cosize: T.int64,
        ):
            dsa_prefill_body(
                q,
                kv,
                indices,
                None,
                None,
                output,
                max_logits_out,
                lse,
                sm_scale,
                persistent_blocks,
                q_cosize,
                kv_cosize,
            )

    return dsa_prefill


def sparse_mla_fwd_interface_v4_mp32_optimized(
    q,
    kv,
    indices,
    topk_length=None,
    sm_scale=None,
    attn_sink=None,
    return_p_sum: bool = False,
    d_v=512,
    threads: int | None = None,
    verbose=False,
    return_max_logits: bool = False,
    is_persistence: bool = True,
    persistent_blocks: int | None = None,
    block_i: int | None = None,
):
    if get_musa_arch(q) != MP32_ARCH:
        return None
    is_causal = True
    assert return_p_sum is False, "This kernel file is for fwd only"
    assert q.dtype == torch.bfloat16, "q must be bfloat16"
    assert kv.dtype == torch.bfloat16, "kv must be bfloat16"
    assert indices.dtype == torch.int32, "indices must be int32"
    q, q_shape = prepare_sparse_mla_strided_tensor("q", q, multiple=8)
    kv, kv_shape = prepare_sparse_mla_strided_tensor("kv", kv, multiple=8)
    indices, indices_shape = prepare_sparse_mla_strided_tensor(
        "indices", indices, multiple=8
    )
    seq_len, heads, dim_q = q_shape
    seq_len_kv, kv_group, _ = kv_shape

    dim = d_v
    assert kv_group == 1, "Only MQA (kv_group == 1) is validated for v4 sparse prefill"
    assert dim_q == dim, (
        f"V4 expects q last_dim == kv last_dim == {dim}, got q={dim_q}, kv={kv_shape[-1]}"
    )
    assert kv_shape[-1] == dim, f"V4 expects kv last_dim == {dim}, got {kv_shape[-1]}"
    assert dim == 512, f"V4 kernel currently expects dim == d_v == 512, got {dim}"

    _, _, topk = indices_shape
    assert indices_shape == (seq_len, kv_group, topk)
    assert topk % 64 == 0, "V4 sparse prefill requires topk to be a multiple of 64"
    if block_i is None:
        block_i = 128 if heads == 64 and topk >= 4096 else 64
    assert block_i in (64, 128), "V4 MP32 optimized supports block_i 64 or 128"
    if block_i == 128:
        assert heads == 64, "block_i=128 is validated only for H64"
        assert topk >= 4096, "block_i=128 is selected only for TOPK >= 4096"
    if threads is None:
        if heads == 64 and block_i == 128:
            threads = 768
        else:
            threads = 896 if heads == 128 and block_i == 64 else 640
    if is_persistence:
        persistent_blocks = resolve_num_mps(q.device, persistent_blocks)
    else:
        persistent_blocks = 0

    topk_length = validate_token_lengths(
        topk_length, seq_len, "topk_length", max_length=topk
    )
    attn_sink = validate_prefill_attn_sink(attn_sink, heads)

    runtime_sm_scale = (1.0 / dim) ** 0.5 if sm_scale is None else float(sm_scale)
    runtime_sm_scale *= 1.44269504
    kernel_kwargs = {
        "kv_group": kv_group,
        "is_causal": is_causal,
        "threads": threads,
        "has_attn_sink": attn_sink is not None,
        "has_topk_length": topk_length is not None,
        "is_persistence": is_persistence,
        "block_i": block_i,
        # Keep the robust 2D/TME source-precompute path disabled until its
        # event-group lowering is available in the public TileLang ABI.  The
        # validated path is the non-TME packed producer used by Case03.
        "use_kv_source_precompute": False,
    }
    if heads == 64 and block_i == 128 and threads in (640, 768):
        kernel_kwargs["asymmetric_pv"] = True
        kernel_kwargs["use_rs_qk"] = True
    # These inputs dominate the contiguous output and auxiliary tensor spans.
    kernel_factory = jit_for_tensor_addressing(
        sparse_attention_fwd_kernel_v4_mp32_optimized,
        q,
        kv,
        indices,
    )
    kernel = kernel_factory(heads, dim, **kernel_kwargs)
    if verbose:
        kernel.show_source()
    raise_complete_if_dry_run()

    args = [q, kv, indices]
    if topk_length is not None:
        args.append(topk_length)
    if attn_sink is not None:
        args.append(attn_sink)
    args.append(runtime_sm_scale)
    args.append(int(persistent_blocks))
    # Keep the MP32 ABI aligned with the pipelined implementation: cosize is
    # computed from the prepared host tensors and passed as an int64 scalar.
    args.append(cosize(q.shape, q.stride()))
    args.append(cosize(kv.shape, kv.stride()))
    out = kernel(*args)

    out_tensor, max_logits, lse_tensor = out
    if return_max_logits:
        return out_tensor, max_logits, lse_tensor
    return out_tensor, lse_tensor
