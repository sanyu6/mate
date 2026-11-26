from typing import Any

import tilelang
import tilelang.language as T
import torch
from tvm import tirx as tir

from ....execution_context import raise_complete_if_dry_run
from ....utils import cosize
from ..sparse_mla_decode_scheduled_common import (
    SCHEDULED_DECODE_COMPILE_FLAGS,
    SCHEDULED_DECODE_PASS_CONFIGS,
    make_scheduled_decode_combine,
    prepare_scheduled_decode_runtime,
    prepare_sparse_mla_decode_strided_tensor,
)
from ..sparse_mla_index_type import (
    jit_for_tensor_addressing,
    needs_index_type_promotion,
)

TARGET = {"kind": "musa", "arch": "mp_32"}

BLOCK_HEADS = 64
BLOCK_TOPK = 64
NOPE_HEAD_DIM = 512
KEY_ROPE_DIM = 64
QK_HEAD_DIM = NOPE_HEAD_DIM + KEY_ROPE_DIM
VALUE_HEAD_DIM = NOPE_HEAD_DIM
PACKED_KV_BYTES = 656
ROPE_BF16_OFFSET = 264
THREADS = 640

CONSUMER_GROUP_THREADS = 256
CONSUMER0_START = 0
CONSUMER1_START = CONSUMER0_START + CONSUMER_GROUP_THREADS
PRODUCER_START = CONSUMER1_START + CONSUMER_GROUP_THREADS
PRODUCER_THREADS = THREADS - PRODUCER_START
LDG_TY_COUNT = CONSUMER_GROUP_THREADS // 8
ROWS_PER_THREAD = BLOCK_TOPK // LDG_TY_COUNT

PH1S_SMEM_CAPACITY_BYTES = 256 * 1024
Q_SMEM_BYTES = BLOCK_HEADS * QK_HEAD_DIM * 2
P_SMEM_BYTES = BLOCK_HEADS * BLOCK_TOPK * 2
KV_SMEM_BYTES = 2 * BLOCK_TOPK * (NOPE_HEAD_DIM // 2) * 2
K_ROPE_SMEM_BYTES = BLOCK_TOPK * KEY_ROPE_DIM * 2
V_STAGE_SMEM_BYTES = 2 * BLOCK_TOPK * (VALUE_HEAD_DIM // 4) * 2
PIPELINE_SMEM_BYTES = (
    Q_SMEM_BYTES + P_SMEM_BYTES + KV_SMEM_BYTES + K_ROPE_SMEM_BYTES + V_STAGE_SMEM_BYTES
)
AUXILIARY_SMEM_BYTES = BLOCK_HEADS * (4 + 4 + 4) + BLOCK_TOPK * (1 + 4 + 4 * 4)
STATIC_SMEM_BYTES = PIPELINE_SMEM_BYTES + AUXILIARY_SMEM_BYTES

assert Q_SMEM_BYTES == 72 * 1024
assert P_SMEM_BYTES == 8 * 1024
assert KV_SMEM_BYTES == 64 * 1024
assert K_ROPE_SMEM_BYTES == 8 * 1024
assert V_STAGE_SMEM_BYTES == 32 * 1024
assert PIPELINE_SMEM_BYTES == 184 * 1024
assert STATIC_SMEM_BYTES < PH1S_SMEM_CAPACITY_BYTES


@tilelang.jit(
    target=TARGET,
    out_idx=[],
    pass_configs=SCHEDULED_DECODE_PASS_CONFIGS,
    compile_flags=SCHEDULED_DECODE_COMPILE_FLAGS,
)
def sparse_mla_decode_kernel(
    num_heads: int,
    topk: int,
    *,
    kv_group: int = 1,
    sm_scale: float | None = None,
    has_topk_length: bool = True,
    has_attn_sink: bool = False,
    max_nums_splits: int = 32,
    threads: int = THREADS,
    use_int64_addressing: bool = False,
):
    """Build the PH1S V32 FP8 sparse-MLA decode kernel.

    The 640-thread CTA is split into two 256-thread consumer groups and one
    128-thread producer group, mirroring the V4 MP32 pipeline. The producer
    only issues raw async gathers from the packed 656-byte cache into shared
    memory; consumers dequantize in place from shared to registers, expand
    back to shared for SQMMA, and stage their dequantized halves through
    v_shared for PV. S01 owns QK, online softmax, and output columns 0:256.
    S23 owns output columns 256:512. KV gather of the next tile overlaps the
    current tile's QK/PV compute.
    """
    assert num_heads % kv_group == 0
    head_kv = num_heads // kv_group
    heads_per_kv = head_kv
    padded_head_kv = max(tilelang.math.next_power_of_2(head_kv), BLOCK_HEADS)
    if padded_head_kv != head_kv:
        assert kv_group == 1, "padded V3.2 heads currently require MQA"
    if head_kv > BLOCK_HEADS:
        assert head_kv % BLOCK_HEADS == 0, "head_kv should be a multiple of BLOCK_HEADS"
        head_repeats = head_kv // BLOCK_HEADS
    else:
        head_repeats = 1
    heads_per_block = padded_head_kv if head_repeats == 1 else BLOCK_HEADS
    head_blocks = T.ceildiv(heads_per_kv, BLOCK_HEADS)
    assert topk > 0 and topk % BLOCK_TOPK == 0
    assert threads == THREADS

    if sm_scale is None:
        sm_scale = QK_HEAD_DIM**-0.5
    log2_scale = float(sm_scale * 1.4426950408889634)

    batch_size = T.dynamic("batch_size")
    queries_per_batch = T.dynamic("queries_per_batch")
    seq_len_kv = T.dynamic("seq_len_kv")
    num_mp_parts = T.dynamic("num_mp_parts")
    stride_dtype = "int64" if use_int64_addressing else "int32"
    q_stride_b = T.dynamic("q_stride_b", stride_dtype)
    q_stride_s = T.dynamic("q_stride_s", stride_dtype)
    q_stride_h = T.dynamic("q_stride_h", stride_dtype)
    kv_stride_s = T.dynamic("kv_stride_s", stride_dtype)
    kv_stride_g = T.dynamic("kv_stride_g", stride_dtype)
    k_pe_stride_s = T.dynamic("k_pe_stride_s", stride_dtype)
    k_pe_stride_g = T.dynamic("k_pe_stride_g", stride_dtype)
    quant_scales_stride_s = T.dynamic("quant_scales_stride_s", stride_dtype)
    quant_scales_stride_g = T.dynamic("quant_scales_stride_g", stride_dtype)
    indices_stride_b = T.dynamic("indices_stride_b", stride_dtype)
    indices_stride_s = T.dynamic("indices_stride_s", stride_dtype)
    indices_stride_g = T.dynamic("indices_stride_g", stride_dtype)
    dtype = T.bfloat16
    accum_dtype = T.float32
    kv_dtype = "float8_e4m3"
    negative_infinity = -(2**30)

    q_shape = [batch_size, queries_per_batch, num_heads, QK_HEAD_DIM]
    kv_shape = [seq_len_kv, kv_group, PACKED_KV_BYTES]
    k_pe_shape = [seq_len_kv, kv_group, PACKED_KV_BYTES // 2]
    quant_scales_shape = [seq_len_kv, kv_group, PACKED_KV_BYTES // 4]
    indices_shape = [batch_size, queries_per_batch, kv_group, topk]
    q_strides = (q_stride_b, q_stride_s, q_stride_h, 1)
    kv_strides = (kv_stride_s, kv_stride_g, 1)
    k_pe_strides = (k_pe_stride_s, k_pe_stride_g, 1)
    quant_scales_strides = (quant_scales_stride_s, quant_scales_stride_g, 1)
    indices_strides = (indices_stride_b, indices_stride_s, indices_stride_g, 1)
    q_type: Any = T.StridedTensor(q_shape, q_strides, dtype)
    kv_type: Any = T.StridedTensor(kv_shape, kv_strides, kv_dtype)
    k_pe_type: Any = T.StridedTensor(k_pe_shape, k_pe_strides, dtype)
    quant_scales_type: Any = T.StridedTensor(
        quant_scales_shape, quant_scales_strides, T.float32
    )
    indices_type: Any = T.StridedTensor(indices_shape, indices_strides, T.int32)

    dsa_combine = make_scheduled_decode_combine(
        batch=batch_size,
        seq_len=queries_per_batch,
        num_heads=num_heads,
        dim=VALUE_HEAD_DIM,
        num_mp_parts=num_mp_parts,
        dtype=dtype,
        accum_dtype=accum_dtype,
        max_nums_splits=max_nums_splits,
        has_attn_sink=has_attn_sink,
        max_lse_init=negative_infinity * log2_scale,
    )

    @T.macro
    def dequant_packed_to_sqmma(
        kv_packed,
        kv_sqmma,
        kv_reg,
        kv_reg_fp16,
        kv_reg_bf16_load,
        kv_reg_fp8,
        quant_local,
        ldg_ty,
        ldg_tx,
    ):
        """Single-pass fp8->bf16 expand: linear packed staging -> swizzled
        SQMMA buffer.  packed[row, 32u + tx*4 + w] holds the fp8 pair for
        expanded columns 64u + tx*8 + 2w(+1); expanded column group
        64u -> scale index u // 2.  Also fills kv_reg for the V staging path.
        """
        for r in T.unroll(ROWS_PER_THREAD):
            row = ldg_ty + r * LDG_TY_COUNT
            for u in T.unroll(4):
                for v in T.vectorized(4):
                    kv_reg_bf16_load[r * 16 + u * 4 + v] = kv_packed[
                        row, 32 * u + ldg_tx * 4 + v
                    ]
        T.lma_wait()
        for r in T.unroll(ROWS_PER_THREAD):
            for u in T.unroll(4):
                for v in T.vectorized(8):
                    idx = r * 32 + u * 8 + v
                    kv_reg_fp16[idx] = kv_reg_fp8[idx]
        for r in T.unroll(ROWS_PER_THREAD):
            for u in T.unroll(4):
                for v in T.vectorized(8):
                    idx = r * 32 + u * 8 + v
                    kv_reg[idx] = T.Cast(
                        dtype,
                        kv_reg_fp16[idx] * quant_local[r, u // 2],
                    )
        for r in T.unroll(ROWS_PER_THREAD):
            row = ldg_ty + r * LDG_TY_COUNT
            for u in T.unroll(4):
                for v in T.vectorized(8):
                    kv_sqmma[row, 64 * u + ldg_tx * 8 + v] = kv_reg[r * 32 + u * 8 + v]
        T.lma_wait()

    @T.macro
    def dequant_packed_nope_half(
        kv_half_shared,
        kv_reg,
        kv_reg_fp16,
        kv_reg_bf16_load,
        kv_reg_fp8,
        quant_local,
        ldg_ty,
        ldg_tx,
    ):
        """In-place expand one packed 64x256 noPE half.

        The producer staged raw FP8 bytes as packed BF16 at (col % 8) < 4.
        Each thread reads its own 4 packed BF16 (8 FP8 values) per 64-column
        chunk, dequantizes through FP16 with the per-128-column FP32 scale,
        and writes the expanded 8 BF16 back to the same columns. Stage A
        covers columns 0:128, stage B columns 128:256; the write-back of A is
        complete before B reads, so packed data is never clobbered.
        """
        for r in T.unroll(ROWS_PER_THREAD):
            for u in T.unroll(2):
                for v in T.vectorized(4):
                    row = ldg_ty + r * LDG_TY_COUNT
                    kv_reg_bf16_load[r * 16 + u * 4 + v] = kv_half_shared[
                        row, 64 * u + ldg_tx * 8 + v
                    ]
        T.lma_wait()
        for r in T.unroll(ROWS_PER_THREAD):
            for u in T.unroll(2):
                for v in T.vectorized(8):
                    idx = r * 32 + u * 8 + v
                    kv_reg_fp16[idx] = kv_reg_fp8[idx]
        for r in T.unroll(ROWS_PER_THREAD):
            for u in T.unroll(2):
                for v in T.vectorized(8):
                    idx = r * 32 + u * 8 + v
                    kv_reg[idx] = T.Cast(
                        dtype,
                        kv_reg_fp16[idx] * quant_local[r, u // 2],
                    )
        for r in T.unroll(ROWS_PER_THREAD):
            for u in T.unroll(2):
                for v in T.vectorized(8):
                    row = ldg_ty + r * LDG_TY_COUNT
                    kv_half_shared[row, 64 * u + ldg_tx * 8 + v] = kv_reg[
                        r * 32 + u * 8 + v
                    ]
        for r in T.unroll(ROWS_PER_THREAD):
            for u in T.unroll(2):
                for v in T.vectorized(4):
                    row = ldg_ty + r * LDG_TY_COUNT
                    kv_reg_bf16_load[r * 16 + (u + 2) * 4 + v] = kv_half_shared[
                        row, 64 * (u + 2) + ldg_tx * 8 + v
                    ]
        T.lma_wait()
        for r in T.unroll(ROWS_PER_THREAD):
            for u in T.unroll(2):
                for v in T.vectorized(8):
                    idx = r * 32 + (u + 2) * 8 + v
                    kv_reg_fp16[idx] = kv_reg_fp8[idx]
        for r in T.unroll(ROWS_PER_THREAD):
            for u in T.unroll(2):
                for v in T.vectorized(8):
                    idx = r * 32 + (u + 2) * 8 + v
                    kv_reg[idx] = T.Cast(
                        dtype,
                        kv_reg_fp16[idx] * quant_local[r, (u + 2) // 2],
                    )
        for r in T.unroll(ROWS_PER_THREAD):
            for u in T.unroll(2):
                for v in T.vectorized(8):
                    idx = r * 32 + (u + 2) * 8 + v
                    row = ldg_ty + r * LDG_TY_COUNT
                    kv_half_shared[row, 64 * (u + 2) + ldg_tx * 8 + v] = kv_reg[idx]
        T.lma_wait()

    @T.macro
    def stage_value_shared_mp32(v_buf, chunk, kv_reg, ldg_ty, ldg_tx, u_start):
        """Stage a linear 64x128 V quarter under the SG16 PV descriptor."""
        T.annotate_layout(
            {
                v_buf[chunk, :, :]: tilelang.layout.make_sqmma_swizzled_layout(
                    v_buf[chunk, :, :],
                    continuity=128,
                    k_major=False,
                    swizzle_granularity=16,
                )
            },
            allow_reannotation=True,
            allow_buffer_region=True,
        )
        for r in T.unroll(ROWS_PER_THREAD):
            for u in T.unroll(2):
                for v in T.vectorized(8):
                    row = ldg_ty + r * LDG_TY_COUNT
                    v_buf[chunk, row, 64 * u + ldg_tx * 8 + v] = kv_reg[
                        r * 32 + (u + u_start) * 8 + v
                    ]

    @T.prim_func
    def dsa_decode(
        Q: q_type,
        KV: kv_type,
        K_pe: k_pe_type,
        Quant_scales: quant_scales_type,
        Indices: indices_type,
        TopkLength: T.Tensor([batch_size], T.int32),  # type: ignore
        AttnSink: T.Tensor([num_heads], accum_dtype),  # type: ignore
        TileSchedulerMetadata: T.Tensor([num_mp_parts, 8], T.int32),  # type: ignore
        NumSplits: T.Tensor([batch_size + 1], T.int32),  # type: ignore
        Glse: T.Tensor(  # type: ignore
            [batch_size + num_mp_parts, queries_per_batch, num_heads], accum_dtype
        ),
        OutputPartial: T.Tensor(  # type: ignore
            [
                batch_size + num_mp_parts,
                queries_per_batch,
                num_heads,
                VALUE_HEAD_DIM,
            ],
            accum_dtype,
        ),
        Output: T.Tensor(  # type: ignore
            [batch_size, queries_per_batch, num_heads, VALUE_HEAD_DIM], dtype
        ),
        Lse: T.Tensor(  # type: ignore
            [batch_size, num_heads, queries_per_batch], accum_dtype
        ),
        q_cosize: T.int64,
    ):
        with T.Kernel(
            queries_per_batch * head_blocks,
            kv_group,
            num_mp_parts,
            threads=threads,
        ) as (bx, by, bz):
            T.assume(q_stride_b % 8 == 0)
            T.assume(q_stride_s % 8 == 0)
            T.assume(q_stride_h % 8 == 0)
            T.assume(kv_stride_s % 8 == 0)
            T.assume(kv_stride_g % 8 == 0)
            T.assume(k_pe_stride_s % 8 == 0)
            T.assume(k_pe_stride_g % 8 == 0)
            T.assume(quant_scales_stride_s % 4 == 0)
            T.assume(quant_scales_stride_g % 4 == 0)
            T.assume(indices_stride_b % 8 == 0)
            T.assume(indices_stride_s % 8 == 0)
            T.assume(indices_stride_g % 8 == 0)
            # 72 KiB Q + 8 KiB P + 64 KiB KV + 8 KiB rope + 32 KiB V stage.
            Q_left_shared = T.alloc_shared([heads_per_block, NOPE_HEAD_DIM // 2], dtype)
            Q_right_shared = T.alloc_shared(
                [heads_per_block, NOPE_HEAD_DIM // 2], dtype
            )
            Q_rope_shared = T.alloc_shared([heads_per_block, KEY_ROPE_DIM], dtype)
            P_shared = T.alloc_shared([heads_per_block, BLOCK_TOPK], dtype)
            KV_l_shared = T.alloc_shared([BLOCK_TOPK, NOPE_HEAD_DIM // 2], dtype)
            KV_r_shared = T.alloc_shared([BLOCK_TOPK, NOPE_HEAD_DIM // 2], dtype)
            K_rope_shared = T.alloc_shared([BLOCK_TOPK, KEY_ROPE_DIM], dtype)
            # Four owned V chunks: S01 stages dims 0:256 into chunks 0/1, S23
            # stages dims 256:512 into chunks 2/3.  Each group then runs its
            # two PV gemms back to back with no cross-group ready/free
            # handshake (was: six per tile through time-shared v_shared_0/1).
            v_buf = T.alloc_shared([4, BLOCK_TOPK, VALUE_HEAD_DIM // 4], dtype)
            # Linear packed staging for the fp8 pairs: the producer gathers
            # contiguous 16-byte ops here, and dequant expands into the
            # swizzled KV buffers in a single pass (no in-place A/B staging).
            kv_packed_l = T.alloc_shared([BLOCK_TOPK, NOPE_HEAD_DIM // 4], dtype)
            kv_packed_r = T.alloc_shared([BLOCK_TOPK, NOPE_HEAD_DIM // 4], dtype)

            alpha_shared = T.alloc_shared([heads_per_block], accum_dtype)
            sumexp_inv_shared = T.alloc_shared([heads_per_block], accum_dtype)
            sink_scale_shared = T.alloc_shared([heads_per_block], accum_dtype)
            index_valid_shared = T.alloc_shared([BLOCK_TOPK], T.bool)
            quant_l_shared = T.alloc_shared([BLOCK_TOPK, 2], T.float32)
            quant_r_shared = T.alloc_shared([BLOCK_TOPK, 2], T.float32)

            bar_kv_mask_ready = T.alloc_barrier(arrive_count=PRODUCER_THREADS)
            bar_kv_mask_free = T.alloc_barrier(arrive_count=CONSUMER_GROUP_THREADS)
            bar_q = T.alloc_barrier(arrive_count=PRODUCER_START)
            bar_indices_ready = T.alloc_barrier(arrive_count=PRODUCER_THREADS)
            bar_kv0_ready = T.alloc_barrier(arrive_count=PRODUCER_THREADS)
            bar_kv1_ready = T.alloc_barrier(arrive_count=PRODUCER_THREADS)
            bar_kv0_quant_ready = T.alloc_barrier(arrive_count=CONSUMER_GROUP_THREADS)
            bar_kv1_quant_ready = T.alloc_barrier(arrive_count=CONSUMER_GROUP_THREADS)
            bar_kv0_free = T.alloc_barrier(arrive_count=CONSUMER_GROUP_THREADS)
            bar_kv1_free = T.alloc_barrier(arrive_count=CONSUMER_GROUP_THREADS)
            bar_vl0_ready = T.alloc_barrier(arrive_count=CONSUMER_GROUP_THREADS)
            bar_vl1_ready = T.alloc_barrier(arrive_count=CONSUMER_GROUP_THREADS)
            bar_vr0_ready = T.alloc_barrier(arrive_count=CONSUMER_GROUP_THREADS)
            bar_vr1_ready = T.alloc_barrier(arrive_count=CONSUMER_GROUP_THREADS)
            bar_vl0_free = T.alloc_barrier(arrive_count=CONSUMER_GROUP_THREADS)
            bar_vl1_free = T.alloc_barrier(arrive_count=CONSUMER_GROUP_THREADS)
            bar_p_ready = T.alloc_barrier(arrive_count=CONSUMER_GROUP_THREADS)
            bar_final = T.alloc_barrier(arrive_count=CONSUMER_GROUP_THREADS)

            q_robust_desc = T.make_robust_desc(
                T.address_of(Q[0, 0, 0, 0]),
                q_cosize * 2,
            )

            T.sync_threads()

            query_i = bx // head_blocks
            head_block_i = bx % head_blocks
            group_i = by
            head_start = group_i * heads_per_kv + head_block_i * BLOCK_HEADS
            begin_batch_i = TileSchedulerMetadata[bz, 0]
            begin_block_i = TileSchedulerMetadata[bz, 1]
            end_batch_i = TileSchedulerMetadata[bz, 2]
            end_block_i = TileSchedulerMetadata[bz, 3]
            begin_split_i = TileSchedulerMetadata[bz, 4]
            phase_count = T.alloc_local([1], T.int32)
            T.fill(phase_count, 0)
            tid = T.get_thread_binding()

            for batch_i in range(begin_batch_i, end_batch_i + 1):
                tir.call_extern("void", "__musa_loop_transparent_outermost")
                dynamic_total_blocks = T.alloc_var(T.int32)
                if has_topk_length:
                    dynamic_total_blocks = T.max(
                        T.ceildiv(TopkLength[batch_i], BLOCK_TOPK), 1
                    )
                else:
                    dynamic_total_blocks = topk // BLOCK_TOPK
                start_block_i = T.if_then_else(
                    batch_i == begin_batch_i, begin_block_i, 0
                )
                stop_block_i = T.if_then_else(
                    batch_i == end_batch_i, end_block_i, dynamic_total_blocks
                )
                split_i = T.if_then_else(batch_i == begin_batch_i, begin_split_i, 0)
                is_unsplit = (NumSplits[batch_i + 1] - NumSplits[batch_i]) == 1

                if tid < PRODUCER_START:
                    if padded_head_kv != head_kv:
                        # TMA loads zero-fill rows past the padded head count,
                        # so padded-head batches need no host-side padding.
                        T.tma_copy(
                            Q[
                                batch_i,
                                query_i,
                                head_start : head_start + heads_per_block,
                                0:256,
                            ],
                            Q_left_shared,
                            barrier=bar_q,
                        )
                        T.tma_copy(
                            Q[
                                batch_i,
                                query_i,
                                head_start : head_start + heads_per_block,
                                256:512,
                            ],
                            Q_right_shared,
                            barrier=bar_q,
                        )
                        T.tma_copy(
                            Q[
                                batch_i,
                                query_i,
                                head_start : head_start + heads_per_block,
                                512:576,
                            ],
                            Q_rope_shared,
                            barrier=bar_q,
                        )
                    else:
                        T.copy(
                            Q[
                                batch_i,
                                query_i,
                                head_start : head_start + heads_per_block,
                                0:256,
                            ],
                            Q_left_shared,
                            src_robust_desc=q_robust_desc,
                        )
                        T.copy(
                            Q[
                                batch_i,
                                query_i,
                                head_start : head_start + heads_per_block,
                                256:512,
                            ],
                            Q_right_shared,
                            src_robust_desc=q_robust_desc,
                        )
                        T.copy(
                            Q[
                                batch_i,
                                query_i,
                                head_start : head_start + heads_per_block,
                                512:576,
                            ],
                            Q_rope_shared,
                            src_robust_desc=q_robust_desc,
                        )
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                    T.barrier_arrive(bar_q)
                    T.barrier_wait(bar_q, (batch_i - begin_batch_i) & 1)

                if tid < CONSUMER1_START:
                    # S01: QK + online softmax + PV for output columns 0:256.
                    c0_ldg_tx = tid % 8
                    c0_ldg_ty = tid // 8

                    acc_s = T.alloc_fragment([heads_per_block, BLOCK_TOPK], accum_dtype)
                    acc_s_cast = T.alloc_fragment([heads_per_block, BLOCK_TOPK], dtype)
                    acc_o_l_0 = T.alloc_fragment([heads_per_block, 128], accum_dtype)
                    acc_o_l_1 = T.alloc_fragment([heads_per_block, 128], accum_dtype)
                    sumexp = T.alloc_fragment([heads_per_block], accum_dtype)
                    sumexp_i = T.alloc_fragment([heads_per_block], accum_dtype)
                    alpha = T.alloc_fragment([heads_per_block], accum_dtype)
                    max_score = T.alloc_fragment([heads_per_block], accum_dtype)
                    max_score_prev = T.alloc_fragment([heads_per_block], accum_dtype)
                    kv_reg_l = T.alloc_local([ROWS_PER_THREAD * 32], dtype)
                    kv_reg_l_fp16 = T.view(kv_reg_l, [ROWS_PER_THREAD * 32], T.float16)
                    kv_reg_l_bf16_load = T.alloc_local(
                        [ROWS_PER_THREAD * 16], T.bfloat16
                    )
                    kv_reg_l_fp8 = T.view(
                        kv_reg_l_bf16_load, [ROWS_PER_THREAD * 32], kv_dtype
                    )
                    quant_local_l = T.alloc_local([ROWS_PER_THREAD, 2], T.float32)

                    T.fill(acc_o_l_0, 0)
                    T.fill(acc_o_l_1, 0)
                    T.fill(sumexp, 0)
                    T.fill(max_score, negative_infinity)

                    for tile_i in range(start_block_i, stop_block_i):
                        phase = phase_count[0] & 1

                        T.barrier_wait(bar_kv0_ready, phase)
                        for r in T.unroll(ROWS_PER_THREAD):
                            row = c0_ldg_ty + r * LDG_TY_COUNT
                            T.copy(quant_l_shared[row, 0:2], quant_local_l[r, :])
                        T.annotate_layout(
                            {
                                KV_l_shared[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    KV_l_shared[:, :],
                                    k_major=True,
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        dequant_packed_to_sqmma(
                            kv_packed_l,
                            KV_l_shared,
                            kv_reg_l,
                            kv_reg_l_fp16,
                            kv_reg_l_bf16_load,
                            kv_reg_l_fp8,
                            quant_local_l,
                            c0_ldg_ty,
                            c0_ldg_tx,
                        )
                        T.barrier_arrive(bar_kv0_quant_ready)

                        for head_i, topk_i in T.Parallel(heads_per_block, BLOCK_TOPK):
                            acc_s[head_i, topk_i] = T.if_then_else(
                                index_valid_shared[topk_i],
                                0,
                                negative_infinity,
                            )
                        T.lma_wait()

                        T.barrier_wait(bar_kv0_quant_ready, phase)
                        T.gemm(
                            Q_left_shared,
                            KV_l_shared[:, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_kv0_free)

                        T.barrier_wait(bar_kv1_quant_ready, phase)
                        T.annotate_layout(
                            {
                                KV_r_shared[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    KV_r_shared[:, :],
                                    k_major=True,
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        T.gemm(
                            Q_right_shared,
                            KV_r_shared[:, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.gemm(
                            Q_rope_shared,
                            K_rope_shared[:, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_kv1_free)

                        # Online softmax (V3.2 numerics: alpha guard, +1e-7
                        # denominator, exact -2**30 masked scores).
                        T.copy(max_score, max_score_prev)
                        T.reduce_max(acc_s, max_score, dim=1, clear=False)
                        for head_i in T.Parallel(heads_per_block):
                            if max_score[head_i] > negative_infinity:
                                alpha[head_i] = T.exp2(
                                    (max_score_prev[head_i] - max_score[head_i])
                                    * log2_scale
                                )
                            else:
                                alpha[head_i] = 1.0
                        for head_i, topk_i in T.Parallel(heads_per_block, BLOCK_TOPK):
                            if max_score[head_i] > negative_infinity:
                                acc_s[head_i, topk_i] = T.exp2(
                                    (acc_s[head_i, topk_i] - max_score[head_i])
                                    * log2_scale
                                )
                            else:
                                acc_s[head_i, topk_i] = 0.0
                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for head_i in T.Parallel(heads_per_block):
                            sumexp[head_i] = (
                                sumexp[head_i] * alpha[head_i] + sumexp_i[head_i]
                            )
                            alpha_shared[head_i] = alpha[head_i]
                        for head_i, dim_i in T.Parallel(heads_per_block, 128):
                            acc_o_l_0[head_i, dim_i] *= alpha[head_i]
                            acc_o_l_1[head_i, dim_i] *= alpha[head_i]

                        # PH1S exposes logical N=64 directly from the C fragment.
                        T.copy(acc_s, acc_s_cast)
                        T.copy(acc_s_cast, P_shared)
                        T.lma_wait()
                        T.barrier_arrive(bar_p_ready)

                        stage_value_shared_mp32(
                            v_buf, 0, kv_reg_l, c0_ldg_ty, c0_ldg_tx, 0
                        )
                        stage_value_shared_mp32(
                            v_buf, 1, kv_reg_l, c0_ldg_ty, c0_ldg_tx, 2
                        )
                        T.lma_wait()
                        T.barrier_arrive(bar_vl0_ready)
                        T.barrier_wait(bar_vl0_ready, phase)

                        T.sched_boundary()
                        T.gemm(
                            P_shared,
                            v_buf[0, :, :],
                            acc_o_l_0,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.gemm(
                            P_shared,
                            v_buf[1, :, :],
                            acc_o_l_1,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        T.sched_boundary()
                        phase_count[0] = phase_count[0] ^ 1

                    for head_i in T.Parallel(heads_per_block):
                        valid_head = head_start + head_i < (group_i + 1) * heads_per_kv
                        if valid_head and max_score[head_i] > negative_infinity:
                            sumexp_i[head_i] = 1.0 / (sumexp[head_i] + 1e-7)
                            lse_log2 = (
                                T.log2(sumexp[head_i]) + max_score[head_i] * log2_scale
                            )
                            if is_unsplit:
                                Lse[batch_i, head_start + head_i, query_i] = (
                                    lse_log2 * 0.6931471805599453
                                )
                            else:
                                Glse[
                                    split_i + NumSplits[batch_i],
                                    query_i,
                                    head_start + head_i,
                                ] = lse_log2
                            if has_attn_sink and is_unsplit:
                                sink_scale_shared[head_i] = 1.0 / (
                                    1.0
                                    + T.exp2(
                                        AttnSink[head_start + head_i]
                                        * 1.4426950408889634
                                        - lse_log2
                                    )
                                )
                            else:
                                sink_scale_shared[head_i] = 1.0
                        else:
                            sumexp_i[head_i] = 0.0
                            if valid_head:
                                if is_unsplit:
                                    Lse[batch_i, head_start + head_i, query_i] = (
                                        T.infinity(accum_dtype)
                                    )
                                else:
                                    Glse[
                                        split_i + NumSplits[batch_i],
                                        query_i,
                                        head_start + head_i,
                                    ] = T.infinity(accum_dtype)
                            sink_scale_shared[head_i] = 1.0
                        sumexp_inv_shared[head_i] = sumexp_i[head_i]
                    T.barrier_arrive(bar_final)
                    for head_i, dim_i in T.Parallel(heads_per_block, 128):
                        scale = sumexp_i[head_i] * sink_scale_shared[head_i]
                        acc_o_l_0[head_i, dim_i] *= scale
                        acc_o_l_1[head_i, dim_i] *= scale
                    for head_i, dim_i in T.Parallel(heads_per_block, 128):
                        if head_start + head_i < (group_i + 1) * heads_per_kv:
                            if is_unsplit:
                                Output[batch_i, query_i, head_start + head_i, dim_i] = (
                                    T.Cast(dtype, acc_o_l_0[head_i, dim_i])
                                )
                                Output[
                                    batch_i,
                                    query_i,
                                    head_start + head_i,
                                    128 + dim_i,
                                ] = T.Cast(dtype, acc_o_l_1[head_i, dim_i])
                            else:
                                OutputPartial[
                                    split_i + NumSplits[batch_i],
                                    query_i,
                                    head_start + head_i,
                                    dim_i,
                                ] = acc_o_l_0[head_i, dim_i]
                                OutputPartial[
                                    split_i + NumSplits[batch_i],
                                    query_i,
                                    head_start + head_i,
                                    128 + dim_i,
                                ] = acc_o_l_1[head_i, dim_i]

                elif tid < PRODUCER_START:
                    # S23: dequantize the right noPE half and compute PV for
                    # output columns 256:512.
                    c1_tid = tid - CONSUMER1_START
                    c1_ldg_tx = c1_tid % 8
                    c1_ldg_ty = c1_tid // 8

                    acc_o_r_0 = T.alloc_fragment([heads_per_block, 128], accum_dtype)
                    acc_o_r_1 = T.alloc_fragment([heads_per_block, 128], accum_dtype)
                    kv_reg_r = T.alloc_local([ROWS_PER_THREAD * 32], dtype)
                    kv_reg_r_fp16 = T.view(kv_reg_r, [ROWS_PER_THREAD * 32], T.float16)
                    kv_reg_r_bf16_load = T.alloc_local(
                        [ROWS_PER_THREAD * 16], T.bfloat16
                    )
                    kv_reg_r_fp8 = T.view(
                        kv_reg_r_bf16_load, [ROWS_PER_THREAD * 32], kv_dtype
                    )
                    quant_local_r = T.alloc_local([ROWS_PER_THREAD, 2], T.float32)

                    T.fill(acc_o_r_0, 0)
                    T.fill(acc_o_r_1, 0)

                    for tile_i in range(start_block_i, stop_block_i):
                        phase = phase_count[0] & 1

                        T.barrier_wait(bar_kv1_ready, phase)
                        for r in T.unroll(ROWS_PER_THREAD):
                            row = c1_ldg_ty + r * LDG_TY_COUNT
                            T.copy(quant_r_shared[row, 0:2], quant_local_r[r, :])
                        T.annotate_layout(
                            {
                                KV_r_shared[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    KV_r_shared[:, :],
                                    k_major=True,
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        dequant_packed_to_sqmma(
                            kv_packed_r,
                            KV_r_shared,
                            kv_reg_r,
                            kv_reg_r_fp16,
                            kv_reg_r_bf16_load,
                            kv_reg_r_fp8,
                            quant_local_r,
                            c1_ldg_ty,
                            c1_ldg_tx,
                        )
                        T.barrier_arrive(bar_kv1_quant_ready)

                        stage_value_shared_mp32(
                            v_buf, 2, kv_reg_r, c1_ldg_ty, c1_ldg_tx, 0
                        )
                        stage_value_shared_mp32(
                            v_buf, 3, kv_reg_r, c1_ldg_ty, c1_ldg_tx, 2
                        )
                        T.lma_wait()
                        T.barrier_arrive(bar_vr0_ready)
                        T.barrier_wait(bar_vr0_ready, phase)

                        T.barrier_wait(bar_p_ready, phase)
                        for head_i, dim_i in T.Parallel(heads_per_block, 128):
                            acc_o_r_0[head_i, dim_i] *= alpha_shared[head_i]
                            acc_o_r_1[head_i, dim_i] *= alpha_shared[head_i]

                        T.sched_boundary()
                        T.gemm(
                            P_shared,
                            v_buf[2, :, :],
                            acc_o_r_0,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.gemm(
                            P_shared,
                            v_buf[3, :, :],
                            acc_o_r_1,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        T.sched_boundary()
                        phase_count[0] = phase_count[0] ^ 1

                    T.barrier_wait(bar_final, (batch_i - begin_batch_i) & 1)
                    for head_i, dim_i in T.Parallel(heads_per_block, 128):
                        scale = sumexp_inv_shared[head_i] * sink_scale_shared[head_i]
                        acc_o_r_0[head_i, dim_i] *= scale
                        acc_o_r_1[head_i, dim_i] *= scale
                    for head_i, dim_i in T.Parallel(heads_per_block, 128):
                        if head_start + head_i < (group_i + 1) * heads_per_kv:
                            if is_unsplit:
                                Output[
                                    batch_i,
                                    query_i,
                                    head_start + head_i,
                                    256 + dim_i,
                                ] = T.Cast(dtype, acc_o_r_0[head_i, dim_i])
                                Output[
                                    batch_i,
                                    query_i,
                                    head_start + head_i,
                                    384 + dim_i,
                                ] = T.Cast(dtype, acc_o_r_1[head_i, dim_i])
                            else:
                                OutputPartial[
                                    split_i + NumSplits[batch_i],
                                    query_i,
                                    head_start + head_i,
                                    256 + dim_i,
                                ] = acc_o_r_0[head_i, dim_i]
                                OutputPartial[
                                    split_i + NumSplits[batch_i],
                                    query_i,
                                    head_start + head_i,
                                    384 + dim_i,
                                ] = acc_o_r_1[head_i, dim_i]

                else:
                    # Producer: indices + raw async KV/scale/rope gather only.
                    producer_tid = tid - PRODUCER_START
                    ldg_tx = producer_tid % 8
                    ldg_ty = producer_tid // 8
                    kperm_mask_local = T.alloc_local([4], T.bool)
                    kperm_indices_local = T.alloc_local([4], T.int32)
                    topk_len_local = T.alloc_local([1], T.int32)
                    if has_topk_length:
                        topk_len_local[0] = TopkLength[batch_i]
                    else:
                        topk_len_local[0] = topk

                    for tile_i in range(start_block_i, stop_block_i):
                        phase = phase_count[0] & 1

                        for r in T.unroll(4):
                            row = r * 16 + ldg_ty
                            token_pos = tile_i * BLOCK_TOPK + row
                            kperm_indices_local[r] = Indices[
                                batch_i, query_i, group_i, token_pos
                            ]
                            kperm_mask_local[r] = (
                                kperm_indices_local[r] >= 0
                                and kperm_indices_local[r] < seq_len_kv
                                and token_pos < topk_len_local[0]
                            )
                            # Invalid rows may read token zero: their logits
                            # are masked before softmax, so their PV weight is
                            # zero. This avoids unsupported robust scalar-copy
                            # lowering in the packed gather path.
                            kperm_indices_local[r] = T.if_then_else(
                                kperm_mask_local[r],
                                kperm_indices_local[r],
                                0,
                            )
                        T.barrier_wait(bar_kv0_free, (phase & 1) ^ 1)
                        if ldg_tx == 0:
                            for r in T.unroll(4):
                                index_valid_shared[r * 16 + ldg_ty] = kperm_mask_local[
                                    r
                                ]

                        T.annotate_layout(
                            {
                                KV_l_shared[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    KV_l_shared[:, :],
                                    k_major=True,
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(4):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    T.copy(
                                        K_pe[
                                            kperm_indices_local[r],
                                            group_i,
                                            64 * u + ldg_tx * 8 + v,
                                        ],
                                        kv_packed_l[
                                            r * 16 + ldg_ty, 64 * u + ldg_tx * 8 + v
                                        ],
                                        force_async_copy=True,
                                    )
                        if ldg_tx == 0:
                            for r in T.unroll(4):
                                for c in T.vectorized(2):
                                    T.copy(
                                        Quant_scales[
                                            kperm_indices_local[r],
                                            group_i,
                                            128 + c,
                                        ],
                                        quant_l_shared[r * 16 + ldg_ty, c],
                                        force_async_copy=True,
                                    )
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        T.barrier_arrive(bar_kv0_ready)

                        T.barrier_wait(bar_kv1_free, (phase & 1) ^ 1)
                        T.annotate_layout(
                            {
                                KV_r_shared[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    KV_r_shared[:, :],
                                    k_major=True,
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        T.annotate_layout(
                            {
                                K_rope_shared[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    K_rope_shared[:, :],
                                    k_major=True,
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(4):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    T.copy(
                                        K_pe[
                                            kperm_indices_local[r],
                                            group_i,
                                            128 + 64 * u + ldg_tx * 8 + v,
                                        ],
                                        kv_packed_r[
                                            r * 16 + ldg_ty, 64 * u + ldg_tx * 8 + v
                                        ],
                                        force_async_copy=True,
                                    )
                        if ldg_tx == 0:
                            for r in T.unroll(4):
                                for c in T.vectorized(2):
                                    T.copy(
                                        Quant_scales[
                                            kperm_indices_local[r],
                                            group_i,
                                            128 + 2 + c,
                                        ],
                                        quant_r_shared[r * 16 + ldg_ty, c],
                                        force_async_copy=True,
                                    )
                        for r in T.unroll(4):
                            for v in T.vectorized(8):
                                T.copy(
                                    K_pe[
                                        kperm_indices_local[r],
                                        group_i,
                                        ROPE_BF16_OFFSET + ldg_tx * 8 + v,
                                    ],
                                    K_rope_shared[r * 16 + ldg_ty, ldg_tx * 8 + v],
                                    force_async_copy=True,
                                )
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        T.barrier_arrive(bar_kv1_ready)
                        phase_count[0] = phase_count[0] ^ 1

        dsa_combine(
            TileSchedulerMetadata,
            NumSplits,
            Glse,
            OutputPartial,
            AttnSink,
            Output,
            Lse,
        )

    return dsa_decode


def sparse_mla_decode(
    q: torch.Tensor,
    packed_kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Run PH1S sparse MLA decode using the V32 656-byte cache layout."""
    assert q.dtype == torch.bfloat16
    assert q.ndim == 3 and q.shape[-1] == QK_HEAD_DIM
    assert packed_kv.ndim == 3 and packed_kv.shape[-1] == PACKED_KV_BYTES
    assert packed_kv.element_size() == 1
    assert indices.dtype == torch.int32

    seq_len, num_heads, _ = q.shape
    _, kv_group, _ = packed_kv.shape
    assert indices.shape[:2] == (seq_len, kv_group)
    topk = indices.shape[-1]

    q_4d, _ = prepare_sparse_mla_decode_strided_tensor(
        "q", q.view(seq_len, 1, num_heads, QK_HEAD_DIM), multiple=8
    )
    packed_kv, _ = prepare_sparse_mla_decode_strided_tensor(
        "packed_kv", packed_kv, multiple=16
    )
    indices_4d, _ = prepare_sparse_mla_decode_strided_tensor(
        "indices", indices.view(seq_len, 1, kv_group, topk), multiple=8
    )
    metadata = torch.zeros((1, 8), dtype=torch.int32, device="cpu")
    metadata[0, 2] = seq_len - 1
    metadata[0, 3] = topk // BLOCK_TOPK
    num_splits = torch.arange(seq_len + 1, dtype=torch.int32, device="cpu")
    output, _ = tilelang_flashmla_interface(
        q_4d,
        packed_kv.view(torch.uint8),
        indices_4d,
        metadata.to(q.device),
        num_splits.to(q.device),
        sm_scale=sm_scale,
    )
    return output.view(seq_len, num_heads, VALUE_HEAD_DIM)


def tilelang_flashmla_interface(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    sm_scale: float | None = None,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    return_p_sum: bool = False,
    d_v: int = VALUE_HEAD_DIM,
    threads: int = THREADS,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the scheduled PH1S V32 decode path through Mate's ABI."""
    assert not return_p_sum, "This kernel file is for fwd only"
    assert d_v == VALUE_HEAD_DIM
    # The 512-thread value is kept ABI-compatible with the pre-pipeline bench
    # callers; the kernel always runs with THREADS.
    assert threads in (512, THREADS)
    assert q.dtype == torch.bfloat16 and q.ndim == 4
    assert kv.dtype == torch.uint8 and kv.ndim == 3
    assert indices.dtype == torch.int32 and indices.ndim == 4
    assert tile_scheduler_metadata is not None
    assert num_splits is not None

    q, q_shape = prepare_sparse_mla_decode_strided_tensor("q", q, multiple=8)
    kv, kv_shape = prepare_sparse_mla_decode_strided_tensor("kv", kv, multiple=16)
    indices, indices_shape = prepare_sparse_mla_decode_strided_tensor(
        "indices", indices, multiple=8
    )
    batch, seq_len_q, num_heads, dim_qk = q_shape
    _, kv_group, packed_bytes = kv_shape
    assert dim_qk == QK_HEAD_DIM
    assert packed_bytes == PACKED_KV_BYTES
    assert kv_group == 1
    assert indices_shape[:3] == (batch, seq_len_q, kv_group)
    topk = int(indices_shape[-1])
    assert topk > 0 and topk % BLOCK_TOPK == 0
    num_mp_parts = int(tile_scheduler_metadata.shape[0])
    assert tile_scheduler_metadata.shape == (num_mp_parts, 8)
    assert int(num_splits.numel()) == batch + 1

    runtime = prepare_scheduled_decode_runtime(
        batch=batch,
        seq_len=seq_len_q,
        heads=num_heads,
        dim=VALUE_HEAD_DIM,
        topk_length=topk_length,
        attn_sink=attn_sink,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        out_dtype=q.dtype,
        device=q.device,
        variant_name="V3.2 MP32",
        support_split=True,
    )
    assert runtime.glse is not None and runtime.out_partial is not None
    topk_length_arg = runtime.topk_length
    if topk_length_arg is None:
        topk_length_arg = indices.as_strided((batch,), (1,))
    attn_sink_arg = runtime.attn_sink
    if attn_sink_arg is None:
        attn_sink_arg = kv.view(torch.float32).as_strided((num_heads,), (1,))

    needs_int64_addressing = needs_index_type_promotion(
        q, kv, indices, runtime.out_partial
    )
    kernel_factory = jit_for_tensor_addressing(
        sparse_mla_decode_kernel, q, kv, indices, runtime.out_partial
    )
    kernel = kernel_factory(
        num_heads,
        topk,
        kv_group=kv_group,
        sm_scale=sm_scale,
        has_topk_length=runtime.topk_length is not None,
        has_attn_sink=runtime.has_attn_sink,
        max_nums_splits=runtime.max_nums_splits,
        threads=THREADS,
        use_int64_addressing=needs_int64_addressing,
    )
    if verbose:
        kernel.show_source()
    raise_complete_if_dry_run()
    # cosize is computed on the host and passed in as a scalar argument.
    q_cosize = cosize(q.shape, q.stride())
    kernel(
        q,
        kv.view(torch.float8_e4m3fn),
        kv.view(torch.bfloat16),
        kv.view(torch.float32),
        indices,
        topk_length_arg,
        attn_sink_arg,
        tile_scheduler_metadata,
        num_splits,
        runtime.glse,
        runtime.out_partial,
        runtime.out,
        runtime.lse,
        q_cosize,
    )
    return runtime.out, runtime.lse
