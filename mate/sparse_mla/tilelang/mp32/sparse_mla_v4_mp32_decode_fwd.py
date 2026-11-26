# ruff: noqa
"""PH1S/MP32 V4 FP8 sparse-MLA decode.

The V4 cache stores 448 FP8 NoPE values, 64 BF16 RoPE values, and seven
E8M0 scale bytes per token.  Its page layout keeps all scale bytes at the end
of each page, so this narrow MP32 path consumes the original paged ABI rather
than materializing a dequantized cache.
"""

from typing import Any, Optional, Tuple

import torch
import tilelang
from tilelang import language as T

from ....execution_context import raise_complete_if_dry_run
from ....mate_runtime import (
    MP32_ARCH,
    get_musa_arch,
)
from ..sparse_mla_decode_scheduled_common import (
    prepare_sparse_mla_decode_strided_tensor,
)
from ..sparse_mla_index_type import jit_for_tensor_addressing


TARGET = {"kind": "musa", "arch": "mp_32"}

BLOCK_HEADS = 64
BLOCK_TOPK = 64
HEAD_DIM = 512
NOPE_HEAD_DIM = 448
ROPE_HEAD_DIM = 64
LEFT_WIDTH = 256
RIGHT_WIDTH = 256
PAGE_BLOCK_SIZE = 64
CONTENT_BYTES_PER_TOKEN = 576
SCALE_BYTES_PER_TOKEN = 8
ROPE_BYTE_OFFSET = 448
THREADS = 512
S01_THREADS = THREADS // 2

PH1S_SMEM_CAPACITY_BYTES = 256 * 1024
Q_SMEM_BYTES = BLOCK_HEADS * HEAD_DIM * 2
P_SMEM_BYTES = BLOCK_HEADS * BLOCK_TOPK * 2
KV_STAGE_SMEM_BYTES = 2 * 2 * BLOCK_TOPK * LEFT_WIDTH * 2
PIPELINE_SMEM_BYTES = Q_SMEM_BYTES + P_SMEM_BYTES + KV_STAGE_SMEM_BYTES
AUXILIARY_SMEM_BYTES = BLOCK_HEADS * 4 * 3 + BLOCK_TOPK
STATIC_SMEM_BYTES = PIPELINE_SMEM_BYTES + AUXILIARY_SMEM_BYTES

assert Q_SMEM_BYTES == 64 * 1024
assert P_SMEM_BYTES == 8 * 1024
assert KV_STAGE_SMEM_BYTES == 128 * 1024
assert PIPELINE_SMEM_BYTES == 200 * 1024
assert STATIC_SMEM_BYTES < PH1S_SMEM_CAPACITY_BYTES


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
    tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
}


@tilelang.jit(target=TARGET, out_idx=[-2, -1], pass_configs=PASS_CONFIGS)
def sparse_attention_decode_kernel_v4_mp32(
    num_heads: int,
    topk: int,
    *,
    sm_scale: float | None = None,
    has_topk_length: bool = True,
    has_attn_sink: bool = False,
    use_8byte_kv_loads: bool = False,
    threads: int = THREADS,
):
    """Build the narrow unsplit V4 decode kernel for PH1S."""

    assert num_heads > 0 and num_heads % BLOCK_HEADS == 0
    assert topk > 0 and topk % BLOCK_TOPK == 0
    assert threads == THREADS

    logits_scale = HEAD_DIM**-0.5 if sm_scale is None else float(sm_scale)
    log2_scale = logits_scale * 1.4426950408889634

    batch = T.dynamic("batch")
    num_blocks = T.dynamic("num_blocks")
    q_stride_b = T.dynamic("q_stride_b")
    q_stride_h = T.dynamic("q_stride_h")
    kv_nope_stride_block = T.dynamic("kv_nope_stride_block")
    kv_rope_stride_block = T.dynamic("kv_rope_stride_block")
    quant_scales_stride_block = T.dynamic("quant_scales_stride_block")
    indices_stride_b = T.dynamic("indices_stride_b")
    indices_stride_g = T.dynamic("indices_stride_g")
    dtype = T.bfloat16
    accum_dtype = T.float32
    kv_dtype = "float8_e4m3"
    negative_infinity = -(2**30)
    seq_len_kv = num_blocks * PAGE_BLOCK_SIZE
    page_block_bytes = PAGE_BLOCK_SIZE * (
        CONTENT_BYTES_PER_TOKEN + SCALE_BYTES_PER_TOKEN
    )
    page_scale_offset = PAGE_BLOCK_SIZE * CONTENT_BYTES_PER_TOKEN
    q_shape = [batch, num_heads, HEAD_DIM]
    kv_nope_shape = [num_blocks, page_block_bytes]
    kv_rope_shape = [num_blocks, page_block_bytes // 2]
    quant_scales_shape = [num_blocks, page_block_bytes]
    indices_shape = [batch, 1, topk]
    q_strides = (q_stride_b, q_stride_h, 1)
    kv_nope_strides = (kv_nope_stride_block, 1)
    kv_rope_strides = (kv_rope_stride_block, 1)
    quant_scales_strides = (quant_scales_stride_block, 1)
    indices_strides = (indices_stride_b, indices_stride_g, 1)
    q_type: Any = T.StridedTensor(q_shape, q_strides, dtype)
    kv_nope_type: Any = T.StridedTensor(kv_nope_shape, kv_nope_strides, kv_dtype)
    kv_rope_type: Any = T.StridedTensor(kv_rope_shape, kv_rope_strides, dtype)
    quant_scales_type: Any = T.StridedTensor(
        quant_scales_shape, quant_scales_strides, "uint8"
    )
    indices_type: Any = T.StridedTensor(indices_shape, indices_strides, T.int32)

    @T.prim_func
    def main(
        Q: q_type,
        KV_nope: kv_nope_type,
        KV_rope: kv_rope_type,
        Quant_scales: quant_scales_type,
        Indices: indices_type,
        TopkLength: T.Tensor([batch], T.int32),  # type: ignore
        AttnSink: T.Tensor([num_heads], accum_dtype),  # type: ignore
        Output: T.Tensor([batch, num_heads, HEAD_DIM], dtype),  # type: ignore
        Lse: T.Tensor([batch, num_heads], accum_dtype),  # type: ignore
    ):
        with T.Kernel(batch, num_heads // BLOCK_HEADS, threads=threads) as (bx, by):
            T.assume(q_stride_b % 8 == 0)
            T.assume(q_stride_h % 8 == 0)
            T.assume(kv_nope_stride_block % 8 == 0)
            if use_8byte_kv_loads:
                T.assume(kv_rope_stride_block % 4 == 0)
            else:
                T.assume(kv_rope_stride_block % 8 == 0)
            T.assume(quant_scales_stride_block % 8 == 0)
            T.assume(indices_stride_b % 8 == 0)
            T.assume(indices_stride_g % 8 == 0)
            # 64 KiB Q + 8 KiB P + 128 KiB two-stage KV = 200 KiB.
            Q_left_shared = T.alloc_shared([BLOCK_HEADS, LEFT_WIDTH], dtype)
            Q_right_shared = T.alloc_shared([BLOCK_HEADS, RIGHT_WIDTH], dtype)
            P_shared = T.alloc_shared([BLOCK_HEADS, BLOCK_TOPK], dtype)
            KV_shared = T.alloc_shared([2, 2, BLOCK_TOPK, LEFT_WIDTH], dtype)

            alpha_shared = T.alloc_shared([BLOCK_HEADS], accum_dtype)
            sumexp_inv_shared = T.alloc_shared([BLOCK_HEADS], accum_dtype)
            sink_scale_shared = T.alloc_shared([BLOCK_HEADS], accum_dtype)
            index_valid_shared = T.alloc_shared([BLOCK_TOPK], T.bool)

            bar_stage_ready = T.alloc_barrier(arrive_count=THREADS)
            bar_qk_done = T.alloc_barrier(arrive_count=S01_THREADS)
            bar_v_ready = T.alloc_barrier(arrive_count=THREADS)
            bar_tile_done = T.alloc_barrier(arrive_count=THREADS)
            bar_final = T.alloc_barrier(arrive_count=S01_THREADS)

            head_start = by * BLOCK_HEADS
            T.copy(
                Q[bx, head_start : head_start + BLOCK_HEADS, :LEFT_WIDTH],
                Q_left_shared,
                disable_tma=True,
            )
            T.copy(
                Q[bx, head_start : head_start + BLOCK_HEADS, LEFT_WIDTH:],
                Q_right_shared,
                disable_tma=True,
            )
            T.sync_threads()

            tid = T.get_thread_binding()
            if tid < S01_THREADS:
                load_x = tid % 8
                load_y = tid // 8

                acc_s = T.alloc_fragment([BLOCK_HEADS, BLOCK_TOPK], accum_dtype)
                acc_s_cast = T.alloc_fragment([BLOCK_HEADS, BLOCK_TOPK], dtype)
                acc_o = T.alloc_fragment([BLOCK_HEADS, LEFT_WIDTH], accum_dtype)
                sumexp = T.alloc_fragment([BLOCK_HEADS], accum_dtype)
                sumexp_i = T.alloc_fragment([BLOCK_HEADS], accum_dtype)
                alpha = T.alloc_fragment([BLOCK_HEADS], accum_dtype)
                max_score = T.alloc_fragment([BLOCK_HEADS], accum_dtype)
                max_score_prev = T.alloc_fragment([BLOCK_HEADS], accum_dtype)

                # PH1S is more reliable when the cache bytes are first read
                # through the BF16 alias.  A direct global FP8 scalar load
                # lowers to an FP8 conversion sequence that can return NaN
                # even for the zero encoding on the cmodel.  Reinterpreting
                # four BF16 values as eight FP8 values matches the working
                # scheduled MP31 loader while preserving the byte ABI.
                kv_raw_bf16 = T.alloc_local([4], dtype)
                kv_fp8 = T.view(kv_raw_bf16, [8], kv_dtype)
                kv_fp16 = T.alloc_local([8], T.float16)
                quant_scale = T.alloc_local([2, 4], accum_dtype)
                index = T.alloc_local([2], T.int32)
                page_index = T.alloc_local([2], T.int32)
                page_offset = T.alloc_local([2], T.int32)
                index_valid = T.alloc_local([2], T.bool)
                topk_length = T.alloc_local([1], T.int32)

                if has_topk_length:
                    topk_length[0] = TopkLength[bx]
                else:
                    topk_length[0] = topk
                T.fill(acc_o, 0)
                T.fill(sumexp, 0)
                T.fill(max_score, negative_infinity)

                for tile_i in range(topk // BLOCK_TOPK):
                    stage = tile_i & 1
                    phase = tile_i & 1

                    for row_i in T.unroll(2):
                        row = load_y + row_i * 32
                        topk_position = tile_i * BLOCK_TOPK + row
                        index[row_i] = Indices[bx, 0, topk_position]
                        index_valid[row_i] = (
                            topk_position < topk_length[0]
                            and index[row_i] >= 0
                            and index[row_i] < seq_len_kv
                        )
                        index[row_i] = T.if_then_else(
                            index_valid[row_i], index[row_i], 0
                        )
                        page_index[row_i] = index[row_i] // PAGE_BLOCK_SIZE
                        page_offset[row_i] = (
                            index[row_i] % PAGE_BLOCK_SIZE
                        ) * CONTENT_BYTES_PER_TOKEN
                        if load_x == 0:
                            index_valid_shared[row] = index_valid[row_i]
                        for scale_i in T.unroll(4):
                            quant_scale[row_i, scale_i] = T.reinterpret(
                                "float32",
                                T.Cast(
                                    "int32",
                                    Quant_scales[
                                        page_index[row_i],
                                        page_scale_offset
                                        + (index[row_i] % PAGE_BLOCK_SIZE)
                                        * SCALE_BYTES_PER_TOKEN
                                        + scale_i,
                                    ],
                                )
                                << 23,
                            )

                    T.annotate_layout(
                        {
                            KV_shared[
                                stage, 0, :, :
                            ]: tilelang.layout.make_sqmma_swizzled_layout(
                                KV_shared[stage, 0, :, :], k_major=True
                            ),
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True,
                    )

                    for row_i in T.unroll(2):
                        row = load_y + row_i * 32
                        for chunk_i in T.unroll(4):
                            if index_valid[row_i]:
                                for vec_i in T.vectorized(4):
                                    kv_raw_bf16[vec_i] = KV_rope[
                                        page_index[row_i],
                                        page_offset[row_i] // 2
                                        + chunk_i * 32
                                        + load_x * 4
                                        + vec_i,
                                    ]
                            else:
                                for vec_i in T.vectorized(4):
                                    kv_raw_bf16[vec_i] = 0
                            for vec_i in T.vectorized(8):
                                kv_fp16[vec_i] = kv_fp8[vec_i]
                            for vec_i in T.vectorized(8):
                                KV_shared[
                                    stage,
                                    0,
                                    row,
                                    chunk_i * 64 + load_x * 8 + vec_i,
                                ] = T.Cast(
                                    dtype,
                                    kv_fp16[vec_i] * quant_scale[row_i, chunk_i],
                                )

                    T.barrier_arrive(bar_stage_ready)
                    T.barrier_wait(bar_stage_ready, phase)
                    T.gemm(
                        Q_left_shared,
                        KV_shared[stage, 0, :, :],
                        acc_s,
                        clear_accum=True,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.gemm(
                        Q_right_shared,
                        KV_shared[stage, 1, :, :],
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.barrier_arrive(bar_qk_done)

                    for head_i, topk_i in T.Parallel(BLOCK_HEADS, BLOCK_TOPK):
                        acc_s[head_i, topk_i] = T.if_then_else(
                            index_valid_shared[topk_i],
                            acc_s[head_i, topk_i],
                            negative_infinity,
                        )
                    T.copy(max_score, max_score_prev)
                    T.reduce_max(acc_s, max_score, dim=1, clear=False)
                    for head_i in T.Parallel(BLOCK_HEADS):
                        if max_score[head_i] > negative_infinity:
                            alpha[head_i] = T.exp2(
                                (max_score_prev[head_i] - max_score[head_i])
                                * log2_scale
                            )
                        else:
                            alpha[head_i] = 1.0
                    for head_i, topk_i in T.Parallel(BLOCK_HEADS, BLOCK_TOPK):
                        if max_score[head_i] > negative_infinity:
                            acc_s[head_i, topk_i] = T.exp2(
                                (acc_s[head_i, topk_i] - max_score[head_i]) * log2_scale
                            )
                        else:
                            acc_s[head_i, topk_i] = 0.0
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for head_i in T.Parallel(BLOCK_HEADS):
                        sumexp[head_i] = (
                            sumexp[head_i] * alpha[head_i] + sumexp_i[head_i]
                        )
                        alpha_shared[head_i] = alpha[head_i]
                    for head_i, dim_i in T.Parallel(BLOCK_HEADS, LEFT_WIDTH):
                        acc_o[head_i, dim_i] *= alpha[head_i]

                    # PH1S consumes N=64 directly.  The PV descriptor changes
                    # metadata only; it does not rewrite this SG16 allocation.
                    T.copy(acc_s, acc_s_cast)
                    T.copy(acc_s_cast, P_shared)
                    T.annotate_layout(
                        {
                            KV_shared[
                                stage, 0, :, :
                            ]: tilelang.layout.make_sqmma_swizzled_layout(
                                KV_shared[stage, 0, :, :],
                                continuity=128,
                                k_major=False,
                                swizzle_granularity=16,
                            )
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True,
                    )
                    T.barrier_arrive(bar_v_ready)
                    T.barrier_wait(bar_v_ready, phase)
                    T.gemm(
                        P_shared,
                        KV_shared[stage, 0, :, :],
                        acc_o,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.barrier_arrive(bar_tile_done)
                    T.barrier_wait(bar_tile_done, phase)

                for head_i in T.Parallel(BLOCK_HEADS):
                    if max_score[head_i] > negative_infinity:
                        sumexp_i[head_i] = 1.0 / (sumexp[head_i] + 1e-7)
                        lse_log2 = (
                            T.log2(sumexp[head_i]) + max_score[head_i] * log2_scale
                        )
                        Lse[bx, head_start + head_i] = lse_log2 * 0.6931471805599453
                        if has_attn_sink:
                            sink_scale_shared[head_i] = 1.0 / (
                                1.0
                                + T.exp2(
                                    AttnSink[head_start + head_i] * 1.4426950408889634
                                    - lse_log2
                                )
                            )
                        else:
                            sink_scale_shared[head_i] = 1.0
                    else:
                        sumexp_i[head_i] = 0.0
                        Lse[bx, head_start + head_i] = T.infinity(accum_dtype)
                        sink_scale_shared[head_i] = 1.0
                    sumexp_inv_shared[head_i] = sumexp_i[head_i]
                T.barrier_arrive(bar_final)
                for head_i, dim_i in T.Parallel(BLOCK_HEADS, LEFT_WIDTH):
                    acc_o[head_i, dim_i] *= sumexp_i[head_i] * sink_scale_shared[head_i]
                T.copy(
                    acc_o,
                    Output[bx, head_start : head_start + BLOCK_HEADS, :LEFT_WIDTH],
                    disable_tma=True,
                )

            elif tid < THREADS:
                right_tid = tid - S01_THREADS
                right_load_x = right_tid % 8
                right_load_y = right_tid // 8
                acc_o = T.alloc_fragment([BLOCK_HEADS, RIGHT_WIDTH], accum_dtype)
                kv_raw_bf16 = T.alloc_local([4], dtype)
                kv_fp8 = T.view(kv_raw_bf16, [8], kv_dtype)
                kv_fp16 = T.alloc_local([8], T.float16)
                quant_scale = T.alloc_local([2, 3], accum_dtype)
                index = T.alloc_local([2], T.int32)
                page_index = T.alloc_local([2], T.int32)
                page_offset = T.alloc_local([2], T.int32)
                index_valid = T.alloc_local([2], T.bool)
                topk_length = T.alloc_local([1], T.int32)

                if has_topk_length:
                    topk_length[0] = TopkLength[bx]
                else:
                    topk_length[0] = topk
                T.fill(acc_o, 0)

                for tile_i in range(topk // BLOCK_TOPK):
                    stage = tile_i & 1
                    phase = tile_i & 1

                    for row_i in T.unroll(2):
                        row = right_load_y + row_i * 32
                        topk_position = tile_i * BLOCK_TOPK + row
                        index[row_i] = Indices[bx, 0, topk_position]
                        index_valid[row_i] = (
                            topk_position < topk_length[0]
                            and index[row_i] >= 0
                            and index[row_i] < seq_len_kv
                        )
                        index[row_i] = T.if_then_else(
                            index_valid[row_i], index[row_i], 0
                        )
                        page_index[row_i] = index[row_i] // PAGE_BLOCK_SIZE
                        page_offset[row_i] = (
                            index[row_i] % PAGE_BLOCK_SIZE
                        ) * CONTENT_BYTES_PER_TOKEN
                        for scale_i in T.unroll(3):
                            quant_scale[row_i, scale_i] = T.reinterpret(
                                "float32",
                                T.Cast(
                                    "int32",
                                    Quant_scales[
                                        page_index[row_i],
                                        page_scale_offset
                                        + (index[row_i] % PAGE_BLOCK_SIZE)
                                        * SCALE_BYTES_PER_TOKEN
                                        + 4
                                        + scale_i,
                                    ],
                                )
                                << 23,
                            )

                    T.annotate_layout(
                        {
                            KV_shared[
                                stage, 1, :, :
                            ]: tilelang.layout.make_sqmma_swizzled_layout(
                                KV_shared[stage, 1, :, :], k_major=True
                            ),
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True,
                    )
                    for row_i in T.unroll(2):
                        row = right_load_y + row_i * 32
                        for chunk_i in T.unroll(3):
                            if index_valid[row_i]:
                                for vec_i in T.vectorized(4):
                                    kv_raw_bf16[vec_i] = KV_rope[
                                        page_index[row_i],
                                        page_offset[row_i] // 2
                                        + LEFT_WIDTH // 2
                                        + chunk_i * 32
                                        + right_load_x * 4
                                        + vec_i,
                                    ]
                            else:
                                for vec_i in T.vectorized(4):
                                    kv_raw_bf16[vec_i] = 0
                            for vec_i in T.vectorized(8):
                                kv_fp16[vec_i] = kv_fp8[vec_i]
                            for vec_i in T.vectorized(8):
                                KV_shared[
                                    stage,
                                    1,
                                    row,
                                    chunk_i * 64 + right_load_x * 8 + vec_i,
                                ] = T.Cast(
                                    dtype,
                                    kv_fp16[vec_i] * quant_scale[row_i, chunk_i],
                                )
                        if index_valid[row_i]:
                            if use_8byte_kv_loads:
                                # A page stride can be 8-byte aligned without
                                # being 16-byte aligned.  Keep each BF16 load
                                # at 8 bytes in that case; vectorized(8)
                                # otherwise lowers to an unaligned 16-byte
                                # global load for the final RoPE values.
                                for half_i in T.unroll(2):
                                    for vec_i in T.vectorized(4):
                                        KV_shared[
                                            stage,
                                            1,
                                            row,
                                            192 + right_load_x * 8 + half_i * 4 + vec_i,
                                        ] = KV_rope[
                                            page_index[row_i],
                                            page_offset[row_i] // 2
                                            + ROPE_BYTE_OFFSET // 2
                                            + right_load_x * 8
                                            + half_i * 4
                                            + vec_i,
                                        ]
                            else:
                                for vec_i in T.vectorized(8):
                                    KV_shared[
                                        stage,
                                        1,
                                        row,
                                        192 + right_load_x * 8 + vec_i,
                                    ] = KV_rope[
                                        page_index[row_i],
                                        page_offset[row_i] // 2
                                        + ROPE_BYTE_OFFSET // 2
                                        + right_load_x * 8
                                        + vec_i,
                                    ]
                        else:
                            for vec_i in T.vectorized(8):
                                KV_shared[
                                    stage,
                                    1,
                                    row,
                                    192 + right_load_x * 8 + vec_i,
                                ] = 0

                    T.barrier_arrive(bar_stage_ready)
                    T.barrier_wait(bar_qk_done, phase)
                    T.annotate_layout(
                        {
                            KV_shared[
                                stage, 1, :, :
                            ]: tilelang.layout.make_sqmma_swizzled_layout(
                                KV_shared[stage, 1, :, :],
                                continuity=128,
                                k_major=False,
                                swizzle_granularity=16,
                            )
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True,
                    )
                    T.barrier_arrive(bar_v_ready)
                    T.barrier_wait(bar_v_ready, phase)
                    for head_i, dim_i in T.Parallel(BLOCK_HEADS, RIGHT_WIDTH):
                        acc_o[head_i, dim_i] *= alpha_shared[head_i]
                    T.gemm(
                        P_shared,
                        KV_shared[stage, 1, :, :],
                        acc_o,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.barrier_arrive(bar_tile_done)
                    if tile_i + 1 < topk // BLOCK_TOPK:
                        T.barrier_wait(bar_tile_done, phase)

                T.barrier_wait(bar_final, 0)
                for head_i, dim_i in T.Parallel(BLOCK_HEADS, RIGHT_WIDTH):
                    acc_o[head_i, dim_i] *= (
                        sumexp_inv_shared[head_i] * sink_scale_shared[head_i]
                    )
                T.copy(
                    acc_o,
                    Output[bx, head_start : head_start + BLOCK_HEADS, LEFT_WIDTH:],
                    disable_tma=True,
                )

    return main


def sparse_mla_decode_fwd_interface_v4_mp32(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Run the unsplit MP32 V4 decode path, or return ``None``."""

    arch = get_musa_arch(q)
    if arch != MP32_ARCH:
        return None
    if (
        q.dtype != torch.bfloat16
        or q.ndim != 4
        or indices.dtype != torch.int32
        or indices.ndim != 4
        or k_cache.ndim != 4
        or k_cache.shape[2] != 1
        or k_cache.shape[-1] != 584
        or k_cache.element_size() != 1
        or k_cache.stride(-1) != 1
        or k_cache.device != q.device
        or indices.device != q.device
    ):
        return None

    batch, seq_len_q, num_heads, dim_q = q.shape
    num_blocks, page_block_size, _, _ = k_cache.shape
    if (
        batch <= 0
        or seq_len_q != 1
        or num_heads <= 0
        or num_heads % BLOCK_HEADS != 0
        or dim_q != HEAD_DIM
        or num_blocks <= 0
        or page_block_size != PAGE_BLOCK_SIZE
        or indices.shape[:3] != (batch, 1, 1)
    ):
        return None
    topk = int(indices.shape[-1])
    if topk <= 0 or topk % BLOCK_TOPK != 0:
        return None

    has_topk_length = topk_length is not None
    # The compiled kernel keeps a fixed argument list.  These aliases avoid a
    # graph-time allocation and are compile-time dead when the matching
    # has_* specialization is false.
    if topk_length is None:
        topk_length_arg = indices.as_strided((batch,), (1,))
    elif (
        topk_length.dtype != torch.int32
        or topk_length.shape != (batch,)
        or topk_length.device != q.device
    ):
        return None
    else:
        topk_length_arg = topk_length.contiguous()

    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink_arg = q.view(torch.float32).as_strided((num_heads,), (1,))
    elif (
        attn_sink.dtype != torch.float32
        or attn_sink.shape != (num_heads,)
        or attn_sink.device != q.device
    ):
        return None
    else:
        attn_sink_arg = attn_sink.contiguous()

    cache_bytes = k_cache.view(torch.uint8)
    page_stride_bytes = cache_bytes.stride(0)
    if page_stride_bytes < PAGE_BLOCK_SIZE * (
        CONTENT_BYTES_PER_TOKEN + SCALE_BYTES_PER_TOKEN
    ):
        return None
    if page_stride_bytes % 8 != 0:
        return None
    use_8byte_kv_loads = page_stride_bytes % 16 != 0
    cache_pages = torch.as_strided(
        cache_bytes,
        (num_blocks, page_stride_bytes),
        (page_stride_bytes, 1),
    )
    logical_page_bytes = PAGE_BLOCK_SIZE * (
        CONTENT_BYTES_PER_TOKEN + SCALE_BYTES_PER_TOKEN
    )
    cache_pages = cache_pages[:, :logical_page_bytes]
    q_view, _ = prepare_sparse_mla_decode_strided_tensor("q", q[:, 0, :, :], multiple=8)
    kv_nope, _ = prepare_sparse_mla_decode_strided_tensor(
        "kv_nope", cache_pages.view(torch.float8_e4m3fn), multiple=8
    )
    kv_rope, _ = prepare_sparse_mla_decode_strided_tensor(
        "kv_rope", cache_pages.view(torch.bfloat16), multiple=4
    )
    quant_scales, _ = prepare_sparse_mla_decode_strided_tensor(
        "quant_scales", cache_pages, multiple=8
    )
    indices_view, _ = prepare_sparse_mla_decode_strided_tensor(
        "indices", indices[:, 0, :, :], multiple=8
    )

    kernel_factory = jit_for_tensor_addressing(
        sparse_attention_decode_kernel_v4_mp32,
        q_view,
        kv_nope,
        kv_rope,
        quant_scales,
        indices_view,
    )
    kernel = kernel_factory(
        num_heads,
        topk,
        sm_scale=sm_scale,
        has_topk_length=has_topk_length,
        has_attn_sink=has_attn_sink,
        use_8byte_kv_loads=use_8byte_kv_loads,
    )
    if verbose:
        kernel.show_source()
    raise_complete_if_dry_run()
    output, lse = kernel(
        q_view,
        kv_nope,
        kv_rope,
        quant_scales,
        indices_view,
        topk_length_arg,
        attn_sink_arg,
    )
    return output.view(batch, 1, num_heads, HEAD_DIM), lse.view(batch, num_heads, 1)
