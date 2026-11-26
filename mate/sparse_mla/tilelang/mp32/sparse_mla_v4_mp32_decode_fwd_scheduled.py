"""PH1S/MP32 scheduled V4 FP8 sparse-MLA decode.

This file is deliberately separate from the MP31 scheduled kernel.  The
metadata/split contract is shared, but PH1S uses SG16 operands directly: the
score tile is kept in logical ``[H, 64]`` order and the PV values are staged
without the MP31 64-to-8x8 N permutation.
"""

# ruff: noqa

from typing import Any

import torch

import tilelang
from tilelang import language as T
from tvm import tirx as tir

from ....utils import cosize
from ..sparse_mla_decode_scheduled_common import (
    SCHEDULED_DECODE_COMPILE_FLAGS,
    SCHEDULED_DECODE_PASS_CONFIGS,
    make_scheduled_decode_combine,
    make_scheduled_decode_finalize_left,
    make_scheduled_decode_finalize_right,
    make_scheduled_decode_indices_loader,
    prepare_sparse_mla_decode_strided_tensor,
    prepare_scheduled_decode_runtime,
    validate_batch_lengths,
)
from ..sparse_mla_index_type import jit_for_tensor_addressing

from ....execution_context import raise_complete_if_dry_run


TARGET = {"kind": "musa", "arch": "mp_32"}
MP32_SCORE_SWIZZLE = False


@tilelang.jit(
    out_idx=[],
    target=TARGET,
    pass_configs=SCHEDULED_DECODE_PASS_CONFIGS,
    compile_flags=SCHEDULED_DECODE_COMPILE_FLAGS,
)
def sparse_attention_decode_fwd_scheduled_kernel_v4(
    num_heads,
    dim,
    *,
    has_extra=False,
    kv_group=1,
    sm_scale=None,
    block_m=64,
    block_i=64,
    threads=0,
    consumer0_threads=256,
    consumer1_threads=256,
    producer_threads=128,
    max_nums_splits=32,
    has_attn_sink=False,
    has_topk_length=False,
    has_extra_topk_length=False,
    page_block_size=64,
    extra_page_block_size=64,
    use_int64_cosize=False,
    use_8byte_kv_loads=False,
    fixed_topk=None,
    cache_format="v4",
    extra_cache_format="v4",
    has_q_rope=False,
    q_position_dtype="int64",
):
    assert dim == tilelang.math.next_power_of_2(dim), (
        f"haven't check padding correctness yet, dim={dim}"
    )
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504
    topk = fixed_topk if fixed_topk is not None else T.dynamic("topk")
    use_three_v_buffers = (
        fixed_topk is not None
        and not has_extra
        and (num_heads >= 128 or fixed_topk >= 8192)
    )
    extra_topk = T.dynamic("extra_topk")
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    rope_positions = T.dynamic("rope_positions")
    assert cache_format in ("v4", "v41")
    assert extra_cache_format in ("v4", "v41", "v41_fp4")
    dim_bytes = 584 if cache_format == "v4" else 528
    extra_dim_bytes = 288 if extra_cache_format == "v41_fp4" else (
        584 if extra_cache_format == "v4" else 528
    )
    rope_bytes_offset = 448
    scale_bytes_offset = 576
    page_block_bytes = page_block_size * dim_bytes
    page_block_bytes_extra = extra_page_block_size * extra_dim_bytes

    num_blocks = T.dynamic("num_blocks")
    num_blocks_extra = T.dynamic("num_blocks_extra")
    page_block_size_extra = extra_page_block_size
    seq_len_kv = num_blocks * page_block_size
    seq_len_kv_extra = num_blocks_extra * page_block_size_extra
    num_mp_parts = T.dynamic("num_mp_parts")
    q_stride_b = T.dynamic("q_stride_b")
    q_stride_s = T.dynamic("q_stride_s")
    q_stride_h = T.dynamic("q_stride_h")
    kv_nope_stride_block = T.dynamic("kv_nope_stride_block")
    kv_rope_stride_block = T.dynamic("kv_rope_stride_block")
    quant_scales_stride_block = T.dynamic("quant_scales_stride_block")
    indices_stride_b = T.dynamic("indices_stride_b")
    indices_stride_s = T.dynamic("indices_stride_s")
    indices_stride_g = T.dynamic("indices_stride_g")
    extra_kv_nope_stride_block = T.dynamic("extra_kv_nope_stride_block")
    extra_kv_rope_stride_block = T.dynamic("extra_kv_rope_stride_block")
    extra_quant_scales_stride_block = T.dynamic("extra_quant_scales_stride_block")
    extra_indices_stride_b = T.dynamic("extra_indices_stride_b")
    extra_indices_stride_s = T.dynamic("extra_indices_stride_s")
    extra_indices_stride_g = T.dynamic("extra_indices_stride_g")

    head_kv = num_heads // kv_group
    q_shape = [batch, seq_len, num_heads, dim]
    o_shape = [batch, seq_len, num_heads, dim]
    lse_shape = [batch, num_heads, seq_len]
    indices_shape = [batch, seq_len, kv_group, topk]
    extra_indices_shape = [batch, seq_len, kv_group, extra_topk]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"
    kv_latent_dtype = "float8_e4m3"
    dtype_bytes = 2
    assert block_m in (16, 64), "V4 decode supports BM16 or BM64"
    assert block_i == 64, "V4 decode supports BN64"
    assert producer_threads == 128, "V4 decode producer currently requires 128 threads"
    assert consumer0_threads in (128, 256), (
        "V4 decode consumer0 must use one or two warpgroups"
    )
    assert consumer1_threads in (128, 256), (
        "V4 decode consumer1 must use one or two warpgroups"
    )
    assert consumer0_threads % 128 == 0 and consumer1_threads % 128 == 0
    assert consumer0_threads % 8 == 0 and consumer1_threads % 8 == 0
    total_role_threads = consumer0_threads + consumer1_threads + producer_threads
    if threads is None or threads == 0:
        threads = total_role_threads
    else:
        assert threads == total_role_threads, (
            "threads must equal consumer0_threads + consumer1_threads + "
            "producer_threads"
        )
    consumer1_start = consumer0_threads
    producer_start = consumer0_threads + consumer1_threads
    consumer0_gemm_threads = 128 if block_m == 16 else consumer0_threads
    assert consumer0_threads >= consumer0_gemm_threads
    consumer0_ldg_ty_count = consumer0_threads // 8
    consumer1_ldg_ty_count = consumer1_threads // 8
    assert block_i % consumer0_ldg_ty_count == 0
    assert block_i % consumer1_ldg_ty_count == 0
    consumer0_rows_per_thread = block_i // consumer0_ldg_ty_count
    consumer1_rows_per_thread = block_i // consumer1_ldg_ty_count
    kv_nope_shape = [num_blocks, page_block_bytes]
    kv_rope_shape = [num_blocks, page_block_bytes // 2]
    quant_scales_shape = [num_blocks, page_block_bytes]
    extra_kv_nope_shape = [num_blocks_extra, page_block_bytes_extra]
    extra_kv_rope_shape = [num_blocks_extra, page_block_bytes_extra // 2]
    extra_quant_scales_shape = [num_blocks_extra, page_block_bytes_extra]
    q_strides = (q_stride_b, q_stride_s, q_stride_h, 1)
    kv_nope_strides = (kv_nope_stride_block, 1)
    kv_rope_strides = (kv_rope_stride_block, 1)
    quant_scales_strides = (quant_scales_stride_block, 1)
    indices_strides = (indices_stride_b, indices_stride_s, indices_stride_g, 1)
    extra_kv_nope_strides = (extra_kv_nope_stride_block, 1)
    extra_kv_rope_strides = (extra_kv_rope_stride_block, 1)
    extra_quant_scales_strides = (extra_quant_scales_stride_block, 1)
    extra_indices_strides = (
        extra_indices_stride_b,
        extra_indices_stride_s,
        extra_indices_stride_g,
        1,
    )
    q_type: Any = T.StridedTensor(q_shape, q_strides, dtype)
    kv_nope_type: Any = T.StridedTensor(kv_nope_shape, kv_nope_strides, kv_latent_dtype)
    kv_rope_type: Any = T.StridedTensor(kv_rope_shape, kv_rope_strides, dtype)
    quant_scales_type: Any = T.StridedTensor(
        quant_scales_shape, quant_scales_strides, "uint8"
    )
    indices_type: Any = T.StridedTensor(indices_shape, indices_strides, indices_dtype)
    extra_kv_nope_type: Any = T.StridedTensor(
        extra_kv_nope_shape, extra_kv_nope_strides, kv_latent_dtype
    )
    extra_kv_rope_type: Any = T.StridedTensor(
        extra_kv_rope_shape, extra_kv_rope_strides, dtype
    )
    extra_quant_scales_type: Any = T.StridedTensor(
        extra_quant_scales_shape, extra_quant_scales_strides, "uint8"
    )
    extra_indices_type: Any = T.StridedTensor(
        extra_indices_shape, extra_indices_strides, indices_dtype
    )

    padded_head_kv = max(tilelang.math.next_power_of_2(head_kv), block_m)
    if padded_head_kv != head_kv:
        assert kv_group == 1

    if head_kv > block_m:
        assert head_kv % block_m == 0, "head_kv should be a multiple of block_m"
        head_repeats = head_kv // block_m
    else:
        head_repeats = 1

    heads_per_block = padded_head_kv if head_repeats == 1 else block_m
    dsa_combine = make_scheduled_decode_combine(
        batch=batch,
        seq_len=seq_len,
        num_heads=num_heads,
        dim=dim,
        num_mp_parts=num_mp_parts,
        dtype=dtype,
        accum_dtype=accum_dtype,
        max_nums_splits=max_nums_splits,
        has_attn_sink=has_attn_sink,
        max_lse_init=-(2**30),
    )

    @T.macro
    def update_online_softmax_mp32(
        acc_s,
        acc_s_cast,
        s_shared,
        m_i,
        m_i_prev,
        sumexp,
        sumexp_i,
        alpha_local,
        alpha_shared,
        acc_o_l_0,
        acc_o_l_1,
    ):
        """Online softmax with a linear SG16 score tile.

        The MP31 helper writes ``[H, 64]`` through a 64-to-8x8 score
        permutation.  PH1S consumes N=64 directly, so the fragment is copied
        without that permutation and the PV descriptor is reannotated SG16.
        """
        T.copy(m_i, m_i_prev)
        T.reduce_max(acc_s, m_i, dim=1, clear=False)
        for h_i in T.Parallel(heads_per_block):
            m_i[h_i] = T.max(m_i_prev[h_i], m_i[h_i])
            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
        for h_i, bi_i in T.Parallel(heads_per_block, block_i):
            acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
        T.reduce_sum(acc_s, sumexp_i, dim=1)
        for h_i in T.Parallel(heads_per_block):
            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
            alpha_shared[h_i] = alpha_local[h_i]
        for h_i, d_i in T.Parallel(heads_per_block, 128):
            acc_o_l_0[h_i, d_i] *= alpha_local[h_i]
            acc_o_l_1[h_i, d_i] *= alpha_local[h_i]
        T.copy(acc_s, acc_s_cast)
        T.copy(acc_s_cast, s_shared)

    finalize_left = make_scheduled_decode_finalize_left(
        h_per_block=heads_per_block,
        out_width=128,
        num_heads=num_heads,
        accum_dtype=accum_dtype,
        has_attn_sink=has_attn_sink,
        use_strict_valid=True,
        add_denominator_epsilon=False,
        sink_only_unsplit=True,
        sink_invalid_zero=True,
        wait_before_final=False,
        l0_start=0,
        l1_start=128,
        out_dtype=dtype,
        guard_invalid_heads=head_kv < block_m,
    )
    finalize_right = make_scheduled_decode_finalize_right(
        h_per_block=heads_per_block,
        out_width=128,
        num_heads=num_heads,
        has_attn_sink=has_attn_sink,
        wait_after_scale=False,
        r0_start=256,
        r1_start=384,
        out_dtype=dtype,
        guard_invalid_heads=head_kv < block_m,
    )

    @T.macro
    def stage_value_shared_mp32(
        v_shared,
        kv_reg,
        ldg_ty,
        ldg_tx,
        u_start,
        ldg_ty_count,
        rows_per_thread,
    ):
        """Stage a linear 64x128 V half under the SG16 PV descriptor."""
        T.annotate_layout(
            {
                v_shared[:, :]: tilelang.layout.make_sqmma_swizzled_layout(
                    v_shared[:, :],
                    continuity=128,
                    k_major=False,
                    swizzle_granularity=16,
                )
            },
            allow_reannotation=True,
            allow_buffer_region=True,
        )
        for r in T.unroll(rows_per_thread):
            for u in T.unroll(2):
                for v in T.vectorized(8):
                    row = ldg_ty + r * ldg_ty_count
                    v_shared[
                        row,
                        64 * u + ldg_tx * 8 + v,
                    ] = kv_reg[r * 32 + (u + u_start) * 8 + v]

    @T.macro
    def load_indices_mp32(
        indices,
        b_i,
        s_i,
        g_i,
        block_index,
        topk_length,
        seq_len_kv_local,
        ldg_ty,
        ldg_tx,
        phase,
        kperm_indices_local,
        kperm_mask_local,
        is_kv_valid,
        kv_indices,
        bar_kv_mask_free,
    ):
        # Producer lanes own two rows each.  Keep token and shared-row order
        # linear; the MP31 64->8x8 index permutation is not legal on MP32.
        for r in T.unroll(4):
            row = r * 16 + ldg_ty
            token_pos = block_index * block_i + row
            kperm_indices_local[r] = indices[b_i, s_i, g_i, token_pos]
            kperm_mask_local[r] = (
                kperm_indices_local[r] >= 0
                and kperm_indices_local[r] < seq_len_kv_local
                and token_pos < topk_length
            )
            kperm_indices_local[r] = T.if_then_else(
                kperm_mask_local[r],
                kperm_indices_local[r],
                seq_len_kv_local,
            )
        T.barrier_wait(bar_kv_mask_free, (phase & 1) ^ 1)
        if ldg_tx == 0:
            for r in T.unroll(4):
                row = r * 16 + ldg_ty
                is_kv_valid[row] = kperm_mask_local[r]
                kv_indices[row] = kperm_indices_local[r]

    pv_gemm_policy = T.GemmWarpPolicy.FullRow
    # BM16 lowers to synchronous PH1S WMMA, whose backend requires wg_wait=0.
    # BM64 uses SQMMA and keeps the explicit commit/wait grouping below.
    use_manual_sqmma_grouping = block_m == 64
    gemm_wg_wait = -1 if use_manual_sqmma_grouping else 0

    @T.macro
    def load_v4_paged_kv_block(
        kv_rope,
        quant_scales,
        rope_robust_desc_arg,
        scale_robust_desc_arg,
        page_size,
        kperm_indices_local,
        kv_indices,
        kv_shared_l,
        kv_shared_r,
        quant_shared,
        ldg_ty,
        ldg_tx,
        ldg_scale_ty,
        ldg_scale_tx,
        phase,
        bar_kv0_free,
        bar_kv_mask_ready,
        bar_indices_ready,
        bar_kv0_ready,
        bar_kv1_free,
        bar_kv1_ready,
    ):
        T.barrier_wait(bar_kv0_free, (phase & 1) ^ 1)
        for r in T.unroll(4):
            for u in T.unroll(4):
                for v in T.vectorized(4):
                    T.copy(
                        kv_rope[
                            kperm_indices_local[r] // page_size,
                            (kperm_indices_local[r] % page_size)
                            * (scale_bytes_offset // 2)
                            + 32 * u
                            + ldg_tx * 4
                            + v,
                        ],
                        kv_shared_l[r * 16 + ldg_ty, 64 * u + ldg_tx * 8 + v],
                        force_async_copy=True,
                        src_robust_desc=rope_robust_desc_arg,
                    )
        T.lma_wait()
        T.barrier_arrive(bar_kv_mask_ready)
        T.barrier_arrive(bar_indices_ready)
        T.barrier_wait(bar_indices_ready, phase & 1)

        for c in T.vectorized(4):
            T.copy(
                quant_scales[
                    kv_indices[ldg_scale_ty] // page_size,
                    page_size * scale_bytes_offset
                    + (kv_indices[ldg_scale_ty] % page_size) * 8
                    + ldg_scale_tx * 4
                    + c,
                ],
                quant_shared[ldg_scale_ty, ldg_scale_tx * 4 + c],
                force_async_copy=True,
                src_robust_desc=scale_robust_desc_arg,
            )
        T.ptx_commit_group()
        T.ptx_wait_group(0)
        T.barrier_arrive(bar_kv0_ready)

        T.barrier_wait(bar_kv1_free, (phase & 1) ^ 1)
        for r in T.unroll(4):
            for u in T.unroll(3):
                for v in T.vectorized(4):
                    T.copy(
                        kv_rope[
                            kperm_indices_local[r] // page_size,
                            (kperm_indices_local[r] % page_size)
                            * (scale_bytes_offset // 2)
                            + dim // 4
                            + 32 * u
                            + ldg_tx * 4
                            + v,
                        ],
                        kv_shared_r[r * 16 + ldg_ty, 64 * u + ldg_tx * 8 + v],
                        force_async_copy=True,
                        src_robust_desc=rope_robust_desc_arg,
                    )

        if use_8byte_kv_loads:
            for r in T.unroll(4):
                for v in T.vectorized(4):
                    T.copy(
                        kv_rope[
                            kperm_indices_local[r] // page_size,
                            (kperm_indices_local[r] % page_size)
                            * (scale_bytes_offset // 2)
                            + rope_bytes_offset // 2
                            + ldg_tx * 8
                            + v,
                        ],
                        kv_shared_r[r * 16 + ldg_ty, 64 * 3 + ldg_tx * 8 + v],
                        force_async_copy=True,
                        src_robust_desc=rope_robust_desc_arg,
                    )
                for v in T.vectorized(4):
                    T.copy(
                        kv_rope[
                            kperm_indices_local[r] // page_size,
                            (kperm_indices_local[r] % page_size)
                            * (scale_bytes_offset // 2)
                            + rope_bytes_offset // 2
                            + ldg_tx * 8
                            + 4
                            + v,
                        ],
                        kv_shared_r[r * 16 + ldg_ty, 64 * 3 + ldg_tx * 8 + 4 + v],
                        force_async_copy=True,
                        src_robust_desc=rope_robust_desc_arg,
                    )
        else:
            for r in T.unroll(4):
                for v in T.vectorized(8):
                    T.copy(
                        kv_rope[
                            kperm_indices_local[r] // page_size,
                            (kperm_indices_local[r] % page_size)
                            * (scale_bytes_offset // 2)
                            + rope_bytes_offset // 2
                            + ldg_tx * 8
                            + v,
                        ],
                        kv_shared_r[r * 16 + ldg_ty, 64 * 3 + ldg_tx * 8 + v],
                        force_async_copy=True,
                        src_robust_desc=rope_robust_desc_arg,
                    )
        T.ptx_commit_group()
        T.ptx_wait_group(0)
        T.barrier_arrive(bar_kv1_ready)

    def make_v41_loader(fp4):
        @T.macro
        def load_v41_paged_kv_block(
            kv_rope,
            quant_scales,
            rope_robust_desc_arg,
            scale_robust_desc_arg,
            page_size,
            kperm_indices_local,
            kv_indices,
            kv_shared_l,
            kv_shared_r,
            quant_shared,
            ldg_ty,
            ldg_tx,
            ldg_scale_ty,
            ldg_scale_tx,
            phase,
            bar_kv0_free,
            bar_kv_mask_ready,
            bar_indices_ready,
            bar_kv0_ready,
            bar_kv1_free,
            bar_kv1_ready,
        ):
            data_bytes = 256 if fp4 else 512
            sf_bytes = 32 if fp4 else 16
            copy_width = 2 if fp4 else 4

            T.barrier_wait(bar_kv0_free, (phase & 1) ^ 1)
            for r in T.unroll(4):
                for u in T.unroll(4):
                    for v in T.vectorized(copy_width):
                        T.copy(
                            kv_rope[
                                kperm_indices_local[r] // page_size,
                                (kperm_indices_local[r] % page_size)
                                * (data_bytes // 2)
                                + (u * 8 + ldg_tx) * copy_width
                                + v,
                            ],
                            kv_shared_l[
                                r * 16 + ldg_ty,
                                u * 64 + ldg_tx * 8 + v,
                            ],
                            force_async_copy=True,
                            src_robust_desc=rope_robust_desc_arg,
                        )
            T.lma_wait()
            T.barrier_arrive(bar_kv_mask_ready)
            T.barrier_arrive(bar_indices_ready)
            T.barrier_wait(bar_indices_ready, phase & 1)

            sf_page = kv_indices[ldg_scale_ty] // page_size
            sf_token = kv_indices[ldg_scale_ty] % page_size
            for c in T.vectorized(sf_bytes // 2):
                T.copy(
                    quant_scales[
                        sf_page,
                        page_size * data_bytes
                        + sf_token * sf_bytes
                        + ldg_scale_tx * (sf_bytes // 2)
                        + c,
                    ],
                    quant_shared[
                        ldg_scale_ty,
                        ldg_scale_tx * (sf_bytes // 2) + c,
                    ],
                    force_async_copy=True,
                    src_robust_desc=scale_robust_desc_arg,
                )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            T.barrier_arrive(bar_kv0_ready)

            T.barrier_wait(bar_kv1_free, (phase & 1) ^ 1)
            for r in T.unroll(4):
                for u in T.unroll(4):
                    for v in T.vectorized(copy_width):
                        T.copy(
                            kv_rope[
                                kperm_indices_local[r] // page_size,
                                (kperm_indices_local[r] % page_size)
                                * (data_bytes // 2)
                                + data_bytes // 4
                                + (u * 8 + ldg_tx) * copy_width
                                + v,
                            ],
                            kv_shared_r[
                                r * 16 + ldg_ty,
                                u * 64 + ldg_tx * 8 + v,
                            ],
                            force_async_copy=True,
                            src_robust_desc=rope_robust_desc_arg,
                        )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            T.barrier_arrive(bar_kv1_ready)

        return load_v41_paged_kv_block

    load_main_kv_block = (
        load_v4_paged_kv_block if cache_format == "v4" else make_v41_loader(False)
    )
    load_extra_kv_block = (
        load_v4_paged_kv_block
        if extra_cache_format == "v4"
        else make_v41_loader(extra_cache_format == "v41_fp4")
    )

    def make_v41_dequantizer(fp4):
        @T.macro
        def dequantize_panel(
            panel, scales, values, ldg_ty, ldg_tx, rows_per_thread, ty_count, col_base
        ):
            raw = T.alloc_local([rows_per_thread * 16], "bfloat16")
            bytes_view = T.view(raw, [rows_per_thread * 32], "uint8")
            fp8_view = T.view(raw, [rows_per_thread * 32], "float8_e4m3")
            T.annotate_layout(
                {
                    panel: tilelang.layout.make_sqmma_swizzled_layout(
                        panel, k_major=True
                    )
                },
                allow_reannotation=True,
            )
            if fp4:
                for r in T.unroll(rows_per_thread):
                    for u in T.unroll(4):
                        for v in T.vectorized(2):
                            raw[r * 16 + u * 4 + v] = panel[
                                ldg_ty + r * ty_count,
                                u * 64 + ldg_tx * 8 + v,
                            ]
            else:
                for r in T.unroll(rows_per_thread):
                    for u in T.unroll(4):
                        for v in T.vectorized(4):
                            raw[r * 16 + u * 4 + v] = panel[
                                ldg_ty + r * ty_count,
                                u * 64 + ldg_tx * 8 + v,
                            ]
            T.lma_wait()
            local_scales = T.alloc_local([rows_per_thread, 4], "float32")
            half_values = T.view(values, [rows_per_thread * 32], "float16")
            for r in T.unroll(rows_per_thread):
                for u in T.unroll(4):
                    if fp4:
                        sf_fp4 = scales[
                            ldg_ty + r * ty_count,
                            col_base // 16 + u * 4 + ldg_tx // 2,
                        ]
                        local_scales[r, u] = T.Cast(
                            "float32", T.reinterpret("float8_e4m3", sf_fp4)
                        )
                    else:
                        sf_fp8 = scales[
                            ldg_ty + r * ty_count,
                            col_base // 32 + u * 2 + ldg_tx // 4,
                        ]
                        bits = (T.Cast("int32", sf_fp8) << 23) | T.if_then_else(
                            (sf_fp8 == 0) | (sf_fp8 == 255), 1 << 22, 0
                        )
                        local_scales[r, u] = T.reinterpret("float32", bits)
            if fp4:
                for r in T.unroll(rows_per_thread):
                    for u in T.unroll(4):
                        for v in T.vectorized(8):
                            code = (
                                T.Cast(
                                    "int32",
                                    bytes_view[r * 32 + u * 8 + v // 2],
                                )
                                >> ((v % 2) * 4)
                            ) & 15
                            magnitude = code & 7
                            half_bits = ((code & 8) << 12) | T.if_then_else(
                                magnitude < 2,
                                magnitude * 0x3800,
                                (magnitude + 28) << 9,
                            )
                            half_values[r * 32 + u * 8 + v] = T.reinterpret(
                                "float16", T.Cast("uint16", half_bits)
                            )
                        for v in T.vectorized(8):
                            values[r * 32 + u * 8 + v] = T.Cast(
                                "bfloat16",
                                half_values[r * 32 + u * 8 + v]
                                * local_scales[r, u],
                            )
                        T.sched_boundary()
            else:
                for r in T.unroll(rows_per_thread):
                    for u in T.unroll(4):
                        for v in T.vectorized(8):
                            half_values[r * 32 + u * 8 + v] = fp8_view[
                                r * 32 + u * 8 + v
                            ]
                for r in T.unroll(rows_per_thread):
                    for u in T.unroll(4):
                        for v in T.vectorized(8):
                            values[r * 32 + u * 8 + v] = T.Cast(
                                "bfloat16",
                                half_values[r * 32 + u * 8 + v]
                                * local_scales[r, u],
                            )
            for r in T.unroll(rows_per_thread):
                for u in T.unroll(4):
                    for v in T.vectorized(8):
                        panel[
                            ldg_ty + r * ty_count,
                            u * 64 + ldg_tx * 8 + v,
                        ] = values[r * 32 + u * 8 + v]
            T.lma_wait()

        return dequantize_panel

    dequantize_v41_fp8_panel = make_v41_dequantizer(False)
    dequantize_v41_fp4_panel = make_v41_dequantizer(True)

    @T.macro
    def dequantize_v41_panel(
        panel,
        scales,
        values,
        ldg_ty,
        ldg_tx,
        rows_per_thread,
        ty_count,
        col_base,
        fp4,
    ):
        if fp4:
            dequantize_v41_fp4_panel(
                panel,
                scales,
                values,
                ldg_ty,
                ldg_tx,
                rows_per_thread,
                ty_count,
                col_base,
            )
        else:
            dequantize_v41_fp8_panel(
                panel,
                scales,
                values,
                ldg_ty,
                ldg_tx,
                rows_per_thread,
                ty_count,
                col_base,
            )

    @T.macro
    def apply_q_rope_in_shared(
        q_shared_r,
        q_rope_positions,
        q_rope_cos_sin,
        batch_idx,
        thread_idx,
        active_threads,
    ):
        # The right shared tile holds head dimensions [256, 512). DSV4 Q RoPE
        # rotates only the adjacent-pair tail [448, 512).
        if has_q_rope:
            position = q_rope_positions[batch_idx]
            even_value = T.alloc_local([1], "float32")
            odd_value = T.alloc_local([1], "float32")
            total_pairs = heads_per_block * 32
            pairs_per_thread = (total_pairs + active_threads - 1) // active_threads
            for iteration in T.unroll(pairs_per_thread):
                linear = thread_idx + iteration * active_threads
                if linear < total_pairs:
                    head = linear // 32
                    pair = linear % 32
                    even_idx = 192 + pair * 2
                    even_value[0] = T.Cast(
                        "float32", q_shared_r[head, even_idx]
                    )
                    odd_value[0] = T.Cast(
                        "float32", q_shared_r[head, even_idx + 1]
                    )
                    cosine = q_rope_cos_sin[position, pair * 2]
                    sine = q_rope_cos_sin[position, pair * 2 + 1]
                    q_shared_r[head, even_idx] = T.Cast(
                        dtype,
                        even_value[0] * cosine - odd_value[0] * sine,
                    )
                    q_shared_r[head, even_idx + 1] = T.Cast(
                        dtype,
                        even_value[0] * sine + odd_value[0] * cosine,
                    )

    @T.prim_func
    def dsa_decode(
        q: q_type,
        kv_nope: kv_nope_type,
        kv_rope: kv_rope_type,
        quant_scales: quant_scales_type,
        indices: indices_type,
        topk_length: T.Tensor([batch], indices_dtype),  # type: ignore
        extra_kv_nope: extra_kv_nope_type,
        extra_kv_rope: extra_kv_rope_type,
        extra_quant_scales: extra_quant_scales_type,
        extra_indices: extra_indices_type,
        extra_topk_length: T.Tensor([batch], indices_dtype),  # type: ignore
        attn_sink: T.Tensor([num_heads], accum_dtype),  # type: ignore
        q_rope_positions: T.Tensor([batch], q_position_dtype),  # type: ignore
        q_rope_cos_sin: T.Tensor([rope_positions, 64], "float32"),  # type: ignore
        tile_scheduler_metadata: T.Tensor([num_mp_parts, 8], T.int32),  # type: ignore
        num_splits: T.Tensor([batch + 1], T.int32),  # type: ignore
        glse: T.Tensor([batch + num_mp_parts, seq_len, num_heads], accum_dtype),  # type: ignore
        output_partial: T.Tensor(
            [batch + num_mp_parts, seq_len, num_heads, dim], accum_dtype
        ),  # type: ignore
        output: T.Tensor(o_shape, dtype),  # type: ignore
        lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        q_cosize: T.int64,
        kv_nope_cosize: T.int64,
        kv_rope_cosize: T.int64,
        quant_scales_cosize: T.int64,
        extra_kv_nope_cosize: T.int64,
        extra_kv_rope_cosize: T.int64,
        extra_quant_scales_cosize: T.int64,
    ):
        # V4 scheduled split main kernel. It follows the FlashMLA scheduler
        # contract: each program consumes one metadata part and either writes the
        # final output for unsplit batches or partial output/LSE for split ones.
        with T.Kernel(
            seq_len * head_repeats, kv_group, num_mp_parts, threads=threads
        ) as (bx, by, bz):
            T.assume(q_stride_b % 8 == 0)
            T.assume(q_stride_s % 8 == 0)
            T.assume(q_stride_h % 8 == 0)
            T.assume(kv_nope_stride_block % 8 == 0)
            T.assume(kv_rope_stride_block % 4 == 0)
            T.assume(quant_scales_stride_block % 8 == 0)
            T.assume(indices_stride_b % 8 == 0)
            T.assume(indices_stride_s % 8 == 0)
            T.assume(indices_stride_g % 8 == 0)
            T.assume(extra_kv_nope_stride_block % 8 == 0)
            T.assume(extra_kv_rope_stride_block % 4 == 0)
            T.assume(extra_quant_scales_stride_block % 8 == 0)
            T.assume(extra_indices_stride_b % 8 == 0)
            T.assume(extra_indices_stride_s % 8 == 0)
            T.assume(extra_indices_stride_g % 8 == 0)
            q_shared_l = T.alloc_shared([heads_per_block, 256], dtype)
            q_shared_r = T.alloc_shared([heads_per_block, 256], dtype)
            kv_shared_l = T.alloc_shared([block_i, 256], dtype)
            kv_shared_r = T.alloc_shared([block_i, 256], dtype)
            v_shared_0 = T.alloc_shared([block_i, 128], dtype)
            v_shared_1 = T.alloc_shared([block_i, 128], dtype)
            if use_three_v_buffers:
                v_shared_2 = T.alloc_shared([block_i, 128], dtype)
            else:
                v_shared_2 = v_shared_0

            scores_shared = T.alloc_shared([heads_per_block, block_i], dtype)
            sum_exp_inv_shared = T.alloc_shared([heads_per_block], accum_dtype)
            sink_scale_shared = T.alloc_shared([heads_per_block], accum_dtype)
            alpha_shared = T.alloc_shared([heads_per_block], accum_dtype)
            is_kv_valid = T.alloc_shared([block_i], "bool", scope="shared")
            kv_indices = T.alloc_shared([block_i], "int32", scope="shared")
            quant_shared = T.alloc_shared([block_i, 32], "uint8")

            bar_kv_mask_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_kv_mask_free = T.alloc_barrier(arrive_count=consumer0_gemm_threads)
            bar_q = T.alloc_barrier(arrive_count=producer_start)
            bar_indices_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_kv0_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_kv1_ready = T.alloc_barrier(arrive_count=producer_threads)
            bar_kv0_quant_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_kv1_quant_ready = T.alloc_barrier(arrive_count=consumer1_threads)

            bar_kv0_free = T.alloc_barrier(arrive_count=consumer0_gemm_threads)
            bar_kv1_free = T.alloc_barrier(arrive_count=consumer0_gemm_threads)

            bar_vl0_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_vl1_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_vr0_ready = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_vr1_ready = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_vl0_free = T.alloc_barrier(arrive_count=consumer0_gemm_threads)
            bar_vl1_free = T.alloc_barrier(arrive_count=consumer0_gemm_threads)

            bar_p_ready = T.alloc_barrier(arrive_count=consumer0_gemm_threads)
            bar_final = T.alloc_barrier(arrive_count=[consumer0_gemm_threads])

            q_robust_desc = T.make_robust_desc(
                T.address_of(q[0, 0, 0, 0]),
                q_cosize * dtype_bytes,
            )
            nope_robust_desc = T.make_robust_desc(
                T.address_of(kv_nope[0, 0]),
                kv_nope_cosize,
            )
            rope_robust_desc = T.make_robust_desc(
                T.address_of(kv_rope[0, 0]),
                kv_rope_cosize * dtype_bytes,
            )
            scale_robust_desc = T.make_robust_desc(
                T.address_of(quant_scales[0, 0]),
                quant_scales_cosize,
            )
            extra_nope_robust_desc = T.make_robust_desc(
                T.address_of(extra_kv_nope[0, 0]),
                extra_kv_nope_cosize,
            )
            extra_rope_robust_desc = T.make_robust_desc(
                T.address_of(extra_kv_rope[0, 0]),
                extra_kv_rope_cosize * dtype_bytes,
            )
            extra_scale_robust_desc = T.make_robust_desc(
                T.address_of(extra_quant_scales[0, 0]),
                extra_quant_scales_cosize,
            )

            T.sync_threads()

            g_i = by
            s_i = bx if head_repeats == 1 else (bx // head_repeats)
            h0 = g_i * padded_head_kv + (
                0 if head_repeats == 1 else (bx % head_repeats) * block_m
            )
            h1 = h0 + heads_per_block
            tid = T.get_thread_binding()

            begin_idx = tile_scheduler_metadata[bz, 0]
            sched_begin_block_idx = tile_scheduler_metadata[bz, 1]
            end_idx = tile_scheduler_metadata[bz, 2]
            sched_end_block_idx = tile_scheduler_metadata[bz, 3]
            begin_n_split_idx = tile_scheduler_metadata[bz, 4]
            phase_count = T.alloc_local([1], T.int32)
            T.fill(phase_count, 0)

            for b_i in range(begin_idx, end_idx + 1, 1):
                tir.call_extern("void", "__musa_loop_transparent_outermost")
                start_block_idx = T.alloc_var(T.int32)
                end_block_idx = T.alloc_var(T.int32)
                n_split_idx = T.alloc_var(T.int32)
                dynamic_main_blocks = T.alloc_var(T.int32)
                dynamic_total_blocks = T.alloc_var(T.int32)
                if has_topk_length:
                    dynamic_main_blocks = T.max(T.ceildiv(topk_length[b_i], block_i), 1)
                else:
                    dynamic_main_blocks = T.max(T.ceildiv(topk, block_i), 1)
                dynamic_total_blocks = dynamic_main_blocks
                if has_extra:
                    if has_extra_topk_length:
                        dynamic_total_blocks += T.ceildiv(
                            extra_topk_length[b_i], block_i
                        )
                    else:
                        dynamic_total_blocks += T.ceildiv(extra_topk, block_i)
                start_block_idx = T.if_then_else(
                    b_i == begin_idx, sched_begin_block_idx, 0
                )
                end_block_idx = T.if_then_else(
                    b_i == end_idx, sched_end_block_idx, dynamic_total_blocks
                )
                n_split_idx = T.if_then_else(b_i == begin_idx, begin_n_split_idx, 0)
                is_unsplit = (num_splits[b_i + 1] - num_splits[b_i]) == 1

                if tid < producer_start:
                    T.copy(
                        q[b_i, s_i, h0:h1, 0:256],
                        q_shared_l,
                        src_robust_desc=q_robust_desc,
                    )
                    T.copy(
                        q[b_i, s_i, h0:h1, 256:512],
                        q_shared_r,
                        src_robust_desc=q_robust_desc,
                    )
                    T.ptx_commit_group()
                    T.ptx_wait_group(0)
                    apply_q_rope_in_shared(
                        q_shared_r,
                        q_rope_positions,
                        q_rope_cos_sin,
                        b_i,
                        tid,
                        producer_start,
                    )
                    T.barrier_arrive(bar_q)
                    T.barrier_wait(bar_q, (b_i - begin_idx) & 1)

                if tid < consumer0_gemm_threads:
                    sumexp = T.alloc_fragment([heads_per_block], accum_dtype)
                    sumexp_i = T.alloc_fragment([heads_per_block], accum_dtype)
                    sumexp_inv = T.alloc_fragment([heads_per_block], accum_dtype)
                    alpha_local = T.alloc_fragment([heads_per_block], accum_dtype)
                    m_i = T.alloc_fragment([heads_per_block], accum_dtype)
                    m_i_prev = T.alloc_fragment([heads_per_block], accum_dtype)
                    acc_s = T.alloc_fragment([heads_per_block, block_i], accum_dtype)
                    acc_s_cast = T.alloc_fragment([heads_per_block, block_i], dtype)
                    acc_o_l_0 = T.alloc_fragment([heads_per_block, 128], accum_dtype)
                    acc_o_l_1 = T.alloc_fragment([heads_per_block, 128], accum_dtype)
                    kv_reg_l = T.alloc_local([consumer0_rows_per_thread * 32], dtype)
                    kv_reg_l_fp16 = T.view(
                        kv_reg_l, [consumer0_rows_per_thread * 32], T.float16
                    )
                    kv_reg_l_bf16_load = T.alloc_local(
                        [consumer0_rows_per_thread * 16], T.bfloat16
                    )
                    kv_reg_l_fp8 = T.view(
                        kv_reg_l_bf16_load,
                        [consumer0_rows_per_thread * 32],
                        kv_latent_dtype,
                    )
                    quant_u8_l = T.alloc_local([consumer0_rows_per_thread, 4], "uint8")
                    quant_local_l = T.alloc_local(
                        [consumer0_rows_per_thread, 4], T.float32
                    )
                    c0_helper_ldg_tx = tid % 8
                    c0_helper_ldg_ty = tid // 8
                    T.fill(sumexp, 0)
                    T.fill(m_i, -(2**30))
                    T.fill(acc_o_l_0, 0)
                    T.fill(acc_o_l_1, 0)

                    for i_i in range(start_block_idx, end_block_idx):
                        T.barrier_wait(bar_kv0_ready, (phase_count[0] & 1))

                        if cache_format != "v4":
                            dequantize_v41_panel(
                                kv_shared_l,
                                quant_shared,
                                kv_reg_l,
                                c0_helper_ldg_ty,
                                c0_helper_ldg_tx,
                                consumer0_rows_per_thread,
                                consumer0_ldg_ty_count,
                                0,
                                (i_i >= dynamic_main_blocks)
                                if extra_cache_format == "v41_fp4"
                                else False,
                            )
                        else:
                            for r in T.unroll(consumer0_rows_per_thread):
                                row = c0_helper_ldg_ty + r * consumer0_ldg_ty_count
                                T.copy(quant_shared[row, 0:4], quant_u8_l[r, :])

                            T.annotate_layout(
                                {
                                    kv_shared_l[
                                        :, :
                                    ]: tilelang.layout.make_sqmma_swizzled_layout(
                                        kv_shared_l[:, :],
                                        k_major=True,
                                    )
                                },
                                allow_reannotation=True,
                                allow_buffer_region=True,
                            )

                            for r in T.unroll(consumer0_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(4):
                                        row = (
                                            c0_helper_ldg_ty
                                            + r * consumer0_ldg_ty_count
                                        )
                                        kv_reg_l_bf16_load[r * 16 + u * 4 + v] = (
                                            kv_shared_l[
                                                row,
                                                64 * u + c0_helper_ldg_tx * 8 + v,
                                            ]
                                        )
                            T.lma_wait()
                            for r in T.unroll(consumer0_rows_per_thread):
                                for u in T.unroll(4):
                                    quant_local_l[r, u] = T.reinterpret(
                                        "float32",
                                        T.Cast("int32", quant_u8_l[r, u]) << 23,
                                    )
                            for r in T.unroll(consumer0_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(8):
                                        idx = r * 32 + u * 8 + v
                                        kv_reg_l_fp16[idx] = kv_reg_l_fp8[idx]
                            for r in T.unroll(consumer0_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(8):
                                        idx = r * 32 + u * 8 + v
                                        kv_reg_l[idx] = T.Cast(
                                            dtype,
                                            kv_reg_l_fp16[idx] * quant_local_l[r, u],
                                        )
                            for r in T.unroll(consumer0_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(8):
                                        row = (
                                            c0_helper_ldg_ty
                                            + r * consumer0_ldg_ty_count
                                        )
                                        kv_shared_l[
                                            row,
                                            64 * u + c0_helper_ldg_tx * 8 + v,
                                        ] = kv_reg_l[r * 32 + u * 8 + v]
                            for r in T.unroll(consumer0_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(4):
                                        row = (
                                            c0_helper_ldg_ty
                                            + r * consumer0_ldg_ty_count
                                        )
                                        kv_reg_l_bf16_load[
                                            r * 16 + (u + 2) * 4 + v
                                        ] = kv_shared_l[
                                            row,
                                            64 * (u + 2)
                                            + c0_helper_ldg_tx * 8
                                            + v,
                                        ]
                            T.lma_wait()
                            for r in T.unroll(consumer0_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(8):
                                        idx = r * 32 + (u + 2) * 8 + v
                                        kv_reg_l_fp16[idx] = kv_reg_l_fp8[idx]
                            for r in T.unroll(consumer0_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(8):
                                        idx = r * 32 + (u + 2) * 8 + v
                                        kv_reg_l[idx] = T.Cast(
                                            dtype,
                                            kv_reg_l_fp16[idx]
                                            * quant_local_l[r, u + 2],
                                        )
                            for r in T.unroll(consumer0_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(8):
                                        idx = r * 32 + (u + 2) * 8 + v
                                        row = (
                                            c0_helper_ldg_ty
                                            + r * consumer0_ldg_ty_count
                                        )
                                        kv_shared_l[
                                            row,
                                            64 * (u + 2)
                                            + c0_helper_ldg_tx * 8
                                            + v,
                                        ] = kv_reg_l[idx]
                        T.lma_wait()
                        T.barrier_arrive(bar_kv0_quant_ready)

                        T.barrier_wait(bar_kv_mask_ready, (phase_count[0] & 1))
                        for h_i, bi_i in T.Parallel(heads_per_block, block_i):
                            acc_s[h_i, bi_i] = T.if_then_else(
                                is_kv_valid[bi_i],
                                0,
                                -(2**30),
                            )
                        T.lma_wait()
                        T.barrier_arrive(bar_kv_mask_free)

                        T.barrier_wait(bar_kv0_quant_ready, (phase_count[0] & 1))
                        T.gemm(
                            q_shared_l,
                            kv_shared_l[:, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=gemm_wg_wait,
                        )
                        if use_manual_sqmma_grouping:
                            T.warpgroup_commit_batch()
                            T.warpgroup_wait(0)
                        T.barrier_arrive(bar_kv0_free)

                        T.barrier_wait(bar_kv1_quant_ready, (phase_count[0] & 1))
                        T.annotate_layout(
                            {
                                kv_shared_r[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_r[:, :],
                                    k_major=True,
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        T.gemm(
                            q_shared_r,
                            kv_shared_r[:, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=gemm_wg_wait,
                        )
                        if use_manual_sqmma_grouping:
                            T.warpgroup_commit_batch()
                            T.warpgroup_wait(0)
                        T.barrier_arrive(bar_kv1_free)

                        update_online_softmax_mp32(
                            acc_s,
                            acc_s_cast,
                            scores_shared,
                            m_i,
                            m_i_prev,
                            sumexp,
                            sumexp_i,
                            alpha_local,
                            alpha_shared,
                            acc_o_l_0,
                            acc_o_l_1,
                        )

                        T.lma_wait()
                        T.barrier_arrive(bar_p_ready)
                        stage_value_shared_mp32(
                            v_shared_0,
                            kv_reg_l,
                            c0_helper_ldg_ty,
                            c0_helper_ldg_tx,
                            0,
                            consumer0_ldg_ty_count,
                            consumer0_rows_per_thread,
                        )
                        T.lma_wait()
                        T.barrier_arrive(bar_vl0_ready)
                        T.barrier_wait(bar_vl0_ready, (phase_count[0] & 1))

                        T.sched_boundary()
                        T.gemm(
                            scores_shared,
                            v_shared_0,
                            acc_o_l_0,
                            policy=pv_gemm_policy,
                            wg_wait=gemm_wg_wait,
                        )
                        if use_manual_sqmma_grouping:
                            T.warpgroup_commit_batch()

                        T.sched_boundary()

                        stage_value_shared_mp32(
                            v_shared_1,
                            kv_reg_l,
                            c0_helper_ldg_ty,
                            c0_helper_ldg_tx,
                            2,
                            consumer0_ldg_ty_count,
                            consumer0_rows_per_thread,
                        )
                        if use_manual_sqmma_grouping:
                            T.warpgroup_wait(0)
                        T.barrier_arrive(bar_vl0_free)

                        T.lma_wait()
                        T.barrier_arrive(bar_vl1_ready)
                        T.barrier_wait(bar_vl1_ready, (phase_count[0] & 1))

                        T.sched_boundary()
                        T.gemm(
                            scores_shared,
                            v_shared_1,
                            acc_o_l_1,
                            policy=pv_gemm_policy,
                            wg_wait=gemm_wg_wait,
                        )
                        if use_manual_sqmma_grouping:
                            T.warpgroup_commit_batch()
                            T.warpgroup_wait(0)
                        T.sched_boundary()
                        T.barrier_arrive(bar_vl1_free)
                        phase_count[0] = phase_count[0] ^ 1

                    # The scheduled common expects a CTA-local sink fragment;
                    # build it once per batch from the global attn_sink.
                    sink_local = T.alloc_fragment([heads_per_block], accum_dtype)
                    if has_attn_sink:
                        for h_i in T.Parallel(heads_per_block):
                            if h0 + h_i < num_heads:
                                sink_local[h_i] = attn_sink[h0 + h_i]
                            else:
                                sink_local[h_i] = 0.0
                    finalize_left(
                        b_i,
                        s_i,
                        h0,
                        h1,
                        is_unsplit,
                        m_i,
                        sumexp,
                        sumexp_inv,
                        sum_exp_inv_shared,
                        sink_scale_shared,
                        acc_o_l_0,
                        acc_o_l_1,
                        output,
                        output_partial,
                        lse,
                        glse,
                        n_split_idx,
                        num_splits,
                        sink_local,
                        bar_final,
                        0,
                        sm_scale,
                    )
                elif tid < consumer0_threads:
                    kv_reg_l = T.alloc_local([consumer0_rows_per_thread * 32], dtype)
                    kv_reg_l_fp16 = T.view(
                        kv_reg_l, [consumer0_rows_per_thread * 32], T.float16
                    )
                    kv_reg_l_bf16_load = T.alloc_local(
                        [consumer0_rows_per_thread * 16], T.bfloat16
                    )
                    kv_reg_l_fp8 = T.view(
                        kv_reg_l_bf16_load,
                        [consumer0_rows_per_thread * 32],
                        kv_latent_dtype,
                    )
                    quant_u8_l = T.alloc_local([consumer0_rows_per_thread, 4], "uint8")
                    quant_local_l = T.alloc_local(
                        [consumer0_rows_per_thread, 4], T.float32
                    )
                    c0_ldg_tx = tid % 8
                    c0_ldg_ty = tid // 8

                    for i_i in range(start_block_idx, end_block_idx):
                        T.barrier_wait(bar_kv0_ready, (phase_count[0] & 1))

                        if cache_format != "v4":
                            dequantize_v41_panel(
                                kv_shared_l,
                                quant_shared,
                                kv_reg_l,
                                c0_ldg_ty,
                                c0_ldg_tx,
                                consumer0_rows_per_thread,
                                consumer0_ldg_ty_count,
                                0,
                                (i_i >= dynamic_main_blocks)
                                if extra_cache_format == "v41_fp4"
                                else False,
                            )
                        else:
                            for r in T.unroll(consumer0_rows_per_thread):
                                row = c0_ldg_ty + r * consumer0_ldg_ty_count
                                T.copy(quant_shared[row, 0:4], quant_u8_l[r, :])

                            T.annotate_layout(
                                {
                                    kv_shared_l[
                                        :, :
                                    ]: tilelang.layout.make_sqmma_swizzled_layout(
                                        kv_shared_l[:, :],
                                        k_major=True,
                                    )
                                },
                                allow_reannotation=True,
                                allow_buffer_region=True,
                            )

                            for r in T.unroll(consumer0_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(4):
                                        row = c0_ldg_ty + r * consumer0_ldg_ty_count
                                        kv_reg_l_bf16_load[r * 16 + u * 4 + v] = (
                                            kv_shared_l[
                                                row,
                                                64 * u + c0_ldg_tx * 8 + v,
                                            ]
                                        )
                            T.lma_wait()
                            for r in T.unroll(consumer0_rows_per_thread):
                                for u in T.unroll(4):
                                    quant_local_l[r, u] = T.reinterpret(
                                        "float32",
                                        T.Cast("int32", quant_u8_l[r, u]) << 23,
                                    )
                            for r in T.unroll(consumer0_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(8):
                                        idx = r * 32 + u * 8 + v
                                        kv_reg_l_fp16[idx] = kv_reg_l_fp8[idx]
                            for r in T.unroll(consumer0_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(8):
                                        idx = r * 32 + u * 8 + v
                                        kv_reg_l[idx] = T.Cast(
                                            dtype,
                                            kv_reg_l_fp16[idx] * quant_local_l[r, u],
                                        )
                            for r in T.unroll(consumer0_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(8):
                                        row = c0_ldg_ty + r * consumer0_ldg_ty_count
                                        kv_shared_l[
                                            row,
                                            64 * u + c0_ldg_tx * 8 + v,
                                        ] = kv_reg_l[r * 32 + u * 8 + v]
                            for r in T.unroll(consumer0_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(4):
                                        row = c0_ldg_ty + r * consumer0_ldg_ty_count
                                        kv_reg_l_bf16_load[
                                            r * 16 + (u + 2) * 4 + v
                                        ] = kv_shared_l[
                                            row,
                                            64 * (u + 2) + c0_ldg_tx * 8 + v,
                                        ]
                            T.lma_wait()
                            for r in T.unroll(consumer0_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(8):
                                        idx = r * 32 + (u + 2) * 8 + v
                                        kv_reg_l_fp16[idx] = kv_reg_l_fp8[idx]
                            for r in T.unroll(consumer0_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(8):
                                        idx = r * 32 + (u + 2) * 8 + v
                                        kv_reg_l[idx] = T.Cast(
                                            dtype,
                                            kv_reg_l_fp16[idx]
                                            * quant_local_l[r, u + 2],
                                        )
                            for r in T.unroll(consumer0_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(8):
                                        idx = r * 32 + (u + 2) * 8 + v
                                        row = c0_ldg_ty + r * consumer0_ldg_ty_count
                                        kv_shared_l[
                                            row,
                                            64 * (u + 2) + c0_ldg_tx * 8 + v,
                                        ] = kv_reg_l[idx]
                        T.lma_wait()
                        T.barrier_arrive(bar_kv0_quant_ready)

                        T.barrier_wait(bar_p_ready, (phase_count[0] & 1))
                        stage_value_shared_mp32(
                            v_shared_0,
                            kv_reg_l,
                            c0_ldg_ty,
                            c0_ldg_tx,
                            0,
                            consumer0_ldg_ty_count,
                            consumer0_rows_per_thread,
                        )
                        T.lma_wait()
                        T.barrier_arrive(bar_vl0_ready)
                        T.barrier_wait(bar_vl0_ready, (phase_count[0] & 1))

                        stage_value_shared_mp32(
                            v_shared_1,
                            kv_reg_l,
                            c0_ldg_ty,
                            c0_ldg_tx,
                            2,
                            consumer0_ldg_ty_count,
                            consumer0_rows_per_thread,
                        )
                        T.lma_wait()
                        T.barrier_arrive(bar_vl1_ready)
                        T.barrier_wait(bar_vl1_ready, (phase_count[0] & 1))
                        phase_count[0] = phase_count[0] ^ 1
                elif tid < producer_start:
                    acc_o_r_0 = T.alloc_fragment([heads_per_block, 128], accum_dtype)
                    acc_o_r_1 = T.alloc_fragment([heads_per_block, 128], accum_dtype)
                    kv_reg_r = T.alloc_local([consumer1_rows_per_thread * 32], dtype)
                    kv_reg_r_fp16 = T.view(
                        kv_reg_r, [consumer1_rows_per_thread * 32], T.float16
                    )
                    kv_reg_r_bf16_load = T.alloc_local(
                        [consumer1_rows_per_thread * 16], T.bfloat16
                    )
                    kv_reg_r_fp8 = T.view(
                        kv_reg_r_bf16_load,
                        [consumer1_rows_per_thread * 32],
                        kv_latent_dtype,
                    )
                    quant_u8_r = T.alloc_local([consumer1_rows_per_thread, 3], "uint8")
                    quant_local_r = T.alloc_local(
                        [consumer1_rows_per_thread, 3], T.float32
                    )
                    T.fill(acc_o_r_0, 0)
                    T.fill(acc_o_r_1, 0)
                    c1_ldg_tx = (tid - consumer1_start) % 8
                    c1_ldg_ty = (tid - consumer1_start) // 8

                    for i_i in range(start_block_idx, end_block_idx):
                        T.barrier_wait(bar_kv1_ready, (phase_count[0] & 1))

                        if cache_format != "v4":
                            dequantize_v41_panel(
                                kv_shared_r,
                                quant_shared,
                                kv_reg_r,
                                c1_ldg_ty,
                                c1_ldg_tx,
                                consumer1_rows_per_thread,
                                consumer1_ldg_ty_count,
                                256,
                                (i_i >= dynamic_main_blocks)
                                if extra_cache_format == "v41_fp4"
                                else False,
                            )
                        else:
                            for r in T.unroll(consumer1_rows_per_thread):
                                row = c1_ldg_ty + r * consumer1_ldg_ty_count
                                T.copy(quant_shared[row, 4:7], quant_u8_r[r, :])

                            T.annotate_layout(
                                {
                                    kv_shared_r[
                                        :, :
                                    ]: tilelang.layout.make_sqmma_swizzled_layout(
                                        kv_shared_r[:, :],
                                        k_major=True,
                                    )
                                },
                                allow_reannotation=True,
                                allow_buffer_region=True,
                            )
                            for r in T.unroll(consumer1_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(4):
                                        row = c1_ldg_ty + r * consumer1_ldg_ty_count
                                        kv_reg_r_bf16_load[r * 16 + u * 4 + v] = (
                                            kv_shared_r[
                                                row,
                                                64 * u + c1_ldg_tx * 8 + v,
                                            ]
                                        )
                            T.lma_wait()
                            for r in T.unroll(consumer1_rows_per_thread):
                                for u in T.unroll(3):
                                    quant_local_r[r, u] = T.reinterpret(
                                        "float32",
                                        T.Cast("int32", quant_u8_r[r, u]) << 23,
                                    )
                            for r in T.unroll(consumer1_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(8):
                                        idx = r * 32 + u * 8 + v
                                        kv_reg_r_fp16[idx] = kv_reg_r_fp8[idx]
                            for r in T.unroll(consumer1_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(8):
                                        idx = r * 32 + u * 8 + v
                                        kv_reg_r[idx] = T.Cast(
                                            dtype,
                                            kv_reg_r_fp16[idx] * quant_local_r[r, u],
                                        )
                            for r in T.unroll(consumer1_rows_per_thread):
                                for u in T.unroll(2):
                                    for v in T.vectorized(8):
                                        row = c1_ldg_ty + r * consumer1_ldg_ty_count
                                        kv_shared_r[
                                            row,
                                            64 * u + c1_ldg_tx * 8 + v,
                                        ] = kv_reg_r[r * 32 + u * 8 + v]

                            for r in T.unroll(consumer1_rows_per_thread):
                                for v in T.vectorized(4):
                                    row = c1_ldg_ty + r * consumer1_ldg_ty_count
                                    kv_reg_r_bf16_load[r * 16 + 8 + v] = kv_shared_r[
                                        row,
                                        128 + c1_ldg_tx * 8 + v,
                                    ]
                            T.lma_wait()
                            for r in T.unroll(consumer1_rows_per_thread):
                                for v in T.vectorized(8):
                                    idx = r * 32 + 16 + v
                                    kv_reg_r_fp16[idx] = kv_reg_r_fp8[idx]
                            for r in T.unroll(consumer1_rows_per_thread):
                                for v in T.vectorized(8):
                                    idx = r * 32 + 16 + v
                                    kv_reg_r[idx] = T.Cast(
                                        dtype,
                                        kv_reg_r_fp16[idx] * quant_local_r[r, 2],
                                    )
                            for r in T.unroll(consumer1_rows_per_thread):
                                for v in T.vectorized(8):
                                    idx = r * 32 + 16 + v
                                    row = c1_ldg_ty + r * consumer1_ldg_ty_count
                                    kv_shared_r[
                                        row,
                                        128 + c1_ldg_tx * 8 + v,
                                    ] = kv_reg_r[idx]

                            for r in T.unroll(consumer1_rows_per_thread):
                                for v in T.vectorized(8):
                                    idx = r * 32 + 24 + v
                                    row = c1_ldg_ty + r * consumer1_ldg_ty_count
                                    kv_reg_r[idx] = kv_shared_r[
                                        row,
                                        192 + c1_ldg_tx * 8 + v,
                                    ]
                        T.lma_wait()
                        T.barrier_arrive(bar_kv1_quant_ready)
                        if not use_three_v_buffers:
                            T.barrier_wait(bar_vl0_free, (phase_count[0] & 1))
                        stage_value_shared_mp32(
                            v_shared_2,
                            kv_reg_r,
                            c1_ldg_ty,
                            c1_ldg_tx,
                            0,
                            consumer1_ldg_ty_count,
                            consumer1_rows_per_thread,
                        )
                        T.lma_wait()
                        T.barrier_arrive(bar_vr0_ready)
                        T.barrier_wait(bar_vr0_ready, (phase_count[0] & 1))

                        T.barrier_wait(bar_p_ready, (phase_count[0] & 1))
                        for h_i, d_i in T.Parallel(heads_per_block, 128):
                            acc_o_r_0[h_i, d_i] *= alpha_shared[h_i]
                            acc_o_r_1[h_i, d_i] *= alpha_shared[h_i]

                        T.sched_boundary()
                        T.gemm(
                            scores_shared,
                            v_shared_2,
                            acc_o_r_0,
                            policy=pv_gemm_policy,
                            wg_wait=gemm_wg_wait,
                        )
                        if use_manual_sqmma_grouping:
                            T.warpgroup_commit_batch()

                        T.sched_boundary()

                        T.barrier_wait(bar_vl1_free, (phase_count[0] & 1))
                        stage_value_shared_mp32(
                            v_shared_1,
                            kv_reg_r,
                            c1_ldg_ty,
                            c1_ldg_tx,
                            2,
                            consumer1_ldg_ty_count,
                            consumer1_rows_per_thread,
                        )
                        T.lma_wait()
                        if use_manual_sqmma_grouping:
                            T.warpgroup_wait(0)
                        T.barrier_arrive(bar_vr1_ready)
                        T.barrier_wait(bar_vr1_ready, (phase_count[0] & 1))

                        T.sched_boundary()
                        T.gemm(
                            scores_shared,
                            v_shared_1,
                            acc_o_r_1,
                            policy=pv_gemm_policy,
                            wg_wait=gemm_wg_wait,
                        )
                        if use_manual_sqmma_grouping:
                            T.warpgroup_commit_batch()
                            T.warpgroup_wait(0)
                        T.sched_boundary()
                        phase_count[0] = phase_count[0] ^ 1

                    finalize_right(
                        b_i,
                        s_i,
                        h0,
                        h1,
                        is_unsplit,
                        acc_o_r_0,
                        acc_o_r_1,
                        sum_exp_inv_shared,
                        sink_scale_shared,
                        output,
                        output_partial,
                        n_split_idx,
                        num_splits,
                        bar_final,
                        0,
                        (b_i - begin_idx) & 1,
                    )
                else:
                    kperm_mask_local = T.alloc_local([4], "bool")
                    kperm_indices_local = T.alloc_local([4], indices_dtype)
                    topk_len_local = T.alloc_local([1], indices_dtype)
                    extra_topk_len_local = T.alloc_local([1], indices_dtype)
                    producer_tid = tid - producer_start
                    ldg_tx = producer_tid % 8
                    ldg_ty = producer_tid // 8
                    ldg_scale_tx = producer_tid % 2
                    ldg_scale_ty = producer_tid // 2
                    if has_topk_length:
                        topk_len_local[0] = topk_length[b_i]
                    else:
                        topk_len_local[0] = topk
                    if has_extra_topk_length:
                        extra_topk_len_local[0] = extra_topk_length[b_i]
                    else:
                        extra_topk_len_local[0] = extra_topk

                    for i_i in range(start_block_idx, end_block_idx):
                        main_block_count = T.max(
                            T.ceildiv(topk_len_local[0], block_i), 1
                        )
                        use_extra = i_i >= main_block_count
                        T.annotate_layout(
                            {
                                kv_shared_l[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_l[:, :],
                                    k_major=True,
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        T.annotate_layout(
                            {
                                kv_shared_r[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_r[:, :],
                                    k_major=True,
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        if use_extra:
                            block_index = i_i - main_block_count
                            load_indices_mp32(
                                extra_indices,
                                b_i,
                                s_i,
                                g_i,
                                block_index,
                                extra_topk_len_local[0],
                                seq_len_kv_extra,
                                ldg_ty,
                                ldg_tx,
                                phase_count[0],
                                kperm_indices_local,
                                kperm_mask_local,
                                is_kv_valid,
                                kv_indices,
                                bar_kv_mask_free,
                            )

                            load_extra_kv_block(
                                extra_kv_rope,
                                extra_quant_scales,
                                extra_rope_robust_desc,
                                extra_scale_robust_desc,
                                page_block_size_extra,
                                kperm_indices_local,
                                kv_indices,
                                kv_shared_l,
                                kv_shared_r,
                                quant_shared,
                                ldg_ty,
                                ldg_tx,
                                ldg_scale_ty,
                                ldg_scale_tx,
                                phase_count[0],
                                bar_kv0_free,
                                bar_kv_mask_ready,
                                bar_indices_ready,
                                bar_kv0_ready,
                                bar_kv1_free,
                                bar_kv1_ready,
                            )
                            phase_count[0] = phase_count[0] ^ 1

                        else:
                            block_index = i_i
                            load_indices_mp32(
                                indices,
                                b_i,
                                s_i,
                                g_i,
                                block_index,
                                topk_len_local[0],
                                seq_len_kv,
                                ldg_ty,
                                ldg_tx,
                                phase_count[0],
                                kperm_indices_local,
                                kperm_mask_local,
                                is_kv_valid,
                                kv_indices,
                                bar_kv_mask_free,
                            )
                            load_main_kv_block(
                                kv_rope,
                                quant_scales,
                                rope_robust_desc,
                                scale_robust_desc,
                                page_block_size,
                                kperm_indices_local,
                                kv_indices,
                                kv_shared_l,
                                kv_shared_r,
                                quant_shared,
                                ldg_ty,
                                ldg_tx,
                                ldg_scale_ty,
                                ldg_scale_tx,
                                phase_count[0],
                                bar_kv0_free,
                                bar_kv_mask_ready,
                                bar_indices_ready,
                                bar_kv0_ready,
                                bar_kv1_free,
                                bar_kv1_ready,
                            )
                            phase_count[0] = phase_count[0] ^ 1

        # V4 scheduled combine kernel. Only batches with more than one split
        # enter this stage; unsplit batches are already written by the split kernel.
        dsa_combine(
            tile_scheduler_metadata,
            num_splits,
            glse,
            output_partial,
            attn_sink,
            output,
            lse,
        )

    return dsa_decode


def sparse_mla_decode_fwd_scheduled_interface_v4(
    q,
    kv_nope,
    kv_rope,
    kv_scales,
    indices,
    tile_scheduler_metadata,
    num_splits,
    *,
    extra_kv_nope=None,
    extra_kv_rope=None,
    extra_kv_scales=None,
    extra_indices=None,
    topk_length=None,
    extra_topk_length=None,
    sm_scale=None,
    attn_sink=None,
    return_p_sum: bool = False,
    d_v=512,
    block_m=64,
    block_i=64,
    threads=0,
    consumer0_threads=256,
    consumer1_threads=256,
    producer_threads=128,
    verbose=False,
    page_block_size=64,
    extra_page_block_size=64,
    cache_format="v4",
    extra_cache_format="v4",
    q_rope_positions=None,
    q_rope_cos_sin=None,
):
    assert return_p_sum is False, "This kernel file is for decode only"
    assert q.dtype == torch.bfloat16, "q must be bfloat16"
    assert kv_nope.dtype == torch.float8_e4m3fn, "kv_nope must be float8_e4m3fn"
    assert kv_rope.dtype == torch.bfloat16, "kv_rope must be bfloat16"
    assert kv_scales.dtype == torch.uint8, "kv_scales must be uint8"
    assert indices.dtype == torch.int32, "indices must be int32"
    q, q_shape = prepare_sparse_mla_decode_strided_tensor("q", q, multiple=8)
    num_blocks = kv_nope.shape[0]
    assert cache_format in ("v4", "v41")
    assert extra_cache_format in ("v4", "v41", "v41_fp4")
    dim_bytes = 584 if cache_format == "v4" else 528
    page_block_bytes = page_block_size * dim_bytes
    assert kv_nope.shape[1] >= page_block_bytes
    assert kv_rope.shape[0] == num_blocks
    assert kv_rope.shape[1] >= page_block_bytes // 2
    assert kv_scales.shape[0] == num_blocks
    assert kv_scales.shape[1] >= page_block_bytes
    kv_nope = kv_nope[:, :page_block_bytes]
    kv_rope = kv_rope[:, : page_block_bytes // 2]
    kv_scales = kv_scales[:, :page_block_bytes]
    kv_nope, kv_nope_shape = prepare_sparse_mla_decode_strided_tensor(
        "kv_nope", kv_nope, multiple=8
    )
    kv_rope, kv_rope_shape = prepare_sparse_mla_decode_strided_tensor(
        "kv_rope", kv_rope, multiple=4
    )
    kv_scales, kv_scales_shape = prepare_sparse_mla_decode_strided_tensor(
        "kv_scales", kv_scales, multiple=8
    )
    indices, indices_shape = prepare_sparse_mla_decode_strided_tensor(
        "indices", indices, multiple=8
    )
    batch, seq_len, heads, dim_q = q_shape
    assert int(num_splits.numel()) == batch + 1
    assert kv_nope_shape == (num_blocks, page_block_bytes)
    kv_group = 1
    assert dim_q == d_v == 512
    assert kv_rope_shape == (num_blocks, page_block_bytes // 2)
    assert kv_scales_shape == (num_blocks, page_block_bytes)
    _, _, _, topk = indices_shape
    assert indices_shape == (batch, seq_len, kv_group, topk)

    extra_inputs = (extra_kv_nope, extra_kv_rope, extra_kv_scales, extra_indices)
    has_extra_inputs = all(tensor is not None for tensor in extra_inputs)
    assert has_extra_inputs or not any(tensor is not None for tensor in extra_inputs), (
        "extra_kv_nope, extra_kv_rope, extra_kv_scales, and extra_indices "
        "must be provided together"
    )
    if not has_extra_inputs:
        extra_indices = indices[:, :, :, :0]
        assert extra_topk_length is None, (
            "extra_topk_length requires extra cache inputs"
        )
    else:
        assert extra_page_block_size is not None, (
            "extra_page_block_size must be provided with extra cache inputs"
        )
    assert extra_indices.dtype == torch.int32, "extra_indices must be int32"
    extra_indices, extra_indices_shape = prepare_sparse_mla_decode_strided_tensor(
        "extra_indices", extra_indices, multiple=8
    )
    extra_topk = extra_indices_shape[-1]
    if has_extra_inputs:
        assert extra_topk > 0, "extra_indices must contain at least one index"
    assert extra_indices_shape == (batch, seq_len, kv_group, extra_topk)
    if not has_extra_inputs:
        extra_kv_nope, extra_kv_rope, extra_kv_scales = kv_nope, kv_rope, kv_scales
        extra_page_block_size = page_block_size
        extra_cache_format = cache_format
    extra_dim_bytes = 288 if extra_cache_format == "v41_fp4" else (
        584 if extra_cache_format == "v4" else 528
    )
    assert extra_kv_nope.dtype == torch.float8_e4m3fn, (
        "extra_kv_nope must be float8_e4m3fn"
    )
    assert extra_kv_rope.dtype == torch.bfloat16, "extra_kv_rope must be bfloat16"
    assert extra_kv_scales.dtype == torch.uint8, "extra_kv_scales must be uint8"
    extra_num_blocks = extra_kv_nope.shape[0]
    extra_page_block_bytes = extra_page_block_size * extra_dim_bytes
    assert extra_kv_nope.shape[1] >= extra_page_block_bytes
    assert extra_kv_rope.shape[0] == extra_num_blocks
    assert extra_kv_rope.shape[1] >= extra_page_block_bytes // 2
    assert extra_kv_scales.shape[0] == extra_num_blocks
    assert extra_kv_scales.shape[1] >= extra_page_block_bytes
    extra_kv_nope = extra_kv_nope[:, :extra_page_block_bytes]
    extra_kv_rope = extra_kv_rope[:, : extra_page_block_bytes // 2]
    extra_kv_scales = extra_kv_scales[:, :extra_page_block_bytes]
    extra_kv_nope, extra_kv_nope_shape = prepare_sparse_mla_decode_strided_tensor(
        "extra_kv_nope", extra_kv_nope, multiple=8
    )
    extra_kv_rope, extra_kv_rope_shape = prepare_sparse_mla_decode_strided_tensor(
        "extra_kv_rope", extra_kv_rope, multiple=4
    )
    extra_kv_scales, extra_kv_scales_shape = prepare_sparse_mla_decode_strided_tensor(
        "extra_kv_scales", extra_kv_scales, multiple=8
    )
    assert extra_kv_nope_shape == (extra_num_blocks, extra_page_block_bytes)
    assert extra_kv_rope_shape == (
        extra_num_blocks,
        extra_page_block_bytes // 2,
    )
    assert extra_kv_scales_shape == (extra_num_blocks, extra_page_block_bytes)

    topk_length_b = validate_batch_lengths(topk_length, batch, "topk_length")
    if topk_length_b is not None:
        assert topk_length_b.device == q.device
    topk_length_arg = (
        topk_length_b
        if topk_length_b is not None
        else indices.as_strided((batch,), (1,))
    )
    extra_topk_length_b = validate_batch_lengths(
        extra_topk_length, batch, "extra_topk_length"
    )
    if extra_topk_length_b is not None:
        assert extra_topk_length_b.device == q.device
    extra_topk_length_arg = (
        extra_topk_length_b
        if extra_topk_length_b is not None
        else indices.as_strided((batch,), (1,))
    )
    has_q_rope = q_rope_positions is not None or q_rope_cos_sin is not None
    if has_q_rope:
        assert q_rope_positions is not None and q_rope_cos_sin is not None, (
            "q_rope_positions and q_rope_cos_sin must be provided together"
        )
        assert q_rope_positions.dtype in (torch.int32, torch.int64)
        assert q_rope_positions.shape == (batch,)
        assert q_rope_cos_sin.dtype == torch.float32
        assert q_rope_cos_sin.ndim == 2 and q_rope_cos_sin.shape[1] == 64
        assert q_rope_positions.device == q.device
        assert q_rope_cos_sin.device == q.device
        q_rope_positions = q_rope_positions.contiguous()
        q_rope_cos_sin = q_rope_cos_sin.contiguous()
        q_position_dtype = str(q_rope_positions.dtype).split(".")[-1]
    else:
        q_rope_positions = num_splits[:batch]
        q_rope_cos_sin = (
            tile_scheduler_metadata.view(torch.float32).flatten()[:64].view(1, 64)
        )
        q_position_dtype = "int32"
    runtime = prepare_scheduled_decode_runtime(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        dim=d_v,
        topk_length=topk_length_b,
        attn_sink=attn_sink,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        out_dtype=q.dtype,
        device=q.device,
        variant_name="V4",
    )

    assert runtime.glse is not None and runtime.out_partial is not None
    address_tensors = [
        q,
        kv_nope,
        kv_rope,
        kv_scales,
        indices,
        extra_kv_nope,
        extra_kv_rope,
        extra_kv_scales,
        extra_indices,
        runtime.out_partial,
        q_rope_positions,
        q_rope_cos_sin,
    ]
    kernel_factory = jit_for_tensor_addressing(
        sparse_attention_decode_fwd_scheduled_kernel_v4, *address_tensors
    )
    use_8byte_kv_loads = kv_rope.stride(0) % 8 != 0
    if has_extra_inputs:
        use_8byte_kv_loads = use_8byte_kv_loads or extra_kv_rope.stride(0) % 8 != 0
    kernel = kernel_factory(
        heads,
        d_v,
        has_extra=extra_topk > 0,
        kv_group=kv_group,
        sm_scale=sm_scale,
        block_m=block_m,
        block_i=block_i,
        threads=threads,
        consumer0_threads=consumer0_threads,
        consumer1_threads=consumer1_threads,
        producer_threads=producer_threads,
        max_nums_splits=runtime.max_nums_splits,
        has_attn_sink=runtime.has_attn_sink,
        has_topk_length=topk_length_b is not None,
        has_extra_topk_length=extra_topk_length_b is not None,
        page_block_size=page_block_size,
        extra_page_block_size=extra_page_block_size,
        use_8byte_kv_loads=use_8byte_kv_loads,
        fixed_topk=topk if topk_length_b is None else None,
        cache_format=cache_format,
        extra_cache_format=extra_cache_format,
        has_q_rope=has_q_rope,
        q_position_dtype=q_position_dtype,
        use_int64_cosize=(
            kernel_factory is not sparse_attention_decode_fwd_scheduled_kernel_v4
        ),
    )
    if verbose:
        kernel.show_source()
    raise_complete_if_dry_run()
    attn_sink_arg = runtime.attn_sink
    if attn_sink_arg is None:
        attn_sink_arg = q.view(torch.float32).as_strided((heads,), (1,))
    kernel(
        q,
        kv_nope,
        kv_rope,
        kv_scales,
        indices,
        topk_length_arg,
        extra_kv_nope,
        extra_kv_rope,
        extra_kv_scales,
        extra_indices,
        extra_topk_length_arg,
        attn_sink_arg,
        q_rope_positions,
        q_rope_cos_sin,
        tile_scheduler_metadata,
        num_splits,
        runtime.glse,
        runtime.out_partial,
        runtime.out,
        runtime.lse,
        # cosize is computed on the host and passed in as scalar arguments.
        cosize(q.shape, q.stride()),
        cosize(kv_nope.shape, kv_nope.stride()),
        cosize(kv_rope.shape, kv_rope.stride()),
        cosize(kv_scales.shape, kv_scales.stride()),
        cosize(extra_kv_nope.shape, extra_kv_nope.stride()),
        cosize(extra_kv_rope.shape, extra_kv_rope.stride()),
        cosize(extra_kv_scales.shape, extra_kv_scales.stride()),
    )

    return runtime.out, runtime.lse


# Keep the public name architecture-explicit at dispatch sites while retaining
# the historical V4 scheduled interface for compatibility imports.
sparse_mla_decode_fwd_scheduled_interface_v4_mp32 = (
    sparse_mla_decode_fwd_scheduled_interface_v4
)
