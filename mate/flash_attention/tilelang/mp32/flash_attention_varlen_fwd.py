# SPDX-License-Identifier: Apache-2.0
# mypy: ignore-errors
"""Production MP32 TileLang FlashAttention forward kernels.

The three configurations below are the supported PH1S tuning points:
128 uses the BN128 pipeline and hand-written 2x32 online softmax; 192/128
uses split K=128/K=64 score issue ordering; 256 uses BN64 and fused 2x16 softmax.
One static kernel factory selects the specialized path at compile time, and
benchmark entry points are intentionally kept out of the runtime package.
"""

import torch
import tilelang
import tilelang.language as T

from ....execution_context import skip_kernel_launch_if_dry_run
from ._flash_attention_fwd_common import (
    _MP32_FORWARD_SOURCE,
    _alloc_phase_registers,
    _annotate_consumer_layout,
    _assume_attention_strides,
    _check_attention_strides,
    _current_musa_cc,
    _make_attention_accessors,
    _make_kernel_types,
    _perm_g2s,
    _perm_r2r,
    _tilelang_dtype_nbytes,
    _to_tilelang_dtype,
    _unpack_kernel_types,
    cosize,
)

_DIM128_COMPILE_DIM = 128
_DIM128_BLOCK_M = 256
_DIM128_BLOCK_N = 128
_DIM128_PRODUCER_THREADS = 128
_DIM128_CONSUMER_THREADS = 512

_D192_QK_DIM = 192
_D192_VO_DIM = 128
_D192_QK_MAIN_DIM = 128
_D192_QK_TAIL_DIM = 64
_D192_QK_TAIL_END = _D192_QK_MAIN_DIM + _D192_QK_TAIL_DIM
_D192_SQUAD_THREADS = 128
_D192_CONSUMER_SQUADS = 4
_D192_BLOCK_M = 256
_D192_BLOCK_N = 128
_D192_PRODUCER_THREADS = 128
_D192_CONSUMER_THREADS = 512

_DIM256_COMPILE_DIM = 256
_DIM256_BLOCK_M = 256
_DIM256_BLOCK_N = 64
_DIM256_PRODUCER_THREADS = 128
_DIM256_CONSUMER_THREADS = 512


def _select_kernel_config(actual_dim_qk, actual_dim_vo):
    dispatch_class = _dispatch_class(actual_dim_qk, actual_dim_vo)
    if dispatch_class == 128:
        return {
            "block_M": _DIM128_BLOCK_M,
            "block_N": _DIM128_BLOCK_N,
            "dim_qk": _DIM128_COMPILE_DIM,
            "dim_vo": _DIM128_COMPILE_DIM,
            "produces_threads": _DIM128_PRODUCER_THREADS,
            "consumer_threads": _DIM128_CONSUMER_THREADS,
        }
    if dispatch_class == 192:
        return {
            "block_M": _D192_BLOCK_M,
            "block_N": _D192_BLOCK_N,
            "dim_qk": _D192_QK_DIM,
            "dim_vo": _D192_VO_DIM,
            "produces_threads": _D192_PRODUCER_THREADS,
            "consumer_threads": _D192_CONSUMER_THREADS,
        }
    if dispatch_class == 256:
        return {
            "block_M": _DIM256_BLOCK_M,
            "block_N": _DIM256_BLOCK_N,
            "dim_qk": _DIM256_COMPILE_DIM,
            "dim_vo": _DIM256_COMPILE_DIM,
            "produces_threads": _DIM256_PRODUCER_THREADS,
            "consumer_threads": _DIM256_CONSUMER_THREADS,
            "use_tma_k": True,
        }
    raise AssertionError(
        "MP32 TileLang FA supports dim classes <=128, (192, 128), and <=256; "
        f"got ({actual_dim_qk}, {actual_dim_vo})"
    )


def _dispatch_class(actual_dim_qk, actual_dim_vo):
    if actual_dim_qk == 192 and actual_dim_vo == 128:
        return 192
    if actual_dim_qk == actual_dim_vo and 0 < actual_dim_qk <= 128:
        return 128
    if actual_dim_qk == actual_dim_vo and 128 < actual_dim_qk <= 256:
        return 256
    raise ValueError(
        "MP32 TileLang FA supports dim classes <=128, (192,128), and <=256; "
        f"got ({actual_dim_qk}, {actual_dim_vo})"
    )


def _select_kernel(actual_dim_qk, actual_dim_vo):
    """Return the production kernel and static config for a supported shape."""
    kernel_config = _select_kernel_config(actual_dim_qk, actual_dim_vo)
    return _flashattn_mp32_kernel, kernel_config


@tilelang.jit(
    target={"kind": "musa", "arch": "mp_32"},
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
    },
    compile_flags=[
        "-fmusa-flush-denormals-to-zero",
        "-fno-signed-zeros",
        "-fno-strict-aliasing",
        "-mllvm",
        "-misched=mtgpu-max-ilp",
        "-mllvm",
        "-mtgpu-if-convert=1",
        "-mllvm",
        "-mtgpu-tiny-offset-hint=1",
        "-mllvm",
        "-misched-recompute-slotindex=1",
        "-mllvm",
        "-mtgpu-combine-fop-instr=1",
        "-mllvm",
        "-mtgpu-combine-instr-with-burst=1",
    ],
)
def _flashattn_mp32_kernel(
    block_M=_DIM128_BLOCK_M,
    block_N=_DIM128_BLOCK_N,
    dim_qk=_DIM128_COMPILE_DIM,
    dim_vo=_DIM128_COMPILE_DIM,
    produces_threads=_DIM128_PRODUCER_THREADS,
    consumer_threads=_DIM128_CONSUMER_THREADS,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
    disable_rs=True,
    use_tma_k=True,
    is_causal=False,
    is_varlen=True,
    heads_q_eq_heads_kv=True,
):
    # Keep the D192 split-QK path a compile-time specialization of the shared
    # pipeline rather than a second, duplicated kernel body.
    is_d192 = dim_qk == _D192_QK_DIM and dim_vo == _D192_VO_DIM
    qk_main_dim = _D192_QK_MAIN_DIM if is_d192 else dim_qk
    use_manual_reduce = (
        block_M == 256
        and block_N in (64, 128)
        and consumer_threads == 512
        and accum_dtype == T.float32
    )
    manual_reduce_max_name = (
        "flashattn_reduce_max_2x16" if block_N == 64 else "flashattn_reduce_max_2x32"
    )
    manual_reduce_sum_name = (
        "flashattn_reduce_sum_2x16" if block_N == 64 else "flashattn_reduce_sum_2x32"
    )
    use_manual_softmax = (
        # BN128/D128 and BN64/D256 have different per-thread score widths;
        # select the matching fused update instead of falling back to the
        # generic elementwise softmax path.
        use_manual_reduce
        and ((block_N == 128 and dim_vo == 128) or (block_N == 64 and dim_vo == 256))
        and not disable_rs
        and dtype in (T.float16, T.bfloat16)
    )
    manual_softmax_name = (
        "flashattn_softmax_update_2x16_2x64"
        if block_N == 64 and dim_vo == 256
        else "flashattn_softmax_update_2x32"
    )
    num_stages = 2
    (
        total_seq_q,
        total_seq_kv,
        batch,
        heads,
        heads_kv,
        seqlen_q,
        seqlen_kv,
        actual_dim_qk,
        actual_dim_vo,
        q_stride_b,
        q_stride_s,
        q_stride_h,
        k_stride_b,
        k_stride_s,
        k_stride_h,
        v_stride_b,
        v_stride_s,
        v_stride_h,
        o_stride_b,
        o_stride_s,
        o_stride_h,
        q_type,
        k_type,
        v_type,
        o_type,
        lse_type,
        q_strides,
        k_strides,
        kv_shape,
    ) = _unpack_kernel_types(
        _make_kernel_types(is_varlen, dtype, accum_dtype, heads_q_eq_heads_kv)
    )
    (
        sequence_start,
        sequence_length,
        input_tile,
        k_tma_tile,
        input_elem,
        input_base,
        output_index,
        lse_index,
    ) = _make_attention_accessors(is_varlen)
    if not use_tma_k:
        dtype_bytes = _tilelang_dtype_nbytes(dtype)
        k_robust_bytes = cosize(kv_shape, k_strides) * dtype_bytes

    @T.prim_func
    def flashatten_fwd_ws(
        Q: q_type,  # type: ignore
        K: k_type,  # type: ignore
        V: v_type,  # type: ignore
        cu_seq_q: T.Tensor((batch + 1,), T.int32),  # type: ignore
        cu_seq_kv: T.Tensor((batch + 1,), T.int32),  # type: ignore
        max_seq_q: T.int32,
        max_seq_kv: T.int32,
        scale: T.float32,
        scale_log2e: T.float32,
        Output: o_type,  # type: ignore
        Logsumexp: lse_type,  # type: ignore
    ):
        with T.Kernel(
            T.ceildiv(max_seq_q, block_M),
            heads,
            batch,
            threads=produces_threads + consumer_threads,
        ) as (bx, by, bz):
            kv_head = by if heads_q_eq_heads_kv else by // (heads // heads_kv)
            T.import_source(_MP32_FORWARD_SOURCE)
            _assume_attention_strides(
                is_varlen,
                (
                    q_stride_s,
                    q_stride_h,
                    k_stride_s,
                    k_stride_h,
                    v_stride_s,
                    v_stride_h,
                    o_stride_s,
                    o_stride_h,
                ),
                (q_stride_b, k_stride_b, v_stride_b, o_stride_b),
            )
            if use_tma_k:
                # TME inner_stride permutes descriptor dim1. Put KV there while
                # preserving the original storage, strides, and dimension bounds.
                if is_varlen:
                    K_tma = T.alias_buffer(
                        K,
                        shape=(heads_kv, total_seq_kv, actual_dim_qk),
                        strides=(k_stride_h, k_stride_s, 1),
                    )
                else:
                    K_tma = T.alias_buffer(
                        K,
                        shape=(batch, heads_kv, seqlen_kv, actual_dim_qk),
                        strides=(k_stride_b, k_stride_h, k_stride_s, 1),
                    )
            Q_shared = T.alloc_shared((block_M, qk_main_dim), dtype)
            K_shared = T.alloc_shared((num_stages, block_N, qk_main_dim), dtype)
            if is_d192:
                Q_tail_shared = T.alloc_shared((block_M, _D192_QK_TAIL_DIM), dtype)
                K_tail_shared = T.alloc_shared(
                    (num_stages, block_N, _D192_QK_TAIL_DIM), dtype
                )
            V_shared = T.alloc_shared((num_stages, block_N, dim_vo), dtype)
            if disable_rs:
                P_shared = T.alloc_shared((block_M, block_N), dtype)

            bar_q_ready = T.alloc_barrier(arrive_count=produces_threads)
            bar_k_ready = T.alloc_barrier(arrive_count=[produces_threads] * num_stages)
            if is_d192:
                bar_k_tail_ready = T.alloc_barrier(
                    arrive_count=[produces_threads] * num_stages
                )
            bar_v_ready = T.alloc_barrier(arrive_count=[produces_threads] * num_stages)

            bar_k_free = T.alloc_barrier(arrive_count=[consumer_threads] * num_stages)
            bar_v_free = T.alloc_barrier(arrive_count=[consumer_threads] * num_stages)
            if is_d192:
                # Order QK-main/QK-tail issue across the four consumer squads.
                bar_qk_squad_order = T.alloc_barrier(
                    arrive_count=[_D192_SQUAD_THREADS] * _D192_CONSUMER_SQUADS
                )
            T.sync_threads()
            tid = T.get_thread_binding()
            producer_phase, consumer_phase = _alloc_phase_registers()
            # Emit initialization in the DSL, not in an ordinary Python helper.
            T.fill(producer_phase, 1)
            T.fill(consumer_phase, 0)
            begin_seq_q = T.alloc_var(T.int32)
            begin_seq_kv = T.alloc_var(T.int32)
            local_seq_q = T.alloc_var(T.int32)
            local_seq_kv = T.alloc_var(T.int32)
            causal_offset = T.alloc_var(T.int32)
            begin_seq_q = sequence_start(cu_seq_q, bz, seqlen_q)
            begin_seq_kv = sequence_start(cu_seq_kv, bz, seqlen_kv)
            local_seq_q = sequence_length(cu_seq_q, bz, seqlen_q)
            local_seq_kv = sequence_length(cu_seq_kv, bz, seqlen_kv)
            causal_offset = local_seq_kv - local_seq_q
            kv_loop_end = local_seq_kv
            if is_causal:
                kv_loop_end = T.max(
                    T.min(
                        kv_loop_end,
                        (bx + 1) * block_M + causal_offset,
                    ),
                    0,
                )
            if not use_tma_k:
                k_robust_desc = T.make_robust_desc(
                    T.address_of(input_base(K)), k_robust_bytes
                )
            if tid < produces_threads:
                T.tma_copy(
                    input_tile(
                        Q,
                        bz,
                        begin_seq_q,
                        bx * block_M,
                        (bx + 1) * block_M,
                        by,
                        0,
                        qk_main_dim,
                    ),
                    Q_shared,
                    barrier=bar_q_ready,
                )
                if is_d192:
                    T.tma_copy(
                        input_tile(
                            Q,
                            bz,
                            begin_seq_q,
                            bx * block_M,
                            (bx + 1) * block_M,
                            by,
                            _D192_QK_MAIN_DIM,
                            _D192_QK_TAIL_END,
                        ),
                        Q_tail_shared,
                        barrier=bar_q_ready,
                    )
                T.barrier_arrive(bar_q_ready)
                for kv_start in range(0, max_seq_kv, block_N):
                    if kv_start < kv_loop_end:
                        kv_tile = kv_start // block_N
                        kv_stage = kv_tile % num_stages
                        T.barrier_wait(bar_k_free[kv_stage], producer_phase[0])
                        if use_tma_k:
                            T.tma_copy(
                                k_tma_tile(
                                    K_tma,
                                    bz,
                                    begin_seq_kv,
                                    kv_start,
                                    kv_start + block_N,
                                    kv_head,
                                    0,
                                    qk_main_dim,
                                ),
                                K_shared[kv_stage, :, :],
                                barrier=bar_k_ready[kv_stage],
                                inner_stride=2,
                            )
                        else:
                            for i, j in T.Parallel(block_N, qk_main_dim):
                                T.copy(
                                    input_elem(
                                        K,
                                        bz,
                                        begin_seq_kv,
                                        kv_start + _perm_g2s(i),
                                        kv_head,
                                        j,
                                    ),
                                    K_shared[kv_stage, i, j],
                                    force_async_copy=True,
                                    src_robust_desc=k_robust_desc,
                                )
                            T.ptx_commit_group()
                            T.ptx_wait_group(0)
                        T.barrier_arrive(bar_k_ready[kv_stage])
                        if is_d192:
                            if use_tma_k:
                                T.tma_copy(
                                    k_tma_tile(
                                        K_tma,
                                        bz,
                                        begin_seq_kv,
                                        kv_start,
                                        kv_start + block_N,
                                        kv_head,
                                        _D192_QK_MAIN_DIM,
                                        _D192_QK_TAIL_END,
                                    ),
                                    K_tail_shared[kv_stage, :, :],
                                    barrier=bar_k_tail_ready[kv_stage],
                                    inner_stride=2,
                                )
                            else:
                                for i, j in T.Parallel(block_N, _D192_QK_TAIL_DIM):
                                    T.copy(
                                        input_elem(
                                            K,
                                            bz,
                                            begin_seq_kv,
                                            kv_start + _perm_g2s(i),
                                            kv_head,
                                            _D192_QK_MAIN_DIM + j,
                                        ),
                                        K_tail_shared[kv_stage, i, j],
                                        force_async_copy=True,
                                        src_robust_desc=k_robust_desc,
                                    )
                                T.ptx_commit_group()
                                T.ptx_wait_group(0)
                            T.barrier_arrive(bar_k_tail_ready[kv_stage])

                        T.barrier_wait(bar_v_free[kv_stage], producer_phase[0])
                        T.tma_copy(
                            input_tile(
                                V,
                                bz,
                                begin_seq_kv,
                                kv_start,
                                kv_start + block_N,
                                kv_head,
                                0,
                                dim_vo,
                            ),
                            V_shared[kv_stage, :, :],
                            barrier=bar_v_ready[kv_stage],
                        )
                        T.barrier_arrive(bar_v_ready[kv_stage])
                        if kv_stage == num_stages - 1:
                            producer_phase[0] = producer_phase[0] ^ 1
            else:
                acc_s = T.alloc_fragment((block_M, block_N), accum_dtype)
                scores_max = T.alloc_fragment((block_M), accum_dtype)
                scores_max_prev = T.alloc_fragment((block_M), accum_dtype)
                scores_scale = T.alloc_fragment((block_M), accum_dtype)
                scores_sum = T.alloc_fragment((block_M), accum_dtype)
                logsum = T.alloc_fragment((block_M), accum_dtype)
                acc_o = T.alloc_fragment((block_M, dim_vo), accum_dtype)
                P_cvt = T.alloc_fragment((block_M, block_N), dtype)

                if block_M == 256 and block_N in (64, 128) and consumer_threads == 512:
                    _annotate_consumer_layout(
                        acc_s,
                        scores_max,
                        scores_max_prev,
                        scores_scale,
                        scores_sum,
                        acc_o,
                        P_cvt,
                        block_N,
                        dim_vo,
                    )

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -(2**30))

                T.barrier_wait(bar_q_ready, 0)
                for kv_start in range(0, max_seq_kv, block_N):
                    if kv_start < kv_loop_end:
                        kv_tile = kv_start // block_N
                        kv_stage = kv_tile % num_stages
                        T.fill(acc_s, 0)
                        if is_causal:
                            # The first query row can see keys through its diagonal.
                            # For valid Q rows, causal masking also excludes KV padding:
                            # q + (local_seq_kv - local_seq_q) < local_seq_kv.
                            if kv_start + block_N > bx * block_M + causal_offset + 1:
                                for i, j in T.Parallel(block_M, block_N):
                                    acc_s[i, j] = T.if_then_else(
                                        bx * block_M + i + causal_offset
                                        >= kv_start + _perm_g2s(j),
                                        0,
                                        -T.infinity(accum_dtype),
                                    )
                        elif kv_start + block_N > local_seq_kv:
                            for i, j in T.Parallel(block_M, block_N):
                                acc_s[i, j] = T.if_then_else(
                                    kv_start + _perm_g2s(j) < local_seq_kv,
                                    0,
                                    -T.infinity(accum_dtype),
                                )
                        T.barrier_wait(bar_k_ready[kv_stage], consumer_phase[0])
                        if is_d192:
                            consumer_squad = (
                                tid - produces_threads
                            ) // _D192_SQUAD_THREADS
                            # Keep QK issue ordered across squads while allowing
                            # softmax to overlap the next squad's main GEMM.
                            if consumer_squad == 0:
                                if kv_tile > 0:
                                    T.barrier_wait(
                                        bar_qk_squad_order[0], (kv_tile - 1) % 2
                                    )
                            else:
                                T.barrier_wait(
                                    bar_qk_squad_order[consumer_squad], kv_tile % 2
                                )
                        T.gemm(
                            Q_shared,
                            K_shared[kv_stage, :, :],
                            acc_s,
                            clear_accum=False,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        if is_d192:
                            T.barrier_wait(
                                bar_k_tail_ready[kv_stage], consumer_phase[0]
                            )
                            T.warpgroup_wait(0)
                            T.gemm(
                                Q_tail_shared,
                                K_tail_shared[kv_stage, :, :],
                                acc_s,
                                clear_accum=False,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                wg_wait=-1,
                            )
                            T.warpgroup_commit_batch()
                            T.barrier_arrive(
                                bar_qk_squad_order[
                                    (consumer_squad + 1) % _D192_CONSUMER_SQUADS
                                ]
                            )
                        T.copy(scores_max, scores_max_prev)
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_k_free[kv_stage])
                        if use_manual_reduce:
                            T.call_extern(
                                manual_reduce_max_name,
                                T.access_ptr(acc_s, "r"),
                                T.access_ptr(scores_max, "w"),
                                dtype="void",
                            )
                        else:
                            T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                        if use_manual_softmax:
                            T.call_extern(
                                manual_softmax_name,
                                T.access_ptr(acc_s, "rw"),
                                T.access_ptr(scores_max, "rw"),
                                T.access_ptr(scores_max_prev, "r"),
                                T.access_ptr(scores_scale, "w"),
                                T.access_ptr(acc_o, "rw"),
                                scale,
                                scale_log2e,
                                dtype="void",
                            )
                            for i, j in T.Parallel(block_M, block_N):
                                P_cvt[i, j] = acc_s[i, _perm_r2r(j)]
                        else:
                            for i in T.Parallel(block_M):
                                scores_max[i] = scores_max[i] * scale
                            for i in T.Parallel(block_M):
                                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                            for i in T.Parallel(block_M):
                                scores_scale[i] = T.exp2(
                                    (scores_max_prev[i] - scores_max[i]) * scale_log2e
                                )
                            for i, j in T.Parallel(block_M, block_N):
                                acc_s[i, j] = (
                                    acc_s[i, j] * scale - scores_max[i]
                                ) * scale_log2e
                            for i, j in T.Parallel(block_M, block_N):
                                acc_s[i, j] = T.exp2(acc_s[i, j])
                            for i, j in T.Parallel(block_M, dim_vo):
                                acc_o[i, j] *= scores_scale[i]
                            if disable_rs:
                                T.copy(acc_s, P_cvt)
                                for i, j in T.Parallel(block_M, block_N):
                                    P_shared[i, j] = P_cvt[i, _perm_r2r(j)]
                                T.sync_warp()
                            else:
                                for i, j in T.Parallel(block_M, block_N):
                                    P_cvt[i, j] = acc_s[i, _perm_r2r(j)]
                        T.barrier_wait(bar_v_ready[kv_stage], consumer_phase[0])
                        if disable_rs:
                            T.gemm(
                                P_shared,
                                V_shared[kv_stage, :, :],
                                acc_o,
                                clear_accum=False,
                                policy=T.GemmWarpPolicy.FullRow,
                                wg_wait=-1,
                            )
                            T.warpgroup_commit_batch()
                        else:
                            T.gemm(
                                P_cvt,
                                V_shared[kv_stage, :, :],
                                acc_o,
                                clear_accum=False,
                                policy=T.GemmWarpPolicy.FullRow,
                                wg_wait=-1,
                            )
                            T.warpgroup_commit_batch()
                        if use_manual_reduce:
                            T.call_extern(
                                manual_reduce_sum_name,
                                T.access_ptr(acc_s, "r"),
                                T.access_ptr(scores_sum, "w"),
                                dtype="void",
                            )
                        else:
                            T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_M):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_v_free[kv_stage])
                        if kv_stage == num_stages - 1:
                            consumer_phase[0] = consumer_phase[0] ^ 1
                for i in T.Parallel(block_M):
                    if bx * block_M + i < local_seq_q:
                        lse_value = T.log2(logsum[i]) / scale_log2e + scores_max[i]
                        Logsumexp[lse_index(bz, begin_seq_q, bx * block_M + i, by)] = (
                            lse_value
                        )
                for i in T.Parallel(block_M):
                    logsum[i] = 1 / (logsum[i] + 1e-8)

                for i, j in T.Parallel(block_M, dim_vo):
                    acc_o[i, j] *= logsum[i]

                for i, j in T.Parallel(block_M, dim_vo):
                    if bx * block_M + i < local_seq_q and j < actual_dim_vo:
                        Output[
                            output_index(bz, begin_seq_q, bx * block_M + i, by, j)
                        ] = acc_o[i, j]

    return flashatten_fwd_ws


def _normalize_fwd_shape(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
):
    is_varlen = cu_seqlens_q is not None or cu_seqlens_k is not None
    if is_varlen and (cu_seqlens_q is None or cu_seqlens_k is None):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be provided together")
    if is_varlen:
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("varlen inputs must have shape [total_seq, heads, dim]")
        if cu_seqlens_q.numel() != cu_seqlens_k.numel():
            raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same length")
        if max_seqlen_q is None or max_seqlen_k is None:
            raise ValueError("max_seqlen_q and max_seqlen_k are required for varlen")
        batch = cu_seqlens_q.numel() - 1
        if q.shape[0] > batch * max_seqlen_q:
            raise ValueError("q has more tokens than max_seqlen_q allows")
        if k.shape[0] > batch * max_seqlen_k or v.shape[0] > batch * max_seqlen_k:
            raise ValueError("k/v have more tokens than max_seqlen_k allows")
    else:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("dense inputs must have shape [batch, seqlen, heads, dim]")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError("q, k and v must have the same batch size")
        if max_seqlen_q is not None and max_seqlen_q != q.shape[1]:
            raise ValueError("max_seqlen_q must match dense q sequence length")
        if max_seqlen_k is not None and max_seqlen_k != k.shape[1]:
            raise ValueError("max_seqlen_k must match dense k sequence length")
        if v.shape[1] != k.shape[1]:
            raise ValueError("v and k must have the same sequence length")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension")
    if k.shape[:-1] != v.shape[:-1]:
        raise ValueError("k and v must have the same sequence and head dimensions")
    heads_q, heads_kv = q.shape[-2], k.shape[-2]
    if heads_q <= 0 or heads_kv <= 0 or heads_q % heads_kv != 0:
        raise ValueError("q head count must be a positive multiple of k/v head count")
    _dispatch_class(q.shape[-1], v.shape[-1])
    return is_varlen


def _validate_fwd_interface_options(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    page_table,
    qv,
    q_descale,
    k_descale,
    v_descale,
    window_size,
    learnable_sink,
    attention_chunk,
    softcap,
    scheduler_metadata,
    num_splits,
    pack_gqa,
    deterministic,
    sm_margin,
    cp_world_size,
    cp_rank,
    cp_tot_seqused_k,
    out,
):
    if not (q.is_musa and k.is_musa and v.is_musa):
        raise ValueError("MP32 TileLang FA requires MUSA q, k and v tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k and v must be on the same device")
    if _current_musa_cc(q) != 32:
        raise ValueError("MP32 TileLang FA requires compute capability 3.2")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k and v must have the same dtype")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("MP32 TileLang FA supports float16 and bfloat16")
    for name, tensor in (
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
    ):
        if tensor is not None:
            if not tensor.is_musa or tensor.device != q.device:
                raise ValueError(f"{name} must be a MUSA tensor on q.device")
            if tensor.dtype != torch.int32 or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous torch.int32")
    unsupported = {
        "seqused_q": seqused_q,
        "seqused_k": seqused_k,
        "page_table": page_table,
        "qv": qv,
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
        "learnable_sink": learnable_sink,
        "scheduler_metadata": scheduler_metadata,
        "pack_gqa": pack_gqa,
        "cp_tot_seqused_k": cp_tot_seqused_k,
        "out": out,
    }
    for name, value in unsupported.items():
        if value is not None:
            raise ValueError(f"MP32 TileLang FA does not support {name}")
    if window_size != (-1, -1):
        raise ValueError("MP32 TileLang FA does not support windowed attention")
    if attention_chunk != 0 or softcap != 0.0:
        raise ValueError("MP32 TileLang FA does not support attention_chunk or softcap")
    if num_splits not in (0, -1):
        raise ValueError("MP32 TileLang FA only supports num_splits=0 or -1")
    if deterministic:
        raise ValueError("MP32 TileLang FA forward does not support deterministic=True")
    if sm_margin != 0 or cp_world_size != 1 or cp_rank != 0:
        raise ValueError("MP32 TileLang FA does not support distributed execution")


def flashattn_varlen_fwd_interface(
    q,
    k,
    v,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    seqused_q=None,
    seqused_k=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    learnable_sink=None,
    attention_chunk=0,
    softcap=0.0,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_softmax_lse=False,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqused_k=None,
    out=None,
):
    if window_size is None:
        window_size = (-1, -1)
    else:
        window_size = tuple(window_size)
    if attention_chunk is None:
        attention_chunk = 0

    _validate_fwd_interface_options(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        page_table,
        qv,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        learnable_sink,
        attention_chunk,
        softcap,
        scheduler_metadata,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
        cp_world_size,
        cp_rank,
        cp_tot_seqused_k,
        out,
    )
    _check_attention_strides("q", q)
    _check_attention_strides("k", k)
    _check_attention_strides("v", v)
    is_varlen = _normalize_fwd_shape(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
    )
    if is_varlen:
        batch = cu_seqlens_q.numel() - 1
        total_seq_q = q.shape[0]
        heads = q.shape[1]
        if max_seqlen_q is None or max_seqlen_k is None:
            raise ValueError("max_seqlen_q and max_seqlen_k are required for varlen")
    else:
        batch = q.shape[0]
        total_seq_q = batch * q.shape[1]
        heads = q.shape[2]
        max_seqlen_q = q.shape[1]
        max_seqlen_k = k.shape[1]
        cu_seqlens_dummy = torch.empty((batch + 1,), dtype=torch.int32, device=q.device)
        cu_seqlens_q = cu_seqlens_dummy
        cu_seqlens_k = cu_seqlens_dummy

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    scale = float(softmax_scale)
    # Scores already include the attention scale; exp2 only needs log2(e).
    scale_log2e = 1.4426950408889634
    kernel_factory, kernel_config = _select_kernel(int(q.shape[-1]), int(v.shape[-1]))
    kernel = kernel_factory(
        **kernel_config,
        dtype=_to_tilelang_dtype(q.dtype),
        disable_rs=False,
        is_causal=causal,
        is_varlen=is_varlen,
        heads_q_eq_heads_kv=q.shape[-2] == k.shape[-2],
    )
    if skip_kernel_launch_if_dry_run():
        output = torch.empty(
            (*q.shape[:-1], v.shape[-1]), dtype=q.dtype, device=q.device
        )
        lse_shape = (heads, total_seq_q) if is_varlen else (batch, heads, max_seqlen_q)
        return output, torch.empty(lse_shape, dtype=torch.float32, device=q.device)
    kernel_args = (
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        scale,
        scale_log2e,
    )
    out_flat, softmax_lse = kernel(*kernel_args)
    if out_flat.shape[-1] != v.shape[-1]:
        out_flat = out_flat[..., : v.shape[-1]]
    return out_flat, softmax_lse


__all__ = ["flashattn_varlen_fwd_interface"]
