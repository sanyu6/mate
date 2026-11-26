# SPDX-License-Identifier: Apache-2.0
# mypy: ignore-errors
"""MP32 D128 backward with producer and two consumer groups.

One CTA owns a KV tile. Producer streams Q/dO; c0 computes QK, dOV and dV;
c1 computes dK and dQ. Causal uses bottom-right alignment; persistent is deferred.
"""

import tilelang
import tilelang.language as T

from .._flash_attention_bwd_common import JIT_COMPILE_FLAGS, JIT_PASS_CONFIGS
from ._flash_attention_bwd_common import _make_kernel_types
from ._flash_attention_fwd_common import (
    _assume_attention_strides,
    _perm_g2s,
    _perm_r2r,
)


D128_JIT_PASS_CONFIGS = {
    **JIT_PASS_CONFIGS,
    tilelang.PassConfigKey.TL_ENABLE_FMA_CONTRACTION: True,
}


def _annotate_sqmma(buffer, k_major, continuity, sg=16):
    T.annotate_layout(
        {
            buffer: tilelang.layout.make_sqmma_swizzled_layout(
                buffer,
                k_major=k_major,
                continuity=continuity,
                swizzle_granularity=sg,
            )
        },
        allow_reannotation=True,
        allow_buffer_region=True,
    )


@tilelang.jit(
    target={"kind": "musa", "arch": "mp_32"},
    out_idx=[],
    pass_configs=D128_JIT_PASS_CONFIGS,
    compile_flags=JIT_COMPILE_FLAGS,
)
def flashattn_bwd_d128(
    is_varlen,
    is_causal=False,
    dtype="bfloat16",
    block_M=64,
    block_N=128,
    consumer0_threads=256,
    consumer1_threads=256,
    producer_threads=128,
    heads_q_eq_heads_kv=True,
):
    dim_qk, dim_v = 128, 128
    block_K = 64
    num_stages = 2
    accum_dtype = T.float32
    log2e = 1.4426950408889634
    kernel_types = _make_kernel_types(
        dim_qk, dim_v, is_varlen, dtype, heads_q_eq_heads_kv
    )
    kv_grad_dtype = kernel_types["kv_grad_dtype"]
    q_type = kernel_types["q_type"]
    k_type = kernel_types["k_type"]
    v_type = kernel_types["v_type"]
    do_type = kernel_types["do_type"]
    lse_type = kernel_types["lse_type"]
    delta_type = T.Tensor(lse_type.shape, accum_dtype)
    cu_q_type = kernel_types["cu_q_type"]
    cu_kv_type = kernel_types["cu_kv_type"]
    dq_accum_type = kernel_types["dq_accum_type"]
    dk_type = kernel_types["dk_type"]
    dv_type = kernel_types["dv_type"]
    heads = kernel_types["heads"]
    heads_kv = kernel_types["heads_kv"]
    batch = kernel_types["batch"]
    seqlen_q = kernel_types["seqlen_q"]
    seqlen_kv = kernel_types["seqlen_kv"]
    total_seq_kv = kernel_types["total_seq_kv"]
    actual_dim_qk = kernel_types["actual_dim_qk"]
    actual_dim_v = kernel_types["actual_dim_v"]
    k_stride_s = kernel_types["k_stride_s"]
    k_stride_h = kernel_types["k_stride_h"]
    v_stride_s = kernel_types["v_stride_s"]
    v_stride_h = kernel_types["v_stride_h"]
    k_stride_b = kernel_types.get("k_stride_b", 0)
    v_stride_b = kernel_types.get("v_stride_b", 0)
    input_strides = tuple(
        kernel_types[f"{name}_stride_{axis}"]
        for name in ("q", "k", "v", "do")
        for axis in ("s", "h")
    )
    batch_strides = tuple(
        kernel_types.get(f"{name}_stride_b", 0) for name in ("q", "k", "v", "do")
    )
    threads = consumer0_threads + consumer1_threads + producer_threads

    if producer_threads != 128:
        raise ValueError("D128 producer roles require one 128-thread squad")

    # FullRow output mapping below needs whole squads, each owning N/num_squads
    # rows in multiples of the MP32 fragment's 32-row distribution unit.
    for consumer_threads in (consumer0_threads, consumer1_threads):
        if (
            consumer_threads <= 0
            or consumer_threads % 128
            or block_N % (consumer_threads // 4)
        ):
            raise ValueError("D128 FullRow requires whole squads and 32 rows per squad")

    def input_tile(tensor, b, begin, start, rows, h, d_start, d_end):
        if is_varlen:
            return tensor[begin + start : begin + start + rows, h, d_start:d_end]
        return tensor[b, start : start + rows, h, d_start:d_end]

    def metadata_elem(tensor, b, begin, row, h):
        if is_varlen:
            return tensor[h, begin + row]
        return tensor[b, h, row]

    def kv_tma_tile(tensor, b, begin, start, h, d_start, d_end, rows=block_N):
        kv_head = h if heads_q_eq_heads_kv else h // (heads // heads_kv)
        if is_varlen:
            return tensor[kv_head, begin + start : begin + start + rows, d_start:d_end]
        return tensor[b, kv_head, start : start + rows, d_start:d_end]

    @T.macro
    def store_dkv(
        accum,
        tensor,
        b,
        begin,
        start,
        h,
        dim_start,
        actual_dim,
        local_seq,
        consumer_tid,
        consumer_threads,
    ):
        # FullRow: one squad owns contiguous KV rows. Four lanes share a row;
        # each lane owns columns lane%4, lane%4+4, ... in this D64 slice.
        rows_per_squad = block_N // (consumer_threads // 128)
        T.annotate_layout(
            {
                accum: tilelang.Fragment(
                    (block_N, block_K),
                    forward_thread_fn=lambda i, j: (i // rows_per_squad) * 128
                    + (i % 32) * 4
                    + j % 4,
                    forward_index_fn=lambda i, j: (i % rows_per_squad // 32)
                    * (block_K // 4)
                    + j // 4,
                )
            }
        )
        for row_group in T.unroll(rows_per_squad // 32, explicit=True):
            row = (
                (consumer_tid // 128) * rows_per_squad
                + row_group * 32
                + (consumer_tid % 128) // 4
            )
            if start + row < local_seq:
                for col_group in T.unroll(block_K // 8, explicit=True):
                    # Actual D is a multiple of 8: one guard covers both local values.
                    if dim_start + col_group * 8 < actual_dim:
                        for element in T.unroll(2, explicit=True):
                            col = col_group * 8 + element * 4 + consumer_tid % 4
                            if is_varlen:
                                tensor[begin + start + row, h, dim_start + col] = accum[
                                    row, col
                                ].astype(kv_grad_dtype)
                            else:
                                tensor[b, start + row, h, dim_start + col] = accum[
                                    row, col
                                ].astype(kv_grad_dtype)

    @T.prim_func
    def flash_atten_bwd_mp32_ws(
        Q: q_type,
        K: k_type,
        V: v_type,
        dO: do_type,
        LSE: lse_type,
        Delta: delta_type,
        cu_seq_q: cu_q_type,
        cu_seq_kv: cu_kv_type,
        dQ_accum: dq_accum_type,
        dK: dk_type,
        dV: dv_type,
        max_seqlen_k: T.int32,
        metadata_bytes: T.int64,
        smscale: T.float32,
        rln2_scale: T.float32,
    ):
        with T.Kernel(
            T.ceildiv(max_seqlen_k, block_N), heads, batch, threads=threads
        ) as (bx, by, bz):
            _assume_attention_strides(is_varlen, input_strides, batch_strides)
            # Descriptor dim1 must be KV for inner_stride=2 to permute rows.
            # Metadata views preserve BSHD/THD strides and actual D bounds.
            if is_varlen:
                K_tma = T.alias_buffer(
                    K,
                    shape=(heads_kv, total_seq_kv, actual_dim_qk),
                    strides=(k_stride_h, k_stride_s, 1),
                )
                V_tma = T.alias_buffer(
                    V,
                    shape=(heads_kv, total_seq_kv, actual_dim_v),
                    strides=(v_stride_h, v_stride_s, 1),
                )
            else:
                K_tma = T.alias_buffer(
                    K,
                    shape=(batch, heads_kv, seqlen_kv, actual_dim_qk),
                    strides=(k_stride_b, k_stride_h, k_stride_s, 1),
                )
                V_tma = T.alias_buffer(
                    V,
                    shape=(batch, heads_kv, seqlen_kv, actual_dim_v),
                    strides=(v_stride_b, v_stride_h, v_stride_s, 1),
                )

            # Ordinary K feeds dQ. The other K/V copies use the same g2s rows,
            # so QK and dOV describe the same logical key in each score column.
            K_shared = T.alloc_shared([2, block_K, dim_qk], dtype)
            K_swizzle_shared = T.alloc_shared([2, block_N, block_K], dtype)
            V_swizzle_shared = T.alloc_shared([2, block_N, block_K], dtype)
            P_shared = T.alloc_shared([block_M, block_N], dtype)
            dS_shared = T.alloc_shared([2, block_M, block_K], dtype)
            dS_dk = T.view(dS_shared, [block_M, block_N], dtype)
            Q_shared = T.alloc_shared([num_stages, 2, block_M, block_K], dtype)
            dO_shared = T.alloc_shared([num_stages, 2, block_M, block_K], dtype)
            LSE_shared = T.alloc_shared([num_stages, block_M], accum_dtype)
            Delta_shared = T.alloc_shared([num_stages, block_M], accum_dtype)
            T.annotate_layout(
                {
                    LSE_shared: tilelang.Layout(
                        (num_stages, block_M), lambda s, i: s * block_M + i
                    ),
                    Delta_shared: tilelang.Layout(
                        (num_stages, block_M), lambda s, i: s * block_M + i
                    ),
                }
            )
            _annotate_sqmma(K_shared[0, :, :], False, block_K, sg=32)
            _annotate_sqmma(K_swizzle_shared[0, :, :], True, block_K)
            _annotate_sqmma(V_swizzle_shared[0, :, :], True, block_K)
            _annotate_sqmma(Q_shared[0, 0, :, :], True, block_K)
            _annotate_sqmma(dO_shared[0, 0, :, :], True, block_K)
            _annotate_sqmma(P_shared, False, block_K)
            _annotate_sqmma(dS_shared[0, :, :], True, block_K)

            # Ready counts belong to producer; free counts include every reader.
            # KV is loaded once: kv_free is only needed for persistent scheduling.
            bar_kv_ready = T.alloc_barrier(arrive_count=producer_threads)
            # Two LSE/Delta warps and one Q warp publish Q-ready; one warp loads dO.
            bar_q_ready = T.alloc_barrier([96] * num_stages)
            bar_do_ready = T.alloc_barrier([32] * num_stages)
            bar_q_free = T.alloc_barrier(
                [consumer0_threads + consumer1_threads] * num_stages
            )
            bar_do_free = T.alloc_barrier([consumer0_threads] * num_stages)
            bar_p_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_p_free = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_ds_ready = T.alloc_barrier(arrive_count=consumer0_threads)
            bar_ds_free = T.alloc_barrier(arrive_count=consumer1_threads)
            bar_loop_stream_control = T.alloc_barrier(arrive_count=consumer1_threads)
            # Q ready/free also protect LSE/Delta in the same ring stage.
            # LSE/Delta warps arrive after LDLMS; c0 frees after LMA and QK.
            # The TME-issuing warp arrives only after registering all transfers.
            # Ready waits cover both producer arrivals and TME completion.
            # Barrier lowering inserts the CTA sync after initialization.

            begin_seq_q = T.alloc_var(T.int32)
            begin_seq_kv = T.alloc_var(T.int32)
            local_seq_q = T.alloc_var(T.int32)
            local_seq_kv = T.alloc_var(T.int32)
            begin_seq_q = cu_seq_q[bz] if is_varlen else bz * seqlen_q
            begin_seq_kv = cu_seq_kv[bz] if is_varlen else bz * seqlen_kv
            local_seq_q = cu_seq_q[bz + 1] - begin_seq_q if is_varlen else seqlen_q
            local_seq_kv = cu_seq_kv[bz + 1] - begin_seq_kv if is_varlen else seqlen_kv
            kv_start = bx * block_N
            tid = T.get_thread_binding()

            if kv_start < local_seq_kv:
                if tid < producer_threads:
                    # Producer: fixed KV, then independent LSE/Delta, Q and dO warps.
                    # Free parity is opposite to the consumers' ready parity.
                    q_do_stage = T.alloc_var(T.int32, init=0)
                    phase_q_do = T.alloc_var(T.int32, init=1)
                    for row_panel in T.unroll(2):
                        T.tma_copy(
                            kv_tma_tile(
                                K_tma,
                                bz,
                                begin_seq_kv,
                                kv_start + row_panel * block_K,
                                by,
                                0,
                                dim_qk,
                                rows=block_K,
                            ),
                            K_shared[row_panel, :, :],
                            barrier=bar_kv_ready,
                        )
                    for d in T.unroll(2):
                        T.tma_copy(
                            kv_tma_tile(
                                K_tma,
                                bz,
                                begin_seq_kv,
                                kv_start,
                                by,
                                d * block_K,
                                (d + 1) * block_K,
                            ),
                            K_swizzle_shared[d, :, :],
                            barrier=bar_kv_ready,
                            inner_stride=2,
                        )
                        T.tma_copy(
                            kv_tma_tile(
                                V_tma,
                                bz,
                                begin_seq_kv,
                                kv_start,
                                by,
                                d * block_K,
                                (d + 1) * block_K,
                            ),
                            V_swizzle_shared[d, :, :],
                            barrier=bar_kv_ready,
                            inner_stride=2,
                        )
                    T.barrier_arrive(bar_kv_ready)

                    lse_robust_desc = T.make_robust_desc(
                        T.address_of(metadata_elem(LSE, 0, 0, 0, 0)), metadata_bytes
                    )
                    delta_robust_desc = T.make_robust_desc(
                        T.address_of(metadata_elem(Delta, 0, 0, 0, 0)), metadata_bytes
                    )

                    for q_start in range(0, local_seq_q, block_M):
                        if tid < 64:
                            T.barrier_wait(bar_q_free[q_do_stage], phase_q_do)
                            for i in T.Parallel(block_M):
                                T.copy(
                                    metadata_elem(
                                        LSE, bz, begin_seq_q, q_start + i, by
                                    ),
                                    LSE_shared[q_do_stage, i],
                                    force_async_copy=True,
                                    src_robust_desc=lse_robust_desc,
                                )
                                T.copy(
                                    metadata_elem(
                                        Delta, bz, begin_seq_q, q_start + i, by
                                    ),
                                    Delta_shared[q_do_stage, i],
                                    force_async_copy=True,
                                    src_robust_desc=delta_robust_desc,
                                )
                            T.ptx_commit_group()
                            T.ptx_wait_group(0)
                            T.barrier_arrive(bar_q_ready[q_do_stage])
                        elif tid < 96:
                            T.barrier_wait(bar_q_free[q_do_stage], phase_q_do)
                            if tid == 64:
                                for d in T.unroll(2):
                                    T.tma_copy(
                                        input_tile(
                                            Q,
                                            bz,
                                            begin_seq_q,
                                            q_start,
                                            block_M,
                                            by,
                                            d * block_K,
                                            (d + 1) * block_K,
                                        ),
                                        Q_shared[q_do_stage, d, :, :],
                                        barrier=bar_q_ready[q_do_stage],
                                    )
                            T.barrier_arrive(bar_q_ready[q_do_stage])
                        else:
                            T.barrier_wait(bar_do_free[q_do_stage], phase_q_do)
                            if tid == 96:
                                for d in T.unroll(2):
                                    T.tma_copy(
                                        input_tile(
                                            dO,
                                            bz,
                                            begin_seq_q,
                                            q_start,
                                            block_M,
                                            by,
                                            d * block_K,
                                            (d + 1) * block_K,
                                        ),
                                        dO_shared[q_do_stage, d, :, :],
                                        barrier=bar_do_ready[q_do_stage],
                                    )
                            T.barrier_arrive(bar_do_ready[q_do_stage])

                        q_do_stage += 1
                        if q_do_stage == num_stages:
                            q_do_stage = 0
                            phase_q_do ^= 1

                elif tid < producer_threads + consumer0_threads:
                    # c0: QK -> dOV -> P -> dV; publish dS while dV runs.
                    acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                    acc_dp = T.alloc_fragment([block_M, block_N], accum_dtype)
                    acc_cast = T.alloc_fragment([block_M, block_N], dtype)
                    # Each D64 output consumes one contiguous dO feature slice.
                    acc_dv0 = T.alloc_fragment([block_N, block_K], accum_dtype)
                    acc_dv1 = T.alloc_fragment([block_N, block_K], accum_dtype)
                    lse_reg = T.alloc_fragment([block_M], accum_dtype)
                    delta_reg = T.alloc_fragment([block_M], accum_dtype)
                    phase_consumer0 = T.alloc_var(T.int32, init=0)
                    q_do_stage = T.alloc_var(T.int32, init=0)
                    phase_q_do = T.alloc_var(T.int32, init=0)
                    T.clear(acc_dv0)
                    T.clear(acc_dv1)
                    T.barrier_wait(bar_kv_ready, 0)

                    for q_start in range(0, local_seq_q, block_M):
                        if not is_causal:
                            T.evaluate(
                                T.call_extern(
                                    "void", "__musa_loop_transparent_outermost"
                                )
                            )
                        if q_start > 0:
                            # c1 has submitted iteration i-1; it need not be done.
                            T.barrier_wait(bar_loop_stream_control, phase_consumer0 ^ 1)
                        # Seed the mask once, in the same permuted KV order as QK
                        # and dOV. For valid Q, bottom-right causal also bounds KV.
                        if is_causal:
                            if (
                                kv_start + block_N
                                > q_start + local_seq_kv - local_seq_q + 1
                            ):
                                for i, j in T.Parallel(block_M, block_N):
                                    acc_s[i, j] = T.if_then_else(
                                        kv_start + _perm_g2s(j)
                                        <= q_start + i + local_seq_kv - local_seq_q,
                                        0.0,
                                        -T.infinity(accum_dtype),
                                    )
                            else:
                                T.clear(acc_s)
                        else:
                            if kv_start + block_N > local_seq_kv:
                                for i, j in T.Parallel(block_M, block_N):
                                    acc_s[i, j] = T.if_then_else(
                                        kv_start + _perm_g2s(j) < local_seq_kv,
                                        0.0,
                                        -T.infinity(accum_dtype),
                                    )
                            else:
                                T.clear(acc_s)
                        T.clear(acc_dp)
                        T.barrier_wait(bar_q_ready[q_do_stage], phase_q_do)
                        for i in T.Parallel(block_M):
                            lse_reg[i] = LSE_shared[q_do_stage, i]
                            delta_reg[i] = Delta_shared[q_do_stage, i]
                        _annotate_sqmma(Q_shared[0, 0, :, :], True, block_K)
                        # Independent score accumulators can occupy the queue
                        # together. Each wait(1) retires the older group before
                        # its accumulator is reused by the next D64 slice.
                        T.gemm(
                            Q_shared[q_do_stage, 0, :, :],
                            K_swizzle_shared[0, :, :],
                            acc_s,
                            clear_accum=False,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.barrier_wait(bar_do_ready[q_do_stage], phase_q_do)
                        _annotate_sqmma(dO_shared[0, 0, :, :], True, block_K)
                        T.gemm(
                            dO_shared[q_do_stage, 0, :, :],
                            V_swizzle_shared[0, :, :],
                            acc_dp,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        # Prepare metadata while the first score slices run.
                        # Completion must precede the common Q/metadata free.
                        T.lma_wait()
                        # Metadata depends on LMA, not on score completion.
                        # Mask Q tails and bottom-right causal rows with no keys
                        # once, before waiting for QK to consume acc_s.
                        for i in T.Parallel(block_M):
                            valid_row = (q_start + i < local_seq_q) & (
                                q_start + i + local_seq_kv >= local_seq_q
                                if is_causal
                                else True
                            )
                            lse_reg[i] = T.if_then_else(
                                valid_row, lse_reg[i] * log2e, T.infinity(accum_dtype)
                            )
                            delta_reg[i] = T.if_then_else(
                                valid_row, delta_reg[i] * smscale, 0
                            )
                        T.sched_boundary()
                        T.warpgroup_wait(1)  # QK0 retired before QK1.
                        T.gemm(
                            Q_shared[q_do_stage, 1, :, :],
                            K_swizzle_shared[1, :, :],
                            acc_s,
                            clear_accum=False,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(1)  # dOV0 retired before dOV1.
                        T.gemm(
                            dO_shared[q_do_stage, 1, :, :],
                            V_swizzle_shared[1, :, :],
                            acc_dp,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(1)  # QK1 retired before exp/Q-free.
                        T.barrier_arrive(bar_q_free[q_do_stage])
                        # The second dOV can overlap P computation below.

                        for i, j in T.Parallel(block_M, block_N, coalesced_width=1):
                            acc_s[i, j] = acc_s[i, j] * rln2_scale - lse_reg[i]
                        T.sched_boundary()
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j])
                        T.barrier_wait(bar_p_free, phase_consumer0 ^ 1)
                        # Cast contiguous registers; restore natural KV only on store.
                        T.copy(acc_s, acc_cast)
                        for i, j in T.Parallel(block_M, block_N):
                            P_shared[i, j] = acc_cast[i, _perm_r2r(j)]
                        T.lma_wait()
                        T.barrier_arrive(bar_p_ready)
                        T.warpgroup_wait(0)  # dOV done before consuming acc_dp.
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = (
                                acc_dp[i, j] * smscale - delta_reg[i]
                            ) * acc_s[i, j]
                        T.barrier_wait(bar_p_ready, phase_consumer0)

                        # dO keeps SG16 and D64 continuity across descriptor majors.
                        _annotate_sqmma(dO_shared[0, 0, :, :], False, block_K)
                        T.gemm(
                            P_shared,
                            dO_shared[q_do_stage, 0, :, :],
                            acc_dv0,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.gemm(
                            P_shared,
                            dO_shared[q_do_stage, 1, :, :],
                            acc_dv1,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()

                        # Single-buffer dS: submission alone does NOT free it.
                        T.barrier_wait(bar_ds_free, phase_consumer0 ^ 1)
                        T.copy(acc_s, acc_cast)
                        for i, j in T.Parallel(block_M, block_N):
                            dS_shared[j // block_K, i, j % block_K] = acc_cast[
                                i, _perm_r2r(j)
                            ]
                        T.lma_wait()
                        T.barrier_arrive(bar_ds_ready)

                        T.warpgroup_wait(0)  # dV is the last dO/P reader.
                        T.barrier_arrive(bar_do_free[q_do_stage])
                        T.barrier_arrive(bar_p_free)
                        phase_consumer0 ^= 1
                        q_do_stage += 1
                        if q_do_stage == num_stages:
                            q_do_stage = 0
                            phase_q_do ^= 1

                    # This CTA owns dV outright, including the zero-Q case.
                    store_dkv(
                        acc_dv0,
                        dV,
                        bz,
                        begin_seq_kv,
                        kv_start,
                        by,
                        0,
                        actual_dim_v,
                        local_seq_kv,
                        tid - producer_threads,
                        consumer0_threads,
                    )
                    store_dkv(
                        acc_dv1,
                        dV,
                        bz,
                        begin_seq_kv,
                        kv_start,
                        by,
                        block_K,
                        actual_dim_v,
                        local_seq_kv,
                        tid - producer_threads,
                        consumer0_threads,
                    )

                else:
                    # c1: dK = dS^T Q, then dQ_partial = dS K.
                    acc_dk0 = T.alloc_fragment([block_N, block_K], accum_dtype)
                    acc_dk1 = T.alloc_fragment([block_N, block_K], accum_dtype)
                    acc_dq = T.alloc_fragment([block_M, dim_qk], accum_dtype)
                    phase_consumer1 = T.alloc_var(T.int32, init=0)
                    q_do_stage = T.alloc_var(T.int32, init=0)
                    phase_q_do = T.alloc_var(T.int32, init=0)
                    T.clear(acc_dk0)
                    T.clear(acc_dk1)
                    T.barrier_wait(bar_kv_ready, 0)

                    for q_start in range(0, local_seq_q, block_M):
                        T.barrier_wait(bar_q_ready[q_do_stage], phase_q_do)
                        T.barrier_wait(bar_ds_ready, phase_consumer1)
                        _annotate_sqmma(Q_shared[0, 0, :, :], False, block_K)
                        _annotate_sqmma(dS_dk, False, block_K)
                        T.gemm(
                            dS_dk,
                            Q_shared[q_do_stage, 0, :, :],
                            acc_dk0,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.gemm(
                            dS_dk,
                            Q_shared[q_do_stage, 1, :, :],
                            acc_dk1,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()

                        # The formal transposed-A layout stores dS as two K64 panels.
                        # Accumulate both panels into the original D128 dQ fragment.
                        _annotate_sqmma(dS_shared[0, :, :], True, block_K)
                        _annotate_sqmma(K_shared[0, :, :], False, block_K, sg=32)
                        T.gemm(
                            dS_shared[0, :, :],
                            K_shared[0, :, :],
                            acc_dq,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                            wg_wait=-1,
                        )
                        T.gemm(
                            dS_shared[1, :, :],
                            K_shared[1, :, :],
                            acc_dq,
                            clear_accum=False,
                            policy=T.GemmWarpPolicy.FullCol,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        # Let c0 submit next QK while these GEMMs are in flight.
                        T.barrier_arrive(bar_loop_stream_control)
                        T.warpgroup_wait(1)  # dK done; c1's Q reader can retire.
                        T.barrier_arrive(bar_q_free[q_do_stage])
                        T.warpgroup_wait(0)  # dQ done; both dS readers retired.
                        T.barrier_arrive(bar_ds_free)

                        # Keep the original transposed 4x4 workspace layout, but let
                        # TileLang vectorize four ordinary atomic adds into one x4.
                        # The workspace base is at least 16B aligned; j * 4 selects
                        # a 16B FP32 group. Host padding keeps the group in bounds.
                        for i, j, lane in T.Parallel(
                            block_M,
                            dim_qk // 4,
                            4,
                        ):
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "__builtin_assume",
                                    T.reinterpret(
                                        T.address_of(
                                            dQ_accum[bz, by, q_start + i, j * 4]
                                        ),
                                        dtype="uint64",
                                    )
                                    % 16
                                    == 0,
                                )
                            )
                            T.atomic_add(
                                dQ_accum[
                                    bz,
                                    by,
                                    q_start + i,
                                    j * 4 + lane,
                                ],
                                acc_dq[
                                    i,
                                    j % 4 + j // 4 * 16 + lane * 4,
                                ],
                            )
                        phase_consumer1 ^= 1
                        q_do_stage += 1
                        if q_do_stage == num_stages:
                            q_do_stage = 0
                            phase_q_do ^= 1
                    store_dkv(
                        acc_dk0,
                        dK,
                        bz,
                        begin_seq_kv,
                        kv_start,
                        by,
                        0,
                        actual_dim_qk,
                        local_seq_kv,
                        tid - producer_threads - consumer0_threads,
                        consumer1_threads,
                    )
                    store_dkv(
                        acc_dk1,
                        dK,
                        bz,
                        begin_seq_kv,
                        kv_start,
                        by,
                        block_K,
                        actual_dim_qk,
                        local_seq_kv,
                        tid - producer_threads - consumer0_threads,
                        consumer1_threads,
                    )

    return flash_atten_bwd_mp32_ws
