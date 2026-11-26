"""MP32 (S6000) GDN prefill.

Single fused kernel: kkt_solve (A = (I + diag(beta) tril(K K^T, -1))^-1) runs in an
extra warpgroup, and the optional q/k L2 normalization is folded in through a
read-only rnorm kernel plus a similarity transform, so q and k are never modified.
``checkpoint_every_n_tokens=N`` additionally stores the state every N tokens
(cuDNN frontend GDN semantics).
MP31 keeps using ``mate.gdn_kernels.tilelang.mp31.gdn_prefill`` with the standalone
l2norm and kkt_solve kernels.
"""

import torch
from mate.utils import cosize, tensor_cache
import tilelang
import tilelang.language as T

__all__ = ["fused_chunk_gdn_prefill", "gdn_qk_rnorm"]


_PREFILL_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
    tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
}
_PREFILL_COMPILE_FLAGS = [
    "-Od3",
    "-fno-signed-zeros",
    "-fmusa-flush-denormals-to-zero",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-mtgpu-enable-postra-sched=1",
    "-mllvm",
    "-misched-recompute-slotindex=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
    "-mllvm",
    "-mtgpu-use-sl=1",
    "-mllvm",
    "-mtgpu-combine-instr-with-burst=1",
]


def _fused_chunk_gdn_prefill_builder(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    h0_dtype,
    ht_dtype,
    o_dtype,
    seqlen_dtype,
    use_initial_state,
    store_final_state,
    is_varlen,
    is_log_space,
    l2norm_eps=None,
    store_h=False,
    hs_dtype="float32",
    ckpt_stride=1,
):
    batch_size = T.dynamic("batch_size")
    fold_l2norm = l2norm_eps is not None
    num_tokens = T.dynamic("num_tokens")
    raw_batch_size = T.dynamic("raw_batch_size")
    block_S = chunk_size
    assert DV == 128, "gdn_prefill tilelang kernel currently supports DV=128 only"

    if is_varlen:
        q_shape = (1, num_tokens, Hg, DK)
        k_shape = (1, num_tokens, Hg, DK)
        v_shape = (1, num_tokens, H, DV)
        o_shape = (1, num_tokens, H, DV)
        a_shape = (1, num_tokens, H, chunk_size)
        g_shape = (1, num_tokens, H)
        b_shape = (1, num_tokens, H)
    else:
        q_shape = (batch_size, num_tokens, Hg, DK)
        k_shape = (batch_size, num_tokens, Hg, DK)
        v_shape = (batch_size, num_tokens, H, DV)
        o_shape = (batch_size, num_tokens, H, DV)
        a_shape = (batch_size, num_tokens, H, chunk_size)
        g_shape = (batch_size, num_tokens, H)
        b_shape = (batch_size, num_tokens, H)
    h0_shape = (raw_batch_size if is_varlen else batch_size, H, DK, DV)
    ht_shape = (raw_batch_size, H, DK, DV)
    seqlens_shape = (raw_batch_size + 1,) if is_varlen else (batch_size + 1,)
    rn_shape = (T.dynamic("rn_b"), T.dynamic("rn_t"), Hg, 2)
    hs_shape = (T.dynamic("hs_rows"), H, DV, DK)
    co_shape = (T.dynamic("co_n"),)

    q_stride_b = T.dynamic("q_stride_b")
    q_stride_t = T.dynamic("q_stride_t")
    q_stride_h = T.dynamic("q_stride_h")
    k_stride_b = T.dynamic("k_stride_b")
    k_stride_t = T.dynamic("k_stride_t")
    k_stride_h = T.dynamic("k_stride_h")
    v_stride_b = T.dynamic("v_stride_b")
    v_stride_t = T.dynamic("v_stride_t")
    v_stride_h = T.dynamic("v_stride_h")
    q_strides = (q_stride_b, q_stride_t, q_stride_h, 1)
    k_strides = (k_stride_b, k_stride_t, k_stride_h, 1)
    v_strides = (v_stride_b, v_stride_t, v_stride_h, 1)

    @T.macro
    def producer_phase(i):
        return T.bitwise_xor((i // 2) % 2, 1)

    @T.macro
    def consumer_phase(i):
        return (i // 2) % 2

    @T.macro
    def perm_dv(i):
        stride = DV // 8
        return (i % 8) * stride + (i // 8)

    @T.macro
    def inv_perm_dv(i):
        stride = DV // 8
        return (i % stride) * 8 + (i // stride)

    @T.macro
    def inv_pair_dv(i):
        stride = DV // 2
        return (i % stride) * 2 + (i // stride)

    @T.macro
    def pair_dv(i):
        stride = DV // 2
        return (i % 2) * stride + (i // 2)

    @T.macro
    def qs_fragment_to_natural_dv(i):
        return inv_pair_dv(inv_perm_dv(i))

    @T.macro
    def tau_dv(i):
        # 4x4 transpose inside each 16-column block (an involution).
        return (i // 16) * 16 + (i % 4) * 4 + (i % 16) // 4

    @T.macro
    def perm_s(i):
        stride = block_S // 8
        return (i % 8) * stride + (i // 8)

    @T.prim_func
    def tilelang_fused_chunk_gdn_prefill_kernel(
        q: T.StridedTensor(q_shape, q_strides, qkva_dtype),
        k: T.StridedTensor(k_shape, k_strides, qkva_dtype),
        v: T.StridedTensor(v_shape, v_strides, qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        cu_seqlens: T.Tensor(seqlens_shape, dtype=seqlen_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
        rn: T.Tensor(rn_shape, dtype="float32"),
        hs: T.Tensor(hs_shape, dtype=hs_dtype),
        chunk_offsets: T.Tensor(co_shape, dtype=seqlen_dtype),
    ):
        launch_batch_size = raw_batch_size if is_varlen else batch_size
        with T.Kernel(launch_batch_size * H, threads=1024) as (bbh,):
            bb, bh = bbh // H, bbh % H
            bhg = bh // (H // Hg)

            batch_idx = T.alloc_var("int32")
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")

            batch_idx = 0 if is_varlen else bb
            seq_start_idx = cu_seqlens[bb] if is_varlen else 0
            seq_end_idx = cu_seqlens[bb + 1] if is_varlen else num_tokens

            seq_len = T.alloc_var("int32")
            seq_len = seq_end_idx - seq_start_idx
            num_iters = T.alloc_var("int32")
            num_iters = T.ceildiv(seq_len, block_S)
            chunk_base = T.alloc_var("int32")
            chunk_base = chunk_offsets[bb] if store_h else 0

            q_shared = T.alloc_shared((2, block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((2, block_S, DK), dtype=qkva_dtype)
            kT_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
            pa_shared = T.alloc_shared((2, block_S, block_S), dtype=qkva_dtype)
            g_shared = T.alloc_shared((2, block_S), dtype=accum_dtype, scope="shared")
            b_shared = T.alloc_shared((2, block_S), dtype=accum_dtype, scope="shared")

            h_shared = T.alloc_shared((DK, DV), dtype=qkva_dtype)
            vd_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
            g_exp_shared = T.alloc_shared(
                (2, block_S), dtype=accum_dtype, scope="shared"
            )
            g_exp_rev_shared = T.alloc_shared(
                (2, block_S), dtype=accum_dtype, scope="shared"
            )
            if fold_l2norm:
                rq_shared = T.alloc_shared(
                    (2, block_S), dtype=accum_dtype, scope="shared"
                )
                rk_shared = T.alloc_shared(
                    (2, block_S), dtype=accum_dtype, scope="shared"
                )
            h_fragment = T.alloc_fragment((DK, DV), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((block_S, DV), dtype=accum_dtype)
            uv_fragment = T.alloc_fragment((block_S, DV), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

            # kkt_solve fused into ws6 (A = (I + diag(b) tril(K K^T, -1))^-1).
            a64_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            a32o_fragment = T.alloc_fragment((32, 32), dtype=accum_dtype)
            a64s_shared = T.alloc_shared((block_S, 68), dtype=accum_dtype)
            a32i0_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
            a32i1_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
            a32o1_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
            a32o2_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
            bd_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
            cd_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
            ed_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
            td_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
            T.annotate_layout(
                {a64s_shared: tilelang.layout.make_linear_layout(a64s_shared)}
            )

            q_is_ready = T.alloc_barrier(arrive_count=[32] * 2)
            q_is_free = T.alloc_barrier(arrive_count=[640] * 2)
            k_is_ready = T.alloc_barrier(arrive_count=[32] * 2)
            k_is_free = T.alloc_barrier(arrive_count=[768] * 2)
            gb_is_ready = T.alloc_barrier(arrive_count=[64] * 2)
            gb_is_free = T.alloc_barrier(arrive_count=[896] * 2)
            v_is_ready = T.alloc_barrier(arrive_count=32)
            v_is_free = T.alloc_barrier(arrive_count=128)

            a_is_ready = T.alloc_barrier(arrive_count=[128] * 2)
            a_is_free = T.alloc_barrier(arrive_count=[128] * 2)
            inv_is_ready = T.alloc_barrier(arrive_count=[512] * 2)
            inv_is_free = T.alloc_barrier(arrive_count=[128] * 2)
            p_is_ready = T.alloc_barrier(arrive_count=[512] * 2)
            vd_is_ready = T.alloc_barrier(arrive_count=128)
            vd_is_free = T.alloc_barrier(arrive_count=512)
            h_is_ready = T.alloc_barrier(arrive_count=512)
            h_is_free = T.alloc_barrier(arrive_count=256)
            u_is_done = T.alloc_barrier(arrive_count=128)
            vpv_is_ready = T.alloc_barrier(arrive_count=128)
            kkt_sync = T.alloc_barrier(arrive_count=128)
            kt_is_ready = T.alloc_barrier(arrive_count=32)
            kt_is_free = T.alloc_barrier(arrive_count=512)

            b_bytes = 4 if b_dtype in (torch.float32, "float32") else 2
            b_robust_desc = T.make_robust_desc(
                T.address_of(b[batch_idx, seq_start_idx, 0]),
                cosize((seq_end_idx - seq_start_idx, H)) * b_bytes,
            )

            tx = T.get_thread_binding()

            with T.ws(0):
                vd_vec = T.alloc_local((16,), qkva_dtype)
                vd_tau = T.alloc_local((16,), qkva_dtype)

                w_v = T.alloc_local((32,), qkva_dtype)
                w_o = T.alloc_local((4,), qkva_dtype)
                w_gr = T.alloc_local((2,), accum_dtype)
                for i_s in T.serial(num_iters):
                    T.barrier_wait(h_is_ready, i_s % 2)
                    T.barrier_wait(k_is_ready[i_s % 2], consumer_phase(i_s))
                    # U = K @ S
                    T.gemm(
                        k_shared[i_s % 2, :, :],
                        h_shared,
                        uv_fragment,
                        clear_accum=True,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()

                    # W = V - g * U
                    T.barrier_wait(gb_is_ready[i_s % 2], consumer_phase(i_s))
                    T.barrier_wait(v_is_ready, i_s % 2)
                    T.warpgroup_wait(0)
                    T.barrier_arrive(u_is_done)
                    T.barrier_arrive(k_is_free[i_s % 2])
                    T.barrier_arrive(h_is_free)

                    # W = V - g * U.  h_shared column j holds S column
                    # inv_perm_dv(tau_dv(j)), so the accumulator columns
                    # 8*m + 4*par + lane this thread owns (rows r*32 + tx/4) are
                    # the contiguous V columns 32*lane + e, e = 16*(m%2) + 8*par + m//2.
                    w_row = tx // 4
                    w_lane = tx % 4
                    for w_r in T.unroll(2):
                        for w_e in T.vectorized(32):
                            w_v[w_e] = v_shared[w_r * 32 + w_row, w_lane * 32 + w_e]
                        if fold_l2norm:
                            w_g = (
                                g_exp_shared[i_s % 2, w_r * 32 + w_row]
                                * rk_shared[i_s % 2, w_r * 32 + w_row]
                            )
                        else:
                            w_g = g_exp_shared[i_s % 2, w_r * 32 + w_row]
                        w_gr[w_r] = g_exp_rev_shared[i_s % 2, w_r * 32 + w_row]
                        for w_e in T.unroll(32):
                            w_v[w_e] = T.cast(
                                T.cast(w_v[w_e], accum_dtype)
                                - w_g
                                * uv_fragment[
                                    w_r * 32 + w_row,
                                    (w_e % 8) * 16
                                    + (w_e // 16) * 8
                                    + ((w_e // 8) % 2) * 4
                                    + w_lane,
                                ],
                                qkva_dtype,
                            )
                        for w_e in T.vectorized(32):
                            v_shared[w_r * 32 + w_row, w_lane * 32 + w_e] = w_v[w_e]
                    T.barrier_arrive(gb_is_free[i_s % 2])

                    T.barrier_wait(inv_is_ready[i_s % 2], consumer_phase(i_s))
                    T.lma_wait()
                    # Vd = Ag @ W
                    T.gemm(
                        pa_shared[i_s % 2, :, :],
                        v_shared,
                        uv_fragment,
                        clear_accum=True,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()

                    T.warpgroup_wait(0)
                    T.barrier_arrive(inv_is_free[i_s % 2])
                    T.barrier_wait(vd_is_free, T.bitwise_xor(i_s % 2, 1))
                    # perm_dv(c) = (c % 8) * 16 + c // 8.  With the MP32 MMA
                    # accumulator layout this thread owns rows r*32 + tx/4 and
                    # columns c = 8*m + 4*par + tx%4, i.e. 16 *contiguous*
                    # destination columns per (r, par): store them as vectors.
                    vd_row = tx // 4
                    vd_lane = tx % 4
                    # vd_shared (S update, on the recurrence chain) first ...
                    for vd_r in T.unroll(2):
                        for vd_par in T.unroll(2):
                            for vd_m in T.unroll(16):
                                vd_vec[vd_m] = T.cast(
                                    uv_fragment[
                                        vd_r * 32 + vd_row,
                                        vd_m * 8 + vd_par * 4 + vd_lane,
                                    ]
                                    * w_gr[vd_r],
                                    qkva_dtype,
                                )
                            for vd_m in T.vectorized(16):
                                vd_shared[
                                    vd_r * 32 + vd_row,
                                    (vd_par * 4 + vd_lane) * 16 + vd_m,
                                ] = vd_vec[vd_m]
                    T.lma_wait()
                    T.barrier_arrive(vd_is_ready)
                    # ... then the tau-ordered copy for O += Pg @ Vd (off chain).
                    for vd_r in T.unroll(2):
                        for vd_par in T.unroll(2):
                            for vd_m in T.unroll(16):
                                vd_tau[vd_m] = T.cast(
                                    uv_fragment[
                                        vd_r * 32 + vd_row,
                                        ((vd_m % 4) * 4 + vd_m // 4) * 8
                                        + vd_par * 4
                                        + vd_lane,
                                    ],
                                    qkva_dtype,
                                )
                            for vd_m in T.vectorized(16):
                                v_shared[
                                    vd_r * 32 + vd_row,
                                    (vd_par * 4 + vd_lane) * 16 + vd_m,
                                ] = vd_tau[vd_m]
                    T.lma_wait()
                    T.barrier_arrive(vpv_is_ready)
            with T.ws(1):
                o_vec = T.alloc_local((8,), o_dtype)
                for i_s in T.serial(num_iters):
                    store_left = seq_start_idx + i_s * block_S
                    valid_seqs = T.min(seq_end_idx - store_left, block_S)

                    T.barrier_wait(q_is_ready[i_s % 2], consumer_phase(i_s))
                    T.barrier_wait(h_is_ready, i_s % 2)
                    # Let the on-chain U = K @ S own the tensor cores first.
                    T.barrier_wait(u_is_done, i_s % 2)
                    T.gemm(
                        q_shared[i_s % 2, :, :],
                        h_shared,
                        o_fragment,
                        clear_accum=True,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()

                    T.warpgroup_wait(0)
                    T.barrier_arrive(q_is_free[i_s % 2])
                    T.barrier_arrive(h_is_free)

                    T.barrier_wait(gb_is_ready[i_s % 2], consumer_phase(i_s))
                    for j_s, j_v in T.Parallel(block_S, DV):
                        if fold_l2norm:
                            o_fragment[j_s, j_v] *= (
                                g_exp_shared[i_s % 2, j_s] * scale
                            ) * rq_shared[i_s % 2, j_s]
                        else:
                            o_fragment[j_s, j_v] *= g_exp_shared[i_s % 2, j_s] * scale
                    T.barrier_arrive(gb_is_free[i_s % 2])

                    T.barrier_wait(p_is_ready[i_s % 2], consumer_phase(i_s))
                    T.barrier_wait(vpv_is_ready, i_s % 2)
                    # O += Pg @ Vd
                    T.gemm(
                        pa_shared[i_s % 2, :, :],
                        v_shared,
                        o_fragment,
                        clear_accum=False,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()
                    T.warpgroup_wait(0)
                    T.barrier_arrive(a_is_free[i_s % 2])
                    T.barrier_arrive(v_is_free)
                    # S2[S] O.  o_fragment's DV register order is not natural;
                    # remap it while storing directly to the global output.
                    # inv_perm_dv(c) = (c % 16) * 8 + c // 16; this thread owns
                    # c = 16*j + 4*q + tx%4 -> 8 contiguous output columns per q.
                    o_row = (tx // 4) % 32
                    o_lane = tx % 4
                    for o_r in T.unroll(2):
                        for o_q in T.unroll(4):
                            for o_j in T.unroll(8):
                                o_vec[o_j] = T.cast(
                                    o_fragment[
                                        o_r * 32 + o_row, o_j * 16 + o_q * 4 + o_lane
                                    ],
                                    o_dtype,
                                )
                            if o_r * 32 + o_row < valid_seqs:
                                for o_j in T.vectorized(8):
                                    o[
                                        batch_idx,
                                        store_left + o_r * 32 + o_row,
                                        bh,
                                        (o_lane * 4 + o_q) * 8 + o_j,
                                    ] = o_vec[o_j]

            with T.ws(2, 3, 4, 5):
                # Initialize S
                if use_initial_state:
                    # 64-bit row: B * H states can exceed 2**31 bytes.
                    for j_k, j_v in T.Parallel(DV, DK):
                        h_fragment[j_k, j_v] = h0[
                            T.cast(bb, "int64"), bh, inv_perm_dv(j_v), j_k
                        ]
                else:
                    for j_k, j_v in T.Parallel(DK, DV):
                        h_fragment[j_k, j_v] = 0.0
                # Main Loop
                for i_s in T.serial(num_iters):
                    valid_seqs = T.min(
                        seq_end_idx - seq_start_idx - i_s * block_S, block_S
                    )
                    T.barrier_wait(h_is_free, T.bitwise_xor(i_s % 2, 1))
                    for j_k, j_v in T.Parallel(DK, DV):
                        h_shared[j_k, tau_dv(j_v)] = h_fragment[j_k, j_v]
                    T.lma_wait()
                    T.barrier_arrive(h_is_ready)
                    if store_h:
                        # Checkpoint = S before tokens 0, N, 2N, ... of this sequence (same
                        # index map as ht). 64-bit row: checkpoints can exceed 2**31 bytes.
                        if i_s % ckpt_stride == 0:
                            for j_k, j_v in T.Parallel(DV, DK):
                                hs[
                                    T.cast(chunk_base + i_s // ckpt_stride, "int64"),
                                    bh,
                                    inv_perm_dv(j_v),
                                    j_k,
                                ] = T.cast(h_fragment[j_k, j_v], hs_dtype)

                    T.barrier_wait(gb_is_ready[i_s % 2], consumer_phase(i_s))
                    # G = Lower(diag(g) @ I @ diag(1/g))
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        g_fragment[j_s, j_t] = (
                            g_shared[i_s % 2, j_s] - g_shared[i_s % 2, j_t]
                        )
                    # Causal mask by token order, not by the sign of G_s - G_t: the
                    # scanned cumsum is not monotone in fp32, so with g == 0 (or |g|
                    # below one ulp of G) the sign test lets future tokens in or drops
                    # past ones. Rounding-positive values become 0; inf / NaN from
                    # padded lanes become -inf.
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        g_fragment[j_s, j_t] = T.if_then_else(
                            j_t <= j_s,
                            T.if_then_else(
                                g_fragment[j_s, j_t] <= 0.0,
                                g_fragment[j_s, j_t],
                                T.if_then_else(
                                    g_fragment[j_s, j_t] < 1.0, 0.0, -float("inf")
                                ),
                            ),
                            -float("inf"),
                        )
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        g_fragment[j_s, j_t] = T.exp2(
                            g_fragment[j_s, j_t] * 1.4426950408889634
                        )

                    # Ag = G * Ar * b
                    T.barrier_wait(a_is_ready[i_s % 2], consumer_phase(i_s))
                    if fold_l2norm:
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            pa_shared[i_s % 2, j_s, j_t] = (
                                g_fragment[j_s, j_t]
                                * pa_shared[i_s % 2, j_s, j_t]
                                * (b_shared[i_s % 2, j_t] * rk_shared[i_s % 2, j_t])
                            )
                    else:
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            pa_shared[i_s % 2, j_s, j_t] = (
                                g_fragment[j_s, j_t]
                                * pa_shared[i_s % 2, j_s, j_t]
                                * b_shared[i_s % 2, j_t]
                            )
                    T.lma_wait()
                    T.barrier_arrive(inv_is_ready[i_s % 2])
                    T.barrier_wait(k_is_ready[i_s % 2], consumer_phase(i_s))
                    T.barrier_wait(q_is_ready[i_s % 2], consumer_phase(i_s))
                    # P = Q K^T is off the chain: issue it after U = K @ S.
                    T.barrier_wait(u_is_done, i_s % 2)
                    T.gemm(
                        q_shared[i_s % 2, :, :],
                        k_shared[i_s % 2, :, :],
                        p_fragment,
                        transpose_B=True,
                        clear_accum=True,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()
                    T.warpgroup_wait(0)
                    if fold_l2norm:
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            p_fragment[j_s, j_t] *= rq_shared[i_s % 2, j_s]
                    T.barrier_arrive(q_is_free[i_s % 2])
                    T.barrier_arrive(k_is_free[i_s % 2])
                    g_last_local = g_exp_shared[i_s % 2, valid_seqs - 1]
                    for j_k, j_v in T.Parallel(DK, DV):
                        h_fragment[j_k, j_v] *= g_last_local
                    T.barrier_arrive(gb_is_free[i_s % 2])

                    T.barrier_wait(inv_is_free[i_s % 2], consumer_phase(i_s))
                    # Pg = s * G * P
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        pa_shared[i_s % 2, j_s, j_t] = (
                            g_fragment[j_s, j_t] * p_fragment[j_s, j_t] * scale
                        )
                    T.lma_wait()
                    T.barrier_arrive(p_is_ready[i_s % 2])

                    T.barrier_wait(vd_is_ready, i_s % 2)
                    T.barrier_wait(kt_is_ready, i_s % 2)
                    # h += K^T (g_rev o Vd): raw K (second TMA, MN-major), ws0 folded g_rev into vd_shared.
                    T.gemm(
                        kT_shared,
                        vd_shared,
                        h_fragment,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=False,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()
                    T.warpgroup_wait(0)
                    T.barrier_arrive(kt_is_free)
                    T.barrier_arrive(vd_is_free)
                # Store final S
                if store_final_state:
                    for j_k, j_v in T.Parallel(DV, DK):
                        ht[T.cast(bb, "int64"), bh, inv_perm_dv(j_v), j_k] = h_fragment[
                            j_k, j_v
                        ]

            with T.ws(6):
                kt = tx - 768
                # zero regions of the block-diagonal operands / A are never rewritten.
                for z_r, z_c in T.Parallel(32, 32):
                    bd_shared[z_r, z_c] = 0
                    cd_shared[z_r, z_c] = 0
                    ed_shared[z_r, z_c] = 0
                    a32i0_shared[z_r, z_c] = 0
                    a32i1_shared[z_r, z_c] = 0
                for z_q, z_r, z_c in T.Parallel(2, block_S, block_S):
                    pa_shared[z_q, z_r, z_c] = 0
                for i_s in T.serial(num_iters):
                    valid_seqs = T.min(
                        seq_end_idx - seq_start_idx - i_s * block_S, block_S
                    )
                    T.barrier_wait(k_is_ready[i_s % 2], consumer_phase(i_s))
                    # A = K @ K^T
                    T.gemm(
                        k_shared[i_s % 2, :, :],
                        k_shared[i_s % 2, :, :],
                        a64_fragment,
                        transpose_B=True,
                        clear_accum=True,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()
                    T.warpgroup_wait(0)
                    T.barrier_arrive(k_is_free[i_s % 2])
                    T.barrier_wait(gb_is_ready[i_s % 2], consumer_phase(i_s))
                    # raw K K^T; -diag(b) (and rk) scaling + mask applied on row reads.
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        # readers: diag 16-blocks, N10 blocks (1,0)/(3,2), rows 32-63 x cols 0-31 (a32o1).
                        if (j_t < 32 or j_s >= 32) and not (
                            j_s // 32 == j_t // 32 and j_s % 32 < 16 and j_t % 32 >= 16
                        ):
                            a64s_shared[j_s, j_t] = a64_fragment[j_s, j_t]
                    T.lma_wait()
                    T.barrier_arrive(kkt_sync)
                    T.barrier_wait(kkt_sync, i_s % 2)

                    T.barrier_wait(a_is_free[i_s % 2], producer_phase(i_s))
                    # 16x16 diagonal inverses, one warp each (warp w owns rows 16w..16w+15).
                    w16 = kt // 32
                    d16 = T.shift_right(T.bitwise_and(kt, 64), 6)
                    h16 = w16 % 2
                    r16 = kt % 16
                    diag_row16 = T.alloc_local([16], accum_dtype)
                    kk_sc = T.alloc_local([1], accum_dtype)
                    y16 = T.alloc_local([16], qkva_dtype)
                    x8 = T.alloc_local([8], accum_dtype)
                    o8 = T.alloc_local([8], accum_dtype)
                    hh16 = (kt % 32) // 16
                    kk_sc[0] = -b_shared[i_s % 2, 32 * (d16) + 16 * h16 + r16]
                    if fold_l2norm:
                        kk_sc[0] = kk_sc[0] * (
                            rk_shared[i_s % 2, 32 * (d16) + 16 * h16 + r16]
                            * rk_shared[i_s % 2, 32 * (d16) + 16 * h16 + r16]
                        )
                    if (kt % 32) >= 16:
                        if h16 == 1:
                            # N10 row of this 32-block (row 32*d16+16+r16, cols 32*d16..+15)
                            for k_t in T.vectorized(16):
                                diag_row16[k_t] = a64s_shared[
                                    32 * (d16) + 16 + r16, 32 * (d16) + k_t
                                ]
                            for k_t in T.unroll(16):
                                y16[k_t] = T.cast(
                                    diag_row16[k_t] * kk_sc[0], qkva_dtype
                                )
                            for k_t in T.vectorized(16):
                                bd_shared[16 * d16 + r16, 16 * d16 + k_t] = y16[k_t]
                    for k_t in T.vectorized(16):
                        diag_row16[k_t] = a64s_shared[
                            32 * (d16) + 16 * h16 + r16, 32 * (d16) + 16 * h16 + k_t
                        ]
                    for k_t in T.unroll(16):
                        diag_row16[k_t] = T.if_then_else(
                            k_t < r16, diag_row16[k_t] * kk_sc[0], 0.0
                        )
                    # half hh16 owns columns 2j+hh16 (x8, updated); o8 keeps the other half's original M values (row scales).
                    for j in T.unroll(8):
                        x8[j] = T.if_then_else(
                            hh16 == 1, diag_row16[2 * j + 1], diag_row16[2 * j]
                        )
                        o8[j] = T.if_then_else(
                            hh16 == 1, diag_row16[2 * j], diag_row16[2 * j + 1]
                        )
                    for s_m1 in T.unroll(14):
                        row_scale = T.if_then_else(
                            hh16 == (s_m1 + 1) % 2,
                            x8[(s_m1 + 1) // 2],
                            o8[(s_m1 + 1) // 2],
                        )
                        for j in T.unroll((s_m1 + 2) // 2):
                            src_row_value = T.shfl_sync(0xFFFFFFFF, x8[j], s_m1 + 1, 16)
                            x8[j] += row_scale * src_row_value
                    for j in T.unroll(8):
                        o8[j] = T.shfl_xor(x8[j], 16)
                        diag_row16[2 * j] = T.if_then_else(hh16 == 1, o8[j], x8[j])
                        diag_row16[2 * j + 1] = T.if_then_else(hh16 == 1, x8[j], o8[j])
                    # MUSA backend workaround: without a (never-taken) consumer of the reassembled row in lanes 16-31, the
                    # column-split inverse is miscompiled (do~1e-2). the guard is never true (grid < 2^31-16).
                    if bbh >= 2147483632:
                        for k_t in T.unroll(16):
                            ht[0, 0, kt, k_t] = diag_row16[k_t]
                    for k_t in T.unroll(16):
                        diag_row16[k_t] = T.if_then_else(
                            k_t == r16, 1.0, diag_row16[k_t]
                        )
                    if (kt % 32) < 16:
                        for k_t in T.unroll(16):
                            y16[k_t] = T.cast(diag_row16[k_t], qkva_dtype)
                        for k_t in T.vectorized(16):
                            pa_shared[i_s % 2, 16 * w16 + r16, 16 * w16 + k_t] = y16[
                                k_t
                            ]
                        if d16 == 0:
                            for k_t in T.vectorized(16):
                                a32i0_shared[16 * h16 + r16, 16 * h16 + k_t] = y16[k_t]
                        else:
                            for k_t in T.vectorized(16):
                                a32i1_shared[16 * h16 + r16, 16 * h16 + k_t] = y16[k_t]
                        if h16 == 0:
                            for k_t in T.vectorized(16):
                                cd_shared[16 * d16 + r16, 16 * d16 + k_t] = y16[k_t]
                        else:
                            for k_t in T.vectorized(16):
                                ed_shared[16 * d16 + r16, 16 * d16 + k_t] = y16[k_t]
                    T.lma_wait()
                    T.barrier_arrive(kkt_sync)
                    T.barrier_wait(kkt_sync, (i_s + 1) % 2)
                    # T = blockdiag(N10) @ blockdiag(Y00)
                    T.gemm(
                        bd_shared,
                        cd_shared,
                        a32o_fragment,
                        clear_accum=True,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()
                    kk_sc[0] = -b_shared[i_s % 2, 32 + kt // 4]
                    if fold_l2norm:
                        kk_sc[0] = kk_sc[0] * (
                            rk_shared[i_s % 2, 32 + kt // 4]
                            * rk_shared[i_s % 2, 32 + kt // 4]
                        )
                    for j in T.unroll(8):
                        a32o1_shared[kt // 4, (kt % 4) * 8 + j] = (
                            a64s_shared[32 + kt // 4, (kt % 4) * 8 + j] * kk_sc[0]
                        )
                    T.warpgroup_wait(0)
                    for k_s, k_t in T.Parallel(32, 32):
                        td_shared[k_s, k_t] = a32o_fragment[k_s, k_t]
                    T.lma_wait()
                    # last b/rk read of this chunk is the a32o1 conversion above.
                    T.barrier_arrive(gb_is_free[i_s % 2])
                    T.barrier_arrive(kkt_sync)
                    T.barrier_wait(kkt_sync, i_s % 2)
                    # Y10 = blockdiag(Y11) @ T
                    T.gemm(
                        ed_shared,
                        td_shared,
                        a32o_fragment,
                        clear_accum=True,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()
                    T.warpgroup_wait(0)
                    for k_s, k_t in T.Parallel(32, 32):
                        if k_s < 16:
                            if k_t < 16:
                                a32i0_shared[16 + k_s, k_t] = a32o_fragment[k_s, k_t]
                                pa_shared[i_s % 2, 16 + k_s, k_t] = a32o_fragment[
                                    k_s, k_t
                                ]
                        else:
                            if k_t >= 16:
                                a32i1_shared[k_s, k_t - 16] = a32o_fragment[k_s, k_t]
                                pa_shared[i_s % 2, 32 + k_s, 16 + k_t] = a32o_fragment[
                                    k_s, k_t
                                ]
                    T.lma_wait()
                    T.barrier_arrive(kkt_sync)
                    T.barrier_wait(kkt_sync, (i_s + 1) % 2)
                    T.gemm(
                        a32i1_shared,
                        a32o1_shared,
                        a32o_fragment,
                        clear_accum=True,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()
                    T.warpgroup_wait(0)
                    for k_s, k_t in T.Parallel(32, 32):
                        a32o2_shared[k_s, k_t] = a32o_fragment[k_s, k_t]
                    T.lma_wait()
                    T.barrier_arrive(kkt_sync)
                    T.barrier_wait(kkt_sync, i_s % 2)
                    T.gemm(
                        a32o2_shared,
                        a32i0_shared,
                        a32o_fragment,
                        clear_accum=True,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()
                    T.warpgroup_wait(0)
                    for k_s, k_t in T.Parallel(32, 32):
                        pa_shared[i_s % 2, 32 + k_s, k_t] = a32o_fragment[k_s, k_t]
                    T.lma_wait()
                    T.barrier_arrive(a_is_ready[i_s % 2])
            with T.ws(7):
                if tx >= 992:
                    for i_s in T.serial(num_iters):
                        load_left = seq_start_idx + i_s * block_S
                        load_right = load_left + block_S
                        T.barrier_wait(v_is_free, T.bitwise_xor(i_s % 2, 1))
                        if tx == 992:
                            T.tma_copy(
                                v[
                                    batch_idx,
                                    load_left:load_right,
                                    bh,
                                    :,
                                ],
                                v_shared,
                                barrier=v_is_ready,
                            )
                        T.barrier_arrive(v_is_ready)
                        T.barrier_wait(kt_is_free, T.bitwise_xor(i_s % 2, 1))
                        if tx == 992:
                            T.tma_copy(
                                k[batch_idx, load_left:load_right, bhg, 0:DK],
                                kT_shared,
                                barrier=kt_is_ready,
                            )
                        T.barrier_arrive(kt_is_ready)
                elif tx >= 960:
                    for i_s in T.serial(num_iters):
                        load_left = seq_start_idx + i_s * block_S
                        load_right = load_left + block_S
                        T.barrier_wait(k_is_free[i_s % 2], producer_phase(i_s))
                        if tx == 960:
                            T.tma_copy(
                                k[batch_idx, load_left:load_right, bhg, 0:DK],
                                k_shared[i_s % 2, :, :],
                                barrier=k_is_ready[i_s % 2],
                            )
                        T.barrier_arrive(k_is_ready[i_s % 2])
                        T.barrier_wait(q_is_free[i_s % 2], producer_phase(i_s))
                        if tx == 960:
                            T.tma_copy(
                                q[batch_idx, load_left:load_right, bhg, 0:DK],
                                q_shared[i_s % 2, :, :],
                                barrier=q_is_ready[i_s % 2],
                            )
                        T.barrier_arrive(q_is_ready[i_s % 2])
                else:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(gb_is_free[i_s % 2], producer_phase(i_s))
                        load_left = seq_start_idx + i_s * block_S
                        load_right = load_left + block_S
                        valid_seqs = T.min(
                            seq_end_idx - seq_start_idx - i_s * block_S, block_S
                        )
                        T.copy(
                            b[batch_idx, load_left:load_right, bh],
                            b_shared[i_s % 2, :],
                            force_async_copy=True,
                            src_robust_desc=b_robust_desc,
                        )
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        T.sync_threads(1, 64)
                        gb_lane = tx - 896
                        gb_lane_in_warp = gb_lane % 32
                        if fold_l2norm:
                            # Clamp the row: the backend fuses both guarded loads into one
                            # predicated load that still reads past the last token, so a
                            # NaN after rn would poison the padded rows of the tail chunk.
                            rn_row = T.min(load_left + gb_lane, num_tokens - 1)
                            rq_shared[i_s % 2, gb_lane] = T.if_then_else(
                                load_left + gb_lane < num_tokens,
                                rn[batch_idx, rn_row, bhg, 0],
                                0.0,
                            )
                            rk_shared[i_s % 2, gb_lane] = T.if_then_else(
                                load_left + gb_lane < num_tokens,
                                rn[batch_idx, rn_row, bhg, 1],
                                0.0,
                            )
                        g_prefix = T.alloc_var(accum_dtype)
                        g_peer = T.alloc_var(accum_dtype)
                        if is_log_space:
                            g_prefix = T.if_then_else(
                                gb_lane < valid_seqs,
                                g[batch_idx, load_left + gb_lane, bh],
                                -float("inf"),
                            )
                        else:
                            g_prefix = T.log(
                                T.if_then_else(
                                    gb_lane < valid_seqs,
                                    g[batch_idx, load_left + gb_lane, bh],
                                    1e-43,
                                )
                            )
                        for j_shift in T.unroll(5):
                            j_offset = 1 << j_shift
                            g_peer = T.shfl_up(g_prefix, j_offset)
                            if gb_lane_in_warp >= j_offset:
                                g_prefix += g_peer
                        g_shared[i_s % 2, gb_lane] = g_prefix
                        T.lma_wait()
                        T.sync_threads(1, 64)
                        if gb_lane >= 32:
                            g_prefix += g_shared[i_s % 2, 31]
                            g_shared[i_s % 2, gb_lane] = g_prefix
                        T.lma_wait()
                        T.sync_threads(1, 64)
                        for j_s in T.Parallel(block_S):
                            g_exp_rev_shared[i_s % 2, j_s] = (
                                g_shared[i_s % 2, valid_seqs - 1]
                                - g_shared[i_s % 2, j_s]
                            )
                        for j_s in T.Parallel(block_S):
                            # Same rounding guard: a valid token must not be dropped
                            # from the state update; padded lanes (+inf) stay -inf.
                            g_exp_rev_shared[i_s % 2, j_s] = T.if_then_else(
                                g_exp_rev_shared[i_s % 2, j_s] <= 0,
                                g_exp_rev_shared[i_s % 2, j_s],
                                T.if_then_else(
                                    g_exp_rev_shared[i_s % 2, j_s] < 1.0,
                                    0.0,
                                    float("-inf"),
                                ),
                            )

                        for j_s in T.Parallel(block_S):
                            g_exp_rev_shared[i_s % 2, j_s] = T.exp2(
                                g_exp_rev_shared[i_s % 2, j_s] * 1.4426950408889634
                            )
                            g_exp_shared[i_s % 2, j_s] = T.exp2(
                                g_shared[i_s % 2, j_s] * 1.4426950408889634
                            )
                        T.lma_wait()
                        T.barrier_arrive(gb_is_ready[i_s % 2])

    def _symbol_part(value):
        return (
            str(value)
            .replace("torch.", "")
            .replace(".", "p")
            .replace("-", "m")
            .replace(" ", "_")
        )

    symbol = (
        "tilelang_fused_chunk_gdn_prefill_kernel"
        f"_h{H}_hg{Hg}_dk{DK}_dv{DV}_cs{chunk_size}"
        f"_s{_symbol_part(scale)}"
        f"_q{_symbol_part(qkva_dtype)}"
        f"_g{_symbol_part(g_dtype)}"
        f"_b{_symbol_part(b_dtype)}"
        f"_h0{_symbol_part(h0_dtype)}"
        f"_ht{_symbol_part(ht_dtype)}"
        f"_o{_symbol_part(o_dtype)}"
        f"_seq{_symbol_part(seqlen_dtype)}"
        f"_init{int(use_initial_state)}_final{int(store_final_state)}_var{int(is_varlen)}"
        f"_log{int(is_log_space)}"
        "_vdv_ov_dop_tau_sp_f56x12_k87b"
        f"_l2n{int(fold_l2norm)}"
        f"_hs{int(store_h)}{_symbol_part(hs_dtype) if store_h else ''}"
        f"{f'_ck{ckpt_stride}' if store_h else ''}"
    )
    return tilelang_fused_chunk_gdn_prefill_kernel.with_attr("global_symbol", symbol)


tilelang_fused_chunk_gdn_prefill = tilelang.jit(
    pass_configs=_PREFILL_PASS_CONFIGS, compile_flags=_PREFILL_COMPILE_FLAGS
)(_fused_chunk_gdn_prefill_builder)
# store_h: machine LICM hoists the per-element h store offsets out of the chunk loop and spills them.
tilelang_fused_chunk_gdn_prefill_hs = tilelang.jit(
    pass_configs=_PREFILL_PASS_CONFIGS,
    compile_flags=_PREFILL_COMPILE_FLAGS + ["-mllvm", "-disable-machine-licm"],
)(_fused_chunk_gdn_prefill_builder)


_RNORM_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
    tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
}
_RNORM_COMPILE_FLAGS = [
    "-O3",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-misched-recompute-slotindex=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
]


def _gdn_qk_rnorm_builder(
    qkva_dtype, rows_per_block, lanes_per_row, large_offsets=False
):
    """rn[b, t, h, 0] = rsqrt(|q[b,t,h,:]|^2 + eps), rn[..., 1] = same for k (inputs untouched)."""
    batch = T.dynamic("batch")
    num_tokens = T.dynamic("num_tokens")
    num_heads = T.dynamic("num_heads")
    D = 128
    vec_size = D // lanes_per_row
    num_shuffles = lanes_per_row.bit_length() - 1
    threads = rows_per_block * lanes_per_row
    q_strides = (
        T.dynamic("q_stride_b"),
        T.dynamic("q_stride_t"),
        T.dynamic("q_stride_h"),
        1,
    )
    k_strides = (
        T.dynamic("k_stride_b"),
        T.dynamic("k_stride_t"),
        T.dynamic("k_stride_h"),
        1,
    )
    x_shape = (batch, num_tokens, num_heads, D)

    @T.prim_func
    def tilelang_gdn_qk_rnorm_kernel(
        q: T.StridedTensor(x_shape, q_strides, qkva_dtype),
        k: T.StridedTensor(x_shape, k_strides, qkva_dtype),
        rn: T.Tensor((batch, num_tokens, num_heads, 2), dtype="float32"),
        eps: T.float32,
    ):
        total_rows = batch * num_tokens * num_heads
        with T.Kernel(T.ceildiv(total_rows, rows_per_block), threads=threads) as (
            block_idx,
        ):
            tid = T.get_thread_binding()
            lane = tid % lanes_per_row
            row = block_idx * rows_per_block + tid // lanes_per_row
            b_i = T.alloc_var("int32")
            t_i = T.alloc_var("int32")
            h_i = T.alloc_var("int32")
            j_d = T.alloc_var("int32")
            xq = T.alloc_local((vec_size,), dtype="float32")
            xk = T.alloc_local((vec_size,), dtype="float32")
            nq = T.alloc_local((1,), dtype="float32")
            nk = T.alloc_local((1,), dtype="float32")
            if row < total_rows:
                b_i = row // (num_tokens * num_heads)
                t_i = (row // num_heads) % num_tokens
                h_i = row % num_heads
                nq[0] = 0.0
                nk[0] = 0.0
                for i in T.vectorized(vec_size):
                    j_d = lane * vec_size + i
                    xq[i] = T.cast(q[b_i, t_i, h_i, j_d], "float32")
                    nq[0] += xq[i] * xq[i]
                for i in T.vectorized(vec_size):
                    j_d = lane * vec_size + i
                    xk[i] = T.cast(k[b_i, t_i, h_i, j_d], "float32")
                    nk[0] += xk[i] * xk[i]
                for offset in T.unroll(num_shuffles):
                    nq[0] += T.shfl_xor(nq[0], (lanes_per_row // 2) >> offset)
                    nk[0] += T.shfl_xor(nk[0], (lanes_per_row // 2) >> offset)
                if lane == 0:
                    rn[b_i, t_i, h_i, 0] = T.rsqrt(nq[0] + eps)
                    rn[b_i, t_i, h_i, 1] = T.rsqrt(nk[0] + eps)

    symbol = (
        f"tilelang_gdn_qk_rnorm2_kernel_q{str(qkva_dtype).replace('torch.', '')}"
        f"_r{rows_per_block}_l{lanes_per_row}"
        f"{'_i64' if large_offsets else ''}"
    )
    return tilelang_gdn_qk_rnorm_kernel.with_attr("global_symbol", symbol)


def _without_tiny_offset_hint(flags):
    i = flags.index("-mtgpu-tiny-offset-hint=1")
    assert flags[i - 1] == "-mllvm"
    return flags[: i - 1] + flags[i + 1 :]


tilelang_gdn_qk_rnorm = tilelang.jit(
    pass_configs=_RNORM_PASS_CONFIGS, compile_flags=_RNORM_COMPILE_FLAGS
)(_gdn_qk_rnorm_builder)
# The tiny-offset hint keeps byte offsets in 32 bits, which faults once q/k span
# 2**31 bytes (e.g. split from a large fused qkv); that build is ~20% slower, so it
# is only used for such inputs.
tilelang_gdn_qk_rnorm_large = tilelang.jit(
    pass_configs=_RNORM_PASS_CONFIGS,
    compile_flags=_without_tiny_offset_hint(_RNORM_COMPILE_FLAGS),
)(_gdn_qk_rnorm_builder)


_RN_DUMMY = {}
_HS_DUMMY = {}


@tensor_cache
def _checkpoint_offsets(cu_seqlens: torch.Tensor, every_n_tokens: int):
    # First checkpoint row of every sequence: ceil(len / N) rows each, packed back
    # to back. Computed on the device, so no host read of cu_seqlens.
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    offsets = torch.zeros_like(cu_seqlens)
    offsets[1:] = torch.cumsum((lens + every_n_tokens - 1) // every_n_tokens, 0)
    return offsets


def _byte_span(x: torch.Tensor) -> int:
    if x.numel() == 0:
        return 0
    last = sum((size - 1) * stride for size, stride in zip(x.shape, x.stride()))
    return (last + 1) * x.element_size()


def gdn_qk_rnorm(q, k, eps, rows_per_block=16, lanes_per_row=4):
    B, Tn, Hg, D = k.shape
    assert q.shape == k.shape and D == 128 and q.stride(-1) == k.stride(-1) == 1
    rn = torch.empty((B, Tn, Hg, 2), dtype=torch.float32, device=k.device)
    if max(_byte_span(q), _byte_span(k), _byte_span(rn)) > 2**31:
        kernel = tilelang_gdn_qk_rnorm_large(
            q.dtype, rows_per_block, lanes_per_row, large_offsets=True
        )
    else:
        kernel = tilelang_gdn_qk_rnorm(q.dtype, rows_per_block, lanes_per_row)
    kernel(q, k, rn, float(eps))
    return rn


def fused_chunk_gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor | None = None,
    output_state: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    output_o: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    is_log_space: bool = True,
    l2norm_eps: float | None = None,
    checkpoint_every_n_tokens: int = 0,
):
    """Fused GDN prefill for MP32.

    ``a`` is accepted for signature compatibility with the MP31 kernel and ignored:
    the kkt solve happens inside the kernel.  With ``l2norm_eps`` set, q and k are
    L2-normalized as ``x * rsqrt(sum(x^2) + eps)`` inside the kernel without being
    modified.

    Returns ``(o, state_checkpoints, final_state)``.  ``checkpoint_every_n_tokens``
    follows the cuDNN frontend GDN op: 0 disables checkpoints (``None`` is returned);
    otherwise it must be a positive multiple of ``chunk_size`` and every sequence
    stores the state *before* its tokens ``0, N, 2N, ...`` (the first row is the
    initial state, the final state is not repeated), ``ceil(len / N)`` rows per
    sequence packed back to back in sequence order.  ``state_checkpoints`` has the
    q/k/v dtype and shape ``[rows, H, V, K]`` (indexed like ``final_state``) with
    ``rows = max(total_tokens // N + num_seqs, 1)`` for varlen inputs, an upper
    bound that avoids a host read of ``cu_seqlens`` (trailing rows are left
    unwritten), and ``rows = B * ceil(T / N)`` for fixed-length inputs.
    """
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H, V = v.shape
    scale = scale or K ** (-0.5)
    assert K == V == 128
    assert chunk_size == 64
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    if checkpoint_every_n_tokens < 0 or checkpoint_every_n_tokens % chunk_size != 0:
        raise ValueError(
            "checkpoint_every_n_tokens must be 0 or a positive multiple of "
            f"{chunk_size}, got {checkpoint_every_n_tokens}."
        )
    store_checkpoints = checkpoint_every_n_tokens > 0

    if cu_seqlens is None:
        real_batch_size = batch_size
        cu_seqlens = torch.empty((batch_size + 1), dtype=torch.int32, device=k.device)
        seqlen_dtype = torch.int32
        is_varlen = False
    else:
        real_batch_size = len(cu_seqlens) - 1
        seqlen_dtype = cu_seqlens.dtype
        is_varlen = True

    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty(
            (real_batch_size, H, K, V), dtype=torch.float32, device=k.device
        )
    state_checkpoints = None
    chunk_offsets = cu_seqlens
    if store_checkpoints:
        every_n = checkpoint_every_n_tokens
        if is_varlen:
            rows = max(num_tokens // every_n + real_batch_size, 1)
            chunk_offsets = _checkpoint_offsets(cu_seqlens, every_n)
        else:
            per_seq = (num_tokens + every_n - 1) // every_n
            rows = batch_size * per_seq
            chunk_offsets = (
                torch.arange(0, batch_size + 1, dtype=torch.int32, device=k.device)
                * per_seq
            )
        state_checkpoints = torch.empty((rows, H, V, K), dtype=q.dtype, device=k.device)
        hs = state_checkpoints
    else:
        key = (H, K, V, k.device)
        if key not in _HS_DUMMY:
            _HS_DUMMY[key] = torch.empty(
                (1, H, V, K), dtype=torch.float32, device=k.device
            )
        hs = _HS_DUMMY[key]

    if output is None:
        o = torch.empty_like(v)
    else:
        if output.shape != v.shape:
            raise ValueError(f"Expected output shape {v.shape}, got {output.shape}.")
        if output.dtype != v.dtype:
            raise ValueError(f"Expected output dtype {v.dtype}, got {output.dtype}.")
        if output.device != v.device:
            raise ValueError(f"Expected output device {v.device}, got {output.device}.")
        o = output

    final_state_shape = (real_batch_size, H, K, V)
    if output_state is None:
        final_state = torch.empty(
            final_state_shape, dtype=torch.float32, device=k.device
        )
    else:
        if output_state.shape != final_state_shape:
            raise ValueError(
                f"Expected output_state shape {final_state_shape}, got {output_state.shape}."
            )
        if output_state.dtype != torch.float32:
            raise ValueError(
                f"Expected output_state dtype torch.float32, got {output_state.dtype}."
            )
        if output_state.device != k.device:
            raise ValueError(
                f"Expected output_state device {k.device}, got {output_state.device}."
            )
        final_state = output_state

    if num_tokens == 0:
        # No token means no chunk: the state passes through unchanged. This is
        # decided from the tensor shape, without reading cu_seqlens on the host,
        # and avoids launching empty grids / zero-extent TMA descriptors.
        if output_final_state:
            if use_initial_state:
                final_state.copy_(initial_state)
            else:
                final_state.zero_()
        return (
            o if output_o else None,
            state_checkpoints,
            final_state if output_final_state else None,
        )

    prefill_jit = (
        tilelang_fused_chunk_gdn_prefill_hs
        if store_checkpoints
        else tilelang_fused_chunk_gdn_prefill
    )
    tilelang_fused_chunk_gdn_prefill_kernel = prefill_jit(
        H,
        Hg,
        K,
        V,
        chunk_size,
        scale,
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        b_dtype=b.dtype,
        h0_dtype=initial_state.dtype,
        ht_dtype=final_state.dtype,
        o_dtype=o.dtype,
        seqlen_dtype=seqlen_dtype,
        accum_dtype="float32",
        use_initial_state=use_initial_state,
        store_final_state=output_final_state,
        is_varlen=is_varlen,
        is_log_space=is_log_space,
        l2norm_eps=l2norm_eps,
        store_h=store_checkpoints,
        hs_dtype=q.dtype if store_checkpoints else torch.float32,
        ckpt_stride=checkpoint_every_n_tokens // chunk_size if store_checkpoints else 1,
    )
    if l2norm_eps is not None:
        rn = gdn_qk_rnorm(q, k, l2norm_eps)
    else:
        rn_key = (Hg, k.device)
        if rn_key not in _RN_DUMMY:
            _RN_DUMMY[rn_key] = torch.zeros(
                (1, 1, Hg, 2), dtype=torch.float32, device=k.device
            )
        rn = _RN_DUMMY[rn_key]
    tilelang_fused_chunk_gdn_prefill_kernel(
        q,
        k,
        v,
        g,
        b,
        initial_state,
        cu_seqlens,
        o,
        final_state,
        rn,
        hs,
        chunk_offsets,
    )

    if not output_final_state:
        final_state = None
    if not output_o:
        o = None

    return o, state_checkpoints, final_state


if __name__ == "__main__":
    tilelang_fused_chunk_gdn_prefill_kernel = tilelang_fused_chunk_gdn_prefill(
        1,
        1,
        128,
        128,
        64,
        1,
        qkva_dtype=torch.float16,
        g_dtype=torch.float32,
        b_dtype=torch.float32,
        h0_dtype=torch.float32,
        ht_dtype=torch.float32,
        o_dtype=torch.float16,
        seqlen_dtype=torch.int32,
        accum_dtype="float32",
        use_initial_state=True,
        store_final_state=True,
        is_varlen=True,
        is_log_space=True,
    )
