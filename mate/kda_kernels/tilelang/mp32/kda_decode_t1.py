import functools
import tilelang
import tilelang.language as T
import torch

from ....execution_context import skip_kernel_launch_if_dry_run

__all__ = ["run_gated_delta_rule_decode_vk_fp32"]

_LOG2E = 1.4426950408889634
_SOFTPLUS_BETA = 1.0
_SOFTPLUS_THRESHOLD = 20.0
# log(x) = ln2 * log2(x); log2 is a hardware op here (see the module notes
# in internal/performance/kda/README.md) while logf goes through the generic
# edge-case path.
_LN2 = 0.6931471805599453094
_KERNEL_THREADS = 128
_STATE_TME_INNER_CACHE_POLICY = "cache_normal"
_STATE_TME_OUTER_CACHE_POLICY = "cache_normal"

_AUTO_TUNED_BATCH_CONFIGS = (
    (2, (8, 8)),
    (4, (8, 4)),
    (16, (8, 2)),
)
_AUTO_TUNED_LARGE_BATCH_CONFIG = (4, 4)
# One 32-row state tile per CTA: measured 1.5x faster than (4, 4) on the
# S6000 (2214 vs 1467 GB/s of state traffic at B=512, D=128).
_AUTO_TUNED_WIDE_BATCH_CONFIG = (32, 4)
# V-first D=128 varlen uses one 16-row state tile per CTA. This keeps the
# eight-way state split while removing three state TME copies/barrier pairs.
_AUTO_TUNED_VARLEN_V_FIRST_LARGE_BATCH_CONFIG = (16, 8)
_AUTO_TUNED_VARLEN_FALLBACK_LARGE_BATCH_CONFIG = (4, 8)
# BF16 V-first uses v_tile=16 to merge each pair of 2 KiB state TME tiles.
# Keeping num_blocks_per_state unchanged preserves the state rows and shared
# memory footprint per CTA while halving the number of TME copies/barriers.
_AUTO_TUNED_BF16_STATE_V_FIRST_BATCH_CONFIGS = (
    (2, (16, 8)),
    (4, (16, 4)),
    (16, (16, 2)),
    (32, (16, 2)),
)
# One 64-row tile per CTA from batch 64 up: bf16 state halves the tile, so 64
# rows cost the same shared memory 32 rows cost in fp32, and the single tile
# removes seven barriered TMA waits per CTA (measured 7-10% faster from B=64).
_AUTO_TUNED_BF16_STATE_V_FIRST_LARGE_BATCH_CONFIG = (64, 2)
# BF16 V-first D=128 varlen uses one 32-row state tile per CTA once B > 16.
_AUTO_TUNED_BF16_STATE_V_FIRST_VARLEN_LARGE_BATCH_CONFIG = (32, 4)
# Preserve the previous V-first configs for dimensions other than the tuned
# D=128 shape, and for the separately tuned K-first path.
_AUTO_TUNED_BF16_STATE_FALLBACK_BATCH_CONFIGS = (
    (2, (8, 8)),
    (4, (8, 4)),
    (16, (8, 2)),
    (64, (8, 4)),
)
_AUTO_TUNED_BF16_STATE_V_FIRST_FALLBACK_LARGE_BATCH_CONFIG = (8, 2)
_AUTO_TUNED_BF16_STATE_K_FIRST_LARGE_BATCH_CONFIG = (8, 4)


def _exp2_f32(value):
    return T.exp2(value * _LOG2E)


def _make_kernel_config(v_tile: int, num_blocks_per_state: int) -> dict:
    return {
        "v_tile": v_tile,
        "num_blocks_per_state": num_blocks_per_state,
    }


def _resolve_autotuned_kernel_config(
    batch: int,
    state_dtype: str = "float32",
    state_v_first: bool = True,
    dim_v: int = 128,
    is_varlen: bool = False,
) -> dict:
    batch_configs: tuple[tuple[int, tuple[int, int]], ...]
    large_batch_config: tuple[int, int]
    if state_dtype == "bfloat16":
        if state_v_first and dim_v == 128:
            if is_varlen and batch > 16:
                return _make_kernel_config(
                    *_AUTO_TUNED_BF16_STATE_V_FIRST_VARLEN_LARGE_BATCH_CONFIG
                )
            batch_configs = _AUTO_TUNED_BF16_STATE_V_FIRST_BATCH_CONFIGS
            large_batch_config = _AUTO_TUNED_BF16_STATE_V_FIRST_LARGE_BATCH_CONFIG
        else:
            batch_configs = _AUTO_TUNED_BF16_STATE_FALLBACK_BATCH_CONFIGS
            large_batch_config = (
                _AUTO_TUNED_BF16_STATE_V_FIRST_FALLBACK_LARGE_BATCH_CONFIG
                if state_v_first
                else _AUTO_TUNED_BF16_STATE_K_FIRST_LARGE_BATCH_CONFIG
            )
    else:
        batch_configs = _AUTO_TUNED_BATCH_CONFIGS
        if is_varlen:
            large_batch_config = (
                _AUTO_TUNED_VARLEN_V_FIRST_LARGE_BATCH_CONFIG
                if state_v_first and dim_v == 128
                else _AUTO_TUNED_VARLEN_FALLBACK_LARGE_BATCH_CONFIG
            )
        else:
            # KDA_MP32_WIDE_STATE_TILE: one wide state tile per CTA removes the
            # per-tile barrier/sync overhead that dominates on the S6000.
            if (
                dim_v % 32 == 0
                and (dim_v // 32) % _AUTO_TUNED_WIDE_BATCH_CONFIG[1] == 0
            ):
                large_batch_config = _AUTO_TUNED_WIDE_BATCH_CONFIG
            else:
                large_batch_config = _AUTO_TUNED_LARGE_BATCH_CONFIG

    # 小 batch 档（batch <= 16）保持旧语义 v_tile = dim_v // nbps，也就是
    # n_tiles == 1，行为与改造前完全一致（这些小档的 num_blocks_per_state 是
    # 为小 batch 的占卡数调过的，不能被大 batch 的调度覆盖）。
    for max_batch, config in batch_configs:
        if batch <= max_batch:
            return _make_kernel_config(dim_v // config[1], config[1])

    # ---- 大 batch 档：T=1 多 tile 调度（本文件专用）----------------------
    # 每个 (batch, head) 只起 1 个 CTA，顺序处理 dim_v // 32 个 32 行 TME tile：
    # TME tile 尺寸、每 head 的 load/store 次数都不变，变的是每 head 的 CTA 数
    # 4 -> 1，于是 prologue（q/k l2norm + gate）只做一次。
    # 实测 B=512 / fp32 state / D=128：nbps=1/2/4 -> 1224/1230/1406 us，
    # 与旧的单 tile 版（nbps=4, 1396 us）位级一致；纯 TMA 地板 1200 us。
    if not is_varlen and state_v_first and state_dtype == "float32" and dim_v == 128:
        return _make_kernel_config(32, 1)

    # bf16 state / varlen / 其它维度：同样保持旧语义。
    return _make_kernel_config(dim_v // large_batch_config[1], large_batch_config[1])


def _build_kda_decode_kernel_factory(
    qk_head: int,
    head: int,
    dim_k: int,
    dim_v: int,
    input_dtype: str,
    gate_batch_dtype: str,
    dt_bias_dtype: str,
    output_dtype: str,
    b_dtype: str,
    use_qk_l2norm: bool,
    v_tile: int,
    num_blocks_per_state: int,
    use_initial_state: bool,
    is_varlen: bool,
    inplace_final_state: bool,
    is_beta_headwise: bool,
    is_continuous_batching: bool,
    is_spec_decoding: bool,
    store_final_state: bool,
    has_dt_bias: bool,
    use_gate_in_kernel: bool,
    use_lower_bound: bool,
    apply_beta_sigmoid: bool,
    allow_neg_eigval: bool,
    state_v_first: bool,
    state_dtype: str = "float32",
):
    if not hasattr(T, "tma_copy"):
        raise RuntimeError(
            "PH1S KDA decode requires TileLang MUSA 0.1.12 or a newer develop build."
        )
    if not state_v_first:
        raise ValueError(
            "MP32 KDA decode only supports state_v_first=True; k-first support "
            "was removed."
        )
    if qk_head <= 0:
        raise ValueError("qk_head must be positive.")
    if head % qk_head != 0:
        raise ValueError(
            f"state/value heads={head} must be divisible by q/k heads={qk_head}."
        )
    if dim_k % 32 != 0:
        raise ValueError(f"dim_k={dim_k} must be divisible by 32.")
    if dim_v != dim_k:
        raise ValueError(
            f"Current decode kernel expects dim_v == dim_k, got dim_v={dim_v}, dim_k={dim_k}."
        )
    if state_dtype not in ("float32", "bfloat16"):
        raise ValueError(f"state_dtype must be float32 or bfloat16, got {state_dtype}.")
    # ---- T=1 专用：一个 CTA 恰好持有一个 state row block ----
    # v_tile 由 num_blocks_per_state 定，num_v_tiles_per_block 恒为 1：不存在跨
    # tile 的流水，也不存在第二块 state smem。T=1 的 state 只读一次、写一次。
    if num_blocks_per_state <= 0:
        raise ValueError("num_blocks_per_state must be positive.")
    if dim_v % num_blocks_per_state != 0:
        raise ValueError(
            f"dim_v={dim_v} must be divisible by "
            f"num_blocks_per_state={num_blocks_per_state}."
        )
    # v_tile（TME tile）与 nbps（每 head 的 CTA 数）解耦：一个 CTA 处理
    # rows_per_cta = dim_v // nbps 行，切成 n_tiles 个 v_tile 大小的 TME tile，
    # 共用同一个 state 缓冲顺序处理（prologue 只做一次）。
    rows_per_cta = dim_v // num_blocks_per_state
    if rows_per_cta % v_tile != 0:
        raise ValueError(
            f"rows_per_cta={rows_per_cta} must be divisible by v_tile={v_tile}"
        )
    n_tiles = rows_per_cta // v_tile

    # 行段的分工（T=1 没有 token 循环；行按趟走）：
    #   一行 = lanes_per_row 个 lane，每 lane 持 k_per_lane 个 K
    #   一个 warp 一趟 rows_per_pass 行，n_passes 趟覆盖 rows_per_warp 行
    lanes_per_row = 16
    rows_per_pass = (_KERNEL_THREADS // 4) // lanes_per_row
    rows_per_warp = v_tile // 4
    if rows_per_warp % rows_per_pass != 0:
        raise ValueError(
            f"rows_per_warp={rows_per_warp} must be divisible by "
            f"rows_per_pass={rows_per_pass}"
        )
    n_passes = rows_per_warp // rows_per_pass
    k_per_lane = dim_k // lanes_per_row
    if k_per_lane % 4 != 0:
        raise ValueError(f"k_per_lane={k_per_lane} must be a multiple of 4")
    n_shfl = lanes_per_row.bit_length() - 1
    if dim_v % v_tile != 0:
        raise ValueError(f"dim_v={dim_v} must be divisible by v_tile={v_tile}")
    if v_tile % 4 != 0:
        raise ValueError(f"v_tile={v_tile} must be divisible by 4 for row mapping.")
    head_group_size = head // qk_head
    num_v_tiles = dim_v // v_tile
    if num_blocks_per_state <= 0:
        raise ValueError("num_blocks_per_state must be positive.")
    if num_blocks_per_state > num_v_tiles:
        raise ValueError(
            f"num_blocks_per_state={num_blocks_per_state} exceeds num_v_tiles={num_v_tiles}."
        )
    if num_v_tiles % num_blocks_per_state != 0:
        raise ValueError(
            f"num_v_tiles={num_v_tiles} must be divisible by "
            f"num_blocks_per_state={num_blocks_per_state}."
        )
    num_v_tiles_per_block = (
        num_v_tiles + num_blocks_per_state - 1
    ) // num_blocks_per_state
    if num_v_tiles_per_block * v_tile > _KERNEL_THREADS:
        raise ValueError(
            f"per-block output elements={num_v_tiles_per_block * v_tile} exceeds "
            f"kernel threads={_KERNEL_THREADS}."
        )
    vec_size = dim_k // 32
    # v/beta 的 load 在 T=1 版里内联进行段，不需要 vec_size_v /
    # num_v_rows_per_block / use_local_v_beta。

    batch = T.dynamic("batch")
    old_batch = T.dynamic("old_batch")
    seq_q = T.dynamic("seq_q")
    seq_kv = T.dynamic("seq_kv")
    seq_o = T.dynamic("seq_o")
    pool_size = T.dynamic("pool_size")

    q_stride_b = T.dynamic("q_stride_b")
    q_stride_t = T.dynamic("q_stride_t")
    q_stride_h = T.dynamic("q_stride_h")
    q_stride_k = T.dynamic("q_stride_k")

    k_stride_b = T.dynamic("k_stride_b")
    k_stride_t = T.dynamic("k_stride_t")
    k_stride_h = T.dynamic("k_stride_h")
    k_stride_k = T.dynamic("k_stride_k")

    v_stride_b = T.dynamic("v_stride_b")
    v_stride_t = T.dynamic("v_stride_t")
    v_stride_h = T.dynamic("v_stride_h")
    v_stride_v = T.dynamic("v_stride_v")

    a_stride_b = T.dynamic("a_stride_b")
    a_stride_t = T.dynamic("a_stride_t")
    a_stride_h = T.dynamic("a_stride_h")
    a_stride_k = T.dynamic("a_stride_k")

    b_stride_b = T.dynamic("b_stride_b")
    b_stride_t = T.dynamic("b_stride_t")
    b_stride_h = T.dynamic("b_stride_h")
    b_stride_v = T.dynamic("b_stride_v")

    o_stride_b = T.dynamic("o_stride_b")
    o_stride_t = T.dynamic("o_stride_t")
    o_stride_h = T.dynamic("o_stride_h")
    o_stride_v = T.dynamic("o_stride_v")

    state_indices_stride_b = T.dynamic("state_indices_stride_b")
    state_indices_stride_t = T.dynamic("state_indices_stride_t")
    token = T.dynamic("token")

    q_shape = (batch, seq_q, qk_head, dim_k)
    k_shape = (batch, seq_kv, qk_head, dim_k)
    v_shape = (batch, seq_kv, head, dim_v)
    g_shape = (batch, seq_kv, head, dim_k)
    # k-first removed: state layout is always (dim_v, dim_k).
    vk_shape = (dim_v, dim_k)
    state_final_shape = (
        (pool_size, head, *vk_shape)
        if inplace_final_state
        else (old_batch, head, *vk_shape)
    )
    state_final_shape = (
        (seq_q, head, *vk_shape)
        if (is_continuous_batching and not inplace_final_state)
        else state_final_shape
    )
    beta_shape = (
        (batch, seq_kv, head, dim_v) if is_beta_headwise else (batch, seq_kv, head)
    )
    o_shape = (batch, seq_o, head, dim_v)
    cu_seqlens_shape = (old_batch + 1,) if is_varlen else (1,)
    # Keep one cheap B-sized argument even when speculative decoding is off.
    # It anchors old_batch for fixed, non-continuous, in-place calls where the
    # state pool can be larger than the logical batch.
    num_accepted_tokens_shape = (old_batch,)
    state_indices_shape = (old_batch, token) if is_continuous_batching else (1, 1)

    q_strides = (q_stride_b, q_stride_t, q_stride_h, q_stride_k)
    k_strides = (k_stride_b, k_stride_t, k_stride_h, k_stride_k)
    v_strides = (v_stride_b, v_stride_t, v_stride_h, v_stride_v)
    a_strides = (a_stride_b, a_stride_t, a_stride_h, a_stride_k)
    b_strides = (
        (b_stride_b, b_stride_t, b_stride_h, b_stride_v)
        if is_beta_headwise
        else (b_stride_b, b_stride_t, b_stride_h)
    )
    o_strides = (o_stride_b, o_stride_t, o_stride_h, o_stride_v)
    state_indices_strides = (state_indices_stride_b, state_indices_stride_t)

    @tilelang.jit(
        target={"kind": "musa", "arch": "mp_32"},
        pass_configs={
            # tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
            tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
            tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
            tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
            tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
        },
        compile_flags=[
            "-Od3",
            "-fno-signed-zeros",
            "-fmusa-flush-denormals-to-zero",
            "-mllvm",
            "-misched=mtgpu-max-ilp",
            "-mllvm",
            "-mtgpu-if-convert=1",
            "-mllvm",
            "-mtgpu-tiny-offset-hint=1",
            "-mllvm",
            "-mtgpu-combine-fop-instr=1",
            "-mllvm",
            "-mtgpu-enable-postra-sched=0",
            "-mllvm",
            "-misched-recompute-slotindex=1",
        ],
    )
    def _decode_func():
        mbarrier_list = [_KERNEL_THREADS] * num_v_tiles_per_block

        @T.prim_func
        def gated_deltanet_kda_decode_t1(
            q: T.StridedTensor(q_shape, q_strides, input_dtype),
            k: T.StridedTensor(k_shape, k_strides, input_dtype),
            v: T.StridedTensor(v_shape, v_strides, input_dtype),
            A_log: T.Tensor([head], "float32"),
            a: T.StridedTensor(g_shape, a_strides, gate_batch_dtype),
            dt_bias: T.Tensor([head, dim_k], dt_bias_dtype),
            b: T.StridedTensor(beta_shape, b_strides, b_dtype),
            cu_seqlens: T.Tensor(cu_seqlens_shape, "int32"),
            num_accepted_tokens: T.Tensor(num_accepted_tokens_shape, "int32"),
            T_fixed_arg: T.int32,
            scale: T.float32,
            lower_bound: T.float32,
            state_indices: T.StridedTensor(
                state_indices_shape, state_indices_strides, "int32"
            ),
            state: T.Tensor([pool_size, head, *vk_shape], state_dtype),
            state_final: T.Tensor(state_final_shape, state_dtype),
            o: T.StridedTensor(o_shape, o_strides, output_dtype),
        ):
            with T.Kernel(
                old_batch * num_blocks_per_state, head, threads=_KERNEL_THREADS
            ) as (bx, hid):
                # T=1: 每个 CTA 一个 (batch, state row block)。wrapper 已经把
                # q/k/v/g/o reshape 成 [1, B, H, *]，所以 token 下标就是 bid。
                bid = bx // num_blocks_per_state
                cta_block = bx % num_blocks_per_state
                cta_row_base = cta_block * rows_per_cta

                # smem: state / q / k / g —— 没有第二块 state buffer，也没有 v。
                state_load_stage = T.alloc_shared([1, v_tile, dim_k], state_dtype)
                # tile 存 fp16：2 B/元素 → 三个 tile 从 1,536 B 降到 768 B，
                # smem 17,920 → 17,152 → 15 CTA/MP；tile 的 LDS 字节也减半。
                q_tile = T.alloc_shared([dim_k], "float16")
                k_tile = T.alloc_shared([dim_k], "float16")
                g_tile = T.alloc_shared([dim_k], "float16")

                q_reg = T.alloc_local([vec_size], "float32")
                k_reg = T.alloc_local([vec_size], "float32")
                g_reg = T.alloc_local([vec_size], "float32")
                # VLOAD: raw (unconverted) staging for the vectorized loads.
                q_h = T.alloc_local([vec_size], input_dtype)
                k_h = T.alloc_local([vec_size], input_dtype)
                g_in = T.alloc_local([vec_size], gate_batch_dtype)
                # 每 lane 持有 k_per_lane 个 K 的 h_dec（= k_per_lane/4 个 float4）
                h_reg = T.alloc_local([k_per_lane], "float32")
                # 把 q/k/g 的 tile 读从趟循环里提出来（每 lane k_per_lane 个值）
                q_l = T.alloc_local([k_per_lane], "float32")
                k_l = T.alloc_local([k_per_lane], "float32")
                g_l = T.alloc_local([k_per_lane], "float32")
                acc_hk = T.alloc_local([4], "float32")
                acc_hq = T.alloc_local([4], "float32")
                acc_kq = T.alloc_local([4], "float32")
                dt_bias_val = T.alloc_local([vec_size], "float32")

                sum_q = T.alloc_local([1], "float32")
                sum_k = T.alloc_local([1], "float32")
                sum_hk = T.alloc_local([1], "float32")
                sum_hq = T.alloc_local([1], "float32")
                sum_kq = T.alloc_local([1], "float32")
                final_state_slot = T.alloc_local([1], "int32")

                mbars = T.alloc_barrier(mbarrier_list)

                qk_hid = hid // head_group_size
                tid = T.get_thread_binding()
                lane = tid % 32
                warp = tid // 32
                k_start = lane * vec_size

                if is_continuous_batching:
                    state_slot = state_indices[bid, 0]
                else:
                    state_slot = bid

                if state_slot >= 0:
                    # prologue 就发 state 的 TMA：和下面 q/k/g 的 load+l2norm 重叠，
                    # 而且 T=1 只需要这一次。
                    if use_initial_state:
                        T.tma_copy(
                            state[state_slot, hid, cta_row_base, 0],
                            state_load_stage[0, :, :],
                            barrier=mbars[0],
                            inner_cache_policy=_STATE_TME_INNER_CACHE_POLICY,
                            outer_cache_policy=_STATE_TME_OUTER_CACHE_POLICY,
                        )
                        T.mbarrier_arrive(mbarrier=mbars[0])

                    A_decay_val = (
                        -_exp2_f32(T.cast(A_log[hid], "float32"))
                        if use_gate_in_kernel
                        else 0.0
                    )
                    for i in T.vectorized(vec_size):
                        dt_bias_val[i] = 0.0
                    if has_dt_bias:
                        for i in T.vectorized(vec_size):
                            dt_bias_val[i] = T.cast(
                                dt_bias[hid, k_start + i], "float32"
                            )

                    final_state_slot[0] = bid
                    if is_continuous_batching:
                        do_store_final = True
                        if inplace_final_state:
                            final_state_slot[0] = state_indices[bid, 0]
                    else:
                        do_store_final = store_final_state

                    # ---- warp 分工：warp0 q、warp1 k、warp3 g（v 不进 smem）----
                    if warp == 0:
                        for i in T.vectorized(vec_size):
                            q_h[i] = q[0, bid, qk_hid, k_start + i]
                        for i in T.vectorized(vec_size):
                            q_reg[i] = T.cast(q_h[i], "float32")
                        if use_qk_l2norm:
                            sum_q[0] = 0.0
                            for i in T.unroll(vec_size):
                                sum_q[0] += q_reg[i] * q_reg[i]
                            for offset in T.unroll(5):
                                sum_q[0] += T.shfl_xor(sum_q[0], 16 >> offset)
                            # RSQRT: raw intrinsic, no libm edge-case path.
                            # 顺手把 scale 一起乘进 inv_norm_q（原本要过两轮乘法）。
                            inv_norm_q = (
                                T.call_extern(
                                    "float32", "__musa_rsqrt_rn_f", sum_q[0] + 1e-6
                                )
                                * scale
                            )
                            for i in T.unroll(vec_size):
                                q_reg[i] = q_reg[i] * inv_norm_q
                        else:
                            for i in T.unroll(vec_size):
                                q_reg[i] = q_reg[i] * scale
                        # fp16 tile 用标量写：T.vectorized 会生成
                        # `half_t tmp[4]` + tl_h4 向量访问（对齐 2B），实测 launch 挂死
                        for i in T.unroll(vec_size):
                            q_tile[k_start + i] = T.cast(q_reg[i], "float16")
                    elif warp == 1:
                        for i in T.vectorized(vec_size):
                            k_h[i] = k[0, bid, qk_hid, k_start + i]
                        for i in T.vectorized(vec_size):
                            k_reg[i] = T.cast(k_h[i], "float32")
                        if use_qk_l2norm:
                            sum_k[0] = 0.0
                            for i in T.unroll(vec_size):
                                sum_k[0] += k_reg[i] * k_reg[i]
                            for offset in T.unroll(5):
                                sum_k[0] += T.shfl_xor(sum_k[0], 16 >> offset)
                            # RSQRT: raw intrinsic, no libm edge-case path.
                            inv_norm_k = T.call_extern(
                                "float32", "__musa_rsqrt_rn_f", sum_k[0] + 1e-6
                            )
                            for i in T.unroll(vec_size):
                                k_reg[i] = k_reg[i] * inv_norm_k
                        # fp16 tile 用标量写：同上
                        for i in T.unroll(vec_size):
                            k_tile[k_start + i] = T.cast(k_reg[i], "float16")
                    elif warp == 3:
                        for i in T.vectorized(vec_size):
                            g_in[i] = a[0, bid, hid, k_start + i]
                        for i in T.unroll(vec_size):
                            g_reg[i] = T.cast(g_in[i], "float32")
                        if use_gate_in_kernel:
                            if has_dt_bias:
                                for i in T.unroll(vec_size):
                                    g_reg[i] += dt_bias_val[i]
                            if use_lower_bound:
                                for i in T.unroll(vec_size):
                                    g_reg[i] = (
                                        lower_bound
                                        * 1.0
                                        / (1.0 + _exp2_f32(A_decay_val * g_reg[i]))
                                    )
                            else:
                                softplus_scale = _LN2 / _SOFTPLUS_BETA
                                for i in T.unroll(vec_size):
                                    g_tmp = _SOFTPLUS_BETA * g_reg[i]
                                    softplus_x = T.if_then_else(
                                        g_tmp <= _SOFTPLUS_THRESHOLD,
                                        softplus_scale * T.log2(1.0 + _exp2_f32(g_tmp)),
                                        g_reg[i],
                                    )
                                    g_reg[i] = A_decay_val * softplus_x
                            for i in T.unroll(vec_size):
                                g_reg[i] = _exp2_f32(g_reg[i])
                        # fp16 tile 用标量写：同上
                        for i in T.unroll(vec_size):
                            g_tile[k_start + i] = T.cast(g_reg[i], "float16")

                    # ---- state 到位 + q/k/g 交接 ----
                    if use_initial_state:
                        T.mbarrier_wait_parity(mbarrier=mbars[0], parity=0)
                    T.sync_threads()

                    # ---- 8 lane 一行；每 lane k_per_lane 个 K（k_per_lane/4 个 float4）----
                    # 一个 warp 一趟管 rows_per_pass 行，共 n_passes 趟。
                    # 一行的 8 个 lane 正好是连续的 8 个 lane，一个 LDS 相位就是
                    # 128 B 连续 → 满速，不需要任何错位/取模，也没有地址运算。
                    rw = lane // lanes_per_row  # 0..rows_per_pass-1
                    part = lane % lanes_per_row  # 0..lanes_per_row-1

                    for m in T.unroll(k_per_lane // 4):
                        piece = lanes_per_row * m + part
                        for i in T.unroll(4):
                            g_l[m * 4 + i] = T.cast(g_tile[4 * piece + i], "float32")
                    for m in T.unroll(k_per_lane // 4):
                        piece = lanes_per_row * m + part
                        for i in T.unroll(4):
                            k_l[m * 4 + i] = T.cast(k_tile[4 * piece + i], "float32")
                    for m in T.unroll(k_per_lane // 4):
                        piece = lanes_per_row * m + part
                        for i in T.unroll(4):
                            q_l[m * 4 + i] = T.cast(q_tile[4 * piece + i], "float32")

                    # sum_kq = Σ_k k·q 与行无关：只算一次（原来每趟重算 + 重归约）
                    for i in T.unroll(4):
                        acc_kq[i] = 0.0
                    for m in T.unroll(k_per_lane // 4):
                        piece = lanes_per_row * m + part
                        for i in T.unroll(4):
                            acc_kq[i] += k_l[m * 4 + i] * q_l[m * 4 + i]
                    sum_kq[0] = (acc_kq[0] + acc_kq[1]) + (acc_kq[2] + acc_kq[3])
                    for step in T.unroll(n_shfl):
                        sum_kq[0] += T.shfl_xor(sum_kq[0], 1 << step)

                    for tile in T.serial(n_tiles):
                        tile_base = cta_row_base + tile * v_tile
                        if tile > 0:
                            # 复用同一个 state 缓冲前，先等上一轮 store 排干
                            if do_store_final:
                                T.tma_store_wait()
                            T.sync_threads()
                            if use_initial_state:
                                T.tma_copy(
                                    state[state_slot, hid, tile_base, 0],
                                    state_load_stage[0, :, :],
                                    barrier=mbars[0],
                                    inner_cache_policy=_STATE_TME_INNER_CACHE_POLICY,
                                    outer_cache_policy=_STATE_TME_OUTER_CACHE_POLICY,
                                )
                                T.mbarrier_arrive(mbarrier=mbars[0])
                        if use_initial_state:
                            T.mbarrier_wait_parity(mbarrier=mbars[0], parity=tile % 2)
                        T.sync_threads()
                        for p in T.serial(n_passes):
                            row_idx = warp * rows_per_warp + p * rows_per_pass + rw
                            global_row = tile_base + row_idx
                            for i in T.unroll(4):
                                acc_hk[i] = 0.0
                                acc_hq[i] = 0.0
                            for m in T.unroll(k_per_lane // 4):
                                piece = lanes_per_row * m + part
                                if use_initial_state:
                                    for i in T.vectorized(4):
                                        h_reg[m * 4 + i] = T.cast(
                                            state_load_stage[0, row_idx, 4 * piece + i],
                                            "float32",
                                        )
                                else:
                                    for i in T.unroll(4):
                                        h_reg[m * 4 + i] = 0.0
                                for i in T.unroll(4):
                                    h_reg[m * 4 + i] = h_reg[m * 4 + i] * g_l[m * 4 + i]
                                    acc_hk[i] += h_reg[m * 4 + i] * k_l[m * 4 + i]
                                    acc_hq[i] += h_reg[m * 4 + i] * q_l[m * 4 + i]
                            # float2 相加 + float 相加（4 个累加器 → 2 级树）
                            sum_hk[0] = (acc_hk[0] + acc_hk[1]) + (
                                acc_hk[2] + acc_hk[3]
                            )
                            sum_hq[0] = (acc_hq[0] + acc_hq[1]) + (
                                acc_hq[2] + acc_hq[3]
                            )
                            # lanes_per_row 个 lane 合一行 -> n_shfl 步 shfl_xor
                            for step in T.unroll(n_shfl):
                                sum_hk[0] += T.shfl_xor(sum_hk[0], 1 << step)
                                sum_hq[0] += T.shfl_xor(sum_hq[0], 1 << step)

                            # v 和 beta 每行只要一个数：同组 lanes_per_row 个 lane 同地址
                            v_val = T.cast(v[0, bid, hid, global_row], "float32")
                            if is_beta_headwise:
                                beta_v = T.cast(b[0, bid, hid, global_row], "float32")
                                if apply_beta_sigmoid:
                                    beta_v = 1.0 / (1.0 + _exp2_f32(-beta_v))
                                    if allow_neg_eigval:
                                        beta_v *= 2
                            else:
                                beta_v = T.cast(b[0, bid, hid], "float32")
                                if apply_beta_sigmoid:
                                    beta_v = 1.0 / (1.0 + _exp2_f32(-beta_v))
                                    if allow_neg_eigval:
                                        beta_v *= 2
                            v_new = (v_val - sum_hk[0]) * beta_v

                            # REASSOC: o = Σ_k h_dec·q + v_new · Σ_k k·q
                            for m in T.unroll(k_per_lane // 4):
                                piece = lanes_per_row * m + part
                                for i in T.vectorized(4):
                                    state_load_stage[0, row_idx, 4 * piece + i] = (
                                        T.cast(
                                            h_reg[m * 4 + i] + k_l[m * 4 + i] * v_new,
                                            state_dtype,
                                        )
                                    )
                            if part == 0:
                                o[0, bid, hid, global_row] = T.cast(
                                    sum_hq[0] + v_new * sum_kq[0], output_dtype
                                )

                        # ---- 就地 store：这一份 state 只写这一次 ----
                        T.sync_threads()
                        if do_store_final:
                            T.tma_copy(
                                state_load_stage[0, :, :],
                                state_final[
                                    final_state_slot[0],
                                    hid,
                                    tile_base : tile_base + v_tile,
                                    :,
                                ],
                                inner_cache_policy=_STATE_TME_INNER_CACHE_POLICY,
                                outer_cache_policy=_STATE_TME_OUTER_CACHE_POLICY,
                            )
                            T.tma_store_wait()
                    if do_store_final:
                        T.tma_store_wait()
                else:
                    for tile in T.serial(n_tiles):
                        if tid < v_tile:
                            o[0, bid, hid, cta_row_base + tile * v_tile + tid] = T.cast(
                                0.0, output_dtype
                            )

        return gated_deltanet_kda_decode_t1

    return _decode_func


@functools.lru_cache(maxsize=32)
def _get_kda_decode_kernel(
    qk_head: int,
    head: int,
    dim_k: int,
    dim_v: int,
    input_dtype: str,
    gate_batch_dtype: str,
    dt_bias_dtype: str,
    output_dtype: str,
    b_dtype: str,
    use_qk_l2norm: bool,
    v_tile: int,
    num_blocks_per_state: int,
    use_initial_state: bool,
    is_varlen: bool,
    inplace_final_state: bool,
    is_beta_headwise: bool,
    is_continuous_batching: bool,
    is_spec_decoding: bool,
    store_final_state: bool,
    has_dt_bias: bool,
    use_gate_in_kernel: bool,
    use_lower_bound: bool,
    apply_beta_sigmoid: bool,
    allow_neg_eigval: bool,
    state_v_first: bool,
    state_dtype: str = "float32",
):
    return _build_kda_decode_kernel_factory(
        qk_head=qk_head,
        head=head,
        dim_k=dim_k,
        dim_v=dim_v,
        input_dtype=input_dtype,
        state_dtype=state_dtype,
        gate_batch_dtype=gate_batch_dtype,
        dt_bias_dtype=dt_bias_dtype,
        output_dtype=output_dtype,
        b_dtype=b_dtype,
        use_qk_l2norm=use_qk_l2norm,
        v_tile=v_tile,
        num_blocks_per_state=num_blocks_per_state,
        use_initial_state=use_initial_state,
        is_varlen=is_varlen,
        inplace_final_state=inplace_final_state,
        is_beta_headwise=is_beta_headwise,
        is_continuous_batching=is_continuous_batching,
        is_spec_decoding=is_spec_decoding,
        store_final_state=store_final_state,
        has_dt_bias=has_dt_bias,
        use_gate_in_kernel=use_gate_in_kernel,
        use_lower_bound=use_lower_bound,
        apply_beta_sigmoid=apply_beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        state_v_first=state_v_first,
    )()


def run_gated_delta_rule_decode_vk_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor | None,
    A_log: torch.Tensor,
    g: torch.Tensor,
    dt_bias: torch.Tensor | None,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    output: torch.Tensor,
    *,
    scale: float,
    lower_bound: float,
    use_qk_l2norm: bool,
    is_varlen: bool,
    inplace_final_state: bool,
    is_beta_headwise: bool,
    is_continuous_batching: bool,
    is_spec_decoding: bool,
    store_final_state: bool,
    has_dt_bias: bool,
    use_gate_in_kernel: bool,
    use_lower_bound: bool,
    apply_beta_sigmoid: bool,
    allow_neg_eigval: bool,
    state_v_first: bool,
    use_initial_state: bool,
):
    if not state_v_first:
        raise ValueError(
            "MP32 KDA decode only supports state_v_first=True; k-first support "
            "was removed."
        )

    if is_varlen:
        raise ValueError(
            "kda_decode_t1 is the T=1 kernel and does not handle varlen; "
            "varlen goes to mp32/kda_decode.py (the general kernel)."
        )
    if is_spec_decoding:
        raise ValueError(
            "kda_decode_t1 does not handle speculative decoding; "
            "use mp32/kda_decode.py (the general kernel)."
        )
    if q.ndim != 4 or q.shape[1] != 1:
        raise ValueError(
            "kda_decode_t1 requires T==1 with q shaped [B, 1, H_qk, K], "
            f"got {tuple(q.shape)}."
        )

    if is_continuous_batching and state_indices is not None:
        if state_indices.ndim == 1:
            state_indices = state_indices.unsqueeze(1)
        elif state_indices.ndim != 2:
            raise ValueError(
                "state_indices must have shape [B] or [B, T], "
                f"got {tuple(state_indices.shape)}."
            )

    if is_varlen:
        if cu_seqlens is None:
            raise ValueError("cu_seqlens must not be None when is_varlen=True")

        B = cu_seqlens.numel() - 1
        _, total_tokens, Hq, K = q.shape
        _, _, HV, V = v.shape

        q_arg = q
        k_arg = k
        v_arg = v
        g_arg = g
        b_arg = b

        output_arg = output
        cu_seqlens_arg = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
        T_fixed_arg = 0
    else:
        B, T_fixed_host, Hq, K = q.shape
        _, _, HV, V = v.shape
        total_tokens = B * T_fixed_host

        q_arg = q.reshape(1, total_tokens, Hq, K)
        k_arg = k.reshape(1, total_tokens, Hq, K)
        v_arg = v.reshape(1, total_tokens, HV, V)
        g_arg = g.reshape(1, total_tokens, HV, K)
        if is_beta_headwise:
            b_arg = b.reshape(1, total_tokens, HV, V)
        else:
            b_arg = b.reshape(1, total_tokens, HV)
        output_arg = output.reshape(1, total_tokens, HV, V)

        cu_seqlens_arg = torch.empty((1,), dtype=torch.int32, device=q.device)
        T_fixed_arg = int(T_fixed_host)

    input_dtype = str(q_arg.dtype).split(".")[-1]
    state_dtype = str(state.dtype).split(".")[-1]
    gate_batch_dtype = str(g_arg.dtype).split(".")[-1]

    output_dtype = str(output_arg.dtype).split(".")[-1]
    b_dtype = str(b_arg.dtype).split(".")[-1]
    if state_dtype not in ("float32", "bfloat16"):
        raise ValueError(
            f"state must have dtype torch.float32 or torch.bfloat16, got {state.dtype}."
        )
    kernel_config = _resolve_autotuned_kernel_config(
        B, state_dtype, state_v_first, V, is_varlen
    )

    needs_dummy_dt_bias = not has_dt_bias
    needs_dummy_num_accepted_tokens = not is_spec_decoding
    needs_dummy_state_indices = not is_continuous_batching

    if needs_dummy_dt_bias:
        dt_bias_arg = torch.zeros((HV, K), dtype=torch.float32, device=q.device)
    else:
        if dt_bias is None:
            raise ValueError("dt_bias must not be None when has_dt_bias=True")
        if dt_bias.ndim == 2:
            dt_bias_arg = dt_bias.to(device=q.device, dtype=dt_bias.dtype).contiguous()
        else:
            raise ValueError(f"dt_bias must be 2D, got ndim={dt_bias.ndim}")

    dt_bias_dtype = str(dt_bias_arg.dtype).split(".")[-1]

    if needs_dummy_state_indices:
        state_indices_arg = torch.empty((1, 1), dtype=torch.int32, device=q.device)
    else:
        if state_indices is None:
            raise ValueError(
                "state_indices must not be None when is_continuous_batching=True"
            )
        state_indices_arg = state_indices.to(
            device=q.device,
            dtype=torch.int32,
        ).contiguous()

    if needs_dummy_num_accepted_tokens:
        num_accepted_tokens_arg = torch.empty((B,), dtype=torch.int32, device=q.device)
    else:
        if num_accepted_tokens is None:
            raise ValueError(
                "num_accepted_tokens must not be None when is_spec_decoding=True"
            )
        num_accepted_tokens_arg = num_accepted_tokens.to(
            device=q.device,
            dtype=torch.int32,
        ).contiguous()

    A_log_arg = (
        A_log
        if A_log is not None
        else torch.zeros((HV,), dtype=torch.float32, device=q.device)
    )

    final_state_slots = None
    if is_continuous_batching:
        final_state_slots = total_tokens
    else:
        final_state_slots = B

    if inplace_final_state:
        assert state is not None
        final_state_arg = state
        final_state_result = state
    else:
        if state_v_first:
            final_state_arg = q.new_empty(
                final_state_slots, HV, V, K, dtype=state.dtype
            )
        else:
            final_state_arg = q.new_empty(
                final_state_slots, HV, K, V, dtype=state.dtype
            )
        final_state_result = final_state_arg
    if is_continuous_batching is not True and store_final_state is not True:
        final_state_result = None

    kernel_fn = _get_kda_decode_kernel(
        qk_head=Hq,
        head=HV,
        dim_k=K,
        dim_v=V,
        input_dtype=input_dtype,
        state_dtype=state_dtype,
        gate_batch_dtype=gate_batch_dtype,
        dt_bias_dtype=dt_bias_dtype,
        output_dtype=output_dtype,
        b_dtype=b_dtype,
        use_qk_l2norm=bool(use_qk_l2norm),
        v_tile=kernel_config["v_tile"],
        num_blocks_per_state=kernel_config["num_blocks_per_state"],
        use_initial_state=bool(use_initial_state),
        is_varlen=bool(is_varlen),
        inplace_final_state=bool(inplace_final_state),
        is_beta_headwise=bool(is_beta_headwise),
        is_continuous_batching=bool(is_continuous_batching),
        is_spec_decoding=bool(is_spec_decoding),
        store_final_state=bool(store_final_state),
        has_dt_bias=bool(has_dt_bias),
        use_gate_in_kernel=bool(use_gate_in_kernel),
        use_lower_bound=bool(use_lower_bound),
        apply_beta_sigmoid=bool(apply_beta_sigmoid),
        allow_neg_eigval=bool(allow_neg_eigval),
        state_v_first=bool(state_v_first),
    )

    if not skip_kernel_launch_if_dry_run():
        kernel_fn(
            q_arg,
            k_arg,
            v_arg,
            A_log_arg,
            g_arg,
            dt_bias_arg,
            b_arg,
            cu_seqlens_arg,
            num_accepted_tokens_arg,
            T_fixed_arg,
            float(scale),
            float(lower_bound),
            state_indices_arg,
            state,
            final_state_arg,
            output_arg,
        )
    if not is_varlen:
        output_arg = output.reshape(B, T_fixed_host, HV, V)
    return output_arg, final_state_result
