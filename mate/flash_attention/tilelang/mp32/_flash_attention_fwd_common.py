# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the PH1S MP32 FlashAttention forward kernels."""

import torch
import tilelang.language as T

from ....utils import cosize

_MP32_FORWARD_SOURCE = r"""
#include <tl_templates/musa/common.h>
#include <tl_templates/musa/fop.h>

struct FlashAttnReduceMax {
  TL_DEVICE tl_f4 operator()(tl_f4 lhs, tl_f4 rhs) const {
    return tl::vec_max_f4(lhs, rhs);
  }
  TL_DEVICE tl_f2 operator()(tl_f2 lhs, tl_f2 rhs) const {
    return tl::vec_max_f2(lhs, rhs);
  }
  TL_DEVICE float operator()(float lhs, float rhs) const {
    return max(lhs, rhs);
  }
};

struct FlashAttnReduceSum {
  TL_DEVICE tl_f4 operator()(tl_f4 lhs, tl_f4 rhs) const {
    float4 lhs_f4 = {lhs[0], lhs[1], lhs[2], lhs[3]};
    float4 rhs_f4 = {rhs[0], rhs[1], rhs[2], rhs[3]};
    float4 out_f4;
    mute::add(out_f4, lhs_f4, rhs_f4);
    return tl_f4{out_f4.x, out_f4.y, out_f4.z, out_f4.w};
  }
  TL_DEVICE tl_f2 operator()(tl_f2 lhs, tl_f2 rhs) const {
    return tl::vec_sum_f2(lhs, rhs);
  }
  TL_DEVICE float operator()(float lhs, float rhs) const {
    return lhs + rhs;
  }
};

template <typename ReduceOp>
TL_DEVICE void flashattn_reduce_2x32(const float* src, float* dst,
                                    ReduceOp op) {
  const tl_f4* values = reinterpret_cast<const tl_f4*>(src);

  tl_f4 row0_l0 = op(values[0], values[1]);
  tl_f4 row1_l0 = op(values[8], values[9]);
  tl_f4 row0_l1 = op(values[2], values[3]);
  tl_f4 row1_l1 = op(values[10], values[11]);
  tl_f4 row0_l2 = op(values[4], values[5]);
  tl_f4 row1_l2 = op(values[12], values[13]);
  tl_f4 row0_l3 = op(values[6], values[7]);
  tl_f4 row1_l3 = op(values[14], values[15]);

  tl_f4 row0_m0 = op(row0_l0, row0_l1);
  tl_f4 row1_m0 = op(row1_l0, row1_l1);
  tl_f4 row0_m1 = op(row0_l2, row0_l3);
  tl_f4 row1_m1 = op(row1_l2, row1_l3);
  tl_f4 row0_root = op(row0_m0, row0_m1);
  tl_f4 row1_root = op(row1_m0, row1_m1);

  tl_f2 row0_pair = op(make_tl_f2(row0_root[0], row0_root[1]),
                       make_tl_f2(row0_root[2], row0_root[3]));
  tl_f2 row1_pair = op(make_tl_f2(row1_root[0], row1_root[1]),
                       make_tl_f2(row1_root[2], row1_root[3]));
  tl_f2 rows = make_tl_f2(op(row0_pair[0], row0_pair[1]),
                          op(row1_pair[0], row1_pair[1]));

  rows = op(rows, tl::shfl_xor_sync(0xffffffffu, rows, 2));
  rows = op(rows, tl::shfl_xor_sync(0xffffffffu, rows, 1));
  *reinterpret_cast<tl_f2*>(dst) = rows;
}

TL_DEVICE void flashattn_reduce_max_2x32(const float* src, float* dst) {
  flashattn_reduce_2x32(src, dst, FlashAttnReduceMax{});
}

TL_DEVICE void flashattn_reduce_sum_2x32(const float* src, float* dst) {
  flashattn_reduce_2x32(src, dst, FlashAttnReduceSum{});
}

template <typename ReduceOp>
TL_DEVICE void flashattn_reduce_2x16(const float* src, float* dst,
                                    ReduceOp op) {
  const tl_f4* values = reinterpret_cast<const tl_f4*>(src);

  tl_f4 row0_l0 = op(values[0], values[1]);
  tl_f4 row1_l0 = op(values[4], values[5]);
  tl_f4 row0_l1 = op(values[2], values[3]);
  tl_f4 row1_l1 = op(values[6], values[7]);
  tl_f4 row0_root = op(row0_l0, row0_l1);
  tl_f4 row1_root = op(row1_l0, row1_l1);

  tl_f2 row0_pair = op(make_tl_f2(row0_root[0], row0_root[1]),
                       make_tl_f2(row0_root[2], row0_root[3]));
  tl_f2 row1_pair = op(make_tl_f2(row1_root[0], row1_root[1]),
                       make_tl_f2(row1_root[2], row1_root[3]));
  tl_f2 rows = make_tl_f2(op(row0_pair[0], row0_pair[1]),
                          op(row1_pair[0], row1_pair[1]));

  rows = op(rows, tl::shfl_xor_sync(0xffffffffu, rows, 2));
  rows = op(rows, tl::shfl_xor_sync(0xffffffffu, rows, 1));
  *reinterpret_cast<tl_f2*>(dst) = rows;
}

TL_DEVICE void flashattn_reduce_max_2x16(const float* src, float* dst) {
  flashattn_reduce_2x16(src, dst, FlashAttnReduceMax{});
}

TL_DEVICE void flashattn_reduce_sum_2x16(const float* src, float* dst) {
  flashattn_reduce_2x16(src, dst, FlashAttnReduceSum{});
}

TL_DEVICE tl_f4 flashattn_fma4(float scale, float scale_log2e, tl_f4 value,
                               float bias) {
  return tl_f4{fmaf(value[0], scale, bias) * scale_log2e,
               fmaf(value[1], scale, bias) * scale_log2e,
               fmaf(value[2], scale, bias) * scale_log2e,
               fmaf(value[3], scale, bias) * scale_log2e};
}

TL_DEVICE void flashattn_softmax_update_2x32(
    float* __restrict__ scores, float* __restrict__ scores_max,
    const float* __restrict__ scores_max_prev,
    float* __restrict__ scores_scale, float* __restrict__ acc_o,
    float scale, float scale_log2e) {
  tl_f4* score4 = reinterpret_cast<tl_f4*>(scores);
  tl_f4* acc_o4 = reinterpret_cast<tl_f4*>(acc_o);

  const tl_f2 scale2 = make_tl_f2(scale, scale);
  const tl_f2 previous = *reinterpret_cast<const tl_f2*>(scores_max_prev);
  const tl_f2 reduced = *reinterpret_cast<const tl_f2*>(scores_max);
  const tl_f2 current =
      tl::vec_max_f2(tl::mul2(reduced, scale2), previous);
  const tl_f2 alpha = tl::vec_exp2_f2(
      tl::mul2(tl::sub2(previous, current), make_tl_f2(scale_log2e, scale_log2e)));
  *reinterpret_cast<tl_f2*>(scores_max) = current;
  *reinterpret_cast<tl_f2*>(scores_scale) = alpha;

  score4[0] = flashattn_fma4(scale, scale_log2e, score4[0], -current[0]);
  score4[1] = flashattn_fma4(scale, scale_log2e, score4[1], -current[0]);
  score4[2] = flashattn_fma4(scale, scale_log2e, score4[2], -current[0]);
  score4[3] = flashattn_fma4(scale, scale_log2e, score4[3], -current[0]);
  score4[4] = flashattn_fma4(scale, scale_log2e, score4[4], -current[0]);
  score4[5] = flashattn_fma4(scale, scale_log2e, score4[5], -current[0]);
  score4[6] = flashattn_fma4(scale, scale_log2e, score4[6], -current[0]);
  score4[7] = flashattn_fma4(scale, scale_log2e, score4[7], -current[0]);
  score4[8] = flashattn_fma4(scale, scale_log2e, score4[8], -current[1]);
  score4[9] = flashattn_fma4(scale, scale_log2e, score4[9], -current[1]);
  score4[10] = flashattn_fma4(scale, scale_log2e, score4[10], -current[1]);
  score4[11] = flashattn_fma4(scale, scale_log2e, score4[11], -current[1]);
  score4[12] = flashattn_fma4(scale, scale_log2e, score4[12], -current[1]);
  score4[13] = flashattn_fma4(scale, scale_log2e, score4[13], -current[1]);
  score4[14] = flashattn_fma4(scale, scale_log2e, score4[14], -current[1]);
  score4[15] = flashattn_fma4(scale, scale_log2e, score4[15], -current[1]);
  const tl_f4 alpha0 = tl_f4{alpha[0], alpha[0], alpha[0], alpha[0]};
  const tl_f4 alpha1 = tl_f4{alpha[1], alpha[1], alpha[1], alpha[1]};
#define FLASHATTN_EXP_SCALE_2(S0, S1, ALPHA)                              \
  score4[S0] = tl::vec_exp2_f4(score4[S0]);                              \
  score4[S1] = tl::vec_exp2_f4(score4[S1]);                              \
  acc_o4[S0] = tl::mul4(acc_o4[S0], ALPHA);                              \
  acc_o4[S1] = tl::mul4(acc_o4[S1], ALPHA);                              \
  asm volatile("" ::: "memory")
  FLASHATTN_EXP_SCALE_2(0, 1, alpha0);
  FLASHATTN_EXP_SCALE_2(2, 3, alpha0);
  FLASHATTN_EXP_SCALE_2(4, 5, alpha0);
  FLASHATTN_EXP_SCALE_2(6, 7, alpha0);
  FLASHATTN_EXP_SCALE_2(8, 9, alpha1);
  FLASHATTN_EXP_SCALE_2(10, 11, alpha1);
  FLASHATTN_EXP_SCALE_2(12, 13, alpha1);
  FLASHATTN_EXP_SCALE_2(14, 15, alpha1);
#undef FLASHATTN_EXP_SCALE_2
}

TL_DEVICE void flashattn_softmax_update_2x16_2x64(
    float* __restrict__ scores, float* __restrict__ scores_max,
    const float* __restrict__ scores_max_prev,
    float* __restrict__ scores_scale, float* __restrict__ acc_o,
    float scale, float scale_log2e) {
  tl_f4* score4 = reinterpret_cast<tl_f4*>(scores);
  tl_f4* acc_o4 = reinterpret_cast<tl_f4*>(acc_o);

  const tl_f2 previous = *reinterpret_cast<const tl_f2*>(scores_max_prev);
  const tl_f2 reduced = *reinterpret_cast<const tl_f2*>(scores_max);
  const tl_f2 current = tl::vec_max_f2(
      tl::mul2(reduced, make_tl_f2(scale, scale)), previous);
  const tl_f2 alpha = tl::vec_exp2_f2(
      tl::mul2(tl::sub2(previous, current),
               make_tl_f2(scale_log2e, scale_log2e)));
  *reinterpret_cast<tl_f2*>(scores_max) = current;
  *reinterpret_cast<tl_f2*>(scores_scale) = alpha;

  score4[0] = flashattn_fma4(scale, scale_log2e, score4[0], -current[0]);
  score4[1] = flashattn_fma4(scale, scale_log2e, score4[1], -current[0]);
  score4[2] = flashattn_fma4(scale, scale_log2e, score4[2], -current[0]);
  score4[3] = flashattn_fma4(scale, scale_log2e, score4[3], -current[0]);
  score4[4] = flashattn_fma4(scale, scale_log2e, score4[4], -current[1]);
  score4[5] = flashattn_fma4(scale, scale_log2e, score4[5], -current[1]);
  score4[6] = flashattn_fma4(scale, scale_log2e, score4[6], -current[1]);
  score4[7] = flashattn_fma4(scale, scale_log2e, score4[7], -current[1]);

  const tl_f4 alpha0 = tl_f4{alpha[0], alpha[0], alpha[0], alpha[0]};
  const tl_f4 alpha1 = tl_f4{alpha[1], alpha[1], alpha[1], alpha[1]};
#define FLASHATTN_EXP_SCALE_4(S, O, ALPHA)                               \
  score4[S] = tl::vec_exp2_f4(score4[S]);                                \
  acc_o4[O] = tl::mul4(acc_o4[O], ALPHA);                                \
  acc_o4[(O) + 1] = tl::mul4(acc_o4[(O) + 1], ALPHA);                    \
  acc_o4[(O) + 2] = tl::mul4(acc_o4[(O) + 2], ALPHA);                    \
  acc_o4[(O) + 3] = tl::mul4(acc_o4[(O) + 3], ALPHA);                    \
  (void)0
  // Each N128 SQMMA tile holds row0 then row1: eight float4s per row.
  // Keep these offsets aligned with output_forward's D256 fragment layout.
  FLASHATTN_EXP_SCALE_4(0, 0, alpha0);
  FLASHATTN_EXP_SCALE_4(1, 4, alpha0);
  FLASHATTN_EXP_SCALE_4(2, 16, alpha0);
  FLASHATTN_EXP_SCALE_4(3, 20, alpha0);
  FLASHATTN_EXP_SCALE_4(4, 8, alpha1);
  FLASHATTN_EXP_SCALE_4(5, 12, alpha1);
  FLASHATTN_EXP_SCALE_4(6, 24, alpha1);
  FLASHATTN_EXP_SCALE_4(7, 28, alpha1);
#undef FLASHATTN_EXP_SCALE_4
}

"""


def _perm_g2s(n):
    return n // 8 * 8 + n % 4 * 2 + (n % 8) // 4


def _perm_r2r(n):
    return n // 8 * 8 + n % 2 * 4 + (n % 8) // 2


def _tilelang_dtype_nbytes(dtype):
    if dtype in (T.float16, T.bfloat16):
        return 2
    if dtype == T.float32:
        return 4
    raise AssertionError(f"unsupported TileLang dtype size query: {dtype}")


def _to_tilelang_dtype(torch_dtype):
    if torch_dtype is torch.float16:
        return T.float16
    if torch_dtype is torch.bfloat16:
        return T.bfloat16
    raise AssertionError(
        f"unsupported TileLang mp32 FlashAttention dtype: {torch_dtype}"
    )


def _current_musa_cc(tensor):
    assert hasattr(torch, "musa")
    device = tensor.device.index
    if device is None:
        device = torch.musa.current_device()
    major, minor = torch.musa.get_device_capability(device)
    return int(major) * 10 + int(minor)


def _check_attention_strides(name, tensor, multiple=8):
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must have a contiguous last dimension")
    for dim, stride in enumerate(tensor.stride()[:-1]):
        if stride <= 0:
            raise ValueError(f"{name} stride({dim}) must be positive, got {stride}")
        if stride % multiple != 0:
            raise ValueError(
                f"{name} stride({dim}) must be divisible by {multiple}, got {stride}"
            )


def _alloc_phase_registers():
    """Allocate phase buffers; initialize them in the kernel DSL body."""
    producer_phase = T.alloc_local([1], T.int32)
    consumer_phase = T.alloc_local([1], T.int32)
    return producer_phase, consumer_phase


def _annotate_consumer_layout(
    acc_s,
    scores_max,
    scores_max_prev,
    scores_scale,
    scores_sum,
    acc_o,
    p_cvt,
    block_n,
    dim_vo,
):
    """Apply the shared two-row consumer layout for BN64 and BN128."""
    score_cols_per_lane = block_n // 4
    output_cols_per_lane = dim_vo // 4

    def score_forward(i, j):
        return (
            i // 64 * 128 + i % 32 * 4 + j % 4,
            i % 64 // 32 * score_cols_per_lane + j // 4,
        )

    def output_forward(i, j):
        if dim_vo == 256:
            # SQMMA stores two N128 tiles, each containing both per-thread rows.
            # The fused D256 softmax must scale the same physical row segments.
            return (
                i // 64 * 128 + i % 32 * 4 + j % 4,
                j // 128 * 64 + i % 64 // 32 * 32 + j % 128 // 4,
            )
        return (
            i // 64 * 128 + i % 32 * 4 + j % 4,
            i % 64 // 32 * output_cols_per_lane + j // 4,
        )

    def row_forward(i, rep):
        return i // 64 * 128 + i % 32 * 4 + rep, i % 64 // 32

    def probability_forward(i, j):
        return (
            i // 64 * 128 + i % 32 * 4 + j % 8 // 2,
            i % 64 // 32 * score_cols_per_lane + j // 8 * 2 + j % 2,
        )

    layout = {
        acc_s: T.Fragment(acc_s.shape, forward_fn=score_forward),
        acc_o: T.Fragment(acc_o.shape, forward_fn=output_forward),
        p_cvt: T.Fragment(p_cvt.shape, forward_fn=probability_forward),
    }
    layout.update(
        {
            buffer: T.Fragment(buffer.shape, forward_fn=row_forward, replicate=4)
            for buffer in (scores_max, scores_max_prev, scores_scale, scores_sum)
        }
    )
    T.annotate_layout(layout)


def _assume_attention_strides(is_varlen, inner_strides, batch_strides):
    """Attach the alignment assumptions shared by all forward paths."""
    for stride in inner_strides:
        T.assume(stride % 8 == 0)
    if not is_varlen:
        for stride in batch_strides:
            T.assume(stride % 8 == 0)


def _make_attention_accessors(is_varlen):
    """Select dense or varlen indexing once while building the kernel."""
    if is_varlen:

        def sequence_start(cu_seqlens, batch_idx, _seqlen):
            return cu_seqlens[batch_idx]

        def sequence_length(cu_seqlens, batch_idx, _seqlen):
            return cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]

        def input_tile(
            tensor,
            _batch_idx,
            begin_seq,
            row_start,
            row_end,
            head,
            dim_start,
            dim_end,
        ):
            return tensor[
                begin_seq + row_start : begin_seq + row_end,
                head,
                dim_start:dim_end,
            ]

        def input_elem(tensor, _batch_idx, begin_seq, row, head, dim_idx):
            return tensor[begin_seq + row, head, dim_idx]

        def input_base(tensor):
            return tensor[0, 0, 0]

        def output_index(_batch_idx, begin_seq, row, head, dim_idx):
            return begin_seq + row, head, dim_idx

        def lse_index(_batch_idx, begin_seq, row, head):
            return head, begin_seq + row

    else:

        def sequence_start(_cu_seqlens, batch_idx, seqlen):
            return batch_idx * seqlen

        def sequence_length(_cu_seqlens, _batch_idx, seqlen):
            return seqlen

        def input_tile(
            tensor,
            batch_idx,
            _begin_seq,
            row_start,
            row_end,
            head,
            dim_start,
            dim_end,
        ):
            return tensor[
                batch_idx,
                row_start:row_end,
                head,
                dim_start:dim_end,
            ]

        def input_elem(tensor, batch_idx, _begin_seq, row, head, dim_idx):
            return tensor[batch_idx, row, head, dim_idx]

        def input_base(tensor):
            return tensor[0, 0, 0, 0]

        def output_index(batch_idx, _begin_seq, row, head, dim_idx):
            return batch_idx, row, head, dim_idx

        def lse_index(batch_idx, _begin_seq, row, head):
            return batch_idx, head, row

    def k_tma_tile(
        tensor, batch_idx, begin_seq, row_start, row_end, head, dim_start, dim_end
    ):
        if is_varlen:
            return tensor[
                head,
                begin_seq + row_start : begin_seq + row_end,
                dim_start:dim_end,
            ]
        return tensor[batch_idx, head, row_start:row_end, dim_start:dim_end]

    return (
        sequence_start,
        sequence_length,
        input_tile,
        k_tma_tile,
        input_elem,
        input_base,
        output_index,
        lse_index,
    )


def _unpack_kernel_types(kernel_types):
    """Return the dynamic kernel signature values in the order used by kernels."""
    return (
        kernel_types["total_seq_q"],
        kernel_types["total_seq_kv"],
        kernel_types["batch"],
        kernel_types["heads"],
        kernel_types["heads_kv"],
        kernel_types["seqlen_q"],
        kernel_types["seqlen_kv"],
        kernel_types["actual_dim_qk"],
        kernel_types["actual_dim_vo"],
        kernel_types["q_stride_b"],
        kernel_types["q_stride_s"],
        kernel_types["q_stride_h"],
        kernel_types["k_stride_b"],
        kernel_types["k_stride_s"],
        kernel_types["k_stride_h"],
        kernel_types["v_stride_b"],
        kernel_types["v_stride_s"],
        kernel_types["v_stride_h"],
        kernel_types["o_stride_b"],
        kernel_types["o_stride_s"],
        kernel_types["o_stride_h"],
        kernel_types["q_type"],
        kernel_types["k_type"],
        kernel_types["v_type"],
        kernel_types["o_type"],
        kernel_types["lse_type"],
        kernel_types["q_strides"],
        kernel_types["k_strides"],
        kernel_types["kv_shape"],
    )


def _make_kernel_types(is_varlen, dtype, accum_dtype, heads_q_eq_heads_kv=True):
    """Build the shared dynamic tensor signature used by all MP32 kernels."""
    total_seq_q = T.dynamic("total_seq_q")
    total_seq_kv = T.dynamic("total_seq_kv")
    batch = T.dynamic("batch")
    heads = T.dynamic("heads")
    heads_kv = heads if heads_q_eq_heads_kv else T.dynamic("heads_kv")
    seqlen_q = T.dynamic("seqlen_q")
    seqlen_kv = T.dynamic("seqlen_kv")
    actual_dim_qk = T.dynamic("actual_dim_qk")
    actual_dim_vo = T.dynamic("actual_dim_vo")
    stride_names = (
        "q_stride_b",
        "q_stride_s",
        "q_stride_h",
        "k_stride_b",
        "k_stride_s",
        "k_stride_h",
        "v_stride_b",
        "v_stride_s",
        "v_stride_h",
        "o_stride_b",
        "o_stride_s",
        "o_stride_h",
    )
    strides = {name: T.dynamic(name) for name in stride_names}
    q_shape = (
        (total_seq_q, heads, actual_dim_qk)
        if is_varlen
        else (batch, seqlen_q, heads, actual_dim_qk)
    )
    kv_shape = (
        (total_seq_kv, heads_kv, actual_dim_qk)
        if is_varlen
        else (batch, seqlen_kv, heads_kv, actual_dim_qk)
    )
    v_shape = (
        (total_seq_kv, heads_kv, actual_dim_vo)
        if is_varlen
        else (batch, seqlen_kv, heads_kv, actual_dim_vo)
    )
    o_shape = (
        (total_seq_q, heads, actual_dim_vo)
        if is_varlen
        else (batch, seqlen_q, heads, actual_dim_vo)
    )
    q_strides = (
        (strides["q_stride_s"], strides["q_stride_h"], 1)
        if is_varlen
        else (
            strides["q_stride_b"],
            strides["q_stride_s"],
            strides["q_stride_h"],
            1,
        )
    )
    k_strides = (
        (strides["k_stride_s"], strides["k_stride_h"], 1)
        if is_varlen
        else (
            strides["k_stride_b"],
            strides["k_stride_s"],
            strides["k_stride_h"],
            1,
        )
    )
    v_strides = (
        (strides["v_stride_s"], strides["v_stride_h"], 1)
        if is_varlen
        else (
            strides["v_stride_b"],
            strides["v_stride_s"],
            strides["v_stride_h"],
            1,
        )
    )
    o_strides = (
        (strides["o_stride_s"], strides["o_stride_h"], 1)
        if is_varlen
        else (
            strides["o_stride_b"],
            strides["o_stride_s"],
            strides["o_stride_h"],
            1,
        )
    )
    return {
        "total_seq_q": total_seq_q,
        "total_seq_kv": total_seq_kv,
        "batch": batch,
        "heads": heads,
        "heads_kv": heads_kv,
        "seqlen_q": seqlen_q,
        "seqlen_kv": seqlen_kv,
        "actual_dim_qk": actual_dim_qk,
        "actual_dim_vo": actual_dim_vo,
        **strides,
        "q_type": T.StridedTensor(q_shape, q_strides, dtype),
        "k_type": T.StridedTensor(kv_shape, k_strides, dtype),
        "v_type": T.StridedTensor(v_shape, v_strides, dtype),
        "o_type": T.StridedTensor(o_shape, o_strides, dtype),
        "lse_type": T.Tensor(
            (heads, total_seq_q) if is_varlen else (batch, heads, seqlen_q),
            accum_dtype,
        ),
        "q_strides": q_strides,
        "k_strides": k_strides,
        "kv_shape": kv_shape,
    }


__all__ = [
    "_MP32_FORWARD_SOURCE",
    "_perm_g2s",
    "_perm_r2r",
    "_tilelang_dtype_nbytes",
    "_to_tilelang_dtype",
    "_current_musa_cc",
    "_check_attention_strides",
    "_alloc_phase_registers",
    "_annotate_consumer_layout",
    "_assume_attention_strides",
    "_make_attention_accessors",
    "_unpack_kernel_types",
    "_make_kernel_types",
    "cosize",
]
