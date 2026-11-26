#pragma once

#include <mute/arch/simd_mp31.hpp>

#include "simd_helper.hpp"
#include "utils.hpp"

namespace mate::attention::fmha {

template <int Rows, bool HasLearnableSink, int MaxOffset = 0, bool SkipUnitRescale = false>
struct Softmax {
  using Element = float;
  using TensorT = decltype(make_tensor<Element>(Shape<Int<Rows>>{}));

  TensorT row_max, row_sum;

  float const sm_scale;
  float const sm_scale_log2;

  MUTLASS_DEVICE
  Softmax(float const softmax_scale_, float const softmax_scale_log2_)
      : sm_scale(softmax_scale_), sm_scale_log2(softmax_scale_log2_) {};

  // TODO: more simd
  template <bool IsFirst, bool CheckInf, bool OutputBf16, class AccQK, class TiledMmaQK, class Probabilities>
  MUTLASS_DEVICE auto online_softmax_impl(AccQK& acc_qk, TiledMmaQK const& tiled_mma_qk, Probabilities& probabilities) {
    auto          reduction_target_qk = reduction_target_n(tiled_mma_qk);
    constexpr int red_rank            = decltype(rank(reduction_target_qk))::value;

    Tensor acc_qk_mn = make_tensor(acc_qk.data(), layout_acc_mn(tiled_mma_qk, acc_qk.layout()));

    static_assert(size<1>(acc_qk_mn) >= 8 && size<1>(acc_qk_mn) % 8 == 0,
                  "Two-chain row reduction requires N to be at least 8 and a multiple of 8");

    TensorT correction_scales;
    Tensor  row_max_prev = make_fragment_like(row_max);

    if constexpr (IsFirst) {
      mute::fill(correction_scales, Element{1.f});
    } else {
      mute::copy(row_max, row_max_prev);
    }

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_qk_mn); ++i) {
      // RowMax
      float4 row_max_even = make_float4(acc_qk_mn(i, 0), acc_qk_mn(i, 1), acc_qk_mn(i, 2), acc_qk_mn(i, 3));
      float4 row_max_odd  = make_float4(acc_qk_mn(i, 4), acc_qk_mn(i, 5), acc_qk_mn(i, 6), acc_qk_mn(i, 7));

      MUTLASS_PRAGMA_UNROLL
      for (int j = 8; j < size<1>(acc_qk_mn); j += 8) {
        float4 vec_even =
            make_float4(acc_qk_mn(i, j + 0), acc_qk_mn(i, j + 1), acc_qk_mn(i, j + 2), acc_qk_mn(i, j + 3));
        float4 vec_odd =
            make_float4(acc_qk_mn(i, j + 4), acc_qk_mn(i, j + 5), acc_qk_mn(i, j + 6), acc_qk_mn(i, j + 7));
        vmax_reduce_rotated(row_max_even, row_max_even, vec_even);
        vmax_reduce_rotated(row_max_odd, row_max_odd, vec_odd);
      }
      float4 row_max_cur;
      vmax_reduce_rotated(row_max_cur, row_max_even, row_max_odd);
      float2 v2_row_max_cur;
      mute::max(v2_row_max_cur, make_float2(row_max_cur.x, row_max_cur.y), make_float2(row_max_cur.z, row_max_cur.w));
      row_max(i) = max(v2_row_max_cur.x, v2_row_max_cur.y);

      for_each(make_seq<red_rank>{}, [&](auto r) {
        MUTLASS_PRAGMA_UNROLL
        for (int j = 1; j < shape<r>(reduction_target_qk); j *= 2) {
          row_max(i) = max(row_max(i), __shfl_xor_sync(uint32_t(-1), row_max(i), stride<r>(reduction_target_qk) * j));
        }
      });

      if (!IsFirst) {
        row_max(i) = max(row_max_prev(i), row_max(i));
      }

      bool    is_all_masked = false;
      Element max_scaled    = row_max(i);
      if constexpr (CheckInf) {
        is_all_masked = row_max(i) == (-std::numeric_limits<Element>::infinity());
        max_scaled    = is_all_masked ? Element{0} : row_max(i);
      }

      // if (i == 0 && threadIdx.x == 128) {
      //   printf("first:%d acc:%f %f %f %f\n", IsFirst, acc_qk_mn(i, 0), acc_qk_mn(i, 1), acc_qk_mn(i, 2), acc_qk_mn(i,
      //   3));
      // }

      // Exp
      Element const scale_max = max_scaled * sm_scale_log2;
      // MUTLASS_PRAGMA_UNROLL
      // for (int j = 0; j < size<1>(acc_qk_mn); j += 4) {
      //   acc_qk_mn(i, j + 0) = acc_qk_mn(i, j + 0) * sm_scale_log2 - scale_max + MaxOffset;
      //   acc_qk_mn(i, j + 1) = acc_qk_mn(i, j + 1) * sm_scale_log2 - scale_max + MaxOffset;
      //   acc_qk_mn(i, j + 2) = acc_qk_mn(i, j + 2) * sm_scale_log2 - scale_max + MaxOffset;
      //   acc_qk_mn(i, j + 3) = acc_qk_mn(i, j + 3) * sm_scale_log2 - scale_max + MaxOffset;

      //   float4 v4f32_src =
      //       make_float4(acc_qk_mn(i, j + 0), acc_qk_mn(i, j + 1), acc_qk_mn(i, j + 2), acc_qk_mn(i, j + 3));

      //   mute::fast_exp2(v4f32_src, v4f32_src);

      //   acc_qk_mn(i, j + 0) = v4f32_src.x;
      //   acc_qk_mn(i, j + 1) = v4f32_src.y;
      //   acc_qk_mn(i, j + 2) = v4f32_src.z;
      //   acc_qk_mn(i, j + 3) = v4f32_src.w;
      // }
      Element const bias = Element(MaxOffset) - scale_max;

      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_qk_mn); j += 4) {
        float32x4_t values = {
            acc_qk_mn(i, j + 0),
            acc_qk_mn(i, j + 1),
            acc_qk_mn(i, j + 2),
            acc_qk_mn(i, j + 3),
        };

        vfma(values, sm_scale_log2, values, bias);

        if constexpr (OutputBf16) {
          Tensor       probabilities_frg = recast<uint64_t>(probabilities);
          bfloat16x4_t values_bf16;
#if defined(MATE_FMHA_WORKAROUND_BST_OVERLAP)
          MUTLASS_PRAGMA_UNROLL
          for (int k = 0; k < 4; ++k) {
            values_bf16[k] = __builtin_bit_cast(__bf16, mutlass::bfloat16_t(exp2f(values[k])));
          }
#else
          vexp2(values_bf16, values);
#endif
          probabilities_frg((i * size<1>(acc_qk_mn) + j) / 4) = __builtin_bit_cast(uint64_t, values_bf16);
        } else {
#if defined(MATE_FMHA_WORKAROUND_BST_OVERLAP)
          values[0] = exp2f(values[0]);
          values[1] = exp2f(values[1]);
          values[2] = exp2f(values[2]);
          values[3] = exp2f(values[3]);
#else
          vexp2(values, values);
#endif

          acc_qk_mn(i, j + 0) = values[0];
          acc_qk_mn(i, j + 1) = values[1];
          acc_qk_mn(i, j + 2) = values[2];
          acc_qk_mn(i, j + 3) = values[3];
        }
      }

      // RowSum
      if constexpr (!IsFirst) {
        correction_scales(i) = is_all_masked ? Element{1.f} : exp2f((row_max_prev(i) - row_max(i)) * sm_scale_log2);
        row_sum(i)           = correction_scales(i) * row_sum(i);
      }

      Element row_sum_block;
      if constexpr (OutputBf16) {
        Tensor       probabilities_frg = recast<uint64_t>(probabilities);
        int const    row_offset        = i * size<1>(acc_qk_mn) / 4;
        bfloat16x4_t row_sum_even      = __builtin_bit_cast(bfloat16x4_t, probabilities_frg(row_offset));
        bfloat16x4_t row_sum_odd       = __builtin_bit_cast(bfloat16x4_t, probabilities_frg(row_offset + 1));
        MUTLASS_PRAGMA_UNROLL
        for (int j = 8; j < size<1>(acc_qk_mn); j += 8) {
          bfloat16x4_t values_even = __builtin_bit_cast(bfloat16x4_t, probabilities_frg(row_offset + j / 4));
          bfloat16x4_t values_odd  = __builtin_bit_cast(bfloat16x4_t, probabilities_frg(row_offset + j / 4 + 1));
          vadd(row_sum_even, values_even, row_sum_even);
          vadd(row_sum_odd, values_odd, row_sum_odd);
        }
        bfloat16x4_t row_sum_bf16;
        vadd(row_sum_bf16, row_sum_even, row_sum_odd);
        row_sum_block = vsum(row_sum_bf16);
      } else {
        float4 row_sum_cur;
        MUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size<1>(acc_qk_mn); j += 4) {
          float4 vec_acc =
              make_float4(acc_qk_mn(i, j + 0), acc_qk_mn(i, j + 1), acc_qk_mn(i, j + 2), acc_qk_mn(i, j + 3));
          if (j == 0) {
            row_sum_cur = vec_acc;
          } else {
            mute::add(row_sum_cur, vec_acc, row_sum_cur);
          }
        }
        float2 v2_row_sum_cur;
        mute::add(v2_row_sum_cur, make_float2(row_sum_cur.x, row_sum_cur.y), make_float2(row_sum_cur.z, row_sum_cur.w));
        row_sum_block = v2_row_sum_cur.x + v2_row_sum_cur.y;
      }
      row_sum(i) = IsFirst ? row_sum_block : (row_sum(i) + row_sum_block);
    }

    return correction_scales;
  }

  template <bool IsFirst, bool CheckInf, class AccQK, class TiledMmaQK>
  MUTLASS_DEVICE auto online_softmax(AccQK& acc_qk, TiledMmaQK const& tiled_mma_qk) {
    return online_softmax_impl<IsFirst, CheckInf, false>(acc_qk, tiled_mma_qk, acc_qk);
  }

  template <bool IsFirst,
            bool CheckInf,
            class AccQK,
            class TiledMmaQK,
            class EngineProbabilities,
            class LayoutProbabilities>
  MUTLASS_DEVICE auto online_softmax_bf16(AccQK&                                            acc_qk,
                                          TiledMmaQK const&                                 tiled_mma_qk,
                                          Tensor<EngineProbabilities, LayoutProbabilities>& probabilities) {
    return online_softmax_impl<IsFirst, CheckInf, true>(acc_qk, tiled_mma_qk, probabilities);
  }

  template <class AccPV, class TiledMmaPV>
  MUTLASS_DEVICE void rescale_o(AccPV& acc_pv, TiledMmaPV const& tiled_mma_pv, TensorT const& correction_scales) {
    if constexpr (SkipUnitRescale) {
      // If correction_scales is 1.0f, there is no need to perform vectorized rescale.
      bool needs_rescale = false;
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Rows; ++i) {
        needs_rescale |= (correction_scales(i) != 1.f);
      }
      if (!__any_sync(uint32_t(-1), needs_rescale)) {
        return;
      }
    }

    Tensor acc_pv_mn = make_tensor(acc_pv.data(), layout_acc_mn(tiled_mma_pv, acc_pv.layout()));
    static_assert(size<0>(acc_pv_mn) == Rows);
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_pv_mn); ++i) {
      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_pv_mn); j += 4) {
        acc_pv_mn(i, j + 0) *= correction_scales(i);
        acc_pv_mn(i, j + 1) *= correction_scales(i);
        acc_pv_mn(i, j + 2) *= correction_scales(i);
        acc_pv_mn(i, j + 3) *= correction_scales(i);
      }
    }
  }

  template <class AccPV, class TiledMmaPV, class SinkVal>
  MUTLASS_DEVICE auto tail(AccPV&            acc_pv,
                           TiledMmaPV const& tiled_mma_pv,
                           SinkVal const&    sink_vals,
                           float             final_scale = 1.f) {
    static_assert(size(SinkVal{}) == Rows);

    Tensor acc_pv_mn = make_tensor(acc_pv.data(), layout_acc_mn(tiled_mma_pv, acc_pv.layout()));

    // update sum across threads
    auto reduction_target  = reduction_target_n(tiled_mma_pv);
    int constexpr red_rank = decltype(rank(reduction_target))::value;

    for_each(make_seq<red_rank>{}, [&](auto r) {
      MUTLASS_PRAGMA_UNROLL
      for (int j = 1; j < shape<r>(reduction_target); j *= 2) {
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<0>(acc_pv_mn); i++) {
          row_sum(i) = row_sum(i) + __shfl_xor_sync(uint32_t(-1), row_sum(i), stride<r>(reduction_target) * j);
        }
      }
    });

    Tensor lse = make_fragment_like(row_sum);

    // if (threadIdx.x == 128) {
    //   printf("row_sum:%f\n", row_sum(0));
    // }

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<0>(acc_pv_mn); ++i) {
      Element sum = row_sum(i);

      if constexpr (HasLearnableSink) {
        sum += exp2f(sink_vals(i) * M_LOG2Ef32 - row_max(i) * sm_scale_log2 + MaxOffset);
      }

      Element inv_sum      = (sum == 0.f || sum != sum) ? 0.f : __frcp_rn(sum);
      Element output_scale = inv_sum * final_scale;

      Element sum_lse = sum;
      if constexpr (MaxOffset > 0) {
        sum_lse *= 1.f / float(1 << MaxOffset);
      }
      lse(i) = (sum == 0.f || sum != sum) ? -std::numeric_limits<float>::infinity()
                                          : row_max(i) * sm_scale + __logf(sum_lse);

      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(acc_pv_mn); j += 4) {
        acc_pv_mn(i, j + 0) *= output_scale;
        acc_pv_mn(i, j + 1) *= output_scale;
        acc_pv_mn(i, j + 2) *= output_scale;
        acc_pv_mn(i, j + 3) *= output_scale;
      }
    }
    return lse;
  }
};

}  // namespace mate::attention::fmha
