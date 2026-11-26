#pragma once

#include <musa_burst.h>
#include <mutlass/bfloat16.h>
#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/tensor.hpp>

namespace mate::deep_gemm {

using namespace mute;

template <class ValueType>
MUTLASS_DEVICE ValueType mqa_warp_inclusive_sum(ValueType value, uint32_t lane_idx) {
#pragma unroll
  for (uint32_t offset = 1; offset < mutlass::NumThreadsPerWarp; offset <<= 1) {
    ValueType const from_lane = __shfl_up_sync(0xffffffff, value, offset);
    if (lane_idx >= offset) {
      value += from_lane;
    }
  }
  return value;
}

template <uint32_t kNumWarps, class ValueType>
MUTLASS_DEVICE ValueType
mqa_cta_exclusive_sum(ValueType value, ValueType* warp_sums, uint32_t lane_idx, uint32_t warp_idx) {
  ValueType const warp_inclusive = mqa_warp_inclusive_sum(value, lane_idx);
  if (lane_idx == mutlass::NumThreadsPerWarp - 1) {
    warp_sums[warp_idx] = warp_inclusive;
  }
  __syncthreads();
  ValueType const warp_total          = lane_idx < kNumWarps ? warp_sums[lane_idx] : ValueType(0);
  ValueType const warp_inclusive_scan = mqa_warp_inclusive_sum(warp_total, lane_idx);
  ValueType const warp_offset         = __shfl_sync(0xffffffff, warp_inclusive_scan - warp_total, warp_idx);
  __syncthreads();
  return warp_offset + warp_inclusive - value;
}

template <uint32_t kTileQ, uint32_t kTileKV, uint32_t kKVTokenAlignment = 4>
MUTLASS_DEVICE mute::tuple<uint32_t, uint32_t> get_mqa_logits_kv_span(uint32_t        q_tile_idx,
                                                                      uint32_t        total_seqlen_q,
                                                                      uint32_t        seqlen_kv,
                                                                      uint32_t const* cu_seqlen_k_start,
                                                                      uint32_t const* cu_seqlen_k_end) {
  uint32_t range_start = seqlen_kv;
  uint32_t range_end   = 0;

  MUTE_UNROLL
  for (uint32_t i = 0; i < kTileQ; ++i) {
    uint32_t const q_row = min(q_tile_idx * kTileQ + i, total_seqlen_q - 1);

    uint32_t const ks = min(__ldg(cu_seqlen_k_start + q_row), seqlen_kv);
    uint32_t const ke = min(__ldg(cu_seqlen_k_end + q_row), seqlen_kv);

    range_start = min(range_start, ks);
    range_end   = max(range_end, ke);
  }

  uint32_t const kv_token_base = range_start / kKVTokenAlignment * kKVTokenAlignment;
  return {kv_token_base, ceil_div(range_end - kv_token_base, kTileKV)};
}

template <class Threshold, class Source, class Reference>
MUTE_HOST_DEVICE constexpr auto mqa_layout_separate(Threshold const& threshold,
                                                    Source const&    source,
                                                    Reference const& reference) {
  auto lower = filter(transform_layout(source, reference, [&](auto const& s, auto const& r) {
    if constexpr (decltype(r < threshold)::value) {
      return s;
    } else {
      return make_layout(_1{}, _0{});
    }
  }));
  auto upper = filter(transform_layout(source, reference, [&](auto const& s, auto const& r) {
    if constexpr (decltype(r >= threshold)::value) {
      return s;
    } else {
      return make_layout(_1{}, _0{});
    }
  }));
  return make_tuple(lower, upper);
}

template <class TiledMma>
MUTE_HOST_DEVICE constexpr auto mqa_reduction_target_n(TiledMma const&) {
  auto separated = mqa_layout_separate(get<0>(typename TiledMma::Shape_MNK{}),
                                       make_layout(shape<0>(typename TiledMma::LayoutC_TV{})),
                                       stride<0>(typename TiledMma::LayoutC_TV{}));
  return get<1>(separated);
}

template <class TiledMma, class Acc>
MUTE_HOST_DEVICE constexpr auto mqa_layout_acc_mn(TiledMma const&, Acc const& acc) {
  auto [value_m, value_n] = mqa_layout_separate(
      get<0>(typename TiledMma::Shape_MNK{}), get<0>(acc), stride<1>(typename TiledMma::LayoutC_TV{}));
  return make_layout(make_layout(value_m, get<1>(acc)), make_layout(value_n, get<2>(acc)));
}

template <class Engine, class Layout>
MUTLASS_DEVICE void apply_relu(Tensor<Engine, Layout>& accum) {
  using F4Vec     = float __attribute__((ext_vector_type(4)));
  auto accum_vec4 = recast<F4Vec>(accum);
  MUTE_UNROLL
  for (int i = 0; i < size(accum_vec4); ++i) {
    accum_vec4(i) = __musa_max_f_bst4_sv(0.0f, accum_vec4(i));
  }
}

template <class Fragment>
MUTLASS_DEVICE auto to_float_fragment(Fragment const& src) {
  Tensor dst = make_fragment_like<float>(src);
  MUTE_UNROLL
  for (int i = 0; i < size(src); ++i) {
    dst(i) = static_cast<float>(src(i));
  }
  return dst;
}

template <int kHeadsPerThread, int kReductionTarget, class TiledMma, class Accum, class Weight>
MUTLASS_DEVICE auto apply_weighted_reduce_head(TiledMma const& tiled_mma, Accum& accum, Weight const& tCrW) {
  Tensor tCrWHQ          = logical_divide(tCrW, Int<kHeadsPerThread>{});
  auto   accum_layout_mn = mqa_layout_acc_mn(tiled_mma, accum.layout());
  Tensor accum_mn        = make_tensor(accum.data(), accum_layout_mn);
  Tensor accum_mhq       = logical_divide(accum_mn, make_tile(_, Int<kHeadsPerThread>{}));
  Tensor final_sum       = make_tensor<float>(
      make_layout(make_shape(size<1>(tCrWHQ), size<0>(accum_layout_mn)), make_stride(size<0>(accum_layout_mn), _1{})));

  static_assert(decltype(size<1>(accum_layout_mn))::value == decltype(size(tCrW))::value,
                "MP32 MQA logits accumulator N fragment must match thread weights");

  constexpr bool EnableWeightedReduceVec = kHeadsPerThread % 4 == 0;
  constexpr bool EnableShflReduceVec     = decltype(size<1>(final_sum))::value % 4 == 0;

  MUTE_UNROLL
  for (int q = 0; q < size<0>(final_sum); ++q) {
    if constexpr (EnableWeightedReduceVec) {
      using F4Vec         = float __attribute__((ext_vector_type(4)));
      auto accum_mhq_vec4 = recast<F4Vec>(accum_mhq);
      auto tCrWHQ_vec4    = recast<F4Vec>(tCrWHQ);

      F4Vec weight4 = tCrWHQ_vec4(0, q);
      MUTE_UNROLL
      for (int m = 0; m < size<1>(final_sum); ++m) {
        accum_mhq_vec4(m, make_coord(0, q)) *= weight4;
      }

      MUTE_UNROLL
      for (int h = 1; h < size<0>(tCrWHQ_vec4); ++h) {
        F4Vec weight4 = tCrWHQ_vec4(h, q);
        MUTE_UNROLL
        for (int m = 0; m < size<1>(final_sum); ++m) {
          accum_mhq_vec4(m, make_coord(h, q)) =
              weight4 * accum_mhq_vec4(m, make_coord(h, q)) + accum_mhq_vec4(m, make_coord(h - 1, q));
        }
      }

      MUTE_UNROLL
      for (int m = 0; m < size<1>(final_sum); ++m) {
        F4Vec sum4      = accum_mhq_vec4(m, make_coord(size<0>(tCrWHQ_vec4) - 1, q));
        final_sum(q, m) = sum4[0] + sum4[1] + sum4[2] + sum4[3];
      }

    } else {
      MUTE_UNROLL
      for (int m = 0; m < size<1>(final_sum); ++m) {
        accum_mhq(m, make_coord(0, q)) *= tCrWHQ(0, q);
      }

      MUTE_UNROLL
      for (int h = 1; h < kHeadsPerThread; ++h) {
        MUTE_UNROLL
        for (int m = 0; m < size<1>(final_sum); ++m) {
          accum_mhq(m, make_coord(h, q)) =
              tCrWHQ(h, q) * accum_mhq(m, make_coord(h, q)) + accum_mhq(m, make_coord(h - 1, q));
        }
      }

      MUTE_UNROLL
      for (int m = 0; m < size<1>(final_sum); ++m) {
        final_sum(q, m) = accum_mhq(m, make_coord(kHeadsPerThread - 1, q));
      }
    }

    if constexpr (EnableShflReduceVec) {
      using F4Vec = float __attribute__((ext_vector_type(4)));

      Tensor shuffled      = make_tensor<float>(make_shape(shape<1>(final_sum)));
      auto   sum_vec4      = recast<F4Vec>(final_sum(q, _));
      auto   shuffled_vec4 = recast<F4Vec>(shuffled);

      MUTE_UNROLL
      for (int offset = 1; offset < kReductionTarget; offset <<= 1) {
        MUTE_UNROLL
        for (int m = 0; m < size<1>(final_sum); ++m) {
          shuffled(m) = __shfl_xor_sync(0xffffffffu, final_sum(q, m), offset);
        }

        MUTE_UNROLL
        for (int m = 0; m < size(sum_vec4); ++m) {
          sum_vec4(m) += shuffled_vec4(m);
        }
      }

    } else {
      MUTE_UNROLL
      for (int m = 0; m < size<1>(final_sum); ++m) {
        MUTE_UNROLL
        for (int offset = 1; offset < kReductionTarget; offset <<= 1) {
          final_sum(q, m) += __shfl_xor_sync(0xffffffffu, final_sum(q, m), offset);
        }
      }
    }
  }

  return final_sum;
}

}  // namespace mate::deep_gemm
