#pragma once

#include "mate/mega_moe/prefill/fp4_fp4/symmetric_layout.hpp"
namespace mega_moe::prefill::mp32::fp4_fp4 {

using PrefillStage3Float4 = float __attribute__((vector_size(16)));

struct Stage3Config {
  static constexpr uint32_t kNumThreads = 128, kBlocksPerMP = 24;
};

template <class L>
__global__ __launch_bounds__(Stage3Config::kNumThreads,
                             1) void stage3_kernel(char* base, uint32_t num_tokens, mutlass::bfloat16_t* output) {
  constexpr uint32_t kHidden = L::kHidden, kNumTopk = L::kTopK, kNumThreads = Stage3Config::kNumThreads;
  auto*              routes  = L::template at<int64_t>(base, L::kRoutes);
  auto*              weights = L::template at<float>(base, L::kWeights);
  auto*              combine = L::template at<mutlass::bfloat16_t>(base, L::kCombine);

  constexpr uint32_t kElemsPerPackedLoad = sizeof(uint4) / sizeof(mutlass::bfloat16_t);
  static_assert(kHidden % kElemsPerPackedLoad == 0, "Stage 3 hidden must pack into uint4 loads");
  static_assert(kElemsPerPackedLoad == 8, "Stage 3 vector conversion expects eight BF16 elements per load");
  constexpr uint32_t kNumPackedLoads = kHidden / kElemsPerPackedLoad;
  constexpr uint32_t kNumWarps       = kNumThreads / mutlass::NumThreadsPerWarp;
  constexpr uint32_t kNumHiddenWarpTiles =
      (kNumPackedLoads + mutlass::NumThreadsPerWarp - 1) / mutlass::NumThreadsPerWarp;
  const uint32_t warp_idx = threadIdx.x / mutlass::NumThreadsPerWarp;
  const uint32_t lane_idx = threadIdx.x % mutlass::NumThreadsPerWarp;

  for (uint32_t token_idx = blockIdx.x * kNumWarps + warp_idx; token_idx < num_tokens;
       token_idx += gridDim.x * kNumWarps) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
    __musa_loop_transparent_outermost();
#endif
    const auto* topk_idx_ptr     = routes + uint64_t(token_idx) * kNumTopk;
    const auto* topk_weights_ptr = weights + uint64_t(token_idx) * kNumTopk;

    for (uint32_t hidden_tile_idx = 0; hidden_tile_idx < kNumHiddenWarpTiles; ++hidden_tile_idx) {
      const uint32_t packed_idx = hidden_tile_idx * mutlass::NumThreadsPerWarp + lane_idx;
      if (packed_idx >= kNumPackedLoads) {
        continue;
      }

      int4                packed_output = make_int4(0, 0, 0, 0);
      PrefillStage3Float4 reduced_lo    = {};
      PrefillStage3Float4 reduced_hi    = {};
#pragma unroll
      for (uint32_t topk_idx = 0; topk_idx < kNumTopk; ++topk_idx) {
        const int64_t expert_idx = __ldg(topk_idx_ptr + topk_idx);
        if (expert_idx < 0) {
          continue;
        }
        const float  route_weight = __ldg(topk_weights_ptr + topk_idx);
        const auto*  partial      = combine + (uint64_t(token_idx) * kNumTopk + topk_idx) * kHidden;
        const auto*  packed_ptr   = reinterpret_cast<int4 const*>(partial) + packed_idx;
        const int4   packed       = __lsu_ld_cache_hint(packed_ptr, 3, 3, 0, 0);
        const auto*  packed_bf16  = reinterpret_cast<const __mt_bfloat164*>(&packed);
        const float4 partial_lo4  = __bfloat1642float4(packed_bf16[0]);
        const float4 partial_hi4  = __bfloat1642float4(packed_bf16[1]);
        const auto&  partial_lo   = reinterpret_cast<const PrefillStage3Float4&>(partial_lo4);
        const auto&  partial_hi   = reinterpret_cast<const PrefillStage3Float4&>(partial_hi4);
        // Preserve FP32 RNE before final BF16 rounding. This SDK's BST4
        // FMA builtin differs from fmaf by an ULP at sensitive boundaries.
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          reduced_lo[j] = fmaf(route_weight, partial_lo[j], reduced_lo[j]);
          reduced_hi[j] = fmaf(route_weight, partial_hi[j], reduced_hi[j]);
        }
      }

      auto* packed_output_bf16 = reinterpret_cast<__mt_bfloat164*>(&packed_output);
      packed_output_bf16[0]    = __float42bfloat164_rn(reinterpret_cast<const float4&>(reduced_lo));
      packed_output_bf16[1]    = __float42bfloat164_rn(reinterpret_cast<const float4&>(reduced_hi));

      auto* output_ptr = reinterpret_cast<int4*>(output + static_cast<uint64_t>(token_idx) * kHidden) + packed_idx;
      __lsu_st_cache_hint(output_ptr, packed_output, 4, 2, 1, 1);
    }
  }
}

}  // namespace mega_moe::prefill::mp32::fp4_fp4
