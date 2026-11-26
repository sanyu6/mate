#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/tensor.hpp>

#include "mp32_mqa_logits_blockscale_utils.hpp"

namespace mate::deep_gemm {

using namespace mute;

MUTLASS_HOST_DEVICE constexpr uint32_t mqa_logits_meta_span_word_offset(uint32_t num_mps) {
  return ((3 * num_mps + 1) / 2) * 2;
}

template <uint32_t kMaxLog2>
MUTLASS_DEVICE uint32_t mqa_logits_prefix_upper_bound(uint32_t const* prefix, uint32_t count, uint32_t target) {
  uint32_t pos = 0;
#pragma unroll
  for (uint32_t step = 1u << (kMaxLog2 - 1); step > 0; step >>= 1) {
    uint32_t const candidate = pos + step;
    if (candidate <= count && prefix[candidate - 1] <= target) {
      pos = candidate;
    }
  }
  return pos;
}

template <uint32_t kTileQ,
          uint32_t kTileKV,
          uint32_t kNumMP,
          uint32_t kSegmentCost       = 1,
          uint32_t kSplitCost         = 1,
          uint32_t kMaxNumQBlocksLog2 = 15>
__global__ __launch_bounds__(256, 1) void mp32_mqa_logits_blockscale_metadata(uint32_t        num_q_tokens,
                                                                              uint32_t        num_kv_tokens,
                                                                              uint32_t const* cu_seq_len_k_start,
                                                                              uint32_t const* cu_seq_len_k_end,
                                                                              uint32_t*       schedule_meta) {
  static_assert(kTileQ > 0 && kTileKV > 0, "non-paged MQA logits requires positive tile_q and tile_kv");
  static_assert(kNumMP > 0, "non-paged MQA logits requires positive num_mps");
  static_assert(kSegmentCost > 0 && kSplitCost > 0, "non-paged MQA logits requires positive costs");

  constexpr uint32_t kNumThreads = 256;
  constexpr uint32_t kNumWarps   = kNumThreads / mutlass::NumThreadsPerWarp;
  static_assert(kNumMP <= kNumThreads, "non-paged MQA logits requires num_mps no greater than the thread count");

  uint32_t const thread_idx = threadIdx.x;
  uint32_t const lane_idx   = mutlass::canonical_lane_idx();
  uint32_t const warp_idx   = mutlass::canonical_warp_idx();

  uint32_t const num_q_tiles = ceil_div(num_q_tokens, kTileQ);

  extern __shared__ uint32_t smem[];
  uint32_t*                  work_prefix = smem;
  uint32_t*                  cost_prefix = smem + num_q_tiles;

  __shared__ uint64_t warp_sums[kNumWarps];
  __shared__ uint64_t carry;
  if (thread_idx == 0) {
    carry = 0;
  }

  auto* kv_spans = reinterpret_cast<uint2*>(schedule_meta + mqa_logits_meta_span_word_offset(kNumMP));

  for (uint32_t tile_q_idx = thread_idx; tile_q_idx < num_q_tiles; tile_q_idx += kNumThreads) {
    auto const [kv_token_base, num_tile_kv] = get_mqa_logits_kv_span<kTileQ, kTileKV>(
        tile_q_idx, num_q_tokens, num_kv_tokens, cu_seq_len_k_start, cu_seq_len_k_end);
    kv_spans[tile_q_idx]    = uint2{kv_token_base, num_tile_kv};
    work_prefix[tile_q_idx] = num_tile_kv;
    cost_prefix[tile_q_idx] = kSplitCost * num_tile_kv + (num_tile_kv == 0 ? 0 : kSegmentCost);
  }
  __syncthreads();

  for (uint32_t base_q_tile = 0; base_q_tile < num_q_tiles; base_q_tile += kNumThreads) {
    uint32_t const tile_q_idx = base_q_tile + thread_idx;
    bool const     active     = tile_q_idx < num_q_tiles;
    uint64_t const value      = active ? (uint64_t(work_prefix[tile_q_idx]) << 32) + cost_prefix[tile_q_idx] : 0ull;

    __syncthreads();
    uint64_t const previous_carry = carry;
    uint64_t const scanned =
        mqa_cta_exclusive_sum<kNumWarps>(value, warp_sums, lane_idx, warp_idx) + value + previous_carry;

    if (active) {
      work_prefix[tile_q_idx] = uint32_t(scanned >> 32);
      cost_prefix[tile_q_idx] = uint32_t(scanned);
    }
    if (thread_idx == kNumThreads - 1) {
      carry = scanned;
    }
    __syncthreads();
  }

  uint32_t const total_work = work_prefix[num_q_tiles - 1];
  uint32_t const total_cost = cost_prefix[num_q_tiles - 1];

  if (thread_idx >= kNumMP) {
    return;
  }

  uint32_t const base      = total_cost / kNumMP;
  uint32_t const remainder = total_cost % kNumMP;

  auto const locate = [&](uint32_t boundary_idx) -> mute::tuple<uint2, uint32_t> {
    uint32_t const target = boundary_idx * base + min(boundary_idx, remainder);
    if (target == total_cost) {
      return {uint2{num_q_tiles, 0}, total_work};
    }
    uint32_t const tile_q_idx     = mqa_logits_prefix_upper_bound<kMaxNumQBlocksLog2>(cost_prefix, num_q_tiles, target);
    uint32_t const cost_before    = tile_q_idx == 0 ? 0 : cost_prefix[tile_q_idx - 1];
    uint32_t const work_before    = tile_q_idx == 0 ? 0 : work_prefix[tile_q_idx - 1];
    uint32_t const tile_kv_offset = min((max(target - cost_before, kSegmentCost) - kSegmentCost) / kSplitCost,
                                        work_prefix[tile_q_idx] - work_before - 1);
    return {uint2{tile_q_idx, tile_kv_offset}, work_before + tile_kv_offset};
  };

  auto const begin                                    = locate(thread_idx);
  auto const end                                      = locate(thread_idx + 1);
  reinterpret_cast<uint2*>(schedule_meta)[thread_idx] = get<0>(begin);
  schedule_meta[2 * kNumMP + thread_idx]              = get<1>(end) - get<1>(begin);
}

}  // namespace mate::deep_gemm
