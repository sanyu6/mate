#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/int_tuple.hpp>

#include "mp32_mqa_logits_blockscale_utils.hpp"

namespace mate::deep_gemm {

template <uint32_t kNextN, uint32_t kTileKV, uint32_t kNumMP, bool kIsContextLens2D>
__global__ __launch_bounds__(1024, 1) void mp32_paged_mqa_logits_blockscale_metadata(uint32_t        num_requests,
                                                                                     uint32_t const* context_lens,
                                                                                     uint32_t*       metadata) {
  constexpr uint32_t kNumThreads = 1024;
  constexpr uint32_t kNumWarps   = kNumThreads / mutlass::NumThreadsPerWarp;

  static_assert(kNextN > 0 && kTileKV > 0, "paged MQA logits requires positive next_n and tile_kv");
  static_assert(kNumMP > 0, "paged MQA logits requires positive num_mps");

  extern __shared__ uint32_t smem[];
  uint32_t*                  request_work_prefix = smem;
  uint32_t*                  warp_sums           = smem + num_requests;

  uint32_t const thread_idx = threadIdx.x;
  uint32_t const lane_idx   = mutlass::canonical_lane_idx();
  uint32_t const warp_idx   = mutlass::canonical_warp_idx();

  for (uint32_t r = thread_idx; r < num_requests; r += kNumThreads) {
    uint32_t const lens_idx    = kIsContextLens2D ? r * kNextN + kNextN - 1 : r;
    uint32_t const context_len = context_lens[lens_idx];
    request_work_prefix[r]     = mute::ceil_div(context_len, kTileKV) * kNextN;
  }
  __syncthreads();

  uint32_t const items_per_thread = mute::ceil_div(num_requests, kNumThreads) | 1u;
  uint32_t const item_begin       = thread_idx * items_per_thread;
  uint32_t const item_end =
      (item_begin + items_per_thread < num_requests) ? item_begin + items_per_thread : num_requests;

  uint32_t local_sum = 0;
  for (uint32_t r = item_begin; r < item_end; ++r) {
    local_sum += request_work_prefix[r];
    request_work_prefix[r] = local_sum;
  }

  uint32_t const thread_inclusive_prefix = mqa_warp_inclusive_sum(local_sum, lane_idx);
  if (lane_idx == mutlass::NumThreadsPerWarp - 1) {
    warp_sums[warp_idx] = thread_inclusive_prefix;
  }
  __syncthreads();

  uint32_t const warp_total            = warp_sums[lane_idx];
  uint32_t const warp_inclusive_prefix = mqa_warp_inclusive_sum(warp_total, lane_idx);
  uint32_t const warp_offset           = __shfl_sync(0xffffffff, warp_inclusive_prefix - warp_total, warp_idx);
  uint32_t const thread_offset         = thread_inclusive_prefix - local_sum + warp_offset;
  for (uint32_t r = item_begin; r < item_end; ++r) {
    request_work_prefix[r] += thread_offset;
  }
  __syncthreads();

  uint32_t const total_work  = num_requests > 0 ? request_work_prefix[num_requests - 1] : 0;
  uint32_t const work_per_mp = total_work / kNumMP;
  uint32_t const remain_work = total_work % kNumMP;
  for (uint32_t mp_idx = thread_idx; mp_idx <= kNumMP; mp_idx += kNumThreads) {
    uint32_t const cu_work = mp_idx * work_per_mp + (mp_idx < remain_work ? mp_idx : remain_work);

    // An exact prefix boundary belongs to the next request.
    uint32_t lo = 0;
    uint32_t hi = num_requests;
    while (lo < hi) {
      uint32_t const mid = lo + (hi - lo) / 2;
      if (request_work_prefix[mid] <= cu_work) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }

    uint32_t const request_idx = lo;

    uint32_t tile_kv_idx_in_request;
    if (request_idx < num_requests) {
      uint32_t const work_before     = request_idx == 0 ? 0 : request_work_prefix[request_idx - 1];
      uint32_t const work_in_request = cu_work - work_before;
      tile_kv_idx_in_request         = work_in_request / kNextN;
    } else {
      tile_kv_idx_in_request = 0;
    }

    metadata[mp_idx * 2]     = request_idx;
    metadata[mp_idx * 2 + 1] = tile_kv_idx_in_request;
  }
}

template <uint32_t kTileKV, uint32_t kNumMP>
__global__ __launch_bounds__(1024,
                             1) void mp32_varlen_paged_mqa_logits_blockscale_metadata(uint32_t num_requests,
                                                                                      uint32_t num_q_tokens_total,
                                                                                      uint32_t const* context_lens,
                                                                                      uint32_t const* indices,
                                                                                      uint32_t*       metadata) {
  constexpr uint32_t kNumThreads = 1024;
  constexpr uint32_t kNumWarps   = kNumThreads / mutlass::NumThreadsPerWarp;

  static_assert(kTileKV > 0 && kNumMP > 0, "paged MQA logits requires positive tile_kv and num_mps");

  extern __shared__ uint32_t smem[];
  uint32_t*                  request_q_token_start_idx   = smem;
  uint32_t*                  request_work_prefix         = request_q_token_start_idx + num_requests;
  uint32_t*                  warp_sums                   = request_work_prefix + num_requests;
  uint32_t*                  num_logical_requests_shared = warp_sums + kNumWarps;

  uint32_t const thread_idx = threadIdx.x;
  uint32_t const lane_idx   = mutlass::canonical_lane_idx();
  uint32_t const warp_idx   = mutlass::canonical_warp_idx_sync();

  uint32_t const num_tokens_per_thread = mute::ceil_div(num_q_tokens_total, kNumThreads) | 1u;
  uint32_t const token_begin           = mute::min(thread_idx * num_tokens_per_thread, num_q_tokens_total);
  uint32_t const token_end             = mute::min(token_begin + num_tokens_per_thread, num_q_tokens_total);

  uint32_t prev_token_request_id     = token_begin > 0 ? indices[token_begin - 1] : 0u;
  uint32_t prev_token_request_id_tmp = prev_token_request_id;
  uint32_t num_request_starts        = 0;
  for (uint32_t token_idx = token_begin; token_idx < token_end; ++token_idx) {
    num_request_starts += (token_idx == 0) || (indices[token_idx] != prev_token_request_id_tmp);
    prev_token_request_id_tmp = indices[token_idx];
  }

  uint32_t request_idx = mqa_cta_exclusive_sum<kNumWarps>(num_request_starts, warp_sums, lane_idx, warp_idx);

  prev_token_request_id_tmp = prev_token_request_id;
  for (uint32_t token_idx = token_begin; token_idx < token_end; ++token_idx) {
    if ((token_idx == 0) || (indices[token_idx] != prev_token_request_id_tmp)) {
      request_q_token_start_idx[request_idx++] = token_idx;
    }
    prev_token_request_id_tmp = indices[token_idx];
  }

  if (thread_idx == kNumThreads - 1) {
    *num_logical_requests_shared = request_idx;
  }
  __syncthreads();

  uint32_t const num_logical_requests = *num_logical_requests_shared;

  for (uint32_t r = thread_idx; r < num_logical_requests; r += kNumThreads) {
    uint32_t const q_token_start_idx = request_q_token_start_idx[r];
    uint32_t const q_token_end_idx =
        (r + 1 < num_logical_requests) ? request_q_token_start_idx[r + 1] : num_q_tokens_total;
    uint32_t const context_len = context_lens[q_token_end_idx - 1];
    request_work_prefix[r]     = mute::ceil_div(context_len, kTileKV) * (q_token_end_idx - q_token_start_idx);
  }
  __syncthreads();

  uint32_t const items_per_thread = mute::ceil_div(num_logical_requests, kNumThreads) | 1u;
  uint32_t const item_begin       = mute::min(thread_idx * items_per_thread, num_logical_requests);
  uint32_t const item_end         = mute::min(item_begin + items_per_thread, num_logical_requests);

  uint32_t local_sum = 0;
  for (uint32_t r = item_begin; r < item_end; ++r) {
    local_sum += request_work_prefix[r];
  }
  uint32_t const thread_offset = mqa_cta_exclusive_sum<kNumWarps>(local_sum, warp_sums, lane_idx, warp_idx);
  uint32_t       thread_prefix = thread_offset;
  for (uint32_t r = item_begin; r < item_end; ++r) {
    thread_prefix += request_work_prefix[r];
    request_work_prefix[r] = thread_prefix;
  }
  __syncthreads();

  uint32_t const num_total_work = num_logical_requests > 0 ? request_work_prefix[num_logical_requests - 1] : 0u;

  uint32_t const work_per_mp = num_total_work / kNumMP;
  uint32_t const remain_work = num_total_work % kNumMP;
  for (uint32_t mp_idx = thread_idx; mp_idx <= kNumMP; mp_idx += kNumThreads) {
    uint32_t const cu_work = mp_idx * work_per_mp + mute::min(mp_idx, remain_work);

    uint32_t lo = 0;
    uint32_t hi = num_logical_requests;
    while (lo < hi) {
      uint32_t const mid = lo + (hi - lo) / 2;
      if (request_work_prefix[mid] <= cu_work) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    uint32_t const request_index = lo;

    uint32_t q_token_idx;
    uint32_t tile_kv_idx_in_request;
    if (request_index < num_logical_requests) {
      uint32_t const work_before         = request_index == 0 ? 0u : request_work_prefix[request_index - 1];
      uint32_t const work_in_request     = cu_work - work_before;
      uint32_t const request_q_token_end = (request_index + 1 < num_logical_requests)
                                               ? request_q_token_start_idx[request_index + 1]
                                               : num_q_tokens_total;
      uint32_t const num_q_tokens        = request_q_token_end - request_q_token_start_idx[request_index];
      q_token_idx                        = request_q_token_start_idx[request_index];
      tile_kv_idx_in_request             = work_in_request / num_q_tokens;
    } else {
      q_token_idx            = num_q_tokens_total;
      tile_kv_idx_in_request = 0;
    }

    metadata[mp_idx * 2]     = q_token_idx;
    metadata[mp_idx * 2 + 1] = tile_kv_idx_in_request;
  }
}

}  // namespace mate::deep_gemm
