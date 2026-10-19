#pragma once

#include <mutlass/mutlass.h>
#include <mutlass/numeric_types.h>

#include "mpxx_params.hpp"

namespace mate::flash_mla {

// Each CTA owns one query/head row and a slice of D. Warps accumulate
// disjoint splits, then reduce their partial outputs in shared memory.
template <typename ElementT, int HEAD_DIM_V, int BLOCK_D, int MAX_SPLITS, int NUM_WARPS, bool VarlenQ>
__global__ void __launch_bounds__(NUM_WARPS * 32, 1) mpxx_mla_combine_parallel_kernel(const MlaCombineParams params) {
  static_assert(BLOCK_D % 32 == 0 && HEAD_DIM_V % BLOCK_D == 0);
  static_assert(MAX_SPLITS <= NUM_WARPS * 32);
  constexpr int ElementsPerLane = BLOCK_D / 32;
  const int     batch_idx       = blockIdx.x;
  const int     row_idx         = blockIdx.y;
  const int     d_start         = blockIdx.z * BLOCK_D;
  const int     warp_idx        = threadIdx.x / 32;
  const int     lane_idx        = threadIdx.x % 32;
  const int     start_split     = __ldg(params.num_splits_ptr + batch_idx);
  const int     num_splits      = __ldg(params.num_splits_ptr + batch_idx + 1) - start_split;
  if (num_splits == 1) return;  // The main kernel already wrote the final result.

  const int rows    = params.max_q_seq_per_hk;
  int       q_start = 0;
  if constexpr (VarlenQ) {
    q_start         = __ldg(params.seqlens_q_ptr + batch_idx);
    const int q_end = __ldg(params.seqlens_q_ptr + batch_idx + 1);
    if (row_idx >= (q_end - q_start) * params.h_r) return;
  }
  const auto*      lse_accum = static_cast<const float*>(params.softmax_lseaccum_ptr);
  const auto*      o_accum   = static_cast<const float*>(params.oaccum_ptr);
  __shared__ float warp_max[NUM_WARPS];
  __shared__ float warp_sum[NUM_WARPS];
  __shared__ float weights[MAX_SPLITS];
  __shared__ float partial[NUM_WARPS][BLOCK_D];

  const float local_lse =
      threadIdx.x < num_splits ? __ldg(lse_accum + (start_split + threadIdx.x) * rows + row_idx) : -INFINITY;
  float max_lse = local_lse;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    max_lse = max(max_lse, __shfl_xor_sync(0xffffffff, max_lse, offset));
  }
  if (lane_idx == 0) warp_max[warp_idx] = max_lse;
  __syncthreads();
  max_lse = lane_idx < NUM_WARPS ? warp_max[lane_idx] : -INFINITY;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    max_lse = max(max_lse, __shfl_xor_sync(0xffffffff, max_lse, offset));
  }
  if (max_lse == -INFINITY) max_lse = 0.f;

  float sum_lse = expf(local_lse - max_lse);
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    sum_lse += __shfl_xor_sync(0xffffffff, sum_lse, offset);
  }
  if (lane_idx == 0) warp_sum[warp_idx] = sum_lse;
  __syncthreads();
  sum_lse = lane_idx < NUM_WARPS ? warp_sum[lane_idx] : 0.f;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    sum_lse += __shfl_xor_sync(0xffffffff, sum_lse, offset);
  }
  const float global_lse = sum_lse == 0.f || sum_lse != sum_lse ? INFINITY : logf(sum_lse) + max_lse;
  if (threadIdx.x < num_splits) weights[threadIdx.x] = expf(local_lse - global_lse);
  if (blockIdx.z == 0 && threadIdx.x == 0) {
    const int lse_offset =
        VarlenQ ? (row_idx % params.h_r) * params.total_q + q_start + row_idx / params.h_r : batch_idx * rows + row_idx;
    static_cast<float*>(params.softmax_lse_ptr)[lse_offset] = global_lse;
  }
  __syncthreads();

  float accum[ElementsPerLane] = {};
  for (int split = warp_idx; split < num_splits; split += NUM_WARPS) {
    const float weight = weights[split];
    // Zero-mass splits can contain undefined partial O (including NaNs).
    if (weight != 0.f) {
      const int64_t offset = (int64_t(start_split + split) * rows + row_idx) * HEAD_DIM_V + d_start;
#pragma unroll
      for (int i = 0; i < ElementsPerLane; ++i) {
        accum[i] += weight * __ldg(o_accum + offset + lane_idx + i * 32);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < ElementsPerLane; ++i) partial[warp_idx][lane_idx + i * 32] = accum[i];
  __syncthreads();

  if (threadIdx.x < BLOCK_D) {
    float result = 0.f;
#pragma unroll
    for (int warp = 0; warp < NUM_WARPS; ++warp) result += partial[warp][threadIdx.x];
    const int64_t offset = (VarlenQ ? q_start * params.h_r * params.o_row_stride : batch_idx * params.o_batch_stride) +
                           row_idx * params.o_row_stride + d_start;
    static_cast<ElementT*>(params.o_ptr)[offset + threadIdx.x] = static_cast<ElementT>(result);
  }
}

}  // namespace mate::flash_mla
