#pragma once
#ifndef MHC_KPARTS
#define MHC_KPARTS 20
#endif
#ifndef MHC_TACTIC
#define MHC_TACTIC 0
#endif
// Experimental b24 mHC: generated TCE projection adapted to native MP32,
// including MP32 accumulator coordinates and last-CTA Sinkhorn completion.
// The pinned model TileLang has an unused FP16 conversion with a narrowing
// diagnostic under this MUSA SDK. This kernel only instantiates BF16 paths.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++11-narrowing"
#include <tl_templates/musa/accelerated_ops.h>
#include <tl_templates/musa/barrier.h>
#include <tl_templates/musa/common.h>
#include <tl_templates/musa/copy.h>
#include <tl_templates/musa/cvt.h>
#include <tl_templates/musa/debug.h>
#include <tl_templates/musa/fop.h>
#include <tl_templates/musa/gemm.h>
#include <tl_templates/musa/intrin.h>
#include <tl_templates/musa/reduce.h>
#include <tl_templates/musa/scan.h>
#include <tl_templates/musa/threadblock_swizzle.h>
#pragma clang diagnostic pop

#include "mate/mega_moe/decode/fp4_fp4/mhc_tce_mma.hpp"
namespace mate::mhc_tce_horizontal {
__device__ void gemm(bfloat16_t* x, bfloat16_t* w, float* acc, int base) {
  mute::task_mhc::GemmTensorOp<32, 32, 128, 4, 1, false, true, false, bfloat16_t, bfloat16_t, float>::body(
      x, w, acc, base);
}
template <SmemSwizzleGranularity sg, SmemSwizzleStride ss, SmemSwizzleLine sl, CacheHint ih, CacheHint oh>
__device__ void load(const MUtensorDescriptor& desc, uint32_t bar, void* ptr, int c0, int c1, int d0, int d1) {
  v2i32_t c{c0, c1}, d{d0, d1};
  __musa_tme_ld_tile_stride_2d(bar,
                               make_ptr_with_address_space<AddressSpace::Shared>(cast_smem_ptr_to_uint(ptr)),
                               reinterpret_cast<uint64_t>(&desc),
                               d,
                               c,
                               int(sg),
                               int(ss),
                               int(sl),
                               0,
                               0,
                               int(ih),
                               int(oh));
}
template <SmemSwizzleGranularity sg, SmemSwizzleStride ss, SmemSwizzleLine sl, CacheHint ih, CacheHint oh>
__device__ void load(
    const MUtensorDescriptor& desc, uint32_t bar, void* ptr, int c0, int c1, int c2, int d0, int d1, int d2) {
  v3i32_t c{c0, c1, c2}, d{d0, d1, d2};
  __musa_tme_ld_tile_stride_3d(bar,
                               make_ptr_with_address_space<AddressSpace::Shared>(cast_smem_ptr_to_uint(ptr)),
                               reinterpret_cast<uint64_t>(&desc),
                               d,
                               c,
                               int(sg),
                               int(ss),
                               int(sl),
                               0,
                               0,
                               int(ih),
                               int(oh));
}

__device__ void project(float*                    out,
                        float*                    sq,
                        const MUtensorDescriptor& w_desc_1,
                        const MUtensorDescriptor& x_desc_0,
                        int                       part,
                        char*                     buf_dyn_shmem) {
  __musa_async_bar_record(16);

  void*      ws0 = ((void*)((char*)buf_dyn_shmem + 0));
  void*      ws1 = ((void*)((char*)buf_dyn_shmem + 16384));
  void*      ws2 = ((void*)((char*)buf_dyn_shmem + 32768));
  void*      xs  = ((void*)((char*)buf_dyn_shmem + 49152));
  float      acc0[8];
  float      acc1[8];
  float      acc2[8];
  float      squares[32];
  float      totals[4];
  bfloat16_t xs_local_cast[8];
  if (tl::tl_shuffle_elect<0>()) {
    __musa_async_init_arrival(1, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(2, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(3, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(4, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(5, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(6, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(7, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(8, ((1 + 31) / 32), 0);
    __musa_async_init_arrival(9, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(10, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(11, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(12, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(13, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(14, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(15, ((128 + 31) / 32), 0);
    __musa_async_init_arrival(16, ((128 + 31) / 32), 0);
  }
  tl::fence_barrier_init();
  __syncthreads_lm();
  if ((((int)threadIdx.x) < 128)) {
    for (int kb = 0; kb < (160 / MHC_KPARTS); ++kb) {
      __musa_async_wait(((kb & 1) + 9), (((kb & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        tl::mbarrier_arrive_expect_tx(((kb & 1) + 1), 8192);
        load<SmemSwizzleGranularity::B16,
             SmemSwizzleStride::B256,
             SmemSwizzleLine::B256,
             CacheHint::CACHE_NORMAL,
             CacheHint::CACHE_NORMAL>(x_desc_0,
                                      ((kb & 1) + 1),
                                      (&(((bfloat16_t*)xs)[((kb & 1) * 4096)])),
                                      ((part * (20480 / MHC_KPARTS)) + (kb * 128)),
                                      0,
                                      128,
                                      32);
      }
      __musa_async_wait(((kb & 1) + 11), (((kb & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        tl::mbarrier_arrive_expect_tx(((kb & 1) + 3), 8192);
        load<SmemSwizzleGranularity::B16,
             SmemSwizzleStride::B256,
             SmemSwizzleLine::B256,
             CacheHint::CACHE_NORMAL,
             CacheHint::CACHE_NORMAL>(w_desc_1,
                                      ((kb & 1) + 3),
                                      (&(((bfloat16_t*)ws0)[((kb & 1) * 4096)])),
                                      ((part * (20480 / MHC_KPARTS)) + (kb * 128)),
                                      0,
                                      0,
                                      128,
                                      32,
                                      1);
      }
      __musa_async_wait(((kb & 1) + 13), (((kb & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        tl::mbarrier_arrive_expect_tx(((kb & 1) + 5), 8192);
        load<SmemSwizzleGranularity::B16,
             SmemSwizzleStride::B256,
             SmemSwizzleLine::B256,
             CacheHint::CACHE_NORMAL,
             CacheHint::CACHE_NORMAL>(w_desc_1,
                                      ((kb & 1) + 5),
                                      (&(((bfloat16_t*)ws1)[((kb & 1) * 4096)])),
                                      ((part * (20480 / MHC_KPARTS)) + (kb * 128)),
                                      0,
                                      1,
                                      128,
                                      32,
                                      1);
      }
      __musa_async_wait(((kb & 1) + 15), (((kb & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        tl::mbarrier_arrive_expect_tx(((kb & 1) + 7), 8192);
        load<SmemSwizzleGranularity::B16,
             SmemSwizzleStride::B256,
             SmemSwizzleLine::B256,
             CacheHint::CACHE_NORMAL,
             CacheHint::CACHE_NORMAL>(w_desc_1,
                                      ((kb & 1) + 7),
                                      (&(((bfloat16_t*)ws2)[((kb & 1) * 4096)])),
                                      ((part * (20480 / MHC_KPARTS)) + (kb * 128)),
                                      0,
                                      2,
                                      128,
                                      32,
                                      1);
      }
    }
  }
  if (threadIdx.x >= 128 && threadIdx.x < 256) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      float broadcast_var       = 0x0p+0f /*0.000000e+00*/;
      *(tl_f4*)(acc0 + (i * 4)) = (tl_f4{static_cast<float>(broadcast_var),
                                         static_cast<float>(broadcast_var),
                                         static_cast<float>(broadcast_var),
                                         static_cast<float>(broadcast_var)});
    }
#pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      float broadcast_var_1       = 0x0p+0f /*0.000000e+00*/;
      *(tl_f4*)(acc1 + (i_1 * 4)) = (tl_f4{static_cast<float>(broadcast_var_1),
                                           static_cast<float>(broadcast_var_1),
                                           static_cast<float>(broadcast_var_1),
                                           static_cast<float>(broadcast_var_1)});
    }
#pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      float broadcast_var_2       = 0x0p+0f /*0.000000e+00*/;
      *(tl_f4*)(acc2 + (i_2 * 4)) = (tl_f4{static_cast<float>(broadcast_var_2),
                                           static_cast<float>(broadcast_var_2),
                                           static_cast<float>(broadcast_var_2),
                                           static_cast<float>(broadcast_var_2)});
    }
#pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      float broadcast_var_3          = 0x0p+0f /*0.000000e+00*/;
      *(tl_f4*)(squares + (i_3 * 4)) = (tl_f4{static_cast<float>(broadcast_var_3),
                                              static_cast<float>(broadcast_var_3),
                                              static_cast<float>(broadcast_var_3),
                                              static_cast<float>(broadcast_var_3)});
    }
    for (int kb_1 = 0; kb_1 < (160 / MHC_KPARTS); ++kb_1) {
      __musa_async_wait(((kb_1 & 1) + 1), ((kb_1 & 3) >> 1));
#pragma unroll
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        *(tl_bf8*)(xs_local_cast + 0) =
            *(tl_bf8*)(((bfloat16_t*)xs) +
                       ((((((kb_1 & 1) * 4096) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) +
                         (((((int)threadIdx.x) & 15) ^ ((((i_4 * 8) + (((int)threadIdx.x) >> 4)) + 8) & 15)) * 8)) -
                        1024));
        for (int vec = 0; vec < 2; ++vec) {
          tl_f4  __1;
          tl_bf4 v_            = *(tl_bf4*)(xs_local_cast + (vec * 4));
          ((float4*)(&__1))[0] = tl::cvt_bfloat16_to_float_x4((reinterpret_cast<tl::bfloat164_t*>(&v_))[0]);
          tl_f4 __2;
          ((float4*)(&__2))[0] = tl::cvt_bfloat16_to_float_x4((reinterpret_cast<tl::bfloat164_t*>(&v_))[0]);
          *(tl_f4*)(squares + ((i_4 * 8) + (vec * 4))) =
              tl::fma4(__1, __2, *(tl_f4*)(squares + ((i_4 * 8) + (vec * 4))));
        }
      }
      __musa_async_wait(((kb_1 & 1) + 3), ((kb_1 & 3) >> 1));
      gemm(
          (&(((bfloat16_t*)xs)[((kb_1 & 1) * 4096)])), (&(((bfloat16_t*)ws0)[((kb_1 & 1) * 4096)])), (&(acc0[0])), 128);
      __musa_async_arrive(((kb_1 & 1) + 11));
      __musa_async_wait(((kb_1 & 1) + 5), ((kb_1 & 3) >> 1));
      gemm(
          (&(((bfloat16_t*)xs)[((kb_1 & 1) * 4096)])), (&(((bfloat16_t*)ws1)[((kb_1 & 1) * 4096)])), (&(acc1[0])), 128);
      __musa_async_arrive(((kb_1 & 1) + 13));
      __musa_async_wait(((kb_1 & 1) + 7), ((kb_1 & 3) >> 1));
      gemm(
          (&(((bfloat16_t*)xs)[((kb_1 & 1) * 4096)])), (&(((bfloat16_t*)ws2)[((kb_1 & 1) * 4096)])), (&(acc2[0])), 128);
      __musa_async_arrive(((kb_1 & 1) + 9));
      __musa_async_arrive(((kb_1 & 1) + 15));
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int tid = int(threadIdx.x) - 128;
      int pos = (tid % 4) * 32 + ((tid / 4) % 8) + (tid / 32) * 8 + (j % 2) * 128 + (j / 2) * 256;
      int row = pos % 32, col = pos / 32;
      if (row < 24 && col < 24) out[(part * 24 + row) * 24 + col] = (acc0[j] + acc1[j]) + acc2[j];
    }
#pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      totals[i_6] = 0x0p+0f /*0.000000e+00*/;
#pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        totals[i_6] = (totals[i_6] + squares[((i_6 * 8) + rv)]);
      }
      totals[i_6] = tl::AllReduce<tl::SumOp, 16, 1, 128>::run(totals[i_6]);
    }
    if ((((int)threadIdx.x) % 16) == 0) {
#pragma unroll
      for (int i_7 = 0; i_7 < 12; ++i_7) {
        if (i_7 < 9) {
          float v__2 = totals[(i_7 / 3)];
          tl::store_global_32(
              (&(sq[(((((part * 72) + ((i_7 / 3) * 24)) + ((((int)threadIdx.x) >> 4) * 3)) + (i_7 % 3)) - 24)])),
              (*(uint*)(&(v__2))));
        }
      }
    }
  }
}
constexpr int kHidden          = 20480;
constexpr int kMix             = 24;
constexpr int kSlices          = 20;
constexpr int kSliceK          = 1024;
constexpr int kNBlocks         = 3;
constexpr int kPartialSqBlocks = 3;
constexpr int kOutputsPerBlock = 8;
constexpr int kWarpSize        = 32;
constexpr int kSinkhornIters   = 20;
constexpr int kMhc             = 4;

__device__ __forceinline__ float reduce_warp_ordered(float value) {
  value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 16));
  value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 8));
  value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 4));
  value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 2));
  value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, 1));
  return value;
}

__device__ __forceinline__ float sinkhorn_reduce_group4_sum(float value) {
#pragma unroll
  for (int offset = 1; offset < kMhc; offset <<= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float sinkhorn_reduce_group4_max(float value) {
#pragma unroll
  for (int offset = 1; offset < kMhc; offset <<= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ __forceinline__ float sinkhorn_reduce_column_sum(float value) {
  value += __shfl_xor_sync(0xffffffffu, value, kMhc);
  value += __shfl_xor_sync(0xffffffffu, value, 2 * kMhc);
  return value;
}

__device__ __forceinline__ float sinkhorn_sigmoid(float value) {
  return 1.0f / (1.0f + __expf(-value));
}

__device__ __forceinline__ void sinkhorn_row(const float* __restrict__ partial_mix,
                                             const float* __restrict__ partial_sq,
                                             const float* __restrict__ hc_scale,
                                             const float* __restrict__ hc_base,
                                             float* __restrict__ pre,
                                             float* __restrict__ post,
                                             float* __restrict__ comb,
                                             int   rows,
                                             int   row,
                                             int   sq_stride,
                                             float rms_eps,
                                             float hc_eps,
                                             int   lane) {
  float mix = 0.0f;
  if (lane < kMix) {
#pragma unroll
    for (int slice = 0; slice < MHC_KPARTS; ++slice) {
      mix = __fadd_rn(mix, partial_mix[(slice * rows + row) * kMix + lane]);
    }
  }
  float sq = 0.0f;
  if (lane == kMix) {
#pragma unroll
    for (int slice = 0; slice < MHC_KPARTS; ++slice) {
      sq = __fadd_rn(sq, partial_sq[(slice * rows + row) * sq_stride]);
    }
  }
  sq              = __shfl_sync(0xffffffffu, sq, kMix);
  const float rms = rsqrtf(sq * (1.0f / 20480.0f) + rms_eps);
  mix *= rms;

  if (lane < kMhc) {
    pre[row * kMhc + lane] = sinkhorn_sigmoid(mix * hc_scale[0] + hc_base[lane]) + hc_eps;
  }

  float       logit     = 0.0f;
  const bool  comb_lane = lane < kMhc * kMhc;
  const int   comb_idx  = comb_lane ? lane : 0;
  const float comb_mix  = __shfl_sync(0xffffffffu, mix, comb_lane ? 2 * kMhc + comb_idx : 0);
  if (comb_lane) {
    logit = __fmaf_rn(comb_mix, hc_scale[2], hc_base[2 * kMhc + comb_idx]);
  }

  const float row_max = sinkhorn_reduce_group4_max(logit);
  if (comb_lane) {
    logit = __expf(logit - row_max);
  }
  float row_sum   = sinkhorn_reduce_group4_sum(comb_lane ? logit : 0.0f);
  float row_recip = comb_lane && (MHC_TACTIC != 0 || (comb_idx & 3) == 0) ? __frcp_rn(row_sum + hc_eps) : 0.0f;
  if constexpr (MHC_TACTIC == 0) row_recip = __shfl_sync(0xffffffffu, row_recip, comb_idx & ~3);
  if (comb_lane) {
    logit = logit * row_recip + hc_eps;
  }

  float col_sum   = sinkhorn_reduce_column_sum(comb_lane ? logit : 0.0f);
  float col_recip = comb_lane && (MHC_TACTIC != 0 || comb_idx < kMhc) ? __frcp_rn(col_sum + hc_eps) : 0.0f;
  if constexpr (MHC_TACTIC == 0) col_recip = __shfl_sync(0xffffffffu, col_recip, comb_idx & (kMhc - 1));
  if (comb_lane) {
    logit = logit * col_recip;
  }

#pragma unroll
  for (int iteration = 1; iteration < kSinkhornIters; ++iteration) {
    row_sum   = sinkhorn_reduce_group4_sum(comb_lane ? logit : 0.0f);
    row_recip = comb_lane && (MHC_TACTIC != 0 || (comb_idx & 3) == 0) ? __frcp_rn(row_sum + hc_eps) : 0.0f;
    if constexpr (MHC_TACTIC == 0) row_recip = __shfl_sync(0xffffffffu, row_recip, comb_idx & ~3);
    if (comb_lane) {
      logit = logit * row_recip;
    }

    col_sum   = sinkhorn_reduce_column_sum(comb_lane ? logit : 0.0f);
    col_recip = comb_lane && (MHC_TACTIC != 0 || comb_idx < kMhc) ? __frcp_rn(col_sum + hc_eps) : 0.0f;
    if constexpr (MHC_TACTIC == 0) col_recip = __shfl_sync(0xffffffffu, col_recip, comb_idx & (kMhc - 1));
    if (comb_lane) {
      logit = logit * col_recip;
    }
  }

  if (comb_lane) {
    comb[row * kMhc * kMhc + comb_idx] = logit;
  }
  if (lane >= kMhc && lane < 2 * kMhc) {
    post[row * kMhc + lane - kMhc] = 2.0f * sinkhorn_sigmoid(mix * hc_scale[1] + hc_base[lane]);
  }
}

// Two independent 16-lane token groups per warp. Every group owns all
// 16 combine coefficients; the first eight lanes also own pre/post.
__device__ void sinkhorn_half(const float* mix,
                              const float* sq,
                              const float* scale,
                              const float* base,
                              float*       out,
                              int          row,
                              float        rms_eps,
                              float        eps) {
  int   lane = threadIdx.x & 15;
  float cm = 0.f, pp = 0.f, ss = 0.f;
#pragma unroll
  for (int part = 0; part < MHC_KPARTS; ++part) {
    cm = __fadd_rn(cm, mix[(part * 24 + row) * 24 + 8 + lane]);
    if (lane < 8) pp = __fadd_rn(pp, mix[(part * 24 + row) * 24 + lane]);
    if (lane == 0) ss = __fadd_rn(ss, sq[(part * 24 + row) * 3]);
  }
  ss        = __shfl_sync(0xffffffffu, ss, 0, 16);
  float inv = rsqrtf(ss * (1.f / 20480.f) + rms_eps);
  if (lane < 4) out[row * 4 + lane] = sinkhorn_sigmoid(pp * inv * scale[0] + base[lane]) + eps;
  if (lane >= 4 && lane < 8) out[96 + row * 4 + lane - 4] = 2.f * sinkhorn_sigmoid(pp * inv * scale[1] + base[lane]);
  float z = __fmaf_rn(cm * inv, scale[2], base[8 + lane]);
  z       = __expf(z - sinkhorn_reduce_group4_max(z));
  z       = z * __frcp_rn(sinkhorn_reduce_group4_sum(z) + eps) + eps;
  z       = z * __frcp_rn(sinkhorn_reduce_column_sum(z) + eps);
#pragma unroll
  for (int iter = 1; iter < 20; ++iter) {
    z = z * __frcp_rn(sinkhorn_reduce_group4_sum(z) + eps);
    z = z * __frcp_rn(sinkhorn_reduce_column_sum(z) + eps);
  }
  out[192 + row * 16 + lane] = z;
}

template <int Threads, int CTAs>
__device__ void run(int                       local,
                    float*                    partial_mix,
                    float*                    partial_sq,
                    int*                      completion,
                    const MUtensorDescriptor& wd,
                    const MUtensorDescriptor& xd,
                    const float*              scale,
                    const float*              base,
                    float*                    output,
                    float                     rms_eps,
                    float                     hc_eps,
                    char*                     smem) {
  for (int part = local; part < MHC_KPARTS; part += CTAs) {
    project(partial_mix, partial_sq, wd, xd, part, smem);
    if constexpr (MHC_TACTIC >= 3) {
      __syncthreads();
      if (threadIdx.x == 0) __threadfence_system();
    } else {
      __threadfence_system();
    }
    __syncthreads();
    int* last = reinterpret_cast<int*>(smem + 65536);
    if (threadIdx.x == 0) *last = (atomicAdd(completion, 1) == MHC_KPARTS - 1);
    __syncthreads();
    if (*last) {
      asm volatile("" ::: "memory");
      // The publishing system fence and acquire-side cache invalidation are
      // kept separate from the dispatch protocol; this group never waits.
      flushInv_byp();
      if constexpr (MHC_TACTIC == 2) {
        for (int row = threadIdx.x / 16; row < 24; row += Threads / 16)
          sinkhorn_half(partial_mix, partial_sq, scale, base, output, row, rms_eps, hc_eps);
      } else {
        for (int row = threadIdx.x / 32; row < 24; row += Threads / 32)
          sinkhorn_row(partial_mix,
                       partial_sq,
                       scale,
                       base,
                       output,
                       output + 96,
                       output + 192,
                       24,
                       row,
                       3,
                       rms_eps,
                       hc_eps,
                       threadIdx.x & 31);
      }
      __syncthreads();
      if (threadIdx.x == 0) atomicExch(completion, 0);
    }
    __syncthreads();
  }
}
}  // namespace mate::mhc_tce_horizontal
