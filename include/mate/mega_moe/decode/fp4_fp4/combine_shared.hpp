#pragma once
#ifndef COMBINE_RING2
#define COMBINE_RING2 0
#endif
#include <cooperative_groups.h>

#include "mate/mega_moe/decode/fp4_fp4/combine_mhc_tail.hpp"
#include "mate/mega_moe/decode/fp4_fp4/dispatch_shared.hpp"
namespace mega_moe::decode::mp32::fp4_fp4 {
// Dedicated symmetric buffer: control prefix and one partial publication plane.
// Each CTA owns the same hidden slices on every rank. Ready/done handshake
// protects publication and reuse without depending on another local CTA.
template <class L, int CTAs, int FeatureTile = 128, int CommThreads = 128, int Stages = 2>
struct CombineShared {
  using Gemm             = SharedFp8Gemm<2304, 5120, FeatureTile, 32, Stages>;
  using Pipeline         = typename Gemm::Pipeline;
  using Barrier          = typename Gemm::Barrier;
  using Params           = typename Gemm::Params;
  static constexpr int R = L::kRanks, Threads = CommThreads + 128 + Gemm::NumMmaThreads;
  struct Shared {
    typename Gemm::Mainloop::SharedStorage gemm;
    uint32_t                               seq, timeout;
    int                                    slots[R * 24 * 6];
    float                                  weights[R * 24 * 6];
  };
  struct Bars {
    uint8_t pipeline[Pipeline::NumBarriers];
    uint8_t comm[2];
  };
  static constexpr int ControlWords = (R + 1) * 32, PayloadOffset = 65536;
  template <class Sync>
  __device__ static void exchange(
      layout::SymBuffer<R> const& sym, uint32_t* state, uint32_t epoch, Shared& sh, Sync sync) {
    int tid = threadIdx.x;
    if (tid < R) PullDispatchOnly<L, CTAs, true>::release_store(sym.map(state + (sym.rank_idx + 1) * 32, tid), epoch);
    sync();
    if (tid < R) {
      auto start = clock64();
      while (int32_t(load_acquire_global_u32(state + (tid + 1) * 32) - epoch) < 0) {
        if (clock64() - start > 4000000000ull) {
          atomicExch(&sh.timeout, 1u);
          break;
        }
      }
    }
    sync();
  }
  template <bool DoCombine = true, bool DoShared = true>
  __device__ static void run(layout::SymBuffer<R> const& sym,
                             Params const&               params,
                             const mutlass::bfloat16_t*  routed,
                             const uint4*                metadata,
                             const int*                  counts,
                             const int*                  offsets,
                             bool                        weighted,
                             float                       route_scale,
                             mutlass::bfloat16_t*        shared,
                             mutlass::bfloat16_t*        reduced,
                             int*                        status,
                             char*                       smem) {
    auto& sh   = *reinterpret_cast<Shared*>(smem);
    auto* bars = reinterpret_cast<Bars*>(0);
    mutlass::arch::allocate_async_barriers(sizeof(Bars) + Barrier::ReservedAsyncBarrierCount);
    typename Pipeline::Params pp;
    pp.transaction_bytes = Gemm::Mainloop::TmeTransactionBytes;
    pp.num_consumers     = Gemm::NumMmaWarps;
    pp.num_producers     = 1;
    Pipeline pipe(pp, reinterpret_cast<uint64_t>(&bars->pipeline));
    auto     id  = [](auto* p) { return uint32_t(reinterpret_cast<uint64_t>(p)) + Barrier::ReservedAsyncBarrierCount; };
    uint32_t bar = id(&bars->comm);
    auto*    base    = sym.template get_base_ptr<char*>();
    auto*    control = L::template at<uint32_t>(base, L::kCombine) + blockIdx.x * ControlWords;
    auto*    partial = reinterpret_cast<mutlass::bfloat16_t*>(L::template at<char>(base, L::kCombine) + PayloadOffset);
    if (threadIdx.x == 0) {
      Barrier::init(bar, CommThreads / 32, 0);
      Barrier::init(bar + 1, CommThreads / 32, 0);
      sh.timeout = 0;
      sh.seq     = load_acquire_global_u32(control) + (DoCombine ? 2 : 0);
      PullDispatchOnly<L, CTAs, true>::release_store(control, sh.seq);
      if (DoCombine && blockIdx.x == 0) *status = 0;
    }
    __syncthreads();
    if constexpr (COMBINE_RING2) partial += ((sh.seq / 2) & 1u) * (R * 24 * 5120);
    int tid = threadIdx.x;
    if (DoCombine && tid < CommThreads) {
      uint32_t phase = 0;
      auto     sync  = [&]() {
        __threadfence_block();
        auto bb = bar + (phase++ & 1u);
        auto p  = Barrier::template arrive<true>(bb);
        Barrier::wait(bb, p);
      };
      for (int route = tid; route < R * 24 * 6; route += CommThreads) {
        sh.slots[route]   = -1;
        sh.weights[route] = 0.f;
      }
      sync();
      for (int e = tid / 32; e < L::kGroups; e += CommThreads / 32) {
        for (int j = tid % 32; j < counts[e]; j += 32) {
          int   slot = offsets[e] + j;
          uint4 m    = metadata[slot];
          if (m.x < R && m.y < 24 && m.z < 6) {
            int route         = (m.x * 24 + m.y) * 6 + m.z;
            sh.slots[route]   = slot;
            sh.weights[route] = weighted ? __uint_as_float(m.w) : 1.f;
          }
        }
      }
      sync();
      for (int tile = blockIdx.x; tile < 5120 / FeatureTile; tile += CTAs) {
        for (int v = tid; v < R * 24 * (FeatureTile / 8); v += CommThreads) {
          int   row = v / (FeatureTile / 8), h = tile * FeatureTile + (v % (FeatureTile / 8)) * 8;
          float acc[8] = {};
#pragma unroll
          for (int k = 0; k < 6; k++) {
            int slot = sh.slots[row * 6 + k];
            if (slot >= 0) {
              uint4 packet = *reinterpret_cast<const uint4*>(routed + uint64_t(slot) * 5120 + h);
              auto* vals   = reinterpret_cast<const mutlass::bfloat16_t*>(&packet);
#pragma unroll
              for (int j = 0; j < 8; j++) acc[j] = __fmaf_rn(float(vals[j]), sh.weights[row * 6 + k], acc[j]);
            }
          }
          uint4 packet;
          auto* vals = reinterpret_cast<mutlass::bfloat16_t*>(&packet);
#pragma unroll
          for (int j = 0; j < 8; j++) vals[j] = mutlass::bfloat16_t(__fmul_rn(acc[j], route_scale));
          *reinterpret_cast<uint4*>(partial + row * 5120 + h) = packet;
        }
      }
      __threadfence_system_noflush();
      sync();
      if (tid == 0) __threadfence_system();
      sync();
      exchange(sym, control, sh.seq - 1, sh, sync);
      if (tid == 0) flushInv_byp();
      __threadfence_system_noflush();
      sync();
      asm volatile("" ::: "memory");
      if (!sh.timeout) {
        for (int tile = blockIdx.x; tile < 5120 / FeatureTile; tile += CTAs) {
          for (int v = tid; v < 24 * (FeatureTile / 8); v += CommThreads) {
            int   row = v / (FeatureTile / 8), h = tile * FeatureTile + (v % (FeatureTile / 8)) * 8;
            int   idx    = (int(sym.rank_idx) * 24 + row) * 5120 + h;
            uint4 packet = *reinterpret_cast<const uint4*>(sym.map(partial + idx, 0));
            float acc[8];
            auto* vals = reinterpret_cast<mutlass::bfloat16_t*>(&packet);
#pragma unroll
            for (int j = 0; j < 8; j++) acc[j] = float(vals[j]);
#pragma unroll
            for (int peer = 1; peer < R; peer++) {
              packet = *reinterpret_cast<const uint4*>(sym.map(partial + idx, peer));
#pragma unroll
              for (int j = 0; j < 8; j++) acc[j] = __fadd_rn(acc[j], float(vals[j]));
            }
#pragma unroll
            for (int j = 0; j < 8; j++) vals[j] = mutlass::bfloat16_t(acc[j]);
            *reinterpret_cast<uint4*>(reduced + row * 5120 + h) = packet;
          }
        }
      }

      __threadfence_system_noflush();
      if constexpr (!COMBINE_RING2) {
        sync();
        if (tid == 0) __threadfence_system();
        sync();
        exchange(sym, control, sh.seq, sh, sync);
      }
      if (sh.timeout && tid == 0) atomicMax(status, 1);
    }
    if (DoShared && tid >= CommThreads && tid < CommThreads + 32) {
      auto inputs = typename Gemm::Mainloop{}.load_init(make_shape(5120, 24, 2304, 1), params);
      auto write  = mutlass::make_producer_start_state<Pipeline>();
      for (int tile = blockIdx.x; tile < 5120 / FeatureTile; tile += CTAs)
        Gemm::load(inputs, tile, 0, reinterpret_cast<char*>(&sh.gemm), params, pipe, write);
    } else if (DoShared && tid >= CommThreads + 128 && tid < Threads) {
      auto                 inputs = typename Gemm::Mainloop{}.load_init(make_shape(5120, 24, 2304, 1), params);
      typename Gemm::State read;
      for (int tile = blockIdx.x; tile < 5120 / FeatureTile; tile += CTAs)
        Gemm::compute(
            inputs, tile, 0, 24, reinterpret_cast<char*>(&sh.gemm), pipe, read, tid - (CommThreads + 128), shared);
    }
  }
};
}  // namespace mega_moe::decode::mp32::fp4_fp4
