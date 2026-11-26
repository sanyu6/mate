#pragma once
#include "mate/mega_moe/decode/fp4_fp4/decode_common.hpp"
#include "mate/mega_moe/stage1_scheduler.hpp"

namespace mega_moe::decode::mp32::fp4_fp4 {
// Dispatch-only extraction: no weights, TME, MMA, activation or combine.
// The dedicated IPC workspace is not shared with an in-flight fused MoE.
template <class L, uint32_t MPs, bool LinePaddedScales>
struct DispatchOnly {
  static constexpr int R = L::kRanks, E = L::kExperts, G = L::kGroups, T = L::kTopK;
  static constexpr int H = L::kHidden, P = L::kPoolTokens;
  static constexpr int kBlockM = 32, kValuesPerByte = 2, kScaleBlock = 32;
  static constexpr int kWarpThreads = 32, kNumDispatchThreads = 128, kDispatchWarps = 4;
  static constexpr int kNumCTAs = MPs, kScalePoolTokens = LinePaddedScales ? 4 * P : P;
  // Only warp-striped receive counts are needed; no expert compute scheduler.
  struct Scheduler {
    layout::Workspace workspace;
    uint32_t          per_lane[(G + 31) / 32] = {};
    __device__ explicit Scheduler(layout::Workspace const& w) : workspace(w) {
    }
    __device__ void fetch_expert_recv_count() {
      MUTE_UNROLL
      for (int i = 0; i < (G + 31) / 32; ++i) {
        int      e     = i * 32 + threadIdx.x % 32;
        uint64_t value = 0;
        if (e < G) {
          do {
            value = ld_volatile_global(workspace.get_expert_recv_count_sum_ptr(e));
          } while (uint32_t(value >> 32) != MPs * R);
        }
        per_lane[i] = uint32_t(value);
      }
      __syncwarp();
    }
    __device__ uint32_t get_num_tokens(int e) const {
      uint32_t value = 0;
      MUTE_UNROLL
      for (int i = 0; i < (G + 31) / 32; ++i)
        if (e / 32 == i) value = per_lane[i];
      return __shfl_sync(0xffffffffu, value, e % 32);
    }
  };
  __device__ static void run(layout::SymBuffer<R> const& sym,
                             uint8_t*                    l1,
                             uint8_t*                    sf1,
                             uint4*                      metadata,
                             int*                        counts,
                             int*                        offsets,
                             int*                        status,
                             int                         tokens,
                             uint32_t*                   expert_counts) {
    int   tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    auto* base          = sym.template get_base_ptr<char*>();
    auto  workspace     = L::workspace(base);
    auto* ids           = L::template at<int64_t>(base, L::kRoutes);
    auto* x             = L::template at<uint8_t>(base, L::kX);
    auto* xs            = L::template at<uint8_t>(base, L::kXS);
    auto* route_weights = L::template at<float>(base, L::kWeights);
    for (int e = tid; e < E; e += 128) expert_counts[e] = 0;
    if (blockIdx.x == 0 && tid == 0) *status = 0;
    if (tid == 0) __threadfence_system();
    __syncthreads();
    auto sync          = []() { __syncthreads(); };
    bool invalid_route = false;
    // Count and allocate source slots, preserving MP31's packed count tags.
    auto for_each_route = [&](auto const& process) {
      for (int token = blockIdx.x * kDispatchWarps + warp; token < tokens; token += kNumCTAs * kDispatchWarps) {
        if (lane < T) {
          int64_t expert = __ldg(ids + uint64_t(token) * T + lane);
          if (expert >= 0 && expert < E)
            process(token * T + lane, uint32_t(expert));
          else if (expert != -1)
            invalid_route = true;
        }
        __syncwarp();
      }
    };
    for_each_route([&](int route, uint32_t e) { atomic_add_shared_u32(expert_counts + e, 1); });
    sync();
    // A CTA with neither a logical tag nor source rows has no contribution.
    // Keep the actual count loop branch-free for every participating CTA.
    if (blockIdx.x < MPs || blockIdx.x * kDispatchWarps < uint32_t(tokens)) {
      for (int e = tid; e < E; e += kNumDispatchThreads) {
        uint64_t count   = (blockIdx.x < MPs ? 1ull << 32 : 0ull) | expert_counts[e];
        expert_counts[e] = uint32_t(atomic_add_global_u64(
            reinterpret_cast<unsigned long long*>(workspace.get_expert_send_count_ptr(e)), count));
      }
    }
    sync();
    for_each_route([&](int route, uint32_t e) {
      uint32_t slot = atomic_add_shared_u32(expert_counts + e, 1);
      st_relaxed_sys_global(sym.map(workspace.get_src_token_topk_idx_ptr(e % G, sym.rank_idx, slot), e / G),
                            uint32_t(route));
    });
    __syncwarp();
    __threadfence_system_noflush();
    grid_sync<kNumCTAs, 0>(workspace, blockIdx.x, tid, sync);
    // CTA0's status initialization precedes this grid barrier. Deferring
    // invalid-ID publication avoids another CTA racing the initial zero.
    if (invalid_route) atomicExch(status, 1);
    if (blockIdx.x == 0) {
      for (int e = tid; e < E; e += kNumDispatchThreads) {
        auto count = ld_volatile_global(workspace.get_expert_send_count_ptr(e));
        st_relaxed_sys_global(sym.map(workspace.get_expert_recv_count_ptr(sym.rank_idx, e % G), e / G),
                              count & 0xffffffffull);
        atomic_add_global_u64(
            reinterpret_cast<unsigned long long*>(sym.map(workspace.get_expert_recv_count_sum_ptr(e % G), e / G)),
            count);
      }
    }

    mtlink_barrier<R, kNumCTAs, kNumDispatchThreads, 0, 1, true>(workspace, sym, blockIdx.x, tid, sync, true, true);

    Scheduler scheduler(workspace);
    scheduler.fetch_expert_recv_count();
    // Rank-major pull preserves MP31's compact expert pool and metadata.
    // Rank round-robin reordering can be evaluated after bring-up.
    uint32_t pool_base = 0;
    for (int e = 0; e < G; ++e) {
      uint32_t count = scheduler.get_num_tokens(e);
      // Stripe the compact pool across all dispatch warps. Restarting at
      // warp zero for every small expert leaves most resident CTAs idle.
      constexpr uint32_t kDispatchWorkers = kNumCTAs * kDispatchWarps;
      uint32_t           first_row =
          (blockIdx.x * kDispatchWarps + warp + kDispatchWorkers - pool_base % kDispatchWorkers) % kDispatchWorkers;
      for (uint32_t row = first_row; row < count; row += kDispatchWorkers) {
        uint32_t src_rank = 0, src_slot = row;
        for (; src_rank < R; ++src_rank) {
          uint32_t n = uint32_t(ld_volatile_global(workspace.get_expert_recv_count_ptr(src_rank, e)));
          if (src_slot < n) break;
          src_slot -= n;
        }
        uint32_t route = ld_volatile_global(workspace.get_src_token_topk_idx_ptr(e, src_rank, src_slot));
        uint32_t token = route / T, slot = route % T, dst_row = pool_base + row;
        auto*    remote_x  = sym.map(x + uint64_t(token) * (H / kValuesPerByte), src_rank);
        auto*    remote_sf = sym.map(xs + uint64_t(token) * (H / kScaleBlock), src_rank);
        for (int word = lane; word < H / kValuesPerByte / int(sizeof(int4)); word += kWarpThreads) {
          auto value = __lsu_ld_cache_hint(reinterpret_cast<int4 const*>(remote_x) + word, 3, 3, 0, 0);
          __lsu_st_cache_hint(
              reinterpret_cast<int4*>(l1 + uint64_t(dst_row) * (H / kValuesPerByte)) + word, value, 4, 2, 1, 1);
        }
        constexpr int kScalesPerWord = sizeof(uint32_t) / sizeof(uint8_t);
        for (int word = lane; word < H / kScaleBlock / kScalesPerWord; word += kWarpThreads) {
          // The invocation prologue invalidates prior peer input lines.
          uint32_t bits = __lsu_ld_cache_hint(reinterpret_cast<uint32_t const*>(remote_sf) + word, 3, 3, 0, 0);
          // Publish all scale bytes before the row-arrival release.
          MUTE_UNROLL
          for (int j = 0; j < kScalesPerWord; ++j)
            __lsu_st_cache_hint(sf1 + uint64_t(word * kScalesPerWord + j) * kScalePoolTokens +
                                    (LinePaddedScales ? (dst_row / 32) * 128 + dst_row % 32 : dst_row),
                                uint8_t(bits >> (j * 8)),
                                LinePaddedScales ? 0 : 4,
                                2,
                                1,
                                1);
        }
        if (lane == 0) {
          static_assert(layout::Workspace::kLogicBufferSlotBytes >= sizeof(uint4));
          __lsu_st_cache_hint(metadata + dst_row,
                              uint4{src_rank,
                                    token,
                                    slot,
                                    __float_as_uint(__lsu_ld_cache_hint(
                                        sym.map(route_weights + uint64_t(token) * T + slot, src_rank), 3, 3, 0, 0))},
                              4,
                              2,
                              1,
                              1);
        }
        // Drain all payload writers; no GEMM arrival counters are needed.
        __threadfence_system_noflush();
        __syncwarp();
      }
      if (blockIdx.x == 0 && tid == 0) {
        counts[e]  = count;
        offsets[e] = pool_base;
      }
      pool_base += math::align(count, uint32_t(kBlockM));
    }
    if (blockIdx.x == 0 && tid == 0) offsets[G] = pool_base;

    // Every CTA publishes its local output before retiring local counters.
    __syncthreads();
    if (tid == 0) __threadfence_system();
    __syncthreads();
    grid_sync<kNumCTAs, 0>(workspace, blockIdx.x, tid, sync);
    if (blockIdx.x == 0) {
      for (int e = tid; e < E; e += 128) st_relaxed_sys_global(workspace.get_expert_send_count_ptr(e), 0ull);
    }
    for (int e = blockIdx.x; e < G; e += kNumCTAs) {
      for (int rank = tid; rank < R; rank += 128)
        st_relaxed_sys_global(workspace.get_expert_recv_count_ptr(rank, e), 0ull);
      if (tid == 0) st_relaxed_sys_global(workspace.get_expert_recv_count_sum_ptr(e), 0ull);
    }
    // All peers have finished pulling inputs and retired their counters before
    // this rank can prepare the next invocation. Persistent phase words survive.
    mtlink_barrier<R, kNumCTAs, 128, 0, 2, true>(workspace, sym, blockIdx.x, tid, sync, true, true);
  }
};
template <class Stage>
__global__ __launch_bounds__(128) void dispatch_only_kernel(MUTLASS_GRID_CONSTANT layout::SymBuffer<Stage::R> const sym,
                                                            uint8_t* payload,
                                                            uint8_t* scales,
                                                            uint4*   metadata,
                                                            int*     counts,
                                                            int*     offsets,
                                                            int*     status,
                                                            int      tokens) {
  __shared__ uint32_t expert_counts[Stage::E];
  Stage::run(sym, payload, scales, metadata, counts, offsets, status, tokens, expert_counts);
}
}  // namespace mega_moe::decode::mp32::fp4_fp4
