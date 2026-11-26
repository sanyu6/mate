#pragma once
#include "mate/mega_moe/decode/fp4_fp4/decode_common.hpp"

namespace mega_moe::decode::mp32::fp4_fp4 {
// Decoder pull: all routes remain distinct. Only small routing metadata is
// read redundantly; FP4 payload is pulled solely by the owning expert rank.
template <class L,
          uint32_t CTAs,
          bool     LinePaddedScales,
          int      FlagWords       = 32,
          int      MaxRows         = 128,
          bool     Compact         = false,
          bool     PrecomputeSlots = false>
struct PullDispatchOnly {
  static_assert(!PrecomputeSlots || Compact);
  static_assert(!PrecomputeSlots || L::kRanks * MaxRows * L::kTopK < 65536);
  static constexpr int R = L::kRanks, E = L::kExperts, G = L::kGroups, T = L::kTopK;
  static constexpr int H = L::kHidden, P = L::kPoolTokens, kNumCTAs = CTAs;
  static constexpr int kScalePoolTokens = LinePaddedScales ? 4 * P : P;
  static constexpr int kControlStride   = (R + 1) * FlagWords;
  static_assert(uint64_t(CTAs) * kControlStride * 4 <= uint64_t(L::kTokens) * T * H * 2);
  struct SharedStorage {
    int      ids[R * MaxRows * T];
    int      owned_routes[Compact ? R * MaxRows * T : 1];
    uint32_t owned_count;
    uint32_t counts[G], offsets[G + 1], sequence, invalid, timeout, max_count;
  };
  __device__ static void release_store(uint32_t* ptr, uint32_t value) {
    auto* p = reinterpret_cast<uint32_t*>(__musa_ptr_gen_to_global(ptr));
    musa::atomic_ref<uint32_t, musa::thread_scope_system> ref(*p);
    ref.store(value, musa::memory_order_release);
  }
  template <class Sync>
  __device__ static bool exchange(
      layout::SymBuffer<R> const& sym, uint32_t* state, uint32_t expected, SharedStorage& shared, Sync const& sync) {
    int tid = threadIdx.x;
    if (tid < R) release_store(sym.map(state + (sym.rank_idx + 1) * FlagWords, tid), expected);
    sync();
    if (tid < R) {
      auto start = clock64();
      while (int32_t(load_acquire_global_u32(state + (tid + 1) * FlagWords) - expected) < 0) {
        if (clock64() - start > 4000000000ull) {
          atomicExch(&shared.timeout, 1u);
          break;
        }
      }
    }
    sync();
    return shared.timeout == 0;
  }
  template <class Sync, bool FenceInput = true>
  __device__ static void run(layout::SymBuffer<R> const& sym,
                             uint8_t*                    payload,
                             uint8_t*                    scales,
                             uint4*                      metadata,
                             int*                        counts,
                             int*                        offsets,
                             int*                        status,
                             int                         tokens,
                             SharedStorage&              shared,
                             Sync const&                 sync,
                             int4*                       tile_schedule  = nullptr,
                             int                         schedule_tiles = 0) {
    int   tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    auto* base       = sym.template get_base_ptr<char*>();
    auto* state      = L::template at<uint32_t>(base, L::kCombine) + blockIdx.x * kControlStride;
    auto* source_ids = L::template at<int64_t>(base, L::kRoutes);
    auto* source_x   = L::template at<uint8_t>(base, L::kX);
    auto* source_sf  = L::template at<uint8_t>(base, L::kXS);
    auto* source_w   = L::template at<float>(base, L::kWeights);
    if (tid == 0) {
      shared.invalid  = 0;
      shared.timeout  = 0;
      shared.sequence = load_acquire_global_u32(state) + 2;
      release_store(state, shared.sequence);
      if (blockIdx.x == 0) atomicExch(status, 0);
      if constexpr (FenceInput) __threadfence_system();
    }
    for (int e = tid; e < G; e += 128) shared.counts[e] = 0;
    sync();
    if (!exchange(sym, state, shared.sequence - 1, shared, sync)) {
      if (tid == 0) atomicMax(status, 2);
      return;
    }
    int routes = R * tokens * T;
    for (int route = tid; route < routes; route += 128) {
      int     source = route / (tokens * T), local = route % (tokens * T);
      int64_t id = __lsu_ld_cache_hint(sym.map(source_ids + local, source), 3, 3, 0, 0);
      if (id >= 0 && id < E) {
        shared.ids[route] = int(id);
        int expert        = int(id) - int(sym.rank_idx) * G;
        if (expert >= 0 && expert < G) atomic_add_shared_u32(shared.counts + expert, 1);
      } else {
        shared.ids[route] = -1;
        if (id != -1 && source == sym.rank_idx) atomicExch(&shared.invalid, 1u);
      }
    }
    sync();
    if (tid < 32) {
      uint32_t carry = 0, maximum = 0;
      for (int group = 0; group < G; group += 32) {
        int      expert = group + lane;
        uint32_t count  = expert < G ? shared.counts[expert] : 0;
        uint32_t value  = (count + 31) / 32 * 32;
        if (tile_schedule != nullptr && schedule_tiles > 0 && blockIdx.x == 0) maximum = max(maximum, count);
        uint32_t inclusive = value;
        MUTE_UNROLL
        for (int delta = 1; delta < 32; delta <<= 1) {
          uint32_t prior = __shfl_up_sync(0xffffffffu, inclusive, delta);
          if (lane >= delta) inclusive += prior;
        }
        if (expert < G) shared.offsets[expert] = carry + inclusive - value;
        carry += __shfl_sync(0xffffffffu, inclusive, 31);
      }
      if (lane == 0) shared.offsets[G] = carry;
      if (tile_schedule != nullptr && schedule_tiles > 0 && blockIdx.x == 0) {
        MUTE_UNROLL
        for (int delta = 16; delta; delta >>= 1) maximum = max(maximum, __shfl_down_sync(0xffffffffu, maximum, delta));
        if (lane == 0) shared.max_count = maximum;
      }
    }
    if (tid == 0 && blockIdx.x == 0) atomicMax(status, int(shared.invalid));
    sync();
    if (blockIdx.x == 0) {
      for (int e = tid; e < G; e += 128) __lsu_st_cache_hint(counts + e, int(shared.counts[e]), 4, 2, 1, 1);
      for (int e = tid; e <= G; e += 128) __lsu_st_cache_hint(offsets + e, int(shared.offsets[e]), 4, 2, 1, 1);
    }
    // Common case preserves the primary task ordering. If an expert exceeds
    // those slots, emit compact expert-major tasks from the existing M32
    // offsets. Writers own disjoint intervals, so no extra barrier or serial
    // scan of global task descriptors is needed. There is no expert row cap.
    if (tile_schedule != nullptr && schedule_tiles > 0 && blockIdx.x == 0) {
      const int capacity = G * schedule_tiles;
      if (shared.max_count <= uint32_t(schedule_tiles * 32)) {
        for (int task = tid; task < capacity; task += 128) {
          const int e = task / schedule_tiles, t = task % schedule_tiles;
          tile_schedule[task] =
              make_int4(e, shared.offsets[e] + t * 32, max(0, min(32, int(shared.counts[e]) - t * 32)), 0);
        }
      } else {
        const int live_tiles = shared.offsets[G] / 32;
        if (live_tiles > capacity) {
          if (tid == 0) atomicMax(status, 3);
        } else {
          for (int e = tid; e < G; e += 128) {
            for (int t = 0; t * 32 < int(shared.counts[e]); ++t) {
              tile_schedule[shared.offsets[e] / 32 + t] =
                  make_int4(e, shared.offsets[e] + t * 32, min(32, int(shared.counts[e]) - t * 32), 0);
            }
          }
          for (int task = live_tiles + tid; task < capacity; task += 128) tile_schedule[task] = make_int4(0, 0, 0, 0);
        }
      }
    }
    // Stable owner-local route queue: distribute actual copies, not global
    // indices containing many routes owned by other ranks. No deduplication.
    if constexpr (Compact) {
      if (tid == 0) {
        uint32_t total = 0;
        for (int e = 0; e < G; ++e) total += shared.counts[e];
        shared.owned_count = total;
      }
      sync();
      if constexpr (PrecomputeSlots) {
        for (int e = tid; e < G; e += 128) shared.counts[e] = 0;
        sync();
      }
      if (shared.owned_count > 0 && (PrecomputeSlots || shared.owned_count != uint32_t(routes)) && tid < 32) {
        uint32_t prefix = 0;
        for (int start = 0; start < routes; start += 32) {
          int      idx = start + lane, id = idx < routes ? shared.ids[idx] : -1;
          bool     mine    = id >= int(sym.rank_idx) * G && id < int(sym.rank_idx + 1) * G;
          uint32_t mask    = __ballot_sync(0xffffffffu, mine);
          uint32_t encoded = uint32_t(idx);
          if constexpr (PrecomputeSlots) {
            uint32_t peers  = __match_any_sync(0xffffffffu, uint32_t(id)) & mask;
            int      leader = peers ? __ffs(peers) - 1 : 0;
            uint32_t prior  = 0;
            if (mine && lane == leader) prior = atomicAdd(shared.counts + id - int(sym.rank_idx) * G, __popc(peers));
            prior         = __shfl_sync(0xffffffffu, prior, leader);
            uint32_t slot = prior + __popc(peers & ((1u << lane) - 1u));
            encoded |= slot << 16;
          }
          if (mine) shared.owned_routes[prefix + __popc(mask & ((1u << lane) - 1u))] = int(encoded);
          prefix += __popc(mask);
        }
      }
      sync();
    }
    int work = Compact ? int(shared.owned_count) : routes;
    for (int ordinal = blockIdx.x * 4 + warp; ordinal < work; ordinal += CTAs * 4) {
      int route = ordinal;
      if constexpr (Compact) {
        if (PrecomputeSlots || shared.owned_count != uint32_t(routes)) route = shared.owned_routes[ordinal] & 0xffff;
      }

      int expert = shared.ids[route], e = expert - int(sym.rank_idx) * G;
      if (expert < 0 || e < 0 || e >= G) continue;
      uint32_t slot = 0;
      if constexpr (PrecomputeSlots) {
        slot = uint32_t(shared.owned_routes[ordinal]) >> 16;
      } else {
        for (int earlier = lane; earlier < route; earlier += 32) slot += shared.ids[earlier] == expert;
        MUTE_UNROLL
        for (int delta = 16; delta; delta >>= 1) slot += __shfl_down_sync(0xffffffffu, slot, delta);
        slot = __shfl_sync(0xffffffffu, slot, 0);
      }
      uint32_t row    = shared.offsets[e] + slot;
      int      source = route / (tokens * T), local = route % (tokens * T), token = local / T, topk = local % T;
      auto*    x  = sym.map(source_x + uint64_t(token) * (H / 2), source);
      auto*    sf = sym.map(source_sf + uint64_t(token) * (H / 32), source);
      for (int word = lane; word < H / 2 / 16; word += 32) {
        auto value = __lsu_ld_cache_hint(reinterpret_cast<int4 const*>(x) + word, 3, 3, 0, 0);
        __lsu_st_cache_hint(reinterpret_cast<int4*>(payload + uint64_t(row) * (H / 2)) + word, value, 4, 2, 1, 1);
      }
      int scale_row = LinePaddedScales ? (row / 32) * 128 + row % 32 : row;
      for (int word = lane; word < H / 32 / 4; word += 32) {
        uint32_t bits = __lsu_ld_cache_hint(reinterpret_cast<uint32_t const*>(sf) + word, 3, 3, 0, 0);
        MUTE_UNROLL
        for (int j = 0; j < 4; ++j)
          __lsu_st_cache_hint(
              scales + uint64_t(word * 4 + j) * kScalePoolTokens + scale_row, uint8_t(bits >> (j * 8)), 4, 2, 1, 1);
      }
      if (lane == 0) {
        auto weight = __lsu_ld_cache_hint(sym.map(source_w + local, source), 3, 3, 0, 0);
        __lsu_st_cache_hint(metadata + row,
                            uint4{uint32_t(source), uint32_t(token), uint32_t(topk), __float_as_uint(weight)},
                            4,
                            2,
                            1,
                            1);
      }
    }
    sync();
    if (tid == 0) __threadfence_system();
    sync();
    // Per-CTA done signals are released after its reads and output publication.
    // A rank returns only when every local CTA has observed every peer's done.
    if (!exchange(sym, state, shared.sequence, shared, sync) && tid == 0) atomicMax(status, 2);
  }
};
template <class Stage>
__global__ __launch_bounds__(128) void dispatch_pull_kernel(MUTLASS_GRID_CONSTANT layout::SymBuffer<Stage::R> const sym,
                                                            uint8_t* payload,
                                                            uint8_t* scales,
                                                            uint4*   metadata,
                                                            int*     counts,
                                                            int*     offsets,
                                                            int*     status,
                                                            int      tokens) {
  __shared__ typename Stage::SharedStorage shared;
  auto                                     sync = []() { __syncthreads(); };
  Stage::run(sym, payload, scales, metadata, counts, offsets, status, tokens, shared, sync);
}
}  // namespace mega_moe::decode::mp32::fp4_fp4
