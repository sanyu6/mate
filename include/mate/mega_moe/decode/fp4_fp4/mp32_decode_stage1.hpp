#pragma once

#include <musa_fp4.h>

#include "mate/mega_moe/decode/fp4_fp4/decode_sglang_epilogue.hpp"
#include "mate/mega_moe/decode/fp4_fp4/decode_swap_ab.hpp"
#include "mate/mega_moe/mp31_activation.hpp"

namespace mega_moe::decode::mp32::fp4_fp4 {

// MP31 decode pattern: dispatch squad, TME producer squad, then MMA squads.
// Dispatch publishes M32 arrivals while FC1 consumes already-arrived tiles.
template <class L,
          uint32_t MPs,
          uint32_t CTAsPerMP        = 1,
          bool     kUseSitu         = false,
          bool     kUseSituLinear   = false,
          bool     SwapAB           = false,
          int      FeatureTile      = 256,
          bool     PackB            = false,
          bool     SglangNumerics   = false,
          bool     LinePaddedScales = false,

          bool DoubleBuffered = false>
struct Stage1
    : DecodeCompute<DoubleBuffered ? 7 : (LinePaddedScales ? (PackB ? 6 : 3) : (SwapAB ? 1 : (PackB ? 2 : 0))),
                    FeatureTile> {
  using Compute =
      DecodeCompute<DoubleBuffered ? 7 : (LinePaddedScales ? (PackB ? 6 : 3) : (SwapAB ? 1 : (PackB ? 2 : 0))),
                    FeatureTile>;
  using typename Compute::Barrier;
  using typename Compute::Mainloop;
  using typename Compute::Pipeline;
  using typename Compute::State;
  using typename Compute::Tile;
  static constexpr int kBlockM = Compute::kBlockM, kBlockN = Compute::kBlockN, kBlockK = Compute::kBlockK;
  static constexpr int kScaleBlock = Compute::kScaleBlock, kValuesPerByte = Compute::kValuesPerByte;
  static constexpr int kWarpThreads = Compute::kWarpThreads, kSquadThreads = Compute::kSquadThreads;
  static constexpr int kNumDispatchThreads = Compute::kNumDispatchThreads;
  static constexpr int kNumMmaWarps = Compute::kNumMmaWarps, kNumMmaThreads = Compute::kNumMmaThreads;
  static constexpr int kAlignment      = Compute::kAlignment;
  static constexpr int kFirstMmaThread = Compute::kFirstMmaThread, kNumThreads = Compute::kNumThreads;

  static constexpr int      R = L::kRanks, E = L::kExperts, G = L::kGroups, T = L::kTopK;
  static constexpr int      H = L::kHidden, I = L::kIntermediate, P = L::kPoolTokens;
  static constexpr int      kScalePoolTokens = LinePaddedScales ? 4 * P : P;
  static constexpr int      kDispatchWarps   = kNumDispatchThreads / kWarpThreads;
  static constexpr int      kComputeWarps    = 1 + kNumMmaWarps;
  static constexpr uint32_t kNumCTAs         = MPs * CTAsPerMP;
  static constexpr int      kMinBlocksPerMp  = DoubleBuffered ? CTAsPerMP : 1;
  static_assert(CTAsPerMP >= 1 && CTAsPerMP <= 3);
  using Pool = std::conditional_t<
      DoubleBuffered,
      std::conditional_t<
          PackB,
          std::conditional_t<LinePaddedScales,
                             LinePaddedPackedDecodePool<2 * I, H, Compute>,
                             PackedDecodePool<2 * I, H, Compute>>,
          std::conditional_t<LinePaddedScales, LinePaddedDecodePool<2 * I, H, Compute>, DecodePool<2 * I, H, Compute>>>,
      typename Compute::template Pool<2 * I, H>>;
  using Params        = typename Pool::Params;
  using BaseScheduler = stage1_sched::MegaMoEL1Scheduler<kBlockM, kBlockN, kBlockK, 2 * I, H, G, 1, MPs, R>;
  // Count tags retain the logical MP contract shared with Stage2; only the
  // resident work stride changes when two CTAs fit on each physical MP.
  static constexpr uint32_t kNBlocksPerChunk = SwapAB ? 512 / kBlockN : 2;
  static_assert((2 * I / kBlockN) % kNBlocksPerChunk == 0);
  struct Scheduler : BaseScheduler {
    MUTLASS_DEVICE explicit Scheduler(layout::Workspace const& workspace) : BaseScheduler(workspace) {
      this->block_idx *= kNBlocksPerChunk;
    }
    MUTLASS_DEVICE bool fetch_next_l1_tile_desc(stage1_sched::L1TileDesc& desc) {
      while (this->current_local_expert_idx < G) {
        if (this->fetch_next_l1_block()) {
          desc = stage1_sched::L1TileDesc::make(this->current_local_expert_idx,
                                                this->get_current_pool_block_offset() + this->m_block_idx,
                                                this->n_block_idx,
                                                this->get_valid_m());
          // Consecutive N tiles reuse the same immutable dispatched M block.
          this->block_idx += this->n_block_idx % kNBlocksPerChunk == kNBlocksPerChunk - 1
                                 ? kNumCTAs * kNBlocksPerChunk - (kNBlocksPerChunk - 1)
                                 : 1;
          return true;
        }
      }
      desc = stage1_sched::L1TileDesc::invalid();
      return false;
    }
  };
  struct BarrierStorage {
    uint8_t pipeline[Pipeline::NumBarriers];
    uint8_t dispatch[2], descriptor[1], consumers[1];
  };
  struct DefaultStorage {
    typename Mainloop::SharedStorage mainloop;
    alignas(kAlignment) uint32_t expert_counts[E];
    uint32_t descriptor[stage1_sched::L1TileDesc::kNumWords];
  };
  struct SwapStorage : DefaultStorage {
    alignas(16) mutlass::bfloat16_t epilogue[kBlockM * kBlockN / 2];
  };
  using SharedStorage           = std::conditional_t<SwapAB, SwapStorage, DefaultStorage>;
  static constexpr int SmemSize = sizeof(SharedStorage);
  static_assert(MPs >= 2 && MPs % 2 == 0);
  static_assert(SmemSize <= 256 * 1024);
  static_assert(sizeof(BarrierStorage) + Barrier::ReservedAsyncBarrierCount <=
                Barrier::HardwareMaxNumAsyncTransactionBarriers);

  MUTLASS_DEVICE static void run(layout::SymBuffer<R> const& sym,
                                 Params const&               params,
                                 uint8_t*                    l1,
                                 uint8_t*                    sf1,
                                 uint8_t*                    l2,
                                 uint8_t*                    sf2,
                                 int*                        status,
                                 int                         tokens,
                                 float                       clamp,
                                 char*                       smem,
                                 float                       activation_beta        = 1.f,
                                 float                       activation_linear_beta = 1.f) {
    auto& shared   = *reinterpret_cast<SharedStorage*>(smem);
    auto* barriers = reinterpret_cast<BarrierStorage*>(0);
    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage) + Barrier::ReservedAsyncBarrierCount);
    typename Pipeline::Params pp;
    pp.transaction_bytes = Mainloop::TmeTransactionBytes;
    pp.num_consumers     = kNumMmaWarps;
    pp.num_producers     = 1;
    Pipeline pipeline(pp, reinterpret_cast<uint64_t>(&barriers->pipeline));
    auto     barrier_id = [](auto* ptr) {
      return uint32_t(reinterpret_cast<uint64_t>(ptr)) + Barrier::ReservedAsyncBarrierCount;
    };
    uint32_t dispatch_bar = barrier_id(&barriers->dispatch);
    uint32_t desc_bar     = barrier_id(&barriers->descriptor);
    uint32_t mma_bar      = barrier_id(&barriers->consumers);
    int      tid = threadIdx.x, warp = tid / kWarpThreads, lane = tid % kWarpThreads;
    if (tid == 0) {
      for (int i = 0; i < 2; ++i) Barrier::init(dispatch_bar + i, kDispatchWarps, 0);
      Barrier::init(desc_bar, kComputeWarps, 0);
      Barrier::init(mma_bar, kNumMmaWarps, 0);
    }
    for (int e = tid; e < E; e += kNumThreads) shared.expert_counts[e] = 0;
    if (blockIdx.x == 0 && tid == 0) *status = 0;
    // Publish source input cache lines before peers can pull this invocation.
    if (tid == 0) __threadfence_system();
    __syncthreads();
    auto* base          = sym.template get_base_ptr<char*>();
    auto  workspace     = L::workspace(base);
    auto* ids           = L::template at<int64_t>(base, L::kRoutes);
    auto* x             = L::template at<uint8_t>(base, L::kX);
    auto* xs            = L::template at<uint8_t>(base, L::kXS);
    auto* route_weights = L::template at<float>(base, L::kWeights);

    if (tid < kNumDispatchThreads) {
      uint32_t dispatch_phase = 0;
      auto     sync           = [&]() {
        auto b     = dispatch_bar + (dispatch_phase++ & 1u);
        auto phase = Barrier::template arrive<true>(b);
        Barrier::wait(b, phase);
      };

      // Count and allocate source slots, preserving MP31's packed count tags.
      auto for_each_route = [&](auto const& process) {
        for (int token = blockIdx.x * kDispatchWarps + warp; token < tokens; token += kNumCTAs * kDispatchWarps) {
          if (lane < T) {
            int64_t expert = __ldg(ids + uint64_t(token) * T + lane);
            if (expert >= 0 && expert < E)
              process(token * T + lane, uint32_t(expert));
            else if (expert != -1)
              atomicExch(status, 1);
          }
          __syncwarp();
        }
      };
      for_each_route([&](int route, uint32_t e) { atomic_add_shared_u32(shared.expert_counts + e, 1); });
      sync();
      // A CTA with neither a logical tag nor source rows has no contribution.
      // Keep the actual count loop branch-free for every participating CTA.
      if (blockIdx.x < MPs || blockIdx.x * kDispatchWarps < uint32_t(tokens)) {
        for (int e = tid; e < E; e += kNumDispatchThreads) {
          uint64_t count          = (blockIdx.x < MPs ? 1ull << 32 : 0ull) | shared.expert_counts[e];
          shared.expert_counts[e] = uint32_t(atomic_add_global_u64(
              reinterpret_cast<unsigned long long*>(workspace.get_expert_send_count_ptr(e)), count));
        }
      }
      sync();
      for_each_route([&](int route, uint32_t e) {
        uint32_t slot = atomic_add_shared_u32(shared.expert_counts + e, 1);
        st_relaxed_sys_global(sym.map(workspace.get_src_token_topk_idx_ptr(e % G, sym.rank_idx, slot), e / G),
                              uint32_t(route));
      });
      __syncwarp();
      __threadfence_system_noflush();
      grid_sync<kNumCTAs, 0>(workspace, blockIdx.x, tid, sync);
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
            __lsu_st_cache_hint(
                reinterpret_cast<uint4*>(workspace.get_token_src_metadata_ptr(dst_row)),
                uint4{src_rank,
                      token,
                      slot,
                      SglangNumerics ? __float_as_uint(__lsu_ld_cache_hint(
                                           sym.map(route_weights + uint64_t(token) * T + slot, src_rank), 3, 3, 0, 0))
                                     : 0u},
                4,
                2,
                1,
                1);
          }
          // Drain every payload writer before the leader publishes arrival.
          __threadfence_system_noflush();
          __syncwarp();
          if (lane == 0) atomic_add_release_global_u32(workspace.get_l1_arrival_count_ptr(dst_row / kBlockM), 1);
        }
        pool_base += math::align(count, uint32_t(kBlockM));
      }
    } else if (warp == kNumDispatchThreads / kWarpThreads || tid >= kFirstMmaThread) {
      auto descriptor_sync = [&]() {
        auto phase = Barrier::template arrive<true>(desc_bar);
        Barrier::wait(desc_bar, phase);
      };
      Mainloop mainloop;
      auto     inputs = Pool::init(P, G, params);
      if (tid < kFirstMmaThread) {
        auto      write = mutlass::make_producer_start_state<Pipeline>();
        Scheduler scheduler(workspace);
        scheduler.init_l1_dispatch_count_scheduler();
        while (true) {
          stage1_sched::L1TileDesc desc;
          scheduler.fetch_next_l1_tile_desc(desc);
          if (lane == 0) {
            if (desc.is_valid())
              while (load_acquire_global_u32(workspace.get_l1_arrival_count_ptr(desc.pool_block_idx)) != desc.valid_m)
                __nanosleep(20);
            desc.store_to_smem(shared.descriptor);
          }
          __syncwarp();
          descriptor_sync();
          descriptor_sync();  // All consumers have copied the descriptor.
          if (!desc.is_valid()) break;
          Pool::load(inputs,
                     make_coord(desc.pool_block_idx, desc.n_block_idx, desc.local_expert_idx),
                     reinterpret_cast<char*>(&shared.mainloop),
                     params,
                     pipeline,
                     write);
        }
      } else {
        int                         mma_tid = tid - kFirstMmaThread;
        typename Mainloop::TiledMma mma;
        State                       read;
        auto                        consumer_sync = [&]() {
          auto phase = Barrier::template arrive<true>(mma_bar);
          Barrier::wait(mma_bar, phase);
        };
        while (true) {
          descriptor_sync();
          auto desc = stage1_sched::L1TileDesc::load_from_smem(shared.descriptor);
          descriptor_sync();
          if (!desc.is_valid()) break;
          // Native SS reads scales before its TME wait, so every consumer
          // warp must acquire the dispatch release before calling mma().
          if (lane == 0)
            while (load_acquire_global_u32(workspace.get_l1_arrival_count_ptr(desc.pool_block_idx)) != desc.valid_m)
              __nanosleep(20);
          __syncwarp();
          // Acquire orders publication, but does not invalidate cached SF
          // lines. Refresh them before native SS eagerly loads its scales.
          // All MMA warps use this CTA's cache. One leader refreshes it,
          // then the consumer barrier holds every warp until CFI/IDF completes.
          // The first feature tile in a chunk refreshes SF. All following
          // tiles reuse the identical immutable published M32 block.
          if constexpr (!LinePaddedScales) {
            if (mma_tid == 0 && desc.n_block_idx % kNBlocksPerChunk == 0) __threadfence_system();
          }
          consumer_sync();
          auto accum = partition_fragment_C(mma, take<0, 2>(Tile{}));
          fill(accum, 0.f);
          mainloop.mma(Pool::mma_inputs(inputs, desc.local_expert_idx),
                       Pool::coord(desc.pool_block_idx, desc.n_block_idx),
                       accum,
                       H / kBlockK,
                       mma_tid,
                       reinterpret_cast<char*>(&shared.mainloop),
                       pipeline,
                       read);
          if constexpr (SglangNumerics) {
            static_assert(!SwapAB && !kUseSitu);
            sglang_activation<Compute>(accum, mma, mma_tid, desc, workspace, l2, sf2, P, I, clamp);
          } else if constexpr (SwapAB) {
            static_assert(!kUseSitu, "swap-AB currently implements SwiGLU only");
            swap_activation<Compute>(
                accum, mma, mma_tid, desc, shared.epilogue, l2, sf2, status, P, I, clamp, consumer_sync);
          } else {
            // Online-swizzled N gives each lane eight adjacent gate values
            // and their eight up values. Four neighboring lanes share a K32
            // quantization group; no shared activation round trip is needed.
            constexpr int      kValuesPerLane     = fp4::kStage2PermutationStride;
            constexpr int      kLanesPerScale     = kScaleBlock / kValuesPerLane;
            constexpr int      kAccumulatorGroup  = 2 * kValuesPerLane;
            constexpr int      kGroupsPerThread   = decltype(size(accum))::value / kAccumulatorGroup;
            constexpr int      kBytesPerLane      = kValuesPerLane / kValuesPerByte;
            constexpr uint32_t kBf16MagnitudeMask = 0x7fffu, kBf16Infinity = 0x7f80u;
            static_assert(kLanesPerScale == 4 && kBytesPerLane == sizeof(uint32_t));
            auto     coords      = mma.get_thread_slice(mma_tid).partition_C(make_identity_tensor(take<0, 2>(Tile{})));
            int      row         = int(get<0>(coords(0)));
            uint32_t valid_lanes = __ballot_sync(0xffffffffu, row < int(desc.valid_m));
            if (row < int(desc.valid_m)) {
              MUTE_UNROLL
              for (int group = 0; group < kGroupsPerThread; ++group) {
                mutlass::bfloat16_t values[kValuesPerLane];
                uint32_t            amax_bits = 0;
                if constexpr (kUseSitu) {
                  // Match MP31 and prefill: sigmoid uses the original BF16 gate;
                  // softcaps run in FP32, followed by one final BF16 round.
                  constexpr int kActivationVector = 4;
                  static_assert(kValuesPerLane % kActivationVector == 0);
                  MUTE_UNROLL
                  for (int col = 0; col < kValuesPerLane; col += kActivationVector) {
                    int          index   = group * kAccumulatorGroup + col;
                    float4       gate    = make_float4(float(mutlass::bfloat16_t(accum(index))),
                                              float(mutlass::bfloat16_t(accum(index + 1))),
                                              float(mutlass::bfloat16_t(accum(index + 2))),
                                              float(mutlass::bfloat16_t(accum(index + 3))));
                    float4       up      = make_float4(float(mutlass::bfloat16_t(accum(index + kValuesPerLane))),
                                            float(mutlass::bfloat16_t(accum(index + kValuesPerLane + 1))),
                                            float(mutlass::bfloat16_t(accum(index + kValuesPerLane + 2))),
                                            float(mutlass::bfloat16_t(accum(index + kValuesPerLane + 3))));
                    const float4 sigmoid = ::mega_moe::mp31::activation_sigmoid_bst4(gate);
                    gate                 = mul(activation_beta,
                               ::mega_moe::mp31::activation_tanh_bst4(mul(__frcp_rn(activation_beta), gate)));
                    gate                 = mul(gate, sigmoid);
                    if constexpr (kUseSituLinear)
                      up = mul(activation_linear_beta,
                               ::mega_moe::mp31::activation_tanh_bst4(mul(__frcp_rn(activation_linear_beta), up)));
                    float4 result   = mul(up, gate);
                    values[col]     = mutlass::bfloat16_t(result.x);
                    values[col + 1] = mutlass::bfloat16_t(result.y);
                    values[col + 2] = mutlass::bfloat16_t(result.z);
                    values[col + 3] = mutlass::bfloat16_t(result.w);
                  }
                }
                if constexpr (!kUseSitu) {
                  constexpr int kActivationVector = sizeof(fp4::ActivationFloat4) / sizeof(float);
                  MUTE_UNROLL
                  for (int col = 0; col < kValuesPerLane; col += kActivationVector) {
                    int                   index = group * kAccumulatorGroup + col;
                    fp4::ActivationFloat4 gate{accum(index), accum(index + 1), accum(index + 2), accum(index + 3)};
                    fp4::ActivationFloat4 up{accum(index + kValuesPerLane),
                                             accum(index + kValuesPerLane + 1),
                                             accum(index + kValuesPerLane + 2),
                                             accum(index + kValuesPerLane + 3)};
                    gate = __builtin_convertvector(__builtin_convertvector(gate, fp4::ActivationBf16x4),
                                                   fp4::ActivationFloat4);
                    up   = __builtin_convertvector(__builtin_convertvector(up, fp4::ActivationBf16x4),
                                                 fp4::ActivationFloat4);
                    MUTE_UNROLL
                    for (int j = 0; j < kActivationVector; ++j) {
                      gate[j] = fminf(gate[j], clamp);
                      up[j]   = fminf(clamp, fmaxf(-clamp, up[j]));
                    }
                    gate      = __builtin_convertvector(__builtin_convertvector(gate, fp4::ActivationBf16x4),
                                                   fp4::ActivationFloat4);
                    auto silu = __builtin_convertvector(
                        __builtin_convertvector(fp4::silu_fp32_bst4(gate), fp4::ActivationBf16x4),
                        fp4::ActivationFloat4);
                    auto activated = __builtin_convertvector(up * silu, fp4::ActivationBf16x4);
                    auto raw       = __builtin_bit_cast(fp4::ActivationBits4, activated);
                    MUTE_UNROLL
                    for (int j = 0; j < kActivationVector; ++j) values[col + j] = mutlass::bfloat16_t::bitcast(raw[j]);
                  }
                }
                MUTE_UNROLL
                for (int col = 0; col < kValuesPerLane; ++col) {
                  mutlass::bfloat16_t act = values[col];
                  values[col]             = act;
                  amax_bits               = max(amax_bits, uint32_t(act.raw()) & kBf16MagnitudeMask);
                }
                // Finite BF16 magnitudes are monotonic in their unsigned encoding.
                // Sanitize only the exceptional group, preserving status and +zero.
                if (amax_bits >= kBf16Infinity) {
                  atomicExch(status, 3);
                  amax_bits = 0;
                  MUTE_UNROLL
                  for (int col = 0; col < kValuesPerLane; ++col) {
                    uint32_t magnitude = uint32_t(values[col].raw()) & kBf16MagnitudeMask;
                    if (magnitude >= kBf16Infinity) {
                      values[col] = mutlass::bfloat16_t(0.f);
                      magnitude   = 0;
                    }
                    amax_bits = max(amax_bits, magnitude);
                  }
                }
                MUTE_UNROLL
                for (int delta = 1; delta < kLanesPerScale; delta *= 2)
                  amax_bits = max(amax_bits, __shfl_xor_sync(valid_lanes, amax_bits, delta));
                int      exponent = fp4::Fp4Quantization::scale_exponent(__uint_as_float(amax_bits << 16));
                float    inv      = fp4::Fp4Quantization::inverse(exponent);
                int      n        = int(get<1>(coords(group * kAccumulatorGroup)));
                int      outgroup = desc.n_block_idx * (kBlockN / (2 * kScaleBlock)) + n / (2 * kScaleBlock);
                uint64_t outrow   = uint64_t(desc.pool_block_idx) * kBlockM + row;
                if (lane % kLanesPerScale == 0)
                  sf2[uint64_t(outgroup) * P + outrow] = uint8_t(exponent + fp4::Fp4Quantization::kScaleBias);
                uint32_t packed = 0;
                MUTE_UNROLL
                for (int pair = 0; pair < kBytesPerLane; ++pair) {
                  uint8_t bits = __musa_cvt_float2_to_fp4x2(
                      float2{float(values[2 * pair]) * inv, float(values[2 * pair + 1]) * inv},
                      __MT_E2M1,
                      musaRoundNearest);
                  packed |= uint32_t(bits) << (pair * 8);
                }
                auto* dst = l2 + outrow * (I / kValuesPerByte) + outgroup * (kScaleBlock / kValuesPerByte) +
                            (n % kScaleBlock) / kValuesPerByte;
                *reinterpret_cast<uint32_t*>(dst) = packed;
              }
            }
          }
          consumer_sync();
        }
      }
    }
  }
};

template <class Stage>
__global__ __launch_bounds__(Stage::kNumThreads, Stage::kMinBlocksPerMp) void stage1_kernel(
    MUTLASS_GRID_CONSTANT typename Stage::Params const params,
    MUTLASS_GRID_CONSTANT layout::SymBuffer<Stage::R> const sym,
    uint8_t*                                                l1,
    uint8_t*                                                sf1,
    uint8_t*                                                l2,
    uint8_t*                                                sf2,
    int*                                                    status,
    int                                                     tokens,
    float                                                   clamp) {
  extern __shared__ __align__(DecodeConfig::kAlignment) char smem[];
  Stage::run(sym, params, l1, sf1, l2, sf2, status, tokens, clamp, smem);
}
// Preserve the default SwiGLU kernel argument ABI.
template <class Stage>
__global__ __launch_bounds__(Stage::kNumThreads, Stage::kMinBlocksPerMp) void situ_stage1_kernel(
    MUTLASS_GRID_CONSTANT typename Stage::Params const params,
    MUTLASS_GRID_CONSTANT layout::SymBuffer<Stage::R> const sym,
    uint8_t*                                                l1,
    uint8_t*                                                sf1,
    uint8_t*                                                l2,
    uint8_t*                                                sf2,
    int*                                                    status,
    int                                                     tokens,
    float                                                   clamp,
    float                                                   beta,
    float                                                   linear_beta) {
  extern __shared__ __align__(DecodeConfig::kAlignment) char smem[];
  Stage::run(sym, params, l1, sf1, l2, sf2, status, tokens, clamp, smem, beta, linear_beta);
}
}  // namespace mega_moe::decode::mp32::fp4_fp4
