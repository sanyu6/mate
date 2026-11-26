#pragma once

#include "mate/mega_moe/decode/fp4_fp4/decode_swap_ab.hpp"

namespace mega_moe::decode::mp32::fp4_fp4 {

// Match MP31 decode: FC2 producer/MMA roles, remote scatter, workspace
// retirement, cross-rank completion, then local weighted combine in this CTA.
template <class L,
          uint32_t MPs,
          uint32_t CTAsPerMP      = 1,
          bool     SwapAB         = false,
          int      FeatureTile    = 256,
          bool     PackB          = false,
          bool     SglangNumerics = false,

          bool DoubleBuffered = false>
struct Stage2 : DecodeCompute<DoubleBuffered ? 7 : (SwapAB ? 1 : (PackB ? 2 : 0)), FeatureTile> {
  using Compute = DecodeCompute<DoubleBuffered ? 7 : (SwapAB ? 1 : (PackB ? 2 : 0)), FeatureTile>;
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
  static constexpr int kAlignment = Compute::kAlignment;

  static constexpr int R = L::kRanks, E = L::kExperts, G = L::kGroups, T = L::kTopK;
  static constexpr int H = L::kHidden, I = L::kIntermediate, P = L::kPoolTokens;
  // Stage2 needs a producer squad and MMA squads, without a dispatch squad.
  static constexpr int      kProducerWarp   = 0;
  static constexpr int      kFirstMmaThread = kSquadThreads;
  static constexpr int      kNumThreads     = kFirstMmaThread + kNumMmaThreads;
  static constexpr uint32_t kNumCTAs        = MPs * CTAsPerMP;
  static constexpr int      kMinBlocksPerMp = DoubleBuffered ? CTAsPerMP : 1;
  static_assert(CTAsPerMP >= 1 && CTAsPerMP <= 3);

  using Pool = std::
      conditional_t<DoubleBuffered && PackB, PackedDecodePool<H, I, Compute>, typename Compute::template Pool<H, I>>;
  using Params    = typename Pool::Params;
  using Scheduler = mate_stage2::WorkspaceTileScheduler<G, H / kBlockN, kNumCTAs, R, kBlockM>;
  struct BarrierStorage {
    uint8_t pipeline[Pipeline::NumBarriers];
  };
  struct SharedStorage {
    typename Mainloop::SharedStorage mainloop;
  };
  static constexpr int SmemSize = sizeof(SharedStorage);
  static_assert(MPs >= 2 && MPs % 2 == 0);
  static_assert(SmemSize <= 256 * 1024);
  static_assert(sizeof(BarrierStorage) + Barrier::ReservedAsyncBarrierCount <=
                Barrier::HardwareMaxNumAsyncTransactionBarriers);

  MUTLASS_DEVICE static void run(
      layout::SymBuffer<R> const& sym, Params const& params, mutlass::bfloat16_t* y, int tokens, char* smem) {
    auto& shared   = *reinterpret_cast<SharedStorage*>(smem);
    auto* barriers = reinterpret_cast<BarrierStorage*>(0);
    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage) + Barrier::ReservedAsyncBarrierCount);
    typename Pipeline::Params pp;
    pp.transaction_bytes = Mainloop::TmeTransactionBytes;
    pp.num_consumers     = kNumMmaWarps;
    pp.num_producers     = 1;
    Pipeline pipeline(pp, reinterpret_cast<uint64_t>(&barriers->pipeline));
    int      tid = threadIdx.x, warp = tid / kWarpThreads;
    __syncthreads();
    auto*     base      = sym.template get_base_ptr<char*>();
    auto      workspace = L::workspace(base);
    auto*     combine   = L::template at<mutlass::bfloat16_t>(base, L::kCombine);
    Mainloop  mainloop;
    auto      inputs = Pool::init(P, G, params);
    Scheduler scheduler;
    auto      work = scheduler.initial_work_tile_info(workspace);

    if (warp == kProducerWarp) {
      auto write = mutlass::make_producer_start_state<Pipeline>();
      while (work.valid) {
        Pool::load(inputs,
                   make_coord(work.pool_block_idx, work.n_block_idx, work.expert_idx),
                   reinterpret_cast<char*>(&shared.mainloop),
                   params,
                   pipeline,
                   write);
        scheduler.advance_to_next_work();
        work = scheduler.get_work_tile_info();
      }
    } else if (tid >= kFirstMmaThread) {
      int                         mma_tid = tid - kFirstMmaThread;
      typename Mainloop::TiledMma mma;
      State                       read;
      while (work.valid) {
        auto accum = partition_fragment_C(mma, take<0, 2>(Tile{}));
        fill(accum, 0.f);
        mainloop.mma(Pool::mma_inputs(inputs, work.expert_idx),
                     Pool::coord(work.pool_block_idx, work.n_block_idx),
                     accum,
                     I / kBlockK,
                     mma_tid,
                     reinterpret_cast<char*>(&shared.mainloop),
                     pipeline,
                     read);
        if constexpr (SwapAB) {
          // W2 rows/scales are prepared in a feature32 x value4 permutation
          // inside each 128-feature block. Native C now owns adjacent output
          // features and scatters packed BF16 directly from registers.
          auto          coords = mma.get_thread_slice(mma_tid).partition_C(make_identity_tensor(take<0, 2>(Tile{})));
          constexpr int Values = decltype(size(accum))::value / 8;
          static_assert(Values == 2 || Values == 4);
          MUTE_UNROLL
          for (int token_index = 0; token_index < 8; ++token_index) {
            int row = int(get<1>(coords(token_index)));
            if (row < int(work.valid_m)) {
              int      physical    = work.n_block_idx * kBlockN + int(get<0>(coords(token_index)));
              int      feature     = (physical / 128) * 128 + (physical % 32) * 4 + (physical % 128) / 32;
              auto     metadata    = *workspace.get_token_src_metadata_ptr(work.pool_block_idx * kBlockM + row);
              auto*    destination = combine + (uint64_t(metadata.token_idx) * T + metadata.topk_idx) * H + feature;
              uint64_t packed      = 0;
              MUTE_UNROLL
              for (int j = 0; j < Values; ++j)
                packed |= uint64_t(mutlass::bfloat16_t(accum(token_index + j * 8)).raw()) << (16 * j);
              if constexpr (Values == 4)
                *reinterpret_cast<uint64_t*>(sym.map(destination, metadata.rank_idx)) = packed;
              else
                *reinterpret_cast<uint32_t*>(sym.map(destination, metadata.rank_idx)) = uint32_t(packed);
            }
          }
        } else {
          // Online swizzle assigns eight adjacent BF16 outputs to each lane.
          // Scatter them directly from MMA registers; no shared-memory staging
          // or producer/store handshake is needed after the accumulator is ready.
          auto          coords = mma.get_thread_slice(mma_tid).partition_C(make_identity_tensor(take<0, 2>(Tile{})));
          constexpr int kValuesPerStore   = sizeof(int4) / sizeof(mutlass::bfloat16_t);
          constexpr int kValuesPerConvert = 4;
          constexpr int kBf16PerWord      = sizeof(uint32_t) / sizeof(mutlass::bfloat16_t);
          constexpr int kWordBits         = std::numeric_limits<uint32_t>::digits;
          using Float4                    = float __attribute__((ext_vector_type(kValuesPerConvert)));
          using Bf16x4                    = __bf16 __attribute__((ext_vector_type(kValuesPerConvert)));
          using StoreWords4               = uint32_t __attribute__((ext_vector_type(sizeof(int4) / sizeof(uint32_t))));
          using GlobalVector              = StoreWords4 __attribute__((address_space(1)));
          static_assert(kValuesPerStore == fp4::kStage2PermutationStride);
          MUTE_UNROLL
          for (int index = 0; index < size(accum); index += kValuesPerStore) {
            auto coord = coords(index);
            int  row   = int(get<0>(coord));
            if (row < int(work.valid_m)) {
              auto  metadata    = *workspace.get_token_src_metadata_ptr(work.pool_block_idx * kBlockM + row);
              auto* destination = combine + (uint64_t(metadata.token_idx) * T + metadata.topk_idx) * H +
                                  work.n_block_idx * kBlockN + int(get<1>(coord));
              StoreWords4 payload;
              MUTE_UNROLL
              for (int group = 0; group < kValuesPerStore / kValuesPerConvert; ++group) {
                int  offset = index + group * kValuesPerConvert;
                auto values = __builtin_convertvector(
                    Float4{accum(offset), accum(offset + 1), accum(offset + 2), accum(offset + 3)}, Bf16x4);
                uint64_t bits                                         = __builtin_bit_cast(uint64_t, values);
                payload[group * kValuesPerConvert / kBf16PerWord]     = uint32_t(bits);
                payload[group * kValuesPerConvert / kBf16PerWord + 1] = uint32_t(bits >> kWordBits);
              }
              *((GlobalVector*)sym.map(destination, metadata.rank_idx)) = payload;
            }
          }
        }
        scheduler.advance_to_next_work();
        work = scheduler.get_work_tile_info();
      }
    }
    // Cached scatter must be flushed by the CTA leader after all store warps
    // finish. A noflush fence alone does not publish those BF16 cache lines.
    __syncthreads();
    if (tid == 0) __threadfence_system();
    __syncthreads();
    grid_sync<kNumCTAs, 0>(workspace, blockIdx.x, tid, []() { __syncthreads(); });
    // Retire only invocation-local counters; keep persistent barrier phases.
    if (blockIdx.x == 0) {
      for (int e = tid; e < E; e += kNumThreads) st_relaxed_sys_global(workspace.get_expert_send_count_ptr(e), 0ull);
    } else {
      for (int e = blockIdx.x - 1; e < G; e += kNumCTAs - 1) {
        uint32_t count  = scheduler.get_num_tokens(e);
        uint32_t offset = scheduler.get_pool_block_offset(e);
        for (int rank = tid; rank < R; rank += kNumThreads)
          st_relaxed_sys_global(workspace.get_expert_recv_count_ptr(rank, e), 0ull);
        for (uint32_t b = tid; b < math::ceil_div(count, uint32_t(kBlockM)); b += kNumThreads)
          st_relaxed_sys_global(workspace.get_l1_arrival_count_ptr(offset + b), 0u);
        if (tid == 0) st_relaxed_sys_global(workspace.get_expert_recv_count_sum_ptr(e), 0ull);
      }
    }
    mtlink_barrier<R, kNumCTAs, kNumThreads, 0, 2, true>(
        workspace, sym, blockIdx.x, tid, []() { __syncthreads(); }, true, true);

    auto*         ids              = L::template at<int64_t>(base, L::kRoutes);
    auto*         weights          = L::template at<float>(base, L::kWeights);
    constexpr int kBf16PerVector   = sizeof(int4) / sizeof(mutlass::bfloat16_t);
    constexpr int kVectorsPerToken = H / kBf16PerVector;
    for (int item = blockIdx.x * kNumThreads + tid; item < tokens * kVectorsPerToken; item += kNumCTAs * kNumThreads) {
      int   token = item / kVectorsPerToken, vec = item % kVectorsPerToken;
      float sum[kBf16PerVector] = {};
      if constexpr (SglangNumerics) {
        // Match combine/RS's BF16 EP-owner partial before the rank-ordered sum.
        MUTE_UNROLL
        for (int owner = 0; owner < R; ++owner) {
          float partial[kBf16PerVector] = {};
          MUTE_UNROLL
          for (int slot = 0; slot < T; ++slot) {
            auto route = __ldg(ids + uint64_t(token) * T + slot);
            if (route < 0 || route >= E || route / G != owner) continue;
            auto* src   = reinterpret_cast<int4 const*>(combine + (uint64_t(token) * T + slot) * H) + vec;
            int4  value = __lsu_ld_cache_hint(src, 3, 3, 0, 0);
            auto* bf16  = reinterpret_cast<mutlass::bfloat16_t const*>(&value);
            MUTE_UNROLL
            for (int j = 0; j < kBf16PerVector; ++j) partial[j] += float(bf16[j]);
          }
          MUTE_UNROLL
          for (int j = 0; j < kBf16PerVector; ++j) sum[j] += float(mutlass::bfloat16_t(partial[j]));
        }
      } else {
        for (int slot = 0; slot < T; ++slot) {
          auto route = __ldg(ids + uint64_t(token) * T + slot);
          if (route < 0 || route >= E) continue;
          float weight = __ldg(weights + uint64_t(token) * T + slot);
          auto* src    = reinterpret_cast<int4 const*>(combine + (uint64_t(token) * T + slot) * H) + vec;
          int4  value  = __lsu_ld_cache_hint(src, 3, 3, 0, 0);
          auto* bf16   = reinterpret_cast<mutlass::bfloat16_t const*>(&value);
          MUTE_UNROLL
          for (int j = 0; j < kBf16PerVector; ++j) sum[j] += float(bf16[j]) * weight;
        }
      }
      int4  value;
      auto* bf16 = reinterpret_cast<mutlass::bfloat16_t*>(&value);
      MUTE_UNROLL
      for (int j = 0; j < kBf16PerVector; ++j) bf16[j] = mutlass::bfloat16_t(sum[j]);
      *(reinterpret_cast<int4*>(y + uint64_t(token) * H) + vec) = value;
    }
  }
};

template <class Stage>
__global__ __launch_bounds__(Stage::kNumThreads, Stage::kMinBlocksPerMp) void stage2_kernel(
    MUTLASS_GRID_CONSTANT typename Stage::Params const params,
    MUTLASS_GRID_CONSTANT layout::SymBuffer<Stage::R> const sym,
    mutlass::bfloat16_t*                                    y,
    int                                                     tokens) {
  extern __shared__ __align__(DecodeConfig::kAlignment) char smem[];
  Stage::run(sym, params, y, tokens, smem);
}
}  // namespace mega_moe::decode::mp32::fp4_fp4
