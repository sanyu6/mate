#pragma once
#include "mate/mega_moe/prefill/mp32_prefill_common.hpp"
namespace mega_moe::prefill::mp32 {
template <uint32_t S, uint32_t H, uint32_t I, uint32_t E, uint32_t T, uint32_t P, uint32_t MPs, uint32_t R>
struct Stage2 : PrefillGemmConfig {
  static constexpr uint32_t kNumRanks_      = R;
  static constexpr int      kMainloopStages = 4;
  using RS                                  = RsMainloop<H, I, kMainloopStages>;
  using Collective                          = typename RS::Collective;
  using Pipeline                            = typename RS::Pipeline;
  using Params                              = typename RS::Params;
  static constexpr int kEpilogueRows        = 64;
  // Eight BF16 padding columns preserve 16-byte alignment and shift shared banks.
  static constexpr int kEpilogueStride       = kBlockN + kBf16StoreElements;
  static constexpr int kNumEpiloguePasses    = kBlockM / kEpilogueRows;
  static constexpr int kRowsPerThreadPass    = kEpilogueRows / kCFragmentRowStride;
  static constexpr int kVectorStoresPerGroup = 4;
  static constexpr int kStoreGroupElements   = kVectorStoresPerGroup * kBf16StoreElements;
  static constexpr int kStoreGroupsPerRow    = kBlockN / kStoreGroupElements;
  static constexpr int kNumConsumerBarriers  = 2;
  static_assert(kBlockM % kEpilogueRows == 0 && kBlockN % kStoreGroupElements == 0);
  static constexpr int SmemSize =
      sizeof(typename Collective::SharedStorage) + kEpilogueRows * kEpilogueStride * sizeof(mutlass::bfloat16_t);
  static_assert(SmemSize <= kSmemCapacityBytes);
  MUTLASS_DEVICE static void run(layout::SymBuffer<R> sym_buffer, Params const& params, char* smem) {
    using ConsumerBarrier            = mutlass::arch::AsyncBarrier;
    constexpr uint32_t consumer_base = Pipeline::NumBarriers + ConsumerBarrier::ReservedAsyncBarrierCount;
    static_assert(Pipeline::NumBarriers + kNumConsumerBarriers + ConsumerBarrier::ReservedAsyncBarrierCount <=
                  ConsumerBarrier::HardwareMaxNumAsyncTransactionBarriers);
    mutlass::arch::allocate_async_barriers(Pipeline::NumBarriers + kNumConsumerBarriers);
    if (threadIdx.x == 0) {
      ConsumerBarrier::init(consumer_base, kNumMmaWarps, 0);
      ConsumerBarrier::init(consumer_base + 1, kNumMmaWarps, 0);
    }
    typename Pipeline::Params pp;
    pp.transaction_bytes = Collective::TmeTransactionBytes;
    pp.num_consumers     = kNumMmaWarps;
    pp.num_producers     = 1;
    Pipeline           pipeline(pp, 0);
    auto               write = mutlass::make_producer_start_state<Pipeline>();
    typename RS::State read;
    auto               layout    = unified::BufferLayout(sym_buffer.get_base_ptr(), R, E, S, T, H, I, P, P);
    auto               workspace = layout.get_prefill_workspace();
    auto               combine   = layout.get_prefill_combine_token_buffer();
    Collective         mainloop;
    auto               inputs = mainloop.load_init(make_shape(H, P, I, E / R), params);
    using Scheduler           = mate_stage2::WorkspaceTileScheduler<E / R, H / kBlockN, MPs, R, kBlockM, true>;
    __shared__ volatile uint32_t coordinate_zero;
    if (threadIdx.x == 0) coordinate_zero = 0;
    __syncthreads();
    int warp = mutlass::canonical_warp_idx_sync();
    if (warp == 0) {
      Scheduler scheduler;
      auto      work = scheduler.initial_work_tile_info(workspace);
      while (work.valid) {
        RS::load(inputs,
                 make_coord(work.n_block_idx, work.pool_block_idx, _, work.expert_idx),
                 smem,
                 params,
                 pipeline,
                 write);
        scheduler.advance_to_next_work();
        work = scheduler.get_work_tile_info();
      }
    } else if (warp >= kFirstMmaWarp) {
      int                   tid = int(threadIdx.x) - kFirstMmaThread;
      typename RS::TiledMma mma;
      auto*    epilogue   = reinterpret_cast<mutlass::bfloat16_t*>(smem + sizeof(typename Collective::SharedStorage));
      uint32_t sync_index = 0;
      auto     consumer_sync = [&]() {
        uint32_t bar   = consumer_base + (sync_index++ & 1);
        auto     phase = ConsumerBarrier::arrive<true>(bar);
        ConsumerBarrier::wait(bar, phase);
      };
      Scheduler scheduler;
      auto      work = scheduler.initial_work_tile_info(workspace);
      while (work.valid) {
        auto accum = partition_fragment_C(mma, take<0, 2>(tile_shape(mma)));
        fill(accum, 0.f);
        mainloop.mma(inputs,
                     make_coord(work.n_block_idx, work.pool_block_idx, _, work.expert_idx),
                     accum,
                     I / kBlockK,
                     tid,
                     smem,
                     pipeline,
                     read);
        // Keep expanded scatter addresses out of the register-heavy MMA loop.
        auto coords =
            mma.get_thread_slice(tid + coordinate_zero).partition_C(make_identity_tensor(take<0, 2>(tile_shape(mma))));
        static_assert(size(accum) == kBlockM * kBlockN / kNumMmaThreads);
        // Each pass gathers a 64-token stripe into row-major shared memory.
        // Keep this scratch separate from the producer's live TME ring.
        MUTE_UNROLL
        for (int pass = 0; pass < kNumEpiloguePasses; ++pass) {
          MUTE_UNROLL
          for (int row_group = 0; row_group < kRowsPerThreadPass; ++row_group) {
            int      row = pass * kRowsPerThreadPass + row_group;
            int      i   = (row / kCFragmentGroupElements) * kCFragmentPairElements + row % kCFragmentGroupElements;
            uint32_t m = get<1>(coords(i)), n = get<0>(coords(i));
            epilogue[(m % kEpilogueRows) * kEpilogueStride + n] = mutlass::bfloat16_t(accum(i));
            epilogue[(m % kEpilogueRows) * kEpilogueStride + n + kCFragmentGroupElements] =
                mutlass::bfloat16_t(accum(i + kCFragmentGroupElements));
          }
          // Publish the entire stripe before remapping threads to vector stores.
          consumer_sync();
          for (int item = tid; item < kEpilogueRows * kStoreGroupsPerRow; item += kNumMmaThreads) {
            uint32_t row = item / kStoreGroupsPerRow, n = (item % kStoreGroupsPerRow) * kStoreGroupElements;
            uint32_t m = pass * kEpilogueRows + row;
            if (m < work.valid_m) {
              auto  metadata    = *workspace.get_token_src_metadata_ptr(work.pool_block_idx * kBlockM + m);
              auto  destination = combine.get_rank_buffer(metadata.topk_idx()).get_data_buffer(metadata.token_idx());
              auto* ptr = sym_buffer.map(destination.template get_base_ptr<mutlass::bfloat16_t>(), metadata.rank_idx());
              // Reuse one route lookup for four contiguous 16-byte stores.
              MUTE_UNROLL
              for (int v = 0; v < kVectorStoresPerGroup; ++v) {
                *reinterpret_cast<int4*>(ptr + work.n_block_idx * kBlockN + n + v * kBf16StoreElements) =
                    *reinterpret_cast<int4 const*>(epilogue + row * kEpilogueStride + n + v * kBf16StoreElements);
              }
            }
          }
          // All vector loads must finish before another stripe reuses the scratch.
          consumer_sync();
        }
        scheduler.advance_to_next_work();
        work = scheduler.get_work_tile_info();
      }
    }
    __threadfence_system_noflush();
    mtlink_barrier<R, MPs, kNumThreads, 0, 1, true>(
        workspace, sym_buffer, blockIdx.x, threadIdx.x, [&]() { __syncthreads_lm(); }, true, true);
    clear_prefill_split_workspace<E, R, MPs, kNumThreads>(workspace);
    __threadfence_system_noflush();
    mtlink_barrier<R, MPs, kNumThreads, 1, 2, true>(
        workspace, sym_buffer, blockIdx.x, threadIdx.x, [&]() { __syncthreads_lm(); }, true, true);
  }
};
template <class Impl>
__global__ __launch_bounds__(Impl::kNumThreads, 1) void stage2_kernel(layout::SymBuffer<Impl::kNumRanks_> sym,
                                                                      typename Impl::Params               params) {
  extern __shared__ __align__(PrefillGemmConfig::kSmemBaseAlignment) char smem[];
  Impl::run(sym, params, smem);
}
}  // namespace mega_moe::prefill::mp32
