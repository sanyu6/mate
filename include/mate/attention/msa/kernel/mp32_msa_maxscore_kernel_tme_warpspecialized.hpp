#pragma once

#include <mute/tensor.hpp>
#include <mutlass/kernel_hardware_info.hpp>

#include "mate/attention/msa/kernel/mp32_msa_maxscore_m64.hpp"

namespace mate::attention::msa {

template <class CollectiveMainloop_, class TileScheduler_, bool NarrowScoreOffsets_ = false>
struct Mp32MsaMaxScoreKernelTmeWarpSpecialized {
  using CollectiveMainloop                 = CollectiveMainloop_;
  using TileScheduler                      = TileScheduler_;
  static constexpr bool NarrowScoreOffsets = NarrowScoreOffsets_;
  using NarrowScoreOffsetKernel = Mp32MsaMaxScoreKernelTmeWarpSpecialized<CollectiveMainloop, TileScheduler, true>;
  static constexpr bool can_use_narrow_score_offsets(int num_heads, int max_k_tiles) {
    return num_heads > 0 && max_k_tiles > 0 &&
           uint64_t(max_k_tiles) * (uint64_t(num_heads) * 32 + 1) * sizeof(float) <= UINT32_MAX;
  }
  static constexpr bool UseM64 = mute::is_same_v<typename CollectiveMainloop::Element, mutlass::float_e4m3_t> &&
                                 CollectiveMainloop::MmaTileQ == 64 && CollectiveMainloop::KStages == 2 &&
                                 CollectiveMainloop::HeadDim == 128 && CollectiveMainloop::HeadRatio == 1 &&
                                 CollectiveMainloop::TileQ == 128;

  struct ProblemSize {
    int batch_size;
    int total_q;
    int total_k;
    int num_qo_heads;
    int num_kv_heads;
    int max_seqlen_q;
    int max_seqlen_k;
    int max_k_tiles;
  };

  using SharedStorage                                  = typename CollectiveMainloop::SharedStorage;
  using BarrierStorage                                 = typename CollectiveMainloop::BarrierStorage;
  static constexpr int      SmemAlignmentBytes         = CollectiveMainloop::SmemAlignmentBytes;
  static constexpr int      SharedStorageSize          = sizeof(SharedStorage);
  static constexpr int      MaxThreadsPerBlock         = CollectiveMainloop::NumThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = UseM64 ? 4 : 1;
  static_assert(CollectiveMainloop::AsyncBarrierCount <=
                    mutlass::arch::AsyncBarrier::HardwareMaxNumAsyncTransactionBarriers,
                "MSA max-score async barrier id exceeds the hardware limit.");

  struct Arguments {
    ProblemSize                            problem_size;
    typename CollectiveMainloop::Arguments mainloop;
    typename TileScheduler::Arguments      scheduler;
    mutlass::KernelHardwareInfo            hw_info{};
  };

  struct Params {
    ProblemSize                         problem_size;
    typename CollectiveMainloop::Params mainloop;
    typename TileScheduler::Params      scheduler;
    mutlass::KernelHardwareInfo         hw_info{};
  };

  static Params to_underlying_arguments(Arguments const& args) {
    int mp_count = args.hw_info.sm_count;
    if (mp_count <= 0) {
      mp_count = mutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    Params params{
        args.problem_size,
        CollectiveMainloop::to_underlying_arguments(args.problem_size, args.mainloop),
        TileScheduler::to_underlying_arguments(args.problem_size, args.scheduler, mp_count),
        mutlass::KernelHardwareInfo{args.hw_info.device_id, mp_count},
    };
    return params;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.scheduler, params.hw_info.sm_count);
  }

  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  MUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    if constexpr (UseM64) {
      Mp32MsaMaxScoreM64<CollectiveMainloop, TileScheduler, NarrowScoreOffsets>::run(params, smem_buf);
      return;
    }
    int thread_idx             = int(threadIdx.x);
    int warp_idx               = mutlass::canonical_warp_idx_sync();
    int warp_squad_idx         = mutlass::canonical_warp_squad_idx();
    int warp_idx_in_warp_squad = warp_idx % mutlass::NumWarpsPerWarpSquad;
    int consumer_thread_idx    = thread_idx - CollectiveMainloop::NumProducerThreads;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    mutlass::arch::allocate_async_barriers(CollectiveMainloop::AsyncBarrierCount);
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);

    CollectiveMainloop                    mainloop;
    TileScheduler                         scheduler;
    typename CollectiveMainloop::Pipeline pipeline(barrier_storage);
    __syncthreads();

    if (warp_squad_idx == 0) {
      int  lane_idx      = thread_idx % mutlass::NumThreadsPerWarp;
      auto schedule_info = scheduler.get_schedule_info(params.scheduler);
      for (auto q_work_tile = scheduler.get_initial_work(params.scheduler, schedule_info);
           q_work_tile.is_valid(params.scheduler);
           q_work_tile = scheduler.get_next_work(params.scheduler, q_work_tile, schedule_info)) {
        __musa_loop_transparent_outermost();
        if (!TileScheduler::is_valid_q_tile(q_work_tile)) {
          continue;
        }
        int k_tile_count = mainloop.k_tile_count(params.mainloop, q_work_tile);
        if (TileScheduler::ParallelKTiles && q_work_tile.k_partition_idx >= k_tile_count) {
          continue;
        }
        if (warp_idx_in_warp_squad != CollectiveMainloop::QLoadWarpInProducer &&
            warp_idx_in_warp_squad != CollectiveMainloop::KLoadWarpInProducer) {
          continue;
        }
        if (lane_idx == 0 && warp_idx_in_warp_squad == CollectiveMainloop::QLoadWarpInProducer) {
          mainloop.load_q(params.mainloop, pipeline, shared_storage, q_work_tile);
        }
        if (warp_idx_in_warp_squad != CollectiveMainloop::KLoadWarpInProducer) {
          continue;
        }
        const int k_tile_begin = TileScheduler::ParallelKTiles ? q_work_tile.k_partition_idx : 0;
        const int k_tile_step  = TileScheduler::ParallelKTiles ? params.scheduler.k_partitions : 1;
        int       page_idx     = 0;
        if constexpr (CollectiveMainloop::EnableKPrefetch) {
          const int first_k_tile_idx = k_tile_begin;
          if (lane_idx == 0 && first_k_tile_idx < k_tile_count) {
            auto first_k_work_tile = TileScheduler::make_k_work_tile(q_work_tile, first_k_tile_idx);
            page_idx               = mainloop.safe_page_idx(params.mainloop, first_k_work_tile);
          }
        }
        MUTLASS_PRAGMA_NO_UNROLL
        for (int k_tile_idx = k_tile_begin; k_tile_idx < k_tile_count; k_tile_idx += k_tile_step) {
          auto k_work_tile = TileScheduler::make_k_work_tile(q_work_tile, k_tile_idx);
          if (lane_idx == 0) {
            if constexpr (!CollectiveMainloop::EnableKPrefetch) {
              page_idx = mainloop.safe_page_idx(params.mainloop, k_work_tile);
            }
            mainloop.load_k(params.mainloop, pipeline, shared_storage, k_work_tile, page_idx);
            if constexpr (CollectiveMainloop::EnableKPrefetch) {
              int next_k_tile_idx = k_tile_idx + k_tile_step;
              if (next_k_tile_idx < k_tile_count) {
                auto next_k_work_tile = TileScheduler::make_k_work_tile(q_work_tile, next_k_tile_idx);
                page_idx              = mainloop.safe_page_idx(params.mainloop, next_k_work_tile);
                mainloop.prefetch_k(params.mainloop, next_k_work_tile, page_idx);
              }
            }
          }
        }
      }
    } else if (warp_squad_idx <= CollectiveMainloop::NumMmaWarpSquads) {
      auto schedule_info = scheduler.get_schedule_info(params.scheduler);
      for (auto q_work_tile = scheduler.get_initial_work(params.scheduler, schedule_info);
           q_work_tile.is_valid(params.scheduler);
           q_work_tile = scheduler.get_next_work(params.scheduler, q_work_tile, schedule_info)) {
        __musa_loop_transparent_outermost();
        if (!TileScheduler::is_valid_q_tile(q_work_tile)) {
          continue;
        }
        int k_tile_count = mainloop.k_tile_count(params.mainloop, q_work_tile);
        if (TileScheduler::ParallelKTiles && q_work_tile.k_partition_idx >= k_tile_count) {
          continue;
        }
        int  first_masked_k_tile = mainloop.first_masked_k_tile(params.mainloop, q_work_tile);
        auto rowmax_store_params = mainloop.make_rowmax_store_params(params.mainloop, q_work_tile, consumer_thread_idx);
        mainloop.wait_q(pipeline, consumer_thread_idx);
        auto      q_fragment   = mainloop.load_q_fragment(pipeline, shared_storage, consumer_thread_idx);
        const int k_tile_begin = TileScheduler::ParallelKTiles ? q_work_tile.k_partition_idx : 0;
        const int k_tile_step  = TileScheduler::ParallelKTiles ? params.scheduler.k_partitions : 1;
        MUTLASS_PRAGMA_NO_UNROLL
        for (int k_tile_idx = k_tile_begin; k_tile_idx < k_tile_count; k_tile_idx += k_tile_step) {
          auto k_work_tile = TileScheduler::make_k_work_tile(q_work_tile, k_tile_idx);
          mainloop.compute_k_tile(params.mainloop,
                                  pipeline,
                                  shared_storage,
                                  k_work_tile,
                                  first_masked_k_tile,
                                  rowmax_store_params,
                                  consumer_thread_idx,
                                  q_fragment);
        }
        mainloop.release_q(pipeline, consumer_thread_idx);
      }
    }
  }
};

}  // namespace mate::attention::msa
