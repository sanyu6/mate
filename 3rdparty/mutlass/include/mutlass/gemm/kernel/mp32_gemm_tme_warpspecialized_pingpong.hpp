/***************************************************************************************************
 * Copyright (c) 2024 - 2026 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of
 * conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of
 * conditions and the following disclaimer in the documentation and/or other materials provided
 * with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to
 * endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
 **************************************************************************************************/

#pragma once

#include "mutlass/mutlass.h"
#include "mutlass/kernel_hardware_info.hpp"
#include "mutlass/epilogue/collective/detail.hpp"
#include "mutlass/epilogue/dispatch_policy.hpp"
#include "mutlass/gemm/kernel/tile_scheduler.hpp"
#include "mutlass/gemm/gemm.h"
#include "mutlass/gemm/dispatch_policy.hpp"
#include "mutlass/pipeline/mp32_pipeline.hpp"
#include "mutlass/trace.h"
#include "mutlass/arch/barrier.hpp"

#include "mute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass::gemm::kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  mute::enable_if_t<
      mute::is_same_v<typename CollectiveMainloop_::ArchTag, arch::Mp32> &&
      mute::is_base_of_v<KernelTmeWarpSpecializedPingpongMp32,
                         typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
{
public:
  using ProblemShape = ProblemShape_;

  static_assert(rank(ProblemShape{}) == 3 || rank(ProblemShape{}) == 4,
                "ProblemShape must be <M,N,K> or <M,N,K,L>");

  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ElementA = typename CollectiveMainloop::ElementA;
  using StrideA = typename CollectiveMainloop::StrideA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using StrideB = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using KernelSchedule = typename DispatchPolicy::Schedule;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // Two consumer groups alternate MMA and epilogue sections.
  using MathOrderBarrier = mutlass::OrderedSequenceBarrier<2, 2>;
  using MathOrderBarrierParams = typename MathOrderBarrier::Params;

  static constexpr int NumConsumerThreads = size(TiledMma{});
  static constexpr int NumConsumerWarps = NumConsumerThreads / mutlass::NumThreadsPerWarp;
  static constexpr int NumProducerThreads = mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumConsumerGroups = MathOrderBarrier::SequenceLength;

  static_assert(NumConsumerThreads > 0 && NumConsumerThreads <= 256,
                "MP32 ping-pong requires at most 256 TiledMma threads");
  static_assert(NumConsumerThreads % mutlass::NumThreadsPerWarpSquad == 0,
                "MP32 ping-pong requires whole 128-thread SQMMA atoms");

  using MainloopPipelineData = typename CollectiveMainloop::MainloopPipelineData;
  using PipelineDataParams = typename MainloopPipelineData::Params;
  using PipelineState = typename MainloopPipelineData::PipelineState;
  static_assert(ArchTag::kMinComputeCapability == 32);

  using CollectiveEpilogue = CollectiveEpilogue_;
  static constexpr bool UsesTmeEpilogue =
      epilogue::collective::detail::IsMp32TmeEpilogue<CollectiveEpilogue>::value;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static constexpr int NumEpilogueBarriers =
      NumConsumerGroups * CollectiveEpilogue::NumBarriers;
  static_assert(UsesTmeEpilogue || CollectiveEpilogue::NumBarriers == 0,
                "MP32 ping-pong requires an LSU or MP32 TME epilogue");
  static constexpr int EpilogueBarrierBase =
      CollectiveMainloop::NumBarriers + MathOrderBarrier::NumBarriers;

  static_assert(mute::is_void_v<TileScheduler_> ||
                mute::is_same_v<TileScheduler_, PersistentScheduler>,
                "This kernel only supports the persistent tile scheduler");

  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
      PersistentScheduler, ArchTag, TileShape, ClusterShape>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;

  static constexpr int EpilogueSmemAlignment = alignof(EpilogueSharedStorage);
  static constexpr bool CanOverlayEpilogueStorage =
      mute::is_empty_v<EpilogueSharedStorage> &&
      alignof(MainloopSharedStorage) >= alignof(EpilogueSharedStorage);
  static constexpr int EpilogueSmemOffset = CanOverlayEpilogueStorage ? 0
      : mute::ceil_div(int(sizeof(MainloopSharedStorage)), EpilogueSmemAlignment) *
            EpilogueSmemAlignment;
  static constexpr int EpilogueSmemStride = CanOverlayEpilogueStorage
      ? 0 : int(sizeof(EpilogueSharedStorage));
  static constexpr int SharedStorageSize = CanOverlayEpilogueStorage
      ? int(sizeof(MainloopSharedStorage))
      : mute::ceil_div(EpilogueSmemOffset + NumConsumerGroups * EpilogueSmemStride,
                       int(alignof(MainloopSharedStorage))) * alignof(MainloopSharedStorage);
  static_assert(SharedStorageSize <= 256 * 1024,
                "MP32 ping-pong shared storage exceeds 256 KiB");

  static constexpr uint32_t MaxThreadsPerBlock =
      NumProducerThreads + NumConsumerGroups * NumConsumerThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int SmemAlignmentBytes = mute::max(
      CollectiveMainloop::SmemAlignmentBytes, int(alignof(EpilogueSharedStorage)));
  static constexpr int NumBarriers =
      CollectiveMainloop::NumBarriers + MathOrderBarrier::NumBarriers +
      NumEpilogueBarriers;

  static_assert(
      NumBarriers <=
          arch::AsyncBarrier::HardwareMaxNumAsyncTransactionBarriers -
              arch::AsyncBarrier::ReservedAsyncBarrierCount,
      "MP32 ping-pong exceeds the hardware async-barrier capacity");

  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop;
    EpilogueParams epilogue;
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
  };

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void)workspace;
    KernelHardwareInfo hw_info{args.hw_info.device_id, args.hw_info.sm_count};
    auto problem_shape_MNKL = append<4>(args.problem_shape, mute::Int<1>{});
    return {
      args.mode,
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(
          args.problem_shape, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(
          args.problem_shape, args.epilogue, workspace),
      hw_info,
      TileScheduler::to_underlying_arguments(
          problem_shape_MNKL, TileShape{}, hw_info, args.scheduler)
    };
  }

  MUTLASS_HOST_DEVICE static bool
  can_implement(Arguments const& args) {
    bool implementable = args.mode == GemmUniversalMode::kGemm ||
        (args.mode == GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);
    if (!implementable) {
      MUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return false;
    }

    implementable &= CollectiveMainloop::can_implement(
        args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(
        args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);
    return implementable;
  }

  static int
  get_workspace_size(Arguments const&) {
    return 0;
  }

  static mutlass::Status
  initialize_workspace(
      Arguments const&, void* = nullptr, musaStream_t = nullptr,
      MusaHostAdapter* = nullptr) {
    return Status::kSuccess;
  }

  static dim3
  get_grid_shape(Params const& params) {
    TileSchedulerArguments args{};
    args.swizzle_size = params.scheduler.swizzle_size_;
    args.raster_order = params.scheduler.raster_order_ ==
            TileScheduler::RasterOrder::AlongN
        ? TileScheduler::RasterOrderOptions::AlongN
        : TileScheduler::RasterOrderOptions::AlongM;
    return TileScheduler::get_grid_shape(
        params.scheduler, params.problem_shape, TileShape{}, params.hw_info, args);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  MUTLASS_DEVICE void
  operator()(Params const& params, char* smem_buf) {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    using namespace mute;

    static_assert(rank(StrideA{}) == 3,
                  "StrideA must be rank-3: [M,K,L]");
    static_assert(rank(StrideB{}) == 3,
                  "StrideB must be rank-3: [N,K,L]");
    static_assert(rank(StrideC{}) == 3,
                  "StrideC must be rank-3: [M,N,L]");
    static_assert(rank(StrideD{}) == 3,
                  "StrideD must be rank-3: [M,N,L]");

    char* mainloop_smem = smem_buf;

    int thread_idx = int(threadIdx.x);
    bool is_producer = thread_idx < NumProducerThreads;
    int consumer_linear_thread = thread_idx - NumProducerThreads;
    int consumer_group_idx = is_producer
        ? 0
        : consumer_linear_thread / NumConsumerThreads;
    int consumer_thread_idx = is_producer
        ? 0
        : consumer_linear_thread - consumer_group_idx * NumConsumerThreads;
    char* epilogue_smem = smem_buf;
    if constexpr (!CanOverlayEpilogueStorage) {
      auto* epilogue_storage =
          reinterpret_cast<EpilogueSharedStorage*>(smem_buf + EpilogueSmemOffset);
      epilogue_smem = reinterpret_cast<char*>(&epilogue_storage[consumer_group_idx]);
    }
    int epilogue_barrier_base = EpilogueBarrierBase +
        consumer_group_idx * CollectiveEpilogue::NumBarriers;

    arch::allocate_async_barriers(NumBarriers);

    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto blk_shape = TileShape{};

    CollectiveMainloop collective_mma;
    auto load_inputs = collective_mma.load_init(
        problem_shape_MNKL, params.mainloop);
    static_assert(tuple_size_v<decltype(load_inputs)> >= 2,
                  "load_init must return A and B tensors first");

    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);
    int k_tile_count = int(size<3>(gA_mkl));

    PipelineDataParams pipeline_params;
    pipeline_params.transaction_bytes =
        CollectiveMainloop::TmeTransactionBytesMK +
        CollectiveMainloop::TmeTransactionBytesNK;
    pipeline_params.num_consumers = NumConsumerWarps;
    pipeline_params.num_producers = 1;
    MainloopPipelineData pipeline_data(pipeline_params);

    MathOrderBarrierParams order_params;
    order_params.group_id = uint32_t(consumer_group_idx);
    order_params.group_size = uint32_t(NumConsumerWarps);
    MathOrderBarrier order_barrier(
        order_params, CollectiveMainloop::NumBarriers);

    if constexpr (UsesTmeEpilogue) {
      MUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < NumConsumerGroups; ++group) {
        CollectiveEpilogue::initialize(thread_idx,
            EpilogueBarrierBase + group * CollectiveEpilogue::NumBarriers);
      }
    }

    __syncthreads();

    if (is_producer) {
      TileScheduler scheduler{params.scheduler};
      PipelineState smem_pipe_write =
          mutlass::make_producer_start_state<MainloopPipelineData>();
      auto work_tile_info = scheduler.initial_work_tile_info();

      MUTLASS_PRAGMA_NO_UNROLL
      for (; work_tile_info.is_valid();
           work_tile_info = scheduler.get_work_tile_info()) {
        __musa_loop_transparent_outermost();

        auto m_coord = idx2crd(int(work_tile_info.M_idx), shape<2>(gA_mkl));
        auto n_coord = idx2crd(int(work_tile_info.N_idx), shape<2>(gB_nkl));
        auto l_coord = idx2crd(int(work_tile_info.L_idx), shape<4>(gB_nkl));
        auto output_tile_coord = make_coord(m_coord, n_coord, _, l_coord);

        Tensor gB = gB_nkl(_, _, n_coord, _, l_coord);
        auto k_tile_iter = make_coord_iterator(shape<2>(gB));

        collective_mma.load(
            load_inputs, output_tile_coord, k_tile_iter, k_tile_count,
            mainloop_smem, params.mainloop, pipeline_data, smem_pipe_write);
        scheduler.advance_to_next_work();
      }
    }
    else {
      TileScheduler scheduler{params.scheduler};
      auto work_tile_info = scheduler.initial_work_tile_info();
      PipelineState smem_pipe_read;

      if (consumer_group_idx == 1) {
        scheduler.advance_to_next_work();
        work_tile_info = scheduler.get_work_tile_info();
        smem_pipe_read.advance(uint32_t(k_tile_count));
      }

      MUTLASS_PRAGMA_NO_UNROLL
      for (; work_tile_info.is_valid();
           work_tile_info = scheduler.get_work_tile_info()) {
        __musa_loop_transparent_outermost();

        auto m_coord = idx2crd(int(work_tile_info.M_idx), shape<2>(gA_mkl));
        auto n_coord = idx2crd(int(work_tile_info.N_idx), shape<2>(gB_nkl));
        auto l_coord = idx2crd(int(work_tile_info.L_idx), shape<4>(gB_nkl));
        auto output_tile_coord = make_coord(m_coord, n_coord, _, l_coord);

        TiledMma tiled_mma;
        Tensor accumulators = partition_fragment_C(
            tiled_mma, take<0,2>(blk_shape));

        order_barrier.wait();
        collective_mma.mma(
            load_inputs, output_tile_coord, accumulators, k_tile_count,
            consumer_thread_idx, mainloop_smem, pipeline_data, smem_pipe_read);
        order_barrier.arrive();

        constexpr int BLK_M_RANK = rank<0>(blk_shape);
        bool m_oob = int(work_tile_info.M_idx) >= size<2>(gA_mkl);
        auto m_max_coord = unwrap(mute::transform(
            make_seq<BLK_M_RANK>{}, [&](auto i) {
              return m_oob ? 0
                           : get<i>(M) - get<0,i>(blk_shape) * get<i>(m_coord);
            }));

        constexpr int BLK_N_RANK = rank<1>(blk_shape);
        bool n_oob = int(work_tile_info.N_idx) >= size<2>(gB_nkl);
        auto n_max_coord = unwrap(mute::transform(
            make_seq<BLK_N_RANK>{}, [&](auto i) {
              return n_oob ? 0
                           : get<i>(N) - get<1,i>(blk_shape) * get<i>(n_coord);
            }));
        auto residue_mnk = make_tuple(m_max_coord, n_max_coord, Int<0>{});

        order_barrier.wait();
        CollectiveEpilogue epilogue = [&]() {
          if constexpr (UsesTmeEpilogue) {
            return CollectiveEpilogue{params.epilogue, epilogue_barrier_base};
          }
          else {
            return CollectiveEpilogue{params.epilogue};
          }
        }();
        epilogue(
            problem_shape_MNKL, blk_shape, output_tile_coord, accumulators,
            tiled_mma, residue_mnk, consumer_thread_idx, epilogue_smem);
        order_barrier.arrive();

        scheduler.advance_to_next_work(NumConsumerGroups);
        smem_pipe_read.advance(uint32_t(k_tile_count) * (NumConsumerGroups - 1));
      }
      if constexpr (UsesTmeEpilogue) {
        CollectiveEpilogue epilogue{params.epilogue, epilogue_barrier_base};
        epilogue.tail(consumer_thread_idx);
      }
    }
#else
    MUTE_INVALID_CONTROL_PATH("MP32 ping-pong GEMM requires MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass::gemm::kernel

/////////////////////////////////////////////////////////////////////////////////////////////////
