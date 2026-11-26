/***************************************************************************************************
 * Copyright (c) 2024 - 2026 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "mutlass/mutlass.h"
#include "mutlass/kernel_hardware_info.hpp"
#include "mutlass/epilogue/collective/detail.hpp"
#include "mutlass/gemm/kernel/tile_scheduler.hpp"
#include "mutlass/gemm/gemm.h"
#include "mutlass/gemm/dispatch_policy.hpp"
#include "mutlass/trace.h"
#include "mutlass/arch/barrier.hpp"

#include "mute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass::gemm::kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Arguments retain public coordinates. Device mainloop and scheduling use
// RS coordinates; output strides are transposed only when SwapAB is enabled.
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
      mute::is_base_of_v<KernelTmeWarpSpecializedBlockScaledMixedInputMp32,
                         typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
{
public:
  using ProblemShape = ProblemShape_;

  static_assert(rank(ProblemShape{}) == 3 || rank(ProblemShape{}) == 4,
                "ProblemShape must be <M,N,K> or <M,N,K,L>");

  using CollectiveMainloop = CollectiveMainloop_;
  static constexpr bool SwapAB = CollectiveMainloop::SwapAB;
  using TileShape = typename CollectiveMainloop::TileShape;
  using InternalTileShape = typename CollectiveMainloop::SwappedTileShape;
  using InternalProblemShape = mute::conditional_t<SwapAB,
      decltype(select<1, 0, 2, 3>(append<4>(ProblemShape{}, mute::Int<1>{}))),
      decltype(append<4>(ProblemShape{}, mute::Int<1>{}))>;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ElementA = typename CollectiveMainloop::ElementA;
  using StrideA = typename CollectiveMainloop::StrideA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using StrideB = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  static constexpr int NumTCEWarps = size(TiledMma{}) / mutlass::NumThreadsPerWarp;
  static constexpr int NumTCEWarpSquads = size(TiledMma{}) / mutlass::NumThreadsPerWarpSquad;
  using MainloopPipelineData = typename CollectiveMainloop::MainloopPipelineData;
  using PipelineDataParams = typename MainloopPipelineData::Params;
  using PipelineState = typename MainloopPipelineData::PipelineState;

  static_assert(ArchTag::kMinComputeCapability == 32);

  using CollectiveEpilogue = CollectiveEpilogue_;
  static constexpr bool UsesTmeEpilogue =
      epilogue::collective::detail::IsMp32TmeEpilogue<CollectiveEpilogue>::value;
  static_assert(!UsesTmeEpilogue || !SwapAB,
    "MP32 RS transposed TME epilogue is not supported; use NoSmemTransposed");
  static_assert(UsesTmeEpilogue ||
    (SwapAB == mute::is_base_of_v<EpilogueTransposed, typename CollectiveEpilogue::DispatchPolicy>),
    "MP32 RS default epilogue transpose must match SwapAB");
  using ElementC = typename CollectiveEpilogue::ElementC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  // Public C/D strides remain in (M, N, L), even when the epilogue uses (N, M, L).
  using StrideC = mute::conditional_t<UsesTmeEpilogue && SwapAB,
      decltype(mute::select<1, 0, 2>(typename CollectiveEpilogue::StrideC{})),
      typename CollectiveEpilogue::StrideC>;
  using StrideD = mute::conditional_t<UsesTmeEpilogue && SwapAB,
      decltype(mute::select<1, 0, 2>(typename CollectiveEpilogue::StrideD{})),
      typename CollectiveEpilogue::StrideD>;
  struct PublicEpilogueArguments {
    typename CollectiveEpilogue::ThreadEpilogueOp::Params thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
  };
  using EpilogueArguments = mute::conditional_t<UsesTmeEpilogue,
      PublicEpilogueArguments, typename CollectiveEpilogue::Arguments>;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static_assert(mute::is_void_v<TileScheduler_> ||
                mute::is_same_v<TileScheduler_, PersistentScheduler>,
                "This kernel only supports the persistent tile scheduler");

  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
      PersistentScheduler, ArchTag, InternalTileShape, ClusterShape>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr int EpilogueSmemAlignment = alignof(typename CollectiveEpilogue::SharedStorage);
  static constexpr int EpilogueSmemOffset = UsesTmeEpilogue
      ? mute::ceil_div(int(sizeof(typename CollectiveMainloop::SharedStorage)),
                       EpilogueSmemAlignment) * EpilogueSmemAlignment : 0;
  static constexpr int SharedStorageSize = UsesTmeEpilogue
      ? EpilogueSmemOffset + sizeof(typename CollectiveEpilogue::SharedStorage)
      : mute::max(sizeof(typename CollectiveMainloop::SharedStorage),
                  sizeof(typename CollectiveEpilogue::SharedStorage));
  static_assert(SharedStorageSize <= 256 * 1024,
                "MP32 RS mainloop plus TME epilogue exceeds shared-memory capacity");

  static constexpr uint32_t MaxThreadsPerBlock =
      mutlass::NumThreadsPerWarpSquad + size(TiledMma{});
  static_assert(MaxThreadsPerBlock <= 1024,
                "MP32 RS warp-specialized kernel exceeds 1024 threads per CTA");
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int SmemAlignmentBytes =
      mute::max(CollectiveMainloop::SmemAlignmentBytes, EpilogueSmemAlignment);
  static constexpr int EpilogueBarriers = [] {
    if constexpr (UsesTmeEpilogue) {
      return CollectiveEpilogue::NumBarriers;
    }
    else {
      return 0;
    }
  }();
  static constexpr int NumBarriers = CollectiveMainloop::NumBarriers + EpilogueBarriers;
  static_assert(NumBarriers + arch::AsyncBarrier::ReservedAsyncBarrierCount <=
                arch::AsyncBarrier::HardwareMaxNumAsyncTransactionBarriers,
                "MP32 RS mainloop and epilogue exceed available barrier slots");

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
    InternalProblemShape problem_shape{};
    MainloopParams mainloop;
    EpilogueParams epilogue;
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
  };

  MUTLASS_HOST_DEVICE static auto
  epilogue_problem_shape(ProblemShape const& shape) {
    if constexpr (UsesTmeEpilogue && SwapAB) {
      return mute::select<1, 0, 2, 3>(append<4>(shape, mute::Int<1>{}));
    }
    else {
      return shape;
    }
  }

  MUTLASS_HOST_DEVICE static typename CollectiveEpilogue::Arguments
  to_epilogue_arguments(EpilogueArguments const& args) {
    if constexpr (UsesTmeEpilogue && SwapAB) {
      return {args.thread, args.ptr_C, mute::select<1, 0, 2>(args.dC),
              args.ptr_D, mute::select<1, 0, 2>(args.dD)};
    }
    else if constexpr (UsesTmeEpilogue) {
      return {args.thread, args.ptr_C, args.dC, args.ptr_D, args.dD};
    }
    else {
      return args;
    }
  }

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void)workspace;
    KernelHardwareInfo hw_info{args.hw_info.device_id, args.hw_info.sm_count};
    auto problem_shape_MNKL = [&] {
      auto shape = append<4>(args.problem_shape, mute::Int<1>{});
      if constexpr (SwapAB) return select<1, 0, 2, 3>(shape);
      else return shape;
    }();
    auto scheduler_args = args.scheduler;
    if (SwapAB && scheduler_args.raster_order == TileScheduler::RasterOrderOptions::AlongM) {
      scheduler_args.raster_order = TileScheduler::RasterOrderOptions::AlongN;
    }
    else if (SwapAB && scheduler_args.raster_order == TileScheduler::RasterOrderOptions::AlongN) {
      scheduler_args.raster_order = TileScheduler::RasterOrderOptions::AlongM;
    }
    return {
      args.mode,
      problem_shape_MNKL,
      CollectiveMainloop::to_underlying_arguments(
          args.problem_shape, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(
          epilogue_problem_shape(args.problem_shape), to_epilogue_arguments(args.epilogue),
          workspace),
      hw_info,
      TileScheduler::to_underlying_arguments(
          problem_shape_MNKL, InternalTileShape{}, hw_info, scheduler_args)
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
        epilogue_problem_shape(args.problem_shape), to_epilogue_arguments(args.epilogue));
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
        params.scheduler, params.problem_shape, InternalTileShape{}, params.hw_info, args);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  MUTLASS_DEVICE void
  operator()(Params const& params, char* smem_buf) {
    using namespace mute;

    static_assert(rank(StrideA{}) == 3,
                  "StrideA must be rank-3: [M,K,L]");
    static_assert(rank(StrideB{}) == 3,
                  "StrideB must be rank-3: [N,K,L]");
    static_assert(rank(StrideC{}) == 3,
                  "StrideC must be rank-3: [M,N,L]");
    static_assert(rank(StrideD{}) == 3,
                  "StrideD must be rank-3: [M,N,L]");

    int thread_idx = int(threadIdx.x);
    int warp_squad_idx = mutlass::canonical_warp_squad_idx();
    bool is_producer = warp_squad_idx == 0;

    arch::allocate_async_barriers(NumBarriers);

    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto blk_shape = InternalTileShape{};

    CollectiveMainloop collective_mma;
    auto load_inputs = collective_mma.load_init(
        problem_shape_MNKL, params.mainloop);
    static_assert(tuple_size_v<decltype(load_inputs)> >= 2,
                  "load_init must return A and B tensors first");

    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    TileScheduler scheduler{params.scheduler};
    auto k_tile_count = size<3>(gA_mkl);

    PipelineDataParams pipeline_params;
    pipeline_params.transaction_bytes =
        CollectiveMainloop::TmeTransactionBytes;
    pipeline_params.num_consumers = NumTCEWarps;
    pipeline_params.num_producers = 1;
    MainloopPipelineData pipeline_data(pipeline_params);

    PipelineState smem_pipe_read;
    PipelineState smem_pipe_write =
        mutlass::make_producer_start_state<MainloopPipelineData>();

    if constexpr (UsesTmeEpilogue) {
      CollectiveEpilogue::initialize(thread_idx, CollectiveMainloop::NumBarriers);
    }
    __syncthreads();

    auto work_tile_info = scheduler.initial_work_tile_info();
    if (is_producer) {
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
            smem_buf, params.mainloop, pipeline_data, smem_pipe_write);
        scheduler.advance_to_next_work();
      }
    }
    else {
      int consumer_thread_idx = thread_idx - mutlass::NumThreadsPerWarpSquad;
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
            tiled_mma, take<0, 2>(tile_shape(tiled_mma)));

        collective_mma.mma(
            load_inputs, output_tile_coord, accumulators, k_tile_count,
            consumer_thread_idx, smem_buf, pipeline_data, smem_pipe_read);

        constexpr int BLK_M_RANK = rank<0>(blk_shape);
        bool m_oob = int(work_tile_info.M_idx) >= size<2>(gA_mkl);
        auto m_max_coord = unwrap(mute::transform(
            make_seq<BLK_M_RANK>{}, [&](auto i) {
              return m_oob ? 0
                           : get<i>(M) - get<0, i>(blk_shape) * get<i>(m_coord);
            }));

        constexpr int BLK_N_RANK = rank<1>(blk_shape);
        bool n_oob = int(work_tile_info.N_idx) >= size<2>(gB_nkl);
        auto n_max_coord = unwrap(mute::transform(
            make_seq<BLK_N_RANK>{}, [&](auto i) {
              return n_oob ? 0
                           : get<i>(N) - get<1, i>(blk_shape) * get<i>(n_coord);
            }));
        auto residue_mnk = make_tuple(m_max_coord, n_max_coord, Int<0>{});

        if constexpr (UsesTmeEpilogue) {
          CollectiveEpilogue epilogue{params.epilogue, CollectiveMainloop::NumBarriers};
          epilogue(
              problem_shape_MNKL, blk_shape, output_tile_coord, accumulators,
              tiled_mma, residue_mnk, consumer_thread_idx,
              smem_buf + EpilogueSmemOffset);
        }
        else {
          CollectiveEpilogue epilogue{params.epilogue};
          epilogue(
              problem_shape_MNKL, blk_shape, output_tile_coord, accumulators,
              tiled_mma, residue_mnk, consumer_thread_idx, smem_buf);
        }
        scheduler.advance_to_next_work();
      }
      if constexpr (UsesTmeEpilogue) {
        CollectiveEpilogue epilogue{params.epilogue, CollectiveMainloop::NumBarriers};
        epilogue.tail(consumer_thread_idx);
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass::gemm::kernel

/////////////////////////////////////////////////////////////////////////////////////////////////
