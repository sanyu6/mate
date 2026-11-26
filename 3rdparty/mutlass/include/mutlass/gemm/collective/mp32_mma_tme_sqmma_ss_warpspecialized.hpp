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

#include <cstdint>

#include "mutlass/mutlass.h"
#include "mutlass/float_subbyte.h"
#include "mutlass/gemm/dispatch_policy.hpp"
#include "mutlass/pipeline/mp32_pipeline.hpp"

#include "mute/atom/copy_atom.hpp"
#include "mute/atom/mma_atom.hpp"
#include "mute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass::gemm::collective {

using namespace mute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages_,
  class KernelSchedule,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopMp32TmeSqmmaWarpSpecialized<Stages_, KernelSchedule>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
  using DispatchPolicy =
      MainloopMp32TmeSqmmaWarpSpecialized<Stages_, KernelSchedule>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static constexpr int NumTCEWarps =
      size(TiledMma{}) / mutlass::NumThreadsPerWarp;
  static constexpr int NumTCEWarpSquads =
      size(TiledMma{}) / mutlass::NumThreadsPerWarpSquad;

  static_assert(rank(SmemLayoutAtomA{}) == 2,
                "SmemLayoutAtomA must be rank 2");
  static_assert(rank(SmemLayoutAtomB{}) == 2,
                "SmemLayoutAtomB must be rank 2");
  static_assert(size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{}) == 0);
  static_assert(size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{}) == 0);
  static_assert(size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{}) == 0);
  static_assert(size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{}) == 0);
  static_assert(Stages_ >= 2 && Stages_ <= 31,
                "MP32 mainloop stage count must be in [2,31]");

  static_assert(
      is_base_of_v<MP32::SQMMA::DescriptorIterator,
                   typename TiledMma::FrgTypeA> &&
      is_base_of_v<MP32::SQMMA::DescriptorIterator,
                   typename TiledMma::FrgTypeB>,
      "MP32 SS SQMMA must source A and B from SMEM descriptors");
  static_assert(is_same_v<GmemTiledCopyA, MP31_TME_LOAD> &&
                is_same_v<GmemTiledCopyB, MP31_TME_LOAD>,
                "MP32 SS mainloop requires MP31_TME_LOAD tags");
  static_assert(is_void_v<SmemCopyAtomA> && is_void_v<SmemCopyAtomB>,
                "MP32 SS SQMMA does not use SMEM copy atoms");
  static_assert(sizeof_bits_v<ElementA> ==
                    sizeof_bits_v<typename TiledMma::ValTypeA> &&
                sizeof_bits_v<ElementB> ==
                    sizeof_bits_v<typename TiledMma::ValTypeB>,
                "GMEM and SQMMA operand widths must match");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}),
                 Int<Stages_>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}),
                 Int<Stages_>{})));

  static_assert(rank(SmemLayoutA{}) == 3 && rank(SmemLayoutB{}) == 3);

  static constexpr uint32_t TmeTransactionBytesMK = mutlass::bits_to_bytes(
      size(take<0,2>(SmemLayoutA{})) * sizeof_bits_v<ElementA>);
  static constexpr uint32_t TmeTransactionBytesNK = mutlass::bits_to_bytes(
      size(take<0,2>(SmemLayoutB{})) * sizeof_bits_v<ElementB>);
  static constexpr uint32_t TmeTransactionBytes =
      TmeTransactionBytesMK + TmeTransactionBytesNK;

  using MainloopPipelineData = mutlass::PipelineTmeAsync<Stages_>;
  using PipelineDataParams = typename MainloopPipelineData::Params;
  using PipelineState = typename MainloopPipelineData::PipelineState;

  static constexpr int NumBarriers = MainloopPipelineData::NumBarriers;
  static constexpr int SmemAlignmentBytes = 256;

  struct SharedStorage {
    alignas(SmemAlignmentBytes)
    ArrayEngine<typename TiledMma::ValTypeA, cosize_v<SmemLayoutA>> smem_A;
    alignas(SmemAlignmentBytes)
    ArrayEngine<typename TiledMma::ValTypeB, cosize_v<SmemLayoutB>> smem_B;
  };

  static_assert(sizeof(SharedStorage) <= 256 * 1024,
                "MP32 mainloop shared storage exceeds 256 KiB");

  using TmeInternalElementA = conditional_t<
      sizeof_bits_v<ElementA> == 4,
      mutlass::detail::float_e2m1x2_t,
      void>;
  using TmeInternalElementB = conditional_t<
      sizeof_bits_v<ElementB> == 4,
      mutlass::detail::float_e2m1x2_t,
      void>;

  struct Arguments {
    ElementA const* ptr_A{nullptr};
    StrideA dA{};
    ElementB const* ptr_B{nullptr};
    StrideB dB{};
  };

  struct Params {
    using TME_A = decltype(make_tme_copy<TmeInternalElementA>(
        GmemTiledCopyA{},
        make_tensor(static_cast<ElementA const*>(nullptr),
                    repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        take<0,2>(SmemLayoutA{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}))));
    using TME_B = decltype(make_tme_copy<TmeInternalElementB>(
        GmemTiledCopyB{},
        make_tensor(static_cast<ElementB const*>(nullptr),
                    repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        take<0,2>(SmemLayoutB{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}))));

    TME_A tme_load_a;
    TME_B tme_load_b;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void)workspace;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    Tensor tensor_a = make_tensor(
        args.ptr_A, make_layout(make_shape(M, K, L), args.dA));
    Tensor tensor_b = make_tensor(
        args.ptr_B, make_layout(make_shape(N, K, L), args.dB));

    typename Params::TME_A tme_load_a = make_tme_copy<TmeInternalElementA>(
        GmemTiledCopyA{}, tensor_a, take<0,2>(SmemLayoutA{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})));
    typename Params::TME_B tme_load_b = make_tme_copy<TmeInternalElementB>(
        GmemTiledCopyB{}, tensor_b, take<0,2>(SmemLayoutB{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})));

    return {tme_load_a, tme_load_b};
  }

  template <class ProblemShape>
  MUTLASS_HOST_DEVICE static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    int64_t m = static_cast<int64_t>(size(M));
    int64_t n = static_cast<int64_t>(size(N));
    int64_t k = static_cast<int64_t>(size(K));
    int64_t l = static_cast<int64_t>(size(L));

    bool implementable = m > 0 && n > 0 && k > 0 && l > 0;
    implementable &= args.ptr_A != nullptr && args.ptr_B != nullptr;

    constexpr int MinAlignedElementsA = 32 / sizeof_bits_v<ElementA>;
    constexpr int MinAlignedElementsB = 32 / sizeof_bits_v<ElementB>;
    implementable &= mutlass::detail::check_alignment<MinAlignedElementsA>(
        make_shape(M, K, L), StrideA{});
    implementable &= mutlass::detail::check_alignment<MinAlignedElementsB>(
        make_shape(N, K, L), StrideB{});

    if (!implementable) {
      MUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: MP32 SS arguments are not TME compatible.\n");
    }
    return implementable;
  }

  MUTLASS_DEVICE static void
  prefetch_tme_descriptors(Params const& params) {
    prefetch_tme_descriptor(params.tme_load_a.get_tme_descriptor());
    prefetch_tme_descriptor(params.tme_load_b.get_tme_descriptor());
  }

  template <class ProblemShape_MNKL>
  MUTLASS_DEVICE auto
  load_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      Params const& params) const {
    using X = Underscore;
    auto [M, N, K, L] = problem_shape_MNKL;

    Tensor mA_mkl = params.tme_load_a.get_tme_tensor(make_shape(M, K, L));
    Tensor mB_nkl = params.tme_load_b.get_tme_tensor(make_shape(N, K, L));

    Tensor gA_mkl = local_tile(
        mA_mkl, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
    Tensor gB_nkl = local_tile(
        mB_nkl, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});

    return make_tuple(gA_mkl, gB_nkl);
  }

  template <class TensorA, class TensorB, class BlockCoord, class KTileIterator>
  MUTLASS_DEVICE void
  load(
      tuple<TensorA, TensorB> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter,
      int k_tile_count,
      char* shared_memory,
      Params const& params,
      MainloopPipelineData& pipeline,
      PipelineState& smem_pipe_write) {
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(
        make_smem_ptr(storage.smem_A.begin()), SmemLayoutA{});
    Tensor sB = make_tensor(
        make_smem_ptr(storage.smem_B.begin()), SmemLayoutB{});

    auto cta_tme_a = params.tme_load_a.get_slice(0);
    auto cta_tme_b = params.tme_load_b.get_slice(0);

    auto [gA_mkl, gB_nkl] = load_inputs;
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    Tensor gA = gA_mkl(_, _, m_coord, _, l_coord);
    Tensor gB = gB_nkl(_, _, n_coord, _, l_coord);

    Tensor tAgA = cta_tme_a.partition_S(gA);
    Tensor tAsA = cta_tme_a.partition_D(sA);
    Tensor tBgB = cta_tme_b.partition_S(gB);
    Tensor tBsB = cta_tme_b.partition_D(sB);

    int warp_idx = canonical_warp_idx() & 3;
    MUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 0; --k_tile_count) {
      if (warp_idx == 0) {
        pipeline.producer_acquire(smem_pipe_write);
        uint32_t bar_id = pipeline.producer_get_barrier_id(smem_pipe_write);
        int write_stage = smem_pipe_write.index();
        copy(params.tme_load_a.with(bar_id),
             tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, write_stage));
        copy(params.tme_load_b.with(bar_id),
             tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, write_stage));
      }
      ++smem_pipe_write;
      ++k_tile_iter;
    }
  }

  template <class TensorA, class TensorB, class BlockCoord, class FrgTensorC>
  MUTLASS_DEVICE void
  mma(
      tuple<TensorA, TensorB> const&,
      BlockCoord const&,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      char* shared_memory,
      MainloopPipelineData& pipeline,
      PipelineState& smem_pipe_read) {
    static_assert(is_rmem<FrgTensorC>::value,
                  "C tensor must be register resident");

    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(
        make_smem_ptr(storage.smem_A.begin()), SmemLayoutA{});
    Tensor sB = make_tensor(
        make_smem_ptr(storage.smem_B.begin()), SmemLayoutB{});

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCsA = thread_mma.partition_A(sA);
    Tensor tCsB = thread_mma.partition_B(sB);
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);

    MUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));

    // Keep the first MMA separate so a zero accumulator selects non-accumulating SQMMA.
    {
      pipeline.consumer_wait(smem_pipe_read);
      int read_stage = smem_pipe_read.index();

      MUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        mute::gemm(tiled_mma,
                   tCrA(_, _, k_block, read_stage),
                   tCrB(_, _, k_block, read_stage),
                   accum);
      }

      warpsquad_commit_batch();
      warpsquad_wait();
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
      --k_tile_count;
    }

    MUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 0; --k_tile_count) {
      pipeline.consumer_wait(smem_pipe_read);
      int read_stage = smem_pipe_read.index();

      MUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        mute::gemm(tiled_mma,
                   tCrA(_, _, k_block, read_stage),
                   tCrB(_, _, k_block, read_stage),
                   accum);
      }

      warpsquad_commit_batch();
      warpsquad_wait();
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
