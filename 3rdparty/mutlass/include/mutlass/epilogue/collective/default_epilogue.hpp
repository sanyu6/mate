/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/*! \file
  \brief Functor performing elementwise operations used by epilogues.
*/

#pragma once

#include "mutlass/mutlass.h"
#include "mutlass/gemm/dispatch_policy.hpp"
#include "mutlass/epilogue/collective/detail.hpp"

#include "mute/tensor.hpp"
#include "mute/numeric/numeric_types.hpp"
#include "mutlass/musa_host_adapter.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies an element wise operation to all elements within the fragment
/// and writes them out to destination storage.
template <
  class StrideC_,
  class StrideD_,
  class ThreadEpilogueOp_,
  class EpilogueSchedule_
>
class DefaultEpilogue {
public:
  //
  // Type Aliases
  //
  using EpilogueSchedule = EpilogueSchedule_;
  using DispatchPolicy = EpilogueSchedule_;

  // derived types of output thread level operator
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementCompute = typename ThreadEpilogueOp::ElementCompute;
  using ElementScalar = ElementCompute;
  using ElementC = typename ThreadEpilogueOp::ElementC;
  using StrideC = StrideC_;
  using ElementD = typename ThreadEpilogueOp::ElementD;
  using StrideD = StrideD_;

  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  static const int kOutputAlignment = ThreadEpilogueOp::kCount;
  using AlignmentType = typename mute::uint_bit<sizeof_bits<ElementOutput>::value * kOutputAlignment>::type;
  static constexpr int NumBarriers = 0;

  static_assert(mute::rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(mute::rank(StrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  struct SharedStorage { };

  using TensorStorage = SharedStorage;

  // Host side epilogue arguments
  struct Arguments {
    typename ThreadEpilogueOp::Params thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
  };

  // Device side epilogue params
  using Params = Arguments;

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      [[maybe_unused]] ProblemShape const& _,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static mutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, musaStream_t stream,
    MusaHostAdapter* musa_adapter = nullptr) {
    return mutlass::Status::kSuccess;
  }

  template<class ProblemShape>
  MUTLASS_HOST_DEVICE static bool
  can_implement(
      [[maybe_unused]] ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    return true;
  }

  // Note: SharedStorage is unused for DefaultEpilogue
  MUTLASS_HOST_DEVICE
  DefaultEpilogue(Params const& params_, SharedStorage const& shared_storage = SharedStorage())
      : params(params_), epilogue_op(params_.thread) { }

  MUTLASS_DEVICE
  bool
  is_source_needed() {
    return epilogue_op.is_source_needed();
  }

  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK
  >
  MUTLASS_HOST_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      mute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      [[maybe_unused]] char* smem_buf)
  {
    using namespace mute;
    using X = Underscore;

    static_assert(mute::rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
    static_assert(mute::rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
    static_assert(mute::rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 3");

    // Separate out problem shape for convenience
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);

    auto stride_c = detail::get_epilogue_stride<EpilogueSchedule>(params.dC);
    auto stride_d = detail::get_epilogue_stride<EpilogueSchedule>(params.dD);

    constexpr bool IsTransposed =
        mute::is_base_of_v<mutlass::gemm::EpilogueTransposed,
                           EpilogueSchedule>;
    // Shape and coordinates already follow the MMA operand order.
    // A transposed epilogue only swaps the public C/D strides.
    Tensor mC_mnl = make_tensor(make_gmem_ptr(params.ptr_C), make_shape(M, N, L), stride_c);
    Tensor mD_mnl = make_tensor(make_gmem_ptr(params.ptr_D), make_shape(M, N, L), stride_d);
    Tensor gC_mnl = local_tile(mC_mnl, blk_shape_MNK, make_coord(_, _, _), Step<_1, _1, X>{});
    Tensor gD_mnl = local_tile(mD_mnl, blk_shape_MNK, make_coord(_, _, _), Step<_1, _1, X>{});

    // Slice to get the tile this CTA is responsible for
    auto m_coord = get<0>(blk_coord_mnkl);
    auto n_coord = get<1>(blk_coord_mnkl);
    auto l_coord = get<3>(blk_coord_mnkl);
    auto output_residue = make_coord(get<0>(residue_mnk), get<1>(residue_mnk));
    Tensor gC = gC_mnl(_, _, m_coord, n_coord, l_coord);
    Tensor gD = gD_mnl(_, _, m_coord, n_coord, l_coord);

    // Partition source and destination tiles to match the accumulator partitioning
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCgD = thr_mma.partition_C(gD);                                       // (VEC,THR_M,THR_N)
    Tensor tCgC = thr_mma.partition_C(gC);                                       // (VEC,THR_M,THR_N)

    static_assert(is_static<FrgLayout>::value, "Accumulator layout must be static");
    MUTE_STATIC_ASSERT_V(size(tCgC) == size(tCgD),
        "Source and destination must have the same number of elements.");
    MUTE_STATIC_ASSERT_V(size(tCgD) == size(accumulators),
        "Accumulator count must have the same destination element count.");

    // Make an identity coordinate tensor for predicating our output MN tile
    auto cD = make_identity_tensor(make_shape(unwrap(shape<0>(gD)), unwrap(shape<1>(gD))));
    Tensor tCcD = thr_mma.partition_C(cD);

    // Coalesce the coordinate layout to get the leading run of output rows.
    // Visit its columns first; the current MP32 layout gives step = 32.
    using CoordLayout = typename decltype(tCcD)::layout_type;
    constexpr int step = detail::is_n_major<StrideD>()
        ? int(size<0>(coalesce(CoordLayout{})))
        : int(size(FrgLayout{}));

    bool const full_tile =
        get<0>(output_residue) >= get<0>(blk_shape_MNK) &&
        get<1>(output_residue) >= get<1>(blk_shape_MNK);

    if (epilogue_op.is_source_needed()) {
      if constexpr (!IsTransposed) {
        if (full_tile) {
          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(accumulators); ++i) {
            tCgD(i) = epilogue_op(accumulators(i), tCgC(i));
          }
        }
        else {
          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(accumulators); ++i) {
            if (elem_less(tCcD(i), output_residue)) {
              tCgD(i) = epilogue_op(accumulators(i), tCgC(i));
            }
          }
        }
      }
      else {
        if (full_tile) {
          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < step; ++i) {
            MUTLASS_PRAGMA_UNROLL
            for (int j = i; j < size(accumulators); j += step) {
              tCgD(j) = epilogue_op(accumulators(j), tCgC(j));
            }
          }
        }
        else {
          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < step; ++i) {
            MUTLASS_PRAGMA_UNROLL
            for (int j = i; j < size(accumulators); j += step) {
              if (elem_less(tCcD(j), output_residue)) {
                tCgD(j) = epilogue_op(accumulators(j), tCgC(j));
              }
            }
          }
        }
      }
    }
    else {
      if constexpr (!IsTransposed) {
        if (full_tile) {
          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(accumulators); ++i) {
            tCgD(i) = epilogue_op(accumulators(i));
          }
        }
        else {
          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(accumulators); ++i) {
            if (elem_less(tCcD(i), output_residue)) {
              tCgD(i) = epilogue_op(accumulators(i));
            }
          }
        }
      }
      else {
        if (full_tile) {
          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < step; ++i) {
            MUTLASS_PRAGMA_UNROLL
            for (int j = i; j < size(accumulators); j += step) {
              tCgD(j) = epilogue_op(accumulators(j));
            }
          }
        }
        else {
          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < step; ++i) {
            MUTLASS_PRAGMA_UNROLL
            for (int j = i; j < size(accumulators); j += step) {
              if (elem_less(tCcD(j), output_residue)) {
                tCgD(j) = epilogue_op(accumulators(j));
              }
            }
          }
        }
      }
    }
  }

private:
  Params params;
  ThreadEpilogueOp epilogue_op;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace mutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
