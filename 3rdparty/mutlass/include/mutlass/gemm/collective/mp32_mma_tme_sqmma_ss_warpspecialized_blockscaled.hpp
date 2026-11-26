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

#include <cstdint>

#include "mutlass/mutlass.h"
#include "mutlass/float_subbyte.h"
#include "mutlass/gemm/dispatch_policy.hpp"
#include "mutlass/pipeline/mp32_pipeline.hpp"

#include "mute/atom/copy_atom.hpp"
#include "mute/atom/mma_atom.hpp"
#include "mute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef MP32_POOL_TME_A_INNER_HINT
#define MP32_POOL_TME_A_INNER_HINT 2
#endif
#ifndef MP32_POOL_TME_A_OUTER_HINT
#define MP32_POOL_TME_A_OUTER_HINT 2
#endif
#ifndef MP32_POOL_TME_B_INNER_HINT
#define MP32_POOL_TME_B_INNER_HINT 2
#endif
#ifndef MP32_POOL_TME_B_OUTER_HINT
#define MP32_POOL_TME_B_OUTER_HINT 2
#endif

namespace mutlass::gemm::collective {

using namespace mute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages_,
  class KernelSchedule,
  class TileShape_,
  class ElementPairA_,
  class StridePairA_,
  class ElementPairB_,
  class StridePairB_,
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
    MainloopMp32TmeSqmmaWarpSpecializedBlockScaled<Stages_, KernelSchedule>,
    TileShape_,
    ElementPairA_,
    StridePairA_,
    ElementPairB_,
    StridePairB_,
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
      MainloopMp32TmeSqmmaWarpSpecializedBlockScaled<Stages_, KernelSchedule>;
  using TileShape = TileShape_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using ElementA = tuple_element_t<0, ElementPairA_>;
  using ElementSFA = tuple_element_t<1, ElementPairA_>;
  using ElementB = tuple_element_t<0, ElementPairB_>;
  using ElementSFB = tuple_element_t<1, ElementPairB_>;
  using StrideA = tuple_element_t<0, StridePairA_>;
  using StrideB = tuple_element_t<0, StridePairB_>;

  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using LayoutSFA = tuple_element_t<1, StridePairA_>;
  using LayoutSFB = tuple_element_t<1, StridePairB_>;
  using TransformA = TransformA_;
  using TransformB = TransformB_;

  static constexpr int NumTCEWarps =
      size(TiledMma{}) / mutlass::NumThreadsPerWarp;
  static constexpr int NumTCEWarpSquads =
      size(TiledMma{}) / mutlass::NumThreadsPerWarpSquad;

  static_assert(rank(SmemLayoutAtomA{}) == 2,
                "SmemLayoutAtomA must be rank 2");
  static_assert(rank(SmemLayoutAtomB{}) == 2,
                "SmemLayoutAtomB must be rank 2");
  static_assert(sizeof_bits_v<ElementSFA> == 8 && sizeof_bits_v<ElementSFB> == 8,
                "MP32 block-scaled scale factors must occupy one byte");
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
      "MP32 SS block-scaled SQMMA must source A and B from SMEM descriptors");
  static_assert(is_same_v<GmemTiledCopyA, MP31_TME_LOAD> &&
                is_same_v<GmemTiledCopyB, MP31_TME_LOAD>,
                "MP32 block-scaled mainloop requires MP31_TME_LOAD tags");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<Stages_>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<Stages_>{})));

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

  using ScaleShape = Shape<int32_t, Shape<_32, int32_t>, int32_t>;
  using ScaleStride = Stride<_1, Stride<_0, int64_t>, int64_t>;
  using ScaleLayout = Layout<ScaleShape, ScaleStride>;

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
    ElementSFA const* ptr_SFA{nullptr};
    ElementSFB const* ptr_SFB{nullptr};
  };

  struct ScaleInputs {
    ElementSFA const* ptr_A;
    ElementSFB const* ptr_B;
    // One byte per 32 K elements; each scale group stores contiguous M/N values.
    int32_t m, n, groups;
  };

  struct Params {
    using TME_A = decltype(make_tme_copy<
        TmeInternalElementA,
        static_cast<TME::CacheHint>(MP32_POOL_TME_A_INNER_HINT),
        static_cast<TME::CacheHint>(MP32_POOL_TME_A_OUTER_HINT)>(
        GmemTiledCopyA{},
        make_tensor(static_cast<ElementA const*>(nullptr),
                    repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        take<0,2>(SmemLayoutA{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}))));
    using TME_B = decltype(make_tme_copy<
        TmeInternalElementB,
        static_cast<TME::CacheHint>(MP32_POOL_TME_B_INNER_HINT),
        static_cast<TME::CacheHint>(MP32_POOL_TME_B_OUTER_HINT)>(
        GmemTiledCopyB{},
        make_tensor(static_cast<ElementB const*>(nullptr),
                    repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        take<0,2>(SmemLayoutB{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}))));

    TME_A tme_load_a;
    TME_B tme_load_b;
    ScaleInputs scales;
  };

  MUTE_HOST_DEVICE static constexpr ScaleLayout
  make_scale_layout(int32_t mn, int32_t k, int32_t l) {
    int32_t groups = mute::ceil_div(k, 32);
    return make_layout(
        make_shape(mn, make_shape(_32{}, groups), l),
        make_stride(_1{}, make_stride(_0{}, int64_t(mn)),
                    int64_t(mn) * groups));
  }

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void)workspace;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    int32_t m = static_cast<int32_t>(size(M));
    int32_t n = static_cast<int32_t>(size(N));
    int32_t k = static_cast<int32_t>(size(K));
    int32_t groups = mute::ceil_div(k, 32);

    Tensor tensor_a = make_tensor(
        args.ptr_A, make_layout(make_shape(M, K, L), args.dA));
    Tensor tensor_b = make_tensor(
        args.ptr_B, make_layout(make_shape(N, K, L), args.dB));

    typename Params::TME_A tme_load_a = make_tme_copy<
        TmeInternalElementA,
        static_cast<TME::CacheHint>(MP32_POOL_TME_A_INNER_HINT),
        static_cast<TME::CacheHint>(MP32_POOL_TME_A_OUTER_HINT)>(
        GmemTiledCopyA{}, tensor_a, take<0,2>(SmemLayoutA{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})));
    typename Params::TME_B tme_load_b = make_tme_copy<
        TmeInternalElementB,
        static_cast<TME::CacheHint>(MP32_POOL_TME_B_INNER_HINT),
        static_cast<TME::CacheHint>(MP32_POOL_TME_B_OUTER_HINT)>(
        GmemTiledCopyB{}, tensor_b, take<0,2>(SmemLayoutB{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})));

    return {tme_load_a, tme_load_b,
            {args.ptr_SFA, args.ptr_SFB, m, n, groups}};
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
    if constexpr (sizeof_bits_v<ElementA> == 4 ||
                  sizeof_bits_v<ElementB> == 4) {
      implementable &= k % 8 == 0;
    }

    implementable &= args.ptr_A != nullptr && args.ptr_B != nullptr &&
                     args.ptr_SFA != nullptr && args.ptr_SFB != nullptr;

    constexpr int MinAlignedElementsA = 32 / sizeof_bits_v<ElementA>;
    constexpr int MinAlignedElementsB = 32 / sizeof_bits_v<ElementB>;
    implementable &= mutlass::detail::check_alignment<MinAlignedElementsA>(
        make_shape(M, K, L), StrideA{});
    implementable &= mutlass::detail::check_alignment<MinAlignedElementsB>(
        make_shape(N, K, L), StrideB{});

    if (!implementable) {
      MUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: MP32 block-scaled arguments are not TME compatible.\n");
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
    return make_tuple(gA_mkl, gB_nkl, params.scales);
  }

  template <
    class TensorA, class TensorB,
    class BlockCoord, class KTileIterator>
  MUTLASS_DEVICE void
  load(
      tuple<TensorA, TensorB, ScaleInputs> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter,
      int k_tile_count,
      char* shared_memory,
      Params const& params,
      MainloopPipelineData& pipeline,
      PipelineState& smem_pipe_write) {
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_A.begin()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_B.begin()), SmemLayoutB{});

    auto cta_tme_a = params.tme_load_a.get_slice(0);
    auto cta_tme_b = params.tme_load_b.get_slice(0);

    auto const& gA_mkl = get<0>(load_inputs);
    auto const& gB_nkl = get<1>(load_inputs);
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

  template <
    class TensorA, class TensorB,
    class BlockCoord, class FrgTensorC>
  MUTLASS_DEVICE void
  mma(
      tuple<TensorA, TensorB, ScaleInputs> const& load_inputs,
      BlockCoord const& blk_coord,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      char* shared_memory,
      MainloopPipelineData& pipeline,
      PipelineState& smem_pipe_read) {
    static_assert(is_rmem<FrgTensorC>::value,
                  "C tensor must be register resident");

    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_A.begin()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_B.begin()), SmemLayoutB{});
    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thread_mma.make_fragment_A(thread_mma.partition_A(sA));
    Tensor tCrB = thread_mma.make_fragment_B(thread_mma.partition_B(sB));
    constexpr int InstK = size<2>(typename TiledMma::AtomShape_MNK{});
    static_assert(size<2>(tCrA) == 1 && size<2>(tCrB) == 1,
                  "MP32 SS block-scaled mainloop requires one K slice per stage");
    static_assert(InstK % 32 == 0);

    auto const& sf = get<2>(load_inputs);
    int m_tile = int(crd2idx(get<0>(blk_coord), shape<2>(get<0>(load_inputs))));
    int n_tile = int(crd2idx(get<1>(blk_coord), shape<2>(get<1>(load_inputs))));
    int batch = int(crd2idx(get<3>(blk_coord), shape<4>(get<0>(load_inputs))));
    uint64_t batch_size_a = uint64_t(sf.m) * sf.groups;
    uint64_t batch_size_b = uint64_t(sf.n) * sf.groups;
    auto batch_a = sf.ptr_A + uint64_t(batch) * batch_size_a;
    auto batch_b = sf.ptr_B + uint64_t(batch) * batch_size_b;
    // Bound tail loads to this batch so another batch's NaN scales cannot leak in.
    auto copy_sfa = Copy_Atom<MP31_ROBUST_LOAD<uint32_t>, ElementSFA>{}.with(
        make_robust_desc(batch_a, batch_size_a));
    auto copy_sfb = Copy_Atom<MP31_ROBUST_LOAD<uint32_t>, ElementSFB>{}.with(
        make_robust_desc(batch_b, batch_size_b));

    Tensor mSFA = make_tensor(make_gmem_ptr(batch_a),
        make_layout(make_shape(sf.m, make_shape(_32{}, sf.groups)),
                    make_stride(_1{}, make_stride(_0{}, int64_t(sf.m)))));
    Tensor mSFB = make_tensor(make_gmem_ptr(batch_b),
        make_layout(make_shape(sf.n, make_shape(_32{}, sf.groups)),
                    make_stride(_1{}, make_stride(_0{}, int64_t(sf.n)))));
    Tensor gSFA = local_tile(mSFA, select<0,2>(TileShape{}), make_coord(m_tile, _));
    Tensor gSFB = local_tile(mSFB, select<1,2>(TileShape{}), make_coord(n_tile, _));
    // (SF_VALUES, MMA_M/N, MMA_K, K_TILE).
    Tensor tCgSFA = MP32::BLOCKSCALED::partition_SFA(gSFA, thread_mma);
    Tensor tCgSFB = MP32::BLOCKSCALED::partition_SFB(gSFB, thread_mma);

    // Match A/B's MMA modes: (SF_VALUES, MMA_M/N, MMA_K).
    auto fragment_layout_a = make_fragment_like(take<0,3>(tCgSFA.layout()));
    auto fragment_layout_b = make_fragment_like(take<0,3>(tCgSFB.layout()));
    static_assert(cosize(decltype(fragment_layout_a){}) % 4 == 0);
    static_assert(cosize(decltype(fragment_layout_b){}) % 4 == 0);
    auto packed_layout_a = recast_layout<ElementSFA, uint32_t>(fragment_layout_a);
    auto packed_layout_b = recast_layout<ElementSFB, uint32_t>(fragment_layout_b);

    // Recast the layout before allocation so both buffers retain dword storage.
    Tensor tCrSFA_storage = make_tensor<uint32_t>(
        append(packed_layout_a, make_layout(_2{}, cosize(packed_layout_a))));
    Tensor tCrSFB_storage = make_tensor<uint32_t>(
        append(packed_layout_b, make_layout(_2{}, cosize(packed_layout_b))));
    Tensor tCrSFA = make_tensor(recast_ptr<ElementSFA>(tCrSFA_storage.data()),
        append(fragment_layout_a, make_layout(_2{}, cosize(fragment_layout_a))));
    Tensor tCrSFB = make_tensor(recast_ptr<ElementSFB>(tCrSFB_storage.data()),
        append(fragment_layout_b, make_layout(_2{}, cosize(fragment_layout_b))));

    auto load_scales = [&](auto buffer) {
      // Copy_Atom recurses through SF_VALUES into dword loads, filtering broadcasts.
      copy(copy_sfa, tCgSFA(_, _, _, _0{}), tCrSFA(_, _, _, buffer));
      copy(copy_sfb, tCgSFB(_, _, _, _0{}), tCrSFB(_, _, _, buffer));
      tCgSFA = domain_offset(make_coord(_0{}, _0{}, _0{}, _1{}), tCgSFA);
      tCgSFB = domain_offset(make_coord(_0{}, _0{}, _0{}, _1{}), tCgSFB);
    };
    auto issue_stage = [&](auto const& sf_a, auto const& sf_b) {
      pipeline.consumer_wait(smem_pipe_read);
      int read_stage = smem_pipe_read.index();
      mute::gemm(tiled_mma,
                 make_zip_tensor(tCrA(_, _, _, read_stage), sf_a),
                 make_zip_tensor(tCrB(_, _, _, read_stage), sf_b), accum);
      warpsquad_commit_batch();
      warpsquad_wait();
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
    };

    load_scales(_0{});
    int tile = 1;
    MUTLASS_PRAGMA_NO_UNROLL
    for (; tile + 1 < k_tile_count; tile += 2) {
      load_scales(_1{});
      issue_stage(tCrSFA(_, _, _, _0{}), tCrSFB(_, _, _, _0{}));
      load_scales(_0{});
      issue_stage(tCrSFA(_, _, _, _1{}), tCrSFB(_, _, _, _1{}));
    }
    // Finish the final stages, including wait and release, before the epilogue.
    if (tile < k_tile_count) {
      load_scales(_1{});
      issue_stage(tCrSFA(_, _, _, _0{}), tCrSFB(_, _, _, _0{}));
      load_scales(_0{});
      issue_stage(tCrSFA(_, _, _, _1{}), tCrSFB(_, _, _, _1{}));
    } else if (tile == k_tile_count) {
      load_scales(_1{});
      issue_stage(tCrSFA(_, _, _, _0{}), tCrSFB(_, _, _, _0{}));
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
