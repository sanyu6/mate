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
#include "mutlass/gemm/collective/collective_mma_decl.hpp"
#include "mutlass/gemm/dispatch_policy.hpp"
#include "mutlass/numeric_conversion.h"
#include "mutlass/pipeline/pipeline.hpp"

#include "mute/arch/copy_mp31.hpp"
#include "mute/atom/copy_atom.hpp"
#include "mute/atom/copy_traits_mp31_tme.hpp"
#include "mute/atom/mma_atom.hpp"
#include "mute/atom/mma_traits_mp32_sqmma.hpp"
#include "mute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass::gemm::collective {

using namespace mute;

namespace detail {

MUTLASS_DEVICE Array<half_t, 4>
rs_fp4_to_fp16_x4(Array<float_e2m1_t, 4> const& source) {
  using Converter = NumericArrayConverter<half_t, float_e2m1_t, 4, FloatRoundStyle::round_to_nearest>;
  return Converter::convert(source);
}

template <class ElementDst>
MUTLASS_DEVICE Array<ElementDst, 4>
rs_fp16_to_fp8_x4(Array<half_t, 4> const& source) {
  using Converter = NumericArrayConverter<ElementDst, half_t, 4, FloatRoundStyle::round_to_nearest>;
  return Converter::convert(source);
}

// Load one complete packed A fragment without waiting for it here. Its first
// conversion consumes the load results after the current SQMMA is issued.
template <int Count, class SrcTensor>
MUTLASS_DEVICE void
load_e2m1_packed(Array<float_e2m1_t, 4> (&source)[Count], SrcTensor const& src) {
  static_assert(is_same_v<typename SrcTensor::value_type, float_e2m1_t>,
    "Direct packed conversion requires an E2M1 source fragment");
  auto packed_src = recast<Array<float_e2m1_t, 4>>(src);
  MUTE_STATIC_ASSERT_V(size(packed_src) == Int<Count>{});
  MUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Count; ++i) {
    source[i] = packed_src(i);
  }
}

// Conversion is separate from the shared-memory load so it can overlap the
// SQMMA that still owns the other FP8 A buffer.
template <class ElementDst, class DstTensor, int Count>
MUTLASS_DEVICE void
convert_e2m1_packed_to_fp8(DstTensor& dst, Array<float_e2m1_t, 4> const (&packed_source)[Count]) {
  static_assert(sizeof_bits_v<ElementDst> == 8, "E2M1 conversion destination must contain 8-bit values");

  auto packed_dst = recast<Array<ElementDst, 4>>(dst);
  static constexpr int ParallelCount = 4;
  MUTE_STATIC_ASSERT_V(Int<Count>{} == size(packed_dst));

  MUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(packed_dst); i += ParallelCount) {
    Array<half_t, 4> fp16[ParallelCount];

    MUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < ParallelCount && i + j < size(packed_dst); ++j) {
      fp16[j] = rs_fp4_to_fp16_x4(packed_source[i + j]);
    }

    MUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < ParallelCount && i + j < size(packed_dst); ++j) {
      packed_dst(i + j) = rs_fp16_to_fp8_x4<ElementDst>(fp16[j]);
    }
  }
}

} // namespace detail

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
  class GmemTiledCopyPairA_,
  class SmemLayoutAtomsA_,
  class SmemCopyAtomsA_,
  class TransformA_,
  class GmemTiledCopyPairB_,
  class SmemLayoutAtomsB_,
  class SmemCopyAtomsB_,
  class TransformB_
>
struct CollectiveMma<
  MainloopMp32TmeSqmmaWarpSpecializedBlockScaledMixedInput<Stages_, KernelSchedule>,
  TileShape_,
  ElementPairA_,
  StridePairA_,
  ElementPairB_,
  StridePairB_,
  TiledMma_,
  GmemTiledCopyPairA_,
  SmemLayoutAtomsA_,
  SmemCopyAtomsA_,
  TransformA_,
  GmemTiledCopyPairB_,
  SmemLayoutAtomsB_,
  SmemCopyAtomsB_,
  TransformB_>
{
  using DispatchPolicy = MainloopMp32TmeSqmmaWarpSpecializedBlockScaledMixedInput<Stages_, KernelSchedule>;
  using TileShape = TileShape_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using ElementA = tuple_element_t<0, ElementPairA_>;
  using ElementSFA = tuple_element_t<1, ElementPairA_>;
  using ElementB = tuple_element_t<0, ElementPairB_>;
  using ElementSFB = tuple_element_t<1, ElementPairB_>;
  // The operand requiring FP4-to-FP8 conversion must occupy RS register A.
  static constexpr bool IsATransformed = is_same_v<ElementA, float_e2m1_t>;
  static constexpr bool SwapAB = !IsATransformed;

  using StrideA = tuple_element_t<0, StridePairA_>;
  using StrideB = tuple_element_t<0, StridePairB_>;

  using GmemTiledCopyA = tuple_element_t<0, GmemTiledCopyPairA_>;
  using GmemTiledCopySFA = tuple_element_t<1, GmemTiledCopyPairA_>;
  using GmemTiledCopyB = tuple_element_t<0, GmemTiledCopyPairB_>;
  using GmemTiledCopySFB = tuple_element_t<1, GmemTiledCopyPairB_>;

  using SmemLayoutAtomA = tuple_element_t<0, SmemLayoutAtomsA_>;
  using SmemLayoutAtomSFA = tuple_element_t<1, SmemLayoutAtomsA_>;
  using SmemLayoutAtomB = tuple_element_t<0, SmemLayoutAtomsB_>;
  using SmemLayoutAtomSFB = tuple_element_t<1, SmemLayoutAtomsB_>;
  using SmemCopyAtomSFA = tuple_element_t<1, SmemCopyAtomsA_>;
  using SmemCopyAtomSFB = tuple_element_t<1, SmemCopyAtomsB_>;
  using TransformA = TransformA_;
  using TransformB = TransformB_;

  // Arguments keep public A/B order. Params and device tensors use RS order.
  using SwappedElementA = conditional_t<SwapAB, ElementB, ElementA>;
  using SwappedElementB = conditional_t<SwapAB, ElementA, ElementB>;
  using SwappedElementSFA = conditional_t<SwapAB, ElementSFB, ElementSFA>;
  using SwappedElementSFB = conditional_t<SwapAB, ElementSFA, ElementSFB>;
  using SwappedStrideA = conditional_t<SwapAB, StrideB, StrideA>;
  using SwappedStrideB = conditional_t<SwapAB, StrideA, StrideB>;
  using SwappedSmemLayoutAtomA = conditional_t<SwapAB, SmemLayoutAtomB, SmemLayoutAtomA>;
  using SwappedSmemLayoutAtomB = conditional_t<SwapAB, SmemLayoutAtomA, SmemLayoutAtomB>;
  using SwappedSmemLayoutAtomSFA = conditional_t<SwapAB, SmemLayoutAtomSFB, SmemLayoutAtomSFA>;
  using SwappedSmemLayoutAtomSFB = conditional_t<SwapAB, SmemLayoutAtomSFA, SmemLayoutAtomSFB>;
  using SwappedTileShape = conditional_t<SwapAB,
      decltype(select<1, 0, 2>(TileShape{})), TileShape>;
  using MmaElementA = typename TiledMma::ValTypeA;

  static constexpr int NumTCEWarps = size(TiledMma{}) / mutlass::NumThreadsPerWarp;
  static constexpr int NumTCEWarpSquads = size(TiledMma{}) / mutlass::NumThreadsPerWarpSquad;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtomA must be rank 2");
  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtomB must be rank 2");
  static_assert(rank(SmemLayoutAtomSFA{}) == 2, "SmemLayoutAtomSFA must be rank 2");
  static_assert(rank(SmemLayoutAtomSFB{}) == 2, "SmemLayoutAtomSFB must be rank 2");
  static_assert(size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{}) == 0);
  static_assert(size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{}) == 0);
  static_assert(size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{}) == 0);
  static_assert(size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{}) == 0);
  static_assert(Stages_ >= 2, "MP32 two-stage lookahead requires at least two pipeline stages");

  static_assert(!is_base_of_v<MP32::SQMMA::DescriptorIterator, typename TiledMma::FrgTypeA> &&
    is_base_of_v<MP32::SQMMA::DescriptorIterator, typename TiledMma::FrgTypeB>,
    "MP32 RS block-scaled mixed-input SQMMA requires register A and SMEM-descriptor B");
  static_assert(is_same_v<GmemTiledCopyA, MP31_TME_LOAD> &&
    is_same_v<GmemTiledCopyB, MP31_TME_LOAD> &&
    is_same_v<GmemTiledCopySFA, MP31_TME_LOAD> &&
    is_same_v<GmemTiledCopySFB, MP31_TME_LOAD>,
    "MP32 mixed-input block-scaled mainloop requires MP31_TME_LOAD tags");

  // Shared storage and TME descriptors follow the internal RS operand order.
  using SmemLayoutA = decltype(tile_to_shape(
    SwappedSmemLayoutAtomA{},
    make_shape(shape<0>(SwappedTileShape{}), shape<2>(SwappedTileShape{}), Int<Stages_>{})));
  using SmemLayoutB = decltype(tile_to_shape(
    SwappedSmemLayoutAtomB{},
    make_shape(shape<1>(SwappedTileShape{}), shape<2>(SwappedTileShape{}), Int<Stages_>{})));

  static_assert(size(filter_zeros(SwappedSmemLayoutAtomSFA{})) == cosize(SwappedSmemLayoutAtomSFA{}),
    "SFA stage stride must equal its physical element count");
  static_assert(size(filter_zeros(SwappedSmemLayoutAtomSFB{})) == cosize(SwappedSmemLayoutAtomSFB{}),
    "SFB stage stride must equal its physical element count");

  using SmemLayoutSFA = decltype(make_layout(
    append(shape(SwappedSmemLayoutAtomSFA{}), Int<Stages_>{}),
    append(stride(SwappedSmemLayoutAtomSFA{}), size(filter_zeros(SwappedSmemLayoutAtomSFA{})))));
  using SmemLayoutSFB = decltype(make_layout(
    append(shape(SwappedSmemLayoutAtomSFB{}), Int<Stages_>{}),
    append(stride(SwappedSmemLayoutAtomSFB{}), size(filter_zeros(SwappedSmemLayoutAtomSFB{})))));

  static_assert(rank(SmemLayoutA{}) == 3 && rank(SmemLayoutB{}) == 3);
  static_assert(rank(SmemLayoutSFA{}) == 3 && rank(SmemLayoutSFB{}) == 3);

  static constexpr uint32_t TmeTransactionBytesA =
    mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * sizeof_bits_v<SwappedElementA>);
  static constexpr uint32_t TmeTransactionBytesB =
    mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * sizeof_bits_v<SwappedElementB>);
  static constexpr uint32_t TmeTransactionBytesSFA =
    mutlass::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFA{})) * sizeof_bits_v<SwappedElementSFA>);
  static constexpr uint32_t TmeTransactionBytesSFB =
    mutlass::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFB{})) * sizeof_bits_v<SwappedElementSFB>);
  static constexpr uint32_t TmeTransactionBytes =
    TmeTransactionBytesA + TmeTransactionBytesB + TmeTransactionBytesSFA + TmeTransactionBytesSFB;
  using MainloopPipelineData = mutlass::PipelineTmeAsync<Stages_>;
  using PipelineDataParams = typename MainloopPipelineData::Params;
  using PipelineState = typename MainloopPipelineData::PipelineState;

  static constexpr int NumBarriers = MainloopPipelineData::NumBarriers;
  static_assert(NumBarriers + arch::AsyncBarrier::ReservedAsyncBarrierCount <=
                arch::AsyncBarrier::HardwareMaxNumAsyncTransactionBarriers,
                "MP32 RS mainloop exceeds available barrier slots");
  using SmemSwizzleA = decltype(mute::detail::get_swizzle_portion(SmemLayoutA{}));
  static constexpr int SmemSwizzleAlignmentBytes =
    SmemSwizzleA::num_bits == 0 ? 1 :
    (1 << (SmemSwizzleA::num_base + SmemSwizzleA::num_bits +
           mute::abs(SmemSwizzleA::num_shft)));
  static constexpr int SmemAlignmentBytes = mute::max(256, SmemSwizzleAlignmentBytes);

  struct SharedStorage
  {
    alignas(SmemAlignmentBytes) ArrayEngine<SwappedElementA, cosize_v<SmemLayoutA>> smem_A;
    alignas(SmemAlignmentBytes) ArrayEngine<SwappedElementB, cosize_v<SmemLayoutB>> smem_B;
    alignas(16) ArrayEngine<SwappedElementSFA, cosize_v<SmemLayoutSFA>> smem_SFA;
    alignas(16) ArrayEngine<SwappedElementSFB, cosize_v<SmemLayoutSFB>> smem_SFB;
  };

  static_assert(sizeof(SharedStorage) <= 256 * 1024, "MP32 mainloop shared storage exceeds 256 KiB");

  using TmeInternalElementA = conditional_t<sizeof_bits_v<SwappedElementA> == 4, mutlass::detail::float_e2m1x2_t, void>;
  using TmeInternalElementB = conditional_t<sizeof_bits_v<SwappedElementB> == 4, mutlass::detail::float_e2m1x2_t, void>;

  using ScaleShape = Shape<int32_t, Shape<_32, int32_t>, int32_t>;
  using ScaleStride = Stride<_1, Stride<_0, int64_t>, int64_t>;
  using ScaleLayout = Layout<ScaleShape, ScaleStride>;

  using TmeInternalElementSF = uint16_t;

  struct Arguments
  {
    ElementA const* ptr_A{nullptr};
    StrideA dA{};
    ElementB const* ptr_B{nullptr};
    StrideB dB{};
    ElementSFA const* ptr_SFA{nullptr};
    ElementSFB const* ptr_SFB{nullptr};
  };

  using TME_A = decltype(make_tme_copy<TmeInternalElementA>(
    GmemTiledCopyA{},
    make_tensor(
      static_cast<SwappedElementA const*>(nullptr),
      repeat_like(SwappedStrideA{}, int32_t(0)),
      SwappedStrideA{}),
    take<0, 2>(SmemLayoutA{}),
    make_shape(shape<0>(SwappedTileShape{}), shape<2>(SwappedTileShape{}))));
  using TME_B = decltype(make_tme_copy<TmeInternalElementB>(
    GmemTiledCopyB{},
    make_tensor(
      static_cast<SwappedElementB const*>(nullptr),
      repeat_like(SwappedStrideB{}, int32_t(0)),
      SwappedStrideB{}),
    take<0, 2>(SmemLayoutB{}),
    make_shape(shape<1>(SwappedTileShape{}), shape<2>(SwappedTileShape{}))));
  using TME_SFA = decltype(make_tme_copy<TmeInternalElementSF>(
    GmemTiledCopySFA{},
    make_tensor(static_cast<SwappedElementSFA const*>(nullptr), ScaleLayout{}),
    take<0, 2>(SmemLayoutSFA{}),
    make_shape(shape<0>(SwappedTileShape{}), shape<2>(SwappedTileShape{}))));
  using TME_SFB = decltype(make_tme_copy<TmeInternalElementSF>(
    GmemTiledCopySFB{},
    make_tensor(static_cast<SwappedElementSFB const*>(nullptr), ScaleLayout{}),
    take<0, 2>(SmemLayoutSFB{}),
    make_shape(shape<1>(SwappedTileShape{}), shape<2>(SwappedTileShape{}))));

  struct Params
  {
    TME_A tme_load_a;
    TME_B tme_load_b;
    TME_SFA tme_load_sfa;
    TME_SFB tme_load_sfb;
    ScaleLayout layout_sfa;
    ScaleLayout layout_sfb;
  };

  MUTLASS_HOST_DEVICE static constexpr ScaleLayout
  make_scale_layout(int32_t mn, int32_t k, int32_t l) {
    int32_t groups = mute::ceil_div(k, 32);
    return make_layout(
      make_shape(mn, make_shape(_32{}, groups), l),
      make_stride(_1{}, make_stride(_0{}, int64_t(mn)), int64_t(mn) * groups));
  }

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace) {
    (void)workspace;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = [&] {
      if constexpr (SwapAB) return select<1, 0, 2, 3>(problem_shape_MNKL);
      else return problem_shape_MNKL;
    }();

    auto ptr_a    = mute::conditional_return<SwapAB>(args.ptr_B, args.ptr_A);
    auto stride_a = mute::conditional_return<SwapAB>(args.dB, args.dA);
    auto ptr_b    = mute::conditional_return<SwapAB>(args.ptr_A, args.ptr_B);
    auto stride_b = mute::conditional_return<SwapAB>(args.dA, args.dB);
    auto ptr_sfa  = mute::conditional_return<SwapAB>(args.ptr_SFB, args.ptr_SFA);
    auto ptr_sfb  = mute::conditional_return<SwapAB>(args.ptr_SFA, args.ptr_SFB);

    Tensor tensor_a = make_tensor(ptr_a, make_layout(make_shape(M, K, L), stride_a));
    Tensor tensor_b = make_tensor(ptr_b, make_layout(make_shape(N, K, L), stride_b));
    TME_A tme_load_a = make_tme_copy<TmeInternalElementA>(
      GmemTiledCopyA{},
      tensor_a,
      take<0, 2>(SmemLayoutA{}),
      make_shape(shape<0>(SwappedTileShape{}), shape<2>(SwappedTileShape{})));
    TME_B tme_load_b = make_tme_copy<TmeInternalElementB>(
      GmemTiledCopyB{},
      tensor_b,
      take<0, 2>(SmemLayoutB{}),
      make_shape(shape<1>(SwappedTileShape{}), shape<2>(SwappedTileShape{})));
    auto layout_sfa = make_scale_layout(size(M), size(K), size(L));
    auto layout_sfb = make_scale_layout(size(N), size(K), size(L));
    Tensor tensor_sfa = make_tensor(ptr_sfa, layout_sfa);
    Tensor tensor_sfb = make_tensor(ptr_sfb, layout_sfb);
    TME_SFA tme_load_sfa = make_tme_copy<TmeInternalElementSF>(
      GmemTiledCopySFA{},
      tensor_sfa,
      take<0, 2>(SmemLayoutSFA{}),
      make_shape(shape<0>(SwappedTileShape{}), shape<2>(SwappedTileShape{})));
    TME_SFB tme_load_sfb = make_tme_copy<TmeInternalElementSF>(
      GmemTiledCopySFB{},
      tensor_sfb,
      take<0, 2>(SmemLayoutSFB{}),
      make_shape(shape<1>(SwappedTileShape{}), shape<2>(SwappedTileShape{})));

    return {tme_load_a, tme_load_b, tme_load_sfa, tme_load_sfb, layout_sfa, layout_sfb};
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
    constexpr int MinAlignedElementsSFA = 32 / sizeof_bits_v<ElementSFA>;
    constexpr int MinAlignedElementsSFB = 32 / sizeof_bits_v<ElementSFB>;
    auto scale_k_shape = make_shape(_32{}, mute::ceil_div(k, int64_t(32)));
    implementable &= mutlass::detail::check_alignment<MinAlignedElementsSFA>(
      make_shape(M, scale_k_shape, L), ScaleStride{});
    implementable &= mutlass::detail::check_alignment<MinAlignedElementsSFB>(
      make_shape(N, scale_k_shape, L), ScaleStride{});

    implementable &= args.ptr_A != nullptr && args.ptr_B != nullptr && args.ptr_SFA != nullptr &&
      args.ptr_SFB != nullptr;
    implementable &= (reinterpret_cast<uintptr_t>(args.ptr_A) % 4 == 0) &&
      (reinterpret_cast<uintptr_t>(args.ptr_B) % 4 == 0) &&
      (reinterpret_cast<uintptr_t>(args.ptr_SFA) % 4 == 0) &&
      (reinterpret_cast<uintptr_t>(args.ptr_SFB) % 4 == 0);

    constexpr int MinAlignedElementsA = 32 / sizeof_bits_v<ElementA>;
    constexpr int MinAlignedElementsB = 32 / sizeof_bits_v<ElementB>;
    implementable &= mutlass::detail::check_alignment<MinAlignedElementsA>(make_shape(M, K, L), StrideA{});
    implementable &= mutlass::detail::check_alignment<MinAlignedElementsB>(make_shape(N, K, L), StrideB{});

    if (!implementable) {
      MUTLASS_TRACE_HOST("  CAN IMPLEMENT: MP32 block-scaled arguments are not TME compatible.\n");
    }
    return implementable;
  }

  MUTLASS_DEVICE static void
  prefetch_tme_descriptors(Params const& params) {
    prefetch_tme_descriptor(params.tme_load_a.get_tme_descriptor());
    prefetch_tme_descriptor(params.tme_load_b.get_tme_descriptor());
    prefetch_tme_descriptor(params.tme_load_sfa.get_tme_descriptor());
    prefetch_tme_descriptor(params.tme_load_sfb.get_tme_descriptor());
  }

  template <class ProblemShape_MNKL>
  MUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& params) const {
    using X = Underscore;
    auto [M, N, K, L] = problem_shape_MNKL;

    Tensor mA_mkl = params.tme_load_a.get_tme_tensor(make_shape(M, K, L));
    Tensor mB_nkl = params.tme_load_b.get_tme_tensor(make_shape(N, K, L));

    Tensor gA_mkl = local_tile(mA_mkl, SwappedTileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
    Tensor gB_nkl = local_tile(mB_nkl, SwappedTileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor mSFA = params.tme_load_sfa.get_tme_tensor(shape(params.layout_sfa));
    Tensor mSFB = params.tme_load_sfb.get_tme_tensor(shape(params.layout_sfb));
    Tensor gSFA = local_tile(mSFA, SwappedTileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
    Tensor gSFB = local_tile(mSFB, SwappedTileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
    return make_tuple(gA_mkl, gB_nkl, gSFA, gSFB);
  }

  template <
    class LoadInputs,
    class BlockCoord,
    class KTileIterator
  >
  MUTLASS_DEVICE void
  load(
      LoadInputs const& load_inputs,
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
    Tensor sSFA = make_tensor(make_smem_ptr(storage.smem_SFA.begin()), SmemLayoutSFA{});
    Tensor sSFB = make_tensor(make_smem_ptr(storage.smem_SFB.begin()), SmemLayoutSFB{});

    auto cta_tme_a = params.tme_load_a.get_slice(0);
    auto cta_tme_b = params.tme_load_b.get_slice(0);
    auto cta_tme_sfa = params.tme_load_sfa.get_slice(0);
    auto cta_tme_sfb = params.tme_load_sfb.get_slice(0);

    auto [gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl] = load_inputs;
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    Tensor gA = gA_mkl(_, _, m_coord, _, l_coord);
    Tensor gB = gB_nkl(_, _, n_coord, _, l_coord);
    Tensor gSFA = gSFA_mkl(_, _, m_coord, _, l_coord);
    Tensor gSFB = gSFB_nkl(_, _, n_coord, _, l_coord);

    Tensor tAgA = cta_tme_a.partition_S(gA);
    Tensor tAsA = cta_tme_a.partition_D(sA);
    Tensor tBgB = cta_tme_b.partition_S(gB);
    Tensor tBsB = cta_tme_b.partition_D(sB);
    Tensor tAgSFA = cta_tme_sfa.partition_S(gSFA);
    Tensor tAsSFA = cta_tme_sfa.partition_D(sSFA);
    Tensor tBgSFB = cta_tme_sfb.partition_S(gSFB);
    Tensor tBsSFB = cta_tme_sfb.partition_D(sSFB);

    int warp_idx = canonical_warp_idx() & 3;
    MUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 0; --k_tile_count) {
      if (warp_idx == 0) {
        pipeline.producer_acquire(smem_pipe_write);
        uint32_t bar_id = pipeline.producer_get_barrier_id(smem_pipe_write);
        int write_stage = smem_pipe_write.index();
        copy(params.tme_load_a.with(bar_id),
             tAgA(_, _, _, *k_tile_iter),
             tAsA(_, _, _, write_stage));
        copy(params.tme_load_b.with(bar_id),
             tBgB(_, _, _, *k_tile_iter),
             tBsB(_, _, _, write_stage));
        copy(params.tme_load_sfa.with(bar_id),
             tAgSFA(_, _, _, *k_tile_iter),
             tAsSFA(_, _, _, write_stage));
        copy(params.tme_load_sfb.with(bar_id),
             tBgSFB(_, _, _, *k_tile_iter),
             tBsSFB(_, _, _, write_stage));
      }
      ++smem_pipe_write;
      ++k_tile_iter;
    }
  }

  template <
    class LoadInputs,
    class BlockCoord,
    class FrgTensorC
  >
  MUTLASS_DEVICE void
  mma(
      LoadInputs const& load_inputs,
      BlockCoord const& blk_coord,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      char* shared_memory,
      MainloopPipelineData& pipeline,
      PipelineState& smem_pipe_read) {
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be register resident");

    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(storage.smem_A.begin()), SmemLayoutA{}));
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_B.begin()), SmemLayoutB{});

    TiledMma rs_tiled_mma;
    auto thread_mma = rs_tiled_mma.get_thread_slice(thread_idx);
    Tensor tCsA = thread_mma.partition_A(sA);
    Tensor tCsB = thread_mma.partition_B(sB);
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);

    Tensor sSFA = make_tensor(make_smem_ptr(storage.smem_SFA.begin()), SmemLayoutSFA{});
    Tensor sSFB = make_tensor(make_smem_ptr(storage.smem_SFB.begin()), SmemLayoutSFB{});
    Tensor tCsSFA = MP32::BLOCKSCALED::partition_SFA(sSFA, thread_mma);
    Tensor tCsSFB = MP32::BLOCKSCALED::partition_SFB(sSFB, thread_mma);
    auto copy_sfa = Copy_Atom<UniversalCopy<uint32_t>, SwappedElementSFA>{};
    auto copy_sfb = Copy_Atom<UniversalCopy<uint32_t>, SwappedElementSFB>{};

    auto fragment_layout_a = make_fragment_like(take<0, 3>(tCsSFA.layout()));
    auto fragment_layout_b = make_fragment_like(take<0, 3>(tCsSFB.layout()));
    using SFAFragmentLayout = decltype(fragment_layout_a);
    using SFBFragmentLayout = decltype(fragment_layout_b);
    static_assert(cosize(SFAFragmentLayout{}) % 4 == 0);
    static_assert(cosize(SFBFragmentLayout{}) % 4 == 0);

    auto sf_cur_a_words = make_tensor<uint32_t>(make_shape(Int<cosize_v<SFAFragmentLayout> / 4>{}));
    auto sf_cur_b_words = make_tensor<uint32_t>(make_shape(Int<cosize_v<SFBFragmentLayout> / 4>{}));
    auto sf_next_a_words = make_fragment_like(sf_cur_a_words);
    auto sf_next_b_words = make_fragment_like(sf_cur_b_words);
    auto tCrSFA_0 = make_tensor(
      make_rmem_ptr(reinterpret_cast<SwappedElementSFA*>(sf_cur_a_words.data())),
      SFAFragmentLayout{});
    auto tCrSFB_0 = make_tensor(
      make_rmem_ptr(reinterpret_cast<SwappedElementSFB*>(sf_cur_b_words.data())),
      SFBFragmentLayout{});
    auto tCrSFA_1 = make_tensor(
      make_rmem_ptr(reinterpret_cast<SwappedElementSFA*>(sf_next_a_words.data())),
      SFAFragmentLayout{});
    auto tCrSFB_1 = make_tensor(
      make_rmem_ptr(reinterpret_cast<SwappedElementSFB*>(sf_next_b_words.data())),
      SFBFragmentLayout{});
    MUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrB));
    MUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrSFA_0));
    MUTE_STATIC_ASSERT_V(size<2>(tCrB) == size<2>(tCrSFB_0));

    auto prefetch_sf = [&](auto& dst_a, auto& dst_b, int stage) {
      copy(copy_sfa, tCsSFA(_, _, _, stage), dst_a);
      copy(copy_sfb, tCsSFB(_, _, _, stage), dst_b);
    };

    if (k_tile_count == 0) {
      return;
    }

    constexpr int KBlockCount = size<2>(tCsA);
    static_assert(KBlockCount == 1, "This pipeline requires one K block per stage");
    auto tCsA_atom = tCsA(_, _, Int<0>{}, Int<0>{});
    auto tCrA_mma_0 = make_fragment_like<MmaElementA>(tCsA_atom);
    auto tCrA_mma_1 = make_fragment_like<MmaElementA>(tCsA_atom);
    constexpr int PackedWords = decltype(size(recast<uint32_t>(tCrA_mma_0)))::value;
    Array<float_e2m1_t, 4> tCrA_load[PackedWords];

    // Prologue: prepare A and scales for the first tile in register buffer 0.
    {
      pipeline.consumer_wait(smem_pipe_read);
      int read_stage = smem_pipe_read.index();

      prefetch_sf(tCrSFA_0, tCrSFB_0, read_stage);
      detail::load_e2m1_packed(tCrA_load, tCsA(_, _, Int<0>{}, read_stage));
      detail::convert_e2m1_packed_to_fp8<MmaElementA>(tCrA_mma_0, tCrA_load);
    }

    // Each iteration consumes tiles k and k+1, leaving tile k+2 prepared
    // in buffer 0. The shared read cursor always identifies the current tile.
    // These operations expose overlap opportunities; instruction scheduling
    // and physical register allocation remain compiler-controlled.
    MUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 2; k_tile_count -= 2) {
      // First half: preload packed A for tile k+1 before issuing tile k.
      auto smem_pipe_read_next = smem_pipe_read;
      ++smem_pipe_read_next;
      pipeline.consumer_wait(smem_pipe_read_next);
      int next_stage = smem_pipe_read_next.index();
      prefetch_sf(tCrSFA_1, tCrSFB_1, next_stage);
      detail::load_e2m1_packed(tCrA_load, tCsA(_, _, Int<0>{}, next_stage));

      // Issue tile k using A/SF buffer 0 and the current shared B stage.
      int read_stage = smem_pipe_read.index();
      for_each(make_seq<KBlockCount>{}, [&](auto k_block) {
        mute::gemm(
          rs_tiled_mma,
          make_zip_tensor(tCrA_mma_0, tCrSFA_0(_, _, k_block)),
          make_zip_tensor(tCrB(_, _, k_block, read_stage), tCrSFB_0(_, _, k_block)),
          accum);
        warpsquad_commit_batch();
        // Prepare A buffer 1 for tile k+1 while tile k is in flight.
        detail::convert_e2m1_packed_to_fp8<MmaElementA>(tCrA_mma_1, tCrA_load);

        warpsquad_wait();
      });

      // SQMMA must finish reading shared B before the producer can reuse this stage.
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
      read_stage = smem_pipe_read.index();

      // Second half: tile k has been released; preload packed A for k+2.
      // The raw load buffer can be reused after conversion of tile k+1.
      ++smem_pipe_read_next;
      pipeline.consumer_wait(smem_pipe_read_next);
      next_stage = smem_pipe_read_next.index();
      prefetch_sf(tCrSFA_0, tCrSFB_0, next_stage);
      detail::load_e2m1_packed(tCrA_load, tCsA(_, _, Int<0>{}, next_stage));

      // Issue tile k+1 using A/SF buffer 1 and its shared B stage.
      for_each(make_seq<KBlockCount>{}, [&](auto k_block) {
        mute::gemm(
          rs_tiled_mma,
          make_zip_tensor(tCrA_mma_1, tCrSFA_1(_, _, k_block)),
          make_zip_tensor(tCrB(_, _, k_block, read_stage), tCrSFB_1(_, _, k_block)),
          accum);
        warpsquad_commit_batch();
        // Refill A buffer 0 for tile k+2 after tile k has completed.
        detail::convert_e2m1_packed_to_fp8<MmaElementA>(tCrA_mma_0, tCrA_load);

        warpsquad_wait();
      });

      // SQMMA must finish reading shared B before the producer can reuse this stage.
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
    }

    // Two-tile tail: prepare only the final tile; never prefetch beyond K.
    if (k_tile_count == 2) {
      // Preload packed A for the final tile while buffer 0 holds the penultimate tile.
      auto smem_pipe_read_next = smem_pipe_read;
      ++smem_pipe_read_next;
      pipeline.consumer_wait(smem_pipe_read_next);
      int next_stage = smem_pipe_read_next.index();
      prefetch_sf(tCrSFA_1, tCrSFB_1, next_stage);
      detail::load_e2m1_packed(tCrA_load, tCsA(_, _, Int<0>{}, next_stage));

      // Issue the penultimate tile using A/SF buffer 0.
      int read_stage = smem_pipe_read.index();
      for_each(make_seq<KBlockCount>{}, [&](auto k_block) {
        mute::gemm(
          rs_tiled_mma,
          make_zip_tensor(tCrA_mma_0, tCrSFA_0(_, _, k_block)),
          make_zip_tensor(tCrB(_, _, k_block, read_stage), tCrSFB_0(_, _, k_block)),
          accum);
        warpsquad_commit_batch();
        // Prepare the final tile in A buffer 1 while SQMMA is in flight.
        detail::convert_e2m1_packed_to_fp8<MmaElementA>(tCrA_mma_1, tCrA_load);

        warpsquad_wait();
      });

      // SQMMA must finish reading shared B before the producer can reuse this stage.
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;

      k_tile_count--;

      // Drain the final tile from buffer 1 without issuing another preload.
      read_stage = smem_pipe_read.index();
      for_each(make_seq<KBlockCount>{}, [&](auto k_block) {
        mute::gemm(
          rs_tiled_mma,
          make_zip_tensor(tCrA_mma_1, tCrSFA_1(_, _, k_block)),
          make_zip_tensor(tCrB(_, _, k_block, read_stage), tCrSFB_1(_, _, k_block)),
          accum);
        warpsquad_commit_batch();
        warpsquad_wait();
      });

      // SQMMA must finish reading shared B before the producer can reuse this stage.
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;

      k_tile_count--;

      return;
    }

    // One-tile tail: buffer 0 is already prepared by the prologue or mainloop.
    if (k_tile_count == 1) {
      int read_stage = smem_pipe_read.index();

      for_each(make_seq<KBlockCount>{}, [&](auto k_block) {
        mute::gemm(
          rs_tiled_mma,
          make_zip_tensor(tCrA_mma_0, tCrSFA_0(_, _, k_block)),
          make_zip_tensor(tCrB(_, _, k_block, read_stage), tCrSFB_0(_, _, k_block)),
          accum);
        warpsquad_commit_batch();
        warpsquad_wait();
      });

      // SQMMA must finish reading shared B before the producer can reuse this stage.
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;

      k_tile_count--;

      return;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
