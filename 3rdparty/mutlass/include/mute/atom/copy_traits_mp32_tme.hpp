/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
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

#include <mute/arch/copy_mp31_tme.hpp>
#include <mute/arch/copy_mp32_tme.hpp>
#include <mute/atom/copy_traits_mp31_tme.hpp>

namespace mute
{

//////////////////////////////////////////////////////////////////////////////
//////////////////////////// MP32 TME LOAD ///////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  TME::CacheHint      inner_hint,
  TME::CacheHint      outer_hint,
  PrefetchSize          prefetch,
  int                PermutationStride = 0
>
struct MP32_TME_LOAD_OP
  : MP32_TME_LOAD_ENTRY<sg, ss, sl, inner_hint, outer_hint, prefetch, PermutationStride> {};


// The non-executable MP32_TME_LOAD_ENTRY with tme_desc and no tme_bar
// Use .with(tme_bar) to construct an executable version
template <
  class NumBitsPerTME,
  class AuxParams_,
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  TME::CacheHint      inner_hint,
  TME::CacheHint      outer_hint,
  PrefetchSize          prefetch,
  int                PermutationStride
>
struct Copy_Traits<MP32_TME_LOAD_ENTRY<sg, ss, sl, inner_hint, outer_hint, prefetch, PermutationStride>,
                   NumBitsPerTME,
                   AuxParams_>
{
  using ThrID    = Layout<_1>;
  using SrcLayout = Layout<Shape<_1,NumBitsPerTME>>;
  using DstLayout = Layout<Shape<_1,NumBitsPerTME>>;
  using RefLayout = SrcLayout;

  TmeDescriptor tme_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;

  MUTE_HOST_DEVICE constexpr
  TmeDescriptor const*
  get_tme_descriptor() const {
    return &tme_desc_;
  }

  MUTE_HOST_DEVICE constexpr
  Copy_Traits<MP32_TME_LOAD_OP<sg, ss, sl, inner_hint, outer_hint, prefetch, PermutationStride>,
              NumBitsPerTME,
              AuxParams>
  with(uint32_t tme_bar) const {
    return {{}, {&tme_desc_, tme_bar, aux_params_}};
  }

  // Construct an executable load with a runtime-selected descriptor.
#if !defined(__MUSA_ARCH__) || defined(MUTE_ARCH_TME_MP32_ACTIVATED)
  MUTE_HOST_DEVICE constexpr
  Copy_Traits<MP32_TME_LOAD_OP<sg, ss, sl, inner_hint, outer_hint, prefetch, PermutationStride>,
              NumBitsPerTME,
              AuxParams>
  with(TmeDescriptor const* tme_desc, uint32_t tme_bar) const {
    return {{}, {tme_desc, tme_bar, aux_params_}};
  }
#else
  MUTE_HOST_DEVICE constexpr
  Copy_Traits<MP32_TME_LOAD_OP<sg, ss, sl, inner_hint, outer_hint, prefetch, PermutationStride>,
              NumBitsPerTME,
              AuxParams>
  with(TmeDescriptor const*, uint32_t) const = delete;
#endif

  template <class GShape>
  MUTE_HOST_DEVICE constexpr
  auto
  get_tme_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  MUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};


// The executable MP32_TME_LOAD with tme_desc, tme_bar & tme_params
template <
  class NumBitsPerTME,
  class AuxParams,
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  TME::CacheHint      inner_hint,
  TME::CacheHint      outer_hint,
  PrefetchSize          prefetch,
  int                PermutationStride
>
struct Copy_Traits<MP32_TME_LOAD_OP<sg, ss, sl, inner_hint, outer_hint, prefetch, PermutationStride>,
                   NumBitsPerTME,
                   AuxParams>
     : TME_LOAD_Unpack<MP32_TME_LOAD_OP<sg, ss, sl, inner_hint, outer_hint, prefetch, PermutationStride>>
{
  using ThrID    = Layout<_1>;
  using SrcLayout = Layout<Shape<_1,NumBitsPerTME>>;
  using DstLayout = Layout<Shape<_1,NumBitsPerTME>>;
  using RefLayout = SrcLayout;

  tuple<
    TmeDescriptor const*,
    uint32_t const&,
    AuxParams const&
  > const opargs_;
};

//////////////////////////////////////////////////////////////////////////////
//////////////////////// MP32 TME REDUCE ADD /////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

template <
  class NumBitsPerTME,
  class AuxParams_,
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  class T
>
struct Copy_Traits<MP32_TME_REDUCE_ADD_ENTRY<sg, ss, sl, T>, NumBitsPerTME, AuxParams_>
{
  using ThrID = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, NumBitsPerTME>>;
  using DstLayout = Layout<Shape<_1, NumBitsPerTME>>;
  using RefLayout = SrcLayout;

  TmeDescriptor tme_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;

  MUTE_HOST_DEVICE constexpr
  TmeDescriptor const*
  get_tme_descriptor() const {
    return &tme_desc_;
  }

  template <class GShape>
  MUTE_HOST_DEVICE constexpr
  auto
  get_tme_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  MUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_smem<TS>::value, "Expected smem src for MP32_TME_REDUCE_ADD");

    void const* const desc_ptr = &traits.tme_desc_;
    void const* const smem_ptr = mute::raw_pointer_cast(src.data());
    auto dst_coord = dst.data().coord_;
    auto block_dim = mute::flatten(traits.aux_params_.block_dim_);

    return detail::explode_tuple(
        detail::CallCOPY<MP32_TME_REDUCE_ADD_ENTRY<sg, ss, sl, T>>{},
        make_tuple(desc_ptr, smem_ptr), seq<0,1>{},
        dst_coord, tuple_seq<decltype(dst_coord)>{},
        block_dim, tuple_seq<decltype(block_dim)>{});
  }
};

namespace detail {

template <class TmeInternalType,
          TME::SmemSwizzleGranularity SwizzleGranularity,
          TME::SmemSwizzleStride SwizzleStride,
          TME::SmemSwizzleLine SwizzleLine,
          TME::CacheHint InnerHint,
          TME::CacheHint OuterHint,
          PrefetchSize Prefetch>
struct TmeCopyOpEntry<MP32_TME_REDUCE_ADD, TmeInternalType,
                      SwizzleGranularity, SwizzleStride, SwizzleLine,
                      InnerHint, OuterHint, Prefetch>
{
  static_assert(InnerHint == TME::CacheHint::CACHE_NORMAL &&
                OuterHint == TME::CacheHint::CACHE_NORMAL,
                "MP32 TME reduce-add currently only supports normal cache hints.");
  using type = MP32_TME_REDUCE_ADD_ENTRY<
      SwizzleGranularity, SwizzleStride, SwizzleLine, TmeInternalType>;
};

} // namespace detail

namespace MP32::detail {

template <int NumValues, class MmaAtom>
MUTE_HOST_DEVICE constexpr
auto
make_mma_c_atom_tiler()
{
  auto mma = make_tiled_mma(MmaAtom{});
  auto layoutC_TV = mma.get_layoutC_TV();
  MUTE_STATIC_ASSERT_V(Int<NumValues>{} <= size<1>(layoutC_TV));

  auto layout_TV = composition(
      layoutC_TV,
      make_layout(make_shape(size<0>(layoutC_TV), Int<NumValues>{})));
  auto mma_tiler = make_shape(tile_size<0>(mma), tile_size<1>(mma));
  auto mma_zeros = repeat_like(mma_tiler, Int<0>{});

  return transform(make_seq<rank(mma_tiler)>{}, [&](auto i) {
    return filter(composition(
        make_layout(mma_tiler, replace<i>(mma_zeros, Int<1>{})),
        layout_TV));
  });
}

template <class MmaAtom>
MUTE_HOST_DEVICE constexpr
auto
deduce_mma_n_thread_span()
{
  constexpr auto tiler_v1 = make_mma_c_atom_tiler<1, MmaAtom>();
  return size(get<1>(tiler_v1));
}

template <class MmaAtom>
MUTE_HOST_DEVICE constexpr
auto
deduce_mma_n_value_stride()
{
  constexpr auto tiler_v1 = make_mma_c_atom_tiler<1, MmaAtom>();
  constexpr auto tiler_v2 = make_mma_c_atom_tiler<2, MmaAtom>();
  MUTE_STATIC_ASSERT_V(size(get<0>(tiler_v1)) == size(get<0>(tiler_v2)),
                       "Expected the first two SQMMA values to differ only in N.");
  return size(get<1>(tiler_v2)) - size(get<1>(tiler_v1));
}

template <class MmaAtom, class PermN>
MUTE_HOST_DEVICE constexpr
auto
deduce_tme_permutation_stride(PermN const& perm_n)
{
  constexpr int NThreadSpan = decltype(deduce_mma_n_thread_span<MmaAtom>())::value;
  constexpr int NValueStride = decltype(deduce_mma_n_value_stride<MmaAtom>())::value;
  constexpr int N = decltype(size(perm_n))::value;
  constexpr int PermutationStride = decltype(perm_n(Int<1>{}) - perm_n(Int<0>{}))::value;
  constexpr int PermutationBlockSize = NThreadSpan * PermutationStride;

  static_assert(NThreadSpan > 0, "Expected SQMMA threads to cover a contiguous N span.");
  static_assert(NValueStride == NThreadSpan, "Expected SQMMA value stride to match its N thread span.");
  static_assert(PermutationStride > 0, "MP32 TME permutation stride must be positive.");
  static_assert(N % PermutationBlockSize == 0,
                "N permutation size must be divisible by its permutation block size.");

  for_each(make_seq<N>{}, [&](auto n) {
    constexpr int n_value = decltype(n)::value;
    constexpr int inner = n_value % PermutationBlockSize;
    constexpr int outer = n_value / PermutationBlockSize * PermutationBlockSize;
    constexpr int expected = outer + (inner * PermutationStride) / PermutationBlockSize +
                             (inner * PermutationStride) % PermutationBlockSize;
    MUTE_STATIC_ASSERT_V(perm_n(n) == Int<expected>{},
                         "TiledMMA N permutation is not representable by MP32 TME permutation stride.");
  });

  return Int<PermutationStride>{};
}

template <class MmaAtom, class AtomMNK, class PermutationMNK>
MUTE_HOST_DEVICE constexpr
auto
deduce_tme_permutation_stride(TiledMMA<MmaAtom, AtomMNK, PermutationMNK> const& tiled_mma)
{
  auto perm_n = tiled_mma.template permutation_mnk<1>();
  if constexpr (is_layout<decltype(perm_n)>::value) {
    return deduce_tme_permutation_stride<MmaAtom>(perm_n);
  } else {
    return Int<0>{};
  }
}

template <class Mp31Entry, int PermutationStride>
struct rebind_mp31_load_to_mp32;

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  TME::CacheHint      inner_hint,
  TME::CacheHint      outer_hint,
  PrefetchSize          prefetch,
  int                PermutationStride
>
struct rebind_mp31_load_to_mp32<MP31_TME_LOAD_ENTRY<sg, ss, sl, inner_hint, outer_hint, prefetch>,
                                PermutationStride>
{
  using type = MP32_TME_LOAD_ENTRY<sg, ss, sl, inner_hint, outer_hint, prefetch, PermutationStride>;
};

template <class TmeGbasis, int PermutationStride, int NThreadSpan>
MUTE_HOST_DEVICE constexpr
void
validate_tme_permutation_gbasis(TmeGbasis const& tme_gbasis)
{
  static_assert(PermutationStride > 0,
                "MP32 TME permutation validation requires a positive permutation stride.");
  MUTE_STATIC_ASSERT_V(rank(tme_gbasis) >= Int<2>{},
                       "MP32 TME permutation requires at least two preserved TME modes.");
  if constexpr (decltype(rank(tme_gbasis) >= Int<2>{})::value) {
    MUTE_STATIC_ASSERT_V(size<1>(tme_gbasis) % Int<NThreadSpan * PermutationStride>{} == Int<0>{},
                         "TME N block dimension must be divisible by its permutation block size.");
  }
}

template <int PermutationStride,
          class TmeInternalType,
          TME::CacheHint InnerHint,
          TME::CacheHint OuterHint,
          class GEngine, class GLayout,
          class SLayout,
          class VShape, class VStride,
          class TmeGbasis>
MUTE_HOST_RTC
auto
make_tme_copy_tiled_from_gbasis(Tensor<GEngine, GLayout> const& gtensor,
                                SLayout                  const& slayout,
                                Layout<VShape, VStride>  const& cta_v_map,
                                TmeGbasis                const& tme_gbasis)
{
  auto smem_swizzle = mute::detail::get_swizzle_portion(slayout);

  auto [tme_desc, aux_params, mp31_copy_op] =
      mute::detail::make_tme_copy_desc<TmeInternalType, MP31_TME_LOAD, InnerHint, OuterHint>(
          gtensor, tme_gbasis, smem_swizzle);

  // MP32 TME encodes the permutation stride in the builtin inner_stride operand.
  using CopyOp = typename rebind_mp31_load_to_mp32<decltype(mp31_copy_op), PermutationStride>::type;
  auto atom = mute::detail::make_tme_copy_atom_from_gbasis<TmeInternalType>(
      CopyOp{}, gtensor, tme_gbasis, tme_desc, aux_params);

  return mute::detail::make_tme_copy_tiled_from_atom(atom, gtensor, slayout, cta_v_map);
}

template <class TmeInternalType,
          TME::CacheHint InnerHint,
          TME::CacheHint OuterHint,
          class GEngine, class GLayout,
          class SLayout,
          class VShape, class VStride,
          class MmaAtom, class AtomMNK, class PermutationMNK>
MUTE_HOST_RTC
auto
make_tme_copy_tiled(MP31_TME_LOAD const&,
                    Tensor<GEngine, GLayout> const& gtensor,
                    SLayout                  const& slayout,
                    Layout<VShape, VStride>  const& cta_v_map,
                    TiledMMA<MmaAtom, AtomMNK, PermutationMNK> const& tiled_mma)
{
  auto smem_layout = mute::detail::get_nonswizzle_portion(slayout);

  auto permutation_stride = deduce_tme_permutation_stride(tiled_mma);
  constexpr int PermutationStride = decltype(permutation_stride)::value;

  if constexpr (PermutationStride == 0) {
    auto tme_gbasis = mute::detail::construct_tme_gbasis<TmeInternalType>(
        gtensor, smem_layout, cta_v_map);
    return make_tme_copy_tiled_from_gbasis<0, TmeInternalType, InnerHint, OuterHint>(
        gtensor, slayout, cta_v_map, tme_gbasis);
  } else {
    constexpr int NThreadSpan = decltype(deduce_mma_n_thread_span<MmaAtom>())::value;
    auto tme_gbasis = mute::detail::construct_tme_gbasis<TmeInternalType, true>(
        gtensor, smem_layout, cta_v_map);
    validate_tme_permutation_gbasis<decltype(tme_gbasis), PermutationStride, NThreadSpan>(tme_gbasis);
    return make_tme_copy_tiled_from_gbasis<PermutationStride, TmeInternalType, InnerHint, OuterHint>(
        gtensor, slayout, cta_v_map, tme_gbasis);
  }
}

} // namespace MP32::detail


template <class TmeInternalType = void,
          TME::CacheHint InnerHint = TME::CacheHint::CACHE_NORMAL,
          TME::CacheHint OuterHint = TME::CacheHint::CACHE_NORMAL,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler,
          class MmaAtom, class AtomMNK, class PermutationMNK>
MUTE_HOST_RTC
auto
make_tme_copy(MP31_TME_LOAD const& copy_op,
              Tensor<GEngine, GLayout> const& gtensor,
              SLayout                  const& slayout,
              CTA_Tiler                const& cta_tiler,
              TiledMMA<MmaAtom, AtomMNK, PermutationMNK> const& tiled_mma)
{
  auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler);

  using TmeType = conditional_t<is_same<void, TmeInternalType>::value, typename GEngine::value_type, TmeInternalType>;
  return MP32::detail::make_tme_copy_tiled<TmeType, InnerHint, OuterHint>(
      copy_op, gtensor, slayout, cta_v_tile, tiled_mma);
}

} // namespace mute

////////////////////////////////////////////////////////////////////////////////////////////////////
