#pragma once

#include "mutlass/gemm/gemm.h"
#include "mute/atom/mma_traits_mp31_sqmma.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

//
// Architecture-specific constants
//
constexpr int mp31_smem_capacity_bytes = 196608;

template <mute::TCE::Major major, class ElementType, class SqmmaOp, class TileShape>
MUTE_HOST_DEVICE constexpr
auto
ss_smem_selector_A()
{
  using AtomOpTraits = MMA_Traits<SqmmaOp>;
  using AtomOpShape = typename AtomOpTraits::Shape_MNK;

  using AtomM = decltype(get<0>(AtomOpShape{}));
  using AtomK = decltype(get<2>(AtomOpShape{}));

  if constexpr (major == mute::TCE::Major::MN) {
    return mute::MP31::SQMMA::make_canonical_gemm_smem_atom_layout<ElementType, major, AtomM, mute::tuple_element_t<2, TileShape>>();
  } else {
    constexpr int SmemAtomBits = 256 * 8;
    constexpr int ElementABits = sizeof_bits_v<typename AtomOpTraits::ValTypeA>;
    constexpr int ElementBBits = sizeof_bits_v<typename AtomOpTraits::ValTypeB>;
    constexpr int MmaElementBits = ElementABits > ElementBBits ? ElementABits : ElementBBits;
    constexpr int TileKBits = size<2>(TileShape{}) * MmaElementBits;
    static_assert(SmemAtomBits % MmaElementBits == 0,
                  "MMA element type must evenly divide a 256-byte SMEM atom");
    constexpr int MaxAtomK = SmemAtomBits / MmaElementBits;

    using SmemAtomK = mute::conditional_t<TileKBits % SmemAtomBits == 0,
                                          Int<min(MaxAtomK, mute::tuple_element_t<2, TileShape>{})>, AtomK>;
    return mute::MP31::SQMMA::make_canonical_gemm_smem_atom_layout<ElementType, major, mute::tuple_element_t<0, TileShape>, SmemAtomK>();
  }
}

template <mute::TCE::Major major, class ElementType, class SqmmaOp, class TileShape>
MUTE_HOST_DEVICE constexpr
auto
ss_smem_selector_B()
{
  using AtomOpTraits = MMA_Traits<SqmmaOp>;
  using AtomOpShape = typename AtomOpTraits::Shape_MNK;

  using AtomN = decltype(get<1>(AtomOpShape{}));
  using AtomK = decltype(get<2>(AtomOpShape{}));

  if constexpr (major == mute::TCE::Major::MN) {
    return mute::MP31::SQMMA::make_canonical_gemm_smem_atom_layout<ElementType, major, AtomN, mute::tuple_element_t<2, TileShape>>();
  } else {
    constexpr int SmemAtomBits = 256 * 8;
    constexpr int ElementABits = sizeof_bits_v<typename AtomOpTraits::ValTypeA>;
    constexpr int ElementBBits = sizeof_bits_v<typename AtomOpTraits::ValTypeB>;
    constexpr int MmaElementBits = ElementABits > ElementBBits ? ElementABits : ElementBBits;
    constexpr int TileKBits = size<2>(TileShape{}) * MmaElementBits;
    static_assert(SmemAtomBits % MmaElementBits == 0,
                  "MMA element type must evenly divide a 256-byte SMEM atom");
    constexpr int MaxAtomK = SmemAtomBits / MmaElementBits;
    using SmemAtomK = mute::conditional_t<TileKBits % SmemAtomBits == 0,
                                          Int<min(MaxAtomK, mute::tuple_element_t<2, TileShape>{})>, AtomK>;
    return mute::MP31::SQMMA::make_canonical_gemm_smem_atom_layout<ElementType, major, mute::tuple_element_t<1, TileShape>, SmemAtomK>();
  }
}

template <class ElementA, class ElementB>
constexpr bool
is_input_fp8() {
  return ((mute::is_same_v<ElementA, float_e4m3_t> || mute::is_same_v<ElementA, float_e5m2_t>) &&
          (mute::is_same_v<ElementB, float_e4m3_t> || mute::is_same_v<ElementB, float_e5m2_t>));
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
