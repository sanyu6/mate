#pragma once

#include "mute/atom/mma_atom.hpp"

#include "mutlass/gemm/gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass::gemm::collective::detail {

constexpr int tme_alignment_bytes = 4;

template <class LayoutA>
constexpr mute::TCE::Major
sqmma_ss_tag_to_major_A() {
  if constexpr (mutlass::gemm::detail::is_mn_major_A<LayoutA>()) {
    return mute::TCE::Major::MN;
  }
  else {
    return mute::TCE::Major::K;
  }
}

template <class LayoutB>
constexpr mute::TCE::Major
sqmma_ss_tag_to_major_B() {
  if constexpr (mutlass::gemm::detail::is_mn_major_B<LayoutB>()) {
    return mute::TCE::Major::MN;
  }
  else {
    return mute::TCE::Major::K;
  }
}

template <class TileShape_MNK, class SqmmaOp>
constexpr auto
make_sqmma_tiled_mma() {
  constexpr int TileM = mute::size<0>(TileShape_MNK{});
  constexpr int TileN = mute::size<1>(TileShape_MNK{});

  constexpr int InstM = mute::size<0>(typename mute::MMA_Traits<SqmmaOp>::Shape_MNK{});
  constexpr int InstN = mute::size<1>(typename mute::MMA_Traits<SqmmaOp>::Shape_MNK{});

  constexpr int AtomM = TileM / InstM;
  constexpr int AtomN = TileN / InstN;

  static_assert(TileM % InstM == 0 && TileN % InstN == 0, "Invalid Tile shape");
  using AtomLayout = decltype(mute::make_layout(
      mute::Shape<mute::Int<AtomM>, mute::Int<AtomN>, mute::Int<1>>{}, mute::LayoutRight{}));

  return mute::make_tiled_mma(SqmmaOp{}, AtomLayout{});
}

template <class ElementA, int AlignmentA, class ElementB, int AlignmentB, int RequiredAlignment>
constexpr bool
is_aligned() {
  constexpr int RequiredAlignmentBits = RequiredAlignment * 8;
  return ((mute::sizeof_bits_v<ElementA> * AlignmentA) % RequiredAlignmentBits == 0) &&
         ((mute::sizeof_bits_v<ElementB> * AlignmentB) % RequiredAlignmentBits == 0);
}

} // namespace mutlass::gemm::collective::detail

/////////////////////////////////////////////////////////////////////////////////////////////////
