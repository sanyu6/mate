/***************************************************************************************************
 * Copyright (c) 2026 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

#include "mutlass/detail/layout.hpp"
#include "mutlass/epilogue/collective/collective_epilogue.hpp"
#include "mutlass/epilogue/thread/linear_combination.h"

namespace mutlass::epilogue::collective {

template <
  class OpClass, class TileShape_MNK, class ClusterShape_MNK,
  class EpilogueTileType, class ElementAccumulator, class ElementCompute,
  class ElementC, class GmemLayoutTagC, int AlignmentC,
  class ElementD, class GmemLayoutTagD, int AlignmentD,
  class MainloopCollective, FloatRoundStyle RoundStyle>
struct CollectiveBuilder<
    arch::Mp32, OpClass, TileShape_MNK, ClusterShape_MNK, EpilogueTileType,
    ElementAccumulator, ElementCompute, ElementC, GmemLayoutTagC, AlignmentC,
    ElementD, GmemLayoutTagD, AlignmentD, WithTme,
    fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute, RoundStyle>,
    MainloopCollective,
    mute::void_t<typename MainloopCollective::TiledMma>> {
  using TiledMma = typename MainloopCollective::TiledMma;
  static_assert(mute::is_same_v<ElementAccumulator, float> &&
                mute::is_same_v<ElementCompute, float> &&
                mute::is_same_v<ElementD, half_t> && mute::is_void_v<ElementC>,
                "MP32 TME epilogue requires FP32 accumulation, FP16 output and void C");
  static_assert(mute::is_same_v<GmemLayoutTagD, layout::RowMajor>,
                "MP32 TME epilogue requires RowMajor output");
  static_assert(mute::is_static<TileShape_MNK>::value);
  static_assert(mute::is_same_v<ClusterShape_MNK, mute::Shape<mute::_1, mute::_1, mute::_1>>);
  static_assert(mute::is_same_v<EpilogueTileType, EpilogueTileAuto> ||
                mute::is_same_v<EpilogueTileType, mute::Shape<mute::_64, mute::_128>>,
                "MP32 TME epilogue uses 64x128 subtiles");
  static_assert(AlignmentD >= 8, "MP32 TME output requires 16-byte alignment");
  using ThreadOp = thread::LinearCombination<
      ElementD, 4, ElementAccumulator, ElementCompute,
      thread::ScaleType::OnlyAlphaScaling, RoundStyle>;
  using CollectiveOp = CollectiveEpilogue<
      Mp32CollectiveEpilogue, mutlass::detail::TagToStrideC_t<GmemLayoutTagD>, ThreadOp,
      TileShape_MNK, TiledMma>;
};

} // namespace mutlass::epilogue::collective
