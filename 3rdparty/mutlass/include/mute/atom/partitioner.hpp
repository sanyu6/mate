/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include <type_traits>

#include <mute/config.hpp>
#include <mute/tensor.hpp>

namespace mute
{

//
// A generic tiling of thread-value layouts
//

template <class Layout_TV_,    // (tid,vid) -> coord   [Need not be 2D...]
          class Tiler_MN_>     // coord space
struct TV_Tiler
{
  using Tiler_MN       = Tiler_MN_;
  using TiledLayout_TV = Layout_TV_;

  // Tile a tensor or a layout from shape
  //   (M,N,...)
  // to shape
  //   ((ThrV,FrgV),(RestM,RestN,...))
  // where
  //   ThrV:  The threads local to a tile.
  //   FrgV:  The values local to a tile.
  //   RestM: The values tiled in M.
  //   RestN: The values tiled in N.
  template <class Tensor>
  MUTE_HOST_DEVICE constexpr static
  auto
  apply(Tensor&& tensor)
  {
    // If Layout_TV and Tiler_MN were composable in general, then this won't be needed!

    // ((thr_id,val_id),(RestM,RestN,...))
    return zipped_divide(tensor, Tiler_MN{}).compose(TiledLayout_TV{}, _);
  }

  template <class SliceCoord>
  struct TV_Partitioner
  {
    SliceCoord coord_;

    template <class TargetTensor>
    MUTE_HOST_DEVICE
    auto
    partition(TargetTensor&& target)
    {
      Tensor thr_tensor = make_tensor(static_cast<TargetTensor&&>(target).data(), apply(target.layout()));
      return thr_tensor(coord_, repeat<rank_v<TargetTensor>>(_));
    }
  };

  template <class SliceCoord>
  MUTE_HOST_DEVICE static
  auto
  get_slice(SliceCoord const& coord)
  {
    return TV_Partitioner<SliceCoord>{coord};
  }
};

template <class Layout_TV,
          class Tiler_MN>
MUTE_HOST_DEVICE
auto
make_tiler_impl(Layout_TV const&,
                Tiler_MN  const&)
{
  return TV_Tiler<Layout_TV, Tiler_MN>{};
}

} // namespace mute
