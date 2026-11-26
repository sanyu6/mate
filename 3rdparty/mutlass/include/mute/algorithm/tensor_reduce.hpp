/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

#include <mute/config.hpp>
#include <mute/tensor_impl.hpp>

#include <mute/algorithm/functional.hpp>
#include <mute/algorithm/fill.hpp>

namespace mute
{

// Reduce @src tensor using binary reduction operator @op and initial value @init and return a scalar.
template <class SrcEngine, class SrcLayout, class T, class BinaryOp = mute::plus>
MUTE_HOST_DEVICE constexpr
T
reduce(Tensor<SrcEngine,SrcLayout> const& src, T init, BinaryOp op = {})
{
  for (int i = 0; i < size(src); ++i) {
    init = op(init, src(i));
  }
  return init;
}

// Reduce @src tensor RedMode using binary reduction operator @op and store the result in @dst tensor
// for each index in @dst/BatchMode.
// @pre @src tensor has rank 2
// @pre size of @src batch mode is equal to size of @dst batch mode
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout,
          class BinaryOp = mute::plus>
MUTE_HOST_DEVICE constexpr
void
batch_reduce(Tensor<SrcEngine, SrcLayout> const& src,       // (RedMode, BatchMode)
             Tensor<DstEngine, DstLayout>      & dst,       // (BatchMode)
             BinaryOp op = {})
{
  // Precondition
  MUTE_STATIC_ASSERT_V(rank(src) == Int<2>{});
  assert(size<1>(src) == size(dst));

  for (int i = 0; i < size(dst); ++i) {
    dst(i) = reduce(src(_,i), dst(i), op);
  }
}

// Reduce @src tensor along selected modes specified in @target_profile using binary reduction operator @op
// and store the result in @dst tensor. @target_profile is a tuple where '_' indicates modes to keep and
// integers indicates modes to reduce.
// @pre @target_profile is compatible with @src layout
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout,
          class TargetProfile,
          class BinaryOp = mute::plus>
MUTE_HOST_DEVICE constexpr
void
logical_reduce(Tensor<SrcEngine, SrcLayout> const& src,
               Tensor<DstEngine, DstLayout>      & dst,
               TargetProfile                const& target_profile,
               BinaryOp op = {})
{
  // Precondition
  assert(compatible(target_profile, shape(src)));

  auto diced_layout = dice(target_profile, src.layout());
  auto sliced_layout = slice(target_profile, src.layout());

  auto red_mode = conditional_return<rank(diced_layout) == Int<0>{}>(Layout<_1,_0>{}, diced_layout);
  auto batch_mode = conditional_return<rank(sliced_layout) == Int<0>{}>(Layout<_1,_0>{}, sliced_layout);

  auto src_tensor = make_tensor(src.data(), make_layout(red_mode, batch_mode));

  batch_reduce(src_tensor, dst, op);
}

} // namespace mute
