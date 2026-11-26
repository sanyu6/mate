/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#if ! defined(__MUSACC_RTC__)
#include <cstdlib>
#endif

#include "mutlass/conv/convnd_problem_shape.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass::conv::collective::detail {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Compute the lower/near corner, returning it in [W,H,D] order.
template <conv::Operator ConvOp, int NumSpatialDimensions>
MUTLASS_HOST_DEVICE
constexpr auto
compute_lower_corner_whd(ConvProblemShape<ConvOp, NumSpatialDimensions> const& problem_shape) {
  mute::array<int, NumSpatialDimensions> lower{};

  if constexpr (ConvOp == conv::Operator::kFprop ||
                ConvOp == conv::Operator::kWgrad) {
    mute::for_each(mute::make_seq<NumSpatialDimensions>{}, [&](auto i) {
      lower[NumSpatialDimensions - 1 - i] = -problem_shape.lower_padding[i];
    });
  }
  else if constexpr (ConvOp == conv::Operator::kDgrad) {
    mute::for_each(mute::make_seq<NumSpatialDimensions>{}, [&](auto i) {
      lower[NumSpatialDimensions - 1 - i] = problem_shape.lower_padding[i] -
        (problem_shape.shape_B[i + 1] - 1) * problem_shape.dilation[i];
    });
  }

  return lower;
}

// Compute the upper/far corner, returning it in [W,H,D] order.
template <conv::Operator ConvOp, int NumSpatialDimensions>
MUTLASS_HOST_DEVICE
constexpr auto
compute_upper_corner_whd(ConvProblemShape<ConvOp, NumSpatialDimensions> const& problem_shape) {
  mute::array<int, NumSpatialDimensions> upper{};

  if constexpr (ConvOp == conv::Operator::kFprop) {
    mute::for_each(mute::make_seq<NumSpatialDimensions>{}, [&](auto i) {
      upper[NumSpatialDimensions - 1 - i] = problem_shape.upper_padding[i] -
        (problem_shape.shape_B[i + 1] - 1) * problem_shape.dilation[i];
    });
  }
  else if constexpr (ConvOp == conv::Operator::kWgrad) {
    mute::for_each(mute::make_seq<NumSpatialDimensions>{}, [&](auto i) {
      upper[NumSpatialDimensions - 1 - i] = problem_shape.upper_padding[i] -
        (problem_shape.shape_C[i + 1] - 1) * problem_shape.dilation[i];
    });
  }
  else if constexpr (ConvOp == conv::Operator::kDgrad) {
    mute::for_each(mute::make_seq<NumSpatialDimensions>{}, [&](auto i) {
      upper[NumSpatialDimensions - 1 - i] = problem_shape.lower_padding[i] -
        (problem_shape.shape_B[i + 1] - 1) * problem_shape.dilation[i] +
        problem_shape.shape_C[i + 1] - problem_shape.shape_A[i + 1];
    });
  }

  return upper;
}

// Compute the lower/near corner of filter modes, returning it in [S,R,T] order.
template <conv::Operator ConvOp, int NumSpatialDimensions>
MUTLASS_HOST_DEVICE
constexpr auto
compute_lower_srt(ConvProblemShape<ConvOp, NumSpatialDimensions> const& problem_shape) {
  mute::array<int, NumSpatialDimensions> lower{};

  if constexpr (ConvOp == conv::Operator::kFprop ||
                ConvOp == conv::Operator::kWgrad) {
    mute::for_each(mute::make_seq<NumSpatialDimensions>{}, [&](auto i) {
      lower[NumSpatialDimensions - 1 - i] = 0;
    });
  }
  else if constexpr (ConvOp == conv::Operator::kDgrad) {
    mute::for_each(mute::make_seq<NumSpatialDimensions>{}, [&](auto i) {
      lower[NumSpatialDimensions - 1 - i] =
        (problem_shape.shape_B[i + 1] - 1) * problem_shape.dilation[i];
    });
  }

  return lower;
}

// Compute filter-mode strides in [S,R,T] order. Dgrad walks filters backward.
template <conv::Operator ConvOp, int NumSpatialDimensions>
MUTLASS_HOST_DEVICE
constexpr auto
compute_stride_srt(ConvProblemShape<ConvOp, NumSpatialDimensions> const& problem_shape) {
  mute::array<int32_t, NumSpatialDimensions> stride{};

  mute::for_each(mute::make_seq<NumSpatialDimensions>{}, [&](auto i) {
    int dilation = problem_shape.dilation[NumSpatialDimensions - 1 - i];
    stride[i] = ConvOp == conv::Operator::kDgrad ? -dilation : dilation;
  });

  return stride;
}

// Support checks stay in the convolution integration layer. The MuTe traits
// only consume already-lowered raw IM2COL parameters.
template <conv::Operator ConvOp, int NumSpatialDimensions>
MUTLASS_HOST_DEVICE
constexpr bool
is_supported_im2col_tme_problem(ConvProblemShape<ConvOp, NumSpatialDimensions> const& problem_shape) {
  bool supported = true;
  auto lower_corner_whd = compute_lower_corner_whd(problem_shape);
  auto upper_corner_whd = compute_upper_corner_whd(problem_shape);
  auto lower_srt = compute_lower_srt(problem_shape);
  auto stride_srt = compute_stride_srt(problem_shape);

  mute::for_each(mute::make_seq<NumSpatialDimensions>{}, [&](auto i) {
    supported = supported && problem_shape.lower_padding[i] >= 0 &&
                              problem_shape.upper_padding[i] >= 0 &&
                              problem_shape.traversal_stride[i] > 0 &&
                              problem_shape.dilation[i] > 0;
    if constexpr (ConvOp == conv::Operator::kDgrad) {
      supported = supported && problem_shape.traversal_stride[i] == 1;
    }

    int whd_idx = NumSpatialDimensions - 1 - i;
    int source_extent = 0;
    if constexpr (ConvOp == conv::Operator::kWgrad) {
      source_extent = problem_shape.shape_B[i + 1];
    }
    else {
      source_extent = problem_shape.shape_A[i + 1];
    }

    int lower_corner = lower_corner_whd[whd_idx];
    int upper_corner = upper_corner_whd[whd_idx];
    int lower_padding = problem_shape.lower_padding[i];
    int upper_padding = problem_shape.upper_padding[i];
    int stride = problem_shape.traversal_stride[i];
    int stride_srt_i = stride_srt[whd_idx];

    if (source_extent <= 0 || stride <= 0 || stride_srt_i == 0) {
      supported = false;
      return;
    }

    int dilation = std::abs(stride_srt_i);
    int shift = lower_corner > 0 ? mute::ceil_div(lower_corner, dilation) : 0;
    int conv_padding = shift * dilation - lower_corner;
    int output_extent = (source_extent + upper_corner - lower_corner - 1) / stride + 1;
    int padding_size = stride_srt_i > 0 ? upper_padding - upper_corner : lower_corner - lower_padding;
    int filter_extent = 1 + padding_size / stride_srt_i;
    int filter_step = stride_srt_i > 0 ? 1 : -1;
    int filter_base = lower_srt[whd_idx] / dilation + shift;
    int filter_last = filter_base + (filter_extent - 1) * filter_step;
    int min_filter_coord = filter_base < filter_last ? filter_base : filter_last;
    int max_filter_coord = filter_base > filter_last ? filter_base : filter_last;

    supported = supported && stride <= 255 && dilation <= 255;
    supported = supported && (lower_srt[whd_idx] % dilation) == 0;
    supported = supported && output_extent > 0 && filter_extent > 0;
    supported = supported && conv_padding >= 0 && conv_padding <= 255;
    supported = supported && min_filter_coord >= 0 && max_filter_coord <= 255;
  });

  return supported;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass::conv::collective::detail

/////////////////////////////////////////////////////////////////////////////////////////////////
