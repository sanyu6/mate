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
#pragma once

#include "mutlass/mutlass.h"

#include "mute/tensor.hpp"

#include "mutlass/epilogue/dispatch_policy.hpp"
#include "mutlass/epilogue/fusion/callbacks.hpp"
#include "mutlass/epilogue/fusion/mp22_visitor.hpp"
#include "mutlass/epilogue/fusion/mp22_visitor_load.hpp"
#include "mutlass/epilogue/fusion/mp22_visitor_store.hpp"
#include "mutlass/epilogue/fusion/mp22_visitor_compute.hpp"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = alpha * acc
template <
  int Stages,
  int FragmentSize,
  class ElementOutput,
  class ElementCompute,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
  epilogue::Mp22CollectiveEpilogue<Stages, FragmentSize>,
  fusion::ScaledAcc<ElementOutput, ElementCompute, ElementScalar, RoundStyle>,
  CtaTileShapeMNK,
  EpilogueTile
> : Mp22EVT<Mp22Compute<multiplies, ElementOutput, ElementCompute, RoundStyle>,
      Mp22ScalarBroadcast<ElementScalar>,
      Mp22AccFetch
    > {
  using Impl =
    Mp22EVT<Mp22Compute<multiplies, ElementOutput, ElementCompute, RoundStyle>,
      Mp22ScalarBroadcast<ElementScalar>,
      Mp22AccFetch
    >;
  using Operation = fusion::ScaledAcc<ElementOutput, ElementCompute, ElementScalar, RoundStyle>;

  struct Arguments {
    // Give a name and flat ordering to the fusion callback args
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    // Conversion to the args expected by the visitor implementation
    // to_underlying_arguments will implicitly call this
    operator typename Impl::Arguments() const {
      return
        {    // binary op : alpha * acc
          {{alpha}, {alpha_ptr}}, // leaf args : alpha
          {},                     // leaf args : acc
          {} // binary args : multiplies
        };   // end binary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = alpha * acc + beta * C
template<
  class ElementOutput,
  class ElementCompute,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Mp22LinearCombination =
  Mp22EVT<Mp22Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + (alpha * acc)
    Mp22ScalarBroadcast<ElementScalar>, // beta
    Mp22SrcFetch<ElementSource>, // C
    Mp22EVT<Mp22Compute<multiplies, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc
      Mp22ScalarBroadcast<ElementScalar>, // alpha
      Mp22AccFetch // acc
    >
  >;

template <
  int Stages,
  int FragmentSize,
  class ElementOutput,
  class ElementCompute,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Mp22CollectiveEpilogue<Stages, FragmentSize>,
    fusion::LinearCombination<ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Mp22LinearCombination<typename mutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementSource, ElementScalar, RoundStyle> {

  using Impl = Mp22LinearCombination<typename mutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinearCombination<ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    operator typename Impl::Arguments() const {
      return
        {    // ternary op : beta * C + (alpha * acc)
          {{beta}, {beta_ptr}}, // leaf args : beta
          {},                   // leaf args : C
          {                     // binary op : alpha * acc
            {{alpha}, {alpha_ptr}}, // leaf args : alpha
            {},                     // leaf args : acc
            {}                  // binary args : multiplies
          },                    // end binary op
          {} // ternary args : multiply_add
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

} // namespace mutlass::epilogue::fusion
