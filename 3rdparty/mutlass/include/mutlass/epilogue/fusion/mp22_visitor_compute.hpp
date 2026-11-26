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
#include "mutlass/array.h"
#include "mutlass/numeric_conversion.h"
#include "mutlass/epilogue/fusion/mp22_visitor.hpp"

#include "mute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass::epilogue::fusion {

using namespace mute;
using namespace detail;

//////////////////////////////////////////////////////////////////////////////////////////////////
//
// N-nary Elementwise Compute Operation
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// The template argument provided for ComputeFn must be able to accept
// exactly one template parameter.  In Standard C++, it's OK for
// ComputeFn to have other template parameters, as long as those have
// defaults.  For example, the following struct Foo would work.
//
// template<class A, class B = A>
// struct Foo {
//   MUTLASS_HOST_DEVICE auto operator() (A a, B b);
// };
//
// However, some compilers, such as Clang, require that the argument
// take _exactly_ one template parameter.  This is nonstandard C++
// behavior.  One work-around for this case is to create a subclass
// with exactly one template parameter, and then use that subclass as
// the template argument.
//
// template<class A>
// struct FooHomogeneous : public Foo<A, B> {};
//
template <
  template <class> class ComputeFn,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class = void
>
struct Mp22Compute {
private:
  using EmptyArguments = typename Mp22VisitorImpl<>::Arguments;

  template <class Fn, class = void>
  struct ComputeArguments {
    using type = EmptyArguments;
  };


  // partial specialization for compute fns that define an Arguments member, e.g. activation hyperparameters
  template <class Fn>
  struct ComputeArguments<Fn, platform::void_t<typename Fn::Arguments>> {
    using type = typename Fn::Arguments;
  };

public:
  struct SharedStorage { };

  using Arguments = typename ComputeArguments<ComputeFn<ElementCompute>>::type;

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static mutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, musaStream_t stream) {
    return mutlass::Status::kSuccess;
  }

  MUTLASS_HOST_DEVICE
  Mp22Compute() { }

  MUTLASS_HOST_DEVICE
  Mp22Compute(Params const& params, SharedStorage const& shared_storage)
      : params(params) {}

  Params const params;

  MUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  struct Callbacks : EmptyCallbacks {
    MUTLASS_DEVICE
    Callbacks(Params const& params)
      : params(params) {}

    Params const& params;

    template <typename ElementAccumulator, typename... ElementInputs, int FragmentSize>
    MUTLASS_DEVICE Array<ElementOutput, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInputs, FragmentSize> const&... frg_inputs) {
      return transform_apply(mute::make_tuple(frg_inputs...),
        [&] (auto&& frg_input) {
          using ElementInput = typename mute::remove_cvref_t<decltype(frg_input)>::Element;
          using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FragmentSize, RoundStyle>;
          ConvertInput convert_input{};

          return convert_input(frg_input);
        },
        [&] (auto&&... cvt_frg_inputs) {
          using ComputeOutput = ComputeFn<Array<ElementCompute, FragmentSize>>;
          using ConvertOutput = NumericArrayConverter<ElementOutput, ElementCompute, FragmentSize, RoundStyle>;
          ComputeOutput compute_output{};
          ConvertOutput convert_output{};

          if constexpr (is_same_v<Arguments, EmptyArguments>) {
            return convert_output(compute_output(cvt_frg_inputs...));
          }
          else {
            return convert_output(compute_output(cvt_frg_inputs..., params));
          }
        }
      );
    }
  };

  template <class... Args>
  MUTLASS_DEVICE auto
  get_callbacks(VisitorArgs<Args...> const& args) {
    return Callbacks(params);
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
