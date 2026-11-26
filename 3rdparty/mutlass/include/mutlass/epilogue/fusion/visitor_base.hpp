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
#include "mutlass/workspace.h"

#include "mute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass::epilogue::fusion::detail {

using namespace mute;
using mute::tuple;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Visitor Implementation
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class... Ops>
struct VisitorImplBase {
  // Shared memory allocation
  using SharedStorage = tuple<typename Ops::SharedStorage...>;
  // Host side fusion arguments
  using Arguments = tuple<typename Ops::Arguments...>;
  // Device side fusion params (Kernel-entry API)
  using Params = tuple<typename Ops::Params...>;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return transform_apply(tuple<Ops...>{}, args,
      [&] (auto&& op, auto const& op_args) {
        using Op = mute::remove_cvref_t<decltype(op)>;
        return Op::to_underlying_arguments(problem_shape, op_args, workspace);
      },
      [] (auto&&... op_params) { return mute::make_tuple(op_params...); }
    );
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return transform_apply(tuple<Ops...>{}, args,
      [&] (auto&& op, auto const& op_args) {
        using Op = mute::remove_cvref_t<decltype(op)>;
        size_t op_workspace_size = Op::get_workspace_size(problem_shape, op_args);
        return round_nearest(op_workspace_size, MinWorkspaceAlignment);
      },
      [&] (auto&&... op_workspace_size) {
        return (0 + ... + op_workspace_size);
      }
    );
  }

  template <class ProblemShape>
  static mutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, musaStream_t stream) {
    Status status = Status::kSuccess;
    uint8_t* op_workspace = reinterpret_cast<uint8_t*>(workspace);
    return transform_apply(tuple<Ops...>{}, args,
      // Initialize each operation's workspace, stopping at the first error
      [&] (auto&& op, auto const& op_args) {
        if (status != Status::kSuccess) {
          return status;
        }

        using Op = mute::remove_cvref_t<decltype(op)>;
        status = Op::initialize_workspace(problem_shape, op_args, op_workspace, stream);
        if (op_workspace != nullptr) {
          size_t op_workspace_size = Op::get_workspace_size(problem_shape, op_args);
          op_workspace += round_nearest(op_workspace_size, MinWorkspaceAlignment);
        }
        return status;
      },
      // Return the final status
      [&] (auto const&...) { return status; }
    );
  }

  MUTLASS_HOST_DEVICE
  VisitorImplBase() {}

  MUTLASS_HOST_DEVICE
  VisitorImplBase(Params const& params, SharedStorage const& shared_storage)
    : ops(transform_apply(tuple<Ops...>{}, params, shared_storage,
        [] (auto&& op, auto const& op_params, auto&& op_storage) {
          using Op = mute::remove_cvref_t<decltype(op)>;
          return Op(op_params, op_storage);
        },
        [] (auto&&... ops) { return mute::make_tuple(ops...); }
      )) {}

  // Ops can store kernel persistent variables (e.g. descriptors, scalars, wave counters)
  tuple<Ops...> ops;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Base specializations so we can have standard layout params and simple aggregate initializers
template <class Op0>
struct VisitorImplBase<Op0> {

  // Retain tuple for SharedStorage because empty structs have 1B alignment
  // tuples use multiple inheritance, avoids this problem
  using SharedStorage = tuple<
    typename Op0::SharedStorage
  >;

  struct Arguments {
    typename Op0::Arguments op_0;
  };

  struct Params {
    typename Op0::Params op_0;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return Params{
      Op0::to_underlying_arguments(problem_shape, args.op_0, workspace)
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    size_t workspace_size = 0;
    workspace_size += Op0::get_workspace_size(problem_shape, args.op_0);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    return workspace_size;
  }

  template <class ProblemShape>
  static mutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, musaStream_t stream) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    status = Op0::initialize_workspace(problem_shape, args.op_0, workspace_ptr + workspace_offset, stream);
    workspace_offset += Op0::get_workspace_size(problem_shape, args.op_0);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  MUTLASS_HOST_DEVICE
  VisitorImplBase() {}

  MUTLASS_HOST_DEVICE
  VisitorImplBase(Params const& params, SharedStorage const& shared_storage)
    : ops({
        Op0(params.op_0, get<0>(shared_storage))
      }) {}

  tuple<Op0> ops;
};

template <class Op0, class Op1>
struct VisitorImplBase<Op0, Op1> {

  using SharedStorage = tuple<
    typename Op0::SharedStorage,
    typename Op1::SharedStorage
  >;

  struct Arguments {
    typename Op0::Arguments op_0;
    typename Op1::Arguments op_1;
  };

  struct Params {
    typename Op0::Params op_0;
    typename Op1::Params op_1;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return Params{
      Op0::to_underlying_arguments(problem_shape, args.op_0, workspace),
      Op1::to_underlying_arguments(problem_shape, args.op_1, workspace)
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    size_t workspace_size = 0;
    workspace_size += Op0::get_workspace_size(problem_shape, args.op_0);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    workspace_size += Op1::get_workspace_size(problem_shape, args.op_1);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    return workspace_size;
  }

  template <class ProblemShape>
  static mutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, musaStream_t stream) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    status = Op0::initialize_workspace(problem_shape, args.op_0, workspace_ptr + workspace_offset, stream);
    workspace_offset += Op0::get_workspace_size(problem_shape, args.op_0);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = Op1::initialize_workspace(problem_shape, args.op_1, workspace_ptr + workspace_offset, stream);
    workspace_offset += Op1::get_workspace_size(problem_shape, args.op_1);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  MUTLASS_HOST_DEVICE
  VisitorImplBase() {}

  MUTLASS_HOST_DEVICE
  VisitorImplBase(Params const& params, SharedStorage const& shared_storage)
    : ops({
        Op0(params.op_0, get<0>(shared_storage)),
        Op1(params.op_1, get<1>(shared_storage))
      }) {}

  tuple<Op0, Op1> ops;
};

template <class Op0, class Op1, class Op2>
struct VisitorImplBase<Op0, Op1, Op2> {

  using SharedStorage = tuple<
    typename Op0::SharedStorage,
    typename Op1::SharedStorage,
    typename Op2::SharedStorage
  >;

  struct Arguments {
    typename Op0::Arguments op_0;
    typename Op1::Arguments op_1;
    typename Op2::Arguments op_2;
  };

  struct Params {
    typename Op0::Params op_0;
    typename Op1::Params op_1;
    typename Op2::Params op_2;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return Params{
      Op0::to_underlying_arguments(problem_shape, args.op_0, workspace),
      Op1::to_underlying_arguments(problem_shape, args.op_1, workspace),
      Op2::to_underlying_arguments(problem_shape, args.op_2, workspace)
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    size_t workspace_size = 0;
    workspace_size += Op0::get_workspace_size(problem_shape, args.op_0);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    workspace_size += Op1::get_workspace_size(problem_shape, args.op_1);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    workspace_size += Op2::get_workspace_size(problem_shape, args.op_2);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    return workspace_size;
  }

  template <class ProblemShape>
  static mutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, musaStream_t stream) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    status = Op0::initialize_workspace(problem_shape, args.op_0, workspace_ptr + workspace_offset, stream);
    workspace_offset += Op0::get_workspace_size(problem_shape, args.op_0);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = Op1::initialize_workspace(problem_shape, args.op_1, workspace_ptr + workspace_offset, stream);
    workspace_offset += Op1::get_workspace_size(problem_shape, args.op_1);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = Op2::initialize_workspace(problem_shape, args.op_2, workspace_ptr + workspace_offset, stream);
    workspace_offset += Op2::get_workspace_size(problem_shape, args.op_2);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  MUTLASS_HOST_DEVICE
  VisitorImplBase() {}

  MUTLASS_HOST_DEVICE
  VisitorImplBase(Params const& params, SharedStorage const& shared_storage)
    : ops({
        Op0(params.op_0, get<0>(shared_storage)),
        Op1(params.op_1, get<1>(shared_storage)),
        Op2(params.op_2, get<2>(shared_storage))
      }) {}

  tuple<Op0, Op1, Op2> ops;
};

template <class Op0, class Op1, class Op2, class Op3>
struct VisitorImplBase<Op0, Op1, Op2, Op3> {

  using SharedStorage = tuple<
    typename Op0::SharedStorage,
    typename Op1::SharedStorage,
    typename Op2::SharedStorage,
    typename Op3::SharedStorage
  >;

  struct Arguments {
    typename Op0::Arguments op_0;
    typename Op1::Arguments op_1;
    typename Op2::Arguments op_2;
    typename Op3::Arguments op_3;
  };

  struct Params {
    typename Op0::Params op_0;
    typename Op1::Params op_1;
    typename Op2::Params op_2;
    typename Op3::Params op_3;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return Params{
      Op0::to_underlying_arguments(problem_shape, args.op_0, workspace),
      Op1::to_underlying_arguments(problem_shape, args.op_1, workspace),
      Op2::to_underlying_arguments(problem_shape, args.op_2, workspace),
      Op3::to_underlying_arguments(problem_shape, args.op_3, workspace)
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    size_t workspace_size = 0;
    workspace_size += Op0::get_workspace_size(problem_shape, args.op_0);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    workspace_size += Op1::get_workspace_size(problem_shape, args.op_1);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    workspace_size += Op2::get_workspace_size(problem_shape, args.op_2);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    workspace_size += Op3::get_workspace_size(problem_shape, args.op_3);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    return workspace_size;
  }

  template <class ProblemShape>
  static mutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, musaStream_t stream) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    status = Op0::initialize_workspace(problem_shape, args.op_0, workspace_ptr + workspace_offset, stream);
    workspace_offset += Op0::get_workspace_size(problem_shape, args.op_0);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = Op1::initialize_workspace(problem_shape, args.op_1, workspace_ptr + workspace_offset, stream);
    workspace_offset += Op1::get_workspace_size(problem_shape, args.op_1);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = Op2::initialize_workspace(problem_shape, args.op_2, workspace_ptr + workspace_offset, stream);
    workspace_offset += Op2::get_workspace_size(problem_shape, args.op_2);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = Op3::initialize_workspace(problem_shape, args.op_3, workspace_ptr + workspace_offset, stream);
    workspace_offset += Op3::get_workspace_size(problem_shape, args.op_3);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  MUTLASS_HOST_DEVICE
  VisitorImplBase() {}

  MUTLASS_HOST_DEVICE
  VisitorImplBase(Params const& params, SharedStorage const& shared_storage)
    : ops({
        Op0(params.op_0, get<0>(shared_storage)),
        Op1(params.op_1, get<1>(shared_storage)),
        Op2(params.op_2, get<2>(shared_storage)),
        Op3(params.op_3, get<3>(shared_storage))
      }) {}

  tuple<Op0, Op1, Op2, Op3> ops;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass::epilogue::fusion::detail

/////////////////////////////////////////////////////////////////////////////////////////////////
