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

#include "visitor_base.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass::epilogue::fusion {

using namespace mute;
using mute::tuple;

namespace detail {

template <
  class ProblemShapeMNKL,
  class TileShapeMNK,
  class TileCoordMNKL,
  class ResidueMN,
  class EpilogueTile,
  class TiledCopy,
  class CoordTensor,
  class ThrCoordTensor,
  class SrcTensor,
  class DstTensor
>
struct VisitorArgs {
  ProblemShapeMNKL problem_shape_mnkl;
  TileShapeMNK tile_shape_mnk;
  TileCoordMNKL tile_coord_mnkl;
  ResidueMN residue_mn;
  EpilogueTile epi_tile;
  TiledCopy tiled_copy;
  int thread_idx;
  CoordTensor cD;
  ThrCoordTensor tCcD;
  SrcTensor const& tCrC;
  DstTensor const& tDgD;

  MUTLASS_DEVICE
  VisitorArgs(
     ProblemShapeMNKL problem_shape_mnkl,
     TileShapeMNK tile_shape_mnk,
     TileCoordMNKL tile_coord_mnkl,
     ResidueMN residue_mn,
     EpilogueTile epi_tile,
     TiledCopy tiled_copy,
     int thread_idx,
     CoordTensor cD,
     ThrCoordTensor tCcD,
     SrcTensor const& tCrC,
     DstTensor const& tDgD)
  : problem_shape_mnkl(problem_shape_mnkl),
    tile_shape_mnk(tile_shape_mnk),
    tile_coord_mnkl(tile_coord_mnkl),
    residue_mn(residue_mn),
    epi_tile(epi_tile),
    tiled_copy(tiled_copy),
    thread_idx(thread_idx),
    cD(cD),
    tCcD(tCcD),
    tCrC(tCrC),
    tDgD(tDgD) {}
};

template <class... Ops>
struct Mp22VisitorImpl : VisitorImplBase<Ops...> {
  using Impl = VisitorImplBase<Ops...>;
  using Params = typename Impl::Params;
  using SharedStorage = typename Impl::SharedStorage;

  MUTLASS_HOST_DEVICE
  Mp22VisitorImpl() {}

  MUTLASS_HOST_DEVICE
  Mp22VisitorImpl(Params const& params, SharedStorage const& shared_storage)
    : Impl(params, shared_storage) {}

  using Impl::ops;

  MUTLASS_DEVICE bool
  is_C_load_needed() const {
    return apply(ops,
      [] (auto const&... op) {
        return (false || ... || op.is_C_load_needed());
      }
    );
  }

  template <class CallbacksTuple>
  struct Callbacks {
    // Callbacks can store non-persistent variables (e.g. tensors) or copies of persistent variables
    CallbacksTuple callbacks_tuple;

    MUTLASS_DEVICE void
    begin_epilogue() {
      for_each(callbacks_tuple,
        [] (auto& callbacks) {
          callbacks.begin_epilogue();
        }
      );
    }

    MUTLASS_DEVICE void
    begin_step(int epi_m, int epi_n, int step_idx) {
      for_each(callbacks_tuple,
        [&] (auto& callbacks) {
          callbacks.begin_step(epi_m, epi_n, step_idx);
        }
      );
    }

    // Perform the fused elementwise computation
    template <typename ElementAccumulator, typename... ElementInputs, int FragmentSize>
    MUTLASS_DEVICE auto // returns an Array
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInputs, FragmentSize> const&... frg_inputs) // depends on the N-naryness of the op
      = delete; // Must be implemented for each operation

    MUTLASS_DEVICE void
    end_step(int epi_m, int epi_n, int step_idx) {
      for_each(callbacks_tuple,
        [&] (auto& callbacks) {
          callbacks.end_step(epi_m, epi_n, step_idx);
        }
      );
    }

    MUTLASS_DEVICE void
    end_epilogue() {
      for_each(callbacks_tuple,
        [&] (auto& callbacks) {
          callbacks.end_epilogue();
        }
      );
    }
  };


  // Callbacks factory
  // All operations must redefine this
  template <class... Args>
  MUTLASS_DEVICE auto
  get_callbacks(VisitorArgs<Args...> const& args) {
    return transform_apply(ops,
        [&] (auto& op) {
          return op.get_callbacks(args);
        },
        [] (auto&&... callbacks) {
          auto callbacks_tuple = mute::make_tuple(callbacks...);
          return Callbacks<decltype(callbacks_tuple)>{callbacks_tuple};
        }
    );
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////


// Convenience aliases
using EmptyCallbacks = Mp22VisitorImpl<>::Callbacks<mute::tuple<>>;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace detail

using namespace detail;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Tree visitor
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class NodeOp, class... ChildOps>
struct Mp22TreeVisitor : Mp22VisitorImpl<ChildOps..., NodeOp> {

  using Impl = Mp22VisitorImpl<ChildOps..., NodeOp>;
  using Params = typename Impl::Params;
  using SharedStorage = typename Impl::SharedStorage;

  MUTLASS_HOST_DEVICE
  Mp22TreeVisitor() {}

  MUTLASS_HOST_DEVICE
  Mp22TreeVisitor(
      Params const& params,
      SharedStorage const& shared_storage)
    : Impl(params, shared_storage) {}


  template <class CallbacksImpl>
  struct Callbacks : CallbacksImpl {
    MUTLASS_DEVICE
    Callbacks(CallbacksImpl&& impl)
      : CallbacksImpl(mute::forward<CallbacksImpl>(impl)) {}

    using CallbacksImpl::callbacks_tuple;

    template <typename ElementAccumulator, int FragmentSize>
    MUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      constexpr int Rm1 = sizeof...(ChildOps);
      return mute::detail::tapply(callbacks_tuple,
        [&] (auto& child_callbacks) {
          return child_callbacks.visit(frg_acc, epi_v, epi_m, epi_n);
        },
        [&] (auto&&... frg_inputs) {
          return get<Rm1>(callbacks_tuple).visit(frg_acc, epi_v, epi_m, epi_n, frg_inputs...);
        },
        make_seq<Rm1>{}
      );
    }
  };

  // Callbacks factory
  template <class... Args>
  MUTLASS_DEVICE auto
  get_callbacks(VisitorArgs<Args...> const& args) {
    return Callbacks<
      decltype(Mp22VisitorImpl<ChildOps..., NodeOp>::get_callbacks(args))>(
        Mp22VisitorImpl<ChildOps..., NodeOp>::get_callbacks(args));
  }
};

template <class NodeOp, class... ChildOps>
using Mp22EVT = Mp22TreeVisitor<NodeOp, ChildOps...>;

} // namespace mutlass::epilogue::fusion
