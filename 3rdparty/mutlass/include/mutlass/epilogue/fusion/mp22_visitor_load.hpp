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
#include "mutlass/epilogue/fusion/mp22_visitor.hpp"

#include "mute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass::epilogue::fusion {

using namespace mute;
using namespace detail;

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Elementwise Fetch Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// returns accumulator
struct Mp22AccFetch : Mp22VisitorImpl<> {

  using Mp22VisitorImpl<>::Mp22VisitorImpl;

  struct Callbacks : EmptyCallbacks {
    template <typename ElementAccumulator, int FragmentSize>
    MUTLASS_DEVICE Array<ElementAccumulator, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      return frg_acc;
    }
  };

  template <class... Args>
  MUTLASS_DEVICE auto
  get_callbacks(VisitorArgs<Args...> const& args) {
    return Callbacks{};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// return C
template <class Element>
struct Mp22SrcFetch : Mp22VisitorImpl<> {
  MUTLASS_DEVICE bool
  is_C_load_needed() const {
    return not is_void_v<Element>;
  }

  using Mp22VisitorImpl<>::Mp22VisitorImpl;

  template <class SrcTensor>
  struct Callbacks : EmptyCallbacks {
    MUTLASS_DEVICE
    Callbacks(SrcTensor const& tCrC)
      : tCrC(tCrC) {}

    SrcTensor const& tCrC;

    template <typename ElementAccumulator, int FragmentSize>
    MUTLASS_DEVICE Array<typename SrcTensor::value_type, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      return recast<Array<typename SrcTensor::value_type, FragmentSize>>(tCrC)(epi_v);
    }
  };

  template <class... Args>
  MUTLASS_DEVICE auto
  get_callbacks(VisitorArgs<Args...> const& args) {
    return Callbacks<mute::remove_cvref_t<decltype(args.tCrC)>>(args.tCrC);
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////
//
// Broadcast Load Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// Scalar broadcast
// Supports reduction over multiple broadcasts to support fusions such as fp8 scaling factors
template <
  class Element,
  class StrideMNL = Stride<_0, _0, _0>,
  int BroadcastCount = 1,
  template <class> class ReductionFn = multiplies
>
struct Mp22ScalarBroadcast {
  static_assert(
    (mute::is_same_v<StrideMNL, Stride<_0, _0, _0>>) || // scalar broadcast, e.g. alpha
    (mute::is_same_v<StrideMNL, Stride<_0, _0, _1>>) || // batched scalar broadcast, e.g. per-batch alpha
    (mute::is_same_v<StrideMNL, Stride<_0, _0,int>>));

  struct SharedStorage {};

  struct Arguments {
    Element scalars[BroadcastCount] = {};
    Element const* scalar_ptrs[BroadcastCount] = {};
    StrideMNL dScalar = {};
  };

  using Params = Arguments;

  template <typename ProblemShape>
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
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args,
                       void* workspace, musaStream_t stream) {
    return mutlass::Status::kSuccess;
  }

  MUTLASS_HOST_DEVICE
  Mp22ScalarBroadcast() {}

  MUTLASS_HOST_DEVICE
  Mp22ScalarBroadcast(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) {
    // Get the scalar for non-batched broadcast
    if constexpr (mute::is_same_v<StrideMNL, Stride<_0, _0, _0>>) {
      update_scalar();
    }
  }

  Element scalar;
  Params const* params_ptr;

  MUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  struct Callbacks : EmptyCallbacks {
    MUTLASS_DEVICE
    Callbacks(Element scalar)
      : scalar(scalar) {}

    Element scalar;

    template <typename ElementAccumulator, int FragmentSize>
    MUTLASS_DEVICE Array<Element, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      Array<Element, FragmentSize> frg_scalar;
      frg_scalar.fill(scalar);

      return frg_scalar;
    }
  };

  template <class... Args>
  MUTLASS_DEVICE auto
  get_callbacks(VisitorArgs<Args...> const& args) {
    if constexpr (
      mute::is_same_v<StrideMNL, Stride<_0, _0, _0>> ||
      mute::is_same_v<StrideMNL, Stride<_0, _0,int>>) {
      auto [m_coord, n_coord, k_coord, l_coord] = args.tile_coord_mnkl;
      update_scalar(l_coord);
    }

    return Callbacks(scalar);
  }

private:
  MUTLASS_DEVICE void
  update_scalar(int l_coord = 0) {
    int l_offset = l_coord * size<2>(params_ptr->dScalar);

    if (params_ptr->scalar_ptrs[0] != nullptr) {
      scalar = params_ptr->scalar_ptrs[0][l_offset];
    } else {
      // batch stride is ignored for nullptr fallback
      scalar = params_ptr->scalars[0];
    }

    // Do reduction over multiple broadcasts if necessary
    ReductionFn<Element> reduction_fn;
    MUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < BroadcastCount; ++i) {
      if (params_ptr->scalar_ptrs[i] != nullptr) {
        scalar = reduction_fn(scalar, params_ptr->scalar_ptrs[i][l_offset]);
      } else {
        // batch stride is ignored for nullptr fallback
        scalar = reduction_fn(scalar, params_ptr->scalars[i]);
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Elementwise Load Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class Element,
  class StrideMNL
>
struct Mp22AuxLoad {

  constexpr static bool is_m_major = epilogue::collective::detail::is_m_major<StrideMNL>();
  constexpr static bool is_n_major = epilogue::collective::detail::is_n_major<StrideMNL>();

  struct SharedStorage {};

  struct Arguments {
    Element* ptr_aux = nullptr;
    Element null_default = Element(0);
    StrideMNL dAux = {};
  };

  struct Params {
    Element* ptr_aux;
    Element null_default = Element(0);
    StrideMNL dAux = {};
    RobustDescriptor desc_aux;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    auto problem_shape_MNKL = mute::append<4>(problem_shape, _1{});
    auto stride_aux         = mute::append<3>(args.dAux, _0{});

    auto [M, N, K, L] = problem_shape_MNKL;
    auto [dM, dN, dL] = stride_aux;

    size_t elem = 0;
    if constexpr (is_n_major) {
      elem = N * dM;
    } else if constexpr (is_m_major) {
      elem = M * dN;
    }

    if (dL != 0) {
      elem *= L;
    }

    auto desc_aux = make_robust_desc(args.ptr_aux, elem);

    return {
      args.ptr_aux,
      args.null_default,
      args.dAux,
      desc_aux
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static mutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args,
                       void* workspace, musaStream_t stream) {
    return mutlass::Status::kSuccess;
  }

  MUTLASS_HOST_DEVICE
  Mp22AuxLoad() {}

  MUTLASS_HOST_DEVICE
  Mp22AuxLoad(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) {}

  Params const* params_ptr;

  MUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  template <class GTensor,
            class RTensor,
            class TiledCopyG2R>
  struct Callbacks : EmptyCallbacks {
    MUTLASS_DEVICE
    Callbacks(
      GTensor&& tGRgAux,
      RTensor&& tGRrAux,
      TiledCopyG2R tiled_g2r,
      Params const* params_ptr
    ):
      tGRgAux(mute::forward<GTensor>(tGRgAux)),
      tGRrAux(mute::forward<RTensor>(tGRrAux)),
      tiled_g2r(tiled_g2r),
      params_ptr(params_ptr) { }

    GTensor tGRgAux;
    RTensor tGRrAux;
    TiledCopyG2R tiled_g2r;
    Params const* params_ptr;

    MUTLASS_DEVICE void
    begin_step(int epi_m, int epi_n, int step_idx) {
      auto src_v = tGRgAux(_, _, _, epi_m, epi_n);
      auto dst_v = tGRrAux;
      copy(tiled_g2r.with(params_ptr->desc_aux), src_v, dst_v);
    }

    template <class ElementAccumulator, int FragmentSize>
    MUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frag_acc, int epi_v, int epi_m, int epi_n) {
      Tensor tGR_rAux_frg = recast<Array<Element, FragmentSize>>(coalesce(tGRrAux));                          // (EPI_V)
      return tGR_rAux_frg(epi_v);
    }
  };

  template <class... Args>
  MUTLASS_DEVICE auto
  get_callbacks(VisitorArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;

    Tensor mAux = make_tensor(
      make_gmem_ptr(params_ptr->ptr_aux),
      make_shape(M, N, L),
      params_ptr->dAux); // (M, N, L)

    Tensor gAux = local_tile(mAux, take<0, 2>(args.tile_shape_mnk), make_coord(m, n, l)); // (BLK_M, BLK_N)
    Tensor gAux_epi = flat_divide(gAux, args.epi_tile); // (EPI_M, EPI_N, TILE_M, TILE_N)
    auto thread_copy = args.tiled_copy.get_thread_slice(args.thread_idx);
    Tensor tDgAux    = thread_copy.partition_S(gAux_epi);
    constexpr int vec_elem = decltype(max_common_vector(tDgAux, args.tDgD))::value;
    constexpr int vec_bits = vec_elem * sizeof_bits_v<Element>;
    using CopyAtomG2R      = Copy_Atom<
            UniversalCopy<uint_bit_t<mute::min(1024, vec_bits)>>, Element>;
    using TiledCopyG2R = TiledCopy<CopyAtomG2R,
                            typename decltype(args.tiled_copy)::TiledLayout_TV,
                            typename decltype(args.tiled_copy)::Tiler_MN>;
    TiledCopyG2R tiled_copy_g2r{};
    auto thr_copy_g2r = tiled_copy_g2r.get_slice(args.thread_idx);
    auto tGRgAux      = thr_copy_g2r.partition_S(gAux_epi);
    auto tGRrAux      = thr_copy_g2r.retile_D(make_tensor<Element>(take<0, 3>(shape(tGRgAux))));

    return Callbacks<
      decltype(tGRgAux), decltype(tGRrAux), TiledCopyG2R>(
          mute::move(tGRgAux),
          mute::move(tGRrAux),
          tiled_copy_g2r,
          params_ptr);
  }
};
} // namespace mutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
