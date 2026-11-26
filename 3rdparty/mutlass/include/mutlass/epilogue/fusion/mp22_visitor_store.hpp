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

#include "mutlass/epilogue/fusion/mp22_visitor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass::epilogue::fusion {

using namespace mute;
using namespace detail;

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Elementwise Store Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class Element,
  FloatRoundStyle RoundStyle,
  class StrideMNL
>
struct Mp22AuxStore {
  constexpr static bool is_m_major = epilogue::collective::detail::is_m_major<StrideMNL>();
  constexpr static bool is_n_major = epilogue::collective::detail::is_n_major<StrideMNL>();

  struct SharedStorage {};

  struct Arguments {
    Element* ptr_aux = nullptr;
    StrideMNL dAux = {};
  };

  struct Params {
    Element* ptr_aux;;
    StrideMNL dAux;
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
  Mp22AuxStore() {}

  MUTLASS_HOST_DEVICE
  Mp22AuxStore(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) {}

  Params const* params_ptr;

  MUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  template <class GTensor,
            class RTensor,
            class CTensor,
            class TiledCopyR2G,
            class ResidueMN,
            class TileShapeMNK>
  struct Callbacks : EmptyCallbacks {
    MUTLASS_DEVICE
    Callbacks(
      GTensor&& tRG_gAux,
      RTensor&& tRG_rAux,
      CTensor&& tRG_cAux,
      TiledCopyR2G tiled_r2g,
      ResidueMN residue_mn,
      TileShapeMNK tile_shape,
      Params const* params_ptr
    ) :
      tRG_gAux(mute::forward<GTensor>(tRG_gAux)),
      tRG_rAux(mute::forward<RTensor>(tRG_rAux)),
      tRG_cAux(mute::forward<CTensor>(tRG_cAux)),
      tiled_r2g(tiled_r2g),
      residue_mn(residue_mn),
      tile_shape(tile_shape),
      params_ptr(params_ptr) {}

    GTensor tRG_gAux;
    RTensor tRG_rAux;
    CTensor tRG_cAux;
    TiledCopyR2G tiled_r2g;
    ResidueMN residue_mn;
    TileShapeMNK tile_shape;
    Params const* params_ptr;

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    MUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
      Array<ElementInput, FragmentSize> const& frg_input) {
      using ConvertInput = NumericArrayConverter<Element, ElementInput, FragmentSize, RoundStyle>;
      ConvertInput convert_input{};

      Tensor tRG_rAux_frg = recast<Array<Element, FragmentSize>>(coalesce(tRG_rAux));                          // (EPI_V)
      tRG_rAux_frg(epi_v) = convert_input(frg_input);

      return frg_input;
    }

    MUTLASS_DEVICE void
    end_step(int epi_m, int epi_n, int step_idx) {
      auto src_v = tRG_rAux;
      auto dst_v = tRG_gAux(_, _, _, epi_m, epi_n);
      auto coord_v = tRG_cAux(_, _, _, epi_m, epi_n);

      if (get<0>(residue_mn) >= get<0>(tile_shape) &&
          get<1>(residue_mn) >= get<1>(tile_shape)) {
        copy(tiled_r2g.with(params_ptr->desc_aux), src_v, dst_v);
      } else {
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(dst_v); ++i) {
          if (elem_less(coord_v(i), residue_mn)) {
            dst_v(i) = src_v(i);
          }
        }
      }
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
    Tensor cAux = make_identity_tensor(make_shape(size<0>(gAux), size<1>(gAux)));
    Tensor cAux_epi = flat_divide(cAux, args.epi_tile);

    auto thread_copy = args.tiled_copy.get_thread_slice(args.thread_idx);
    Tensor tDgAux    = thread_copy.partition_D(gAux_epi);
    constexpr int vec_elem = decltype(max_common_vector(tDgAux, args.tDgD))::value;
    constexpr int vec_bits = vec_elem * sizeof_bits_v<Element>;
    using CopyAtomR2G      = Copy_Atom<
            UniversalCopy<uint_bit_t<mute::min(1024, vec_bits)>>, Element>;
    using TiledCopyR2G = TiledCopy<CopyAtomR2G,
                            typename decltype(args.tiled_copy)::TiledLayout_TV,
                            typename decltype(args.tiled_copy)::Tiler_MN>;

    TiledCopyR2G tiled_copy_r2g{};
    auto thr_copy_r2g = tiled_copy_r2g.get_slice(args.thread_idx);
    auto tRG_gAux     = thr_copy_r2g.partition_D(gAux_epi);
    auto tRG_rAux     = thr_copy_r2g.retile_S(make_tensor<Element>(take<0, 3>(shape(tRG_gAux))));
    auto tRG_cAux     = thr_copy_r2g.partition_D(cAux_epi);

    return Callbacks<
      decltype(tRG_gAux), decltype(tRG_rAux), decltype(tRG_cAux),
      TiledCopyR2G, decltype(args.residue_mn), decltype(args.tile_shape_mnk)>(
          mute::move(tRG_gAux),
          mute::move(tRG_rAux),
          mute::move(tRG_cAux),
          tiled_copy_r2g,
          args.residue_mn,
          args.tile_shape_mnk,
          params_ptr);
  }
};

} // namespace mutlass::epilogue::fusion
