/***************************************************************************************************
 * Copyright (c) 2026 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

#include "mute/atom/copy_atom.hpp"
#include "mute/atom/mma_atom.hpp"
#include "mute/tensor.hpp"
#include "mutlass/arch/barrier.hpp"
#include "mutlass/array.h"
#include "mutlass/epilogue/dispatch_policy.hpp"
#include "mutlass/mutlass.h"
#include "mutlass/numeric_conversion.h"

// Adapted from MUTLASS mp32_epilogue_tme.hpp at 33ef62e8.
// Native SS TME output geometry is retained; conversion/storage uses BF16.
namespace mutlass::epilogue::collective {

template <class StrideD_, class ThreadEpilogueOp_, class TileShape_, class TiledMma_>
class MateFp4Bf16Epilogue {
 public:
  using DispatchPolicy     = Mp32CollectiveEpilogue;
  using EpilogueSchedule   = WithTme;
  using ThreadEpilogueOp   = ThreadEpilogueOp_;
  using ElementAccumulator = float;
  using ElementCompute     = float;
  using ElementC           = void;
  using ElementD           = bfloat16_t;
  using StrideC            = StrideD_;
  using StrideD            = StrideD_;
  using GmemTiledCopyC     = void;
  using GmemTiledCopyD     = mute::MP31_TME_STORE;
  using TileShape          = TileShape_;
  using TiledMma           = TiledMma_;
  using AtomShape          = typename TiledMma::AtomShape_MNK;
  static_assert(mute::size(TiledMma{}) % mutlass::NumThreadsPerWarpSquad == 0,
                "MP32 TME epilogue requires whole consumer warp squads");
  static_assert(mute::size<0>(AtomShape{}) == 128 && mute::size<1>(AtomShape{}) == 128,
                "MP32 TME epilogue requires a 128x128 SQMMA accumulator layout");
  static constexpr int NumSquads            = mute::size(TiledMma{}) / mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumBarriers          = NumSquads;
  static constexpr int ConversionVectorSize = 4;
  using EpilogueTile                        = mute::Shape<mute::_64, mute::_128>;
  static constexpr int BufferElements       = mute::size(EpilogueTile{});
  using SmemSwizzle                         = mute::Swizzle<4, 4, 4>;
  // A full swizzle period makes swizzling byte offsets equivalent to swizzling addresses.
  static constexpr int SmemAlignmentBytes =
      1 << (SmemSwizzle::num_base + SmemSwizzle::num_bits + mute::abs(SmemSwizzle::num_shft));
  using SmemLayout = mute::ComposedLayout<SmemSwizzle,
                                          mute::smem_ptr_flag_bits<mute::sizeof_bits_v<ElementD>>,
                                          decltype(mute::make_layout(EpilogueTile{}, mute::LayoutRight{}))>;

  // Storage for one consumer group; each squad reuses its buffer for two subtiles.
  struct SharedStorage {
    mute::array_aligned<ElementD, NumSquads * BufferElements, SmemAlignmentBytes> data;
  };
  using TensorStorage = SharedStorage;

  struct Arguments {
    typename ThreadEpilogueOp::Params thread{};
    ElementC const*                   ptr_C = nullptr;
    StrideC                           dC{};
    ElementD*                         ptr_D = nullptr;
    StrideD                           dD{};
  };
  using TmeStore = decltype(mute::make_tme_copy(
      GmemTiledCopyD{},
      mute::make_tensor(
          mute::make_gmem_ptr(static_cast<ElementD*>(nullptr)), mute::make_shape(int(0), int(0), int(0)), StrideD{}),
      SmemLayout{},
      EpilogueTile{}));
  struct Params {
    TmeStore tme_store;
  };

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem, Arguments const& args, void*) {
    auto shape  = mute::append<4>(problem, 1);
    auto tensor = mute::make_tensor(mute::make_gmem_ptr(args.ptr_D),
                                    mute::make_shape(mute::get<0>(shape), mute::get<1>(shape), mute::get<3>(shape)),
                                    args.dD);
    return {mute::make_tme_copy(GmemTiledCopyD{}, tensor, SmemLayout{}, EpilogueTile{})};
  }

  template <class ProblemShape>
  MUTLASS_HOST_DEVICE static bool can_implement(ProblemShape const& problem, Arguments const& args) {
    auto    shape = mute::append<4>(problem, 1);
    int64_t m = mute::size<0>(shape), n = mute::size<1>(shape), l = mute::size<3>(shape);
    int64_t row_stride = mute::get<0>(args.dD), batch_stride = mute::get<2>(args.dD);
    // TME bounds stores using the descriptor extents; M/N need not be tile-aligned.
    return m > 0 && n > 0 && l > 0 && args.ptr_D != nullptr && row_stride >= n && row_stride % 8 == 0 &&
           (l == 1 || (batch_stride >= m * row_stride && batch_stride % 8 == 0)) && args.thread.alpha == 1.0f &&
           args.thread.beta == 0.0f && args.thread.alpha_ptr == nullptr && args.thread.beta_ptr == nullptr &&
           args.thread.alpha_ptr_array == nullptr && args.thread.beta_ptr_array == nullptr;
  }

  MUTLASS_DEVICE explicit MateFp4Bf16Epilogue(Params const& params, int barrier_base)
      : params_(params), barrier_base_(barrier_base) {
  }

  MUTLASS_DEVICE static void initialize(int thread_idx, int barrier_base) {
    if (thread_idx < mutlass::NumThreadsPerWarp) {
      MUTLASS_PRAGMA_UNROLL
      for (int squad = 0; squad < NumSquads; ++squad) {
        arch::AsyncBarrier(barrier_base + squad).init(mutlass::NumWarpsPerWarpSquad);
      }
    }
  }

  template <class ProblemShape, class TileCoord, class AccEngine, class AccLayout, class Residue>
  MUTLASS_DEVICE void operator()(ProblemShape problem,
                                 TileShape,
                                 TileCoord                                 tile_coord,
                                 mute::Tensor<AccEngine, AccLayout> const& accum,
                                 TiledMma                                  mma,
                                 Residue,
                                 int   thread_idx,
                                 char* smem_buf) {
    using namespace mute;
    static_assert(size(AccLayout{}) == 128, "MP32 TME epilogue requires one SQMMA accumulator atom per squad");
    unsigned      squad            = unsigned(thread_idx) / mutlass::NumThreadsPerWarpSquad;
    unsigned      local_thread     = unsigned(thread_idx) % mutlass::NumThreadsPerWarpSquad;
    bool          issue_store      = local_thread < mutlass::NumThreadsPerWarp;
    auto          barrier          = arch::AsyncBarrier(barrier_base_ + squad);
    auto&         storage          = *reinterpret_cast<SharedStorage*>(smem_buf);
    ElementD*     buffer           = storage.data.data() + squad * BufferElements;
    auto          sD               = make_tensor(make_smem_ptr(buffer), SmemLayout{});
    auto          identity         = make_identity_tensor(take<0, 2>(TileShape{}));
    auto          coords           = mma.get_thread_slice(thread_idx).partition_C(identity);
    auto          first            = coords(0);
    int           squad_m          = int(get<0>(first)) / size<0>(AtomShape{});
    int           squad_n          = int(get<1>(first)) / size<1>(AtomShape{});
    constexpr int ValuesPerSubtile = BufferElements / mutlass::NumThreadsPerWarpSquad;
    // ((CVT_VALUES, GROUP), SUBTILE), in accumulator order.
    auto rD       = logical_divide(logical_divide(coalesce(accum), Int<ValuesPerSubtile>{}),
                             make_tile(Int<ConversionVectorSize>{}));
    auto atom_mma = make_tiled_mma(typename TiledMma::Atom{});
    // Keep the swizzle in the layout so stores calculate element offsets, not swizzled pointers.
    auto sD_store  = make_tensor(make_smem_ptr(buffer), as_position_independent_swizzle_layout(SmemLayout{}));
    auto tCsD      = atom_mma.get_thread_slice(local_thread).partition_C(sD_store);
    auto sD_chunks = logical_divide(coalesce(tCsD), Int<ValuesPerSubtile>{});
    // Reuse the first SMEM subtile for every accumulator chunk.
    auto sD_vectors = logical_divide(sD_chunks(_, _0{}), Int<ConversionVectorSize>{});
    MUTE_STATIC_ASSERT_V((size<1>(sD_vectors) == size<0, 1>(rD)));
    auto gD         = params_.tme_store.get_tme_tensor(make_shape(get<0>(problem), get<1>(problem), get<3>(problem)));
    auto copy_slice = params_.tme_store.get_slice(Int<0>{});
    auto source     = copy_slice.partition_S(sD);

    // A squad reuses its 16 KiB buffer for both halves and subsequent output tiles.
    MUTLASS_PRAGMA_UNROLL
    for (int chunk = 0; chunk < size<1>(rD); ++chunk) {
      if (issue_store) tme_store_wait();
      barrier.sync();
      MUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < size<0, 1>(rD); ++group) {
        mutlass::Array<ElementAccumulator, ConversionVectorSize> input;
        MUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < ConversionVectorSize; ++j) {
          input[j] = rD(make_coord(j, group), chunk);
        }
        auto output =
            NumericArrayConverter<ElementD, ElementAccumulator, ConversionVectorSize, ThreadEpilogueOp::kRound>{}(
                input);
        MUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < ConversionVectorSize; ++j) {
          sD_vectors(j, group) = output[j];
        }
      }
      barrier.sync();
      if (issue_store) {
        auto tile        = local_tile(gD,
                               EpilogueTile{},
                               make_coord(int(get<0>(tile_coord)) * (size<0>(TileShape{}) / size<0>(sD)) +
                                              squad_m * (size<0>(AtomShape{}) / size<0>(sD)) + chunk,
                                          int(get<1>(tile_coord)) * (size<1>(TileShape{}) / size<1>(sD)) + squad_n,
                                          get<3>(tile_coord)));
        auto destination = copy_slice.partition_D(tile);
        copy(params_.tme_store, source, destination);
        tme_store_arrive();
      }
    }
  }

  MUTLASS_DEVICE void tail(int thread_idx) const {
    if (thread_idx % mutlass::NumThreadsPerWarpSquad < mutlass::NumThreadsPerWarp) {
      mute::tme_store_wait();
    }
  }

 private:
  Params const& params_;
  int           barrier_base_;
};

}  // namespace mutlass::epilogue::collective
