#pragma once

// The MUTLASS builders require the mainloop definitions before the epilogue.
// clang-format off
#include <musa_runtime.h>
#include <mute/tensor.hpp>
#include <mutlass/device_kernel.h>
#include <mutlass/float_subbyte.h>
#include <mutlass/float8.h>
#include <mutlass/gemm/collective/collective_builder.hpp>
#include <mutlass/epilogue/collective/collective_builder.hpp>
#include <mutlass/epilogue/fusion/operations.hpp>
#include <mutlass/gemm/kernel/gemm_universal.hpp>
#include <mutlass/util/packed_stride.hpp>
#include "mate/mega_moe/prefill/fp4_fp4/ss_bf16_epilogue.hpp"
// clang-format on

namespace mega_moe::prefill::mp32::fp4_fp4 {

// Equal-length groups use the native batch dimension in one persistent launch.
// Ragged expert lengths and MoE dispatch are separate milestones.
template <int TileM, int TileN, int TileK = 256>
struct SsGroupedGemm {
  static_assert(TileK == 256 || TileK == 512);
  static constexpr int kBlockK           = TileK;
  static constexpr int kScaleBlock       = 32;
  static constexpr int kValuesPerByte    = 2;
  using Input                            = mutlass::mx_float4_t<mutlass::float_e2m1_t>;
  using Element                          = typename Input::DataType;
  using Scale                            = typename Input::ScaleFactorType;
  using Output                           = mutlass::bfloat16_t;
  using Tile                             = mute::Shape<mute::Int<TileM>, mute::Int<TileN>, mute::Int<kBlockK>>;
  using Cluster                          = mute::Shape<mute::_1, mute::_1, mute::_1>;
  static constexpr int kOperandAlignment = 128 / mutlass::sizeof_bits<Element>::value;
  static constexpr int kOutputAlignment  = 128 / mutlass::sizeof_bits<Output>::value;

  template <class Stages>
  using Builder        = mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp32,
                                                                      mutlass::arch::OpClassBlockScaledTensorOp,
                                                                      Input,
                                                                      mutlass::layout::RowMajor,
                                                                      kOperandAlignment,
                                                                      Input,
                                                                      mutlass::layout::ColumnMajor,
                                                                      kOperandAlignment,
                                                                      float,
                                                                      Tile,
                                                                      Cluster,
                                                                      Stages,
                                                                      mutlass::gemm::KernelTmeWarpSpecializedMp32>;
  using MainloopConfig = Builder<mutlass::gemm::collective::StageCount<2>>;
  using ThreadOp       = mutlass::epilogue::thread::
      LinearCombination<Output, 4, float, float, mutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;
  using Epilogue =
      mutlass::epilogue::collective::MateFp4Bf16Epilogue<mutlass::detail::TagToStrideC_t<mutlass::layout::RowMajor>,
                                                         ThreadOp,
                                                         Tile,
                                                         typename MainloopConfig::TiledMma>;
  using Mainloop = typename Builder<
      mutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>>::CollectiveOp;
  using Kernel = mutlass::gemm::kernel::
      GemmUniversal<mute::Shape<int, int, int, int>, Mainloop, Epilogue, mutlass::gemm::PersistentScheduler>;
  using Arguments = typename Kernel::Arguments;
  using Params    = typename Kernel::Params;
  static_assert(mute::is_same_v<typename MainloopConfig::TiledMma, typename Mainloop::TiledMma>);
  static_assert(mute::size<2>(typename Mainloop::TiledMma::AtomShape_MNK{}) == kBlockK);
  static_assert(Kernel::SharedStorageSize <= 256 * 1024);

  static Arguments arguments(void const* a,
                             void const* sa,
                             void const* b,
                             void const* sb,
                             void*       out,
                             int         groups,
                             int         m,
                             int         n,
                             int         k,
                             int         device,
                             int         mps) {
    auto      da = mutlass::make_mute_packed_stride(typename Kernel::StrideA{}, mute::make_shape(m, k, groups));
    auto      db = mutlass::make_mute_packed_stride(typename Kernel::StrideB{}, mute::make_shape(n, k, groups));
    auto      dc = mutlass::make_mute_packed_stride(typename Kernel::StrideC{}, mute::make_shape(m, n, groups));
    auto      dd = mutlass::make_mute_packed_stride(typename Kernel::StrideD{}, mute::make_shape(m, n, groups));
    Arguments args{mutlass::gemm::GemmUniversalMode::kBatched,
                   {m, n, k, groups},
                   {reinterpret_cast<Element const*>(a),
                    da,
                    reinterpret_cast<Element const*>(b),
                    db,
                    reinterpret_cast<Scale const*>(sa),
                    reinterpret_cast<Scale const*>(sb)},
                   {{1.0f}, nullptr, dc, static_cast<Output*>(out), dd},
                   {device, mps}};
    args.scheduler.swizzle_size = 2;
    args.scheduler.raster_order = Kernel::TileScheduler::RasterOrderOptions::AlongN;
    return args;
  }
};

template <class Impl>
__global__ __launch_bounds__(Impl::Kernel::MaxThreadsPerBlock) void grouped_gemm_kernel(
    MUTLASS_GRID_CONSTANT typename Impl::Params const params) {
  using Kernel = typename Impl::Kernel;
  extern __shared__ char __attribute__((aligned(Kernel::SmemAlignmentBytes))) smem[];
  Kernel                                                                      kernel;
  kernel(params, smem);
}

}  // namespace mega_moe::prefill::mp32::fp4_fp4
