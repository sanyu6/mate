#pragma once
#include "mate/mega_moe/prefill/fp4_fp4/ss_grouped_gemm.hpp"
#include "mate/mega_moe/prefill/mp32_prefill_common.hpp"
namespace mega_moe::prefill::mp32::fp4_fp4 {
using namespace mute;

// Original prefill compact-pool contract: shared activation batch zero,
// per-expert weight batches. The producer warp is chosen by the stage caller.
template <int N, int K, int TileN = 256, int TileK = 256>
struct SsPoolMainloop {
  using Impl       = SsGroupedGemm<256, TileN, TileK>;
  using Collective = typename Impl::Mainloop;
  using Params     = typename Collective::Params;
  using Pipeline   = typename Collective::MainloopPipelineData;
  using State      = typename Pipeline::PipelineState;
  static Params params(int pool, int groups, void const* a, void const* sa, void const* b, void const* sb) {
    typename Collective::Arguments args{static_cast<typename Collective::ElementA const*>(a),
                                        typename Collective::StrideA{K, _1{}, 0},
                                        static_cast<typename Collective::ElementB const*>(b),
                                        typename Collective::StrideB{K, _1{}, int64_t(N) * K},
                                        static_cast<typename Collective::ElementSFA const*>(sa),
                                        static_cast<typename Collective::ElementSFB const*>(sb)};
    return Collective::to_underlying_arguments(make_shape(pool, N, K, groups), args, nullptr);
  }
  template <class Inputs, class Coord>
  MUTLASS_DEVICE static void load(
      Inputs const& inputs, Coord const& coord, char* smem, Params const& params, Pipeline& pipeline, State& write) {
    auto& storage = *reinterpret_cast<typename Collective::SharedStorage*>(smem);
    auto  sa      = make_tensor(make_smem_ptr(storage.smem_A.begin()), typename Collective::SmemLayoutA{});
    auto  sb      = make_tensor(make_smem_ptr(storage.smem_B.begin()), typename Collective::SmemLayoutB{});
    auto  ca = params.tme_load_a.get_slice(0), cb = params.tme_load_b.get_slice(0);
    auto [m, n, e] = coord;
    auto ga        = ca.partition_S(get<0>(inputs)(_, _, m, _, 0));
    auto gb        = cb.partition_S(get<1>(inputs)(_, _, n, _, e));
    auto da        = ca.partition_D(sa);
    auto db        = cb.partition_D(sb);
    MUTLASS_PRAGMA_NO_UNROLL
    for (int k = 0; k < K / Impl::kBlockK; ++k) {
      pipeline.producer_acquire(write);
      auto bar   = pipeline.producer_get_barrier_id(write);
      int  stage = write.index();
      copy(params.tme_load_a.with(bar), ga(_, _, _, k), da(_, _, _, stage));
      copy(params.tme_load_b.with(bar), gb(_, _, _, k), db(_, _, _, stage));
      ++write;
    }
  }
};
}  // namespace mega_moe::prefill::mp32::fp4_fp4
