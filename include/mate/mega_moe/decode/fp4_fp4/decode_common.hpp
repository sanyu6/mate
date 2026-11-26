#pragma once

#include "mate/mega_moe/prefill/fp4_fp4/prefill_common.hpp"
#include "mate/mega_moe/prefill/fp4_fp4/stage2_online_common.hpp"

namespace mega_moe::decode::mp32::fp4_fp4 {
using namespace mute;
namespace fp4 = ::mega_moe::prefill::mp32::fp4_fp4;

struct DecodeConfig {
  static constexpr int kBlockM = 32, kBlockN = 256, kBlockK = 256;
  static constexpr int kScaleBlock = 32, kValuesPerByte = 2, kMaxTokens = 128;
  static constexpr int kWarpThreads        = mutlass::NumThreadsPerWarp;
  static constexpr int kSquadThreads       = mutlass::NumThreadsPerWarpSquad;
  static constexpr int kNumDispatchThreads = kSquadThreads;
  static constexpr int kFirstMmaThread     = 2 * kSquadThreads;
  static constexpr int kPipelineStages     = 3;
  static constexpr int kAlignment          = 256;
  using Input                              = mutlass::mx_float4_t<mutlass::float_e2m1_t>;
  using Tile                               = Shape<Int<kBlockM>, Int<kBlockN>, Int<kBlockK>>;
  // Fixed staging avoids spending all 256 KiB on a small decode tile.
  using Base =
      typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp32,
                                                            mutlass::arch::OpClassBlockScaledTensorOp,
                                                            Input,
                                                            mutlass::layout::RowMajor,
                                                            32,
                                                            Input,
                                                            mutlass::layout::ColumnMajor,
                                                            32,
                                                            float,
                                                            Tile,
                                                            Shape<_1, _1, _1>,
                                                            mutlass::gemm::collective::StageCount<kPipelineStages>,
                                                            mutlass::gemm::KernelTmeWarpSpecializedMp32>::CollectiveOp;
  using Mainloop                      = fp4::Stage2OnlineMainloop<Base>;
  using Pipeline                      = typename Mainloop::MainloopPipelineData;
  using State                         = typename Pipeline::PipelineState;
  using Barrier                       = mutlass::arch::AsyncBarrier;
  static constexpr int kNumMmaWarps   = Mainloop::NumTCEWarps;
  static constexpr int kNumMmaThreads = kNumMmaWarps * kWarpThreads;
  static constexpr int kNumThreads    = kFirstMmaThread + kNumMmaThreads;
  static_assert(size<0>(typename Mainloop::TiledMma::AtomShape_MNK{}) == kBlockM);
  static_assert(size<2>(typename Mainloop::TiledMma::AtomShape_MNK{}) == kBlockK);
  static_assert(kNumThreads <= 1024);
};

template <int R, int T, int H, int I, int E, int TopK>
struct DecodeLayout : fp4::SymmetricLayout<R, T, H, I, E, TopK> {
  using Prefill = fp4::SymmetricLayout<R, T, H, I, E, TopK>;
  // Reuse the prefill input/combine prefix and its local activation allocation.
  // Only control words need a separate M32 view, appended in the same IPC block.
  static constexpr int kBlockM       = DecodeConfig::kBlockM;
  static constexpr int kDecodeTokens = DecodeConfig::kMaxTokens;
  static constexpr int kPoolTokens   = layout::get_num_max_pool_tokens(R, T, TopK, E / R, Prefill::kBlockM);
  static_assert(T >= kDecodeTokens);
  static_assert(kPoolTokens >= layout::get_num_max_pool_tokens(R, kDecodeTokens, TopK, E / R, kBlockM));
  __host__ __device__ static layout::Workspace workspace(void* base) {
    auto prefill_bytes =
        Prefill::kBytes + ::mega_moe::prefill::PrefillSplitWorkspace(nullptr, R, E, T, TopK).get_num_bytes();
    return {math::advance_ptr(base, prefill_bytes), R, E, kDecodeTokens, TopK, kBlockM};
  }
  static uint64_t allocation_bytes() {
    return Prefill::allocation_bytes() + layout::Workspace(nullptr, R, E, kDecodeTokens, TopK, kBlockM).get_num_bytes();
  }
};

template <int N, int K, class Compute = DecodeConfig>
struct DecodePool : Compute {
  using typename Compute::Mainloop;
  using typename Compute::Pipeline;
  using typename Compute::State;
  static constexpr int kBlockK = Compute::kBlockK;
  using Params                 = typename Mainloop::Params;
  static Params params(int pool, int groups, void const* a, void const* sa, void const* b, void const* sb) {
    typename Mainloop::Arguments args{static_cast<typename Mainloop::ElementA const*>(a),
                                      typename Mainloop::StrideA{K, _1{}, 0},
                                      static_cast<typename Mainloop::ElementB const*>(b),
                                      typename Mainloop::StrideB{K, _1{}, int64_t(N) * K},
                                      static_cast<typename Mainloop::ElementSFA const*>(sa),
                                      static_cast<typename Mainloop::ElementSFB const*>(sb)};
    return Mainloop::to_underlying_arguments(make_shape(pool, N, K, groups), args, nullptr);
  }
  MUTLASS_DEVICE static auto init(int pool, int groups, Params const& params) {
    return Mainloop{}.load_init(make_shape(pool, N, K, groups), params);
  }
  template <class Inputs>
  MUTLASS_DEVICE static auto mma_inputs(Inputs const& inputs, int expert) {
    auto scales = get<2>(inputs);
    scales.ptr_B += uint64_t(expert) * scales.n * scales.groups;
    return make_tuple(get<0>(inputs), get<1>(inputs), scales);
  }
  MUTLASS_DEVICE static auto coord(int pool_block, int feature_block) {
    return make_coord(pool_block, feature_block, _, 0);
  }
  template <class Inputs, class Coord>
  MUTLASS_DEVICE static void load(
      Inputs const& inputs, Coord const& coord, char* smem, Params const& params, Pipeline& pipeline, State& write) {
    auto& storage  = *reinterpret_cast<typename Mainloop::SharedStorage*>(smem);
    auto  a        = make_tensor(make_smem_ptr(storage.smem_A.begin()), typename Mainloop::SmemLayoutA{});
    auto  b        = make_tensor(make_smem_ptr(storage.smem_B.begin()), typename Mainloop::SmemLayoutB{});
    auto  ca       = params.tme_load_a.get_slice(0);
    auto  cb       = params.tme_load_b.get_slice(0);
    auto [m, n, e] = coord;
    auto ga        = ca.partition_S(get<0>(inputs)(_, _, m, _, 0));
    auto gb        = cb.partition_S(get<1>(inputs)(_, _, n, _, e));
    auto da        = ca.partition_D(a);
    auto db        = cb.partition_D(b);
    MUTLASS_PRAGMA_NO_UNROLL
    for (int k = 0; k < K / kBlockK; ++k) {
      pipeline.producer_acquire(write);
      auto bar = pipeline.producer_get_barrier_id(write);
      copy(params.tme_load_a.with(bar), ga(_, _, _, k), da(_, _, _, write.index()));
      copy(params.tme_load_b.with(bar), gb(_, _, _, k), db(_, _, _, write.index()));
      ++write;
    }
  }
};
}  // namespace mega_moe::decode::mp32::fp4_fp4
