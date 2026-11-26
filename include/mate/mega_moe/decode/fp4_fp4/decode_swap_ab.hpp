#pragma once

#include <musa_fp4.h>

#include "mate/mega_moe/decode/fp4_fp4/decode_common.hpp"

namespace mega_moe::decode::mp32::fp4_fp4 {

// Logical scheduler M remains the token pool (32); the physical MMA exchanges
// A/B so that features occupy M and the small token dimension occupies N.
template <int ComputeKind, int FeatureTile = 256>
struct DecodeCompute : DecodeConfig {
  template <int N, int K>
  using Pool = DecodePool<N, K>;
};

template <int N, int K, int FeatureTile>
struct SwapDecodePool;

template <int FeatureTile>
struct DecodeCompute<true, FeatureTile> : DecodeConfig {
  static constexpr int kBlockN = FeatureTile, kPipelineStages = 3;
  using Tile = Shape<Int<kBlockN>, Int<kBlockM>, Int<kBlockK>>;
  using Mainloop =
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
  using Pipeline                      = typename Mainloop::MainloopPipelineData;
  using State                         = typename Pipeline::PipelineState;
  static constexpr int kNumMmaWarps   = Mainloop::NumTCEWarps;
  static constexpr int kNumMmaThreads = kNumMmaWarps * kWarpThreads;
  static constexpr int kNumThreads    = kFirstMmaThread + kNumMmaThreads;
  template <int N, int K>
  using Pool = SwapDecodePool<N, K, FeatureTile>;
};

template <int N, int K, int FeatureTile>
struct SwapDecodePool : DecodeCompute<true, FeatureTile> {
  using Compute = DecodeCompute<true, FeatureTile>;
  using typename Compute::Mainloop;
  using typename Compute::Pipeline;
  using typename Compute::State;
  static constexpr int kBlockN = Compute::kBlockN, kBlockK = Compute::kBlockK;
  static_assert(N % kBlockN == 0 && K % kBlockK == 0);
  struct Params {
    typename Mainloop::Params logical;
    typename Mainloop::Params packed;
  };
  static Params params(int pool, int groups, void const* a, void const* sa, void const* b, void const* sb) {
    // Public launch ABI is activation first; the native mainloop sees weight A.
    typename Mainloop::Arguments args{static_cast<typename Mainloop::ElementA const*>(b),
                                      typename Mainloop::StrideA{K, _1{}, int64_t(N) * K},
                                      static_cast<typename Mainloop::ElementB const*>(a),
                                      typename Mainloop::StrideB{K, _1{}, 0},
                                      static_cast<typename Mainloop::ElementSFA const*>(sb),
                                      static_cast<typename Mainloop::ElementSFB const*>(sa)};
    auto logical = Mainloop::to_underlying_arguments(make_shape(N, pool, K, groups), args, nullptr);
    args.dA      = typename Mainloop::StrideA{kBlockK, _1{}, int64_t(kBlockN) * kBlockK};
    auto packed  = Mainloop::to_underlying_arguments(
        make_shape(kBlockN, pool, kBlockK, groups * (N / kBlockN) * (K / kBlockK)), args, nullptr);
    return {logical, packed};
  }
  MUTLASS_DEVICE static auto init(int pool, int groups, Params const& params) {
    return make_tuple(Mainloop{}.load_init(make_shape(N, pool, K, groups), params.logical),
                      Mainloop{}.load_init(make_shape(kBlockN, pool, kBlockK, groups * (N / kBlockN) * (K / kBlockK)),
                                           params.packed));
  }
  template <class Inputs>
  MUTLASS_DEVICE static auto mma_inputs(Inputs const& inputs, int expert) {
    auto native = get<0>(inputs);
    auto scales = get<2>(native);
    scales.ptr_A += uint64_t(expert) * scales.m * scales.groups;
    return make_tuple(get<0>(native), get<1>(native), scales);
  }
  MUTLASS_DEVICE static auto coord(int pool_block, int feature_block) {
    return make_coord(feature_block, pool_block, _, 0);
  }
  template <class Inputs, class Coord>
  MUTLASS_DEVICE static void load(
      Inputs const& inputs, Coord const& coord, char* smem, Params const& params, Pipeline& pipeline, State& write) {
    auto& storage = *reinterpret_cast<typename Mainloop::SharedStorage*>(smem);
    auto  a       = make_tensor(make_smem_ptr(storage.smem_A.begin()), typename Mainloop::SmemLayoutA{});
    auto  b       = make_tensor(make_smem_ptr(storage.smem_B.begin()), typename Mainloop::SmemLayoutB{});
    auto  ca      = params.packed.tme_load_a.get_slice(0);
    auto  cb      = params.logical.tme_load_b.get_slice(0);
    auto [token_block, feature_block, expert] = coord;
    auto gb                                   = cb.partition_S(get<1>(get<0>(inputs))(_, _, token_block, _, 0));
    auto da                                   = ca.partition_D(a);
    auto db                                   = cb.partition_D(b);
    MUTLASS_PRAGMA_NO_UNROLL
    for (int k = 0; k < K / kBlockK; ++k) {
      pipeline.producer_acquire(write);
      auto bar   = pipeline.producer_get_barrier_id(write);
      int  batch = (expert * (N / kBlockN) + feature_block) * (K / kBlockK) + k;
      auto ga    = ca.partition_S(get<0>(get<1>(inputs))(_, _, 0, _, batch));
      copy(params.packed.tme_load_a.with(bar), ga(_, _, _, 0), da(_, _, _, write.index()));
      copy(params.logical.tme_load_b.with(bar), gb(_, _, _, k), db(_, _, _, write.index()));
      ++write;
    }
  }
};

// Preserve the M32/token-A arithmetic and register epilogues while adopting
// the K-tile-contiguous weight packing of the swap-AB path.
template <int N, int K, class Config = DecodeConfig>
struct PackedDecodePool : Config {
  using typename Config::Mainloop;
  using typename Config::Pipeline;
  using typename Config::State;
  using typename Config::Tile;
  static constexpr int kBlockN = Config::kBlockN, kBlockK = Config::kBlockK;
  struct Params {
    typename Mainloop::Params        logical;
    typename Mainloop::Params::TME_B packed_b;
  };
  static Params params(int pool, int groups, void const* a, void const* sa, void const* b, void const* sb) {
    typename Mainloop::Arguments args{static_cast<typename Mainloop::ElementA const*>(a),
                                      typename Mainloop::StrideA{K, _1{}, 0},
                                      static_cast<typename Mainloop::ElementB const*>(b),
                                      typename Mainloop::StrideB{K, _1{}, int64_t(N) * K},
                                      static_cast<typename Mainloop::ElementSFA const*>(sa),
                                      static_cast<typename Mainloop::ElementSFB const*>(sb)};
    auto logical = Mainloop::to_underlying_arguments(make_shape(pool, N, K, groups), args, nullptr);
    args.dB      = typename Mainloop::StrideB{kBlockK, _1{}, int64_t(kBlockN) * kBlockK};
    auto packed  = Mainloop::to_underlying_arguments(
        make_shape(pool, kBlockN, kBlockK, groups * (N / kBlockN) * (K / kBlockK)), args, nullptr);
    return {logical, packed.tme_load_b};
  }
  MUTLASS_DEVICE static auto init(int pool, int groups, Params const& params) {
    auto logical = Mainloop{}.load_init(make_shape(pool, N, K, groups), params.logical);
    auto b       = params.packed_b.get_tme_tensor(make_shape(kBlockN, kBlockK, groups * (N / kBlockN) * (K / kBlockK)));
    auto tiled   = local_tile(b, Tile{}, make_coord(_, _, _), Step<Underscore, _1, _1>{});
    return make_tuple(logical, tiled);
  }
  template <class Inputs>
  MUTLASS_DEVICE static auto mma_inputs(Inputs const& inputs, int expert) {
    return DecodePool<N, K>::mma_inputs(get<0>(inputs), expert);
  }
  MUTLASS_DEVICE static auto coord(int pool_block, int feature_block) {
    return DecodePool<N, K>::coord(pool_block, feature_block);
  }
  template <class Inputs, class Coord>
  MUTLASS_DEVICE static void load(
      Inputs const& inputs, Coord const& coord, char* smem, Params const& params, Pipeline& pipeline, State& write) {
    auto& storage  = *reinterpret_cast<typename Mainloop::SharedStorage*>(smem);
    auto  a        = make_tensor(make_smem_ptr(storage.smem_A.begin()), typename Mainloop::SmemLayoutA{});
    auto  b        = make_tensor(make_smem_ptr(storage.smem_B.begin()), typename Mainloop::SmemLayoutB{});
    auto  ca       = params.logical.tme_load_a.get_slice(0);
    auto  cb       = params.packed_b.get_slice(0);
    auto [m, n, e] = coord;
    auto ga        = ca.partition_S(get<0>(get<0>(inputs))(_, _, m, _, 0));
    auto da        = ca.partition_D(a);
    auto db        = cb.partition_D(b);
    MUTLASS_PRAGMA_NO_UNROLL
    for (int k = 0; k < K / kBlockK; ++k) {
      pipeline.producer_acquire(write);
      auto bar   = pipeline.producer_get_barrier_id(write);
      int  batch = (e * (N / kBlockN) + n) * (K / kBlockK) + k;
      auto gb    = cb.partition_S(get<1>(inputs)(_, _, 0, _, batch));
      copy(params.logical.tme_load_a.with(bar), ga(_, _, _, k), da(_, _, _, write.index()));
      copy(params.packed_b.with(bar), gb(_, _, _, 0), db(_, _, _, write.index()));
      ++write;
    }
  }
};

template <int FeatureTile>
struct DecodeCompute<2, FeatureTile> : DecodeConfig {
  template <int N, int K>
  using Pool = PackedDecodePool<N, K>;
};

// Each M32 token block owns an entire 128-byte SF cache line. Publication
// of another block cannot invalidate a partially populated line that a
// consumer already prefetched. The once-per-invocation cache reset remains.
template <int N, int K, class Config = DecodeConfig>
struct LinePaddedDecodePool : DecodePool<N, K, Config> {
  using Parent = DecodePool<N, K, Config>;
  using Params = typename Parent::Params;
  static Params params(int pool, int groups, void const* a, void const* sa, void const* b, void const* sb) {
    auto result     = Parent::params(pool, groups, a, sa, b, sb);
    result.scales.m = pool * 4;
    return result;
  }
  MUTLASS_DEVICE static auto coord(int pool_block, int feature_block) {
    // MMA's block coordinate indexes scales only; TME uses the descriptor's
    // original compact pool block in Parent::load().
    return make_coord(pool_block * 4, feature_block, _, 0);
  }
};

template <int FeatureTile>
struct DecodeCompute<3, FeatureTile> : DecodeConfig {
  template <int N, int K>
  using Pool = LinePaddedDecodePool<N, K>;
};

template <int N, int K, class Config = DecodeConfig>
struct LinePaddedPackedDecodePool : PackedDecodePool<N, K, Config> {
  using Parent = PackedDecodePool<N, K, Config>;
  using Params = typename Parent::Params;
  static Params params(int pool, int groups, void const* a, void const* sa, void const* b, void const* sb) {
    auto result             = Parent::params(pool, groups, a, sa, b, sb);
    result.logical.scales.m = pool * 4;
    return result;
  }
  MUTLASS_DEVICE static auto coord(int pool_block, int feature_block) {
    return make_coord(pool_block * 4, feature_block, _, 0);
  }
};
template <int FeatureTile>
struct DecodeCompute<6, FeatureTile> : DecodeConfig {
  template <int N, int K>
  using Pool = LinePaddedPackedDecodePool<N, K>;
};

struct DoubleBufferedDecodeConfig {
  static constexpr int kBlockM = 32, kBlockN = 256, kBlockK = 256;
  static constexpr int kScaleBlock = 32, kValuesPerByte = 2, kMaxTokens = 128;
  static constexpr int kWarpThreads        = mutlass::NumThreadsPerWarp;
  static constexpr int kSquadThreads       = mutlass::NumThreadsPerWarpSquad;
  static constexpr int kNumDispatchThreads = kSquadThreads;
  static constexpr int kFirstMmaThread     = 2 * kSquadThreads;
  static constexpr int kPipelineStages     = 2;
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

template <int FeatureTile>
struct DecodeCompute<7, FeatureTile> : DoubleBufferedDecodeConfig {
  template <int N, int K>
  using Pool = DecodePool<N, K, DoubleBufferedDecodeConfig>;
};

// Register SwiGLU followed by a compact BF16 SMEM transpose. Preserve the
// existing BF16/BST4/UE8M0 arithmetic and eight quantized values per lane.
// No global intermediate or extra launch is introduced.
template <class Compute, class Accum, class Mma, class Desc, class Sync>
MUTLASS_DEVICE void swap_activation(Accum const&         accum,
                                    Mma const&           mma,
                                    int                  tid,
                                    Desc const&          desc,
                                    mutlass::bfloat16_t* tile,
                                    uint8_t*             l2,
                                    uint8_t*             sf2,
                                    int*                 status,
                                    int                  pool,
                                    int                  intermediate,
                                    float                clamp,
                                    Sync const&          sync) {
  constexpr int F = Compute::kBlockN, Threads = Compute::kNumMmaThreads;
  auto coords = mma.get_thread_slice(tid).partition_C(make_identity_tensor(take<0, 2>(typename Compute::Tile{})));
  static_assert(decltype(size(accum))::value % 16 == 0);
  // Native C holds eight token positions for one gate feature followed by
  // the matching eight up values. Activate in registers and stage only the
  // BF16 product, halving the SMEM epilogue traffic.
  MUTE_UNROLL
  for (int index = 0; index < size(accum); index += 16) {
    MUTE_UNROLL
    for (int col = 0; col < 8; col += 4) {
      fp4::ActivationFloat4 gate{
          accum(index + col), accum(index + col + 1), accum(index + col + 2), accum(index + col + 3)};
      fp4::ActivationFloat4 up{
          accum(index + col + 8), accum(index + col + 9), accum(index + col + 10), accum(index + col + 11)};
      gate = __builtin_convertvector(__builtin_convertvector(gate, fp4::ActivationBf16x4), fp4::ActivationFloat4);
      up   = __builtin_convertvector(__builtin_convertvector(up, fp4::ActivationBf16x4), fp4::ActivationFloat4);
      MUTE_UNROLL
      for (int j = 0; j < 4; ++j) {
        gate[j] = fminf(gate[j], clamp);
        up[j]   = fminf(clamp, fmaxf(-clamp, up[j]));
      }
      gate      = __builtin_convertvector(__builtin_convertvector(gate, fp4::ActivationBf16x4), fp4::ActivationFloat4);
      auto silu = __builtin_convertvector(__builtin_convertvector(fp4::silu_fp32_bst4(gate), fp4::ActivationBf16x4),
                                          fp4::ActivationFloat4);
      auto raw  = __builtin_bit_cast(fp4::ActivationBits4, __builtin_convertvector(up * silu, fp4::ActivationBf16x4));
      MUTE_UNROLL
      for (int j = 0; j < 4; ++j) {
        int f = int(get<0>(coords(index + col + j))), row = int(get<1>(coords(index + col + j)));
        tile[row * (F / 2) + (f / 64) * 32 + f % 32] = mutlass::bfloat16_t::bitcast(raw[j]);
      }
    }
  }
  sync();
  constexpr int Groups = F / 64;
  int           lane   = tid % 4;
  for (int q = tid / 4; q < 32 * Groups; q += Threads / 4) {
    int row = q / Groups, group = q % Groups;
    if (row >= int(desc.valid_m)) continue;
    mutlass::bfloat16_t values[8];
    uint32_t            amax = 0;
    MUTE_UNROLL
    for (int col = 0; col < 8; ++col) {
      values[col]        = tile[row * (F / 2) + group * 32 + lane * 8 + col];
      uint32_t magnitude = uint32_t(values[col].raw()) & 0x7fffu;
      if (magnitude >= 0x7f80u) {
        atomicExch(status, 3);
        values[col] = mutlass::bfloat16_t(0.f);
        magnitude   = 0;
      }
      amax = max(amax, magnitude);
    }
    unsigned mask     = __activemask();
    amax              = max(amax, __shfl_xor_sync(mask, amax, 1));
    amax              = max(amax, __shfl_xor_sync(mask, amax, 2));
    int      exponent = fp4::Fp4Quantization::scale_exponent(__uint_as_float(amax << 16));
    float    inv      = fp4::Fp4Quantization::inverse(exponent);
    int      outgroup = desc.n_block_idx * Groups + group;
    uint64_t outrow   = uint64_t(desc.pool_block_idx) * 32 + row;
    if (lane == 0) sf2[uint64_t(outgroup) * pool + outrow] = uint8_t(exponent + fp4::Fp4Quantization::kScaleBias);
    uint32_t packed = 0;
    MUTE_UNROLL
    for (int pair = 0; pair < 4; ++pair)
      packed |=
          uint32_t(__musa_cvt_float2_to_fp4x2(
              float2{float(values[2 * pair]) * inv, float(values[2 * pair + 1]) * inv}, __MT_E2M1, musaRoundNearest))
          << (pair * 8);
    *reinterpret_cast<uint32_t*>(l2 + outrow * (intermediate / 2) + outgroup * 16 + lane * 4) = packed;
  }
}
}  // namespace mega_moe::decode::mp32::fp4_fp4
