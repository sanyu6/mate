#pragma once
#include "mate/mega_moe/decode/fp4_fp4/dispatch_pull.hpp"

namespace mega_moe::decode::mp32::fp4_fp4 {
using namespace mute;
template <int K, int N, int FeatureTile = 128, int TokenTile = 32, int Stages = 2>
struct SharedFp8Gemm {
  using Input = mutlass::mx_float8_t<mutlass::float_e4m3_t>;
  using Tile  = Shape<Int<FeatureTile>, Int<TokenTile>, Int<128>>;
  using Mainloop =
      typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp32,
                                                            mutlass::arch::OpClassBlockScaledTensorOp,
                                                            Input,
                                                            mutlass::layout::RowMajor,
                                                            16,
                                                            Input,
                                                            mutlass::layout::ColumnMajor,
                                                            16,
                                                            float,
                                                            Tile,
                                                            Shape<_1, _1, _1>,
                                                            mutlass::gemm::collective::StageCount<Stages>,
                                                            mutlass::gemm::KernelTmeWarpSpecializedMp32>::CollectiveOp;
  using Params                       = typename Mainloop::Params;
  using Pipeline                     = typename Mainloop::MainloopPipelineData;
  using State                        = typename Pipeline::PipelineState;
  using Barrier                      = mutlass::arch::AsyncBarrier;
  static constexpr int NumMmaThreads = Mainloop::NumTCEWarps * 32;
  static constexpr int NumMmaWarps   = Mainloop::NumTCEWarps;
  static Params        params(int tokens, void const* w, void const* sw, void const* x, void const* sx) {
    typename Mainloop::Arguments args{static_cast<typename Mainloop::ElementA const*>(w),
                                      typename Mainloop::StrideA{K, _1{}, 0},
                                      static_cast<typename Mainloop::ElementB const*>(x),
                                      typename Mainloop::StrideB{K, _1{}, 0},
                                      static_cast<typename Mainloop::ElementSFA const*>(sw),
                                      static_cast<typename Mainloop::ElementSFB const*>(sx)};
    return Mainloop::to_underlying_arguments(make_shape(N, tokens, K, 1), args, nullptr);
  }
  template <class Inputs>
  __device__ static void load(
      Inputs const& inputs, int feature, int token, char* smem, Params const& p, Pipeline& pipe, State& write) {
    auto& storage = *reinterpret_cast<typename Mainloop::SharedStorage*>(smem);
    auto  sa      = make_tensor(make_smem_ptr(storage.smem_A.begin()), typename Mainloop::SmemLayoutA{});
    auto  sb      = make_tensor(make_smem_ptr(storage.smem_B.begin()), typename Mainloop::SmemLayoutB{});
    auto  ca      = p.tme_load_a.get_slice(0);
    auto  cb      = p.tme_load_b.get_slice(0);
    auto  ga      = ca.partition_S(get<0>(inputs)(_, _, feature, _, 0));
    auto  gb      = cb.partition_S(get<1>(inputs)(_, _, token, _, 0));
    auto  da      = ca.partition_D(sa);
    auto  db      = cb.partition_D(sb);
    MUTLASS_PRAGMA_NO_UNROLL
    for (int k = 0; k < K / 128; ++k) {
      pipe.producer_acquire(write);
      auto bar   = pipe.producer_get_barrier_id(write);
      int  stage = write.index();
      copy(p.tme_load_a.with(bar), ga(_, _, _, k), da(_, _, _, stage));
      copy(p.tme_load_b.with(bar), gb(_, _, _, k), db(_, _, _, stage));
      ++write;
    }
  }
  template <class Inputs>
  __device__ static void compute(Inputs const&        inputs,
                                 int                  feature,
                                 int                  token,
                                 int                  tokens,
                                 char*                smem,
                                 Pipeline&            pipe,
                                 State&               read,
                                 int                  mma_tid,
                                 mutlass::bfloat16_t* output) {
    typename Mainloop::TiledMma mma;
    auto                        accum = partition_fragment_C(mma, take<0, 2>(Tile{}));
    fill(accum, 0.f);
    Mainloop{}.mma(inputs, make_coord(feature, token, _, 0), accum, K / 128, mma_tid, smem, pipe, read);
    auto coords = mma.get_thread_slice(mma_tid).partition_C(make_identity_tensor(take<0, 2>(Tile{})));
    MUTE_UNROLL
    for (int j = 0; j < size(accum); ++j) {
      auto coord = coords(j);
      int  f     = feature * FeatureTile + int(get<0>(coord));
      int  row   = token * TokenTile + int(get<1>(coord));
      if (row < tokens && f < N) output[uint64_t(row) * N + f] = mutlass::bfloat16_t(accum(j));
    }
  }
};

template <class L,
          int      N,
          uint32_t CTAs,
          bool     Padded,
          bool     Fused,
          int      FeatureTile = 128,
          int      TokenTile   = 32,
          int      Stages      = 2,
          int      GemmCTAs    = CTAs>
struct DispatchSharedGemm {
  using Gemm                           = SharedFp8Gemm<L::kHidden, N, FeatureTile, TokenTile, Stages>;
  using Dispatch                       = PullDispatchOnly<L, CTAs, Padded>;
  using Pipeline                       = typename Gemm::Pipeline;
  using Barrier                        = typename Gemm::Barrier;
  using Params                         = typename Gemm::Params;
  static constexpr int kProducerThread = Fused ? 128 : 0;
  static constexpr int kMmaThread      = Fused ? 256 : 128;
  static constexpr int kNumThreads     = kMmaThread + Gemm::NumMmaThreads;
  static constexpr int kNumCTAs        = Fused ? (CTAs > GemmCTAs ? CTAs : GemmCTAs) : GemmCTAs;
  static constexpr int R               = L::kRanks;
  struct Empty {};
  struct SharedStorage {
    typename Gemm::Mainloop::SharedStorage                             gemm;
    std::conditional_t<Fused, typename Dispatch::SharedStorage, Empty> dispatch;
  };
  static constexpr int SmemSize = sizeof(SharedStorage);
  struct BarrierStorage {
    uint8_t pipeline[Pipeline::NumBarriers];
    uint8_t dispatch[2];
  };
  static_assert(kNumThreads <= 1024 && SmemSize <= 256 * 1024);
  static_assert(sizeof(BarrierStorage) + Barrier::ReservedAsyncBarrierCount <=
                Barrier::HardwareMaxNumAsyncTransactionBarriers);
  __device__ static void run(layout::SymBuffer<R> const& sym,
                             Params const&               params,
                             uint8_t*                    payload,
                             uint8_t*                    scales,
                             uint4*                      metadata,
                             int*                        counts,
                             int*                        offsets,
                             int*                        status,
                             int                         tokens,
                             mutlass::bfloat16_t*        out,
                             char*                       smem) {
    auto& shared   = *reinterpret_cast<SharedStorage*>(smem);
    auto* barriers = reinterpret_cast<BarrierStorage*>(0);
    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage) + Barrier::ReservedAsyncBarrierCount);
    typename Pipeline::Params pp;
    pp.transaction_bytes = Gemm::Mainloop::TmeTransactionBytes;
    pp.num_consumers     = Gemm::NumMmaWarps;
    pp.num_producers     = 1;
    Pipeline pipe(pp, reinterpret_cast<uint64_t>(&barriers->pipeline));
    auto     barrier_id = [](auto* ptr) {
      return uint32_t(reinterpret_cast<uint64_t>(ptr)) + Barrier::ReservedAsyncBarrierCount;
    };
    uint32_t dispatch_bar = barrier_id(&barriers->dispatch);
    if constexpr (Fused) {
      if (threadIdx.x == 0)
        for (int i = 0; i < 2; ++i) Barrier::init(dispatch_bar + i, 4, 0);
    }
    if (threadIdx.x == 0) __threadfence_system();
    __syncthreads();
    int tid = threadIdx.x;
    if constexpr (Fused) {
      if (tid < 128 && blockIdx.x < CTAs) {
        uint32_t phase = 0;
        auto     sync  = [&]() {
          auto b = dispatch_bar + (phase++ & 1u);
          auto p = Barrier::template arrive<true>(b);
          Barrier::wait(b, p);
        };
        Dispatch::template run<decltype(sync), false>(
            sym, payload, scales, metadata, counts, offsets, status, tokens, shared.dispatch, sync);
      }
    }
    if (blockIdx.x < GemmCTAs && tid >= kProducerThread && tid < kProducerThread + 32) {
      auto inputs       = typename Gemm::Mainloop{}.load_init(make_shape(N, tokens, L::kHidden, 1), params);
      auto write        = mutlass::make_producer_start_state<Pipeline>();
      int  token_blocks = (tokens + TokenTile - 1) / TokenTile;
      for (int task = blockIdx.x; task < (N / FeatureTile) * token_blocks; task += GemmCTAs) {
        Gemm::load(inputs,
                   task / token_blocks,
                   task % token_blocks,
                   reinterpret_cast<char*>(&shared.gemm),
                   params,
                   pipe,
                   write);
      }
    } else if (blockIdx.x < GemmCTAs && tid >= kMmaThread) {
      auto                 inputs = typename Gemm::Mainloop{}.load_init(make_shape(N, tokens, L::kHidden, 1), params);
      typename Gemm::State read;
      int                  token_blocks = (tokens + TokenTile - 1) / TokenTile;
      for (int task = blockIdx.x; task < (N / FeatureTile) * token_blocks; task += GemmCTAs) {
        Gemm::compute(inputs,
                      task / token_blocks,
                      task % token_blocks,
                      tokens,
                      reinterpret_cast<char*>(&shared.gemm),
                      pipe,
                      read,
                      tid - kMmaThread,
                      out);
      }
    }
  }
};
template <class Stage>
__global__ __launch_bounds__(Stage::kNumThreads) void dispatch_shared_kernel(
    MUTLASS_GRID_CONSTANT layout::SymBuffer<Stage::R> const sym,
    MUTLASS_GRID_CONSTANT typename Stage::Params const      params,
    uint8_t*                                                payload,
    uint8_t*                                                scales,
    uint4*                                                  metadata,
    int*                                                    counts,
    int*                                                    offsets,
    int*                                                    status,
    int                                                     tokens,
    mutlass::bfloat16_t*                                    out) {
  extern __shared__ __align__(256) char smem[];
  Stage::run(sym, params, payload, scales, metadata, counts, offsets, status, tokens, out, smem);
}
}  // namespace mega_moe::decode::mp32::fp4_fp4
