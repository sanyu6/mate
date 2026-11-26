#pragma once
#ifndef DISPATCH_TACTIC
#define DISPATCH_TACTIC 0
#endif
#ifndef SHARED_EPILOGUE_TACTIC
#define SHARED_EPILOGUE_TACTIC 0
#endif
#include <musa_fp8.h>

#include "mate/mega_moe/decode/fp4_fp4/dispatch_pull.hpp"

namespace mega_moe::decode::mp32::fp4_fp4 {
using namespace mute;
template <int K, int N, int FeatureTile = 128, int TokenTile = 32, int Stages = 2>
struct SharedFp8SwiGlu {
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
                                 mutlass::bfloat16_t* scratch,
                                 uint8_t*             output_q,
                                 uint8_t*             output_sf,
                                 float                limit,
                                 uint32_t             epibar,
                                 uint32_t&            epiphase) {
    typename Mainloop::TiledMma mma;
    auto                        accum = partition_fragment_C(mma, take<0, 2>(Tile{}));
    fill(accum, 0.f);
    Mainloop{}.mma(inputs, make_coord(feature, token, _, 0), accum, K / 128, mma_tid, smem, pipe, read);
    auto coords = mma.get_thread_slice(mma_tid).partition_C(make_identity_tensor(take<0, 2>(Tile{})));
    if constexpr (SHARED_EPILOGUE_TACTIC == 2) {
      static_assert(NumMmaThreads == 128 && (FeatureTile == 64 || FeatureTile == 128));
      constexpr int Half = FeatureTile / 2, Groups = Half / 32, I = N / 2;
      float         values[size(accum) / 2];
      auto*         maxima = reinterpret_cast<float*>(scratch);
      MUTE_UNROLL
      for (int j = 0; j < size(accum) / 2; ++j) {
        auto  c = coords(j);
        int   f = int(get<0>(c)), row = int(get<1>(c));
        float gate = float(mutlass::bfloat16_t(accum(j))), up = float(mutlass::bfloat16_t(accum(j + size(accum) / 2)));
        if (limit >= 0) {
          gate = fminf(gate, limit);
          up   = fminf(fmaxf(up, -limit), limit);
        }
        values[j] = float(mutlass::bfloat16_t((__fdividef(gate, 1.f + __expf(-gate))) * up));
        float am  = fabsf(values[j]);
        am        = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
        am        = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8));
        am        = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 16));
        if ((mma_tid & 31) < 4) maxima[(row * Groups + f / 32) * 4 + mma_tid / 32] = am;
      }
      __threadfence_block();
      __musa_async_arrive(epibar);
      __musa_async_wait(epibar, epiphase++ & 1u);
      asm volatile("" ::: "memory");
      MUTE_UNROLL
      for (int j = 0; j < size(accum) / 2; ++j) {
        auto  c = coords(j);
        int   f = int(get<0>(c)), row = int(get<1>(c)), dstrow = token * TokenTile + row;
        float am = 0;
#pragma unroll
        for (int w = 0; w < 4; ++w) am = fmaxf(am, maxima[(row * Groups + f / 32) * 4 + w]);
        float    raw  = fmaxf(fmaxf(am, 1.e-30f) * (1.f / 448.f), 1.e-30f);
        uint32_t bits = (__float_as_uint(raw) + 0x7fffffu) & 0xff800000u, code = min((bits >> 23) & 255u, 254u);
        float    scale = exp2f(float(code) - 127.f);
        float    value = fminf(fmaxf(__fdividef(values[j], scale), -448.f), 448.f);
        int      lane  = mma_tid & 31;
        float    v1    = __shfl_sync(0xffffffffu, value, (lane & ~12) + 4);
        float    v2    = __shfl_sync(0xffffffffu, value, (lane & ~12) + 8);
        float    v3    = __shfl_sync(0xffffffffu, value, (lane & ~12) + 12);
        if (dstrow < tokens && f % 4 == 0) {
          float4 q = {value, v1, v2, v3};
          *reinterpret_cast<uint32_t*>(output_q + dstrow * I + feature * Half + f) =
              __musa_cvt_float4_to_fp8x4(q, __MT_SATFINITE, __MT_E4M3);
          if (f % 32 == 0) output_sf[((feature * Half + f) / 32) * tokens + dstrow] = uint8_t(code);
        }
      }
      __musa_async_arrive(epibar);
      __musa_async_wait(epibar, epiphase++ & 1u);
      return;
    }
    if constexpr (SHARED_EPILOGUE_TACTIC == 1) {
      static_assert(FeatureTile == 64 || FeatureTile == 128);
      MUTE_UNROLL
      for (int j = 0; j < size(accum) / 2; ++j) {
        auto  coord = coords(j);
        float gate  = float(mutlass::bfloat16_t(accum(j)));
        float up    = float(mutlass::bfloat16_t(accum(j + size(accum) / 2)));
        if (limit >= 0) {
          gate = fminf(gate, limit);
          up   = fminf(fmaxf(up, -limit), limit);
        }
        float silu                                                           = __fdividef(gate, 1.f + __expf(-gate));
        scratch[int(get<1>(coord)) * (FeatureTile / 2) + int(get<0>(coord))] = mutlass::bfloat16_t(silu * up);
      }
    } else {
      // Paired tile layout: [gate half-tile, up half-tile]. No HBM gate/up tensor.
      MUTE_UNROLL
      for (int j = 0; j < size(accum); ++j) {
        auto coord                                                     = coords(j);
        scratch[int(get<1>(coord)) * FeatureTile + int(get<0>(coord))] = mutlass::bfloat16_t(accum(j));
      }
    }
    __threadfence_block();
    __musa_async_arrive(epibar);
    __musa_async_wait(epibar, epiphase++ & 1u);
    asm volatile("" ::: "memory");
    constexpr int I = N / 2, Half = FeatureTile / 2, Groups = Half / 32;
    for (int task = mma_tid / 8; task < TokenTile * Groups; task += NumMmaThreads / 8) {
      int row = task / Groups, group = task % Groups, lane = mma_tid % 8;
      int dstrow = token * TokenTile + row;
      if (dstrow < tokens) {
        float v[4], am = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          int c = group * 32 + lane * 4 + j;
          if constexpr (SHARED_EPILOGUE_TACTIC == 1) {
            v[j] = float(scratch[row * Half + c]);
          } else {
            float gate = float(scratch[row * FeatureTile + c]), up = float(scratch[row * FeatureTile + Half + c]);
            if (limit >= 0) {
              gate = fminf(gate, limit);
              up   = fminf(fmaxf(up, -limit), limit);
            }
            float silu = __fdividef(gate, 1.f + __expf(-gate));
            v[j]       = float(mutlass::bfloat16_t(silu * up));
          }
          am = fmaxf(am, fabsf(v[j]));
        }
        am             = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
        am             = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
        am             = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
        float    raw   = fmaxf(fmaxf(am, 1.e-30f) * (1.f / 448.f), 1.e-30f);
        uint32_t bits  = (__float_as_uint(raw) + 0x7fffffu) & 0xff800000u;
        uint32_t code  = min((bits >> 23) & 255u, 254u);
        float    scale = exp2f(float(code) - 127.f);
        float4   q     = {fminf(fmaxf(__fdividef(v[0], scale), -448.f), 448.f),
                          fminf(fmaxf(__fdividef(v[1], scale), -448.f), 448.f),
                          fminf(fmaxf(__fdividef(v[2], scale), -448.f), 448.f),
                          fminf(fmaxf(__fdividef(v[3], scale), -448.f), 448.f)};
        int      col   = feature * Half + group * 32 + lane * 4;
        *reinterpret_cast<uint32_t*>(output_q + dstrow * I + col) =
            __musa_cvt_float4_to_fp8x4(q, __MT_SATFINITE, __MT_E4M3);
        if (lane == 0) output_sf[(col / 32) * tokens + dstrow] = uint8_t(code);
      }
    }
    __musa_async_arrive(epibar);
    __musa_async_wait(epibar, epiphase++ & 1u);
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
struct DispatchSharedSwiGlu {
  using Gemm                           = SharedFp8SwiGlu<L::kHidden, N, FeatureTile, TokenTile, Stages>;
  using Dispatch                       = PullDispatchOnly<L,
                                                          CTAs,
                                                          Padded,
                                                          32,
                                                          (DISPATCH_TACTIC >= 2 ? 24 : 128),
                                                          (DISPATCH_TACTIC >= 2),
                                                          (DISPATCH_TACTIC == 4)>;
  using Pipeline                       = typename Gemm::Pipeline;
  using Barrier                        = typename Gemm::Barrier;
  using Params                         = typename Gemm::Params;
  static constexpr int kProducerThread = Fused ? 128 : 0;
  static constexpr int kMmaThread      = Fused ? 256 : 128;
  static constexpr int kNumThreads     = kMmaThread + Gemm::NumMmaThreads;
  static constexpr int kNumCTAs        = Fused ? (CTAs > GemmCTAs ? CTAs : GemmCTAs) : GemmCTAs;
  static constexpr int R               = L::kRanks;
  static constexpr int kDispatchCTAs = CTAs, kGemmCTAs = GemmCTAs;
  struct Empty {};
  struct SharedStorage {
    typename Gemm::Mainloop::SharedStorage                             gemm;
    mutlass::bfloat16_t                                                epilogue[FeatureTile * TokenTile];
    std::conditional_t<Fused, typename Dispatch::SharedStorage, Empty> dispatch;
  };
  static constexpr int SmemSize = sizeof(SharedStorage);
  struct BarrierStorage {
    uint8_t pipeline[Pipeline::NumBarriers];
    uint8_t dispatch[2];
    uint8_t epilogue;
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
                             char*                       smem,
                             uint8_t*                    output_q,
                             uint8_t*                    output_sf,
                             float                       limit,
                             int                         cta_index      = -1,
                             int4*                       tile_schedule  = nullptr,
                             int                         schedule_tiles = 0) {
    int   cta      = cta_index < 0 ? int(blockIdx.x) : cta_index;
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
    uint32_t epibar = barrier_id(&barriers->epilogue);
    if (threadIdx.x == 0) Barrier::init(epibar, Gemm::NumMmaWarps, 0);
    uint32_t dispatch_bar = barrier_id(&barriers->dispatch);
    if constexpr (Fused) {
      if (threadIdx.x == 0)
        for (int i = 0; i < 2; ++i) Barrier::init(dispatch_bar + i, 4, 0);
    }
    if (Fused && threadIdx.x == 0 && (DISPATCH_TACTIC == 0 || cta < CTAs)) __threadfence_system();
    __syncthreads();
    int tid = threadIdx.x;
    if constexpr (Fused) {
      if (tid < 128 && cta < CTAs) {
        uint32_t phase = 0;
        auto     sync  = [&]() {
          auto b = dispatch_bar + (phase++ & 1u);
          auto p = Barrier::template arrive<true>(b);
          Barrier::wait(b, p);
        };
        Dispatch::template run<decltype(sync), false>(sym,
                                                      payload,
                                                      scales,
                                                      metadata,
                                                      counts,
                                                      offsets,
                                                      status,
                                                      tokens,
                                                      shared.dispatch,
                                                      sync,
                                                      tile_schedule,
                                                      schedule_tiles);
      }
    }
    if (cta < GemmCTAs && tid >= kProducerThread && tid < kProducerThread + 32) {
      auto inputs       = typename Gemm::Mainloop{}.load_init(make_shape(N, tokens, L::kHidden, 1), params);
      auto write        = mutlass::make_producer_start_state<Pipeline>();
      int  token_blocks = (tokens + TokenTile - 1) / TokenTile;
      for (int task = cta; task < (N / FeatureTile) * token_blocks; task += GemmCTAs) {
        Gemm::load(inputs,
                   task / token_blocks,
                   task % token_blocks,
                   reinterpret_cast<char*>(&shared.gemm),
                   params,
                   pipe,
                   write);
      }
    } else if (cta < GemmCTAs && tid >= kMmaThread && tid < kMmaThread + Gemm::NumMmaThreads) {
      auto                 inputs = typename Gemm::Mainloop{}.load_init(make_shape(N, tokens, L::kHidden, 1), params);
      typename Gemm::State read;
      uint32_t             epiphase     = 0;
      int                  token_blocks = (tokens + TokenTile - 1) / TokenTile;
      for (int task = cta; task < (N / FeatureTile) * token_blocks; task += GemmCTAs) {
        Gemm::compute(inputs,
                      task / token_blocks,
                      task % token_blocks,
                      tokens,
                      reinterpret_cast<char*>(&shared.gemm),
                      pipe,
                      read,
                      tid - kMmaThread,
                      shared.epilogue,
                      output_q,
                      output_sf,
                      limit,
                      epibar,
                      epiphase);
      }
    }
  }
};
template <class Stage>
__global__ __launch_bounds__(Stage::kNumThreads) void dispatch_shared_swiglu_kernel(
    MUTLASS_GRID_CONSTANT layout::SymBuffer<Stage::R> const sym,
    MUTLASS_GRID_CONSTANT typename Stage::Params const      params,
    uint8_t*                                                payload,
    uint8_t*                                                scales,
    uint4*                                                  metadata,
    int*                                                    counts,
    int*                                                    offsets,
    int*                                                    status,
    int                                                     tokens,
    mutlass::bfloat16_t*                                    out,
    uint8_t*                                                output_q,
    uint8_t*                                                output_sf,
    float                                                   limit) {
  extern __shared__ __align__(256) char smem[];
  Stage::run(
      sym, params, payload, scales, metadata, counts, offsets, status, tokens, out, smem, output_q, output_sf, limit);
}
}  // namespace mega_moe::decode::mp32::fp4_fp4
