#pragma once
#include "mate/mega_moe/prefill/fp4_fp4/stage2_online_common.hpp"
#include "mate/mega_moe/prefill/fp4_fp4/symmetric_layout.hpp"
namespace mega_moe::prefill::mp32::fp4_fp4 {

using Stage2StoreWords4  = uint32_t __attribute__((ext_vector_type(4)));
using Stage2GlobalVector = Stage2StoreWords4 __attribute__((address_space(1)));

template <class L, uint32_t MPs>
struct Stage2 : PrefillGemmConfig {
  static constexpr uint32_t R = L::kRanks, E = L::kExperts, S = L::kTokens, T = L::kTopK, H = L::kHidden,
                            I          = L::kIntermediate;
  static constexpr uint32_t kNumRanks_ = R;
  static constexpr uint32_t P          = layout::get_num_max_pool_tokens(R, S, T, E / R, uint32_t(256));
  using SS                             = Stage2OnlinePool<H, I>;
  using Collective                     = typename SS::Collective;
  using Pipeline                       = typename SS::Pipeline;
  using Params                         = typename SS::Params;
  static constexpr int kBlockM = 256, kBlockN = 256, kBlockK = SS::Impl::kBlockK;
  static constexpr int kNumMmaWarps   = Collective::NumTCEWarps;
  static constexpr int kNumMmaThreads = kNumMmaWarps * mutlass::NumThreadsPerWarp;
  static constexpr int kNumThreads    = kFirstMmaThread + kNumMmaThreads;
  static constexpr int SmemSize       = sizeof(typename Collective::SharedStorage);
  static_assert(SmemSize <= kSmemCapacityBytes);
  MUTLASS_DEVICE static void run(layout::SymBuffer<R> sym_buffer, Params const& params, int* counts, char* smem) {
    mutlass::arch::allocate_async_barriers(Pipeline::NumBarriers);
    typename Pipeline::Params pp;
    pp.transaction_bytes = Collective::TmeTransactionBytes;
    pp.num_consumers     = kNumMmaWarps;
    pp.num_producers     = 1;
    Pipeline           pipeline(pp, 0);
    auto               write = mutlass::make_producer_start_state<Pipeline>();
    typename SS::State read;
    auto*              base      = sym_buffer.template get_base_ptr<char*>();
    auto               workspace = PrefillSplitWorkspace(L::template at<char>(base, L::kBytes), R, E, S, T);
    auto*              combine   = L::template at<mutlass::bfloat16_t>(base, L::kCombine);
    Collective         mainloop;
    auto               inputs = mainloop.load_init(make_shape(P, H, I, E / R), params);
    using Scheduler           = mate_stage2::WorkspaceTileScheduler<E / R, H / kBlockN, MPs, R, kBlockM, true>;
    __syncthreads();
    int warp = mutlass::canonical_warp_idx_sync();
    if (warp == 0) {
      Scheduler scheduler;
      auto      work = scheduler.initial_work_tile_info(workspace);
      while (work.valid) {
        SS::load(
            inputs, make_coord(work.pool_block_idx, work.n_block_idx, work.expert_idx), smem, params, pipeline, write);
        scheduler.advance_to_next_work();
        work = scheduler.get_work_tile_info();
      }
    } else if (warp >= kFirstMmaWarp) {
      int                           tid = int(threadIdx.x) - kFirstMmaThread;
      typename Collective::TiledMma mma;
      Scheduler                     scheduler;
      auto                          work = scheduler.initial_work_tile_info(workspace);
      while (work.valid) {
        auto accum = partition_fragment_C(mma, take<0, 2>(tile_shape(mma)));
        fill(accum, 0.f);
        auto scales = get<2>(inputs);
        scales.ptr_B += uint64_t(work.expert_idx) * scales.n * scales.groups;
        auto expert_inputs = make_tuple(get<0>(inputs), get<1>(inputs), scales);
        mainloop.mma(expert_inputs,
                     make_coord(work.pool_block_idx, work.n_block_idx, _, 0),
                     accum,
                     I / kBlockK,
                     tid,
                     smem,
                     pipeline,
                     read);
        // TME online swizzle makes each lane own groups of eight adjacent
        // BF16 outputs; pack and store them without exchanging lanes.
        auto          coords = mma.get_thread_slice(tid).partition_C(make_identity_tensor(take<0, 2>(tile_shape(mma))));
        constexpr int kRowFragments    = 4;
        constexpr int kRowElements     = size(accum) / kRowFragments;
        constexpr int kBf16PerWord     = sizeof(uint32_t) / sizeof(mutlass::bfloat16_t);
        constexpr int kWordBits        = std::numeric_limits<uint32_t>::digits;
        constexpr int kConvertElements = 4;
        constexpr int kLaneGroup       = 4;
        constexpr int kRowWords        = kRowElements / kBf16PerWord;
        constexpr int kPackedWords     = size(accum) / kBf16PerWord;
        using Float4                   = float __attribute__((ext_vector_type(kConvertElements)));
        using Bf16x4                   = __bf16 __attribute__((ext_vector_type(kConvertElements)));
        static_assert(kRowElements == 32 && kLaneGroup == 4);
        uint32_t rounded[kPackedWords];
        MUTE_UNROLL
        for (int index = 0; index < size(accum); index += kConvertElements) {
          auto values = __builtin_convertvector(
              Float4{accum(index), accum(index + 1), accum(index + 2), accum(index + 3)}, Bf16x4);
          uint64_t bits                     = __builtin_bit_cast(uint64_t, values);
          rounded[index / kBf16PerWord]     = uint32_t(bits);
          rounded[index / kBf16PerWord + 1] = uint32_t(bits >> kWordBits);
        }
        MUTLASS_PRAGMA_NO_UNROLL
        for (int row_fragment = 0; row_fragment < kRowFragments; ++row_fragment) {
          int m = int(get<0>(coords(row_fragment * kRowElements)));
          if (m < work.valid_m) {
            auto          metadata        = *workspace.get_token_src_metadata_ptr(work.pool_block_idx * kBlockM + m);
            auto*         destination     = combine + (uint64_t(metadata.token_idx()) * T + metadata.topk_idx()) * H;
            auto*         ptr             = sym_buffer.map(destination, metadata.rank_idx());
            int           native_n        = int(get<1>(coords(row_fragment * kRowElements)));
            constexpr int kValuesPerStore = sizeof(int4) / sizeof(mutlass::bfloat16_t);
            constexpr int kWordsPerStore  = sizeof(int4) / sizeof(uint32_t);
            constexpr int kGroupStride    = kLaneGroup * kValuesPerStore;
            MUTE_UNROLL
            for (int group = 0; group < kRowElements / kValuesPerStore; ++group) {
              Stage2StoreWords4 payload{rounded[group * kWordsPerStore],
                                        rounded[group * kWordsPerStore + 1],
                                        rounded[group * kWordsPerStore + 2],
                                        rounded[group * kWordsPerStore + 3]};
              auto* dst = (Stage2GlobalVector*)(ptr + work.n_block_idx * kBlockN + native_n + group * kGroupStride);
              *dst      = payload;
            }
          }
          MUTE_UNROLL
          for (int word = 0; word < kPackedWords - kRowWords; ++word) rounded[word] = rounded[word + kRowWords];
        }
        scheduler.advance_to_next_work();
        work = scheduler.get_work_tile_info();
      }
    }
    // Order every writer warp before the CTA leader flushes its local cache.
    // Peers may start Stage3 as soon as the noflush MTLink barrier completes.
    __syncthreads();
    if (threadIdx.x == 0) __threadfence_system();
    __syncthreads();
    mtlink_barrier<R, MPs, kNumThreads, 0, 1, true>(
        workspace, sym_buffer, blockIdx.x, threadIdx.x, [&]() { __syncthreads_lm(); }, true, true);
    clear_prefill_split_workspace<E, R, MPs, kNumThreads>(workspace);
    for (uint32_t e = blockIdx.x * kNumThreads + threadIdx.x; e < E / R; e += MPs * kNumThreads) counts[e] = 0;
    __threadfence_system_noflush();
    mtlink_barrier<R, MPs, kNumThreads, 1, 2, true>(
        workspace, sym_buffer, blockIdx.x, threadIdx.x, [&]() { __syncthreads_lm(); }, true, true);
  }
};
template <class Impl>
__global__ __launch_bounds__(Impl::kNumThreads, 1) void stage2_kernel(layout::SymBuffer<Impl::kNumRanks_> sym,
                                                                      typename Impl::Params               params,
                                                                      int*                                counts) {
  extern __shared__ __align__(PrefillGemmConfig::kSmemBaseAlignment) char smem[];
  Impl::run(sym, params, counts, smem);
}
}  // namespace mega_moe::prefill::mp32::fp4_fp4
