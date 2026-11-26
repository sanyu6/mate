#pragma once
#include <limits>
#include <mute/tensor.hpp>
#include <mutlass/arch/barrier.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
#include <mutlass/pipeline/mp31_pipeline.hpp>

#include "mate/mega_moe/mega_barrier.hpp"
#include "mate/mega_moe/mega_moe_unified_layout.hpp"
#include "mate/mega_moe/mp31_activation.hpp"
#include "mate/mega_moe/stage1_scheduler.hpp"
#include "mate/mega_moe/stage2_scheduler.hpp"

namespace mega_moe::prefill::mp32 {
using namespace mute;

// Fixed geometry shared by both RS mainloops and their launch/epilogue code.
struct PrefillGemmConfig {
  static constexpr int kBlockM = 256, kBlockN = 256, kBlockK = 128;
  using TileShape = Shape<Int<kBlockM>, Int<kBlockN>, Int<kBlockK>>;
  // Warps before the MMA consumers serve dispatch/TME or remain idle.
  static constexpr int kFirstMmaWarp = 4, kNumMmaWarps = 16;
  static constexpr int kFirstMmaThread    = kFirstMmaWarp * mutlass::NumThreadsPerWarp;
  static constexpr int kNumMmaThreads     = kNumMmaWarps * mutlass::NumThreadsPerWarp;
  static constexpr int kNumThreads        = kFirstMmaThread + kNumMmaThreads;
  static constexpr int kVectorStoreBytes  = sizeof(int4);
  static constexpr int kBf16StoreElements = kVectorStoreBytes / sizeof(mutlass::bfloat16_t);
  // Native C fragments alternate two groups of 32 registers. This is a
  // fragment-layout property, independent of the MX scale block size below.
  static constexpr int      kCFragmentGroupElements  = 32;
  static constexpr int      kCFragmentPairElements   = 2 * kCFragmentGroupElements;
  static constexpr int      kCFragmentRowStride      = 4;
  static constexpr uint32_t kSmemAllocationAlignment = 4096u;
  static constexpr int      kSmemBaseAlignment       = 1024;
  static constexpr int      kTmeSmemAlignment        = 256;
  static constexpr int      kSmemCapacityBytes       = 256 * 1024;
};

struct PrefillFp8Scale {
  static constexpr int      kBlockElements = 32;
  static constexpr int      kMantissaBits  = std::numeric_limits<float>::digits - 1;
  static constexpr int      kExponentBias  = std::numeric_limits<float>::max_exponent - 1;
  static constexpr int      kMinExponent   = std::numeric_limits<float>::min_exponent - 1;
  static constexpr int      kMaxExponent   = kExponentBias;
  static constexpr uint32_t kMantissaMask  = (1u << kMantissaBits) - 1u;
  // E4M3 max finite is 448 = 1.75 * 2^8. The BF16 amax needs the
  // next scale exponent exactly when its FP32 mantissa exceeds .75.
  static constexpr int      kE4M3MaxExponent       = 8;
  static constexpr int      kScaleExponentOffset   = kExponentBias + kE4M3MaxExponent;
  static constexpr uint32_t kE4M3MantissaThreshold = 3u << (kMantissaBits - 2);
  // Reciprocal of 2^127 is the FP32 subnormal 2^-127.
  static constexpr uint32_t kInverseAtMaxExponent = 1u << (kMantissaBits - 1);
};
template <int N, int K, int Stages>
struct RsMainloop : PrefillGemmConfig {
  static constexpr int kOperandAlignmentBytes = 16;
  static constexpr int kFp8AlignmentElements  = kOperandAlignmentBytes / sizeof(mutlass::float_e4m3_t);
  // Each byte packs two FP4 elements.
  static constexpr int kFp4AlignmentElements = 2 * kOperandAlignmentBytes;
  using A                                    = mutlass::mx_float8_t<mutlass::float_e4m3_t>;
  using B                                    = mutlass::mx_float4_t<mutlass::float_e2m1_t>;
  using StrideA                              = Stride<Int<K>, _1, int64_t>;
  using StrideB                              = Stride<Int<K>, _1, int64_t>;
  using Collective                           = typename mutlass::gemm::collective::CollectiveBuilder<
                                mutlass::arch::Mp32,
                                mutlass::arch::OpClassBlockScaledTensorOp,
                                A,
                                StrideA,
                                kFp8AlignmentElements,
                                B,
                                StrideB,
                                kFp4AlignmentElements,
                                float,
                                TileShape,
                                Shape<_1, _1, _1>,
                                mutlass::gemm::collective::StageCount<Stages>,
                                mutlass::gemm::KernelTmeWarpSpecializedBlockScaledMixedInputMp32>::CollectiveOp;
  using Pipeline = typename Collective::MainloopPipelineData;
  using State    = typename Pipeline::PipelineState;
  using Params   = typename Collective::Params;
  using TiledMma = typename Collective::TiledMma;
  static_assert(Collective::SwapAB && Collective::NumTCEWarpSquads == 4);
  static Params params(int pool, int groups, const void* a, const void* sa, const void* b, const void* sb) {
    // The activation pool is shared by all experts, including its scale tensor.
    // Construct native descriptors here to avoid changing the mutlass submodule.
    auto                           shape = make_shape(pool, N, K, groups);
    typename Collective::Arguments args{static_cast<const mutlass::float_e4m3_t*>(a),
                                        StrideA{Int<K>{}, _1{}, 0},
                                        static_cast<const mutlass::float_e2m1_t*>(b),
                                        StrideB{Int<K>{}, _1{}, int64_t(N) * K},
                                        static_cast<const mutlass::float_ue8m0_t*>(sa),
                                        static_cast<const mutlass::float_ue8m0_t*>(sb)};
    auto                           p = Collective::to_underlying_arguments(shape, args, nullptr);
    // Internally B is the FP8 activation operand after SwapAB.
    p.layout_sfb = make_layout(
        make_shape(
            pool, make_shape(Int<PrefillFp8Scale::kBlockElements>{}, K / PrefillFp8Scale::kBlockElements), groups),
        make_stride(_1{}, make_stride(_0{}, int64_t(pool)), int64_t(0)));
    auto sf = make_tensor(static_cast<const mutlass::float_ue8m0_t*>(sa), p.layout_sfb);
    p.tme_load_sfb =
        make_tme_copy<typename Collective::TmeInternalElementSF>(typename Collective::GmemTiledCopySFB{},
                                                                 sf,
                                                                 take<0, 2>(typename Collective::SmemLayoutSFB{}),
                                                                 make_shape(Int<kBlockM>{}, Int<kBlockK>{}));
    return p;
  }
  // Issue from the sole producer warp (warp 3 in Stage 1, warp 0 in Stage 2).
  // Collective::load hardcodes warp 0; the copies below use the same partitions.
  template <class Inputs, class Coord>
  MUTLASS_DEVICE static void load(
      Inputs const& inputs, Coord const& coord, char* smem, Params const& p, Pipeline& pipe, State& write) {
    auto& s             = *reinterpret_cast<typename Collective::SharedStorage*>(smem);
    auto  sA            = make_tensor(make_smem_ptr(s.smem_A.begin()), typename Collective::SmemLayoutA{});
    auto  sB            = make_tensor(make_smem_ptr(s.smem_B.begin()), typename Collective::SmemLayoutB{});
    auto  sSA           = make_tensor(make_smem_ptr(s.smem_SFA.begin()), typename Collective::SmemLayoutSFA{});
    auto  sSB           = make_tensor(make_smem_ptr(s.smem_SFB.begin()), typename Collective::SmemLayoutSFB{});
    auto [a, b, sa, sb] = inputs;
    auto [n, m, k, e]   = coord;
    auto ca             = p.tme_load_a.get_slice(0);
    auto cb             = p.tme_load_b.get_slice(0);
    auto csa            = p.tme_load_sfa.get_slice(0);
    auto csb            = p.tme_load_sfb.get_slice(0);
    auto ga             = ca.partition_S(a(_, _, n, _, e));
    auto gb             = cb.partition_S(b(_, _, m, _, e));
    auto gsa            = csa.partition_S(sa(_, _, n, _, e));
    auto gsb            = csb.partition_S(sb(_, _, m, _, e));
    auto da             = ca.partition_D(sA);
    auto db             = cb.partition_D(sB);
    auto dsa            = csa.partition_D(sSA);
    auto dsb            = csb.partition_D(sSB);
    MUTLASS_PRAGMA_NO_UNROLL
    for (int kb = 0; kb < K / kBlockK; ++kb) {
      pipe.producer_acquire(write);
      auto bar = pipe.producer_get_barrier_id(write);
      int  st  = write.index();
      copy(p.tme_load_a.with(bar), ga(_, _, _, kb), da(_, _, _, st));
      copy(p.tme_load_b.with(bar), gb(_, _, _, kb), db(_, _, _, st));
      copy(p.tme_load_sfa.with(bar), gsa(_, _, _, kb), dsa(_, _, _, st));
      copy(p.tme_load_sfb.with(bar), gsb(_, _, _, kb), dsb(_, _, _, st));
      ++write;
    }
  }
};
template <uint32_t kNumExperts, uint32_t kNumRanks, uint32_t kNumMPs, uint32_t kNumThreads>
__device__ __forceinline__ void clear_prefill_split_workspace(const PrefillSplitWorkspace& workspace) {
  constexpr uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks;
  const uint32_t     global_thread_idx  = blockIdx.x * kNumThreads + threadIdx.x;
  constexpr uint32_t kNumGlobalThreads  = kNumMPs * kNumThreads;

  for (uint32_t expert_idx = global_thread_idx; expert_idx < kNumExperts; expert_idx += kNumGlobalThreads) {
    st_relaxed_sys_global(workspace.get_expert_send_count_ptr(expert_idx), uint64_t{0});
  }
  for (uint32_t slot_idx = global_thread_idx; slot_idx < kNumExperts; slot_idx += kNumGlobalThreads) {
    const uint32_t rank_idx         = slot_idx / kNumExpertsPerRank;
    const uint32_t local_expert_idx = slot_idx % kNumExpertsPerRank;
    st_relaxed_sys_global(workspace.get_expert_recv_count_ptr(rank_idx, local_expert_idx), uint64_t{0});
  }
  for (uint32_t local_expert_idx = global_thread_idx; local_expert_idx < kNumExpertsPerRank;
       local_expert_idx += kNumGlobalThreads) {
    st_relaxed_sys_global(workspace.get_expert_recv_count_sum_ptr(local_expert_idx), uint64_t{0});
  }
  const uint64_t kNumRouteCountEntries = static_cast<uint64_t>(kNumRanks) * workspace.num_max_tokens_per_rank;
  for (uint64_t route_idx = global_thread_idx; route_idx < kNumRouteCountEntries; route_idx += kNumGlobalThreads) {
    const uint32_t source_rank_idx  = static_cast<uint32_t>(route_idx / workspace.num_max_tokens_per_rank);
    const uint32_t source_token_idx = static_cast<uint32_t>(route_idx % workspace.num_max_tokens_per_rank);
    st_relaxed_sys_global(workspace.get_src_route_count_ptr(source_rank_idx, source_token_idx), uint32_t{0});
  }
  for (uint32_t pool_block_idx = global_thread_idx; pool_block_idx < workspace.num_max_pool_blocks;
       pool_block_idx += kNumGlobalThreads) {
    st_relaxed_sys_global(workspace.get_l1_arrival_count_ptr(pool_block_idx), uint32_t{0});
    st_relaxed_sys_global(workspace.get_l2_arrival_mask_ptr(pool_block_idx), uint64_t{0});
    st_relaxed_sys_global(workspace.get_l1_consumed_mask_ptr(pool_block_idx), uint64_t{0});
  }
}

}  // namespace mega_moe::prefill::mp32
