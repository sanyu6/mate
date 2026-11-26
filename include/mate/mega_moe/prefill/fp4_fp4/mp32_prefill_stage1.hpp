#pragma once
#include <musa_fp4.h>

#include "mate/mega_moe/mp31_activation.hpp"
#include "mate/mega_moe/prefill/fp4_fp4/prefill_common.hpp"
#include "mate/mega_moe/prefill/fp4_fp4/stage2_online_common.hpp"

namespace mega_moe::prefill::mp32::fp4_fp4 {
using namespace mute;
enum class Stage1Mode { Full, Compute, Dispatch };
union MUTE_ALIGNAS(64) Pull512 {
  int4 words[4];
};

// Keep W4A8's CTA roles, dispatch and compact-pool scheduler. Only the
// mainloop operand representation changes from mixed RS to native FP4 SS.
struct FusedStage1Config : PrefillGemmConfig {
  static constexpr int kM = 256, kN = 128, kBlockK = 256;
  static constexpr int kWideBlockK  = 512;
  static constexpr int kActivationN = kN / 2, kInterleave = 32, kFragmentGate = 8;
  using Impl                                = SsGroupedGemm<kM, kN>;
  using Mainloop                            = Stage2OnlineMainloop<typename Impl::Mainloop>;
  static constexpr int kConsumerGroups      = 2;
  static constexpr int kConsumerSyncBuffers = 2;
  static constexpr int kTeamWarps           = Mainloop::NumTCEWarps;
  static constexpr int kTeamThreads         = kTeamWarps * mutlass::NumThreadsPerWarp;
  static constexpr int kNumMmaWarps         = kConsumerGroups * kTeamWarps;
  static constexpr int kNumMmaThreads       = kNumMmaWarps * mutlass::NumThreadsPerWarp;
  static constexpr int kNumThreads          = kFirstMmaThread + kNumMmaThreads;
  static constexpr int kThreads             = kNumThreads;
};

template <class L, uint32_t kNumMPs, Stage1Mode Mode, bool kUseSitu = false, bool kUseSituLinear = false>
struct Stage1 : FusedStage1Config {
  static constexpr uint32_t kNumRanks = L::kRanks, kRanks = L::kRanks;
  static constexpr uint32_t kNumExperts = L::kExperts, kNumTopk = L::kTopK;
  static constexpr uint32_t kNumMaxTokensPerRank = L::kTokens;
  static constexpr uint32_t kHidden = L::kHidden, kIntermediateHidden = L::kIntermediate;
  // Use the wider native SQMMA only when the complete hidden dimension fits.
  // Keep K256 for smaller or non-K512-aligned supported shapes.
  static constexpr int kBlockK = kHidden % kWideBlockK == 0 ? kWideBlockK : FusedStage1Config::kBlockK;
  using Impl                   = SsGroupedGemm<kM, kN, kBlockK>;
  using Mainloop               = Stage2OnlineMainloop<typename Impl::Mainloop>;
  static_assert(kTeamWarps == Mainloop::NumTCEWarps);

  static constexpr uint32_t kNumMaxPoolTokens =
      layout::get_num_max_pool_tokens(kNumRanks, kNumMaxTokensPerRank, kNumTopk, kNumExperts / kNumRanks, uint32_t(kM));
  static constexpr uint32_t kTokenBytes = kHidden / Impl::kValuesPerByte;
  static constexpr uint32_t kBlockM = kM, kBlockN = kN;
  // Native SS C lanes cover four columns and eight rows per warp. Offset
  // neighboring columns by eight bank words instead of a full bank period.
  static constexpr int kActivationRowPadding =
      (mutlass::NumThreadsPerWarp / (kInterleave / kFragmentGate)) * sizeof(uint32_t) / sizeof(mutlass::bfloat16_t);
  static constexpr int      kActivationRowStride    = kBlockM + kActivationRowPadding;
  static constexpr int      kScaleBlockElements     = Impl::kScaleBlock;
  static constexpr int      kNumDispatchCountArrays = 2, kPackedScaleElements = sizeof(uint32_t);
  static constexpr int      kNumExpertsPerRank = L::kGroups, kNumOwnerCtas = 4;
  static constexpr int      kNumDispatchWarps = 3, kNumDispatchSetupWarps = 8;
  static constexpr int      kTileDescBuffers = kConsumerGroups;
  static constexpr bool     kComputeOnly = Mode == Stage1Mode::Compute, kDispatchOnly = Mode == Stage1Mode::Dispatch;
  static constexpr uint32_t kNumSF             = kHidden / kScaleBlockElements;
  static constexpr uint32_t kScaleCopyRows     = sizeof(uint32_t) / sizeof(uint8_t);
  static constexpr uint32_t kScaleScratchBytes = kNumDispatchWarps * kScaleCopyRows * kNumSF;
  static_assert(kBlockM % kScaleCopyRows == 0 && kNumMaxPoolTokens % kScaleCopyRows == 0);
  using CollectiveMma   = Mainloop;
  using PipelineAB      = typename CollectiveMma::MainloopPipelineData;
  using ConsumerBarrier = mutlass::arch::AsyncBarrier;
  using SS              = Stage2OnlinePool<2 * kIntermediateHidden, kHidden, kN, kBlockK>;
  using Params          = typename SS::Params;
  static_assert(kNumMPs % kNumOwnerCtas == 0);
  static_assert(kFirstMmaWarp == 4 && kNumMmaWarps >= 4);
  struct BarrierStorage {
    uint8_t PipelineAB[PipelineAB::NumBarriers];
    uint8_t Consumer[kConsumerSyncBuffers * kConsumerGroups], TileDesc[kTileDescBuffers], TokenPull[kNumDispatchWarps],
        Dispatch[2], DispatchSetup[2], DispatchReady[1];
  };
  union SharedStorage {
    typename CollectiveMma::SharedStorage mainloop;
    alignas(kTmeSmemAlignment) mutlass::bfloat16_t gemm[kActivationRowStride * kBlockN];
  };
  static constexpr int SmemSize =
      math::constexpr_align<uint32_t>(sizeof(SharedStorage), kSmemAllocationAlignment) +
      math::constexpr_align<uint32_t>(kNumDispatchCountArrays * kNumExperts * sizeof(uint32_t) + kScaleScratchBytes,
                                      kSmemAllocationAlignment) +
      kNumDispatchWarps * math::constexpr_align<uint32_t>(kTokenBytes, kSmemAllocationAlignment);
  static_assert(SmemSize <= kSmemCapacityBytes);
  static_assert(sizeof(BarrierStorage) + ConsumerBarrier::ReservedAsyncBarrierCount <=
                ConsumerBarrier::HardwareMaxNumAsyncTransactionBarriers);

  MUTLASS_DEVICE static void run(uint32_t                     num_tokens,
                                 layout::SymBuffer<kNumRanks> sym_buffer,
                                 uint8_t*                     l1_acts,
                                 uint8_t*                     l1_acts_sf,
                                 uint8_t*                     l2_acts,
                                 uint8_t*                     l2_acts_sf,
                                 int*                         counts,
                                 int64_t*                     metadata,
                                 int*                         status,
                                 Params const&                params,
                                 float                        activation_clamp,
                                 char*                        smem_buffer,
                                 float                        activation_beta        = 1.f,
                                 float                        activation_linear_beta = 1.f) {
    __shared__ uint32_t smem_l1_tile_desc[kTileDescBuffers][stage1_sched::L1TileDesc::kNumWords];

    auto& shared          = *reinterpret_cast<SharedStorage*>(smem_buffer);
    auto* barrier_storage = reinterpret_cast<BarrierStorage*>(0);
    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage) + ConsumerBarrier::ReservedAsyncBarrierCount);
    auto workspace = PrefillSplitWorkspace(L::template at<char>(sym_buffer.template get_base_ptr<char*>(), L::kBytes),
                                           kNumRanks,
                                           kNumExperts,
                                           kNumMaxTokensPerRank,
                                           kNumTopk);
    typename PipelineAB::Params pp;
    pp.transaction_bytes = CollectiveMma::TmeTransactionBytes;
    pp.num_consumers     = kTeamWarps;
    pp.num_producers     = 1;
    PipelineAB                         pipeline(pp, reinterpret_cast<uint64_t>(&barrier_storage->PipelineAB));
    auto                               pipe_write = mutlass::make_producer_start_state<PipelineAB>();
    typename PipelineAB::PipelineState pipe_read;
    int                                warp_idx          = mutlass::canonical_warp_idx_sync();
    bool                               is_consumer_squad = warp_idx >= kFirstMmaWarp;
    bool                               is_tme_issue_warp = warp_idx == kNumDispatchWarps;
    uint32_t                           consumer_barrier_base =
        reinterpret_cast<uint64_t>(&barrier_storage->Consumer) + ConsumerBarrier::ReservedAsyncBarrierCount;
    uint32_t tile_desc_barrier_base =
        reinterpret_cast<uint64_t>(&barrier_storage->TileDesc) + ConsumerBarrier::ReservedAsyncBarrierCount;
    if (threadIdx.x == 0) {
      for (int team = 0; team < kConsumerGroups; ++team) {
        for (int slot = 0; slot < kConsumerSyncBuffers; ++slot)
          ConsumerBarrier::init(consumer_barrier_base + kConsumerSyncBuffers * team + slot, kTeamWarps, 0);
        ConsumerBarrier::init(tile_desc_barrier_base + team, 1 + kTeamWarps, 0);
      }
      const uint32_t dispatch_barrier_base =
          static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->Dispatch)) +
          ConsumerBarrier::ReservedAsyncBarrierCount;
      const uint32_t token_pull_barrier_base =
          static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->TokenPull)) +
          ConsumerBarrier::ReservedAsyncBarrierCount;
      for (uint32_t i = 0; i < kNumDispatchWarps; ++i) {
        ConsumerBarrier::init(token_pull_barrier_base + i, 1, 0);
      }
      ConsumerBarrier::init(dispatch_barrier_base, kNumDispatchWarps, 0);
      ConsumerBarrier::init(dispatch_barrier_base + 1, kNumDispatchWarps, 0);
      const uint32_t dispatch_setup_barrier =
          static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->DispatchSetup)) +
          ConsumerBarrier::ReservedAsyncBarrierCount;
      ConsumerBarrier::init(dispatch_setup_barrier, kNumDispatchSetupWarps, 0);
      ConsumerBarrier::init(dispatch_setup_barrier + 1, kNumDispatchSetupWarps, 0);
      const uint32_t dispatch_ready_barrier =
          static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->DispatchReady)) +
          ConsumerBarrier::ReservedAsyncBarrierCount;
      ConsumerBarrier::init(dispatch_ready_barrier, kNumDispatchWarps, 0);
    }
    __syncthreads();

    const int consumer_group             = is_consumer_squad ? (warp_idx - kFirstMmaWarp) / kTeamWarps : 0;
    uint32_t  tile_desc_barrier_call_idx = consumer_group;
    auto      tile_desc_sync             = [&]() {
      const uint32_t barrier_idx = tile_desc_barrier_base + (tile_desc_barrier_call_idx & (kTileDescBuffers - 1));
      const unsigned phase       = ConsumerBarrier::arrive<true>(barrier_idx);
      ConsumerBarrier::wait(barrier_idx, phase);
      tile_desc_barrier_call_idx += is_consumer_squad ? kConsumerGroups : 1;
    };

    constexpr uint32_t kNumDispatchThreads      = kNumDispatchWarps * mutlass::NumThreadsPerWarp;
    constexpr uint32_t kNumDispatchSetupThreads = kNumDispatchSetupWarps * mutlass::NumThreadsPerWarp;
    const bool         is_dispatch_warp         = warp_idx < int(kNumDispatchWarps);
    const bool         is_dispatch_setup_warp   = warp_idx < int(kNumDispatchSetupWarps);
    if constexpr (!kComputeOnly) {
      if (is_dispatch_setup_warp) {
        const uint32_t     thread_idx        = threadIdx.x;
        const uint32_t     lane_idx          = mutlass::canonical_lane_idx();
        const uint32_t     mp_idx            = blockIdx.x;
        const uint32_t     dispatch_warp_idx = warp_idx;
        constexpr uint32_t kDispatchScratchOffset =
            math::constexpr_align<uint32_t>(static_cast<uint32_t>(sizeof(SharedStorage)), kSmemAllocationAlignment);
        constexpr uint32_t kDispatchCountScratch = math::constexpr_align<uint32_t>(
            kNumDispatchCountArrays * kNumExperts * sizeof(uint32_t) + kScaleScratchBytes, kSmemAllocationAlignment);
        auto*              smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer + kDispatchScratchOffset);
        auto*              smem_rank_prefix  = smem_expert_count + kNumExperts;
        constexpr uint32_t kDispatchTokenStride =
            math::constexpr_align<uint32_t>(kTokenBytes, kSmemAllocationAlignment);
        auto* smem_token_pull =
            reinterpret_cast<int4*>(smem_buffer + kDispatchScratchOffset + kDispatchCountScratch +
                                    (is_dispatch_warp ? dispatch_warp_idx : 0) * kDispatchTokenStride);

        auto* smem_sf_pull = reinterpret_cast<uint8_t*>(smem_expert_count + kNumDispatchCountArrays * kNumExperts) +
                             (is_dispatch_warp ? dispatch_warp_idx : 0) * kScaleCopyRows * kNumSF;

        uint32_t       dispatch_barrier_call_idx = 0;
        const uint32_t dispatch_barrier_base =
            static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->Dispatch)) +
            ConsumerBarrier::ReservedAsyncBarrierCount;
        auto dispatch_sync = [&]() {
          const uint32_t barrier_idx = dispatch_barrier_base + (dispatch_barrier_call_idx & 1u);
          const unsigned phase       = ConsumerBarrier::arrive<true>(barrier_idx);
          ConsumerBarrier::wait(barrier_idx, phase);
          ++dispatch_barrier_call_idx;
        };
        const uint32_t dispatch_setup_barrier =
            static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->DispatchSetup)) +
            ConsumerBarrier::ReservedAsyncBarrierCount;
        uint32_t dispatch_setup_barrier_call_idx = 0;
        auto     dispatch_setup_sync             = [&]() {
          const uint32_t barrier_idx = dispatch_setup_barrier + (dispatch_setup_barrier_call_idx & 1u);
          const unsigned phase       = ConsumerBarrier::arrive<true>(barrier_idx);
          ConsumerBarrier::wait(barrier_idx, phase);
          ++dispatch_setup_barrier_call_idx;
        };

        // Every writer drains its stores before the CTA leader flushes the
        // shared L1 domain; all writers then finish before publishing signals.
        auto publish_setup_writes = [&]() {
          __threadfence_system_noflush();
          dispatch_setup_sync();
          if (thread_idx == 0) __threadfence_system();
          dispatch_setup_sync();
        };

        auto* base      = sym_buffer.template get_base_ptr<char*>();
        auto* input_ids = L::template at<int64_t>(base, L::kRoutes);
        auto* input_x   = L::template at<uint8_t>(base, L::kX);
        auto* input_sf  = L::template at<uint8_t>(base, L::kXS);

        // Full/Dispatch diagnostics may replay without Stage2. Clear only
        // invocation-local control words; preserve persistent barrier phases.
        clear_prefill_split_workspace<kNumExperts, kNumRanks, kNumMPs, kNumDispatchSetupThreads>(workspace);
        if (mp_idx == 0 && thread_idx == 0) *status = 0;
        publish_setup_writes();
        mtlink_barrier<kNumRanks, kNumMPs, kNumDispatchSetupThreads, 0, 3, true>(
            workspace, sym_buffer, mp_idx, thread_idx, dispatch_setup_sync, true, true);

        for (uint32_t expert_idx = thread_idx; expert_idx < kNumExperts; expert_idx += kNumDispatchSetupThreads) {
          smem_expert_count[expert_idx] = 0;
        }
        dispatch_setup_sync();

        const auto for_each_source_token = [&](const auto& process) {
          for (uint32_t token_idx = (mp_idx * kNumDispatchSetupWarps + dispatch_warp_idx); token_idx < num_tokens;
               token_idx += kNumMPs * kNumDispatchSetupWarps) {
            process(token_idx);
            __syncwarp();
          }
        };
        for_each_source_token([&](uint32_t token_idx) {
          int32_t expert_idx = -1;
          if (lane_idx < kNumTopk) {
            expert_idx = static_cast<int32_t>(__ldg(input_ids + token_idx * kNumTopk + lane_idx));
          }
          if (lane_idx < kNumTopk && (expert_idx < -1 || expert_idx >= int(kNumExperts))) atomicExch(status, 1);
          if (expert_idx >= 0 && expert_idx < static_cast<int32_t>(kNumExperts)) {
            atomic_add_shared_u32(smem_expert_count + static_cast<uint32_t>(expert_idx), 1);
          }
        });
        dispatch_setup_sync();

        for (uint32_t expert_idx = thread_idx; expert_idx < kNumExperts; expert_idx += kNumDispatchSetupThreads) {
          const uint64_t send_value     = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[expert_idx]);
          smem_expert_count[expert_idx] = static_cast<uint32_t>(atomic_add_global_u64(
              reinterpret_cast<unsigned long long*>(workspace.get_expert_send_count_ptr(expert_idx)), send_value));
        }
        dispatch_setup_sync();

        if (mp_idx == 0 && thread_idx < kNumRanks) {
          auto* remote_active_count =
              sym_buffer.map(workspace.get_src_active_count_ptr(sym_buffer.rank_idx), thread_idx);
          st_relaxed_sys_global(remote_active_count, 0u);
        }
        publish_setup_writes();
        grid_sync<kNumMPs, 0>(workspace, mp_idx, thread_idx, dispatch_setup_sync);

        for_each_source_token([&](uint32_t token_idx) {
          int32_t expert_idx = -1;
          if (lane_idx < kNumTopk) {
            expert_idx = static_cast<int32_t>(__ldg(input_ids + token_idx * kNumTopk + lane_idx));
          }
          const bool     routed           = expert_idx >= 0 && expert_idx < static_cast<int32_t>(kNumExperts);
          const uint32_t dst_rank_idx     = routed ? static_cast<uint32_t>(expert_idx) / kNumExpertsPerRank : 0;
          const uint32_t local_expert_idx = routed ? static_cast<uint32_t>(expert_idx) % kNumExpertsPerRank : 0;

          uint32_t dst_slot_idx = 0;
          if (routed) {
            dst_slot_idx = atomic_add_shared_u32(smem_expert_count + static_cast<uint32_t>(expert_idx), 1);
            *sym_buffer.map(workspace.get_src_token_topk_idx_ptr(local_expert_idx, sym_buffer.rank_idx, dst_slot_idx),
                            dst_rank_idx) = token_idx * kNumTopk + lane_idx;
          }
        });

        __syncwarp();
        if (lane_idx == 0) {
          __threadfence_system();
        }
        __syncwarp();

        grid_sync<kNumMPs, 0>(workspace, mp_idx, thread_idx, dispatch_setup_sync);
        if (mp_idx == 0) {
          for (uint32_t expert_idx = thread_idx; expert_idx < kNumExperts; expert_idx += kNumDispatchSetupThreads) {
            const uint32_t dst_rank_idx         = expert_idx / kNumExpertsPerRank;
            const uint32_t dst_local_expert_idx = expert_idx % kNumExpertsPerRank;
            const uint64_t expert_status        = ld_volatile_global(workspace.get_expert_send_count_ptr(expert_idx));
            st_relaxed_sys_global(
                sym_buffer.map(workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                               dst_rank_idx),
                expert_status & 0xffffffffull);
            atomic_add_global_u64(reinterpret_cast<unsigned long long*>(sym_buffer.map(
                                      workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx)),
                                  static_cast<unsigned long long>(expert_status));
          }
        }

        publish_setup_writes();
        mtlink_barrier<kNumRanks, kNumMPs, kNumDispatchSetupThreads, 0, 1, true>(
            workspace, sym_buffer, mp_idx, thread_idx, dispatch_setup_sync, true, true);
        const uint32_t dispatch_ready_barrier =
            static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->DispatchReady)) +
            ConsumerBarrier::ReservedAsyncBarrierCount;
        if (is_dispatch_warp) {
          ConsumerBarrier::arrive<false>(dispatch_ready_barrier);
        }

        if (is_dispatch_warp) {
          // The route loop below is latency-sensitive.  Build one CTA-wide prefix
          // table with a single warp, using all 32 lanes to avoid serial HBM loads.
          if (dispatch_warp_idx == 0) {
            uint32_t pool_block_prefix = 0;
            for (uint32_t expert_base = 0; expert_base < kNumExperts; expert_base += mutlass::NumThreadsPerWarp) {
              const uint32_t expert_idx = expert_base + lane_idx;
              const uint32_t expert_tokens =
                  (expert_idx < kNumExpertsPerRank
                       ? static_cast<uint32_t>(ld_volatile_global(workspace.get_expert_recv_count_sum_ptr(expert_idx)))
                       : 0u);
              if (mp_idx == 0 && expert_idx < kNumExpertsPerRank) counts[expert_idx] = expert_tokens;
              const uint32_t expert_blocks = math::ceil_div(expert_tokens, kBlockM);
              uint32_t       inclusive     = expert_blocks;
#pragma unroll
              for (uint32_t offset = 1; offset < mutlass::NumThreadsPerWarp; offset <<= 1) {
                const uint32_t previous = __shfl_up_sync(0xffffffffu, inclusive, offset);
                if (lane_idx >= offset) {
                  inclusive += previous;
                }
              }
              smem_expert_count[expert_idx] = pool_block_prefix + inclusive - expert_blocks;
              pool_block_prefix += __shfl_sync(0xffffffffu, inclusive, mutlass::NumThreadsPerWarp - 1);
            }
            for (uint32_t local_expert_idx = lane_idx; local_expert_idx < kNumExpertsPerRank;
                 local_expert_idx += mutlass::NumThreadsPerWarp) {
              uint32_t rank_token_prefix = 0;
              for (uint32_t source_rank_idx = 0; source_rank_idx < kNumRanks; ++source_rank_idx) {
                smem_rank_prefix[local_expert_idx * kNumRanks + source_rank_idx] = rank_token_prefix;
                rank_token_prefix +=
                    ld_volatile_global(workspace.get_expert_recv_count_ptr(source_rank_idx, local_expert_idx));
              }
            }
          }
          dispatch_sync();

          // kNumOwnerCtas CTAs own each pool block. Their copy warps fill
          // disjoint rows, making complete M tiles available without waiting
          // for the whole active-token stripe. Publish only after every lane's
          // stores are system-visible; each warp contributes its actual row count.
          constexpr uint32_t kNumCopyWarpsPerBlock = kNumOwnerCtas * kNumDispatchWarps;
          uint32_t           local_expert_idx      = 0;
          const uint32_t     last_expert           = kNumExpertsPerRank - 1;
          const uint32_t     pool_blocks =
              smem_expert_count[last_expert] +
              math::ceil_div(
                  static_cast<uint32_t>(ld_volatile_global(workspace.get_expert_recv_count_sum_ptr(last_expert))),
                  kBlockM);
          for (uint32_t pool_block = mp_idx / kNumOwnerCtas; pool_block < pool_blocks;
               pool_block += kNumMPs / kNumOwnerCtas) {
            while (local_expert_idx + 1 < kNumExpertsPerRank && smem_expert_count[local_expert_idx + 1] <= pool_block) {
              ++local_expert_idx;
            }
            const uint32_t expert_tokens =
                static_cast<uint32_t>(ld_volatile_global(workspace.get_expert_recv_count_sum_ptr(local_expert_idx)));
            const uint32_t expert_row_base = (pool_block - smem_expert_count[local_expert_idx]) * kBlockM;
            const uint32_t valid_m         = min(kBlockM, expert_tokens - expert_row_base);
            // The physical slot is only overwritten once every L1 N block of the
            // previous occupant has released it.
            const uint32_t ring_pool_block = pool_block;
            uint32_t       copied          = 0;
            for (uint32_t row_base =
                     ((mp_idx % kNumOwnerCtas) * kNumDispatchWarps + dispatch_warp_idx) * kScaleCopyRows;
                 row_base < valid_m;
                 row_base += kNumCopyWarpsPerBlock * kScaleCopyRows) {
              MUTLASS_PRAGMA_NO_UNROLL
              for (uint32_t chunk_row = 0; chunk_row < kScaleCopyRows; ++chunk_row) {
                uint32_t row = row_base + chunk_row;
                if (row >= valid_m) {
                  constexpr uint32_t kPaddingScales =
                      uint32_t(Fp4Quantization::kScaleBias) *
                      (std::numeric_limits<uint32_t>::max() / std::numeric_limits<uint8_t>::max());
                  for (uint32_t word = lane_idx; word < kNumSF / kPackedScaleElements;
                       word += mutlass::NumThreadsPerWarp)
                    reinterpret_cast<uint32_t*>(smem_sf_pull + chunk_row * kNumSF)[word] = kPaddingScales;
                  __syncwarp();
                  continue;
                }
                const uint32_t expert_row      = expert_row_base + row;
                uint32_t       source_rank_idx = 0;
                for (uint32_t r = 1; r < kNumRanks; ++r) {
                  if (smem_rank_prefix[local_expert_idx * kNumRanks + r] <= expert_row) source_rank_idx = r;
                }
                const uint32_t source_slot =
                    expert_row - smem_rank_prefix[local_expert_idx * kNumRanks + source_rank_idx];
                const uint32_t token_topk = ld_volatile_global(
                    workspace.get_src_token_topk_idx_ptr(local_expert_idx, source_rank_idx, source_slot));
                const uint32_t source_token_idx = token_topk / kNumTopk;
                const uint32_t topk_idx         = token_topk % kNumTopk;
                const uint32_t pool_token       = pool_block * kBlockM + row;
                const uint32_t ring_pool_token  = ring_pool_block * kBlockM + row;
                if (lane_idx == 0) {
                  *workspace.get_token_src_metadata_ptr(pool_token) =
                      PackedTokenSrcMetadata::pack(source_rank_idx, source_token_idx, topk_idx, local_expert_idx);
                  metadata[uint64_t(local_expert_idx) * L::kCapacity + expert_row] =
                      (int64_t(source_rank_idx) * L::kTokens + source_token_idx) * L::kTopK + topk_idx;
                }
                const auto* src_token =
                    sym_buffer.map(input_x + uint64_t(source_token_idx) * kTokenBytes, source_rank_idx);
                const auto* src_sf = sym_buffer.map(input_sf + uint64_t(source_token_idx) * kNumSF, source_rank_idx);
                // Pull one complete token asynchronously. Scale copies overlap the
                // transfer; the per-warp barrier protects shared reads and reuse.
                const uint32_t pull_bar = uint32_t(reinterpret_cast<uint64_t>(&barrier_storage->TokenPull)) +
                                          ConsumerBarrier::ReservedAsyncBarrierCount + dispatch_warp_idx;
                uint32_t pull_phase = 0;
                if (lane_idx == 0) {
                  mutlass::arch::AsyncTransactionBarrier::expect_transaction(pull_bar, kTokenBytes);
                  pull_phase = ConsumerBarrier::arrive<true>(pull_bar);
                  MP31_BULK_COPY_G2S::copy(src_token, pull_bar, smem_token_pull, kTokenBytes);
                }
                pull_phase = __shfl_sync(0xffffffffu, pull_phase, 0);
                static_assert(kNumSF % kPackedScaleElements == 0);
                for (uint32_t word = lane_idx; word < kNumSF / kPackedScaleElements; word += mutlass::NumThreadsPerWarp)
                  reinterpret_cast<uint32_t*>(smem_sf_pull + chunk_row * kNumSF)[word] =
                      __ldg(reinterpret_cast<uint32_t const*>(src_sf) + word);
                ConsumerBarrier::wait(pull_bar, pull_phase);
                using StoreVector = int32_t __attribute__((ext_vector_type(4)));
                auto* dst = reinterpret_cast<volatile StoreVector*>(l1_acts + uint64_t(ring_pool_token) * kTokenBytes);
                auto const* src = reinterpret_cast<StoreVector const*>(smem_token_pull);
                for (uint32_t i = lane_idx; i < kTokenBytes / sizeof(StoreVector); i += mutlass::NumThreadsPerWarp)
                  dst[i] = src[i];
                __syncwarp();
                ++copied;
              }
              __syncwarp();
              for (uint32_t col = lane_idx; col < kNumSF; col += mutlass::NumThreadsPerWarp) {
                uint32_t scales = 0;
                MUTE_UNROLL
                for (uint32_t row = 0; row < kScaleCopyRows; ++row)
                  scales |= uint32_t(smem_sf_pull[row * kNumSF + col]) << (std::numeric_limits<uint8_t>::digits * row);
                uint64_t offset = uint64_t(col) * kNumMaxPoolTokens + ring_pool_block * kBlockM + row_base;
                *reinterpret_cast<volatile uint32_t*>(l1_acts_sf + offset) = scales;
              }
              __syncwarp();
            }
            if (expert_row_base + valid_m == expert_tokens) {
              for (uint32_t row = valid_m + (mp_idx % kNumOwnerCtas) * kNumDispatchWarps + dispatch_warp_idx;
                   row < kBlockM;
                   row += kNumCopyWarpsPerBlock) {
                uint64_t erow = uint64_t(pool_block) * kBlockM + row;
                for (int col = lane_idx; col < L::kIntermediate / 2; col += mutlass::NumThreadsPerWarp)
                  l2_acts[erow * (L::kIntermediate / 2) + col] = 0;
                for (int col = lane_idx; col < L::kIntermediate / kScaleBlockElements;
                     col += mutlass::NumThreadsPerWarp)
                  l2_acts_sf[uint64_t(col) * kNumMaxPoolTokens + erow] = Fp4Quantization::kScaleBias;
              }
            }
            __syncwarp();
            // Order volatile payload and packed scale writes before publishing
            // this warp's contribution to the arrival count.
            __threadfence_system_noflush();
            __syncwarp();
            if (lane_idx == 0 && copied != 0)
              atomic_add_global_u32(workspace.get_l1_arrival_count_ptr(pool_block), copied);
          }

          mtlink_barrier<kNumRanks, kNumMPs, kNumDispatchThreads, 1, 2, true>(
              workspace, sym_buffer, mp_idx, thread_idx, dispatch_sync, true, false);
        }
      }
    }

    if constexpr (!kDispatchOnly) {
      if (!is_dispatch_warp) {
        auto ready = uint32_t(reinterpret_cast<uint64_t>(&barrier_storage->DispatchReady)) +
                     ConsumerBarrier::ReservedAsyncBarrierCount;
        if constexpr (!kComputeOnly) ConsumerBarrier::wait(ready, 0);
        CollectiveMma mainloop;
        auto          inputs = mainloop.load_init(
            make_shape(kNumMaxPoolTokens, 2 * kIntermediateHidden, kHidden, kNumExpertsPerRank), params);
        if (is_tme_issue_warp) {
          using Scheduler = stage1_sched::L1TileScheduler<kBlockM,
                                                          kBlockN,
                                                          kBlockK,
                                                          2 * kIntermediateHidden,
                                                          kHidden,
                                                          kNumExpertsPerRank,
                                                          1,
                                                          kNumMPs,
                                                          kNumRanks,
                                                          false>;
          Scheduler sched(workspace);
          sched.init_l1_dispatch_count_scheduler();
          int tile_desc_slot  = 0;
          int finished_groups = 0;
          while (true) {
            stage1_sched::L1TileDesc next;
            bool                     valid = sched.fetch_next_l1_tile_desc(next);
            if (mutlass::canonical_lane_idx() == 0) {
              if (valid)
                while (load_acquire_global_u32(workspace.get_l1_arrival_count_ptr(next.pool_block_idx)) != next.valid_m)
                  __nanosleep(20);
              next.store_to_smem(smem_l1_tile_desc[tile_desc_slot]);
            }
            __syncwarp();
            tile_desc_sync();
            auto desc = stage1_sched::L1TileDesc::load_from_smem(smem_l1_tile_desc[tile_desc_slot]);
            if (!desc.is_valid()) {
              if (++finished_groups == kConsumerGroups) break;
              tile_desc_slot ^= kTileDescBuffers - 1;
              continue;
            }
            SS::load(inputs,
                     make_coord(desc.pool_block_idx, desc.n_block_idx, desc.local_expert_idx),
                     smem_buffer,
                     params,
                     pipeline,
                     pipe_write);
            tile_desc_slot ^= kTileDescBuffers - 1;
          }
        } else if (is_consumer_squad) {
          int tid = (int(threadIdx.x) - kFirstMmaThread) % kTeamThreads;
          pipe_read.advance(consumer_group * (kHidden / kBlockK));
          typename CollectiveMma::TiledMma mma;
          uint32_t                         sync_index = 0;
          auto                             sync       = [&]() {
            uint32_t bar = consumer_barrier_base + kConsumerSyncBuffers * consumer_group +
                           (sync_index++ & (kConsumerSyncBuffers - 1));
            auto phase = ConsumerBarrier::arrive<true>(bar);
            ConsumerBarrier::wait(bar, phase);
          };
          int tile_desc_slot = consumer_group;
          tile_desc_sync();
          auto desc = stage1_sched::L1TileDesc::load_from_smem(smem_l1_tile_desc[tile_desc_slot]);
          while (desc.is_valid()) {
            // The TME producer acquires arrival before publishing this
            // descriptor to the complete consumer team. Order each reader's
            // global scale loads after that descriptor handoff.
            if constexpr (!kComputeOnly) __threadfence_system_noflush();
            auto accum = partition_fragment_C(mma, take<0, 2>(typename Impl::Tile{}));
            fill(accum, 0.f);
            auto scales = get<2>(inputs);
            scales.ptr_B += uint64_t(desc.local_expert_idx) * scales.n * scales.groups;
            auto expert_inputs = make_tuple(get<0>(inputs), get<1>(inputs), scales);
            mainloop.mma(expert_inputs,
                         make_coord(desc.pool_block_idx, desc.n_block_idx, _, 0),
                         accum,
                         kHidden / kBlockK,
                         tid,
                         smem_buffer,
                         pipeline,
                         pipe_read);
            sync();
            // The other consumer team can execute its MMA while this team
            // activates and quantizes its completed tile.
            pipe_read.advance((kConsumerGroups - 1) * (kHidden / kBlockK));
            auto coords =
                mma.get_thread_slice(tid).partition_C(make_identity_tensor(take<0, 2>(typename Impl::Tile{})));
            constexpr int kVectorElements = sizeof(ActivationFloat4) / sizeof(float);
            constexpr int kLaneGroup      = kScaleBlockElements / kFragmentGate;
            // The native SS accumulator gives each lane four row fragments.
            constexpr int      kRowFragments      = 4;
            constexpr int      kCRowElements      = size(accum) / kRowFragments;
            constexpr int      kGroupsPerRow      = kCRowElements / (2 * kFragmentGate);
            constexpr int      kBf16PerWord       = sizeof(uint32_t) / sizeof(mutlass::bfloat16_t);
            constexpr int      kBf16Bits          = std::numeric_limits<uint16_t>::digits;
            constexpr uint32_t kBf16MagnitudeMask = 0x7fffu;
            constexpr uint32_t kBf16Infinity      = 0x7f80u;
            constexpr uint32_t kBf16StorageMask   = std::numeric_limits<uint16_t>::max();
            constexpr int      kPackedCWords      = size(accum) / kBf16PerWord;
            constexpr int      kRowCWords         = kCRowElements / kBf16PerWord;
            constexpr int      kGateWords         = kFragmentGate / kBf16PerWord;
            constexpr int      kScaleGroups       = kActivationN / kScaleBlockElements;
            static_assert(kLaneGroup == 4 && kFragmentGate == 8 && kRowFragments == 4);
            static_assert(kScaleBlockElements == kInterleave);

            // Each four-lane group owns one 32-value gate/up scale group.
            // Keep BF16 rounding and SwiGLU in registers. TME online swizzle
            // makes each lane's packed FP4 word contiguous without a lane transpose.
            uint32_t rounded[kPackedCWords];
            MUTE_UNROLL
            for (int i = 0; i < size(accum); i += kVectorElements) {
              auto values = __builtin_convertvector(
                  ActivationFloat4{accum(i), accum(i + 1), accum(i + 2), accum(i + 3)}, ActivationBf16x4);
              auto bits                     = __builtin_bit_cast(uint64_t, values);
              rounded[i / kBf16PerWord]     = uint32_t(bits);
              rounded[i / kBf16PerWord + 1] = uint32_t(bits >> std::numeric_limits<uint32_t>::digits);
            }
            const int lane_in_group = tid % kLaneGroup;
            MUTLASS_PRAGMA_NO_UNROLL
            for (int row_fragment = 0; row_fragment < kRowFragments; ++row_fragment) {
              int row = int(get<0>(coords(row_fragment * kCRowElements)));
              if (row < desc.valid_m) {
                MUTE_UNROLL
                for (int group = 0; group < kGroupsPerRow; ++group) {
                  uint32_t activated[kGateWords];
                  uint32_t amax_bits = 0;
                  MUTE_UNROLL
                  for (int col = 0; col < kFragmentGate; col += kVectorElements) {
                    constexpr int kWordBits = std::numeric_limits<uint32_t>::digits;
                    int           word      = group * (2 * kGateWords) + col / kBf16PerWord;
                    uint64_t      gate_bits = uint64_t(rounded[word]) | (uint64_t(rounded[word + 1]) << kWordBits);
                    uint64_t      up_bits =
                        uint64_t(rounded[word + kGateWords]) | (uint64_t(rounded[word + kGateWords + 1]) << kWordBits);
                    ActivationFloat4 gate =
                        __builtin_convertvector(__builtin_bit_cast(ActivationBf16x4, gate_bits), ActivationFloat4);
                    ActivationFloat4 up =
                        __builtin_convertvector(__builtin_bit_cast(ActivationBf16x4, up_bits), ActivationFloat4);
                    ActivationBf16x4 values;
                    if constexpr (kUseSitu) {
                      // Match MP31: sigmoid sees the original BF16 gate,
                      // then gate/up softcaps run in FP32 before one BF16 round.
                      float4       gate4   = make_float4(gate[0], gate[1], gate[2], gate[3]);
                      const float4 sigmoid = ::mega_moe::mp31::activation_sigmoid_bst4(gate4);
                      gate4                = mul(activation_beta,
                                  ::mega_moe::mp31::activation_tanh_bst4(mul(__frcp_rn(activation_beta), gate4)));
                      gate4                = mul(gate4, sigmoid);
                      float4 up4           = make_float4(up[0], up[1], up[2], up[3]);
                      if constexpr (kUseSituLinear)
                        up4 = mul(activation_linear_beta,
                                  ::mega_moe::mp31::activation_tanh_bst4(mul(__frcp_rn(activation_linear_beta), up4)));
                      values = __builtin_convertvector(
                          ActivationFloat4{up4.x * gate4.x, up4.y * gate4.y, up4.z * gate4.z, up4.w * gate4.w},
                          ActivationBf16x4);
                    } else {
                      if (activation_clamp > 0.f) {
                        MUTE_UNROLL
                        for (int j = 0; j < kVectorElements; ++j) {
                          gate[j] = gate[j] < activation_clamp ? gate[j] : activation_clamp;
                          up[j]   = up[j] > -activation_clamp ? up[j] : -activation_clamp;
                          up[j]   = up[j] < activation_clamp ? up[j] : activation_clamp;
                        }
                      } else {
                        MUTE_UNROLL
                        for (int j = 0; j < kVectorElements; ++j) {
                          gate[j] = fminf(gate[j], activation_clamp);
                          up[j]   = fminf(activation_clamp, fmaxf(-activation_clamp, up[j]));
                        }
                      }
                      gate = __builtin_convertvector(__builtin_convertvector(gate, ActivationBf16x4), ActivationFloat4);
                      ActivationFloat4 act = __builtin_convertvector(
                          __builtin_convertvector(silu_fp32_bst4(gate), ActivationBf16x4), ActivationFloat4);
                      values = __builtin_convertvector(up * act, ActivationBf16x4);
                    }
                    auto bits = __builtin_bit_cast(ActivationBits4, values);
                    MUTE_UNROLL
                    for (int j = 0; j < kVectorElements; ++j) {
                      uint32_t magnitude = uint32_t(bits[j]) & kBf16MagnitudeMask;
                      amax_bits          = max(amax_bits, magnitude);
                    }
                    uint64_t packed                   = __builtin_bit_cast(uint64_t, bits);
                    activated[col / kBf16PerWord]     = uint32_t(packed);
                    activated[col / kBf16PerWord + 1] = uint32_t(packed >> kWordBits);
                  }
                  // Finite activations take the common path without per-value
                  // status stores or sanitizing selects. Preserve the original
                  // non-finite handling in the exceptional scale group.
                  if (amax_bits >= kBf16Infinity) {
                    atomicExch(status, 3);
                    amax_bits = 0;
                    MUTE_UNROLL
                    for (int word = 0; word < kGateWords; ++word) {
                      uint32_t raw    = activated[word];
                      uint32_t lo     = raw & kBf16StorageMask;
                      uint32_t hi     = raw >> kBf16Bits;
                      lo              = (lo & kBf16MagnitudeMask) >= kBf16Infinity ? 0u : lo;
                      hi              = (hi & kBf16MagnitudeMask) >= kBf16Infinity ? 0u : hi;
                      activated[word] = lo | (hi << kBf16Bits);
                      amax_bits       = max(amax_bits, max(lo & kBf16MagnitudeMask, hi & kBf16MagnitudeMask));
                    }
                  }
                  uint32_t mask = __activemask();
                  MUTE_UNROLL
                  for (int offset = 1; offset < kLaneGroup; offset *= 2)
                    amax_bits = max(amax_bits, __shfl_xor_sync(mask, amax_bits, offset));
                  float    amax      = __uint_as_float(amax_bits << 16);
                  int      exponent  = Fp4Quantization::scale_exponent(amax);
                  float    inv       = Fp4Quantization::inverse(exponent);
                  uint32_t lane_word = 0;
                  MUTE_UNROLL
                  for (int pair = 0; pair < kGateWords; ++pair) {
                    uint32_t raw    = activated[pair];
                    float    lo     = __uint_as_float((raw & 0xffffu) << 16) * inv;
                    float    hi     = __uint_as_float(raw & 0xffff0000u) * inv;
                    uint8_t  packed = __musa_cvt_float2_to_fp4x2(float2{lo, hi}, __MT_E2M1, musaRoundNearest);
                    lane_word |= uint32_t(packed) << (pair * std::numeric_limits<uint8_t>::digits);
                  }
                  uint32_t output_word  = lane_word;
                  uint64_t outrow       = uint64_t(desc.pool_block_idx) * kBlockM + row;
                  int      native_group = int(get<1>(coords(row_fragment * kCRowElements))) / (2 * kInterleave) + group;
                  int      outgroup     = desc.n_block_idx * kScaleGroups + native_group;
                  if (lane_in_group == 0)
                    l2_acts_sf[uint64_t(outgroup) * kNumMaxPoolTokens + outrow] =
                        uint8_t(exponent + Fp4Quantization::kScaleBias);
                  auto* dst = reinterpret_cast<uint32_t*>(l2_acts + outrow * (L::kIntermediate / Impl::kValuesPerByte) +
                                                          outgroup * (kScaleBlockElements / Impl::kValuesPerByte));
                  dst[lane_in_group] = output_word;
                }
              }
              // Slide the next native row into constant register indices.
              MUTE_UNROLL
              for (int word = 0; word < kPackedCWords - kRowCWords; ++word) rounded[word] = rounded[word + kRowCWords];
            }
            sync();
            tile_desc_sync();
            desc = stage1_sched::L1TileDesc::load_from_smem(smem_l1_tile_desc[tile_desc_slot]);
          }
        }
      }
    }
  }
};

template <class Stage>
__global__ __launch_bounds__(Stage::kThreads) void fused_stage1_kernel(MUTLASS_GRID_CONSTANT
                                                                       typename Stage::Params const     params,
                                                                       layout::SymBuffer<Stage::kRanks> sym,
                                                                       uint8_t*                         l1,
                                                                       uint8_t*                         sf1,
                                                                       uint8_t*                         l2,
                                                                       uint8_t*                         sf2,
                                                                       int*                             counts,
                                                                       int64_t*                         metadata,
                                                                       int*                             status,
                                                                       int                              tokens,
                                                                       float                            clamp) {
  extern __shared__ char __attribute__((aligned(1024))) shared[];
  Stage::run(tokens, sym, l1, sf1, l2, sf2, counts, metadata, status, params, clamp, shared);
}
// Keep the existing SwiGLU FFI/kernel ABI unchanged. Only the SiTU
// specialization carries the two runtime softcap parameters.
template <class Stage>
__global__ __launch_bounds__(Stage::kThreads) void fused_situ_stage1_kernel(MUTLASS_GRID_CONSTANT
                                                                            typename Stage::Params const     params,
                                                                            layout::SymBuffer<Stage::kRanks> sym,
                                                                            uint8_t*                         l1,
                                                                            uint8_t*                         sf1,
                                                                            uint8_t*                         l2,
                                                                            uint8_t*                         sf2,
                                                                            int*                             counts,
                                                                            int64_t*                         metadata,
                                                                            int*                             status,
                                                                            int                              tokens,
                                                                            float                            clamp,
                                                                            float                            beta,
                                                                            float linear_beta) {
  extern __shared__ char __attribute__((aligned(1024))) shared[];
  Stage::run(tokens, sym, l1, sf1, l2, sf2, counts, metadata, status, params, clamp, shared, beta, linear_beta);
}
}  // namespace mega_moe::prefill::mp32::fp4_fp4
