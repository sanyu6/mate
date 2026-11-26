#pragma once
#include "mate/gemm/deep_gemm/mp32_fp8_fp4_gemm_1d1d.hpp"
#include "mate/mega_moe/prefill/mp32_prefill_common.hpp"
namespace mega_moe::prefill::mp32 {
using ActivationFloat4 = float __attribute__((ext_vector_type(4)));
using ActivationBf16x4 = __bf16 __attribute__((ext_vector_type(4)));
using ActivationBits4  = uint16_t __attribute__((ext_vector_type(4)));
// This SDK exposes the LLVM intrinsic but not a C++ RCP BST4 builtin.
extern __device__ ActivationFloat4 activation_rcp_bst4(ActivationFloat4) __asm__("llvm.musa.rcp.f.bst4");
union MUTE_ALIGNAS(64) Pull512 {
  int4 words[4];
};

template <uint32_t kNumMaxTokensPerRank,
          uint32_t kHidden,
          uint32_t kIntermediateHidden,
          uint32_t kNumExperts,
          uint32_t kNumTopk,
          uint32_t kNumMaxPoolTokens,
          uint32_t kNumMPs,
          uint32_t kNumRanks,
          bool     kUseSitu,
          bool     kUseSituLinear,
          bool     kDispatchOnly = false,
          bool     kComputeOnly  = false,
          uint32_t kStages       = 4,
          uint32_t kOwnerCtas    = 4>
struct Stage1 : PrefillGemmConfig {
  using Fp8Scale                       = PrefillFp8Scale;
  static constexpr uint32_t kNumRanks_ = kNumRanks;
  using RS                             = RsMainloop<2 * kIntermediateHidden, kHidden, kStages>;
  using CollectiveMma                  = typename RS::Collective;
  using PipelineAB                     = typename RS::Pipeline;
  using ConsumerBarrier                = mutlass::arch::AsyncBarrier;
  static constexpr uint32_t kBlockM = RS::kBlockM, kBlockN = RS::kBlockN, kBlockK = RS::kBlockK;
  static constexpr int      kScaleBlockElements = Fp8Scale::kBlockElements;
  static constexpr int      kActivationRows = kBlockM, kActivationColumns = kBlockN / 2;
  static constexpr int      kActivationScaleBlocks  = kActivationColumns / kScaleBlockElements;
  static constexpr int      kBstVectorElements      = sizeof(ActivationFloat4) / sizeof(float);
  static constexpr int      kNumDispatchCountArrays = 2;
  static constexpr uint32_t kPackedScaleElements    = sizeof(uint32_t) / sizeof(uint8_t);
  static constexpr int      kNumExpertsPerRank = kNumExperts / kNumRanks, kNumOwnerCtas = kOwnerCtas;
  static constexpr int      kNumDispatchWarps = 3, kNumDispatchSetupWarps = 8;
  static constexpr int      kL1RingTokens  = kNumMaxPoolTokens;
  static constexpr bool     kL1RingEnabled = false;
  static_assert(!kComputeOnly || !kDispatchOnly);
  static_assert(kNumMPs % kNumOwnerCtas == 0);
  struct BarrierStorage {
    uint8_t PipelineAB[PipelineAB::NumBarriers];
    uint8_t Consumer[2], TileDesc[2], TokenPull[kNumDispatchWarps], Dispatch[2], DispatchSetup[2], DispatchReady[1];
  };
  MUTLASS_DEVICE static ActivationFloat4 native_silu_bst4(ActivationFloat4 gate) {
    ActivationFloat4 x;
    MUTE_UNROLL
    for (int j = 0; j < kBstVectorElements; ++j)
      x[j] = fminf(float(Fp8Scale::kMaxExponent + 1),
                   fmaxf(float(Fp8Scale::kMinExponent), -gate[j] * ::mega_moe::mp31::kActivationLog2e));
    ActivationFloat4 value = __musa_exp2_f_bst4(x);
    return gate * activation_rcp_bst4(value + 1.f);
  }
  union SharedStorage {
    typename CollectiveMma::SharedStorage mainloop;
    alignas(kTmeSmemAlignment) mutlass::bfloat16_t activation[kActivationColumns * kActivationRows];
  };
  static constexpr int SmemSize =
      math::constexpr_align<uint32_t>(sizeof(SharedStorage), kSmemAllocationAlignment) +
      math::constexpr_align<uint32_t>(kNumDispatchCountArrays * kNumExperts * sizeof(uint32_t),
                                      kSmemAllocationAlignment) +
      kNumDispatchWarps * math::constexpr_align<uint32_t>(kHidden, kSmemAllocationAlignment);
  static_assert(SmemSize <= kSmemCapacityBytes);
  static_assert(sizeof(BarrierStorage) + ConsumerBarrier::ReservedAsyncBarrierCount <=
                ConsumerBarrier::HardwareMaxNumAsyncTransactionBarriers);
  using Params = typename RS::Params;
  MUTLASS_DEVICE static void run(uint32_t                     num_tokens,
                                 layout::SymBuffer<kNumRanks> sym_buffer,
                                 mutlass::float_e4m3_t*       l1_acts,
                                 uint8_t*                     l1_acts_sf,
                                 mutlass::float_e4m3_t*       l2_acts,
                                 uint8_t*                     l2_acts_sf,
                                 Params const&                params,
                                 float                        activation_clamp,
                                 float                        activation_beta,
                                 float                        activation_linear_beta,
                                 char*                        smem_buffer) {
    __shared__ uint32_t          smem_l1_tile_desc[stage1_sched::L1TileDesc::kNumWords];
    __shared__ volatile uint32_t coordinate_zero;
    auto&                        shared          = *reinterpret_cast<SharedStorage*>(smem_buffer);
    auto*                        barrier_storage = reinterpret_cast<BarrierStorage*>(0);
    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage) + ConsumerBarrier::ReservedAsyncBarrierCount);
    auto                        buffer_layout = unified::BufferLayout(sym_buffer.get_base_ptr(),
                                               kNumRanks,
                                               kNumExperts,
                                               kNumMaxTokensPerRank,
                                               kNumTopk,
                                               kHidden,
                                               kIntermediateHidden,
                                               kNumMaxPoolTokens,
                                               kL1RingTokens);
    auto                        workspace     = buffer_layout.get_prefill_workspace();
    typename PipelineAB::Params pp;
    pp.transaction_bytes = CollectiveMma::TmeTransactionBytes;
    pp.num_consumers     = kNumMmaWarps;
    pp.num_producers     = 1;
    PipelineAB         pipeline(pp, reinterpret_cast<uint64_t>(&barrier_storage->PipelineAB));
    auto               pipe_write = mutlass::make_producer_start_state<PipelineAB>();
    typename RS::State pipe_read;
    int                warp_idx          = mutlass::canonical_warp_idx_sync();
    bool               is_consumer_squad = warp_idx >= kFirstMmaWarp;
    bool               is_tme_issue_warp = warp_idx == kNumDispatchWarps;
    uint32_t           consumer_barrier_base =
        reinterpret_cast<uint64_t>(&barrier_storage->Consumer) + ConsumerBarrier::ReservedAsyncBarrierCount;
    uint32_t tile_desc_barrier_base =
        reinterpret_cast<uint64_t>(&barrier_storage->TileDesc) + ConsumerBarrier::ReservedAsyncBarrierCount;
    if (threadIdx.x == 0) {
      coordinate_zero = 0;
      ConsumerBarrier::init(consumer_barrier_base, kNumMmaWarps, 0);
      ConsumerBarrier::init(consumer_barrier_base + 1, kNumMmaWarps, 0);
      ConsumerBarrier::init(tile_desc_barrier_base, 1 + kNumMmaWarps, 0);
      ConsumerBarrier::init(tile_desc_barrier_base + 1, 1 + kNumMmaWarps, 0);
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

    uint32_t tile_desc_barrier_call_idx = 0;
    auto     tile_desc_sync             = [&]() {
      const uint32_t barrier_idx = tile_desc_barrier_base + (tile_desc_barrier_call_idx & 1u);
      const unsigned phase       = ConsumerBarrier::arrive<true>(barrier_idx);
      ConsumerBarrier::wait(barrier_idx, phase);
      ++tile_desc_barrier_call_idx;
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
            kNumDispatchCountArrays * kNumExperts * sizeof(uint32_t), kSmemAllocationAlignment);
        auto*              smem_expert_count    = reinterpret_cast<uint32_t*>(smem_buffer + kDispatchScratchOffset);
        auto*              smem_rank_prefix     = smem_expert_count + kNumExperts;
        constexpr uint32_t kDispatchTokenStride = math::constexpr_align<uint32_t>(kHidden, kSmemAllocationAlignment);
        auto*              smem_token_pull =
            reinterpret_cast<int4*>(smem_buffer + kDispatchScratchOffset + kDispatchCountScratch +
                                    (is_dispatch_warp ? dispatch_warp_idx : 0) * kDispatchTokenStride);

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

        const auto input_token_buffer        = buffer_layout.get_input_token_buffer();
        const auto input_sf_buffer           = buffer_layout.get_input_sf_buffer();
        const auto input_topk_idx_buffer     = buffer_layout.get_input_topk_idx_buffer();
        const auto input_topk_weights_buffer = buffer_layout.get_input_topk_weights_buffer();
        (void)input_topk_weights_buffer;

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
            expert_idx = static_cast<int32_t>(
                __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx));
          }
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
        __threadfence_system();
        grid_sync<kNumMPs, 0>(workspace, mp_idx, thread_idx, dispatch_setup_sync);

        for_each_source_token([&](uint32_t token_idx) {
          int32_t expert_idx = -1;
          if (lane_idx < kNumTopk) {
            expert_idx = static_cast<int32_t>(
                __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx));
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

        __threadfence_system();
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

          constexpr uint32_t kNumSF = kHidden / kScaleBlockElements;
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
            for (uint32_t row = (mp_idx % kNumOwnerCtas) * kNumDispatchWarps + dispatch_warp_idx;
                 dispatch_warp_idx < kNumDispatchWarps && row < valid_m;
                 row += kNumCopyWarpsPerBlock) {
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
              }
              const auto* src_token =
                  sym_buffer.map(input_token_buffer.get_data_buffer(source_token_idx).get_base_ptr(), source_rank_idx);
              const auto* src_sf = sym_buffer.map(
                  input_sf_buffer.get_data_buffer(source_token_idx).get_base_ptr<uint8_t>(), source_rank_idx);
              // Pull one complete token asynchronously. Scale copies overlap the
              // transfer; the per-warp barrier protects shared reads and reuse.
              const uint32_t pull_bar = uint32_t(reinterpret_cast<uint64_t>(&barrier_storage->TokenPull)) +
                                        ConsumerBarrier::ReservedAsyncBarrierCount + dispatch_warp_idx;
              uint32_t pull_phase = 0;
              if (lane_idx == 0) {
                mutlass::arch::AsyncTransactionBarrier::expect_transaction(pull_bar, kHidden);
                pull_phase = ConsumerBarrier::arrive<true>(pull_bar);
                MP31_BULK_COPY_G2S::copy(src_token, pull_bar, smem_token_pull, kHidden);
              }
              pull_phase = __shfl_sync(0xffffffffu, pull_phase, 0);
              static_assert(kNumSF % kPackedScaleElements == 0);
              for (uint32_t word = lane_idx; word < kNumSF / kPackedScaleElements; word += mutlass::NumThreadsPerWarp) {
                uint32_t values = __ldg(reinterpret_cast<uint32_t const*>(src_sf) + word);
                MUTLASS_PRAGMA_UNROLL
                for (uint32_t j = 0; j < kPackedScaleElements; ++j)
                  l1_acts_sf[uint64_t(word * kPackedScaleElements + j) * kNumMaxPoolTokens + ring_pool_token] =
                      uint8_t(values >> (std::numeric_limits<uint8_t>::digits * j));
              }
              ConsumerBarrier::wait(pull_bar, pull_phase);
              auto* dst = reinterpret_cast<Pull512*>(l1_acts + static_cast<uint64_t>(ring_pool_token) * kHidden);
              for (uint32_t i = lane_idx; i < kHidden / sizeof(Pull512); i += mutlass::NumThreadsPerWarp) {
                dst[i] = reinterpret_cast<const Pull512*>(smem_token_pull)[i];
              }
              __syncwarp();
              ++copied;
            }
            __syncwarp();
            // MP32 TME reads require flushed pool stores before ready is published.
            // A noflush fence produced zero-valued early tiles in the interface test.
            __threadfence_system();
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
        // Native RS inputs and coordinates use weight N first, token M second.
        auto inputs = mainloop.load_init(
            make_shape(2 * kIntermediateHidden, kNumMaxPoolTokens, kHidden, kNumExpertsPerRank), params);
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
                                                          true>;
          Scheduler sched(workspace);
          sched.init_l1_dispatch_count_scheduler();
          while (true) {
            stage1_sched::L1TileDesc next;
            bool                     valid = sched.fetch_next_l1_tile_desc(next);
            if (mutlass::canonical_lane_idx() == 0) {
              if (valid)
                while (load_acquire_global_u32(workspace.get_l1_arrival_count_ptr(next.pool_block_idx)) != next.valid_m)
                  __nanosleep(20);
              next.store_to_smem(smem_l1_tile_desc);
            }
            __syncwarp();
            tile_desc_sync();
            auto desc = stage1_sched::L1TileDesc::load_from_smem(smem_l1_tile_desc);
            if (!desc.is_valid()) break;
            RS::load(inputs,
                     make_coord(desc.n_block_idx, desc.pool_block_idx, _, desc.local_expert_idx),
                     smem_buffer,
                     params,
                     pipeline,
                     pipe_write);
          }
        } else if (is_consumer_squad) {
          int                   tid = int(threadIdx.x) - kFirstMmaThread;
          typename RS::TiledMma mma;
          uint32_t              sync_index = 0;
          auto                  sync       = [&]() {
            uint32_t bar   = consumer_barrier_base + (sync_index++ & 1);
            auto     phase = ConsumerBarrier::arrive<true>(bar);
            ConsumerBarrier::wait(bar, phase);
          };
          while (true) {
            tile_desc_sync();
            auto desc = stage1_sched::L1TileDesc::load_from_smem(smem_l1_tile_desc);
            if (!desc.is_valid()) break;
            auto accum = partition_fragment_C(mma, take<0, 2>(tile_shape(mma)));
            fill(accum, 0.f);

            using GroupedGemm = ::mate::deep_gemm::Mp32Fp8Fp4Gemm1D1D< ::mate::deep_gemm::GemmType::MGroupedContiguous,
                                                                       typename RS::A,
                                                                       typename RS::B,
                                                                       mutlass::bfloat16_t,
                                                                       typename RS::StrideA,
                                                                       typename RS::StrideB,
                                                                       Stride<int64_t, _1, int64_t>,
                                                                       typename RS::TileShape,
                                                                       kStages>;
            static_assert(is_same_v<typename GroupedGemm::CollectiveMma, CollectiveMma>);
            GroupedGemm::mma(inputs,
                             make_coord(desc.n_block_idx, desc.pool_block_idx, _, desc.local_expert_idx),
                             accum,
                             kHidden / kBlockK,
                             tid,
                             smem_buffer,
                             pipeline,
                             pipe_read);

            // All consumer squads must retire SQMMA before activation aliases
            // the mainloop buffers. The producer cannot start the next tile
            // until the consumers reach tile_desc_sync after the epilogue.
            sync();
            // Materialize output coordinates after MMA so their expanded addresses
            // do not stay live across the register-heavy mainloop.
            int  epilogue_tid = tid + coordinate_zero;
            auto coords =
                mma.get_thread_slice(epilogue_tid).partition_C(make_identity_tensor(take<0, 2>(tile_shape(mma))));
            // Round the uniform gate clamp once: min of two BF16 values is
            // already BF16, so each gate needs no second conversion.
            float up_limit   = activation_clamp >= 0 ? activation_clamp : std::numeric_limits<float>::infinity();
            float gate_limit = float(mutlass::bfloat16_t(up_limit));
            // Native RS fragments alternate 32 gate and 32 up registers.
            // Visit only gate fragments; the paired up register is always i+32.
            // All gate fragments share N; successive fragments advance token M by four.
            // Derive this base through MUTE after MMA, then use constant offsets.
            int gate_n = get<0>(coords(0));
            int gate_m = get<1>(coords(0));
            int gate_col =
                (gate_n / kCFragmentPairElements) * kCFragmentGroupElements + gate_n % kCFragmentGroupElements;
            auto* activation_base = shared.activation + gate_col * kActivationRows + gate_m;
            MUTE_UNROLL
            for (int gate_idx = 0; gate_idx < size(accum) / 2; gate_idx += kBstVectorElements) {
              int i =
                  (gate_idx / kCFragmentGroupElements) * kCFragmentPairElements + gate_idx % kCFragmentGroupElements;
              if constexpr (kUseSitu) {
                MUTE_UNROLL
                for (int j = 0; j < kBstVectorElements; ++j) {
                  int n = get<0>(coords(i + j)), m = get<1>(coords(i + j));
                  if (m < desc.valid_m) {
                    float gate = float(mutlass::bfloat16_t(accum(i + j)));
                    float up   = float(mutlass::bfloat16_t(accum(i + j + kCFragmentGroupElements)));
                    float sig  = .5f * (1.f + tanhf(.5f * gate));
                    gate       = activation_beta * tanhf(gate / activation_beta) * sig;
                    if constexpr (kUseSituLinear) up = activation_linear_beta * tanhf(up / activation_linear_beta);
                    int col = (n / kCFragmentPairElements) * kCFragmentGroupElements + n % kCFragmentGroupElements;
                    shared.activation[col * kActivationRows + m] = mutlass::bfloat16_t(up * gate);
                  }
                }
              } else {
                ActivationFloat4 gate = __builtin_convertvector(
                    __builtin_convertvector(ActivationFloat4{accum(i), accum(i + 1), accum(i + 2), accum(i + 3)},
                                            ActivationBf16x4),
                    ActivationFloat4);
                ActivationFloat4 up = __builtin_convertvector(
                    __builtin_convertvector(ActivationFloat4{accum(i + kCFragmentGroupElements),
                                                             accum(i + kCFragmentGroupElements + 1),
                                                             accum(i + kCFragmentGroupElements + 2),
                                                             accum(i + kCFragmentGroupElements + 3)},
                                            ActivationBf16x4),
                    ActivationFloat4);
                MUTE_UNROLL
                for (int j = 0; j < kBstVectorElements; ++j) {
                  gate[j] = fminf(gate[j], gate_limit);
                  up[j]   = fminf(fmaxf(up[j], -up_limit), up_limit);
                }
                ActivationFloat4 act = __builtin_convertvector(
                    __builtin_convertvector(native_silu_bst4(gate), ActivationBf16x4), ActivationFloat4);
                ActivationBf16x4 output = __builtin_convertvector(up * act, ActivationBf16x4);
                ActivationBits4  bits   = __builtin_bit_cast(ActivationBits4, output);
                MUTE_UNROLL
                for (int j = 0; j < kBstVectorElements; ++j) {
                  int offset = (gate_idx + j) * kCFragmentRowStride;
                  if (gate_m + offset < desc.valid_m) activation_base[offset] = mutlass::bfloat16_t::bitcast(bits[j]);
                }
              }
            }
            sync();
            // One consumer thread handles a token/K32 block. This staging keeps
            // native CLayout and logical output order separate and easy to audit.
            for (int item = tid; item < kActivationRows * kActivationScaleBlocks; item += kNumMmaThreads) {
              int m = item % kActivationRows, scale_block = item / kActivationRows;
              if (m >= desc.valid_m) continue;
              float amax = kSGLangSwiGLUFP8AmaxFloor;
              for (int col = 0; col < kScaleBlockElements; ++col)
                amax = fmaxf(
                    amax,
                    fabsf(float(shared.activation[(scale_block * kScaleBlockElements + col) * kActivationRows + m])));
              uint32_t amax_bits = __float_as_uint(amax);
              int      exponent  = int(amax_bits >> Fp8Scale::kMantissaBits) - Fp8Scale::kScaleExponentOffset +
                             int((amax_bits & Fp8Scale::kMantissaMask) > Fp8Scale::kE4M3MantissaThreshold);
              exponent              = max(Fp8Scale::kMinExponent, min(Fp8Scale::kMaxExponent, exponent));
              uint32_t inverse_bits = exponent == Fp8Scale::kMaxExponent
                                          ? Fp8Scale::kInverseAtMaxExponent
                                          : uint32_t(Fp8Scale::kExponentBias - exponent) << Fp8Scale::kMantissaBits;
              float    inv          = __uint_as_float(inverse_bits);
              uint64_t row          = uint64_t(desc.pool_block_idx) * kActivationRows + m;
              l2_acts_sf[uint64_t(desc.n_block_idx * kActivationScaleBlocks + scale_block) * kNumMaxPoolTokens + row] =
                  uint8_t(exponent + Fp8Scale::kExponentBias);

              alignas(kVectorStoreBytes) mutlass::Array<mutlass::float_e4m3_t, kScaleBlockElements> packed;
              MUTE_UNROLL
              for (int col = 0; col < kScaleBlockElements; col += kBstVectorElements) {
                ActivationBits4 raw;
                MUTE_UNROLL
                for (int j = 0; j < kBstVectorElements; ++j)
                  raw[j] = shared.activation[(scale_block * kScaleBlockElements + col + j) * kActivationRows + m].raw();
                ActivationFloat4 y =
                    __builtin_convertvector(__builtin_bit_cast(ActivationBf16x4, raw), ActivationFloat4) * inv;
                MUTE_UNROLL
                for (int j = 0; j < kBstVectorElements; ++j) y[j] = fminf(kFinfoAmaxE4M3, fmaxf(-kFinfoAmaxE4M3, y[j]));
                auto values = __musa_f2e4m3_rn_bst4(y);
                reinterpret_cast<uint32_t*>(packed.data())[col / kBstVectorElements] =
                    __builtin_bit_cast(uint32_t, values);
              }
              auto* dst =
                  reinterpret_cast<int4*>(l2_acts + row * kIntermediateHidden + desc.n_block_idx * kActivationColumns +
                                          scale_block * kScaleBlockElements);
              auto const* words = reinterpret_cast<int4 const*>(packed.data());
              dst[0]            = words[0];
              dst[1]            = words[1];
            }
            // All epilogue readers must finish before any squad starts the next tile.
            sync();
          }
        }
      }
    }
  }
};
template <class Impl>
__global__ __launch_bounds__(Impl::kNumThreads, 1) void stage1_kernel(uint32_t                            tokens,
                                                                      layout::SymBuffer<Impl::kNumRanks_> sym,
                                                                      mutlass::float_e4m3_t*              a,
                                                                      uint8_t*                            sa,
                                                                      mutlass::float_e4m3_t*              d,
                                                                      uint8_t*                            sd,
                                                                      typename Impl::Params               params,
                                                                      float                               clamp,
                                                                      float                               beta,
                                                                      float                               linear_beta) {
  extern __shared__ __align__(PrefillGemmConfig::kSmemBaseAlignment) char smem[];
  Impl::run(tokens, sym, a, sa, d, sd, params, clamp, beta, linear_beta, smem);
}
}  // namespace mega_moe::prefill::mp32
