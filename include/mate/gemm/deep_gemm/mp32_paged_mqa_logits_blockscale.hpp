#pragma once

#include <mutlass/float8.h>
#include <mutlass/float_subbyte.h>
#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/arch/simd_mp31.hpp>
#include <mute/atom/copy_traits_mp31.hpp>
#include <mute/atom/copy_traits_mp32_tme.hpp>
#include <mute/atom/mma_atom.hpp>
#include <mute/atom/mma_traits_mp32_sqmma.hpp>
#include <mute/tensor.hpp>
#include <mute/tensor_zip.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
#include <mutlass/pipeline/pipeline.hpp>

#include "mp32_mqa_logits_blockscale_utils.hpp"
#include "mp32_paged_mqa_logits_blockscale_scheduler.hpp"

namespace mate::deep_gemm {

using namespace mute;

template <class ElementQKV_,
          class ElementW_,
          class ElementLogits_,
          uint32_t kNextN_,
          uint32_t kNumHeads_,
          uint32_t kHeaddim_,
          uint32_t kTileQH_,
          uint32_t kPageSize_,
          uint32_t kTileKV_,
          bool     kIsContextLens2D_,
          bool     kIsVarlen_,
          class StrideQ_,
          class StrideKV_,
          class StrideSFQ_,
          class StrideSFKV_,
          class StrideW_,
          class StridePageTable_,
          class StrideLogits_,
          uint32_t kStagesQ_           = 1,
          uint32_t kStagesKV_          = 1,
          uint32_t kScaleStorageBytes_ = 4,
          bool     IsQScaleMajorM_     = false,
          bool     IsKVScaleMajorM_    = false>
struct MP32PagedMQALogitsBlockScale {
  using ElementQKV         = ElementQKV_;
  using ElementAccumulator = float;
  using ElementSF          = mutlass::float_ue8m0_t;
  using ElementW           = ElementW_;
  using ElementLogits      = ElementLogits_;

  using StrideQ         = StrideQ_;
  using StrideKV        = StrideKV_;
  using StrideSFQ       = StrideSFQ_;
  using StrideSFKV      = StrideSFKV_;
  using StrideW         = StrideW_;
  using StridePageTable = StridePageTable_;
  using StrideLogits    = StrideLogits_;

  static constexpr int  kNextN                 = int(kNextN_);
  static constexpr int  kNumHeads              = int(kNumHeads_);
  static constexpr int  kHeaddim               = int(kHeaddim_);
  static constexpr int  kPageSize              = int(kPageSize_);
  static constexpr bool kIsContextLens2D       = kIsContextLens2D_;
  static constexpr bool kIsVarlen              = kIsVarlen_;
  static constexpr int  kStagesQ               = int(kStagesQ_);
  static constexpr int  kStagesKV              = int(kStagesKV_);
  static constexpr int  kMmaKV                 = 128;
  static constexpr int  kTileKV                = int(kTileKV_);
  static constexpr int  kNumConsumerWarpSquads = kTileKV / kMmaKV;
  static constexpr int  kNumConsumerWarps      = kNumConsumerWarpSquads * mutlass::NumWarpsPerWarpSquad;
  static constexpr int  kTileQH                = int(kTileQH_);
  static constexpr int  kNumProducerWarpSquads = 1;
  // One producer warp per consumer squad, all living in the producer warp squad.
  static constexpr int kNumValidProducerWarps = kNumConsumerWarpSquads;
  static_assert(kNumValidProducerWarps == mutlass::NumWarpsPerWarpSquad,
                "MP32 paged MQA logits expects one producer warp per consumer squad");
  static constexpr int  QuantBlockTile     = 32;
  static constexpr int  kElementQKVBits    = sizeof_bits_v<ElementQKV>;
  static constexpr bool kIsMXFP8           = is_same_v<ElementQKV, mutlass::float_e4m3_t>;
  static constexpr bool kIsMXFP4           = is_same_v<ElementQKV, mutlass::float_e2m1_t>;
  static constexpr int  kMmaHeaddim        = kIsMXFP4 ? ceil_div(kHeaddim, 64) * 64 : kHeaddim;
  static constexpr int  kValidScaleBytes   = kHeaddim / QuantBlockTile;
  static constexpr int  kNumMmaScaleGroups = kMmaHeaddim / QuantBlockTile;
  static constexpr int  kScaleStorageBytes = int(kScaleStorageBytes_);
  static constexpr bool IsQScaleMajorM     = IsQScaleMajorM_;
  static constexpr bool IsKVScaleMajorM    = IsKVScaleMajorM_;

  using TmeScaleWord =
      conditional_t<kScaleStorageBytes == 1, uint8_t, conditional_t<kScaleStorageBytes == 2, uint16_t, uint32_t>>;

  static_assert(kStagesQ > 0 && kStagesKV > 0, "MP32 paged blockscale MQA logits requires pipeline stages");
  static_assert(!kIsVarlen || kNextN == 1, "varlen paged MQA logits requires next_n=1");
  static_assert(kNumHeads > 0 && kTileQH % kNumHeads == 0,
                "MP32 paged blockscale MQA logits requires heads to divide TileQH");
  static_assert(kTileKV >= kMmaKV && kTileKV % kMmaKV == 0,
                "MP32 paged blockscale MQA logits requires TileKV to be a positive multiple of MMA_KV");
  static_assert(kNumConsumerWarpSquads <= mutlass::NumWarpsPerWarpSquad,
                "MP32 paged blockscale MQA logits supports at most four consumer warpsquads");
  static_assert(kPageSize > 0 && kMmaKV % kPageSize == 0,
                "MP32 paged blockscale MQA logits requires pages to divide MMA_KV");
  static_assert(kIsMXFP8 || kIsMXFP4, "MP32 paged blockscale MQA logits requires MXFP8 or MXFP4 operands");
  static_assert(kHeaddim > 0, "MP32 paged blockscale MQA logits requires a positive head dimension");
  static_assert(kHeaddim % QuantBlockTile == 0,
                "MP32 paged blockscale MQA logits requires head dimension divisible by 32");
  static_assert(!kIsMXFP4 || kHeaddim <= 128, "MP32 paged MXFP4 MQA logits requires head dimension up to 128");
  static_assert(kScaleStorageBytes == 1 || kScaleStorageBytes == 2 || kScaleStorageBytes == 4,
                "MP32 paged blockscale MQA logits scale storage must use 1, 2, or 4 bytes per row");
  static_assert(kValidScaleBytes <= kScaleStorageBytes,
                "MP32 paged blockscale MQA logits scale storage must cover all valid scale groups");
  static_assert(kNumMmaScaleGroups <= kScaleStorageBytes,
                "MP32 paged blockscale MQA logits scale storage must cover all MMA scale groups");

  static constexpr int kTileQ             = kTileQH / kNumHeads;
  static constexpr int kNumPagesPerTileKV = kTileKV / kPageSize;
  static constexpr int kPagesPerMMA       = kMmaKV / kPageSize;

  static constexpr uint32_t kDispatchStop   = 0xFFFFFFFFu;
  static constexpr int      kDispatchStages = 2;

  // Per-task message broadcast by the producer warp to the consumer squads.
  struct DispatchSlot {
    uint4 head;  // {num_tile_kv, token_q_base, tile_kv_base, num_token_q}
  };

  using Scheduler = conditional_t<kIsVarlen,
                                  MP32VarlenPagedMQALogitsScheduler<kTileQ, kTileKV, kPageSize>,
                                  MP32PagedMQALogitsScheduler<kNextN, kTileQ, kTileKV, kPageSize, kIsContextLens2D>>;

  using MMATileShape = Shape<Int<kMmaKV>, Int<kTileQH>, Int<kMmaHeaddim>>;

  using ElementPair = conditional_t<kIsMXFP4, mutlass::mx_float4_t<ElementQKV>, mutlass::mx_float8_t<ElementQKV>>;
  static constexpr int kAlignmentQKV = 32 / kElementQKVBits;
  using CollectiveMetadata           = typename mutlass::gemm::collective::CollectiveBuilder<
                mutlass::arch::Mp32,
                mutlass::arch::OpClassBlockScaledTensorOp,
                ElementPair,
                mutlass::layout::RowMajor,
                kAlignmentQKV,
                ElementPair,
                mutlass::layout::ColumnMajor,
                kAlignmentQKV,
                ElementAccumulator,
                MMATileShape,
                Shape<_1, _1, _1>,
                mutlass::gemm::collective::StageCount<2>,
                mutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using TiledMma         = typename CollectiveMetadata::TiledMma;
  using SmemCopyAtomSFKV = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;
  using SmemCopyAtomSFQ  = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;
  using SmemCopyAtomW    = Copy_Atom<DefaultCopy, ElementW>;

  static constexpr int kReductionTarget = size(mqa_reduction_target_n(TiledMma{}));
  static constexpr int kHeadsPerThread  = kNumHeads / kReductionTarget;
  static constexpr int kNumThreadsWS2R  = mutlass::NumThreadsPerWarpSquad;

  using SmemLayoutKV             = decltype(tile_to_shape(typename CollectiveMetadata::SmemLayoutAtomA{},
                                              Shape<Int<kTileKV>, Int<kMmaHeaddim>, Int<kStagesKV>>{}));
  using SmemLayoutKVLoad         = decltype(mute::detail::get_nonswizzle_portion(SmemLayoutKV{}));
  using SmemLayoutQ              = decltype(tile_to_shape(typename CollectiveMetadata::SmemLayoutAtomB{},
                                             Shape<Int<kTileQH>, Int<kMmaHeaddim>, Int<kStagesQ>>{}));
  using SmemLayoutLogits         = Layout<Shape<Int<kTileQ>, Int<kTileKV>>, Stride<Int<kTileKV>, _1>>;
  using SmemLayoutLogitsSquadRow = Layout<Shape<_1, Int<kMmaKV>>, Stride<Int<kTileKV>, _1>>;

  static constexpr int kSmemSFQRowStride   = IsQScaleMajorM ? 1 : kScaleStorageBytes;
  static constexpr int kSmemSFQStageSize   = kTileQH * kScaleStorageBytes;
  static constexpr int kSmemSFQGroupStride = IsQScaleMajorM ? kTileQH : 1;
  using LoadLayoutSFQ                      = Layout<Shape<Int<kTileQH>, Int<kScaleStorageBytes>, Int<kStagesQ>>,
                                                    Stride<Int<kSmemSFQRowStride>, Int<kSmemSFQGroupStride>, Int<kSmemSFQStageSize>>>;
  using LoadLayoutSFQStage                 = decltype(take<0, 2>(LoadLayoutSFQ{}));
  using MMALayoutSFQ =
      Layout<Shape<Int<kTileQH>, Shape<Int<QuantBlockTile>, Int<kNumMmaScaleGroups>>, Int<kStagesQ>>,
             Stride<Int<kSmemSFQRowStride>, Stride<_0, Int<kSmemSFQGroupStride>>, Int<kSmemSFQStageSize>>>;

  static constexpr int kSmemSFKVRowStride   = IsKVScaleMajorM ? 1 : kScaleStorageBytes;
  static constexpr int kSmemSFKVPageSize    = kPageSize * kScaleStorageBytes;
  static constexpr int kSmemSFKVStageSize   = kTileKV * kScaleStorageBytes;
  static constexpr int kSmemSFKVGroupStride = IsKVScaleMajorM ? kPageSize : 1;
  using LoadLayoutSFKVPage                  = Layout<Shape<Int<kPageSize>, Int<kScaleStorageBytes>>,
                                                     Stride<Int<kSmemSFKVRowStride>, Int<kSmemSFKVGroupStride>>>;
  using LoadLayoutSFKV                      = Layout<
                           Shape<Int<kPageSize>, Int<kScaleStorageBytes>, Int<kNumPagesPerTileKV>, Int<kStagesKV>>,
                           Stride<Int<kSmemSFKVRowStride>, Int<kSmemSFKVGroupStride>, Int<kSmemSFKVPageSize>, Int<kSmemSFKVStageSize>>>;
  using LoadLayoutSFKVStage = decltype(take<0, 3>(LoadLayoutSFKV{}));
  using MMALayoutSFKV =
      Layout<Shape<Shape<Shape<Int<kPageSize>, Int<kPagesPerMMA>>, Int<kNumConsumerWarpSquads>>,
                   Shape<Int<QuantBlockTile>, Int<kNumMmaScaleGroups>>,
                   Int<kStagesKV>>,
             Stride<Stride<Stride<Int<kSmemSFKVRowStride>, Int<kSmemSFKVPageSize>>, Int<kMmaKV * kScaleStorageBytes>>,
                    Stride<_0, Int<kSmemSFKVGroupStride>>,
                    Int<kSmemSFKVStageSize>>>;

  using SmemLayoutKVStage =
      decltype(tile_to_shape(typename CollectiveMetadata::SmemLayoutAtomA{}, Shape<Int<kMmaKV>, Int<kMmaHeaddim>>{}));
  using SmemLayoutQStage = decltype(take<0, 2>(SmemLayoutQ{}));
  using SmemLayoutW      = Layout<Shape<Int<kTileQH>, Int<kStagesQ>>, Stride<_1, Int<kTileQH>>>;
  using SmemLayoutWStage = decltype(take<0, 1>(SmemLayoutW{}));
  using SmemLayoutWS2R   = Layout<Shape<Shape<Int<kReductionTarget>, Int<kNumThreadsWS2R / kReductionTarget>>,
                                        Int<kTileQ * kHeadsPerThread * kStagesQ>>,
                                  Stride<Stride<_1, _0>, Int<kReductionTarget>>>;
  using SmemCopyThreadLayoutWS2R =
      Layout<Shape<Shape<Int<kReductionTarget>, Int<kNumThreadsWS2R / kReductionTarget>>, _1>,
             Stride<Stride<_1, Int<kReductionTarget>>, _0>>;
  using SmemCopyValueLayoutWS2R = Layout<Shape<_1, Int<kTileQ * kHeadsPerThread>>, Stride<_0, _1>>;

  using TmeQKVInternalType = conditional_t<(sizeof_bits_v<ElementQKV> == 4), mutlass::detail::float_e2m1x2_t, void>;

  using KVPageShape = Shape<Int<kPageSize>, Int<kMmaHeaddim>>;
  using SmemLayoutKVPage =
      decltype(MP32::SQMMA::
                   make_canonical_gemm_smem_atom_layout<ElementQKV, TCE::Major::K, Int<kPageSize>, Int<kMmaHeaddim>>());

  using TME_Q      = decltype(make_tme_copy<TmeQKVInternalType>(
      MP31_TME_LOAD{},
      make_tensor(
          make_gmem_ptr(static_cast<ElementQKV const*>(nullptr)), repeat_like(StrideQ{}, int32_t(0)), StrideQ{}),
      SmemLayoutQStage{},
      select<1, 2>(MMATileShape{})));
  using TME_W      = decltype(make_tme_copy(MP31_TME_LOAD{},
                                       make_tensor(make_gmem_ptr(static_cast<ElementW const*>(nullptr)),
                                                   make_layout(make_shape(int32_t(0)), make_stride(get<0>(StrideW{})))),
                                       SmemLayoutWStage{},
                                       shape(SmemLayoutWStage{})));
  using TME_SFQ    = decltype(make_tme_copy<TmeScaleWord>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementSF const*>(nullptr)),
                  make_layout(make_shape(int32_t(0), Int<kScaleStorageBytes>{}), StrideSFQ{})),
      LoadLayoutSFQStage{},
      shape(LoadLayoutSFQStage{})));
  using TME_KV     = decltype(make_tme_copy<TmeQKVInternalType>(
      MP31_TME_LOAD{},
      make_tensor(
          make_gmem_ptr(static_cast<ElementQKV const*>(nullptr)), repeat_like(StrideKV{}, int32_t(0)), StrideKV{}),
      SmemLayoutKVPage{},
      KVPageShape{}));
  using TME_SFKV   = decltype(make_tme_copy<TmeScaleWord>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementSF const*>(nullptr)),
                  make_layout(make_shape(Int<kPageSize>{}, Int<kScaleStorageBytes>{}, int32_t(0)), StrideSFKV{})),
      LoadLayoutSFKVPage{},
      shape(LoadLayoutSFKVPage{})));
  using TME_Logits = decltype(make_tme_copy(
      MP31_TME_STORE{},
      make_tensor(
          make_gmem_ptr(static_cast<ElementLogits*>(nullptr)), repeat_like(StrideLogits{}, int32_t(0)), StrideLogits{}),
      SmemLayoutLogitsSquadRow{},
      shape(SmemLayoutLogitsSquadRow{})));

  using PipelineQ              = mutlass::PipelineTmeAsync<kStagesQ>;
  using PipelineKV             = mutlass::PipelineTmeAsync<kStagesKV>;
  using PipelineDispatch       = mutlass::PipelineAsync<kDispatchStages>;
  using PipelineQParams        = typename PipelineQ::Params;
  using PipelineKVParams       = typename PipelineKV::Params;
  using PipelineDispatchParams = typename PipelineDispatch::Params;
  using PipelineQState         = typename PipelineQ::PipelineState;
  using PipelineKVState        = typename PipelineKV::PipelineState;
  using PipelineDispatchState  = typename PipelineDispatch::PipelineState;

  static constexpr int kTmeTransactionBytesQ =
      ceil_div(int(cosize(SmemLayoutQStage{})) * sizeof_bits_v<ElementQKV>, 8) +
      ceil_div(int(cosize(SmemLayoutWStage{})) * sizeof_bits_v<ElementW>, 8) + int(cosize(LoadLayoutSFQStage{}));
  static constexpr int kTmeTransactionBytesKV =
      kNumConsumerWarpSquads * ceil_div(int(cosize(SmemLayoutKVStage{})) * sizeof_bits_v<ElementQKV>, 8) +
      int(cosize(LoadLayoutSFKVStage{}));
  // One producer warp feeds one consumer squad: kPagesPerMMA KV pages and their scales.
  static_assert(kTmeTransactionBytesKV % kNumConsumerWarpSquads == 0,
                "MP32 paged MQA logits KV transaction bytes must split evenly across consumer squads");
  static constexpr int kTmeTransactionBytesKVPerSquad = kTmeTransactionBytesKV / kNumConsumerWarpSquads;

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t PipelineQ[PipelineQ::NumBarriers];
    uint8_t PipelineKV[PipelineKV::NumBarriers * kNumConsumerWarpSquads];
    uint8_t SquadLogits[kNumConsumerWarpSquads];
    uint8_t PipelineDispatch[PipelineDispatch::NumBarriers];
  };

  static constexpr int kSmemAlignment = 256;

  struct SharedStorage {
    alignas(kSmemAlignment) ArrayEngine<ElementQKV, cosize_v<SmemLayoutQ>> smem_q;
    alignas(kSmemAlignment) ArrayEngine<ElementQKV, cosize_v<SmemLayoutKV>> smem_kv;
    alignas(kSmemAlignment) ArrayEngine<ElementW, cosize_v<SmemLayoutW>> smem_weights;
    alignas(kSmemAlignment) ArrayEngine<ElementSF, cosize_v<LoadLayoutSFKV>> smem_sfkv;
    alignas(kSmemAlignment) ArrayEngine<ElementSF, cosize_v<LoadLayoutSFQ>> smem_sfq;
    alignas(kSmemAlignment) ArrayEngine<ElementLogits, cosize_v<SmemLayoutLogits>> smem_logits;
    alignas(kSmemAlignment) ArrayEngine<DispatchSlot, kDispatchStages> smem_dispatch;
  };

  static constexpr int      SharedStorageSize  = sizeof(SharedStorage);
  static constexpr int      SmemAlignmentBytes = kSmemAlignment;
  static constexpr uint32_t MaxThreadsPerBlock =
      (kNumProducerWarpSquads + kNumConsumerWarpSquads) * mutlass::NumThreadsPerWarpSquad;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  struct Arguments {
    uint32_t total_num_token_q;
    uint32_t total_qh;
    uint32_t num_kv_blocks;

    ElementQKV const* ptr_q;
    StrideQ           stride_q;
    ElementQKV const* ptr_kv;
    StrideKV          stride_kv;
    ElementSF const*  ptr_sfq;
    StrideSFQ         stride_sfq;
    ElementSF const*  ptr_sfkv;
    StrideSFKV        stride_sfkv;
    ElementW const*   ptr_weights;
    StrideW           stride_weights;

    uint32_t const* context_lens;
    uint32_t const* metadata;
    uint32_t const* indices;
    uint32_t const* page_table;
    StridePageTable stride_page_table;
    uint64_t        page_table_num_elements;

    ElementLogits* ptr_logits;
    StrideLogits   stride_logits;
    uint32_t       max_seqlen_kv;
  };

  struct Params {
    uint32_t total_num_token_q;
    uint32_t total_qh;
    uint32_t num_kv_blocks;

    TME_Q      tme_q;
    TME_W      tme_w;
    TME_SFQ    tme_sfq;
    TME_KV     tme_kv;
    TME_SFKV   tme_sfkv;
    TME_Logits tme_logits;

    uint32_t const*  context_lens;
    uint32_t const*  metadata;
    uint32_t const*  indices;
    uint32_t const*  page_table;
    StridePageTable  stride_page_table;
    RobustDescriptor desc_page_table;

    ElementLogits* ptr_logits;
    StrideLogits   stride_logits;
    uint32_t       max_seqlen_kv;
  };

  MUTLASS_DEVICE static Scheduler make_scheduler(Params const& params) {
    if constexpr (kIsVarlen) {
      return Scheduler(uint32_t(blockIdx.x),
                       params.context_lens,
                       params.metadata,
                       params.indices,
                       uint32_t(params.total_num_token_q));
    } else {
      return Scheduler(
          uint32_t(blockIdx.x), params.context_lens, params.metadata, uint32_t(params.total_num_token_q) / kNextN);
    }
  }

  static Params to_underlying_arguments(Arguments const& args) {
    TME_Q tme_q = make_tme_copy<TmeQKVInternalType>(
        MP31_TME_LOAD{},
        make_tensor(make_gmem_ptr(args.ptr_q), make_shape(args.total_qh, kHeaddim), args.stride_q),
        SmemLayoutQStage{},
        select<1, 2>(MMATileShape{}));
    TME_W tme_w =
        make_tme_copy(MP31_TME_LOAD{},
                      make_tensor(make_gmem_ptr(args.ptr_weights),
                                  make_layout(make_shape(args.total_qh), make_stride(get<0>(args.stride_weights)))),
                      SmemLayoutWStage{},
                      shape(SmemLayoutWStage{}));
    TME_SFQ tme_sfq = make_tme_copy<TmeScaleWord>(
        MP31_TME_LOAD{},
        make_tensor(make_gmem_ptr(args.ptr_sfq), make_shape(args.total_qh, Int<kScaleStorageBytes>{}), args.stride_sfq),
        LoadLayoutSFQStage{},
        shape(LoadLayoutSFQStage{}));

    // Paged descriptors retain the physical-page mode because fused blocks append scales after values.
    TME_KV tme_kv =
        make_tme_copy<TmeQKVInternalType>(MP31_TME_LOAD{},
                                          make_tensor(make_gmem_ptr(args.ptr_kv),
                                                      make_shape(Int<kPageSize>{}, Int<kHeaddim>{}, args.num_kv_blocks),
                                                      args.stride_kv),
                                          SmemLayoutKVPage{},
                                          KVPageShape{});
    TME_SFKV tme_sfkv = make_tme_copy<TmeScaleWord>(
        MP31_TME_LOAD{},
        make_tensor(make_gmem_ptr(args.ptr_sfkv),
                    make_shape(Int<kPageSize>{}, Int<kScaleStorageBytes>{}, args.num_kv_blocks),
                    args.stride_sfkv),
        LoadLayoutSFKVPage{},
        shape(LoadLayoutSFKVPage{}));
    auto mLogits = make_tensor(
        make_gmem_ptr(args.ptr_logits), make_shape(args.total_num_token_q, args.max_seqlen_kv), args.stride_logits);
    TME_Logits tme_logits =
        make_tme_copy(MP31_TME_STORE{}, mLogits, SmemLayoutLogitsSquadRow{}, shape(SmemLayoutLogitsSquadRow{}));

    RobustDescriptor desc_page_table =
        make_robust_desc(args.page_table, static_cast<size_t>(args.page_table_num_elements));

    return Params{
        .total_num_token_q = args.total_num_token_q,
        .total_qh          = args.total_qh,
        .num_kv_blocks     = args.num_kv_blocks,
        .tme_q             = tme_q,
        .tme_w             = tme_w,
        .tme_sfq           = tme_sfq,
        .tme_kv            = tme_kv,
        .tme_sfkv          = tme_sfkv,
        .tme_logits        = tme_logits,
        .context_lens      = args.context_lens,
        .metadata          = args.metadata,
        .indices           = args.indices,
        .page_table        = args.page_table,
        .stride_page_table = args.stride_page_table,
        .desc_page_table   = desc_page_table,
        .ptr_logits        = args.ptr_logits,
        .stride_logits     = args.stride_logits,
        .max_seqlen_kv     = args.max_seqlen_kv,
    };
  }

  // Per-squad epilogue: write the reduced logits slice, then let the squad's first warp store it.
  template <class FinalSum>
  MUTLASS_DEVICE void epilogue(int                          consumer_idx,
                               int                          thread_idx_in_warpsquad,
                               Params const&                params,
                               SharedStorage&               shared_storage,
                               mutlass::arch::AsyncBarrier& logits_barrier,
                               uint32_t                     token_q_base,
                               uint32_t                     num_token_q,
                               int32_t                      kv_tile_base,
                               FinalSum const&              final_sum) {
    bool const is_store_warp = thread_idx_in_warpsquad < mutlass::NumThreadsPerWarp;

    // Wait for this squad's previous logits store before the slice is written again.
    tme_store_wait();

    if (thread_idx_in_warpsquad % kReductionTarget == 0) {
      TiledMma  tiled_mma;
      auto      thr_mma  = tiled_mma.get_thread_slice(thread_idx_in_warpsquad);
      Tensor    cMN      = make_identity_tensor(take<0, 2>(MMATileShape{}));
      Tensor    tCcMN    = thr_mma.partition_C(cMN);
      Tensor    tCcMN_mn = make_tensor(tCcMN.data(), mqa_layout_acc_mn(tiled_mma, tCcMN.layout()));
      int const kv_base  = consumer_idx * kMmaKV;

      Tensor sLogits = make_tensor(make_smem_ptr(shared_storage.smem_logits.begin()), SmemLayoutLogits{});
      MUTE_UNROLL
      for (int q = 0; q < kTileQ; ++q) {
        MUTE_UNROLL
        for (int m = 0; m < size<1>(final_sum); ++m) {
          int const kv_local   = kv_base + int(get<0>(tCcMN_mn(m, _0{})));
          sLogits(q, kv_local) = static_cast<ElementLogits>(final_sum(q, m));
        }
      }
    }

    __syncwarp();

    // This squad's slice is complete: store it while the squad moves on to the next tile.
    logits_barrier.sync();

    if (is_store_warp) {
      auto cta_tme_logits = params.tme_logits.get_slice(0);
      auto mLogits = params.tme_logits.get_tme_tensor(make_shape(params.total_num_token_q, params.max_seqlen_kv));
      auto sLogits = make_tensor(make_smem_ptr(shared_storage.smem_logits.begin()), SmemLayoutLogits{});

      for (uint32_t q = 0; q < num_token_q; ++q) {
        auto sLogitsSquadRow   = local_tile(sLogits, shape(SmemLayoutLogitsSquadRow{}), make_coord(q, consumer_idx));
        auto tSsLogitsSquadRow = cta_tme_logits.partition_S(sLogitsSquadRow);
        auto mLogitsBase =
            domain_offset(make_coord(int32_t(token_q_base + q), kv_tile_base + consumer_idx * kMmaKV), mLogits);
        auto gLogitsSquadRow   = local_tile(mLogitsBase, shape(SmemLayoutLogitsSquadRow{}), make_coord(_0{}, _0{}));
        auto tSgLogitsSquadRow = cta_tme_logits.partition_D(gLogitsSquadRow);
        copy(params.tme_logits, tSsLogitsSquadRow, tSgLogitsSquadRow);
        tme_store_arrive();
      }
    }
  }

  // Stages the KV pages (values + scales) of one consumer squad for one task.
  MUTLASS_DEVICE void load_kv_tile(Params const&                   params,
                                   SharedStorage&                  shared_storage,
                                   typename Scheduler::Task const& task,
                                   int                             producer_idx,
                                   PipelineKV&                     pipeline_kv,
                                   PipelineKVState&                pipeline_kv_write) {
    auto cta_tme_kv   = params.tme_kv.get_slice(0);
    auto cta_tme_sfkv = params.tme_sfkv.get_slice(0);

    auto mKV     = params.tme_kv.get_tme_tensor(make_shape(Int<kPageSize>{}, Int<kHeaddim>{}, params.num_kv_blocks));
    auto gKVFull = local_tile(mKV, KVPageShape{}, make_coord(_, _, _));
    auto tKVgKV  = cta_tme_kv.partition_S(gKVFull(_, _, _0{}, _0{}, _));
    auto sKVLoad = make_tensor(make_smem_ptr(shared_storage.smem_kv.begin()), SmemLayoutKVLoad{});

    auto mSFKV =
        params.tme_sfkv.get_tme_tensor(make_shape(Int<kPageSize>{}, Int<kScaleStorageBytes>{}, params.num_kv_blocks));
    auto gSFKVFull  = local_tile(mSFKV, shape(LoadLayoutSFKVPage{}), make_coord(_, _, _));
    auto tSFKVgSFKV = cta_tme_sfkv.partition_S(gSFKVFull(_, _, _0{}, _0{}, _));
    auto sSFKV      = make_tensor(make_smem_ptr(shared_storage.smem_sfkv.begin()), LoadLayoutSFKV{});
    auto tSFKVsSFKV = cta_tme_sfkv.partition_D(sSFKV);

    uint32_t const        num_tile_kv  = task.num_tile_kv;
    uint32_t const        tile_kv_base = task.tile_kv_base;
    uint32_t const        num_page_kv  = task.num_page_kv;
    uint32_t const        first_page   = uint32_t(kPagesPerMMA * producer_idx);
    uint32_t const* const page_row = params.page_table + uint64_t(task.request_id) * uint64_t(params.stride_page_table);

    for (uint32_t tile_kv_idx = 0; tile_kv_idx < num_tile_kv; ++tile_kv_idx) {
      pipeline_kv.producer_acquire(pipeline_kv_write);
      uint32_t const kv_barrier_id = pipeline_kv.producer_get_barrier_id(pipeline_kv_write);
      int const      kv_stage      = pipeline_kv_write.index();

      uint32_t const tile_kv_idx_in_request = tile_kv_base + tile_kv_idx;
      uint32_t const tile_page_base         = tile_kv_idx_in_request * kNumPagesPerTileKV;

      // This producer warp owns pages [first_page, first_page + kPagesPerMMA).
      uint32_t page_coords[kPagesPerMMA];
      MUTE_UNROLL
      for (int page = 0; page < kPagesPerMMA; ++page) {
        uint32_t const page_offset = tile_page_base + first_page + uint32_t(page);
        uint32_t       coord       = 0;
        MP31_ROBUST_LOAD<uint32_t>::copy(page_row[page_offset], coord, true, params.desc_page_table);
        if (page_offset >= num_page_kv) {
          coord = 0;
        }
        page_coords[page] = coord;
      }

      MUTE_UNROLL
      for (int page = 0; page < kPagesPerMMA; ++page) {
        uint32_t const page_idx_in_tile_kv = first_page + uint32_t(page);
        uint32_t const page_coord          = page_coords[page];

        auto sKVPageLoad = local_tile(sKVLoad, KVPageShape{}, make_coord(page_idx_in_tile_kv, _0{}, kv_stage));
        auto tKVsKVPage  = cta_tme_kv.partition_D(sKVPageLoad);
        auto sKVPageDst  = make_tensor(sKVPageLoad.data(), tKVsKVPage.layout());
        copy(params.tme_kv.with(kv_barrier_id), tKVgKV(_, _, _, page_coord), sKVPageDst);
        copy(params.tme_sfkv.with(kv_barrier_id),
             tSFKVgSFKV(_, _, _, page_coord),
             tSFKVsSFKV(_, _, _, page_idx_in_tile_kv, kv_stage));
      }

      ++pipeline_kv_write;
    }
  }

  // Producer warp 0: owns the scheduler, publishes dispatch records, stages Q, weights, and Q
  // scales, and stages the KV pages (and scales) of consumer squad 0.
  MUTLASS_DEVICE void load0(Params const&          params,
                            SharedStorage&         shared_storage,
                            PipelineQ&             pipeline_q,
                            PipelineQState&        pipeline_q_write,
                            PipelineKV&            pipeline_kv,
                            PipelineKVState&       pipeline_kv_write,
                            PipelineDispatch&      pipeline_dispatch,
                            PipelineDispatchState& pipeline_dispatch_write) {
    auto cta_tme_w   = params.tme_w.get_slice(0);
    auto cta_tme_q   = params.tme_q.get_slice(0);
    auto cta_tme_sfq = params.tme_sfq.get_slice(0);

    auto mW   = params.tme_w.get_tme_tensor(make_shape(params.total_qh));
    auto sW   = make_tensor(make_smem_ptr(shared_storage.smem_weights.begin()), SmemLayoutW{});
    auto tWsW = cta_tme_w.partition_D(sW);

    auto mQ   = params.tme_q.get_tme_tensor(make_shape(params.total_qh, kHeaddim));
    auto sQ   = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()), SmemLayoutQ{});
    auto tQsQ = cta_tme_q.partition_D(sQ);

    auto mSFQ     = params.tme_sfq.get_tme_tensor(make_shape(params.total_qh, Int<kScaleStorageBytes>{}));
    auto sSFQ     = make_tensor(make_smem_ptr(shared_storage.smem_sfq.begin()), LoadLayoutSFQ{});
    auto tSFQsSFQ = cta_tme_sfq.partition_D(sSFQ);

    uint32_t const           lane_idx  = mutlass::canonical_lane_idx();
    Scheduler                scheduler = make_scheduler(params);
    typename Scheduler::Task task{};

    while (scheduler.get_next_task(task)) {
      __musa_loop_transparent_outermost();
      pipeline_dispatch.producer_acquire(pipeline_dispatch_write);

      if (lane_idx == 0) {
        DispatchSlot& slot = shared_storage.smem_dispatch.begin()[pipeline_dispatch_write.index()];
        slot.head          = uint4{task.num_tile_kv, task.token_q_base, task.tile_kv_base, task.num_token_q};
      }

      __syncwarp();
      pipeline_dispatch.producer_commit(pipeline_dispatch_write);
      ++pipeline_dispatch_write;

      pipeline_q.producer_acquire(pipeline_q_write);
      uint32_t const q_barrier_id = pipeline_q.producer_get_barrier_id(pipeline_q_write);
      int const      q_stage      = pipeline_q_write.index();
      int32_t const  qh_base      = int32_t(task.token_q_base) * kNumHeads;

      auto mWBase = domain_offset(make_coord(qh_base), mW);
      auto gW     = local_tile(mWBase, shape(SmemLayoutWStage{}), make_coord(_));
      auto tWgW   = cta_tme_w.partition_S(gW);
      copy(params.tme_w.with(q_barrier_id), tWgW(_, _, _0{}), tWsW(_, _, q_stage));

      auto mQBase = domain_offset(make_coord(qh_base, _0{}), mQ);
      auto gQ     = local_tile(mQBase, select<1, 2>(MMATileShape{}), make_coord(_, _0{}));
      auto tQgQ   = cta_tme_q.partition_S(gQ);
      copy(params.tme_q.with(q_barrier_id), tQgQ(_, _, _, _0{}), tQsQ(_, _, _, q_stage));

      auto mSFQBase = domain_offset(make_coord(qh_base, _0{}), mSFQ);
      auto gSFQ     = local_tile(mSFQBase, shape(LoadLayoutSFQStage{}), make_coord(_, _0{}));
      auto tSFQgSFQ = cta_tme_sfq.partition_S(gSFQ);
      copy(params.tme_sfq.with(q_barrier_id), tSFQgSFQ(_, _, _, _0{}), tSFQsSFQ(_, _, _, q_stage));

      ++pipeline_q_write;

      load_kv_tile(params, shared_storage, task, 0, pipeline_kv, pipeline_kv_write);
    }

    // Stop record for the consumers.
    pipeline_dispatch.producer_acquire(pipeline_dispatch_write);

    if (lane_idx == 0) {
      DispatchSlot& slot = shared_storage.smem_dispatch.begin()[pipeline_dispatch_write.index()];
      slot.head          = uint4{kDispatchStop, 0u, 0u, 0u};
    }

    __syncwarp();
    pipeline_dispatch.producer_commit(pipeline_dispatch_write);
  }

  // Producer warps 1..3: stage the KV pages (and scales) of their consumer squad.
  MUTLASS_DEVICE void load(Params const&    params,
                           SharedStorage&   shared_storage,
                           int              producer_idx,
                           PipelineKV&      pipeline_kv,
                           PipelineKVState& pipeline_kv_write) {
    Scheduler                scheduler = make_scheduler(params);
    typename Scheduler::Task task{};

    while (scheduler.get_next_task(task)) {
      __musa_loop_transparent_outermost();
      load_kv_tile(params, shared_storage, task, producer_idx, pipeline_kv, pipeline_kv_write);
    }
  }

  // Regular compute role: consumes dispatch records, runs SQMMA per KV tile, and feeds logits.
  MUTLASS_DEVICE void compute(Params const&                params,
                              SharedStorage&               shared_storage,
                              int                          consumer_idx,
                              PipelineQ&                   pipeline_q,
                              PipelineQState&              pipeline_q_read,
                              PipelineKV&                  pipeline_kv,
                              PipelineKVState&             pipeline_kv_read,
                              PipelineDispatch&            pipeline_dispatch,
                              PipelineDispatchState&       pipeline_dispatch_read,
                              mutlass::arch::AsyncBarrier& logits_barrier) {
    int const thread_idx_in_warpsquad = int(threadIdx.x) % mutlass::NumThreadsPerWarpSquad;

    TiledMma tiled_mma;
    auto     thr_mma = tiled_mma.get_thread_slice(thread_idx_in_warpsquad);

    Tensor sKVAll = make_tensor(make_smem_ptr(shared_storage.smem_kv.begin()), SmemLayoutKV{});
    Tensor sKV    = sKVAll(make_coord(_, consumer_idx), _, _);
    Tensor sQ     = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()), SmemLayoutQ{});
    Tensor sW     = make_tensor(make_smem_ptr(shared_storage.smem_weights.begin()), SmemLayoutWS2R{});
    Tensor sSFKVAll =
        make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(shared_storage.smem_sfkv.begin())), MMALayoutSFKV{});
    Tensor sSFKV = sSFKVAll(make_coord(_, consumer_idx), _, _);
    Tensor sSFQ =
        make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(shared_storage.smem_sfq.begin())), MMALayoutSFQ{});

    Tensor tCsKV   = thr_mma.partition_A(sKV);
    Tensor tCsQ    = thr_mma.partition_B(sQ);
    Tensor tCrKV   = thr_mma.make_fragment_A(tCsKV);
    Tensor tCrQ    = thr_mma.make_fragment_B(tCsQ);
    Tensor tCrSFKV = MP32::BLOCKSCALED::partition_fragment_SFA(sSFKV(_, _, Int<0>{}), thr_mma);
    Tensor tCrSFQ  = MP32::BLOCKSCALED::partition_fragment_SFB(sSFQ(_, _, Int<0>{}), thr_mma);
    Tensor accum   = partition_fragment_C(tiled_mma, take<0, 2>(MMATileShape{}));

    auto   tile_shape_mnk       = tile_shape(tiled_mma);
    auto   smem_tiled_copy_sfkv = make_tiled_copy_impl(SmemCopyAtomSFKV{},
                                                     MP32::BLOCKSCALED::get_layoutSFA_TV(tiled_mma),
                                                     make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto   smem_tiled_copy_sfq  = make_tiled_copy_impl(SmemCopyAtomSFQ{},
                                                    MP32::BLOCKSCALED::get_layoutSFB_TV(tiled_mma),
                                                    make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto   smem_thr_copy_sfkv   = smem_tiled_copy_sfkv.get_thread_slice(thread_idx_in_warpsquad);
    auto   smem_thr_copy_sfq    = smem_tiled_copy_sfq.get_thread_slice(thread_idx_in_warpsquad);
    Tensor tCsSFKV              = smem_thr_copy_sfkv.partition_S(sSFKV);
    Tensor tCsSFQ               = smem_thr_copy_sfq.partition_S(sSFQ);
    Tensor tCrSFKVCopy          = smem_thr_copy_sfkv.retile_D(tCrSFKV);
    Tensor tCrSFQCopy           = smem_thr_copy_sfq.retile_D(tCrSFQ);

    auto   smem_tiled_copy_w = make_tiled_copy(SmemCopyAtomW{}, SmemCopyThreadLayoutWS2R{}, SmemCopyValueLayoutWS2R{});
    auto   smem_thr_copy_w   = smem_tiled_copy_w.get_thread_slice(thread_idx_in_warpsquad);
    Tensor tCsW              = smem_thr_copy_w.partition_S(sW);
    Tensor tCrW              = make_fragment_like<ElementW>(tCsW(_, _, Int<0>{}));

    static_assert(size(accum) % 4 == 0, "MP32 paged MQA logits accumulator fragment must be divisible by BST4");
    static_assert(decltype(size<2>(tCrKV))::value == 1 && decltype(size<2>(tCrQ))::value == 1,
                  "MP32 paged MQA logits requires one MMA K block");

    while (true) {
      __musa_loop_transparent_outermost();
      pipeline_dispatch.consumer_wait(pipeline_dispatch_read);

      uint4 const head = shared_storage.smem_dispatch.begin()[pipeline_dispatch_read.index()].head;

      uint32_t const num_tile_kv  = head.x;
      uint32_t const token_q_base = head.y;
      uint32_t const tile_kv_base = head.z;
      uint32_t const num_token_q  = head.w;

      __syncwarp();
      pipeline_dispatch.consumer_release(pipeline_dispatch_read);
      ++pipeline_dispatch_read;

      if (num_tile_kv == kDispatchStop) {
        break;
      }

      pipeline_q.consumer_wait(pipeline_q_read);
      int const q_stage = pipeline_q_read.index();

      copy(tCsW(_, _, q_stage), tCrW);
      copy(tCsSFQ(_, _, Int<0>{}, q_stage), tCrSFQCopy(_, _, Int<0>{}));
      auto tCrWF = to_float_fragment(tCrW);

      for (uint32_t tile_kv_idx = 0; tile_kv_idx < num_tile_kv; ++tile_kv_idx) {
        pipeline_kv.consumer_wait(pipeline_kv_read);
        int const kv_stage = pipeline_kv_read.index();

        copy(tCsSFKV(_, _, Int<0>{}, kv_stage), tCrSFKVCopy(_, _, Int<0>{}));
        clear(accum);
        gemm(tiled_mma,
             make_zip_tensor(tCrKV(_, _, Int<0>{}, kv_stage), tCrSFKV(_, _, Int<0>{})),
             make_zip_tensor(tCrQ(_, _, Int<0>{}, q_stage), tCrSFQ(_, _, Int<0>{})),
             accum);
        warpsquad_commit_batch();
        warpsquad_wait();

        pipeline_kv.consumer_release(pipeline_kv_read);
        ++pipeline_kv_read;

        apply_relu(accum);
        Tensor final_sum = apply_weighted_reduce_head<kHeadsPerThread, kReductionTarget>(tiled_mma, accum, tCrWF);
        epilogue(consumer_idx,
                 thread_idx_in_warpsquad,
                 params,
                 shared_storage,
                 logits_barrier,
                 token_q_base,
                 num_token_q,
                 int32_t((tile_kv_base + tile_kv_idx) * kTileKV),
                 final_sum);
      }

      pipeline_q.consumer_release(pipeline_q_read);
      ++pipeline_q_read;
    }
  }

  MUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);

    PipelineQParams pipeline_q_params;
    pipeline_q_params.transaction_bytes = kTmeTransactionBytesQ;
    pipeline_q_params.num_consumers     = kNumConsumerWarps;
    pipeline_q_params.num_producers     = 1;

    PipelineKVParams pipeline_kv_params;
    pipeline_kv_params.transaction_bytes = kTmeTransactionBytesKVPerSquad;
    pipeline_kv_params.num_consumers     = mutlass::NumWarpsPerWarpSquad;
    pipeline_kv_params.num_producers     = 1;

    PipelineDispatchParams pipeline_dispatch_params;
    pipeline_dispatch_params.producer_arv_count = 1;
    pipeline_dispatch_params.consumer_arv_count = kNumConsumerWarps;

    int const warp_idx       = mutlass::canonical_warp_idx_sync();
    int const warp_squad_idx = mutlass::canonical_warp_squad_idx();
    int const consumer_idx   = warp_squad_idx - kNumProducerWarpSquads;

    // One barrier per consumer squad: the squad arrives after writing its logits slice.
    uint32_t const logits_barrier_base = uint32_t(reinterpret_cast<uint64_t>(&barrier_storage->SquadLogits));
    if (warp_idx == 0) {
      MUTE_UNROLL
      for (int squad = 0; squad < kNumConsumerWarpSquads; ++squad) {
        mutlass::arch::AsyncBarrier(logits_barrier_base + uint32_t(squad)).init(mutlass::NumWarpsPerWarpSquad);
      }
    }
    mutlass::arch::AsyncBarrier logits_barrier(logits_barrier_base + uint32_t(consumer_idx));

    PipelineQ        pipeline_q(pipeline_q_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineQ));
    PipelineDispatch pipeline_dispatch(pipeline_dispatch_params,
                                       reinterpret_cast<uint64_t>(&barrier_storage->PipelineDispatch));

    // One KV pipeline (values + scales) per consumer squad; every warp points at its own group.
    // Producer warps 0..kNumValidProducerWarps initialise their group's barriers (init_warps).
    int const  kv_group_idx = (warp_squad_idx < kNumProducerWarpSquads) ? int(warp_idx) : consumer_idx;
    PipelineKV pipeline_kv(pipeline_kv_params,
                           uint32_t(reinterpret_cast<uint64_t>(&barrier_storage->PipelineKV)) +
                               uint32_t(kv_group_idx) * PipelineKV::NumBarriers,
                           mutlass::NumWarpsPerWarpSquad);

    PipelineQState        pipeline_q_read;
    PipelineQState        pipeline_q_write = mutlass::make_producer_start_state<PipelineQ>();
    PipelineKVState       pipeline_kv_read;
    PipelineKVState       pipeline_kv_write = mutlass::make_producer_start_state<PipelineKV>();
    PipelineDispatchState pipeline_dispatch_read;
    PipelineDispatchState pipeline_dispatch_write = mutlass::make_producer_start_state<PipelineDispatch>();

    __syncthreads();

    if (warp_idx == 0) {
      load0(params,
            shared_storage,
            pipeline_q,
            pipeline_q_write,
            pipeline_kv,
            pipeline_kv_write,
            pipeline_dispatch,
            pipeline_dispatch_write);
    } else if (warp_idx < kNumValidProducerWarps) {
      load(params, shared_storage, int(warp_idx), pipeline_kv, pipeline_kv_write);
    } else if (warp_squad_idx >= kNumProducerWarpSquads) {
      compute(params,
              shared_storage,
              consumer_idx,
              pipeline_q,
              pipeline_q_read,
              pipeline_kv,
              pipeline_kv_read,
              pipeline_dispatch,
              pipeline_dispatch_read,
              logits_barrier);
    }
  }
};

}  // namespace mate::deep_gemm
