#pragma once

#include <mutlass/float8.h>
#include <mutlass/float_subbyte.h>
#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/arch/simd_mp31.hpp>
#include <mute/atom/copy_traits_mp32_tme.hpp>
#include <mute/atom/mma_atom.hpp>
#include <mute/atom/mma_traits_mp32_sqmma.hpp>
#include <mute/tensor.hpp>
#include <mute/tensor_zip.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
#include <mutlass/pipeline/pipeline.hpp>

#include "mp32_mqa_logits_blockscale_scheduler.hpp"
#include "mp32_mqa_logits_blockscale_utils.hpp"

namespace mate::deep_gemm {

using namespace mute;

template <class ElementQKV_,
          class ElementW_,
          class ElementLogits_,
          uint32_t kNumHeads_,
          uint32_t kHeaddim_,
          uint32_t kTileKV_,
          uint32_t kTileQH_,
          class StrideQ_,
          class StrideKV_,
          class StrideSFQ_,
          class StrideSFKV_,
          class StrideW_,
          class StrideLogits_,
          uint32_t kStagesQ_            = 1,
          uint32_t kStagesKV_           = 1,
          uint32_t kKVTokenAlignment_   = 4,
          bool     kIsCompressedLogits_ = false,
          uint32_t kScaleStorageBytes_  = 4,
          bool     IsQScaleMajorM       = false,
          bool     IsKVScaleMajorM      = false,
          bool     UseSchedule_         = false,
          uint32_t kNumMP_              = 0>
struct MP32MQALogitsBlockScale {
  using ElementQKV         = ElementQKV_;
  using ElementAccumulator = float;
  using ElementSF          = mutlass::float_ue8m0_t;
  using ElementW           = ElementW_;
  using ElementLogits      = ElementLogits_;

  using StrideQ      = StrideQ_;
  using StrideKV     = StrideKV_;
  using StrideSFQ    = StrideSFQ_;
  using StrideSFKV   = StrideSFKV_;
  using StrideW      = StrideW_;
  using StrideLogits = StrideLogits_;

  static constexpr int  kNumHeads              = int(kNumHeads_);
  static constexpr int  kHeaddim               = int(kHeaddim_);
  static constexpr int  kStagesQ               = int(kStagesQ_);
  static constexpr int  kStagesKV              = int(kStagesKV_);
  static constexpr int  kKVTokenAlignment      = int(kKVTokenAlignment_);
  static constexpr bool kIsCompressedLogits    = kIsCompressedLogits_;
  static constexpr int  kMmaKV                 = 128;
  static constexpr int  kTileKV                = int(kTileKV_);
  static constexpr int  kTileQH                = int(kTileQH_);
  static constexpr int  kNumProducerWarpSquads = 1;
  static constexpr int  kNumConsumerWarpSquads = kTileKV / kMmaKV;
  static constexpr int  kTileQ                 = kTileQH / kNumHeads;
  static constexpr int  QuantBlockTile         = 32;
  static constexpr int  kElementQKVBits        = sizeof_bits_v<ElementQKV>;
  static constexpr bool kIsMXFP8               = is_same_v<ElementQKV, mutlass::float_e4m3_t>;
  static constexpr bool kIsMXFP4               = is_same_v<ElementQKV, mutlass::float_e2m1_t>;
  static constexpr int  kMmaHeaddim            = kIsMXFP4 ? ceil_div(kHeaddim, 64) * 64 : kHeaddim;
  static constexpr int  kValidScaleBytes       = kHeaddim / QuantBlockTile;
  static constexpr int  kNumMmaScaleGroups     = kMmaHeaddim / QuantBlockTile;
  static constexpr int  kScaleStorageBytes     = int(kScaleStorageBytes_);

  static_assert(kStagesQ > 0, "MP32 block-scale MQA logits requires at least one Q stage");
  static_assert(kStagesKV > 0, "MP32 block-scale MQA logits requires at least one KV stage");
  static_assert(kTileQH % kNumHeads == 0, "MP32 block-scale MQA logits requires heads to divide TileQH");
  static_assert(kTileKV >= kMmaKV && kTileKV % kMmaKV == 0,
                "MP32 block-scale MQA logits requires TileKV to be a positive multiple of MMA_KV");
  static_assert(kNumConsumerWarpSquads <= mutlass::NumWarpsPerWarpSquad,
                "MP32 block-scale MQA logits requires one producer warp per consumer warpsquad");
  static_assert(kIsMXFP8 || kIsMXFP4, "MP32 block-scale MQA logits requires MXFP8 or MXFP4 operands");
  static_assert(kHeaddim > 0, "MP32 block-scale MQA logits requires a positive head dimension");
  static_assert(kHeaddim % QuantBlockTile == 0, "MP32 block-scale MQA logits requires head dimension divisible by 32");
  static_assert(!kIsMXFP4 || kHeaddim <= 128, "MP32 MXFP4 MQA logits requires head dimension up to 128");
  static_assert(kValidScaleBytes == 1 || kValidScaleBytes == 2 || kValidScaleBytes == 4,
                "MP32 block-scale MQA logits supports 1, 2, or 4 scale groups");
  static_assert(kValidScaleBytes <= kScaleStorageBytes,
                "MP32 block-scale MQA logits scale storage must cover valid scale groups");
  static_assert(kNumMmaScaleGroups <= kScaleStorageBytes,
                "MP32 block-scale MQA logits scale storage must cover MMA scale groups");

  using MMATileShape = Shape<Int<kMmaKV>, Int<kTileQH>, Int<kMmaHeaddim>>;
  using Scheduler    = conditional_t<UseSchedule_,
                                     MP32MQALogitsStaticScheduler<kTileKV, kTileQ, kNumMP_, kKVTokenAlignment>,
                                     MP32MQALogitsLegacyScheduler<kTileKV, kTileQ, kKVTokenAlignment>>;
  using Task         = typename Scheduler::Task;

  using ElementPair = conditional_t<kIsMXFP4, mutlass::mx_float4_t<ElementQKV>, mutlass::mx_float8_t<ElementQKV>>;
  static constexpr int kAlignmentQKV = 32 / kElementQKVBits;
  // The collective supplies MMA/layout metadata only; MQA owns independently staged Q and KV pipelines.
  using CollectiveMetadata = typename mutlass::gemm::collective::CollectiveBuilder<
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
  using SqmmaShape       = typename TiledMma::AtomShape_MNK;
  using SmemCopyAtomSFKV = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;
  using SmemCopyAtomSFQ  = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;
  using SmemCopyAtomW    = Copy_Atom<DefaultCopy, ElementW>;

  static constexpr int kReductionTarget = size(mqa_reduction_target_n(TiledMma{}));
  static constexpr int kHeadsPerThread  = kNumHeads / kReductionTarget;
  static constexpr int kNumThreadsWS2R  = mutlass::NumThreadsPerWarpSquad;

  using SmemLayoutKV      = decltype(tile_to_shape(typename CollectiveMetadata::SmemLayoutAtomA{},
                                              Shape<Int<kTileKV>, Int<kMmaHeaddim>, Int<kStagesKV>>{}));
  using SmemLayoutQ       = decltype(tile_to_shape(typename CollectiveMetadata::SmemLayoutAtomB{},
                                             Shape<Int<kTileQH>, Int<kMmaHeaddim>, Int<kStagesQ>>{}));
  using SmemLayoutKVStage = decltype(take<0, 2>(SmemLayoutKV{}));
  using SmemLayoutQStage  = decltype(take<0, 2>(SmemLayoutQ{}));
  using SmemLayoutLogits  = Layout<Shape<Int<kTileQ>, Int<kTileKV>>, Stride<Int<kTileKV>, _1>>;

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
  static constexpr int kSmemSFKVStageSize   = kTileKV * kScaleStorageBytes;
  static constexpr int kSmemSFKVGroupStride = IsKVScaleMajorM ? kTileKV : 1;
  using LoadLayoutSFKV                      = Layout<Shape<Int<kTileKV>, Int<kScaleStorageBytes>, Int<kStagesKV>>,
                                                     Stride<Int<kSmemSFKVRowStride>, Int<kSmemSFKVGroupStride>, Int<kSmemSFKVStageSize>>>;
  using LoadLayoutSFKVStage                 = decltype(take<0, 2>(LoadLayoutSFKV{}));
  using MMALayoutSFKV                       = Layout<Shape<Shape<Int<kMmaKV>, Int<kNumConsumerWarpSquads>>,
                                                           Shape<Int<QuantBlockTile>, Int<kNumMmaScaleGroups>>,
                                                           Int<kStagesKV>>,
                                                     Stride<Stride<Int<kSmemSFKVRowStride>, Int<kMmaKV * kSmemSFKVRowStride>>,
                                                            Stride<_0, Int<kSmemSFKVGroupStride>>,
                                                            Int<kSmemSFKVStageSize>>>;

  using SmemLayoutWG2SStage = Layout<Shape<Int<kTileQH>>, Stride<_1>>;
  using SmemLayoutWG2S      = Layout<Shape<Int<kTileQH>, Int<kStagesQ>>, Stride<_1, Int<kTileQH>>>;
  using SmemLayoutWS2R      = Layout<Shape<Shape<Int<kReductionTarget>, Int<kNumThreadsWS2R / kReductionTarget>>,
                                           Int<kTileQ * kHeadsPerThread * kStagesQ>>,
                                     Stride<Stride<_1, _0>, Int<kReductionTarget>>>;
  using SmemCopyThreadLayoutWS2R =
      Layout<Shape<Shape<Int<kReductionTarget>, Int<kNumThreadsWS2R / kReductionTarget>>, _1>,
             Stride<Stride<_1, Int<kReductionTarget>>, _0>>;
  using SmemCopyValueLayoutWS2R = Layout<Shape<_1, Int<kTileQ * kHeadsPerThread>>, Stride<_0, _1>>;

  using PipelineQ           = mutlass::PipelineTmeAsync<kStagesQ>;
  using PipelineKV          = mutlass::PipelineTmeAsync<kStagesKV>;
  using PipelineLogits      = mutlass::PipelineAsync<1>;
  using PipelineQParams     = typename PipelineQ::Params;
  using PipelineKVParams    = typename PipelineKV::Params;
  using PipelineQState      = typename PipelineQ::PipelineState;
  using PipelineKVState     = typename PipelineKV::PipelineState;
  using PipelineLogitsState = typename PipelineLogits::PipelineState;

  static constexpr int kNumConsumerWarps = kNumConsumerWarpSquads * mutlass::NumWarpsPerWarpSquad;
  static constexpr int kTmeTransactionBytesQ =
      ceil_div(int(cosize(SmemLayoutQStage{})) * sizeof_bits_v<ElementQKV>, 8) +
      ceil_div(int(cosize(SmemLayoutWG2SStage{})) * sizeof_bits_v<ElementW>, 8) + kSmemSFQStageSize;
  static constexpr int kTmeTransactionBytesKV =
      ceil_div(int(cosize(SmemLayoutKVStage{})) * sizeof_bits_v<ElementQKV>, 8) + kSmemSFKVStageSize;
  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t PipelineQ[PipelineQ::NumBarriers];
    uint8_t PipelineKV[PipelineKV::NumBarriers];
    uint8_t PipelineLogits[PipelineLogits::NumBarriers];
  };

  using TmeScaleWord =
      conditional_t<kScaleStorageBytes == 1, uint8_t, conditional_t<kScaleStorageBytes == 2, uint16_t, uint32_t>>;
  using TmeQKVInternalType = conditional_t<(sizeof_bits_v<ElementQKV> == 4), mutlass::detail::float_e2m1x2_t, void>;

  using TME_Q = decltype(make_tme_copy<TmeQKVInternalType>(
      MP31_TME_LOAD{},
      make_tensor(
          make_gmem_ptr(static_cast<ElementQKV const*>(nullptr)), repeat_like(StrideQ{}, int32_t(0)), StrideQ{}),
      SmemLayoutQStage{},
      select<1, 2>(MMATileShape{}),
      TiledMma{}));

  using TME_KV = decltype(make_tme_copy<TmeQKVInternalType>(
      MP31_TME_LOAD{},
      make_tensor(
          make_gmem_ptr(static_cast<ElementQKV const*>(nullptr)), repeat_like(StrideKV{}, int32_t(0)), StrideKV{}),
      SmemLayoutKVStage{},
      make_shape(Int<kTileKV>{}, Int<kMmaHeaddim>{}),
      TiledMma{}));

  using TME_W = decltype(make_tme_copy(MP31_TME_LOAD{},
                                       make_tensor(make_gmem_ptr(static_cast<ElementW const*>(nullptr)),
                                                   make_layout(make_shape(int32_t(0)), make_stride(get<0>(StrideW{})))),
                                       SmemLayoutWG2SStage{},
                                       shape(SmemLayoutWG2SStage{})));

  using TME_SFQ = decltype(make_tme_copy<TmeScaleWord>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementSF const*>(nullptr)),
                  make_layout(make_shape(int32_t(0), Int<kScaleStorageBytes>{}), StrideSFQ{})),
      LoadLayoutSFQStage{},
      shape(LoadLayoutSFQStage{})));

  using TME_SFKV = decltype(make_tme_copy<TmeScaleWord>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementSF const*>(nullptr)),
                  make_layout(make_shape(int32_t(0), Int<kScaleStorageBytes>{}), StrideSFKV{})),
      LoadLayoutSFKVStage{},
      shape(LoadLayoutSFKVStage{})));

  using TME_Logits = decltype(make_tme_copy(
      MP31_TME_STORE{},
      make_tensor(
          make_gmem_ptr(static_cast<ElementLogits*>(nullptr)), repeat_like(StrideLogits{}, int32_t(0)), StrideLogits{}),
      SmemLayoutLogits{},
      shape(SmemLayoutLogits{})));

  using SmemLayoutLogitsRow = Layout<Shape<_1, Int<kTileKV>>, Stride<Int<kTileKV>, _1>>;
  using TME_LogitsRow       = decltype(make_tme_copy(
      MP31_TME_STORE{},
      make_tensor(
          make_gmem_ptr(static_cast<ElementLogits*>(nullptr)), repeat_like(StrideLogits{}, int32_t(0)), StrideLogits{}),
      SmemLayoutLogitsRow{},
      shape(SmemLayoutLogitsRow{})));

  static constexpr int kSmemAlignment = 256;

  struct SharedStorage {
    alignas(kSmemAlignment) ArrayEngine<ElementQKV, cosize_v<SmemLayoutQ>> smem_q;
    alignas(kSmemAlignment) ArrayEngine<ElementQKV, cosize_v<SmemLayoutKV>> smem_kv;
    alignas(kSmemAlignment) ArrayEngine<ElementW, cosize_v<SmemLayoutWG2S>> smem_weights;
    alignas(kSmemAlignment) ArrayEngine<TmeScaleWord, cosize_v<LoadLayoutSFKV> / kScaleStorageBytes> smem_sfkv;
    alignas(kSmemAlignment) ArrayEngine<TmeScaleWord, cosize_v<LoadLayoutSFQ> / kScaleStorageBytes> smem_sfq;
    alignas(kSmemAlignment) ArrayEngine<ElementLogits, cosize_v<SmemLayoutLogits>> smem_logits;
  };

  static constexpr int      SharedStorageSize  = sizeof(SharedStorage);
  static constexpr int      SmemAlignmentBytes = kSmemAlignment;
  static constexpr uint32_t MaxThreadsPerBlock =
      (kNumProducerWarpSquads + kNumConsumerWarpSquads) * mutlass::NumThreadsPerWarpSquad;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  struct Arguments {
    int32_t num_tile_q;
    int32_t seqlen_kv;
    int32_t num_heads;
    int32_t headdim;
    int32_t total_seqlen_q;

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

    int32_t const*  cu_seqlen_k_start;
    int32_t const*  cu_seqlen_k_end;
    uint32_t const* schedule_meta;

    ElementLogits* ptr_logits;
    StrideLogits   stride_logits;
    int32_t        max_seqlen_kv;
  };

  struct Params {
    int32_t num_tile_q;
    int32_t seqlen_kv;
    int32_t num_heads;
    int32_t headdim;
    int32_t total_seqlen_q;
    int32_t num_kv_blocks;

    TME_Q         tme_q;
    TME_KV        tme_kv;
    TME_SFQ       tme_sfq;
    TME_SFKV      tme_sfkv;
    TME_W         tme_w;
    TME_Logits    tme_logits;
    TME_LogitsRow tme_logits_row;

    int32_t const*  cu_seqlen_k_start;
    int32_t const*  cu_seqlen_k_end;
    uint32_t const* schedule_meta;

    ElementLogits* ptr_logits;
    StrideLogits   stride_logits;
    int32_t        max_seqlen_kv;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    int32_t const num_kv_blocks = ceil_div(args.seqlen_kv, int32_t(kTileKV));
    int32_t const max_seqlen_kv = args.max_seqlen_kv > 0 ? args.max_seqlen_kv : args.seqlen_kv;

    TME_Q tme_q = make_tme_copy<TmeQKVInternalType>(
        MP31_TME_LOAD{},
        make_tensor(
            make_gmem_ptr(args.ptr_q), make_shape(args.total_seqlen_q * args.num_heads, args.headdim), args.stride_q),
        SmemLayoutQStage{},
        select<1, 2>(MMATileShape{}),
        TiledMma{});
    TME_KV tme_kv = make_tme_copy<TmeQKVInternalType>(
        MP31_TME_LOAD{},
        make_tensor(make_gmem_ptr(args.ptr_kv), make_shape(args.seqlen_kv, args.headdim), args.stride_kv),
        SmemLayoutKVStage{},
        make_shape(Int<kTileKV>{}, Int<kMmaHeaddim>{}),
        TiledMma{});
    TME_W tme_w = make_tme_copy(MP31_TME_LOAD{},
                                make_tensor(make_gmem_ptr(args.ptr_weights),
                                            make_layout(make_shape(args.total_seqlen_q * args.num_heads),
                                                        make_stride(get<0>(args.stride_weights)))),
                                SmemLayoutWG2SStage{},
                                shape(SmemLayoutWG2SStage{}));

    auto    sfq_shape = make_shape(args.total_seqlen_q * args.num_heads, Int<kScaleStorageBytes>{});
    TME_SFQ tme_sfq   = make_tme_copy<TmeScaleWord>(MP31_TME_LOAD{},
                                                  make_tensor(make_gmem_ptr(args.ptr_sfq), sfq_shape, args.stride_sfq),
                                                  LoadLayoutSFQStage{},
                                                  shape(LoadLayoutSFQStage{}));

    auto     sfkv_shape = make_shape(args.seqlen_kv, Int<kScaleStorageBytes>{});
    TME_SFKV tme_sfkv =
        make_tme_copy<TmeScaleWord>(MP31_TME_LOAD{},
                                    make_tensor(make_gmem_ptr(args.ptr_sfkv), sfkv_shape, args.stride_sfkv),
                                    LoadLayoutSFKVStage{},
                                    shape(LoadLayoutSFKVStage{}));
    TME_Logits    tme_logits{};
    TME_LogitsRow tme_logits_row{};
    if constexpr (!kIsCompressedLogits) {
      tme_logits = make_tme_copy(MP31_TME_STORE{},
                                 make_tensor(make_gmem_ptr(args.ptr_logits),
                                             make_shape(args.num_tile_q * kTileQ, get<0>(args.stride_logits)),
                                             args.stride_logits),
                                 SmemLayoutLogits{},
                                 shape(SmemLayoutLogits{}));
    } else {
      // Row-granular store: dim1 = max_seqlen_kv clips the right edge through TME OOB handling.
      tme_logits_row = make_tme_copy(
          MP31_TME_STORE{},
          make_tensor(
              make_gmem_ptr(args.ptr_logits), make_shape(args.num_tile_q * kTileQ, max_seqlen_kv), args.stride_logits),
          SmemLayoutLogitsRow{},
          shape(SmemLayoutLogitsRow{}));
    }

    return Params{
        .num_tile_q        = args.num_tile_q,
        .seqlen_kv         = args.seqlen_kv,
        .num_heads         = args.num_heads,
        .headdim           = args.headdim,
        .total_seqlen_q    = args.total_seqlen_q,
        .num_kv_blocks     = num_kv_blocks,
        .tme_q             = tme_q,
        .tme_kv            = tme_kv,
        .tme_sfq           = tme_sfq,
        .tme_sfkv          = tme_sfkv,
        .tme_w             = tme_w,
        .tme_logits        = tme_logits,
        .tme_logits_row    = tme_logits_row,
        .cu_seqlen_k_start = args.cu_seqlen_k_start,
        .cu_seqlen_k_end   = args.cu_seqlen_k_end,
        .schedule_meta     = args.schedule_meta,
        .ptr_logits        = args.ptr_logits,
        .stride_logits     = args.stride_logits,
        .max_seqlen_kv     = max_seqlen_kv,
    };
  }

  MUTLASS_DEVICE void load(Params const&    params,
                           SharedStorage&   shared_storage,
                           PipelineQ&       pipeline_q,
                           PipelineQState&  pipeline_q_write,
                           PipelineKV&      pipeline_kv,
                           PipelineKVState& pipeline_kv_write,
                           Scheduler&       scheduler) {
    auto cta_tme_w    = params.tme_w.get_slice(0);
    auto cta_tme_q    = params.tme_q.get_slice(0);
    auto cta_tme_sfq  = params.tme_sfq.get_slice(0);
    auto cta_tme_kv   = params.tme_kv.get_slice(0);
    auto cta_tme_sfkv = params.tme_sfkv.get_slice(0);

    auto mW   = params.tme_w.get_tme_tensor(make_shape(params.total_seqlen_q * params.num_heads));
    auto gW   = local_tile(mW, shape(SmemLayoutWG2SStage{}), make_coord(_));
    auto sW   = make_tensor(make_smem_ptr(shared_storage.smem_weights.begin()), SmemLayoutWG2S{});
    auto tWgW = cta_tme_w.partition_S(gW);
    auto tWsW = cta_tme_w.partition_D(sW);

    auto mQ   = params.tme_q.get_tme_tensor(make_shape(params.total_seqlen_q * params.num_heads, params.headdim));
    auto gQ   = local_tile(mQ, select<1, 2>(MMATileShape{}), make_coord(_, _));
    auto sQ   = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()), SmemLayoutQ{});
    auto tQgQ = cta_tme_q.partition_S(gQ(_, _, _, _0{}));
    auto tQsQ = cta_tme_q.partition_D(sQ);

    auto sfq_shape = make_shape(params.total_seqlen_q * params.num_heads, Int<kScaleStorageBytes>{});
    auto mSFQ      = params.tme_sfq.get_tme_tensor(sfq_shape);
    auto gSFQ      = local_tile(mSFQ, shape(LoadLayoutSFQStage{}), make_coord(_, _0{}));
    auto sSFQ =
        make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(shared_storage.smem_sfq.begin())), LoadLayoutSFQ{});
    auto tSFQgSFQ = cta_tme_sfq.partition_S(gSFQ);
    auto tSFQsSFQ = cta_tme_sfq.partition_D(sSFQ);

    auto mKV    = params.tme_kv.get_tme_tensor(make_shape(params.seqlen_kv, params.headdim));
    auto sKV    = make_tensor(make_smem_ptr(shared_storage.smem_kv.begin()), SmemLayoutKV{});
    auto tKVsKV = cta_tme_kv.partition_D(sKV);

    auto sfkv_shape = make_shape(params.seqlen_kv, Int<kScaleStorageBytes>{});
    auto mSFKV      = params.tme_sfkv.get_tme_tensor(sfkv_shape);
    auto sSFKV =
        make_tensor(make_smem_ptr(reinterpret_cast<ElementSF*>(shared_storage.smem_sfkv.begin())), LoadLayoutSFKV{});
    auto tSFKVsSFKV = cta_tme_sfkv.partition_D(sSFKV);

    Task task{};
    while (scheduler.get_next_task(task)) {
      __musa_loop_transparent_outermost();
      pipeline_q.producer_acquire(pipeline_q_write);
      uint32_t const q_barrier_id = pipeline_q.producer_get_barrier_id(pipeline_q_write);
      int const      q_stage      = pipeline_q_write.index();

      copy(params.tme_w.with(q_barrier_id), tWgW(_, _, task.tile_q_idx), tWsW(_, _, q_stage));
      copy(params.tme_q.with(q_barrier_id), tQgQ(_, _, _, task.tile_q_idx), tQsQ(_, _, _, q_stage));
      copy(params.tme_sfq.with(q_barrier_id), tSFQgSFQ(_, _, _, task.tile_q_idx), tSFQsSFQ(_, _, _, q_stage));
      ++pipeline_q_write;

      auto mKVBase = domain_offset(make_coord(int32_t(task.kv_token_base), _0{}), mKV);
      auto gKV     = local_tile(mKVBase, make_shape(Int<kTileKV>{}, Int<kMmaHeaddim>{}), make_coord(_, _0{}));
      auto tKVgKV  = cta_tme_kv.partition_S(gKV);

      auto mSFKVBase  = domain_offset(make_coord(int32_t(task.kv_token_base), _0{}), mSFKV);
      auto gSFKV      = local_tile(mSFKVBase, shape(LoadLayoutSFKVStage{}), make_coord(_, _0{}));
      auto tSFKVgSFKV = cta_tme_sfkv.partition_S(gSFKV);

      for (uint32_t kv_tile_idx = 0; kv_tile_idx < task.num_tile_kv; ++kv_tile_idx) {
        pipeline_kv.producer_acquire(pipeline_kv_write);
        uint32_t const kv_barrier_id = pipeline_kv.producer_get_barrier_id(pipeline_kv_write);
        int const      kv_stage      = pipeline_kv_write.index();

        copy(params.tme_kv.with(kv_barrier_id), tKVgKV(_, _, _, kv_tile_idx), tKVsKV(_, _, _, kv_stage));
        copy(params.tme_sfkv.with(kv_barrier_id), tSFKVgSFKV(_, _, _, kv_tile_idx), tSFKVsSFKV(_, _, _, kv_stage));
        ++pipeline_kv_write;
      }
    }
  }

  MUTLASS_DEVICE void write(Params const&        params,
                            SharedStorage&       shared_storage,
                            PipelineLogits&      pipeline_logits,
                            PipelineLogitsState& pipeline_logits_read,
                            Scheduler&           scheduler) {
    if constexpr (!kIsCompressedLogits) {
      auto cta_tme_logits = params.tme_logits.get_slice(0);
      auto mLogits =
          params.tme_logits.get_tme_tensor(make_shape(params.num_tile_q * kTileQ, get<0>(params.stride_logits)));
      auto sLogits   = make_tensor(make_smem_ptr(shared_storage.smem_logits.begin()), SmemLayoutLogits{});
      auto tSsLogits = cta_tme_logits.partition_S(sLogits);

      Task task{};
      while (scheduler.get_next_task(task)) {
        __musa_loop_transparent_outermost();
        auto mLogitsBase = domain_offset(make_coord(_0{}, int32_t(task.kv_token_base)), mLogits);

        for (uint32_t kv_tile_idx = 0; kv_tile_idx < task.num_tile_kv; ++kv_tile_idx) {
          pipeline_logits.consumer_wait(pipeline_logits_read);

          auto gLogits   = local_tile(mLogitsBase, shape(SmemLayoutLogits{}), make_coord(task.tile_q_idx, kv_tile_idx));
          auto tSgLogits = cta_tme_logits.partition_D(gLogits);
          copy(params.tme_logits, tSsLogits, tSgLogits);
          tme_store_arrive();
          tme_store_wait();

          pipeline_logits.consumer_release(pipeline_logits_read);
          ++pipeline_logits_read;
        }
      }
    } else {
      // Per-row destinations: column = kv token - ks[row]; TME OOB handling drops the
      // pre-window (negative) part and clips at dim1 = max_seqlen_kv on the right.
      auto cta_tme_logits_row = params.tme_logits_row.get_slice(0);
      auto mLogitsRow =
          params.tme_logits_row.get_tme_tensor(make_shape(params.num_tile_q * kTileQ, params.max_seqlen_kv));

      Task task{};
      while (scheduler.get_next_task(task)) {
        __musa_loop_transparent_outermost();

        for (uint32_t kv_tile_idx = 0; kv_tile_idx < task.num_tile_kv; ++kv_tile_idx) {
          pipeline_logits.consumer_wait(pipeline_logits_read);

          int64_t const tile_start = int64_t(task.kv_token_base) + int64_t(kv_tile_idx) * kTileKV;
          MUTE_UNROLL
          for (int q = 0; q < kTileQ; ++q) {
            int64_t const ks = int64_t(scheduler.seqlen_k_start[q]);
            auto          sRow =
                make_tensor(make_smem_ptr(shared_storage.smem_logits.begin() + q * kTileKV), SmemLayoutLogitsRow{});
            auto tSsRow = cta_tme_logits_row.partition_S(sRow);
            auto mBase =
                domain_offset(make_coord(int32_t(task.tile_q_idx * kTileQ + q), int32_t(tile_start - ks)), mLogitsRow);
            auto gRow   = local_tile(mBase, shape(SmemLayoutLogitsRow{}), make_coord(_0{}, _0{}));
            auto tSgRow = cta_tme_logits_row.partition_D(gRow);
            copy(params.tme_logits_row, tSsRow, tSgRow);
            tme_store_arrive();
          }
          tme_store_wait();

          pipeline_logits.consumer_release(pipeline_logits_read);
          ++pipeline_logits_read;
        }
      }
    }
  }

  template <class FinalSum>
  MUTLASS_DEVICE void epilogue(Params const&        params,
                               Scheduler const&     scheduler,
                               Task const&          task,
                               uint32_t             kv_tile_idx,
                               int                  consumer_idx,
                               int                  thread_idx_in_warpsquad,
                               SharedStorage&       shared_storage,
                               PipelineLogits&      pipeline_logits,
                               PipelineLogitsState& pipeline_logits_write,
                               FinalSum const&      final_sum) {
    pipeline_logits.producer_acquire(pipeline_logits_write);

    if (thread_idx_in_warpsquad % kReductionTarget == 0) {
      TiledMma tiled_mma;
      auto     thr_mma  = tiled_mma.get_thread_slice(thread_idx_in_warpsquad);
      Tensor   cMN      = make_identity_tensor(take<0, 2>(MMATileShape{}));
      Tensor   tCcMN    = thr_mma.partition_C(cMN);
      Tensor   tCcMN_mn = make_tensor(tCcMN.data(), mqa_layout_acc_mn(tiled_mma, tCcMN.layout()));

      Tensor    sLogits = make_tensor(make_smem_ptr(shared_storage.smem_logits.begin()), SmemLayoutLogits{});
      int const kv_base = consumer_idx * kMmaKV;
      MUTE_UNROLL
      for (int q = 0; q < kTileQ; ++q) {
        MUTE_UNROLL
        for (int m = 0; m < size<1>(final_sum); ++m) {
          int kv_local         = kv_base + int(get<0>(tCcMN_mn(m, _0{})));
          sLogits(q, kv_local) = static_cast<ElementLogits>(final_sum(q, m));
        }
      }
    }

    __syncwarp();
    pipeline_logits.producer_commit(pipeline_logits_write);
    ++pipeline_logits_write;
  }

  MUTLASS_DEVICE void compute(Params const&        params,
                              SharedStorage&       shared_storage,
                              int                  consumer_idx,
                              PipelineQ&           pipeline_q,
                              PipelineQState&      pipeline_q_read,
                              PipelineKV&          pipeline_kv,
                              PipelineKVState&     pipeline_kv_read,
                              PipelineLogits&      pipeline_logits,
                              PipelineLogitsState& pipeline_logits_write,
                              Scheduler&           scheduler) {
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

    static_assert(size(accum) % 4 == 0, "MP32 MQA logits accumulator fragment must be divisible by BST4");
    static_assert(decltype(size<2>(tCrKV))::value == 1 && decltype(size<2>(tCrQ))::value == 1,
                  "MP32 MQA logits requires one MMA K block");

    Task task{};
    while (scheduler.get_next_task(task)) {
      __musa_loop_transparent_outermost();
      pipeline_q.consumer_wait(pipeline_q_read);
      int const q_stage = pipeline_q_read.index();

      copy(tCsW(_, _, q_stage), tCrW);
      copy(tCsSFQ(_, _, Int<0>{}, q_stage), tCrSFQCopy(_, _, Int<0>{}));
      auto tCrWF = to_float_fragment(tCrW);

      for (uint32_t kv_tile_idx = 0; kv_tile_idx < task.num_tile_kv; ++kv_tile_idx) {
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

        epilogue(params,
                 scheduler,
                 task,
                 kv_tile_idx,
                 consumer_idx,
                 thread_idx_in_warpsquad,
                 shared_storage,
                 pipeline_logits,
                 pipeline_logits_write,
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
    pipeline_kv_params.transaction_bytes = kTmeTransactionBytesKV;
    pipeline_kv_params.num_consumers     = kNumConsumerWarps;
    pipeline_kv_params.num_producers     = 1;

    typename PipelineLogits::Params pipeline_logits_params;
    pipeline_logits_params.producer_arv_count = kNumConsumerWarps;
    pipeline_logits_params.consumer_arv_count = 1;

    int const warp_idx       = mutlass::canonical_warp_idx_sync();
    int const warp_squad_idx = mutlass::canonical_warp_squad_idx();
    int const consumer_idx   = warp_squad_idx - kNumProducerWarpSquads;

    PipelineQ      pipeline_q(pipeline_q_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineQ));
    PipelineKV     pipeline_kv(pipeline_kv_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineKV));
    PipelineLogits pipeline_logits(pipeline_logits_params,
                                   reinterpret_cast<uint64_t>(&barrier_storage->PipelineLogits));

    PipelineQState      pipeline_q_read;
    PipelineQState      pipeline_q_write = mutlass::make_producer_start_state<PipelineQ>();
    PipelineKVState     pipeline_kv_read;
    PipelineKVState     pipeline_kv_write = mutlass::make_producer_start_state<PipelineKV>();
    PipelineLogitsState pipeline_logits_read;
    PipelineLogitsState pipeline_logits_write = mutlass::make_producer_start_state<PipelineLogits>();

    __syncthreads();

    Scheduler scheduler(uint32_t(blockIdx.x),
                        uint32_t(gridDim.x),
                        uint32_t(params.num_tile_q),
                        uint32_t(params.total_seqlen_q),
                        uint32_t(params.seqlen_kv),
                        params.cu_seqlen_k_start,
                        params.cu_seqlen_k_end,
                        params.schedule_meta);

    if (warp_idx == 0) {
      load(params, shared_storage, pipeline_q, pipeline_q_write, pipeline_kv, pipeline_kv_write, scheduler);
    } else if (warp_idx == 1) {
      write(params, shared_storage, pipeline_logits, pipeline_logits_read, scheduler);
    } else if (warp_squad_idx >= kNumProducerWarpSquads) {
      compute(params,
              shared_storage,
              consumer_idx,
              pipeline_q,
              pipeline_q_read,
              pipeline_kv,
              pipeline_kv_read,
              pipeline_logits,
              pipeline_logits_write,
              scheduler);
    }
  }
};

}  // namespace mate::deep_gemm
