#pragma once

#include "mute/atom/copy_atom.hpp"
#include "mute/atom/mma_atom.hpp"

#include "mutlass/arch/mma.h"
#include "mutlass/arch/barrier.hpp"
#include "mutlass/gemm/dispatch_policy.hpp"
#include "mutlass/gemm/collective/collective_builder_decl.hpp"
#include "mutlass/gemm/collective/collective_mma_decl.hpp"
#include "mutlass/gemm/collective/builders/mp3x_common.inl"
#include "mutlass/pipeline/mp32_pipeline.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass::gemm::collective {

namespace detail {

constexpr int mp32_smem_capacity_bytes = 256 * 1024;
constexpr int mp32_blockscaled_sf_vector_size = 32;

template <class KernelScheduleType>
constexpr bool
is_mp32_pingpong_schedule() {
  return mute::is_base_of_v<
      KernelTmeWarpSpecializedPingpongMp32, KernelScheduleType>;
}

template <class KernelScheduleType>
constexpr int
mp32_max_pipeline_stages() {
  constexpr int ReservedBarriers =
      arch::AsyncBarrier::ReservedAsyncBarrierCount;
  constexpr int OrderedBarriers =
      is_mp32_pingpong_schedule<KernelScheduleType>()
          ? mutlass::OrderedSequenceBarrier<2, 2>::NumBarriers : 0;
  constexpr int AvailableBarriers =
      arch::AsyncBarrier::HardwareMaxNumAsyncTransactionBarriers -
      ReservedBarriers - OrderedBarriers;
  using Pipeline = mutlass::PipelineTmeAsync<1>;
  constexpr int BarriersPerStage = Pipeline::NumBarriers / Pipeline::Stages;
  return AvailableBarriers / BarriersPerStage;
}

template <class KernelScheduleType>
constexpr bool
is_mp32_warp_specialized_schedule() {
  return mute::is_same_v<KernelScheduleType, KernelScheduleAuto> ||
         mute::is_same_v<KernelScheduleType, KernelTmeWarpSpecialized> ||
         (mute::is_base_of_v<KernelTmeWarpSpecializedMp32, KernelScheduleType> &&
          !mute::is_base_of_v<
              KernelTmeWarpSpecializedBlockScaledMixedInputMp32,
              KernelScheduleType>);
}

template <class KernelScheduleType>
using mp32_kernel_schedule_t = mute::conditional_t<
    mute::is_same_v<KernelScheduleType, KernelScheduleAuto> ||
        mute::is_same_v<KernelScheduleType, KernelTmeWarpSpecialized>,
    KernelTmeWarpSpecializedMp32,
    KernelScheduleType>;

template <class KernelScheduleType>
constexpr bool
is_mp32_blockscaled_mixed_input_schedule() {
  return mute::is_same_v<KernelScheduleType, KernelScheduleAuto> ||
         mute::is_base_of_v<
             KernelTmeWarpSpecializedBlockScaledMixedInputMp32,
             KernelScheduleType>;
}

template <class KernelScheduleType>
using mp32_blockscaled_mixed_input_kernel_schedule_t = mute::conditional_t<
    mute::is_same_v<KernelScheduleType, KernelScheduleAuto>,
    KernelTmeWarpSpecializedBlockScaledMixedInputMp32,
    KernelScheduleType>;

template <class ElementPair>
struct is_mp32_blockscaled_input_supported : mute::false_type {};

template <class Element>
struct is_mp32_blockscaled_input_supported<mx_float8_t<Element>>
    : mute::bool_constant<
          mute::is_same_v<Element, float_e4m3_t> ||
          mute::is_same_v<Element, float_e5m2_t>> {};

template <class Element>
struct is_mp32_blockscaled_input_supported<mx_float6_t<Element>>
    : mute::bool_constant<
          mute::is_same_v<Element, float_e2m3_t> ||
          mute::is_same_v<Element, float_e3m2_t>> {};

template <class Element>
struct is_mp32_blockscaled_input_supported<mx_float4_t<Element>>
    : mute::bool_constant<mute::is_same_v<Element, float_e2m1_t>> {};

template <class Element>
struct is_mp32_blockscaled_input_supported<mt_float4_t<Element>>
    : mute::bool_constant<mute::is_same_v<Element, float_e2m1_t>> {};

template <class Element>
using mp32_blockscaled_element_t = mute::conditional_t<
    mute::is_same_v<Element, float_e2m3_t>,
    float_e2m3_unpack8bits_t,
    mute::conditional_t<
        mute::is_same_v<Element, float_e3m2_t>,
        float_e3m2_unpack8bits_t,
        Element>>;

template <class ElementA, class ElementB, class ElementSFA, class ElementSFB>
constexpr bool
is_mp32_blockscaled_pair_supported() {
  constexpr bool IsSameScale = mute::is_same_v<ElementSFA, ElementSFB>;
  constexpr bool IsF8 =
      (mute::is_same_v<ElementA, float_e4m3_t> ||
       mute::is_same_v<ElementA, float_e5m2_t>) &&
      (mute::is_same_v<ElementB, float_e4m3_t> ||
       mute::is_same_v<ElementB, float_e5m2_t>);
  constexpr bool IsF6 =
      (mute::is_same_v<ElementA, float_e2m3_unpack8bits_t> ||
       mute::is_same_v<ElementA, float_e3m2_unpack8bits_t>) &&
      (mute::is_same_v<ElementB, float_e2m3_unpack8bits_t> ||
       mute::is_same_v<ElementB, float_e3m2_unpack8bits_t>);
  constexpr bool IsF4 =
      mute::is_same_v<ElementA, float_e2m1_t> &&
      mute::is_same_v<ElementB, float_e2m1_t>;
  constexpr bool IsMxScale = mute::is_same_v<ElementSFA, float_ue8m0_t>;
  constexpr bool IsMtScale = mute::is_same_v<ElementSFA, float_ue4m3_t>;

  return IsSameScale &&
         ((IsF8 && IsMxScale) ||
          (IsF6 && IsMxScale) ||
          (IsF4 && (IsMxScale || IsMtScale)));
}

template <class ElementPairA, class ElementPairB>
constexpr bool
is_mp32_blockscaled_ss_pair_supported() {
  if constexpr (is_mp32_blockscaled_input_supported<ElementPairA>::value &&
                is_mp32_blockscaled_input_supported<ElementPairB>::value) {
    using ElementA = mp32_blockscaled_element_t<typename ElementPairA::DataType>;
    using ElementB = mp32_blockscaled_element_t<typename ElementPairB::DataType>;
    using ElementSFA = typename ElementPairA::ScaleFactorType;
    using ElementSFB = typename ElementPairB::ScaleFactorType;
    return is_mp32_blockscaled_pair_supported<
        ElementA, ElementB, ElementSFA, ElementSFB>();
  }
  else {
    return false;
  }
}

template <class ElementPairA, class ElementPairB>
struct is_mp32_blockscaled_mixed_input_pair_supported : mute::false_type {};

template <class ElementA, class ElementB>
struct is_mp32_blockscaled_mixed_input_pair_supported<
    mx_float8_t<ElementA>, mx_float4_t<ElementB>>
    : mute::bool_constant<
          (mute::is_same_v<ElementA, float_e4m3_t> ||
           mute::is_same_v<ElementA, float_e5m2_t>) &&
          mute::is_same_v<ElementB, float_e2m1_t>> {};

template <class ElementA, class ElementB>
struct is_mp32_blockscaled_mixed_input_pair_supported<
    mx_float4_t<ElementA>, mx_float8_t<ElementB>>
    : mute::bool_constant<
          mute::is_same_v<ElementA, float_e2m1_t> &&
          (mute::is_same_v<ElementB, float_e4m3_t> ||
           mute::is_same_v<ElementB, float_e5m2_t>)> {};

template <class Element, mute::TCE::Major Major>
constexpr bool
is_mp32_blockscaled_major_supported() {
  return mute::sizeof_bits_v<Element> != 4 || Major == mute::TCE::Major::K;
}

template <int MN, int K>
using mp32_scale_factor_layout = mute::Layout<
    mute::Shape<mute::Int<MN>,
                mute::Shape<mute::_32, mute::Int<K / mp32_blockscaled_sf_vector_size>>>,
    mute::Stride<mute::_1, mute::Stride<mute::_0, mute::Int<MN>>>>;

template <class ElementA, class ElementB, class TileShapeMNK>
constexpr int
mp32_dense_stage_bytes() {
  constexpr int TileM = mute::size<0>(TileShapeMNK{});
  constexpr int TileN = mute::size<1>(TileShapeMNK{});
  constexpr int TileK = mute::size<2>(TileShapeMNK{});
  return mutlass::bits_to_bytes(mute::sizeof_bits_v<ElementA> * TileM * TileK) +
         mutlass::bits_to_bytes(mute::sizeof_bits_v<ElementB> * TileN * TileK);
}

template <class ElementA, class ElementB,
          class ElementSFA, class ElementSFB,
          class TileShapeMNK>
constexpr int
mp32_blockscaled_stage_bytes() {
  constexpr int TileM = mute::size<0>(TileShapeMNK{});
  constexpr int TileN = mute::size<1>(TileShapeMNK{});
  constexpr int TileK = mute::size<2>(TileShapeMNK{});
  static_assert(TileK % mp32_blockscaled_sf_vector_size == 0,
                "MP32 block-scaled Tile_K must be divisible by SFVecSize");

  constexpr int DataBits =
      mute::sizeof_bits_v<ElementA> * TileM * TileK +
      mute::sizeof_bits_v<ElementB> * TileN * TileK;
  constexpr int ScaleBits =
      mute::sizeof_bits_v<ElementSFA> * TileM *
          (TileK / mp32_blockscaled_sf_vector_size) +
      mute::sizeof_bits_v<ElementSFB> * TileN *
          (TileK / mp32_blockscaled_sf_vector_size);
  return mutlass::bits_to_bytes(DataBits + ScaleBits);
}

template <int CapacityBytes, int StageBytes, int MaxStages, int Stages>
constexpr int
compute_mp32_stage_count_or_override(StageCount<Stages>) {
  static_assert(Stages >= 2 && Stages <= MaxStages,
                "MP32 mainloop stage count exceeds the schedule's barrier capacity");
  static_assert(Stages * StageBytes <= CapacityBytes,
                "MP32 mainloop stage count exceeds shared-memory capacity");
  return Stages;
}

template <int CapacityBytes, int StageBytes, int MaxStages, int Stages>
constexpr int
compute_mp32_stage_count_or_override(mute::Int<Stages>) {
  static_assert(Stages >= 2 && Stages <= MaxStages,
                "MP32 mainloop stage count exceeds the schedule's barrier capacity");
  static_assert(Stages * StageBytes <= CapacityBytes,
                "MP32 mainloop stage count exceeds shared-memory capacity");
  return Stages;
}

template <int CapacityBytes, int StageBytes, int MaxStages, int CarveoutBytes>
constexpr int
compute_mp32_stage_count_or_override(StageCountAutoCarveout<CarveoutBytes>) {
  constexpr int MaximumStages = (CapacityBytes - CarveoutBytes) / StageBytes;
  static_assert(MaximumStages >= 2, "Mainloop needs at least 2 stages");
  return mute::min(MaximumStages, MaxStages);
}

template <mute::TCE::Major Major, class ElementType, class SqmmaOp, class TileShape>
MUTE_HOST_DEVICE constexpr auto
mp32_ss_smem_selector_A() {
  using AtomOpShape = typename mute::MMA_Traits<SqmmaOp>::Shape_MNK;
  using AtomM = decltype(mute::get<0>(AtomOpShape{}));
  using AtomK = decltype(mute::get<2>(AtomOpShape{}));

  if constexpr (Major == mute::TCE::Major::MN) {
    return mute::MP32::SQMMA::make_canonical_gemm_smem_atom_layout<
        ElementType, Major, AtomM, mute::tuple_element_t<2, TileShape>>();
  }
  else {
    constexpr int SmemAtomBits = 256 * 8;
    constexpr int ElementBits = mute::sizeof_bits_v<ElementType>;
    constexpr int TileKBits = mute::size<2>(TileShape{}) * ElementBits;
    static_assert(SmemAtomBits % ElementBits == 0,
                  "Element type must evenly divide a 256-byte SMEM atom");
    constexpr int MaxAtomK = SmemAtomBits / ElementBits;
    using SmemAtomK = mute::conditional_t<
        TileKBits % SmemAtomBits == 0,
        mute::Int<mute::min(MaxAtomK, mute::tuple_element_t<2, TileShape>{})>,
        AtomK>;
    return mute::MP32::SQMMA::make_canonical_gemm_smem_atom_layout<
        ElementType, Major, mute::tuple_element_t<0, TileShape>, SmemAtomK>();
  }
}

template <mute::TCE::Major Major, class ElementType, class SqmmaOp, class TileShape>
MUTE_HOST_DEVICE constexpr auto
mp32_ss_smem_selector_B() {
  using AtomOpShape = typename mute::MMA_Traits<SqmmaOp>::Shape_MNK;
  using AtomN = decltype(mute::get<1>(AtomOpShape{}));
  using AtomK = decltype(mute::get<2>(AtomOpShape{}));

  if constexpr (Major == mute::TCE::Major::MN) {
    return mute::MP32::SQMMA::make_canonical_gemm_smem_atom_layout<
        ElementType, Major, AtomN, mute::tuple_element_t<2, TileShape>>();
  }
  else {
    constexpr int SmemAtomBits = 256 * 8;
    constexpr int ElementBits = mute::sizeof_bits_v<ElementType>;
    constexpr int TileKBits = mute::size<2>(TileShape{}) * ElementBits;
    static_assert(SmemAtomBits % ElementBits == 0,
                  "Element type must evenly divide a 256-byte SMEM atom");
    constexpr int MaxAtomK = SmemAtomBits / ElementBits;
    using SmemAtomK = mute::conditional_t<
        TileKBits % SmemAtomBits == 0,
        mute::Int<mute::min(MaxAtomK, mute::tuple_element_t<2, TileShape>{})>,
        AtomK>;
    return mute::MP32::SQMMA::make_canonical_gemm_smem_atom_layout<
        ElementType, Major, mute::tuple_element_t<1, TileShape>, SmemAtomK>();
  }
}

template <mute::TCE::Major Major, class ElementType, class SqmmaOp, class TileShape>
MUTE_HOST_DEVICE constexpr auto
mp32_rs_smem_selector_A() {
  static_assert(Major == mute::TCE::Major::K,
                "MP32 RS register-A shared layout requires K-major storage");
  return mute::MP32::SQMMA::Layout_SL128_SS128_SG16_Atom<
      mute::size<0>(TileShape{}),
      mute::size<2>(TileShape{}),
      ElementType,
      Major>{};
}

template <mute::TCE::Major Major, class ElementType, class SqmmaOp, class TileShape>
MUTE_HOST_DEVICE constexpr auto
mp32_rs_smem_selector_B() {
  return mp32_ss_smem_selector_B<Major, ElementType, SqmmaOp, TileShape>();
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA_TME_SS_WS
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
  arch::Mp32,
  arch::OpClassTensorOp,
  ElementA,
  GmemLayoutATag,
  AlignmentA,
  ElementB,
  GmemLayoutBTag,
  AlignmentB,
  ElementAccumulator,
  TileShape_MNK,
  ClusterShape_MNK,
  StageCountType,
  KernelScheduleType,
  mute::enable_if_t<
      detail::is_mp32_warp_specialized_schedule<KernelScheduleType>()>
> {
  static_assert(mute::is_static<TileShape_MNK>::value);
  static_assert(mute::is_static<ClusterShape_MNK>::value);
  static_assert(mute::is_same_v<ClusterShape_MNK, mute::Shape<mute::_1, mute::_1, mute::_1>>,
                "MP32 currently supports only a 1x1x1 cluster shape");

  static_assert(
      detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tme_alignment_bytes>(),
      "Should meet TME alignment requirement\n");

  using BuilderSchedule = detail::mp32_kernel_schedule_t<KernelScheduleType>;

  using ElementAMma = mute::conditional_t<mute::is_same_v<ElementA, float>, mute::tfloat32_t, ElementA>;
  using ElementBMma = mute::conditional_t<mute::is_same_v<ElementB, float>, mute::tfloat32_t, ElementB>;

  static constexpr mute::TCE::Major SqmmaMajorA =
      detail::sqmma_ss_tag_to_major_A<GmemLayoutATag>();
  static constexpr mute::TCE::Major SqmmaMajorB =
      detail::sqmma_ss_tag_to_major_B<GmemLayoutBTag>();

  using SqmmaOp = decltype(mute::MP32::SQMMA::ss_op_selector<
      ElementAMma,
      ElementBMma,
      ElementAccumulator,
      TileShape_MNK,
      SqmmaMajorA,
      SqmmaMajorB>());

  using TiledMma = decltype(detail::make_sqmma_tiled_mma<TileShape_MNK, SqmmaOp>());

  static_assert(
      !detail::is_mp32_pingpong_schedule<BuilderSchedule>() ||
          mute::size(TiledMma{}) <= 256,
      "MP32 ping-pong requires a TiledMma with at most 256 threads");

  using GmemTiledCopyA = mute::MP31_TME_LOAD;
  using GmemTiledCopyB = mute::MP31_TME_LOAD;

  using SmemLayoutAtomA = decltype(detail::mp32_ss_smem_selector_A<
      SqmmaMajorA, ElementAMma, SqmmaOp, TileShape_MNK>());
  using SmemLayoutAtomB = decltype(detail::mp32_ss_smem_selector_B<
      SqmmaMajorB, ElementBMma, SqmmaOp, TileShape_MNK>());

  using SmemCopyAtomA = void;
  using SmemCopyAtomB = void;

  static constexpr int StageBytes = detail::mp32_dense_stage_bytes<
      ElementAMma, ElementBMma, TileShape_MNK>();
  static constexpr int PipelineStages = detail::compute_mp32_stage_count_or_override<
      detail::mp32_smem_capacity_bytes,
      StageBytes,
      detail::mp32_max_pipeline_stages<BuilderSchedule>()>(StageCountType{});

  using DispatchPolicy = MainloopMp32TmeSqmmaWarpSpecialized<PipelineStages, BuilderSchedule>;
  using ClusterShape_MNK_ = mute::Shape<mute::_1, mute::_1, mute::_1>;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      mute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      mute::identity>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA_TME_SS_WS_BLOCKSCALED
template <
  class ElementPairA_,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementPairB_,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
  arch::Mp32,
  arch::OpClassBlockScaledTensorOp,
  ElementPairA_,
  GmemLayoutATag,
  AlignmentA,
  ElementPairB_,
  GmemLayoutBTag,
  AlignmentB,
  ElementAccumulator,
  TileShape_MNK,
  ClusterShape_MNK,
  StageCountType,
  KernelScheduleType,
  mute::enable_if_t<
      detail::is_mp32_warp_specialized_schedule<KernelScheduleType>() &&
      detail::is_mp32_blockscaled_ss_pair_supported<
          ElementPairA_, ElementPairB_>()>
> {
  static_assert(mute::is_static<TileShape_MNK>::value);
  static_assert(mute::is_static<ClusterShape_MNK>::value);
  static_assert(mute::is_same_v<ClusterShape_MNK, mute::Shape<mute::_1, mute::_1, mute::_1>>,
                "MP32 currently supports only a 1x1x1 cluster shape");

  using ElementPairATag = ElementPairA_;
  using ElementPairBTag = ElementPairB_;
  using ElementARaw = typename ElementPairATag::DataType;
  using ElementBRaw = typename ElementPairBTag::DataType;
  using ElementA = detail::mp32_blockscaled_element_t<ElementARaw>;
  using ElementB = detail::mp32_blockscaled_element_t<ElementBRaw>;
  using ElementSFA = typename ElementPairATag::ScaleFactorType;
  using ElementSFB = typename ElementPairBTag::ScaleFactorType;
  using ElementSF = ElementSFA;
  using BuilderSchedule = detail::mp32_kernel_schedule_t<KernelScheduleType>;

  static constexpr int ScaleVectorSize = detail::mp32_blockscaled_sf_vector_size;
  static_assert(
      detail::is_mp32_blockscaled_pair_supported<
          ElementA, ElementB, ElementSFA, ElementSFB>(),
      "Unsupported MP32 block-scaled operand or scale-factor combination");
  static_assert(
      detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB,
                         detail::tme_alignment_bytes>(),
      "Should meet TME alignment requirement\n");

  static constexpr mute::TCE::Major SqmmaMajorA =
      detail::sqmma_ss_tag_to_major_A<GmemLayoutATag>();
  static constexpr mute::TCE::Major SqmmaMajorB =
      detail::sqmma_ss_tag_to_major_B<GmemLayoutBTag>();
  static_assert(
      detail::is_mp32_blockscaled_major_supported<ElementA, SqmmaMajorA>(),
      "MP32 block-scaled F4 operand A requires a K-major layout");
  static_assert(
      detail::is_mp32_blockscaled_major_supported<ElementB, SqmmaMajorB>(),
      "MP32 block-scaled F4 operand B requires a K-major layout");

  using SqmmaOp = decltype(mute::MP32::SQMMA::ss_op_selector<
      ElementA,
      ElementB,
      ElementAccumulator,
      ElementSF,
      TileShape_MNK,
      SqmmaMajorA,
      SqmmaMajorB>());
  using TiledMma = decltype(detail::make_sqmma_tiled_mma<TileShape_MNK, SqmmaOp>());

  static_assert(
      !detail::is_mp32_pingpong_schedule<BuilderSchedule>() ||
          mute::size(TiledMma{}) <= 256,
      "MP32 ping-pong requires a TiledMma with at most 256 threads");

  using GmemTiledCopyA = mute::MP31_TME_LOAD;
  using GmemTiledCopyB = mute::MP31_TME_LOAD;

  using SmemLayoutAtomA = decltype(detail::mp32_ss_smem_selector_A<
      SqmmaMajorA, ElementA, SqmmaOp, TileShape_MNK>());
  using SmemLayoutAtomB = decltype(detail::mp32_ss_smem_selector_B<
      SqmmaMajorB, ElementB, SqmmaOp, TileShape_MNK>());
  using LayoutSFA = detail::mp32_scale_factor_layout<
      mute::size<0>(TileShape_MNK{}), mute::size<2>(TileShape_MNK{})>;
  using LayoutSFB = detail::mp32_scale_factor_layout<
      mute::size<1>(TileShape_MNK{}), mute::size<2>(TileShape_MNK{})>;

  using StrideA = TagToStrideA_t<GmemLayoutATag>;
  using StrideB = TagToStrideB_t<GmemLayoutBTag>;
  using StridePairA = mute::tuple<StrideA, LayoutSFA>;
  using StridePairB = mute::tuple<StrideB, LayoutSFB>;
  using ElementPairA = mute::tuple<ElementA, ElementSFA>;
  using ElementPairB = mute::tuple<ElementB, ElementSFB>;

  using SmemCopyAtomA = void;
  using SmemCopyAtomB = void;

  static constexpr int StageBytes = detail::mp32_dense_stage_bytes<
      ElementA, ElementB, TileShape_MNK>();
  static constexpr int PipelineStages =
      detail::compute_mp32_stage_count_or_override<
          detail::mp32_smem_capacity_bytes,
          StageBytes,
          detail::mp32_max_pipeline_stages<BuilderSchedule>()>(StageCountType{});

  using DispatchPolicy = MainloopMp32TmeSqmmaWarpSpecializedBlockScaled<
      PipelineStages, BuilderSchedule>;
  using ClusterShape_MNK_ = mute::Shape<mute::_1, mute::_1, mute::_1>;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementPairA,
      StridePairA,
      ElementPairB,
      StridePairB,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      mute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      mute::identity>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// SQMMA_TME_RS_WS_BLOCKSCALED_MIXED_INPUT
template <
  class ElementPairA_,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementPairB_,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
  arch::Mp32,
  arch::OpClassBlockScaledTensorOp,
  ElementPairA_,
  GmemLayoutATag,
  AlignmentA,
  ElementPairB_,
  GmemLayoutBTag,
  AlignmentB,
  ElementAccumulator,
  TileShape_MNK,
  ClusterShape_MNK,
  StageCountType,
  KernelScheduleType,
  mute::enable_if_t<
      detail::is_mp32_blockscaled_mixed_input_schedule<KernelScheduleType>() &&
      detail::is_mp32_blockscaled_mixed_input_pair_supported<
          ElementPairA_, ElementPairB_>::value>
> {
  static_assert(mute::is_static<TileShape_MNK>::value);
  static_assert(mute::is_static<ClusterShape_MNK>::value);
  static_assert(mute::is_same_v<ClusterShape_MNK, mute::Shape<mute::_1, mute::_1, mute::_1>>,
                "MP32 currently supports only a 1x1x1 cluster shape");

  using ElementPairATag = ElementPairA_;
  using ElementPairBTag = ElementPairB_;
  using ElementA = typename ElementPairATag::DataType;
  using ElementB = typename ElementPairBTag::DataType;
  using ElementSFA = typename ElementPairATag::ScaleFactorType;
  using ElementSFB = typename ElementPairBTag::ScaleFactorType;
  static_assert(mute::is_same_v<ElementSFA, ElementSFB>,
                "MP32 mixed-input requires identical scale factor types");
  using ElementSF = ElementSFA;
  using BuilderSchedule =
      detail::mp32_blockscaled_mixed_input_kernel_schedule_t<KernelScheduleType>;

  static constexpr bool IsATransformed = mute::is_same_v<ElementPairATag, mx_float4_t<ElementA>>;
  static constexpr bool SwapAB = !IsATransformed;

  static_assert(mute::is_same_v<ElementSFA, float_ue8m0_t> &&
                mute::is_same_v<ElementSFB, float_ue8m0_t>,
                "MP32 WF4AFP8 requires UE8M0 scale factors");

  static constexpr int ScaleVectorSize = detail::mp32_blockscaled_sf_vector_size;
  static_assert(mute::size<2>(TileShape_MNK{}) % ScaleVectorSize == 0,
                "MP32 mixed-input Tile_K must be divisible by SFVecSize");
  static_assert(
      detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB,
                         detail::tme_alignment_bytes>(),
      "Should meet TME alignment requirement\n");

  static constexpr mute::TCE::Major SqmmaMajorA = detail::sqmma_ss_tag_to_major_A<GmemLayoutATag>();
  static constexpr mute::TCE::Major SqmmaMajorB = detail::sqmma_ss_tag_to_major_B<GmemLayoutBTag>();
  static_assert((SwapAB ? SqmmaMajorB : SqmmaMajorA) == mute::TCE::Major::K,
                "MP32 mixed input requires the transformed FP4 operand to be K-major");

  // Place the transformed FP4 operand on the register side of RS SQMMA.
  static constexpr mute::TCE::Major TiledMmaSqmmaMajorB = SwapAB ? SqmmaMajorA : SqmmaMajorB;
  using RealElementA = mute::conditional_t<SwapAB, ElementB, ElementA>;
  using RealElementB = mute::conditional_t<SwapAB, ElementA, ElementB>;
  using RealElementAMma = RealElementB;
  using RealElementBMma = RealElementB;

  // Swapping computes C^T in (N,M,K); otherwise retain public (M,N,K).
  using RsTileShape_MNK = mute::conditional_t<SwapAB, decltype(mute::select<1, 0, 2>(TileShape_MNK{})), TileShape_MNK>;

  // Convert the FP4 operand to the other operand's FP8 format for RS SQMMA.
  using SqmmaOp = decltype(mute::MP32::SQMMA::rs_op_selector<
      RealElementAMma,
      RealElementBMma,
      ElementAccumulator,
      ElementSF,
      RsTileShape_MNK,
      mute::TCE::Major::K,
      TiledMmaSqmmaMajorB,
      mute::_64,
      mute::_128>());
  using RsAtomShape_MNK = typename mute::MMA_Traits<SqmmaOp>::Shape_MNK;
  static constexpr int RsTileM = mute::size<0>(RsTileShape_MNK{});
  static constexpr int RsTileN = mute::size<1>(RsTileShape_MNK{});
  static constexpr int RsAtomM = mute::size<0>(RsAtomShape_MNK{});
  static constexpr int RsAtomN = mute::size<1>(RsAtomShape_MNK{});

  // Reuse register A across two instruction tiles in N when the CTA permits.
  static constexpr int RsSquadTileN = RsTileN % (2 * RsAtomN) == 0 ? 2 * RsAtomN : RsAtomN;
  static_assert(
      mute::size<2>(RsTileShape_MNK{}) == mute::size<2>(RsAtomShape_MNK{}),
      "MP32 RS warp-specialized mainloop requires one instruction K block per stage");
  using RsSquadLayout = decltype(mute::make_layout(
      mute::Shape<mute::Int<RsTileM / RsAtomM>,
                  mute::Int<RsTileN / RsSquadTileN>, mute::_1>{},
      mute::LayoutRight{}));

  // Expose the entire RS CTA tile to fragment allocation and the epilogue.
  // A 256x256 public tile needs four 64x256 squads; 256x128 needs two.
  using TiledMma = decltype(mute::make_tiled_mma(
      SqmmaOp{}, RsSquadLayout{},
      mute::Tile<mute::Int<RsTileM>, mute::Int<RsTileN>, mute::Underscore>{}));

  // Include the producer squad in the CTA thread budget.
  static_assert(mutlass::NumThreadsPerWarpSquad + mute::size(TiledMma{}) <= 1024,
                "MP32 RS warp-specialized tile exceeds 1024 threads per CTA");

  using GmemTiledCopyA = mute::MP31_TME_LOAD;
  using GmemTiledCopyB = mute::MP31_TME_LOAD;
  using GmemTiledCopySFA = mute::MP31_TME_LOAD;
  using GmemTiledCopySFB = mute::MP31_TME_LOAD;
  using GmemTiledCopyPairA = mute::tuple<GmemTiledCopyA, GmemTiledCopySFA>;
  using GmemTiledCopyPairB = mute::tuple<GmemTiledCopyB, GmemTiledCopySFB>;

  using RsSmemLayoutAtomA = decltype(detail::mp32_rs_smem_selector_A<
      mute::TCE::Major::K, RealElementA, SqmmaOp, RsTileShape_MNK>());
  using RsSmemLayoutAtomB = decltype(detail::mp32_rs_smem_selector_B<
      TiledMmaSqmmaMajorB, RealElementB, SqmmaOp, RsTileShape_MNK>());
  // Keep public layout atoms paired with their public operand.
  using SmemLayoutAtomA = mute::conditional_t<SwapAB, RsSmemLayoutAtomB, RsSmemLayoutAtomA>;
  using SmemLayoutAtomB = mute::conditional_t<SwapAB, RsSmemLayoutAtomA, RsSmemLayoutAtomB>;

  // These layouts remain attached to the public operands, even though RS
  // swaps their SQMMA roles below.
  using SmemLayoutAtomSFA = detail::mp32_scale_factor_layout<
      mute::size<0>(TileShape_MNK{}), mute::size<2>(TileShape_MNK{})>;
  using SmemLayoutAtomSFB = detail::mp32_scale_factor_layout<
      mute::size<1>(TileShape_MNK{}), mute::size<2>(TileShape_MNK{})>;

  // Template arguments and host Arguments keep public A/B order. The
  // collective maps types and Params to RS order before device execution.
  // Scale factors stay paired with their data operand through that mapping.
  using SmemLayoutAtomsA =
      mute::tuple<SmemLayoutAtomA, SmemLayoutAtomSFA>;
  using SmemLayoutAtomsB =
      mute::tuple<SmemLayoutAtomB, SmemLayoutAtomSFB>;

  using StrideA = TagToStrideA_t<GmemLayoutATag>;
  using StrideB = TagToStrideB_t<GmemLayoutBTag>;
  using StridePairA = mute::tuple<StrideA, SmemLayoutAtomSFA>;
  using StridePairB = mute::tuple<StrideB, SmemLayoutAtomSFB>;
  using ElementPairA = mute::tuple<ElementA, ElementSFA>;
  using ElementPairB = mute::tuple<ElementB, ElementSFB>;

  using SmemCopyAtomSFA =
      mute::Copy_Atom<mute::UniversalCopy<ElementSFA>, ElementSFA>;
  using SmemCopyAtomSFB =
      mute::Copy_Atom<mute::UniversalCopy<ElementSFB>, ElementSFB>;
  using SmemCopyAtomA = mute::tuple<mute::identity, SmemCopyAtomSFA>;
  using SmemCopyAtomB = mute::tuple<mute::identity, SmemCopyAtomSFB>;

  static constexpr int StageBytes =
    detail::mp32_blockscaled_stage_bytes<
        RealElementA, RealElementB, ElementSFA, ElementSFB, RsTileShape_MNK>();

  static constexpr int PipelineStages =
    detail::compute_mp32_stage_count_or_override<
        detail::mp32_smem_capacity_bytes,
        StageBytes,
        detail::mp32_max_pipeline_stages<BuilderSchedule>()>(StageCountType{});

  static_assert(PipelineStages >= 2,
                "MP32 RS two-stage lookahead requires at least two pipeline stages");

  using DispatchPolicy =
      MainloopMp32TmeSqmmaWarpSpecializedBlockScaledMixedInput<
          PipelineStages, BuilderSchedule>;
  using ClusterShape_MNK_ = mute::Shape<mute::_1, mute::_1, mute::_1>;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementPairA,
      StridePairA,
      ElementPairB,
      StridePairB,
      TiledMma,
      GmemTiledCopyPairA,
      SmemLayoutAtomsA,
      SmemCopyAtomA,
      mute::identity,
      GmemTiledCopyPairB,
      SmemLayoutAtomsB,
      SmemCopyAtomB,
      mute::identity>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
