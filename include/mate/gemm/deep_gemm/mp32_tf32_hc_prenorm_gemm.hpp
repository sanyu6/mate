#pragma once

#include <mutlass/mutlass.h>

#include <mute/atom/copy_traits_mp32_tme.hpp>
#include <mute/atom/mma_traits_mp32_sqmma.hpp>
#include <mute/tensor.hpp>
#include <mutlass/pipeline/pipeline.hpp>

#include "mate/common/layout.hpp"

namespace mate::deep_gemm {

using namespace mute;

namespace detail {

template <class Threshold, class Source, class Reference>
MUTE_HOST_DEVICE constexpr auto layout_separate(Threshold const& threshold,
                                                Source const&    source,
                                                Reference const& reference) {
  auto lower = filter(transform_layout(source, reference, [&](auto const& src, auto const& ref) {
    if constexpr (decltype(ref < threshold)::value) {
      return src;
    } else {
      return make_layout(_1{}, _0{});
    }
  }));

  auto upper = filter(transform_layout(source, reference, [&](auto const& src, auto const& ref) {
    if constexpr (decltype(ref >= threshold)::value) {
      return src;
    } else {
      return make_layout(_1{}, _0{});
    }
  }));
  return make_tuple(lower, upper);
}

template <class TiledMma, class FragmentLayout>
MUTE_HOST_DEVICE constexpr auto layout_a_mk(TiledMma const& tiled_mma, FragmentLayout const& fragment) {
  auto [value_m, value_k] = layout_separate(
      get<0>(typename TiledMma::Shape_MNK{}), get<0>(fragment), stride<1>(typename TiledMma::LayoutA_TV{}));
  return make_layout(make_layout(value_m, get<1>(fragment)), make_layout(value_k, get<2>(fragment)));
}

template <class TiledMma>
MUTE_HOST_DEVICE constexpr auto reduction_target_k(TiledMma const& tiled_mma) {
  auto separated = layout_separate(get<0>(typename TiledMma::Shape_MNK{}),
                                   make_layout(shape<0>(typename TiledMma::LayoutA_TV{})),
                                   stride<0>(typename TiledMma::LayoutA_TV{}));
  return get<1>(separated);
}

}  // namespace detail

template <typename ElementA_,
          typename ElementB_,
          typename ElementD_,
          typename StrideA_,
          typename StrideB_,
          typename StrideD_,
          typename StrideSqrSum_,
          typename TileShape_,
          uint32_t kStages_,
          uint32_t kNumMmaWarpSquads_>
struct Mp32Tf32HcPrenormGemm {
  using ElementA = ElementA_;
  // B arrives as FP32 and is consumed as TF32 throughout; TF32 shares FP32's 32-bit container.
  using ElementBInput = ElementB_;
  using ElementB      = mutlass::tfloat32_t;
  using ElementD      = ElementD_;
  using StrideA       = StrideA_;
  using StrideB       = StrideB_;
  using StrideD       = StrideD_;
  using StrideSqrSum  = StrideSqrSum_;
  using TileShape     = TileShape_;

  static constexpr uint32_t kStages           = kStages_;
  static constexpr uint32_t kNumMmaWarpSquads = kNumMmaWarpSquads_;

  static constexpr int BlockM = get<0>(TileShape{});
  static constexpr int BlockN = get<1>(TileShape{});
  static constexpr int BlockK = get<2>(TileShape{});

  static_assert(kNumMmaWarpSquads > 0, "MP32 HC prenorm requires at least one GEMM squad");
  static_assert(BlockM % int(kNumMmaWarpSquads) == 0, "BlockM must divide evenly across the GEMM squads");
  static constexpr int SquadTileM = BlockM / int(kNumMmaWarpSquads);
  static_assert(SquadTileM % 32 == 0, "Each MP32 GEMM squad must cover a multiple of 32 rows");
  static constexpr int MaxInstructionM = SquadTileM % 128 == 0 ? 128 : SquadTileM % 64 == 0 ? 64 : 32;
  // K32 decomposition for M64/M128: the native K64 MATE path has not been revalidated end to end.
  static constexpr int MaxInstructionKBytes = MaxInstructionM == 32 ? 256 : 128;
  static_assert(kStages >= 2, "MP32 HC prenorm requires at least two pipeline stages");
  static_assert(is_same_v<ElementA, mutlass::bfloat16_t>, "MP32 HC prenorm requires BF16 A");
  static_assert(is_same_v<ElementB_, float>, "MP32 HC prenorm requires an FP32 B input");
  static_assert(is_same_v<ElementD, float>, "MP32 HC prenorm requires an FP32 D");

  using ElementMma         = mutlass::tfloat32_t;
  using ElementAccumulator = float;

  using SqmmaOp       = decltype(mute::MP32::SQMMA::rs_op_selector<ElementMma,
                                                                   ElementMma,
                                                                   ElementAccumulator,
                                                                   TileShape,
                                                                   TCE::Major::K,
                                                                   TCE::Major::K,
                                                                   Int<MaxInstructionM>,
                                                                   _128,
                                                                   Int<MaxInstructionKBytes>>());
  using MmaAtomLayout = Layout<Shape<Int<kNumMmaWarpSquads>, _1, _1>>;
  using TiledMma      = decltype(make_tiled_mma(SqmmaOp{}, MmaAtomLayout{}));
  using MmaAtomShape  = typename TiledMma::AtomShape_MNK;

  static constexpr int MmaM = get<0>(MmaAtomShape{});
  static constexpr int MmaN = get<1>(MmaAtomShape{});
  static constexpr int MmaK = get<2>(MmaAtomShape{});

  using StoreVector                  = mute::uint128_t;
  static constexpr int StoreElements = sizeof_bits_v<StoreVector> / sizeof_bits_v<ElementD>;
  // N-lane span derived from the atom rather than hardcoded.
  static constexpr int MmaNThreadSpan =
      decltype(mute::MP32::detail::deduce_mma_n_thread_span<typename TiledMma::Atom>())::value;
  static constexpr int MmaNPermuteRepeats = MmaN / (MmaNThreadSpan * StoreElements);

  static_assert(MmaN % (MmaNThreadSpan * StoreElements) == 0, "MP32 SQMMA N must divide into complete store vectors");

  // Reorder N so each thread's StoreElements consecutive accumulator registers map to consecutive D columns.
  using MmaPermuteN     = decltype(filter(make_ordered_layout(
      Shape<Int<MmaNThreadSpan>, Int<StoreElements>, Int<MmaNPermuteRepeats>>{}, Step<_2, _1, _3>{})));
  using MmaPermutation  = Tile<Underscore, MmaPermuteN, Underscore>;
  using PermuteTiledMma = decltype(mate::convert_to_permuted_sqmma(TiledMma{}, MmaPermutation{}));

  static_assert(decltype(mute::MP32::detail::deduce_tme_permutation_stride(PermuteTiledMma{}))::value == StoreElements,
                "MP32 SQMMA N permutation must match the store vector width");

  static_assert(MmaM == MaxInstructionM, "Unexpected MP32 SQMMA M instruction shape");
  static_assert(BlockM % (int(kNumMmaWarpSquads) * MmaM) == 0 && BlockN % MmaN == 0 && BlockK % MmaK == 0,
                "The MP32 SQMMA atom layout must divide the CTA tile");

  static constexpr uint32_t NumLoadWarpSquads      = 1;
  static constexpr uint32_t NumGemmSquads          = kNumMmaWarpSquads;
  static constexpr uint32_t NumThreadsPerWarp      = mutlass::NumThreadsPerWarp;
  static constexpr uint32_t NumThreadsPerWarpSquad = mutlass::NumThreadsPerWarpSquad;
  static constexpr uint32_t WarpsPerWarpSquad      = NumThreadsPerWarpSquad / NumThreadsPerWarp;
  static constexpr uint32_t NumConsumerWarps       = NumGemmSquads * WarpsPerWarpSquad;
  static constexpr uint32_t MaxThreadsPerBlock     = (NumLoadWarpSquads + NumGemmSquads) * NumThreadsPerWarpSquad;
  // Two CTAs per MP: 640 threads x 2 leaves 131072/1280 = 102 -> 96 registers per thread, and a third
  // CTA fits neither the register file nor the staging smem. Class-scope, so it covers every MP32 config.
  static constexpr uint32_t MinBlocksPerMultiprocessor = 2;
  // Swizzle<3,4,3> touches byte-address bits 4..9, so a 1 KiB-aligned storage base lets the
  // position-independent consumer below apply the swizzle to layout offsets instead of to the running
  // address. Publishing it here is also what makes mutlass::device_kernel align its dynamic smem base.
  static constexpr int SmemAlignmentBytes = 1024;

  static_assert(size(TiledMma{}) == NumGemmSquads * NumThreadsPerWarpSquad,
                "TiledMma must contain exactly one thread squad per M tile");
  static_assert(MaxThreadsPerBlock <= 1024, "MP32 supports at most 1024 threads per block");

  // SG16 granularity with swizzle stride 128 / line 128 = Swizzle<3,4,3>.
  using SmemLayoutAtomA = decltype(upcast<sizeof_bits_v<ElementA>>(
      ComposedLayout<Swizzle<3, 4, 3>,
                     smem_ptr_flag,
                     mute::MP32::SQMMA::Layout_Atom_Bits<MmaM, MmaK, ElementA, TCE::Major::K>>{}));
  using SmemLayoutAtomB =
      decltype(mute::MP32::SQMMA::
                   make_canonical_gemm_smem_atom_layout<ElementB, TCE::Major::K, Int<MmaN>, Int<MmaK>>());
  using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, append<3>(select<0, 2>(TileShape{}), Int<kStages>{})));
  using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, append<3>(select<1, 2>(TileShape{}), Int<kStages>{})));
  // TME load cache hints: A is streamed through once (outer CACHE_NONE), B is reused across M blocks
  // (outer CACHE_PERSIST); both use CACHE_NORMAL for the inner field.
  static constexpr TME::CacheHint TmeHintAInner = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeHintAOuter = TME::CacheHint::CACHE_NONE;
  static constexpr TME::CacheHint TmeHintBInner = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeHintBOuter = TME::CacheHint::CACHE_PERSIST;
  using TME_A                                   = decltype(make_tme_copy<void, TmeHintAInner, TmeHintAOuter>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementA const*>(nullptr)), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
      take<0, 2>(SmemLayoutA{}),
      select<0, 2>(TileShape{}),
      TiledMma{}));
  using TME_B                                   = decltype(make_tme_copy<void, TmeHintBInner, TmeHintBOuter>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementB const*>(nullptr)), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
      take<0, 2>(SmemLayoutB{}),
      select<1, 2>(TileShape{}),
      PermuteTiledMma{}));

  // ---- TME-store epilogue --------------------------------------------------------------------
  // The (BlockM x BlockN) fp32 accumulator tile is staged in smem by the consumer warps and drained by
  // the GEMM squads, each owning SquadTileM consecutive rows of it. The staging layout is built like the
  // A operand's TME layout (the same Swizzle<3,4,3> atom upcast to the element type), so the descriptor's
  // swizzle and the register->smem copy's swizzle come from the same layout object. One buffer suffices:
  // the tile is staged once and drained once.
  static constexpr int EpiStages = 1;
  using EpiSmemAtomD             = decltype(upcast<sizeof_bits_v<ElementD>>(
      ComposedLayout<Swizzle<3, 4, 3>,
                                 smem_ptr_flag,
                                 mute::MP32::SQMMA::Layout_Atom_Bits<BlockM, BlockN, ElementD, TCE::Major::K>>{}));
  using EpiSmemLayoutD = decltype(tile_to_shape(EpiSmemAtomD{}, Shape<Int<BlockM>, Int<BlockN>, Int<EpiStages>>{}));
  // 16-byte register->smem vector copy on the same permuted accumulator tiling the store uses.
  using EpiR2SCopyAtom  = Copy_Atom<UniversalCopy<StoreVector>, ElementD>;
  using EpiR2STiledCopy = decltype(make_tiled_copy_C(EpiR2SCopyAtom{}, PermuteTiledMma{}));

  // Per-squad write-back: a squad owns a contiguous SquadTileM-row slice, so it can describe that slice
  // with the same SW128 atom at a squad-local base. The slice base is a multiple of 4 KiB, so
  // swizzle(base + off) == base + swizzle(off).
  using EpiSmemAtomDSquad = decltype(upcast<sizeof_bits_v<ElementD>>(
      ComposedLayout<Swizzle<3, 4, 3>,
                     smem_ptr_flag,
                     mute::MP32::SQMMA::Layout_Atom_Bits<SquadTileM, BlockN, ElementD, TCE::Major::K>>{}));
  using EpiSmemLayoutDSquad =
      decltype(tile_to_shape(EpiSmemAtomDSquad{}, Shape<Int<SquadTileM>, Int<BlockN>, Int<EpiStages>>{}));
  static constexpr int EpiSquadStageElems = SquadTileM * BlockN * EpiStages;
  // D is a [S][M][N] allocation, addressed here as a rank-3 (m, n, num_splits) view -- the same order
  // Arguments::stride_d uses. A TME box and its smem layout must have the same top-level shape, so the box's
  // two leading modes are the (m, n) tile a squad drains and the split rides along as the trailing mode.
  // Keeping the split a real mode rather than folding it into the row index is what stops a partial M tile
  // from running into the neighbouring split's rows.
  using TME_D_Squad = decltype(make_tme_copy(
      MP31_TME_STORE{},
      make_tensor(make_gmem_ptr(static_cast<ElementD*>(nullptr)), repeat_like(StrideD{}, int32_t(0)), StrideD{}),
      EpiSmemLayoutDSquad{},
      Shape<Int<SquadTileM>, Int<BlockN>, Int<EpiStages>>{}));

  static constexpr uint32_t TmeTransactionBytesA =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * sizeof_bits_v<ElementA>);
  static constexpr uint32_t TmeTransactionBytesB =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * sizeof_bits_v<ElementB>);

  // MP32 pipeline: the plain producer-consumer ring from mutlass/pipeline/pipeline.hpp (NumBarriers =
  // 2*kStages, empty barrier = stage + Stages), the same shape mutlass' own MP32 SQMMA collectives use. The
  // generalized Mp31PipelineTmeAsync defaults to BarPerStageRatio = 2, which the MP32 ring does not need.
  using PipelineMain  = mutlass::PipelineTmeAsync<kStages>;
  using PipelineState = typename PipelineMain::PipelineState;

  // One extra async barrier per GEMM squad for the write-back rendezvous: MP32 has no barrier that
  // synchronizes only part of the block, so a subset of the warps can only rendezvous through the TME
  // async barriers. Laying them out after the pipeline keeps the two id sets disjoint.
  using EpiSquadBarrier                    = mutlass::arch::AsyncBarrier;
  static constexpr uint32_t NumEpiBarriers = NumGemmSquads;

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t PipelineMain[PipelineMain::NumBarriers];
    uint8_t Epilogue[NumEpiBarriers];
  };
  static_assert(sizeof(BarrierStorage) + mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount <=
                    mutlass::arch::AsyncBarrier::HardwareMaxNumAsyncTransactionBarriers,
                "MP32 HC prenorm exceeds the async barrier budget");

  struct SharedStorage {
    mute::array_aligned<ElementA, cosize_v<SmemLayoutA>, SmemAlignmentBytes>    smem_a;
    mute::array_aligned<ElementB, cosize_v<SmemLayoutB>, SmemAlignmentBytes>    smem_b;
    mute::array_aligned<ElementD, cosize_v<EpiSmemLayoutD>, SmemAlignmentBytes> smem_d;
  };

  struct Arguments {
    ElementA const*      ptr_a;
    ElementBInput const* ptr_b;
    ElementD*            ptr_d;
    float*               ptr_sqr_sum;
    StrideA              stride_a;
    StrideB              stride_b;
    StrideD              stride_d;
    StrideSqrSum         stride_sqr_sum;
    int                  m, n, k;
    int                  num_splits;
  };

  struct Params {
    TME_A        tme_a;
    TME_B        tme_b;
    TME_D_Squad  tme_d_squad;
    float*       ptr_sqr_sum;
    StrideSqrSum stride_sqr_sum;
    int          m, n, k;
    int          num_splits;
    int          num_m_blocks;
    int          num_n_blocks;
    int          num_k_blocks_per_split;
    int          remain_k_blocks;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    // The same bytes viewed as TF32, so the TME descriptor is encoded as TFLOAT32.
    auto mB = recast<ElementB>(make_tensor(make_gmem_ptr(args.ptr_b), make_shape(args.n, args.k), args.stride_b));

    auto  mA    = make_tensor(make_gmem_ptr(args.ptr_a), make_shape(args.m, args.k), args.stride_a);
    TME_A tme_a = make_tme_copy<void, TmeHintAInner, TmeHintAOuter>(
        MP31_TME_LOAD{}, mA, take<0, 2>(SmemLayoutA{}), select<0, 2>(TileShape{}), TiledMma{});
    TME_B tme_b = make_tme_copy<void, TmeHintBInner, TmeHintBOuter>(
        MP31_TME_LOAD{}, mB, take<0, 2>(SmemLayoutB{}), select<1, 2>(TileShape{}), PermuteTiledMma{});

    const int num_splits             = max(args.num_splits, 1);
    const int num_m_blocks           = ceil_div(args.m, BlockM);
    const int num_n_blocks           = ceil_div(args.n, BlockN);
    const int num_k_blocks           = ceil_div(args.k, BlockK);
    const int num_k_blocks_per_split = num_k_blocks / num_splits;
    const int remain_k_blocks        = num_k_blocks % num_splits;

    // The runtime tensor must reproduce the type-level one exactly (make_tme_copy's return type depends
    // on the gmem layout type), so the shape stays Shape<int32_t,int32_t,int32_t> like repeat_like.
    auto        mD = make_tensor(make_gmem_ptr(args.ptr_d), make_shape(args.m, args.n, int(num_splits)), args.stride_d);
    TME_D_Squad tme_d_squad = make_tme_copy(
        MP31_TME_STORE{}, mD, EpiSmemLayoutDSquad{}, Shape<Int<SquadTileM>, Int<BlockN>, Int<EpiStages>>{});

    return Params{tme_a,
                  tme_b,
                  tme_d_squad,
                  args.ptr_sqr_sum,
                  args.stride_sqr_sum,
                  args.m,
                  args.n,
                  args.k,
                  num_splits,
                  num_m_blocks,
                  num_n_blocks,
                  num_k_blocks_per_split,
                  remain_k_blocks};
  }

  static constexpr int SharedStorageSize = int(sizeof(SharedStorage));

  static dim3 get_grid_shape(Params const& p) {
    return dim3(uint32_t(p.num_splits), uint32_t(p.num_n_blocks), uint32_t(p.num_m_blocks));
  }

  MUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem);

    // A barrier id is its byte offset from zero, so the storage pointer doubles as the id space.
    auto* bs = reinterpret_cast<BarrierStorage*>(0);
    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));

    typename PipelineMain::Params pipeline_params;
    pipeline_params.transaction_bytes = TmeTransactionBytesA + TmeTransactionBytesB;
    pipeline_params.num_consumers     = NumConsumerWarps;
    pipeline_params.num_producers     = 1;
    PipelineMain pipeline_main(pipeline_params, static_cast<uint32_t>(reinterpret_cast<uint64_t>(&bs->PipelineMain)));

    const int tid           = int(threadIdx.x);
    const int squad         = mutlass::canonical_warp_squad_idx();
    const int warp_idx      = mutlass::canonical_warp_idx_sync();
    const int warp_in_squad = warp_idx % int(WarpsPerWarpSquad);

    // Loader squads come first; the write-back keys off the same constant.
    const bool is_producer_squad = squad < int(NumLoadWarpSquads);
    const bool is_tme_warp       = is_producer_squad && warp_in_squad == 0;

    const int k_split_idx = int(blockIdx.x);
    const int n_block_idx = int(blockIdx.y);
    const int m_block_idx = int(blockIdx.z);
    const int m_offset    = m_block_idx * BlockM;

    // Epilogue state, declared ahead of the split: the write-back ends the consumer branch.
    const int epi_compute_tid = is_producer_squad ? 0 : (tid - int(NumLoadWarpSquads * NumThreadsPerWarpSquad));
    const int epi_valid_m     = min(BlockM, params.m - m_offset);

    // This squad's write-back barrier: one arrival per warp (not per thread), released by the squad's
    // first warp before it issues the squad's TME store. Initialised before the prologue __syncthreads()
    // below, so no arrival can precede it; the id is clamped for producer squads.
    const uint32_t squad_in_gemm = squad < int(NumLoadWarpSquads) ? 0u : uint32_t(squad - int(NumLoadWarpSquads));
    const EpiSquadBarrier epi_barrier(static_cast<uint32_t>(reinterpret_cast<uint64_t>(&bs->Epilogue[squad_in_gemm])));
    if (!is_producer_squad && warp_in_squad == 0) {
      epi_barrier.init(int(WarpsPerWarpSquad), 0);
    }

    TiledMma epi_tiled_mma;
    auto     epi_thr_mma = epi_tiled_mma.get_thread_slice(epi_compute_tid);
    auto     epi_rAcc    = partition_fragment_C(epi_tiled_mma, take<0, 2>(TileShape{}));
    Tensor   epi_cA      = make_identity_tensor(select<0, 2>(TileShape{}));
    Tensor   epi_tCcA    = epi_thr_mma.partition_A(epi_cA);
    Tensor   epi_tCcA_mk = make_tensor(epi_tCcA.data(), detail::layout_a_mk(epi_tiled_mma, epi_tCcA.layout()));
    static constexpr int EpiRowsPerThread   = size<0>(epi_tCcA_mk);
    Tensor               epi_thread_sqr_sum = make_tensor<float>(make_shape(Int<EpiRowsPerThread>{}));
    auto                 reduction_target   = detail::reduction_target_k(epi_tiled_mma);
    using ReductionTarget                   = decltype(reduction_target);
    static constexpr int ReductionRank      = decltype(rank(ReductionTarget{}))::value;
    clear(epi_rAcc);
    clear(epi_thread_sqr_sum);

    const int k_blocks_this_split = params.num_k_blocks_per_split + (k_split_idx < params.remain_k_blocks ? 1 : 0);
    const int k_block_start = k_split_idx * params.num_k_blocks_per_split + min(k_split_idx, params.remain_k_blocks);

    Tensor sA = make_tensor(make_smem_ptr(shared.smem_a.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared.smem_b.data()), SmemLayoutB{});

    auto   cta_tme_b = params.tme_b.get_slice(0);
    auto   cta_tme_a = params.tme_a.get_slice(0);
    Tensor mA        = params.tme_a.get_tme_tensor(make_shape(params.m, params.k));
    Tensor gA_full   = local_tile(mA, select<0, 2>(TileShape{}), make_coord(_, _));
    Tensor mB        = params.tme_b.get_tme_tensor(make_shape(params.n, params.k));
    Tensor gB_full   = local_tile(mB, select<1, 2>(TileShape{}), make_coord(_, _));

    __syncthreads();

    if (is_producer_squad) {
      if (is_tme_warp) {
        PipelineState pipe_write = mutlass::make_producer_start_state<PipelineMain>();

        for (int kb = 0; kb < k_blocks_this_split; ++kb) {
          const int gkb = k_block_start + kb;
          pipeline_main.producer_acquire(pipe_write);

          const uint32_t barrier_id = pipeline_main.producer_get_barrier_id(pipe_write);
          const int      stage      = int(pipe_write.index());

          auto tAgA = cta_tme_a.partition_S(gA_full(_, _, m_block_idx, gkb));
          auto tAsA = cta_tme_a.partition_D(sA(_, _, stage));
          copy(params.tme_a.with(barrier_id), tAgA, tAsA);

          auto tBgB  = cta_tme_b.partition_S(gB_full(_, _, n_block_idx, gkb));
          auto sBTme = make_tensor(sB(_, _, stage).data(), take<0, 2>(SmemLayoutB{}));
          auto tBsB  = cta_tme_b.partition_D(sBTme);
          copy(params.tme_b.with(barrier_id), tBgB, tBsB);

          ++pipe_write;
        }
      }
    } else {
      TiledMma tiled_mma = epi_tiled_mma;
      auto     thr_mma   = epi_thr_mma;

      // TME has already permuted B while loading it, so SQMMA consumes canonical A/B descriptors. A is
      // read out of smem element by element, so give the consumer the swizzle-free view: the swizzle then
      // depends only on the layout offset. The producer's TME destination above still uses sA.
      Tensor sA_rs = as_position_independent_swizzle_tensor(sA);
      Tensor tCsA  = thr_mma.partition_A(sA_rs);
      Tensor tCsB  = thr_mma.partition_B(sB);
      Tensor tCrB  = thr_mma.make_fragment_B(tCsB);

      // The consumer state advances by one per iteration, so iteration kb always reads stage kb % kStages.
      // Unrolling the K loop by kStages and passing the stage as a compile-time tag makes every per-element
      // smem address a loop invariant, so the swizzle arithmetic is hoisted out of the loop.
      PipelineState pipe_read;

      auto consume = [&](auto stage_tag) {
        pipeline_main.consumer_wait(pipe_read);

        Tensor tCsAStage = tCsA(_, _, _, stage_tag);
        Tensor tCrAStage = thr_mma.make_fragment_A(tCsAStage);
        // BF16 is the high half of an FP32/TF32 word, so this recast is the widening. The generic typed
        // copy would instead round, which only ever touches bits TF32 discards.
        Tensor tCsAStageU16 = recast<uint16_t>(tCsAStage);
        // The low half of the word is already zero, which __builtin_assume lets the compiler rely on
        // instead of adding a truncation mask. Draining the whole stage out of smem first and widening in
        // registers afterwards keeps the reads back to back instead of each widen waiting on its own read.
        Tensor               tCrAStageI32     = recast<int32_t>(tCrAStage);
        static constexpr int kAElemsPerThread = size(tCrAStageI32);
        uint16_t             a_stage_bits[kAElemsPerThread];
        MUTE_UNROLL
        for (int i = 0; i < kAElemsPerThread; ++i) a_stage_bits[i] = tCsAStageU16(i);
        MUTE_UNROLL
        for (int i = 0; i < kAElemsPerThread; ++i) {
          __mt_bfloat16_raw bfr;
          bfr.x        = a_stage_bits[i];
          int32_t bits = __float_as_int(__musa_bfloat162float(bfr.data));
          __builtin_assume((bits & 0xFFFF) == 0);
          tCrAStageI32(i) = bits;
        }

        MUTE_UNROLL
        for (int mma_k = 0; mma_k < size<2>(tCrAStage); ++mma_k) {
          mute::gemm(tiled_mma, tCrAStage(_, _, mma_k), tCrB(_, _, mma_k, stage_tag), epi_rAcc);
        }

        // Keep the SQMMA commit/wait pair in one straight-line block so the whole batch is visible to the
        // dependency graph at once; the A sqrsum is computed unconditionally to overlap with SQMMA.
        mute::warpsquad_commit_batch();
        Tensor tCrAStage_mk = make_tensor(tCrAStage.data(), detail::layout_a_mk(tiled_mma, tCrAStage.layout()));
        using Float4        = float __attribute__((ext_vector_type(4)));
        MUTE_UNROLL
        for (int local_m = 0; local_m < size<0>(tCrAStage_mk); ++local_m) {
          Float4 local_sqr_sum = {0.f, 0.f, 0.f, 0.f};

          MUTE_UNROLL
          for (int local_k = 0; local_k < size<1>(tCrAStage_mk); local_k += 4) {
            const float v0 = float(tCrAStage_mk(local_m, local_k + 0));
            const float v1 = float(tCrAStage_mk(local_m, local_k + 1));
            const float v2 = float(tCrAStage_mk(local_m, local_k + 2));
            const float v3 = float(tCrAStage_mk(local_m, local_k + 3));

            const Float4 values = {v0, v1, v2, v3};
            local_sqr_sum       = local_sqr_sum + values * values;
          }

          epi_thread_sqr_sum(local_m) += (local_sqr_sum[0] + local_sqr_sum[2]) + (local_sqr_sum[1] + local_sqr_sum[3]);
        }

        mute::warpsquad_wait<0>();
        pipeline_main.consumer_release(pipe_read);
        ++pipe_read;
      };

      constexpr int kStageCount = int(kStages);

      int kb = 0;
      for (; kb + kStageCount <= k_blocks_this_split; kb += kStageCount) {
        for_each(make_seq<kStageCount>{}, [&](auto s) { consume(s); });
      }
      // Fewer than kStages iterations remain; the stage index restarts at 0.
      for_each(make_seq<kStageCount>{}, [&](auto s) {
        if (kb + int(decltype(s)::value) < k_blocks_this_split) {
          consume(s);
        }
      });
      // ---- epilogue: sqrsum + D write-back ------------------------------------------------------
      // Tail of the consumer branch: reduce the row sums, stage the tile in smem, rendezvous inside the
      // squad, then let the squad's first warp drain its SquadTileM rows with a TME store.
      if (n_block_idx == 0) {
        for_each(make_seq<ReductionRank>{}, [&](auto r) {
          MUTE_UNROLL
          for (int j = 1; j < shape<r>(reduction_target); j *= 2) {
            MUTE_UNROLL
            for (int local_m = 0; local_m < EpiRowsPerThread; ++local_m) {
              epi_thread_sqr_sum(local_m) +=
                  __shfl_xor_sync(0xffffffffu, epi_thread_sqr_sum(local_m), stride<r>(reduction_target) * j);
            }
          }
        });

        MUTE_UNROLL
        for (int local_m = 0; local_m < EpiRowsPerThread; ++local_m) {
          const int row = int(get<0>(epi_tCcA_mk(local_m, _0{})));
          if (get<1>(epi_tCcA_mk(local_m, _0{})) == 0 && row < epi_valid_m) {
            params.ptr_sqr_sum[k_split_idx * get<0>(params.stride_sqr_sum) +
                               (m_offset + row) * get<1>(params.stride_sqr_sum)] = epi_thread_sqr_sum(local_m);
          }
        }
      }

      // Stage the tile: the (BlockM, BlockN) stage is exactly the box the TME store drains, and the
      // descriptor drops rows past a partial M tile and columns past n, so no bounds check is needed.
      Tensor          sDStage = take<0, 2>(make_tensor(make_smem_ptr(shared.smem_d.data()), EpiSmemLayoutD{}));
      EpiR2STiledCopy epi_tiled_copy;
      auto            epi_thr_copy = epi_tiled_copy.get_thread_slice(epi_compute_tid);
      Tensor          tCrDEpi      = epi_thr_copy.retile_S(epi_rAcc);
      Tensor          tDsDEpi      = epi_thr_copy.partition_D(sDStage);
      copy(epi_tiled_copy, tCrDEpi, tDsDEpi);

      // Each squad drains its own row slice, so the staging buffer needs no CTA-wide rendezvous.
      __syncwarp();
      epi_barrier.arrive();
      if (warp_in_squad == 0) {
        epi_barrier.wait(0);
        Tensor sDSquad = make_tensor(make_smem_ptr(shared.smem_d.data() + size_t(squad_in_gemm) * EpiSquadStageElems),
                                     EpiSmemLayoutDSquad{});
        Tensor bSG_sD  = params.tme_d_squad.get_slice(Int<0>{}).partition_S(sDSquad);
        Tensor mD      = params.tme_d_squad.get_tme_tensor(make_shape(params.m, params.n, int(params.num_splits)));
        // The squad-local descriptor boxes SquadTileM rows: M coord = global row / SquadTileM.
        Tensor gD     = local_tile(mD,
                               Shape<Int<SquadTileM>, Int<BlockN>, Int<EpiStages>>{},
                               make_coord(m_block_idx * int(NumGemmSquads) + squad_in_gemm, n_block_idx, k_split_idx));
        Tensor bSG_gD = params.tme_d_squad.get_slice(Int<0>{}).partition_D(gD);
        copy(params.tme_d_squad, bSG_sD, bSG_gD);
        mute::tme_store_arrive();
        mute::tme_store_wait();
      }
    }
  }
};

}  // namespace mate::deep_gemm
