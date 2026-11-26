#pragma once

#include <mutlass/fast_math.h>
#include <mutlass/gemm/gemm.h>
#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/tensor.hpp>
#include <mutlass/arch/barrier.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
#include <mutlass/pipeline/mp32_pipeline.hpp>

#include "mate/gemm/deep_gemm/gemm_type.hpp"
#include "mate/gemm/deep_gemm/mp31_scheduler.hpp"

namespace mate::deep_gemm {

using namespace mute;

template <GemmType kType,
          typename ElementA,
          typename ElementB,
          typename ElementD,
          typename StrideA,
          typename StrideB,
          typename StrideD,
          typename TileShape,
          uint32_t kStages>
struct Mp32Fp8Fp4Gemm1D1D {
  static_assert(kType == GemmType::MGroupedContiguous || kType == GemmType::MGroupedMasked,
                "MP32 FP8 x FP4 GEMM supports contiguous and masked M-grouped modes");

  static_assert(is_same_v<ElementA, mutlass::mx_float8_t<typename ElementA::DataType>>, "MP32 W4A8 requires MXFP8 A");
  static_assert(is_same_v<ElementB, mutlass::mx_float4_t<typename ElementB::DataType>>, "MP32 W4A8 requires MXFP4 B");

  using ElementDataA  = typename ElementA::DataType;
  using ElementDataB  = typename ElementB::DataType;
  using ElementScaleA = typename ElementA::ScaleFactorType;
  using ElementScaleB = typename ElementB::ScaleFactorType;

  static_assert(is_same_v<ElementScaleA, mutlass::float_ue8m0_t> && is_same_v<ElementScaleB, mutlass::float_ue8m0_t>,
                "MP32 W4A8 requires UE8M0 scales");

  static constexpr int BlockM = size<0>(TileShape{});
  static constexpr int BlockN = size<1>(TileShape{});
  static constexpr int BlockK = size<2>(TileShape{});

  static_assert(is_same_v<ElementD, mutlass::bfloat16_t>, "MP32 W4A8 currently supports BF16 output");
  static_assert(is_static<TileShape>::value && rank(TileShape{}) == 3, "MP32 W4A8 requires a static rank-3 tile");
  static_assert((BlockM == 128 || BlockM == 256) && (BlockN == 128 || BlockN == 256) && BlockK == 128,
                "MP32 W4A8 requires tile M/N=128 or 256 and K=128");
  static_assert(kStages >= 3 && kStages <= 31, "MP32 W4A8 pipeline supports 3 to 31 stages");
  static_assert(rank(StrideA{}) == 3 && rank(StrideB{}) == 3 && rank(StrideD{}) == 3,
                "MP32 W4A8 requires rank-3 strides");
  // Public B becomes the register-A operand of RS SQMMA.
  static_assert(mutlass::gemm::detail::is_major<1, StrideB>(), "MP32 W4A8 requires K-major W");

  static constexpr int AlignmentA = 128 / sizeof_bits_v<ElementDataA>;
  static constexpr int AlignmentB = 128 / sizeof_bits_v<ElementDataB>;

  using ElementAccumulator = float;
  using KernelSchedule     = mutlass::gemm::KernelTmeWarpSpecializedBlockScaledMixedInputMp32;

  using CollectiveMma =
      typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp32,
                                                            mutlass::arch::OpClassBlockScaledTensorOp,
                                                            ElementA,
                                                            StrideA,
                                                            AlignmentA,
                                                            ElementB,
                                                            StrideB,
                                                            AlignmentB,
                                                            ElementAccumulator,
                                                            TileShape,
                                                            Shape<_1, _1, _1>,
                                                            mutlass::gemm::collective::StageCount<kStages>,
                                                            KernelSchedule>::CollectiveOp;

  // Public A is FP8 and public B is FP4. RS SQMMA consumes B as register A.
  // Keep all device mainloop tensors in this swapped operand order.
  using SwappedElementA      = ElementDataB;
  using SwappedElementB      = ElementDataA;
  using SwappedElementScaleA = ElementScaleB;
  using SwappedElementScaleB = ElementScaleA;
  using SwappedTileShape     = decltype(select<1, 0, 2>(TileShape{}));
  using MmaElementA          = typename CollectiveMma::MmaElementA;
  using TiledMma             = typename CollectiveMma::TiledMma;
  using PipelineAB           = typename CollectiveMma::MainloopPipelineData;
  using PipelineABParams     = typename PipelineAB::Params;
  using PipelineABState      = typename PipelineAB::PipelineState;
  using TensorGroup          = detail::GemmTensorGroupTraits<kType>;
  using Scheduler            = detail::Mp31PersistentTileScheduler<kType, BlockM, BlockN, 4>;

  static constexpr int NumThreadsPerWarpSquad = mutlass::NumThreadsPerWarpSquad;

  static constexpr int NumLoadWarpSquads = 1;
  static constexpr int NumMmaWarpSquads  = CollectiveMma::NumTCEWarpSquads;
  static constexpr int NumMmaWarps       = CollectiveMma::NumTCEWarps;
  static constexpr int TotalWarpSquads   = NumLoadWarpSquads + NumMmaWarpSquads;

  using SmemLayoutA   = typename CollectiveMma::SmemLayoutA;
  using SmemLayoutB   = typename CollectiveMma::SmemLayoutB;
  using SmemLayoutSFA = typename CollectiveMma::SmemLayoutSFA;
  using SmemLayoutSFB = typename CollectiveMma::SmemLayoutSFB;

  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t MaxThreadsPerBlock         = TotalWarpSquads * NumThreadsPerWarpSquad;
  static constexpr int      SmemAlignmentBytes         = CollectiveMma::SmemAlignmentBytes;

  static_assert(MaxThreadsPerBlock <= 1024, "MP32 W4A8 exceeds the CTA thread limit");
  static_assert(PipelineAB::NumBarriers + mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount <=
                    mutlass::arch::AsyncBarrier::HardwareMaxNumAsyncTransactionBarriers,
                "MP32 W4A8 exceeds the async barrier limit");

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t PipelineAB[PipelineAB::NumBarriers];
  };

  static constexpr uint32_t TmeTransactionBytes = CollectiveMma::TmeTransactionBytes;
  using SharedStorage                           = typename CollectiveMma::SharedStorage;
  static constexpr int SharedStorageSize        = int(sizeof(SharedStorage));
  static_assert(SharedStorageSize <= 256 * 1024, "MP32 grouped GEMM shared storage exceeds 256 KiB");

  using ScaleShape = Shape<int32_t, Shape<_32, int32_t>, int32_t>;
  template <bool Grouped>
  using ScaleLayout          = Layout<ScaleShape, Stride<_1, Stride<_0, int64_t>, conditional_t<Grouped, int64_t, _0>>>;
  using ScaleLayoutA         = ScaleLayout<TensorGroup::kScaleB>;
  using ScaleLayoutB         = ScaleLayout<TensorGroup::kScaleA>;
  using TmeInternalElementSF = typename CollectiveMma::TmeInternalElementSF;

  using TME_A   = typename CollectiveMma::TME_A;
  using TME_B   = typename CollectiveMma::TME_B;
  using TME_SFA = decltype(make_tme_copy<TmeInternalElementSF>(
      MP31_TME_LOAD{},
      make_tensor(static_cast<SwappedElementScaleA const*>(nullptr), ScaleLayoutA{}),
      take<0, 2>(SmemLayoutSFA{}),
      make_shape(Int<BlockN>{}, Int<BlockK>{})));
  using TME_SFB = decltype(make_tme_copy<TmeInternalElementSF>(
      MP31_TME_LOAD{},
      make_tensor(static_cast<SwappedElementScaleB const*>(nullptr), ScaleLayoutB{}),
      take<0, 2>(SmemLayoutSFB{}),
      make_shape(Int<BlockM>{}, Int<BlockK>{})));

  struct MainloopParams {
    TME_A        tme_load_a;
    TME_B        tme_load_b;
    TME_SFA      tme_load_sfa;
    TME_SFB      tme_load_sfb;
    ScaleLayoutA layout_sfa;
    ScaleLayoutB layout_sfb;
  };

  struct Arguments {
    ElementDataA const* ptr_a = nullptr;
    // Scale planes are MN-contiguous with an MN stride of M or N.
    ElementScaleA const* ptr_scale_a = nullptr;
    ElementDataB const*  ptr_b       = nullptr;
    ElementScaleB const* ptr_scale_b = nullptr;
    ElementD*            ptr_d       = nullptr;

    int32_t const* ptr_grouped_layout = nullptr;

    StrideA stride_a;
    StrideB stride_b;
    StrideD stride_d;

    int m          = 0;
    int n          = 0;
    int k          = 0;
    int num_groups = 1;
    int expected_m = 0;
    int num_mps    = 0;
  };

  struct Params {
    MainloopParams mainloop;

    ElementD*      ptr_d;
    StrideD        stride_d;
    int32_t const* ptr_grouped_layout;

    int m;
    int n;
    int k;
    int num_groups;
    int expected_m;
    int num_mps;
    int k_blocks;
  };

  template <bool Grouped>
  static ScaleLayout<Grouped> make_scale_layout(int32_t mn, int32_t k, int32_t groups) {
    int32_t k_groups = ceil_div(k, 32);
    int64_t leading  = mn;
    return make_layout(
        make_shape(mn, make_shape(_32{}, k_groups), groups),
        make_stride(_1{}, make_stride(_0{}, leading), conditional_return<Grouped>(leading * k_groups, _0{})));
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMma::can_implement(
        make_shape(args.m, args.n, args.k, args.num_groups),
        {args.ptr_a, args.stride_a, args.ptr_b, args.stride_b, args.ptr_scale_a, args.ptr_scale_b});
  }

  static Params to_underlying_arguments(Arguments const& args) {
    const int k_blocks = ceil_div(args.k, BlockK);
    // The collective's default scale descriptors assume dense batch strides.
    // Build grouped descriptors here, keeping public A/ScaleA flattened for contiguous.
    auto tme_load_a = make_tme_copy<typename CollectiveMma::TmeInternalElementA>(
        MP31_TME_LOAD{},
        make_tensor(args.ptr_b, make_layout(make_shape(args.n, args.k, args.num_groups), args.stride_b)),
        take<0, 2>(SmemLayoutA{}),
        make_shape(Int<BlockN>{}, Int<BlockK>{}));
    auto tme_load_b = make_tme_copy<typename CollectiveMma::TmeInternalElementB>(
        MP31_TME_LOAD{},
        make_tensor(args.ptr_a, make_layout(make_shape(args.m, args.k, args.num_groups), args.stride_a)),
        take<0, 2>(SmemLayoutB{}),
        make_shape(Int<BlockM>{}, Int<BlockK>{}));
    auto           layout_sfa   = make_scale_layout<TensorGroup::kScaleB>(args.n, args.k, args.num_groups);
    auto           layout_sfb   = make_scale_layout<TensorGroup::kScaleA>(args.m, args.k, args.num_groups);
    auto           tme_load_sfa = make_tme_copy<TmeInternalElementSF>(MP31_TME_LOAD{},
                                                            make_tensor(args.ptr_scale_b, layout_sfa),
                                                            take<0, 2>(SmemLayoutSFA{}),
                                                            make_shape(Int<BlockN>{}, Int<BlockK>{}));
    auto           tme_load_sfb = make_tme_copy<TmeInternalElementSF>(MP31_TME_LOAD{},
                                                            make_tensor(args.ptr_scale_a, layout_sfb),
                                                            take<0, 2>(SmemLayoutSFB{}),
                                                            make_shape(Int<BlockM>{}, Int<BlockK>{}));
    MainloopParams mainloop_params{tme_load_a, tme_load_b, tme_load_sfa, tme_load_sfb, layout_sfa, layout_sfb};
    return Params{mainloop_params,
                  args.ptr_d,
                  args.stride_d,
                  args.ptr_grouped_layout,
                  args.m,
                  args.n,
                  args.k,
                  args.num_groups,
                  args.expected_m,
                  args.num_mps,
                  k_blocks};
  }

  static dim3 get_grid_shape_persistent(Params const& params) {
    const int num_mps = params.num_mps > 0 ? params.num_mps : 1;
    return dim3(uint32_t(num_mps), 1u, 1u);
  }

  MUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);

    PipelineABParams pipeline_params;
    pipeline_params.transaction_bytes = TmeTransactionBytes;
    pipeline_params.num_consumers     = NumMmaWarps;
    pipeline_params.num_producers     = 1;
    PipelineAB pipeline(pipeline_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineAB));

    const int  squad             = mutlass::canonical_warp_squad_idx();
    const bool is_producer_squad = squad == 0;
    const bool is_consumer_squad = squad >= NumLoadWarpSquads && squad < TotalWarpSquads;

    auto problem_shape = make_shape(params.n, params.m, params.k, params.num_groups);
    auto load_inputs   = load_init(problem_shape, params.mainloop);

    Scheduler scheduler(static_cast<uint32_t>(params.m),
                        static_cast<uint32_t>(params.n),
                        static_cast<uint32_t>(params.num_groups),
                        params.ptr_grouped_layout,
                        static_cast<uint32_t>(params.expected_m));

    PipelineABState pipe_write = mutlass::make_producer_start_state<PipelineAB>();
    PipelineABState pipe_read;

    __syncthreads();

    if (is_producer_squad) {
      auto work = scheduler.initial_work_tile_info();

      MUTLASS_PRAGMA_NO_UNROLL
      for (; work.is_valid();) {
        __musa_loop_transparent_outermost();

        if (work.is_compute_valid()) {
          const uint32_t m_block_idx = work.m_row_offset / uint32_t(BlockM);
          const uint32_t n_block_idx = uint32_t(work.N_idx);
          const uint32_t gid         = uint32_t(work.G_idx);
          const uint32_t a_gid       = TensorGroup::kB ? gid : 0u;
          auto           tile_coord  = make_coord(n_block_idx, m_block_idx, _, gid);

          auto gA          = get<0>(load_inputs);
          auto tile_a      = gA(_, _, n_block_idx, _, a_gid);
          auto k_tile_iter = make_coord_iterator(shape<2>(tile_a));
          load(load_inputs, tile_coord, k_tile_iter, params.k_blocks, smem, params.mainloop, pipeline, pipe_write);
        }

        scheduler.advance_to_next_work();
        work = scheduler.get_work_tile_info();
      }
    } else if (is_consumer_squad) {
      const int     thread_idx = int(threadIdx.x) - NumLoadWarpSquads * NumThreadsPerWarpSquad;
      CollectiveMma collective_mma;
      TiledMma      tiled_mma;
      // RS accumulators use (public N, public M) coordinates.
      auto   stride_d   = make_stride(get<1>(params.stride_d), get<0>(params.stride_d), get<2>(params.stride_d));
      Tensor mD         = make_tensor(make_gmem_ptr(params.ptr_d),
                              make_shape(params.n, params.m, TensorGroup::kD ? params.num_groups : 1),
                              stride_d);
      auto   thread_mma = tiled_mma.get_thread_slice(thread_idx);
      Tensor identity   = make_identity_tensor(make_shape(Int<BlockN>{}, Int<BlockM>{}));
      Tensor coords     = thread_mma.partition_C(identity);

      auto work = scheduler.initial_work_tile_info();

      MUTLASS_PRAGMA_NO_UNROLL
      for (; work.is_valid();) {
        __musa_loop_transparent_outermost();

        const uint32_t m_block_idx = work.m_row_offset / uint32_t(BlockM);
        const uint32_t n_block_idx = uint32_t(work.N_idx);
        const uint32_t gid         = uint32_t(work.G_idx);
        const uint32_t d_gid       = TensorGroup::kD ? gid : 0u;
        auto           tile_coord  = make_coord(n_block_idx, m_block_idx, _, gid);

        // The producer also skips these tiles, so no pipeline stages need draining.
        if (!work.is_compute_valid()) {
          scheduler.advance_to_next_work();
          work = scheduler.get_work_tile_info();
          continue;
        }

        auto accum = partition_fragment_C(tiled_mma, take<0, 2>(tile_shape(tiled_mma)));
        fill(accum, ElementAccumulator(0));
        collective_mma.mma(load_inputs, tile_coord, accum, params.k_blocks, thread_idx, smem, pipeline, pipe_read);

        int valid_m = params.m;
        if constexpr (kType == GemmType::MGroupedMasked) {
          valid_m = min(params.m, max(0, int(__ldg(params.ptr_grouped_layout + gid))));
        }

        Tensor gD =
            local_tile(mD, make_shape(Int<BlockN>{}, Int<BlockM>{}), make_coord(n_block_idx, m_block_idx, d_gid));
        Tensor tD = thread_mma.partition_C(gD);
        MUTE_STATIC_ASSERT_V(size(tD) == size(accum));
        auto residue = make_coord(min(BlockN, max(0, params.n - int(n_block_idx) * BlockN)),
                                  min(BlockM, max(0, valid_m - int(work.m_row_offset))));

        // Match DefaultEpilogue's transposed traversal for public N-major D.
        using CoordLayout  = typename decltype(coords)::layout_type;
        constexpr int Step = mutlass::gemm::detail::is_major<1, StrideD>()
                                 ? int(size<0>(coalesce(CoordLayout{})))
                                 : int(size(typename decltype(accum)::layout_type{}));
        if (get<0>(residue) >= BlockN && get<1>(residue) >= BlockM) {
          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < Step; ++i) {
            MUTLASS_PRAGMA_UNROLL
            for (int j = i; j < size(accum); j += Step) {
              tD(j) = ElementD(accum(j));
            }
          }
        } else {
          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < Step; ++i) {
            MUTLASS_PRAGMA_UNROLL
            for (int j = i; j < size(accum); j += Step) {
              if (elem_less(coords(j), residue)) {
                tD(j) = ElementD(accum(j));
              }
            }
          }
        }

        scheduler.advance_to_next_work();
        work = scheduler.get_work_tile_info();
      }
    }
  }

 private:
  template <class ProblemShape>
  MUTLASS_DEVICE static auto load_init(ProblemShape const& problem, MainloopParams const& params) {
    using X           = Underscore;
    auto [M, N, K, L] = problem;
    Tensor mA         = params.tme_load_a.get_tme_tensor(make_shape(M, K, L));
    Tensor mB         = params.tme_load_b.get_tme_tensor(make_shape(N, K, L));
    Tensor mSFA       = params.tme_load_sfa.get_tme_tensor(shape(params.layout_sfa));
    Tensor mSFB       = params.tme_load_sfb.get_tme_tensor(shape(params.layout_sfb));
    auto   gA         = local_tile(mA, SwappedTileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
    auto   gB         = local_tile(mB, SwappedTileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
    auto   gSFA       = local_tile(mSFA, SwappedTileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
    auto   gSFB       = local_tile(mSFB, SwappedTileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
    return make_tuple(gA, gB, gSFA, gSFB);
  }

  template <class LoadInputs, class BlockCoord, class KTileIterator>
  MUTLASS_DEVICE static void load(LoadInputs const&     load_inputs,
                                  BlockCoord const&     tile_coord,
                                  KTileIterator         k_tile_iter,
                                  int                   k_tile_count,
                                  char*                 shared_memory,
                                  MainloopParams const& params,
                                  PipelineAB&           pipeline,
                                  PipelineABState&      pipe_write) {
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor         sA      = make_tensor(make_smem_ptr(storage.smem_A.begin()), SmemLayoutA{});
    Tensor         sB      = make_tensor(make_smem_ptr(storage.smem_B.begin()), SmemLayoutB{});
    Tensor         sSFA    = make_tensor(make_smem_ptr(storage.smem_SFA.begin()), SmemLayoutSFA{});
    Tensor         sSFB    = make_tensor(make_smem_ptr(storage.smem_SFB.begin()), SmemLayoutSFB{});

    auto cta_tme_a                            = params.tme_load_a.get_slice(0);
    auto cta_tme_b                            = params.tme_load_b.get_slice(0);
    auto cta_tme_sfa                          = params.tme_load_sfa.get_slice(0);
    auto cta_tme_sfb                          = params.tme_load_sfb.get_slice(0);
    auto [gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl] = load_inputs;
    auto [m_coord, n_coord, k_coord, gid]     = tile_coord;
    auto   a_gid                              = TensorGroup::kB ? gid : 0;
    auto   b_gid                              = TensorGroup::kA ? gid : 0;
    auto   sfa_gid                            = TensorGroup::kScaleB ? gid : 0;
    auto   sfb_gid                            = TensorGroup::kScaleA ? gid : 0;
    Tensor gA                                 = gA_mkl(_, _, m_coord, _, a_gid);
    Tensor gB                                 = gB_nkl(_, _, n_coord, _, b_gid);
    Tensor gSFA                               = gSFA_mkl(_, _, m_coord, _, sfa_gid);
    Tensor gSFB                               = gSFB_nkl(_, _, n_coord, _, sfb_gid);

    Tensor tAgA   = cta_tme_a.partition_S(gA);
    Tensor tAsA   = cta_tme_a.partition_D(sA);
    Tensor tBgB   = cta_tme_b.partition_S(gB);
    Tensor tBsB   = cta_tme_b.partition_D(sB);
    Tensor tAgSFA = cta_tme_sfa.partition_S(gSFA);
    Tensor tAsSFA = cta_tme_sfa.partition_D(sSFA);
    Tensor tBgSFB = cta_tme_sfb.partition_S(gSFB);
    Tensor tBsSFB = cta_tme_sfb.partition_D(sSFB);

    int warp_idx = mutlass::canonical_warp_idx() & 3;
    MUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 0; --k_tile_count) {
      if (warp_idx == 0) {
        pipeline.producer_acquire(pipe_write);
        uint32_t bar_id = pipeline.producer_get_barrier_id(pipe_write);
        int      stage  = pipe_write.index();
        copy(params.tme_load_a.with(bar_id), tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, stage));
        copy(params.tme_load_b.with(bar_id), tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, stage));
        copy(params.tme_load_sfa.with(bar_id), tAgSFA(_, _, _, *k_tile_iter), tAsSFA(_, _, _, stage));
        copy(params.tme_load_sfb.with(bar_id), tBgSFB(_, _, _, *k_tile_iter), tBsSFB(_, _, _, stage));
      }
      ++pipe_write;
      ++k_tile_iter;
    }
  }

 public:
  // Reuse the same RS conversion/SQMMA pipeline in fused MegaMoE Stage1.
  template <class LoadInputs, class BlockCoord, class FrgTensorC>
  MUTLASS_DEVICE static void mma(LoadInputs const& load_inputs,
                                 BlockCoord const& blk_coord,
                                 FrgTensorC&       accum,
                                 int               k_tile_count,
                                 int               thread_idx,
                                 char*             shared_memory,
                                 PipelineAB&       pipeline,
                                 PipelineABState&  smem_pipe_read) {
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be register resident");

    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor         sA =
        as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(storage.smem_A.begin()), SmemLayoutA{}));
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_B.begin()), SmemLayoutB{});

    TiledMma rs_tiled_mma;
    auto     thread_mma = rs_tiled_mma.get_thread_slice(thread_idx);
    Tensor   tCsA       = thread_mma.partition_A(sA);
    Tensor   tCsB       = thread_mma.partition_B(sB);
    Tensor   tCrB       = thread_mma.make_fragment_B(tCsB);

    Tensor sSFA     = make_tensor(make_smem_ptr(storage.smem_SFA.begin()), SmemLayoutSFA{});
    Tensor sSFB     = make_tensor(make_smem_ptr(storage.smem_SFB.begin()), SmemLayoutSFB{});
    Tensor tCsSFA   = MP32::BLOCKSCALED::partition_SFA(sSFA, thread_mma);
    Tensor tCsSFB   = MP32::BLOCKSCALED::partition_SFB(sSFB, thread_mma);
    auto   copy_sfa = Copy_Atom<UniversalCopy<uint32_t>, SwappedElementScaleA>{};
    auto   copy_sfb = Copy_Atom<UniversalCopy<uint32_t>, SwappedElementScaleB>{};

    auto fragment_layout_a  = make_fragment_like(take<0, 3>(tCsSFA.layout()));
    auto fragment_layout_b  = make_fragment_like(take<0, 3>(tCsSFB.layout()));
    using SFAFragmentLayout = decltype(fragment_layout_a);
    using SFBFragmentLayout = decltype(fragment_layout_b);
    static_assert(cosize(SFAFragmentLayout{}) % 4 == 0);
    static_assert(cosize(SFBFragmentLayout{}) % 4 == 0);

    auto sf_cur_a_words  = make_tensor<uint32_t>(make_shape(Int<cosize_v<SFAFragmentLayout> / 4>{}));
    auto sf_cur_b_words  = make_tensor<uint32_t>(make_shape(Int<cosize_v<SFBFragmentLayout> / 4>{}));
    auto sf_next_a_words = make_fragment_like(sf_cur_a_words);
    auto sf_next_b_words = make_fragment_like(sf_cur_b_words);
    auto tCrSFA_0 =
        make_tensor(make_rmem_ptr(reinterpret_cast<SwappedElementScaleA*>(sf_cur_a_words.data())), SFAFragmentLayout{});
    auto tCrSFB_0 =
        make_tensor(make_rmem_ptr(reinterpret_cast<SwappedElementScaleB*>(sf_cur_b_words.data())), SFBFragmentLayout{});
    auto tCrSFA_1 = make_tensor(make_rmem_ptr(reinterpret_cast<SwappedElementScaleA*>(sf_next_a_words.data())),
                                SFAFragmentLayout{});
    auto tCrSFB_1 = make_tensor(make_rmem_ptr(reinterpret_cast<SwappedElementScaleB*>(sf_next_b_words.data())),
                                SFBFragmentLayout{});
    MUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrB));
    MUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrSFA_0));
    MUTE_STATIC_ASSERT_V(size<2>(tCrB) == size<2>(tCrSFB_0));

    auto prefetch_sf = [&](auto& dst_a, auto& dst_b, int stage) {
      copy(copy_sfa, tCsSFA(_, _, _, stage), dst_a);
      copy(copy_sfb, tCsSFB(_, _, _, stage), dst_b);
    };

    if (k_tile_count == 0) {
      return;
    }

    constexpr int KBlockCount = size<2>(tCsA);
    static_assert(KBlockCount == 1, "This pipeline requires one K block per stage");
    auto                                     tCsA_atom   = tCsA(_, _, Int<0>{}, Int<0>{});
    auto                                     tCrA_mma_0  = make_fragment_like<MmaElementA>(tCsA_atom);
    auto                                     tCrA_mma_1  = make_fragment_like<MmaElementA>(tCsA_atom);
    constexpr int                            PackedWords = decltype(size(recast<uint32_t>(tCrA_mma_0)))::value;
    mutlass::Array<mutlass::float_e2m1_t, 4> tCrA_load[PackedWords];

    // Prologue: prepare A and scales for the first tile in register buffer 0.
    {
      pipeline.consumer_wait(smem_pipe_read);
      int read_stage = smem_pipe_read.index();

      prefetch_sf(tCrSFA_0, tCrSFB_0, read_stage);
      mutlass::gemm::collective::detail::load_e2m1_packed(tCrA_load, tCsA(_, _, Int<0>{}, read_stage));
      mutlass::gemm::collective::detail::convert_e2m1_packed_to_fp8<MmaElementA>(tCrA_mma_0, tCrA_load);
    }

    // Each iteration consumes tiles k and k+1, leaving tile k+2 prepared
    // in buffer 0. The shared read cursor always identifies the current tile.
    // These operations expose overlap opportunities; instruction scheduling
    // and physical register allocation remain compiler-controlled.
    MUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 2; k_tile_count -= 2) {
      // First half: preload packed A for tile k+1 before issuing tile k.
      auto smem_pipe_read_next = smem_pipe_read;
      ++smem_pipe_read_next;
      pipeline.consumer_wait(smem_pipe_read_next);
      int next_stage = smem_pipe_read_next.index();
      prefetch_sf(tCrSFA_1, tCrSFB_1, next_stage);
      mutlass::gemm::collective::detail::load_e2m1_packed(tCrA_load, tCsA(_, _, Int<0>{}, next_stage));

      // Issue tile k using A/SF buffer 0 and the current shared B stage.
      int read_stage = smem_pipe_read.index();
      for_each(make_seq<KBlockCount>{}, [&](auto k_block) {
        mute::gemm(rs_tiled_mma,
                   make_zip_tensor(tCrA_mma_0, tCrSFA_0(_, _, k_block)),
                   make_zip_tensor(tCrB(_, _, k_block, read_stage), tCrSFB_0(_, _, k_block)),
                   accum);
        warpsquad_commit_batch();
        // Prepare A buffer 1 for tile k+1 while tile k is in flight.
        mutlass::gemm::collective::detail::convert_e2m1_packed_to_fp8<MmaElementA>(tCrA_mma_1, tCrA_load);

        warpsquad_wait();
      });

      // SQMMA must finish reading shared B before the producer can reuse this stage.
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
      read_stage = smem_pipe_read.index();

      // Second half: tile k has been released; preload packed A for k+2.
      // The raw load buffer can be reused after conversion of tile k+1.
      ++smem_pipe_read_next;
      pipeline.consumer_wait(smem_pipe_read_next);
      next_stage = smem_pipe_read_next.index();
      prefetch_sf(tCrSFA_0, tCrSFB_0, next_stage);
      mutlass::gemm::collective::detail::load_e2m1_packed(tCrA_load, tCsA(_, _, Int<0>{}, next_stage));

      // Issue tile k+1 using A/SF buffer 1 and its shared B stage.
      for_each(make_seq<KBlockCount>{}, [&](auto k_block) {
        mute::gemm(rs_tiled_mma,
                   make_zip_tensor(tCrA_mma_1, tCrSFA_1(_, _, k_block)),
                   make_zip_tensor(tCrB(_, _, k_block, read_stage), tCrSFB_1(_, _, k_block)),
                   accum);
        warpsquad_commit_batch();
        // Refill A buffer 0 for tile k+2 after tile k has completed.
        mutlass::gemm::collective::detail::convert_e2m1_packed_to_fp8<MmaElementA>(tCrA_mma_0, tCrA_load);

        warpsquad_wait();
      });

      // SQMMA must finish reading shared B before the producer can reuse this stage.
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
    }

    // Two-tile tail: prepare only the final tile; never prefetch beyond K.
    if (k_tile_count == 2) {
      // Preload packed A for the final tile while buffer 0 holds the penultimate tile.
      auto smem_pipe_read_next = smem_pipe_read;
      ++smem_pipe_read_next;
      pipeline.consumer_wait(smem_pipe_read_next);
      int next_stage = smem_pipe_read_next.index();
      prefetch_sf(tCrSFA_1, tCrSFB_1, next_stage);
      mutlass::gemm::collective::detail::load_e2m1_packed(tCrA_load, tCsA(_, _, Int<0>{}, next_stage));

      // Issue the penultimate tile using A/SF buffer 0.
      int read_stage = smem_pipe_read.index();
      for_each(make_seq<KBlockCount>{}, [&](auto k_block) {
        mute::gemm(rs_tiled_mma,
                   make_zip_tensor(tCrA_mma_0, tCrSFA_0(_, _, k_block)),
                   make_zip_tensor(tCrB(_, _, k_block, read_stage), tCrSFB_0(_, _, k_block)),
                   accum);
        warpsquad_commit_batch();
        // Prepare the final tile in A buffer 1 while SQMMA is in flight.
        mutlass::gemm::collective::detail::convert_e2m1_packed_to_fp8<MmaElementA>(tCrA_mma_1, tCrA_load);

        warpsquad_wait();
      });

      // SQMMA must finish reading shared B before the producer can reuse this stage.
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;

      k_tile_count--;

      // Drain the final tile from buffer 1 without issuing another preload.
      read_stage = smem_pipe_read.index();
      for_each(make_seq<KBlockCount>{}, [&](auto k_block) {
        mute::gemm(rs_tiled_mma,
                   make_zip_tensor(tCrA_mma_1, tCrSFA_1(_, _, k_block)),
                   make_zip_tensor(tCrB(_, _, k_block, read_stage), tCrSFB_1(_, _, k_block)),
                   accum);
        warpsquad_commit_batch();
        warpsquad_wait();
      });

      // SQMMA must finish reading shared B before the producer can reuse this stage.
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;

      k_tile_count--;

      return;
    }

    // One-tile tail: buffer 0 is already prepared by the prologue or mainloop.
    if (k_tile_count == 1) {
      int read_stage = smem_pipe_read.index();

      for_each(make_seq<KBlockCount>{}, [&](auto k_block) {
        mute::gemm(rs_tiled_mma,
                   make_zip_tensor(tCrA_mma_0, tCrSFA_0(_, _, k_block)),
                   make_zip_tensor(tCrB(_, _, k_block, read_stage), tCrSFB_0(_, _, k_block)),
                   accum);
        warpsquad_commit_batch();
        warpsquad_wait();
      });

      // SQMMA must finish reading shared B before the producer can reuse this stage.
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;

      k_tile_count--;

      return;
    }
  }
};

}  // namespace mate::deep_gemm
