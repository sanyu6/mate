#pragma once

#include <musa_runtime.h>
#include <mutlass/numeric_conversion.h>

#include <cstddef>
#include <mute/arch/copy_mp31.hpp>
#include <mute/atom/copy_traits_mp31.hpp>
#include <mute/atom/copy_traits_mp31_tme.hpp>
#include <mute/atom/mma_traits_mp32_sqmma.hpp>
#include <mute/tensor.hpp>
#include <mutlass/arch/barrier.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>

#include "mate/common/layout.hpp"
#include "mate/flat/kda/chunk_kda_components.hpp"
#include "mate/flat/kda/chunk_kda_mp32_policy.hpp"
#include "mate/flat/kda/kda_permutes.hpp"
#include "mate/flat/kda/kernel/chunk_kda_recurrence_work.hpp"
#include "mate/flat/kda/named_barrier.hpp"
#include "mate/flat/simd_helper.hpp"

namespace mate::flat::kda::recurrence {

using namespace mute;

// Two consumer squads share one CTA and one 64-column value tile, each running
// half of the recurrence's five GEMMs:
//   * the STATE squad runs Scale (state @ diag) and UK (U^T @ k_restored); its
//     accumulator is the whole [kValueTile, 128] state fragment, 64 floats a
//     lane.
//   * the OUTPUT squad runs DS (decayed @ state), IP (inverse/P @ residual) and
//     PU (P @ U); its accumulators are the [32, kValueTile] DS and IPorPU
//     fragments, 16 floats a lane each.
// Scale and DS both read the state at the chunk start, and UK and PU both need
// only U, so the two squads overlap instead of serialising one five-GEMM chain.
// They exchange exactly two handshakes a chunk: `BarrierMap::UReady` (output -> state, U is
// in smem) and `BarrierMap::StateCommitted` (state -> output, the updated state is in smem
// for the next chunk's DS).  The CTA covers `kValueTile` columns of the
// head -- half of it -- and the CTA index selects the matching work, so two
// CTAs share a head.

template <class Components_, class TileShape_, class StateElement_, class StrideV_, class StrideO_>
struct Mp32ChunkKdaRecurrenceCollectiveTmeWarpSpecialized {
  using Components               = Components_;
  using ArchTag                  = mutlass::arch::Mp32;
  using Element                  = typename Components::Element;
  using StateElement             = StateElement_;
  using StrideV                  = StrideV_;
  using StrideO                  = StrideO_;
  using Workspace                = typename Components::Workspace;
  static constexpr int kChunk    = Components::kChunk;
  static constexpr int kHeadDim  = Components::kHeadDim;
  static constexpr int kValueDim = Components::kValueDim;
  // AlignmentA/AlignmentB for every CollectiveBuilder below: the operands
  // are 16-byte aligned.
  static constexpr int kOperandAlignment = 8;
  // The structure is named, not parameterised: one producer squad and exactly two
  // consumer squads, a state one and an output one.  Every thread and warp count
  // below is derived from these three, and `run()` dispatches exactly three ways.
  static constexpr int NumProducerSquads = 1;
  static constexpr int NumStateSquads    = 1;
  static constexpr int NumOutputSquads   = 1;
  // The state squad IS a state-MMA squad: it runs Scale and UK.  The kernel's arch
  // check uses this to tell the split recurrence from the MP31 one.
  static constexpr int NumProducerThreads = NumProducerSquads * mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumStateThreads    = NumStateSquads * mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumOutputThreads   = NumOutputSquads * mutlass::NumThreadsPerWarpSquad;
  // Per-squad warp counts: what a per-squad barrier counts.
  static constexpr int NumProducerWarps = NumProducerSquads * mutlass::NumWarpsPerWarpSquad;
  static constexpr int NumStateWarps    = NumStateSquads * mutlass::NumWarpsPerWarpSquad;
  static constexpr int NumOutputWarps   = NumOutputSquads * mutlass::NumWarpsPerWarpSquad;
  static_assert(NumStateWarps == 4 && NumOutputWarps == 4, "one warp squad a side");
  struct WarpSpecRole {
    static constexpr int ProducerSquad = 0;
    static constexpr int StateSquad    = 1;
    static constexpr int OutputSquad   = 2;
    static constexpr int DecayedWarp   = 0;
    static constexpr int RestoredWarp  = 1;
    static constexpr int InverseWarp   = 2;
    static constexpr int ScaleWarp     = 3;
    static constexpr int Producer      = ProducerSquad;
    static constexpr int State         = StateSquad;
    static constexpr int Output        = OutputSquad;
  };
  // The recurrence's input rings -- the staged smem tiles and their full/empty
  // barrier pairs -- are one stage deep.  A second TME in flight only takes smem
  // bandwidth away from the SQMMAs reading the first one: two stages measured
  // 2.398 ms at 8192x48 against 1.572 ms here, and neither version ever stalls a
  // producer waiting for a stage to come back.
  static constexpr int TmeStages          = 1;
  static constexpr int SmemAlignmentBytes = 256;
  static_assert(kChunk == 16 && kHeadDim == 128 && kValueDim == 128);
  // The CTA's value tile: 64 columns, half a head.  It is not a free parameter.
  // The state squad's accumulator is one float a lane per value column, so 64
  // columns is what still fits three CTAs an MP at ~108 registers -- 128 columns
  // would need a second state fragment a lane and cost the residency that the
  // H=144 round depends on -- and two CTAs then cover a head.  The tile is the
  // policy's number; the assert pins what the rest of this file assumes.
  static constexpr int kValueTile = Components::kRecurrenceValueTile;
  static_assert(kValueTile == 64, "the work item's value tile must be the CTA's 64 columns");
  static_assert(!Components::HasChunkStatesOut ||
                    (is_same_v<Element, mutlass::bfloat16_t> || is_same_v<Element, mutlass::half_t>),
                "MP32 chunk-state snapshots require a bf16/fp16 activation: the pool is io-dtype "
                "(the activation-typed snapshot), independent of the recurrent state");

  // The mainloop tag the builder instantiates is `MainloopMp32TmeSqmmaWarpSpecialized`
  // with this stage count baked in, and it static_asserts that it is at least 2, so
  // the collectives keep a two-stage pipeline they never run -- the recurrence
  // drives the SQMMAs itself and only takes the instruction, the atom and the
  // operand layouts off them.
  using DummyStageCount = mutlass::gemm::collective::StageCount<2>;

  // The five GEMMs are mutlass' own MP32 collectives.  The builder selects the
  // SQMMA instruction, its smem atom and the staged smem layout; the recurrence
  // consumes exactly those and never runs the collective's operator(), because it
  // drives the SQMMA itself around the diagonal handshake.  ClusterShape must be
  // 1x1x1 (the builder static_asserts it) -- each squad works on its own tile.
  //
  // Tag -> majorness is NOT symmetric between the operands: the A tag is the tag
  // of (M, K) while the B tag is the tag of (K, N) (mapped onto [N, K, L]), so
  // TagToStrideB<RowMajor> is Stride<_1, int64_t, int64_t>, i.e. MN-major.
  //   A: RowMajor -> Major::K,  ColumnMajor -> Major::MN
  //   B: RowMajor -> Major::MN, ColumnMajor -> Major::K

  // decayed @ state        : A K-major (gmem), B K-major (smem, Major::K)
  using CollectiveMmaDS = typename mutlass::gemm::collective::CollectiveBuilder<
      mutlass::arch::Mp32,
      mutlass::arch::OpClassTensorOp,
      Element,
      mutlass::layout::RowMajor,
      kOperandAlignment,
      Element,
      mutlass::layout::ColumnMajor,
      kOperandAlignment,
      float,
      Shape<_32, Int<kValueTile>, _128>,
      Shape<_1, _1, _1>,
      DummyStageCount,
      mutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  // inverse_P @ residual   : A K-major, B MN-major
  using CollectiveMmaIP = typename mutlass::gemm::collective::CollectiveBuilder<
      mutlass::arch::Mp32,
      mutlass::arch::OpClassTensorOp,
      Element,
      mutlass::layout::RowMajor,
      kOperandAlignment,
      Element,
      mutlass::layout::RowMajor,
      kOperandAlignment,
      float,
      Shape<_32, Int<kValueTile>, _16>,
      Shape<_1, _1, _1>,
      DummyStageCount,
      mutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  // inverse_P @ sU         : same shapes and majorness as IP, other operand
  using CollectiveMmaPU = typename mutlass::gemm::collective::CollectiveBuilder<
      mutlass::arch::Mp32,
      mutlass::arch::OpClassTensorOp,
      Element,
      mutlass::layout::RowMajor,
      kOperandAlignment,
      Element,
      mutlass::layout::RowMajor,
      kOperandAlignment,
      float,
      Shape<_32, Int<kValueTile>, _16>,
      Shape<_1, _1, _1>,
      DummyStageCount,
      mutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  // sU @ k_restored        : A MN-major, B MN-major (gmem (128,16):(1,128))
  using CollectiveMmaUK = typename mutlass::gemm::collective::CollectiveBuilder<
      mutlass::arch::Mp32,
      mutlass::arch::OpClassTensorOp,
      Element,
      mutlass::layout::ColumnMajor,
      kOperandAlignment,
      Element,
      mutlass::layout::RowMajor,
      kOperandAlignment,
      float,
      Shape<Int<kValueTile>, _128, _16>,
      Shape<_1, _1, _1>,
      DummyStageCount,
      mutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  // state @ diag           : A K-major, B MN-major
  using CollectiveMmaScale = typename mutlass::gemm::collective::CollectiveBuilder<
      mutlass::arch::Mp32,
      mutlass::arch::OpClassTensorOp,
      Element,
      mutlass::layout::RowMajor,
      kOperandAlignment,
      Element,
      mutlass::layout::RowMajor,
      kOperandAlignment,
      float,
      Shape<Int<kValueTile>, _128, _128>,
      Shape<_1, _1, _1>,
      DummyStageCount,
      mutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  // One stage of a staged smem tile, taken straight off the staged layout: its
  // first two modes are the tile and its stage mode strides by cosize(tile).  The
  // TME and SQMMA descriptor paths build their descriptor from the smem POINTER,
  // so a stage has to stay a pointer offset with the plain tile layout on top --
  // slicing the staged layout would carry the offset inside the layout, where
  // those paths cannot see it.

  // Every smem operand is taken straight off the collective that consumes it:
  // take<0, 2> of the collective's own staged SmemLayoutA/B is the tile, and
  // unstage_smem_layout re-stages it with the recurrence's own depth (TmeStages
  // for the pipelined inputs, kTileStages for the state and U tiles).  Nothing
  // here is hand-rolled any more.
  // Only kHeadDim entries of the diagonal are ever non-zero; the rest are written
  // once at kernel start and stay zero.  One stage, shared by both squads,
  // because they both walk the same chunk -- so it is the one operand that is a
  // plain tile, with no staging to carry.
  // The state and U tiles are single copies: one state squad writes the state and
  // one output squad owns U, so there is no per-squad stage to carry.
  static constexpr int kTileStages = 1;
  using SmemTileDiag               = decltype(take<0, 2>(typename CollectiveMmaScale::SmemLayoutB{}));
  using SmemLayoutDecayed =
      decltype(mate::unstage_smem_layout(typename CollectiveMmaDS::SmemLayoutA{}, Int<TmeStages>{}));
  using SmemLayoutKRestored =
      decltype(mate::unstage_smem_layout(typename CollectiveMmaUK::SmemLayoutB{}, Int<TmeStages>{}));
  using SmemLayoutInverseP =
      decltype(mate::unstage_smem_layout(typename CollectiveMmaIP::SmemLayoutA{}, Int<TmeStages>{}));
  // The state tile is read by the output squad's DS (as its B operand) and by the
  // state squad's Scale (as its A operand), so the two must be the same layout;
  // the static_assert below checks it rather than assuming it.
  using SmemLayoutState =
      decltype(mate::unstage_smem_layout(typename CollectiveMmaDS::SmemLayoutB{}, Int<kTileStages>{}));
  static_assert(
      is_same_v<SmemLayoutState,
                decltype(mate::unstage_smem_layout(typename CollectiveMmaScale::SmemLayoutA{}, Int<kTileStages>{}))>,
      "the state tile read by DS differs from the one read by Scale");
  // Physical state row = 4*hi + lo; natural value row = 16*lo + hi.
  // Exposing those two modes lets TME undo the value permutation while
  // retaining the SQMMA swizzle, without a second shared-memory tile.
  using SmemTileState          = decltype(take<0, 2>(SmemLayoutState{}));
  using SmemLayoutStateBlocked = decltype(composition(
      SmemTileState{}.layout_a(),
      SmemTileState{}.offset(),
      make_layout(make_shape(make_shape(_16{}, _4{}), _128{}), make_stride(make_stride(_512{}, _128{}), _1{}))));
  using TilerStateChunkStates =
      decltype(make_layout(make_shape(make_shape(_16{}, _4{}), _128{}), make_stride(make_stride(_1{}, _16{}), _128{})));
  using TMEChunkStates = decltype(make_tme_copy(MP31_TME_STORE{},
                                                make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)),
                                                            make_layout(make_shape(_16{}, _8{}, _128{}, int32_t{}),
                                                                        make_stride(_128{}, _2048{}, _1{}, _16384{}))),
                                                SmemLayoutStateBlocked{},
                                                TilerStateChunkStates{}));
  // The [kValueTile,16] MN-major operand tile.  inverse/P's lower half -- U --
  // is written by the IP gemm and read by the PU gemm as its B operand and by the
  // UK gemm as its A operand; the residual is the same tile, read by the IP gemm
  // as its B.  One layout serves all of them, and the collectives are each asked
  // for it so the equality is checked rather than assumed.
  using SmemLayoutU = decltype(mate::unstage_smem_layout(typename CollectiveMmaPU::SmemLayoutB{}, Int<kTileStages>{}));
  static_assert(
      is_same_v<SmemLayoutU,
                decltype(mate::unstage_smem_layout(typename CollectiveMmaUK::SmemLayoutA{}, Int<kTileStages>{}))>,
      "U read by UK differs from U read by PU");
  static_assert(
      is_same_v<SmemLayoutU,
                decltype(mate::unstage_smem_layout(typename CollectiveMmaIP::SmemLayoutB{}, Int<kTileStages>{}))>,
      "the residual tile differs from the U tile");

  using TiledMmaDS = typename CollectiveMmaDS::TiledMma;
  using TiledMmaIP = typename CollectiveMmaIP::TiledMma;
  using TiledMmaPU = typename CollectiveMmaPU::TiledMma;
  using TiledMmaUK = typename CollectiveMmaUK::TiledMma;
  // The state update is two SQMMAs on one accumulator:
  //   acc = U^T @ K_restored            (TiledMmaUK,  K = kChunk)
  //   acc = State @ diag(G_total)       (TiledMmaScale, K = kHeadDim)
  // The second one scales the carried state by the chunk decay, and its B operand
  // is a diagonal tile: MmaKBytes = kHeadDim * 16 / 8 = 256, one of the K sizes
  // the atom allows, and its CLayout is the same 32x128 / ThrID 128 the UK atom
  // uses, so both accumulate into the same fragment with no extra move.
  using TiledMmaScale = typename CollectiveMmaScale::TiledMma;

  // The state update's epilogue wants the state tile's key axis in the LOGICAL
  // frame, while the UK SQMMA's own N axis is the workspace SLOT frame.  Instead
  // of undoing the frame with index arithmetic, the epilogue reads its
  // coordinates off the same atom with its N axis relabelled by the workspace
  // frame (TiledMMA's PermutationMNK), so partition_C reports the logical key
  // directly.  Only coordinates come from this one: the gemm and the accumulator
  // fragment stay on TiledMmaUK, whose register order the vector stores rely on.
  using HeadDimPermute      = PermuteN<mutlass::arch::Mp32, Workspace::kHeadDim>;
  using SlotToLogical       = typename HeadDimPermute::SlotToLogical;
  using UKSlotToLogicalTile = Tile<Underscore, SlotToLogical, Underscore>;
  using TiledMmaUKCoords    = decltype(mate::convert_to_permuted_sqmma(TiledMmaUK{}, UKSlotToLogicalTile{}));

  // The widened path does not use a UValueFrame: at 64 columns a lane's IP
  // registers are stride-4 in the value axis, so the stores take the column the
  // atom reports (see the identity-frame note below) instead.  The committed
  // 32-wide path keeps its frame; only this build's widening drops it.

  // One flat struct.  The input tiles carry their stage as a layout mode; the
  // state and U tiles are single copies, so the struct-of-stages the collective
  // used to hand-roll (InputStage / ConsumerStorage) buys nothing.
  struct SharedStorage {
    array_aligned<Element, cosize_v<SmemLayoutDecayed>, SmemAlignmentBytes>   smem_decayed;
    array_aligned<Element, cosize_v<SmemLayoutKRestored>, SmemAlignmentBytes> smem_k_restored;
    array_aligned<Element, cosize_v<SmemLayoutInverseP>, SmemAlignmentBytes>  smem_inverse;
    array_aligned<Element, cosize_v<SmemLayoutInverseP>, SmemAlignmentBytes>  smem_p;
    array_aligned<Element, cosize_v<SmemTileDiag>, SmemAlignmentBytes>        smem_diag;
    // The state is kept in the Element representation.  Each KU iteration
    // loads it into the accumulator before the MMA and writes it back after
    // the MMA; no FP32 state cache is kept in shared memory or registers.
    array_aligned<Element, cosize_v<SmemLayoutState>, SmemAlignmentBytes> smem_state;
    // Reuse the tile as V, residual, then U. BarrierMap::StateCommitted protects the
    // previous UK reads before V prefetch; IP retires residual before U stores.
    union {
      array_aligned<Element, cosize_v<SmemLayoutU>, SmemAlignmentBytes> smem_residual;
      array_aligned<Element, cosize_v<SmemLayoutU>, SmemAlignmentBytes> smem_u;
    };
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static_assert(SharedStorageSize <= 64 * 1024);

  // Logical barrier ranges (hardware slot zero is reserved).  The ids are
  // RecurrenceBarriers' (named_barrier.hpp); this arch stages, per input ring, a
  // full/empty pair, and consumes them as:
  //   Decayed   decayed workspace   -> OUTPUT squad (DS A operand)
  //   Restored  k_restored + diagonal -> STATE squad (UK B operand)
  //   InvP      inverse/P/total     -> OUTPUT (inverse, P) + STATE (total)
  //   BarrierMap::Local     one rendezvous per squad
  //   BarrierMap::Seeded    producer -> consumer squads, once per work item
  //   BarrierMap::Drained   state squad -> producer, once per work item
  //   BarrierMap::UReady          OUTPUT squad -> STATE squad: U is in smem, run UK
  //   BarrierMap::StateCommitted  STATE squad  -> OUTPUT squad: the updated state is in smem
  // There is no V ring: a residual lane loads its own V word, so nothing is shared.
  using BarrierMap = RecurrenceBarriers<TmeStages,
                                        NumStateSquads + NumOutputSquads,
                                        /*HasVRing=*/false,
                                        /*HasStateSnapshot=*/false>;
  static_assert(BarrierMap::VReady == -1 && BarrierMap::VFree == -1 && BarrierMap::NumBarriers == 12 &&
                    BarrierMap::StateCommitted == 11,
                "the MP32 recurrence barrier map size moved");
  // A `full` barrier takes one arrival per owning warp -- two for the pair
  // `k_restored` and the diagonal share -- and its `empty` mate one per consumer
  // warp of the squad that consumes the ring.  `BarrierMap::Local` counts that squad's four
  // warps; `BarrierMap::Seeded` counts the four producer warps and `BarrierMap::Drained` the four state
  // warps; `BarrierMap::UReady` and `BarrierMap::StateCommitted` count the four warps of the squad that
  // publishes them.
  // Entries of the diagonal one diagonal-warp lane owns.
  static constexpr int kDiagPerLane = kHeadDim / mutlass::NumThreadsPerWarp;
  // Warps 2/3 of a consumer squad own the decayed K rows and form the residual
  // from V, so they are the ones that carry V.
  static constexpr int kResidualThreads = 2 * mutlass::NumThreadsPerWarp;

  using Arguments = ChunkKdaRecurrenceArguments<Element, StrideV, StrideO, Workspace>;

  // `p` and `inverse` are adjacent in the record (ChunkKdaWorkspace), so ONE
  // 32-row tile covers both: rows 0..15 are P and rows 16..31 the inverse.
  // The C fragment hands rows 0..15 to warps 0/1 -- the same warps that own the
  // decayed Q rows, so they close the output themselves -- and rows 16..31 to
  // warps 2/3, which own the K rows and therefore the residual and U.
  using PInverseTileLayout = Layout<Shape<_32, _16>, Stride<_16, _1>>;

  // The packed [P;inverse] tile is 1 KiB, so TME is the wrong tool for it: the
  // descriptor cost dwarfs the transfer, and a TME issue makes the warp wait on
  // the arrive before it can do anything else.  One producer warp moves it with
  // 16 B robust LDGSTS, which are fire-and-forget; a single ldgsts_wait() retires
  // the copies.  This is MP31's g pipeline (see mp31_chunk_kda_recurrence_...:
  // GmemCopyAtomInvP), and it is also the machinery the V tile will reuse.
  static constexpr int kInvPVectorBits = 128;
  static constexpr int kInvPFragment   = kInvPVectorBits / sizeof_bits_v<Element>;
  static_assert(kChunk % kInvPFragment == 0);
  using GmemCopyAtomInvP = mute::MP31_ROBUST_LDGSTS<mute::uint_bit_t<kInvPVectorBits>>;
  // The two TME loads are the collectives' own operand descriptors: an MP32
  // collective's Params is nothing but the TME copy for A and for B.  The DS A
  // operand is the decayed workspace and the UK B operand is k_restored, so
  // those are exactly the copies this kernel needs; they are built from the
  // collectives' Arguments in to_underlying_arguments below.
  using TmeLoadDecayed   = typename CollectiveMmaDS::Params::TME_A;
  using TmeLoadKRestored = typename CollectiveMmaUK::Params::TME_B;
  // The [P;inverse] tile is a robust LDGSTS copy, so it needs a robust
  // buffer-access descriptor instead of a TME descriptor.
  //
  // The pool extent is not here: it rides the inherited `Arguments` contract
  // (`checkpoint_pool_rows`, resolved by `to_underlying_arguments`), so this
  // payload keeps only the descriptor, which is what a build without the
  // chunk-state pool must omit.
  struct ChunkStatesParams {
    TMEChunkStates tme_chunk_states;
  };
  using ChunkStatesParamsBase = FeatureGatedParams<Components::HasChunkStatesOut, ChunkStatesParams>;
  struct Params : Arguments, ChunkStatesParamsBase {
    TmeLoadDecayed         tme_load_decayed;
    TmeLoadKRestored       tme_load_k_restored;
    mute::RobustDescriptor desc_inverse_p;
  };

  static_assert(sizeof(FeatureGatedParams<false, ChunkStatesParams>) == 1);
  static_assert(alignof(FeatureGatedParams<false, ChunkStatesParams>) == 1);
  static constexpr std::size_t align_up_params(std::size_t value, std::size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
  }
  static constexpr std::size_t kFeatureBaseEnd =
      Components::HasChunkStatesOut
          ? align_up_params(sizeof(Arguments), alignof(ChunkStatesParams)) + sizeof(ChunkStatesParams)
          : sizeof(Arguments);
  static_assert(offsetof(Params, tme_load_decayed) == align_up_params(kFeatureBaseEnd, alignof(TmeLoadDecayed)),
                "feature-gated MP32 Params suffix moved");

  template <class Problem>
  static Params to_underlying_arguments(Problem const& problem, Arguments const& args) {
    // `Params` inherits the argument contract, and the device store reads the
    // pool extent back out of it, so resolve the caller's 0 to the packed
    // record count here (once, on the host) instead of leaving a branch on the
    // device.
    Arguments resolved = args;
    resolved.checkpoint_pool_rows =
        args.checkpoint_pool_rows > 0 ? args.checkpoint_pool_rows : Components::workspace_records(problem);
    int  records = Components::workspace_records(problem) * problem.H;
    auto ip      = make_tensor(
        args.workspace.p,
        make_layout(make_shape(_32{}, _16{}, records), make_stride(_16{}, _1{}, Int<Workspace::RecordStrideInt>{})));

    // The collectives build their own operand descriptors from their Arguments.
    // The operand each of the two gemms does NOT read from gmem (state for DS, sU
    // for UK) still has to carry a majorness-correct pointer and stride, or the
    // TME descriptor encode rejects it; those copies are never issued.
    //
    //   decayed    : (M, K, record) = (32, 128, records), K contiguous
    //   k_restored : (N, K, record) = (128, 16, records), N contiguous
    using ArgsDS     = typename CollectiveMmaDS::Arguments;
    using StrideA_DS = mutlass::detail::TagToStrideA_t<mutlass::layout::RowMajor>;
    using StrideB_DS = mutlass::detail::TagToStrideB_t<mutlass::layout::ColumnMajor>;
    auto ds          = CollectiveMmaDS::to_underlying_arguments(
        make_shape(_32{}, Int<kValueTile>{}, _128{}, records),
        ArgsDS{args.workspace.decayed,
               StrideA_DS{int64_t(kHeadDim), _1{}, int64_t(Workspace::RecordStrideInt)},
               args.workspace.decayed,
               StrideB_DS{int64_t(kHeadDim), _1{}, int64_t(Workspace::RecordStrideInt)}},
        nullptr);

    using ArgsUK     = typename CollectiveMmaUK::Arguments;
    using StrideA_UK = mutlass::detail::TagToStrideA_t<mutlass::layout::ColumnMajor>;
    using StrideB_UK = mutlass::detail::TagToStrideB_t<mutlass::layout::RowMajor>;
    auto uk          = CollectiveMmaUK::to_underlying_arguments(
        make_shape(Int<kValueTile>{}, _128{}, _16{}, records),
        ArgsUK{args.workspace.k_restored,
               StrideA_UK{_1{}, int64_t(kHeadDim), int64_t(Workspace::RecordStrideInt)},
               args.workspace.k_restored,
               StrideB_UK{_1{}, int64_t(kHeadDim), int64_t(Workspace::RecordStrideInt)}},
        nullptr);

    // One descriptor per LDGSTS stream; `ip` spans P and the inverse of every
    // record, so a copy that runs past the last in-record element still resolves
    // to valid workspace memory rather than silently zero-filling.
    ChunkStatesParamsBase chunk_states{};
    if constexpr (Components::HasChunkStatesOut) {
      // The pool's outer extent is the caller's row count times H; the
      // checkpoint cadence pool is compact, so it is smaller than the packed
      // chunk-slot space the workspace tensors use.
      int32_t const pool_outer = resolved.checkpoint_pool_rows * problem.H;
      auto          pool       = make_tensor(make_gmem_ptr(static_cast<Element*>(args.ptr_chunk_states)),
                              make_layout(make_shape(_16{}, _8{}, _128{}, int32_t(pool_outer)),
                                          make_stride(_128{}, _2048{}, _1{}, _16384{})));
      chunk_states.tme_chunk_states =
          make_tme_copy(MP31_TME_STORE{}, pool, SmemLayoutStateBlocked{}, TilerStateChunkStates{});
    }
    return {resolved,
            chunk_states,
            ds.tme_load_a,
            uk.tme_load_b,
            mute::make_robust_desc(args.workspace.p, static_cast<size_t>(mute::cosize(ip.layout())))};
  }

  template <class Problem, class WorkDesc>
  MUTLASS_DEVICE static void store_chunk_states(
      SharedStorage& shared, Params const& params, Problem const& problem, WorkDesc const& work, int chunk) {
    if constexpr (Components::HasChunkStatesOut) {
      // The host selects the row mapping; the kernel only picks between the two
      // compile-time layouts.  `kStride` is a constant, so the cadence modulus
      // below folds away when the cadence equals the chunk (every chunk stores).
      constexpr int kStride = Components::kCheckpointStride;
      int64_t       pool_row;
      if constexpr (Components::HasCheckpointCadence) {
        // Checkpoint cadence: only the chunk whose state enters token
        // `j * every_n_tokens`, packed per sequence.  Skipping the store here
        // is safe because the caller's post-store rendezvous is unconditional;
        // only the TME issue is conditional.
        if ((chunk % kStride) != 0) {
          return;
        }
        // The work descriptor already holds this sequence's checkpoint row base
        // (see `make_recurrence_work`), so nothing is scanned here.
        pool_row = int64_t(work.checkpoint_base) + chunk / kStride;
        pool_row = pool_row * problem.H + work.head_idx;
      } else {
        pool_row = int64_t(Workspace::record(problem, work, chunk));
      }
      int32_t const pool_rows = params.checkpoint_pool_rows * problem.H;
      auto          sState    = make_tensor(make_smem_ptr(shared.smem_state.data()), SmemLayoutStateBlocked{});
      auto          gcoord    = params.tme_chunk_states.get_tme_tensor(make_shape(_16{}, _8{}, _128{}, pool_rows));
      // Offset the coordinate tensor before partition_D so the second CTA
      // actually selects rows 64..127 in the emitted TME instruction.
      auto gc      = domain_offset(make_coord(_0{}, int32_t(work.value_start / 16), _0{}, int32_t(pool_row)), gcoord);
      auto cta_tme = params.tme_chunk_states.get_slice(_0{});
      auto tS      = group_modes<0, 3>(cta_tme.partition_S(sState));
      auto tG      = group_modes<0, 3>(cta_tme.partition_D(gc));
      copy(params.tme_chunk_states, tS, tG(_, _0{}, _0{}));
      mute::tme_store_arrive();
    }
  }
  // Each role handles the one work owned by this CTA, without scheduler state.
  template <class Work, class Problem>
  MUTLASS_DEVICE static void run_role(SharedStorage& shared, Params const& params, Problem const& problem) {
    auto work = make_recurrence_work<Components>(problem, uint32_t(blockIdx.x));
    if (work.seq_idx >= 0) {
      int sequence = 0;
      Work{}(shared, params, problem, work, sequence);
    }
  }

  // A residual lane needs exactly the eight V columns its DS fragment row covers,
  // so it reads that one 16 B word straight into registers: no smem staging, no
  // barrier and no syncwarp.
  using ElementVec = mutlass::Array<Element, 8>;
  using AccumVec   = mutlass::Array<float, 8>;

  // One lane's own V word, taken from V's tensor rather than from hand-computed
  // strides: those eight consecutive value columns are one 16 B element of the
  // value axis, so recasting the column slice makes the read a single 128-bit
  // access.

  // k_restored's headDim axis is stored in the MP32 slot frame; the frame and
  // its inverse (slot -> logical key) live in kda_permutes.hpp, so
  // K1 and K2 share one definition instead of a closed form each.

  // The value axis carries a FRAME: logical value column v lives at tile row
  // `TileRowFromLogical{}(v)`, chosen so that a lane's accumulator registers
  // land on CONSECUTIVE logical columns.  This is the STATE and U tile frame; the
  // residual tile's frame is re-oriented below, which is what lets both smem
  // stores be one 128-bit access per group.
  //
  // The map, pinned by the asserts below: for the 32x64x128 DS atom, register r
  // of lane t reports (row, column) =
  // (t / 4, t % 4 + 4r) -- the value axis advances by 4 per register, so a lane's
  // raw registers are stride-4 columns and no per-lane packed store exists in the
  // ATOM frame.  The same map holds for the 32x64 IP/PU atoms (one map for all
  // three C fragments), and the 64x128x16 UK atom is (t/4 + 32*(r/32), t%4 + 4*(r%32)).
  static_assert(get<0>(TiledMmaDS{}.get_thread_slice(0).partition_C(
                    make_identity_tensor(Shape<_32, Int<kValueTile>>{}))(0)) == 0 &&
                    get<1>(TiledMmaDS{}.get_thread_slice(0).partition_C(
                        make_identity_tensor(Shape<_32, Int<kValueTile>>{}))(0)) == 0,
                "DS atom map changed: (lane 0, reg 0) must be (0,0)");
  static_assert(
      get<1>(TiledMmaDS{}.get_thread_slice(0).partition_C(make_identity_tensor(Shape<_32, Int<kValueTile>>{}))(1)) == 4,
      "DS atom map changed: (lane 0, reg 1) column must be 4");
  static_assert(
      get<1>(TiledMmaDS{}.get_thread_slice(1).partition_C(make_identity_tensor(Shape<_32, Int<kValueTile>>{}))(0)) == 1,
      "DS atom map changed: (lane 1, reg 0) column must be 1");
  static_assert(get<1>(TiledMmaDS{}.get_thread_slice(0).partition_C(
                    make_identity_tensor(Shape<_32, Int<kValueTile>>{}))(8)) == 32,
                "DS atom map changed: (lane 0, reg 8) column must be 32");
  static_assert(
      get<0>(TiledMmaDS{}.get_thread_slice(4).partition_C(make_identity_tensor(Shape<_32, Int<kValueTile>>{}))(0)) == 1,
      "DS atom map changed: (lane 4, reg 0) row must be 1");

  // A lane's DS/IP/PU fragment is kValueTile / 4 registers (32*64/128 = 16), so
  // it holds (kValueTile / 4) / 8 = kValueTile / 32 groups of eight.  Bounding
  // the vector loops by kValueTile / 8 is WRONG -- that counts KEYS (the UK
  // fragment's groups), not this fragment's -- and reads past the fragment.
  static constexpr int kValueGroups = kValueTile / 32;
  static_assert(kValueGroups * 8 == kValueTile / 4, "a lane's value fragment must be whole groups of eight registers");

  // The tile frame: the STATE and U tiles store a logical column at a permuted
  // row, and MP32's PermuteV owns that map (kda_permutes.hpp).  The atom map
  // asserted above is what says the atom's C column is already the tile row.
  // The residual tile uses the re-oriented frame derived below instead, because
  // it is the one value-axis tile whose frame every reader lets us choose.
  using ValuePermute       = PermuteV<mutlass::arch::Mp32, kValueTile>;
  using TileRowFromLogical = typename ValuePermute::TileRowFromLogical;
  using LogicalFromTileRow = typename ValuePermute::LogicalFromTileRow;

  // ---- why the residual and U stores are one 128-bit access per group ------
  //
  // The residual tile is the PU/UK B/A operand un-staged, i.e.
  //   SmemLayoutU = ComposedLayout<Swizzle<4,4,4>, smem_ptr_flag,
  //                                ((64,1),(16,1)):((1,0),(64,0))>
  // (printed by a host probe of the collectives), so addr(n, k) = sw(n + 64k)
  // with sw(y) = y ^ ((y >> 4) & 0xF0): the value axis is element-contiguous and
  // the swizzle only ever XORs bits 4..7, so EIGHT CONSECUTIVE 8-ALIGNED value
  // indices are eight contiguous smem elements -- one 128-bit access.  The state
  // tile ((64,128):(128,1), same swizzle) gets its 16 B stores from the same
  // property along the key axis.
  //
  // What has to be true for a lane to use one is a frame question, not a layout
  // one.  Register r of lane t (t % 4 = c) sits on C column p = c + 4r, so the
  // lane's group h = registers 8h..8h+7 covers the stride-4 coordinate set
  //   P(c,h) = { c + 32h + 4i : i < 8 }.
  // A packed residual store therefore needs the residual tile's frame -- a
  // bijection coordinate -> logical value column, free because the residual tile
  // is written and read only here and by `thread_mma_ip.partition_B` -- to map
  // the eight coordinates of P(c,h) onto eight CONSECUTIVE tile coordinates.
  //
  // The other three value-axis paths pin their frames, so they stay as they are:
  //   * the V word is two 16 B gmem reads at logical columns 16c + 8h .. + 7, so
  //     the DS register (c,h,i) must hold the value column nat(p) = 16c + 8h + i
  //     while the state tile's coordinate p holds it: the state frame is nat;
  //   * the output is q@S + P@U accumulated REGISTER-WISE into tCrIPorPU, so the
  //     PU register (c,h,i) must hold the same value column as the DS one, and
  //     the output store is one 16 B store per group: the U frame is nat too.
  // Only the residual frame is free, and with the residual store at frame order
  // sigma(i) the U store's target is forced: the U store's coordinate for the IP
  // register (c,h,i) is phys(nat_res(p)) and must ALSO be eight consecutive
  // coordinates.  Solving both at once gives a frame that cuts the value axis
  // into one block of eight per (lane column c, group h) pair:
  //
  //   block index g = c + 4h   ->   the eight value columns [8g, 8g + 8)
  //   order sigma(i) = 4 * (i % 2) + i / 2, i.e. position j of the packed store
  //                  holds register sigma^-1(j) = 2 * (j % 4) + j / 4
  //
  // -- the even registers first, then the odd ones -- for BOTH stores, at the
  // SAME coordinate block.  Both tiles are written by this same lane/group pair,
  // so one helper serves both.
  //
  // g is mixed-radix over (c, h), so the pairs account for the axis exactly, and
  // writing the base as 8 * g is what makes that tiling hold for any kValueTile:
  // the form `v0 / 2 + 32 * group` only equals it when kValueTile / 4 is 16.
  struct ResidualValueStoreFrame {
    static constexpr int kBlock       = 8;               // registers holding one group
    static constexpr int kLaneColumns = kValueTile / 4;  // logical columns per c

    MUTLASS_HOST_DEVICE static constexpr int block_base(int v0, int group) {
      return kBlock * (v0 / kLaneColumns + 4 * group);
    }

    MUTLASS_HOST_DEVICE static constexpr int register_for_store(int position) {
      return 2 * (position % 4) + position / 4;
    }

    template <class Vec8>
    MUTLASS_DEVICE static Vec8 reorder_for_store(Vec8 const& v) {
      Vec8 out;
      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 8; ++j) {
        out[j] = v[register_for_store(j)];
      }
      return out;
    }
  };

  // v0 is the lane's first logical column = kLaneColumns * c, so c steps one block
  // and group h steps four.  The store reads even registers first, then odd ones.
  static_assert(ResidualValueStoreFrame::block_base(0, 0) == 0 &&
                    ResidualValueStoreFrame::block_base(ResidualValueStoreFrame::kLaneColumns, 0) ==
                        ResidualValueStoreFrame::kBlock &&
                    ResidualValueStoreFrame::block_base(0, 1) == 4 * ResidualValueStoreFrame::kBlock &&
                    ResidualValueStoreFrame::register_for_store(0) == 0 &&
                    ResidualValueStoreFrame::register_for_store(4) == 1 &&
                    ResidualValueStoreFrame::register_for_store(7) == 7,
                "residual value store frame moved");

  // The four producer warps seed and drain the single [kValueTile, 128] state
  // tile between them: warp `Part` covers value rows
  // [Part * kPartRows, (Part + 1) * kPartRows).  All four arrive on `BarrierMap::Seeded`, and
  // all four wait on `BarrierMap::Drained` before exporting their final rows.
  template <int Part, class Problem, class Work>
  MUTLASS_DEVICE static void initialize_state(SharedStorage& shared,
                                              Params const&  params,
                                              Problem const& problem,
                                              Work const&    work) {
    constexpr int kPartRows = kValueTile / NumProducerWarps;
    auto          sState    = make_tensor(make_smem_ptr(shared.smem_state.data()), SmemLayoutState{})(_, _, 0);
    int           lane_idx  = mutlass::canonical_lane_idx();
    int           state_row = params.ptr_state_indices[work.seq_idx];
    int64_t       offset    = (int64_t(state_row) * problem.H + work.head_idx) * kValueDim * kHeadDim +
                     (work.value_start + Part * kPartRows) * kHeadDim;
    MUTLASS_PRAGMA_UNROLL
    for (int i = lane_idx; i < kPartRows * kHeadDim; i += mutlass::NumThreadsPerWarp) {
      float value = 0.f;
      if constexpr (Components::HasStateIn) {
        if (state_row >= 0) {
          value = float(static_cast<StateElement const*>(params.ptr_initial_state)[offset + i]);
        }
      }
      // The tile's value axis is the physical frame, so the seed and the drain
      // both go through TileRowFromLogical.
      sState(int(TileRowFromLogical{}(Part * kPartRows + i / kHeadDim)), i % kHeadDim) = Element(value);
    }
    arrive_warp(BarrierMap::Seeded);
  }

  template <int Part, class Problem, class Work>
  MUTLASS_DEVICE static void finalize_state(SharedStorage const& shared,
                                            Params const&        params,
                                            Problem const&       problem,
                                            Work const&          work) {
    if constexpr (Components::HasStateOut) {
      constexpr int kPartRows = kValueTile / NumProducerWarps;
      int           state_row = params.ptr_state_indices[work.seq_idx];
      if (state_row >= 0) {
        auto sState =
            make_tensor(make_smem_ptr(const_cast<Element*>(shared.smem_state.data())), SmemLayoutState{})(_, _, 0);
        int     lane_idx = mutlass::canonical_lane_idx();
        int64_t offset   = (int64_t(state_row) * problem.H + work.head_idx) * kValueDim * kHeadDim +
                         (work.value_start + Part * kPartRows) * kHeadDim;
        MUTLASS_PRAGMA_UNROLL
        for (int i = lane_idx; i < kPartRows * kHeadDim; i += mutlass::NumThreadsPerWarp) {
          int v = Part * kPartRows + i / kHeadDim;
          int k = i % kHeadDim;
          static_cast<StateElement*>(params.ptr_final_state)[offset + i] =
              StateElement(sState(int(TileRowFromLogical{}(v)), k));
        }
      }
    }
  }

  // ---- the warp roles -------------------------------------------------------
  // A role processes the single work owned by its CTA.  Each producer role carries
  // its own input ring: a NamedBarrierRing's `full` publishes a stage to the squads
  // and its `empty` says they have retired it.  Every ring is one stage deep, the
  // diagonal included.

  // Producer warp 0: the decayed tile, one 32x128 TME tile per chunk, lane 0.
  struct DecayedProducer : NamedBarrierRing<BarrierMap::TmeStages, BarrierMap::DecayedReady, BarrierMap::DecayedFree> {
    static constexpr int ValueSlice = 0;
    // Bytes the TME delivers into one stage -- taken from the DS collective's own
    // operand accounting, so it follows the layout the builder picked.
    static constexpr uint32_t bytes = CollectiveMmaDS::TmeTransactionBytesMK;
    // Wait for this chunk's stage and hand back the slice of the ring it is on: the
    // wait and the address arithmetic both stay in here.
    MUTLASS_DEVICE static auto yield(SharedStorage& shared, int sequence) {
      wait(sequence);
      auto sStaged = make_tensor(make_smem_ptr(shared.smem_decayed.data()), SmemLayoutDecayed{});
      return sStaged(_, _, stage(sequence));
    }

    MUTLASS_DEVICE void load(
        SharedStorage& shared, Params const& params, int lane_idx, int sequence, int64_t record, int records) const {
      if (lane_idx != 0) {
        return;
      }
      auto gD   = params.tme_load_decayed.get_tme_tensor(make_shape(_32{}, _128{}, records));
      auto sD   = make_tensor(make_smem_ptr(shared.smem_decayed.data()), SmemLayoutDecayed{})(_, _, stage(sequence));
      auto tmeD = params.tme_load_decayed.get_slice(0);
      copy(params.tme_load_decayed.with(named_barrier_id(full(sequence))),
           tmeD.partition_S(gD(_, _, record)),
           tmeD.partition_D(sD));
    }

    template <class Problem, class WorkDesc>
    MUTLASS_DEVICE void operator()(SharedStorage&  shared,
                                   Params const&   params,
                                   Problem const&  problem,
                                   WorkDesc const& work,
                                   int&            sequence) const {
      int lane_idx = mutlass::canonical_lane_idx();
      int records  = Components::workspace_records(problem) * problem.H;
      initialize_state<ValueSlice>(shared, params, problem, work);
      MUTLASS_PRAGMA_NO_UNROLL
      for (int chunk = 0; chunk < work.n_chunks; ++chunk, ++sequence) {
        // Take the stage back from the squads: they arrive on empty once they
        // have retired it.
        wait_empty(sequence);
        // Announce the TME transaction -- it has to be on the barrier BEFORE the
        // copy is issued -- then issue it.
        expect_tx_warp(full(sequence), bytes);
        load(shared, params, lane_idx, sequence, Workspace::record(problem, work, chunk), records);
      }
      wait_warp(BarrierMap::Drained, 0);
      finalize_state<ValueSlice>(shared, params, problem, work);
    }
  };

  // Producer warp 1: k_restored, one 128x16 TME tile per chunk, lane 0.  Record
  // stride is in Element units -- the step the TME tile's third mode walks.
  struct RestoredProducer
      : NamedBarrierRing<BarrierMap::TmeStages, BarrierMap::RestoredReady, BarrierMap::RestoredFree> {
    static constexpr int ValueSlice = 1;
    // Bytes the TME delivers into one stage, from the UK collective's B operand.
    static constexpr uint32_t bytes = CollectiveMmaUK::TmeTransactionBytesNK;
    // Wait for this chunk's stage and hand back the slice of the ring it is on: the
    // wait and the address arithmetic both stay in here.
    MUTLASS_DEVICE static auto yield(SharedStorage& shared, int sequence) {
      wait(sequence);
      auto sStaged = make_tensor(make_smem_ptr(shared.smem_k_restored.data()), SmemLayoutKRestored{});
      return sStaged(_, _, stage(sequence));
    }

    MUTLASS_DEVICE void load(
        SharedStorage& shared, Params const& params, int lane_idx, int sequence, int64_t record, int records) const {
      if (lane_idx != 0) {
        return;
      }
      auto gK = params.tme_load_k_restored.get_tme_tensor(make_shape(_128{}, _16{}, records));
      auto sK = make_tensor(make_smem_ptr(shared.smem_k_restored.data()), SmemLayoutKRestored{})(_, _, stage(sequence));
      auto tmeK = params.tme_load_k_restored.get_slice(0);
      copy(params.tme_load_k_restored.with(named_barrier_id(full(sequence))),
           tmeK.partition_S(gK(_, _, record)),
           tmeK.partition_D(sK));
    }

    template <class Problem, class WorkDesc>
    MUTLASS_DEVICE void operator()(SharedStorage&  shared,
                                   Params const&   params,
                                   Problem const&  problem,
                                   WorkDesc const& work,
                                   int&            sequence) const {
      int lane_idx = mutlass::canonical_lane_idx();
      int records  = Components::workspace_records(problem) * problem.H;
      initialize_state<ValueSlice>(shared, params, problem, work);
      MUTLASS_PRAGMA_NO_UNROLL
      for (int chunk = 0; chunk < work.n_chunks; ++chunk, ++sequence) {
        // Take the stage back from the squads: they arrive on empty once they
        // have retired it.
        wait_empty(sequence);
        // Announce the TME transaction -- it has to be on the barrier BEFORE the
        // copy is issued -- then issue it.
        expect_tx_warp(full(sequence), bytes);
        load(shared, params, lane_idx, sequence, Workspace::record(problem, work, chunk), records);
      }
      wait_warp(BarrierMap::Drained, 0);
      finalize_state<ValueSlice>(shared, params, problem, work);
    }
  };

  // Producer warp 2: the packed [P;inverse] tile.  It is tiny, so it rides async
  // LDGSTS -- the issuing lanes never block and one wait retires the copies.  The
  // whole warp takes part; the `publish` arrive publishes it to the consumers.
  // The chunk's decay total is NOT staged: the diagonal producer reads it straight
  // from gmem, so a smem copy would only be written and never read.
  struct InversePProducer : NamedBarrierRing<BarrierMap::TmeStages, BarrierMap::InvPReady, BarrierMap::InvPFree> {
    static constexpr int ValueSlice = 2;
    // Wait for this chunk's stage and hand back the slice of the ring it is on: the
    // wait and the address arithmetic both stay in here.
    MUTLASS_DEVICE static auto yield(SharedStorage& shared, int sequence) {
      wait(sequence);
      auto sStaged = make_tensor(make_smem_ptr(shared.smem_inverse.data()), SmemLayoutInverseP{});
      return sStaged(_, _, stage(sequence));
    }

    MUTLASS_DEVICE void load(
        SharedStorage& shared, Params const& params, int lane_idx, int sequence, int64_t record) const {
      auto sInv = make_tensor(make_smem_ptr(shared.smem_inverse.data()), SmemLayoutInverseP{})(_, _, stage(sequence));
      auto sP   = make_tensor(make_smem_ptr(shared.smem_p.data()), SmemLayoutInverseP{})(_, _, stage(sequence));
      auto gIP  = make_tensor(make_gmem_ptr(params.workspace.p + int64_t(record) * Workspace::RecordStride),
                             PInverseTileLayout{});
      // Keep the packed workspace ABI, but load padded operands [P; 0] and
      // [0; Inv]. The producer never overwrites their initialized zero halves.
      int row    = lane_idx / 2;
      int col    = (lane_idx % 2) * kInvPFragment;
      using Word = mute::uint_bit_t<128>;
      GmemCopyAtomInvP::copy(*reinterpret_cast<Word const*>(&gIP(row, col)),
                             *reinterpret_cast<Word*>(&sP(row, col)),
                             true,
                             params.desc_inverse_p);
      GmemCopyAtomInvP::copy(*reinterpret_cast<Word const*>(&gIP(row + kChunk, col)),
                             *reinterpret_cast<Word*>(&sInv(row + kChunk, col)),
                             true,
                             params.desc_inverse_p);
    }

    template <class Problem, class WorkDesc>
    MUTLASS_DEVICE void operator()(SharedStorage&  shared,
                                   Params const&   params,
                                   Problem const&  problem,
                                   WorkDesc const& work,
                                   int&            sequence) const {
      int lane_idx = mutlass::canonical_lane_idx();
      initialize_state<ValueSlice>(shared, params, problem, work);
      MUTLASS_PRAGMA_NO_UNROLL
      for (int chunk = 0; chunk < work.n_chunks; ++chunk, ++sequence) {
        // Take the stage back from the squads: they arrive on empty once they
        // have retired it.
        wait_empty(sequence);
        load(shared, params, lane_idx, sequence, Workspace::record(problem, work, chunk));
        // The LDGSTS copies are fire-and-forget: wait them out, then hand the
        // stage to the consumers.
        mute::ldgsts_wait();
        publish(sequence);
      }
      wait_warp(BarrierMap::Drained, 0);
      finalize_state<ValueSlice>(shared, params, problem, work);
    }
  };

  // Producer warp 3: the diagonal the scale SQMMA needs.  It is a single stage --
  // four squads on the same chunk read the same one -- so its ring is one deep
  // and gated on the squads RETIRING the previous chunk, not on the input ring.
  // gmem -> register -> diagonal is kDiagPerLane Elements per lane and finishes
  // long before sU does, which is what lets the two SQMMAs at the end of the
  // chunk go back to back.
  struct ScaleDiagProducer
      : NamedBarrierRing<BarrierMap::TmeStages, BarrierMap::RestoredReady, BarrierMap::RestoredFree> {
    static constexpr int ValueSlice = 3;
    // This warp shares the ring k_restored publishes on: its arrival is the second
    // of the pair's two, and the squads hand the stage back through the same
    // `empty` when they retire the state update.

    // The diagonal is not staged -- it is one tile shared by the four squads -- and
    // the ring was waited for by the k_restored half of the pair, so this only hands
    // the tile back.
    MUTLASS_DEVICE static auto tile(SharedStorage& shared) {
      return make_tensor(make_smem_ptr(shared.smem_diag.data()), SmemTileDiag{});
    }

    MUTLASS_DEVICE void load(SharedStorage& shared, Params const& params, int lane_idx, int64_t record) const {
      auto const* gTotal = params.workspace.total + int64_t(record) * Workspace::RecordStride;
      auto        sDiag  = make_tensor(make_smem_ptr(shared.smem_diag.data()), SmemTileDiag{});
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kDiagPerLane; ++i) {
        // The two SQMMAs share the accumulator's column axis, and the UK gemm's B
        // operand carries headDim in the workspace's SLOT frame: its fragment
        // column n stands for the logical key SlotToLogical(n).  So the scale
        // SQMMA's B must contract the state's column SlotToLogical(n) into
        // fragment column n -- a diagonal in (k, n) space, not in (n, n) -- and
        // the value on it is total[SlotToLogical(n)], exactly the column the
        // accumulator's n stands for.
        int n       = kDiagPerLane * lane_idx + i;
        int k       = SlotToLogical{}(n);
        sDiag(n, k) = gTotal[k];
      }
    }

    template <class Problem, class WorkDesc>
    MUTLASS_DEVICE void operator()(SharedStorage&  shared,
                                   Params const&   params,
                                   Problem const&  problem,
                                   WorkDesc const& work,
                                   int&            sequence) const {
      int lane_idx = mutlass::canonical_lane_idx();
      initialize_state<ValueSlice>(shared, params, problem, work);
      MUTLASS_PRAGMA_NO_UNROLL
      for (int chunk = 0; chunk < work.n_chunks; ++chunk, ++sequence) {
        // Take the stage back from the squads: they arrive on empty once they
        // have retired it.
        wait_empty(sequence);
        load(shared, params, lane_idx, Workspace::record(problem, work, chunk));
        // Hand the diagonal to the squads.
        publish(sequence);
      }
      wait_warp(BarrierMap::Drained, 0);
      finalize_state<ValueSlice>(shared, params, problem, work);
    }
  };

  // The OUTPUT consumer (warp_squad 2): DS -> residual -> IP -> U -> PU, plus the
  // output store.  It publishes U through `BarrierMap::UReady` and waits for the state
  // consumer's `BarrierMap::StateCommitted` before the next chunk's DS.
  struct OutputConsumer {
    static constexpr int local  = BarrierMap::Local + 1;
    static constexpr int seeded = BarrierMap::Seeded;

    using VVec = mutlass::Array<Element, 8>;

    template <class Problem, class WorkDesc>
    MUTLASS_DEVICE void operator()(SharedStorage&  shared,
                                   Params const&   params,
                                   Problem const&  problem,
                                   WorkDesc const& work,
                                   int&            sequence) const {
      int consumer_thread_idx = (int(threadIdx.x) - NumProducerThreads) & 127;
      // V's gmem view: the value axis is contiguous, so a lane's word is the eight
      // elements at its own value column and token row.
      auto v = make_tensor(
          make_gmem_ptr(params.ptr_v),
          make_layout(
              make_shape(_128{}, problem.T, problem.H, problem.B),
              make_stride(_1{}, get<1>(params.stride_v), get<2, 0>(params.stride_v), get<2, 1>(params.stride_v))));
      wait_warp(seeded, 0);
      // The DS C fragment's coordinates are a property of the tiled MMA rather than
      // of the chunk, so the token row and the eight value columns a lane owns are
      // known before the loop.  Only that row's token changes per chunk, and a tail
      // chunk would read past the token axis, so the token is clamped -- those rows
      // are multiplied by a P/inverse that K1 zeroed.
      static_assert(mute::is_same_v<typename TiledMmaDS::CLayout, typename TiledMmaPU::CLayout>);
      static_assert(mute::is_same_v<typename TiledMmaDS::CLayout, typename TiledMmaIP::CLayout>);
      TiledMmaDS tiled_mma_ds;
      TiledMmaPU tiled_mma_pu;
      TiledMmaPU tiled_mma_ip;
      auto       thread_mma_ds = tiled_mma_ds.get_thread_slice(consumer_thread_idx);
      auto       tCcDS         = thread_mma_ds.partition_C(make_identity_tensor(Shape<_32, Int<kValueTile>>{}));
      auto       tCrDSorIPorPU = partition_fragment_C(tiled_mma_ds, Shape<_32, Int<kValueTile>>{});
      int const  row           = get<0>(tCcDS(0));
      int const  v0            = int(LogicalFromTileRow{}(get<1>(tCcDS(0))));
      int const  residual_row  = row - kChunk;
      auto       thread_mma_pu = tiled_mma_pu.get_thread_slice(consumer_thread_idx);

      int const token_max = problem.T - 1;
      // One state tile and one U tile per CTA, so there is no per-squad stage.
      auto sState    = make_tensor(make_smem_ptr(shared.smem_state.data()), SmemLayoutState{})(_, _, 0);
      auto sResidual = make_tensor(make_smem_ptr(shared.smem_residual.data()), SmemLayoutU{})(_, _, 0);
      auto sU        = make_tensor(make_smem_ptr(shared.smem_u.data()), SmemLayoutU{})(_, _, 0);

      // Rows 0..15 of the [P ; inverse] tile are P, so those rows carry the output.
      // With the value frame above, a lane's kValueTile/4 accumulators come back
      // as kValueTile/4 CONSECUTIVE logical columns starting at v0 -- so the
      // output goes back to one packed convert + one 16 B store per group of eight,
      // exactly as the committed 32-wide build did.
      auto store_output = [&](int c, auto& acc) {
        if (row < work.actual_len(c)) {
          int64_t offset = (work.tme_token(c) + row) * get<1>(params.stride_out) +
                           work.head_idx * get<2, 0>(params.stride_out) +
                           work.tme_batch() * get<2, 1>(params.stride_out) + work.value_start + v0;
          MUTLASS_PRAGMA_UNROLL
          for (int h = 0; h < kValueGroups; ++h) {
            *reinterpret_cast<VVec*>(params.ptr_out + offset + 8 * h) =
                mutlass::NumericArrayConverter<Element, float, 8, mutlass::FloatRoundStyle::round_to_nearest>{}(
                    recast<mutlass::Array<float, 8>>(acc)(h));
          }
        }
      };
      // One FP32 fragment evolves [QS; KS] -> [QS; U] -> [output; U]. Padded
      // operands preserve QS in the upper half, so it never goes through BF16.
      MUTLASS_PRAGMA_NO_UNROLL
      for (int chunk = 0; chunk < work.n_chunks; ++chunk, ++sequence) {
        {
          // The state this chunk's DS reads was published by the state squad at
          // the end of the previous chunk; chunk 0 uses the item's seed.
          if (chunk > 0) {
            wait_warp(BarrierMap::StateCommitted, (sequence - 1) & 1);
          }
          if constexpr (Components::HasChunkStatesOut) {
            // Snapshot the state entering this chunk. The state squad cannot
            // overwrite it until this squad publishes BarrierMap::UReady below.
            if (consumer_thread_idx == 0) store_chunk_states(shared, params, problem, work, chunk);
          }
          // Each ring is waited for right before its first reader and handed back
          // right after its last one retires, so a producer restarts its next tile
          // as early as the chunk allows.
          auto sDecayed   = DecayedProducer::yield(shared, sequence);
          auto tCsDecayed = thread_mma_ds.partition_A(sDecayed);
          auto tCrDecayed = thread_mma_ds.make_fragment_A(tCsDecayed);
          auto tCsState   = thread_mma_ds.partition_B(sState);
          auto tCrState   = thread_mma_ds.make_fragment_B(tCsState);
          clear(tCrDSorIPorPU);
          mute::gemm(tiled_mma_ds, tCrDecayed, tCrState, tCrDSorIPorPU);
          // Keep DS and the V copies in flight together. Emitting LDGSTS before
          // DS makes the backend drain those copies before issuing SQMMA.
          if (consumer_thread_idx >= kResidualThreads) {
            int token = int(work.tme_token(chunk)) + residual_row;
            token     = token < token_max ? token : token_max;
            MUTLASS_PRAGMA_UNROLL
            for (int h = 0; h < kValueGroups; ++h) {
              auto  src = recast<mute::uint128_t>(local_tile(
                  v(_, token, work.head_idx, work.tme_batch()), Shape<_8>{}, (work.value_start + v0 + 8 * h) / 8));
              auto& dst = *reinterpret_cast<mute::uint128_t*>(
                  &sResidual(ResidualValueStoreFrame::block_base(v0, h), residual_row));
              mute::MP31_LDGSTS<mute::uint128_t>::copy(src(0), dst);
            }
          }

          mute::warpsquad_commit_batch();
          mute::warpsquad_wait();
          // The DS gemm was the decayed tile's only reader.
          DecayedProducer::release(sequence);
          // DS register r of lane l sits on tile row (l % 4) + 4 * r, which under the
          // value frame above is logical value column (C / 4) * (l % 4) + r -- i.e.
          // this lane's own consecutive run starting at v0, which is what lets the
          // output store and the V word below go back to packed 16 B accesses.
          //
          // The C fragment hands the decayed tile's rows out by warp: warps 0/1 own
          // the Q rows [0,16), whose half is the output's and stays in the
          // accumulator, and warps 2/3 own the K rows [16,32), which form the
          // residual straight into U's staging tile.  The branch is warp-uniform.
          if (consumer_thread_idx >= kResidualThreads) {
            mute::ldgsts_wait();
            // No bounds guard is needed here: K1 zeroes P and the inverse outside
            // actual_len, so the V of a clamped token row cannot reach U or the
            // output.
            // residual = V - K_decayed @ State.  V is contiguous in the value axis
            // and the lane's registers are consecutive LOGICAL columns, so the V
            // side stays two packed 8-element words.  The smem side is one packed
            // 128-bit store per group as well: the eight registers of a group land
            // on the residual frame's block `ResidualValueStoreFrame::block_base(v0, h)`, in the
            // frame's register order (see the frame note above).
            auto acc_ds_vec = recast<AccumVec const>(tCrDSorIPorPU);
            MUTLASS_PRAGMA_UNROLL
            for (int h = 0; h < kValueGroups; ++h) {
              mute::uint128_t const v_bits = *reinterpret_cast<mute::uint128_t const*>(
                  &sResidual(ResidualValueStoreFrame::block_base(v0, h), residual_row));
              ElementVec const v_values        = *reinterpret_cast<ElementVec const*>(&v_bits);
              AccumVec         residual        = mutlass::NumericArrayConverter<float, Element, 8>{}(v_values);
              auto*            residual_float4 = reinterpret_cast<float4*>(&residual);
              auto*            acc_ds_float4   = reinterpret_cast<float4 const*>(&acc_ds_vec(h));
              MUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < 2; ++i) {
                residual_float4[i] = simd::vsub(residual_float4[i], acc_ds_float4[i]);
              }
              *reinterpret_cast<ElementVec*>(&sResidual(ResidualValueStoreFrame::block_base(v0, h), residual_row)) =
                  mutlass::NumericArrayConverter<Element, float, 8, mutlass::FloatRoundStyle::round_to_nearest>{}(
                      ResidualValueStoreFrame::reorder_for_store(residual));
            }
            // Clear only KS after materializing residual. The upper QS must
            // survive IP and serve as PU's accumulation seed.
            clear(tCrDSorIPorPU);
          }
          sync_warp_squad(local);
          auto sInverseP = InversePProducer::yield(shared, sequence);
          // [0; Inv] @ residual updates only the lower half to U. IP and PU
          // both accumulate onto C; neither may discard the preserved QS.
          auto thread_mma_ip = tiled_mma_ip.get_thread_slice(consumer_thread_idx);
          auto tCsInverseP   = thread_mma_ip.partition_A(sInverseP);
          auto tCsResidual   = thread_mma_ip.partition_B(sResidual);
          auto tCrInverseP   = thread_mma_ip.make_fragment_A(tCsInverseP);
          auto tCrResidual   = thread_mma_ip.make_fragment_B(tCsResidual);
          mute::gemm(tiled_mma_ip, tCrInverseP, tCrResidual, tCrDSorIPorPU);
          mute::warpsquad_commit_batch();
          mute::warpsquad_wait();
          // Rows 16..31 of the inverse operand produce U, and the
          // row axis is handed out by warp, so the two halves split by branch rather
          // than by a coordinate test: the K-half warps own U and move it to sU,
          // while the Q-half ones take the P@U accumulator seed, i.e. the DS half
          // landing where inverse/P's own (dead) P rows were.  Which row is the
          // residual's own: the three 32x32 C fragments are one map.
          if (consumer_thread_idx >= kResidualThreads) {
            // U lands in the same value frame as the residual above -- the frame is
            // what makes the tile coordinate the UK A operand's m axis and the PU B
            // operand's n axis index stand for the logical value column, and the
            // residual's frame forces this store's coordinate to
            // `ResidualValueStoreFrame::block_base(v0, h)` in the frame's register order.  One packed
            // float->Element convert and ONE 128-bit store per group replaces the
            // sixteen stride-4 scalar stores this used to be.
            auto ip_f = recast<AccumVec const>(tCrDSorIPorPU);
            MUTLASS_PRAGMA_UNROLL
            for (int h = 0; h < kValueGroups; ++h) {
              AccumVec const ip_vec = ip_f(h);
              *reinterpret_cast<ElementVec*>(&sU(ResidualValueStoreFrame::block_base(v0, h), residual_row)) =
                  mutlass::NumericArrayConverter<Element, float, 8, mutlass::FloatRoundStyle::round_to_nearest>{}(
                      ResidualValueStoreFrame::reorder_for_store(ip_vec));
            }
          }
          if constexpr (Components::HasChunkStatesOut) {
            // The issuing lane drains the snapshot before the existing squad
            // rendezvous. BarrierMap::UReady then protects S from every state writer,
            // without an extra barrier or state-squad registers.
            if (consumer_thread_idx == 0) mute::tme_store_wait();
          }
          sync_warp_squad(local);
          // Publish U to the state squad before PU: UK and PU both read the same
          // tile, so they overlap instead of serialising.
          arrive_warp(BarrierMap::UReady);
          auto tCsU = thread_mma_pu.partition_B(sU);
          auto tCrU = thread_mma_pu.make_fragment_B(tCsU);
          auto sP   = make_tensor(make_smem_ptr(shared.smem_p.data()), SmemLayoutInverseP{})(_, _, 0);
          auto tCsP = thread_mma_pu.partition_A(sP);
          auto tCrP = thread_mma_pu.make_fragment_A(tCsP);
          mute::gemm(tiled_mma_pu, tCrP, tCrU, tCrDSorIPorPU);
          mute::warpsquad_commit_batch();
          mute::warpsquad_wait();
          // The IP and PU gemms were inverse/P's only readers.
          InversePProducer::release(sequence);

          // Retire this output before the next DS overwrites the single C fragment.
          if (consumer_thread_idx < kResidualThreads) {
            store_output(chunk, tCrDSorIPorPU);
          }
          sync_warp_squad(local);
        }
      }
      // No BarrierMap::Drained arrival here: the STATE squad owns the state export.
    }
  };

  // The STATE consumer (warp_squad 1): Scale (state @ diag) -> wait BarrierMap::UReady -> UK
  // (U^T @ k_restored) -> state writeback -> BarrierMap::StateCommitted.  It publishes the
  // state tile that the output consumer's next DS reads.
  struct StateConsumer {
    static constexpr int local  = BarrierMap::Local + 0;
    static constexpr int seeded = BarrierMap::Seeded;

    template <class Problem, class WorkDesc>
    MUTLASS_DEVICE void operator()(SharedStorage&  shared,
                                   Params const&   params,
                                   Problem const&  problem,
                                   WorkDesc const& work,
                                   int&            sequence) const {
      int consumer_thread_idx = (int(threadIdx.x) - NumProducerThreads) & 127;
      wait_warp(seeded, 0);
      auto       sState = make_tensor(make_smem_ptr(shared.smem_state.data()), SmemLayoutState{})(_, _, 0);
      TiledMmaUK tiled_mma_uk;
      auto       thread_mma_uk = tiled_mma_uk.get_thread_slice(consumer_thread_idx);
      // Coordinates from the frame-relabelled atom: its N index is the slot and
      // the reported column is already the logical key.
      auto tCcUK = TiledMmaUKCoords{}
                       .get_thread_slice(consumer_thread_idx)
                       .partition_C(make_identity_tensor(Shape<Int<kValueTile>, _128>{}));
      auto tCrUK        = partition_fragment_C(tiled_mma_uk, Shape<Int<kValueTile>, _128>{});
      using StateVector = mutlass::Array<Element, 8>;
      TiledMmaScale tiled_mma_scale;
      auto          thread_mma_scale = tiled_mma_scale.get_thread_slice(consumer_thread_idx);
      auto          tCsScale         = thread_mma_scale.partition_A(sState);
      auto          tCrScale         = thread_mma_scale.make_fragment_A(tCsScale);

      MUTLASS_PRAGMA_NO_UNROLL
      for (int chunk = 0; chunk < work.n_chunks; ++chunk, ++sequence) {
        {
          // The k_restored + diagonal pair publishes this chunk's tiles; the
          // diagonal feeds Scale, which reads the previous state.
          auto sKRestored = RestoredProducer::yield(shared, sequence);
          auto sDiag      = ScaleDiagProducer::tile(shared);
          auto tCsDiag    = thread_mma_scale.partition_B(sDiag);
          auto tCrDiag    = thread_mma_scale.make_fragment_B(tCsDiag);
          // clear, Scale, UK == clear, UK, Scale: both accumulate into tCrUK.
          clear(tCrUK);
          mute::gemm(tiled_mma_scale, tCrScale, tCrDiag, tCrUK);
          // U is the output squad's; UK cannot start until BarrierMap::UReady.
          wait_warp(BarrierMap::UReady, sequence & 1);
          auto sU      = make_tensor(make_smem_ptr(shared.smem_u.data()), SmemLayoutU{})(_, _, 0);
          auto tCsUK_A = thread_mma_uk.partition_A(sU);
          auto tCsUK_B = thread_mma_uk.partition_B(sKRestored);
          auto tCrUK_A = thread_mma_uk.make_fragment_A(tCsUK_A);
          auto tCrUK_B = thread_mma_uk.make_fragment_B(tCsUK_B);
          mute::gemm(tiled_mma_uk, tCrUK_A, tCrUK_B, tCrUK);
          mute::warpsquad_commit_batch();
          mute::warpsquad_wait();
          // k_restored and the diagonal are only read by the state update.
          ScaleDiagProducer::release(sequence);
          RestoredProducer::release(sequence);
          // One packed float->bf16 convert per eight elements, then one 16 B store:
          // the slot frame turns a lane's eight registers into eight consecutive
          // keys, and the group's row is the fragment's m (the state tile's row).
          using StateFloatVec = mutlass::Array<float, 8>;
          auto tCrUKVec       = recast<StateFloatVec const>(tCrUK);
          MUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < kValueTile / 8; ++j) {
            int v = get<0>(tCcUK(8 * j));
            int k = get<1>(tCcUK(8 * j));
            *reinterpret_cast<StateVector*>(&sState(v, k)) =
                mutlass::NumericArrayConverter<Element, float, 8, mutlass::FloatRoundStyle::round_to_nearest>{}(
                    tCrUKVec(j));
          }
          sync_warp_squad(local);
          // Publish the updated state for the output squad's next DS.
          arrive_warp(BarrierMap::StateCommitted);
        }
      }
      // Publish the final state once per CTA, matching each producer's one
      // wait_warp(BarrierMap::Drained, 0) before finalize_state.
      arrive_warp(BarrierMap::Drained);
    }
  };

  using ProducerWarp0 = DecayedProducer;
  using ProducerWarp1 = RestoredProducer;
  using ProducerWarp2 = InversePProducer;
  using ProducerWarp3 = ScaleDiagProducer;

  static MUTLASS_DEVICE void init_barriers(int tid) {
    mutlass::arch::allocate_async_barriers(BarrierMap::NumBarriers);
    if (tid != 0) {
      return;
    }
    for (int s = 0; s < TmeStages; ++s) {
      named_barrier_init(BarrierMap::DecayedReady + s, 1);
      named_barrier_init(BarrierMap::DecayedFree + s, NumOutputWarps);
      named_barrier_init(BarrierMap::RestoredReady + s, 2);
      named_barrier_init(BarrierMap::RestoredFree + s, 2 * NumStateWarps);
      named_barrier_init(BarrierMap::InvPReady + s, 1);
      named_barrier_init(BarrierMap::InvPFree + s, NumOutputWarps);
    }
    for (int s = 0; s < NumStateSquads + NumOutputSquads; ++s) {
      named_barrier_init(BarrierMap::Local + s, mutlass::NumWarpsPerWarpSquad);
    }
    named_barrier_init(BarrierMap::Seeded, NumProducerWarps);
    named_barrier_init(BarrierMap::Drained, NumStateWarps);
    named_barrier_init(BarrierMap::UReady, NumOutputWarps);
    named_barrier_init(BarrierMap::StateCommitted, NumStateWarps);
  }

  static MUTLASS_DEVICE void init_shared_storage(SharedStorage& shared, int tid) {
    for (int i = tid; i < cosize_v<SmemTileDiag>; i += blockDim.x) {
      shared.smem_diag[i] = Element(0);
    }
    for (int i = tid; i < cosize_v<SmemLayoutInverseP>; i += blockDim.x) {
      shared.smem_inverse[i] = Element(0);
      shared.smem_p[i]       = Element(0);
    }
  }
};

}  // namespace mate::flat::kda::recurrence
