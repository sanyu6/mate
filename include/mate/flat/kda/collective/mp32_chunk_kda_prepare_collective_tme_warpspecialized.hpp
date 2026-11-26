#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>
#include <mutlass/numeric_conversion.h>

#include <cstdint>
#include <limits>
#include <mute/arch/copy_mp31.hpp>
#include <mute/arch/copy_mp31_tme.hpp>
#include <mute/atom/copy_atom.hpp>
#include <mute/atom/copy_traits_mp31.hpp>
#include <mute/swizzle_layout.hpp>
#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
#include <mutlass/pipeline/pipeline.hpp>
#include <type_traits>

#include "mate/common/layout.hpp"
#include "mate/common/mma_mp31_sqmma.hpp"
#include "mate/flat/collective/flat_collective_inverse_nxn.hpp"
#include "mate/flat/collective/flat_collective_mma_tme_sqmma.hpp"
#include "mate/flat/kda/chunk_kda_components.hpp"
#include "mate/flat/kda/kda_permutes.hpp"
#include "mate/flat/kda/named_barrier.hpp"
#include "mate/flat/simd_helper.hpp"

// The MP31 frontend exposes the correctly-rounded reciprocal-square-root
// builtin in BuiltinsMTGPU.def but does not declare it in the public math
// headers.  Declare the device entries explicitly so mcc can select the
// native instructions without the generic rsqrtf/division edge-case paths.
extern "C" __device__ float __musa_rsqrt_rn_f(float);

namespace mate::flat::kda {

namespace prepare {

using namespace mute;

template <class Components_, class TileShape_, class StrideQ_, class StrideK_, class StrideG_>
struct Mp32ChunkKdaPrepareCollectiveTmeWarpSpecialized {
  static constexpr float kLog2E = 1.4426950408889634074f;

  using ArchTag                       = mutlass::arch::Mp32;
  using Components                    = Components_;
  using Element                       = typename Components::Element;
  using TileShape                     = TileShape_;
  using StrideQ                       = StrideQ_;
  using StrideK                       = StrideK_;
  using StrideV                       = ChunkKdaVoStride;
  using StrideG                       = StrideG_;
  static constexpr bool HasGateParams = Components::HasGateParams;
  static constexpr bool NormalizeQK   = Components::NormalizeQK;
  static constexpr bool IsVarlen      = Components::IsVarlen;

  static constexpr int kChunk    = Components::kChunk;
  static constexpr int kHeadDim  = Components::kHeadDim;
  static constexpr int kValueDim = Components::kValueDim;
  using Workspace                = typename Components::Workspace;
  using HeadDimPermute           = PermuteN<mutlass::arch::Mp32, Workspace::kHeadDim>;
  using LogicalToSlot            = typename HeadDimPermute::LogicalToSlot;
  static_assert(decltype(get<0>(TileShape{}))::value == kChunk);
  static_assert(decltype(get<1>(TileShape{}))::value == kChunk);
  static_assert(decltype(get<2>(TileShape{}))::value == kHeadDim);
  static constexpr int kValueMmaN = kValueDim < 64 ? kValueDim : 64;
  static_assert(decltype(get<0>(TileShape{}))::value == decltype(get<1>(TileShape{}))::value);
  static_assert(kChunk == 16, "the 32x32 SQMMA warp-role permutation requires chunk size 16");

  // Two warp squads: consumer warps 0-3 run the SQMMA/epilogue path, while
  // producer-squad warps 0-3 (global warps 4-7) feed it and store the result.
  //
  //   producer-squad warp 0: load Q -> registers -> RMS norm -> smem_norm_q
  //   producer-squad warp 1: load K -> registers -> RMS norm -> smem_norm_k
  //   producer-squad warp 2: load G -> gate cumsum (fp32) + workspace.total
  //   producer-squad warp 3: TMA store after its full barrier
  //   consumer warps:   decayed = norm*q * gate, k_inverse = norm*k / gate,
  //                     SQMMA gemm, inverse + k_restored + P epilogue
  static constexpr int NumProducerWarps       = 4;
  static constexpr int NumConsumerWarps       = 4;
  static constexpr int NumThreadsPerBlock     = (NumProducerWarps + NumConsumerWarps) * mutlass::NumThreadsPerWarp;
  static constexpr int NormalizeQWarp         = 0;
  static constexpr int NormalizeKWarp         = 1;
  static constexpr int GateWarp               = 2;
  static constexpr int StoreWarp              = 3;
  static constexpr int ConsumerWarpBase       = 0;
  static constexpr int ProducerWarpBase       = NumConsumerWarps;
  static constexpr int ProducerNormalizeQWarp = ProducerWarpBase + NormalizeQWarp;
  static constexpr int ProducerNormalizeKWarp = ProducerWarpBase + NormalizeKWarp;
  static constexpr int ProducerGateWarp       = ProducerWarpBase + GateWarp;
  static constexpr int ProducerStoreWarp      = ProducerWarpBase + StoreWarp;
  static_assert(NumConsumerWarps * mutlass::NumThreadsPerWarp == kHeadDim);

  static constexpr int NumStageBarriers   = static_cast<int>(PrepareBarrier::NumBarriers);
  static constexpr int SmemAlignmentBytes = 256;
  static constexpr int VecSize            = 4;

  // Async barrier ids are hardware ids, not shared-memory byte offsets.
  // Keep the named barrier range and the two pipeline ranges disjoint.

  // The MUTLASS builder requires at least two stages, but the producer
  // register path only needs the consumer's operand tiles.
  using BuilderStageCount = mutlass::gemm::collective::StageCount<2>;
  using TileShapeQK       = Shape<Int<2 * kChunk>, Int<2 * kChunk>, Int<kHeadDim>>;
  // The SQMMA operand layouts follow ArchTag: MP32 needs its own SQMMA
  // encoding (the MP31 atom does not survive the backend on an mp_32 target),
  // while the workspace drain path is selected separately below.
  using CollectiveMmaQK =
      typename mutlass::gemm::collective::CollectiveBuilder<ArchTag,
                                                            mutlass::arch::OpClassTensorOp,
                                                            Element,
                                                            StrideQ,
                                                            2,
                                                            Element,
                                                            StrideK,
                                                            2,
                                                            float,
                                                            TileShapeQK,
                                                            Shape<_1, _1, _1>,
                                                            BuilderStageCount,
                                                            mutlass::gemm::KernelTmeWarpSpecialized>::CollectiveOp;
  static constexpr int BuilderStages = CollectiveMmaQK::DispatchPolicy::Stages;
  static_assert(BuilderStages == BuilderStageCount::value);

  using TiledMmaQK = typename CollectiveMmaQK::TiledMma;
  static_assert(decltype(size(TiledMmaQK{}))::value == NumConsumerWarps * mutlass::NumThreadsPerWarp);

  static constexpr int PipelineStages = 1;
  using SmemLayoutDecayed =
      decltype(mate::unstage_smem_layout(typename CollectiveMmaQK::SmemLayoutA{}, Int<PipelineStages>{}));
  using SmemLayoutKInverse =
      decltype(mate::unstage_smem_layout(typename CollectiveMmaQK::SmemLayoutB{}, Int<PipelineStages>{}));
  using CollectiveInverseNxN = collective::CollectiveLowerTriangularInverseNxN<Element, kChunk>;

  // Producer/consumer tiles: plain row-major layouts (no TME, no SQMMA
  // access), fp32 for the gate cumsum, bf16 for the normalized tiles.
  using NormLayout   = Layout<Shape<Int<kChunk>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>;
  using CumsumLayout = NormLayout;

  // Staging tile for the P publish: the SQMMA C fragment scatters into it, and
  // it is read back one contiguous vector per lane.
  using PStagingLayout = Layout<Shape<Int<kChunk>, Int<kChunk>>, Stride<Int<kChunk>, _1>>;
  // The whole 16x16 staging tile is published by a single warp: 32 lanes x 8
  // bf16 = 512 bytes, one 16-byte store per lane.
  static constexpr int kPublishPerLane = kChunk * kChunk / mutlass::NumThreadsPerWarp;
  static_assert(kChunk * kChunk % mutlass::NumThreadsPerWarp == 0);

  static constexpr int KInverseValidBytes = kChunk * kHeadDim * sizeof(Element);

  // k_restored lives in the k_inverse pad half and uses the same canonical
  // 16x128 tile layout as the fused-pass stores.
  using SmemLayoutKRestoredTile = decltype(take<0, 2>(SmemLayoutKInverse{}));

  // Async LSU (robust LDGSTS) copies for the Q/K raw tiles: the hardware
  // descriptor bounds-checks every access, so partial-chunk tail rows need
  // no invalid-line handling on the load side.
  using LsuCopyAtom    = mute::MP31_ROBUST_LDGSTS<mute::uint128_t>;
  using GmemTiledCopyQ = decltype(mutlass::gemm::collective::detail::make_simt_tiled_copy<mutlass::NumThreadsPerWarp,
                                                                                          Element,
                                                                                          128 / sizeof_bits_v<Element>,
                                                                                          StrideQ,
                                                                                          kChunk,
                                                                                          kHeadDim,
                                                                                          LsuCopyAtom>());
  using GmemTiledCopyK = decltype(mutlass::gemm::collective::detail::make_simt_tiled_copy<mutlass::NumThreadsPerWarp,
                                                                                          Element,
                                                                                          128 / sizeof_bits_v<Element>,
                                                                                          StrideK,
                                                                                          kChunk,
                                                                                          kHeadDim,
                                                                                          LsuCopyAtom>());
  using GmemTiledCopyG = decltype(mutlass::gemm::collective::detail::make_simt_tiled_copy<mutlass::NumThreadsPerWarp,
                                                                                          Element,
                                                                                          128 / sizeof_bits_v<Element>,
                                                                                          StrideG,
                                                                                          kChunk,
                                                                                          kHeadDim,
                                                                                          LsuCopyAtom>());

  template <class StorageElement, class SmemLayout>
  using SmemArray = array_aligned<StorageElement, cosize_v<SmemLayout>, SmemAlignmentBytes>;

  struct SharedStorage {
    SmemArray<Element, NormLayout>        smem_norm_q;
    SmemArray<Element, NormLayout>        smem_norm_k;
    SmemArray<Element, CumsumLayout>      smem_g_cumsum;
    SmemArray<Element, SmemLayoutDecayed> smem_decayed;
    // k_inverse is the 2x-swizzled SQMMA B tile: 4 KiB data plus 4 KiB pad.
    // The union uses [0, 4 KiB) for k_inverse and [4 KiB, 8 KiB) for
    // k_restored; k_restored must stay at offset 4 KiB because the drain
    // describes it with the k_inverse layout. The two physical pipeline pairs
    // defer reuse until the corresponding consumer phase has completed.
    //
    // The NxN inverse workspace (16x16 float matrix plus the solve scratch)
    // deliberately lives OUTSIDE this union. Aliasing it with smem_k_inverse
    // lets a consumer warp that has already finished its P scatter run ahead
    // into the next chunk's scale pass and overwrite the matrix while the
    // solve is still reading it, which corrupts the inverse output. The extra
    // 1 KiB is the price of keeping the matrix private.
    union {
      SmemArray<Element, SmemLayoutKInverse> smem_k_inverse;
      struct {
        uint8_t                                                       dummy_pad[KInverseValidBytes];
        array_aligned<Element, kChunk * kHeadDim, SmemAlignmentBytes> smem_k_restored;
      };
    };
    typename CollectiveInverseNxN::SharedStorage smem_inverse_workspace;
    // 16x16 bf16 P tile staged by consumer warps 0/1 before the coalesced write.
    array_aligned<Element, kChunk * kChunk, 16> smem_p_stage;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static_assert(SharedStorageSize <= 32 * 1024, "shared memory must fit 8 CTAs per MP");

  struct Params;

  struct Arguments {
    Element const*                      ptr_q;
    Element const*                      ptr_k;
    Element const*                      ptr_g;
    Element const*                      ptr_beta;
    float const*                        ptr_A_log;
    float const*                        ptr_dt_bias;
    StrideQ                             stride_q;
    StrideK                             stride_k;
    StrideG                             stride_g;
    float                               scale;
    float                               lower_bound;
    typename Workspace::MutablePointers workspace;
  };

  struct Params {
    Element const*                      ptr_q;
    Element const*                      ptr_k;
    Element const*                      ptr_g;
    Element const*                      ptr_beta;
    float const*                        ptr_A_log;
    float const*                        ptr_dt_bias;
    StrideQ                             stride_q;
    StrideK                             stride_k;
    StrideG                             stride_g;
    float                               scale;
    float                               lower_bound;
    typename Workspace::MutablePointers workspace;
    mute::RobustDescriptor              desc_q;
    mute::RobustDescriptor              desc_k;
    mute::RobustDescriptor              desc_g;
  };

  // Warp-role synchronization: norm producers are warps 4/5/6 and consumers
  // are warps 0-3; decayed producers are warps 0-3 and the store consumer is
  // warp 7.  Norm is released after the fused pass to overlap the next fill.
  using PipelineNorm         = mutlass::PipelineAsync<PipelineStages>;
  using PipelineDecayed      = mutlass::PipelineAsync<PipelineStages>;
  using PipelineNormState    = typename PipelineNorm::PipelineState;
  using PipelineDecayedState = typename PipelineDecayed::PipelineState;

  struct BarrierStorage;

  struct Pipeline {
    PipelineNorm         norm;
    PipelineDecayed      decayed;
    PipelineNormState    norm_read;
    PipelineNormState    norm_write;
    PipelineDecayedState decayed_read;
    PipelineDecayedState decayed_write;

    static MUTLASS_DEVICE typename PipelineNorm::Params make_norm_params() {
      return {uint32_t(NumProducerWarps - 1), uint32_t(NumConsumerWarps)};
    }

    static MUTLASS_DEVICE typename PipelineDecayed::Params make_decayed_params() {
      return {uint32_t(NumConsumerWarps), 1};
    }

    MUTLASS_DEVICE explicit Pipeline(BarrierStorage* storage)
        : norm(make_norm_params(), static_cast<uint32_t>(reinterpret_cast<uint64_t>(&storage->pipeline_norm))),
          decayed(make_decayed_params(), static_cast<uint32_t>(reinterpret_cast<uint64_t>(&storage->pipeline_decayed))),
          norm_read{},
          norm_write(mutlass::make_producer_start_state<PipelineNorm>()),
          decayed_read{},
          decayed_write(mutlass::make_producer_start_state<PipelineDecayed>()) {
    }
  };

  struct alignas(1) BarrierStorage {
    uint8_t stage[NumStageBarriers];
    uint8_t pipeline_norm[PipelineNorm::NumBarriers];
    uint8_t pipeline_decayed[PipelineDecayed::NumBarriers];
  };

  // The inverse runs on one warp after a block-wide rendezvous, so its own
  // entry barrier would be a barrier of one.
  struct PStagedBarrierSync {
    int local_tid;

    MUTLASS_DEVICE void operator()() const {
      if (mutlass::canonical_lane_idx() == 0) {
        named_barrier_arrive_and_wait(PrepareBarrier::PStagedReady);
      }
      __syncwarp();
    }
  };

  struct InverseSmemBarrierSync {
    int local_tid;

    MUTLASS_DEVICE void operator()() const {
      if (mutlass::canonical_lane_idx() == 0) {
        named_barrier_arrive_and_wait(PrepareBarrier::InverseSmemReady);
      }
      __syncwarp();
    }
  };

  static MUTLASS_DEVICE void init_stage_barriers(BarrierStorage* barrier_storage, int tid) {
    (void)barrier_storage;
    if (tid == 0) {
      // MP31: only the two KKT warps publish the matrix.  MP32: the ABI row
      // interleaving puts KKT rows in both warp pairs, so all four arrive.
      // One rendezvous per warp pair; the pairs never wait on each other.
      named_barrier_init(PrepareBarrier::InverseSmemReady, 2);
      named_barrier_init(PrepareBarrier::PStagedReady, 2);
    }
  }

  template <class ProblemSize>
  static Params to_underlying_arguments(ProblemSize const& problem_size, Arguments const& args) {
    int32_t                records = Components::workspace_records(problem_size);
    mute::RobustDescriptor desc_q  = mute::make_robust_desc(
        args.ptr_q, static_cast<size_t>(problem_size.B) * problem_size.T * problem_size.Hqk * kHeadDim);
    mute::RobustDescriptor desc_k = mute::make_robust_desc(
        args.ptr_k, static_cast<size_t>(problem_size.B) * problem_size.T * problem_size.Hqk * kHeadDim);
    mute::RobustDescriptor desc_g = mute::make_robust_desc(
        args.ptr_g, static_cast<size_t>(problem_size.B) * problem_size.T * problem_size.Hqk * kHeadDim);
    return Params{
        args.ptr_q,
        args.ptr_k,
        args.ptr_g,
        args.ptr_beta,
        args.ptr_A_log,
        args.ptr_dt_bias,
        args.stride_q,
        args.stride_k,
        args.stride_g,
        args.scale,
        args.lower_bound,
        args.workspace,
        desc_q,
        desc_k,
        desc_g,
    };
  }

  static MUTLASS_DEVICE float sigmoid_fast(float x) {
    return 1.0f / (1.0f + ::mate::flat::detail::fast_exp2(-x * kLog2E));
  }

  static MUTLASS_DEVICE Element inactive_gate_raw_element() {
    if constexpr (HasGateParams) {
      return -std::numeric_limits<Element>::infinity();
    } else {
      return Element(0.0f);
    }
  }

  // simd::vrcp expresses this as a real <4 x float> division so the backend can
  // fold it into one AF32.RCP.BST4; see that helper for why four scalar
  // __musa_rcp_rn_f calls cannot reach the same combine.
  static MUTLASS_DEVICE float4 rcp_native(float4 const& x) {
    return simd::vrcp(x);
  }

  template <class TensorQK>
  static MUTLASS_DEVICE float4 load_qk_vec4(TensorQK&& sQk, int row, int col_base) {
    return simd::load_packed4_as_float4(&sQk(make_coord(row, col_base)));
  }

  template <class TensorQK>
  static MUTLASS_DEVICE void store_qk_vec4(TensorQK&& sQk, int row, int col_base, float4 value) {
    simd::store_float4_to_packed4_rn(&sQk(make_coord(row, col_base)), value);
  }

  // Reduce over SubWarpSize lanes.  MP31 has a native SHUFFLE.IDX
  // instruction, but the butterfly form (__shfl_xor_sync) is emulated
  // through a global-memory buffer; express the reduction as index shuffles
  // with an absolute source lane so it lowers to the native instruction.
  template <int SubWarpSize>
  static MUTLASS_DEVICE void subwarp_reduce_sum(float& value) {
    int lane = mutlass::canonical_lane_idx();
    MUTE_UNROLL
    for (int offset = SubWarpSize / 2; offset > 0; offset >>= 1) {
      value += __shfl_sync(uint32_t(-1), value, lane ^ offset, SubWarpSize);
    }
  }

  template <class ProblemSize, class WorkDesc>
  static MUTLASS_DEVICE float load_beta_register(Params const&      params,
                                                 ProblemSize const& problem_size,
                                                 WorkDesc const&    work_desc,
                                                 int                chunk_idx,
                                                 int                lane,
                                                 bool               is_not_full_chunk) {
    bool    token_valid = lane < kChunk && (!is_not_full_chunk || lane < work_desc.actual_len(chunk_idx));
    int64_t offset =
        (int64_t(work_desc.tme_batch()) * problem_size.T + int64_t(work_desc.tme_token(chunk_idx) + lane)) *
            problem_size.H +
        int64_t(work_desc.head_idx);
    float raw_beta =
        token_valid ? static_cast<float>(params.ptr_beta[offset]) : -std::numeric_limits<float>::infinity();
    return sigmoid_fast(raw_beta);
  }

  // -------------------------------------------------------------------------
  // Producer squad (global warps 4-7)
  // -------------------------------------------------------------------------

  // Producer-squad warps 0/1: normalize the raw Q/K tile in smem in place.
  // The raw tile was filled by a robust LDGSTS copy, so every row is present
  // (tail rows hold descriptor-safe garbage that the consumers mask out).
  // Each pass covers RowsPerPass rows at once: a row is split over LanesPerRow
  // lanes and the warp's remaining lanes take the following rows. Widening the
  // per-lane element count is what pays here, because the two per-row serial
  // costs - the shuffle reduction and the correctly-rounded rsqrt sequence -
  // are issued once per pass and shared by every row in it. Eight lanes per row
  // puts four rows in flight and costs three shuffle levels instead of five,
  // which is roughly half the normalize instructions.
  static MUTLASS_DEVICE void normalize_qk(Element* s_norm_data, float scale, int actual_len, bool is_not_full_chunk) {
    constexpr int LanesPerRow = 8;
    constexpr int ElemPerLane = kHeadDim / LanesPerRow;                    // 16
    constexpr int VecPerLane  = ElemPerLane / VecSize;                     // 4
    constexpr int RowsPerPass = mutlass::NumThreadsPerWarp / LanesPerRow;  // 4
    int           lane        = mutlass::canonical_lane_idx();
    int           lane_in_row = lane % LanesPerRow;
    int           row_in_pass = lane / LanesPerRow;
    int           col_base    = lane_in_row * ElemPerLane;
    auto          sNorm       = make_tensor(make_smem_ptr(s_norm_data), NormLayout{});
    MUTLASS_PRAGMA_UNROLL
    for (int pass = 0; pass < kChunk / RowsPerPass; ++pass) {
      int    row = pass * RowsPerPass + row_in_pass;
      float4 v[VecPerLane];
      MUTE_UNROLL
      for (int i = 0; i < VecPerLane; ++i) {
        v[i] = load_qk_vec4(sNorm, row, col_base + i * VecSize);
      }
      float sum = 0.0f;
      if constexpr (NormalizeQK) {
        // Independent partial chains, then a short tree: dot4 is a serial
        // four-deep fma chain and four of them in a row would make the
        // reduction latency-bound instead of throughput-bound.
        float part[VecPerLane];
        MUTE_UNROLL
        for (int i = 0; i < VecPerLane; ++i) {
          part[i] = 0.0f;
          simd::dot4(v[i], v[i], part[i]);
        }
        MUTE_UNROLL
        for (int i = 0; i < VecPerLane; ++i) {
          sum += part[i];
        }
        subwarp_reduce_sum<LanesPerRow>(sum);
        sum = __musa_rsqrt_rn_f(sum + 1.0e-6f);
      } else {
        sum = 1.0f;
      }
      // Robust-copy tail rows hold garbage: zero them like the sync-load path.
      bool  valid  = !is_not_full_chunk || row < actual_len;
      float factor = sum * scale;
      MUTE_UNROLL
      for (int i = 0; i < VecPerLane; ++i) {
        store_qk_vec4(sNorm, row, col_base + i * VecSize, valid ? simd::vmul(v[i], factor) : simd::splat4(0.0f));
      }
    }
  }

  // Producer-squad warp 2: gate cumsum straight from gmem.  Each lane owns one column
  // strip and keeps the running prefix; the fp32 exp2(prefix) is published
  // to smem, the final value to workspace.total.
  template <class ProblemSize, class WorkDesc>
  static MUTLASS_DEVICE void compute_g_cumsum(SharedStorage&     shared_storage,
                                              Params const&      params,
                                              ProblemSize const& problem_size,
                                              WorkDesc const&    work_desc,
                                              int                chunk_idx) {
    auto  sGCumsum   = make_tensor(make_smem_ptr(shared_storage.smem_g_cumsum.data()), CumsumLayout{});
    int   lane       = mutlass::canonical_lane_idx();
    int   col_base   = lane * VecSize;
    int   actual_len = work_desc.actual_len(chunk_idx);
    float a_log_exp  = 1.0f;
    if constexpr (HasGateParams) {
      if (lane == 0) {
        a_log_exp = ::mate::flat::detail::fast_exp2(params.ptr_A_log[work_desc.head_idx] * kLog2E);
      }
      a_log_exp = __shfl_sync(uint32_t(-1), a_log_exp, 0, mutlass::NumThreadsPerWarp);
    }
    float4 dt = simd::splat4(0.0f);
    if constexpr (HasGateParams) {
      dt = simd::load_float4(params.ptr_dt_bias + int64_t(work_desc.head_idx) * kHeadDim + col_base);
    }
    float  gate_scale = -a_log_exp * kLog2E;
    float  gate_coeff = params.lower_bound * kLog2E;
    float4 cumsum     = simd::splat4(0.0f);
    for (int row = 0; row < kChunk; ++row) {
      float4 raw_g = simd::splat4(static_cast<float>(inactive_gate_raw_element()));
      if (row < actual_len) {
        raw_g = load_qk_vec4(sGCumsum, row, col_base);
      }
      float4 gate = raw_g;
      if constexpr (HasGateParams) {
        float4 exp_arg = simd::vmul(simd::vadd(raw_g, dt), gate_scale);
        gate           = simd::vmul(rcp_native(simd::vadd(simd::fast_exp2(exp_arg), 1.0f)), gate_coeff);
      }
      cumsum = simd::vadd(cumsum, gate);
      store_qk_vec4(sGCumsum, row, col_base, simd::fast_exp2(cumsum));
    }
    int    record = Workspace::record(problem_size, work_desc, chunk_idx);
    float4 total  = simd::fast_exp2(cumsum);
    // `total` is an Element row: one rounding to the workspace type here, so K2
    // and the reference replay read back exactly the same decay product.
    using TotalVec = mutlass::Array<float, 4>;
    mutlass::NumericArrayConverter<Element, float, 4, mutlass::FloatRoundStyle::round_to_nearest> round_total;
    *reinterpret_cast<mutlass::Array<Element, 4>*>(params.workspace.total + int64_t(record) * Workspace::RecordStride +
                                                   col_base) = round_total(*reinterpret_cast<TotalVec const*>(&total));
  }

  // Producer-squad warp 3: drain the decayed tile and k_restored once consumers
  // published them.
  template <class ProblemSize, class WorkDesc>
  static MUTLASS_DEVICE void store_decayed(SharedStorage&     shared_storage,
                                           Params const&      params,
                                           ProblemSize const& problem_size,
                                           WorkDesc const&    work_desc,
                                           int                chunk_idx) {
    // MP32 has no codegen-usable TME store (mcc lowers __musa_tme_st_* to a
    // form its own assembler rejects), so the store warp drains both tiles
    // with vectorized LSU stores straight from the SQMMA operand layouts --
    // the same drain the native MP32 K1 collective uses.
    int  lane         = mutlass::canonical_lane_idx();
    int  record       = Workspace::record(problem_size, work_desc, chunk_idx);
    auto s_decayed    = make_tensor(make_smem_ptr(shared_storage.smem_decayed.data()), SmemLayoutDecayed{})(_, _, 0);
    auto s_k_restored = make_tensor(make_smem_ptr(shared_storage.smem_k_restored.data()), SmemLayoutKRestoredTile{});
    // 16 bytes per lane per iteration. recast<> asks mute to divide the
    // swizzled layout into vectors itself, so the addresses come from the same
    // algebra that the scalar accesses use (a hand computed offset does not:
    // the swizzle is not part of the sliced layout() offset).
    constexpr int kDrainVec = VecSize * 2;
    constexpr int kRowVecs  = kHeadDim / kDrainVec;
    using StoreVec          = mutlass::Array<Element, kDrainVec>;
    static_assert(kHeadDim % kDrainVec == 0);
    auto s_decayed_vec    = recast<StoreVec>(s_decayed);
    auto s_k_restored_vec = recast<StoreVec>(s_k_restored);

    // `record` selects the chunk-major record block: stride by
    // Workspace::RecordStride.  The vector COUNT below is a different quantity
    // -- it is the number of 16 B vectors in the *field* (DecayedElements),
    // not the stride -- which is why the two must not share a constant.
    Element*      g_decayed      = params.workspace.decayed + int64_t(record) * Workspace::RecordStride;
    constexpr int DecayedVectors = Workspace::DecayedElements / kDrainVec;
    for (int vec = lane; vec < DecayedVectors; vec += mutlass::NumThreadsPerWarp) {
      int row     = vec / kRowVecs;
      int col_vec = vec % kRowVecs;
      *reinterpret_cast<StoreVec*>(g_decayed + row * kHeadDim + col_vec * kDrainVec) =
          s_decayed_vec(make_coord(row, col_vec));
    }

    // k_restored goes out through the MP32 headDim slot frame
    // (kda_permutes.hpp), which is what makes K2's UK accumulator a
    // lane's eight consecutive logical keys.
    Element*      g_k_restored     = params.workspace.k_restored + int64_t(record) * Workspace::RecordStride;
    constexpr int KRestoredVectors = Workspace::KRestoredElements / kDrainVec;
    for (int vec = lane; vec < KRestoredVectors; vec += mutlass::NumThreadsPerWarp) {
      int            row     = vec / kRowVecs;
      int            col_vec = vec % kRowVecs;
      StoreVec const packed  = s_k_restored_vec(make_coord(row, col_vec));
      Element*       dst     = g_k_restored + row * kHeadDim;
      MUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < kDrainVec; ++s) {
        int logical_k                   = col_vec * kDrainVec + s;
        dst[LogicalToSlot{}(logical_k)] = packed[s];
      }
    }
  }

  // -------------------------------------------------------------------------
  // Consumer warps
  // -------------------------------------------------------------------------

  // All four consumer warps: gate the producer's normalized tiles and
  // produce decayed + k_inverse in the SQMMA operand layouts, in one fused
  // pass over the rows (each gate row is loaded once and applied to both
  // its Q and K rows).  k_restored = k_inverse * total is computed here
  // too: total is the last row of the gate cumsum tile, so no separate
  // g_total buffer is needed.  No validity checks: the producers already
  // publish zeros for inactive norm rows and zero gates for inactive
  // cumsum rows; the reciprocal epsilon keeps 0 * rcp(0) finite.
  template <class TensorK,
            class TensorQ,
            class TensorG,
            class TensorDecayed,
            class TensorKInverse,
            class TensorKRestored>
  static MUTLASS_DEVICE void scale_normalized_qk(TensorK&&         sK,
                                                 TensorQ&&         sQ,
                                                 TensorG&&         sGCumsum,
                                                 TensorDecayed&&   sDecayed,
                                                 TensorKInverse&&  sKInverse,
                                                 TensorKRestored&& sKRestored,
                                                 int               local_tid) {
    constexpr int   ThreadsPerRow     = 32;
    constexpr int   RowsPerWarp       = mutlass::NumThreadsPerWarp / ThreadsPerRow;
    constexpr int   ElementsPerThread = kHeadDim / ThreadsPerRow;
    constexpr int   VectorsPerThread  = ElementsPerThread / VecSize;
    constexpr int   RowsPerWave       = NumConsumerWarps * RowsPerWarp;
    constexpr float kReciprocalEps    = 1.0e-38f;
    static_assert(kChunk % RowsPerWave == 0);
    int warp_idx      = mutlass::canonical_warp_idx();
    int lane_idx      = mutlass::canonical_lane_idx();
    int row_in_warp   = lane_idx / ThreadsPerRow;
    int lane_in_row   = lane_idx % ThreadsPerRow;
    int thread_col_lo = lane_in_row * ElementsPerThread;

    constexpr int RowsPerThread = kChunk / RowsPerWave;

    // g_total = exp2(final cumsum) = the last row of the cumsum tile. It does
    // not depend on the row, and col_base only varies with row_in_warp, so load
    // it once per column slot instead of once per (row, slot).
    float4 g_total[VectorsPerThread];
    MUTE_UNROLL
    for (int vec_iter = 0; vec_iter < VectorsPerThread; ++vec_iter) {
      int vec_slot      = (vec_iter + row_in_warp) % VectorsPerThread;
      g_total[vec_iter] = load_qk_vec4(sGCumsum, kChunk - 1, thread_col_lo + vec_slot * VecSize);
    }

    MUTE_UNROLL
    for (int r = 0; r < RowsPerThread; ++r) {
      int row = r * RowsPerWave + warp_idx * RowsPerWarp + row_in_warp;
      // Row layout of the 32-row decayed operand tile, i.e. the K1/K2
      // workspace contract for this architecture.
      //   MP31 (32x32 SQMMA, rows r/r+16 per warp): K 0-7, Q 0-7, K 8-15,
      //     Q 8-15, which leaves warps 0/1 with the KKT rows and warps 2/3
      //     with the P rows.
      //   MP32 (M32N32, C rows [0,16) to warps 0/1, [16,32) to warps 2/3):
      //     Q 0-15 in the top half and K 0-15 in the bottom half, which gives
      //     warps 0/1 the P rows and warps 2/3 the KKT rows.
      int q_row = row;
      int k_row = row + kChunk;
      MUTE_UNROLL
      for (int vec_iter = 0; vec_iter < VectorsPerThread; ++vec_iter) {
        int vec_slot = (vec_iter + row_in_warp) % VectorsPerThread;
        int col_base = thread_col_lo + vec_slot * VecSize;

        float4 q    = load_qk_vec4(sQ, row, col_base);
        float4 k    = load_qk_vec4(sK, row, col_base);
        float4 gate = load_qk_vec4(sGCumsum, row, col_base);

        // 0 * rcp(0 + eps) stays finite for the inactive rows.
        float4 k_inverse  = simd::vmul(k, rcp_native(simd::vadd(gate, simd::splat4(kReciprocalEps))));
        float4 k_restored = simd::vmul(k_inverse, g_total[vec_iter]);

        store_qk_vec4(sDecayed, q_row, col_base, simd::vmul(q, gate));
        store_qk_vec4(sDecayed, k_row, col_base, simd::vmul(k, gate));
        store_qk_vec4(sKInverse, row, col_base, k_inverse);
        store_qk_vec4(sKRestored, row, col_base, k_restored);
      }
    }
  }

  // Consumer warps 0/1: scatter the KKT matrix and solve the NxN inverse;
  // consumer warps 2/3 scatter P.
  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void compute_inverse_and_p(SharedStorage&     shared_storage,
                                                   Params const&      params,
                                                   ProblemSize const& problem_size,
                                                   WorkDesc const&    work_desc,
                                                   int                chunk_idx,
                                                   int                local_tid,
                                                   bool               is_not_full_chunk) {
    auto sDecayed  = make_tensor(make_smem_ptr(shared_storage.smem_decayed.data()), SmemLayoutDecayed{})(_, _, 0);
    auto sKInverse = make_tensor(make_smem_ptr(shared_storage.smem_k_inverse.data()), SmemLayoutKInverse{})(_, _, 0);

    TiledMmaQK tiled_mma;
    auto       thr_mma = tiled_mma.get_thread_slice(local_tid);
    auto       acc     = partition_fragment_C(tiled_mma, make_shape(Int<2 * kChunk>{}, Int<2 * kChunk>{}));
    static_assert(decltype(size(acc))::value == 8);
    clear(acc);

    auto tSsDecayed  = thr_mma.partition_A(sDecayed);
    auto tSsKInverse = thr_mma.partition_B(sKInverse);
    auto tSrDecayed  = thr_mma.make_fragment_A(tSsDecayed);
    auto tSrKInverse = thr_mma.make_fragment_B(tSsKInverse);

    // Only K-inverse rows 0..15 are initialized.  They produce the valid C
    // columns 0..15; columns 16..31 are intentionally computed and discarded.
    gemm(tiled_mma, acc, tSrDecayed, tSrKInverse, acc);
    mute::warpsquad_commit_batch();
    mute::warpsquad_wait<0>();

    int                  warp_idx = mutlass::canonical_warp_idx();
    int                  lane     = local_tid % mutlass::NumThreadsPerWarp;
    CollectiveInverseNxN inverse_nxn;
    auto                 cC  = make_identity_tensor(make_shape(Int<2 * kChunk>{}, Int<2 * kChunk>{}));
    auto                 tCc = thr_mma.partition_C(cC);
    // M32N32 hands C rows [0, 16) to warps 0/1 and [16, 32) to warps 2/3, and the
    // decayed tile is [Q 0-15, K 0-15], so warps 0/1 own the Q rows (the P block)
    // and warps 2/3 own the K rows (the KKT matrix). The two tiles never coexist
    // in one thread's fragment, which is what lets each pair run its own
    // scatter -> rendezvous -> drain with no cross-pair barrier at all:
    //
    //   warps 0/1: scatter P into smem, rendezvous, warp 0 drains it to gmem
    //   warps 2/3: scatter KKT into the matrix, rendezvous, warp 2 solves it
    //
    // Nothing is hoisted above this split. In particular beta belongs to the KKT
    // pair only, so the P pair never even loads it, and each pair computes its
    // own output pointer inside its branch.
    if (warp_idx < 2) {
      int actual_len = work_desc.actual_len(chunk_idx);
      // Q rows: tile row is already the Q index. Scatter the fragment into
      // smem rather than straight to gmem. A 2-byte global store per element
      // leaves every 32-byte P row partially written, so the block costs ~256
      // partial row writes instead of 16 full ones and L2 read-modify-writes
      // each line; the staged form publishes it with one contiguous store per
      // lane.
      auto sP = make_tensor(make_smem_ptr(shared_storage.smem_p_stage.data()), PStagingLayout{});
      MUTLASS_PRAGMA_UNROLL
      for (int base = 0; base < size(acc); base += VecSize) {
        float4  values = *reinterpret_cast<float4 const*>(&acc(base));
        Element packed[VecSize];
        simd::store_float4_to_packed4_rn(packed, values);
        MUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < VecSize; ++j) {
          int row = int(get<0>(tCc(base + j)));
          int col = int(get<1>(tCc(base + j)));
          if (col < kChunk) {
            sP(make_coord(row, col)) = row < actual_len && row >= col ? packed[j] : Element(0.0f);
          }
        }
      }

      PStagedBarrierSync{local_tid}();
      if (warp_idx == 0) {
        int64_t  record                          = Workspace::record(problem_size, work_desc, chunk_idx);
        Element* gP                              = params.workspace.p + record * Workspace::RecordStride;
        auto*    sPFlat                          = shared_storage.smem_p_stage.data();
        using PublishVec                         = mutlass::Array<Element, kPublishPerLane>;
        int idx                                  = lane * kPublishPerLane;
        *reinterpret_cast<PublishVec*>(gP + idx) = *reinterpret_cast<PublishVec const*>(sPFlat + idx);
      }
    } else {
      // K rows live in tile rows [kChunk, 2 * kChunk).
      float beta_lane = load_beta_register(params, problem_size, work_desc, chunk_idx, lane, is_not_full_chunk);
      auto  matrix    = make_tensor(make_smem_ptr(shared_storage.smem_inverse_workspace.matrix),
                                typename CollectiveInverseNxN::SmemLayout{});
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(acc); ++i) {
        int row = int(get<0>(tCc(i))) - kChunk;
        int col = int(get<1>(tCc(i)));
        if (col < kChunk) {
          float beta_row               = __shfl_sync(uint32_t(-1), beta_lane, row, mutlass::NumThreadsPerWarp);
          matrix(make_coord(row, col)) = row == col ? 1.0f : (row > col ? acc(i) * beta_row : 0.0f);
        }
      }

      // The KKT matrix lives in this pair's accumulators, so this pair solves
      // it. M32N32 gives warps 2/3 the K rows, so the solve runs on warp 2:
      // indexing relative to this pair makes warp 2's lanes 0..15 the 16 columns
      // and leaves warp 3 out, exactly as local_tid does for warp 0 on MP31.
      inverse_nxn.solve_columns(shared_storage.smem_inverse_workspace,
                                beta_lane,
                                InverseSmemBarrierSync{local_tid},
                                local_tid - 2 * mutlass::NumThreadsPerWarp);
      InverseSmemBarrierSync{local_tid}();
      // The output address is only needed here. Computing it after the solve
      // keeps its 64-bit form out of the solve's live range, which is what
      // lands this CTA at 62 registers - eight resident CTAs instead of seven.
      int64_t  record      = Workspace::record(problem_size, work_desc, chunk_idx);
      Element* gInverseOut = params.workspace.inverse + record * Workspace::RecordStride;
      inverse_nxn.publish_solution(
          shared_storage.smem_inverse_workspace, gInverseOut, local_tid - 2 * mutlass::NumThreadsPerWarp);
    }
  }

  // Consumer warps: fused pass + gemm + epilogue for every chunk.
  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void consumer_loop(SharedStorage&     shared_storage,
                                           Params const&      params,
                                           ProblemSize const& problem_size,
                                           WorkDesc const&    work_desc,
                                           Pipeline&          pipeline,
                                           int                thread_idx,
                                           int                chunk_idx) {
    int lane = mutlass::canonical_lane_idx();

    auto sGCumsum   = make_tensor(make_smem_ptr(shared_storage.smem_g_cumsum.data()), CumsumLayout{});
    auto sNormQ     = make_tensor(make_smem_ptr(shared_storage.smem_norm_q.data()), NormLayout{});
    auto sNormK     = make_tensor(make_smem_ptr(shared_storage.smem_norm_k.data()), NormLayout{});
    auto sDecayed   = make_tensor(make_smem_ptr(shared_storage.smem_decayed.data()), SmemLayoutDecayed{})(_, _, 0);
    auto sKInverse  = make_tensor(make_smem_ptr(shared_storage.smem_k_inverse.data()), SmemLayoutKInverse{})(_, _, 0);
    auto sKRestored = make_tensor(make_smem_ptr(shared_storage.smem_k_restored.data()), SmemLayoutKRestoredTile{});

    {
      int  actual_len        = work_desc.actual_len(chunk_idx);
      bool is_not_full_chunk = actual_len < kChunk;

      pipeline.norm.consumer_wait(pipeline.norm_read);
      pipeline.decayed.producer_acquire(pipeline.decayed_write);
      scale_normalized_qk(sNormK, sNormQ, sGCumsum, sDecayed, sKInverse, sKRestored, thread_idx);
      __syncwarp();
      if (lane == 0) {
        pipeline.norm.consumer_release(pipeline.norm_read);
        pipeline.decayed.producer_commit(pipeline.decayed_write);
      }
      __syncwarp();
      ++pipeline.norm_read;
      ++pipeline.decayed_write;
      compute_inverse_and_p(shared_storage, params, problem_size, work_desc, chunk_idx, thread_idx, is_not_full_chunk);
    }
  }

  // Producer warp: normalize Q for every chunk.
  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void producer_normalize_q_loop(SharedStorage&     shared_storage,
                                                       Params const&      params,
                                                       ProblemSize const& problem_size,
                                                       WorkDesc const&    work_desc,
                                                       Pipeline&          pipeline,
                                                       int                thread_idx,
                                                       int                chunk_idx) {
    (void)problem_size;
    (void)shared_storage;
    (void)params;
    (void)thread_idx;
    int lane = mutlass::canonical_lane_idx();
    {
      pipeline.norm.producer_acquire(pipeline.norm_write);
      GmemTiledCopyQ tiled_copy;
      auto           thr_copy          = tiled_copy.get_thread_slice(lane);
      int            actual_len        = work_desc.actual_len(chunk_idx);
      bool           is_not_full_chunk = actual_len < kChunk;
      auto           outer             = get<2>(params.stride_q);
      int64_t        base              = int64_t(work_desc.tme_token(0)) * int64_t(get<0>(params.stride_q)) +
                     int64_t(work_desc.qk_head_idx) * int64_t(get<0>(outer)) +
                     int64_t(work_desc.tme_batch()) * int64_t(get<1>(outer));
      auto gQ = make_tensor(
          make_gmem_ptr(params.ptr_q + base),
          make_layout(make_shape(Int<kChunk>{}, Int<kHeadDim>{}), make_stride(get<0>(params.stride_q), _1{})));
      auto sQraw = make_tensor(make_smem_ptr(shared_storage.smem_norm_q.data()), NormLayout{});
      copy(tiled_copy.with(params.desc_q), thr_copy.partition_S(gQ), thr_copy.partition_D(sQraw));
      mute::ldgsts_wait();
      normalize_qk(shared_storage.smem_norm_q.data(), params.scale, actual_len, is_not_full_chunk);

      __syncwarp();
      if (lane == 0) {
        pipeline.norm.producer_commit(pipeline.norm_write);
      }
      __syncwarp();
      ++pipeline.norm_write;
    }
  }

  // Producer warp: normalize K for every chunk.
  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void producer_normalize_k_loop(SharedStorage&     shared_storage,
                                                       Params const&      params,
                                                       ProblemSize const& problem_size,
                                                       WorkDesc const&    work_desc,
                                                       Pipeline&          pipeline,
                                                       int                thread_idx,
                                                       int                chunk_idx) {
    (void)problem_size;
    (void)shared_storage;
    (void)params;
    (void)thread_idx;
    int lane = mutlass::canonical_lane_idx();
    {
      pipeline.norm.producer_acquire(pipeline.norm_write);
      GmemTiledCopyK tiled_copy;
      auto           thr_copy          = tiled_copy.get_thread_slice(lane);
      int            actual_len        = work_desc.actual_len(chunk_idx);
      bool           is_not_full_chunk = actual_len < kChunk;
      auto           outer             = get<2>(params.stride_k);
      int64_t        base              = int64_t(work_desc.tme_token(0)) * int64_t(get<0>(params.stride_k)) +
                     int64_t(work_desc.qk_head_idx) * int64_t(get<0>(outer)) +
                     int64_t(work_desc.tme_batch()) * int64_t(get<1>(outer));
      auto gK = make_tensor(
          make_gmem_ptr(params.ptr_k + base),
          make_layout(make_shape(Int<kChunk>{}, Int<kHeadDim>{}), make_stride(get<0>(params.stride_k), _1{})));
      auto sKraw = make_tensor(make_smem_ptr(shared_storage.smem_norm_k.data()), NormLayout{});
      copy(tiled_copy.with(params.desc_k), thr_copy.partition_S(gK), thr_copy.partition_D(sKraw));
      mute::ldgsts_wait();
      normalize_qk(shared_storage.smem_norm_k.data(), 1.0f, actual_len, is_not_full_chunk);

      __syncwarp();
      if (lane == 0) {
        pipeline.norm.producer_commit(pipeline.norm_write);
      }
      __syncwarp();
      ++pipeline.norm_write;
    }
  }

  // Producer warp: gate cumsum for every chunk.
  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void producer_gate_loop(SharedStorage&     shared_storage,
                                                Params const&      params,
                                                ProblemSize const& problem_size,
                                                WorkDesc const&    work_desc,
                                                Pipeline&          pipeline,
                                                int                thread_idx,
                                                int                chunk_idx) {
    (void)shared_storage;
    (void)params;
    (void)problem_size;
    (void)thread_idx;
    int lane = mutlass::canonical_lane_idx();
    {
      pipeline.norm.producer_acquire(pipeline.norm_write);
      GmemTiledCopyG tiled_copy;
      auto           thr_copy = tiled_copy.get_thread_slice(lane);
      auto           outer    = get<2>(params.stride_g);
      int64_t        base     = int64_t(work_desc.tme_token(chunk_idx)) * int64_t(get<0>(params.stride_g)) +
                     int64_t(work_desc.head_idx) * int64_t(get<0>(outer)) +
                     int64_t(work_desc.tme_batch()) * int64_t(get<1>(outer));
      auto gG = make_tensor(
          make_gmem_ptr(params.ptr_g + base),
          make_layout(make_shape(Int<kChunk>{}, Int<kHeadDim>{}), make_stride(get<0>(params.stride_g), _1{})));
      auto sGraw = make_tensor(make_smem_ptr(shared_storage.smem_g_cumsum.data()), CumsumLayout{});
      copy(tiled_copy.with(params.desc_g), thr_copy.partition_S(gG), thr_copy.partition_D(sGraw));
      mute::ldgsts_wait();
      compute_g_cumsum(shared_storage, params, problem_size, work_desc, chunk_idx);

      __syncwarp();
      if (lane == 0) {
        pipeline.norm.producer_commit(pipeline.norm_write);
      }
      __syncwarp();
      ++pipeline.norm_write;
    }
  }

  // Producer warp: TMA store of the decayed tile for every chunk.
  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void producer_store_loop(SharedStorage&     shared_storage,
                                                 Params const&      params,
                                                 ProblemSize const& problem_size,
                                                 WorkDesc const&    work_desc,
                                                 Pipeline&          pipeline,
                                                 int                thread_idx,
                                                 int                chunk_idx) {
    (void)shared_storage;
    (void)params;
    (void)problem_size;
    (void)thread_idx;
    int lane = mutlass::canonical_lane_idx();
    {
      pipeline.decayed.consumer_wait(pipeline.decayed_read);
      int32_t records = Components::workspace_records(problem_size);
      store_decayed(shared_storage, params, problem_size, work_desc, chunk_idx);

      __syncwarp();
      if (lane == 0) {
        pipeline.decayed.consumer_release(pipeline.decayed_read);
      }
      __syncwarp();
      ++pipeline.decayed_read;
    }
  }
};

}  // namespace prepare

}  // namespace mate::flat::kda
