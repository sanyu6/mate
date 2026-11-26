#pragma once

#include "mate/attention/msa/collective/msa_fwd_collective_tme_warpspecialized.hpp"
#include "mate/attention/msa/collective/msa_fwd_epilogue.hpp"
#include "mate/attention/msa/collective/msa_fwd_pair_epilogue.hpp"
#include "mate/attention/msa/kernel/msa_fwd_kernel_tme_warpspecialized.hpp"
#include "mate/attention/msa/kernel/msa_fwd_pair_tile_scheduler.hpp"
#include "mate/attention/msa/kernel/msa_fwd_pair_union_kernel_tme_warpspecialized.hpp"
#include "mate/attention/msa/kernel/msa_fwd_tile_scheduler.hpp"
#include "mate/attention/msa/msa_options.hpp"

namespace mate::attention::msa {

// MSA forward implementation:
//   Element       = FP8 E4M3, FP16, or BF16 for Q/K/V/P
//   ElementOutput = FP8 E4M3, FP16, or BF16 for O
//   accumulator   = FP32
//   CTA           = one query x one KV head, logical ratio 8 or 16
//   MP31 = physical M16 SS QK/PV; MP32 = physical M32 SS QK and RS PV
//   unused physical head rows use TME OOB fill
//   sparse blocks = q2k[total_q, Hkv, 16], block/page size 128
//   output        = direct O + LSE (no split, partial output, or combine)
template <class Element_, class ElementOutput_, int HeadRatio_, class... Options_>
struct MsaFwdBuilder {
  using Element                  = Element_;
  using ElementOutput            = ElementOutput_;
  static constexpr int HeadRatio = HeadRatio_;
  static constexpr int QStages   = 1;
  static constexpr int KStages   = 1;
  static constexpr int VStages   = 1;
  static_assert(HeadRatio == 8 || HeadRatio == 16, "MSA forward supports head ratios 8 and 16.");

  using CollectiveMainloop =
      collective::MsaFwdCollectiveTmeWarpSpecialized<Element, QStages, KStages, VStages, Options_...>;
  using CollectiveEpilogue = collective::MsaFwdEpilogue<ElementOutput>;
  using TileScheduler      = MsaFwdTileScheduler;
  using Kernel             = MsaFwdKernelTmeWarpSpecialized<CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
};

// Ratio-8 union path: two queries fill M16 on MP31, four fill M32 on MP32.
// Their top-k lists share K/V loads through per-query membership masks,
// reusing one QK/PV sequence for each union entry.
template <class Element_, class ElementOutput_, class... Options_>
struct MsaFwdPairUnionBuilder {
  using Element                = Element_;
  using ElementOutput          = ElementOutput_;
  static constexpr int QStages = 1;
  static constexpr int KStages = 1;
  static constexpr int VStages = 1;

  using CollectiveMainloop = collective::MsaFwdCollectiveTmeWarpSpecialized<Element,
                                                                            QStages,
                                                                            KStages,
                                                                            VStages,
                                                                            Options_...,
                                                                            PackQueryPair<std::true_type>>;
  using CollectiveEpilogue = collective::MsaFwdPairEpilogue<ElementOutput>;
  // Re-decode MP32 metadata to shorten its lifetime across the compute body.
  using TileScheduler = std::conditional_t<CollectiveMainloop::IsMP32,
                                           MsaFwdQueryGroupTileScheduler<CollectiveMainloop::QueriesPerTile, false>,
                                           Mp31MsaFwdPairTileScheduler>;
  using Kernel        = MsaFwdPairUnionKernelTmeWarpSpecialized<CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
};

}  // namespace mate::attention::msa
