#pragma once

#include <type_traits>

#include "mate/flat/kda/chunk_kda_components.hpp"
#include "mate/flat/kda/chunk_kda_mp31_policy.hpp"
#include "mate/flat/kda/chunk_kda_mp32_policy.hpp"
#include "mate/flat/kda/collective/mp31_chunk_kda_prepare_collective_tme_warpspecialized.hpp"
#include "mate/flat/kda/collective/mp32_chunk_kda_prepare_collective_tme_warpspecialized.hpp"
#include "mate/flat/kda/kda_arch_traits.hpp"
#include "mate/flat/kda/kernel/chunk_kda_metadata_kernel.hpp"
#include "mate/flat/kda/kernel/chunk_kda_prepare_kernel.hpp"
#include "mate/flat/kda/kernel/chunk_kda_tile_scheduler.hpp"

namespace mate::flat::kda {

// K1 (prepare) builder.  The two architectures need architecture-specific
// collectives -- MP31 runs one 128-thread squad per CTA against a 32x32 SQMMA
// with the K0-7 / Q0-7 / K8-15 / Q8-15 decayed row layout and TME workspace
// stores, while MP32 lays the decayed tile out as Q[0:kChunk], K[0:kChunk],
// drives an M32N32 SQMMA and drains the workspace with LSU stores -- so ArchTag
// selects between two collectives while the problem/workspace/scheduler
// contract stays shared.
template <class ArchTag_,
          class Element_,
          class CuSeqlensElement_,
          class LogicalShape_,
          class TileShape_,
          class StrideQ_,
          class StrideK_,
          class StrideG_,
          class... Options_>
struct ChunkKdaPrepareBuilder {
  using ArchTag                = ArchTag_;
  static constexpr bool IsMp32 = arch_traits<ArchTag>::is_mp32;
  using SchedulePolicy         = std::conditional_t<IsMp32, ChunkKdaMp32SchedulePolicy, ChunkKdaMp31SchedulePolicy>;
  using Components = ChunkKdaPrefillComponents<Element_, CuSeqlensElement_, LogicalShape_, SchedulePolicy, Options_...>;
  using CollectivePrepare = std::conditional_t<
      IsMp32,
      prepare::Mp32ChunkKdaPrepareCollectiveTmeWarpSpecialized<Components, TileShape_, StrideQ_, StrideK_, StrideG_>,
      prepare::Mp31ChunkKdaPrepareCollectiveTmeWarpSpecialized<Components, TileShape_, StrideQ_, StrideK_, StrideG_>>;
  using TileScheduler = typename Components::PrepareTileScheduler;
  using Kernel        = ChunkKdaPrepareKernel<CollectivePrepare, TileScheduler>;
};

}  // namespace mate::flat::kda
