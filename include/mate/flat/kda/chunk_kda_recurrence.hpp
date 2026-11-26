#pragma once

#include <type_traits>

#include "mate/flat/kda/chunk_kda_components.hpp"
#include "mate/flat/kda/chunk_kda_mp31_policy.hpp"
#include "mate/flat/kda/chunk_kda_mp32_policy.hpp"
#include "mate/flat/kda/collective/mp31_chunk_kda_recurrence_collective_tme_warpspecialized.hpp"
#include "mate/flat/kda/collective/mp32_chunk_kda_recurrence_collective_tme_warpspecialized.hpp"
#include "mate/flat/kda/kda_arch_traits.hpp"
#include "mate/flat/kda/kernel/chunk_kda_recurrence_kernel.hpp"
#include "mate/flat/kda/kernel/chunk_kda_tile_scheduler.hpp"

namespace mate::flat::kda {

// K2 (recurrence) builder.  The two architectures need architecture-specific
// collectives -- MP32 widens the Q/P MMA to 2*kChunk rows, loads the
// inverse/P staging tile with plain loads and lays the decayed tile out as
// Q[0:kChunk], K[0:kChunk] -- so ArchTag selects between two collectives
// while the problem/workspace contract stays shared. MP32 assigns each CTA one
// recurrence chain directly; MP31 retains its persistent scheduler.
template <class ArchTag_,
          class Element_,
          class StateElement_,
          class CuSeqlensElement_,
          class LogicalShape_,
          class TileShape_,
          class StrideV_,
          class StrideO_,
          class... Options_>
struct ChunkKdaRecurrenceBuilder {
  using ArchTag                = ArchTag_;
  static constexpr bool IsMp32 = arch_traits<ArchTag>::is_mp32;
  using SchedulePolicy         = std::conditional_t<IsMp32, ChunkKdaMp32SchedulePolicy, ChunkKdaMp31SchedulePolicy>;
  using Components = ChunkKdaPrefillComponents<Element_, CuSeqlensElement_, LogicalShape_, SchedulePolicy, Options_...>;
  using CollectiveRecurrence = std::conditional_t<
      IsMp32,
      recurrence::
          Mp32ChunkKdaRecurrenceCollectiveTmeWarpSpecialized<Components, TileShape_, StateElement_, StrideV_, StrideO_>,
      recurrence::Mp31ChunkKdaRecurrenceCollectiveTmeWarpSpecialized<Components,
                                                                     TileShape_,
                                                                     StateElement_,
                                                                     StrideV_,
                                                                     StrideO_>>;
  using TileScheduler = ChunkKdaRecurrenceScheduler<typename Components::LogicalShape,
                                                    typename Components::CuSeqlensElement,
                                                    typename Components::SchedulePolicy,
                                                    Options_...>;
  using Kernel        = ChunkKdaRecurrenceKernel<CollectiveRecurrence, TileScheduler>;
};

}  // namespace mate::flat::kda
