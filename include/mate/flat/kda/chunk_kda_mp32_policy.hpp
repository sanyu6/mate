#pragma once

#include "mate/flat/kda/chunk_kda_problem.hpp"

namespace mate::flat::kda {

// One CTA owns a 64-column value slice and has one producer, one state, and one
// output squad. Four CTAs per MP is only a launch-bounds budget: the recurrence
// grid covers each (sequence, head, value-slice) work exactly once.
struct ChunkKdaMp32SchedulePolicy : ChunkKdaPrefillSchedulePolicy<8, 4, 64, 512, 192 * 1024> {};

}  // namespace mate::flat::kda
