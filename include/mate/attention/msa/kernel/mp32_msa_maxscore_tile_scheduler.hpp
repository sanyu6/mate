#pragma once

#include "mate/attention/msa/kernel/msa_maxscore_tile_scheduler.hpp"

namespace mate::attention::msa {

template <class TileShape, int HeadRatio, bool ParallelKTiles, bool IsVarlen, bool HasMetadata, int CtasPerMp>
struct Mp32MsaMaxScoreTileScheduler
    : MsaMaxScoreTileScheduler<TileShape, HeadRatio, ParallelKTiles, IsVarlen, HasMetadata, CtasPerMp> {
  using Base = MsaMaxScoreTileScheduler<TileShape, HeadRatio, ParallelKTiles, IsVarlen, HasMetadata, CtasPerMp>;
  using typename Base::Params;
  using typename Base::ScheduleInfo;
  using typename Base::WorkTileInfo;

  static MUTLASS_DEVICE ScheduleInfo get_schedule_info(Params const& params) {
    if constexpr (!HasMetadata) {
      return Base::get_schedule_info(params);
    } else {
      ScheduleInfo info{params.total_works, params.total_works, params.k_partitions, 0};
      int          mp_slot = Base::schedule_mp_slot(params);
      if (mp_slot >= params.schedule_num_mps) return info;
      int row              = 0;
      int group            = Base::schedule_partition_group(params, mp_slot, row);
      int group_partitions = min(Base::CtasPerMp, params.k_partitions - group * Base::CtasPerMp);
      if (group_partitions <= 0) return info;
      int q_begin = params.ptr_schedule_meta[2 * mp_slot];
      int q_count = params.ptr_schedule_meta[2 * mp_slot + 1];
      int slot    = Base::schedule_cta_slot();
      int part    = slot % group_partitions;
      int q_lane  = slot / group_partitions;
      // Each partition's slots interleave the Q range, including a partial group.
      int q_lanes = (Base::CtasPerMp + group_partitions - 1 - part) / group_partitions;
      if (q_lane >= q_count) return info;
      int partition    = group * Base::CtasPerMp + part;
      info.work_begin  = (q_begin + q_lane) * params.k_partitions + partition;
      info.work_end    = (q_begin + q_count) * params.k_partitions;
      info.work_stride = q_lanes * params.k_partitions;
      info.active      = 1;
      return info;
    }
  }

  MUTLASS_DEVICE WorkTileInfo get_next_work(Params const&       params,
                                            WorkTileInfo const& work,
                                            ScheduleInfo const& info) const {
    if constexpr (!HasMetadata) {
      return Base::get_next_work(params, work, info);
    } else {
      int next = work.work_idx + info.work_stride;
      if (!info.active || next >= info.work_end || next >= params.total_works) {
        WorkTileInfo invalid{};
        invalid.work_idx = params.total_works;
        return invalid;
      }
      auto next_work = Base::make_q_work_tile(params, next);
      if (next_work.batch_idx == work.batch_idx) {
        Base::copy_k_meta(work, next_work);
      }
      return next_work;
    }
  }
};

}  // namespace mate::attention::msa
