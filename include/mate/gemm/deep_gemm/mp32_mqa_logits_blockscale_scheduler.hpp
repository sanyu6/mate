#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/tensor.hpp>

#include "mp32_mqa_logits_blockscale_metadata.hpp"

namespace mate::deep_gemm {

using namespace mute;

template <uint32_t kTileKV_, uint32_t kTileQ_, uint32_t kKVTokenAlignment_ = 4>
struct MP32MQALogitsLegacyScheduler {
  static constexpr uint32_t kTileKV           = kTileKV_;
  static constexpr uint32_t kTileQ            = kTileQ_;
  static constexpr uint32_t kKVTokenAlignment = kKVTokenAlignment_;

  static_assert(kTileKV > 0, "tileKV must be positive");
  static_assert(kTileQ > 0, "tileQ must be positive");
  static_assert(kKVTokenAlignment > 0, "KV token alignment must be positive");

  struct Task {
    uint32_t tile_q_idx;
    uint32_t kv_token_base;
    uint32_t num_tile_kv;
  };

  uint32_t       current_tile_q_idx;
  uint32_t const worker_stride;
  uint32_t const num_tile_q;
  uint32_t const total_seqlen_q;
  uint32_t const seqlen_kv;
  int32_t const* cu_seqlen_k_start;
  int32_t const* cu_seqlen_k_end;
  uint32_t       seqlen_k_start[kTileQ];
  uint32_t       seqlen_k_end[kTileQ];

  MUTLASS_DEVICE MP32MQALogitsLegacyScheduler(uint32_t        worker_idx,
                                              uint32_t        num_workers,
                                              uint32_t        num_tile_q,
                                              uint32_t        total_seqlen_q,
                                              uint32_t        seqlen_kv,
                                              int32_t const*  cu_seqlen_k_start,
                                              int32_t const*  cu_seqlen_k_end,
                                              uint32_t const* schedule_meta = nullptr)
      : current_tile_q_idx(num_workers == 0 || total_seqlen_q == 0 ? num_tile_q : worker_idx),
        worker_stride(num_workers),
        num_tile_q(num_tile_q),
        total_seqlen_q(total_seqlen_q),
        seqlen_kv(seqlen_kv),
        cu_seqlen_k_start(cu_seqlen_k_start),
        cu_seqlen_k_end(cu_seqlen_k_end) {
  }

  MUTLASS_DEVICE bool get_next_task(Task& task) {
    if (current_tile_q_idx >= num_tile_q) {
      return false;
    }

    uint32_t const tile_q_idx = current_tile_q_idx;
    current_tile_q_idx += worker_stride;

    uint32_t range_start = seqlen_kv;
    uint32_t range_end   = 0;

    MUTE_UNROLL
    for (uint32_t i = 0; i < kTileQ; ++i) {
      uint32_t const q_row = min(tile_q_idx * kTileQ + i, total_seqlen_q - 1);

      int32_t ks = __ldg(cu_seqlen_k_start + q_row);
      int32_t ke = __ldg(cu_seqlen_k_end + q_row);
      ks         = max(int32_t(0), min(ks, int32_t(seqlen_kv)));
      ke         = max(int32_t(0), min(ke, int32_t(seqlen_kv)));

      seqlen_k_start[i] = uint32_t(ks);
      seqlen_k_end[i]   = uint32_t(ke);

      range_start = min(range_start, uint32_t(ks));
      range_end   = max(range_end, uint32_t(ke));
    }

    uint32_t const kv_token_base = range_start / kKVTokenAlignment * kKVTokenAlignment;
    task                         = Task{
                                .tile_q_idx    = tile_q_idx,
                                .kv_token_base = kv_token_base,
                                .num_tile_kv   = ceil_div(range_end - kv_token_base, kTileKV),
    };
    return true;
  }
};

template <uint32_t kTileKV_, uint32_t kTileQ_, uint32_t kNumMP_, uint32_t kKVTokenAlignment_ = 4>
struct MP32MQALogitsStaticScheduler {
  static constexpr uint32_t kTileKV           = kTileKV_;
  static constexpr uint32_t kTileQ            = kTileQ_;
  static constexpr uint32_t kNumMP            = kNumMP_;
  static constexpr uint32_t kKVTokenAlignment = kKVTokenAlignment_;

  static_assert(kTileKV > 0, "tileKV must be positive");
  static_assert(kTileQ > 0, "tileQ must be positive");
  static_assert(kNumMP > 0, "numMP must be positive");
  static_assert(kKVTokenAlignment > 0, "KV token alignment must be positive");

  struct Task {
    uint32_t tile_q_idx;
    uint32_t kv_token_base;
    uint32_t num_tile_kv;
  };

  uint32_t       current_tile_q_idx;
  uint32_t       current_split_offset;
  uint32_t       remaining_splits;
  uint2 const*   spans;
  uint32_t const num_tile_q;
  uint32_t const total_seqlen_q;
  uint32_t const seqlen_kv;
  int32_t const* cu_seqlen_k_start;
  int32_t const* cu_seqlen_k_end;
  uint32_t       seqlen_k_start[kTileQ];
  uint32_t       seqlen_k_end[kTileQ];

  MUTLASS_DEVICE MP32MQALogitsStaticScheduler(uint32_t        mp_idx,
                                              uint32_t        num_workers,
                                              uint32_t        num_tile_q,
                                              uint32_t        total_seqlen_q,
                                              uint32_t        seqlen_kv,
                                              int32_t const*  cu_seqlen_k_start,
                                              int32_t const*  cu_seqlen_k_end,
                                              uint32_t const* schedule_meta)
      : num_tile_q(num_tile_q),
        total_seqlen_q(total_seqlen_q),
        seqlen_kv(seqlen_kv),
        cu_seqlen_k_start(cu_seqlen_k_start),
        cu_seqlen_k_end(cu_seqlen_k_end) {
    current_tile_q_idx   = reinterpret_cast<uint2 const*>(schedule_meta)[mp_idx].x;
    current_split_offset = reinterpret_cast<uint2 const*>(schedule_meta)[mp_idx].y;
    remaining_splits     = schedule_meta[2 * kNumMP + mp_idx];
    spans                = reinterpret_cast<uint2 const*>(schedule_meta + mqa_logits_meta_span_word_offset(kNumMP));
  }

  MUTLASS_DEVICE void load_seq_bounds(uint32_t tile_q_idx) {
    MUTE_UNROLL
    for (uint32_t i = 0; i < kTileQ; ++i) {
      uint32_t const q_row = min(tile_q_idx * kTileQ + i, total_seqlen_q - 1);

      int32_t ks = __ldg(cu_seqlen_k_start + q_row);
      int32_t ke = __ldg(cu_seqlen_k_end + q_row);
      ks         = max(int32_t(0), min(ks, int32_t(seqlen_kv)));
      ke         = max(int32_t(0), min(ke, int32_t(seqlen_kv)));

      seqlen_k_start[i] = uint32_t(ks);
      seqlen_k_end[i]   = uint32_t(ke);
    }
  }

  MUTLASS_DEVICE bool get_next_task(Task& task) {
    while (remaining_splits > 0 && current_tile_q_idx < num_tile_q) {
      uint2 const    span         = spans[current_tile_q_idx];
      uint32_t const split_offset = current_split_offset;
      current_split_offset        = 0;

      if (split_offset < span.y) {
        load_seq_bounds(current_tile_q_idx);

        task = Task{
            .tile_q_idx    = current_tile_q_idx,
            .kv_token_base = span.x + split_offset * kTileKV,
            .num_tile_kv   = min(span.y - split_offset, remaining_splits),
        };
        remaining_splits -= task.num_tile_kv;
        ++current_tile_q_idx;
        return true;
      }
      ++current_tile_q_idx;
    }
    return false;
  }
};

}  // namespace mate::deep_gemm
