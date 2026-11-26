#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/int_tuple.hpp>

namespace mate::deep_gemm {

template <uint32_t kNextN, uint32_t kTileQ, uint32_t kTileKV, uint32_t kPageSize, bool kIsContextLens2D>
struct MP32PagedMQALogitsRequestInfo {
  uint32_t request_idx;
  uint32_t num_tile_kv;
  uint32_t num_page_kv;

  static_assert(kNextN > 0 && kTileQ > 0 && kTileKV > 0 && kPageSize > 0,
                "paged MQA logits requires positive next_n and tile sizes");
  static constexpr uint32_t kNumTileQ = mute::ceil_div(kNextN, kTileQ);

  MUTLASS_DEVICE static MP32PagedMQALogitsRequestInfo from_request_idx(uint32_t        request_idx,
                                                                       uint32_t const* context_lens) {
    uint32_t const context_len_offset = kIsContextLens2D ? request_idx * kNextN + kNextN - 1 : request_idx;
    uint32_t const context_len        = context_lens[context_len_offset];

    MP32PagedMQALogitsRequestInfo info{};
    info.request_idx = request_idx;
    info.num_tile_kv = mute::ceil_div(context_len, kTileKV);
    info.num_page_kv = mute::ceil_div(context_len, kPageSize);
    return info;
  }

  MUTLASS_DEVICE mute::tuple<uint32_t, uint32_t> get_tile_q_span(uint32_t tile_q_idx) const {
    constexpr uint32_t q_tokens_per_tile     = kNextN / kNumTileQ;
    constexpr uint32_t q_tokens_per_tile_rem = kNextN % kNumTileQ;

    uint32_t const q_offset_in_request =
        tile_q_idx * q_tokens_per_tile + (tile_q_idx < q_tokens_per_tile_rem ? tile_q_idx : q_tokens_per_tile_rem);
    uint32_t const num_q_in_tile = q_tokens_per_tile + (tile_q_idx < q_tokens_per_tile_rem ? 1u : 0u);

    return mute::make_tuple(q_offset_in_request, num_q_in_tile);
  }
};

template <uint32_t kNextN, uint32_t kTileQ, uint32_t kTileKV, uint32_t kPageSize, bool kIsContextLens2D>
struct MP32PagedMQALogitsScheduler {
  using RequestInfo = MP32PagedMQALogitsRequestInfo<kNextN, kTileQ, kTileKV, kPageSize, kIsContextLens2D>;
  static constexpr uint32_t kTilesPerChunk = 16;

  struct Task {
    uint32_t token_q_base;
    uint32_t num_token_q;
    uint32_t tile_kv_base;
    uint32_t num_tile_kv;
    uint32_t request_id;
    uint32_t num_page_kv;
  };

  uint32_t const* context_lens;
  uint32_t        num_requests;

  RequestInfo cur_request_info{};

  uint32_t cur_tile_kv_idx_in_request;
  uint32_t end_request_idx;
  uint32_t end_tile_kv_idx_in_request;

  uint32_t cur_tile_q_idx_in_request;

  bool done;

  MUTLASS_DEVICE MP32PagedMQALogitsScheduler(uint32_t        mp_idx,
                                             uint32_t const* context_lens_,
                                             uint32_t const* metadata,
                                             uint32_t        num_requests_) {
    context_lens = context_lens_;
    num_requests = num_requests_;

    uint2 const start = __ldg(reinterpret_cast<uint2 const*>(metadata) + mp_idx);
    uint2 const end   = __ldg(reinterpret_cast<uint2 const*>(metadata) + mp_idx + 1);

    cur_tile_kv_idx_in_request = start.y;
    cur_tile_q_idx_in_request  = 0;
    end_request_idx            = end.x;
    end_tile_kv_idx_in_request = end.y;

    done = start.x >= num_requests || (start.x == end_request_idx && start.y >= end_tile_kv_idx_in_request);

    if (!done) {
      cur_request_info = RequestInfo::from_request_idx(start.x, context_lens);
    }
  }

  MUTLASS_DEVICE bool get_next_task(Task& task) {
    while (!done) {
      uint32_t const tile_kv_upper =
          cur_request_info.request_idx == end_request_idx ? end_tile_kv_idx_in_request : cur_request_info.num_tile_kv;
      // Skip requests with no remaining KV tiles, including zero-length requests.
      if (cur_tile_kv_idx_in_request >= tile_kv_upper) {
        if (cur_request_info.request_idx == end_request_idx) {
          done = true;
          break;
        }

        uint32_t const next_request_idx = cur_request_info.request_idx + 1;
        cur_tile_kv_idx_in_request      = 0;
        cur_tile_q_idx_in_request       = 0;
        if (next_request_idx >= num_requests ||
            (next_request_idx == end_request_idx && end_tile_kv_idx_in_request == 0)) {
          done = true;
          break;
        }
        cur_request_info = RequestInfo::from_request_idx(next_request_idx, context_lens);
        continue;
      }

      auto [q_offset_in_request, num_q_in_tile] = cur_request_info.get_tile_q_span(cur_tile_q_idx_in_request);

      uint32_t const rem_tile_kv_in_request = tile_kv_upper - cur_tile_kv_idx_in_request;

      task.token_q_base = cur_request_info.request_idx * kNextN + q_offset_in_request;
      task.num_token_q  = num_q_in_tile;
      task.tile_kv_base = cur_tile_kv_idx_in_request;
      task.num_tile_kv  = RequestInfo::kNumTileQ == 1
                              ? rem_tile_kv_in_request
                              : (rem_tile_kv_in_request < kTilesPerChunk ? rem_tile_kv_in_request : kTilesPerChunk);
      task.request_id   = cur_request_info.request_idx;
      task.num_page_kv  = cur_request_info.num_page_kv;

      ++cur_tile_q_idx_in_request;
      if (cur_tile_q_idx_in_request == RequestInfo::kNumTileQ) {
        cur_tile_q_idx_in_request = 0;
        cur_tile_kv_idx_in_request += task.num_tile_kv;
      }
      return true;
    }
    return false;
  }
};

template <uint32_t kTileQ, uint32_t kTileKV, uint32_t kPageSize>
struct MP32VarlenPagedMQALogitsRequestInfo {
  uint32_t token_q_start;
  uint32_t num_token_q;
  uint32_t num_tile_q;
  uint32_t num_tile_kv;
  uint32_t num_page_kv;

  MUTLASS_DEVICE static MP32VarlenPagedMQALogitsRequestInfo from_token_q(uint32_t        token_q_idx,
                                                                         uint32_t        num_q_tokens_total,
                                                                         uint32_t const* context_lens,
                                                                         uint32_t const* indices) {
    uint32_t const request_id  = indices[token_q_idx];
    uint32_t       token_q_end = token_q_idx;
    while (token_q_end + 1 < num_q_tokens_total && indices[token_q_end + 1] == request_id) {
      ++token_q_end;
    }

    uint32_t const context_len = context_lens[token_q_end];

    MP32VarlenPagedMQALogitsRequestInfo info{};
    info.token_q_start = token_q_idx;
    info.num_token_q   = token_q_end - token_q_idx + 1;
    info.num_tile_q    = mute::ceil_div(info.num_token_q, kTileQ);
    info.num_tile_kv   = mute::ceil_div(context_len, kTileKV);
    info.num_page_kv   = mute::ceil_div(context_len, kPageSize);
    return info;
  }

  MUTLASS_DEVICE mute::tuple<uint32_t, uint32_t> get_tile_q_span(uint32_t tile_q_idx) const {
    uint32_t const base     = num_token_q / num_tile_q;
    uint32_t const rem      = num_token_q % num_tile_q;
    uint32_t const q_offset = tile_q_idx * base + (tile_q_idx < rem ? tile_q_idx : rem);
    uint32_t const num_q    = base + (tile_q_idx < rem ? 1u : 0u);
    return mute::make_tuple(q_offset, num_q);
  }
};

template <uint32_t kTileQ, uint32_t kTileKV, uint32_t kPageSize>
struct MP32VarlenPagedMQALogitsScheduler {
  using RequestInfo = MP32VarlenPagedMQALogitsRequestInfo<kTileQ, kTileKV, kPageSize>;

  static constexpr uint32_t kTilesPerChunk = 16;

  struct Task {
    uint32_t token_q_base;
    uint32_t num_token_q;
    uint32_t tile_kv_base;
    uint32_t num_tile_kv;
    uint32_t request_id;
    uint32_t num_page_kv;
  };

  uint32_t const* context_lens;
  uint32_t const* indices;
  uint32_t        num_q_tokens_total;

  RequestInfo cur_request_info{};

  uint32_t cur_tile_kv_idx_in_request;
  uint32_t cur_tile_q_idx_in_request;
  uint32_t end_token_q_idx;
  uint32_t end_tile_kv_idx_in_request;

  bool done;

  MUTLASS_DEVICE MP32VarlenPagedMQALogitsScheduler(uint32_t        mp_idx,
                                                   uint32_t const* context_lens_,
                                                   uint32_t const* metadata,
                                                   uint32_t const* indices_,
                                                   uint32_t        num_q_tokens_total_) {
    context_lens       = context_lens_;
    indices            = indices_;
    num_q_tokens_total = num_q_tokens_total_;

    uint2 const start = __ldg(reinterpret_cast<uint2 const*>(metadata) + mp_idx);
    uint2 const end   = __ldg(reinterpret_cast<uint2 const*>(metadata) + mp_idx + 1);

    cur_tile_kv_idx_in_request = start.y;
    cur_tile_q_idx_in_request  = 0;
    end_token_q_idx            = end.x;
    end_tile_kv_idx_in_request = end.y;

    done = start.x >= num_q_tokens_total || (start.x == end_token_q_idx && start.y >= end_tile_kv_idx_in_request);

    if (!done) {
      cur_request_info = RequestInfo::from_token_q(start.x, num_q_tokens_total, context_lens, indices);
    }
  }

  MUTLASS_DEVICE bool get_next_task(Task& task) {
    while (!done) {
      uint32_t const tile_kv_upper =
          cur_request_info.token_q_start == end_token_q_idx ? end_tile_kv_idx_in_request : cur_request_info.num_tile_kv;
      if (cur_tile_kv_idx_in_request >= tile_kv_upper) {
        if (cur_request_info.token_q_start == end_token_q_idx) {
          done = true;
          break;
        }

        uint32_t const next_token_q = cur_request_info.token_q_start + cur_request_info.num_token_q;
        cur_tile_kv_idx_in_request  = 0;
        cur_tile_q_idx_in_request   = 0;
        if (next_token_q >= num_q_tokens_total ||
            (next_token_q == end_token_q_idx && end_tile_kv_idx_in_request == 0)) {
          done = true;
          break;
        }
        cur_request_info = RequestInfo::from_token_q(next_token_q, num_q_tokens_total, context_lens, indices);
        continue;
      }

      auto [q_offset_in_request, num_q_in_tile] = cur_request_info.get_tile_q_span(cur_tile_q_idx_in_request);

      uint32_t const rem_tile_kv_in_request = tile_kv_upper - cur_tile_kv_idx_in_request;

      task.token_q_base = cur_request_info.token_q_start + q_offset_in_request;
      task.num_token_q  = num_q_in_tile;
      task.tile_kv_base = cur_tile_kv_idx_in_request;
      task.num_tile_kv  = cur_request_info.num_tile_q == 1
                              ? rem_tile_kv_in_request
                              : (rem_tile_kv_in_request < kTilesPerChunk ? rem_tile_kv_in_request : kTilesPerChunk);
      task.request_id   = cur_request_info.token_q_start;
      task.num_page_kv  = cur_request_info.num_page_kv;

      ++cur_tile_q_idx_in_request;
      if (cur_tile_q_idx_in_request == cur_request_info.num_tile_q) {
        cur_tile_q_idx_in_request = 0;
        cur_tile_kv_idx_in_request += task.num_tile_kv;
      }
      return true;
    }
    return false;
  }
};

}  // namespace mate::deep_gemm
