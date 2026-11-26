#pragma once

#include <cstdint>

#include "mate/mega_moe/mega_moe_layout.hpp"

namespace mega_moe::prefill {

constexpr uint32_t kBlockM                 = 256;
constexpr uint32_t kBlockK                 = 128;
constexpr uint32_t kMaxIntermediateHidden  = 8192;
constexpr uint32_t kMaxL2ArrivalMaskBlocks = 64;

// A source token may fan out to multiple experts on the same destination rank.
// Stage 1 dispatch pulls that token once and expands these compact entries locally.
struct PackedRouteEntry {
  uint32_t value;

  static constexpr uint32_t kTopkBits        = 6;
  static constexpr uint32_t kLocalExpertBits = 10;
  static constexpr uint32_t kPoolSlotBits    = 16;

  static constexpr uint32_t kTopkMask        = (1u << kTopkBits) - 1;
  static constexpr uint32_t kLocalExpertMask = (1u << kLocalExpertBits) - 1;
  static constexpr uint32_t kPoolSlotMask    = (1u << kPoolSlotBits) - 1;

  __host__ __device__ static constexpr PackedRouteEntry pack(uint32_t pool_slot,
                                                             uint32_t local_expert_idx,
                                                             uint32_t topk_idx) {
    return {(pool_slot << (kLocalExpertBits + kTopkBits)) | (local_expert_idx << kTopkBits) | topk_idx};
  }

  __host__ __device__ constexpr uint32_t pool_slot() const {
    return (value >> (kLocalExpertBits + kTopkBits)) & kPoolSlotMask;
  }

  __host__ __device__ constexpr uint32_t local_expert_idx() const {
    return (value >> kTopkBits) & kLocalExpertMask;
  }

  __host__ __device__ constexpr uint32_t topk_idx() const {
    return value & kTopkMask;
  }
};
static_assert(sizeof(PackedRouteEntry) == sizeof(uint32_t));

// Stage 2 uses this metadata to direct-store each expert partial back to the
// originating rank/token/top-k lane. The remaining upper bits are reserved.
struct PackedTokenSrcMetadata {
  uint64_t value;

  static constexpr uint32_t kRankShift   = 32;
  static constexpr uint32_t kTopkShift   = 40;
  static constexpr uint32_t kExpertShift = 48;
  static constexpr uint64_t kValidBit    = 1ull << 63;

  __host__ __device__ static constexpr PackedTokenSrcMetadata pack(uint32_t rank_idx,
                                                                   uint32_t token_idx,
                                                                   uint32_t topk_idx,
                                                                   uint32_t local_expert_idx = 0) {
    return {static_cast<uint64_t>(token_idx) | (static_cast<uint64_t>(rank_idx) << kRankShift) |
            (static_cast<uint64_t>(topk_idx) << kTopkShift) |
            (static_cast<uint64_t>(local_expert_idx) << kExpertShift) | kValidBit};
  }

  __host__ __device__ constexpr uint32_t rank_idx() const {
    return static_cast<uint32_t>((value >> kRankShift) & 0xffu);
  }

  __host__ __device__ constexpr uint32_t token_idx() const {
    return static_cast<uint32_t>(value);
  }

  __host__ __device__ constexpr uint32_t topk_idx() const {
    return static_cast<uint32_t>((value >> kTopkShift) & 0xffu);
  }

  __host__ __device__ constexpr uint32_t local_expert_idx() const {
    return static_cast<uint32_t>((value >> kExpertShift) & 0x3ffu);
  }

  __host__ __device__ constexpr bool valid() const {
    return (value & kValidBit) != 0;
  }
};
static_assert(sizeof(PackedTokenSrcMetadata) == sizeof(uint64_t));

// Preserve the base Workspace control region so the existing grid/MTLink
// barriers can accept this object through a Workspace reference. Replace only
// the decode-oriented, 64B-per-item metadata tail with compact prefill arrays.
struct PrefillSplitWorkspace : public layout::Workspace {
  uint32_t num_topk;

  __host__ __device__ PrefillSplitWorkspace(
      void* base, uint32_t num_ranks, uint32_t num_experts, uint32_t num_max_tokens_per_rank, uint32_t num_topk)
      : layout::Workspace(base, num_ranks, num_experts, num_max_tokens_per_rank, num_topk, kBlockM),
        num_topk(num_topk) {
  }

  __host__ __device__ uint64_t get_num_bytes() const {
    return math::align(get_token_src_metadata_offset() + get_num_token_src_metadata_bytes(),
                       layout::kBufferAlignmentBytes);
  }

  __host__ __device__ void* get_end_ptr() const {
    return math::advance_ptr(base, get_num_bytes());
  }

  __host__ __device__ uint32_t* get_src_token_topk_idx_ptr(uint32_t local_expert_idx = 0,
                                                           uint32_t source_rank_idx  = 0,
                                                           uint32_t source_token_idx = 0) const {
    const uint64_t index =
        (static_cast<uint64_t>(local_expert_idx) * num_ranks + source_rank_idx) * num_max_tokens_per_rank +
        source_token_idx;
    return math::advance_ptr<uint32_t>(base, get_src_token_topk_idx_offset() + index * sizeof(uint32_t));
  }

  __host__ __device__ uint32_t* get_src_route_count_ptr(uint32_t source_rank_idx  = 0,
                                                        uint32_t source_token_idx = 0) const {
    const uint64_t index = static_cast<uint64_t>(source_rank_idx) * num_max_tokens_per_rank + source_token_idx;
    return math::advance_ptr<uint32_t>(base, get_src_route_count_offset() + index * sizeof(uint32_t));
  }

  __host__ __device__ PackedRouteEntry* get_src_route_entry_ptr(uint32_t source_rank_idx  = 0,
                                                                uint32_t source_token_idx = 0,
                                                                uint32_t route_idx        = 0) const {
    const uint64_t index =
        (static_cast<uint64_t>(source_rank_idx) * num_max_tokens_per_rank + source_token_idx) * num_topk + route_idx;
    return math::advance_ptr<PackedRouteEntry>(base, get_src_route_entry_offset() + index * sizeof(PackedRouteEntry));
  }

  __host__ __device__ uint32_t* get_src_active_count_ptr(uint32_t source_rank_idx = 0) const {
    return math::advance_ptr<uint32_t>(
        base, get_src_active_count_offset() + static_cast<uint64_t>(source_rank_idx) * sizeof(uint32_t));
  }

  __host__ __device__ uint32_t* get_src_active_token_ptr(uint32_t source_rank_idx = 0, uint32_t active_idx = 0) const {
    const uint64_t index = static_cast<uint64_t>(source_rank_idx) * num_max_tokens_per_rank + active_idx;
    return math::advance_ptr<uint32_t>(base, get_src_active_token_offset() + index * sizeof(uint32_t));
  }

  __host__ __device__ PackedTokenSrcMetadata* get_token_src_metadata_ptr(uint32_t pool_token_idx = 0) const {
    return math::advance_ptr<PackedTokenSrcMetadata>(
        base, get_token_src_metadata_offset() + static_cast<uint64_t>(pool_token_idx) * sizeof(PackedTokenSrcMetadata));
  }

  // One bit per L1 N block of a logical pool block. The L1 consumer sets its
  // bit once the pool block's activation source and scales are fully read, so
  // the dispatcher can recycle the physical ring slot.
  __host__ __device__ uint64_t* get_l1_consumed_mask_ptr(uint32_t pool_block_idx = 0) const {
    return math::advance_ptr<uint64_t>(
        base, get_l1_consumed_mask_offset() + static_cast<uint64_t>(pool_block_idx) * kLogicBufferSlotBytes);
  }

  __host__ __device__ uint64_t get_control_region_num_bytes() const {
    return get_l1_consumed_mask_offset() + static_cast<uint64_t>(num_max_pool_blocks) * kLogicBufferSlotBytes;
  }

  __host__ __device__ uint64_t get_l1_consumed_mask_offset() const {
    uint64_t offset = kNumBarrierSignalBytes;
    offset += static_cast<uint64_t>(num_experts) * kLogicBufferSlotBytes * 2;
    offset += static_cast<uint64_t>(num_experts_per_rank) * kLogicBufferSlotBytes;
    offset += static_cast<uint64_t>(math::align(num_max_pool_blocks, 2u)) * kLogicBufferSlotBytes;
    offset += static_cast<uint64_t>(num_max_pool_blocks) * kLogicBufferSlotBytes;
    return offset;
  }

  __host__ __device__ uint64_t get_src_token_topk_idx_offset() const {
    return math::align(get_control_region_num_bytes(), layout::kBufferAlignmentBytes);
  }

  __host__ __device__ uint64_t get_src_route_count_offset() const {
    return math::align(get_src_token_topk_idx_offset() + get_num_src_token_topk_idx_bytes(),
                       layout::kBufferAlignmentBytes);
  }

  __host__ __device__ uint64_t get_src_route_entry_offset() const {
    return math::align(get_src_route_count_offset() + get_num_src_route_count_bytes(), layout::kBufferAlignmentBytes);
  }

  __host__ __device__ uint64_t get_token_src_metadata_offset() const {
    return math::align(get_src_active_token_offset() + get_num_src_active_token_bytes(), layout::kBufferAlignmentBytes);
  }

  __host__ __device__ uint64_t get_src_active_count_offset() const {
    return math::align(get_src_route_entry_offset() + get_num_src_route_entry_bytes(), layout::kBufferAlignmentBytes);
  }

  __host__ __device__ uint64_t get_src_active_token_offset() const {
    return math::align(get_src_active_count_offset() + static_cast<uint64_t>(num_ranks) * sizeof(uint32_t),
                       layout::kBufferAlignmentBytes);
  }

 private:
  __host__ __device__ uint64_t get_num_src_token_topk_idx_bytes() const {
    return static_cast<uint64_t>(num_experts_per_rank) * num_ranks * num_max_tokens_per_rank * sizeof(uint32_t);
  }

  __host__ __device__ uint64_t get_num_src_route_count_bytes() const {
    return static_cast<uint64_t>(num_ranks) * num_max_tokens_per_rank * sizeof(uint32_t);
  }

  __host__ __device__ uint64_t get_num_src_route_entry_bytes() const {
    return static_cast<uint64_t>(num_ranks) * num_max_tokens_per_rank * num_topk * sizeof(PackedRouteEntry);
  }

  __host__ __device__ uint64_t get_num_token_src_metadata_bytes() const {
    return static_cast<uint64_t>(num_max_pool_tokens) * sizeof(PackedTokenSrcMetadata);
  }

  __host__ __device__ uint64_t get_num_src_active_token_bytes() const {
    return static_cast<uint64_t>(num_ranks) * num_max_tokens_per_rank * sizeof(uint32_t);
  }
};

struct PrefillSymmBufferLayoutInfo {
  uint64_t num_bytes;
  uint64_t workspace_num_bytes;
  uint32_t num_max_pool_tokens;
  uint32_t num_padded_sf_pool_tokens;
  uint64_t input_token_offset;
  uint64_t input_sf_offset;
  uint64_t input_topk_idx_offset;
  uint64_t input_topk_weights_offset;
  uint64_t l1_token_offset;
  uint64_t l1_sf_offset;
  uint64_t l2_token_offset;
  uint64_t l2_sf_offset;
  uint64_t combine_token_offset;
};

__host__ inline PrefillSymmBufferLayoutInfo get_symm_buffer_layout_info(const uint32_t num_ranks,
                                                                        const uint32_t num_experts,
                                                                        const uint32_t num_max_tokens_per_rank,
                                                                        const uint32_t num_topk,
                                                                        const uint32_t hidden,
                                                                        const uint32_t intermediate_hidden) {
  constexpr uint64_t kSyntheticBase = 1ull << 20;
  auto*              base           = reinterpret_cast<void*>(kSyntheticBase);
  const auto         offset_of      = [](const void* ptr) { return reinterpret_cast<uint64_t>(ptr) - kSyntheticBase; };

  const auto num_experts_per_rank = num_experts / num_ranks;
  const auto num_max_pool_tokens =
      layout::get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank, kBlockM);
  const auto num_padded_sf_pool_tokens = layout::get_num_padded_sf_pool_tokens(num_max_pool_tokens, kBlockM);

  const auto workspace         = PrefillSplitWorkspace(base, num_ranks, num_experts, num_max_tokens_per_rank, num_topk);
  const auto fp8_token_layout  = layout::Data(hidden);
  const auto bf16_token_layout = layout::Data(hidden * sizeof(uint16_t));
  const auto fp8_intermediate_token_layout = layout::Data(intermediate_hidden);
  const auto fp8_sf_layout                 = layout::Data(hidden / 32, false);
  const auto fp8_intermediate_sf_layout    = layout::Data((intermediate_hidden / kBlockK) * sizeof(float));
  const auto input_topk_idx_layout         = layout::Data(num_topk * sizeof(int64_t), false);
  const auto input_topk_weights_layout     = layout::Data(num_topk * sizeof(float), false);

  const auto input_token_buffer = layout::Buffer(fp8_token_layout, 1, num_max_tokens_per_rank, workspace.get_end_ptr());
  const auto input_sf_buffer =
      layout::Buffer(fp8_sf_layout, 1, num_max_tokens_per_rank, input_token_buffer.get_end_ptr());
  const auto input_topk_idx_buffer =
      layout::Buffer(input_topk_idx_layout, 1, num_max_tokens_per_rank, input_sf_buffer.get_end_ptr());
  const auto input_topk_weights_buffer =
      layout::Buffer(input_topk_weights_layout, 1, num_max_tokens_per_rank, input_topk_idx_buffer.get_end_ptr());
  const auto l1_token_buffer =
      layout::Buffer(fp8_token_layout, 1, num_max_pool_tokens, input_topk_weights_buffer.get_end_ptr());
  const auto l1_sf_buffer = layout::Buffer(fp8_sf_layout, 1, num_padded_sf_pool_tokens, l1_token_buffer.get_end_ptr());
  const auto l2_token_buffer =
      layout::Buffer(fp8_intermediate_token_layout, 1, num_max_pool_tokens, l1_sf_buffer.get_end_ptr());
  const auto l2_sf_buffer =
      layout::Buffer(fp8_intermediate_sf_layout, 1, num_padded_sf_pool_tokens, l2_token_buffer.get_end_ptr());
  const auto combine_token_buffer =
      layout::Buffer(bf16_token_layout, num_topk, num_max_tokens_per_rank, l2_sf_buffer.get_end_ptr());

  return {
      offset_of(combine_token_buffer.get_end_ptr()),
      workspace.get_num_bytes(),
      num_max_pool_tokens,
      num_padded_sf_pool_tokens,
      offset_of(input_token_buffer.get_base_ptr()),
      offset_of(input_sf_buffer.get_base_ptr()),
      offset_of(input_topk_idx_buffer.get_base_ptr()),
      offset_of(input_topk_weights_buffer.get_base_ptr()),
      offset_of(l1_token_buffer.get_base_ptr()),
      offset_of(l1_sf_buffer.get_base_ptr()),
      offset_of(l2_token_buffer.get_base_ptr()),
      offset_of(l2_sf_buffer.get_base_ptr()),
      offset_of(combine_token_buffer.get_base_ptr()),
  };
}

}  // namespace mega_moe::prefill
