#pragma once
// Load MUTLASS before the FFI Tensor alias introduced by the barrier helpers.
// clang-format off
#include "mate/mega_moe/prefill/fp4_fp4/ss_grouped_gemm.hpp"
#include "mate/mega_moe/mega_barrier.hpp"
#include "mate/mega_moe/prefill/prefill_workspace.hpp"
// clang-format on

namespace mega_moe::prefill::mp32::fp4_fp4 {

template <int R, int T, int H, int I, int E, int TopK>
struct SymmetricLayout {
  static constexpr int      kRanks = R, kTokens = T, kHidden = H, kIntermediate = I, kExperts = E, kTopK = TopK;
  static constexpr int      kGroups = E / R, kCapacity = R * T, kScaleBlock = 32, kBlockM = 256;
  static constexpr uint64_t kAlignment = 256, kBarrierBytes = 512;
  static constexpr uint64_t align(uint64_t value) {
    return (value + kAlignment - 1) / kAlignment * kAlignment;
  }
  static constexpr uint64_t kX       = kBarrierBytes;
  static constexpr uint64_t kXS      = align(kX + uint64_t(T) * (H / 2));
  static constexpr uint64_t kRoutes  = align(kXS + uint64_t(T) * (H / kScaleBlock));
  static constexpr uint64_t kWeights = align(kRoutes + uint64_t(T) * TopK * sizeof(int64_t));
  static constexpr uint64_t kCombine = align(kWeights + uint64_t(T) * TopK * sizeof(float));
  static constexpr uint64_t kBytes   = align(kCombine + uint64_t(T) * TopK * H * sizeof(mutlass::bfloat16_t));
  static_assert(layout::Workspace::kNumBarrierSignalBytes <= kBarrierBytes);
  static_assert(T % kBlockM == 0 && H % 256 == 0 && I % 256 == 0 && E % R == 0);

  // The compact dispatch workspace has independent persistent phase words;
  // IPC probes retain their independent prefix counters.
  static uint64_t dispatch_workspace_bytes() {
    return PrefillSplitWorkspace(nullptr, R, E, T, TopK).get_num_bytes();
  }
  static uint64_t allocation_bytes() {
    return kBytes + dispatch_workspace_bytes();
  }

  template <class U>
  __device__ static U* at(char* base, uint64_t offset) {
    return reinterpret_cast<U*>(base + offset);
  }
};

struct CtaSync {
  __device__ void operator()() const {
    __syncthreads();
  }
};

template <class L>
__global__ void ipc_barrier_kernel(layout::SymBuffer<L::kRanks> sym) {
  layout::Workspace workspace(sym.get_base_ptr(), L::kRanks, L::kExperts, L::kTokens, L::kTopK, L::kBlockM);
  // This one-CTA barrier is launched at a kernel boundary. Its phase counters
  // persist across calls and are never included in per-invocation workspace clear.
  mtlink_barrier<L::kRanks, 1, 128, 0, 0, true>(workspace, sym, 0, threadIdx.x, CtaSync{});
}

template <class L>
__global__ void ipc_probe_kernel(layout::SymBuffer<L::kRanks> sym, uint8_t* out) {
  int rank = threadIdx.x;
  if (rank < L::kRanks) {
    auto local = L::template at<uint8_t>(sym.template get_base_ptr<char*>(), L::kX);
    out[rank]  = uint8_t(__lsu_ld_cache_hint(reinterpret_cast<uint32_t const*>(sym.map(local, rank)), 3, 3, 0, 0));
  }
}

}  // namespace mega_moe::prefill::mp32::fp4_fp4
