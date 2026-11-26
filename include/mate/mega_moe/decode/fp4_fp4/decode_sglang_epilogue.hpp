#pragma once

#include "mate/mega_moe/decode/fp4_fp4/decode_common.hpp"

namespace mega_moe::decode::mp32::fp4_fp4 {

// SGLang's packed W4A4 contract: BF16 gate/up, FP32 exp2 SwiGLU and
// route weighting before E2M1 quantization. Keep the M32 online-N swizzle.
__device__ inline uint32_t sg_scale_code(float value) {
  uint32_t bits     = __float_as_uint(value);
  uint32_t exponent = min((bits + 0x007fffffu) >> 23, 254u);
  return bits - 0x00400001u <= 0x7f3fffffu ? exponent : 0u;
}

template <class Compute, class Accum, class Mma, class Desc, class Workspace>
MUTLASS_DEVICE void sglang_activation(Accum const&     accum,
                                      Mma const&       mma,
                                      int              tid,
                                      Desc const&      desc,
                                      Workspace const& workspace,
                                      uint8_t*         l2,
                                      uint8_t*         sf2,
                                      int              pool,
                                      int              intermediate,
                                      float            clamp) {
  constexpr int Values = 8, Lanes = 4, Group = 16;
  auto     coords = mma.get_thread_slice(tid).partition_C(make_identity_tensor(take<0, 2>(typename Compute::Tile{})));
  int      row    = int(get<0>(coords(0)));
  unsigned valid  = __ballot_sync(0xffffffffu, row < int(desc.valid_m));
  if (row >= int(desc.valid_m)) return;
  uint64_t outrow = uint64_t(desc.pool_block_idx) * 32 + row;
  auto     metadata =
      __lsu_ld_cache_hint(reinterpret_cast<uint4 const*>(workspace.get_token_src_metadata_ptr(outrow)), 0, 2, 1, 0);
  float route_weight = __uint_as_float(metadata.w);
  MUTE_UNROLL
  for (int group = 0; group < size(accum) / Group; ++group) {
    float values[Values];
    float amax = 0.f;
    MUTE_UNROLL
    for (int col = 0; col < Values; ++col) {
      float gate = float(mutlass::bfloat16_t(accum(group * Group + col)));
      float up   = float(mutlass::bfloat16_t(accum(group * Group + Values + col)));
      if (clamp > 0.f) {
        gate = fminf(gate, clamp);
        up   = fminf(fmaxf(up, -clamp), clamp);
      }
      float sigmoid = 1.f / (exp2f(gate * -1.4426950408889634f) + 1.f);
      values[col]   = (gate * sigmoid * up) * route_weight;
      amax          = fmaxf(amax, fabsf(values[col]));
    }
    amax              = fmaxf(amax, __shfl_xor_sync(valid, amax, 2));
    amax              = fmaxf(amax, __shfl_xor_sync(valid, amax, 1));
    uint32_t code     = sg_scale_code(amax * (1.f / 6.f));
    float    scale    = __uint_as_float(code == 0u ? 0x00400000u : code << 23);
    float    inverse  = scale > 0.f ? 1.f / scale : 0.f;
    int      n        = int(get<1>(coords(group * Group)));
    int      outgroup = desc.n_block_idx * (Compute::kBlockN / 64) + n / 64;
    if (tid % Lanes == 0) sf2[uint64_t(outgroup) * pool + outrow] = uint8_t(code);
    uint16_t lo = static_cast<uint16_t>(
        __mt_fp4x4_e2m1(make_float4(values[0] * inverse, values[1] * inverse, values[2] * inverse, values[3] * inverse))
            .__x);
    uint16_t hi = static_cast<uint16_t>(
        __mt_fp4x4_e2m1(make_float4(values[4] * inverse, values[5] * inverse, values[6] * inverse, values[7] * inverse))
            .__x);
    auto* dst                         = l2 + outrow * (intermediate / 2) + outgroup * 16 + (n % 32) / 2;
    *reinterpret_cast<uint32_t*>(dst) = uint32_t(lo) | (uint32_t(hi) << 16);
  }
}
}  // namespace mega_moe::decode::mp32::fp4_fp4
