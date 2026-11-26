#pragma once
#include "mate/mega_moe/prefill/fp4_fp4/symmetric_layout.hpp"

namespace mega_moe::prefill::mp32::fp4_fp4 {

struct Fp4Quantization {
  static constexpr int      kScaleBias = 127, kMantissaBits = 23, kMaxFiniteExponent = 2;
  static constexpr uint32_t kMantissaMask         = (1u << kMantissaBits) - 1u;
  static constexpr uint32_t kMaxFiniteMantissa    = 1u << (kMantissaBits - 1);  // 6 = 1.5 * 2^2
  static constexpr uint32_t kInverseAtMaxExponent = 1u << (kMantissaBits - 1);
  __device__ static int     scale_exponent(float amax) {
    if (amax == 0.f) return 0;
    uint32_t bits = __float_as_uint(amax);
    int      exponent =
        int(bits >> kMantissaBits) - kScaleBias - kMaxFiniteExponent + int((bits & kMantissaMask) > kMaxFiniteMantissa);
    return max(-kScaleBias, min(kScaleBias, exponent));
  }
  __device__ static float inverse(int exponent) {
    uint32_t bits = exponent == kScaleBias ? kInverseAtMaxExponent : uint32_t(kScaleBias - exponent) << kMantissaBits;
    return __uint_as_float(bits);
  }
};

__device__ inline float silu_fp32(float gate) {
  float denominator = 1.f + __expf(-gate);
  float inverse     = __frcp_rn(denominator);
  float quotient    = gate * inverse;
  if (isfinite(denominator) && quotient != 0.f) quotient = fmaf(fmaf(-quotient, denominator, gate), inverse, quotient);
  return quotient;
}

using ActivationFloat4 = float __attribute__((ext_vector_type(4)));
using ActivationBf16x4 = __bf16 __attribute__((ext_vector_type(4)));
using ActivationBits4  = uint16_t __attribute__((ext_vector_type(4)));
// The retained SDK exposes RCP BST4 through LLVM rather than a C++ builtin.
extern __device__ ActivationFloat4 activation_rcp_bst4(ActivationFloat4) __asm__("llvm.musa.rcp.f.bst4");

__device__ inline ActivationFloat4 silu_fp32_bst4(ActivationFloat4 gate) {
  constexpr float  kLog2e   = 1.4426950408889634074f;
  constexpr float  kMinExp2 = -126.f, kMaxExp2 = 128.f;
  constexpr int    kVectorElements = sizeof(ActivationFloat4) / sizeof(float);
  ActivationFloat4 x;
  MUTE_UNROLL
  for (int j = 0; j < kVectorElements; ++j) x[j] = fminf(kMaxExp2, fmaxf(kMinExp2, -gate[j] * kLog2e));
  ActivationFloat4 denominator = __musa_exp2_f_bst4(x) + 1.f;
  ActivationFloat4 inverse     = activation_rcp_bst4(denominator);
  ActivationFloat4 quotient    = gate * inverse;
  constexpr float  kMaxFinite  = std::numeric_limits<float>::max();
  MUTE_UNROLL
  for (int j = 0; j < kVectorElements; ++j) {
    // Infinity has reciprocal zero. A finite denominator keeps the correction
    // defined in that case, and restoring the gate sign preserves negative zero.
    float finite_denominator = denominator[j] < kMaxFinite ? denominator[j] : kMaxFinite;
    float residual           = fmaf(-quotient[j], finite_denominator, gate[j]);
    quotient[j]              = copysignf(fmaf(residual, inverse[j], quotient[j]), gate[j]);
  }
  return quotient;
}

__device__ inline uint8_t e2m1_rne(float value) {
  float magnitude = fminf(6.f, fabsf(value));
  // Crossing each midpoint advances the code once. Alternate > and >=
  // so an exact tie selects the even E2M1 code without divergent branches.
  int code = int(magnitude > 0.25f) + int(magnitude >= 0.75f) + int(magnitude > 1.25f) + int(magnitude >= 1.75f) +
             int(magnitude > 2.5f) + int(magnitude >= 3.5f) + int(magnitude > 5.f);
  return uint8_t(code | ((__float_as_uint(value) >> 31) << 3));
}

}  // namespace mega_moe::prefill::mp32::fp4_fp4
