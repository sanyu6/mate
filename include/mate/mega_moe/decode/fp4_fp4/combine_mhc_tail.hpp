#pragma once
#ifndef COMBINE_FAST_QUANT
#define COMBINE_FAST_QUANT 0
#endif
__device__ __forceinline__ float combine_quant_div(float a, float b) {
  if constexpr (COMBINE_FAST_QUANT)
    return __fdividef(a, b);
  else
    return a / b;
}
// Adapted from the validated SGLang MUSA post/combine/norm quant boundary.
// Keep intermediate BF16 rounding and the explicit FP32 reduction tree.
#include <musa_bf16.h>
#include <musa_fp8.h>
namespace mate::combine_mhc_tail {
constexpr int kChannels         = 4;
constexpr int kHidden           = 5120;
constexpr int kGroupSize        = 8;
constexpr int kReductionThreads = 128;
constexpr int kGroups           = kReductionThreads;
static_assert(kGroups * 5 * kGroupSize == kHidden);
constexpr int kParts              = 20;
constexpr int kPartHidden         = kHidden / kParts;
constexpr int kQuantGroup         = 32;
constexpr int kValuesPerLane      = 4;
constexpr int kLanesPerQuantGroup = kQuantGroup / kValuesPerLane;
constexpr int kFp8Max             = 448;

__device__ __forceinline__ uint32_t quant_scale_code(float amax) {
  const float raw     = fmaxf(fmaxf(amax, 1.0e-30f) * (1.0f / kFp8Max), 1.0e-30f);
  uint32_t    bits    = __float_as_uint(raw);
  bits                = (bits + 0x7fffffu) & 0xff800000u;
  const uint32_t code = (bits >> 23) & 0xffu;
  return code < 255u ? code : 254u;
}

template <bool Quantize, int Parts, bool PadResidual = false>
__device__ void mhc_post_combine_norm_body(const __mt_bfloat16* __restrict__ x,
                                           const __mt_bfloat16* __restrict__ residual,
                                           const float* __restrict__ post,
                                           const float* __restrict__ comb,
                                           const float* __restrict__ pre,
                                           const __mt_bfloat16* __restrict__ weight,
                                           __mt_bfloat16* __restrict__ residual_out,
                                           __mt_bfloat16* __restrict__ output,
                                           __mt_fp8_e4m3* __restrict__ output_q,
                                           uint8_t* __restrict__ output_scale,
                                           int                  rows,
                                           int64_t              x_row_stride,
                                           int64_t              pre_row_stride,
                                           float                eps,
                                           int                  row_index,
                                           const __mt_bfloat16* local_shared,
                                           char*                scratch) {
  auto*         combined            = reinterpret_cast<__mt_bfloat16*>(scratch);
  auto*         group_sum           = reinterpret_cast<float*>(scratch + 10240);
  auto*         part_values         = combined;
  constexpr int kPartHidden         = kHidden / Parts;
  constexpr int kQuantGroupsPerPart = kPartHidden / kQuantGroup;

  const int row        = row_index;
  const int part       = 0;
  const int tid        = static_cast<int>(threadIdx.x);
  const int part_begin = part * kPartHidden;
  const int part_end   = part_begin + kPartHidden;

  if constexpr (PadResidual) {
    if (row >= rows) {
      for (int channel = 0; channel < kChannels; ++channel) {
        for (int h = part_begin + tid; h < part_end; h += blockDim.x) {
          residual_out[(row * kChannels + channel) * kHidden + h] = __float2bfloat16(0.0f);
        }
      }
      return;
    }
  }

  for (int h = tid; h < kHidden; h += blockDim.x) {
    const float xv = float(
        __float2bfloat16(__fadd_rn(float(x[row * x_row_stride + h]), float(local_shared[row * x_row_stride + h]))));
    __mt_bfloat16 post_split[4];
    const float   residual0 = float(residual[(row * kChannels) * kHidden + h]);
#pragma unroll
    for (int channel = 0; channel < kChannels; ++channel) {
      const float post_value = post[row * kChannels + channel];
      const float comb_value = comb[row * kChannels * kChannels + channel];
      float       acc        = __fmaf_rn(post_value, xv, __fmul_rn(comb_value, residual0));
#pragma unroll
      for (int i = 1; i < kChannels; ++i) {
        const float residual_value = float(residual[(row * kChannels + i) * kHidden + h]);
        const float comb_ij        = comb[row * kChannels * kChannels + i * kChannels + channel];
        acc                        = __fmaf_rn(comb_ij, residual_value, acc);
      }
      const __mt_bfloat16 rounded = __float2bfloat16(acc);
      post_split[channel]         = rounded;
      if (h >= part_begin && h < part_end) {
        residual_out[(row * kChannels + channel) * kHidden + h] = rounded;
      }
    }

    float value = 0.0f;
#pragma unroll
    for (int channel = 0; channel < kChannels; ++channel) {
      const float xv_post = float(post_split[channel]);
      const float pv      = pre[row * pre_row_stride + channel];
      value               = __fadd_rn(value, __fmul_rn(xv_post, pv));
    }
    combined[h] = __float2bfloat16(value);
  }
  __syncthreads();

  if (tid < kGroups) {
    float acc = 0.0f;
#pragma unroll
    for (int block = 0; block < 5; ++block) {
#pragma unroll
      for (int j = 0; j < kGroupSize; ++j) {
        const int   h     = block * 1024 + tid * kGroupSize + j;
        const float value = combined[h];
        acc               = __fmaf_rn(value, value, acc);
      }
    }
    group_sum[tid] = acc;
  }
  __syncthreads();

  if (tid < 32) {
    float       value = __fadd_rn(group_sum[tid], group_sum[tid + 64]);
    const float upper = __fadd_rn(group_sum[tid + 32], group_sum[tid + 96]);
    value             = __fadd_rn(value, upper);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    if (tid == 0) {
      group_sum[0] = value;
    }
  }
  __syncthreads();

  const float mean = __fadd_rn(group_sum[0] * (1.0f / 5120.0f), eps);
  float       inv  = rsqrtf(mean);
  inv              = __fmul_rn(inv, __fmaf_rn(__fmul_rn(-0.5f, mean) * inv, inv, 1.5f));
  for (int h = part_begin + tid; h < part_end; h += blockDim.x) {
    const float         value      = __fmul_rn(float(combined[h]), inv);
    const float         result     = __fmul_rn(value, float(weight[h]));
    const __mt_bfloat16 normalized = __float2bfloat16(result);
    output[row * kHidden + h]      = normalized;
    if constexpr (Quantize) {
      part_values[h - part_begin] = normalized;
    }
  }
  if constexpr (Quantize) {
    __syncthreads();
    for (int quant_tid = tid; quant_tid < kQuantGroupsPerPart * kLanesPerQuantGroup; quant_tid += blockDim.x) {
      const int group_in_part = quant_tid / kLanesPerQuantGroup;
      const int lane          = quant_tid % kLanesPerQuantGroup;
      const int local_col     = group_in_part * kQuantGroup + lane * kValuesPerLane;
      float     values[kValuesPerLane];
      float     local_absmax = 0.0f;
#pragma unroll
      for (int vec = 0; vec < kValuesPerLane; ++vec) {
        values[vec]  = float(part_values[local_col + vec]);
        local_absmax = fmaxf(local_absmax, fabsf(values[vec]));
      }
      local_absmax              = fmaxf(local_absmax, __shfl_xor_sync(0xffffffffu, local_absmax, 4));
      local_absmax              = fmaxf(local_absmax, __shfl_xor_sync(0xffffffffu, local_absmax, 2));
      local_absmax              = fmaxf(local_absmax, __shfl_xor_sync(0xffffffffu, local_absmax, 1));
      const uint32_t scale_code = quant_scale_code(local_absmax);
      const float    scale      = exp2f(static_cast<float>(scale_code) - 127.0f);
      const float4   quantized  = {fminf(fmaxf(combine_quant_div(values[0], scale), -kFp8Max), kFp8Max),
                                   fminf(fmaxf(combine_quant_div(values[1], scale), -kFp8Max), kFp8Max),
                                   fminf(fmaxf(combine_quant_div(values[2], scale), -kFp8Max), kFp8Max),
                                   fminf(fmaxf(combine_quant_div(values[3], scale), -kFp8Max), kFp8Max)};
      const uint32_t packed     = __musa_cvt_float4_to_fp8x4(quantized, __MT_SATFINITE, __MT_E4M3);
      *reinterpret_cast<uint32_t*>(output_q + row * kHidden + part_begin + local_col) = packed;
      if (lane == 0) {
        const int group                  = part * kQuantGroupsPerPart + group_in_part;
        output_scale[group * rows + row] = static_cast<uint8_t>(scale_code);
      }
    }
  }
}

}  // namespace mate::combine_mhc_tail
