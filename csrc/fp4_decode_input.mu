#include <musa_bf16.h>
#include <musa_fp4.h>
#include <musa_runtime.h>

#include "op_utils.hpp"

namespace {
__device__ __forceinline__ uint32_t scale_code(float value) {
  uint32_t bits     = __float_as_uint(value);
  uint32_t exponent = min((bits + 0x007fffffu) >> 23, 254u);
  return bits - 0x00400001u <= 0x7f3fffffu ? exponent : 0u;
}

template <class Index>
__global__ void prepare_input_kernel(__mt_bfloat16 const* x,
                                     Index const*         ids,
                                     float const*         weights,
                                     uint8_t*             q,
                                     uint8_t*             sf,
                                     int64_t*             output_ids,
                                     float*               output_weights,
                                     int                  hidden,
                                     int                  groups,
                                     int                  topk) {
  int row = blockIdx.x, group = blockIdx.y * blockDim.x + threadIdx.x;
  if (blockIdx.y == 0 && threadIdx.x < topk) {
    int route = row * topk + threadIdx.x;
    __lsu_st_cache_hint(output_ids + route, int64_t(ids[route]), 0, 2, 1, 1);
    __lsu_st_cache_hint(
        reinterpret_cast<uint32_t*>(output_weights) + route, __float_as_uint(weights[route]), 0, 2, 1, 1);
  }
  if (group >= groups) {
    __threadfence_system_noflush();
    return;
  }
  auto* input = x + uint64_t(row) * hidden + group * 32;
  float values[32];
  float amax = 0.f;
#pragma unroll
  for (int vector = 0; vector < 4; ++vector) {
    int4          raw = *reinterpret_cast<int4 const*>(input + vector * 8);
    __mt_bfloat16 bf16[8];
    *reinterpret_cast<int4*>(bf16) = raw;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float value            = __bfloat162float(bf16[i]);
      values[vector * 8 + i] = value;
      amax                   = fmaxf(amax, fabsf(value));
    }
  }
  uint32_t code  = scale_code(amax / 6.f);
  float    scale = __uint_as_float(code == 0u ? 0x00400000u : code << 23);
  float    inv   = scale > 0.f ? 1.f / scale : 0.f;
  auto*    out   = reinterpret_cast<uint32_t*>(q + uint64_t(row) * (hidden / 2) + group * 16);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int      base = i * 8;
    uint16_t lo   = static_cast<uint16_t>(
        __mt_fp4x4_e2m1(
            make_float4(values[base] * inv, values[base + 1] * inv, values[base + 2] * inv, values[base + 3] * inv))
            .__x);
    uint16_t hi = static_cast<uint16_t>(
        __mt_fp4x4_e2m1(
            make_float4(values[base + 4] * inv, values[base + 5] * inv, values[base + 6] * inv, values[base + 7] * inv))
            .__x);
    __lsu_st_cache_hint(out + i, uint32_t(lo) | (uint32_t(hi) << 16), 0, 2, 1, 1);
  }
  __lsu_st_cache_hint(sf + uint64_t(row) * groups + group, uint8_t(code), 0, 2, 1, 1);
  __threadfence_system_noflush();
}
}  // namespace

void fp4_decode_input(ffi::TensorView x,
                      ffi::TensorView ids,
                      ffi::TensorView weights,
                      ffi::TensorView q,
                      ffi::TensorView sf,
                      ffi::TensorView output_ids,
                      ffi::TensorView output_weights) {
  for (auto tensor : {x, ids, weights, q, sf, output_ids, output_weights}) {
    CHECK_INPUT(tensor);
    CHECK_DEVICE(tensor, x);
  }
  CHECK_INPUT_TYPE(x, dl_bfloat16);
  CHECK_INPUT_TYPE(q, dl_uint8);
  CHECK_INPUT_TYPE(sf, dl_uint8);
  CHECK_INPUT_TYPE(weights, dl_float32);
  CHECK_INPUT_TYPE(output_weights, dl_float32);
  CHECK_INPUT_TYPE(output_ids, dl_int64);
  TVM_FFI_ICHECK(ids.dtype() == dl_int32 || ids.dtype() == dl_int64);
  TVM_FFI_ICHECK_EQ(x.ndim(), 2);
  TVM_FFI_ICHECK_EQ(ids.ndim(), 2);
  int rows = x.size(0), hidden = x.size(1), topk = ids.size(1);
  TVM_FFI_ICHECK(rows > 0 && rows <= 128 && hidden > 0 && hidden % 256 == 0 && topk > 0 && topk <= 32);
  expect_shape(ids, {rows, topk}, "ids");
  expect_shape(weights, {rows, topk}, "weights");
  expect_shape(output_ids, {rows, topk}, "output_ids");
  expect_shape(output_weights, {rows, topk}, "output_weights");
  expect_shape(q, {rows, hidden / 2}, "q");
  expect_shape(sf, {rows, hidden / 32}, "sf");
  ffi::MUSADeviceGuard guard(x.device().device_id);
  dim3                 grid(rows, (hidden / 32 + 255) / 256);
  if (ids.dtype() == dl_int32)
    prepare_input_kernel<<<grid, 256, 0, get_stream(x.device())>>>(static_cast<__mt_bfloat16 const*>(x.data_ptr()),
                                                                   static_cast<int32_t const*>(ids.data_ptr()),
                                                                   static_cast<float const*>(weights.data_ptr()),
                                                                   static_cast<uint8_t*>(q.data_ptr()),
                                                                   static_cast<uint8_t*>(sf.data_ptr()),
                                                                   static_cast<int64_t*>(output_ids.data_ptr()),
                                                                   static_cast<float*>(output_weights.data_ptr()),
                                                                   hidden,
                                                                   hidden / 32,
                                                                   topk);
  else
    prepare_input_kernel<<<grid, 256, 0, get_stream(x.device())>>>(static_cast<__mt_bfloat16 const*>(x.data_ptr()),
                                                                   static_cast<int64_t const*>(ids.data_ptr()),
                                                                   static_cast<float const*>(weights.data_ptr()),
                                                                   static_cast<uint8_t*>(q.data_ptr()),
                                                                   static_cast<uint8_t*>(sf.data_ptr()),
                                                                   static_cast<int64_t*>(output_ids.data_ptr()),
                                                                   static_cast<float*>(output_weights.data_ptr()),
                                                                   hidden,
                                                                   hidden / 32,
                                                                   topk);
  MATE_CHECK_MUSA_KERNEL_LAUNCH();
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp4_decode_input, fp4_decode_input);
