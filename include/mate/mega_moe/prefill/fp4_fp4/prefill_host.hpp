#pragma once
#include "mate/mega_moe/prefill/fp4_fp4/symmetric_layout.hpp"
#include "op_utils.hpp"

namespace mate::fp4_prefill_host {

inline void tensor(
    ffi::TensorView value, DLDataType dtype, DLDevice device, std::initializer_list<int64_t> shape, char const* name) {
  CHECK_INPUT(value);
  CHECK_INPUT_TYPE(value, dtype);
  TVM_FFI_ICHECK_EQ(value.device().device_type, device.device_type);
  TVM_FFI_ICHECK_EQ(value.device().device_id, device.device_id);
  expect_shape(value, shape, name);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(value.data_ptr()) % 16, 0);
}

inline void mps(int value, DLDevice device) {
  musaDeviceProp props{};
  MATE_MUSA_RUNTIME_CHECK(musaGetDeviceProperties(&props, device.device_id));
  TVM_FFI_ICHECK_GT(value, 0);
  TVM_FFI_ICHECK_LE(value, props.multiProcessorCount);
}

template <class L>
inline auto sym(ffi::TensorView buffer, ffi::TensorView ptrs, int rank) {
  tensor(buffer, dl_uint8, buffer.device(), {buffer.numel()}, "buffer");
  // Decode-enabled unified allocations append their independent M32 control
  // region after prefill. The common prefix and prefill offsets stay fixed.
  TVM_FFI_ICHECK_GE(buffer.numel(), L::allocation_bytes());
  check_mp32(buffer.device(), "FP4 Mega MoE");
  auto result = mate::mega_moe::make_sym_buffer<L::kRanks>(ptrs, rank);
  TVM_FFI_ICHECK_EQ(result.base, reinterpret_cast<int64_t>(buffer.data_ptr()));
  return result;
}

}  // namespace mate::fp4_prefill_host
