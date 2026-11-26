#include <musa_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/stl.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ffi = tvm::ffi;

namespace {

std::mutex                               event_mutex;
std::unordered_map<int64_t, musaEvent_t> events;
int64_t                                  next_event_handle = 1;

void check_musa(musaError_t status, const char* operation) {
  if (status != musaSuccess) {
    TVM_FFI_THROW(RuntimeError) << operation << " failed: " << musaGetErrorString(status);
  }
}

struct ManagedMusaTensor {
  DLManagedTensorVersioned managed{};
  std::vector<int64_t>     shape;
  std::vector<int64_t>     strides;
};

void managed_tensor_deleter(DLManagedTensorVersioned* managed) {
  auto* holder = static_cast<ManagedMusaTensor*>(managed->manager_ctx);
  if (managed->dl_tensor.data != nullptr) {
    musaFree(managed->dl_tensor.data);
  }
  delete holder;
}

int allocate_managed_tensor(DLTensor*                  prototype,
                            DLManagedTensorVersioned** out,
                            void*                      error_ctx,
                            void (*set_error)(void*, const char*, const char*)) {
  try {
    if (prototype->device.device_type != kDLExtDev) {
      set_error(error_ctx, "ValueError", "JIT dump loader only allocates MUSA tensors");
      return -1;
    }

    auto holder = std::make_unique<ManagedMusaTensor>();
    holder->shape.assign(prototype->shape, prototype->shape + prototype->ndim);
    holder->strides.resize(holder->shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(holder->shape.size()) - 2; i >= 0; --i) {
      holder->strides[i] = holder->strides[i + 1] * holder->shape[i + 1];
    }

    holder->managed.version               = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    holder->managed.manager_ctx           = holder.get();
    holder->managed.deleter               = managed_tensor_deleter;
    holder->managed.flags                 = 0;
    holder->managed.dl_tensor             = *prototype;
    holder->managed.dl_tensor.data        = nullptr;
    holder->managed.dl_tensor.shape       = holder->shape.data();
    holder->managed.dl_tensor.strides     = holder->strides.data();
    holder->managed.dl_tensor.byte_offset = 0;

    check_musa(musaSetDevice(prototype->device.device_id), "musaSetDevice");
    const size_t nbytes = ffi::GetDataSize(holder->managed.dl_tensor);
    if (nbytes != 0) {
      check_musa(musaMalloc(&holder->managed.dl_tensor.data, nbytes), "musaMalloc");
    }
    *out = &holder->managed;
    holder.release();
    return 0;
  } catch (const std::exception& exc) {
    set_error(error_ctx, "RuntimeError", exc.what());
    return -1;
  }
}

ffi::Tensor empty_tensor(std::vector<int64_t> shape, ffi::String dtype_name, int device_id) {
  ffi::Shape ffi_shape(std::move(shape));
  DLDevice   device{kDLExtDev, device_id};
  DLDataType dtype = ffi::StringToDLDataType(dtype_name);
  return ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, ffi_shape, dtype, device);
}

ffi::Tensor load_tensor(ffi::String path, std::vector<int64_t> shape, ffi::String dtype_name, int device_id) {
  ffi::Tensor  tensor          = empty_tensor(std::move(shape), dtype_name, device_id);
  const size_t expected_nbytes = ffi::GetDataSize(*tensor.GetDLTensorPtr());

  std::ifstream input(std::string(path), std::ios::binary | std::ios::ate);
  TVM_FFI_ICHECK(input.is_open()) << "cannot open tensor file: " << path;
  const std::streamsize file_size = input.tellg();
  TVM_FFI_ICHECK_GE(file_size, 0) << "cannot determine tensor file size: " << path;
  TVM_FFI_ICHECK_EQ(static_cast<size_t>(file_size), expected_nbytes) << "tensor file size mismatch for " << path;

  std::vector<char> host_data(expected_nbytes);
  input.seekg(0, std::ios::beg);
  if (expected_nbytes != 0) {
    TVM_FFI_ICHECK(input.read(host_data.data(), file_size)) << "cannot read tensor file: " << path;
    check_musa(musaMemcpy(tensor.data_ptr(), host_data.data(), expected_nbytes, musaMemcpyHostToDevice),
               "musaMemcpyHostToDevice");
  }
  return tensor;
}

void save_tensor(ffi::Tensor tensor, ffi::String path) {
  const DLTensor* dl_tensor = tensor.GetDLTensorPtr();
  TVM_FFI_ICHECK_EQ(dl_tensor->device.device_type, kDLExtDev) << "JIT dump loader only saves MUSA tensors";

  int64_t expected_stride = 1;
  if (dl_tensor->strides != nullptr) {
    for (int index = dl_tensor->ndim - 1; index >= 0; --index) {
      TVM_FFI_ICHECK_EQ(dl_tensor->strides[index], expected_stride) << "JIT dump loader only saves contiguous tensors";
      expected_stride *= dl_tensor->shape[index];
    }
  }

  check_musa(musaSetDevice(dl_tensor->device.device_id), "musaSetDevice");
  const size_t      nbytes = ffi::GetDataSize(*dl_tensor);
  std::vector<char> host_data(nbytes);
  if (nbytes != 0) {
    check_musa(musaMemcpy(host_data.data(), tensor.data_ptr(), nbytes, musaMemcpyDeviceToHost),
               "musaMemcpyDeviceToHost");
  }

  std::ofstream output(std::string(path), std::ios::binary | std::ios::trunc);
  TVM_FFI_ICHECK(output.is_open()) << "cannot open tensor output file: " << path;
  if (nbytes != 0) {
    output.write(host_data.data(), static_cast<std::streamsize>(nbytes));
    TVM_FFI_ICHECK(output.good()) << "cannot write tensor output file: " << path;
  }
}

void install_allocator() {
  TVM_FFI_CHECK_SAFE_CALL(TVMFFIEnvSetDLPackManagedTensorAllocator(allocate_managed_tensor, 1, nullptr));
}

void synchronize(int device_id) {
  check_musa(musaSetDevice(device_id), "musaSetDevice");
  check_musa(musaDeviceSynchronize(), "musaDeviceSynchronize");
}

musaEvent_t lookup_event_unlocked(int64_t handle) {
  auto iterator = events.find(handle);
  TVM_FFI_ICHECK(iterator != events.end()) << "unknown JIT dump loader event: " << handle;
  return iterator->second;
}

int64_t create_event(int device_id) {
  check_musa(musaSetDevice(device_id), "musaSetDevice");
  musaEvent_t event = nullptr;
  check_musa(musaEventCreate(&event), "musaEventCreate");

  std::lock_guard<std::mutex> lock(event_mutex);
  const int64_t               handle = next_event_handle++;
  events.emplace(handle, event);
  return handle;
}

void record_event(int64_t handle, int device_id) {
  check_musa(musaSetDevice(device_id), "musaSetDevice");
  std::lock_guard<std::mutex> lock(event_mutex);
  check_musa(musaEventRecord(lookup_event_unlocked(handle)), "musaEventRecord");
}

float elapsed_time_ms(int64_t start_handle, int64_t end_handle) {
  float                       elapsed_ms = 0.0f;
  std::lock_guard<std::mutex> lock(event_mutex);
  check_musa(musaEventElapsedTime(&elapsed_ms, lookup_event_unlocked(start_handle), lookup_event_unlocked(end_handle)),
             "musaEventElapsedTime");
  return elapsed_ms;
}

void destroy_event(int64_t handle) {
  musaEvent_t event = nullptr;
  {
    std::lock_guard<std::mutex> lock(event_mutex);
    event = lookup_event_unlocked(handle);
    events.erase(handle);
  }
  check_musa(musaEventDestroy(event), "musaEventDestroy");
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(jit_dump_loader_empty_tensor, empty_tensor);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jit_dump_loader_load_tensor, load_tensor);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jit_dump_loader_save_tensor, save_tensor);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jit_dump_loader_install_allocator, install_allocator);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jit_dump_loader_synchronize, synchronize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jit_dump_loader_create_event, create_event);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jit_dump_loader_record_event, record_event);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jit_dump_loader_elapsed_time_ms, elapsed_time_ms);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jit_dump_loader_destroy_event, destroy_event);
