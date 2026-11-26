#include <mutlass/device_kernel.h>

#include <algorithm>
#include <optional>

#include "attention_combine.hpp"
#include "attention_scheduler.hpp"
#include "mate/attention/flash_mla/mpxx_params.hpp"
#include "op_utils.hpp"

using namespace mute;

struct DecodingAttnImplMeta {
  int num_mp_parts;
  int fixed_overhead_num_blocks;
  int k_block_size;
};

DecodingAttnImplMeta get_attn_impl_meta(Arch                   arch,
                                        int                    mp_count,
                                        int                    num_q_tokens_per_head_k,
                                        int                    h_k,
                                        ffi::Optional<int64_t> h_q,
                                        bool                   is_fp8_kvcache,
                                        bool                   is_sparse_attn,
                                        int                    tile_m = 128) {
  (void)h_q;
  if (is_sparse_attn) {
    TVM_FFI_ICHECK(arch.is_mp31() || arch.is_mp32()) << "Sparse MLA only supports MP31 and MP32";
    if (is_fp8_kvcache) {
      return {std::max(mp_count / h_k / mutlass::ceil_div(num_q_tokens_per_head_k, tile_m), 1), 5, 64};
    } else {
      TVM_FFI_ICHECK(false) << "Sparse BF16 MLA is not supported on MP31 or MP32";
    }
  } else {
    TVM_FFI_ICHECK(arch.is_mp31()) << "Dense MLA only supports MP31";
    if (is_fp8_kvcache) {
      TVM_FFI_ICHECK(false) << "Dense FP8 MLA is not supported on MP31";
    } else {
      return {std::max(mp_count / h_k / mutlass::ceil_div(num_q_tokens_per_head_k, tile_m), 1), 5, 64};
    }
  }
  return {0, 0, 0};
}

namespace {

struct MetadataResult {
  ffi::Tensor     tile_scheduler_metadata_storage;
  ffi::Tensor     num_splits_storage;
  ffi::TensorView tile_scheduler_metadata;
  ffi::TensorView num_splits;
};

}  // namespace

MetadataResult get_mla_decoding_metadata_impl(ffi::Optional<ffi::TensorView> seqlens_k,
                                              int64_t                        num_q_tokens_per_head_k,
                                              int64_t                        h_k,
                                              ffi::Optional<int64_t>         h_q,
                                              bool                           is_fp8_kvcache,
                                              ffi::Optional<int64_t>         topk,
                                              ffi::Optional<ffi::TensorView> tile_scheduler_metadata_opt,
                                              ffi::Optional<ffi::TensorView> num_splits_opt,
                                              ffi::Optional<ffi::TensorView> q,
                                              ffi::Optional<int64_t>         bs,
                                              ffi::Optional<ffi::TensorView> topk_length,
                                              ffi::Optional<ffi::TensorView> extra_topk_length,
                                              ffi::Optional<int64_t>         extra_topk) {
  FFI_CHECK(seqlens_k.has_value() || q.has_value(), "seqlens_k or q must be provided");

  auto                 tensor_device = seqlens_k.has_value() ? seqlens_k.value().device() : q.value().device();
  auto                 device_id     = tensor_device.device_id;
  ffi::MUSADeviceGuard device_guard(device_id);

  musaDeviceProp dprops{};
  MATE_MUSA_RUNTIME_CHECK(musaGetDeviceProperties(&dprops, device_id));
  const int mp_count = dprops.multiProcessorCount;
  Arch      arch     = {dprops.major, dprops.minor};
  TVM_FFI_ICHECK(arch.is_mp31() || arch.is_mp32()) << "Only MP31 and MP32 are supported";

  const int batch_size = static_cast<int>(
      bs.has_value() ? bs.value() : (seqlens_k.has_value() ? seqlens_k.value().size(0) : q.value().size(0)));
  const bool      is_sparse_attn    = topk.has_value();
  ffi::TensorView device_cmp_tensor = seqlens_k.has_value() ? seqlens_k.value() : q.value();
  if (extra_topk.has_value()) {
    FFI_CHECK(is_sparse_attn, "extra_topk may only be provided when topk is provided");
    FFI_CHECK(extra_topk.value() >= 0, "extra_topk must be non-negative");
  }

  if (is_sparse_attn) {
    CHECK_HAS_VALUE_WITH_MSG(h_q, "num_heads_q must be provided when topk is provided");
  } else {
    CHECK_HAS_VALUE_WITH_MSG(seqlens_k, "seqlens_k must be provided when topk is not provided");
    CHECK_MUSA(seqlens_k.value());
    CHECK_CONTIGUOUS(seqlens_k.value());
    CHECK_INPUT_TYPE(seqlens_k.value(), dl_int32);
  }
  if (topk_length.has_value()) {
    CHECK_MUSA(topk_length.value());
    CHECK_CONTIGUOUS(topk_length.value());
    CHECK_INPUT_TYPE(topk_length.value(), dl_int32);
    CHECK_DEVICE(topk_length.value(), device_cmp_tensor);
    expect_shape(topk_length.value(), {batch_size}, "topk_length");
  }
  if (extra_topk_length.has_value()) {
    CHECK_MUSA(extra_topk_length.value());
    CHECK_CONTIGUOUS(extra_topk_length.value());
    CHECK_INPUT_TYPE(extra_topk_length.value(), dl_int32);
    CHECK_DEVICE(extra_topk_length.value(), device_cmp_tensor);
    expect_shape(extra_topk_length.value(), {batch_size}, "extra_topk_length");
  }

  const int            tile_m         = is_sparse_attn ? 64 : 128;
  DecodingAttnImplMeta attn_impl_meta = get_attn_impl_meta(arch,
                                                           mp_count,
                                                           static_cast<int>(num_q_tokens_per_head_k),
                                                           static_cast<int>(h_k),
                                                           h_q,
                                                           is_fp8_kvcache,
                                                           is_sparse_attn,
                                                           tile_m);

  ffi::Tensor     tile_scheduler_metadata_storage;
  ffi::Tensor     num_splits_storage;
  ffi::TensorView tile_scheduler_metadata =
      tile_scheduler_metadata_opt.has_value()
          ? tile_scheduler_metadata_opt.value()
          : ffi::TensorView(tile_scheduler_metadata_storage = alloc_tensor(
                                ffi::Shape{attn_impl_meta.num_mp_parts, mate::flash_mla::TileSchedulerMetaDataSize},
                                dl_int32,
                                tensor_device));
  ffi::TensorView num_splits =
      num_splits_opt.has_value()
          ? num_splits_opt.value()
          : ffi::TensorView(num_splits_storage = alloc_tensor(ffi::Shape{batch_size + 1}, dl_int32, tensor_device));

  CHECK_DEVICE(tile_scheduler_metadata, device_cmp_tensor);
  CHECK_DEVICE(num_splits, device_cmp_tensor);
  CHECK_CONTIGUOUS(tile_scheduler_metadata);
  CHECK_CONTIGUOUS(num_splits);
  CHECK_INPUT_TYPE(tile_scheduler_metadata, dl_int32);
  CHECK_INPUT_TYPE(num_splits, dl_int32);
  expect_shape(tile_scheduler_metadata,
               {attn_impl_meta.num_mp_parts, mate::flash_mla::TileSchedulerMetaDataSize},
               "tile_scheduler_metadata");
  expect_shape(num_splits, {batch_size + 1}, "num_splits");

  mate::flash_mla::GetDecodingMetadataParams params = {};
  params.seqlens_k_ptr = seqlens_k.has_value() ? static_cast<int32_t*>(seqlens_k.value().data_ptr()) : nullptr;
  params.tile_scheduler_metadata_ptr = static_cast<int32_t*>(tile_scheduler_metadata.data_ptr());
  params.num_splits_ptr              = static_cast<int32_t*>(num_splits.data_ptr());
  params.topk_length_ptr = topk_length.has_value() ? static_cast<int32_t*>(topk_length.value().data_ptr()) : nullptr;
  params.extra_topk_length_ptr =
      extra_topk_length.has_value() ? static_cast<int32_t*>(extra_topk_length.value().data_ptr()) : nullptr;
  params.batch_size                = batch_size;
  params.block_size_n              = attn_impl_meta.k_block_size;
  params.fixed_overhead_num_blocks = attn_impl_meta.fixed_overhead_num_blocks;
  params.num_mp_parts              = attn_impl_meta.num_mp_parts;
  params.topk                      = is_sparse_attn ? static_cast<int>(topk.value()) : -1;
  params.extra_topk                = extra_topk.has_value() ? static_cast<int>(extra_topk.value()) : -1;
  run_get_mla_metadata_kernel(params,
                              get_stream(seqlens_k.has_value() ? seqlens_k.value().device() : q.value().device()),
                              dprops.sharedMemPerBlockOptin);

  return {tile_scheduler_metadata_storage, num_splits_storage, tile_scheduler_metadata, num_splits};
}

ffi::Array<ffi::Any> get_mla_decoding_metadata(ffi::Optional<ffi::TensorView> seqlens_k,
                                               int64_t                        num_q_tokens_per_head_k,
                                               int64_t                        h_k,
                                               ffi::Optional<int64_t>         h_q,
                                               bool                           is_fp8_kvcache,
                                               ffi::Optional<int64_t>         topk,
                                               ffi::Optional<ffi::Tensor>     tile_scheduler_metadata,
                                               ffi::Optional<ffi::Tensor>     num_splits,
                                               ffi::Optional<ffi::TensorView> q,
                                               ffi::Optional<int64_t>         bs,
                                               ffi::Optional<ffi::TensorView> topk_length,
                                               ffi::Optional<ffi::TensorView> extra_topk_length,
                                               ffi::Optional<int64_t>         extra_topk) {
  TVM_FFI_ICHECK(tile_scheduler_metadata.has_value() == num_splits.has_value())
      << "tile_scheduler_metadata and num_splits must be both provided or both omitted";

  ffi::Optional<ffi::TensorView> tile_scheduler_metadata_view;
  ffi::Optional<ffi::TensorView> num_splits_view;

  if (tile_scheduler_metadata.has_value()) {
    const ffi::Tensor& tile_scheduler_metadata_tensor = tile_scheduler_metadata.value();
    const ffi::Tensor& num_splits_tensor              = num_splits.value();
    tile_scheduler_metadata_view                      = ffi::TensorView(tile_scheduler_metadata_tensor);
    num_splits_view                                   = ffi::TensorView(num_splits_tensor);
  }

  auto result = get_mla_decoding_metadata_impl(seqlens_k,
                                               num_q_tokens_per_head_k,
                                               h_k,
                                               h_q,
                                               is_fp8_kvcache,
                                               topk,
                                               tile_scheduler_metadata_view,
                                               num_splits_view,
                                               q,
                                               bs,
                                               topk_length,
                                               extra_topk_length,
                                               extra_topk);

  ffi::Tensor tile_scheduler_metadata_result =
      tile_scheduler_metadata.has_value() ? tile_scheduler_metadata.value() : result.tile_scheduler_metadata_storage;
  ffi::Tensor num_splits_result = num_splits.has_value() ? num_splits.value() : result.num_splits_storage;

  return ffi::Array<ffi::Any>{ffi::Any(tile_scheduler_metadata_result), ffi::Any(num_splits_result)};
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_mla_decoding_metadata, get_mla_decoding_metadata);
