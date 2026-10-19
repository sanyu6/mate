#include "attention_combine.hpp"
#include "op_utils.hpp"

void test_mla_combine(ffi::TensorView o_accum,
                      ffi::TensorView lse_accum,
                      ffi::TensorView splits,
                      ffi::TensorView out,
                      ffi::TensorView lse,
                      int             num_mp_parts) {
  ffi::MUSADeviceGuard              guard(out.device().device_id);
  mate::flash_mla::MlaCombineParams params{};
  params.o_ptr                = out.data_ptr();
  params.softmax_lse_ptr      = lse.data_ptr();
  params.oaccum_ptr           = o_accum.data_ptr();
  params.softmax_lseaccum_ptr = lse_accum.data_ptr();
  params.num_splits_ptr       = static_cast<int*>(splits.data_ptr());
  params.o_batch_stride       = out.stride(0);
  params.o_row_stride         = out.stride(1);
  params.o_head_stride        = 512;
  params.max_q_seq_per_hk     = out.size(1);
  params.total_q              = out.size(0);
  params.h_r                  = out.size(1);
  params.h_k                  = 1;
  params.batch_size           = out.size(0);
  params.num_mp_parts         = num_mp_parts;
  if (dtype_equal(out.dtype(), dl_bfloat16)) {
    run_mla_combine_kernel<mutlass::bfloat16_t, false>(params, get_stream(out.device()));
  } else {
    run_mla_combine_kernel<mutlass::half_t, false>(params, get_stream(out.device()));
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_mla_combine, test_mla_combine);
