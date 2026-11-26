/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#if !defined(__MUSACC_RTC__)
#include <musa.h>
#include <cstdlib>
#endif

#include <mute/numeric/integral_constant.hpp>
#include <mute/atom/copy_traits.hpp>
#include <mute/atom/copy_traits_mp31_tme.hpp>
#include <mute/atom/copy_atom.hpp>
#include <mutlass/fast_math.h>

namespace mute
{

template <
  class TmeSwizzle_,
  class TmeIm2ColBlockDim_,
  class TmeIm2ColConvParam_,
  class TmeIm2ColOutputDim_
>
struct AuxTmeIm2ColParams
{
  using TmeSwizzle = TmeSwizzle_;
  static_assert(is_static<TmeSwizzle>::value);

  using TmeIm2ColBlockDim = TmeIm2ColBlockDim_;
  TmeIm2ColBlockDim block_dim_;

  using TmeIm2ColConvParam = TmeIm2ColConvParam_;
  TmeIm2ColConvParam conv_param_;

  using TmeIm2ColOutputDim = TmeIm2ColOutputDim_;
  TmeIm2ColOutputDim output_dim_;
};

template <class CopyOp>
struct TME_LOAD_IM2COL_Unpack
{
  template <
    class... Args,
    class TS, class SLayout,
    class TD, class DLayout
  >
  MUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout>           const& src,
              Tensor<TD,DLayout>                & dst)
  {
    static_assert(is_smem<TD>::value, "MP31_TME_LOAD_IM2COL requires the destination be shared memory.");

    // The coord tensor is mode-aware but always lowered to MUSA's
    // (c, q, p, z, n, s, r, t) intrinsic argument order.
    auto src_coord = flatten(src(Int<0>{}));

    void* dst_ptr = mute::raw_pointer_cast(dst.data());

    // block_dim: (range_c, range_whdn), conv_param: (padding, stride, dilation),
    // output_dim: (Q, P, Z).
    const auto &aux_param  = get<2>(traits.opargs_);
    const auto &block_dim  = aux_param.block_dim_;
    const auto &conv_param = aux_param.conv_param_;
    const auto &output_dim = aux_param.output_dim_;

    return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                 traits.opargs_, make_seq<2>{},
                                 make_tuple(dst_ptr), seq<0>{},
                                 src_coord, tuple_seq<decltype(src_coord)>{},
                                 block_dim, seq<0, 1>{},
                                 make_tuple(conv_param), seq<0>{},
                                 make_tuple(output_dim), seq<0>{});
  }
};

//////////////////////////////////////////////////////////////////////////////
///////////////////////////// TME_IM2COL_LOAD ////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  PrefetchSize          prefetch
>
struct MP31_TME_LOAD_IM2COL_OP : MP31_TME_LOAD_IM2COL_ENTRY<sg, ss, sl, prefetch> {};


// The non-executable MP31_TME_IM2COL_LOAD_ENTRY with tme_desc and no tme_bar
// Use .with(tme_bar) to construct an executable version
template <
  class NumBitsPerTME,
  class AuxParams,
  class TMETensor,
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  PrefetchSize          prefetch
>
struct Copy_Traits<MP31_TME_LOAD_IM2COL_ENTRY<sg, ss, sl, prefetch>,
                   NumBitsPerTME,
                   AuxParams,
                   TMETensor>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTME>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTME>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // MP31_TME_IM2COL_LOAD arguments
  TmeDescriptor tme_desc_;

  AuxParams aux_params_;

  TMETensor tme_tensor_;

  // Return TmeDescriptor
  MUTE_HOST_DEVICE constexpr
  TmeDescriptor const*
  get_tme_descriptor() const {
    return &tme_desc_;
  }

  // Construct an executable MP31_TME_LOAD_IM2COL with tme_bar
  MUTE_HOST_DEVICE constexpr
  Copy_Traits<MP31_TME_LOAD_IM2COL_OP<sg, ss, sl, prefetch>, NumBitsPerTME, AuxParams>
  with(uint32_t tme_bar) const {
    return {{}, {&tme_desc_, tme_bar, aux_params_}};
  }

  // Generate the TME coord tensor
  template <class GShape>
  MUTE_HOST_DEVICE constexpr
  auto
  get_tme_tensor(GShape const&) const {
    return tme_tensor_;
  }

  // Don't try to execute a copy with MP31_TME_LOAD_IM2COL before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  MUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

// The executable MP31_TME_IM2COL_LOAD with tme_desc, tme_im2col_desc, tme_bar & tme_params
template <
  class NumBitsPerTME,
  class AuxParams,
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  PrefetchSize          prefetch
>
struct Copy_Traits<MP31_TME_LOAD_IM2COL_OP<sg, ss, sl, prefetch>, NumBitsPerTME, AuxParams>
     : TME_LOAD_IM2COL_Unpack<MP31_TME_LOAD_IM2COL_OP<sg, ss, sl, prefetch>>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTME>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTME>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // MP31_TME_IM2COL_LOAD arguments
  tuple<
    TmeDescriptor const*,
    uint32_t const&,
    AuxParams const&
  > const opargs_;
};

template <
  class NumBitsPerTME,
  class AuxParams,
  class TMETensor
>
struct Copy_Traits<MP31_TME_IM2COL_PREFETCH, NumBitsPerTME, AuxParams, TMETensor>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTME>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTME>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  tuple<TmeDescriptor const*, AuxParams const&> const opargs_;

  template <class... CopyArgs>
  MUTE_HOST_DEVICE constexpr
  Copy_Traits(Copy_Traits<CopyArgs...> const& traits)
    : opargs_({&traits.tme_desc_, traits.aux_params_}) {}

  template <class TS, class SLayout,
            class TD, class DLayout>
  MUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits                  const& traits,
              Tensor<TS,SLayout>           const& src,
              Tensor<TD,DLayout>                & dst)
  {
    (void)dst;

    auto src_coord = flatten(src(Int<0>{}));

    const auto &aux_param  = get<1>(traits.opargs_);
    const auto &block_dim  = aux_param.block_dim_;
    const auto &conv_param = aux_param.conv_param_;
    const auto &output_dim = aux_param.output_dim_;

    return detail::explode_tuple(detail::CallCOPY<MP31_TME_IM2COL_PREFETCH>{},
                                 traits.opargs_, make_seq<1>{},
                                 src_coord, tuple_seq<decltype(src_coord)>{},
                                 block_dim, seq<0, 1>{},
                                 make_tuple(conv_param), seq<0>{},
                                 make_tuple(output_dim), seq<0>{});
  }
};

namespace detail {

/// @brief Creates the descriptor payloads used by MP31 TME IM2COL load.
///
/// MUSA TME IM2COL uses a regular tensor descriptor plus a compact
/// `TmeIm2ColParam` carrying `(padding, stride, dilation)` for each spatial
/// mode. The caller passes already-lowered MUSA residual parameters.
///
/// @param tensor_cwhdn Global tensor in (C,W,H,D,N) order, already recast to the TME access type.
/// @param conv_padding_whd Non-negative MUSA im2col lower padding in [W,H,D] order.
/// @param conv_stride_whd MUSA im2col traversal stride in [W,H,D] order.
/// @param conv_dilation_srt MUSA im2col dilation magnitude for filter coordinates in [S,R,T] order.
/// @return Tuple of `(TmeDescriptor, encoded_im2col_param)`.
template <
  class EngineA, class LayoutA,
  class ConvPadding,
  class ConvStride,
  class ConvDilation
>
MUTE_HOST_RTC
auto
make_tme_im2col_copy_desc(Tensor<EngineA, LayoutA> const& tensor_cwhdn,
                          ConvPadding             const& conv_padding_whd,
                          ConvStride              const& conv_stride_whd,
                          ConvDilation            const& conv_dilation_srt)
{
  using value_type = typename EngineA::value_type;

  void* gmem_address = (void*)raw_pointer_cast(tensor_cwhdn.data());

  constexpr uint32_t num_modes = LayoutA::rank;

  mute::array<uint64_t, 5> gmem_prob_shape = {1, 1, 1, 1, 1};
  for_each(make_seq<num_modes>{}, [&](auto i) {
    gmem_prob_shape[i] = static_cast<uint64_t>(shape<i>(tensor_cwhdn));
  });

  // Gmem strides are byte strides of the recast tensor.
  mute::array<uint64_t, 5> gmem_prob_stride = {0, 0, 0, 0, 0};
  for_each(make_seq<num_modes>{}, [&](auto i) {
    gmem_prob_stride[i] = static_cast<uint64_t>(sizeof(value_type) * stride<i>(tensor_cwhdn));
  });

  TmeDescriptor tme_desc{};

  MUtensorDescriptorDataType   tme_format     = TME::to_MUtensorDescriptorDataType<value_type>();
  MUtensorDescriptorInterleave tme_interleave = MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE;
  uint64_t                     tme_oobFill    = 0;

  MUresult result = MUTLASS_MUSA_DRIVER_WRAPPER_CALL(muTensorDescriptorEncode)(
      &tme_desc,
      tme_format,
      num_modes,
      gmem_address,
      gmem_prob_shape.data(),
      gmem_prob_stride.data() + 1,
      tme_interleave,
      tme_oobFill);

  if (result != MUSA_SUCCESS) {
    std::cerr << "TME Desc Addr:    " << &tme_desc
              << "\nformat          " << tme_format
              << "\nnum_modes       " << num_modes
              << "\ngmem_address    " << gmem_address
              << "\nglobalDim       " << gmem_prob_shape
              << "\nglobalStrides   " << gmem_prob_stride
              << "\ninterleave      " << tme_interleave
              << "\noobFill         " << tme_oobFill << std::endl;

    std::cerr << "Error: Failed to initialize the TME descriptor: "
              << result << std::endl;
    assert(false);
  }

  TmeIm2ColParam tme_im2col_param{};

  constexpr int            num_spatial_modes    = num_modes - 2;
  mute::array<uint32_t, 3> im2col_lower_padding = {0, 0, 0};
  mute::array<uint32_t, 3> im2col_stride        = {0, 0, 0};
  mute::array<uint32_t, 3> im2col_dilation      = {0, 0, 0};

  for_each(make_seq<num_spatial_modes>{}, [&](auto i) {
    im2col_lower_padding[i] = static_cast<uint32_t>(get<i>(conv_padding_whd));
    im2col_stride[i]        = static_cast<uint32_t>(get<i>(conv_stride_whd));
    im2col_dilation[i]      = static_cast<uint32_t>(get<i>(conv_dilation_srt));
  });

  result = MUTLASS_MUSA_DRIVER_WRAPPER_CALL(muTensorIm2colConvParamEncode)(
      &tme_im2col_param,
      num_spatial_modes,
      im2col_lower_padding.data(),
      im2col_stride.data(),
      im2col_dilation.data());

  if (result != MUSA_SUCCESS) {
    std::cerr << "TME im2col Param Addr:    " << &tme_im2col_param
              << "\nnum_spatial_modes       " << num_spatial_modes
              << "\nlower_padding           " << im2col_lower_padding
              << "\nstride                  " << im2col_stride
              << "\ndilation                " << im2col_dilation << std::endl;
    std::cerr << "Error: Failed to initialize the TME Im2Col Parameters: "
              << result << std::endl;
    assert(false);
  }

  mute::v3i32_t conv_param = {static_cast<int32_t>(tme_im2col_param.params[0]),
                              static_cast<int32_t>(tme_im2col_param.params[1]),
                              static_cast<int32_t>(tme_im2col_param.params[2])};

  return mute::make_tuple(tme_desc, conv_param);
}

/// @brief Creates the descriptor and coordinate tensor used by MP31 TME IM2COL load.
///
/// The atom builder first derives the TME tiling, then this helper builds both
/// descriptor payloads and the logical im2col coordinate tensor for that tiling.
///
/// @param tensor_cwhdn Global tensor in (C,W,H,D,N) order.
/// @param tme_layout_trunc TME tile layout after SMEM/GMEM basis truncation.
/// @return Tuple of `(TmeDescriptor, encoded_im2col_param, tme_tensor, block_dim, output_dim)`.
template <
  class EngineA, class LayoutA,
  class TmeLayout,
  class LowerCorner,
  class UpperCorner,
  class LowerPadding,
  class UpperPadding,
  class TraversalStride,
  class LowerSRT,
  class StrideSRT
>
MUTE_HOST_RTC
auto
make_tme_atom_im2col_desc(Tensor<EngineA, LayoutA> const& tensor_cwhdn,
                          TmeLayout               const& tme_layout_trunc,
                          LowerCorner             const& lower_corner_whd,
                          UpperCorner             const& upper_corner_whd,
                          LowerPadding            const& lower_padding_whd,
                          UpperPadding            const& upper_padding_whd,
                          TraversalStride         const& stride_whd,
                          LowerSRT                const& lower_srt,
                          StrideSRT               const& stride_srt)
{
  constexpr int num_total_modes = LayoutA::rank;
  constexpr int num_spatial_modes = num_total_modes - 2;
  static_assert(num_spatial_modes >= 1 && num_spatial_modes <= 3, "IM2COL supports 1D, 2D, and 3D conv tensors.");

  auto shape_cwhdn = shape(tensor_cwhdn);

  //
  // Lower logical IM2COL bounding-box parameters to MUSA TME residual form.
  //
  // Logical IM2COL coordinates:
  //   coord = out * stride_whd + lower_corner_whd + lower_srt + i * stride_srt
  //
  // MUSA intrinsic pseudo-code:
  //   coord = out * conv_stride - conv_padding + filter_coord * conv_dilation
  //
  // For each spatial mode:
  //   dilation     = abs(stride_srt)
  //   shift        = lower_corner > 0 ? ceil_div(lower_corner, dilation) : 0
  //   conv_padding = shift * dilation - lower_corner          // always non-negative
  //   filter_coord = lower_srt / dilation + shift + i * sign(stride_srt)
  //
  // This maps the logical bounding box to MUSA's unsigned `conv_param.padding`
  // and u8 packed filter-coordinate ABI.
  auto conv_padding_whd = mute::transform(mute::make_seq<num_spatial_modes>{}, [&](auto i) {
    auto lower_corner = get<i>(lower_corner_whd);
    auto dilation = abs(get<i>(stride_srt));
    auto shift = conditional_return(lower_corner > Int<0>{},
                                    ceil_div(lower_corner, dilation),
                                    Int<0>{});
    return shift * dilation - lower_corner;
  });

  auto conv_dilation_srt = mute::transform(mute::make_seq<num_spatial_modes>{}, [&](auto i) {
    return abs(get<i>(stride_srt));
  });

  auto filter_base_srt = mute::transform(mute::make_seq<num_spatial_modes>{}, [&](auto i) {
    auto lower_corner = get<i>(lower_corner_whd);
    auto dilation = abs(get<i>(stride_srt));
    auto shift = conditional_return(lower_corner > Int<0>{},
                                    ceil_div(lower_corner, dilation),
                                    Int<0>{});
    return get<i>(lower_srt) / dilation + shift;
  });

  auto filter_stride_srt = mute::transform(mute::make_seq<num_spatial_modes>{}, [&](auto i) {
    return conditional_return(get<i>(stride_srt) > Int<0>{}, Int<1>{}, Int<-1>{});
  });

#if defined(MUTLASS_TME_IM2COL_DEBUG)
  print("lower_corner_whd : "); print(lower_corner_whd); print("\n");
  print("upper_corner_whd : "); print(upper_corner_whd); print("\n");
  print("lower_padding_whd: "); print(lower_padding_whd); print("\n");
  print("upper_padding_whd: "); print(upper_padding_whd); print("\n");
  print("stride_whd       : "); print(stride_whd); print("\n");
  print("lower_srt        : "); print(lower_srt); print("\n");
  print("stride_srt       : "); print(stride_srt); print("\n");
  print("conv_padding_whd : "); print(conv_padding_whd); print("\n");
  print("conv_dilation_srt: "); print(conv_dilation_srt); print("\n");
  print("filter_base_srt  : "); print(filter_base_srt); print("\n");
  print("filter_stride_srt: "); print(filter_stride_srt); print("\n");
#endif

  auto [tme_desc, tme_im2col_param] = make_tme_im2col_copy_desc(tensor_cwhdn,
                                                                 conv_padding_whd,
                                                                 stride_whd,
                                                                 conv_dilation_srt);

  //
  // Calculate GEMM shapes and linearized shapes based on TME layout tiling.
  //

  // Compute [w, h, d, n]:
  // q/p/z = (w/h/d + upper_corner_whd - lower_corner_whd - 1) / stride_whd + 1
  auto gemm_mn_ = mute::transform(mute::make_seq<num_spatial_modes>{}, [&](auto i) {
    return (get<i+1>(shape_cwhdn)
            + get<i>(upper_corner_whd)
            - get<i>(lower_corner_whd)
            - Int<1>{}) / get<i>(stride_whd) + Int<1>{};
  });
  auto gemm_mn = append(gemm_mn_, get<num_spatial_modes+1>(shape_cwhdn));

  // Compute [c, s, r, t]:
  // fprop/wgrad use the far padding side; dgrad uses the near padding side
  // with negative stride_srt.
  auto gemm_k_ = mute::transform(mute::make_seq<num_spatial_modes>{}, [&](auto i) {
    auto padding_size = conditional_return(get<i>(stride_srt) > Int<0>{},
                                           get<i>(upper_padding_whd) - get<i>(upper_corner_whd),
                                           get<i>(lower_corner_whd)  - get<i>(lower_padding_whd));
    return Int<1>{} + padding_size / get<i>(stride_srt);
  });
  auto gemm_k = prepend(gemm_k_, get<0>(shape_cwhdn));

  // For fprop/dgrad kernels, gemm shapes are ((q,p,z,n), (c,s,r,t)).
  // For wgrad kernels, the caller's cta_v_map swaps these roles.
  // Dynamic Q/P/Z/N extents need repeated divmod when the linear GEMM
  // coordinate is decoded back to output-space coordinates; static MN leaves
  // remain compile-time constants.
  auto gemm_mn_divmod = transform_leaf(gemm_mn, [](auto s) {
    return conditional_return(is_static<decltype(s)>{}, s, mutlass::FastDivmod(static_cast<int>(s)));
  });

  auto gemm_shapes_ = make_shape(gemm_mn_divmod, gemm_k);
  auto gemm_shape = make_shape(basis_get(stride<1>(tme_layout_trunc), gemm_shapes_),
                               basis_get(stride<0>(tme_layout_trunc), gemm_shapes_));

  auto linear_shape_common = make_shape(size(gemm_mn), gemm_k);
  auto linear_shape = make_shape(basis_get(stride<1>(tme_layout_trunc), linear_shape_common),
                                 basis_get(stride<0>(tme_layout_trunc), linear_shape_common));

  //
  // Calculate global-memory basis strides based on TME layout tiling.
  //

  auto tme_basis_scale = make_shape(_1{},
                                    repeat_like(stride_whd, _1{}),
                                    _1{},
                                    filter_stride_srt);
  auto tme_basis = elem_scale(tme_basis_scale, make_basis_like(tme_basis_scale));

  auto gbasis_strides_common = make_stride(append(get<1>(tme_basis), get<2>(tme_basis)),
                                           prepend(get<3>(tme_basis), get<0>(tme_basis)));
  auto gbasis_strides = make_stride(basis_get(stride<1>(tme_layout_trunc), gbasis_strides_common),
                                    basis_get(stride<0>(tme_layout_trunc), gbasis_strides_common));

  //
  // Create TME coordinate tensor.
  //

  auto lower_corner = make_arithmetic_tuple(_0{},
                                            repeat_like(stride_whd, _0{}),
                                            _0{},
                                            filter_base_srt);
  auto tensor_multimode = make_tensor(ArithmeticTupleIterator(lower_corner),
                                      gemm_shape,
                                      gbasis_strides);

  auto tensor_linear = make_identity_tensor(linear_shape);
  auto tme_tensor = make_tensor(tensor_multimode.data(), composition(tensor_multimode.layout(),
                                                                     tensor_linear(_0{}),
                                                                     tensor_linear.layout()));

#if defined(MUTLASS_TME_IM2COL_DEBUG)
  print("gemm_mn_        : "); print(gemm_mn_); print("\n");
  print("gemm_k_         : "); print(gemm_k_); print("\n");
  print("gemm_shape      : "); print(gemm_shape); print("\n");
  print("linear_shape    : "); print(linear_shape); print("\n");
  print("tme_basis       : "); print(tme_basis); print("\n");
  print("gbasis_strides  : "); print(gbasis_strides); print("\n");
  print("tme_tensor      : "); print(tme_tensor); print("\n");
#endif

  auto range_c    = size<0>(tme_layout_trunc);
  auto range_whdn = size<1>(tme_layout_trunc);
  auto blk_dim    = make_shape(range_c, range_whdn);

  // MUSA IM2COL intrinsics overload output_dim by spatial rank:
  // 1D -> int32_t(Q), 2D -> v2i32_t(Q,P), 3D -> v3i32_t(Q,P,Z).
  auto output_dim = [&] {
    if constexpr (num_spatial_modes == 3) {
      return mute::v3i32_t{static_cast<int32_t>(size<0>(gemm_mn_)),
                           static_cast<int32_t>(size<1>(gemm_mn_)),
                           static_cast<int32_t>(size<2>(gemm_mn_))};
    }
    else if constexpr (num_spatial_modes == 2) {
      return mute::v2i32_t{static_cast<int32_t>(size<0>(gemm_mn_)),
                           static_cast<int32_t>(size<1>(gemm_mn_))};
    }
    else {
      return static_cast<int32_t>(size<0>(gemm_mn_));
    }
  }();

#if defined(MUTLASS_TME_IM2COL_DEBUG)
  print("blk_dim       : "); print(blk_dim); print("\n");
  print("output_dim    : ");
  if constexpr (num_spatial_modes == 3) {
    print(make_tuple(output_dim[0], output_dim[1], output_dim[2]));
  }
  else if constexpr (num_spatial_modes == 2) {
    print(make_tuple(output_dim[0], output_dim[1]));
  }
  else {
    print(output_dim);
  }
  print("\n");
#endif

  return mute::make_tuple(tme_desc, tme_im2col_param, tme_tensor, blk_dim, output_dim);
}

template <
  class CopyOp,
  class GEngine, class GLayout,
  class SLayout,
  class VShape, class VStride,
  class LowerCorner,
  class UpperCorner,
  class LowerPadding,
  class UpperPadding,
  class TraversalStride,
  class LowerSRT,
  class StrideSRT
>
MUTE_HOST_RTC
auto
make_tme_atom_im2col(CopyOp,
                     Tensor<GEngine,GLayout> const& gtensor,
                     SLayout                 const& slayout,
                     Layout<VShape,VStride>  const& cta_v_map,
                     LowerCorner             const& lower_corner_whd,
                     UpperCorner             const& upper_corner_whd,
                     LowerPadding            const& lower_padding_whd,
                     UpperPadding            const& upper_padding_whd,
                     TraversalStride         const& stride_whd,
                     LowerSRT                const& lower_srt,
                     StrideSRT               const& stride_srt)
{
  // gtensor shape: ((w, h, d, n), c). Spatial rank is 1, 2, or 3.
  constexpr int num_spatial_modes = rank<0>(GLayout{}) - 1;
  static_assert(num_spatial_modes >= 1 && num_spatial_modes <= 3, "IM2COL supports 1D, 2D, and 3D conv tensors.");

  //
  // TME parameter checking
  //

  MUTE_STATIC_ASSERT_V(product_each(shape(slayout)) == product_each(shape(cta_v_map)),
    "TME requires CTA_Tile and SLayout top-level shape equivalence.");

  //
  // TME slayout manipulation
  //

  // Invert SMEM to find the largest contiguous vector, then map that vector
  // back to the global-memory logical modes covered by the CTA tile.
  auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));
  auto sidx_to_gmode = coalesce(composition(cta_v_map, inv_smem_layout));

#if defined(MUTLASS_TME_IM2COL_DEBUG)
  print("g_layout         : "); print(gtensor.layout()); print("\n");
  print("s_layout         : "); print(slayout); print("\n");
  print("cta_v_map        : "); print(cta_v_map); print("\n");
  print("inv_smem         : "); print(inv_smem_layout); print("\n");
  print("sidx_to_gmode    : "); print(sidx_to_gmode); print("\n");
#endif

  //
  // TME gtensor manipulation
  //

  // Generate a TupleBasis for gtensor, then tile its modes with
  // cta_v_map o inv_smem_layout. TME must not start in the middle of a
  // gmem mode, so truncate the layout at the first partially-covered mode.
  auto glayout_basis = make_identity_layout(product_each(shape(gtensor)));
  auto tme_layout_full = flatten(composition(glayout_basis, sidx_to_gmode));

  auto smem_rank = find_if(stride(tme_layout_full), [](auto e) {
    [[maybe_unused]] auto v = basis_value(e);
    return not is_constant<1,decltype(v)>{};
  });

  static_assert(smem_rank >= 2, "IM2COL expects at least 2 modes of the smem to vectorize with gmem.");

  constexpr int smem_tme_rank = mute::min(int(smem_rank), 2);
  auto tme_layout_trunc = take<0,smem_tme_rank>(tme_layout_full);

#if defined(MUTLASS_TME_IM2COL_DEBUG)
  print("glayout_basis    : "); print(glayout_basis); print("\n");
  print("tme_layout_full  : "); print(tme_layout_full); print("\n");
  print("tme_layout_trunc : "); print(tme_layout_trunc); print("\n");
#endif

  auto layout_cwhdn = flatten(make_layout(
      make_layout(basis_get(stride<0>(tme_layout_trunc), gtensor.shape()),
                  basis_get(stride<0>(tme_layout_trunc), gtensor.stride())),
      make_layout(basis_get(stride<1>(tme_layout_trunc), gtensor.shape()),
                  basis_get(stride<1>(tme_layout_trunc), gtensor.stride()))));
  Tensor tensor_cwhdn = make_tensor(gtensor.data(), layout_cwhdn);

  auto [tme_desc, tme_im2col_param, tme_tensor, blk_dim, output_dim] =
    make_tme_atom_im2col_desc(tensor_cwhdn,
                              tme_layout_trunc,
                              lower_corner_whd,
                              upper_corner_whd,
                              lower_padding_whd,
                              upper_padding_whd,
                              stride_whd,
                              lower_srt,
                              stride_srt);

  using value_type = typename GEngine::value_type;
  constexpr uint64_t num_bits_per_tme = size(tme_layout_trunc) * sizeof_bits_v<value_type>;

  constexpr auto smem_swizzle_attr      = get_tme_swizzle_attributes(decltype(get_swizzle_portion(slayout)){});
  constexpr auto tme_SwizzleGranularity = get<0>(smem_swizzle_attr);
  constexpr auto tme_SwizzleStride      = get<1>(smem_swizzle_attr);
  constexpr auto tme_SwizzleSwizzleLine = get<2>(smem_swizzle_attr);
  constexpr auto tme_L2Prefetch         = PrefetchSize::B128;

#if defined(MUTLASS_TME_IM2COL_DEBUG)
  print("num_bits      : "); print(num_bits_per_tme); print("\n");
  print("blk_dim       : "); print(blk_dim); print("\n");
  print("output_dim    : ");
  if constexpr (num_spatial_modes == 3) {
    print(make_tuple(output_dim[0], output_dim[1], output_dim[2]));
  }
  else if constexpr (num_spatial_modes == 2) {
    print(make_tuple(output_dim[0], output_dim[1]));
  }
  else {
    print(output_dim);
  }
  print("\n");
#endif

  //
  // Construct the Copy_Traits and return the Copy_Atom.
  //

  using CopyOpEntry = MP31_TME_LOAD_IM2COL_ENTRY<tme_SwizzleGranularity,
                                                 tme_SwizzleStride,
                                                 tme_SwizzleSwizzleLine,
                                                 tme_L2Prefetch>;
  using AuxParam = AuxTmeIm2ColParams<decltype(get_swizzle_portion(slayout)),
                                      decltype(blk_dim),
                                      decltype(tme_im2col_param),
                                      decltype(output_dim)>;
  using Traits = Copy_Traits<CopyOpEntry,
                             mute::C<num_bits_per_tme>,
                             AuxParam,
                             decltype(tme_tensor)>;
  using Atom = Copy_Atom<Traits, value_type>;

  AuxParam aux_param{blk_dim, tme_im2col_param, output_dim};
  Traits tme_traits{tme_desc, aux_param, tme_tensor};

  return Atom{tme_traits};
}

template <
  class CopyOp,
  class GEngine, class GLayout,
  class SLayout,
  class VShape, class VStride,
  class LowerCorner,
  class UpperCorner,
  class LowerPadding,
  class UpperPadding,
  class TraversalStride,
  class LowerSRT,
  class StrideSRT
>
MUTE_HOST_RTC
auto
make_tme_copy_im2col(CopyOp                    const& copy_op,
                     Tensor<GEngine, GLayout>  const& tensor_cwhdn,
                     SLayout                   const& slayout,
                     Layout<VShape, VStride>   const& cta_v_map,
                     LowerCorner               const& lower_corner_whd,
                     UpperCorner               const& upper_corner_whd,
                     LowerPadding              const& lower_padding_whd,
                     UpperPadding              const& upper_padding_whd,
                     TraversalStride           const& stride_whd,
                     LowerSRT                  const& lower_srt,
                     StrideSRT                 const& stride_srt)
{
  //
  // Construct the Copy_Atom.
  //

  Copy_Atom atom = make_tme_atom_im2col(copy_op,
                                        tensor_cwhdn,
                                        slayout,
                                        cta_v_map,
                                        lower_corner_whd,
                                        upper_corner_whd,
                                        lower_padding_whd,
                                        upper_padding_whd,
                                        stride_whd,
                                        lower_srt,
                                        stride_srt);

  [[maybe_unused]] auto cta_tiler = product_each(shape(cta_v_map));

  //
  // Construct the TiledCopy.
  //

  auto num_elems_per_tme = size<1>(typename decltype(atom)::RefLayout{}) /
                           static_value<sizeof_bits<typename GEngine::value_type>>();

  auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));
  auto layout_v = composition(inv_smem_layout, num_elems_per_tme);
  auto layout_V = tile_to_shape(make_layout(layout_v), size(cta_v_map));

  auto layout_T = Layout<_1, _0>{};
  [[maybe_unused]] auto layout_TV = make_layout(layout_T, layout_V);

#if defined(MUTLASS_TME_IM2COL_DEBUG)
  print("cta_tiler : "); print(cta_tiler); print("\n");
  print("layout_v  : "); print(layout_v); print("\n");
  print("layout_V  : "); print(layout_V); print("\n");
  print("layout_T  : "); print(layout_T); print("\n");
  print("layout_TV : "); print(layout_TV); print("\n");
#endif

  return TiledCopy<decltype(atom), decltype(layout_TV), decltype(cta_tiler)>{atom};
}

} // namespace detail

template <
  class TmeInternalType = void,
  class CopyOp,
  class GEngine, class GLayout,
  class SLayout,
  class CTATiler,
  class LowerCorner,
  class UpperCorner,
  class LowerPadding,
  class UpperPadding,
  class TraversalStride,
  class LowerSRT,
  class StrideSRT
>
MUTE_HOST_RTC
auto
make_im2col_tme_copy(CopyOp                   const& copy_op,
                     Tensor<GEngine, GLayout> const& gtensor,
                     SLayout                  const& slayout,
                     CTATiler                 const& cta_tiler,
                     LowerCorner              const& lower_corner_whd,
                     UpperCorner              const& upper_corner_whd,
                     LowerPadding             const& lower_padding_whd,
                     UpperPadding             const& upper_padding_whd,
                     TraversalStride          const& stride_whd,
                     LowerSRT                 const& lower_srt,
                     StrideSRT                const& stride_srt)
{
  using TMEType = conditional_t<is_same<void, TmeInternalType>::value,
                                typename GEngine::value_type,
                                TmeInternalType>;

  auto gtensor_tme = recast<TMEType>(gtensor);
  auto cta_v_tile = make_identity_layout(product_each(shape(gtensor_tme))).compose(cta_tiler);

  return detail::make_tme_copy_im2col(copy_op,
                                      gtensor_tme,
                                      slayout,
                                      cta_v_tile,
                                      lower_corner_whd,
                                      upper_corner_whd,
                                      lower_padding_whd,
                                      upper_padding_whd,
                                      stride_whd,
                                      lower_srt,
                                      stride_srt);
}

template <
  class TmeInternalType = void,
  class CopyOp,
  class GEngine, class GLayout,
  class SLayout,
  class LowerCorner,
  class UpperCorner,
  class LowerPadding,
  class UpperPadding,
  class TraversalStride,
  class LowerSRT,
  class StrideSRT
>
MUTE_HOST_RTC
auto
make_im2col_tme_copy(CopyOp                   const& copy_op,
                     Tensor<GEngine, GLayout> const& gtensor,
                     SLayout                  const& slayout,
                     LowerCorner              const& lower_corner_whd,
                     UpperCorner              const& upper_corner_whd,
                     LowerPadding             const& lower_padding_whd,
                     UpperPadding             const& upper_padding_whd,
                     TraversalStride          const& stride_whd,
                     LowerSRT                 const& lower_srt,
                     StrideSRT                const& stride_srt)
{
  return make_im2col_tme_copy<TmeInternalType>(copy_op,
                                               gtensor,
                                               slayout,
                                               product_each(shape(slayout)),
                                               lower_corner_whd,
                                               upper_corner_whd,
                                               lower_padding_whd,
                                               upper_padding_whd,
                                               stride_whd,
                                               lower_srt,
                                               stride_srt);
}

} // namespace mute
