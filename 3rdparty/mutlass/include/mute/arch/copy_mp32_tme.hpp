/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
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

#include <mute/arch/config.hpp>

#include <mute/arch/copy.hpp>
#include <mute/arch/copy_mp31_desc.hpp>
#include <mute/numeric/int.hpp>

#include <type_traits>

namespace mute
{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// MP32 TME tile-load instructions with optional permutation stride
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  TME::CacheHint      inner_hint,
  TME::CacheHint      outer_hint,
  PrefetchSize          prefetch,
  int                PermutationStride = 0
>
struct MP32_TME_LOAD_1D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& dim0)
  {
#if defined(MUTE_ARCH_TME_MP32_ACTIVATED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    __musa_tme_ld_tile_stride_1d(
        bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc,
        dim0, crd0, static_cast<int32_t>(sg), static_cast<int32_t>(ss),
        static_cast<int32_t>(sl), static_cast<int32_t>(prefetch), PermutationStride,
        static_cast<int32_t>(inner_hint), static_cast<int32_t>(outer_hint));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use MP32 TME load without MUTE_ARCH_TME_MP32_ACTIVATED.");
#endif
  }
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  TME::CacheHint      inner_hint,
  TME::CacheHint      outer_hint,
  PrefetchSize          prefetch,
  int                PermutationStride = 0
>
struct MP32_TME_LOAD_2D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& dim0,
       int32_t const& dim1)
  {
#if defined(MUTE_ARCH_TME_MP32_ACTIVATED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    mute::v2i32_t crd {crd0, crd1};
    mute::v2i32_t dim {dim0, dim1};
    __musa_tme_ld_tile_stride_2d(
        bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc,
        dim, crd, static_cast<int32_t>(sg), static_cast<int32_t>(ss),
        static_cast<int32_t>(sl), static_cast<int32_t>(prefetch), PermutationStride,
        static_cast<int32_t>(inner_hint), static_cast<int32_t>(outer_hint));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use MP32 TME load without MUTE_ARCH_TME_MP32_ACTIVATED.");
#endif
  }
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  TME::CacheHint      inner_hint,
  TME::CacheHint      outer_hint,
  PrefetchSize          prefetch,
  int                PermutationStride = 0
>
struct MP32_TME_LOAD_3D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& crd2,
       int32_t const& dim0,
       int32_t const& dim1,
       int32_t const& dim2)
  {
#if defined(MUTE_ARCH_TME_MP32_ACTIVATED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    mute::v3i32_t crd {crd0, crd1, crd2};
    mute::v3i32_t dim {dim0, dim1, dim2};
    __musa_tme_ld_tile_stride_3d(
        bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc,
        dim, crd, static_cast<int32_t>(sg), static_cast<int32_t>(ss),
        static_cast<int32_t>(sl), static_cast<int32_t>(prefetch), PermutationStride,
        static_cast<int32_t>(inner_hint), static_cast<int32_t>(outer_hint));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use MP32 TME load without MUTE_ARCH_TME_MP32_ACTIVATED.");
#endif
  }
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  TME::CacheHint      inner_hint,
  TME::CacheHint      outer_hint,
  PrefetchSize          prefetch,
  int                PermutationStride = 0
>
struct MP32_TME_LOAD_4D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& crd2,
       int32_t const& crd3,
       int32_t const& dim0,
       int32_t const& dim1,
       int32_t const& dim2,
       int32_t const& dim3)
  {
#if defined(MUTE_ARCH_TME_MP32_ACTIVATED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    mute::v4i32_t crd {crd0, crd1, crd2, crd3};
    mute::v4i32_t dim {dim0, dim1, dim2, dim3};
    __musa_tme_ld_tile_stride_4d(
        bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc,
        dim, crd, static_cast<int32_t>(sg), static_cast<int32_t>(ss),
        static_cast<int32_t>(sl), static_cast<int32_t>(prefetch), PermutationStride,
        static_cast<int32_t>(inner_hint), static_cast<int32_t>(outer_hint));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use MP32 TME load without MUTE_ARCH_TME_MP32_ACTIVATED.");
#endif
  }
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  TME::CacheHint      inner_hint,
  TME::CacheHint      outer_hint,
  PrefetchSize          prefetch,
  int                PermutationStride = 0
>
struct MP32_TME_LOAD_5D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& crd2,
       int32_t const& crd3,
       int32_t const& crd4,
       int32_t const& dim0,
       int32_t const& dim1,
       int32_t const& dim2,
       int32_t const& dim3,
       int32_t const& dim4)
  {
#if defined(MUTE_ARCH_TME_MP32_ACTIVATED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    mute::v5i32_t crd {crd0, crd1, crd2, crd3, crd4};
    mute::v5i32_t dim {dim0, dim1, dim2, dim3, dim4};
    __musa_tme_ld_tile_stride_5d(
        bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc,
        dim, crd, static_cast<int32_t>(sg), static_cast<int32_t>(ss),
        static_cast<int32_t>(sl), static_cast<int32_t>(prefetch), PermutationStride,
        static_cast<int32_t>(inner_hint), static_cast<int32_t>(outer_hint));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use MP32 TME load without MUTE_ARCH_TME_MP32_ACTIVATED.");
#endif
  }
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  TME::CacheHint      inner_hint,
  TME::CacheHint      outer_hint,
  PrefetchSize          prefetch,
  int                PermutationStride = 0
>
struct MP32_TME_LOAD_ENTRY
{
  static_assert(PermutationStride >= 0,
                "MP32_TME_LOAD_ENTRY requires a non-negative permutation stride.");

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& dim0)
  {
    return MP32_TME_LOAD_1D<sg, ss, sl, inner_hint, outer_hint, prefetch, PermutationStride>::copy(desc_ptr, bar_id, smem_ptr, crd0, dim0);
  }

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& dim0,
       int32_t const& dim1)
  {
    return MP32_TME_LOAD_2D<sg, ss, sl, inner_hint, outer_hint, prefetch, PermutationStride>::copy(desc_ptr, bar_id, smem_ptr, crd0, crd1, dim0, dim1);
  }

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& crd2,
       int32_t const& dim0,
       int32_t const& dim1,
       int32_t const& dim2)
  {
    return MP32_TME_LOAD_3D<sg, ss, sl, inner_hint, outer_hint, prefetch, PermutationStride>::copy(desc_ptr, bar_id, smem_ptr, crd0, crd1, crd2, dim0, dim1, dim2);
  }

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& crd2,
       int32_t const& crd3,
       int32_t const& dim0,
       int32_t const& dim1,
       int32_t const& dim2,
       int32_t const& dim3)
  {
    return MP32_TME_LOAD_4D<sg, ss, sl, inner_hint, outer_hint, prefetch, PermutationStride>::copy(desc_ptr, bar_id, smem_ptr, crd0, crd1, crd2, crd3, dim0, dim1, dim2, dim3);
  }

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& crd2,
       int32_t const& crd3,
       int32_t const& crd4,
       int32_t const& dim0,
       int32_t const& dim1,
       int32_t const& dim2,
       int32_t const& dim3,
       int32_t const& dim4)
  {
    return MP32_TME_LOAD_5D<sg, ss, sl, inner_hint, outer_hint, prefetch, PermutationStride>::copy(desc_ptr, bar_id, smem_ptr, crd0, crd1, crd2, crd3, crd4, dim0, dim1, dim2, dim3, dim4);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// TME_REDUCE_ADD : Initiates an MP32 TME tile reduce-add from shared memory to global memory
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace TME {

// Semantic data type encoding expected by MP32 TME reduce builtins.
enum class AtomicDType : uint32_t {
  FP16   = 0,
  BF16   = 1,
  INT32  = 2,
  UINT32 = 3,
  FP32   = 4,
  INT64  = 5,
  UINT64 = 6,
  FP64   = 7,
};

} // namespace TME

struct MP32_TME_REDUCE_ADD {};

template <class T>
struct MP32_TME_REDUCE_ADD_ATOMIC
{
  static constexpr bool valid =
      std::is_same<T, half_t>::value ||
      std::is_same<T, bfloat16_t>::value ||
      std::is_same<T, float>::value ||
      std::is_same<T, double>::value ||
      std::is_same<T, int32_t>::value ||
      std::is_same<T, uint32_t>::value ||
      std::is_same<T, int64_t>::value ||
      std::is_same<T, uint64_t>::value;

  static constexpr TME::AtomicDType dtype =
      std::is_same<T, half_t>::value     ? TME::AtomicDType::FP16 :
      std::is_same<T, bfloat16_t>::value ? TME::AtomicDType::BF16 :
      std::is_same<T, int32_t>::value    ? TME::AtomicDType::INT32 :
      std::is_same<T, uint32_t>::value   ? TME::AtomicDType::UINT32 :
      std::is_same<T, float>::value      ? TME::AtomicDType::FP32 :
      std::is_same<T, int64_t>::value    ? TME::AtomicDType::INT64 :
      std::is_same<T, uint64_t>::value   ? TME::AtomicDType::UINT64 :
                                           TME::AtomicDType::FP64;
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  class T
>
struct MP32_TME_REDUCE_ADD_1D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, void const* smem_ptr,
       int32_t const& crd0, int32_t const& dim0)
  {
#if defined(MUTE_ARCH_TME_MP32_ACTIVATED)
    static_assert(MP32_TME_REDUCE_ADD_ATOMIC<T>::valid,
                  "Unsupported MP32 TME reduce-add element type.");
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    __musa_tme_tensor_tile_reduce_1d(
        /* reduce_op = */ 0, static_cast<uint32_t>(MP32_TME_REDUCE_ADD_ATOMIC<T>::dtype),
        make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc,
        dim0, crd0, static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use MP32 TME reduce-add without MUTE_ARCH_TME_MP32_ACTIVATED.");
#endif
  }
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  class T
>
struct MP32_TME_REDUCE_ADD_2D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, void const* smem_ptr,
       int32_t const& crd0, int32_t const& crd1,
       int32_t const& dim0, int32_t const& dim1)
  {
#if defined(MUTE_ARCH_TME_MP32_ACTIVATED)
    static_assert(MP32_TME_REDUCE_ADD_ATOMIC<T>::valid,
                  "Unsupported MP32 TME reduce-add element type.");
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    mute::v2i32_t crd {crd0, crd1};
    mute::v2i32_t dim {dim0, dim1};
    __musa_tme_tensor_tile_reduce_2d(
        /* reduce_op = */ 0, static_cast<uint32_t>(MP32_TME_REDUCE_ADD_ATOMIC<T>::dtype),
        make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc,
        dim, crd, static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use MP32 TME reduce-add without MUTE_ARCH_TME_MP32_ACTIVATED.");
#endif
  }
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  class T
>
struct MP32_TME_REDUCE_ADD_3D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, void const* smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2,
       int32_t const& dim0, int32_t const& dim1, int32_t const& dim2)
  {
#if defined(MUTE_ARCH_TME_MP32_ACTIVATED)
    static_assert(MP32_TME_REDUCE_ADD_ATOMIC<T>::valid,
                  "Unsupported MP32 TME reduce-add element type.");
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    mute::v3i32_t crd {crd0, crd1, crd2};
    mute::v3i32_t dim {dim0, dim1, dim2};
    __musa_tme_tensor_tile_reduce_3d(
        /* reduce_op = */ 0, static_cast<uint32_t>(MP32_TME_REDUCE_ADD_ATOMIC<T>::dtype),
        make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc,
        dim, crd, static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use MP32 TME reduce-add without MUTE_ARCH_TME_MP32_ACTIVATED.");
#endif
  }
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  class T
>
struct MP32_TME_REDUCE_ADD_4D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, void const* smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3,
       int32_t const& dim0, int32_t const& dim1, int32_t const& dim2, int32_t const& dim3)
  {
#if defined(MUTE_ARCH_TME_MP32_ACTIVATED)
    static_assert(MP32_TME_REDUCE_ADD_ATOMIC<T>::valid,
                  "Unsupported MP32 TME reduce-add element type.");
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    mute::v4i32_t crd {crd0, crd1, crd2, crd3};
    mute::v4i32_t dim {dim0, dim1, dim2, dim3};
    __musa_tme_tensor_tile_reduce_4d(
        /* reduce_op = */ 0, static_cast<uint32_t>(MP32_TME_REDUCE_ADD_ATOMIC<T>::dtype),
        make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc,
        dim, crd, static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use MP32 TME reduce-add without MUTE_ARCH_TME_MP32_ACTIVATED.");
#endif
  }
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  class T
>
struct MP32_TME_REDUCE_ADD_5D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, void const* smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2,
       int32_t const& crd3, int32_t const& crd4,
       int32_t const& dim0, int32_t const& dim1, int32_t const& dim2,
       int32_t const& dim3, int32_t const& dim4)
  {
#if defined(MUTE_ARCH_TME_MP32_ACTIVATED)
    static_assert(MP32_TME_REDUCE_ADD_ATOMIC<T>::valid,
                  "Unsupported MP32 TME reduce-add element type.");
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    mute::v5i32_t crd {crd0, crd1, crd2, crd3, crd4};
    mute::v5i32_t dim {dim0, dim1, dim2, dim3, dim4};
    __musa_tme_tensor_tile_reduce_5d(
        /* reduce_op = */ 0, static_cast<uint32_t>(MP32_TME_REDUCE_ADD_ATOMIC<T>::dtype),
        make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc,
        dim, crd, static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use MP32 TME reduce-add without MUTE_ARCH_TME_MP32_ACTIVATED.");
#endif
  }
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  class T
>
struct MP32_TME_REDUCE_ADD_ENTRY
{
  static_assert(MP32_TME_REDUCE_ADD_ATOMIC<T>::valid,
                "Unsupported MP32 TME reduce-add element type.");

  template <class... Args>
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, void const* smem_ptr, Args const&... args)
  {
    constexpr int Rank = sizeof...(Args) / 2;
    static_assert(sizeof...(Args) % 2 == 0 && Rank >= 1 && Rank <= 5,
                  "MP32 TME reduce-add requires matching 1D-5D coordinates and dimensions.");
    if constexpr (Rank == 1) {
      MP32_TME_REDUCE_ADD_1D<sg, ss, sl, T>::copy(desc_ptr, smem_ptr, args...);
    } else if constexpr (Rank == 2) {
      MP32_TME_REDUCE_ADD_2D<sg, ss, sl, T>::copy(desc_ptr, smem_ptr, args...);
    } else if constexpr (Rank == 3) {
      MP32_TME_REDUCE_ADD_3D<sg, ss, sl, T>::copy(desc_ptr, smem_ptr, args...);
    } else if constexpr (Rank == 4) {
      MP32_TME_REDUCE_ADD_4D<sg, ss, sl, T>::copy(desc_ptr, smem_ptr, args...);
    } else {
      MP32_TME_REDUCE_ADD_5D<sg, ss, sl, T>::copy(desc_ptr, smem_ptr, args...);
    }
  }
};

} // namespace mute

////////////////////////////////////////////////////////////////////////////////////////////////////
