/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/*!
    \file
    \brief Top-level include for all MUTLASS numeric types.
*/
#pragma once

#include <cstddef>

#include "mutlass/numeric_size.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <std::size_t... Seq>
struct index_sequence;

template <std::size_t N, std::size_t... Next>
struct index_sequence_helper : index_sequence_helper<N - 1, N - 1, Next...> {};

template <std::size_t... Next>
struct index_sequence_helper<0, 0, Next...> {
  using type = index_sequence<0, Next...>;
};

template <std::size_t N>
using make_index_sequence = typename index_sequence_helper<N>::type;


/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Get the register type used in kernel
//
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<typename T>
struct get_unpacked_element_type {
  using type = T;
};

} // namespace detail

}  // namespace mutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "mutlass/integer_subbyte.h"
#include "mutlass/half.h"
#include "mutlass/bfloat16.h"
#include "mutlass/tfloat32.h"
#include "mutlass/float8.h"
#include "mutlass/uint128.h"
#include "mutlass/uint256.h"
#include "mutlass/float_subbyte.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {

// Default case - no negative zero
template <typename T>
struct has_negative_zero : platform::false_type {};

// Float types that support negative zero
template <> struct has_negative_zero<mx_float4_t<float_e2m1_t>> : platform::true_type {};
template <> struct has_negative_zero<mx_float6_t<float_e2m3_t>> : platform::true_type {};
template <> struct has_negative_zero<mx_float8_t<float_e4m3_t>> : platform::true_type {};
template <> struct has_negative_zero<mx_float8_t<float_e5m2_t>> : platform::true_type {};
template <> struct has_negative_zero<float_e2m1_t> : platform::true_type {};
template <> struct has_negative_zero<float_e2m3_t> : platform::true_type {};
template <> struct has_negative_zero<float_e4m3_t> : platform::true_type {};
template <> struct has_negative_zero<float_e5m2_t> : platform::true_type {};
template <> struct has_negative_zero<half_t> : platform::true_type {};
template <> struct has_negative_zero<bfloat16_t> : platform::true_type {};
template <> struct has_negative_zero<float> : platform::true_type {};
template <> struct has_negative_zero<double> : platform::true_type {};
template <> struct has_negative_zero<tfloat32_t> : platform::true_type {};

template <typename T>
constexpr bool has_negative_zero_v = has_negative_zero<T>::value;

}  // namespace mutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
