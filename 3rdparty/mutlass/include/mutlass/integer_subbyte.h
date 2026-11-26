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
    \brief Defines a class for using integer types smaller than one byte in host or
      device code.
*/

#pragma once

#if defined(__MUSACC_RTC__)
#include <musa/std/cstdint>
#else
#include <cstdint>
#endif

#include "mutlass/mutlass.h"
#include "mutlass/numeric_size.h"
#include "mutlass/platform/platform.h"

namespace mutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int Bits, bool Signed = true>
struct integer_subbyte {
  /// Storage type
  using Storage = uint8_t;

  /// Number of bits
  static_assert(Bits <= 8*sizeof(Storage), "Require a subbyte of bits in integer_subbyte");

  /// External type
  using xint_t = typename platform::conditional<Signed, int, unsigned>::type;

  /// Bitmask for truncation from larger integers
  static constexpr Storage bits_mask_ = Storage(Storage(-1) >> (8 - Bits));
  /// Bitmask for the sign bit
  static constexpr Storage sign_mask_ = Storage((Signed ? 1 : 0) << (Bits - 1));

  //
  // Data members
  //

  Storage storage;

  //
  // Methods
  //

  /// No operation
  integer_subbyte() = default;

  /// Conversion from integer type
  MUTLASS_HOST_DEVICE explicit
  integer_subbyte(int value)
      : storage(reinterpret_cast<Storage const&>(value) & bits_mask_) {}

  MUTLASS_HOST_DEVICE explicit
  integer_subbyte(unsigned value)
      : storage(reinterpret_cast<Storage const&>(value) & bits_mask_) {}

  MUTLASS_HOST_DEVICE explicit
  integer_subbyte(double value) {
    xint_t tmp = static_cast<xint_t>(value);
    storage = reinterpret_cast<Storage const &>(tmp) & bits_mask_;
  }

  /// Convert to int or unsigned
  MUTLASS_HOST_DEVICE
  operator xint_t() const {
    if (sign_mask_ & storage) {  // Sign extend
      return xint_t(storage) | ~xint_t(bits_mask_);
    } else {
      return xint_t(storage);
    }
  }

  /// Equality
  MUTLASS_HOST_DEVICE
  bool operator==(integer_subbyte const& rhs) const {
    return storage == rhs.storage;
  }

  /// Inequality
  MUTLASS_HOST_DEVICE
  bool operator!=(integer_subbyte const& rhs) const {
    return storage != rhs.storage;
  }

  /// Less than or equal
  MUTLASS_HOST_DEVICE
  bool operator<=(integer_subbyte const& rhs) const {
    if ((sign_mask_ & storage) == (sign_mask_ & rhs.storage)) {
      return storage <= rhs.storage;
    }
    return sign_mask_ & storage;
  }

  /// Less than
  MUTLASS_HOST_DEVICE
  bool operator<(integer_subbyte const& rhs) const {
    if ((sign_mask_ & storage) == (sign_mask_ & rhs.storage)) {
      return storage < rhs.storage;
    }
    return sign_mask_ & storage;
  }

  /// Greater than or equal
  MUTLASS_HOST_DEVICE
  bool operator>=(integer_subbyte const& rhs) const {
    return !(*this < rhs);
  }

  /// Greater than
  MUTLASS_HOST_DEVICE
  bool operator>(integer_subbyte const& rhs) const {
    return !(*this <= rhs);
  }

};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int Bits, bool Signed>
MUTLASS_HOST_DEVICE
integer_subbyte<Bits, Signed> conj(integer_subbyte<Bits, Signed> const& x) {
  return x;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// 1-bit Unsigned integer type
using uint1b_t = integer_subbyte<1, false>;

/// 2-bit Integer type
using int2b_t = integer_subbyte<2, true>;

/// 2-bit Unsigned integer type
using uint2b_t = integer_subbyte<2, false>;

/// 4-bit Integer type
using int4b_t = integer_subbyte<4, true>;

/// 4-bit Unsigned integer type
using uint4b_t = integer_subbyte<4, false>;

/// 6-bit Integer type
using int6b_t = integer_subbyte<6, true>;

/// 6-bit Unsigned integer type
using uint6b_t = integer_subbyte<6, false>;

/// 1-bit binary type
using bin1_t = bool;

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int Bits, bool Signed>
struct sizeof_bits<integer_subbyte<Bits,Signed>> {
  static constexpr int value = Bits;
};

/// Defines the size of an element in bits - specialized for bin1_t
template <>
struct sizeof_bits<bin1_t> {
  static constexpr int value = 1;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace platform {

// Specialization for signed integer_subbyte
template<int NumBits>
struct numeric_limits<mutlass::integer_subbyte<NumBits, true>> {
private:
  using value_type = mutlass::integer_subbyte<NumBits, true>;

public:
  MUTLASS_HOST_DEVICE static value_type lowest() noexcept {
    return value_type{-(1 << (NumBits - 1))};
  }

  MUTLASS_HOST_DEVICE static value_type max() noexcept {
    return value_type{(1 << (NumBits - 1)) - 1};
  }

  MUTLASS_HOST_DEVICE static value_type min() noexcept {
    return lowest();
  }

  static constexpr bool is_integer = true;
  static constexpr bool is_signed = true;
  static constexpr bool has_infinity = false;
};

// Specialization for unsigned integer_subbyte
template<int NumBits>
struct numeric_limits<mutlass::integer_subbyte<NumBits, false>> {
private:
  using value_type = mutlass::integer_subbyte<NumBits, false>;

public:
  MUTLASS_HOST_DEVICE static value_type lowest() noexcept {
    return value_type{0u};
  }

  MUTLASS_HOST_DEVICE static value_type max() noexcept {
    return value_type{(1u << NumBits) - 1u};
  }

  MUTLASS_HOST_DEVICE static value_type min() noexcept {
    return lowest();
  }

  static constexpr bool is_integer = true;
  static constexpr bool is_signed = false;
  static constexpr bool has_infinity = false;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace platform
} // namespace mutlass
