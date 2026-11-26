#pragma once

#include <mute/arch/config.hpp>
#include <mute/arch/mma.hpp>
#include <mute/arch/tce_desc.hpp>
#include <mute/numeric/numeric_types.hpp>

namespace mute {

namespace MP32 {

// TCE Src Operand enum
enum class TceABDtype {
  U8U8     = 0x00,
  S8S8     = 0x01,
  F16F16   = 0x02,
  BF16BF16 = 0x03,
  TF32TF32 = 0x04,
  E4M3E4M3 = 0x05,
  E5M2E5M2 = 0x06,
  E4M3E5M2 = 0x0B,
  E5M2E4M3 = 0x0C,
  E2M1E2M1 = 0x11,
  E2M3E2M3 = 0x12,
  E2M3E3M2 = 0x13,
  E3M2E2M3 = 0x14,
  E3M2E3M2 = 0x15,
};

namespace detail {

template <class ElementA, class ElementB>
struct deduce_tce_dtype_enum {
  static_assert(mute::dependent_false<ElementA, ElementB>, "Unsupported TCE DType combination.");
};

template <>
struct deduce_tce_dtype_enum<uint8_t, uint8_t> {
  static constexpr TceABDtype value = TceABDtype::U8U8;
};

template <>
struct deduce_tce_dtype_enum<int8_t, int8_t> {
  static constexpr TceABDtype value = TceABDtype::S8S8;
};

template <>
struct deduce_tce_dtype_enum<half_t, half_t> {
  static constexpr TceABDtype value = TceABDtype::F16F16;
};

template <>
struct deduce_tce_dtype_enum<bfloat16_t, bfloat16_t> {
  static constexpr TceABDtype value = TceABDtype::BF16BF16;
};

template <>
struct deduce_tce_dtype_enum<tfloat32_t, tfloat32_t> {
  static constexpr TceABDtype value = TceABDtype::TF32TF32;
};

template <>
struct deduce_tce_dtype_enum<float_e4m3_t, float_e4m3_t> {
  static constexpr TceABDtype value = TceABDtype::E4M3E4M3;
};

template <>
struct deduce_tce_dtype_enum<float_e5m2_t, float_e5m2_t> {
  static constexpr TceABDtype value = TceABDtype::E5M2E5M2;
};

template <>
struct deduce_tce_dtype_enum<float_e4m3_t, float_e5m2_t> {
  static constexpr TceABDtype value = TceABDtype::E4M3E5M2;
};

template <>
struct deduce_tce_dtype_enum<float_e5m2_t, float_e4m3_t> {
  static constexpr TceABDtype value = TceABDtype::E5M2E4M3;
};

template <>
struct deduce_tce_dtype_enum<float_e2m1_t, float_e2m1_t> {
  static constexpr TceABDtype value = TceABDtype::E2M1E2M1;
};

template <>
struct deduce_tce_dtype_enum<float_e2m3_unpack8bits_t,
                             float_e2m3_unpack8bits_t> {
  static constexpr TceABDtype value = TceABDtype::E2M3E2M3;
};

template <>
struct deduce_tce_dtype_enum<float_e2m3_unpack8bits_t,
                             float_e3m2_unpack8bits_t> {
  static constexpr TceABDtype value = TceABDtype::E2M3E3M2;
};

template <>
struct deduce_tce_dtype_enum<float_e3m2_unpack8bits_t,
                             float_e2m3_unpack8bits_t> {
  static constexpr TceABDtype value = TceABDtype::E3M2E2M3;
};

template <>
struct deduce_tce_dtype_enum<float_e3m2_unpack8bits_t,
                             float_e3m2_unpack8bits_t> {
  static constexpr TceABDtype value = TceABDtype::E3M2E3M2;
};

template <class ElementA, class ElementB>
static constexpr TceABDtype deduce_tce_dtype_enum_v =
    deduce_tce_dtype_enum<ElementA, ElementB>::value;

template <class ElementA, class ElementB,
          TceABDtype = deduce_tce_dtype_enum_v<ElementA, ElementB>>
struct deduce_tce_acc_type {
  using type = float;
};

template <>
struct deduce_tce_acc_type<uint8_t, uint8_t, TceABDtype::U8U8> {
  using type = uint32_t;
};

template <>
struct deduce_tce_acc_type<int8_t, int8_t, TceABDtype::S8S8> {
  using type = int32_t;
};

template <class ElementA, class ElementB>
using deduce_tce_acc_type_t =
    typename deduce_tce_acc_type<ElementA, ElementB>::type;

template <class ElementA, class ElementB,
          TceABDtype = deduce_tce_dtype_enum_v<ElementA, ElementB>>
struct deduce_tce_bits {
  static constexpr int bits_a = mute::sizeof_bits_v<ElementA>;
  static constexpr int bits_b = mute::sizeof_bits_v<ElementB>;

  static_assert(bits_a == bits_b, "MP32 TCE operands must have the same bit width.");

  static constexpr int value = bits_a <= 4 ? 4 : bits_a <= 8 ? 8 : bits_a;
  static_assert((value & (value - 1)) == 0, "TCE bits must be a power of two.");
};

template <class ElementA, class ElementB>
static constexpr int deduce_tce_bits_v =
    deduce_tce_bits<ElementA, ElementB>::value;

} // namespace detail

} // namespace MP32

} // namespace mute

#include <mute/arch/mma_mp32_desc.hpp>
#include <mute/arch/mma_mp32_sqmma.hpp>
