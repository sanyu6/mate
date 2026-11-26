#pragma once

#include <mute/config.hpp>

#include <mute/arch/mma.hpp>
#include <mute/arch/mma_mp31_desc.hpp>

namespace mute {

namespace MP32::SQMMA {

using MP31::SQMMA::LeadingStrideType;
using MP31::SQMMA::to_string;

enum class SwizzleGranularityType : uint8_t {
  B16  = 0,
  B32  = 1,
  B64  = 2,
  B128 = 3,
};

enum class SwizzleStrideType : uint8_t {
  B32  = 0,
  B64  = 1,
  B128 = 2,
  B256 = 3,
};

enum class Fp4Type : uint8_t {
  MXF4 = 0,
  MTF4 = 1,
};

MUTE_HOST_DEVICE char const* to_string(SwizzleStrideType const& t) {
  switch (t) {
    case SwizzleStrideType::B32:  return "SS_32B";
    case SwizzleStrideType::B64:  return "SS_64B";
    case SwizzleStrideType::B128: return "SS_128B";
    case SwizzleStrideType::B256: return "SS_256B";
  }
  return nullptr;
}

MUTE_HOST_DEVICE char const* to_string(SwizzleGranularityType const& t) {
  switch (t) {
    case SwizzleGranularityType::B16:  return "SG_16B";
    case SwizzleGranularityType::B32:  return "SG_32B";
    case SwizzleGranularityType::B64:  return "SG_64B";
    case SwizzleGranularityType::B128: return "SG_128B";
  }
  return nullptr;
}

MUTE_HOST_DEVICE char const* to_string(Fp4Type const& t) {
  switch (t) {
    case Fp4Type::MXF4: return "MXF4";
    case Fp4Type::MTF4: return "MTF4";
  }
  return nullptr;
}

#if !defined(__MUSACC_RTC__)
using MP31::SQMMA::operator<<;

MUTE_HOST std::ostream& operator<<(std::ostream& os, SwizzleStrideType const& t) {
  char const* s = to_string(t);
  if (s) {
    std::operator<<(os, s);  // Explicit call to avoid ambiguity
  } else {
    os.setstate(std::ios_base::failbit);
  }
  return os;
}

MUTE_HOST std::ostream& operator<<(std::ostream& os, SwizzleGranularityType const& t) {
  char const* s = to_string(t);
  if (s) {
    std::operator<<(os, s);
  } else {
    os.setstate(std::ios_base::failbit);
  }
  return os;
}

MUTE_HOST std::ostream& operator<<(std::ostream& os, Fp4Type const& t) {
  char const* s = to_string(t);
  if (s) {
    std::operator<<(os, s);  // Explicit call to avoid ambiguity
  } else {
    os.setstate(std::ios_base::failbit);
  }
  return os;
}
#endif

union SqmmaDescriptor
{

  MUTE_HOST_DEVICE constexpr
  SqmmaDescriptor() noexcept : desc_(0) {}
  MUTE_HOST_DEVICE constexpr
  SqmmaDescriptor(uint32_t desc) noexcept : desc_(desc) {}
  MUTE_HOST_DEVICE constexpr
  SqmmaDescriptor(SqmmaDescriptor const& t) noexcept : desc_(t.desc_) {}
  MUTE_HOST_DEVICE constexpr
  SqmmaDescriptor(SqmmaDescriptor && t) noexcept : desc_(t.desc_) {}

  MUTE_HOST_DEVICE constexpr
  SqmmaDescriptor& operator=(SqmmaDescriptor const& t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  MUTE_HOST_DEVICE constexpr
  SqmmaDescriptor& operator=(SqmmaDescriptor && t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  int32_t desc_;

  // Bitfield implementation avoids the need for shifts in assignment
  // All members must share one underlying type: the MSVC ABI does not pack bitfields with
  // mixed underlying types into one storage unit.
  struct BitField {
    // start_address, bit [0, 18)
    uint32_t start_address_ : 18;
    // leading dimension byte offset type, bit [18, 21)
    // 8B:1,16B:2,32B:3,64B:4,128B:5,256B:6
    uint32_t leading_stride_type_ : 3;
    // swizzle type, bit [21, 22)
    // DISABLE:0, ENABLE:1, always ENABLE
    uint32_t swizzle_type_ : 1;
    // swizzle granularity type, bit [22, 24)
    // SG_16B=0,SG_32B=1,SG_64B=2,SG_128B=3(reserved)
    uint32_t swizzle_granularity_type_ : 2;
    // swizzle stride type, bit [24, 26)
    // SS_32B=0,SS_64B=1,SS_128B=2,SS_256B=3
    uint32_t swizzle_stride_type_ : 2;
    uint32_t reserved_                :  5;  // [26:30]
    // FP4 type, bit [31, 32)
    // MXF4:0(UE8M0 scale), MTF4:1(E4M3 scale)
    uint32_t fp4_type_ : 1;
  } bitfield;

  static_assert(sizeof_bits_v<BitField> == 32, "The underlying bits of BitField must be 32.");

  // Decay to a int32_t
  MUTE_HOST_DEVICE constexpr
  operator int32_t() const noexcept { return desc_; }

  MUTE_HOST_DEVICE friend void print(SqmmaDescriptor const& t)
  {
    #if !defined(__MUSACC_RTC__)
    printf("SqmmaDescriptor: 0x%08x\n", t.desc_);
    printf("     start_addr: 0x%05x\n", t.bitfield.start_address_);
    printf(" leading_stride: 0x%01x(%s)\n", t.bitfield.leading_stride_type_,
           to_string(static_cast<SQMMA::LeadingStrideType>(t.bitfield.leading_stride_type_)));
    printf("   swizzle_type: 0x%01x\n", t.bitfield.swizzle_type_);
    printf(" sw_granularity: 0x%01x(%s)\n", t.bitfield.swizzle_granularity_type_,
           to_string(static_cast<SQMMA::SwizzleGranularityType>(t.bitfield.swizzle_granularity_type_)));
    printf("swizzle_stride: 0x%01x(%s)\n", t.bitfield.swizzle_stride_type_,
           to_string(static_cast<SQMMA::SwizzleStrideType>(t.bitfield.swizzle_stride_type_)));
    printf("      fp4_type: 0x%01x(%s)\n", t.bitfield.fp4_type_,
           to_string(static_cast<SQMMA::Fp4Type>(t.bitfield.fp4_type_)));
    #endif
  }

};

} // namespace MP32::SQMMA

} // namespace mute
