#pragma once

#include <mute/config.hpp>
#include <mute/arch/mma.hpp>
#include <mute/arch/mma_mp31_desc.hpp>
#include <mute/arch/mma_mp32.hpp>
#include <mute/int_tuple.hpp>

namespace mute {

////////////////////////////////////////////////////////////////////////////////////////////////////
//// MP32 SS-SQMMA
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_32x32_SS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static_assert((K * MP32::detail::deduce_tce_bits_v<a_type, b_type>) % 8 == 0,
                "MP32 SQMMA K bit count must be byte-aligned.");
  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");
  static constexpr bool RequireAKMajor = sizeof_bits_v<a_type> == 4;
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireAKMajor || tnspA == TCE::Major::K,
                "MP32 SQMMA A operand requires K-major layout.");
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m32n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32_32x32_SS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_32x64_SS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static_assert((K * MP32::detail::deduce_tce_bits_v<a_type, b_type>) % 8 == 0,
                "MP32 SQMMA K bit count must be byte-aligned.");
  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");
  static constexpr bool RequireAKMajor = sizeof_bits_v<a_type> == 4;
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireAKMajor || tnspA == TCE::Major::K,
                "MP32 SQMMA A operand requires K-major layout.");
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m32n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32_32x64_SS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_32x128_SS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static_assert((K * MP32::detail::deduce_tce_bits_v<a_type, b_type>) % 8 == 0,
                "MP32 SQMMA K bit count must be byte-aligned.");
  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");
  static constexpr bool RequireAKMajor = sizeof_bits_v<a_type> == 4;
  static constexpr bool RequireBKMajor =
      sizeof_bits_v<b_type> == 4 || mute::is_same_v<b_type, tfloat32_t>;
  static_assert(!RequireAKMajor || tnspA == TCE::Major::K,
                "MP32 SQMMA A operand requires K-major layout.");
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m32n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32_32x128_SS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_64x32_SS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static_assert((K * MP32::detail::deduce_tce_bits_v<a_type, b_type>) % 8 == 0,
                "MP32 SQMMA K bit count must be byte-aligned.");
  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");
  static constexpr bool RequireAKMajor = sizeof_bits_v<a_type> == 4;
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireAKMajor || tnspA == TCE::Major::K,
                "MP32 SQMMA A operand requires K-major layout.");
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m64n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32_64x32_SS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_64x64_SS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static_assert((K * MP32::detail::deduce_tce_bits_v<a_type, b_type>) % 8 == 0,
                "MP32 SQMMA K bit count must be byte-aligned.");
  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");
  static constexpr bool RequireAKMajor = sizeof_bits_v<a_type> == 4;
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireAKMajor || tnspA == TCE::Major::K,
                "MP32 SQMMA A operand requires K-major layout.");
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m64n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32_64x64_SS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_64x128_SS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static_assert((K * MP32::detail::deduce_tce_bits_v<a_type, b_type>) % 8 == 0,
                "MP32 SQMMA K bit count must be byte-aligned.");
  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");
  static constexpr bool RequireAKMajor = sizeof_bits_v<a_type> == 4;
  static constexpr bool RequireBKMajor =
      sizeof_bits_v<b_type> == 4 || mute::is_same_v<b_type, tfloat32_t>;
  static_assert(!RequireAKMajor || tnspA == TCE::Major::K,
                "MP32 SQMMA A operand requires K-major layout.");
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m64n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32_64x128_SS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_128x32_SS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static_assert((K * MP32::detail::deduce_tce_bits_v<a_type, b_type>) % 8 == 0,
                "MP32 SQMMA K bit count must be byte-aligned.");
  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");
  static constexpr bool RequireAKMajor =
      sizeof_bits_v<a_type> == 4 || mute::is_same_v<a_type, tfloat32_t>;
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireAKMajor || tnspA == TCE::Major::K,
                "MP32 SQMMA A operand requires K-major layout.");
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m128n32_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32_128x32_SS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_128x64_SS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static_assert((K * MP32::detail::deduce_tce_bits_v<a_type, b_type>) % 8 == 0,
                "MP32 SQMMA K bit count must be byte-aligned.");
  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");
  static constexpr bool RequireAKMajor =
      sizeof_bits_v<a_type> == 4 || mute::is_same_v<a_type, tfloat32_t>;
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireAKMajor || tnspA == TCE::Major::K,
                "MP32 SQMMA A operand requires K-major layout.");
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m128n64_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32_128x64_SS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_128x128_SS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static_assert((K * MP32::detail::deduce_tce_bits_v<a_type, b_type>) % 8 == 0,
                "MP32 SQMMA K bit count must be byte-aligned.");
  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");
  static constexpr bool RequireAKMajor =
      sizeof_bits_v<a_type> == 4 || mute::is_same_v<a_type, tfloat32_t>;
  static constexpr bool RequireBKMajor =
      sizeof_bits_v<b_type> == 4 || mute::is_same_v<b_type, tfloat32_t>;
  static_assert(!RequireAKMajor || tnspA == TCE::Major::K,
                "MP32 SQMMA A operand requires K-major layout.");
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m128n128_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32_128x128_SS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//// MP32 RS-SQMMA
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_32x32_RS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static_assert((K * MP32::detail::deduce_tce_bits_v<a_type, b_type>) % 8 == 0,
                "MP32 SQMMA K bit count must be byte-aligned.");
  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");
  static_assert(tnspA == TCE::Major::K,
                "MP32 RS-SQMMA A operand requires K-major layout.");
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[MmaKBytes / 16];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  MUTE_HOST_DEVICE static void
  fma(int32_t const* a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m32n32_RS_mma(
      d, a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32_32x32_RS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_32x64_RS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static_assert((K * MP32::detail::deduce_tce_bits_v<a_type, b_type>) % 8 == 0,
                "MP32 SQMMA K bit count must be byte-aligned.");
  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");
  static_assert(tnspA == TCE::Major::K,
                "MP32 RS-SQMMA A operand requires K-major layout.");
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[MmaKBytes / 16];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const* a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m32n64_RS_mma(
      d, a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32_32x64_RS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_32x128_RS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static_assert((K * MP32::detail::deduce_tce_bits_v<a_type, b_type>) % 8 == 0,
                "MP32 SQMMA K bit count must be byte-aligned.");
  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");
  static_assert(tnspA == TCE::Major::K,
                "MP32 RS-SQMMA A operand requires K-major layout.");
  static constexpr bool RequireBKMajor =
      sizeof_bits_v<b_type> == 4 || mute::is_same_v<b_type, tfloat32_t>;
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[MmaKBytes / 16];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const* a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m32n128_RS_mma(
      d, a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32_32x128_RS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_64x32_RS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static_assert((K * MP32::detail::deduce_tce_bits_v<a_type, b_type>) % 8 == 0,
                "MP32 SQMMA K bit count must be byte-aligned.");
  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");
  static_assert(tnspA == TCE::Major::K,
                "MP32 RS-SQMMA A operand requires K-major layout.");
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[MmaKBytes / 8];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  MUTE_HOST_DEVICE static void
  fma(int32_t const* a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m64n32_RS_mma(
      d, a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32_64x32_RS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_64x64_RS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static_assert((K * MP32::detail::deduce_tce_bits_v<a_type, b_type>) % 8 == 0,
                "MP32 SQMMA K bit count must be byte-aligned.");
  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");
  static_assert(tnspA == TCE::Major::K,
                "MP32 RS-SQMMA A operand requires K-major layout.");
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[MmaKBytes / 8];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const* a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m64n64_RS_mma(
      d, a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32_64x64_RS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_64x128_RS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static_assert((K * MP32::detail::deduce_tce_bits_v<a_type, b_type>) % 8 == 0,
                "MP32 SQMMA K bit count must be byte-aligned.");
  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");
  static_assert(tnspA == TCE::Major::K,
                "MP32 RS-SQMMA A operand requires K-major layout.");
  static constexpr bool RequireBKMajor =
      sizeof_bits_v<b_type> == 4 || mute::is_same_v<b_type, tfloat32_t>;
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[MmaKBytes / 8];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const* a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m64n128_RS_mma(
      d, a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32_64x128_RS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_128x32_RS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static_assert((K * MP32::detail::deduce_tce_bits_v<a_type, b_type>) % 8 == 0,
                "MP32 SQMMA K bit count must be byte-aligned.");
  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");
  static_assert(tnspA == TCE::Major::K,
                "MP32 RS-SQMMA A operand requires K-major layout.");
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[MmaKBytes / 4];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  MUTE_HOST_DEVICE static void
  fma(int32_t const* a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m128n32_RS_mma(
      d, a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32_128x32_RS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_128x64_RS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static_assert((K * MP32::detail::deduce_tce_bits_v<a_type, b_type>) % 8 == 0,
                "MP32 SQMMA K bit count must be byte-aligned.");
  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");
  static_assert(tnspA == TCE::Major::K,
                "MP32 RS-SQMMA A operand requires K-major layout.");
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[MmaKBytes / 4];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  MUTE_HOST_DEVICE static void
  fma(int32_t const* a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m128n64_RS_mma(
      d, a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32_128x64_RS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_128x128_RS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static_assert((K * MP32::detail::deduce_tce_bits_v<a_type, b_type>) % 8 == 0,
                "MP32 SQMMA K bit count must be byte-aligned.");
  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");
  static_assert(tnspA == TCE::Major::K,
                "MP32 RS-SQMMA A operand requires K-major layout.");
  static constexpr bool RequireBKMajor =
      sizeof_bits_v<b_type> == 4 || mute::is_same_v<b_type, tfloat32_t>;
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[MmaKBytes / 4];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  MUTE_HOST_DEVICE static void
  fma(int32_t const* a,
      int32_t const& desc_b,
      int32_t      * d,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m128n128_RS_mma(
      d, a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType));
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32_128x128_RS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
//// MP32 BLOCKSCALED SS-SQMMA
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace MP32::BLOCKSCALED {

// Operand and scale types select the concrete MXF8/F6/F4 or MTF4 format.

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_32x32_SS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static constexpr int SFVecSize = 32;
  static_assert(K % SFVecSize == 0, "MP32 block-scaled MMA K must be divisible by SFVecSize.");

  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");

  static constexpr int MmaBits = sizeof_bits_v<a_type>;
  static_assert(MmaBits <= 8,
                "MP32 block-scaled SQMMA only supports operand types up to 8 bits.");
  static_assert(MmaBits == 4 || mute::is_same_v<sf_type, float_ue8m0_t>,
                "MP32 block-scaled SQMMA F6/F8 operands require float_ue8m0_t scale factor.");
  static_assert(MmaBits != 4 ||
                mute::is_same_v<sf_type, float_ue8m0_t> ||
                mute::is_same_v<sf_type, float_ue4m3_t>,
                "MP32 block-scaled SQMMA F4 operands require float_ue8m0_t or float_ue4m3_t scale factor.");

  static constexpr bool RequireAKMajor = sizeof_bits_v<a_type> == 4;
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireAKMajor || tnspA == TCE::Major::K,
                "MP32 SQMMA A operand requires K-major layout.");
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  using SFARegisters = int32_t[mute::ceil_div(32 * (K / SFVecSize), 512)];
  using SFBRegisters = int32_t[mute::ceil_div(32 * (K / SFVecSize), 512)];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      int32_t const* sfa,
      int32_t const* sfb,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m32n32_mxscale_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType), sfa, sfb);
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32::BLOCKSCALED::MP32_32x32_SS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_32x64_SS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static constexpr int SFVecSize = 32;
  static_assert(K % SFVecSize == 0, "MP32 block-scaled MMA K must be divisible by SFVecSize.");

  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");

  static constexpr int MmaBits = sizeof_bits_v<a_type>;
  static_assert(MmaBits <= 8,
                "MP32 block-scaled SQMMA only supports operand types up to 8 bits.");
  static_assert(MmaBits == 4 || mute::is_same_v<sf_type, float_ue8m0_t>,
                "MP32 block-scaled SQMMA F6/F8 operands require float_ue8m0_t scale factor.");
  static_assert(MmaBits != 4 ||
                mute::is_same_v<sf_type, float_ue8m0_t> ||
                mute::is_same_v<sf_type, float_ue4m3_t>,
                "MP32 block-scaled SQMMA F4 operands require float_ue8m0_t or float_ue4m3_t scale factor.");

  static constexpr bool RequireAKMajor = sizeof_bits_v<a_type> == 4;
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireAKMajor || tnspA == TCE::Major::K,
                "MP32 SQMMA A operand requires K-major layout.");
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  using SFARegisters = int32_t[mute::ceil_div(32 * (K / SFVecSize), 512)];
  using SFBRegisters = int32_t[mute::ceil_div(64 * (K / SFVecSize), 512)];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      int32_t const* sfa,
      int32_t const* sfb,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m32n64_mxscale_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType), sfa, sfb);
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32::BLOCKSCALED::MP32_32x64_SS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_32x128_SS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static constexpr int SFVecSize = 32;
  static_assert(K % SFVecSize == 0, "MP32 block-scaled MMA K must be divisible by SFVecSize.");

  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");

  static constexpr int MmaBits = sizeof_bits_v<a_type>;
  static_assert(MmaBits <= 8,
                "MP32 block-scaled SQMMA only supports operand types up to 8 bits.");
  static_assert(MmaBits == 4 || mute::is_same_v<sf_type, float_ue8m0_t>,
                "MP32 block-scaled SQMMA F6/F8 operands require float_ue8m0_t scale factor.");
  static_assert(MmaBits != 4 ||
                mute::is_same_v<sf_type, float_ue8m0_t> ||
                mute::is_same_v<sf_type, float_ue4m3_t>,
                "MP32 block-scaled SQMMA F4 operands require float_ue8m0_t or float_ue4m3_t scale factor.");

  static constexpr bool RequireAKMajor = sizeof_bits_v<a_type> == 4;
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireAKMajor || tnspA == TCE::Major::K,
                "MP32 SQMMA A operand requires K-major layout.");
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  using SFARegisters = int32_t[mute::ceil_div(32 * (K / SFVecSize), 512)];
  using SFBRegisters = int32_t[mute::ceil_div(128 * (K / SFVecSize), 512)];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      int32_t const* sfa,
      int32_t const* sfb,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m32n128_mxscale_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType), sfa, sfb);
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32::BLOCKSCALED::MP32_32x128_SS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_64x32_SS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static constexpr int SFVecSize = 32;
  static_assert(K % SFVecSize == 0, "MP32 block-scaled MMA K must be divisible by SFVecSize.");

  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");

  static constexpr int MmaBits = sizeof_bits_v<a_type>;
  static_assert(MmaBits <= 8,
                "MP32 block-scaled SQMMA only supports operand types up to 8 bits.");
  static_assert(MmaBits == 4 || mute::is_same_v<sf_type, float_ue8m0_t>,
                "MP32 block-scaled SQMMA F6/F8 operands require float_ue8m0_t scale factor.");
  static_assert(MmaBits != 4 ||
                mute::is_same_v<sf_type, float_ue8m0_t> ||
                mute::is_same_v<sf_type, float_ue4m3_t>,
                "MP32 block-scaled SQMMA F4 operands require float_ue8m0_t or float_ue4m3_t scale factor.");

  static constexpr bool RequireAKMajor = sizeof_bits_v<a_type> == 4;
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireAKMajor || tnspA == TCE::Major::K,
                "MP32 SQMMA A operand requires K-major layout.");
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  using SFARegisters = int32_t[mute::ceil_div(64 * (K / SFVecSize), 512)];
  using SFBRegisters = int32_t[mute::ceil_div(32 * (K / SFVecSize), 512)];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      int32_t const* sfa,
      int32_t const* sfb,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m64n32_mxscale_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType), sfa, sfb);
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32::BLOCKSCALED::MP32_64x32_SS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_64x64_SS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static constexpr int SFVecSize = 32;
  static_assert(K % SFVecSize == 0, "MP32 block-scaled MMA K must be divisible by SFVecSize.");

  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");

  static constexpr int MmaBits = sizeof_bits_v<a_type>;
  static_assert(MmaBits <= 8,
                "MP32 block-scaled SQMMA only supports operand types up to 8 bits.");
  static_assert(MmaBits == 4 || mute::is_same_v<sf_type, float_ue8m0_t>,
                "MP32 block-scaled SQMMA F6/F8 operands require float_ue8m0_t scale factor.");
  static_assert(MmaBits != 4 ||
                mute::is_same_v<sf_type, float_ue8m0_t> ||
                mute::is_same_v<sf_type, float_ue4m3_t>,
                "MP32 block-scaled SQMMA F4 operands require float_ue8m0_t or float_ue4m3_t scale factor.");

  static constexpr bool RequireAKMajor = sizeof_bits_v<a_type> == 4;
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireAKMajor || tnspA == TCE::Major::K,
                "MP32 SQMMA A operand requires K-major layout.");
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  using SFARegisters = int32_t[mute::ceil_div(64 * (K / SFVecSize), 512)];
  using SFBRegisters = int32_t[mute::ceil_div(64 * (K / SFVecSize), 512)];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      int32_t const* sfa,
      int32_t const* sfb,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m64n64_mxscale_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType), sfa, sfb);
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32::BLOCKSCALED::MP32_64x64_SS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_64x128_SS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static constexpr int SFVecSize = 32;
  static_assert(K % SFVecSize == 0, "MP32 block-scaled MMA K must be divisible by SFVecSize.");

  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");

  static constexpr int MmaBits = sizeof_bits_v<a_type>;
  static_assert(MmaBits <= 8,
                "MP32 block-scaled SQMMA only supports operand types up to 8 bits.");
  static_assert(MmaBits == 4 || mute::is_same_v<sf_type, float_ue8m0_t>,
                "MP32 block-scaled SQMMA F6/F8 operands require float_ue8m0_t scale factor.");
  static_assert(MmaBits != 4 ||
                mute::is_same_v<sf_type, float_ue8m0_t> ||
                mute::is_same_v<sf_type, float_ue4m3_t>,
                "MP32 block-scaled SQMMA F4 operands require float_ue8m0_t or float_ue4m3_t scale factor.");

  static constexpr bool RequireAKMajor = sizeof_bits_v<a_type> == 4;
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireAKMajor || tnspA == TCE::Major::K,
                "MP32 SQMMA A operand requires K-major layout.");
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  using SFARegisters = int32_t[mute::ceil_div(64 * (K / SFVecSize), 512)];
  using SFBRegisters = int32_t[mute::ceil_div(128 * (K / SFVecSize), 512)];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      int32_t const* sfa,
      int32_t const* sfb,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m64n128_mxscale_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType), sfa, sfb);
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32::BLOCKSCALED::MP32_64x128_SS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_128x32_SS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static constexpr int SFVecSize = 32;
  static_assert(K % SFVecSize == 0, "MP32 block-scaled MMA K must be divisible by SFVecSize.");

  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");

  static constexpr int MmaBits = sizeof_bits_v<a_type>;
  static_assert(MmaBits <= 8,
                "MP32 block-scaled SQMMA only supports operand types up to 8 bits.");
  static_assert(MmaBits == 4 || mute::is_same_v<sf_type, float_ue8m0_t>,
                "MP32 block-scaled SQMMA F6/F8 operands require float_ue8m0_t scale factor.");
  static_assert(MmaBits != 4 ||
                mute::is_same_v<sf_type, float_ue8m0_t> ||
                mute::is_same_v<sf_type, float_ue4m3_t>,
                "MP32 block-scaled SQMMA F4 operands require float_ue8m0_t or float_ue4m3_t scale factor.");

  static constexpr bool RequireAKMajor = sizeof_bits_v<a_type> == 4;
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireAKMajor || tnspA == TCE::Major::K,
                "MP32 SQMMA A operand requires K-major layout.");
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  using SFARegisters = int32_t[mute::ceil_div(128 * (K / SFVecSize), 512)];
  using SFBRegisters = int32_t[mute::ceil_div(32 * (K / SFVecSize), 512)];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      int32_t const* sfa,
      int32_t const* sfb,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m128n32_mxscale_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType), sfa, sfb);
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32::BLOCKSCALED::MP32_128x32_SS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_128x64_SS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static constexpr int SFVecSize = 32;
  static_assert(K % SFVecSize == 0, "MP32 block-scaled MMA K must be divisible by SFVecSize.");

  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");

  static constexpr int MmaBits = sizeof_bits_v<a_type>;
  static_assert(MmaBits <= 8,
                "MP32 block-scaled SQMMA only supports operand types up to 8 bits.");
  static_assert(MmaBits == 4 || mute::is_same_v<sf_type, float_ue8m0_t>,
                "MP32 block-scaled SQMMA F6/F8 operands require float_ue8m0_t scale factor.");
  static_assert(MmaBits != 4 ||
                mute::is_same_v<sf_type, float_ue8m0_t> ||
                mute::is_same_v<sf_type, float_ue4m3_t>,
                "MP32 block-scaled SQMMA F4 operands require float_ue8m0_t or float_ue4m3_t scale factor.");

  static constexpr bool RequireAKMajor = sizeof_bits_v<a_type> == 4;
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireAKMajor || tnspA == TCE::Major::K,
                "MP32 SQMMA A operand requires K-major layout.");
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  using SFARegisters = int32_t[mute::ceil_div(128 * (K / SFVecSize), 512)];
  using SFBRegisters = int32_t[mute::ceil_div(64 * (K / SFVecSize), 512)];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      int32_t const* sfa,
      int32_t const* sfb,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m128n64_mxscale_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType), sfa, sfb);
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32::BLOCKSCALED::MP32_128x64_SS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_128x128_SS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static constexpr int SFVecSize = 32;
  static_assert(K % SFVecSize == 0, "MP32 block-scaled MMA K must be divisible by SFVecSize.");

  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");

  static constexpr int MmaBits = sizeof_bits_v<a_type>;
  static_assert(MmaBits <= 8,
                "MP32 block-scaled SQMMA only supports operand types up to 8 bits.");
  static_assert(MmaBits == 4 || mute::is_same_v<sf_type, float_ue8m0_t>,
                "MP32 block-scaled SQMMA F6/F8 operands require float_ue8m0_t scale factor.");
  static_assert(MmaBits != 4 ||
                mute::is_same_v<sf_type, float_ue8m0_t> ||
                mute::is_same_v<sf_type, float_ue4m3_t>,
                "MP32 block-scaled SQMMA F4 operands require float_ue8m0_t or float_ue4m3_t scale factor.");

  static constexpr bool RequireAKMajor = sizeof_bits_v<a_type> == 4;
  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireAKMajor || tnspA == TCE::Major::K,
                "MP32 SQMMA A operand requires K-major layout.");
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  using SFARegisters = int32_t[mute::ceil_div(128 * (K / SFVecSize), 512)];
  using SFBRegisters = int32_t[mute::ceil_div(128 * (K / SFVecSize), 512)];

  MUTE_HOST_DEVICE static void
  fma(int32_t const& desc_a,
      int32_t const& desc_b,
      int32_t      * d,
      int32_t const* sfa,
      int32_t const* sfb,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m128n128_mxscale_mma(
      d, &desc_a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType), sfa, sfb);
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32::BLOCKSCALED::MP32_128x128_SS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//// MP32 BLOCKSCALED RS-SQMMA
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_32x32_RS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static constexpr int SFVecSize = 32;
  static_assert(K % SFVecSize == 0, "MP32 block-scaled MMA K must be divisible by SFVecSize.");

  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");

  static constexpr int MmaBits = sizeof_bits_v<a_type>;
  static_assert(MmaBits <= 8,
                "MP32 block-scaled SQMMA only supports operand types up to 8 bits.");
  static_assert(MmaBits == 4 || mute::is_same_v<sf_type, float_ue8m0_t>,
                "MP32 block-scaled SQMMA F6/F8 operands require float_ue8m0_t scale factor.");
  static_assert(MmaBits != 4 ||
                mute::is_same_v<sf_type, float_ue8m0_t> ||
                mute::is_same_v<sf_type, float_ue4m3_t>,
                "MP32 block-scaled SQMMA F4 operands require float_ue8m0_t or float_ue4m3_t scale factor.");

  static_assert(tnspA == TCE::Major::K,
                "MP32 RS-SQMMA A operand requires K-major layout.");

  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[MmaKBytes / 16];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[8];

  using SFARegisters = int32_t[mute::ceil_div(32 * (K / SFVecSize), 512)];
  using SFBRegisters = int32_t[mute::ceil_div(32 * (K / SFVecSize), 512)];

  MUTE_HOST_DEVICE static void
  fma(int32_t const* a,
      int32_t const& desc_b,
      int32_t      * d,
      int32_t const* sfa,
      int32_t const* sfb,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m32n32_RS_mxscale_mma(
      d, a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType), sfa, sfb);
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32::BLOCKSCALED::MP32_32x32_RS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_32x64_RS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static constexpr int SFVecSize = 32;
  static_assert(K % SFVecSize == 0, "MP32 block-scaled MMA K must be divisible by SFVecSize.");

  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");

  static constexpr int MmaBits = sizeof_bits_v<a_type>;
  static_assert(MmaBits <= 8,
                "MP32 block-scaled SQMMA only supports operand types up to 8 bits.");
  static_assert(MmaBits == 4 || mute::is_same_v<sf_type, float_ue8m0_t>,
                "MP32 block-scaled SQMMA F6/F8 operands require float_ue8m0_t scale factor.");
  static_assert(MmaBits != 4 ||
                mute::is_same_v<sf_type, float_ue8m0_t> ||
                mute::is_same_v<sf_type, float_ue4m3_t>,
                "MP32 block-scaled SQMMA F4 operands require float_ue8m0_t or float_ue4m3_t scale factor.");

  static_assert(tnspA == TCE::Major::K,
                "MP32 RS-SQMMA A operand requires K-major layout.");

  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[MmaKBytes / 16];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  using SFARegisters = int32_t[mute::ceil_div(32 * (K / SFVecSize), 512)];
  using SFBRegisters = int32_t[mute::ceil_div(64 * (K / SFVecSize), 512)];

  MUTE_HOST_DEVICE static void
  fma(int32_t const* a,
      int32_t const& desc_b,
      int32_t      * d,
      int32_t const* sfa,
      int32_t const* sfb,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m32n64_RS_mxscale_mma(
      d, a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType), sfa, sfb);
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32::BLOCKSCALED::MP32_32x64_RS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_32x128_RS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static constexpr int SFVecSize = 32;
  static_assert(K % SFVecSize == 0, "MP32 block-scaled MMA K must be divisible by SFVecSize.");

  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");

  static constexpr int MmaBits = sizeof_bits_v<a_type>;
  static_assert(MmaBits <= 8,
                "MP32 block-scaled SQMMA only supports operand types up to 8 bits.");
  static_assert(MmaBits == 4 || mute::is_same_v<sf_type, float_ue8m0_t>,
                "MP32 block-scaled SQMMA F6/F8 operands require float_ue8m0_t scale factor.");
  static_assert(MmaBits != 4 ||
                mute::is_same_v<sf_type, float_ue8m0_t> ||
                mute::is_same_v<sf_type, float_ue4m3_t>,
                "MP32 block-scaled SQMMA F4 operands require float_ue8m0_t or float_ue4m3_t scale factor.");

  static_assert(tnspA == TCE::Major::K,
                "MP32 RS-SQMMA A operand requires K-major layout.");

  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[MmaKBytes / 16];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  using SFARegisters = int32_t[mute::ceil_div(32 * (K / SFVecSize), 512)];
  using SFBRegisters = int32_t[mute::ceil_div(128 * (K / SFVecSize), 512)];

  MUTE_HOST_DEVICE static void
  fma(int32_t const* a,
      int32_t const& desc_b,
      int32_t      * d,
      int32_t const* sfa,
      int32_t const* sfb,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m32n128_RS_mxscale_mma(
      d, a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType), sfa, sfb);
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32::BLOCKSCALED::MP32_32x128_RS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_64x32_RS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static constexpr int SFVecSize = 32;
  static_assert(K % SFVecSize == 0, "MP32 block-scaled MMA K must be divisible by SFVecSize.");

  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");

  static constexpr int MmaBits = sizeof_bits_v<a_type>;
  static_assert(MmaBits <= 8,
                "MP32 block-scaled SQMMA only supports operand types up to 8 bits.");
  static_assert(MmaBits == 4 || mute::is_same_v<sf_type, float_ue8m0_t>,
                "MP32 block-scaled SQMMA F6/F8 operands require float_ue8m0_t scale factor.");
  static_assert(MmaBits != 4 ||
                mute::is_same_v<sf_type, float_ue8m0_t> ||
                mute::is_same_v<sf_type, float_ue4m3_t>,
                "MP32 block-scaled SQMMA F4 operands require float_ue8m0_t or float_ue4m3_t scale factor.");

  static_assert(tnspA == TCE::Major::K,
                "MP32 RS-SQMMA A operand requires K-major layout.");

  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[MmaKBytes / 8];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[16];

  using SFARegisters = int32_t[mute::ceil_div(64 * (K / SFVecSize), 512)];
  using SFBRegisters = int32_t[mute::ceil_div(32 * (K / SFVecSize), 512)];

  MUTE_HOST_DEVICE static void
  fma(int32_t const* a,
      int32_t const& desc_b,
      int32_t      * d,
      int32_t const* sfa,
      int32_t const* sfb,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m64n32_RS_mxscale_mma(
      d, a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType), sfa, sfb);
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32::BLOCKSCALED::MP32_64x32_RS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_64x64_RS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static constexpr int SFVecSize = 32;
  static_assert(K % SFVecSize == 0, "MP32 block-scaled MMA K must be divisible by SFVecSize.");

  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");

  static constexpr int MmaBits = sizeof_bits_v<a_type>;
  static_assert(MmaBits <= 8,
                "MP32 block-scaled SQMMA only supports operand types up to 8 bits.");
  static_assert(MmaBits == 4 || mute::is_same_v<sf_type, float_ue8m0_t>,
                "MP32 block-scaled SQMMA F6/F8 operands require float_ue8m0_t scale factor.");
  static_assert(MmaBits != 4 ||
                mute::is_same_v<sf_type, float_ue8m0_t> ||
                mute::is_same_v<sf_type, float_ue4m3_t>,
                "MP32 block-scaled SQMMA F4 operands require float_ue8m0_t or float_ue4m3_t scale factor.");

  static_assert(tnspA == TCE::Major::K,
                "MP32 RS-SQMMA A operand requires K-major layout.");

  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[MmaKBytes / 8];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  using SFARegisters = int32_t[mute::ceil_div(64 * (K / SFVecSize), 512)];
  using SFBRegisters = int32_t[mute::ceil_div(64 * (K / SFVecSize), 512)];

  MUTE_HOST_DEVICE static void
  fma(int32_t const* a,
      int32_t const& desc_b,
      int32_t      * d,
      int32_t const* sfa,
      int32_t const* sfb,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m64n64_RS_mxscale_mma(
      d, a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType), sfa, sfb);
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32::BLOCKSCALED::MP32_64x64_RS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_64x128_RS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static constexpr int SFVecSize = 32;
  static_assert(K % SFVecSize == 0, "MP32 block-scaled MMA K must be divisible by SFVecSize.");

  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");

  static constexpr int MmaBits = sizeof_bits_v<a_type>;
  static_assert(MmaBits <= 8,
                "MP32 block-scaled SQMMA only supports operand types up to 8 bits.");
  static_assert(MmaBits == 4 || mute::is_same_v<sf_type, float_ue8m0_t>,
                "MP32 block-scaled SQMMA F6/F8 operands require float_ue8m0_t scale factor.");
  static_assert(MmaBits != 4 ||
                mute::is_same_v<sf_type, float_ue8m0_t> ||
                mute::is_same_v<sf_type, float_ue4m3_t>,
                "MP32 block-scaled SQMMA F4 operands require float_ue8m0_t or float_ue4m3_t scale factor.");

  static_assert(tnspA == TCE::Major::K,
                "MP32 RS-SQMMA A operand requires K-major layout.");

  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[MmaKBytes / 8];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  using SFARegisters = int32_t[mute::ceil_div(64 * (K / SFVecSize), 512)];
  using SFBRegisters = int32_t[mute::ceil_div(128 * (K / SFVecSize), 512)];

  MUTE_HOST_DEVICE static void
  fma(int32_t const* a,
      int32_t const& desc_b,
      int32_t      * d,
      int32_t const* sfa,
      int32_t const* sfb,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m64n128_RS_mxscale_mma(
      d, a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType), sfa, sfb);
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32::BLOCKSCALED::MP32_64x128_RS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_128x32_RS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static constexpr int SFVecSize = 32;
  static_assert(K % SFVecSize == 0, "MP32 block-scaled MMA K must be divisible by SFVecSize.");

  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");

  static constexpr int MmaBits = sizeof_bits_v<a_type>;
  static_assert(MmaBits <= 8,
                "MP32 block-scaled SQMMA only supports operand types up to 8 bits.");
  static_assert(MmaBits == 4 || mute::is_same_v<sf_type, float_ue8m0_t>,
                "MP32 block-scaled SQMMA F6/F8 operands require float_ue8m0_t scale factor.");
  static_assert(MmaBits != 4 ||
                mute::is_same_v<sf_type, float_ue8m0_t> ||
                mute::is_same_v<sf_type, float_ue4m3_t>,
                "MP32 block-scaled SQMMA F4 operands require float_ue8m0_t or float_ue4m3_t scale factor.");

  static_assert(tnspA == TCE::Major::K,
                "MP32 RS-SQMMA A operand requires K-major layout.");

  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[MmaKBytes / 4];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[32];

  using SFARegisters = int32_t[mute::ceil_div(128 * (K / SFVecSize), 512)];
  using SFBRegisters = int32_t[mute::ceil_div(32 * (K / SFVecSize), 512)];

  MUTE_HOST_DEVICE static void
  fma(int32_t const* a,
      int32_t const& desc_b,
      int32_t      * d,
      int32_t const* sfa,
      int32_t const* sfb,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m128n32_RS_mxscale_mma(
      d, a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType), sfa, sfb);
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32::BLOCKSCALED::MP32_128x32_RS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_128x64_RS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static constexpr int SFVecSize = 32;
  static_assert(K % SFVecSize == 0, "MP32 block-scaled MMA K must be divisible by SFVecSize.");

  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");

  static constexpr int MmaBits = sizeof_bits_v<a_type>;
  static_assert(MmaBits <= 8,
                "MP32 block-scaled SQMMA only supports operand types up to 8 bits.");
  static_assert(MmaBits == 4 || mute::is_same_v<sf_type, float_ue8m0_t>,
                "MP32 block-scaled SQMMA F6/F8 operands require float_ue8m0_t scale factor.");
  static_assert(MmaBits != 4 ||
                mute::is_same_v<sf_type, float_ue8m0_t> ||
                mute::is_same_v<sf_type, float_ue4m3_t>,
                "MP32 block-scaled SQMMA F4 operands require float_ue8m0_t or float_ue4m3_t scale factor.");

  static_assert(tnspA == TCE::Major::K,
                "MP32 RS-SQMMA A operand requires K-major layout.");

  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[MmaKBytes / 4];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[64];

  using SFARegisters = int32_t[mute::ceil_div(128 * (K / SFVecSize), 512)];
  using SFBRegisters = int32_t[mute::ceil_div(64 * (K / SFVecSize), 512)];

  MUTE_HOST_DEVICE static void
  fma(int32_t const* a,
      int32_t const& desc_b,
      int32_t      * d,
      int32_t const* sfa,
      int32_t const* sfb,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m128n64_RS_mxscale_mma(
      d, a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType), sfa, sfb);
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32::BLOCKSCALED::MP32_128x64_RS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
struct MP32_128x128_RS
{
  static constexpr auto MmaType = MP32::detail::deduce_tce_dtype_enum_v<a_type, b_type>;

  static constexpr int SFVecSize = 32;
  static_assert(K % SFVecSize == 0, "MP32 block-scaled MMA K must be divisible by SFVecSize.");

  static constexpr int MmaKBytes = K * MP32::detail::deduce_tce_bits_v<a_type, b_type> / 8;
  static_assert(MmaKBytes == 32 || MmaKBytes == 64 || MmaKBytes == 128 || MmaKBytes == 256,
                "Invalid SQMMA Instruction K");

  static constexpr int MmaBits = sizeof_bits_v<a_type>;
  static_assert(MmaBits <= 8,
                "MP32 block-scaled SQMMA only supports operand types up to 8 bits.");
  static_assert(MmaBits == 4 || mute::is_same_v<sf_type, float_ue8m0_t>,
                "MP32 block-scaled SQMMA F6/F8 operands require float_ue8m0_t scale factor.");
  static_assert(MmaBits != 4 ||
                mute::is_same_v<sf_type, float_ue8m0_t> ||
                mute::is_same_v<sf_type, float_ue4m3_t>,
                "MP32 block-scaled SQMMA F4 operands require float_ue8m0_t or float_ue4m3_t scale factor.");

  static_assert(tnspA == TCE::Major::K,
                "MP32 RS-SQMMA A operand requires K-major layout.");

  static constexpr bool RequireBKMajor = sizeof_bits_v<b_type> == 4;
  static_assert(!RequireBKMajor || tnspB == TCE::Major::K,
                "MP32 SQMMA B operand requires K-major layout.");

  using DRegisters = void;
  using ARegisters = int32_t[MmaKBytes / 4];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[128];

  using SFARegisters = int32_t[mute::ceil_div(128 * (K / SFVecSize), 512)];
  using SFBRegisters = int32_t[mute::ceil_div(128 * (K / SFVecSize), 512)];

  MUTE_HOST_DEVICE static void
  fma(int32_t const* a,
      int32_t const& desc_b,
      int32_t      * d,
      int32_t const* sfa,
      int32_t const* sfb,
      MP31::SQMMA::ScaleOut scaleD = MP31::SQMMA::ScaleOut::One)
  {
#if defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
    __musa_sqmma_m128n128_RS_mxscale_mma(
      d, a, &desc_b, d, static_cast<int32_t>(tnspA), 1 - static_cast<int32_t>(tnspB),
      static_cast<int32_t>(scaleA), static_cast<int32_t>(scaleB), static_cast<int32_t>(scaleD),
      /* sat = */ 0, /* k = */ K, /* mma_type = */ static_cast<int>(MmaType), sfa, sfb);
#else
    MUTE_INVALID_CONTROL_PATH("Attempting to use MP32::BLOCKSCALED::MP32_128x128_RS without MUTE_ARCH_SQMMA_MP32_ENABLED");
#endif
  }
};

} // namespace MP32::BLOCKSCALED

////////////////////////////////////////////////////////////////////////////////////////////////////
//// MP32 SQMMA instruction selectors
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace MP32::SQMMA::detail {

MUTE_HOST_DEVICE constexpr int
select_mn_extent(int tile_extent, int max_extent)
{
  return tile_extent % 128 == 0 && max_extent >= 128 ? 128 :
         tile_extent %  64 == 0 && max_extent >=  64 ?  64 :
         tile_extent %  32 == 0 && max_extent >=  32 ?  32 : 0;
}

MUTE_HOST_DEVICE constexpr int
select_k_bytes(int tile_k_bytes, int max_k_bytes)
{
  return tile_k_bytes % 256 == 0 && max_k_bytes >= 256 ? 256 :
         tile_k_bytes % 128 == 0 && max_k_bytes >= 128 ? 128 :
         tile_k_bytes %  64 == 0 && max_k_bytes >=  64 ?  64 :
         tile_k_bytes %  32 == 0 && max_k_bytes >=  32 ?  32 : 0;
}

template <class ElementA,
          class ElementB,
          class ElementC,
          class TileShape_MNK,
          class MaxInstructionM,
          class MaxInstructionN,
          class MaxInstructionKBytes,
          int EligibleMaxM,
          int EligibleMaxN>
struct op_selection
{
  static_assert(is_static<TileShape_MNK>::value, "TileShape_MNK must be static.");
  static_assert(rank(TileShape_MNK{}) == 3, "TileShape_MNK must be rank 3.");

  static constexpr int TileM = size<0>(TileShape_MNK{});
  static constexpr int TileN = size<1>(TileShape_MNK{});
  static constexpr int TileK = size<2>(TileShape_MNK{});
  static_assert(TileM > 0 && TileN > 0 && TileK > 0,
                "TileShape_MNK extents must be positive.");

  static constexpr int MaxM = MaxInstructionM{};
  static constexpr int MaxN = MaxInstructionN{};
  static constexpr int MaxKBytes = MaxInstructionKBytes{};

  static_assert(MaxM == 32 || MaxM == 64 || MaxM == 128,
                "MaxInstructionM must be 32, 64, or 128.");
  static_assert(MaxN == 32 || MaxN == 64 || MaxN == 128,
                "MaxInstructionN must be 32, 64, or 128.");
  static_assert(MaxKBytes == 32 || MaxKBytes == 64 ||
                MaxKBytes == 128 || MaxKBytes == 256,
                "MaxInstructionKBytes must be 32, 64, 128, or 256.");

  static_assert(TileM % 32 == 0, "Tile_M must be a multiple of 32.");
  static_assert(TileN % 32 == 0, "Tile_N must be a multiple of 32.");

  static constexpr int EffectiveMaxM = MaxM < EligibleMaxM ? MaxM : EligibleMaxM;
  static constexpr int EffectiveMaxN = MaxN < EligibleMaxN ? MaxN : EligibleMaxN;
  static constexpr int M = select_mn_extent(TileM, EffectiveMaxM);
  static constexpr int N = select_mn_extent(TileN, EffectiveMaxN);

  using ExpectedElementC = MP32::detail::deduce_tce_acc_type_t<ElementA, ElementB>;
  static_assert(is_same_v<ElementC, ExpectedElementC>,
                "ElementC does not match the MP32 SQMMA accumulator type.");

  static constexpr int MmaBits = MP32::detail::deduce_tce_bits_v<ElementA, ElementB>;
  static_assert(256 % MmaBits == 0, "MP32 SQMMA element bits must divide 256.");
  static constexpr int MinInstructionK = 32 * 8 / MmaBits;
  static_assert(TileK % MinInstructionK == 0,
                "Tile_K must contain a whole 32-byte SQMMA instruction.");

  static constexpr int TileKBytes = TileK * MmaBits / 8;
  static constexpr int KBytes = select_k_bytes(TileKBytes, MaxKBytes);
  static_assert(KBytes != 0, "Unable to select an MP32 SQMMA instruction K.");
  static constexpr int K = KBytes * 8 / MmaBits;
};

template <int M,
          int N,
          bool IsRS,
          class ElementSF,
          class ElementA,
          class ElementB,
          int K,
          TCE::Major MajorA,
          TCE::Major MajorB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
MUTE_HOST_DEVICE constexpr auto
make_operation()
{
  if constexpr (!IsRS && is_void_v<ElementSF>) {
    if constexpr (M == 128 && N == 128) {
      return ::mute::MP32_128x128_SS<ElementA, ElementB, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 128 && N == 64) {
      return ::mute::MP32_128x64_SS<ElementA, ElementB, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 128 && N == 32) {
      return ::mute::MP32_128x32_SS<ElementA, ElementB, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 64 && N == 128) {
      return ::mute::MP32_64x128_SS<ElementA, ElementB, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 64 && N == 64) {
      return ::mute::MP32_64x64_SS<ElementA, ElementB, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 64 && N == 32) {
      return ::mute::MP32_64x32_SS<ElementA, ElementB, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 32 && N == 128) {
      return ::mute::MP32_32x128_SS<ElementA, ElementB, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 32 && N == 64) {
      return ::mute::MP32_32x64_SS<ElementA, ElementB, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 32 && N == 32) {
      return ::mute::MP32_32x32_SS<ElementA, ElementB, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else {
      static_assert(dependent_false<Int<M>, Int<N>>, "Unsupported MP32 SS-SQMMA shape.");
    }
  }
  else if constexpr (IsRS && is_void_v<ElementSF>) {
    if constexpr (M == 128 && N == 128) {
      return ::mute::MP32_128x128_RS<ElementA, ElementB, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 128 && N == 64) {
      return ::mute::MP32_128x64_RS<ElementA, ElementB, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 128 && N == 32) {
      return ::mute::MP32_128x32_RS<ElementA, ElementB, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 64 && N == 128) {
      return ::mute::MP32_64x128_RS<ElementA, ElementB, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 64 && N == 64) {
      return ::mute::MP32_64x64_RS<ElementA, ElementB, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 64 && N == 32) {
      return ::mute::MP32_64x32_RS<ElementA, ElementB, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 32 && N == 128) {
      return ::mute::MP32_32x128_RS<ElementA, ElementB, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 32 && N == 64) {
      return ::mute::MP32_32x64_RS<ElementA, ElementB, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 32 && N == 32) {
      return ::mute::MP32_32x32_RS<ElementA, ElementB, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else {
      static_assert(dependent_false<Int<M>, Int<N>>, "Unsupported MP32 RS-SQMMA shape.");
    }
  }
  else if constexpr (!IsRS) {
    if constexpr (M == 128 && N == 128) {
      return MP32::BLOCKSCALED::MP32_128x128_SS<ElementA, ElementB, ElementSF, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 128 && N == 64) {
      return MP32::BLOCKSCALED::MP32_128x64_SS<ElementA, ElementB, ElementSF, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 128 && N == 32) {
      return MP32::BLOCKSCALED::MP32_128x32_SS<ElementA, ElementB, ElementSF, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 64 && N == 128) {
      return MP32::BLOCKSCALED::MP32_64x128_SS<ElementA, ElementB, ElementSF, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 64 && N == 64) {
      return MP32::BLOCKSCALED::MP32_64x64_SS<ElementA, ElementB, ElementSF, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 64 && N == 32) {
      return MP32::BLOCKSCALED::MP32_64x32_SS<ElementA, ElementB, ElementSF, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 32 && N == 128) {
      return MP32::BLOCKSCALED::MP32_32x128_SS<ElementA, ElementB, ElementSF, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 32 && N == 64) {
      return MP32::BLOCKSCALED::MP32_32x64_SS<ElementA, ElementB, ElementSF, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 32 && N == 32) {
      return MP32::BLOCKSCALED::MP32_32x32_SS<ElementA, ElementB, ElementSF, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else {
      static_assert(dependent_false<Int<M>, Int<N>>, "Unsupported MP32 block-scaled SS-SQMMA shape.");
    }
  }
  else {
    if constexpr (M == 128 && N == 128) {
      return MP32::BLOCKSCALED::MP32_128x128_RS<ElementA, ElementB, ElementSF, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 128 && N == 64) {
      return MP32::BLOCKSCALED::MP32_128x64_RS<ElementA, ElementB, ElementSF, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 128 && N == 32) {
      return MP32::BLOCKSCALED::MP32_128x32_RS<ElementA, ElementB, ElementSF, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 64 && N == 128) {
      return MP32::BLOCKSCALED::MP32_64x128_RS<ElementA, ElementB, ElementSF, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 64 && N == 64) {
      return MP32::BLOCKSCALED::MP32_64x64_RS<ElementA, ElementB, ElementSF, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 64 && N == 32) {
      return MP32::BLOCKSCALED::MP32_64x32_RS<ElementA, ElementB, ElementSF, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 32 && N == 128) {
      return MP32::BLOCKSCALED::MP32_32x128_RS<ElementA, ElementB, ElementSF, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 32 && N == 64) {
      return MP32::BLOCKSCALED::MP32_32x64_RS<ElementA, ElementB, ElementSF, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else if constexpr (M == 32 && N == 32) {
      return MP32::BLOCKSCALED::MP32_32x32_RS<ElementA, ElementB, ElementSF, K, MajorA, MajorB, scaleA, scaleB>{};
    }
    else {
      static_assert(dependent_false<Int<M>, Int<N>>, "Unsupported MP32 block-scaled RS-SQMMA shape.");
    }
  }
}

template <bool IsRS,
          class ElementSF,
          class ElementA,
          class ElementB,
          class ElementC,
          class TileShape_MNK,
          TCE::Major MajorA,
          TCE::Major MajorB,
          class MaxInstructionM,
          class MaxInstructionN,
          class MaxInstructionKBytes,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
MUTE_HOST_DEVICE constexpr auto
op_selector_impl()
{
  constexpr bool IsBlockScaled = !is_void_v<ElementSF>;
  constexpr int EligibleMaxM =
      !IsBlockScaled && !IsRS && is_same_v<ElementA, tfloat32_t> && MajorA != TCE::Major::K ? 64 : 128;
  constexpr int EligibleMaxN =
      !IsBlockScaled && is_same_v<ElementB, tfloat32_t> && MajorB != TCE::Major::K ? 64 : 128;

  using Selection = op_selection<ElementA, ElementB, ElementC, TileShape_MNK,
                                 MaxInstructionM, MaxInstructionN, MaxInstructionKBytes,
                                 EligibleMaxM, EligibleMaxN>;

  return make_operation<Selection::M, Selection::N, IsRS, ElementSF,
                        ElementA, ElementB, Selection::K, MajorA, MajorB, scaleA, scaleB>();
}

} // namespace MP32::SQMMA::detail

namespace MP32::SQMMA {

// Unscaled SQMMA selectors.
template <class ElementA,
          class ElementB,
          class ElementC,
          class TileShape_MNK,
          TCE::Major MajorA = TCE::Major::K,
          TCE::Major MajorB = TCE::Major::K,
          class MaxInstructionM = _128,
          class MaxInstructionN = _128,
          class MaxInstructionKBytes = _256,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
MUTE_HOST_DEVICE constexpr auto
ss_op_selector()
{
  return detail::op_selector_impl<false, void,
                                  ElementA, ElementB, ElementC, TileShape_MNK,
                                  MajorA, MajorB,
                                  MaxInstructionM, MaxInstructionN, MaxInstructionKBytes,
                                  scaleA, scaleB>();
}

template <class ElementA,
          class ElementB,
          class ElementC,
          class TileShape_MNK,
          TCE::Major MajorA = TCE::Major::K,
          TCE::Major MajorB = TCE::Major::K,
          class MaxInstructionM = _128,
          class MaxInstructionN = _128,
          class MaxInstructionKBytes = _256,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
MUTE_HOST_DEVICE constexpr auto
rs_op_selector()
{
  return detail::op_selector_impl<true, void,
                                  ElementA, ElementB, ElementC, TileShape_MNK,
                                  MajorA, MajorB,
                                  MaxInstructionM, MaxInstructionN, MaxInstructionKBytes,
                                  scaleA, scaleB>();
}

// Block-scaled SQMMA selector overloads.
template <class ElementA,
          class ElementB,
          class ElementC,
          class ElementSF,
          class TileShape_MNK,
          TCE::Major MajorA = TCE::Major::K,
          TCE::Major MajorB = TCE::Major::K,
          class MaxInstructionM = _128,
          class MaxInstructionN = _128,
          class MaxInstructionKBytes = _256,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
MUTE_HOST_DEVICE constexpr auto
ss_op_selector()
{
  static_assert(!is_void_v<ElementSF>, "ElementSF must be a scale factor type.");
  return MP32::SQMMA::detail::op_selector_impl<false, ElementSF,
                                               ElementA, ElementB, ElementC, TileShape_MNK,
                                               MajorA, MajorB,
                                               MaxInstructionM, MaxInstructionN, MaxInstructionKBytes,
                                               scaleA, scaleB>();
}

template <class ElementA,
          class ElementB,
          class ElementC,
          class ElementSF,
          class TileShape_MNK,
          TCE::Major MajorA = TCE::Major::K,
          TCE::Major MajorB = TCE::Major::K,
          class MaxInstructionM = _128,
          class MaxInstructionN = _128,
          class MaxInstructionKBytes = _256,
          MP31::SQMMA::ScaleIn scaleA = MP31::SQMMA::ScaleIn::One,
          MP31::SQMMA::ScaleIn scaleB = MP31::SQMMA::ScaleIn::One>
MUTE_HOST_DEVICE constexpr auto
rs_op_selector()
{
  static_assert(!is_void_v<ElementSF>, "ElementSF must be a scale factor type.");
  return MP32::SQMMA::detail::op_selector_impl<true, ElementSF,
                                               ElementA, ElementB, ElementC, TileShape_MNK,
                                               MajorA, MajorB,
                                               MaxInstructionM, MaxInstructionN, MaxInstructionKBytes,
                                               scaleA, scaleB>();
}

} // namespace MP32::SQMMA

} // namespace mute
