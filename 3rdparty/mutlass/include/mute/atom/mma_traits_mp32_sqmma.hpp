#pragma once

#include <mute/arch/mma_mp32.hpp>
#include <mute/atom/mma_traits.hpp>
#include <mute/atom/mma_traits_mp31_sqmma.hpp>
#include <mute/tensor_zip.hpp>

namespace mute {

///////////////////////////////////////////////////////////////////////////////
//////////////////////////// MMA_TRAITS ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

namespace MP32::SQMMA {

////////////////////////////////////////////
// Common layouts for SQMMA Shared Memory //
////////////////////////////////////////////
template <int MN, int K, class Type, TCE::Major tnsp>
using Layout_Atom_Bits = MP31::SQMMA::Layout_Atom_Bits<MN, K, Type, tnsp>;

template <int MN, int K, class Type, TCE::Major tnsp>
using Layout_SL256_SS256_NONE_Atom = MP31::SQMMA::Layout_SL256_SS256_NONE_Atom<MN, K, Type, tnsp>;

template <int MN, int K, class Type, TCE::Major tnsp>
using Layout_SL256_SS256_SG16_Atom = MP31::SQMMA::Layout_SL256_SS256_SG16_Atom<MN, K, Type, tnsp>;

template <int MN, int K, class Type, TCE::Major tnsp>
using Layout_SL256_SS256_SG32_Atom = MP31::SQMMA::Layout_SL256_SS256_SG32_Atom<MN, K, Type, tnsp>;

template <int MN, int K, class Type, TCE::Major tnsp>
using Layout_SL256_SS256_SG64_Atom = MP31::SQMMA::Layout_SL256_SS256_SG64_Atom<MN, K, Type, tnsp>;

template <int MN, int K, class Type, TCE::Major tnsp>
using Layout_SL128_SS128_SG32_Atom = MP31::SQMMA::Layout_SL128_SS128_SG32_Atom<MN, K, Type, tnsp>;

template <int MN, int K, class Type, TCE::Major tnsp>
using Layout_SL128_SS128_SG16_Atom = MP31::SQMMA::Layout_SL128_SS128_SG16_Atom<MN, K, Type, tnsp>;

// Only used for canonical gemm
template <class Type, TCE::Major tnsp, class AtomMN, class AtomK>
MUTE_HOST_DEVICE constexpr
auto
make_canonical_gemm_smem_atom_layout() {
  constexpr int BITS = sizeof_bits_v<Type>;
  constexpr int ATOM_MN = size(AtomMN{});
  constexpr int ATOM_K  = size(AtomK{});
  if constexpr (tnsp == TCE::Major::MN) {
    // S4/S8/U8/FP8
    if constexpr (BITS <= 8) {
      return Layout_SL256_SS256_SG16_Atom<ATOM_MN, ATOM_K, Type, tnsp>{};
    }
    // FP16/BF16, MP32 supports both SG16 and SG32 for FP16/BF16; prefer SG16.
    else if constexpr (BITS == 16) {
      return Layout_SL256_SS256_SG16_Atom<ATOM_MN, ATOM_K, Type, tnsp>{};
    }
    // TF32
    else if constexpr (BITS == 32) {
      return Layout_SL256_SS256_SG64_Atom<ATOM_MN, ATOM_K, Type, tnsp>{};
    }
    else {
      static_assert(BITS <= 32, "Unsupported Sqmma Type");
    }
  }
  else if constexpr (tnsp == TCE::Major::K) {
    return Layout_SL256_SS256_SG16_Atom<ATOM_MN, ATOM_K, Type, tnsp>{};
  }
  else {
    static_assert(tnsp != TCE::Major::MN && tnsp != TCE::Major::K, "Unrecognized MajorMode!");
  }
}

///////////////////////////////////////////////////////////////////////////////
// Construction method for SQMMA Descriptors
///////////////////////////////////////////////////////////////////////////////
template <TCE::Major MajorMode, bool IsMtF4 = false, class TEngine, class TLayout>
MUTE_HOST_DEVICE constexpr
SqmmaDescriptor
make_sqmma_desc(Tensor<TEngine, TLayout> const& tensor)
{
  static_assert(is_smem<TEngine>::value, "SQMMA Descriptors can only be constructed on smem.");
  static_assert(TLayout::rank == 2, "SQMMA Descriptors can only be constructed on rank-2 tensors.");

  Tensor u8_tensor = recast<uint8_t const>(tensor);

  SqmmaDescriptor desc;

  constexpr auto MP31_SG_TYPE = MP31::SQMMA::swizzle_granularity_type(u8_tensor);
  static_assert(MP31_SG_TYPE != MP31::SQMMA::SwizzleGranularityType::SWIZZLE_GRANULARITY_NONE,
                "MP32 SQMMA requires an enabled shared-memory swizzle.");
  desc.bitfield.swizzle_granularity_type_ = static_cast<uint8_t>(MP31_SG_TYPE) - 1;
  desc.bitfield.swizzle_type_ = 1;
  desc.bitfield.swizzle_stride_type_ = static_cast<uint8_t>(SwizzleStrideType::B256);
  desc.bitfield.fp4_type_ = static_cast<uint8_t>(IsMtF4 ? Fp4Type::MTF4 : Fp4Type::MXF4);

  desc.bitfield.start_address_ = cast_smem_ptr_to_uint(raw_pointer_cast(u8_tensor.data()));

  if constexpr (MajorMode == TCE::Major::MN)
  {
    MUTE_STATIC_ASSERT_V(size<0>(u8_tensor) % Int<16>{} == Int<0>{},
                         "Not a canonical SQMMA_MN Layout: Expected MN-size multiple of 16.");

    MUTE_STATIC_ASSERT_V(size<1>(u8_tensor) % Int<8>{} == Int<0>{} &&
                         size<1>(u8_tensor) <= Int<256>{},
                         "Not a canonical SQMMA_MN Layout: Expected K-size multiple of 8 (in units of uint8_t).");

    constexpr uint32_t stride_1 = stride<1>(u8_tensor);
    static_assert(8 <= stride_1 && stride_1 <= 256, "Not a canonical SQMMA_MN Layout: Expected stride failure.");

    switch (stride_1) {
      case   8: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B8);   break;
      case  16: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B16);  break;
      case  32: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B32);  break;
      case  64: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B64);  break;
      case 128: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B128); break;
      case 256: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B256); break;
    }
  }
  else if constexpr (MajorMode == TCE::Major::K)
  {
    MUTE_STATIC_ASSERT_V(size<0>(u8_tensor) % Int<16>{} == Int<0>{},
                         "Not a canonical SQMMA_K Layout: Expected MN-size multiple of 16.");
    MUTE_STATIC_ASSERT_V(size<1>(u8_tensor) % Int<8>{} == Int<0>{} &&
                         size<1>(u8_tensor) <= Int<256>{},
                         "Not a canonical SQMMA_K Layout: Expected K-size multiple of 8 (in units of uint8_t).");

    constexpr uint32_t stride_0 = stride<0>(u8_tensor);
    static_assert(8 <= stride_0 && stride_0 <= 256, "Not a canonical SQMMA_K Layout: Expected stride failure.");

    switch (stride_0) {
      case   8: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B8);   break;
      case  16: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B16);  break;
      case  32: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B32);  break;
      case  64: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B64);  break;
      case 128: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B128); break;
      case 256: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B256); break;
    }
  }
  else {
    static_assert(MajorMode != TCE::Major::MN && MajorMode != TCE::Major::K, "Unrecognized MajorMode!");
  }

  return desc;
}

///////////////////////////////////////////////////////////////////////////////
// Higher level SQMMA Descriptor utilities
///////////////////////////////////////////////////////////////////////////////

struct DescriptorIterator
{
  using reference    = SqmmaDescriptor;
  using element_type = SqmmaDescriptor;
  using value_type   = SqmmaDescriptor;

  SqmmaDescriptor desc_;

  MUTE_HOST_DEVICE constexpr
  reference operator*() const { return desc_; }

  template <class Index>
  MUTE_HOST_DEVICE constexpr
  reference operator[](Index const& i) const { return *(*this + i); }

  template <class Index>
  MUTE_HOST_DEVICE constexpr
  DescriptorIterator operator+(Index const& offset) const
  {
    return { SqmmaDescriptor{desc_ + uint64_t(offset)} };
  }

  MUTE_HOST_DEVICE constexpr
  operator int32_t() const { return static_cast<int32_t>(desc_); }

  MUTE_HOST_DEVICE friend void
  print(DescriptorIterator) { printf("SQMMA::DescriptorIterator"); }
};

MUTE_HOST_DEVICE constexpr
SqmmaDescriptor
raw_pointer_cast(DescriptorIterator const& ptr) {
  return ptr.desc_;
}

template <class NewT>
MUTE_HOST_DEVICE constexpr
DescriptorIterator
recast_ptr(DescriptorIterator const& iter) {
  static_assert(is_same<NewT, int32_t>::value, "Can only cast SqmmaDescriptorIterator to int32_t.");
  return iter;
}

template <TCE::Major, bool IsMtF4 = false>
struct smem_desc : DescriptorIterator {};

// Accumulator layouts
template <int M, int N>
using CLayout = Layout<Shape <Shape <    _4, _8, _4>, Shape <      _2, Int<N/8>, Int<M/32>>>,
                       Stride<Stride<Int<M>, _1, _8>, Stride<Int<M*4>, Int<M*8>,      _32>>>;


// Register source layout for K-Major
template <int M, int K, int Bits>
using ALayout = Layout<Shape <Shape <            _4, _8, _4>, Shape <Int<32/Bits>, Int<Bits*K/128>, Int<M/32>>>,
                       Stride<Stride<Int<M*32/Bits>, _1, _8>, Stride<      Int<M>, Int<M*128/Bits>,      _32>>>;


// Shared memory source layouts for any value type
template <int M, int K>
using ABLayout  = Layout<Shape <_128, Shape <Int<M>,Int<K>>>,
                         Stride<  _0, Stride<    _1,Int<M>>>>;


// Scale-factor layout: four consecutive MN scales form one dword, and
// consecutive threads advance through MN before advancing to the next SF group.
template <int MN, int K>
using SFLayout = Layout<
  Shape<Shape<Int<MN / 4>,
              Int<mute::min(K / 32, 128 / (MN / 4))>,
              Int<128 / ((MN / 4) * mute::min(K / 32, 128 / (MN / 4)))>>,
        Shape<_4, _32, Int<mute::ceil_div((MN / 4) * (K / 32), 128)>>>,
  Stride<Stride<_4, Int<MN * 32>, _0>,
         Stride<_1, Int<MN>, Int<128 * 4 * 32>>>>;
} // namespace MP32::SQMMA

template <TCE::Major MajorMode, bool IsMtF4>
struct MakeTensor<MP32::SQMMA::smem_desc<MajorMode, IsMtF4>>
{
  template <class TEngine, class TLayout>
  MUTE_HOST_DEVICE constexpr auto
  operator()(Tensor<TEngine,TLayout> const& smem_tensor)
  {
    static_assert(is_smem<TEngine>::value, "Expected SMEM Tensor to construct a SQMMA Desc Tensor");
    return make_tensor(MP32::SQMMA::DescriptorIterator{MP32::SQMMA::make_sqmma_desc<MajorMode, IsMtF4>(tensor<0>(smem_tensor))},
                       replace<0>(recast<uint8_t const>(smem_tensor).layout(), Layout<_1,_0>{}));
  }
};

namespace MP32::BLOCKSCALED {

// SF tensors retain the full logical MN-by-K shape; their source layout
// represents the scale-vector broadcast with a zero stride.
namespace detail {

template <class SFATensor, class TiledMma>
MUTE_HOST_DEVICE constexpr auto
thrfrg_SFA(SFATensor&& sfatensor, TiledMma const& mma)
{
  MUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});

  using AtomShape_MNK  = typename TiledMma::AtomShape_MNK;
  using AtomLayoutSFA  = typename TiledMma::Atom::Traits::SFALayout;

  auto t_tensor = logical_divide(
      sfatensor,
      make_tile(mma.template permutation_mnk<0>(),
                mma.template permutation_mnk<2>()));
  auto atom_tensor = zipped_divide(
      t_tensor,
      make_tile(make_layout(size<0>(AtomShape_MNK{})),
                make_layout(size<2>(AtomShape_MNK{}))));
  auto tv_tensor = atom_tensor.compose(AtomLayoutSFA{}, _);
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  return zipped_divide(
      tv_tensor,
      make_tile(_, make_tile(make_layout(size<1>(thr_layout_vmnk)),
                             make_layout(size<3>(thr_layout_vmnk)))));
}

template <class SFBTensor, class TiledMma>
MUTE_HOST_DEVICE constexpr auto
thrfrg_SFB(SFBTensor&& sfbtensor, TiledMma const& mma)
{
  MUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});

  using AtomShape_MNK  = typename TiledMma::AtomShape_MNK;
  using AtomLayoutSFB  = typename TiledMma::Atom::Traits::SFBLayout;

  auto t_tensor = logical_divide(
      sfbtensor,
      make_tile(mma.template permutation_mnk<1>(),
                mma.template permutation_mnk<2>()));
  auto atom_tensor = zipped_divide(
      t_tensor,
      make_tile(make_layout(size<1>(AtomShape_MNK{})),
                make_layout(size<2>(AtomShape_MNK{}))));
  auto tv_tensor = atom_tensor.compose(AtomLayoutSFB{}, _);
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  return zipped_divide(
      tv_tensor,
      make_tile(_, make_tile(make_layout(size<2>(thr_layout_vmnk)),
                             make_layout(size<3>(thr_layout_vmnk)))));
}

} // namespace detail

template <class SFATensor, class ThrMma>
MUTE_HOST_DEVICE constexpr auto
partition_SFA(SFATensor&& sfatensor, ThrMma const& thread_mma)
{
  auto thr_tensor = make_tensor(
      static_cast<SFATensor&&>(sfatensor).data(),
      detail::thrfrg_SFA(sfatensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk = make_coord(get<0>(thr_vmnk),
                            make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
  return thr_tensor(thr_vmk,
                    make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
}

template <class SFBTensor, class ThrMma>
MUTE_HOST_DEVICE constexpr auto
partition_SFB(SFBTensor&& sfbtensor, ThrMma const& thread_mma)
{
  auto thr_tensor = make_tensor(
      static_cast<SFBTensor&&>(sfbtensor).data(),
      detail::thrfrg_SFB(sfbtensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vnk = make_coord(get<0>(thr_vmnk),
                            make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
  return thr_tensor(thr_vnk,
                    make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
}

template <class SFATensor, class ThrMma>
MUTE_HOST_DEVICE constexpr auto
partition_fragment_SFA(SFATensor&& sfatensor, ThrMma const& thread_mma)
{
  using ValTypeSFA = typename ThrMma::Atom::Traits::ValTypeSFA;
  return make_fragment_like<ValTypeSFA>(
      partition_SFA(static_cast<SFATensor&&>(sfatensor), thread_mma));
}

template <class SFBTensor, class ThrMma>
MUTE_HOST_DEVICE constexpr auto
partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma const& thread_mma)
{
  using ValTypeSFB = typename ThrMma::Atom::Traits::ValTypeSFB;
  return make_fragment_like<ValTypeSFB>(
      partition_SFB(static_cast<SFBTensor&&>(sfbtensor), thread_mma));
}

template <class TiledMma>
MUTE_HOST_DEVICE constexpr auto
get_layoutSFA_TV(TiledMma const& mma)
{
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto ref_A = make_layout(
      make_shape(mma.template tile_size_mnk<0>(),
                 mma.template tile_size_mnk<2>()));
  auto atile = make_tile(
      _, make_tile(make_layout(
                       make_shape(size<1>(thr_layout_vmnk),
                                  size<2>(thr_layout_vmnk)),
                       make_stride(_1{}, _0{})),
                   _));
  auto thridx_2_thrid = composition(
      make_layout(make_shape(size(thr_layout_vmnk), _1{}),
                  make_stride(_1{}, _0{})),
      right_inverse(make_layout(thr_layout_vmnk,
                                complement(thr_layout_vmnk))));
  return detail::thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
}

template <class TiledMma>
MUTE_HOST_DEVICE constexpr auto
get_layoutSFB_TV(TiledMma const& mma)
{
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
  auto ref_B = make_layout(
      make_shape(mma.template tile_size_mnk<1>(),
                 mma.template tile_size_mnk<2>()));
  auto btile = make_tile(
      _, make_tile(make_layout(
                       make_shape(size<1>(thr_layout_vmnk),
                                  size<2>(thr_layout_vmnk)),
                       make_stride(_0{}, _1{})),
                   _));
  auto thridx_2_thrid = composition(
      make_layout(make_shape(size(thr_layout_vmnk), _1{}),
                  make_stride(_1{}, _0{})),
      right_inverse(make_layout(thr_layout_vmnk,
                                complement(thr_layout_vmnk))));
  return detail::thrfrg_SFB(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
}

template <class MMAOp,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
MUTE_HOST_DEVICE constexpr void
mma_unpack(MMA_Traits<MMAOp>         const& traits,
           Tensor<TD, DLayout>            & D,
           Tensor<TA, ALayout>       const& A_zipped,
           Tensor<TB, BLayout>       const& B_zipped,
           Tensor<TC, CLayout>       const& C)
{
  // MMA_Atom::call consumes register fragments. SS operands are represented by
  // register-resident descriptors, while zipped operands also keep SFA/SFB in registers.
  static_assert(is_rmem<TD>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TA>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TB>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TC>::value, "Expected registers in MMA_Atom::call");

  using RegTypeD   = typename remove_extent<typename MMAOp::DRegisters>::type;
  using RegTypeA   = typename remove_extent<typename MMAOp::ARegisters>::type;
  using RegTypeB   = typename remove_extent<typename MMAOp::BRegisters>::type;
  using RegTypeC   = typename remove_extent<typename MMAOp::CRegisters>::type;
  using RegTypeSFA = typename remove_extent<typename MMAOp::SFARegisters>::type;
  using RegTypeSFB = typename remove_extent<typename MMAOp::SFBRegisters>::type;
  using ValTypeSFA = typename MMA_Traits<MMAOp>::ValTypeSFA;
  using ValTypeSFB = typename MMA_Traits<MMAOp>::ValTypeSFB;

  constexpr int RegNumD   = extent<typename MMAOp::DRegisters>::value;
  constexpr int RegNumA   = extent<typename MMAOp::ARegisters>::value;
  constexpr int RegNumB   = extent<typename MMAOp::BRegisters>::value;
  constexpr int RegNumC   = extent<typename MMAOp::CRegisters>::value;
  constexpr int RegNumSFA = extent<typename MMAOp::SFARegisters>::value;
  constexpr int RegNumSFB = extent<typename MMAOp::SFBRegisters>::value;

  auto [A, SFA] = unzip_tensor(A_zipped);
  auto [B, SFB] = unzip_tensor(B_zipped);

  constexpr int SFVecSize = MMAOp::SFVecSize;
  static_assert(sizeof_bits_v<RegTypeSFA> % sizeof_bits_v<ValTypeSFA> == 0);
  static_assert(sizeof_bits_v<RegTypeSFB> % sizeof_bits_v<ValTypeSFB> == 0);
  constexpr int SFAValuesPerRegister = sizeof_bits_v<RegTypeSFA> / sizeof_bits_v<ValTypeSFA>;
  constexpr int SFBValuesPerRegister = sizeof_bits_v<RegTypeSFB> / sizeof_bits_v<ValTypeSFB>;

  MUTE_STATIC_ASSERT_V(size(SFA) == Int<RegNumSFA * SFAValuesPerRegister * SFVecSize>{});
  MUTE_STATIC_ASSERT_V(size(SFB) == Int<RegNumSFB * SFBValuesPerRegister * SFVecSize>{});
  MUTE_STATIC_ASSERT_V(cosize(layout(SFA)) == Int<RegNumSFA * SFAValuesPerRegister>{});
  MUTE_STATIC_ASSERT_V(cosize(layout(SFB)) == Int<RegNumSFB * SFBValuesPerRegister>{});

  Tensor rA = recast<RegTypeA>(A);
  Tensor rB = recast<RegTypeB>(B);
  MUTE_STATIC_ASSERT_V(size(rA) == Int<RegNumA>{});
  MUTE_STATIC_ASSERT_V(size(rB) == Int<RegNumB>{});

  Tensor rSFA = recast<RegTypeSFA>(filter_zeros(SFA));
  Tensor rSFB = recast<RegTypeSFB>(filter_zeros(SFB));
  MUTE_STATIC_ASSERT_V(size(rSFA) == Int<RegNumSFA>{});
  MUTE_STATIC_ASSERT_V(size(rSFB) == Int<RegNumSFB>{});

  if constexpr (is_same<RegTypeD, void>::value) {
    static_assert(is_same<typename TD::value_type, typename TC::value_type>::value,
                  "SQMMA C and D value_type must match.");
    static_assert(is_same<DLayout, CLayout>::value,
                  "SQMMA C and D layouts must match.");

    Tensor rC = recast<RegTypeC>(D);
    MUTE_STATIC_ASSERT_V(size(rC) == Int<RegNumC>{});
    MMAOp::fma(rA.data(), rB.data(), rC.data(),
               rSFA.data(), rSFB.data(), traits.accumulate_);
  }
  else {
    Tensor rD = recast<RegTypeD>(D);
    Tensor rC = recast<RegTypeC>(C);
    MUTE_STATIC_ASSERT_V(size(rD) == Int<RegNumD>{});
    MUTE_STATIC_ASSERT_V(size(rC) == Int<RegNumC>{});
    MMAOp::fma(rD.data(), rA.data(), rB.data(), rC.data(),
               rSFA.data(), rSFB.data());
  }
}

} // namespace MP32::BLOCKSCALED

///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
///// SS SQMMA Traits
///////////////////////////////////////////////////////////////////////////////

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32_32x32_SS<a_type, b_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;

  using FrgTypeA = MP32::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ABLayout< 32,  K>;
  using BLayout   = MP32::SQMMA::ABLayout< 32,  K>;
  using CLayout   = MP32::SQMMA::CLayout<  32, 32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32_32x64_SS<a_type, b_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;

  using FrgTypeA = MP32::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ABLayout< 32,  K>;
  using BLayout   = MP32::SQMMA::ABLayout< 64,  K>;
  using CLayout   = MP32::SQMMA::CLayout<  32, 64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32_32x128_SS<a_type, b_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;

  using FrgTypeA = MP32::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ABLayout< 32,   K>;
  using BLayout   = MP32::SQMMA::ABLayout<128,   K>;
  using CLayout   = MP32::SQMMA::CLayout<  32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32_64x32_SS<a_type, b_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;

  using FrgTypeA = MP32::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ABLayout< 64,  K>;
  using BLayout   = MP32::SQMMA::ABLayout< 32,  K>;
  using CLayout   = MP32::SQMMA::CLayout<  64, 32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32_64x64_SS<a_type, b_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;

  using FrgTypeA = MP32::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ABLayout< 64,  K>;
  using BLayout   = MP32::SQMMA::ABLayout< 64,  K>;
  using CLayout   = MP32::SQMMA::CLayout<  64, 64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32_64x128_SS<a_type, b_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;

  using FrgTypeA = MP32::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ABLayout< 64,   K>;
  using BLayout   = MP32::SQMMA::ABLayout<128,   K>;
  using CLayout   = MP32::SQMMA::CLayout<  64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32_128x32_SS<a_type, b_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;

  using FrgTypeA = MP32::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ABLayout<128,  K>;
  using BLayout   = MP32::SQMMA::ABLayout< 32,  K>;
  using CLayout   = MP32::SQMMA::CLayout< 128, 32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32_128x64_SS<a_type, b_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;

  using FrgTypeA = MP32::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ABLayout<128,  K>;
  using BLayout   = MP32::SQMMA::ABLayout< 64,  K>;
  using CLayout   = MP32::SQMMA::CLayout< 128, 64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32_128x128_SS<a_type, b_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;

  using FrgTypeA = MP32::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ABLayout<128,   K>;
  using BLayout   = MP32::SQMMA::ABLayout<128,   K>;
  using CLayout   = MP32::SQMMA::CLayout< 128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

///////////////////////////////////////////////////////////////////////////////
///// RS SQMMA Traits
///////////////////////////////////////////////////////////////////////////////

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32_32x32_RS<a_type, b_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;

  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ALayout<32, K, sizeof_bits_v<ValTypeA>>;
  using BLayout   = MP32::SQMMA::ABLayout< 32,   K>;
  using CLayout   = MP32::SQMMA::CLayout<  32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32_32x64_RS<a_type, b_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;

  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ALayout<32, K, sizeof_bits_v<ValTypeA>>;
  using BLayout   = MP32::SQMMA::ABLayout< 64,  K>;
  using CLayout   = MP32::SQMMA::CLayout<  32, 64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32_32x128_RS<a_type, b_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;

  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ALayout<32, K, sizeof_bits_v<ValTypeA>>;
  using BLayout   = MP32::SQMMA::ABLayout<128,   K>;
  using CLayout   = MP32::SQMMA::CLayout<  32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32_64x32_RS<a_type, b_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;

  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ALayout<64, K, sizeof_bits_v<ValTypeA>>;
  using BLayout   = MP32::SQMMA::ABLayout< 32,  K>;
  using CLayout   = MP32::SQMMA::CLayout<  64, 32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32_64x64_RS<a_type, b_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;

  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ALayout<64, K, sizeof_bits_v<ValTypeA>>;
  using BLayout   = MP32::SQMMA::ABLayout< 64,  K>;
  using CLayout   = MP32::SQMMA::CLayout<  64, 64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32_64x128_RS<a_type, b_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;

  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ALayout<64, K, sizeof_bits_v<ValTypeA>>;
  using BLayout   = MP32::SQMMA::ABLayout<128,   K>;
  using CLayout   = MP32::SQMMA::CLayout<  64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32_128x32_RS<a_type, b_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;

  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ALayout<128, K, sizeof_bits_v<ValTypeA>>;
  using BLayout   = MP32::SQMMA::ABLayout< 32,  K>;
  using CLayout   = MP32::SQMMA::CLayout< 128, 32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32_128x64_RS<a_type, b_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;

  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ALayout<128, K, sizeof_bits_v<ValTypeA>>;
  using BLayout   = MP32::SQMMA::ABLayout< 64,  K>;
  using CLayout   = MP32::SQMMA::CLayout< 128, 64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32_128x128_RS<a_type, b_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;

  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ALayout<128, K, sizeof_bits_v<ValTypeA>>;
  using BLayout   = MP32::SQMMA::ABLayout<128,   K>;
  using CLayout   = MP32::SQMMA::CLayout< 128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

///////////////////////////////////////////////////////////////////////////////
///// BLOCKSCALED SS SQMMA Traits
///////////////////////////////////////////////////////////////////////////////
template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32::BLOCKSCALED::MP32_32x32_SS<a_type, b_type, sf_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  static_assert(sizeof_bits_v<a_type> <= 8 && sizeof_bits_v<b_type> <= 8, "MP32 block-scaled SQMMA supports types with leq 8bit types");

  using FrgTypeA = MP32::SQMMA::smem_desc<tnspA, sizeof_bits_v<ValTypeA> == 4 && is_same_v<ValTypeSFA, float_ue4m3_t>>;
  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB, sizeof_bits_v<ValTypeB> == 4 && is_same_v<ValTypeSFB, float_ue4m3_t>>;

  using Shape_MNK = Shape<_32, _32, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ABLayout< 32,  K>;
  using BLayout   = MP32::SQMMA::ABLayout< 32,  K>;
  using CLayout   = MP32::SQMMA::CLayout<  32, 32>;
  using SFALayout = MP32::SQMMA::SFLayout< 32,  K>;
  using SFBLayout = MP32::SQMMA::SFLayout< 32,  K>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32::BLOCKSCALED::MP32_32x64_SS<a_type, b_type, sf_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  static_assert(sizeof_bits_v<a_type> <= 8 && sizeof_bits_v<b_type> <= 8, "MP32 block-scaled SQMMA supports types with leq 8bit types");

  using FrgTypeA = MP32::SQMMA::smem_desc<tnspA, sizeof_bits_v<ValTypeA> == 4 && is_same_v<ValTypeSFA, float_ue4m3_t>>;
  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB, sizeof_bits_v<ValTypeB> == 4 && is_same_v<ValTypeSFB, float_ue4m3_t>>;

  using Shape_MNK = Shape<_32, _64, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ABLayout< 32,  K>;
  using BLayout   = MP32::SQMMA::ABLayout< 64,  K>;
  using CLayout   = MP32::SQMMA::CLayout<  32, 64>;
  using SFALayout = MP32::SQMMA::SFLayout< 32,  K>;
  using SFBLayout = MP32::SQMMA::SFLayout< 64,  K>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32::BLOCKSCALED::MP32_32x128_SS<a_type, b_type, sf_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  static_assert(sizeof_bits_v<a_type> <= 8 && sizeof_bits_v<b_type> <= 8, "MP32 block-scaled SQMMA supports types with leq 8bit types");

  using FrgTypeA = MP32::SQMMA::smem_desc<tnspA, sizeof_bits_v<ValTypeA> == 4 && is_same_v<ValTypeSFA, float_ue4m3_t>>;
  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB, sizeof_bits_v<ValTypeB> == 4 && is_same_v<ValTypeSFB, float_ue4m3_t>>;

  using Shape_MNK = Shape<_32, _128, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ABLayout< 32,   K>;
  using BLayout   = MP32::SQMMA::ABLayout<128,   K>;
  using CLayout   = MP32::SQMMA::CLayout<  32, 128>;
  using SFALayout = MP32::SQMMA::SFLayout< 32,   K>;
  using SFBLayout = MP32::SQMMA::SFLayout<128,   K>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32::BLOCKSCALED::MP32_64x32_SS<a_type, b_type, sf_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  static_assert(sizeof_bits_v<a_type> <= 8 && sizeof_bits_v<b_type> <= 8, "MP32 block-scaled SQMMA supports types with leq 8bit types");

  using FrgTypeA = MP32::SQMMA::smem_desc<tnspA, sizeof_bits_v<ValTypeA> == 4 && is_same_v<ValTypeSFA, float_ue4m3_t>>;
  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB, sizeof_bits_v<ValTypeB> == 4 && is_same_v<ValTypeSFB, float_ue4m3_t>>;

  using Shape_MNK = Shape<_64, _32, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ABLayout< 64,  K>;
  using BLayout   = MP32::SQMMA::ABLayout< 32,  K>;
  using CLayout   = MP32::SQMMA::CLayout<  64, 32>;
  using SFALayout = MP32::SQMMA::SFLayout< 64,  K>;
  using SFBLayout = MP32::SQMMA::SFLayout< 32,  K>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32::BLOCKSCALED::MP32_64x64_SS<a_type, b_type, sf_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  static_assert(sizeof_bits_v<a_type> <= 8 && sizeof_bits_v<b_type> <= 8, "MP32 block-scaled SQMMA supports types with leq 8bit types");

  using FrgTypeA = MP32::SQMMA::smem_desc<tnspA, sizeof_bits_v<ValTypeA> == 4 && is_same_v<ValTypeSFA, float_ue4m3_t>>;
  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB, sizeof_bits_v<ValTypeB> == 4 && is_same_v<ValTypeSFB, float_ue4m3_t>>;

  using Shape_MNK = Shape<_64, _64, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ABLayout< 64,  K>;
  using BLayout   = MP32::SQMMA::ABLayout< 64,  K>;
  using CLayout   = MP32::SQMMA::CLayout<  64, 64>;
  using SFALayout = MP32::SQMMA::SFLayout< 64,  K>;
  using SFBLayout = MP32::SQMMA::SFLayout< 64,  K>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32::BLOCKSCALED::MP32_64x128_SS<a_type, b_type, sf_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  static_assert(sizeof_bits_v<a_type> <= 8 && sizeof_bits_v<b_type> <= 8, "MP32 block-scaled SQMMA supports types with leq 8bit types");

  using FrgTypeA = MP32::SQMMA::smem_desc<tnspA, sizeof_bits_v<ValTypeA> == 4 && is_same_v<ValTypeSFA, float_ue4m3_t>>;
  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB, sizeof_bits_v<ValTypeB> == 4 && is_same_v<ValTypeSFB, float_ue4m3_t>>;

  using Shape_MNK = Shape<_64, _128, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ABLayout< 64,   K>;
  using BLayout   = MP32::SQMMA::ABLayout<128,   K>;
  using CLayout   = MP32::SQMMA::CLayout<  64, 128>;
  using SFALayout = MP32::SQMMA::SFLayout< 64,   K>;
  using SFBLayout = MP32::SQMMA::SFLayout<128,   K>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32::BLOCKSCALED::MP32_128x32_SS<a_type, b_type, sf_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  static_assert(sizeof_bits_v<a_type> <= 8 && sizeof_bits_v<b_type> <= 8, "MP32 block-scaled SQMMA supports types with leq 8bit types");

  using FrgTypeA = MP32::SQMMA::smem_desc<tnspA, sizeof_bits_v<ValTypeA> == 4 && is_same_v<ValTypeSFA, float_ue4m3_t>>;
  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB, sizeof_bits_v<ValTypeB> == 4 && is_same_v<ValTypeSFB, float_ue4m3_t>>;

  using Shape_MNK = Shape<_128, _32, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ABLayout<128,  K>;
  using BLayout   = MP32::SQMMA::ABLayout< 32,  K>;
  using CLayout   = MP32::SQMMA::CLayout< 128, 32>;
  using SFALayout = MP32::SQMMA::SFLayout<128,  K>;
  using SFBLayout = MP32::SQMMA::SFLayout< 32,  K>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32::BLOCKSCALED::MP32_128x64_SS<a_type, b_type, sf_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  static_assert(sizeof_bits_v<a_type> <= 8 && sizeof_bits_v<b_type> <= 8, "MP32 block-scaled SQMMA supports types with leq 8bit types");

  using FrgTypeA = MP32::SQMMA::smem_desc<tnspA, sizeof_bits_v<ValTypeA> == 4 && is_same_v<ValTypeSFA, float_ue4m3_t>>;
  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB, sizeof_bits_v<ValTypeB> == 4 && is_same_v<ValTypeSFB, float_ue4m3_t>>;

  using Shape_MNK = Shape<_128, _64, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ABLayout<128,  K>;
  using BLayout   = MP32::SQMMA::ABLayout< 64,  K>;
  using CLayout   = MP32::SQMMA::CLayout< 128, 64>;
  using SFALayout = MP32::SQMMA::SFLayout<128,  K>;
  using SFBLayout = MP32::SQMMA::SFLayout< 64,  K>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32::BLOCKSCALED::MP32_128x128_SS<a_type, b_type, sf_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  static_assert(sizeof_bits_v<a_type> <= 8 && sizeof_bits_v<b_type> <= 8, "MP32 block-scaled SQMMA supports types with leq 8bit types");

  using FrgTypeA = MP32::SQMMA::smem_desc<tnspA, sizeof_bits_v<ValTypeA> == 4 && is_same_v<ValTypeSFA, float_ue4m3_t>>;
  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB, sizeof_bits_v<ValTypeB> == 4 && is_same_v<ValTypeSFB, float_ue4m3_t>>;

  using Shape_MNK = Shape<_128, _128, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ABLayout<128,   K>;
  using BLayout   = MP32::SQMMA::ABLayout<128,   K>;
  using CLayout   = MP32::SQMMA::CLayout< 128, 128>;
  using SFALayout = MP32::SQMMA::SFLayout<128,   K>;
  using SFBLayout = MP32::SQMMA::SFLayout<128,   K>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

///////////////////////////////////////////////////////////////////////////////
///// BLOCKSCALED RS SQMMA Traits
///////////////////////////////////////////////////////////////////////////////

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32::BLOCKSCALED::MP32_32x32_RS<a_type, b_type, sf_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  static_assert(sizeof_bits_v<a_type> <= 8 && sizeof_bits_v<b_type> <= 8, "MP32 block-scaled SQMMA supports types with leq 8bit types");

  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB, sizeof_bits_v<ValTypeB> == 4 && is_same_v<ValTypeSFB, float_ue4m3_t>>;

  using Shape_MNK = Shape<_32, _32, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ALayout<32, K, sizeof_bits_v<ValTypeA>>;
  using BLayout   = MP32::SQMMA::ABLayout< 32,  K>;
  using CLayout   = MP32::SQMMA::CLayout<  32, 32>;
  using SFALayout = MP32::SQMMA::SFLayout< 32,  K>;
  using SFBLayout = MP32::SQMMA::SFLayout< 32,  K>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32::BLOCKSCALED::MP32_32x64_RS<a_type, b_type, sf_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  static_assert(sizeof_bits_v<a_type> <= 8 && sizeof_bits_v<b_type> <= 8, "MP32 block-scaled SQMMA supports types with leq 8bit types");

  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB, sizeof_bits_v<ValTypeB> == 4 && is_same_v<ValTypeSFB, float_ue4m3_t>>;

  using Shape_MNK = Shape<_32, _64, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ALayout<32, K, sizeof_bits_v<ValTypeA>>;
  using BLayout   = MP32::SQMMA::ABLayout< 64,  K>;
  using CLayout   = MP32::SQMMA::CLayout<  32, 64>;
  using SFALayout = MP32::SQMMA::SFLayout< 32,  K>;
  using SFBLayout = MP32::SQMMA::SFLayout< 64,  K>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32::BLOCKSCALED::MP32_32x128_RS<a_type, b_type, sf_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  static_assert(sizeof_bits_v<a_type> <= 8 && sizeof_bits_v<b_type> <= 8, "MP32 block-scaled SQMMA supports types with leq 8bit types");

  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB, sizeof_bits_v<ValTypeB> == 4 && is_same_v<ValTypeSFB, float_ue4m3_t>>;

  using Shape_MNK = Shape<_32, _128, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ALayout<32, K, sizeof_bits_v<ValTypeA>>;
  using BLayout   = MP32::SQMMA::ABLayout<128,   K>;
  using CLayout   = MP32::SQMMA::CLayout<  32, 128>;
  using SFALayout = MP32::SQMMA::SFLayout< 32,   K>;
  using SFBLayout = MP32::SQMMA::SFLayout<128,   K>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32::BLOCKSCALED::MP32_64x32_RS<a_type, b_type, sf_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  static_assert(sizeof_bits_v<a_type> <= 8 && sizeof_bits_v<b_type> <= 8, "MP32 block-scaled SQMMA supports types with leq 8bit types");

  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB, sizeof_bits_v<ValTypeB> == 4 && is_same_v<ValTypeSFB, float_ue4m3_t>>;

  using Shape_MNK = Shape<_64, _32, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ALayout<64, K, sizeof_bits_v<ValTypeA>>;
  using BLayout   = MP32::SQMMA::ABLayout< 32,  K>;
  using CLayout   = MP32::SQMMA::CLayout<  64, 32>;
  using SFALayout = MP32::SQMMA::SFLayout< 64,  K>;
  using SFBLayout = MP32::SQMMA::SFLayout< 32,  K>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32::BLOCKSCALED::MP32_64x64_RS<a_type, b_type, sf_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  static_assert(sizeof_bits_v<a_type> <= 8 && sizeof_bits_v<b_type> <= 8, "MP32 block-scaled SQMMA supports types with leq 8bit types");

  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB, sizeof_bits_v<ValTypeB> == 4 && is_same_v<ValTypeSFB, float_ue4m3_t>>;

  using Shape_MNK = Shape<_64, _64, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ALayout<64, K, sizeof_bits_v<ValTypeA>>;
  using BLayout   = MP32::SQMMA::ABLayout< 64,  K>;
  using CLayout   = MP32::SQMMA::CLayout<  64, 64>;
  using SFALayout = MP32::SQMMA::SFLayout< 64,  K>;
  using SFBLayout = MP32::SQMMA::SFLayout< 64,  K>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32::BLOCKSCALED::MP32_64x128_RS<a_type, b_type, sf_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  static_assert(sizeof_bits_v<a_type> <= 8 && sizeof_bits_v<b_type> <= 8, "MP32 block-scaled SQMMA supports types with leq 8bit types");

  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB, sizeof_bits_v<ValTypeB> == 4 && is_same_v<ValTypeSFB, float_ue4m3_t>>;

  using Shape_MNK = Shape<_64, _128, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ALayout<64, K, sizeof_bits_v<ValTypeA>>;
  using BLayout   = MP32::SQMMA::ABLayout<128,   K>;
  using CLayout   = MP32::SQMMA::CLayout<  64, 128>;
  using SFALayout = MP32::SQMMA::SFLayout< 64,   K>;
  using SFBLayout = MP32::SQMMA::SFLayout<128,   K>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32::BLOCKSCALED::MP32_128x32_RS<a_type, b_type, sf_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  static_assert(sizeof_bits_v<a_type> <= 8 && sizeof_bits_v<b_type> <= 8, "MP32 block-scaled SQMMA supports types with leq 8bit types");

  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB, sizeof_bits_v<ValTypeB> == 4 && is_same_v<ValTypeSFB, float_ue4m3_t>>;

  using Shape_MNK = Shape<_128, _32, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ALayout<128, K, sizeof_bits_v<ValTypeA>>;
  using BLayout   = MP32::SQMMA::ABLayout< 32,  K>;
  using CLayout   = MP32::SQMMA::CLayout< 128, 32>;
  using SFALayout = MP32::SQMMA::SFLayout<128,  K>;
  using SFBLayout = MP32::SQMMA::SFLayout< 32,  K>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32::BLOCKSCALED::MP32_128x64_RS<a_type, b_type, sf_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  static_assert(sizeof_bits_v<a_type> <= 8 && sizeof_bits_v<b_type> <= 8, "MP32 block-scaled SQMMA supports types with leq 8bit types");

  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB, sizeof_bits_v<ValTypeB> == 4 && is_same_v<ValTypeSFB, float_ue4m3_t>>;

  using Shape_MNK = Shape<_128, _64, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ALayout<128, K, sizeof_bits_v<ValTypeA>>;
  using BLayout   = MP32::SQMMA::ABLayout< 64,  K>;
  using CLayout   = MP32::SQMMA::CLayout< 128, 64>;
  using SFALayout = MP32::SQMMA::SFLayout<128,  K>;
  using SFBLayout = MP32::SQMMA::SFLayout< 64,  K>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

template <class a_type, class b_type, class sf_type, int K,
          TCE::Major tnspA,
          TCE::Major tnspB,
          MP31::SQMMA::ScaleIn scaleA,
          MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP32::BLOCKSCALED::MP32_128x128_RS<a_type, b_type, sf_type, K, tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeA   = a_type;
  using ValTypeB   = b_type;
  using ValTypeC   = MP32::detail::deduce_tce_acc_type_t<a_type, b_type>;
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;

  static_assert(sizeof_bits_v<a_type> <= 8 && sizeof_bits_v<b_type> <= 8, "MP32 block-scaled SQMMA supports types with leq 8bit types");

  using FrgTypeB = MP32::SQMMA::smem_desc<tnspB, sizeof_bits_v<ValTypeB> == 4 && is_same_v<ValTypeSFB, float_ue4m3_t>>;

  using Shape_MNK = Shape<_128, _128, Int<K>>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP32::SQMMA::ALayout<128, K, sizeof_bits_v<ValTypeA>>;
  using BLayout   = MP32::SQMMA::ABLayout<128,   K>;
  using CLayout   = MP32::SQMMA::CLayout< 128, 128>;
  using SFALayout = MP32::SQMMA::SFLayout<128,   K>;
  using SFBLayout = MP32::SQMMA::SFLayout<128,   K>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

} // namespace mute
