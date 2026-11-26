#pragma once
#include <mute/atom/mma_traits_mp32_sqmma.hpp>
#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
namespace mute {
namespace task_mhc {

template <int  M,
          int  N,
          int  K,
          int  num_warp_m,
          int  num_warp_n,
          bool trans_A,
          bool trans_B,
          bool clear_accum,
          typename A_type_raw,
          typename B_type_raw,
          typename C_type_raw>
class GemmTensorOp {
 public:
  using A_type_mute = typename tl::to_mute_type<A_type_raw>::type;
  using B_type_mute = typename tl::to_mute_type<B_type_raw>::type;
  using A_type      = conditional_t<std::is_same<A_type_mute, float>::value, tfloat32_t, A_type_mute>;
  using B_type      = conditional_t<std::is_same<B_type_mute, float>::value, tfloat32_t, B_type_mute>;
  using C_type      = C_type_raw;

  static constexpr bool IsFp8A = std::is_same<A_type_mute, mutlass::float_e4m3_t>::value ||
                                 std::is_same<A_type_mute, mutlass::float_e5m2_t>::value;
  static constexpr bool IsFp8B = std::is_same<B_type_mute, mutlass::float_e4m3_t>::value ||
                                 std::is_same<B_type_mute, mutlass::float_e5m2_t>::value;
  static constexpr bool IsFp8SQMMA = IsFp8A && IsFp8B;

  static constexpr TCE::Major SqmmaMajorA = IsFp8SQMMA ? TCE::Major::K : (trans_A ? TCE::Major::MN : TCE::Major::K);
  static constexpr TCE::Major SqmmaMajorB = IsFp8SQMMA ? TCE::Major::K : (trans_B ? TCE::Major::K : TCE::Major::MN);

  // Tile handled by one squad warp
  using AtomShape_MNK = Shape<Int<M / (num_warp_m / 4)>, Int<N / num_warp_n>, Int<K>>;
  using TileShape_MNK = Shape<Int<M>, Int<N>, Int<K>>;

  using SqmmaOp     = mute::MP32_32x32_SS<mutlass::bfloat16_t, mutlass::bfloat16_t, 64, TCE::Major::K, TCE::Major::K>;
  using SqmmaTraits = MMA_Traits<SqmmaOp>;
  using InstructionShape_MNK = typename SqmmaTraits::Shape_MNK;

  static_assert(size<0>(TileShape_MNK{}) % size<0>(InstructionShape_MNK{}) == 0, "TileM must align to SQMMA M.");
  static_assert(size<1>(TileShape_MNK{}) % size<1>(InstructionShape_MNK{}) == 0, "TileN must align to SQMMA N.");
  static_assert(size<2>(TileShape_MNK{}) % size<2>(InstructionShape_MNK{}) == 0, "TileK must align to SQMMA K.");

  using AtomLayout = Layout<Shape<Int<num_warp_m / 4>, Int<num_warp_n>, _1>>;

  using TiledMma = decltype(make_tiled_mma(SqmmaOp{}, AtomLayout{}));

  using SmemLayoutAtomA =
      decltype(mutlass::gemm::collective::detail::ss_smem_selector_A<SqmmaMajorA, A_type, SqmmaOp, TileShape_MNK>());
  using SmemLayoutAtomB =
      decltype(mutlass::gemm::collective::detail::ss_smem_selector_B<SqmmaMajorB, B_type, SqmmaOp, TileShape_MNK>());

  using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, Shape<Int<M>, Int<K>, _1>{}));
  using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, Shape<Int<N>, Int<K>, _1>{}));

  static_assert(num_warp_m % 4 == 0, "num_warp_m must be a multiple of 4 for sqmma");

  template <int wg_wait = 0>
  static TL_DEVICE void body(A_type_raw *pA, B_type_raw *pB, C_type_raw *pC, int thread_idx_base = 0) {
    // Fragment layouts are bound to the local warp-specialized thread window.
    const int tid = threadIdx.x - thread_idx_base;
    Tensor    sA  = make_tensor(make_smem_ptr(reinterpret_cast<A_type *>(pA)), SmemLayoutA{});
    Tensor    sB  = make_tensor(make_smem_ptr(reinterpret_cast<B_type *>(pB)), SmemLayoutB{});
    TiledMma  tiled_mma;
    auto      thr_mma = tiled_mma.get_thread_slice(tid);

    // Allocate registers for pipelining
    Tensor tCsA = thr_mma.partition_A(sA);  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);  // (MMA,MMA_N,MMA_K,PIPE)

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);

    Tensor acc = make_tensor(make_rmem_ptr(reinterpret_cast<C_type *>(pC)),
                             partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));

    if constexpr (clear_accum) {
      tiled_mma.accumulate_ = mute::MP31::SQMMA::ScaleOut::Zero;
    }

    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      // (V,M) x (V,N) => (V,M,N)
      gemm(tiled_mma, tCrA(_, _, k_block, 0), tCrB(_, _, k_block, 0), acc);
      tiled_mma.accumulate_ = mute::MP31::SQMMA::ScaleOut::One;
    }

    if constexpr (wg_wait >= 0) {
      warpsquad_wait<wg_wait>();
    }
  }
};

}  // namespace task_mhc
}  // namespace mute
