#pragma once
#include <mute/atom/copy_traits_mp32_tme.hpp>

#include "mate/mega_moe/prefill/fp4_fp4/ss_prefill_common.hpp"

namespace mega_moe::prefill::mp32::fp4_fp4 {

// Each lane packs eight adjacent BF16 outputs into one aligned vector store.
inline constexpr int kStage2PermutationStride = sizeof(int4) / sizeof(mutlass::bfloat16_t);

template <class Mma>
struct Stage2OutputMma;
template <class Atom, class AtomLayout, class OldPermutation>
struct Stage2OutputMma<TiledMMA<Atom, AtomLayout, OldPermutation>> {
  using Original                         = TiledMMA<Atom, AtomLayout, OldPermutation>;
  static constexpr int kNThreadSpan      = decltype(MP32::detail::deduce_mma_n_thread_span<Atom>())::value;
  static constexpr int kTileN            = decltype(size<1>(tile_shape(Original{})))::value;
  static constexpr int kPermutationBlock = kNThreadSpan * kStage2PermutationStride;
  static_assert(kTileN % kPermutationBlock == 0);
  using NPermutation = Layout<Shape<Int<kNThreadSpan>, Int<kStage2PermutationStride>, Int<kTileN / kPermutationBlock>>,
                              Stride<Int<kStage2PermutationStride>, _1, Int<kPermutationBlock>>>;
  using type         = TiledMMA<Atom, AtomLayout, decltype(make_tile(_, NPermutation{}, _))>;
};

// Keep the original physical SS descriptors and contiguous SFB loads in mma().
// Only B's TME load and the logical C coordinates use the N permutation.
template <class Base>
struct Stage2OnlineMainloop : Base {
  using TiledMma            = typename Stage2OutputMma<typename Base::TiledMma>::type;
  using TileShape           = typename Base::TileShape;
  using ElementB            = typename Base::ElementB;
  using StrideB             = typename Base::StrideB;
  using SmemLayoutB         = typename Base::SmemLayoutB;
  using GmemTiledCopyB      = typename Base::GmemTiledCopyB;
  using TmeInternalElementB = typename Base::TmeInternalElementB;
  struct Params {
    using TME_A = typename Base::Params::TME_A;
    using TME_B = decltype(make_tme_copy<TmeInternalElementB>(
        GmemTiledCopyB{},
        make_tensor(static_cast<ElementB const*>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        take<0, 2>(SmemLayoutB{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        TiledMma{}));
    TME_A                      tme_load_a;
    TME_B                      tme_load_b;
    typename Base::ScaleInputs scales;
  };
  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const&             problem,
                                        typename Base::Arguments const& args,
                                        void*                           workspace) {
    auto original           = Base::to_underlying_arguments(problem, args, workspace);
    auto [m, n, k, batches] = append<4>(problem, 1);
    auto b                  = make_tensor(args.ptr_B, make_layout(make_shape(n, k, batches), args.dB));
    auto load_b             = make_tme_copy<TmeInternalElementB>(GmemTiledCopyB{},
                                                     b,
                                                     take<0, 2>(SmemLayoutB{}),
                                                     make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
                                                     TiledMma{});
    return {original.tme_load_a, load_b, original.scales};
  }
  template <class ProblemShape>
  MUTLASS_DEVICE auto load_init(ProblemShape const& problem, Params const& params) const {
    auto [m, n, k, batches] = problem;
    auto a                  = params.tme_load_a.get_tme_tensor(make_shape(m, k, batches));
    auto b                  = params.tme_load_b.get_tme_tensor(make_shape(n, k, batches));
    auto tiled_a            = local_tile(a, TileShape{}, make_coord(_, _, _), Step<_1, Underscore, _1>{});
    auto tiled_b            = local_tile(b, TileShape{}, make_coord(_, _, _), Step<Underscore, _1, _1>{});
    return make_tuple(tiled_a, tiled_b, params.scales);
  }
};

template <int N, int K, int TileN = 256, int TileK = 256>
struct Stage2OnlinePool : SsPoolMainloop<N, K, TileN, TileK> {
  using Original   = SsPoolMainloop<N, K, TileN, TileK>;
  using Collective = Stage2OnlineMainloop<typename Original::Collective>;
  using Params     = typename Collective::Params;
  using Pipeline   = typename Original::Pipeline;
  using State      = typename Original::State;
  using Impl       = typename Original::Impl;
  static Params params(int pool, int groups, void const* a, void const* sa, void const* b, void const* sb) {
    typename Collective::Arguments args{static_cast<typename Collective::ElementA const*>(a),
                                        typename Collective::StrideA{K, _1{}, 0},
                                        static_cast<typename Collective::ElementB const*>(b),
                                        typename Collective::StrideB{K, _1{}, int64_t(N) * K},
                                        static_cast<typename Collective::ElementSFA const*>(sa),
                                        static_cast<typename Collective::ElementSFB const*>(sb)};
    return Collective::to_underlying_arguments(make_shape(pool, N, K, groups), args, nullptr);
  }
  template <class Inputs, class Coord>
  MUTLASS_DEVICE static void load(
      Inputs const& inputs, Coord const& coord, char* smem, Params const& params, Pipeline& pipeline, State& write) {
    auto& storage  = *reinterpret_cast<typename Collective::SharedStorage*>(smem);
    auto  sa       = make_tensor(make_smem_ptr(storage.smem_A.begin()), typename Collective::SmemLayoutA{});
    auto  sb       = make_tensor(make_smem_ptr(storage.smem_B.begin()), typename Collective::SmemLayoutB{});
    auto  ca       = params.tme_load_a.get_slice(0);
    auto  cb       = params.tme_load_b.get_slice(0);
    auto [m, n, e] = coord;
    auto ga        = ca.partition_S(get<0>(inputs)(_, _, m, _, 0));
    auto gb        = cb.partition_S(get<1>(inputs)(_, _, n, _, e));
    auto da        = ca.partition_D(sa);
    auto db        = cb.partition_D(sb);
    MUTLASS_PRAGMA_NO_UNROLL
    for (int k = 0; k < K / Impl::kBlockK; ++k) {
      pipeline.producer_acquire(write);
      auto bar   = pipeline.producer_get_barrier_id(write);
      int  stage = write.index();
      copy(params.tme_load_a.with(bar), ga(_, _, _, k), da(_, _, _, stage));
      copy(params.tme_load_b.with(bar), gb(_, _, _, k), db(_, _, _, stage));
      ++write;
    }
  }
};
}  // namespace mega_moe::prefill::mp32::fp4_fp4
