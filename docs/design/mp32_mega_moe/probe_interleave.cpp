#include <cstdio>
#include <mute/atom/mma_traits_mp32_sqmma.hpp>
#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/builders/mp3x_common.inl>

using namespace mute;

template <class TM>
void dump(const char* name, TM tm, bool swapped) {
  auto identity = make_identity_tensor(take<0, 2>(tile_shape(tm)));
  std::printf("TYPE %s atom=%d,%d,%d threads=%d swapped=%d\n",
              name,
              int(size<0>(typename TM::AtomShape_MNK{})),
              int(size<1>(typename TM::AtomShape_MNK{})),
              int(size<2>(typename TM::AtomShape_MNK{})),
              int(size(tm)),
              int(swapped));
  for (int t = 0; t < int(size(tm)); ++t) {
    auto coords = tm.get_thread_slice(t).partition_C(identity);
    for (int i = 0; i < int(size(coords)); ++i) {
      auto c = coords(i);
      int  m = int(get<0>(c)), n = int(get<1>(c));
      std::printf("C %s %d %d %d %d\n", name, t, i, swapped ? n : m, swapped ? m : n);
    }
  }
}

int main() {
  using TS  = Shape<_256, _256, _128>;
  using F8  = mutlass::float_e4m3_t;
  using SF  = mutlass::float_ue8m0_t;
  auto ssop = MP32::SQMMA::ss_op_selector<F8, F8, float, SF, TS>();
  auto ss   = mutlass::gemm::collective::detail::make_sqmma_tiled_mma<TS, decltype(ssop)>();
  dump("SS", ss, false);
  auto rsop        = MP32::SQMMA::rs_op_selector<F8, F8, float, SF, TS, TCE::Major::K, TCE::Major::K, _64, _128>();
  using ROp        = decltype(rsop);
  constexpr int AM = size<0>(typename MMA_Traits<ROp>::Shape_MNK{});
  constexpr int AN = size<1>(typename MMA_Traits<ROp>::Shape_MNK{});
  constexpr int SN = 256 % (2 * AN) == 0 ? 2 * AN : AN;
  auto          sl = make_layout(Shape<Int<256 / AM>, Int<256 / SN>, _1>{}, LayoutRight{});
  auto          rs = make_tiled_mma(rsop, sl, Tile<_256, _256, Underscore>{});
  dump("RS_W4A8", rs, true);
}
