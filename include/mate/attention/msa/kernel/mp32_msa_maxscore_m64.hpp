#pragma once

#include <musa_burst.h>

#include <cstdint>
#include <mute/tensor.hpp>
#include <type_traits>

namespace mate::attention::msa {

using namespace mute;

// M64 SS core with generic scheduler/TME coordinates and sequence boundaries.
template <class Mainloop, class Scheduler, bool NarrowScoreOffsets = false>
struct Mp32MsaMaxScoreM64 {
  using SharedStorage                  = typename Mainloop::SharedStorage;
  using WorkTile                       = typename Scheduler::WorkTileInfo;
  using ScoreOffset                    = std::conditional_t<NarrowScoreOffsets, uint32_t, uint64_t>;
  static constexpr int ProducerThreads = Mainloop::NumProducerThreads;
  static constexpr int ConsumerWarps   = Mainloop::NumConsumerWarps;
  static constexpr int HeadRatio       = Mainloop::HeadRatio;
  static constexpr int HeadDim         = Mainloop::HeadDim;
  static constexpr int KStages         = Mainloop::KStages;
  static_assert(HeadRatio == 1);
  static_assert(KStages >= 2 && 1 + 2 * KStages <= 63);

  MUTLASS_DEVICE static void anchor(int* acc) {
    asm volatile(""
                 : "+R"(acc[0]),
                   "+R"(acc[4]),
                   "+R"(acc[8]),
                   "+R"(acc[12]),
                   "+R"(acc[32]),
                   "+R"(acc[36]),
                   "+R"(acc[40]),
                   "+R"(acc[44])
                 :
                 : "memory");
  }

  template <class Params>
  MUTLASS_DEVICE static void load_q(Params const& params, SharedStorage& storage, WorkTile const& work) {
    __musa_async_add_trans(1, Mainloop::TmeTransactionBytesQ);
    __musa_async_arrive(1);
    auto const& load = params.mainloop.load_q.tme_load;
    auto        cta  = load.get_slice(0);
    auto        gQ   = Mainloop::make_tme_q_gmem(params.mainloop.load_q, work);
    auto        sQ   = mute::make_tensor(mute::make_smem_ptr(storage.smem_q.data()),
                                mute::take<0, 3>(typename Mainloop::SmemLayoutQTme{}));
    mute::copy(load.with(1), cta.partition_S(gQ), cta.partition_D(sQ));
  }

  template <class CoordinateLayout, class Source, class Destinations>
  struct KTransfer {
    CoordinateLayout coordinates;
    Source           origin;
    Source           source;
    Destinations     destinations;
    int32_t const*   page_row;
    int              kv_len;
    int              page_count;
  };

  template <class Params>
  MUTLASS_DEVICE static int32_t const* page_row(Params const& params, WorkTile const& work) {
    if constexpr (!Mainloop::IsPagedKV) {
      return nullptr;
    } else if constexpr (Mainloop::PageTableMode == PageTableKind::Batched2D) {
      return params.mainloop.args.ptr_page_table +
             int64_t(work.batch_idx) * get<0>(params.mainloop.args.stride_page_table);
    } else {
      return params.mainloop.args.ptr_page_table + int64_t(params.mainloop.args.ptr_kv_page_indptr[work.batch_idx]);
    }
  }

  template <class Transfer>
  MUTLASS_DEVICE static int physical_page(Transfer const& transfer, int page) {
    if constexpr (Mainloop::IsPagedKV) {
      if (page * Mainloop::TileKV < transfer.kv_len && page < transfer.page_count) {
        return transfer.page_row[page];
      }
    }
    return 0;
  }

  template <class CoordinateLayout, class Source>
  MUTE_HOST_DEVICE static auto k_source(CoordinateLayout const& coordinates,
                                        Source const&           origin,
                                        int                     page,
                                        int                     physical_page) {
    auto delta = [&] {
      if constexpr (Mainloop::IsPagedKV) {
        return coordinates(make_coord(_0{}, _0{}, _0{}, physical_page));
      } else {
        return coordinates(make_coord(page * Mainloop::TileKV, _0{}, _0{}, _0{}));
      }
    }();
    return make_tensor(origin.data() + delta, origin.layout());
  }

  template <class Params>
  MUTLASS_DEVICE static auto prepare_k(Params const& params, SharedStorage& storage, WorkTile const& work, int begin) {
    auto const& load        = params.mainloop.load_k.tme_load;
    auto        cta         = load.get_slice(0);
    auto        coordinates = load.get_tme_tensor(params.mainloop.load_k.shape).layout();
    auto        origin_work = Scheduler::make_k_work_tile(work, 0);
    auto        origin      = cta.partition_S(Mainloop::make_tme_k_gmem(params.mainloop.load_k, origin_work, 0));
    auto        sK          = make_tensor(make_smem_ptr(storage.smem_k.data()), typename Mainloop::SmemLayoutK{});
    auto destinations = transform(make_seq<KStages>{}, [&](auto stage) { return cta.partition_D(sK(_, _, stage)); });
    int  physical     = Mainloop::safe_page_idx(params.mainloop, Scheduler::make_k_work_tile(work, begin));
    auto source       = k_source(coordinates, origin, begin, physical);
    return KTransfer<decltype(coordinates), decltype(origin), decltype(destinations)>{
        coordinates,
        origin,
        source,
        destinations,
        page_row(params, work),
        work.kv_len,
        get<1>(params.mainloop.args.shape_page_table)};
  }

  template <int Stage, class Params, class Transfer>
  MUTLASS_DEVICE static void produce(Params const& params, Transfer& transfer, int page, int end, int step, int phase) {
    __musa_async_wait(2 + KStages + Stage, phase);
    __musa_async_add_trans(2 + Stage, Mainloop::TmeTransactionBytesK);
    __musa_async_arrive(2 + Stage);
    auto const& load = params.mainloop.load_k.tme_load;
    mute::copy(load.with(2 + Stage), transfer.source, get<Stage>(transfer.destinations));
    if (page + step < end) {
      int physical    = physical_page(transfer, page + step);
      transfer.source = k_source(transfer.coordinates, transfer.origin, page + step, physical);
      if constexpr (Mainloop::EnableKPrefetch) {
        mute::prefetch(load, transfer.source);
      }
    }
  }

  template <bool Mask>
  MUTLASS_DEVICE static float4 masked_group(float4 v, int group, int limit) {
    if constexpr (Mask) {
      int   col = group * 16;
      float inf = -__builtin_inff();
      if (col > limit) v.x = inf;
      if (col + 4 > limit) v.y = inf;
      if (col + 8 > limit) v.z = inf;
      if (col + 12 > limit) v.w = inf;
    }
    return v;
  }

  template <bool Mask>
  MUTLASS_DEVICE static float2 reduce_local(int* acc, int lane, int limit0, int limit1) {
    float4* groups = reinterpret_cast<float4*>(acc);
    // Immediate columns avoid keeping lane+column coordinates live through MMA.
    int limits[2] = {limit0 - lane % 4, limit1 - lane % 4};
    MUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 4; ++j) {
      MUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < 2; ++r) {
        float4 a          = masked_group<Mask>(groups[r * 8 + j], j, limits[r]);
        float4 b          = masked_group<Mask>(groups[r * 8 + j + 4], j + 4, limits[r]);
        groups[r * 8 + j] = max(a, b);
      }
    }
    MUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 2; ++j) {
      MUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < 2; ++r) {
        groups[r * 8 + j] = max(groups[r * 8 + j], groups[r * 8 + j + 2]);
      }
    }
    float maximum[2];
    MUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < 2; ++r) {
      float4 v    = max(groups[r * 8], groups[r * 8 + 1]);
      float2 pair = max(make_float2(v.x, v.y), make_float2(v.z, v.w));
      maximum[r]  = max(pair.x, pair.y);
    }
    return make_float2(maximum[0], maximum[1]);
  }

  MUTLASS_DEVICE static float2 reduce_wave(float2 partial) {
    float maximum[2] = {partial.x, partial.y};
    MUTLASS_PRAGMA_UNROLL
    for (int delta = 1; delta <= 2; delta *= 2) {
      MUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < 2; ++r) {
        maximum[r] = max(maximum[r], __shfl_xor_sync(0xffffffffu, maximum[r], delta));
      }
    }
    return make_float2(maximum[0], maximum[1]);
  }

  struct ConsumerState {
    uint64_t    row_base;
    ScoreOffset row_delta;
    int         lane;
    int         step;
    int         desc_a;
    bool        valid0;
    bool        valid1;
  };

  MUTLASS_DEVICE static void finish_partial(float2 partial, int page, ConsumerState const& output) {
    uint32_t    byte_offset        = uint32_t(page) * sizeof(float);
    ScoreOffset second_byte_offset = output.row_delta + byte_offset;
    asm volatile("" : "+R"(byte_offset), "+R"(second_byte_offset) : : "memory");
    float2 maxima = reduce_wave(partial);
    // Retire both maxima before predicated stores can reuse their registers.
    asm volatile("" : "+R"(maxima.x), "+R"(maxima.y) : : "memory");
    using GlobalFloat = float __attribute__((address_space(1)));
    if (output.valid0) {
      *reinterpret_cast<GlobalFloat*>(output.row_base + byte_offset) = maxima.x;
    }
    if (output.valid1) {
      *reinterpret_cast<GlobalFloat*>(output.row_base + second_byte_offset) = maxima.y;
    }
  }

  template <int Stage, bool EmitPrevious, bool Mask = false>
  MUTLASS_DEVICE static float2 advance_partial(int                  desc_b,
                                               int                  page,
                                               int                  phase,
                                               int*                 acc,
                                               float2               previous,
                                               ConsumerState const& output,
                                               int                  last_key0 = 0,
                                               int                  last_key1 = 0) {
    __musa_async_wait(2 + Stage, phase);
    int desc_a = output.desc_a;
    __musa_sqmma_m64n128_mma(acc, &desc_a, &desc_b, acc, 0, 1, 0, 0, 0, 0, HeadDim, 5);
    __musa_tce_commit_group();
    if constexpr (EmitPrevious) {
      // Only old scalar maxima survive; the new accumulator remains unread until wait0.
      finish_partial(previous, page - output.step, output);
    }
    __musa_tce_wait_group(0);
    __syncwarp();
    __musa_async_arrive_none_phaseid(2 + KStages + Stage);
    anchor(acc);
    float2 partial = reduce_local<Mask>(acc, output.lane, last_key0 - page * 128, last_key1 - page * 128);
    asm volatile("" : "+R"(partial.x), "+R"(partial.y) : : "memory");
    return partial;
  }

  template <class Params>
  MUTLASS_DEVICE static void consumer(
      Params const& params, SharedStorage& storage, WorkTile const& work, int begin, int full_end, int end, int step) {
    int         consumer_idx = int(threadIdx.x) - ProducerThreads;
    int         lane         = consumer_idx % 128;
    int         squad        = consumer_idx / 128;
    int         packed_row0  = squad * 64 + lane / 4;
    int         token0       = packed_row0 / HeadRatio;
    int         token1       = (packed_row0 + 32) / HeadRatio;
    int         head0        = work.head_kv * HeadRatio + packed_row0 % HeadRatio;
    auto const& args         = params.mainloop.args;
    int64_t     offset0      = ((int64_t(work.q_abs_begin) + token0) * args.num_qo_heads + head0) * args.max_k_tiles;
    uint64_t    row_base     = reinterpret_cast<uint64_t>(args.ptr_max_score + offset0);
    ScoreOffset row_delta    = static_cast<ScoreOffset>(args.score_row_pair_bytes);
    asm volatile("" : "+R"(row_base) : : "memory");
    bool valid0 = lane % 4 == 0 && token0 < work.q_count;
    bool valid1 = lane % 4 == 0 && token1 < work.q_count;
    __musa_async_wait(1, 0);
    auto sQ = mute::make_tensor(mute::make_smem_ptr(storage.smem_q.data()), typename Mainloop::SmemLayoutQ{})(
        _, _, mute::_0{});
    auto q_squad =
        mute::local_tile(sQ, mute::make_shape(mute::_64{}, mute::Int<HeadDim>{}), mute::make_coord(squad, mute::_0{}));
    int desc_a = mute::MP32::SQMMA::make_sqmma_desc<mute::TCE::Major::K>(
        mute::coalesce(q_squad, mute::Shape<mute::_1, mute::_1>{}));
    auto          sK = mute::make_tensor(mute::make_smem_ptr(storage.smem_k.data()), typename Mainloop::SmemLayoutK{});
    auto          descriptors = transform(make_seq<KStages>{}, [&](auto stage) {
      int desc = mute::MP32::SQMMA::make_sqmma_desc<mute::TCE::Major::K>(
          mute::coalesce(sK(_, _, stage), mute::Shape<mute::_1, mute::_1>{}));
      return desc;
    });
    int           page        = begin;
    int           visits      = 0;
    ConsumerState output{row_base, row_delta, lane, step, desc_a, valid0, valid1};
    if (page < full_end) {
      alignas(16) int acc[64];
      float2 partial = advance_partial<0, false>(get<0>(descriptors), page, 0, acc, make_float2(0.f, 0.f), output);
      page += step;
      ++visits;
      // Priming consumed visit0, so each complete ring starts at visit1.
      MUTLASS_PRAGMA_NO_UNROLL
      for (; page + (2 * KStages - 1) * step < full_end; page += 2 * KStages * step, visits += 2 * KStages) {
        for_each(make_seq<2 * KStages>{}, [&](auto offset) {
          constexpr int Offset = decltype(offset)::value;
          constexpr int Visit  = Offset + 1;
          constexpr int Stage  = Visit % KStages;
          partial              = advance_partial<Stage, true>(
              get<Stage>(descriptors), page + Offset * step, (Visit / KStages) & 1, acc, partial, output);
        });
      }
      MUTLASS_PRAGMA_NO_UNROLL
      for (; page < full_end; page += step, ++visits) {
        int phase = (visits / KStages) & 1;
        for_each(make_seq<KStages>{}, [&](auto stage) {
          constexpr int Stage = decltype(stage)::value;
          if (visits % KStages == Stage) {
            partial = advance_partial<Stage, true>(get<Stage>(descriptors), page, phase, acc, partial, output);
          }
        });
      }
      finish_partial(partial, page - step, output);
    }
    int last_key0 = work.kv_len - 1;
    int last_key1 = last_key0;
    if constexpr (Mainloop::IsCausal) {
      int offset = args.ptr_qo_offset[work.batch_idx];
      last_key0  = min(last_key0, work.q_local_begin + token0 + offset);
      last_key1  = min(last_key1, work.q_local_begin + token1 + offset);
    }
    MUTLASS_PRAGMA_NO_UNROLL
    for (; page < end; page += step, ++visits) {
      int             phase = (visits / KStages) & 1;
      alignas(16) int acc[64];
      float2          partial;
      for_each(make_seq<KStages>{}, [&](auto stage) {
        constexpr int Stage = decltype(stage)::value;
        if (visits % KStages == Stage) {
          partial = advance_partial<Stage, false, true>(
              get<Stage>(descriptors), page, phase, acc, make_float2(0.f, 0.f), output, last_key0, last_key1);
        }
      });
      finish_partial(partial, page, output);
    }
  }

  template <class Params>
  MUTLASS_DEVICE static void run_work(Params const& params, char* smem_buf, WorkTile const& work) {
    if (!Scheduler::is_valid_q_tile(work)) return;
    int end   = Mainloop::k_tile_count(params.mainloop, work);
    int begin = work.k_partition_idx;
    int step  = params.scheduler.k_partitions;
    if (begin >= end) return;
    int            full_end = min(end, Mainloop::first_masked_k_tile(params.mainloop, work));
    SharedStorage& storage  = *reinterpret_cast<SharedStorage*>(smem_buf);
    __musa_async_bar_record(1 + 2 * KStages);
    if (threadIdx.x < 32) {
      __musa_async_init_arrival(1, 1, 0);
      for_each(make_seq<KStages>{}, [&](auto stage) {
        constexpr int Stage = decltype(stage)::value;
        __musa_async_init_arrival(2 + Stage, 1, 0);
        __musa_async_init_arrival(2 + KStages + Stage, ConsumerWarps, 0);
      });
    }
    __syncthreads();
    if (threadIdx.x < ProducerThreads) {
      if (threadIdx.x >= 32) return;
      load_q(params, storage, work);
      auto transfer = prepare_k(params, storage, work, begin);
      MUTLASS_PRAGMA_NO_UNROLL
      for (int page = begin; page < full_end; page += 2 * KStages * step) {
        for_each(make_seq<2 * KStages>{}, [&](auto visit) {
          constexpr int Visit = decltype(visit)::value;
          constexpr int Stage = Visit % KStages;
          if constexpr (Visit == 0) {
            produce<Stage>(params, transfer, page, full_end, step, 1);
          } else if (page + Visit * step < full_end) {
            produce<Stage>(params, transfer, page + Visit * step, full_end, step, (Visit / KStages) ^ 1);
          }
        });
      }
      int visits = max(0, (full_end - begin + step - 1) / step);
      MUTLASS_PRAGMA_NO_UNROLL
      for (int page = begin + visits * step; page < end; page += step, ++visits) {
        int physical    = physical_page(transfer, page);
        transfer.source = k_source(transfer.coordinates, transfer.origin, page, physical);
        int phase       = ((visits / KStages) & 1) ^ 1;
        for_each(make_seq<KStages>{}, [&](auto stage) {
          constexpr int Stage = decltype(stage)::value;
          if (visits % KStages == Stage) {
            produce<Stage>(params, transfer, page, page + 1, step, phase);
          }
        });
      }
    } else {
      consumer(params, storage, work, begin, full_end, end, step);
    }
  }

  template <class Params>
  MUTLASS_DEVICE static void run(Params const& params, char* smem_buf) {
    Scheduler scheduler;
    auto      info = scheduler.get_schedule_info(params.scheduler);
    for (auto work = scheduler.get_initial_work(params.scheduler, info); work.is_valid(params.scheduler);
         work      = scheduler.get_next_work(params.scheduler, work, info)) {
      run_work(params, smem_buf, work);
      __syncthreads();
    }
  }
};

}  // namespace mate::attention::msa
