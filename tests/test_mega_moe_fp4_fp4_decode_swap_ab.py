"""EP4 decode-only exact replay; compares active FC1 bytes before FC2."""

import os
import argparse
import subprocess
import sys
from pathlib import Path
import pytest
import torch
import torch.distributed as dist

from mate.jit import env
from mate.jit.core import gen_jit_spec
from mate.jit.utils import TVM_HEADER
from mate.jit.mega_moe_fp4_fp4_decode import get_fp4_decode_module
from mate.mega_moe_fp4_fp4 import (
    FP4FP4SymmBuffer,
    fp4_fp4_decode_stage1,
    fp4_fp4_decode_stage2,
    fp4_fp4_mega_moe_decode,
    transform_weights_for_fp4_fp4_mega_moe,
)
from mega_moe_fp4_fp4_reference import (
    rank_inputs,
    rank_weights,
    reference_partials,
    routes_for,
)
from test_mega_moe_fp4_fp4_pipeline import collective_check

torch_musa = pytest.importorskip("torch_musa")


@pytest.mark.parametrize("backend", ["swap_ab", "pack_b"])
@pytest.mark.parametrize("ranks", [1, 4])
def test_swap_ab_decode_exact_replay(ranks, backend):
    if not torch.musa.is_available() or torch.musa.device_count() < ranks:
        pytest.skip("requires MP32 devices")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={ranks}",
            str(Path(__file__).resolve()),
            "--backend",
            backend,
        ],
        check=True,
        timeout=600,
    )


def build_scheduler_audit(config, mps, backend="swap_ab"):
    """Execute the production scheduler for both legal resident CTA counts.

    Numerical comparison alone cannot detect identical duplicate writes.
    This probe checks one visit per active pool/feature tile, with no copied
    scheduler arithmetic in the reference.
    """
    values = ", ".join(str(v) for v in config.__dict__.values())
    name = (
        "mate_fp4_scheduler_"
        + backend
        + "_f256_c2_regact_"
        + "_".join(str(v) for v in config.__dict__.values())
    )
    source = (
        TVM_HEADER
        + r"""
#include "mate/mega_moe/decode/fp4_fp4/mp32_decode_stage1.hpp"
#include "mate/mega_moe/prefill/fp4_fp4/prefill_host.hpp"
namespace dev = ::mega_moe::decode::mp32::fp4_fp4;
using L = dev::DecodeLayout<CONFIG>;
template<int C> using S = dev::Stage1<L, MPS, C, false, false, SWAPAB, 256, PACKB>;
template<int C> __global__ void audit(mega_moe::layout::SymBuffer<L::kRanks> sym, int* visits) {
  typename S<C>::Scheduler scheduler(L::workspace(sym.template get_base_ptr<char*>()));
  scheduler.init_l1_dispatch_count_scheduler();
  mega_moe::stage1_sched::L1TileDesc desc;
  while (scheduler.fetch_next_l1_tile_desc(desc))
    if (threadIdx.x == 0) atomicAdd(visits + desc.pool_block_idx * (2 * L::kIntermediate / S<C>::kBlockN) + desc.n_block_idx, 1);
}
void AUDIT_ENTRY(ffi::TensorView buffer, ffi::TensorView ptrs, ffi::TensorView visits, int rank) {
  ffi::MUSADeviceGuard guard(buffer.device().device_id);
  auto sym = mate::fp4_prefill_host::sym<L>(buffer, ptrs, rank);
  auto* v = static_cast<int*>(visits.data_ptr());
  audit<1><<<S<1>::kNumCTAs, 32, 0, get_stream(buffer.device())>>>(sym, v);
  audit<2><<<S<2>::kNumCTAs, 32, 0, get_stream(buffer.device())>>>(sym, v + visits.size(1) * visits.size(2));
  MATE_CHECK_MUSA_KERNEL_LAUNCH();
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(AUDIT_ENTRY, AUDIT_ENTRY);
""".replace("SWAPAB", "true" if backend == "swap_ab" else "false")
        .replace("PACKB", "true" if backend == "pack_b" else "false")
        .replace("CONFIG", values)
        .replace("MPS", str(mps))
        .replace("AUDIT_ENTRY", name)
    )
    arch = env.MusaArch.parse("3.2")
    path = env.resolve_musa_generated_dir(arch) / (name + ".mu")
    spec = gen_jit_spec(
        name,
        [path],
        generated_sources={path: source},
        target_arch=arch,
        extra_cuda_cflags=["-Od3", "-std=c++17"],
        extra_include_paths=[
            env.MATE_INCLUDE_DIR,
            env.MATE_CSRC_DIR,
            env.MUTLASS_INCLUDE_DIR,
            env.MUTLASS_INCLUDE_DIR.parent / "tools" / "util" / "include",
        ],
    )
    return spec.build_and_load().get_function(name)


def main(backend="swap_ab"):
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(4)
    torch.musa.set_device(local)
    dist.init_process_group("gloo")
    s = FP4FP4SymmBuffer(
        dist.group.WORLD, world * 4, 256, 2, 256, 256, enable_decode=True
    )
    w1, w2 = transform_weights_for_fp4_fp4_mega_moe(
        tuple(
            t.to("musa") for t in rank_weights(rank, 4, 512, 256, 1, logical_only=True)
        ),
        tuple(
            t.to("musa") for t in rank_weights(rank, 4, 256, 256, 2, logical_only=True)
        ),
    )
    s.allocate_workspace()
    for swap in (False, True):
        for stage in (1, 2):
            get_fp4_decode_module(
                s.config, stage, num_mps=s.common_mps, **{backend: swap}
            )
        if swap:
            prepare = (
                s.swap_decode_weights
                if backend == "swap_ab"
                else s.packed_decode_weights
            )
            prepare(w1, 1)
            prepare(w2, 2)
    audit = build_scheduler_audit(s.config, s.common_mps, backend)
    visits = torch.empty(
        (2, s.stage1_pool_tokens // 32, 512 // 256), dtype=torch.int32, device="musa"
    )
    x, sf = rank_inputs(rank, 256, 256)

    def snapshot(tokens, pattern):
        first = s.stage1_l1.cpu()
        fs = s.stage1_l1_sf.cpu().T.contiguous()
        second = s.stage2_l2.cpu()
        ss = s.stage2_l2_sf.cpu().T.contiguous()
        base = 0
        result = []
        allids = torch.cat(
            [routes_for(r, tokens, 2, world * 4, pattern) for r in range(world)]
        )
        for e in range(4):
            n = int((allids == rank * 4 + e).sum())
            result.append(
                sorted(
                    bytes(torch.cat([first[j], fs[j], second[j], ss[j]]).tolist())
                    for j in range(base, base + n)
                )
            )
            base += (n + 31) // 32 * 32
        return result

    try:
        for phase, (tokens, pattern) in enumerate(
            [
                (1, "balanced"),
                (24, "balanced"),
                (31, "skew_disabled"),
                (33, "skew_disabled"),
                (127, "all_disabled"),
                (128, "balanced"),
            ]
        ):
            changed = x[:tokens].clone()
            changed[:, 0] ^= phase & 1
            ids = routes_for(rank, tokens, 2, world * 4, pattern)
            s.x[:tokens].copy_(changed)
            s.x_sf[:tokens].copy_(sf[:tokens])
            s.topk_idx[:tokens].copy_(ids)
            s.topk_weights[:tokens].fill_(0.5)
            expected = (
                reference_partials(changed, sf[:tokens], ids, 256).float().sum(1) * 0.5
            ).bfloat16()
            previous = None
            for swap in (False, True):
                print("CASE_BEGIN", rank, tokens, pattern, swap, flush=True)
                out = torch.empty(tokens, 256, dtype=torch.bfloat16, device="musa")
                fp4_fp4_decode_stage1(s, w1, tokens, **{backend: swap})
                torch.musa.synchronize()
                snap = snapshot(tokens, pattern)
                if swap:
                    visits.zero_()
                    audit(s.buffer, s.ptrs, visits, rank)
                    got_visits = visits.cpu()
                    expected_visits = torch.zeros_like(got_visits)
                    base = 0
                    allids = torch.cat(
                        [
                            routes_for(r, tokens, 2, world * 4, pattern)
                            for r in range(world)
                        ]
                    )
                    for expert in range(4):
                        count = int((allids == rank * 4 + expert).sum())
                        blocks = (count + 31) // 32
                        expected_visits[:, base : base + blocks, :] = 1
                        base += blocks
                    collective_check(
                        lambda: torch.testing.assert_close(
                            got_visits, expected_visits, rtol=0, atol=0
                        )
                    )
                fp4_fp4_decode_stage2(out, s, w2, **{backend: swap})
                torch.musa.synchronize()

                def verify():
                    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
                    assert s.status.cpu().item() == 0
                    if previous is not None:
                        assert previous == snap, "active FC1 payload/scales differ"

                collective_check(verify)
                previous = snap
                if rank == 0:
                    print("CASE_PASS", tokens, pattern, swap, flush=True)
        # Actual device replay without synchronization between calls. Each output
        # has separate storage and each input update precedes its own invocation.
        queued = []
        for phase, (tokens, pattern) in enumerate(
            [
                (1, "balanced"),
                (31, "skew_disabled"),
                (33, "skew_disabled"),
                (127, "all_disabled"),
                (128, "balanced"),
            ]
        ):
            changed = x[:tokens].clone()
            changed[:, 0] ^= phase & 1
            ids = routes_for(rank, tokens, 2, world * 4, pattern)
            expected = (
                reference_partials(changed, sf[:tokens], ids, 256).float().sum(1) * 0.5
            ).bfloat16()
            for swap in (False, True):
                s.x[:tokens].copy_(changed)
                s.x_sf[:tokens].copy_(sf[:tokens])
                s.topk_idx[:tokens].copy_(ids)
                out = torch.empty(tokens, 256, dtype=torch.bfloat16, device="musa")
                fp4_fp4_mega_moe_decode(out, w1, w2, s, **{backend: swap})
                queued.append((out, expected))
        torch.musa.synchronize()

        def verify_queue():
            for out, expected in queued:
                torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

        collective_check(verify_queue)
        if rank == 0:
            print("EXACT_SWAP_REPLAY_PASS", flush=True)
    finally:
        s.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("swap_ab", "pack_b"), default="swap_ab")
    main(parser.parse_args().backend)
