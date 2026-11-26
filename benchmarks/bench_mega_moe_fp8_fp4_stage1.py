"""EP8 MP32 Stage1 decomposition using the DeepSeek Flash MoE dimensions."""

import argparse
import json
import os
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
import torch_musa  # noqa: F401

from mate.jit.mega_moe_fp8_fp4 import PrefillConfig, _get_stage_module
from mate.mega_moe_fp8_fp4 import (
    get_symm_buffer_for_fp8_fp4_mega_moe,
    transform_weights_for_fp8_fp4_mega_moe,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_mega_moe_fp8_fp4 import _weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--num-mps", type=int, default=44)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--owner-ctas", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--flush-mb", type=int, default=256)
    parser.add_argument("--label", default="group_vector")
    parser.add_argument("--output")
    args = parser.parse_args()
    rank, world, local = (
        int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    torch.set_num_threads(4)
    torch.musa.set_device(local)
    dist.init_process_group("mccl", device_id=torch.device("musa", local))
    sym = get_symm_buffer_for_fp8_fp4_mega_moe(
        dist.group.WORLD,
        args.experts,
        args.tokens,
        args.topk,
        args.hidden,
        args.intermediate,
    )
    rng = torch.Generator().manual_seed(1000 + rank)
    sym.x.copy_(
        torch.randn(args.tokens, args.hidden, generator=rng).to(torch.float8_e4m3fn)
    )
    sym.x_sf.copy_(
        torch.randint(125, 129, sym.x_sf.shape, generator=rng, dtype=torch.uint8)
    )
    routes = (
        torch.arange(args.tokens)[:, None] + torch.arange(args.topk)[None, :] + rank
    ) % args.experts
    sym.topk_idx.copy_(routes)
    scores = torch.rand(args.tokens, args.topk, generator=rng)
    sym.topk_weights.copy_(scores / scores.sum(-1, keepdim=True))
    e = args.experts // world
    w1, w2 = transform_weights_for_fp8_fp4_mega_moe(
        tuple(
            t.to("musa")
            for t in _weights(e, 2 * args.intermediate, args.hidden, 2000 + rank)
        ),
        tuple(
            t.to("musa")
            for t in _weights(e, args.hidden, args.intermediate, 3000 + rank)
        ),
    )
    out = torch.empty(args.tokens, args.hidden, dtype=torch.bfloat16, device="musa")
    config = PrefillConfig(
        world,
        args.experts,
        sym.num_max_tokens_per_rank,
        args.topk,
        args.hidden,
        args.intermediate,
        sym.prefill_num_max_pool_tokens,
        num_mps=args.num_mps,
        stage1_stages=args.stages,
        owner_ctas=args.owner_ctas,
    )

    def build(c, stage):
        if rank == 0:
            print(f"COMPILE stage={stage} mode={c.profile_mode}", flush=True)
        name, module = _get_stage_module(c, stage)
        return module.get_function(name)

    functions = {
        mode: build(replace(config, profile_mode=mode), 1)
        for mode in ("full", "dispatch_only", "compute_only")
    }
    stage2, stage3 = build(config, 2), build(config, 3)
    ptrs = sym.handle.buffer_ptrs_tensor

    def stage1(mode):
        functions[mode](
            sym.l1_acts,
            sym.l1_acts_sf,
            sym.l2_acts,
            sym.l2_acts_sf,
            *w1,
            ptrs,
            rank,
            args.tokens,
            10.0,
            1.0,
            -1.0,
        )

    def cleanup():
        # Stage2 resets route counters after a system-wide barrier. Keep this
        # outside Stage1 timing, including after dispatch-only measurements.
        stage2(sym.l2_acts, sym.l2_acts_sf, *w2, ptrs, rank)

    def combine():
        stage3(out, ptrs, rank, args.tokens)

    def e2e():
        stage1("full")
        cleanup()
        combine()

    torch.musa.synchronize()
    dist.barrier()
    e2e()
    torch.musa.synchronize()
    expected = out.clone()
    stage1("dispatch_only")
    stage1("compute_only")
    cleanup()
    combine()
    torch.musa.synchronize()
    assert torch.isfinite(out).all()
    assert torch.equal(out, expected), (
        "split dispatch/compute differs from fused Stage1"
    )
    if rank == 0:
        print("SPLIT_EQUIVALENCE_PASS", flush=True)

    flush = torch.empty(args.flush_mb * 1024 * 1024, dtype=torch.uint8, device="musa")
    start, end = (
        torch.musa.Event(enable_timing=True),
        torch.musa.Event(enable_timing=True),
    )

    def measure(mode):
        if mode == "compute_only":
            stage1("dispatch_only")
            torch.musa.synchronize()
            dist.barrier()
        samples = []
        for iteration in range(args.warmup + args.iters):
            flush.zero_()
            torch.musa.synchronize()
            dist.barrier()
            start.record()
            e2e() if mode == "e2e" else stage1(mode)
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end)
            if mode in ("full", "dispatch_only"):
                cleanup()
            torch.musa.synchronize()
            dist.barrier()
            value = torch.tensor(ms, device="musa")
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            if iteration >= args.warmup:
                samples.append(float(value))
        if mode == "compute_only":
            cleanup()
            torch.musa.synchronize()
            dist.barrier()
        return {
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "samples_ms": samples,
        }

    result = {
        "label": args.label,
        "shape": vars(args),
        "world_size": world,
        "reference": "split/fused bitwise equivalence, all tokens",
    }
    for mode in ("dispatch_only", "compute_only", "full", "e2e"):
        result[mode] = measure(mode)
        if rank == 0:
            print(f"TIMING {mode} {result[mode]}", flush=True)
    td, tc, tf = (
        result[m]["median_ms"] for m in ("dispatch_only", "compute_only", "full")
    )
    result["hidden_time_ms"] = td + tc - tf
    result["overlap_efficiency"] = (td + tc - tf) / min(td, tc)
    result["stage1_serial_speedup"] = (td + tc) / tf
    # Counts are uniform for this cyclic routing workload. Local expert pulls
    # are excluded; each remote route transfers FP8 values and K32 UE8M0 scales.
    payload = args.hidden + args.hidden // 32
    remote_bytes = args.tokens * args.topk * payload * (world - 1) / world
    flops = 2 * args.tokens * args.topk * args.hidden * (2 * args.intermediate)
    result["equivalent_metrics"] = {
        "remote_payload_bytes_per_gpu": remote_bytes,
        "one_way_rx_GBps_per_gpu": remote_bytes / (td * 1e6),
        "one_way_rx_GBps_aggregate": world * remote_bytes / (td * 1e6),
        "FC1_flops_per_gpu": flops,
        "FC1_TFLOPs_per_gpu": flops / (tc * 1e9),
        "communication_note": "logical activation+scale RX payload, excludes metadata and local routes; not physical link counters",
        "compute_note": "useful FC1 GEMM FLOPs divided by compute-only time, including activation and quantization",
    }
    if rank == 0:
        print("RESULT " + json.dumps(result), flush=True)
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    sym.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
