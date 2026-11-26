"""Measure upstream MP32 grouped W4A8 FC1 as a Flash Stage1 compute reference."""

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

from mate.gemm import ragged_m_moe_gemm_mxfp
from mate.jit.gemm.deep_gemm.gemm import (
    DEEP_GEMM_FP8_FP4_CONFIGS,
    _select_fp8_fp4_config,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_mega_moe_fp8_fp4 import _dequant_weight, _weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="4096,8192")
    parser.add_argument("--num-mps", default="40,44")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--stages", type=int, choices=range(3, 6), default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--flush-mb", type=int, default=256)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    DEEP_GEMM_FP8_FP4_CONFIGS[(256, 256)] = replace(
        DEEP_GEMM_FP8_FP4_CONFIGS[(256, 256)], stages=args.stages
    )
    rank, world, local = (
        int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    torch.set_num_threads(4)
    torch.musa.set_device(local)
    dist.init_process_group("mccl", device_id=torch.device("musa", local))
    device_mps = [None] * world
    dist.all_gather_object(
        device_mps, torch.musa.get_device_properties(local).multi_processor_count
    )
    if rank == 0:
        print(f"DEVICE_MPS {device_mps}", flush=True)
    assert max(map(int, args.num_mps.split(","))) <= min(device_mps)
    e, n, k = args.experts // world, 2 * args.intermediate, args.hidden
    w_cpu, sfw_cpu = _weights(e, n, k, 2000 + rank)
    w = w_cpu.to("musa")
    sfw = sfw_cpu.transpose(-1, -2).contiguous().to("musa").transpose(-1, -2)
    flush = torch.empty(args.flush_mb * 1024 * 1024, dtype=torch.uint8, device="musa")
    start, end = (
        torch.musa.Event(enable_timing=True),
        torch.musa.Event(enable_timing=True),
    )
    results = []
    for tokens in map(int, args.tokens.split(",")):
        m = tokens * args.topk
        assert m % e == 0 and (m // e) % 256 == 0
        rng = torch.Generator().manual_seed(1000 + rank)
        a_cpu = torch.randn(m, k, generator=rng).to(torch.float8_e4m3fn)
        sfa_cpu = torch.randint(
            125, 129, (m, k // 32), generator=rng, dtype=torch.uint8
        )
        a = a_cpu.to("musa")
        sfa = sfa_cpu.T.contiguous().to("musa").T
        info = torch.arange(e, device="musa", dtype=torch.int32).repeat_interleave(
            m // e
        )
        output = torch.empty(m, n, dtype=torch.bfloat16, device="musa")
        config = _select_fp8_fp4_config(m, n)
        assert config == DEEP_GEMM_FP8_FP4_CONFIGS[(256, 256)]

        def run(mps, a=a, sfa=sfa, output=output, info=info):
            ragged_m_moe_gemm_mxfp(
                (a, sfa),
                (w, sfw),
                info,
                output,
                recipe=(1, 1, 32),
                num_mp=mps,
                alignment_m=256,
                backend="mutlass",
            )

        run(44)
        torch.musa.synchronize()
        # Independent sampled CPU dequantized reference, outside timing.
        for expert in (0, e // 2, e - 1):
            row = expert * (m // e)
            x = a_cpu[row].float() * torch.exp2(
                sfa_cpu[row].float() - 127
            ).repeat_interleave(32)
            ref = x @ _dequant_weight(w_cpu[expert], sfw_cpu[expert]).T
            got = output[row].float().cpu()
            rel = float((got - ref).norm() / ref.norm().clamp_min(1e-12))
            assert rel < 0.01, (rank, expert, rel)
        assert torch.isfinite(output).all()
        if rank == 0:
            print(
                f"REFERENCE_PASS tokens={tokens} groups={e} M={m} N={n} K={k}",
                flush=True,
            )
        for mps in map(int, args.num_mps.split(",")):
            assert mps <= torch.musa.get_device_properties(local).multi_processor_count
            samples = []
            for iteration in range(args.warmup + args.iters):
                flush.zero_()
                torch.musa.synchronize()
                dist.barrier()
                start.record()
                run(mps)
                end.record()
                end.synchronize()
                measured = torch.tensor(start.elapsed_time(end), device="musa")
                dist.all_reduce(measured, op=dist.ReduceOp.MAX)
                if iteration >= args.warmup:
                    samples.append(float(measured))
            median = statistics.median(samples)
            flops = 2 * m * n * k
            record = {
                "tokens_per_rank": tokens,
                "experts_per_rank": e,
                "M": m,
                "M_per_expert": m // e,
                "N": n,
                "K": k,
                "num_mps": mps,
                "tile": [256, 256, 128],
                "stages": config.stages,
                "world_size": world,
                "flops_per_gpu": flops,
                "median_ms": median,
                "min_ms": min(samples),
                "max_ms": max(samples),
                "TFLOPs_per_gpu": flops / (median * 1e9),
                "aggregate_TFLOPs": world * flops / (median * 1e9),
                "samples_ms": samples,
                "reference": "3 sampled rows/rank against CPU dequantized FP32 GEMM; rel_l2 < 0.01",
            }
            results.append(record)
            if rank == 0:
                print("SOL " + json.dumps(record), flush=True)
                Path(args.output).write_text(
                    json.dumps(
                        {
                            "settings": vars(args),
                            "device_mps": device_mps,
                            "results": results,
                        },
                        indent=2,
                    )
                    + "\n"
                )
        del a, sfa, output, info, a_cpu, sfa_cpu
    if rank == 0:
        Path(args.output).write_text(
            json.dumps(
                {"settings": vars(args), "device_mps": device_mps, "results": results},
                indent=2,
            )
            + "\n"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
