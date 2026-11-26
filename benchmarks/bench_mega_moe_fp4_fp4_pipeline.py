"""EP4/EP8 Flash prefill validation and split/full stage measurements."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import torch
import torch.distributed as dist
import torch_musa  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from mega_moe_fp4_fp4_reference import (
    rank_inputs,
    rank_weights,
    routes_for,
    reference_partials,
)  # noqa: E402
from mate.mega_moe_fp4_fp4 import (
    FP4FP4SymmBuffer,
    transform_weights_for_fp4_fp4_mega_moe,
    fp4_fp4_mega_moe,
    fp4_fp4_stage1,
    fp4_fp4_stage2,
)  # noqa: E402
from mate.jit.mega_moe_fp4_fp4 import (
    get_fp4_prefill_stage_module,
)  # noqa: E402


# Keep control traffic off the devices occupied by resident prefill grids.
CONTROL_GROUP = None


def check_collectively(function):
    error = None
    try:
        function()
    except Exception as exc:
        error = repr(exc)
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error, group=CONTROL_GROUP)
    assert not any(errors), errors


def main():
    global CONTROL_GROUP
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--tokens", type=int, choices=(4096, 8192), required=True)
    parser.add_argument("--activation", choices=("swiglu", "situ"), default="swiglu")
    parser.add_argument("--activation-beta", type=float, default=4.0)
    parser.add_argument(
        "--activation-linear-beta",
        type=lambda value: None if value.lower() == "none" else float(value),
        default=25.0,
    )
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument(
        "--activation-clamp",
        type=lambda value: None if value.lower() == "none" else float(value),
        default=10.0,
    )
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.activation == "situ" and args.activation_clamp is not None:
        parser.error("SiTU requires --activation-clamp none")
    activation_options = {
        "activation": args.activation,
        "activation_clamp": args.activation_clamp,
    }
    reference_options = {"activation": args.activation}
    jit_activation_options = {
        "activation": args.activation,
        "use_situ_linear": args.activation == "situ"
        and args.activation_linear_beta is not None,
    }
    if args.activation == "situ":
        activation_options.update(
            activation_beta=args.activation_beta,
            activation_linear_beta=args.activation_linear_beta,
        )
        reference_options.update(
            activation_beta=args.activation_beta,
            activation_linear_beta=args.activation_linear_beta,
        )

    if min(args.samples, args.rounds) <= 0:
        parser.error("samples/rounds must be positive")
    rank, world, local = (
        int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    if world not in (4, 8):
        parser.error("requires EP4 or EP8")
    torch.set_num_threads(4)
    torch.musa.set_device(local)
    # All control traffic stays on CPU, outside the resident prefill grids.
    dist.init_process_group("gloo")
    CONTROL_GROUP = dist.group.WORLD
    if torch.musa.device_count() != world:
        parser.error("visible device count must match EP; set MUSA_VISIBLE_DEVICES")
    physical_mps = [None] * world
    dist.all_gather_object(
        physical_mps,
        torch.musa.get_device_properties(local).multi_processor_count,
        group=CONTROL_GROUP,
    )
    t, h, i, e, k = args.tokens, args.hidden, args.intermediate, args.experts, args.topk
    s = FP4FP4SymmBuffer(CONTROL_GROUP, e, t, k, h, i)
    s.allocate_workspace()
    mps = s.common_mps
    if rank == 0:
        print(
            "FP4_PREFILL_ENVIRONMENT",
            world,
            t,
            physical_mps,
            "common_mps",
            mps,
            flush=True,
        )
    x, sf = rank_inputs(rank, t, h)
    s.x.copy_(x)
    s.x_sf.copy_(sf)
    ids = routes_for(rank, t, k, e, "balanced")
    s.topk_idx.copy_(ids)
    s.topk_weights.fill_(1.0 / k)
    logical_w1 = rank_weights(rank, e // world, 2 * i, h, 1, logical_only=True)
    logical_w2 = rank_weights(rank, e // world, h, i, 2, logical_only=True)
    w1, w2 = transform_weights_for_fp4_fp4_mega_moe(
        tuple(t.to("musa") for t in logical_w1),
        tuple(t.to("musa") for t in logical_w2),
    )
    out = torch.empty((t, h), dtype=torch.bfloat16, device="musa")
    get_fp4_prefill_stage_module(s.config, 2, num_mps=mps)
    if rank == 0:
        print("FP4_PREFILL_COMPILED", t, "stage2", flush=True)
    get_fp4_prefill_stage_module(s.config, 3)
    for mode in ("full", "compute", "dispatch"):
        name, module = get_fp4_prefill_stage_module(
            s.config, 1, mode=mode, num_mps=mps, **jit_activation_options
        )
        if rank == 0:
            print("FP4_PREFILL_COMPILED", t, "stage1", mode, flush=True)
    shared = module.get_function(name + "_shared")()
    threads = module.get_function(name + "_threads")()
    s.fused_stage1_weights(w1)
    # Drain every rank's preparation before any resident IPC grid can start.
    torch.musa.synchronize()
    dist.barrier(group=CONTROL_GROUP)
    torch.musa.synchronize()

    def fused(active=t):
        fp4_fp4_mega_moe(out[:active], w1, w2, s, num_mps=mps, **activation_options)

    def first(mode="full"):
        fp4_fp4_stage1(s, w1, t, num_mps=mps, mode=mode, **activation_options)

    name3, module3 = get_fp4_prefill_stage_module(s.config, 3)
    combine = module3.get_function(name3)

    def second():
        fp4_fp4_stage2(s, w2, num_mps=mps)

    def third():
        combine(out, s.buffer, t, mps)

    try:
        if rank == 0:
            print("FP4_PREFILL_BASELINE_BEGIN", world, t, physical_mps, flush=True)
        first()
        torch.musa.synchronize()
        dist.barrier(group=CONTROL_GROUP)
        if rank == 0:
            print("FP4_PREFILL_STAGE1_READY", t, flush=True)
        second()
        torch.musa.synchronize()
        dist.barrier(group=CONTROL_GROUP)
        if rank == 0:
            print("FP4_PREFILL_STAGE2_READY", t, flush=True)
        third()
        torch.musa.synchronize()
        baseline = out.clone()
        # Finish the runtime D2D snapshot before resident IPC work can start
        # again on a peer. Keep verification copies outside measured events.
        torch.musa.synchronize()
        dist.barrier(group=CONTROL_GROUP)
        torch.musa.synchronize()

        def compare():
            assert s.status.cpu().item() == 0
            assert torch.isfinite(out).all().item()
            torch.testing.assert_close(out, baseline, rtol=0, atol=0)

        check_collectively(compare)
        selection = torch.tensor(
            [
                rank * 32 + 13,
                t // 2 + ((rank + 3) % world) * 32 + 9,
                t - 1 - ((rank + 5) % world) * 32 - 9,
            ]
        )
        metrics = {}

        def reference():
            partials = reference_partials(
                x[selection],
                sf[selection],
                ids[selection],
                i,
                clamp=args.activation_clamp,
                **reference_options,
            )
            ref = torch.zeros((len(selection), h))
            for slot in range(k):
                ref = torch.addcmul(
                    ref, partials[:, slot].float(), torch.tensor(1.0 / k)
                )
            ref = ref.bfloat16().float()
            got = out[selection.to("musa")].cpu().float()
            rel = ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()
            metrics.update(
                tokens=selection.tolist(),
                experts=ids[selection].tolist(),
                rel_l2=rel,
                max_abs=(got - ref).abs().max().item(),
            )
            assert rel < 0.01, metrics

        check_collectively(reference)
        fused(t // 2)
        check_collectively(
            lambda: torch.testing.assert_close(
                out[: t // 2], baseline[: t // 2], rtol=0, atol=0
            )
        )
        fused()
        check_collectively(compare)

        def check_replay(rows, packed, route_ids):
            partials = reference_partials(
                packed[rows],
                sf[rows],
                route_ids[rows],
                i,
                clamp=args.activation_clamp,
                **reference_options,
            )
            ref = torch.zeros((len(rows), h))
            for slot in range(k):
                ref = (
                    ref.double()
                    + partials[:, slot].double() * float(torch.tensor(1.0 / k))
                ).float()
            torch.testing.assert_close(
                out[rows.to("musa")].cpu(), ref.bfloat16(), rtol=0, atol=0
            )

        # Include ranks with no local experts and disabled routes.
        skew = routes_for(rank, t, k, e, "skew")
        skew[::3, 0] = -1
        s.topk_idx.copy_(skew)
        fused(129)
        check_collectively(lambda: check_replay(torch.tensor([0, 64, 128]), x, skew))
        s.topk_idx.copy_(ids)
        updated = x.clone()
        updated[:, 0].bitwise_xor_(1)
        s.x.copy_(updated)
        fused()
        check_collectively(lambda: check_replay(selection, updated, ids))
        s.x.copy_(x)
        fused()
        check_collectively(compare)
        all_metrics = [None] * world
        dist.all_gather_object(all_metrics, metrics, group=CONTROL_GROUP)
        if rank == 0:
            print(
                "FP4_PREFILL_VALIDATED",
                t,
                "mps",
                mps,
                "shared",
                shared,
                json.dumps(all_metrics),
                flush=True,
            )

        flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="musa")
        cases = {
            "full_stage1": (lambda: None, first),
            "dispatch_only": (lambda: None, lambda: first("dispatch")),
            "compute_only": (lambda: first("dispatch"), lambda: first("compute")),
            "full_stage2": (first, second),
            "stage3": (fused, third),
            "e2e": (lambda: None, fused),
        }
        measurements = {name: [] for name in cases}
        for round_index in range(args.rounds):
            order = list(cases)
            if round_index % 2:
                order.reverse()
            for label in order:
                if rank == 0:
                    print("FP4_PREFILL_BEGIN", t, round_index, label, flush=True)
                setup, run = cases[label]
                samples = []
                for index in range(5 + args.samples):
                    setup()
                    torch.musa.synchronize()
                    dist.barrier(group=CONTROL_GROUP)
                    # CPU control collectives cannot occupy an MP or enqueue
                    # competing GPU copies before a resident prefill launch.
                    torch.musa.synchronize()
                    flush.zero_()
                    start, end = (
                        torch.musa.Event(enable_timing=True) for _ in range(2)
                    )
                    start.record()
                    run()
                    end.record()
                    end.synchronize()
                    if index >= 5:
                        samples.append(start.elapsed_time(end))
                ranks = [None] * world
                dist.all_gather_object(ranks, samples, group=CONTROL_GROUP)
                maxima = [max(values) for values in zip(*ranks)]
                entry = {
                    "median_ms": statistics.median(maxima),
                    "rank_samples_ms": ranks,
                    "max_rank_samples_ms": maxima,
                }
                measurements[label].append(entry)
                if rank == 0:
                    print(
                        "FP4_PREFILL_TIME",
                        t,
                        round_index,
                        label,
                        entry["median_ms"],
                        flush=True,
                    )
        fused()
        check_collectively(compare)
        medians = {
            label: statistics.median(row["median_ms"] for row in rounds)
            for label, rounds in measurements.items()
        }
        td, tc, tf = (
            medians[label] for label in ("dispatch_only", "compute_only", "full_stage1")
        )
        overlap = td + tc - tf
        f1 = 2 * t * k * h * (2 * i)
        f2 = 2 * t * k * h * i
        summary = {
            "median_ms": medians,
            "stage1_tflops_per_rank": f1 / tf / 1e9,
            "compute_tflops_per_rank": f1 / tc / 1e9,
            "e2e_tflops_per_rank": (f1 + f2) / medians["e2e"] / 1e9,
            "stage2_tflops_per_rank": (2 * t * k * h * i)
            / medians["full_stage2"]
            / 1e9,
            "overlap_ms": overlap,
            "overlap_efficiency": overlap / min(td, tc),
        }
        if rank == 0:
            repo = Path(__file__).resolve().parents[1]
            files = [
                Path(__file__).resolve(),
                repo / "mate/mega_moe_fp4_fp4.py",
                repo / "mate/jit/mega_moe_fp4_fp4.py",
                repo / "mate/_fp4_moe_buffer.py",
                *sorted((repo / "include/mate/mega_moe/prefill/fp4_fp4").glob("*.hpp")),
                *sorted((repo / "csrc/templates/mega_moe").glob("fp4_fp4_*.j2")),
                repo / "tests/mega_moe_fp4_fp4_reference.py",
            ]
            data = {
                "shape": dict(
                    ep=world,
                    tokens=t,
                    hidden=h,
                    intermediate=i,
                    experts=e,
                    topk=k,
                    activation=args.activation,
                    activation_clamp=args.activation_clamp,
                    activation_beta=args.activation_beta
                    if args.activation == "situ"
                    else None,
                    activation_linear_beta=args.activation_linear_beta
                    if args.activation == "situ"
                    else None,
                    common_mps=mps,
                    physical_mps=physical_mps,
                    local_experts=e // world,
                    balanced_expert_tokens=t * k * world // e,
                ),
                "git_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
                ).strip(),
                "source_sha256": {
                    str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in files
                },
                "checks": all_metrics,
                "resources": dict(
                    tile=[256, 256, 256],
                    threads=threads,
                    shared_bytes=shared,
                    workspace_bytes=sum(
                        getattr(s, name).numel() * getattr(s, name).element_size()
                        for name in (
                            "stage1_l1",
                            "stage1_l1_sf",
                            "stage2_l2",
                            "stage2_l2_sf",
                            "counts",
                            "metadata",
                            "status",
                        )
                    ),
                ),
                "summary": summary,
                "measurements": measurements,
                "scope": "prequantized inputs; full includes setup/reset/IPC; compute includes GEMM/gated activation/FP4; dispatch includes setup/pull/tails; max rank then median; 256MiB flush outside events; five warmup samples per case per round; un-clipped overlap",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(data, indent=2) + "\n")
            print("FP4_PREFILL_PASS", t, json.dumps(summary), flush=True)
    finally:
        s.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
