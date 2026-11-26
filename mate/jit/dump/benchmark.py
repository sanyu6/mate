#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from typing import Any


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summarize(
    elapsed_ms: list[float],
    function_name: str,
    device_id: int,
    warmup_iters: int,
    profile_iters: int,
    flops: int | None = None,
) -> dict[str, Any]:
    elapsed_us = [value * 1000.0 for value in elapsed_ms]
    result = {
        "function": function_name,
        "device_id": device_id,
        "warmup_iters": warmup_iters,
        "profile_iters": profile_iters,
        "latency_us": {
            "min": min(elapsed_us),
            "median": statistics.median(elapsed_us),
            "mean": statistics.fmean(elapsed_us),
            "p90": _percentile(elapsed_us, 0.9),
            "max": max(elapsed_us),
        },
    }
    if flops is not None:
        if any(value <= 0 for value in elapsed_ms):
            raise ValueError("event elapsed time must be positive to compute TFLOPS")
        throughput_tflops = [flops / (value * 1e9) for value in elapsed_ms]
        result["workload_flops"] = flops
        result["throughput_tflops"] = {
            "min": min(throughput_tflops),
            "median": statistics.median(throughput_tflops),
            "mean": statistics.fmean(throughput_tflops),
            "p90": _percentile(throughput_tflops, 0.9),
            "max": max(throughput_tflops),
        }
    return result


def benchmark(
    warmup_iters: int,
    profile_iters: int,
    flops: int | None = None,
) -> dict[str, Any]:
    import replay

    for _ in range(warmup_iters):
        replay.invoke()
    replay.synchronize()

    create_event = replay.loader.get_function("jit_dump_loader_create_event")
    record_event = replay.loader.get_function("jit_dump_loader_record_event")
    elapsed_time_ms = replay.loader.get_function("jit_dump_loader_elapsed_time_ms")
    destroy_event = replay.loader.get_function("jit_dump_loader_destroy_event")

    event_pairs: list[tuple[int, int]] = []
    event_handles: list[int] = []
    try:
        for _ in range(profile_iters):
            start_event = create_event(replay.DEVICE_ID)
            event_handles.append(start_event)
            end_event = create_event(replay.DEVICE_ID)
            event_handles.append(end_event)
            event_pairs.append((start_event, end_event))

        for start_event, end_event in event_pairs:
            record_event(start_event, replay.DEVICE_ID)
            replay.invoke()
            record_event(end_event, replay.DEVICE_ID)
        replay.synchronize()

        elapsed = [
            float(elapsed_time_ms(start_event, end_event))
            for start_event, end_event in event_pairs
        ]
    finally:
        for event in event_handles:
            destroy_event(event)

    return _summarize(
        elapsed,
        replay.FUNCTION_NAME,
        replay.DEVICE_ID,
        warmup_iters,
        profile_iters,
        flops,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark one dumped TVM-FFI call with MUSA events."
    )
    parser.add_argument("--warmup-iters", type=_non_negative_int, default=10)
    parser.add_argument("--profile-iters", type=_positive_int, default=100)
    parser.add_argument(
        "--flops",
        type=_positive_int,
        help="theoretical floating-point operations per kernel call",
    )
    args = parser.parse_args(argv)

    result = benchmark(args.warmup_iters, args.profile_iters, args.flops)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
