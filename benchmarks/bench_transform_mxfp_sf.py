"""Measure MXFP scale conversion with preallocated outputs and MUSA graphs.

GB/s counts unique input bytes plus output bytes, not hardware memory traffic.
The large profile's output alone exceeds the 32 MiB L2 on MP32/S6000.
"""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import statistics
import time

import torch

from mate.gemm import transform_mxfp_sf
from mate.gemm_kernels.tilelang.transform_mxfp_sf import (
    _get_transform_mxfp_sf_kernel,
    _select_transform_mxfp_sf_kernel,
)
from mate.mate_runtime import get_arch
from mate.utils import ceil_div


PROFILES = {
    "small": (1, 128, 4096),
    "medium": (8, 4096, 4096),
    "large": (64, 4096, 8192),
}
DTYPES = {"fp32": torch.float32, "int32": torch.int32, "uint8": torch.uint8}
RECIPES = {"1x32": (1, 32), "1x128": (1, 128), "128x128": (128, 128)}


def summarize(samples):
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "p95_us": sorted(samples)[min(len(samples) - 1, int(len(samples) * 0.95))],
        "samples_us": samples,
    }


def bench_graphs(functions, io_bytes, rounds, launches):
    launches = launches or max(8, min(256, (256 << 20) // max(1, io_bytes)))
    graphs = []
    for fn in functions:
        for _ in range(5):
            fn()
        torch.musa.synchronize()
        graph = torch.musa.MUSAGraph()
        with torch.musa.graph(graph):
            for _ in range(launches):
                fn()
        graphs.append(graph)
    for _ in range(3):
        for graph in graphs:
            graph.replay()
    torch.musa.synchronize()
    samples = [[] for _ in graphs]
    for round_idx in range(rounds):
        order = (
            range(len(graphs)) if round_idx % 2 == 0 else reversed(range(len(graphs)))
        )
        for idx in order:
            begin = torch.musa.Event(enable_timing=True)
            end = torch.musa.Event(enable_timing=True)
            begin.record()
            graphs[idx].replay()
            end.record()
            end.synchronize()
            samples[idx].append(begin.elapsed_time(end) * 1000 / launches)
    results = [summarize(values) for values in samples]
    for result in results:
        result.update(
            effective_gbps=io_bytes / result["median_us"] / 1000,
            launches_per_sample=launches,
        )
    return results


def bench_graph(fn, io_bytes, rounds, launches):
    return bench_graphs([fn], io_bytes, rounds, launches)[0]


def bench_host(fn, rounds):
    submit, complete = [], []
    torch.musa.synchronize()
    for _ in range(rounds):
        begin = time.perf_counter_ns()
        result = fn()
        submitted = time.perf_counter_ns()
        torch.musa.synchronize()
        end = time.perf_counter_ns()
        submit.append((submitted - begin) / 1000)
        complete.append((end - begin) / 1000)
        del result
    return {"host": summarize(submit), "synchronized": summarize(complete)}


def make_bandwidth_probe(n, read_factor, write_factor, threads=128, vector=16):
    """Stream a known read/write ratio; all loaded words contribute to output."""
    import tilelang
    import tilelang.language as T

    inputs = n * read_factor // write_factor
    load_width, store_width = min(4, vector * read_factor), min(4, vector)

    @tilelang.jit(
        target="musa",
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True},
    )
    def bandwidth_probe(
        src: T.Tensor[(inputs,), T.int32], dst: T.Tensor[(n,), T.int32]
    ):
        with T.Kernel(T.ceildiv(n, threads * vector), threads=threads) as bx:
            idx = (bx * threads + T.get_thread_binding()) * vector
            value = T.alloc_local((vector,), T.int32)
            loaded = T.alloc_local((vector * read_factor,), T.int32)
            if write_factor == 1:
                for chunk in T.unroll(vector * read_factor // load_width):
                    for v in T.vectorized(load_width):
                        loaded[chunk * load_width + v] = src[
                            idx * read_factor + chunk * load_width + v
                        ]
                for v in T.unroll(vector):
                    value[v] = loaded[v]
                for r in T.unroll(1, read_factor):
                    for v in T.unroll(vector):
                        value[v] = value[v] ^ loaded[r * vector + v]
            else:
                for chunk in T.unroll(max(1, vector // write_factor)):
                    scalar = src[idx // write_factor + chunk]
                    for v in T.unroll(min(vector, write_factor)):
                        value[chunk * write_factor + v] = scalar
            for chunk in T.unroll(vector // store_width):
                for v in T.vectorized(store_width):
                    dst[idx + chunk * store_width + v] = value[chunk * store_width + v]

    return bandwidth_probe


def bench_bandwidth_probes(args):
    results = []
    n = (64 << 20) // 4
    for read_factor, write_factor in ((4, 1), (1, 1), (1, 4), (1, 128), (1, 512)):
        src = torch.randint(
            1,
            1000000,
            (n * read_factor // write_factor,),
            dtype=torch.int32,
            device="musa",
        )
        dst = torch.empty((n,), dtype=torch.int32, device="musa")
        vector = 4 if write_factor == 1 else (8 if write_factor == 4 else 16)
        kernel = make_bandwidth_probe(n, read_factor, write_factor, vector=vector)
        kernel(src, dst)
        for idx in (0, 17, n // 2, n - 1):
            if write_factor == 1:
                base = idx // vector * vector * read_factor + idx % vector
                expected = 0
                for r in range(read_factor):
                    expected ^= src[base + r * vector].item()
            else:
                expected = src[idx // write_factor].item()
            assert dst[idx].item() == expected
        row = {
            "kind": "streaming_reference",
            "read_write_ratio": [read_factor, write_factor],
            "input_bytes": 4 * src.numel(),
            "output_bytes": 4 * dst.numel(),
            "threads": 128,
            "words_per_thread": vector,
        }
        row.update(
            bench_graph(
                lambda: kernel(src, dst),
                4 * (src.numel() + dst.numel()),
                args.rounds,
                args.launches,
            )
        )
        print(json.dumps(row), flush=True)
        results.append(row)
    return results


def make_input(groups, mn, k, dtype, recipe, layout):
    rows, cols = ceil_div(mn, recipe[0]), ceil_div(k, recipe[1])
    if dtype == torch.int32:
        cols = ceil_div(cols, 4)
        values = torch.randint(
            1, 255, (groups, rows, cols * 4), dtype=torch.uint8, device="musa"
        ).view(torch.int32)
    else:
        codes = torch.randint(
            1, 255, (groups, rows, cols), dtype=torch.int32, device="musa"
        )
        values = (
            (codes << 23).view(torch.float32)
            if dtype == torch.float32
            else codes.to(torch.uint8)
        )
    if layout == "k":
        return values
    stride_mn = ceil_div(rows, 4) * 4 + 4 if layout == "mn_padded" else rows
    if layout == "mn_offset":
        storage = torch.empty(1 + groups * rows * cols, dtype=dtype, device="musa")
        result = storage.as_strided((groups, rows, cols), (rows * cols, 1, rows), 1)
        return result.copy_(values)
    result = torch.empty_strided(
        (groups, rows, cols),
        (stride_mn * cols, 1, stride_mn),
        dtype=dtype,
        device="musa",
    )
    return result.copy_(values)


def check_samples(sf, out, recipe):
    groups, mn, q = out.shape
    indices = list(
        itertools.product(
            sorted({0, groups - 1}),
            sorted({0, mn // 2, mn - 1}),
            sorted({0, q // 2, q - 1}),
        )
    )
    g, i, j = (
        torch.tensor(values, dtype=torch.int64, device="musa")
        for values in zip(*indices)
    )
    src_j = j // (recipe[1] // 32)
    if sf.dtype == torch.int32:
        words = sf[g, i // recipe[0], src_j // 4].cpu().to(torch.int64)
        expected = ((words >> (src_j.cpu() % 4 * 8)) & 255).to(torch.uint8)
    else:
        values = sf[g, i // recipe[0], src_j].cpu()
        expected = (
            (values.view(torch.int32) >> 23).to(torch.uint8)
            if sf.dtype == torch.float32
            else values
        )
    torch.testing.assert_close(out[g, i, j].cpu(), expected, rtol=0, atol=0)


def bench_case(profile, dtype_name, recipe_name, layout, args):
    groups, mn, k = PROFILES[profile]
    dtype, recipe = DTYPES[dtype_name], RECIPES[recipe_name]
    q = ceil_div(k, 32)
    sf = make_input(groups, mn, k, dtype, recipe, layout)
    torch.musa.synchronize()
    first_call_start = time.perf_counter_ns()
    public_out = transform_mxfp_sf(sf, mn, k, recipe)
    torch.musa.synchronize()
    first_call_us = (time.perf_counter_ns() - first_call_start) / 1000
    check_samples(sf, public_out, recipe)
    input_bytes = sf.numel() * sf.element_size()
    output_bytes = groups * mn * q
    result = {
        "profile": profile,
        "shape_g_mn_k": [groups, mn, k],
        "dtype": dtype_name,
        "recipe": list(recipe),
        "layout": layout,
        "input_strides": list(sf.stride()),
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "logical_io_bytes": input_bytes + output_bytes,
        "first_call_us": first_call_us,
    }
    if public_out is sf:
        samples = []
        for _ in range(args.rounds):
            begin = time.perf_counter_ns()
            for _ in range(1000):
                transform_mxfp_sf(sf, mn, k, recipe)
            samples.append((time.perf_counter_ns() - begin) / 1e6)
        result.update(kind="identity", host_us=statistics.median(samples))
        return result

    del public_out
    storage = torch.empty((groups, q, mn), dtype=torch.uint8, device="musa")
    kernel = _select_transform_mxfp_sf_kernel(sf, mn, k, recipe)
    run = lambda: kernel(sf, storage)
    result.update(kind="kernel")
    functions = [run]
    if args.compare_generic:
        arch = get_arch(sf.device)
        generic = _get_transform_mxfp_sf_kernel(
            str(dtype).split(".")[-1], *recipe, f"mp_{arch.major}{arch.minor}"
        )
        functions.append(lambda: generic(sf, storage))
    timings = bench_graphs(
        functions, input_bytes + output_bytes, args.rounds, args.launches
    )
    result.update(timings[0])
    if args.compare_generic:
        result["generic"] = timings[1]
        result["speedup"] = timings[1]["median_us"] / timings[0]["median_us"]
    if args.measure_host:
        result["public_api"] = bench_host(
            lambda: transform_mxfp_sf(sf, mn, k, recipe), args.rounds
        )
        result["cached_launcher"] = bench_host(run, args.rounds)
    run()
    check_samples(sf, storage.transpose(-1, -2), recipe)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles", nargs="+", choices=PROFILES, default=list(PROFILES)
    )
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument("--recipes", nargs="+", choices=RECIPES, default=list(RECIPES))
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=["k", "mn", "mn_padded", "mn_offset"],
        default=["k", "mn"],
    )
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument(
        "--compare-generic",
        action="store_true",
        help="Interleave A/B with the general strided kernel",
    )
    parser.add_argument(
        "--measure-host",
        action="store_true",
        help="Also measure public API (including allocation) and cached-launcher host/synchronized times",
    )
    parser.add_argument(
        "--bandwidth-probes",
        action="store_true",
        help="Measure 64 MiB streaming references with matching read/write byte ratios",
    )
    parser.add_argument(
        "--launches",
        type=int,
        default=0,
        help="0 chooses 8–256 launches based on I/O size",
    )
    parser.add_argument(
        "--copy-mib",
        type=int,
        default=0,
        help="0 disables the contiguous device-copy reference",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 1 or args.launches < 0 or args.copy_mib < 0:
        parser.error(
            "rounds must be positive; launches and copy-mib must be non-negative"
        )
    torch.musa.set_device(args.device)
    torch.manual_seed(20260929)
    props = torch.musa.get_device_properties(args.device)
    kernel_path = (
        Path(__file__).resolve().parents[1]
        / "mate/gemm_kernels/tilelang/transform_mxfp_sf.py"
    )
    report = {
        "device": props.name,
        "architecture": repr(get_arch(torch.device("musa", args.device))),
        "mp_count": props.multi_processor_count,
        "l2_bytes": getattr(props, "L2_cache_size", None),
        "torch_version": torch.__version__,
        "tilelang_version": importlib.metadata.version("tilelang-musa"),
        "kernel_sha256": hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
        "method": "Preallocated output; warm JIT; MUSA-event timed graph replays, alternating A/B order when comparing the general kernel. Report median/min/p95; host timings are separate. GB/s = (unique logical input bytes + output bytes) / GPU time; not a DRAM counter. Streaming probes are measured references, not a physical upper bound. Small profiles can be cache-resident. Native UINT8 uses identity and has no GPU bandwidth.",
        "results": [],
    }
    print(
        json.dumps({key: value for key, value in report.items() if key != "results"}),
        flush=True,
    )
    if args.bandwidth_probes:
        report["bandwidth_references"] = bench_bandwidth_probes(args)
    if args.copy_mib:
        size = args.copy_mib << 20
        src = torch.randint(0, 256, (size,), dtype=torch.uint8, device="musa")
        dst = torch.empty_like(src)
        copy_result = {"kind": "device_copy", "input_bytes": size, "output_bytes": size}
        copy_result.update(
            bench_graph(lambda: dst.copy_(src), 2 * size, args.rounds, args.launches)
        )
        report["copy_reference"] = copy_result
        print(json.dumps(copy_result), flush=True)
        del src, dst
    for case in itertools.product(
        args.profiles, args.dtypes, args.recipes, args.layouts
    ):
        result = bench_case(*case, args)
        report["results"].append(result)
        print(json.dumps(result), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
