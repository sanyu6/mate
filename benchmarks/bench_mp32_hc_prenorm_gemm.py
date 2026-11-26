"""Benchmark MP32 HC prenorm GEMM for V4.1 Flash MHC decode.

Covers the business M cases (13, 137, 4096, 8192) at the production decode shape
(N=24, K=20480, split_K=32) plus a split-K sweep per M. Only the kernel under
test is timed: there is no reference or competitor column, the report is
latency and bandwidth.

Pass --m/--n/--k to benchmark one custom shape instead of the business cases.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Callable

import torch

import mate.deep_gemm as deep_gemm


M_CASES = (13, 137, 4096, 8192)
N_DEFAULT = 24
K_DEFAULT = 20480
SPLIT_K_DEFAULT = 32
SPLIT_K_SWEEP = (1, 4, 8, 16, 32, 64)


@dataclass(frozen=True)
class BenchResult:
    name: str
    us: float
    bytes_rw: int
    flops: int

    @property
    def bandwidth_gbps(self) -> float:
        return self.bytes_rw / (self.us * 1.0e-6) / 1.0e9 if self.us > 0 else 0.0

    @property
    def tflops(self) -> float:
        """Theoretical GEMM throughput for this benchmark shape."""
        return self.flops / (self.us * 1.0e-6) / 1.0e12 if self.us > 0 else 0.0


def _sync() -> None:
    torch.musa.synchronize()


def _bench_us(fn: Callable[[], None], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    events = [
        (torch.musa.Event(enable_timing=True), torch.musa.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for start_ev, end_ev in events:
        start_ev.record()
        fn()
        end_ev.record()
    _sync()
    times = [start.elapsed_time(end) * 1000.0 for start, end in events]
    return statistics.median(times)


def _num_bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _make_inputs(
    m: int,
    n: int,
    k: int,
    *,
    device: str = "musa",
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16) * 0.1
    b = torch.randn((n, k), device=device, dtype=torch.float32) * 0.01
    return a, b


def bench_prenorm_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    num_splits: int,
    *,
    warmup: int,
    iters: int,
) -> BenchResult:
    m, k = a.shape
    n = b.shape[0]

    if num_splits <= 1:
        d = torch.empty((m, n), device=a.device, dtype=torch.float32)
        s = torch.empty((m,), device=a.device, dtype=torch.float32)
    else:
        d = torch.empty((num_splits, m, n), device=a.device, dtype=torch.float32)
        s = torch.empty((num_splits, m), device=a.device, dtype=torch.float32)

    def run_kernel() -> None:
        deep_gemm.tf32_hc_prenorm_gemm(a, b.contiguous(), d, s, num_splits=num_splits)

    kernel_us = _bench_us(run_kernel, warmup=warmup, iters=iters)
    # Traffic of one invocation: read A/B and write D/sqrsum. The split-K partial
    # layout is part of the kernel contract, so the selected num_splits decides
    # the actual D/sqrsum output footprint that is counted.
    bytes_rw = _num_bytes(a, b, d, s)

    return BenchResult(
        name=f"prenorm_m{m}_n{n}_k{k}_splits{num_splits}",
        us=kernel_us,
        bytes_rw=bytes_rw,
        flops=2 * m * n * k,
    )


def _print_header() -> None:
    print(f"{'Kernel':<45s} {'Latency':>12s} {'BW':>12s} {'TFLOPS':>10s}")
    print("-" * 83)


def _print_result(result: BenchResult) -> None:
    print(
        f"{result.name:<45s} "
        f"{result.us:>7.1f} us "
        f"{result.bandwidth_gbps:>8.2f} GB/s "
        f"{result.tflops:>9.2f}"
    )


def _print_section(title: str) -> None:
    print()
    print("=" * 83)
    print(title)
    print("=" * 83)
    _print_header()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MP32 HC prenorm GEMM (V4.1 Flash MHC decode)."
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help="benchmark a single custom M instead of the business M cases",
    )
    parser.add_argument("--n", type=int, default=N_DEFAULT, help="N of the custom case")
    parser.add_argument("--k", type=int, default=K_DEFAULT, help="K of the custom case")
    parser.add_argument(
        "--splits",
        type=int,
        nargs="+",
        default=None,
        help=f"split-K values of the custom case (default: {SPLIT_K_SWEEP})",
    )
    args = parser.parse_args()

    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        raise RuntimeError("MUSA device is not available.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.mudnn.allow_tf32 = True

    device_name = torch.musa.get_device_name(0)
    print(f"Device: {device_name}")
    if args.m is None:
        print(f"M cases: {M_CASES}  N: {N_DEFAULT}  K: {K_DEFAULT}")
    else:
        print(f"Custom case: M={args.m}  N={args.n}  K={args.k}")
    print(f"Warmup: {args.warmup}  Iters: {args.iters}")

    if args.m is not None:
        splits = args.splits if args.splits is not None else list(SPLIT_K_SWEEP)
        _print_section(f"Split-K sweep: M={args.m}, N={args.n}, K={args.k}")
        a, b = _make_inputs(args.m, args.n, args.k)
        for sk in splits:
            _print_result(
                bench_prenorm_kernel(
                    a, b, num_splits=sk, warmup=args.warmup, iters=args.iters
                )
            )
        return

    # ---------------------------------------------------------------
    # 1. Business M cases at the production decode shape
    # ---------------------------------------------------------------
    _print_section(
        f"1. Business M cases: N={N_DEFAULT}, K={K_DEFAULT}, split_K={SPLIT_K_DEFAULT}"
    )

    for m in M_CASES:
        a, b = _make_inputs(m, N_DEFAULT, K_DEFAULT)
        _print_result(
            bench_prenorm_kernel(
                a, b, num_splits=SPLIT_K_DEFAULT, warmup=args.warmup, iters=args.iters
            )
        )

    # ---------------------------------------------------------------
    # 2. Split-K sweep per M
    # ---------------------------------------------------------------
    _print_section(f"2. Split-K sweep: N={N_DEFAULT}, K={K_DEFAULT}")

    for m in M_CASES:
        a, b = _make_inputs(m, N_DEFAULT, K_DEFAULT)
        for sk in SPLIT_K_SWEEP:
            _print_result(
                bench_prenorm_kernel(
                    a, b, num_splits=sk, warmup=args.warmup, iters=args.iters
                )
            )
        print("-" * 83)


if __name__ == "__main__":
    main()
