"""Attention backward benchmarks with independent gradient verification."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

import numpy as np
import torch

from mate.execution_context import (
    MateDryRunComplete,
    is_dry_run_enabled,
    raise_complete_if_dry_run,
)
from mate.testing.metrics import get_flash_attention_bwd_metrics
from mate.testing.operators.attention_bwd import (
    AttentionBwdInputs,
    AttentionBwdOperator,
    AttentionBwdWorkload,
)
from mate.testing.operators.operator import OpMode
from mate.testing.utils import bench_gpu_time
from .mate_benchmark_utils import (
    dtype_str_to_torch_dtype,
    filter_backends_by_arch,
    maybe_print_kineto_metrics,
    print_perf_metrics,
)

logger = logging.getLogger(__name__)


def get_attention_bwd_metrics(
    w: AttentionBwdWorkload, x: AttentionBwdInputs, median_ms: float
) -> tuple[float, float]:
    kernel_dim = 128 if max(w.headdim_qk, w.headdim_v or w.headdim_qk) <= 128 else 256
    terms = (
        7
        if w.deterministic or (kernel_dim == 128 and w.num_head_q != w.num_head_kv)
        else 5
    )
    # dq/dk/dv have the same shapes and dtypes as q/k/v; count their storage
    # without retaining the first call's gradients throughout timing.
    tensors = (x.q, x.k, x.v, x.out, x.dout, x.lse, x.cu_q, x.cu_kv, x.q, x.k, x.v)
    tensors += tuple(t for t in (x.seqused_q, x.seqused_k) if t is not None)
    metrics = get_flash_attention_bwd_metrics(
        tensors,
        q_lens=x.q_lengths,
        kv_lens=x.kv_lengths,
        num_head_q=w.num_head_q,
        headdim_qk=w.headdim_qk,
        headdim_v=w.headdim_v or w.headdim_qk,
        terms=terms,
        causal=w.causal,
        window_size=w.window_size,
    )
    return metrics.to_perf(median_ms)


def _workload(args: argparse.Namespace) -> AttentionBwdWorkload:
    if args.routine != "flashattn_varlen_bwd_interface":
        raise ValueError(f"Unsupported attention backward routine: {args.routine}")
    if args.q_dtype != args.kv_dtype:
        raise ValueError("attention backward requires matching Q/KV dtypes")
    if args.seqlen_mode_q == "bshd" and args.seqlen_mode_kv != "bshd":
        raise ValueError("BSHD Q requires BSHD KV")
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive")
    return AttentionBwdWorkload(
        batch_size=args.batch_size,
        seqlen_q=args.seqlen_q,
        seqlen_kv=args.seqlen_kv,
        num_head_q=args.num_head_q,
        num_head_kv=args.num_head_kv,
        headdim_qk=args.headdim_qk,
        headdim_v=args.headdim_v,
        dtype=dtype_str_to_torch_dtype(args.q_dtype),
        is_bshd=args.seqlen_mode_q == "bshd",
        q_length_mode=args.varlen_seqlen_mode_q,
        kv_length_mode=args.varlen_seqlen_mode_kv,
        seqused_q=args.seqused_q,
        seqused_k=args.seqused_k,
        causal=args.causal,
        window_size=(args.window_left, args.window_right),
        softcap=args.softcap,
        deterministic=args.deterministic,
        seed=args.random_seed,
    )


def run_attention_bwd_test(args: argparse.Namespace) -> None:
    workload = _workload(args)
    backends = filter_backends_by_arch(
        list(dict.fromkeys(args.backends)), args.routine, torch.device("musa")
    )
    if not backends:
        logger.error("[ERROR] No backends to test. Exiting.")
        return
    operator = AttentionBwdOperator()
    mode = OpMode.DRY_RUN if is_dry_run_enabled() else OpMode.NORMAL
    for backend in backends:
        with operator.use_mode(mode):
            inputs = operator.generate(workload)
            reference = operator.reference(inputs) if mode is OpMode.NORMAL else None
            try:
                outputs = operator(inputs)
                raise_complete_if_dry_run()
            except MateDryRunComplete:
                logger.info("[DRY_RUN] %s completed for %s", args.routine, backend)
                continue
            operator.verify(inputs, outputs, reference)
            logger.info("[VERIFY] %s passed for %s (dq, dk, dv)", args.routine, backend)
            del reference, outputs
            torch.musa.synchronize()

            def run_fn():
                operator(inputs)

            times = bench_gpu_time(
                run_fn,
                dry_run_iters=args.warmup,
                repeat_iters=args.repeat,
                l2_flush=True,
                l2_flush_device="musa",
                sleep_after_run=True,
            )
            ms, sd = float(np.median(times)), float(np.std(times))
            tflops, bandwidth = get_attention_bwd_metrics(workload, inputs, ms)
            print_perf_metrics(backend, ms, sd, tflops, bandwidth)
            maybe_print_kineto_metrics(args, backend, run_fn, ms, tflops, bandwidth)


def parse_attention_bwd_args(
    line: Sequence[str] | None, parser: argparse.ArgumentParser
) -> argparse.Namespace:
    """Add and parse arguments for a FlashAttention backward kernel routine."""
    parser.add_argument(
        "--backends", nargs="+", choices=("tilelang",), default=["tilelang"]
    )
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seqlen_q", type=int, required=True)
    parser.add_argument("--seqlen_kv", type=int, required=True)
    parser.add_argument("--seqlen_mode_q", choices=("varlen", "bshd"), default="varlen")
    parser.add_argument(
        "--seqlen_mode_kv", choices=("varlen", "bshd"), default="varlen"
    )
    parser.add_argument(
        "--varlen_seqlen_mode_q", choices=("random", "full"), default="full"
    )
    parser.add_argument(
        "--varlen_seqlen_mode_kv", choices=("random", "full"), default="full"
    )
    parser.add_argument("--num_head_q", type=int, required=True)
    parser.add_argument("--num_head_kv", type=int, required=True)
    parser.add_argument("--headdim_qk", type=int, required=True)
    parser.add_argument("--headdim_v", type=int, default=None)
    parser.add_argument(
        "--q_dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--kv_dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--causal", action="store_true", default=False)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--window_left", type=int, default=-1)
    parser.add_argument("--window_right", type=int, default=-1)
    parser.add_argument("--softcap", type=float, default=0.0)
    parser.add_argument("--seqused_q", action="store_true", default=False)
    parser.add_argument("--seqused_k", action="store_true", default=False)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    return parser.parse_args(line)
