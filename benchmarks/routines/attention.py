"""FMHA benchmarks using the shared operator for inputs and verification."""

import argparse
import logging
import random
from dataclasses import replace
from collections.abc import Sequence

import numpy as np
import torch

from mate.execution_context import (
    MateDryRunComplete,
    is_dry_run_enabled,
    raise_complete_if_dry_run,
)
from mate.testing.operators.fmha import (
    FmhaApi,
    FmhaLayout,
    FmhaOperator,
    FmhaTensorSpec,
    FmhaWorkload,
)
from mate.testing.operators.operator import OpMode
from mate.testing.metrics import get_flash_attention_metrics
from mate.testing.utils import bench_gpu_time

from .mate_benchmark_utils import (
    dtype_str_to_torch_dtype,
    filter_backends_by_arch,
    maybe_print_kineto_metrics,
    print_perf_metrics,
)

logger = logging.getLogger(__name__)


def get_attention_metrics(
    query_lengths: list[int],
    kv_lengths: list[int],
    num_head_q: int,
    num_head_kv: int,
    headdim_qk: int,
    headdim_v: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    out_dtype: torch.dtype,
    causal: bool,
    window_size: tuple[int, int],
    median_time_ms: float,
) -> tuple[float, float]:
    """Calculate logical attention throughput and bandwidth."""
    metrics = get_flash_attention_metrics(
        query_lengths,
        kv_lengths,
        num_head_q=num_head_q,
        num_head_kv=num_head_kv,
        headdim_qk=headdim_qk,
        headdim_v=headdim_v,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        out_dtype=out_dtype,
        causal=causal,
        window_size=window_size,
    )
    return metrics.to_perf(median_time_ms)


def _workload(args: argparse.Namespace) -> FmhaWorkload:
    """Translate the legacy CLI layouts and lengths to tensor specifications."""
    apis = {
        "flash_attn_varlen_func": FmhaApi.VARLEN,
        "flash_attn_with_kvcache": FmhaApi.KVCACHE,
    }
    if args.routine not in apis:
        raise ValueError(f"Unsupported attention routine: {args.routine}")
    api = apis[args.routine]
    if args.headdim_qk is None:
        raise ValueError(f"{args.routine} requires --headdim_qk")
    head_dim_v = args.headdim_v if args.headdim_v is not None else args.headdim_qk
    dimensions = (
        args.batch_size,
        args.seqlen_q,
        args.seqlen_kv,
        args.num_head_q,
        args.num_head_kv,
        args.headdim_qk,
        head_dim_v,
    )
    if any(value <= 0 for value in dimensions):
        raise ValueError(
            "Batch, sequence, head count, and head dimensions must be positive"
        )
    if args.causal and args.seqlen_q > args.seqlen_kv:
        raise ValueError("causal attention requires seqlen_q <= seqlen_kv")
    if (
        args.window_left < -1
        or args.window_right < -1
        or (args.window_left < 0) != (args.window_right < 0)
    ):
        raise ValueError(
            "--window_left and --window_right must both be -1 or both >= 0"
        )

    q_layout = FmhaLayout.NORMAL if args.seqlen_mode_q == "bshd" else FmhaLayout.RAGGED
    if args.seqlen_mode_kv == "paged":
        if api is FmhaApi.VARLEN:
            raise ValueError("paged KV requires flash_attn_with_kvcache")
        kv_layout = FmhaLayout.PAGED
    elif args.seqlen_mode_kv == "varlen" and api is FmhaApi.VARLEN:
        kv_layout = FmhaLayout.RAGGED
    else:
        # The cache API stores variable-length sequences in a dense cache and
        # describes the valid prefix of each batch item with cache_seqlens.
        kv_layout = FmhaLayout.NORMAL

    dtype = dtype_str_to_torch_dtype(args.q_dtype)
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    out_dtype = (
        dtype if args.out_dtype is None else dtype_str_to_torch_dtype(args.out_dtype)
    )
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Attention benchmarks support bfloat16 and float16")
    if dtype != kv_dtype or out_dtype != dtype:
        raise ValueError("Attention requires matching Q, KV, and output dtypes")

    # Lengths are host metadata, so random workloads also work in fake mode.
    rng = random.Random(args.random_seed)
    q_lengths = tuple(
        rng.randint(1, args.seqlen_q)
        if args.seqlen_mode_q == "varlen" and args.varlen_seqlen_mode_q == "random"
        else args.seqlen_q
        for _ in range(args.batch_size)
    )
    kv_lengths = tuple(
        max(rng.randint(1, args.seqlen_kv), q_len if args.causal else 1)
        if args.seqlen_mode_kv == "varlen" and args.varlen_seqlen_mode_kv == "random"
        else args.seqlen_kv
        for q_len in q_lengths
    )
    kv_spec = FmhaTensorSpec(layout=kv_layout)
    tensor_specs = (
        dict(k=kv_spec, v=kv_spec)
        if api is FmhaApi.VARLEN
        else dict(k_cache=kv_spec, v_cache=kv_spec)
    )
    operator_window = (
        (None, None)
        if args.window_left == -1 and args.window_right == -1
        else (args.window_left, args.window_right)
    )
    return FmhaWorkload(
        api=api,
        batch_size=args.batch_size,
        seqlen_q=args.seqlen_q,
        seqlen_k=args.seqlen_kv,
        num_heads_q=args.num_head_q,
        num_heads_kv=args.num_head_kv,
        head_dim_qk=args.headdim_qk,
        head_dim_v=head_dim_v,
        dtype=dtype,
        reference_dtype=dtype,
        device="musa",
        seed=args.random_seed,
        q=FmhaTensorSpec(layout=q_layout),
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        page_size=args.page_size if kv_layout is FmhaLayout.PAGED else None,
        causal=args.causal,
        window_size=operator_window,
        num_splits=args.num_splits,
        backend="auto",
        return_softmax_lse=True,
        compare_lse=True,
        **tensor_specs,
    )


def run_attention_test(args: argparse.Namespace) -> None:
    """Verify each selected backend, then measure only operator calls."""
    workload = _workload(args)
    backends = filter_backends_by_arch(
        list(dict.fromkeys(args.backends)), args.routine, torch.device("musa")
    )
    if not backends:
        logger.error("[ERROR] No backends to test. Exiting.")
        return
    operator = FmhaOperator()
    mode = OpMode.DRY_RUN if is_dry_run_enabled() else OpMode.NORMAL
    for backend in backends:
        with operator.use_mode(mode):
            inputs = operator.generate(replace(workload, backend=backend))
            reference = operator.reference(inputs) if mode is OpMode.NORMAL else None
            try:
                outputs = operator(inputs)
                raise_complete_if_dry_run()
            except MateDryRunComplete:
                logger.info("[DRY_RUN] %s completed for %s", args.routine, backend)
                continue
            operator.verify(inputs, outputs, reference)
            logger.info(
                "[VERIFY] %s passed for %s (output and LSE)", args.routine, backend
            )
            del reference, outputs
            torch.musa.synchronize()

            def run_fn() -> None:
                operator(inputs)

            times = bench_gpu_time(
                run_fn,
                l2_flush=True,
                l2_flush_device="musa",
                sleep_after_run=True,
            )
            ms = float(np.median(times))
            sd = float(np.std(times))
            tflops, bandwidth = get_attention_metrics(
                list(inputs.q_lengths),
                list(inputs.kv_lengths),
                workload.num_heads_q,
                workload.num_heads_kv,
                workload.head_dim_qk,
                workload.head_dim_v,
                workload.dtype,
                workload.dtype,
                workload.dtype,
                workload.causal,
                (args.window_left, args.window_right),
                ms,
            )
            print_perf_metrics(backend, ms, sd, tflops, bandwidth)
            maybe_print_kineto_metrics(args, backend, run_fn, ms, tflops, bandwidth)
            del inputs


def parse_attention_args(
    line: Sequence[str] | None, parser: argparse.ArgumentParser
) -> argparse.Namespace:
    """Add attention arguments to the shared parser and parse the command line."""
    routine_parser = argparse.ArgumentParser(add_help=False)
    routine_parser.add_argument("-R", "--routine")
    routine_args, _ = routine_parser.parse_known_args(line)
    if routine_args.routine == "flash_attn_varlen_bwd":
        from .attention_bwd import parse_attention_bwd_args

        return parse_attention_bwd_args(line, parser)

    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("auto", "mutlass", "mubin"),
        default=["auto"],
        help="MATE attention backends to benchmark.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size of the benchmark case.",
    )
    parser.add_argument(
        "--seqlen_q",
        type=int,
        required=True,
        help="Maximum query sequence length.",
    )
    parser.add_argument(
        "--seqlen_mode_q",
        choices=("bshd", "varlen"),
        default="varlen",
        help="Query tensor sequence layout.",
    )
    parser.add_argument(
        "--varlen_seqlen_mode_q",
        choices=("random", "full"),
        default=None,
        help="Query lengths for varlen Q; defaults to full.",
    )
    parser.add_argument(
        "--seqlen_kv",
        type=int,
        required=True,
        help="Maximum key/value sequence length.",
    )
    parser.add_argument(
        "--seqlen_mode_kv",
        choices=("bshd", "varlen", "paged"),
        default="bshd",
        help="Key/value cache sequence layout.",
    )
    parser.add_argument(
        "--varlen_seqlen_mode_kv",
        choices=("random", "full"),
        default=None,
        help="Key/value lengths for varlen KV; defaults to full.",
    )
    parser.add_argument(
        "--page_size",
        type=int,
        default=64,
        help="Page size for paged KV; defaults to 64.",
    )
    parser.add_argument(
        "--num_head_q",
        type=int,
        required=True,
        help="Number of query/output heads.",
    )
    parser.add_argument(
        "--num_head_kv",
        type=int,
        required=True,
        help="Number of key/value heads.",
    )
    parser.add_argument(
        "--headdim_qk",
        type=int,
        default=None,
        help="Query/key head dimension.",
    )
    parser.add_argument(
        "--headdim_v",
        type=int,
        default=None,
        help="Value/output head dimension; defaults to headdim_qk.",
    )
    parser.add_argument(
        "--q_dtype",
        default="bfloat16",
        help="Query data type.",
    )
    parser.add_argument(
        "--kv_dtype",
        default="bfloat16",
        help="Key/value data type.",
    )
    parser.add_argument(
        "--out_dtype",
        default=None,
        help="Output data type; defaults to q_dtype.",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        default=False,
        help="Enable causal masking for prefill attention.",
    )
    parser.add_argument(
        "--window_left",
        type=int,
        default=-1,
        help="Left window size for sliding-window attention; -1 disables local windowing.",
    )
    parser.add_argument(
        "--window_right",
        type=int,
        default=-1,
        help="Right window size for sliding-window attention; -1 disables local windowing.",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=0,
        help="KV split count; zero selects the heuristic and negative disables metadata.",
    )
    args = parser.parse_args(line)
    if args.seqlen_mode_q == "varlen":
        args.varlen_seqlen_mode_q = args.varlen_seqlen_mode_q or "full"
    elif args.varlen_seqlen_mode_q is not None:
        parser.error("--varlen_seqlen_mode_q requires --seqlen_mode_q varlen")

    if args.seqlen_mode_kv == "varlen":
        args.varlen_seqlen_mode_kv = args.varlen_seqlen_mode_kv or "full"
    elif args.varlen_seqlen_mode_kv is not None:
        parser.error("--varlen_seqlen_mode_kv requires --seqlen_mode_kv varlen")

    if args.verbose >= 1:
        logger.info(f"[INFO] {args = }")
    return args
