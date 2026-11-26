"""GDN benchmarks using structured operators and independent verification."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import torch

from mate.execution_context import (
    MateDryRunComplete,
    is_dry_run_enabled,
    raise_complete_if_dry_run,
)
from mate.testing.metrics import get_gdn_decode_metrics, get_gdn_prefill_metrics
from mate.testing.operators.gdn import GdnApi, GdnWorkload, GdnInputs, GdnOperator
from mate.testing.operators.operator import OpMode
from mate.testing.utils import bench_gpu_time
from .mate_benchmark_utils import (
    dtype_str_to_torch_dtype,
    filter_backends_by_arch,
    maybe_print_kineto_metrics,
    print_perf_metrics,
)


logger = logging.getLogger(__name__)


def get_gdn_metrics(
    workload: GdnWorkload, inputs: GdnInputs, median_ms: float
) -> tuple[float, float]:
    x, w = inputs, workload
    if w.api is GdnApi.DECODE:
        metrics = get_gdn_decode_metrics(
            (
                x.q,
                x.k,
                x.v,
                x.state,
                x.A_log,
                x.a,
                x.dt_bias,
                x.b,
                x.output,
                x.state_indices,
            ),
            batch_size=w.batch_size,
            head_o=w.num_head_v,
            head_size=w.head_size,
        )
    else:
        metrics = get_gdn_prefill_metrics(
            (
                x.q,
                x.k,
                x.v,
                x.g,
                x.beta,
                x.initial_state,
                x.output,
                x.output_state,
                x.cu_seqlens,
            ),
            total_tokens=sum(w.lengths),
            head_o=w.num_head_v,
            head_size=w.head_size,
        )
    return metrics.to_perf(median_ms)


def _workload(args: argparse.Namespace) -> GdnWorkload:
    common = dict(
        num_head_q=args.num_head_q,
        num_head_v=args.num_head_v,
        head_size=args.head_size,
        dtype=dtype_str_to_torch_dtype(
            {"bf16": "bfloat16", "fp16": "float16"}.get(args.dtype, args.dtype)
        ),
        seed=args.random_seed,
    )
    if args.routine == "gdn_decode":
        return GdnWorkload(
            api=GdnApi.DECODE,
            batch_size=args.batch_size,
            seqlen=1,
            cache_indices="none"
            if args.cache_indices == "both"
            else args.cache_indices,
            **common,
        )
    if args.routine != "gdn_prefill":
        raise ValueError(f"Unsupported GDN routine: {args.routine}")
    lengths, batch_size, _ = _resolve_prefill_lengths(args)
    return GdnWorkload(
        api=GdnApi.PREFILL,
        batch_size=batch_size,
        seqlen=max(lengths),
        seq_lens=tuple(lengths) if args.seqlen_mode == "varlen" else None,
        initial_state=args.initial_state,
        output_final_state=args.output_final_state,
        qk_l2norm=args.qk_l2norm,
        chunk_size=args.chunk_size,
        is_log_space=args.is_log_space,
        **common,
    )


def run_gdn_test(args: argparse.Namespace) -> None:
    """Verify from the initial state, restore it, then time continuous updates."""
    workload = _workload(args)
    backends = filter_backends_by_arch(
        [args.backend], args.routine, torch.device("musa")
    )
    if not backends:
        logger.error("[ERROR] No backends to test. Exiting.")
        return
    cache_modes = (
        ("none", "identity")
        if workload.api is GdnApi.DECODE and args.cache_indices == "both"
        else (workload.cache_indices,)
    )
    operator = GdnOperator()
    mode = OpMode.DRY_RUN if is_dry_run_enabled() else OpMode.NORMAL
    for backend in backends:
        for cache_mode in cache_modes:
            label = backend if len(cache_modes) == 1 else f"{backend}:{cache_mode}"
            with operator.use_mode(mode):
                inputs = operator.generate(replace(workload, cache_indices=cache_mode))
                snapshot = inputs.clone() if mode is OpMode.NORMAL else None
                reference = (
                    operator.reference(inputs) if mode is OpMode.NORMAL else None
                )
                try:
                    outputs = operator(inputs)
                    raise_complete_if_dry_run()
                except MateDryRunComplete:
                    logger.info("[DRY_RUN] %s completed for %s", args.routine, label)
                    continue
                operator.verify(inputs, outputs, reference)
                logger.info(
                    "[VERIFY] %s passed for %s (output and state)", args.routine, label
                )
                inputs._copy_from(snapshot)
                del snapshot, reference, outputs
                torch.musa.synchronize()

                def run_fn() -> None:
                    operator(inputs)

                times = bench_gpu_time(
                    run_fn,
                    repeat_iters=args.num_tests,
                    l2_flush=True,
                    l2_flush_device="musa",
                    sleep_after_run=True,
                )
                ms, sd = float(np.median(times)), float(np.std(times))
                tflops, bandwidth = get_gdn_metrics(workload, inputs, ms)
                print_perf_metrics(label, ms, sd, tflops, bandwidth)
                maybe_print_kineto_metrics(args, label, run_fn, ms, tflops, bandwidth)
                del inputs


def _parse_seq_lens(value: str) -> list[int]:
    """Parse comma- or colon-separated positive sequence lengths."""
    try:
        seq_lens = [int(item) for item in value.replace(":", ",").split(",") if item]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid sequence lengths: {value}") from exc
    if not seq_lens or any(seq_len <= 0 for seq_len in seq_lens):
        raise argparse.ArgumentTypeError(
            "--seq_lens must contain one or more positive integers"
        )
    return seq_lens


def _resolve_prefill_lengths(args: argparse.Namespace) -> tuple[list[int], int, int]:
    if args.seqlen_mode == "dense":
        if args.seq_lens is not None:
            raise ValueError("--seq_lens requires --seqlen_mode varlen")
        if args.seqlen is None or args.seqlen <= 0:
            raise ValueError("Dense gdn_prefill requires a positive --seqlen")
        batch_size = 1 if args.batch_size is None else args.batch_size
        if batch_size <= 0:
            raise ValueError("--batch_size must be positive")
        seq_lens = [args.seqlen] * batch_size
        return seq_lens, batch_size, batch_size * args.seqlen

    if args.seq_lens is None:
        raise ValueError("Varlen gdn_prefill requires --seq_lens")
    batch_size = len(args.seq_lens)
    if args.batch_size is not None and args.batch_size != batch_size:
        raise ValueError(
            "For varlen gdn_prefill, --batch_size must match the number of "
            "entries in --seq_lens"
        )
    if args.seqlen is not None:
        raise ValueError("--seqlen is only used with --seqlen_mode dense")
    return args.seq_lens, batch_size, sum(args.seq_lens)


def parse_gdn_args(
    line: Sequence[str] | None, parser: argparse.ArgumentParser
) -> argparse.Namespace:
    """Add GDN arguments to the shared parser and parse the command line."""
    routine_parser = argparse.ArgumentParser(add_help=False)
    routine_parser.add_argument("-R", "--routine")
    routine_args, _ = routine_parser.parse_known_args(line)
    if routine_args.routine == "gdn_decode":
        parser.add_argument(
            "--backend",
            choices=("auto",),
            default="auto",
            help="MATE GDN backend.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            required=True,
            help="Batch size of the benchmark case.",
        )
        parser.add_argument(
            "--num_head_q",
            type=int,
            required=True,
            help="Number of query heads.",
        )
        parser.add_argument(
            "--num_head_v",
            type=int,
            required=True,
            help="Number of value heads.",
        )
        parser.add_argument(
            "--head_size",
            type=int,
            default=128,
            help="Head dimension.",
        )
        parser.add_argument(
            "--dtype",
            choices=("fp16", "bf16"),
            default="bf16",
            help="Input dtype for q/k/v.",
        )
        parser.add_argument(
            "--cache_indices",
            choices=("none", "identity", "both"),
            default="none",
            help="Benchmark without cache_indices, with identity cache_indices, or both.",
        )
        parser.add_argument(
            "--num-tests",
            type=int,
            default=10,
            help="Number of timing iterations.",
        )
        args = parser.parse_args(line)
        if args.verbose >= 1:
            logger.info(f"[INFO] {args = }")
        return args

    if routine_args.routine == "gdn_prefill":
        parser.add_argument(
            "--backend",
            choices=("auto",),
            default="auto",
            help="MATE GDN backend.",
        )
        parser.add_argument(
            "--seqlen_mode",
            choices=("dense", "varlen"),
            default="varlen",
            help="Dense BSHD input or packed variable-length input.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=None,
            help="Dense batch size; for varlen it must match --seq_lens.",
        )
        parser.add_argument(
            "--seqlen",
            type=int,
            default=None,
            help="Sequence length for dense mode.",
        )
        parser.add_argument(
            "--seq_lens",
            type=_parse_seq_lens,
            default=None,
            help="Comma-separated sequence lengths for varlen mode.",
        )
        parser.add_argument(
            "--num_head_q",
            type=int,
            required=True,
            help="Number of query heads.",
        )
        parser.add_argument(
            "--num_head_v",
            type=int,
            required=True,
            help="Number of value heads.",
        )
        parser.add_argument(
            "--head_size",
            type=int,
            default=128,
            help="Head dimension.",
        )
        parser.add_argument(
            "--dtype",
            choices=("fp16", "bf16"),
            default="bf16",
            help="Input dtype for q/k/v.",
        )
        parser.add_argument(
            "--chunk_size",
            type=int,
            default=64,
            help="Chunk size for the prefill path.",
        )
        parser.add_argument(
            "--num-tests",
            type=int,
            default=10,
            help="Number of timing iterations.",
        )
        parser.add_argument(
            "--qk-l2norm",
            dest="qk_l2norm",
            action="store_true",
        )
        parser.add_argument(
            "--no-qk-l2norm",
            dest="qk_l2norm",
            action="store_false",
        )
        parser.set_defaults(qk_l2norm=True)
        parser.add_argument(
            "--is-log-space",
            dest="is_log_space",
            action="store_true",
        )
        parser.add_argument(
            "--no-is-log-space",
            dest="is_log_space",
            action="store_false",
        )
        parser.set_defaults(is_log_space=True)
        parser.add_argument(
            "--initial-state",
            dest="initial_state",
            action="store_true",
        )
        parser.add_argument(
            "--no-initial-state",
            dest="initial_state",
            action="store_false",
        )
        parser.set_defaults(initial_state=True)
        parser.add_argument(
            "--output-final-state",
            dest="output_final_state",
            action="store_true",
        )
        parser.add_argument(
            "--no-output-final-state",
            dest="output_final_state",
            action="store_false",
        )
        parser.set_defaults(output_final_state=True)
        args = parser.parse_args(line)
        if args.verbose >= 1:
            logger.info(f"[INFO] {args = }")
        return args

    raise ValueError(f"Unsupported GDN routine: {routine_args.routine}")
