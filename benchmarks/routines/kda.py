"""KDA benchmarks using structured operators and independent verification."""

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
from mate.testing.metrics import get_kda_metrics as get_chunk_kda_metrics
from mate.testing.operators.kda import KdaApi, KdaWorkload, KdaInputs, KdaOperator
from mate.testing.operators.operator import OpMode
from mate.testing.utils import bench_gpu_time
from .mate_benchmark_utils import (
    dtype_str_to_torch_dtype,
    filter_backends_by_arch,
    maybe_print_kineto_metrics,
    print_perf_metrics,
)


logger = logging.getLogger(__name__)


def get_kda_metrics(
    workload: KdaWorkload, inputs: KdaInputs, median_ms: float
) -> tuple[float, float]:
    x, w = inputs, workload
    metrics = get_chunk_kda_metrics(
        (
            x.q,
            x.k,
            x.v,
            x.g,
            x.beta,
            x.output,
            x.initial_state,
            x.final_state,
            x.cu_seqlens,
            x.A_log,
            x.dt_bias,
        ),
        total_tokens=sum(w.lengths),
        num_head_v=w.num_head_v,
        head_dim=w.headdim,
    )
    return metrics.to_perf(median_ms)


def _workload(args: argparse.Namespace) -> KdaWorkload:
    if args.routine != "chunk_kda":
        raise ValueError(f"Unsupported KDA routine: {args.routine}")
    lengths, batch_size, _ = _resolve_sequence_shape(args)
    return KdaWorkload(
        api=KdaApi.CHUNK,
        batch_size=batch_size,
        seqlen=max(lengths),
        seq_lens=tuple(lengths) if args.seqlen_mode == "varlen" else None,
        num_head_qk=args.num_head_qk,
        num_head_v=args.num_head_v,
        headdim=args.headdim,
        dtype=dtype_str_to_torch_dtype(args.input_dtype),
        seed=args.random_seed,
        state_mode=args.state_mode,
        state_in=args.state_in,
        state_out=args.state_out,
        gate_params=args.gate_params,
        qk_l2norm=args.qk_l2norm,
        lower_bound=args.lower_bound,
    )


def run_kda_test(args: argparse.Namespace) -> None:
    """Verify the output and recurrent state before timing operator calls."""
    workload = _workload(args)
    backends = filter_backends_by_arch(
        [args.backend], args.routine, torch.device("musa")
    )
    if not backends:
        logger.error("[ERROR] No backends to test. Exiting.")
        return
    operator = KdaOperator()
    mode = OpMode.DRY_RUN if is_dry_run_enabled() else OpMode.NORMAL
    for backend in backends:
        with operator.use_mode(mode):
            inputs = operator.generate(workload)
            snapshot = inputs.clone() if mode is OpMode.NORMAL else None
            reference = operator.reference(inputs) if mode is OpMode.NORMAL else None
            try:
                outputs = operator(inputs)
                raise_complete_if_dry_run()
            except MateDryRunComplete:
                logger.info("[DRY_RUN] %s completed for %s", args.routine, backend)
                continue
            operator.verify(inputs, outputs, reference)
            logger.info(
                "[VERIFY] %s passed for %s (output and state)", args.routine, backend
            )
            inputs._copy_from(snapshot)
            del snapshot, reference, outputs
            torch.musa.synchronize()

            def run_fn() -> None:
                operator(inputs)

            times = bench_gpu_time(
                run_fn, l2_flush=True, l2_flush_device="musa", sleep_after_run=True
            )
            ms, sd = float(np.median(times)), float(np.std(times))
            tflops, bandwidth = get_kda_metrics(workload, inputs, ms)
            print_perf_metrics(backend, ms, sd, tflops, bandwidth)
            maybe_print_kineto_metrics(args, backend, run_fn, ms, tflops, bandwidth)
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


def _resolve_sequence_shape(args: argparse.Namespace) -> tuple[list[int], int, int]:
    """Return per-sequence lengths, batch size, and total token count."""
    if args.seqlen_mode == "dense":
        if args.seq_lens is not None:
            raise ValueError("--seq_lens requires --seqlen_mode varlen")
        if args.seqlen is None or args.seqlen <= 0:
            raise ValueError("Dense chunk_kda requires a positive --seqlen")
        batch_size = 1 if args.batch_size is None else args.batch_size
        if batch_size <= 0:
            raise ValueError("--batch_size must be positive")
        seq_lens = [args.seqlen] * batch_size
        return seq_lens, batch_size, batch_size * args.seqlen

    if args.seq_lens is None:
        raise ValueError("Varlen chunk_kda requires --seq_lens")
    batch_size = len(args.seq_lens)
    if args.batch_size is not None and args.batch_size != batch_size:
        raise ValueError(
            "For varlen chunk_kda, --batch_size must match the number of "
            "entries in --seq_lens"
        )
    if args.seqlen is not None:
        raise ValueError("--seqlen is only used with --seqlen_mode dense")
    return args.seq_lens, batch_size, sum(args.seq_lens)


def parse_kda_args(
    line: Sequence[str] | None, parser: argparse.ArgumentParser
) -> argparse.Namespace:
    """Add KDA arguments to the shared parser and parse the command line."""
    parser.add_argument(
        "--backend",
        choices=("auto",),
        default="auto",
        help="MATE KDA backend.",
    )
    parser.add_argument(
        "--seqlen_mode",
        choices=("dense", "varlen"),
        default="dense",
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
    parser.add_argument("--num_head_qk", type=int, required=True)
    parser.add_argument("--num_head_v", type=int, required=True)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument(
        "--input_dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--state_mode",
        choices=("none", "bfloat16", "fp32"),
        default="none",
    )
    parser.add_argument("--state_in", action="store_true")
    parser.add_argument("--state_out", action="store_true")

    gate_group = parser.add_mutually_exclusive_group()
    gate_group.add_argument("--gate-params", dest="gate_params", action="store_true")
    gate_group.add_argument(
        "--no-gate-params", dest="gate_params", action="store_false"
    )
    parser.set_defaults(gate_params=True)

    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument("--qk-l2norm", dest="qk_l2norm", action="store_true")
    norm_group.add_argument("--no-qk-l2norm", dest="qk_l2norm", action="store_false")
    parser.set_defaults(qk_l2norm=True)
    parser.add_argument("--lower_bound", type=float, default=-5.0)

    args = parser.parse_args(line)
    if args.verbose >= 1:
        logger.info(f"[INFO] {args = }")
    return args
