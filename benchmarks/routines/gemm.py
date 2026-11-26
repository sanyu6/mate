"""GEMM benchmarks using one structured operator lifecycle for every routine."""

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
from mate.testing.operators.gemm import GemmApi, GemmInputs, GemmOperator, GemmWorkload
from mate.testing.operators.operator import OpMode
from mate.testing.metrics import (
    get_bmm_metrics,
    get_masked_moe_gemm_metrics,
    get_mixed_dtype_moe_gemm_metrics,
    get_ragged_k_moe_gemm_metrics,
    get_ragged_m_moe_gemm_metrics,
)
from mate.testing.utils import bench_gpu_time
from .mate_benchmark_utils import (
    dtype_str_to_torch_dtype,
    filter_backends_by_arch,
    maybe_print_kineto_metrics,
    print_perf_metrics,
)

logger = logging.getLogger(__name__)


def get_gemm_metrics(
    w: GemmWorkload, x: GemmInputs, median_ms: float
) -> tuple[float, float]:
    """Select the logical metric helper for this GEMM API and convert to rates."""
    assert x.a is not None and x.b is not None and x.out is not None
    if w.api is GemmApi.BMM:
        metrics = get_bmm_metrics(x.a, x.b, x.out, k=w.k)
    elif w.api is GemmApi.RAGGED_K_MOE:
        metrics = get_ragged_k_moe_gemm_metrics(
            x.a,
            x.b,
            x.group_k_idx,
            x.out,
            total_k=x.total_k,
            m=w.m,
            n=w.n,
            extra_load_tensors=(x.scale_a, x.scale_b),
        )
    elif w.api is GemmApi.MASKED_MOE:
        metrics = get_masked_moe_gemm_metrics(
            x.a,
            x.b,
            x.masked_m,
            x.out,
            valid_m=x.valid_m,
            n=w.n,
            k=w.k,
            a_scale=x.scale_a,
            granu_k=x.scale_granularity_mnk[2],
            b_scale=x.scale_b,
        )
    elif w.api is GemmApi.RAGGED_M_MOE:
        metrics = get_ragged_m_moe_gemm_metrics(
            x.a,
            x.b,
            x.m_indices,
            x.out,
            valid_m=x.valid_m,
            n=w.n,
            k=w.k,
            a_scale=x.scale_a,
            granu_k=x.scale_granularity_mnk[2],
            extra_load_tensors=(x.scale_b,),
        )
    elif w.api is GemmApi.MIXED_DTYPE_MOE:
        metrics = get_mixed_dtype_moe_gemm_metrics(
            x.a,
            x.b,
            x.out,
            mixed_dtype=w.mixed_dtype,
            valid_m=x.valid_m,
            n=w.n,
            k=w.k,
            masked_m=x.masked_m,
            m_indices=x.m_indices,
            a_scale=x.scale_a,
            b_scale=x.scale_b,
            residual_e8m0=x.residual_e8m0,
            epilogue_fp32=x.epilogue_fp32,
        )
    else:
        raise ValueError(f"Unsupported GEMM API: {w.api}")
    return metrics.to_perf(median_ms)


def _workload(args: argparse.Namespace) -> GemmWorkload:
    apis = {
        "bmm_fp16": (GemmApi.BMM, False),
        "bmm_fp8": (GemmApi.BMM, True),
        "masked_moe_gemm_16bit": (GemmApi.MASKED_MOE, False),
        "masked_moe_gemm_8bit": (GemmApi.MASKED_MOE, True),
        "ragged_m_moe_gemm_16bit": (GemmApi.RAGGED_M_MOE, False),
        "ragged_m_moe_gemm_8bit": (GemmApi.RAGGED_M_MOE, True),
        "ragged_k_moe_gemm_16bit": (GemmApi.RAGGED_K_MOE, False),
        "ragged_k_moe_gemm_8bit": (GemmApi.RAGGED_K_MOE, True),
        "ragged_moe_gemm_mixed_dtype": (GemmApi.MIXED_DTYPE_MOE, True),
        "masked_moe_gemm_mixed_dtype": (GemmApi.MIXED_DTYPE_MOE, True),
    }
    if args.routine not in apis:
        raise ValueError(f"Unsupported GEMM routine: {args.routine}")
    api, quantized = apis[args.routine]
    return GemmWorkload(
        api=api,
        m=args.m,
        n=args.n,
        k=args.k,
        batch_size=args.batch_size,
        group_size=args.group_size,
        input_dtype=dtype_str_to_torch_dtype(args.input_dtype),
        out_dtype=dtype_str_to_torch_dtype(args.out_dtype),
        backend=args.backend,
        quantized=quantized,
        mixed_dtype=args.mixed_dtype,
        mixed_layout="masked" if args.routine.startswith("masked") else "ragged",
        quant_tile=args.quant_tile,
        scale_major_mode=args.scale_major_mode,
        alignment_m=args.alignment_m,
        group_length=args.group_length,
        seed=args.random_seed,
    )


def run_gemm_test(args: argparse.Namespace) -> None:
    workload = _workload(args)
    backends = filter_backends_by_arch(
        [args.backend], args.routine, torch.device("musa")
    )
    if not backends:
        logger.error("[ERROR] No backends to test. Exiting.")
        return
    operator = GemmOperator()
    mode = OpMode.DRY_RUN if is_dry_run_enabled() else OpMode.NORMAL
    for backend in backends:
        w = replace(workload, backend=backend)
        with operator.use_mode(mode):
            inputs = operator.generate(w)
            initial_out = (
                inputs.out.clone()
                if w.api is GemmApi.RAGGED_K_MOE and mode is OpMode.NORMAL
                else None
            )
            reference = operator.reference(inputs) if mode is OpMode.NORMAL else None
            try:
                outputs = operator(inputs)
                raise_complete_if_dry_run()
            except MateDryRunComplete:
                logger.info("[DRY_RUN] %s completed for %s", args.routine, backend)
                continue
            operator.verify(inputs, outputs, reference)
            logger.info("[VERIFY] %s passed for %s", args.routine, backend)
            if initial_out is not None:
                inputs.out.copy_(initial_out)
            del initial_out, reference, outputs
            torch.musa.synchronize()

            def run_fn():
                operator(inputs)

            times = bench_gpu_time(
                run_fn, l2_flush=True, l2_flush_device="musa", sleep_after_run=True
            )
            ms, sd = float(np.median(times)), float(np.std(times))
            tflops, bandwidth = get_gemm_metrics(w, inputs, ms)
            print_perf_metrics(backend, ms, sd, tflops, bandwidth)
            maybe_print_kineto_metrics(args, backend, run_fn, ms, tflops, bandwidth)


def parse_gemm_args(
    line: Sequence[str] | None, parser: argparse.ArgumentParser
) -> argparse.Namespace:
    """Add GEMM arguments to the shared parser and parse the command line."""
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size of the benchmark case.",
    )
    parser.add_argument("--m", type=int, required=True, help="GEMM M dimension.")
    parser.add_argument("--n", type=int, required=True, help="GEMM N dimension.")
    parser.add_argument("--k", type=int, required=True, help="GEMM K dimension.")
    parser.add_argument(
        "--quant_tile",
        type=int,
        default=128,
        help="Quantization tile size used by quantized GEMM routines.",
    )
    parser.add_argument(
        "--alignment_m",
        type=int,
        default=128,
        help="M-alignment requirement used by ragged MoE routines.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=1,
        help="Number of groups in a grouped GEMM operation.",
    )
    parser.add_argument(
        "--group-length",
        choices=("random", "fixed"),
        default="fixed",
        help=(
            "Per-group M or K distribution: random around the requested "
            "dimension, or fixed for every group."
        ),
    )
    parser.add_argument(
        "--scale_major_mode",
        choices=("MN", "K"),
        default="K",
        help="Scale tensor layout.",
    )
    parser.add_argument(
        "--input_dtype",
        default="fp8_e4m3",
        help="Input data type.",
    )
    parser.add_argument(
        "--mixed_dtype",
        choices=("s4fp8", "fp4fp8"),
        default="s4fp8",
        help="Mixed input format.",
    )
    parser.add_argument(
        "--out_dtype",
        default="bfloat16",
        help="Output data type.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "mubin", "mudnn", "mutlass"),
        default="auto",
        help="MATE GEMM backend.",
    )
    args = parser.parse_args(line)
    has_input_dtype_arg = any(
        token == "--input_dtype" or token.startswith("--input_dtype=")
        for token in line or ()
    )
    has_out_dtype_arg = any(
        token == "--out_dtype" or token.startswith("--out_dtype=")
        for token in line or ()
    )
    sixteen_bit_routines = {
        "bmm_fp16",
        "masked_moe_gemm_16bit",
        "ragged_k_moe_gemm_16bit",
        "ragged_m_moe_gemm_16bit",
    }
    if args.routine in sixteen_bit_routines and not has_input_dtype_arg:
        args.input_dtype = "bfloat16"
    ragged_k_routines = {
        "ragged_k_moe_gemm_16bit",
        "ragged_k_moe_gemm_8bit",
    }
    if args.routine in ragged_k_routines and not has_out_dtype_arg:
        args.out_dtype = "float32"
    if args.verbose >= 1:
        logger.info(f"[INFO] {args = }")
    return args
