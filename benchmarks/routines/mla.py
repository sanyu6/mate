"""MLA benchmarks using the shared operator for inputs and verification."""

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
from mate.testing.metrics import (
    get_dense_mla_metrics,
    get_pack8_mla_metrics,
    get_sparse_mla_metrics,
)
from mate.testing.operators.mla import MlaApi, MlaInputs, MlaOperator, MlaWorkload
from mate.testing.operators.operator import OpMode
from mate.testing.utils import bench_gpu_time

from .mate_benchmark_utils import (
    dtype_str_to_torch_dtype,
    filter_backends_by_arch,
    maybe_print_kineto_metrics,
    print_perf_metrics,
)

logger = logging.getLogger(__name__)

DEFAULT_HEADDIM_LATENT = 512
DEFAULT_HEADDIM_ROPE = 64
DEFAULT_PAGE_SIZE = 64


def get_mla_metrics(
    workload: MlaWorkload, inputs: MlaInputs, median_ms: float
) -> tuple[float, float]:
    """Select the existing logical metric definition for this MLA API."""
    assert inputs.q is not None
    if workload.api is MlaApi.SPARSE_FWD_PACK8:
        assert inputs.kv is not None and inputs.indices is not None
        metrics = get_pack8_mla_metrics(
            inputs.q, inputs.kv, inputs.indices, head_dim_v=inputs.head_dim_v
        )
    elif workload.api is MlaApi.KVCACHE and not workload.is_fp8_kvcache:
        assert inputs.cache_seqlens is not None
        metrics = get_dense_mla_metrics(
            inputs.q,
            inputs.cache_seqlens,
            seqlen_q=workload.seqlen_q,
            num_head_q=workload.num_head_q,
            headdim_latent=workload.headdim_latent,
            headdim_rope=workload.headdim_rope,
            causal=workload.causal,
        )
    else:
        kv = inputs.k_cache if workload.is_fp8_kvcache else inputs.kv
        assert kv is not None and inputs.indices is not None
        metrics = get_sparse_mla_metrics(
            inputs.q,
            inputs.indices,
            inputs.topk_length,
            batch_size=workload.batch_size,
            seqlen_q=workload.seqlen_q,
            seqlen_kv=workload.seqlen_kv,
            num_head_q=workload.num_head_q,
            headdim_latent=workload.headdim_latent,
            headdim_rope=workload.headdim_rope,
            kv_bytes_per_token=kv.shape[-1] * kv.element_size(),
        )
    return metrics.to_perf(median_ms)


def _workload(args: argparse.Namespace) -> MlaWorkload:
    """Translate CLI choices; tensor preparation belongs to the operator."""
    apis = {
        "flash_mla_sparse_fwd": MlaApi.SPARSE_FWD,
        "flash_mla_sparse_fwd_pack8": MlaApi.SPARSE_FWD_PACK8,
        "flash_mla_with_kvcache": MlaApi.KVCACHE,
    }
    if args.routine not in apis:
        raise ValueError(f"Unsupported MLA routine: {args.routine}")
    api = apis[args.routine]
    if api is MlaApi.KVCACHE and args.seqlen_mode_q != "bshd":
        raise ValueError("flash_mla_with_kvcache requires BSHD Q layout")
    return MlaWorkload(
        api=api,
        batch_size=args.batch_size,
        seqlen_q=args.seqlen_q,
        seqlen_kv=args.seqlen_kv,
        num_head_q=args.num_head_q,
        num_head_kv=args.num_head_kv,
        headdim_latent=args.headdim_latent,
        headdim_rope=args.headdim_rope,
        dtype=dtype_str_to_torch_dtype(args.q_dtype),
        kv_dtype=dtype_str_to_torch_dtype(args.kv_dtype),
        seed=args.random_seed,
        page_size=args.page_size,
        topk=args.topk,
        topk_length_mode=args.topk_length_mode,
        seqlen_mode_kv=args.seqlen_mode_kv,
        causal=args.causal,
        causal_window=args.causal_window,
        is_fp8_kvcache=args.is_fp8_kvcache,
        attn_sink=args.attn_sink,
    )


def run_mla_test(args: argparse.Namespace) -> None:
    """Verify each selected backend, then measure only operator calls."""
    workload = _workload(args)
    backends = filter_backends_by_arch(
        list(dict.fromkeys(args.backends)), args.routine, torch.device("musa")
    )
    if not backends:
        logger.error("[ERROR] No backends to test. Exiting.")
        return
    operator = MlaOperator()
    mode = OpMode.DRY_RUN if is_dry_run_enabled() else OpMode.NORMAL
    for backend in backends:
        with operator.use_mode(mode):
            # MLA's public APIs select their implementation internally and do
            # not accept a backend argument. Only supported labels reach here.
            inputs = operator.generate(workload)
            if inputs.q_padded is not None:
                logger.info(
                    "[INFO] V3.2 prefill: %d logical heads padded to 64 kernel heads; "
                    "throughput and bandwidth use logical workload counts",
                    workload.num_head_q,
                )
            reference = operator.reference(inputs) if mode is OpMode.NORMAL else None
            try:
                outputs = operator(inputs)
                raise_complete_if_dry_run()
            except MateDryRunComplete:
                logger.info("[DRY_RUN] %s completed for %s", args.routine, backend)
                continue
            operator.verify(inputs, outputs, reference)
            logger.info(
                "[VERIFY] %s passed for %s (all outputs)", args.routine, backend
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
            tflops, bandwidth = get_mla_metrics(workload, inputs, ms)
            print_perf_metrics(backend, ms, sd, tflops, bandwidth)
            maybe_print_kineto_metrics(args, backend, run_fn, ms, tflops, bandwidth)
            del inputs


def parse_mla_args(
    line: Sequence[str] | None, parser: argparse.ArgumentParser
) -> argparse.Namespace:
    parser.add_argument(
        "--backends", nargs="+", default=None, choices=("auto", "tilelang")
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seqlen_q", type=int, default=1)
    parser.add_argument("--seqlen_mode_q", choices=("bshd", "varlen"), default="bshd")
    parser.add_argument("--seqlen_mode_kv", choices=("full", "random"), default="full")
    parser.add_argument("--seqlen_kv", type=int, required=True)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--num_head_q", type=int, default=128)
    parser.add_argument("--num_head_kv", type=int, default=1)
    parser.add_argument("--headdim_latent", type=int, default=DEFAULT_HEADDIM_LATENT)
    parser.add_argument("--headdim_rope", type=int, default=DEFAULT_HEADDIM_ROPE)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument(
        "--topk_length_mode", choices=("full", "random"), default="full"
    )
    parser.add_argument("--q_dtype", default="bfloat16")
    parser.add_argument("--kv_dtype", default="bfloat16")
    parser.add_argument("--causal", action="store_true", default=False)
    parser.add_argument("--attn_sink", action="store_true", default=False)
    parser.add_argument("--is-fp8-kvcache", action="store_true", default=False)
    parser.add_argument("--causal-window", type=int, default=0)
    args = parser.parse_args(line)
    if args.backends is None:
        args.backends = (
            ["auto"] if args.routine == "flash_mla_with_kvcache" else ["tilelang"]
        )
    if args.verbose >= 1:
        logger.info(f"[INFO] args = {args}")
    return args
