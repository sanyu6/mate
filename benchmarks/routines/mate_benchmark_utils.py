import logging
import random
import sys

import numpy as np
import torch
import torch_musa  # noqa: F401

from mate.testing.utils import bench_kineto

logger = logging.getLogger(__name__)


benchmark_apis = {
    "attention": [
        "flash_attn_varlen_func",
        "flash_attn_with_kvcache",
    ],
    "attention_bwd": [
        "flashattn_varlen_bwd_interface",
    ],
    "mla": [
        "flash_mla_sparse_fwd",
        "flash_mla_sparse_fwd_pack8",
        "flash_mla_with_kvcache",
    ],
    "gemm": [
        "bmm_fp16",
        "bmm_fp8",
        "masked_moe_gemm_16bit",
        "masked_moe_gemm_8bit",
        "ragged_m_moe_gemm_16bit",
        "ragged_m_moe_gemm_8bit",
        "ragged_k_moe_gemm_16bit",
        "ragged_k_moe_gemm_8bit",
        "ragged_moe_gemm_mixed_dtype",
        "masked_moe_gemm_mixed_dtype",
    ],
    "gdn": [
        "gdn_decode",
        "gdn_prefill",
    ],
    "kda": [
        "chunk_kda",
    ],
}


routine_arch_to_supported_backends = {
    "flash_attn_varlen_func": {
        "mp31": ["auto", "mutlass", "mubin"],
    },
    "flashattn_varlen_bwd_interface": {
        "mp31": ["tilelang"],
    },
    "flash_attn_with_kvcache": {
        "mp31": ["auto"],
    },
    "flash_mla_sparse_fwd": {
        "mp31": ["tilelang"],
    },
    "flash_mla_sparse_fwd_pack8": {
        "mp31": ["tilelang"],
    },
    "flash_mla_with_kvcache": {
        "mp31": ["auto"],
    },
    "bmm_fp16": {
        "mp31": ["auto", "mudnn"],
    },
    "bmm_fp8": {
        "mp31": ["auto", "mudnn"],
    },
    "masked_moe_gemm_16bit": {
        "mp31": ["auto", "mubin", "mutlass"],
    },
    "masked_moe_gemm_8bit": {
        "mp31": ["auto", "mubin", "mutlass"],
    },
    "ragged_m_moe_gemm_16bit": {
        "mp31": ["auto", "mubin", "mutlass"],
    },
    "ragged_m_moe_gemm_8bit": {
        "mp31": ["auto", "mubin", "mutlass"],
    },
    "ragged_k_moe_gemm_16bit": {
        "mp31": ["auto", "mubin"],
    },
    "ragged_k_moe_gemm_8bit": {
        "mp31": ["auto", "mubin"],
    },
    "ragged_moe_gemm_mixed_dtype": {
        "mp31": ["auto", "mubin"],
    },
    "masked_moe_gemm_mixed_dtype": {
        "mp31": ["auto", "mubin"],
    },
    "gdn_decode": {
        "mp31": ["auto"],
    },
    "gdn_prefill": {
        "mp31": ["auto"],
    },
    "chunk_kda": {
        "mp31": ["auto"],
    },
}


def set_random_seed(seed: int) -> None:
    """Seed benchmark input generators across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.musa.manual_seed(seed)


def configure_benchmark_logging(verbose: int) -> None:
    """Configure benchmark logging while preserving existing line formats."""
    logging.basicConfig(
        level=logging.DEBUG if verbose >= 2 else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )


def dtype_str_to_torch_dtype(dtype: str) -> torch.dtype:
    """Convert a benchmark dtype name to a PyTorch dtype."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }
    try:
        return dtype_map[dtype]
    except KeyError:
        raise ValueError(f"Unsupported dtype: {dtype}") from None


def print_perf_metrics(
    backend: str,
    median_time: float,
    std_time: float,
    tflops: float,
    bandwidth_gb_per_sec: float,
) -> None:
    """Log benchmark latency, compute throughput, and memory throughput."""
    output_backend_width = max(15, len(backend))
    logger.info(
        f"[PERF] {backend.ljust(output_backend_width)}:: "
        f"median time {median_time:.3f} ms; std {std_time:.3f} ms; "
        f"achieved TFLOP {tflops:.3f} TFLOP/s; "
        f"achieved Bandwidth {bandwidth_gb_per_sec:.3f} GB/s"
    )


def maybe_print_kineto_metrics(
    args,
    backend: str,
    run_fn,
    event_median_ms: float,
    event_tflops: float,
    event_bandwidth_gb_per_sec: float,
) -> None:
    """Optionally profile one implementation kernel and print its performance."""
    kernel_name = getattr(args, "mupti_kernel_name", None)
    if not kernel_name:
        return

    kineto_seconds = bench_kineto(
        run_fn,
        kernel_name,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    kineto_ms = kineto_seconds * 1e3
    if kineto_ms <= 0:
        logger.warning(
            f"[WARNING] Kineto did not find kernel name containing {kernel_name!r}."
        )
        return

    logical_flops = event_tflops * 1e9 * event_median_ms
    logical_bytes = event_bandwidth_gb_per_sec * 1e6 * event_median_ms
    kineto_tflops = logical_flops / (1e9 * kineto_ms)
    kineto_bandwidth = logical_bytes / (1e6 * kineto_ms)
    print_perf_metrics(
        f"{backend}:kineto",
        kineto_ms,
        0.0,
        kineto_tflops,
        kineto_bandwidth,
    )


def filter_backends_by_arch(
    backends: list[str], routine: str, device: object
) -> list[str]:
    """Remove backends unsupported by the current MUSA architecture."""
    if routine not in routine_arch_to_supported_backends:
        raise ValueError(f"Unsupported routine: {routine}")

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("MUSA is not available")

    try:
        major, minor = torch.musa.get_device_capability(device)
    except Exception as capability_exc:
        try:
            properties = torch.musa.get_device_properties(device)
            major, minor = properties.major, properties.minor
        except Exception as properties_exc:
            raise RuntimeError(
                "Unable to determine the MUSA device architecture: "
                f"{capability_exc}; fallback failed: {properties_exc}"
            ) from properties_exc

    arch = f"mp{int(major)}{int(minor)}"
    supported_backends = routine_arch_to_supported_backends[routine].get(arch, [])

    for backend in list(backends):
        if backend not in supported_backends:
            backends.remove(backend)
            logger.warning(
                f"[WARNING] {backend} for routine {routine} is not supported "
                f"on MUSA architecture {arch.upper()}. Skipping."
            )

    return backends
