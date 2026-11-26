#!/usr/bin/env python3
"""Run one small smoke case for every routine registered by mate_benchmark.py.

This is a dispatch/return-value check, not a performance sweep.  Run it inside
the MUSA container with the same Python environment as the benchmark runner.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    name: str
    args: tuple[str, ...]


COMMON = ("--random-seed", "42")

CASES = (
    Case(
        "flash_attn_varlen_func",
        (
            "-R",
            "flash_attn_varlen_func",
            "--backends",
            "auto",
            "--batch_size",
            "1",
            "--seqlen_q",
            "16",
            "--seqlen_kv",
            "16",
            "--seqlen_mode_q",
            "varlen",
            "--seqlen_mode_kv",
            "varlen",
            "--num_head_q",
            "2",
            "--num_head_kv",
            "2",
            "--headdim_qk",
            "64",
            "--q_dtype",
            "bfloat16",
            "--kv_dtype",
            "bfloat16",
        ),
    ),
    Case(
        "flash_attn_with_kvcache",
        (
            "-R",
            "flash_attn_with_kvcache",
            "--backends",
            "auto",
            "--batch_size",
            "1",
            "--seqlen_q",
            "1",
            "--seqlen_kv",
            "16",
            "--seqlen_mode_q",
            "bshd",
            "--seqlen_mode_kv",
            "bshd",
            "--num_head_q",
            "2",
            "--num_head_kv",
            "2",
            "--headdim_qk",
            "64",
            "--q_dtype",
            "bfloat16",
            "--kv_dtype",
            "bfloat16",
        ),
    ),
    Case(
        "flashattn_varlen_bwd_interface",
        (
            "-R",
            "flashattn_varlen_bwd_interface",
            "--backends",
            "tilelang",
            "--batch_size",
            "1",
            "--seqlen_q",
            "16",
            "--seqlen_kv",
            "16",
            "--num_head_q",
            "2",
            "--num_head_kv",
            "2",
            "--headdim_qk",
            "64",
            "--q_dtype",
            "bfloat16",
            "--kv_dtype",
            "bfloat16",
            "--warmup",
            "0",
            "--repeat",
            "1",
        ),
    ),
    Case(
        "gdn_decode",
        (
            "-R",
            "gdn_decode",
            "--batch_size",
            "1",
            "--num_head_q",
            "2",
            "--num_head_v",
            "2",
            "--head_size",
            "128",
            "--dtype",
            "bf16",
            "--num-tests",
            "1",
        ),
    ),
    Case(
        "gdn_prefill",
        (
            "-R",
            "gdn_prefill",
            "--seqlen_mode",
            "dense",
            "--batch_size",
            "1",
            "--seqlen",
            "16",
            "--num_head_q",
            "2",
            "--num_head_v",
            "2",
            "--head_size",
            "128",
            "--dtype",
            "bf16",
            "--num-tests",
            "1",
        ),
    ),
    Case(
        "chunk_kda",
        (
            "-R",
            "chunk_kda",
            "--seqlen_mode",
            "dense",
            "--batch_size",
            "1",
            "--seqlen",
            "16",
            "--num_head_qk",
            "2",
            "--num_head_v",
            "2",
            "--headdim",
            "128",
            "--input_dtype",
            "bfloat16",
        ),
    ),
    Case(
        "flash_mla_sparse_fwd",
        (
            "-R",
            "flash_mla_sparse_fwd",
            "--seqlen_q",
            "1",
            "--seqlen_kv",
            "64",
            "--num_head_q",
            "128",
            "--num_head_kv",
            "1",
            "--headdim_latent",
            "512",
            "--headdim_rope",
            "0",
            "--topk",
            "64",
            "--q_dtype",
            "bfloat16",
            "--kv_dtype",
            "bfloat16",
            "--backends",
            "tilelang",
        ),
    ),
    Case(
        "flash_mla_sparse_fwd_pack8",
        (
            "-R",
            "flash_mla_sparse_fwd_pack8",
            "--seqlen_q",
            "8",
            "--seqlen_kv",
            "64",
            "--num_head_q",
            "8",
            "--num_head_kv",
            "1",
            "--headdim_latent",
            "512",
            "--headdim_rope",
            "0",
            "--topk",
            "64",
            "--q_dtype",
            "bfloat16",
            "--kv_dtype",
            "bfloat16",
            "--backends",
            "tilelang",
        ),
    ),
    Case(
        "flash_mla_with_kvcache",
        (
            "-R",
            "flash_mla_with_kvcache",
            "--seqlen_q",
            "1",
            "--seqlen_kv",
            "64",
            "--num_head_q",
            "128",
            "--num_head_kv",
            "1",
            "--headdim_latent",
            "512",
            "--headdim_rope",
            "64",
            "--q_dtype",
            "bfloat16",
            "--kv_dtype",
            "bfloat16",
            "--backends",
            "auto",
        ),
    ),
    Case(
        "bmm_fp16",
        (
            "-R",
            "bmm_fp16",
            "--batch_size",
            "1",
            "--m",
            "16",
            "--n",
            "16",
            "--k",
            "16",
            "--input_dtype",
            "bfloat16",
            "--out_dtype",
            "bfloat16",
            "--backend",
            "auto",
        ),
    ),
    Case(
        "bmm_fp8",
        (
            "-R",
            "bmm_fp8",
            "--batch_size",
            "1",
            "--m",
            "16",
            "--n",
            "16",
            "--k",
            "16",
            "--input_dtype",
            "fp8_e4m3",
            "--out_dtype",
            "bfloat16",
            "--backend",
            "auto",
        ),
    ),
)

# MoE variants share the common GEMM parser.  Their kernels impose additional
# alignment/layout constraints, so use the smallest generally valid tile.
CASES += tuple(
    Case(
        name,
        (
            "-R",
            name,
            "--m",
            "128",
            "--n",
            "128",
            "--k",
            "128",
            "--group_size",
            "2",
            "--backend",
            "auto",
        ),
    )
    for name in (
        "masked_moe_gemm_16bit",
        "masked_moe_gemm_8bit",
        "ragged_m_moe_gemm_16bit",
        "ragged_m_moe_gemm_8bit",
        "ragged_k_moe_gemm_16bit",
        "ragged_k_moe_gemm_8bit",
        "ragged_moe_gemm_mixed_dtype",
        "masked_moe_gemm_mixed_dtype",
    )
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Set MATE_DRY_RUN=1 and require each case to report [DRY_RUN].",
    )
    args = parser.parse_args()

    runner = [sys.executable, "benchmarks/mate_benchmark.py"]
    passed = failed = 0
    total = len(CASES)
    for index, case in enumerate(CASES, start=1):
        command = runner + list(case.args) + list(COMMON)
        environment = os.environ.copy()
        if args.dry_run:
            environment["MATE_DRY_RUN"] = "1"
        started = time.monotonic()
        try:
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=args.timeout,
                env=environment,
            )
            elapsed = time.monotonic() - started
            output = result.stdout + result.stderr
            dry_run_marker = "[DRY_RUN]" in output
            ok = result.returncode == 0 and (not args.dry_run or dry_run_marker)
            label = f"{index}/{total} {case.name}"
            print(f"\n===== {label}: benchmark output =====")
            print(output.rstrip() if output.strip() else "<no output>")
            if args.dry_run and result.returncode == 0 and not dry_run_marker:
                print("ERROR: exit code was 0, but no [DRY_RUN] marker was found")
            print(
                f"===== {label}: {'PASS' if ok else 'FAIL'} "
                f"({elapsed:.1f}s), returncode={result.returncode} ====="
            )
            passed += ok
            failed += not ok
        except subprocess.TimeoutExpired as error:
            failed += 1
            label = f"{index}/{total} {case.name}"
            print(f"\n===== {label}: benchmark output before timeout =====")
            partial_output = (error.stdout or "") + (error.stderr or "")
            print(partial_output.rstrip() if partial_output else "<no output>")
            print(f"===== {label}: TIMEOUT after {args.timeout:.0f}s =====")

    print(f"\nSummary: passed={passed}, failed={failed}, total={len(CASES)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
