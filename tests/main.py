#!/usr/bin/env python3
"""MATE test catalog and incremental-test resolver entry point."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import stat
import subprocess
import sys
from typing import Literal, Mapping, Sequence


# Put repo into sys.path
REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from mate.testing.resolver import (  # noqa: E402
    OperationCatalog,
    SelectionPolicy,
    TestRunRequest,
    TestRunner,
)
from mate.testing.resolver.cli import main as resolver_main  # noqa: E402


TEST_CATALOG = OperationCatalog(
    definition_files=(Path(__file__).resolve().relative_to(REPOSITORY),),
)

TEST_SELECTION_POLICY = SelectionPolicy(
    ignored_paths=(
        ".devcontainer/**",
        "benchmarks/**",
        "docker/**",
        "docs/**",
        "internal/**",
        "licenses/**",
        "skills/**",
        "scripts/**",
        "tools/**",
        "wrappers/**",
        ".clang-format",
        ".dockerignore",
        ".pre-commit-config.yaml",
        "AGENTS.md",
        "CLAUDE.md",
        "compose.local.yaml.example",
        "compose.yaml",
        "LICENSE",
        "pyproject.toml",
        "README.md",
        "version.txt",
    ),
)

TEST_CATALOG.register(
    "mate.ops.moe.dispatch_shared",
    interfaces=(
        "mate/ops/moe/dispatch_shared/op.py",
        "mate/ops/moe/dispatch_shared/impl.py",
        "mate/jit/dispatch_shared.py",
    ),
    tests=(
        "tests/test_dispatch_shared_contract.py",
        "tests/test_dispatch_shared_gpu.py",
    ),
)

TEST_CATALOG.register(
    "mate.ops.moe.fp4_dispatch",
    interfaces=(
        "mate/ops/moe/fp4_dispatch/op.py",
        "mate/ops/moe/fp4_dispatch/buffer.py",
        "mate/jit/fp4_dispatch_only.py",
    ),
    tests=("tests/test_fp4_dispatch_contract.py", "tests/test_fp4_dispatch_gpu.py"),
)

TEST_CATALOG.register(
    "mate.ops.attention.dsa",
    interfaces=(
        "mate/flashmla.py",
        "mate/sparse_mla_interface.py",
    ),
    tests=("tests/test_sparse_mla.py",),
)

TEST_CATALOG.register(
    "mate.ops.attention.fa",
    interfaces=("mate/mha_interface.py",),
    tests=(
        "tests/test_fmha.py",
        "tests/test_fmha_bwd.py",
    ),
)

TEST_CATALOG.register(
    "mate.ops.attention.mla",
    interfaces=("mate/mha_interface.py", "mate/flashmla.py"),
    tests=("tests/test_mla.py",),
)

# NOTE: We probably can remove test_msa_interface.py
TEST_CATALOG.register(
    "mate.ops.attention.msa",
    interfaces=("mate/msa_interface.py",),
    tests=(
        "tests/test_msa_fwd.py",
        "tests/test_msa_interface.py",
    ),
)

TEST_CATALOG.register(
    "mate.ops.attention.sage",
    interfaces=("mate/sage_attention_interface.py",),
    tests=("tests/test_sage_attention_interface.py",),
)

TEST_CATALOG.register(
    "mate.ops.attention.kda",
    interfaces=("mate/kda.py",),
    tests=(
        "tests/test_kda.py",
        "tests/test_kda_decode.py",
        "tests/test_kda_chunk_states.py",
    ),
)

TEST_CATALOG.register(
    "mate.ops.attention.gdn",
    interfaces=(
        "mate/gdn_decode.py",
        "mate/gdn_prefill.py",
    ),
    tests=(
        "tests/test_gdn_decode.py",
        "tests/test_gdn_mtp.py",
        "tests/test_gdn_prefill.py",
    ),
)

TEST_CATALOG.register(
    "mate.ops.gemm.gemm",
    interfaces=("mate/gemm.py",),
    tests=(
        "tests/test_gemm.py",
        "tests/test_deep_gemm.py",
    ),
)

TEST_CATALOG.register(
    "mate.ops.moe.deep_gemm",
    interfaces=("mate/deep_gemm.py",),
    tests=("tests/test_deep_gemm.py",),
)

TEST_CATALOG.register(
    "mate.ops.moe.router.hash_topk",
    interfaces=("mate/hash_topk.py",),
    tests=("tests/test_hash_topk.py",),
)

TEST_CATALOG.register(
    "mate.ops.moe.gating.moe_fused_gate",
    interfaces=("mate/moe_fused_gate.py",),
    tests=("tests/test_fused_moe_gate.py",),
)

TEST_CATALOG.register(
    "mate.ops.moe.mega_moe",
    interfaces=(
        "mate/mega_moe.py",
        "mate/jit/mega_moe.py",
        "mate/jit/mega_moe_prefill.py",
        "mate/mega_moe_fp8_fp4.py",
        "mate/jit/mega_moe_fp8_fp4.py",
        "mate/mega_moe_fp4_fp4.py",
        "mate/jit/mega_moe_fp4_fp4.py",
        "mate/jit/mega_moe_fp4_fp4_decode.py",
        "mate/_fp4_moe_buffer.py",
    ),
    tests=(
        "tests/test_mega_moe_decode.py",
        "tests/test_mega_moe_prefill.py",
        "tests/test_mega_moe_fp8_fp4.py",
        "tests/test_mega_moe_fp4_fp4_contract.py",
        "tests/test_mega_moe_fp4_fp4_situ_contract.py",
        "tests/test_mega_moe_fp4_fp4_grouped.py",
        "tests/test_mega_moe_fp4_fp4_ipc.py",
        "tests/test_mega_moe_fp4_fp4_pipeline.py",
        "tests/test_mega_moe_fp4_fp4_publication.py",
        "tests/test_mega_moe_fp4_fp4_decode_contract.py",
        "tests/test_mega_moe_fp4_fp4_decode.py",
        "tests/test_mega_moe_fp4_fp4_decode_swap_ab.py",
        "tests/test_mega_moe_fp4_fp4_decode_w4a4.py",
    ),
)

TEST_CATALOG.register(
    "mate.ops.norm",
    interfaces=("mate/norm/__init__.py",),
    tests=("tests/test_norm.py",),
)

TEST_CATALOG.register(
    "mate.ops.hyper_connect",
    interfaces=("mate/hyperconnection.py",),
    tests=("tests/test_hyperconnection.py",),
)

TEST_CATALOG.register(
    "mate.ops.quantization",
    interfaces=("mate/quantization/__init__.py", "mate/quantization/api.py"),
    tests=("tests/test_quantization.py",),
)


def _mudmp_files(directory: Path) -> dict[Path, tuple[int, int, int]]:
    files = {}
    for path in directory.glob("*.mudmp"):
        try:
            info = path.lstat()
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode):
            files[path] = (info.st_ino, info.st_mtime_ns, info.st_size)
    return files


def _run_with_mudmp_output(
    request: TestRunRequest,
    command: Sequence[str],
    *,
    environment: Mapping[str, str | None],
) -> int:
    before = _mudmp_files(request.repository)
    status = request.run(command, environment=environment)
    if status:
        after = _mudmp_files(request.repository)
        for path in sorted(after):
            if after[path] == before.get(path):
                continue
            print(f"\n=== mudmp: {path.name} ===", flush=True)
            subprocess.run(("cat", "--", str(path)), check=False)
            print(f"\n=== end mudmp: {path.name} ===", flush=True)
    return status


def run_selected_tests(
    request: TestRunRequest,
    *,
    dry_run_workers: int | Literal["auto"] = "auto",
) -> int:
    """Preflight marked selected tests, then run all selections for correctness."""
    tests = request.selection.selected_tests
    base_environment = {
        "MATE_PYTEST_SHARD_TOTAL": "1",
        "MATE_PYTEST_SHARD_INDEX": "0",
    }
    status = _run_with_mudmp_output(
        request,
        (
            sys.executable,
            "-m",
            "pytest",
            "--dry-run",
            "-n",
            str(dry_run_workers),
            "-v",
            *tests,
            "--max-worker-restart",
            "0",
        ),
        environment=base_environment,
    )
    if status:
        return status

    return _run_with_mudmp_output(
        request,
        (
            sys.executable,
            "-m",
            "pytest",
            "-v",
            *tests,
        ),
        environment={**base_environment, "MATE_DRY_RUN": None},
    )


def _parse_dry_run_workers(value: str) -> int | Literal["auto"]:
    if value == "auto":
        return "auto"
    try:
        workers = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "must be 'auto' or a non-negative integer"
        ) from None
    if workers < 0:
        raise argparse.ArgumentTypeError("must be 'auto' or a non-negative integer")
    return workers


def _configure_run_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run-workers",
        type=_parse_dry_run_workers,
        default="auto",
        metavar="N|auto",
        help="pytest-xdist workers for dry-run: auto or a non-negative integer "
        "(0 disables xdist; default: auto)",
    )


def _create_test_runner(args: argparse.Namespace) -> TestRunner:
    return partial(run_selected_tests, dry_run_workers=args.dry_run_workers)


def main(argv: Sequence[str] | None = None) -> int:
    return resolver_main(
        TEST_CATALOG,
        argv,
        policy=TEST_SELECTION_POLICY,
        configure_run_parser=_configure_run_parser,
        runner_factory=_create_test_runner,
    )


if __name__ == "__main__":
    raise SystemExit(main())
