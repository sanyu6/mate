import argparse
import sys
from collections.abc import Sequence

from routines.attention import parse_attention_args, run_attention_test
from routines.attention_bwd import parse_attention_bwd_args, run_attention_bwd_test
from routines.gdn import parse_gdn_args, run_gdn_test
from routines.gemm import parse_gemm_args, run_gemm_test
from routines.kda import parse_kda_args, run_kda_test
from routines.mla import parse_mla_args, run_mla_test
from routines.mate_benchmark_utils import (
    benchmark_apis,
    configure_benchmark_logging,
    set_random_seed,
)


def format_available_routines() -> str:
    """Format registered benchmark routines for the CLI help epilog."""
    lines = ["Available routines:"]
    for family, routines in benchmark_apis.items():
        lines.append(f"  {family.upper()}:")
        lines.extend(f"    {routine}" for routine in routines)
    return "\n".join(lines)


def run_bench(args: argparse.Namespace) -> None:
    """Route a MATE benchmark case to its routine implementation."""
    configure_benchmark_logging(args.verbose)
    set_random_seed(args.random_seed)
    if args.routine in benchmark_apis["attention"]:
        run_attention_test(args)
        return
    if args.routine in benchmark_apis["attention_bwd"]:
        run_attention_bwd_test(args)
        return
    if args.routine in benchmark_apis["gemm"]:
        run_gemm_test(args)
        return
    if args.routine in benchmark_apis["gdn"]:
        run_gdn_test(args)
        return
    if args.routine in benchmark_apis["mla"]:
        run_mla_test(args)
        return
    if args.routine in benchmark_apis["kda"]:
        run_kda_test(args)
        return

    raise ValueError(f"Unsupported routine: {args.routine}")


def parse_args(line: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse shared arguments followed by routine-specific arguments."""
    if line is None:
        line = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Run a MATE benchmark routine.",
        epilog=format_available_routines(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    routine_choices = [
        routine for routines in benchmark_apis.values() for routine in routines
    ]
    parser.add_argument(
        "-R",
        "--routine",
        required=True,
        choices=routine_choices,
        metavar="ROUTINE",
        help="Benchmark routine to run.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output verbosity; use -vv for tensor details.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible benchmark inputs.",
    )
    parser.add_argument(
        "--mupti-kernel-name",
        default=None,
        metavar="STR",
        help=(
            "Also profile the exact kernel name with Kineto/MUPTI and print a "
            "separate single-kernel performance line."
        ),
    )
    routine_parser = argparse.ArgumentParser(add_help=False)
    routine_parser.add_argument("-R", "--routine", choices=routine_choices)
    routine_parser.add_argument("-v", "--verbose", action="count", default=0)
    routine_args, _ = routine_parser.parse_known_args(line)
    configure_benchmark_logging(routine_args.verbose)

    if routine_args.routine in benchmark_apis["attention"]:
        return parse_attention_args(line, parser)
    if routine_args.routine in benchmark_apis["attention_bwd"]:
        return parse_attention_bwd_args(line, parser)
    if routine_args.routine in benchmark_apis["gemm"]:
        return parse_gemm_args(line, parser)
    if routine_args.routine in benchmark_apis["gdn"]:
        return parse_gdn_args(line, parser)
    if routine_args.routine in benchmark_apis["mla"]:
        return parse_mla_args(line, parser)
    if routine_args.routine in benchmark_apis["kda"]:
        return parse_kda_args(line, parser)

    return parser.parse_args(line)


def main() -> int:
    args = parse_args()
    run_bench(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
