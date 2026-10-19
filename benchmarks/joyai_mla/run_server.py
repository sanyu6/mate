"""Launch SGLang with default, optimized, or numerical-validation decode."""

import argparse
import os
import runpy
import sys


def configure():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--mla-mode",
        choices=["default", "optimized", "validate"],
        default=os.environ.get("JOYAI_MLA_MODE", "default"),
    )
    parser.add_argument("--mla-library", default=os.environ.get("MATE_MLA_LIBRARY"))
    parser.add_argument(
        "--validation-output", default=os.environ.get("JOYAI_VALIDATION_OUTPUT")
    )
    parser.add_argument(
        "--validation-min-kv",
        type=int,
        default=int(os.environ.get("JOYAI_VALIDATION_MIN_KV", "20000")),
    )
    args, remaining = parser.parse_known_args()
    if args.mla_mode != "default" and not args.mla_library:
        parser.error("--mla-library is required for optimized/validate mode")
    if args.mla_mode == "validate" and not args.validation_output:
        parser.error("--validation-output is required in validate mode")
    os.environ["JOYAI_MLA_MODE"] = args.mla_mode
    for key, value in [
        ("MATE_MLA_LIBRARY", args.mla_library),
        ("JOYAI_VALIDATION_OUTPUT", args.validation_output),
        ("JOYAI_VALIDATION_MIN_KV", str(args.validation_min_kv)),
    ]:
        if value is not None:
            os.environ[key] = value
    # Spawned ranks inherit both mode and paths even after own flags are removed.
    sys.argv = [sys.argv[0], *remaining]
    from profiling import enable_musa_activity

    enable_musa_activity()
    if args.mla_mode == "validate":
        from validate_real_attention import install

        install(args.mla_library, args.validation_output, args.validation_min_kv)
    else:
        from adapter import install

        install(args.mla_mode, args.mla_library)


if __name__ in ("__main__", "__mp_main__"):
    configure()
    if __name__ == "__main__":
        runpy.run_module("sglang.launch_server", run_name="__main__")
