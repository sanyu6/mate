"""Build the MLA module from this optimized MATE checkout."""

import argparse
import hashlib
import json


def main():
    from mate.jit import env
    from mate.jit.mla_ops import get_mla_ops_module

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", help="Optional JSON manifest output")
    args = parser.parse_args()
    source = env.MATE_CSRC_DIR / "mla_pybind.mu"
    if "SmallM" not in source.read_text():
        parser.error(
            "The imported MATE source lacks the small-M optimization; set PYTHONPATH to this checkout"
        )
    get_mla_ops_module()
    library = env.MATE_JIT_DIR / "mla_ops" / "mla_ops.so"
    result = {
        "library": str(library.resolve()),
        "source": str(source),
        "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }
    if args.record:
        from pathlib import Path

        Path(args.record).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
