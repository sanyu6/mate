#!/usr/bin/env python3
"""Apply the pinned MP32/FP4 header delta to the public MUTLASS submodule."""

from pathlib import Path
import argparse
import subprocess


BASE = "435707c2f62e0ebf9067f5057412ebab14643781"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Validate without applying"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dependency = root / "3rdparty/mutlass"
    patch = root / "patches/mutlass-mp32-fp4.patch"
    if not (dependency / ".git").exists():
        parser.error("Run git submodule update --init 3rdparty/mutlass first")

    def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(dependency), *arguments],
            check=check,
            capture_output=True,
            text=True,
        )

    if git("rev-parse", "HEAD").stdout.strip() != BASE:
        parser.error(f"Expected MUTLASS {BASE}; refusing to patch a different revision")
    if git("apply", "--reverse", "--check", str(patch), check=False).returncode == 0:
        print("MP32/FP4 header patch is already applied")
        return
    if git("status", "--porcelain").stdout:
        parser.error("MUTLASS has local changes; refusing to overwrite them")
    result = git("apply", "--check", str(patch), check=False)
    if result.returncode:
        parser.error(result.stderr.strip())
    if args.check:
        print("MP32/FP4 header patch applies cleanly")
        return
    git("apply", str(patch))
    print("Applied MP32/FP4 headers, including the W4A4 pool TME cache hints")


if __name__ == "__main__":
    main()
