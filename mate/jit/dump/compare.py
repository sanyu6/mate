#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


_STRUCT_FORMATS = {
    "bool": "?",
    "uint8": "B",
    "int8": "b",
    "int16": "h",
    "int32": "i",
    "int64": "q",
    "float16": "e",
    "float32": "f",
    "float64": "d",
}


def _numel(shape: list[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _read_values(path: Path, dtype: str, count: int) -> Iterator[float | int | bool]:
    data = path.read_bytes()
    if dtype == "bfloat16":
        item_size = 2
        if len(data) != count * item_size:
            raise ValueError(
                f"{path}: expected {count * item_size} bytes, got {len(data)}"
            )
        for (bits,) in struct.iter_unpack("<H", data):
            yield struct.unpack("<f", struct.pack("<I", bits << 16))[0]
        return

    format_code = _STRUCT_FORMATS.get(dtype)
    if format_code is None:
        raise ValueError(f"{path}: unsupported dtype {dtype!r}")
    item_size = struct.calcsize(format_code)
    if len(data) != count * item_size:
        raise ValueError(f"{path}: expected {count * item_size} bytes, got {len(data)}")
    for (value,) in struct.iter_unpack(f"<{format_code}", data):
        yield value


def _unravel_index(index: int, shape: list[int]) -> tuple[int, ...]:
    coordinates = []
    for dimension in reversed(shape):
        coordinates.append(index % dimension)
        index //= dimension
    return tuple(reversed(coordinates))


def _compare_close(root: Path, spec: dict[str, Any]) -> bool:
    shape = spec["shape"]
    count = _numel(shape)
    actual_values = _read_values(root / spec["output"], spec["output_dtype"], count)
    reference_values = _read_values(
        root / spec["reference"], spec["reference_dtype"], count
    )
    atol = float(spec["atol"])
    rtol = float(spec["rtol"])
    equal_nan = bool(spec.get("equal_nan", False))

    mismatches = 0
    max_abs = 0.0
    max_rel = 0.0
    first_mismatch = None
    for linear_index, (actual, reference) in enumerate(
        zip(actual_values, reference_values)
    ):
        actual_float = float(actual)
        reference_float = float(reference)
        if actual_float == reference_float:
            continue
        if equal_nan and math.isnan(actual_float) and math.isnan(reference_float):
            continue

        absolute_error = abs(actual_float - reference_float)
        relative_error = (
            absolute_error / abs(reference_float)
            if reference_float != 0.0
            else math.inf
        )
        max_abs = max(max_abs, absolute_error)
        max_rel = max(max_rel, relative_error)
        if absolute_error <= atol + rtol * abs(reference_float):
            continue

        mismatches += 1
        if first_mismatch is None:
            first_mismatch = (
                _unravel_index(linear_index, shape),
                actual_float,
                reference_float,
                absolute_error,
            )

    name = spec["name"]
    if mismatches == 0:
        logger.info(
            f"PASS {name}: {count} values, max_abs={max_abs:.8g}, max_rel={max_rel:.8g}"
        )
        return True

    coordinate, actual, reference, absolute_error = first_mismatch
    logger.error(
        f"FAIL {name}: {mismatches}/{count} values differ, "
        f"max_abs={max_abs:.8g}, max_rel={max_rel:.8g}"
    )
    logger.error(
        f"  first mismatch at {coordinate}: output={actual:.8g}, "
        f"reference={reference:.8g}, abs={absolute_error:.8g}, "
        f"allowed={atol + rtol * abs(reference):.8g}"
    )
    return False


def _compare_error_ratio(root: Path, spec: dict[str, Any]) -> bool:
    shape = spec["shape"]
    count = _numel(shape)
    actual_values = list(
        _read_values(root / spec["output"], spec["output_dtype"], count)
    )
    reference_values = list(
        _read_values(root / spec["reference"], spec["reference_dtype"], count)
    )
    baseline_values = list(
        _read_values(root / spec["baseline"], spec["baseline_dtype"], count)
    )

    actual_errors = [
        abs(float(actual) - float(reference))
        for actual, reference in zip(actual_values, reference_values)
    ]
    baseline_errors = [
        abs(float(baseline) - float(reference))
        for baseline, reference in zip(baseline_values, reference_values)
    ]
    name = spec["name"]
    if not all(math.isfinite(error) for error in actual_errors + baseline_errors):
        logger.error(f"FAIL {name}: comparison contains a non-finite error")
        return False

    actual_max = max(actual_errors, default=0.0)
    baseline_max = max(baseline_errors, default=0.0)
    mean_denominator = int(spec.get("mean_denominator", count))
    if mean_denominator < count:
        raise ValueError(
            f"mean denominator {mean_denominator} is smaller than value count {count}"
        )
    actual_mean = sum(actual_errors) / mean_denominator if mean_denominator else 0.0
    baseline_mean = sum(baseline_errors) / mean_denominator if mean_denominator else 0.0
    epsilon = float(spec.get("epsilon", 1e-5))
    allowed_max = float(spec["max_error_factor"]) * baseline_max + epsilon
    allowed_mean = float(spec["mean_error_factor"]) * baseline_mean + epsilon
    passed = actual_max <= allowed_max and actual_mean <= allowed_mean
    status = "PASS" if passed else "FAIL"
    logger.log(
        logging.INFO if passed else logging.ERROR,
        f"{status} {name}: {count} values, max_abs={actual_max:.8g}, "
        f"baseline_max_abs={baseline_max:.8g}, allowed_max={allowed_max:.8g}, "
        f"mean_abs={actual_mean:.8g}, baseline_mean_abs={baseline_mean:.8g}, "
        f"allowed_mean={allowed_mean:.8g}",
    )
    return passed


def _compare(root: Path, spec: dict[str, Any]) -> bool:
    comparison = spec.get("comparison", "close")
    if comparison == "close":
        return _compare_close(root, spec)
    if comparison == "error_ratio":
        return _compare_error_ratio(root, spec)
    raise ValueError(f"unsupported comparison mode: {comparison!r}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Compare replay outputs with dumped pytest references."
    )
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1:
        raise ValueError(f"unsupported reference manifest: {manifest.get('version')}")
    comparisons = manifest.get("comparisons", [])
    if not comparisons:
        raise ValueError(f"reference manifest has no comparisons: {manifest_path}")

    results = [_compare(manifest_path.parent, spec) for spec in comparisons]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
