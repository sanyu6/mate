from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, TypedDict

import torch

from ..env import MusaArch
from .context import JITDumpContext, get_jit_dump_context

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOADER_SOURCE = _REPO_ROOT / "tools" / "jit_dump" / "loader.cpp"
_COMPARE_SOURCE = Path(__file__).with_name("compare.py")
_BENCHMARK_SOURCE = Path(__file__).with_name("benchmark.py")


class MateJITDumpComplete(RuntimeError):
    """Raised after one JIT dump is complete and before kernel launch."""

    def __init__(self, artifact_dir: Path, function_name: str) -> None:
        self.artifact_dir = artifact_dir
        self.function_name = function_name
        super().__init__(f"dumped {function_name} in {artifact_dir}")


@dataclasses.dataclass(frozen=True)
class _DumpedOutput:
    ordinal: int
    name: str
    filename: str
    tensor: torch.Tensor


@dataclasses.dataclass(frozen=True)
class _RegisteredReference:
    name: str
    tensor: torch.Tensor
    atol: float | None
    rtol: float | None
    equal_nan: bool
    baseline: torch.Tensor | None = None
    max_error_factor: float | None = None
    mean_error_factor: float | None = None
    mean_denominator: int | None = None
    epsilon: float = 1e-5


@dataclasses.dataclass
class _PytestDumpSession:
    root: Path
    context: JITDumpContext | None = None
    dumps: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    current_nodeid: str | None = None
    current_dumped: bool = False


class _Comparison(TypedDict, total=False):
    name: str
    output: str
    reference: str
    shape: list[int]
    output_dtype: str
    reference_dtype: str
    atol: float
    rtol: float
    equal_nan: bool
    output_ordinal: int
    comparison: str
    baseline: str
    baseline_dtype: str
    max_error_factor: float
    mean_error_factor: float
    mean_denominator: int
    epsilon: float


_registered_references: dict[str, _RegisteredReference] = {}
_pytest_session: _PytestDumpSession | None = None


def _write_pytest_manifest(session: _PytestDumpSession) -> None:
    manifest = {
        "version": 2,
        "dump_arch": session.context.dump_arch.name if session.context else None,
        "dumps": session.dumps,
    }
    temporary_path = session.root / ".manifest.json.tmp"
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(session.root / "manifest.json")


def start_pytest_dump_session(
    root: Path,
    context: JITDumpContext | None = None,
) -> None:
    global _pytest_session

    if _pytest_session is not None:
        raise RuntimeError("a JIT dump pytest session is already active")
    resolved_root = root.expanduser().resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    _pytest_session = _PytestDumpSession(root=resolved_root, context=context)
    _write_pytest_manifest(_pytest_session)


def finish_pytest_dump_session() -> int:
    global _pytest_session, _registered_references

    if _pytest_session is None:
        return 0
    dump_count = len(_pytest_session.dumps)
    _pytest_session = None
    _registered_references = {}
    return dump_count


def begin_pytest_dump_case(nodeid: str) -> None:
    global _registered_references

    if _pytest_session is None:
        raise RuntimeError("no JIT dump pytest session is active")
    if _pytest_session.current_nodeid is not None:
        raise RuntimeError(
            f"JIT dump pytest case {_pytest_session.current_nodeid!r} is still active"
        )
    _pytest_session.current_nodeid = nodeid
    _pytest_session.current_dumped = False
    _registered_references = {}


def end_pytest_dump_case(nodeid: str) -> bool:
    global _registered_references

    if _pytest_session is None:
        raise RuntimeError("no JIT dump pytest session is active")
    if _pytest_session.current_nodeid != nodeid:
        raise RuntimeError(
            "JIT dump pytest case mismatch: "
            f"{_pytest_session.current_nodeid!r} != {nodeid!r}"
        )
    dumped = _pytest_session.current_dumped
    _pytest_session.current_nodeid = None
    _pytest_session.current_dumped = False
    _registered_references = {}
    return dumped


def pytest_dump_count() -> int:
    return len(_pytest_session.dumps) if _pytest_session is not None else 0


def _require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"{description} does not exist: {resolved}")
    return resolved


def _resolve_loader_path() -> Path:
    override = os.environ.get("MATE_JIT_DUMP_LOADER_SO")
    if override:
        return _require_file(Path(override), "MATE_JIT_DUMP_LOADER_SO")

    if not _LOADER_SOURCE.is_file():
        raise RuntimeError(
            "automatic JIT dump loader build requires a MATE source checkout; "
            "set MATE_JIT_DUMP_LOADER_SO when running from an installed package"
        )

    from ..core import gen_jit_spec
    from ..cpp_ext import get_musa_home

    spec = gen_jit_spec(
        "jit_dump_loader",
        [_LOADER_SOURCE],
        extra_include_paths=[get_musa_home() / "include"],
    )
    return spec.build(verbose=os.environ.get("MATE_JIT_VERBOSE", "0") == "1")


def _musart_soname(path: Path) -> str:
    candidate = path.name
    if candidate == "libmusart.so" and path.is_symlink():
        candidate = Path(os.readlink(path)).name

    match = re.match(r"^(libmusart\.so\.\d+)(?:\..*)?$", candidate)
    if match:
        return match.group(1)

    resolved_name = path.resolve().name
    match = re.match(r"^(libmusart\.so\.\d+)(?:\..*)?$", resolved_name)
    if match:
        return match.group(1)
    raise RuntimeError(f"cannot determine libmusart SONAME from {path}")


def _resolve_musart() -> tuple[Path, str]:
    override = os.environ.get("MATE_JIT_DUMP_MUSART_SO")
    if override:
        path = Path(override).expanduser()
    else:
        from ..cpp_ext import get_musa_home

        path = get_musa_home() / "lib" / "libmusart.so"

    soname = _musart_soname(path)
    return _require_file(path, "libmusart"), soname


def _tensor_device_id(tensor: torch.Tensor) -> int:
    if tensor.device.type != "musa":
        raise ValueError(f"JIT dump only supports MUSA tensors, got {tensor.device}")
    return tensor.device.index or 0


def _collect_tensor_device_ids(value: Any, device_ids: set[int]) -> None:
    if isinstance(value, torch.Tensor):
        device_ids.add(_tensor_device_id(value))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _collect_tensor_device_ids(item, device_ids)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    host_tensor = tensor.detach().contiguous().cpu().clone()
    return host_tensor.view(torch.uint8).reshape(-1).numpy().tobytes()


def _tensor_expression(tensor: torch.Tensor, index: int, output_dir: Path) -> str:
    device_id = _tensor_device_id(tensor)
    if not tensor.is_contiguous():
        raise ValueError(
            "JIT dump only supports contiguous tensors, "
            f"got shape={tuple(tensor.shape)} stride={tensor.stride()}"
        )

    filename = f"tensor_{index}.bin"
    (output_dir / filename).write_bytes(_tensor_bytes(tensor))
    dtype_name = str(tensor.dtype).removeprefix("torch.")
    return (
        f"load_tensor({filename!r}, {list(tensor.shape)!r}, "
        f"{dtype_name!r}, {device_id})"
    )


def _empty_tensor_expression(tensor: torch.Tensor) -> str:
    device_id = _tensor_device_id(tensor)
    if not tensor.is_contiguous():
        raise ValueError(
            "JIT dump only supports contiguous tensors, "
            f"got shape={tuple(tensor.shape)} stride={tensor.stride()}"
        )
    dtype_name = str(tensor.dtype).removeprefix("torch.")
    return f"empty_tensor({list(tensor.shape)!r}, {dtype_name!r}, {device_id})"


def _value_expression(
    value: Any,
    output_dir: Path,
    tensor_index: list[int],
    tensor_bindings: list[str],
) -> str:
    if isinstance(value, torch.Tensor):
        index = tensor_index[0]
        tensor_index[0] += 1
        variable_name = f"tensor_{index}"
        expression = _tensor_expression(value, index, output_dir)
        tensor_bindings.append(f"{variable_name} = {expression}")
        return variable_name
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, tuple):
        items = [
            _value_expression(item, output_dir, tensor_index, tensor_bindings)
            for item in value
        ]
        suffix = "," if len(items) == 1 else ""
        return f"({', '.join(items)}{suffix})"
    if isinstance(value, list):
        items = [
            _value_expression(item, output_dir, tensor_index, tensor_bindings)
            for item in value
        ]
        return f"[{', '.join(items)}]"
    raise TypeError(f"unsupported JIT dump argument: {type(value).__name__}")


def _write_replay(
    output_dir: Path,
    function_name: str,
    arguments: tuple[Any, ...],
    empty_tensor_indices: frozenset[int],
    output_names: Mapping[int, str],
) -> None:
    device_ids: set[int] = set()
    for argument in arguments:
        _collect_tensor_device_ids(argument, device_ids)
    if len(device_ids) > 1:
        raise ValueError(
            "JIT dump requires all tensors on one MUSA device, "
            f"got device ids {sorted(device_ids)}"
        )
    replay_device_id = next(iter(device_ids), 0)

    tensor_index = [0]
    tensor_bindings: list[str] = []
    output_bindings: list[tuple[str, str]] = []
    expressions = []
    for index, argument in enumerate(arguments):
        if index in empty_tensor_indices:
            if not isinstance(argument, torch.Tensor):
                raise TypeError(f"empty argument {index} is not a tensor")
            variable_name = f"tensor_{tensor_index[0]}"
            tensor_index[0] += 1
            expression = _empty_tensor_expression(argument)
            tensor_bindings.append(f"{variable_name} = {expression}")
            expressions.append(variable_name)
            output_ordinal = len(output_bindings)
            output_name = output_names.get(index, f"argument_{index}")
            output_bindings.append(
                (variable_name, f"output_{output_ordinal}_{output_name}.bin")
            )
        else:
            expressions.append(
                _value_expression(argument, output_dir, tensor_index, tensor_bindings)
            )
    call_arguments = ",\n        ".join(expressions)
    tensor_setup = "\n".join(tensor_bindings)
    output_saves = (
        "\n".join(
            (
                'loader.get_function("jit_dump_loader_save_tensor")('
                f"{variable_name}, str(ROOT / {filename!r}))"
            )
            for variable_name, filename in output_bindings
        )
        or "pass"
    )
    replay = f"""import os
import sys
from pathlib import Path

os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")
sys.modules["torch"] = None
import tvm_ffi


ROOT = Path(__file__).resolve().parent
FUNCTION_NAME = {function_name!r}
DEVICE_ID = {replay_device_id}
loader = tvm_ffi.load_module(ROOT / "loader.so")
loader.get_function("jit_dump_loader_install_allocator")()


def load_tensor(filename, shape, dtype, device_id):
    print(f"loading {{filename}}", flush=True)
    tensor = loader.get_function("jit_dump_loader_load_tensor")(
        str(ROOT / filename), shape, dtype, device_id
    )
    print(f"loaded {{filename}}", flush=True)
    return tensor


def empty_tensor(shape, dtype, device_id):
    print(f"allocating output {{shape}} {{dtype}}", flush=True)
    return loader.get_function("jit_dump_loader_empty_tensor")(
        shape, dtype, device_id
    )


{tensor_setup}

kernel = tvm_ffi.load_module(ROOT / "kernel.so")
kernel_function = kernel.get_function(FUNCTION_NAME)


def invoke():
    kernel_function(
        {call_arguments}
    )


def synchronize():
    loader.get_function("jit_dump_loader_synchronize")(DEVICE_ID)


def save_outputs():
    {output_saves.replace(chr(10), chr(10) + "    ")}


def main():
    print(f"launching {{FUNCTION_NAME}}", flush=True)
    invoke()
    print("kernel call returned; synchronizing", flush=True)
    synchronize()
    save_outputs()
    print(f"replay completed: {{FUNCTION_NAME}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
    (output_dir / "replay.py").write_text(replay)


def _write_run_script(output_dir: Path) -> None:
    script = """#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export LD_LIBRARY_PATH="$ROOT/runtime:/usr/local/musa/lib:${LD_LIBRARY_PATH:-}"
MODE=${1:-all}
if [ "$#" -gt 0 ]; then
    shift
fi
REPLAY_PID=

stop_replay() {
    if [ -n "$REPLAY_PID" ]; then
        kill -TERM "$REPLAY_PID" 2>/dev/null || true
    fi
    exit 130
}

run_replay() {
    trap stop_replay HUP INT TERM
    python3 -u "$ROOT/replay.py" &
    REPLAY_PID=$!
    set +e
    wait "$REPLAY_PID"
    STATUS=$?
    set -e
    REPLAY_PID=
    trap - HUP INT TERM
    return "$STATUS"
}

case "$MODE" in
    replay)
        run_replay
        ;;
    compare)
        if [ ! -f "$ROOT/references.json" ]; then
            echo "references.json is required for compare mode" >&2
            exit 2
        fi
        exec python3 -u "$ROOT/compare.py" "$ROOT/references.json"
        ;;
    bench)
        exec python3 -u "$ROOT/benchmark.py" "$@"
        ;;
    all)
        run_replay
        if [ -f "$ROOT/references.json" ]; then
            exec python3 -u "$ROOT/compare.py" "$ROOT/references.json"
        fi
        ;;
    *)
        echo "usage: $0 [all|replay|compare|bench] [bench options]" >&2
        exit 2
        ;;
esac
"""
    path = output_dir / "run.sh"
    path.write_text(script)
    path.chmod(0o755)


def _commit_dump(staging_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / "DUMP_COMPLETE"
    complete_path.unlink(missing_ok=True)

    for pattern in (
        "tensor_*.bin",
        "output_*.bin",
        "reference_*.bin",
        "baseline_*.bin",
    ):
        for stale_tensor in output_dir.glob(pattern):
            stale_tensor.unlink()
    for name in ("references.json", "REFERENCE_COMPLETE"):
        (output_dir / name).unlink(missing_ok=True)

    runtime_dir = output_dir / "runtime"
    if runtime_dir.is_dir() and not runtime_dir.is_symlink():
        shutil.rmtree(runtime_dir)
    elif runtime_dir.exists() or runtime_dir.is_symlink():
        runtime_dir.unlink()

    for name in (
        "kernel.so",
        "loader.so",
        "replay.py",
        "run.sh",
        "compare.py",
        "benchmark.py",
    ):
        (staging_dir / name).replace(output_dir / name)
    (staging_dir / "runtime").replace(runtime_dir)
    for tensor_path in staging_dir.glob("tensor_*.bin"):
        tensor_path.replace(output_dir / tensor_path.name)

    (staging_dir / "DUMP_COMPLETE").replace(complete_path)


def _dumped_outputs(
    arguments: tuple[Any, ...],
    empty_tensor_indices: frozenset[int],
    output_names: Mapping[int, str],
) -> tuple[_DumpedOutput, ...]:
    outputs: list[_DumpedOutput] = []
    for argument_index, argument in enumerate(arguments):
        if argument_index not in empty_tensor_indices:
            continue
        if not isinstance(argument, torch.Tensor):
            raise TypeError(f"empty argument {argument_index} is not a tensor")
        name = output_names.get(argument_index, f"argument_{argument_index}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"invalid dump output name: {name!r}")
        ordinal = len(outputs)
        outputs.append(
            _DumpedOutput(
                ordinal=ordinal,
                name=name,
                filename=f"output_{ordinal}_{name}.bin",
                tensor=argument,
            )
        )
    return tuple(outputs)


def _reference_dump_enabled() -> bool:
    return os.environ.get("MATE_JIT_DUMP_REFERENCE", "0") == "1" and bool(
        os.environ.get("MATE_JIT_DUMP_DIR")
    )


def register_dump_reference(
    name: str,
    reference: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    equal_nan: bool = False,
) -> None:
    """Register a reference before the target FFI call is dumped."""
    if not _reference_dump_enabled():
        return
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"invalid dump reference name: {name!r}")
    if not isinstance(reference, torch.Tensor):
        raise TypeError(f"dump reference {name!r} is not a tensor")
    _registered_references[name] = _RegisteredReference(
        name=name,
        tensor=reference,
        atol=float(atol),
        rtol=float(rtol),
        equal_nan=equal_nan,
    )


def register_dump_reference_with_baseline(
    name: str,
    reference: torch.Tensor,
    baseline: torch.Tensor,
    *,
    max_error_factor: float,
    mean_error_factor: float,
    mean_denominator: int | None = None,
    epsilon: float = 1e-5,
) -> None:
    """Register a reference using the test's baseline max/mean error rule."""
    if not _reference_dump_enabled():
        return
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"invalid dump reference name: {name!r}")
    if not isinstance(reference, torch.Tensor):
        raise TypeError(f"dump reference {name!r} is not a tensor")
    if not isinstance(baseline, torch.Tensor):
        raise TypeError(f"dump baseline {name!r} is not a tensor")
    if tuple(reference.shape) != tuple(baseline.shape):
        raise ValueError(
            f"dump baseline shape for {name!r} does not match the reference: "
            f"{tuple(baseline.shape)} != {tuple(reference.shape)}"
        )
    if max_error_factor < 0 or mean_error_factor < 0 or epsilon < 0:
        raise ValueError("dump baseline comparison parameters must be non-negative")
    reference_numel = reference.numel()
    if mean_denominator is not None and mean_denominator < reference_numel:
        raise ValueError(
            "dump baseline mean denominator must be at least the reference numel"
        )
    _registered_references[name] = _RegisteredReference(
        name=name,
        tensor=reference,
        atol=None,
        rtol=None,
        equal_nan=False,
        baseline=baseline,
        max_error_factor=float(max_error_factor),
        mean_error_factor=float(mean_error_factor),
        mean_denominator=mean_denominator,
        epsilon=float(epsilon),
    )


def _write_registered_references(
    output_dir: Path,
    function_name: str,
    outputs: tuple[_DumpedOutput, ...],
    references: tuple[_RegisteredReference, ...],
) -> None:
    outputs_by_name = {output.name: output for output in outputs}
    unknown_names = sorted(
        reference.name
        for reference in references
        if reference.name not in outputs_by_name
    )
    if unknown_names:
        raise ValueError(
            f"dump references do not match dumped outputs: {unknown_names}"
        )

    comparisons: list[_Comparison] = []
    for reference in references:
        output = outputs_by_name[reference.name]
        if tuple(reference.tensor.shape) != tuple(output.tensor.shape):
            raise ValueError(
                f"dump reference shape for {output.name!r} does not match the "
                f"dumped output: {tuple(reference.tensor.shape)} != "
                f"{tuple(output.tensor.shape)}"
            )
        reference_filename = f"reference_{output.ordinal}_{output.name}.bin"
        comparison: _Comparison = {
            "name": output.name,
            "output": output.filename,
            "reference": reference_filename,
            "shape": list(output.tensor.shape),
            "output_dtype": str(output.tensor.dtype).removeprefix("torch."),
            "reference_dtype": str(reference.tensor.dtype).removeprefix("torch."),
            "output_ordinal": output.ordinal,
        }
        if reference.baseline is None:
            comparison.update(
                {
                    "comparison": "close",
                    "atol": reference.atol,
                    "rtol": reference.rtol,
                    "equal_nan": reference.equal_nan,
                }
            )
        else:
            comparison.update(
                {
                    "comparison": "error_ratio",
                    "baseline": f"baseline_{output.ordinal}_{output.name}.bin",
                    "baseline_dtype": str(reference.baseline.dtype).removeprefix(
                        "torch."
                    ),
                    "max_error_factor": reference.max_error_factor,
                    "mean_error_factor": reference.mean_error_factor,
                    "mean_denominator": reference.mean_denominator
                    or reference.tensor.numel(),
                    "epsilon": reference.epsilon,
                }
            )
        comparisons.append(comparison)
    comparisons.sort(key=lambda item: item["output_ordinal"])
    manifest = {
        "version": 1,
        "function": function_name,
        "comparisons": comparisons,
    }

    with tempfile.TemporaryDirectory(
        prefix=".references.staging-", dir=output_dir
    ) as staging_value:
        staging_dir = Path(staging_value)
        for comparison in comparisons:
            reference = next(
                item for item in references if item.name == comparison["name"]
            )
            (staging_dir / comparison["reference"]).write_bytes(
                _tensor_bytes(reference.tensor)
            )
            if reference.baseline is not None:
                (staging_dir / comparison["baseline"]).write_bytes(
                    _tensor_bytes(reference.baseline)
                )
        (staging_dir / "references.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (staging_dir / "REFERENCE_COMPLETE").write_text(
            "\n".join(item["name"] for item in comparisons) + "\n"
        )

        for comparison in comparisons:
            (staging_dir / comparison["reference"]).replace(
                output_dir / comparison["reference"]
            )
            if "baseline" in comparison:
                (staging_dir / comparison["baseline"]).replace(
                    output_dir / comparison["baseline"]
                )
        (staging_dir / "references.json").replace(output_dir / "references.json")
        (staging_dir / "REFERENCE_COMPLETE").replace(output_dir / "REFERENCE_COMPLETE")


def _resolve_dump_outputs(
    arguments: tuple[Any, ...],
    outputs: Mapping[str, torch.Tensor] | None,
) -> dict[int, str]:
    resolved: dict[int, str] = {}
    for name, tensor in (outputs or {}).items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"dump output {name!r} is not a tensor")
        matches = [
            index for index, argument in enumerate(arguments) if argument is tensor
        ]
        if len(matches) != 1:
            raise ValueError(
                f"dump output {name!r} must occur exactly once in the FFI "
                f"arguments, found {len(matches)} occurrences"
            )
        index = matches[0]
        if index in resolved:
            raise ValueError(
                f"FFI argument {index} is registered as both "
                f"{resolved[index]!r} and {name!r}"
            )
        resolved[index] = name
    return resolved


class _DumpingModule:
    def __init__(
        self,
        module: Any,
        library_path: Path,
        config_arch: MusaArch | None,
        binary_arch: MusaArch | None,
    ) -> None:
        self._module = module
        self._library_path = library_path
        self._config_arch = config_arch
        self._binary_arch = binary_arch

    def get_function(self, function_name: str):
        def dumped_function(
            *arguments: Any,
            _dump_outputs: Mapping[str, torch.Tensor] | None = None,
        ):
            maybe_dump_ffi_call(
                self._library_path,
                function_name,
                arguments,
                outputs=_dump_outputs,
                config_arch=self._config_arch,
                binary_arch=self._binary_arch,
            )
            raise RuntimeError("JIT dump returned without completing")

        return dumped_function

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


def wrap_ffi_module_for_dump(
    module: Any,
    library_path: Path,
    *,
    config_arch: MusaArch | None = None,
    binary_arch: MusaArch | None = None,
) -> Any:
    return _DumpingModule(
        module,
        library_path,
        config_arch,
        binary_arch,
    )


def _dump_output_dir(output_value: str) -> Path:
    if _pytest_session is None:
        return Path(output_value).expanduser().resolve()
    if _pytest_session.current_nodeid is None:
        raise RuntimeError("JIT dump occurred outside a pytest item")
    if _pytest_session.current_dumped:
        raise RuntimeError(
            f"pytest item {_pytest_session.current_nodeid!r} already produced a JIT dump"
        )
    return _pytest_session.root / f"{len(_pytest_session.dumps):04d}"


def _record_pytest_dump(
    output_dir: Path,
    function_name: str,
    config_arch: MusaArch | None,
    binary_arch: MusaArch | None,
) -> None:
    if _pytest_session is None:
        return
    assert _pytest_session.current_nodeid is not None
    _pytest_session.dumps.append(
        {
            "index": len(_pytest_session.dumps),
            "nodeid": _pytest_session.current_nodeid,
            "directory": output_dir.name,
            "function": function_name,
            "config_arch": config_arch.name if config_arch else None,
            "binary_arch": binary_arch.name if binary_arch else None,
            "status": "dumped",
        }
    )
    _pytest_session.current_dumped = True
    _write_pytest_manifest(_pytest_session)


def _resolve_dump_arches(
    config_arch: MusaArch | None,
    binary_arch: MusaArch | None,
) -> tuple[MusaArch | None, MusaArch | None]:
    context = _pytest_session.context if _pytest_session is not None else None
    if context is None:
        context = get_jit_dump_context()
    if context is None:
        return config_arch, binary_arch
    if context.current_arch != context.dump_arch and (
        config_arch is None or binary_arch is None
    ):
        raise RuntimeError(
            "cross-architecture dump requires the operator to declare "
            "config_arch and binary_arch"
        )
    return config_arch or context.dump_arch, binary_arch or context.dump_arch


def maybe_dump_ffi_call(
    library_path: Path,
    function_name: str,
    arguments: tuple[Any, ...],
    *,
    outputs: Mapping[str, torch.Tensor] | None = None,
    config_arch: MusaArch | None = None,
    binary_arch: MusaArch | None = None,
) -> None:
    global _registered_references

    output_value = os.environ.get("MATE_JIT_DUMP_DIR")
    if not output_value:
        return

    config_arch, binary_arch = _resolve_dump_arches(config_arch, binary_arch)

    dump_reference = _reference_dump_enabled()
    references = tuple(_registered_references.values())
    if dump_reference and not references:
        raise RuntimeError(
            "MATE_JIT_DUMP_REFERENCE=1 requires references to be registered "
            "before the target FFI call"
        )

    output_dir = _dump_output_dir(output_value)
    resolved_output_names = _resolve_dump_outputs(arguments, outputs)
    empty_tensor_indices = frozenset(resolved_output_names)
    dumped_outputs = _dumped_outputs(
        arguments, empty_tensor_indices, resolved_output_names
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
    ) as staging_value:
        staging_dir = Path(staging_value)
        shutil.copy2(library_path, staging_dir / "kernel.so")
        shutil.copy2(_resolve_loader_path(), staging_dir / "loader.so")
        shutil.copy2(
            _require_file(_COMPARE_SOURCE, "dump comparator"),
            staging_dir / "compare.py",
        )
        shutil.copy2(
            _require_file(_BENCHMARK_SOURCE, "dump benchmark"),
            staging_dir / "benchmark.py",
        )

        musart_path, musart_soname = _resolve_musart()
        runtime_dir = staging_dir / "runtime"
        runtime_dir.mkdir()
        shutil.copy2(musart_path, runtime_dir / musart_soname)

        _write_replay(
            staging_dir,
            function_name,
            arguments,
            empty_tensor_indices,
            resolved_output_names,
        )
        _write_run_script(staging_dir)
        (staging_dir / "DUMP_COMPLETE").write_text(f"{function_name}\n")

        _commit_dump(staging_dir, output_dir)

    if dump_reference:
        _write_registered_references(
            output_dir,
            function_name,
            dumped_outputs,
            references,
        )
        _registered_references = {}

    _record_pytest_dump(output_dir, function_name, config_arch, binary_arch)
    print(f"JIT dump completed without launching {function_name}", flush=True)
    raise MateJITDumpComplete(output_dir, function_name)
