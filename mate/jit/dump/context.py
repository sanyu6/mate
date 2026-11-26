from __future__ import annotations

import dataclasses
import os

from ..env import MusaArch

JIT_DUMP_DIR_ENV = "MATE_JIT_DUMP_DIR"
MUSA_ARCH_LIST_ENV = "MATE_MUSA_ARCH_LIST"


@dataclasses.dataclass(frozen=True)
class JITDumpContext:
    current_arch: MusaArch
    dump_arch: MusaArch


def detect_current_musa_arch() -> MusaArch:
    try:
        import torch
        import torch_musa  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "JIT dump requires torch and torch_musa to detect the current architecture"
        ) from exc

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("JIT dump requires an available MUSA device")

    properties = torch.musa.get_device_properties(torch.musa.current_device())
    return MusaArch(major=int(properties.major), minor=int(properties.minor))


def get_jit_dump_context() -> JITDumpContext | None:
    if not os.environ.get(JIT_DUMP_DIR_ENV):
        return None

    current_arch = detect_current_musa_arch()
    arch_list_value = os.environ.get(MUSA_ARCH_LIST_ENV)
    if arch_list_value is None:
        dump_arch = current_arch
    else:
        dump_targets = arch_list_value.split()
        if len(dump_targets) != 1:
            raise ValueError("JIT dump requires exactly one MATE_MUSA_ARCH_LIST target")
        dump_arch = MusaArch.parse(dump_targets[0])
    return JITDumpContext(current_arch=current_arch, dump_arch=dump_arch)
