"""Runtime helpers shared by MATE Python APIs and compatibility wrappers."""

import functools
from collections.abc import Collection
from typing import Optional, overload

import torch

from .arch import Arch, MP31_ARCH, MP32_ARCH

_num_mps: Optional[int] = None


def _device_index(device: Optional[torch.device] = None) -> int:
    if device is None or device.index is None:
        return torch.musa.current_device()
    return device.index


@functools.cache
def _get_physical_num_mps(device_index: int) -> int:
    return torch.musa.get_device_properties(device_index).multi_processor_count


def get_physical_num_mps(device: Optional[torch.device] = None) -> int:
    """Return the physical MUSA MP count for ``device``."""
    return _get_physical_num_mps(_device_index(device))


@functools.cache
def _get_arch(device_index: int) -> Arch:
    major, minor = torch.musa.get_device_capability(device_index)
    return Arch(major=int(major), minor=int(minor))


def get_arch(device: Optional[torch.device] = None) -> Arch:
    """Return the architecture of ``device``, defaulting to the current device."""
    return _get_arch(_device_index(device))


@overload
def get_musa_arch(device: torch.Tensor) -> Arch | None: ...


@overload
def get_musa_arch(device: torch.device | None = None) -> Arch: ...


def get_musa_arch(
    device: torch.Tensor | torch.device | None = None,
) -> Arch | None:
    """Return the exact MUSA architecture of ``device``.

    Resolve the current device before entering the cache so an omitted device
    never becomes a stale cache key when the process changes current devices.
    Tensor inputs are accepted for operator dispatch; non-MUSA tensors have no
    MUSA architecture and return ``None``.
    """
    if isinstance(device, torch.Tensor):
        if device.device.type != "musa":
            return None
        device = device.device
    return _get_arch(_device_index(device))


def require_musa_arch(
    tensor: torch.Tensor,
    supported: Collection[Arch],
    operation: str,
) -> Arch | None:
    """Validate a MUSA tensor architecture while allowing host-side tests."""
    arch = get_musa_arch(tensor)
    if arch is None:
        return None
    if arch not in supported:
        supported_names = ", ".join(
            f"MP{supported_arch.major}{supported_arch.minor}"
            for supported_arch in supported
        )
        raise RuntimeError(
            f"{operation} only supports {supported_names}; "
            f"got MP{arch.major}{arch.minor}"
        )
    return arch


def get_num_mps() -> int:
    """Return the effective MP count for MATE kernel dispatch."""
    if _num_mps is not None:
        return _num_mps
    return get_physical_num_mps()


def resolve_num_mps(
    device: torch.device | None = None, num_mps: int | None = None
) -> int:
    """Resolve an explicit MP count or fall back to the effective device count."""
    if num_mps is not None and num_mps > 0:
        return num_mps
    if _num_mps is not None:
        return _num_mps
    return get_physical_num_mps(device)


def set_num_mps(num: int | None) -> None:
    """Limit the maximum MP count available to MATE JIT kernels.

    Pass ``None`` to reset to the physical MP count of the target device.
    """
    global _num_mps
    assert num is None or (isinstance(num, int) and num > 0), (
        f"num_mps must be a positive int or None, got {num}"
    )
    _num_mps = num


def get_compute_capability(as_int: bool = False) -> int | tuple[int, int]:
    """Return the compute capability of the current MUSA device."""
    arch = get_arch()
    return arch.major * 10 + arch.minor if as_int else (arch.major, arch.minor)


__all__ = [
    "MP31_ARCH",
    "MP32_ARCH",
    "get_arch",
    "get_compute_capability",
    "get_musa_arch",
    "get_num_mps",
    "get_physical_num_mps",
    "require_musa_arch",
    "resolve_num_mps",
    "set_num_mps",
]
