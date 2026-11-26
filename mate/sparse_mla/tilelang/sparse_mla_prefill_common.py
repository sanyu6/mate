# ruff: noqa
"""Shared sparse prefill host/JIT plumbing for V4 and V3.2."""

from __future__ import annotations

from typing import Optional

import torch
import tilelang


SPARSE_PREFILL_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
    tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
}


SPARSE_PREFILL_COMPILE_FLAGS = [
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-fno-strict-aliasing",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
    "-mllvm",
    "-mtgpu-load-store-2d=1",
    "-mllvm",
    "-mtgpu-enable-postra-sched=1",
]

# Keep the measured min-register schedule on the optimized MP32 factory, so
# direct compilation and interface calls agree without changing unrelated
# MP31/V32 factories that also use the generic flags above.
SPARSE_PREFILL_MP32_OPTIMIZED_COMPILE_FLAGS = [
    *SPARSE_PREFILL_COMPILE_FLAGS,
    "-mllvm",
    "-mtgpu-enchanced-minreg-schedule=1",
]


def validate_token_lengths(
    lengths: Optional[torch.Tensor],
    seq_len: int,
    name: str,
    max_length: Optional[int] = None,
) -> Optional[torch.Tensor]:
    if lengths is None:
        return None
    assert lengths.dtype == torch.int32, f"{name} must be int32"
    assert lengths.shape == (seq_len,), f"{name} must have shape [S_q]"
    if lengths.shape[-1] != 1:
        assert lengths.stride(-1) == 1, f"{name} last dimension must be contiguous"
    # Do not read device values back to Python here.  `.item()` on a MUSA
    # tensor synchronizes the stream and is illegal during graph capture.  CPU
    # callers (fixture/preflight paths) still get an eager contract check;
    # device callers must validate before capture or rely on the kernel's
    # masked input contract.
    if max_length is not None and lengths.device.type == "cpu":
        min_value = int(lengths.min().item())
        max_value = int(lengths.max().item())
        assert min_value >= 0, f"{name} values must be >= 0, got {min_value}"
        assert max_value <= max_length, (
            f"{name} values must be <= {max_length}, got {max_value}"
        )
    return lengths.contiguous()


def validate_prefill_attn_sink(
    attn_sink: Optional[torch.Tensor],
    heads: int,
) -> Optional[torch.Tensor]:
    if attn_sink is None:
        return None
    assert attn_sink.dtype == torch.float32, "attn_sink must be float32"
    assert attn_sink.shape == (heads,), "attn_sink must have shape [H_q]"
    if attn_sink.shape[-1] != 1:
        assert attn_sink.stride(-1) == 1, "attn_sink last dimension must be contiguous"
    return attn_sink.contiguous()


def prepare_sparse_mla_strided_tensor(
    name: str, tensor: torch.Tensor, multiple: int
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Validate a runtime-strided input once and normalize singleton strides."""

    shape = tensor.shape
    strides = tensor.stride()
    normalized = None
    if strides[-1] != 1:
        assert shape[-1] == 1, f"{name} last dimension must be contiguous"
        normalized = list(strides)
        normalized[-1] = 1
    for dim in range(len(strides) - 1):
        stride = strides[dim]
        if stride > 0 and stride % multiple == 0:
            continue
        if shape[dim] != 1:
            assert stride > 0, f"{name} stride({dim}) must be positive, got {stride}"
            raise AssertionError(
                f"{name} stride({dim}) must be divisible by {multiple}, got {stride}"
            )
        if normalized is None:
            normalized = list(strides)
        normalized[dim] = multiple
    if normalized is None:
        return tensor, shape
    return (
        torch.as_strided(tensor, shape, normalized, tensor.storage_offset()),
        shape,
    )
