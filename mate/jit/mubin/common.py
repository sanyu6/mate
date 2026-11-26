from collections.abc import Sequence

import torch

from mate.arch import Arch, MP31_ARCH, MP32_ARCH


def _resolve_grouped_fp8_recipe(
    recipe: tuple[int, int, int] | None,
    arch: Arch,
    *,
    k_grouped: bool = False,
) -> tuple[int, int, int]:
    """Resolve the quantization contract before selecting a grouped kernel."""

    if arch not in (MP31_ARCH, MP32_ARCH):
        raise RuntimeError(
            f"MATE grouped GEMM MUBINs support only MP31 and MP32, got {arch}"
        )
    if arch == MP32_ARCH:
        expected = (1, 1, 32)
    else:
        expected = (1, 1, 128) if k_grouped else (1, 128, 128)
    if recipe is None:
        return expected
    if (arch == MP32_ARCH or k_grouped) and recipe != expected:
        raise ValueError(f"{arch} grouped FP8 GEMM requires recipe={expected}")
    return recipe


def get_asm_dtype_from_torch_dtype(
    tensor_dtype: torch.dtype,
    pack_bits: int = 8,
) -> str:
    if pack_bits == 4:
        if tensor_dtype != torch.int8:
            raise ValueError("4-bit packed ASM dtype requires torch.int8 storage")
        return "int4"
    if pack_bits != 8:
        raise ValueError(f"pack_bits must be 4 or 8, got {pack_bits}")

    return {
        torch.float16: "half",
        torch.bfloat16: "bfloat16",
        torch.float32: "float",
        torch.float8_e4m3fn: "fp8_e4m3",
        torch.float8_e5m2: "fp8_e5m2",
        torch.int8: "int8",
    }[tensor_dtype]


def check_tensor_same_device(tensors: Sequence[torch.Tensor]) -> None:
    if not tensors:
        return
    reference = tensors[0]
    if any(tensor.device != reference.device for tensor in tensors[1:]):
        raise ValueError("all tensors must be on the same device")


def check_musa(tensor: torch.Tensor) -> None:
    if tensor.device.type != "musa":
        raise ValueError("tensor must be on MUSA device")
