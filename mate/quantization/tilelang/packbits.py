"""TileLang MUSA implementations of bit packing utilities."""

from typing import Literal

import torch
import tilelang
import tilelang.language as T

from ._common import _JIT_CONFIG


BitOrder = Literal["big", "little"]
_BYTES_PER_BLOCK = 256
_THREADS_PER_BLOCK = 128
_BYTES_PER_THREAD = 2
_BITS_PER_THREAD = _BYTES_PER_THREAD * 8


def _check_bitorder(bitorder: str) -> BitOrder:
    if bitorder not in ("big", "little"):
        raise ValueError(f"bitorder must be 'big' or 'little', got {bitorder!r}.")
    return bitorder  # type: ignore[return-value]


def _check_input(x: torch.Tensor) -> None:
    if x.dim() != 1:
        raise ValueError(f"x must be 1D, got shape {tuple(x.shape)}.")
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("A MUSA device is required.")
    if x.device.type != "musa":
        raise ValueError(f"x must be on a MUSA device, got {x.device}.")


@tilelang.jit(**_JIT_CONFIG)
def tilelang_packbits(num_elements: int, bitorder: BitOrder):
    num_bytes = (num_elements + 7) // 8
    num_blocks = (num_bytes + _BYTES_PER_BLOCK - 1) // _BYTES_PER_BLOCK
    full_blocks = num_elements // (_BYTES_PER_BLOCK * 8)

    @T.prim_func
    def tilelang_packbits_kernel(
        x: T.Tensor[(num_elements,), T.uint8],
        y: T.Tensor[(num_bytes,), T.uint8],
    ) -> None:
        with T.Kernel(num_blocks, threads=_THREADS_PER_BLOCK) as block_idx:
            thread_idx = T.get_thread_binding()
            byte_idx = block_idx * _BYTES_PER_BLOCK + thread_idx * _BYTES_PER_THREAD
            if block_idx < full_blocks:
                values = T.alloc_local((_BITS_PER_THREAD,), T.uint8)
                packed = T.alloc_local((_BYTES_PER_THREAD,), T.uint8)
                for input_idx in T.vectorized(_BITS_PER_THREAD):
                    values[input_idx] = x[
                        block_idx * _BYTES_PER_BLOCK * 8
                        + thread_idx * _BITS_PER_THREAD
                        + input_idx
                    ]
                for output_idx in T.unroll(_BYTES_PER_THREAD):
                    packed[output_idx] = 0
                    for bit in T.serial(8):
                        shift = 7 - bit if bitorder == "big" else bit
                        value = T.cast(values[output_idx * 8 + bit], T.uint8)
                        packed[output_idx] = packed[output_idx] | (value << shift)
                    y[byte_idx + output_idx] = packed[output_idx]
            else:
                for output_idx in T.unroll(_BYTES_PER_THREAD):
                    output_idx_global = byte_idx + output_idx
                    if output_idx_global < num_bytes:
                        packed = T.alloc_local((1,), T.uint8)
                        packed[0] = 0
                        for bit in T.serial(8):
                            shift = 7 - bit if bitorder == "big" else bit
                            source_idx = output_idx_global * 8 + bit
                            source_valid = source_idx < num_elements
                            safe_source_idx = T.if_then_else(
                                source_valid, source_idx, 0
                            )
                            value = T.cast(
                                T.if_then_else(source_valid, x[safe_source_idx], False),
                                T.uint8,
                            )
                            packed[0] = packed[0] | (value << shift)
                        y[output_idx_global] = packed[0]

    return tilelang_packbits_kernel


def _packbits_impl(x: torch.Tensor, bitorder: BitOrder = "big") -> torch.Tensor:
    """Validate inputs and launch the TileLang backend for ``packbits``."""
    _check_input(x)
    bitorder = _check_bitorder(bitorder)
    num_elements = x.numel()
    output = torch.empty((num_elements + 7) // 8, dtype=torch.uint8, device=x.device)
    if num_elements == 0:
        return output
    x_u8 = x.to(torch.bool).view(torch.uint8)
    tilelang_packbits(num_elements, bitorder)(x_u8, output)
    return output


def main() -> None:
    from mate.quantization import packbits

    import argparse

    from mate.testing.utils import bench_kineto

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-source", action="store_true")
    parser.add_argument("--num-elements", type=int, default=1024 * (1 << 20))
    parser.add_argument("--rep", type=int, default=30)
    args = parser.parse_args()
    if args.num_elements < 0:
        parser.error("--num-elements must be non-negative")
    if args.rep <= 0:
        parser.error("--rep must be positive")

    device = torch.device("musa")
    x = torch.randint(
        0, 2, (args.num_elements,), device=device, dtype=torch.int32
    ).bool()
    packbits(x)
    if args.show_source and args.num_elements:
        tilelang_packbits(args.num_elements, "big").show_source()
    seconds = bench_kineto(
        lambda: packbits(x),
        kernel_names="tilelang_packbits_kernel",
        num_tests=args.rep,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    input_bytes = x.numel() * x.element_size()
    output_bytes = (x.numel() + 7) // 8
    logical_bytes = input_bytes + output_bytes
    gbps = logical_bytes / seconds / 1e9
    print(f"packbits: {gbps:.3f} logical GB/s (read + write)")
    print(f"packed output: {output_bytes} bytes")


if __name__ == "__main__":
    main()
