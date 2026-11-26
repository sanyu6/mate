"""TileLang MUSA implementation of segmented bit packing."""

from typing import Literal

import torch
import tilelang
import tilelang.language as T

from ._common import _JIT_CONFIG


BitOrder = Literal["big", "little"]
_BYTES_PER_BLOCK = 256


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
def tilelang_segment_packbits(
    batch_size: int, max_packed_segment_length: int, bitorder: BitOrder
):
    num_blocks = (max_packed_segment_length + _BYTES_PER_BLOCK - 1) // _BYTES_PER_BLOCK

    @T.prim_func
    def tilelang_segment_packbits_kernel(
        x: T.Tensor[(T.dynamic("num_elements"),), T.uint8],
        indptr: T.Tensor[(batch_size + 1,), T.int32],
        new_indptr: T.Tensor[(batch_size + 1,), T.int32],
        y: T.Tensor[(T.dynamic("num_output_bytes"),), T.uint8],
    ) -> None:
        with T.Kernel(num_blocks, batch_size, threads=_BYTES_PER_BLOCK) as (
            block_idx,
            segment_idx,
        ):
            thread_idx = T.get_thread_binding()
            byte_idx = block_idx * _BYTES_PER_BLOCK + thread_idx
            segment_start = indptr[segment_idx]
            segment_end = indptr[segment_idx + 1]
            segment_length = segment_end - segment_start
            full_packed_bytes = segment_length // 8
            packed_length = (segment_length + 7) // 8

            output_start = new_indptr[segment_idx]
            if byte_idx < full_packed_bytes:
                packed = T.alloc_local((1,), T.uint8)
                packed[0] = 0
                input_start = segment_start + byte_idx * 8
                values = T.reinterpret(x[input_start : input_start + 8], T.uint64)
                low = values & 0xFFFFFFFF
                high = values >> 32
                if bitorder == "big":
                    low_magic = 0x80402010
                    high_magic = 0x08040201
                else:
                    low_magic = 0x01020408
                    high_magic = 0x10204080
                packed[0] = T.cast(
                    (((low * low_magic) >> 24) | ((high * high_magic) >> 24)) & 0xFF,
                    T.uint8,
                )
                y[output_start + byte_idx] = packed[0]
            elif byte_idx < packed_length:
                packed = T.alloc_local((1,), T.uint8)
                packed[0] = 0
                for bit in T.serial(8):
                    shift = 7 - bit if bitorder == "big" else bit
                    source_idx = segment_start + byte_idx * 8 + bit
                    source_valid = source_idx < segment_end
                    safe_source_idx = T.if_then_else(source_valid, source_idx, 0)
                    value = T.cast(
                        T.if_then_else(source_valid, x[safe_source_idx], False),
                        T.uint8,
                    )
                    packed[0] = packed[0] | (value << shift)
                y[output_start + byte_idx] = packed[0]

    return tilelang_segment_packbits_kernel


def _segment_packbits_impl(
    x: torch.Tensor,
    indptr: torch.Tensor,
    bitorder: BitOrder = "big",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate inputs and launch the TileLang backend for ``segment_packbits``."""
    _check_input(x)
    bitorder = _check_bitorder(bitorder)
    if indptr.dim() != 1 or indptr.numel() < 1:
        raise ValueError("indptr must be a non-empty 1D tensor.")
    if indptr.device != x.device:
        raise ValueError("indptr and x must be on the same MUSA device.")
    if indptr.dtype not in (torch.int32, torch.int64):
        raise ValueError("indptr must have dtype torch.int32 or torch.int64.")
    indptr_i32 = indptr.to(torch.int32)
    segment_lengths = indptr_i32[1:] - indptr_i32[:-1]
    if torch.any(segment_lengths < 0):
        raise ValueError("indptr must be monotonically non-decreasing.")
    if indptr_i32[-1].item() != x.numel():
        raise ValueError("indptr[-1] must equal x.numel().")

    packed_lengths = (segment_lengths + 7) // 8
    new_indptr = torch.zeros_like(indptr_i32)
    new_indptr[1:] = torch.cumsum(packed_lengths, dim=0)
    output_bytes = int(new_indptr[-1].item())
    output = torch.empty((output_bytes,), dtype=torch.uint8, device=x.device)
    if output_bytes == 0:
        return output, new_indptr

    x_u8 = x.to(torch.bool).view(torch.uint8)
    max_packed_length = int(packed_lengths.max().item())
    tilelang_segment_packbits(indptr.numel() - 1, max_packed_length, bitorder)(
        x_u8, indptr_i32, new_indptr, output
    )
    return output, new_indptr


def main() -> None:
    from mate.quantization import segment_packbits

    import argparse

    from mate.testing.utils import bench_kineto

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-source", action="store_true")
    parser.add_argument("--num-elements", type=int, default=1024 * (1 << 20))
    parser.add_argument("--num-segments", type=int, default=128)
    parser.add_argument("--rep", type=int, default=30)
    args = parser.parse_args()
    if args.num_elements < 0:
        parser.error("--num-elements must be non-negative")
    if args.num_segments <= 0:
        parser.error("--num-segments must be positive")
    if args.rep <= 0:
        parser.error("--rep must be positive")

    device = torch.device("musa")
    x = torch.randint(
        0, 2, (args.num_elements,), device=device, dtype=torch.int32
    ).bool()
    lengths = torch.full(
        (args.num_segments,),
        args.num_elements // args.num_segments,
        dtype=torch.int32,
        device=device,
    )
    lengths[: args.num_elements % args.num_segments] += 1
    indptr = torch.cat(
        (torch.zeros(1, dtype=torch.int32, device=device), lengths.cumsum(0))
    )

    segment_packbits(x, indptr)
    if args.show_source and args.num_elements:
        tilelang_segment_packbits(
            args.num_segments, int(((lengths + 7) // 8).max().item()), "big"
        ).show_source()
    segment_seconds = bench_kineto(
        lambda: segment_packbits(x, indptr),
        kernel_names="tilelang_segment_packbits_kernel",
        num_tests=args.rep,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    input_bytes = x.numel() * x.element_size()
    metadata_bytes = 2 * indptr.numel() * 4
    segment_lengths = indptr[1:] - indptr[:-1]
    output_bytes = int(((segment_lengths + 7) // 8).sum().item())
    logical_bytes = input_bytes + metadata_bytes + output_bytes
    segment_gbps = logical_bytes / segment_seconds / 1e9
    print(f"segment_packbits: {segment_gbps:.3f} logical GB/s (read + write)")
    print(f"packed output: {output_bytes} bytes; segments: {args.num_segments}")


if __name__ == "__main__":
    main()
