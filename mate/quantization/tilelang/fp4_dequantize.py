"""TileLang MUSA FP4 dequantization with strided block scales."""

from typing import Literal

import torch
import tilelang
import tilelang.language as T

from ._common import _JIT_CONFIG, _can_vectorize, _ceil_div


SfType = Literal["e8m0", "e4m3"]


def _decode_scale(raw, sf_type: SfType):
    """Decode one raw scale byte to float32 inside the device kernel."""
    scale_type = T.float8_e4m3fn if sf_type == "e4m3" else T.float8_e8m0fnu
    return T.cast(T.reinterpret(raw, scale_type), T.float32)


@tilelang.jit(**_JIT_CONFIG)
def tilelang_fp4_dequantize(
    sf_type: SfType,
    output_dtype: torch.dtype,
    groups_per_block: int,
    values_per_thread: int,
):
    """Dequantize packed FP4 values with strided 32-value scales."""
    batch, rows, cols = T.dynamic("batch, rows, cols")
    qs_b, qs_m = T.dynamic("qs_b, qs_m", "int64")
    ss_b, ss_m, ss_k = T.dynamic("ss_b, ss_m, ss_k", "int64")
    assert groups_per_block > 0
    assert values_per_thread in (2, 16, 32)
    groups = T.ceildiv(cols, 32)
    threads_per_group = 32 // values_per_thread
    threads = groups_per_block * threads_per_group

    @T.prim_func
    def kernel(
        q: T.StridedTensor[
            (batch, rows, cols),
            (qs_b, qs_m, 1),
            T.float4_e2m1fn,
        ],
        sf: T.StridedTensor[
            (batch, rows, groups),
            (ss_b, ss_m, ss_k),
            T.uint8,
        ],
        out: T.Tensor[
            (batch, rows, cols),
            output_dtype,
        ],
    ) -> None:
        with T.Kernel(
            batch,
            rows,
            (groups + groups_per_block - 1) // groups_per_block,
            threads=threads,
        ) as (batch_idx, row, block_group):
            T.assume(cols % values_per_thread == 0)
            T.assume(qs_b % values_per_thread == 0)
            T.assume(qs_m % values_per_thread == 0)
            tid = T.get_thread_binding()
            group_in_block = tid // threads_per_group
            byte_in_group = tid % threads_per_group
            group = block_group * groups_per_block + group_in_block
            group_valid = group < groups
            safe_group = T.if_then_else(group_valid, group, 0)
            safe_col = safe_group * 32 + byte_in_group * values_per_thread
            scale_byte = sf[batch_idx, row, safe_group]
            scale = _decode_scale(scale_byte, sf_type)
            col = group * 32 + byte_in_group * values_per_thread
            # Two FP4 values occupy one byte, so even K always permits a pair.
            # Wider chunks require aligned addresses and K divisible by their width.
            if col < cols:
                for i in T.vectorized(values_per_thread):
                    value = q[batch_idx, row, safe_col + i]
                    out[batch_idx, row, col + i] = T.cast(
                        T.cast(value, T.float32) * scale, output_dtype
                    )

    return kernel.with_attr("global_symbol", "tilelang_fp4_dequantize_kernel")


def _check_args(
    q: torch.Tensor, sf: torch.Tensor, sf_type: SfType, output_dtype: torch.dtype
) -> tuple[SfType, torch.dtype]:
    if q.dim() not in (2, 3):
        raise ValueError(f"q must be rank 2 or 3, got shape {tuple(q.shape)}")
    if q.dtype not in (torch.uint8, torch.float4_e2m1fn_x2):
        raise ValueError("q must have dtype torch.uint8 or torch.float4_e2m1fn_x2")
    if sf.dtype != torch.uint8:
        raise ValueError("sf must have dtype torch.uint8")
    if q.device.type != "musa" or sf.device != q.device:
        raise ValueError("q and sf must be on the same MUSA device")
    if sf.dim() != q.dim():
        raise ValueError("q and sf must have the same rank")
    if sf_type not in ("e8m0", "e4m3"):
        raise ValueError(f"unsupported sf_type: {sf_type!r}")
    if output_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            "output_dtype must be torch.float32, torch.float16, or torch.bfloat16"
        )
    if q.shape[-1] == 0 or q.shape[-2] == 0:
        raise ValueError("q must have non-empty dimensions")
    if q.dim() == 3 and q.shape[0] == 0:
        raise ValueError("q must have a non-empty batch dimension")
    if q.shape[:-1] != sf.shape[:-1]:
        raise ValueError(
            "q and sf must have matching batch and row dimensions, got "
            f"{tuple(q.shape)} and {tuple(sf.shape)}"
        )
    packed_cols = q.shape[-1]
    cols = packed_cols * 2
    expected_groups = _ceil_div(cols, 32)
    if sf.shape[-1] != expected_groups:
        raise ValueError(f"sf.shape[-1] must be {expected_groups}, got {sf.shape[-1]}")
    if q.stride(-1) != 1:
        raise ValueError("q must be contiguous in the K dimension")
    return sf_type, output_dtype


def _fp4_dequantize_impl(
    q: torch.Tensor,
    sf: torch.Tensor,
    sf_type: SfType = "e8m0",
    sf_vec_size: int = 32,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Validate inputs and launch the TileLang backend for ``fp4_dequantize``."""
    if sf_vec_size != 32:
        raise ValueError("only sf_vec_size=32 is supported")
    sf_type, output_dtype = _check_args(q, sf, sf_type, output_dtype)
    input_is_3d = q.dim() == 3
    batch = q.shape[0] if input_is_3d else 1
    rows = q.shape[-2]
    packed_cols = q.shape[-1]
    cols = packed_cols * 2
    groups = _ceil_div(cols, sf_vec_size)
    q3 = q if input_is_3d else q.unsqueeze(0)
    sf3 = sf if input_is_3d else sf.unsqueeze(0)
    if q3.dtype != torch.uint8:
        q3 = q3.view(torch.uint8)

    out3 = torch.empty((batch, rows, cols), dtype=output_dtype, device=q.device)
    # Read aligned full or half groups where possible. Arbitrary row starts
    # retain packed-byte reads without requiring a contiguous input tensor.
    values_per_thread = 32 if cols % 32 == 0 and _can_vectorize(q3, 16) else 2
    if values_per_thread == 2 and cols % 16 == 0 and _can_vectorize(q3, 8):
        values_per_thread = 16
    if values_per_thread >= 16:
        groups_per_block = 32 if groups <= 32 else (64 if groups <= 64 else 128)
    else:
        groups_per_block = 8
    kernel = tilelang_fp4_dequantize(
        sf_type, output_dtype, groups_per_block, values_per_thread
    )
    kernel(q3, sf3, out3)
    return out3 if input_is_3d else out3[0]


def main() -> None:
    from mate.quantization import fp4_dequantize

    import argparse
    import math

    from mate.testing.utils import bench_kineto

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--cols", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--output-dtype", choices=("fp32", "bf16", "fp16"), default="fp32"
    )
    parser.add_argument("--sf-type", choices=("e8m0", "e4m3"), default="e8m0")
    parser.add_argument("--rep", type=int, default=30)
    args = parser.parse_args()
    if args.rows <= 0 or args.cols <= 0 or args.batch_size <= 0 or args.rep <= 0:
        parser.error("rows, cols, batch-size, and rep must be positive")
    if args.cols % 2 != 0:
        parser.error("cols must be even")

    from mate.quantization import fp4_quantize

    device = torch.device("musa")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    output_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.output_dtype]
    x = torch.randn((args.batch_size, args.rows, args.cols), device=device, dtype=dtype)
    q, sf = fp4_quantize(x, sf_type=args.sf_type)
    del x

    def run():
        return fp4_dequantize(q, sf, sf_type=args.sf_type, output_dtype=output_dtype)

    # Compile and warm up before profiling; quantization is outside the timed call.
    out = run()

    logical_bytes = (
        q.numel() * q.element_size()
        + sf.numel() * sf.element_size()
        + out.numel() * out.element_size()
    )
    print(
        f"shape={tuple(out.shape)}, output_dtype={out.dtype}, "
        f"sf_type={args.sf_type}, rep={args.rep}, flush_l2=True"
    )
    del out
    torch.musa.synchronize()
    name = "tilelang_fp4_dequantize_kernel"
    seconds = bench_kineto(
        run,
        kernel_names=name,
        num_tests=args.rep,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError(f"Kineto did not record a valid duration for {name}")
    gbps = logical_bytes / seconds / 1e9
    print(f"{name}: {gbps:.3f} logical GB/s (q + sf read, output write)")


if __name__ == "__main__":
    main()
