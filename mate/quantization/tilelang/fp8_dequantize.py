"""Dequantize FP8 payloads with strided block scales."""

from typing import Literal

import torch
import tilelang
import tilelang.language as T

from ._common import _JIT_CONFIG, _can_vectorize, _ceil_div


SfType = Literal["e8m0", "e4m3"]
_SF_VEC_SIZE = 32


def _decode_e8m0(code):
    code = T.cast(code, T.uint32)
    # Match fp8_quantize: E8M0 byte 0 represents the smallest finite scale.
    bits = T.if_then_else(
        code == 0,
        T.uint32(0x00400000),
        code << 23,
    )
    return T.reinterpret(bits, T.float32)


@tilelang.jit(**_JIT_CONFIG)
def tilelang_fp8_dequantize(
    sf_type: SfType,
    q_dtype: torch.dtype,
    output_dtype: torch.dtype,
    groups_per_block: int,
    values_per_thread: int = 32,
):
    batch, rows, cols = T.dynamic("batch, rows, cols")
    qs_b, qs_m = T.dynamic("qs_b, qs_m", "int64")
    ss_b, ss_m, ss_k = T.dynamic("ss_b, ss_m, ss_k", "int64")
    assert groups_per_block > 0
    assert values_per_thread in (2, 16, 32)
    groups = T.ceildiv(cols, _SF_VEC_SIZE)
    threads_per_group = _SF_VEC_SIZE // values_per_thread
    threads = groups_per_block * threads_per_group
    assert threads <= 512
    q_type = T.float8_e4m3 if q_dtype == torch.float8_e4m3fn else T.float8_e5m2
    out_type = {
        torch.float32: T.float32,
        torch.float16: T.float16,
        torch.bfloat16: T.bfloat16,
    }[output_dtype]

    @T.prim_func
    def kernel(
        q: T.StridedTensor[(batch, rows, cols), (qs_b, qs_m, 1), q_type],
        sf: T.StridedTensor[(batch, rows, groups), (ss_b, ss_m, ss_k), T.uint8],
        out: T.Tensor[(batch, rows, cols), out_type],
    ) -> None:
        with T.Kernel(
            batch,
            rows,
            (groups + groups_per_block - 1) // groups_per_block,
            threads=threads,
        ) as (batch_idx, row, block_group):
            if values_per_thread >= 16:
                T.assume(cols % values_per_thread == 0)
                T.assume(qs_b % values_per_thread == 0)
                T.assume(qs_m % values_per_thread == 0)
            else:
                T.assume(cols % 2 == 0)
            tid = T.get_thread_binding()
            lane = tid % threads_per_group
            warp_lane = tid % 32
            group_in_block = tid // threads_per_group
            group = block_group * groups_per_block + group_in_block
            group_valid = group < groups
            safe_group = T.if_then_else(group_valid, group, 0)
            base = safe_group * _SF_VEC_SIZE + lane * values_per_thread
            col = group * _SF_VEC_SIZE + lane * values_per_thread
            scale_local = T.alloc_local((1,), T.float32)
            values = T.alloc_local((values_per_thread,), T.float32)

            scale_local[0] = 0.0
            if lane == 0:
                scale_byte = sf[batch_idx, row, safe_group]
                if sf_type == "e8m0":
                    scale_local[0] = _decode_e8m0(scale_byte)
                else:
                    scale_value = T.reinterpret(scale_byte, T.float8_e4m3)
                    scale_local[0] = T.cast(scale_value, T.float32)
            scale_local[0] = T.shfl_sync(
                scale_local[0],
                (warp_lane // threads_per_group) * threads_per_group,
                32,
            )

            if col < cols:
                if values_per_thread >= 16:
                    for i in T.vectorized(values_per_thread):
                        values[i] = T.cast(q[batch_idx, row, base + i], T.float32)
                else:
                    # Adjacent FP8 bytes need not have a two-byte-aligned address.
                    for i in T.unroll(values_per_thread):
                        values[i] = T.cast(q[batch_idx, row, base + i], T.float32)
                for i in T.vectorized(values_per_thread):
                    out[batch_idx, row, base + i] = T.cast(
                        values[i] * scale_local[0], out_type
                    )

    return kernel.with_attr("global_symbol", "tilelang_fp8_dequantize_kernel")


def _check_args(
    q: torch.Tensor,
    sf: torch.Tensor,
    sf_type: SfType,
    sf_vec_size: int,
    output_dtype: torch.dtype,
) -> None:
    if q.dim() not in (2, 3):
        raise ValueError("q must have shape [M,K] or [B,M,K]")
    if q.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("q must have dtype torch.float8_e4m3fn or torch.float8_e5m2")
    if sf.dtype != torch.uint8:
        raise ValueError("sf must have dtype torch.uint8")
    if q.device.type != "musa" or sf.device != q.device:
        raise ValueError("q and sf must be on the same MUSA device")
    if sf.dim() != q.dim() or sf.shape[:-1] != q.shape[:-1]:
        raise ValueError("q and sf must have matching rank and batch/row dimensions")
    if sf_type not in ("e8m0", "e4m3"):
        raise ValueError("sf_type must be e8m0 or e4m3")
    if sf_vec_size != _SF_VEC_SIZE:
        raise ValueError("only sf_vec_size=32 is supported")
    if output_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            "output_dtype must be torch.float32, torch.float16, or torch.bfloat16"
        )
    if q.shape[-1] == 0 or q.shape[-2] == 0:
        raise ValueError("q must have non-empty dimensions")
    if q.dim() == 3 and q.shape[0] == 0:
        raise ValueError("q must have a non-empty batch dimension")
    if q.shape[-1] % 2:
        raise ValueError("q must have an even K dimension")
    expected_groups = _ceil_div(q.shape[-1], _SF_VEC_SIZE)
    if sf.shape[-1] != expected_groups:
        raise ValueError(f"sf.shape[-1] must be {expected_groups}")
    if q.stride(-1) != 1:
        raise ValueError("q must be contiguous in the K dimension")


def _fp8_dequantize_impl(
    q: torch.Tensor,
    sf: torch.Tensor,
    sf_type: SfType = "e8m0",
    sf_vec_size: int = 32,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Validate inputs and launch the TileLang backend for ``fp8_dequantize``."""
    _check_args(q, sf, sf_type, sf_vec_size, output_dtype)
    input_is_3d = q.dim() == 3
    q3 = q if input_is_3d else q.unsqueeze(0)
    sf3 = sf if input_is_3d else sf.unsqueeze(0)
    batch, rows, cols = q3.shape
    groups = _ceil_div(cols, _SF_VEC_SIZE)
    out = torch.empty((batch, rows, cols), dtype=output_dtype, device=q.device)
    # Aligned full or half groups use vector loads. Other rows retain byte
    # loads with the same tail guard. K itself is always contiguous.
    values_per_thread = 32 if cols % 32 == 0 and _can_vectorize(q3, 32) else 2
    if values_per_thread == 2 and cols % 16 == 0 and _can_vectorize(q3, 16):
        values_per_thread = 16
    if values_per_thread >= 16:
        groups_per_block = 32 if groups <= 32 else 64
    else:
        groups_per_block = 8
    kernel = tilelang_fp8_dequantize(
        sf_type, q.dtype, output_dtype, groups_per_block, values_per_thread
    )
    kernel(q3, sf3, out)
    return out if input_is_3d else out[0]


def main() -> None:
    from mate.quantization import fp8_dequantize

    import argparse
    import math

    from mate.testing.utils import bench_kineto

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--cols", type=int, default=16384)
    parser.add_argument("--q-type", choices=("e4m3", "e5m2"), default="e4m3")
    parser.add_argument("--sf-type", choices=("e8m0", "e4m3"), default="e8m0")
    parser.add_argument(
        "--output-dtype", choices=("fp32", "fp16", "bf16"), default="fp32"
    )
    parser.add_argument("--rep", type=int, default=30)
    args = parser.parse_args()
    if min(args.batch_size, args.rows, args.cols, args.rep) <= 0 or args.cols % 2:
        parser.error("positive batch/rows/cols/rep and even K are required")
    q_dtype = torch.float8_e4m3fn if args.q_type == "e4m3" else torch.float8_e5m2
    output_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.output_dtype]
    device = torch.device("musa")
    torch.manual_seed(0)
    x = torch.randn(
        (args.batch_size, args.rows, args.cols), device=device, dtype=torch.float32
    )
    q = x.to(q_dtype)
    sf = torch.full(
        (args.batch_size, args.rows, _ceil_div(args.cols, 32)),
        127 if args.sf_type == "e8m0" else 0x38,
        dtype=torch.uint8,
        device=device,
    )
    out = fp8_dequantize(q, sf, args.sf_type, output_dtype=output_dtype)
    logical_bytes = q.numel() + sf.numel() + out.numel() * out.element_size()
    seconds = bench_kineto(
        lambda: fp8_dequantize(q, sf, args.sf_type, output_dtype=output_dtype),
        kernel_names="tilelang_fp8_dequantize_kernel",
        num_tests=args.rep,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError("Kineto did not record the FP8 dequantization kernel")
    print(
        f"input={(args.batch_size, args.rows, args.cols)}, output={out.shape} {output_dtype}, "
        f"q={q_dtype}, sf={args.sf_type}, "
        f"{logical_bytes / seconds / 1e9:.3f} logical GB/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
