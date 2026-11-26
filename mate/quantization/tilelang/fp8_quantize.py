"""Blockwise FP8 quantization with one MN-major scale per 32 K elements."""

from dataclasses import dataclass
from typing import Literal

import torch
import tilelang
import tilelang.language as T

from ._common import (
    _JIT_CONFIG,
    _can_vectorize,
    _ceil_div,
    _scale_store_bytes,
    _e8m0_code,
    _e8m0_float,
    _max_abs,
    _can_use_tme,
    _TME_COLS,
)
from ._quantize import tilelang_quantize_masked
from ._quantize_grouped import tilelang_quantize_grouped


SfType = Literal["e8m0", "e4m3"]
_SF_VEC_SIZE = 32
_E4M3_MIN = 2.0**-9
_VALUES_PER_THREAD = 8


@dataclass(frozen=True)
class _RowConfig:
    rows_per_block: int
    threads_per_row: int
    values_per_thread: int
    cols_per_block: int


def _resolve_row_config(rows: int, cols: int) -> _RowConfig:
    # Small M needs enough independent CTAs before coarsening across rows.
    if rows >= 1024 and rows % 32 == 0 and cols % 4096 == 0:
        return _RowConfig(32, 8, 32, 4096)
    return _RowConfig(1, 256, 32, min(8192, max(32, 1 << (cols - 1).bit_length())))


@tilelang.jit(**_JIT_CONFIG)
def tilelang_fp8_quantize(
    sf_type: SfType,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    rows_per_block: int,
    threads_per_row: int,
    values_per_thread: int,
    cols_per_block: int,
    store_bytes: int,
    use_tme: bool = False,
    has_k_tail: bool = False,
):
    batch, rows, cols = T.dynamic("batch, rows, cols")
    stride_b, stride_m = T.dynamic("stride_b, stride_m", "int64")
    assert cols_per_block % _SF_VEC_SIZE == 0
    assert values_per_thread in (8, 16, 32)
    assert not has_k_tail or (use_tme and values_per_thread == 16)
    groups = T.dynamic("groups")
    threads = rows_per_block * threads_per_row
    tile_width = threads_per_row * values_per_thread
    threads_per_scale = _SF_VEC_SIZE // values_per_thread
    groups_per_step = tile_width // _SF_VEC_SIZE
    groups_per_block = cols_per_block // _SF_VEC_SIZE
    steps = (cols_per_block + tile_width - 1) // tile_width
    shared_groups = steps * groups_per_step
    store_steps = (rows_per_block * shared_groups + threads * store_bytes - 1) // (
        threads * store_bytes
    )
    # MUSA's vector FP8 conversions use SATFINITE, so a separate clamp of the
    # normalized values would duplicate the conversion's saturation.
    # The MUSA backend's vector-conversion lowering recognizes the canonical
    # E4M3 type.  ``float8_e4m3fn`` remains the public torch storage dtype, but
    # using it here makes the backend expand the cast to scalar conversions.
    q_type = T.float8_e4m3 if output_dtype == torch.float8_e4m3fn else T.float8_e5m2
    q_max = 448.0 if output_dtype == torch.float8_e4m3fn else 57344.0

    @T.prim_func
    def kernel(
        x: T.StridedTensor[(batch, rows, cols), (stride_b, stride_m, 1), input_dtype],
        q: T.Tensor[(batch, rows, cols), q_type],
        sf: T.StridedTensor[(batch, rows, groups), (rows * groups, 1, rows), T.uint8],
    ) -> None:
        with T.Kernel(
            (cols + cols_per_block - 1) // cols_per_block if use_tme else batch,
            rows // rows_per_block,
            batch if use_tme else (cols + cols_per_block - 1) // cols_per_block,
            threads=threads,
        ) as (block_x, block_m, block_z):
            # K tiles are adjacent in the launch order to keep large inputs local.
            batch_idx = block_z if use_tme else block_x
            block_k = block_x if use_tme else block_z
            T.assume(cols % (16 if has_k_tail else 32) == 0)
            if rows_per_block > 1 and not has_k_tail:
                # The row-coarsened dispatch requires complete K tiles.
                T.assume(cols % cols_per_block == 0)
            T.assume(stride_b % 8 == 0)
            T.assume(stride_m % 8 == 0)
            T.assume(rows % store_bytes == 0)
            tid = T.get_thread_binding()
            lane = tid % threads_per_row
            row_in_block = tid // threads_per_row
            T.assume(rows % rows_per_block == 0)
            row = block_m * rows_per_block + row_in_block
            warp_lane = tid % 32
            values = T.alloc_local((values_per_thread,), input_dtype)
            values_float = T.alloc_local((values_per_thread,), T.float32)
            amax = T.alloc_local((1,), T.float32)
            scale_local = T.alloc_local((1,), T.float32)
            sf_shared = T.alloc_shared((shared_groups, rows_per_block), T.uint8)

            if use_tme:
                x_shared = T.alloc_shared((rows_per_block, cols_per_block), input_dtype)
                T.annotate_layout(
                    {x_shared: tilelang.layout.make_linear_layout(x_shared)}
                )
                x_ready = T.alloc_barrier(arrive_count=threads)
                T.tma_copy(
                    x[
                        batch_idx,
                        block_m * rows_per_block : (block_m + 1) * rows_per_block,
                        block_k * cols_per_block : (block_k + 1) * cols_per_block,
                    ],
                    x_shared,
                    barrier=x_ready,
                )
                T.mbarrier_arrive(x_ready)
                T.mbarrier_wait_parity(x_ready, 0)

            for step in T.serial(steps):
                base = (
                    block_k * cols_per_block
                    + step * tile_width
                    + lane * values_per_thread
                )
                # K and each thread's start are multiples of values_per_thread:
                # a thread owns either an entirely valid vector or no elements.
                for i in T.vectorized(values_per_thread):
                    values[i] = 0
                    if use_tme:
                        values[i] = x_shared[
                            row_in_block,
                            step * tile_width + lane * values_per_thread + i,
                        ]
                    elif rows_per_block > 1 or base < cols:
                        values[i] = x[batch_idx, row, base + i]
                for i in T.vectorized(values_per_thread):
                    values_float[i] = T.cast(values[i], T.float32)
                if use_tme:
                    amax[0] = _max_abs(values_float, 0, values_per_thread)
                else:
                    amax[0] = 0.0
                    for i in T.unroll(values_per_thread):
                        amax[0] = T.max(amax[0], T.abs(values_float[i]))
                if threads_per_scale >= 4:
                    amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 2))
                if threads_per_scale >= 2:
                    amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 1))

                scale_local[0] = 0.0
                if lane % threads_per_scale == 0:
                    scale_local[0] = amax[0] * (1.0 / q_max)
                    if sf_type == "e8m0":
                        code = _e8m0_code(scale_local[0])
                        scale_byte = T.cast(code, T.uint8)
                        scale_local[0] = _e8m0_float(code)
                    else:
                        # E5M2's large range can otherwise round a normal
                        # block's E4M3 scale to zero. Zero blocks keep scale 0.
                        scale_local[0] = T.if_then_else(
                            amax[0] > 0.0,
                            T.min(T.max(scale_local[0], _E4M3_MIN), 448.0),
                            0.0,
                        )
                        scale_value = T.cast(scale_local[0], T.float8_e4m3fn)
                        scale_byte = T.reinterpret(scale_value, T.uint8)
                        scale_local[0] = T.cast(scale_value, T.float32)
                    local_group = step * groups_per_step + lane // threads_per_scale
                    group = block_k * groups_per_block + local_group
                    if rows_per_block > 1 or group < groups:
                        sf_shared[local_group, row_in_block] = scale_byte
                if threads_per_scale > 1:
                    scale_local[0] = T.shfl_sync(
                        scale_local[0],
                        (warp_lane // threads_per_scale) * threads_per_scale,
                        32,
                    )
                inv_scale = T.if_then_else(
                    scale_local[0] > 0.0, 1.0 / scale_local[0], 0.0
                )
                for i in T.vectorized(values_per_thread):
                    if (rows_per_block > 1 and not has_k_tail) or base < cols:
                        q[batch_idx, row, base + i] = T.cast(
                            values_float[i] * inv_scale,
                            q_type,
                        )

            # Scale writers and readers may belong to different warps.
            T.sync_threads()
            for store_step in T.serial(store_steps):
                for i in T.vectorized(store_bytes):
                    out_index = (store_step * threads + tid) * store_bytes + i
                    out_row = out_index % rows_per_block
                    out_group = out_index // rows_per_block
                    group = block_k * groups_per_block + out_group
                    if (
                        out_index < rows_per_block * shared_groups
                        and out_group < groups_per_block
                        and ((rows_per_block > 1 and not has_k_tail) or group < groups)
                    ):
                        sf[batch_idx, block_m * rows_per_block + out_row, group] = (
                            sf_shared[out_group, out_row]
                        )

    name = (
        "tilelang_fp8_quantize_kernel_tme"
        if use_tme
        else "tilelang_fp8_quantize_kernel"
    )
    return kernel.with_attr("global_symbol", name)


@dataclass(frozen=True)
class _GroupedConfig:
    groups_per_block: int
    groups_in_thread: int


def _resolve_grouped_config(
    rows: int, cols: int, output_dtype: torch.dtype, sf_type: SfType
) -> _GroupedConfig | None:
    groups = _ceil_div(cols, _SF_VEC_SIZE)
    # The grouped mapping needs enough rows to amortize one-row CTAs.
    if rows >= 8192 and groups >= 64:
        return _GroupedConfig(64, 4)
    # At medium M, only the E4M3 output or scale cases were faster in the A/B
    # sweep. E5M2+E8M0 stays on the original row-coarsened kernel.
    if (
        rows >= 2048
        and groups >= 64
        and (output_dtype == torch.float8_e4m3fn or sf_type == "e4m3")
    ):
        return _GroupedConfig(64, 4)
    return None


@tilelang.jit(**_JIT_CONFIG)
def tilelang_fp8_quantize_grouped(
    sf_type: SfType,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    groups_per_block: int,
    groups_in_thread: int,
    has_tail: bool,
):
    values_per_thread = _VALUES_PER_THREAD
    batch, rows, cols = T.dynamic("batch, rows, cols")
    stride_b, stride_m = T.dynamic("stride_b, stride_m", "int64")
    assert groups_per_block % groups_in_thread == 0
    groups = T.dynamic("groups")
    threads_per_group = 32 // values_per_thread
    thread_groups = groups_per_block // groups_in_thread
    threads = thread_groups * threads_per_group
    q_type = T.float8_e4m3 if output_dtype == torch.float8_e4m3fn else T.float8_e5m2
    q_max = 448.0 if output_dtype == torch.float8_e4m3fn else 57344.0
    assert values_per_thread == 8
    assert threads in (32, 64, 128, 256, 512)

    @T.prim_func
    def kernel(
        x: T.StridedTensor[(batch, rows, cols), (stride_b, stride_m, 1), input_dtype],
        q: T.Tensor[(batch, rows, cols), q_type],
        sf: T.StridedTensor[(batch, rows, groups), (rows * groups, 1, rows), T.uint8],
    ):
        with T.Kernel(
            batch,
            rows,
            (groups + groups_per_block - 1) // groups_per_block,
            threads=threads,
        ) as (b, row, block_group):
            T.assume(cols % 32 == 0)
            T.assume(stride_b % 8 == 0)
            T.assume(stride_m % 8 == 0)
            tid = T.get_thread_binding()
            group_lane = tid % threads_per_group
            warp_lane = tid % 32
            input_values = T.alloc_local((values_per_thread,), input_dtype)
            values = T.alloc_local((groups_in_thread * values_per_thread,), T.float32)
            amax = T.alloc_local((1,), T.float32)
            scale = T.alloc_local((1,), T.float32)
            for part in T.unroll(groups_in_thread):
                group = (
                    block_group * groups_per_block
                    + part * thread_groups
                    + tid // threads_per_group
                )
                if has_tail:
                    safe_group = T.if_then_else(group < groups, group, 0)
                else:
                    safe_group = group
                base = safe_group * 32 + group_lane * values_per_thread
                for i in T.vectorized(values_per_thread):
                    input_values[i] = x[b, row, base + i]
                for i in T.vectorized(values_per_thread):
                    values[part * values_per_thread + i] = T.cast(
                        input_values[i], T.float32
                    )
            for part in T.unroll(groups_in_thread):
                local_group = part * thread_groups + tid // threads_per_group
                group = block_group * groups_per_block + local_group
                if has_tail:
                    safe_group = T.if_then_else(group < groups, group, 0)
                else:
                    safe_group = group
                base = safe_group * 32 + group_lane * values_per_thread
                v0 = T.max(T.abs(values[part * 8]), T.abs(values[part * 8 + 1]))
                v1 = T.max(T.abs(values[part * 8 + 2]), T.abs(values[part * 8 + 3]))
                v2 = T.max(T.abs(values[part * 8 + 4]), T.abs(values[part * 8 + 5]))
                v3 = T.max(T.abs(values[part * 8 + 6]), T.abs(values[part * 8 + 7]))
                amax[0] = T.max(T.max(v0, v1), T.max(v2, v3))
                amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 2, 32))
                amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 1, 32))
                scale[0] = 0.0
                if group_lane == 0:
                    scale[0] = amax[0] * (1.0 / q_max)
                    if sf_type == "e8m0":
                        code = _e8m0_code(scale[0])
                        scale_byte = T.cast(code, T.uint8)
                        scale[0] = _e8m0_float(code)
                    else:
                        scale[0] = T.if_then_else(
                            amax[0] > 0.0, T.min(T.max(scale[0], _E4M3_MIN), 448.0), 0.0
                        )
                        encoded = T.cast(scale[0], T.float8_e4m3fn)
                        scale_byte = T.reinterpret(encoded, T.uint8)
                        scale[0] = T.cast(encoded, T.float32)
                    if not has_tail or group < groups:
                        sf[b, row, group] = scale_byte
                scale[0] = T.shfl_sync(
                    scale[0], warp_lane // threads_per_group * threads_per_group, 32
                )
                inv_scale = T.if_then_else(scale[0] > 0.0, 1.0 / scale[0], 0.0)
                if has_tail:
                    if group < groups:
                        for i in T.vectorized(values_per_thread):
                            q[b, row, base + i] = T.cast(
                                values[part * values_per_thread + i] * inv_scale,
                                q_type,
                            )
                else:
                    for i in T.vectorized(values_per_thread):
                        q[b, row, base + i] = T.cast(
                            values[part * values_per_thread + i] * inv_scale,
                            q_type,
                        )

    return kernel.with_attr("global_symbol", "tilelang_fp8_quantize_grouped_kernel")


def _resolve_burst8_config(rows: int, cols: int) -> _GroupedConfig | None:
    groups = _ceil_div(cols, _SF_VEC_SIZE)
    # Explicit x8 casts lower to two BURST4 conversions for both output types.
    # Medium/small M keeps the existing mapping, whose casts also vectorize.
    if rows >= 8192 and groups >= 64:
        return _GroupedConfig(64, 4)
    return None


def _check_args(x, sf_vec_size, sf_type, output_dtype, *, fused=False):
    if x.dim() not in (2, 3):
        raise ValueError("x must have shape [M,K] or [B,M,K]")
    if x.stride(-1) != 1:
        raise ValueError("input must be contiguous in the K dimension")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("x must have dtype torch.float16 or torch.bfloat16")
    if x.device.type != "musa":
        raise ValueError("x must be on a MUSA device")
    if any(dim == 0 for dim in x.shape):
        raise ValueError("x dimensions must be non-empty")
    if sf_vec_size != _SF_VEC_SIZE:
        raise ValueError("only sf_vec_size=32 is supported")
    if x.shape[-1] % (4 if fused else 2):
        raise ValueError("the logical K dimension must be even")
    sf_type = "e8m0" if sf_type is None else sf_type
    if sf_type not in ("e8m0", "e4m3"):
        raise ValueError("sf_type must be e8m0 or e4m3")
    if output_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(
            "output_dtype must be torch.float8_e4m3fn or torch.float8_e5m2"
        )
    return sf_type


def _get_fp8_kernel(x, sf_type, output_dtype):
    _, rows, cols = x.shape
    groups = _ceil_div(cols, _SF_VEC_SIZE)
    # TME amortizes descriptor/barrier work on larger M. A single 16x512
    # input tile plus scales uses 16,640 B/CTA, without double buffering.
    if rows >= 2048 and _can_use_tme(x, rows_per_block=16):
        kernel = tilelang_fp8_quantize(
            sf_type,
            x.dtype,
            output_dtype,
            16,
            8,
            16,
            _TME_COLS,
            _scale_store_bytes(16),
            use_tme=True,
            has_k_tail=cols % _TME_COLS != 0,
        )
    elif cols % 16 or not _can_vectorize(x):
        kernel = tilelang_quantize_masked(x.dtype, output_dtype, sf_type)
    elif cols % 32:
        groups_per_block = 32 if rows <= 1024 else 64
        groups_in_thread = 1 if rows <= 1024 else (4 if rows >= 8192 else 2)
        kernel = tilelang_quantize_grouped(
            sf_type,
            x.dtype,
            output_dtype,
            groups_per_block,
            groups_in_thread,
            groups % groups_per_block != 0,
            has_partial_group=True,
        )
    else:
        burst8 = _resolve_burst8_config(rows, cols)
        grouped = _resolve_grouped_config(rows, cols, output_dtype, sf_type)
        if burst8 is not None:
            kernel = tilelang_quantize_grouped(
                sf_type,
                x.dtype,
                output_dtype,
                burst8.groups_per_block,
                burst8.groups_in_thread,
                groups % burst8.groups_per_block != 0,
            )
        elif grouped is not None:
            kernel = tilelang_fp8_quantize_grouped(
                sf_type,
                x.dtype,
                output_dtype,
                grouped.groups_per_block,
                grouped.groups_in_thread,
                groups % grouped.groups_per_block != 0,
            )
        else:
            config = _resolve_row_config(rows, cols)
            kernel = tilelang_fp8_quantize(
                sf_type,
                x.dtype,
                output_dtype,
                config.rows_per_block,
                config.threads_per_row,
                config.values_per_thread,
                config.cols_per_block,
                _scale_store_bytes(config.rows_per_block),
            )
    return kernel


def _fp8_quantize_impl(
    x: torch.Tensor,
    sf_vec_size: int = 32,
    sf_type: SfType | None = None,
    output_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate inputs and launch the TileLang backend for ``fp8_quantize``."""
    sf_type = _check_args(x, sf_vec_size, sf_type, output_dtype)
    input_is_3d = x.dim() == 3
    x3 = x if input_is_3d else x.unsqueeze(0)
    batch, rows, cols = x3.shape
    groups = _ceil_div(cols, _SF_VEC_SIZE)
    q = torch.empty((batch, rows, cols), dtype=output_dtype, device=x.device)
    sf = torch.empty_strided(
        (batch, rows, groups),
        (rows * groups, 1, rows),
        dtype=torch.uint8,
        device=x.device,
    )
    kernel = _get_fp8_kernel(x3, sf_type, output_dtype)
    kernel(x3, q, sf)
    return (q, sf) if input_is_3d else (q[0], sf[0])


def main() -> None:
    from mate.quantization import fp8_quantize

    import argparse
    import math

    from mate.testing.utils import bench_kineto

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--cols", type=int, default=16384)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--output-type", choices=("e4m3", "e5m2"), default="e4m3")
    parser.add_argument("--sf-type", choices=("e8m0", "e4m3"), default="e8m0")
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--show-source", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.rows, args.cols, args.rep) <= 0 or args.cols % 2:
        parser.error("positive batch/rows/cols/rep and even K are required")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    output_dtype = (
        torch.float8_e4m3fn if args.output_type == "e4m3" else torch.float8_e5m2
    )
    device = torch.device("musa")
    torch.manual_seed(0)
    print(
        f"torch={torch.__version__}, tilelang={tilelang.__version__}, device={torch.musa.get_device_name(0)}",
        flush=True,
    )

    x = torch.randn((args.batch_size, args.rows, args.cols), dtype=dtype, device=device)
    q, sf = fp8_quantize(
        x,
        sf_type=args.sf_type,
        output_dtype=output_dtype,
    )
    kernel = _get_fp8_kernel(x, args.sf_type, output_dtype)
    kernel_name = str(kernel.prim_func.attrs["global_symbol"])
    if args.show_source:
        print(kernel.get_kernel_source())
    torch.musa.synchronize()
    seconds = bench_kineto(
        lambda: kernel(x, q, sf),
        kernel_names=kernel_name,
        num_tests=args.rep,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError("Kineto did not record the FP8 quantization kernel")
    logical_bytes = x.numel() * x.element_size() + q.numel() + sf.numel()
    print(
        f"input={tuple(x.shape)}, q={tuple(q.shape)} {q.dtype}, sf={tuple(sf.shape)} {args.sf_type}, input_dtype={dtype}, rep={args.rep}, flush_l2=True",
        flush=True,
    )
    print(
        f"{kernel_name}: {logical_bytes / seconds / 1e9:.3f} logical GB/s (input read + q/sf write; preallocated buffers)",
        flush=True,
    )


if __name__ == "__main__":
    main()
