"""TileLang MUSA FP4 quantization with MN-major scale layout."""

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


SfType = Literal["e8m0", "e4m3"]
_FP4_MAX = 6.0
_TME_ROWS = 8


@dataclass(frozen=True)
class _RowConfig:
    rows_per_block: int
    threads_per_row: int
    values_per_thread: int
    cols_per_block: int


def _resolve_row_config(rows: int, cols: int, use_tme: bool = False) -> _RowConfig:
    """Choose the row mapping; scales are staged in MN-major shared memory."""
    if use_tme:
        # 8x512 input plus scales uses 8,320 B/CTA. Keep 16 values/lane
        # for both full tiles and K16 tails to limit registers.
        return _RowConfig(_TME_ROWS, 8, 16, _TME_COLS)
    if rows % 32 == 0 and cols % 4096 == 0:
        # Fewer values per lane reduce register pressure from dynamic addressing.
        return _RowConfig(8, 16, 16, 4096)
    # Bound shared memory for long rows; the last K tile may be partial.
    return _RowConfig(1, 256, 32, min(8192, max(32, 1 << (cols - 1).bit_length())))


def _check_args(x: torch.Tensor, sf_type: SfType | None) -> SfType:
    if x.dim() not in (2, 3):
        raise ValueError(f"x must be rank 2 or 3, got shape {tuple(x.shape)}")
    if x.stride(-1) != 1:
        raise ValueError("input must be contiguous in the K dimension")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("x must have dtype torch.float16 or torch.bfloat16")
    if x.device.type != "musa":
        raise ValueError(f"x must be on a MUSA device, got {x.device}")
    if sf_type is None:
        sf_type = "e8m0"
    if sf_type not in ("e8m0", "e4m3"):
        raise ValueError(f"unsupported sf_type: {sf_type!r}")
    if x.shape[-1] == 0:
        raise ValueError("x must have a non-empty K dimension")
    if x.shape[-1] % 2:
        raise ValueError("x must have an even K dimension")
    if x.shape[-2] == 0 or (x.dim() == 3 and x.shape[0] == 0):
        raise ValueError("x must have non-empty batch and M dimensions")
    return sf_type


@tilelang.jit(**_JIT_CONFIG)
def tilelang_fp4_quantize(
    sf_type: SfType,
    input_dtype: torch.dtype,
    use_global_scale: bool,
    is_global_scale_inversed: bool,
    rows_per_block: int,
    threads_per_row: int,
    values_per_thread: int,
    cols_per_block: int,
    store_bytes: int,
    use_tme: bool = False,
    has_k_tail: bool = False,
):
    """Stage MN-major scales, then cooperatively write vectors along M."""
    batch, rows, cols = T.dynamic("batch, rows, cols")
    stride_b, stride_m = T.dynamic("stride_b, stride_m", "int64")
    assert cols_per_block % 32 == 0
    assert values_per_thread in (8, 16, 32)
    assert not has_k_tail or use_tme
    if use_tme:
        # Keep the shared-memory budget used by the host dispatch exact.
        assert (rows_per_block, threads_per_row, cols_per_block) == (
            _TME_ROWS,
            8,
            _TME_COLS,
        )
        assert values_per_thread in (16, 32)
    groups = T.dynamic("groups")
    threads = rows_per_block * threads_per_row
    tile_width = threads_per_row * values_per_thread
    threads_per_scale = 32 // values_per_thread
    scale_blocks_per_step = tile_width // 32
    block_cols = cols_per_block
    steps = (block_cols + tile_width - 1) // tile_width
    groups_per_block = block_cols // 32
    shared_groups = steps * scale_blocks_per_step
    # Each scale group stores a vector of adjacent rows.
    store_steps = (rows_per_block * shared_groups + threads * store_bytes - 1) // (
        threads * store_bytes
    )
    scale_type = T.float8_e4m3fn if sf_type == "e4m3" else T.float8_e8m0fnu
    replicate_scale = use_tme and sf_type == "e8m0"

    @T.prim_func
    def kernel(
        x: T.StridedTensor[(batch, rows, cols), (stride_b, stride_m, 1), input_dtype],
        q: T.Tensor[(batch, rows, cols), T.float4_e2m1fn],
        sf: T.StridedTensor[(batch, rows, groups), (rows * groups, 1, rows), T.uint8],
        global_scale: T.Tensor[(1,), T.float32],
    ) -> None:
        with T.Kernel(
            (cols + block_cols - 1) // block_cols if use_tme else batch,
            rows // rows_per_block,
            batch if use_tme else (cols + block_cols - 1) // block_cols,
            threads=threads,
        ) as (
            block_x,
            block_m,
            block_z,
        ):
            # Visit adjacent K tiles before moving to another row tile.
            # This keeps the TME working set local on large matrices.
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
                x_shared = T.alloc_shared((rows_per_block, block_cols), input_dtype)
                # The descriptor and SIMT consumer must agree on this layout.
                T.annotate_layout(
                    {x_shared: tilelang.layout.make_linear_layout(x_shared)}
                )
                x_ready = T.alloc_barrier(arrive_count=threads)
                T.tma_copy(
                    x[
                        batch_idx,
                        block_m * rows_per_block : (block_m + 1) * rows_per_block,
                        block_k * block_cols : (block_k + 1) * block_cols,
                    ],
                    x_shared,
                    barrier=x_ready,
                )
                T.mbarrier_arrive(x_ready)
                T.mbarrier_wait_parity(x_ready, 0)

            for step in T.serial(steps):
                base = (
                    block_k * block_cols + step * tile_width + lane * values_per_thread
                )
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
                # Build a balanced expression at compile time; a serial max
                # chain lengthens the dependency path of each scale group.
                amax[0] = _max_abs(values_float, 0, values_per_thread)
                if threads_per_scale >= 4:
                    amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 2))
                if threads_per_scale >= 2:
                    amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 1))

                scale_local[0] = 0.0
                if replicate_scale or lane % threads_per_scale == 0:
                    # For finite FP16/BF16 input these forms encode identical
                    # scales without global_scale. The multiply avoids the
                    # compiler's reciprocal/refinement sequence for division.
                    if use_tme and not use_global_scale:
                        scale_local[0] = amax[0] * (1.0 / _FP4_MAX)
                    else:
                        scale_local[0] = amax[0] / _FP4_MAX
                    if use_global_scale:
                        if is_global_scale_inversed:
                            scale_local[0] = scale_local[0] / global_scale[0]
                        else:
                            scale_local[0] = scale_local[0] * global_scale[0]
                    if sf_type == "e8m0":
                        code = _e8m0_code(scale_local[0])
                        scale_byte = T.cast(code, T.uint8)
                        scale_local[0] = _e8m0_float(code)
                    else:
                        scale_value = T.cast(scale_local[0], scale_type)
                        scale_byte = T.reinterpret(scale_value, T.uint8)
                        scale_local[0] = T.cast(scale_value, T.float32)
                    local_group = (
                        step * scale_blocks_per_step + lane // threads_per_scale
                    )
                    group = block_k * groups_per_block + local_group
                    if lane % threads_per_scale == 0 and (
                        rows_per_block > 1 or group < groups
                    ):
                        sf_shared[local_group, row_in_block] = scale_byte
                if threads_per_scale > 1 and not replicate_scale:
                    scale_local[0] = T.shfl_sync(
                        scale_local[0],
                        (warp_lane // threads_per_scale) * threads_per_scale,
                        32,
                    )
                inv_scale = T.if_then_else(
                    scale_local[0] > 0.0, 1.0 / scale_local[0], 0.0
                )
                if has_k_tail:
                    # TME zero-fills the input tail. Store only complete 16-value
                    # chunks; packed FP4 rows are then at least 8-byte aligned.
                    for chunk in T.unroll(values_per_thread // 16):
                        if base + chunk * 16 < cols:
                            for i in T.vectorized(16):
                                q[batch_idx, row, base + chunk * 16 + i] = T.cast(
                                    values_float[chunk * 16 + i] * inv_scale,
                                    T.float4_e2m1fn,
                                )
                else:
                    for i in T.vectorized(values_per_thread):
                        if rows_per_block > 1 or base < cols:
                            q[batch_idx, row, base + i] = T.cast(
                                values_float[i] * inv_scale, T.float4_e2m1fn
                            )

            # Quantization's JIT configuration disables automatic barriers.
            # All producing warps must finish before other threads read sf.
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
        "tilelang_fp4_quantize_kernel_tme"
        if use_tme
        else "tilelang_fp4_quantize_kernel"
    )
    return kernel.with_attr("global_symbol", name)


def _fp4_quantize_impl(
    x: torch.Tensor,
    global_scale: torch.Tensor | None = None,
    sf_vec_size: int = 32,
    sf_type: SfType | None = None,
    is_global_scale_inversed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate inputs and launch the TileLang backend for ``fp4_quantize``."""
    sf_type = _check_args(x, sf_type)
    if sf_vec_size != 32:
        raise ValueError("only sf_vec_size=32 is supported")
    if global_scale is not None:
        if global_scale.shape != (1,) or global_scale.dtype != torch.float32:
            raise ValueError("global_scale must be a one-element float32 tensor")
        if global_scale.device != x.device:
            raise ValueError("global_scale must be on the same device as x")
        if not bool(torch.isfinite(global_scale).item()):
            raise ValueError("global_scale must be finite")
        if is_global_scale_inversed and not bool((global_scale > 0).item()):
            raise ValueError("global_scale must be positive")

    input_is_3d = x.dim() == 3
    batch = x.shape[0] if input_is_3d else 1
    rows, cols = x.shape[-2:]
    groups = _ceil_div(cols, sf_vec_size)
    x3 = x if input_is_3d else x.unsqueeze(0)
    q_storage = torch.empty(
        (batch, rows, cols // 2), dtype=torch.float4_e2m1fn_x2, device=x.device
    )
    sf = torch.empty_strided(
        (batch, rows, groups),
        (rows * groups, 1, rows),
        dtype=torch.uint8,
        device=x.device,
    )
    use_tme = _can_use_tme(x3)
    if use_tme or (cols % 32 == 0 and _can_vectorize(x3)):
        row_config = _resolve_row_config(rows, cols, use_tme)
        kernel = tilelang_fp4_quantize(
            sf_type,
            x.dtype,
            global_scale is not None,
            is_global_scale_inversed,
            row_config.rows_per_block,
            row_config.threads_per_row,
            row_config.values_per_thread,
            row_config.cols_per_block,
            _scale_store_bytes(row_config.rows_per_block),
            use_tme,
            has_k_tail=use_tme and cols % _TME_COLS != 0,
        )
        kernel(x3, q_storage, sf, global_scale)
    else:
        kernel = tilelang_quantize_masked(
            x.dtype,
            torch.uint8,
            sf_type,
            False,
            global_scale is not None,
            is_global_scale_inversed,
        )
        args = (x3, q_storage.view(torch.uint8), sf)
        if global_scale is not None:
            kernel(*args, global_scale)
        else:
            kernel(*args)
    if input_is_3d:
        return q_storage.view(torch.uint8), sf
    return q_storage[0].view(torch.uint8), sf[0]


def main() -> None:
    from mate.quantization import fp4_quantize

    import argparse

    from mate.testing.utils import bench_kineto

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--cols", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, help="Use a rank-3 input", default=1)
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--sf-type", choices=("e8m0", "e4m3"))
    parser.add_argument("--rep", type=int, default=30)
    args = parser.parse_args()
    if args.rows <= 0 or args.cols <= 0 or args.rep <= 0:
        parser.error("rows, cols, and rep must be positive")

    device = torch.device("musa")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if args.batch_size is not None:
        import math

        if args.batch_size <= 0 or not (0 < args.global_scale < math.inf):
            parser.error("batch-size and global-scale must be finite and positive")
        x = torch.randn(
            (args.batch_size, args.rows, args.cols), device=device, dtype=dtype
        )
        gs = torch.tensor([args.global_scale], dtype=torch.float32, device=device)
        run = lambda: fp4_quantize(x, global_scale=gs, sf_type=args.sf_type)
        name = "tilelang_fp4_quantize_kernel"
    else:
        x = torch.randn((args.rows, args.cols), device=device, dtype=dtype)
        run = lambda: fp4_quantize(x, sf_type=args.sf_type)
        name = "tilelang_fp4_quantize_kernel"
    q, sf = run()

    seconds = bench_kineto(
        run,
        kernel_names=name,
        num_tests=args.rep,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    logical_bytes = x.numel() * x.element_size() + q.numel() + sf.numel()
    gbps = logical_bytes / seconds / 1e9
    print(f"{name}: {gbps:.3f} logical GB/s (read + write)")


if __name__ == "__main__":
    main()
