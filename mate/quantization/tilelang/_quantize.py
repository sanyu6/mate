"""Shared quantization for partial groups and unaligned rows with unit K stride."""

import tilelang
import tilelang.language as T
import torch

from ._common import (
    _JIT_CONFIG,
    _e8m0_code,
    _e8m0_float,
    _gated_activation,
)


@tilelang.jit(**_JIT_CONFIG)
def tilelang_quantize_masked(
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    sf_type: str,
    fused: bool = False,
    use_global_scale: bool = False,
    is_global_scale_inversed: bool = False,
    sf_vec_size: int = 32,
    activation: str = "swiglu",
    swiglu_limit: float | None = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
):
    """Handle tails and unaligned rows; BF16 skips reduction and scale output."""
    batch, rows, input_cols = T.dynamic("batch, rows, input_cols")
    stride_b, stride_m = T.dynamic("stride_b, stride_m", "int64")
    cols = input_cols // 2 if fused else input_cols
    groups = T.ceildiv(cols, sf_vec_size)
    sf_strides = (rows * groups, 1, rows)
    threads_per_group = sf_vec_size // 8
    groups_per_block = 128 // threads_per_group
    scale_type = T.float32 if sf_type == "fp32" else T.uint8
    is_bf16 = output_dtype == torch.bfloat16
    is_fp4 = output_dtype == torch.uint8
    assert not is_bf16 or fused
    q_type = (
        T.bfloat16
        if is_bf16
        else T.uint8
        if is_fp4
        else T.float8_e4m3
        if output_dtype == torch.float8_e4m3fn
        else T.float8_e5m2
    )
    q_max = (
        6.0 if is_fp4 else (448.0 if output_dtype == torch.float8_e4m3fn else 57344.0)
    )

    @T.macro
    def body(x, q, sf, global_scale):
        with T.Kernel(
            batch, rows, T.ceildiv(groups, groups_per_block), threads=128
        ) as (
            b,
            row,
            block,
        ):
            tid = T.get_thread_binding()
            lane = tid % threads_per_group
            group = block * groups_per_block + tid // threads_per_group
            base = group * sf_vec_size + lane * 8
            values = T.alloc_local((8,), T.float32)
            amax = T.alloc_local((1,), T.float32)
            scale = T.alloc_local((1,), T.float32)
            for i in T.unroll(8):
                values[i] = 0.0
            # Keep one predicate around full lanes so address calculations
            # are shared across their loads. Only the last lane needs masks.
            if base + 8 <= cols:
                for i in T.unroll(8):
                    value = T.cast(x[b, row, base + i], T.float32)
                    if fused:
                        up = T.cast(x[b, row, cols + base + i], T.float32)
                        values[i] = _gated_activation(
                            value,
                            up,
                            activation,
                            swiglu_limit,
                            situ_beta,
                            situ_linear_beta,
                        )
                    else:
                        values[i] = value
            elif base < cols:
                for i in T.unroll(8):
                    if base + i < cols:
                        value = T.cast(x[b, row, base + i], T.float32)
                        if fused:
                            up = T.cast(x[b, row, cols + base + i], T.float32)
                            values[i] = _gated_activation(
                                value,
                                up,
                                activation,
                                swiglu_limit,
                                situ_beta,
                                situ_linear_beta,
                            )
                        else:
                            values[i] = value
            if is_bf16:
                for i in T.unroll(8):
                    if base + i < cols:
                        q[b, row, base + i] = T.cast(values[i], T.bfloat16)
            else:
                amax[0] = 0.0
                for i in T.unroll(8):
                    amax[0] = T.max(amax[0], T.abs(values[i]))
                if sf_vec_size == 128:
                    amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 8, 32))
                    amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 4, 32))
                amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 2, 32))
                amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 1, 32))
                scale[0] = 0.0
                if lane == 0:
                    if is_fp4 and not fused:
                        scale[0] = amax[0] / q_max
                    else:
                        scale[0] = amax[0] * (1.0 / q_max)
                    if use_global_scale:
                        if is_global_scale_inversed:
                            scale[0] = scale[0] / global_scale[0]
                        else:
                            scale[0] = scale[0] * global_scale[0]
                    if sf_type == "fp32":
                        amax[0] = T.max(amax[0], 1e-4)
                        scale_byte = amax[0] * (1.0 / q_max)
                        scale[0] = q_max / amax[0]
                    elif sf_type == "e8m0":
                        code = _e8m0_code(scale[0])
                        scale_byte = T.cast(code, T.uint8)
                        scale[0] = _e8m0_float(code)
                    else:
                        if not is_fp4:
                            scale[0] = T.if_then_else(
                                amax[0] > 0.0,
                                T.min(T.max(scale[0], 2.0**-9), 448.0),
                                0.0,
                            )
                        encoded = T.cast(scale[0], T.float8_e4m3fn)
                        scale_byte = T.reinterpret(encoded, T.uint8)
                        scale[0] = T.cast(encoded, T.float32)
                    if group < groups:
                        sf[b, row, group] = scale_byte
                scale[0] = T.shfl_sync(
                    scale[0], (tid % 32) // threads_per_group * threads_per_group, 32
                )
                if sf_type == "fp32":
                    inv_scale = scale[0]
                else:
                    inv_scale = T.if_then_else(scale[0] > 0.0, 1.0 / scale[0], 0.0)
                if base + 8 <= cols:
                    if is_fp4:
                        for pair in T.unroll(4):
                            packed = T.cast(
                                values[T.Ramp(pair * 2, 1, 2)]
                                * T.Broadcast(inv_scale, 2),
                                T.float4_e2m1fnx2,
                            )
                            q[b, row, base // 2 + pair] = T.reinterpret(packed, T.uint8)
                    else:
                        for i in T.unroll(8):
                            q[b, row, base + i] = T.cast(
                                T.min(T.max(values[i] * inv_scale, -q_max), q_max),
                                q_type,
                            )
                elif base < cols:
                    if is_fp4:
                        for pair in T.unroll(4):
                            if base + pair * 2 < cols:
                                packed = T.cast(
                                    values[T.Ramp(pair * 2, 1, 2)]
                                    * T.Broadcast(inv_scale, 2),
                                    T.float4_e2m1fnx2,
                                )
                                q[b, row, base // 2 + pair] = T.reinterpret(
                                    packed, T.uint8
                                )
                    else:
                        for i in T.unroll(8):
                            if base + i < cols:
                                q[b, row, base + i] = T.cast(
                                    T.min(T.max(values[i] * inv_scale, -q_max), q_max),
                                    q_type,
                                )

    if is_bf16:

        @T.prim_func
        def kernel(
            x: T.StridedTensor[
                (batch, rows, input_cols), (stride_b, stride_m, 1), input_dtype
            ],
            # The backend allocates the output as a contiguous tensor.
            q: T.Tensor[(batch, rows, cols), q_type],
        ):
            body(x, q, x, x)
    elif use_global_scale:

        @T.prim_func
        def kernel(
            x: T.StridedTensor[
                (batch, rows, input_cols), (stride_b, stride_m, 1), input_dtype
            ],
            q: T.Tensor[(batch, rows, cols // 2 if is_fp4 else cols), q_type],
            sf: T.StridedTensor[(batch, rows, groups), sf_strides, scale_type],
            global_scale: T.Tensor[(1,), T.float32],
        ):
            body(x, q, sf, global_scale)
    else:

        @T.prim_func
        def kernel(
            x: T.StridedTensor[
                (batch, rows, input_cols), (stride_b, stride_m, 1), input_dtype
            ],
            q: T.Tensor[(batch, rows, cols // 2 if is_fp4 else cols), q_type],
            sf: T.StridedTensor[(batch, rows, groups), sf_strides, scale_type],
        ):
            body(x, q, sf, x)

    return kernel.with_attr("global_symbol", "tilelang_quantize_masked_kernel")
