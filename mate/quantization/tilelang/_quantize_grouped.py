"""Vectorized quantization and activation, including unquantized BF16 output."""

import torch
import tilelang
import tilelang.language as T

from ._common import (
    _JIT_CONFIG,
    _e8m0_code,
    _e8m0_float,
    _e8m0_inv,
    _gated_activation,
    _gated_activation_vector,
)


@tilelang.jit(**_JIT_CONFIG)
def tilelang_quantize_grouped(
    sf_type: str,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    groups_per_block: int,
    groups_in_thread: int,
    has_tail: bool,
    fused: bool = False,
    has_partial_group: bool = False,
    sf_vec_size: int = 32,
    activation: str = "swiglu",
    swiglu_limit: float | None = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
    use_global_scale: bool = False,
    is_global_scale_inversed: bool = False,
    replicate_scale: bool | None = None,
):
    """Share vector loads and activation; BF16 bypasses the scale reduction."""
    is_bf16 = output_dtype == torch.bfloat16
    is_fp4 = output_dtype == torch.float4_e2m1fn_x2
    assert not is_bf16 or fused
    assert not is_fp4 or (fused and sf_vec_size == 32)
    assert not use_global_scale or is_fp4
    assert (sf_vec_size == 32 and sf_type != "fp32") or (
        sf_vec_size == 128 and sf_type == "fp32"
    )
    # Explicit vector stages preserve burst operations across the softcaps.
    # A 512-thread clamp tail CTA favors fewer vector temporaries.
    vector_activation = fused and (
        activation == "situ"
        or (
            swiglu_limit is not None
            and groups_in_thread > 1
            and groups_per_block // groups_in_thread * (sf_vec_size // 8) < 512
        )
    )
    k_first = fused and (activation == "situ" or swiglu_limit is not None)
    if (swiglu_limit is not None and sf_vec_size == 32 and groups_in_thread == 1) or (
        activation == "situ"
        and sf_vec_size == 128
        and groups_per_block == 16
        and groups_in_thread == 1
    ):
        k_first = False
    # Amax is already replicated across each group. Encoding the cheap E8M0
    # scale in every lane removes its leader predicate and broadcast dependency.
    if replicate_scale is None:
        replicate_scale = fused and sf_type == "e8m0" and not is_fp4
    integer_reciprocal = replicate_scale and activation == "situ" and not is_fp4
    scale_type = T.float32 if sf_type == "fp32" else T.uint8
    values_per_thread = 8
    batch, rows, cols = T.dynamic("batch, rows, cols")
    # For fusion, bind logical K from the FP8 output shape. This also lets
    # the vectorizer prove alignment of the up half without a dynamic divide.
    input_cols = cols * 2 if fused else cols
    stride_b, stride_m = T.dynamic("stride_b, stride_m", "int64")
    assert groups_per_block % groups_in_thread == 0
    compact_scale = (
        sf_type == "e8m0"
        and groups_in_thread == 4
        and not has_tail
        and not has_partial_group
    )
    groups = (
        T.ceildiv(cols, sf_vec_size)
        if is_bf16 or (is_fp4 and not compact_scale)
        else T.dynamic("groups")
    )
    sf_strides = (rows * groups, 1, rows)
    threads_per_group = sf_vec_size // values_per_thread
    thread_groups = groups_per_block // groups_in_thread
    threads = thread_groups * threads_per_group
    q_type = (
        T.bfloat16
        if is_bf16
        else T.float4_e2m1fn
        if is_fp4
        else (T.float8_e4m3 if output_dtype == torch.float8_e4m3fn else T.float8_e5m2)
    )
    q_max = (
        6.0 if is_fp4 else (448.0 if output_dtype == torch.float8_e4m3fn else 57344.0)
    )
    q_vec_type = (
        "bfloat16x8"
        if is_bf16
        else T.float4_e2m1fnx8
        if is_fp4
        else (
            T.float8_e4m3x8 if output_dtype == torch.float8_e4m3fn else T.float8_e5m2x8
        )
    )
    assert threads in (32, 64, 128, 256, 512)

    @T.macro
    def body(x, q, sf, global_scale, b, row, block_group, check_groups, check_values):
        tid = T.get_thread_binding()
        group_lane = tid % threads_per_group
        warp_lane = tid % 32
        input_values = T.alloc_local((values_per_thread,), input_dtype)
        up_values = T.alloc_local((values_per_thread,), input_dtype)
        values = T.alloc_local((groups_in_thread * values_per_thread,), T.float32)
        amax = T.alloc_local((1,), T.float32)
        scale = T.alloc_local((1,), T.float32)
        for part in T.unroll(groups_in_thread):
            group = (
                block_group * groups_per_block
                + part * thread_groups
                + tid // threads_per_group
            )
            if check_groups and not check_values:
                safe_group = T.if_then_else(group < groups, group, 0)
            else:
                safe_group = group
            base = safe_group * sf_vec_size + group_lane * values_per_thread
            if not check_values or base < cols:
                for i in T.vectorized(values_per_thread):
                    input_values[i] = x[b, row, base + i]
                if fused:
                    for i in T.vectorized(values_per_thread):
                        up_values[i] = x[b, row, cols + base + i]
            else:
                for i in T.vectorized(values_per_thread):
                    input_values[i] = 0
                    if fused:
                        up_values[i] = 0
            if vector_activation:
                _gated_activation_vector(
                    input_values,
                    up_values,
                    values,
                    part * 8,
                    activation,
                    swiglu_limit,
                    situ_beta,
                    situ_linear_beta,
                    is_bf16 and activation == "situ" and situ_beta == 4.0,
                )
            else:
                for i in T.vectorized(values_per_thread):
                    if fused:
                        values[part * values_per_thread + i] = _gated_activation(
                            T.cast(input_values[i], T.float32),
                            T.cast(up_values[i], T.float32),
                            activation,
                            swiglu_limit,
                            situ_beta,
                            situ_linear_beta,
                        )
                    else:
                        values[part * values_per_thread + i] = T.cast(
                            input_values[i], T.float32
                        )
        for part in T.unroll(groups_in_thread):
            local_group = part * thread_groups + tid // threads_per_group
            group = block_group * groups_per_block + local_group
            if check_groups and not check_values:
                safe_group = T.if_then_else(group < groups, group, 0)
            else:
                safe_group = group
            base = safe_group * sf_vec_size + group_lane * values_per_thread
            # BF16 specializes out all scale work, including shuffle reduction.
            if not is_bf16:
                v0 = T.max(T.abs(values[part * 8]), T.abs(values[part * 8 + 1]))
                v1 = T.max(T.abs(values[part * 8 + 2]), T.abs(values[part * 8 + 3]))
                v2 = T.max(T.abs(values[part * 8 + 4]), T.abs(values[part * 8 + 5]))
                v3 = T.max(T.abs(values[part * 8 + 6]), T.abs(values[part * 8 + 7]))
                amax[0] = T.max(T.max(v0, v1), T.max(v2, v3))
                if sf_vec_size == 128:
                    amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 8, 32))
                    amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 4, 32))
                amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 2, 32))
                amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 1, 32))
                scale[0] = 0.0
                if replicate_scale or group_lane == 0:
                    if sf_type == "fp32":
                        # Match RMSNorm's 1x128 scale calculation.
                        amax[0] = T.max(amax[0], 1e-4)
                        scale_byte = amax[0] * (1.0 / q_max)
                        scale[0] = q_max / amax[0]
                    else:
                        scale[0] = amax[0] * (1.0 / q_max)
                    if use_global_scale:
                        if is_global_scale_inversed:
                            scale[0] = scale[0] / global_scale[0]
                        else:
                            scale[0] = scale[0] * global_scale[0]
                    if sf_type == "e8m0":
                        code = _e8m0_code(scale[0])
                        scale_byte = T.cast(code, T.uint8)
                        scale[0] = (
                            _e8m0_inv(code) if integer_reciprocal else _e8m0_float(code)
                        )
                    elif sf_type == "e4m3":
                        if not is_fp4:
                            scale[0] = T.if_then_else(
                                amax[0] > 0.0,
                                T.min(T.max(scale[0], 2.0**-9), 448.0),
                                0.0,
                            )
                        encoded = T.cast(scale[0], T.float8_e4m3fn)
                        scale_byte = T.reinterpret(encoded, T.uint8)
                        scale[0] = T.cast(encoded, T.float32)
                    if group_lane == 0 and (not check_groups or group < groups):
                        # A CTA owns one row; its MN-major K groups are strided.
                        sf[b, row, group] = scale_byte
                if not replicate_scale:
                    scale[0] = T.shfl_sync(
                        scale[0], warp_lane // threads_per_group * threads_per_group, 32
                    )
                if sf_type == "fp32" or integer_reciprocal:
                    inv_scale = scale[0]
                else:
                    inv_scale = T.if_then_else(scale[0] > 0.0, 1.0 / scale[0], 0.0)
            if not (check_groups or check_values) or (
                group < groups and (not check_values or base < cols)
            ):
                if is_bf16:
                    q[b, row, T.Ramp(base, 1, 8)] = T.cast(
                        values[T.Ramp(part * values_per_thread, 1, 8)], q_vec_type
                    )
                else:
                    q[b, row, T.Ramp(base, 1, 8)] = T.cast(
                        values[T.Ramp(part * values_per_thread, 1, 8)]
                        * T.Broadcast(inv_scale, 8),
                        q_vec_type,
                    )

    @T.macro
    def launch(x, q, sf, global_scale):
        with T.Kernel(
            (groups + groups_per_block - 1) // groups_per_block if k_first else batch,
            rows,
            batch if k_first else (groups + groups_per_block - 1) // groups_per_block,
            threads=threads,
        ) as (block_x, row, block_z):
            b = block_z if k_first else block_x
            block_group = block_x if k_first else block_z
            T.assume(cols % (16 if has_partial_group else sf_vec_size) == 0)
            T.assume(stride_b % 8 == 0)
            T.assume(stride_m % 8 == 0)
            if has_tail or has_partial_group:
                # One uniform branch per CTA keeps lane predicates and safe
                # index selection out of every complete K block.
                if block_group < cols // (groups_per_block * sf_vec_size):
                    body(x, q, sf, global_scale, b, row, block_group, False, False)
                else:
                    body(
                        x,
                        q,
                        sf,
                        global_scale,
                        b,
                        row,
                        block_group,
                        has_tail,
                        has_partial_group,
                    )
            else:
                body(x, q, sf, global_scale, b, row, block_group, False, False)

    # Packed FP4 needs logical K explicitly; FP8/BF16 bind it from the output.
    if is_bf16:

        @T.prim_func
        def kernel(
            x: T.StridedTensor[
                (batch, rows, input_cols), (stride_b, stride_m, 1), input_dtype
            ],
            # The backend allocates the output as a contiguous tensor.
            q: T.Tensor[(batch, rows, cols), q_type],
        ):
            launch(x, q, x, x)
    elif is_fp4:

        @T.prim_func
        def kernel(
            x: T.StridedTensor[
                (batch, rows, input_cols), (stride_b, stride_m, 1), input_dtype
            ],
            q: T.Tensor[(batch, rows, cols), q_type],
            sf: T.StridedTensor[(batch, rows, groups), sf_strides, scale_type],
            global_scale: T.Tensor[(1,), T.float32],
            cols: T.int32,
        ):
            launch(x, q, sf, global_scale)
    else:

        @T.prim_func
        def kernel(
            x: T.StridedTensor[
                (batch, rows, input_cols), (stride_b, stride_m, 1), input_dtype
            ],
            q: T.Tensor[(batch, rows, cols), q_type],
            sf: T.StridedTensor[(batch, rows, groups), sf_strides, scale_type],
        ):
            launch(x, q, sf, x)

    return kernel.with_attr("global_symbol", "tilelang_quantize_grouped_kernel")
