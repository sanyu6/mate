import tilelang
import tilelang.language as T
import torch


_JIT_CONFIG = {
    "pass_configs": {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
    "compile_flags": [
        "-O3",
        "-fno-signed-zeros",
        "-mllvm",
        "-mtgpu-if-convert=1",
        "-mllvm",
        "-misched=mtgpu-max-ilp",
        "-mllvm",
        "-mtgpu-tiny-offset-hint=1",
        "-mllvm",
        "-misched-recompute-slotindex=1",
    ],
}


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _can_vectorize(x: torch.Tensor, width: int = 8) -> bool:
    """Prove alignment for each vector, including sliced and padded rows."""
    return (
        x.stride(-1) == 1
        and x.data_ptr() % (width * x.element_size()) == 0
        and all(stride % width == 0 for stride in x.stride()[:-1])
    )


def _scale_store_bytes(rows_per_block: int) -> int:
    return 16 if rows_per_block % 16 == 0 else (4 if rows_per_block % 4 == 0 else 1)


def _e8m0_code(value):
    bits = T.reinterpret(value, T.uint32)
    exponent = T.min(T.cast((bits + T.uint32(0x007FFFFF)) >> 23, T.int32), 254)
    # One unsigned range check covers 2^-127 < value <= +inf. Values below
    # that interval wrap around on subtraction; negative values and NaNs
    # lie above it. Preserve code 0 for those inputs and 254 for +inf.
    return T.if_then_else(
        bits - T.uint32(0x00400001) <= T.uint32(0x7F3FFFFF), exponent, 0
    )


def _e8m0_float(code):
    return T.reinterpret(
        T.if_then_else(code == 0, T.int32(0x00400000), code << 23), T.float32
    )


def _e8m0_inv(code):
    """Decode the exact reciprocal of an E8M0 scale code in [0, 254]."""
    # Code 0 is 2^-127, so its reciprocal is the normal value 2^127.
    # Code 254 needs the subnormal reciprocal 2^-127 instead of exponent 0.
    return T.reinterpret(
        T.if_then_else(code == 254, T.int32(0x00400000), (254 - code) << 23),
        T.float32,
    )


def _silu(gate):
    return gate * (1.0 / (T.exp2(-gate * 1.4426950408889634) + 1.0))


def _situ_tanh(value):
    """Express the softcap through sigmoid within the FP32 activation budget."""
    return 2.0 * (1.0 / (1.0 + T.exp2(value * -2.8853900817779268))) - 1.0


def _gated_activation(gate, up, activation, swiglu_limit, situ_beta, situ_linear_beta):
    """Evaluate the model activation in FP32 before fused quantization."""
    if activation == "situ":
        # Kimi K3: the sigmoid takes the original gate, before tanh softcapping.
        gate_out = (
            situ_beta
            * _situ_tanh(gate * (1.0 / situ_beta))
            * (1.0 / (T.exp2(-gate * 1.4426950408889634) + 1.0))
        )
        if situ_linear_beta is not None:
            up = situ_linear_beta * _situ_tanh(up * (1.0 / situ_linear_beta))
        return gate_out * up
    if swiglu_limit is not None:
        gate = T.min(gate, swiglu_limit)
        up = T.min(T.max(up, -swiglu_limit), swiglu_limit)
    return _silu(gate) * up


_TME_COLS = 512


def _can_use_tme(x: torch.Tensor, rows_per_block: int = 8) -> bool:
    """Check tile geometry and alignment; the target is assumed to support TME."""
    return (
        x.shape[-2] % rows_per_block == 0
        and x.shape[-1] >= _TME_COLS
        and x.shape[-1] % 16 == 0
        # TME setup costs more than direct loads for small, narrow matrices.
        and (x.shape[-1] >= 4096 or x.shape[0] * x.shape[-2] >= 256)
        and _can_vectorize(x)
        # Broadcast outer dimensions need direct loads instead of a descriptor.
        and all(stride > 0 for stride in x.stride()[:-1])
    )


def _max_abs(values, start: int, count: int):
    """Expand a balanced max tree while constructing the kernel."""
    if count == 1:
        return T.abs(values[start])
    half = count // 2
    return T.max(_max_abs(values, start, half), _max_abs(values, start + half, half))


@T.macro
def _situ_tanh_vector(input_values, output):
    for chunk in T.unroll(2):
        index = T.Ramp(chunk * 4, 1, 4)
        output[index] = _situ_tanh(input_values[index])


@T.macro
def _gated_activation_vector(
    gate_input,
    up_input,
    output,
    offset,
    activation,
    swiglu_limit,
    situ_beta,
    situ_linear_beta,
    reuse_situ_exp=False,
):
    gate = T.alloc_local((8,), T.float32)
    up = T.alloc_local((8,), T.float32)
    sigmoid = T.alloc_local((8,), T.float32)
    soft_gate = T.alloc_local((8,), T.float32)
    soft_up = T.alloc_local((8,), T.float32)
    exp_gate = T.alloc_local((8,), T.float32)
    for chunk in T.unroll(2):
        index = T.Ramp(chunk * 4, 1, 4)
        gate[index] = T.cast(gate_input[index], T.float32x4)
        up[index] = T.cast(up_input[index], T.float32x4)
        if swiglu_limit is not None:
            gate[index] = T.min(gate[index], swiglu_limit)
            up[index] = T.min(T.max(up[index], -swiglu_limit), swiglu_limit)
        if reuse_situ_exp:
            # For beta=4, exp(-gate) = exp(-gate/2)^2. BF16 output can
            # share this exponential between sigmoid and gate softcapping.
            exp_gate[index] = T.exp2(gate[index] * -0.7213475204444817)
            sigmoid[index] = 1.0 / (exp_gate[index] * exp_gate[index] + 1.0)
            soft_gate[index] = 1.0 / (exp_gate[index] + 1.0)
            if situ_linear_beta is not None:
                soft_up[index] = 1.0 / (
                    T.exp2(up[index] * (-2.8853900817779268 / situ_linear_beta)) + 1.0
                )
        else:
            sigmoid[index] = 1.0 / (T.exp2(gate[index] * -1.4426950408889634) + 1.0)
            if activation == "situ":
                gate[index] = gate[index] * (1.0 / situ_beta)
                if situ_linear_beta is not None:
                    up[index] = up[index] * (1.0 / situ_linear_beta)
    if activation == "situ":
        if reuse_situ_exp:
            for i in T.unroll(8):
                gate[i] = T.ieee_fmaf(soft_gate[i], 2.0 * situ_beta, -situ_beta)
                if situ_linear_beta is not None:
                    up[i] = T.ieee_fmaf(
                        soft_up[i], 2.0 * situ_linear_beta, -situ_linear_beta
                    )
        else:
            _situ_tanh_vector(gate, soft_gate)
            if situ_linear_beta is not None:
                _situ_tanh_vector(up, soft_up)
    for chunk in T.unroll(2):
        index = T.Ramp(chunk * 4, 1, 4)
        if activation == "situ" and not reuse_situ_exp:
            gate[index] = situ_beta * soft_gate[index]
            if situ_linear_beta is not None:
                up[index] = situ_linear_beta * soft_up[index]
        output[T.Ramp(offset + chunk * 4, 1, 4)] = (gate[index] * sigmoid[index]) * up[
            index
        ]
