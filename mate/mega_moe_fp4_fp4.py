"""Packed E2M1/UE8M0 contracts for the independent MP32 FP4 MoE path."""

from __future__ import annotations

import math
from typing import NamedTuple

from mate.jit.mega_moe_common import _normalize_activation

import torch

FP4_SCALE_BLOCK = 32
FP4_VALUES_PER_BYTE = 2
FP4_MAX_FINITE = 6.0
UE8M0_BIAS = 127
_FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """Pack uint8 E2M1 codes, first logical element in the low nibble."""
    if codes.dtype != torch.uint8 or codes.ndim < 1:
        raise TypeError("FP4 codes must be a uint8 tensor with a last dimension")
    if codes.shape[-1] % FP4_VALUES_PER_BYTE:
        raise ValueError("FP4 logical last dimension must be even")
    if bool((codes > 15).any()):
        raise ValueError("FP4 codes must be in [0,15]")
    return (codes[..., ::2] | (codes[..., 1::2] << 4)).contiguous()


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack low/high nibbles without dequantizing their values."""
    if packed.dtype != torch.uint8 or packed.ndim < 1:
        raise TypeError("packed FP4 must be uint8 with a last dimension")
    return torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)


def quantize_fp4_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU golden quantizer: K32 power-of-two scales and nearest-even E2M1.

    Returns logical row-major packed values [...,K/2] and scales [...,K/32].
    Zero blocks use scale=1 (byte 127). Finite inputs only. Scale exponents
    saturate to [-127,127]; byte 255 is reserved and never generated.
    This is an offline reference, not an inference device quantization kernel.
    """
    if x.device.type != "cpu":
        raise ValueError("quantize_fp4_reference requires a CPU tensor")
    if x.ndim < 1 or x.shape[-1] <= 0 or x.shape[-1] % FP4_SCALE_BLOCK:
        raise ValueError("logical K must be a positive multiple of 32")
    if not x.is_floating_point():
        raise TypeError("FP4 reference input must be floating point")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("FP4 reference requires finite input")
    # Float64 keeps power-of-two scale boundaries and tiny FP32 values exact.
    blocks = x.double().reshape(*x.shape[:-1], -1, FP4_SCALE_BLOCK)
    amax = blocks.abs().amax(dim=-1)
    exponent = torch.ceil(
        torch.log2(amax.clamp_min(torch.finfo(torch.float64).tiny) / FP4_MAX_FINITE)
    )
    exponent = torch.where(amax == 0, 0, exponent).clamp(-UE8M0_BIAS, UE8M0_BIAS)
    normalized = blocks * torch.exp2(-exponent).unsqueeze(-1)
    magnitude = normalized.abs().clamp_max(FP4_MAX_FINITE)
    values = torch.tensor(_FP4_MAGNITUDES, dtype=torch.float64)
    midpoints = (values[:-1] + values[1:]) * 0.5
    code = torch.bucketize(magnitude.contiguous(), midpoints)
    # bucketize chooses the lower code at a midpoint; odd lower codes round up.
    tie = (code < 7) & (magnitude == midpoints[code.clamp_max(6)]) & ((code & 1) != 0)
    code = code + tie.to(code.dtype)
    code = (code | (torch.signbit(normalized).to(code.dtype) << 3)).to(torch.uint8)
    return pack_fp4(code.reshape(x.shape)), (exponent + UE8M0_BIAS).to(torch.uint8)


def dequantize_fp4_reference(
    packed: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Decode CPU logical row-major packed FP4 and K32 UE8M0 to FP32."""
    if packed.device.type != "cpu" or scales.device.type != "cpu":
        raise ValueError("dequantize_fp4_reference requires CPU tensors")
    codes = unpack_fp4(packed)
    if scales.dtype != torch.uint8:
        raise TypeError("UE8M0 scales must be uint8")
    if codes.shape[-1] % FP4_SCALE_BLOCK or tuple(scales.shape) != (
        *codes.shape[:-1],
        codes.shape[-1] // FP4_SCALE_BLOCK,
    ):
        raise ValueError("logical scale shape must be [...,K/32]")
    if bool((scales == 255).any()):
        raise ValueError("UE8M0 byte 255 is reserved")
    values = torch.tensor(_FP4_MAGNITUDES, dtype=torch.float64)[(codes & 7).long()]
    values = torch.where((codes & 8) != 0, -values, values)
    scale = torch.exp2(scales.double() - UE8M0_BIAS).repeat_interleave(
        FP4_SCALE_BLOCK, -1
    )
    return (values * scale).float()


def _validate_logical_fp4_operand(packed: torch.Tensor, scales: torch.Tensor):
    if packed.dtype != torch.uint8 or packed.ndim != 3:
        raise TypeError("packed operand must be uint8 [groups,rows,K/2]")
    if scales.dtype != torch.uint8 or scales.ndim != 3:
        raise TypeError("scales must be uint8 [groups,rows,K/32]")
    groups, rows, packed_k = packed.shape
    if min(groups, rows, packed_k) <= 0 or packed_k * 2 % FP4_SCALE_BLOCK:
        raise ValueError("invalid FP4 operand dimensions")
    if scales.shape != (groups, rows, packed_k * 2 // FP4_SCALE_BLOCK):
        raise ValueError("logical operand/scale shape mismatch")
    if packed.device != scales.device:
        raise ValueError("packed operand and scales must share a device")
    if bool((scales == 255).any()):
        raise ValueError("UE8M0 byte 255 is reserved")


def prepare_fp4_operand(
    packed: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare ordinary SS GEMM storage; scales become [groups,K/32,rows].

    Mega MoE W2 requires transform_weights_for_fp4_fp4_mega_moe instead.
    """
    _validate_logical_fp4_operand(packed, scales)
    return packed.contiguous(), scales.transpose(-1, -2).contiguous()


def _validate_fp4_grouped_gemm(
    out: torch.Tensor,
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    *,
    num_mps: int | None = None,
) -> int:
    """Launch native SS FP4 GEMM for equal-length groups on MP32.

    ``a``/``b`` are prepared operands from :func:`prepare_fp4_operand`:
    uint8 [G,M,K/2] / [G,N,K/2], with scales [G,K/32,M/N].
    ``out`` is caller-allocated contiguous BF16 [G,M,N]. K must be 256
    aligned and N 8 aligned. This primitive uses one batched device launch;
    the separate Mega MoE entry uses the compact prefill pool.
    """
    if isinstance(a, _Stage2Weights) or isinstance(b, _Stage2Weights):
        raise ValueError("ordinary group GEMM requires prepare_fp4_operand storage")
    if out.device.type != "musa" or out.dtype != torch.bfloat16:
        raise ValueError("out must be a MUSA BF16 tensor")
    if out.ndim != 3 or not out.is_contiguous():
        raise ValueError("out must be contiguous [groups,M,N]")
    av, sa = a
    bv, sb = b
    for name, tensor in (("a", av), ("sa", sa), ("b", bv), ("sb", sb)):
        if tensor.dtype != torch.uint8 or tensor.ndim != 3:
            raise TypeError(f"{name} must be a prepared uint8 3D tensor")
        if tensor.device != out.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on the output device")
    groups, m, packed_k = av.shape
    n = bv.shape[1]
    k = packed_k * FP4_VALUES_PER_BYTE
    if min(groups, m, n, k) <= 0 or max(groups, m, n, k) > 2**31 - 1:
        raise ValueError("GEMM dimensions must be positive int32 values")
    if k % 256 or n % 8:
        raise ValueError("native FP4 SS requires K aligned to 256 and N to 8")
    if bv.shape != (groups, n, packed_k):
        raise ValueError("packed A/B group or K dimensions differ")
    if sa.shape != (groups, k // FP4_SCALE_BLOCK, m) or sb.shape != (
        groups,
        k // FP4_SCALE_BLOCK,
        n,
    ):
        raise ValueError("prepared scales must be contiguous [groups,K/32,rows]")
    if out.shape != (groups, m, n):
        raise ValueError("output shape must be [groups,M,N]")
    for tensor in (out, av, sa, bv, sb):
        if tensor.data_ptr() % 16:
            raise ValueError("native FP4 storage must be 16-byte aligned")
    available_mps = torch.musa.get_device_properties(out.device).multi_processor_count
    if num_mps is None:
        num_mps = available_mps
    if (
        isinstance(num_mps, bool)
        or not isinstance(num_mps, int)
        or not 0 < num_mps <= available_mps
    ):
        raise ValueError("num_mps must be within the device MP count")
    return num_mps


def fp4_fp4_grouped_gemm(
    out: torch.Tensor,
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    *,
    num_mps: int | None = None,
) -> torch.Tensor:
    """Launch native SS FP4 GEMM for equal-length groups on MP32.

    ``a``/``b`` are prepared operands from :func:`prepare_fp4_operand`:
    uint8 [G,M,K/2] / [G,N,K/2], with scales [G,K/32,M/N].
    ``out`` is caller-allocated contiguous BF16 [G,M,N]. K must be 256
    aligned and N 8 aligned. This primitive uses one batched device launch;
    the separate Mega MoE entry uses the compact prefill pool.
    """
    num_mps = _validate_fp4_grouped_gemm(out, a, b, num_mps=num_mps)
    av, sa = a
    bv, sb = b
    from mate.jit.mega_moe_fp4_fp4 import get_fp4_grouped_module

    name, module = get_fp4_grouped_module()
    module.get_function(name)(out, av, sa, bv, sb, num_mps)
    return out


from mate._fp4_moe_buffer import (  # noqa: E402,F401
    FP4FP4SymmBuffer,
    get_symm_buffer_for_fp4_fp4_mega_moe,
)


# Match Stage2's eight BF16 values per 16-byte store and four N lanes.
_STAGE2_ONLINE_STRIDE = 8
_STAGE2_N_LANES = 4


class _Stage2Weights(NamedTuple):
    """Opaque prepared pair: its SFB layout is specific to Mega MoE Stage2."""

    packed: torch.Tensor
    scales: torch.Tensor


def transform_weights_for_fp4_fp4_mega_moe(w1, w2):
    """Prepare logical packed FP4 weights once, before inference.

    W1 is [local_experts,2*intermediate,hidden/2], gate half then up half.
    W2 is [local_experts,hidden,intermediate/2]. Each input is a pair of
    packed E2M1 bytes and logical row-major UE8M0 scales on the same device.
    W1 retains logical gate/up order; Stage1 caches gate32/up32 internally.
    W2 data retains logical row order. Its SFB transpose and N permutation
    are combined in one copy, matching Stage2 TME online swizzle stride 8.
    Treat the returned W2 pair as opaque; retain its type when passing it
    to Stage2 or mega_moe. It cannot be used with ordinary group GEMM.
    Re-run this function after changing the original logical weights.
    """
    packed, scales = w2
    _validate_logical_fp4_operand(packed, scales)
    groups, rows, scale_groups = scales.shape
    block = _STAGE2_N_LANES * _STAGE2_ONLINE_STRIDE
    if rows % block:
        raise ValueError("Mega MoE W2 rows must be aligned to 32")
    # dst[..., p] = src[..., f(p)], f(p) = outer + inner*8/32 + inner*8%32.
    sf = scales.reshape(
        groups, rows // block, _STAGE2_N_LANES, _STAGE2_ONLINE_STRIDE, scale_groups
    )
    sf = sf.permute(0, 4, 1, 3, 2).contiguous().view(groups, scale_groups, rows)
    return prepare_fp4_operand(*w1), _Stage2Weights(packed.contiguous(), sf)


def _fp4_prefill_activation_options(
    activation, activation_clamp, activation_beta, activation_linear_beta
):
    """Validate the MP31-compatible gated-activation contract before IPC."""
    activation = _normalize_activation(activation)
    if activation == "swiglu":
        if activation_beta is not None or activation_linear_beta is not None:
            raise ValueError(
                "activation_beta/activation_linear_beta are only valid for SiTU"
            )
        return activation, 1.0, 1.0
    if activation_clamp is not None:
        raise ValueError("activation_clamp is only valid for SwiGLU; use None for SiTU")
    if activation_beta is None:
        raise ValueError("activation_beta is required for SiTU")
    beta = float(activation_beta)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("SiTU beta must be finite and positive")
    linear_beta = (
        1.0 if activation_linear_beta is None else float(activation_linear_beta)
    )
    if not math.isfinite(linear_beta) or linear_beta <= 0:
        raise ValueError("SiTU linear_beta must be finite and positive")
    return activation, beta, linear_beta


def fp4_fp4_stage1(
    sym_buffer,
    w1,
    tokens,
    *,
    num_mps=None,
    activation_clamp=10.0,
    mode="full",
    activation="swiglu",
    activation_beta=None,
    activation_linear_beta=None,
):
    """Fused dispatch / SS FP4 GEMM / gated activation-FP4 on device counts.

    ``mode='compute'`` reuses the most recent full/dispatch routing and L1
    payload; it excludes planning/dispatch. Dispatch mode plans and fills L1
    without GEMM. Full mode includes IPC, setup and dispatch in the same resident kernel.
    W1 retains the public logical gate-half/up-half prepared contract; a
    gate32/up32 device copy is cached once and invalidated on in-place updates.
    Compile all diagnostic modes before collective benchmarking.
    For Kimi K3 SiTU, set activation="situ", activation_clamp=None,
    activation_beta=4.0 and activation_linear_beta=25.0. None linear beta
    disables the up softcap. Gate/up round to BF16 before FP32 activation;
    the activation rounds once to BF16 before block-scaled FP4 quantization.
    """
    import math
    from mate.jit.mega_moe_fp4_fp4 import get_fp4_prefill_stage_module

    activation, beta, linear_beta = _fp4_prefill_activation_options(
        activation, activation_clamp, activation_beta, activation_linear_beta
    )

    if not isinstance(sym_buffer, FP4FP4SymmBuffer) or sym_buffer.group is None:
        raise ValueError("sym_buffer must be live")
    s = sym_buffer
    if getattr(s, "_decode_plan", None) is not None:
        raise ValueError("finish decode Stage2 before reusing the shared data pool")
    if (
        not isinstance(tokens, int)
        or isinstance(tokens, bool)
        or not 0 < tokens <= s.config.tokens
    ):
        raise ValueError("tokens must be within symmetric capacity")
    if num_mps is None:
        num_mps = s.common_mps
    if (
        not isinstance(num_mps, int)
        or isinstance(num_mps, bool)
        or not 0 < num_mps <= s.common_mps
    ):
        raise ValueError("num_mps exceeds the common EP MP count")
    clamp = float("inf") if activation_clamp is None else float(activation_clamp)
    if math.isnan(clamp) or clamp < 0:
        raise ValueError("invalid activation clamp")
    groups = s.config.experts // s.config.ranks
    expected = (
        (groups, 2 * s.config.intermediate, s.config.hidden // 2),
        (groups, s.config.hidden // 32, 2 * s.config.intermediate),
    )
    if not isinstance(w1, tuple) or len(w1) != 2:
        raise ValueError("W1 must be a prepared tuple")
    for value, shape in zip(w1, expected):
        if (
            value.shape != shape
            or value.dtype != torch.uint8
            or value.device != s.buffer.device
            or not value.is_contiguous()
            or value.data_ptr() % 16
        ):
            raise ValueError("W1 must use aligned prepared FP4 storage")
    if mode == "compute" and (
        getattr(s, "_stage1_plan_tokens", None) != tokens
        or getattr(s, "_stage1_plan_mps", None) != num_mps
    ):
        raise ValueError(
            "compute mode requires a full/dispatch plan with the same tokens and MP count"
        )
    if num_mps % 4:
        raise ValueError(
            "fused Stage1 requires an MP count divisible by four owner CTAs"
        )
    name, module = get_fp4_prefill_stage_module(
        s.config,
        1,
        mode=mode,
        num_mps=num_mps,
        activation=activation,
        use_situ_linear=activation_linear_beta is not None,
    )
    s.allocate_workspace()
    weights = s.fused_stage1_weights(w1)
    module.get_function(name)(
        s.buffer,
        s.ptrs,
        s.stage1_l1,
        s.stage1_l1_sf,
        s.stage2_l2,
        s.stage2_l2_sf,
        s.counts,
        s.metadata,
        s.status,
        *weights,
        s.rank,
        tokens,
        num_mps,
        clamp,
        *((beta, linear_beta) if activation == "situ" else ()),
    )
    if mode != "compute":
        s._stage1_plan_tokens = tokens
        s._stage1_plan_mps = num_mps


def fp4_fp4_stage2(s, w2, *, num_mps=None):
    """Native SS FC2 and original prefill scatter/barrier/clear sequence.

    W2 must come from transform_weights_for_fp4_fp4_mega_moe.
    Call after fused Stage1; consumes its compact L2 pool and route workspace.
    All ranks call in the same order. The workspace is retired after this call.
    """
    from mate.jit.mega_moe_fp4_fp4 import get_fp4_prefill_stage_module

    if not isinstance(s, FP4FP4SymmBuffer) or s.group is None:
        raise ValueError("s must be a live FP4 symmetric buffer")
    if getattr(s, "_stage1_plan_tokens", None) is None:
        raise ValueError("Stage2 requires a live fused Stage1 plan")
    if num_mps is None:
        num_mps = s.common_mps
    if (
        isinstance(num_mps, bool)
        or not isinstance(num_mps, int)
        or not 0 < num_mps <= s.common_mps
    ):
        raise ValueError("num_mps exceeds the common EP MP count")
    c = s.config
    expected = (
        (c.experts // c.ranks, c.hidden, c.intermediate // 2),
        (c.experts // c.ranks, c.intermediate // 32, c.hidden),
    )
    if not isinstance(w2, _Stage2Weights):
        raise ValueError("W2 must come from transform_weights_for_fp4_fp4_mega_moe")
    for value, shape in zip(w2, expected):
        if (
            value.shape != shape
            or value.dtype != torch.uint8
            or value.device != s.buffer.device
            or not value.is_contiguous()
            or value.data_ptr() % 16
        ):
            raise ValueError("W2 must use aligned prepared FP4 storage")
    name, module = get_fp4_prefill_stage_module(c, 2, num_mps=num_mps)
    module.get_function(name)(
        s.buffer, s.ptrs, s.stage2_l2, s.stage2_l2_sf, s.counts, *w2, s.rank, num_mps
    )
    s._stage1_plan_tokens = None


def fp4_fp4_mega_moe(
    out,
    w1,
    w2,
    sym_buffer,
    *,
    num_mps=None,
    activation_clamp=10.0,
    activation="swiglu",
    activation_beta=None,
    activation_linear_beta=None,
):
    """FP4 prefill with fused Stage1/Stage2 and BF16 combine.

    All ranks call with the same active token count and distinct expert IDs per
    token (-1 disables a slot). Inputs are prequantized E2M1/UE8M0; W1/W2 use
    transform_weights_for_fp4_fp4_mega_moe storage. BF16 output is caller allocated.
    Stage1 setup/dispatch/load/MMA/SwiGLU-FP4 execute in one resident kernel;
    no full FC1 tensor is written. Stage2 uses the original prefill compact-pool scheduler, vector scatter and
    two-barrier workspace retirement; Stage3 performs weighted combine.
    Compilation/weight interleave/workspace setup precede the first IPC barrier.
    The default MP count is the minimum of participating devices. Synchronize
    then inspect sym_buffer.status (0 is OK); destroy the buffer collectively.
    SiTU uses activation="situ", activation_clamp=None and a positive
    activation_beta; activation_linear_beta optionally softcaps up (K3:4/25).
    Weight preparation and the unified workspace are shared with SwiGLU.
    """
    import math

    activation, beta, linear_beta = _fp4_prefill_activation_options(
        activation, activation_clamp, activation_beta, activation_linear_beta
    )

    if not isinstance(sym_buffer, FP4FP4SymmBuffer) or sym_buffer.group is None:
        raise ValueError("sym_buffer must be a live FP4FP4SymmBuffer")
    c = sym_buffer.config
    if getattr(sym_buffer, "_decode_plan", None) is not None:
        raise ValueError("finish decode Stage2 before reusing the shared data pool")
    if (
        out.device != sym_buffer.buffer.device
        or out.dtype != torch.bfloat16
        or out.ndim != 2
        or not out.is_contiguous()
        or out.data_ptr() % 16
        or out.shape[1] != c.hidden
        or not 0 < out.shape[0] <= c.tokens
    ):
        raise ValueError(
            "out must be contiguous BF16 [active_tokens,hidden] on the buffer device"
        )
    if num_mps is None:
        num_mps = sym_buffer.common_mps
    if (
        isinstance(num_mps, bool)
        or not isinstance(num_mps, int)
        or not 0 < num_mps <= sym_buffer.common_mps
    ):
        raise ValueError("num_mps must be within the common EP MP count")
    clamp = float("inf") if activation_clamp is None else float(activation_clamp)
    if math.isnan(clamp) or clamp < 0:
        raise ValueError("activation_clamp must be nonnegative or None")
    if not isinstance(w2, _Stage2Weights):
        raise ValueError("W2 must come from transform_weights_for_fp4_fp4_mega_moe")
    groups = c.experts // c.ranks
    for name, operand, n, k in (
        ("w1", w1, 2 * c.intermediate, c.hidden),
        ("w2", w2, c.hidden, c.intermediate),
    ):
        if not isinstance(operand, tuple) or len(operand) != 2:
            raise ValueError(f"{name} must be a prepared (packed,scales) tuple")
        for value, shape in zip(
            operand, ((groups, n, k // 2), (groups, k // FP4_SCALE_BLOCK, n))
        ):
            if (
                value.device != out.device
                or value.dtype != torch.uint8
                or value.shape != shape
                or not value.is_contiguous()
                or value.data_ptr() % 16
            ):
                raise ValueError(
                    f"{name} must use aligned contiguous prepared FP4 storage"
                )
    from mate.jit.mega_moe_fp4_fp4 import get_fp4_prefill_stage_module

    # Resolve all stages before the first IPC operation.
    functions = []
    for stage in (1, 2, 3):
        name, module = get_fp4_prefill_stage_module(
            c,
            stage,
            **({"num_mps": num_mps} if stage < 3 else {}),
            **(
                {
                    "activation": activation,
                    "use_situ_linear": activation_linear_beta is not None,
                }
                if stage == 1
                else {}
            ),
        )
        functions.append(module.get_function(name))
    sym_buffer.allocate_workspace()
    s = sym_buffer
    tokens = out.shape[0]
    packed_w1 = s.fused_stage1_weights(w1)
    functions[0](
        s.buffer,
        s.ptrs,
        s.stage1_l1,
        s.stage1_l1_sf,
        s.stage2_l2,
        s.stage2_l2_sf,
        s.counts,
        s.metadata,
        s.status,
        *packed_w1,
        s.rank,
        tokens,
        num_mps,
        clamp,
        *((beta, linear_beta) if activation == "situ" else ()),
    )
    functions[1](
        s.buffer,
        s.ptrs,
        s.stage2_l2,
        s.stage2_l2_sf,
        s.counts,
        *w2,
        s.rank,
        num_mps,
    )
    s._stage1_plan_tokens = None
    functions[2](out, s.buffer, tokens, num_mps)
    return out


def _decode_settings(s, tokens, num_mps, activation_clamp):
    import math

    if not isinstance(s, FP4FP4SymmBuffer) or s.group is None:
        raise ValueError("sym_buffer must be live")
    if not getattr(s, "decode_enabled", False):
        raise ValueError("create the unified buffer with enable_decode=True")
    if (
        isinstance(tokens, bool)
        or not isinstance(tokens, int)
        or not 0 < tokens <= min(s.config.tokens, 128)
    ):
        raise ValueError("M32 decode supports 1..128 active tokens per rank")
    if num_mps is None:
        num_mps = s.common_mps
    if (
        isinstance(num_mps, bool)
        or not isinstance(num_mps, int)
        or not 2 <= num_mps <= s.common_mps
        or num_mps % 2
    ):
        raise ValueError(
            "decode requires an even resident MP count within the common device count"
        )
    clamp = float("inf") if activation_clamp is None else float(activation_clamp)
    if math.isnan(clamp) or clamp < 0:
        raise ValueError("activation_clamp must be nonnegative or None")
    return num_mps, clamp


def _decode_weight(s, weights, stage):
    c = s.config
    n, k = (2 * c.intermediate, c.hidden) if stage == 1 else (c.hidden, c.intermediate)
    if not isinstance(weights, tuple) or len(weights) != 2:
        raise ValueError("decode weights must be a prepared pair")
    if stage == 2 and not isinstance(weights, _Stage2Weights):
        raise ValueError("W2 must come from transform_weights_for_fp4_fp4_mega_moe")
    for value, shape in zip(
        weights, ((c.experts // c.ranks, n, k // 2), (c.experts // c.ranks, k // 32, n))
    ):
        if (
            value.shape != shape
            or value.dtype != torch.uint8
            or value.device != s.buffer.device
            or not value.is_contiguous()
            or value.data_ptr() % 16
        ):
            raise ValueError("decode weights must use aligned prepared FP4 storage")


def fp4_fp4_decode_stage1(
    s,
    w1,
    tokens,
    *,
    num_mps=None,
    activation_clamp=10.0,
    activation="swiglu",
    activation_beta=None,
    activation_linear_beta=None,
    swap_ab=False,
    pack_b=False,
    sglang_numerics=False,
    line_padded_scales=False,
    double_buffered=False,
):
    """Experimental M32 dispatch/FC1/gated activation-FP4 in one resident kernel.

    Uses the same input tensors, W1 preparation and local activation pools as
    prefill. M32 control/arrival words occupy a separate region of the unified
    IPC buffer. Finish Stage2 before switching to prefill or another decode.
    Validate target configurations with the device regression suite before deployment.
    """
    from mate.jit.mega_moe_fp4_fp4_decode import get_fp4_decode_module

    activation, beta, linear_beta = _fp4_prefill_activation_options(
        activation, activation_clamp, activation_beta, activation_linear_beta
    )
    num_mps, clamp = _decode_settings(s, tokens, num_mps, activation_clamp)
    if (
        getattr(s, "_decode_plan", None) is not None
        or getattr(s, "_stage1_plan_tokens", None) is not None
    ):
        raise ValueError(
            "finish the previous Stage2 before reusing the shared data pool"
        )
    _decode_weight(s, w1, 1)
    name, module = get_fp4_decode_module(
        s.config,
        1,
        num_mps=num_mps,
        swap_ab=swap_ab,
        pack_b=pack_b,
        sglang_numerics=sglang_numerics,
        double_buffered=double_buffered,
        line_padded_scales=line_padded_scales,
        **(
            {
                "activation": activation,
                "use_situ_linear": activation_linear_beta is not None,
            }
            if activation == "situ"
            else {}
        ),
    )
    s.allocate_workspace()
    packed_w1 = (
        s.swap_decode_weights(w1, 1)
        if swap_ab
        else s.packed_decode_weights(w1, 1)
        if pack_b
        else s.fused_stage1_weights(w1)
    )
    if line_padded_scales:
        if getattr(s, "_decode_line_padded_sf", None) is None:
            s._decode_line_padded_sf = torch.empty(
                (s.config.hidden // 32, s.stage1_pool_tokens * 4),
                dtype=torch.uint8,
                device=s.buffer.device,
            )
        input_sf = s._decode_line_padded_sf
    else:
        input_sf = s.stage1_l1_sf
    module.get_function(name)(
        s.buffer,
        s.ptrs,
        s.stage1_l1,
        input_sf,
        s.stage2_l2,
        s.stage2_l2_sf,
        s.status,
        *packed_w1,
        s.rank,
        tokens,
        num_mps,
        clamp,
        *((beta, linear_beta) if activation == "situ" else ()),
    )
    s._decode_plan = (tokens, num_mps)
    s._decode_sglang_numerics = sglang_numerics


def fp4_fp4_decode_stage2(
    out,
    s,
    w2,
    *,
    num_mps=None,
    swap_ab=False,
    pack_b=False,
    sglang_numerics=False,
    double_buffered=False,
):
    """Experimental M32 FC2/scatter/weighted combine in the second kernel."""
    from mate.jit.mega_moe_fp4_fp4_decode import get_fp4_decode_module

    if not isinstance(s, FP4FP4SymmBuffer) or s.group is None:
        raise ValueError("sym_buffer must be live")
    num_mps, _ = _decode_settings(
        s, out.shape[0] if out.ndim == 2 else 0, num_mps, None
    )
    if (
        out.device.type != "musa"
        or out.device != s.buffer.device
        or out.dtype != torch.bfloat16
        or out.ndim != 2
        or out.shape[1] != s.config.hidden
        or not out.is_contiguous()
        or out.data_ptr() % 16
    ):
        raise ValueError("out must be contiguous MUSA BF16 [tokens,hidden]")
    if getattr(s, "_decode_plan", None) != (out.shape[0], num_mps):
        raise ValueError("Stage2 requires a matching M32 Stage1 token/MP plan")
    if bool(getattr(s, "_decode_sglang_numerics", False)) != sglang_numerics:
        raise ValueError("Stage2 must match Stage1 numerical mode")
    _decode_weight(s, w2, 2)
    name, module = get_fp4_decode_module(
        s.config,
        2,
        num_mps=num_mps,
        swap_ab=swap_ab,
        pack_b=pack_b,
        sglang_numerics=sglang_numerics,
        double_buffered=double_buffered,
    )
    module.get_function(name)(
        out,
        s.buffer,
        s.ptrs,
        s.stage2_l2,
        s.stage2_l2_sf,
        *(
            s.swap_decode_weights(w2, 2)
            if swap_ab
            else s.packed_decode_weights(w2, 2)
            if pack_b
            else w2
        ),
        s.rank,
        out.shape[0],
        num_mps,
    )
    s._decode_plan = None
    s._decode_sglang_numerics = False
    return out


def fp4_fp4_mega_moe_decode(
    out,
    w1,
    w2,
    sym_buffer,
    *,
    num_mps=None,
    activation_clamp=10.0,
    activation="swiglu",
    activation_beta=None,
    activation_linear_beta=None,
    swap_ab=False,
    pack_b=False,
    sglang_numerics=False,
    line_padded_scales=False,
    double_buffered=False,
):
    """Experimental MP32 FP4 x FP4 two-launch decode, with M=32 tiles.

    All ranks use 1..128 active tokens, the same resident even MP count and
    the same call order. Each token has distinct expert IDs; -1 disables a
    route. Create the common buffer with enable_decode=True and prepare W1/W2
    with transform_weights_for_fp4_fp4_mega_moe. Prefill remains an explicit,
    separate entry; do not use the shared data pools concurrently.

    SiTU uses activation="situ", activation_clamp=None, activation_beta=4.0
    and activation_linear_beta=25.0 for Kimi K3. None linear beta disables
    the up softcap. Gate/up and final rounding match the prefill contract.

    ``swap_ab=True`` opts into a decode-only weight-A/activation-B native
    FP4 mainloop. Features occupy MMA M and tokens N. Weight packing is
    cached per buffer and invalidated by tensor mutation; prepare/warm up
    before timing. The default and prefill paths retain their layouts.

    ``pack_b=True`` instead retains the M32 token-A mainloop and its online
    register epilogues, packing only weight B into contiguous K256 tiles.
    It is mutually exclusive with swap_ab and currently supports SwiGLU.

    This is an experimental entry. The device regression suite covers exact
    small-shape EP1/4/8 replay; validate the intended production configuration
    before deployment.
    """
    from mate.jit.mega_moe_fp4_fp4_decode import get_fp4_decode_module

    activation, _, _ = _fp4_prefill_activation_options(
        activation, activation_clamp, activation_beta, activation_linear_beta
    )
    num_mps, _ = _decode_settings(
        sym_buffer, out.shape[0] if out.ndim == 2 else 0, num_mps, activation_clamp
    )
    s = sym_buffer
    if (
        out.device.type != "musa"
        or out.device != s.buffer.device
        or out.dtype != torch.bfloat16
        or out.ndim != 2
        or out.shape[1] != s.config.hidden
        or not out.is_contiguous()
        or out.data_ptr() % 16
    ):
        raise ValueError("out must be contiguous MUSA BF16 [tokens,hidden]")
    _decode_weight(s, w1, 1)
    _decode_weight(s, w2, 2)
    if swap_ab:
        if activation != "swiglu":
            raise ValueError("swap-AB currently supports SwiGLU only")
        s.swap_decode_weights(w1, 1)
        s.swap_decode_weights(w2, 2)
    if pack_b:
        s.packed_decode_weights(w1, 1)
        s.packed_decode_weights(w2, 2)
    # Build both stages before the first IPC launch. A rank compiling Stage2
    # must never leave peers spinning inside the Stage2 collective barrier.
    for stage in (1, 2):
        get_fp4_decode_module(
            s.config,
            stage,
            num_mps=num_mps,
            swap_ab=swap_ab,
            pack_b=pack_b,
            sglang_numerics=sglang_numerics,
            double_buffered=double_buffered,
            line_padded_scales=line_padded_scales if stage == 1 else False,
            **(
                {
                    "activation": activation,
                    "use_situ_linear": activation_linear_beta is not None,
                }
                if stage == 1 and activation == "situ"
                else {}
            ),
        )
    fp4_fp4_decode_stage1(
        s,
        w1,
        out.shape[0],
        num_mps=num_mps,
        activation_clamp=activation_clamp,
        activation=activation,
        activation_beta=activation_beta,
        activation_linear_beta=activation_linear_beta,
        swap_ab=swap_ab,
        pack_b=pack_b,
        sglang_numerics=sglang_numerics,
        double_buffered=double_buffered,
        line_padded_scales=line_padded_scales,
    )
    return fp4_fp4_decode_stage2(
        out,
        s,
        w2,
        num_mps=num_mps,
        swap_ab=swap_ab,
        pack_b=pack_b,
        sglang_numerics=sglang_numerics,
        double_buffered=double_buffered,
    )
