"""MP32 three-stage MegaMoE prefill with E4M3 activations and packed E2M1 weights."""

from __future__ import annotations

import math
import torch

from mate.mega_moe import MegaMoESymmBuffer, _align
from mate.mate_runtime import get_compute_capability
from mate.jit.mega_moe_fp8_fp4 import PrefillConfig, _get_stage_module


class FP8FP4SymmBuffer(MegaMoESymmBuffer):
    """Full pools; UE8M0 input scales are row major, pool scales are K32 major.

    A K32 UE8M0 scale takes the same space as the MP31 K128 FP32 scale.
    Reuse the IPC/workspace layout, exposing byte views instead of FP32 views.
    """

    def __init__(
        self,
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        activation="swiglu",
    ):
        if get_compute_capability(as_int=True) != 32:
            raise ValueError("FP8 x FP4 MegaMoE requires MP32")
        if group.size() not in (1, 2, 4, 8):
            raise ValueError("MP32 prefill supports EP1/2/4/8")
        if (
            hidden <= 0
            or intermediate_hidden <= 0
            or hidden % 256
            or intermediate_hidden % 256
        ):
            raise ValueError("hidden and intermediate_hidden must be 256 aligned")
        if num_topk <= 0 or num_topk > min(num_experts, 32):
            raise ValueError("invalid top-k")
        if (
            num_experts <= 0
            or num_experts % group.size()
            or num_experts // group.size() > 1024
        ):
            raise ValueError("invalid expert count")
        if not 0 < num_max_tokens_per_rank <= 65536:
            raise ValueError("token capacity must be within [1,65536]")
        super().__init__(
            group,
            num_experts,
            _align(num_max_tokens_per_rank, 256),
            num_topk,
            hidden,
            intermediate_hidden,
            activation=activation,
            prefill_pool_capacity_ratio=1.0,
            prefill_l1_ring=False,
        )
        self.x_sf = self.x_sf.view(torch.uint8).reshape(
            self.num_max_tokens_per_rank, hidden // 32
        )
        self.l1_acts_sf = self.prefill_l1_acts_sf = self.l1_acts_sf.view(
            torch.uint8
        ).reshape(hidden // 32, self.prefill_num_max_pool_tokens)
        self.l2_acts_sf = self.prefill_l2_acts_sf = self.l2_acts_sf.view(
            torch.uint8
        ).reshape(intermediate_hidden // 32, self.prefill_num_max_pool_tokens)


def get_symm_buffer_for_fp8_fp4_mega_moe(
    group,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    activation="swiglu",
):
    return FP8FP4SymmBuffer(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        activation,
    )


def _logical_scales(scale, rows, k):
    if scale.dtype == torch.int32:
        if tuple(scale.shape) != (rows[0], rows[1], k // 128):
            raise ValueError("packed UE8M0 scales must have shape [E,N,K/128]")
        scale = scale.contiguous().view(torch.uint8)
    elif scale.dtype != torch.uint8:
        raise TypeError("scales must be UE8M0 uint8 or four packed bytes in int32")
    if tuple(scale.shape) != (rows[0], rows[1], k // 32):
        raise ValueError("logical UE8M0 scales must have shape [E,N,K/32]")
    return scale


def transform_weights_for_fp8_fp4_mega_moe(l1_weights, l2_weights):
    """Prepare logical gate/up W1 and logical W2 once, before inference.

    Weights: int8 [E,N,K/2], two E2M1 values per byte (low nibble first).
    Scales: uint8 [E,N,K/32], or int32 [E,N,K/128] packing four UE8M0 bytes.
    Returned scales: contiguous [E,K/32,N]. W1 and its scales use gate32/up32;
    W2 stays in logical intermediate order.
    """
    result = []
    for index, (weight, scale) in enumerate((l1_weights, l2_weights)):
        if weight.dtype != torch.int8 or weight.ndim != 3:
            raise TypeError("weights must be packed E2M1 in int8 [E,N,K/2]")
        e, n, packed_k = weight.shape
        if (
            e <= 0
            or n <= 0
            or packed_k <= 0
            or weight.device != scale.device
            or n % 256
            or packed_k % 128
        ):
            raise ValueError("invalid weight shape or scale device")
        scale = _logical_scales(scale, (e, n), packed_k * 2)
        if index == 0:
            if n % 512:
                raise ValueError("W1 must contain 256-aligned gate and up halves")
            weight = (
                weight.reshape(e, 2, n // 64, 32, packed_k)
                .transpose(1, 2)
                .reshape(e, n, packed_k)
            )
            scale = (
                scale.reshape(e, 2, n // 64, 32, packed_k // 16)
                .transpose(1, 2)
                .reshape(e, n, packed_k // 16)
            )
        result.append((weight.contiguous(), scale.transpose(1, 2).contiguous()))
    return tuple(result)


def fp8_fp4_mega_moe(
    out,
    l1_weights,
    l2_weights,
    sym_buffer,
    *,
    num_mps=40,
    fast_math=True,
    activation=None,
    activation_clamp=None,
    activation_beta=None,
    activation_linear_beta=None,
):
    """Execute prefill through the independent MP32 JIT and three kernels.

    Input x, x_sf, topk_idx and topk_weights are filled in the symmetric buffer.
    Route IDs must be unique per token; -1 disables a route. All ranks call this
    collectively with the same token count and configuration. Weight tuples
    must be prepared by transform_weights_for_fp8_fp4_mega_moe.
    """
    if not isinstance(sym_buffer, FP8FP4SymmBuffer):
        raise TypeError("use get_symm_buffer_for_fp8_fp4_mega_moe")
    sym = sym_buffer
    if out.dtype != torch.bfloat16 or out.ndim != 2 or not out.is_contiguous():
        raise ValueError("out must be contiguous BF16 [tokens,hidden]")
    tokens, h = out.shape
    if (
        out.device != sym.buffer.device
        or h != sym.hidden
        or not 0 < tokens <= sym.num_max_tokens_per_rank
    ):
        raise ValueError("invalid output device or shape")
    available_mps = torch.musa.get_device_properties(out.device).multi_processor_count
    if (
        not isinstance(num_mps, int)
        or num_mps <= 0
        or num_mps % 4
        or num_mps > available_mps
    ):
        raise ValueError(
            "num_mps must be a positive multiple of 4 within the device MP count"
        )
    activation = (activation or sym.activation).lower()
    if activation not in ("swiglu", "situ"):
        raise ValueError("activation must be swiglu or situ")
    if activation != "situ" and (
        activation_beta is not None or activation_linear_beta is not None
    ):
        raise ValueError("beta and linear_beta are SiTU parameters")
    if activation == "situ" and (
        activation_clamp is not None or activation_beta is None
    ):
        raise ValueError("SiTU requires beta and does not accept clamp")
    for value in (activation_beta, activation_linear_beta):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("activation beta must be finite and positive")
    if activation_clamp is not None and (
        not math.isfinite(activation_clamp) or activation_clamp < 0
    ):
        raise ValueError("activation clamp must be finite and nonnegative")
    experts = sym.num_experts // sym.group.size()
    for weights, n, k in (
        (l1_weights, 2 * sym.intermediate_hidden, sym.hidden),
        (l2_weights, sym.hidden, sym.intermediate_hidden),
    ):
        w, sf = weights
        if w.dtype != torch.int8 or sf.dtype != torch.uint8:
            raise TypeError("prepared weights/scales must be int8/UE8M0 uint8")
        if tuple(w.shape) != (experts, n, k // 2) or tuple(sf.shape) != (
            experts,
            k // 32,
            n,
        ):
            raise ValueError("invalid prepared weight/scale shape")
        if any(t.device != out.device or not t.is_contiguous() for t in (w, sf)):
            raise ValueError(
                "weights and scales must be contiguous on the output device"
            )
    config = PrefillConfig(
        sym.group.size(),
        sym.num_experts,
        sym.num_max_tokens_per_rank,
        sym.num_topk,
        sym.hidden,
        sym.intermediate_hidden,
        sym.prefill_num_max_pool_tokens,
        num_mps,
        fast_math,
        activation == "situ",
        activation_linear_beta is not None,
    )
    # Compile all modules before entering any cross-rank barrier.
    modules = [_get_stage_module(config, stage) for stage in (1, 2, 3)]
    name, module = modules[0]
    module.get_function(name)(
        sym.l1_acts,
        sym.l1_acts_sf,
        sym.l2_acts,
        sym.l2_acts_sf,
        *l1_weights,
        sym.handle.buffer_ptrs_tensor,
        sym.group.rank(),
        tokens,
        -1.0 if activation_clamp is None else activation_clamp,
        1.0 if activation_beta is None else activation_beta,
        -1.0 if activation_linear_beta is None else activation_linear_beta,
    )
    name, module = modules[1]
    module.get_function(name)(
        sym.l2_acts,
        sym.l2_acts_sf,
        *l2_weights,
        sym.handle.buffer_ptrs_tensor,
        sym.group.rank(),
    )
    name, module = modules[2]
    module.get_function(name)(
        out, sym.handle.buffer_ptrs_tensor, sym.group.rank(), tokens
    )
    return out


__all__ = [
    "fp8_fp4_mega_moe",
    "get_symm_buffer_for_fp8_fp4_mega_moe",
    "transform_weights_for_fp8_fp4_mega_moe",
    "FP8FP4SymmBuffer",
]
