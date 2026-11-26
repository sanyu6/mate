"""Independent CPU references and deterministic inputs for FP4 prefill tests."""

import torch
from mate.mega_moe_fp4_fp4 import (
    pack_fp4,
    prepare_fp4_operand,
    dequantize_fp4_reference,
    quantize_fp4_reference,
)


def expert_weight(expert, n, k, layer):
    rng = torch.Generator().manual_seed(20000 + layer * 1000 + expert)
    values = pack_fp4(torch.randint(0, 16, (n, k), generator=rng, dtype=torch.uint8))
    scales = torch.randint(120, 123, (n, k // 32), generator=rng, dtype=torch.uint8)
    return values, scales


def rank_weights(rank, groups, n, k, layer, *, logical_only=False):
    parts = [expert_weight(rank * groups + e, n, k, layer) for e in range(groups)]
    logical = (torch.stack([x[0] for x in parts]), torch.stack([x[1] for x in parts]))
    if logical_only:
        return logical
    return logical, tuple(t.to("musa") for t in prepare_fp4_operand(*logical))


def rank_inputs(rank, tokens, hidden):
    rng = torch.Generator().manual_seed(1000 + rank)
    x = pack_fp4(
        torch.randint(0, 16, (tokens, hidden), generator=rng, dtype=torch.uint8)
    )
    sf = torch.randint(
        125, 129, (tokens, hidden // 32), generator=rng, dtype=torch.uint8
    )
    return x, sf


def routes_for(rank, tokens, topk, experts, pattern):
    if pattern == "all_disabled":
        return torch.full((tokens, topk), -1, dtype=torch.int64)
    if pattern == "skew":
        return torch.arange(topk).expand(tokens, topk).clone()
    ids = (torch.arange(tokens)[:, None] + torch.arange(topk)[None, :] + rank) % experts
    if pattern == "disabled":
        ids[::3, 0] = -1
    return ids


def activation_reference(
    first,
    clamp,
    *,
    activation="swiglu",
    activation_beta=4.0,
    activation_linear_beta=None,
):
    gate, up = first.float().chunk(2, -1)
    if activation == "situ":
        if clamp is not None:
            raise ValueError("SiTU does not use a SwiGLU clamp")
        gate = (
            activation_beta * torch.tanh(gate / activation_beta) * torch.sigmoid(gate)
        )
        if activation_linear_beta is not None:
            up = activation_linear_beta * torch.tanh(up / activation_linear_beta)
        return (gate * up).bfloat16().float()
    if activation != "swiglu":
        raise ValueError("unsupported activation")
    if clamp is not None:
        gate = gate.clamp(max=clamp).bfloat16().float()
        up = up.clamp(-clamp, clamp)
    return (up * (gate / (1 + torch.exp(-gate))).bfloat16().float()).bfloat16().float()


def reference_partials(
    x,
    sf,
    ids,
    intermediate,
    clamp=10.0,
    *,
    activation="swiglu",
    activation_beta=4.0,
    activation_linear_beta=None,
):
    """CPU reference from original inputs/weights, independent of dispatched storage."""
    hidden = x.shape[-1] * 2
    a = dequantize_fp4_reference(x, sf)
    result = torch.zeros((*ids.shape, hidden), dtype=torch.bfloat16)
    for expert in ids.unique().tolist():
        if expert < 0:
            continue
        tokens, slots = torch.where(ids == expert)
        b1 = dequantize_fp4_reference(
            *expert_weight(expert, 2 * intermediate, hidden, 1)
        )
        first = (a[tokens] @ b1.T).bfloat16()
        act = activation_reference(
            first,
            clamp,
            activation=activation,
            activation_beta=activation_beta,
            activation_linear_beta=activation_linear_beta,
        )
        aq = dequantize_fp4_reference(*quantize_fp4_reference(act))
        b2 = dequantize_fp4_reference(*expert_weight(expert, hidden, intermediate, 2))
        result[tokens, slots] = (aq @ b2.T).bfloat16()
    return result


def w4a4_partials_reference(x, sf, ids, route_weights, intermediate, clamp=10.0):
    """Packed W4A4 semantics: FP32 SiLU and route factor before FP4 quant."""
    hidden = x.shape[-1] * 2
    a = dequantize_fp4_reference(x, sf)
    result = torch.zeros((*ids.shape, hidden), dtype=torch.bfloat16)
    for expert in ids.unique().tolist():
        if expert < 0:
            continue
        tokens, slots = torch.where(ids == expert)
        w1 = dequantize_fp4_reference(
            *expert_weight(expert, 2 * intermediate, hidden, 1)
        )
        first = (a[tokens] @ w1.T).bfloat16().float()
        gate, up = first.chunk(2, -1)
        if clamp is not None:
            gate = gate.clamp(max=clamp)
            up = up.clamp(-clamp, clamp)
        sigmoid = 1.0 / (torch.exp2(gate * -1.4426950408889634) + 1.0)
        activation = (gate * sigmoid * up) * route_weights[tokens, slots, None]
        q, qs = quantize_fp4_reference(activation)
        w2 = dequantize_fp4_reference(*expert_weight(expert, hidden, intermediate, 2))
        result[tokens, slots] = (dequantize_fp4_reference(q, qs) @ w2.T).bfloat16()
    return result


def w4a4_combine_reference(partials, ids, experts_per_rank, world):
    """Round each EP owner's partial before rank-ordered FP32 reduction."""
    total = torch.zeros(partials.shape[0], partials.shape[-1])
    for owner in range(world):
        local = torch.zeros_like(total)
        for slot in range(ids.shape[1]):
            mask = (ids[:, slot] >= 0) & (ids[:, slot] // experts_per_rank == owner)
            local[mask] += partials[mask, slot].float()
        total += local.bfloat16().float()
    return total.bfloat16()
