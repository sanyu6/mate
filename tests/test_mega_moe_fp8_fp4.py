"""Public FP8/FP4 MegaMoE lifecycle against a sampled dequantized CPU reference.

Run: torchrun --standalone --nproc_per_node=8 tests/test_mega_moe_fp8_fp4.py
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch_musa  # noqa: F401
from mate.mega_moe_fp8_fp4 import (
    fp8_fp4_mega_moe,
    get_symm_buffer_for_fp8_fp4_mega_moe,
    transform_weights_for_fp8_fp4_mega_moe,
)


def _dequant_weight(w, sf):
    w = w.view(torch.uint8)
    code = torch.stack((w & 15, w >> 4), dim=-1).flatten(-2).long()
    table = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]
    )
    return table[code] * torch.exp2(sf.float() - 127).repeat_interleave(32, -1)


def _weights(e, n, k, seed):
    rng = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (e, n, k), generator=rng, dtype=torch.uint8)
    packed = (codes[..., ::2] | (codes[..., 1::2] << 4)).view(torch.int8)
    sf = torch.randint(120, 123, (e, n, k // 32), generator=rng, dtype=torch.uint8)
    return packed, sf


def _reference_weight_payload(logical1, logical2, rank, needed):
    experts = logical1[0].shape[0]
    # Object collectives pickle each tensor view with its entire backing storage.
    # Clone selected experts so the payload contains only the requested weights.
    return {
        rank * experts + expert: (
            (logical1[0][expert].clone(), logical1[1][expert].clone()),
            (logical2[0][expert].clone(), logical2[1][expert].clone()),
        )
        for expert in range(experts)
        if rank * experts + expert in needed
    }


def _reference(x, sf, routes, scores, w1, w2, activation, clamp, beta, linear_beta):
    x = x.float() * torch.exp2(sf.float() - 127).repeat_interleave(32, -1)
    output = torch.zeros_like(x)
    experts = w1.keys() if isinstance(w1, dict) else range(w1.shape[0])
    for expert in experts:
        rows, slots = torch.where(routes == expert)
        if not rows.numel():
            continue
        first = _dequant_weight(*w1[expert]) if isinstance(w1, dict) else w1[expert]
        second = _dequant_weight(*w2[expert]) if isinstance(w2, dict) else w2[expert]
        c = (x[rows] @ first.T).bfloat16().float()
        gate, up = c.chunk(2, dim=-1)
        if activation == "situ":
            g = beta * torch.tanh(gate / beta) * (0.5 * (1 + torch.tanh(gate * 0.5)))
            if linear_beta is not None:
                up = linear_beta * torch.tanh(up / linear_beta)
            y = (up * g).bfloat16().float()
        else:
            if clamp is not None:
                gate = gate.clamp(max=clamp).bfloat16().float()
                up = up.clamp(-clamp, clamp)
            y = (
                (up * (gate * torch.sigmoid(gate)).bfloat16().float())
                .bfloat16()
                .float()
            )
        shape = y.shape
        blocks = y.reshape(shape[0], -1, 32)
        scale = torch.exp2(
            torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-10) / 448.0))
        )
        quant = (
            (blocks / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        )
        y = (quant * scale[..., None]).reshape(shape)
        partial = (y @ second.T).bfloat16().float()
        output.index_add_(0, rows, partial * scores[rows, slots, None])
    return output.bfloat16()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--intermediate-hidden", type=int, default=256)
    parser.add_argument("--experts-per-rank", type=int, default=4)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--activation", choices=("swiglu", "situ"), default="swiglu")
    parser.add_argument("--clamp", type=float)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--linear-beta", type=float)
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--num-mps", type=int, default=40)
    parser.add_argument(
        "--result-json", help="save per-rank reference and replay results"
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--routing", choices=("uniform", "skew"), default="uniform")
    parser.add_argument("--all-routes-valid", action="store_true")
    parser.add_argument(
        "--reference-samples",
        type=int,
        default=8,
        help="reference tokens per rank (default: 8); 0 checks every token",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.musa.set_device(local)
    dist.init_process_group("mccl", device_id=torch.device("musa", local))
    group = dist.group.WORLD
    s, h, i, e, t = (
        args.tokens,
        args.hidden,
        args.intermediate_hidden,
        args.experts_per_rank,
        args.topk,
    )
    sym = get_symm_buffer_for_fp8_fp4_mega_moe(
        group, e * world, s, t, h, i, activation=args.activation
    )
    print(f"PHASE rank={rank} buffer_ready", flush=True)
    rng = torch.Generator().manual_seed(1000 + rank)
    x = torch.randn((s, h), generator=rng).to(torch.float8_e4m3fn)
    sf = torch.randint(125, 129, (s, h // 32), dtype=torch.uint8, generator=rng)
    # Unique top-k, skew, empty experts, and disabled routes are exercised.
    routes = (torch.arange(s)[:, None] + torch.arange(t)[None, :] + rank) % (e * world)
    if args.routing == "skew":
        routes %= min(e * world, t * 2)
    if not args.all_routes_valid:
        routes[::7, -1] = -1
    scores = torch.rand((s, t), generator=rng)
    scores /= scores.sum(-1, keepdim=True)
    sym.x[:s].copy_(x)
    sym.x_sf[:s].copy_(sf)
    sym.topk_idx[:s].copy_(routes)
    sym.topk_weights[:s].copy_(scores)
    logical1 = _weights(e, 2 * i, h, 2000 + rank)
    logical2 = _weights(e, h, i, 3000 + rank)
    print(f"PHASE rank={rank} logical_weights_ready", flush=True)
    w1, w2 = transform_weights_for_fp8_fp4_mega_moe(
        tuple(v.to("musa") for v in logical1), tuple(v.to("musa") for v in logical2)
    )
    # Compare actual physical preprocessing to explicit logical indices.
    idx = torch.stack(
        (torch.arange(i).reshape(-1, 32), torch.arange(i, 2 * i).reshape(-1, 32)), dim=1
    ).flatten()
    assert torch.equal(w1[0].cpu(), logical1[0][:, idx, :])
    assert torch.equal(w1[1].cpu(), logical1[1][:, idx, :].transpose(1, 2))
    print(f"PHASE rank={rank} transformed_weights_ready", flush=True)
    sample = (
        torch.arange(s)
        if args.reference_samples <= 0
        else torch.linspace(0, s - 1, min(s, args.reference_samples)).long()
    )
    if not args.benchmark:
        needed = [None] * world
        dist.all_gather_object(
            needed,
            set(routes[sample].flatten().tolist()) - {-1},
            group=group,
        )
        global_needed = set.union(*needed)
        payload = _reference_weight_payload(logical1, logical2, rank, global_needed)
        gathered = [None] * world
        dist.all_gather_object(gathered, payload, group=group)
        first_weights = {
            key: value[0] for item in gathered for key, value in item.items()
        }
        second_weights = {
            key: value[1] for item in gathered for key, value in item.items()
        }
        print(f"PHASE rank={rank} reference_inputs_ready", flush=True)
        reference = _reference(
            x[sample],
            sf[sample],
            routes[sample],
            scores[sample],
            first_weights,
            second_weights,
            args.activation,
            args.clamp,
            args.beta,
            args.linear_beta,
        )
    print(f"PHASE rank={rank} reference_ready", flush=True)
    output = torch.empty((s, h), dtype=torch.bfloat16, device="musa")
    options = dict(
        num_mps=args.num_mps,
        activation=args.activation,
        activation_clamp=args.clamp,
        activation_beta=args.beta if args.activation == "situ" else None,
        activation_linear_beta=args.linear_beta,
    )
    fp8_fp4_mega_moe(output, w1, w2, sym, **options)
    torch.musa.synchronize()
    dist.barrier(group=group)
    first = output.cpu()
    if not args.benchmark:
        diff = (first[sample].float() - reference.float()).abs()
        rel_l2 = float(diff.norm() / reference.float().norm().clamp_min(1e-12))
        local_metrics = {
            "rank": rank,
            "reference_tokens": int(sample.numel()),
            "reference_elements": int(diff.numel()),
            "rel_l2": rel_l2,
            "max_abs": float(diff.max()),
            "finite": bool(torch.isfinite(first).all()),
        }
        # BF16 gate rounding can move a few values across FP8 rounding boundaries.
        assert rel_l2 < 0.01, (rank, rel_l2, float(diff.max()))
        assert torch.isfinite(first).all()
        print(
            f"REFERENCE_PASS rank={rank} rel_l2={rel_l2:.6g} max_abs={float(diff.max()):.6g}",
            flush=True,
        )
    for replay in range(args.replays):
        active = s if replay % 2 == 0 else max(1, s // 2)
        fp8_fp4_mega_moe(output[:active], w1, w2, sym, **options)
        torch.musa.synchronize()
        dist.barrier(group=group)
        assert torch.equal(output[:active].cpu(), first[:active]), (
            f"rank {rank} replay {replay} changed"
        )
    if args.benchmark:
        start, end = (
            torch.musa.Event(enable_timing=True),
            torch.musa.Event(enable_timing=True),
        )
        start.record()
        for _ in range(10):
            fp8_fp4_mega_moe(output, w1, w2, sym, **options)
        end.record()
        end.synchronize()
        print(
            f"E2E_MS rank={rank} tokens={s} ms={start.elapsed_time(end) / 10:.6f}",
            flush=True,
        )
    if rank == 0:
        print(
            f"PUBLIC_REPLAY_PASS EP={world} tokens={s} activation={args.activation} mps={args.num_mps}",
            flush=True,
        )
    if args.result_json and not args.benchmark:
        metrics = [None] * world
        dist.all_gather_object(metrics, local_metrics, group=group)
        if rank == 0:
            Path(args.result_json).write_text(
                json.dumps(
                    {
                        "shape": vars(args),
                        "world_size": world,
                        "reference": "independent dequantized CPU FP32 matmul with BF16/FP8 rounding",
                        "criterion": "per-rank final BF16 output rel_l2 < 0.01",
                        "all_tokens_checked": int(sample.numel()) == s,
                        "replays_passed": args.replays,
                        "rank_metrics": metrics,
                    },
                    indent=2,
                )
                + "\n"
            )
    sym.destroy()
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--clamp", "2"],
        ["--activation", "situ", "--beta", "2", "--linear-beta", "1"],
    ],
)
def test_public_fp8_fp4_prefill(options):
    if not torch.musa.is_available():
        pytest.skip("requires MP32")
    props = torch.musa.get_device_properties(0)
    if (props.major, props.minor) != (3, 2):
        pytest.skip("requires MP32")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=1",
            str(Path(__file__).resolve()),
            "--experts-per-rank",
            "32",
            *options,
        ],
        check=True,
        timeout=300,
    )


def test_fp8_fp4_weight_preparation():
    logical1 = _weights(2, 512, 256, 123)
    logical2 = _weights(2, 256, 256, 456)
    a, b = transform_weights_for_fp8_fp4_mega_moe(logical1, logical2)
    index = torch.stack(
        (torch.arange(256).reshape(-1, 32), torch.arange(256, 512).reshape(-1, 32)),
        dim=1,
    ).flatten()
    assert torch.equal(a[0], logical1[0][:, index, :])
    assert torch.equal(a[1], logical1[1][:, index, :].transpose(1, 2))
    assert torch.equal(b[0], logical2[0])
    assert torch.equal(b[1], logical2[1].transpose(1, 2))
    packed1 = (logical1[0], logical1[1].view(torch.int32))
    packed2 = (logical2[0], logical2[1].view(torch.int32))
    packed = transform_weights_for_fp8_fp4_mega_moe(packed1, packed2)
    for normal, alternate in zip((a, b), packed):
        for x, y in zip(normal, alternate):
            assert torch.equal(x, y)


def test_reference_payload_serialization():
    logical1 = _weights(8, 512, 256, 123)
    logical2 = _weights(8, 256, 256, 456)
    needed = {0, 3}
    payload = _reference_weight_payload(logical1, logical2, 0, needed)
    serialized = pickle.dumps(payload)
    selected_bytes = sum(
        tensor.numel() * tensor.element_size()
        for pair in payload.values()
        for weights in pair
        for tensor in weights
    )
    # Serialization must scale with selected experts, not whole local storage.
    assert len(serialized) < 2 * selected_bytes
    restored = pickle.loads(serialized)
    assert set(restored) == needed
    for expert, pair in restored.items():
        for received, logical in zip(pair, (logical1, logical2)):
            for tensor, source in zip(received, logical):
                assert torch.equal(tensor, source[expert])


if __name__ == "__main__":
    main()
