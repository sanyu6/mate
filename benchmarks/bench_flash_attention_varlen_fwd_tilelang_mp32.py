#!/usr/bin/env python3
"""Benchmark the shipped PH1S MP32 FlashAttention forward configurations.

This is the product benchmark entry point.  It exercises the public MATE
interface and reports median GPU latency and achieved attention TFLOP/s for
the three supported dispatch classes: 128, 192/128, and 256.

Examples
--------
    python benchmarks/bench_flash_attention_varlen_fwd_tilelang_mp32.py
    python benchmarks/bench_flash_attention_varlen_fwd_tilelang_mp32.py \
        --dims 192-128 --seqlen-q 2048 --seqlen-kv 2048 --check
"""

from __future__ import annotations

import argparse
import statistics

import torch
import torch_musa  # noqa: F401

from mate import flash_attn_varlen_func


_DIMS = {
    "128": (128, 128),
    "192-128": (192, 128),
    "256": (256, 256),
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _make_inputs(args, dim_qk: int, dim_v: int):
    device = torch.device("musa")
    dtype = getattr(torch, args.dtype)
    shape_q = (args.batch, args.seqlen_q, args.heads, dim_qk)
    shape_k = (args.batch, args.seqlen_kv, args.heads, dim_qk)
    shape_v = (args.batch, args.seqlen_kv, args.heads, dim_v)
    q = torch.randn(shape_q, device=device, dtype=dtype)
    k = torch.randn(shape_k, device=device, dtype=dtype)
    v = torch.randn(shape_v, device=device, dtype=dtype)
    if not args.varlen:
        return q, k, v, None, None

    q = q.reshape(args.batch * args.seqlen_q, args.heads, dim_qk)
    k = k.reshape(args.batch * args.seqlen_kv, args.heads, dim_qk)
    v = v.reshape(args.batch * args.seqlen_kv, args.heads, dim_v)
    q_offsets = torch.arange(
        0,
        (args.batch + 1) * args.seqlen_q,
        args.seqlen_q,
        device=device,
        dtype=torch.int32,
    )
    kv_offsets = torch.arange(
        0,
        (args.batch + 1) * args.seqlen_kv,
        args.seqlen_kv,
        device=device,
        dtype=torch.int32,
    )
    return q, k, v, q_offsets, kv_offsets


def _run(args, inputs):
    q, k, v, cu_q, cu_k = inputs
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=args.seqlen_q if args.varlen else None,
        max_seqlen_k=args.seqlen_kv if args.varlen else None,
        causal=args.causal,
        backend=args.backend,
    )
    return flash_attn_varlen_func(**kwargs)


def _measure_ms(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    starts = [torch.musa.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.musa.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.musa.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def _reference(args, inputs, dim_qk: int):
    q, k, v, cu_q, cu_k = inputs
    q = q.float()
    k = k.float()
    v = v.float()
    scale = dim_qk**-0.5
    outputs = []
    if args.varlen:
        q_lens = cu_q[1:].cpu() - cu_q[:-1].cpu()
        k_lens = cu_k[1:].cpu() - cu_k[:-1].cpu()
        q_begin = k_begin = 0
        for q_len, k_len in zip(q_lens.tolist(), k_lens.tolist()):
            q_part = q[q_begin : q_begin + q_len].transpose(0, 1)
            k_part = k[k_begin : k_begin + k_len].transpose(0, 1)
            v_part = v[k_begin : k_begin + k_len].transpose(0, 1)
            q_begin += q_len
            k_begin += k_len
            scores = torch.matmul(q_part, k_part.transpose(-1, -2)) * scale
            if args.causal:
                q_idx = torch.arange(q_len, device=q.device).unsqueeze(1)
                k_idx = torch.arange(k_len, device=q.device).unsqueeze(0)
                scores = scores.masked_fill(k_idx > q_idx + k_len - q_len, -torch.inf)
            outputs.append(torch.softmax(scores, dim=-1).matmul(v_part).transpose(0, 1))
        return torch.cat(outputs, dim=0)

    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    if args.causal:
        mask = torch.triu(
            torch.ones(
                args.seqlen_q,
                args.seqlen_kv,
                dtype=torch.bool,
                device=q.device,
            ),
            diagonal=args.seqlen_kv - args.seqlen_q + 1,
        )
        scores = scores.masked_fill(mask, -torch.inf)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhv", probs, v)


def _bench_one(args, label: str, dim_qk: int, dim_v: int) -> None:
    inputs = _make_inputs(args, dim_qk, dim_v)
    run = lambda: _run(args, inputs)
    output = run()
    torch.musa.synchronize()
    if args.check:
        reference = _reference(args, inputs, dim_qk).to(device="cpu")
        actual = output.to(device="cpu")
        torch.testing.assert_close(
            actual.float(), reference.float(), rtol=3e-2, atol=3e-2
        )
    samples = _measure_ms(run, args.warmup, args.iters)
    median_ms = statistics.median(samples)
    tokens = args.batch * args.seqlen_q * args.seqlen_kv * args.heads
    # QK^T and PV each perform one multiply-add per q/k/v dimension.
    tflops = 2.0 * tokens * (dim_qk + dim_v) / (median_ms * 1e9)
    print(
        f"config={label} backend={args.backend} varlen={args.varlen} "
        f"causal={args.causal} shape=({args.batch},{args.seqlen_q},{args.seqlen_kv},"
        f"{args.heads},{dim_qk},{dim_v}) median_ms={median_ms:.4f} "
        f"p50_tflops={tflops:.2f} samples={len(samples)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", choices=[*list(_DIMS), "all"], default="all")
    parser.add_argument("--batch", type=_positive_int, default=1)
    parser.add_argument("--seqlen-q", type=_positive_int, default=2048)
    parser.add_argument("--seqlen-kv", type=_positive_int, default=2048)
    parser.add_argument("--heads", type=_positive_int, default=12)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--warmup", type=_nonnegative_int, default=10)
    parser.add_argument("--iters", type=_positive_int, default=50)
    parser.add_argument("--backend", choices=["auto", "tilelang"], default="tilelang")
    parser.add_argument("--varlen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="run a PyTorch reference check"
    )
    args = parser.parse_args()
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        parser.error("this benchmark requires a MUSA device")
    torch.manual_seed(0)
    torch.musa.manual_seed_all(0)
    labels = list(_DIMS) if args.dims == "all" else [args.dims]
    for label in labels:
        _bench_one(args, label, *_DIMS[label])


if __name__ == "__main__":
    main()
