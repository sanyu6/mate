"""Same-input A/A, A/B and FP32 reference checks on actual model Q/KV."""

import json
from pathlib import Path

import torch
from adapter import fixed_decode, load_library


def install(library, output_dir, min_kv=20000):
    import sglang.srt.hardware_backend.musa.attention.flashattention_backend as fa

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    module = load_library(library)
    original_attention = fa.flash_attn_with_kvcache
    original_decode = fa.MusaFlashAttentionBackend.forward_decode
    state = {"decode": False, "layer": -1, "count": 0}

    def metrics(actual, reference):
        actual, reference = actual.float(), reference.float()
        delta = actual - reference
        return {
            "max_abs": delta.abs().max().item(),
            "relative_l2": (delta.norm() / reference.norm().clamp_min(1e-20)).item(),
            "mean_abs": delta.abs().mean().item(),
            "finite": bool(torch.isfinite(actual).all().item()),
        }

    def attention(*args, **kwargs):
        q, qv = kwargs.get("q"), kwargs.get("qv")
        eligible = (
            state["decode"]
            and not args
            and qv is not None
            and q is not None
            and q.ndim == 3
            and q.shape[0] == 1
            and q.shape[-1] == 64
            and qv.shape[-1] == 512
            and kwargs.get("max_seqlen_q") == 1
            and isinstance(kwargs.get("scheduler_metadata"), tuple)
        )
        if not eligible:
            return original_attention(*args, **kwargs)
        length = int(kwargs["cache_seqlens"][0].item())
        if length < min_kv:
            return original_attention(*args, **kwargs)
        assert not torch.musa.is_current_stream_capturing()
        base = original_attention(*args, **kwargs)
        again_kwargs = dict(
            kwargs, scheduler_metadata=(kwargs["scheduler_metadata"][0], True)
        )
        repeat = original_attention(**again_kwargs)
        optimized, lse = fixed_decode(module, kwargs, metadata_initialized=True)
        page = kwargs["v_cache"].shape[1]
        page_ids = kwargs["page_table"][0, : (length + page - 1) // page].long()
        latent = (
            kwargs["v_cache"]
            .index_select(0, page_ids)
            .reshape(-1, 512)[:length]
            .float()
        )
        rope = (
            kwargs["k_cache"].index_select(0, page_ids).reshape(-1, 64)[:length].float()
        )
        scores = (qv[0].float() @ latent.T + q[0].float() @ rope.T) * kwargs[
            "softmax_scale"
        ]
        reference = (scores.softmax(-1) @ latent).unsqueeze(0)
        ref_lse = scores.logsumexp(-1).unsqueeze(0).unsqueeze(1)
        row = {
            "call": state["count"],
            "layer": state["layer"],
            "device": str(q.device),
            "kv_length": length,
            "heads": q.shape[-2],
            "aa": metrics(repeat, base),
            "ab": metrics(optimized, base),
            "default_fp32": metrics(base, reference),
            "optimized_fp32": metrics(optimized, reference),
            "lse_fp32": metrics(lse, ref_lse),
        }
        if state["count"] == 0:
            # Cross-check the GPU float reference against an independent CPU FP32
            # evaluation on the same actual model inputs.
            ql_cpu, qr_cpu = qv[0].float().cpu(), q[0].float().cpu()
            latent_cpu, rope_cpu = latent.cpu(), rope.cpu()
            cpu_scores = (ql_cpu @ latent_cpu.T + qr_cpu @ rope_cpu.T) * kwargs[
                "softmax_scale"
            ]
            cpu_reference = (cpu_scores.softmax(-1) @ latent_cpu).unsqueeze(0)
            row["gpu_reference_vs_cpu"] = metrics(reference.cpu(), cpu_reference)
            row["default_cpu_fp32"] = metrics(base.cpu(), cpu_reference)
            row["optimized_cpu_fp32"] = metrics(optimized.cpu(), cpu_reference)
        row["close_default_ref"] = bool(
            torch.allclose(base.float(), reference, atol=1.5e-2, rtol=1e-2)
        )
        row["close_optimized_ref"] = bool(
            torch.allclose(optimized.float(), reference, atol=1.5e-2, rtol=1e-2)
        )
        row["aa_exact"] = bool(torch.equal(base, repeat))
        with (output_dir / f"attention-numerics-{q.device.index}.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        state["count"] += 1
        # Always propagate the baseline output. Every comparison sees baseline
        # hidden states and identical tokens/KV, even if candidate errors exist.
        return base

    def decode(self, *args, **kwargs):
        previous = (state["decode"], state["layer"])
        state["decode"] = True
        layer = args[3] if len(args) > 3 else kwargs.get("layer")
        state["layer"] = getattr(layer, "layer_id", -1)
        try:
            return original_decode(self, *args, **kwargs)
        finally:
            state["decode"], state["layer"] = previous

    fa.flash_attn_with_kvcache = attention
    fa.MusaFlashAttentionBackend.forward_decode = decode
