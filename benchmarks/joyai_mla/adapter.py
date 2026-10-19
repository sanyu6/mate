"""Decode-only SGLang adapter for the fixed-query small-batch MLA kernels."""

from functools import cache
from pathlib import Path

import torch
import mate.mha_interface as mi


@cache
def load_library(path):
    import tvm_ffi

    library = Path(path).expanduser().resolve()
    if not library.is_file():
        raise FileNotFoundError(f"MLA library does not exist: {library}")
    return tvm_ffi.load_module(str(library))


def supports_fixed_decode(kwargs):
    q, qv = kwargs.get("q"), kwargs.get("qv")
    lengths = kwargs.get("cache_seqlens")
    return (
        q is not None
        and qv is not None
        and lengths is not None
        and q.ndim == qv.ndim == 3
        and 0 < q.shape[0] <= 5
        and q.shape[0] == lengths.shape[0]
        and q.shape[-2] <= 32
        and q.shape[-1] == 64
        and qv.shape[-1] == 512
        and q.shape[:-1] == qv.shape[:-1]
        and q.dtype in (torch.bfloat16, torch.float16)
        and q.dtype == qv.dtype
        and kwargs.get("max_seqlen_q") == 1
        and not kwargs.get("return_softmax_lse", False)
        and isinstance(kwargs.get("scheduler_metadata"), tuple)
        and kwargs.get("k") is None
        and kwargs.get("v") is None
        and not kwargs.get("softcap", 0.0)
        and kwargs.get("k_descale") is None
        and kwargs.get("v_descale") is None
    )


def fixed_decode(module, kwargs, *, metadata_initialized=None):
    """Run one query/request; preserve the caller's scheduler workspace."""
    q, qv = kwargs["q"], kwargs["qv"]
    q4 = mi._prepare_mla_query_input(q, require_seq_dense=True).unsqueeze(1)
    qv4 = mi._prepare_mla_query_input(qv, require_seq_dense=True).unsqueeze(1)
    workspace, initialized = kwargs["scheduler_metadata"]
    if metadata_initialized is not None:
        initialized = metadata_initialized
    metadata, splits = mi._prepare_mla_scheduler_metadata_from_workspace(
        workspace, initialized, qv4, kwargs["v_cache"], kwargs["cache_seqlens"], None, 1
    )
    out = torch.empty((*q4.shape[:-1], 512), dtype=q.dtype, device=q.device)
    lse = torch.empty(q4.shape[:-1], dtype=torch.float32, device=q.device)
    module.get_function("mla_with_kvcache")(
        qv4,
        q4,
        kwargs["v_cache"],
        kwargs["k_cache"],
        kwargs["cache_seqlens"],
        None,
        None,
        kwargs["page_table"],
        metadata,
        splits,
        out,
        lse,
        kwargs["softmax_scale"],
        kwargs.get("causal", False),
    )
    return out.squeeze(1), lse


def install(mode, library=None):
    """Patch only normal decode; retain SGLang's prefill/fallback functions."""
    import sglang.srt.hardware_backend.musa.attention.flashattention_backend as fa

    if mode not in ("default", "optimized"):
        raise ValueError(f"Unknown adapter mode: {mode}")
    module = load_library(library) if mode == "optimized" else None
    original_attention = fa.flash_attn_with_kvcache
    original_decode = fa.MusaFlashAttentionBackend.forward_decode
    state = {"decode": False}

    def attention(*args, **kwargs):
        if (
            mode == "optimized"
            and state["decode"]
            and not args
            and supports_fixed_decode(kwargs)
        ):
            return fixed_decode(module, kwargs)[0]
        return original_attention(*args, **kwargs)

    def decode(self, *args, **kwargs):
        previous = state["decode"]
        state["decode"] = True
        try:
            return original_decode(self, *args, **kwargs)
        finally:
            state["decode"] = previous

    fa.flash_attn_with_kvcache = attention
    fa.MusaFlashAttentionBackend.forward_decode = decode
