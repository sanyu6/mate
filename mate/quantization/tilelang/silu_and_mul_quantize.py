"""Fused SwiGLU/SiTU with FP4/FP8 quantization or direct BF16 output."""

from typing import Literal
import math

import torch

from ._common import (
    _can_vectorize,
    _ceil_div,
)
from ._quantize import tilelang_quantize_masked
from ._quantize_grouped import tilelang_quantize_grouped


SfType = Literal["e8m0", "e4m3", "fp32"]


def _resolve_silu_config(rows: int) -> int:
    """Return the number of 32-value scale groups owned by one CTA."""
    # Keep more CTAs for small M. Large M owns 64 groups, with the per-thread
    # work selected separately by _resolve_groups_in_thread.
    return 32 if rows <= 1024 else 64


def _resolve_groups_in_thread(
    rows: int,
    cols: int,
    sf_type: SfType,
    groups_per_block: int,
) -> int:
    """Choose how many independent scale groups each thread participates in."""
    if groups_per_block < 64:
        return 1
    # Four groups hide more latency on wide E8M0 inputs, including the full
    # blocks preceding a tail. E4M3 uses two groups per thread.
    if sf_type == "e8m0" and rows >= 8192 and cols >= 16384:
        return 4
    return 2


def _check_args(
    x: torch.Tensor,
    global_scale: torch.Tensor | None,
    sf_type: SfType,
    *,
    require_even_k: bool = True,
) -> SfType:
    if x.dim() not in (2, 3):
        raise ValueError(f"input must be rank 2 or 3, got {tuple(x.shape)}")
    if x.stride(-1) != 1:
        raise ValueError("input must be contiguous in the K dimension")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("input must have dtype torch.float16 or torch.bfloat16")
    if x.device.type != "musa":
        raise ValueError(f"input must be on a MUSA device, got {x.device}")
    if x.shape[-1] == 0 or x.shape[-2] == 0:
        raise ValueError("input must have non-empty dimensions")
    if x.dim() == 3 and x.shape[0] == 0:
        raise ValueError("input must have a non-empty batch dimension")
    if x.shape[-1] % 2 != 0:
        raise ValueError("input.shape[-1] must be even because it contains gate and up")
    cols = x.shape[-1] // 2
    if require_even_k and cols % 2:
        raise ValueError("the output K dimension must be even")
    if sf_type not in ("e8m0", "e4m3", "fp32"):
        raise ValueError(f"unsupported sf_type: {sf_type!r}")
    if global_scale is not None:
        if global_scale.shape != (1,) or global_scale.dtype != torch.float32:
            raise ValueError("global_scale must be a one-element float32 tensor")
        if global_scale.device != x.device:
            raise ValueError("global_scale must be on the same device as input")
        if not bool(torch.isfinite(global_scale).item()):
            raise ValueError("global_scale must be finite")
    return sf_type


def _silu_and_mul_quantize_impl(
    input: torch.Tensor,
    output_dtype: torch.dtype = torch.float4_e2m1fn_x2,
    *,
    global_scale: torch.Tensor | None = None,
    sf_vec_size: int = 32,
    sf_type: SfType | None = None,
    is_global_scale_inversed: bool = False,
    activation: Literal["swiglu", "situ"] = "swiglu",
    swiglu_limit: float | None = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Validate inputs and launch the TileLang backend for ``silu_and_mul_quantize``."""
    if activation not in ("swiglu", "situ"):
        raise ValueError("activation must be swiglu or situ")
    if swiglu_limit is not None:
        if (
            activation != "swiglu"
            or not math.isfinite(swiglu_limit)
            or swiglu_limit <= 0
        ):
            raise ValueError("swiglu_limit must be finite and positive for swiglu")
    if activation == "situ":
        if not math.isfinite(situ_beta) or situ_beta <= 0:
            raise ValueError("situ_beta must be finite and positive")
        if situ_linear_beta is not None and (
            not math.isfinite(situ_linear_beta) or situ_linear_beta <= 0
        ):
            raise ValueError("situ_linear_beta must be None or finite and positive")
    is_bf16 = output_dtype == torch.bfloat16
    if output_dtype not in (
        torch.bfloat16,
        torch.float4_e2m1fn_x2,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        raise ValueError(
            "output_dtype must be torch.float4_e2m1fn_x2, "
            "torch.float8_e4m3fn, torch.float8_e5m2 or torch.bfloat16"
        )
    if is_bf16 and (
        global_scale is not None
        or sf_vec_size != 32
        or sf_type is not None
        or is_global_scale_inversed
    ):
        raise ValueError("BF16 output does not accept quantization scale options")
    is_fp4 = output_dtype == torch.float4_e2m1fn_x2
    if not is_fp4 and global_scale is not None:
        raise ValueError("global_scale is supported for FP4 output only")
    if sf_vec_size not in ((32,) if is_fp4 else (32, 128)):
        raise ValueError("sf_vec_size must be 32 for FP4, or 32/128 for FP8")
    sf_type = ("fp32" if sf_vec_size == 128 else "e8m0") if sf_type is None else sf_type
    if (sf_vec_size == 128) != (sf_type == "fp32"):
        raise ValueError("FP32 scales require sf_vec_size=128 with FP8 output")
    sf_type = _check_args(input, global_scale, sf_type, require_even_k=not is_bf16)
    if global_scale is not None and is_global_scale_inversed:
        if not bool((global_scale > 0).item()):
            raise ValueError("global_scale must be positive in inverse mode")

    input_is_3d = input.dim() == 3
    x = input if input_is_3d else input.unsqueeze(0)
    batch, rows, input_cols = x.shape
    cols = input_cols // 2
    groups = _ceil_div(cols, sf_vec_size)
    q = torch.empty(
        (batch, rows, cols // 2 if is_fp4 else cols),
        dtype=output_dtype,
        device=input.device,
    )
    sf = None
    if not is_bf16:
        sf = torch.empty_strided(
            (batch, rows, groups),
            (rows * groups, 1, rows),
            dtype=torch.float32 if sf_type == "fp32" else torch.uint8,
            device=input.device,
        )
    if cols % 16 == 0 and _can_vectorize(x):
        # A 128-value group needs four times as many lanes as a 32-value
        # group. Keep its tile at 16 groups to avoid oversized CTAs and
        # mostly idle lanes on medium-width rows and tails.
        groups_per_block = 16 if sf_vec_size == 128 else _resolve_silu_config(rows)
        # A 4096-value row plus one scale group is faster in one CTA than
        # two full CTAs followed by a nearly empty third CTA.
        if sf_vec_size == 128 and rows > 1024 and groups == 33:
            groups_per_block = 64
        if activation == "situ":
            # Keep small rows and 1x128 groups at eight outputs per thread.
            groups_per_block = 8 if sf_vec_size == 128 and rows > 1024 else 16
            groups_in_thread = 1
            if (
                sf_type == "e8m0"
                and rows > 1024
                and cols >= 2048
                and (cols % 1024 == 0 or cols >= 8192)
            ):
                # Sixteen outputs amortize indexing and expose independent
                # activation work without the register pressure of 32 outputs.
                # Short tails retain eight outputs; wide tails use a smaller
                # CTA so masked lanes do not dominate the final block.
                groups_per_block = 32 if is_fp4 or cols % 2048 else 64
                groups_in_thread = 2
        elif swiglu_limit is not None and sf_vec_size == 32:
            groups_per_block = 32
            groups_in_thread = 1 if rows <= 1024 else 2
        elif is_fp4:
            groups_in_thread = _resolve_groups_in_thread(
                rows,
                cols,
                sf_type,
                groups_per_block,
            )
        else:
            groups_in_thread = (
                (1 if rows <= 1024 else 2)
                if sf_vec_size == 128
                else (1 if rows <= 1024 else (4 if rows >= 8192 else 2))
            )
        has_tail = groups % groups_per_block != 0
        kernel = tilelang_quantize_grouped(
            sf_type,
            input.dtype,
            output_dtype,
            groups_per_block,
            groups_in_thread,
            has_tail,
            fused=True,
            has_partial_group=cols % sf_vec_size != 0,
            sf_vec_size=sf_vec_size,
            activation=activation,
            swiglu_limit=swiglu_limit,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            use_global_scale=global_scale is not None,
            is_global_scale_inversed=is_global_scale_inversed,
            replicate_scale=(
                sf_type == "e8m0" and activation == "situ" and rows > 1024
                if is_fp4
                else None
            ),
        )
        if is_bf16:
            kernel(x, q)
        elif is_fp4:
            kernel(x, q, sf, global_scale, cols)
        else:
            kernel(x, q, sf)

    else:
        kernel = tilelang_quantize_masked(
            input.dtype,
            torch.uint8 if is_fp4 else output_dtype,
            sf_type,
            True,
            global_scale is not None,
            is_global_scale_inversed,
            sf_vec_size=sf_vec_size,
            activation=activation,
            swiglu_limit=swiglu_limit,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )
        args = (x, q.view(torch.uint8) if is_fp4 else q, sf)
        if is_bf16:
            kernel(x, q)
        elif global_scale is not None:
            kernel(*args, global_scale)
        else:
            kernel(*args)
    if is_fp4:
        q = q.view(torch.uint8)
    return (q, sf) if input_is_3d else (q[0], sf[0] if sf is not None else None)


def main() -> None:
    from mate.quantization import silu_and_mul_quantize

    import argparse
    import math

    from mate.testing.utils import bench_kineto

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--cols", type=int, default=16384, help="Output K dimension")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--sf-type", choices=("e8m0", "e4m3", "fp32"))
    parser.add_argument("--sf-vec-size", choices=(32, 128), type=int, default=32)
    parser.add_argument("--global-scale", type=float)
    parser.add_argument(
        "--output-dtype", choices=("fp4", "e4m3", "e5m2", "bf16"), default="fp4"
    )
    parser.add_argument("--is-global-scale-inversed", action="store_true")
    parser.add_argument("--activation", choices=("swiglu", "situ"), default="swiglu")
    parser.add_argument("--swiglu-limit", type=float)
    parser.add_argument("--situ-beta", type=float, default=4.0)
    parser.add_argument("--situ-linear-beta", type=float, default=25.0)
    parser.add_argument("--no-situ-up-softcap", action="store_true")
    parser.add_argument("--rep", type=int, default=30)
    args = parser.parse_args()
    if args.rows <= 0 or args.cols <= 0 or args.batch_size <= 0 or args.rep <= 0:
        parser.error("rows, cols, batch-size, and rep must be positive")
    if (args.cols % 2 != 0 and args.output_dtype != "bf16") or (
        args.global_scale is not None and not (0.0 < args.global_scale < math.inf)
    ):
        parser.error("cols must be even and global-scale must be finite and positive")

    if args.output_dtype != "fp4" and args.global_scale is not None:
        parser.error("global-scale is supported for FP4 output only")
    output_dtype = {
        "fp4": torch.float4_e2m1fn_x2,
        "e4m3": torch.float8_e4m3fn,
        "e5m2": torch.float8_e5m2,
        "bf16": torch.bfloat16,
    }[args.output_dtype]
    device = torch.device("musa")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(0)

    x = torch.randn(
        (args.batch_size, args.rows, args.cols * 2), device=device, dtype=dtype
    )
    global_scale = (
        torch.tensor([args.global_scale], dtype=torch.float32, device=device)
        if args.global_scale is not None
        else None
    )
    activation_options = dict(
        activation=args.activation,
        swiglu_limit=args.swiglu_limit,
        situ_beta=args.situ_beta,
        situ_linear_beta=None if args.no_situ_up_softcap else args.situ_linear_beta,
    )
    q, sf = silu_and_mul_quantize(
        x,
        output_dtype=output_dtype,
        global_scale=global_scale,
        sf_vec_size=args.sf_vec_size,
        sf_type=args.sf_type,
        is_global_scale_inversed=args.is_global_scale_inversed,
        **activation_options,
    )

    def run():
        return silu_and_mul_quantize(
            x,
            output_dtype=output_dtype,
            global_scale=global_scale,
            sf_vec_size=args.sf_vec_size,
            sf_type=args.sf_type,
            is_global_scale_inversed=args.is_global_scale_inversed,
            **activation_options,
        )

    torch.musa.synchronize()
    print(
        f"input={tuple(x.shape)}, q={tuple(q.shape)}, sf={tuple(sf.shape) if sf is not None else None}, "
        f"dtype={args.dtype}, sf_type={args.sf_type}, sf_vec_size={args.sf_vec_size}, "
        f"activation={args.activation}, swiglu_limit={args.swiglu_limit}, "
        f"rep={args.rep}, flush_l2=True",
        flush=True,
    )
    seconds = bench_kineto(
        run,
        kernel_names="tilelang",
        num_tests=args.rep,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError("Kineto did not record the fused quantization kernel")
    logical_bytes = (
        x.numel() * x.element_size()
        + q.numel() * q.element_size()
        + (sf.numel() * sf.element_size() if sf is not None else 0)
        + (
            global_scale.numel() * global_scale.element_size()
            if global_scale is not None
            else 0
        )
    )
    print(
        f"silu_and_mul_quantize ({args.output_dtype}): "
        f"{logical_bytes / seconds / 1e9:.3f} logical GB/s "
        f"(input/global_scale read + q/sf write)"
    )


if __name__ == "__main__":
    main()
