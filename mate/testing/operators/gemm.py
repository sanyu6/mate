"""Structured GEMM workloads, public API calls, and independent references."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from importlib import import_module

import torch

from mate.testing.utils import group_quantize_fp8, tensor_quantize_fp8
from .operator import (
    Operator,
    OperatorInputs,
    OperatorOutputs,
    OperatorReference,
    OperatorWorkload,
)

_FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)


class GemmApi(Enum):
    BMM = "bmm"
    MASKED_MOE = "masked_moe"
    RAGGED_M_MOE = "ragged_m_moe"
    RAGGED_K_MOE = "ragged_k_moe"
    MIXED_DTYPE_MOE = "mixed_dtype_moe"


@dataclass(frozen=True, kw_only=True)
class GemmWorkload(OperatorWorkload):
    m: int
    n: int
    k: int
    api: GemmApi = GemmApi.BMM
    batch_size: int = 1
    group_size: int = 1
    input_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = torch.bfloat16
    backend: str = "auto"
    quantized: bool = False
    mixed_dtype: str = "s4fp8"
    mixed_layout: str = "ragged"
    quant_tile: int = 128
    scale_major_mode: str = "K"
    alignment_m: int = 128
    group_length: str = "fixed"
    group_lengths: tuple[int, ...] | None = None
    trans_a: bool = False
    trans_b: bool = True
    device: str | torch.device = "musa"
    seed: int | None = 42

    def __post_init__(self):
        if not isinstance(self.api, GemmApi):
            raise ValueError("expected GemmApi")
        if min(self.m, self.n, self.k, self.batch_size, self.group_size) <= 0:
            raise ValueError("GEMM dimensions and group counts must be positive")
        if self.group_length not in ("fixed", "random"):
            raise ValueError("group_length must be fixed or random")
        if self.group_lengths is not None and (
            len(self.group_lengths) != self.group_size or min(self.group_lengths) < 0
        ):
            raise ValueError(
                "group_lengths must contain a non-negative count per expert"
            )
        mixed = self.api is GemmApi.MIXED_DTYPE_MOE
        if self.input_dtype not in (
            _FP8 if self.quantized or mixed else (torch.float16, torch.bfloat16)
        ):
            raise ValueError("input_dtype does not match the GEMM quantization path")
        if self.out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("generated GEMM requires FP16/BF16/FP32 output")
        if self.api is GemmApi.BMM:
            return
        if self.trans_a or not self.trans_b:
            raise ValueError("generated MoE workloads use the established API layouts")
        if self.api is GemmApi.RAGGED_K_MOE:
            if self.out_dtype != torch.float32:
                raise ValueError("ragged-K benchmark requires FP32 output")
            if self.quantized and self.quant_tile != 128:
                raise ValueError("FP8 ragged-K requires quant_tile=128")
        elif not self.quantized and not mixed:
            if self.out_dtype != self.input_dtype:
                raise ValueError("16-bit MoE requires matching input and output dtypes")
            if self.backend == "mutlass" and self.input_dtype != torch.bfloat16:
                raise ValueError("16-bit MoE mutlass requires BF16")
        elif self.out_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("quantized MoE requires BF16/FP16 output")
        if self.quantized or mixed:
            if self.scale_major_mode != "K":
                raise ValueError("quantized MoE requires scale_major_mode=K")
        if self.api is GemmApi.RAGGED_M_MOE or (
            mixed and self.mixed_layout == "ragged"
        ):
            if self.alignment_m not in (128, 256):
                raise ValueError("ragged-M alignment_m must be 128 or 256")
        if mixed:
            if self.mixed_dtype not in ("s4fp8", "fp4fp8") or self.mixed_layout not in (
                "masked",
                "ragged",
            ):
                raise ValueError("unsupported mixed dtype or layout")
            tile = 32 if self.mixed_dtype == "fp4fp8" else 128
            if self.quant_tile != tile or self.k % tile:
                raise ValueError(
                    f"{self.mixed_dtype} requires quant_tile={tile} and aligned K"
                )
            if self.mixed_dtype == "fp4fp8" and (
                self.input_dtype != torch.float8_e4m3fn
                or self.out_dtype != torch.bfloat16
            ):
                raise ValueError("FP4FP8 requires E4M3 A and BF16 output")
            if (
                self.mixed_dtype == "fp4fp8"
                and self.mixed_layout == "ragged"
                and self.alignment_m != 256
            ):
                raise ValueError("FP4FP8 ragged-M requires alignment_m=256")
        elif self.quantized and self.api is not GemmApi.RAGGED_K_MOE:
            tile = 128 if self.api is GemmApi.MASKED_MOE else self.quant_tile
            if tile <= 0 or self.n % tile or self.k % tile:
                raise ValueError("FP8 MoE requires N and K divisible by quant_tile")


@dataclass(kw_only=True)
class GemmInputs(OperatorInputs):
    """Physical tensors and dispatch arguments; workload is optional provenance."""

    api: GemmApi = GemmApi.BMM
    workload: GemmWorkload | None = None
    a: torch.Tensor | None = None
    b: torch.Tensor | None = None
    out: torch.Tensor | None = None
    c: torch.Tensor | None = None
    scale_a: torch.Tensor | None = None
    scale_b: torch.Tensor | None = None
    scale_out: torch.Tensor | None = None
    residual_e8m0: torch.Tensor | None = None
    epilogue_fp32: torch.Tensor | None = None
    masked_m: torch.Tensor | None = None
    m_indices: torch.Tensor | None = None
    group_k_idx: torch.Tensor | None = None
    signal: torch.Tensor | None = None
    group_lengths: tuple[int, ...] | None = None
    valid_m: int | None = None
    total_k: int | None = None
    logical_k: int | None = None
    trans_a: bool = False
    trans_b: bool = True
    recipe_a: tuple[int, int] | None = None
    recipe_b: tuple[int, int] | None = None
    scale_granularity_mnk: tuple[int, int, int] = (1, 128, 128)
    fixed_scale_layout: bool | None = None
    out_dtype: torch.dtype | None = None
    backend: str | None = None
    mixed_dtype: str = "s4fp8"
    mixed_layout: str = "ragged"
    alignment_m: int = 128
    expect_tokens: int | None = None
    gemm_mode: str = "per_token"
    major_a_mode: str = "K"
    major_b_mode: str = "K"
    num_mp: int | None = None
    enable_overlap: bool = False

    def __post_init__(self):
        if self.backend is None:
            self.backend = (
                self.workload.backend if self.workload is not None else "auto"
            )
        if self.out_dtype is None:
            self.out_dtype = (
                self.out.dtype
                if self.out is not None
                else self.workload.out_dtype
                if self.workload is not None
                else torch.bfloat16
            )
        if (
            self.workload is not None
            and self.workload.quantized
            and self.api is GemmApi.BMM
        ):
            if self.recipe_a is None:
                self.recipe_a = (-1, -1)
            if self.recipe_b is None:
                self.recipe_b = (-1, -1)


@dataclass(frozen=True, kw_only=True)
class GemmOutputs(OperatorOutputs):
    out: torch.Tensor
    auxiliary: tuple = ()


@dataclass(frozen=True, kw_only=True)
class GemmReference(OperatorReference):
    out: torch.Tensor
    valid_rows: torch.Tensor | None = None


class GemmOperator(Operator):
    def generate(self, workload: OperatorWorkload) -> GemmInputs:
        if not isinstance(workload, GemmWorkload):
            raise TypeError("expected GemmWorkload")
        w = workload
        a_shape: tuple[int, ...]
        out_shape: tuple[int, ...]
        if w.seed is not None:
            torch.manual_seed(w.seed)
        rng = random.Random(w.seed) if w.seed is not None else random
        x = GemmInputs(
            api=w.api,
            workload=w,
            out_dtype=w.out_dtype,
            backend=w.backend,
            logical_k=w.k,
            trans_a=w.trans_a,
            trans_b=w.trans_b,
            mixed_dtype=w.mixed_dtype,
            mixed_layout=w.mixed_layout,
            alignment_m=w.alignment_m,
            expect_tokens=w.m,
        )
        if w.api is GemmApi.BMM:
            a_shape = (
                (w.batch_size, w.k, w.m) if w.trans_a else (w.batch_size, w.m, w.k)
            )
            b_shape = (
                (w.batch_size, w.n, w.k) if w.trans_b else (w.batch_size, w.k, w.n)
            )
            if w.quantized:
                x.a, x.scale_a = tensor_quantize_fp8(
                    torch.randn(a_shape, device=w.device), w.input_dtype
                )
                x.b, x.scale_b = tensor_quantize_fp8(
                    torch.randn(b_shape, device=w.device), w.input_dtype
                )
                x.recipe_a = x.recipe_b = (-1, -1)
            else:
                x.a = torch.randn(a_shape, device=w.device, dtype=w.input_dtype)
                x.b = torch.randn(b_shape, device=w.device, dtype=w.input_dtype)
            x.out = torch.empty(
                w.batch_size, w.m, w.n, dtype=w.out_dtype, device=w.device
            )
            return x
        expected = w.k if w.api is GemmApi.RAGGED_K_MOE else w.m
        lengths = w.group_lengths or tuple(
            int(expected * rng.uniform(0.7, 1.3))
            if w.group_length == "random"
            else expected
            for _ in range(w.group_size)
        )
        x.group_lengths = lengths
        if w.api is GemmApi.RAGGED_K_MOE:
            x.group_k_idx = torch.tensor(lengths, device=w.device, dtype=torch.int32)
            x.total_k = sum(lengths)
            x.gemm_mode, x.major_a_mode, x.major_b_mode = "per_expert", "M", "N"
            x.scale_granularity_mnk = (1, 1, 128)
            if w.quantized:
                a_parts, b_parts, sa_parts, sb_parts = [], [], [], []
                for length in lengths:
                    if not length:
                        continue
                    padded = math.ceil(length / 128) * 128
                    a, sa = _group_quantize_fp8(
                        torch.randn(w.m, padded, device=w.device),
                        (w.m, padded // 128),
                        (1, 128),
                        w.input_dtype,
                        "K",
                    )
                    b, sb = _group_quantize_fp8(
                        torch.randn(w.n, padded, device=w.device),
                        (w.n, padded // 128),
                        (1, 128),
                        w.input_dtype,
                        "K",
                    )
                    a_parts.append(a[:, :length].T.contiguous())
                    b_parts.append(b[:, :length].T.contiguous())
                    sa_parts.append(sa.T.contiguous())
                    sb_parts.append(sb.T.contiguous())
                x.a = (
                    torch.cat(a_parts)
                    if a_parts
                    else torch.empty(0, w.m, device=w.device, dtype=w.input_dtype)
                )
                x.b = (
                    torch.cat(b_parts)
                    if b_parts
                    else torch.empty(0, w.n, device=w.device, dtype=w.input_dtype)
                )
                x.scale_a = (
                    torch.cat(sa_parts)
                    if sa_parts
                    else torch.empty(0, w.m, device=w.device)
                )
                x.scale_b = (
                    torch.cat(sb_parts)
                    if sb_parts
                    else torch.empty(0, w.n, device=w.device)
                )
            else:
                x.a = torch.randn(
                    sum(lengths), w.m, device=w.device, dtype=w.input_dtype
                )
                x.b = torch.randn(
                    sum(lengths), w.n, device=w.device, dtype=w.input_dtype
                )
            x.out = torch.zeros(
                w.group_size, w.m, w.n, dtype=w.out_dtype, device=w.device
            )
            return x
        masked = w.api is GemmApi.MASKED_MOE or (
            w.api is GemmApi.MIXED_DTYPE_MOE and w.mixed_layout == "masked"
        )
        x.valid_m = sum(lengths)
        if masked:
            max_m = max(
                max(lengths), w.m if w.group_length == "fixed" else math.ceil(w.m * 1.3)
            )
            x.masked_m = torch.tensor(lengths, device=w.device, dtype=torch.int32)
            a_shape, out_shape = (w.group_size, max_m, w.k), (w.group_size, max_m, w.n)
        else:
            ids = []
            for expert, length in enumerate(lengths):
                padded = (
                    math.ceil(length / w.alignment_m) * w.alignment_m
                    if w.group_length == "random" or w.group_lengths is not None
                    else length
                )
                ids.extend([expert] * length + [-1] * (padded - length))
            x.m_indices = torch.tensor(ids, device=w.device, dtype=torch.int32)
            a_shape, out_shape = (len(ids), w.k), (len(ids), w.n)
        x.out = torch.empty(out_shape, dtype=w.out_dtype, device=w.device)
        # Preserve the mixed-dtype benchmark's non-negative activation distribution.
        a = (torch.rand if w.api is GemmApi.MIXED_DTYPE_MOE else torch.randn)(
            a_shape, device=w.device
        )
        if w.api is GemmApi.MIXED_DTYPE_MOE:
            x.a, x.scale_a = _group_quantize_fp8(
                a,
                (*a_shape[:-1], 1),
                (*([1] * (len(a_shape) - 1)), w.k),
                w.input_dtype,
                "K",
            )
            x.recipe_a, x.recipe_b = (1, -1), (1, w.quant_tile)
            if w.mixed_dtype == "s4fp8":
                x.b, x.scale_b = _make_int4_weights(w.group_size, w.n, w.k, w.device)
            else:
                x.b, x.residual_e8m0, x.epilogue_fp32 = _make_fp4_weights(
                    w.group_size, w.n, w.k, w.device
                )
        elif w.quantized:
            tile = 128 if masked else w.quant_tile
            x.scale_granularity_mnk = (1, tile, tile)
            x.a, x.scale_a = _group_quantize_fp8(
                a,
                (*a_shape[:-1], w.k // tile),
                (*([1] * (len(a_shape) - 1)), tile),
                w.input_dtype,
                "K",
            )
            x.b, x.scale_b = _group_quantize_fp8(
                torch.randn(w.group_size, w.n, w.k, device=w.device),
                (w.group_size, w.n // tile, w.k // tile),
                (1, tile, tile),
                w.input_dtype,
                "K",
            )
        else:
            x.a = a.to(w.input_dtype)
            x.b = torch.randn(
                w.group_size, w.n, w.k, device=w.device, dtype=w.input_dtype
            )
        return x

    def call(self, inputs: OperatorInputs) -> GemmOutputs:
        if not isinstance(inputs, GemmInputs):
            raise TypeError("expected GemmInputs")
        x = inputs
        if x.a is None or x.b is None:
            raise ValueError("GEMM requires a and b")
        api = import_module("mate.gemm")
        if x.api is GemmApi.BMM:
            result = api.bmm(
                x.a,
                x.b,
                out=x.out,
                trans_a=x.trans_a,
                trans_b=x.trans_b,
                scale_a=x.scale_a,
                scale_b=x.scale_b,
                scale_out=x.scale_out,
                recipe_a=x.recipe_a,
                recipe_b=x.recipe_b,
                c=x.c,
                out_dtype=x.out_dtype,
                fixed_scale_layout=x.fixed_scale_layout,
                backend=x.backend,
            )
            return GemmOutputs(out=result)
        if x.out is None:
            raise ValueError("MoE GEMM requires a preallocated out")
        quantized = x.a.dtype in _FP8
        a, b = (
            (x.a, x.scale_a) if quantized else x.a,
            (x.b, x.scale_b) if quantized else x.b,
        )
        if x.api is GemmApi.MIXED_DTYPE_MOE:
            b = (
                (x.b, (x.residual_e8m0, x.epilogue_fp32))
                if x.mixed_dtype == "fp4fp8"
                else (x.b, x.scale_b)
            )
            common = dict(
                mixed_dtype=x.mixed_dtype,
                backend=x.backend,
                a_quant_recipe=x.recipe_a,
                b_quant_recipe=x.recipe_b,
            )
            if x.mixed_layout == "masked":
                result = api.masked_moe_gemm_mixed_dtype(
                    a,
                    b,
                    x.masked_m,
                    x.out,
                    expect_tokens=x.expect_tokens,
                    enable_overlap=x.enable_overlap,
                    signal=x.signal,
                    **common,
                )
            else:
                result = api.ragged_moe_gemm_mixed_dtype(
                    a, b, x.m_indices, x.out, alignment_m=x.alignment_m, **common
                )
        elif x.api is GemmApi.MASKED_MOE:
            fn = api.masked_moe_gemm_8bit if quantized else api.masked_moe_gemm_16bit
            extra: dict[str, object] = (
                dict(scale_granularity_mnk=x.scale_granularity_mnk) if quantized else {}
            )
            result = fn(
                a,
                b,
                x.masked_m,
                x.out,
                expect_tokens=x.expect_tokens,
                enable_overlap=x.enable_overlap,
                signal=x.signal,
                backend=x.backend,
                **extra,
            )
        elif x.api in (GemmApi.RAGGED_M_MOE, GemmApi.RAGGED_K_MOE):
            extra = (
                dict(scale_granularity_mnk=x.scale_granularity_mnk) if quantized else {}
            )
            if x.api is GemmApi.RAGGED_M_MOE:
                fn = (
                    api.ragged_m_moe_gemm_8bit
                    if quantized
                    else api.ragged_m_moe_gemm_16bit
                )
                extra.update(alignment_m=x.alignment_m, backend=x.backend)
                indices = x.m_indices
            else:
                fn = (
                    api.ragged_k_moe_gemm_8bit
                    if quantized
                    else api.ragged_k_moe_gemm_16bit
                )
                indices = x.group_k_idx
            result = fn(
                a,
                b,
                indices,
                x.out,
                gemm_mode=x.gemm_mode,
                major_a_mode=x.major_a_mode,
                major_b_mode=x.major_b_mode,
                num_mp=x.num_mp,
                **extra,
            )
        else:
            raise ValueError("unsupported GEMM API")
        return GemmOutputs(
            out=x.out, auxiliary=tuple(result[1:]) if isinstance(result, tuple) else ()
        )

    def reference(self, inputs: OperatorInputs) -> GemmReference:
        if not isinstance(inputs, GemmInputs):
            raise TypeError("expected GemmInputs")
        x = inputs
        if x.a is None or x.b is None:
            raise ValueError("GEMM requires a and b")
        if x.api is GemmApi.BMM:
            a = x.a.float().transpose(-1, -2) if x.trans_a else x.a.float()
            b = x.b.float().transpose(-1, -2) if not x.trans_b else x.b.float()
            if x.a.dtype in _FP8 and (x.scale_a is not None or x.scale_b is not None):
                if (
                    x.scale_a is None
                    or x.scale_b is None
                    or x.recipe_a is None
                    or x.recipe_b is None
                ):
                    raise ValueError("scaled BMM requires both scales and recipes")
                a = _dequantize(a, x.scale_a, x.recipe_a)
                b = _dequantize(b, x.scale_b, x.recipe_b)
            out = torch.bmm(a, b.transpose(-1, -2))
            if x.c is not None:
                out = out + x.c.float()
            if x.scale_out is not None:
                out = (
                    out
                    / x.scale_out.float().repeat_interleave(128, -1)[
                        ..., : out.shape[-1]
                    ]
                )
                out = out.clamp(-448, 448)
            return GemmReference(
                out=out.to(x.out.dtype if x.out is not None else x.out_dtype)
            )
        if x.out is None:
            raise ValueError("MoE reference requires out shape and dtype")
        if x.api is GemmApi.RAGGED_K_MOE:
            if x.group_k_idx is None:
                raise ValueError("ragged-K requires group_k_idx")
            out = x.out.float().clone()
            cursor = scale_cursor = 0
            for expert, length in enumerate(x.group_k_idx.tolist()):
                a, b = (
                    x.a[cursor : cursor + length].float(),
                    x.b[cursor : cursor + length].float(),
                )
                blocks = math.ceil(length / x.scale_granularity_mnk[2])
                if x.scale_a is not None:
                    a *= (
                        x.scale_a[scale_cursor : scale_cursor + blocks]
                        .float()
                        .repeat_interleave(x.scale_granularity_mnk[2], 0)[:length]
                    )
                    b *= (
                        x.scale_b[scale_cursor : scale_cursor + blocks]
                        .float()
                        .repeat_interleave(x.scale_granularity_mnk[2], 0)[:length]
                    )
                out[expert] += a.T @ b
                cursor += length
                scale_cursor += blocks
            if cursor != x.a.shape[0]:
                raise ValueError("ragged-K lengths do not sum to the input K")
            return GemmReference(out=out.to(x.out.dtype))
        a = x.a.float()
        if x.api is GemmApi.RAGGED_M_MOE and x.major_a_mode == "M":
            a = a.transpose(-1, -2)
        if x.api is GemmApi.MIXED_DTYPE_MOE:
            if x.scale_a is None or x.recipe_a is None or x.recipe_b is None:
                raise ValueError("mixed GEMM requires A scales and explicit recipes")
            a = _dequantize(a, x.scale_a, x.recipe_a)
            b = _unpack_weights(x)
        else:
            b = x.b.float()
            if x.api is GemmApi.RAGGED_M_MOE and x.major_b_mode == "N":
                b = b.transpose(-1, -2)
            if x.scale_a is not None:
                gm, gn, gk = x.scale_granularity_mnk
                a, b = (
                    _dequantize(a, x.scale_a, (gm, gk)),
                    _dequantize(b, x.scale_b, (gn, gk)),
                )
        out = torch.zeros(x.out.shape, device=x.out.device, dtype=torch.float32)
        if x.masked_m is not None:
            valid = (
                torch.arange(out.shape[1], device=out.device)[None, :]
                < x.masked_m[:, None]
            )
            for expert, length in enumerate(x.masked_m.tolist()):
                out[expert, :length] = a[expert, :length] @ b[expert].T
        else:
            if x.m_indices is None:
                raise ValueError("ragged-M requires m_indices")
            ids = x.m_indices.long()
            if x.gemm_mode != "per_token":
                counts = (
                    ids
                    if x.gemm_mode == "per_expert"
                    else torch.diff(ids, prepend=ids.new_zeros(1))
                )
                ids = torch.arange(b.shape[0], device=ids.device).repeat_interleave(
                    counts
                )
            if ids.shape != (a.shape[0],):
                raise ValueError("ragged-M indices must describe each row")
            valid = ids >= 0
            if (ids[valid] >= b.shape[0]).any():
                raise ValueError("expert index out of range")
            for expert in range(b.shape[0]):
                rows = ids == expert
                out[rows] = a[rows] @ b[expert].T
        if x.logical_k == 0 or a.shape[-1] == 0:
            valid = torch.ones_like(valid)
        return GemmReference(out=out.to(x.out.dtype), valid_rows=valid)

    def verify(
        self,
        inputs: OperatorInputs,
        outputs: OperatorOutputs,
        reference: OperatorReference,
    ) -> None:
        if (
            not isinstance(inputs, GemmInputs)
            or not isinstance(outputs, GemmOutputs)
            or not isinstance(reference, GemmReference)
        ):
            raise TypeError("expected GEMM lifecycle types")
        if inputs.out is not None and outputs.out.data_ptr() != inputs.out.data_ptr():
            raise AssertionError("GEMM did not reuse the supplied output")
        if (
            outputs.out.shape != reference.out.shape
            or outputs.out.dtype != reference.out.dtype
        ):
            raise AssertionError("GEMM output shape or dtype differs from reference")
        actual, expected = outputs.out, reference.out
        if reference.valid_rows is not None:
            actual, expected = (
                actual[reference.valid_rows],
                expected[reference.valid_rows],
            )
        quantized = inputs.api is GemmApi.MIXED_DTYPE_MOE or inputs.a.dtype in _FP8
        tolerance = (
            5e-2 if quantized else 2e-2 if inputs.a.dtype == torch.bfloat16 else 5e-3
        )
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=tolerance, rtol=tolerance
        )


def _group_quantize_fp8(value, scale_shape, tile_shape, dtype, major):
    # einops cannot infer a tile dimension from a zero-sized group. Empty
    # experts need no quantization work but still carry correctly shaped scales.
    if value.numel() == 0:
        return value.to(dtype), torch.empty(
            scale_shape, device=value.device, dtype=torch.float32
        )
    return group_quantize_fp8(value, scale_shape, tile_shape, dtype, major)


def _dequantize(
    value: torch.Tensor, scale: torch.Tensor, recipe: tuple[int, int]
) -> torch.Tensor:
    if scale is None:
        raise ValueError("missing quantization scale")
    if scale.numel() == 1:
        return value.float() * scale.float()
    gm, gk = recipe
    gm = value.shape[-2] if gm == -1 else gm
    gk = value.shape[-1] if gk == -1 else gk
    expected = (
        *value.shape[:-2],
        math.ceil(value.shape[-2] / gm),
        math.ceil(value.shape[-1] / gk),
    )
    if tuple(scale.shape) != expected:
        if scale.numel() != math.prod(expected):
            raise ValueError(
                f"scale shape {tuple(scale.shape)} does not match {expected}"
            )
        scale = scale.reshape(expected)
    expanded = scale.float().repeat_interleave(gm, -2).repeat_interleave(gk, -1)
    return value.float() * expanded[..., : value.shape[-2], : value.shape[-1]]


def _make_int4_weights(groups, n, k, device):
    weight = torch.rand(groups, n, k, device=device) * 2 - 1
    blocks = weight.reshape(groups, n, k // 128, 128)
    scale = (blocks.abs().amax(-1).clamp_min(1e-4) / 7).to(torch.bfloat16)
    codes = (
        (blocks / scale.float().unsqueeze(-1))
        .round()
        .clamp(-8, 7)
        .to(torch.int8)
        .reshape(groups, n, k)
    )
    packed = ((codes[..., 0::2] & 15) | ((codes[..., 1::2] & 15) << 4)).to(torch.int8)
    return packed, scale


def _make_fp4_weights(groups, n, k, device):
    codes = (
        2
        + 2
        * (
            torch.arange(groups * n * (k // 2), device=device).reshape(
                groups, n, k // 2
            )
            % 2
        )
    ).to(torch.uint8)
    packed = (codes | (codes << 4)).view(torch.int8)
    residual = (
        127
        + torch.arange(groups * n * (k // 32), device=device).reshape(
            groups, n, k // 32
        )
        % 2
    ).to(torch.uint8)
    epilogue = (
        0.5 + torch.arange(groups * n, device=device).reshape(groups, n) % 2
    ).float()
    return packed, residual, epilogue


def _unpack_weights(x: GemmInputs) -> torch.Tensor:
    assert x.b is not None and x.a is not None and x.recipe_b is not None
    raw = x.b.to(torch.uint8)
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)
    k = x.logical_k if x.logical_k is not None else x.a.shape[-1]
    codes = codes[..., :k]
    if x.mixed_dtype == "s4fp8":
        weight = torch.where(
            codes >= 8, codes.to(torch.int16) - 16, codes.to(torch.int16)
        ).float()
        return _dequantize(weight, x.scale_b, x.recipe_b)
    if x.mixed_dtype != "fp4fp8" or x.residual_e8m0 is None or x.epilogue_fp32 is None:
        raise ValueError("FP4 requires E8M0 residual and FP32 epilogue scales")
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=codes.device)
    weight = lut[(codes & 7).long()] * torch.where(codes < 8, 1.0, -1.0)
    scale = torch.pow(2.0, x.residual_e8m0.float() - 127)
    weight = _dequantize(weight, scale, x.recipe_b).to(torch.float8_e4m3fn).float()
    return weight * x.epilogue_fp32.float().unsqueeze(-1)


# Preserve the existing BMM construction and import surface.
BmmWorkload = GemmWorkload
BmmInputs = GemmInputs
BmmOutputs = GemmOutputs
BmmReference = GemmReference
BmmOperator = GemmOperator

__all__ = [
    "GemmApi",
    "GemmWorkload",
    "GemmInputs",
    "GemmOutputs",
    "GemmReference",
    "GemmOperator",
    "BmmWorkload",
    "BmmInputs",
    "BmmOutputs",
    "BmmReference",
    "BmmOperator",
]
