import torch
from typing import Optional, Tuple, Union

from mate.deep_gemm import (
    bf16_gemm_nt as mate_bf16_gemm_nt,
    fp8_einsum as mate_fp8_einsum,
    fp8_gemm_nt as mate_fp8_gemm_nt,
    fp8_gemm_nt_skip_head_mid as mate_fp8_gemm_nt_skip_head_mid,
    get_mqa_logits_metadata as mate_get_mqa_logits_metadata,
    get_paged_mqa_logits_metadata as mate_get_paged_mqa_logits_metadata,
    m_grouped_fp8_gemm_nn_contiguous as mate_m_grouped_fp8_gemm_nn_contiguous,
    tf32_hc_prenorm_gemm as mate_tf32_hc_prenorm_gemm,
)
from mate.deep_gemm import (
    mqa_logits as mate_mqa_logits,
)
from mate.deep_gemm import (
    paged_mqa_logits as mate_paged_mqa_logits,
)
from mate.gemm import (
    bmm as mate_bmm,
    masked_moe_gemm_16bit,
    masked_moe_gemm_8bit,
    masked_moe_gemm_mxfp,
    masked_moe_gemm_mixed_dtype,
    ragged_k_moe_gemm_8bit,
    ragged_k_moe_gemm_16bit,
    ragged_m_moe_gemm_16bit,
    ragged_m_moe_gemm_8bit,
    ragged_m_moe_gemm_mxfp,
    ragged_moe_gemm_mixed_dtype,
)
from mate.arch import MP31_ARCH, MP32_ARCH
from mate.jit.mubin.common import _resolve_grouped_fp8_recipe
from mate.mate_runtime import get_arch
from mate.jit.mubin.gemm.gemm_api import (
    _validate_mxfp_ks,
    ragged_k_moe_gemm_mxfp_mubin,
)
from mate.utils import ceil_div
from .utils import get_mk_alignment_for_contiguous_layout
from .utils.layout import _transform_sf_pair_into_required_layout
from .utils.w4a8 import (
    _is_packed_fp4,
    _prepare_m_grouped_w4a8_operands,
)


def bf16_gemm_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk",
    backend: str = "auto",
):
    mate_bf16_gemm_nt(
        a,
        b,
        d,
        c=c,
        compiled_dims=compiled_dims,
        backend=backend,
    )


def bf16_gemm_nn(a, b, d, c=None, compiled_dims: str = "nk", backend: str = "auto"):
    mate_bmm(
        a.unsqueeze(0),
        b.unsqueeze(0),
        d.unsqueeze(0),
        c=c.unsqueeze(0) if c is not None else None,
        trans_a=False,
        trans_b=False,
        backend=backend,
    )


def bf16_gemm_tn(a, b, d, c=None, compiled_dims: str = "mn", backend: str = "auto"):
    mate_bmm(
        a.unsqueeze(0),
        b.unsqueeze(0),
        d.unsqueeze(0),
        c=c.unsqueeze(0) if c is not None else None,
        trans_a=True,
        trans_b=False,
        backend=backend,
    )


def bf16_gemm_tt(a, b, d, c=None, compiled_dims: str = "mn", backend: str = "auto"):
    mate_bmm(
        a.unsqueeze(0),
        b.unsqueeze(0),
        d.unsqueeze(0),
        c=c.unsqueeze(0) if c is not None else None,
        trans_a=True,
        trans_b=True,
        backend=backend,
    )


def cublaslt_gemm_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
) -> None:
    mate_bmm(
        a.unsqueeze(0),
        b.unsqueeze(0),
        d.unsqueeze(0),
        c=c.unsqueeze(0) if c is not None else None,
        trans_a=False,
        trans_b=True,
        backend="mudnn",
    )


def cublaslt_gemm_nn(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
) -> None:
    mate_bmm(
        a.unsqueeze(0),
        b.unsqueeze(0),
        d.unsqueeze(0),
        c=c.unsqueeze(0) if c is not None else None,
        trans_a=False,
        trans_b=False,
        backend="mudnn",
    )


def cublaslt_gemm_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
) -> None:
    mate_bmm(
        a.unsqueeze(0),
        b.unsqueeze(0),
        d.unsqueeze(0),
        c=c.unsqueeze(0) if c is not None else None,
        trans_a=True,
        trans_b=False,
        backend="mudnn",
    )


def cublaslt_gemm_tt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
) -> None:
    mate_bmm(
        a.unsqueeze(0),
        b.unsqueeze(0),
        d.unsqueeze(0),
        c=c.unsqueeze(0) if c is not None else None,
        trans_a=True,
        trans_b=True,
        backend="mudnn",
    )


def m_grouped_bf16_gemm_nt_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    compiled_dims: str = "nk",
    use_psum_layout: Optional[bool] = None,
    ensure_zero_padding: Optional[bool] = None,
    expected_m_for_psum_layout: Optional[int] = None,
    alignment_m: Optional[int] = None,
    backend: str = "auto",
):
    _ = compiled_dims
    assert use_psum_layout is None
    assert ensure_zero_padding is None
    assert expected_m_for_psum_layout is None
    if alignment_m is None:
        alignment_m = get_mk_alignment_for_contiguous_layout()

    ragged_m_moe_gemm_16bit(
        a,
        b,
        grouped_layout,
        d,
        alignment_m=alignment_m,
        backend=backend,
    )


def m_grouped_bf16_gemm_nt_masked(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    compiled_dims: str = "nk",
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
    backend: str = "auto",
):
    res = masked_moe_gemm_16bit(
        a,
        b,
        masked_m,
        d,
        expect_tokens=expected_m,
        enable_overlap=enable_overlap,
        signal=signal,
        backend=backend,
    )

    return res[2:] if enable_overlap else None


def m_grouped_fp8_fp4_gemm_nt_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[
        torch.Tensor,
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ],
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    recipe: Optional[Tuple[int, int, int]] = None,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: Optional[bool] = None,
    use_psum_layout: Optional[bool] = None,
    ensure_zero_padding: Optional[bool] = None,
    expected_m_for_psum_layout: Optional[int] = None,
    alignment_m: Optional[int] = None,
    backend: str = "auto",
):
    """Run an M-grouped NT GEMM, converting supported source SF on MP32.

    On MP32, FP8 x FP8, FP4 x FP4 and FP8 E4M3 x FP4 accept FP32 scales,
    INT32 words packing four K UE8M0 codes, or UINT8 codes, in K-major or
    MN-major positive-stride layouts (including padding and offsets).
    A SF is 2D and B SF is 3D. Recipes use logical K, including unpacked FP4
    values. Specify either ``recipe=(gran_m,gran_n,gran_k)`` or both
    ``recipe_a``/``recipe_b``.

    For FP8 x FP8 and FP4 x FP4, each side supports input recipes
    ``(1,32)``, ``(1,128)`` and ``(128,128)``. When omitted, B's SF dtype
    selects ``(1,128,128)`` for FP32, ``(1,1,128)`` for INT32, or
    ``(1,1,32)`` for UINT8.

    ``disable_ue8m0_cast=None`` or ``False`` permits FP32 exponent extraction;
    ``True`` rejects FP32 but still permits INT32/UINT8 layout conversion.
    Conversion preserves the operands' quantized values and produces UINT8
    MN-major SF with qtile 32 for the native GEMM. Already-native SF is
    reused without loading TileLang. Explicitly preconverted SF must be
    passed with the default UINT8 recipe or ``(1,1,32)``.
    See :func:`transform_sf_into_required_layout` for SF shapes and limits.

    For FP8 E4M3 x FP4, A/B recipes default to ``(1,32)``; A also supports
    ``(1,128)``, while B requires ``(1,32)``. Output is BF16.
    PSUM and overlap are unsupported.
    """
    a_is_fp4 = _is_packed_fp4(a[0])
    b_is_fp4 = _is_packed_fp4(b[0])
    a_is_fp8 = a[0].dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    b_is_fp8 = b[0].dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    arch = get_arch(a[0].device)
    if arch == MP32_ARCH:
        if (a_is_fp8 and b_is_fp8) or (a_is_fp4 and b_is_fp4):
            assert use_psum_layout is None
            assert ensure_zero_padding is None
            assert expected_m_for_psum_layout is None
        elif a_is_fp8 and b_is_fp4:
            if compiled_dims != "nk":
                raise NotImplementedError(
                    "MP32 W4A8 currently supports compiled_dims='nk' only"
                )
            if use_psum_layout or expected_m_for_psum_layout is not None:
                raise NotImplementedError("MP32 W4A8 does not support PSUM layout")
            if ensure_zero_padding:
                raise NotImplementedError("MP32 W4A8 does not guarantee zeroed padding")
        else:
            raise ValueError(
                f"Unsupported MP32 grouped GEMM operand pair: {a[0].dtype} x {b[0].dtype}"
            )
        if not isinstance(b[1], torch.Tensor):
            raise TypeError("MP32 grouped GEMM requires B scales as a single tensor")

        sfa, sfb = _transform_sf_pair_into_required_layout(
            a[1],
            b[1],
            a[0].shape[-2],
            b[0].shape[-2],
            a[0].shape[-1] * (2 if a_is_fp4 else 1),
            recipe,
            recipe_a,
            recipe_b,
            None,
            b[0].shape[0],
            bool(disable_ue8m0_cast),
            dtype_a=a[0].dtype,
            dtype_b=b[0].dtype,
        )
        a, b = (a[0], sfa), (b[0], sfb)
        recipe, recipe_a, recipe_b = (1, 1, 32), None, None
        if alignment_m is None:
            alignment_m = get_mk_alignment_for_contiguous_layout(a[0].device)
        ragged_m_moe_gemm_mxfp(
            a,
            b,
            grouped_layout,
            d,
            recipe=recipe,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
            disable_ue8m0_cast=True,
            alignment_m=alignment_m,
            backend=backend,
        )
        return None
    elif arch == MP31_ARCH:
        # Legacy FP8 and W4A8 operand formats.
        assert use_psum_layout is None
        assert ensure_zero_padding is None
        assert expected_m_for_psum_layout is None
        assert disable_ue8m0_cast is None or disable_ue8m0_cast is True
        if a_is_fp4:
            raise NotImplementedError("Packed FP4 A requires MP32 grouped GEMM")
        if a_is_fp8 and b_is_fp4:
            w4a8_a, w4a8_b = _prepare_m_grouped_w4a8_operands(
                a, b, recipe, recipe_a, recipe_b
            )

            ragged_moe_gemm_mixed_dtype(
                w4a8_a,
                w4a8_b,
                grouped_layout,
                d,
                alignment_m=alignment_m,
                mixed_dtype="fp4fp8",
                backend=backend,
                a_quant_recipe=(1, -1),
                b_quant_recipe=(1, 32),
            )
            return None
        if b_is_fp4:
            raise ValueError("Mate W4A8 requires FP8 E4M3 A and packed FP4 B")

        assert recipe_a is None
        assert recipe_b is None
        if not a_is_fp8 or not b_is_fp8:
            raise ValueError(
                "m_grouped_fp8_fp4_gemm_nt_contiguous expects FP8 A and B tensors"
            )
        if alignment_m is None:
            alignment_m = get_mk_alignment_for_contiguous_layout(a[0].device)

        ragged_m_moe_gemm_8bit(
            a,
            b,
            grouped_layout,
            d,
            scale_granularity_mnk=recipe,
            alignment_m=alignment_m,
            backend=backend,
        )
    else:
        raise NotImplementedError(
            f"M-grouped contiguous GEMM supports MP31 and MP32, got {arch}"
        )


def m_grouped_fp8_gemm_nn_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = True,
    alignment_m: Optional[int] = None,
    backend: str = "auto",
):
    if alignment_m is None:
        alignment_m = get_mk_alignment_for_contiguous_layout()
    mate_m_grouped_fp8_gemm_nn_contiguous(
        a,
        b,
        d,
        grouped_layout,
        recipe=recipe,
        compiled_dims=compiled_dims,
        disable_ue8m0_cast=disable_ue8m0_cast,
        alignment_m=alignment_m,
        backend=backend,
    )


def m_grouped_fp8_fp4_gemm_nt_masked(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[
        torch.Tensor,
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    recipe: Optional[Tuple[int, int, int]] = None,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: Optional[bool] = None,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
    backend: str = "auto",
):
    """Run a masked NT GEMM with automatic source SF conversion on MP32.

    Both A and B SF tensors are 3D, with logical MN before the K-scale
    dimension. Supported SF dtypes, input recipes, defaults and
    ``disable_ue8m0_cast`` follow
    :func:`m_grouped_fp8_fp4_gemm_nt_contiguous`. Source scales are converted
    on the current stream on every call. Graph replay observes updated SF,
    and native UINT8 MN-major SF is reused.
    """
    a_is_fp4 = _is_packed_fp4(a[0])
    b_is_fp4 = _is_packed_fp4(b[0])
    a_is_fp8 = a[0].dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    b_is_fp8 = b[0].dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    arch = get_arch(a[0].device)
    if arch == MP32_ARCH:
        if (a_is_fp8 and b_is_fp8) or (a_is_fp4 and b_is_fp4):
            assert not enable_overlap
            assert signal is None
        elif a_is_fp8 and b_is_fp4:
            if compiled_dims != "nk":
                raise NotImplementedError(
                    "MP32 W4A8 currently supports compiled_dims='nk' only"
                )
            if enable_overlap or signal is not None:
                raise NotImplementedError(
                    "MP32 W4A8 does not support overlap or signal"
                )
        else:
            raise ValueError(
                f"Unsupported MP32 grouped GEMM operand pair: {a[0].dtype} x {b[0].dtype}"
            )
        if not isinstance(b[1], torch.Tensor):
            raise TypeError("MP32 grouped GEMM requires B scales as a single tensor")

        sfa, sfb = _transform_sf_pair_into_required_layout(
            a[1],
            b[1],
            a[0].shape[-2],
            b[0].shape[-2],
            a[0].shape[-1] * (2 if a_is_fp4 else 1),
            recipe,
            recipe_a,
            recipe_b,
            a[0].shape[0],
            b[0].shape[0],
            bool(disable_ue8m0_cast),
            dtype_a=a[0].dtype,
            dtype_b=b[0].dtype,
        )
        a, b = (a[0], sfa), (b[0], sfb)
        recipe, recipe_a, recipe_b = (1, 1, 32), None, None
        masked_moe_gemm_mxfp(
            a,
            b,
            masked_m,
            d,
            expect_tokens=expected_m,
            recipe=recipe,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
            disable_ue8m0_cast=True,
            backend=backend,
        )
        return None
    elif arch == MP31_ARCH:
        # Legacy FP8 and W4A8 operand formats.
        assert disable_ue8m0_cast is None or disable_ue8m0_cast is True
        if a_is_fp4:
            raise NotImplementedError("Packed FP4 A requires MP32 grouped GEMM")
        if a_is_fp8 and b_is_fp4:
            w4a8_a, w4a8_b = _prepare_m_grouped_w4a8_operands(
                a, b, recipe, recipe_a, recipe_b
            )

            res = masked_moe_gemm_mixed_dtype(
                w4a8_a,
                w4a8_b,
                masked_m,
                d,
                expect_tokens=expected_m,
                enable_overlap=enable_overlap,
                signal=signal,
                mixed_dtype="fp4fp8",
                backend=backend,
                a_quant_recipe=(1, -1),
                b_quant_recipe=(1, 32),
            )
            return res[2:] if enable_overlap else None
        if b_is_fp4:
            raise ValueError("Mate W4A8 requires FP8 E4M3 A and packed FP4 B")

        assert recipe_a is None
        assert recipe_b is None
        if not a_is_fp8 or not b_is_fp8:
            raise ValueError(
                "m_grouped_fp8_fp4_gemm_nt_masked expects FP8 A and B tensors"
            )

        res = masked_moe_gemm_8bit(
            a,
            b,
            masked_m,
            d,
            recipe,
            expected_m,
            enable_overlap=enable_overlap,
            signal=signal,
            backend=backend,
        )

        return res[2:] if enable_overlap else None
    else:
        raise NotImplementedError(
            f"M-grouped masked GEMM supports MP31 and MP32, got {arch}"
        )


def k_grouped_fp8_gemm_tn_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    ks_cpu: Optional[list[int]],
    grouped_layout: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "mn",
    use_psum_layout: Optional[bool] = None,
):
    assert use_psum_layout is None
    arch = get_arch(a[0].device)
    recipe = _resolve_grouped_fp8_recipe(recipe, arch, k_grouped=True)
    is_mp32 = arch == MP32_ARCH
    if is_mp32:
        assert compiled_dims == "mn"
        _validate_mxfp_ks(a[0], ks_cpu, grouped_layout)
        assert grouped_layout.cpu().tolist() == list(ks_cpu)
        # The launcher validates every tensor before initializing the accumulator.
        ragged_k_moe_gemm_mxfp_mubin(
            a, b, grouped_layout, recipe, d, ks_cpu=ks_cpu, c=c, zero_output=c is None
        )
        return d
    assert ks_cpu is None
    if a[0].shape[0] != b[0].shape[0]:
        raise ValueError("a.shape[0] and b.shape[0] must be the same")
    if c is not None:
        if c.shape != d.shape:
            raise ValueError("c and d must have the same shape")
        if c.dtype != d.dtype:
            raise ValueError("c and d must have the same dtype")
        if c.device != d.device:
            raise ValueError("c and d must be on the same device")
        if not c.is_contiguous():
            raise ValueError("c must be contiguous")
        if c is not d:
            d.copy_(c)
    else:
        d.zero_()

    ragged_k_moe_gemm_8bit(a, b, grouped_layout, d, scale_granularity_mnk=recipe)

    return d


def k_grouped_fp8_gemm_nt_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    ks_cpu: Optional[list[int]],
    grouped_layout: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 128),
    compiled_dims: str = "mn",
    use_psum_layout: bool = False,
) -> None:
    a_fp8, scale_a = a
    b_fp8, scale_b = b
    if recipe != (1, 1, 128):
        raise ValueError("recipe must be (1, 1, 128)")
    if use_psum_layout:
        raise ValueError(
            "k_grouped_fp8_gemm_nt_contiguous does not support PSUM layout"
        )
    if not ks_cpu:
        raise ValueError("ks_cpu must be nonempty")

    num_groups, m, n = d.shape
    if not grouped_layout.is_contiguous() or grouped_layout.dtype != torch.int32:
        raise ValueError("grouped_layout must be a contiguous int32 tensor")
    if grouped_layout.numel() != num_groups or len(ks_cpu) != num_groups:
        raise ValueError("grouped_layout, ks_cpu and d must have matching group counts")

    sum_k = 0
    for group_k in ks_cpu:
        if group_k % 128 != 0:
            raise ValueError("each K group must be aligned to 128")
        sum_k += group_k

    if a_fp8.numel() != sum_k * m or b_fp8.numel() != sum_k * n:
        raise ValueError("FP8 payload sizes must match ks_cpu and d")
    if not a_fp8.is_contiguous() or not b_fp8.is_contiguous():
        raise ValueError("FP8 payloads must be contiguous")
    if not d.is_contiguous():
        raise ValueError("d must be contiguous")
    if c is None or not c.is_contiguous():
        raise ValueError("c must be provided and contiguous")

    a_fp8 = a_fp8.view(-1)
    b_fp8 = b_fp8.view(-1)

    granularity_m, granularity_n, granularity_k = recipe
    recipe_a = (granularity_m, granularity_k)
    recipe_b = (granularity_n, granularity_k)

    a_offset = 0
    b_offset = 0
    scale_offset = 0

    for group, group_k in enumerate(ks_cpu):
        a_size = m * group_k
        b_size = n * group_k
        group_scale_k = ceil_div(group_k, granularity_k)

        input_a = a_fp8[a_offset : a_offset + a_size].view(1, m, group_k)
        input_b = b_fp8[b_offset : b_offset + b_size].view(1, n, group_k)
        input_scale_a = (
            scale_a[:, scale_offset : scale_offset + group_scale_k]
            .unsqueeze(0)
            .contiguous()
        )
        input_scale_b = (
            scale_b[:, scale_offset : scale_offset + group_scale_k]
            .unsqueeze(0)
            .contiguous()
        )
        output = d[group : group + 1]
        input_c = c[group : group + 1] if c is not None else None

        mate_bmm(
            input_a,
            input_b,
            output,
            scale_a=input_scale_a,
            scale_b=input_scale_b,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
            c=input_c,
            trans_a=False,
            trans_b=True,
            backend="mudnn",
        )

        a_offset += a_size
        b_offset += b_size
        scale_offset += group_scale_k


def k_grouped_bf16_gemm_tn_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    ks_cpu: Optional[list[int]],
    grouped_layout: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "mn",
    use_psum_layout: Optional[bool] = None,
):
    _ = compiled_dims
    assert ks_cpu is None
    assert use_psum_layout is None
    if a.shape[0] != b.shape[0]:
        raise ValueError("a.shape[0] and b.shape[0] must be the same")
    if c is not None:
        if c.shape != d.shape:
            raise ValueError("c and d must have the same shape")
        if c.dtype != d.dtype:
            raise ValueError("c and d must have the same dtype")
        if c.device != d.device:
            raise ValueError("c and d must be on the same device")
        if not c.is_contiguous():
            raise ValueError("c must be contiguous")
        if c is not d:
            d.copy_(c)
    else:
        d.zero_()
    ragged_k_moe_gemm_16bit(a, b, grouped_layout, d)

    return d


# Legacy DeepGEMM API.
m_grouped_fp8_gemm_nt_contiguous = m_grouped_fp8_fp4_gemm_nt_contiguous
m_grouped_fp8_gemm_nt_masked = m_grouped_fp8_fp4_gemm_nt_masked
fp8_m_grouped_gemm_nt_masked = m_grouped_fp8_gemm_nt_masked
bf16_m_grouped_gemm_nt_masked = m_grouped_bf16_gemm_nt_masked


def _fp8_fp4_gemm_dispatch(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor],
    recipe: Optional[Tuple[int, int, int]],
    recipe_a: Optional[Tuple[int, int]],
    recipe_b: Optional[Tuple[int, int]],
    major_a_mode: str,
    major_b_mode: str,
    backend: str,
):
    if recipe_a is None and recipe is None:
        recipe = (1, 1, 128) if b[1].dtype == torch.int32 else (1, 128, 128)
    if (recipe_a is None) != (recipe_b is None):
        raise ValueError("recipe_a and recipe_b must be provided together")
    if (recipe_a is None) == (recipe is None):
        raise ValueError("provide either recipe or recipe_a and recipe_b")
    if recipe is not None:
        recipe_a = (recipe[0], recipe[2])
        recipe_b = (recipe[1], recipe[2])
    mate_bmm(
        a[0].unsqueeze(0),
        b[0].unsqueeze(0),
        d.unsqueeze(0),
        scale_a=a[1].unsqueeze(0),
        scale_b=b[1].unsqueeze(0),
        c=c.unsqueeze(0) if c is not None else None,
        trans_a=major_a_mode == "M",
        trans_b=major_b_mode == "K",
        recipe_a=recipe_a,
        recipe_b=recipe_b,
        backend=backend,
    )


def fp8_fp4_gemm_nt(
    a,
    b,
    d,
    c=None,
    recipe=None,
    recipe_a=None,
    recipe_b=None,
    compiled_dims="nk",
    disable_ue8m0_cast=False,
    backend="auto",
):
    _ = compiled_dims, disable_ue8m0_cast
    _fp8_fp4_gemm_dispatch(a, b, d, c, recipe, recipe_a, recipe_b, "K", "K", backend)


def fp8_fp4_gemm_nn(
    a,
    b,
    d,
    c=None,
    recipe=None,
    recipe_a=None,
    recipe_b=None,
    compiled_dims="nk",
    disable_ue8m0_cast=False,
    backend="auto",
):
    _ = compiled_dims, disable_ue8m0_cast
    _fp8_fp4_gemm_dispatch(a, b, d, c, recipe, recipe_a, recipe_b, "K", "N", backend)


def fp8_fp4_gemm_tn(
    a,
    b,
    d,
    c=None,
    recipe=None,
    recipe_a=None,
    recipe_b=None,
    compiled_dims="mn",
    disable_ue8m0_cast=False,
    backend="auto",
):
    _ = compiled_dims, disable_ue8m0_cast
    _fp8_fp4_gemm_dispatch(a, b, d, c, recipe, recipe_a, recipe_b, "M", "N", backend)


def fp8_fp4_gemm_tt(
    a,
    b,
    d,
    c=None,
    recipe=None,
    recipe_a=None,
    recipe_b=None,
    compiled_dims="mn",
    disable_ue8m0_cast=False,
    backend="auto",
):
    _ = compiled_dims, disable_ue8m0_cast
    _fp8_fp4_gemm_dispatch(a, b, d, c, recipe, recipe_a, recipe_b, "M", "K", backend)


def fp8_gemm_nt(
    a,
    b,
    d,
    c=None,
    recipe=None,
    compiled_dims="nk",
    disable_ue8m0_cast=True,
    backend="auto",
):
    mate_fp8_gemm_nt(
        a,
        b,
        d,
        c=c,
        recipe=recipe,
        compiled_dims=compiled_dims,
        disable_ue8m0_cast=disable_ue8m0_cast,
        backend=backend,
    )


def fp8_gemm_nn(
    a,
    b,
    d,
    c=None,
    recipe=None,
    compiled_dims="nk",
    disable_ue8m0_cast=True,
    backend="auto",
):
    _ = compiled_dims, disable_ue8m0_cast
    _fp8_fp4_gemm_dispatch(a, b, d, c, recipe, None, None, "K", "N", backend)


def fp8_gemm_tn(
    a,
    b,
    d,
    c=None,
    recipe=None,
    compiled_dims="mn",
    disable_ue8m0_cast=True,
    backend="auto",
):
    _ = compiled_dims, disable_ue8m0_cast
    _fp8_fp4_gemm_dispatch(a, b, d, c, recipe, None, None, "M", "N", backend)


def fp8_gemm_tt(
    a,
    b,
    d,
    c=None,
    recipe=None,
    compiled_dims="mn",
    disable_ue8m0_cast=True,
    backend="auto",
):
    _ = compiled_dims, disable_ue8m0_cast
    _fp8_fp4_gemm_dispatch(a, b, d, c, recipe, None, None, "M", "K", backend)


def einsum(
    expr: str,
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
):
    supported_expressions = (
        "bmk,bnk->mn",
        "bhr,hdr->bhd",
        "bhd,hdr->bhr",
    )
    if expr not in supported_expressions:
        raise ValueError(f"Unsupported einsum expression: {expr}")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError("einsum expects BF16 a and b tensors")
    if d.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("einsum expects BF16 or FP32 d tensor")
    if expr != "bmk,bnk->mn" and d.dtype != torch.bfloat16:
        raise ValueError(f"einsum('{expr}') expects a BF16 d tensor")
    if c is not None:
        if expr != "bmk,bnk->mn":
            raise ValueError(f"einsum('{expr}') does not support C accumulation")
        if c.dtype != torch.float32 or d.dtype != torch.float32:
            raise ValueError("einsum with c expects FP32 c and d tensors")
        if c.data_ptr() != d.data_ptr():
            raise ValueError("einsum expects c to alias d")
    elif d.dtype == torch.float32:
        raise ValueError("einsum with FP32 d requires c")

    result = torch.einsum(expr, a, b)
    if c is None:
        d.copy_(result)
    else:
        d.add_(result)


def fp8_gemm_nt_skip_head_mid(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    head_splits: Tuple[int, int, int],
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = True,
):
    return mate_fp8_gemm_nt_skip_head_mid(
        a,
        b,
        d,
        head_splits,
        recipe=recipe,
        compiled_dims=compiled_dims,
        disable_ue8m0_cast=disable_ue8m0_cast,
    )


def fp8_einsum(
    expr: str,
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 128, 128),
):
    return mate_fp8_einsum(expr, a, b, d, c=c, recipe=recipe)


def tf32_hc_prenorm_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    sqr_sum: torch.Tensor,
    num_splits: Optional[int] = None,
):
    return mate_tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=num_splits)


def get_mqa_logits_metadata(
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
    num_kv_tokens: int,
    num_heads: int,
) -> torch.Tensor:
    r"""Get metadata for MQA logits

    Parameters
    ----------
    cu_seq_len_k_start: Tensor
        Inclusive KV start offsets, shape ``(seqlen_q,)``.
    cu_seq_len_k_end: Tensor
        Exclusive KV end offsets, shape ``(seqlen_q,)``.
    num_kv_tokens: int
        Total number of KV tokens.
    num_heads: int
        Number of query heads.

    Returns
    -------
    Tensor
        Schedule metadata.
    """
    return mate_get_mqa_logits_metadata(
        cu_seq_len_k_start,
        cu_seq_len_k_end,
        num_kv_tokens,
        num_heads,
    )


def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor,
    block_kv: int,
    num_sms: int,
    indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Get metadata for paged MQA logits

    Parameters
    ----------
    context_lens: Tensor
        Context lengths of each query, shape ``(batch_size, next_n)`` on MP32.
    block_kv: int
        Block size of the KV cache. MP32 supports 32, 64, and 128.
    num_sms: int
        Number of processors represented by the metadata.
    indices: Tensor, optional
        Optional varlen request indices.

    Returns
    -------
    Tensor
        Schedule metadata, shape ``(num_mps + 1, 2)``
    """
    return mate_get_paged_mqa_logits_metadata(
        context_lens,
        block_kv,
        num_mps=num_sms,
        indices=indices,
    )


def _unpack_int32_scale(scale: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    # DeepGEMM packs four UE8M0 bytes into one int32; Mate takes the raw bytes.
    if scale is None or scale.dtype != torch.int32:
        return scale
    return scale.view(torch.uint8).view(*scale.shape, 4)


def fp8_fp4_paged_mqa_logits(
    q: tuple[torch.Tensor, Optional[torch.Tensor]],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    clean_logits: bool = False,
    logits_dtype: torch.dtype = torch.float32,
    indices: Optional[torch.Tensor] = None,
    is_kv_scale_m_major: bool = False,
) -> torch.Tensor:
    q_values, scale_q = q
    scale_q = _unpack_int32_scale(scale_q)
    return mate_paged_mqa_logits(
        q_values,
        kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits,
        logits_dtype,
        indices,
        is_kv_scale_m_major=is_kv_scale_m_major,
        scale_q=scale_q,
    )


def fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    clean_logits: bool = False,
    indices: Optional[torch.Tensor] = None,
    logits_dtype: torch.dtype = torch.float32,
    is_kv_scale_m_major: bool = False,
    scale_q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""FP8 Paged MQA logits

    On MP32 the KV-cache footer scale is raw UE8M0 bytes and ``q`` is MXFP8, so
    ``scale_q`` must be provided; MP31 uses plain FP8 with an fp32 footer scale.

    Parameters
    ----------
    q: Tensor
        The FP8 query tensor with shape ``(batch_size, next_n, heads, index_dim)``
    kv_cache: Tensor
        The FP8 kv cache with fp32 scale (MP31) or raw UE8M0 footer (MP32),
        shape ``(num_blocks, block_size, 1, index_dim + 4)``
    weights: Tensor
        The FP32 weight tensor for each query, shape ``(batch_size * next_n, heads)``
    context_lens: Tensor
        Context lengths tensor, supports two layouts:

        - **1D** ``(batch_size,)`` — all ``next_n`` draft tokens of request ``i`` share the
          same context length ``context_lens[i]``.  The visible KV range for draft token
          ``j`` is implicitly ``[0, context_lens[i] - next_n + j]``.
        - **2D** ``(batch_size, next_n)`` — each draft token has an independent context
          length ``context_lens[i, j]``, with visible KV range ``[0, context_lens[i, j] - 1]``.
          Useful for tree-based speculative decoding (e.g. Medusa / EAGLE) where tokens
          on different branches see different KV prefixes.

        The shape is auto-detected; ``get_paged_mqa_logits_metadata`` must be called with
        the same ``context_lens`` tensor.
    block_table: Tensor
        Block table tensor with shape ``(batch_size, max_blocks)``
    schedule_meta: Tensor
        Schedule metadata tensor with shape ``(num_mps + 1, 2)``, produced by
        :func:`get_paged_mqa_logits_metadata`
    max_context_len: int
        Maximum context length
    clean_logits: bool
        Whether to zero-fill logit positions that are out of the valid KV range
    indices: Tensor, optional
        Optional varlen request indices with shape ``(total_q,)`` on MP32;
        ignored on MP31.
    logits_dtype: torch.dtype, default=torch.float32
        Output dtype; ``torch.float32`` or ``torch.bfloat16``. BF16 weights
        require BF16 logits (MP32 only).
    is_kv_scale_m_major: bool, default=False
        MP32 fused-KV footer byte order; ignored on MP31.
    scale_q: Tensor, optional
        Raw UE8M0 query scales for MP32, shape ``(batch_size, next_n, num_heads, 4)``;
        ignored on MP31.

    Returns
    -------
    Tensor
        Logits, shape ``(batch_size * next_n, max_context_len)``; dtype
        ``logits_dtype``.
    """
    scale_q = _unpack_int32_scale(scale_q)
    return mate_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits,
        logits_dtype,
        indices,
        is_kv_scale_m_major=is_kv_scale_m_major,
        scale_q=scale_q,
    )


def fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
    clean_logits: bool = False,
    max_seqlen_k: int = 0,
    scale_q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""FP8 MQA logits.

    This operator computes MQA (multi-query attention) logits for a query sequence against
    a *non-paged* KV tensor. It supports both full logits and "compressed logits" mode
    (when ``max_seqlen_k > 0``), where the output width is limited to a window size.
    On MP32 the operand scales are raw UE8M0 bytes (MXFP8) and ``scale_q`` must be
    provided; MP31 uses plain FP8 with fp32 scales.

    Parameters
    ----------
    q : torch.Tensor
        FP8 query tensor with shape ``(seq_len, heads, head_dim)`` and dtype
        ``torch.float8_e4m3fn``.
    kv : tuple[torch.Tensor, torch.Tensor]
        A tuple ``(kv_fp8, kv_scale)``:
        - ``kv_fp8``: FP8 KV tensor with shape ``(seq_len_kv, head_dim)`` and dtype
          ``torch.float8_e4m3fn``. (MQA uses a single KV head.)
        - ``kv_scale``: FP32 scale tensor with shape ``(seq_len_kv,)`` on MP31;
          raw UE8M0 bytes with shape ``(seq_len_kv, 4)`` on MP32.
    weights : torch.Tensor
        FP32 weight tensor with shape ``(seq_len, heads)`` and dtype ``torch.float32``.
    cu_seq_len_k_start : torch.Tensor
        Per-row valid KV start offsets (inclusive) for each query row, with shape
        ``(seq_len,)`` and dtype ``torch.int32``.
    cu_seq_len_k_end : torch.Tensor
        Per-row valid KV end offsets (exclusive) for each query row, with shape
        ``(seq_len,)`` and dtype ``torch.int32``.
    scale_q : torch.Tensor, optional
        Raw UE8M0 query scales for MP32, shape ``(seq_len, heads, 4)``; ignored
        on MP31.
    clean_logits : bool, default=False
        Whether to clean logits outside valid KV range. Must be ``False`` when
        ``max_seqlen_k > 0``.
    max_seqlen_k : int, default=0
        If > 0, enables compressed logits mode. The output width becomes ``max_seqlen_k``
        (a windowed logits range per row); columns at and beyond each row's window
        (``ke - ks``) are outside the contract and may contain values computed from KV
        tokens outside the window. In this mode, ``clean_logits`` must be ``False``.

    Returns
    -------
    torch.Tensor
        FP32 logits tensor with shape:
        - ``(seq_len, seq_len_kv)`` if ``max_seqlen_k == 0``
        - ``(seq_len, max_seqlen_k)`` if ``max_seqlen_k > 0``
        and dtype ``torch.float32``.
    """
    kv_fp8, kv_scale = kv
    if max_seqlen_k > 0 and clean_logits:
        raise ValueError("max_seq_len_k is not supported with clean_logits")
    scale_q = _unpack_int32_scale(scale_q)
    kv_scale = _unpack_int32_scale(kv_scale)
    return mate_mqa_logits(
        q,
        kv_fp8,
        kv_scale,
        weights,
        cu_seq_len_k_start,
        cu_seq_len_k_end,
        clean_logits,
        int(max_seqlen_k),
        scale_q=scale_q,
    )


def fp8_fp4_mqa_logits(
    q: tuple[torch.Tensor, Optional[torch.Tensor]],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
    clean_logits: bool = False,
    max_seqlen_k: int = 0,
    logits_dtype: torch.dtype = torch.float32,
    schedule_meta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    q_values, scale_q = q
    kv_values, scale_kv = kv
    scale_q = _unpack_int32_scale(scale_q)
    scale_kv = _unpack_int32_scale(scale_kv)
    return mate_mqa_logits(
        q_values,
        kv_values,
        scale_kv,
        weights,
        cu_seq_len_k_start,
        cu_seq_len_k_end,
        clean_logits,
        max_seqlen_k,
        logits_dtype,
        scale_q=scale_q,
        schedule_meta=schedule_meta,
    )
