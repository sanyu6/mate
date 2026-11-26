import functools
import os

import torch
from mate.arch import MP32_ARCH
from mate.gemm import transform_mxfp_sf
from mate.mate_runtime import get_arch
from mate.utils import ceil_div, round_up

__all__ = [
    "get_tma_aligned_size",
    "get_mn_major_tma_aligned_packed_ue8m0_tensor",
    "get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor",
    "transform_sf_into_required_layout",
    "get_mk_alignment_for_contiguous_layout",
    "get_col_major_tma_aligned_tensor",
    "get_mn_major_tma_aligned_tensor",
]

_MK_ALIGNMENT_ENV = "MATE_DEEPGEMM_MK_ALIGNMENT"
_VALID_MK_ALIGNMENTS = (128, 256)


def get_tma_aligned_size(x: int, element_size: int) -> int:
    alignment = 16 // element_size
    return round_up(x, alignment)


def get_mn_major_tma_aligned_packed_ue8m0_tensor(
    sf: torch.Tensor,
) -> torch.Tensor:
    ue8m0_tensor = (sf.view(torch.int32) >> 23).to(torch.uint8)

    mn, k = sf.shape[-2:]
    remove_dim = sf.dim() == 2
    if remove_dim:
        sf = sf.unsqueeze(0)
    batch = sf.shape[0]

    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = round_up(k, 4)
    padded = torch.zeros(
        (batch, aligned_mn, aligned_k), device=sf.device, dtype=torch.uint8
    )
    padded[:, :mn, :k] = ue8m0_tensor
    padded = padded.view(-1).view(torch.int32).view(batch, aligned_mn, aligned_k // 4)

    transposed = torch.empty(
        (batch, aligned_k // 4, aligned_mn), device=sf.device, dtype=torch.int32
    ).mT
    transposed[:, :, :] = padded
    aligned_x = transposed[:, :mn, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x


def get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
    sf: torch.Tensor,
    ks_tensor: torch.Tensor,
    ks: list[int],
) -> torch.Tensor:
    sf_chunks = sf.split([ceil_div(k, 128) for k in ks])
    packed_chunks = [
        get_mn_major_tma_aligned_packed_ue8m0_tensor(chunk.T).T for chunk in sf_chunks
    ]
    return torch.cat(packed_chunks)


def get_mk_alignment_for_contiguous_layout(device: torch.device | None = None) -> int:
    """Return the M-axis alignment requirement for contiguous grouped GEMM.

    Each expert segment in a contiguous-layout grouped GEMM must have its
    token count padded to a multiple of this value before being passed to
    m_grouped_{fp8,bf16}_gemm_nt_contiguous.

    Defaults to 128 on MP31 and MP32. MATE_DEEPGEMM_MK_ALIGNMENT
    can override the default to 128 or 256.
    """
    return _get_mk_alignment()


@functools.cache
def _get_mk_alignment() -> int:
    value = os.environ.get(_MK_ALIGNMENT_ENV)
    if value is None:
        return 128

    value = value.strip()
    if not value:
        raise ValueError(
            f"{_MK_ALIGNMENT_ENV} must be one of {_VALID_MK_ALIGNMENTS}, "
            "got an empty value"
        )

    try:
        alignment = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{_MK_ALIGNMENT_ENV} must be one of {_VALID_MK_ALIGNMENTS}, got {value!r}"
        ) from exc

    if alignment not in _VALID_MK_ALIGNMENTS:
        raise ValueError(
            f"{_MK_ALIGNMENT_ENV} must be one of {_VALID_MK_ALIGNMENTS}, got {value!r}"
        )

    return alignment


def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """Return a column-major, TMA-aligned view of a scale tensor.

    This function returns a contiguous copy of x without reordering data.

    Callers porting DeepGEMM code should call this on the LHS FP32 scale
    tensor (shape [m, ceil(k/128)]) before packaging it as a (fp8, scale)
    tuple — the call is a no-op here but keeps the call-site portable.
    """
    return x.contiguous()


get_mn_major_tma_aligned_tensor = get_col_major_tma_aligned_tensor


def _transform_sf_pair_into_required_layout(
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    m: int,
    n: int,
    k: int,
    recipe: tuple[int, int, int] | None,
    recipe_a: tuple[int, int] | None,
    recipe_b: tuple[int, int] | None,
    num_groups_a: int | None,
    num_groups_b: int | None,
    disable_ue8m0_cast: bool,
    *,
    dtype_a: torch.dtype | None = None,
    dtype_b: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve source recipes and prepare native SF for MP32 grouped GEMM.

    FP8 x FP4 defaults to K32 for every source scale dtype. Other operand
    pairs retain the legacy defaults selected by B's scale dtype.
    """
    is_fp8_fp4 = (
        dtype_a in (torch.float8_e4m3fn, torch.float8_e5m2) and dtype_b == torch.int8
    )
    if not is_fp8_fp4 and (recipe_a is None) != (recipe_b is None):
        raise ValueError("recipe_a and recipe_b must be provided together")
    if recipe is not None and (recipe_a is not None or recipe_b is not None):
        raise ValueError("provide either recipe or recipe_a and recipe_b")
    if recipe is None and recipe_a is None and recipe_b is None:
        # Match DeepGEMM's legacy defaults; native UINT8 uses qtile 32.
        if is_fp8_fp4:
            recipe = (1, 1, 32)
        elif sfb.dtype == torch.float32:
            recipe = (1, 128, 128)
        elif sfb.dtype == torch.int32:
            recipe = (1, 1, 128)
        elif sfb.dtype == torch.uint8:
            recipe = (1, 1, 32)
        else:
            raise TypeError("sf must have dtype float32, int32 or uint8")
    if recipe is not None:
        if not isinstance(recipe, tuple) or len(recipe) != 3:
            raise ValueError("recipe must contain three granularity values")
        recipe_a, recipe_b = (recipe[0], recipe[2]), (recipe[1], recipe[2])
    if is_fp8_fp4 and (recipe_a not in ((1, 32), (1, 128)) or recipe_b != (1, 32)):
        raise NotImplementedError("W4A8 RS supports A=(1,32/128), B=(1,32)")
    assert recipe_a is not None and recipe_b is not None
    return (
        transform_sf_into_required_layout(
            sfa,
            m,
            k,
            recipe_a,
            num_groups_a,
            disable_ue8m0_cast=disable_ue8m0_cast,
        ),
        transform_sf_into_required_layout(
            sfb,
            n,
            k,
            recipe_b,
            num_groups_b,
            disable_ue8m0_cast=disable_ue8m0_cast,
        ),
    )


def transform_sf_into_required_layout(
    sf: torch.Tensor,
    mn: int,
    k: int,
    recipe: tuple[int, int] | tuple[int, int, int],
    num_groups: int | None = None,
    is_sfa: bool = False,
    disable_ue8m0_cast: bool = False,
) -> torch.Tensor:
    """Transform scales explicitly for the current architecture.

    On MP32, accept FP32 power-of-two scales, UINT8 UE8M0 codes, or INT32
    words packing four successive K codes, least significant byte first.
    Inputs may be K-major or MN-major positive-stride tensors, including
    physical padding and storage offsets.

    ``recipe`` describes the input as ``(gran_mn, gran_k)`` or
    ``(gran_m, gran_n, gran_k)``. For a three-value recipe, ``is_sfa=True``
    selects ``gran_m`` and ``False`` selects ``gran_n``. The resulting
    per-operand recipe must be ``(1,32)``, ``(1,128)`` or ``(128,128)``.
    With ``R=ceil(mn/gran_mn)`` and ``S=ceil(k/gran_k)``, FP32/UINT8 inputs
    have shape ``[R,S]`` or ``[G,R,S]``; packed INT32 inputs replace ``S``
    with ``ceil(S/4)``. Both majors preserve this logical axis order.
    ``num_groups``, when supplied, requires a 3D input with ``G=num_groups``.
    For packed FP4 operands, ``k`` is the logical K before packing.

    Return UINT8 MN-major scales with shape ``[mn,Q]`` or ``[G,mn,Q]``,
    ``Q=ceil(k/32)``, and strides ``[1,mn]`` or ``[mn*Q,1,mn]``. Coarser
    input scales are broadcast; use ``(1,1,32)`` for the subsequent GEMM.
    Native UINT8 input with recipe ``(1,32)`` and these exact strides is
    returned unchanged. FP32 conversion requires ``disable_ue8m0_cast=False``
    and extracts exponent bits without rounding. Valid UE8M0 codes must not
    contain ``0xFF``; unused bytes in the last packed word are ignored.

    MP32 conversion uses :func:`mate.gemm.transform_mxfp_sf` and loads
    TileLang only when conversion is needed. Hardware-swizzled storage,
    independently packed K-grouped scales and PSUM gap layouts are unsupported.
    On MP31, return ``sf.contiguous()`` with the input dtype and granularity.
    """
    if get_arch(sf.device) == MP32_ARCH:
        if disable_ue8m0_cast and sf.dtype == torch.float32:
            raise ValueError(
                "MP32 FP32 scale conversion requires disable_ue8m0_cast=False"
            )
        if num_groups is not None and (sf.ndim != 3 or sf.shape[0] != num_groups):
            raise ValueError("num_groups must match the leading scale dimension")
        if not isinstance(recipe, tuple) or len(recipe) not in (2, 3):
            raise ValueError("recipe must contain two or three granularity values")
        input_recipe = (
            (recipe[0 if is_sfa else 1], recipe[2]) if len(recipe) == 3 else recipe
        )
        return transform_mxfp_sf(sf, mn, k, input_recipe)
    return get_mn_major_tma_aligned_tensor(sf)
