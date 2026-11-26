import torch
from typing import Optional, Tuple

from mate.api_logging import mate_api
from mate.arch import MP32_ARCH
from mate.mate_runtime import (
    get_arch,
    get_musa_arch,
    get_num_mps as get_num_mps,
    resolve_num_mps,
    set_num_mps as set_num_mps,
)
from mate.gemm import (
    bmm as mate_bmm,
    ragged_m_moe_gemm_16bit,
    masked_moe_gemm_16bit,
    ragged_k_moe_gemm_16bit,
    ragged_m_moe_gemm_8bit,
    masked_moe_gemm_8bit,
    ragged_k_moe_gemm_8bit,
    ragged_k_moe_gemm_mxfp,
    ragged_m_moe_gemm_mxfp,
    masked_moe_gemm_mxfp,
)
from mate.jit.gemm.deep_gemm.gemm import (
    DEEP_GEMM_EPILOGUE_HEAD_SPLITS,
    GEMM_TYPE_NORMAL,
    get_deep_gemm_gemm_module,
)
from mate.jit.gemm.deep_gemm.hyperconnection import get_hyperconnection_module
from mate.jit.gemm.deep_gemm.mqa_logits import get_mqa_logits_module
from mate.jit.gemm.deep_gemm.mqa_logits_blockscale import (
    _get_mqa_logit_blockscale_alignment,
    _get_mqa_logits_blockscale_tile,
    get_mqa_logits_blockscale_module,
)
from mate.jit.gemm.deep_gemm.mqa_logits_blockscale_metadata import (
    get_mqa_logits_blockscale_metadata_module,
)
from mate.jit.gemm.deep_gemm.paged_mqa_logits import (
    get_paged_mqa_logits_metadata_module,
    get_paged_mqa_logits_module,
)
from mate.jit.gemm.deep_gemm.paged_mqa_logits_blockscale import (
    get_paged_mqa_logits_blockscale_module,
)
from mate.jit.gemm.deep_gemm.paged_mqa_logits_blockscale_metadata import (
    get_paged_mqa_logits_blockscale_metadata_module,
)
from mate.jit.runtime import ffi_to_torch
from mate.mega_moe import (
    SymmBuffer as SymmBuffer,
    fp8_fp8_mega_moe as fp8_fp8_mega_moe,
    get_symm_buffer_for_mega_moe as get_symm_buffer_for_mega_moe,
    get_token_alignment_for_mega_moe as get_token_alignment_for_mega_moe,
    transform_weights_for_mega_moe as transform_weights_for_mega_moe,
)
from mate.utils import ceil_div, round_up


_MXFP4_DTYPE = getattr(torch, "float4_e2m1fn_x2", None)


def m_grouped_bf16_gemm_nt_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    m_indices: torch.Tensor,
    alignment_m: int = 128,
    backend: str = "auto",
):
    ragged_m_moe_gemm_16bit(
        a,
        b,
        m_indices,
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
    signal: torch.Tensor = None,
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


def m_grouped_fp8_gemm_nt_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    m_indices: torch.Tensor,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = True,
    alignment_m: Optional[int] = None,
    backend: str = "auto",
):
    if not disable_ue8m0_cast:
        raise Exception("m_grouped_fp8_gemm_nt_contiguous UE8M0 cast is not supported!")

    if get_arch(a[0].device) == MP32_ARCH:
        ragged_m_moe_gemm_mxfp(
            a,
            b,
            m_indices,
            d,
            recipe=recipe,
            alignment_m=128 if alignment_m is None else alignment_m,
            backend=backend,
        )
        return None

    ragged_m_moe_gemm_8bit(
        a,
        b,
        m_indices,
        d,
        scale_granularity_mnk=recipe,
        alignment_m=alignment_m,
        backend=backend,
    )


def m_grouped_fp8_gemm_nn_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    m_indices: torch.Tensor,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = True,
    alignment_m: int = 128,
    backend: str = "auto",
):
    _ = compiled_dims
    if not disable_ue8m0_cast:
        raise ValueError(
            "m_grouped_fp8_gemm_nn_contiguous does not support UE8M0 casting"
        )
    if recipe is None:
        recipe = (1, 128, 128)
    if recipe != (1, 128, 128):
        raise ValueError(
            "m_grouped_fp8_gemm_nn_contiguous requires recipe (1, 128, 128)"
        )

    ragged_m_moe_gemm_8bit(
        a,
        b,
        m_indices,
        d,
        major_a_mode="K",
        major_b_mode="N",
        scale_granularity_mnk=recipe,
        alignment_m=alignment_m,
        backend=backend,
    )


def m_grouped_fp8_gemm_nt_masked(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = True,
    enable_overlap: bool = False,
    signal: torch.Tensor = None,
    backend: str = "auto",
):
    if not disable_ue8m0_cast:
        raise Exception("m_grouped_fp8_gemm_nt_masked UE8M0 cast is not supported!")

    if get_arch(a[0].device) == MP32_ARCH:
        if enable_overlap:
            raise ValueError("MP32 masked MXFP does not support overlap")
        masked_moe_gemm_mxfp(
            a, b, masked_m, d, expected_m, recipe=recipe, backend=backend
        )
        return None

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


def k_grouped_fp8_gemm_tn_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    ks: Optional[list[int]] = None,
    ks_tensor: Optional[torch.Tensor] = None,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
):
    if ks_tensor is None:
        if ks is None:
            raise Exception("Must give the ks whether on host or device!")
        ks_tensor = torch.tensor(ks, device="musa", dtype=torch.int32)

    if get_arch(a[0].device) == MP32_ARCH:
        ragged_k_moe_gemm_mxfp(a, b, ks_tensor, d, recipe=recipe, ks_cpu=ks)
    else:
        ragged_k_moe_gemm_8bit(a, b, ks_tensor, d, scale_granularity_mnk=recipe)

    return d


def k_grouped_bf16_gemm_tn_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    ks: Optional[list[int]] = None,
    ks_tensor: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk",
):
    if ks_tensor is None:
        if ks is None:
            raise Exception("Must give the ks whether on host or device!")
        ks_tensor = torch.tensor(ks, device="musa", dtype=torch.int32)

    ragged_k_moe_gemm_16bit(
        a,
        b,
        ks_tensor,
        d,
    )

    return d


# legacy deepgemm api
fp8_m_grouped_gemm_nt_masked = m_grouped_fp8_gemm_nt_masked
bf16_m_grouped_gemm_nt_masked = m_grouped_bf16_gemm_nt_masked


def bf16_gemm_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk",
    backend: str = "auto",
) -> torch.Tensor:
    r"""Perform BF16 NT GEMM and write the result to ``d``.

    This compatibility API computes ``d = a @ b.T`` or ``d = c + a @ b.T``
    through :func:`mate.gemm.bmm`. ``compiled_dims`` is accepted for
    DeepGEMM compatibility and is ignored.
    """
    _ = compiled_dims
    mate_bmm(
        a.unsqueeze(0),
        b.unsqueeze(0),
        d.unsqueeze(0),
        c=c.unsqueeze(0) if c is not None else None,
        trans_a=False,
        trans_b=True,
        backend=backend,
    )
    return d


def fp8_gemm_nt(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = True,
    backend: str = "auto",
) -> torch.Tensor:
    r"""Perform scaled FP8 NT GEMM and write the result to ``d``.

    This compatibility API computes ``d = a @ b.T`` or ``d = c + a @ b.T``
    through :func:`mate.gemm.bmm`. ``compiled_dims`` is accepted for
    DeepGEMM compatibility and is ignored. UE8M0 scale casting is not
    supported.
    """
    _ = compiled_dims
    if not disable_ue8m0_cast:
        raise ValueError("fp8_gemm_nt does not support UE8M0 casting")
    if recipe is None:
        recipe = (1, 128, 128)

    mate_bmm(
        a[0].unsqueeze(0),
        b[0].unsqueeze(0),
        d.unsqueeze(0),
        scale_a=a[1].unsqueeze(0),
        scale_b=b[1].unsqueeze(0),
        c=c.unsqueeze(0) if c is not None else None,
        trans_a=False,
        trans_b=True,
        recipe_a=(recipe[0], recipe[2]),
        recipe_b=(recipe[1], recipe[2]),
        backend=backend,
    )
    return d


@mate_api
def fp8_gemm_nt_skip_head_mid(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    head_splits: Tuple[int, int, int],
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = True,
):
    if not disable_ue8m0_cast:
        raise Exception("fp8_gemm_nt_skip_head_mid UE8M0 cast is not supported!")
    if len(head_splits) != 3:
        raise ValueError(f"head_splits must have 3 elements, got {head_splits}")

    left, mid, right = head_splits
    if left <= 0 or mid < 0 or right <= 0:
        raise ValueError(f"Invalid head_splits: {head_splits}")

    compact_head_dim = left + right
    if b[0].shape[0] % compact_head_dim != 0:
        raise ValueError(
            f"b.shape[0] must be divisible by left + right, got {b[0].shape[0]}"
        )

    full_n = b[0].shape[0] // compact_head_dim * (left + mid + right)
    if d.shape[0] != a[0].shape[0] or d.shape[1] != full_n:
        raise ValueError(
            f"d shape must be ({a[0].shape[0]}, {full_n}), got {tuple(d.shape)}"
        )

    dispatch_name, mod = get_deep_gemm_gemm_module(
        kind="fp8",
        gemm_type=GEMM_TYPE_NORMAL,
        config_m=a[0].shape[0],
        epilogue=DEEP_GEMM_EPILOGUE_HEAD_SPLITS,
        head_splits=head_splits,
    )
    mod.get_function(dispatch_name)(
        a[0],
        a[1],
        b[0],
        b[1],
        d,
        None,
        0,
        resolve_num_mps(a[0].device),
    )


def _validate_fp8_einsum_pair(
    name: str, value: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a tuple of (fp8_tensor, scale_tensor)")
    tensor, scale = value
    if tensor.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"{name}[0] must be an FP8 tensor")
    if scale.dtype != torch.float32:
        raise ValueError(f"{name}[1] must be a float32 scale tensor")
    if tensor.device != scale.device:
        raise ValueError(f"{name}[0] and {name}[1] must be on the same device")
    return tensor, scale


@mate_api
def fp8_einsum(
    expr: str,
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 128, 128),
) -> None:
    r"""DeepGEMM-compatible FP8 einsum.

    Supported expressions are ``"bhr,hdr->bhd"``, ``"bhd,hdr->bhr"``, and
    ``"bhd,bhr->hdr"``. The quantization recipe must be ``(1, 128, 128)`` or
    ``(1, 1, 128)``.
    """
    recipe_values = tuple(int(x) for x in recipe)
    if len(recipe_values) != 3:
        raise ValueError("fp8_einsum recipe must be a 3-tuple")
    recipe = (recipe_values[0], recipe_values[1], recipe_values[2])
    if recipe not in ((1, 128, 128), (1, 1, 128)):
        raise ValueError("fp8_einsum only supports recipe (1, 128, 128) or (1, 1, 128)")
    recipe_a = (recipe[0], recipe[2])
    recipe_b = (recipe[1], recipe[2])

    a_fp8, a_scale = _validate_fp8_einsum_pair("a", a)
    b_fp8, b_scale = _validate_fp8_einsum_pair("b", b)
    if a_fp8.device != b_fp8.device or a_fp8.device != d.device:
        raise ValueError("a, b and d must be on the same device")
    if c is not None:
        if c.device != d.device:
            raise ValueError("c must be on the same device as d")
        if c.dtype != torch.float32 or d.dtype != torch.float32:
            raise ValueError("fp8_einsum with c expects fp32 c and d tensors")

    if expr == "bhr,hdr->bhd":
        if a_fp8.dim() != 3 or b_fp8.dim() != 3 or d.dim() != 3:
            raise ValueError("fp8_einsum('bhr,hdr->bhd') expects 3D tensors")
        batch, heads, r_dim = a_fp8.shape
        h_b, d_dim, r_b = b_fp8.shape
        if heads != h_b or r_dim != r_b or tuple(d.shape) != (batch, heads, d_dim):
            raise ValueError("expected a[b,h,r], b[h,d,r] and d[b,h,d]")
        c_view = c.permute(1, 0, 2) if c is not None else None
        mate_bmm(
            a_fp8.permute(1, 0, 2),
            b_fp8,
            d.permute(1, 0, 2),
            scale_a=a_scale.permute(1, 0, 2),
            scale_b=b_scale,
            c=c_view,
            trans_a=False,
            trans_b=True,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )
        return None

    if expr == "bhd,hdr->bhr":
        if a_fp8.dim() != 3 or b_fp8.dim() != 3 or d.dim() != 3:
            raise ValueError("fp8_einsum('bhd,hdr->bhr') expects 3D tensors")
        batch, heads, d_dim = a_fp8.shape
        h_b, d_b, r_dim = b_fp8.shape
        if heads != h_b or d_dim != d_b or tuple(d.shape) != (batch, heads, r_dim):
            raise ValueError("expected a[b,h,d], b[h,d,r] and d[b,h,r]")
        c_view = c.permute(1, 0, 2) if c is not None else None
        mate_bmm(
            a_fp8.permute(1, 0, 2),
            b_fp8,
            d.permute(1, 0, 2),
            scale_a=a_scale.permute(1, 0, 2),
            scale_b=b_scale,
            c=c_view,
            trans_a=False,
            trans_b=False,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )
        return None

    if expr == "bhd,bhr->hdr":
        if a_fp8.dim() != 3 or b_fp8.dim() != 3 or d.dim() != 3:
            raise ValueError("fp8_einsum('bhd,bhr->hdr') expects 3D tensors")
        batch, heads, d_dim = a_fp8.shape
        b_batch, h_b, r_dim = b_fp8.shape
        if batch != b_batch or heads != h_b or tuple(d.shape) != (heads, d_dim, r_dim):
            raise ValueError("expected a[b,h,d], b[b,h,r] and d[h,d,r]")
        mate_bmm(
            a_fp8.permute(1, 0, 2),
            b_fp8.permute(1, 0, 2),
            d,
            scale_a=a_scale.permute(1, 0, 2),
            scale_b=b_scale.permute(1, 0, 2),
            c=c,
            trans_a=True,
            trans_b=False,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )
        return None

    raise ValueError(f"Unsupported fp8_einsum expression: {expr}")


def _get_paged_mqa_logits_fp8_metadata(
    context_lens: torch.Tensor, block_kv: int, num_mps: int = 0
) -> torch.Tensor:
    r"""Get metadata for paged MQA logits

    Parameters
    ----------
    context_lens: Tensor
        Context lengths of each query, shape ``(batch_size)``
    block_kv: Tensor
        Block size of kv cache, **must be 64 now**.
    num_mps: int
        Number of MP to execute. 0 means use all MPs of the current device

    Returns
    -------
    Tensor
        Schedule metadata, shape ``(num_mps + 1, 2)``
    """
    num_mps = resolve_num_mps(context_lens.device, num_mps)
    schedule_meta = torch.empty(
        (num_mps + 1, 2), device=context_lens.device, dtype=torch.int32
    )
    batch_size = context_lens.shape[0]
    get_paged_mqa_logits_metadata_module(batch_size).get_function(
        "get_paged_mqa_logits_metadata"
    )(context_lens, block_kv, schedule_meta)
    return schedule_meta


def _get_paged_mqa_logits_blockscale_metadata(
    seqlens_kv: torch.Tensor,
    tile_kv: int,
    num_mps: Optional[int] = None,
    indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Get MP32 paged block-scaled MQA logits schedule metadata.

    Parameters
    ----------
    seqlens_kv : torch.Tensor
        Int32 context lengths with shape ``(num_requests, next_n)`` for regular
        requests, or ``(num_requests,)`` with one length per request. In varlen
        mode, shape ``(total_q,)`` or ``(total_q, 1)`` stores one length per Q
        token.
    tile_kv : int
        KV tile size used by the paged MQA compute kernel. The current MP32
        paged blockscale compute kernel uses 512; other values are ignored.
    num_mps : int, optional
        Number of MPs represented by the metadata. ``None`` uses all MPs on
        the current device.
    indices : torch.Tensor, optional
        Contiguous int32 tensor with shape ``(total_q,)``. Each maximal run of
        equal values defines one varlen request. Varlen mode requires
        ``next_n=1``.

    Returns
    -------
    torch.Tensor
        Int32 metadata with shape ``(num_mps + 1, 2)``. Regular-request rows
        contain ``(request_idx, tile_kv_idx_in_request)``. Varlen rows contain
        ``(token_q_idx, tile_kv_idx_in_request)``.
    """
    is_context_lens_2d = seqlens_kv.dim() == 2
    is_varlen = indices is not None
    next_n = seqlens_kv.shape[1] if is_context_lens_2d else 1

    num_mps = resolve_num_mps(seqlens_kv.device, num_mps)
    metadata = torch.empty(
        (num_mps + 1, 2), device=seqlens_kv.device, dtype=torch.int32
    )
    metadata_module = get_paged_mqa_logits_blockscale_metadata_module(
        next_n,
        is_context_lens_2d,
        tile_kv,
        num_mps,
        is_varlen,
    )
    metadata_function = metadata_module.get_function(
        "get_paged_mqa_logits_blockscale_metadata"
    )
    metadata_function(seqlens_kv, indices, metadata)
    return metadata


@mate_api
def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor,
    block_kv: int,
    num_mps: Optional[int] = None,
    indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Get paged MQA logits schedule metadata.

    Dispatches to the MP31 or MP32 implementation for ``context_lens``'s
    device. MP31 requires ``block_kv == 64`` and ignores ``indices``;
    ``indices`` selects MP32 varlen mode.

    Parameters
    ----------
    context_lens : torch.Tensor
        Int32 context lengths. MP31 accepts ``(batch_size,)`` or
        ``(batch_size, next_n)``; MP32 accepts ``(num_requests, next_n)``,
        ``(num_requests,)``, and varlen ``(total_q,)`` / ``(total_q, 1)``.
    block_kv : int
        KV block size; MP32 uses the fixed 512 KV tile and ignores this value.
    num_mps : int, optional
        Number of MPs represented by the metadata; ``None`` uses all MPs.
    indices : torch.Tensor, optional
        MP32 varlen request indices with shape ``(total_q,)``.

    Returns
    -------
    torch.Tensor
        Int32 schedule metadata with shape ``(num_mps + 1, 2)``.
    """
    if get_musa_arch(context_lens) == MP32_ARCH:
        return _get_paged_mqa_logits_blockscale_metadata(
            context_lens, block_kv, num_mps, indices
        )
    return _get_paged_mqa_logits_fp8_metadata(context_lens, block_kv, num_mps or 0)


def _get_mqa_logits_blockscale_metadata(
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
    num_kv_tokens: int,
    num_heads: int,
) -> torch.Tensor:
    r"""Get MP32 block-scaled MQA logits schedule metadata (non-paged).

    Parameters
    ----------
    cu_seq_len_k_start : torch.Tensor
        Int32 inclusive KV start offsets with shape ``(seqlen_q,)``.
    cu_seq_len_k_end : torch.Tensor
        Int32 exclusive KV end offsets with shape ``(seqlen_q,)``.
    num_kv_tokens : int
        Total KV length.
    num_heads : int
        Number of query heads; selects the Q tile used by the compute kernel.

    Returns
    -------
    torch.Tensor
        Int32 schedule metadata workspace.
    """
    if not (0 < num_heads <= 128 and num_heads % 4 == 0):
        raise ValueError(
            f"num_heads must be a multiple of 4 up to 128, got {num_heads}"
        )

    tile_q, _, tile_kv, _, _ = _get_mqa_logits_blockscale_tile(num_heads)
    num_q_tokens = cu_seq_len_k_start.shape[0]
    num_q_tiles = ceil_div(num_q_tokens, tile_q)

    num_mps = resolve_num_mps(cu_seq_len_k_start.device, None)
    required_words = ((3 * num_mps + 1) // 2) * 2 + 2 * num_q_tiles
    schedule_meta = torch.empty(
        (required_words,), device=cu_seq_len_k_start.device, dtype=torch.int32
    )

    metadata_module = get_mqa_logits_blockscale_metadata_module(
        tile_q,
        tile_kv,
        num_mps,
    )
    metadata_function = metadata_module.get_function(
        "get_mqa_logits_blockscale_metadata"
    )
    metadata_function(
        cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, schedule_meta
    )
    return schedule_meta


@mate_api
def get_mqa_logits_metadata(
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
    num_kv_tokens: int,
    num_heads: int,
) -> torch.Tensor:
    r"""Get non-paged MQA logits schedule metadata for MP32.

    Parameters
    ----------
    cu_seq_len_k_start : torch.Tensor
        Int32 inclusive KV start offsets with shape ``(seqlen_q,)``.
    cu_seq_len_k_end : torch.Tensor
        Int32 exclusive KV end offsets with shape ``(seqlen_q,)``.
    num_kv_tokens : int
        Total KV length.
    num_heads : int
        Number of query heads.

    Returns
    -------
    torch.Tensor
        Int32 schedule metadata consumed by the MP32 MQA logits kernels.
    """
    if get_musa_arch(cu_seq_len_k_start) != MP32_ARCH:
        raise ValueError("get_mqa_logits_metadata is only supported on MP32")
    return _get_mqa_logits_blockscale_metadata(
        cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, num_heads
    )


def _paged_mqa_logits_blockscale(
    q: torch.Tensor,
    scale_q: torch.Tensor,
    kv_cache_with_scale: torch.Tensor,
    weights: torch.Tensor,
    seqlens_kv: torch.Tensor,
    page_table: torch.Tensor,
    metadata: torch.Tensor,
    max_seqlen_kv: int,
    clean_logits: bool = False,
    logits_dtype: torch.dtype = torch.float32,
    indices: Optional[torch.Tensor] = None,
    is_kv_scale_m_major: bool = False,
) -> torch.Tensor:
    r"""MXFP8 or MXFP4 block-scaled paged MQA logits for MP32.

    Parameters
    ----------
    q : torch.Tensor
        MXFP8 values have shape ``(batch_size, next_n, num_heads, headdim)``.
        MXFP4 values pack two E2M1 elements per ``torch.float4_e2m1fn_x2``
        element and have shape ``(batch_size, next_n, num_heads, headdim / 2)``.
    scale_q : torch.Tensor
        UE8M0 scales with shape
        ``(batch_size, next_n, num_heads, 4)`` and dtype ``torch.uint8``.
        Q scale layout is inferred from its strides.
    kv_cache_with_scale : torch.Tensor
        Fused uint8 KV cache with shape
        ``(num_blocks, page_size, 1, packed_headdim + 4)``. Within each block,
        all KV value bytes precede four scale bytes per token.
    weights : torch.Tensor
        ``torch.float32`` or ``torch.bfloat16`` weights with shape
        ``(batch_size * next_n, num_heads)``. BF16 requires MP32 and
        ``logits_dtype=torch.bfloat16``.
    seqlens_kv : torch.Tensor
        Int32 context lengths with shape ``(batch_size, next_n)``. A
        ``(batch_size,)`` tensor is also accepted; the visible KV range for
        draft token ``j`` of request ``i`` is then implicitly
        ``[0, seqlens_kv[i] - next_n + j]``.
    page_table : torch.Tensor
        Int32 physical-page indices with shape ``(batch_size, max_pages)``.
        Varlen mode uses one row per Q token, repeating the request's row for
        every token in the same run.
    metadata : torch.Tensor
        Int32 schedule metadata with shape ``(num_mps + 1, 2)``.
    max_seqlen_kv : int
        Logical output width.
    clean_logits : bool, default=False
        Set positions outside each row's sequence length to negative infinity.
    logits_dtype : torch.dtype, default=torch.float32
        Output dtype; ``torch.float32`` or ``torch.bfloat16``. BF16 requires
        BF16 ``weights``.
    indices : torch.Tensor, optional
        Contiguous int32 tensor with shape ``(total_q,)``. Equal-value runs
        define varlen requests and select varlen mode, which requires
        ``next_n=1``. The final context length in each run determines its KV
        split count.
    is_kv_scale_m_major : bool, default=False
        Selects the physical byte order of the scale footer in each fused KV
        page. Q scale byte order is inferred from ``scale_q`` strides.

    Returns
    -------
    torch.Tensor
        Logits with shape ``(batch_size * next_n, max_seqlen_kv)`` and dtype
        ``logits_dtype``.
    """
    if q.dtype == torch.float8_e4m3fn:
        pack_factor = 1
    elif _MXFP4_DTYPE is not None and q.dtype == _MXFP4_DTYPE:
        pack_factor = 2
    else:
        raise ValueError("q values must use E4M3 MXFP8 or packed E2M1 MXFP4")

    batch_size, next_n, num_heads, physical_headdim = q.shape
    headdim = physical_headdim * pack_factor
    if scale_q.dtype == torch.float32:
        raise ValueError(
            "MP32 scale_q must use raw UE8M0 bytes; fp32 scales are MP31-only"
        )
    if weights.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("weights must be float32 or bfloat16")
    if logits_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("logits_dtype must be torch.float32 or torch.bfloat16")
    if weights.dtype == torch.bfloat16 and logits_dtype != torch.bfloat16:
        raise ValueError("bfloat16 weights require bfloat16 logits")
    page_size = kv_cache_with_scale.shape[1]
    total_q = batch_size * next_n
    is_varlen = indices is not None

    page_table_capacity = page_table.shape[1] * page_size
    if max_seqlen_kv <= 0 or max_seqlen_kv > page_table_capacity:
        raise ValueError(
            "max_seqlen_kv must be in [1, page_table.shape[1] * page_size]"
        )

    logits_stride = round_up(max_seqlen_kv, 512)
    logits_storage = torch.empty(
        (total_q, logits_stride), dtype=logits_dtype, device=q.device
    )
    out = logits_storage[:, :max_seqlen_kv]

    paged_module = get_paged_mqa_logits_blockscale_module(
        num_heads,
        headdim,
        page_size,
        next_n,
        seqlens_kv.dim() == 2,
        is_varlen=is_varlen,
        clean_logits=clean_logits,
        qkv_dtype=q.dtype,
        is_kv_scale_m_major=is_kv_scale_m_major,
        weights_dtype=weights.dtype,
        logits_dtype=logits_dtype,
    )
    paged_function = paged_module.get_function("paged_mqa_logits_blockscale")
    arguments = (
        q,
        scale_q,
        kv_cache_with_scale,
        weights,
        seqlens_kv,
        page_table,
        metadata,
    )
    paged_function(*arguments, indices, out)
    return out


@mate_api
def fp8_paged_mqa_logits(
    q: torch.Tensor,
    fused_kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    clean_logits: bool,
    logits_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""FP8 Paged MQA logits

    Parameters
    ----------
    q: Tensor
        The FP8 query tensor with shape ``(batch_size, next_n, heads, index_dim)``
    fused_kv_cache: Tensor
        The FP8 kv cache with fp32 scale, shape ``(num_blocks, block_size, 1, index_dim + 4)``
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
    logits_dtype: torch.dtype, default=torch.float32
        Output dtype; ``torch.float32`` or ``torch.bfloat16``.

    Returns
    -------
    Tensor
        Logits with dtype ``logits_dtype``, shape
        ``(batch_size * next_n, max_context_len)``
    """
    next_n = q.shape[1]
    num_heads = q.shape[2]
    head_dim = q.shape[3]
    block_kv = fused_kv_cache.shape[1]
    is_context_lens_2d = context_lens.dim() == 2

    if logits_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("logits_dtype must be torch.float32 or torch.bfloat16")

    return ffi_to_torch(
        get_paged_mqa_logits_module(
            next_n,
            num_heads,
            head_dim,
            block_kv,
            is_context_lens_2d,
            logits_dtype,
        ).get_function("fp8_paged_mqa_logits")(
            q,
            fused_kv_cache,
            weights,
            context_lens,
            block_table,
            schedule_meta,
            max_context_len,
            clean_logits,
        )
    )


@mate_api
def paged_mqa_logits(
    q: torch.Tensor,
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
    scale_q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Paged MQA logits, dispatching to the MP31 or MP32 implementation.

    MP32 requires MXFP8/MXFP4 ``q`` with ``scale_q``; MP31 takes plain FP8
    ``q`` with the legacy fused KV cache and ignores ``scale_q``/``indices``.

    Parameters
    ----------
    q : torch.Tensor
        Query values.
    kv_cache : torch.Tensor
        Fused KV cache (block-scaled on MP32, FP8 with fp32 scale on MP31).
    weights : torch.Tensor
        ``torch.float32`` or ``torch.bfloat16`` weights with shape
        ``(batch_size * next_n, num_heads)``. BF16 requires MP32 and
        ``logits_dtype=torch.bfloat16``.
    context_lens : torch.Tensor
        Int32 context lengths (MP32: ``(batch_size, next_n)`` or 1D).
    block_table : torch.Tensor
        Int32 physical-page indices with shape ``(batch_size, max_pages)``.
    schedule_meta : torch.Tensor
        Int32 schedule metadata from :func:`get_paged_mqa_logits_metadata`.
    max_context_len : int
        Logical output width.
    clean_logits : bool, default=False
        Set positions outside each row's sequence length to negative infinity.
    logits_dtype : torch.dtype, default=torch.float32
        Output dtype; ``torch.float32`` or ``torch.bfloat16``. BF16 weights
        require BF16 logits (MP32 only).
    indices : torch.Tensor, optional
        MP32 varlen request indices with shape ``(total_q,)``.
    is_kv_scale_m_major : bool, default=False
        MP32 fused-KV scale-footer byte order. Query-scale layout is inferred
        from ``scale_q`` strides.
    scale_q : torch.Tensor, optional
        Raw UE8M0 query scales for MP32 with shape
        ``(batch_size, next_n, num_heads, 4)``.

    Returns
    -------
    torch.Tensor
        Logits with shape ``(batch_size * next_n, max_context_len)`` and dtype
        ``logits_dtype``.
    """
    if get_musa_arch(q) == MP32_ARCH:
        if scale_q is None:
            raise ValueError("MP32 paged MQA logits requires MX scales")
        return _paged_mqa_logits_blockscale(
            q,
            scale_q,
            kv_cache,
            weights,
            context_lens,
            block_table,
            schedule_meta,
            max_context_len,
            clean_logits,
            logits_dtype,
            indices,
            is_kv_scale_m_major,
        )
    return fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits,
        logits_dtype,
    )


@mate_api
def fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
    clean_logits: bool = False,
    max_seqlen_k: int = 0,
    logits_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""FP8 MQA logits.

    This operator computes MQA (multi-query attention) logits for a query sequence against
    a *non-paged* KV tensor. It supports both full logits and "compressed logits" mode
    (when ``max_seqlen_k > 0``), where the output width is limited to a window size.

    Parameters
    ----------
    q : torch.Tensor
        FP8 query tensor with shape ``(seq_len, heads, head_dim)`` and dtype
        ``torch.float8_e4m3fn``.
    kv : tuple[torch.Tensor, torch.Tensor]
        A tuple ``(kv_fp8, kv_scale)`` containing:

        - ``kv_fp8``: FP8 KV tensor with shape ``(seq_len_kv, head_dim)`` and dtype
          ``torch.float8_e4m3fn``. (MQA uses a single KV head.)
        - ``kv_scale``: FP32 scale tensor with shape ``(seq_len_kv,)`` and dtype
          ``torch.float32``.
    weights : torch.Tensor
        FP32 weight tensor with shape ``(seq_len, heads)`` and dtype ``torch.float32``.
    cu_seq_len_k_start : torch.Tensor
        Per-row valid KV start offsets (inclusive) for each query row, with shape
        ``(seq_len,)`` and dtype ``torch.int32``.
    cu_seq_len_k_end : torch.Tensor
        Per-row valid KV end offsets (exclusive) for each query row, with shape
        ``(seq_len,)`` and dtype ``torch.int32``.
    clean_logits : bool, default=False
        Whether to clean logits outside valid KV range. Must be ``False`` when
        ``max_seqlen_k > 0``.
    max_seqlen_k : int, default=0
        If > 0, enables compressed logits mode. The output width becomes ``max_seqlen_k``
        (a windowed logits range per row). In this mode, ``clean_logits`` must be ``False``.
    logits_dtype : torch.dtype, default=torch.float32
        Output dtype; ``torch.float32`` or ``torch.bfloat16``.

    Returns
    -------
    torch.Tensor
        Logits tensor with dtype ``logits_dtype`` and shape:

        - ``(seq_len, seq_len_kv)`` if ``max_seqlen_k == 0``
        - ``(seq_len, max_seqlen_k)`` if ``max_seqlen_k > 0``
    """
    kv_fp8, kv_scale = kv
    if kv_scale.dtype == torch.int32:
        raise ValueError("int32 packed scales are MP32-only")
    if max_seqlen_k > 0 and clean_logits:
        raise ValueError("max_seqlen_k is not supported with clean_logits")
    if logits_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("logits_dtype must be torch.float32 or torch.bfloat16")

    seq_len = q.shape[0]
    seq_len_kv = kv_fp8.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    compressed_logits = max_seqlen_k > 0
    num_mps = resolve_num_mps(q.device)

    return ffi_to_torch(
        get_mqa_logits_module(
            seq_len,
            seq_len_kv,
            num_heads,
            head_dim,
            compressed_logits,
            num_mps,
            logits_dtype,
        ).get_function("fp8_mqa_logits")(
            q,
            kv_fp8,
            weights,
            cu_seq_len_k_start,
            cu_seq_len_k_end,
            kv_scale,
            clean_logits,
            int(max_seqlen_k),
        )
    )


def _mqa_logits_blockscale(
    q: torch.Tensor,
    kv: torch.Tensor,
    scale_q: torch.Tensor,
    scale_kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
    clean_logits: bool = False,
    max_seqlen_k: int = 0,
    logits_dtype: torch.dtype = torch.float32,
    schedule_meta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""MXFP8 or MXFP4 block-scaled MQA logits for MP32.

    Parameters
    ----------
    q : torch.Tensor
        MXFP8 values have shape
        ``(seqlen_q, num_heads, headdim)`` and dtype ``torch.float8_e4m3fn``.
        MXFP4 values pack two E2M1 elements per ``torch.float4_e2m1fn_x2``
        element and have shape ``(seqlen_q, num_heads, headdim / 2)``.
    kv : torch.Tensor
        Values using the same MX format as ``q``. MXFP8 values have shape
        ``(seqlen_kv, headdim)``; packed MXFP4 values have shape
        ``(seqlen_kv, headdim / 2)``.
    scale_q : torch.Tensor
        UE8M0 scales with shape ``(seqlen_q, num_heads, 4)`` and dtype ``torch.uint8``.
        Scale layout is inferred from tensor strides.
    scale_kv : torch.Tensor
        UE8M0 scales with shape ``(seqlen_kv, 4)`` and dtype ``torch.uint8``.
        Scale layout is inferred from tensor strides.
    weights : torch.Tensor
        ``torch.float32`` or ``torch.bfloat16`` weights with shape
        ``(seqlen_q, num_heads)``. BF16 requires MP32 and
        ``logits_dtype=torch.bfloat16``.
    cu_seq_len_k_start : torch.Tensor
        Inclusive valid KV start offsets with shape ``(seqlen_q,)`` and dtype
        ``torch.int32``.
    cu_seq_len_k_end : torch.Tensor
        Exclusive valid KV end offsets with shape ``(seqlen_q,)`` and dtype
        ``torch.int32``.
    clean_logits : bool, default=False
        Whether to fill positions outside each valid KV range with negative
        infinity. Must be ``False`` in compressed mode.
    max_seqlen_k : int, default=0
        Enables compressed logits when greater than zero and sets the logical
        output width. Columns at and beyond each row's ``ke - ks`` (up to
        ``max_seqlen_k``) are outside the contract and may contain values
        computed from KV tokens outside the window.
    logits_dtype : torch.dtype, default=torch.float32
        Output dtype; ``torch.float32`` or ``torch.bfloat16``. BF16 requires
        BF16 ``weights``.
    schedule_meta : torch.Tensor, optional
        MP32 schedule metadata produced by ``get_mqa_logits_metadata``;
        ``None`` selects the grid-stride schedule.

    Returns
    -------
    torch.Tensor
        Logits with shape ``(seqlen_q, seqlen_kv)`` in full mode or
        ``(seqlen_q, max_seqlen_k)`` in compressed mode, with dtype
        ``logits_dtype``.

    """
    if q.dtype == torch.float8_e4m3fn:
        pack_factor = 1
    elif _MXFP4_DTYPE is not None and q.dtype == _MXFP4_DTYPE:
        pack_factor = 2
    else:
        raise ValueError("q values must use E4M3 MXFP8 or packed E2M1 MXFP4")

    seqlen_q = q.shape[0]
    total_seqlen_kv = kv.shape[0]
    num_heads = q.shape[1]
    headdim = q.shape[2] * pack_factor
    if scale_q.dtype == torch.float32:
        raise ValueError(
            "MP32 scale_q must use raw UE8M0 bytes; fp32 scales are MP31-only"
        )
    if scale_kv.dtype == torch.float32:
        raise ValueError(
            "MP32 scale_kv must use raw UE8M0 bytes; fp32 scales are MP31-only"
        )
    if weights.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("weights must be float32 or bfloat16")
    if logits_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("logits_dtype must be torch.float32 or torch.bfloat16")
    if weights.dtype == torch.bfloat16 and logits_dtype != torch.bfloat16:
        raise ValueError("bfloat16 weights require bfloat16 logits")
    compressed_logits = max_seqlen_k > 0

    seqlen_kv = max_seqlen_k if compressed_logits else total_seqlen_kv
    aligned_seqlen_q, aligned_seqlen_kv = _get_mqa_logit_blockscale_alignment(
        seqlen_q, seqlen_kv, num_heads, compressed_logits
    )
    logits_storage = torch.empty(
        (aligned_seqlen_q, aligned_seqlen_kv),
        dtype=logits_dtype,
        device=q.device,
    )
    out = logits_storage[:seqlen_q, :seqlen_kv]

    get_mqa_logits_blockscale_module(
        num_heads,
        headdim,
        compressed_logits,
        q.dtype,
        weights_dtype=weights.dtype,
        logits_dtype=logits_dtype,
        use_schedule=schedule_meta is not None,
        num_mps=resolve_num_mps(q.device),
    ).get_function("mqa_logits_blockscale")(
        q,
        kv,
        scale_q,
        scale_kv,
        weights,
        cu_seq_len_k_start,
        cu_seq_len_k_end,
        out,
        clean_logits,
        schedule_meta,
    )
    return out


@mate_api
def mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    scale_kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
    clean_logits: bool = False,
    max_seqlen_k: int = 0,
    logits_dtype: torch.dtype = torch.float32,
    scale_q: Optional[torch.Tensor] = None,
    schedule_meta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""MQA logits, dispatching to the MP31 or MP32 implementation.

    MP32 requires MXFP8/MXFP4 ``q``/``kv`` with ``scale_q``/``scale_kv``; MP31
    takes plain FP8 ``q`` with fp32 ``scale_kv`` and ignores ``scale_q``.

    Parameters
    ----------
    q : torch.Tensor
        Query values.
    kv : torch.Tensor
        KV values using the same format as ``q``.
    scale_kv : torch.Tensor
        KV scales; raw UE8M0 ``(seqlen_kv, 4)`` on MP32,
        fp32 ``(seqlen_kv,)`` on MP31.
    weights : torch.Tensor
        ``torch.float32`` or ``torch.bfloat16`` weights with shape
        ``(seqlen_q, num_heads)``. BF16 requires MP32 and
        ``logits_dtype=torch.bfloat16``.
    cu_seq_len_k_start : torch.Tensor
        Inclusive valid KV start offsets with shape ``(seqlen_q,)``.
    cu_seq_len_k_end : torch.Tensor
        Exclusive valid KV end offsets with shape ``(seqlen_q,)``.
    clean_logits : bool, default=False
        Fill positions outside each valid KV range with negative infinity.
    max_seqlen_k : int, default=0
        Enables compressed logits when greater than zero: each row holds its
        ``[ks, ke)`` window packed from column 0. On MP32, columns at and beyond
        ``ke - ks`` (up to ``max_seqlen_k``) are outside the contract and may
        contain values computed from KV tokens outside the window.
    logits_dtype : torch.dtype, default=torch.float32
        Output dtype; ``torch.float32`` or ``torch.bfloat16``. BF16 weights
        require BF16 logits (MP32 only).
    scale_q : torch.Tensor, optional
        Raw MP32 query scales, ``(seqlen_q, num_heads, 4)``; ignored on MP31.
    schedule_meta : torch.Tensor, optional
        MP32 schedule metadata produced by ``get_mqa_logits_metadata``; must
        be ``None`` on MP31.

    Returns
    -------
    torch.Tensor
        Logits with shape ``(seqlen_q, seqlen_kv)`` or
        ``(seqlen_q, max_seqlen_k)`` in compressed mode, with dtype
        ``logits_dtype``.
    """
    if get_musa_arch(q) == MP32_ARCH:
        if scale_q is None:
            raise ValueError("MP32 MQA logits requires MX scales")
        return _mqa_logits_blockscale(
            q,
            kv,
            scale_q,
            scale_kv,
            weights,
            cu_seq_len_k_start,
            cu_seq_len_k_end,
            clean_logits,
            max_seqlen_k,
            logits_dtype,
            schedule_meta,
        )
    if schedule_meta is not None:
        raise ValueError("schedule_meta is only supported on MP32")
    return fp8_mqa_logits(
        q,
        (kv, scale_kv),
        weights,
        cu_seq_len_k_start,
        cu_seq_len_k_end,
        clean_logits,
        max_seqlen_k,
        logits_dtype,
    )


@mate_api
def tf32_hc_prenorm_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    sqr_sum: torch.Tensor,
    num_splits: Optional[int] = None,
) -> None:
    r"""TF32 HyperConnection prenorm GEMM.

    Parameters
    ----------
    a : torch.Tensor
        Input tensor with shape ``(M, K)`` and dtype ``torch.bfloat16``.
    b : torch.Tensor
        Weight tensor with shape ``(N, K)`` and dtype ``torch.float32``.
    d : torch.Tensor
        Output GEMM tensor with dtype ``torch.float32``. Shape is ``(M, N)`` when
        ``num_splits`` is ``None`` or ``<= 1``; otherwise ``(num_splits, M, N)``.
    sqr_sum : torch.Tensor
        Output row-wise squared-sum tensor with dtype ``torch.float32``. Shape is
        ``(M,)`` when ``num_splits`` is ``None`` or ``<= 1``; otherwise
        ``(num_splits, M)``.
    num_splits : int, default=None
        Optional split-K factor. When greater than 1, the kernel writes per-split
        partial outputs and callers should reduce them along dim 0.

    Returns
    -------
    None
    """

    m = a.shape[0]
    n = b.shape[0]
    num_splits = 1 if num_splits is None or num_splits <= 1 else int(num_splits)
    num_mps = resolve_num_mps(a.device)

    get_hyperconnection_module(m, n, num_splits, num_mps).get_function(
        "tf32_hc_prenorm_gemm"
    )(
        a,
        b,
        d,
        sqr_sum,
        num_splits,
    )
