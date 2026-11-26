import torch
import mate
from mate import gemm
from wrappers.DeepGEMM import deep_gemm as wrapper
import pytest
import random
from mate.utils import ceil_div
from mate.testing.utils import (
    align,
    group_quantize_fp8,
    group_dequantize_fp8,
    check_gemm_sbo_signal,
    calc_diff,
)
from mate.testing import (
    dequantize_mxfp4,
    dequantize_mxfp8,
    quantize_mxfp4,
    quantize_mxfp8,
    supported_musa_compute_capability,
)


_MXFP4_DTYPE = getattr(torch, "float4_e2m1fn_x2", None)
_MXFP4_QKV_DTYPES = (
    (torch.float8_e4m3fn, _MXFP4_DTYPE)
    if _MXFP4_DTYPE is not None
    else (torch.float8_e4m3fn,)
)
_MXFP4_QKV_IDS = ("mxfp8", "mxfp4") if _MXFP4_DTYPE is not None else ("mxfp8",)


def _quantize_einsum_a(x: torch.Tensor):
    return group_quantize_fp8(
        x,
        (*x.shape[:-1], ceil_div(x.size(-1), 128)),
        (*([1] * (x.dim() - 1)), 128),
        torch.float8_e4m3fn,
        "K",
    )


def _quantize_einsum_b_nk(x: torch.Tensor, recipe):
    _, scale_granularity_n, scale_granularity_k = recipe
    return group_quantize_fp8(
        x,
        (
            x.size(0),
            ceil_div(x.size(1), scale_granularity_n),
            ceil_div(x.size(2), scale_granularity_k),
        ),
        (1, scale_granularity_n, scale_granularity_k),
        torch.float8_e4m3fn,
        "K",
    )


def _quantize_einsum_batch_k(x: torch.Tensor):
    return group_quantize_fp8(
        x,
        (ceil_div(x.size(0), 128), x.size(1), x.size(2)),
        (128, 1, 1),
        torch.float8_e4m3fn,
        "K",
    )


def _apply_skip_head_mid(
    d: torch.Tensor,
    head_splits: tuple[int, int, int],
) -> torch.Tensor:
    left, mid, right = head_splits
    m, n = d.shape
    assert n % (left + right) == 0
    num_heads = n // (left + right)

    d = d.view(m, num_heads, left + right)
    d_left = d[:, :, :left]
    d_right = d[:, :, -right:]
    d_mid = torch.zeros((m, num_heads, mid), dtype=d.dtype, device=d.device)
    return torch.cat([d_left, d_mid, d_right], dim=2).view(m, -1)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("m, compact_n, k", [(128, 8192, 512), (4096, 8192, 512)])
def test_fp8_gemm_nt_skip_head_mid(m, compact_n, k):
    torch.manual_seed(0)
    head_splits = (128, 64, 128)
    left, mid, right = head_splits
    assert compact_n % (left + right) == 0
    num_heads = compact_n // (left + right)
    full_n = num_heads * (left + mid + right)

    a = torch.randn((m, k), device="musa", dtype=torch.bfloat16)
    b = torch.randn((compact_n, k), device="musa", dtype=torch.bfloat16)
    fp8_a, scale_a = group_quantize_fp8(
        a,
        (m, ceil_div(k, 128)),
        (1, 128),
        torch.float8_e4m3fn,
        "K",
    )
    fp8_b, scale_b = group_quantize_fp8(
        b,
        (ceil_div(compact_n, 128), ceil_div(k, 128)),
        (128, 128),
        torch.float8_e4m3fn,
        "K",
    )
    scale_a = scale_a.float()
    scale_b = scale_b.float()

    d = torch.zeros((m, full_n), device="musa", dtype=torch.bfloat16)
    ref_d = (
        group_dequantize_fp8(fp8_a, scale_a, "K")
        @ group_dequantize_fp8(fp8_b, scale_b, "K").T
    )
    ref_d = _apply_skip_head_mid(ref_d, head_splits)

    mate.deep_gemm.fp8_gemm_nt_skip_head_mid(
        (fp8_a, scale_a),
        (fp8_b, scale_b),
        d,
        head_splits,
        disable_ue8m0_cast=True,
    )

    diff = calc_diff(d, ref_d)
    assert diff < 1e-3, f"{m=}, {compact_n=}, {k=}, {head_splits=}, {diff=}"


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("recipe", [(1, 128, 128), (1, 1, 128)])
def test_fp8_einsum_bhr_hdr_bhd(recipe):
    batch, heads, r_dim, d_dim = 4, 2, 128, 256
    x = torch.rand((batch, heads, r_dim), device="musa", dtype=torch.float32)
    y = torch.rand((heads, d_dim, r_dim), device="musa", dtype=torch.float32)
    x_fp8, x_scale = _quantize_einsum_a(x)
    y_fp8, y_scale = _quantize_einsum_b_nk(y, recipe)
    out = torch.empty((batch, heads, d_dim), device="musa", dtype=torch.bfloat16)

    ref = torch.einsum(
        "bhr,hdr->bhd",
        group_dequantize_fp8(x_fp8, x_scale, "K"),
        group_dequantize_fp8(y_fp8, y_scale, "K"),
    )

    mate.deep_gemm.fp8_einsum(
        "bhr,hdr->bhd", (x_fp8, x_scale), (y_fp8, y_scale), out, recipe=recipe
    )

    torch.testing.assert_close(out.float(), ref, rtol=5e-3, atol=5e-3)


@supported_musa_compute_capability([31])
def test_fp8_einsum_bhd_hdr_bhr():
    batch, heads, d_dim, r_dim = 4, 2, 128, 256
    recipe = (1, 128, 128)
    x = torch.rand((batch, heads, d_dim), device="musa", dtype=torch.float32)
    y = torch.rand((heads, d_dim, r_dim), device="musa", dtype=torch.float32)
    x_fp8, x_scale = _quantize_einsum_a(x)
    y_fp8, y_scale = _quantize_einsum_b_nk(y, recipe)
    out = torch.empty((batch, heads, r_dim), device="musa", dtype=torch.bfloat16)

    ref = torch.einsum(
        "bhd,hdr->bhr",
        group_dequantize_fp8(x_fp8, x_scale, "K"),
        group_dequantize_fp8(y_fp8, y_scale, "K"),
    )

    mate.deep_gemm.fp8_einsum(
        "bhd,hdr->bhr", (x_fp8, x_scale), (y_fp8, y_scale), out, recipe=recipe
    )

    torch.testing.assert_close(out.float(), ref, rtol=5e-3, atol=5e-3)


@supported_musa_compute_capability([31])
def test_fp8_einsum_bhd_bhr_hdr_with_c():
    batch, heads, d_dim, r_dim = 128, 2, 128, 128
    recipe = (1, 1, 128)
    x = torch.rand((batch, heads, d_dim), device="musa", dtype=torch.float32)
    y = torch.rand((batch, heads, r_dim), device="musa", dtype=torch.float32)
    x_fp8, x_scale = _quantize_einsum_batch_k(x)
    y_fp8, y_scale = _quantize_einsum_batch_k(y)
    out = torch.randn((heads, d_dim, r_dim), device="musa", dtype=torch.float32)
    c = out.clone()

    ref = c + torch.einsum(
        "bhd,bhr->hdr",
        group_dequantize_fp8(x_fp8, x_scale, "K"),
        group_dequantize_fp8(y_fp8, y_scale, "K"),
    )

    mate.deep_gemm.fp8_einsum(
        "bhd,bhr->hdr", (x_fp8, x_scale), (y_fp8, y_scale), out, c, recipe=recipe
    )

    torch.testing.assert_close(out, ref, rtol=5e-3, atol=5e-3)


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(dtype=torch.uint8)
    x_fp8[:, block_size * head_dim :] = sf.view(num_blocks, block_size).view(
        dtype=torch.uint8
    )
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4), x_scaled, sf


def ref_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,  # 1D [batch] or 2D [batch, next_n]
    block_tables: torch.Tensor,
    max_model_len: int,
):
    """
    context_lens layout:
      1D [batch]         — shared context length for all next_n positions in a batch row
      2D [batch, next_n] — per-(batch, next_n) context length;
                           context_lens[i, j] is the number of valid KV tokens when
                           computing attention for the j-th draft token of batch i.
                           Must satisfy context_lens[i, j] <= context_lens[i, j+1].
    """
    batch_size, next_n, heads, dim = q.size()
    num_block, block_size, _, _ = kv_cache.size()
    is_2d = context_lens.dim() == 2

    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )

    ctx_lens_cpu = context_lens.tolist()

    for i in range(batch_size):
        if is_2d:
            per_slot_lens = ctx_lens_cpu[i]  # list of length next_n
        else:
            per_slot_lens = [ctx_lens_cpu[i]] * next_n
        max_ctx_len = per_slot_lens[-1]

        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )  # [heads, next_n]

        for block_rk in range(ceil_div(max_ctx_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx = q[i]  # [next_n, heads, dim]
            kx = kv_cache[block_idx]  # [block_size, 1, dim]

            k_offsets = torch.arange(
                block_rk * block_size,
                (block_rk + 1) * block_size,
                device=q.device,
            )  # [block_size]

            q_offsets = torch.tensor(
                [per_slot_lens[j] - 1 for j in range(next_n)],
                device=q.device,
                dtype=torch.long,
            )  # [next_n]

            mask = k_offsets[None, :] <= q_offsets[:, None]  # [next_n, block_size]

            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]  # [heads, next_n, block_size]
            s = s.sum(dim=0)  # [next_n, block_size]

            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(mask, s, float("-inf"))

    return logits


def _pack_mqa_blockscale(
    scale: torch.Tensor, is_scale_m_major: bool = False
) -> torch.Tensor:
    """Pad UE8M0 scales to four storage slots and set the row/group byte order.

    K-major keeps the padded rows contiguous; M-major returns a same-shape
    strided view with unit row stride and group stride equal to the row count.
    """
    packed = torch.zeros((*scale.shape[:-1], 4), dtype=torch.uint8, device=scale.device)
    packed[..., : scale.shape[-1]] = scale

    if not is_scale_m_major:
        return packed

    num_rows = packed.numel() // 4
    storage = packed.reshape(num_rows, 4).transpose(0, 1).contiguous()
    row_strides = []
    stride = 1
    for size in reversed(packed.shape[:-1]):
        row_strides.append(stride)
        stride *= size
    return storage.as_strided(packed.shape, tuple(reversed(row_strides)) + (num_rows,))


def _fuse_paged_mqa_kv(
    kv_cache: torch.Tensor,
    kv_cache_scale: torch.Tensor,
    is_scale_m_major: bool = False,
) -> torch.Tensor:
    num_pages, page_size, num_kv_heads, headdim = kv_cache.shape
    fused = torch.empty(
        (num_pages, page_size * (headdim + 4)),
        dtype=torch.uint8,
        device=kv_cache.device,
    )
    fused[:, : page_size * headdim] = kv_cache.view(torch.uint8).reshape(
        num_pages, page_size * headdim
    )
    packed_scales = torch.zeros(
        (*kv_cache_scale.shape[:-1], 4), dtype=torch.uint8, device=kv_cache_scale.device
    )
    packed_scales[..., : kv_cache_scale.shape[-1]] = kv_cache_scale
    packed_scales = packed_scales.reshape(num_pages, page_size, 4)

    if is_scale_m_major:
        packed_scales = packed_scales.transpose(1, 2).contiguous()

    fused[:, page_size * headdim :] = packed_scales.reshape(num_pages, page_size * 4)
    return fused.view(num_pages, page_size, num_kv_heads, headdim + 4)


def ref_paged_mqa_logits_blockscale(
    q_mx: tuple[torch.Tensor, torch.Tensor],
    kv_mx: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    seqlen_kv: torch.Tensor,
    page_table: torch.Tensor,
) -> torch.Tensor:
    dequantize = (
        dequantize_mxfp4
        if _MXFP4_DTYPE is not None and q_mx[0].dtype == _MXFP4_DTYPE
        else dequantize_mxfp8
    )
    weights = weights.float()
    q = dequantize(*q_mx)
    kv_cache_paged = dequantize(*kv_mx)
    batch_size, next_n, num_heads, head_dim = q.shape
    max_seqlen_kv = page_table.shape[1] * kv_cache_paged.shape[1]
    kv_cache = kv_cache_paged[page_table.flatten()].view(
        batch_size, max_seqlen_kv, 1, head_dim
    )

    scores = torch.einsum("bnhd,bkd->bnhk", q, kv_cache[:, :, 0])
    logits = (torch.relu(scores) * weights.view(batch_size, next_n, num_heads, 1)).sum(
        dim=2
    )
    logits = logits.view(batch_size * next_n, max_seqlen_kv)

    if seqlen_kv.dim() == 2:
        row_seqlens_kv = seqlen_kv.reshape(-1)
    else:
        next_n_offsets = torch.arange(next_n, device=q.device, dtype=torch.int32)
        row_seqlens_kv = (
            seqlen_kv[:, None] - next_n + next_n_offsets[None, :] + 1
        ).reshape(-1)
    positions = torch.arange(max_seqlen_kv, device=q.device)
    return logits.masked_fill(
        positions[None, :] >= row_seqlens_kv[:, None], float("-inf")
    )


@supported_musa_compute_capability([32])
@pytest.mark.parametrize("page_size", [32, 64, 128])
@pytest.mark.parametrize("next_n", [1, 2, 6])
@pytest.mark.parametrize("num_heads", [8, 16, 32, 64])
@pytest.mark.parametrize("headdim", [32, 64, 128])
@pytest.mark.parametrize("seqlens_kv", [[1024, 1024], [1234, 256, 1234]])
@pytest.mark.parametrize("is_q_scale_m_major", [False, True])
@pytest.mark.parametrize("is_kv_scale_m_major", [False, True])
@pytest.mark.parametrize(
    "qkv_dtype",
    _MXFP4_QKV_DTYPES,
    ids=_MXFP4_QKV_IDS,
)
@pytest.mark.parametrize("weights_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
def test_paged_mqa_logits_blockscale(
    page_size,
    next_n,
    num_heads,
    headdim,
    seqlens_kv,
    is_q_scale_m_major,
    is_kv_scale_m_major,
    qkv_dtype,
    weights_dtype,
    logits_dtype,
):
    if weights_dtype == torch.bfloat16 and logits_dtype == torch.float32:
        pytest.skip("bfloat16 weights require bfloat16 logits")
    torch.manual_seed(666)

    device = torch.device("musa")
    batch_size = len(seqlens_kv)
    q = torch.randn(
        (batch_size, next_n, num_heads, headdim),
        dtype=torch.bfloat16,
    )
    max_seqlen_kv = ceil_div(max(seqlens_kv), page_size) * page_size
    seqlen_kv = torch.tensor(seqlens_kv, dtype=torch.int32).unsqueeze(1)
    seqlen_kv = (
        seqlen_kv - next_n + torch.arange(1, next_n + 1, dtype=torch.int32).unsqueeze(0)
    )
    num_pages_per_query = max_seqlen_kv // page_size
    num_total_pages = batch_size * num_pages_per_query
    kv_cache_paged = torch.randn(
        (num_total_pages, page_size, 1, headdim),
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        (batch_size * next_n, num_heads),
        dtype=torch.float32,
    ).to(weights_dtype)
    page_table = torch.randperm(num_total_pages, dtype=torch.int32).view(
        batch_size,
        num_pages_per_query,
    )

    quantize = (
        quantize_mxfp4
        if _MXFP4_DTYPE is not None and qkv_dtype == _MXFP4_DTYPE
        else quantize_mxfp8
    )
    q_mx = quantize(q)
    kv_mx = quantize(kv_cache_paged)
    ref_logits = ref_paged_mqa_logits_blockscale(
        q_mx,
        kv_mx,
        weights,
        seqlen_kv,
        page_table,
    )
    q_values, q_scale = q_mx
    kv_cache, kv_cache_scale = kv_mx
    fused_kv_cache = _fuse_paged_mqa_kv(kv_cache, kv_cache_scale, is_kv_scale_m_major)

    q_values_musa = q_values.to(device)
    q_scale_musa = _pack_mqa_blockscale(q_scale.to(device), is_q_scale_m_major)
    fused_kv_cache_musa = fused_kv_cache.to(device)
    weights_musa = weights.to(device)
    seqlen_kv_musa = seqlen_kv.to(device)
    page_table_musa = page_table.to(device)
    metadata = mate.deep_gemm.get_paged_mqa_logits_metadata(
        seqlen_kv_musa,
        512,
    )

    logits = mate.deep_gemm.paged_mqa_logits(
        q_values_musa,
        fused_kv_cache_musa,
        weights_musa,
        seqlen_kv_musa,
        page_table_musa,
        metadata,
        max_seqlen_kv,
        logits_dtype=logits_dtype,
        is_kv_scale_m_major=is_kv_scale_m_major,
        scale_q=q_scale_musa,
    )
    torch.musa.synchronize()
    logits = logits.cpu()

    positions = torch.arange(max_seqlen_kv).unsqueeze(0)
    valid = positions < seqlen_kv.reshape(-1, 1)
    logits_masked = logits.masked_fill(~valid, 0)
    ref_logits_masked = ref_logits.masked_fill(~valid, 0)

    diff = calc_diff(logits_masked, ref_logits_masked)
    tol = 3e-5 if torch.bfloat16 in (weights_dtype, logits_dtype) else 5e-6
    assert diff < tol, f"{diff=}"


@supported_musa_compute_capability([32])
@pytest.mark.parametrize("page_size", [128, 32, 64])
@pytest.mark.parametrize("num_heads", [8, 16, 32, 64])
@pytest.mark.parametrize("headdim", [32, 64, 128])
@pytest.mark.parametrize(
    "seqlens", [[(512, 512), (512, 512), (512, 512)], [(1234, 1234), (1234, 1234)]]
)
@pytest.mark.parametrize("is_q_scale_m_major", [False, True])
@pytest.mark.parametrize("is_kv_scale_m_major", [False, True])
@pytest.mark.parametrize(
    "qkv_dtype",
    _MXFP4_QKV_DTYPES,
)
@pytest.mark.parametrize("weights_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
def test_paged_mqa_logits_blockscale_varlen(
    page_size,
    num_heads,
    headdim,
    seqlens,
    is_q_scale_m_major,
    is_kv_scale_m_major,
    qkv_dtype,
    weights_dtype,
    logits_dtype,
):
    if weights_dtype == torch.bfloat16 and logits_dtype == torch.float32:
        pytest.skip("bfloat16 weights require bfloat16 logits")
    torch.manual_seed(666)
    seqlens_q = torch.tensor([seqlen_q for seqlen_q, _ in seqlens], dtype=torch.int32)
    seqlens_kv = torch.tensor(
        [seqlen_kv for _, seqlen_kv in seqlens], dtype=torch.int32
    )
    raw_batch_size = len(seqlens)
    indices = torch.arange(raw_batch_size, dtype=torch.int32).repeat_interleave(
        seqlens_q
    )
    total_q = indices.numel()

    q = torch.randn(
        (total_q, 1, num_heads, headdim),
        dtype=torch.bfloat16,
    )
    num_blocks_per_request = ceil_div(seqlens_kv, page_size)
    max_context_len = int(num_blocks_per_request.max().item()) * page_size
    num_total_blocks = int(num_blocks_per_request.sum().item())
    kv_cache = torch.randn(
        (num_total_blocks, page_size, 1, headdim),
        dtype=torch.bfloat16,
    )
    weights = torch.randn((total_q, num_heads), dtype=torch.float32).to(weights_dtype)

    request_page_table = torch.zeros(
        (raw_batch_size, int(num_blocks_per_request.max().item())),
        dtype=torch.int32,
    )
    block_idx_pool = torch.randperm(num_total_blocks, dtype=torch.int32)
    offset = 0
    for request_idx, num_blocks in enumerate(num_blocks_per_request.tolist()):
        request_page_table[request_idx, :num_blocks] = block_idx_pool[
            offset : offset + num_blocks
        ]
        offset += num_blocks
    page_table = request_page_table.repeat_interleave(seqlens_q, dim=0)

    offsets_within_request = torch.cat(
        [torch.arange(seqlen_q, dtype=torch.int32) for seqlen_q, _ in seqlens]
    )
    context_lens = (
        (seqlens_kv - seqlens_q + 1).repeat_interleave(seqlens_q)
        + offsets_within_request
    ).view(total_q, 1)

    quantize = (
        quantize_mxfp4
        if _MXFP4_DTYPE is not None and qkv_dtype == _MXFP4_DTYPE
        else quantize_mxfp8
    )
    q_mx = quantize(q)
    kv_mx = quantize(kv_cache)
    ref_logits = ref_paged_mqa_logits_blockscale(
        q_mx,
        kv_mx,
        weights,
        context_lens,
        page_table,
    )
    q_values, q_scale = q_mx
    kv_cache, kv_cache_scale = kv_mx
    fused_kv_cache = _fuse_paged_mqa_kv(kv_cache, kv_cache_scale, is_kv_scale_m_major)

    device = torch.device("musa")
    q_values_musa = q_values.to(device)
    q_scale_musa = _pack_mqa_blockscale(q_scale.to(device), is_q_scale_m_major)
    fused_kv_cache_musa = fused_kv_cache.to(device)
    weights_musa = weights.to(device)
    context_lens_musa = context_lens.to(device)
    page_table_musa = page_table.to(device)
    indices_musa = indices.to(device)
    metadata = mate.deep_gemm.get_paged_mqa_logits_metadata(
        context_lens_musa,
        512,
        num_mps=8,
        indices=indices_musa,
    )

    logits = mate.deep_gemm.paged_mqa_logits(
        q_values_musa,
        fused_kv_cache_musa,
        weights_musa,
        context_lens_musa,
        page_table_musa,
        metadata,
        max_context_len,
        logits_dtype=logits_dtype,
        indices=indices_musa,
        is_kv_scale_m_major=is_kv_scale_m_major,
        scale_q=q_scale_musa,
    )
    torch.musa.synchronize()
    logits = logits.cpu()

    valid = torch.arange(max_context_len).unsqueeze(0) < context_lens
    logits_masked = logits.masked_fill(~valid, 0)
    ref_logits_masked = ref_logits.masked_fill(~valid, 0)

    diff = calc_diff(logits_masked, ref_logits_masked)
    assert diff < 0.02, f"{diff=}"


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("batch_size, next_n", [(64, 1), (64, 4), (64, 2), (128, 1)])
@pytest.mark.parametrize("avg_kv", [8192, 32768])
@pytest.mark.parametrize("is_context_lens_2d", [False, True])
@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
def test_paged_mqa_logits(batch_size, next_n, avg_kv, is_context_lens_2d, logits_dtype):
    # 2D context_lens only makes sense when next_n > 1
    if is_context_lens_2d and next_n == 1:
        pytest.skip(
            "2D context_lens with next_n=1 is identical to 1D; skip to avoid redundancy"
        )

    torch.manual_seed(0)
    random.seed(0)

    max_model_len = 111 * 1000
    for heads, index_dim in [(32, 128)]:
        num_blocks, blocksize = max_model_len * 3, 64

        q = torch.randn(
            (batch_size, next_n, heads, index_dim),
            device="musa",
            dtype=torch.bfloat16,
        )
        kv_cache = torch.randn(
            (num_blocks, blocksize, 1, index_dim),
            device="musa",
            dtype=torch.bfloat16,
        )
        weights = torch.randn(
            (batch_size * next_n, heads), device="musa", dtype=torch.float32
        )

        base_lens = torch.randint(
            int(0.7 * avg_kv), int(1.3 * avg_kv), (batch_size,)
        ).to(torch.int32)
        base_lens = torch.ones(batch_size, dtype=torch.int32) * avg_kv

        if is_context_lens_2d:
            offsets = torch.arange(next_n, dtype=torch.int32)  # [0, 1, ..., next_n-1]
            context_lens_cpu = (
                base_lens.unsqueeze(1) - (next_n - 1) + offsets.unsqueeze(0)
            )  # [batch, next_n]
            context_lens = context_lens_cpu.musa()
            max_context_len_per_row = context_lens[:, -1]
        else:
            context_lens = base_lens.musa()
            max_context_len_per_row = context_lens

        max_block_len = (
            (max_context_len_per_row.max().item() + blocksize - 1)
            // blocksize
            * blocksize
        )
        block_tables = torch.zeros(
            (batch_size, max_block_len), device="musa", dtype=torch.int32
        )

        counter = 0
        block_idx_pool = list(range(num_blocks))
        random.shuffle(block_idx_pool)
        for i in range(batch_size):
            ctx_len = max_context_len_per_row[i].item()
            for j in range(ceil_div(ctx_len, blocksize)):
                block_tables[i][j] = block_idx_pool[counter]
                counter += 1

        ref_logits = ref_fp8_paged_mqa_logits(
            q, kv_cache, weights, context_lens, block_tables, max_model_len
        )

        q_fp8 = q.to(torch.float8_e4m3fn)
        kv_cache_fp8, x_scaled, sf = kv_cache_cast_to_fp8(kv_cache)

        schedule_metadata = mate.deep_gemm.get_paged_mqa_logits_metadata(
            context_lens, blocksize
        )
        logits = mate.deep_gemm.fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len,
            True,
            logits_dtype=logits_dtype,
        )

        positions = (
            torch.arange(max_model_len, device="musa")
            .unsqueeze(0)
            .expand(batch_size * next_n, -1)
        )  # [batch*next_n, max_model_len]

        if is_context_lens_2d:
            per_slot_lens = context_lens.view(-1)  # [batch*next_n]
            ref_neginf_mask = positions >= per_slot_lens.unsqueeze(1)
        else:
            row_indices = torch.arange(batch_size * next_n, device="musa") // next_n
            next_n_offset = torch.arange(batch_size * next_n, device="musa") % next_n
            q_positions = (
                context_lens[row_indices] - next_n + next_n_offset
            ).unsqueeze(1)
            ref_neginf_mask = positions > q_positions

        neginf_mask = logits == float("-inf")
        assert torch.equal(neginf_mask, ref_neginf_mask), (
            "neginf mask mismatch between kernel and reference"
        )

        logits = logits.masked_fill(ref_neginf_mask, 0)
        ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
        diff = calc_diff(logits.float(), ref_logits)
        assert diff < 1e-3, f"{diff=}"


def get_deepgemm_group_gemm_contig_cases():
    return [
        [4096 for _ in range(8)],
        [8192 for _ in range(4)],
    ]


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("ms_per_group", get_deepgemm_group_gemm_contig_cases())
@pytest.mark.parametrize("n", [4096])
@pytest.mark.parametrize("k", [2048])
@pytest.mark.parametrize("a_fp8_type", [torch.float8_e4m3fn])
@pytest.mark.parametrize("b_fp8_type", [torch.float8_e4m3fn])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
@pytest.mark.parametrize("alignment_m", [128, 256])
@pytest.mark.parametrize("backend", ["mubin", "mutlass"])
@pytest.mark.parametrize("explicit_recipe", [False, True], ids=["default", "explicit"])
def test_m_grouped_fp8_gemm_nt_contiguous(
    ms_per_group,
    n,
    k,
    a_fp8_type,
    b_fp8_type,
    out_dtype,
    alignment_m,
    backend,
    explicit_recipe,
):
    quant_tile = 128
    scale_granularity_mnk = (1, quant_tile, quant_tile)

    num_expert = len(ms_per_group)
    aligned_ms = [align(m, alignment_m) for m in ms_per_group]
    m = sum(aligned_ms)

    a = torch.rand((m, k), device="musa", dtype=torch.float)
    b = torch.rand((num_expert, n, k), device="musa", dtype=torch.float)
    m_indices = torch.full((m,), -1, device="musa", dtype=torch.int32)

    d = torch.empty((m, n), device="musa", dtype=out_dtype)
    ref_d = torch.zeros((m, n), device="musa", dtype=torch.float)

    quant_tile_shape_a = (1, quant_tile)
    quant_tile_shape_b = (1, quant_tile, quant_tile)
    scale_a_shape = (m, k // quant_tile)
    scale_b_shape = (num_expert, n // quant_tile, k // quant_tile)
    fp8_a, scale_a = group_quantize_fp8(
        a, scale_a_shape, quant_tile_shape_a, a_fp8_type, "K"
    )
    fp8_b, scale_b = group_quantize_fp8(
        b, scale_b_shape, quant_tile_shape_b, b_fp8_type, "K"
    )

    dequant_a = group_dequantize_fp8(fp8_a, scale_a, "K")
    dequant_b = group_dequantize_fp8(fp8_b, scale_b, "K")

    # calc ref
    m_base = 0
    for i in range(num_expert):
        r = slice(m_base, m_base + ms_per_group[i])
        m_indices[r] = i
        ref_d[r] = torch.matmul(dequant_a[r], dequant_b[i].t())
        m_base += aligned_ms[i]

    mate.deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (fp8_a, scale_a),
        (fp8_b, scale_b),
        d,
        m_indices,
        **({"recipe": scale_granularity_mnk} if explicit_recipe else {}),
        alignment_m=alignment_m,
        backend=backend,
    )

    d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
    torch.testing.assert_close(d.to(torch.float), ref_d, rtol=5e-3, atol=5e-3)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("alignment_m", [128, 256])
@pytest.mark.parametrize("backend", ["auto", "mutlass"])
def test_m_grouped_fp8_gemm_nn_contiguous(alignment_m, backend):
    ms_per_group = [333, 444]
    n, k = 4096, 2048
    quant_tile = 128
    recipe = (1, quant_tile, quant_tile)

    num_expert = len(ms_per_group)
    aligned_ms = [align(m, alignment_m) for m in ms_per_group]
    m = sum(aligned_ms)

    a = torch.rand((m, k), device="musa", dtype=torch.float)
    b = torch.rand((num_expert, n, k), device="musa", dtype=torch.float)
    m_indices = torch.full((m,), -1, device="musa", dtype=torch.int32)
    d = torch.empty((m, n), device="musa", dtype=torch.bfloat16)
    ref_d = torch.zeros((m, n), device="musa", dtype=torch.float)

    fp8_a, scale_a = group_quantize_fp8(
        a,
        (m, k // quant_tile),
        (1, quant_tile),
        torch.float8_e4m3fn,
        "K",
    )
    fp8_b, scale_b = group_quantize_fp8(
        b,
        (num_expert, n // quant_tile, k // quant_tile),
        (1, quant_tile, quant_tile),
        torch.float8_e4m3fn,
        "K",
    )

    dequant_a = group_dequantize_fp8(fp8_a, scale_a, "K")
    dequant_b = group_dequantize_fp8(fp8_b, scale_b, "K")
    m_base = 0
    for group, group_m in enumerate(ms_per_group):
        rows = slice(m_base, m_base + group_m)
        m_indices[rows] = group
        ref_d[rows] = torch.matmul(dequant_a[rows], dequant_b[group].t())
        m_base += aligned_ms[group]

    mate.deep_gemm.m_grouped_fp8_gemm_nn_contiguous(
        (fp8_a, scale_a),
        (fp8_b.transpose(1, 2).contiguous(), scale_b.transpose(1, 2)),
        d,
        m_indices,
        recipe,
        alignment_m=alignment_m,
        backend=backend,
    )

    d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
    torch.testing.assert_close(d.float(), ref_d, rtol=5e-3, atol=5e-3)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("ms_per_group", get_deepgemm_group_gemm_contig_cases())
@pytest.mark.parametrize("n", [4096])
@pytest.mark.parametrize("k", [2048])
@pytest.mark.parametrize("data_type", [torch.bfloat16])
@pytest.mark.parametrize("alignment_m", [128, 256])
@pytest.mark.parametrize("backend", ["mubin", "mutlass"])
def test_m_grouped_bf16_gemm_nt_contiguous(
    ms_per_group, n, k, data_type, alignment_m, backend
):
    num_expert = len(ms_per_group)
    aligned_ms = [align(m, alignment_m) for m in ms_per_group]
    m = sum(aligned_ms)

    a = torch.rand((m, k), device="musa", dtype=data_type)
    b = torch.rand((num_expert, n, k), device="musa", dtype=data_type)
    m_indices = torch.full((m,), -1, device="musa", dtype=torch.int32)

    d = torch.empty((m, n), device="musa", dtype=data_type)
    ref_d = torch.zeros((m, n), device="musa", dtype=data_type)

    # calc ref
    m_base = 0
    for i in range(num_expert):
        r = slice(m_base, m_base + ms_per_group[i])
        m_indices[r] = i
        ref_d[r] = torch.matmul(a[r], b[i].t())
        m_base += aligned_ms[i]

    mate.deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
        a,
        b,
        d,
        m_indices,
        alignment_m=alignment_m,
        backend=backend,
    )

    d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
    torch.testing.assert_close(d, ref_d, rtol=5e-3, atol=5e-3)


def get_m_grouped_gemm_nt_masked_cases():
    return [
        [256, 256, 256, 256],
        [333, 444, 555, 666],
    ]


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("ms_per_group", get_m_grouped_gemm_nt_masked_cases())
@pytest.mark.parametrize("n", [4096])
@pytest.mark.parametrize("k", [4096])
@pytest.mark.parametrize("expected_m", [None, 8192])
@pytest.mark.parametrize("a_fp8_type", [torch.float8_e4m3fn])
@pytest.mark.parametrize("b_fp8_type", [torch.float8_e4m3fn])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
@pytest.mark.parametrize("enable_overlap", [False, True])
@pytest.mark.parametrize("backend", ["mubin", "mutlass"])
@pytest.mark.parametrize("explicit_recipe", [False, True], ids=["default", "explicit"])
def test_m_grouped_fp8_gemm_nt_masked(
    ms_per_group,
    n,
    k,
    expected_m,
    a_fp8_type,
    b_fp8_type,
    out_dtype,
    enable_overlap,
    backend,
    explicit_recipe,
):
    if backend == "mutlass" and enable_overlap:
        pytest.skip('backend="mutlass" does not support enable_overlap')

    tile_signal = 64
    quant_tile = 128
    scale_granularity_mnk = (1, quant_tile, quant_tile)

    max_m = max(ms_per_group)
    expected_m = expected_m if expected_m is not None else max_m

    num_expert = len(ms_per_group)

    a = torch.rand((num_expert, max_m, k), device="musa", dtype=torch.float)
    b = torch.rand((num_expert, n, k), device="musa", dtype=torch.float)
    masked_m = torch.tensor(ms_per_group, device="musa", dtype=torch.int32)

    d = torch.empty((num_expert, max_m, n), device="musa", dtype=out_dtype)
    signal = torch.zeros(
        num_expert * ceil_div(max_m, tile_signal), dtype=torch.int32, device=a.device
    )

    quant_tile_shape_a = (1, 1, quant_tile)
    quant_tile_shape_b = (1, quant_tile, quant_tile)
    scale_a_shape = (num_expert, max_m, k // quant_tile)
    scale_b_shape = (num_expert, n // quant_tile, k // quant_tile)
    fp8_a, scale_a = group_quantize_fp8(
        a, scale_a_shape, quant_tile_shape_a, a_fp8_type, "K"
    )
    fp8_b, scale_b = group_quantize_fp8(
        b, scale_b_shape, quant_tile_shape_b, b_fp8_type, "K"
    )

    dequant_a = group_dequantize_fp8(fp8_a, scale_a, "K")
    dequant_b = group_dequantize_fp8(fp8_b, scale_b, "K")
    ref_d = torch.einsum("bmk,bnk->bmn", dequant_a, dequant_b).to(torch.float)

    res = mate.deep_gemm.fp8_m_grouped_gemm_nt_masked(
        (fp8_a, scale_a),
        (fp8_b, scale_b),
        d,
        masked_m,
        expected_m,
        **({"recipe": scale_granularity_mnk} if explicit_recipe else {}),
        enable_overlap=enable_overlap,
        signal=signal,
        backend=backend,
    )

    d = d.to(torch.float)
    for i in range(num_expert):
        torch.testing.assert_close(
            d[i, : ms_per_group[i], :],
            ref_d[i, : ms_per_group[i], :],
            rtol=5e-3,
            atol=5e-3,
        )

    if enable_overlap:
        block_m = res[0]
        threshold = res[1]
        check_gemm_sbo_signal(num_expert, max_m, block_m, threshold, signal, masked_m)


def kv_cache_cast_to_fp8_for_nonpaged(kv: torch.Tensor, block_size: int = 64):
    assert kv.dim() == 2
    seq_kv, head_dim = kv.shape
    num_blocks = (seq_kv + block_size - 1) // block_size
    padded = num_blocks * block_size

    if padded != seq_kv:
        kv_pad = torch.zeros((padded, head_dim), device=kv.device, dtype=kv.dtype)
        kv_pad[:seq_kv].copy_(kv)
    else:
        kv_pad = kv

    x = kv_pad.view(num_blocks, block_size, 1, head_dim)
    _, x_scaled, sf = kv_cache_cast_to_fp8(x)

    kv_fp8 = x_scaled.view(padded, head_dim)[:seq_kv]
    kv_scale = sf.view(padded)[:seq_kv]
    return kv_fp8, kv_scale


def generate_cp_test_data(seq_len, seq_len_kv, device):
    chunk_size0 = seq_len // 2
    chunk_size1 = seq_len - chunk_size0

    ks = torch.zeros(seq_len, dtype=torch.int32, device=device)
    ke = torch.empty(seq_len, dtype=torch.int32, device=device)

    base0 = seq_len_kv // 3
    base1 = (seq_len_kv * 2) // 3

    t0 = torch.arange(chunk_size0, dtype=torch.int32, device=device)
    t1 = torch.arange(chunk_size1, dtype=torch.int32, device=device)
    ke0 = base0 + t0
    ke1 = base1 + t1

    ke0 = ke0.clamp(min=0, max=seq_len_kv)
    ke1 = ke1.clamp(min=0, max=seq_len_kv)

    ke[:chunk_size0] = ke0
    ke[chunk_size0:] = ke1
    return ks, ke


def ref_fp8_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    cost_only: bool = False,
    kv_chunk_size: int = 4096,
):
    device = q.device
    seq_len_q, num_heads, head_dim = q.shape
    seq_len_kv = kv.shape[0]

    ks = cu_seqlen_ks.clamp(min=0, max=seq_len_kv)
    ke = cu_seqlen_ke.clamp(min=0, max=seq_len_kv)

    if cost_only:
        count_ones_per_row = (ke - ks).clamp(min=0)
        return count_ones_per_row.sum()

    q_f = q.to(torch.float32)
    k_f = kv.to(torch.float32)
    w_f = weights.to(torch.float32)

    cost = (ke - ks).clamp(min=0).sum()
    logits = torch.zeros(
        (seq_len_q, seq_len_kv),
        device=device,
        dtype=torch.float32,
    )

    for kv_start in range(0, seq_len_kv, kv_chunk_size):
        kv_end = min(seq_len_kv, kv_start + kv_chunk_size)
        cur_kv_len = kv_end - kv_start
        k_block = k_f[kv_start:kv_end, :]
        logits_block = torch.zeros(
            (seq_len_q, cur_kv_len),
            device=device,
            dtype=torch.float32,
        )

        for h in range(num_heads):
            q_h = q_f[:, h, :]
            w_h = w_f[:, h].unsqueeze(-1)
            score_block = q_h @ k_block.t()
            logits_block += torch.relu(score_block) * w_h

        positions_block = torch.arange(
            kv_start,
            kv_end,
            device=device,
        ).unsqueeze(0)

        mask_lo = positions_block >= ks.unsqueeze(1)
        mask_hi = positions_block < ke.unsqueeze(1)
        mask_block = mask_lo & mask_hi

        logits_block = logits_block.masked_fill(
            ~mask_block,
            float("-inf"),
        )

        logits[:, kv_start:kv_end] = logits_block

    return logits, cost


def ref_mqa_logits_blockscale(
    q: torch.Tensor,
    kv: torch.Tensor,
    scale_q: torch.Tensor,
    scale_kv: torch.Tensor,
    weights: torch.Tensor,
):
    dequantize = (
        dequantize_mxfp4
        if _MXFP4_DTYPE is not None and q.dtype == _MXFP4_DTYPE
        else dequantize_mxfp8
    )
    weights = weights.float()
    q_f = dequantize(q, scale_q).float()
    kv_f = dequantize(kv, scale_kv).float()
    scores = torch.einsum("qhd,kd->qhk", q_f, kv_f)
    return (torch.relu(scores) * weights[:, :, None]).sum(dim=1)


@supported_musa_compute_capability([32])
@pytest.mark.parametrize("seqlen", [(1024, 1234), (1024, 1236), (1234, 2468)])
@pytest.mark.parametrize("is_q_scale_m_major", [False, True])
@pytest.mark.parametrize("is_kv_scale_m_major", [False, True])
@pytest.mark.parametrize("num_heads", [8, 16, 32, 64])
@pytest.mark.parametrize("headdim", [32, 64, 128])
@pytest.mark.parametrize(
    "compressed_logits,clean_logits",
    [(False, False), (False, True), (True, False)],
    ids=["full", "full-clean", "compressed"],
)
@pytest.mark.parametrize(
    "qkv_dtype",
    _MXFP4_QKV_DTYPES,
)
@pytest.mark.parametrize("weights_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_scheduler", [False, True])
def test_mqa_logits_blockscale(
    seqlen,
    is_q_scale_m_major,
    is_kv_scale_m_major,
    num_heads,
    headdim,
    compressed_logits,
    clean_logits,
    qkv_dtype,
    weights_dtype,
    logits_dtype,
    use_scheduler,
):
    if weights_dtype == torch.bfloat16 and logits_dtype == torch.float32:
        pytest.skip("bfloat16 weights require bfloat16 logits")
    torch.manual_seed(666)

    seqlen_q, seqlen_kv = seqlen
    if is_kv_scale_m_major and seqlen_kv % 4 != 0:
        pytest.skip("M-major KV scales require seqlen_kv divisible by 4")
    device = torch.device("musa")
    q = torch.randn((seqlen_q, num_heads, headdim), dtype=torch.float32)
    kv = torch.randn((seqlen_kv, headdim), dtype=torch.float32)
    weights = torch.randn((seqlen_q, num_heads), dtype=torch.float32).to(weights_dtype)

    ks = torch.zeros(seqlen_q, dtype=torch.int32)
    ke = torch.arange(seqlen_q, dtype=torch.int32) + (seqlen_kv - seqlen_q + 1)

    quantize = (
        quantize_mxfp4
        if _MXFP4_DTYPE is not None and qkv_dtype == _MXFP4_DTYPE
        else quantize_mxfp8
    )
    q_values, q_scale = quantize(q)
    kv_values, kv_scale = quantize(kv)

    ref_logits = ref_mqa_logits_blockscale(
        q_values,
        kv_values,
        q_scale,
        kv_scale,
        weights,
    )
    positions = torch.arange(seqlen_kv).unsqueeze(0)
    valid = (positions >= ks.unsqueeze(1)) & (positions < ke.unsqueeze(1))

    q_values_musa = q_values.to(device)
    kv_values_musa = kv_values.to(device)
    scale_q_musa = _pack_mqa_blockscale(q_scale.to(device), is_q_scale_m_major)
    scale_kv_musa = _pack_mqa_blockscale(kv_scale.to(device), is_kv_scale_m_major)
    weights_musa = weights.to(device)
    ks_musa = ks.to(device)
    ke_musa = ke.to(device)
    schedule_meta = None
    if use_scheduler:
        schedule_meta = mate.deep_gemm.get_mqa_logits_metadata(
            ks_musa,
            ke_musa,
            seqlen_kv,
            num_heads,
        )

    if compressed_logits:
        max_seqlen_k = int((ke - ks).max().item())
        logits_comp = mate.deep_gemm.mqa_logits(
            q_values_musa,
            kv_values_musa,
            scale_kv_musa,
            weights_musa,
            ks_musa,
            ke_musa,
            clean_logits=clean_logits,
            logits_dtype=logits_dtype,
            max_seqlen_k=max_seqlen_k,
            scale_q=scale_q_musa,
            schedule_meta=schedule_meta,
        )
        torch.musa.synchronize()
        logits_comp = logits_comp.cpu()
        compressed_positions = torch.arange(max_seqlen_k).unsqueeze(0)
        compressed_valid = compressed_positions < (ke - ks).unsqueeze(1)
        logits_masked = logits_comp.masked_fill(~compressed_valid, 0)
        ref_logits_masked = ref_logits[:, :max_seqlen_k].masked_fill(
            ~compressed_valid, 0
        )
    else:
        logits = mate.deep_gemm.mqa_logits(
            q_values_musa,
            kv_values_musa,
            scale_kv_musa,
            weights_musa,
            ks_musa,
            ke_musa,
            clean_logits=clean_logits,
            logits_dtype=logits_dtype,
            scale_q=scale_q_musa,
            schedule_meta=schedule_meta,
        )
        torch.musa.synchronize()
        logits = logits.cpu()
        if clean_logits:
            assert torch.equal(torch.isneginf(logits), ~valid)
        logits_masked = logits.masked_fill(~valid, 0)
        ref_logits_masked = ref_logits.masked_fill(~valid, 0)

    diff = calc_diff(logits_masked, ref_logits_masked)
    tol = 3e-5 if torch.bfloat16 in (weights_dtype, logits_dtype) else 5e-6
    assert diff < tol, f"{diff=}"


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("seq_q", [613, 2049, 3889])
@pytest.mark.parametrize("compressed_logits", [True, False])
@pytest.mark.parametrize(
    "seq_kv",
    [4001, 4096],
)
@pytest.mark.parametrize("disable_cp", [False, True])
@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
def test_mqa_logits(seq_q, seq_kv, compressed_logits, disable_cp, logits_dtype):
    torch.manual_seed(0)
    random.seed(0)

    device = torch.device("musa")
    num_heads, head_dim = 32, 32
    q = torch.randn(
        (seq_q, num_heads, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    kv = torch.randn(
        (seq_kv, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        (seq_q, num_heads),
        device=device,
        dtype=torch.float32,
    )

    if disable_cp:
        ks = torch.zeros(seq_q, dtype=torch.int32, device=device)
        ke = torch.arange(seq_q, dtype=torch.int32, device=device) + (seq_kv - seq_q)
    else:
        ks, ke = generate_cp_test_data(seq_q, seq_kv, device=device)

    ks = ks.clamp(0, seq_kv)
    ke = ke.clamp(0, seq_kv)
    ke = torch.maximum(ke, ks)

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, kv_scale = kv_cache_cast_to_fp8_for_nonpaged(kv)
    seq_kv_orig = seq_kv

    if compressed_logits:
        max_seqlen_k = (ke - ks).max().item()
        logits_comp = mate.deep_gemm.fp8_mqa_logits(
            q_fp8,
            (kv_fp8, kv_scale),
            weights,
            ks,
            ke,
            clean_logits=False,
            max_seqlen_k=max_seqlen_k,
            logits_dtype=logits_dtype,
        )
        assert logits_comp.shape == (seq_q, max_seqlen_k)

        logits = torch.full(
            (seq_q, seq_kv),
            float("-inf"),
            device=device,
            dtype=torch.float32,
        )
        for i in range(seq_q):
            start_i = int(ks[i].item())
            end_i = int(ke[i].item())
            length_i = max(0, end_i - start_i)
            if length_i > 0:
                logits[i, start_i:end_i] = logits_comp[i, :length_i]
    else:
        logits_pad = mate.deep_gemm.fp8_mqa_logits(
            q_fp8,
            (kv_fp8, kv_scale),
            weights,
            ks,
            ke,
            clean_logits=True,
            logits_dtype=logits_dtype,
        )
        logits = logits_pad[:, :seq_kv_orig]

    torch.musa.synchronize()

    do_check = seq_kv < 20000
    if do_check:
        ref_logits, ref_cost = ref_fp8_mqa_logits(
            q=q,
            kv=kv,
            weights=weights,
            cu_seqlen_ks=ks,
            cu_seqlen_ke=ke,
        )

        ref_neginf_mask = ref_logits == float("-inf")
        neginf_mask = logits == float("-inf")
        assert torch.equal(neginf_mask, ref_neginf_mask)

        logits_masked = logits.masked_fill(neginf_mask, 0)
        ref_logits_masked = ref_logits.masked_fill(ref_neginf_mask, 0)

        diff = calc_diff(logits_masked.float(), ref_logits_masked)
        assert diff < 1e-3, f"{diff=}"


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("ms_per_group", get_m_grouped_gemm_nt_masked_cases())
@pytest.mark.parametrize("n", [4096])
@pytest.mark.parametrize("k", [4096])
@pytest.mark.parametrize("expected_m", [None, 8192])
@pytest.mark.parametrize("data_type", [torch.bfloat16])
@pytest.mark.parametrize("enable_overlap", [False, True])
@pytest.mark.parametrize("backend", ["mubin", "mutlass"])
def test_m_grouped_bf16_gemm_nt_masked(
    ms_per_group,
    n,
    k,
    expected_m,
    data_type,
    enable_overlap,
    backend,
):
    if backend == "mutlass" and enable_overlap:
        pytest.skip('backend="mutlass" does not support enable_overlap')

    tile_signal = 64

    max_m = max(ms_per_group)
    expected_m = expected_m if expected_m is not None else max_m

    num_expert = len(ms_per_group)

    a = torch.rand((num_expert, max_m, k), device="musa", dtype=data_type)
    b = torch.rand((num_expert, n, k), device="musa", dtype=data_type)
    masked_m = torch.tensor(ms_per_group, device="musa", dtype=torch.int32)

    d = torch.empty((num_expert, max_m, n), device="musa", dtype=data_type)
    signal = torch.zeros(
        num_expert * ceil_div(max_m, tile_signal), dtype=torch.int32, device=a.device
    )

    ref_d = torch.einsum("bmk,bnk->bmn", a, b)
    res = mate.deep_gemm.bf16_m_grouped_gemm_nt_masked(
        a,
        b,
        d,
        masked_m,
        expected_m,
        enable_overlap=enable_overlap,
        signal=signal,
        backend=backend,
    )

    for i in range(num_expert):
        torch.testing.assert_close(
            d[i, : ms_per_group[i], :],
            ref_d[i, : ms_per_group[i], :],
            rtol=5e-3,
            atol=5e-3,
        )

    if enable_overlap:
        block_m = res[0]
        threshold = res[1]
        check_gemm_sbo_signal(num_expert, max_m, block_m, threshold, signal, masked_m)


def _quantize_fp8_fp4_operand(x, gran_k, fp4=False):
    rows, k = x.shape
    padded = torch.nn.functional.pad(x.float(), (0, align(k, gran_k) - k))
    if fp4:
        if _MXFP4_DTYPE is None:
            pytest.skip("Reference MXFP4 quantization requires torch.float4_e2m1fn_x2")
        quantized, codes = quantize_mxfp4(padded)
        payload = quantized.view(torch.int8)[:, : k // 2].contiguous()
        scale = torch.pow(2.0, codes.float() - 127)
        dequantized = dequantize_mxfp4(quantized, codes)
    elif gran_k == 32:
        quantized, codes = quantize_mxfp8(padded)
        payload = quantized[:, :k].contiguous()
        scale = torch.pow(2.0, codes.float() - 127)
        dequantized = dequantize_mxfp8(quantized, codes)
    else:
        # group_quantize_fp8 requires nonempty rows to infer its tile shape.
        if rows == 0:
            return (
                x.to(torch.float8_e4m3fn),
                x.new_empty((0, ceil_div(k, gran_k)), dtype=torch.float32),
                x.float(),
            )
        quantized, scale = group_quantize_fp8(
            padded,
            (rows, padded.size(-1) // gran_k),
            (1, gran_k),
        )
        payload = quantized[:, :k].contiguous()
        dequantized = quantized.float() * scale.repeat_interleave(gran_k, -1)
    return payload, scale, dequantized[:, :k]


def _run_fp8_fp4_grouped_case(
    counts,
    n,
    k,
    gran_k_a,
    scale_dtype,
    *,
    masked,
    alignment_m=256,
    max_m=4096,
    expected_m=1,
    check_quantization=True,
):
    groups = len(counts)
    aligned_ms = [align(count, alignment_m) for count in counts]
    m = max_m if masked else sum(aligned_ms)
    shape_a = (groups, m, k) if masked else (m, k)
    shape_d = (*shape_a[:-1], n)
    a = torch.empty(shape_a, dtype=torch.float8_e4m3fn, device="musa")
    b = torch.empty((groups, n, k // 2), dtype=torch.int8, device="musa")
    sfa = torch.empty((*shape_a[:-1], ceil_div(k, gran_k_a)), device="musa")
    sfb = torch.empty((groups, n, ceil_div(k, 32)), device="musa")
    d = torch.full(shape_d, -123.0, dtype=torch.bfloat16, device="musa")
    metadata = (
        torch.tensor(counts, dtype=torch.int32, device="musa")
        if masked
        else torch.full((m,), -1, dtype=torch.int32, device="musa")
    )
    references = []
    offset = 0
    # Generate and quantize one group at a time to bound temporary memory.
    for group, count in enumerate(counts):
        capacity = m if masked else aligned_ms[group]
        source_a = torch.randn((capacity, k), dtype=torch.bfloat16, device="musa")
        source_b = torch.randn((n, k), dtype=torch.bfloat16, device="musa")
        if not masked:
            source_a[count:] = 0
        qa, sa, da = _quantize_fp8_fp4_operand(source_a, gran_k_a)
        qb, sb, db = _quantize_fp8_fp4_operand(source_b, 32, fp4=True)
        references.append(
            (
                (source_a[:count].float() @ source_b.float().t()).bfloat16(),
                da[:count] @ db.t(),
            )
        )
        if masked:
            sa[count:] = 0
            a[group], sfa[group] = qa, sa
        else:
            rows = slice(offset, offset + capacity)
            a[rows], sfa[rows] = qa, sa
            metadata[offset : offset + count] = group
            offset += capacity
        b[group], sfb[group] = qb, sb
    sfa = _encode_sf(sfa, scale_dtype)
    sfb = _encode_sf(sfb, scale_dtype)
    kwargs = dict(
        recipe_a=(1, gran_k_a),
        recipe_b=(1, 32),
        disable_ue8m0_cast=scale_dtype != torch.float32,
    )
    if masked:
        wrapper.m_grouped_fp8_fp4_gemm_nt_masked(
            (a, sfa),
            (b, sfb),
            d,
            metadata,
            expected_m=expected_m,
            **kwargs,
        )
    else:
        wrapper.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a, sfa),
            (b, sfb),
            d,
            metadata,
            alignment_m=alignment_m,
            **kwargs,
        )
    offset = 0
    for group, count in enumerate(counts):
        actual = d[group, :count] if masked else d[offset : offset + count]
        original, quantized = references[group]
        if count:
            assert torch.isfinite(actual).all()
            assert torch.count_nonzero(original) > 0
            # DeepGEMM's W4A8 bound includes FP8/FP4 quantization error.
            if check_quantization:
                diff = calc_diff(actual, original)
                assert diff < 0.01, f"{group=}, {count=}, {n=}, {k=}, {diff=}"
            # Keep this reference in FP32: adjacent BF16 rounding choices can
            # differ by one ULP even when both are close to the FP32 result.
            torch.testing.assert_close(actual.float(), quantized, rtol=5e-3, atol=5e-3)
        if masked:
            assert torch.all(d[group, count:] == -123)
        offset += aligned_ms[group]
    # Contiguous padding output is unspecified and is not compared.


@supported_musa_compute_capability([32])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize(
    "counts,max_m,n,k,gran_k_a,scale_dtype,alignment_m",
    [
        ([0, 1, 4], 4, 4, 32, 32, torch.uint8, 128),
        ([0, 20, 257], 260, 260, 160, 128, torch.int32, 256),
        ([0, 0, 0], 4, 68, 96, 32, torch.float32, 128),
        ([1, 0, 3], 4, 4, 8, 32, torch.uint8, 128),
        ([0, 1, 4], 4, 68, 40, 32, torch.uint8, 128),
        ([17, 0, 129], 132, 132, 136, 32, torch.uint8, 256),
    ],
)
def test_m_grouped_fp8_fp4_gemm_boundaries(
    masked,
    counts,
    max_m,
    n,
    k,
    gran_k_a,
    scale_dtype,
    alignment_m,
    disable_mudnn_tf32,
):
    torch.manual_seed(0)
    _run_fp8_fp4_grouped_case(
        counts,
        n,
        k,
        gran_k_a,
        scale_dtype,
        masked=masked,
        alignment_m=alignment_m,
        max_m=max_m,
        expected_m=16,
        check_quantization=False,
    )


if __name__ == "__main__":
    test_paged_mqa_logits()
    test_mqa_logits()

# Grouped MX coverage follows DeepGEMM 78b69000794d0937b47ae3387eff7663410264d1,
# tests/test_fp8_fp4.py and tests/generators.py, adapted to MATE's native
# UE8M0 scale layout and supported MP32 kernels.
_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
_M_CASES = [
    pytest.param([255, 0, 256, 257], 512, 272, k, False, id=f"tails-k{k}")
    for k in (32, 128, 160, 4096)
] + [
    pytest.param([1], 256, 16, 64, False, id="single-token"),
    pytest.param([0, 0, 17, 0], 256, 128, 96, False, id="single-active-expert"),
    pytest.param([0, 0, 0, 0], 256, 256, 128, False, id="all-empty"),
    pytest.param([383, 0, 384], 512, 128, 160, False, id="large-m-small-n"),
    pytest.param([383, 0, 384], 512, 272, 160, False, id="large-m-large-n"),
    pytest.param([17, 97, 513, 0, 29, 256], 768, 512, 1024, True, id="random-uneven"),
]
_K_CASES = [
    pytest.param([32, 96, 160, 256], id="unaligned-expert-boundaries"),
    pytest.param([128, 4096], id="long-k"),
    pytest.param([32], id="single-expert"),
    pytest.param([32] * 8, id="many-small-experts"),
]


def _recipe_kwargs(api, mode):
    if mode == "default":
        return {}
    if mode == "separate":
        return {"recipe_a": (1, 32), "recipe_b": (1, 32)}
    return {"scale_granularity_mnk" if api == "mate_8bit" else "recipe": (1, 1, 32)}


_FP8_PAIRS = [
    pytest.param(a, b, id=f"{a_name}-{b_name}")
    for a_name, a in (("e4m3", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2))
    for b_name, b in (("e4m3", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2))
]


def _make_operand(shape, fp4, rng, random_quantization, fp8_dtype=torch.float8_e4m3fn):
    """Return payload, E8M0 bytes, decoded values and original CPU input."""
    if random_quantization:
        original = torch.randn(shape, generator=rng).to(torch.bfloat16).float()
        blocks = original.reshape(*shape[:-1], shape[-1] // 32, 32)
        exponent = (
            (
                blocks.abs().amax(-1).clamp_min(1e-4)
                / (6 if fp4 else torch.finfo(fp8_dtype).max)
            )
            .log2()
            .ceil()
        )
        scale = torch.pow(2.0, exponent)
        codes = (exponent + 127).to(torch.uint8)
        scaled = (blocks / scale.unsqueeze(-1)).reshape(shape)
        if fp4:
            boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
            magnitude = scaled.abs().contiguous()
            indices = torch.bucketize(magnitude, boundaries)
            # Round ties to the even E2M1 encoding.
            indices += (
                (indices % 2 == 1) & (magnitude == boundaries[indices.clamp_max(6)])
            ).long()
            indices |= ((scaled < 0) & (indices != 0)).long() << 3
            decoded = _E2M1[indices]
        else:
            payload = scaled.to(fp8_dtype)
            decoded = payload.float()
    else:
        if fp4:
            indices = torch.randint(0, 16, shape, generator=rng)
            decoded = _E2M1[indices]
        else:
            payload = (torch.randint(-6, 7, shape, generator=rng).float() / 2).to(
                fp8_dtype
            )
            decoded = payload.float()
        # Every group within a 128-element compute tile has a distinct scale.
        rows = torch.arange(shape[-2]).view(-1, 1)
        groups = torch.arange(shape[0]).view(-1, 1, 1)
        codes = (125 + (groups + rows + torch.arange(shape[-1] // 32)) % 4).to(
            torch.uint8
        )
        scale = torch.pow(2.0, codes.float() - 127)
        original = decoded * scale.repeat_interleave(32, -1)
    if fp4:
        payload = (indices[..., 0::2] | (indices[..., 1::2] << 4)).to(torch.int8)
    return payload, codes, decoded, original


def _mx_scale(codes):
    fsl = codes.transpose(-1, -2).contiguous().to("musa")
    # Expose logical [...,MN,Q] with MN-major physical storage.
    # Explicit strides matter when Q == 1: contiguous() can preserve the
    # arbitrary singleton stride of the original transpose.
    rows, k_groups = codes.shape[-2:]
    strides = (rows * k_groups, 1, rows) if codes.dim() == 3 else (1, rows)
    return fsl.as_strided(codes.shape, strides)


def _encode_sf(sf, dtype):
    if sf.dtype == torch.float32:
        if dtype == torch.float32:
            return sf
        codes = (sf.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    else:
        codes = sf
    if dtype == torch.float32:
        return torch.pow(2.0, codes.float() - 127)
    if dtype == torch.uint8:
        return codes
    # Poison unused K bytes: the transform must discard them, including 0xFF.
    padded = torch.full(
        (*codes.shape[:-1], ceil_div(codes.shape[-1], 4) * 4),
        255,
        dtype=torch.int64,
        device=codes.device,
    )
    padded[..., : codes.shape[-1]] = codes
    return sum(padded[..., i::4] << (8 * i) for i in range(4)).to(torch.int32)


def _sf_to_musa(sf, layout):
    if layout == "mn":
        return _mx_scale(sf)
    if layout == "mn_padded":
        rows, cols = sf.shape[-2:]
        storage = torch.empty(
            (*sf.shape[:-2], cols, rows + 4), dtype=sf.dtype, device="musa"
        )
        return storage[..., :rows].transpose(-1, -2).copy_(sf)
    if layout == "offset":
        storage = torch.empty(sf.numel() + 1, dtype=sf.dtype, device="musa")
        return storage[1:].view(sf.shape).copy_(sf)
    assert layout == "k"
    return sf.to("musa")


def _expand_sf(codes, mn, k, recipe):
    gran_mn, gran_k = recipe
    return codes.repeat_interleave(gran_mn, -2).repeat_interleave(gran_k // 32, -1)[
        ..., :mn, : ceil_div(k, 32)
    ]


def _check_transformed_sf_gemm(
    fp4,
    dtypes,
    recipes,
    masked,
    k=512,
    layouts=("k", "mn"),
    recipe_mode="triple",
    disable_ue8m0_cast=None,
    capture=False,
    fp4_b=None,
):
    fp4_b = fp4 if fp4_b is None else fp4_b
    rng = torch.Generator().manual_seed(912)
    groups, m, n = 2, 256, 272
    counts = [127, 17]
    inputs, references = [], []
    for i, (rows, dtype, recipe, layout) in enumerate(
        zip((m, n), dtypes, recipes, layouts)
    ):
        payload, _, decoded, _ = _make_operand(
            (groups, rows, k),
            fp4 if i == 0 else fp4_b,
            rng,
            False,
            torch.float8_e4m3fn if i == 0 else torch.float8_e5m2,
        )
        codes = torch.randint(
            125,
            129,
            (groups, ceil_div(rows, recipe[0]), ceil_div(k, recipe[1])),
            generator=rng,
            dtype=torch.uint8,
        )
        # Build the reference from source codes, independently of the transform.
        expanded = _expand_sf(codes, rows, k, recipe)
        references.append(
            decoded * torch.pow(2.0, expanded.float() - 127).repeat_interleave(32, -1)
        )
        if i == 0 and not masked:
            payload, codes = payload.flatten(0, 1), codes.flatten(0, 1)
        sf = _sf_to_musa(_encode_sf(codes, dtype), layout)
        inputs.append((payload.to("musa"), sf))
    valid = torch.arange(m).unsqueeze(0) < torch.tensor(counts).unsqueeze(1)
    metadata = (
        torch.tensor(counts, dtype=torch.int32, device="musa")
        if masked
        else torch.where(valid, torch.arange(groups).unsqueeze(1), -1)
        .flatten()
        .int()
        .to("musa")
    )
    shape = (groups, m, n) if masked else (groups * m, n)
    out = torch.full(shape, float("nan"), dtype=torch.bfloat16, device="musa")
    if masked:
        op = (
            wrapper.m_grouped_fp8_fp4_gemm_nt_masked
            if fp4 or fp4_b
            else wrapper.m_grouped_fp8_gemm_nt_masked
        )
        kwargs = {"expected_m": 72}
    else:
        op = (
            wrapper.m_grouped_fp8_fp4_gemm_nt_contiguous
            if fp4 or fp4_b
            else wrapper.m_grouped_fp8_gemm_nt_contiguous
        )
        kwargs = {"alignment_m": 128}
    if recipe_mode == "triple":
        kwargs["recipe"] = (recipes[0][0], recipes[1][0], recipes[0][1])
    elif recipe_mode == "pair":
        kwargs.update(recipe_a=recipes[0], recipe_b=recipes[1])
    else:
        assert recipe_mode == "default"
    kwargs["disable_ue8m0_cast"] = disable_ue8m0_cast

    def launch(**overrides):
        assert (
            op(
                *inputs,
                out,
                metadata,
                backend="mutlass" if not fp4 and fp4_b else "mubin",
                **(kwargs | overrides),
            )
            is None
        )

    if (
        not fp4
        and not fp4_b
        and disable_ue8m0_cast is False
        and recipe_mode == "triple"
        and not capture
    ):
        with pytest.raises(ValueError, match="disable_ue8m0_cast=False"):
            launch(disable_ue8m0_cast=True)
        with pytest.raises(ValueError, match="must be provided together"):
            launch(recipe=None, recipe_a=recipes[0])
        with pytest.raises(ValueError, match="either recipe or"):
            launch(recipe_a=recipes[0], recipe_b=recipes[1])

    if capture:
        stream = torch.musa.Stream()
        stream.wait_stream(torch.musa.current_stream())
        with torch.musa.stream(stream):
            launch()
        stream.synchronize()
        graph = torch.musa.MUSAGraph()
        with torch.musa.graph(graph, stream=stream):
            launch()
        with torch.musa.stream(stream):
            inputs[0][1].mul_(2)
            graph.replay()
        stream.synchronize()
        references[0].mul_(2)
    else:
        launch()

    # Explicit preprocessing and the native GEMM must match the wrapper path.
    operands = []
    for (payload, sf), recipe in zip(inputs, recipes):
        mn, q = payload.shape[-2], ceil_div(k, 32)
        scale = wrapper.transform_sf_into_required_layout(
            sf, mn, k, recipe, groups if sf.ndim == 3 else None
        )
        assert scale.dtype == torch.uint8
        assert scale.shape == (*payload.shape[:-1], q)
        assert scale.stride() == ((mn * q, 1, mn) if scale.ndim == 3 else (1, mn))
        operands.append((payload, scale))
    explicit_out = torch.full_like(out, float("nan"))
    _launch_m(
        "mate_mxfp", operands, explicit_out, metadata, masked, 72, "default", None, 128
    )
    actual = out.cpu().reshape(groups, m, n)[valid]
    torch.testing.assert_close(
        actual, explicit_out.cpu().reshape(groups, m, n)[valid], rtol=0, atol=0
    )
    golden = torch.cat(
        [
            references[0][g, :count] @ references[1][g].T
            for g, count in enumerate(counts)
        ]
    ).bfloat16()
    torch.testing.assert_close(actual, golden, rtol=0, atol=0)


@supported_musa_compute_capability([32])
@pytest.mark.parametrize(
    "fp4,fp4_b",
    [(False, False), (True, True), (False, True)],
    ids=["fp8", "fp4", "fp8-fp4"],
)
@pytest.mark.parametrize(
    "dtype,recipes,k,layouts,recipe_mode,disable_ue8m0_cast",
    [
        pytest.param(
            torch.float32,
            ((1, 32), (1, 32)),
            512,
            ("k", "mn"),
            "triple",
            False,
            id="fp32-native-explicit",
        ),
        pytest.param(
            torch.int32,
            ((1, 32), (1, 32)),
            512,
            ("mn", "k"),
            "pair",
            True,
            id="int32-native-pair",
        ),
        pytest.param(
            torch.uint8,
            ((1, 32), (1, 32)),
            512,
            ("k", "mn"),
            "default",
            None,
            id="uint8-native-default",
        ),
        pytest.param(
            torch.float32,
            ((1, 128), (128, 128)),
            512,
            ("k", "mn"),
            "default",
            None,
            id="fp32-coarse-default",
        ),
        pytest.param(
            torch.int32,
            ((1, 128), (1, 128)),
            160,
            ("offset", "offset"),
            "default",
            None,
            id="int32-coarse-default-offset-tail",
        ),
        pytest.param(
            torch.uint8,
            ((1, 128), (128, 128)),
            512,
            ("mn_padded", "k"),
            "triple",
            True,
            id="uint8-coarse-explicit-padded",
        ),
        pytest.param(
            torch.float32,
            ((1, 128), (1, 32)),
            160,
            ("offset", "mn_padded"),
            "pair",
            False,
            id="fp32-a-k128",
        ),
        pytest.param(
            torch.int32,
            ((1, 128), (1, 32)),
            160,
            ("offset", "mn_padded"),
            "pair",
            True,
            id="int32-a-k128",
        ),
        pytest.param(
            torch.uint8,
            ((1, 128), (1, 32)),
            160,
            ("offset", "mn_padded"),
            "pair",
            True,
            id="uint8-a-k128",
        ),
    ],
)
def test_mp32_masked_gemm_transformed_sf(
    fp4, fp4_b, dtype, recipes, k, layouts, recipe_mode, disable_ue8m0_cast
):
    if not fp4 and fp4_b and recipes[1] != (1, 32):
        pytest.skip("W4A8 source B scales require recipe (1, 32)")
    _check_transformed_sf_gemm(
        fp4,
        (dtype, dtype),
        recipes,
        masked=True,
        k=k,
        layouts=layouts,
        recipe_mode=recipe_mode,
        disable_ue8m0_cast=disable_ue8m0_cast,
        fp4_b=fp4_b,
    )


@supported_musa_compute_capability([32])
@pytest.mark.parametrize(
    "fp4,fp4_b",
    [(False, False), (True, True), (False, True)],
    ids=["fp8", "fp4", "fp8-fp4"],
)
def test_mp32_ragged_gemm_transformed_sf(fp4, fp4_b):
    _check_transformed_sf_gemm(
        fp4,
        (torch.float32, torch.int32),
        ((1, 128), (1, 32)) if not fp4 and fp4_b else ((1, 32), (1, 128)),
        masked=False,
        k=160,
        layouts=("offset", "offset"),
        recipe_mode="pair",
        fp4_b=fp4_b,
    )


@supported_musa_compute_capability([32])
def test_mp32_masked_gemm_transformed_sf_stream_capture():
    _check_transformed_sf_gemm(
        False,
        (torch.float32, torch.float32),
        ((1, 32), (1, 32)),
        masked=True,
        capture=True,
    )


def _launch_m(
    api, operands, out, metadata, masked, expected_m, recipe_mode, num_mp, alignment_m
):
    kwargs = _recipe_kwargs(api, recipe_mode)
    backend = (
        "mutlass"
        if operands[0][0].dtype == torch.float8_e4m3fn
        and operands[1][0].dtype == torch.int8
        else "mubin"
    )
    if not masked and alignment_m is not None:
        kwargs["alignment_m"] = alignment_m
    if api == "mate_mxfp":
        kwargs["num_mp"] = num_mp
    if api in ("mate_mxfp", "mate_8bit"):
        if masked:
            op = (
                gemm.masked_moe_gemm_mxfp
                if api == "mate_mxfp"
                else gemm.masked_moe_gemm_8bit
            )
            kwargs["expect_tokens"] = expected_m
        else:
            op = (
                gemm.ragged_m_moe_gemm_mxfp
                if api == "mate_mxfp"
                else gemm.ragged_m_moe_gemm_8bit
            )
        assert op(*operands, metadata, out, backend=backend, **kwargs) is out
    else:
        op = (
            wrapper.m_grouped_fp8_fp4_gemm_nt_masked
            if masked
            else wrapper.m_grouped_fp8_fp4_gemm_nt_contiguous
        )
        if masked:
            kwargs["expected_m"] = expected_m
        assert op(*operands, out, metadata, backend=backend, **kwargs) is None


def _check_m_grouped(
    api,
    fp4,
    masked,
    recipe_mode,
    counts,
    slot_m,
    n,
    k,
    random_quantization,
    dtype_a=torch.float8_e4m3fn,
    dtype_b=torch.float8_e4m3fn,
    num_mp=None,
    alignment_m=None,
    fp4_b=None,
):
    fp4_b = fp4 if fp4_b is None else fp4_b
    is_w4a8 = not fp4 and fp4_b
    rng = torch.Generator().manual_seed(1701)
    groups = len(counts)
    operands, fp8_operands, references, original_inputs = [], [], [], []
    for is_a, rows in ((True, slot_m), (False, n)):
        payload, codes, decoded, original = _make_operand(
            (groups, rows, k),
            fp4 if is_a else fp4_b,
            rng,
            random_quantization,
            dtype_a if is_a else dtype_b,
        )
        references.append(
            decoded * torch.pow(2.0, codes.float() - 127).repeat_interleave(32, -1)
        )
        original_inputs.append(original)
        if is_a and not masked:
            payload, codes, decoded = (
                x.flatten(0, 1) for x in (payload, codes, decoded)
            )
        scale = _mx_scale(codes)
        operands.append((payload.to("musa"), scale))
        if fp4 or fp4_b:
            # E4M3 can represent each E2M1 value exactly; preserve its scales.
            fp8_operands.append((decoded.to(torch.float8_e4m3fn).to("musa"), scale))
    valid = torch.arange(slot_m).unsqueeze(0) < torch.tensor(counts).unsqueeze(1)
    padding_value = -123.0 if is_w4a8 else float("nan")
    out = torch.full(
        (groups, slot_m, n), padding_value, dtype=torch.bfloat16, device="musa"
    )
    if masked:
        metadata = torch.tensor(counts, dtype=torch.int32, device="musa")
    else:
        metadata = (
            torch.where(valid, torch.arange(groups).unsqueeze(1), -1)
            .flatten()
            .int()
            .to("musa")
        )
        out = out.flatten(0, 1)
    # An estimate may be smaller or larger than the actual per-expert counts.
    expected_m = max(1, int(sum(counts) / groups * 1.2))
    launch = lambda ops, dest, mode=recipe_mode: _launch_m(
        api, ops, dest, metadata, masked, expected_m, mode, num_mp, alignment_m
    )
    original_metadata = metadata.clone()
    launch(operands, out)
    output = out.cpu().reshape(groups, slot_m, n)
    actual = output[valid]
    if masked and is_w4a8:
        assert torch.all(output[~valid] == padding_value)
    golden = torch.cat(
        [
            references[0][g, :count] @ references[1][g].T
            for g, count in enumerate(counts)
        ]
    )
    assert torch.isfinite(actual).all()
    if is_w4a8:
        # Compare W4A8 against the FP32 reference before BF16 output rounding.
        torch.testing.assert_close(actual.float(), golden, rtol=5e-3, atol=5e-3)
    else:
        torch.testing.assert_close(
            actual,
            golden.to(torch.bfloat16),
            atol=0 if fp4 else 2.0,
            rtol=0 if fp4 else 0.05,
        )
    if actual.numel() and golden.count_nonzero():
        assert calc_diff(actual, golden.to(torch.bfloat16)) < 1e-6
    if random_quantization:
        original_ref = torch.cat(
            [
                original_inputs[0][g, :count] @ original_inputs[1][g].T
                for g, count in enumerate(counts)
            ]
        )
        quantization_tolerance = (
            0.01
            if is_w4a8
            else 0.02
            if fp4 or fp4_b
            else 0.005
            if torch.float8_e5m2 in (dtype_a, dtype_b)
            else 0.001
        )
        assert calc_diff(actual, original_ref) < quantization_tolerance
    # Compare only valid rows: padding is not a zero-output guarantee.
    for _ in range(3):
        out.fill_(padding_value)
        launch(operands, out)
        output = out.cpu().reshape(groups, slot_m, n)
        torch.testing.assert_close(output[valid], actual, rtol=0, atol=0)
        if masked and is_w4a8:
            assert torch.all(output[~valid] == padding_value)
    torch.testing.assert_close(metadata, original_metadata, rtol=0, atol=0)
    if fp4 or fp4_b:
        out.fill_(padding_value)
        launch(fp8_operands, out, mode="explicit")
        torch.testing.assert_close(
            out.cpu().reshape(groups, slot_m, n)[valid], actual, rtol=0, atol=0
        )


_MX_FORMATS = [
    pytest.param(False, False, *pair.values, id=pair.id) for pair in _FP8_PAIRS
] + [
    pytest.param(True, True, torch.float8_e4m3fn, torch.float8_e4m3fn, id="fp4"),
    pytest.param(False, True, torch.float8_e4m3fn, torch.float8_e4m3fn, id="fp8-fp4"),
]


@supported_musa_compute_capability([32])
@pytest.mark.parametrize("counts,slot_m,n,k,random_quantization", _M_CASES)
@pytest.mark.parametrize("fp4,fp4_b,dtype_a,dtype_b", _MX_FORMATS)
@pytest.mark.parametrize(
    "api,num_mp",
    [("mate_mxfp", 1), ("mate_mxfp", None), ("wrapper", None)],
    ids=["native-one-cta", "native-device-mps", "wrapper"],
)
def test_mp32_masked_moe_gemm_mxfp(
    api, counts, slot_m, n, k, random_quantization, fp4, fp4_b, dtype_a, dtype_b, num_mp
):
    # One CTA forces persistent reuse across tiles and expert boundaries.
    _check_m_grouped(
        api,
        fp4,
        True,
        "default",
        counts,
        slot_m,
        n,
        k,
        random_quantization,
        dtype_a,
        dtype_b,
        num_mp,
        fp4_b=fp4_b,
    )


@supported_musa_compute_capability([32])
@pytest.mark.parametrize(
    "counts,slot_m,n,k,random_quantization",
    [_M_CASES[0], _M_CASES[3], _M_CASES[-3], _M_CASES[-2], _M_CASES[-1]],
)
@pytest.mark.parametrize("fp4,fp4_b,dtype_a,dtype_b", _MX_FORMATS)
@pytest.mark.parametrize("alignment_m", [128, 256])
@pytest.mark.parametrize("api", ["mate_mxfp", "wrapper"])
def test_mp32_ragged_moe_gemm_mxfp(
    api,
    counts,
    slot_m,
    n,
    k,
    random_quantization,
    fp4,
    fp4_b,
    dtype_a,
    dtype_b,
    alignment_m,
):
    # M128 must handle expert boundaries that are not multiples of 256.
    if alignment_m == 128:
        slot_m -= 128
    _check_m_grouped(
        api,
        fp4,
        False,
        "default",
        counts,
        slot_m,
        n,
        k,
        random_quantization,
        dtype_a,
        dtype_b,
        num_mp=1,
        alignment_m=alignment_m,
        fp4_b=fp4_b,
    )


@supported_musa_compute_capability([32])
@pytest.mark.parametrize("fp4,fp4_b,dtype_a,dtype_b", _MX_FORMATS)
def test_mp32_ragged_mxfp_default_alignment(fp4, fp4_b, dtype_a, dtype_b):
    _check_m_grouped(
        "mate_mxfp",
        fp4,
        False,
        "default",
        [127, 0, 129],
        384,
        272,
        160,
        False,
        dtype_a,
        dtype_b,
        num_mp=1,
        fp4_b=fp4_b,
    )


@supported_musa_compute_capability([32])
@pytest.mark.parametrize("fp4,fp4_b,dtype_a,dtype_b", _MX_FORMATS)
def test_mp32_contiguous_mxfp_wrapper_default_alignment(fp4, fp4_b, dtype_a, dtype_b):
    assert wrapper.get_mk_alignment_for_contiguous_layout() == 128
    _check_m_grouped(
        "wrapper",
        fp4,
        False,
        "default",
        [127, 0, 129],
        384,
        272,
        160,
        False,
        dtype_a,
        dtype_b,
        fp4_b=fp4_b,
    )


def _check_k_grouped(
    api,
    out_dtype,
    ks,
    quant_tile,
    explicit_recipe,
    c_mode="separate",
    dtype_a=torch.float8_e4m3fn,
    dtype_b=torch.float8_e4m3fn,
    num_mp=None,
):
    rng = torch.Generator().manual_seed(1702)
    m, n = (16, 32) if len(ks) == 1 else (272, 272)
    operands, references = [], []
    for rows, dtype in ((m, dtype_a), (n, dtype_b)):
        values = (
            torch.randint(-40, 41, (sum(ks), rows), generator=rng).float() / 7
        ).to(dtype)
        codes = (
            125
            + (torch.arange(sum(ks) // quant_tile).view(-1, 1) + torch.arange(rows)) % 4
        ).to(torch.uint8)
        scales = torch.pow(2.0, codes.float() - 127)
        references.append(values.float() * scales.repeat_interleave(quant_tile, 0))
        operands.append(
            (values.to("musa"), (codes if quant_tile == 32 else scales).to("musa"))
        )
    metadata = torch.tensor(ks, dtype=torch.int32, device="musa")
    initial = ((torch.arange(len(ks) * m * n).reshape(len(ks), m, n) % 13 - 6) / 4).to(
        out_dtype
    )
    out = initial.to("musa")
    c = (
        initial.to("musa")
        if c_mode == "separate"
        else out
        if c_mode == "alias"
        else None
    )
    recipe = (1, 1, quant_tile)

    def launch(dest, accumulator):
        kwargs = {"recipe": recipe} if explicit_recipe else {}
        if api == "mate_mxfp":
            result = gemm.ragged_k_moe_gemm_mxfp(
                *operands, metadata, dest, ks_cpu=ks, num_mp=num_mp, **kwargs
            )
        elif api == "mate_8bit":
            kwargs = {"scale_granularity_mnk": recipe} if explicit_recipe else {}
            result = gemm.ragged_k_moe_gemm_8bit(
                *operands, metadata, dest, ks_cpu=ks, num_mp=num_mp, **kwargs
            )
        else:
            result = wrapper.k_grouped_fp8_gemm_tn_contiguous(
                *operands,
                dest,
                ks if quant_tile == 32 else None,
                metadata,
                c=accumulator,
                **kwargs,
            )
        assert result is dest

    if c_mode == "none":
        out.fill_(float("nan"))
    launch(out, c)
    actual = out.cpu()
    offset = 0
    for g, k in enumerate(ks):
        golden = (
            references[0][offset : offset + k].T @ references[1][offset : offset + k]
        )
        golden += initial[g] if c_mode != "none" else 0
        torch.testing.assert_close(actual[g], golden.to(out_dtype), atol=2.0, rtol=0.05)
        assert calc_diff(actual[g], golden.to(out_dtype)) < 1e-6
        offset += k
    for _ in range(3):
        out.copy_(initial) if c_mode != "none" else out.fill_(float("nan"))
        launch(out, c)
        torch.testing.assert_close(out.cpu(), actual, rtol=0, atol=0)
    if api == "deep_gemm" and c_mode == "separate":
        torch.testing.assert_close(c.cpu(), initial, rtol=0, atol=0)
    if c_mode == "none":
        fp32_baseline = torch.zeros_like(out, dtype=torch.float32)
        launch(fp32_baseline, fp32_baseline)
        torch.testing.assert_close(
            actual, fp32_baseline.cpu().to(out_dtype), rtol=0, atol=0
        )


@supported_musa_compute_capability([32])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("ks", _K_CASES)
@pytest.mark.parametrize("dtype_a,dtype_b", _FP8_PAIRS)
def test_mp32_k_grouped_mxfp(out_dtype, ks, dtype_a, dtype_b):
    _check_k_grouped(
        "mate_mxfp",
        out_dtype,
        ks,
        32,
        False,
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        num_mp=1,
    )


@supported_musa_compute_capability([32])
@pytest.mark.parametrize("c_mode", ["none", "alias", "separate"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_mp32_k_grouped_wrapper_accumulator(c_mode, out_dtype):
    _check_k_grouped(
        "deep_gemm",
        out_dtype,
        [32, 96, 160, 256],
        32,
        False,
        c_mode,
        torch.float8_e5m2,
        torch.float8_e4m3fn,
    )


@supported_musa_compute_capability([32])
@pytest.mark.parametrize("fp4,fp4_b,dtype_a,dtype_b", _MX_FORMATS)
def test_mp32_masked_mxfp_wrapper(fp4, fp4_b, dtype_a, dtype_b):
    _check_m_grouped(
        "deep_gemm",
        fp4,
        True,
        "separate",
        [17, 0, 257],
        512,
        272,
        160,
        False,
        dtype_a,
        dtype_b,
        fp4_b=fp4_b,
    )


@supported_musa_compute_capability([32])
@pytest.mark.parametrize("recipe_mode", ["explicit", "separate"])
def test_mp32_masked_mxfp_recipes(recipe_mode):
    _check_m_grouped(
        "mate_mxfp",
        False,
        True,
        recipe_mode,
        [17, 0, 257],
        512,
        272,
        160,
        False,
        num_mp=1,
    )


@supported_musa_compute_capability([32])
def test_mp32_8bit_masked_compatibility():
    _check_m_grouped(
        "mate_8bit",
        False,
        True,
        "default",
        [17, 0, 257],
        512,
        272,
        160,
        False,
        torch.float8_e5m2,
        torch.float8_e4m3fn,
    )


@supported_musa_compute_capability([32])
def test_mp32_8bit_k_grouped_compatibility():
    _check_k_grouped(
        "mate_8bit",
        torch.float32,
        [32, 160],
        32,
        False,
        dtype_a=torch.float8_e4m3fn,
        dtype_b=torch.float8_e5m2,
        num_mp=1,
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("explicit_recipe", [False, True], ids=["default", "explicit"])
def test_mp31_k_grouped_fp8(explicit_recipe):
    _check_k_grouped("mate_8bit", torch.float32, [128, 256], 128, explicit_recipe)
