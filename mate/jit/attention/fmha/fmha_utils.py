from functools import cache, lru_cache

import torch

from ... import env as jit_env
from ....utils import ceil_div, cc_switch

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_ELEMENT_NAME_SUFFIX = {
    "mutlass::float_e4m3_t": "fp8_e4m3",
    "mutlass::float_e5m2_t": "fp8_e5m2",
    "mutlass::bfloat16_t": "bf16",
    "mutlass::half_t": "fp16",
}

FMHA_EXTRA_CUDA_CFLAGS = [
    "-Od3",
    "-DNDEBUG",
    "-fno-strict-aliasing",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-load-cluster-mutation=1",
    "-mllvm",
    "--num-dwords-of-load-in-mutation=64",
]


def fmha_extra_include_paths():
    return [
        jit_env.MATE_INCLUDE_DIR,
        jit_env.MATE_CSRC_DIR,
        jit_env.MUTLASS_INCLUDE_DIR,
    ]


def _resolve_mask(
    seqlen_q,
    seqlen_k,
    is_causal,
    window_size_left,
    window_size_right,
    attention_chunk=0,
):
    if window_size_left is None or window_size_left >= seqlen_k - 1:
        window_size_left = -1
    if window_size_right is None or window_size_right >= seqlen_q - 1:
        window_size_right = -1

    if is_causal:
        window_size_right = 0

    is_causal = window_size_left < 0 and window_size_right == 0 and attention_chunk == 0
    is_local = (
        window_size_left >= 0 or window_size_right >= 0 or attention_chunk >= 1
    ) and not is_causal

    # chunk
    if window_size_left < 0:
        window_size_left = seqlen_k - 1
    if window_size_right < 0:
        window_size_right = seqlen_q - 1
    if attention_chunk > 0:
        window_size_left = min(window_size_left, attention_chunk - 1)
        window_size_right = min(window_size_right, attention_chunk - 1)

    return is_causal, is_local, window_size_left, window_size_right


@lru_cache
def _roundup_headdim(headdim: int, headdim_v: int):
    # round up headdim
    if headdim <= 64:
        headdim = 64
    elif headdim <= 128:
        headdim = 128
    elif headdim <= 192:
        headdim = 192
    elif headdim <= 256:
        headdim = 256
    elif headdim <= 384:
        headdim = 384
    else:
        headdim = 512

    if headdim_v <= 64:
        headdim_v = 64
    elif headdim_v <= 128:
        headdim_v = 128
    elif headdim_v <= 192:
        headdim_v = 192
    elif headdim_v <= 256:
        headdim_v = 256
    elif headdim_v <= 384:
        headdim_v = 384
    else:
        headdim_v = 512

    return (headdim, headdim_v)


@lru_cache
def _check_enable_packgqa(m: int):
    return m <= 256


@lru_cache
def _get_tile_m(m: int, head_ratio: int, enable_packgqa: bool = None):
    enable_packgqa = (
        _check_enable_packgqa(m) if enable_packgqa is None else enable_packgqa
    )

    assert m is not None
    assert head_ratio is not None

    m = m * head_ratio if enable_packgqa else m
    if m <= 16:
        tile_m = 16
    elif m <= 32:
        tile_m = 32
    elif m <= 64:
        tile_m = 64
    elif m <= 128:
        tile_m = 128
    elif m <= 192:
        tile_m = 192
    else:
        tile_m = 256

    return tile_m, enable_packgqa


@cache
def _get_mp31_qv_stages_by_smem(
    tile_m: int, tile_n: int, headdim: int, headdim_v: int, element_size: int
):
    smem_limit = 192 * 1024

    def estimate(stages: int):
        smem_q = tile_m * headdim
        smem_qv = tile_m * headdim_v
        smem_k = tile_n * headdim * stages
        smem_v_for_qv = tile_n * headdim_v * stages
        smem_p = tile_m * tile_n
        smem_v_for_pv = tile_n * headdim_v * stages
        return (
            smem_q + smem_qv + smem_k + smem_v_for_qv + smem_p + smem_v_for_pv
        ) * element_size

    return (2, 2) if estimate(2) <= smem_limit else (1, 1)


@cache
def round_multiple(x, m):
    return (x + m - 1) // m * m


_MP32_SMEM_CAPACITY_BYTES = 256 * 1024
_SMEM_ARRAY_ALIGNMENT_BYTES = 16


@cache
def _get_mp32_stages_by_smem(
    tile_m: int,
    tile_n: int,
    headdim: int,
    headdim_v: int,
    element_size: int,
    has_qv: bool,
    mma_pv_rs: bool,
) -> tuple[int, int]:
    def array_aligned_size(numel: int) -> int:
        return round_multiple(max(numel * element_size, 1), _SMEM_ARRAY_ALIGNMENT_BYTES)

    def shared_storage_size(stages: int) -> int:
        size = sum(
            (
                array_aligned_size(tile_m * headdim),  # smem_q
                array_aligned_size(stages * tile_n * headdim),  # smem_k
                array_aligned_size(stages * tile_n * headdim_v),  # smem_v
            )
        )
        if has_qv:
            size += array_aligned_size(tile_m * headdim_v)  # smem_qv
            size += array_aligned_size(stages * tile_n * headdim_v)  # smem_vt
        if not mma_pv_rs:
            size += array_aligned_size(tile_m * tile_n)  # smem_p
        return size

    for stages in (2, 1):
        if shared_storage_size(stages) <= _MP32_SMEM_CAPACITY_BYTES:
            return stages, stages

    raise ValueError(
        "MP32 FMHA shared storage exceeds 256 KiB for "
        f"tile_m={tile_m}, tile_n={tile_n}, "
        f"headdim={headdim}, headdim_v={headdim_v}"
    )


@lru_cache
def _dtype_bits(dtype: torch.dtype) -> int:
    return dtype.itemsize * 8


@cc_switch()
def _get_fwd_kernel_config(
    m: int,
    head_ratio: int,
    headdim: int,
    headdim_v: int,
    element_size: int,
    enable_packgqa: bool = None,
    has_qv: bool = False,
    is_fp8: bool = False,
    is_high_regpressure: bool = False,
):
    raise NotImplementedError("This function should be specialized for each CC")


@_get_fwd_kernel_config.register(31)
@lru_cache
def _get_fwd_kernel_config_mp31(
    m: int,
    head_ratio: int,
    headdim: int,
    headdim_v: int,
    element_size: int,
    enable_packgqa: bool = None,
    has_qv: bool = False,
    is_fp8: bool = False,
    is_high_regpressure: bool = False,
):
    mma_pv_rs = False
    packgqa_alignment = 128 // (element_size * 8)
    if headdim % packgqa_alignment != 0 or headdim_v % packgqa_alignment != 0:
        enable_packgqa = False
    headdim, headdim_v = _roundup_headdim(headdim, headdim_v)

    candidate_tile_m, enable_packgqa = _get_tile_m(m, head_ratio, enable_packgqa)
    if headdim % 128 != 0:
        candidate_tile_m = max(candidate_tile_m, 64 if is_fp8 else 32)
    if is_fp8:
        candidate_tile_m = max(candidate_tile_m, 32)
    tile_head_dim = 0

    if is_fp8 and has_qv:
        tile_m = min(candidate_tile_m, 32)
        tile_n = 128
        stages_k, stages_v = _get_mp31_qv_stages_by_smem(
            tile_m, tile_n, headdim, headdim_v, element_size
        )
    elif not is_fp8:
        tile_n = 64
        if has_qv:
            match (headdim, headdim_v):
                case (64, 64):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 256), 4, 4
                case (64, 128):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 256), 1, 1
                case (64, 256):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 128), 1, 1
                case (64, 512):
                    tile_m = min(candidate_tile_m, 128)
                    stages_k, stages_v, tile_head_dim = (
                        (4, 4, 64) if tile_m > 32 else (1, 1, 0)
                    )
                case (128, 128):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 192), 1, 1
                case (192, 128):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 128), 1, 1
                case _:
                    assert False, f"Add QV config for headdim {headdim}-{headdim_v}"
        else:
            match (headdim, headdim_v):
                case (64, 64):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 256), 2, 2
                case (64, 256):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 256), 3, 3
                case (64, 512):
                    tile_m = min(candidate_tile_m, 128)
                    stages_k, stages_v, tile_head_dim = (
                        (4, 4, 64) if tile_m > 32 else (1, 1, 0)
                    )
                case (128, 128):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 256), 3, 3
                case (192, 128) | (192, 192):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 256), 1, 1
                case (256, 256):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 192), 1, 1
                case (384, 384):
                    tile_m = min(candidate_tile_m, 128)
                    stages_k, stages_v, tile_head_dim = (
                        (4, 4, 128) if tile_m > 32 else (1, 1, 0)
                    )
                case (512, 512):
                    tile_m = min(candidate_tile_m, 128)
                    stages_k, stages_v, tile_head_dim = (
                        (3, 3, 128) if tile_m > 32 else (1, 1, 0)
                    )
                case _:
                    assert False, f"Add config for headdim {headdim}-{headdim_v}"

    if is_fp8 and not has_qv:
        match (headdim, headdim_v, is_high_regpressure):
            case (
                (64, 64, _) | (128, 128, False) | (192, 128, False) | (192, 192, False)
            ):
                tile_m = min(candidate_tile_m, 256)
                tile_n = 128
                stages_k = 2
                stages_v = 2
            case (256, 256, False):
                tile_m = min(candidate_tile_m, 192)
                tile_n = 128
                stages_k = 1
                stages_v = 1
            case (
                (128, 128, True)
                | (192, 128, True)
                | (192, 192, True)
                | (256, 256, True)
                | (384, 384, _)
            ):
                tile_m = min(candidate_tile_m, 128)
                tile_n = 128
                stages_k = 1
                stages_v = 1
            case (512, 512, _):
                tile_m = min(candidate_tile_m, 32)
                tile_n = 128
                stages_k = 1
                stages_v = 1
            case _:
                assert False, f"Add FP8 config for headdim {headdim}-{headdim_v}"

    multi_chunk = tile_head_dim > 0
    consumers_qk = consumers_pv = ceil_div(tile_m, 32 if multi_chunk else 64)

    return (
        tile_m,
        tile_n,
        stages_k,
        stages_v,
        headdim,
        headdim_v,
        consumers_qk,
        consumers_pv,
        tile_head_dim,
        enable_packgqa,
        mma_pv_rs,
        31,  # cc
    )


@_get_fwd_kernel_config.register(32)
@lru_cache
def _get_fwd_kernel_config_mp32(
    m: int,
    head_ratio: int,
    headdim: int,
    headdim_v: int,
    element_size: int,
    enable_packgqa: bool = None,
    has_qv: bool = False,
    is_fp8: bool = False,
    is_high_regpressure: bool = False,
):
    headdim, headdim_v = _roundup_headdim(headdim, headdim_v)
    if enable_packgqa is None:
        enable_packgqa = _check_enable_packgqa(m)

    candidate_tile_m, enable_packgqa = _get_tile_m(m, head_ratio, enable_packgqa)
    if candidate_tile_m == 192 and headdim % 128 != 0:
        candidate_tile_m = 128

    match (element_size, has_qv, (headdim, headdim_v)):
        # Qv
        # NOTE: For Qv, tile_n must be 64. Otherwise, smem will explode.
        case (1, True, _):  # FP8 Qv
            tile_m, tile_n, consumer_m = 128, 64, 64
        case (2, True, _):  # BF16 Qv
            tile_m, tile_n, consumer_m = 32, 64, 32

        # FP8 Normal Attention
        case (1, False, (128, 128) | (256, 256)):
            # 7 Consumers
            tile_m, tile_n, consumer_m = 224, 128, 32
        case (1, False, (192, 128)):
            # 6 Consumers
            tile_m, tile_n, consumer_m = 192, 128, 32
        case (1, False, (384, 384) | (512, 512)):
            tile_m, tile_n, consumer_m = 128, 128, 64
        case (1, False, _):
            tile_m, tile_n, consumer_m = max(candidate_tile_m, 128), 128, 64

        # BF16 / FP16 Normal Attention
        case (2, False, (256, 256)):
            tile_m, tile_n, consumer_m = candidate_tile_m, 64, 64
        case (2, False, (384, 384)):
            tile_m, tile_n, consumer_m = min(candidate_tile_m, 128), 128, 64
        case (2, False, (512, 512)):
            tile_m, tile_n, consumer_m = min(candidate_tile_m, 64), 128, 64
        case (2, False, _):
            tile_m, tile_n, consumer_m = candidate_tile_m, 128, 64

        case _:
            raise ValueError(f"Unsupported dtype with {element_size}B")

    # NOTE: No tile_m = 16 for MP32.
    tile_m = max(tile_m, 32)
    mma_pv_rs = True

    stages_k, stages_v = _get_mp32_stages_by_smem(
        tile_m,
        tile_n,
        headdim,
        headdim_v,
        element_size,
        has_qv,
        mma_pv_rs,
    )

    consumers_qk = consumers_pv = ceil_div(tile_m, consumer_m)
    # TODO: TileN must be greater than AppendKVRows
    # (NumMmaThreads / GmemThreadsPerRow). When consumers = 3, NumMmaThreads = 384,
    # GmemThreadsPerRow = 4, AppendKVRows = 96, cannot use TileN = 64.

    return (
        tile_m,
        tile_n,
        stages_k,
        stages_v,
        headdim,
        headdim_v,
        consumers_qk,
        consumers_pv,
        0,  # tile_head_dim
        enable_packgqa,
        mma_pv_rs,
        32,  # cc
    )


@cc_switch()
def _select_lsu_load_kv(
    *,
    paged_kv: bool,
    page_size: int,
    tile_n: int,
    has_cu_seqlens_k: bool,
    has_leftpad_k: bool,
    has_qv: bool,
    dtype: torch.dtype,
) -> tuple[bool, bool]:
    raise NotImplementedError("This function should be specialized for each CC")


@_select_lsu_load_kv.register(31)
@lru_cache
def _select_lsu_load_kv_mp31(
    *,
    paged_kv: bool,
    page_size: int,
    tile_n: int,
    has_cu_seqlens_k: bool,
    has_leftpad_k: bool,
    has_qv: bool,
    dtype: torch.dtype,
) -> tuple[bool, bool]:
    paged_tme_ok = paged_kv and page_size == tile_n and not has_leftpad_k
    use_lsu_load_k = not paged_tme_ok
    use_lsu_load_v = (paged_kv and not paged_tme_ok) or (has_qv and not paged_kv)
    return use_lsu_load_k, use_lsu_load_v


@_select_lsu_load_kv.register(32)
@lru_cache
def _select_lsu_load_kv_mp32(
    *,
    paged_kv: bool,
    page_size: int,
    tile_n: int,
    has_cu_seqlens_k: bool,
    has_leftpad_k: bool,
    has_qv: bool,
    dtype: torch.dtype,
) -> tuple[bool, bool]:
    tme_ok = not (paged_kv and (page_size != tile_n or has_leftpad_k))
    return not tme_ok, not tme_ok


@cc_switch()
def _check_valid_asm_input(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    page_table,
    seqused_q,
    seqused_k,
    qv,
    window_size,
    learnable_sink,
    attention_chunk,
    softcap,
    cp_world_size=1,
):
    raise NotImplementedError("This function should be specialized for each CC")


@_check_valid_asm_input.register(31)
def _check_valid_asm_input_mp31(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    page_table,
    seqused_q,
    seqused_k,
    qv,
    window_size,
    learnable_sink,
    attention_chunk,
    softcap,
    cp_world_size=1,
):
    if not (q.is_musa and k.is_musa and v.is_musa):
        return False

    supported_dtypes = (torch.float16, torch.bfloat16)
    if (
        q.dtype not in supported_dtypes
        or k.dtype not in supported_dtypes
        or v.dtype not in supported_dtypes
        or q.dtype != k.dtype
        or q.dtype != v.dtype
    ):
        return False

    if (
        q.dim() not in (3, 4)
        or k.dim() not in (3, 4)
        or v.dim() not in (3, 4)
        or q.dim() != k.dim()
        or q.dim() != v.dim()
    ):
        return False

    headdim_qk = q.shape[-1]
    headdim_v = v.shape[-1]

    is_192_128 = headdim_qk == 192 and headdim_v == 128
    is_128_128_or_less = headdim_qk == headdim_v and headdim_qk <= 128

    window_size_left, window_size_right = window_size
    if (
        not (is_192_128 or is_128_128_or_less)
        or page_table is not None
        or seqused_q is not None
        or seqused_k is not None
        or (window_size_left is not None and window_size_left >= 0)
        or (window_size_right is not None and window_size_right >= 0)
        or qv is not None
        or softcap != 0.0
        or learnable_sink is not None
        or attention_chunk != 0
        or cp_world_size != 1
    ):
        return False

    if q.dim() == 3:
        total_seq_q, nr_heads, headdim_qk = q.shape
        total_seq_kv, nr_heads_kv, _ = k.shape
        _, _, headdim_v = v.shape

        if (
            cu_seqlens_q is None
            or cu_seqlens_k is None
            or max_seqlen_q is None
            or max_seqlen_k is None
        ):
            return False

        return (
            k.shape == (total_seq_kv, nr_heads_kv, headdim_qk)
            and v.shape == (total_seq_kv, nr_heads_kv, headdim_v)
            and cu_seqlens_q.is_musa
            and cu_seqlens_k.is_musa
            and cu_seqlens_k.numel() == cu_seqlens_q.numel()
        )

    batch, seq_q, nr_heads, headdim_qk = q.shape
    _, seq_kv, nr_heads_kv, _ = k.shape
    _, _, _, headdim_v = v.shape

    return k.shape == (
        batch,
        seq_kv,
        nr_heads_kv,
        headdim_qk,
    ) and v.shape == (batch, seq_kv, nr_heads_kv, headdim_v)


@_check_valid_asm_input.register(32)
def _check_valid_asm_input_mp32(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    page_table,
    seqused_q,
    seqused_k,
    qv,
    window_size,
    learnable_sink,
    attention_chunk,
    softcap,
    cp_world_size=1,
):
    # For now, no MP32 asm kernel ready.
    return False
