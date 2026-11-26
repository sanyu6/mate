from dataclasses import asdict, replace
import functools
import hashlib
import json
import math
from pathlib import Path

import torch

from mate.artifacts import (
    ArtifactPath,
    KernelMapHash,
    ensure_mubin_kernel_artifact,
    load_kernel_map,
)
from mate.utils import ceil_div

from ..common import (
    Arch,
    MP31_ARCH,
    MP32_ARCH,
    get_asm_dtype_from_torch_dtype,
)
from .types import (
    AsmDType,
    GemmKernelBlock,
    GemmMubinId,
    MoeGemmMode,
    MxScaleMode,
    TensorMajor,
    TensorQuantMode,
)

RAGGED_BLOCK_BY_ALIGNMENT_M = {
    128: GemmKernelBlock(num_thread=256, tile_m=128, num_buffer=4, num_squad_m=1),
    256: GemmKernelBlock(num_thread=512, tile_m=256, num_buffer=2, num_squad_m=2),
}


MASKED_BLOCK_BY_SELECT_M = {
    128: GemmKernelBlock(num_thread=256, tile_m=128, num_buffer=4, num_squad_m=1),
    256: GemmKernelBlock(num_thread=512, tile_m=256, num_buffer=2, num_squad_m=2),
}


PACKED_4BIT_BLOCK_BY_SELECT_M = {
    128: GemmKernelBlock(
        num_thread=256,
        tile_m=128,
        tile_n=256,
        tile_k=128,
        num_buffer=2,
        num_squad_m=1,
        num_squad_n=2,
    ),
    256: GemmKernelBlock(
        num_thread=512,
        tile_m=256,
        tile_n=256,
        tile_k=128,
        num_buffer=2,
        num_squad_m=1,
        num_squad_n=4,
    ),
}


# Standard MXFP4 profiles are deliberately kept separate from the MP31 W4A8
# table and target MP32.  As in Mate-ASM, ``tile_k`` is a byte budget.  For
# pure FP4 the 128-byte profile covers 256 logical K elements, while the
# launcher passes the 128-byte physical TME tile dimensions for both packed
# operands.
MP32_GROUPED_BLOCK = GemmKernelBlock(
    num_thread=512,
    tile_m=256,
    tile_n=256,
    tile_k=128,
    num_buffer=4,
    blk_per_mp=1,
    num_squad_m=2,
    num_squad_n=2,
    macro_tile_x=2,
    switch_swizzle=1,
)

MP32_RAGGED_BLOCK_BY_ALIGNMENT_M = {
    128: replace(MP32_GROUPED_BLOCK, num_thread=256, tile_m=128, num_squad_m=1),
    256: MP32_GROUPED_BLOCK,
}


def _get_mp32_ragged_block(alignment_m: int) -> GemmKernelBlock:
    if alignment_m not in MP32_RAGGED_BLOCK_BY_ALIGNMENT_M:
        raise ValueError("alignment_m must be 128 or 256")
    return MP32_RAGGED_BLOCK_BY_ALIGNMENT_M[alignment_m]


RAGGED_EXPERT_LAYOUT_BLOCK_CANDIDATES = [
    (
        GemmKernelBlock(
            num_thread=512,
            tile_m=256,
            tile_n=256,
            tile_k=128,
            num_buffer=2,
            num_squad_m=2,
            num_squad_n=2,
        ),
        1.0,
    ),
    (
        GemmKernelBlock(
            num_thread=256,
            tile_m=128,
            tile_n=256,
            tile_k=128,
            num_buffer=4,
            num_squad_m=1,
            num_squad_n=2,
        ),
        0.8,
    ),
    (
        GemmKernelBlock(
            num_thread=256,
            tile_m=256,
            tile_n=128,
            tile_k=128,
            num_buffer=4,
            num_squad_m=2,
            num_squad_n=1,
        ),
        0.8,
    ),
]


@functools.cache
def get_gemm_mubin_id_hash(asm_id: GemmMubinId) -> str:
    payload = asdict(asm_id)
    for enum_key in (
        "group_mode",
        "quant_mode_a",
        "quant_mode_b",
        "major_a",
        "major_b",
        "mx_scale",
    ):
        payload[enum_key] = getattr(asm_id, enum_key).value
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class MoeGemmMubinDispatcher:
    def __init__(self, kernel_map_path: Path, arch: Arch):
        if arch not in (MP31_ARCH, MP32_ARCH):
            raise ValueError(f"Unsupported GEMM architecture: {arch}")
        self.kernel_map_path = Path(kernel_map_path)
        self.module_dir = self.kernel_map_path.parent
        self.arch = arch
        self._kernel_hash_map = {
            entry.dispatch_hash: entry
            for entry in load_kernel_map(self.kernel_map_path)
        }

    def estimate_block_score(
        self,
        block: GemmKernelBlock,
        m: int,
        n: int,
        num_expert: int,
        num_mp: int,
        base_score: float,
    ) -> float:
        if num_mp <= 0:
            raise ValueError("num_mp must be greater than 0")
        if base_score == 0 or m == 0 or n == 0 or num_expert == 0:
            return base_score

        average_group_m = m / num_expert
        upper_average_group_m = average_group_m * 1.04
        lower_average_group_m = average_group_m * 0.96
        deviate_group_count = num_expert // 3
        lower_average_group_m_tile = math.ceil(lower_average_group_m / block.tile_m)
        upper_average_group_m_tile = math.ceil(upper_average_group_m / block.tile_m)
        average_group_m_tile = math.ceil(upper_average_group_m / block.tile_m)
        m_tile = (
            lower_average_group_m_tile * deviate_group_count
            + upper_average_group_m_tile * deviate_group_count
            + average_group_m_tile * (num_expert - deviate_group_count * 2)
        )
        n_tile = ceil_div(n, block.tile_n)
        wave = ceil_div(m_tile * n_tile, num_mp)
        wave_ratio = m * n / (block.tile_m * block.tile_n * wave * num_mp)
        return base_score * wave_ratio

    def get_ragged_asm_id(
        self,
        *,
        type_a: torch.dtype,
        type_b: torch.dtype,
        type_d: torch.dtype,
        tme_cache_hint_b: bool,
        alignment_m: int,
        quant_tile: int,
        quant_mode_a: TensorQuantMode,
        quant_mode_b: TensorQuantMode,
        major_a: TensorMajor = TensorMajor.K,
        major_b: TensorMajor = TensorMajor.K,
        major_scale_a: TensorMajor = TensorMajor.K,
    ) -> GemmMubinId:
        try:
            kernel_block = RAGGED_BLOCK_BY_ALIGNMENT_M[alignment_m]
        except KeyError as err:
            raise ValueError("alignment_m must be 128 or 256") from err
        fixed_scale_layout_a = (
            type_a in (torch.float8_e4m3fn, torch.float8_e5m2)
            and major_scale_a == TensorMajor.MN
        )

        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(type_a),
            b_type=get_asm_dtype_from_torch_dtype(type_b),
            d_type=get_asm_dtype_from_torch_dtype(type_d),
            dtype_bias_scale=get_asm_dtype_from_torch_dtype(torch.float32),
            group_mode=MoeGemmMode.RAGGED,
            quant_mode_a=quant_mode_a,
            quant_mode_b=quant_mode_b,
            quant_tile=quant_tile,
            kernel_block=kernel_block,
            major_a=major_a,
            major_b=major_b,
            tme_cache_hint_b=tme_cache_hint_b,
            fixed_scale_layout_a=fixed_scale_layout_a,
            enable_tme_split=True,
            enable_persistence=True,
        )

    def get_w4a8_ragged_asm_id(
        self,
        *,
        type_a: torch.dtype,
        type_d: torch.dtype,
        type_scale_b: torch.dtype,
        tme_cache_hint_b: bool,
        alignment_m: int,
        quant_tile: int,
        quant_mode_a: TensorQuantMode,
        quant_mode_b: TensorQuantMode,
        mx_scale: MxScaleMode,
    ) -> GemmMubinId:
        try:
            kernel_block = PACKED_4BIT_BLOCK_BY_SELECT_M[alignment_m]
        except KeyError as err:
            raise ValueError("alignment_m must be 128 or 256") from err
        is_fp4 = mx_scale == MxScaleMode.FP4_E8M0
        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(type_a),
            b_type=(AsmDType.FP4_E2M1.value if is_fp4 else AsmDType.INT4.value),
            d_type=get_asm_dtype_from_torch_dtype(type_d),
            dtype_bias_scale=(
                AsmDType.FP8_E8M0.value
                if is_fp4
                else get_asm_dtype_from_torch_dtype(type_scale_b)
            ),
            group_mode=MoeGemmMode.RAGGED,
            quant_mode_a=quant_mode_a,
            quant_mode_b=quant_mode_b,
            quant_tile=quant_tile,
            kernel_block=kernel_block,
            major_a=TensorMajor.K,
            major_b=TensorMajor.K,
            tme_cache_hint_b=tme_cache_hint_b,
            enable_tme_split=True,
            enable_persistence=True,
            b_pack_bits=4,
            mx_scale=mx_scale,
            n_split=alignment_m == 256,
        )

    def get_masked_asm_id(
        self,
        *,
        type_a: torch.dtype,
        type_b: torch.dtype,
        type_d: torch.dtype,
        alignment_m: int,
        tme_cache_hint_b: bool,
        quant_tile: int,
        quant_mode_a: TensorQuantMode,
        quant_mode_b: TensorQuantMode,
        major_scale_a: TensorMajor = TensorMajor.K,
    ) -> GemmMubinId:
        try:
            kernel_block = MASKED_BLOCK_BY_SELECT_M[alignment_m]
        except KeyError as err:
            raise ValueError("masked alignment_m must be 128 or 256") from err

        fixed_scale_layout_a = (
            type_a in (torch.float8_e4m3fn, torch.float8_e5m2)
            and major_scale_a == TensorMajor.MN
        )

        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(type_a),
            b_type=get_asm_dtype_from_torch_dtype(type_b),
            d_type=get_asm_dtype_from_torch_dtype(type_d),
            dtype_bias_scale=get_asm_dtype_from_torch_dtype(torch.float32),
            group_mode=MoeGemmMode.MASKED,
            quant_mode_a=quant_mode_a,
            quant_mode_b=quant_mode_b,
            quant_tile=quant_tile,
            kernel_block=kernel_block,
            major_a=TensorMajor.K,
            major_b=TensorMajor.K,
            tme_cache_hint_b=tme_cache_hint_b,
            fixed_scale_layout_a=fixed_scale_layout_a,
            enable_tme_split=True,
            enable_persistence=True,
        )

    def get_w4a8_masked_asm_id(
        self,
        *,
        type_a: torch.dtype,
        type_d: torch.dtype,
        type_scale_b: torch.dtype,
        alignment_m: int,
        tme_cache_hint_b: bool,
        quant_tile: int,
        quant_mode_a: TensorQuantMode,
        quant_mode_b: TensorQuantMode,
        mx_scale: MxScaleMode,
    ) -> GemmMubinId:
        try:
            kernel_block = PACKED_4BIT_BLOCK_BY_SELECT_M[alignment_m]
        except KeyError as err:
            raise ValueError("masked alignment_m must be 128 or 256") from err
        is_fp4 = mx_scale == MxScaleMode.FP4_E8M0
        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(type_a),
            b_type=(AsmDType.FP4_E2M1.value if is_fp4 else AsmDType.INT4.value),
            d_type=get_asm_dtype_from_torch_dtype(type_d),
            dtype_bias_scale=(
                AsmDType.FP8_E8M0.value
                if is_fp4
                else get_asm_dtype_from_torch_dtype(type_scale_b)
            ),
            group_mode=MoeGemmMode.MASKED,
            quant_mode_a=quant_mode_a,
            quant_mode_b=quant_mode_b,
            quant_tile=quant_tile,
            kernel_block=kernel_block,
            major_a=TensorMajor.K,
            major_b=TensorMajor.K,
            tme_cache_hint_b=tme_cache_hint_b,
            enable_tme_split=True,
            enable_persistence=True,
            b_pack_bits=4,
            mx_scale=mx_scale,
            n_split=alignment_m == 256,
        )

    def get_k_contig_asm_id(
        self,
        *,
        type_a: torch.dtype,
        type_b: torch.dtype,
        type_d: torch.dtype,
        quant_tile: int,
        quant_mode_a: TensorQuantMode,
        quant_mode_b: TensorQuantMode,
        major_scale_a: TensorMajor,
    ) -> GemmMubinId:
        fixed_scale_layout_a = (
            type_a in (torch.float8_e4m3fn, torch.float8_e5m2)
            and major_scale_a == TensorMajor.MN
        )

        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(type_a),
            b_type=get_asm_dtype_from_torch_dtype(type_b),
            d_type=get_asm_dtype_from_torch_dtype(type_d),
            dtype_bias_scale=get_asm_dtype_from_torch_dtype(torch.float32),
            group_mode=MoeGemmMode.K_CONTIG,
            quant_mode_a=quant_mode_a,
            quant_mode_b=quant_mode_b,
            quant_tile=quant_tile,
            kernel_block=GemmKernelBlock(),
            major_a=TensorMajor.MN,
            major_b=TensorMajor.MN,
            fixed_scale_layout_a=fixed_scale_layout_a,
            enable_tme_split=True,
            enable_persistence=True,
        )

    def get_mp32_ragged_asm_id(self, **kwargs) -> GemmMubinId:
        block = _get_mp32_ragged_block(kwargs.pop("alignment_m"))
        return self._get_mp32_fp8_asm_id(
            MoeGemmMode.RAGGED, kernel_block=block, **kwargs
        )

    def get_mp32_masked_asm_id(self, **kwargs) -> GemmMubinId:
        return self._get_mp32_fp8_asm_id(MoeGemmMode.MASKED, **kwargs)

    def get_mp32_k_contig_asm_id(self, **kwargs) -> GemmMubinId:
        return self._get_mp32_fp8_asm_id(MoeGemmMode.K_CONTIG, **kwargs)

    def _get_mp32_fp8_asm_id(
        self,
        group_mode: MoeGemmMode,
        *,
        kernel_block: GemmKernelBlock = MP32_GROUPED_BLOCK,
        **kwargs,
    ) -> GemmMubinId:
        fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        if kwargs["type_a"] not in fp8_dtypes or kwargs["type_b"] not in fp8_dtypes:
            raise ValueError("MP32 grouped GEMM requires E4M3 or E5M2 A and B")
        output_types = (
            (torch.bfloat16, torch.float32)
            if group_mode == MoeGemmMode.K_CONTIG
            else (torch.bfloat16,)
        )
        if kwargs["type_d"] not in output_types:
            raise ValueError(
                "MP32 grouped GEMM output dtype is unsupported for this mode"
            )
        if kwargs["quant_tile"] != 32:
            raise ValueError("MP32 grouped GEMM requires quant_tile=32")
        major = TensorMajor.MN if group_mode == MoeGemmMode.K_CONTIG else TensorMajor.K
        return GemmMubinId(
            arch=MP32_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(kwargs["type_a"]),
            b_type=get_asm_dtype_from_torch_dtype(kwargs["type_b"]),
            d_type=get_asm_dtype_from_torch_dtype(kwargs["type_d"]),
            dtype_bias_scale=AsmDType.FP8_E8M0.value,
            group_mode=group_mode,
            quant_mode_a=TensorQuantMode.GROUP,
            quant_mode_b=TensorQuantMode.GROUP,
            quant_tile=32,
            kernel_block=kernel_block,
            major_a=major,
            major_b=major,
            fixed_scale_layout_a=True,
            fixed_scale_layout_b=True,
            enable_tme_split=True,
            enable_persistence=True,
            mx_scale=MxScaleMode.E8M0,
        )

    def _get_mp32_mxfp4_asm_id(
        self,
        group_mode: MoeGemmMode,
        kernel_block: GemmKernelBlock = MP32_GROUPED_BLOCK,
    ) -> GemmMubinId:
        if group_mode not in (MoeGemmMode.RAGGED, MoeGemmMode.MASKED):
            raise ValueError(f"Unsupported MP32 MXFP4 group mode: {group_mode}")
        return GemmMubinId(
            arch=MP32_ARCH,
            a_type=AsmDType.FP4_E2M1.value,
            b_type=AsmDType.FP4_E2M1.value,
            d_type=get_asm_dtype_from_torch_dtype(torch.bfloat16),
            dtype_bias_scale=AsmDType.FP8_E8M0.value,
            group_mode=group_mode,
            quant_mode_a=TensorQuantMode.GROUP,
            quant_mode_b=TensorQuantMode.GROUP,
            quant_tile=32,
            kernel_block=kernel_block,
            major_a=TensorMajor.K,
            major_b=TensorMajor.K,
            fixed_scale_layout_a=True,
            fixed_scale_layout_b=True,
            enable_tme_split=True,
            enable_persistence=True,
            b_pack_bits=4,
            mx_scale=MxScaleMode.E8M0,
        )

    def get_mp32_mxfp4_ragged_asm_id(self, alignment_m: int = 128) -> GemmMubinId:
        return self._get_mp32_mxfp4_asm_id(
            MoeGemmMode.RAGGED, _get_mp32_ragged_block(alignment_m)
        )

    def get_mp32_mxfp4_masked_asm_id(self) -> GemmMubinId:
        return self._get_mp32_mxfp4_asm_id(MoeGemmMode.MASKED)

    def get_ragged_expert_layout_asm_id(
        self,
        *,
        type_a: torch.dtype,
        type_b: torch.dtype,
        type_d: torch.dtype,
        candidate_block_id: int,
        quant_tile: int,
        quant_mode_a: TensorQuantMode,
        quant_mode_b: TensorQuantMode,
        major_b: TensorMajor,
    ) -> GemmMubinId:
        kernel_block = RAGGED_EXPERT_LAYOUT_BLOCK_CANDIDATES[candidate_block_id][0]

        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(type_a),
            b_type=get_asm_dtype_from_torch_dtype(type_b),
            d_type=get_asm_dtype_from_torch_dtype(type_d),
            dtype_bias_scale=get_asm_dtype_from_torch_dtype(torch.float32),
            group_mode=MoeGemmMode.RAGGED_EXPERT_LAYOUT,
            quant_mode_a=quant_mode_a,
            quant_mode_b=quant_mode_b,
            quant_tile=quant_tile,
            kernel_block=kernel_block,
            major_a=TensorMajor.K,
            major_b=major_b,
            enable_tme_split=True,
            enable_persistence=True,
        )

    def get_normal_gemm_id(
        self,
        *,
        type_a: torch.dtype,
        type_b: torch.dtype,
        type_d: torch.dtype,
        quant_tile: int,
        major_a: TensorMajor,
        major_b: TensorMajor,
    ) -> GemmMubinId:
        kernel_block = GemmKernelBlock()
        return GemmMubinId(
            arch=MP31_ARCH,
            a_type=get_asm_dtype_from_torch_dtype(type_a),
            b_type=get_asm_dtype_from_torch_dtype(type_b),
            d_type=get_asm_dtype_from_torch_dtype(type_d),
            dtype_bias_scale=get_asm_dtype_from_torch_dtype(torch.float32),
            group_mode=MoeGemmMode.NO_GROUP,
            quant_mode_a=TensorQuantMode.GROUP,
            quant_mode_b=TensorQuantMode.BLOCK,
            quant_tile=quant_tile,
            kernel_block=kernel_block,
            major_a=major_a,
            major_b=major_b,
            fixed_scale_layout_a=major_a == TensorMajor.MN,
            fixed_scale_layout_b=False,
            enable_tme_split=True,
            enable_persistence=True,
        )

    @functools.cache  # noqa: B019 - dispatcher instances are process-global caches
    def resolve_kernel_entry(self, asm_id: GemmMubinId):
        if asm_id.arch != self.arch:
            raise ValueError("GEMM dispatch architecture does not match kernel map")
        asm_id_hash = get_gemm_mubin_id_hash(asm_id)
        entry = self._kernel_hash_map.get(asm_id_hash)
        if entry is None:
            raise ValueError(f"No MoE GEMM mubin kernel found for hash {asm_id_hash}")
        return entry

    @functools.cache  # noqa: B019 - dispatcher instances are process-global caches
    def resolve_kernel_artifact(self, asm_id: GemmMubinId):
        entry = self.resolve_kernel_entry(asm_id)
        path = ensure_mubin_kernel_artifact(
            "gemm",
            self.module_dir,
            entry,
            artifact_path=ArtifactPath.for_module("gemm", self.arch),
            kernel_map_sha256=KernelMapHash.for_module("gemm", self.arch),
            arch=self.arch,
        )
        return entry, path


@functools.cache
def get_moe_gemm_mubin_dispatcher(kernel_map_path: Path, arch: Arch):
    return MoeGemmMubinDispatcher(kernel_map_path, arch)
