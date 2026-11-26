"""Independent FP4 symmetric IPC storage and compact prefill pools."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from mate.jit.mega_moe import get_mega_moe_runtime_utils_module
from mate.jit.mega_moe_fp4_fp4_decode import get_fp4_decode_input_module
from mate.jit.mega_moe_fp4_fp4 import FP4PrefillConfig, get_fp4_layout_module
from mate.jit.runtime import ffi_to_torch


@dataclass(frozen=True)
class FP4SymmetricLayout:
    tokens: int
    hidden: int
    topk: int

    @property
    def offsets(self):
        def align(value):
            return (value + 255) // 256 * 256

        x = 512
        xs = align(x + self.tokens * (self.hidden // 2))
        routes = align(xs + self.tokens * (self.hidden // 32))
        weights = align(routes + self.tokens * self.topk * 8)
        combine = align(weights + self.tokens * self.topk * 4)
        return {
            "x": x,
            "x_sf": xs,
            "topk_idx": routes,
            "topk_weights": weights,
            "combine": combine,
            "bytes": align(combine + self.tokens * self.topk * self.hidden * 2),
        }


class FP4FP4SymmBuffer:
    """Collectively owned IPC buffer; destroy collectively after all launches.

    Packed inputs, dispatch control/source metadata, routes and BF16 combine
    slots are shared. Compact compute pools are local and allocated lazily. Route IDs must be unique per
    token; -1 disables a route. Every rank supplies the same active token count.
    dedicated_ipc=True owns a separate musaMalloc allocation for IPC handles.
    """

    def __init__(
        self,
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        *,
        enable_decode=False,
        dedicated_ipc=False,
    ):
        if not isinstance(enable_decode, bool):
            raise TypeError("enable_decode must be a bool")
        if not isinstance(dedicated_ipc, bool):
            raise TypeError("dedicated_ipc must be a bool")
        self.group = group
        self.config = FP4PrefillConfig(
            group.size(),
            (num_max_tokens_per_rank + 255) // 256 * 256,
            hidden,
            intermediate_hidden,
            num_experts,
            num_topk,
        )
        c = self.config
        props = torch.musa.get_device_properties(torch.musa.current_device())
        if (props.major, props.minor) != (3, 2):
            raise ValueError("FP4 Mega MoE requires MP32")
        settings = [None] * group.size()
        dist.all_gather_object(
            settings,
            (c, props.multi_processor_count, enable_decode, dedicated_ipc),
            group=group,
        )
        if any(
            setting[0] != c or setting[2:] != (enable_decode, dedicated_ipc)
            for setting in settings
        ):
            raise ValueError("FP4 symmetric configurations must match across ranks")
        # Four owner CTAs cooperate per compact-pool tile, as in W4A8.
        self.common_mps = min(setting[1] for setting in settings) // 4 * 4
        self.rank = group.rank()
        self.layout = FP4SymmetricLayout(c.tokens, c.hidden, c.topk)
        offsets = self.layout.offsets
        prefix, module = get_fp4_layout_module(c)
        if module.get_function(prefix + "_bytes")() != offsets["bytes"]:
            raise RuntimeError("Python/C++ FP4 IPC layout mismatch")
        self._layout_module = (prefix, module)
        self.dispatch_workspace_bytes = module.get_function(
            prefix + "_workspace_bytes"
        )()
        self.dispatch_control_bytes = module.get_function(prefix + "_control_bytes")()
        self.dispatch_barrier_bytes = module.get_function(prefix + "_barrier_bytes")()
        self.decode_enabled = bool(enable_decode)
        allocation_bytes = offsets["bytes"] + self.dispatch_workspace_bytes
        if self.decode_enabled:
            from mate.jit.mega_moe_fp4_fp4_decode import get_fp4_decode_module

            name, decode_module = get_fp4_decode_module(c, 0)
            allocation_bytes = decode_module.get_function(name)()
        self._utils = get_mega_moe_runtime_utils_module()
        if dedicated_ipc:
            # IPC handles describe an allocation base, not a caching-allocator
            # suballocation. Retain the owning FFI allocation through DLPack.
            self.buffer = ffi_to_torch(
                self._utils.get_function("mega_moe_alloc_buffer")(
                    allocation_bytes, torch.musa.current_device()
                )
            ).view(torch.uint8)
            self.buffer.zero_()
        else:
            self.buffer = torch.zeros(
                allocation_bytes,
                dtype=torch.uint8,
                device="musa",
            )

        def view(name, dtype, shape):
            size = 1
            for dim in shape:
                size *= dim
            itemsize = torch.empty((), dtype=dtype).element_size()
            return (
                self.buffer[offsets[name] : offsets[name] + size * itemsize]
                .view(dtype)
                .view(shape)
            )

        self.x = view("x", torch.uint8, (c.tokens, c.hidden // 2))
        self.x_sf = view("x_sf", torch.uint8, (c.tokens, c.hidden // 32))
        self.topk_idx = view("topk_idx", torch.int64, (c.tokens, c.topk))
        self.topk_weights = view("topk_weights", torch.float32, (c.tokens, c.topk))
        self.combine = view("combine", torch.bfloat16, (c.tokens, c.topk, c.hidden))
        if group.size() == 1:
            self.ptrs = torch.tensor(
                [self.buffer.data_ptr()], dtype=torch.int64, device="cpu"
            )
        else:
            size = int(self._utils.get_function("mega_moe_get_ipc_handle_size")())
            handle = torch.empty(size, dtype=torch.uint8, device="cpu")
            self._utils.get_function("mega_moe_get_ipc_handle")(self.buffer, handle)
            handles = [None] * group.size()
            dist.all_gather_object(handles, bytes(handle.tolist()), group=group)
            packed = torch.tensor(
                [list(value) for value in handles], dtype=torch.uint8, device="cpu"
            )
            self.ptrs = torch.empty(group.size(), dtype=torch.int64, device="cpu")
            self._utils.get_function("mega_moe_open_ipc_handles")(
                packed, self.buffer, self.rank, self.ptrs
            )
        self._workspace_ready = False
        torch.musa.synchronize()
        dist.barrier(group=group)

    def allocate_workspace(self):
        if self._workspace_ready:
            return
        c = self.config
        g, cap = c.experts // c.ranks, c.ranks * c.tokens

        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=self.buffer.device)

        self.counts = empty((g,), torch.int32)
        self.metadata = empty((g, cap), torch.int64)
        self.status = empty((1,), torch.int32)
        # Same compact pool capacity formula and M256 padding as W4A8.
        self.stage1_pool_tokens = (cap * min(c.topk, g) + g * 255 + 255) // 256 * 256
        self.stage1_l1 = empty((self.stage1_pool_tokens, c.hidden // 2), torch.uint8)
        self.stage1_l1_sf = empty(
            (c.hidden // 32, self.stage1_pool_tokens), torch.uint8
        )
        self.stage2_l2 = empty(
            (self.stage1_pool_tokens, c.intermediate // 2), torch.uint8
        )
        self.stage2_l2_sf = empty(
            (c.intermediate // 32, self.stage1_pool_tokens), torch.uint8
        )
        self._workspace_ready = True

    def prepare_decode_input(self, hidden, ids, weights):
        """Write BF16 hidden and current routing into the decode IPC prefix.

        Quantization matches the packed W4A4 E2M1/UE8M0 contract. Scale
        layout conversion, int32-to-int64 IDs and route copies are fused.
        Warm up once before capture. Returns no newly allocated tensors.
        """
        if not self.decode_enabled or self.group is None:
            raise ValueError("prepare_decode_input requires a live decode buffer")
        if (
            getattr(self, "_decode_plan", None) is not None
            or getattr(self, "_stage1_plan_tokens", None) is not None
        ):
            raise ValueError("finish Stage2 before overwriting decode input")
        rows = hidden.shape[0] if hidden.ndim == 2 else 0
        if not 0 < rows <= min(128, self.config.tokens):
            raise ValueError("decode input supports 1..128 local rows")
        get_fp4_decode_input_module().fp4_decode_input(
            hidden,
            ids,
            weights,
            self.x[:rows],
            self.x_sf[:rows],
            self.topk_idx[:rows],
            self.topk_weights[:rows],
        )

    def fused_stage1_weights(self, w1):
        """Cache gate/up interleave and stride-8 TME scales, tracking mutations."""
        key = tuple((id(t), t._version) for t in w1)
        if getattr(self, "_fused_weight_key", None) != key:
            c = self.config
            g = c.experts // c.ranks
            scale_block = 32
            online_stride = 8
            groups = c.intermediate // scale_block
            packed = w1[0].view(g, 2, groups, scale_block, c.hidden // 2)
            # Interleave gate/up groups and apply TME's N permutation within
            # each scale block in one copy. FP4 weight bytes stay unswizzled.
            scales = w1[1].view(
                g,
                c.hidden // scale_block,
                2,
                groups,
                scale_block // online_stride,
                online_stride,
            )
            self._fused_weights = (
                packed.permute(0, 2, 1, 3, 4).contiguous().view_as(w1[0]),
                scales.permute(0, 1, 3, 2, 5, 4).contiguous().view_as(w1[1]),
            )
            self._fused_weight_source = w1
            self._fused_weight_key = key
        return self._fused_weights

    def swap_decode_weights(self, weights, stage, feature_tile=256):
        """Decode-only K-tile-contiguous weight A and plain MN-major scales.

        Accept the existing public prepared pairs, retaining their source and
        tracking in-place updates. This does not change prefill preparation.
        """
        if stage not in (1, 2):
            raise ValueError("decode weight stage must be 1 or 2")
        key = (feature_tile, tuple((id(t), t._version) for t in weights))
        cache_id = (stage, feature_tile, tuple(id(t) for t in weights))
        cache = getattr(self, "_swap_weight_cache", {})
        if cache_id not in cache or cache[cache_id][0] != key:
            packed, scales = weights
            g, n, half_k = packed.shape
            if feature_tile not in (64, 128, 256) or n % feature_tile or half_k % 128:
                raise ValueError("swap-AB weights need tile feature and K256 alignment")
            if stage == 1:
                # Input W1 is still gate half followed by up half.
                packed = (
                    packed.view(g, 2, n // 64, 32, half_k)
                    .permute(0, 2, 1, 3, 4)
                    .contiguous()
                    .view_as(packed)
                )
                scales = (
                    scales.view(g, half_k // 16, 2, n // 64, 32)
                    .permute(0, 1, 3, 2, 4)
                    .contiguous()
                    .view_as(scales)
                )
            else:
                # Undo the opaque Stage2 online-N scale permutation.
                scales = (
                    scales.view(g, half_k // 16, n // 32, 8, 4)
                    .permute(0, 1, 2, 4, 3)
                    .contiguous()
                    .view_as(scales)
                )
            if stage == 2:
                # Give each native accumulator lane 2/4 adjacent features.
                packed = (
                    packed.view(g, n // 128, 32, 4, half_k)
                    .permute(0, 1, 3, 2, 4)
                    .contiguous()
                    .view_as(packed)
                )
                scales = (
                    scales.view(g, half_k // 16, n // 128, 32, 4)
                    .permute(0, 1, 2, 4, 3)
                    .contiguous()
                    .view_as(scales)
                )
            packed = (
                packed.reshape(g, n // feature_tile, feature_tile, half_k // 128, 128)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
                .view_as(packed)
            )
            result = (packed, scales)
            cache[cache_id] = (key, weights, result)
            self._swap_weight_cache = cache
        return cache[cache_id][2]

    def packed_decode_weights(self, weights, stage):
        """Cache K256-contiguous weight B without changing the online swizzle."""
        if stage not in (1, 2):
            raise ValueError("decode weight stage must be 1 or 2")
        key = tuple((id(t), t._version) for t in weights)
        cache_id = (stage, tuple(id(t) for t in weights))
        cache = getattr(self, "_packed_decode_cache", {})
        if cache_id not in cache or cache[cache_id][0] != key:
            prepared = self.fused_stage1_weights(weights) if stage == 1 else weights
            packed, scales = prepared
            g, n, half_k = packed.shape
            packed = (
                packed.view(g, n // 256, 256, half_k // 128, 128)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
                .view_as(packed)
            )
            cache[cache_id] = (key, weights, (packed, scales))
            self._packed_decode_cache = cache
        return cache[cache_id][2]

    def ipc_probe(self):
        prefix, module = self._layout_module
        result = torch.empty(
            self.config.ranks, dtype=torch.uint8, device=self.buffer.device
        )
        module.get_function(prefix + "_probe")(
            self.buffer, self.ptrs, result, self.rank
        )
        return result

    def destroy(self):
        if self.group is None:
            return
        torch.musa.synchronize()
        dist.barrier(group=self.group)
        if self.group.size() > 1:
            self._utils.get_function("mega_moe_close_ipc_handles")(self.ptrs, self.rank)
        for key, value in tuple(self.__dict__.items()):
            if isinstance(value, torch.Tensor):
                setattr(self, key, None)
        self.group = None
        self._fused_weights = None
        self._fused_weight_source = None
        self._swap_weight_cache = {}
        self._packed_decode_cache = {}


def get_symm_buffer_for_fp4_fp4_mega_moe(
    group,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    *,
    enable_decode=False,
):
    return FP4FP4SymmBuffer(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        enable_decode=enable_decode,
    )
