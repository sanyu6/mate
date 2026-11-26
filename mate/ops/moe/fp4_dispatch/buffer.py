"""Dedicated IPC state and caller-visible dispatch outputs."""

from dataclasses import dataclass
from importlib import import_module

import torch


@dataclass(frozen=True)
class Fp4DispatchResult:
    """Expert-major M32 pools, valid rows described by counts and offsets.

    payload: uint8 [P,H/2], packed E2M1, K contiguous.
    scales: uint8 [H/32,P] or [H/32,4P], MN-major UE8M0. In the
        line-padded variant, logical row r is stored at (r//32)*128+r%32.
    metadata: int32 [P,4], {source_rank, token, topk_slot, float_weight_bits}.
    counts: int32 [local_experts], number of valid rows per expert.
    offsets: int32 [local_experts+1], exclusive prefix of M32-padded counts.
    status: int32 [1], 0 success, 1 invalid expert ID (except disabled -1),
        2 peer-handshake timeout (discard the invalid output/buffer).

    No ordering guarantee within an expert/source-rank group. Padding rows
    are undefined. Views are overwritten on the next dispatch on this buffer.
    """

    payload: torch.Tensor
    scales: torch.Tensor
    metadata: torch.Tensor
    counts: torch.Tensor
    offsets: torch.Tensor
    status: torch.Tensor
    line_padded_scales: bool

    def clone(self):
        """Snapshot all output storage on the current stream, also in graphs."""
        return Fp4DispatchResult(
            *(t.clone() for t in self.tensors), self.line_padded_scales
        )

    @property
    def tensors(self):
        return (
            self.payload,
            self.scales,
            self.metadata,
            self.counts,
            self.offsets,
            self.status,
        )


class Fp4DispatchBuffer:
    """Collectively allocate a dedicated MegaMoE symmetric IPC workspace.

    Every rank must use the same configuration and active token count, and
    invoke dispatch in the same order. IDs are unique per token; -1 disables
    a route. There is no token deduplication. Use one ordered stream per buffer;
    overlapping invocations or mixing this state with a fused MoE is unsupported.
    Construct and warm up before graph capture. close() collectively after
    all graph instances using this buffer have been reset.
    Set dedicated_ipc=True when embedding in a model to use independent IPC
    allocations rather than possible PyTorch caching-allocator suballocations.
    """

    def __init__(
        self,
        group,
        num_experts: int,
        hidden: int,
        topk: int,
        *,
        max_tokens: int = 128,
        line_padded_scales: bool = True,
        dedicated_ipc: bool = False,
    ):
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 1 <= max_tokens <= 128
        ):
            raise ValueError("max_tokens must be an integer in 1..128")
        if not isinstance(line_padded_scales, bool):
            raise TypeError("line_padded_scales must be bool")
        # I is an unused layout parameter: no GEMM or expert weights are allocated.
        factory = import_module("mate._fp4_moe_buffer").FP4FP4SymmBuffer
        self._sym = factory(
            group,
            num_experts,
            max_tokens,
            topk,
            hidden,
            256,
            enable_decode=True,
            dedicated_ipc=dedicated_ipc,
        )
        self.max_tokens = max_tokens
        self.line_padded_scales = line_padded_scales
        self.closed = False
        c = self._sym.config
        g, cap = c.experts // c.ranks, c.ranks * c.tokens
        self.pool_tokens = (cap * min(c.topk, g) + g * 255 + 255) // 256 * 256
        p = self.pool_tokens
        device = self._sym.buffer.device
        self.result = Fp4DispatchResult(
            torch.empty((p, hidden // 2), dtype=torch.uint8, device=device),
            torch.empty(
                (hidden // 32, p * (4 if line_padded_scales else 1)),
                dtype=torch.uint8,
                device=device,
            ),
            torch.empty((p, 4), dtype=torch.int32, device=device),
            torch.empty(g, dtype=torch.int32, device=device),
            torch.empty(g + 1, dtype=torch.int32, device=device),
            torch.empty(1, dtype=torch.int32, device=device),
            line_padded_scales,
        )

    def prepare_input(
        self, hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor
    ):
        """Separately fuse BF16->FP4, scales and route copies into the IPC prefix.

        This preparation is not part of the dispatch-only Op. Prequantized
        callers may use prepare_fp4() instead.
        """
        if self.closed:
            raise ValueError("dispatch buffer is closed")
        if hidden.ndim != 2 or not 1 <= hidden.shape[0] <= self.max_tokens:
            raise ValueError("active rows exceed dispatch capacity")
        self._sym.prepare_decode_input(hidden, ids, weights)

    @property
    def device(self):
        return self._sym.buffer.device

    def prepare_fp4(
        self,
        payload: torch.Tensor,
        scales: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
    ):
        """Copy already quantized input into the IPC prefix, outside dispatch.

        payload is uint8 [tokens,H/2], scales is row-major uint8 [tokens,H/32],
        ids is int32/int64 [tokens,topk], weights is float32 [tokens,topk].
        All inputs must be contiguous on this buffer's device. The copies are
        issued on the current stream and may themselves be captured.
        """
        if self.closed:
            raise ValueError("dispatch buffer is closed")
        c = self._sym.config
        rows = payload.shape[0] if payload.ndim == 2 else 0
        if not 1 <= rows <= self.max_tokens:
            raise ValueError("active rows exceed dispatch capacity")
        tensors = (payload, scales, ids, weights)
        shapes = (
            (rows, c.hidden // 2),
            (rows, c.hidden // 32),
            (rows, c.topk),
            (rows, c.topk),
        )
        dtypes = (
            (torch.uint8,),
            (torch.uint8,),
            (torch.int32, torch.int64),
            (torch.float32,),
        )
        for t, shape, dtype in zip(tensors, shapes, dtypes):
            if (
                t.shape != shape
                or t.dtype not in dtype
                or not t.is_contiguous()
                or t.device != self.device
            ):
                raise ValueError("invalid prequantized input shape/dtype/device/stride")
        for dst, src in zip(
            (self._sym.x, self._sym.x_sf, self._sym.topk_idx, self._sym.topk_weights),
            tensors,
        ):
            dst[:rows].copy_(src)

    def close(self):
        if not self.closed:
            self._sym.destroy()
            self.closed = True
