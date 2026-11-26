from importlib import import_module

from mate.ops import Impl
from .._input import _Fp4DispatchInput
from ..buffer import Fp4DispatchBuffer


class Mp32DispatchImpl(Impl):
    def __init__(self, pull=False):
        self.pull = pull

    def sanitize(self, inputs: _Fp4DispatchInput) -> None:
        if not isinstance(inputs, _Fp4DispatchInput):
            raise TypeError("requires _Fp4DispatchInput")
        if not isinstance(inputs.buffer, Fp4DispatchBuffer) or inputs.buffer.closed:
            raise ValueError("requires a live Fp4DispatchBuffer")
        if (
            isinstance(inputs.tokens, bool)
            or not isinstance(inputs.tokens, int)
            or not 1 <= inputs.tokens <= inputs.buffer.max_tokens
        ):
            raise ValueError("tokens must be an integer within dispatch capacity")
        if inputs.num_ctas is not None:
            if (
                isinstance(inputs.num_ctas, bool)
                or not isinstance(inputs.num_ctas, int)
                or not 1 <= inputs.num_ctas <= inputs.buffer._sym.common_mps
            ):
                raise ValueError("num_ctas must be within the common MP count")

    def _settings(self, inputs):
        b = inputs.buffer
        if not self.pull and inputs.num_ctas is not None:
            raise ValueError("num_ctas is supported by the pull backend only")
        # The b24 choice is measured. Larger batches retain the full common grid.
        default = (
            4
            if inputs.tokens <= 4
            else 24
            if inputs.tokens <= 32
            else b._sym.common_mps
        )
        ctas = (
            inputs.num_ctas
            if inputs.num_ctas is not None
            else min(default, b._sym.common_mps)
        )
        return b._sym.config, b._sym.common_mps, b.line_padded_scales, ctas

    def build(self, inputs=None, *, aot=False):
        if aot:
            raise ValueError("dispatch-only has no predefined AOT configurations")
        self.sanitize(inputs)
        config, mps, padded, ctas = self._settings(inputs)
        jit = import_module("mate.jit.fp4_dispatch_only")
        return jit.get_fp4_dispatch_module(
            config, mps, padded, pull=self.pull, num_ctas=ctas if self.pull else None
        )

    def __call__(self, inputs: _Fp4DispatchInput):
        self.sanitize(inputs)
        config, mps, padded, ctas = self._settings(inputs)
        jit = import_module("mate.jit.fp4_dispatch_only")
        launcher = jit.get_fp4_dispatch_launcher(
            config, mps, padded, pull=self.pull, num_ctas=ctas if self.pull else None
        )
        b = inputs.buffer
        s = b._sym
        launcher(s.buffer, s.ptrs, *b.result.tensors, s.rank, inputs.tokens)
        return b.result
