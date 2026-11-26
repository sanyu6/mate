from mate.arch import MP32_ARCH
from mate.ops import Op

from ._input import _Fp4DispatchInput
from .buffer import Fp4DispatchBuffer, Fp4DispatchResult
from .impls.mp32 import Mp32DispatchImpl


class Fp4Dispatch(Op):
    """Dispatch each FP4 route from symmetric input into its owner's expert pool.

    No GEMM, activation, route deduplication, output combine or hidden input
    quantization is performed. A complete invocation includes publication,
    peer pull/pack and workspace retirement. Does not retain per-call inputs.
    """

    @classmethod
    def resolve_backend(cls, target, *args, override_backend=None, **kwargs):
        if target != MP32_ARCH or override_backend not in (None, "mp32", "mp32_pull"):
            raise ValueError("Fp4Dispatch supports the mp32 backend only")
        return Mp32DispatchImpl(pull=override_backend in (None, "mp32_pull"))

    @classmethod
    def run(
        cls,
        buffer: Fp4DispatchBuffer,
        tokens: int,
        *,
        override_backend: str | None = None,
        num_ctas: int | None = None,
    ) -> Fp4DispatchResult:
        """Dispatch already prepared FP4 input; return overwritten output views.

        Parameters:
            buffer: Dedicated collectively allocated IPC and output storage.
            tokens: Same active input rows on every rank, within 1..max_tokens.
            override_backend: None/'mp32_pull' selects optimized peer-pull; 'mp32' selects the original protocol.
            num_ctas: Optional common resident CTA count for the pull backend.
        """
        inputs = _Fp4DispatchInput(buffer, tokens, num_ctas)
        # Validate before querying a potentially closed buffer's device.
        Mp32DispatchImpl().sanitize(inputs)
        target = cls.resolve_target(device=buffer._sym.buffer.device)
        return cls.resolve_backend(target, override_backend=override_backend)(inputs)

    @classmethod
    def build(cls, inputs=None, *, aot=False):
        """Compile an explicit _Fp4DispatchInput without dispatching data."""
        if aot:
            raise ValueError("dispatch-only has no predefined AOT configurations")
        impl = Mp32DispatchImpl()
        impl.sanitize(inputs)
        target = cls.resolve_target(device=inputs.buffer._sym.buffer.device)
        return cls.resolve_backend(target).build(inputs)
