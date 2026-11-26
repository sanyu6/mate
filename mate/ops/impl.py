"""Interface for a complete implementation, potentially spanning targets."""

from abc import ABC, abstractmethod
from typing import Any


class Impl(ABC):
    """A callable implementation that owns validation, building and execution.

    Each operator defines a mutable input object for one call. Implementations
    receive that object throughout preparation and execution; reusable instances
    keep build resources and caches, rather than per-call inputs.

    One implementation may share code across architectures and orchestrate
    multiple kernels. Its constructor, build resources and cache lifetime are
    defined by the concrete implementation.
    """

    @abstractmethod
    def sanitize(self, inputs: Any) -> None:
        """Validate and normalize this call's inputs in place.

        May replace tensor fields with prepared tensors, allocating memory and
        copying data when permitted by the Op and this implementation.
        Subsequent building and execution consume the updated fields. Return ``None`` on success; raise ``TypeError`` or
        ``ValueError`` for invalid or unsupported inputs.
        """
        raise NotImplementedError

    @abstractmethod
    def build(self, inputs: Any = None, *, aot: bool = False):
        """Prepare implementation artifacts without launching operation kernels.

        Parameters:
            inputs (Any, optional):
                Per-call inputs used to prepare the required specialization.
                Existing AOT artifacts or cached builds may be reused.
            aot (bool, optional):
                Prepare predefined configurations where this Impl supports AOT.
                Configuration enumeration belongs to this implementation;
                report unsupported modes instead of silently ignoring them.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, inputs: Any) -> Any:
        """Execute the implementation and return the operator's results.

        The concrete implementation arranges validation, building and kernel
        launches as needed. Callers need not call ``sanitize`` or ``build``
        separately before invoking the instance.
        """
        raise NotImplementedError
