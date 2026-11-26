"""Base operator interface and target architecture lookup."""

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Iterable, Mapping, cast

import torch

from ..mate_runtime import get_arch
from ..arch import Arch
from .impl import Impl


class _OpMeta(ABCMeta):
    """Execute an operator class without constructing an instance."""

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls.__abstractmethods__:
            methods = ", ".join(sorted(cls.__abstractmethods__))
            raise TypeError(
                f"Cannot call abstract operator {cls.__name__} "
                f"with abstract methods: {methods}"
            )
        return cast("type[Op]", cls).run(*args, **kwargs)


class Op(ABC, metaclass=_OpMeta):
    """Base class for selecting an implementation and running an operator.

    Each operator declares its public arguments in ``run``, selects an Impl
    through ``resolve_backend``, and calls that Impl. The operator also decides
    when metadata must be prepared.
    """

    impl_map: Mapping[Arch, Iterable[Impl]]

    @classmethod
    @abstractmethod
    def run(cls, *args: Any, **kwargs: Any) -> Any:
        """Run the operator with its public arguments.

        ``ConcreteOp(...)`` calls this class method without creating an Op
        instance. Create a new internal input object, select an Impl, and
        return its result.
        """
        raise NotImplementedError

    @classmethod
    def resolve_target(
        cls, override_cc: int | None = None, *, device: torch.device | None = None
    ) -> Arch:
        """Resolve an explicit compute capability or query the input/current MUSA device.

        An override such as ``31`` takes precedence over ``device`` and can be
        resolved without querying or initializing a device.
        Concrete ops should pass their input device when querying hardware.
        """
        if override_cc is not None:
            return Arch.from_cc(override_cc)

        return get_arch(device)

    @classmethod
    @abstractmethod
    def resolve_backend(
        cls,
        target: Arch,
        *args: Any,
        override_backend: str | None = None,
        **kwargs: Any,
    ) -> Impl:
        """Select a callable implementation instance for this target and input.

        The concrete op owns automatic selection, backend names, explicit
        overrides and errors for unsupported requests. Selecting an instance
        must not normalize inputs, compile, or launch kernels.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build(cls, inputs: Any = None, *, aot: bool = False):
        """Prepare artifacts without launching operation kernels.

        The ``aot`` parameter preserves existing implementation-specific build
        paths; it does not define a shared AOT API. Concrete operators document
        which build modes they support.

        Parameters:
            inputs (Any):
                Operator-specific inputs used to select an implementation and
                prepare the required specialization. Existing AOT artifacts or
                cached builds may be reused. Required when aot=False.
            aot (bool):
                Prepare predefined configurations where this operator supports
                AOT. Report unsupported modes instead of silently ignoring them.

        Examples:
            An operator with existing AOT support may build as shown below.
            This example assumes inputs contain a query tensor q and selected
            Impls support ``build(aot=True)``. It does not require every Op to
            use ``impl_map`` or this enumeration strategy.

            >>> @classmethod
            ... def build(cls, inputs=None, *, aot=False):
            ...     if aot:
            ...         if inputs is not None:
            ...             raise ValueError(
            ...                 "inputs and aot=True are mutually exclusive"
            ...             )
            ...         return {
            ...             arch: tuple(impl.build(aot=True) for impl in impls)
            ...             for arch, impls in cls.impl_map.items()
            ...         }
            ...
            ...     if inputs is None:
            ...         raise ValueError("inputs are required when aot=False")
            ...
            ...     target = cls.resolve_target(device=inputs.q.device)
            ...     impl = cls.resolve_backend(target, inputs)
            ...     return impl.build(inputs=inputs)
        """
        raise NotImplementedError
