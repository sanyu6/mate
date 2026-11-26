"""Shared MATE Python helpers."""

import functools
from typing import Any
from collections import OrderedDict
from collections.abc import Callable

import torch

from .arch import Arch, MP31_ARCH, MP32_ARCH


def tensor_cache(
    fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent results of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    The cache is limited to a fixed size (default is 256). When the cache is full, the oldest entry will be removed.

    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """

    cache: "OrderedDict[tuple[tuple[int, ...], tuple[tuple[str, int], ...]], tuple[tuple[Any, ...], dict[str, Any], Any]]" = OrderedDict()
    cache_size = 256

    def get_id(x: Any):
        if (type(x) is int) or (type(x) is float) or (type(x) is str):
            return x
        else:
            return id(x)

    def make_identity_key(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[int, ...], tuple[tuple[str, int], ...]]:
        args_key = tuple(get_id(a) for a in args)
        kwargs_key = tuple(sorted((k, get_id(v)) for k, v in kwargs.items()))
        return args_key, kwargs_key

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache, cache_size
        key = make_identity_key(args, kwargs)
        if key in cache:
            cache.move_to_end(key, last=True)
            _, _, cached_result = cache[key]
            return cached_result

        result = fn(*args, **kwargs)
        cache[key] = (args, kwargs, result)
        cache.move_to_end(key, last=True)
        if len(cache) > cache_size:
            cache.popitem(last=False)
        return result

    return wrapper


def _row_major_strides(shape):
    stride = 1
    strides = []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return tuple(reversed(strides))


def cosize(shape, strides=None):
    """Return the storage span for a shape/stride layout, measured in elements."""

    shape = tuple(shape)
    if strides is None:
        strides = _row_major_strides(shape)
    else:
        strides = tuple(strides)
        assert len(shape) == len(strides), "shape and strides must have the same rank"

    size = 1
    for extent, stride in zip(shape, strides):
        size += (extent - 1) * stride
    return size


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def round_up(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def cc_switch():
    """Dispatch to a registered implementation for the selected compute capability."""

    def decorator(default):
        implementations = {}

        @functools.wraps(default)
        def wrapper(*args, **kwargs):
            cc = kwargs.pop("override_cc", None)
            if cc is None:
                from .mate_runtime import get_musa_arch

                arch = get_musa_arch()
                cc = arch.major * 10 + arch.minor
            return implementations.get(cc, default)(*args, **kwargs)

        def register(cc):
            def register_implementation(implementation):
                implementations[cc] = implementation
                return implementation

            return register_implementation

        wrapper.register = register
        return wrapper

    return decorator


__all__ = [
    "Arch",
    "MP31_ARCH",
    "MP32_ARCH",
    "cc_switch",
    "ceil_div",
    "cosize",
    "round_up",
    "tensor_cache",
]
