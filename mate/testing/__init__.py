"""Testing utilities package."""

from importlib import import_module


_LAZY_ATTR_MODULES = {
    "KernelMetrics": "metrics",
    "quantize_sage_attention_tensor": "sage_attention_quantization",
    "repeat_check": "repeatability",
    "requires_musa_compute_capability_ge": "arch",
    "supported_musa_compute_capability": "arch",
    "quantize_mxfp4": "quantize",
    "quantize_mxfp8": "quantize",
    "dequantize_mxfp4": "quantize",
    "dequantize_mxfp8": "quantize",
}
__all__ = [
    "KernelMetrics",
    "quantize_sage_attention_tensor",
    "repeat_check",
    "requires_musa_compute_capability_ge",
    "supported_musa_compute_capability",
    "quantize_mxfp4",
    "quantize_mxfp8",
    "dequantize_mxfp4",
    "dequantize_mxfp8",
]


def __getattr__(name: str):
    module_name = _LAZY_ATTR_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
