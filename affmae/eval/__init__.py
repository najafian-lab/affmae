"""Evaluation: segmentation metrics, foot-process-width geometry, comparisons.

Submodules are imported lazily so that pulling in the metrics does not also
pull in matplotlib or the dataset stack.
"""

__all__ = [
    "FpwParams",
    "amp_dtype_for",
    "collect_predictions",
    "compare_backends",
    "compare_models",
    "evaluate_fpw",
    "evaluate_fpw_across_seeds",
    "evaluate_segmentation",
    "iterate_predictions",
    "load_for_eval",
    "resolve_checkpoint",
]

_ORIGINS = {
    "FpwParams": "fpw",
    "amp_dtype_for": "loader",
    "collect_predictions": "segmentation",
    "compare_backends": "segmentation",
    "compare_models": "segmentation",
    "evaluate_fpw": "fpw",
    "evaluate_fpw_across_seeds": "fpw",
    "evaluate_segmentation": "segmentation",
    "iterate_predictions": "segmentation",
    "load_for_eval": "loader",
    "resolve_checkpoint": "loader",
}


def __getattr__(name):
    """Resolve public names to their submodule on first access (PEP 562)."""
    if name not in _ORIGINS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{_ORIGINS[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
