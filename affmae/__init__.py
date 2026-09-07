"""AFF-MAE: masked autoencoding over adaptive feature fields.

The curated public API. Importing this module is deliberately cheap — it pulls
in torch and nothing else, so the reusable pieces can be lifted into another
project without dragging along training, plotting, or optional dependencies.

Layering, enforced by ``tests/test_import_hygiene.py``:

* ``affmae.ops``      — standalone operators, one file per backend. torch only,
  no global state.
* ``affmae.layers``   — ``nn.Module`` classes built from those operators.
* ``affmae.eval``     — metrics and evaluation drivers.
* ``affmae.models``   — assembled architectures, selectable via the registry.
* ``affmae.viz``      — renderers. Imports matplotlib; nothing above may import it.
* ``affmae.training`` — training loops.

Model classes and the registry are exposed lazily via ``__getattr__`` so that
``import affmae`` does not construct anything or import every architecture.
"""

__version__ = "2.0.0"

__all__ = [
    "__version__",
    # Inference
    "AFFMAE",
    "AFFMAEPredictor",
    "SegmentationResult",
    "ReconstructionResult",
    "Mode",
    # Registry
    "get_model_spec",
    "available_models",
    "ModelSpec",
]

_LAZY = {
    "AFFMAE": ("affmae.inference", "AFFMAE"),
    # Previous name, kept so existing scripts keep working.
    "AFFMAEPredictor": ("affmae.inference", "AFFMAE"),
    "SegmentationResult": ("affmae.inference", "SegmentationResult"),
    "ReconstructionResult": ("affmae.inference", "ReconstructionResult"),
    "Mode": ("affmae.ops.policy", "Mode"),
    "get_model_spec": ("affmae.models.registry", "get_model_spec"),
    "available_models": ("affmae.models.registry", "available_models"),
    "ModelSpec": ("affmae.models.registry", "ModelSpec"),
}


def __getattr__(name):
    """Resolve public names on first access (PEP 562)."""
    try:
        module_path, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'affmae' has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module_path), attr)


def __dir__():
    return sorted(__all__)
