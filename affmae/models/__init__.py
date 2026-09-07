"""Model architectures, selectable by name through the registry.

Kept import-light: the registry loads a spec's module on first lookup, so
``import affmae.models`` does not import every architecture.
"""

from .registry import ModelSpec, available_models, get_model_spec, register

__all__ = ["ModelSpec", "available_models", "get_model_spec", "register"]
