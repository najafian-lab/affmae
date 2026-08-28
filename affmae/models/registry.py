"""Model registry: one ``ModelSpec`` per method, looked up by ``cfg.model_type``.

A spec bundles five concerns into one object registered next to the model it
describes -- building the pretrain model, building the segmentation model,
remapping a checkpoint's keys, assigning layer-decay ids, and knowing which aux
losses the model emits. Adding a method means adding a module, not editing every
driver::

    from affmae.models.registry import get_model_spec

    spec  = get_model_spec(cfg.model_type)
    model = spec.build_segmentation(cfg)
    state = spec.adapt_state_dict(state_dict, model, cfg)
    optim = build_optimizer_with_llrd(model, cfg, spec.default_llrd, spec.layer_decay_plan)

Contract, so that wrapping the model for DDP stays possible:

* Builders return a **bare, unwrapped** ``nn.Module``. Wrapping —
  ``accelerator.prepare``, ``SyncBatchNorm`` conversion — is the driver's job.
* ``adapt_state_dict`` and ``layer_decay_plan`` are always handed the
  **unwrapped** module and may assume so; drivers call
  :func:`affmae.utils.dist.unwrap_model` first.
* Specs are frozen and process-global. Never stash a model, device, or rank on
  one.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

__all__ = [
    "LayerDecayPlan",
    "ModelSpec",
    "register",
    "get_model_spec",
    "available_models",
    "resolve_alias",
]


@dataclass(frozen=True)
class LayerDecayPlan:
    """How to spread layer-wise LR decay over one architecture's parameters.

    Attributes:
        num_layers: int, depth of the backbone trunk for decay purposes.
        layer_id: callable, maps a parameter name to its trunk depth in
            ``[0, num_layers)``. Parameters outside the trunk (decoder, heads)
            never reach this function.
    """

    num_layers: int
    layer_id: Callable[[str], int]


def _identity_adapt(state_dict, model, cfg):
    """Default checkpoint adapter: pass the keys through untouched."""
    return state_dict


@dataclass(frozen=True)
class ModelSpec:
    """Everything the drivers need to know about one model family.

    Attributes:
        name: str, canonical ``model_type`` value.
        build_segmentation: callable, ``(cfg) -> nn.Module`` for finetuning.
        build_pretrain: callable or None, ``(cfg) -> nn.Module`` for MAE
            pretraining. None means this method is finetune-only.
        adapt_state_dict: callable, ``(state_dict, model, cfg) -> state_dict``,
            applied to a pretrained checkpoint before loading.
        layer_decay_plan: callable or None, ``(model, cfg) -> LayerDecayPlan``.
            None falls back to flat (undecayed) parameter groups.
        aux_names: tuple of str, deep-supervision outputs of the *segmentation*
            model, in the order it returns them. The last entry is the primary
            head.
        pretrain_aux_names: tuple of str or None, the same for the *pretraining*
            model, which need not match. ViT is the case in point: its
            segmentation model emits one head, but ``VanillaViTMAE`` returns no
            auxiliary losses at all. None means "same as ``aux_names``"; use an
            empty tuple to mean "none".
        default_llrd: float, layer-wise LR decay this architecture wants.
        supports_token_viz: bool, whether ``affmae.viz.model_figures.render_tokens``
            can render this model's token layout. The renderers are free
            functions, not model methods, so models stay free of matplotlib.
        reconstruction_renderer: str or None, name of the function in
            ``affmae.viz.model_figures`` that renders this model's MAE
            reconstructions. None means the model has no reconstruction view.
        aliases: tuple of str, legacy ``model_type`` values that map here.
    """

    name: str
    build_segmentation: Callable
    build_pretrain: Optional[Callable] = None
    adapt_state_dict: Callable = _identity_adapt
    layer_decay_plan: Optional[Callable] = None
    aux_names: Tuple[str, ...] = ("res2",)
    pretrain_aux_names: Optional[Tuple[str, ...]] = None
    default_llrd: float = 0.8
    supports_token_viz: bool = False
    reconstruction_renderer: Optional[str] = None
    aliases: Tuple[str, ...] = ()

    @property
    def pretrain_aux(self) -> Tuple[str, ...]:
        """Auxiliary loss names for pretraining, defaulting to ``aux_names``."""
        return self.aux_names if self.pretrain_aux_names is None else self.pretrain_aux_names


_REGISTRY: Dict[str, ModelSpec] = {}
_ALIASES: Dict[str, str] = {}
_SPECS_LOADED = False


def register(spec: ModelSpec) -> ModelSpec:
    """Add a spec to the registry.

    Args:
        spec: ModelSpec to register.
    Returns:
        The same spec, so this can be used as a module-level expression.
    Raises:
        ValueError: on a duplicate name or alias collision.
    """
    if spec.name in _REGISTRY:
        raise ValueError(f"Model '{spec.name}' is already registered.")
    for alias in spec.aliases:
        if alias in _ALIASES or alias in _REGISTRY:
            raise ValueError(
                f"Alias '{alias}' for model '{spec.name}' collides with an "
                f"existing model or alias."
            )
    _REGISTRY[spec.name] = spec
    for alias in spec.aliases:
        _ALIASES[alias] = spec.name
    return spec


def _load_builtin_specs() -> None:
    """Import the built-in specs, then any optional contrib specs.

    Deferred to first lookup so that ``import affmae.models.registry`` stays cheap
    and free of circular imports: spec modules import model classes, and model
    classes may import from this package.
    """
    global _SPECS_LOADED
    if _SPECS_LOADED:
        return
    _SPECS_LOADED = True
    import affmae.models.specs  # noqa: F401  (registration side effects)

    try:
        # Optional: baseline comparison models live on the `rebuttals` branch
        # and drop in here without any edit to the release code.
        import affmae.models.contrib  # noqa: F401
    except ImportError:
        pass


def resolve_alias(name: str) -> str:
    """Map a legacy ``model_type`` onto its canonical name.

    Args:
        name: str, possibly-legacy model type.
    Returns:
        The canonical name, or ``name`` unchanged if it is not an alias.
    """
    _load_builtin_specs()
    return _ALIASES.get(name, name)


def get_model_spec(name: str) -> ModelSpec:
    """Look up a registered spec by name or legacy alias.

    Args:
        name: str, value of ``cfg.model_type``.
    Returns:
        The matching ModelSpec.
    Raises:
        KeyError: if nothing matches, listing what is available.
    """
    _load_builtin_specs()
    canonical = _ALIASES.get(name, name)
    try:
        return _REGISTRY[canonical]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        aliased = ", ".join(sorted(_ALIASES)) or "(none)"
        raise KeyError(
            f"Unknown model_type '{name}'. Registered models: {known}. "
            f"Legacy aliases: {aliased}. Baseline models (mixmae, hiera, "
            f"hivit, greenmim, swin) live on the 'rebuttals' branch."
        ) from None


def available_models() -> list:
    """Return the sorted canonical names of every registered model."""
    _load_builtin_specs()
    return sorted(_REGISTRY)
