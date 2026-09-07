"""Mode policy for forward-only inference and gradient-enabled use. Thanks for claude for most of this code.

Inference does not build tables or values consumed only by backward. This reduces forward overhead, but deliberately makes backward unavailable. Finetune and pretrain modes retain that state and allow gradients.

"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional

__all__ = ["Mode", "KernelPolicy", "InferenceOnlyError"]


class InferenceOnlyError(RuntimeError):
    """Raised when a gradient is requested from a model in inference mode."""


class Mode(str, Enum):
    """How a loaded model is going to be used.

    Attributes:
        INFERENCE: forward only. No backward state is built and parameters do
            not require grad.
        FINETUNE: training a segmentation head and backbone from a pretrained
            checkpoint.
        PRETRAIN: masked-autoencoding pretraining.
    """

    INFERENCE = "inference"
    FINETUNE = "finetune"
    PRETRAIN = "pretrain"

    @classmethod
    def parse(cls, value: "str | Mode") -> "Mode":
        """Accept a Mode or its string name.

        Args:
            value: a Mode, or one of ``"inference"``, ``"finetune"``,
                ``"pretrain"``.
        Returns:
            The Mode.
        Raises:
            ValueError: on anything else.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError:
            raise ValueError(
                f"unknown mode {value!r}; expected one of "
                f"{', '.join(m.value for m in cls)}.") from None


@dataclass(frozen=True)
class KernelPolicy:
    """Kernel settings for one mode, at one resolution.

    Args:
        mode: which regime this policy is for.
        knn_cache: cache the dense KNN table across the blocks of a decoder
            stage. Bit-identical either way, so this is purely a speed choice.
        build_backward_state: build the structures only the backward pass reads
            (the KV-owner edge index and edge-coefficient CSR). False under
            inference.
        params_requires_grad: whether model parameters require grad.
        launch_overrides: optional per-kernel launch parameters.
    """

    mode: Mode
    knn_cache: bool = True
    build_backward_state: bool = True
    params_requires_grad: bool = True
    launch_overrides: tuple = ()

    @classmethod
    def for_mode(cls, mode: "str | Mode", img_size: Optional[int] = None,
                 device: Any = None) -> "KernelPolicy":
        """Recommended settings for a mode.

        Args:
            mode: see :class:`Mode`.
            img_size: reserved for resolution-dependent policy settings.
            device: reserved for device-dependent policy settings.
        Returns:
            A KernelPolicy.
        """
        resolved = Mode.parse(mode)
        if resolved is Mode.INFERENCE:
            return cls(mode=resolved, knn_cache=True,
                       build_backward_state=False,
                       params_requires_grad=False)
        return cls(mode=resolved, knn_cache=True, build_backward_state=True,
                   params_requires_grad=True)

    def apply_to_config(self, cfg: Any) -> Any:
        """Write the mode-dependent settings onto a config, in place.

        Backend selection remains independent and is not modified here.

        Args:
            cfg: a loaded Config.
        Returns:
            ``cfg``.
        """
        cfg.decoder_knn_cache = self.knn_cache
        return cfg

    def with_(self, **changes) -> "KernelPolicy":
        """Return a copy with ``changes`` applied.

        Args:
            **changes: any field of this dataclass.
        Returns:
            A new KernelPolicy.
        """
        return replace(self, **changes)
