"""Run a trained model on an image, with no dataloader and no training config.

Example:
    from affmae import AFFMAEPredictor

    predictor = AFFMAEPredictor.from_checkpoint("best_model.pth",
                                               config="configs/aff_base_finetune_512_fpw.yaml")
    result = predictor.predict("docs/assets/sample1.png")
    result.labels          # [H, W] class indices
    result.save_overlay("out.png")

"""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from affmae.ops.policy import InferenceOnlyError, KernelPolicy, Mode
from affmae.utils.dist import resolve_device

__all__ = ["SegmentationResult", "ReconstructionResult", "AFFMAE",
           "AFFMAEPredictor", "Mode", "InferenceOnlyError"]

ImageSource = Union[str, os.PathLike, np.ndarray, torch.Tensor]



def _split_encoder_stages(captured: dict, sample: int = 0):
    """Turn the encoder's flat output dict into parallel per-stage lists.

    The encoder returns ``res2``, ``res2_pos``, ``res2_spatial_shape``, ``res3``,
    ... in one dict. Sorting the ``resN`` keys gives coarse-to-fine order.

    Args:
        captured: the encoder's output dict, or empty if nothing was captured.
        sample: batch index to take.
    Returns:
        ``(names, locations, features)``, all empty if there was nothing to take
        (a fixed-grid backbone does not produce per-stage positions).
    """
    names = sorted(key for key in captured
                   if key.startswith("res") and not key.endswith(
                       ("_pos", "_spatial_shape")))
    locations, features = [], []
    for name in names:
        position = captured.get(f"{name}_pos")
        feature = captured.get(name)
        if position is None or feature is None:
            return [], [], []
        locations.append(position[sample].detach().float().cpu())
        features.append(feature[sample].detach().float().cpu())
    return names, locations, features


@contextmanager
def _capture_encoder_stages(model):
    """Capture the encoder's per-stage output during a forward pass.

    A hook rather than a second encoder pass: ``segment`` already runs the
    encoder, and re-running it to collect positions would double the cost of
    every prediction for callers who never look at them.
    """
    from affmae.utils.dist import unwrap_model

    captured: dict = {}
    encoder = getattr(unwrap_model(model), "encoder", None)
    if encoder is None:
        yield captured
        return

    def hook(_module, _args, output):
        if isinstance(output, dict):
            captured.update(output)

    handle = encoder.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()



class _EncoderStages:
    """Per-stage token locations and features, shared by both result types.

    The encoder already returns both -- ``resN`` and ``resN_pos`` for each stage
    -- so a forward hook captures them during the prediction pass rather than
    paying for a second one. Downstream projects usually want exactly this: where
    the adaptive tokens landed, and the features at those points.
    """

    #: Filled in by the predictor; parallel lists, coarse index 0 to fine.
    locations: List[torch.Tensor]
    features: List[torch.Tensor]
    stage_names: List[str]
    patch_size: int

    def stage(self, name_or_index) -> "tuple[torch.Tensor, torch.Tensor]":
        """``(locations, features)`` for one stage, by name or index.

        Args:
            name_or_index: ``"res3"``, or an int index (``-1`` is the last stage).
        Returns:
            ``([N, 2], [N, C])`` on the CPU.
        Raises:
            KeyError: if the name is not one of :attr:`stage_names`.
        """
        if isinstance(name_or_index, str):
            if name_or_index not in self.stage_names:
                raise KeyError(
                    f"no stage {name_or_index!r}; have {self.stage_names}.")
            index = self.stage_names.index(name_or_index)
        else:
            index = int(name_or_index)
        return self.locations[index], self.features[index]

    def render_locations(self, path: str, stage=None, config=None) -> str:
        """Render token locations over the input image.

        Args:
            path: output image path.
            stage: None renders every stage side by side; a name or index renders
                just that one.
            config: VizConfig, or None for the paper defaults.
        Returns:
            ``path``.
        Raises:
            RuntimeError: if this result carries no stage data, which means the
                backbone is not an AFF encoder (a plain ViT keeps a fixed grid).
        """
        import os

        from affmae.viz import PAPER, draw_token_positions, render_token_layout

        if not self.locations:
            raise RuntimeError(
                "this result has no per-stage token locations; adaptive token "
                "layout is an AFF capability and a plain ViT has a fixed grid.")

        viz = config or PAPER
        image = self._stage_image()
        if stage is None:
            return render_token_layout(
                image.unsqueeze(0),
                [p.unsqueeze(0) for p in self.locations],
                self.patch_size, path,
                stage_names=self.stage_names, config=viz)

        positions, _ = self.stage(stage)
        canvas = draw_token_positions(image, positions, self.patch_size, viz)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        from PIL import Image

        Image.fromarray(canvas).save(path)
        return path

    def _stage_image(self) -> torch.Tensor:
        """The image the tokens sit on; overridden where it is not ``image``."""
        return self.image


@dataclass
class SegmentationResult(_EncoderStages):
    """One image's prediction.

    Attributes:
        labels: [H, W] long tensor of class indices; 0 is background.
        logits: [K, H, W] tensor from the finest head.
        image: [1, H, W] preprocessed input, kept so renderers need no reload.
        source: str description of where the image came from.
        num_classes: int, including background.
        locations: per-stage token positions, coarse to fine.
        features: per-stage token features, aligned with ``locations``.
        stage_names: encoder stage names, e.g. ``["res2", "res3", ...]``.
        patch_size: encoder patch size, needed to place tokens in pixels.
    """

    labels: torch.Tensor
    logits: torch.Tensor
    image: torch.Tensor
    source: str
    num_classes: int
    #: Per-stage token positions in patch units, coarse to fine. Empty for a
    #: fixed-grid backbone.
    locations: List[torch.Tensor] = field(default_factory=list)
    #: Per-stage token features, aligned with :attr:`locations`.
    features: List[torch.Tensor] = field(default_factory=list)
    stage_names: List[str] = field(default_factory=list)
    patch_size: int = 8

    @property
    def class_pixel_counts(self) -> "dict[int, int]":
        """Pixels assigned to each class, useful as a quick sanity check."""
        values, counts = self.labels.flatten().unique(return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}

    def save_overlay(self, path: str, config=None) -> str:
        """Render the prediction over its input.

        Args:
            path: output image path.
            config: VizConfig, or None for the paper defaults.
        Returns:
            ``path``.
        """
        from affmae.viz import PAPER, render_segmentation

        return render_segmentation(
            self.image.unsqueeze(0), self.logits.unsqueeze(0),
            self.num_classes, path, targets=None,
            titles=[self.source], config=config or PAPER)


@dataclass
class ReconstructionResult(_EncoderStages):
    """One image's MAE reconstruction.

    Attributes:
        original: [1, H, W] preprocessed input.
        masked: [1, H, W] input with masked patches zeroed.
        reconstructions: per-stage [1, H, W] predictions, coarse to fine.
        source: str description of the input.
        locations: per-stage positions of the *visible* tokens.
        features: per-stage token features, aligned with ``locations``.
        stage_names: encoder stage names.
        patch_size: encoder patch size.
    """

    original: torch.Tensor
    masked: torch.Tensor
    reconstructions: List[torch.Tensor]
    source: str
    #: Per-stage token positions in patch units, coarse to fine. These are the
    #: *visible* tokens: the MAE encoder never saw the masked ones.
    locations: List[torch.Tensor] = field(default_factory=list)
    #: Per-stage token features, aligned with :attr:`locations`.
    features: List[torch.Tensor] = field(default_factory=list)
    stage_names: List[str] = field(default_factory=list)
    patch_size: int = 8

    def _stage_image(self) -> torch.Tensor:
        """Tokens belong over the masked input: that is what the encoder saw."""
        return self.masked

    def save(self, path: str, config=None) -> str:
        """Render original / masked / per-stage reconstructions."""
        from affmae.viz import PAPER, render_reconstruction

        return render_reconstruction(
            self.original.unsqueeze(0), self.masked.unsqueeze(0),
            [r.unsqueeze(0) for r in self.reconstructions], path,
            config=config or PAPER, show_residual=True)


class AFFMAE:
    """A loaded AFF-MAE model plus the preprocessing it was trained with.

    What this object can do depends on the checkpoint it was loaded from, not on
    the class:

    * a **finetuned** checkpoint segments -- :meth:`segment`,
      :meth:`segment_batch`
    * a **pretraining** checkpoint reconstructs masked images --
      :meth:`reconstruct`
    * either one can show the adaptive token layout -- :meth:`token_layout`

    Ask before calling, rather than catching:

        model = AFFMAE.from_checkpoint("run/last_model.pth")
        if model.can_segment:
            result = model.segment("docs/assets/sample1.png")

    Args:
        model: an already-built ``nn.Module``.
        img_size: side length the model expects.
        num_classes: classes including background.
        device: torch.device the model lives on.
        patch_size: encoder patch size.
        mode: ``"inference"`` (default), ``"finetune"`` or ``"pretrain"``. In
            inference mode no backward state is built and the parameters do not
            require grad, so asking for a gradient raises
            :class:`~affmae.ops.policy.InferenceOnlyError` rather than failing
            somewhere deep in a kernel.
    """

    def __init__(self, model, img_size: int, num_classes: int, device,
                 patch_size: int = 8, mode: "str | Mode" = Mode.INFERENCE):
        self.model = model
        self.img_size = int(img_size)
        self.num_classes = int(num_classes)
        self.patch_size = int(patch_size)
        self.device = torch.device(device)
        self.mode = Mode.parse(mode)
        self.policy = KernelPolicy.for_mode(self.mode, img_size=self.img_size,
                                            device=self.device)
        if not self.policy.params_requires_grad:
            self.model.requires_grad_(False)
            self.model.eval()

    # -- capability introspection -----------------------------------------

    @property
    def capabilities(self) -> "frozenset[str]":
        """What this checkpoint supports: any of ``segment``, ``reconstruct``,
        ``token_layout``."""
        from affmae.utils.dist import unwrap_model

        model = unwrap_model(self.model)
        found = set()
        if hasattr(model, "seg_head") or hasattr(model, "decode_head") or \
                self.num_classes > 0:
            found.add("segment")
        if hasattr(model, "patchify") and hasattr(model, "unpatchify"):
            found.add("reconstruct")
        encoder = getattr(model, "encoder", None)
        if encoder is not None and hasattr(encoder, "forward_with_pos"):
            found.add("token_layout")
        return frozenset(found)

    @property
    def can_segment(self) -> bool:
        """True if :meth:`segment` will work on this checkpoint."""
        return "segment" in self.capabilities

    @property
    def can_reconstruct(self) -> bool:
        """True if :meth:`reconstruct` will work on this checkpoint."""
        return "reconstruct" in self.capabilities

    def _forbid_grad(self, what: str) -> None:
        """Raise if a gradient is being asked of an inference-mode model.

        Args:
            what: the operation being attempted, for the message.
        Raises:
            InferenceOnlyError: always, when the mode is INFERENCE.
        """
        if self.mode is Mode.INFERENCE:
            raise InferenceOnlyError(
                f"this model was loaded with mode='inference', so {what} is "
                f"not available: parameters do not require grad and the "
                f"kernels skip the state a backward pass would need. Reload "
                f"with mode='finetune' or mode='pretrain' to train.")

    def forward(self, images: torch.Tensor):
        """Run the model on an already-preprocessed batch.

        Args:
            images: [B, C, H, W] normalized input on any device.
        Returns:
            Whatever the underlying model returns -- for segmentation, a list
            of deep-supervision heads with the finest last.
        Raises:
            InferenceOnlyError: if the mode is ``"inference"`` and a gradient
                is being requested.
        """
        if self.mode is Mode.INFERENCE:
            if torch.is_grad_enabled() and images.requires_grad:
                self._forbid_grad("backpropagating through the model")
            with torch.no_grad():
                return self.model(images.to(self.device))
        return self.model(images.to(self.device))

    __call__ = forward

    @classmethod
    def from_checkpoint(cls, checkpoint: str, config=None,
                        device: Union[str, torch.device, None] = None,
                        model_type: Optional[str] = None,
                        task: str = "auto",
                        mode: "str | Mode" = Mode.INFERENCE,
                        cluster_attention_backend: str = "auto",
                        decoder_deform_backend: str = "auto") -> "AFFMAE":
        """Build a predictor from a checkpoint and its config.

        Args:
            checkpoint: a path to a ``.pth``, an ``http(s)`` URL (including a
                Google Drive share link), or an :class:`~affmae.data.weights.EMWeights`
                member. URLs and registry members are downloaded once into
                ``$CHECKPOINT_ROOT`` and reused after that.
            config: a config object, a path to a YAML, or None. None looks for
                ``config.yaml`` beside the checkpoint (the training scripts copy
                it there); for an EMWeights member it falls back to the config
                that checkpoint was trained with.
            device: ``"cuda"``, ``"cpu"``, ``"mps"``, or None for the best
                available. An unavailable choice is downgraded with a warning.
            model_type: overrides ``config.model_type``.
            task: which head to build. ``"auto"`` (default) reads the config: a
                config with ``num_classes`` builds the segmentation model, one
                with ``mask_ratio`` and no ``num_classes`` builds the MAE.
                ``"segmentation"`` and ``"pretrain"`` force one. Without this
                every checkpoint built a segmentation model, so
                :meth:`reconstruct` was unreachable from a checkpoint despite
                being advertised in :attr:`capabilities`.
            mode: ``"inference"`` (default), ``"finetune"`` or ``"pretrain"``.
                Inference omits state used only by backward and rejects gradient
                requests. All modes otherwise use the same forward backends.
            cluster_attention_backend: neighbourhood-attention implementation.
                ``"auto"`` uses fused Triton on a supported CUDA/ROCm device and
                the PyTorch fallback elsewhere.
            decoder_deform_backend: point-decoder implementation. ``"auto"``
                and ``"fused"`` select ``"csr_knn_cached"``; that algorithm has
                a fused Triton implementation and an equivalent PyTorch fallback.
        Returns:
            An AFFMAE in eval mode.
        Raises:
            FileNotFoundError: if the checkpoint or an inferred config is absent.
        """
        from affmae.config import load_config
        from affmae.eval.loader import load_state_dict_into
        from affmae.models.registry import get_model_spec
        from affmae.ops.deform_attn_torch import resolve_deform_backend
        from affmae.ops.nbhd_attn import resolve_backend

        from affmae.data.weights import resolve_source
        from affmae.utils.paths import repo_root as _repo_root

        # Accepts a registry member, a URL, or a path. A registry member also
        # supplies task/img_size/num_classes, so `from_checkpoint(EMWeights.X)`
        # needs no config argument at all.
        checkpoint, spec = resolve_source(checkpoint)

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

        if config is None and spec is not None:
            config = os.path.join(str(_repo_root()), spec.config)
            if not os.path.exists(config):
                raise FileNotFoundError(
                    f"{spec.filename} was trained with {spec.config}, which is "
                    f"not in this checkout. Pass config= explicitly.")

        if config is None:
            beside = os.path.join(os.path.dirname(os.path.abspath(checkpoint)),
                                  "config.yaml")
            if not os.path.exists(beside):
                raise FileNotFoundError(
                    f"no config given and none found at {beside}. Pass "
                    f"config=<path to the YAML used for training>.")
            config = beside
        cfg = load_config(config) if isinstance(config, (str, os.PathLike)) else config

        if model_type is not None:
            cfg.model_type = model_type

        resolved = resolve_device(device if device is not None
                                  else getattr(cfg, "device", None))
        cfg.device = resolved

        resolved_mode = Mode.parse(mode)
        KernelPolicy.for_mode(resolved_mode, img_size=cfg.img_size,
                              device=resolved).apply_to_config(cfg)

        # Validate and normalize the two public component controls before model
        # construction. Explicit arguments intentionally win over the YAML.
        cfg.cluster_attention_backend = resolve_backend(
            cluster_attention_backend)
        cfg.decoder_deform_backend = resolve_deform_backend(
            decoder_deform_backend)

        if task not in ("auto", "segmentation", "pretrain"):
            raise ValueError(
                f"task must be auto|segmentation|pretrain, got {task!r}.")
        if task == "auto":
            task = spec.task if spec is not None else (
                "segmentation" if getattr(cfg, "num_classes", None)
                else "pretrain")

        spec = get_model_spec(cfg.model_type)
        if task == "pretrain":
            model = spec.build_pretrain(cfg)
        else:
            model = spec.build_segmentation(cfg)
        load_state_dict_into(model, checkpoint, map_location="cpu")

        return cls(model.to(resolved).eval(), cfg.img_size,
                   getattr(cfg, "num_classes", 0) or 0,
                   resolved, patch_size=getattr(cfg, "patch_size", 8),
                   mode=resolved_mode)

    @classmethod
    def from_model(cls, model, img_size: int, num_classes: int,
                   patch_size: int = 8, device=None,
                   mode: "str | Mode" = Mode.INFERENCE) -> "AFFMAE":
        """Wrap a model that is already built, e.g. mid-training.

        Lets the training loop reuse the same extraction and rendering path as
        the demo instead of keeping a second copy of it.

        Args:
            model: a built ``nn.Module``.
            img_size: side length the model expects.
            num_classes: classes including background.
            patch_size: encoder patch size.
            device: where the model lives; inferred from its parameters if None.
            mode: see :meth:`from_checkpoint`.
        Returns:
            An AFFMAE wrapping ``model``. The model is not copied.
        """
        if device is None:
            device = next(model.parameters()).device
        return cls(model, img_size, num_classes, device,
                   patch_size=patch_size, mode=mode)

    def _prepare(self, source: ImageSource) -> "tuple[torch.Tensor, str]":
        from affmae.data.preprocess import preprocess_image

        label = source if isinstance(source, (str, os.PathLike)) else "<array>"
        tensor = preprocess_image(source, self.img_size)
        return tensor.to(self.device), str(label)

    @torch.no_grad()
    def segment(self, source: ImageSource) -> SegmentationResult:
        """Segment one image. Needs a finetuned checkpoint.

        Args:
            source: path, numpy array, PIL image, or tensor.
        Returns:
            A SegmentationResult.
        """
        tensor, label = self._prepare(source)
        with _capture_encoder_stages(self.model) as captured:
            outputs = self.model(tensor)
        # forward returns a list of deep-supervision heads, finest last. Every
        # call site used to re-implement this unpacking.
        logits = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
        names, locations, features = _split_encoder_stages(captured)
        return SegmentationResult(
            locations=locations, features=features, stage_names=names,
            patch_size=self.patch_size,
            labels=logits[0].argmax(dim=0).cpu(),
            logits=logits[0].float().cpu(),
            image=tensor[0].cpu(),
            source=label,
            num_classes=self.num_classes,
        )

    @torch.no_grad()
    def reconstruct(self, source: ImageSource,
                    mask_ratio: float = 0.5) -> ReconstructionResult:
        """Reconstruct a masked image with an MAE model.

        Args:
            source: image source.
            mask_ratio: fraction of patches to hide.
        Returns:
            A ReconstructionResult.
        Raises:
            NotImplementedError: if this checkpoint has no MAE decoder. A
                finetuned segmentation model has replaced that head, so
                reconstruction needs a *pretraining* checkpoint.
        """
        from affmae.utils.dist import unwrap_model

        model = unwrap_model(self.model)
        if not hasattr(model, "_forward_internal") or not hasattr(model, "unpatchify"):
            raise NotImplementedError(
                f"{type(model).__name__} has no MAE reconstruction head; load a "
                f"pretraining checkpoint instead of a finetuned one.")

        tensor, label = self._prepare(source)
        previous = getattr(model, "mask_ratio", None)
        try:
            if previous is not None:
                model.mask_ratio = float(mask_ratio)
            with _capture_encoder_stages(self.model) as captured:
                outputs = model._forward_internal(tensor)
        finally:
            if previous is not None:
                model.mask_ratio = previous

        patches = model.patchify(tensor)
        keep = outputs["ids_keep"]
        restore = outputs["ids_restore"]
        visible = torch.gather(
            patches, 1, keep.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))

        def assemble(predicted):
            merged = torch.cat([visible, predicted], dim=1)
            ordered = torch.gather(
                merged, 1,
                restore.unsqueeze(-1).expand(-1, -1, merged.shape[-1]))
            return model.unpatchify(ordered)[0].float().cpu()

        stages = [assemble(p) for p in outputs["all_preds"]]
        blanked = assemble(torch.zeros_like(outputs["all_preds"][0]))

        names, locations, features = _split_encoder_stages(captured)
        return ReconstructionResult(original=tensor[0].cpu(), masked=blanked,
                                    locations=locations, features=features,
                                    stage_names=names,
                                    patch_size=self.patch_size,
                                    reconstructions=stages, source=label)

    def segment_batch(self, sources: Sequence[ImageSource]
                      ) -> List[SegmentationResult]:
        """Segment several images. Needs a finetuned checkpoint.

        Args:
            sources: image sources, each anything :meth:`segment` accepts.
        Returns:
            One SegmentationResult per input, in order.

        Note:
            Loops rather than batching: inputs may differ in original size, and
            the memory profile at 1024 makes one-at-a-time the safe default.
        """
        return [self.segment(source) for source in sources]

    @torch.no_grad()
    def token_layout(self, source: ImageSource,
                     mask_ratio: "float | None" = None
                     ) -> "tuple[torch.Tensor, list]":
        """Return the encoder's token positions, one entry per stage.

        This is the method's signature visualization: it shows where the
        adaptive tokens actually land, rather than on a fixed grid.

        Args:
            source: image source.
            mask_ratio: None (default) embeds the whole image, which is what a
                finetuned model sees. A float in (0, 1) instead applies the MAE's
                Perlin mask and returns the positions of the *visible* tokens --
                what the pretraining encoder actually received. Needs a
                pretraining checkpoint.

                Each call draws a **new** mask. So do not pair this with a
                separate :meth:`reconstruct` and plot them together: the two
                masks differ, and only about half the tokens will sit on a patch
                the image shows as visible. Use ``reconstruct(...).locations``,
                which comes from the same pass.
        Returns:
            ``(image, positions_per_stage)``: the preprocessed [1, H, W] input,
            and per stage an [N_s, 2] tensor of positions in patch units.
        Raises:
            NotImplementedError: if the encoder does not expose per-stage
                positions -- a plain ViT keeps a fixed grid, so there is nothing
                adaptive to show -- or if ``mask_ratio`` is given for a
                checkpoint with no MAE masking.
        """
        from affmae.utils.dist import unwrap_model

        tensor, _ = self._prepare(source)
        model = unwrap_model(self.model)
        encoder = getattr(model, "encoder", None)
        if encoder is None or not hasattr(encoder, "forward_with_pos"):
            raise NotImplementedError(
                f"{type(model).__name__} has no encoder exposing "
                f"forward_with_pos; adaptive token layout is an AFF capability "
                f"(a plain ViT uses a fixed grid).")

        if mask_ratio is None:
            # ids_masked=None makes the patch embed behave as a plain conv stem.
            pos, feat, height, width = encoder.patch_embed(tensor, ids_masked=None)
        else:
            if not hasattr(model, "mask_and_embed"):
                raise NotImplementedError(
                    f"{type(model).__name__} does not mask its input, so "
                    f"mask_ratio has no meaning here; load a pretraining "
                    f"checkpoint or leave mask_ratio as None.")
            previous = model.mask_ratio
            try:
                model.mask_ratio = float(mask_ratio)
                embedded = model.mask_and_embed(tensor)
            finally:
                model.mask_ratio = previous
            feat = embedded["visible_tokens"]
            pos = embedded["visible_pos"]
            height, width = embedded["h"], embedded["w"]

        stage_positions = encoder.forward_with_pos(feat, pos, height, width)
        return tensor[0].cpu(), [p[0].float().cpu() for p in stage_positions]


#: Previous name for :class:`AFFMAE`. ``predict``/``predict_batch`` were renamed
#: to ``segment``/``segment_batch``, because the class also reconstructs -- which
#: of the two a given instance can do depends on the checkpoint.
AFFMAEPredictor = AFFMAE
