"""Gradio demo: upload images, get masks, overlays, reconstructions, tokens.

Launched by ``python inference.py --checkpoint <ckpt> --gradio``. Gradio is an
optional dependency::

    pip install -e ".[demo]"
"""

import getpass
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from types import MappingProxyType

import numpy as np

from affmae.inference import AFFMAE
from affmae.viz import VizConfig

logger = logging.getLogger(__name__)

__all__ = ["build_interface", "launch"]

def _sample_images():
    """The bundled example glomerulus crops, for the example galleries.

    Paths, not arrays: Gradio loads an example lazily when it is clicked, and a
    missing file is skipped rather than breaking the whole interface, so a user
    who pip-installed the package without the repo still gets a working UI.
    """
    from affmae.utils.paths import repo_root

    assets = repo_root() / "docs" / "assets"
    return [[str(assets / f"sample{n}.png")] for n in (1, 2, 3, 4)
            if (assets / f"sample{n}.png").is_file()]



#: Finetuned checkpoints the resolution picker can offer.
_FT_BY_RESOLUTION = MappingProxyType({512: "AFFMAE_BASE_FT_512",
                                      768: "AFFMAE_BASE_FT_768",
                                      1024: "AFFMAE_BASE_FT_1024"})


def _available_resolutions(loaded: AFFMAE):
    """Resolutions the picker can actually serve, cheapest check first.

    Only offers a resolution whose checkpoint is already on disk, plus whatever
    was passed on the command line. A demo should not download 800 MB because
    someone clicked a radio button.
    """
    from affmae.data.weights import EMWeights

    found = {int(loaded.img_size)}
    for resolution, member in _FT_BY_RESOLUTION.items():
        entry = getattr(EMWeights, member)
        if os.path.isfile(entry.download_path):
            found.add(int(resolution))
    return sorted(found)


def _predictor_cache(loaded: AFFMAE, device):
    """Return ``get(resolution) -> (predictor, error)``, loading lazily.

    A finetuned checkpoint is resolution-specific: the token count scales with
    img_size/patch_size, so running the 512px weights at 1024 is not the same
    model. Each resolution therefore needs its own checkpoint.
    """
    from affmae.data.weights import EMWeights

    cache = {int(loaded.img_size): loaded}

    def get(resolution):
        resolution = int(resolution)
        if resolution in cache:
            return cache[resolution], None
        member = _FT_BY_RESOLUTION.get(resolution)
        if member is None:
            return None, f"No checkpoint is registered for {resolution}px."
        entry = getattr(EMWeights, member)
        if not os.path.isfile(entry.download_path):
            return None, (f"The {resolution}px checkpoint is not downloaded. "
                          f"Fetch it with "
                          f"`python -c \"from affmae.data.weights import "
                          f"EMWeights; EMWeights.{member}.fetch()\"`.")
        try:
            cache[resolution] = AFFMAE.from_checkpoint(entry, device=device)
        except Exception as error:                        # pragma: no cover
            return None, f"Could not load the {resolution}px checkpoint: {error}"
        logger.info("loaded %dpx checkpoint for the demo", resolution)
        return cache[resolution], None

    return get


def _viz_config(dpi, alpha, radius, scale):
    """Build a VizConfig from the slider values."""
    return VizConfig(
        dpi=int(dpi),
        overlay_alpha=float(alpha),
        token_radius=int(radius) if radius else None,
        token_render_scale=float(scale),
    )


def _segment(resolve, resolution, image, dpi, alpha):
    """Segmentation tab: overlay plus a per-class pixel share table."""
    if image is None:
        return None, "Upload an image first."
    predictor, error = resolve(resolution)
    if error:
        return None, error
    result = predictor.segment(np.asarray(image))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        path = handle.name
    result.save_overlay(path, config=_viz_config(dpi, alpha, None, 1.0))

    counts = result.class_pixel_counts
    total = sum(counts.values()) or 1
    lines = [f"**{result.labels.shape[0]}x{result.labels.shape[1]}** prediction",
             "", "| class | pixels | share |", "|---|---|---|"]
    for class_id in sorted(counts):
        label = "background" if class_id == 0 else f"class {class_id}"
        lines.append(f"| {label} | {counts[class_id]:,} | "
                     f"{100 * counts[class_id] / total:.1f}% |")
    return path, "\n".join(lines)


def _reconstruct(predictor, image, mask_ratio):
    """Reconstruction tab. Only available for models with an MAE decoder."""
    if image is None:
        return None, "Upload an image first."
    try:
        result = predictor.reconstruct(np.asarray(image), mask_ratio=mask_ratio)
    except (AttributeError, NotImplementedError) as exc:
        return None, (f"This checkpoint has no reconstruction head: {exc}\n\n"
                      f"Reconstruction needs a pretraining (MAE) checkpoint; "
                      f"a finetuned segmentation model has replaced that head.")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        path = handle.name
    result.save(path)
    return path, f"Masked {mask_ratio:.0%} of patches."


def _tokens(resolve, resolution, image, radius, scale):
    """Token-layout tab: where the adaptive tokens land, per encoder stage."""
    if image is None:
        return None, "Upload an image first."
    predictor, error = resolve(resolution)
    if error:
        return None, error
    try:
        rendered, positions = predictor.token_layout(np.asarray(image))
    except (RuntimeError, AttributeError) as exc:
        return None, (f"Token layout unavailable for this model: {exc}\n\n"
                      f"It is an AFF capability -- a plain ViT has a fixed grid.")
    from affmae.viz import render_token_layout

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        path = handle.name
    render_token_layout(rendered.unsqueeze(0),
                        [p.unsqueeze(0) for p in positions],
                        patch_size=predictor.patch_size,
                        save_path=path,
                        config=_viz_config(150, 0.65, radius, scale))
    counts = ", ".join(str(p.shape[0]) for p in positions)
    return path, f"Tokens per stage: {counts}"


def _batch(predictor, files, dpi, alpha):
    """Batch tab: several images in, one zip of masks and overlays out."""
    if not files:
        return None, "Upload one or more images."
    config = _viz_config(dpi, alpha, None, 1.0)
    workdir = Path(tempfile.mkdtemp())
    written = []
    for item in files:
        source = item.name if hasattr(item, "name") else str(item)
        stem = Path(source).stem
        try:
            result = predictor.segment(source)
        except Exception as exc:                       # keep going on one bad file
            logger.warning("skipped %s: %s", source, exc)
            continue
        from PIL import Image

        mask_path = workdir / f"{stem}_mask.png"
        Image.fromarray(result.labels.numpy().astype("uint8")).save(mask_path)
        overlay_path = workdir / f"{stem}_overlay.png"
        result.save_overlay(str(overlay_path), config=config)
        written.extend([mask_path, overlay_path])

    if not written:
        return None, "Every upload failed to load; see the log."

    archive = workdir / "affmae_predictions.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in written:
            bundle.write(path, arcname=path.name)
    return str(archive), f"Processed {len(written) // 2} image(s)."


def _scope_gradio_temp_dir():
    """Point GRADIO_TEMP_DIR at a per-user directory if the caller has not.

    Gradio caches served files under /tmp/gradio, which is shared between users
    on a multi-user host: whoever starts first owns it and everyone else gets
    PermissionError -> HTTP 500 on every result image. build_interface needs
    this too, not just launch, because gr.Examples writes into the cache while
    the interface is being built.
    """
    if os.environ.get("GRADIO_TEMP_DIR"):
        return
    private = os.path.join(tempfile.gettempdir(),
                           f"gradio-{getpass.getuser()}")
    os.makedirs(private, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = private


def build_interface(predictor: AFFMAE, mae: AFFMAE = None,
                    device=None, gpu=None):
    """Assemble the Gradio interface around a loaded predictor.

    Args:
        predictor: an AFFMAE with a segmentation head.
        mae: an AFFMAE loaded from a *pretraining* checkpoint, for the
            reconstruction tab. A finetuned model has replaced the MAE head, so
            without this that tab can only explain why it cannot run.
        device: device the resolution-specific checkpoints load onto. None uses
            the predictor's.
        gpu: a decorator applied to every handler that runs the model, or None
            for no wrapping. This exists for HuggingFace ZeroGPU, where the GPU
            is attached per call and only inside a ``spaces.GPU``-decorated
            function -- so the Space passes that decorator in rather than this
            module importing ``spaces``, which exists only on a Space.
    Returns:
        A ``gradio.Blocks`` ready to ``.launch()``.
    Raises:
        ImportError: if gradio is not installed.
    """
    try:
        import gradio as gr
    except ImportError as exc:                          # pragma: no cover
        raise ImportError(
            "the demo needs gradio: pip install -e \".[demo]\"") from exc

    _scope_gradio_temp_dir()

    if gpu is None:
        def gpu(fn):
            return fn

    _SAMPLES = _sample_images()
    _resolve = _predictor_cache(predictor, device or predictor.device)
    _RESOLUTIONS = _available_resolutions(predictor)
    _DEFAULT_RES = 512 if 512 in _RESOLUTIONS else _RESOLUTIONS[0]

    with gr.Blocks(title="AFF-MAE", analytics_enabled=False) as demo:
        gr.Markdown(
            f"# AFF-MAE\n"
            f"Run EM trained AFFMAE at "
            f"**{', '.join(f'{r}' for r in _RESOLUTIONS[:-1])} or "
            f"{_RESOLUTIONS[-1]}px** on segmentation, reconstruction, and "
            f"token location generation."
            if len(_RESOLUTIONS) > 1 else
            f"# AFF-MAE\n"
            f"Run EM trained AFFMAE at **{_RESOLUTIONS[0]}px** on "
            f"segmentation, reconstruction, and token location generation.")

        with gr.Tab("Segmentation"):
            with gr.Row():
                with gr.Column():
                    seg_in = gr.Image(label="Input", type="numpy")
                    seg_res = gr.Radio(_RESOLUTIONS, value=_DEFAULT_RES,
                                       label="Resolution (px)")
                    seg_dpi = gr.Slider(72, 300, value=150, step=1, label="DPI")
                    seg_alpha = gr.Slider(0.0, 1.0, value=0.65, step=0.05,
                                          label="Overlay opacity")
                    seg_go = gr.Button("Segment", variant="primary")
                with gr.Column():
                    seg_out = gr.Image(label="Prediction")
                    seg_text = gr.Markdown()
            @gpu
            def on_segment(image, resolution, dpi, alpha):
                return _segment(_resolve, resolution, image, dpi, alpha)

            seg_go.click(on_segment,
                         [seg_in, seg_res, seg_dpi, seg_alpha],
                         [seg_out, seg_text])
            if _SAMPLES:
                gr.Examples(examples=_SAMPLES, inputs=[seg_in],
                            label="Example glom images", examples_per_page=4)

        with gr.Tab("Reconstruction"):
            with gr.Row():
                with gr.Column():
                    rec_in = gr.Image(label="Input", type="numpy")
                    rec_ratio = gr.Slider(0.0, 0.95, value=0.5, step=0.05,
                                          label="Mask ratio")
                    rec_go = gr.Button("Reconstruct", variant="primary")
                with gr.Column():
                    rec_out = gr.Image(label="Original / masked / reconstruction")
                    rec_text = gr.Markdown()
            @gpu
            def on_reconstruct(image, mask_ratio):
                return _reconstruct(mae or predictor, image, mask_ratio)

            rec_go.click(on_reconstruct, [rec_in, rec_ratio],
                         [rec_out, rec_text])
            if _SAMPLES:
                gr.Examples(examples=_SAMPLES, inputs=[rec_in],
                            label="Example glom images", examples_per_page=4)

        with gr.Tab("Token layout"):
            with gr.Row():
                with gr.Column():
                    tok_in = gr.Image(label="Input", type="numpy")
                    tok_res = gr.Radio(_RESOLUTIONS, value=_DEFAULT_RES,
                                       label="Resolution (px)")
                    tok_radius = gr.Slider(0, 12, value=0, step=1,
                                           label="Dot radius (0 = scale to image)")
                    tok_scale = gr.Slider(1.0, 4.0, value=1.0, step=0.5,
                                          label="Supersampling")
                    tok_go = gr.Button("Show tokens", variant="primary")
                with gr.Column():
                    tok_out = gr.Image(label="Token positions per stage")
                    tok_text = gr.Markdown()
            @gpu
            def on_tokens(image, resolution, radius, scale):
                return _tokens(_resolve, resolution, image, radius, scale)

            tok_go.click(on_tokens,
                         [tok_in, tok_res, tok_radius, tok_scale],
                         [tok_out, tok_text])
            if _SAMPLES:
                gr.Examples(examples=_SAMPLES, inputs=[tok_in],
                            label="Example glom images", examples_per_page=4)

        with gr.Tab("Batch"):
            with gr.Row():
                with gr.Column():
                    bat_in = gr.File(label="Images", file_count="multiple")
                    bat_dpi = gr.Slider(72, 300, value=150, step=1, label="DPI")
                    bat_alpha = gr.Slider(0.0, 1.0, value=0.65, step=0.05,
                                          label="Overlay opacity")
                    bat_go = gr.Button("Run batch", variant="primary")
                with gr.Column():
                    bat_out = gr.File(label="Masks + overlays (zip)")
                    bat_text = gr.Markdown()
            @gpu
            def on_batch(files, dpi, alpha):
                return _batch(predictor, files, dpi, alpha)

            bat_go.click(on_batch, [bat_in, bat_dpi, bat_alpha],
                         [bat_out, bat_text])
            if _SAMPLES:
                # File(file_count="multiple") takes a list, so one example is
                # all four crops rather than four one-image examples.
                gr.Examples(examples=[[[row[0] for row in _SAMPLES]]],
                            inputs=[bat_in],
                            label="Example glom images", examples_per_page=1)

    return demo


def _cached_pretrain_checkpoint():
    """The released MAE checkpoint if it is already on disk, else None.

    Deliberately does not download: a demo should not pull 832 MB uninvited.
    """
    from affmae.data.weights import EMWeights

    entry = EMWeights.AFFMAE_BASE_PRETRAIN_512
    return entry.download_path if os.path.isfile(entry.download_path) else None


def launch(checkpoint: str, config=None, device=None, share: bool = False,
           port: int = 7860, pretrain_checkpoint=None, pretrain_config=None):
    """Load a checkpoint and serve the demo.

    Args:
        checkpoint: path to a trained ``.pth``.
        config: training YAML, or None to find ``config.yaml`` beside it.
        device: cuda | cpu | mps, or None for the best available.
        share: create a public Gradio link.
        port: port to serve on.
        pretrain_checkpoint: MAE checkpoint for the reconstruction tab. None
            uses the released one if it is already cached locally, so the tab
            works out of the box without downloading anything.
        pretrain_config: config for that checkpoint.
    """
    _scope_gradio_temp_dir()

    predictor = AFFMAE.from_checkpoint(checkpoint, config=config,
                                                device=device)

    mae = None
    source = pretrain_checkpoint or _cached_pretrain_checkpoint()
    if source:
        try:
            mae = AFFMAE.from_checkpoint(source, config=pretrain_config,
                                         device=device)
            logger.info("reconstruction tab using %s", source)
        except Exception as error:
            logger.warning("could not load %s for the reconstruction tab: %s",
                           source, error)
    else:
        logger.info("no MAE checkpoint cached; the reconstruction tab will "
                    "explain that it needs one")

    logger.info("serving on http://127.0.0.1:%d (device=%s)", port,
                predictor.device)
    # show_error surfaces handler exceptions in the UI; without it a failure
    # is an opaque 500 in the browser and the traceback only reaches stderr.
    build_interface(predictor, mae, device=predictor.device).launch(
        server_port=port, share=share, show_error=True)
