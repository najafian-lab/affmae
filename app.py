"""AFF-MAE on HuggingFace ZeroGPU.

ZeroGPU attaches a GPU per request rather than for the life of the Space, and
only inside a ``spaces.GPU``-decorated function. ``affmae.demo.build_interface``
takes that decorator as its ``gpu`` argument, so the library never imports
``spaces`` -- which exists only on a Space.

Checkpoints come from a HuggingFace model repo, placed at the paths the
EMWeights registry already looks in. ``download_file`` returns early when the
file is present, so nothing here reaches Google Drive.

Runs locally too: without ``spaces`` installed the decorator is the identity, so
``python app.py`` is the same app on a normal GPU.
"""

import logging
import os

# Before torch: ZeroGPU patches CUDA initialisation, and importing torch first
# means those patches land too late.
try:
    import spaces
except ImportError:                     # running off-Space
    spaces = None

#: True only on ZeroGPU hardware. The `spaces` package being importable is not
#: enough -- it is in requirements.txt and so present on a CPU Space too, where
#: decorating a handler with spaces.GPU and forcing device="cuda" would fail at
#: startup. HuggingFace sets SPACES_ZERO_GPU only when a GPU is actually
#: attachable.
HAS_ZEROGPU = spaces is not None and bool(os.environ.get("SPACES_ZERO_GPU"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("affmae.space")

#: Where the checkpoints live. A model repo, not this Space, so the Space stays
#: small and the weights are versioned on their own.
WEIGHTS_REPO = os.environ.get("AFFMAE_WEIGHTS_REPO", "smerkd/affmae")

#: Resolutions to offer. Each finetuned checkpoint is ~300 MB, fetched at
#: startup, and a finetuned model is resolution-specific -- the token count
#: scales with img_size/patch_size, so 512px weights are not the same model at
#: 1024. They load lazily, one per resolution the visitor actually picks. Set
#: AFFMAE_SPACE_RESOLUTIONS="512" to trade the choice for a faster cold start.
RESOLUTIONS = [int(part) for part in
               os.environ.get("AFFMAE_SPACE_RESOLUTIONS",
                              "512,768,1024").split(",")
               if part.strip()]

#: Seconds of GPU per request. The first call on a fresh worker also pays the
#: Triton JIT compile (~11 s with the static kernel config), so this is well
#: above the steady-state cost of a single 1024px forward.
GPU_DURATION = int(os.environ.get("AFFMAE_SPACE_GPU_SECONDS", "120"))


def stage_checkpoint(entry) -> bool:
    """Put one registry checkpoint where EMWeights expects to find it.

    Args:
        entry: an EMWeights member.
    Returns:
        True if the file is now in place, False if it could not be fetched --
        the demo then simply does not offer that resolution.
    """
    from huggingface_hub import hf_hub_download

    target = entry.download_path
    if os.path.exists(target):
        return True
    try:
        cached = hf_hub_download(repo_id=WEIGHTS_REPO, filename=entry.filename)
    except Exception as error:
        logger.warning("could not fetch %s from %s: %s",
                       entry.filename, WEIGHTS_REPO, error)
        return False
    os.makedirs(os.path.dirname(target), exist_ok=True)
    # Symlink rather than copy: the hub cache already holds the bytes, and a
    # Space without persistent storage has limited disk.
    os.symlink(cached, target)
    logger.info("staged %s", entry.filename)
    return True


def main() -> None:
    # Keep the checkpoint tree inside the app directory, so download_path()
    # resolves somewhere writable regardless of how the app is launched. Set
    # here rather than at module scope: importing this module should have no
    # side effects, or a test that imports it changes where every other test
    # thinks the weights live.
    os.environ.setdefault(
        "CHECKPOINT_ROOT",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights"))

    from affmae import AFFMAE
    from affmae.data.weights import EMWeights
    from affmae.demo import build_interface

    finetuned = {512: EMWeights.AFFMAE_BASE_FT_512,
                 768: EMWeights.AFFMAE_BASE_FT_768,
                 1024: EMWeights.AFFMAE_BASE_FT_1024}

    staged = [res for res in RESOLUTIONS
              if res in finetuned and stage_checkpoint(finetuned[res])]
    if not staged:
        raise SystemExit(
            f"no finetuned checkpoint could be staged from {WEIGHTS_REPO}. "
            f"Check the repo exists and holds the .pth files named in "
            f"affmae.data.weights.")

    # The reconstruction tab needs a pretraining checkpoint: a finetuned model
    # has replaced the MAE head, so without this it can only explain itself.
    has_mae = stage_checkpoint(EMWeights.AFFMAE_BASE_PRETRAIN_512)

    # None, not "cuda": AFFMAE auto-detects, and on ZeroGPU the spaces package
    # has already patched torch.cuda so the probe reports a device. Hardcoding
    # cuda would instead hard-fail on any non-GPU Space.
    device = None
    predictor = AFFMAE.from_checkpoint(finetuned[staged[0]], device=device)
    mae = (AFFMAE.from_checkpoint(EMWeights.AFFMAE_BASE_PRETRAIN_512,
                                  device=device) if has_mae else None)

    gpu = spaces.GPU(duration=GPU_DURATION) if HAS_ZEROGPU else None
    logger.info("serving %s at %s (zerogpu=%s, device=%s)", WEIGHTS_REPO,
                staged, HAS_ZEROGPU, predictor.device)

    demo = build_interface(predictor, mae, device=device, gpu=gpu)
    # queue() so concurrent visitors are serialised onto the one GPU slot
    # instead of racing and timing out.
    demo.queue(max_size=20).launch(show_error=True)


if __name__ == "__main__":
    main()
