"""The inference API: checkpoint plus image, no dataloader and no training config.

None of this was possible before. Every path went through a config and a
paired-mask dataloader, and ``EMTestDatasetMultiClass`` intersects image and mask
filenames — so an image with no ground-truth mask produced *zero* samples.
Preprocessing lived inside ``__getitem__``, so any inference path had to
reimplement it, and drift between the two degrades predictions in a way that
looks like a model problem.

The test that matters most is
:meth:`TestPreprocessing.test_matches_the_dataset_chain`: if the predictor
preprocesses differently from training, everything downstream is quietly wrong.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from affmae.config import load_config
from affmae.inference import AFFMAE, SegmentationResult
from affmae.models.registry import get_model_spec

REPO = Path(__file__).resolve().parents[1]
CONFIG = "configs/aff_base_finetune_512_fpw.yaml"
SMALL = 256   # a valid AFF geometry that is bearable on CPU


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """An untrained checkpoint plus its config, laid out as training would."""
    directory = tmp_path_factory.mktemp("ckpt")
    cfg = load_config(CONFIG)
    cfg.img_size = SMALL
    model = get_model_spec(cfg.model_type).build_segmentation(cfg)
    path = directory / "best_model.pth"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 0}, path)

    # The training scripts copy the config into the experiment directory, which
    # is what lets from_checkpoint work with no --config.
    text = (REPO / CONFIG).read_text().replace("img_size: 512",
                                               f"img_size: {SMALL}")
    (directory / "config.yaml").write_text(text)
    return path


@pytest.fixture(scope="module")
def predictor(checkpoint):
    return AFFMAE.from_checkpoint(str(checkpoint), device="cpu")


@pytest.fixture
def image():
    return (np.random.RandomState(0).rand(300, 220) * 255).astype(np.uint8)


class TestPreprocessing:
    def test_matches_the_dataset_chain(self, image):
        """Bit-identical to what training feeds the model.

        The dataset runs ToImage -> ConvertImageDtype(float32) -> Resize before
        normalizing; ConvertImageDtype is what maps uint8 0..255 into 0..1.
        Omitting it makes every value ~255x too large, which is exactly the kind
        of silent drift this test exists to prevent.
        """
        from torchvision.transforms import v2

        from affmae.data.finetune_dataset import IMAGE_MEAN, IMAGE_STD
        from affmae.data.preprocess import apply_clahe, preprocess_image

        mine = preprocess_image(image, img_size=64)

        array = apply_clahe(image)[..., None]
        expected = v2.Compose([
            v2.ToImage(),
            v2.ConvertImageDtype(torch.float32),
            v2.Resize((64, 64), antialias=True,
                      interpolation=v2.InterpolationMode.BILINEAR),
        ])(array).float().sub_(IMAGE_MEAN).div_(IMAGE_STD).unsqueeze(0)

        torch.testing.assert_close(mine, expected, rtol=0, atol=0)

    def test_output_shape_and_dtype(self, image):
        from affmae.data.preprocess import preprocess_image

        out = preprocess_image(image, img_size=128)
        assert out.shape == (1, 1, 128, 128)
        assert out.dtype == torch.float32

    def test_handles_high_bit_depth(self):
        """Micrographs are often 12- or 16-bit; CLAHE needs 8-bit input."""
        from affmae.data.preprocess import apply_clahe

        deep = (np.random.RandomState(1).rand(64, 64) * 65535).astype(np.uint16)
        assert apply_clahe(deep).dtype == np.uint8

    def test_handles_rgb_by_converting_to_grey(self):
        from affmae.data.preprocess import apply_clahe

        rgb = (np.random.RandomState(2).rand(32, 32, 3) * 255).astype(np.uint8)
        assert apply_clahe(rgb).ndim == 2

    def test_multichannel_mask_becomes_labels(self):
        from affmae.data.preprocess import multichannel_mask_to_labels

        mask = np.zeros((2, 8, 8))
        mask[0, :4] = 255
        mask[1, 4:] = 255
        labels = multichannel_mask_to_labels(mask)
        assert labels[0, 0] == 1 and labels[7, 0] == 2

    def test_missing_file_is_a_clear_error(self):
        from affmae.data.preprocess import load_image

        with pytest.raises(FileNotFoundError, match="image not found"):
            load_image("/nonexistent/image.tif")


class TestFromCheckpoint:
    def test_finds_the_config_beside_the_checkpoint(self, predictor):
        assert predictor.img_size == SMALL
        assert predictor.num_classes >= 2
        assert predictor.device.type == "cpu"

    def test_missing_checkpoint_is_a_clear_error(self):
        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            AFFMAE.from_checkpoint("/nonexistent/model.pth")

    def test_missing_config_names_what_to_pass(self, checkpoint, tmp_path):
        orphan = tmp_path / "alone.pth"
        shutil.copy(checkpoint, orphan)
        with pytest.raises(FileNotFoundError, match="no config given"):
            AFFMAE.from_checkpoint(str(orphan))

    def test_unavailable_device_downgrades(self, checkpoint):
        """Asking for a device you do not have must warn, not crash."""
        loaded = AFFMAE.from_checkpoint(str(checkpoint), device="mps")
        assert loaded.device.type in ("mps", "cuda", "cpu")


class TestPredict:
    def test_accepts_a_numpy_array(self, predictor, image):
        result = predictor.segment(image)
        assert isinstance(result, SegmentationResult)
        assert result.labels.shape == (SMALL, SMALL)
        assert result.logits.shape[0] == predictor.num_classes

    def test_accepts_a_file_path(self, predictor, image, tmp_path):
        from PIL import Image

        path = tmp_path / "img.png"
        Image.fromarray(image).save(path)
        result = predictor.segment(str(path))
        assert result.source.endswith("img.png")

    def test_accepts_a_tensor(self, predictor):
        result = predictor.segment(torch.rand(1, 64, 64) * 255)
        assert result.labels.shape == (SMALL, SMALL)

    def test_labels_are_in_range(self, predictor, image):
        result = predictor.segment(image)
        assert int(result.labels.min()) >= 0
        assert int(result.labels.max()) < predictor.num_classes

    def test_class_pixel_counts_sums_to_the_image(self, predictor, image):
        result = predictor.segment(image)
        assert sum(result.class_pixel_counts.values()) == SMALL * SMALL

    def test_batch_preserves_order_and_length(self, predictor, image):
        other = np.flipud(image).copy()
        results = predictor.segment_batch([image, other])
        assert len(results) == 2
        assert not torch.equal(results[0].labels, results[1].labels)

    def test_is_deterministic(self, predictor, image):
        first = predictor.segment(image)
        second = predictor.segment(image)
        torch.testing.assert_close(first.logits, second.logits, rtol=0, atol=0)

    def test_does_not_leave_the_model_in_train_mode(self, predictor, image):
        """A renderer used to call model.train() on the way out."""
        predictor.segment(image)
        assert not predictor.model.training


class TestSaveOverlay:
    def test_writes_a_figure_without_ground_truth(self, predictor, image, tmp_path):
        out = predictor.segment(image).save_overlay(str(tmp_path / "o.png"))
        assert Path(out).stat().st_size > 0

    def test_honours_a_viz_config(self, predictor, image, tmp_path):
        from affmae.viz import VizConfig

        result = predictor.segment(image)
        low = tmp_path / "low.png"
        high = tmp_path / "high.png"
        result.save_overlay(str(low), config=VizConfig(dpi=72))
        result.save_overlay(str(high), config=VizConfig(dpi=200))
        assert high.stat().st_size != low.stat().st_size


class TestTokenLayout:
    def test_returns_positions_per_stage(self, predictor, image):
        rendered, positions = predictor.token_layout(image)
        assert rendered.shape == (1, SMALL, SMALL)
        assert len(positions) >= 2
        assert all(p.shape[-1] == 2 for p in positions)

    def test_token_count_decreases_across_stages(self, predictor, image):
        """The point of adaptive downsampling."""
        _, positions = predictor.token_layout(image)
        counts = [p.shape[0] for p in positions]
        assert counts == sorted(counts, reverse=True), counts
        assert counts[0] > counts[-1]


class TestReconstruct:
    def test_declines_clearly_on_a_segmentation_checkpoint(self, predictor, image):
        """A finetuned model has replaced the MAE head; say so."""
        with pytest.raises(NotImplementedError, match="pretraining checkpoint"):
            predictor.reconstruct(image)


class TestDemoWiring:
    """The Gradio app is a thin shell over the predictor; check the seams."""

    def test_interface_builds(self, predictor):
        pytest.importorskip("gradio")
        from affmae.demo import build_interface

        assert build_interface(predictor) is not None

    def test_segmentation_tab_returns_an_image_and_a_table(self, predictor, image):
        pytest.importorskip("gradio")
        from affmae.demo import _predictor_cache, _segment

        resolve = _predictor_cache(predictor, predictor.device)
        path, text = _segment(resolve, predictor.img_size, image, 100, 0.6)
        assert path and Path(path).stat().st_size > 0
        assert "class" in text

    def test_segmentation_tab_reports_a_missing_resolution(self, predictor, image):
        """A resolution with no checkpoint on disk must explain itself, not raise.

        The picker only offers downloaded resolutions, but the API accepts any
        value, so the handler is the last line of defence.
        """
        pytest.importorskip("gradio")
        from affmae.demo import _predictor_cache, _segment

        resolve = _predictor_cache(predictor, predictor.device)
        path, text = _segment(resolve, 999, image, 100, 0.6)
        assert path is None
        assert "999" in text

    def test_tabs_handle_no_upload(self, predictor):
        pytest.importorskip("gradio")
        from affmae.demo import (_predictor_cache, _reconstruct, _segment,
                                 _tokens)

        resolve = _predictor_cache(predictor, predictor.device)
        for handler in (_segment, _tokens):
            _, message = handler(resolve, predictor.img_size, None, 100, 0.6)
            assert "Upload" in message
        _, message = _reconstruct(predictor, None, 0.5)
        assert "Upload" in message

    def test_the_gpu_hook_wraps_every_model_handler(self, predictor):
        """ZeroGPU needs each handler wrapped; a missed one would run on CPU.

        The Space passes spaces.GPU in here. If a tab were wired with a bare
        lambda again, that tab would silently lose its GPU on the Space.
        """
        pytest.importorskip("gradio")
        from affmae.demo import build_interface

        wrapped = []

        def fake_gpu(fn):
            wrapped.append(fn.__name__)
            return fn

        assert build_interface(predictor, gpu=fake_gpu) is not None
        assert sorted(wrapped) == ["on_batch", "on_reconstruct", "on_segment",
                                   "on_tokens"], wrapped

    def test_the_gpu_hook_is_optional(self, predictor):
        """Local use passes nothing and must behave exactly as before."""
        pytest.importorskip("gradio")
        from affmae.demo import build_interface

        assert build_interface(predictor) is not None

    def test_reconstruction_tab_degrades_gracefully(self, predictor, image):
        pytest.importorskip("gradio")
        from affmae.demo import _reconstruct

        path, message = _reconstruct(predictor, image, 0.5)
        assert path is None
        assert "pretraining" in message

    def test_batch_tab_returns_a_zip(self, predictor, image, tmp_path):
        pytest.importorskip("gradio")
        import zipfile

        from PIL import Image

        from affmae.demo import _batch

        paths = []
        for index in range(2):
            path = tmp_path / f"i{index}.png"
            Image.fromarray(image).save(path)
            paths.append(str(path))

        archive, message = _batch(predictor, paths, 100, 0.6)
        assert archive and zipfile.is_zipfile(archive)
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
        assert len(names) == 4, names          # mask + overlay per image
        assert "2 image(s)" in message

    def test_batch_tab_handles_no_files(self, predictor):
        pytest.importorskip("gradio")
        from affmae.demo import _batch

        archive, message = _batch(predictor, [], 100, 0.6)
        assert archive is None and "Upload" in message


class TestModes:
    """Inference skips backward-only work; training modes retain gradients."""

    def test_defaults_to_inference(self, predictor):
        from affmae.ops.policy import Mode

        assert predictor.mode is Mode.INFERENCE

    def test_inference_mode_freezes_parameters(self, checkpoint):
        model = AFFMAE.from_checkpoint(str(checkpoint), device="cpu",
                                       mode="inference")
        assert not any(p.requires_grad for p in model.model.parameters())

    def test_training_modes_keep_parameters_trainable(self, checkpoint):
        for mode in ("finetune", "pretrain"):
            model = AFFMAE.from_checkpoint(str(checkpoint), device="cpu",
                                           mode=mode)
            assert any(p.requires_grad for p in model.model.parameters()), mode

    def test_inference_mode_refuses_a_gradient_with_a_useful_message(self, checkpoint):
        from affmae.ops.policy import InferenceOnlyError

        model = AFFMAE.from_checkpoint(str(checkpoint), device="cpu",
                                       mode="inference")
        images = torch.randn(1, 1, SMALL, SMALL, requires_grad=True)
        with pytest.raises(InferenceOnlyError, match="mode='finetune'"):
            model(images)

    def test_finetune_mode_allows_a_backward_pass(self, checkpoint):
        model = AFFMAE.from_checkpoint(str(checkpoint), device="cpu",
                                       mode="finetune")
        images = torch.randn(1, 1, SMALL, SMALL)
        outputs = model(images)
        logits = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
        logits.sum().backward()
        assert any(p.grad is not None for p in model.model.parameters())

    def test_rejects_an_unknown_mode(self, checkpoint):
        with pytest.raises(ValueError, match="unknown mode"):
            AFFMAE.from_checkpoint(str(checkpoint), device="cpu", mode="fast")

    def test_mode_policy_does_not_select_component_backends(self):
        from affmae.config import load_config
        from affmae.ops.policy import KernelPolicy, Mode

        for mode in Mode:
            cfg = load_config(str(REPO / CONFIG))
            before = (cfg.decoder_deform_backend, cfg.cluster_attention_backend)
            KernelPolicy.for_mode(mode).apply_to_config(cfg)
            after = (cfg.decoder_deform_backend, cfg.cluster_attention_backend)
            assert before == after, f"{mode} changed the algorithm: {before} -> {after}"

    @pytest.mark.parametrize("alias", ["auto", "fused", "csr_cached",
                                        "csr_knn_cached"])
    def test_decoder_fused_aliases(self, checkpoint, alias):
        loaded = AFFMAE.from_checkpoint(
            str(checkpoint), device="cpu", decoder_deform_backend=alias)
        decoder = loaded.model.cross_attention_decoder
        assert decoder.deform_backend == "csr_knn_cached"

    def test_backend_arguments_override_the_yaml(self, checkpoint):
        loaded = AFFMAE.from_checkpoint(
            str(checkpoint), device="cpu",
            cluster_attention_backend="torch",
            decoder_deform_backend="unfused")
        first_attention = loaded.model.encoder.layers[0].blocks[0].attn
        assert first_attention.backend == "torch"
        assert loaded.model.cross_attention_decoder.deform_backend == "unfused"

    def test_rejects_unknown_component_backends(self, checkpoint):
        with pytest.raises(ValueError, match="neighbourhood-attention backend"):
            AFFMAE.from_checkpoint(
                str(checkpoint), device="cpu",
                cluster_attention_backend="mystery")
        with pytest.raises(ValueError, match="decoder deform backend"):
            AFFMAE.from_checkpoint(
                str(checkpoint), device="cpu",
                decoder_deform_backend="mystery")

    def test_inference_enables_the_free_knn_cache(self):
        """Caching the KNN table is bit-identical, so there is no reason not to."""
        from affmae.config import load_config
        from affmae.ops.policy import KernelPolicy, Mode

        cfg = load_config(str(REPO / CONFIG))
        KernelPolicy.for_mode(Mode.INFERENCE).apply_to_config(cfg)
        assert cfg.decoder_knn_cache is True

    def test_capabilities_are_introspectable(self, predictor):
        """Ask, rather than call and catch."""
        assert "segment" in predictor.capabilities
        assert predictor.can_segment is True
        assert isinstance(predictor.can_reconstruct, bool)

    def test_from_model_wraps_without_copying(self, predictor):
        wrapped = AFFMAE.from_model(predictor.model, img_size=SMALL,
                                    num_classes=predictor.num_classes,
                                    mode="inference")
        assert wrapped.model is predictor.model


PRETRAIN_CONFIG = "configs/aff_base_pretrain_0.4ds_0.5mask_last_local.yaml"


@pytest.fixture(scope="module")
def pretrain_checkpoint(tmp_path_factory):
    """An untrained *pretraining* checkpoint plus its config."""
    directory = tmp_path_factory.mktemp("mae_ckpt")
    cfg = load_config(PRETRAIN_CONFIG)
    cfg.img_size = SMALL
    model = get_model_spec(cfg.model_type).build_pretrain(cfg)
    path = directory / "ckpt.pth"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 0}, path)

    text = (REPO / PRETRAIN_CONFIG).read_text().replace(
        "img_size: 512", f"img_size: {SMALL}")
    (directory / "config.yaml").write_text(text)
    return path


class TestPretrainingCheckpoints:
    """``from_checkpoint`` used to hardcode ``build_segmentation``.

    So :meth:`AFFMAE.reconstruct` was unreachable from a checkpoint even though
    :attr:`capabilities` advertised it -- the only way in was ``from_model``.
    """

    def test_a_pretraining_checkpoint_can_reconstruct(self, pretrain_checkpoint):
        model = AFFMAE.from_checkpoint(str(pretrain_checkpoint), device="cpu")
        assert model.can_reconstruct
        assert not model.can_segment

    def test_task_can_be_forced(self, pretrain_checkpoint):
        model = AFFMAE.from_checkpoint(str(pretrain_checkpoint), device="cpu",
                                       task="pretrain")
        assert model.can_reconstruct

    def test_an_unknown_task_is_rejected(self, pretrain_checkpoint):
        with pytest.raises(ValueError, match="auto|segmentation|pretrain"):
            AFFMAE.from_checkpoint(str(pretrain_checkpoint), device="cpu",
                                   task="reconstruction")

    def test_reconstruction_keeps_the_visible_patches_exactly(
            self, pretrain_checkpoint, image):
        """The point of the composite: only the masked region is model output.

        A figure that quietly re-renders the *whole* image from the decoder would
        overstate reconstruction quality, so pin the visible half to the input.
        """
        from affmae.utils.dist import unwrap_model

        predictor = AFFMAE.from_checkpoint(str(pretrain_checkpoint), device="cpu")
        result = predictor.reconstruct(image, mask_ratio=0.5)
        model = unwrap_model(predictor.model)

        original = model.patchify(result.original[None])
        composite = model.patchify(result.reconstructions[-1][None])
        blanked = model.patchify(result.masked[None])
        visible = blanked.abs().sum(-1) > 0

        assert 0 < int(visible.sum()) < visible.numel()
        assert torch.equal(original[visible], composite[visible])

    def test_masked_token_layout_is_sparser_than_dense(self, pretrain_checkpoint,
                                                       image):
        """``mask_ratio`` shows the tokens the encoder actually received."""
        predictor = AFFMAE.from_checkpoint(str(pretrain_checkpoint), device="cpu")
        _, dense = predictor.token_layout(image)
        _, sparse = predictor.token_layout(image, mask_ratio=0.5)

        assert len(dense) == len(sparse)
        for stage, (d, s) in enumerate(zip(dense, sparse)):
            assert s.shape[0] < d.shape[0], f"stage {stage} was not sparsened"

    def test_the_mask_comes_from_the_model_not_a_copy(self, pretrain_checkpoint,
                                                     image):
        """Layout and reconstruction share ``mask_and_embed``.

        They were separate code paths, so a figure could show tokens from one
        mask beside a reconstruction from another.
        """
        from affmae.utils.dist import unwrap_model

        predictor = AFFMAE.from_checkpoint(str(pretrain_checkpoint), device="cpu")
        model = unwrap_model(predictor.model)
        assert hasattr(model, "mask_and_embed")
        tensor, _ = predictor._prepare(image)
        embedded = model.mask_and_embed(tensor)
        kept = embedded["ids_keep"].shape[1]
        hidden = embedded["ids_masked"].shape[1]
        assert kept + hidden == embedded["img_patches"].shape[1]

    def test_mask_ratio_is_rejected_for_a_segmentation_checkpoint(self, predictor,
                                                                 image):
        with pytest.raises(NotImplementedError, match="mask_ratio"):
            predictor.token_layout(image, mask_ratio=0.5)

    def test_reconstruction_tokens_sit_on_visible_patches(self, pretrain_checkpoint,
                                                          image):
        """The tokens and the masked image must come from the same mask.

        ``render_examples.py`` used to take the reconstruction from
        ``reconstruct()`` and the tokens from a second
        ``token_layout(mask_ratio=...)`` call. Each draws its own Perlin mask, so
        the figure showed one mask's tokens over another mask's image: measured,
        only ~50% of tokens landed on a patch the image showed as visible, which
        is chance. Taking both from one result makes it 100%.
        """
        from affmae.utils.dist import unwrap_model

        predictor = AFFMAE.from_checkpoint(str(pretrain_checkpoint), device="cpu")
        model = unwrap_model(predictor.model)
        result = predictor.reconstruct(image, mask_ratio=0.5)
        assert result.locations, "no per-stage locations to check"

        visible = model.patchify(result.masked[None])[0].abs().sum(-1) > 0
        grid = SMALL // model.encoder_patch_size

        for name, positions in zip(result.stage_names, result.locations):
            linear = (positions[:, 1].round().long().clamp(0, grid - 1) * grid
                      + positions[:, 0].round().long().clamp(0, grid - 1))
            on_visible = visible[linear].float().mean().item()
            assert on_visible == 1.0, (
                f"{name}: only {on_visible:.1%} of tokens are on visible "
                f"patches; the tokens and the masked image disagree.")

    def test_a_second_token_layout_call_draws_a_different_mask(self,
                                                               pretrain_checkpoint,
                                                               image):
        """Pin the footgun itself, so the docstring warning stays true."""
        predictor = AFFMAE.from_checkpoint(str(pretrain_checkpoint), device="cpu")
        _, first = predictor.token_layout(image, mask_ratio=0.5)
        _, second = predictor.token_layout(image, mask_ratio=0.5)
        assert not torch.equal(first[0], second[0]), (
            "two token_layout calls produced the same mask; if masking became "
            "deterministic, the warning on token_layout should be removed.")
