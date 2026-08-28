"""Model registry behaviour and per-spec completeness.

These tests are the contract that lets baseline models live on the
``rebuttals`` branch: a driver only ever touches ``ModelSpec`` fields, so a new
method plugs in by registering a spec rather than by editing every driver.
"""

import pathlib

import pytest

from affmae.models.registry import (
    ModelSpec,
    LayerDecayPlan,
    available_models,
    get_model_spec,
    register,
    resolve_alias,
)

RELEASE_MODELS = ["affmae", "vit"]


class TestLookup:
    def test_ships_expected_models(self):
        assert available_models() == sorted(RELEASE_MODELS)

    def test_legacy_alias_resolves(self):
        """Every shipped config still says `model_type: aff`."""
        assert resolve_alias("aff") == "affmae"
        assert get_model_spec("aff").name == "affmae"

    def test_canonical_name_is_not_an_alias(self):
        assert resolve_alias("affmae") == "affmae"

    def test_unknown_name_is_actionable(self):
        """The error should name what is available and where baselines went."""
        with pytest.raises(KeyError) as exc:
            get_model_spec("mixmae")
        msg = str(exc.value)
        assert "affmae" in msg
        assert "rebuttals" in msg

    def test_duplicate_registration_rejected(self):
        with pytest.raises(ValueError, match="already registered"):
            register(ModelSpec(name="affmae", build_segmentation=lambda cfg: None))

    def test_alias_collision_rejected(self):
        with pytest.raises(ValueError, match="collides"):
            register(ModelSpec(name="brand_new_model",
                               build_segmentation=lambda cfg: None,
                               aliases=("aff",)))


@pytest.mark.parametrize("name", RELEASE_MODELS)
class TestSpecCompleteness:
    """Every shipped spec must fill in what the drivers actually read."""

    def test_has_builders(self, name):
        spec = get_model_spec(name)
        assert callable(spec.build_segmentation)
        assert spec.build_pretrain is None or callable(spec.build_pretrain)

    def test_has_state_dict_adapter(self, name):
        assert callable(get_model_spec(name).adapt_state_dict)

    def test_has_layer_decay_plan(self, name):
        """A missing plan silently trains the whole trunk at the base LR."""
        assert callable(get_model_spec(name).layer_decay_plan)

    def test_aux_names_non_empty_and_primary_last(self, name):
        spec = get_model_spec(name)
        assert spec.aux_names, "segmentation models always emit a primary head"
        assert spec.aux_names[-1] == "res2"

    def test_llrd_in_open_unit_interval(self, name):
        assert 0.0 < get_model_spec(name).default_llrd <= 1.0

    def test_spec_is_immutable(self, name):
        """Specs are process-global; mutating one would leak across runs."""
        with pytest.raises(Exception):
            get_model_spec(name).default_llrd = 0.5


class TestPretrainAuxNames:
    def test_vit_reports_no_pretrain_aux(self):
        """VanillaViTMAE.forward returns `loss, []`.

        Reusing the segmentation aux names here would allocate meters the
        pretraining loop never fills.
        """
        assert get_model_spec("vit").pretrain_aux == ()

    def test_affmae_pretrain_aux_matches_segmentation(self):
        spec = get_model_spec("affmae")
        assert spec.pretrain_aux == spec.aux_names == ("res5", "res4", "res2")

    def test_defaults_to_aux_names_when_unset(self):
        spec = ModelSpec(name="_probe", build_segmentation=lambda cfg: None,
                         aux_names=("a", "b"))
        assert spec.pretrain_aux == ("a", "b")


class TestLayerDecayPlan:
    def test_plan_shape(self):
        plan = LayerDecayPlan(num_layers=3, layer_id=lambda n: 0)
        assert plan.num_layers == 3
        assert plan.layer_id("anything") == 0


class TestRendererNamesResolve:
    """``reconstruction_renderer`` holds a function *name*, not a reference.

    So renaming or moving a renderer breaks these specs silently -- no importer
    to fail, no attribute to miss, until a pretraining run reaches its first
    eval epoch and dies there.
    """

    def test_every_spec_renderer_exists(self):
        from affmae.models.registry import available_models, get_model_spec
        from affmae.viz import model_figures

        checked = 0
        for name in available_models():
            renderer = get_model_spec(name).reconstruction_renderer
            if renderer is None:
                continue
            assert hasattr(model_figures, renderer), (
                f"model {name!r} names renderer {renderer!r}, which is not in "
                f"affmae.viz.model_figures")
            checked += 1
        assert checked > 0, "no spec declared a reconstruction renderer"

    def test_token_viz_models_have_a_token_renderer(self):
        from affmae.models.registry import available_models, get_model_spec
        from affmae.viz import model_figures

        if any(get_model_spec(n).supports_token_viz for n in available_models()):
            assert hasattr(model_figures, "render_tokens")
            assert hasattr(model_figures, "run_pca_visualization")


class TestDecoderTransfersOnFinetune:
    """Finetuning must reuse the pretrained pixel decoder, not re-init it.

    ``RANDOM_INIT_DECODER_ON_FINETUNE`` was True, which dropped every key
    containing "decoder". The released checkpoints disprove that: measured
    between the released pretrain and 512px finetuned weights,
    ``cross_attention_decoder.input_proj.0.0.weight`` correlates +0.97, and each
    ``decoder_pred_head`` block correlates +0.37 with the pretrained head while
    the blocks correlate ~0.01 with each other -- the signature of the head
    expansion, whose copies drift apart during training. A re-initialized
    decoder cannot produce +0.97.

    True also made the expansion dead code, since ``decoder_pred_head`` contains
    "decoder" and was deleted twenty lines after being built.
    """

    REPO = pathlib.Path(__file__).resolve().parents[1]

    def _adapted(self):

        from affmae.config import load_config
        from affmae.models.registry import get_model_spec

        cfg = load_config(str(self.REPO / "configs" / "aff_base_finetune_512_fpw.yaml"))
        cfg.img_size = 256
        spec = get_model_spec(cfg.model_type)
        model = spec.build_segmentation(cfg)

        pretrain_cfg = load_config(
            str(self.REPO / "configs" / "aff_base_pretrain_0.4ds_0.5mask_last_local.yaml"))
        pretrain_cfg.img_size = 256
        mae = spec.build_pretrain(pretrain_cfg)

        state = {k: v.clone() for k, v in mae.state_dict().items()}
        return state, spec.adapt_state_dict(dict(state), model, cfg), cfg

    def test_decoder_weights_survive_adaptation(self):
        original, adapted, _ = self._adapted()
        decoder_keys = [k for k in original if "decoder" in k]
        assert decoder_keys, "the MAE has no decoder keys to transfer"
        missing = [k for k in decoder_keys if k not in adapted]
        assert missing == [], (
            f"adapt_state_dict dropped {len(missing)} decoder keys, e.g. "
            f"{missing[:3]}. Finetuning would start from a random decoder.")

    def test_the_head_expansion_is_not_discarded(self):
        """The expansion is pointless if the key it writes is deleted after."""
        original, adapted, cfg = self._adapted()
        key = "decoder_pred_head.weight"
        assert key in adapted, f"{key} was expanded and then dropped"
        assert adapted[key].shape[0] == original[key].shape[0] * cfg.num_classes

    def test_the_flag_is_off(self):
        from affmae.models.specs.affmae import RANDOM_INIT_DECODER_ON_FINETUNE

        assert RANDOM_INIT_DECODER_ON_FINETUNE is False, (
            "encoder-only transfer is a valid ablation, but it is not how the "
            "released checkpoints were trained; leaving it on silently changes "
            "what finetuning reproduces.")
