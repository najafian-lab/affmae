"""Guards that keep distributed training addable without a codebase sweep.

None of this starts a process group. It wraps a model in a stand-in that has
the one property every distributed wrapper shares — a ``.module`` attribute —
and asserts the code paths that used to break still work.

What used to break:

* ``isinstance(model, AFFSegmentation)`` gated aux-loss meters and token
  visualization. Wrapped, it silently returns False and those quietly vanish.
* ``save_checkpoint`` called ``model.state_dict()`` on whatever it was handed,
  emitting ``module.``-prefixed keys no single-GPU run can load.
* ``build_optimizer_with_llrd`` reached straight into ``model.encoder.layers``.
"""

import pytest
import torch
import torch.nn as nn

from affmae.models.registry import get_model_spec
from affmae.utils.dist import (
    convert_sync_batchnorm,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    reduce_metric,
    unwrap_model,
)
from affmae.utils.misc import (
    AverageMeter,
    load_checkpoint,
    save_checkpoint,
    strip_module_prefix,
)


class WrapperStandIn(nn.Module):
    """Minimal stand-in for DDP/Accelerate: holds the model at ``.module``."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *a, **kw):
        return self.module(*a, **kw)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 4))
        self.decoder_pred_head = nn.Linear(4, 2)
        self.bn = nn.BatchNorm2d(3)

    def forward(self, x):
        return self.decoder_pred_head(self.encoder(x))


class TestUnwrapModel:
    def test_unwraps_single_layer(self):
        m = TinyModel()
        assert unwrap_model(WrapperStandIn(m)) is m

    def test_unwraps_nested_layers(self):
        m = TinyModel()
        assert unwrap_model(WrapperStandIn(WrapperStandIn(m))) is m

    def test_bare_model_passes_through(self):
        m = TinyModel()
        assert unwrap_model(m) is m

    def test_terminates_on_pathological_nesting(self):
        """Must not spin forever if something self-references."""
        m = TinyModel()
        wrapped = m
        for _ in range(50):
            wrapped = WrapperStandIn(wrapped)
        unwrap_model(wrapped)  # returns rather than hanging


class TestCheckpointRoundTrip:
    def test_saving_wrapped_yields_bare_keys(self, tmp_path):
        """The regression: wrapped saves must not emit `module.` keys."""
        model = TinyModel()
        wrapped = WrapperStandIn(model)
        opt = torch.optim.SGD(wrapped.parameters(), lr=0.1)
        path = tmp_path / "ckpt.pth"

        save_checkpoint(wrapped, opt, epoch=1, step=10, loss=0.5, path=str(path))
        keys = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]

        assert not any(k.startswith("module.") for k in keys), sorted(keys)[:3]

    def test_wrapped_save_loads_into_bare_model(self, tmp_path):
        model = TinyModel()
        with torch.no_grad():
            model.decoder_pred_head.bias.fill_(1.234)
        wrapped = WrapperStandIn(model)
        opt = torch.optim.SGD(wrapped.parameters(), lr=0.1)
        path = tmp_path / "ckpt.pth"
        save_checkpoint(wrapped, opt, 1, 10, 0.5, str(path))

        fresh = TinyModel()
        fresh_opt = torch.optim.SGD(fresh.parameters(), lr=0.1)
        epoch, step = load_checkpoint(fresh, fresh_opt, str(path))

        assert (epoch, step) == (1, 10)
        torch.testing.assert_close(fresh.decoder_pred_head.bias,
                                   model.decoder_pred_head.bias)

    def test_legacy_prefixed_checkpoint_still_loads(self, tmp_path):
        """Checkpoints written before the fix carry the prefix."""
        model = TinyModel()
        legacy = {f"module.{k}": v for k, v in model.state_dict().items()}
        path = tmp_path / "legacy.pth"
        torch.save({"epoch": 0, "step": 0, "loss": 0.0,
                    "model_state_dict": legacy,
                    "optimizer_state_dict": {}}, path)

        fresh = TinyModel()
        load_checkpoint(fresh, None, str(path))  # must not raise

    def test_strip_prefix_leaves_mixed_keys_alone(self):
        """Only strip when the prefix is uniform, else we'd corrupt names."""
        mixed = {"module.a": 1, "b": 2}
        assert strip_module_prefix(mixed) == mixed


class TestSpecHooksOnWrappedModels:
    """Spec fields replace the isinstance checks that wrapping broke."""

    @pytest.mark.parametrize("name", ["affmae", "vit"])
    def test_capabilities_are_data_not_isinstance(self, name):
        spec = get_model_spec(name)
        assert isinstance(spec.aux_names, tuple)
        assert isinstance(spec.supports_token_viz, bool)

    def test_layer_decay_plan_accepts_unwrapped_model(self):
        """Drivers must unwrap before calling; the plan may then reach in."""
        from affmae.models.specs.vit import layer_decay_plan

        class Cfg:
            vit_depth = 12

        model = TinyModel()
        plan = layer_decay_plan(unwrap_model(WrapperStandIn(model)), Cfg())
        assert plan.num_layers == 12
        assert plan.layer_id("encoder_blocks.3.attn.qkv.weight") == 3


class TestDistHelpersDegradeGracefully:
    """Every helper must be a no-op with no process group initialized."""

    def test_flags(self):
        assert is_distributed() is False
        assert is_main_process() is True
        assert get_rank() == 0
        assert get_world_size() == 1

    def test_reduce_metric_is_identity(self):
        assert reduce_metric(2.5) == pytest.approx(2.5)

    def test_sync_batchnorm_is_noop_when_not_distributed(self):
        model = TinyModel()
        out = convert_sync_batchnorm(model)
        assert isinstance(out.bn, nn.BatchNorm2d)
        assert not isinstance(out.bn, nn.SyncBatchNorm)

    def test_vit_segmentation_has_batchnorm_to_convert(self):
        """Pins why convert_sync_batchnorm exists at all.

        ViTSegmentationUperNet is the shipped ViT model and uses BatchNorm2d in
        its FPN/PPM blocks, whose running stats diverge per rank under DDP.
        """
        from affmae.models.vit_fpn_segmentation import ViTSegmentationUperNet

        model = ViTSegmentationUperNet(
            patch_size=16, img_size=224, in_chans=1, embed_dim=64,
            depth=2, num_heads=2, decoder_conv_dim=32, num_classes=3)
        n_bn = sum(1 for m in model.modules()
                   if isinstance(m, nn.modules.batchnorm._BatchNorm))
        assert n_bn > 0


class TestAverageMeter:
    def test_default_behaviour_unchanged(self):
        m = AverageMeter()
        m.update(1.0, n=2)
        m.update(3.0, n=2)
        assert m.avg == pytest.approx(2.0)

    def test_reduce_fn_is_applied_to_avg(self):
        """The seam DDP will use to average across ranks."""
        m = AverageMeter(reduce_fn=lambda v: v * 10)
        m.update(1.0, n=2)
        m.update(3.0, n=2)
        assert m.avg == pytest.approx(20.0)

    def test_reduce_metric_wires_in_cleanly(self):
        m = AverageMeter(reduce_fn=reduce_metric)
        m.update(4.0)
        assert m.avg == pytest.approx(4.0)
