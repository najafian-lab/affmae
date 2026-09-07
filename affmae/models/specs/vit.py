"""Registry spec for the ViT + UperNet reference model."""

import logging

from affmae.models.registry import LayerDecayPlan, ModelSpec, register
from affmae.layers.pos_embed import interpolate_pos_embed

logger = logging.getLogger(__name__)


def build_pretrain(cfg):
    """Build the vanilla ViT masked autoencoder for pretraining."""
    from affmae.models.vit_mae import VanillaViTMAE

    return VanillaViTMAE(
        patch_size=cfg.patch_size,
        img_size=cfg.img_size,
        in_chans=cfg.in_channels,
        mask_ratio=cfg.mask_ratio,
        embed_dim=cfg.vit_embed_dim,
        depth=cfg.vit_depth,
        num_heads=cfg.vit_num_heads,
        decoder_embed_dim=cfg.decoder_embed_dim,
        decoder_depth=cfg.decoder_depth,
        decoder_num_heads=cfg.decoder_num_heads,
        mlp_ratio=4.0,
    )


def build_segmentation(cfg):
    """Build ViT + UperNet for finetuning."""
    from affmae.models.vit_fpn_segmentation import ViTSegmentationUperNet

    return ViTSegmentationUperNet(
        patch_size=cfg.patch_size,
        img_size=cfg.img_size,
        in_chans=cfg.in_channels,
        embed_dim=cfg.vit_embed_dim,
        depth=cfg.vit_depth,
        num_heads=cfg.vit_num_heads,
        decoder_conv_dim=cfg.decoder_embed_dim,
        num_classes=cfg.num_classes,
    )


def adapt_state_dict(state_dict, model, cfg):
    """Resize the MAE positional embeddings to the finetuning resolution."""
    interpolate_pos_embed(state_dict, model, key="pos_embed")
    interpolate_pos_embed(state_dict, model, key="decoder_pos_embed")
    return state_dict


def layer_decay_plan(model, cfg):
    """Layer ids for the flat ViT trunk.

    Blocks are ``encoder_blocks.<i>``, so the index is the depth directly. The
    trailing encoder norm sits on top of the trunk.

    Args:
        model: nn.Module, **unwrapped** ViT model.
        cfg: Config, supplies ``vit_depth`` as the trunk length.
    Returns:
        LayerDecayPlan over the encoder trunk.
    """
    num_layers = getattr(cfg, "vit_depth", 24)

    def layer_id(name):
        if "encoder_blocks" in name:
            try:
                return int(name.split("encoder_blocks.")[1].split(".")[0])
            except (IndexError, ValueError):
                return 0
        if "encoder_norm" in name:
            return num_layers - 1
        return 0

    return LayerDecayPlan(num_layers=num_layers, layer_id=layer_id)


register(ModelSpec(
    name="vit",
    build_pretrain=build_pretrain,
    build_segmentation=build_segmentation,
    adapt_state_dict=adapt_state_dict,
    layer_decay_plan=layer_decay_plan,
    aux_names=("res2",),
    pretrain_aux_names=(),  # VanillaViTMAE.forward returns `loss, []`
    default_llrd=0.8,
    reconstruction_renderer="render_vit_reconstruction",
))
