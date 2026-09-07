"""Registry spec for AFF-MAE, the method this repository implements."""

import logging

import torch

from affmae.models.registry import LayerDecayPlan, ModelSpec, register

logger = logging.getLogger(__name__)


# Setting it True creates the transfer learning ablation for encoder-only transfer for finetuning.
RANDOM_INIT_DECODER_ON_FINETUNE = False


def build_pretrain(cfg):
    """Build the AFF masked autoencoder for pretraining."""
    from affmae.models.aff_mae import AFFMaskedAutoEncoder

    return AFFMaskedAutoEncoder(
        patch_size=cfg.patch_size,
        img_size=cfg.img_size,
        in_channels=cfg.in_channels,
        mask_ratio=cfg.mask_ratio,
        encoder_embed_dim=cfg.aff_embed_dims,
        encoder_depth=cfg.aff_depths,
        encoder_num_heads=cfg.aff_num_heads,
        encoder_nbhd_size=cfg.aff_nbhd_sizes,
        ds_rate=cfg.aff_ds_rates,
        cluster_size=cfg.aff_cluster_size,
        mlp_ratio=cfg.aff_mlp_ratio,
        alpha=cfg.aff_alpha,
        global_attention=cfg.aff_global_attention,
        merging_method=cfg.aff_merging_method,
        decoder_embed_dim=cfg.decoder_embed_dim,
        decoder_num_heads=cfg.decoder_num_heads,
        decoder_deform_backend=getattr(cfg, "decoder_deform_backend", "auto"),
        decoder_knn_cache=getattr(cfg, "decoder_knn_cache", True),
        cluster_attention_backend=getattr(cfg, "cluster_attention_backend", "auto"),
    )


def build_segmentation(cfg):
    """Build the AFF segmentation model for finetuning and evaluation."""
    from affmae.models.aff_segmentation import AFFSegmentation

    return AFFSegmentation(
        patch_size=cfg.patch_size,
        img_size=cfg.img_size,
        in_channels=cfg.in_channels,
        encoder_embed_dim=cfg.aff_embed_dims,
        encoder_depth=cfg.aff_depths,
        encoder_num_heads=cfg.aff_num_heads,
        encoder_nbhd_size=cfg.aff_nbhd_sizes,
        ds_rate=cfg.aff_ds_rates,
        cluster_size=cfg.aff_cluster_size,
        global_attention=cfg.aff_global_attention,
        decoder_embed_dim=cfg.decoder_embed_dim,
        decoder_num_heads=cfg.decoder_num_heads,
        merging_method=cfg.aff_merging_method,
        mlp_ratio=cfg.aff_mlp_ratio,
        alpha=cfg.aff_alpha,
        num_classes=cfg.num_classes,
        decoder_deform_backend=getattr(cfg, "decoder_deform_backend", "auto"),
        decoder_knn_cache=getattr(cfg, "decoder_knn_cache", True),
        cluster_attention_backend=getattr(cfg, "cluster_attention_backend", "auto"),
    )


def adapt_state_dict(state_dict, model, cfg):
    """Expand the pretrained reconstruction head to ``num_classes`` logits.

    Pretraining regresses ``patch_size**2 * in_channels`` pixels; segmentation
    predicts ``patch_size**2 * num_classes``. Tiling the pretrained rows once per
    class and adding a little noise starts finetuning from the reconstruction
    head rather than from scratch, which measurably reduces early-epoch shock.

    Args:
        state_dict: dict, checkpoint weights (mutated in place).
        model: nn.Module, the **unwrapped** target segmentation model.
        cfg: Config, needs ``num_classes``.
    Returns:
        The adapted state dict.
    """
    weight_key, bias_key = "decoder_pred_head.weight", "decoder_pred_head.bias"

    if weight_key in state_dict and bias_key in state_dict:
        old_weight, old_bias = state_dict[weight_key], state_dict[bias_key]
        logger.info(
            "Expanding decoder head from %d to %d output units.",
            old_weight.shape[0], model.decoder_pred_head.out_features,
        )
        new_weight = old_weight.repeat(cfg.num_classes, 1)
        new_bias = old_bias.repeat(cfg.num_classes)
        state_dict[weight_key] = new_weight + torch.randn_like(new_weight) * 1e-4
        state_dict[bias_key] = new_bias + torch.randn_like(new_bias) * 1e-4
    else:
        logger.warning(
            "'%s' absent from checkpoint; skipping head expansion.", weight_key
        )

    if RANDOM_INIT_DECODER_ON_FINETUNE:
        state_dict = {k: v for k, v in state_dict.items() if "decoder" not in k}

    return state_dict


def layer_decay_plan(model, cfg):
    """Layer ids for AFF's staged encoder.

    Blocks live at ``encoder.layers.<stage>.blocks.<i>``, so a parameter's trunk
    depth is its stage offset plus its within-stage index. Downsampling and
    routing modules belong to the end of the stage they close; per-stage norms
    likewise.

    Args:
        model: nn.Module, **unwrapped** AFF model.
        cfg: Config, unused, present for signature uniformity.
    Returns:
        LayerDecayPlan over the encoder trunk.
    """
    depths = [len(layer.blocks) for layer in model.encoder.layers]
    num_layers = sum(depths)

    offsets = [0]
    for d in depths[:-1]:
        offsets.append(offsets[-1] + d)

    def layer_id(name):
        parts = name.split(".")
        if "encoder.layers" in name:
            stage_idx = int(parts[parts.index("layers") + 1])
            if "blocks" in name:
                return offsets[stage_idx] + int(parts[parts.index("blocks") + 1])
            if "downsample" in name or "prob_net" in name:
                return offsets[stage_idx] + depths[stage_idx] - 1
        if "encoder.norms" in name:
            stage_idx = int(parts[parts.index("norms") + 1])
            return offsets[stage_idx] + depths[stage_idx] - 1
        return 0

    return LayerDecayPlan(num_layers=num_layers, layer_id=layer_id)


register(ModelSpec(
    name="affmae",
    aliases=("aff",),  # every shipped config still says model_type: aff
    build_pretrain=build_pretrain,
    build_segmentation=build_segmentation,
    adapt_state_dict=adapt_state_dict,
    layer_decay_plan=layer_decay_plan,
    aux_names=("res5", "res4", "res2"),
    default_llrd=0.8,
    supports_token_viz=True,
    reconstruction_renderer="render_mae_reconstruction",
))
