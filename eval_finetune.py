import os
import logging
from argparse import ArgumentParser
import torch

from src.config import load_config
from src.data.finetune_dataset import build_test_dataloader
from src.models.aff_segmentation import AFFSegmentation
from src.models.vit_segmentation import ViTSegmentation
from src.utils.misc import visualize_model_comparison_zoom, setup_logging, set_seed

def build_model(cfg):
    """Helper to build model based on config."""
    if cfg.model_type == "aff":
        model = AFFSegmentation(
            patch_size=cfg.patch_size, img_size=cfg.img_size, in_channels=cfg.in_channels,
            encoder_embed_dim=cfg.aff_embed_dims, encoder_depth=cfg.aff_depths,
            encoder_num_heads=cfg.aff_num_heads, encoder_nbhd_size=cfg.aff_nbhd_sizes,
            ds_rate=cfg.aff_ds_rates, cluster_size=cfg.aff_cluster_size,
            global_attention=cfg.aff_global_attention, decoder_embed_dim=cfg.decoder_embed_dim,
            decoder_num_heads=cfg.decoder_num_heads, merging_method=cfg.aff_merging_method,
            mlp_ratio=cfg.aff_mlp_ratio, alpha=cfg.aff_alpha, num_classes=cfg.num_classes
        )
    elif cfg.model_type == 'vit':
        model = ViTSegmentation(
            patch_size=cfg.patch_size, img_size=cfg.img_size, in_chans=cfg.in_channels,
            embed_dim=cfg.vit_embed_dim, depth=cfg.vit_depth, num_heads=cfg.vit_num_heads,
            decoder_embed_dim=cfg.decoder_embed_dim, decoder_depth=cfg.decoder_depth,
            decoder_num_heads=cfg.decoder_num_heads, mlp_ratio=4.0, num_classes=cfg.num_classes
        )
    
    # Load weights
    ckpt_path = os.path.join(cfg.output_dir, cfg.name, "last_model.pth")
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)["model_state_dict"]
        state_dict = {k: v for k, v in state_dict.items() if "pre_table" not in k}
        model.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded weights for {cfg.name}")
    else:
        logging.warning(f"Weights not found for {cfg.name} at {ckpt_path}")
        
    model.to(cfg.device)
    model.eval()
    return model

def get_predictions(model, dataloader, cfg):
    """Runs inference and returns stacked tensors."""
    results = []
    with torch.no_grad():
        for images, targets, paths in dataloader:
            images, targets = images.to(cfg.device), targets.to(cfg.device).long()
            pred_logits = model(images)
            
            final_logits = pred_logits[-1].cpu()
            
            batch_paths = paths[0] if isinstance(paths, (tuple, list)) else paths
            for b in range(images.size(0)):
                results.append({
                    'img': images.cpu()[b],
                    'tgt': targets.cpu()[b],
                    'pred': final_logits[b],
                    'path': batch_paths[b]
                })
                
    results.sort(key=lambda x: x['path'])
    return results

def main():
    parser = ArgumentParser()
    parser.add_argument('--config1', type=str, required=True, help="Config for Model 1 (e.g., ViT)")
    parser.add_argument('--config2', type=str, required=True, help="Config for Model 2 (e.g., AFF)")
    args = parser.parse_args()

    cfg1 = load_config(args.config1)
    cfg2 = load_config(args.config2)
    
    # Setup environment using cfg1 as base
    set_seed(7)
    exp_dir = os.path.join(cfg1.output_dir, "model_comparisons")
    os.makedirs(exp_dir, exist_ok=True)
    setup_logging(exp_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg1.device = device
    cfg2.device = device

    logging.info("Building models...")
    model1 = build_model(cfg1)
    model2 = build_model(cfg2)

    test_dl = build_test_dataloader(cfg1)

    logging.info("Running inference for Model 1...")
    res1 = get_predictions(model1, test_dl, cfg1)
    
    logging.info("Running inference for Model 2...")
    res2 = get_predictions(model2, test_dl, cfg2)

    # Stack results
    val_xs = torch.stack([x['img'] for x in res1], dim=0)
    val_ys = torch.stack([x['tgt'] for x in res1], dim=0)
    val_preds_1 = torch.stack([x['pred'] for x in res1], dim=0)
    val_preds_2 = torch.stack([x['pred'] for x in res2], dim=0)

    # Zoom boxes
    boxes = [
        (560, 220, 180, 200),
        (250, 320, 180, 180),
        (90, 290, 180, 180)
    ]

    logging.info("Generating comparison visualization...")
    visualize_model_comparison_zoom(
        x=val_xs, 
        y_true=val_ys, 
        pred_logits_1=val_preds_1, 
        pred_logits_2=val_preds_2, 
        model_1_name=cfg1.name, 
        model_2_name=cfg2.name, 
        num_classes=cfg1.num_classes, 
        indices=[41, 42, 43], 
        folder=exp_dir, 
        zoom_boxes=boxes,
        filename=f"compare_{cfg1.name}_vs_{cfg2.name}.png"
    )
    logging.info(f"Saved visualization to {exp_dir}")

if __name__ == "__main__":
    main()