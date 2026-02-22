import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm

# Import your modules
from src.config import load_config
from src.models.aff_mae import AFFMaskedAutoEncoder
from src.models.vit_mae import VanillaViTMAE
from src.data.pretrain_dataset import get_stable_visualization_batch
from src.utils.misc import load_checkpoint

def compute_effective_rank(features):
    """
    Args:
        features: [B, N, C] tensor
    Returns:
        avg_rank: scalar (average effective rank across batch)
        avg_max_rank: scalar (average theoretical max rank)
    """
    # center
    features = features - features.mean(dim=1, keepdim=True)
    
    # svd
    try:
        S = torch.linalg.svdvals(features) # [B, min(N, C)]
    except RuntimeError: 
        return 0.0, 0.0

    # P_i = \sigma_i / \sum \sigma_j
    S_sum = S.sum(dim=-1, keepdim=True) + 1e-10
    p = S / S_sum

    # shannon Entropy
    # H = - \sum p_i log(p_i)
    entropy = -torch.sum(p * torch.log(p + 1e-10), dim=-1)

    eff_rank = torch.exp(entropy)
    
    # theoretical max rank
    B, N, C = features.shape
    print(features.shape)
    max_rank = min(N, C)
    
    return eff_rank.mean().item(), max_rank


encoder_outputs = {}
def encoder_hook(module, input, output):
    """
    Hooks the return value of AFFEncoder.forward()
    Output is expected to be a dictionary {'res2': ..., 'res5': ...}
    """
    if isinstance(output, dict):
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                encoder_outputs[k] = v.detach()
    else:
        if isinstance(output, (list, tuple)):
            for i, feat in enumerate(output):
                encoder_outputs[f'block_{i}'] = feat.detach()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='/homes/iws/ziawang/Documents/lab/affmae_root/output/rank_analysis')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading {config.model_type}...")
    if config.model_type == 'aff':
        model = AFFMaskedAutoEncoder(
            # general model configs
            patch_size=config.patch_size,
            img_size=config.img_size,
            in_channels=config.in_channels,
            mask_ratio=config.mask_ratio,
            # aff encoder configs
            encoder_embed_dim=config.aff_embed_dims,
            encoder_depth=config.aff_depths,
            encoder_num_heads=config.aff_num_heads,
            encoder_nbhd_size=config.aff_nbhd_sizes,
            ds_rate=config.aff_ds_rates,
            cluster_size=config.aff_cluster_size,
            mlp_ratio=config.aff_mlp_ratio,
            alpha=config.aff_alpha,
            global_attention=config.aff_global_attention,
            merging_method=config.aff_merging_method,
            # decoder configs
            decoder_embed_dim=config.decoder_embed_dim,
            decoder_num_heads=config.decoder_num_heads,
        ).to(device)
    else:
        pass 

    load_checkpoint(model, None, args.resume)
    model.eval()

    handle = model.encoder.register_forward_hook(encoder_hook)

    images = get_stable_visualization_batch(config, device)
    
    with torch.no_grad():
        model(images)
    
    stages = []
    ranks = []
    normalized_ranks = []
    
    sorted_keys = sorted([k for k in encoder_outputs.keys() if 'res' in k and 'pos' not in k])
    
    for key in sorted_keys:
        feat = encoder_outputs[key] # [B, N, C]
        
        rank, max_rank = compute_effective_rank(feat)
        
        # normalized rank
        norm_rank = rank / max_rank
        
        stages.append(key)
        ranks.append(rank)
        normalized_ranks.append(norm_rank)
        
        print(f"{key}: Rank={rank:.2f} / Max={max_rank} (Norm={norm_rank:.2f})")

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.set_xlabel('Stage')
    ax1.set_ylabel('Effective Rank', color='tab:blue')
    ax1.plot(stages, ranks, marker='o', color='tab:blue', linewidth=2, label='Effective Rank')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx() 
    ax2.set_ylabel('Normalized Rank (Rank / Min(N,C))', color='tab:orange')
    ax2.plot(stages, normalized_ranks, marker='s', linestyle='--', color='tab:orange', label='Normalized Rank')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylim(0, 1.05)

    plt.title(f"Encoder Feature Rank: {config.model_type}\n(Collapse = Low Normalized Rank)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'feature_collapse.png'), dpi=300)
    print(f"Plot saved to {args.output_dir}/feature_collapse.png")
    
    handle.remove()

if __name__ == '__main__':
    main()