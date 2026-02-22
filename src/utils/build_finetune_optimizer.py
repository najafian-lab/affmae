import torch
import logging

def build_optimizer_with_llrd(model, cfg, ld):
    layer_decay = ld  
    base_lr = cfg.learning_rate
    weight_decay = cfg.weight_decay
    
    if cfg.model_type == 'aff':
        aff_depths = [len(layer.blocks) for layer in model.encoder.layers]
        num_layers = sum(aff_depths)
        aff_offsets = [0]
        for d in aff_depths[:-1]:
            aff_offsets.append(aff_offsets[-1] + d)
    else:
        num_layers = getattr(cfg, 'vit_depth', 24)

    def get_layer_id_aff(name):
        if 'encoder.layers' in name:
            parts = name.split('.')
            stage_idx = int(parts[parts.index('layers') + 1])
            stage_offset = aff_offsets[stage_idx]
            
            if 'blocks' in name:
                block_idx = int(parts[parts.index('blocks') + 1])
                return stage_offset + block_idx
            elif 'downsample' in name or 'prob_net' in name:
                return stage_offset + aff_depths[stage_idx] - 1
                
        if 'encoder.norms' in name:
            parts = name.split('.')
            stage_idx = int(parts[parts.index('norms') + 1])
            return aff_offsets[stage_idx] + aff_depths[stage_idx] - 1

        return 0

    def get_layer_id_vit(name):
        if 'encoder_blocks' in name:
            try:
                return int(name.split('encoder_blocks.')[1].split('.')[0])
            except:
                return 0
        elif 'encoder_norm' in name:
            return num_layers - 1
        return 0

    param_groups = {} 

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim == 1 or name.endswith(".bias") or \
           "pos_embed" in name or "cls_token" in name or "masked_token" in name:
            this_wd = 0.0
        else:
            this_wd = weight_decay

        # Determine Scale
        if any(k in name for k in ['decoder', 'mask_token', 'head', 'masked_token']):
            scale = 1.0
            layer_id = "Decoder" # Just for tracking
        else:
            if cfg.model_type == 'aff':
                layer_id = get_layer_id_aff(name)
            else:
                layer_id = get_layer_id_vit(name)
            
            scale = layer_decay ** (num_layers - layer_id)

        # Store group
        group_key = (scale, this_wd)
        if group_key not in param_groups:
            param_groups[group_key] = []
        param_groups[group_key].append(param)

    optimizer_grouped_parameters = []
    for (scale, wd), params in param_groups.items():
        optimizer_grouped_parameters.append({
            "params": params,
            "weight_decay": wd,
            "lr": base_lr * scale,
            "initial_lr": base_lr * scale 
        })
    
    logging.info(f"\n{'='*50}")
    logging.info(f" LLRD CONFIGURATION CHECK (Decay={layer_decay})")
    logging.info(f"{'='*50}")
    logging.info(f"{'Layer / Group':<25} | {'LR Multiplier':<15} | {'Actual LR':<15}")
    logging.info(f"{'-'*60}")
    
    logged_scales = set()
    
    # Sort groups by LR (descending) so Decoder is first
    sorted_groups = sorted(optimizer_grouped_parameters, key=lambda x: x['lr'], reverse=True)
    
    for group in sorted_groups:
        scale = group['lr'] / base_lr
        
        if abs(scale - 1.0) < 1e-6:
            label = "Decoder / Head"
        elif abs(scale - (layer_decay ** 1)) < 1e-6:
            label = "Backbone TOP Layer"
        elif abs(scale - (layer_decay ** num_layers)) < 1e-6:
            label = "Backbone BOTTOM Layer"
        else:
            label = f"Backbone Intermediate"

        scale_key = round(scale, 5)
        if scale_key not in logged_scales:
            logging.info(f"{label:<25} | {scale:<15.4f} | {group['lr']:<15.2e}")
            logged_scales.add(scale_key)
            
    logging.info(f"{'='*50}\n")

    return torch.optim.AdamW(optimizer_grouped_parameters)