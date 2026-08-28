"""The pretraining loop. """

import datetime
import logging
import os
import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from affmae.config import create_experiment_dir
from affmae.data.pretrain_dataset import (
    build_pretrain_dataloader,
    get_stable_visualization_batch,
)
from affmae.models.registry import get_model_spec
from affmae.training import tracking
from affmae.utils.dist import (
    autocast_context,
    cleanup_distributed,
    get_rank,
    get_world_size,
    has_unused_parameters,
    init_distributed,
    is_main_process,
    resolve_device,
    reduce_metric,
    unwrap_model,
    wrap_for_distributed,
)
from affmae.utils.misc import (
    AverageMeter,
    cosine_lr_schedule,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    setup_logging,
)

Config = Any
AMP = True

__all__ = ["run_pretrain", "train_epoch", "run_evaluation",
           "calculate_dataset_size"]


def calculate_dataset_size(dataloader: DataLoader) -> int:
    """
    iterates through the dataloader once to calculate the exact number of batches.
    this is needed as webdataset loaders don't have __len__
    """
    log_prefix = "dataset"
    logging.info(f"[{log_prefix}] Dry run started to calculate total batches/steps...")
    start = time.time()
    count = 0
    for _ in tqdm(dataloader, desc=f"Counting {log_prefix} batches"):
        count += 1
    
    duration = time.time() - start
    logging.info(f"[{log_prefix}] Count finished in {duration:.2f}s. Total batches: {count}")
    return count


def train_epoch(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                    loss_scaler: torch.amp.GradScaler, config: 'Config', epoch: int, global_step: int, 
                    total_max_steps: int, aux_names=(), params=None) -> tuple[float, int]:
    model.train()

    meters = {'loss': AverageMeter()}
    for name in aux_names:
        meters[name] = AverageMeter()

    log_start_time = time.time()
    num_accum = config.num_accum
    current_step = global_step

    optimizer.zero_grad()
    
    for batch_idx, (samples, _) in enumerate(dataloader):
        samples = samples.to(config.device)

        if AMP:
            with autocast_context(config.device):
                loss, aux_losses_list = model(samples)
                loss = loss / num_accum
            
            loss_scaler.scale(loss).backward()
        else:
            loss, aux_losses_list = model(samples)
            loss = loss / num_accum
            loss.backward()

        batch_size = samples.size(0)
        meters['loss'].update(loss.item() * num_accum, batch_size) 
        
        # update aux meters
        for i, val in enumerate(aux_losses_list):
            meters[aux_names[i]].update(val.item(), batch_size)
        
        # grad accum
        if (batch_idx + 1) % num_accum == 0:
            GRAD_SCALE = 5.0
            if AMP:
                loss_scaler.unscale_(optimizer)
            
            # Clip grads
            torch.nn.utils.clip_grad_norm_(params, GRAD_SCALE, error_if_nonfinite=False)

            if AMP:
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            
            current_step += 1
            
            # apply lr sched based on the training steps
            cosine_lr_schedule(optimizer, current_step, total_max_steps, config.base_lr, 1e-6, config.warmup_steps)

        # logging
        if (batch_idx + 1) % config.log_freq == 0:
            elapsed_time = time.time() - log_start_time
            samples_per_sec = (config.log_freq * samples.size(0)) / elapsed_time

            aux_str = " ".join([f"{k}:{m.avg:.4f}" for k, m in meters.items() if k != 'loss'])
            
            curr_lr = optimizer.param_groups[0]['lr']
            steps_per_epoch = total_max_steps // config.epochs

            logging.info(
                f"Epoch: {epoch} [{batch_idx+1:>4}/{steps_per_epoch*num_accum}] | "
                f"Loss: {meters['loss'].avg:.4f} ({aux_str}) | "
                f"LR: {curr_lr:.8f} | "
                f"Spd: {samples_per_sec:.1f}/s"
            )

            if tracking.is_active():
                log_dict = {
                    "batch_loss": meters['loss'].avg,
                    "lr": curr_lr,
                    "global_step": current_step,
                    "samples_per_sec": samples_per_sec
                }
                for name in aux_names:
                    log_dict[f"loss_{name}"] = meters[name].avg
                
                tracking.log(log_dict)

            log_start_time = time.time()

    return meters['loss'].avg, current_step


def run_evaluation(model: torch.nn.Module, vis_images: torch.Tensor,
                    epoch: int, output_dir: str, config: Config,
                    supports_token_viz: bool = False,
                    reconstruction_renderer: str = None):
    model.eval()
    logging.info(f"Running evaluation for epoch {epoch}...")

    eval_epoch_dir = os.path.join(output_dir, f'epoch_{epoch}')
    os.makedirs(eval_epoch_dir, exist_ok=True)

    # Imported here rather than at module scope so the training loop does not
    # pull matplotlib until it actually renders. Which renderer to use comes
    # from the model spec, by name.
    from affmae.viz import model_figures

    bare = unwrap_model(model)
    num_vis = min(5, vis_images.shape[0])

    if reconstruction_renderer is not None:
        recon_path = os.path.join(eval_epoch_dir, 'recon.png')
        getattr(model_figures, reconstruction_renderer)(
            bare, vis_images, num_vis, recon_path, seed=config.seed)

    # Token layout and per-stage PCA are an AFF capability.
    if supports_token_viz:
        token_path = os.path.join(eval_epoch_dir, 'token_loc.png')
        pca_path = os.path.join(eval_epoch_dir, 'pca_feats.png')
        model_figures.render_tokens(bare, vis_images, num_vis, token_path,
                                    seed=config.seed)
        model_figures.run_pca_visualization(bare, vis_images[:num_vis], pca_path,
                                            config.device)

    model.train()


def _probe_unused_parameters(model, config, device):
    """Run one tiny forward/backward to find parameters that get no gradient.

    DDP errors out on unused parameters unless ``find_unused_parameters=True``,
    and which parameters are used depends on the config (deep supervision in
    particular). Probing is cheaper and more honest than hardcoding the flag.

    The probe runs before wrapping, and gradients are zeroed afterwards so it
    cannot perturb training.

    Args:
        model: nn.Module, unwrapped pretraining model.
        config: Config, supplies input geometry.
        device: torch.device.
    Returns:
        bool, True if any trainable parameter received no gradient.
    """
    if get_world_size() == 1:
        return False
    try:
        model.train()
        probe = torch.randn(2, config.in_channels, config.img_size, config.img_size,
                            device=device)
        loss, _ = model(probe)
        loss.backward()
        unused = has_unused_parameters(model, loss)
    except Exception as exc:
        logging.warning("Unused-parameter probe failed (%s); assuming True, which "
                        "is correct but slower.", exc)
        return True
    finally:
        model.zero_grad(set_to_none=True)

    if unused:
        logging.info("DDP: %d parameter(s) receive no gradient "
                     "(find_unused_parameters=True): %s",
                     len(unused), unused[:5])
    return bool(unused)


def build_optimizer(model, config):
    """AdamW with weight decay off for 1-D parameters and biases.

    Args:
        model: the (possibly DDP-wrapped) model.
        config: Config; reads ``weight_decay``, ``base_lr``, ``beta1``, ``beta2``.
    Returns:
        ``(optimizer, params)`` where ``params`` is every trainable parameter, in
        the order the loop needs for gradient clipping.
    """
    decay_params = []
    no_decay_params = []
    for name, parameter in unwrap_model(model).named_parameters():
        if not parameter.requires_grad:
            continue
        if len(parameter.shape) == 1 or name.endswith('.bias'):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    optimizer = torch.optim.AdamW(
        [{'params': decay_params, 'weight_decay': config.weight_decay},
         {'params': no_decay_params, 'weight_decay': 0.0}],
        lr=config.base_lr,
        betas=(config.beta1, config.beta2),
    )
    return optimizer, decay_params + no_decay_params


def run_pretrain(config, config_path):
    """Run masked-autoencoder pretraining to completion.

    Args:
        config: a loaded Config. ``resume_path`` is honoured if set.
        config_path: path to the YAML, copied into the experiment directory.
    Returns:
        The experiment directory.
    """
    # torchrun sets RANK/WORLD_SIZE/LOCAL_RANK; absent them this is a no-op and
    # the loop runs exactly as it does single-process. The caller is responsible
    # for load_dotenv() before this point.
    distributed, local_rank = init_distributed()
    world_size = get_world_size()
    if distributed and torch.cuda.is_available():
        config.device = f"cuda:{local_rank}"

    # Only rank 0 creates the experiment directory and logs, or ranks race on
    # mkdir and every rank writes interleaved output to the same file.
    exp_dir = create_experiment_dir(config, config_path)
    if is_main_process():
        setup_logging(exp_dir)

    tracking.start_run(config,
                       resume="allow" if config.resume_path else None)

    # Offset by rank so each rank draws different masks, reproducibly.
    set_seed(config.seed + get_rank())
    # Honour the config when possible, downgrade with a warning when not.
    device = resolve_device(getattr(config, 'device', None))
    config.device = device 

    logging.info(f"Experiment started: {config.experiment_name}")
    logging.info(f"Using device: {device}")


    # we create a temporary loader just to count batches.
    logging.info("Pre-calculating dataset length...")
    count_loader, planned_batches = build_pretrain_dataloader(config, world_size)
    if world_size > 1:
        # Epoch length is pinned by with_epoch, so trust it rather than paying a
        # full pass over the shards; an uneven count here is what desyncs ranks.
        total_batches_per_epoch = planned_batches
    else:
        total_batches_per_epoch = calculate_dataset_size(count_loader)

    # calculate total training steps (batches / accumulation)
    steps_per_epoch = total_batches_per_epoch // config.num_accum
    total_max_steps = config.epochs * steps_per_epoch

    if world_size > 1:
        logging.info(f"DDP: world_size={world_size} "
                     f"global_batch={config.batch_size} "
                     f"per_rank_batch={max(1, config.batch_size // world_size)}")
    logging.info(f"Batches per epoch (per rank): {total_batches_per_epoch}")
    logging.info(f"Steps per epoch: {steps_per_epoch}")
    logging.info(f"Total steps (Training Length): {total_max_steps}")

    # main dataloader
    dataloader, _ = build_pretrain_dataloader(config, world_size)

    vis_images = get_stable_visualization_batch(config, device)

    logging.info(f"Initializing model of type: '{config.model_type}'")

    spec = get_model_spec(config.model_type)
    if spec.build_pretrain is None:
        raise ValueError(f"Model '{spec.name}' does not support pretraining.")
    model = spec.build_pretrain(config).to(device)

    # Determine whether any parameter goes gradient-less on this config before
    # wrapping: DDP raises on unused parameters unless told to look for them.
    # Configs that disable deep supervision are the case that matters.
    find_unused = _probe_unused_parameters(model, config, device)

    model = wrap_for_distributed(model, device=device,
                                 find_unused_parameters=find_unused)

    optimizer, params = build_optimizer(model, config)

    loss_scaler = torch.amp.GradScaler()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logging.info(f'Model initialized with {param_count:.2f}M trainable parameters.')



    start_epoch = 0
    global_step = 0

    resume_path = config.resume_path

    if resume_path and os.path.exists(resume_path):
        logging.info(f"Resuming from checkpoint: {resume_path}")
        start_epoch, _ = load_checkpoint(model, optimizer, resume_path)
        start_epoch += 1

        # we just start at the beginning of next epoch for the lr sched
        global_step = start_epoch * steps_per_epoch

        logging.info(f'Resumed successfully. Starting Epoch: {start_epoch}, Global Step: {global_step}')


    logging.info(f"Starting training loop from epoch {start_epoch} to {config.epochs}.")
    total_start_time = time.time()

    try:
        for epoch in range(start_epoch, config.epochs):
            epoch_start_time = time.time()

            train_loss, global_step = train_epoch(
                model=model, 
                dataloader=dataloader, 
                optimizer=optimizer, 
                loss_scaler=loss_scaler, 
                config=config, 
                epoch=epoch, 
                aux_names=spec.pretrain_aux, 
                global_step=global_step, 
                total_max_steps=total_max_steps, 
                params=params
            )

            # Each rank saw a disjoint shard; the epoch loss is the mean.
            train_loss = reduce_metric(train_loss, device)
            epoch_time = time.time() - epoch_start_time
            lr = optimizer.param_groups[0]['lr']

            logging.info("-" * 80)
            logging.info(f"Epoch {epoch} Done | Avg Loss: {train_loss:.4f} | Time: {epoch_time:.2f}s | LR: {lr:.8f}")
            logging.info("-" * 80)

            tracking.log({'epoch': epoch,
                          'epoch_avg_loss': train_loss,
                          'epoch_time_s': epoch_time})

            # Checkpointing, save on first epoch
            if (epoch + 1) % config.save_freq == 0 or epoch == config.epochs - 1 or epoch == 0:
                checkpoint_path = os.path.join(exp_dir, 'checkpoints', f'ckpt_epoch_{epoch}.pth') 
                if is_main_process():
                    save_checkpoint(model, optimizer, epoch, global_step, train_loss, checkpoint_path)

                # Visualization
                if is_main_process():
                    run_evaluation(model, vis_images, epoch, os.path.join(exp_dir, 'evaluations'), config,
                                   supports_token_viz=spec.supports_token_viz,
                                   reconstruction_renderer=spec.reconstruction_renderer)

    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise e
    finally:
        total_time_taken = time.time() - total_start_time
        logging.info(f"Training finished in {str(datetime.timedelta(seconds=int(total_time_taken)))}.")

        tracking.finish()

        # Barrier then destroy the group, so a rank that finishes early does not
        # tear down communication while others are still logging or saving.
        cleanup_distributed()

    return exp_dir
