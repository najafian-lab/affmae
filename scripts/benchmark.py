#!/usr/bin/env python
"""Compare AFF and ViT forward/loss/backward on real microscopy images.

    python scripts/benchmark.py --data-root data/fpwdata
    python scripts/benchmark.py --inputs real_batch.pt --task pretrain --resolution 1024

Uses FP16 and FlashAttention, with CUDA graphs on NVIDIA by default. Input
preparation, capture, gradient clearing, and optimizer/scaler steps are not timed.
"""

import argparse
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

from affmae.config import load_config  # noqa: E402
from affmae.data.preprocess import (  # noqa: E402
    load_image, multichannel_mask_to_labels, preprocess_image,
)
from affmae.data.weights import resolve_source  # noqa: E402
from affmae.eval.loader import legacy_checkpoint_compat  # noqa: E402
from affmae.models.specs import affmae, vit  # noqa: E402
from affmae.training.finetune_engine import build_loss_fn, _combine_losses  # noqa: E402
from affmae.utils.env import load_dotenv  # noqa: E402


def seed(value=90211):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def config_path(model, task, resolution):
    if task == "pretrain":
        name = ("aff_base_pretrain_0.4ds_0.5mask_last_local.yaml" if model == "aff"
                else "vit_base_pretrain_0.5mask.yaml")
    elif model == "aff":
        name = f"aff_base_finetune_{resolution}_fpw.yaml"
    else:
        name = "vit_base_finetune.yaml" if resolution == 512 else "vit_base_finetune_1024.yaml"
    return REPO / "configs" / name


def prepare_inputs(args):
    """Load real inputs once at 1024px, matching the published resize protocol."""
    if args.inputs:
        batch = torch.load(args.inputs, map_location="cpu", weights_only=True)
        provenance = {"cache": str(Path(args.inputs).resolve())}
    else:
        from PIL import Image

        cfg = load_config(config_path("aff", "finetune", 1024))
        root = Path(args.data_root or cfg.base_path)
        paths = sorted((root / "train" / "images").glob("*.tiff"))
        if len(paths) < args.batch_size:
            raise ValueError(f"Need {args.batch_size} real .tiff images under {root / 'train/images'}")
        paths = paths[:args.batch_size]
        images = torch.cat([preprocess_image(path, 1024) for path in paths])
        targets = []
        if args.task != "pretrain":
            for path in paths:
                mask = load_image(root / "train" / "masks" / path.name)
                if mask.ndim == 4 and mask.shape[-1] == 1:
                    mask = mask[..., 0]
                labels = multichannel_mask_to_labels(mask[cfg.indices]) if mask.ndim == 3 else mask
                labels = np.asarray(Image.fromarray(labels.astype("uint8")).resize(
                    (1024, 1024), Image.Resampling.NEAREST)).copy()
                targets.append(torch.from_numpy(labels).long())
        batch = {"images": images, "targets": torch.stack(targets) if targets else None}
        provenance = {"images": [str(p.resolve()) for p in paths], "data_root": str(root.resolve())}
    images, targets = batch["images"], batch.get("targets")
    if images.ndim != 4 or images.shape[1:] != (1, 1024, 1024) or len(images) < args.batch_size:
        raise ValueError("Images must contain at least batch-size real samples, shaped [B,1,1024,1024]")
    if not torch.isfinite(images).all():
        raise ValueError("Input images contain non-finite values")
    if args.task != "pretrain":
        if targets is None or targets.shape != (len(images), 1024, 1024):
            raise ValueError("Finetuning requires paired labels shaped [B,1024,1024]")
        if targets.is_floating_point() or targets.min() < 0 or targets.max() > 2:
            raise ValueError("The shipped FPW configurations require integer labels 0, 1, 2")
    return {"images": images[:args.batch_size],
            "targets": targets[:args.batch_size] if targets is not None else None}, provenance


def prepare_checkpoint(source, destination):
    """Resolve a released checkpoint and save only its model tensors for workers."""
    path, _ = resolve_source(source)
    with legacy_checkpoint_compat():
        state = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "model", "state_dict"):
        if key in state:
            state = state[key]
            break
    state = {key.removeprefix("module."): value for key, value in state.items()}
    torch.save(state, destination)
    return str(Path(path).resolve())


def step(model, images, targets, criterion, task):
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(images)
        loss = output[0] if task == "pretrain" else _combine_losses(output, targets, criterion)[1]
    loss.backward()
    return loss, output


def measure(model, images, targets, criterion, args):
    """Time complete microbatches, including forward-to-backward metadata."""
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup):
        model.zero_grad(set_to_none=True)
        step(model, images, targets, criterion, args.task)
    torch.cuda.synchronize()
    graph = None
    if not args.eager:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss, _ = step(model, images, targets, criterion, args.task)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
    wall, events = [], []
    for _ in range(args.iterations):
        if graph is None:
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        if graph is None:
            loss, _ = step(model, images, targets, criterion, args.task)
        else:
            graph.replay()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - start) * 1000)
        events.append(begin.elapsed_time(end))
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite loss after timing")
    return {"wall_ms": wall, "event_ms": events, "median_ms": statistics.median(wall),
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(), "loss": loss.item()}


def run_worker(args):
    """Run one model/round in its own process so allocator pools cannot carry over."""
    import timm
    import triton

    torch.set_num_threads(4)
    seed()
    resolution = args.resolution[0]
    cfg = load_config(config_path(args.worker, args.task, resolution))
    cfg.img_size, cfg.device = resolution, "cuda"
    spec = affmae if args.worker == "aff" else vit
    model = spec.build_pretrain(cfg) if args.task == "pretrain" else spec.build_segmentation(cfg)
    if args.worker == "aff":
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if args.task == "finetune":
            state = spec.adapt_state_dict(state, model, cfg)
        print("Checkpoint:", model.load_state_dict(state, strict=args.task == "pretrain"), flush=True)
    model = model.cuda().train()
    batch = torch.load(args.inputs, map_location="cpu", weights_only=True)
    images = batch["images"]
    targets = batch["targets"] if args.task == "finetune" else None
    if resolution != 1024:
        images = torch.nn.functional.interpolate(images, size=(resolution, resolution),
                                                 mode="bilinear", align_corners=False, antialias=True)
        if targets is not None:
            targets = torch.nn.functional.interpolate(targets[:, None].float(),
                size=(resolution, resolution), mode="nearest")[:, 0].long()
    images = images.cuda()
    targets = targets.cuda() if targets is not None else None
    criterion = build_loss_fn(cfg) if args.task == "finetune" else None
    report = {"model": args.worker, "task": args.task, "resolution": resolution,
              "batch_size": args.batch_size, "graph": not args.eager,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "triton": triton.__version__, "timm": timm.__version__,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "initialization": "released AFF pretraining weights" if args.worker == "aff" else "seeded ViT initialization"}
    # Fail if FlashAttention is unavailable; do not silently time math attention.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        seed()
        loss, output = step(model, images, targets, criterion, args.task)
        outputs = [output[0], *output[1]] if args.task == "pretrain" else output
        if not all(torch.isfinite(t).all() for t in outputs):
            raise FloatingPointError("Non-finite model output")
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError("Non-finite model gradient")
        del loss, output, outputs
        report["measurement"] = measure(model, images, targets, criterion, args)
    Path(args.result).write_text(json.dumps(report, indent=2))


def summarize(folder, tasks, resolutions, rounds):
    lines = ["| Task | Resolution | ViT ms | AFF ms | Speedup | Memory reduction |",
             "|---|---:|---:|---:|---:|---:|"]
    rows = []
    for task in tasks:
        for resolution in resolutions:
            results = {name: [json.loads((folder / f"{task}_{resolution}_{name}_{i}.json").read_text())
                              for i in range(rounds)] for name in ("vit", "aff")}
            med = lambda name, key: statistics.median(r["measurement"][key] for r in results[name])
            speedup = statistics.median(results["vit"][i]["measurement"]["median_ms"] /
                                        results["aff"][i]["measurement"]["median_ms"] for i in range(rounds))
            memory = med("vit", "peak_allocated") / med("aff", "peak_allocated")
            row = {"task": task, "resolution": resolution, "vit_ms": med("vit", "median_ms"),
                   "aff_ms": med("aff", "median_ms"), "speedup": speedup, "memory_reduction": memory,
                   "vit_peak_gib": med("vit", "peak_allocated") / 2**30,
                   "aff_peak_gib": med("aff", "peak_allocated") / 2**30}
            rows.append(row)
            lines.append(f"| {task} | {resolution} | {row['vit_ms']:.2f} | {row['aff_ms']:.2f} | {speedup:.2f}× | {memory:.2f}× |")
    (folder / "summary.json").write_text(json.dumps(rows, indent=2))
    table = "\n".join(lines) + "\n"
    (folder / "TABLE.md").write_text(table)
    print(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", help="FPW directory containing train/images and train/masks")
    parser.add_argument("--inputs", help="Instead load a real-data .pt cache with images and targets tensors")
    parser.add_argument("--checkpoint", default="AFFMAE_BASE_PRETRAIN_512", help="AFF pretraining path or released weight name")
    parser.add_argument("--task", choices=("all", "pretrain", "finetune"), default="all")
    parser.add_argument("--resolution", type=int, nargs="+", choices=(512, 1024), default=[512, 1024])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--eager", action="store_true", help="Disable CUDA graphs for both models (automatic on ROCm)")
    parser.add_argument("--output", default="output/benchmark", help="Directory for per-round JSON, logs, and the table")
    parser.add_argument("--worker", choices=("aff", "vit"), help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    args = parser.parse_args()
    load_dotenv()
    if min(args.batch_size, args.rounds, args.warmup, args.iterations) < 1:
        parser.error("batch-size, rounds, warmup, and iterations must be positive")
    if not torch.cuda.is_available():
        parser.error("A CUDA or ROCm GPU is required; install affmae[cuda,baselines,inference,train]")
    args.eager = args.eager or torch.version.hip is not None
    if args.worker:
        run_worker(args)
        return
    folder = Path(args.output).resolve()
    if folder.exists() and any(folder.iterdir()):
        parser.error("Output directory is not empty; choose a new --output to avoid mixing runs")
    folder.mkdir(parents=True, exist_ok=True)
    tasks = ("pretrain", "finetune") if args.task == "all" else (args.task,)
    with tempfile.TemporaryDirectory(prefix="affmae-benchmark-") as temporary:
        temporary = Path(temporary)
        batch, provenance = prepare_inputs(args)
        torch.save(batch, temporary / "inputs.pt")
        provenance["checkpoint"] = prepare_checkpoint(args.checkpoint, temporary / "weights.pt")
        (folder / "run.json").write_text(json.dumps({"arguments": vars(args), "data": provenance}, indent=2))
        for task in tasks:
            for resolution in args.resolution:
                for round_index in range(args.rounds):
                    for model in (("vit", "aff") if round_index % 2 == 0 else ("aff", "vit")):
                        stem = f"{task}_{resolution}_{model}_{round_index}"
                        command = [sys.executable, str(Path(__file__).resolve()), "--worker", model,
                                   "--task", task, "--resolution", str(resolution), "--batch-size", str(args.batch_size),
                                   "--inputs", str(temporary / "inputs.pt"), "--checkpoint", str(temporary / "weights.pt"),
                                   "--warmup", str(args.warmup), "--iterations", str(args.iterations),
                                   "--result", str(folder / f"{stem}.json")]
                        if args.eager:
                            command.append("--eager")
                        print(f"Running {stem}", flush=True)
                        with (folder / f"{stem}.log").open("w") as log:
                            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                        if result.returncode:
                            raise SystemExit(f"{stem} failed; see {folder / (stem + '.log')}. "
                                             "For OOM, retry both models with a smaller --batch-size and a new --output.")
    summarize(folder, tasks, args.resolution, args.rounds)


if __name__ == "__main__":
    main()
