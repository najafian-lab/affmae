"""Gradient correctness of the DDP path.

This machine has one GPU, so these tests verify the *math and wiring* rather
than multi-GPU execution:

* Gradient averaging is device- and backend-independent, so a 2-rank ``gloo``
  group on CPU proves the claim that matters — DDP's averaged gradient over two
  disjoint half-batches equals the single-process gradient over the full batch.
  If sharding, reduction, or loss normalization were wrong, this fails.
* Wrapping must not perturb a single-rank run: a 1-rank DDP group must produce
  bit-comparable gradients to the bare model, on GPU, for the real AFF and ViT
  models.

What this does **not** cover: NCCL, multi-device placement, SyncBatchNorm's
cross-device statistics, and throughput. See the DDP note in README.md.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from conftest import requires_cuda
from affmae.utils.dist import (
    get_world_size,
    has_unused_parameters,
    is_distributed,
    unwrap_model,
)


class ToyModel(nn.Module):
    """Deterministic, tiny, and CPU-friendly so gloo tests stay fast."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, x):
        return self.net(x)


class ToyWithUnusedHead(ToyModel):
    """Mimics an auxiliary head that a config leaves out of the loss."""

    def __init__(self):
        super().__init__()
        self.aux_head = nn.Linear(16, 4)  # never used in forward

    def forward(self, x):
        return self.net(x)


def _fixed_batch(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 8, generator=g)


def _single_process_grads(model_cls, batch):
    """Reference gradients: one process, whole batch, mean-reduced loss."""
    torch.manual_seed(1234)
    model = model_cls()
    model(batch).pow(2).mean().backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _ddp_worker(rank, world_size, batch, out_dir, tmpfile):
    """One rank: build the same model, take a shard, save averaged grads.

    Results go through the filesystem rather than a multiprocessing queue —
    queues created in the parent are not reliably usable from spawned children.
    """
    dist.init_process_group(backend="gloo", init_method=f"file://{tmpfile}",
                            rank=rank, world_size=world_size)
    try:
        torch.manual_seed(1234)  # identical init across ranks
        model = nn.parallel.DistributedDataParallel(ToyModel())

        shard = batch.chunk(world_size)[rank]
        # Mean over the shard; DDP averages grads across ranks, so equal-sized
        # shards reproduce the mean over the full batch.
        model(shard).pow(2).mean().backward()

        grads = {n: p.grad.clone()
                 for n, p in unwrap_model(model).named_parameters()
                 if p.grad is not None}
        if rank == 0:
            torch.save(grads, os.path.join(out_dir, "grads.pt"))
    finally:
        dist.destroy_process_group()


class TestGradientAveraging:
    """DDP over N shards must equal one process over the whole batch."""

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_matches_single_process(self, world_size, tmp_path):
        batch = _fixed_batch(n=8)
        want = _single_process_grads(ToyModel, batch)

        out_dir = tmp_path / f"ws{world_size}"
        out_dir.mkdir()
        mp.start_processes(
            _ddp_worker,
            args=(world_size, batch, str(out_dir), str(tmp_path / f"pg_{world_size}")),
            nprocs=world_size, start_method="spawn", join=True,
        )
        got = torch.load(out_dir / "grads.pt", weights_only=False)

        assert set(got) == set(want)
        for name in want:
            torch.testing.assert_close(
                got[name], want[name], rtol=1e-5, atol=1e-6,
                msg=lambda s, n=name: f"gradient mismatch for {n}: {s}")

    def test_uneven_shards_would_not_match(self, tmp_path):
        """Documents why epoch length must be pinned across ranks.

        Averaging per-rank means only equals the global mean when every rank
        contributes the same number of samples. This is the arithmetic reason
        ``build_pretrain_dataloader`` pins ``batches_per_rank``.
        """
        batch = _fixed_batch(n=8)
        full_mean = batch.mean(0)
        uneven = (batch[:6].mean(0) + batch[6:].mean(0)) / 2
        assert not torch.allclose(full_mean, uneven), (
            "if these matched, uneven shards would be harmless")


class TestUnusedParameterDetection:
    def test_reports_gradientless_parameter(self):
        model = ToyWithUnusedHead()
        loss = model(_fixed_batch()).pow(2).mean()
        loss.backward()

        unused = has_unused_parameters(model, loss)
        assert "aux_head.weight" in unused
        assert "aux_head.bias" in unused

    def test_reports_nothing_when_all_used(self):
        model = ToyModel()
        loss = model(_fixed_batch()).pow(2).mean()
        loss.backward()
        assert has_unused_parameters(model, loss) == []

    def test_works_through_a_wrapper(self):
        """Must read the unwrapped module, or every name gains a `module.`."""
        model = ToyWithUnusedHead()
        loss = model(_fixed_batch()).pow(2).mean()
        loss.backward()

        class Wrap(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.module = m

        unused = has_unused_parameters(Wrap(model), loss)
        assert all(not n.startswith("module.") for n in unused), unused


def _single_rank_worker(rank, world_size, cfg_path, out_dir, tmpfile):
    """Build a real model bare and under 1-rank DDP; compare gradients."""
    import sys
    sys.path.insert(0, os.getcwd())
    from affmae.config import load_config
    from affmae.models.registry import get_model_spec
    from affmae.utils.dist import wrap_for_distributed

    dist.init_process_group(backend="gloo", init_method=f"file://{tmpfile}",
                            rank=rank, world_size=world_size)
    try:
        cfg = load_config(cfg_path)
        # Do NOT shrink img_size: AFF's neighborhood sizes are tuned to the
        # configured token count, and a smaller grid leaves fewer tokens than
        # nbhd_size, which fails inside the encoder rather than in DDP.
        device = torch.device("cuda")
        spec = get_model_spec(cfg.model_type)

        torch.manual_seed(7)
        bare = spec.build_pretrain(cfg).to(device)
        torch.manual_seed(7)
        wrapped = wrap_for_distributed(
            spec.build_pretrain(cfg).to(device), device=device,
            find_unused_parameters=True)

        x = torch.randn(2, cfg.in_channels, cfg.img_size, cfg.img_size,
                        device=device, generator=torch.Generator(device=device).manual_seed(3))

        torch.manual_seed(11)
        bare(x)[0].backward()
        torch.manual_seed(11)
        wrapped(x)[0].backward()

        diffs = {}
        wrapped_params = dict(unwrap_model(wrapped).named_parameters())
        for name, p in bare.named_parameters():
            q = wrapped_params[name]
            if p.grad is None and q.grad is None:
                continue
            if (p.grad is None) != (q.grad is None):
                diffs[name] = "one side has no gradient"
                continue
            delta = (p.grad - q.grad).abs().max().item()
            if delta > 1e-4:
                diffs[name] = delta
        torch.save(diffs, os.path.join(out_dir, "diffs.pt"))
    finally:
        dist.destroy_process_group()


@requires_cuda
@pytest.mark.slow
@pytest.mark.parametrize("cfg_path", [
    "configs/aff_base_pretrain_0.4ds_0.5mask_last_local.yaml",  # affmae
    "configs/vit_base_pretrain_0.5mask.yaml",                          # vit
])
def test_single_rank_ddp_matches_bare_model(cfg_path, tmp_path):
    """Wrapping must not change gradients for the real AFF and ViT models.

    This is the closest check to real DDP available on one GPU: it exercises the
    actual models, the actual wrap path, and DDP's reducer, and would catch a
    parameter-ordering or bucketing problem. It cannot catch anything that only
    manifests with more than one device.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    mp.start_processes(
        _single_rank_worker,
        args=(1, cfg_path, str(out_dir), str(tmp_path / "pg1")),
        nprocs=1, start_method="spawn", join=True,
    )
    diffs = torch.load(out_dir / "diffs.pt", weights_only=False)
    assert diffs == {}, f"gradients diverged after DDP wrap: {diffs}"


class TestDataloaderShardingArithmetic:
    """The batch/epoch arithmetic that keeps ranks in lockstep."""

    @pytest.mark.parametrize("world_size,expected", [(1, 64), (2, 32), (8, 8)])
    def test_per_rank_batch_divides_global(self, world_size, expected):
        global_batch = 64
        assert max(1, global_batch // world_size) == expected

    def test_batches_per_rank_is_uniform(self):
        """Every rank must plan the same number of batches."""
        from affmae.data.pretrain_dataset import TOTAL_SAMPLES

        global_batch = 64
        counts = set()
        for world_size in (1, 2, 4, 8):
            per_rank = max(1, global_batch // world_size)
            counts.add(((TOTAL_SAMPLES // world_size) // per_rank))
        # Same effective batch regardless of world size, so the planned number
        # of optimizer steps per epoch should not drift with GPU count.
        assert len(counts) == 1, f"steps per epoch varies with world size: {counts}"
