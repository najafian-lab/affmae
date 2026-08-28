# CLUSTEN CUDA extensions (optional)

These are the original hand-written CUDA kernels for cluster and deformable
point attention. **The release does not need them.** Everything trains and
evaluates on the Triton kernels in `affmae/ops/`, which are the default
on every code path.

They are kept for one reason: they are the independent reference the Triton
kernels are tested against. `tests/test_forward.py`, `tests/test_backward.py`
and `tests/test_integration.py` run the same `ClusterAttention` module twice —
once with `backend="flash_nbhd_attn"` (Triton) and once with `backend="cuda"`
(these kernels) — and compare outputs and gradients. Without the extension
built those tests skip; the suite still passes, but you lose the differential
check.

`tests/test_kernels.py` does not need them: it validates Triton against the
pure-PyTorch references that ship in the kernel modules, so there is real
kernel coverage either way.

## Building

Requires a CUDA toolkit whose `nvcc` matches the CUDA version your PyTorch was
built against, plus a C++ compiler:

```bash
python -c "import torch; print(torch.version.cuda)"   # must match nvcc --version
cd affmae/ops/cuda_ext/src
python setup.py build_ext --inplace
```

Then verify:

```bash
python -c "from affmae.ops.cuda_ext.clusten import CLUSTENQKFunction; print('ok')"
pytest tests/ -q            # the 24 previously-skipped tests now run
```

## Kernels

| Source | Function |
|---|---|
| `clustenqk_cuda_kernel.cu` | cluster attention QK<sup>T</sup> |
| `clustenav_cuda_kernel.cu` | cluster attention attn·V |
| `clustenwf_cuda_kernel.cu` | weighted feature gather |
| `msdetrpc_cuda_kernel.cu` | multi-scale deformable point cross-attention |
| `weighted_gather_cuda_kernel.cu` | weighted gather used by the decoder |

`test_msdetrpc_kernel.py` and `test_wg_kernel.py` in this directory are
standalone smoke scripts for two of them; run them directly with `python`, not
under pytest.

## Using them at runtime

`backend="cuda"` on `ClusterAttention`, or `decoder_deform_backend="cuda"` on
the deformable attention, selects these kernels. Both raise a `RuntimeError`
pointing back here if the extension is not importable. There is no reason to
prefer them for training — they are slower than the Triton path and exist as an
oracle.
