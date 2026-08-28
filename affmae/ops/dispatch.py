"""Backend probing for the Triton kernels, general utils and probes across hardware
"""

import os

import torch

__all__ = [
    "HAS_TRITON",
    "is_hip",
    "is_rocm_build",
    "can_use_keops",
    "is_cuda",
    "has_triton_backend",
    "can_use_triton",
    "supports_host_descriptor",
    "custom_fwd",
    "custom_bwd",
    "autotune_extra_kwargs",
    "NUM_STAGES_OPTIONS",
    "warp_size",
    "clamp_num_warps",
    "clamp_num_stages",
    "_env_flag",
]

class _MissingTriton:
    """Stand-in for ``triton``/``triton.language`` when Triton is not installed.

    The kernel modules decorate with ``@triton.jit`` and annotate arguments with
    ``tl.constexpr``, both of which are evaluated when the module is imported.
    So the names have to exist on a torch-only install or ``import
    affmae.layers.attention`` fails outright, which would contradict the CPU and
    MPS fallbacks. Every attribute and call resolves back to this object, and
    only launching a kernel (``kernel[grid](...)``) raises.
    """

    __slots__ = ()

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        # Covers @triton.jit (bare) and @triton.autotune(...)/@triton.heuristics(...)
        # (called first, then applied), plus triton.Config(...) in module-level
        # config lists.
        return self

    def __getitem__(self, grid):
        return self._launch

    @staticmethod
    def _launch(*args, **kwargs):
        raise ImportError(
            "This operation needs a Triton kernel, but Triton is not "
            "installed. Install it with `pip install affmae[cuda]`, or run on "
            "a backend with a PyTorch fallback (the default backends pick the "
            "fallback automatically off CUDA)."
        )

    def __repr__(self):
        return "<triton unavailable>"


try:
    import triton
    import triton.language as tl  # noqa: F401  (re-exported for kernel modules)

    HAS_TRITON = True
except ImportError:  # pragma: no cover - environment dependent
    triton = _MissingTriton()
    tl = triton
    HAS_TRITON = False


def _backend():
    """Active Triton backend name, or None when unavailable.

    Returns:
        str or None: ``"cuda"``, ``"hip"``, or None if Triton is missing or has
        no live driver (CPU-only host, Apple silicon, no GPU visible).
    """
    if not HAS_TRITON:
        return None
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except Exception:
        # Probing the driver raises rather than returning a sentinel when no
        # GPU is present, so this must not be allowed to escape at import time.
        return None


def has_triton_backend() -> bool:
    """True if a Triton kernel can be launched on this host at all."""
    return _backend() is not None


def can_use_triton(*tensors) -> bool:
    """True if Triton can run on these specific tensors.

    A live backend is not enough: on a GPU host a model may still be running on
    CPU (common when testing locally), and launching a kernel on CPU pointers
    fails with "Pointer argument cannot be accessed from Triton".

    Args:
        *tensors: tensors the kernel would read.
    Returns:
        True only if a backend is live *and* every tensor is on an accelerator.
    """
    if not has_triton_backend():
        return False
    return all(t is not None and t.is_cuda for t in tensors)


def is_hip() -> bool:
    """True on AMD ROCm."""
    return _backend() == "hip"


def is_cuda() -> bool:
    """True on NVIDIA CUDA."""
    return _backend() == "cuda"


def is_rocm_build() -> bool:
    """True if torch itself was built against ROCm.

    Distinct from :func:`is_hip`, which asks Triton. This asks the torch build,
    which is what matters for a *tensor*: a ROCm tensor reports
    ``is_cuda == True`` and ``device.type == "cuda"``, so ``is_cuda`` cannot tell
    NVIDIA from AMD and is the wrong predicate for anything CUDA-specific.
    """
    return torch.version.hip is not None


def can_use_keops(*tensors) -> bool:
    """True if PyKeOps can run on these specific tensors.

    KeOps generates and compiles CUDA C++, or runs on CPU. It has **no ROCm and
    no MPS backend**, and a ROCm tensor looks like a CUDA tensor to
    ``is_cuda`` -- so a plain ``is_cuda`` check sends ROCm straight into a
    backend that does not exist.

    Args:
        *tensors: tensors the call would read.
    Returns:
        True only if PyKeOps is importable and every tensor is on CPU, or on an
        NVIDIA CUDA device.
    """
    import importlib.util

    if importlib.util.find_spec("pykeops") is None:
        return False
    if is_rocm_build():
        return False
    for tensor in tensors:
        if tensor is None:
            continue
        kind = tensor.device.type
        if kind == "cpu":
            continue
        if kind == "cuda":
            continue          # NVIDIA, since the ROCm build was excluded above
        return False           # mps, xpu, meta, anything else
    return True


def supports_host_descriptor() -> bool:
    """True on CUDA compute capability 9.0+ (Hopper and later)."""
    if not is_cuda():
        return False
    try:
        return torch.cuda.get_device_capability()[0] >= 9
    except Exception:
        return False


def _env_flag(name: str, default: str = "0") -> bool:
    """Read a boolean environment variable.

    Args:
        name: variable name.
        default: value used when unset.
    Returns:
        True for 1/true/yes/on, case-insensitive.
    """
    value = os.environ.get(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# ROCm autotuning is unstable above a single pipeline stage. Evaluated at import;
# clamp_num_stages() probes is_hip() live rather than trusting this.
NUM_STAGES_OPTIONS = [1] if is_hip() else [1, 2, 3, 4]


def warp_size() -> int:
    """Lanes per warp (NVIDIA) or wavefront (AMD).

    64 on AMD, 32 on NVIDIA. Not cosmetic: it sets how many lanes a requested
    ``num_warps`` actually reserves, so a tile smaller than that product leaves
    whole warps with no work.
    """
    return 64 if is_hip() else 32


def clamp_num_warps(num_warps: int, block_elements: int) -> int:
    """Clamp a config's ``num_warps`` to what its tile can actually fill.

    A no-op on CUDA, deliberately: the shipped configs were tuned on 32-lane
    warps, and changing them there would move performance without fixing
    anything. On ROCm a wavefront is 64 lanes, so a config asking for 2 warps
    over an 8x8 tile reserves 128 lanes for 64 elements, and the cross-warp
    reductions in the neighbourhood-attention kernel then return garbage.

    Measured on an MI300X (gfx942, warp_size 64), fp16, BLOCK_Q=8, BLOCK_MN=8:
    num_warps 1 gives finite output while 2 and 4 give NaN in 60-70% of
    elements, at both BLOCK_D 16 and 32 and at both num_stages 1 and 2 -- so the
    warp count is the cause, not the tile depth or the pipeline depth. Across a
    270-cell sweep this took ROCm from 56 non-finite cells to 0 and broke none.

    Args:
        num_warps: the config's requested warp count.
        block_elements: elements one program handles, i.e. ``BLOCK_Q * BLOCK_MN``
            for the tiled attention kernels.
    Returns:
        A warp count of at least 1, never more than the tile can fill.
    """
    requested = max(1, int(num_warps))
    if not is_hip():
        return requested
    lanes = warp_size()
    if int(block_elements) < lanes:
        return 1
    return max(1, min(requested, int(block_elements) // lanes))


def clamp_num_stages(stages: int) -> int:
    """Clamp a pipeline depth to what this backend tolerates.

    A no-op on CUDA. On ROCm it returns 1, because ROCm autotuning is unstable
    above a single stage -- the restriction :data:`NUM_STAGES_OPTIONS` encodes.

    That constant used to be consumed by exactly one of the six Triton modules
    (the v1 neighbourhood kernel), while the others hardcoded 2 to 4 stages, so
    the guard was reassurance rather than protection. Every stage choice now
    goes through here.

    Args:
        stages: the depth the kernel would like.
    Returns:
        A depth of at least 1, no greater than this backend allows.
    """
    # Probe live; NUM_STAGES_OPTIONS is evaluated at import, so reading it made
    # the ROCm branch untestable.
    return 1 if is_hip() else max(1, min(int(stages), max(NUM_STAGES_OPTIONS)))

def _amp_custom_decorators():
    """``custom_fwd``/``custom_bwd`` that work across torch versions.

    ``torch.amp.custom_fwd(device_type=...)`` is the 2.4-stable spelling. Older
    builds -- including 2.4.0.dev nightlies, which is what RunPod's ROCm image
    ships -- only have ``torch.cuda.amp.custom_fwd``, which takes no
    ``device_type``. Six kernel modules imported the new spelling at *module
    scope*, so on such a build importing any of them raised ImportError and the
    whole Triton path became unimportable -- defeating the ``_MissingTriton``
    design, which exists precisely so those modules always import.

    Returns:
        ``(custom_fwd, custom_bwd)``, each accepting ``device_type`` and
        ``cast_inputs`` and dropping what the installed torch cannot use.
    """
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "custom_fwd"):
        return amp.custom_fwd, amp.custom_bwd

    legacy = getattr(getattr(torch, "cuda", None), "amp", None)
    if legacy is None or not hasattr(legacy, "custom_fwd"):
        raise ImportError(
            "this torch exposes neither torch.amp.custom_fwd nor "
            "torch.cuda.amp.custom_fwd; affmae needs torch>=2.0.")

    def custom_fwd(fwd=None, *, device_type="cuda", cast_inputs=None):
        # The legacy decorator is CUDA-only and has no device_type parameter.
        decorator = legacy.custom_fwd(cast_inputs=cast_inputs)
        return decorator if fwd is None else decorator(fwd)

    def custom_bwd(bwd=None, *, device_type="cuda"):
        return legacy.custom_bwd if bwd is None else legacy.custom_bwd(bwd)

    return custom_fwd, custom_bwd


#: Re-exported so the kernel modules never import them from torch directly.
custom_fwd, custom_bwd = _amp_custom_decorators()

def autotune_extra_kwargs(cache_results: bool) -> dict:
    """Autotune kwargs the installed Triton actually accepts.

    ``triton.autotune(cache_results=...)`` is newer than ``triton>=3.0``, which
    is all pyproject requires. Triton 3.0.0 -- the version ``pytorch-triton-rocm``
    ships against a torch 2.4 ROCm build -- rejects the *keyword itself* with
    ``TypeError: autotune() got an unexpected keyword argument 'cache_results'``,
    even when the value is False. Passing it unconditionally made every
    autotuned neighbourhood-attention kernel unimportable there.

    Args:
        cache_results: whether to ask Triton to persist autotune results.
    Returns:
        ``{"cache_results": ...}`` when supported, else ``{}``.
    """
    if not HAS_TRITON:
        return {}
    try:
        import inspect

        params = inspect.signature(triton.autotune).parameters
    except (TypeError, ValueError):   # a stub, or a C-implemented callable
        return {}
    return {"cache_results": cache_results} if "cache_results" in params else {}
