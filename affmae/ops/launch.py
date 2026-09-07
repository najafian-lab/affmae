"""Deriving Triton launch parameters from device properties. Again thanks to claude for most of this code
I dispatched various kernels on RTX 5090, H100, GH200, and A100 and tested performance of kernels with different block sizes/num_warps/num_stages
then also logged features of each card and saved the results. Then asked claude to write the code to derive the launch parameters from the device properties.
This isn't perfect, but it's a good starting point and whoever is reading this please feel free to improve it.

The kernels ship static launch parameters rather than ``@triton.autotune``,
because autotuning picks per-rank under DDP and races on the shared disk cache.
Static constants tuned on one card are wrong on the next one, so the constants
here are *computed* from properties torch already exposes.

    from affmae.ops.launch import DeviceProfile, warps_for_width

    profile = DeviceProfile.current()
    num_warps = warps_for_width(BLOCK_P, profile)

Two hard constraints apply to every rule:

* A Triton block size reaching ``tl.arange(0, BLOCK)`` must be a **power of
  two**; 48 fails to compile. :func:`largest_pow2_at_most` and
  :func:`clamp_pow2` exist so a rule cannot return an illegal value.
* Changing a block size changes which tiles are partial, so a kernel's masking
  has to be right for ``total_work % BLOCK != 0``. That is what
  ``TestPartialTileMasking`` covers.

Two cautions when sweeping candidates to find these values:

* An illegal memory access surfaces at the *next* ``torch.cuda.synchronize()``,
  which in a sweep is usually a different candidate -- so a fault gets
  attributed to the wrong parameters, and the poisoned context takes the rest of
  the sweep with it. Run each candidate in its own process before believing that
  a specific pair crashes. One such report did not reproduce across ten
  configurations here.
* The backward pass accumulates ``d_v`` through atomics, so it is not
  bit-reproducible run to run (~1e-6 absolute). "Bit-identical" is therefore the
  wrong acceptance test for backward launch parameters; compare against a
  same-configuration baseline instead.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

__all__ = [
    "DeviceProfile",
    "warps_for_width",
    "block_for_saturation",
    "largest_pow2_at_most",
    "clamp_pow2",
]


def largest_pow2_at_most(value: int) -> int:
    """Largest power of two not exceeding ``value``, at least 1.

    Args:
        value: any integer.
    Returns:
        A power of two in [1, value], or 1 when ``value`` < 1.
    """
    if value < 1:
        return 1
    return 1 << (int(value).bit_length() - 1)


def clamp_pow2(value: int, low: int, high: int) -> int:
    """Round ``value`` down to a power of two and clamp it to [low, high].

    Args:
        value: desired size.
        low: smallest acceptable size; rounded up to a power of two.
        high: largest acceptable size; rounded down to a power of two.
    Returns:
        A power of two within the range.
    Raises:
        ValueError: if the range contains no power of two.
    """
    lo = 1 << max(0, (int(low) - 1).bit_length())      # round low UP
    hi = largest_pow2_at_most(high)
    if lo > hi:
        raise ValueError(f"no power of two in [{low}, {high}].")
    return min(max(largest_pow2_at_most(value), lo), hi)


@dataclass(frozen=True)
class DeviceProfile:
    """The device properties the launch rules depend on.

    Args:
        name: device name, for logging and per-card overrides.
        warp_size: lanes per warp (32 on CUDA) or per wavefront (64 on ROCm).
        sm_count: streaming multiprocessors / compute units.
        max_threads_per_sm: resident-thread ceiling per SM.
        max_threads_per_block: thread ceiling for one program.
        shared_mem_per_block: opt-in shared memory per block, in bytes.
        regs_per_sm: registers per SM.
        capability: (major, minor) compute capability; (0, 0) off CUDA.
        l2_bytes: L2 cache size, in bytes.
        sm_clock_mhz: peak SM clock, in MHz.
        memory_clock_mhz: peak memory clock, in MHz.
        bandwidth_gb_s: theoretical peak memory bandwidth, or None if the
            properties needed to compute it are unavailable.
        is_hip: True on ROCm.
    """

    name: str
    warp_size: int
    sm_count: int
    max_threads_per_sm: int
    max_threads_per_block: int
    shared_mem_per_block: int
    regs_per_sm: int
    capability: tuple
    l2_bytes: int
    sm_clock_mhz: Optional[float]
    memory_clock_mhz: Optional[float]
    bandwidth_gb_s: Optional[float]
    is_hip: bool

    @classmethod
    def current(cls, device=None) -> "DeviceProfile":
        """Profile of the given (or current) CUDA/ROCm device.

        Args:
            device: torch device or index; None for the current device.
        Returns:
            A DeviceProfile. Falls back to a conservative CPU-ish profile when
            no accelerator is present, so callers need no special case.
        """
        import torch

        if not torch.cuda.is_available():
            return cls(name="cpu", warp_size=32, sm_count=1,
                       max_threads_per_sm=1024, max_threads_per_block=1024,
                       shared_mem_per_block=49152, regs_per_sm=65536,
                       capability=(0, 0), l2_bytes=0, sm_clock_mhz=None,
                       memory_clock_mhz=None, bandwidth_gb_s=None,
                       is_hip=False)
        index = torch.cuda.current_device() if device is None else device
        return _profile_for(int(getattr(index, "index", index)))

    @property
    def max_warps_per_block(self) -> int:
        """Warps that fit in one program, from the thread ceiling."""
        return max(1, self.max_threads_per_block // self.warp_size)

    @property
    def is_bandwidth_rich(self) -> Optional[bool]:
        """True on HBM-class parts (>= 1.5 TB/s), or None if unknown.

        A memory-bound kernel wants larger tiles on a bandwidth-poor card, to
        amortize each byte over more arithmetic; on an HBM part the same kernel
        is more often occupancy-limited instead.

        None rather than False when the bandwidth is unavailable: whether torch
        reports ``memory_bus_width`` depends on the torch version, not the card
        (2.9.1 does not report it on a 4090; 2.13 gives 1008 GB/s), so a False
        here would be a silent wrong answer rather than a missing one.
        """
        if self.bandwidth_gb_s is None:
            return None
        return self.bandwidth_gb_s >= 1500.0

    @property
    def bytes_per_sm_clock(self) -> Optional[float]:
        """Bandwidth per SM per clock -- how memory-starved each SM is.

        The ratio that decides whether a tile should grow: a low value means
        each SM is bandwidth-starved and wants larger tiles to amortize the
        bytes it does get, a high value means the SMs are fed and occupancy
        matters more.
        """
        if self.bandwidth_gb_s is None or not self.sm_clock_mhz:
            return None
        return (self.bandwidth_gb_s * 1e9) / (
            self.sm_count * self.sm_clock_mhz * 1e6)



def _warp_size_for(props) -> int:
    """Lanes per warp/wavefront, asked of the device rather than assumed.

    ``torch.cuda.get_device_properties`` does not expose ``warp_size`` on every
    ROCm build -- torch 2.4.0.dev+rocm6.0 on an MI300X does not -- so a plain
    ``getattr(p, "warp_size", 32)`` silently applied NVIDIA's 32 to a 64-lane
    CDNA wavefront. Every launch heuristic derived from it was then off by 2x,
    and it disagreed with ``deform_attn_triton._warps_for_width``, which already
    hardcodes 64 on ROCm.

    Order: what torch reports, then what Triton reports for the live target,
    then the architecture default -- 64 for CDNA/GCN, 32 for NVIDIA.

    Args:
        props: a ``torch.cuda.get_device_properties`` result.
    Returns:
        Lanes per warp.
    """
    import torch

    reported = getattr(props, "warp_size", None)
    if reported:
        return int(reported)

    try:
        import triton

        target = triton.runtime.driver.active.get_current_target()
        from_triton = getattr(target, "warp_size", None)
        if from_triton:
            return int(from_triton)
    except Exception:
        pass

    return 64 if torch.version.hip is not None else 32


@lru_cache(maxsize=8)
def _profile_for(index: int) -> DeviceProfile:
    """Build and cache a DeviceProfile for one device index."""
    import torch

    p = torch.cuda.get_device_properties(index)
    # bits -> bytes, kHz -> Hz, and x2 for double data rate.
    bandwidth = None
    bus = getattr(p, "memory_bus_width", 0)
    clock = getattr(p, "memory_clock_rate", 0)
    if bus and clock:
        bandwidth = (bus / 8) * (clock * 1e3) * 2 / 1e9

    return DeviceProfile(
        name=p.name,
        warp_size=_warp_size_for(p),
        sm_count=p.multi_processor_count,
        max_threads_per_sm=getattr(p, "max_threads_per_multi_processor", 2048),
        max_threads_per_block=getattr(p, "max_threads_per_block", 1024),
        shared_mem_per_block=getattr(p, "shared_memory_per_block_optin", 0)
        or getattr(p, "shared_memory_per_block", 49152),
        regs_per_sm=getattr(p, "regs_per_multiprocessor", 65536),
        capability=(p.major, p.minor),
        l2_bytes=getattr(p, "L2_cache_size", None) or None,
        sm_clock_mhz=(getattr(p, "clock_rate", 0) or None) and
        p.clock_rate / 1e3,
        memory_clock_mhz=(clock or None) and clock / 1e3,
        bandwidth_gb_s=bandwidth,
        is_hip=torch.version.hip is not None,
    )


def warps_for_width(width: int, profile: Optional[DeviceProfile] = None) -> int:
    """Warps needed to cover a ``width``-element vector, at least one.

    For a kernel whose values are all ``[BLOCK]``-shaped -- elementwise work, or
    a serial scan holding a few registers per lane -- a program wider than the
    data leaves lanes idle. ``BLOCK=32`` with Triton's default of 4 warps uses 8
    of each warp's 32 lanes.

    Measured on a GH200 for ``dense_top4_knn``, bit-identical, on token
    positions captured from real micrographs -- synthetic positions understate
    it by about half, because they collide on 36% of cells and change the
    insertion-sort branching:

        1024px, 16384 tokens   5.44 -> 1.57 ms    3.47x
        1024px,  6553 tokens   2.20 -> 0.65 ms    3.35x
        1024px,  2621 tokens   0.90 -> 0.29 ms    3.14x
         512px,  4096 tokens   0.44 -> 0.22 ms    1.95x
         512px,   655 tokens   0.13 -> 0.13 ms    1.00x

    The gain grows with the tile count, and vanishes on the small late-stage
    shapes, where the kernel cannot fill the device anyway.

    **Scope, measured rather than assumed.** Of the eight Triton kernels in this
    package, ``dense_top4_knn`` is the only one whose widest vector is narrower
    than a full warp complement, and it is the only one where reducing warps
    helps. ``num_warps=1`` wins nowhere else -- in the deformable backward, 4
    warps beat 1 by 1.10x at a fixed ``BLOCK_Q``. So apply this only to kernels
    that are elementwise or a serial scan over ``[BLOCK]`` vectors; a kernel
    whose inner loop blocks a second axis has real work to split across warps.

    Args:
        width: elements per program, e.g. BLOCK_P.
        profile: device profile; the current device when None.
    Returns:
        A warp count in [1, ``max_warps_per_block``].
    """
    profile = profile or DeviceProfile.current()
    needed = max(1, int(width) // profile.warp_size)
    return min(needed, profile.max_warps_per_block)


def block_for_saturation(total_work: int, profile: Optional[DeviceProfile] = None,
                         min_waves: int = 2, low: int = 16,
                         high: int = 256) -> int:
    """Largest legal block size that still fills the device.

    A kernel launched as ``cdiv(total_work, BLOCK)`` programs stops being
    throughput-bound once the grid drops below the SM count: at 512px the top-4
    KNN grid is ``4096 / 32 = 128`` tiles on a 132-SM GH200, under a single
    wave, which is why its speedup there (1.10x) is far below the 1024px case
    (1.85x). Bigger blocks amortize setup, so this takes the largest block that
    still leaves ``min_waves`` waves of work.

    Args:
        total_work: independent units, e.g. ``grid_h * grid_w`` per batch entry.
        profile: device profile; the current device when None.
        min_waves: waves of programs to keep resident, for latency hiding.
        low: smallest acceptable block.
        high: largest acceptable block.
    Returns:
        A power of two in [low, high].
    """
    profile = profile or DeviceProfile.current()
    target_programs = max(1, profile.sm_count * max(1, min_waves))
    return clamp_pow2(max(1, int(total_work) // target_programs), low, high)
