"""KeOps nearest-neighbour backend for token positions. """

import logging
import os

import torch

logger = logging.getLogger(__name__)


# PyKeOps is an optional dependency: it is only needed by knn_keops(), which is
# the fallback path for large k. Importing it at module scope made every model
# import require a working KeOps install (and JIT compiler), so it is resolved
# lazily on first use instead.
_KEOPS_READY = False


def have_keops():
    """True if PyKeOps is importable.

    PyKeOps is an optional CPU/CUDA accelerator. Missing or unsupported installs
    are handled by the exact PyTorch fallback.
    """
    try:
        import pykeops  # noqa: F401
        return True
    except ImportError:
        return False


def _ensure_keops():
    """Import and initialize PyKeOps on first use.

    Raises:
        ImportError: if PyKeOps is not installed.
    """
    global _KEOPS_READY
    if _KEOPS_READY:
        return
    try:
        import pykeops
    except ImportError as exc:
        raise ImportError(
            "PyKeOps is not installed (pip install 'affmae[keops]')."
        ) from exc
    try:
        build_folder = pykeops.get_build_folder()
        os.makedirs(build_folder, exist_ok=True)
        os.environ.setdefault("KEOPS_VERBOSE", "0")
        pykeops.set_verbose(False)
        logger.debug("KeOps build folder: %s", build_folder)
    except Exception as exc:  # diagnostics only; KeOps can still work
        logger.warning("KeOps init diagnostics failed: %s", exc)
    _KEOPS_READY = True
# from affmae.ops.knn_triton import knn as triton_knn


# @torch.compile()
def knn_keops(query, database, k, return_dist=False):
    """
    Compute k-nearest neighbors using the Keops library
    Backward pass turned off; Keops does not provide backward pass for distance
    Args:
        query - b x n_ x c, the position of tokens looking for knn
        database - b x n x c, the candidate tokens for knn
        k - int, the nunmber of neighbors to be found
        return_dist - bool, whether to return distance to the neighbors
    Returns:
        nn_dix - b x n x k, the indices of the knn
        nn_dist - b x n x k, if return_dist, the distance to the knn
    """
    # KeOps has no ROCm and no MPS backend, and a ROCm tensor reports
    # is_cuda True (I think? I didn't check) -- so the device has to be checked properly, not with
    # is_cuda. This function is called from the AFF encoder's clustering on
    # every forward, so getting it wrong breaks the model outright on those
    # backends rather than just slowing it down.
    from affmae.ops.dispatch import can_use_keops, can_use_triton

    if not can_use_keops(query, database):
        if can_use_triton(query, database) and k <= 8 and query.shape[-1] == 2:
            from affmae.ops.knn_triton import knn as knn_triton_compat

            indices = knn_triton_compat(query, database, k)
            if not return_dist:
                return indices
            gathered = torch.gather(
                database.unsqueeze(1).expand(-1, query.shape[1], -1, -1), 2,
                indices.long().unsqueeze(-1).expand(-1, -1, -1,
                                                    database.shape[-1]))
            dist = (gathered - query.unsqueeze(2)).pow(2).sum(-1).sqrt()
            return indices, dist
        # Exact, device-agnostic, and the only option on MPS or ROCm.
        from affmae.ops.knn_triton import knn_pytorch_large

        indices = knn_pytorch_large(query, database, k).to(torch.int32)
        if not return_dist:
            return indices
        gathered = torch.gather(
            database.unsqueeze(1).expand(-1, query.shape[1], -1, -1), 2,
            indices.long().unsqueeze(-1).expand(-1, -1, -1,
                                                database.shape[-1]))
        dist = (gathered - query.unsqueeze(2)).pow(2).sum(-1).sqrt()
        return indices, dist

    if not have_keops():
        from affmae.ops.knn_triton import knn as knn_triton_compat

        return knn_triton_compat(query, database, k, return_dist=return_dist)

    _ensure_keops()
    b, n, c = database.shape

    # disable amp/not supported for knn keops computation
    with torch.amp.autocast(device_type='cuda', enabled=False):
        with torch.no_grad():
            # detach from graph
            query = query.detach()
            database = database.detach()

            # Keops does not support half precision
            if query.dtype != torch.float32:
                query = query.to(torch.float32)
            if database.dtype != torch.float32:
                database = database.to(torch.float32)
            from pykeops.torch import LazyTensor
            query_ = LazyTensor(query[:, None, :, :])
            database_ = LazyTensor(database[:, :, None, :])
            dist = ((query_-database_) ** 2).sum(-1) ** 0.5  # b x n x n_

        if return_dist:
            nn_dist, nn_idx = dist.Kmin_argKmin(k, dim=1)  # b x n_ x k
            return nn_idx, nn_dist
        else:
            nn_idx = dist.argKmin(k, dim=1)  # b x n_ x k
            return nn_idx
