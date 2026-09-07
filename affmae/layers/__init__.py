"""Neural network modules: the ``nn.Module`` layer of the stack.

Holds parameters and composes operators; the operators themselves, and every
backend of them, live in :mod:`affmae.ops`. So a module here never branches on
which kernel to call -- it passes ``backend=`` down.

Imports only torch, ``affmae.ops`` and ``affmae.utils`` -- no renderers, no
optional dependencies. See ``tests/test_import_hygiene.py``.
"""

from .drop_path import DropPath, drop_path

__all__ = ["DropPath", "drop_path"]
