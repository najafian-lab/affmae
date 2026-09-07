"""Shared output locations. """

import os
from pathlib import Path

__all__ = ["OUTPUT_ENV_VAR", "output_root", "output_path", "plots_dir",
           "default_plot_path"]

OUTPUT_ENV_VAR = "AFFMAE_OUTPUT_DIR"
_DEFAULT_DIRNAME = "output"


def repo_root():
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parents[2]


def output_root(create=False):
    """Root directory for all generated artifacts.

    Args:
        create: bool, create the directory if missing.
    Returns:
        Path to the output root.
    """
    env = os.environ.get(OUTPUT_ENV_VAR)
    root = Path(env).expanduser() if env else repo_root() / _DEFAULT_DIRNAME
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def output_path(*parts, create_parent=False):
    """Join ``parts`` under the output root.

    Args:
        *parts: path components.
        create_parent: bool, create the containing directory.
    Returns:
        Path under the output root.
    """
    path = output_root().joinpath(*parts)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plots_dir(create=True):
    """Directory for figures: ``<output root>/plots``."""
    path = output_root() / "plots"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def default_plot_path(filename):
    """Default destination for a named figure, as a string."""
    return str(plots_dir() / filename)
