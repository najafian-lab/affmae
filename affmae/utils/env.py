"""Load credentials from a local ``.env`` file.

Precedence is deliberate: **already-set environment variables always win** over
``.env``. That keeps ``WANDB_API_KEY=... python pretrain.py`` and CI secrets
authoritative, and makes ``.env`` a convenience rather than a hidden override.

No dependency on ``python-dotenv``; the format needed is one ``KEY=value`` per
line.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["load_dotenv", "wandb_available"]


def load_dotenv(path=None, override=False):
    """Read ``KEY=value`` lines from ``.env`` into ``os.environ``.

    Args:
        path: str, Path, or None. Defaults to ``.env`` beside the repository
            root (two levels up from this file).
        override: bool, if True let file values replace existing environment
            variables. Defaults to False, so the real environment wins.
    Returns:
        list of str, names of the variables this call actually set.
    """
    if path is None:
        path = Path(__file__).resolve().parents[2] / ".env"
    path = Path(path)
    if not path.is_file():
        return []

    applied = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # Strip optional surrounding quotes; a blank value means "not set".
        value = value.strip().strip('"').strip("'")
        if not key or not value:
            continue
        if key in os.environ and not override:
            continue
        os.environ[key] = value
        applied.append(key)

    if applied:
        logger.info("Loaded %d variable(s) from %s: %s",
                    len(applied), path.name, ", ".join(sorted(applied)))
    return applied


def wandb_available():
    """True if W&B has credentials to authenticate with.

    Checks ``WANDB_API_KEY`` and the netrc that ``wandb login`` writes, so a
    driver can skip logging instead of dying inside ``wandb.init``.
    """
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path.home() / ".netrc"
    return netrc.is_file() and "api.wandb.ai" in netrc.read_text()
