import logging
import os
import re

import yaml
import shutil
from dataclasses import dataclass

@dataclass
class Config:
    def __init__(self, dictionary=None):
        if dictionary:
            for k, v in dictionary.items():
                setattr(self, k, v)

    def __str__(self):
        return "\n".join([f"{k}: {v}" for k, v in self.__dict__.items()])

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value):
    """Expand ``${VAR}`` and ``${VAR:-default}`` inside a string value.

    Configs reference machine-specific dataset and checkpoint locations. Env
    expansion lets one committed config work on several machines: set
    ``DATA_ROOT`` once instead of editing every file.

    Args:
        value: any config value; non-strings pass through untouched.
    Returns:
        The value with any ``${...}`` references replaced.
    Raises:
        KeyError: if a referenced variable is unset and has no default, with a
            message naming the variable.
    """
    if not isinstance(value, str) or "${" not in value:
        return value

    def substitute(match):
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise KeyError(
            f"Config references ${{{name}}} but that variable is not set. "
            f"Export it, add it to .env, or write ${{{name}:-/some/default}}.")

    return _ENV_PATTERN.sub(substitute, value)


def _expand_env_tree(node):
    """Recursively apply :func:`_expand_env` through dicts and lists."""
    if isinstance(node, dict):
        return {k: _expand_env_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env_tree(v) for v in node]
    return _expand_env(node)


def _flatten_dict_strict(d, flattened=None, seen_keys=None, parent_path=""):
    """
    Recursively flattens a dictionary.
    Raises ValueError if a key name is repeated anywhere in the hierarchy.
    """
    if flattened is None: flattened = {}
    if seen_keys is None: seen_keys = {}  # map key -> original path for better error msgs

    for key, value in d.items():
        # current path for debugging (e.g. "model.aff.embed_dims")
        current_path = f"{parent_path}.{key}" if parent_path else key

        if isinstance(value, dict):
            # recurse into sub-dictionary
            _flatten_dict_strict(value, flattened, seen_keys, current_path)
        else:
            # it's a leaf value (setting)
            if key in seen_keys:
                existing_path = seen_keys[key]
                raise ValueError(
                    f"Config Collision Detected!\n"
                    f"The key '{key}' is defined in two places:\n"
                    f"  1. {existing_path}\n"
                    f"  2. {current_path}\n"
                    f"Please rename one of them in your YAML file."
                )

            seen_keys[key] = current_path
            flattened[key] = value

    return flattened

#: Config keys holding a checkpoint path. A relative value is resolved against
#: the repository root, not the caller's cwd, so `python evaluate.py` works from
#: anywhere. Absolute values are left alone.
CHECKPOINT_KEYS = ("pretrained_ckpt_path", "resume_path")

#: Config keys holding a dataset path, resolved the same way. ``base_path`` is a
#: directory of image/mask splits; ``path`` is the pretraining WebDataset shard
#: pattern, which may contain a brace range and is still just a path to join.
#: Every shipped config writes both under ``${DATA_ROOT:-data}``, so one variable
#: relocates the whole dataset tree and the committed default is a `data/` folder
#: in the repository root.
DATA_KEYS = ("base_path", "path")


def _resolve_checkpoint_paths(config) -> None:
    """Make relative checkpoint and data paths absolute, against the repo root.

    The shipped configs write ``${CHECKPOINT_ROOT:-weights}/pretrain/...``, which
    keeps them portable: no absolute path to anyone's home directory, and one
    environment variable relocates every weight at once.

    Args:
        config: a Config, modified in place.
    """
    from affmae.utils.paths import repo_root

    for key in CHECKPOINT_KEYS + DATA_KEYS:
        value = getattr(config, key, None)
        if not value or not isinstance(value, str):
            continue
        if not os.path.isabs(value):
            setattr(config, key, os.path.join(str(repo_root()), value))


def load_config(config_path: str) -> Config:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        raw_dict = yaml.safe_load(f)

    # Resolve ${VAR} / ${VAR:-default} before anything reads the values.
    raw_dict = _expand_env_tree(raw_dict)

    # flatten with strict collision detection
    flat_config_dict = _flatten_dict_strict(raw_dict)

    # create config object
    config = Config(flat_config_dict)

    # Canonicalize legacy model names (e.g. 'aff' -> 'affmae') so shipped
    # configs keep working against the registry. Imported here to keep config
    # loading free of a module-import cycle.
    _resolve_checkpoint_paths(config)

    model_type = getattr(config, "model_type", None)
    if model_type is not None:
        from affmae.models.registry import resolve_alias

        canonical = resolve_alias(model_type)
        if canonical != model_type:
            logging.info(
                "config: model_type '%s' is a legacy alias for '%s'.",
                model_type, canonical,
            )
            config.model_type = canonical

    # keep raw dict in case W&B needs the nested structure for organization
    config._nested_config = raw_dict

    return config

def create_experiment_dir(config: Config, config_path: str, exp_name_override: str = None) -> str:
    # use provided experiment name or create one from timestamp
    if exp_name_override:
        exp_name = exp_name_override
    else:
        # fallback to config name
        name_base = getattr(config, 'experiment_name', 'no_name_exp')
        exp_name = f"{name_base}"

    exp_dir = os.path.join(config.output_dir, exp_name)

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'evaluations'), exist_ok=True)

    shutil.copy2(config_path, os.path.join(exp_dir, 'config.yaml'))

    return exp_dir