import yaml
import os
import shutil
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Config:
    def __init__(self, dictionary=None):
        if dictionary:
            for k, v in dictionary.items():
                setattr(self, k, v)

    def __str__(self):
        return "\n".join([f"{k}: {v}" for k, v in self.__dict__.items()])

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

def load_config(config_path: str) -> Config:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        raw_dict = yaml.safe_load(f)

    # flatten with strict collision detection
    flat_config_dict = _flatten_dict_strict(raw_dict)
    
    # create config object
    config = Config(flat_config_dict)
    
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