"""Configuration utilities for ad_generalization_benchmark."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml

CONFIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONFIG_DIR.parent
PROJECT_ROOT = REPO_ROOT.parent


def get_config_path(filename: str) -> Path:
    """Get absolute path to a configuration file."""
    return CONFIG_DIR / filename


def load_yaml(filepath: Path) -> Dict[str, Any]:
    """Safely load a YAML configuration file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_paths_config() -> Dict[str, Any]:
    """Load paths configuration."""
    return load_yaml(get_config_path("paths.yaml"))


def load_experiment_config() -> Dict[str, Any]:
    """Load experiment hyperparameters and configuration."""
    return load_yaml(get_config_path("experiment.yaml"))


def load_features_config() -> Dict[str, Any]:
    """Load feature mappings and label mappings configuration."""
    return load_yaml(get_config_path("features.yaml"))


def resolve_dataset_path(key: str) -> Path:
    """Resolve dataset path using fallback strategies:
    1. Exact configured path
    2. Path relative to PROJECT_ROOT
    3. Path relative to current working directory
    """
    cfg = load_paths_config()
    raw_path_str = cfg.get("datasets", {}).get(key)
    if not raw_path_str:
        raise KeyError(f"Dataset key '{key}' not found in paths.yaml")

    raw_path = Path(raw_path_str)
    if raw_path.exists():
        return raw_path

    # Try relative to PROJECT_ROOT
    candidate = PROJECT_ROOT / "datasets" / raw_path.name
    if candidate.exists():
        return candidate

    # Try relative to cwd
    candidate = Path.cwd() / "datasets" / raw_path.name
    if candidate.exists():
        return candidate

    # Return raw_path if none exist so standard FileNotFoundError shows the configured path
    return raw_path
