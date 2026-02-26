"""
Configuration utilities for Channel Finder service.

This module provides configuration access for the Channel Finder service,
wrapping Osprey's centralized config system.
"""

from pathlib import Path
from typing import Any

# Use Osprey's public config API
from osprey.utils.config import get_config_builder
from osprey.utils.config import load_config as osprey_load_config


def get_config() -> dict[str, Any]:
    """
    Get default configuration dictionary.

    Returns the raw configuration dictionary from the default config file
    (config.yml in project root or CONFIG_FILE environment variable).

    Returns:
        Configuration dictionary as loaded from YAML
    """
    return osprey_load_config()


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load configuration from a specific file path.

    Args:
        config_path: Path to the configuration YAML file

    Returns:
        Configuration dictionary as loaded from YAML
    """
    return osprey_load_config(config_path)


def resolve_path(path_str: str) -> Path:
    """
    Resolve path relative to project root.

    Args:
        path_str: Path string (absolute or relative to project root)

    Returns:
        Resolved absolute Path object
    """
    config_builder = get_config_builder()
    project_root = Path(config_builder.get("project_root"))
    path = Path(path_str)

    if path.is_absolute():
        return path
    return project_root / path
