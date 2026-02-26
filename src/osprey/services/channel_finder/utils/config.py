"""Configuration utilities for Channel Finder service."""

from pathlib import Path
from typing import Any

from osprey.utils.config import get_config_builder
from osprey.utils.config import load_config as osprey_load_config


def get_config() -> dict[str, Any]:
    """Get default configuration dictionary."""
    return osprey_load_config()


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from a specific file path."""
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
