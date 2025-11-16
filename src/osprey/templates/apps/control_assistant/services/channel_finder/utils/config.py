"""
Configuration utilities for Channel Finder service.

This module has been migrated to use Osprey's centralized config system.
All configuration is now loaded from the main Osprey config (config.yml at project root).

For backward compatibility, we provide minimal stubs that redirect to Osprey's config.
"""

from pathlib import Path
from typing import Dict, Any

# Use Osprey's config system
from osprey.utils.config import _get_config as get_osprey_config


def get_config() -> Dict[str, Any]:
    """
    Get configuration dictionary.
    
    DEPRECATED: This function is maintained for backward compatibility only.
    New code should use Osprey's config system directly:
        from osprey.utils.config import _get_config
        config_builder = _get_config()
        value = config_builder.get('some.path')
    
    Returns:
        Configuration dictionary (raw_config from Osprey's ConfigBuilder)
    """
    config_builder = get_osprey_config()
    return config_builder.raw_config


def resolve_path(path_str: str) -> Path:
    """
    Resolve path relative to project root.
    
    DEPRECATED: This function is maintained for backward compatibility only.
    New code should use Osprey's config system for path resolution.
    
    Args:
        path_str: Path string (absolute or relative to project root)
        
    Returns:
        Resolved absolute Path object
    """
    config_builder = get_osprey_config()
    project_root = Path(config_builder.get('project_root'))
    path = Path(path_str)
    
    if path.is_absolute():
        return path
    return project_root / path
