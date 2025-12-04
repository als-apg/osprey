"""Utilities for channel finder service."""

# Only export the backward compatibility functions
from .config import get_config, resolve_path

__all__ = ['get_config', 'resolve_path']
