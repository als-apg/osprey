"""
Context Loading Utilities

Utilities for loading context data from various sources (JSON files, etc.).
Used by the generated code wrapper to provide context access in subprocess execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from osprey.utils.logger import get_logger

logger = get_logger("context_loader")


class _DictNamespace:
    """Lightweight namespace providing dot-notation access to nested dicts.

    Replaces the deleted ContextManager for simple read-only context access
    in generated code wrappers.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(f"No context key '{name}'") from None
        if isinstance(value, dict):
            return _DictNamespace(value)
        return value

    def __repr__(self) -> str:
        return f"_DictNamespace({list(self._data.keys())})"


def load_context(context_file: str = "context.json") -> _DictNamespace | None:
    """Load agent execution context from a JSON file in the current directory.

    Provides dot-notation access to context data stored as nested JSON.
    Used by the generated code wrapper for subprocess execution.

    Args:
        context_file: Name of the context file (default: "context.json")

    Returns:
        Namespace with dot-notation access to context data, or None if loading fails

    Usage:
        >>> from osprey.context import load_context
        >>> context = load_context()
        >>> data = context.ARCHIVER_DATA.beam_current_historical_data
    """
    try:
        context_path = Path.cwd() / context_file

        if not context_path.exists():
            logger.warning(f"Context file not found: {context_path}")
            return None

        with open(context_path, encoding="utf-8") as f:
            context_data = json.load(f)

        if context_data:
            logger.info("Context loaded successfully")
            logger.info(f"Available context types: {list(context_data.keys())}")
            return _DictNamespace(context_data)
        else:
            logger.warning("Context loaded but no data found")
            return None

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in context file: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading context: {e}")
        return None
