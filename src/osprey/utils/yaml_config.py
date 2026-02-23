"""Comment-preserving YAML configuration utilities.

Uses ruamel.yaml (round-trip mode) to read, modify, and write YAML files
without stripping comments, formatting, or ordering. All config mutations
in the codebase should go through these functions.

Typical usage:
    from osprey.utils.yaml_config import config_add_to_list, config_update_fields

    # Add entry to a YAML list
    config_add_to_list(Path("config.yml"), ["prompts", "user_owned"], "rules/facility")

    # Apply structured key-value updates
    config_update_fields(Path("config.yml"), {
        "control_system.writes_enabled": True,
        "approval.global_mode": "selective",
    })
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

logger = logging.getLogger(__name__)

# Shared YAML instance — round-trip mode preserves comments, order, and quoting.
_yaml = YAML(typ="rt")
_yaml.preserve_quotes = True
_yaml.width = 200  # prevent aggressive line-wrapping


def _load(path: Path) -> Any:
    """Load a YAML file with comment preservation."""
    with open(path, encoding="utf-8") as f:
        data = _yaml.load(f)
    return data if data is not None else {}


def _save(path: Path, data: Any) -> None:
    """Write data back to a YAML file, preserving comments."""
    with open(path, "w", encoding="utf-8") as f:
        _yaml.dump(data, f)


def config_add_to_list(
    config_path: Path,
    key_path: list[str],
    value: str,
) -> bool:
    """Append *value* to a YAML list at *key_path*, creating parents if needed.

    Args:
        config_path: Path to the YAML file.
        key_path: List of nested keys, e.g. ``["prompts", "user_owned"]``.
        value: The scalar value to append.

    Returns:
        True if the value was added, False if it was already present.
    """
    data = _load(config_path)

    node = data
    for key in key_path[:-1]:
        if key not in node:
            node[key] = {}
        node = node[key]

    leaf = key_path[-1]
    if leaf not in node:
        node[leaf] = []

    lst = node[leaf]
    if value in lst:
        return False

    lst.append(value)
    _save(config_path, data)
    logger.debug("config_add_to_list: %s += %s in %s", ".".join(key_path), value, config_path)
    return True


def config_remove_from_list(
    config_path: Path,
    key_path: list[str],
    value: str,
    *,
    prune_empty: bool = True,
) -> bool:
    """Remove *value* from a YAML list at *key_path*.

    Args:
        config_path: Path to the YAML file.
        key_path: List of nested keys.
        value: The scalar value to remove.
        prune_empty: If True, delete empty parent keys after removal.

    Returns:
        True if the value was removed, False if it was not present.
    """
    data = _load(config_path)

    parents: list[tuple[Any, str]] = []
    node = data
    for key in key_path[:-1]:
        if key not in node:
            return False
        parents.append((node, key))
        node = node[key]

    leaf = key_path[-1]
    if leaf not in node:
        return False

    lst = node[leaf]
    if value not in lst:
        return False

    lst.remove(value)

    if prune_empty:
        if not lst:
            del node[leaf]
        for parent_node, parent_key in reversed(parents):
            child = parent_node[parent_key]
            if isinstance(child, dict) and not child:
                del parent_node[parent_key]

    _save(config_path, data)
    logger.info(
        "config_remove_from_list: %s -= %s in %s", ".".join(key_path), value, config_path
    )
    return True


def config_update_fields(
    config_path: Path,
    updates: dict[str, Any],
) -> None:
    """Apply structured field updates to a YAML config, preserving comments.

    Keys use dot-notation to address nested values.
    Array values in *updates* are written as YAML sequences.

    Args:
        config_path: Path to the YAML file.
        updates: Mapping of ``"dot.separated.key"`` → new value.
    """
    data = _load(config_path)

    for dotted_key, value in updates.items():
        parts = dotted_key.split(".")
        node = data
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    _save(config_path, data)
    logger.info("config_update_fields: updated %d field(s) in %s", len(updates), config_path)


def config_read(config_path: Path) -> dict:
    """Load a YAML config as a plain dict (no ruamel wrapper types).

    Useful when you need a JSON-serializable dict for API responses.
    """
    import copy
    import json

    data = _load(config_path)
    return json.loads(json.dumps(copy.deepcopy(data), default=str))
