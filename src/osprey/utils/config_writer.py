"""Comment-preserving YAML configuration writer.

Unified module for all comment-preserving YAML mutations. Uses ruamel.yaml
(round-trip mode) to read, modify, and write YAML files without stripping
comments, formatting, or ordering.

This module consolidates the former ``yaml_config.py`` and ``config_updater.py``
into a single source of truth for config file modifications.

For read-only config access, use ``utils/config.py`` (ConfigBuilder, get_config_value).

Typical usage:
    from osprey.utils.config_writer import config_add_to_list, config_update_fields

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

from ruamel.yaml import YAML, CommentedMap

logger = logging.getLogger(__name__)

# Round-trip mode ("rt") preserves comments, key order, and quoting style.
_yaml = YAML(typ="rt")
_yaml.preserve_quotes = True
_yaml.width = 4096  # prevent aggressive line-wrapping


# =============================================================================
# Core I/O
# =============================================================================


def _load(path: Path) -> Any:
    """Load a YAML file with comment preservation."""
    with open(path, encoding="utf-8") as f:
        data = _yaml.load(f)
    return data if data is not None else CommentedMap()


def _save(path: Path, data: Any) -> None:
    """Write data back to a YAML file, preserving comments."""
    with open(path, "w", encoding="utf-8") as f:
        _yaml.dump(data, f)


# =============================================================================
# List Operations
# =============================================================================


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
    logger.info("config_remove_from_list: %s -= %s in %s", ".".join(key_path), value, config_path)
    return True


# =============================================================================
# Field Updates
# =============================================================================


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


# =============================================================================
# Comment-Preserving YAML File Update
# =============================================================================


def update_yaml_file(
    file_path: Path,
    updates: dict[str, Any],
    create_backup: bool = True,
    section_comments: dict[str, str] | None = None,
) -> Path | None:
    """Update a YAML file while preserving comments and formatting.

    Uses the shared ruamel.yaml handler to maintain comments, blank lines,
    and original formatting when modifying YAML configuration files.

    Args:
        file_path: Path to the YAML file to update
        updates: Dictionary of updates to apply. Supports nested paths using
                dot notation in keys (e.g., {"control_system.type": "epics"})
                or nested dictionaries that will be merged.
        create_backup: If True, creates a .bak file before modifying
        section_comments: Optional dict mapping top-level keys to comment headers.
                         Comments are added with a blank line before them.
                         Example: {"simulation": "Simulation Configuration"}

    Returns:
        Path to backup file if created, None otherwise

    Examples:
        >>> # Simple update
        >>> update_yaml_file(Path("config.yml"), {"control_system.type": "epics"})

        >>> # Nested update
        >>> update_yaml_file(Path("config.yml"), {
        ...     "control_system": {
        ...         "type": "epics",
        ...         "connector": {"epics": {"gateways": {"read_only": {"port": 5064}}}}
        ...     }
        ... })
    """
    data = _load(file_path)

    # Create backup before modifying
    backup_path = None
    if create_backup:
        backup_path = file_path.with_suffix(".yml.bak")
        backup_path.write_text(file_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Track which keys are being added (for comment placement)
    new_keys = set()
    for key in updates:
        top_key = key.split(".")[0] if "." in key else key
        if top_key not in data:
            new_keys.add(top_key)

    # Apply updates
    _apply_nested_updates(data, updates)

    # Add section comments for new top-level keys
    if section_comments:
        for key, comment in section_comments.items():
            if key in new_keys and key in data:
                separator = "=" * 60
                comment_text = f"\n{separator}\n{comment}\n{separator}"
                data.yaml_set_comment_before_after_key(key, before=comment_text)

    _save(file_path, data)

    return backup_path


def _apply_nested_updates(data: dict, updates: dict) -> None:
    """Apply nested updates to a dictionary, supporting dot notation keys.

    Args:
        data: Target dictionary to update (modified in place)
        updates: Updates to apply
    """
    for key, value in updates.items():
        if "." in key:
            _set_nested_value(data, key, value)
        elif isinstance(value, dict) and key in data and isinstance(data[key], dict):
            _apply_nested_updates(data[key], value)
        else:
            data[key] = value


def _set_nested_value(data: dict, path: str, value: Any) -> None:
    """Set a value at a nested path using dot notation.

    Args:
        data: Target dictionary
        path: Dot-separated path (e.g., "control_system.connector.epics.port")
        value: Value to set
    """
    keys = path.split(".")
    current = data

    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


# =============================================================================
# Config File Discovery
# =============================================================================


def find_config_file() -> Path | None:
    """Find the config.yml file in current directory.

    Returns:
        Path to config.yml or None if not found
    """
    config_path = Path.cwd() / "config.yml"
    return config_path if config_path.exists() else None


# =============================================================================
# Control System Type Configuration
# =============================================================================


def get_control_system_type(config_path: Path, key: str = "control_system.type") -> str | None:
    """Get control system or archiver type from config.yml.

    Args:
        config_path: Path to config.yml
        key: Config key to read (e.g., 'control_system.type' or 'archiver.type')

    Returns:
        Type string or None if not found
    """
    try:
        data = _load(config_path)

        keys = key.split(".")
        value = data
        for k in keys:
            value = value.get(k)
            if value is None:
                return None

        return value
    except Exception:
        pass  # Config read/parse failed; return default below

    return None


def set_control_system_type(
    config_path: Path,
    control_type: str,
    archiver_type: str | None = None,
    create_backup: bool = True,
) -> tuple[str, str]:
    """Update control system and optionally archiver type in config.yml.

    Uses comment-preserving YAML update via update_yaml_file().

    Args:
        config_path: Path to config.yml
        control_type: 'mock' or 'epics'
        archiver_type: Optional archiver type ('mock_archiver', 'epics_archiver')
        create_backup: If True, creates a .bak file before modifying

    Returns:
        Tuple of (updated_content, preview) where updated_content is the new file content
    """
    updates: dict[str, Any] = {"control_system.type": control_type}

    if archiver_type:
        updates["archiver.type"] = archiver_type

    update_yaml_file(config_path, updates, create_backup=create_backup)

    updated_content = config_path.read_text()

    preview_lines = [
        "[bold]Control System Configuration[/bold]\n",
        f"control_system.type: {control_type}",
    ]

    if archiver_type:
        preview_lines.append(f"archiver.type: {archiver_type}")

    preview_lines.append("\n[dim]Updated config.yml[/dim]")

    preview = "\n".join(preview_lines)

    return updated_content, preview


# =============================================================================
# EPICS Gateway Configuration
# =============================================================================


def set_epics_gateway_config(
    config_path: Path,
    facility: str,
    custom_config: dict | None = None,
    create_backup: bool = True,
) -> tuple[str, str]:
    """Update EPICS gateway configuration in config.yml.

    Updates the control_system.connector.epics.gateways section with
    facility-specific or custom gateway settings.

    Args:
        config_path: Path to config.yml
        facility: 'aps', 'als', or 'custom'
        custom_config: For 'custom', dict with 'read_only' and 'write_access' gateways
        create_backup: If True, creates a .bak file before modifying

    Returns:
        Tuple of (updated_content, preview) where updated_content is the new file content
    """
    from osprey.templates.data import get_facility_config

    if facility == "custom":
        if not custom_config:
            raise ValueError("custom_config required when facility='custom'")
        gateway_config = custom_config
        facility_name = "Custom"
    else:
        preset = get_facility_config(facility)
        if not preset:
            raise ValueError(f"Unknown facility: {facility}")
        gateway_config = preset["gateways"]
        facility_name = preset["name"]

    update_yaml_file(
        config_path,
        {"control_system.connector.epics.gateways": gateway_config},
        create_backup=create_backup,
    )

    updated_content = config_path.read_text()

    gateway_yaml = _format_gateway_yaml(gateway_config)
    preview = f"""
[bold]EPICS Gateway Configuration - {facility_name}[/bold]

{gateway_yaml.rstrip()}

[dim]Updated control_system.connector.epics.gateways in config.yml[/dim]
"""

    return updated_content, preview


def _format_gateway_yaml(gateway_config: dict) -> str:
    """Format gateway configuration as YAML string.

    Args:
        gateway_config: Dict with 'read_only' and 'write_access' gateway configs

    Returns:
        Formatted YAML string with proper indentation
    """
    lines = []

    for gateway_type in ["read_only", "write_access"]:
        if gateway_type in gateway_config:
            gw = gateway_config[gateway_type]
            lines.append(f"        {gateway_type}:")
            lines.append(f"          address: {gw['address']}")
            lines.append(f"          port: {gw['port']}")
            lines.append(
                f"          use_name_server: {str(gw.get('use_name_server', False)).lower()}"
            )

    return "\n".join(lines) + "\n"


def get_epics_gateway_config(config_path: Path) -> dict | None:
    """Get current EPICS gateway configuration from config.yml.

    Args:
        config_path: Path to config.yml

    Returns:
        Dict with gateway configuration or None if not found
    """
    try:
        data = _load(config_path)

        gateways = (
            data.get("control_system", {}).get("connector", {}).get("epics", {}).get("gateways")
        )
        return gateways
    except Exception:
        pass  # Config read/parse failed; return default below

    return None


def get_facility_from_gateway_config(config_path: Path) -> str | None:
    """Detect which facility preset is configured (if any).

    Compares current gateway configuration against known facility presets.

    Args:
        config_path: Path to config.yml

    Returns:
        Facility name ('APS', 'ALS', 'Custom') or None if using defaults
    """
    from osprey.templates.data import FACILITY_GATEWAYS

    current_gateways = get_epics_gateway_config(config_path)
    if not current_gateways:
        return None

    for _facility_id, preset in FACILITY_GATEWAYS.items():
        preset_gateways = preset["gateways"]

        if "read_only" in current_gateways and "read_only" in preset_gateways:
            current_read = current_gateways["read_only"]
            preset_read = preset_gateways["read_only"]

            if (
                current_read.get("address") == preset_read["address"]
                and current_read.get("port") == preset_read["port"]
            ):
                return preset["name"]

    if "read_only" in current_gateways:
        read_addr = current_gateways["read_only"].get("address", "")
        if read_addr and read_addr != "cagw-alsdmz.als.lbl.gov":
            return "Custom"

    return None
