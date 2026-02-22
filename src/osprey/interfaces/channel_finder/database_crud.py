"""Channel Finder Database CRUD Operations.

Pure functions for creating, reading, updating, and deleting nodes/channels
in the three channel-finder database formats (hierarchical, middle_layer,
in_context). All mutations use atomic writes with backup.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Metadata keys to skip during tree traversal
_HIER_META_KEYS = frozenset(
    {
        "_description",
        "_expansion",
        "_channel_part",
        "_is_leaf",
        "_separator",
    }
)
_ML_META_KEYS = frozenset(
    {
        "_description",
        "setup",
        "pyat",
        "ChannelNames",
        "DataType",
        "Mode",
        "Units",
        "HW2PhysicsParams",
        "Physics2HWParams",
        "Tol",
        "Range",
        "Description",
        "CommonNames",
        "DeviceList",
    }
)


class CrudError(Exception):
    """Raised on invalid CRUD operations."""

    def __init__(self, message: str, error_type: str = "crud_error"):
        super().__init__(message)
        self.error_type = error_type


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    """Load and parse a JSON file."""
    with open(path) as f:
        return json.load(f)


def _atomic_write(path: Path, data: dict | list) -> None:
    """Write JSON atomically: backup current file, write via temp, os.replace.

    Creates a `.json.bak` backup of the existing file before overwriting.
    """
    path = Path(path)

    # Create backup if file exists
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)

    # Write to temp file in same directory, then atomic replace
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        # Clean up temp file on error
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _reload_registry(pipeline_type: str) -> None:
    """Reload the in-memory database from the registry singleton."""
    try:
        if pipeline_type == "hierarchical":
            from osprey.services.channel_finder.mcp.hierarchical.registry import (
                get_cf_hier_registry,
            )

            get_cf_hier_registry().database.load_database()

        elif pipeline_type == "middle_layer":
            from osprey.services.channel_finder.mcp.middle_layer.registry import (
                get_cf_ml_registry,
            )

            get_cf_ml_registry().database.load_database()

        else:  # in_context
            from osprey.services.channel_finder.mcp.in_context.registry import (
                get_cf_ic_registry,
            )

            get_cf_ic_registry().database.load_database()

    except Exception:
        logger.warning("Failed to reload %s registry (may not be initialized)", pipeline_type)


# ---------------------------------------------------------------------------
# Hierarchical CRUD — operates on data["tree"]
# ---------------------------------------------------------------------------


def _hier_level_info(data: dict) -> tuple[list[str], dict[str, str]]:
    """Extract level names and level types from hierarchy config.

    Args:
        data: The full database dict (must contain ``hierarchy.levels``).

    Returns:
        Tuple of (level_names, level_types) where level_types maps name → type.
        Levels without an explicit type default to ``"tree"``.
    """
    hierarchy = data.get("hierarchy", {})
    level_names: list[str] = []
    level_types: dict[str, str] = {}
    for lvl in hierarchy.get("levels", []):
        if isinstance(lvl, str):
            name = lvl
            ltype = "tree"
        else:
            name = lvl.get("name") or lvl.get("level", "")
            ltype = lvl.get("type", "tree")
        level_names.append(name)
        level_types[name] = ltype
    return level_names, level_types


def _hier_navigate(
    tree: dict,
    level_names: list[str],
    selections: dict[str, str],
    level_types: dict[str, str] | None = None,
) -> dict:
    """Navigate the hierarchical tree to a parent node.

    Args:
        tree: The root tree dict.
        level_names: Ordered list of hierarchy level names.
        selections: Mapping of level_name → selected_value for parent path.
        level_types: Optional mapping of level_name → type (``"tree"`` or
            ``"instances"``).  When an instance-type level is encountered its
            selection value is *not* used as a tree key; instead the function
            enters the child that owns an ``_expansion`` key.

    Returns:
        The target subtree dict.

    Raises:
        CrudError: If navigation path is invalid.
    """
    if level_types is None:
        level_types = {}

    node = tree
    for lvl in level_names:
        val = selections.get(lvl)
        if val is None:
            break

        ltype = level_types.get(lvl, "tree")

        if ltype == "instances":
            # Instance levels: navigate into the container that has _expansion
            # (the selected value like "B01" is an expanded instance, not a key)
            found = False
            for key, child in node.items():
                if key.startswith("_") or not isinstance(child, dict):
                    continue
                if "_expansion" in child:
                    node = child
                    found = True
                    break
            if not found:
                raise CrudError(
                    f"No expansion container found at instance level '{lvl}'",
                    "invalid_path",
                )
        else:
            # Tree levels: navigate using the selection value as a key
            if val not in node:
                raise CrudError(f"Node '{val}' not found at level '{lvl}'", "not_found")
            node = node[val]
            if not isinstance(node, dict):
                raise CrudError(
                    f"Node '{val}' at level '{lvl}' is not navigable", "invalid_path"
                )
    return node


def _hier_child_keys(node: dict) -> list[str]:
    """Return non-metadata child keys of a hierarchical node."""
    return [k for k in node if k not in _HIER_META_KEYS and not k.startswith("_")]


def _hier_count_channels(node: dict) -> int:
    """Recursively count leaf channels under a node."""
    if not isinstance(node, dict):
        return 0
    count = 0
    for k, v in node.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            count += _hier_count_channels(v)
        else:
            count += 1
    # A dict with no non-meta children is itself a leaf node
    children = _hier_child_keys(node)
    if not children:
        count += 1
    return count


def hier_add_node(
    db_path: str | Path,
    level: str,
    parent_selections: dict[str, str],
    name: str,
    description: str = "",
    channel_part: str | None = None,
) -> dict:
    """Add a new node at a given hierarchy level.

    Args:
        db_path: Path to the hierarchical JSON database.
        level: The level name where the node will be added.
        parent_selections: Dict of level→value for parent navigation.
        name: Name of the new node.
        description: Optional description.
        channel_part: Optional channel part override.

    Returns:
        Success dict with node info.
    """
    data = _load_json(Path(db_path))
    tree = data.get("tree", data)

    # Navigate to parent
    level_names, level_types = _hier_level_info(data)
    parent = _hier_navigate(tree, level_names, parent_selections, level_types)

    if name in parent:
        raise CrudError(f"Node '{name}' already exists at this level", "duplicate")

    # Create the new node
    new_node: dict = {}
    if description:
        new_node["_description"] = description
    if channel_part:
        new_node["_channel_part"] = channel_part

    parent[name] = new_node

    _atomic_write(Path(db_path), data)
    _reload_registry("hierarchical")

    return {"success": True, "name": name, "level": level}


def hier_edit_node(
    db_path: str | Path,
    level: str,
    selections: dict[str, str],
    old_name: str,
    new_name: str | None = None,
    description: str | None = None,
) -> dict:
    """Edit a node's name and/or description at a hierarchy level.

    Args:
        db_path: Path to the hierarchical JSON database.
        level: The level where the node exists.
        selections: Parent selections (excluding the target level).
        old_name: Current name of the node.
        new_name: New name for the node (None to keep current).
        description: New description (None to leave unchanged,
            empty string to clear).

    Returns:
        Success dict.
    """
    data = _load_json(Path(db_path))
    tree = data.get("tree", data)

    level_names, level_types = _hier_level_info(data)
    parent = _hier_navigate(tree, level_names, selections, level_types)

    if old_name not in parent:
        raise CrudError(f"Node '{old_name}' not found", "not_found")

    effective_name = new_name if new_name and new_name != old_name else old_name

    # Rename if needed
    if effective_name != old_name:
        if effective_name in parent:
            raise CrudError(f"Node '{effective_name}' already exists", "duplicate")
        parent[effective_name] = parent.pop(old_name)

    # Update description if provided
    if description is not None:
        node = parent[effective_name]
        if isinstance(node, dict):
            if description:
                node["_description"] = description
            else:
                node.pop("_description", None)

    _atomic_write(Path(db_path), data)
    _reload_registry("hierarchical")

    return {"success": True, "old_name": old_name, "new_name": effective_name}


def hier_delete_node(
    db_path: str | Path,
    level: str,
    selections: dict[str, str],
    name: str,
) -> dict:
    """Delete a node (and all descendants) at a hierarchy level.

    Args:
        db_path: Path to the hierarchical JSON database.
        level: The level where the node exists.
        selections: Parent selections (excluding the target level).
        name: Name of the node to delete.

    Returns:
        Success dict with affected channel count.
    """
    data = _load_json(Path(db_path))
    tree = data.get("tree", data)

    level_names, level_types = _hier_level_info(data)
    parent = _hier_navigate(tree, level_names, selections, level_types)

    if name not in parent:
        raise CrudError(f"Node '{name}' not found", "not_found")

    affected = _hier_count_channels(parent[name])
    del parent[name]

    _atomic_write(Path(db_path), data)
    _reload_registry("hierarchical")

    return {"success": True, "name": name, "affected_channels": affected}


def hier_edit_expansion(
    db_path: str | Path,
    level: str,
    selections: dict[str, str],
    pattern: str | None = None,
    range_start: int | None = None,
    range_end: int | None = None,
) -> dict:
    """Edit the expansion config for an instance-type level.

    Navigates to the instance container (the child dict that has ``_expansion``)
    and updates its ``_expansion`` fields.

    Args:
        db_path: Path to the hierarchical JSON database.
        level: The instance-type level name (used for selection navigation).
        selections: Parent selections (the instance-level value is used only
            for navigation, not as a tree key).
        pattern: New pattern string (e.g. ``"B{:02d}"``).  ``None`` leaves unchanged.
        range_start: New range start.  ``None`` leaves unchanged.
        range_end: New range end.  ``None`` leaves unchanged.

    Returns:
        Success dict with updated expansion info.
    """
    data = _load_json(Path(db_path))
    tree = data.get("tree", data)

    level_names, level_types = _hier_level_info(data)

    # Navigate to the parent of the instance level, then find the expansion container
    # We need to navigate *up to but not including* the instance level
    parent_selections = {}
    for lvl in level_names:
        if lvl == level:
            break
        val = selections.get(lvl)
        if val is not None:
            parent_selections[lvl] = val

    parent = _hier_navigate(tree, level_names, parent_selections, level_types)

    # Find the child that has _expansion
    expansion_container = None
    for key, child in parent.items():
        if key.startswith("_") or not isinstance(child, dict):
            continue
        if "_expansion" in child:
            expansion_container = child
            break

    if expansion_container is None:
        raise CrudError(
            f"No expansion container found at level '{level}'",
            "not_found",
        )

    expansion = expansion_container["_expansion"]

    if pattern is not None:
        expansion["pattern"] = pattern
    if range_start is not None and range_end is not None:
        expansion["range"] = [range_start, range_end]
    elif range_start is not None:
        r = expansion.get("range", [1, 1])
        expansion["range"] = [range_start, r[1] if len(r) > 1 else range_start]
    elif range_end is not None:
        r = expansion.get("range", [1, 1])
        expansion["range"] = [r[0] if r else 1, range_end]

    _atomic_write(Path(db_path), data)
    _reload_registry("hierarchical")

    return {"success": True, "level": level, "expansion": expansion}


def hier_get_expansion(
    db_path: str | Path,
    level: str,
    selections: dict[str, str],
) -> dict:
    """Read the current expansion config for an instance-type level.

    Args:
        db_path: Path to the hierarchical JSON database.
        level: The instance-type level name.
        selections: Parent selections for navigation.

    Returns:
        Dict with ``expansion`` key containing the expansion config.
    """
    data = _load_json(Path(db_path))
    tree = data.get("tree", data)

    level_names, level_types = _hier_level_info(data)

    parent_selections = {}
    for lvl in level_names:
        if lvl == level:
            break
        val = selections.get(lvl)
        if val is not None:
            parent_selections[lvl] = val

    parent = _hier_navigate(tree, level_names, parent_selections, level_types)

    for key, child in parent.items():
        if key.startswith("_") or not isinstance(child, dict):
            continue
        if "_expansion" in child:
            return {"expansion": child["_expansion"]}

    raise CrudError(
        f"No expansion container found at level '{level}'",
        "not_found",
    )


def hier_count_descendants(
    db_path: str | Path,
    level: str,
    selections: dict[str, str],
    name: str,
) -> int:
    """Count channels that would be affected by deleting a node.

    Returns:
        Number of descendant channels.
    """
    data = _load_json(Path(db_path))
    tree = data.get("tree", data)

    level_names, level_types = _hier_level_info(data)
    parent = _hier_navigate(tree, level_names, selections, level_types)

    if name not in parent:
        raise CrudError(f"Node '{name}' not found", "not_found")

    return _hier_count_channels(parent[name])


# ---------------------------------------------------------------------------
# Middle Layer CRUD — operates on top-level data dict
# ---------------------------------------------------------------------------


def _ml_navigate(data: dict, system: str, family: str | None = None) -> dict:
    """Navigate to a system or family node in a middle layer database."""
    if system not in data:
        raise CrudError(f"System '{system}' not found", "not_found")
    node = data[system]
    if family is not None:
        if family not in node:
            raise CrudError(f"Family '{family}' not found in system '{system}'", "not_found")
        node = node[family]
    return node


def _ml_count_channels_in_family(family_node: dict) -> int:
    """Count total channels in a middle layer family."""
    count = 0
    for key, val in family_node.items():
        if key in _ML_META_KEYS or key.startswith("_"):
            continue
        if isinstance(val, dict):
            if "ChannelNames" in val:
                ch = val["ChannelNames"]
                count += len(ch) if isinstance(ch, list) else 1
            else:
                # Subfield level
                for _sk, sv in val.items():
                    if isinstance(sv, dict) and "ChannelNames" in sv:
                        ch = sv["ChannelNames"]
                        count += len(ch) if isinstance(ch, list) else 1
    return count


def ml_add_family(
    db_path: str | Path,
    system: str,
    family: str,
    description: str = "",
) -> dict:
    """Add a new family to a system.

    Args:
        db_path: Path to the middle layer JSON database.
        system: System name (must already exist).
        family: New family name.
        description: Optional description.

    Returns:
        Success dict.
    """
    data = _load_json(Path(db_path))
    sys_node = _ml_navigate(data, system)

    if family in sys_node:
        raise CrudError(f"Family '{family}' already exists in system '{system}'", "duplicate")

    new_family: dict = {}
    if description:
        new_family["_description"] = description

    sys_node[family] = new_family

    _atomic_write(Path(db_path), data)
    _reload_registry("middle_layer")

    return {"success": True, "system": system, "family": family}


def ml_delete_family(
    db_path: str | Path,
    system: str,
    family: str,
) -> dict:
    """Delete a family and all its channels.

    Returns:
        Success dict with affected channel count.
    """
    data = _load_json(Path(db_path))
    sys_node = _ml_navigate(data, system)

    if family not in sys_node:
        raise CrudError(f"Family '{family}' not found in system '{system}'", "not_found")

    affected = _ml_count_channels_in_family(sys_node[family])
    del sys_node[family]

    _atomic_write(Path(db_path), data)
    _reload_registry("middle_layer")

    return {"success": True, "system": system, "family": family, "affected_channels": affected}


def ml_add_channel(
    db_path: str | Path,
    system: str,
    family: str,
    field: str,
    channel_name: str,
    subfield: str | None = None,
) -> dict:
    """Add a channel to a family's field.

    Creates the field/subfield path if it doesn't exist.

    Args:
        db_path: Path to the middle layer JSON database.
        system: System name.
        family: Family name.
        field: Field name (e.g., "Monitor").
        channel_name: Channel PV name to add.
        subfield: Optional subfield name.

    Returns:
        Success dict.
    """
    data = _load_json(Path(db_path))
    fam_node = _ml_navigate(data, system, family)

    # Navigate/create to the target field node
    if field not in fam_node:
        fam_node[field] = {}
    target = fam_node[field]

    if subfield is not None:
        if not isinstance(target, dict):
            raise CrudError(f"Field '{field}' is not a dict, cannot add subfield", "invalid_path")
        if subfield not in target:
            target[subfield] = {}
        target = target[subfield]

    # Add channel to ChannelNames list
    if "ChannelNames" not in target:
        target["ChannelNames"] = []
    names = target["ChannelNames"]
    if isinstance(names, str):
        names = [names]
        target["ChannelNames"] = names

    if channel_name in names:
        raise CrudError(f"Channel '{channel_name}' already exists in this field", "duplicate")

    names.append(channel_name)

    _atomic_write(Path(db_path), data)
    _reload_registry("middle_layer")

    return {"success": True, "channel": channel_name, "path": f"{system}:{family}:{field}"}


def ml_delete_channel(
    db_path: str | Path,
    system: str,
    family: str,
    field: str,
    channel_name: str,
    subfield: str | None = None,
) -> dict:
    """Delete a channel from a family's field.

    Returns:
        Success dict.
    """
    data = _load_json(Path(db_path))
    fam_node = _ml_navigate(data, system, family)

    if field not in fam_node:
        raise CrudError(f"Field '{field}' not found in family '{family}'", "not_found")

    target = fam_node[field]
    if subfield is not None:
        if not isinstance(target, dict) or subfield not in target:
            raise CrudError(f"Subfield '{subfield}' not found", "not_found")
        target = target[subfield]

    names = target.get("ChannelNames", [])
    if isinstance(names, str):
        names = [names]

    if channel_name not in names:
        raise CrudError(f"Channel '{channel_name}' not found", "not_found")

    names.remove(channel_name)
    target["ChannelNames"] = names

    _atomic_write(Path(db_path), data)
    _reload_registry("middle_layer")

    return {"success": True, "channel": channel_name}


def ml_count_family_channels(
    db_path: str | Path,
    system: str,
    family: str,
) -> int:
    """Count channels in a family (for delete impact preview).

    Returns:
        Number of channels.
    """
    data = _load_json(Path(db_path))
    fam_node = _ml_navigate(data, system, family)
    return _ml_count_channels_in_family(fam_node)


# ---------------------------------------------------------------------------
# In-Context CRUD — operates on flat channel list
# ---------------------------------------------------------------------------


def _ic_load_channels(data: dict | list) -> list:
    """Extract the channels list from in-context data (handles dict or list)."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("channels", [])
    return []


def _ic_set_channels(data: dict | list, channels: list) -> dict | list:
    """Return the data with channels updated."""
    if isinstance(data, dict):
        data["channels"] = channels
        return data
    return channels


def ic_add_channel(
    db_path: str | Path,
    channel: str,
    address: str = "",
    description: str = "",
) -> dict:
    """Add a channel to the in-context database.

    Args:
        db_path: Path to the in-context JSON database.
        channel: Channel name.
        address: PV address (defaults to channel name).
        description: Human-readable description.

    Returns:
        Success dict.
    """
    data = _load_json(Path(db_path))
    channels = _ic_load_channels(data)

    # Check for duplicates
    for ch in channels:
        name = ch.get("channel") or ch.get("name", "")
        if name == channel:
            raise CrudError(f"Channel '{channel}' already exists", "duplicate")

    new_entry = {"channel": channel, "address": address or channel}
    if description:
        new_entry["description"] = description

    channels.append(new_entry)
    data = _ic_set_channels(data, channels)

    _atomic_write(Path(db_path), data)
    _reload_registry("in_context")

    return {"success": True, "channel": channel}


def ic_delete_channel(
    db_path: str | Path,
    channel: str,
) -> dict:
    """Delete a channel from the in-context database.

    Args:
        db_path: Path to the in-context JSON database.
        channel: Channel name to delete.

    Returns:
        Success dict.
    """
    data = _load_json(Path(db_path))
    channels = _ic_load_channels(data)

    original_len = len(channels)
    channels = [ch for ch in channels if (ch.get("channel") or ch.get("name", "")) != channel]

    if len(channels) == original_len:
        raise CrudError(f"Channel '{channel}' not found", "not_found")

    data = _ic_set_channels(data, channels)

    _atomic_write(Path(db_path), data)
    _reload_registry("in_context")

    return {"success": True, "channel": channel}


def ic_update_channel(
    db_path: str | Path,
    channel: str,
    new_description: str | None = None,
    new_address: str | None = None,
) -> dict:
    """Update a channel's description and/or address.

    Args:
        db_path: Path to the in-context JSON database.
        channel: Channel name to update.
        new_description: New description (if not None).
        new_address: New PV address (if not None).

    Returns:
        Success dict.
    """
    data = _load_json(Path(db_path))
    channels = _ic_load_channels(data)

    found = False
    for ch in channels:
        name = ch.get("channel") or ch.get("name", "")
        if name == channel:
            if new_description is not None:
                ch["description"] = new_description
            if new_address is not None:
                ch["address"] = new_address
            found = True
            break

    if not found:
        raise CrudError(f"Channel '{channel}' not found", "not_found")

    data = _ic_set_channels(data, channels)

    _atomic_write(Path(db_path), data)
    _reload_registry("in_context")

    return {"success": True, "channel": channel}
