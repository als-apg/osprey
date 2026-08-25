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
    config_add_to_list(Path("config.yml"), ["scaffold", "user_owned"], "rules/facility")

    # Apply structured key-value updates
    config_update_fields(Path("config.yml"), {
        "control_system.writes_enabled": True,
        "approval.tools.channel_write": "always",
    })
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML, CommentedMap, CommentedSeq
from ruamel.yaml.error import CommentMark
from ruamel.yaml.tokens import CommentToken

logger = logging.getLogger(__name__)

# Round-trip mode ("rt") preserves comments, key order, and quoting style.
_yaml = YAML(typ="rt")
_yaml.preserve_quotes = True
_yaml.width = 4096  # prevent aggressive line-wrapping
# Emit block sequences indented under their key, which is how every YAML this
# framework generates is written. ruamel's default (offset 0) would pull each
# list item back to its parent's column on the first write, so a one-key
# `osprey set` reflowed ~80 lines of a hand-formatted, git-tracked profile —
# comments preserved, but every list in the file re-indented around them.
_yaml.indent(mapping=2, sequence=4, offset=2)


# =============================================================================
# Section-Comment Anchoring
# =============================================================================
# ruamel attaches a comment block that sits between two YAML sections to the
# deepest last node of the PRECEDING section — not to the key that follows it.
# A plain append to that section's list/mapping therefore renders the new
# entry *after* the comment block: a list split in two around a section
# banner, or a new service block emitted below the next section's header.
# The helpers below detach such a trailing block before appending and
# re-attach it to the new last entry, so section banners stay at section
# boundaries. Inline comments (same line as a value) are never moved.


def _deepest_trailing_slot(container: Any) -> tuple[Any, Any, int] | None:
    """Locate the comment slot trailing *container*'s last entry.

    Descends through nested non-empty containers, because that is where ruamel
    parks a section-trailing comment block. Returns ``(owner, key_or_index,
    slot_index)`` addressing ``owner.ca.items[key_or_index][slot_index]``, or
    None for empty/non-commented containers.
    """
    while True:
        if isinstance(container, CommentedMap) and len(container):
            last: Any = next(reversed(container))
            slot = 2  # eol/post-value comment slot for mappings
        elif isinstance(container, CommentedSeq) and len(container):
            last = len(container) - 1
            slot = 0  # post-item comment slot for sequences
        else:
            return None
        value = container[last]
        if isinstance(value, (CommentedMap, CommentedSeq)) and len(value):
            container = value
            continue
        return container, last, slot


def _steal_section_comment(container: Any) -> str | None:
    """Detach and return the comment block trailing *container*'s last entry.

    An inline comment on the entry's own line stays put; only the block that
    follows on subsequent lines is removed and returned (with its leading
    newline, ready for re-attachment). Returns None when there is nothing to
    move.
    """
    found = _deepest_trailing_slot(container)
    if found is None:
        return None
    owner, last, slot = found
    entry = owner.ca.items.get(last)
    token = entry[slot] if entry else None
    if token is None:
        return None
    text = token.value
    newline = text.find("\n")
    if newline == -1 or not text[newline + 1 :].strip():
        return None  # inline-only comment — nothing trails onto later lines
    first_line, rest = text[: newline + 1], text[newline + 1 :]
    if first_line.strip():
        # Keep the inline part; the extra "\n" restores the line terminator
        # the inline comment's token no longer provides at the new location.
        token.value = first_line
        return "\n" + rest
    entry[slot] = None
    if not any(entry):
        del owner.ca.items[last]
    return text


def _attach_section_comment(container: Any, text: str) -> None:
    """Attach *text* as the comment block trailing *container*'s last entry."""
    found = _deepest_trailing_slot(container)
    if found is None:
        return
    owner, last, slot = found
    if not text.startswith("\n"):
        text = "\n" + text
    entry = owner.ca.items.setdefault(last, [None, None, None, None])
    if entry[slot] is not None:
        entry[slot].value += text
    else:
        entry[slot] = CommentToken(text, CommentMark(0), None)


def _to_commented(obj: Any) -> Any:
    """Deep-convert plain dicts/lists so comment slots exist along the spine."""
    if isinstance(obj, dict) and not isinstance(obj, CommentedMap):
        return CommentedMap((k, _to_commented(v)) for k, v in obj.items())
    if isinstance(obj, list) and not isinstance(obj, CommentedSeq):
        return CommentedSeq(_to_commented(item) for item in obj)
    return obj


def anchored_append(seq: Any, value: Any) -> None:
    """Append *value* to *seq*, keeping a trailing section comment at the end.

    Drop-in replacement for ``seq.append(value)`` on a round-trip-loaded
    sequence whose tail may carry the comment block introducing the next
    section. Plain lists are appended to unchanged.
    """
    if not isinstance(seq, CommentedSeq) or not len(seq):
        seq.append(value)
        return
    moved = _steal_section_comment(seq)
    seq.append(_to_commented(value))
    if moved:
        _attach_section_comment(seq, moved)


def anchored_put(mapping: Any, key: Any, value: Any) -> None:
    """Set ``mapping[key] = value``, keeping a trailing section comment at the end.

    Drop-in replacement for item assignment on a round-trip-loaded mapping:
    when *key* is new (ruamel appends it after the last entry — and after any
    comment block trailing that entry), the block is re-anchored below the new
    entry. Existing keys are updated in place with no layout change — except
    the last key, whose replacement value would silently discard a comment
    block parked inside the old value; that block is carried over. *value*
    must be complete — an empty dict/list cannot carry the re-anchored
    comment beneath it, so such values are assigned without relocation.
    """
    if not isinstance(mapping, CommentedMap) or not len(mapping):
        mapping[key] = value
        return
    if key in mapping and next(reversed(mapping)) != key:
        mapping[key] = value
        return
    value = _to_commented(value)
    if isinstance(value, (CommentedMap, CommentedSeq)) and not len(value):
        mapping[key] = value
        return
    moved = _steal_section_comment(mapping)
    mapping[key] = value
    if moved:
        _attach_section_comment(mapping, moved)


# =============================================================================
# Core I/O
# =============================================================================


#: Directory, under the agent-data (state) root, holding config backups.
#:
#: Config backups deliberately do NOT sit beside the file they copy. A project's
#: ``config.yml`` lives in the render, and the render becomes root-owned when the
#: container split lands: creating a *new* file next to it needs write permission
#: on the render directory, which the admin image will not have. Every backup
#: here runs *before* its writer touches anything, so under the old sibling
#: scheme the ``PermissionError`` would have turned each of these config writes
#: into a 500 in that image, without a byte being written. The state zone is the
#: tree that stays writable across the split and survives a rebuild, so backups
#: belong there. This is a hard precondition for the split, not a tidy-up.
#:
#: The zone is never spelled literally: it is read from ``agent_data.base_dir``
#: through :func:`agent_data_base_dir` and anchored with :func:`anchored_path`,
#: so a project that relocates its state zone moves its config backups with it.
CONFIG_BACKUP_DIRNAME = "config-backups"


def config_backup_path(config_path: Path) -> Path:
    """Where a backup of *config_path* belongs: inside the agent-data state zone.

    The single spelling of the backup scheme. Three writers take config backups
    — this module's :func:`update_yaml_file`, the Web Terminal's ``/api/config``
    and ARIEL's ``PUT /config`` — and each of them writes into a project that the
    container split will make read-only. One helper so the three cannot drift on
    a location that has to be right in all three for the split to work at all.

    **The anchor is a fact about the config file, not a caller's choice.**
    ``agent_data.base_dir`` is documented and rendered relative to the deployment
    repo ROOT — the directory holding ``profile.yml`` and ``var/`` — never
    relative to the render that ``config.yml`` sits in. So the anchor is derived
    from the config itself, applying the same precedence
    :func:`~osprey.utils.workspace.resolve_project_root` applies at runtime: the
    config's own ``project_root`` key when it names a directory that exists
    here, and :func:`repo_root_for_config` otherwise. That is the helper every
    other state-zone consumer resolves the root with:

    * flat layout (``<project>/config.yml``) → ``<project>``, whose state zone
      is ``<project>/var/agent_data``;
    * deployment repo (``<repo>/build/config.yml``) → ``<repo>``, so the backup
      lands in the durable ``var/`` and not in the disposable render;
    * container (``/app/<project>/build/config.yml``) → ``/app/<project>``,
      which is exactly the directory ``Dockerfile.j2`` pre-creates and chowns
      ``var/agent_data/config-backups`` under.

    That last one is why this takes no ``project_dir`` override any more. A
    caller passing the *render* directory (the web panel's ``project_cwd``) made
    the route mkdir ``<render>/var`` inside the root-owned render zone, which
    fails with ``PermissionError`` before a byte of the save is written — the
    admin Config panel could not save at all. One anchor rule, derived from the
    one input every caller already holds, so no caller can disagree with the
    runtime about where the state zone is.

    The filename keeps the source file's own name, so two configs backed up into
    one zone cannot land on each other; there is one slot per name, overwritten
    on each save, which is the retention the sibling scheme had.

    Args:
        config_path: The config file that would be backed up.

    Returns:
        The path the backup should be written to. Nothing is created.
    """
    from osprey.utils.workspace import agent_data_base_dir, anchored_path, repo_root_for_config

    try:
        config = _load(config_path)
    except Exception:  # noqa: BLE001 -- an unreadable config still deserves a backup
        # Falls back to the framework default rather than to the file's own
        # directory: a config too broken to name its own state zone is precisely
        # the one worth having a copy of, and the default is where every other
        # reader would look for that copy anyway.
        config = None

    # The SAME precedence workspace.resolve_project_root applies, in the same
    # order: the config's own ``project_root`` when it names a directory that
    # exists here, and filesystem truth otherwise. Mirrored rather than
    # delegated because resolve_project_root() re-reads the ambient config
    # (OSPREY_CONFIG / the process-global cache), which would anchor a panel
    # driving another project on whichever project this PROCESS was started in.
    # The config is already loaded above, so the key can be read from it
    # directly and the two cannot disagree.
    #
    # They diverge on exactly one shape, and it is not hypothetical: a project
    # directory literally NAMED ``build`` reads as a render to
    # repo_root_for_config (its documented non-goal), so without this the
    # backup would land one level above the project while every runtime reader
    # of the state zone looked inside it.
    configured = (config or {}).get("project_root")
    candidate = (
        Path(configured).expanduser() if isinstance(configured, str) and configured else None
    )
    root = (
        candidate
        if candidate is not None and candidate.is_dir()
        else repo_root_for_config(config_path)
    )

    zone = anchored_path(agent_data_base_dir(config), root)
    return zone / CONFIG_BACKUP_DIRNAME / f"{config_path.name}.bak"


def write_config_backup(config_path: Path) -> Path:
    """Copy *config_path* into the state zone and return where it landed.

    Not guarded: a backup that cannot be taken must stop the write, which is the
    contract every caller already had. The failure mode this location exists to
    prevent is exactly that stop happening for a reason nobody can fix — see
    :data:`CONFIG_BACKUP_DIRNAME`.

    Args:
        config_path: The config file to copy.

    Returns:
        Path to the backup that was written.
    """
    backup_path = config_backup_path(config_path)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, backup_path)
    return backup_path


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


def _ensure_parent_node(data: Any, key_path: list[str]) -> Any:
    """Walk *key_path*'s parent keys from *data*, creating empty maps for any missing.

    Shared by every writer that needs to address a leaf under a key_path that
    may not exist yet (:func:`config_add_to_list`, :func:`config_replace_list`)
    — as opposed to :func:`config_remove_from_list`, which must NOT create
    missing parents (removing from a list that was never there is a no-op,
    not a reason to create the empty scaffolding for it).

    Args:
        data: The loaded YAML document (or any nested mapping within it).
        key_path: List of nested keys; only ``key_path[:-1]`` (the parents)
            are walked/created — the leaf itself is left for the caller.

    Returns:
        The parent node ``key_path[:-1]`` resolves to.
    """
    node = data
    for key in key_path[:-1]:
        if key not in node:
            node[key] = {}
        node = node[key]
    return node


def config_add_to_list(
    config_path: Path,
    key_path: list[str],
    value: str,
) -> bool:
    """Append *value* to a YAML list at *key_path*, creating parents if needed.

    Args:
        config_path: Path to the YAML file.
        key_path: List of nested keys, e.g. ``["scaffold", "user_owned"]``.
        value: The scalar value to append.

    Returns:
        True if the value was added, False if it was already present.
    """
    data = _load(config_path)
    node = _ensure_parent_node(data, key_path)

    leaf = key_path[-1]
    if leaf not in node:
        node[leaf] = []

    lst = node[leaf]
    if value in lst:
        return False

    anchored_append(lst, value)
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
    logger.debug("config_remove_from_list: %s -= %s in %s", ".".join(key_path), value, config_path)
    return True


# =============================================================================
# Field Updates
# =============================================================================


def config_replace_list(
    config_path: Path,
    key_path: list[str],
    new_list: list[Any],
) -> None:
    """Replace an entire YAML list at *key_path* with *new_list*, preserving comments.

    :func:`config_add_to_list` and :func:`config_remove_from_list` mutate a list of
    *scalars* in place via ``value in lst`` membership tests, which never match a
    dict against a string — so neither can edit a list of *objects* (e.g. explicit
    ``{"name": ..., "index": ...}`` roster entries). This function sidesteps that by
    replacing the whole list value at once; the surrounding document (other keys,
    comments) is untouched because only the addressed subtree is reassigned before
    the round-trip save.

    Args:
        config_path: Path to the YAML file.
        key_path: List of nested keys, e.g. ``["modules", "web_terminals", "users"]``.
            Parent keys are created as empty maps if missing, matching
            :func:`config_add_to_list`.
        new_list: The full replacement list (scalars, or nested dicts/lists).
    """
    data = _load(config_path)
    node = _ensure_parent_node(data, key_path)

    node[key_path[-1]] = new_list
    _save(config_path, data)
    logger.debug(
        "config_replace_list: %s = <%d item(s)> in %s",
        ".".join(key_path),
        len(new_list),
        config_path,
    )


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
        _set_dotted_anchored(data, data, dotted_key, value, create_only=True)

    _save(config_path, data)
    logger.debug("config_update_fields: updated %d field(s) in %s", len(updates), config_path)


def _set_dotted_anchored(
    root: Any, data: Any, dotted_key: str, value: Any, *, create_only: bool
) -> None:
    """Set a dotted key, re-anchoring a section comment displaced by new keys.

    Walks ``dotted_key`` from *data*, creating missing intermediate maps. The
    first key created inside an existing non-root mapping is where a trailing
    section comment would be displaced — it is detached there, and re-attached
    once the full subtree exists. New keys at the *root* mapping keep their
    tail-append placement: the shipped templates document appended top-level
    sections at the file tail, below banners written for exactly that purpose.

    Args:
        root: The document's top-level mapping (anchor exemption).
        data: Mapping to walk from (the root itself, or a nested map).
        dotted_key: ``"dot.separated.key"`` path.
        value: Value to assign at the leaf.
        create_only: If True, walk into whatever an existing intermediate key
            holds (:func:`config_update_fields` semantics); if False, replace
            non-mapping intermediates with empty maps
            (:func:`_set_nested_value` semantics).
    """
    parts = dotted_key.split(".")
    node = data
    moved: str | None = None
    anchor: Any = None

    def _steal_here(current: Any) -> str | None:
        if current is root or not isinstance(current, CommentedMap) or not len(current):
            return None
        return _steal_section_comment(current)

    for part in parts[:-1]:
        if part not in node:
            if moved is None:
                moved = _steal_here(node)
                anchor = node if moved else None
            node[part] = CommentedMap()
        elif not create_only and not isinstance(node[part], dict):
            node[part] = CommentedMap()
        node = node[part]

    leaf = parts[-1]
    if leaf not in node and moved is None:
        moved = _steal_here(node)
        anchor = node if moved else None
    node[leaf] = _to_commented(value)

    if moved:
        _attach_section_comment(anchor, moved)


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
        create_backup: If True, copies the file into the agent-data state zone
                before modifying it (see :func:`config_backup_path`)
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
        # Into the state zone, not beside the file -- see CONFIG_BACKUP_DIRNAME.
        backup_path = write_config_backup(file_path)

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


def _apply_nested_updates(data: dict, updates: dict, root: Any = None) -> None:
    """Apply nested updates to a dictionary, supporting dot notation keys.

    Args:
        data: Target dictionary to update (modified in place)
        updates: Updates to apply
        root: The document's top-level mapping, threaded through recursion for
            section-comment anchoring (defaults to *data*).
    """
    if root is None:
        root = data
    for key, value in updates.items():
        if "." in key:
            _set_nested_value(data, key, value, root=root)
        elif isinstance(value, dict) and key in data and isinstance(data[key], dict):
            _apply_nested_updates(data[key], value, root=root)
        elif data is root:
            data[key] = value
        else:
            anchored_put(data, key, value)


def _set_nested_value(data: dict, path: str, value: Any, root: Any = None) -> None:
    """Set a value at a nested path using dot notation.

    Args:
        data: Target dictionary
        path: Dot-separated path (e.g., "control_system.connector.epics.port")
        value: Value to set
        root: The document's top-level mapping, for section-comment anchoring
            (defaults to *data*).
    """
    _set_dotted_anchored(root if root is not None else data, data, path, value, create_only=False)


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

    Retained as the config-side write path; the live CLI writes through the
    profile (``osprey set`` + ``osprey build``), so this is currently
    test-only.

    Uses comment-preserving YAML update via update_yaml_file().

    Args:
        config_path: Path to config.yml
        control_type: 'mock', 'epics', or 'virtual_accelerator'
        archiver_type: Optional archiver type ('mock_archiver', 'epics_archiver')
        create_backup: If True, copies the file into the agent-data state zone
                before modifying it (see :func:`config_backup_path`)

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
        create_backup: If True, copies the file into the agent-data state zone
                before modifying it (see :func:`config_backup_path`)

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
