"""Profile inheritance merging and content hashing.

Resolves the ``extends`` chain into a single raw YAML dict — deep-merging each
layer, applying ``exclude:`` list subtraction, and detecting circular
references — and derives the canonical content hashes stamped into build
manifests. Hashing lives here rather than beside preset discovery because it
hashes the *resolved* profile, which requires ``extends`` resolution; keeping it
here leaves :mod:`osprey.cli.build_profile_presets` a leaf.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from osprey.errors import BuildProfileError

from .build_profile_presets import _load_preset_raw, _preset_exists, list_presets

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profile inheritance helpers
# ---------------------------------------------------------------------------


def _merge_lists(base: list, child: list) -> list:
    """Merge two YAML lists.

    String lists: union with dedup, base order preserved.
    Other lists (e.g. lifecycle step dicts): concatenate.
    """
    if not base and not child:
        return []
    all_items = base + child
    if all(isinstance(x, str) for x in all_items):
        seen: set[str] = set()
        merged: list[str] = []
        for item in all_items:
            if item not in seen:
                seen.add(item)
                merged.append(item)
        return merged
    return list(base) + list(child)


def _deep_merge(base: dict, child: dict) -> dict:
    """Deep-merge two raw YAML profile dicts (child wins on conflict)."""
    merged = dict(base)
    for key, child_val in child.items():
        if key not in base:
            merged[key] = child_val
        else:
            base_val = base[key]
            if isinstance(base_val, dict) and isinstance(child_val, dict):
                merged[key] = _deep_merge(base_val, child_val)
            elif isinstance(base_val, list) and isinstance(child_val, list):
                merged[key] = _merge_lists(base_val, child_val)
            else:
                merged[key] = child_val
    return merged


# String-list profile fields that ``exclude:`` may subtract from. Deliberately
# excludes dict-shaped fields (config, overlay, mcp_servers, services, ...) —
# list subtraction only makes sense for the plain string collections a child
# inherits via ``extends``.
_EXCLUDABLE_FIELDS: frozenset[str] = frozenset(
    {
        "skills",
        "rules",
        "hooks",
        "agents",
        "output_styles",
        "web_panels",
        "dependencies",
    }
)


def _apply_exclude(merged: dict[str, Any], exclude: Any) -> None:
    """Subtract ``exclude`` entries from the string-list fields of ``merged`` in place.

    ``exclude`` is a mapping of field name (one of :data:`_EXCLUDABLE_FIELDS`) to a
    list of entries to remove. Excluding an entry that is not present is a silent
    no-op. Because this runs after each ``_deep_merge`` in :func:`_resolve_extends`,
    a deeper ``extends`` layer that re-adds an entry merges in afterwards and wins;
    an entry re-added by an override file or ``--set`` merges *before* extends
    resolution and is stripped again here, so it cannot win.

    Args:
        merged: The merged raw profile dict (mutated in place).
        exclude: The raw ``exclude`` value from a profile layer.

    Raises:
        BuildProfileError: If ``exclude`` is not a mapping, names an unknown or
            non-list-shaped field, or maps a field to a non-list value.
    """
    if not isinstance(exclude, dict):
        raise BuildProfileError(
            f"Profile 'exclude' must be a mapping of field name to list "
            f"(got {type(exclude).__name__})"
        )
    for field_name, entries in exclude.items():
        if field_name not in _EXCLUDABLE_FIELDS:
            raise BuildProfileError(
                f"exclude: unknown or non-list field {field_name!r} "
                f"(must be one of {sorted(_EXCLUDABLE_FIELDS)})"
            )
        if not isinstance(entries, list):
            raise BuildProfileError(
                f"exclude.{field_name} must be a list of entries to remove "
                f"(got {type(entries).__name__})"
            )
        current = merged.get(field_name)
        if not isinstance(current, list):
            continue
        removal = set(entries)
        merged[field_name] = [item for item in current if item not in removal]


def _resolve_extends(
    raw: dict[str, Any], profile_path: Path, chain: list[Path] | None = None
) -> dict[str, Any]:
    """Resolve ``extends`` chain, returning a fully merged raw YAML dict.

    Args:
        raw: The raw YAML dict from the current file.
        profile_path: Resolved path to the current YAML file.
        chain: Paths already visited (for circular-reference detection).

    Returns:
        Merged raw dict with ``extends`` consumed.

    Raises:
        BuildProfileError: On missing base, circular reference, or bad YAML.
    """
    if chain is None:
        chain = []

    resolved = profile_path.resolve()
    if resolved in chain:
        cycle = " -> ".join(str(p) for p in chain) + f" -> {resolved}"
        raise BuildProfileError(f"Circular extends detected: {cycle}")
    chain.append(resolved)

    extends_value = raw.pop("extends", None)
    if extends_value is None:
        # No base to subtract from — ``exclude`` here can only touch this file's
        # own declarations, which is an author mistake. Apply-to-self (a no-op in
        # practice) and log so it's discoverable, matching the recursive path's
        # "pop exclude before returning" contract.
        exclude_value = raw.pop("exclude", None)
        if exclude_value is not None:
            _LOGGER.debug(
                "Profile %s declares 'exclude' without 'extends'; it can only "
                "affect its own declarations (no inherited entries to remove).",
                resolved,
            )
            _apply_exclude(raw, exclude_value)
        return raw

    # Try a bundled preset by name first; fall through to filesystem-path
    # resolution. Path-shaped values like ``als-base.yml`` correctly miss the
    # preset probe (it looks up ``als-base.yml.yml``) and resolve as paths,
    # preserving the sibling-file semantics ALS-style profiles depend on.
    preset_path = _preset_exists(extends_value)
    if preset_path is not None:
        base_path = preset_path
    else:
        base_path = (profile_path.parent / extends_value).resolve()
        if not base_path.exists():
            available = ", ".join(list_presets()) or "(none)"
            raise BuildProfileError(
                f"Cannot resolve extends: {extends_value!r}. "
                f"No bundled preset by that name (available: {available}), "
                f"and no file at {base_path}."
            )

    try:
        base_raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise BuildProfileError(f"Invalid YAML in {base_path}: {e}") from e

    if not isinstance(base_raw, dict):
        raise BuildProfileError(f"Extended profile must be a YAML mapping: {base_path}")

    # Recurse: the base may itself extend another profile
    base_raw = _resolve_extends(base_raw, base_path, chain)

    merged = _deep_merge(base_raw, raw)
    # Apply this layer's ``exclude`` to the merged result and consume it. The
    # recursively-resolved ``base_raw`` has already had its own ``exclude``
    # popped, so the only ``exclude`` present here is this layer's own.
    exclude_value = merged.pop("exclude", None)
    if exclude_value is not None:
        _apply_exclude(merged, exclude_value)
    return merged


# ---------------------------------------------------------------------------
# Profile content hashing
# ---------------------------------------------------------------------------


def _hash_resolved_profile(raw: dict[str, Any], profile_path: Path) -> str:
    """Canonical content hash of a profile dict after ``extends`` resolution.

    Hashes the *resolved* content (canonical JSON, sorted keys) rather than
    file bytes, so comment/ordering churn is invisible while a change in any
    ``extends`` parent is not.
    """
    import hashlib
    import json

    resolved = _resolve_extends(dict(raw), profile_path)
    canonical = json.dumps(resolved, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def compute_preset_hash(preset_name: str) -> str | None:
    """Content hash of a bundled preset as resolved (post-``extends``).

    Stamped into ``.osprey-manifest.json`` at build time and compared by the
    deploy-side staleness advisory
    (:mod:`osprey.deployment.staleness`). Returns ``None`` when the preset is
    unknown or unreadable — callers treat that as "cannot compare", never as
    drift.
    """
    try:
        raw, path = _load_preset_raw(preset_name)
        return _hash_resolved_profile(raw, path)
    except Exception:
        return None


def compute_profile_hash(profile_path: Path) -> str | None:
    """Content hash of a positional profile YAML as resolved (post-``extends``).

    Counterpart of :func:`compute_preset_hash` for ``osprey build NAME
    PROFILE.yml`` invocations. Returns ``None`` when the file is missing or
    unreadable.
    """
    try:
        path = Path(profile_path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        return _hash_resolved_profile(raw, path)
    except Exception:
        return None
