"""Profile inheritance merging and content hashing.

Resolves the ``extends`` chain into a single raw YAML dict — deep-merging each
layer, applying ``exclude:`` list subtraction, and detecting circular
references — and derives the canonical content hashes stamped into build
manifests. Hashing lives here rather than beside preset discovery because it
hashes the *resolved* profile, which requires ``extends`` resolution; keeping it
here leaves :mod:`osprey.cli.build_profile_presets` a leaf.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from osprey.utils.workspace import LAYER_DIR_KINDS, PROFILE_FILENAME, RETIRED_LAYER_DIRS
from osprey_connectors.errors import BuildProfileError
from osprey_connectors.types import (
    configured_targets,
    resolve_target,
    tier_writes,
)

from . import build_profile_document as _document
from .build_profile_document import RENDER_KEYS, _read_profile_document, retired_write_origin
from .build_profile_presets import (
    PRESET_ONLY_KEYS,
    _load_preset_raw,
    _preset_exists,
    list_presets,
    resolve_triggers_path,
)
from .derived_keys import derived_key_errors
from .profile_root import ProfileKind, resolve_profile_root

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
# excludes ``config`` — list subtraction only makes sense for the plain string
# collections a child inherits via ``extends``, not for a mapping whose entries
# are key/value settings rather than names. The dict-shaped convention fields
# (``mcp_servers``, ``services``) are reachable through ``exclude:`` by a
# different route: _apply_exclude drops the named declaration alongside the
# files, which is why they are absent here rather than unreachable.
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


def _convention_source_for_field(field_name: str) -> str | None:
    """The convention directory, if any, an ``exclude:`` field names.

    Args:
        field_name: One key of an ``exclude:`` mapping.

    Returns:
        The matching entry of
        :data:`~osprey.cli.profile_conventions.CONVENTION_SOURCES`, or ``None``
        when the field names no convention directory. A profile spells its list
        fields with underscores (``output_styles``) while the directory is
        hyphenated (``output-styles/``), so an underscore fallback bridges the
        two; the identity match runs first, which keeps ``mcp_servers`` — whose
        directory really does carry an underscore — from being rewritten.
    """
    from .profile_conventions import CONVENTION_SOURCES

    if field_name in CONVENTION_SOURCES:
        return field_name
    hyphenated = field_name.replace("_", "-")
    return hyphenated if hyphenated in CONVENTION_SOURCES else None


# ``exclude:`` fields whose entries name built-in library artifacts rather than
# profile-authored files. A bare name in one of these profile lists must be one
# the built-in library carries (``validate_artifacts``); the profile's own
# ``agents/orbit-writer.md`` is selected by its directory-prefixed spelling,
# ``agents/orbit-writer``, in the same list. The two namespaces are disjoint,
# which is why the two ``exclude:`` spellings below have to mean different
# things.
#
# ``hooks`` is the one field whose two namespaces cannot collide by accident:
# the library selects a hook by a readable short name (``writes-check``) while
# a profile-shipped one is named by the filename the render wires into
# ``settings.json`` (``hooks/osprey_writes_check.py``). The library probe below
# still decides, as it does for every other field — the two spellings simply
# cannot be confused for one another the way ``rules/safety`` and ``safety``
# can.
_ARTIFACT_LIBRARY_FIELDS: frozenset[str] = frozenset(
    {"hooks", "rules", "skills", "agents", "output_styles"}
)


def _selecting_list_field(source: str | None) -> str | None:
    """The profile list field that selects convention directory *source*, if any."""
    from .profile_conventions import SELECTED_CONVENTION_SOURCES

    return next(
        (field for field, dirname in SELECTED_CONVENTION_SOURCES.items() if dirname == source),
        None,
    )


def _is_library_selection(field_name: str, name: str) -> bool:
    """Whether ``name`` under ``field_name`` names a built-in library artifact.

    Args:
        field_name: The ``exclude:`` key the entry was listed under.
        name: The bare entry.

    Returns:
        ``True`` only for a field the built-in library governs whose entry that
        library actually carries. An unreadable library answers ``False``: the
        exclusion still applies, it is merely also recorded as a file omission,
        which is the harmless direction to be wrong in.
    """
    if field_name not in _ARTIFACT_LIBRARY_FIELDS:
        return False
    from .templates.artifact_library import list_artifacts

    try:
        return name in set(list_artifacts(field_name))
    except Exception:
        return False


def _split_exclude_entry(field_name: str, source: str | None, entry: Any) -> tuple[str, bool]:
    """Split one ``exclude:`` entry into its bare name and its spelling.

    The spelling is the contract, because the two things a profile can select
    by name live in disjoint namespaces. ``channel-finder`` is a built-in
    library artifact the profile *selects*; ``agents/channel-finder`` is a file
    the profile *ships*, which shadows the built-in one. Excluding the shadow
    must leave the selection standing — that is what makes the built-in version
    render again — so the two spellings cannot be treated as equivalent.

    Args:
        field_name: The ``exclude:`` key the entry was listed under.
        source: That key's convention directory, or ``None``.
        entry: The raw list entry.

    Returns:
        ``(name, qualified)`` — the bare name, and whether it was written with
        its convention directory as a prefix.

    Raises:
        BuildProfileError: If the entry is not a string, is qualified with a
            *different* convention directory than the key it sits under, or is
            a prefix naming nothing. Each of those would otherwise exclude the
            wrong artifact, or nothing at all, while looking like it worked.
    """
    if not isinstance(entry, str):
        raise BuildProfileError(
            f"exclude.{field_name}: entries must be strings naming one artifact "
            f"(got {type(entry).__name__}: {entry!r})"
        )
    head, sep, tail = entry.partition("/")
    if not sep:
        return entry, False
    if source is not None and head in (source, field_name):
        if not tail:
            raise BuildProfileError(
                f"exclude.{field_name}: entry {entry!r} names no artifact after the "
                f"{head + '/'!r} prefix."
            )
        return tail, True
    # Checked for every field, not only convention-backed ones: a qualified
    # entry under an unrelated key (``hooks: [agents/x]``) is the same mistake
    # and must not pass as a literal hook name that matches nothing.
    if _convention_source_for_field(head) is not None:
        raise BuildProfileError(
            f"exclude.{field_name}: entry {entry!r} names a {head!r} artifact under "
            f"the {field_name!r} key. List it under {head!r} instead, or drop the "
            f"{head + '/'!r} prefix if you did mean a {field_name!r} entry."
        )
    return entry, False


def _apply_config_exclude(merged: dict[str, Any], entries: Any) -> None:
    """Subtract ``exclude.config`` entries from list-valued ``config:`` keys.

    Each key names a ``config:`` entry by its dotted spelling and maps to the
    items to remove from the list held there. A key the merged config does not
    carry is a silent no-op, like any other absent exclusion; a key that is
    present but holds something other than a list is refused, because the
    profile would then be making two statements about one fact.

    Args:
        merged: The merged raw profile dict (mutated in place).
        entries: The raw ``exclude.config`` value.

    Raises:
        BuildProfileError: If ``entries`` is not a mapping, maps a key to a
            non-list, or names a key whose inherited value is not a list.
    """
    if not isinstance(entries, dict):
        raise BuildProfileError(
            "exclude.config must be a mapping of dotted config key to the list of "
            f"entries to remove (got {type(entries).__name__})"
        )
    config = merged.get("config")
    for key, removal in entries.items():
        if not isinstance(removal, list):
            raise BuildProfileError(
                f"exclude.config.{key} must be a list of entries to remove "
                f"(got {type(removal).__name__})"
            )
        if not isinstance(config, dict) or key not in config:
            continue
        current = config[key]
        if not isinstance(current, list):
            raise BuildProfileError(
                f"exclude.config.{key}: the inherited value is not a list "
                f"(got {type(current).__name__}). exclude: takes entries out of a "
                "list; to change a scalar, state it under config: instead."
            )
        config[key] = [item for item in current if item not in removal]


def _refuse_excluded_helpers(names: Iterable[str]) -> None:
    """Refuse a bare helper under ``exclude: hooks:``.

    A helper is never selected, so there is nothing to subtract: it ships with
    the hooks that need it. A profile's own copy is omitted by its qualified
    spelling (``hooks/<name>``), which this leaves alone.

    Raises:
        BuildProfileError: Naming the first helper among *names*.
    """
    from osprey.agent_runner.build_artifacts.catalog import BuildArtifactCatalog
    from osprey.cli.templates.artifact_library import helper_refusal

    helpers = BuildArtifactCatalog.default().helpers()
    for name in names:
        if f"hooks/{name}" in helpers:
            raise BuildProfileError(helper_refusal(f"hooks/{name}"))


def _apply_exclude(
    merged: dict[str, Any],
    exclude: Any,
    *,
    artifacts: set[str] | None = None,
    shadow_candidates: set[str] | None = None,
) -> None:
    """Subtract ``exclude`` entries from ``merged`` in place, recording artifacts.

    ``exclude`` is a mapping of field name to a list of entries to remove, and
    the SPELLING of each entry decides which of the two things it removes:

    * bare (``channel-finder``) subtracts from the string list
      (:data:`_EXCLUDABLE_FIELDS`), so a built-in artifact an ``extends`` base
      selected is no longer selected;
    * qualified (``agents/channel-finder``) records the entry in ``artifacts``
      as ``"<source>/<name>"`` so the build omits the profile's own file for
      it, and drops that same qualified selection from the list, leaving the
      bare library selection alone.

    Keeping them apart is what makes FR-10's promise true: a profile that ships
    ``agents/channel-finder.md`` shadows the built-in agent of that name, and
    excluding the shadow by its qualified spelling drops only the file, so the
    still-selected built-in version renders again. Conflating the two would
    delete both and the artifact would vanish entirely.

    A bare entry that the built-in library does *not* carry is recorded as a
    file omission too — it cannot be a library selection (``validate_artifacts``
    would reject it), so the only thing it can mean is the profile's own file.
    Dict-shaped fields (``mcp_servers``, ``services``) drop the declaration on
    either spelling: the declaration names the profile's own directory, so
    omitting the files while leaving it standing would deploy a server whose
    source is missing, and there is no built-in version to fall back to.

    ``config`` is the one key whose value is a mapping rather than a list: each
    entry names a list-valued ``config:`` key by its dotted spelling and lists
    the items to subtract from it (``exclude: {config: {deployed_services:
    [mongodb]}}``). That is how a layer — a host-variant overlay above all —
    deploys fewer services than the profile it sits on: the inheritance merge
    unions lists, so a layer cannot state a shorter one, and taking away has
    to be said. See :func:`_apply_config_exclude`.

    Excluding an entry that is not present is a silent no-op. Because this runs
    after each ``_deep_merge`` in :func:`_resolve_extends`, a deeper ``extends``
    layer that re-adds an entry merges in afterwards and wins; an entry re-added
    by an override file or ``--set`` merges *before* extends resolution and is
    stripped again here, so it cannot win.

    Args:
        merged: The merged raw profile dict (mutated in place).
        exclude: The raw ``exclude`` value from a profile layer.
        artifacts: Set collecting convention-artifact exclusions, mutated in
            place. ``None`` skips recording, for callers that only need list
            subtraction.
        shadow_candidates: Set collecting the *other* half — bare entries that
            named a built-in library artifact, as ``"<source>/<name>"`` — so a
            caller that knows the profile root can tell whether the profile also
            shadows that name (see :func:`_warn_shadowed_bare_exclusions`).
            Deliberately disjoint from ``artifacts``: these entries record no
            file omission, which is exactly why they need a separate diagnostic.

    Raises:
        BuildProfileError: If ``exclude`` is not a mapping, names a field that
            is neither an excludable list nor a convention directory, or maps a
            field to a non-list value.
    """
    from .profile_conventions import CONVENTION_SOURCES

    if not isinstance(exclude, dict):
        raise BuildProfileError(
            f"Profile 'exclude' must be a mapping of field name to list "
            f"(got {type(exclude).__name__})"
        )
    for field_name, entries in exclude.items():
        if field_name == "config":
            _apply_config_exclude(merged, entries)
            continue
        source = _convention_source_for_field(field_name)
        if field_name not in _EXCLUDABLE_FIELDS and source is None:
            raise BuildProfileError(
                f"exclude: unknown or non-list field {field_name!r} (must be one of "
                f"{sorted(_EXCLUDABLE_FIELDS)}, a convention directory: "
                f"{sorted(CONVENTION_SOURCES)}, or 'config' mapping a dotted key to "
                "the entries to remove from its list)"
            )
        if not isinstance(entries, list):
            raise BuildProfileError(
                f"exclude.{field_name} must be a list of entries to remove "
                f"(got {type(entries).__name__})"
            )
        split = [_split_exclude_entry(field_name, source, entry) for entry in entries]
        if field_name == "hooks":
            _refuse_excluded_helpers(name for name, qualified in split if not qualified)
        if source is not None:
            for name, qualified in split:
                # One library probe per entry: list_artifacts() walks a
                # directory, and both records below key off the same answer.
                library = not qualified and _is_library_selection(field_name, name)
                if artifacts is not None and not library:
                    artifacts.add(f"{source}/{name}")
                if shadow_candidates is not None and library:
                    shadow_candidates.add(f"{source}/{name}")

        current = merged.get(field_name)
        if isinstance(current, dict):
            for name, _qualified in split:
                current.pop(name, None)
        elif isinstance(current, list):
            removal = {name for name, qualified in split if not qualified}
            merged[field_name] = [item for item in current if item not in removal]
        # A qualified entry also takes the profile's own artifact out of the
        # list that selects it (`exclude: {skills: [skills/beam-check]}` drops
        # `skills/beam-check` from `skills:`), whichever spelling the key used.
        list_field = _selecting_list_field(source)
        selecting = merged.get(list_field) if list_field else None
        if list_field is not None and isinstance(selecting, list):
            qualified_removal = {f"{source}/{name}" for name, qualified in split if qualified}
            merged[list_field] = [item for item in selecting if item not in qualified_removal]


def _resolve_extends(
    raw: dict[str, Any],
    profile_path: Path,
    chain: list[Path] | None = None,
    *,
    artifacts: set[str] | None = None,
    shadow_candidates: set[str] | None = None,
    preset_lineage: list[str] | None = None,
) -> dict[str, Any]:
    """Resolve ``extends`` chain, returning a fully merged raw YAML dict.

    Args:
        raw: The raw YAML dict from the current file.
        profile_path: Resolved path to the current YAML file.
        chain: Paths already visited (for circular-reference detection).
        artifacts: Set collecting convention-artifact exclusions from every
            layer in the chain, mutated in place. ``None`` skips recording.
        shadow_candidates: Set collecting bare library-selection exclusions from
            every layer, mutated in place. ``None`` skips recording.
        preset_lineage: List collecting the bundled presets the chain extends,
            nearest layer first, mutated in place. ``None`` skips recording.

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
            _apply_exclude(
                raw, exclude_value, artifacts=artifacts, shadow_candidates=shadow_candidates
            )
        return raw

    # Try a bundled preset by name first; fall through to filesystem-path
    # resolution. Path-shaped values like ``site-base.yml`` correctly miss the
    # preset probe (it looks up ``site-base.yml.yml``) and resolve as paths, so
    # a profile that extends a sibling file keeps those semantics.
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

    base_raw = _read_profile_document(base_path)

    if not isinstance(base_raw, dict):
        raise BuildProfileError(f"Extended profile must be a YAML mapping: {base_path}")

    if preset_path is not None:
        # A bundled preset reached as a base is consumed exactly as one reached
        # by name is (:func:`~.build_profile_presets._load_preset_raw`), so an
        # inherited ``app_template:`` or ``facility:`` never becomes part of
        # what the child resolves to. Confined to the preset branch: the same
        # key in a hand-written parent profile is a profile key, and is refused
        # as one.
        for key in PRESET_ONLY_KEYS:
            base_raw.pop(key, None)
        # Consuming the keys would otherwise lose the one fact they carried:
        # which packaged data this profile's tree came from. Record the preset
        # instead, so a reader can ask
        # :func:`~.build_profile_presets.preset_data_bundle` the same question
        # the deep merge used to answer. Recorded before the recursion, so the
        # list runs nearest-first — the order the merge resolves by.
        if preset_lineage is not None:
            preset_lineage.append(str(extends_value))

    # Recurse: the base may itself extend another profile
    base_raw = _resolve_extends(
        base_raw,
        base_path,
        chain,
        artifacts=artifacts,
        shadow_candidates=shadow_candidates,
        preset_lineage=preset_lineage,
    )

    merged = _deep_merge(base_raw, raw)
    # Apply this layer's ``exclude`` to the merged result and consume it. The
    # recursively-resolved ``base_raw`` has already had its own ``exclude``
    # popped, so the only ``exclude`` present here is this layer's own.
    exclude_value = merged.pop("exclude", None)
    if exclude_value is not None:
        _apply_exclude(
            merged, exclude_value, artifacts=artifacts, shadow_candidates=shadow_candidates
        )
    return merged


def merge_layer_delta(
    root_resolved: dict[str, Any],
    delta: dict[str, Any],
    *,
    artifacts: set[str] | None = None,
    shadow_candidates: set[str] | None = None,
) -> dict[str, Any]:
    """Merge a layer over its resolved root profile.

    The implicit merge a ``tiers/<name>.yml`` file stands for: it carries
    only what differs from the profile it lives under, so it is meaningless on
    its own. The sequence mirrors one ``extends`` layer exactly — the delta's
    ``exclude:`` is consumed first, then the delta wins key-by-key over the
    root, then ``exclude:`` subtracts from the *merged* result — so a render
    resolves identically whether it is spelled as a delta or as an ``extends``
    child.

    Args:
        root_resolved: The root profile, already ``extends``-resolved.
        delta: The render's own layer.
        artifacts: Set collecting the delta's convention-artifact exclusions,
            mutated in place. ``None`` skips recording.
        shadow_candidates: Set collecting the delta's bare library-selection
            exclusions, mutated in place. ``None`` skips recording.

    Returns:
        A new merged dict; neither argument is mutated.
    """
    delta = dict(delta)
    exclude_value = delta.pop("exclude", None)
    merged = _deep_merge(root_resolved, delta)
    if exclude_value is not None:
        _apply_exclude(
            merged, exclude_value, artifacts=artifacts, shadow_candidates=shadow_candidates
        )
    return merged


def _profile_ships(root_dir: Path, artifact: str) -> bool:
    """Whether the profile carries its own file for a ``"<source>/<name>"`` entry.

    Args:
        root_dir: The profile root the exclusions anchor at.
        artifact: A recorded exclusion, ``"<source>/<name>"``.

    Returns:
        ``True`` when the profile ships that artifact. Markdown categories name
        the artifact without its extension; the directory-shaped ones name the
        directory itself, so both spellings are probed.

        File-shaped categories (``hooks/``) need a third probe. A profile ships
        a hook under the filename the render wires into ``settings.json``
        (``hooks/osprey_writes_check.py``), while the built-in library selects
        the same hook by a readable short name (``writes-check``) — so a bare
        library exclusion arrives here spelled neither way, and only the
        library's own resolution maps one to the other. A short name that
        resolves to nothing is not shipped, which is the answer that keeps the
        unmatched-exclusion warning loud about a misspelling.
    """
    from .profile_conventions import EntryShape, convention_for

    source, _, name = artifact.partition("/")
    base = root_dir / source / name
    if base.exists() or (base.parent / f"{base.name}.md").exists():
        return True

    convention = convention_for(source)
    if convention is None or convention.shape is not EntryShape.FILE:
        return False
    from .templates.artifact_library import resolve_artifact

    try:
        library_file = resolve_artifact(source, name)
    except Exception:
        return False
    return (base.parent / library_file.name).is_file()


def _warn_shadowed_bare_exclusions(root_dir: Path, shadow_candidates: set[str]) -> None:
    """Warn when a bare exclusion names a library artifact the profile shadows.

    The one exclusion mistake neither :func:`_warn_unmatched_exclusions` nor the
    build itself can see. A bare entry unselects the built-in artifact, and that
    is all it does — so when the profile *also* ships a file of that name, the
    file keeps rendering and the project comes out byte-identical. Nothing is
    recorded in the exclusion record (by design: no file is omitted), so the
    unmatched-exclusion warning has nothing to look at, and the author reads a
    successful build as a successful exclusion.

    The remedy is the other spelling, so the warning names it: the qualified
    ``<source>/<name>`` form drops the profile's own file, which is what someone
    who shadows an artifact and then excludes it almost always meant.

    Args:
        root_dir: The profile root the exclusions anchor at.
        shadow_candidates: Bare library-selection exclusions recorded by
            :func:`_apply_exclude`, as ``"<source>/<name>"``.
    """
    for artifact in sorted(shadow_candidates):
        if not _profile_ships(root_dir, artifact):
            continue
        source, _, name = artifact.partition("/")
        _LOGGER.warning(
            "Profile %s excludes the bare name %r under %r, but it also ships its own "
            "%r in %s. That file still renders, so the exclusion has no effect. "
            "Did you mean %r? A bare name only unselects the BUILT-IN artifact, which "
            "this profile already shadows; the qualified spelling omits the profile's "
            "own file, which is what makes the built-in version render again.",
            root_dir,
            name,
            source,
            name,
            root_dir / source,
            artifact,
        )


def _warn_unmatched_exclusions(root_dir: Path, artifacts: set[str]) -> None:
    """Warn about recorded exclusions that match no file under the profile root.

    A misspelled shadow exclusion is otherwise a silent no-op: the artifact it
    meant to drop keeps rendering, and nothing says why. This warns rather than
    raises because a root profile may legitimately exclude something only some
    of its renders carry.

    Args:
        root_dir: The profile root the exclusions anchor at.
        artifacts: Recorded ``"<source>/<name>"`` exclusions.
    """
    for artifact in sorted(artifacts):
        if _profile_ships(root_dir, artifact):
            continue
        source, _, _name = artifact.partition("/")
        _LOGGER.warning(
            "Profile %s excludes %r, but no such artifact exists under %s. The "
            "exclusion has no effect. Check the spelling.",
            root_dir,
            artifact,
            root_dir / source,
        )


#: The kill-switch hook every app render selects, whatever its file lists.
_WRITES_CHECK_HOOK = "writes-check"


#: The dotted ``config:`` key that gives a render its own web-terminal tier.
_WEB_TERMINALS_ENABLED = "modules.web_terminals.enabled"

#: What an attached render is, whatever its file says: no services of its own,
#: no web-terminal tier of its own. ``(key, value)`` for the top-level facts.
_ATTACHMENT_FACTS: tuple[tuple[str, Any], ...] = (("deploy_services", False), ("services", {}))


def _find_config_path(config: Any, parts: list[str]) -> list[Any]:
    """Every value in ``config`` spelled at the key path ``parts``.

    A ``config:`` block may spell one leaf fully dotted, partly nested or fully
    nested; all three reach the same rendered key.
    """
    found: list[Any] = []
    if not isinstance(config, dict):
        return found
    for key, value in config.items():
        key_parts = str(key).split(".")
        if key_parts != parts[: len(key_parts)]:
            continue
        if len(key_parts) == len(parts):
            found.append(value)
        elif isinstance(value, dict):
            found.extend(_find_config_path(value, parts[len(key_parts) :]))
    return found


def _drop_config_path(config: dict[str, Any], parts: list[str]) -> dict[str, Any]:
    """A copy of ``config`` with every spelling of ``parts`` removed."""
    result: dict[str, Any] = {}
    for key, value in config.items():
        key_parts = str(key).split(".")
        if key_parts == parts[: len(key_parts)]:
            if len(key_parts) == len(parts):
                continue
            if isinstance(value, dict):
                value = _drop_config_path(value, parts[len(key_parts) :])
                if not value:
                    continue
        result[key] = value
    return result


def _refuse_layer_spellings(
    layer: dict[str, Any], layer_rel: str, kind: Any, *, legacy: bool
) -> None:
    """Refuse what a tier or app file may not say: subtraction, attachment, derived keys.

    A tier adds by name, so ``exclude:`` has no meaning in it; the same holds
    for an app, whose lists are its own. Attachment is a fact of the kind, set
    by the resolver, so a file spelling it is either redundant or wrong. A
    file in the retired layer directory (``legacy``) keeps ``exclude:`` and may restate the
    attachment it gets anyway, but never contradict it. A ``config:`` key the
    build renders (:mod:`~osprey.cli.derived_keys`) is refused here, on the
    file's own block, so the refusal names this file rather than the merged
    render's ``profile.yml``.

    Raises:
        BuildProfileError: Naming the file and the offending key.
    """
    article = "an" if kind.value[0] in "aeiou" else "a"
    if not legacy and "exclude" in layer:
        raise BuildProfileError(
            f"{layer_rel} spells 'exclude:'. A {kind.value} cannot take anything away "
            f"from the deployment it belongs to: {article} {kind.value} adds by name; "
            f"edit the deployment's own lists in {PROFILE_FILENAME} instead."
        )
    spelled: list[tuple[str, Any, Any]] = [
        (key, layer[key], forced) for key, forced in _ATTACHMENT_FACTS if key in layer
    ]
    spelled.extend(
        (_WEB_TERMINALS_ENABLED, value, False)
        for value in _find_config_path(layer.get("config"), _WEB_TERMINALS_ENABLED.split("."))
    )
    for key, value, forced in spelled:
        if legacy and value == forced:
            continue
        raise BuildProfileError(
            f"{layer_rel} spells {key!r}. Every {kind.value} render is attached to its "
            f"deployment: it runs no services and no web-terminal tier of its own, and "
            f"dials the deployment's. The build sets that, so remove {key!r} from "
            f"{layer_rel}; to change what the deployment runs, edit its "
            f"{PROFILE_FILENAME}."
        )
    derived = derived_key_errors(layer.get("config"), layer_rel)
    if derived:
        raise BuildProfileError("Profile validation failed:\n  " + "\n  ".join(derived))


def _strip_attachment(layer: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``layer`` without its (already checked) attachment spellings."""
    stripped = {k: v for k, v in layer.items() if k not in dict(_ATTACHMENT_FACTS)}
    if isinstance(stripped.get("config"), dict):
        stripped["config"] = _drop_config_path(
            stripped["config"], _WEB_TERMINALS_ENABLED.split(".")
        )
    return stripped


def _attach(merged: dict[str, Any]) -> dict[str, Any]:
    """Set the attachment facts every tier and app render carries."""
    attached = dict(merged)
    for key, forced in _ATTACHMENT_FACTS:
        attached[key] = forced.copy() if isinstance(forced, dict) else forced
    config = attached.get("config")
    config = (
        _drop_config_path(config, _WEB_TERMINALS_ENABLED.split("."))
        if isinstance(config, dict)
        else {}
    )
    config[_WEB_TERMINALS_ENABLED] = False
    attached["config"] = config
    return attached


#: The top-level keys an app file owns outright: what the agent IS. The host's
#: values never reach an app render, so a key the app file leaves out is
#: absent from the render rather than inherited.
_APP_OWNED_KEYS: tuple[str, ...] = (
    "name",
    "hooks",
    "rules",
    "skills",
    "agents",
    "output_styles",
    "web_panels",
    "default_panel",
    "claude_md_template",
    "channel_finder_mode",
)

#: The one top-level key an app file may set over the host's value rather than
#: own outright: an app that names no model runs the host's.
_APP_MODEL_KEY = "model"

_CLAUDE_CODE_SECTION = "claude_code"
_CONTROL_SYSTEM_SECTION = "control_system"
#: The one posture an app render holds, whatever its host or its file says.
_APP_WRITES = "never"


def _flatten_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    """A copy of ``config`` with every key under ``section`` spelled fully dotted.

    Keys outside ``section`` keep their spelling and order. Two configs whose
    ``section`` entries are both flattened merge leaf by leaf — a nested
    ``claude_code: {permissions: {deny: [...]}}`` and a dotted
    ``claude_code.permissions.deny`` become one key whose lists union — rather
    than surviving as two spellings of one path.

    A partly-nested key such as ``claude_code: {permissions.deny: [...]}`` is
    one more spelling of the same path and is flattened like the rest. Below
    the second segment (``claude_code.servers``, ``claude_code.permissions``)
    the keys are names, not paths: a mapping there whose own keys contain a
    dot (a server named with one) is kept whole under its path, since the
    build applies a mapping-form key verbatim and splitting it would address a
    different, nested place.
    """
    result: dict[str, Any] = {}

    def walk(node: dict[str, Any], prefix: list[str]) -> None:
        if len(prefix) >= 2 and any("." in str(key) for key in node):
            result[".".join(prefix)] = node
            return
        for key, value in node.items():
            path = [*prefix, *str(key).split(".")]
            if isinstance(value, dict) and value:
                walk(value, path)
            else:
                result[".".join(path)] = value

    for key, value in config.items():
        parts = str(key).split(".")
        if parts[0] != section:
            result[key] = value
        elif isinstance(value, dict) and value:
            walk(value, parts)
        else:
            result[key] = value
    return result


def _drop_config_leaves(
    node: dict[str, Any], drop: Any, prefix: tuple[str, ...] = ()
) -> dict[str, Any]:
    """A copy of a config block without the leaves whose dotted path ``drop`` accepts.

    ``drop`` sees each leaf's full path as segments, whatever the spelling —
    dotted, partly nested, nested, or a custom connector's dotted type key —
    and mappings the drop empties are pruned.
    """
    result: dict[str, Any] = {}
    for key, value in node.items():
        path = prefix + tuple(str(key).split("."))
        if isinstance(value, dict) and value:
            value = _drop_config_leaves(value, drop, path)
            if not value:
                continue
        elif drop(path):
            continue
        result[key] = value
    return result


def _is_server_switch(path: tuple[str, ...]) -> bool:
    """Whether a leaf path lies under ``claude_code.servers``."""
    return path[:2] == (_CLAUDE_CODE_SECTION, "servers")


def _is_write_switch(path: tuple[str, ...]) -> bool:
    """Whether a leaf path is a write-posture switch.

    The tier key ``control_system.writes`` in any spelling, and every retired
    ``control_system.*writes_enabled`` leaf (the flat one and each connector
    type's) — an app render holds none of them but its own pin.
    """
    if not path or path[0] != _CONTROL_SYSTEM_SECTION:
        return False
    return (
        path == (_CONTROL_SYSTEM_SECTION, _document.WRITES_KEY_LEAF)
        or path[-1] == _document.RETIRED_WRITES_LEAF
    )


def _is_deny_lift(path: tuple[str, ...]) -> bool:
    """Whether a leaf path is a ``claude_code.permissions.remove_deny`` list.

    A lift grants the host's own render a tool its floor denies. It is a
    privilege of that render, not a fact of the deployment, so it never
    reaches an app.
    """
    return path[:3] == (_CLAUDE_CODE_SECTION, "permissions", "remove_deny")


def _edit_door_switches() -> tuple[tuple[str, ...], ...]:
    """The path of every switch door in the deployment-edit door table.

    Read from :data:`~osprey.deployment.web_terminals.roster.DEPLOYMENT_EDIT_DOORS`
    rather than restated, imported here because that module is heavy and this
    one is on the path of every profile read.
    """
    from osprey.deployment.web_terminals.roster import DEPLOYMENT_EDIT_DOORS

    return tuple(tuple(door.key.split(".")) for door in DEPLOYMENT_EDIT_DOORS if door.is_switch)


def _is_edit_door_switch(path: tuple[str, ...]) -> bool:
    """Whether a leaf path spells one of the deployment-edit door switches.

    Every spelling counts — dotted, partly nested or nested — and so does a
    scalar written where the door's section belongs (``web.config_panel: true``).
    """
    for segments in _edit_door_switches():
        if path[: len(segments)] == segments:
            return True
        if len(path) >= 2 and segments[: len(path)] == path:
            return True
    return False


def _drop_write_switches(config: dict[str, Any]) -> dict[str, Any]:
    """A copy of a config block without any write-posture switch (:func:`_is_write_switch`)."""
    return _drop_config_leaves(config, _is_write_switch)


def _pin_write_switches(config: dict[str, Any]) -> dict[str, Any]:
    """``config`` with every write switch removed and ``control_system.writes: never`` pinned.

    Every spelling of the tier key and every retired ``writes_enabled`` leaf is
    dropped, and the tier key is set again, dotted, to ``never``. No retired
    leaf is pinned: the tier key alone states the posture, and a per-type pin
    would keep a retired key alive in every app render. Dropping a leaf prunes
    a connector block it was the only key of; a host's resolved profile holds
    no such block, because the normalizer has already removed its retired
    leaves (and judged the emptied block).
    """
    from osprey_connectors.types import WRITES_KEY

    pinned = _drop_write_switches(config)
    pinned[WRITES_KEY] = _APP_WRITES
    return pinned


def _pin_edit_doors(config: dict[str, Any]) -> dict[str, Any]:
    """``config`` with every deployment-edit door set closed, dotted.

    Each row of :data:`~osprey.deployment.web_terminals.roster.DEPLOYMENT_EDIT_DOORS`
    is pinned: a switch door (the Config panel, gallery writes) has every
    spelling dropped and is set to its closed value; the list door has its
    tool appended to the dotted list, which keeps the floor denies already
    there. The setup tool rewrites the render's own ``config.yml``, the panel
    edits the same file and gallery writes add to what the agent reads, so an
    app render holding any of them could change the deployment it is a guest
    of, whatever the host or the app file set.
    """
    from osprey.deployment.web_terminals.roster import DEPLOYMENT_EDIT_DOORS

    pinned = _drop_config_leaves(config, _is_edit_door_switch)
    for door in DEPLOYMENT_EDIT_DOORS:
        if door.is_switch:
            pinned[door.key] = door.closed
            continue
        entries = pinned.get(door.key)
        entries = list(entries) if isinstance(entries, list) else []
        if door.closed not in entries:
            entries.append(door.closed)
        pinned[door.key] = entries
    return pinned


def _judge_app_posture(app_config: dict[str, Any], app_rel: str) -> None:
    """Refuse an app that spells ``control_system.writes: allowed``; warn about one that arms.

    An app is always ``never``. The tier key spelled ``allowed`` in an app is
    refused (the parse refuses it too; this is the composer's own door, for a
    caller that hands it an unparsed document). An app on the retired
    ``writes_enabled`` leaves that arm is pinned ``never`` as it always was,
    and warned about once: the normalizer has already rewritten a parsed one,
    so this fires only for an app it did not see.
    """
    from osprey_connectors.types import WRITES_KEY

    from .build_profile_document import retired_write_answer

    if any(value == "allowed" for value in _document._tier_key_values(app_config)):
        raise BuildProfileError(
            f"{app_rel}: an app cannot spell `{WRITES_KEY}: allowed` — an app is always "
            f"`{WRITES_KEY}: {_APP_WRITES}`; give write access through a tier."
        )
    if retired_write_answer(app_config) == "allowed":
        _warn_once(
            app_rel,
            "app-arms",
            f"{app_rel}: the retired `writes_enabled` keys arm writes, but an app is always "
            f"`{WRITES_KEY}: {_APP_WRITES}`, so this app renders `{_APP_WRITES}`. Delete the "
            "retired keys",
        )


def _host_guard_wiring(config: dict[str, Any]) -> dict[str, Any]:
    """The host's hook wiring every app keeps: its guards, and nothing else.

    Guards are a floor, the way the host's floor denies are: a
    ``claude_code.hooks.PreToolUse`` declaration in force on the host reaches
    every app render, inert where no tool it matches is present, so an app can
    add a guard but never drop one by omission. A declaration on any other
    event is the host's own; an app runs it only when its own ``config:``
    declares it.

    Args:
        config: The host's config facts, ``claude_code`` keys fully dotted.

    Returns:
        A copy of *config* without the non-guard declarations; an event left
        with none is dropped.
    """
    from .profile_conventions import GUARD_EVENT

    prefix = f"{_CLAUDE_CODE_SECTION}.hooks"
    kept: dict[str, Any] = {}
    for key, value in config.items():
        if key == prefix and isinstance(value, dict):
            guards = value.get(GUARD_EVENT)
            if isinstance(guards, list) and guards:
                kept[f"{prefix}.{GUARD_EVENT}"] = guards
        elif key.startswith(f"{prefix}."):
            if key == f"{prefix}.{GUARD_EVENT}":
                kept[key] = value
        else:
            kept[key] = value
    return kept


def _guard_floor(host_hooks: Any, app_hooks: Any) -> list[Any]:
    """The app's ``hooks:`` list with every host library guard it omits appended.

    A guard is a library hook on :data:`~osprey.cli.profile_conventions.GUARD_EVENT`
    (approval, the kill switch, limits, the memory guard). Each one the host
    selects reaches the app as the host spells it — the deployment's
    replacement file where the host names one — unless the app selects it
    already, in either spelling. The kill switch is pinned whatever the host
    says: an app render can never write.
    """
    from .profile_conventions import library_guard_hooks, library_selection

    hooks = list(app_hooks) if isinstance(app_hooks, list) else []
    selected = set(library_selection("hooks", hooks))
    guards = library_guard_hooks()
    for entry in host_hooks if isinstance(host_hooks, list) else []:
        names = library_selection("hooks", [entry])
        if names and names[0] in guards and names[0] not in selected:
            hooks.append(entry)
            selected.add(names[0])
    if _WRITES_CHECK_HOOK not in selected:
        hooks.append(_WRITES_CHECK_HOOK)
    return hooks


def compose_app_render(host: dict[str, Any], app: dict[str, Any], app_rel: str) -> dict[str, Any]:
    """Compose an app render from its host deployment and its complete app file.

    The app file owns what the agent is: the keys in :data:`_APP_OWNED_KEYS`
    and :data:`~osprey.cli.build_profile_document.RENDER_KEYS` come from it alone. Its ``config:`` sits over
    the host's facts — the host's config minus every ``claude_code.servers``
    switch (the app states its own server set), minus every
    ``claude_code.hooks`` declaration on an event other than ``PreToolUse``,
    and minus every write-posture switch (:func:`_is_write_switch`);
    ``claude_code.permissions`` is
    kept, so the host's floor denies union with the app's own — all but the
    host's ``remove_deny`` lifts, which grant the host's own render. The app's
    ``mcp_servers`` declarations are added to the host's. A ``model:`` the
    app names replaces the host's, served by the host's provider; an app that
    names none runs the host's model. Every other top-level key, ``provider``,
    ``provenance`` and ``data`` among them, is the host's verbatim, which is
    what keeps the data bundle the deployment's. Nothing is
    read from the package: the app file is a materialized, explicit copy.

    Guards are a floor: every ``PreToolUse`` hook in force on the host — a
    library guard its ``hooks:`` selects, or a facility guard its
    ``claude_code.hooks.PreToolUse`` declares — reaches the app whatever its
    own list says, so an app adds guards but never drops one by omission.

    Whatever either side says, the render cannot write or edit the deployment:
    ``control_system.writes`` is pinned ``never`` and no retired
    ``writes_enabled`` leaf is kept, every door in
    :data:`~osprey.deployment.web_terminals.roster.DEPLOYMENT_EDIT_DOORS` is set
    closed (the setup tool denied, the Config panel and gallery writes off),
    ``writes-check`` is selected (the one guard pinned even where the host
    selects none), and
    :func:`~osprey.deployment.web_terminals.lint.lint_app_render` checks the
    result before it is returned.

    Args:
        host: The host's resolved root profile.
        app: The app file's document, unresolved.
        app_rel: The app file's repo-relative label, for messages.

    Returns:
        The composed render, before the attachment facts are set.

    Raises:
        BuildProfileError: If the app file spells ``control_system.writes:
            allowed``, or if the composed render could still write.
    """
    render = {
        key: value
        for key, value in host.items()
        if key not in _APP_OWNED_KEYS and key not in RENDER_KEYS and key != "config"
    }
    for key in (*_APP_OWNED_KEYS, *sorted(RENDER_KEYS)):
        if key in app:
            render[key] = app[key]
    # The model is the app's when it names one, else the host's; the provider
    # that serves it is always the host's.
    if app.get(_APP_MODEL_KEY) is not None:
        render[_APP_MODEL_KEY] = app[_APP_MODEL_KEY]

    app_servers = app.get("mcp_servers")
    if isinstance(app_servers, dict) and app_servers:
        host_servers = host.get("mcp_servers")
        render["mcp_servers"] = _deep_merge(
            host_servers if isinstance(host_servers, dict) else {}, app_servers
        )

    host_config = host.get("config")
    host_config = dict(host_config) if isinstance(host_config, dict) else {}
    app_config = app.get("config")
    app_config = dict(app_config) if isinstance(app_config, dict) else {}
    host_facts = _drop_config_leaves(
        host_config,
        lambda path: (
            _is_server_switch(path)
            or _is_write_switch(path)
            or _is_deny_lift(path)
            or _is_edit_door_switch(path)
        ),
    )
    config = _deep_merge(
        _host_guard_wiring(_flatten_section(host_facts, _CLAUDE_CODE_SECTION)),
        _flatten_section(app_config, _CLAUDE_CODE_SECTION),
    )
    _judge_app_posture(app_config, app_rel)
    render["config"] = config
    # Guards are a floor: every host guard the app omits is appended, as the
    # host spells it; the seal then pins the kill switch.
    render["hooks"] = _guard_floor(host.get("hooks"), render.get("hooks"))
    return seal_app_render(render, app_rel)


def seal_app_render(render: dict[str, Any], app_rel: str) -> dict[str, Any]:
    """Pin an app render shut and check it: the composer's last step, and the edit's.

    ``control_system.writes`` is pinned ``never`` with every retired leaf
    dropped, every deployment-edit door is closed, ``writes-check`` is
    selected, and :func:`~osprey.deployment.web_terminals.lint.lint_app_render`
    checks the result. A ``--set`` edit lands after the composer, so the build
    seals an edited app render again: an app is ``never`` whatever the
    command line says.

    Args:
        render: The app render, composed or edited.
        app_rel: The app file's repo-relative label, for messages.

    Returns:
        ``render`` with its ``config`` and ``hooks`` pinned.

    Raises:
        BuildProfileError: If the pinned render could still write or edit the
            deployment.
    """
    config = render.get("config")
    config = dict(config) if isinstance(config, dict) else {}
    render["config"] = _pin_edit_doors(_pin_write_switches(config))

    # The kill switch is pinned whatever the list says, in either spelling: an
    # app render can never write. The host floor was applied by the composer.
    render["hooks"] = _guard_floor((), render.get("hooks"))

    from osprey.deployment.web_terminals.lint import lint_app_render

    findings = lint_app_render(render)
    if findings:
        raise BuildProfileError(f"{app_rel}: {findings[0].message}")
    return render


# ── Today's write posture beside the tier's ─────────────────────────────────
#
# The retired posture leaves are rewritten to ``control_system.writes`` per
# document (C-NORM), which is PR-3's per-layer rule. Today's reading was per
# machine instead (``_retired_session_posture`` of the merged section), where a per-type
# ``false`` overrides a flat ``true`` and a leaf on a type no configured target
# reaches arms nothing. The two can differ on a merged render, and an operator
# is told where they do (R2 CF-1 / BE-3, CC-2), once per render (BE-22).

_ALLOWED = "allowed"
_NEVER = "never"
_REVALIDATE = "run `osprey init --revalidate`"
_POSTURE_CHANGE = "posture-change"
_WIDENING = "widening"
_EMPTIED_BLOCK = "emptied-block"


def _set_path(node: dict[str, Any], path: Sequence[str], value: Any) -> None:
    """Set ``value`` at ``path`` beneath ``node``, creating mappings on the way."""
    for part in path[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[path[-1]] = value


def _retired_leaf_paths(origin: dict[str, Any]) -> dict[tuple[str, ...], Any]:
    """Each leaf of a :func:`retired_write_origin` fragment, by its literal path.

    The paths start at ``control_system``; a connector type is one segment,
    dots and all, as a mapping spells it.
    """
    leaves: dict[tuple[str, ...], Any] = {}
    if _document.RETIRED_WRITES_LEAF in origin:
        leaves[(_CONTROL_SYSTEM_SECTION, _document.RETIRED_WRITES_LEAF)] = origin[
            _document.RETIRED_WRITES_LEAF
        ]
    connector = origin.get("connector")
    if isinstance(connector, dict):
        for name, block in connector.items():
            if isinstance(block, dict) and _document.RETIRED_WRITES_LEAF in block:
                path = (
                    _CONTROL_SYSTEM_SECTION,
                    "connector",
                    str(name),
                    _document.RETIRED_WRITES_LEAF,
                )
                leaves[path] = block[_document.RETIRED_WRITES_LEAF]
    return leaves


def _rendered_control_system(
    config: Any, retired: dict[tuple[str, ...], Any] | None = None
) -> dict[str, Any]:
    """The ``control_system`` section the render writes from a merged ``config:`` mapping.

    Each key applied in order, split on its dots, as ``config_update_fields``
    applies it: a mapping value is assigned whole, and keys inside it are
    written literally. In memory, and only the control-system keys.

    *retired* (leaf paths from :func:`_retired_leaf_paths`) reads the render
    as it was before the normalizer removed those leaves: they are written
    AHEAD of the config, so a later mapping assigned over a leaf's block
    replaces it as it replaces the block's other keys. A mapping that still
    holds the leaf's block is where the leaf was spelled (the normalizer
    leaves the block in place), so the leaf is put back into it.
    """
    tree: dict[str, Any] = {}
    if not isinstance(config, dict):
        return {}
    for path, leaf in (retired or {}).items():
        _set_path(tree, path, copy.deepcopy(leaf))
    for key, value in config.items():
        if not isinstance(key, str):
            continue
        if key != _CONTROL_SYSTEM_SECTION and not key.startswith(f"{_CONTROL_SYSTEM_SECTION}."):
            continue
        parts = key.split(".")
        node: Any = tree
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]
            if not isinstance(node, dict):
                break
        else:
            assigned = copy.deepcopy(value)
            node[parts[-1]] = assigned
            if isinstance(assigned, dict):
                _reseat_retired(assigned, tuple(parts), retired or {})
    section = tree.get(_CONTROL_SYSTEM_SECTION)
    return section if isinstance(section, dict) else {}


def _reseat_retired(
    assigned: dict[str, Any], at: tuple[str, ...], retired: dict[tuple[str, ...], Any]
) -> None:
    """Put each retired leaf below *at* back into *assigned* when its block is there."""
    for path, leaf in retired.items():
        if len(path) <= len(at) or path[: len(at)] != at:
            continue
        block: Any = assigned
        for part in path[len(at) : -1]:
            block = block.get(part) if isinstance(block, dict) else None
        if isinstance(block, dict):
            block[path[-1]] = copy.deepcopy(leaf)


def _machine(section: dict[str, Any], target: str) -> str:
    try:
        return f"`{target}` (`{resolve_target(section, target)}`)"
    except ValueError:
        return f"`{target}`"


def _warn_once(dedupe: str, check: str, message: str) -> None:
    key = (dedupe, check)
    if key in _document._WARNED_RETIRED_KEYS:
        return
    _document._WARNED_RETIRED_KEYS.add(key)
    _LOGGER.warning("%s; %s", message, _REVALIDATE)


def judge_retired_posture(
    config: Any,
    origin_paths: Sequence[Path],
    label: str,
    dedupe: str,
    *,
    warn: bool = True,
) -> None:
    """Compare today's write posture of a merged render with its tier, and say where they differ.

    Today's reading is what the retired leaves armed per target
    (:func:`~.build_profile_document._retired_session_posture`) in the
    rendered ``control_system`` section with the retired leaves the normalizer removed along the chain
    (:func:`~.build_profile_document.retired_write_origin` of each of
    *origin_paths*, later over earlier) put back as the render applies them
    (:func:`_rendered_control_system`), so a block a later layer's mapping
    replaced keeps no leaf of it: ``allowed`` when it arms
    at least one configured target, else ``never``. The new reading is
    ``tier_writes`` of the rendered section. A render with no retired origin
    on its chain is skipped.

    * A machine configured today only because a retired leaf was its block's
      one key is no longer configured once the leaf is gone: refused when
      today's reading armed it, warned about when it did not (a shipped
      previous-release tier spells an unarmed ``epics`` leaf on deployments
      that configure no EPICS machine, and those keep building for the
      release window).
    * A difference warns "was <today>, now <tier>".
    * An unchanged ``allowed`` whose configured machines today's reading only
      partly armed warns, naming the machines that become armable.

    Each warning is logged once per *dedupe* key, and never with *warn* off.

    Args:
        config: The merged render's ``config:`` mapping.
        origin_paths: The chain's files, base first, the layer last.
        label: How to name the render in messages.
        dedupe: The render's resolved path, keying the once-only warnings.
        warn: Whether to log the warnings (the refusal is raised either way).

    Raises:
        BuildProfileError: When a retired leaf that armed a configured machine
            was the only key of its connector block.
    """
    origins = [o for o in (retired_write_origin(p) for p in origin_paths) if o]
    if not origins:
        return
    section = _rendered_control_system(config)
    tier = tier_writes(section)
    retired: dict[tuple[str, ...], Any] = {}
    for origin in origins:
        retired.update(_retired_leaf_paths(origin))
    today_section = {
        k: v
        for k, v in _rendered_control_system(config, retired).items()
        if k != _document.WRITES_KEY_LEAF
    }

    configured_now = configured_targets(section)
    lost = [t for t in configured_targets(today_section) if t not in configured_now]
    for machine in lost:
        connector_type = resolve_target(today_section, machine)
        emptied = (
            f"{label}: the `{_CONTROL_SYSTEM_SECTION}.connector.{connector_type}` block holds only "
            f"the retired `{_document.RETIRED_WRITES_LEAF}` leaf, so without it the "
            f"{_machine(today_section, machine)} machine is no longer configured. Add a "
            "connection key to the block, or delete the block"
        )
        if _document._retired_target_armed(today_section, machine):
            raise BuildProfileError(f"{emptied}.")
        if warn:
            _warn_once(dedupe, f"{_EMPTIED_BLOCK}:{connector_type}", emptied)
    if not warn:
        return

    armed = [
        target for target, on in _document._retired_session_posture(today_section).items() if on
    ]
    today = _ALLOWED if armed else _NEVER
    if today != tier:
        _warn_once(
            dedupe,
            _POSTURE_CHANGE,
            f"{label}: write posture was {today}, now {tier}. The retired `writes_enabled` "
            f"keys read per machine (armed today: "
            f"{', '.join(_machine(today_section, t) for t in armed) or 'none'}); "
            f"`{_CONTROL_SYSTEM_SECTION}.writes` reads `{tier}` for every machine. State "
            f"`{_CONTROL_SYSTEM_SECTION}.writes` explicitly",
        )
        return
    if tier == _ALLOWED:
        armable = [t for t in configured_now if t not in armed]
        if armable:
            _warn_once(
                dedupe,
                _WIDENING,
                f"{label}: `{_CONTROL_SYSTEM_SECTION}.writes: {_ALLOWED}` lets a session write to "
                f"every configured machine; today only "
                f"{', '.join(_machine(today_section, t) for t in armed)} was armed, so "
                f"{', '.join(_machine(today_section, t) for t in armable)} become armable",
            )


def _refuse_root_lists(resolved: dict[str, Any], where: str) -> None:
    """Refuse a deployment profile spelling ``lists:``, a login-only key.

    ``lists`` says whether one login's card lists the deployment's applications
    or only its own; spelled on the root, every login silent on it would
    inherit it through the merge.
    """
    if "lists" in resolved:
        raise BuildProfileError(
            f"{where} spells `lists`, a login key: it says whether one login's card "
            f"lists the deployment's applications or only its own. Remove it from "
            f"{PROFILE_FILENAME} and set it in tiers/<name>.yml."
        )


def _refuse_unlisted_selection(
    host_resolved: dict[str, Any], layer: dict[str, Any], layer_rel: str
) -> None:
    """Refuse a login selecting an application the deployment does not list.

    An application is on when the deployment's profile lists it; a login
    selects from that list and never adds to it.
    """
    selected = layer.get("apps")
    if not isinstance(selected, dict):
        return
    listed = host_resolved.get("apps")
    listed_names = {str(name) for name in listed} if isinstance(listed, dict) else set()
    unlisted = sorted({str(name) for name in selected} - listed_names)
    if unlisted:
        names = ", ".join(f"apps.{name}" for name in unlisted)
        raise BuildProfileError(
            f"{layer_rel} selects {names}, which the deployment's {PROFILE_FILENAME} "
            f"does not list. A login selects only applications the deployment runs; "
            f"list them under apps: in {PROFILE_FILENAME} or remove them from {layer_rel}."
        )


def render_document(
    kind: ProfileKind,
    host_resolved: dict[str, Any],
    layer: dict[str, Any],
    layer_rel: str,
    *,
    legacy: bool = False,
    artifacts: set[str] | None = None,
    shadow_candidates: set[str] | None = None,
    host_chain: Sequence[Path] = (),
    layer_path: Path | None = None,
    warn: bool = True,
) -> dict[str, Any]:
    """The profile document one layer renders from, decided by its kind.

    The one answer to "what does this tier or app render": init, lint and the
    build each read a render's privileges off this document, so none of them
    can judge an app as the tier merge it never is.

    * ``legacy`` — a file in the retired layer directory keeps its old
      meaning, ``exclude:`` included, for the one release it is still read
      (``osprey init --revalidate`` rewrites it into ``tiers/``);
    * an app is composed from the host and its complete file
      (:func:`compose_app_render`);
    * a tier is merged over the host.

    Args:
        kind: The layer's kind.
        host_resolved: The host's resolved root profile (only its ``config``,
            ``hooks`` and ``mcp_servers`` decide what an app holds).
        layer: The layer's document, unresolved.
        layer_rel: The layer's repo-relative label, for messages.
        legacy: Whether the layer sits in the retired layer directory.
        artifacts: Passed to :func:`merge_layer_delta` for a legacy layer.
        shadow_candidates: Passed to :func:`merge_layer_delta` for a legacy layer.
        host_chain: The host's ``extends`` chain as resolved paths, base first.
            With *layer_path*, where the retired posture leaves each document
            spelled are looked up (:func:`judge_retired_posture`); a caller
            holding no files passes neither, and the check is skipped.
        layer_path: The layer's file.
        warn: Whether :func:`judge_retired_posture` logs its warnings.

    Returns:
        The rendered document, before the attachment facts are set.

    Raises:
        BuildProfileError: If an app's composed render could still write or
            edit the deployment, or if an armed machine's connector block held
            only a retired posture leaf.
    """
    if legacy:
        rendered = merge_layer_delta(
            host_resolved,
            _strip_attachment(layer),
            artifacts=artifacts,
            shadow_candidates=shadow_candidates,
        )
    elif kind is ProfileKind.APP:
        rendered = compose_app_render(host_resolved, layer, layer_rel)
    else:
        _refuse_unlisted_selection(host_resolved, layer, layer_rel)
        rendered = _deep_merge(host_resolved, layer)
    if kind is not ProfileKind.APP or legacy:
        origin_paths = [*host_chain, *([layer_path] if layer_path is not None else [])]
        judge_retired_posture(
            rendered.get("config"),
            origin_paths,
            layer_rel,
            str(Path(layer_path).resolve()) if layer_path is not None else layer_rel,
            warn=warn,
        )
    # The per-render keys are the card's, read from the file by the render
    # projection; no profile field holds them, so the document they are parsed
    # into does not carry them.
    return {key: value for key, value in rendered.items() if key not in RENDER_KEYS}


@dataclass(frozen=True)
class ResolvedProfileDocument:
    """One profile document resolved into what the build actually reads.

    Attributes:
        raw: The fully resolved profile mapping — ``extends`` consumed and, for
            a layer, merged over its root.
        root_dir: The profile root every profile-relative path anchors at
            (``data:``, convention directories, ``.env``, hash material). For a
            layer this is the directory *above* ``tiers/``, never the
            delta's own parent.
        is_layer: Whether the document was resolved as a layer.
        excluded_artifacts: Convention-dir artifacts the resolved profile
            excludes, as ``"<source>/<name>"`` (e.g. ``"agents/orbit-writer"``).
            The build omits these files, and ownership derivation scans the
            post-exclude set — so excluding a shadow restores the framework's
            own version of that artifact.
        inherited_preset: The bundled preset nearest this document on its
            ``extends`` chain, or ``None`` when the chain reaches none. It is
            the only surviving trace of the preset-side ``app_template:`` the
            chain consumed, and so the answer to "which packaged data bundle is
            this profile's tree from" for a profile that records no
            ``provenance:`` block of its own.
        app_rel: The app file's repo-relative label when the document was
            composed as an app render (:func:`compose_app_render`), else
            ``None``.
    """

    raw: dict[str, Any]
    root_dir: Path
    is_layer: bool
    excluded_artifacts: frozenset[str]
    inherited_preset: str | None = None
    app_rel: str | None = None


def resolve_profile_document(
    raw: dict[str, Any],
    profile_path: Path,
    *,
    warn: bool = True,
    root_overlays: Sequence[dict[str, Any]] = (),
) -> ResolvedProfileDocument:
    """Resolve a profile document, following the implicit render merge.

    The one place that decides what a profile file *means*. A file under
    ``tiers/`` beside a ``profile.yml`` carries only a delta: living there
    is its inheritance, so it is resolved against that root rather than on its
    own, and everything it names anchors at the root. Every other file resolves
    standalone through its ``extends`` chain.

    Both the loader and the content hash go through here, which is what keeps a
    render's built project and its staleness hash describing the same merge.

    Args:
        raw: The profile document as read.
        profile_path: Where that document lives — the classification is made
            from its location, not its contents.
        warn: Whether to log the exclusion diagnostics — exclusions that match
            no file, and bare exclusions a profile file shadows. ``False`` for
            the hash path, which resolves the same
            document a build already loaded — without it every build reports
            each one twice, and a warning that repeats is one people learn to
            skip. The loader is the user-facing surface, so it keeps the
            warnings. The same flag gates the write-posture warnings of
            :func:`judge_retired_posture`.
        root_overlays: Host-variant overlays, already read. They lay over the
            DEPLOYMENT — the root ``profile.yml`` — before its ``extends`` is
            resolved: over ``raw`` when it is the root, over the root beside
            it when ``raw`` is a tier or app. A layer therefore sees the
            variant only through the root it merges over, and the layer-file
            refusals judge what the layer file itself spells, never what the
            host's variant said.

    Returns:
        The resolved document and its anchor.

    Raises:
        BuildProfileError: If a layer declares ``extends``, if the root
            of a layer is missing or is not a YAML mapping, or if
            ``extends`` resolution fails.
        FacilityBuildError: ``profile-invalid`` when a layer names
            ``data:`` (every render shares the root profile's facility tree)
            or sets ``simulation.models`` on a VA-baselined render.
    """
    root_dir, kind = resolve_profile_root(profile_path)
    is_layer = bool(kind)
    normalized = dict(raw)
    artifacts: set[str] = set()
    shadowed: set[str] = set()
    lineage: list[str] = []

    def finish(
        resolved: dict[str, Any], as_delta: bool, app_rel: str | None = None
    ) -> ResolvedProfileDocument:
        """Report the exclusion diagnostics, then package the result.

        Both branches end the same way, and they must: the diagnostics read the
        exclusion sets every layer of either branch fed, so a branch that
        skipped one would leave that whole shape of profile without the warning.
        """
        if warn:
            _warn_unmatched_exclusions(root_dir, artifacts)
            _warn_shadowed_bare_exclusions(root_dir, shadowed)
        return ResolvedProfileDocument(
            resolved,
            root_dir,
            as_delta,
            frozenset(artifacts),
            lineage[0] if lineage else None,
            app_rel,
        )

    if not is_layer:
        for overlay in root_overlays:
            normalized = _deep_merge(normalized, overlay)
        chain: list[Path] = []
        resolved = _resolve_extends(
            normalized,
            profile_path,
            chain,
            artifacts=artifacts,
            shadow_candidates=shadowed,
            preset_lineage=lineage,
        )
        judge_retired_posture(
            resolved.get("config"),
            chain[::-1],
            str(profile_path),
            str(Path(profile_path).resolve()),
            warn=warn,
        )
        _refuse_root_lists(resolved, str(profile_path))
        return finish(resolved, False)

    layer_dir = Path(profile_path).resolve().parent.name
    layer_rel = f"{layer_dir}/{Path(profile_path).name}"
    retired_tier = layer_dir in RETIRED_LAYER_DIRS

    if "extends" in normalized:
        raise BuildProfileError(
            f"{kind.value.capitalize()} {profile_path} declares 'extends'. A file in "
            f"{layer_dir}/ already inherits from the {PROFILE_FILENAME} beside "
            f"it, so a second base would be ambiguous. "
            f"Remove the 'extends' key; to change what every render of this deployment "
            f"inherits, edit {root_dir / PROFILE_FILENAME} instead."
        )

    if "data" in normalized:
        from osprey.facility.errors import FacilityBuildError

        raise FacilityBuildError(
            "profile-invalid",
            layer_rel,
            [layer_rel],
            f"remove `data:` from {layer_rel}",
            record_kind="path",
            detail=(
                f"a {kind.value} names `data:`, and every render of a deployment shares "
                f"the facility tree the root {PROFILE_FILENAME}'s `data:` names"
            ),
        )

    _refuse_layer_spellings(normalized, layer_rel, kind, legacy=retired_tier)

    root_path = root_dir / PROFILE_FILENAME
    root_raw = _read_profile_document(root_path)
    if not isinstance(root_raw, dict):
        raise BuildProfileError(f"Profile root must be a YAML mapping: {root_path}")
    for overlay in root_overlays:
        root_raw = _deep_merge(root_raw, overlay)
    host_chain: list[Path] = []
    root_resolved = _resolve_extends(
        dict(root_raw),
        root_path,
        host_chain,
        artifacts=artifacts,
        shadow_candidates=shadowed,
        preset_lineage=lineage,
    )
    _refuse_root_lists(root_resolved, str(root_path))
    # The delta goes into the merge unresolved on purpose: it carries no
    # ``extends:`` of its own (rejected above), and _resolve_extends would
    # consume its ``exclude:`` against its own layer instead of against the
    # root, silently dropping the exclusion.
    merged = _attach(
        render_document(
            kind,
            root_resolved,
            normalized,
            layer_rel,
            legacy=retired_tier,
            artifacts=artifacts,
            shadow_candidates=shadowed,
            host_chain=host_chain[::-1],
            layer_path=Path(profile_path),
            warn=warn,
        )
    )
    from .build_profile_load import render_served_models_error

    served_error = render_served_models_error(normalized, merged, layer_rel)
    if served_error is not None:
        raise served_error
    is_app = kind is ProfileKind.APP and not retired_tier
    return finish(merged, True, layer_rel if is_app else None)


# ---------------------------------------------------------------------------
# Profile content hashing
# ---------------------------------------------------------------------------


def _fold_source_tree(
    digest: Any, label: str, source: Path, *, skip_runtime_output: bool = False
) -> None:
    """Fold one file-or-directory build input into ``digest``, in place.

    Every regular file under ``source`` contributes its path relative to
    ``source`` (posix separators, so the digest matches across platforms) NUL-
    joined with the SHA-256 of its bytes. Entries are sorted by that relative
    path, and directories themselves contribute nothing — only their files —
    so the digest depends on content alone and not on filesystem walk order.

    ``skip_runtime_output`` is passed for the ``data:`` tree only: the
    top-level :data:`~osprey.utils.workspace.RUNTIME_DATA_DIR_NAME`
    subtree is where ``osprey up`` mints its own material (per-lane CURVE
    certificates), and hashing it would make every successful start mark its
    own build OUT OF DATE (#716). The subtree is skipped statically — by its
    reserved name, not by a manifest — so the fold and the writer cannot fall
    out of sync about which files a start is allowed to create.

    A ``source`` that does not exist folds only its ``label``: the profile keys
    that name it are already in the canonical JSON, so a vanished tree still
    reads differently from a populated one without this function having to
    invent a marker for it.

    Two limits of the walk, both deliberate:

    * Symlinks. A symlinked *file* is folded — ``is_file`` and ``read_bytes``
      both follow it — and so is a ``source`` that is itself a symlink to a
      directory, which arrives here already resolved. A symlinked *directory
      nested inside* the tree is not: ``rglob`` does not recurse into one, so
      its contents are invisible to the digest while ``shutil.copytree``
      (``symlinks=False``) does follow it and copy them into the project. A
      data tree that reaches its content through a nested directory symlink
      therefore has a known blind spot in the staleness advisory: edits behind
      that link do not move the hash. Recursing resolved targets would change
      the pinned algorithm and is left to the maintainer.
    * Unicode. Relative paths are folded as their UTF-8 bytes with no
      normalization, so cross-platform determinism is guaranteed for ASCII
      names; a non-ASCII name stored NFD on one filesystem and NFC on another
      hashes differently.
    """
    import hashlib

    from osprey.utils.workspace import RUNTIME_DATA_DIR_NAME

    if source.is_dir():
        entries = []
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(source).as_posix()
            if skip_runtime_output and rel.split("/", 1)[0] == RUNTIME_DATA_DIR_NAME:
                continue
            entries.append((rel, path))
        entries.sort(key=lambda entry: entry[0])
    elif source.is_file():
        entries = [(source.name, source)]
    else:
        entries = []

    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    for rel_posix, path in entries:
        digest.update(rel_posix.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\0")


def profile_triggers_material(
    resolved: dict[str, Any], profile_dir: Path
) -> tuple[str, Path] | None:
    """The trigger file ``dispatch.triggers`` names, when it lives with the profile.

    Resolved by :func:`~osprey.cli.build_profile_presets.resolve_triggers_path`,
    the resolver the build copies the file through. A bundled trigger file is
    never returned: it moves with the package, not with the profile, and
    folding it would mark every build out of date on an upgrade. The value is
    read from the unvalidated resolved dict, so any shape other than a
    non-empty string yields ``None`` rather than an error.

    Args:
        resolved: The profile after ``extends`` (and layer) resolution.
        profile_dir: The profile root the trigger path anchors at.

    Returns:
        The spelled path normalised to POSIX form (``./triggers.yml`` keys as
        ``triggers.yml``) and the file it resolves to, or ``None`` when the
        profile names no trigger file beside it.
    """
    dispatch = resolved.get("dispatch")
    if not isinstance(dispatch, dict):
        return None
    triggers = dispatch.get("triggers")
    if not isinstance(triggers, str) or not triggers:
        return None
    source = resolve_triggers_path(profile_dir, triggers)
    if source is None or source.bundled:
        return None
    return PurePosixPath(triggers).as_posix(), source.path


def _fold_profile_material(
    digest: Any, resolved: dict[str, Any], profile_dir: Path, *, conventions: bool = True
) -> None:
    """Fold the *file* inputs a resolved profile names into ``digest``.

    The canonical JSON alone captures what a profile says; it cannot see what
    the trees it points at contain. Without this, editing a facility's channel
    database, a rule the agent reads, or the trigger table a dispatcher fires
    on leaves the profile hash untouched and the deploy-side staleness advisory
    stays silent about a project that no longer matches its source.

    Four kinds of input are covered:

    * the ``data:`` tree, anchored via
      :meth:`~osprey.cli.build_profile_model.BuildProfile.resolved_data_root`
      so it resolves exactly where the build copies it from — minus the
      reserved ``data/.runtime/`` subtree
      (:data:`~osprey.utils.workspace.RUNTIME_DATA_DIR_NAME`), which is
      runtime-minted material (the Bluesky lanes' CURVE certificates) and
      never build input: fold it and every successful ``osprey up`` marks its
      own build OUT OF DATE, stranding a bare ``osprey up -d`` boot unit at
      the drift gate (#716);
    * every convention directory the profile carries
      (:data:`~osprey.cli.profile_conventions.CONVENTION_SOURCES`, which
      includes the ``project/`` verbatim mirror), folded in sorted name order
      so reordering the mapping table cannot move a hash;
    * the trigger file ``dispatch.triggers`` names, when it lives with the
      profile (a bundled trigger file is package content, not profile
      material);
    * every layer file in ``tiers/``, ``apps/`` and the retired layer directory
      (:func:`~osprey.cli.profile_root.layer_files`, the same listing the build
      loop renders from) — the files a deployment's own per-layer projects are
      rendered from. Without them, editing a delta
      moves no hash at all: the deploy-side check would call the build clean,
      no rebuild would follow, and the render already on disk (which
      ``up`` re-renders only when it is *absent*) would keep shipping the
      superseded delta into that render's image.

    Layers are folded as the *direct children* of their layer directory and
    nothing deeper, because that is exactly the set that can be built: a file is
    read as a layer only when its parent directory IS a layer directory
    (:func:`~osprey.cli.profile_root.resolve_profile_root`), so a tree nested
    below it is no more build input than a ``README`` beside ``profile.yml``.
    Suffixes are not filtered — a delta is named by the render catalog, which
    accepts any direct child — while dot-prefixed entries are skipped, matching
    the walk every other convention read uses: a ``.DS_Store`` must not report a
    deployment as stale.

    One consequence, accepted rather than worked around: this is folded for a
    layer too (its root_dir is the profile root), so editing ONE delta
    moves every sibling render's hash as well. That is the same over-report as
    a root edit reaching every render — it costs an idempotent rebuild, where
    subtracting siblings would cost a render that silently no longer matches
    its source.

    A convention directory, the named trigger file or ``tiers/`` that is absent
    folds nothing at all, so a profile carrying none of them hashes exactly as
    it did before they existed. Creating an empty convention directory does
    move the hash — it folds its label — which is honest: the profile tree
    changed.

    Args:
        digest: The hash object to update in place.
        resolved: The profile after ``extends`` (and layer) resolution.
        profile_dir: The profile root every relative path anchors at.
        conventions: Whether to fold convention material. ``False`` for a
            bundled preset, which is one file among many in a shared package
            directory rather than a profile root of its own — folding there
            would make every preset's hash depend on every other preset's
            neighbouring directories.
    """
    from .build_profile_model import BuildProfile
    from .profile_conventions import CONVENTION_SOURCES
    from .profile_root import layer_files

    data_root = BuildProfile(name="", data=resolved.get("data")).resolved_data_root(profile_dir)
    if data_root is not None:
        # data/.runtime/ is runtime-minted material, never build input — see
        # the docstring above and _fold_source_tree's skip contract (#716).
        _fold_source_tree(digest, "data", data_root, skip_runtime_output=True)

    if not conventions:
        return

    for source in sorted(CONVENTION_SOURCES):
        candidate = profile_dir / source
        if candidate.exists():
            _fold_source_tree(digest, f"convention:{source}", candidate)

    triggers = profile_triggers_material(resolved, profile_dir)
    if triggers is not None:
        _fold_source_tree(digest, "triggers", triggers[1])

    # Sorted by name, which is the whole ordering rule a flat directory needs;
    # `_fold_source_tree` contributes each delta's own name beside its content
    # digest, so a rename moves the hash exactly as an edit does.
    for _kind, layer in layer_files(profile_dir):
        _fold_source_tree(digest, layer_fold_label(layer), layer)


def layer_fold_label(layer: Path) -> str:
    """The label a layer file folds under, named by its directory.

    By directory rather than by kind, so a file in a retired layer directory
    keeps the label it always folded under (:data:`~osprey.cli.build_profile_document.RETIRED_TIER_KEY`)
    — a repo that has not moved its files into ``tiers/`` keeps its hash —
    while ``tiers/`` and ``apps/`` files fold as ``tier`` and ``app``.

    Strict: *layer* must be a layer file, one :func:`~osprey.cli.profile_root.layer_files`
    lists. A path in any other directory raises :class:`KeyError` rather than
    folding under a label it was never given.
    """
    from .build_profile_document import RETIRED_TIER_KEY

    name = layer.parent.name
    return RETIRED_TIER_KEY if name in RETIRED_LAYER_DIRS else LAYER_DIR_KINDS[name]


def _hash_resolved_profile(
    raw: dict[str, Any],
    profile_path: Path,
    *,
    conventions: bool = True,
    overlay: Path | None = None,
) -> str:
    """Canonical content hash of a profile dict after ``extends`` resolution.

    Hashes the *resolved* content (canonical JSON, sorted keys) rather than
    file bytes, so comment/ordering churn is invisible while a change in any
    ``extends`` parent is not. The file inputs the resolved profile names — its
    ``data:`` tree, convention directories, the named trigger file and the render
    deltas in ``tiers/`` — are folded in on top by
    :func:`_fold_profile_material`, so a project whose facility data changed
    under an unchanged profile still reads as stale.

    Resolution goes through :func:`resolve_profile_document`, the same call the
    loader makes: the hash can only say honestly whether a built project still
    matches its source if it resolves the *same* document the build resolved.
    That is also what makes a root edit move every render project's hash.

    The excluded-artifact record is folded in as well. It is derived state, not
    profile content, so the canonical JSON cannot see it — yet two renders
    differing only by a ``commands:`` exclusion build different projects, and
    without this they would hash identically and neither would ever read stale.

    What goes in is the profile's declared selection lists, not the *effective*
    post-shadow ones: a bare exclusion of a name the profile also shadows moves
    the hash while rendering byte-identical output, and that over-report is
    accepted on purpose. Subtracting the effective selection would mean
    re-deriving the build's shadow rule inside the hash, and any drift between
    the two would make the hash miss a real divergence — the one failure this
    advisory exists to prevent. Over-reporting costs an idempotent rebuild;
    under-reporting costs a project that silently no longer matches its source,
    so the conservative input is the right one.
    :func:`_warn_shadowed_bare_exclusions` keeps that case loud rather than
    letting an author sit on a no-op exclusion forever.

    ``raw`` is normalized to the canonical field spellings first, so a caller
    that hands over a dict which did not come from
    :func:`~osprey.cli.build_profile_document._read_profile_document` hashes
    the same as one that did — the YAML-surface rename must not move any
    profile's hash. Normalizing before ``extends`` resolution (rather than
    after) keeps the deep merge child-wins across a mixed-spelling chain.

    ``overlay`` is the host-variant profile this repo builds, when one is
    selected — the layer ``raw`` does not carry, because the hash resolves the
    TRACKED document while the render merges the overlay over it. It is folded
    as file material rather than merged in: :func:`_fold_source_tree`
    contributes the file's NAME beside its content, so an edit to the selected
    overlay and a switch between two byte-identical overlays both move the hash,
    while an overlay this host did not select folds nothing. No overlay folds
    nothing at all, which is what keeps every repo that defines no variants
    hashing exactly as it did before variants existed.
    """
    import hashlib
    import json

    document = resolve_profile_document(raw, profile_path, warn=False)
    canonical = json.dumps(document.raw, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8"))
    _fold_profile_material(digest, document.raw, document.root_dir, conventions=conventions)
    if overlay is not None:
        _fold_source_tree(digest, "variant", overlay)
    # Sorted so the record's iteration order cannot move a hash; a profile with
    # no exclusions folds nothing and keeps the digest it had before.
    for artifact in sorted(document.excluded_artifacts):
        digest.update(b"exclude:")
        digest.update(artifact.encode("utf-8"))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def compute_preset_hash(preset_name: str) -> str | None:
    """Content hash of a bundled preset as resolved (post-``extends``).

    Stamped into ``.osprey-manifest.json`` at build time and compared by the
    deploy-side staleness advisory
    (:mod:`osprey.deployment.staleness`). Returns ``None`` when the preset is
    unknown or unreadable — callers treat that as "cannot compare", never as
    drift.

    Convention material is deliberately not folded: bundled presets share one
    package directory, so folding it would make every preset's hash move when
    any other preset's neighbouring directories changed.
    """
    try:
        raw, path = _load_preset_raw(preset_name)
        if "/" in preset_name:
            # A kind-qualified tier or app (`tiers/readonly`) is a layer, not a
            # profile: resolving it would lay it over a `profile.yml` beside it
            # that the package does not have. Its content is its own document.
            import hashlib
            import json

            canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
            return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
        return _hash_resolved_profile(raw, path, conventions=False)
    except Exception:
        return None


def compute_profile_hash(profile_path: Path) -> str | None:
    """Content hash of a repo's own ``profile.yml`` as resolved (post-``extends``).

    Counterpart of :func:`compute_preset_hash` for profile-built repos — the
    drift-fingerprint input, hashed against the profile path the manifest
    recorded. Returns ``None`` when the file is missing or unreadable.

    The host variant is folded in from the profile's own directory rather than
    passed by the caller. That is deliberate: this one function is what the
    build stamps and what ``osprey up``/``status`` recompute, and a fingerprint
    whose variant arrived as an argument would be a fingerprint two call sites
    could spell differently — the exact way a stale build goes unnoticed. Read
    off the directory, both sides of the comparison see the same selection, and
    a directory with no ``.env.variant`` (a preset's) folds nothing. A tier or
    app reads it off its deployment root, because the variant reaches its
    render through that root.
    """
    try:
        from .variant_selection import active_variant_overlay

        path = Path(profile_path)
        raw = _read_profile_document(path)
        if not isinstance(raw, dict):
            return None
        root_dir, _kind = resolve_profile_root(path)
        return _hash_resolved_profile(raw, path, overlay=active_variant_overlay(root_dir))
    except Exception:
        return None
