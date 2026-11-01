"""Raw profile-YAML document reads — the single parse point.

Every layer of the profile pipeline reads its YAML through
:func:`_read_profile_document`: the bundled preset, each host-variant overlay,
the ``extends:`` parent inside :func:`~osprey.cli.build_profile_merge._resolve_extends`,
the positional profile file, and the hashing path. One read point is what lets
the empty-collection flattening and the retired-roster-key rewrite below apply
to every layer identically, so an emptied selection list means the same thing
in the merge as in the parser, and a profile spelled the retired way hashes,
merges and renders exactly like its rewrite.

Kept a leaf — it imports nothing else from the profile pipeline — so
:mod:`osprey.cli.build_profile_presets` can use it without gaining a dependency
on the parser.
"""

from __future__ import annotations

import copy
import functools
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from osprey.utils.workspace import LAYER_DIR_KINDS, RETIRED_LAYER_DIRS
from osprey_connectors import yaml_loader
from osprey_connectors.errors import BuildProfileError
from osprey_connectors.types import (
    TIER_WORDS,
    baseline_target,
    configured_targets,
    resolve_control_system_type,
    resolve_target,
    switch_capable,
)

_LOGGER = logging.getLogger(__name__)

# Top-level collection keys whose value is a plain selection — a list of names,
# or the panel-layout mapping — and whose empty spelling therefore has exactly
# one reading: "none of these". YAML parses a key written with nothing under it
# (every entry commented out, or a delta stub whose list was cleared) to
# ``None``, which is neither the collection the field's type promises nor the
# value the author wrote, so it is flattened to the empty collection here.
#
# Confined to these fields on purpose. The block-shaped keys (``config:``,
# ``lifecycle:``, ``env:``, the connector blocks) are read by parsers that
# already narrow their own empty spelling, and an empty block there is a
# question about inheritance rather than a selection — so they keep the reading
# their own parser gives them.
_EMPTY_COLLECTION_KEYS: dict[str, type] = {
    "hooks": list,
    "rules": list,
    "skills": list,
    "agents": list,
    "output_styles": list,
    "web_panels": list,
    "dependencies": list,
    "panel_presets": dict,
}


def _normalize_empty_collections(raw: dict[str, Any]) -> dict[str, Any]:
    """Rewrite present-but-empty selection keys to the empty collection, in place.

    Run on every document as it is parsed, and so *before* the ``extends`` and
    tier-delta merges. That placement is the point: it makes an empty
    ``web_panels:`` mean precisely what ``web_panels: []`` means, in the merge
    as well as in the parser. Normalizing only at parse time would leave
    ``None`` to win the merge key-by-key, and a tier delta whose list the
    author had emptied would silently *subtract* the root profile's whole
    selection — a power no list spelling has (that is what ``exclude:`` is
    for).

    A value of the wrong shape is left exactly as written: the schema
    validation downstream names it far better than a silent rewrite would.

    Args:
        raw: One raw profile mapping, mutated in place.

    Returns:
        ``raw``.
    """
    for key, empty in _EMPTY_COLLECTION_KEYS.items():
        if key in raw and raw[key] is None:
            raw[key] = empty()
    return raw


# ── Retired roster spellings ─────────────────────────────────────────────────
#
# The roster's access level is a *tier*: ``users[].tier``, ``default_tier`` and
# ``authorization.roles.<r>.tier`` name a file in ``tiers/``, and the build
# derives every render from the files themselves. The ``persona`` spellings and
# the ``personas:`` catalog that preceded them are read for one release window
# by rewriting them here, at the single parse point, so nothing downstream ever
# sees them: the merge, the hash and the render all read the new spelling.

#: Where the roster lives under ``config:``, as path segments.
_WEB_TERMINALS = ("modules", "web_terminals")
_WT_LABEL = ".".join(_WEB_TERMINALS)

#: The retired tier pin: ``users[].<key>`` and ``authorization.roles.<r>.<key>``.
#: It is also the fold label of a file in a retired layer directory
#: (``osprey.utils.workspace.RETIRED_LAYER_DIRS``), kept so such a repo
#: hashes as it always did.
RETIRED_TIER_KEY = "persona"

#: The retired spelling of ``modules.web_terminals.default_tier``.
RETIRED_DEFAULT_TIER_KEY = "default_persona"

#: The retired render catalog under ``modules.web_terminals``.
RETIRED_CATALOG_KEY = "personas"

#: Catalog fields the build derives from the tier or app file's location.
_DERIVED_CATALOG_KNOBS = frozenset({"build_profile", "project", "project_path"})

#: Per-render keys a tier or app file may spell and a root profile never does:
#: what the card running that render mounts, seeds from, and runs as. They are
#: also the retired catalog's fields that moved to the top of those files. Read
#: from the file by the render projection
#: (:func:`~osprey.deployment.web_terminals.render_naming.render_projection`);
#: no profile field holds them, so a resolved render document does not carry
#: them either.
RENDER_KEYS = frozenset({"extra_mounts", "seed_base", "image"})

#: ``(source, key)`` pairs already warned about. A document is parsed several
#: times per command (load, hash, lint), and one line per key is the contract.
_WARNED_RETIRED_KEYS: set[tuple[str, str]] = set()


@dataclass
class _RosterSpellings:
    """Every place one document spells a roster key, old way and new way.

    Each rename entry is ``(container, key)``: the mapping holding the key and
    the key exactly as written there (possibly dotted). Catalog entries are
    ``(container, key, path)`` with *path* the key's segments below
    ``personas``.
    """

    default_old: list[tuple[dict, str]] = field(default_factory=list)
    default_new: bool = False
    users_old: list[tuple[dict, str]] = field(default_factory=list)
    users_new: bool = False
    roles_old: list[tuple[dict, str]] = field(default_factory=list)
    roles_new: bool = False
    catalog: list[tuple[dict, str, tuple[str, ...], Any]] = field(default_factory=list)
    renders: bool = False


def _scan_roster(mapping: dict, prefix: tuple[str, ...], found: _RosterSpellings) -> None:
    """Record every roster key under *mapping*, whose path from ``config`` is *prefix*.

    Handles every spelling a config block accepts — fully dotted, partly
    nested, fully nested — by splitting each key on dots and matching the
    accumulated path. A value of the wrong shape is skipped, never coerced: the
    schema names it better.
    """
    depth = len(_WEB_TERMINALS)
    for key, value in mapping.items():
        if not isinstance(key, str):
            continue
        path = prefix + tuple(key.split("."))
        if path[:depth] != _WEB_TERMINALS:
            if _WEB_TERMINALS[: len(path)] == path and isinstance(value, dict):
                _scan_roster(value, path, found)
            continue
        rel = path[depth:]
        if not rel:
            if isinstance(value, dict):
                _scan_roster(value, path, found)
            continue
        head = rel[0]
        if rel == ("default_persona",):
            found.default_old.append((mapping, key))
        elif rel == ("default_tier",):
            found.default_new = True
        elif rel == ("users",):
            for entry in value if isinstance(value, list) else ():
                if isinstance(entry, dict):
                    if "persona" in entry:
                        found.users_old.append((entry, "persona"))
                    if "tier" in entry:
                        found.users_new = True
        elif head == "personas":
            found.catalog.append((mapping, key, rel[1:], value))
        elif head == "renders":
            found.renders = True
        elif head == "authorization":
            if len(rel) > 1 and rel[1] != "roles":
                continue
            if len(rel) == 4 and rel[3] == "persona":
                found.roles_old.append((mapping, key))
            elif len(rel) == 4 and rel[3] == "tier":
                found.roles_new = True
            elif len(rel) < 4 and isinstance(value, dict):
                _scan_roster(value, path, found)


def _rename_last_segment(container: dict, key: str, new_leaf: str) -> None:
    """Rename *key* in *container* to end in *new_leaf*, keeping its position."""
    renamed = key.rsplit(".", 1)[0] + "." + new_leaf if "." in key else new_leaf
    items = [(renamed if k == key else k, v) for k, v in container.items()]
    container.clear()
    container.update(items)


def _catalog_knobs(path: tuple[str, ...], value: Any) -> Iterator[tuple[str, str, Any]]:
    """``(name, knob, value)`` for every field one catalog key carried, any spelling."""
    if len(path) >= 2:
        yield path[0], path[1], value
    elif isinstance(value, dict):
        for key, inner in value.items():
            if isinstance(key, str):
                yield from _catalog_knobs(path + tuple(key.split(".")), inner)


def _warn_retired(source: str, key: str, message: str) -> None:
    if (source, key) in _WARNED_RETIRED_KEYS:
        return
    _WARNED_RETIRED_KEYS.add((source, key))
    _LOGGER.warning("%s: %s; run `osprey init --revalidate`", source, message)


def _refuse_mixed(source: str, old: str, new: str) -> BuildProfileError:
    return BuildProfileError(
        f"{source} spells both `{old}` and `{new}`, two spellings of one key. "
        f"Keep `{new}` only — run `osprey init --revalidate` to rewrite the "
        "retired spelling."
    )


def _refuse_mixed_roster(found: _RosterSpellings, source: str) -> None:
    """Refuse a document that spells one roster key both the retired and the tier way."""
    for old, new, mixed in (
        (RETIRED_DEFAULT_TIER_KEY, "default_tier", found.default_old and found.default_new),
        (f"users[].{RETIRED_TIER_KEY}", "users[].tier", found.users_old and found.users_new),
        (
            f"authorization.roles.<role>.{RETIRED_TIER_KEY}",
            "authorization.roles.<role>.tier",
            found.roles_old and found.roles_new,
        ),
        (RETIRED_CATALOG_KEY, "renders", found.catalog and found.renders),
    ):
        if mixed:
            raise _refuse_mixed(source, f"{_WT_LABEL}.{old}", f"{_WT_LABEL}.{new}")


def _normalize_retired_roster_keys(raw: dict[str, Any], source: str) -> dict[str, Any]:
    """Rewrite the retired roster spellings under ``config:`` to the tier ones, in place.

    ``users[].persona`` → ``users[].tier``, ``default_persona`` →
    ``default_tier``, ``authorization.roles.<r>.persona`` →
    ``authorization.roles.<r>.tier``, each renamed where it stands (same
    container, same dotted spelling, same position) so the result is the
    document a new-style author would have written and hashes identically.
    The ``personas:`` catalog is dropped: a render's project and path derive
    from its file in ``tiers/`` or ``apps/``; ``extra_mounts``, ``seed_base``
    and ``image`` are top-level keys of that file and ``landing_group`` has no
    successor, so each of those is named in a warning of its own.

    Each retired key logs one deprecation line per source. A document spelling
    one key both ways is refused before anything is rewritten.

    Args:
        raw: One raw profile mapping, mutated in place.
        source: How to name the document in warnings and errors.

    Returns:
        ``raw``.

    Raises:
        BuildProfileError: When one key is spelled both ways.
    """
    config = raw.get("config")
    if not isinstance(config, dict):
        return raw
    found = _RosterSpellings()
    _scan_roster(config, (), found)

    default_old = f"{_WT_LABEL}.default_persona"
    users_old = f"{_WT_LABEL}.users[].persona"
    roles_old = f"{_WT_LABEL}.authorization.roles.<role>.persona"
    catalog_old = f"{_WT_LABEL}.personas"
    _refuse_mixed_roster(found, source)

    for container, key in found.default_old:
        _rename_last_segment(container, key, "default_tier")
    if found.default_old:
        _warn_retired(
            source, default_old, f"`{default_old}` is retired; read as `{_WT_LABEL}.default_tier`"
        )
    for container, key in found.users_old:
        _rename_last_segment(container, key, "tier")
    if found.users_old:
        _warn_retired(
            source, users_old, f"`{users_old}` is retired; read as `{_WT_LABEL}.users[].tier`"
        )
    for container, key in found.roles_old:
        _rename_last_segment(container, key, "tier")
    if found.roles_old:
        _warn_retired(
            source,
            roles_old,
            f"`{roles_old}` is retired; read as `{_WT_LABEL}.authorization.roles.<role>.tier`",
        )
    if found.catalog:
        _warn_retired(
            source,
            catalog_old,
            f"`{catalog_old}` is retired and ignored; each render is a file in tiers/ or apps/",
        )
    for container, key, path, value in found.catalog:
        del container[key]
        for name, knob, _value in _catalog_knobs(path, value):
            if knob in _DERIVED_CATALOG_KNOBS:
                continue
            spelled = f"{catalog_old}.{name}.{knob}"
            if knob in RENDER_KEYS:
                remedy = f"set `{knob}` at the top of tiers/{name}.yml"
            elif knob == "landing_group":
                remedy = f"group landing cards with `{_WT_LABEL}.landing.groups`"
            else:
                remedy = "it has no successor"
            _warn_retired(source, spelled, f"`{spelled}` is dropped; {remedy}")
    return raw


# ── Retired write-posture spellings ──────────────────────────────────────────
#
# A login tier's write ceiling is one key, ``control_system.writes: never |
# asked | allowed``. The posture it replaces — the flat ``control_system.writes_enabled``
# and every ``control_system.connector.<type>.writes_enabled`` — is read for one
# release window by rewriting it here, at the single parse point, so the merge,
# the hash and the render all read the new key. The window closes at
# ``build_profile_load.RETIRED_WRITE_KEYS_REFUSED_FROM``: a test tripwire, not
# a version compare here, so this module keeps importing nothing from the
# profile pipeline.

#: The control-system section under ``config:``.
_CONTROL_SYSTEM = "control_system"

#: The connector table inside it, as path segments from ``config``. Each direct
#: child of this mapping is ONE connector type, dots and all
#: (``mypkg.Mod``), exactly as :func:`_retired_type_armed` looks it up.
_CONNECTOR_PATH = (_CONTROL_SYSTEM, "connector")

#: The tier key and the retired posture leaf it replaces.
WRITES_KEY_LEAF = "writes"
RETIRED_WRITES_LEAF = "writes_enabled"

_WRITES_LABEL = f"{_CONTROL_SYSTEM}.{WRITES_KEY_LEAF}"
_FLAT_RETIRED_LABEL = f"{_CONTROL_SYSTEM}.{RETIRED_WRITES_LEAF}"

_NEVER, _ASKED, _ALLOWED = TIER_WORDS

#: The tier key's words that arm anything; an app may spell neither.
_ARMING_WORDS = (_ASKED, _ALLOWED)

#: Resolved path -> the retired posture leaves the normalizer removed from the
#: document there (C-ORIGIN). Set or deleted on every parse of a path.
_RETIRED_WRITE_ORIGINS: dict[Path, dict[str, Any]] = {}


@dataclass
class _WriteSpellings:
    """Every place one ``config:`` mapping spells a write-posture key.

    ``flat`` and ``per_type`` entries are ``(container, key, top)``: the
    mapping holding the leaf, the key exactly as written there (possibly
    dotted) and the ``config:`` key the leaf was reached through. ``per_type``
    entries also carry the connector type.
    """

    flat: list[tuple[dict, str, str]] = field(default_factory=list)
    per_type: list[tuple[dict, str, str, str]] = field(default_factory=list)
    new: bool = False
    #: ``config["control_system"]`` when the document spells it nested.
    block: dict | None = None


def _scan_writes(
    mapping: dict, prefix: tuple[str, ...], top: str | None, found: _WriteSpellings
) -> None:
    """Record every write-posture key under *mapping*, whose path from ``config`` is *prefix*.

    Every spelling a config block accepts is matched by splitting keys on dots,
    except a direct child of the connector table, which is one connector type
    whole. A dotted key that runs through the connector table is split all the
    way, as the render splits it, so ``control_system.connector.a.b.writes_enabled``
    is no type's leaf.
    """
    for key, value in mapping.items():
        if not isinstance(key, str):
            continue
        parts = (key,) if prefix == _CONNECTOR_PATH else tuple(key.split("."))
        path = prefix + parts
        entry = key if top is None else top
        if path[0] != _CONTROL_SYSTEM:
            continue
        if path == (_CONTROL_SYSTEM,):
            if isinstance(value, dict):
                if top is None and found.block is None:
                    found.block = value
                _scan_writes(value, path, entry, found)
        elif path == (_CONTROL_SYSTEM, RETIRED_WRITES_LEAF):
            found.flat.append((mapping, key, entry))
        elif path == (_CONTROL_SYSTEM, WRITES_KEY_LEAF):
            found.new = True
        elif path[:2] == _CONNECTOR_PATH and len(path) == 4 and path[3] == RETIRED_WRITES_LEAF:
            found.per_type.append((mapping, key, entry, path[2]))
        elif path[:2] == _CONNECTOR_PATH and len(path) < 4 and isinstance(value, dict):
            _scan_writes(value, path, entry, found)


def _expand_config(mapping: dict, prefix: tuple[str, ...] = ()) -> dict[str, Any]:
    """The nested tree a ``config:`` mapping addresses, dotted keys folded in.

    A leaf spelled more than once holds the value written last in key order,
    and a mapping merges into what earlier keys addressed. A direct child of the
    connector table stays one key (a dotted connector type is one block).
    """
    tree: dict[str, Any] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            continue
        parts = (key,) if prefix == _CONNECTOR_PATH else tuple(key.split("."))
        node = tree
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        leaf = parts[-1]
        if isinstance(value, dict):
            existing = node.get(leaf)
            expanded = _expand_config(value, prefix + parts)
            node[leaf] = {**existing, **expanded} if isinstance(existing, dict) else expanded
        else:
            node[leaf] = value
    return tree


def _retired_leaves(config: dict) -> dict[str, Any]:
    """The retired posture leaves *config* addresses, as a ``control_system`` fragment.

    ``{"writes_enabled": v, "connector": {<type>: {"writes_enabled": v}}}``,
    holding only the leaves present, each with the value the render would
    read. Empty when the document spells none.
    """
    section = _expand_config(config).get(_CONTROL_SYSTEM)
    if not isinstance(section, dict):
        return {}
    leaves: dict[str, Any] = {}
    if RETIRED_WRITES_LEAF in section:
        leaves[RETIRED_WRITES_LEAF] = section[RETIRED_WRITES_LEAF]
    connector = section.get("connector")
    if isinstance(connector, dict):
        per_type = {
            name: {RETIRED_WRITES_LEAF: block[RETIRED_WRITES_LEAF]}
            for name, block in connector.items()
            if isinstance(block, dict) and RETIRED_WRITES_LEAF in block
        }
        if per_type:
            leaves["connector"] = per_type
    return leaves


# ---------------------------------------------------------------------------
# Readers of the retired spellings
#
# The runtime reads only ``control_system.writes``. For the release window the
# build still has to say what a document's retired ``writes_enabled`` leaves
# meant, so the one reading of them lives here, private to the migration code:
# the normalizer's rewrite and the PR-3 posture warnings. They read exactly as
# the runtime read those leaves before the tier key replaced them.
# ---------------------------------------------------------------------------


def _retired_flat_armed(section: Any) -> bool:
    """Whether the flat ``control_system.writes_enabled`` is a literal ``True``."""
    return isinstance(section, dict) and section.get(RETIRED_WRITES_LEAF) is True


def _retired_type_armed(section: Any, connector_type: str | None) -> bool:
    """Whether the retired leaves armed one connector *type*.

    The type's own ``writes_enabled`` leaf answers when its block carries one,
    and only a literal ``True`` arms. A type whose block says nothing (no
    table, no block, a block that is not a mapping, no leaf) inherits the flat
    leaf. The type is one key, never split on its dots.
    """
    connector = section.get("connector") if isinstance(section, dict) else None
    block = (
        connector.get(connector_type) if connector_type and isinstance(connector, dict) else None
    )
    if not isinstance(block, dict) or RETIRED_WRITES_LEAF not in block:
        return _retired_flat_armed(section)
    return block[RETIRED_WRITES_LEAF] is True


def _retired_target_armed(section: Any, target: Any) -> bool:
    """Whether the retired leaves armed one session *target*.

    :func:`_retired_type_armed` for the type *target* resolves to; a target
    that does not resolve has no type, and answers the flat leaf.
    """
    try:
        connector_type = resolve_target(section, target)
    except ValueError:
        return _retired_flat_armed(section)
    return _retired_type_armed(section, connector_type)


def _retired_session_posture(section: Any) -> dict[str, bool]:
    """What the retired leaves armed, per target a session here can select.

    Every configured target when the deployment renders the switch, else the
    one built connector type under its baseline target.
    """
    if switch_capable(section):
        return {
            target: _retired_target_armed(section, target) for target in configured_targets(section)
        }
    built = resolve_control_system_type(section)
    return {baseline_target(section): _retired_type_armed(section, built)}


def _retired_tier(section: Any) -> str:
    """The tier the retired leaves amount to: ``allowed`` or ``never``.

    ``allowed`` when the flat leaf or any connector block's leaf is a literal
    ``True``. Target-blind, as the tier is: one armed machine allows writes for
    every machine. The tier key itself is not consulted.
    """
    if _retired_flat_armed(section):
        return _ALLOWED
    connector = section.get("connector") if isinstance(section, dict) else None
    if isinstance(connector, dict) and any(
        isinstance(block, dict) and block.get(RETIRED_WRITES_LEAF) is True
        for block in connector.values()
    ):
        return _ALLOWED
    return _NEVER


def retired_write_answer(config_mapping: Any) -> str | None:
    """The tier a document's retired write-posture leaves amount to.

    ``"allowed"`` when any retired leaf — the flat
    ``control_system.writes_enabled`` or any connector type's
    ``writes_enabled`` — is a literal ``True``, else ``"never"``
    (:func:`_retired_tier`). ``None`` when the mapping spells no
    retired leaf. Every spelling is read (dotted, nested, mixed); a connector
    type is one key, never split on its dots. Pure: *config_mapping* is not
    changed, and the tier key itself is not consulted.

    Args:
        config_mapping: A document's ``config:`` mapping.

    Returns:
        ``"allowed"``, ``"never"`` or ``None``.
    """
    if not isinstance(config_mapping, dict):
        return None
    leaves = _retired_leaves(config_mapping)
    if not leaves:
        return None
    return _retired_tier(leaves)


def retired_write_origin(path: Path) -> dict[str, Any] | None:
    """The retired posture leaves the normalizer removed from the document at *path*.

    A ``control_system`` fragment (see :func:`_retired_leaves`) keyed by the
    resolved path, or ``None`` when the last parse of that path found none.
    Recorded only for documents read from a file. Read-only: callers get a copy.
    """
    origin = _RETIRED_WRITE_ORIGINS.get(Path(path).resolve())
    return None if origin is None else copy.deepcopy(origin)


def _document_kind(path: Path | None) -> str:
    """``tier``, ``layer`` or ``root``, from where the document lives.

    Only the directory holding the file counts, so a root profile in a repo
    that happens to sit under a ``tiers/`` directory of something else is
    still a root. A document with no file is a root.
    """
    if path is None:
        return "root"
    parent = path.parent.name
    if parent in LAYER_DIR_KINDS:
        return LAYER_DIR_KINDS[parent]
    if parent in RETIRED_LAYER_DIRS:
        return "layer"
    return "root"


def _kept_flat_leaf(found: _WriteSpellings) -> tuple[dict, str] | None:
    """The retired flat leaf the tier key takes the place of, as ``(container, key)``.

    With a nested ``control_system:`` block, the last flat leaf written inside
    it; else the last dotted flat leaf (the one the render reads); ``None``
    when neither exists. Shared by the parse-point rewrite
    (:func:`_place_tier_key`) and ``osprey init --revalidate``, so both put the
    tier key in the same place.
    """
    if found.block is not None:
        inside = [(c, k) for c, k, top in found.flat if top == _CONTROL_SYSTEM]
        return inside[-1] if inside else None
    if found.flat:
        container, key, _top = found.flat[-1]
        return (container, key)
    return None


def _mixed_writes_error(found: _WriteSpellings, source: str) -> BuildProfileError:
    """The refusal of a document spelling a retired write leaf beside ``control_system.writes``."""
    old = (
        _FLAT_RETIRED_LABEL
        if found.flat
        else f"{_CONTROL_SYSTEM}.connector.<type>.{RETIRED_WRITES_LEAF}"
    )
    return _refuse_mixed(source, old, _WRITES_LABEL)


def _place_tier_key(config: dict, found: _WriteSpellings, answer: str) -> None:
    """Write ``control_system.writes`` where the retired posture stood, and drop the retired leaves.

    The new leaf takes the place of a retired flat leaf, in its spelling, so
    the rewrite is the document a new-style author would have written:

    - When the document spells a nested ``control_system:`` block, the leaf
      goes inside it — in place of the last flat leaf written there, else at
      the block's end. A dotted ``control_system.writes`` beside a nested
      block would sit apart from it in every merge, and whichever came later
      in key order would decide the render.
    - Otherwise, in place of the last dotted flat leaf (the one the render
      reads), else as a dotted key at the position of the first per-type
      leaf's ``config:`` key — never appended at the end, where it would
      outrank a later layer's key by order alone.
    """
    kept = _kept_flat_leaf(found)
    if kept is not None:
        container, key = kept
        _rename_last_segment(container, key, WRITES_KEY_LEAF)
        renamed = key.rsplit(".", 1)[0] + "." + WRITES_KEY_LEAF if "." in key else WRITES_KEY_LEAF
        container[renamed] = answer
    elif found.block is not None:
        found.block[WRITES_KEY_LEAF] = answer
    else:
        top = found.per_type[0][2]
        items: list[tuple[Any, Any]] = []
        for key, value in config.items():
            if key == top:
                items.append((_WRITES_LABEL, answer))
            items.append((key, value))
        config.clear()
        config.update(items)

    for container, key, _top in found.flat:
        if kept is None or container is not kept[0] or key != kept[1]:
            del container[key]
    for container, key, _top, _name in found.per_type:
        del container[key]


def _warn_retired_writes(found: _WriteSpellings, source: str, answer: str) -> None:
    labels: list[tuple[str, bool]] = []
    for container, key, _top in found.flat:
        labels.append((_FLAT_RETIRED_LABEL, container.get(key) is True))
    for container, key, _top, name in found.per_type:
        labels.append(
            (
                f"{_CONTROL_SYSTEM}.connector.{name}.{RETIRED_WRITES_LEAF}",
                container.get(key) is True,
            )
        )
    for label, _arms in labels:
        reading = f"this document reads as `{_WRITES_LABEL}: {answer}`"
        _warn_retired(source, label, f"`{label}` is retired; {reading}")


def _tier_key_values(mapping: dict, prefix: tuple[str, ...] = ()) -> Iterator[Any]:
    """Every value *mapping* (a ``config:`` block, path *prefix*) gives ``control_system.writes``.

    Every spelling — dotted, nested, mixed — is visited, not only the one the
    render would keep, so a value a later key shadows is still checked.
    """
    for key, value in mapping.items():
        if not isinstance(key, str):
            continue
        path = prefix + tuple(key.split("."))
        if path[0] != _CONTROL_SYSTEM:
            continue
        if path == (_CONTROL_SYSTEM, WRITES_KEY_LEAF):
            yield value
        elif path == (_CONTROL_SYSTEM,) and isinstance(value, dict):
            yield from _tier_key_values(value, path)


def validate_tier_key_value(raw: dict[str, Any], source: str) -> None:
    """Refuse a ``control_system.writes`` that is not ``never``, ``asked`` or ``allowed``.

    Runs after :func:`normalize_retired_write_keys` on every document it
    sees — files, text and the ``--set`` layer — so a value it wrote is
    checked too. Only the exact lowercase strings pass: a boolean, ``null``,
    ``"Allowed"`` or ``"yes"`` is refused rather than read as a tier. Whether
    a login may spell an arming word is its audience's question, judged where
    the render is composed.

    Args:
        raw: One raw profile mapping; not changed.
        source: How to name the document in the error.

    Raises:
        BuildProfileError: On any other value.
    """
    config = raw.get("config")
    if not isinstance(config, dict):
        return
    values = list(_tier_key_values(config))
    if not values:
        return
    for value in values:
        if not (isinstance(value, str) and value in TIER_WORDS):
            raise BuildProfileError(
                f"{source}: `{_WRITES_LABEL}` must be `{_NEVER}`, `{_ASKED}` or `{_ALLOWED}`, "
                f"got {value!r}"
            )


def normalize_retired_write_keys(
    raw: dict[str, Any], source: str, *, path: Path | None = None
) -> dict[str, Any]:
    """Rewrite the retired write-posture leaves under ``config:`` to the tier key, in place.

    Acts only on a document that spells at least one retired leaf — the flat
    ``control_system.writes_enabled`` or a ``control_system.connector.<type>.
    writes_enabled`` — in any spelling; a silent document is returned untouched,
    so a silent tier, overlay or ``extends`` child inherits its host's tier.
    The retired leaves are removed (a connector block they empty is left for
    the render's checks) and ``control_system.writes`` is written where they
    stood (:func:`_place_tier_key`): :func:`retired_write_answer`, for every
    document. A login named by ``shared:`` is sealed ``never`` where its render
    is composed, which reads the recorded origin to tell a rewritten
    ``allowed`` from an authored one.

    With a path, the removed leaves are recorded for
    :func:`retired_write_origin`, and a parse that finds none deletes the
    record, so a file rewritten in the same process leaves nothing stale.

    Each retired key logs one deprecation line per source. A document spelling
    a retired leaf beside ``control_system.writes`` is refused before anything
    is rewritten. Every document, silent or rewritten, then goes through
    :func:`validate_tier_key_value`, so each caller of this function — the
    parse point and the ``--set`` layer — checks the tier key's value too.

    Args:
        raw: One raw profile mapping, mutated in place.
        source: How to name the document in warnings and errors.
        path: The file the document was read from, when there is one.

    Returns:
        ``raw``.

    Raises:
        BuildProfileError: When the document spells both the retired keys and
            ``control_system.writes``, or the tier key has a value
            :func:`validate_tier_key_value` refuses.
    """
    key = None if path is None else Path(path).resolve()
    config = raw.get("config")
    found = _WriteSpellings()
    if isinstance(config, dict):
        _scan_writes(config, (), None, found)
    if not (found.flat or found.per_type):
        if key is not None:
            _RETIRED_WRITE_ORIGINS.pop(key, None)
        validate_tier_key_value(raw, source)
        return raw
    assert isinstance(config, dict)
    if found.new:
        if key is not None:
            _RETIRED_WRITE_ORIGINS.pop(key, None)
        raise _mixed_writes_error(found, source)

    leaves = _retired_leaves(config)
    answer = _retired_tier(leaves)
    _warn_retired_writes(found, source, answer)

    _place_tier_key(config, found, answer)

    if key is not None:
        _RETIRED_WRITE_ORIGINS[key] = leaves
    validate_tier_key_value(raw, source)
    return raw


# ── Retired per-tool policies for the governed tools ────────────────────────
#
# The write switch's slider (off / asked / allowed) decides whether a governed
# write asks a person first, so ``approval.tools.<tool>`` for one of those tools
# states a second answer nothing reads. It is dropped here, at the single parse
# point, with one warning per document, until
# ``build_profile_load.RETIRED_APPROVAL_POLICY_REFUSED_FROM`` (a test tripwire,
# as for the write keys). Every other tool keeps its policy.

#: Where the per-tool policies live under ``config:``, as path segments.
APPROVAL_TOOLS_PATH = ("approval", "tools")

#: The warning's dedupe key: one line per document, however many tools it names.
_APPROVAL_POLICY_LABEL = ".".join(APPROVAL_TOOLS_PATH)

#: The hook presets a facility server attaches to every tool that together make
#: its writes the switch's to decide (the registry's ``rule_carries_both_write_hooks``).
_BOTH_WRITE_PRESETS = frozenset({"writes_check", "approval"})


@functools.cache
def _framework_governed_tools() -> frozenset[str]:
    """Bare names of every framework tool the switch decides for, enabled by default or not.

    Read from the registry's one producer (``governed_write_tools``) over every
    framework server's own hook rules, so a tool that joins or leaves the
    governed set there joins or leaves it here. Imported lazily: the registry
    is a heavier module than this parse point otherwise needs.
    """
    from osprey.registry.mcp import FRAMEWORK_SERVERS, _hook_rule_to_dict, governed_write_tools

    servers = [
        {"enabled": True, "hooks_pre": [_hook_rule_to_dict(rule) for rule in sdef.hooks_pre]}
        for sdef in FRAMEWORK_SERVERS.values()
    ]
    return frozenset(tool.rsplit("__", 1)[-1] for tool in governed_write_tools(servers))


def _declared_governed_tools(raw: dict[str, Any]) -> set[str]:
    """Bare names of the tools *raw*'s own ``mcp_servers`` hand to the switch.

    A server's ``writes:`` tools (and the warning-window tools gated beside
    them) carry both write hooks; a server whose ``hooks.pre_tool_use`` names
    both ``writes_check`` and ``approval`` carries them on every tool, so each
    tool its ``permissions`` name is governed. Malformed entries are skipped:
    the loader refuses them with its own message.
    """
    servers = raw.get("mcp_servers")
    if not isinstance(servers, dict):
        return set()
    from osprey.registry.mcp import _declared_write_rule_tools

    tools: set[str] = set()
    for name, spec in servers.items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            continue
        perms = spec.get("permissions")
        perms = perms if isinstance(perms, dict) else {}
        perms = {
            k: [e for e in v if isinstance(e, str)] if isinstance(v, list) else []
            for k, v in perms.items()
            if k in ("allow", "ask")
        }
        writes = spec.get("writes")
        if isinstance(writes, list):
            declared = tuple(str(tool) for tool in writes if isinstance(tool, str))
            tools.update(_declared_write_rule_tools(name, declared, perms))
        hooks = spec.get("hooks")
        presets = hooks.get("pre_tool_use") if isinstance(hooks, dict) else None
        if isinstance(presets, list) and set(presets) >= _BOTH_WRITE_PRESETS:
            prefix = f"mcp__{name}__"
            for entry in (*perms.get("allow", []), *perms.get("ask", [])):
                tool = entry.removeprefix(prefix)
                if tool and not set("*.|()[]") & set(tool):
                    tools.add(tool)
    return tools


def governed_policy_tools(raw: dict[str, Any]) -> frozenset[str]:
    """The bare tool names whose ``approval.tools`` policy *raw* may no longer state.

    The framework's governed tools plus the ones this same document's servers
    declare (:func:`_declared_governed_tools`). A tier or app declares no
    servers, so the framework set is what applies there.
    """
    return _framework_governed_tools() | _declared_governed_tools(raw)


def governed_policy_spellings(
    config: dict, governed: frozenset[str], prefix: tuple[str, ...] = ()
) -> list[tuple[dict, str, str, tuple[tuple[dict, str], ...]]]:
    """Every ``approval.tools.<tool>`` under *config* for a tool in *governed*.

    Each hit is ``(container, key, tool, parents)``: the mapping holding the
    key, the key as spelled there (dotted, nested or mixed), the bare tool, and
    the ``(mapping, key)`` chain from *config* down to the container, so a
    caller can prune what the removal empties.
    """
    hits: list[tuple[dict, str, str, tuple[tuple[dict, str], ...]]] = []
    depth = len(APPROVAL_TOOLS_PATH)
    for key, value in list(config.items()):
        if not isinstance(key, str):
            continue
        path = prefix + tuple(key.split("."))
        if len(path) == depth + 1 and path[:depth] == APPROVAL_TOOLS_PATH:
            if path[depth] in governed:
                hits.append((config, key, path[depth], ()))
        elif (
            len(path) <= depth
            and path == APPROVAL_TOOLS_PATH[: len(path)]
            and isinstance(value, dict)
        ):
            for container, inner, tool, parents in governed_policy_spellings(value, governed, path):
                hits.append((container, inner, tool, ((config, key), *parents)))
    return hits


def normalize_retired_approval_policies(raw: dict[str, Any], source: str) -> dict[str, Any]:
    """Drop ``approval.tools.<tool>`` for every governed tool under ``config:``, in place.

    The governed set is :func:`governed_policy_tools` of *raw*. A silent
    document is untouched and quiet; one that spells any such policy loses
    those keys (and any ``approval.tools`` / ``approval`` mapping the removal
    empties) and logs one warning per *source* naming the slider and the three
    consequences. Every other tool's policy, and every other ``approval.*``
    key, survives.

    Args:
        raw: One raw profile mapping, mutated in place.
        source: How to name the document in the warning.

    Returns:
        ``raw``.
    """
    config = raw.get("config")
    if not isinstance(config, dict):
        return raw
    hits = governed_policy_spellings(config, governed_policy_tools(raw))
    if not hits:
        return raw
    for container, key, _tool, parents in hits:
        container.pop(key, None)
        node: dict = container
        for parent, parent_key in reversed(parents):
            if node or parent.get(parent_key) is not node:
                break
            del parent[parent_key]
            node = parent
    tools = sorted({tool for _c, _k, tool, _p in hits})
    _warn_retired(source, _APPROVAL_POLICY_LABEL, retired_approval_policy_message(tools))
    return raw


def retired_approval_policy_message(tools: list[str]) -> str:
    """The warning (and revalidate report) for the dropped policies of *tools*."""
    spelled = ", ".join(f"`{_APPROVAL_POLICY_LABEL}.{tool}`" for tool in tools)
    return (
        f"{spelled} dropped: the write switch's slider on the control-target chip "
        "(off / asked / allowed) decides whether a governed write asks, so a "
        "per-tool policy for it is no longer read. Three consequences: a `skip` "
        "policy no longer runs it unprompted (it prompts at asked; pick allowed on "
        "an allowed tier for that); an `always` policy no longer prompts at "
        "allowed (an `execute: always` loses its prompt on read-only code); and "
        "`approval.enabled: false` no longer silences governed tools (it still "
        "silences every other tool)"
    )


def _read_profile_document(path: Path, source: str | None = None) -> Any:
    """Read one raw profile-YAML document.

    Args:
        path: File to read.
        source: How to name the file in error messages (defaults to ``path``).

    Returns:
        The parsed document — normalized in place when it is a mapping.

    Raises:
        BuildProfileError: On invalid YAML.
    """
    return _parse_profile_document(
        path.read_text(encoding="utf-8"), str(path) if source is None else source, path=path
    )


def _parse_profile_document(text: str, source: str, *, path: Path | None = None) -> Any:
    """Parse one raw profile-YAML document.

    The only YAML parse of a profile document in the pipeline (pinned
    by a guard test), reached from files through :func:`_read_profile_document`
    and directly for a document that exists only as text — the profile the
    emitter would write. Mapping-shape checks stay with the callers so each
    keeps naming its own layer ("Override must be a YAML mapping", ...); a
    non-mapping document is returned exactly as parsed.

    Args:
        text: The document.
        source: How to name it in error messages.
        path: Private: the file the text was read from, passed by
            :func:`_read_profile_document`. The retired write keys read the
            document's kind from it and record their origin under it.

    Returns:
        The parsed document — normalized in place when it is a mapping.

    Raises:
        BuildProfileError: On invalid YAML.
    """
    try:
        raw = yaml_loader.safe_load(text)
    except yaml.YAMLError as e:
        raise BuildProfileError(f"Invalid YAML in {source}: {e}") from e
    if isinstance(raw, dict):
        _normalize_empty_collections(raw)
        _normalize_retired_roster_keys(raw, source)
        normalize_retired_write_keys(raw, source, path=path)
        normalize_retired_approval_policies(raw, source)
    elif path is not None:
        _RETIRED_WRITE_ORIGINS.pop(Path(path).resolve(), None)
    return raw
