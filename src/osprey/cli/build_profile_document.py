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

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from osprey_connectors import yaml_loader
from osprey_connectors.errors import BuildProfileError

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
#: (``osprey_connectors.workspace.RETIRED_LAYER_DIRS``), kept so such a repo
#: hashes as it always did.
RETIRED_TIER_KEY = "persona"

#: The retired spelling of ``modules.web_terminals.default_tier``.
RETIRED_DEFAULT_TIER_KEY = "default_persona"

#: The retired render catalog under ``modules.web_terminals``.
RETIRED_CATALOG_KEY = "personas"

#: Catalog fields the build derives from the tier or app file's location.
_DERIVED_CATALOG_KNOBS = frozenset({"build_profile", "project", "project_path"})

#: Catalog fields that are top-level keys of the tier or app file instead.
_LAYER_FILE_KNOBS = ("extra_mounts", "seed_base", "image")

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
            if knob in _LAYER_FILE_KNOBS:
                remedy = f"set `{knob}` at the top of tiers/{name}.yml (or apps/{name}.yml)"
            elif knob == "landing_group":
                remedy = f"group landing cards with `{_WT_LABEL}.landing.groups`"
            else:
                remedy = "it has no successor"
            _warn_retired(source, spelled, f"`{spelled}` is dropped; {remedy}")
    return raw


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
        path.read_text(encoding="utf-8"), str(path) if source is None else source
    )


def _parse_profile_document(text: str, source: str) -> Any:
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
    return raw
