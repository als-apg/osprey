"""``osprey init --revalidate``: bring an existing deployment repo to this OSPREY, in place.

A repo made before tiers and apps spells its roster with the retired keys
(the ``RETIRED_*`` constants of :mod:`osprey.cli.build_profile_document`), keeps
its per-card files in the retired layer directory and runs the logbook and
knowledge cards as fake users pinned to a tier. This module rewrites such a repo
in place, with the round-trip writer so every comment the operator kept survives:

1. The retired layer directory becomes ``tiers/`` (``git mv`` when the files are
   tracked).
2. The retired default-tier, tier-pin and role-target keys are renamed to
   ``default_tier``, ``users[].tier`` and ``authorization.roles.<r>.tier`` where
   they stand, in ``profile.yml`` and every variant overlay; the retired render
   catalog is dropped and the ``extra_mounts``, ``seed_base``
   and ``image`` it carried move to the top of the tier or app file they name.
3. A file made from a retired preset and never edited (its data and its text
   below ``name:``, comments included, hash to what the previous release
   emitted) is re-emitted from the bundled source: a tier as
   ``tiers/<n>.yml``, the logbook and knowledge cards as the bundled app in
   ``apps/``. The users that pinned such an app leave the roster and the app
   joins ``shared:``. An edited file stays where it is, and the step left to
   do by hand is printed, as it is for every catalog field, role or pin the
   rewrite cannot carry over.
4. Every ``config:`` leaf the profile leaves to its preset is filled in from
   that preset, under the preset's own comment; the retired ``app_template:``
   key is dropped. The deployment's own selections (the keys the manifest
   marks ``deployment-data`` or ``data-map``) are never filled: the output
   names each one left out.
5. ``providers.yml`` is refreshed: the entries OSPREY ships come back fresh,
   and every entry the operator added is kept.
6. ``provenance:`` is restamped with the preset, its hash and the catalog's
   hash, and ``requires_osprey_version`` at the schema floor, so a release
   that predates the vocabulary refuses the rewritten repo.

Every file is read and rewritten in memory before the repo is touched, so a
file that cannot be read leaves the repo as it was. The result is reported as
a unified diff of every file changed.
"""

from __future__ import annotations

import difflib
import hashlib
import io
import json
import re
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import yaml
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import CommentMark, YAMLError
from ruamel.yaml.tokens import CommentToken

from osprey.profiles.providers import PROVIDERS_FILENAME
from osprey.utils.config_writer import _yaml as _ROUND_TRIP
from osprey_connectors.errors import BuildProfileError
from osprey_connectors.workspace import (
    APPS_DIRNAME,
    LAYER_DIRNAMES,
    PROFILE_FILENAME,
    RETIRED_LAYER_DIRS,
    TIERS_DIRNAME,
    layer_dir_kind,
)

from .build_profile_document import (
    _LAYER_FILE_KNOBS,
    _WEB_TERMINALS,
    _WT_LABEL,
    RETIRED_CATALOG_KEY,
    _catalog_knobs,
    _refuse_mixed_roster,
    _RosterSpellings,
    _scan_roster,
)
from .build_profile_emit import _drop_key_and_pre_comment
from .build_profile_load import _PROFILE_SCHEMA_MIN_OSPREY
from .build_profile_merge import compute_preset_hash
from .build_profile_presets import PRESET_DATA_BUNDLE_KEY
from .variant_selection import known_variants, variant_profile_path

#: The directories whose files are read as tiers, retired ones first, so a file
#: in ``tiers/`` wins over a same-named one still in a retired directory.
_TIER_DIRS_RETIRED_FIRST: tuple[str, ...] = tuple(
    reversed([d for d in LAYER_DIRNAMES if layer_dir_kind(d) == "tier"])
)

CONTEXT_DIRNAME = "web-terminal-context"


class RetiredPreset(NamedTuple):
    """One retired per-card preset, as the previous release emitted it."""

    kind: str
    """``tiers`` or ``apps``: what the card is now."""
    successor: str
    """The bundled tier or app that replaces it."""
    content_hash: str
    """:func:`_content_hash` of the preset's data."""
    text_hash: str
    """:func:`_text_hash` of the file the previous release emitted from it."""


#: The retired per-card presets by name. Both hashes are frozen here because
#: the retired presets leave the package. A file whose data AND text below its
#: ``name:`` line (comments included) still hash to them was never edited, so
#: re-emitting it from the successor loses nothing the operator wrote.
RETIRED_PRESETS: dict[str, RetiredPreset] = {
    "control-assistant-readonly": RetiredPreset(
        TIERS_DIRNAME,
        "readonly",
        "sha256:2cec7785b7fdb90d79c76d008a256ad1ecd34fe034eb417e5e6795a5e84c82f3",
        "sha256:a02b9e42d12e2de3bbdb609e7a638a8bf4f4d89b7609f3772de8adebba75c972",
    ),
    "control-assistant-readwrite": RetiredPreset(
        TIERS_DIRNAME,
        "readwrite",
        "sha256:2acb650dbf5027318c40cda9b4adf5acbe22caee21da6c98b2a781a32146b105",
        "sha256:5288b7f8f39680ca03c86302e8d9b131f63a6cfc6d589e5d921f7c763e4e012f",
    ),
    "control-assistant-admin": RetiredPreset(
        TIERS_DIRNAME,
        "admin",
        "sha256:fa466c4004df3ef3a2625f1d8d2c2aafbf793a754c1ea1cccf7e747998357b95",
        "sha256:a2ae0df9d1108dc3072d450abdaae9e91f8fea135606457928e271b19cc04647",
    ),
    "control-assistant-logbook": RetiredPreset(
        APPS_DIRNAME,
        "logbook",
        "sha256:87a0465d2273266ff1cd5a1087ed49397e33fb3ecc4408e78ad813cedcbc5bf6",
        "sha256:f5afe72dceb1e5b17ef978dae78173b02a97699264e3974c4cf099de9cced910",
    ),
    "control-assistant-knowledge": RetiredPreset(
        APPS_DIRNAME,
        "knowledge",
        "sha256:1fd8929fce26154def1b9aabb1115ddd7a9429c06abf80fb399587de03730e75",
        "sha256:c399b55a15895532bbbb591d115d1a6dd8df9f0e3efeb7d00b15a645053bd2e3",
    ),
}

#: Keys a materialized layer carries that say where it came from, not what it is.
_IDENTITY_KEYS = ("name", "extends", "provenance")

#: The deployment preset whose roster comments a rewritten roster borrows.
_HOST_PRESET = "control-assistant"

_MADE_FROM = re.compile(r"^#\s*Made from the bundled `([^`]+)` preset\.", re.MULTILINE)


@dataclass
class RevalidateResult:
    """What one rewrite changed, and what it left for the operator."""

    diff: str = ""
    manual_steps: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    preset: str | None = None
    """The preset the profile was filled from; ``None`` when none is known."""
    filled: list[str] = field(default_factory=list)
    """Config keys written from the preset."""
    skipped: dict[str, str] = field(default_factory=dict)
    """Lacking config keys left out on purpose → why."""
    providers_kept: tuple[str, ...] = ()
    """The operator's own ``providers.yml`` entries carried over."""
    drift_refusals: int = 0
    """Differences from the preset ``osprey validate`` now refuses."""
    deviation_marker: str = ""
    """The tag a drift finding is claimed with."""


# ─────────────────────────────────────────────────────────────────────────────
# Comments under ruamel's round trip
#
# A comment that precedes a node is stored on the LAST node before it: the
# trailing slot of the previous key (or item), as one token holding that node's
# end-of-line comment, a newline, and every full comment line up to the next
# node. Removing or inserting a node therefore means re-cutting that token:
# lines indented deeper than the node's column still belong to the node before
# it; the rest precede what comes next.
# ─────────────────────────────────────────────────────────────────────────────


def _last_slot(container: Any, key: Any) -> tuple[dict, Any, int]:
    """The comment slot that trails ``container[key]``'s whole subtree."""
    value = container[key]
    if isinstance(value, CommentedMap) and len(value):
        return _last_slot(value, list(value.keys())[-1])
    if isinstance(value, CommentedSeq) and len(value):
        return _last_slot(value, len(value) - 1)
    return container.ca.items, key, 0 if isinstance(container, CommentedSeq) else 2


def _get_tail(slot: tuple[dict, Any, int]) -> tuple[str, int]:
    items, key, pos = slot
    entry = items.get(key)
    token = entry[pos] if entry and len(entry) > pos else None
    if isinstance(token, CommentToken):
        return token.value, token.column
    return "", 0


def _set_tail(slot: tuple[dict, Any, int], text: str, column: int = 0) -> None:
    items, key, pos = slot
    entry = items.setdefault(key, [None, None, None, None])
    while len(entry) <= pos:
        entry.append(None)
    entry[pos] = None if text in ("", "\n") else CommentToken(text, CommentMark(column))


def _split_eol(text: str) -> tuple[str, str]:
    """``(end-of-line comment, the full lines after it)`` of one trailing token."""
    if "\n" not in text:
        return text, ""
    eol, rest = text.split("\n", 1)
    return eol, rest


def _split_inner(rest: str, column: int) -> tuple[str, str]:
    """Cut *rest* into the leading comment lines indented past *column*, and the remainder."""
    lines = rest.splitlines(keepends=True)
    cut = 0
    for line in lines:
        stripped = line.lstrip(" ")
        if not stripped.startswith("#") or len(line) - len(stripped) <= column:
            break
        cut += 1
    return "".join(lines[:cut]), "".join(lines[cut:])


def _join_tail(eol: str, rest: str) -> str:
    return eol + "\n" + rest if (eol or rest) else ""


def _drop_node(container: Any, key: Any, pred: tuple[dict, Any, int] | None, column: int) -> None:
    """Delete ``container[key]`` with its own comments, keeping the next node's.

    *pred* is the slot trailing the node before it; ``None`` when the node is
    the first of its block, whose leading comment then stays where it is.
    """
    _x_eol, x_rest = _split_eol(_get_tail(_last_slot(container, key))[0])
    _inner_x, outer_x = _split_inner(x_rest, column)
    if pred is not None:
        p_text, p_col = _get_tail(pred)
        p_eol, p_rest = _split_eol(p_text)
        inner_p, _ = _split_inner(p_rest, column)
        _set_tail(pred, _join_tail(p_eol, inner_p + outer_x), p_col)
    container.ca.items.pop(key, None)
    del container[key]


def _place_after(
    pred: tuple[dict, Any, int], container: Any, key: Any, column: int, comment: list[str]
) -> None:
    """Hand what followed *pred* to ``container[key]``, just inserted after it, under *comment*.

    *pred* is read before the insertion; the new node's trailing slot after it.
    """
    p_text, p_col = _get_tail(pred)
    p_eol, p_rest = _split_eol(p_text)
    inner, outer = _split_inner(p_rest, column)
    lead = "".join(" " * column + line + "\n" for line in comment)
    _set_tail(pred, _join_tail(p_eol, inner + lead), p_col)
    slot = _last_slot(container, key)
    k_text, k_col = _get_tail(slot)
    k_eol, _ = _split_eol(k_text)
    _set_tail(slot, _join_tail(k_eol, outer), k_col)


# ─────────────────────────────────────────────────────────────────────────────
# What the bundled deployment preset says about the new roster keys
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _PresetRoster:
    """Comments the bundled deployment preset writes around ``shared:``."""

    shared_lead: list[str] = field(default_factory=list)
    shared_items: dict[str, tuple[str, int]] = field(default_factory=dict)
    apps_group_lead: list[str] = field(default_factory=list)
    apps_group: dict[str, str] = field(default_factory=dict)


def _comment_block_above(lines: list[str], index: int) -> list[str]:
    """The contiguous comment lines right above ``lines[index]`` at its indentation."""
    indent = len(lines[index]) - len(lines[index].lstrip(" "))
    block: list[str] = []
    for line in reversed(lines[:index]):
        stripped = line.lstrip(" ")
        if not stripped.startswith("#") or len(line) - len(stripped) != indent:
            break
        block.append(stripped)
    return list(reversed(block))


def _preset_roster() -> _PresetRoster:
    """Read the shared-card comments out of the bundled deployment preset.

    They are the preset's own words, so a rewritten roster reads exactly like
    the one ``osprey init`` writes today.
    """
    from .build_profile_presets import _load_preset_raw

    found = _PresetRoster()
    try:
        _raw, path = _load_preset_raw(_HOST_PRESET)
    except BuildProfileError:
        return found
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "shared:" and not found.shared_lead:
            found.shared_lead = _comment_block_above(lines, i)
            for item in lines[i + 1 :]:
                match = re.match(r"^(\s*- ([^\s#]+)\s*)(#.*)?$", item)
                if not match:
                    break
                if match.group(3):
                    found.shared_items[match.group(2)] = (match.group(3), len(match.group(1)))
        elif stripped == "- type: apps" and not found.apps_group_lead:
            found.apps_group_lead = _comment_block_above(lines, i)
            found.apps_group = {"type": "apps"}
            indent = len(line) - len(line.lstrip(" ")) + 2
            for item in lines[i + 1 :]:
                match = re.match(rf"^ {{{indent}}}(\w+): (.+)$", item)
                if not match:
                    break
                found.apps_group[match.group(1)] = match.group(2)
    return found


# ─────────────────────────────────────────────────────────────────────────────
# The roster rewrite
# ─────────────────────────────────────────────────────────────────────────────


def _content_hash(data: Any) -> str:
    """Hash of a layer's own keys, without the ones that only say where it came from."""
    if isinstance(data, dict):
        data = {k: v for k, v in data.items() if k not in _IDENTITY_KEYS}
    canonical = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _layer_source(text: str, data: Any) -> str | None:
    """The preset a materialized layer file names as its source, if any."""
    if isinstance(data, dict):
        provenance = data.get("provenance")
        preset = provenance.get("preset") if isinstance(provenance, dict) else None
        if isinstance(preset, str):
            return preset
    match = _MADE_FROM.search(text)
    return match.group(1) if match else None


def _locate(
    mapping: Any, target: tuple[str, ...], prefix: tuple[str, ...] = ()
) -> Iterator[tuple[Any, str]]:
    """Every ``(container, key)`` under *mapping* whose dotted path from it is *target*."""
    if not isinstance(mapping, dict):
        return
    for key in list(mapping.keys()):
        if not isinstance(key, str):
            continue
        path = prefix + tuple(key.split("."))
        if path == target:
            yield mapping, key
        elif target[: len(path)] == path:
            yield from _locate(mapping[key], target, path)


def _rename_key(container: Any, key: str, new_leaf: str) -> None:
    """Rename *key* to end in *new_leaf*, keeping its position and its comments."""
    renamed = key.rsplit(".", 1)[0] + "." + new_leaf if "." in key else new_leaf
    position = list(container.keys()).index(key)
    comment = container.ca.items.pop(key, None)
    value = container.pop(key)
    container.insert(position, renamed, value)
    if comment is not None:
        # An end-of-line comment is stored at an absolute column; keep the gap
        # the author left after the value rather than the column it sat at.
        token = comment[2] if len(comment) > 2 else None
        if isinstance(token, CommentToken) and not token.value.startswith("\n"):
            comment[2] = CommentToken(
                token.value, CommentMark(max(0, token.column + len(renamed) - len(key)))
            )
        container.ca.items[renamed] = comment


def _predecessor(container: Any, key: Any) -> tuple[dict, Any, int] | None:
    keys = (
        list(range(len(container)))
        if isinstance(container, CommentedSeq)
        else list(container.keys())
    )
    position = keys.index(key)
    return _last_slot(container, keys[position - 1]) if position else None


def _key_column(container: Any, key: Any, default: int) -> int:
    try:
        return int(container.lc.key(key)[1])
    except (AttributeError, KeyError, TypeError, IndexError):
        return default


def _item_column(seq: Any, default: int) -> int:
    """The column of the ``-`` of *seq*'s items."""
    try:
        line_col = seq.lc.item(0)
    except (AttributeError, KeyError, TypeError, IndexError):
        return default
    return int(line_col[1]) - 2


def _rewrite_roster_keys(document: Any, source: str) -> dict[str, dict[str, Any]]:
    """Rename the retired roster keys in place; drop the catalog.

    Returns:
        Every field the dropped catalog carried, as ``{name: {knob: value}}``.

    Raises:
        BuildProfileError: When one key is spelled both ways.
    """
    config = document.get("config") if isinstance(document, dict) else None
    if not isinstance(config, dict):
        return {}
    found = _RosterSpellings()
    _scan_roster(config, (), found)
    _refuse_mixed_roster(found, source)

    for container, key in found.default_old:
        _rename_key(container, key, "default_tier")
    for container, key in found.users_old + found.roles_old:
        _rename_key(container, key, "tier")

    catalog: dict[str, dict[str, Any]] = {}
    for container, key, path, value in found.catalog:
        for name, knob, knob_value in _catalog_knobs(path, value):
            catalog.setdefault(name, {})[knob] = knob_value
        column = _key_column(container, key, 0)
        _drop_node(container, key, _predecessor(container, key), column)
    return catalog


def _tier_pins(config: Any) -> list[tuple[str, Any, Any]]:
    """``(label, container, key)`` of every place *config* names a tier, new spelling.

    ``default_tier``, ``users[].tier`` and ``authorization.roles.<r>.tier``,
    whichever way each is spelled (dotted, nested, mixed).
    """
    pins: list[tuple[str, Any, Any]] = []
    depth = len(_WEB_TERMINALS)

    def walk(mapping: Any, prefix: tuple[str, ...]) -> None:
        for key, value in mapping.items():
            if not isinstance(key, str):
                continue
            path = prefix + tuple(key.split("."))
            if path[:depth] != _WEB_TERMINALS:
                if _WEB_TERMINALS[: len(path)] == path and isinstance(value, dict):
                    walk(value, path)
                continue
            rel = path[depth:]
            if not rel:
                if isinstance(value, dict):
                    walk(value, path)
            elif rel == ("default_tier",):
                pins.append((f"{_WT_LABEL}.default_tier", mapping, key))
            elif rel == ("users",) and isinstance(value, list):
                for entry in value:
                    if isinstance(entry, dict) and "tier" in entry:
                        label = f"{_WT_LABEL}.users[{entry.get('name')}].tier"
                        pins.append((label, entry, "tier"))
            elif rel[0] == "authorization" and (len(rel) < 2 or rel[1] == "roles"):
                if len(rel) == 4 and rel[3] == "tier":
                    pins.append((f"{_WT_LABEL}.authorization.roles.{rel[2]}.tier", mapping, key))
                elif len(rel) < 4 and isinstance(value, dict):
                    walk(value, path)

    if isinstance(config, dict):
        walk(config, ())
    return pins


def _roster_container(config: Any) -> tuple[Any, str] | None:
    """``(container, key)`` of the roster's ``users`` list."""
    for container, key in _locate(config, ("modules", "web_terminals", "users")):
        if isinstance(container[key], list):
            return container, key
    return None


def _move_users_to_shared(
    config: Any, apps: dict[str, tuple[str, str]], preset: _PresetRoster
) -> list[str]:
    """Take every user pinned to a migrated app off the roster; list the app in ``shared:``.

    Args:
        config: The profile's ``config:`` mapping, round-trip loaded.
        apps: ``{tier name users pin: (app name, bundled app it was made from)}``.
        preset: The bundled deployment preset's comments.

    Returns:
        The names of the users removed.
    """
    located = _roster_container(config)
    if located is None:
        return []
    container, users_key = located
    users = container[users_key]
    column = _item_column(users, _key_column(container, users_key, 0) + 2)
    removed: list[str] = []
    order: list[str] = []
    bundled: dict[str, str] = dict(apps.values())
    for index in reversed(range(len(users))):
        entry = users[index]
        tier = entry.get("tier") if isinstance(entry, dict) else None
        if tier not in apps:
            continue
        removed.append(str(entry.get("name")))
        order.append(apps[tier][0])
        pred = _last_slot(users, index - 1) if index else None
        _drop_node(users, index, pred, column)
    removed.reverse()
    order = list(dict.fromkeys(reversed(order)))
    if not order:
        return removed

    shared_key = users_key.rsplit(".", 1)[0] + ".shared" if "." in users_key else "shared"
    key_column = _key_column(container, users_key, 0)
    if shared_key not in container:
        container.insert(list(container.keys()).index(users_key) + 1, shared_key, CommentedSeq())
        seq = container[shared_key]
        seq.extend(order)
        _place_after(
            _last_slot(container, users_key),
            container,
            shared_key,
            key_column,
            preset.shared_lead,
        )
    else:
        seq = container[shared_key]
        fresh = [name for name in order if name not in seq]
        for offset, name in enumerate(fresh):
            seq.insert(offset, name)
        order = fresh
    last = len(seq) - 1
    for index, name in enumerate(order):
        comment = preset.shared_items.get(bundled.get(name, name))
        if comment is None:
            continue
        text, column_at = comment
        slot = (seq.ca.items, index, 0)
        rest = _split_eol(_get_tail(slot)[0])[1] if index == last else ""
        _set_tail(slot, text + "\n" + rest, column_at)
    return removed


def _ensure_apps_group(config: Any, preset: _PresetRoster) -> bool:
    """Give an explicit ``landing.groups`` the apps section ``shared:`` cards show in."""
    for container, key in _locate(config, ("modules", "web_terminals", "landing", "groups")):
        groups = container[key]
        if not isinstance(groups, list) or not groups:
            continue
        if any(isinstance(g, dict) and g.get("type") == "apps" for g in groups):
            return False
        group = CommentedMap()
        for name, value in (preset.apps_group or {"type": "apps"}).items():
            group[name] = value
        column = _item_column(groups, 0)
        pred = _last_slot(groups, len(groups) - 1)
        groups.append(group)
        _place_after(pred, groups, len(groups) - 1, column, preset.apps_group_lead)
        return True
    return False


def _floor_met(spec: str) -> bool:
    """True when *spec* already admits no release older than the schema floor."""
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import InvalidVersion, Version

    try:
        floor = Version(_PROFILE_SCHEMA_MIN_OSPREY)
        lows = [
            Version(s.version)
            for s in SpecifierSet(spec)
            if s.operator in (">=", ">", "==", "~=", "===")
        ]
    except (InvalidSpecifier, InvalidVersion):
        return False
    return any(low >= floor for low in lows)


def _restamp_floor(document: Any) -> None:
    """Raise ``requires_osprey_version`` to the schema floor; never lower it."""
    key = "requires_osprey_version"
    stamp = f">={_PROFILE_SCHEMA_MIN_OSPREY}"
    old = document.get(key)
    if isinstance(old, str):
        if not _floor_met(old):
            document[key] = type(old)(stamp)
    else:
        from osprey.utils.config_writer import anchored_put

        anchored_put(document, key, stamp)


# ─────────────────────────────────────────────────────────────────────────────
# Files
# ─────────────────────────────────────────────────────────────────────────────


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )


def _tracked(root: Path, rel: str) -> bool:
    result = _git(root, "ls-files", "--error-unmatch", rel)
    return result.returncode == 0


def _refuse_dir_clash(root: Path) -> None:
    """Refuse when a retired layer directory and its successor hold a file of the same name."""
    for retired, successor in RETIRED_LAYER_DIRS.items():
        source, target = root / retired, root / successor
        if not (source.is_dir() and target.is_dir()):
            continue
        clashes = sorted(p.name for p in source.iterdir() if (target / p.name).exists())
        if clashes:
            raise BuildProfileError(
                f"{retired}/ and {successor}/ both hold "
                f"{', '.join(clashes)}; keep one of each and run `osprey init --revalidate` again."
            )


def _git_move(root: Path, old: str, new: str) -> None:
    """Move *old* to *new* (repo-relative), by ``git mv`` when git tracks it."""
    if not (_tracked(root, old) and _git(root, "mv", old, new).returncode == 0):
        (root / old).rename(root / new)


def _move_retired_layer_dir(root: Path) -> list[str]:
    """Each retired layer directory → its successor, by ``git mv`` when git tracks the files."""
    _refuse_dir_clash(root)
    moved: list[str] = []
    for retired, successor in RETIRED_LAYER_DIRS.items():
        source = root / retired
        if not source.is_dir():
            continue
        target = root / successor
        files = sorted(p for p in source.iterdir())
        tracked = _tracked(root, retired)
        if not target.exists():
            if not (tracked and _git(root, "mv", retired, successor).returncode == 0):
                source.rename(target)
        else:
            for path in files:
                rel = f"{retired}/{path.name}"
                if not (
                    tracked and _git(root, "mv", rel, f"{successor}/{path.name}").returncode == 0
                ):
                    path.rename(target / path.name)
            if source.exists() and not any(source.iterdir()):
                source.rmdir()
        moved += [p.name for p in files]
    return moved


def _snapshot(root: Path) -> dict[str, str]:
    """The text of every file this rewrite may touch, by repo-relative path."""
    texts: dict[str, str] = {}
    candidates = [root / PROFILE_FILENAME, root / PROVIDERS_FILENAME]
    for dirname in (*LAYER_DIRNAMES, "profiles", CONTEXT_DIRNAME):
        base = root / dirname
        if base.is_dir():
            candidates.extend(p for p in base.rglob("*") if p.is_file())
    for path in candidates:
        if path.is_file():
            try:
                texts[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
    return texts


def _diff(
    before: dict[str, str], after: dict[str, str], renamed: dict[str, str]
) -> tuple[str, list[str]]:
    chunks: list[str] = []
    changed: list[str] = []
    origin = {new: old for old, new in renamed.items()}
    seen: set[str] = set()
    for rel in sorted(set(before) | set(after)):
        if rel in seen:
            continue
        if rel in after:
            old_rel = origin.get(rel, rel)
            seen.add(old_rel)
            old_text = before.get(old_rel, "")
            from_name = f"a/{old_rel}" if old_rel in before else "/dev/null"
        else:
            if rel in renamed and renamed[rel] in after:
                continue
            old_rel, old_text, from_name = rel, before[rel], f"a/{rel}"
        new_text = after.get(rel, "")
        to_name = f"b/{rel}" if rel in after else "/dev/null"
        if old_text == new_text and old_rel == rel:
            continue
        changed.append(rel)
        chunks.extend(
            difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=from_name,
                tofile=to_name,
            )
        )
        if old_text == new_text:
            chunks.append(f"rename {from_name[2:]} => {rel}\n")
    return "".join(c if c.endswith("\n") else c + "\n" for c in chunks), changed


def _load_text(text: str, rel: str) -> Any:
    """Round-trip load *text*; a parse error names the file."""
    try:
        document = _ROUND_TRIP.load(text)
    except YAMLError as e:
        raise BuildProfileError(f"{rel} is not valid YAML, so nothing was rewritten: {e}") from e
    return document if document is not None else CommentedMap()


def _safe_load(text: str, rel: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise BuildProfileError(f"{rel} is not valid YAML, so nothing was rewritten: {e}") from e


def _dump_text(document: Any) -> str:
    """*document* as the round-trip writer every profile edit uses would save it."""
    buffer = io.StringIO()
    _ROUND_TRIP.dump(document, buffer)
    return buffer.getvalue()


def _text_hash(text: str) -> str:
    """Hash of a layer file's text below its ``name:`` line, comments included.

    The header above ``name:`` carries the deployment's name, the emitting
    release and the preset hash; everything below is the preset's own text.
    """
    lines = text.splitlines(keepends=True)
    start = next((i + 1 for i, line in enumerate(lines) if line.startswith("name:")), 0)
    return "sha256:" + hashlib.sha256("".join(lines[start:]).encode("utf-8")).hexdigest()


def _put_knobs(text: str, rel: str, values: dict[str, Any]) -> str:
    """*text* with *values* set at the top of the layer file, right under its ``name:``."""
    document = _load_text(text, rel)
    if not isinstance(document, CommentedMap):
        raise BuildProfileError(f"{rel} is not a mapping")
    keys = list(document.keys())
    position = keys.index("name") + 1 if "name" in keys else 0
    pred = _last_slot(document, "name") if position else None
    fresh = [knob for knob in values if knob not in document]
    for knob, value in values.items():
        if knob in document:
            document[knob] = value
    for offset, knob in enumerate(fresh):
        document.insert(position + offset, knob, values[knob])
    if pred is not None and fresh:
        # What followed `name:` introduced the key after it; it still does.
        _place_after(pred, document, fresh[-1], 0, [])
    return _dump_text(document)


def _context_slot_empty(slot: Path) -> bool:
    return slot.is_dir() and all(p.name == ".gitkeep" for p in slot.iterdir())


# ─────────────────────────────────────────────────────────────────────────────
# ── Lacking config leaves from the preset ──
#
# A profile emitted before the preset carried the whole configuration spells
# only what its facility changed. The rewrite writes every ``config:`` leaf the
# preset documents and the profile does not
# (:func:`~osprey.cli.build_profile_drift.lacking_config_keys`), each under the
# preset's own comment, so the filled file reads like one ``osprey init``
# emitted. Values already in the file are never touched: the fill only ADDS.
# Two kinds of leaf are never written, because their absence is correct: one
# another block of the profile already owns (:func:`_claimed_elsewhere`), and
# the deployment's own selections (:func:`_deployment_data_keys`).
# ─────────────────────────────────────────────────────────────────────────────

#: How the emitted profile opens a section of its ``config:`` block. Recognized
#: rather than re-authored: a copied block that starts with one is a section
#: heading, and a heading wants the blank line above it the emitter gives it.
_SECTION_DIVIDER = "──"

#: The column a ``config:`` block's own keys sit at. Only ever needed for a
#: block the fill CREATES: a mapping built in memory carries no position, and
#: without one every comment copied into it renders flush left, which is the
#: emitted file's section-header convention and reads as a lie inside a block.
_CONFIG_COLUMN = 2

#: The two header lines that name the profile's source in prose. ``--preset``
#: re-points the machine-readable stamp; these say the same thing to a person,
#: and a file whose prose and data disagree is worse than either alone.
_HEADER_PRESET = re.compile(r"(Made from the bundled `)[^`]+(` preset)")
_HEADER_HASH = re.compile(r"(#\s+preset content hash: )\S+")

#: How much deeper than its parent a nested mapping's keys sit. The emitted
#: profile is two-space indented throughout, and a mapping built in memory has
#: to be told the column its keys will render at or every comment written into
#: it lands flush left (:func:`~osprey.utils.config_writer._mapping_column`).
_INDENT_STEP = 2

#: The manual step when no preset this OSPREY ships can be read off the profile.
_NO_PRESET_STEP = (
    "no preset this OSPREY ships is recorded for this profile, so no key was "
    "filled in; name one: `osprey init --revalidate --preset NAME`"
)


def _comment_text(lines: list[str]) -> str | None:
    """Turn ruamel comment LINES into the text ``anchored_put`` re-renders.

    The two spellings are inverses: a stored line is ``  # prose`` — the
    mapping's indent, a hash, the text — and ``comment=`` takes the text alone
    and writes the indent and hash itself, at the column of the mapping it is
    putting into. So this strips exactly what the writer puts back, which is
    what lets a comment copied out of the emitted preset land at the profile's
    own indentation rather than at the preset's.

    A blank line inside the block survives as a blank line. A line that is a
    bare ``#`` becomes one too: the renderer prefixes ``# `` to every non-empty
    line, so there is no input that produces a lone hash, and a blank separator
    is the honest approximation.

    A block that OPENS with a section divider gets a blank line put back in
    front of it. The emitted profile separates its sections that way, and the
    reader ruamel stores blocks with trims exactly that blank as edge layout —
    so without this a copied section would butt against the key above it and
    the filled file would read less like the emitted one it is copying.

    Args:
        lines: Raw comment lines as ruamel stored them, hash included.

    Returns:
        The block as text, one line per line, or ``None`` when it is empty.
    """
    text: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped[1:]
            if stripped.startswith(" "):
                stripped = stripped[1:]
        text.append(stripped)
    while text and not text[-1]:
        text.pop()
    if not any(text):
        return None
    if text[0].startswith(_SECTION_DIVIDER):
        text.insert(0, "")
    return "\n".join(text)


def _trailing_spill(cm: CommentedMap) -> list[str]:
    """The comment lines *cm*'s last entry holds for whatever follows *cm*.

    ruamel parks them on the deepest last node: a map key's trailing slot, or
    — when the last value is a list — the last ITEM's
    (:func:`~osprey.utils.config_writer._deepest_trailing_slot`). The token's
    first line is that node's own end-of-line comment; the rest is the block.
    """
    from osprey.utils.config_writer import _deepest_trailing_slot

    from .build_profile_emit import _comment_block

    found = _deepest_trailing_slot(cm)
    if found is None:
        return []
    owner, last, slot = found
    entry = owner.ca.items.get(last)
    token = entry[slot] if entry and len(entry) > slot else None
    if token is None or not isinstance(getattr(token, "value", None), str):
        return []
    return _comment_block(token.value.partition("\n")[2])


def _leaf_comments(
    cm: CommentedMap, parent_entry: list[Any] | None = None, prefix: str = ""
) -> tuple[dict[str, str], list[str]]:
    """Every leaf of ``cm``, dotted, mapped to the comment block above it.

    The emitted ``config:`` block is written flat — one dotted key per rendered
    leaf — so this is usually one call to
    :func:`~osprey.cli.build_profile_emit._extract_visual`. It recurses anyway
    because a preset may spell a subtree nested (``modules.web_terminals:`` in
    the control-assistant presets is), and the leaves under such a branch are
    addressed by the same dotted keys the comparison reports as lacking.

    A leaf with no comment of its own simply has no entry. The comment above a
    nested BRANCH is not inherited by its leaves: it introduces the branch, and
    copying it onto each leaf beneath would repeat one paragraph a dozen times.

    The second return value is what makes the recursion whole. ruamel parks the
    block introducing a key on the node BEFORE it — and when that predecessor is
    a nested branch or a list, on the deepest last leaf or item inside it, a
    level the parent's own ``_extract_visual`` never looks at. Each level
    therefore hands its caller the lines its last entry holds
    (:func:`_trailing_spill`), and the caller gives them to the next sibling.
    Without it the header of the key that follows a nested branch
    (``execution.execution_method`` after ``modules.web_terminals:``) is lost.

    Args:
        cm: A ``config:`` mapping from an emitted profile.
        parent_entry: The ruamel ``ca`` entry of the key ``cm`` hangs under,
            which is where the block above its FIRST key is stored.
        prefix: Dotted path of ``cm`` itself, for the recursion.

    Returns:
        Dotted key → comment text in the spelling ``anchored_put`` takes, and
        the comment lines ``cm``'s last entry holds for whatever follows ``cm``.
    """
    from .build_profile_emit import _extract_visual

    pre, _eol = _extract_visual(cm, parent_entry)
    comments: dict[str, str] = {}
    keys = list(cm.keys())
    # Lines the previous sibling held for this one, when the parent level could
    # not see them: they were stored inside that sibling's own subtree.
    spill: list[str] = []
    for index, key in enumerate(keys):
        value = cm[key]
        block, spill = spill + pre.get(key, []), []
        if hasattr(value, "items") and len(value):
            nested, spill = _leaf_comments(value, cm.ca.items.get(key), f"{prefix}{key}.")
            comments.update(nested)
        else:
            text = _comment_text(block)
            if text is not None:
                comments[f"{prefix}{key}"] = text
            if index == len(keys) - 1:
                spill = _trailing_spill(cm)
    return comments, spill


def _leaf_values(mapping: Any, prefix: str = "") -> dict[str, Any]:
    """Every leaf of ``mapping``, dotted, mapped to its value.

    The value half of :func:`_leaf_comments`, walked the same way so the two
    agree on what a leaf is. Values come back as plain Python — a ruamel
    container carries the comments of the document it was parsed from, and
    putting one into the profile would drag the preset's layout in with it.
    """
    leaves: dict[str, Any] = {}
    for key, value in mapping.items():
        dotted = f"{prefix}{key}"
        if hasattr(value, "items") and len(value):
            leaves.update(_leaf_values(value, f"{dotted}."))
        else:
            leaves[dotted] = _plain(value)
    return leaves


def _plain(value: Any) -> Any:
    """``value`` with every ruamel container replaced by its plain equivalent."""
    if hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _deployment_data_keys() -> dict[str, bool]:
    """Config keys whose value is the deployment's own, never the preset's.

    Read off the packaged manifest rather than listed here, because that is
    where the fact is recorded and where ``scripts/check_config_keys.py`` reads
    it. Two marks say it. ``deployment-data: true`` is a selection — who is on
    the roster, which apps are shared, which tier is the default, which tier a
    role gets, how the landing page groups its cards: a preset's value there is
    an example, and writing it into a profile would invent a choice the
    facility never made. ``data-map: true`` is a catalog whose KEYS a facility
    invents — the provider table, the named panel layouts, the render catalog —
    and it implies the first.

    Both are TERMINI for the fill: neither the key nor anything below it is
    ever written.

    Returns:
        Key → whether it is a data map (whose entries, rather than whose value,
        are the deployment's own).
    """
    from .config_cmd import _load_key_manifest

    return {
        key: bool(spec.get("data-map"))
        for key, spec in _load_key_manifest().items()
        if isinstance(spec, dict) and (spec.get("deployment-data") or spec.get("data-map"))
    }


def _claimed_elsewhere(document: Any) -> dict[str, str]:
    """Config leaves whose single home is another block of this same profile.

    A key the preset documents can still be wrong to write here, because the
    profile already states the same fact somewhere the build reads first.
    ``deploy.image_source`` is the shipped case: a build propagates it into the
    rendered config's ``modules.web_terminals.image_source``, and a profile that
    spells that leaf as well gives one fact two homes, which ``osprey validate``
    refuses outright. Filling it in would break the build of every repo that
    carries deploy coordinates.

    A profile whose ``config:`` block ALREADY spells the leaf is not claimed:
    that profile has a two-homes problem the fill did not create, and silently
    treating the key as handled would hide it from the refusal that reports it.

    Returns:
        Dotted config key → the key that owns the fact instead.
    """
    from .build_profile_deploy import IMAGE_SOURCE_CONFIG_KEY, config_image_source_spelling

    deploy = document.get("deploy")
    if not hasattr(deploy, "items"):
        return {}
    if config_image_source_spelling(document.get("config")) is not None:
        return {}
    return {IMAGE_SOURCE_CONFIG_KEY: "deploy.image_source"}


def _skipped_keys(document: Any, lacking: list[str]) -> dict[str, str]:
    """The lacking keys the fill must NOT write, each with the reason.

    Two reasons, both about a key whose absence is correct rather than
    accidental: it is, or sits below, the deployment's own data
    (:func:`_deployment_data_keys`), or its fact is stated in another block of
    this profile (:func:`_claimed_elsewhere`).
    """
    if not lacking:
        return {}
    owned = _deployment_data_keys()
    claimed = _claimed_elsewhere(document)
    skipped: dict[str, str] = {}
    for key in lacking:
        if key in claimed:
            skipped[key] = f"stated by `{claimed[key]}`. The build writes it here for you."
            continue
        parent = next(
            (prefix for prefix in owned if key == prefix or key.startswith(f"{prefix}.")), None
        )
        if parent is None:
            continue
        if owned[parent] and key != parent:
            skipped[key] = f"below `{parent}`, whose entries are this deployment's own data."
        else:
            skipped[key] = (
                f"`{parent}` is this deployment's own selection; a preset never fills it in."
            )
    return skipped


def _detach_trailing_section(mapping: Any) -> str | None:
    """Take the comment block trailing *mapping*, so an append can land above it.

    ``anchored_put`` does this itself, through
    :func:`~osprey.utils.config_writer._steal_trailing_block`, which splits the
    trailing token at the first line indented less far than the token starts.
    That is the right split when the entry's own inline comment WRAPS — moving
    its continuation lines would tear one comment in half — but it detaches
    NOTHING when the token starts at column 0, which is exactly where a section
    header introducing the next top-level block sits. Left in place, every key
    the fill appends lands under that header and the block it introduces loses
    it.

    So: the careful split first, and when it declines, the whole token — but
    only once the token's first line is known to be blank, which is what says
    the entry has no inline comment of its own and there is nothing to tear.
    """
    from osprey.utils.config_writer import (
        _deepest_trailing_slot,
        _steal_section_comment,
        _steal_trailing_block,
    )

    moved = _steal_trailing_block(mapping)
    if moved is not None:
        return moved
    found = _deepest_trailing_slot(mapping)
    if found is None:
        return None
    owner, last, slot = found
    entry = owner.ca.items.get(last)
    token = entry[slot] if entry else None
    if token is None or str(token.value).partition("\n")[0].strip():
        return None
    return _steal_section_comment(mapping)


def _put_target(config: Any, dotted: str) -> tuple[Any, str]:
    """Which mapping a lacking leaf goes into, and under which key.

    A dotted key is a SPELLING, not a location, but only at the TOP LEVEL of
    ``config:``: that is where the build's applier expands the dots, so
    ``modules.web_terminals.enabled`` beside a ``modules.web_terminals:``
    mapping and ``enabled`` inside it address one rendered leaf. Writing the
    dotted spelling beside a mapping the profile already nests puts one block at
    two depths — the shape ``osprey validate`` refuses outright for
    ``control_system:`` — and leaves the next reader looking in two places for
    one setting. The nested home wins whenever the profile has one.

    INSIDE a nested mapping the same spelling is not a spelling at all: nothing
    expands it, so ``approval: {tools.channel_write: always}`` renders a literal
    key named ``tools.channel_write`` and the runtime reading
    ``approval['tools']`` never sees the value. So the descent stops at the
    deepest mapping that EXISTS, and a remainder that still carries a dot is
    the caller's signal to create the intermediate mappings
    (:func:`_put_leaf`) rather than to write the dots down as a key.

    Args:
        config: The profile's ``config:`` mapping.
        dotted: The lacking leaf's dotted key.

    Returns:
        The deepest existing mapping on the path and the rest of the dotted key.
        A remainder with dots in it means the path runs through mappings that do
        not exist yet — legal as a key only when the mapping is ``config``
        itself.
    """
    container = config
    segments = dotted.split(".")
    while len(segments) > 1:
        for length in range(len(segments) - 1, 0, -1):
            nested = container.get(".".join(segments[:length]))
            if isinstance(nested, CommentedMap) and len(nested):
                container, segments = nested, segments[length:]
                break
        else:
            break
    return container, ".".join(segments)


def _put_leaf(config: Any, dotted: str, value: Any, comment: str | None) -> None:
    """Write one lacking leaf into ``config:``, under the comment it came with.

    Two homes, decided by :func:`_put_target`:

    * The dotted key at the top level of ``config:`` — the emitted profile's own
      flat spelling, and the only place the build expands dots. A branch the
      profile does not mention AT ALL lands this way, one dotted leaf per line,
      which is what the file would have said had it been emitted today.
    * Inside the mapping the profile nests, when it nests one. If the leaf sits
      below intermediate mappings that do not exist yet, they are created here:
      writing the remainder as a dotted key inside a nested mapping would render
      a literal key with dots in its name, which the runtime never reads.

    The created branch is filled BEFORE it is put, so the single
    :func:`~osprey.utils.config_writer.anchored_put` that attaches it sees a
    complete value and can re-anchor the comment block trailing the mapping
    below it — an empty mapping cannot carry that block, and the section header
    it usually is would end up introducing the branch instead of what follows.
    """
    from osprey.utils.config_writer import _attach_section_comment, anchored_put

    container, remainder = _put_target(config, dotted)
    segments = remainder.split(".")
    branch: Any = value
    key = remainder
    if len(segments) > 1 and container is not config:
        column = _mapping_column(container)
        leaf_depth = len(segments) - 1
        for depth in range(leaf_depth, 0, -1):
            nested = CommentedMap()
            nested.lc.col = column + _INDENT_STEP * depth
            # Only the LEAF carries the preset's prose: the mappings above it
            # are the fill's own scaffolding, which the preset documents no
            # comment for.
            anchored_put(
                nested, segments[depth], branch, comment=comment if depth == leaf_depth else None
            )
            branch = nested
        key, comment = segments[0], None
    # The block trailing the target introduces whatever comes AFTER it, so it
    # has to end up below the append rather than above it.
    moved = _detach_trailing_section(container)
    anchored_put(container, key, branch, comment=comment)
    if moved:
        _attach_section_comment(container, moved)


def _mapping_column(mapping: Any) -> int:
    """The column *mapping*'s keys sit at, or the ``config:`` block's default."""
    column = getattr(getattr(mapping, "lc", None), "col", None)
    return column if isinstance(column, int) else _CONFIG_COLUMN


def _restamp_header(text: str, preset: str, preset_hash: str) -> str:
    """Re-point the emitted header's two preset lines at what the stamp now says.

    ``--preset`` moves ``provenance:``; the header says the same thing in prose,
    and a file whose prose and data disagree sends its next reader to the wrong
    preset. Only the leading comment block is touched, and only the two lines
    that name the preset and its hash — the emitting VERSION stays as it was,
    because that line is history and history did not change.
    """
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.startswith("#"):
            break
        line = _HEADER_PRESET.sub(lambda m: m.group(1) + preset + m.group(2), line)
        lines[index] = _HEADER_HASH.sub(lambda m: m.group(1) + preset_hash, line)
    return "".join(lines)


def _reference_document(preset: str, *, project_name: str, profile_name: str) -> CommentedMap:
    """The profile ``osprey init --preset <preset>`` would write, with comments.

    :func:`~osprey.cli.build_profile_emit.materialized_profile` answers the same
    question as DATA, and the comparison the fill asks its keys of uses it. The
    comments are the other half of the answer here — the fill copies each key's
    documentation along with it — and they only exist in the emitted TEXT, so the
    text is what this parses. The layering is the materializer's, so the two
    documents are the same document read two ways.

    ``project_name`` is the profile's own ``project_name:``, so the reference
    states the same value and that key never reads as one to copy in.
    """
    from ruamel.yaml import YAML

    from .build_profile_emit import materialized_profile_text

    text = materialized_profile_text(preset, project_name=project_name, profile_name=profile_name)
    document: CommentedMap = YAML().load(text)
    return document


def _resolve_preset(document: Any, preset: str | None, profile_path: Path) -> str | None:
    """Which preset this profile is filled from, or ``None`` when none is known.

    Three sources, most explicit first. ``--preset`` is the operator saying it.
    ``provenance.preset`` is what ``osprey init`` recorded, and it names the
    exact preset, so it outranks the retired ``app_template:`` key, which only
    ever named one of the packaged bundles. ``app_template:`` is the last
    resort, and the only source a profile emitted before ``provenance:``
    existed has.

    ``--preset`` does not fall through: a preset the operator named and this
    OSPREY does not ship — or a tier or app, which is no deployment — is a
    refusal, not a reason to quietly fill from something else. The two
    file-borne sources DO fall through to each other, which is how a profile
    from a release that shipped a preset this one dropped still fills from its
    bundle; when neither names a shipped preset the answer is ``None``, and the
    caller leaves the fill for a run that names one.

    Args:
        document: The raw profile document.
        preset: The ``--preset`` value, if given.
        profile_path: Named in the refusal, so an operator knows which file.

    Returns:
        A bundled preset name in the on-disk spelling, or ``None``.

    Raises:
        BuildProfileError: ``--preset`` names a tier or app, or a preset this
            installation does not ship.
    """
    from .build_profile_presets import (
        PRESET_DATA_BUNDLE_KEY,
        _normalize_preset_name,
        layer_preset_refusal,
        list_presets,
    )

    if preset:
        refusal = layer_preset_refusal(preset)
        if refusal is not None:
            raise BuildProfileError(f"{refusal}. Nothing was written to {profile_path}.")
        known = list_presets()
        if _normalize_preset_name(preset) in known:
            return _normalize_preset_name(preset)
        raise BuildProfileError(
            f"--preset names preset {preset!r}, which this OSPREY does not ship. "
            f"Nothing was written.\n\nBundled presets: {', '.join(known)}"
        )

    known = list_presets()
    provenance = document.get("provenance")
    recorded = provenance.get("preset") if hasattr(provenance, "get") else None
    for value in (recorded, document.get(PRESET_DATA_BUNDLE_KEY)):
        if value and _normalize_preset_name(str(value)) in known:
            return _normalize_preset_name(str(value))
    return None


def _refresh_provenance_comment(document: CommentedMap) -> None:
    """Replace the comment OSPREY emitted above ``provenance:`` with today's.

    The block is OSPREY's own text, not the operator's — recognized by its first
    line — so a profile emitted by an earlier release reads like one emitted by
    this one once it is brought up to date. A block the operator wrote is kept.
    ruamel stores the block on the trailing slot of the node before
    ``provenance:``.
    """
    from osprey.utils.config_writer import _deepest_trailing_slot

    from .build_profile_emit import _PROVENANCE_COMMENT

    keys = list(document.keys())
    index = keys.index("provenance")
    if index == 0:
        return
    previous = keys[index - 1]
    value = document[previous]
    found = (
        _deepest_trailing_slot(value)
        if isinstance(value, (CommentedMap, CommentedSeq)) and len(value)
        else None
    ) or (document, previous, 2)
    owner, last, slot = found
    entry = owner.ca.items.get(last)
    token = entry[slot] if entry and len(entry) > slot else None
    if token is None or not isinstance(getattr(token, "value", None), str):
        return
    lines = token.value.split("\n")
    end = len(lines)
    while end > 1 and not lines[end - 1].strip():
        end -= 1
    start = end
    while start > 1 and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    if start == end or lines[start].strip() != _PROVENANCE_COMMENT[0]:
        return
    indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    lines[start:end] = [indent + line for line in _PROVENANCE_COMMENT]
    token.value = "\n".join(lines)


def _stamp_provenance(document: Any, preset: str, preset_hash: str, providers_hash: str) -> str:
    """Record what the profile now carries: the preset, its hash, the catalog's.

    Written after the fill rather than before, because that is what makes it
    true: the profile spells every key the preset documents, so the stamp the
    emitter would have written is the honest one. A ``deviation_marker:`` the
    operator set is left alone — it is the one key in the block that is
    theirs. The comment OSPREY emitted above the block is brought up to date
    (:func:`_refresh_provenance_comment`).

    Returns:
        The deviation marker now in force, which is what a drift finding has to
        be claimed with.
    """
    from osprey.utils.config_writer import anchored_put

    from .build_profile_emit import _PROVENANCE_COMMENT
    from .build_profile_schema import DEFAULT_DEVIATION_MARKER

    stamps = {
        "preset": preset,
        "preset_hash": preset_hash,
        "providers_hash": providers_hash,
    }
    block = document.get("provenance")
    if not isinstance(block, CommentedMap):
        block = CommentedMap()
        for key, value in stamps.items():
            block[key] = value
        anchored_put(
            document, "provenance", block, comment=_comment_text(list(_PROVENANCE_COMMENT))
        )
        return DEFAULT_DEVIATION_MARKER
    _refresh_provenance_comment(document)
    for key, value in stamps.items():
        block[key] = value
    marker = block.get("deviation_marker")
    return marker if isinstance(marker, str) and marker else DEFAULT_DEVIATION_MARKER


class _Fill(NamedTuple):
    """What the fill wrote into ``config:``, and what it left out on purpose."""

    filled: list[str]
    skipped: dict[str, str]


def _fill_lacking_keys(document: CommentedMap, repo_root: Path, preset: str) -> _Fill:
    """Write every ``config:`` leaf *preset* documents and *document* lacks.

    Planned in memory like the rest of the rewrite: an unbacked key — one the
    comparison reports missing that the preset's own emitted profile does not
    carry — refuses the run before anything is written, because the
    alternative is writing ``key: null`` into the facility's source of truth
    and calling it the preset's value.

    Raises:
        BuildProfileError: The preset cannot be read, or reports a key its own
            emitted profile does not carry.
    """
    from osprey.utils.config_writer import anchored_put

    from .build_profile_drift import lacking_config_keys

    profile_path = repo_root / PROFILE_FILENAME
    profile_name = str(document.get("name") or repo_root.name)
    try:
        reference = _reference_document(
            preset,
            project_name=str(document.get("project_name") or ""),
            profile_name=profile_name,
        )
        lacking = lacking_config_keys(document, preset)
    except BuildProfileError as e:
        raise BuildProfileError(f"Cannot read preset {preset!r} to fill from: {e}") from e

    reference_config = reference.get("config")
    comments = (
        _leaf_comments(reference_config, reference.ca.items.get("config"))[0]
        if hasattr(reference_config, "items")
        else {}
    )
    values = _leaf_values(reference_config) if hasattr(reference_config, "items") else {}

    # Keys whose absence is correct rather than accidental. Removed before the
    # backing check below, because a key that is never written needs no value.
    skipped = _skipped_keys(document, lacking)
    lacking = [key for key in lacking if key not in skipped]

    unbacked = [key for key in lacking if key not in values]
    if unbacked:
        raise BuildProfileError(
            f"Preset {preset!r} reports {len(unbacked)} key(s) as missing that its own "
            f"emitted profile does not carry: {', '.join(unbacked[:5])}. Nothing was "
            f"written. This is a framework inconsistency, not a problem with "
            f"{profile_path}."
        )

    if lacking:
        config = document.get("config")
        if not isinstance(config, CommentedMap):
            config = CommentedMap()
            # A mapping built here carries no position, and a comment written
            # into one renders at column 0 — the emitted file's section-header
            # convention, inside a block where it means nothing.
            config.lc.col = _CONFIG_COLUMN
            anchored_put(document, "config", config)
        for key in lacking:
            _put_leaf(config, key, values.get(key), comments.get(key))
    return _Fill(lacking, skipped)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _Layer:
    """One tier file the rewrite plans for."""

    origin: str
    """Repo-relative path before the rewrite."""
    rel: str
    """Repo-relative path after it."""
    text: str


def _drop_helper_selections(document: Any) -> list[str]:
    """Remove every helper file a ``hooks:`` list names, in place.

    A helper ships as the closure of the hooks that import it, so the build
    refuses one named in the list; the previous release's presets named them.
    Each entry's own line comment goes with it.

    Returns:
        The helper names removed, in list order.
    """
    from osprey.agent_runner.build_artifacts.catalog import BuildArtifactCatalog

    hooks = document.get("hooks") if isinstance(document, dict) else None
    if not isinstance(hooks, CommentedSeq):
        return []
    helpers = BuildArtifactCatalog.default().helpers()
    removed: list[str] = []
    for index in reversed(range(len(hooks))):
        entry = hooks[index]
        if isinstance(entry, str) and f"hooks/{entry}" in helpers:
            removed.insert(0, entry)
            del hooks[index]
    return removed


def _catalog_stem(build_profile: Any) -> str | None:
    """``<s>`` when a catalog ``build_profile`` names ``<d>/<s>.yml`` for a directory ``<d>`` read as tiers."""
    if not isinstance(build_profile, str):
        return None
    dirs = "|".join(re.escape(d) for d in _TIER_DIRS_RETIRED_FIRST)
    match = re.fullmatch(rf"(?:\./)?(?:{dirs})/([^/]+)\.yml", build_profile.strip())
    return match.group(1) if match else None


def revalidate_repo(root: Path, *, preset: str | None = None) -> RevalidateResult:
    """Bring the deployment repo at *root* to this OSPREY, in place.

    Every file is read, parsed and rewritten in memory first; the repo is
    touched only once the whole rewrite is planned, so a file that cannot be
    read leaves the repo exactly as it was.

    Args:
        root: The repo root; holds ``profile.yml``.
        preset: The preset to fill from; default: the one the profile records.

    Returns:
        The diff of every changed file, what the fill wrote and left out, and
        the steps left to do by hand.

    Raises:
        BuildProfileError: When the repo cannot be rewritten as it stands (a
            file that is not valid YAML, one key spelled both ways,
            the retired layer directory and ``tiers/`` holding the same file,
            a ``preset`` this OSPREY does not ship, a ``providers.yml`` that
            does not parse).
    """
    from .build_profile_emit import emit_layer_delta_yaml

    root = Path(root)
    before = _snapshot(root)
    result = RevalidateResult()
    steps = result.manual_steps
    writes: dict[str, str] = {}
    deletes: set[str] = set()
    sweeps: list[str] = []

    # ── The profile and its overlays, renamed in memory ─────────────────────
    profile_text = (root / PROFILE_FILENAME).read_text(encoding="utf-8")
    document = _load_text(profile_text, PROFILE_FILENAME)
    if not isinstance(document, CommentedMap):
        raise BuildProfileError(f"{root / PROFILE_FILENAME} is not a mapping")
    catalog = _rewrite_roster_keys(document, PROFILE_FILENAME)
    overlays: dict[str, Any] = {}
    overlay_catalogs: dict[str, dict[str, dict[str, Any]]] = {}
    for variant in known_variants(root):
        overlay_path = variant_profile_path(root, variant)
        rel = overlay_path.relative_to(root).as_posix()
        overlay = _load_text(overlay_path.read_text(encoding="utf-8"), rel)
        if not isinstance(overlay, CommentedMap):
            continue
        overlay_catalogs[rel] = _rewrite_roster_keys(overlay, rel)
        overlays[rel] = overlay
    documents = {PROFILE_FILENAME: document, **overlays}
    for doc in documents.values():
        _drop_helper_selections(doc)

    # ── Which file each catalog entry built ─────────────────────────────────
    _refuse_dir_clash(root)
    on_disk = {
        p.stem
        for d in _TIER_DIRS_RETIRED_FIRST
        if (root / d).is_dir()
        for p in (root / d).glob("*.yml")
    }
    pinned = {
        container[key]
        for doc in documents.values()
        for _label, container, key in _tier_pins(doc.get("config"))
        if isinstance(container[key], str)
    }
    built_from: dict[str, list[str]] = {}
    for name, fields in catalog.items():
        stem = _catalog_stem(fields.get("build_profile"))
        if "build_profile" in fields and stem != name:
            built_from.setdefault(stem or "", []).append(name)
    file_renames: dict[str, str] = {}
    for stem, names in built_from.items():
        name = names[0]
        if len(names) == 1 and stem in on_disk and name not in on_disk and stem not in pinned:
            file_renames[stem] = name
            continue
        for name in names:
            steps.append(
                f"the `{RETIRED_CATALOG_KEY}:` catalog built {name!r} from "
                f"{catalog[name]['build_profile']}, and a tier is now built from "
                f"{TIERS_DIRNAME}/{name}.yml; rename that file to {TIERS_DIRNAME}/{name}.yml "
                f"or pin its users to the file's own name"
            )
    for rel, overlay_catalog in overlay_catalogs.items():
        for name, fields in overlay_catalog.items():
            stem = _catalog_stem(fields.get("build_profile"))
            if "build_profile" in fields and stem != name and file_renames.get(stem or "") != name:
                steps.append(
                    f"{rel} built {name!r} from {fields['build_profile']} in its `{RETIRED_CATALOG_KEY}:` "
                    f"catalog, which is gone; a tier is built from {TIERS_DIRNAME}/{name}.yml"
                )

    # ── Tier files: keep, re-emit, or turn into an app ──────────────────────
    layers: dict[str, _Layer] = {}
    for dirname in _TIER_DIRS_RETIRED_FIRST:
        base = root / dirname
        for path in sorted(base.glob("*.yml")) if base.is_dir() else ():
            stem = file_renames.get(path.stem, path.stem)
            layers[stem] = _Layer(
                f"{dirname}/{path.name}",
                f"{TIERS_DIRNAME}/{stem}.yml",
                path.read_text(encoding="utf-8"),
            )
    apps_dir = root / APPS_DIRNAME
    existing_apps = {p.stem: p for p in sorted(apps_dir.glob("*.yml"))} if apps_dir.is_dir() else {}

    profile_name = str(document.get("name") or root.name)
    apps: dict[str, tuple[str, str]] = {}
    for stem, layer in layers.items():
        data = _safe_load(layer.text, layer.origin)
        source = _layer_source(layer.text, data)
        retired = RETIRED_PRESETS.get(source) if source else None
        if retired is None:
            continue
        rel = layer.rel
        successor = retired.successor
        app_rel = f"{APPS_DIRNAME}/{successor}.yml"
        edited = (
            _content_hash(data) != retired.content_hash
            or _text_hash(layer.text) != retired.text_hash
        )
        if retired.kind == APPS_DIRNAME and not edited and successor in existing_apps:
            steps.append(
                f"{rel} was made from `{source}` and never edited, but {app_rel} already "
                f"exists, so it stays a tier. To run it as the `{successor}` app: list "
                f"`{successor}` under `shared:` and remove {rel} and the users pinned to it"
            )
            continue
        if edited:
            if retired.kind == APPS_DIRNAME:
                steps.append(
                    f"{rel} was made from `{source}` and has been edited, so it stays a tier. "
                    f"To run it as the `{successor}` app: run `osprey scaffold apps`, carry "
                    f"your edits into {app_rel}, list `{successor}` under `shared:`, and "
                    f"remove {rel} and the users pinned to it"
                )
            else:
                steps.append(
                    f"{rel} was made from `{source}` and has been edited, so it is kept as "
                    f"it is; it has no provenance block, so `osprey validate` cannot report "
                    f"its drift from the bundled `{successor}` tier"
                )
            continue
        if retired.kind == APPS_DIRNAME:
            deletes.add(rel)
            if app_rel not in writes:
                writes[app_rel] = emit_layer_delta_yaml(
                    f"{APPS_DIRNAME}/{successor}", profile_name, app_rel
                )
            apps[stem] = (successor, successor)
        else:
            emitted = emit_layer_delta_yaml(f"{TIERS_DIRNAME}/{successor}", profile_name, rel)
            if stem != successor:
                # Titled after the file, as the previous release titled it.
                emitted = emitted.replace(
                    f"{profile_name} ({successor})", f"{profile_name} ({stem})"
                )
            writes[rel] = emitted

    # An app made by an earlier run that stopped before its users moved: the
    # users still pinned to it move now.
    tiers_left = {stem for stem, layer in layers.items() if layer.rel not in deletes}
    for stem, path in existing_apps.items():
        if stem in apps or stem in tiers_left:
            continue
        rel = f"{APPS_DIRNAME}/{path.name}"
        text = path.read_text(encoding="utf-8")
        source = _layer_source(text, _safe_load(text, rel))
        if source and source not in RETIRED_PRESETS:
            # Materialized from a bundled app (``logbook`` or ``apps/logbook``).
            apps[stem] = (stem, source.rsplit("/", 1)[-1])

    # ── The roster ──────────────────────────────────────────────────────────
    config = document.get("config")
    pinned_apps = {
        container[key]
        for _label, container, key in _tier_pins(config)
        if isinstance(container[key], str) and container[key] in apps
    }
    if pinned_apps and isinstance(config, dict):
        roster = _preset_roster()
        removed = _move_users_to_shared(config, apps, roster)
        if removed:
            _ensure_apps_group(config, roster)
        for user in removed:
            slot = f"{CONTEXT_DIRNAME}/{user}"
            if _context_slot_empty(root / slot):
                sweeps.append(slot)
            elif (root / slot).exists():
                steps.append(
                    f"{slot}/ held context for the {user!r} card, which is now a shared app "
                    f"with no per-user context; move what you need and delete the folder"
                )

    for rel, doc in documents.items():
        for label, container, key in _tier_pins(doc.get("config")):
            value = container[key]
            if not isinstance(value, str):
                continue
            if value in apps:
                steps.append(
                    f"{rel}: `{label}` names {value!r}, which is now the shared app "
                    f"`{apps[value][0]}` that everyone on the roster opens; name a tier from "
                    f"{TIERS_DIRNAME}/ instead"
                )
            elif value not in tiers_left:
                steps.append(
                    f"{rel}: `{label}` names {value!r}, which has no file in "
                    f"{TIERS_DIRNAME}/; add {TIERS_DIRNAME}/{value}.yml or name a tier that has one"
                )

    # ── What the dropped catalogs carried ───────────────────────────────────
    knobs: dict[str, dict[str, Any]] = {}
    for rel, cat in {PROFILE_FILENAME: catalog, **overlay_catalogs}.items():
        overlay_knobs: dict[str, dict[str, Any]] = {}
        for name, fields in cat.items():
            for knob, value in fields.items():
                if knob == "build_profile":
                    continue
                if knob in _LAYER_FILE_KNOBS:
                    target = knobs if rel == PROFILE_FILENAME else overlay_knobs
                    target.setdefault(name, {})[knob] = value
                elif knob == "landing_group":
                    if name not in apps:
                        steps.append(
                            f"{rel} put {name!r} under the landing heading {value!r} through "
                            f"its `{RETIRED_CATALOG_KEY}:` catalog, which is gone; group its cards with "
                            f"`{_WT_LABEL}.landing.groups`"
                        )
                else:
                    steps.append(
                        f"{rel} set `{knob}` for {name!r} in its `{RETIRED_CATALOG_KEY}:` catalog, which "
                        f"is gone; the key has no successor and was dropped"
                    )
        for name, values in overlay_knobs.items():
            steps.append(
                f"{rel} set {', '.join(sorted(values))} for {name!r} in its "
                f"`{RETIRED_CATALOG_KEY}:` catalog, which is gone; set them at the top of "
                f"{TIERS_DIRNAME}/{name}.yml if every variant should have them"
            )
    for name, values in knobs.items():
        if name in apps:
            rel = f"{APPS_DIRNAME}/{apps[name][0]}.yml"
            text = writes.get(rel) or (root / rel).read_text(encoding="utf-8")
        elif name in tiers_left:
            rel = layers[name].rel
            text = writes.get(rel, layers[name].text)
        else:
            steps.append(
                f"the `{RETIRED_CATALOG_KEY}:` catalog set {', '.join(sorted(values))} for {name!r}, "
                f"which has no file in {TIERS_DIRNAME}/ or {APPS_DIRNAME}/; set them at the "
                f"top of that file once it exists"
            )
            continue
        writes[rel] = _put_knobs(text, rel, values)

    # ── Lacking config leaves, providers.yml, provenance ────────────────────
    resolved = _resolve_preset(document, preset, root / PROFILE_FILENAME)
    result.preset = resolved
    if resolved is not None:
        fill = _fill_lacking_keys(document, root, resolved)
        result.filled, result.skipped = fill.filled, fill.skipped
        if PRESET_DATA_BUNDLE_KEY in document:
            _drop_key_and_pre_comment(document, PRESET_DATA_BUNDLE_KEY)
    else:
        steps.append(_NO_PRESET_STEP)
    plan = _plan_providers(root)
    if (root / PROVIDERS_FILENAME).is_file():
        writes[PROVIDERS_FILENAME] = plan.text
        result.providers_kept = plan.carried
    preset_hash = ""
    if resolved is not None:
        preset_hash = compute_preset_hash(resolved) or "(unavailable)"
        result.deviation_marker = _stamp_provenance(
            document, resolved, preset_hash, plan.content_hash
        )

    _restamp_floor(document)
    for rel, doc in documents.items():
        writes[rel] = _dump_text(doc)
    if resolved is not None:
        writes[PROFILE_FILENAME] = _restamp_header(writes[PROFILE_FILENAME], resolved, preset_hash)

    # ── Apply: nothing above touched the repo ───────────────────────────────
    _move_retired_layer_dir(root)
    for old, new in file_renames.items():
        old_rel, new_rel = f"{TIERS_DIRNAME}/{old}.yml", f"{TIERS_DIRNAME}/{new}.yml"
        if (root / old_rel).is_file():
            _git_move(root, old_rel, new_rel)
    for rel in sorted(deletes):
        (root / rel).unlink(missing_ok=True)
    for rel, text in writes.items():
        path = root / rel
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for rel in sweeps:
        shutil.rmtree(root / rel)

    renamed = {layer.origin: layer.rel for layer in layers.values() if layer.origin != layer.rel}
    result.diff, result.changed = _diff(before, _snapshot(root), renamed)
    if resolved is not None:
        result.drift_refusals = _drift_refusals(
            root, resolved, preset_hash, plan.content_hash, result.deviation_marker
        )
    return result


def _plan_providers(root: Path) -> Any:
    """The ``providers.yml`` this repo is brought to; refuses before anything is written."""
    import click

    from .profile_cmd import _plan_provider_catalog

    try:
        return _plan_provider_catalog(root)
    except click.UsageError as e:
        raise BuildProfileError(e.message) from e


def _drift_refusals(
    root: Path, preset: str, preset_hash: str, providers_hash: str, marker: str
) -> int:
    """How many differences from *preset* ``osprey validate`` now refuses.

    Stamping ``provenance:`` is what turns the preset-drift check on: before it
    there is nothing to compare the profile with, and after it every structural
    difference from the preset is a refusal until a marker comment claims it.
    A profile that carried no stamp — or one ``--preset`` just re-pointed — can
    therefore validate green before the rewrite and be refused after, which the
    operator has to hear from the verb that did it rather than from the next
    build. Never fatal: a comparison that cannot be made counts nothing.
    """
    from .build_profile_drift import preset_drift_report
    from .build_profile_schema import ProfileProvenance

    try:
        card = preset_drift_report(
            root / PROFILE_FILENAME,
            ProfileProvenance(
                preset=preset,
                preset_hash=preset_hash,
                providers_hash=providers_hash,
                deviation_marker=marker,
            ),
        )
    except BuildProfileError:
        return 0
    return len(card.refusals)
