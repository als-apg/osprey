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
4. The retired write-posture keys (``control_system.writes_enabled`` and every
   ``control_system.connector.<type>.writes_enabled``) become the tier key
   ``control_system.writes`` by the rule the build applies at parse, in the
   profile, every overlay and every tier or app file kept. A comment stanza the
   previous release shipped about them is replaced by the bundled preset's
   current one, a commented retired-key example is dropped, and any other
   prose about them, or a connector block the rewrite empties, is printed as a
   step left to do by hand.
5. Every ``config:`` leaf the profile leaves to its preset is filled in from
   that preset, under the preset's own comment; the retired ``app_template:``
   key is dropped. The deployment's own selections (the keys the manifest
   marks ``deployment-data`` or ``data-map``) are never filled: the output
   names each one left out, and so is ``control_system.writes``, so a root
   that never stated it stays ``never``.
6. ``providers.yml`` is completed, never rewritten: every value it states is
   kept, an edited entry OSPREY ships included, and only the entries OSPREY
   ships that it lacks, and the keys an existing one lacks, are added.
   ``osprey validate`` reports each remaining difference as drift.
7. ``provenance:`` is restamped with the preset, its hash and the catalog's
   hash, and ``requires_osprey_version`` at the schema floor, so a release
   that predates the vocabulary refuses the rewritten repo.
7. The repo's own skills, agents, rules, output styles and hooks are named by
   path in ``profile.yml`` and each app: each now renders only where its path
   is named, and before every render carried all of them. A bare entry the
   repo ships a file for (``safety`` beside ``rules/safety.md``) becomes the
   path in place; every other unnamed one is appended. Tiers inherit
   ``profile.yml``'s lists.

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
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

import yaml
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import CommentMark, YAMLError
from ruamel.yaml.tokens import CommentToken

from osprey.profiles.providers import PROVIDERS_FILENAME
from osprey.utils.config_writer import _yaml as _ROUND_TRIP
from osprey.utils.workspace import (
    APPS_DIRNAME,
    LAYER_DIRNAMES,
    PROFILE_FILENAME,
    RETIRED_LAYER_DIRS,
    TIERS_DIRNAME,
    layer_dir_kind,
)
from osprey_connectors.errors import BuildProfileError

from .build_profile_document import (
    _WEB_TERMINALS,
    _WT_LABEL,
    RENDER_KEYS,
    RETIRED_CATALOG_KEY,
    RETIRED_WRITES_LEAF,
    WRITES_KEY_LEAF,
    _catalog_knobs,
    _kept_flat_leaf,
    _mixed_writes_error,
    _refuse_mixed_roster,
    _RosterSpellings,
    _scan_roster,
    _scan_writes,
    _tier_key_values,
    _WriteSpellings,
    governed_policy_spellings,
    governed_policy_tools,
    retired_approval_policy_message,
    retired_write_answer,
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


#: :attr:`RetiredPreset.kind` of a card that is now a shared login.
_SHARED = "shared"


class RetiredPreset(NamedTuple):
    """One retired per-card preset, as the previous release emitted it."""

    kind: str
    """``tiers`` (a person's tier) or ``shared`` (a shared login): what the card is now."""
    successor: str
    """The bundled login, in ``tiers/``, that replaces it."""
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
        _SHARED,
        "logbook",
        "sha256:87a0465d2273266ff1cd5a1087ed49397e33fb3ecc4408e78ad813cedcbc5bf6",
        "sha256:f5afe72dceb1e5b17ef978dae78173b02a97699264e3974c4cf099de9cced910",
    ),
    "control-assistant-knowledge": RetiredPreset(
        _SHARED,
        "knowledge",
        "sha256:1fd8929fce26154def1b9aabb1115ddd7a9429c06abf80fb399587de03730e75",
        "sha256:c399b55a15895532bbbb591d115d1a6dd8df9f0e3efeb7d00b15a645053bd2e3",
    ),
}


class MovedLogin(NamedTuple):
    """A login file ``osprey init`` wrote into ``apps/`` before logins moved to ``tiers/``."""

    content_hash: str
    """:func:`_content_hash` of the file's data as ``osprey init`` emitted it."""
    text_hash: str
    """:func:`_text_hash` of that file's text."""


#: The bundled logins ``osprey init`` emitted into ``apps/<name>.yml``, by name,
#: hashed as it wrote them (shipped shadows named by path). Frozen here because
#: those files leave the package: one that still hashes to them was never
#: edited, so re-emitting it from ``tiers/<name>.yml`` loses nothing.
MOVED_APP_LOGINS: dict[str, MovedLogin] = {
    "logbook": MovedLogin(
        "sha256:3112606cd043738e4024400db26d283745fed0b87cdb1d4c8f60ba6f14a2d53a",
        "sha256:4770797d63a26b848f237c83205b797d86d9d59160d5bb0cef94fab300ac9b8b",
    ),
    "knowledge": MovedLogin(
        "sha256:093a14fb72f1270df22fdf1b80d4ede6a194907c51e1d9652e1d8405e4534d72",
        "sha256:90d4e76d0450d958eb1b8bdf8547c1b3d97b688f2b675558341b2e78bce1e102",
    ),
    "channel-finder": MovedLogin(
        "sha256:9a1397d6b6f266ebfd16857229e851c9aaa6e2ad3f4d33a0e338facada104c42",
        "sha256:94e9d2ea21ca44229c30f5ca8583c23d0ba6409d8cd34faa04bd271d6dee2557",
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
    app_drift: list[str] = field(default_factory=list)
    """Each copied application whose files differ from what ``osprey init`` copied,
    unclaimed, and each whose preset has moved on since. Reported, never rewritten."""
    retired_policies: list[str] = field(default_factory=list)
    """One line per file that lost ``approval.tools.<tool>`` for a governed tool, naming them."""


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
    """The comment slot that trails ``container[key]``'s whole subtree.

    A flow collection (``access: [roster]``) is one line, so it is a leaf here:
    its own items' slots sit inside its brackets, and a comment block handed to
    one would be written there.
    """
    value = container[key]
    if _is_flow(value):
        return container.ca.items, key, 0 if isinstance(container, CommentedSeq) else 2
    if isinstance(value, CommentedMap) and len(value):
        return _last_slot(value, list(value.keys())[-1])
    if isinstance(value, CommentedSeq) and len(value):
        return _last_slot(value, len(value) - 1)
    return container.ca.items, key, 0 if isinstance(container, CommentedSeq) else 2


def _is_flow(value: Any) -> bool:
    """Whether *value* is a collection written in flow style (``[a, b]``, ``{k: v}``)."""
    return isinstance(value, (CommentedMap, CommentedSeq)) and bool(value.fa.flow_style())


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
    shared_group_lead: list[str] = field(default_factory=list)
    shared_group: dict[str, str] = field(default_factory=dict)


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


#: The landing section shared cards show in, and its retired spelling.
_SHARED_GROUP_TYPE = "shared"
_RETIRED_SHARED_GROUP_TYPE = "apps"
_SHARED_GROUP_LINES = (
    f"- type: {_SHARED_GROUP_TYPE}",
    f"- type: {_RETIRED_SHARED_GROUP_TYPE}",
)


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
        elif stripped in _SHARED_GROUP_LINES and not found.shared_group_lead:
            found.shared_group_lead = _comment_block_above(lines, i)
            found.shared_group = {"type": _SHARED_GROUP_TYPE}
            indent = len(line) - len(line.lstrip(" ")) + 2
            for item in lines[i + 1 :]:
                match = re.match(rf"^ {{{indent}}}(\w+): (.+)$", item)
                if not match:
                    break
                found.shared_group[match.group(1)] = match.group(2)
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
    # An item added in memory has no position yet.
    return default if line_col is None else int(line_col[1]) - 2


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


# ─────────────────────────────────────────────────────────────────────────────
# The write-posture rewrite
#
# The retired ``control_system.writes_enabled`` and every
# ``control_system.connector.<type>.writes_enabled`` become the one tier key,
# ``control_system.writes``, by the rule the parse point applies
# (:func:`~osprey.cli.build_profile_document.retired_write_answer`; ``never``
# in an app). The keys are rewritten in the round-trip tree; the comments that
# described them are rewritten as text afterwards, because a comment is not a
# node: a stanza the previous release shipped is replaced by the current
# preset's, a commented retired-key example is dropped, and any other prose
# about the retired keys is left for the operator.
# ─────────────────────────────────────────────────────────────────────────────


class WriteStanza(NamedTuple):
    """One comment stanza the previous release shipped about the retired write keys.

    The stanza is the ``old_lines`` lines right above the ``anchor`` line, in
    the file AFTER the keys are rewritten; it is replaced by the
    ``new_lines`` lines right above that line in the bundled ``preset`` today,
    or above ``preset_anchor`` where the preset's line has since moved on from
    the one an operator's file still holds. Lines are compared without their
    indentation, and the replacement takes the anchor's indentation in the file
    it lands in.
    """

    preset: str
    """The bundled preset (``tiers/<n>`` or a deployment preset) holding the successor."""
    anchor: str
    """The line below the stanza, stripped; the same in both releases."""
    old_lines: int
    old_hash: str
    """:func:`_stanza_hash` of the previous release's stanza."""
    new_lines: int
    new_hash: str
    """:func:`_stanza_hash` of the preset's stanza today, which locates it there."""
    preset_anchor: str | None = None
    """The line below the stanza in the bundled preset, stripped, when it is not ``anchor``."""


#: The stanzas the previous release shipped, by frozen hash: the old text
#: leaves the package with this release, so only its hash is kept. A stanza
#: that still hashes to one of these was never edited, so replacing it with
#: the preset's current words loses nothing the operator wrote.
WRITE_STANZAS: tuple[WriteStanza, ...] = (
    WriteStanza(
        "control-assistant",
        "control_system.writes: allowed",
        7,
        "sha256:ac361105ab01d4cd2a662d5f42d71ee2fa27841d59519084cb1313f05c12e940",
        6,
        "sha256:cc7a570eed0b6ebf581c8f97ed5ff6ac4526148138141ce38a46aa8082fee24e",
        "control_system.writes: asked",
    ),
    WriteStanza(
        "control-assistant",
        "# Where the simulator runs: `served` is its container over Channel Access; `in_process` is inside this process, with no container.",
        4,
        "sha256:af5020608bfe8231dea1508462dd9880cbb16c3e7cc680e01c565692e127d15a",
        0,
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    WriteStanza(
        "control-assistant",
        "# Channel the target switch reads to prove the live machine is reachable.",
        3,
        "sha256:ce6a8cf56ab8a75fd66cda2b9d67d9a9e465bae5e3d03f25c8aba1ca6e265e75",
        0,
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    WriteStanza(
        "control-assistant",
        "# Seconds an ENS lookup, a property read or a property set is given.",
        1,
        "sha256:eacda895bd3b9b217dea11b3de9c5bdaf04bdba82a45e527c43ced885e4329e2",
        1,
        "sha256:c71d12de9f95e34dfe6e9a4e3c62c6b4a26e59b90b1c73eee9373493289b4c6b",
    ),
    WriteStanza(
        "control-assistant",
        "# Property the target switch reads to prove this machine is reachable.",
        3,
        "sha256:268822f74f972ad272b373bc687b5419e76c3c45be3fd9d141daa8562253af2f",
        0,
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    WriteStanza(
        "control-assistant",
        "# control_system.connector.tango.probe_channel: sys/tg_test/1/ampli",
        2,
        "sha256:09c939480a710bcd937052fb6ca9076e2b1f1ba29bbb7f5a92e64cd4fe68ae90",  # osprey:not-a-port — digest digits
        1,
        "sha256:a8668b9914da648f2c88cf1cef503f5d2ef4ff64bd642957c7f2d0b594d6df1e",
    ),
    # The writes-check hook's entry, which named the retired key; a stanza is
    # compared line by line, so a list entry's own comment is one too.
    WriteStanza(
        "control-assistant",
        "- limits            # Enforce per-channel min/max limits before writes",
        1,
        "sha256:e80aca8e8c01cb79c19c83637e2958d48b8e86843a5b865a4ea618be86d3806e",
        1,
        "sha256:8f18a376a22a4d0943887f7020bc7a6c0536d21039ee1f22ab541b117a133999",
    ),
    # The stand-in's comment, which named the banner the real machine wears.
    WriteStanza(
        "control-assistant",
        "# Delete the line on a laptop: the simulator image is amd64-only, so a",
        3,
        "sha256:7557ddf3c86a9245e4d5b1dc31a72d46a89e8c2fd425b42702f6c42170e11afe",
        3,
        "sha256:24247dd971887136c2a7e38327cc656e7a5c6d9ee9b6bcaa6bb5fed6e8c0b477",
    ),
    # The approval block's words, which now say the slider decides for the
    # tools the write switch governs.
    WriteStanza(
        "control-assistant",
        "approval.enabled: true",
        3,
        "sha256:f559167d66c10f91eb024fd3f8cd27ff24e77ae987ec7e6c96464710acadfdaf",
        4,
        "sha256:cdcc4f753da2040b7472a81ef99f4c8ce6c0db5f8a99f1d329927456cb2b828c",
    ),
    WriteStanza(
        "control-assistant",
        "approval.tools.channel_read: skip",
        3,
        "sha256:fcde28785b5bed12ec45b719243409fcb553a64337e45424854b6380f7bd1c43",
        6,
        "sha256:b5b4f69cc82513f1dbe833a6bb79f28a3ceaae681c69ac712c1aacc6c0392abc",
    ),
    WriteStanza(
        "hello-world",
        "# write is refused before the limits check or the approval prompt is even",
        1,
        "sha256:cae435052c5791048c3acd5a74b7d278beac811cf47826bd2bd16e17c49823e9",
        1,
        "sha256:383412beeae8fa33449e63d1ebdb2e794e1a6ee70ec4c6d3d20334dd15d88562",
    ),
    WriteStanza(
        "hello-world",
        "control_system.writes: never",
        1,
        "sha256:88194b282cd242f33511a448710834fec5058da5c5925ba1d331d998f33bcfcb",
        1,
        "sha256:0d878f39d94a995eb416fd0ff186946e2619cb5347f9deef398c7a3a1febfe5b",
    ),
    WriteStanza(
        "tiers/readonly",
        "# at every write surface: channel-write tools land in the rendered",
        4,
        "sha256:2651efc04017af2746b818e064ecf4b10f6748a79877c69bea8d5421162c60ba",
        3,
        "sha256:afe4814f5c9f725d1aa17a0ab628d0ed68f29c3e53a4cc1b79f7695ef16b3312",
    ),
    WriteStanza(
        "tiers/readonly",
        "control_system.writes: never",
        6,
        "sha256:6e8ec4d3d2c95777f8c7e9233505b1d3b700599c74998cbe972141ed9e484ed7",
        3,
        "sha256:4560e9dbfe539a16a4d4776cb1eaa868e0fe95154859b489c59799534ab51466",
    ),
    WriteStanza(
        "tiers/readonly",
        "#",
        1,
        "sha256:cfacbbb8563ef4d25cce60db32cb514f44a113b30d3b2cd90d2cdb2bd475bc8a",
        1,
        "sha256:31af145311961adff92ca6de4e8826a1d8131b9b0594cf1c029bf7869843b560",
    ),
    WriteStanza(
        "tiers/readwrite",
        "#",
        6,
        "sha256:b27562646623c3588a64e7185477ae24ef4b3cad9757733725209f53f60bcf86",
        4,
        "sha256:389c501b048b278a8c50d9a86457a73edbb07952510527fe9d504465e8c1bd31",
    ),
    WriteStanza(
        "tiers/readwrite",
        "control_system.writes: allowed",
        6,
        "sha256:f570e2a1d6767f477cb13bf517f0012496dc1fc8604a710519ac91bddca0f2dc",
        2,
        "sha256:5bad65a1da7fdf15488a9490dbd180e08ca051f1b0f4b0ebff6e6bed261d4434",
        "control_system.writes: asked",
    ),
    WriteStanza(
        "tiers/readwrite",
        "web.ui_mode: expert",
        1,
        "sha256:10d72c4916bfd1c7bb01e2a0f8302c118ed14fdd1c05aec617e6f1b2140058fe",
        1,
        "sha256:577f2e4790d551676cccda3ec34d49f56a0e6e1dc1255b99ce2872d261bd0fdc",
    ),
    WriteStanza(
        "tiers/admin",
        "# What makes this the admin tier is the deployment-editing block below, not a",
        5,
        "sha256:10020b2115f76dd4b0400534961374c024fcc9795a1c579f4dea2635433abf30",
        2,
        "sha256:e83a36e54cb84e7195fe19a4ee5bfe3e5b1ed0a87d7c0d3855a0314569cc6cf1",
    ),
    WriteStanza(
        "tiers/logbook",
        "#",
        3,
        "sha256:7c4114f59e4681876a8e717a2b4473341cb08217693ec7e6b6a874383d5cea71",
        2,
        "sha256:45f8414a6ae749ae66d67bab745ac0e474b11e48197a6c6ee4f523604e6eb3bc",
    ),
    WriteStanza(
        "tiers/logbook",
        "control_system.writes: never",
        2,
        "sha256:95d2f2142ddaed64959e9ae2e542ce94bc197b53e6b7e8471578f356d70582fd",
        1,
        "sha256:b9c78100d99e0727d05425c92d85d96b59af2bc35f680a150f8be51d51815def",
    ),
    WriteStanza(
        "tiers/knowledge",
        "#",
        3,
        "sha256:7c4114f59e4681876a8e717a2b4473341cb08217693ec7e6b6a874383d5cea71",
        2,
        "sha256:45f8414a6ae749ae66d67bab745ac0e474b11e48197a6c6ee4f523604e6eb3bc",
    ),
    WriteStanza(
        "tiers/knowledge",
        "control_system.writes: never",
        2,
        "sha256:95d2f2142ddaed64959e9ae2e542ce94bc197b53e6b7e8471578f356d70582fd",
        1,
        "sha256:b9c78100d99e0727d05425c92d85d96b59af2bc35f680a150f8be51d51815def",
    ),
    WriteStanza(
        "tiers/channel-finder",
        "#",
        3,
        "sha256:7c4114f59e4681876a8e717a2b4473341cb08217693ec7e6b6a874383d5cea71",
        2,
        "sha256:45f8414a6ae749ae66d67bab745ac0e474b11e48197a6c6ee4f523604e6eb3bc",
    ),
    WriteStanza(
        "tiers/channel-finder",
        "control_system.writes: never",
        2,
        "sha256:95d2f2142ddaed64959e9ae2e542ce94bc197b53e6b7e8471578f356d70582fd",
        1,
        "sha256:b9c78100d99e0727d05425c92d85d96b59af2bc35f680a150f8be51d51815def",
    ),
)

#: A commented example of a retired write key, any spelling: dropped.
_RETIRED_WRITE_EXAMPLE = re.compile(r"^\s*#\s*(?:[^\s#:]+\.)?writes_enabled\s*:")

#: The tier key, dotted from ``config:``.
_TIER_KEY = f"control_system.{WRITES_KEY_LEAF}"

#: The tier key in its dotted spelling, as a line of a ``config:`` block.
_TIER_KEY_LINE = re.compile(r"^\s*control_system\.writes\s*:")


def _stanza_hash(lines: list[str]) -> str:
    """Hash of a stanza's lines with their indentation and line ends removed."""
    canonical = "\n".join(line.strip() for line in lines)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _preset_stanza(stanza: WriteStanza) -> list[str] | None:
    """The lines the bundled preset writes above *stanza*'s anchor today; ``None`` if it no longer does."""
    from .build_profile_presets import _presets_dir

    path = _presets_dir() / f"{stanza.preset}.yml"
    try:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        return None
    anchor = stanza.preset_anchor or stanza.anchor
    for index, line in enumerate(lines):
        if line.strip() != anchor or index < stanza.new_lines:
            continue
        region = lines[index - stanza.new_lines : index]
        if _stanza_hash(region) == stanza.new_hash:
            column = _indent(line)
            return [entry[min(column, _indent(entry)) :] for entry in region]
    return None


def _rewrite_write_keys(document: Any, rel: str, steps: list[str], *, shared: bool = False) -> bool:
    """Rewrite the retired write-posture keys of one document to the tier key, in place.

    The decision is the parse point's (:func:`retired_write_answer`), except
    for a shared login (*shared*) and a login file still in ``apps/``, which
    is moved to ``tiers/`` as one: both are always ``never``, so a shared
    login's file never says ``allowed``; the placement is
    :func:`~osprey.cli.build_profile_document._place_tier_key`'s, done with the
    comment-keeping helpers: the tier key takes the place of the retired flat
    key the render reads, else goes at the end of a nested ``control_system:``
    block, else in front of the first per-type key. A connector block the
    rewrite empties is reported in *steps*, not removed.

    Returns:
        Whether the document spelled a retired key.

    Raises:
        BuildProfileError: When it also spells ``control_system.writes``.
    """
    config = document.get("config") if isinstance(document, dict) else None
    if not isinstance(config, dict):
        return False
    found = _WriteSpellings()
    _scan_writes(config, (), None, found)
    if not (found.flat or found.per_type):
        return False
    if found.new:
        raise _mixed_writes_error(found, rel)
    answer = (
        "never"
        if shared or PurePosixPath(rel).parts[0] == APPS_DIRNAME
        else retired_write_answer(config) or "never"
    )

    kept = _kept_flat_leaf(found)

    if kept is None and found.block is not None:
        block: Any = found.block  # a CommentedMap: the document is round-trip loaded
        last = list(block.keys())[-1]
        pred = _last_slot(block, last)
        column = _key_column(block, next(iter(block)), 2)
        block.insert(len(block), WRITES_KEY_LEAF, answer)
        _place_after(pred, block, WRITES_KEY_LEAF, column, [])
    elif kept is None:
        top = found.per_type[0][2]
        commented: Any = config
        commented.insert(list(config.keys()).index(top), _TIER_KEY, answer)

    for container, key, _top, name in found.per_type:
        _drop_node(container, key, _predecessor(container, key), _key_column(container, key, 0))
        if container is not config and not len(container):
            steps.append(
                f"{rel}: `control_system.connector.{name}` held only the retired "
                f"`{RETIRED_WRITES_LEAF}`, so it is now empty; delete it, or give "
                f"that machine the settings it needs"
            )
    for container, key, _top in found.flat:
        if kept is not None and container is kept[0] and key == kept[1]:
            continue
        _drop_node(container, key, _predecessor(container, key), _key_column(container, key, 0))
    if kept is not None:
        container, key = kept
        _rename_key(container, key, WRITES_KEY_LEAF)
        renamed = key.rsplit(".", 1)[0] + "." + WRITES_KEY_LEAF if "." in key else WRITES_KEY_LEAF
        old = container[renamed]
        old_text = str(old).lower() if isinstance(old, bool) else str(old)
        container[renamed] = answer
        # The end-of-line comment keeps the gap the author left after the value.
        commented_map: Any = container  # a CommentedMap: the document is round-trip loaded
        entry = commented_map.ca.items.get(renamed)
        token = entry[2] if entry and len(entry) > 2 else None
        if isinstance(token, CommentToken) and not token.value.startswith("\n"):
            column = max(0, token.column + len(answer) - len(old_text))
            entry[2] = CommentToken(token.value, CommentMark(column))
    return True


def _drop_governed_policies(document: Any, rel: str, report: list[str]) -> bool:
    """Remove ``approval.tools.<tool>`` for every governed tool of one document, in place.

    The same set and spellings the parse point drops
    (:func:`~osprey.cli.build_profile_document.governed_policy_tools`), removed
    with the comment-keeping helpers; an ``approval.tools`` or ``approval``
    mapping the removal empties goes too. Each document that lost a line is
    reported in *report* (:attr:`RevalidateResult.retired_policies`) with the
    consequences the build's warning names: nothing is left to do by hand.

    Returns:
        Whether anything was removed.
    """
    config = document.get("config") if isinstance(document, dict) else None
    if not isinstance(config, dict):
        return False
    hits = governed_policy_spellings(config, governed_policy_tools(document))
    if not hits:
        return False
    for container, key, _tool, parents in hits:
        if key not in container:
            continue
        _drop_node(container, key, _predecessor(container, key), _key_column(container, key, 0))
        node = container
        for parent, parent_key in reversed(parents):
            if len(node) or parent.get(parent_key) is not node:
                break
            _drop_node(
                parent,
                parent_key,
                _predecessor(parent, parent_key),
                _key_column(parent, parent_key, 0),
            )
            node = parent
    tools = sorted({tool for _c, _k, tool, _p in hits})
    report.append(f"{rel}: {retired_approval_policy_message(tools)}")
    return True


def _rewrite_write_prose(text: str, rel: str, steps: list[str], *, rewrote: bool) -> str:
    """*text* with the previous release's write-posture comments brought to this one.

    A stanza in :data:`WRITE_STANZAS` is replaced by the bundled preset's
    current stanza, and every commented example of a retired write key is
    dropped. Prose left describing the retired keys — a comment line that names
    ``writes_enabled``, the comment above a dropped example, or (when
    *rewrote*) the comment above the tier key the rewrite wrote — is named in
    *steps* for a hand edit. A line that is not a whole comment (the end of a
    ``hooks:`` entry, say) is never named; only a stanza replaces one.
    """
    lines = text.splitlines(keepends=True)
    replacements: list[tuple[int, int, list[str]]] = []
    for stanza in WRITE_STANZAS:
        for index, line in enumerate(lines):
            if line.strip() != stanza.anchor or index < stanza.old_lines:
                continue
            start = index - stanza.old_lines
            if _stanza_hash(lines[start:index]) != stanza.old_hash:
                continue
            if any(s < index and start < e for s, e, _ in replacements):
                continue
            current = _preset_stanza(stanza)
            if current is None:
                continue
            pad = " " * _indent(line)
            replacements.append(
                (start, index, [pad + entry if entry.strip() else entry for entry in current])
            )
    # Each line paired with whether a replaced stanza now sits right above it.
    marked = [(line, False) for line in lines]
    for start, end, new in sorted(replacements, reverse=True):
        marked[start : end + 1] = [(line, True) for line in new] + [(marked[end][0], True)]

    prose: list[str] = []
    kept: list[tuple[str, bool]] = []
    for line, fresh in marked:
        if _RETIRED_WRITE_EXAMPLE.match(line):
            above = kept[-1] if kept else None
            if above and not above[1] and _is_prose(above[0]):
                prose.append(f"`{above[0].strip()}`, above the dropped `{line.strip()}`")
            continue
        kept.append((line, fresh))
    for index, (line, fresh) in enumerate(kept):
        stripped = line.strip()
        if stripped.startswith("#") and RETIRED_WRITES_LEAF in stripped:
            prose.append(f"`{stripped}`")
        elif rewrote and not fresh and _TIER_KEY_LINE.match(line):
            first = index
            while first and not kept[first - 1][1] and _is_prose(kept[first - 1][0]):
                first -= 1
            if first < index:
                prose.append(f"the comment from `{kept[first][0].strip()}` down to `{stripped}`")
    if prose:
        unique = list(dict.fromkeys(prose))
        steps.append(
            f"{rel}: comments written for the retired `{RETIRED_WRITES_LEAF}` keys "
            f"remain; reword them for `{_TIER_KEY}` (`allowed` or `never`): " + "; ".join(unique)
        )
    return "".join(line for line, _fresh in kept)


def _is_prose(line: str) -> bool:
    """A whole-line comment with words in it (not a ``#`` spacer, not a key line)."""
    stripped = line.strip()
    if not stripped.startswith("#"):
        return False
    words = stripped.lstrip("#").strip()
    return bool(words) and not _COMMENTED_KEY.match(words)


#: A commented-out key line (``control_system.connector.epics.timeout_s: 5.0``).
_COMMENTED_KEY = re.compile(r"^[\w\-]*[._][\w.\-]*:(\s|$)")


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


def _rename_apps_groups(config: Any) -> bool:
    """Respell every ``landing.groups`` entry ``type: apps`` as ``type: shared``, in place.

    The retired spelling is read as ``shared`` for one release; rewriting it
    here is what keeps a rewritten deployment from showing two sections of
    shared cards once :func:`_ensure_shared_group` runs.
    """
    renamed = False
    for container, key in _locate(config, ("modules", "web_terminals", "landing", "groups")):
        groups = container[key]
        for group in groups if isinstance(groups, list) else []:
            if isinstance(group, dict) and group.get("type") == _RETIRED_SHARED_GROUP_TYPE:
                group["type"] = _SHARED_GROUP_TYPE
                renamed = True
    return renamed


def _ensure_shared_group(config: Any, preset: _PresetRoster) -> bool:
    """Give an explicit ``landing.groups`` the section ``shared:`` cards show in."""
    for container, key in _locate(config, ("modules", "web_terminals", "landing", "groups")):
        groups = container[key]
        if not isinstance(groups, list) or not groups:
            continue
        if any(
            isinstance(g, dict)
            and g.get("type") in (_SHARED_GROUP_TYPE, _RETIRED_SHARED_GROUP_TYPE)
            for g in groups
        ):
            return False
        group = CommentedMap()
        for name, value in (preset.shared_group or {"type": _SHARED_GROUP_TYPE}).items():
            group[name] = value
        column = _item_column(groups, 0)
        pred = _last_slot(groups, len(groups) - 1)
        groups.append(group)
        _place_after(pred, groups, len(groups) - 1, column, preset.shared_group_lead)
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
    # `apps/` is not a layer directory, but its login files are moved from it.
    for dirname in (*LAYER_DIRNAMES, APPS_DIRNAME, "profiles", CONTEXT_DIRNAME):
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
            stripped = stripped.removeprefix(" ")
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

    Three reasons, each about a key whose absence is correct rather than
    accidental: it is, or sits below, the deployment's own data
    (:func:`_deployment_data_keys`); its fact is stated in another block of
    this profile (:func:`_claimed_elsewhere`); or it is the tier key
    ``control_system.writes``, which a silent root reads as ``never`` and a
    preset never decides for a deployment.
    """
    if not lacking:
        return {}
    owned = _deployment_data_keys()
    claimed = _claimed_elsewhere(document)
    skipped: dict[str, str] = {}
    for key in lacking:
        if key == _TIER_KEY:
            skipped[key] = (
                "a root that does not state it reads as `never`; whether this "
                "deployment allows writes is never filled in from a preset."
            )
            continue
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


def _stated_tier_word(config: Any) -> Any:
    """The ``control_system.writes`` value *config* states, in any spelling; ``None`` when silent."""
    if not hasattr(config, "items"):
        return None
    return _leaf_values(config).get(_TIER_KEY)


def _tier_drift_note(stated: Any, preset: str, preset_word: Any) -> str:
    """Why the fill leaves a stated tier word alone, and what ``osprey validate`` says of it."""
    shipped = f"`{preset_word}`" if preset_word is not None else "no value"
    return (
        f"this profile states `{stated}` and preset `{preset}` ships {shipped}; a preset "
        "never decides whether this deployment allows writes, so it is kept. `osprey "
        "validate` reports the difference as a preset-drift NOTE, not a refusal; a "
        "`# DEVIATION:` comment above the line, or spelling the preset's word, silences it."
    )


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
    stated = _stated_tier_word(document.get("config"))
    preset_word = values.get(_TIER_KEY)
    if _TIER_KEY not in skipped and stated is not None and stated != preset_word:
        skipped[_TIER_KEY] = _tier_drift_note(stated, preset, preset_word)
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


def _rewrite_shipped_shadows(document: Any, artifacts: list[str]) -> list[str]:
    """Turn every bare entry the deployment ships a file for into that file's path, in place.

    Before the by-name rule such a file replaced the library artifact wherever
    the bare name was selected; now it does only where its path is named. The
    entry keeps its position and its line comment.

    Args:
        document: A round-trip profile or app document.
        artifacts: The repo's list-selected artifacts, as the build names them.

    Returns:
        The paths written, in list order.
    """
    from .profile_conventions import SELECTED_CONVENTION_SOURCES, library_entry_shadow

    shipped = set(artifacts)
    written: list[str] = []
    if not isinstance(document, CommentedMap):
        return written
    for field_name in SELECTED_CONVENTION_SOURCES:
        seq = document.get(field_name)
        if not isinstance(seq, CommentedSeq):
            continue
        for index, entry in enumerate(seq):
            path = library_entry_shadow(field_name, entry)
            if path is not None and path in shipped and path not in seq:
                seq[index] = path
                written.append(path)
    return written


def _owns_lists(text: str, rel: str) -> bool:
    """Whether the login file *text* says ``lists: own``."""
    data = _safe_load(text, rel)
    return isinstance(data, dict) and data.get("lists") == "own"


def _own_list_logins(root: Path) -> list[str]:
    """The ``tiers/<name>.yml`` files of *root* that say ``lists: own``, repo-relative."""
    tiers_dir = root / TIERS_DIRNAME
    if not tiers_dir.is_dir():
        return []
    return [
        f"{TIERS_DIRNAME}/{path.name}"
        for path in sorted(tiers_dir.glob("*.yml"))
        if _owns_lists(path.read_text(encoding="utf-8"), f"{TIERS_DIRNAME}/{path.name}")
    ]


def name_shipped_shadows(root: Path) -> list[str]:
    """Name every library-named file the repo at *root* ships by its path, in place.

    ``osprey init`` seeds the deployment's own copy of the facility description
    beside a profile that selects the library rule by its bare name; this is
    what makes the profile, and every ``lists: own`` login, select the copy
    instead.

    Args:
        root: The repo root; holds ``profile.yml`` and maybe ``tiers/``.

    Returns:
        The files rewritten, repo-relative.
    """
    from .profile_conventions import unselected_convention_artifacts

    shipped = unselected_convention_artifacts(root, frozenset())
    if not shipped:
        return []
    rels = [PROFILE_FILENAME, *_own_list_logins(root)]
    changed: list[str] = []
    for rel in rels:
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        rewritten = _rewrite_shadow_lines(text, rel, shipped)
        if rewritten != text:
            path.write_text(rewritten, encoding="utf-8")
            changed.append(rel)
    return changed


def _name_convention_artifacts_text(text: str, rel: str, artifacts: list[str]) -> str:
    """:func:`_name_convention_artifacts` on a file's text, changing nothing else in it.

    For a file revalidate does not otherwise re-dump — an app — so its quoting,
    its empty values and its comments come back byte for byte: a bare entry the
    repo ships a file for becomes that path on its own line, and every artifact
    still unnamed is added as a line of its own.
    """
    from .profile_conventions import SELECTED_CONVENTION_SOURCES, convention_selections

    text = _rewrite_shadow_lines(text, rel, artifacts)
    document = _load_text(text, rel)
    if not isinstance(document, CommentedMap):
        return text
    selected = convention_selections(
        {field: document.get(field) or [] for field in SELECTED_CONVENTION_SOURCES}
    )
    field_for = {source: field for field, source in SELECTED_CONVENTION_SOURCES.items()}
    missing: dict[str, list[str]] = {}
    for artifact in artifacts:
        if artifact not in selected:
            missing.setdefault(field_for[artifact.partition("/")[0]], []).append(artifact)
    for field_name, entries in missing.items():
        text = _append_list_lines(text, rel, field_name, entries)
    return text


def _append_list_lines(text: str, rel: str, field_name: str, entries: list[str]) -> str:
    """*text* with *entries* added to its top-level list *field_name*, line by line.

    A block list gains one line per entry under its last item, a one-line flow
    list gains them inside its brackets, an empty key gains a block under it,
    and an absent key is added as a block above ``config:`` and the comment
    that introduces it (at the end of a file with no ``config:``).
    """
    document = _load_text(text, rel)
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    value = document.get(field_name)
    key_line, key_col = document.lc.key(field_name) if field_name in document else (0, 0)
    # Read off the text: ruamel does not mark an empty `[]` as flow style.
    after_key = lines[key_line][key_col:].split(":", 1)[-1].lstrip() if value is not None else ""
    if isinstance(value, CommentedSeq) and after_key.startswith("["):
        line = lines[key_line]
        start = line.find("[", key_col)
        end = line.find("]", start)
        if start < 0 or end < 0:
            raise BuildProfileError(
                f"{rel}: `{field_name}:` is a flow list over several lines; write it one "
                f"entry per line and run `osprey init --revalidate` again"
            )
        inner = line[start + 1 : end].rstrip()
        joined = ", ".join([inner, *entries]) if inner.strip() else ", ".join(entries)
        lines[key_line] = line[: start + 1] + joined + line[end:]
    elif isinstance(value, CommentedSeq) and len(value):
        first_line, first_col = value.lc.item(0)
        prefix = lines[first_line][:first_col]
        last_line = value.lc.item(len(value) - 1)[0]
        lines[last_line + 1 : last_line + 1] = [f"{prefix}{entry}\n" for entry in entries]
    elif field_name in document:
        block = [f"{' ' * (key_col + 2)}- {entry}\n" for entry in entries]
        lines[key_line + 1 : key_line + 1] = block
    else:
        at = len(lines)
        if "config" in document:
            at = document.lc.key("config")[0]
            while at and lines[at - 1].startswith("#"):
                at -= 1
        lines[at:at] = [f"{field_name}:\n", *(f"  - {entry}\n" for entry in entries)]
    return "".join(lines)


def shipped_shadow_texts(root: Path, texts: dict[str, str]) -> dict[str, str]:
    """*texts* (repo-relative path → YAML) as :func:`name_shipped_shadows` leaves them in *root*."""
    from .profile_conventions import unselected_convention_artifacts

    shipped = unselected_convention_artifacts(root, frozenset())
    if not shipped:
        return dict(texts)
    return {rel: _rewrite_shadow_lines(text, rel, shipped) for rel, text in texts.items()}


def _rewrite_shadow_lines(text: str, rel: str, shipped: list[str]) -> str:
    """*text* with each bare entry the repo ships a file for spelled as its path.

    Edited line by line rather than re-dumped, so a file ``osprey init`` just
    emitted changes in those entries and nowhere else: the round-trip writer
    does not keep every scalar's spelling (an empty value comes back ``null``).
    A line comment keeps its column where the longer entry leaves room.
    """
    from .profile_conventions import SELECTED_CONVENTION_SOURCES, library_entry_shadow

    document = _load_text(text, rel)
    if not isinstance(document, CommentedMap):
        return text
    lines = text.splitlines(keepends=True)
    shipped_set = set(shipped)
    for field_name in SELECTED_CONVENTION_SOURCES:
        seq = document.get(field_name)
        if not isinstance(seq, CommentedSeq) or seq.fa.flow_style():
            continue
        for index, entry in enumerate(seq):
            path = library_entry_shadow(field_name, entry)
            if path is None or path not in shipped_set or path in seq:
                continue
            line_no, column = seq.lc.item(index)
            line = lines[line_no]
            if line[column : column + len(entry)] != entry:
                continue
            rest = line[column + len(entry) :]
            pad = len(rest) - len(rest.lstrip(" "))
            grow = len(path) - len(entry)
            if rest.lstrip(" ").startswith("#") and pad > grow:
                rest = rest[grow:]
            lines[line_no] = line[:column] + path + rest
    return "".join(lines)


def _name_convention_artifacts(document: Any, artifacts: list[str]) -> list[str]:
    """Append every artifact in *artifacts* that *document*'s lists do not select, in place.

    Before the by-name rule a profile's own skills, agents, rules, output styles
    and hooks rendered into every render, and one named like a library artifact
    replaced it wherever the bare name was selected; now each renders only where
    its path is named. So a bare entry the deployment ships a file for becomes
    that path, in place, and every artifact still unnamed is appended — which
    keeps the file's render exactly what it was. No kind is left out for any
    file: trimming one is an edit the operator makes afterwards.

    Args:
        document: A round-trip profile or app document.
        artifacts: Every list-selected convention artifact the repo ships, as
            the build names them (``skills/beam-check``, ``hooks/osprey_limits.py``);
            a facility hook is never among them (its declaration wires it).

    Returns:
        The paths written: rewritten bare entries first, then those appended.
    """
    from .profile_conventions import SELECTED_CONVENTION_SOURCES, convention_selections

    if not isinstance(document, CommentedMap):
        return []
    added = _rewrite_shipped_shadows(document, artifacts)
    lists = {
        field: value if isinstance(value, list) else []
        for field in SELECTED_CONVENTION_SOURCES
        for value in (document.get(field),)
    }
    selected = convention_selections(lists)
    field_for = {source: field for field, source in SELECTED_CONVENTION_SOURCES.items()}
    for artifact in artifacts:
        if artifact in selected:
            continue
        field_name = field_for[artifact.partition("/")[0]]
        seq = document.get(field_name)
        if not isinstance(seq, CommentedSeq):
            keys = list(document.keys())
            if field_name in document:
                position = keys.index(field_name)
                document.pop(field_name)
            else:
                position = keys.index("config") if "config" in keys else len(keys)
            pred = _last_slot(document, keys[position - 1]) if position else None
            document.insert(position, field_name, CommentedSeq())
            seq = document[field_name]
            seq.append(artifact)
            if pred is not None:
                # What followed the key before it introduced the next key; it
                # still does, rather than now sitting above this list.
                _place_after(pred, document, field_name, 0, [])
        elif not len(seq):
            # `skills: []` grows into a block list, the way every list with
            # entries is spelled.
            seq.fa.set_block_style()
            seq.append(artifact)
        else:
            column = _item_column(seq, 2)
            pred = _last_slot(seq, len(seq) - 1)
            seq.append(artifact)
            _place_after(pred, seq, len(seq) - 1, column, [])
        added.append(artifact)
    return added


def _listed_artifacts(artifacts: list[str]) -> list[str]:
    """The artifacts of *artifacts* a list may name: every one but a facility hook.

    A facility hook is wired, and so copied, by its ``claude_code.hooks``
    declaration alone; only a file replacing an OSPREY hook is listed.
    """
    from .profile_conventions import replaces_library_artifact

    return [a for a in artifacts if not a.startswith("hooks/") or replaces_library_artifact(a)]


def _host_session_hooks(config: Any) -> dict[str, list[Any]]:
    """The host's facility hook declarations on every event but the guard event.

    Before the by-name rule an app inherited every host declaration. A guard
    still reaches every app (guards are a floor); any other event reaches only
    an app whose own ``config:`` declares it, so the migration writes it there.
    """
    from .profile_conventions import GUARD_EVENT, declared_hooks

    return {
        event: entries
        for event, entries in declared_hooks(config if isinstance(config, dict) else {}).items()
        if event != GUARD_EVENT and entries
    }


def _yaml_flow(entry: Any) -> str:
    """One declaration entry as a single-line YAML value."""
    if isinstance(entry, str):
        return entry
    return yaml.safe_dump(dict(entry), default_flow_style=True, sort_keys=False).strip()


def _carry_host_hooks_text(text: str, rel: str, declared: dict[str, list[Any]]) -> str:
    """*text* (an app file) declaring each of the host's *declared* hooks it does not yet.

    Each missing entry is added under the app's own dotted
    ``claude_code.hooks.<event>`` key — appended to it when the app has one,
    added at the top of ``config:`` otherwise — and nothing else in the file
    changes.

    Raises:
        BuildProfileError: If the app declares that event in another spelling,
            which a line edit cannot extend safely.
    """
    from .profile_conventions import declared_hook_path, declared_hooks

    for event, entries in declared.items():
        document = _load_text(text, rel)
        if not isinstance(document, CommentedMap):
            return text
        config = document.get("config")
        own = declared_hooks(config) if isinstance(config, CommentedMap) else {}
        have = {declared_hook_path(e) for e in own.get(event, [])}
        missing = [e for e in entries if declared_hook_path(e) not in have]
        if not missing:
            continue
        key = f"claude_code.hooks.{event}"
        lines = text.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        value = config.get(key) if isinstance(config, CommentedMap) else None
        if isinstance(value, CommentedSeq) and len(value) and not value.fa.flow_style():
            first_line, first_col = value.lc.item(0)
            prefix = lines[first_line][:first_col]
            last_line = value.lc.item(len(value) - 1)[0]
            lines[last_line + 1 : last_line + 1] = [f"{prefix}{_yaml_flow(e)}\n" for e in missing]
        elif event in own:
            raise BuildProfileError(
                f"{rel}: declares claude_code.hooks.{event} in a spelling revalidate cannot "
                f"extend; add {', '.join(_yaml_flow(e) for e in missing)} to it by hand, or "
                f"spell it as one `{key}:` block list and run `osprey init --revalidate` again"
            )
        else:
            if isinstance(config, CommentedMap) and len(config):
                at, column = config.lc.key(next(iter(config)))
                indent = " " * column
            elif "config" in document:
                at, indent = document.lc.key("config")[0] + 1, "  "
            else:
                lines.append("config:\n")
                at, indent = len(lines), "  "
            block = [
                f"{indent}# Carried from profile.yml: an app runs a host hook on {event} "
                "only where it declares it.\n",
                f"{indent}{key}:\n",
                *(f"{indent}  - {_yaml_flow(e)}\n" for e in missing),
            ]
            lines[at:at] = block
        text = "".join(lines)
    return text


def _catalog_stem(build_profile: Any) -> str | None:
    """``<s>`` when a catalog ``build_profile`` names ``<d>/<s>.yml`` for a directory ``<d>`` read as tiers."""
    if not isinstance(build_profile, str):
        return None
    dirs = "|".join(re.escape(d) for d in _TIER_DIRS_RETIRED_FIRST)
    match = re.fullmatch(rf"(?:\./)?(?:{dirs})/([^/]+)\.yml", build_profile.strip())
    return match.group(1) if match else None


def _moved_login_text(
    root: Path, text: str, rel: str, target: str, profile_name: str, steps: list[str]
) -> str:
    """The ``tiers/`` file a login file left in ``apps/`` becomes.

    One that was never edited — it hashes to what ``osprey init`` emitted into
    ``apps/`` (:data:`MOVED_APP_LOGINS`), or to what it emits into ``tiers/``
    today, shipped shadows named by path either way — is re-emitted from the
    bundled login it names, which restamps its provenance. An edited one keeps
    its lines and gains ``lists: own`` and, unless it spells it,
    ``control_system.writes: never``; *steps* says how ``osprey validate``
    reports where it differs from the bundled login.
    """
    from .build_profile_emit import emit_layer_delta_yaml
    from .build_profile_presets import list_tiers

    data = _safe_load(text, rel)
    source = _layer_source(text, data)
    bundled = source.rsplit("/", 1)[-1] if source else None
    if bundled in list_tiers():
        normalized = shipped_shadow_texts(root, {rel: text})[rel]
        emitted = shipped_shadow_texts(
            root,
            {target: emit_layer_delta_yaml(f"{TIERS_DIRNAME}/{bundled}", profile_name, target)},
        )[target]
        hashes = (_content_hash(_safe_load(normalized, rel)), _text_hash(normalized))
        frozen = MOVED_APP_LOGINS.get(bundled)
        current = (_content_hash(_safe_load(emitted, target)), _text_hash(emitted))
        if (frozen is not None and hashes == tuple(frozen)) or hashes == current:
            return emitted

    moved = (
        text
        if isinstance(data, dict) and data.get("lists") == "own"
        else (_put_knobs(text, rel, {"lists": "own"}))
    )
    config = _safe_load(moved, rel).get("config") if isinstance(data, dict) else None
    if not (isinstance(config, dict) and list(_tier_key_values(config))):
        moved = _pin_never(moved, rel)
    note = (
        f"; `osprey validate` reports where it differs from the bundled `{bundled}` login — "
        f"claim each difference with a `# DEVIATION:` comment, or take the bundled file with "
        f"`osprey scaffold tiers --force`"
        if bundled in list_tiers()
        else ""
    )
    steps.append(
        f"{rel} has been edited, so it moved to {target} as written, with `lists: own` and "
        f"`{_TIER_KEY}: never`{note}"
    )
    return moved


def _pin_never(text: str, rel: str) -> str:
    """*text* with ``control_system.writes: never`` added at the end of its ``config:``."""
    document = _load_text(text, rel)
    if not isinstance(document, CommentedMap):
        return text
    config = document.get("config")
    if not isinstance(config, CommentedMap):
        document["config"] = CommentedMap({_TIER_KEY: "never"})
    else:
        config[_TIER_KEY] = "never"
    return _dump_text(document)


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

    # ── Tier files: keep, re-emit, or turn into a shared login ──────────────
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
    # Login files `osprey init` wrote into apps/ before logins moved to tiers/;
    # each is moved below, after the write-posture pass.
    apps_dir = root / APPS_DIRNAME
    existing_apps = {p.stem: p for p in sorted(apps_dir.glob("*.yml"))} if apps_dir.is_dir() else {}

    profile_name = str(document.get("name") or root.name)
    apps: dict[str, tuple[str, str]] = {}
    # The file each shared login is written to, by its name.
    shared_rels: dict[str, str] = {}
    for stem, layer in layers.items():
        data = _safe_load(layer.text, layer.origin)
        source = _layer_source(layer.text, data)
        retired = RETIRED_PRESETS.get(source) if source else None
        if retired is None:
            continue
        rel = layer.rel
        successor = retired.successor
        successor_rel = f"{TIERS_DIRNAME}/{successor}.yml"
        edited = (
            _content_hash(data) != retired.content_hash
            or _text_hash(layer.text) != retired.text_hash
        )
        taken = successor != stem and (successor in layers or successor in existing_apps)
        if retired.kind == _SHARED and not edited and taken:
            steps.append(
                f"{rel} was made from `{source}` and never edited, but {successor_rel} already "
                f"exists, so it stays a tier. To run it as the `{successor}` shared login: list "
                f"`{successor}` under `shared:` and remove {rel} and the users pinned to it"
            )
            continue
        if edited:
            if retired.kind == _SHARED:
                # The one edit worth naming: a model the card ran on, which the
                # bundled login does not set, so it stays in the file.
                model = data.get("model") if isinstance(data, dict) else None
                kept = f", its `model: {model}` among them," if model else ""
                steps.append(
                    f"{rel} was made from `{source}` and has been edited, so it stays a tier "
                    f"with your edits{kept} in it. To run it as the `{successor}` shared "
                    f"login: give {successor_rel} `lists: own` and the bundled "
                    f"`{successor}` login's lists (`osprey scaffold tiers --force` writes "
                    f"them; carry your edits over), list `{successor}` under `shared:`, and "
                    f"remove the users pinned to it"
                )
            else:
                steps.append(
                    f"{rel} was made from `{source}` and has been edited, so it is kept as "
                    f"it is; it has no provenance block, so `osprey validate` cannot report "
                    f"its drift from the bundled `{successor}` tier"
                )
            continue
        if retired.kind == _SHARED:
            if rel != successor_rel:
                deletes.add(rel)
            if successor_rel not in writes:
                writes[successor_rel] = emit_layer_delta_yaml(
                    f"{TIERS_DIRNAME}/{successor}", profile_name, successor_rel
                )
            apps[stem] = (successor, successor)
            shared_rels[successor] = successor_rel
        else:
            emitted = emit_layer_delta_yaml(f"{TIERS_DIRNAME}/{successor}", profile_name, rel)
            if stem != successor:
                # Titled after the file, as the previous release titled it.
                emitted = emitted.replace(
                    f"{profile_name} ({successor})", f"{profile_name} ({stem})"
                )
            writes[rel] = emitted

    # A shared login made by an earlier run that stopped before its users
    # moved: the users still pinned to it move now.
    tiers_left = {
        stem
        for stem, layer in layers.items()
        if layer.rel not in deletes and stem not in shared_rels
    }
    # The model provider is a deployment fact: a tier that spells one is
    # refused at build, and which provider the deployment runs is the
    # operator's call, so the key is named here rather than moved or dropped.
    for stem in sorted(tiers_left):
        layer = layers[stem]
        data = _safe_load(writes.get(layer.rel, layer.text), layer.rel)
        if isinstance(data, dict) and "provider" in data:
            steps.append(
                f"{layer.rel} spells `provider: {data['provider']}`, which a tier may not: "
                f"the provider lives in {PROFILE_FILENAME}; remove it from {layer.rel} and set "
                f"it in {PROFILE_FILENAME} if every login should run on it"
            )
    for stem, layer in layers.items():
        if stem in apps or stem in shared_rels or layer.rel in deletes:
            continue
        if _owns_lists(layer.text, layer.rel):
            source = _layer_source(layer.text, _safe_load(layer.text, layer.rel))
            if source and source not in RETIRED_PRESETS:
                # A shared login a run already moved (``logbook`` or ``tiers/logbook``).
                apps[stem] = (stem, source.rsplit("/", 1)[-1])
                shared_rels[stem] = layer.rel
                tiers_left.discard(stem)
    for stem, path in existing_apps.items():
        if stem in apps or stem in tiers_left:
            continue
        rel = f"{APPS_DIRNAME}/{path.name}"
        text = path.read_text(encoding="utf-8")
        source = _layer_source(text, _safe_load(text, rel))
        if source and source not in RETIRED_PRESETS:
            # Materialized from a bundled login (``logbook`` or ``apps/logbook``).
            apps[stem] = (stem, source.rsplit("/", 1)[-1])
            shared_rels.setdefault(stem, rel)

    # ── The roster ──────────────────────────────────────────────────────────
    config = document.get("config")
    if isinstance(config, dict):
        _rename_apps_groups(config)
    pinned_apps = {
        container[key]
        for _label, container, key in _tier_pins(config)
        if isinstance(container[key], str) and container[key] in apps
    }
    if pinned_apps and isinstance(config, dict):
        roster = _preset_roster()
        removed = _move_users_to_shared(config, apps, roster)
        if removed:
            _ensure_shared_group(config, roster)
        for user in removed:
            slot = f"{CONTEXT_DIRNAME}/{user}"
            if _context_slot_empty(root / slot):
                sweeps.append(slot)
            elif (root / slot).exists():
                steps.append(
                    f"{slot}/ held context for the {user!r} card, which is now a shared login "
                    f"with no per-user context; move what you need and delete the folder"
                )

    for rel, doc in documents.items():
        for label, container, key in _tier_pins(doc.get("config")):
            value = container[key]
            if not isinstance(value, str):
                continue
            if value in apps:
                steps.append(
                    f"{rel}: `{label}` names {value!r}, which is now the shared login "
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
                if knob in RENDER_KEYS:
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
            rel = shared_rels.get(apps[name][0], f"{TIERS_DIRNAME}/{apps[name][0]}.yml")
            text = writes.get(rel) or (root / rel).read_text(encoding="utf-8")
        elif name in tiers_left:
            rel = layers[name].rel
            text = writes.get(rel, layers[name].text)
        else:
            steps.append(
                f"the `{RETIRED_CATALOG_KEY}:` catalog set {', '.join(sorted(values))} for {name!r}, "
                f"which has no file in {TIERS_DIRNAME}/; set them at the "
                f"top of that file once it exists"
            )
            continue
        writes[rel] = _put_knobs(text, rel, values)

    # ── The write posture, in every file kept ───────────────────────────────
    # After the tier files are classified (a retired preset is recognized by
    # its text as the previous release wrote it) and before the fill, which
    # must see the tier key where the retired one stood rather than add it.
    for rel, doc in list(documents.items()):
        _drop_governed_policies(doc, rel, result.retired_policies)
        rewrote = _rewrite_write_keys(doc, rel, steps)
        dumped = _dump_text(doc)
        text = _rewrite_write_prose(dumped, rel, steps, rewrote=rewrote)
        if text != dumped:
            documents[rel] = _load_text(text, rel)
    document = documents[PROFILE_FILENAME]
    kept_layers = {layers[stem].rel: layers[stem].text for stem in sorted(tiers_left)}
    for stem, layer in layers.items():
        if stem in shared_rels and layer.rel == shared_rels[stem]:
            kept_layers.setdefault(layer.rel, layer.text)
    for path in existing_apps.values():
        kept_layers.setdefault(f"{APPS_DIRNAME}/{path.name}", path.read_text(encoding="utf-8"))
    shared_files = set(shared_rels.values())
    for rel, original in kept_layers.items():
        if rel in deletes:
            continue
        text = writes.get(rel, original)
        try:
            layer_doc = _load_text(text, rel)
        except BuildProfileError:
            # The round-trip reader is stricter than the build's (a key
            # written twice, say), so a file it refuses is not rewritten.
            data = _safe_load(text, rel)
            if isinstance(data, dict) and retired_write_answer(data.get("config")) is not None:
                steps.append(
                    f"{rel} spells the retired `{RETIRED_WRITES_LEAF}` keys but could not be "
                    f"rewritten in place; replace them with `{_TIER_KEY}: allowed` or "
                    f"`{_TIER_KEY}: never` by hand"
                )
            continue
        dropped = _drop_governed_policies(layer_doc, rel, result.retired_policies)
        rewrote = _rewrite_write_keys(layer_doc, rel, steps, shared=rel in shared_files)
        rewritten = _rewrite_write_prose(
            _dump_text(layer_doc) if rewrote or dropped else text, rel, steps, rewrote=rewrote
        )
        if rewritten != text:
            writes[rel] = rewritten
    # ── Login files left in apps/: moved to tiers/ ──────────────────────────
    # After the write-posture pass, so the text compared and moved is the one
    # that pass left: a moved file always says `never`.
    moved: dict[str, str] = {}
    for stem, path in existing_apps.items():
        rel = f"{APPS_DIRNAME}/{path.name}"
        moved_rel = f"{TIERS_DIRNAME}/{path.name}"
        if moved_rel in writes or (stem in layers and layers[stem].rel not in deletes):
            steps.append(
                f"{rel} is a login file, but {moved_rel} already exists, so it was not moved; "
                f"merge the two into {moved_rel} by hand and delete {rel}"
            )
            continue
        kept_text = writes.pop(rel, "") or path.read_text(encoding="utf-8")
        writes[moved_rel] = _moved_login_text(root, kept_text, rel, moved_rel, profile_name, steps)
        deletes.add(rel)
        moved[rel] = moved_rel
        if shared_rels.get(stem) == rel:
            shared_rels[stem] = moved_rel

    # ── Own skills, agents, rules, output styles, hooks: named ──────────────
    # Each now renders only where a list names it. profile.yml and every
    # `lists: own` login name all of them, which is what each render carried
    # before; a tier that adds inherits profile.yml's lists. A facility hook is
    # the exception: its declaration wires and copies it, so it is never
    # listed, and each `lists: own` login declares the host's non-guard hooks
    # it used to inherit.
    from .build_persistence import (
        FACILITY_RULE_NAME,
        FACILITY_RULE_PATH_ENTRY,
        FACILITY_RULE_RELPATH,
    )
    from .profile_conventions import BUILD_OUTPUT_DIR, unselected_convention_artifacts

    shipped = _listed_artifacts(unselected_convention_artifacts(root, frozenset()))
    # A repo from before the facility description moved into the profile keeps
    # its only, edited copy in build/; the bare `facility` it selects now means
    # OSPREY's rule, so that copy is moved into the profile (below, with the
    # other writes) and named by its path like every shipped shadow.
    rescued_facility_rule = (
        root / BUILD_OUTPUT_DIR / ".claude" / FACILITY_RULE_RELPATH
        if not (root / FACILITY_RULE_RELPATH).exists()
        and FACILITY_RULE_NAME in (document.get("rules") or [])
        and (root / BUILD_OUTPUT_DIR / ".claude" / FACILITY_RULE_RELPATH).is_file()
        else None
    )
    if rescued_facility_rule is not None:
        shipped = sorted({*shipped, FACILITY_RULE_PATH_ENTRY})
        writes[FACILITY_RULE_RELPATH.as_posix()] = rescued_facility_rule.read_text(encoding="utf-8")
    session_hooks = _host_session_hooks(document.get("config"))
    if shipped or session_hooks:
        if shipped:
            _name_convention_artifacts(document, shipped)
        own_rels = {
            rel
            for rel in (*_own_list_logins(root), *writes)
            if rel.startswith(f"{TIERS_DIRNAME}/") and rel.endswith(".yml")
        }
        own_rels = {
            rel
            for rel in own_rels
            if _owns_lists(writes.get(rel) or (root / rel).read_text(encoding="utf-8"), rel)
        }
        for rel in sorted(own_rels - deletes):
            text = writes.get(rel) or (root / rel).read_text(encoding="utf-8")
            named = _name_convention_artifacts_text(text, rel, shipped) if shipped else text
            named = _carry_host_hooks_text(named, rel, session_hooks)
            if named != text:
                writes[rel] = named

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
    renamed.update(moved)
    result.diff, result.changed = _diff(before, _snapshot(root), renamed)
    if resolved is not None:
        result.drift_refusals = _drift_refusals(
            root, resolved, preset_hash, plan.content_hash, result.deviation_marker
        )
    result.app_drift = _app_drift(root, result.deviation_marker)
    return result


def _app_drift(root: Path, marker: str) -> list[str]:
    """The application differences ``osprey validate`` reports, one line each.

    An application's files are the operator's to edit, so a difference is said
    and never rewritten: the copy stays as it is, and a marker in its
    ``.provenance`` sidecar claims it. Never fatal: a sidecar that does not
    parse reports nothing here and refuses where ``osprey validate`` reads it.
    """
    from .build_profile_drift import _compare_apps
    from .build_profile_schema import DEFAULT_DEVIATION_MARKER

    try:
        findings, _stale, notes = _compare_apps(root, marker or DEFAULT_DEVIATION_MARKER)
    except BuildProfileError:
        return []
    return [finding.render() for finding in findings if not finding.marked] + notes


def _plan_providers(root: Path) -> Any:
    """The ``providers.yml`` this repo is brought to; refuses before anything is written."""
    import click

    from .profile_cmd import _plan_provider_catalog

    try:
        return _plan_provider_catalog(root, keep_values=True)
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
