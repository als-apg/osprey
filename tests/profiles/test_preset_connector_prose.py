"""The connector and archiver types the root presets name in prose.

A preset is a static document: it cannot render its "the alternatives are ..."
comment from :mod:`osprey_connectors.types`, so the enumeration is hand-kept and
goes stale silently the moment a connector is added or removed. These tests are
the pin that makes that loud.

What is read is narrow on purpose: the value of ``control_system.type`` /
``archiver.type`` plus the comment block directly above it — the prose that
introduces the key and lists its alternatives. Inside that block every
double-quoted lower_snake token is a type name; elsewhere in the file a quoted
token is as likely to be an approval policy or a persona id.

Three rules, and the third is what keeps a partial list honest:

1. every type name the block mentions is a type the framework actually ships —
   a renamed or deleted connector cannot be left behind in the copy;
2. ``control-assistant``, the reference document, mentions every shipped type —
   a new connector nobody documented fails here;
3. a block that mentions only some of them points the reader at
   ``osprey config --defaults``, so a short list never reads as a complete one,
   and that ledger really does name every type — a pointer at a surface that
   enumerates nothing is worse than the short list it excuses.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from osprey_connectors.types import (
    CLI_ARCHIVER_TYPES,
    CLI_CONTROL_SYSTEM_TYPES,
    SET_CONTROL_SYSTEM_TYPES,
)

PRESET_DIR = Path(__file__).resolve().parents[2] / "src" / "osprey" / "profiles" / "presets"

#: The four documents an operator starts from.
ROOT_PRESETS = ("control-assistant", "hello-world", "ariel-standalone", "channel-finder-standalone")

#: The preset that enumerates the alternatives, and is therefore held to the
#: complete list. The other three describe only the one type they set.
REFERENCE_PRESET = "control-assistant"

#: Where a block sends a reader whose type it does not name.
FULL_LIST_POINTER = "osprey config --defaults"

#: ``live_standin`` is settable but not initable, so prose may name it while the
#: CLI choice list does not carry it.
SHIPPED = {
    "control_system.type": frozenset(SET_CONTROL_SYSTEM_TYPES),
    "archiver.type": frozenset(CLI_ARCHIVER_TYPES),
}

#: The complete list each block is measured against — what ``osprey init`` will
#: actually materialize, which is the narrower of the two control-system lists.
ADVERTISED = {
    "control_system.type": frozenset(CLI_CONTROL_SYSTEM_TYPES),
    "archiver.type": frozenset(CLI_ARCHIVER_TYPES),
}

_QUOTED_TOKEN_RE = re.compile(r'"([a-z][a-z0-9_]*)"')
#: The ledger writes prose, and prose quotes a name in backticks.
_BACKTICK_TOKEN_RE = re.compile(r"`([a-z][a-z0-9_]*)`")


def _type_block(preset: str, key: str) -> tuple[set[str], str] | None:
    """The names *key*'s block in *preset* mentions, and the block's text.

    Returns ``None`` when the preset does not set *key* at all — the two
    standalone presets set neither, and a preset that stays silent makes no
    claim to check.
    """
    lines = (PRESET_DIR / f"{preset}.yml").read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip().startswith(f"{key}:"):
            continue
        value = line.split(":", 1)[1].strip()
        comments: list[str] = []
        cursor = index - 1
        while cursor >= 0 and lines[cursor].strip().startswith("#"):
            comments.append(lines[cursor])
            cursor -= 1
        names = {value} if value else set()
        for comment in comments:
            names.update(_QUOTED_TOKEN_RE.findall(comment))
        return names, "\n".join(reversed(comments))
    return None


_CASES = [(preset, key) for preset in ROOT_PRESETS for key in SHIPPED]


@pytest.mark.parametrize(("preset", "key"), _CASES)
def test_prose_names_only_shipped_types(preset: str, key: str) -> None:
    """No preset advertises a connector or archiver the framework does not have."""
    block = _type_block(preset, key)
    if block is None:
        pytest.skip(f"{preset} does not set {key}")
    assert block[0] <= SHIPPED[key]


@pytest.mark.parametrize("key", sorted(ADVERTISED))
def test_the_reference_preset_names_every_shipped_type(key: str) -> None:
    """A connector added without a line in the reference document fails here."""
    block = _type_block(REFERENCE_PRESET, key)
    assert block is not None
    assert ADVERTISED[key] <= block[0]


@pytest.mark.parametrize(("preset", "key"), _CASES)
def test_a_partial_list_points_at_the_full_one(preset: str, key: str) -> None:
    """Prose that names some of the types must say where the rest are.

    A block naming all of them passes on its own; the failure is the middle
    case, which reads as a complete list and is not one.
    """
    block = _type_block(preset, key)
    if block is None:
        pytest.skip(f"{preset} does not set {key}")
    names, comment_text = block
    if ADVERTISED[key] <= names:
        return
    assert FULL_LIST_POINTER in comment_text


def _ledger_note(key: str) -> str:
    """The comment block ``osprey config --defaults`` prints above *key*."""
    from osprey.cli.config_cmd import _render_defaults_ledger

    lines = _render_defaults_ledger().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(f"{key}: "))
    note: list[str] = []
    cursor = index - 1
    while cursor >= 0 and lines[cursor].startswith("# ") and not lines[cursor].startswith("# ──"):
        note.append(lines[cursor])
        cursor -= 1
    return "\n".join(reversed(note))


@pytest.mark.parametrize("key", sorted(ADVERTISED))
def test_the_pointed_at_ledger_names_every_type(key: str) -> None:
    """The surface a partial list sends the reader to lists the whole set.

    ``control_system.type`` and ``archiver.type`` have no fallback, so the
    ledger's ``default:`` column prints ``<required>`` and nothing else. Their
    admissible values are a closed set with nowhere else to appear for a reader
    holding only the ledger, so the manifest's ``default_note`` names them and
    this is the pin that keeps that list whole.
    """
    named = set(_BACKTICK_TOKEN_RE.findall(_ledger_note(key)))
    assert ADVERTISED[key] <= named
    assert named <= SHIPPED[key]
