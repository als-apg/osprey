"""The profile page's key tables name exactly the keys the loader accepts.

``docs/source/reference/configuration/profile.rst`` opens by promising every
key a ``profile.yml`` accepts, and several of its blocks are closed: an
unknown key fails the build or warns. Each table is hand-written, and the
loader's closed sets are derived from dataclasses, so a field added in code
reaches the loader at once while the table never mentions it, and a field
renamed in code leaves the table advertising a spelling the build refuses.

``test_profile_dispatch_keys`` and ``test_profile_bluesky_keys`` guard two of
the blocks. This module guards the rest the same way, as a set equality in
both directions:

* the top-level field table against ``_KNOWN_PROFILE_KEYS``;
* the ``mcp_servers.<name>`` key table against ``McpServerDef``'s fields;
* the ``virtual_accelerator``, ``optimizer`` and ``environment`` blocks
  against their loader sets.

Each section is found by its RST anchor, and only the first key listing after
it is read, so a later table in the same section (the virtual accelerator's
``config:`` keys, say) is not mistaken for the block's own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from pathlib import Path

import pytest

from osprey.cli.build_profile_load import (
    _KNOWN_ENVIRONMENT_KEYS,
    _KNOWN_OPTIMIZER_KEYS,
    _KNOWN_PROFILE_KEYS,
    _KNOWN_VA_KEYS,
)
from osprey.cli.build_profile_schema import McpServerDef

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The page this sweep reads, relative to the repo root.
_PAGE = "docs/source/reference/configuration/profile.rst"

#: A ``list-table`` row's first cell: ``* - `` and the rest of the line.
_ROW_PATTERN = re.compile(r"^\s*\* - (.*)$")

#: One RST inline literal: how a table cell or a definition-list term writes a key.
_LITERAL_PATTERN = re.compile(r"^``([^`\n]+)``$")

#: Number words the prose uses to state a block's size.
_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


@dataclass(frozen=True)
class _Table:
    """One key listing on the page and the loader set it must equal."""

    anchor: str
    accepted: frozenset[str]
    style: str = "list-table"
    #: A regex whose group 1 is the number word the prose states, or None.
    count: str | None = None


_TABLES: dict[str, _Table] = {
    "top-level": _Table(".. _reference-profile:", _KNOWN_PROFILE_KEYS),
    "mcp_servers": _Table(
        ".. _profile-mcp-servers:", frozenset(f.name for f in fields(McpServerDef))
    ),
    "virtual_accelerator": _Table(".. _profile-virtual-accelerator:", _KNOWN_VA_KEYS),
    "optimizer": _Table(
        ".. _profile-optimizer:",
        _KNOWN_OPTIMIZER_KEYS,
        count=r"accepts exactly (\w+) keys",
    ),
    "environment": _Table(
        ".. _profile-environment:",
        _KNOWN_ENVIRONMENT_KEYS,
        style="definitions",
        count=r"All (\w+) keys are optional",
    ),
}


def _section(anchor: str, root_dir: Path | None = None) -> list[str]:
    """The lines between *anchor* and the next anchor after it."""
    path = (root_dir if root_dir is not None else _REPO_ROOT) / _PAGE
    lines = path.read_text(encoding="utf-8").splitlines()
    starts = [n for n, line in enumerate(lines) if line.strip() == anchor]
    if not starts:
        return []
    start = starts[0] + 1
    for offset, line in enumerate(lines[start:], start=start):
        if line.startswith(".. _") and line.rstrip().endswith(":"):
            return lines[start:offset]
    return lines[start:]


def _first_list_table(section: list[str]) -> list[str]:
    """The body of the first ``list-table`` in *section*: its indented lines."""
    for n, line in enumerate(section):
        if line.strip() == ".. list-table::":
            body = []
            for following in section[n + 1 :]:
                if following.strip() and not following.startswith(" "):
                    break
                body.append(following)
            return body
    return []


def _list_table_keys(section: list[str]) -> set[str]:
    """Every key a first cell names; ``a`` / ``b`` cells name several."""
    found: set[str] = set()
    for line in _first_list_table(section):
        row = _ROW_PATTERN.match(line)
        if row is None:
            continue
        parts = [part.strip() for part in row.group(1).split(" / ")]
        literals = [_LITERAL_PATTERN.match(part) for part in parts]
        if all(literals):
            found.update(m.group(1) for m in literals if m is not None)
    return found


def _definition_keys(section: list[str]) -> set[str]:
    """Every definition-list term that is a literal followed by an indented line."""
    found: set[str] = set()
    for line, following in zip(section, section[1:], strict=False):
        term = _LITERAL_PATTERN.match(line)
        if term is not None and following.startswith("   ") and following.strip():
            found.add(term.group(1))
    return found


def _documented(table: _Table, root_dir: Path | None = None) -> set[str]:
    section = _section(table.anchor, root_dir)
    if table.style == "definitions":
        return _definition_keys(section)
    return _list_table_keys(section)


@pytest.mark.parametrize("block", sorted(_TABLES))
def test_the_documented_keys_are_the_accepted_ones(block: str) -> None:
    """A table and its loader set name the same keys, in both directions."""
    table = _TABLES[block]
    documented = _documented(table)
    assert documented, (
        f"no keys parsed from the {table.anchor} section of {_PAGE}: "
        "the anchor or the table shape has moved"
    )
    undocumented = sorted(table.accepted - documented)
    unaccepted = sorted(documented - table.accepted)
    assert (undocumented, unaccepted) == ([], []), (
        f"The {block} keys in {_PAGE} must match the loader's set.\n"
        f"Accepted but undocumented: {undocumented}\n"
        f"Documented but not accepted: {unaccepted}"
    )


@pytest.mark.parametrize("block", sorted(b for b, t in _TABLES.items() if t.count))
def test_the_stated_key_count_is_the_accepted_one(block: str) -> None:
    """Prose that states a block's size states the loader's size."""
    table = _TABLES[block]
    assert table.count is not None
    text = " ".join(line.strip() for line in _section(table.anchor))
    match = re.search(table.count, text)
    assert match is not None, f"no {table.count!r} sentence in the {block} section of {_PAGE}"
    expected = _NUMBER_WORDS.get(len(table.accepted), str(len(table.accepted)))
    assert match.group(1) == expected, (
        f"{_PAGE} says the {block} block has {match.group(1)} keys; "
        f"the loader accepts {len(table.accepted)}"
    )


def test_only_the_first_table_after_the_anchor_is_read(tmp_path: Path) -> None:
    """A later table, or a later anchor's table, is not collected."""
    page = tmp_path / _PAGE
    page.parent.mkdir(parents=True)
    page.write_text(
        ".. _profile-virtual-accelerator:\n\n"
        ".. list-table::\n\n"
        "   * - Key\n"
        "     - Meaning\n"
        "   * - ``port``\n"
        "     - what it does\n"
        "   * - ``a`` / ``b``\n"
        "     - two keys, one row\n\n"
        "Prose.\n\n"
        ".. list-table::\n\n"
        "   * - ``control_system.type``\n"
        "     - a config key\n\n"
        ".. _elsewhere:\n\n"
        ".. list-table::\n\n"
        "   * - ``not_this_block``\n"
        "     - what it does\n",
        encoding="utf-8",
    )
    table = _Table(".. _profile-virtual-accelerator:", frozenset())
    assert _documented(table, tmp_path) == {"port", "a", "b"}
