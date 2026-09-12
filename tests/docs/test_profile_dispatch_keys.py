"""The documented ``dispatch:`` keys are exactly the ones the loader accepts.

``docs/source/reference/configuration/profile.rst`` calls the ``dispatch:``
block closed --- "every key it accepts is listed below" --- and then lists them
in a hand-written table. The loader enforces its own closed set,
:data:`~osprey.cli.build_profile_load._KNOWN_DISPATCH_KEYS`, derived from
``DispatchConfig``; until now nothing compared the two.

Both directions of that drift are quiet and both are expensive, because the
page's own claim is what makes them so:

* a field added to ``DispatchConfig`` reaches the loader immediately and the
  table never mentions it, so a closed list a reader trusts is missing a knob
  that exists;
* a field renamed or dropped leaves the table advertising a spelling the loader
  now refuses outright --- the reader writes it, and the build fails naming a
  key the documentation told them to use.

The comparison is a set equality rather than a one-way check, since either
half going stale breaks the same promise. The producer is the loader's frozen
set, not the dataclass: the loader is what a profile is actually measured
against, and reading it here keeps this sweep honest if the two ever part.
"""

from __future__ import annotations

import re
from pathlib import Path

from osprey.cli.build_profile_load import _KNOWN_DISPATCH_KEYS

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The page this sweep reads, relative to the repo root.
_PAGE = "docs/source/reference/configuration/profile.rst"

#: The section the key table lives in, delimited by RST anchors. Anchors rather
#: than headings or line numbers: a heading can be reworded and a line number
#: moves whenever anything above it grows, but an anchor is referenced from
#: elsewhere in the docs and cannot be renamed silently.
_SECTION_ANCHOR = ".. _profile-dispatch-block:"

#: A ``list-table`` row's first cell: ``* - `` and the rest of the line. Later
#: cells are written ``- `` at the same indent and are not rows.
_ROW_PATTERN = re.compile(r"^\s*\* - (.*)$")

#: A first cell that is exactly one RST inline literal, which is how the table
#: writes a key. The header row's plain ``Key`` and any prose cell fail it.
_KEY_CELL_PATTERN = re.compile(r"^``([^`\n]+)``$")


def _section(root_dir: Path | None = None) -> list[str]:
    """The lines between the dispatch anchor and the next anchor after it."""
    path = (root_dir if root_dir is not None else _REPO_ROOT) / _PAGE
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    starts = [n for n, line in enumerate(lines) if line.strip() == _SECTION_ANCHOR]
    if not starts:
        return []
    start = starts[0] + 1
    for offset, line in enumerate(lines[start:], start=start):
        if line.startswith(".. _") and line.rstrip().endswith(":"):
            return lines[start:offset]
    return lines[start:]


def _documented_keys(root_dir: Path | None = None) -> set[str]:
    """Every key the dispatch section's table names in a row's first cell."""
    found: set[str] = set()
    for line in _section(root_dir):
        row = _ROW_PATTERN.match(line)
        if row is None:
            continue
        cell = _KEY_CELL_PATTERN.match(row.group(1).strip())
        if cell is not None:
            found.add(cell.group(1))
    return found


def test_the_documented_dispatch_keys_are_the_accepted_ones() -> None:
    """The rule: a closed block's table and its loader name the same set."""
    documented = _documented_keys()
    undocumented = sorted(set(_KNOWN_DISPATCH_KEYS) - documented)
    unaccepted = sorted(documented - set(_KNOWN_DISPATCH_KEYS))
    assert (undocumented, unaccepted) == ([], []), (
        f"The ``dispatch:`` table in {_PAGE} claims to list every key the block "
        "accepts, so it must match the loader's closed set exactly.\n"
        f"Accepted but undocumented: {undocumented}\n"
        f"Documented but refused by the loader: {unaccepted}"
    )


def test_the_section_yields_keys() -> None:
    """Guard against the table moving out from under the anchor.

    A section that parses to nothing would make the equality above fail with a
    confusing message, or pass vacuously if the loader's set ever emptied.
    Pinning the parse says plainly which artefact broke.
    """
    assert _documented_keys(), (
        f"no key rows parsed from the {_SECTION_ANCHOR} section of {_PAGE} — "
        "the anchor or the table shape has moved"
    )


def test_the_section_stops_at_the_next_anchor(tmp_path: Path) -> None:
    """A key row belonging to a later block must not be collected as a dispatch one.

    The two tables on this page are written identically, so the delimiter is
    the only thing separating them; a parser that ran to end-of-file would
    inherit every key on the page and pass no matter what the table said.
    """
    page = tmp_path / _PAGE
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(
        f"{_SECTION_ANCHOR}\n\n"
        ".. list-table::\n\n"
        "   * - ``triggers``\n"
        "     - what it does\n\n"
        ".. _profile-somewhere-else:\n\n"
        ".. list-table::\n\n"
        "   * - ``not_a_dispatch_key``\n"
        "     - what it does\n",
        encoding="utf-8",
    )

    assert _documented_keys(tmp_path) == {"triggers"}
