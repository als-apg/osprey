"""The documented ``bluesky:`` keys are exactly the ones the loader accepts.

``docs/source/reference/configuration/profile.rst`` says the ``bluesky:``
section "accepts exactly <n> keys" and lists them in a hand-written table. The
loader enforces its own closed set,
:data:`~osprey.cli.build_profile_load._KNOWN_BLUESKY_KEYS`, derived from
``BlueskyConfig``. The table and the spelled-out count must both match it, so a
field added to or dropped from the dataclass cannot leave the page behind.
"""

from __future__ import annotations

import re
from pathlib import Path

from osprey.cli.build_profile_load import _KNOWN_BLUESKY_KEYS

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The page this sweep reads, relative to the repo root.
_PAGE = "docs/source/reference/configuration/profile.rst"

#: The section's heading text; the section runs to the next ``=`` heading.
_HEADING = "bluesky"

#: A ``list-table`` row's first cell: ``* - `` and the rest of the line.
_ROW_PATTERN = re.compile(r"^\s*\* - (.*)$")

#: A first cell that is exactly one RST inline literal, which is how the table
#: writes a key.
_KEY_CELL_PATTERN = re.compile(r"^``([^`\n]+)``$")

#: The sentence that spells the section's key count.
_COUNT_PATTERN = re.compile(r"accepts\s+exactly\s+(\w+)\s+keys")

_NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
).split()


def _is_underline(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) == {"="}


def _section(root_dir: Path | None = None) -> list[str]:
    """The lines from the ``bluesky`` heading to the next ``=`` heading."""
    path = (root_dir if root_dir is not None else _REPO_ROOT) / _PAGE
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for n in range(len(lines) - 1):
        if lines[n].strip() == _HEADING and _is_underline(lines[n + 1]):
            start = n + 2
            break
    else:
        return []
    for offset in range(start, len(lines) - 1):
        if lines[offset].strip() and _is_underline(lines[offset + 1]):
            return lines[start:offset]
    return lines[start:]


def _documented_keys(section: list[str]) -> set[str]:
    """Every key the section's table names in a row's first cell."""
    found: set[str] = set()
    for line in section:
        row = _ROW_PATTERN.match(line)
        if row is None:
            continue
        cell = _KEY_CELL_PATTERN.match(row.group(1).strip())
        if cell is not None:
            found.add(cell.group(1))
    return found


def test_the_documented_bluesky_keys_are_the_accepted_ones() -> None:
    """The table names the loader's set and the prose spells its size."""
    section = _section()
    documented = _documented_keys(section)
    assert documented, f"no key rows parsed from the bluesky section of {_PAGE}"
    undocumented = sorted(set(_KNOWN_BLUESKY_KEYS) - documented)
    unaccepted = sorted(documented - set(_KNOWN_BLUESKY_KEYS))
    assert (undocumented, unaccepted) == ([], []), (
        f"The ``bluesky:`` table in {_PAGE} must match the loader's closed set.\n"
        f"Accepted but undocumented: {undocumented}\n"
        f"Documented but refused by the loader: {unaccepted}"
    )
    count = _COUNT_PATTERN.search(" ".join(line.strip() for line in section))
    assert count is not None, f"no 'accepts exactly <n> keys' sentence in {_PAGE}"
    assert count.group(1) == _NUMBER_WORDS[len(_KNOWN_BLUESKY_KEYS)], (
        f"{_PAGE} says the bluesky section accepts {count.group(1)} keys; "
        f"the loader accepts {len(_KNOWN_BLUESKY_KEYS)}"
    )
