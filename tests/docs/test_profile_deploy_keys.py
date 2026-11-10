"""The profile page's ``deploy`` row names exactly the deploy keys the loader accepts.

``docs/source/reference/configuration/profile.rst`` lists the ``deploy`` block's
keys in the first sentence of that row's description cell, and the loader
refuses any other key. The sentence is read up to its first full stop after
``run:``, so the row's later literals (a command, a sibling block) are not keys.
"""

from __future__ import annotations

import re
from pathlib import Path

from osprey.cli.build_profile_deploy import _KNOWN_DEPLOY_KEYS

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The page this test reads, relative to the repo root.
_PAGE = "docs/source/reference/configuration/profile.rst"

#: A ``list-table`` row: ``* - `` and the rest of the line.
_ROW_PATTERN = re.compile(r"^\s*\* - (.*)$")

#: One RST inline literal.
_LITERAL_PATTERN = re.compile(r"``([^`\n]+)``")


def _deploy_description() -> str:
    """The ``deploy`` row's description cell, joined into one line."""
    lines = (_REPO_ROOT / _PAGE).read_text(encoding="utf-8").splitlines()
    for n, line in enumerate(lines):
        row = _ROW_PATTERN.match(line)
        if row is None or row.group(1).strip() != "``deploy``":
            continue
        cells: list[list[str]] = []
        for following in lines[n + 1 :]:
            if _ROW_PATTERN.match(following) or (
                following.strip() and not following.startswith(" ")
            ):
                break
            cell = re.match(r"^\s*- (.*)$", following)
            if cell is not None:
                cells.append([cell.group(1)])
            elif cells:
                cells[-1].append(following.strip())
        return " ".join(cells[-1]) if cells else ""
    return ""


def _documented_deploy_keys() -> set[str]:
    """The literals of the description's first sentence, the one ending after ``run:``."""
    text = _deploy_description()
    match = re.search(r"^(.*?run:.*?\.)(?:\s|$)", text)
    assert match is not None, f"no 'run:' sentence in the deploy row of {_PAGE}"
    return set(_LITERAL_PATTERN.findall(match.group(1)))


def test_the_deploy_row_names_the_accepted_keys() -> None:
    """The row and ``_KNOWN_DEPLOY_KEYS`` name the same keys, in both directions."""
    documented = _documented_deploy_keys()
    undocumented = sorted(_KNOWN_DEPLOY_KEYS - documented)
    unaccepted = sorted(documented - _KNOWN_DEPLOY_KEYS)
    assert (undocumented, unaccepted) == ([], []), (
        f"The deploy row of {_PAGE} must name the loader's deploy keys.\n"
        f"Accepted but undocumented: {undocumented}\n"
        f"Documented but not accepted: {unaccepted}"
    )
