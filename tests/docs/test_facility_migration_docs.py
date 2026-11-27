"""The pages that tell an operator how a repository leaves its retired paths state what the code does.

- ``how-to/build-profiles.rst`` holds the one retired-to-current map: every
  move of :func:`osprey.repo_layout.retired_moves` and, under its facility
  marker row, every move of :func:`osprey_facility.source_tree.retired_moves`,
  and nothing else. It names the deck-value rewrite.
- The ``--revalidate`` entry of ``reference/cli.rst`` links to that map and
  names the heading the hand-done steps print under.
- The refusal note of ``how-to/describe-your-facility.rst`` names the
  alternative command the refusal prints.
- ``how-to/control-systems/machine-repository.rst`` states the upgrade order of
  a shared machine repository, and both other pages link to it.
"""

from __future__ import annotations

import re
from pathlib import Path

from osprey import repo_layout
from osprey_facility import source_tree

_DOCS = Path(__file__).resolve().parents[2] / "docs" / "source"
_CLI_PAGE = _DOCS / "reference" / "cli.rst"
_PROFILES_PAGE = _DOCS / "how-to" / "build-profiles.rst"
_FACILITY_PAGE = _DOCS / "how-to" / "describe-your-facility.rst"
_MACHINE_PAGE = _DOCS / "how-to" / "control-systems" / "machine-repository.rst"

#: The label the machine repository page's upgrade section carries.
_UPGRADE_LABEL = "machine-repository-upgrade"
#: The label of the ``--revalidate`` entry.
_REVALIDATE_LABEL = "cli-osprey-init-revalidate"
#: The label of the section that holds the map.
_MAP_LABEL = "profile-root-migration"
#: The row after which the map's rows are relative to a facility tree.
_FACILITY_MARKER = "*Inside a facility tree:*"
#: A current-path cell that keeps each retired name under one directory.
_SAME_NAME = re.compile(r"^the same name under ``([^`]+)``$")
#: The per-lane certificate directories, which the map writes as one row.
_LANE = re.compile(r"^(data|var/runtime)/\w+_curve$")

Moves = set[tuple[str, str]]


def _map_section() -> str:
    """The map's section, from its label up to the first paragraph after the table."""
    text = _PROFILES_PAGE.read_text(encoding="utf-8")
    start = text.index(f".. _{_MAP_LABEL}:")
    return text[start : text.index("Every command that reads the sources", start)]


def _revalidate_entry() -> str:
    """The ``--revalidate`` entry, from its label up to the next option."""
    text = _CLI_PAGE.read_text(encoding="utf-8")
    start = text.index(f".. _{_REVALIDATE_LABEL}:")
    return text[start : text.index("``--up``", start)]


def _rows(section: str) -> list[tuple[str, str]]:
    """Every body row of the section's list-table as (retired cell, current cell)."""
    cells: list[list[str]] = []
    for line in section.splitlines():
        if line.startswith("   * - "):
            cells.append([line[7:].strip(), ""])
        elif line.startswith("     - ") or line == "     -":
            cells[-1][1] = line[6:].strip()
        elif line.startswith("       ") and cells and not cells[-1][1]:
            cells[-1][0] += " " + line.strip()
    return [(old, new) for old, new in cells[1:]]


def _table_moves(section: str) -> tuple[Moves, Moves]:
    """The (root, facility) moves the map states, trailing ``/`` dropped."""
    root: Moves = set()
    facility: Moves = set()
    target = root
    for old_cell, new_cell in _rows(section):
        if old_cell == _FACILITY_MARKER:
            target = facility
            continue
        same = _SAME_NAME.match(new_cell)
        for old in re.findall(r"``([^`]+)``", old_cell):
            old = old.rstrip("/")
            name = old.rsplit("/", 1)[-1]
            new = f"{same.group(1)}{name}" if same else new_cell.strip("`").rstrip("/")
            target.add((old, new))
    return root, facility


def _lane(path: str) -> str:
    """*path* with a lane's certificate directory written as the map writes it."""
    return re.sub(r"\w+_curve$", "<lane>_curve", path) if _LANE.match(path) else path


def _code_moves() -> tuple[Moves, Moves]:
    root = {(_lane(move.old), _lane(move.new)) for move in repo_layout.retired_moves()}
    facility = {(old, new) for old, new, _content in source_tree.retired_moves()}
    return root, facility


def test_the_map_states_every_retired_path_and_no_other() -> None:
    assert _table_moves(_map_section()) == _code_moves()


def test_the_map_is_the_only_one() -> None:
    """No other page tables a retired path beside its current one."""
    old, new = next(iter(sorted(_code_moves()[1])))
    row = re.compile(rf"``{re.escape(old)}/?``.*``{re.escape(new)}/?``")
    holders = [
        page.relative_to(_DOCS).as_posix()
        for page in sorted(_DOCS.rglob("*.rst"))
        if row.search(page.read_text(encoding="utf-8"))
    ]
    assert holders == []


def test_map_reader_sees_a_dropped_path() -> None:
    """Negative control: a path missing from the page is a difference."""
    section = _map_section()
    for moves in _code_moves():
        old, _new = sorted(moves)[0]
        pruned = re.sub(rf"``{re.escape(old)}/?``,?\s*", "", section, count=1)
        assert pruned != section
        assert _table_moves(pruned) != _code_moves()


def test_the_map_names_the_deck_rewrite() -> None:
    section = _map_section()
    old_prefix, new_prefix = source_tree.retired_deck_prefix()
    assert f"``{old_prefix}``" in section
    assert f"``{new_prefix}``" in section


def test_revalidate_entry_links_the_map_and_names_the_hand_steps() -> None:
    entry = _revalidate_entry()
    assert f":ref:`{_MAP_LABEL}`" in entry
    assert '"Left for you to do by hand"' in entry
    assert f"<{_UPGRADE_LABEL}>`" in entry


def test_refusal_note_names_revalidate() -> None:
    text = _FACILITY_PAGE.read_text(encoding="utf-8")
    assert "osprey init --revalidate" in text
    assert f"<{_REVALIDATE_LABEL}>`" in text


def test_machine_page_states_the_upgrade_order() -> None:
    text = _MACHINE_PAGE.read_text(encoding="utf-8")
    start = text.index(f".. _{_UPGRADE_LABEL}:")
    section = text[start:]
    assert "every deployment that mounts" in section
    assert "osprey init --revalidate" in section
