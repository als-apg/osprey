"""The pages that tell an operator how a facility tree leaves its retired paths state what the code does.

- The ``--revalidate`` entry of ``reference/cli.rst`` tables every retired path
  with its current path, exactly :func:`osprey_facility.source_tree.retired_moves`
  in its order, names the deck-value rewrite, and names the heading the
  hand-done steps print under.
- The refusal note of ``how-to/describe-your-facility.rst`` names the
  alternative command the refusal prints.
- ``how-to/control-systems/machine-repository.rst`` states the upgrade order of
  a shared machine repository, and both other pages link to it.
"""

from __future__ import annotations

import re
from pathlib import Path

from osprey_facility import source_tree

_DOCS = Path(__file__).resolve().parents[2] / "docs" / "source"
_CLI_PAGE = _DOCS / "reference" / "cli.rst"
_FACILITY_PAGE = _DOCS / "how-to" / "describe-your-facility.rst"
_MACHINE_PAGE = _DOCS / "how-to" / "control-systems" / "machine-repository.rst"

#: The label the machine repository page's upgrade section carries.
_UPGRADE_LABEL = "machine-repository-upgrade"
#: The label of the ``--revalidate`` entry.
_REVALIDATE_LABEL = "cli-osprey-init-revalidate"
#: One body row of the retired-path simple table: two inline literals.
_ROW = re.compile(r"^\s*``([^`]+)``\s+``([^`]+)``\s*$")


def _revalidate_entry() -> str:
    """The ``--revalidate`` entry, from its label up to the next option."""
    text = _CLI_PAGE.read_text(encoding="utf-8")
    start = text.index(f".. _{_REVALIDATE_LABEL}:")
    end = text.index("``--up``", start)
    return text[start:end]


def _table_moves(entry: str) -> list[tuple[str, str]]:
    """Every (retired, current) row the entry's table holds, trailing ``/`` dropped."""
    return [
        (m.group(1).rstrip("/"), m.group(2).rstrip("/"))
        for line in entry.splitlines()
        if (m := _ROW.match(line))
    ]


def _code_moves() -> list[tuple[str, str]]:
    return [(old, new) for old, new, _content in source_tree.retired_moves()]


def test_revalidate_entry_tables_every_retired_path() -> None:
    assert _table_moves(_revalidate_entry()) == _code_moves()


def test_table_reader_sees_a_dropped_row() -> None:
    """Negative control: a row missing from the page is a difference."""
    entry = _revalidate_entry()
    old, new = _code_moves()[0]
    pruned = "\n".join(
        line for line in entry.splitlines() if not line.strip().startswith(f"``{old}")
    )
    assert _table_moves(pruned) != _code_moves()


def test_revalidate_entry_names_the_deck_rewrite_and_the_hand_steps() -> None:
    entry = _revalidate_entry()
    old_prefix, new_prefix = source_tree.retired_deck_prefix()
    assert f"``{old_prefix}``" in entry
    assert f"``{new_prefix}``" in entry
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
