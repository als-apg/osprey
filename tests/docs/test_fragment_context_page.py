"""The contract pages that enumerate a surface the code owns, held equal to it.

``reference/contracts/facility-kinds.rst`` names the measurement-kind
declaration's fields, the roles the pyAML view ships with their sorts, the
member sorts, the entry-point group and the built-in kinds. Each is read here
from :mod:`osprey.facility.kinds`, so a field, role, sort or built-in kind added
to or dropped from the code without the page following fails.
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

from osprey.facility import kinds

DOCS_SOURCE = Path(__file__).resolve().parents[2] / "docs" / "source"
KINDS_PAGE = DOCS_SOURCE / "reference" / "contracts" / "facility-kinds.rst"

_LITERAL = re.compile(r"``([^`]+)``")


def _section(text: str, heading: str) -> str:
    """The body of the section titled ``heading``, up to the next section title."""
    lines = text.splitlines()
    start = lines.index(heading) + 2
    end = next(
        (i - 1 for i in range(start + 1, len(lines)) if set(lines[i]) == {"="}),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _table_rows(section: str) -> list[list[str]]:
    """The rows of the section's first list-table, each its cells' first literals."""
    rows: list[list[str]] = []
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("* - "):
            rows.append([])
            stripped = stripped[2:]
        if stripped.startswith("- ") and rows:
            found = _LITERAL.findall(stripped)
            rows[-1].append(found[0] if found else stripped[2:])
    return rows[1:]


def _page() -> str:
    return KINDS_PAGE.read_text(encoding="utf-8")


def test_the_page_lists_every_declaration_field() -> None:
    rows = _table_rows(_section(_page(), "The declaration"))

    named = [row[0] for row in rows]

    assert len(named) == len(set(named))
    assert set(named) == {f.name for f in fields(kinds.KindDeclaration)}


def test_the_page_lists_the_shipped_roles_with_their_sorts() -> None:
    rows = _table_rows(_section(_page(), "Member roles"))

    assert {row[0]: row[1] for row in rows} == dict(kinds.SHIPPED_SORTS)
    assert all(row[2] == ("instruments" if row[1] == "instrument" else "groups") for row in rows)


def test_the_page_names_the_member_sorts() -> None:
    field_rows = {row[0]: row for row in _table_rows(_section(_page(), "The declaration"))}
    sorts_cell = _page().split("* - ``member_sorts``", 1)[1].split("* - ", 1)[0]

    assert "member_sorts" in field_rows
    assert [w for w in _LITERAL.findall(sorts_cell) if w in kinds.MEMBER_SORTS] == list(
        kinds.MEMBER_SORTS
    )


def test_the_page_names_the_entry_point_group_and_the_built_in_kinds() -> None:
    registering = _section(_page(), "Registering a kind")
    sentence = registering.split("*built-in* kinds:", 1)[1].split(".", 1)[0]

    assert f"``{kinds.ENTRY_POINT_GROUP}``" in registering
    assert f"``{kinds.BUILTIN_DISTRIBUTION}``" in registering
    named = _LITERAL.findall(sentence)

    assert len(named) == len(set(named))
    assert set(named) == set(kinds.builtin_kinds())
