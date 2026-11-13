"""The relations layer's tables: CSV read into rows, and refusals folded per value.

Every table is UTF-8 CSV with a header row naming its columns in any order. A
byte-order mark is dropped, every cell is stripped, a row with no non-empty
cell is skipped and an empty cell is absent. Four tables, each a
:class:`TableSpec`::

    relations  type, subject, object; subject_kind, object_kind,
               subject_port, object_port, object_unit, attrs, basis, rule,
               subject_class, object_class
    names      thing, name, scheme; basis, rule
    places     place; within, basis, rule
    members    group, member; member_kind

and every table also takes the citation columns ``document``, ``version``,
``state``, ``locator`` and ``imported``. Required columns come first above.

Non-UTF-8 input, a file that does not parse as CSV, an unknown, repeated or
missing required column, a row with more cells than the header, a row with a
required cell empty, a citation column filled with no ``document``, and an
``imported`` that is not an ISO date are refusals. A refusal goes into a
:class:`Refusals`, which every translator of the layer appends to and which
folds them per problem and value: one error per value naming its first three
rows and how many more, and past :data:`FOLD_LIMIT` errors one closing line
with the count of the rest.
"""

from __future__ import annotations

import csv
import datetime
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from osprey.facility.errors import FacilityBuildError, quoted_slots
from osprey.facility.layers import LayerSourceInvalid

__all__ = [
    "CITATION_COLUMNS",
    "FOLD_LIMIT",
    "MEMBERS",
    "NAMES",
    "PLACES",
    "RELATIONS",
    "Refusals",
    "Row",
    "TableSpec",
    "citation",
    "parse_attrs",
    "read_table",
]

#: The citation columns every table takes, in the order a ``Citation`` holds them.
CITATION_COLUMNS: tuple[str, ...] = ("document", "version", "state", "locator", "imported")

#: How many folded errors :meth:`Refusals.errors` lists before its closing line.
FOLD_LIMIT = 20

#: How many row numbers one folded error names before ``and N more``.
_ROWS_NAMED = 3


@dataclass(frozen=True)
class TableSpec:
    """The columns one table may name.

    Attributes:
        name: The table's word in messages (``relations``, ``names``, ...).
        required: The columns the header must name and every row must fill.
        optional: The columns the header may name.
    """

    name: str
    required: tuple[str, ...]
    optional: tuple[str, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        """Every column the header may name, required first."""
        return self.required + self.optional


#: The relations table: one row is one claim.
RELATIONS = TableSpec(
    "relations",
    ("type", "subject", "object"),
    (
        "subject_kind",
        "object_kind",
        "subject_port",
        "object_port",
        "object_unit",
        "attrs",
        "basis",
        "rule",
        *CITATION_COLUMNS,
        "subject_class",
        "object_class",
    ),
)

#: The names table: one row is one name of a thing the relations table uses.
NAMES = TableSpec("names", ("thing", "name", "scheme"), ("basis", "rule", *CITATION_COLUMNS))

#: The places table: one row places one place within another.
PLACES = TableSpec("places", ("place",), ("within", "basis", "rule", *CITATION_COLUMNS))

#: The members table: one row is one member of a group.
MEMBERS = TableSpec("members", ("group", "member"), ("member_kind", *CITATION_COLUMNS))


class Row(NamedTuple):
    """One table row: its 1-based line in the file and its filled cells."""

    line: int
    cells: dict[str, str]


@dataclass
class _Fold:
    record_kind: str
    value: str
    problem: str
    remedy: str
    table: str
    lines: list[int] = field(default_factory=list)


@dataclass
class Refusals:
    """The refusals of one import, folded per problem and value.

    Every translator appends with :meth:`refuse`; the orchestrator raises
    ``LayerSourceInvalid(refusals.errors())`` when any exist.
    """

    _folds: dict[tuple[str, str, str, str], _Fold] = field(default_factory=dict)

    def refuse(
        self,
        table: str,
        line: int | None,
        *,
        record_kind: str,
        value: str,
        problem: str,
        remedy: str,
    ) -> None:
        """Record one refusal.

        Args:
            table: The file name of the table refused.
            line: The row's line number, or ``None`` for the file as a whole.
            record_kind: What ``value`` is (``type``, ``place``, ``path``, ...).
            value: The refused value; refusals of one problem fold per value.
            problem: What is wrong, such as ``unknown relation type``.
            remedy: What the author changes to clear it.
        """
        key = (table, record_kind, value, problem)
        fold = self._folds.get(key)
        if fold is None:
            fold = self._folds[key] = _Fold(record_kind, value, problem, remedy, table)
        if line is not None and line not in fold.lines:
            fold.lines.append(line)

    def __bool__(self) -> bool:
        return bool(self._folds)

    def __len__(self) -> int:
        return len(self._folds)

    def errors(self) -> list[FacilityBuildError]:
        """The folded errors in the order their first refusal came.

        Returns:
            One ``source-invalid`` error per (problem, value), at most
            :data:`FOLD_LIMIT` of them, then one closing error counting the rest.
        """
        folds = list(self._folds.values())
        errors = [_error(fold) for fold in folds[:FOLD_LIMIT]]
        rest = folds[FOLD_LIMIT:]
        if rest:
            tables = sorted({fold.table for fold in rest})
            errors.append(
                FacilityBuildError(
                    "source-invalid",
                    "relations",
                    tables,
                    "clear the refusals above and import again to see the rest",
                    record_kind="import",
                    detail=f"{len(rest)} more refused {'value' if len(rest) == 1 else 'values'}",
                )
            )
        return errors


def _error(fold: _Fold) -> FacilityBuildError:
    return FacilityBuildError(
        "source-invalid",
        fold.value,
        [fold.table],
        fold.remedy,
        record_kind=fold.record_kind,
        detail=_detail(fold),
    )


def _detail(fold: _Fold) -> str:
    lines = sorted(fold.lines)
    if not lines:
        return fold.problem
    named = ", ".join(str(line) for line in lines[:_ROWS_NAMED])
    more = len(lines) - _ROWS_NAMED
    if more > 0:
        named += f" and {more} more"
    word = "row" if len(lines) == 1 else "rows"
    return f"{fold.problem} ({fold.table} {word} {named})"


def read_table(path: Path, spec: TableSpec, refusals: Refusals | None = None) -> list[Row]:
    """Read one table into rows.

    Args:
        path: The CSV file.
        spec: The table's columns.
        refusals: Where refusals go; when omitted, any refusal raises.

    Returns:
        The rows that passed, in file order; a refused row is left out.

    Raises:
        LayerSourceInvalid: ``refusals`` was omitted and the table was refused.
    """
    own = Refusals() if refusals is None else refusals
    rows = _read(path, spec, own)
    if refusals is None and own:
        raise LayerSourceInvalid(own.errors())
    return rows


def _read(path: Path, spec: TableSpec, refusals: Refusals) -> list[Row]:
    name = path.name

    def refuse_file(problem: str, remedy: str) -> None:
        refusals.refuse(name, None, record_kind="path", value=name, problem=problem, remedy=remedy)

    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        refuse_file(f"is not UTF-8: {error.reason}", "save it as UTF-8")
        return []
    try:
        lines = [
            (number, [cell.strip() for cell in cells])
            for number, cells in enumerate(csv.reader(io.StringIO(text, newline="")), start=1)
            if any(cell.strip() for cell in cells)
        ]
    except csv.Error as error:
        refuse_file(f"does not parse as CSV: {error}", "fix the CSV")
        return []
    if not lines:
        refuse_file("holds no header row", f"add a header row naming {quoted_slots(spec.required)}")
        return []
    header = lines[0][1]
    columns = ", ".join(spec.columns)
    problems = [
        (f"unknown column `{column}`", f"remove the column; the columns are {columns}")
        for column in header
        if column not in spec.columns
    ]
    problems += [
        (f"names column `{column}` twice", "keep one")
        for column in sorted({column for column in header if header.count(column) > 1})
    ]
    problems += [
        (f"has no `{column}` column", f"add the column `{column}`")
        for column in spec.required
        if column not in header
    ]
    for problem, remedy in problems:
        refuse_file(problem, remedy)
    if problems:
        return []
    rows: list[Row] = []
    for number, cells in lines[1:]:
        if len(cells) > len(header):
            refusals.refuse(
                name,
                number,
                record_kind="path",
                value=name,
                problem=f"holds more cells than the header names ({len(header)})",
                remedy="fill one cell per header column",
            )
            continue
        filled = {column: cell for column, cell in zip(header, cells, strict=False) if cell}
        if _check(name, number, spec, filled, refusals):
            rows.append(Row(number, filled))
    return rows


def _check(
    name: str, number: int, spec: TableSpec, cells: dict[str, str], refusals: Refusals
) -> bool:
    good = True
    for column in spec.required:
        if column not in cells:
            refusals.refuse(
                name,
                number,
                record_kind="column",
                value=column,
                problem="empty required cell",
                remedy=f"fill `{column}` or remove the row",
            )
            good = False
    cited = [column for column in CITATION_COLUMNS if column in cells]
    if cited and "document" not in cells:
        refusals.refuse(
            name,
            number,
            record_kind="column",
            value="document",
            problem=f"a citation with no document ({quoted_slots(cited)} filled)",
            remedy="fill `document`, or empty the citation columns",
        )
        good = False
    imported = cells.get("imported")
    if imported is not None and not _is_date(imported):
        refusals.refuse(
            name,
            number,
            record_kind="imported",
            value=imported,
            problem="not an ISO date",
            remedy="write `imported` as YYYY-MM-DD",
        )
        good = False
    return good


def _is_date(text: str) -> bool:
    try:
        datetime.date.fromisoformat(text)
    except ValueError:
        return False
    return len(text) == 10


def citation(cells: dict[str, str]) -> dict[str, str] | None:
    """The ``Citation`` map a row's citation columns state.

    Args:
        cells: The row's filled cells.

    Returns:
        ``document``, ``version``, ``state``, ``locator`` and ``imported`` as
        filled, absent ones omitted and ``imported`` kept as written; ``None``
        when the row names no document.
    """
    if "document" not in cells:
        return None
    return {column: cells[column] for column in CITATION_COLUMNS if column in cells}


def parse_attrs(text: str, refusals: Refusals, table: str, line: int) -> dict[str, str] | None:
    """Read an ``attrs`` cell: ``key=value`` pairs separated by ``;``.

    Keys and values are stripped; an empty segment (a trailing ``;``) is
    skipped and a value may hold ``=``.

    Args:
        text: The cell.
        refusals: Where a malformed pair goes.
        table: The table's file name.
        line: The row's line number.

    Returns:
        The pairs in cell order, or ``None`` when a pair is malformed (no
        ``=``, an empty key, or one key given twice).
    """
    attrs: dict[str, str] = {}
    good = True
    for segment in text.split(";"):
        pair = segment.strip()
        if not pair:
            continue
        key, equals, value = pair.partition("=")
        key = key.strip()
        if not equals or not key:
            problem = "an attribute pair with no `=`" if not equals else "an attribute with no key"
        elif key in attrs:
            problem = f"attribute `{key}` given twice"
        else:
            attrs[key] = value.strip()
            continue
        refusals.refuse(
            table,
            line,
            record_kind="attrs",
            value=pair,
            problem=problem,
            remedy="write `attrs` as key=value pairs separated by `;`",
        )
        good = False
    return attrs if good else None
