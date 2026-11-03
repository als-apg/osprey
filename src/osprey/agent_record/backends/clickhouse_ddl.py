"""The agent record's two ClickHouse tables, rendered, fingerprinted and reconciled.

This module is the only author of the ClickHouse record's schema. Each table is
declared once as a :class:`TableSpec`; :func:`render_create` turns it into the
``CREATE TABLE`` the stage applies, stamped with a ``COMMENT`` that is the
sha256 of the render itself (:func:`fingerprint`).

**The record table** ``entries`` is a plain ``MergeTree``: no deduplicating
engine, no TTL and no person column. Its ``ORDER BY (identity, session_id,
key, chunk_index)`` is permanent: a change of the sorting key, the partition
key or a column's type is a migration, never something a stage applies in
place. ``PARTITION BY toDate(inserted_at)`` makes one partition per arrival
day, so retention drops whole days with ``DROP PARTITION`` and an export reads
one partition. Uniqueness of ``(key, chunk_index)`` is the backend's job (it
checks before it inserts), because no MergeTree enforces it.

**The state table** ``state`` is a ``ReplacingMergeTree(version)`` keyed by
name, read with ``FINAL``, so the row with the highest version is the value.

On every stage the live tables are compared before anything is applied
(:func:`reconcile`). A live table whose comment equals the render's
fingerprint needs nothing. Otherwise the render is created in the scratch
database :data:`SCRATCH_DATABASE`, so both sides of the comparison are
normalized by the store itself, and:

* a different engine, sorting, partition or primary key, a column whose type
  or default differs, or a column the render no longer has is refused
  (:class:`SchemaRefused`);
* a missing column is added, and the comment is rewritten last.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

DATABASE = "agent_record"
#: Where the render is created for comparison, and dropped again.
SCRATCH_DATABASE = "agent_record_shape"

ENTRIES_TABLE = "entries"
STATE_TABLE = "state"

_TIME = "DateTime64(6, 'UTC')"


@dataclass(frozen=True)
class Column:
    """One column: its name and its type."""

    name: str
    type: str

    def render(self) -> str:
        return f"`{self.name}` {self.type}"


@dataclass(frozen=True)
class TableSpec:
    """One table as the agent record declares it."""

    name: str
    columns: tuple[Column, ...]
    engine: str
    order_by: str
    partition_by: str = ""

    def column_names(self) -> list[str]:
        return [column.name for column in self.columns]


ENTRIES = TableSpec(
    name=ENTRIES_TABLE,
    columns=(
        Column("key", "String"),
        Column("chunk_index", "UInt32"),
        Column("chunk_total", "UInt32"),
        Column("kind", "String"),
        Column("ts", _TIME),
        Column("identity", "String"),
        Column("on_behalf_of", "String"),
        Column("session_id", "String"),
        Column("terminal_session", "String"),
        Column("run_id", "String"),
        Column("agent_id", "String"),
        Column("parent_key", "String"),
        Column("tool_use_id", "String"),
        Column("body", "String"),
        Column("inserted_at", _TIME),
    ),
    engine="MergeTree",
    order_by="(identity, session_id, key, chunk_index)",
    partition_by="toDate(inserted_at)",
)

STATE = TableSpec(
    name=STATE_TABLE,
    columns=(
        Column("name", "String"),
        Column("value", "String"),
        Column("version", "UInt64"),
    ),
    engine="ReplacingMergeTree(version)",
    order_by="name",
)

#: Every table, in creation order.
TABLES: tuple[TableSpec, ...] = (ENTRIES, STATE)


def quote(text: str) -> str:
    """*text* as a ClickHouse string literal."""
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def qualified(table: str, database: str = DATABASE) -> str:
    """*table* in *database*, as a quoted ClickHouse identifier."""
    return f"`{database}`.`{table}`"


def _render_body(table: TableSpec, database: str) -> str:
    lines = [column.render() for column in table.columns]
    text = (
        f"CREATE TABLE IF NOT EXISTS {qualified(table.name, database)}\n(\n    "
        + ",\n    ".join(lines)
        + f"\n)\nENGINE = {table.engine}\n"
    )
    if table.partition_by:
        text += f"PARTITION BY {table.partition_by}\n"
    return text + f"ORDER BY {table.order_by}"


def fingerprint(table: TableSpec) -> str:
    """The sha256 of *table*'s normalized render, the value its ``COMMENT`` carries.

    Normalized: rendered against :data:`DATABASE` with every run of whitespace
    collapsed, so neither the target database nor layout moves it.
    """
    normalized = " ".join(_render_body(table, DATABASE).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def render_create(table: TableSpec, database: str = DATABASE) -> str:
    """The ``CREATE TABLE`` for *table* in *database*, fingerprint comment included."""
    return _render_body(table, database) + f"\nCOMMENT '{fingerprint(table)}'"


def render_schema(database: str = DATABASE) -> list[str]:
    """The database and every table, as statements, in creation order."""
    return [
        f"CREATE DATABASE IF NOT EXISTS `{database}`",
        *(render_create(table, database) for table in TABLES),
    ]


# --------------------------------------------------------------------------- reconcile


class SchemaRefused(Exception):
    """A live table differs from the render in a way only a migration may change."""

    def __init__(self, table: str, reason: str) -> None:
        self.table = table
        self.reason = reason
        super().__init__(
            f"agent record table {DATABASE}.{table} differs from this release's schema: "
            f"{reason}. A schema change is migrated by hand: rename the table to "
            f"{table}_pre_<version>, stage again to create it anew, copy the rows with "
            f"INSERT ... SELECT one partition at a time as the store's admin, then drop "
            f"the renamed table."
        )


@dataclass
class TableShape:
    """One table as the store reports it, from ``system.tables`` and ``system.columns``."""

    engine: str = ""
    sorting_key: str = ""
    partition_key: str = ""
    primary_key: str = ""
    comment: str = ""
    #: name -> (type, default_kind, default_expression), in table order.
    columns: dict[str, tuple[str, str, str]] = field(default_factory=dict)


def _engine(engine_full: str) -> str:
    """The engine and its arguments, without the clauses ``engine_full`` appends."""
    for clause in (" PARTITION BY ", " PRIMARY KEY ", " ORDER BY ", " SETTINGS "):
        engine_full = engine_full.split(clause)[0]
    return engine_full


def shapes_from_rows(
    tables: Sequence[Mapping[str, object]],
    columns: Sequence[Mapping[str, object]],
) -> dict[str, TableShape]:
    """Assemble :class:`TableShape` per table from the two ``system.*`` row sets."""
    shapes: dict[str, TableShape] = {}
    for row in tables:
        shapes[str(row["name"])] = TableShape(
            engine=_engine(str(row.get("engine_full", ""))),
            sorting_key=str(row.get("sorting_key", "")),
            partition_key=str(row.get("partition_key", "")),
            primary_key=str(row.get("primary_key", "")),
            comment=str(row.get("comment", "")),
        )
    for row in columns:
        shape = shapes.get(str(row["table"]))
        if shape is not None:
            shape.columns[str(row["name"])] = (
                str(row.get("type", "")),
                str(row.get("default_kind", "")),
                str(row.get("default_expression", "")),
            )
    return shapes


def tables_query(database: str) -> str:
    return (
        "SELECT name, sorting_key, partition_key, primary_key, comment, engine_full "
        f"FROM system.tables WHERE database = '{database}' FORMAT JSONEachRow"
    )


def columns_query(database: str) -> str:
    return (
        "SELECT table, name, type, default_kind, default_expression FROM system.columns "
        f"WHERE database = '{database}' ORDER BY table, position FORMAT JSONEachRow"
    )


def needs_scratch(live: Mapping[str, TableShape]) -> bool:
    """Whether any table is missing or carries a comment other than its fingerprint.

    When this is ``False`` the fingerprint fast path holds and the reconcile
    issues no statement at all.
    """
    return any(
        live.get(table.name) is None or live[table.name].comment != fingerprint(table)
        for table in TABLES
    )


def reconcile(live: Mapping[str, TableShape], scratch: Mapping[str, TableShape]) -> list[str]:
    """The statements that bring the live tables to the render, in apply order.

    *scratch* is the render as the store created it in
    :data:`SCRATCH_DATABASE`; it may be empty when no live table needs a
    comparison (each is either missing or carries its fingerprint).

    Raises:
        SchemaRefused: A live table differs in its engine, a key, a column's
            type or default, or has a column the render no longer has.
    """
    statements: list[str] = []
    for table in TABLES:
        current = live.get(table.name)
        if current is None:
            statements.append(render_create(table))
            continue
        wanted = fingerprint(table)
        if current.comment == wanted:
            continue
        statements.extend(_alter_statements(table, current, scratch[table.name], wanted))
    return statements


def _alter_statements(
    table: TableSpec, live: TableShape, shape: TableShape, wanted: str
) -> list[str]:
    for key in ("engine", "sorting_key", "partition_key", "primary_key"):
        if getattr(live, key) != getattr(shape, key):
            raise SchemaRefused(
                table.name,
                f"its {key.replace('_', ' ')} is {getattr(live, key)!r}, "
                f"this release's is {getattr(shape, key)!r}",
            )
    for name, definition in live.columns.items():
        if name not in shape.columns:
            raise SchemaRefused(table.name, f"column {name} is not in this release's schema")
        if definition != shape.columns[name]:
            raise SchemaRefused(
                table.name,
                f"column {name} is {definition!r}, this release's is {shape.columns[name]!r}",
            )
    target = qualified(table.name)
    rendered = {column.name: column for column in table.columns}
    statements = [
        f"ALTER TABLE {target} ADD COLUMN IF NOT EXISTS {rendered[name].render()}"
        for name in shape.columns
        if name not in live.columns and name in rendered
    ]
    statements.append(f"ALTER TABLE {target} MODIFY COMMENT '{wanted}'")
    return statements
