"""Every table of the telemetry store, rendered, fingerprinted and reconciled.

The gateway's exporters write with ``create_schema: false``, so this module is
the only author of the schema. Each table is declared once as a
:class:`TableSpec`; :func:`render_create` turns it into the ``CREATE TABLE``
``osprey up`` applies, stamped with a ``COMMENT`` that is the sha256 of the
render itself (:func:`fingerprint`).

On every ``osprey up`` the stage compares before it applies
(:func:`reconcile`). A live table whose comment equals the render's
fingerprint needs nothing. Otherwise the render is created in the scratch
database :data:`SCRATCH_DATABASE`, so both sides of the comparison are
normalized by the store itself, and:

* a different sorting, partition or primary key, a column whose type or
  default expression differs, a column the render no longer has, or an index
  whose definition changed is refused (:class:`SchemaRefused`): that is a
  schema change, migrated by hand with the rename-and-copy steps the deploy
  docs give;
* anything else is applied in place: a missing column is added, an index this
  run adds is added and materialized (an existing one never is, since
  ``MATERIALIZE INDEX`` rewrites every part), a changed TTL or setting is
  modified, and the comment is rewritten last.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

DATABASE = "telemetry"
#: Where the render is created for comparison, and dropped again.
SCRATCH_DATABASE = "telemetry_shape"

#: The column every table's TTL, partitions, reader row policies and clamp read.
TIME_COLUMNS: Mapping[str, str] = {
    "otel_logs": "Timestamp",
    "otel_traces": "Timestamp",
    "audit_ledger": "Timestamp",
    "otel_metrics_gauge": "TimeUnix",
    "otel_metrics_sum": "TimeUnix",
    "otel_metrics_histogram": "TimeUnix",
    "otel_metrics_exponential_histogram": "TimeUnix",
    "otel_metrics_summary": "TimeUnix",
}

#: Which table each exporter instance of the gateway writes, per signal. The
#: gateway renderer names its exporters' tables from here, and the column
#: contract test checks every pair against the exporter's insert statements.
EXPORTER_TABLES: Mapping[str, Mapping[str, str]] = {
    "clickhouse/otlp": {
        "logs": "otel_logs",
        "traces": "otel_traces",
        "metrics_gauge": "otel_metrics_gauge",
        "metrics_sum": "otel_metrics_sum",
        "metrics_histogram": "otel_metrics_histogram",
        "metrics_exponential_histogram": "otel_metrics_exponential_histogram",
        "metrics_summary": "otel_metrics_summary",
    },
    "clickhouse/ledger": {"logs": "audit_ledger"},
}

#: Every table carries these, besides its own index granularity.
TABLE_SETTINGS: Mapping[str, str] = {
    "index_granularity": "8192",
    "ttl_only_drop_parts": "1",
    "materialize_ttl_recalculate_only": "1",
    "non_replicated_deduplication_window": "1000",
}

_ZSTD = "CODEC(ZSTD(1))"


@dataclass(frozen=True)
class Column:
    """One column: its name and everything after the name in its definition."""

    name: str
    definition: str

    def render(self) -> str:
        return f"`{self.name}` {self.definition}"


@dataclass(frozen=True)
class Index:
    """One data-skipping index."""

    name: str
    expression: str
    kind: str
    granularity: int = 1

    def render(self) -> str:
        return (
            f"INDEX {self.name} {self.expression} TYPE {self.kind} GRANULARITY {self.granularity}"
        )


@dataclass(frozen=True)
class TableSpec:
    """One table as OSPREY declares it."""

    name: str
    columns: tuple[Column, ...]
    indexes: tuple[Index, ...]
    order_by: str
    partition_by: str

    @property
    def time_column(self) -> str:
        return TIME_COLUMNS[self.name]

    def column_names(self) -> list[str]:
        """Every insertable column name, a ``Nested`` one expanded to ``Name.Field``."""
        names: list[str] = []
        for column in self.columns:
            if " MATERIALIZED " in f" {column.definition} ":
                continue
            nested = re.match(r"Nested\s*\((.*)\)\s*(CODEC.*)?$", column.definition, re.S)
            if nested:
                names.extend(
                    f"{column.name}.{field_def.split()[0]}"
                    for field_def in _split_top_level(nested.group(1))
                )
            else:
                names.append(column.name)
        return names


def _split_top_level(text: str) -> list[str]:
    """*text* split on the commas that sit outside every parenthesis."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if "".join(current).strip():
        parts.append("".join(current).strip())
    return parts


def _c(name: str, definition: str) -> Column:
    return Column(name, definition)


#: The exporter's fifteen base log columns (``logs_insert.sql``), shared by
#: ``otel_logs`` and ``audit_ledger``. No ``EventName``: the exporter detects
#: that column once at start, so a gateway started while the store was down
#: would skip it; the gateway copies the name into ``LogAttributes`` instead.
_LOG_BASE_COLUMNS: tuple[Column, ...] = (
    _c("Timestamp", "DateTime64(9) CODEC(Delta(8), ZSTD(1))"),
    _c("TraceId", f"String {_ZSTD}"),
    _c("SpanId", f"String {_ZSTD}"),
    _c("TraceFlags", "UInt8"),
    _c("SeverityText", f"LowCardinality(String) {_ZSTD}"),
    _c("SeverityNumber", "UInt8"),
    _c("ServiceName", f"LowCardinality(String) {_ZSTD}"),
    _c("Body", f"String {_ZSTD}"),
    _c("ResourceSchemaUrl", f"LowCardinality(String) {_ZSTD}"),
    _c("ResourceAttributes", f"Map(LowCardinality(String), String) {_ZSTD}"),
    _c("ScopeSchemaUrl", f"LowCardinality(String) {_ZSTD}"),
    _c("ScopeName", f"String {_ZSTD}"),
    _c("ScopeVersion", f"LowCardinality(String) {_ZSTD}"),
    _c("ScopeAttributes", f"Map(LowCardinality(String), String) {_ZSTD}"),
    _c("LogAttributes", f"Map(LowCardinality(String), String) {_ZSTD}"),
)

_USER = _c("User", "LowCardinality(String) MATERIALIZED ResourceAttributes['osprey.user']")
_INSERTED_AT = _c("InsertedAt", "DateTime DEFAULT now()")
_IDX_INSERTED = Index("idx_inserted", "InsertedAt", "minmax")

OTEL_LOGS = TableSpec(
    name="otel_logs",
    columns=(
        *_LOG_BASE_COLUMNS,
        _USER,
        _c("SessionId", "String MATERIALIZED LogAttributes['session.id']"),
        _c("TerminalSession", "String MATERIALIZED ResourceAttributes['terminal.session']"),
        _c("Event", "LowCardinality(String) MATERIALIZED LogAttributes['event.name']"),
        _INSERTED_AT,
    ),
    indexes=(
        Index("idx_session", "SessionId", "bloom_filter"),
        Index("idx_event", "Event", "set(64)"),
        _IDX_INSERTED,
    ),
    # User and Event are both low-cardinality, so the session list (prompt
    # events only) prunes to prompt granules by key analysis alone.
    order_by="(User, Event, SessionId, Timestamp)",
    partition_by="toDate(Timestamp)",
)

AUDIT_LEDGER = TableSpec(
    name="audit_ledger",
    columns=(
        *_LOG_BASE_COLUMNS,
        _USER,
        _c("SessionId", "String MATERIALIZED LogAttributes['osprey.audit.session_id']"),
        _c("ToolUseId", "String MATERIALIZED LogAttributes['osprey.audit.tool_use_id']"),
        _c("TerminalSession", "String MATERIALIZED LogAttributes['osprey.audit.session']"),
        _c("Surface", "LowCardinality(String) MATERIALIZED LogAttributes['osprey.audit.surface']"),
        _c("OnBehalfOf", "String MATERIALIZED LogAttributes['osprey.audit.on_behalf_of']"),
        _c("Actor", "LowCardinality(String) MATERIALIZED LogAttributes['osprey.audit.actor']"),
        _c(
            "Decision",
            "LowCardinality(String) MATERIALIZED LogAttributes['osprey.audit.decision']",
        ),
        _c(
            "DurationMs",
            "Nullable(Float64) MATERIALIZED "
            "toFloat64OrNull(LogAttributes['osprey.audit.duration_ms'])",
        ),
        # The gateway's markers for a line it could not read whole.
        _c("Incomplete", "Bool MATERIALIZED LogAttributes['osprey.audit.incomplete'] = 'true'"),
        _c("Truncated", "Bool MATERIALIZED LogAttributes['osprey.audit.truncated'] = 'true'"),
        _c("Unparsed", "Bool MATERIALIZED LogAttributes['osprey.audit.unparsed'] = 'true'"),
        _c(
            "TimeUnparsed",
            "Bool MATERIALIZED LogAttributes['osprey.audit.time_unparsed'] = 'true'",
        ),
        _INSERTED_AT,
    ),
    indexes=(Index("idx_tool", "ToolUseId", "bloom_filter"), _IDX_INSERTED),
    order_by="(User, SessionId, Timestamp)",
    partition_by="toDate(Timestamp)",
)

_ATTRS = f"Map(LowCardinality(String), String) {_ZSTD}"

OTEL_TRACES = TableSpec(
    name="otel_traces",
    columns=(
        _c("Timestamp", "DateTime64(9) CODEC(Delta, ZSTD(1))"),
        _c("TraceId", f"String {_ZSTD}"),
        _c("SpanId", f"String {_ZSTD}"),
        _c("ParentSpanId", f"String {_ZSTD}"),
        _c("TraceState", f"String {_ZSTD}"),
        _c("SpanName", f"LowCardinality(String) {_ZSTD}"),
        _c("SpanKind", f"LowCardinality(String) {_ZSTD}"),
        _c("ServiceName", f"LowCardinality(String) {_ZSTD}"),
        _c("ResourceAttributes", _ATTRS),
        _c("ScopeName", f"String {_ZSTD}"),
        _c("ScopeVersion", f"String {_ZSTD}"),
        _c("SpanAttributes", _ATTRS),
        _c("Duration", f"UInt64 {_ZSTD}"),
        _c("StatusCode", f"LowCardinality(String) {_ZSTD}"),
        _c("StatusMessage", f"String {_ZSTD}"),
        _c(
            "Events",
            "Nested(Timestamp DateTime64(9), Name LowCardinality(String), "
            f"Attributes Map(LowCardinality(String), String)) {_ZSTD}",
        ),
        _c(
            "Links",
            "Nested(TraceId String, SpanId String, TraceState String, "
            f"Attributes Map(LowCardinality(String), String)) {_ZSTD}",
        ),
        _USER,
        _c("SessionId", "String MATERIALIZED SpanAttributes['session.id']"),
        _INSERTED_AT,
    ),
    indexes=(
        Index("idx_trace_id", "TraceId", "bloom_filter(0.001)"),
        Index("idx_res_attr_key", "mapKeys(ResourceAttributes)", "bloom_filter(0.01)"),
        Index("idx_res_attr_value", "mapValues(ResourceAttributes)", "bloom_filter(0.01)"),
        Index("idx_span_attr_key", "mapKeys(SpanAttributes)", "bloom_filter(0.01)"),
        Index("idx_span_attr_value", "mapValues(SpanAttributes)", "bloom_filter(0.01)"),
        Index("idx_duration", "Duration", "minmax"),
        _IDX_INSERTED,
    ),
    order_by="(ServiceName, SpanName, toDateTime(Timestamp))",
    partition_by="toDate(Timestamp)",
)

_METRIC_HEAD: tuple[Column, ...] = (
    _c("ResourceAttributes", _ATTRS),
    _c("ResourceSchemaUrl", f"String {_ZSTD}"),
    _c("ScopeName", f"String {_ZSTD}"),
    _c("ScopeVersion", f"String {_ZSTD}"),
    _c("ScopeAttributes", _ATTRS),
    _c("ScopeDroppedAttrCount", f"UInt32 {_ZSTD}"),
    _c("ScopeSchemaUrl", f"String {_ZSTD}"),
    _c("ServiceName", f"LowCardinality(String) {_ZSTD}"),
    _c("MetricName", f"LowCardinality(String) {_ZSTD}"),
    _c("MetricDescription", f"String {_ZSTD}"),
    _c("MetricUnit", f"String {_ZSTD}"),
    _c("Attributes", _ATTRS),
    _c("StartTimeUnix", "DateTime CODEC(Delta, ZSTD(1))"),
    _c("TimeUnix", "DateTime CODEC(Delta, ZSTD(1))"),
)
_EXEMPLARS = _c(
    "Exemplars",
    "Nested(FilteredAttributes Map(LowCardinality(String), String), TimeUnix DateTime, "
    f"Value Float64, SpanId String, TraceId String) {_ZSTD}",
)
_METRIC_INDEXES: tuple[Index, ...] = (
    Index("idx_res_attr_key", "mapKeys(ResourceAttributes)", "bloom_filter(0.01)"),
    Index("idx_res_attr_value", "mapValues(ResourceAttributes)", "bloom_filter(0.01)"),
    Index("idx_scope_attr_key", "mapKeys(ScopeAttributes)", "bloom_filter(0.01)"),
    Index("idx_scope_attr_value", "mapValues(ScopeAttributes)", "bloom_filter(0.01)"),
    Index("idx_attr_key", "mapKeys(Attributes)", "bloom_filter(0.01)"),
    Index("idx_attr_value", "mapValues(Attributes)", "bloom_filter(0.01)"),
    Index("idx_time_minmax", "TimeUnix", "minmax"),
    _IDX_INSERTED,
)
_METRIC_ORDER = (
    "(ServiceName, MetricName, toStartOfHour(TimeUnix), cityHash64(Attributes), TimeUnix)"
)


def _metric(name: str, *columns: Column) -> TableSpec:
    return TableSpec(
        name=name,
        columns=(*_METRIC_HEAD, *columns, _INSERTED_AT),
        indexes=_METRIC_INDEXES,
        order_by=_METRIC_ORDER,
        partition_by="toDate(TimeUnix)",
    )


_FLAGS = _c("Flags", f"UInt32 {_ZSTD}")
_COUNT = _c("Count", "UInt64 CODEC(Delta, ZSTD(1))")
_SUM = _c("Sum", f"Float64 {_ZSTD}")
_MIN = _c("Min", f"Float64 {_ZSTD}")
_MAX = _c("Max", f"Float64 {_ZSTD}")
_TEMPORALITY = _c("AggregationTemporality", f"Int32 {_ZSTD}")

OTEL_METRICS_GAUGE = _metric(
    "otel_metrics_gauge", _c("Value", f"Float64 {_ZSTD}"), _FLAGS, _EXEMPLARS
)
OTEL_METRICS_SUM = _metric(
    "otel_metrics_sum",
    _c("Value", f"Float64 {_ZSTD}"),
    _FLAGS,
    _EXEMPLARS,
    _TEMPORALITY,
    _c("IsMonotonic", "Boolean CODEC(Delta, ZSTD(1))"),
)
OTEL_METRICS_HISTOGRAM = _metric(
    "otel_metrics_histogram",
    _COUNT,
    _SUM,
    _c("BucketCounts", f"Array(UInt64) {_ZSTD}"),
    _c("ExplicitBounds", f"Array(Float64) {_ZSTD}"),
    _EXEMPLARS,
    _FLAGS,
    _MIN,
    _MAX,
    _TEMPORALITY,
)
OTEL_METRICS_EXPONENTIAL_HISTOGRAM = _metric(
    "otel_metrics_exponential_histogram",
    _COUNT,
    _SUM,
    _c("Scale", f"Int32 {_ZSTD}"),
    _c("ZeroCount", f"UInt64 {_ZSTD}"),
    _c("PositiveOffset", f"Int32 {_ZSTD}"),
    _c("PositiveBucketCounts", f"Array(UInt64) {_ZSTD}"),
    _c("NegativeOffset", f"Int32 {_ZSTD}"),
    _c("NegativeBucketCounts", f"Array(UInt64) {_ZSTD}"),
    _EXEMPLARS,
    _FLAGS,
    _MIN,
    _MAX,
    _TEMPORALITY,
)
OTEL_METRICS_SUMMARY = _metric(
    "otel_metrics_summary",
    _COUNT,
    _SUM,
    _c("ValueAtQuantiles", f"Nested(Quantile Float64, Value Float64) {_ZSTD}"),
    _FLAGS,
)

#: Every table, in the order it is created.
TABLES: tuple[TableSpec, ...] = (
    OTEL_LOGS,
    OTEL_TRACES,
    AUDIT_LEDGER,
    OTEL_METRICS_GAUGE,
    OTEL_METRICS_SUM,
    OTEL_METRICS_HISTOGRAM,
    OTEL_METRICS_EXPONENTIAL_HISTOGRAM,
    OTEL_METRICS_SUMMARY,
)
TABLES_BY_NAME: Mapping[str, TableSpec] = {table.name: table for table in TABLES}


def ttl_clause(table: TableSpec, retention_days: int) -> str:
    """The TTL expression of *table* (the probe-verified form)."""
    return f"toDateTime({table.time_column}) + INTERVAL {int(retention_days)} DAY"


def _render_body(table: TableSpec, retention_days: int, database: str) -> str:
    lines = [*(column.render() for column in table.columns), *(i.render() for i in table.indexes)]
    settings = ", ".join(f"{key} = {value}" for key, value in TABLE_SETTINGS.items())
    return (
        f"CREATE TABLE IF NOT EXISTS `{database}`.`{table.name}`\n(\n    "
        + ",\n    ".join(lines)
        + "\n)\nENGINE = MergeTree\n"
        f"PARTITION BY {table.partition_by}\n"
        f"ORDER BY {table.order_by}\n"
        f"TTL {ttl_clause(table, retention_days)}\n"
        f"SETTINGS {settings}"
    )


def fingerprint(table: TableSpec, retention_days: int) -> str:
    """The sha256 of *table*'s normalized render, the value its ``COMMENT`` carries.

    Normalized: rendered against :data:`DATABASE` with every run of whitespace
    collapsed, so neither the target database nor layout moves it.
    """
    normalized = " ".join(_render_body(table, retention_days, DATABASE).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def render_create(table: TableSpec, retention_days: int, database: str = DATABASE) -> str:
    """The ``CREATE TABLE`` for *table* in *database*, fingerprint comment included."""
    return (
        _render_body(table, retention_days, database)
        + f"\nCOMMENT '{fingerprint(table, retention_days)}'"
    )


def render_schema(retention_days: int, database: str = DATABASE) -> list[str]:
    """The database and every table, as statements, in creation order."""
    return [
        f"CREATE DATABASE IF NOT EXISTS `{database}`",
        *(render_create(table, retention_days, database) for table in TABLES),
    ]


# --------------------------------------------------------------------------- reconcile


class SchemaRefused(Exception):
    """A live table differs from the render in a way only a migration may change."""

    def __init__(self, table: str, reason: str) -> None:
        self.table = table
        self.reason = reason
        super().__init__(
            f"telemetry store table {DATABASE}.{table} differs from this release's schema: "
            f"{reason}. A schema change is migrated by hand: rename the table to "
            f"{table}_pre_<version>, run `osprey up` to create it anew, copy the rows "
            f"with INSERT ... SELECT one partition at a time as the store's admin, then "
            f"drop the renamed table (deploy docs, telemetry store section)."
        )


@dataclass
class TableShape:
    """One table as the store reports it, from ``system.tables``, ``.columns``
    and ``.data_skipping_indices``."""

    sorting_key: str = ""
    partition_key: str = ""
    primary_key: str = ""
    comment: str = ""
    engine_full: str = ""
    #: name -> (type, default_kind, default_expression), in table order.
    columns: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    #: name -> (expr, type_full, granularity).
    indexes: dict[str, tuple[str, str, int]] = field(default_factory=dict)

    @property
    def ttl(self) -> str:
        match = re.search(r"\bTTL (.*?)(?: SETTINGS |$)", self.engine_full)
        return match.group(1).strip() if match else ""

    @property
    def settings(self) -> dict[str, str]:
        _, sep, tail = self.engine_full.partition(" SETTINGS ")
        if not sep:
            return {}
        pairs = (item.partition("=") for item in _split_top_level(tail))
        return {key.strip(): value.strip() for key, _eq, value in pairs}


def shapes_from_rows(
    tables: Sequence[Mapping[str, object]],
    columns: Sequence[Mapping[str, object]],
    indexes: Sequence[Mapping[str, object]],
) -> dict[str, TableShape]:
    """Assemble :class:`TableShape` per table from the three ``system.*`` row sets."""
    shapes: dict[str, TableShape] = {}
    for row in tables:
        shapes[str(row["name"])] = TableShape(
            sorting_key=str(row.get("sorting_key", "")),
            partition_key=str(row.get("partition_key", "")),
            primary_key=str(row.get("primary_key", "")),
            comment=str(row.get("comment", "")),
            engine_full=str(row.get("engine_full", "")),
        )
    for row in columns:
        shape = shapes.get(str(row["table"]))
        if shape is not None:
            shape.columns[str(row["name"])] = (
                str(row.get("type", "")),
                str(row.get("default_kind", "")),
                str(row.get("default_expression", "")),
            )
    for row in indexes:
        shape = shapes.get(str(row["table"]))
        if shape is not None:
            shape.indexes[str(row["name"])] = (
                str(row.get("expr", "")),
                str(row.get("type_full", "")),
                int(str(row.get("granularity") or 1)),
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


def indexes_query(database: str) -> str:
    return (
        "SELECT table, name, expr, type_full, granularity FROM system.data_skipping_indices "
        f"WHERE database = '{database}' FORMAT JSONEachRow"
    )


def needs_scratch(live: Mapping[str, TableShape], retention_days: int) -> bool:
    """Whether any table is missing or carries a comment other than its fingerprint.

    When this is ``False`` the fingerprint fast path holds and the reconcile
    issues no statement at all.
    """
    return any(
        live.get(table.name) is None
        or live[table.name].comment != fingerprint(table, retention_days)
        for table in TABLES
    )


def reconcile(
    live: Mapping[str, TableShape],
    scratch: Mapping[str, TableShape],
    retention_days: int,
) -> list[str]:
    """The statements that bring the live tables to the render, in apply order.

    *scratch* is the render as the store created it in
    :data:`SCRATCH_DATABASE`; it may be empty when every live table already
    carries its fingerprint, which yields no statement.

    Raises:
        SchemaRefused: A live table differs in a key, a column's type or
            default expression, a column the render no longer has, or an
            existing index's definition.
    """
    statements: list[str] = []
    for table in TABLES:
        wanted = fingerprint(table, retention_days)
        current = live.get(table.name)
        if current is None:
            statements.append(render_create(table, retention_days))
            continue
        if current.comment == wanted:
            continue
        shape = scratch[table.name]
        statements.extend(_alter_statements(table, current, shape, retention_days, wanted))
    return statements


def _alter_statements(
    table: TableSpec,
    live: TableShape,
    shape: TableShape,
    retention_days: int,
    wanted: str,
) -> list[str]:
    for key in ("sorting_key", "partition_key", "primary_key"):
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
    for name, live_index in live.indexes.items():
        if name in shape.indexes and live_index != shape.indexes[name]:
            raise SchemaRefused(table.name, f"index {name} changed its definition")
        if name not in shape.indexes:
            raise SchemaRefused(table.name, f"index {name} is not in this release's schema")

    target = f"`{DATABASE}`.`{table.name}`"
    statements: list[str] = []
    rendered_columns = {column.name: column for column in table.columns}
    for name in shape.columns:
        if name not in live.columns and name in rendered_columns:
            statements.append(
                f"ALTER TABLE {target} ADD COLUMN IF NOT EXISTS {rendered_columns[name].render()}"
            )
    for index in table.indexes:
        if index.name not in live.indexes:
            statements.append(f"ALTER TABLE {target} ADD INDEX IF NOT EXISTS {index.render()[6:]}")
            statements.append(f"ALTER TABLE {target} MATERIALIZE INDEX {index.name}")
    if live.ttl != shape.ttl:
        statements.append(f"ALTER TABLE {target} MODIFY TTL {ttl_clause(table, retention_days)}")
    live_settings = live.settings
    changed = [
        f"{key} = {value}"
        for key, value in shape.settings.items()
        if live_settings.get(key) != value
    ]
    if changed:
        statements.append(f"ALTER TABLE {target} MODIFY SETTING {', '.join(changed)}")
    statements.append(f"ALTER TABLE {target} MODIFY COMMENT '{wanted}'")
    return statements
