"""The store's schema: the exporter column contract, the pinned keys, the reconcile."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from osprey.deployment.telemetry_store import ddl

_FIXTURES = Path(__file__).parent / "fixtures"
_EXPORTER = _FIXTURES / "clickhouseexporter-0.162.0"

#: The exporter's insert template per signal, as vendored from v0.162.0.
_INSERT_TEMPLATES = {
    "logs": "logs_insert.sql",
    "traces": "traces_insert.sql",
    "metrics_gauge": "metrics_gauge_insert.sql",
    "metrics_sum": "metrics_sum_insert.sql",
    "metrics_histogram": "metrics_histogram_insert.sql",
    "metrics_exponential_histogram": "metrics_exp_histogram_insert.sql",
    "metrics_summary": "metrics_summary_insert.sql",
}


def _inserted_columns(template: str) -> list[str]:
    """The column list of an insert template rendered with ``FeatureColumnNames = ''``."""
    text = (_EXPORTER / template).read_text().replace("{{.FeatureColumnNames}}", "")
    head = text.split(") VALUES", 1)[0]
    return [name.strip() for name in head.split("(", 1)[1].split(",") if name.strip()]


@pytest.mark.parametrize(
    ("instance", "signal"),
    [(i, s) for i, signals in ddl.EXPORTER_TABLES.items() for s in signals],
)
def test_every_inserted_column_exists(instance: str, signal: str) -> None:
    table = ddl.TABLES_BY_NAME[ddl.EXPORTER_TABLES[instance][signal]]
    missing = set(_inserted_columns(_INSERT_TEMPLATES[signal])) - set(table.column_names())
    assert not missing, f"{instance} -> {table.name} inserts columns the DDL lacks: {missing}"


def test_no_table_has_event_name() -> None:
    for table in ddl.TABLES:
        assert "EventName" not in {column.name for column in table.columns}
        assert "EventName" not in ddl.render_create(table, 14)


def test_registry_covers_every_table() -> None:
    assert set(ddl.TIME_COLUMNS) == set(ddl.TABLES_BY_NAME)


_KEYS = {
    "otel_logs": ("(User, Event, SessionId, Timestamp)", "toDate(Timestamp)"),
    "audit_ledger": ("(User, SessionId, Timestamp)", "toDate(Timestamp)"),
    "otel_traces": ("(ServiceName, SpanName, toDateTime(Timestamp))", "toDate(Timestamp)"),
    **{
        name: (
            "(ServiceName, MetricName, toStartOfHour(TimeUnix), cityHash64(Attributes), TimeUnix)",
            "toDate(TimeUnix)",
        )
        for name in ddl.TIME_COLUMNS
        if name.startswith("otel_metrics_")
    },
}

_MATERIALIZED = {
    "otel_logs": {
        "User": "ResourceAttributes['osprey.user']",
        "SessionId": "LogAttributes['session.id']",
        "TerminalSession": "ResourceAttributes['terminal.session']",
        "Event": "LogAttributes['event.name']",
    },
    "audit_ledger": {
        "User": "ResourceAttributes['osprey.user']",
        "SessionId": "LogAttributes['osprey.audit.session_id']",
        "ToolUseId": "LogAttributes['osprey.audit.tool_use_id']",
        "TerminalSession": "LogAttributes['osprey.audit.session']",
        "Surface": "LogAttributes['osprey.audit.surface']",
        "OnBehalfOf": "LogAttributes['osprey.audit.on_behalf_of']",
        "Actor": "LogAttributes['osprey.audit.actor']",
        "Decision": "LogAttributes['osprey.audit.decision']",
        "DurationMs": "toFloat64OrNull(LogAttributes['osprey.audit.duration_ms'])",
        "Incomplete": "LogAttributes['osprey.audit.incomplete'] = 'true'",
        "Truncated": "LogAttributes['osprey.audit.truncated'] = 'true'",
        "Unparsed": "LogAttributes['osprey.audit.unparsed'] = 'true'",
        "TimeUnparsed": "LogAttributes['osprey.audit.time_unparsed'] = 'true'",
    },
    "otel_traces": {
        "User": "ResourceAttributes['osprey.user']",
        "SessionId": "SpanAttributes['session.id']",
    },
}


@pytest.mark.parametrize("name", sorted(ddl.TIME_COLUMNS))
def test_keys_and_materialized_columns_are_pinned(name: str) -> None:
    table = ddl.TABLES_BY_NAME[name]
    assert (table.order_by, table.partition_by) == _KEYS[name]
    materialized = {
        column.name: column.definition.split(" MATERIALIZED ", 1)[1]
        for column in table.columns
        if " MATERIALIZED " in column.definition
    }
    assert materialized == _MATERIALIZED.get(name, {})
    assert "InsertedAt" in {column.name for column in table.columns}
    create = ddl.render_create(table, 14)
    assert f"TTL toDateTime({ddl.TIME_COLUMNS[name]}) + INTERVAL 14 DAY" in create
    for setting in (
        "ttl_only_drop_parts = 1",
        "materialize_ttl_recalculate_only = 1",
        "non_replicated_deduplication_window = 1000",
    ):
        assert setting in create
    assert create.endswith(f"COMMENT '{ddl.fingerprint(table, 14)}'")


def test_fingerprint_moves_with_the_render_only() -> None:
    table = ddl.OTEL_LOGS
    assert ddl.fingerprint(table, 14) == ddl.fingerprint(table, 14)
    assert ddl.fingerprint(table, 14) != ddl.fingerprint(table, 30)
    assert ddl.render_create(table, 14, ddl.SCRATCH_DATABASE).endswith(
        f"COMMENT '{ddl.fingerprint(table, 14)}'"
    )


# --------------------------------------------------------------------------- reconcile


def _canned() -> dict[str, ddl.TableShape]:
    """The tables as a real 26.8.18.2 store reported them, created at retention 30."""
    rows = json.loads((_FIXTURES / "live_shapes_retention_30.json").read_text())
    return ddl.shapes_from_rows(rows["tables"], rows["columns"], rows["indexes"])


def test_equal_fingerprint_issues_nothing() -> None:
    live = _canned()
    assert not ddl.needs_scratch(live, 30)
    assert ddl.reconcile(live, {}, 30) == []


def test_fresh_store_creates_every_table() -> None:
    statements = ddl.reconcile({}, {}, 14)
    assert statements == [ddl.render_create(table, 14) for table in ddl.TABLES]


def test_one_new_index_adds_and_materializes_it_once() -> None:
    live, scratch = _canned(), _canned()
    del live["otel_logs"].indexes["idx_event"]
    live["otel_logs"].comment = "stale"
    statements = ddl.reconcile(live, scratch, 30)
    adds = [s for s in statements if " ADD INDEX " in s]
    materializes = [s for s in statements if " MATERIALIZE INDEX " in s]
    assert adds == [
        "ALTER TABLE `telemetry`.`otel_logs` ADD INDEX IF NOT EXISTS "
        "idx_event Event TYPE set(64) GRANULARITY 1"
    ]
    assert materializes == ["ALTER TABLE `telemetry`.`otel_logs` MATERIALIZE INDEX idx_event"]
    assert statements[-1].startswith("ALTER TABLE `telemetry`.`otel_logs` MODIFY COMMENT ")
    assert len(statements) == 3


def test_changed_ttl_modifies_it_once_per_table() -> None:
    live = _canned()
    scratch = _canned()
    for shape in scratch.values():
        shape.engine_full = shape.engine_full.replace("toIntervalDay(30)", "toIntervalDay(14)")
    statements = ddl.reconcile(live, scratch, 14)
    ttl = [s for s in statements if " MODIFY TTL " in s]
    assert len(ttl) == len(ddl.TABLES)
    assert (
        "ALTER TABLE `telemetry`.`otel_logs` MODIFY TTL toDateTime(Timestamp) + INTERVAL 14 DAY"
        in ttl
    )
    assert not [s for s in statements if "INDEX" in s or "SETTING" in s or "ADD COLUMN" in s]


def test_missing_column_and_changed_setting_are_applied() -> None:
    live, scratch = _canned(), _canned()
    shape = live["audit_ledger"]
    del shape.columns["InsertedAt"]
    shape.engine_full = shape.engine_full.replace(
        "non_replicated_deduplication_window = 1000", "non_replicated_deduplication_window = 10"
    )
    shape.comment = "stale"
    statements = ddl.reconcile(live, scratch, 30)
    assert statements[0] == (
        "ALTER TABLE `telemetry`.`audit_ledger` ADD COLUMN IF NOT EXISTS "
        "`InsertedAt` DateTime DEFAULT now()"
    )
    assert (
        "ALTER TABLE `telemetry`.`audit_ledger` MODIFY SETTING "
        "non_replicated_deduplication_window = 1000"
    ) in statements


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: setattr(s, "sorting_key", "User, SessionId"),
        lambda s: setattr(s, "partition_key", "toYYYYMM(Timestamp)"),
        lambda s: s.columns.__setitem__("Body", ("LowCardinality(String)", "", "")),
        lambda s: s.columns.__setitem__(
            "User", ("LowCardinality(String)", "MATERIALIZED", "LogAttributes['user']")
        ),
        lambda s: s.columns.__setitem__("Stray", ("String", "", "")),
        lambda s: s.indexes.__setitem__("idx_session", ("SessionId", "minmax", 1)),
    ],
    ids=["sorting-key", "partition-key", "type", "materialized", "extra-column", "index"],
)
def test_shape_changes_are_refused(mutate) -> None:
    live, scratch = _canned(), _canned()
    live["otel_logs"].comment = "stale"
    mutate(live["otel_logs"])
    with pytest.raises(ddl.SchemaRefused, match=re.escape("telemetry.otel_logs")) as info:
        ddl.reconcile(live, scratch, 30)
    assert "otel_logs_pre_<version>" in str(info.value)
