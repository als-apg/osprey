"""The ClickHouse record's schema: render, fingerprint, reconcile and stage.

Unit cases run on canned shapes and a fake executor. The live cases stage the
pinned image (the contract suite's store) and check the fast path, a drifted
comment's repair and the account's exact grant set.
"""

from __future__ import annotations

import subprocess
from dataclasses import replace

import pytest

from osprey.agent_record.backends import clickhouse_ddl as ddl
from osprey.agent_record.backends import clickhouse_stage as stage
from osprey.agent_record.credentials import CLICKHOUSE_USER
from tests.agent_record.contract_suite.conftest import clickhouse_store  # noqa: F401

# --- render and fingerprint -------------------------------------------------------------------


def test_the_record_table_is_a_plain_mergetree_by_arrival_day_with_no_ttl():
    create = ddl.render_create(ddl.ENTRIES)
    assert "ENGINE = MergeTree\n" in create
    assert "ORDER BY (identity, session_id, key, chunk_index)" in create
    assert "PARTITION BY toDate(inserted_at)" in create
    assert "TTL" not in create
    assert "person" not in create
    assert create.startswith("CREATE TABLE IF NOT EXISTS `agent_record`.`entries`")


def test_the_record_columns_are_the_contracts_in_its_order():
    from osprey.agent_record.contract import ENTRY_COLUMNS

    assert tuple(ddl.ENTRIES.column_names()) == ENTRY_COLUMNS


def test_state_is_a_replacing_mergetree_on_its_version():
    create = ddl.render_create(ddl.STATE)
    assert "ENGINE = ReplacingMergeTree(version)" in create
    assert "ORDER BY name" in create


def test_the_fingerprint_moves_with_the_render_only():
    base = ddl.fingerprint(ddl.ENTRIES)
    assert ddl.render_create(ddl.ENTRIES).endswith(f"COMMENT '{base}'")
    # The target database does not move it; the scratch render carries the same comment.
    assert ddl.render_create(ddl.ENTRIES, ddl.SCRATCH_DATABASE).endswith(f"COMMENT '{base}'")
    assert ddl.fingerprint(replace(ddl.ENTRIES, order_by="(key, chunk_index)")) != base
    extra = replace(ddl.ENTRIES, columns=(*ddl.ENTRIES.columns, ddl.Column("x", "String")))
    assert ddl.fingerprint(extra) != base
    assert ddl.fingerprint(ddl.STATE) != base


def test_the_schema_creates_the_database_then_every_table():
    statements = ddl.render_schema()
    assert statements[0] == "CREATE DATABASE IF NOT EXISTS `agent_record`"
    assert [s.split("`.`")[1].split("`")[0] for s in statements[1:]] == ["entries", "state"]


# --- reconcile --------------------------------------------------------------------------------


def _shape(table: ddl.TableSpec, **changes) -> ddl.TableShape:
    shape = ddl.TableShape(
        engine=table.engine,
        sorting_key=table.order_by.strip("()"),
        partition_key=table.partition_by,
        primary_key=table.order_by.strip("()"),
        comment=ddl.fingerprint(table),
        columns={c.name: (c.type, "", "") for c in table.columns},
    )
    for key, value in changes.items():
        setattr(shape, key, value)
    return shape


def _shapes(**overrides) -> dict[str, ddl.TableShape]:
    return {t.name: overrides.get(t.name, _shape(t)) for t in ddl.TABLES}


def test_a_fresh_store_creates_every_table():
    assert ddl.needs_scratch({})
    assert ddl.reconcile({}, {}) == [ddl.render_create(t) for t in ddl.TABLES]


def test_equal_fingerprints_issue_nothing():
    live = _shapes()
    assert not ddl.needs_scratch(live)
    assert ddl.reconcile(live, {}) == []


def test_a_drifted_comment_on_an_equal_shape_is_rewritten_only():
    live = _shapes(entries=_shape(ddl.ENTRIES, comment="old"))
    assert ddl.needs_scratch(live)
    assert ddl.reconcile(live, _shapes()) == [
        f"ALTER TABLE `agent_record`.`entries` MODIFY COMMENT '{ddl.fingerprint(ddl.ENTRIES)}'"
    ]


def test_a_missing_column_is_added_before_the_comment():
    columns = dict(_shape(ddl.ENTRIES).columns)
    del columns["tool_use_id"]
    live = _shapes(entries=_shape(ddl.ENTRIES, comment="old", columns=columns))
    statements = ddl.reconcile(live, _shapes())
    assert statements[0] == (
        "ALTER TABLE `agent_record`.`entries` ADD COLUMN IF NOT EXISTS `tool_use_id` String"
    )
    assert statements[-1].endswith(f"MODIFY COMMENT '{ddl.fingerprint(ddl.ENTRIES)}'")


@pytest.mark.parametrize(
    ("table", "changes", "named"),
    [
        (ddl.ENTRIES, {"sorting_key": "key, chunk_index"}, "sorting key"),
        (ddl.ENTRIES, {"partition_key": "toYYYYMM(inserted_at)"}, "partition key"),
        (ddl.ENTRIES, {"engine": "ReplacingMergeTree"}, "engine"),
        (ddl.STATE, {"engine": "ReplacingMergeTree"}, "engine"),
        (ddl.ENTRIES, {"primary_key": "identity"}, "primary key"),
    ],
)
def test_a_key_or_engine_change_is_refused(table, changes, named):
    live = _shapes(**{table.name: _shape(table, comment="old", **changes)})
    with pytest.raises(ddl.SchemaRefused, match=named) as raised:
        ddl.reconcile(live, _shapes())
    assert raised.value.table == table.name
    assert "migrated by hand" in str(raised.value)


def test_a_column_type_change_or_an_extra_column_is_refused():
    columns = dict(_shape(ddl.ENTRIES).columns)
    columns["chunk_index"] = ("UInt64", "", "")
    live = _shapes(entries=_shape(ddl.ENTRIES, comment="old", columns=columns))
    with pytest.raises(ddl.SchemaRefused, match="column chunk_index"):
        ddl.reconcile(live, _shapes())
    columns = {**_shape(ddl.ENTRIES).columns, "person": ("String", "", "")}
    live = _shapes(entries=_shape(ddl.ENTRIES, comment="old", columns=columns))
    with pytest.raises(ddl.SchemaRefused, match="column person"):
        ddl.reconcile(live, _shapes())


def test_shapes_are_read_from_system_rows():
    shapes = ddl.shapes_from_rows(
        [
            {
                "name": "entries",
                "sorting_key": "identity, session_id, key, chunk_index",
                "partition_key": "toDate(inserted_at)",
                "primary_key": "identity, session_id, key, chunk_index",
                "comment": "c",
                "engine_full": "MergeTree PARTITION BY toDate(inserted_at) ORDER BY (identity)"
                " SETTINGS index_granularity = 8192",
            },
            {
                "name": "state",
                "engine_full": "ReplacingMergeTree(version) ORDER BY name SETTINGS x = 1",
            },
        ],
        [{"table": "entries", "name": "key", "type": "String"}],
    )
    assert shapes["entries"].engine == "MergeTree"
    assert shapes["state"].engine == "ReplacingMergeTree(version)"
    assert shapes["entries"].columns == {"key": ("String", "", "")}


# --- stage, with a fake executor --------------------------------------------------------------


class FakeStore:
    """Answers ``clickhouse-client`` calls from canned system rows; records scripts."""

    def __init__(self, live_tables=(), live_columns=(), *, fail_script=None):
        self.live = (list(live_tables), list(live_columns))
        self.scripts: list[str] = []
        self.fail_script = fail_script

    def __call__(self, args, stdin):
        if args[:1] == ["--multiquery"]:
            self.scripts.append(stdin)
            if self.fail_script and self.fail_script in stdin:
                reached = []
                for line in stdin.splitlines():
                    if "osprey-step-" in line:
                        reached.append(line.split("'")[1])
                    elif self.fail_script in line:
                        break
                out = "".join(m + "\n" for m in reached)
                return subprocess.CompletedProcess(args, 1, out, "Code: 62. boom")
            return subprocess.CompletedProcess(args, 0, "", "")
        query = args[1]
        if query == "SELECT 1":
            return subprocess.CompletedProcess(args, 0, "1\n", "")
        rows = []
        if "system.tables" in query and "'agent_record'" in query:
            rows = self.live[0]
        if "system.columns" in query and "'agent_record'" in query:
            rows = self.live[1]
        import json

        return subprocess.CompletedProcess(args, 0, "".join(json.dumps(r) + "\n" for r in rows), "")


def _statements(script: str) -> list[str]:
    parts = [p.strip() for p in script.split(";\n") if p.strip()]
    return [p for p in parts if not p.startswith("SELECT 'osprey-step-")]


def test_staging_a_fresh_store_creates_schema_then_the_account_with_exact_grants():
    store = FakeStore()
    stage.stage(store, "pw'1", sleep=lambda s: None)
    schema, accounts = (_statements(s) for s in store.scripts)
    assert schema == ["CREATE DATABASE IF NOT EXISTS `agent_record`", *ddl.render_schema()[1:]]
    assert accounts == [
        f"CREATE USER IF NOT EXISTS `{CLICKHOUSE_USER}` IDENTIFIED WITH sha256_password BY 'pw\\'1'",
        f"ALTER USER `{CLICKHOUSE_USER}` IDENTIFIED WITH sha256_password BY 'pw\\'1'",
        f"REVOKE ALL ON *.* FROM `{CLICKHOUSE_USER}`",
        "GRANT SELECT, INSERT, ALTER DELETE ON `agent_record`.`entries` TO `agent_record`",
        "GRANT SELECT, INSERT ON `agent_record`.`state` TO `agent_record`",
    ]


def test_a_staged_store_gets_only_the_account_script():
    tables = [{"name": t.name, "comment": ddl.fingerprint(t)} for t in ddl.TABLES]
    store = FakeStore(tables)
    stage.stage(store, "pw", sleep=lambda s: None)
    assert len(store.scripts) == 1
    assert "CREATE USER" in store.scripts[0]
    assert "CREATE TABLE" not in store.scripts[0]


def test_a_failed_account_statement_never_shows_the_password():
    store = FakeStore(fail_script="CREATE USER")
    with pytest.raises(stage.AgentRecordStoreError) as raised:
        stage.stage(store, "s3cret-pw", sleep=lambda s: None)
    assert "s3cret-pw" not in str(raised.value)
    assert "CREATE USER IF NOT EXISTS `agent_record` IDENTIFIED ..." in str(raised.value)


def test_a_failed_schema_statement_is_named():
    store = FakeStore(fail_script="CREATE TABLE")
    with pytest.raises(
        stage.AgentRecordStoreError,
        match="refused `CREATE TABLE IF NOT EXISTS `agent_record`.`entries`",
    ):
        stage.stage(store, "pw", sleep=lambda s: None)


def test_an_empty_password_is_refused_before_anything_runs():
    store = FakeStore()
    with pytest.raises(stage.AgentRecordStoreError, match="no password"):
        stage.stage(store, "", sleep=lambda s: None)
    assert store.scripts == []


def test_a_store_that_never_answers_names_the_wait():
    times = iter([0.0, 0.0, 200.0])

    def down(args, _stdin):
        return subprocess.CompletedProcess(args, 210, "", "Connection refused")

    with pytest.raises(stage.AgentRecordStoreError, match="did not answer within 120 s"):
        stage.wait_until_ready(down, sleep=lambda s: None, clock=lambda: next(times))


def test_a_busy_store_is_retried():
    answers = iter(
        [subprocess.CompletedProcess([], 1, "", "Code: 202. busy")] * 2
        + [subprocess.CompletedProcess([], 0, "", "")]
    )
    calls = []

    def busy(args, _stdin):
        calls.append(args)
        return next(answers)

    stage.apply_script(busy, ["SELECT 2"], sleep=lambda s: None)
    assert len(calls) == 3


def test_the_compose_executor_execs_the_client_in_the_record_store(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(stage.subprocess, "run", fake_run)
    stage.compose_executor(["docker", "compose"], {})(["--query", "SELECT 1"], None)
    assert seen["cmd"] == [
        "docker",
        "compose",
        "exec",
        "-T",
        "agent_record_clickhouse",
        "clickhouse-client",
        "--query",
        "SELECT 1",
    ]


# --- stage, on the pinned image ---------------------------------------------------------------


def _client_rows(execute, query):
    result = execute(["--query", query], None)
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if line.strip()]


@pytest.mark.xdist_group("docker")
def test_a_staged_store_restages_through_the_fast_path(clickhouse_store):  # noqa: F811
    assert stage.schema_statements(clickhouse_store.admin) == []
    stage.stage(clickhouse_store.admin, clickhouse_store.password)


@pytest.mark.xdist_group("docker")
def test_a_drifted_comment_is_repaired_through_the_scratch_database(clickhouse_store):  # noqa: F811
    admin = clickhouse_store.admin
    stage.apply_script(admin, ["ALTER TABLE `agent_record`.`state` MODIFY COMMENT 'old'"])
    statements = stage.schema_statements(admin)
    assert statements == [
        "CREATE DATABASE IF NOT EXISTS `agent_record`",
        f"ALTER TABLE `agent_record`.`state` MODIFY COMMENT '{ddl.fingerprint(ddl.STATE)}'",
    ]
    stage.apply_script(admin, statements)
    assert stage.schema_statements(admin) == []
    assert _client_rows(admin, "SHOW DATABASES LIKE 'agent_record_shape'") == []


@pytest.mark.xdist_group("docker")
def test_the_account_holds_exactly_its_grants(clickhouse_store):  # noqa: F811
    grants = sorted(_client_rows(clickhouse_store.admin, f"SHOW GRANTS FOR {CLICKHOUSE_USER}"))
    assert grants == [
        "GRANT SELECT, INSERT ON agent_record.state TO agent_record",
        "GRANT SELECT, INSERT, ALTER DELETE ON agent_record.entries TO agent_record",
    ]
