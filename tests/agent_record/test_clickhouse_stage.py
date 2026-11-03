"""``osprey up`` stages the agent record's ClickHouse store beside the telemetry store.

The unit cases drive ``_stage_agent_record_clickhouse`` with compose stubbed
and a fake executor, and check the exact statements it sends. The live cases
run on the pinned image (the contract suite's store) and check that the
account's grant works, and that no other account but the admin can touch the
record table.
"""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from datetime import date, timedelta

import httpx
import pytest

from osprey.agent_record.backends import clickhouse_ddl as ddl
from osprey.agent_record.backends import clickhouse_stage
from osprey.agent_record.backends.clickhouse import ClickHouseBackend
from osprey.agent_record.contract import KIND_PROMPT
from osprey.agent_record.credentials import (
    CLICKHOUSE_ADMIN_USER,
    CLICKHOUSE_FILE,
    CLICKHOUSE_PASSWORD_ENV,
    CLICKHOUSE_USER,
)
from osprey.deployment import container_lifecycle as lifecycle
from tests.agent_record.contract_suite import _kit as k
from tests.agent_record.contract_suite.conftest import clickhouse_store  # noqa: F401

_BASE_CMD = ["docker", "compose", "-f", "docker-compose.yml"]


class FakeStore:
    """A fresh store: answers the readiness and shape queries, records every script."""

    def __init__(self):
        self.scripts: list[str] = []

    def __call__(self, args, stdin):
        if args[:1] == ["--multiquery"]:
            self.scripts.append(stdin)
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[1] == "SELECT 1":
            return subprocess.CompletedProcess(args, 0, "1\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")


def _statements(script: str) -> list[str]:
    parts = [p.strip() for p in script.split(";\n") if p.strip()]
    return [p for p in parts if not p.startswith("SELECT 'osprey-step-")]


@pytest.fixture
def staged(monkeypatch):
    """Compose stubbed out: the commands run, and the executors handed out."""
    seen: dict = {"run": [], "executors": []}
    store = FakeStore()

    @contextmanager
    def no_watch(_cmd, **_kw):
        yield {}

    def fake_compose_executor(base_cmd, _env, _service=None):
        seen["executors"].append(list(base_cmd))
        return store

    monkeypatch.setattr(lifecycle, "compose_base_cmd", lambda *a, **kw: list(_BASE_CMD))
    monkeypatch.setattr(lifecycle, "get_runtime_command", lambda config: ["docker"])
    monkeypatch.setattr(lifecycle, "_env_file_args", lambda *a, **kw: [])
    monkeypatch.setattr(lifecycle, "runtime_env", lambda config, env, **kw: dict(env))
    monkeypatch.setattr(lifecycle, "compose_provider_env", lambda *a, **kw: {})
    monkeypatch.setattr(lifecycle, "watched_compose_up", no_watch)
    monkeypatch.setattr(lifecycle, "run_captured", lambda cmd, **kw: seen["run"].append(cmd))
    monkeypatch.setattr(clickhouse_stage, "compose_executor", fake_compose_executor)
    monkeypatch.setattr(clickhouse_stage.time, "sleep", lambda s: None)
    seen["store"] = store
    return seen


def _clickhouse_config() -> dict:
    return {"agent_record": {"backend": "clickhouse"}}


def _write_password(root, password: str = "pw'1") -> None:
    (root / CLICKHOUSE_FILE).write_text(
        f"OSPREY_AGENT_RECORD_CLICKHOUSE_USER={CLICKHOUSE_USER}\n"
        f"{CLICKHOUSE_PASSWORD_ENV}={password}\n",
        encoding="utf-8",
    )


# --- with compose stubbed ---------------------------------------------------------------------


def test_up_starts_the_store_alone_then_stages_its_exact_statements(staged, tmp_path):
    _write_password(tmp_path)
    lifecycle._stage_agent_record_clickhouse(_clickhouse_config(), [], {}, tmp_path)

    assert staged["run"] == [[*_BASE_CMD, "up", "-d", "agent_record_clickhouse"]]
    assert staged["executors"] == [_BASE_CMD]
    schema, accounts = (_statements(s) for s in staged["store"].scripts)
    assert schema == ["CREATE DATABASE IF NOT EXISTS `agent_record`", *ddl.render_schema()[1:]]
    assert accounts == [
        "CREATE USER IF NOT EXISTS `agent_record` IDENTIFIED WITH sha256_password BY 'pw\\'1'",
        "ALTER USER `agent_record` IDENTIFIED WITH sha256_password BY 'pw\\'1'",
        "REVOKE ALL ON *.* FROM `agent_record`",
        "GRANT SELECT, INSERT, ALTER DELETE ON `agent_record`.`entries` TO `agent_record`",
        "GRANT SELECT, INSERT ON `agent_record`.`state` TO `agent_record`",
    ]


@pytest.mark.parametrize(
    "config",
    [{}, {"agent_record": {}}, {"agent_record": {"backend": "embedded"}}],
    ids=["no-block", "default-backend", "embedded"],
)
def test_nothing_runs_unless_the_record_is_on_clickhouse(staged, tmp_path, config):
    _write_password(tmp_path)
    lifecycle._stage_agent_record_clickhouse(config, [], {}, tmp_path)
    assert staged["run"] == []
    assert staged["store"].scripts == []


@pytest.mark.parametrize(
    "content", [None, f"{CLICKHOUSE_PASSWORD_ENV}=\n"], ids=["absent", "empty"]
)
def test_a_missing_password_stops_up_before_the_store_starts(staged, tmp_path, content):
    if content is not None:
        (tmp_path / CLICKHOUSE_FILE).write_text(content, encoding="utf-8")
    with pytest.raises(clickhouse_stage.AgentRecordStoreError, match=CLICKHOUSE_FILE):
        lifecycle._stage_agent_record_clickhouse(_clickhouse_config(), [], {}, tmp_path)
    assert staged["run"] == []


@pytest.mark.usefixtures("staged")
def test_a_refused_statement_stops_up(tmp_path, monkeypatch):
    _write_password(tmp_path)

    def refuses(args, _stdin):
        if args[:1] == ["--multiquery"]:
            return subprocess.CompletedProcess(args, 1, "", "Code: 497. not enough privileges")
        return subprocess.CompletedProcess(args, 0, "1\n" if args[1] == "SELECT 1" else "", "")

    monkeypatch.setattr(clickhouse_stage, "compose_executor", lambda *a, **kw: refuses)
    with pytest.raises(clickhouse_stage.AgentRecordStoreError, match="refused"):
        lifecycle._stage_agent_record_clickhouse(_clickhouse_config(), [], {}, tmp_path)


def test_the_start_flow_stages_the_record_store_before_the_stack_starts():
    import inspect

    source = inspect.getsource(lifecycle)
    image = source.index(
        "_build_project_image(config, dev_mode, env, build_context, compose_files=compose_files)"
    )
    record = source.index(
        "_stage_agent_record_clickhouse(config, compose_files, env, Path(repo_root), "
        "provider=provider)"
    )
    site_ca = source.index("stage_service_site_ca(config, compose_files, repo_root)")
    assert image < record < site_ca


# --- on the pinned image ----------------------------------------------------------------------


def _admin_rows(execute, query: str) -> list[str]:
    result = execute(["--query", query], None)
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if line.strip()]


def _as_account(store, sql: str) -> httpx.Response:
    return httpx.post(
        store.url,
        content=sql.encode(),
        headers={"X-ClickHouse-User": CLICKHOUSE_USER, "X-ClickHouse-Key": store.password},
        timeout=10.0,
    )


@pytest.mark.xdist_group("docker")
def test_the_account_writes_reads_and_drops_partitions_of_the_record(clickhouse_store):  # noqa: F811
    tables = [f"`{ddl.DATABASE}`.`{t.name}`" for t in ddl.TABLES]
    clickhouse_stage.apply_script(clickhouse_store.admin, [f"TRUNCATE TABLE {t}" for t in tables])
    with ClickHouseBackend(clickhouse_store.url, clickhouse_store.password) as backend:
        backend.append([k.entry("operator", "grant-probe", KIND_PROMPT, 0.0)])
        backend.put_state("grant-probe", {"n": 1})
        assert backend.get_state("grant-probe") == {"n": 1}
        assert [e.key for e in k.rows(backend)] == ["grant-probe"]
        assert backend.delete_arrived_before(date.today() + timedelta(days=2)) == 1
        assert k.rows(backend) == []


@pytest.mark.xdist_group("docker")
def test_the_account_can_do_nothing_beyond_its_grants(clickhouse_store):  # noqa: F811
    refused = [
        f"CREATE TABLE `{ddl.DATABASE}`.`intruder` (x UInt8) ENGINE = Memory",
        f"TRUNCATE TABLE `{ddl.DATABASE}`.`{ddl.ENTRIES_TABLE}`",
        f"ALTER TABLE `{ddl.DATABASE}`.`{ddl.STATE_TABLE}` DELETE WHERE 1",
        "SELECT name FROM system.users",
        "CREATE USER intruder IDENTIFIED WITH sha256_password BY 'x'",
    ]
    for sql in refused:
        response = _as_account(clickhouse_store, sql)
        assert response.status_code != 200, sql
        assert "ACCESS_DENIED" in response.text or "Code: 497" in response.text, response.text


@pytest.mark.xdist_group("docker")
def test_only_agent_record_reaches_the_record_table_besides_the_admin(clickhouse_store):  # noqa: F811
    # Any grant that covers the record table: on it, on its database, or global.
    covering = (
        "FROM system.grants WHERE (database IS NULL OR database = 'agent_record') "
        "AND (`table` IS NULL OR `table` = 'entries')"
    )
    users = set(
        _admin_rows(
            clickhouse_store.admin,
            f"SELECT DISTINCT user_name {covering} AND user_name IS NOT NULL",
        )
    )
    roles = _admin_rows(
        clickhouse_store.admin, f"SELECT DISTINCT role_name {covering} AND role_name IS NOT NULL"
    )
    assert users - {CLICKHOUSE_ADMIN_USER} == {CLICKHOUSE_USER}
    assert roles == []
    assert set(_admin_rows(clickhouse_store.admin, "SELECT name FROM system.users")) - {
        CLICKHOUSE_ADMIN_USER
    } == {CLICKHOUSE_USER}
