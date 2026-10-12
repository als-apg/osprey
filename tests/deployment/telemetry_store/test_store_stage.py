"""Applying the script: the failing statement is named, 202 is retried, the fast path is silent."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from osprey.deployment.telemetry_store import ddl, stage


def _done(code: int = 0, out: str = "", err: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], code, out, err)


def test_failure_names_the_failing_statement() -> None:
    def execute(args, _stdin):
        assert args == ["--multiquery"]
        return _done(1, "osprey-step-0\nosprey-step-1\n", "Code: 62. Syntax error")

    with pytest.raises(stage.TelemetryStoreError, match="refused `GRANT b`: Code: 62"):
        stage.apply_script(execute, ["CREATE a", "GRANT b", "CREATE c"])


def test_busy_store_is_retried() -> None:
    calls = []

    def execute(_args, stdin):
        calls.append(stdin)
        return (
            _done(1, "", "Code: 202. TOO_MANY_SIMULTANEOUS_QUERIES") if len(calls) < 3 else _done()
        )

    stage.apply_script(execute, ["SELECT 1"], sleep=lambda _s: None)
    assert len(calls) == 3


def test_fingerprint_fast_path_runs_nothing() -> None:
    rows = json.loads(
        (Path(__file__).parent / "fixtures" / "live_shapes_retention_30.json").read_text()
    )
    answers = {
        ddl.tables_query(ddl.DATABASE): rows["tables"],
        ddl.columns_query(ddl.DATABASE): rows["columns"],
        ddl.indexes_query(ddl.DATABASE): rows["indexes"],
    }

    def execute(args, _stdin):
        assert args[0] == "--query", "a fast path writes nothing"
        return _done(0, "".join(json.dumps(row) + "\n" for row in answers[args[1]]))

    assert stage.schema_statements(execute, 30) == []


def test_store_not_ready_times_out() -> None:
    ticks = iter(range(1000))
    with pytest.raises(stage.TelemetryStoreError, match="did not answer within 3 s"):
        stage.wait_until_ready(
            lambda args, stdin: _done(1, "", "Connection refused"),
            timeout=3,
            sleep=lambda _s: None,
            clock=lambda: next(ticks),
        )
