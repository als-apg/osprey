"""Apply the schema and the accounts to a running store, during ``osprey up``.

Everything goes through ``compose exec -T <store> clickhouse-client``, which
reads the admin's name and password from the container's own environment
(its ``env_file``), so no secret is on a host command line. The script is
handed over on stdin and stops at its first error; a marker query before
each statement lets the failure name the statement that failed. A store
answering ``TOO_MANY_SIMULTANEOUS_QUERIES`` (202) is retried with backoff:
every statement is safe to repeat.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from osprey.deployment import telemetry_config as tc
from osprey.deployment.errors import DeploymentError
from osprey.deployment.telemetry_store import accounts, ddl

#: Runs ``clickhouse-client`` in the store container: (client args, stdin) -> result.
Executor = Callable[[Sequence[str], "str | None"], subprocess.CompletedProcess]

_BUSY_CODE = "Code: 202."
_RETRIES = 5
_HEALTH_TIMEOUT_S = 120.0
_MARKER = "osprey-step-"
_SHOWN_CHARS = 300


class TelemetryStoreError(DeploymentError):
    """The store could not be brought to this release's schema and accounts."""


def compose_executor(base_cmd: Sequence[str], env: Mapping[str, str]) -> Executor:
    """An :data:`Executor` over ``<compose> exec -T telemetry_store clickhouse-client``."""

    def run(args: Sequence[str], stdin: str | None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*base_cmd, "exec", "-T", tc.STORE_SERVICE, "clickhouse-client", *args],
            input=stdin,
            capture_output=True,
            text=True,
            env=dict(env),
            check=False,
        )

    return run


def _run(
    execute: Executor,
    args: Sequence[str],
    stdin: str | None = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> subprocess.CompletedProcess:
    """*args* through *execute*, retried while the store answers 202."""
    result = execute(args, stdin)
    for attempt in range(_RETRIES):
        if result.returncode == 0 or _BUSY_CODE not in (result.stderr or ""):
            break
        sleep(0.5 * 2**attempt)
        result = execute(args, stdin)
    return result


def _rows(execute: Executor, query: str, **kw: Any) -> list[dict[str, Any]]:
    result = _run(execute, ["--query", query], **kw)
    if result.returncode != 0:
        raise TelemetryStoreError(
            f"the telemetry store did not answer {query.split(' FROM ')[-1][:80]!r}: "
            f"{(result.stderr or '').strip()[:_SHOWN_CHARS]}"
        )
    return [json.loads(line) for line in (result.stdout or "").splitlines() if line.strip()]


def wait_until_ready(
    execute: Executor,
    *,
    timeout: float = _HEALTH_TIMEOUT_S,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Block until the store answers a query, or raise after *timeout* seconds."""
    deadline = clock() + timeout
    while True:
        result = execute(["--query", "SELECT 1"], None)
        if result.returncode == 0:
            return
        if clock() >= deadline:
            raise TelemetryStoreError(
                f"the telemetry store did not answer within {int(timeout)} s: "
                f"{(result.stderr or '').strip()[:_SHOWN_CHARS]}"
            )
        sleep(1.0)


def _shapes(execute: Executor, database: str, **kw: Any) -> dict[str, ddl.TableShape]:
    return ddl.shapes_from_rows(
        _rows(execute, ddl.tables_query(database), **kw),
        _rows(execute, ddl.columns_query(database), **kw),
        _rows(execute, ddl.indexes_query(database), **kw),
    )


def apply_script(execute: Executor, statements: Sequence[str], **kw: Any) -> None:
    """Run *statements* as one script, naming the first that fails.

    Raises:
        TelemetryStoreError: A statement failed; the message names it.
    """
    if not statements:
        return
    script = "".join(
        f"SELECT '{_MARKER}{index}';\n{statement};\n" for index, statement in enumerate(statements)
    )
    result = _run(execute, ["--multiquery"], script, **kw)
    if result.returncode == 0:
        return
    reached = [int(n) for n in re.findall(rf"{_MARKER}(\d+)", result.stdout or "")]
    failed = statements[max(reached)] if reached else statements[0]
    shown = " ".join(failed.split())
    if len(shown) > _SHOWN_CHARS:
        shown = shown[:_SHOWN_CHARS] + " ..."
    raise TelemetryStoreError(
        f"the telemetry store refused `{shown}`: {(result.stderr or '').strip()[:_SHOWN_CHARS]}"
    )


def schema_statements(execute: Executor, retention_days: int, **kw: Any) -> list[str]:
    """The reconcile's statements for the live store (see :func:`ddl.reconcile`).

    Raises:
        TelemetryStoreError: A live table needs a migration
            (:class:`ddl.SchemaRefused`).
    """
    live = _shapes(execute, ddl.DATABASE, **kw)
    if not ddl.needs_scratch(live, retention_days):
        return []
    scratch: dict[str, ddl.TableShape] = {}
    if live:
        drop = f"DROP DATABASE IF EXISTS `{ddl.SCRATCH_DATABASE}`"
        apply_script(
            execute, [drop, *ddl.render_schema(retention_days, ddl.SCRATCH_DATABASE)], **kw
        )
        try:
            scratch = _shapes(execute, ddl.SCRATCH_DATABASE, **kw)
        finally:
            apply_script(execute, [drop], **kw)
    try:
        statements = ddl.reconcile(live, scratch, retention_days)
    except ddl.SchemaRefused as exc:
        raise TelemetryStoreError(str(exc)) from exc
    return [f"CREATE DATABASE IF NOT EXISTS `{ddl.DATABASE}`", *statements]


def live_accounts(execute: Executor, **kw: Any) -> accounts.LiveAccounts:
    """The store's access entities, read through *execute*."""
    return accounts.live_from_rows(
        {key: _rows(execute, query, **kw) for key, query in accounts.LIVE_QUERIES.items()}
    )


def stage_store(
    config: Mapping[str, Any],
    repo_root: Path | str,
    execute: Executor,
    **kw: Any,
) -> None:
    """Reconcile the schema and apply the accounts script to the running store.

    Withdrawals and revokes run first, so a schema statement that fails never
    leaves a withdrawn account able to read.

    Raises:
        TelemetryStoreError: Anything that leaves the store short of this
            release's schema or accounts; the message names the cause.
    """
    try:
        passwords = accounts.read_passwords(repo_root, config)
    except accounts.AccountsError as exc:
        raise TelemetryStoreError(str(exc)) from exc
    wait_until_ready(execute, **{k: v for k, v in kw.items() if k == "sleep"})
    live = live_accounts(execute, **kw)
    try:
        account_statements = accounts.render_account_statements(config, passwords, live)
    except accounts.AccountsError as exc:
        raise TelemetryStoreError(str(exc)) from exc
    early = [s.sql for s in account_statements if s.phase < accounts.Phase.CREATE]
    late = [s.sql for s in account_statements if s.phase >= accounts.Phase.CREATE]
    apply_script(execute, early, **kw)
    apply_script(execute, schema_statements(execute, tc.retention_days(config), **kw), **kw)
    apply_script(execute, late, **kw)
