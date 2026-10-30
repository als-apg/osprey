"""Stage the agent record's ClickHouse store: database, tables, account, grants.

Everything goes through an :data:`Executor` that runs ``clickhouse-client`` in
the store container as its admin, whose name and password the client reads
from the container's own environment, so no admin secret is on a host command
line. ``osprey up`` uses a ``compose exec -T`` executor
(:func:`compose_executor`); the contract suite uses ``docker exec``; unit tests
use a fake. The script is handed over on stdin and stops at its first error; a
marker query before each statement lets the failure name the statement that
failed. A store answering ``TOO_MANY_SIMULTANEOUS_QUERIES`` (202) is retried
with backoff: every statement is safe to repeat.

The one account, :data:`~osprey.agent_record.credentials.CLICKHOUSE_USER`, is
granted ``INSERT`` and ``SELECT`` on both tables, plus ``ALTER DELETE`` on the
record table (what ``DROP PARTITION`` needs), and nothing else: every earlier
grant is revoked first, so a stage always leaves exactly this set.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from osprey.agent_record.backends import clickhouse_ddl as ddl
from osprey.agent_record.credentials import CLICKHOUSE_SERVICE, CLICKHOUSE_USER
from osprey.deployment.errors import DeploymentError

#: Runs ``clickhouse-client`` in the store container: (client args, stdin) -> result.
Executor = Callable[[Sequence[str], "str | None"], subprocess.CompletedProcess]

_BUSY_CODE = "Code: 202."
_RETRIES = 5
_HEALTH_TIMEOUT_S = 120.0
_MARKER = "osprey-step-"
_SHOWN_CHARS = 300


class AgentRecordStoreError(DeploymentError):
    """The agent record store could not be brought to this release's schema and account."""


def compose_executor(
    base_cmd: Sequence[str], env: Mapping[str, str], service: str = CLICKHOUSE_SERVICE
) -> Executor:
    """An :data:`Executor` over ``<compose> exec -T <service> clickhouse-client``."""

    def run(args: Sequence[str], stdin: str | None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*base_cmd, "exec", "-T", service, "clickhouse-client", *args],
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
        raise AgentRecordStoreError(
            f"the agent record store did not answer {query.split(' FROM ')[-1][:80]!r}: "
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
            raise AgentRecordStoreError(
                f"the agent record store did not answer within {int(timeout)} s: "
                f"{(result.stderr or '').strip()[:_SHOWN_CHARS]}"
            )
        sleep(1.0)


def apply_script(execute: Executor, statements: Sequence[str], **kw: Any) -> None:
    """Run *statements* as one script, naming the first that fails.

    A statement that carries a secret is never shown: the message names its
    position instead.

    Raises:
        AgentRecordStoreError: A statement failed; the message names it.
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
    index = max(reached) if reached else 0
    shown = " ".join(statements[index].split())
    if " IDENTIFIED " in shown:
        shown = shown.split(" IDENTIFIED ")[0] + " IDENTIFIED ..."
    if len(shown) > _SHOWN_CHARS:
        shown = shown[:_SHOWN_CHARS] + " ..."
    stderr = (result.stderr or "").strip()
    if " IDENTIFIED " in stderr:
        stderr = "(the store's message is withheld: it may quote the password)"
    raise AgentRecordStoreError(
        f"the agent record store refused `{shown}`: {stderr[:_SHOWN_CHARS]}"
    )


def _shapes(execute: Executor, database: str, **kw: Any) -> dict[str, ddl.TableShape]:
    return ddl.shapes_from_rows(
        _rows(execute, ddl.tables_query(database), **kw),
        _rows(execute, ddl.columns_query(database), **kw),
    )


def schema_statements(execute: Executor, **kw: Any) -> list[str]:
    """The reconcile's statements for the live store (see :func:`ddl.reconcile`).

    Raises:
        AgentRecordStoreError: A live table needs a migration
            (:class:`ddl.SchemaRefused`).
    """
    live = _shapes(execute, ddl.DATABASE, **kw)
    if not ddl.needs_scratch(live):
        return []
    scratch: dict[str, ddl.TableShape] = {}
    if any(
        table.name in live and live[table.name].comment != ddl.fingerprint(table)
        for table in ddl.TABLES
    ):
        drop = f"DROP DATABASE IF EXISTS `{ddl.SCRATCH_DATABASE}`"
        apply_script(execute, [drop, *ddl.render_schema(ddl.SCRATCH_DATABASE)], **kw)
        try:
            scratch = _shapes(execute, ddl.SCRATCH_DATABASE, **kw)
        finally:
            apply_script(execute, [drop], **kw)
    try:
        statements = ddl.reconcile(live, scratch)
    except ddl.SchemaRefused as exc:
        raise AgentRecordStoreError(str(exc)) from exc
    return [f"CREATE DATABASE IF NOT EXISTS `{ddl.DATABASE}`", *statements]


def account_statements(password: str, user: str = CLICKHOUSE_USER) -> list[str]:
    """The account and its grants, idempotent: the password is (re)set and the
    grant set replaced by exactly this one."""
    if not password:
        raise AgentRecordStoreError(f"the agent record account {user} has no password")
    identified = f"IDENTIFIED WITH sha256_password BY {ddl.quote(password)}"
    entries = ddl.qualified(ddl.ENTRIES_TABLE)
    state = ddl.qualified(ddl.STATE_TABLE)
    return [
        f"CREATE USER IF NOT EXISTS `{user}` {identified}",
        f"ALTER USER `{user}` {identified}",
        f"REVOKE ALL ON *.* FROM `{user}`",
        f"GRANT SELECT, INSERT, ALTER DELETE ON {entries} TO `{user}`",
        f"GRANT SELECT, INSERT ON {state} TO `{user}`",
    ]


def stage(execute: Executor, password: str, **kw: Any) -> None:
    """Bring the running store to this release's schema, account and grants.

    Args:
        execute: Runs ``clickhouse-client`` as the store's admin.
        password: The agent-record account's password.
        **kw: ``sleep`` (and ``clock``, for the readiness wait) for tests.

    Raises:
        AgentRecordStoreError: Anything that leaves the store short of this
            release's schema or account; the message names the cause.
    """
    accounts = account_statements(password)
    wait_until_ready(execute, **{k: v for k, v in kw.items() if k in ("sleep", "clock")})
    run_kw = {k: v for k, v in kw.items() if k == "sleep"}
    apply_script(execute, schema_statements(execute, **run_kw), **run_kw)
    apply_script(execute, accounts, **run_kw)
