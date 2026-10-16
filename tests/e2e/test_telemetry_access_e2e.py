"""The telemetry access end to end: reader accounts, refusals, limits, and history scoping.

Runs in the model-free lane against public images only; no test here names a
provider or reaches a model.

One deployment serves the whole module. Its profile declares ``bluesky`` in
``deployed_services`` and authenticates a two-person roster by password, so the
render names the queueserver in the ledger's trust condition and in the
own-scope policy, and a view user exists per person. Only the store and its
gateway are started, from the rendered compose: the store through the same
``_stage_telemetry_store`` call ``osprey up`` makes (which applies the schema
and the accounts script), then the gateway. No web tier, no bluesky image.

The history cases send what the page sends: ``EXECUTE AS <view user> <query>``
by POST as the ``history`` account, with ``log_comment``, ``quota_key`` and
``http_wait_end_of_query=1`` on the URL, and the view contract's own query
files (``osprey.history.queries``). Rows reach the store the way they do in a
deployment (OTLP posted to the gateway with a sender's login, ledger lines
under ``var/audit/<identity>/``), except where a case needs a volume or a row
size no sender produces: those rows are written as the store's admin.

The cases run in file order; the ones that re-stage the accounts restore the
module's own config before they return.

CONTAINER SAFETY: every docker invocation names an exact container of this
project. Teardown is ``compose down`` of this project followed by the
project-scoped volume sweep (``tests/e2e/_volumes.py``); never a prune, never
a wildcard.
"""

from __future__ import annotations

import base64
import copy
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from osprey.audit.envelope import POSTURE_SOURCE_APP, POSTURE_SOURCE_SPAWN, AuditEnvelope
from osprey.audit.tool_call import build_record
from osprey.deployment import telemetry_config as tc
from osprey.deployment import telemetry_credentials as creds
from osprey.deployment.compose_generator import compose_base_cmd, compose_provider_env
from osprey.deployment.container_lifecycle import (
    _ensure_bluesky_control_plane_keys,
    _ensure_bluesky_document_plane_certs,
    _env_file_args,
    _stage_telemetry_store,
    as_built_compose_files,
)
from osprey.deployment.runtime_helper import get_runtime_command, runtime_env
from osprey.deployment.telemetry_gateway import trusted_on_behalf_of
from osprey.deployment.telemetry_store import accounts
from osprey.deployment.telemetry_store.stage import (
    Executor,
    apply_script,
    compose_executor,
    live_accounts,
    stage_store,
)
from osprey.history import queries
from tests.e2e._otlp_replay import rebase_otlp_times
from tests.e2e._volumes import remove_project_volumes
from tests.e2e.profile_edits import set_pairs

pytestmark = [
    pytest.mark.model_free,
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
]

#: This module's compose project. A compose project is addressed by name, so
#: no other e2e module may deploy under it (``tests/deployment/test_e2e_project_names.py``).
PROJECT_NAME = "osprey-e2e-telemetry-access"
#: This module's port block (``deployment.port_base``); no other e2e module uses it.
PORT_BASE = 17500

STORE_CONTAINER = f"{PROJECT_NAME}-telemetry-store"
GATEWAY_CONTAINER = f"{PROJECT_NAME}-telemetry-gateway"

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "telemetry" / "claude_cli_session.otlp.json"
)

ALICE = "alice"
BOB = "bob"
#: The reader the narrowing and ``days_back`` cases hold to ``otel_logs``, and
#: the reader the recipe runs as (every table, no ``days_back``).
READER_X = "x"
READER_Y = "y"
READER_X_DAYS = 3

VERB_TIMEOUT_S = 600
ROW_TIMEOUT_S = 120.0
HTTP_TIMEOUT_S = 60.0

#: The probe day (``research/probe-clickhouse.md``): one UTC day the archive
#: exported, 920,509 rows. The session-list load is 14 of those days.
PROBE_DAY_ROWS = 920_509
LOAD_DAYS = 14
#: The probe day's event mix: one prompt per twenty events, each session's
#: prompts interleaved among its other events, and sessions of these sizes in
#: turn (rows per session). One block of them is 5,760 rows, so a probe day is
#: about 800 sessions and the default window holds well over 1,000.
PROMPT_EVERY = 20
SESSION_SIZES = (40, 120, 400, 1_200, 4_000)
LOAD_USERS = 7
OTHER_EVENTS = ("api_request", "tool_result", "tool_decision", "assistant_response")

#: The 2,000-event session: twenty results of 512 KiB among them.
LONG_SESSION_TOOL_CALLS = 2_000
LONG_SESSION_EVENTS = 1_000
BIG_RESULTS = 20
BIG_RESULT_BYTES = 512 * 1024

#: Variables a developer shell or an earlier test may carry that would change
#: what a spawned ``osprey`` verb resolves.
_SCRUBBED_ENV = (
    "CLAUDECODE",
    tc.GATEWAY_HOST_ENV,
    tc.GATEWAY_PORT_ENV,
    creds.LOGIN_ENV,
    creds.AUDIT_IDENTITY_ENV,
    "OSPREY_TERMINAL_SESSION",
    "OSPREY_TERMINAL_USER",
)


def _profile_edits() -> dict[str, Any]:
    """The pins this module's assertions stand on, as ``osprey init --set`` edits.

    Roster entries are objects with an explicit index. The persona catalog is
    added after ``init`` (:func:`_complete_profile`): ``init`` materializes one
    profile per catalog entry, and this deployment builds no persona.
    """
    return {
        "config": {
            "system.timezone": "UTC",
            "deploy.fqdn": "127.0.0.1",
            "registry.url": "127.0.0.1:5999/osprey-e2e",
            "deployed_services": [tc.STORE_SERVICE, tc.GATEWAY_SERVICE, "bluesky"],
            "modules.web_terminals": {
                "enabled": True,
                "image_source": "registry",
                "default_persona": "assistant",
                "users": [{"name": ALICE, "index": 0}, {"name": BOB, "index": 1}],
                "auth": {"method": "password", "allow_insecure_http": True},
            },
            "services.telemetry_store.readers": [
                {
                    "name": READER_X,
                    "tables": ["otel_logs", "audit_ledger"],
                    "days_back": READER_X_DAYS,
                },
                {"name": READER_Y},
            ],
        }
    }


def _complete_profile(repo: Path) -> None:
    """Add the persona catalog and the bluesky block ``init --set`` cannot spell."""
    path = repo / "profile.yml"
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    profile["config"]["modules.web_terminals.personas"] = {
        "assistant": {"project": f"{PROJECT_NAME}-assistant"}
    }
    profile["bluesky"] = {"tiled_enabled": False}
    path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# The deployment
# --------------------------------------------------------------------------- #


@dataclass
class Stack:
    """The built repo, its rendered config, and how the cases reach the store."""

    repo: Path
    config: dict[str, Any]
    compose_files: list[str]
    env: dict[str, str]
    base_cmd: list[str] = field(default_factory=list)
    run_env: dict[str, str] = field(default_factory=dict)

    @property
    def store_url(self) -> str:
        return f"http://127.0.0.1:{tc.store_port(self.config)}/"

    @property
    def gateway_url(self) -> str:
        return f"http://127.0.0.1:{tc.gateway_port(self.config)}"

    @property
    def execute(self) -> Executor:
        return compose_executor(self.base_cmd, self.run_env)

    def compose(self, *args: str, timeout: int = VERB_TIMEOUT_S) -> None:
        result = subprocess.run(
            [*self.base_cmd, *args],
            env=self.run_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        assert result.returncode == 0, f"compose {' '.join(args)}: {result.stderr}"

    def file_value(self, name: str, key: str) -> str:
        for line in (self.repo / name).read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() == key:
                return v.strip().strip("'\"")
        raise AssertionError(f"{name} holds no {key}")

    def history_password(self) -> str:
        return self.file_value(creds.HISTORY_FILE, creds.HISTORY_PASSWORD_ENV)

    def reader_password(self, name: str) -> str:
        return self.file_value(creds.reader_file(name), creds.READER_KEY)

    def sender_auth(self, identity: str) -> str:
        login = creds.read_sender_login(self.repo, identity)
        assert login, f"no sender login for {identity!r}"
        return "Basic " + base64.b64encode(login.encode()).decode()

    def restage(self, config: Mapping[str, Any]) -> None:
        """Mint what *config* needs and apply its accounts, as ``osprey up`` does."""
        creds.ensure_telemetry_credentials(self.repo, config, reader_keys=True)
        stage_store(config, self.repo, self.execute)


def _osprey_bin() -> Path:
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


def _env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _SCRUBBED_ENV}


def _run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=VERB_TIMEOUT_S, env=_env()
    )
    if result.returncode != 0:
        pytest.fail(
            f"{' '.join(cmd[1:3])} failed (rc={result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


def _docker(*args: str, timeout: int = 120, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout, check=False
    )
    if check and result.returncode != 0:
        raise AssertionError(f"docker {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def _remove_project_containers() -> None:
    for name in (STORE_CONTAINER, GATEWAY_CONTAINER):
        _docker("rm", "-f", name, check=False)


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    """Render the roster deployment, then start only its store and gateway."""
    osprey = _osprey_bin()
    base = tmp_path_factory.mktemp("telemetry_access_e2e")
    repo = base / PROJECT_NAME

    _remove_project_containers()
    remove_project_volumes(PROJECT_NAME)

    _run(
        [
            str(osprey),
            "init",
            str(repo),
            "--preset",
            "hello-world",
            "--no-git",
            "--set",
            f"port_base={PORT_BASE}",
            *set_pairs(_profile_edits()),
        ],
        cwd=base,
    )
    _complete_profile(repo)
    _run([str(osprey), "build", "--skip-deps", "--skip-lifecycle"], cwd=repo)
    (repo / ".env").touch()
    config = yaml.safe_load((repo / "build" / "config.yml").read_text())
    if not tc.history_page_deployed(config):
        pytest.fail("the render does not deploy the history page; its view users would not exist")
    if trusted_on_behalf_of(config)[1] != ("queueserver",):
        pytest.fail(f"the render trusts no queueserver: {trusted_on_behalf_of(config)!r}")

    deployed = Stack(
        repo=repo,
        config=config,
        compose_files=as_built_compose_files(config, repo),
        env=_env(),
    )
    deployed.run_env = runtime_env(config, {**deployed.env, **compose_provider_env(None, repo)})
    deployed.base_cmd = compose_base_cmd(
        get_runtime_command(config), deployed.compose_files, repo, _env_file_args(repo, None)
    )
    try:
        # What `osprey up` mints before any compose call: the rendered compose
        # names the bluesky services' keys, and compose interpolates every
        # service of a file even when it starts one.
        _ensure_bluesky_control_plane_keys(config, repo / ".env")
        _ensure_bluesky_document_plane_certs(config, repo / ".env")
        creds.ensure_telemetry_credentials(repo, config, reader_keys=True)
        _stage_telemetry_store(config, deployed.compose_files, deployed.env, repo)
        deployed.compose("up", "-d", tc.GATEWAY_SERVICE)
        _wait_ready(deployed)
        yield deployed
    finally:
        down = subprocess.run(
            [*deployed.base_cmd, "down", "--timeout", "10"],
            env=deployed.run_env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if down.returncode != 0:
            print(f"compose down rc={down.returncode}\n{down.stdout}\n{down.stderr}")
        _remove_project_containers()
        remove_project_volumes(PROJECT_NAME)


def _wait_ready(stack: Stack, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            if _sql("SELECT 1").strip() == "1":
                httpx.get(
                    f"http://127.0.0.1:{tc.metrics_port(stack.config)}/metrics", timeout=3
                ).raise_for_status()
                return
        except (AssertionError, httpx.HTTPError) as exc:
            last = str(exc)
        time.sleep(1.0)
    raise AssertionError(f"store and gateway not ready after {timeout:.0f} s: {last}")


# --------------------------------------------------------------------------- #
# Reaching the store
# --------------------------------------------------------------------------- #


def _sql(query: str, fmt: str = "TSV", stdin: str | None = None) -> str:
    """Run *query* as the store's admin account, inside the store container."""
    result = subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            STORE_CONTAINER,
            "clickhouse-client",
            "--format",
            fmt,
            "-q",
            query,
        ],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"store query failed: {result.stderr.strip()}\n{query[:500]}")
    return result.stdout


def _scalar(query: str) -> int:
    return int(_sql(query).strip() or 0)


def _wait_count(query: str, expected: int, timeout: float = ROW_TIMEOUT_S) -> int:
    deadline = time.monotonic() + timeout
    value = -1
    while time.monotonic() < deadline:
        try:
            value = _scalar(query)
        except AssertionError:
            value = -1
        if value >= expected:
            return value
        time.sleep(1.0)
    return value


def _http(
    stack: Stack,
    user: str,
    password: str,
    query: str,
    *,
    params: Mapping[str, str] | None = None,
    method: str = "POST",
) -> httpx.Response:
    """One query over the store's HTTP interface, as *user*."""
    headers = {"X-ClickHouse-User": user, "X-ClickHouse-Key": password}
    if method == "GET":
        return httpx.get(
            stack.store_url,
            params={**(params or {}), "query": query},
            headers=headers,
            timeout=HTTP_TIMEOUT_S,
        )
    return httpx.post(
        stack.store_url,
        params=dict(params or {}),
        content=query.encode(),
        headers=headers,
        timeout=HTTP_TIMEOUT_S,
    )


def _page_params(person: str, view: str, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The URL parameters the page puts on every store call."""
    return {
        **(extra or {}),
        "log_comment": f"{person}:{view}",
        "quota_key": person,
        "http_wait_end_of_query": "1",
    }


def _as_view(
    stack: Stack,
    view: str,
    query: str,
    *,
    person: str = ALICE,
    params: Mapping[str, str] | None = None,
    log_comment: str | None = None,
) -> httpx.Response:
    """``EXECUTE AS <view> <query>`` by POST as ``history``, the way the page sends it."""
    url = _page_params(person, view, params)
    if log_comment is not None:
        url["log_comment"] = log_comment
    return _http(
        stack,
        accounts.HISTORY_USER,
        stack.history_password(),
        f"EXECUTE AS `{view}` {query.strip().rstrip(';')}\nFORMAT JSONEachRow",
        params=url,
    )


def _rows(response: httpx.Response) -> list[dict[str, Any]]:
    assert response.status_code == 200, f"{response.status_code}: {response.text[:600]}"
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def _view_rows(stack: Stack, view: str, query: str, **kw: Any) -> list[dict[str, Any]]:
    return _rows(_as_view(stack, view, query, **kw))


def _view_count(stack: Stack, view: str, table: str, where: str) -> int:
    rows = _view_rows(stack, view, f"SELECT count() AS n FROM telemetry.{table} WHERE {where}")
    return int(rows[0]["n"])


def _contract(
    stack: Stack, view: str, name: str, values: Mapping[str, Any], **kw: Any
) -> httpx.Response:
    """A view-contract query, its values sent as the store's typed parameters."""
    return _as_view(
        stack, view, queries.load(name), params=queries.query_parameters(name, values), **kw
    )


def _refused_with(response: httpx.Response, code: int) -> None:
    assert response.status_code != 200, f"expected a refusal, got 200: {response.text[:300]}"
    assert f"Code: {code}" in response.text, response.text[:600]


# --------------------------------------------------------------------------- #
# Sending rows the way a deployment does
# --------------------------------------------------------------------------- #


def _kv(attrs: Mapping[str, str]) -> list[dict[str, Any]]:
    return [{"key": k, "value": {"stringValue": v}} for k, v in attrs.items()]


def _log_body(case: str, session: str, terminal: str, *, event: str, time_ns: int) -> dict:
    return {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": _kv({"service.name": "e2e", "terminal.session": terminal})
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "e2e"},
                        "logRecords": [
                            {
                                "timeUnixNano": str(time_ns),
                                "body": {"stringValue": f"{case}-{event}"},
                                "attributes": _kv(
                                    {"e2e.case": case, "session.id": session, "event.name": event}
                                ),
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _span_body(case: str, session: str, *, time_ns: int) -> dict:
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _kv({"service.name": "e2e"})},
                "scopeSpans": [
                    {
                        "scope": {"name": "e2e"},
                        "spans": [
                            {
                                "traceId": uuid.uuid4().hex,
                                "spanId": uuid.uuid4().hex[:16],
                                "name": "e2e-span",
                                "kind": 1,
                                "startTimeUnixNano": str(time_ns),
                                "endTimeUnixNano": str(time_ns + 1_000_000),
                                "attributes": _kv({"e2e.case": case, "session.id": session}),
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _post(stack: Stack, identity: str, signal: str, body: Mapping[str, Any]) -> None:
    response = httpx.post(
        f"{stack.gateway_url}/v1/{signal}",
        content=json.dumps(body),
        headers={"Content-Type": "application/json", "Authorization": stack.sender_auth(identity)},
        timeout=30,
    )
    assert response.status_code == 200, f"/v1/{signal} as {identity}: {response.status_code}"


def _stamp(when: datetime.datetime) -> str:
    return when.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _ledger(stack: Stack, identity: str, stem: str, record: Mapping[str, Any]) -> None:
    """Append one ledger line under ``var/audit/<identity>/<stem>.jsonl``."""
    directory = stack.repo / "var" / "audit" / identity
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{stem}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _tool_call(
    *,
    when: datetime.datetime,
    actor: str,
    terminal: str,
    session_id: str,
    tool_use_id: str,
    subject: str,
    duration_ms: float = 120.0,
    result: Any = "ok",
) -> dict[str, Any]:
    return build_record(
        ts=_stamp(when),
        actor=actor,
        posture="sandbox",
        posture_source=POSTURE_SOURCE_SPAWN,
        session=terminal,
        session_id=session_id,
        tool_use_id=tool_use_id,
        server="workspace",
        subject=subject,
        decision="allowed",
        reason="e2e",
        approval=None,
        target=None,
        generation=None,
        arguments={"e2e": True},
        arguments_ref=None,
        result=result,
        result_ref=None,
        error=None,
        is_error=False,
        facts={},
        duration_ms=duration_ms,
    )


def _envelope(
    *,
    when: datetime.datetime,
    surface: str,
    actor: str,
    subject: str,
    decision: str,
    session: str | None = None,
    on_behalf_of: str | None = None,
    tool_use_id: str | None = None,
    posture_source: str = POSTURE_SOURCE_APP,
) -> dict[str, Any]:
    return AuditEnvelope(
        ts=_stamp(when),
        surface=surface,
        actor=actor,
        posture="sandbox",
        posture_source=posture_source,
        session=session,
        on_behalf_of=on_behalf_of,
        subject=subject,
        decision=decision,
        reason="e2e",
        tool_use_id=tool_use_id,
    ).to_dict()


# --------------------------------------------------------------------------- #
# The scoping rows
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Seed:
    """The rows every scoping case reads, tagged by one case marker."""

    case: str
    alice_session: str
    bob_session: str
    old_case: str


@pytest.fixture(scope="module")
def seed(stack: Stack) -> Seed:
    """Each person's log, span and tool call; the sidecar's rows; a 4-day-old row."""
    case = uuid.uuid4().hex[:12]
    old_case = f"{case}-old"
    now = datetime.datetime.now(datetime.UTC)
    ns = time.time_ns()
    sessions = {person: f"e2e-{case}-{person}" for person in (ALICE, BOB)}
    for person, session in sessions.items():
        _post(
            stack,
            person,
            "logs",
            _log_body(case, session, f"t-{person}", event="user_prompt", time_ns=ns),
        )
        _post(stack, person, "traces", _span_body(case, session, time_ns=ns))
        _ledger(
            stack,
            person,
            "workspace",
            _tool_call(
                when=now,
                actor=person,
                terminal=f"t-{person}",
                session_id=session,
                tool_use_id=f"toolu_{case}_{person}",
                subject=case,
            ),
        )
        # The sidecar decides a sign-in on each person's behalf.
        _ledger(
            stack,
            "sidecar",
            "auth_sidecar",
            _envelope(
                when=now,
                surface="auth_sidecar",
                actor="sidecar",
                subject=case,
                decision="allowed",
                on_behalf_of=person,
            ),
        )
    four_days_ago = ns - 4 * 86_400 * 1_000_000_000
    _post(
        stack,
        ALICE,
        "logs",
        _log_body(old_case, sessions[ALICE], "t-alice", event="user_prompt", time_ns=four_days_ago),
    )

    assert (
        _wait_count(
            f"SELECT count() FROM telemetry.otel_logs WHERE LogAttributes['e2e.case'] = '{case}'", 2
        )
        == 2
    )
    assert (
        _wait_count(
            f"SELECT count() FROM telemetry.otel_logs WHERE LogAttributes['e2e.case'] = '{old_case}'",
            1,
        )
        == 1
    )
    assert (
        _wait_count(
            f"SELECT count() FROM telemetry.otel_traces WHERE SpanAttributes['e2e.case'] = '{case}'",
            2,
        )
        == 2
    )
    assert (
        _wait_count(f"SELECT count() FROM telemetry.audit_ledger WHERE Body LIKE '%{case}%'", 4)
        == 4
    )
    return Seed(
        case=case,
        alice_session=sessions[ALICE],
        bob_session=sessions[BOB],
        old_case=old_case,
    )


def _case_logs(seed: Seed, user: str) -> str:
    return f"User = '{user}' AND LogAttributes['e2e.case'] = '{seed.case}'"


def _all_views(stack: Stack) -> list[str]:
    return accounts.configured_view_users(stack.config)


# --------------------------------------------------------------------------- #
# EXECUTE AS: who may, and how
# --------------------------------------------------------------------------- #


def test_execute_as_by_post_works_for_history_and_is_refused_for_readers_and_by_get(
    stack: Stack,
) -> None:
    view = accounts.person_view_user(ALICE)
    query = f"EXECUTE AS `{view}` SELECT currentUser() AS who FORMAT JSONEachRow"

    by_post = _http(
        stack,
        accounts.HISTORY_USER,
        stack.history_password(),
        query,
        params=_page_params(ALICE, view),
    )
    assert _rows(by_post) == [{"who": view}]

    by_get = _http(
        stack,
        accounts.HISTORY_USER,
        stack.history_password(),
        query,
        params=_page_params(ALICE, view),
        method="GET",
    )
    assert by_get.status_code != 200, f"EXECUTE AS by GET answered 200: {by_get.text[:300]}"

    for reader in (READER_X, READER_Y):
        response = _http(stack, reader, stack.reader_password(reader), query)
        _refused_with(response, 497)


def test_every_view_user_is_readonly_under_the_page_limits(stack: Stack) -> None:
    limits = tc.store_limits(stack.config)
    expected = {
        "readonly": "1",
        "max_result_rows": str(limits["result_rows"]),
        "max_result_bytes": str(2 * tc.gateway_max_log_size(stack.config)),
        "max_rows_to_read": str(limits["rows_read"]),
        "max_bytes_to_read": str(limits["bytes_read"]),
        "result_overflow_mode": "throw",
        "read_overflow_mode": "throw",
        "max_execution_time": str(accounts.HISTORY_SECONDS),
    }
    select = ", ".join(f"toString(getSetting('{name}')) AS {name}" for name in expected)
    views = _all_views(stack)
    assert len(views) == 5, views
    for view in views:
        (row,) = _view_rows(stack, view, f"SELECT {select}")
        assert row == expected, f"{view}: {row}"


def test_an_over_cap_load_full_is_a_non_200_naming_the_error_never_a_partial_200(
    stack: Stack,
) -> None:
    session = f"overcap-{uuid.uuid4().hex[:8]}"
    cap = 2 * tc.gateway_max_log_size(stack.config)
    piece = 512 * 1024
    _sql(
        "INSERT INTO telemetry.otel_logs (Timestamp, ServiceName, Body, ResourceAttributes, LogAttributes) "
        "SELECT now64(9), 'e2e', "
        # repeat() stops at 1,000,000 characters: join pieces of 512 KiB.
        f"arrayStringConcat(arrayMap(i -> repeat('x', {piece}), range({cap // piece + 2}))), "
        f"map('osprey.user', '{ALICE}'), map('session.id', '{session}', 'event.name', 'assistant_response')"
    )
    ts, body_hash = (
        _sql(
            "SELECT toString(Timestamp), cityHash64(Body) FROM telemetry.otel_logs "
            f"WHERE User = '{ALICE}' AND SessionId = '{session}'"
        )
        .strip()
        .split("\t")
    )
    response = _contract(
        stack,
        accounts.person_view_user(ALICE),
        "load_full_log",
        {"user": ALICE, "session": session, "ts": ts, "body_hash": body_hash},
    )
    assert response.status_code != 200, (
        f"an over-cap row came back as 200 with {len(response.content)} bytes"
    )
    assert "TOO_MANY_ROWS_OR_BYTES" in response.text, response.text[:600]


# --------------------------------------------------------------------------- #
# Own scope
# --------------------------------------------------------------------------- #


def test_current_user_inside_the_own_policy_names_the_impersonated_user(
    stack: Stack, seed: Seed
) -> None:
    for person in (ALICE, BOB):
        view = accounts.person_view_user(person)
        (row,) = _view_rows(stack, view, "SELECT currentUser() AS who")
        assert row["who"] == view
        # The own_otel_logs policy reads substring(currentUser(), 6): the
        # person's own row passes it only because currentUser() is the view user.
        assert (
            _view_count(stack, view, "otel_logs", f"LogAttributes['e2e.case'] = '{seed.case}'") == 1
        )
        assert _view_count(stack, view, "otel_logs", _case_logs(seed, person)) == 1


def test_view_alice_sees_no_bob_row_and_the_sidecars_row_for_alice(
    stack: Stack, seed: Seed
) -> None:
    view = accounts.person_view_user(ALICE)
    assert _view_count(stack, view, "otel_logs", f"User = '{BOB}'") == 0
    assert _view_count(stack, view, "otel_traces", f"User = '{BOB}'") == 0
    assert _view_count(stack, view, "audit_ledger", f"User = '{BOB}'") == 0
    assert _view_count(stack, view, "audit_ledger", f"OnBehalfOf = '{BOB}'") == 0
    sidecar = f"User = 'sidecar' AND OnBehalfOf = '{ALICE}' AND Body LIKE '%{seed.case}%'"
    assert _scalar(f"SELECT count() FROM telemetry.audit_ledger WHERE {sidecar}") == 1
    assert _view_count(stack, view, "audit_ledger", sidecar) == 1
    assert (
        _view_count(stack, view, "audit_ledger", f"User = '{ALICE}' AND Body LIKE '%{seed.case}%'")
        == 1
    )


def test_view_all_sees_both_people(stack: Stack, seed: Seed) -> None:
    view = accounts.VIEW_ALL_USER
    for person in (ALICE, BOB):
        assert _view_count(stack, view, "otel_logs", _case_logs(seed, person)) == 1
        assert (
            _view_count(
                stack,
                view,
                "otel_traces",
                f"User = '{person}' AND SpanAttributes['e2e.case'] = '{seed.case}'",
            )
            == 1
        )
        assert (
            _view_count(
                stack,
                view,
                "audit_ledger",
                f"OnBehalfOf = '{person}' AND Body LIKE '%{seed.case}%'",
            )
            == 1
        )


def test_traces_are_scoped_by_user(stack: Stack, seed: Seed) -> None:
    for person, other in ((ALICE, BOB), (BOB, ALICE)):
        view = accounts.person_view_user(person)
        rows = _view_rows(
            stack,
            view,
            "SELECT User FROM telemetry.otel_traces "
            f"WHERE SpanAttributes['e2e.case'] = '{seed.case}'",
        )
        assert [r["User"] for r in rows] == [person]
        # The page's agent list, as the person.
        agents = _rows(
            _contract(
                stack,
                view,
                "agents",
                {"user": other, "session": f"e2e-{seed.case}-{other}"},
                person=person,
            )
        )
        assert agents == []


def test_query_log_names_the_view_user_and_the_log_comment(stack: Stack) -> None:
    view = accounts.person_view_user(ALICE)
    marker = f"{ALICE}:{view}"
    tag = f"probe-{uuid.uuid4().hex[:12]}"
    _view_rows(stack, view, f"SELECT count() AS n, '{tag}' AS probe FROM telemetry.otel_logs")
    _sql("SYSTEM FLUSH LOGS")
    rows = _sql(
        "SELECT user, initial_user, log_comment FROM system.query_log "
        f"WHERE type = 'QueryFinish' AND query LIKE '%{tag}%' AND query NOT LIKE '%system.query_log%' "
        "ORDER BY event_time_microseconds",
        fmt="JSONEachRow",
    )
    logged = [json.loads(line) for line in rows.splitlines() if line.strip()]
    assert logged, "the page's query left no query_log row"
    assert {row["log_comment"] for row in logged} == {marker}
    # One row for the authenticated outer query, one for the impersonated one.
    assert sorted(row["user"] for row in logged) == sorted([accounts.HISTORY_USER, view]), logged


# --------------------------------------------------------------------------- #
# Reader views: tables and days_back, and narrowing
# --------------------------------------------------------------------------- #


def _config_with_reader(stack: Stack, reader: Mapping[str, Any]) -> dict[str, Any]:
    """The module's config with reader ``reader['name']`` replaced or added."""
    config = copy.deepcopy(stack.config)
    store = config["services"]["telemetry_store"]
    store["readers"] = [r for r in store["readers"] if r["name"] != reader["name"]] + [dict(reader)]
    return config


def _row_policies(prefix: str) -> set[str]:
    out = _sql(f"SELECT short_name FROM system.row_policies WHERE short_name LIKE '{prefix}%'")
    return set(out.split())


def test_narrowing_reader_x_to_one_table_withdraws_the_other_and_keeps_days_back(
    stack: Stack, seed: Seed
) -> None:
    view = accounts.reader_view_user(READER_X)
    assert _row_policies(f"viewreader_{READER_X}_") == {
        f"viewreader_{READER_X}_otel_logs",
        f"viewreader_{READER_X}_audit_ledger",
    }
    assert _view_count(stack, view, "audit_ledger", f"Body LIKE '%{seed.case}%'") == 4

    narrowed = _config_with_reader(
        stack, {"name": READER_X, "tables": ["otel_logs"], "days_back": READER_X_DAYS}
    )
    try:
        stack.restage(narrowed)
        _refused_with(_as_view(stack, view, "SELECT count() FROM telemetry.audit_ledger"), 497)
        assert _row_policies(f"viewreader_{READER_X}_") == {f"viewreader_{READER_X}_otel_logs"}
        assert _row_policies(f"reader_{READER_X}_") == {f"reader_{READER_X}_otel_logs"}
        # days_back 3: the 4-day-old row is out of the reader's view, not the person's.
        assert (
            _view_count(stack, view, "otel_logs", f"LogAttributes['e2e.case'] = '{seed.old_case}'")
            == 0
        )
        assert (
            _view_count(stack, view, "otel_logs", f"LogAttributes['e2e.case'] = '{seed.case}'") == 2
        )
        assert (
            _view_count(
                stack,
                accounts.person_view_user(ALICE),
                "otel_logs",
                f"LogAttributes['e2e.case'] = '{seed.old_case}'",
            )
            == 1
        )
    finally:
        stack.restage(stack.config)
    assert _view_count(stack, view, "audit_ledger", f"Body LIKE '%{seed.case}%'") == 4


def test_a_reader_named_own_leaves_view_alice_scoped(stack: Stack, seed: Seed) -> None:
    """The profile refuses the name; the accounts script, given it anyway, stays scoped."""
    with_own = _config_with_reader(stack, {"name": "own"})
    try:
        stack.restage(with_own)
        assert accounts.reader_view_user("own") in _sql("SELECT name FROM system.users").split()
        assert _row_policies(accounts.OWN_POLICY_PREFIX) == {
            f"own_{t}" for t in accounts.VIEW_TABLES
        }
        view = accounts.person_view_user(ALICE)
        assert _view_count(stack, view, "otel_logs", f"User = '{BOB}'") == 0
        assert (
            _view_count(stack, view, "audit_ledger", f"User = '{BOB}' OR OnBehalfOf = '{BOB}'") == 0
        )
        assert _view_count(stack, view, "otel_logs", _case_logs(seed, ALICE)) == 1
    finally:
        stack.restage(stack.config)
    assert accounts.reader_view_user("own") not in _sql("SELECT name FROM system.users").split()


# --------------------------------------------------------------------------- #
# A cut accounts script
# --------------------------------------------------------------------------- #


def _naming(statements: list[accounts.Statement], name: str) -> list[str]:
    ident = f"`{name}`"
    return [s.sql for s in statements if ident in s.sql]


def _bob_rows_visible(stack: Stack, view: str) -> int:
    """Bob's rows *view* returns, or 0 when the store refuses the impersonation."""
    total = 0
    for table, where in (
        ("otel_logs", f"User = '{BOB}'"),
        ("otel_traces", f"User = '{BOB}'"),
        ("audit_ledger", f"User = '{BOB}' OR OnBehalfOf = '{BOB}'"),
    ):
        response = _as_view(
            stack, view, f"SELECT count() AS n FROM telemetry.{table} WHERE {where}"
        )
        if response.status_code == 200:
            total += int(_rows(response)[0]["n"])
    return total


def test_the_accounts_script_cut_after_any_view_user_statement_never_shows_alice_a_bob_row(
    stack: Stack, seed: Seed
) -> None:
    view = accounts.person_view_user(ALICE)
    passwords = accounts.read_passwords(stack.repo, stack.config)

    # The create path, from no view user: cut after each statement. The page's
    # IMPERSONATE is granted at every cut, as if it had been held all along.
    _sql(f"DROP USER IF EXISTS `{view}`")
    created = _naming(
        accounts.render_account_statements(stack.config, passwords, live_accounts(stack.execute)),
        view,
    )
    assert len(created) >= 5, created
    try:
        for cut in range(1, len(created) + 1):
            _sql(f"DROP USER IF EXISTS `{view}`")
            apply_script(stack.execute, created[:cut])
            if view in _sql("SELECT name FROM system.users").split():
                _sql(f"GRANT IMPERSONATE ON `{view}` TO `{accounts.HISTORY_USER}`")
            assert _bob_rows_visible(stack, view) == 0, f"cut after: {created[cut - 1]}"
    finally:
        stack.restage(stack.config)
    assert _view_count(stack, view, "otel_logs", _case_logs(seed, ALICE)) == 1

    # The withdraw path: alice leaves the roster; cut after each statement.
    without_alice = copy.deepcopy(stack.config)
    web = without_alice["modules"]["web_terminals"]
    web["users"] = [u for u in web["users"] if u["name"] != ALICE]
    withdrawn = _naming(
        accounts.render_account_statements(without_alice, passwords, live_accounts(stack.execute)),
        view,
    )
    assert withdrawn and withdrawn[-1].startswith("DROP USER"), withdrawn
    try:
        for cut in range(len(withdrawn)):
            stack.restage(stack.config)
            apply_script(stack.execute, withdrawn[:cut])
            assert _bob_rows_visible(stack, view) == 0, f"cut before: {withdrawn[cut]}"
    finally:
        stack.restage(stack.config)
    assert _view_count(stack, view, "otel_logs", _case_logs(seed, ALICE)) == 1


# --------------------------------------------------------------------------- #
# Preflight rows: the queueserver arm
# --------------------------------------------------------------------------- #


def test_a_preflight_row_on_alices_behalf_is_stored_and_read_only_on_the_preflight_arm(
    stack: Stack,
) -> None:
    """Stored as written; alice's view reads it through the narrowed queueserver arm.

    Whether the page then shows it (only when a queue-submit of alice's
    precedes it within 60 s) is the page's matcher, proven with the page.
    """
    tag = uuid.uuid4().hex[:12]
    now = datetime.datetime.now(datetime.UTC)
    planted = f"planted-{tag}"
    _ledger(
        stack,
        "queueserver",
        "preflight",
        _envelope(
            when=now,
            surface="preflight",
            actor="queueserver",
            subject=planted,
            decision="allowed",
            on_behalf_of=ALICE,
            posture_source=POSTURE_SOURCE_SPAWN,
        ),
    )
    # The same claim on any other surface is not on the arm.
    _ledger(
        stack,
        "queueserver",
        "queue",
        _envelope(
            when=now,
            surface="queue",
            actor="queueserver",
            subject=f"other-{tag}",
            decision="allowed",
            on_behalf_of=ALICE,
            posture_source=POSTURE_SOURCE_SPAWN,
        ),
    )
    stored = f"SELECT count() FROM telemetry.audit_ledger WHERE User = 'queueserver' AND Body LIKE '%{tag}%'"
    assert _wait_count(stored, 2) == 2
    assert (
        _scalar(
            "SELECT count() FROM telemetry.audit_ledger WHERE User = 'queueserver' "
            f"AND Surface = 'preflight' AND OnBehalfOf = '{ALICE}' AND Body LIKE '%{planted}%'"
        )
        == 1
    )

    window = {
        "from": now - datetime.timedelta(minutes=5),
        "to": now + datetime.timedelta(minutes=5),
    }
    own = _rows(
        _contract(
            stack,
            accounts.person_view_user(ALICE),
            "preflight",
            {"queueservers": ["queueserver"], "user": ALICE, **window},
        )
    )
    assert [r["Surface"] for r in own if planted in r["body"]] == ["preflight"]
    assert (
        _view_count(
            stack, accounts.person_view_user(ALICE), "audit_ledger", f"Body LIKE '%other-{tag}%'"
        )
        == 0
    )
    assert (
        _view_count(stack, accounts.person_view_user(BOB), "audit_ledger", f"Body LIKE '%{tag}%'")
        == 0
    )


# --------------------------------------------------------------------------- #
# The recipe: a replayed CLI session, reconstructed as a reader
# --------------------------------------------------------------------------- #


def _attrs(record: Mapping[str, Any]) -> dict[str, Any]:
    return {a["key"]: next(iter(a["value"].values()), "") for a in record.get("attributes", [])}


def _log_records(capture: Mapping[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any], int]]:
    """(resource attributes, record attributes, time ns) for every log record."""
    found = []
    for request in capture["requests"]:
        for resource_logs in request["body"].get("resourceLogs", []):
            resource = _attrs(resource_logs.get("resource", {}))
            for scope in resource_logs.get("scopeLogs", []):
                for record in scope.get("logRecords", []):
                    found.append((resource, _attrs(record), int(record["timeUnixNano"])))
    return found


def _reader_recipe(stack: Stack, reader: str, user: str, session: str) -> list[dict[str, Any]]:
    """Every row of the reconstruct-session recipe, paged to the end, as *reader*."""
    sql = queries.load(queries.RECIPE) + "\nFORMAT JSONEachRow"
    cursor: dict[str, Any] = {"after_at": queries.FIRST_CURSOR, "after_source": "", "after_hash": 0}
    found: list[dict[str, Any]] = []
    for _ in range(100):
        response = _http(
            stack,
            reader,
            stack.reader_password(reader),
            sql,
            params={
                **queries.query_parameters(
                    queries.RECIPE, {"user": user, "session": session, **cursor}
                ),
                "http_wait_end_of_query": "1",
            },
        )
        assert response.status_code == 200, f"{response.status_code}: {response.text[:600]}"
        page = _rows(response)
        found += page
        if len(page) < queries.PAGE_LIMIT:
            return found
        last = page[-1]
        cursor = {
            "after_at": last["at"],
            "after_source": last["source"],
            "after_hash": last["row_hash"],
        }
    raise AssertionError("the recipe never reached its last page")


def test_the_replayed_cli_session_is_reconstructed_by_a_reader(stack: Stack) -> None:
    capture = rebase_otlp_times(
        json.loads(FIXTURE.read_text(encoding="utf-8")), datetime.datetime.now(datetime.UTC)
    )
    records = _log_records(capture)
    sessions = {attrs["session.id"] for _, attrs, _ in records if attrs.get("session.id")}
    assert len(sessions) == 1, sessions
    (session,) = sessions
    terminal = records[0][0]["terminal.session"]
    prompts = [r for r in records if r[1].get("event.name") == queries.USER_PROMPT_EVENT]
    tools = {
        attrs["tool_use_id"]: (
            attrs.get("tool_name", "tool"),
            at,
            float(attrs.get("duration_ms") or 50),
        )
        for _, attrs, at in records
        if attrs.get("event.name") == "tool_result" and attrs.get("tool_use_id")
    }
    assert prompts and tools, "the capture lacks a prompt or a tool result"

    for request in capture["requests"]:
        _post(stack, ALICE, request["signal"], request["body"])

    first_at = None
    for tool_use_id, (tool_name, at, duration) in tools.items():
        when = datetime.datetime.fromtimestamp(at / 1e9, datetime.UTC)
        first_at = first_at or when
        _ledger(
            stack,
            ALICE,
            "workspace",
            _tool_call(
                when=when,
                actor=ALICE,
                terminal=terminal,
                session_id=session,
                tool_use_id=tool_use_id,
                subject=tool_name,
                duration_ms=duration,
            ),
        )
        _ledger(
            stack,
            ALICE,
            "hook_approval",
            _envelope(
                when=when - datetime.timedelta(milliseconds=duration + 10),
                surface="hook_approval",
                actor=ALICE,
                subject=tool_name,
                decision="approved",
                session=terminal,
                tool_use_id=tool_use_id,
                posture_source=POSTURE_SOURCE_SPAWN,
            ),
        )
    assert first_at is not None
    _ledger(
        stack,
        "queueserver",
        "preflight",
        _envelope(
            when=first_at + datetime.timedelta(seconds=1),
            surface="preflight",
            actor="queueserver",
            subject=f"plan-{session}",
            decision="allowed",
            on_behalf_of=ALICE,
            posture_source=POSTURE_SOURCE_SPAWN,
        ),
    )

    assert _wait_count(
        f"SELECT count() FROM telemetry.otel_logs WHERE User = '{ALICE}' AND SessionId = '{session}'",
        len(records),
    ) == len(records)
    assert _wait_count(
        f"SELECT count() FROM telemetry.audit_ledger WHERE User = '{ALICE}' AND ToolUseId IN "
        f"({', '.join(repr(t) for t in tools)})",
        2 * len(tools),
    ) == 2 * len(tools)
    assert (
        _wait_count(
            f"SELECT count() FROM telemetry.audit_ledger WHERE Surface = 'preflight' AND Body LIKE '%plan-{session}%'",
            1,
        )
        == 1
    )

    rows = _reader_recipe(stack, READER_Y, ALICE, session)
    kinds = [(r["source"], r["kind"]) for r in rows]
    assert sum(1 for s, _ in kinds if s == "event") == len(records), kinds
    assert kinds.count(("event", queries.USER_PROMPT_EVENT)) == len(prompts)
    assert {r["tool_use_id"] for r in rows if r["source"] == "tool_call"} == set(tools)
    assert {r["tool_use_id"] for r in rows if r["source"] == "approval"} == set(tools)
    assert [r["kind"] for r in rows if r["source"] == "preflight"] == ["allowed"]
    assert [r["at"] for r in rows] == sorted(r["at"] for r in rows)


# --------------------------------------------------------------------------- #
# Volume: long sessions and a busy store
# --------------------------------------------------------------------------- #


def _insert_log_rows(select: str) -> None:
    _sql(
        "INSERT INTO telemetry.otel_logs "
        "(Timestamp, ServiceName, Body, ResourceAttributes, LogAttributes) " + select
    )


def test_a_2000_event_session_with_twenty_512_kib_results_pages_to_the_end(stack: Stack) -> None:
    session = f"long-{uuid.uuid4().hex[:8]}"
    terminal = f"t-{session}"
    start = "now64(9) - INTERVAL 2 HOUR"
    big_every = LONG_SESSION_TOOL_CALLS // BIG_RESULTS
    _sql(
        "INSERT INTO telemetry.audit_ledger "
        "(Timestamp, ServiceName, Body, ResourceAttributes, LogAttributes) "
        f"SELECT {start} + toIntervalMillisecond(number * 100), 'e2e', "
        'concat(\'{"ts":"x","surface":"tool_call","tool_use_id":"\', '
        f"concat('toolu_{session}_', toString(number)), '\",\"result\":\"', "
        f"if(number % {big_every} = 0, repeat('r', {BIG_RESULT_BYTES}), 'ok'), '\"}}'), "
        f"map('osprey.user', '{ALICE}'), "
        f"map('osprey.audit.surface', 'tool_call', 'osprey.audit.session_id', '{session}', "
        f"'osprey.audit.session', '{terminal}', "
        f"'osprey.audit.tool_use_id', concat('toolu_{session}_', toString(number)), "
        "'osprey.audit.duration_ms', '40', 'osprey.audit.decision', 'allowed') "
        f"FROM numbers({LONG_SESSION_TOOL_CALLS})"
    )
    _insert_log_rows(
        f"SELECT {start} + toIntervalMillisecond(number * 200 + 50), 'e2e', concat('event-', toString(number)), "
        f"map('osprey.user', '{ALICE}', 'terminal.session', '{terminal}'), "
        f"map('session.id', '{session}', 'event.name', "
        "if(number % 2 = 0, 'user_prompt', 'assistant_response'), "
        "if(number % 2 = 0, 'prompt', 'response'), concat('text-', toString(number))) "
        f"FROM numbers({LONG_SESSION_EVENTS})"
    )

    view = accounts.person_view_user(ALICE)
    cursor: dict[str, Any] = {"after_ts": queries.FIRST_CURSOR, "after_hash": 0}
    seen: list[dict[str, Any]] = []
    pages = 0
    while True:
        response = _contract(stack, view, "tool_log", {"user": ALICE, "session": session, **cursor})
        page = _rows(response)
        pages += 1
        seen += page
        if len(page) < queries.PAGE_LIMIT:
            break
        cursor = {"after_ts": page[-1]["Timestamp"], "after_hash": page[-1]["body_hash"]}
        assert pages < 20, "the tool log never reached its last page"
    assert len({r["ToolUseId"] for r in seen}) == LONG_SESSION_TOOL_CALLS
    big = [r for r in seen if int(r["body_length"]) > BIG_RESULT_BYTES]
    assert len(big) == BIG_RESULTS
    assert all(len(r["body"]) <= queries.PREVIEW_CHARS for r in seen)

    rows = _reader_recipe(stack, READER_Y, ALICE, session)
    assert len(rows) == LONG_SESSION_TOOL_CALLS + LONG_SESSION_EVENTS
    assert len(rows) > tc.store_limits(stack.config)["result_rows"]


def _seed_probe_days(now: datetime.datetime) -> int:
    """14 probe days of agent events, one INSERT per day; returns the sessions written.

    Each day ends half an hour before *now* and spans 22 hours, so the oldest
    row stays a day inside the retention window.
    """
    block = sum(SESSION_SIZES)
    bounds = [sum(SESSION_SIZES[: i + 1]) for i in range(len(SESSION_SIZES))]
    which = (
        "multiIf("
        + ", ".join(f"o < {b}, {i}" for i, b in enumerate(bounds[:-1]))
        + f", {len(bounds) - 1})"
    )
    starts = "[" + ", ".join(str(b - s) for b, s in zip(bounds, SESSION_SIZES, strict=True)) + "]"
    others = "[" + ", ".join(f"'{e}'" for e in OTHER_EVENTS) + "]"
    usable_ns = (86_400 - 2 * 3600) * 1_000_000_000
    for day in range(LOAD_DAYS):
        newest = now - datetime.timedelta(days=day, minutes=30)
        _insert_log_rows(
            f"SELECT toDateTime64('{newest:%Y-%m-%d %H:%M:%S}', 9, 'UTC') "
            f"- toIntervalNanosecond(intDiv(n * {usable_ns}, {PROBE_DAY_ROWS})), 'e2e', 'load', "
            f"map('osprey.user', concat('load', toString((intDiv(n, {block}) * {len(SESSION_SIZES)} + k) % {LOAD_USERS}))), "
            f"map('session.id', concat('load-{day}-', toString(intDiv(n, {block})), '-', toString(k)), "
            f"'event.name', if((o - {starts}[k + 1]) % {PROMPT_EVERY} = 0, 'user_prompt', "
            f"{others}[(o % {len(OTHER_EVENTS)}) + 1])) "
            f"FROM (SELECT number AS n, n % {block} AS o, {which} AS k FROM numbers({PROBE_DAY_ROWS}))"
        )
    return _scalar("SELECT uniqExact(User, SessionId) FROM telemetry.otel_logs WHERE Body = 'load'")


def _read_rows(comment: str) -> int:
    _sql("SYSTEM FLUSH LOGS")
    return _scalar(
        "SELECT max(read_rows) FROM system.query_log "
        f"WHERE type = 'QueryFinish' AND log_comment = '{comment}'"
    )


def test_the_session_list_at_14_probe_days_reads_fewer_rows_than_the_cap(stack: Stack) -> None:
    now = datetime.datetime.now(datetime.UTC).replace(microsecond=0)
    written = _seed_probe_days(now)
    assert written > 1_000
    assert _scalar(
        "SELECT count() FROM telemetry.otel_logs WHERE ServiceName = 'e2e' AND Body = 'load'"
    ) == (PROBE_DAY_ROWS * LOAD_DAYS)
    # Whole days elapsed, not calendar-day boundaries crossed: the oldest row
    # is under 14 x 24 hours old at any time of day.
    oldest_day = _scalar(
        "SELECT intDiv(dateDiff('second', min(Timestamp), now64(9)), 86400) "
        "FROM telemetry.otel_logs WHERE Body = 'load'"
    )
    assert oldest_day <= tc.retention_days(stack.config) - 1

    cap = tc.store_limits(stack.config)["rows_read"]
    view = accounts.VIEW_ALL_USER
    to = now + datetime.timedelta(minutes=1)
    window = {"from": to - datetime.timedelta(days=queries.SESSION_LIST_WINDOW_DAYS), "to": to}
    cursor: dict[str, Any] = {"cursor_ts": to, "cursor_user": "", "cursor_session": ""}
    run = uuid.uuid4().hex[:8]
    listed = 0
    page_no = 0
    read: list[int] = []
    while True:
        page_no += 1
        comment = f"load-{run}-{page_no}:{view}"
        page = _rows(
            _contract(
                stack,
                view,
                "session_list",
                {**window, **cursor},
                person="load",
                log_comment=comment,
            )
        )
        listed += len(page)
        read.append(_read_rows(comment))
        if len(page) < queries.SESSION_LIST_LIMIT:
            break
        last = page[-1]
        cursor = {
            "cursor_ts": last["last_ts"],
            "cursor_user": last["User"],
            "cursor_session": last["SessionId"],
        }
        assert page_no < 100, "the session list never reached its last page"
    assert listed > 1_000, f"only {listed} sessions in the default window"
    assert read[0] < cap, f"page 1 read {read[0]} rows; the cap is {cap}"
    assert read[-1] < cap, f"the last page read {read[-1]} rows; the cap is {cap}"


# =========================================================================== #
# Readers: refusals, scope, limits and the account lifecycle
# =========================================================================== #
#
# A reader reaches the store directly here, with its key, the way a script
# does; the readers' origin in front of it is proven in `full-chain-auth-e2e`.
# The cases that re-stage the accounts run them the way `osprey up` does
# (mint the key files, then the accounts script) and restore the module's
# config before they return. The reset case is last: it removes the store's
# volume.

#: The case marker of the rows the limit cases read.
LIMIT_CASE = f"limits-{uuid.uuid4().hex[:8]}"
#: Rows of :data:`LIMIT_CASE`, and the body size of the rows the byte cap reads.
LIMIT_ROWS = 1_500
BIG_ROWS = 900
BIG_BODY_BYTES = 12_500
#: How long one held slot lasts: this many three-second sleeps.
HOLD_SLEEPS = 5
#: Readers this section adds and withdraws again.
READER_A = "a"
READER_B = "b"
READER_Z = "z"
READER_HAND = "hand"
READER_ROTATE = "rot"
READER_BURST = "burst"


def _reader(
    stack: Stack, name: str, query: str, *, params: Mapping[str, str] | None = None, **kw: Any
) -> httpx.Response:
    return _http(stack, name, stack.reader_password(name), query, params=params, **kw)


def _key_response(stack: Stack, name: str, password: str) -> httpx.Response:
    return _http(stack, name, password, "SELECT currentUser()")


def _authenticates(stack: Stack, name: str, password: str) -> bool:
    response = _key_response(stack, name, password)
    if response.status_code == 200:
        assert response.text.strip() == name, response.text
        return True
    assert "Code: 516" in response.text, f"{response.status_code}: {response.text[:300]}"
    return False


def _refused_named(response: httpx.Response, name: str) -> None:
    assert response.status_code != 200, f"expected {name}, got 200: {response.text[:300]}"
    assert f"({name})" in response.text, f"expected {name}: {response.text[:600]}"


def _count_of(response: httpx.Response) -> int:
    assert response.status_code == 200, f"{response.status_code}: {response.text[:600]}"
    return int(response.text.strip())


def _slow(sleeps: int, marker: str) -> str:
    """A reader query that takes ``3 * sleeps`` seconds before it reads a row.

    Each scalar subquery runs in turn while the query is planned, inside the
    query's own time limit; the distinct offsets keep the store from reusing
    one subquery's result for the next.
    """
    waits = " AND ".join(f"(SELECT sleep(3) + {i}) = {i}" for i in range(sleeps))
    return (
        f"SELECT count() AS n, '{marker}' AS marker FROM telemetry.otel_logs "
        f"WHERE Timestamp > now() - INTERVAL 1 HOUR AND {waits}"
    )


def _reader_config(stack: Stack, *entries: Mapping[str, Any], drop: tuple[str, ...] = ()) -> dict:
    config = copy.deepcopy(stack.config)
    store = config["services"]["telemetry_store"]
    kept = [r for r in store["readers"] if r["name"] not in drop]
    names = {e["name"] for e in entries}
    store["readers"] = [r for r in kept if r["name"] not in names] + [dict(e) for e in entries]
    return config


def _access_snapshot() -> dict[str, list[str]]:
    """The schema and every access entity, as the admin reads them."""
    queries_ = {
        "tables": "SELECT database, name, engine, create_table_query FROM system.tables "
        "WHERE database NOT IN ('system', 'INFORMATION_SCHEMA', 'information_schema') "
        "ORDER BY database, name",
        "users": "SELECT name, auth_type, host_ip, host_names, default_roles_list, default_database "
        "FROM system.users ORDER BY name",
        "grants": "SELECT * FROM system.grants ORDER BY ALL",
        "row_policies": "SELECT name, database, table, select_filter, is_restrictive, apply_to_list "
        "FROM system.row_policies ORDER BY name",
    }
    return {key: _sql(q, fmt="JSONEachRow").splitlines() for key, q in queries_.items()}


#: FR3's refused list as a reader sends it, and the error the store names.
REFUSED_STATEMENTS: tuple[tuple[str, str], ...] = (
    ("SELECT * FROM remote('127.0.0.1', system.one)", "ACCESS_DENIED"),
    ("SELECT * FROM cluster('default', system.one)", "ACCESS_DENIED"),
    ("SELECT * FROM url('http://127.0.0.1:8123/', CSV, 'a String')", "ACCESS_DENIED"),
    ("SELECT * FROM file('x.csv', CSV, 'a String')", "ACCESS_DENIED"),
    ("SELECT * FROM loop(telemetry.otel_logs) LIMIT 1", "ACCESS_DENIED"),
    ("SELECT * FROM executable('x.sh', CSV, 'a String')", "ACCESS_DENIED"),
    ("SELECT * FROM mergeTreeIndex(telemetry, otel_logs) LIMIT 1", "ACCESS_DENIED"),
    ("KILL QUERY WHERE 1", "ACCESS_DENIED"),
    (f"EXECUTE AS `{accounts.VIEW_ALL_USER}` SELECT 1", "ACCESS_DENIED"),
    ("SET readonly = 0", "READONLY"),
    ("SELECT 1 SETTINGS max_execution_time = 600", "SETTING_CONSTRAINT_VIOLATION"),
    ("INSERT INTO telemetry.otel_logs (Body) VALUES ('x')", "ACCESS_DENIED"),
    ("ALTER TABLE telemetry.otel_logs DELETE WHERE 1", "ACCESS_DENIED"),
    ("TRUNCATE TABLE telemetry.otel_logs", "ACCESS_DENIED"),
    ("DROP TABLE telemetry.otel_logs", "ACCESS_DENIED"),
    ("CREATE TABLE telemetry.e2e_x (a UInt8) ENGINE = Memory", "ACCESS_DENIED"),
    ("CREATE USER e2e_x", "ACCESS_DENIED"),
    (f"GRANT SELECT ON telemetry.* TO `{accounts.READERS_ROLE}`", "ACCESS_DENIED"),
    ("SYSTEM FLUSH LOGS", "ACCESS_DENIED"),
)


def _run_refusals(stack: Stack, reader: str) -> list[str]:
    """Send every refused statement as *reader*; return their query ids."""
    ids: list[str] = []
    wrong: list[str] = []
    for statement, error in REFUSED_STATEMENTS:
        query_id = f"refused-{uuid.uuid4().hex}"
        response = _reader(stack, reader, statement, params={"query_id": query_id})
        if response.status_code == 200 or f"({error})" not in response.text:
            wrong.append(f"{statement!r} -> {response.status_code} {response.text[:200]!r}")
        ids.append(query_id)
    assert not wrong, "\n".join(wrong)
    return ids


def test_each_refused_statement_is_refused_by_name_and_changes_no_access_entity(
    stack: Stack,
) -> None:
    before = _access_snapshot()
    _run_refusals(stack, READER_Y)
    assert _access_snapshot() == before


def test_subquery_join_merge_view_and_days_back_read_no_out_of_scope_row_and_are_logged(
    stack: Stack, seed: Seed, limit_rows: str
) -> None:
    assert limit_rows
    old = f"LogAttributes['e2e.case'] = '{seed.old_case}'"
    tag = uuid.uuid4().hex[:10]
    shapes = {
        "subquery": f"SELECT count() FROM (SELECT * FROM telemetry.otel_logs WHERE {old})",
        "join": "SELECT count() FROM telemetry.otel_logs AS a INNER JOIN telemetry.otel_logs AS b "
        f"ON a.Timestamp = b.Timestamp WHERE a.{old}",
        "merge": f"SELECT count() FROM merge('telemetry', '^otel_logs$') WHERE {old}",
        "view": f"SELECT count() FROM view(SELECT * FROM telemetry.otel_logs WHERE {old})",
        "days_back": f"SELECT count() FROM telemetry.otel_logs WHERE {old}",
    }
    assert _scalar(f"SELECT count() FROM telemetry.otel_logs WHERE {old}") == 1
    for shape, query in shapes.items():
        response = _reader(stack, READER_X, f"{query} /* {tag}-{shape} */")
        assert _count_of(response) == 0, shape
    # The same row, through every shape, for the reader whose days_back holds it.
    assert _count_of(_reader(stack, READER_Y, shapes["days_back"])) == 1
    _sql("SYSTEM FLUSH LOGS")
    logged = _sql(
        "SELECT DISTINCT extract(query, '/\\\\* ([^ ]+) \\\\*/') FROM system.query_log "
        f"WHERE user = '{READER_X}' AND type = 'QueryFinish' AND query LIKE '%{tag}-%'"
    ).split()
    assert sorted(logged) == sorted(f"{tag}-{shape}" for shape in shapes)


@pytest.fixture(scope="module")
def limit_rows(stack: Stack) -> str:  # noqa: ARG001 - the rows need the running store
    """:data:`LIMIT_ROWS` small rows and :data:`BIG_ROWS` large ones, written as the admin.

    The session-list load above leaves fourteen days of rows behind; a reader
    query that filters on an attribute reads past the rows-read cap through
    them, so they go first.
    """
    _sql(
        "ALTER TABLE telemetry.otel_logs DELETE WHERE ServiceName = 'e2e' AND Body = 'load' "
        "SETTINGS mutations_sync = 2"
    )
    _insert_log_rows(
        f"SELECT now64(9), 'e2e', 'row', map('osprey.user', '{ALICE}'), "
        f"map('e2e.case', '{LIMIT_CASE}') FROM numbers({LIMIT_ROWS})"
    )
    _insert_log_rows(
        f"SELECT now64(9), 'e2e', repeat('b', {BIG_BODY_BYTES}), map('osprey.user', '{ALICE}'), "
        f"map('e2e.case', '{LIMIT_CASE}-big') FROM numbers({BIG_ROWS})"
    )
    return LIMIT_CASE


def test_limit_1001_an_11_mb_result_and_a_31_s_query_are_refused_with_the_limit_named(
    stack: Stack, limit_rows: str
) -> None:
    where = f"LogAttributes['e2e.case'] = '{limit_rows}'"
    assert (
        len(
            _rows(
                _reader(
                    stack,
                    READER_Y,
                    f"SELECT Body FROM telemetry.otel_logs WHERE {where} LIMIT 1000 FORMAT JSONEachRow",
                )
            )
        )
        == 1000
    )
    _refused_named(
        _reader(stack, READER_Y, f"SELECT Body FROM telemetry.otel_logs WHERE {where} LIMIT 1001"),
        "TOO_MANY_ROWS_OR_BYTES",
    )
    big = _reader(
        stack,
        READER_Y,
        f"SELECT Body FROM telemetry.otel_logs WHERE LogAttributes['e2e.case'] = '{limit_rows}-big'",
    )
    _refused_named(big, "TOO_MANY_ROWS_OR_BYTES")
    assert "max bytes" in big.text, big.text[:300]

    started = time.monotonic()
    slow = _reader(stack, READER_Y, _slow(11, "timeout"))
    _refused_named(slow, "TIMEOUT_EXCEEDED")
    assert 30 <= time.monotonic() - started < 45


def _quota_queries(reader: str) -> int:
    out = _sql(
        f"SELECT sum(queries) FROM system.quotas_usage WHERE quota_name = '{accounts.READER_QUOTA}' "
        f"AND quota_key = '{reader}'"
    ).strip()
    return int(out or 0) if out != "\\N" else 0


def _at_a_fresh_minute() -> datetime.datetime:
    """Sleep to just past the next clock minute, the quota interval's boundary."""
    now = datetime.datetime.now(datetime.UTC)
    time.sleep(60 - now.second - now.microsecond / 1e6 + 0.5)
    return datetime.datetime.now(datetime.UTC)


def test_the_61st_query_in_a_minute_is_refused_by_the_quota_started_before_second_20(
    stack: Stack,
) -> None:
    per_minute = tc.store_limits(stack.config)["per_minute"]
    read = "SELECT count() FROM telemetry.otel_logs WHERE Timestamp > now() - INTERVAL 1 MINUTE"
    # A reader of its own, so no other case's queries share its minute.
    try:
        stack.restage(_reader_config(stack, {"name": READER_BURST}))
        started = datetime.datetime.now(datetime.UTC)
        if started.second >= 20:
            started = _at_a_fresh_minute()
        assert started.second < 20, started
        for n in range(per_minute):
            response = _reader(stack, READER_BURST, read)
            assert response.status_code == 200, f"query {n + 1}: {response.text[:300]}"
        over = _reader(stack, READER_BURST, read)
        crossed = datetime.datetime.now(datetime.UTC).minute != started.minute
        assert not crossed, "the burst crossed a minute"
        _refused_named(over, "QUOTA_EXCEEDED")
        assert f"queries = {per_minute + 1}/{per_minute}" in over.text, over.text[:300]
    finally:
        stack.restage(stack.config)


def test_the_from_less_shape_returns_more_than_1000_rows_with_200_and_leaves_the_quota(
    stack: Stack,
) -> None:
    """FR4's documented exception, pinned so an image that closes it is noticed."""
    now = datetime.datetime.now(datetime.UTC)
    if now.second > 45:
        _at_a_fresh_minute()
    before = _quota_queries(READER_X)
    response = _reader(stack, READER_X, "SELECT arrayJoin(range(5000)) AS n")
    assert response.status_code == 200, response.text[:300]
    assert len(response.text.split()) == 5000
    assert _quota_queries(READER_X) == before


def _spawn(pool: Any, stack: Stack, reader: str, query: str) -> Any:
    def run() -> tuple[httpx.Response, float]:
        response = _reader(stack, reader, query)
        return response, time.monotonic()

    return pool.submit(run)


def _wait_running(reader: str, count: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _scalar(f"SELECT count() FROM system.processes WHERE user = '{reader}'") >= count:
            return
        time.sleep(0.2)
    raise AssertionError(f"{reader} never had {count} queries running")


def test_a_third_query_is_refused_per_reader_while_another_reader_holds_a_slot(
    stack: Stack,
) -> None:
    import concurrent.futures

    limit = tc.store_limits(stack.config)["concurrent"]
    assert limit == 2
    # A pool one wider than one reader's limit, so the other reader's query
    # and this reader's two all run: the refusal is then the per-reader one.
    wider = copy.deepcopy(stack.config)
    wider["services"]["telemetry_store"].setdefault("limits", {})["reader_slots"] = limit + 1
    try:
        stack.restage(wider)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            other = _spawn(pool, stack, READER_Y, _slow(HOLD_SLEEPS + 3, "other"))
            _wait_running(READER_Y, 1)
            held = [
                _spawn(pool, stack, READER_X, _slow(HOLD_SLEEPS, f"held-{i}")) for i in range(limit)
            ]
            _wait_running(READER_X, limit)
            running = _sql(
                "SELECT user, count() FROM system.processes "
                f"WHERE user IN ('{READER_X}', '{READER_Y}') GROUP BY user ORDER BY user"
            ).split()
            assert running == [READER_X, str(limit), READER_Y, "1"], running
            third = _reader(stack, READER_X, "SELECT count() FROM telemetry.otel_logs")
            _refused_named(third, "TOO_MANY_SIMULTANEOUS_QUERIES")
            assert f"for user {READER_X}" in third.text, third.text[:300]
            for future in (other, *held):
                assert future.result(timeout=120)[0].status_code == 200
    finally:
        stack.restage(stack.config)


def test_a_query_beyond_the_shared_pool_waits_while_an_archive_query_still_runs(
    stack: Stack,
) -> None:
    """Two readers fill the pool; a third reader query waits, the archive's does not."""
    import concurrent.futures

    assert tc.store_limits(stack.config)["reader_slots"] == 2
    archive_password = stack.file_value(creds.ARCHIVE_FILE, creds.ARCHIVE_PASSWORD_ENV)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        holders = [
            _spawn(pool, stack, READER_X, _slow(HOLD_SLEEPS, "pool-x")),
            _spawn(pool, stack, READER_Y, _slow(HOLD_SLEEPS, "pool-y")),
        ]
        _wait_running(READER_X, 1)
        _wait_running(READER_Y, 1)
        sent = time.monotonic()
        beyond = _spawn(pool, stack, READER_Y, "SELECT count() FROM telemetry.otel_logs")
        archive_started = time.monotonic()
        archive = _http(
            stack,
            accounts.ARCHIVE_USER,
            archive_password,
            "SELECT count() FROM telemetry.otel_logs",
        )
        archive_took = time.monotonic() - archive_started
        assert archive.status_code == 200, archive.text[:300]
        assert archive_took < 3 * HOLD_SLEEPS / 2, f"the archive waited {archive_took:.1f} s"
        # Read before the result is awaited: a query that waited is still open here.
        still_open = not beyond.done()
        response, answered = beyond.result(timeout=120)
        ended = min(future.result(timeout=120)[1] for future in holders)
        if response.status_code == 200:
            assert still_open, "the query beyond the pool did not wait for a slot"
            assert answered >= ended - 0.5, "the query beyond the pool ran before a slot freed"
            assert answered - sent >= 3 * HOLD_SLEEPS - 5
        else:
            assert "readers" in response.text, response.text[:300]


def test_session_id_and_an_encoded_quota_key_change_neither_scope_nor_limits(
    stack: Stack, seed: Seed, limit_rows: str
) -> None:
    for params in ({"session_id": f"x-{uuid.uuid4().hex[:8]}"}, {"quota%5Fkey": READER_Y}):
        # httpx would encode the `%` again; the raw query string is what the store parses.
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{stack.store_url}?{query_string}"
        headers = {
            "X-ClickHouse-User": READER_X,
            "X-ClickHouse-Key": stack.reader_password(READER_X),
        }

        def ask(sql: str, url: str = url, headers: dict[str, str] = headers) -> httpx.Response:
            return httpx.post(url, content=sql.encode(), headers=headers, timeout=HTTP_TIMEOUT_S)

        assert ask("SELECT currentUser()").text.strip() == READER_X
        assert (
            _count_of(
                ask(
                    "SELECT count() FROM telemetry.otel_logs "
                    f"WHERE LogAttributes['e2e.case'] = '{seed.old_case}'"
                )
            )
            == 0
        )
        _refused_named(ask("SELECT count() FROM telemetry.otel_traces"), "ACCESS_DENIED")
        _refused_named(
            ask(
                "SELECT Body FROM telemetry.otel_logs "
                f"WHERE LogAttributes['e2e.case'] = '{limit_rows}' LIMIT 1001"
            ),
            "TOO_MANY_ROWS_OR_BYTES",
        )


def test_a_reader_narrowed_to_one_table_gets_497_on_the_other_and_its_policy_is_gone(
    stack: Stack, seed: Seed
) -> None:
    query = f"SELECT count() FROM telemetry.audit_ledger WHERE Body LIKE '%{seed.case}%'"
    assert _count_of(_reader(stack, READER_X, query)) == 4
    try:
        stack.restage(
            _reader_config(
                stack, {"name": READER_X, "tables": ["otel_logs"], "days_back": READER_X_DAYS}
            )
        )
        _refused_named(_reader(stack, READER_X, query), "ACCESS_DENIED")
        assert _row_policies(f"reader_{READER_X}_") == {f"reader_{READER_X}_otel_logs"}
    finally:
        stack.restage(stack.config)
    assert _count_of(_reader(stack, READER_X, query)) == 4


def test_a_withdrawn_reader_gets_516_and_re_adding_the_name_mints_a_new_key(
    stack: Stack,
) -> None:
    config = _reader_config(stack, {"name": READER_A})
    stack.restage(config)
    old = stack.reader_password(READER_A)
    assert _authenticates(stack, READER_A, old)
    try:
        stack.restage(stack.config)
        assert not _authenticates(stack, READER_A, old)
        assert not (stack.repo / creds.reader_file(READER_A)).exists()
        withdrawn = list(stack.repo.glob(f"{creds.reader_file(READER_A)}.withdrawn-*"))
        assert len(withdrawn) == 1, withdrawn

        stack.restage(config)
        new = stack.reader_password(READER_A)
        assert new != old
        assert _authenticates(stack, READER_A, new)
        assert not _authenticates(stack, READER_A, old)
    finally:
        stack.restage(stack.config)
    assert not _authenticates(
        stack, READER_A, stack.file_value(withdrawn[0].name, creds.READER_KEY)
    )


def test_a_rotated_reader_key_leaves_the_old_key_516_and_the_new_file_key_working(
    stack: Stack,
) -> None:
    config = _reader_config(stack, {"name": READER_ROTATE})
    try:
        stack.restage(config)
        old = stack.reader_password(READER_ROTATE)
        assert _authenticates(stack, READER_ROTATE, old)
        (stack.repo / creds.reader_file(READER_ROTATE)).unlink()
        stack.restage(config)
        new = stack.reader_password(READER_ROTATE)
        assert new != old
        assert not _authenticates(stack, READER_ROTATE, old)
        assert _authenticates(stack, READER_ROTATE, new)
    finally:
        stack.restage(stack.config)


def test_a_hand_made_user_of_a_reader_name_is_refused_by_name_and_never_adopted(
    stack: Stack,
) -> None:
    from osprey.deployment.telemetry_store.stage import TelemetryStoreError

    hand_password = uuid.uuid4().hex
    _sql(f"CREATE USER `{READER_HAND}` IDENTIFIED WITH sha256_password BY '{hand_password}'")
    try:
        with pytest.raises(TelemetryStoreError, match=f"reader '{READER_HAND}'"):
            stack.restage(_reader_config(stack, {"name": READER_HAND}))
        assert _authenticates(stack, READER_HAND, hand_password)
        roles = _sql(
            f"SELECT granted_role_name FROM system.role_grants WHERE user_name = '{READER_HAND}'"
        ).split()
        assert accounts.READERS_ROLE not in roles
        assert _row_policies(f"reader_{READER_HAND}_") == set()
    finally:
        _sql(f"DROP USER IF EXISTS `{READER_HAND}`")
        stack.restage(stack.config)


def test_the_script_cut_after_any_statement_for_a_new_reader_leaves_no_out_of_scope_row_readable(
    stack: Stack,
) -> None:
    """Reader z, ``audit_ledger`` only and 3 days back, cut after each of its statements."""
    tag = uuid.uuid4().hex[:10]
    _sql(
        "INSERT INTO telemetry.audit_ledger (Timestamp, ServiceName, Body, ResourceAttributes, "
        f"LogAttributes) SELECT now64(9) - INTERVAL 4 DAY, 'e2e', 'old-{tag}', "
        f"map('osprey.user', '{ALICE}'), map('osprey.audit.surface', 'e2e')"
    )
    config = _reader_config(stack, {"name": READER_Z, "tables": ["audit_ledger"], "days_back": 3})
    creds.ensure_telemetry_credentials(stack.repo, config, reader_keys=True)
    passwords = accounts.read_passwords(stack.repo, config)
    statements = _naming(
        accounts.render_account_statements(config, passwords, live_accounts(stack.execute)),
        READER_Z,
    )
    assert len(statements) >= 4, statements

    def out_of_scope() -> int:
        seen = 0
        for query in (
            f"SELECT count() FROM telemetry.audit_ledger WHERE Body = 'old-{tag}'",
            "SELECT count() FROM telemetry.otel_logs",
            "SELECT count() FROM telemetry.otel_traces",
        ):
            response = _reader(stack, READER_Z, query)
            if response.status_code == 200:
                seen += int(response.text.strip())
            else:
                assert "Code: 497" in response.text or "Code: 516" in response.text, response.text[
                    :300
                ]
        return seen

    try:
        for cut in range(1, len(statements) + 1):
            _sql(f"DROP USER IF EXISTS `{READER_Z}`")
            apply_script(stack.execute, statements[:cut])
            assert out_of_scope() == 0, f"cut after: {statements[cut - 1]}"
        assert _authenticates(stack, READER_Z, stack.reader_password(READER_Z))
    finally:
        stack.restage(stack.config)
    assert READER_Z not in _sql("SELECT name FROM system.users").split()


def test_removing_reader_a_beside_a_failing_create_for_new_reader_b_still_cuts_a_off(
    stack: Stack,
) -> None:
    from osprey.deployment.telemetry_store.stage import TelemetryStoreError

    with_a = _reader_config(stack, {"name": READER_A})
    stack.restage(with_a)
    key_a = stack.reader_password(READER_A)
    assert _authenticates(stack, READER_A, key_a)

    with_b = _reader_config(stack, {"name": READER_B}, drop=(READER_A,))
    creds.ensure_telemetry_credentials(stack.repo, with_b, reader_keys=True)
    passwords = accounts.read_passwords(stack.repo, with_b)
    script = accounts.render_account_statements(with_b, passwords, live_accounts(stack.execute))
    early = [s.sql for s in script if s.phase < accounts.Phase.CREATE]
    late = [s.sql for s in script if s.phase >= accounts.Phase.CREATE]
    first_b = next(i for i, sql in enumerate(late) if f"`{READER_B}`" in sql)
    late[first_b] = late[first_b].replace(
        f"SETTINGS PROFILE '{accounts.READER_PROFILE}'", "SETTINGS PROFILE 'no_such_profile'"
    )
    assert "no_such_profile" in late[first_b], late[first_b]
    try:
        apply_script(stack.execute, early)
        with pytest.raises(TelemetryStoreError):
            apply_script(stack.execute, late)
        assert not _authenticates(stack, READER_A, key_a)
    finally:
        stack.restage(stack.config)


def test_the_archive_key_gets_497_on_insert_and_164_on_set_readonly_0(stack: Stack) -> None:
    password = stack.file_value(creds.ARCHIVE_FILE, creds.ARCHIVE_PASSWORD_ENV)
    insert = _http(
        stack,
        accounts.ARCHIVE_USER,
        password,
        "INSERT INTO telemetry.otel_logs (Body) VALUES ('x')",
    )
    _refused_with(insert, 497)
    _refused_with(_http(stack, accounts.ARCHIVE_USER, password, "SET readonly = 0"), 164)


def test_an_archive_pass_after_the_refusal_suite_holds_the_refused_statements_by_query_id(
    stack: Stack, tmp_path: Path
) -> None:
    import duckdb

    from osprey.deployment.telemetry_store import ddl
    from osprey.services.archive.run import run_pass
    from osprey.services.archive.telemetry_export import QUERY_LOG, TelemetryExporter

    refused = _run_refusals(stack, READER_Y)
    _sql("SYSTEM FLUSH LOGS")
    today = datetime.datetime.now(datetime.UTC).date()
    sources, dest = tmp_path / "sources", tmp_path / "archive"
    sources.mkdir()
    dest.mkdir()
    exporter = TelemetryExporter(
        url=stack.store_url,
        backfill_days=1,
        environ={
            creds.ARCHIVE_PASSWORD_ENV: stack.file_value(
                creds.ARCHIVE_FILE, creds.ARCHIVE_PASSWORD_ENV
            )
        },
    )
    # The pass runs as if today had just closed: the explicit day is today.
    now = datetime.datetime.combine(
        today + datetime.timedelta(days=1), datetime.time(0, 20), tzinfo=datetime.UTC
    )
    result = run_pass(sources, dest, now=now, telemetry=exporter)
    assert not result.errors, result.errors
    assert today.isoformat() in result.telemetry_days

    def parquet(table: str) -> str:
        (path,) = dest.rglob(f"telemetry/{today.isoformat()}/{table}.parquet")
        return str(path)

    logged = {
        row[0]
        for row in duckdb.execute(
            "SELECT query_id FROM read_parquet(?) WHERE user = ?", [parquet(QUERY_LOG), READER_Y]
        ).fetchall()
    }
    missing = sorted(set(refused) - logged)
    assert not missing, f"{len(missing)} refused statements are not in the query_log export"
    for table in ddl.TABLES:
        exported = duckdb.execute(
            "SELECT count(*) FROM read_parquet(?)", [parquet(table.name)]
        ).fetchone()[0]
        counted = _scalar(
            f"SELECT count() FROM telemetry.{table.name} WHERE toDate(InsertedAt, 'UTC') = '{today}'"
        )
        assert exported == counted, f"{table.name}: exported {exported}, the store holds {counted}"


@pytest.mark.usefixtures("stack")
def test_the_store_was_not_oom_killed_during_the_suite() -> None:
    state = json.loads(_docker("inspect", "--format", "{{json .State}}", STORE_CONTAINER).stdout)
    assert state["Running"], state
    assert not state["OOMKilled"], state
    assert state["ExitCode"] != 137, state
    restarts = _docker("inspect", "--format", "{{.RestartCount}}", STORE_CONTAINER).stdout.strip()
    assert restarts == "0", f"the store restarted {restarts} times"


def _up_again(stack: Stack) -> None:
    """What `osprey up` does for the store and gateway: mint, stage, start."""
    _ensure_bluesky_control_plane_keys(stack.config, stack.repo / ".env")
    _ensure_bluesky_document_plane_certs(stack.config, stack.repo / ".env")
    creds.ensure_telemetry_credentials(stack.repo, stack.config, reader_keys=True)
    _stage_telemetry_store(stack.config, stack.compose_files, stack.env, stack.repo)
    stack.compose("up", "-d", tc.GATEWAY_SERVICE)
    _wait_ready(stack)


def _reader_files(repo: Path) -> dict[str, bytes]:
    return {name: (repo / creds.reader_file(name)).read_bytes() for name in (READER_X, READER_Y)}


def test_reader_keys_survive_a_rebuild_from_scratch(stack: Stack) -> None:
    before = _reader_files(stack.repo)
    shutil.rmtree(stack.repo / "build")
    _run([str(_osprey_bin()), "build", "--skip-deps", "--skip-lifecycle"], cwd=stack.repo)
    assert _reader_files(stack.repo) == before, "osprey build changed a reader key"
    stack.restage(stack.config)
    assert _reader_files(stack.repo) == before, "the staging after a rebuild changed a reader key"
    for name in (READER_X, READER_Y):
        assert _authenticates(stack, name, stack.reader_password(name))


def test_reset_then_up_keeps_the_old_reader_key_working(stack: Stack) -> None:
    """Last in the module: the reset removes the store's volume and every row in it."""
    keys = {name: stack.reader_password(name) for name in (READER_X, READER_Y)}
    _run([str(_osprey_bin()), "reset", "--yes"], cwd=stack.repo)
    assert {name: stack.reader_password(name) for name in keys} == keys
    _run([str(_osprey_bin()), "build", "--skip-deps", "--skip-lifecycle"], cwd=stack.repo)
    (stack.repo / ".env").touch()
    _up_again(stack)
    for name, key in keys.items():
        assert _authenticates(stack, name, key), f"{name}'s key stopped working across a reset"
