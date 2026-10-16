"""The telemetry store end to end: ingest through the gateway, store outages, the archive pass, and health.

Runs in the model-free lane against public images only; no test here names a
provider or reaches a model.

One deployment serves the whole module: the hello-world preset, which deploys
exactly the store and its gateway, initialised under this module's own project
name and port block, built and started with the real ``osprey`` verbs. Every
case posts OTLP the way a sender does — HTTP to the gateway with the sender's
Basic login — and reads the result back from the store as its admin account,
through ``clickhouse-client`` inside the store container (the admin account is
confined to loopback there).

Each case tags what it sends with its own ``session.id`` (and a ``e2e.case``
attribute), so no case reads another case's rows. The cases run in file order:
the read-only ones first, then the ones that stop, pause, rebuild, rotate and
reset the deployment, each leaving it running for the next.

The ledger feed comes last. Its cases write ledger lines the way the writers
do — through :mod:`osprey.audit.writer` where a writer exists, raw bytes where
the case is about a line no writer produces — into ``var/audit/<identity>/``,
and read them back from ``audit_ledger``. Each case owns its identity
directory; the directories exist before the build that renders the gateway, so
each has its own receiver.

CONTAINER SAFETY: every docker invocation names an exact container, image or
network of this project. Teardown is ``osprey down`` followed by the
project-scoped volume sweep (``tests/e2e/_volumes.py``); never a prune, never
a wildcard.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from osprey.audit import writer as audit_writer
from osprey.audit.envelope import (
    DECISION_ALLOWED,
    DECISION_APPROVED,
    DECISION_REFUSED,
    POSTURE_SOURCE_APP,
    SURFACE_EXECUTOR,
    AuditEnvelope,
)
from osprey.audit.tool_call import build_record
from osprey.deployment import telemetry_config as tc
from osprey.deployment import telemetry_credentials as creds
from osprey.deployment.compose_generator import compose_base_cmd, compose_provider_env
from osprey.deployment.container_lifecycle import (
    _ensure_bluesky_control_plane_keys,
    _ensure_bluesky_document_plane_certs,
    _env_file_args,
    as_built_compose_files,
)
from osprey.deployment.runtime_helper import get_runtime_command, runtime_env
from osprey.deployment.telemetry_gateway import (
    CLAMP,
    COLLECTOR_CONFIG_FILE,
    LEDGER_PATH_FILTER,
    REFUSED_SUFFIX,
    ledger_receiver_id,
    trusted_on_behalf_of,
)
from osprey.health.core.telemetry_gateway import fed_identities
from osprey.utils.workspace import AUDIT_DIR_RELPATH
from tests.e2e._volumes import remove_project_volumes

pytestmark = [
    pytest.mark.model_free,
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
]

#: This module's compose project. A compose project is addressed by name, so
#: no other e2e module may deploy under it (``tests/deployment/test_e2e_project_names.py``).
PROJECT_NAME = "osprey-e2e-telemetry-store"
#: This module's port block (``deployment.port_base``); no other e2e module uses it.
PORT_BASE = 17000

STORE_CONTAINER = f"{PROJECT_NAME}-telemetry-store"
GATEWAY_CONTAINER = f"{PROJECT_NAME}-telemetry-gateway"
PROJECT_NETWORK = f"{PROJECT_NAME}_osprey-network"

#: The image a sender container runs: Python with exactly what ``osprey health``
#: imports, and the checkout's own source mounted at run time.
SENDER_BASE_IMAGE = "python:3.11-slim"
SENDER_PACKAGES = (
    "click",
    "rich",
    "pyyaml",
    "httpx",
    "python-dotenv",
    "jinja2",
    "questionary",
    "ruamel.yaml",
    "requests",
    "pydantic",
    "fastapi",
)

VERB_TIMEOUT_S = 600
ROW_TIMEOUT_S = 120.0
#: After the expected count is reached, how long a case keeps watching for a
#: duplicate before it calls the count exact.
SETTLE_S = 10.0
#: The outage cases: a pause longer than the exporter's 30 s insert timeout,
#: and a stop long enough to span a full gateway restart.
PAUSE_S = 35.0
STOP_S = 60.0

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


# --------------------------------------------------------------------------- #
# The deployment
# --------------------------------------------------------------------------- #


@dataclass
class Stack:
    """The deployed repo and what the cases address it by."""

    repo: Path
    osprey: Path
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def gateway_port(self) -> int:
        return tc.gateway_port(self.config)

    @property
    def store_port(self) -> int:
        return tc.store_port(self.config)

    @property
    def metrics_port(self) -> int:
        return tc.metrics_port(self.config)

    @property
    def identity(self) -> str:
        identity = tc.host_identity(self.config)
        assert identity, "the render records no deployment.host_identity"
        return identity

    @property
    def gateway_url(self) -> str:
        return f"http://127.0.0.1:{self.gateway_port}"

    @property
    def store_url(self) -> str:
        return f"http://127.0.0.1:{self.store_port}"

    def reload(self) -> None:
        self.config = yaml.safe_load((self.repo / "build" / "config.yml").read_text())

    def login(self) -> str:
        login = creds.read_sender_login(self.repo, self.identity)
        assert login, f"no sender login file for {self.identity!r}"
        return login

    def auth(self) -> str:
        return basic(self.login())

    def file_value(self, name: str, key: str) -> str:
        for line in (self.repo / name).read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() == key:
                return v.strip()
        raise AssertionError(f"{name} holds no {key}")


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


def _run(cmd: list[str], cwd: Path, timeout: int = VERB_TIMEOUT_S) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, env=_env()
    )


def _verb(stack: Stack, *args: str, timeout: int = VERB_TIMEOUT_S) -> subprocess.CompletedProcess:
    """Run an ``osprey`` verb in the repo; ``pytest.fail`` on a non-zero exit."""
    result = _run([str(stack.osprey), *args], cwd=stack.repo, timeout=timeout)
    if result.returncode != 0:
        pytest.fail(
            f"osprey {' '.join(args)} failed (rc={result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


def _docker(*args: str, timeout: int = 120, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout, check=False
    )
    if check and result.returncode != 0:
        raise AssertionError(f"docker {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def _remove_project_containers() -> None:
    """Remove this project's own containers by exact name, should a run leave any."""
    for name in (STORE_CONTAINER, GATEWAY_CONTAINER):
        _docker("rm", "-f", name, check=False)


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    """``osprey init`` + ``build`` + ``up`` the hello-world preset; tear it down after."""
    osprey = _osprey_bin()
    base = tmp_path_factory.mktemp("telemetry_store_e2e")
    repo = base / PROJECT_NAME

    _remove_project_containers()
    remove_project_volumes(PROJECT_NAME)

    init = _run(
        [
            str(osprey),
            "init",
            str(repo),
            "--preset",
            "hello-world",
            "--no-git",
            "--set",
            f"port_base={PORT_BASE}",
        ],
        cwd=base,
    )
    if init.returncode != 0:
        pytest.fail(f"osprey init failed (rc={init.returncode}):\n{init.stdout}\n{init.stderr}")
    deployed = Stack(repo=repo, osprey=osprey)
    _verb(deployed, "build", "--skip-deps", "--skip-lifecycle")
    deployed.reload()
    services = deployed.config.get("deployed_services") or []
    if not {tc.STORE_SERVICE, tc.GATEWAY_SERVICE} <= set(services):
        pytest.fail(f"the hello-world render does not deploy the store and gateway: {services!r}")
    # `osprey up` refuses a repo without a .env; this preset needs nothing in it.
    (repo / ".env").touch()
    try:
        _verb(deployed, "up", "-d")
        _wait_ready(deployed)
        yield deployed
    finally:
        down = _run([str(osprey), "down"], cwd=repo, timeout=300)
        if down.returncode != 0:
            print(f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}")
        _remove_project_containers()
        remove_project_volumes(PROJECT_NAME)
        # `osprey up --build` chains a full build, whose venv every later
        # re-render carries forward; nothing reads it once the module's stacks
        # are down.
        shutil.rmtree(repo / "build" / ".venv", ignore_errors=True)


def _wait_ready(stack: Stack, timeout: float = 120.0) -> None:
    """Wait until the store answers queries and the gateway answers its counters page."""
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            if _sql("SELECT 1").strip() == "1":
                httpx.get(
                    f"http://127.0.0.1:{stack.metrics_port}/metrics", timeout=3
                ).raise_for_status()
                return
        except (AssertionError, httpx.HTTPError) as exc:
            last = str(exc)
        time.sleep(1.0)
    raise AssertionError(f"store and gateway not ready after {timeout:.0f} s: {last}")


# --------------------------------------------------------------------------- #
# Reading the store
# --------------------------------------------------------------------------- #


def _sql(query: str, fmt: str = "TSV") -> str:
    """Run *query* as the store's admin account, inside the store container, in *fmt*."""
    result = _docker(
        "exec", STORE_CONTAINER, "clickhouse-client", "--format", fmt, "-q", query, check=False
    )
    if result.returncode != 0:
        raise AssertionError(f"store query failed: {result.stderr.strip()}\n{query}")
    return result.stdout


def _scalar(query: str) -> int:
    return int(_sql(query).strip() or 0)


def _wait_value(read: Callable[[], int], expected: int, timeout: float = ROW_TIMEOUT_S) -> int:
    """Poll *read* until it reaches *expected* (or the timeout); return the last value."""
    deadline = time.monotonic() + timeout
    value = -1
    while time.monotonic() < deadline:
        with contextlib.suppress(AssertionError):
            value = read()
            if value >= expected:
                return value
        time.sleep(1.0)
    return value


def _exact(read: Callable[[], int], expected: int, what: str) -> None:
    """Assert *read* reaches *expected* and is still exactly that after :data:`SETTLE_S`."""
    reached = _wait_value(read, expected)
    assert reached == expected, f"{what}: {reached} stored, expected {expected}"
    time.sleep(SETTLE_S)
    settled = read()
    assert settled == expected, f"{what}: {settled} stored after settling, expected {expected}"


def _logs_count(case: str) -> int:
    return _scalar(
        f"SELECT count() FROM telemetry.otel_logs WHERE LogAttributes['e2e.case'] = '{case}'"
    )


def _counter(stack: Stack, name: str, **labels: str) -> float:
    """One series of the gateway's counters page; 0 when it is not exported yet."""
    text = httpx.get(f"http://127.0.0.1:{stack.metrics_port}/metrics", timeout=5).text
    for line in text.splitlines():
        if not line.startswith(name + "{") and not line.startswith(name + " "):
            continue
        if all(f'{k}="{v}"' in line for k, v in labels.items()):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


# --------------------------------------------------------------------------- #
# Sending OTLP
# --------------------------------------------------------------------------- #


def basic(login: str) -> str:
    return "Basic " + base64.b64encode(login.encode()).decode()


def _kv(attrs: Mapping[str, str]) -> list[dict[str, Any]]:
    return [{"key": k, "value": {"stringValue": v}} for k, v in attrs.items()]


def _new_case() -> tuple[str, str]:
    """A case tag and a session id, both unique to the calling case."""
    tag = uuid.uuid4().hex[:12]
    return tag, f"e2e-{tag}"


def log_payload(
    case: str,
    session: str | None,
    *,
    seq: int = 0,
    time_ns: int | None = None,
    resource: Mapping[str, str] | None = None,
    scope: Mapping[str, str] | None = None,
    record: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    attrs = {"e2e.case": case, "e2e.seq": str(seq), "event.name": "user_prompt", **(record or {})}
    if session is not None:
        attrs["session.id"] = session
    return {
        "resourceLogs": [
            {
                "resource": {"attributes": _kv({"service.name": "e2e", **(resource or {})})},
                "scopeLogs": [
                    {
                        "scope": {"name": "e2e", "attributes": _kv(scope or {})},
                        "logRecords": [
                            {
                                "timeUnixNano": str(time_ns or time.time_ns()),
                                "body": {"stringValue": f"{case}-{seq}"},
                                "attributes": _kv(attrs),
                            }
                        ],
                    }
                ],
            }
        ]
    }


def span_payload(
    case: str,
    session: str,
    *,
    time_ns: int | None = None,
    forged: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    start = time_ns or time.time_ns()
    forged = dict(forged or {})
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _kv({"service.name": "e2e", **forged})},
                "scopeSpans": [
                    {
                        "scope": {"name": "e2e", "attributes": _kv(forged)},
                        "spans": [
                            {
                                "traceId": uuid.uuid4().hex,
                                "spanId": uuid.uuid4().hex[:16],
                                "name": "e2e-span",
                                "kind": 1,
                                "startTimeUnixNano": str(start),
                                "endTimeUnixNano": str(start + 1_000_000),
                                "attributes": _kv(
                                    {"e2e.case": case, "session.id": session, **forged}
                                ),
                                "links": [
                                    {
                                        "traceId": uuid.uuid4().hex,
                                        "spanId": uuid.uuid4().hex[:16],
                                        "attributes": _kv({"e2e.case": case, **forged}),
                                    }
                                ],
                                "events": [
                                    {
                                        "timeUnixNano": str(start),
                                        "name": "e2e-event",
                                        "attributes": _kv({"e2e.case": case, **forged}),
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }


def metric_payload(
    case: str,
    session: str,
    *,
    time_ns: int | None = None,
    forged: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    now = time_ns or time.time_ns()
    forged = dict(forged or {})
    return {
        "resourceMetrics": [
            {
                "resource": {"attributes": _kv({"service.name": "e2e", **forged})},
                "scopeMetrics": [
                    {
                        "scope": {"name": "e2e", "attributes": _kv(forged)},
                        "metrics": [
                            {
                                "name": "e2e.counter",
                                "sum": {
                                    "aggregationTemporality": 2,
                                    "isMonotonic": True,
                                    "dataPoints": [
                                        {
                                            "asInt": "1",
                                            "startTimeUnixNano": str(now - 1_000_000_000),
                                            "timeUnixNano": str(now),
                                            "attributes": _kv(
                                                {"e2e.case": case, "session.id": session, **forged}
                                            ),
                                            "exemplars": [
                                                {
                                                    "asInt": "1",
                                                    "timeUnixNano": str(now),
                                                    "filteredAttributes": _kv(forged),
                                                }
                                            ],
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    }


def post(
    stack: Stack,
    signal: str,
    payload: Mapping[str, Any],
    auth: str | None,
    *,
    timeout: float = 15.0,
) -> httpx.Response:
    headers = {"Content-Type": "application/json"}
    if auth is not None:
        headers["Authorization"] = auth
    return httpx.post(
        f"{stack.gateway_url}/v1/{signal}",
        content=json.dumps(payload),
        headers=headers,
        timeout=timeout,
    )


def _accept(stack: Stack, signal: str, payload: Mapping[str, Any]) -> None:
    response = post(stack, signal, payload, stack.auth())
    assert response.status_code == 200, f"/v1/{signal}: {response.status_code} {response.text}"


class Sender(threading.Thread):
    """Posts one log record every *interval* seconds until stopped; counts what the gateway took.

    Only a 200 counts as sent: a record the gateway refused, or never
    answered, is not one the store owes.
    """

    def __init__(self, stack: Stack, case: str, session: str, interval: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.stack = stack
        self.case = case
        self.session = session
        self.interval = interval
        self.sent: list[int] = []
        self.failed = 0
        self._stop_event = threading.Event()
        self._auth = stack.auth()

    def run(self) -> None:
        seq = 0
        while not self._stop_event.is_set():
            seq += 1
            try:
                response = post(
                    self.stack,
                    "logs",
                    log_payload(self.case, self.session, seq=seq),
                    self._auth,
                    timeout=5.0,
                )
                if response.status_code == 200:
                    self.sent.append(seq)
                else:
                    self.failed += 1
            except httpx.HTTPError:
                self.failed += 1
            self._stop_event.wait(self.interval)

    def finish(self) -> list[int]:
        self._stop_event.set()
        self.join(timeout=30)
        return list(self.sent)


def _assert_exactly_once(case: str, sent: list[int]) -> None:
    assert sent, "the sender had no record accepted"
    _exact(lambda: _logs_count(case), len(sent), f"rows for case {case}")
    unique = _scalar(
        "SELECT uniqExact(LogAttributes['e2e.seq']) FROM telemetry.otel_logs "
        f"WHERE LogAttributes['e2e.case'] = '{case}'"
    )
    assert unique == len(sent), f"{unique} distinct records stored, {len(sent)} sent"
    stored = {
        int(s)
        for s in _sql(
            "SELECT LogAttributes['e2e.seq'] FROM telemetry.otel_logs "
            f"WHERE LogAttributes['e2e.case'] = '{case}'"
        ).split()
    }
    assert stored == set(sent), f"stored and sent differ: {sorted(stored ^ set(sent))[:20]}"


def _health(stack: Stack, *categories: str) -> dict[str, dict[str, Any]]:
    """``osprey health --json`` on the host for *categories*; rows by check name."""
    args = [str(stack.osprey), "health", "--json"]
    for category in categories:
        args += ["--category", category]
    result = _run(args, cwd=stack.repo, timeout=180)
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"health printed no JSON:\n{result.stdout}\n{result.stderr}") from exc
    return {row["name"]: row for row in report["results"]}


def _start_store(stack: Stack) -> None:
    _docker("start", STORE_CONTAINER)
    _wait_ready(stack)


# --------------------------------------------------------------------------- #
# The sender container
# --------------------------------------------------------------------------- #


def _source_roots() -> list[Path]:
    """The import roots of this checkout's ``osprey`` and ``osprey_connectors``."""
    import osprey
    import osprey_connectors

    return [
        Path(osprey.__file__).resolve().parents[1],
        Path(osprey_connectors.__file__).resolve().parents[1],
    ]


@pytest.fixture(scope="module")
def sender_image(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Build (once per content) the image a sender container runs."""
    dockerfile = textwrap.dedent(
        f"""\
        FROM {SENDER_BASE_IMAGE}
        RUN pip install --no-cache-dir --root-user-action=ignore {" ".join(SENDER_PACKAGES)}
        """
    )
    tag = f"osprey-e2e-telemetry-sender:{hashlib.sha256(dockerfile.encode()).hexdigest()[:12]}"
    if _docker("image", "inspect", tag, check=False).returncode == 0:
        return tag
    context = tmp_path_factory.mktemp("sender_image")
    (context / "Dockerfile").write_text(dockerfile)
    _docker("build", "-t", tag, str(context), timeout=900)
    return tag


_SENDER_SCRIPT = textwrap.dedent(
    """\
    import base64, json, os, sys, time
    import httpx
    login = os.environ["OSPREY_TELEMETRY_LOGIN"]
    payload = json.loads(sys.argv[2])
    response = httpx.post(
        sys.argv[1],
        content=json.dumps(payload),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Basic " + base64.b64encode(login.encode()).decode(),
        },
        timeout=15,
    )
    print(response.status_code)
    """
)


def _sender_container(
    stack: Stack,
    image: str,
    command: list[str],
    *,
    network: str,
    env: Mapping[str, str],
) -> subprocess.CompletedProcess:
    """Run *command* in a throwaway sender container with the repo and the source mounted."""
    roots = _source_roots()
    mounts: list[str] = ["-v", f"{stack.repo}:/repo:ro"]
    for index, root in enumerate(roots):
        mounts += ["-v", f"{root}:/osprey-src/{index}:ro"]
    pythonpath = ":".join(f"/osprey-src/{index}" for index in range(len(roots)))
    env_args: list[str] = []
    for key, value in {**env, "PYTHONPATH": pythonpath}.items():
        env_args += ["-e", f"{key}={value}"]
    return _docker(
        "run",
        "--rm",
        "--network",
        network,
        "-w",
        "/repo",
        *mounts,
        *env_args,
        image,
        *command,
        timeout=300,
        check=False,
    )


# =========================================================================== #
# Ingest
# =========================================================================== #


FORGED = {"osprey.user": "mallory", "osprey.audit.surface": "forged", "osprey.anything": "x"}


def _osprey_keys(attrs: Mapping[str, str]) -> set[str]:
    return {k for k in attrs if k.startswith("osprey.")}


def test_forged_keys_are_dropped_and_every_record_is_stamped_with_its_login(stack: Stack) -> None:
    """Forged ``osprey.*`` keys at every level, a span link's included, are gone; the login is stamped."""
    case, session = _new_case()
    _accept(
        stack,
        "logs",
        log_payload(case, session, resource=FORGED, scope=FORGED, record=FORGED),
    )
    _accept(stack, "traces", span_payload(case, session, forged=FORGED))
    _accept(stack, "metrics", metric_payload(case, session, forged=FORGED))

    where = f"WHERE LogAttributes['e2e.case'] = '{case}'"
    assert _wait_value(lambda: _logs_count(case), 1) == 1
    user, log_session, *maps = json.loads(
        _sql(
            "SELECT User, SessionId, toJSONString(ResourceAttributes), "
            "toJSONString(ScopeAttributes), toJSONString(LogAttributes) "
            f"FROM telemetry.otel_logs {where}",
            "JSONCompactEachRow",
        )
    )
    resource, scope, record = (json.loads(v) for v in maps)
    assert (user, log_session) == (stack.identity, session)
    assert _osprey_keys(resource) == {"osprey.user"} and resource["osprey.user"] == stack.identity
    assert _osprey_keys(scope) == set()
    assert _osprey_keys(record) == {"osprey.user"} and record["osprey.user"] == stack.identity

    span_where = f"WHERE SpanAttributes['e2e.case'] = '{case}'"
    assert (
        _wait_value(lambda: _scalar(f"SELECT count() FROM telemetry.otel_traces {span_where}"), 1)
        == 1
    )
    span = json.loads(
        _sql(
            "SELECT User, SessionId, toJSONString(ResourceAttributes), "
            "toJSONString(SpanAttributes), toJSONString(`Links.Attributes`), "
            f"toJSONString(`Events.Attributes`) FROM telemetry.otel_traces {span_where}",
            "JSONCompactEachRow",
        )
    )
    user, span_session, resource, attributes, links, events = span
    assert (user, span_session) == (stack.identity, session)
    assert _osprey_keys(json.loads(resource)) == {"osprey.user"}
    attributes = json.loads(attributes)
    assert (
        _osprey_keys(attributes) == {"osprey.user"} and attributes["osprey.user"] == stack.identity
    )
    for link in json.loads(links):
        assert _osprey_keys(link) == set(), f"a span link kept a forged key: {link}"
    for event in json.loads(events):
        assert _osprey_keys(event) == set(), f"a span event kept a forged key: {event}"

    point_where = f"WHERE Attributes['e2e.case'] = '{case}'"
    assert (
        _wait_value(
            lambda: _scalar(f"SELECT count() FROM telemetry.otel_metrics_sum {point_where}"), 1
        )
        == 1
    )
    point = json.loads(
        _sql(
            "SELECT toJSONString(ResourceAttributes), toJSONString(ScopeAttributes), "
            "toJSONString(Attributes), toJSONString(`Exemplars.FilteredAttributes`) "
            f"FROM telemetry.otel_metrics_sum {point_where}",
            "JSONCompactEachRow",
        )
    )
    resource, scope, attributes, exemplars = (json.loads(v) for v in point)
    assert resource["osprey.user"] == stack.identity and _osprey_keys(resource) == {"osprey.user"}
    assert _osprey_keys(scope) == set()
    assert attributes["osprey.user"] == stack.identity and _osprey_keys(attributes) == {
        "osprey.user"
    }
    for exemplar in exemplars:
        assert _osprey_keys(exemplar) == set(), f"an exemplar kept a forged key: {exemplar}"


@pytest.mark.parametrize("auth", ["wrong", "missing"])
def test_a_wrong_or_missing_login_is_refused_with_401_and_stores_nothing(
    stack: Stack, auth: str
) -> None:
    case, session = _new_case()
    before = _scalar("SELECT count() FROM telemetry.otel_logs")
    header = basic(f"{stack.identity}:not-the-password") if auth == "wrong" else None
    for signal, payload in (
        ("logs", log_payload(case, session)),
        ("traces", span_payload(case, session)),
        ("metrics", metric_payload(case, session)),
    ):
        response = post(stack, signal, payload, header)
        assert response.status_code == 401, f"/v1/{signal} answered {response.status_code}"

    # A control record sent afterwards with the real login lands, so the
    # refused ones had every chance to land before it.
    control, control_session = _new_case()
    _accept(stack, "logs", log_payload(control, control_session))
    assert _wait_value(lambda: _logs_count(control), 1) == 1
    assert _logs_count(case) == 0
    assert (
        _scalar(
            f"SELECT count() FROM telemetry.otel_traces WHERE SpanAttributes['e2e.case'] = '{case}'"
        )
        == 0
    )
    assert (
        _scalar(
            f"SELECT count() FROM telemetry.otel_metrics_sum WHERE Attributes['e2e.case'] = '{case}'"
        )
        == 0
    )
    assert _scalar("SELECT count() FROM telemetry.otel_logs") == before + 1


def test_a_record_without_a_session_id_is_stored_with_an_empty_session(stack: Stack) -> None:
    case, _session = _new_case()
    _accept(stack, "logs", log_payload(case, None))
    assert _wait_value(lambda: _logs_count(case), 1) == 1
    row = _sql(
        "SELECT SessionId, User, toUInt8(mapContains(LogAttributes, 'session.id')) "
        f"FROM telemetry.otel_logs WHERE LogAttributes['e2e.case'] = '{case}'"
    ).rstrip("\n")
    assert row.split("\t") == ["", stack.identity, "0"]


def test_records_outside_the_time_window_are_dropped_and_counted(stack: Stack) -> None:
    """A record a year ahead or older than the retention is dropped and counted; a current one lands."""
    case, session = _new_case()
    ahead = time.time_ns() + int(timedelta(days=365).total_seconds() * 1e9)
    behind = time.time_ns() - int(
        timedelta(days=tc.DEFAULT_RETENTION_DAYS + 1).total_seconds() * 1e9
    )
    clamp = {"filter": "filter/clamp"}
    before = {
        series: _counter(stack, series, **clamp)
        for series in (
            "otelcol_processor_filter_logs_filtered",
            "otelcol_processor_filter_spans_filtered",
            "otelcol_processor_filter_datapoints_filtered",
        )
    }

    _accept(stack, "logs", log_payload(case, session, seq=1, time_ns=ahead))
    _accept(stack, "logs", log_payload(case, session, seq=2, time_ns=behind))
    _accept(stack, "traces", span_payload(case, session, time_ns=ahead))
    _accept(stack, "metrics", metric_payload(case, session, time_ns=ahead))
    _accept(stack, "logs", log_payload(case, session, seq=3))

    _exact(lambda: _logs_count(case), 1, "current records of the clamp case")
    assert (
        _sql(
            f"SELECT LogAttributes['e2e.seq'] FROM telemetry.otel_logs WHERE LogAttributes['e2e.case'] = '{case}'"
        ).strip()
        == "3"
    )
    assert (
        _scalar(
            f"SELECT count() FROM telemetry.otel_traces WHERE SpanAttributes['e2e.case'] = '{case}'"
        )
        == 0
    )
    assert (
        _scalar(
            f"SELECT count() FROM telemetry.otel_metrics_sum WHERE Attributes['e2e.case'] = '{case}'"
        )
        == 0
    )

    after = {series: _counter(stack, series, **clamp) for series in before}
    assert (
        after["otelcol_processor_filter_logs_filtered"]
        - before["otelcol_processor_filter_logs_filtered"]
        >= 2
    )
    assert (
        after["otelcol_processor_filter_spans_filtered"]
        - before["otelcol_processor_filter_spans_filtered"]
        >= 1
    )
    assert (
        after["otelcol_processor_filter_datapoints_filtered"]
        - before["otelcol_processor_filter_datapoints_filtered"]
        >= 1
    )


def test_a_host_launch_environment_from_the_real_launch_builder_posts_to_the_gateway(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env ``osprey chat`` builds on the host carries a login the deployed gateway takes."""
    from osprey.agent_runner.provider_env import load_provider_spec
    from osprey.build.claude_code_telemetry import export_host_terminal_session
    from osprey.deployment.telemetry_credentials import export_built_host_identity

    # The exports below write into os.environ. Setting each name before
    # deleting it makes monkeypatch record it, so the teardown removes what
    # the exports wrote even where the name was absent to begin with.
    for name in _SCRUBBED_ENV:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    exported = export_built_host_identity(os.environ, stack.repo)
    assert set(exported) <= set(_SCRUBBED_ENV), (
        f"the host launch exports a name the teardown does not restore: {exported}"
    )
    assert creds.LOGIN_ENV in exported, f"the host launch carries no login: {exported}"
    terminal = export_host_terminal_session(os.environ)

    spec = load_provider_spec(stack.repo / "build", env_dir=stack.repo)
    assert spec is not None
    env = spec.env_block
    endpoint = env["OTEL_EXPORTER_OTLP_ENDPOINT"].rstrip("/")
    assert endpoint == stack.gateway_url, endpoint
    headers = dict(
        pair.split("=", 1) for pair in env["OTEL_EXPORTER_OTLP_HEADERS"].split(",") if pair
    )
    resource = dict(
        pair.split("=", 1) for pair in env["OTEL_RESOURCE_ATTRIBUTES"].split(",") if pair
    )
    assert resource.get("terminal.session") == terminal

    case, session = _new_case()
    response = httpx.post(
        f"{endpoint}/v1/logs",
        content=json.dumps(log_payload(case, session, resource=resource)),
        headers={"Content-Type": "application/json", **headers},
        timeout=15,
    )
    assert response.status_code == 200, response.text
    assert _wait_value(lambda: _logs_count(case), 1) == 1
    row = _sql(
        "SELECT User, TerminalSession, SessionId FROM telemetry.otel_logs "
        f"WHERE LogAttributes['e2e.case'] = '{case}'"
    ).strip()
    assert row.split("\t") == [stack.identity, terminal, session]


def test_a_host_networked_sender_container_posts_to_the_gateway(
    stack: Stack, sender_image: str
) -> None:
    case, session = _new_case()
    result = _sender_container(
        stack,
        sender_image,
        [
            "python",
            "-c",
            _SENDER_SCRIPT,
            f"http://127.0.0.1:{stack.gateway_port}/v1/logs",
            json.dumps(log_payload(case, session)),
        ],
        network="host",
        env={creds.LOGIN_ENV: stack.login()},
    )
    assert result.returncode == 0 and result.stdout.strip() == "200", (
        f"host-networked sender: rc={result.returncode}\n{result.stdout}\n{result.stderr}"
    )
    assert _wait_value(lambda: _logs_count(case), 1) == 1
    assert (
        _sql(
            f"SELECT User FROM telemetry.otel_logs WHERE LogAttributes['e2e.case'] = '{case}'"
        ).strip()
        == stack.identity
    )


def test_osprey_health_inside_a_sender_container_reports_the_stateless_rules(
    stack: Stack, sender_image: str
) -> None:
    """A container on the compose network posts, then ``osprey health`` there reports the stateless tier."""
    case, session = _new_case()
    env = {creds.LOGIN_ENV: stack.login(), tc.GATEWAY_HOST_ENV: tc.GATEWAY_SERVICE}
    sent = _sender_container(
        stack,
        sender_image,
        [
            "python",
            "-c",
            _SENDER_SCRIPT,
            f"http://{tc.GATEWAY_SERVICE}:{tc.GATEWAY_OTLP_CONTAINER_PORT}/v1/logs",
            json.dumps(log_payload(case, session)),
        ],
        network=PROJECT_NETWORK,
        env=env,
    )
    assert sent.returncode == 0 and sent.stdout.strip() == "200", sent.stderr
    assert _wait_value(lambda: _logs_count(case), 1) == 1

    health = _sender_container(
        stack,
        sender_image,
        [
            "python",
            "-c",
            "from osprey.cli.main import main; main()",
            "health",
            "--json",
            "--category",
            "telemetry_gateway",
            "--category",
            "telemetry_store",
        ],
        network=PROJECT_NETWORK,
        env=env,
    )
    try:
        report = json.loads(health.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"in-container health printed no JSON:\n{health.stdout}\n{health.stderr}"
        ) from exc
    rows = {row["name"]: row for row in report["results"]}
    for name in (
        "telemetry_gateway_metrics",
        "telemetry_gateway_queue",
        "telemetry_gateway_refusals",
        "telemetry_gateway_audit_files",
        "telemetry_gateway_unfed",
        "telemetry_store_ping",
    ):
        assert name in rows, f"{name} missing from the in-container report: {sorted(rows)}"
        assert rows[name]["status"] == "ok", rows[name]
    assert (
        f"{tc.GATEWAY_SERVICE}:{tc.GATEWAY_METRICS_CONTAINER_PORT}"
        in rows["telemetry_gateway_metrics"]["message"]
    )
    assert "telemetry_gateway_trend" not in rows, "a container took the host tier"


def test_provisioning_is_idempotent(stack: Stack) -> None:
    """A second ``osprey up`` leaves users, roles, grants, policies, profiles and quotas unchanged."""
    queries = [
        "SELECT * FROM system.users ORDER BY name FORMAT TSV",
        "SELECT * FROM system.roles ORDER BY name FORMAT TSV",
        "SELECT * FROM system.role_grants ORDER BY ALL FORMAT TSV",
        "SELECT * FROM system.grants ORDER BY ALL FORMAT TSV",
        # A policy is re-stated whole on every run, so its id is not part of
        # what has to stay the same; its name, table, filter and holders are.
        "SELECT * EXCEPT (id) FROM system.row_policies ORDER BY name FORMAT TSV",
        "SELECT * FROM system.settings_profiles ORDER BY name FORMAT TSV",
        "SELECT * FROM system.settings_profile_elements ORDER BY ALL FORMAT TSV",
        "SELECT * FROM system.quotas ORDER BY name FORMAT TSV",
    ]
    before = [_sql(q) for q in queries]
    _verb(stack, "up", "-d")
    _wait_ready(stack)
    after = [_sql(q) for q in queries]
    for query, first, second in zip(queries, before, after, strict=True):
        assert first == second, f"{query} changed across a second osprey up"


# =========================================================================== #
# Health on the host
# =========================================================================== #


def test_host_health_reports_store_and_gateway_up_down_and_degraded(stack: Stack) -> None:
    rows = _health(stack, "telemetry_store", "telemetry_gateway")
    assert rows["telemetry_store_ping"]["status"] == "ok", rows["telemetry_store_ping"]
    assert rows["telemetry_gateway_metrics"]["status"] == "ok", rows["telemetry_gateway_metrics"]
    assert "telemetry_gateway_trend" in rows

    _docker("stop", GATEWAY_CONTAINER, timeout=120)
    try:
        rows = _health(stack, "telemetry_gateway")
        assert rows["telemetry_gateway_metrics"]["status"] == "warning", rows[
            "telemetry_gateway_metrics"
        ]
    finally:
        _docker("start", GATEWAY_CONTAINER)
    _docker("stop", STORE_CONTAINER, timeout=120)
    try:
        rows = _health(stack, "telemetry_store")
        assert rows["telemetry_store_ping"]["status"] == "warning", rows["telemetry_store_ping"]
    finally:
        _start_store(stack)

    # Degraded: the store refuses the ingest account, so the gateway's queue
    # holds data and sends nothing. The host tier judges that against its
    # previous sample; the sample is aged by the stall window rather than the
    # case waiting it out.
    baseline_file = stack.repo / "var" / "health" / "telemetry_gateway.json"
    baseline_file.unlink(missing_ok=True)
    taken = _health(stack, "telemetry_gateway")["telemetry_gateway_trend"]
    assert taken["message"] == "Baseline taken", taken
    case, session = _new_case()
    _sql("ALTER USER ingest IDENTIFIED WITH sha256_password BY 'e2e-changed-password'")
    try:
        _accept(stack, "logs", log_payload(case, session))
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if _counter(stack, "otelcol_exporter_queue_size", exporter="clickhouse/otlp") > 0:
                break
            time.sleep(1.0)
        baseline = json.loads(baseline_file.read_text())
        aged = datetime.fromisoformat(baseline["ts"]) - timedelta(seconds=301)
        baseline["ts"] = aged.isoformat()
        baseline["sent_since"] = dict.fromkeys(baseline.get("sent_since") or {}, aged.isoformat())
        baseline_file.write_text(json.dumps(baseline))
        rows = _health(stack, "telemetry_gateway")
        trend = rows["telemetry_gateway_trend"]
        metrics = httpx.get(f"http://127.0.0.1:{stack.metrics_port}/metrics", timeout=5).text
        assert trend["status"] == "warning", (
            trend,
            baseline,
            [line for line in metrics.splitlines() if "exporter_" in line and "#" not in line],
        )
        assert "clickhouse/otlp" in trend["message"], trend
    finally:
        # `osprey up` re-asserts the ingest account from its file.
        _verb(stack, "up", "-d")
        _wait_ready(stack)
    # The record held while the store refused the account lands once it is restored.
    _exact(lambda: _logs_count(case), 1, "the record held through the refused ingest login")


# =========================================================================== #
# Outages
# =========================================================================== #


def test_the_gateway_starts_while_the_store_is_down_and_still_writes_every_column(
    stack: Stack,
) -> None:
    case, session = _new_case()
    terminal = f"term-{case}"
    _docker("stop", STORE_CONTAINER, timeout=120)
    try:
        _docker("restart", GATEWAY_CONTAINER, timeout=120)
        deadline = time.monotonic() + 60
        while True:
            try:
                httpx.get(
                    f"http://127.0.0.1:{stack.metrics_port}/metrics", timeout=3
                ).raise_for_status()
                break
            except httpx.HTTPError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(1.0)
        _accept(stack, "logs", log_payload(case, session, resource={"terminal.session": terminal}))
        _accept(stack, "traces", span_payload(case, session))
        _accept(stack, "metrics", metric_payload(case, session))
    finally:
        _start_store(stack)

    assert _wait_value(lambda: _logs_count(case), 1) == 1
    row = _sql(
        "SELECT User, SessionId, TerminalSession, Event, ServiceName FROM telemetry.otel_logs "
        f"WHERE LogAttributes['e2e.case'] = '{case}'"
    ).strip()
    assert row.split("\t") == [stack.identity, session, terminal, "user_prompt", "e2e"]
    assert (
        _wait_value(
            lambda: _scalar(
                "SELECT count() FROM telemetry.otel_traces WHERE SpanAttributes['e2e.case'] = "
                f"'{case}' AND User = '{stack.identity}' AND SessionId = '{session}'"
            ),
            1,
        )
        == 1
    )
    assert (
        _wait_value(
            lambda: _scalar(
                "SELECT count() FROM telemetry.otel_metrics_sum WHERE Attributes['e2e.case'] = "
                f"'{case}' AND ResourceAttributes['osprey.user'] = '{stack.identity}'"
            ),
            1,
        )
        == 1
    )


def test_otlp_is_stored_exactly_once_across_a_store_pause(stack: Stack) -> None:
    case, session = _new_case()
    sender = Sender(stack, case, session)
    sender.start()
    try:
        time.sleep(3)
        _docker("pause", STORE_CONTAINER)
        try:
            time.sleep(PAUSE_S)
        finally:
            _docker("unpause", STORE_CONTAINER)
        time.sleep(3)
    finally:
        sent = sender.finish()
    _assert_exactly_once(case, sent)


@dataclass
class _Outage:
    case: str
    sent: list[int]


@pytest.fixture(scope="module")
def stop_outage(stack: Stack) -> _Outage:
    """Records sent across a 60 s store stop with a clean gateway restart inside it."""
    case, session = _new_case()
    sender = Sender(stack, case, session)
    sender.start()
    try:
        time.sleep(3)
        _docker("stop", STORE_CONTAINER, timeout=120)
        stopped = time.monotonic()
        try:
            time.sleep(10)
            _docker("restart", GATEWAY_CONTAINER, timeout=120)
            time.sleep(max(0.0, STOP_S - (time.monotonic() - stopped)))
        finally:
            _start_store(stack)
        time.sleep(3)
    finally:
        sent = sender.finish()
    return _Outage(case, sent)


def test_otlp_is_stored_exactly_once_across_a_store_stop_and_a_gateway_restart(
    stop_outage: _Outage,
) -> None:
    _assert_exactly_once(stop_outage.case, stop_outage.sent)


def test_an_archive_pass_after_the_stop_exports_the_late_rows_in_their_arrival_day(
    stack: Stack, stop_outage: _Outage, tmp_path: Path
) -> None:
    import duckdb

    from osprey.services.archive.run import run_pass
    from osprey.services.archive.telemetry_export import TelemetryExporter

    _assert_exactly_once(stop_outage.case, stop_outage.sent)
    days = sorted(
        _sql(
            "SELECT DISTINCT toDate(InsertedAt, 'UTC') FROM telemetry.otel_logs "
            f"WHERE LogAttributes['e2e.case'] = '{stop_outage.case}'"
        ).split()
    )
    assert days
    last = datetime.fromisoformat(days[-1]).replace(tzinfo=UTC)
    # The pass runs as if the arrival day had just closed.
    now = last + timedelta(days=1, minutes=20)
    sources = tmp_path / "sources"
    dest = tmp_path / "archive"
    sources.mkdir()
    dest.mkdir()
    exporter = TelemetryExporter(
        url=stack.store_url,
        backfill_days=len(days) + 1,
        environ={
            creds.ARCHIVE_PASSWORD_ENV: stack.file_value(
                creds.ARCHIVE_FILE, creds.ARCHIVE_PASSWORD_ENV
            )
        },
    )
    result = run_pass(sources, dest, now=now, telemetry=exporter)
    assert not result.errors, result.errors
    assert set(days) <= set(result.telemetry_days), result.telemetry_days

    exported: list[int] = []
    for day in days:
        files = list(dest.rglob(f"telemetry/{day}/otel_logs.parquet"))
        assert len(files) == 1, f"otel_logs Parquet for {day}: {files}"
        rows = duckdb.execute(
            "SELECT LogAttributes['e2e.seq'] FROM read_parquet(?) "
            "WHERE LogAttributes['e2e.case'] = ?",
            [str(files[0]), stop_outage.case],
        ).fetchall()
        exported.extend(int(seq) for (seq,) in rows)
    assert sorted(exported) == sorted(stop_outage.sent)


# =========================================================================== #
# Keys across rebuild, rotation and reset
# =========================================================================== #


def _credential_digests(repo: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(repo.glob(creds.CREDENTIAL_GLOB))
        if path.is_file()
    }


def _container_ids() -> dict[str, str]:
    out = _docker(
        "ps",
        "-a",
        "--filter",
        f"label=com.docker.compose.project={PROJECT_NAME}",
        "--format",
        "{{.Names}}\t{{.ID}}",
    ).stdout
    return dict(line.split("\t", 1) for line in out.splitlines() if line.strip())


def _assert_ingest_lands(stack: Stack) -> None:
    case, session = _new_case()
    _accept(stack, "logs", log_payload(case, session))
    assert _wait_value(lambda: _logs_count(case), 1) == 1


def _assert_archive_authenticates(stack: Stack) -> None:
    password = stack.file_value(creds.ARCHIVE_FILE, creds.ARCHIVE_PASSWORD_ENV)
    response = httpx.post(
        stack.store_url,
        content=b"SELECT count() FROM telemetry.otel_logs",
        auth=("archive", password),
        timeout=15,
    )
    assert response.status_code == 200, f"archive: {response.status_code} {response.text}"


def test_every_build_minted_key_survives_a_rebuild_from_scratch(stack: Stack) -> None:
    before = _credential_digests(stack.repo)
    assert creds.INGEST_FILE in before and creds.HTPASSWD_FILE in before, sorted(before)
    shutil.rmtree(stack.repo / "build")
    _verb(stack, "build", "--skip-deps", "--skip-lifecycle")
    stack.reload()
    assert _credential_digests(stack.repo) == before, "a rebuild changed a credential file"
    _verb(stack, "up", "-d")
    _wait_ready(stack)
    assert _credential_digests(stack.repo) == before, "osprey up changed a credential file"
    _assert_ingest_lands(stack)
    _assert_archive_authenticates(stack)


def test_deleting_the_ingest_file_recreates_only_the_gateway(stack: Stack) -> None:
    before_ids = _container_ids()
    assert GATEWAY_CONTAINER in before_ids and STORE_CONTAINER in before_ids, before_ids
    old = (stack.repo / creds.INGEST_FILE).read_bytes()
    (stack.repo / creds.INGEST_FILE).unlink()

    _verb(stack, "up", "-d")
    _wait_ready(stack)
    assert (stack.repo / creds.INGEST_FILE).read_bytes() != old, "the ingest key was not rotated"
    after_ids = _container_ids()
    assert after_ids[GATEWAY_CONTAINER] != before_ids[GATEWAY_CONTAINER], (
        "the gateway was not recreated"
    )
    others = {name: cid for name, cid in before_ids.items() if name != GATEWAY_CONTAINER}
    assert {name: after_ids.get(name) for name in others} == others, (
        "another container was recreated"
    )
    _assert_ingest_lands(stack)


def test_reset_then_up_keeps_every_key_and_ingest_and_archive_authenticate(stack: Stack) -> None:
    before = _credential_digests(stack.repo)
    _verb(stack, "reset", "--yes")
    assert _credential_digests(stack.repo) == before, "osprey reset changed a credential file"
    _verb(stack, "up", "-d", "--build")
    stack.reload()
    _wait_ready(stack)
    assert _credential_digests(stack.repo) == before, "osprey up after a reset changed a key"
    _assert_ingest_lands(stack)
    _assert_archive_authenticates(stack)


# =========================================================================== #
# The ledger feed
# =========================================================================== #


#: One identity directory per case, each created before the build that renders
#: the gateway, so each has its own receiver. ``bob`` is the identity whose feed
#: keeps flowing beside hostile files; the three service names are the writers
#: whose ``on_behalf_of`` the gateway judges.
LEDGER = {
    "days": "e2e-days",
    "maintenance": "e2e-maintenance",
    "channel_finder": "e2e-channel-finder",
    "hook": "e2e-hook",
    "executor": "e2e-executor",
    "links": "e2e-links",
    "victim": "e2e-victim",
    "overcap": "e2e-overcap",
    "torn": "e2e-torn",
    "incomplete": "e2e-incomplete",
    "nonjson": "e2e-nonjson",
    "badts": "e2e-badts",
    "copytruncate": "e2e-copytruncate",
    "nan": "e2e-nan",
    "fifo": "e2e-fifo",
    "zero": "e2e-zero",
    "backfill": "e2e-backfill",
    "pause": "e2e-pause",
    "stop": "e2e-stop",
    "kill": "e2e-kill",
    "bob": "bob",
    "sidecar": "sidecar",
    "dispatch": "dispatch-worker-1",
    "queueserver": "queueserver",
}
#: Created by their cases, after the render: no receiver reads them.
LATE_IDENTITY = "e2e-late"
LOCKED_IDENTITY = "e2e-locked"
#: The roster the roster-removal case renders, then shrinks.
ROSTER_KEPT = "alice"
ROSTER_REMOVED = "carol"

#: "Half a line, then this much silence": past the reader's 60 s flush.
INCOMPLETE_SILENCE_S = 65.0
#: "A line in two halves this far apart": inside the flush, so one line.
TORN_GAP_S = 11.0
#: Lines per identity in the at-start burst.
BURST_LINES = 100
#: Days of history the back-fill case writes while the store is down.
BACKFILL_DAYS = 14
BACKFILL_PER_DAY = 5


def _stamp(when: datetime) -> str:
    """A ledger ``ts`` the way the writers spell it."""
    return when.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _micros(when: datetime) -> int:
    return int(when.timestamp()) * 1_000_000 + when.microsecond


def _audit_root(stack: Stack) -> Path:
    return stack.repo / AUDIT_DIR_RELPATH


def _audit_dir(stack: Stack, identity: str) -> Path:
    directory = _audit_root(stack) / identity
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _envelope(
    identity: str,
    *,
    surface: str,
    subject: str,
    when: datetime | None = None,
    decision: str = DECISION_ALLOWED,
    on_behalf_of: str | None = None,
    tool_use_id: str | None = None,
    source: str | None = None,
) -> AuditEnvelope:
    return AuditEnvelope(
        ts=_stamp(when or datetime.now(UTC)),
        surface=surface,
        actor=identity,
        posture="sandbox",
        posture_source=POSTURE_SOURCE_APP,
        session=f"term-{identity}",
        on_behalf_of=on_behalf_of,
        subject=subject,
        decision=decision,
        reason="e2e",
        tool_use_id=tool_use_id,
        source=source,
    )


def _envelope_line(envelope: AuditEnvelope) -> bytes:
    """One envelope as the writers encode it: compact JSON and a newline."""
    return (json.dumps(envelope.to_dict(), separators=(",", ":")) + "\n").encode()


def _write_envelope(stack: Stack, identity: str, stem: str, envelope: AuditEnvelope) -> None:
    """Append *envelope* through the real writer to ``<identity>/<stem>.jsonl``."""
    path = _audit_dir(stack, identity) / f"{stem}.jsonl"
    assert audit_writer.append_envelope(path, envelope), f"the writer did not append to {path}"


def _tool_call(
    identity: str,
    *,
    tool_use_id: str,
    subject: str = "e2e_tool",
    duration_ms: float = 12.5,
    result: Any = "ok",
    when: datetime | None = None,
) -> dict[str, Any]:
    return build_record(
        ts=_stamp(when or datetime.now(UTC)),
        actor=identity,
        posture="sandbox",
        posture_source=POSTURE_SOURCE_APP,
        session=f"term-{identity}",
        session_id=f"sess-{identity}",
        tool_use_id=tool_use_id,
        server="workspace",
        subject=subject,
        decision=DECISION_ALLOWED,
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


def _write_tool_call(stack: Stack, identity: str, record: dict[str, Any]) -> None:
    """Append *record* through the writer's ``tool_call`` entry point, into *identity*."""
    path = _audit_dir(stack, identity) / "tool_call.jsonl"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(audit_writer, "ledger_path", lambda surface, identity=None: path)
        assert audit_writer.append_record("tool_call", record) == path


def _write_raw(stack: Stack, identity: str, stem: str, data: bytes) -> Path:
    path = _audit_dir(stack, identity) / f"{stem}.jsonl"
    with path.open("ab") as handle:
        handle.write(data)
    return path


def _ledger_count(identity: str) -> int:
    return _scalar(f"SELECT count() FROM telemetry.audit_ledger WHERE User = '{identity}'")


def _ledger_rows(identity: str, columns: str) -> list[list[Any]]:
    """*columns* of every ``audit_ledger`` row of *identity*, oldest first."""
    out = _sql(
        f"SELECT {columns} FROM telemetry.audit_ledger WHERE User = '{identity}' "
        "ORDER BY Timestamp, Body",
        "JSONCompactEachRow",
    )
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def _ledger_subjects(identity: str) -> list[str]:
    return sorted(
        _sql(
            "SELECT JSONExtractString(Body, 'subject') FROM telemetry.audit_ledger "
            f"WHERE User = '{identity}'"
        ).split()
    )


def _today_utc() -> str:
    return datetime.now(UTC).date().isoformat()


def _wait_metrics(stack: Stack, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            httpx.get(
                f"http://127.0.0.1:{stack.metrics_port}/metrics", timeout=3
            ).raise_for_status()
            return
        except httpx.HTTPError:
            if time.monotonic() > deadline:
                raise
            time.sleep(1.0)


def _collector(stack: Stack) -> dict[str, Any]:
    path = stack.repo / "build" / "services" / tc.GATEWAY_SERVICE / COLLECTOR_CONFIG_FILE
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _wide_verb(stack: Stack, *args: str, check: bool = True) -> tuple[int, str]:
    """Run an ``osprey`` verb with lines wide enough that no path wraps; rc and both streams."""
    result = subprocess.run(
        [str(stack.osprey), *args],
        cwd=str(stack.repo),
        capture_output=True,
        text=True,
        timeout=VERB_TIMEOUT_S,
        env={**_env(), "COLUMNS": "4000"},
    )
    output = result.stdout + result.stderr
    if check and result.returncode != 0:
        pytest.fail(f"osprey {' '.join(args)} failed (rc={result.returncode}):\n{output}")
    return result.returncode, output


def _flat(text: str) -> str:
    """*text* without whitespace, so a path a console wrapped still matches."""
    return "".join(text.split())


class LedgerWriter(threading.Thread):
    """Appends one envelope every *interval* seconds through the real writer until stopped."""

    def __init__(self, stack: Stack, identity: str, interval: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.stack = stack
        self.identity = identity
        self.interval = interval
        self.sent: list[int] = []
        self._stop_event = threading.Event()

    def run(self) -> None:
        seq = 0
        while not self._stop_event.is_set():
            seq += 1
            _write_envelope(
                self.stack,
                self.identity,
                "e2e",
                _envelope(self.identity, surface="e2e", subject=f"seq-{seq:05d}"),
            )
            self.sent.append(seq)
            self._stop_event.wait(self.interval)

    def finish(self) -> list[int]:
        self._stop_event.set()
        self.join(timeout=30)
        return list(self.sent)


def _assert_ledger_exactly_once(identity: str, sent: list[int]) -> None:
    assert sent, "the writer appended nothing"
    _exact(lambda: _ledger_count(identity), len(sent), f"ledger rows of {identity}")
    assert _ledger_subjects(identity) == [f"seq-{n:05d}" for n in sorted(sent)]


def _render_receivers(osprey: Path, base: Path, preset: str) -> int | None:
    """How many ledger receivers *preset* renders; ``None`` when it does not build."""
    repo = base / preset
    if _run([str(osprey), "init", str(repo), "--preset", preset, "--no-git"], cwd=base).returncode:
        return None
    if _run([str(osprey), "build", "--skip-deps", "--skip-lifecycle"], cwd=repo).returncode:
        return None
    path = repo / "build" / "services" / tc.GATEWAY_SERVICE / COLLECTOR_CONFIG_FILE
    if not path.exists():
        return 0
    return len(fed_identities(yaml.safe_load(path.read_text(encoding="utf-8"))))


def _largest_preset_receivers(osprey: Path, base: Path) -> int:
    """R: the most ledger receivers any bundled preset renders on a fresh build."""
    from osprey.profiles import presets

    names = sorted(path.stem for path in Path(presets.__file__).parent.glob("*.yml"))
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        counts = dict(
            zip(
                names,
                pool.map(lambda name: _render_receivers(osprey, base, name), names),
                strict=True,
            )
        )
    rendered = {name: n for name, n in counts.items() if n is not None}
    if not any(rendered.values()):
        pytest.fail(f"no bundled preset renders a ledger receiver: {counts!r}")
    print(f"ledger receivers per preset: {counts!r}")
    return max(rendered.values())


def _burst_identities(receivers: int) -> list[str]:
    return [f"e2e-burst-{index:02d}" for index in range(receivers)]


def _half_tool_call(identity: str) -> tuple[bytes, dict[str, Any]]:
    """A ``tool_call`` line cut right after its ``decision``: every field the store keeps."""
    record = _tool_call(
        identity, tool_use_id=f"toolu_{identity.replace('-', '_')}", duration_ms=7.25
    )
    line = json.dumps(record, separators=(",", ":"))
    cut = line.index(',"reason"')
    return line[:cut].encode(), record


@dataclass
class Ledger:
    stack: Stack
    #: R, the most ledger receivers any bundled preset renders.
    receivers: int
    #: When the half line was in place and the gateway reading it.
    half_line_since: float
    half_line_record: dict[str, Any]


@pytest.fixture(scope="module")
def ledger(stack: Stack, tmp_path_factory: pytest.TempPathFactory) -> Ledger:
    """Every case's identity directory, then a build and an ``osprey up`` that feed each."""
    receivers = _largest_preset_receivers(stack.osprey, tmp_path_factory.mktemp("presets"))
    identities = [*LEDGER.values(), *_burst_identities(receivers)]
    for identity in identities:
        _audit_dir(stack, identity)
    half, record = _half_tool_call(LEDGER["incomplete"])
    _write_raw(stack, LEDGER["incomplete"], "tool_call", half)

    _verb(stack, "build", "--skip-deps", "--skip-lifecycle")
    stack.reload()
    _verb(stack, "up", "-d")
    _wait_ready(stack)
    since = time.monotonic()
    missing = set(identities) - fed_identities(_collector(stack))
    if missing:
        pytest.fail(f"the build renders no receiver for {sorted(missing)}")
    return Ledger(stack, receivers, since, record)


def test_ledger_lines_dated_40_3_and_0_days_ago_yield_two_rows_in_their_partitions(
    ledger: Ledger,
) -> None:
    stack, identity = ledger.stack, LEDGER["days"]
    now = datetime.now(UTC)
    dates = {age: now - timedelta(days=age) for age in (40, 3, 0)}
    before = _counter(stack, "otelcol_processor_filter_logs_filtered", filter=CLAMP)
    for age, when in dates.items():
        _write_envelope(
            stack,
            identity,
            "e2e",
            _envelope(identity, surface="e2e", subject=f"age-{age}", when=when),
        )
    _exact(lambda: _ledger_count(identity), 2, "rows of the dated lines")
    rows = _ledger_rows(identity, "JSONExtractString(Body, 'subject'), _partition_id")
    assert sorted(rows) == sorted(
        [f"age-{age}", dates[age].strftime("%Y%m%d")] for age in (3, 0)
    ), rows
    after = _counter(stack, "otelcol_processor_filter_logs_filtered", filter=CLAMP)
    assert after - before >= 1, "the 40-day line was not counted as dropped"


def test_a_maintenance_ledger_line_keeps_its_surface(ledger: Ledger) -> None:
    stack, identity = ledger.stack, LEDGER["maintenance"]
    _write_envelope(
        stack,
        identity,
        audit_writer.WRITER_MAINTENANCE,
        _envelope(identity, surface="setup_patch", subject="config.key"),
    )
    _exact(lambda: _ledger_count(identity), 1, "rows of the maintenance line")
    rows = _ledger_rows(identity, "Surface, LogAttributes['log.file.name']")
    assert rows == [["setup_patch", f"{audit_writer.WRITER_MAINTENANCE}.jsonl"]], rows


def test_a_channel_finder_envelope_keeps_its_ts_and_surface(ledger: Ledger) -> None:
    stack, identity = ledger.stack, LEDGER["channel_finder"]
    when = datetime.now(UTC).replace(microsecond=123456) - timedelta(hours=2)
    _write_envelope(
        stack,
        identity,
        "channel-finder",
        _envelope(identity, surface="channel-finder", subject="search", when=when),
    )
    _exact(lambda: _ledger_count(identity), 1, "rows of the channel-finder line")
    rows = _ledger_rows(identity, "Surface, toUnixTimestamp64Micro(Timestamp), TimeUnparsed")
    assert rows == [["channel-finder", _micros(when), False]], rows


def test_a_hook_approval_keeps_its_decision_and_tool_use_id(ledger: Ledger) -> None:
    """The hook's own record shape (``osprey_hook_log.emit_audit``): ``tool_use_id`` last."""
    stack, identity = ledger.stack, LEDGER["hook"]
    record = {
        "ts": _stamp(datetime.now(UTC)),
        "surface": "hook_approval",
        "actor": identity,
        "posture": "writes",
        "posture_source": "process",
        "session": None,
        "subject": "mcp__controls__channel_write",
        "decision": DECISION_APPROVED,
        "reason": "user_approved",
        "tool_use_id": "toolu_hook_approval_01",
    }
    _write_raw(
        stack,
        identity,
        "hook_approval",
        (json.dumps(record, separators=(",", ":")) + "\n").encode(),
    )
    _exact(lambda: _ledger_count(identity), 1, "rows of the hook approval")
    rows = _ledger_rows(identity, "Surface, Decision, ToolUseId, Unparsed")
    assert rows == [["hook_approval", DECISION_APPROVED, "toolu_hook_approval_01", False]], rows


def test_an_executor_envelope_keeps_its_actor_decision_and_tool_use_id(ledger: Ledger) -> None:
    """An executor record carries ``source`` after ``tool_use_id`` and still parses."""
    stack, identity = ledger.stack, LEDGER["executor"]
    _write_envelope(
        stack,
        identity,
        SURFACE_EXECUTOR,
        _envelope(
            identity,
            surface=SURFACE_EXECUTOR,
            subject="execute",
            decision=DECISION_REFUSED,
            tool_use_id="toolu_executor_01",
            source="print('e2e')",
        ),
    )
    _exact(lambda: _ledger_count(identity), 1, "rows of the executor record")
    rows = _ledger_rows(identity, "Surface, Actor, Decision, ToolUseId")
    assert rows == [[SURFACE_EXECUTOR, identity, DECISION_REFUSED, "toolu_executor_01"]], rows


def test_symlinked_ledgers_yield_no_row_and_are_counted_as_drops(ledger: Ledger) -> None:
    """A link to another ledger, to ``/proc/self/environ`` and to the htpasswd: no row each."""
    stack, identity, victim = ledger.stack, LEDGER["links"], LEDGER["victim"]
    for n in range(2):
        _write_envelope(stack, victim, "e2e", _envelope(victim, surface="e2e", subject=f"v-{n}"))
    _exact(lambda: _ledger_count(victim), 2, "the victim's own rows")

    links = _audit_dir(stack, identity)
    before = _counter(stack, "otelcol_processor_filter_logs_filtered", filter=LEDGER_PATH_FILTER)
    targets = {
        "ledger.jsonl": f"../{victim}/e2e.jsonl",
        "environ.jsonl": "/proc/self/environ",
        "htpasswd.jsonl": "/etc/otelcol/senders.htpasswd",
    }
    for name, target in targets.items():
        (links / name).symlink_to(target)
    try:
        # Two victim lines, at least one htpasswd line, and the environment
        # block — which has no newline, so it arrives at the reader's flush.
        expected = 2 + 1 + 1
        deadline = time.monotonic() + 180
        drops = 0.0
        while time.monotonic() < deadline:
            drops = (
                _counter(stack, "otelcol_processor_filter_logs_filtered", filter=LEDGER_PATH_FILTER)
                - before
            )
            if drops >= expected:
                break
            time.sleep(2.0)
        assert drops >= expected, f"{drops} lines counted as dropped by the path filter"
        assert _ledger_count(identity) == 0
        assert _ledger_count(victim) == 2, "a linked ledger was stored twice"
    finally:
        for name in targets:
            (links / name).unlink(missing_ok=True)


def test_an_over_cap_tool_call_line_is_one_row_marked_truncated_only(ledger: Ledger) -> None:
    stack, identity = ledger.stack, LEDGER["overcap"]
    cap = tc.gateway_max_log_size(stack.config)
    _write_tool_call(
        stack,
        identity,
        _tool_call(
            identity, tool_use_id="toolu_overcap_01", duration_ms=321.5, result="r" * (cap + 4096)
        ),
    )
    _exact(lambda: _ledger_count(identity), 1, "rows of the over-cap line")
    rows = _ledger_rows(
        identity,
        "Surface, ToolUseId, DurationMs, Truncated, Incomplete, Unparsed, TimeUnparsed, length(Body)",
    )
    assert rows == [["tool_call", "toolu_overcap_01", 321.5, True, False, False, False, cap]], rows


def test_a_line_written_in_two_halves_11_s_apart_is_one_intact_row(ledger: Ledger) -> None:
    stack, identity = ledger.stack, LEDGER["torn"]
    line = _envelope_line(_envelope(identity, surface="e2e", subject="torn"))
    middle = len(line) // 2
    _write_raw(stack, identity, "e2e", line[:middle])
    time.sleep(TORN_GAP_S)
    _write_raw(stack, identity, "e2e", line[middle:])
    _exact(lambda: _ledger_count(identity), 1, "rows of the torn line")
    rows = _ledger_rows(identity, "Body, Surface, Incomplete, Truncated, Unparsed")
    assert rows == [[line.decode().rstrip("\n"), "e2e", False, False, False]], rows


def test_a_non_json_line_is_one_unparsed_row_with_todays_time(ledger: Ledger) -> None:
    stack, identity = ledger.stack, LEDGER["nonjson"]
    _write_raw(stack, identity, "e2e", b"this line is not json at all\n")
    _exact(lambda: _ledger_count(identity), 1, "rows of the non-JSON line")
    rows = _ledger_rows(
        identity,
        "Surface, Unparsed, TimeUnparsed, toString(toDate(Timestamp, 'UTC')), "
        "abs(dateDiff('second', Timestamp, now64(9))) < 900",
    )
    assert rows == [["unparsed", True, False, _today_utc(), True]], rows


def test_an_unparseable_ts_is_one_row_with_todays_time_its_surface_and_the_marker(
    ledger: Ledger,
) -> None:
    stack, identity = ledger.stack, LEDGER["badts"]
    record = _envelope(identity, surface="e2e-badts", subject="bad-time").to_dict()
    record["ts"] = "not-a-time"
    _write_raw(stack, identity, "e2e", (json.dumps(record, separators=(",", ":")) + "\n").encode())
    _exact(lambda: _ledger_count(identity), 1, "rows of the bad-time line")
    rows = _ledger_rows(
        identity,
        "Surface, TimeUnparsed, Unparsed, toString(toDate(Timestamp, 'UTC')), "
        "abs(dateDiff('second', Timestamp, now64(9))) < 900",
    )
    assert rows == [["e2e-badts", True, False, _today_utc(), True]], rows


def test_a_copytruncate_then_3_lines_yields_3_rows(ledger: Ledger) -> None:
    stack, identity = ledger.stack, LEDGER["copytruncate"]
    for n in range(5):
        _write_envelope(
            stack, identity, "e2e", _envelope(identity, surface="e2e", subject=f"old-{n}")
        )
    _exact(lambda: _ledger_count(identity), 5, "rows before the rotation")
    path = _audit_dir(stack, identity) / "e2e.jsonl"
    shutil.copyfile(path, path.with_name("e2e.jsonl.1"))
    with path.open("r+b") as handle:
        handle.truncate(0)
    for n in range(3):
        _write_envelope(
            stack, identity, "e2e", _envelope(identity, surface="e2e", subject=f"new-{n}")
        )
    _exact(lambda: _ledger_count(identity), 8, "rows after the rotation")
    assert _ledger_subjects(identity) == sorted(
        [f"old-{n}" for n in range(5)] + [f"new-{n}" for n in range(3)]
    )


def test_a_nan_record_is_stored_as_valid_json(ledger: Ledger) -> None:
    stack, identity = ledger.stack, LEDGER["nan"]
    _write_tool_call(
        stack, identity, _tool_call(identity, tool_use_id="toolu_nan_01", duration_ms=float("nan"))
    )
    _exact(lambda: _ledger_count(identity), 1, "rows of the NaN record")
    rows = _ledger_rows(
        identity, "isValidJSON(Body), JSONExtractString(Body, 'duration_ms'), ToolUseId, Unparsed"
    )
    assert rows == [[1, "NaN", "toolu_nan_01", False]], rows


def test_half_a_line_then_65_s_of_silence_is_one_incomplete_row_with_its_prefix_fields(
    ledger: Ledger,
) -> None:
    """The half line was written before the gateway started reading (see :func:`ledger`)."""
    identity = LEDGER["incomplete"]
    time.sleep(max(0.0, INCOMPLETE_SILENCE_S - (time.monotonic() - ledger.half_line_since)))
    _exact(lambda: _ledger_count(identity), 1, "rows of the half line")
    record = ledger.half_line_record
    rows = _ledger_rows(
        identity, "Surface, ToolUseId, DurationMs, SessionId, Incomplete, Truncated, Unparsed"
    )
    assert rows == [
        [
            "tool_call",
            record["tool_use_id"],
            record["duration_ms"],
            record["session_id"],
            True,
            False,
            False,
        ]
    ], rows


def _burst_line(n: int, size: int) -> bytes:
    """A ledger line of exactly *size* bytes, newline not counted."""
    head = f'{{"ts":"{_stamp(datetime.now(UTC))}","surface":"burst","subject":"{n:03d}","pad":"'
    tail = '"}'
    pad = size - len(head) - len(tail)
    assert pad > 0
    return (head + "x" * pad + tail + "\n").encode()


def test_r_identities_of_cap_sized_lines_at_gateway_start_neither_restart_nor_oom_kill_it(
    ledger: Ledger,
) -> None:
    """100 lines of exactly ``max_log_size`` in each of R identities, all waiting at start."""
    stack = ledger.stack
    size = tc.gateway_max_log_size(stack.config)
    identities = _burst_identities(ledger.receivers)
    _docker("stop", GATEWAY_CONTAINER, timeout=120)
    try:
        for identity in identities:
            path = _audit_dir(stack, identity) / "burst.jsonl"
            with path.open("wb") as handle:
                for n in range(BURST_LINES):
                    handle.write(_burst_line(n, size))
    finally:
        _docker("start", GATEWAY_CONTAINER)
    try:
        _wait_metrics(stack)
        names = ", ".join(f"'{identity}'" for identity in identities)
        total = BURST_LINES * len(identities)

        def stored() -> int:
            return _scalar(f"SELECT count() FROM telemetry.audit_ledger WHERE User IN ({names})")

        reached = _wait_value(stored, total, timeout=900)
        assert reached == total, f"{reached} of {total} cap-sized lines stored"
        state = _docker(
            "inspect", "-f", "{{.RestartCount}} {{.State.OOMKilled}}", GATEWAY_CONTAINER
        ).stdout.split()
        assert state == ["0", "false"], f"RestartCount, OOMKilled = {state}"
        per_identity = dict(
            line.split("\t")
            for line in _sql(
                "SELECT User, count() FROM telemetry.audit_ledger "
                f"WHERE User IN ({names}) GROUP BY User"
            ).splitlines()
        )
        assert per_identity == {identity: str(BURST_LINES) for identity in identities}
    finally:
        for identity in identities:
            (_audit_root(stack) / identity / "burst.jsonl").unlink(missing_ok=True)


def test_a_fifo_and_a_dev_zero_link_leave_bob_flowing_and_osprey_up_moves_both_aside(
    ledger: Ledger,
) -> None:
    """Health names both; ``osprey up`` renames both and prints them; the next stop is clean."""
    stack, bob = ledger.stack, LEDGER["bob"]
    fifo = _audit_dir(stack, LEDGER["fifo"]) / "e2e.jsonl"
    zero = _audit_dir(stack, LEDGER["zero"]) / "e2e.jsonl"
    writer = LedgerWriter(stack, bob)
    writer.start()
    try:
        time.sleep(3)
        os.mkfifo(fifo)
        zero.symlink_to("/dev/zero")
        time.sleep(10)
        row = _health(stack, "telemetry_gateway")["telemetry_gateway_audit_files"]
        assert row["status"] == "warning", row
        assert str(fifo) in row["details"] and str(zero) in row["details"], row

        _rc, output = _wide_verb(stack, "up", "-d")
        _wait_ready(stack)
        flat = _flat(output)
        assert _flat(str(fifo)) in flat and _flat(str(zero)) in flat, output
        for path in (fifo, zero):
            assert not os.path.lexists(path), f"{path} was not moved aside"
            moved = [
                p.name
                for p in path.parent.iterdir()
                if p.name.startswith(path.name + REFUSED_SUFFIX)
            ]
            assert len(moved) == 1, (
                f"{path.parent}: {sorted(p.name for p in path.parent.iterdir())}"
            )
        time.sleep(3)
    finally:
        sent = writer.finish()
    _assert_ledger_exactly_once(bob, sent)

    started = time.monotonic()
    _docker("stop", "--time", "30", GATEWAY_CONTAINER, timeout=120)
    took = time.monotonic() - started
    exit_code = _docker("inspect", "-f", "{{.State.ExitCode}}", GATEWAY_CONTAINER).stdout.strip()
    _docker("start", GATEWAY_CONTAINER)
    _wait_metrics(stack)
    assert exit_code == "0" and took < 30, (
        f"the stop was not clean: exit {exit_code} after {took:.1f} s"
    )


def test_an_unrenameable_non_regular_ledger_path_fails_osprey_up_naming_it(ledger: Ledger) -> None:
    stack = ledger.stack
    directory = _audit_dir(stack, LOCKED_IDENTITY)
    hostile = directory / "e2e.jsonl"
    hostile.symlink_to("/dev/null")
    directory.chmod(0o555)
    try:
        rc, output = _wide_verb(stack, "up", "-d", check=False)
        assert rc != 0, f"osprey up succeeded with an unrenameable ledger path:\n{output}"
        assert _flat(str(hostile)) in _flat(output), output
    finally:
        directory.chmod(0o755)
        hostile.unlink(missing_ok=True)
        directory.rmdir()
    _verb(stack, "up", "-d")
    _wait_ready(stack)


def test_a_14_day_backfill_while_the_store_is_down_then_5_current_lines_are_stored_once(
    ledger: Ledger,
) -> None:
    stack, identity = ledger.stack, LEDGER["backfill"]
    now = datetime.now(UTC)
    expected: list[str] = []
    days: set[str] = set()
    _docker("stop", STORE_CONTAINER, timeout=120)
    try:
        for age in range(BACKFILL_DAYS):
            when = now - timedelta(days=age, minutes=5)
            days.add(when.date().isoformat())
            for n in range(BACKFILL_PER_DAY):
                subject = f"back-{age:02d}-{n}"
                _write_envelope(
                    stack,
                    identity,
                    "e2e",
                    _envelope(identity, surface="e2e", subject=subject, when=when),
                )
                expected.append(subject)
        time.sleep(10)
    finally:
        _start_store(stack)
    for n in range(5):
        subject = f"current-{n}"
        _write_envelope(stack, identity, "e2e", _envelope(identity, surface="e2e", subject=subject))
        expected.append(subject)
    _exact(lambda: _ledger_count(identity), len(expected), "back-filled and current rows")
    assert _ledger_subjects(identity) == sorted(expected)
    stored_days = set(
        _sql(
            "SELECT DISTINCT toString(toDate(Timestamp, 'UTC')) FROM telemetry.audit_ledger "
            f"WHERE User = '{identity}' AND startsWith(JSONExtractString(Body, 'subject'), 'back-')"
        ).split()
    )
    assert stored_days == days


def test_ledger_lines_are_stored_exactly_once_across_a_store_pause(ledger: Ledger) -> None:
    stack, identity = ledger.stack, LEDGER["pause"]
    writer = LedgerWriter(stack, identity)
    writer.start()
    try:
        time.sleep(3)
        _docker("pause", STORE_CONTAINER)
        try:
            time.sleep(PAUSE_S)
        finally:
            _docker("unpause", STORE_CONTAINER)
        time.sleep(3)
    finally:
        sent = writer.finish()
    _assert_ledger_exactly_once(identity, sent)


def test_ledger_lines_are_stored_exactly_once_across_a_store_stop_and_a_gateway_restart(
    ledger: Ledger,
) -> None:
    stack, identity = ledger.stack, LEDGER["stop"]
    writer = LedgerWriter(stack, identity)
    writer.start()
    try:
        time.sleep(3)
        _docker("stop", STORE_CONTAINER, timeout=120)
        stopped = time.monotonic()
        try:
            time.sleep(10)
            _docker("restart", GATEWAY_CONTAINER, timeout=120)
            time.sleep(max(0.0, STOP_S - (time.monotonic() - stopped)))
        finally:
            _start_store(stack)
        time.sleep(3)
    finally:
        sent = writer.finish()
    _assert_ledger_exactly_once(identity, sent)


def test_a_gateway_killed_right_after_a_committed_batch_stores_every_line_once_after_limit_1_by(
    ledger: Ledger,
) -> None:
    """At least once across SIGKILL; exactly once read through the natural key."""
    stack, identity = ledger.stack, LEDGER["kill"]
    sent = 200
    for n in range(sent):
        _write_envelope(
            stack, identity, "e2e", _envelope(identity, surface="e2e", subject=f"k-{n:03d}")
        )
    reached = _wait_value(lambda: _ledger_count(identity), sent)
    _docker("kill", "--signal", "KILL", GATEWAY_CONTAINER)
    assert reached >= sent, f"{reached} of {sent} lines committed before the kill"
    _docker("start", GATEWAY_CONTAINER)
    _wait_metrics(stack)
    # A line written after the restart marks the point the reader is past
    # everything it may re-read.
    _write_envelope(stack, identity, "e2e", _envelope(identity, surface="e2e", subject="k-after"))
    _wait_value(
        lambda: _scalar(
            "SELECT count() FROM telemetry.audit_ledger WHERE User = "
            f"'{identity}' AND JSONExtractString(Body, 'subject') = 'k-after'"
        ),
        1,
    )
    time.sleep(SETTLE_S)
    assert _ledger_count(identity) >= sent + 1
    deduplicated = _scalar(
        "SELECT count() FROM (SELECT Body FROM telemetry.audit_ledger "
        f"WHERE User = '{identity}' LIMIT 1 BY User, Timestamp, cityHash64(Body))"
    )
    assert deduplicated == sent + 1


# --------------------------------------------------------------------------- #
# Renders the cases switch to and back from
# --------------------------------------------------------------------------- #


def _render_variant(
    stack: Stack, base: str, *, roster: list[str] | None = None, bluesky: bool = False
) -> dict[str, Any]:
    """Build *base* plus a roster or a declared ``bluesky``, and recreate only the gateway.

    The variant's other services are never started: only the gateway's
    Collector file depends on them, and it is all these cases read.
    """
    profile = yaml.safe_load(base)
    config = profile["config"]
    if roster:
        config["deploy.fqdn"] = "127.0.0.1"
        config["registry.url"] = "127.0.0.1:5999/osprey-e2e"
        config["modules.web_terminals"] = {
            "enabled": True,
            "image_source": "registry",
            "default_persona": "assistant",
            "users": [{"name": name, "index": index} for index, name in enumerate(roster)],
            "auth": {"method": "password", "allow_insecure_http": True},
        }
        config["modules.web_terminals.personas"] = {
            "assistant": {"project": f"{PROJECT_NAME}-assistant"}
        }
    if bluesky:
        config["deployed_services"] = [*config["deployed_services"], "bluesky"]
        profile["bluesky"] = {"tiled_enabled": False}
    (stack.repo / "profile.yml").write_text(yaml.safe_dump(profile, sort_keys=False))
    _verb(stack, "build", "--skip-deps", "--skip-lifecycle")
    stack.reload()
    if bluesky:
        # Compose interpolates every service of a file even when it starts one.
        _ensure_bluesky_control_plane_keys(stack.config, stack.repo / ".env")
        _ensure_bluesky_document_plane_certs(stack.config, stack.repo / ".env")
    _recreate_gateway(stack)
    return _collector(stack)


def _recreate_gateway(stack: Stack) -> None:
    files = as_built_compose_files(stack.config, stack.repo)
    base_cmd = compose_base_cmd(
        get_runtime_command(stack.config), files, stack.repo, _env_file_args(stack.repo, None)
    )
    run_env = runtime_env(stack.config, {**_env(), **compose_provider_env(None, stack.repo)})
    result = subprocess.run(
        [*base_cmd, "up", "-d", "--no-deps", "--force-recreate", tc.GATEWAY_SERVICE],
        env=run_env,
        capture_output=True,
        text=True,
        timeout=VERB_TIMEOUT_S,
        check=False,
    )
    assert result.returncode == 0, f"recreating the gateway: {result.stderr}"
    _wait_metrics(stack)


def _restore(stack: Stack, base: str) -> None:
    """Put the module's own profile back, build it and bring the deployment up on it."""
    (stack.repo / "profile.yml").write_text(base)
    _verb(stack, "build", "--skip-deps", "--skip-lifecycle")
    stack.reload()
    _verb(stack, "up", "-d")
    _recreate_gateway(stack)
    _wait_ready(stack)


def test_on_behalf_of_is_kept_from_the_sidecar_and_a_queueserver_preflight_only(
    ledger: Ledger,
) -> None:
    """Admin-side: ``OnBehalfOf`` as stored. Who may read which row is the access module's."""
    stack = ledger.stack
    sidecar, dispatch, queueserver = LEDGER["sidecar"], LEDGER["dispatch"], LEDGER["queueserver"]
    base = (stack.repo / "profile.yml").read_text()
    try:
        _render_variant(stack, base, bluesky=True)
        assert trusted_on_behalf_of(stack.config) == ("sidecar", ("queueserver",))
        claim = "alice"
        _write_envelope(
            stack,
            sidecar,
            "web_auth",
            _envelope(sidecar, surface="web_auth", subject="login", on_behalf_of=claim),
        )
        _write_envelope(
            stack,
            dispatch,
            "dispatch",
            _envelope(dispatch, surface="dispatch", subject="job", on_behalf_of=claim),
        )
        _write_envelope(
            stack,
            queueserver,
            "preflight",
            _envelope(queueserver, surface="preflight", subject="plan", on_behalf_of=claim),
        )
        _write_envelope(
            stack,
            queueserver,
            "queue",
            _envelope(queueserver, surface="queue", subject="submit", on_behalf_of=claim),
        )
        _write_tool_call(stack, queueserver, _tool_call(queueserver, tool_use_id="toolu_qs_01"))
        for identity, count in ((sidecar, 1), (dispatch, 1), (queueserver, 3)):
            _exact(lambda identity=identity: _ledger_count(identity), count, f"rows of {identity}")
        assert _ledger_rows(sidecar, "Surface, OnBehalfOf") == [["web_auth", claim]]
        assert _ledger_rows(dispatch, "Surface, OnBehalfOf") == [["dispatch", ""]]
        assert sorted(_ledger_rows(queueserver, "Surface, OnBehalfOf")) == [
            ["preflight", claim],
            ["queue", ""],
            ["tool_call", ""],
        ]
    finally:
        _restore(stack, base)


def test_removing_a_roster_user_keeps_its_receiver_and_duplicates_nothing(ledger: Ledger) -> None:
    stack, identity = ledger.stack, ROSTER_REMOVED
    base = (stack.repo / "profile.yml").read_text()
    receiver = ledger_receiver_id(identity)
    try:
        first = _render_variant(stack, base, roster=[ROSTER_KEPT, ROSTER_REMOVED])
        assert receiver in first["receivers"]
        for n in range(3):
            _write_envelope(
                stack, identity, "e2e", _envelope(identity, surface="e2e", subject=f"r-{n}")
            )
        _exact(lambda: _ledger_count(identity), 3, "rows while on the roster")

        second = _render_variant(stack, base, roster=[ROSTER_KEPT])
        users = [user["name"] for user in stack.config["modules"]["web_terminals"]["users"]]
        assert users == [ROSTER_KEPT], users
        assert receiver in second["receivers"], "the removed user's receiver changed its id"
        assert second["receivers"][receiver] == first["receivers"][receiver]
        for n in range(3, 5):
            _write_envelope(
                stack, identity, "e2e", _envelope(identity, surface="e2e", subject=f"r-{n}")
            )
        _exact(lambda: _ledger_count(identity), 5, "rows after the removal")
        assert _ledger_subjects(identity) == [f"r-{n}" for n in range(5)]
    finally:
        _restore(stack, base)


def test_a_directory_created_after_the_render_is_unfed_listed_and_fed_once_after_the_next_build(
    ledger: Ledger,
) -> None:
    stack, identity = ledger.stack, LATE_IDENTITY
    assert identity not in fed_identities(_collector(stack))
    for n in range(2):
        _write_envelope(
            stack, identity, "e2e", _envelope(identity, surface="e2e", subject=f"late-{n}")
        )

    row = _health(stack, "telemetry_gateway")["telemetry_gateway_unfed"]
    assert identity in row["message"], row
    _rc, output = _wide_verb(stack, "up", "-d")
    _wait_ready(stack)
    assert "not fed until the next build" in output and identity in output, output
    time.sleep(SETTLE_S)
    assert _ledger_count(identity) == 0, "an unfed directory was read"

    _verb(stack, "build", "--skip-deps", "--skip-lifecycle")
    stack.reload()
    assert identity in fed_identities(_collector(stack))
    _verb(stack, "up", "-d")
    _wait_ready(stack)
    _exact(lambda: _ledger_count(identity), 2, "rows of the late directory")
    assert _ledger_subjects(identity) == ["late-0", "late-1"]
