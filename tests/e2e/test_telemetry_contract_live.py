"""The OTLP contract between the pinned Claude Code CLI and agent-record's receiver, live.

agent-record's OTLP receiver reads names the CLI chooses: event names,
attribute keys, resource keys. This module runs the real CLI — the one version
the deployment pins, ``claude_code.cli_version`` — with the OTLP block the real
launch builder writes (:func:`osprey.build.claude_code_telemetry.record_telemetry_env`)
for a launch handed a record address and a record login, pointed at an
in-process receiver that captures what is posted to ``/v1/logs``, for one model
turn that runs one tool.

With ``OSPREY_CAPTURE_TELEMETRY_FIXTURE=1`` it writes what the receiver got,
decoded to OTLP/JSON, to ``tests/fixtures/telemetry/claude_cli_session.otlp.json``
with the CLI's version and the capture time; ``tests/agent_record/test_otlp_receiver.py``
posts that capture to the real receiver. Otherwise it compares the live run
with that fixture structurally: the set of event names, the attribute keys of
each event, the resource keys and the shape of the Authorization header. Ids,
times and values differ on every run and are not compared.

The same switch records the launches the agent record replays. A host launch
(``osprey chat --print`` and ``osprey query``, the real commands) runs against
the built project with its record address and login pointed at the receiver
and its config directory in the test's temp tree, so it writes its session
transcripts there; each turn starts a subagent and thinks, and the query's
subagent reads a file large enough to give an oversized tool result. Each
launch is written to ``tests/fixtures/agent_record/<launch>/``: ``capture.json``
(the CLI version, the capture time, the command and what the session holds),
``otlp.json`` (what the receiver got, scrubbed as the contract fixture is) and
``transcripts/`` (the session's directory under ``projects/``, with the
capturing host's paths and account name replaced). The launches that run in a
container, the web terminal and dispatch, need a running stack and are not
captured here. ``tests/build_pipeline/test_telemetry_fixture_cli_pin.py``
holds every capture to the pinned version and the scrub.

Reaches a model, so it carries no ``model_free`` marker and runs in the
``e2e-tests`` lane under ``full-ci``.
"""

from __future__ import annotations

import base64
import getpass
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from google.protobuf import json_format

from osprey.agent_record.ingest.otlp import PROTOBUF_CONTENT_TYPE, decode_logs
from osprey.build.claude_code_telemetry import (
    HOST_SESSION_PREFIX,
    LOGS_PATH,
    OTLP_PROTOCOL,
    RECORD_LOGIN_ENV,
    RECORD_OFF_ENV,
    RECORD_URL_ENV,
    TERMINAL_SESSION_ENV,
)
from osprey.cli.templates.scaffolding import _DEFAULT_CLAUDE_CLI_VERSION
from tests.e2e._deploy_support import find_osprey_console_script
from tests.e2e.profile_edits import set_pairs
from tests.e2e.provider import build_model, build_provider, e2e_provider

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.requires_e2e_provider,
]

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "telemetry" / "claude_cli_session.otlp.json"
)
#: Where each launch's capture is written, one directory per launch.
CAPTURES = Path(__file__).resolve().parents[1] / "fixtures" / "agent_record"
CAPTURE_ENV = "OSPREY_CAPTURE_TELEMETRY_FIXTURE"
#: The project's ``deployment.host_identity``: a host launch records under it,
#: and the capture carries it in place of the capturing account's name.
HOST_IDENTITY = "e2e-builder"

PROJECT_NAME = "osprey-e2e-telemetry-contract"
CLI_PACKAGE = "@anthropic-ai/claude-code"
TURN_TIMEOUT_S = 300
#: The prompt: one model turn that runs one shell command, so the session
#: carries a prompt, an API request, a tool decision and a tool result.
PROMPT = "Run the shell command `echo osprey-contract` and reply with exactly what it printed."
ALLOWED_TOOLS = "Bash(echo:*)"
#: Attribute keys whose values identify the account the capture ran under.
#: The keys stay in the fixture; their values do not.
_IDENTIFYING = re.compile(r"^(user\.|organization\.|host\.)")
#: Attribute keys whose values carry session content (API bodies, the prompt,
#: the reply, tool input and output). The contract pins keys, never content,
#: so the fixture keeps the keys and drops these values.
_CONTENT = frozenset(
    {
        "body",
        "prompt",
        "user_prompt",
        "response",
        "tool_input",
        "tool_parameters",
        "output",
        "full_command",
        "bash_command",
    }
)
#: The longest string value the fixture keeps. Ids, names, model and session
#: identifiers fit; anything longer is content and is dropped whatever its key.
#: ``tests/build_pipeline/test_telemetry_fixture_cli_pin.py`` holds the fixture to it.
MAX_KEPT_VALUE = 64
REDACTED = "<redacted>"
#: Event names a turn emits only when a request fails and is retried. They
#: say nothing about the CLI's shape, so a live run may carry them although
#: the fixture does not.
_TRANSIENT_EVENTS = frozenset({"api_error"})
#: Event names the CLI emits from work that runs beside the turn rather than
#: in it: the retention sweep starts on a timer that does not hold the process
#: open, and ``mcp_server_connection`` reports the project's MCP servers coming
#: up while the turn is already under way. Whether a run carries them depends
#: on how the two raced, not on what shape the CLI speaks: one capture of the
#: pinned version has them and the next does not. Neither side of the
#: comparison holds them.
_HOUSEKEEPING_EVENTS = frozenset({"retention_sweep", "mcp_server_connection"})
_BASIC = re.compile(r"^Basic ([A-Za-z0-9+/]+=*)$")
#: Log-record fields OTLP/JSON spells as lowercase hex, where the protobuf JSON
#: printer writes base64.
_HEX_ID_FIELDS = ("traceId", "spanId")
#: Launch variables the host running the test may carry and the launch builder
#: reads; none of them may reach the CLI except as the builder writes them.
_LAUNCH_VARS = (RECORD_URL_ENV, RECORD_LOGIN_ENV, RECORD_OFF_ENV, TERMINAL_SESSION_ENV)


# --------------------------------------------------------------------------- #
# The sink
# --------------------------------------------------------------------------- #


@dataclass
class Sink:
    """An in-process OTLP/HTTP receiver that keeps every request it is sent.

    A ``/v1/logs`` body is decoded the way agent-record decodes it
    (:func:`~osprey.agent_record.ingest.otlp.decode_logs`) and kept as
    OTLP/JSON; a request to any other path is kept with an empty body, so a
    CLI that exports a signal the launch turned off is seen, not hidden.
    """

    requests: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    server: ThreadingHTTPServer | None = None

    @property
    def port(self) -> int:
        assert self.server is not None
        return self.server.server_address[1]

    def signals(self) -> set[str]:
        with self.lock:
            return {r["signal"] for r in self.requests}


def _handler(sink: Sink) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            path = self.path.split("?", 1)[0].rstrip("/")
            content_type = self.headers.get("Content-Type", "")
            body: dict[str, Any] = {}
            if path == LOGS_PATH:
                body = otlp_json(
                    decode_logs(raw, content_type, self.headers.get("Content-Encoding"))
                )
            record = {
                "path": path,
                "signal": path.rsplit("/", 1)[-1],
                "content_type": content_type,
                "authorization": self.headers.get("Authorization", ""),
                "body": body,
            }
            with sink.lock:
                sink.requests.append(record)
            # An empty ExportLogsServiceResponse: full success, in protobuf.
            self.send_response(200)
            self.send_header("Content-Type", PROTOBUF_CONTENT_TYPE)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            return

    return Handler


def otlp_json(request: Any) -> dict[str, Any]:
    """*request* (an ``ExportLogsServiceRequest``) as the OTLP/JSON body a CLI would post."""
    data = json_format.MessageToDict(request)
    for resource_logs in data.get("resourceLogs", []):
        for scope_logs in resource_logs.get("scopeLogs", []):
            for record in scope_logs.get("logRecords", []):
                for name in _HEX_ID_FIELDS:
                    if name in record:
                        record[name] = base64.b64decode(record[name]).hex()
    return data


@pytest.fixture
def sink() -> Iterator[Sink]:
    collected = Sink()
    collected.server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(collected))
    thread = threading.Thread(target=collected.server.serve_forever, daemon=True)
    thread.start()
    try:
        yield collected
    finally:
        collected.server.shutdown()
        collected.server.server_close()


# --------------------------------------------------------------------------- #
# The pinned CLI and the project
# --------------------------------------------------------------------------- #


def _clean_env() -> dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if k != "CLAUDECODE"
        and not k.startswith(("OTEL_", "CLAUDE_CODE_", "OSPREY_AGENT_RECORD_"))
        and k not in _LAUNCH_VARS
    }


@pytest.fixture(scope="module")
def pinned_cli(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """The CLI at the deployment's pinned version, installed into a directory of its own."""
    npm = shutil.which("npm")
    if npm is None:
        pytest.fail("npm is required to install the pinned Claude Code CLI")
    prefix = tmp_path_factory.mktemp("pinned_cli")
    install = subprocess.run(
        [
            npm,
            "install",
            "--prefix",
            str(prefix),
            "--no-save",
            "--no-audit",
            "--no-fund",
            f"{CLI_PACKAGE}@{_DEFAULT_CLAUDE_CLI_VERSION}",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert install.returncode == 0, f"npm install failed:\n{install.stdout}\n{install.stderr}"
    cli = prefix / "node_modules" / ".bin" / "claude"
    version = subprocess.run(
        [str(cli), "--version"], capture_output=True, text=True, timeout=60, check=False
    )
    assert version.returncode == 0, version.stderr
    reported = version.stdout.split()[0]
    assert reported == _DEFAULT_CLAUDE_CLI_VERSION, (
        f"the installed CLI reports {reported!r}, the pin is {_DEFAULT_CLAUDE_CLI_VERSION!r}"
    )
    return cli, reported


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A built hello-world repo: the provider and model the launch builder resolves."""
    osprey = find_osprey_console_script()
    base = tmp_path_factory.mktemp("telemetry_contract")
    repo = base / PROJECT_NAME
    provider = build_provider(e2e_provider())
    edits = {
        "provider": provider,
        "model": build_model(None),
        "config": {"deployment": {"host_identity": HOST_IDENTITY}},
    }
    for args, cwd in (
        (["init", str(repo), "--preset", "hello-world", "--no-git", *set_pairs(edits)], base),
        (["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"], base),
    ):
        result = subprocess.run(
            [str(osprey), *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=300,
            env=_clean_env(),
            check=False,
        )
        if result.returncode != 0:
            pytest.fail(f"osprey {args[0]} failed:\n{result.stdout}\n{result.stderr}")
    return repo


def _launch_env(
    project: Path, sink: Sink, login: str, terminal: str, monkeypatch: pytest.MonkeyPatch
) -> dict[str, str]:
    """The CLI's environment as the real launch builder writes it, aimed at *sink*.

    *sink* stands where agent-record stands: the launch is handed its address
    and a record login, exactly the two variables a deployment hands it.
    """
    from osprey.agent_runner.provider_env import load_provider_spec

    url = f"http://127.0.0.1:{sink.port}"
    monkeypatch.delenv(RECORD_OFF_ENV, raising=False)
    for key, value in {
        RECORD_URL_ENV: url,
        RECORD_LOGIN_ENV: login,
        TERMINAL_SESSION_ENV: terminal,
    }.items():
        monkeypatch.setenv(key, value)
    spec = load_provider_spec(project / "build", env_dir=project)
    assert spec is not None, "the project names no provider"
    if spec.needs_proxy:
        pytest.fail(f"provider {spec.provider!r} needs the translation proxy; name a native one")
    env = {**_clean_env(), **spec.env_block}
    if spec.auth_env_var != spec.auth_secret_env:
        secret = os.environ.get(spec.auth_secret_env)
        assert secret, f"{spec.auth_secret_env} is not set"
        env[spec.auth_env_var] = secret
    assert env["OTEL_LOGS_EXPORTER"] == "otlp", env.get("OTEL_LOGS_EXPORTER")
    assert env["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"] == f"{url}{LOGS_PATH}"
    assert env["OTEL_EXPORTER_OTLP_PROTOCOL"] == OTLP_PROTOCOL
    return env


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


def _attr_keys(attributes: list[Mapping[str, Any]] | None) -> set[str]:
    return {a["key"] for a in attributes or []}


def _attr_value(attributes: list[Mapping[str, Any]] | None, key: str) -> str | None:
    for a in attributes or []:
        if a["key"] == key:
            value = a.get("value") or {}
            return value.get("stringValue")
    return None


def _event_name(record: Mapping[str, Any]) -> str:
    name = _attr_value(record.get("attributes"), "event.name")
    if name:
        return name
    return str((record.get("body") or {}).get("stringValue", ""))


def structure(requests: list[Mapping[str, Any]]) -> dict[str, Any]:
    """What the contract pins: names and keys, never values."""
    events: dict[str, set[str]] = {}
    resource_keys: set[str] = set()
    metrics: set[str] = set()
    for request in requests:
        body = request["body"]
        for resource_logs in body.get("resourceLogs", []):
            resource_keys |= _attr_keys((resource_logs.get("resource") or {}).get("attributes"))
            for scope_logs in resource_logs.get("scopeLogs", []):
                for record in scope_logs.get("logRecords", []):
                    keys = _attr_keys(record.get("attributes"))
                    events.setdefault(_event_name(record), set()).update(keys)
        for resource_metrics in body.get("resourceMetrics", []):
            resource_keys |= _attr_keys((resource_metrics.get("resource") or {}).get("attributes"))
            for scope_metrics in resource_metrics.get("scopeMetrics", []):
                metrics.update(m["name"] for m in scope_metrics.get("metrics", []))
    return {
        "events": {name: sorted(keys) for name, keys in sorted(events.items())},
        "resource_keys": sorted(resource_keys),
        "metrics": sorted(metrics),
    }


def contract_events(
    expected: Mapping[str, list[str]], live: Mapping[str, list[str]]
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    """The events both sides are held to, and the event names that differ.

    Housekeeping events leave both sides; a transient event may appear live
    without the fixture. Returns ``(expected, live, drift)``, where *drift*
    maps ``missing`` and ``extra`` to sorted event names.
    """
    kept_expected = {n: k for n, k in expected.items() if n not in _HOUSEKEEPING_EVENTS}
    kept_live = {n: k for n, k in live.items() if n not in _HOUSEKEEPING_EVENTS}
    drift = {
        "missing": sorted(set(kept_expected) - set(kept_live)),
        "extra": sorted(set(kept_live) - set(kept_expected) - _TRANSIENT_EVENTS),
    }
    return kept_expected, kept_live, drift


def header_shape(authorization: str, login: str | None = None) -> str:
    """``Basic <identity>:<password>`` when the header is a Basic login, else what it is."""
    match = _BASIC.match(authorization)
    if not match:
        return f"not Basic: {authorization.split(' ', 1)[0]!r}"
    decoded = base64.b64decode(match.group(1)).decode()
    identity, sep, password = decoded.partition(":")
    if not (sep and identity and password):
        return "Basic <not identity:password>"
    if login is not None and decoded != login:
        return "Basic <a different login>"
    return "Basic <identity>:<password>"


def _redact(node: Any, replacements: Mapping[str, str]) -> Any:
    """The OTLP body with identifying values and session content replaced.

    Identifying attributes, content attributes and any string attribute value
    longer than :data:`MAX_KEPT_VALUE` lose their value; keys stay. Inside
    every other string, each key of *replacements* (the capturing host's home
    directory and account name) is replaced by its value.
    """
    if isinstance(node, list):
        return [_redact(item, replacements) for item in node]
    if isinstance(node, dict):
        if set(node) == {"key", "value"}:
            key = str(node["key"])
            value = node["value"] if isinstance(node["value"], dict) else {}
            text = value.get("stringValue")
            if (
                _IDENTIFYING.match(key)
                or key in _CONTENT
                or (isinstance(text, str) and len(text) > MAX_KEPT_VALUE)
            ):
                return {"key": key, "value": {"stringValue": REDACTED}}
        return {k: _redact(v, replacements) for k, v in node.items()}
    if isinstance(node, str):
        for old, new in replacements.items():
            node = node.replace(old, new)
    return node


def _host_replacements(temp_roots: Mapping[str, str] | None = None) -> dict[str, str]:
    """The capturing host's home directory and account name, longest first.

    *temp_roots* maps further directories of the capturing host (the test's
    temp tree, the system temp directory) to what stands in their place. Each
    is replaced in both spellings a macOS host writes, the resolved one under
    ``/private`` and the one the process was handed.
    """
    pairs = {str(Path.home()): "~", getpass.getuser(): "<user>"}
    for root, placeholder in (temp_roots or {}).items():
        for spelling in {root, os.path.realpath(root)}:
            pairs[spelling.rstrip("/")] = placeholder
            if spelling.startswith("/private/"):
                pairs[spelling[len("/private") :].rstrip("/")] = placeholder
    return dict(sorted(pairs.items(), key=lambda item: -len(item[0])))


# --------------------------------------------------------------------------- #
# The test
# --------------------------------------------------------------------------- #


def test_the_pinned_cli_speaks_the_otlp_shape_the_fixture_records(
    project: Path,
    pinned_cli: tuple[Path, str],
    sink: Sink,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli, version = pinned_cli
    login = f"contract:{uuid.uuid4().hex}"
    terminal = f"contract-{uuid.uuid4().hex[:12]}"
    env = _launch_env(project, sink, login, terminal, monkeypatch)
    # A config dir of its own: no user settings, plugins or hooks of the host
    # running the test reach the CLI.
    config_dir = tmp_path / "claude-config"
    config_dir.mkdir()
    env["CLAUDE_CONFIG_DIR"] = str(config_dir)
    workdir = tmp_path / "work"
    workdir.mkdir()

    turn = subprocess.run(
        [
            str(cli),
            "-p",
            PROMPT,
            "--output-format",
            "json",
            "--allowedTools",
            ALLOWED_TOOLS,
        ],
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=TURN_TIMEOUT_S,
        check=False,
    )
    assert turn.returncode == 0, f"the CLI turn failed:\n{turn.stdout}\n{turn.stderr}"
    assert "osprey-contract" in turn.stdout, turn.stdout

    requests = list(sink.requests)
    # Logs only, to the one path agent-record receives them on: traces and
    # metrics are off in the launch, and agent-record answers 404 on both.
    assert requests, "the CLI exported nothing"
    assert {r["path"] for r in requests} == {LOGS_PATH}, sorted({r["path"] for r in requests})
    for request in requests:
        assert request["content_type"].startswith(PROTOBUF_CONTENT_TYPE), request["content_type"]
    shapes = {header_shape(r["authorization"], login) for r in requests}
    assert shapes == {"Basic <identity>:<password>"}, shapes

    live = structure(requests)
    ran_the_tool = any(
        _attr_value(record.get("attributes"), "event.name") == "tool_result"
        and _attr_value(record.get("attributes"), "tool_name") == "Bash"
        for r in requests
        for rl in r["body"].get("resourceLogs", [])
        for sl in rl.get("scopeLogs", [])
        for record in sl.get("logRecords", [])
    )
    assert ran_the_tool, (
        f"the model did not run the tool; not a contract change. Re-run the case:\n{turn.stdout}"
    )
    assert "terminal.session" in live["resource_keys"], live["resource_keys"]
    terminals = {
        _attr_value((rl.get("resource") or {}).get("attributes"), "terminal.session")
        for r in requests
        for rl in r["body"].get("resourceLogs", [])
    }
    assert terminals == {terminal}, terminals
    for name, keys in live["events"].items():
        assert "session.id" in keys, f"event {name!r} carries no session.id: {keys}"

    if os.environ.get(CAPTURE_ENV) == "1":
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        captured = {
            "cli_version": version,
            "captured_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "protocol": OTLP_PROTOCOL,
            "authorization": "Basic <identity>:<password>",
            "requests": [
                {"signal": r["signal"], "body": _redact(r["body"], _host_replacements())}
                for r in requests
            ],
        }
        FIXTURE.write_text(json.dumps(captured, indent=1, sort_keys=True) + "\n")
        return

    assert FIXTURE.is_file(), f"no fixture at {FIXTURE}; capture it with {CAPTURE_ENV}=1"
    recorded = json.loads(FIXTURE.read_text())
    assert recorded["cli_version"] == version, (
        f"the fixture was captured with CLI {recorded['cli_version']}, the pin is {version}; "
        f"re-capture with {CAPTURE_ENV}=1 under a provider key"
    )
    expected = structure(recorded["requests"])
    expected_events, live_events, drift = contract_events(expected["events"], live["events"])
    assert not drift["missing"] and not drift["extra"], (
        "event names differ from the fixture",
        drift,
    )
    for name, keys in expected_events.items():
        assert live_events[name] == keys, (
            f"attribute keys of {name!r} differ from the fixture",
            sorted(set(live_events[name]) ^ set(keys)),
        )
    assert live["resource_keys"] == expected["resource_keys"], (
        "resource keys differ from the fixture",
        sorted(set(live["resource_keys"]) ^ set(expected["resource_keys"])),
    )
    assert recorded["authorization"] == "Basic <identity>:<password>"


# --------------------------------------------------------------------------- #
# The launches the agent record replays
# --------------------------------------------------------------------------- #

#: A tool result at least this large, serialized, is oversized: the replay
#: carries one result well past every other entry of a session.
OVERSIZED_RESULT_BYTES = 64 * 1024
#: Lines in the file the query's subagent reads; with the line prefix the
#: Read tool adds, its result passes :data:`OVERSIZED_RESULT_BYTES`.
_LARGE_FILE_LINES = 1200
#: Environment variables whose values the capture must never hold: a launch's
#: credentials, whatever provider holds them.
_SECRET_SUFFIXES = ("_API_KEY", "_AUTH_TOKEN", "_SECRET", "_PASSWORD")


@dataclass(frozen=True)
class HostLaunch:
    """One command that starts a session on the host, as an operator types it."""

    name: str
    command: tuple[str, ...]
    #: The prompt; ``{file}`` is the file the subagent reads.
    prompt: str
    #: What the file holds on its last line, and so what the answer carries.
    marker: str
    oversized: bool


HOST_LAUNCHES = (
    HostLaunch(
        name="host-chat",
        command=("chat", "--print"),
        prompt=(
            "Think it through first. Then use the Agent tool to start one general-purpose "
            "subagent that reads the file {file} with the Read tool and reports its last "
            "line. Reply with exactly that line."
        ),
        marker="osprey-capture-host-chat",
        oversized=False,
    ),
    HostLaunch(
        name="host-query",
        command=("query", "--json"),
        prompt=(
            "Think it through first. Then use the Agent tool to start one general-purpose "
            "subagent that reads the whole file {file} with one Read call (no offset or "
            "limit) and reports its first line. Reply with exactly that line."
        ),
        marker="osprey-capture-host-query",
        oversized=True,
    ),
)


def _write_read_file(project: Path, launch: HostLaunch) -> Path:
    """The file *launch*'s subagent reads, in the uploads directory the agent may read."""
    uploads = project / "var" / "agent_data" / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / f"{launch.name}.txt"
    if launch.oversized:
        filler = "abcdefghij" * 5
        lines = [launch.marker] + [
            f"{i:05d} osprey-capture-line {filler}" for i in range(1, _LARGE_FILE_LINES)
        ]
    else:
        lines = ["osprey capture", launch.marker]
    path.write_text("\n".join(lines) + "\n")
    return path


def _entries(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _blocks(entries: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for entry in entries:
        content = (entry.get("message") or {}).get("content")
        if isinstance(content, list):
            yield from (block for block in content if isinstance(block, dict))


def _session_dir(config_dir: Path) -> Path:
    """The one ``projects/<encoded build dir>/`` directory the launch wrote."""
    projects = config_dir / "projects"
    found = [d for d in projects.iterdir() if d.is_dir()] if projects.is_dir() else []
    assert len(found) == 1, f"expected one project directory under {projects}: {found}"
    return found[0]


def _scrub(text: str, replacements: Mapping[str, str]) -> str:
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _secrets() -> list[str]:
    return [
        value
        for name, value in os.environ.items()
        if name.endswith(_SECRET_SUFFIXES) and len(value) >= 16
    ]


@pytest.mark.parametrize("launch", HOST_LAUNCHES, ids=[launch.name for launch in HOST_LAUNCHES])
def test_a_host_launch_records_a_subagent_and_thinking(
    launch: HostLaunch,
    project: Path,
    pinned_cli: tuple[Path, str],
    sink: Sink,
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    cli, version = pinned_cli
    read_file = _write_read_file(project, launch)
    config_dir = tmp_path / "claude-config"
    home = tmp_path / "home"
    config_dir.mkdir()
    home.mkdir()
    env = _clean_env()
    # The launch's own config dir and home: its transcripts land in the
    # test's temp tree, and no settings, memory or hooks of the host reach it.
    env["CLAUDE_CONFIG_DIR"] = str(config_dir)
    env.setdefault("npm_config_cache", str(Path.home() / ".npm"))
    env["HOME"] = str(home)
    # The pinned CLI first on PATH: a headless run executes the `claude` it
    # finds there, and `osprey chat` runs the pin through npx.
    env["PATH"] = f"{cli.parent}{os.pathsep}{env.get('PATH', '')}"
    env[RECORD_URL_ENV] = f"http://127.0.0.1:{sink.port}"
    env[RECORD_LOGIN_ENV] = f"{HOST_IDENTITY}:{uuid.uuid4().hex}"

    run = subprocess.run(
        [
            str(find_osprey_console_script()),
            *launch.command,
            "--repo",
            str(project),
            launch.prompt.format(file=read_file),
        ],
        cwd=str(project),
        env=env,
        capture_output=True,
        text=True,
        timeout=TURN_TIMEOUT_S,
        check=False,
    )
    assert run.returncode == 0, f"osprey {launch.command[0]} failed:\n{run.stdout}\n{run.stderr}"
    assert launch.marker in run.stdout, (
        f"the answer does not carry {launch.marker!r}; the model did not run the subagent's "
        f"read. Re-run the case:\n{run.stdout}"
    )

    session_dir = _session_dir(config_dir)
    mains = sorted(session_dir.glob("*.jsonl"))
    subagents = sorted(session_dir.glob("*/subagents/*.jsonl"))
    assert len(mains) == 1, mains
    assert subagents, f"the session started no subagent: {sorted(session_dir.rglob('*'))}"
    main = _entries(mains[0])
    kinds = {block.get("type") for block in _blocks(main)}
    assert "thinking" in kinds, f"the session holds no thinking block: {sorted(map(str, kinds))}"
    assert any(b.get("name") == "Agent" for b in _blocks(main) if b.get("type") == "tool_use")
    if launch.oversized:
        sizes = [
            len(json.dumps(block.get("content")))
            for path in subagents
            for block in _blocks(_entries(path))
            if block.get("type") == "tool_result"
        ]
        assert max(sizes, default=0) >= OVERSIZED_RESULT_BYTES, sizes

    requests = list(sink.requests)
    assert requests, "the launch exported nothing"
    assert {r["path"] for r in requests} == {LOGS_PATH}, sorted({r["path"] for r in requests})
    sessions = {
        _attr_value((rl.get("resource") or {}).get("attributes"), "terminal.session")
        for r in requests
        for rl in r["body"].get("resourceLogs", [])
    }
    assert len(sessions) == 1, sessions
    (terminal,) = sessions
    assert terminal and terminal.startswith(HOST_SESSION_PREFIX), terminal

    dest = CAPTURES / launch.name
    if os.environ.get(CAPTURE_ENV) != "1":
        recorded = json.loads((dest / "capture.json").read_text())
        assert recorded["cli_version"] == version, (
            f"the {launch.name} capture was made with CLI {recorded['cli_version']}, the pin "
            f"is {version}; re-capture with {CAPTURE_ENV}=1 under a provider key"
        )
        return

    replacements = _host_replacements(
        {
            str(tmp_path_factory.getbasetemp()): "<tmp>",
            tempfile.gettempdir(): "<tmpdir>",
            str(project): "<project>",
        }
    )
    # The CLI names the session's directory after the build directory's path.
    replacements = dict(
        sorted(
            {**replacements, session_dir.name: "<project-dir>"}.items(), key=lambda i: -len(i[0])
        )
    )
    scrubbed = {
        path.relative_to(session_dir): _scrub(path.read_text(), replacements)
        for path in sorted(session_dir.rglob("*"))
        if path.is_file()
    }
    otlp = {
        "cli_version": version,
        "captured_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "protocol": OTLP_PROTOCOL,
        "authorization": "Basic <identity>:<password>",
        "requests": [
            {"signal": r["signal"], "body": _redact(r["body"], replacements)} for r in requests
        ],
    }
    otlp_text = json.dumps(otlp, indent=1, sort_keys=True) + "\n"
    for secret in _secrets():
        assert all(secret not in text for text in [otlp_text, *scrubbed.values()]), (
            "a credential of this environment reached the capture; nothing was written"
        )
    left = sorted(str(p) for p, text in scrubbed.items() if str(Path.home()) in text)
    assert not left, f"the home directory survived the scrub in {left}"

    if dest.exists():
        shutil.rmtree(dest)
    (dest / "transcripts").mkdir(parents=True)
    for relpath, text in scrubbed.items():
        target = dest / "transcripts" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    (dest / "otlp.json").write_text(otlp_text)
    holds = ["subagent", "thinking"] + (["oversized tool result"] if launch.oversized else [])
    capture = {
        "cli_version": version,
        "captured_at": otlp["captured_at"],
        "launch": launch.name,
        "command": " ".join(("osprey", *launch.command)),
        "identity": HOST_IDENTITY,
        "terminal_session_prefix": HOST_SESSION_PREFIX,
        "holds": holds,
    }
    (dest / "capture.json").write_text(json.dumps(capture, indent=1, sort_keys=True) + "\n")
