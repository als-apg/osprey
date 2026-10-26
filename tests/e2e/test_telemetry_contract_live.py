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
import sys
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
    LOGS_PATH,
    OTLP_PROTOCOL,
    RECORD_LOGIN_ENV,
    RECORD_OFF_ENV,
    RECORD_URL_ENV,
    TERMINAL_SESSION_ENV,
)
from osprey.cli.templates.scaffolding import _DEFAULT_CLAUDE_CLI_VERSION
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
CAPTURE_ENV = "OSPREY_CAPTURE_TELEMETRY_FIXTURE"

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
#: Event names the CLI's background housekeeping emits. It starts on a timer
#: that does not hold the process open, so a turn that ends first never emits
#: them; whether a run carries them says how long it lasted, not what shape
#: the CLI speaks. Neither side of the comparison holds them.
_HOUSEKEEPING_EVENTS = frozenset({"retention_sweep"})
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


def _osprey_bin() -> Path:
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


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
    osprey = _osprey_bin()
    base = tmp_path_factory.mktemp("telemetry_contract")
    repo = base / PROJECT_NAME
    provider = build_provider(e2e_provider())
    edits = {"provider": provider, "model": build_model(None)}
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


def _host_replacements() -> dict[str, str]:
    """The capturing host's home directory and account name, longest first."""
    pairs = {str(Path.home()): "~", getpass.getuser(): "<user>"}
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
    for events in (expected["events"], live["events"]):
        for name in _HOUSEKEEPING_EVENTS:
            events.pop(name, None)
    missing = sorted(set(expected["events"]) - set(live["events"]))
    extra = sorted(set(live["events"]) - set(expected["events"]) - _TRANSIENT_EVENTS)
    assert not missing and not extra, (
        "event names differ from the fixture",
        {"missing": missing, "extra": extra},
    )
    for name, keys in expected["events"].items():
        assert live["events"][name] == keys, (
            f"attribute keys of {name!r} differ from the fixture",
            sorted(set(live["events"][name]) ^ set(keys)),
        )
    assert live["resource_keys"] == expected["resource_keys"], (
        "resource keys differ from the fixture",
        sorted(set(live["resource_keys"]) ^ set(expected["resource_keys"])),
    )
    assert recorded["authorization"] == "Basic <identity>:<password>"
