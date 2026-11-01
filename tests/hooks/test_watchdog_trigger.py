"""The watchdog trigger hook: which prompts it hands on, and how.

The child path runs as real processes against a stub terminal on loopback, with
a stand-in ``osprey.services.watchdog`` on ``PYTHONPATH`` that reports the argv
and the submission it was given. The decisions the parent makes before it spawns
are also pinned in-process, where a spawn can be counted.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

import osprey.templates.claude_code.claude.hooks.osprey_watchdog_trigger as trigger
from osprey.agent_runner.runner import WATCHDOG_SKIP_ENV_VAR
from osprey_connectors.posture_store import DISPATCH_RUN_ENV_VAR
from osprey_connectors_tests._harness._ports import bound_socket

HOOK = Path(trigger.__file__)
SESSION_KEY = "pool-key-1"
RECORD_SESSION = "11111111-2222-4333-8444-555555555555"
DEADLINE_SECONDS = 20.0

STAND_IN_MAIN = """
import json, os, sys, uuid
args = sys.argv[1:]
path = args[args.index("--submission-file") + 1]
with open(path, encoding="utf-8") as stream:
    submission = json.load(stream)
out = os.path.join(os.environ["TRIGGER_TEST_RECORDS"], uuid.uuid4().hex + ".json")
with open(out + ".tmp", "w", encoding="utf-8") as stream:
    json.dump(
        {
            "args": args,
            "submission": submission,
            "safe_path_env": os.environ.get("PYTHONSAFEPATH"),
        },
        stream,
    )
os.rename(out + ".tmp", out)
"""


class Sandbox:
    """One hook run's environment, temp directory and record-entry stand-in."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path / "tmp"
        self.records = tmp_path / "records"
        self.project = tmp_path / "project"
        for directory in (self.tmp, self.records, self.project):
            directory.mkdir()
        watchdog = tmp_path / "stub" / "osprey" / "services" / "watchdog"
        watchdog.mkdir(parents=True)
        for package in (watchdog, watchdog.parent, watchdog.parent.parent):
            (package / "__init__.py").write_text("")
        (watchdog / "__main__.py").write_text(STAND_IN_MAIN)
        self.env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
            "TMPDIR": str(self.tmp),
            "PYTHONPATH": str(tmp_path / "stub"),
            "CLAUDE_PROJECT_DIR": str(self.project),
            "TRIGGER_TEST_RECORDS": str(self.records),
        }

    def run(self, prompt: Any = "raise the limit", *, hook: Path = HOOK) -> None:
        """Run the hook as Claude Code does; stdout must stay empty."""
        started = subprocess.run(
            [sys.executable, str(hook)],
            input=json.dumps(
                {
                    "hook_event_name": "UserPromptSubmit",
                    "session_id": RECORD_SESSION,
                    "prompt": prompt,
                }
            ),
            capture_output=True,
            text=True,
            env=self.env,
            cwd=self.project,
            timeout=30,
        )
        assert started.returncode == 0
        assert started.stdout == ""

    def submission_files(self) -> list[Path]:
        return list(self.tmp.glob("osprey_watchdog_*"))

    def wait_for_records(self, count: int) -> list[dict[str, Any]]:
        """Wait for *count* record-entry reports, and return them."""
        deadline = time.monotonic() + DEADLINE_SECONDS
        while time.monotonic() < deadline:
            found = list(self.records.glob("*.json"))
            if len(found) >= count:
                return [json.loads(path.read_text()) for path in found]
            time.sleep(0.02)
        raise AssertionError(f"{count} record entries expected, got {len(found)}")

    def wait_until_consumed(self) -> None:
        """Wait until every submission file is gone."""
        deadline = time.monotonic() + DEADLINE_SECONDS
        while self.submission_files():
            assert time.monotonic() < deadline, "the submission file outlived the child"
            time.sleep(0.02)

    def web_session(self, port: int) -> None:
        self.env.update(
            OSPREY_WEB_PORT=str(port),
            OSPREY_SESSION_ID=SESSION_KEY,
            OSPREY_PANEL_TOKEN="panel-secret",
        )


class Terminal:
    """A stub web terminal answering every POST with one fixed status."""

    def __init__(self, status: int) -> None:
        self.requests: list[dict[str, Any]] = []
        requests = self.requests

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                requests.append(
                    {
                        "path": self.path,
                        "auth": self.headers.get("Authorization"),
                        "body": json.loads(body),
                    }
                )
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *_args: Any) -> None:
                return None

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


@pytest.fixture
def terminal_factory():
    terminals: list[Terminal] = []

    def _make(status: int) -> Terminal:
        terminals.append(Terminal(status))
        return terminals[-1]

    yield _make
    for terminal in terminals:
        terminal.close()


class _Spawns:
    """Counts what the hook spawns when it runs in-process."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error=None) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.error = error
        self.tmp = tmp_path
        monkeypatch.setattr(subprocess, "Popen", self._popen)
        monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    def _popen(self, argv: list[str], **kwargs: Any) -> None:
        self.calls.append((argv, kwargs))
        if self.error is not None:
            raise self.error


def _run_in_process(monkeypatch: pytest.MonkeyPatch, stdin: str) -> None:
    import io

    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    assert trigger.main([]) == 0


GOOD_STDIN = json.dumps({"session_id": RECORD_SESSION, "prompt": "raise the limit"})


@pytest.mark.parametrize("name", [trigger.SKIP_ENV_VAR, trigger.DISPATCH_RUN_ENV_VAR])
def test_a_marked_run_exits_without_a_child_or_a_file(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(trigger.SKIP_ENV_VAR, raising=False)
    monkeypatch.delenv(trigger.DISPATCH_RUN_ENV_VAR, raising=False)
    monkeypatch.setenv(name, "1")
    spawns = _Spawns(monkeypatch, tmp_path)

    _run_in_process(monkeypatch, GOOD_STDIN)

    assert spawns.calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["OSPREY_WATCHDOG_SKIP", "OSPREY_DISPATCH_RUN"])
def test_a_marked_process_run_leaves_no_trace(sandbox: Sandbox, name: str) -> None:
    sandbox.env[name] = "1"

    sandbox.run()

    assert sandbox.submission_files() == []
    time.sleep(0.5)
    assert list(sandbox.records.iterdir()) == []


def test_the_marker_names_agree_with_the_code_that_stamps_them() -> None:
    assert trigger.SKIP_ENV_VAR == WATCHDOG_SKIP_ENV_VAR
    assert trigger.DISPATCH_RUN_ENV_VAR == DISPATCH_RUN_ENV_VAR


def test_the_child_is_detached_and_holds_no_pipe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(trigger.SKIP_ENV_VAR, raising=False)
    monkeypatch.delenv(trigger.DISPATCH_RUN_ENV_VAR, raising=False)
    spawns = _Spawns(monkeypatch, tmp_path)

    _run_in_process(monkeypatch, GOOD_STDIN)

    [(argv, kwargs)] = spawns.calls
    assert argv[:3] == [sys.executable, str(HOOK), "--child"]
    assert kwargs["start_new_session"] is True
    for stream in ("stdin", "stdout", "stderr"):
        assert kwargs[stream] == subprocess.DEVNULL


def test_the_submission_file_is_private_and_carries_the_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(trigger.SKIP_ENV_VAR, raising=False)
    monkeypatch.delenv(trigger.DISPATCH_RUN_ENV_VAR, raising=False)
    monkeypatch.setenv("OSPREY_SESSION_ID", SESSION_KEY)
    spawns = _Spawns(monkeypatch, tmp_path)

    _run_in_process(monkeypatch, GOOD_STDIN)

    path = Path(spawns.calls[0][0][3])
    assert path.stat().st_mode & 0o777 == 0o600
    submission = json.loads(path.read_text())
    assert set(submission) == {"session_key", "record_session_id", "submission_id", "prompt"}
    assert submission["session_key"] == SESSION_KEY
    assert submission["record_session_id"] == RECORD_SESSION
    assert submission["prompt"] == "raise the limit"


def test_a_failed_spawn_leaves_no_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(trigger.SKIP_ENV_VAR, raising=False)
    monkeypatch.delenv(trigger.DISPATCH_RUN_ENV_VAR, raising=False)
    spawns = _Spawns(monkeypatch, tmp_path, error=OSError("no fork"))

    _run_in_process(monkeypatch, GOOD_STDIN)

    assert len(spawns.calls) == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "stdin",
    [
        "not json",
        "",
        "[]",
        json.dumps({"session_id": RECORD_SESSION}),
        json.dumps({"session_id": RECORD_SESSION, "prompt": ""}),
        json.dumps({"session_id": RECORD_SESSION, "prompt": 7}),
        json.dumps({"prompt": "raise the limit"}),
    ],
)
def test_a_payload_without_a_prompt_and_session_spawns_nothing(
    stdin: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(trigger.SKIP_ENV_VAR, raising=False)
    monkeypatch.delenv(trigger.DISPATCH_RUN_ENV_VAR, raising=False)
    spawns = _Spawns(monkeypatch, tmp_path)

    _run_in_process(monkeypatch, stdin)

    assert spawns.calls == []
    assert list(tmp_path.iterdir()) == []


def test_a_bare_claude_takes_the_record_only_path(sandbox: Sandbox) -> None:
    sandbox.run()

    [record] = sandbox.wait_for_records(1)
    sandbox.wait_until_consumed()
    args = record["args"]
    assert args[0] == "record"
    assert args[args.index("--project-dir") + 1] == str(sandbox.project)
    assert "--session-key" not in args
    assert record["submission"]["session_key"] is None
    assert record["submission"]["record_session_id"] == RECORD_SESSION
    assert record["submission"]["prompt"] == "raise the limit"


def test_a_session_key_without_a_web_terminal_is_record_only(sandbox: Sandbox) -> None:
    sandbox.env["OSPREY_SESSION_ID"] = SESSION_KEY

    sandbox.run()

    [record] = sandbox.wait_for_records(1)
    assert "--session-key" not in record["args"]


def test_a_web_session_posts_the_submission_with_the_panel_token(
    sandbox: Sandbox, terminal_factory
) -> None:
    terminal = terminal_factory(202)
    sandbox.web_session(terminal.port)

    sandbox.run()
    sandbox.wait_until_consumed()

    [request] = terminal.requests
    assert request["path"] == "/api/watchdog/prompt"
    assert request["auth"] == "Bearer panel-secret"
    assert request["body"]["session_key"] == SESSION_KEY
    assert request["body"]["record_session_id"] == RECORD_SESSION
    assert list(sandbox.records.iterdir()) == []


@pytest.mark.parametrize("status", [500, 503, 429, 403])
def test_a_refused_answer_runs_the_record_entry_with_the_same_submission(
    sandbox: Sandbox, terminal_factory, status: int
) -> None:
    terminal = terminal_factory(status)
    sandbox.web_session(terminal.port)

    sandbox.run()

    [record] = sandbox.wait_for_records(1)
    sandbox.wait_until_consumed()
    args = record["args"]
    assert args[args.index("--session-key") + 1] == SESSION_KEY
    [request] = terminal.requests
    assert record["submission"]["submission_id"] == request["body"]["submission_id"]


def test_a_refused_connection_runs_the_record_entry_with_the_session_key(
    sandbox: Sandbox,
) -> None:
    # Bound and never listening: a connect is refused, and no other bind can take the port.
    with bound_socket() as held:
        sandbox.web_session(held.getsockname()[1])

        sandbox.run()

        [record] = sandbox.wait_for_records(1)
        sandbox.wait_until_consumed()
    args = record["args"]
    assert args[args.index("--session-key") + 1] == SESSION_KEY


def _hook_alone(tmp_path: Path) -> Path:
    """Copy the hook to a directory holding none of its build siblings."""
    alone = tmp_path / "alone"
    alone.mkdir()
    return Path(shutil.copy(HOOK, alone / HOOK.name))


def test_the_hook_posts_with_no_sibling_beside_it(
    sandbox: Sandbox, tmp_path: Path, terminal_factory
) -> None:
    """The catalog declares no dependency, so the file must work shipped alone."""
    terminal = terminal_factory(202)
    sandbox.web_session(terminal.port)

    sandbox.run(hook=_hook_alone(tmp_path))
    sandbox.wait_until_consumed()

    [request] = terminal.requests
    assert request["auth"] == "Bearer panel-secret"
    assert list(sandbox.records.iterdir()) == []


def test_a_failed_post_with_no_sibling_still_holds(
    sandbox: Sandbox, tmp_path: Path, terminal_factory
) -> None:
    terminal = terminal_factory(503)
    sandbox.web_session(terminal.port)

    sandbox.run(hook=_hook_alone(tmp_path))

    [record] = sandbox.wait_for_records(1)
    assert "--session-key" in record["args"]


def test_a_terminal_that_never_answers_falls_back_to_the_record_entry(
    sandbox: Sandbox,
) -> None:
    # Listening and never accepting: the connect succeeds and the answer never comes.
    with bound_socket() as held:
        held.listen(1)
        sandbox.web_session(held.getsockname()[1])

        sandbox.run()

        [record] = sandbox.wait_for_records(1)
        sandbox.wait_until_consumed()
    args = record["args"]
    assert args[args.index("--session-key") + 1] == SESSION_KEY


@pytest.mark.parametrize("token", ["", "   "])
def test_a_blank_panel_token_sends_no_credential(
    sandbox: Sandbox, terminal_factory, token: str
) -> None:
    terminal = terminal_factory(202)
    sandbox.web_session(terminal.port)
    sandbox.env["OSPREY_PANEL_TOKEN"] = token

    sandbox.run()
    sandbox.wait_until_consumed()

    [request] = terminal.requests
    assert request["auth"] is None


def test_a_package_planted_in_the_working_directory_does_not_replace_the_record_entry(
    sandbox: Sandbox, tmp_path: Path
) -> None:
    """A watched agent can write to the project dir, so it must not reach this import."""
    marker = tmp_path / "planted-ran"
    planted = sandbox.project / "osprey" / "services" / "watchdog"
    planted.mkdir(parents=True)
    for package in (planted, planted.parent, planted.parent.parent):
        (package / "__init__.py").write_text("")
    (planted / "__main__.py").write_text(f"open({str(marker)!r}, 'w').close()\n")

    sandbox.run()

    [record] = sandbox.wait_for_records(1)
    sandbox.wait_until_consumed()
    assert record["args"][0] == "record"
    assert not marker.exists()


def test_the_record_entry_does_not_pass_a_path_setting_to_its_own_children(
    sandbox: Sandbox,
) -> None:
    """Hooks under a child agent import their peers from the script's directory."""
    sandbox.run()

    [record] = sandbox.wait_for_records(1)
    assert record["safe_path_env"] is None


@pytest.mark.parametrize("name", ["http_proxy", "HTTP_PROXY"])
def test_the_post_never_goes_through_a_configured_proxy(
    sandbox: Sandbox, terminal_factory, name: str
) -> None:
    """The prompt and the panel token stay on loopback whatever the host's proxy is."""
    proxy = terminal_factory(200)
    terminal = terminal_factory(202)
    sandbox.web_session(terminal.port)
    sandbox.env[name] = f"http://127.0.0.1:{proxy.port}"

    sandbox.run()
    sandbox.wait_until_consumed()

    assert proxy.requests == []
    assert len(terminal.requests) == 1
    assert list(sandbox.records.iterdir()) == []


def test_the_file_is_gone_when_the_record_entry_cannot_run(sandbox: Sandbox) -> None:
    (Path(sandbox.env["PYTHONPATH"]) / "osprey/services/watchdog/__main__.py").write_text(
        "raise SystemExit(3)\n"
    )

    sandbox.run()

    sandbox.wait_until_consumed()


def test_the_same_prompt_twice_is_two_submissions(sandbox: Sandbox) -> None:
    sandbox.run("same words")
    sandbox.run("same words")

    records = sandbox.wait_for_records(2)
    ids = {record["submission"]["submission_id"] for record in records}
    assert len(ids) == 2


def test_the_hook_returns_while_a_slow_child_is_still_running(sandbox: Sandbox) -> None:
    (Path(sandbox.env["PYTHONPATH"]) / "osprey/services/watchdog/__main__.py").write_text(
        "import time\ntime.sleep(5)\n"
    )

    sandbox.run()

    assert len(sandbox.submission_files()) == 1


def test_the_hook_is_stdlib_only_and_3_9_safe() -> None:
    tree = ast.parse(HOOK.read_text(), feature_version=(3, 9))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported - set(sys.stdlib_module_names) == set()


def test_the_parent_path_imports_only_what_a_prompt_can_afford() -> None:
    top_level = {
        alias.name
        for node in ast.parse(HOOK.read_text()).body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert top_level == {"json", "os", "sys"}
