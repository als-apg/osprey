"""The host launchers hold the agent record: ``osprey web``, ``chat`` and ``query``.

Pinned here: each launcher, on the host, holds an ``ensure_host_record`` lease
(with its supervisor) for its lifetime and hands its children the record's
loopback URL, the host login and ``OSPREY_SPOOL_DIR``; the lease ends when the
launcher returns, and ``osprey web stop`` ends the detached server's lease and
leaves no record process behind; a ClickHouse project with no stack refuses
to start, naming the fix; with the URL and a terminal login already set (a
terminal container), ``osprey web`` takes no lease, spawns nothing and does
not refuse; and the supervisor starts the record again after it is stopped
mid-session.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import yaml
from click.testing import CliRunner

from osprey.agent_record import credentials as creds
from osprey.agent_record import host as hr
from osprey.agent_record.client import URL_ENV
from osprey.agent_record.contract import SPOOL_DIR_ENV
from osprey.agent_runner import launcher, provider_env
from osprey.cli.chat_cmd import chat
from osprey.cli.query_cmd import EXIT_USAGE, query
from osprey.cli.web_cmd import PID_FILE, web
from osprey_connectors_tests._harness._ports import reserved_port
from tests.cli._lifecycle_build import stub_build
from tests.cli._scoped_subprocess import patch_subprocess

# Imported at collection, before ``tests/cli/conftest.py`` stubs them per test.
REAL_HOST_RECORD_LAUNCH = provider_env.host_record_launch
REAL_HOST_RECORD_REFUSAL = hr.host_record_refusal

HOST = "builder"


#: The port this test's record would bind, reserved for the test by
#: :func:`_record_port`: the config names it before any record starts.
_RECORD_PORT: list[int] = []


@pytest.fixture(autouse=True)
def _record_port() -> Iterator[None]:
    with reserved_port() as port:
        _RECORD_PORT[:] = [port]
        yield


def _config(repo: Path, *, backend: str = "embedded", port: int | None = None) -> dict:
    return {
        "project_name": "hostlaunch",
        "project_root": str(repo),
        "build_dir": "./build",
        "deployment": {"host_identity": HOST},
        "agent_record": {"port": port or _RECORD_PORT[0], "backend": backend},
    }


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _restore_process_state(tmp_path, monkeypatch):
    """Undo what an in-process launch does to cwd and ``os.environ``."""
    claude = tmp_path / "claude"
    claude.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude))
    monkeypatch.delenv(URL_ENV, raising=False)
    monkeypatch.delenv(creds.LOGIN_ENV, raising=False)
    monkeypatch.delenv(SPOOL_DIR_ENV, raising=False)
    cwd = Path.cwd()
    environ = dict(os.environ)
    yield
    os.chdir(cwd)
    os.environ.clear()
    os.environ.update(environ)


@pytest.fixture(autouse=True)
def _launch_stubs(monkeypatch):
    """Pin what the launchers read from the machine, so only the record varies."""
    monkeypatch.setattr(
        "osprey.agent_runner.provider_env.read_managed_policy_env", lambda paths=None: {}
    )
    monkeypatch.setattr(launcher, "_locate_claude", lambda: "/usr/local/bin/claude")
    monkeypatch.setattr(launcher, "_probe_cli_version", lambda executable: "2.1.300")
    monkeypatch.setattr("osprey.mcp_env.load_dotenv_from_project", lambda **_kw: None)
    monkeypatch.setattr("osprey.cli.chat_cmd._launch_companion_servers", lambda project_dir: [])


@pytest.fixture
def real_record(monkeypatch):
    """Put back the record functions the conftest guard replaces."""
    monkeypatch.setattr(provider_env, "host_record_launch", REAL_HOST_RECORD_LAUNCH)
    monkeypatch.setattr(hr, "host_record_refusal", REAL_HOST_RECORD_REFUSAL)


def _deploy(repo: Path, config: dict, *, credentials: bool = True) -> Path:
    stub_build(repo, config=yaml.safe_dump(config))
    (repo / "build" / ".mcp.json").write_text('{"mcpServers": {}}')
    if credentials:
        creds.ensure_agent_record_credentials(repo, config, reader_keys=False)
    return repo


@dataclass
class FakeEnsure:
    """Stands in for ``ensure_host_record``: records each lease and its end."""

    record: hr.HostRecord = field(
        default_factory=lambda: hr.HostRecord("http://127.0.0.1:4242", "host-builder", "tok")
    )
    calls: list[dict] = field(default_factory=list)
    open: int = 0
    ended: int = 0
    seen_inside: list[dict] = field(default_factory=list)

    @contextlib.contextmanager
    def __call__(self, repo_root, _config, *, config_path=None, supervise_s=None, **_kw):
        self.calls.append(
            {"repo_root": Path(repo_root), "config_path": config_path, "supervise_s": supervise_s}
        )
        self.open += 1
        try:
            yield self.record
        finally:
            self.open -= 1
            self.ended += 1


@pytest.fixture
def fake_ensure(monkeypatch, request) -> FakeEnsure:
    request.getfixturevalue("real_record")
    fake = FakeEnsure()
    monkeypatch.setattr(hr, "ensure_host_record", fake)
    return fake


def _expected_env(repo: Path, record: hr.HostRecord) -> dict[str, str]:
    return {
        URL_ENV: record.url,
        creds.LOGIN_ENV: f"{record.login}:{record.token}",
        SPOOL_DIR_ENV: str(repo / "var" / "spool" / HOST),
    }


def _record_env(environ) -> dict[str, str | None]:
    return {key: environ.get(key) for key in (URL_ENV, creds.LOGIN_ENV, SPOOL_DIR_ENV)}


# ---------------------------------------------------------------------------
# Each launcher holds a lease and passes the env
# ---------------------------------------------------------------------------


def test_web_holds_a_supervised_lease_and_passes_the_env(
    runner, lifecycle_repo, fake_ensure, monkeypatch
):
    repo = _deploy(lifecycle_repo, _config(lifecycle_repo), credentials=False)
    seen: dict = {}

    def _run_web(**_kw):
        seen["env"] = _record_env(os.environ)
        seen["open"] = fake_ensure.open

    monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", _run_web)
    monkeypatch.chdir(repo)

    result = runner.invoke(web, ["--shell", "true", "--skip-preflight"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert seen["env"] == _expected_env(repo, fake_ensure.record)
    assert seen["open"] == 1
    assert fake_ensure.ended == 1  # the lease ended when the server returned
    assert fake_ensure.calls[0]["supervise_s"] == hr.SUPERVISE_INTERVAL_S
    assert Path(fake_ensure.calls[0]["config_path"]) == repo / "build" / "config.yml"
    assert _record_env(os.environ) == {URL_ENV: None, creds.LOGIN_ENV: None, SPOOL_DIR_ENV: None}
    assert (repo / "var" / "spool" / HOST).is_dir()


def test_web_ends_the_lease_when_the_server_raises(
    runner, lifecycle_repo, fake_ensure, monkeypatch
):
    repo = _deploy(lifecycle_repo, _config(lifecycle_repo), credentials=False)

    def _run_web(**_kw):
        raise RuntimeError("server crashed")

    monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", _run_web)
    monkeypatch.chdir(repo)

    result = runner.invoke(web, ["--shell", "true", "--skip-preflight"])

    assert isinstance(result.exception, RuntimeError)
    assert fake_ensure.ended == 1


def test_web_ends_the_lease_when_stopped_by_sigterm(
    runner, lifecycle_repo, fake_ensure, monkeypatch
):
    """uvicorn re-raises the SIGTERM it shut down on; the lease still ends."""
    repo = _deploy(lifecycle_repo, _config(lifecycle_repo), credentials=False)
    before = signal.getsignal(signal.SIGTERM)

    def _run_web(**_kw):
        signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", _run_web)
    monkeypatch.chdir(repo)

    result = runner.invoke(web, ["--shell", "true", "--skip-preflight"])

    assert result.exit_code == 0, result.output
    assert fake_ensure.ended == 1
    assert signal.getsignal(signal.SIGTERM) is before


def test_chat_holds_a_supervised_lease_and_passes_the_env(runner, lifecycle_repo, fake_ensure):
    repo = _deploy(lifecycle_repo, _config(lifecycle_repo), credentials=False)
    launched: list[dict] = []

    def fake_run(_argv, *_args, env=None, **_kwargs):
        launched.append({"env": dict(env or {}), "open": fake_ensure.open})
        return SimpleNamespace(returncode=3)

    with patch_subprocess("osprey.cli.chat_cmd", side_effect=fake_run):
        result = runner.invoke(chat, ["--repo", str(repo)])

    assert result.exit_code == 3, result.output
    assert len(launched) == 1
    assert _record_env(launched[0]["env"]) == _expected_env(repo, fake_ensure.record)
    assert launched[0]["open"] == 1
    assert fake_ensure.ended == 1
    assert fake_ensure.calls[0]["supervise_s"] == hr.SUPERVISE_INTERVAL_S


def test_query_holds_a_supervised_lease_and_passes_the_env(runner, lifecycle_repo, fake_ensure):
    repo = _deploy(lifecycle_repo, _config(lifecycle_repo), credentials=False)
    seen: dict = {}

    async def _run_query(*_args, **_kw):
        seen["env"] = _record_env(os.environ)
        seen["open"] = fake_ensure.open
        return MagicMock(text_blocks=["ok"], is_error=False, mcp_servers=[], tool_traces=[])

    with (
        patch("osprey.cli.query_cmd.run_query", new=AsyncMock(side_effect=_run_query)),
        patch("osprey.cli.query_cmd.read_only_disallowed_tools", return_value=[]),
        patch("osprey.cli.query_cmd.expected_mcp_servers", return_value=set()),
        patch("osprey.cli.query_cmd.evaluate_verdict", return_value=0),
    ):
        result = runner.invoke(query, ["--repo", str(repo), "hello"])

    assert result.exit_code == 0, result.output
    assert seen["env"] == _expected_env(repo, fake_ensure.record)
    assert seen["open"] == 1
    assert fake_ensure.ended == 1
    assert fake_ensure.calls[0]["supervise_s"] == hr.SUPERVISE_INTERVAL_S


# ---------------------------------------------------------------------------
# ClickHouse with no stack refuses, naming the fix
# ---------------------------------------------------------------------------


def _clickhouse(lifecycle_repo: Path) -> Path:
    return _deploy(lifecycle_repo, _config(lifecycle_repo, backend="clickhouse"))


def _assert_named_fix(output: str) -> None:
    assert "osprey up" in output
    assert "agent_record.backend: embedded" in output


@pytest.mark.usefixtures("real_record")
def test_web_refuses_a_clickhouse_project_with_no_stack(runner, lifecycle_repo, monkeypatch):
    repo = _clickhouse(lifecycle_repo)
    run_web = MagicMock()
    monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", run_web)
    monkeypatch.chdir(repo)

    result = runner.invoke(web, ["--shell", "true", "--skip-preflight"])

    assert result.exit_code == 1
    _assert_named_fix(result.output)
    run_web.assert_not_called()


@patch("osprey.cli.web_cmd._wait_for_server", return_value=True)
@patch_subprocess("osprey.cli.web_cmd", popen=True)
@pytest.mark.usefixtures("real_record")
def test_web_detach_refuses_a_clickhouse_project_in_the_foreground(
    fake_subprocess, _wait, runner, lifecycle_repo, monkeypatch
):
    repo = _clickhouse(lifecycle_repo)
    monkeypatch.chdir(repo)

    result = runner.invoke(web, ["--shell", "true", "--skip-preflight", "--detach"])

    assert result.exit_code == 1
    _assert_named_fix(result.output)
    fake_subprocess.Popen.assert_not_called()


@pytest.mark.usefixtures("real_record")
def test_chat_refuses_a_clickhouse_project_with_no_stack(runner, lifecycle_repo):
    repo = _clickhouse(lifecycle_repo)
    launched = []
    with patch_subprocess("osprey.cli.chat_cmd", side_effect=lambda *a, **k: launched.append(a)):
        result = runner.invoke(chat, ["--repo", str(repo)])

    assert result.exit_code == 1
    _assert_named_fix(result.output)
    assert launched == []


@pytest.mark.usefixtures("real_record")
def test_query_refuses_a_clickhouse_project_with_no_stack(runner, lifecycle_repo):
    repo = _clickhouse(lifecycle_repo)
    run = AsyncMock()
    with (
        patch("osprey.cli.query_cmd.run_query", new=run),
        patch("osprey.cli.query_cmd.read_only_disallowed_tools", return_value=[]),
        patch("osprey.cli.query_cmd.expected_mcp_servers", return_value=set()),
    ):
        result = runner.invoke(query, ["--repo", str(repo), "hello"])

    assert result.exit_code == EXIT_USAGE
    _assert_named_fix(result.output)
    run.assert_not_called()


# ---------------------------------------------------------------------------
# Inside a terminal container: the URL is used as it is
# ---------------------------------------------------------------------------


PRESET = {URL_ENV: "http://agent-record:8000", creds.LOGIN_ENV: "terminal-alice:secret"}


@pytest.mark.usefixtures("real_record")
def test_a_preset_url_takes_no_lease_spawns_nothing_and_never_refuses(
    runner, lifecycle_repo, monkeypatch
):
    # ClickHouse, nothing answering, and not even a host login filed: on the
    # host each of these would refuse.
    repo = _deploy(lifecycle_repo, _config(lifecycle_repo, backend="clickhouse"), credentials=False)
    for key, value in PRESET.items():
        monkeypatch.setenv(key, value)

    def _never(*_a, **_kw):
        raise AssertionError("no lease or process may be taken in a terminal container")

    monkeypatch.setattr(hr, "ensure_host_record", _never)
    monkeypatch.setattr(hr, "_spawn", _never)
    seen: dict = {}
    monkeypatch.setattr(
        "osprey.interfaces.web_terminal.run_web",
        lambda **_kw: seen.setdefault("env", _record_env(os.environ)),
    )
    monkeypatch.chdir(repo)

    result = runner.invoke(web, ["--shell", "true", "--skip-preflight"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert seen["env"] == {**PRESET, SPOOL_DIR_ENV: None}
    assert not (repo / "var" / "state").exists()


@patch("osprey.cli.web_cmd._wait_for_server", return_value=True)
@patch_subprocess("osprey.cli.web_cmd", popen=True)
@pytest.mark.usefixtures("real_record")
def test_a_preset_url_never_refuses_a_detached_launch(
    fake_subprocess, _wait, runner, lifecycle_repo, monkeypatch
):
    repo = _deploy(lifecycle_repo, _config(lifecycle_repo, backend="clickhouse"), credentials=False)
    for key, value in PRESET.items():
        monkeypatch.setenv(key, value)
    fake_subprocess.Popen.return_value = MagicMock(pid=4321)
    monkeypatch.chdir(repo)

    result = runner.invoke(web, ["--shell", "true", "--skip-preflight", "--detach"])

    assert result.exit_code == 0, result.output
    fake_subprocess.Popen.assert_called_once()


# ---------------------------------------------------------------------------
# `osprey web stop` ends the lease
# ---------------------------------------------------------------------------

_HOLDER = """
import signal, sys, time
from pathlib import Path
from osprey.agent_record import host as hr
lease = hr._write_lease(Path(sys.argv[1]))
def _bye(*_):
    lease.unlink()
    sys.exit(0)
signal.signal(signal.SIGTERM, _bye)
print("ready", flush=True)
while True:
    time.sleep(0.05)
"""


def _wait_dead(proc: subprocess.Popen, wait_s: float = 15.0) -> bool:
    try:
        proc.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        return False
    return True


@contextlib.contextmanager
def _record_and_server(repo: Path, config: dict):
    """A stand-in record process (pid file) and a web server holding a lease on it."""
    state_dir = hr.host_state_dir(repo, config)
    state_dir.mkdir(parents=True, exist_ok=True)
    record = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True
    )
    server = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(state_dir / hr.LEASE_DIR_NAME)],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        start = None
        for _ in range(100):
            start = hr._process_start(record.pid)
            if start:
                break
            time.sleep(0.02)
        assert start
        hr._write_identity(state_dir / hr.PROCESS_FILE_NAME, record.pid, start)
        assert server.stdout is not None
        assert server.stdout.readline().strip() == "ready"
        path = repo / PID_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(server.pid))
        yield state_dir, record, server
    finally:
        for proc in (server, record):
            if proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def test_web_stop_ends_the_lease_and_leaves_no_record(runner, lifecycle_repo, monkeypatch):
    config = _config(lifecycle_repo)
    repo = _deploy(lifecycle_repo, config, credentials=False)
    monkeypatch.chdir(repo)

    with _record_and_server(repo, config) as (state_dir, record, server):
        result = runner.invoke(web, ["stop"], catch_exceptions=False)

        assert result.exit_code == 0, result.output
        assert _wait_dead(server)
        assert _wait_dead(record)
        assert hr.live_leases(state_dir / hr.LEASE_DIR_NAME) == []
        assert not (state_dir / hr.PROCESS_FILE_NAME).exists()
        assert not (repo / PID_FILE).exists()


def test_web_stop_leaves_a_record_another_session_still_holds(runner, lifecycle_repo, monkeypatch):
    config = _config(lifecycle_repo)
    repo = _deploy(lifecycle_repo, config, credentials=False)
    monkeypatch.chdir(repo)

    with _record_and_server(repo, config) as (state_dir, record, server):
        mine = hr._write_lease(state_dir / hr.LEASE_DIR_NAME)  # an `osprey chat` still open
        try:
            result = runner.invoke(web, ["stop"], catch_exceptions=False)

            assert result.exit_code == 0, result.output
            assert _wait_dead(server)
            assert record.poll() is None
        finally:
            mine.unlink()


# ---------------------------------------------------------------------------
# The supervisor
# ---------------------------------------------------------------------------


def _supervisor(tmp_path: Path, *, embedded: bool = True) -> tuple[hr._Supervisor, Path]:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    lease = state_dir / hr.LEASE_DIR_NAME / "mine.lease"
    sup = hr._Supervisor(
        tmp_path,
        {},
        state_dir,
        hr.HostRecord("http://127.0.0.1:1", "host-builder", "tok"),
        1,
        embedded,
        lease,
        tmp_path / "config.yml",
        60.0,
    )
    return sup, lease


def test_the_supervisor_never_spawns_under_a_live_marker(tmp_path, monkeypatch):
    sup, _ = _supervisor(tmp_path)
    monkeypatch.setattr(hr, "_probe", lambda *a: (hr._DOWN, "nothing answers"))
    monkeypatch.setattr(hr, "marker_is_live", lambda state_dir: True)
    monkeypatch.setattr(hr, "_spawn", MagicMock(side_effect=AssertionError("spawned")))

    assert sup.tick() is False


def test_the_supervisor_never_spawns_a_clickhouse_record(tmp_path, monkeypatch):
    sup, _ = _supervisor(tmp_path, embedded=False)
    monkeypatch.setattr(hr, "_probe", lambda *a: (hr._DOWN, "nothing answers"))
    monkeypatch.setattr(hr, "marker_is_live", lambda state_dir: False)
    monkeypatch.setattr(hr, "_spawn", MagicMock(side_effect=AssertionError("spawned")))

    assert sup.tick() is False


def test_the_supervisor_leaves_an_answering_record_alone(tmp_path, monkeypatch):
    sup, _ = _supervisor(tmp_path)
    monkeypatch.setattr(hr, "_probe", lambda *a: (hr._READY, ""))
    monkeypatch.setattr(hr, "_spawn", MagicMock(side_effect=AssertionError("spawned")))

    assert sup.tick() is False


def test_the_supervisor_restarts_under_the_holders_own_lease(tmp_path, monkeypatch):
    sup, lease = _supervisor(tmp_path)
    monkeypatch.setattr(hr, "_probe", lambda *a: (hr._DOWN, "nothing answers"))
    monkeypatch.setattr(hr, "marker_is_live", lambda state_dir: False)
    spawn = MagicMock()
    monkeypatch.setattr(hr, "_spawn", spawn)

    assert sup.tick() is True
    spawn.assert_called_once()
    # The lease something removed is written again, naming this process.
    assert [held.path for held in hr.live_leases(lease.parent)] == [lease]


def test_the_supervisor_restarts_the_record_stopped_mid_session(tmp_path):
    """A real ``osprey agent-record serve``: stopped under the lock, back within seconds."""
    repo = tmp_path / "repo"
    (repo / "build").mkdir(parents=True)
    config = _config(repo)
    (repo / "build" / "config.yml").write_text(yaml.safe_dump(config))
    creds.ensure_agent_record_credentials(repo, config, reader_keys=False)
    state_dir = hr.host_state_dir(repo, config)
    try:
        with hr.ensure_host_record(repo, config, supervise_s=0.3) as record:
            first = hr._running_process(state_dir)
            assert first is not None
            with hr.host_record_lock(state_dir):
                assert hr.stop_host_process(state_dir, wait_s=10)

            deadline = time.monotonic() + 90
            again = None
            while time.monotonic() < deadline:
                again = hr._running_process(state_dir)
                if again is not None and again != first:
                    if hr._probe(record.url, record.login, record.token)[0] == hr._READY:
                        break
                time.sleep(0.2)
            assert again is not None and again != first
            health = httpx.get(
                f"{record.url}/health/sources",
                auth=(record.login, record.token),
                trust_env=False,
            )
            assert health.status_code == 200
        # The holder's release stops the restarted record too.
        assert hr._running_process(state_dir) is None
    finally:
        if state_dir.exists():
            hr.stop_host_process(state_dir, wait_s=10)


# ---------------------------------------------------------------------------
# A host launch records under the identity the build recorded
# ---------------------------------------------------------------------------

AUDIT_IDENTITY = "OSPREY_AUDIT_IDENTITY"


def _rendered(repo: Path, config: dict | str) -> Path:
    path = repo / "build" / "config.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config if isinstance(config, str) else yaml.safe_dump(config))
    return repo


def test_export_built_host_identity_exports_the_identity_alone(tmp_path):
    repo = _rendered(tmp_path, _config(tmp_path))
    environ: dict[str, str] = {}
    assert creds.export_built_host_identity(environ, repo) == [AUDIT_IDENTITY]
    # The identity alone: no sender login, no address.
    assert environ == {AUDIT_IDENTITY: HOST}


def test_export_built_host_identity_leaves_a_terminal_container_alone(tmp_path):
    repo = _rendered(tmp_path, _config(tmp_path))
    environ = {"OSPREY_TERMINAL_USER": "alice"}
    assert creds.export_built_host_identity(environ, repo) == []
    assert environ == {"OSPREY_TERMINAL_USER": "alice"}


@pytest.mark.parametrize(
    "render",
    [None, "{not: [yaml", "- a list\n", {"project_name": "no-identity"}],
    ids=["no-render", "unreadable", "not-a-mapping", "no-host-identity"],
)
def test_export_built_host_identity_exports_nothing_without_one(tmp_path, render):
    repo = tmp_path if render is None else _rendered(tmp_path, render)
    environ: dict[str, str] = {}
    assert creds.export_built_host_identity(environ, repo) == []
    assert environ == {}


@pytest.mark.usefixtures("fake_ensure")
def test_chat_launches_under_the_built_host_identity(runner, lifecycle_repo, monkeypatch):
    monkeypatch.delenv(AUDIT_IDENTITY, raising=False)
    monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
    repo = _deploy(lifecycle_repo, _config(lifecycle_repo), credentials=False)
    launched: list[dict] = []

    def fake_run(_argv, *_args, env=None, **_kwargs):
        launched.append(dict(env or {}))
        return SimpleNamespace(returncode=0)

    with patch_subprocess("osprey.cli.chat_cmd", side_effect=fake_run):
        result = runner.invoke(chat, ["--repo", str(repo)])

    assert result.exit_code == 0, result.output
    assert launched[0][AUDIT_IDENTITY] == HOST
    assert "OSPREY_TELEMETRY_LOGIN" not in launched[0]


@pytest.mark.parametrize(
    ("module", "anchor"),
    [
        ("chat_cmd", "_overlay_repo_env(repo_root)\n"),
        ("query_cmd", "_overlay_repo_env(repo_root)\n"),
        ("web_cmd", "load_dotenv_from_project(include_sensitive=True)\n"),
    ],
)
def test_every_host_launcher_exports_the_identity_after_the_env_chain(
    module: str, anchor: str
) -> None:
    """Each launcher makes the one call, after the env chain is loaded.

    After, because the chain is loaded with override: an identity exported
    first could be replaced by a stale line in ``.env``.
    """
    import importlib
    import inspect

    source = inspect.getsource(importlib.import_module(f"osprey.cli.{module}"))
    call = "export_built_host_identity(os.environ, repo_root)"

    assert source.count(call) == 1
    assert source.rindex(anchor) < source.index(call)
    assert "from osprey.agent_record.credentials import export_built_host_identity" in source


def test_web_resolves_its_render_with_the_identity_exported(tmp_path, monkeypatch) -> None:
    from osprey.cli import web_cmd

    (tmp_path / "profile.yml").write_text("project_name: p\n")
    _rendered(tmp_path, _config(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(AUDIT_IDENTITY, "")
    monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
    monkeypatch.delenv("OSPREY_TELEMETRY_LOGIN", raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", "")

    web_cmd._resolve_render(tmp_path)

    assert os.environ[AUDIT_IDENTITY] == HOST
    assert "OSPREY_TELEMETRY_LOGIN" not in os.environ
