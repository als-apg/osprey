"""Lifecycle tests for agent-record, the service every deployment runs.

- The project image is built for it even when the profile deploys no other
  service that runs that image, and ``_start_stack`` starts a stack with no
  ``deployed_services`` and no web terminals.
- Removing a user's volumes first removes the agent-record container, which
  mounts every user's transcripts volume.
- The image entrypoint joins the spool's and the record dir's groups and never
  chowns either.
- ``osprey up`` takes the record over from the host-local agent-record under
  the ensure lock (marker written, host process stopped, compose started, the
  stack's record waited for); an ensure arriving meanwhile waits and uses the
  stack, never spawning; after ``osprey down`` an ensure spawns again; a record
  file not group-writable by its directory's group is refused before compose.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.agent_record import credentials as creds
from osprey.agent_record import host as hr
from osprey.agent_runner.project_paths import CLAUDE_CONFIG_DIR_ENV
from osprey.deployment import container_lifecycle
from osprey.deployment.compose_generator import (
    AGENT_RECORD_TEMPLATE_DIR,
    resolve_project_name,
    resolve_user_projects_volume_name,
    resolve_user_volume_names,
)
from osprey.deployment.deployment_record import DeploymentRecord, write_deployment_record
from osprey.deployment.web_terminals import lifecycle
from tests.deployment.test_collect_all_preflight import stubbed_start  # noqa: F401
from tests.deployment.web_terminals.test_lifecycle import _config as _web_config
from tests.deployment.web_terminals.test_lifecycle import _write_config

ENTRYPOINT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "osprey"
    / "templates"
    / "project"
    / "entrypoint.sh"
)
SPOOL_VAR = "OSPREY_AGENT_RECORD_SPOOL_ROOT"
RECORD_VAR = "OSPREY_AGENT_RECORD_DIR"


# ── The project image and the start ─────────────────────────────────────────


def test_a_profile_with_no_project_image_service_builds_the_image_for_agent_record() -> None:
    config: dict[str, Any] = {"project_name": "hello", "deployed_services": []}

    assert AGENT_RECORD_TEMPLATE_DIR in container_lifecycle._deployed_project_image_services(config)
    assert container_lifecycle._project_image_build_target(config, {}) == "hello:local"


def test_an_image_override_takes_the_agent_record_build_away() -> None:
    config: dict[str, Any] = {"project_name": "hello", "deployed_services": []}

    env = {"OSPREY_WORKER_IMAGE": "registry.example.org/osprey:1"}
    assert container_lifecycle._project_image_build_target(config, env) is None


def _record_compose_file(tmp_path: Path, services: dict[str, Any]) -> list[str]:
    """A rendered agent-record compose file holding *services*, as a start's compose files."""
    path = tmp_path / "build" / "services" / AGENT_RECORD_TEMPLATE_DIR / "docker-compose.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"services": services}), encoding="utf-8")
    return [str(path)]


def test_an_attached_project_up_plans_no_agent_record_image_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The render of an attached project holds no agent-record, so nothing is built for it."""
    config: dict[str, Any] = {"project_name": "hello", "deployed_services": []}

    def _no_build(_config: Any = None) -> list[str]:
        raise AssertionError("an attached project builds no project image")

    monkeypatch.setattr(container_lifecycle, "get_runtime_command", _no_build)

    services = container_lifecycle._deployed_project_image_services(config, [])
    assert AGENT_RECORD_TEMPLATE_DIR not in services
    assert container_lifecycle._project_image_build_target(config, {}, []) is None
    assert container_lifecycle._unreleased_pin_problem(config, {}, False, []) is None
    container_lifecycle._build_project_image(config, False, {}, None, [])


def test_an_up_that_is_not_attached_still_plans_the_agent_record_image(tmp_path: Path) -> None:
    config: dict[str, Any] = {"project_name": "hello", "deployed_services": []}
    compose_files = _record_compose_file(tmp_path, {creds.RECORD_SERVICE: {"image": "hello:local"}})

    services = container_lifecycle._deployed_project_image_services(config, compose_files)
    assert AGENT_RECORD_TEMPLATE_DIR in services
    assert container_lifecycle._project_image_build_target(config, {}, compose_files) == (
        "hello:local"
    )


def test_a_clickhouse_backed_up_that_is_not_attached_still_pulls_the_clickhouse_image(
    tmp_path: Path,
) -> None:
    """The record's store is a pulled image beside agent-record; neither is dropped."""
    config: dict[str, Any] = {
        "project_name": "hello",
        "deployed_services": [],
        "agent_record": {"backend": "clickhouse"},
    }
    store_image = "clickhouse/clickhouse-server:26.8.18.2"
    compose_files = _record_compose_file(
        tmp_path,
        {
            creds.RECORD_SERVICE: {"image": "hello:local"},
            creds.CLICKHOUSE_SERVICE: {"image": store_image},
        },
    )

    assert container_lifecycle._project_image_build_target(config, {}, compose_files) == (
        "hello:local"
    )
    rendered = container_lifecycle.rendered_compose_services(Path(compose_files[0]))
    assert rendered is not None
    store = rendered[creds.CLICKHOUSE_SERVICE]
    assert store["image"] == store_image
    assert "build" not in store, "compose pulls the store's image; nothing builds it"


@pytest.mark.usefixtures("stubbed_start")
@pytest.mark.parametrize("attached", [True, False], ids=["attached-project", "own-stack"])
def test_the_start_plans_the_project_image_from_the_compose_files_it_starts(
    attached: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The build and the compose call read one service set: the start's compose files."""
    config: dict[str, Any] = {"project_name": "hello", "deployed_services": []}
    compose_files = [] if attached else _record_compose_file(tmp_path, {creds.RECORD_SERVICE: {}})
    planned: list[Any] = []

    def _after_build(*_args: Any, **_kwargs: Any) -> None:
        raise _Reached

    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)
    monkeypatch.setattr(
        container_lifecycle,
        "_build_project_image",
        lambda *a, compose_files=None, **k: planned.append(compose_files),
    )
    monkeypatch.setattr(container_lifecycle, "_stage_agent_record_clickhouse", _after_build)

    with pytest.raises(_Reached):
        container_lifecycle._start_stack(config, compose_files, tmp_path, detached=True)

    assert planned == [compose_files]
    assert container_lifecycle._project_image_build_target(config, {}, planned[0]) == (
        None if attached else "hello:local"
    )


class _Reached(Exception):
    """Raised by the first step after the removed early return."""


def test_start_stack_starts_a_stack_with_no_services_and_no_terminals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _reached(_config: dict) -> None:
        raise _Reached

    monkeypatch.setattr(container_lifecycle, "_check_shared_disk_preflight", _reached)
    config: dict[str, Any] = {"project_name": "hello", "deployed_services": []}

    with pytest.raises(_Reached):
        container_lifecycle._start_stack(config, [], tmp_path)


# ── Removing a user's volumes ───────────────────────────────────────────────


@pytest.fixture
def runtime_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def _run(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lifecycle.subprocess, "run", _run)
    monkeypatch.setattr(lifecycle, "get_runtime_command", lambda config=None: ["docker", "compose"])
    monkeypatch.setattr(lifecycle, "verify_runtime_is_running", lambda config=None, **_: (True, ""))
    return calls


def test_purging_a_user_removes_agent_record_before_the_transcripts_volume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_calls: list[list[str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _web_config(["alice", "bob"])
    config_path = _write_config(tmp_path, config)
    claude_vol, agent_vol = resolve_user_volume_names(config, "alice")
    projects_vol = resolve_user_projects_volume_name(config, "alice")

    lifecycle.decommission_user(str(config_path), "alice", purge=True, assume_yes=True)

    removals = [
        c for c in runtime_calls if "demo-project-agent-record" in c or c[1:3] == ["volume", "rm"]
    ]
    assert removals == [
        ["docker", "rm", "-f", "demo-project-agent-record"],
        ["docker", "volume", "rm", claude_vol],
        ["docker", "volume", "rm", projects_vol],
        ["docker", "volume", "rm", agent_vol],
    ]
    out = " ".join(capsys.readouterr().out.split())
    assert "agent-record container was removed" in out
    assert "Run `osprey build`, then `osprey up`" in out


def test_retaining_a_users_volumes_leaves_agent_record_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime_calls: list[list[str]]
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(tmp_path, _web_config(["alice", "bob"]))

    lifecycle.decommission_user(str(config_path), "alice", assume_yes=True)

    assert not [c for c in runtime_calls if "demo-project-agent-record" in c]


def test_a_missing_agent_record_container_is_not_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def _run(argv, **_kwargs):
        missing = argv[1] == "rm"
        return subprocess.CompletedProcess(argv, returncode=1 if missing else 0, stdout="")

    monkeypatch.setattr(lifecycle.subprocess, "run", _run)

    lifecycle._apply_volume_policy(
        "docker",
        ["demo-project_alice-claude-projects"],
        archive=False,
        purge=True,
        repo_root=tmp_path,
        record_container="demo-project-agent-record",
    )

    assert "agent-record" not in capsys.readouterr().out


# ── The entrypoint ──────────────────────────────────────────────────────────


def _function(text: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.S | re.M)
    assert match, name
    return match.group(0)


def test_the_entrypoint_joins_both_directories_groups() -> None:
    body = _function(ENTRYPOINT.read_text(encoding="utf-8"), "join_mounted_groups")

    for var in (SPOOL_VAR, RECORD_VAR):
        assert f'join_mounted_group {var} "${{{var}:-}}"' in body
        assert f'probe_mounted_dir {var} "${{{var}:-}}" write' in body


def test_the_hand_back_prunes_the_spool_root_and_the_record_dir(tmp_path: Path) -> None:
    """Run the real function with ``find`` stubbed: both binds are pruned, so
    root never chowns the group-shared spool tree or the record dir."""
    text = ENTRYPOINT.read_text(encoding="utf-8")
    state = tmp_path / "var"
    spool = state / "spool"
    record = state / "agent_record"
    for directory in (spool / ".otlp", record):
        directory.mkdir(parents=True)
    args_file = tmp_path / "find-args"
    script = "\n".join(
        [
            f'find() {{ printf "%s\\n" "$*" > "{args_file}"; }}',
            "log() { :; }",
            _function(text, "hand_back_state_zone"),
            "hand_back_state_zone",
        ]
    )

    subprocess.run(
        ["sh", "-c", script],
        check=True,
        env={
            "PATH": "/usr/bin:/bin",
            "STATE_DIR": str(state),
            SPOOL_VAR: str(spool),
            RECORD_VAR: str(record),
        },
    )

    args = args_file.read_text(encoding="utf-8")
    assert f"-path {spool} -prune -o" in args, args
    assert f"-path {record} -prune -o" in args, args
    assert "! -user osprey -exec chown osprey:osprey" in args, args


def test_the_hand_back_comment_names_both_exceptions() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    comment = text[: text.index("hand_back_state_zone() {")]

    for var in (SPOOL_VAR, RECORD_VAR):
        assert f"#   * {var}" in comment, var


# ── The stack takes the record over from the host ───────────────────────────


class _Alive:
    """A spawned host-local child that keeps running."""

    pid = 999_999
    returncode = None

    def poll(self) -> None:
        return None


class _Stack:
    """The stack's agent-record as the host sees it: answering once compose started it."""

    def __init__(self) -> None:
        self.up = threading.Event()
        self.spawned: list[int] = []

    def probe(self, *_args: Any) -> tuple[str, str]:
        if self.up.is_set() or self.spawned:
            return hr._READY, ""
        return hr._DOWN, "nothing answers"


@pytest.fixture
def host_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A host-sessions deployment repo with its host login minted."""
    monkeypatch.setenv(CLAUDE_CONFIG_DIR_ENV, str(tmp_path / "claude-home"))
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    config: dict[str, Any] = {
        "project_name": "hello",
        "project_root": str(root),
        "build_dir": "./build",
        "deployed_services": [],
        "deployment": {"host_identity": "thorsten"},
        "agent_record": {"port": 1, "backend": "embedded"},
    }
    (root / "build" / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    creds.ensure_agent_record_credentials(root, config, reader_keys=False)
    return config


@pytest.fixture
def stack(monkeypatch: pytest.MonkeyPatch) -> _Stack:
    """Probes, container liveness and spawns answered by one scripted stack."""
    fake = _Stack()
    monkeypatch.setattr(hr, "_probe", fake.probe)
    monkeypatch.setattr(hr, "_PROBE_INTERVAL_S", 0.01)
    monkeypatch.setattr(hr, "_container_running", lambda name: fake.up.is_set())

    def _spawn(*_args: Any) -> _Alive:
        fake.spawned.append(1)
        return _Alive()

    monkeypatch.setattr(hr, "_spawn", _spawn)
    # Lease identity without `ps`: the start stubs out every subprocess.
    monkeypatch.setattr(hr, "_process_start", lambda pid: "now" if pid == os.getpid() else None)
    monkeypatch.setattr(container_lifecycle, "STACK_RECORD_HEALTH_WAIT_S", 2.0)
    return fake


def _root(config: dict[str, Any]) -> Path:
    return Path(config["project_root"])


def _state_dir(config: dict[str, Any]) -> Path:
    return hr.host_state_dir(_root(config), config)


def _lock_is_held(state_dir: Path) -> bool:
    if not state_dir.is_dir():
        return False
    fd = os.open(state_dir / hr.LOCK_FILE_NAME, os.O_RDWR | os.O_CREAT, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _web_terminal_config(config: dict[str, Any]) -> dict[str, Any]:
    """*config* started through the web-terminal path, whose compose ``up`` is one stub."""
    return {
        **config,
        "modules": {
            "web_terminals": {
                "enabled": True,
                "image_source": "local",
                "default_tier": "ops",
                "renders": {
                    "ops": {"kind": "tier", "project": "ops-app", "project_path": "build/ops-app"}
                },
            }
        },
    }


def _record_compose(config: dict[str, Any]) -> list[str]:
    """The stack's rendered agent-record compose file, as a start's compose files."""
    path = _root(config) / "build" / "services" / AGENT_RECORD_TEMPLATE_DIR / "docker-compose.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"services": {creds.RECORD_SERVICE: {"image": "hello:local"}}}),
        encoding="utf-8",
    )
    return [str(path)]


def _is_record_up(cmd: list[str]) -> bool:
    return "up" in cmd and cmd[-1] == creds.RECORD_SERVICE


class _Compose:
    """``run_captured`` for a start: the agent-record ``up`` starts the scripted stack."""

    def __init__(self, stack: _Stack, state_dir: Path, events: list[str]) -> None:
        self.stack, self.state_dir, self.events = stack, state_dir, events
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        if _is_record_up(cmd):
            assert _lock_is_held(self.state_dir), "agent-record starts under the ensure lock"
            assert cmd[-4:] == ["up", "-d", "--no-build", creds.RECORD_SERVICE]
            self.events.append("record up")
            self.stack.up.set()
        elif "build" in cmd:
            assert not _lock_is_held(self.state_dir), "no build runs under the ensure lock"
            self.events.append("stack build")
        elif "up" in cmd:
            assert not _lock_is_held(self.state_dir), "the rest of the start runs unlocked"
            assert self.stack.up.is_set(), "the record was started before the rest"
            self.events.append("stack up")
        return subprocess.CompletedProcess(cmd, returncode=0)


@pytest.fixture
def handover(
    host_repo: dict[str, Any], stack: _Stack, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], _Compose]:
    """Stop, record ``up`` and health wait, each logged in order of happening."""
    state_dir = _state_dir(host_repo)
    events: list[str] = []

    def _stop(directory: Path, **_kwargs: Any) -> bool:
        assert directory == state_dir
        assert _lock_is_held(state_dir), "the host process is stopped under the ensure lock"
        marker = (state_dir / hr.MARKER_FILE_NAME).read_text(encoding="utf-8")
        assert marker.splitlines()[0] == "hello-agent-record"
        events.append("stop host-local")
        return True

    wait = container_lifecycle._wait_for_stack_record

    def _wait(*args: Any) -> None:
        wait(*args)
        assert _lock_is_held(state_dir), "the health wait runs under the ensure lock"
        events.append("record healthy")

    compose = _Compose(stack, state_dir, events)
    monkeypatch.setattr(hr, "stop_host_process", _stop)
    monkeypatch.setattr(container_lifecycle, "_wait_for_stack_record", _wait)
    monkeypatch.setattr(container_lifecycle, "run_captured", compose)
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)
    return events, compose


@pytest.mark.usefixtures("stubbed_start")
def test_up_stops_the_host_local_record_before_compose_starts_agent_record(
    host_repo: dict[str, Any],
    handover: tuple[list[str], _Compose],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _web_terminal_config(host_repo)
    state_dir = _state_dir(config)
    events, _compose = handover

    def _web_up(*_args: Any, **_kwargs: Any) -> None:
        assert not _lock_is_held(state_dir), "render and service builds run unlocked"
        events.append("web up")

    monkeypatch.setattr(container_lifecycle, "deploy_up_web_terminals", _web_up)

    container_lifecycle._start_stack(config, _record_compose(config), _root(config), detached=True)

    assert events == ["stop host-local", "record up", "record healthy", "web up"]
    assert not _lock_is_held(state_dir)
    assert (state_dir / hr.MARKER_FILE_NAME).is_file()


@pytest.mark.usefixtures("stubbed_start")
def test_a_detached_up_hands_the_record_over_before_the_rest_of_the_stack(
    host_repo: dict[str, Any],
    handover: tuple[list[str], _Compose],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = host_repo
    events, compose = handover
    monkeypatch.setattr(
        container_lifecycle, "watched_compose_up", lambda *a, **k: contextlib.nullcontext({})
    )

    container_lifecycle._start_stack(config, _record_compose(config), _root(config), detached=True)

    assert events == ["stop host-local", "record up", "record healthy", "stack up"]
    assert "--remove-orphans" in compose.calls[-1] and "-d" in compose.calls[-1]
    assert not _lock_is_held(_state_dir(config))


class _Execed(Exception):
    """Raised where an attached start replaces this process with compose."""


@pytest.mark.usefixtures("stubbed_start")
def test_an_attached_up_has_the_record_running_and_answering_before_the_exec(
    host_repo: dict[str, Any],
    stack: _Stack,
    handover: tuple[list[str], _Compose],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = host_repo
    state_dir = _state_dir(config)
    events, _compose = handover
    monkeypatch.setattr(container_lifecycle, "compose_build_step_reporter", contextlib.nullcontext)

    def _execvpe(_file: str, argv: list[str], _env: Any) -> None:
        assert stack.up.is_set(), "the record container runs before compose takes over"
        assert not _lock_is_held(state_dir), "the lock is released before the exec"
        assert hr.marker_is_live(state_dir), "the marker stays live across the exec"
        events.append("exec up")
        assert "-d" not in argv
        raise _Execed

    monkeypatch.setattr(container_lifecycle.os, "execvpe", _execvpe)

    with pytest.raises(_Execed):
        container_lifecycle._start_stack(
            config, _record_compose(config), _root(config), detached=False
        )

    assert events == ["stack build", "stop host-local", "record up", "record healthy", "exec up"]


@pytest.mark.usefixtures("stubbed_start")
def test_an_ensure_during_a_slow_compose_start_waits_and_uses_the_stack(
    host_repo: dict[str, Any],
    stack: _Stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _web_terminal_config(host_repo)
    root = _root(config)
    got: list[hr.HostRecord] = []
    failed: list[BaseException] = []

    def _session() -> None:
        try:
            with hr.ensure_host_record(root, config) as record:
                got.append(record)
        except BaseException as exc:  # pragma: no cover - surfaced below
            failed.append(exc)

    session = threading.Thread(target=_session)

    def _slow_compose(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if _is_record_up(cmd):
            session.start()
            time.sleep(0.5)
            assert session.is_alive(), "the ensure waits while compose starts the stack"
            assert not got and not stack.spawned
            stack.up.set()
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(container_lifecycle, "run_captured", _slow_compose)
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)

    container_lifecycle._start_stack(config, _record_compose(config), root, detached=True)
    session.join(timeout=30)

    assert not failed, failed
    assert [record.started for record in got] == [False]
    assert got[0].url == "http://127.0.0.1:1"
    assert stack.spawned == [], "an ensure under the stack's start never spawns"


@pytest.mark.usefixtures("stubbed_start")
def test_a_stack_without_agent_record_leaves_the_host_record_alone(
    host_repo: dict[str, Any],
    handover: tuple[list[str], _Compose],
) -> None:
    config = _web_terminal_config(host_repo)
    events, compose = handover

    container_lifecycle._start_stack(config, [], _root(config), detached=True)

    assert events == [] and not any(_is_record_up(cmd) for cmd in compose.calls)
    assert not (_state_dir(config) / hr.MARKER_FILE_NAME).exists()


def test_after_down_an_ensure_spawns_again(
    host_repo: dict[str, Any], stack: _Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = host_repo
    root = _root(config)
    state_dir = _state_dir(config)
    with container_lifecycle._stack_takes_over_host_record(config, root):
        stack.up.set()
    assert (state_dir / hr.MARKER_FILE_NAME).is_file()

    downs: list[list[str]] = []

    def _run_captured(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        downs.append(list(cmd))
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(container_lifecycle, "as_built_compose_files", lambda c, r: ["x.yml"])
    monkeypatch.setattr(container_lifecycle, "run_captured", _run_captured)
    monkeypatch.setattr(
        container_lifecycle, "verify_runtime_is_running", lambda config=None, **_: (True, "")
    )
    write_deployment_record(root, DeploymentRecord("docker", resolve_project_name(config)))
    container_lifecycle.down_deployment(root)

    assert downs and downs[0][-1] == "down"
    assert not (state_dir / hr.MARKER_FILE_NAME).exists()
    # The stack's record answers no more, while a container runtime would still
    # call the container running: only the marker's removal lets the next host
    # session start its own record.
    stack.up.clear()
    monkeypatch.setattr(hr, "_container_running", lambda name: True)
    with hr.ensure_host_record(root, config) as record:
        assert record.started
    assert stack.spawned == [1]


def test_a_clean_osprey_down_removes_the_marker(
    host_repo: dict[str, Any], stack: _Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osprey.deployment import compose_generator

    config = host_repo
    root = _root(config)
    state_dir = _state_dir(config)
    with container_lifecycle._stack_takes_over_host_record(config, root):
        stack.up.set()
    monkeypatch.setattr(container_lifecycle, "load_project_config", lambda *a, **k: config)
    monkeypatch.setattr(container_lifecycle, "resolve_repo_root", lambda c, p: root)
    monkeypatch.setattr(compose_generator, "find_existing_compose_files", lambda *a, **k: ["x.yml"])
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config=None: ["docker", "compose"]
    )
    returncodes = iter([1, 0])
    monkeypatch.setattr(
        container_lifecycle,
        "run_captured",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, returncode=next(returncodes)),
    )

    with pytest.raises(SystemExit) as failed:
        container_lifecycle.deploy_down("config.yml")
    assert failed.value.code == 1
    assert (state_dir / hr.MARKER_FILE_NAME).is_file(), "a failed down keeps the marker"

    with pytest.raises(SystemExit) as done:
        container_lifecycle.deploy_down("config.yml")
    assert done.value.code == 0
    assert not (state_dir / hr.MARKER_FILE_NAME).exists()


def test_while_the_stack_owns_the_record_an_ensure_never_spawns(
    host_repo: dict[str, Any], stack: _Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The counterfactual of the test above: before the down, the marker keeps
    an ensure from spawning even while the stack's record does not answer."""
    config = host_repo
    root = _root(config)
    with container_lifecycle._stack_takes_over_host_record(config, root):
        stack.up.set()
    stack.up.clear()
    monkeypatch.setattr(hr, "_container_running", lambda name: True)

    with (
        pytest.raises(hr.HostRecordRefused, match="the stack owns the record"),
        hr.ensure_host_record(root, config, wait_s=0.2),
    ):
        pass
    assert stack.spawned == []


@pytest.mark.usefixtures("stack")
def test_a_deployment_without_host_sessions_starts_without_the_handover(tmp_path: Path) -> None:
    config = {"project_name": "hello", "project_root": str(tmp_path)}
    ran: list[str] = []

    with container_lifecycle._stack_takes_over_host_record(config, tmp_path):
        ran.append("up")

    assert ran == ["up"]
    assert not (tmp_path / "var" / "state").exists()


# ── The record store is shared through its directory's group ────────────────


def _record_dir(root: Path) -> Path:
    path = root / "var" / "agent_record"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.parametrize("name", ["record.duckdb", "record.duckdb.wal"])
def test_a_record_file_that_is_not_group_writable_is_refused_with_the_fix(
    tmp_path: Path, name: str
) -> None:
    store = _record_dir(tmp_path) / name
    store.write_bytes(b"")
    store.chmod(0o644)

    with pytest.raises(RuntimeError, match=re.escape(f"chmod g+w {store}")):
        container_lifecycle._preflight_record_store_group_writable(tmp_path)


def test_group_writable_record_files_and_no_record_pass(tmp_path: Path) -> None:
    container_lifecycle._preflight_record_store_group_writable(tmp_path)  # no record dir
    record_dir = _record_dir(tmp_path)
    container_lifecycle._preflight_record_store_group_writable(tmp_path)  # no store yet
    for name in ("record.duckdb", "record.duckdb.wal"):
        (record_dir / name).write_bytes(b"")
        (record_dir / name).chmod(0o664)

    container_lifecycle._preflight_record_store_group_writable(tmp_path)


@pytest.mark.usefixtures("stack")
def test_up_refuses_an_unwritable_record_before_compose_starts(
    host_repo: dict[str, Any],
    stubbed_start: list[str],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _web_terminal_config(host_repo)
    store = _record_dir(_root(config)) / "record.duckdb"
    store.write_bytes(b"")
    store.chmod(0o644)
    ups: list[str] = []
    monkeypatch.setattr(
        container_lifecycle, "deploy_up_web_terminals", lambda *a, **k: ups.append("up")
    )

    with pytest.raises(RuntimeError, match="chmod g\\+w"):
        container_lifecycle._start_stack(
            config, _record_compose(config), _root(config), detached=True
        )

    assert ups == [] and stubbed_start == []
    assert not (_state_dir(config) / hr.MARKER_FILE_NAME).exists()


@pytest.mark.usefixtures("stubbed_start", "handover")
@pytest.mark.parametrize("shape", ["no agent-record rendered", "clickhouse backend"])
def test_up_asks_only_a_stack_whose_agent_record_opens_the_embedded_store(
    host_repo: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    shape: str,
) -> None:
    config = _web_terminal_config(host_repo)
    store = _record_dir(_root(config)) / "record.duckdb"
    store.write_bytes(b"")
    store.chmod(0o644)
    compose_files: list[str] = []
    if shape == "clickhouse backend":
        config["agent_record"] = {"port": 1, "backend": "clickhouse"}
        compose_files = _record_compose(config)
        monkeypatch.setattr(
            container_lifecycle, "_stage_agent_record_clickhouse", lambda *a, **k: None
        )
    ups: list[str] = []
    monkeypatch.setattr(
        container_lifecycle, "deploy_up_web_terminals", lambda *a, **k: ups.append("up")
    )

    container_lifecycle._start_stack(config, compose_files, _root(config), detached=True)

    assert ups == ["up"], "the start was not refused over a file no container opens"
