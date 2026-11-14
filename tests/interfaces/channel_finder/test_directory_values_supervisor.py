"""The supervisor of the Channels panel's value child.

The supervision tests run a fake child: a script that records the environment
it was started with, then exits with the status its plan names or serves batch
reads on the socket it inherits. The refusal test runs the real child under a
forbidden variable.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from osprey.interfaces.channel_finder import directory_values_child as child
from osprey.interfaces.channel_finder import directory_values_ipc as ipc
from osprey.interfaces.channel_finder import directory_values_supervisor as supervision
from osprey.interfaces.channel_finder.directory_values_supervisor import ValueSupervisor
from osprey.interfaces.web_terminal.supervised_child import BACKOFF_S, PREVIOUS_EXIT_ENV
from osprey.runtime.session_env import FORBIDDEN_ENV
from osprey_connectors import posture_store
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV

_FAKE_CHILD = """
import json, os, socket, struct, sys, time

fd = int(sys.argv[sys.argv.index("--socket-fd") + 1])
log = os.environ["FAKE_CHILD_LOG"]
record = json.dumps({"at": time.monotonic(), "env": dict(os.environ)}) + "\\n"
with open(log, "ab", buffering=0) as handle:
    handle.write(record.encode())
with open(log) as handle:
    starts = sum(1 for _ in handle)
with open(os.environ["FAKE_CHILD_PLAN"]) as handle:
    plan = json.load(handle)
step = plan[min(starts, len(plan)) - 1]
if "wait_for" in step:
    while not os.path.exists(step["wait_for"]):
        time.sleep(0.02)
if "stderr" in step:
    sys.stderr.write(step["stderr"] + "\\n")
    sys.stderr.flush()
if "exit" in step:
    sys.exit(step["exit"])
listener = socket.socket(fileno=fd)
while True:
    conn, _ = listener.accept()
    with conn:
        header = conn.recv(4, socket.MSG_WAITALL)
        body = conn.recv(struct.unpack(">I", header)[0], socket.MSG_WAITALL)
        names = json.loads(body)["names"]
        rows = [
            {"name": n, "outcome": "found", "value": 1.5, "timestamp": None,
             "severity": 0, "status": "NO_ALARM"}
            for n in names
        ]
        out = json.dumps(rows).encode()
        conn.sendall(struct.pack(">I", len(out)) + out)
"""


def _wait_until(condition: Callable[[], bool], timeout: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


class FakeChild:
    """The fake child's script, plan and start log."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.script = root / "fake_child.py"
        self.script.write_text(_FAKE_CHILD)
        self.log = root / "starts.ndjson"
        self.plan_file = root / "plan.json"
        self.release = root / "release"
        monkeypatch.setenv("FAKE_CHILD_LOG", str(self.log))
        monkeypatch.setenv("FAKE_CHILD_PLAN", str(self.plan_file))

    def plan(self, *steps: dict[str, Any]) -> None:
        self.plan_file.write_text(json.dumps(list(steps)))

    def argv(self, fd: int) -> list[str]:
        return [sys.executable, "-I", str(self.script), ipc.SOCKET_FD_ARG, str(fd)]

    def starts(self) -> list[dict[str, Any]]:
        if not self.log.exists():
            return []
        # A record is a line with its newline: the child may be part-way through one.
        complete, _, _ = self.log.read_text().rpartition("\n")
        return [json.loads(line) for line in complete.splitlines() if line]


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.yml"
    path.write_text("project_name: test\n")
    return path


@pytest.fixture()
def session_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_file: Path) -> Path:
    """The inherited variables of Value child IPC, and none of the forbidden ones."""
    state = tmp_path / "control_target" / "operator"
    state.mkdir(parents=True)
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(state))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "test-operator")
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(config_file))
    for name in list(os.environ):
        if name in FORBIDDEN_ENV or name.startswith("OSPREY_CONTROL_TARGET"):
            monkeypatch.delenv(name)
    return state


@pytest.fixture()
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, session_env: Path) -> FakeChild:  # noqa: ARG001
    return FakeChild(tmp_path, monkeypatch)


@pytest.fixture()
def supervise(config_file: Path) -> Iterator[Callable[..., ValueSupervisor]]:
    started: list[ValueSupervisor] = []

    def make(fake: FakeChild | None = None, **kwargs: Any) -> ValueSupervisor:
        if fake is not None:
            kwargs.setdefault("argv_for", fake.argv)
        supervisor = ValueSupervisor(config_file, **kwargs)
        started.append(supervisor)
        supervisor.start()
        return supervisor

    yield make
    for supervisor in started:
        supervisor.stop()


def _read(supervisor: ValueSupervisor, names: list[str]) -> list[ipc.ValueRow]:
    return asyncio.run(supervisor.read(names))


# -- the D8 schedule -----------------------------------------------------------


def test_the_default_schedule_is_d8() -> None:
    assert BACKOFF_S == (0.5, 1.0, 2.0, 4.0)
    assert supervision.RESPAWN_NOW_CODES == {75, 76}
    assert supervision.NEVER_RESPAWN_CODES == {78}


def test_the_client_waits_past_the_row_bound() -> None:
    assert supervision.READ_WAIT_S > child.PV_READ_BOUND_S + child.ROW_MARGIN_S


@pytest.mark.parametrize("status", [75, 76])
def test_a_move_or_stale_exit_is_replaced_at_once(fake, supervise, status: int) -> None:
    fake.plan({"exit": status}, {})
    # A backoff no test could outwait: only a replacement at once is seen.
    supervisor = supervise(fake, backoff_s=(60.0,), quick_exit_s=0.0)

    assert _wait_until(lambda: len(fake.starts()) == 2)
    assert fake.starts()[1]["env"][PREVIOUS_EXIT_ENV] == str(status)
    assert supervisor.unavailable_reason is None


def test_a_refused_start_is_not_replaced_and_says_why(fake, supervise) -> None:
    fake.plan({"exit": 78, "stderr": f"{ipc.START_REFUSED_PREFIX}OSPREY_X must not be set"})
    supervisor = supervise(fake, backoff_s=(0.01,), quick_exit_s=0.0)

    assert _wait_until(lambda: supervisor.unavailable_reason is not None)
    time.sleep(0.3)
    assert len(fake.starts()) == 1
    assert supervisor.unavailable_reason == "OSPREY_X must not be set"
    rows = _read(supervisor, ["SR:C01:BPM1:X", "SR:C01:BPM1:Y"])
    assert [row.outcome for row in rows] == [ipc.OUTCOME_UNAVAILABLE] * 2
    assert {row.reason for row in rows} == {"OSPREY_X must not be set"}
    assert [row.name for row in rows] == ["SR:C01:BPM1:X", "SR:C01:BPM1:Y"]


def test_crashes_are_replaced_after_a_growing_wait(fake, supervise) -> None:
    fake.plan({"exit": 1}, {"exit": 1}, {"exit": 1}, {})
    backoff = (0.2, 0.4, 0.8)
    supervise(fake, backoff_s=backoff, quick_exit_s=0.0)

    assert _wait_until(lambda: len(fake.starts()) == 4)
    at = [start["at"] for start in fake.starts()]
    gaps = [later - earlier for earlier, later in zip(at, at[1:], strict=False)]
    for gap, wait in zip(gaps, backoff, strict=True):
        assert gap >= wait


def test_five_crashes_in_the_window_make_values_unavailable(fake, supervise) -> None:
    # A shell, not an interpreter: five starts must fit the 60 s window on a busy host.
    def crash(fd: int) -> list[str]:
        return ["/bin/sh", "-c", 'echo "$0" >> "$FAKE_CHILD_LOG"; exit 3', str(fd)]

    supervisor = supervise(argv_for=crash, backoff_s=(0.01,), quick_exit_s=0.0)

    assert _wait_until(lambda: supervisor.unavailable_reason is not None)
    assert len(fake.log.read_text().splitlines()) == 5
    assert "5 times" in supervisor.unavailable_reason
    assert "status 3" in supervisor.unavailable_reason
    (row,) = _read(supervisor, ["SR:C01:BPM1:X"])
    assert row.outcome == ipc.OUTCOME_UNAVAILABLE
    assert row.reason == supervisor.unavailable_reason


def test_a_replacement_after_a_move_is_rendered_again(
    fake, supervise, tmp_path: Path, monkeypatch
) -> None:
    fake.plan({"wait_for": str(fake.release), "exit": 75}, {})
    supervise(fake, backoff_s=(60.0,), quick_exit_s=0.0)
    assert _wait_until(lambda: len(fake.starts()) == 1)

    moved = tmp_path / "control_target" / "moved"
    moved.mkdir(parents=True)
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(moved))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "second-operator")
    fake.release.touch()

    assert _wait_until(lambda: len(fake.starts()) == 2)
    first, second = (start["env"] for start in fake.starts())
    assert first[AUDIT_IDENTITY_ENV] == "test-operator"
    assert second[posture_store.CONTROL_CONTEXT_DIR_ENV_VAR] == str(moved)
    assert second[AUDIT_IDENTITY_ENV] == "second-operator"


# -- what the child is given ---------------------------------------------------


def test_the_socket_directory_is_private(fake, supervise) -> None:
    fake.plan({})
    supervisor = supervise(fake)

    directory = supervisor.socket_path.parent
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_ISSOCK(supervisor.socket_path.stat().st_mode)


def test_the_rendered_environment(session_env: Path, config_file: Path, monkeypatch) -> None:
    monkeypatch.setenv("OSPREY_TERMINAL_SECRET", "not-a-secret")
    monkeypatch.setenv("CONFIG_FILE", "/elsewhere/config.yml")
    supervisor = ValueSupervisor(config_file)

    env = supervisor.render_environment()

    assert "OSPREY_TERMINAL_SECRET" not in set(env)
    assert env["OSPREY_CONFIG"] == env["CONFIG_FILE"] == str(config_file.resolve())
    assert env[posture_store.CONTROL_CONTEXT_DIR_ENV_VAR] == str(session_env)
    assert env[AUDIT_IDENTITY_ENV] == "test-operator"


@pytest.mark.usefixtures("session_env")
def test_the_audit_identity_is_the_panels_operator(config_file, monkeypatch) -> None:
    # The terminal rung is dropped with the terminal's names; the operator stays.
    monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
    env = ValueSupervisor(config_file).render_environment()

    assert TERMINAL_USER_ENV not in set(env)
    assert env[AUDIT_IDENTITY_ENV] == "alice"


@pytest.mark.usefixtures("session_env")
def test_a_forbidden_variable_is_passed_through(config_file, monkeypatch) -> None:
    monkeypatch.setenv("OSPREY_CONTROL_OWNER", "someone")
    env = ValueSupervisor(config_file).render_environment()

    assert env["OSPREY_CONTROL_OWNER"] == "someone"


@pytest.mark.usefixtures("session_env")
def test_a_forbidden_variable_makes_values_unavailable_with_the_reason(
    supervise, monkeypatch
) -> None:
    monkeypatch.setenv("OSPREY_CONTROL_OWNER", "someone")
    supervisor = supervise()  # the real child

    assert _wait_until(lambda: supervisor.unavailable_reason is not None)
    assert supervisor.unavailable_reason == "OSPREY_CONTROL_OWNER must not be set"
    (row,) = _read(supervisor, ["SR:C01:BPM1:X"])
    assert row.outcome == ipc.OUTCOME_UNAVAILABLE
    assert row.reason == "OSPREY_CONTROL_OWNER must not be set"


# -- reads and the end ---------------------------------------------------------


def test_reads_go_through_the_child(fake, supervise) -> None:
    fake.plan({})
    supervisor = supervise(fake)

    rows = _read(supervisor, ["SR:C01:BPM1:X", "SR:C01:BPM1:Y"])

    assert [(row.name, row.outcome, row.value) for row in rows] == [
        ("SR:C01:BPM1:X", ipc.OUTCOME_FOUND, 1.5),
        ("SR:C01:BPM1:Y", ipc.OUTCOME_FOUND, 1.5),
    ]


def test_a_read_waits_through_a_replacement(fake, supervise, monkeypatch) -> None:
    # The backlog is under test, not the wait: two interpreter starts on a busy host.
    monkeypatch.setattr(supervision, "READ_WAIT_S", 30.0)
    fake.plan({"exit": 75}, {})
    supervisor = supervise(fake, quick_exit_s=0.0)

    (row,) = _read(supervisor, ["SR:C01:BPM1:X"])

    assert row.outcome == ipc.OUTCOME_FOUND


def test_more_than_a_batch_is_refused(fake, supervise) -> None:
    fake.plan({})
    supervisor = supervise(fake)

    with pytest.raises(ValueError, match="50"):
        _read(supervisor, [f"PV{i}" for i in range(ipc.MAX_NAMES + 1)])


def test_an_empty_read_asks_nothing(fake, supervise) -> None:
    fake.plan({})
    supervisor = supervise(fake)

    assert _read(supervisor, []) == []


def test_a_child_that_does_not_answer_gives_unavailable_rows(fake, supervise, monkeypatch) -> None:
    fake.plan({"wait_for": str(fake.release)})
    supervisor = supervise(fake)
    monkeypatch.setattr(supervision, "READ_WAIT_S", 0.3)

    (row,) = _read(supervisor, ["SR:C01:BPM1:X"])

    assert row.outcome == ipc.OUTCOME_UNAVAILABLE
    assert "did not answer" in row.reason


def test_a_clean_stop_ends_the_child_and_removes_the_socket(fake, supervise) -> None:
    fake.plan({})
    supervisor = supervise(fake)
    assert _wait_until(lambda: supervisor.pid is not None)
    pid = supervisor.pid
    directory = supervisor.socket_path.parent

    supervisor.stop()

    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not directory.exists()
    (row,) = _read(supervisor, ["SR:C01:BPM1:X"])
    assert row.outcome == ipc.OUTCOME_UNAVAILABLE


def test_a_child_that_cannot_start_makes_values_unavailable(fake, supervise) -> None:
    supervisor = supervise(fake, argv_for=lambda fd: ["/nonexistent/python", str(fd)])

    assert supervisor.unavailable_reason is not None
    assert "could not be started" in supervisor.unavailable_reason


# -- the Channels lifespan -----------------------------------------------------


@pytest.mark.usefixtures("mock_registry")
def test_the_lifespan_supervises_only_with_the_section(monkeypatch, config_file) -> None:
    pytest.importorskip("osprey_connectors.service_answers")
    from fastapi.testclient import TestClient

    from osprey.interfaces.channel_finder.app import create_app

    events: list[str] = []

    class Recording:
        def __init__(self, config_path: Any) -> None:
            events.append(f"init {Path(config_path).name}")

        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")

    monkeypatch.setattr(supervision, "ValueSupervisor", Recording)
    monkeypatch.setenv("OSPREY_CONFIG", str(config_file))
    base: dict[str, Any] = {
        "channel_finder": {
            "pipeline_mode": "in_context",
            "pipelines": {"in_context": {"database": {"path": "/tmp/db.json", "type": "flat"}}},
        },
        "control_system": {"type": "virtual_accelerator"},
        "services": {"virtual_accelerator": {"port": 5064}},
    }
    seam = "osprey_connectors.workspace.load_osprey_config"

    monkeypatch.setattr(seam, lambda: base)
    with TestClient(create_app(project_cwd="/tmp/test-project")) as client:
        assert getattr(client.app.state, "directory_values", None) is None
    assert events == []

    monkeypatch.setattr(seam, lambda: {**base, "pv_directory": {}})
    with TestClient(create_app(project_cwd="/tmp/test-project")) as client:
        assert isinstance(client.app.state.directory_values, Recording)
        assert events == ["init config.yml", "start"]
    assert events == ["init config.yml", "start", "stop"]
