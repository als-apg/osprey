"""What the hand-off promises the operator about the commands the outgoing terminal agent started.

Taking a session key from the Expert view ends the terminal agent, and ending
it ends every process it started. A command the agent started in a process
group of its own (a scan script, a polling loop) is therefore ended by the
hand-off, so the door asks before it acts: it refuses with the list until the
request agrees to end them, and with nothing running it goes straight through.

Real processes under the real door: the outgoing terminal is a real PTY child
pooled under the key, built by ``_pty_child.detaching_child_script`` to start
its commands the way the agent CLI does, and the door runs through
``_handoff_harness`` on the fake clock, so the interrupt grace is not spent.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from osprey.deployment import process_table
from osprey.deployment.subprocess_capture import group_is_gone
from osprey.interfaces.web_terminal.pty_manager import PtySession
from osprey.interfaces.web_terminal.session_handoff import (
    ERROR_HANDOFF_STARTED_COMMANDS,
    HandoffRefused,
)
from osprey.interfaces.web_terminal.turn_state import IDLE
from tests._processes import kill_quietly, wait_gone
from tests.interfaces.web_terminal._handoff_harness import (
    KEY,
    acquire,
    assert_released,
    chat_spawner,
    chats,
    make_app,
    registry,
    set_store,
)
from tests.interfaces.web_terminal._pty_child import (
    detaching_child_script,
    sleeper_script,
    wait_for_pids,
)

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="no process groups on Windows"),
    pytest.mark.usefixtures("readable_process_table"),
]


def _pool_terminal(
    app: SimpleNamespace, tmp_path: Path, names: tuple[str, ...]
) -> tuple[PtySession, Path, Path]:
    """Pool a real terminal under :data:`KEY` that starts one command per name in *names*.

    Returns the session, the file its pids are written to (one per command,
    then the helper in the child's own group), and the file its input lands in.
    """
    pid_file = tmp_path / "pids"
    input_file = tmp_path / "input"
    input_file.touch()
    scripts = [sleeper_script(tmp_path, name) for name in names]
    source = detaching_child_script(pid_file, scripts=scripts, input_file=input_file)
    session, reused = registry(app).get_or_create_session(KEY, [sys.executable, "-c", source])
    assert not reused
    return session, pid_file, input_file


def _hold_writes_until_read(session: PtySession, input_file: Path) -> None:
    """Return from *session*'s input write only once the real child has recorded it.

    The fake clock spends the interrupt grace at once, so without this the door
    can terminate the child before it has read what it was sent.
    """
    write = session.write_input

    def write_then_wait(data: bytes) -> None:
        write(data)
        deadline = time.monotonic() + 30.0
        while not input_file.read_bytes() and time.monotonic() < deadline:
            time.sleep(0.01)

    session.write_input = write_then_wait  # type: ignore[method-assign]


async def test_stop_and_switch_asks_before_a_started_command_is_ended(tmp_path):
    app = make_app(hook=False)
    session, pid_file, input_file = _pool_terminal(
        app, tmp_path, ("magnet_scan.py", "orbit_poll.py")
    )
    pids: list[int] = []
    try:
        pids = wait_for_pids(session, pid_file, 3)
        grandchild = pids[0]

        with pytest.raises(HandoffRefused) as refused:
            await acquire(app, KEY, "simple", object(), interrupt=True, spawn=chat_spawner(app))

        assert refused.value.error == ERROR_HANDOFF_STARTED_COMMANDS
        # The refusal lists each started command.
        labels = [command["label"] for command in refused.value.extra["commands"]]
        assert sorted(labels) == ["magnet_scan.py", "orbit_poll.py"]
        assert registry(app).get_session(KEY) is session
        assert session.is_alive
        assert input_file.read_bytes() == b""
        assert not group_is_gone(grandchild)
        assert KEY not in chats(app).sessions
        assert_released(app)
    finally:
        registry(app).cleanup_all()
        kill_quietly(pids)


async def test_stop_both_ends_the_agent_and_its_commands(tmp_path):
    app = make_app(hook=False)
    session, pid_file, input_file = _pool_terminal(app, tmp_path, ("magnet_scan.py",))
    pids: list[int] = []
    try:
        pids = wait_for_pids(session, pid_file, 2)
        _hold_writes_until_read(session, input_file)

        await acquire(
            app,
            KEY,
            "simple",
            object(),
            interrupt=True,
            spawn=chat_spawner(app),
            end_started=True,
        )

        assert input_file.read_bytes() == b"\x1b"
        assert not session.is_alive
        await wait_gone(pids)
        assert KEY in chats(app).sessions
        assert registry(app).get_session(KEY) is None
        assert_released(app)
    finally:
        registry(app).cleanup_all()
        kill_quietly(pids)


async def test_a_plain_flip_that_finds_the_agent_idle_asks_too(tmp_path):
    app = make_app(hook=True)
    session, pid_file, _ = _pool_terminal(app, tmp_path, ("magnet_scan.py",))
    set_store(app, IDLE, app.clock.now)
    pids: list[int] = []
    try:
        pids = wait_for_pids(session, pid_file, 2)

        with pytest.raises(HandoffRefused) as refused:
            await acquire(app, KEY, "simple", object(), spawn=chat_spawner(app))

        assert refused.value.error == ERROR_HANDOFF_STARTED_COMMANDS
        assert registry(app).get_session(KEY) is session
        assert session.is_alive
        assert not group_is_gone(pids[0])
        assert KEY not in chats(app).sessions
        assert_released(app)
    finally:
        registry(app).cleanup_all()
        kill_quietly(pids)


async def test_with_nothing_started_the_stop_goes_straight_through(tmp_path):
    app = make_app(hook=False)
    session, pid_file, _ = _pool_terminal(app, tmp_path, ())
    pids: list[int] = []
    try:
        pids = wait_for_pids(session, pid_file, 1)

        await acquire(app, KEY, "simple", object(), interrupt=True, spawn=chat_spawner(app))

        assert not session.is_alive
        await wait_gone([pids[0]])
        assert KEY in chats(app).sessions
        assert_released(app)
    finally:
        registry(app).cleanup_all()
        kill_quietly(pids)


def _fail_reads(monkeypatch) -> None:
    """Make every process-table read fail the way no retry can answer: a refused read."""

    def unreadable():
        raise process_table.ProcessTableUnreadable(PermissionError("process table"))

    monkeypatch.setattr(process_table, "snapshot", unreadable)


def _fail_first_read(monkeypatch) -> list[bool]:
    """Make the first process-table read fail the way a retry can answer: a busy table.

    Returns the log of reads: each entry says whether that read reached the real process table.
    """
    real = process_table.snapshot
    reads: list[bool] = []

    def flaky():
        reads.append(bool(reads))
        if not reads[-1]:
            raise process_table.ProcessTableUnreadable(OSError("process table"))
        return real()

    monkeypatch.setattr(process_table, "snapshot", flaky)
    return reads


async def test_a_failed_process_table_read_is_looked_at_again_before_the_door_acts(
    tmp_path, monkeypatch
):
    app = make_app(hook=False)
    session, pid_file, input_file = _pool_terminal(app, tmp_path, ("magnet_scan.py",))
    pids: list[int] = []
    try:
        pids = wait_for_pids(session, pid_file, 2)
        reads = _fail_first_read(monkeypatch)

        with pytest.raises(HandoffRefused) as refused:
            await acquire(app, KEY, "simple", object(), interrupt=True, spawn=chat_spawner(app))

        assert reads[0] is False
        assert refused.value.error == ERROR_HANDOFF_STARTED_COMMANDS
        assert [c["label"] for c in refused.value.extra["commands"]] == ["magnet_scan.py"]
        assert refused.value.extra["unread"] is False
        assert session.is_alive
        assert input_file.read_bytes() == b""
        assert not group_is_gone(pids[0])
    finally:
        monkeypatch.undo()
        registry(app).cleanup_all()
        kill_quietly(pids)


async def test_a_process_table_that_never_reads_asks_instead_of_ending_unseen_commands(
    tmp_path, monkeypatch
):
    app = make_app(hook=False)
    session, pid_file, input_file = _pool_terminal(app, tmp_path, ("magnet_scan.py",))
    pids: list[int] = []
    try:
        pids = wait_for_pids(session, pid_file, 2)
        _fail_reads(monkeypatch)

        with pytest.raises(HandoffRefused) as refused:
            await acquire(app, KEY, "simple", object(), interrupt=True, spawn=chat_spawner(app))

        assert refused.value.error == ERROR_HANDOFF_STARTED_COMMANDS
        assert refused.value.extra == {"commands": [], "unread": True}
        assert registry(app).get_session(KEY) is session
        assert session.is_alive
        assert input_file.read_bytes() == b""
        assert KEY not in chats(app).sessions
        assert_released(app)
        monkeypatch.undo()
        assert not group_is_gone(pids[0])
    finally:
        monkeypatch.undo()
        registry(app).cleanup_all()
        kill_quietly(pids)


async def test_agreeing_to_end_unseen_commands_goes_through_without_a_look(tmp_path, monkeypatch):
    app = make_app(hook=False)
    session, pid_file, input_file = _pool_terminal(app, tmp_path, ("magnet_scan.py",))
    pids: list[int] = []
    try:
        pids = wait_for_pids(session, pid_file, 2)
        _hold_writes_until_read(session, input_file)
        _fail_reads(monkeypatch)

        await acquire(
            app,
            KEY,
            "simple",
            object(),
            interrupt=True,
            spawn=chat_spawner(app),
            end_started=True,
        )

        assert input_file.read_bytes() == b"\x1b"
        assert not session.is_alive
        assert KEY in chats(app).sessions
        assert_released(app)
    finally:
        monkeypatch.undo()
        registry(app).cleanup_all()
        kill_quietly(pids)
