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

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from osprey.interfaces.web_terminal.pty_manager import PtySession
from osprey.interfaces.web_terminal.session_handoff import (
    ERROR_HANDOFF_STARTED_COMMANDS,
    HandoffRefused,
)
from osprey.interfaces.web_terminal.turn_state import IDLE
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
    kill_quietly,
    pid_gone,
    sleeper_script,
    wait_for_pids,
)

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="no process groups on Windows"),
    pytest.mark.skipif(shutil.which("ps") is None, reason="needs ps to read the process tree"),
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


async def test_stop_and_switch_asks_before_a_started_command_is_ended(tmp_path):
    app = make_app(hook=False)
    session, pid_file, input_file = _pool_terminal(app, tmp_path, ("magnet_scan.py",))
    pids: list[int] = []
    try:
        pids = wait_for_pids(session, pid_file, 2)
        grandchild = pids[0]

        with pytest.raises(HandoffRefused) as refused:
            await acquire(app, KEY, "simple", object(), interrupt=True, spawn=chat_spawner(app))

        assert refused.value.error == ERROR_HANDOFF_STARTED_COMMANDS
        assert registry(app).get_session(KEY) is session
        assert session.is_alive
        assert input_file.read_bytes() == b""
        assert not pid_gone(grandchild, within=0.0)
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
        assert all(pid_gone(pid) for pid in pids)
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
        assert not pid_gone(pids[0], within=0.0)
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
        assert pid_gone(pids[0])
        assert KEY in chats(app).sessions
        assert_released(app)
    finally:
        registry(app).cleanup_all()
        kill_quietly(pids)


async def test_the_refusal_lists_each_started_command(tmp_path):
    app = make_app(hook=False)
    session, pid_file, _ = _pool_terminal(app, tmp_path, ("magnet_scan.py", "orbit_poll.py"))
    pids: list[int] = []
    try:
        pids = wait_for_pids(session, pid_file, 3)

        with pytest.raises(HandoffRefused) as refused:
            await acquire(app, KEY, "simple", object(), interrupt=True, spawn=chat_spawner(app))

        labels = [command["label"] for command in refused.value.extra["commands"]]
        assert sorted(labels) == ["magnet_scan.py", "orbit_poll.py"]
    finally:
        registry(app).cleanup_all()
        kill_quietly(pids)
