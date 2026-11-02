"""Ending the chat's agent, on every path that ends it, also ends what the agent started.

Real processes: the chat child is the ``_chat_child`` stand-in, held by a real
``OperatorSession`` through the agent runner, and the commands it starts are
real sleepers in sessions of their own, beside a helper in the server's group.
"""

from __future__ import annotations

import asyncio
import logging
import math
import signal
import sys
import time
from collections.abc import Callable

import pytest

from osprey.deployment import process_table
from osprey.deployment.subprocess_capture import group_is_gone
from osprey.interfaces.web_terminal import operator_session, process_tree
from osprey.interfaces.web_terminal.chat_session_pool import ChatSessionPool
from osprey.interfaces.web_terminal.operator_session import OperatorRegistry, OperatorSession
from osprey.interfaces.web_terminal.session_handoff import HandoffRefused
from tests._live import async_wait_until
from tests._processes import kill_quietly, wait_gone
from tests.interfaces.web_terminal._chat_child import (
    ChildClient,
    child_factory,
    start_chat,
    wait_for_chat_pids,
)
from tests.interfaces.web_terminal._fakes import sdk_seam
from tests.interfaces.web_terminal._handoff_harness import (
    KEY,
    OTHER_KEY,
    acquire,
    assert_released,
    chats,
    make_app,
    pty_spawner,
    registry,
)
from tests.interfaces.web_terminal._pty_child import CHILD_HANG_CEILING

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="no process groups on Windows"),
    pytest.mark.usefixtures("readable_process_table"),
]


IDLE = 60.0


class _Clock:
    """A monotonic clock a test moves by hand."""

    def __init__(self) -> None:
        self.now = time.monotonic()

    def __call__(self) -> float:
        return self.now


def _an_idle_window_after(session: OperatorSession, idle_seconds: float) -> float:
    """A whole-second clock reading at least *idle_seconds* after the session's last turn."""
    return math.ceil(session.last_activity) + idle_seconds


def _pool(
    idle_seconds: float = 900.0,
    max_sessions: int = 5,
    now: Callable[[], float] = time.monotonic,
) -> ChatSessionPool:
    """A real pool that builds ``OperatorSession``s, as ``OperatorRegistry`` does."""
    return ChatSessionPool(
        factory=lambda cwd, env, session_key: OperatorSession(
            cwd=cwd, env=env, session_key=session_key
        ),
        max_sessions=max_sessions,
        idle_seconds=idle_seconds,
        now=now,
    )


async def test_stop_ends_and_logs_the_processes_the_chat_agent_started(tmp_path, caplog):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py", "orbit_poll.py"))
    session = await start_chat(factory)
    pids: list[int] = []
    try:
        pids = wait_for_chat_pids(session, factory.pid_files[0], 3)

        with caplog.at_level(logging.INFO, logger=operator_session.__name__):
            await session.stop()

        assert session.process_exited is True
        await wait_gone(pids)
        grandchild = pids[0]
        assert [
            record
            for record in caplog.records
            if "magnet_scan.py" in record.getMessage() and str(grandchild) in record.getMessage()
        ]
        assert session.started_commands() == []
    finally:
        await session.stop()
        kill_quietly(pids)


async def test_stop_ends_started_processes_that_ignore_sigterm(tmp_path):
    factory = child_factory(
        tmp_path,
        scripts=("magnet_scan.py",),
        helper_ignores_term=True,
        script_ignores=(signal.SIGTERM, signal.SIGHUP),
    )
    session = await start_chat(factory)
    pids: list[int] = []
    try:
        pids = wait_for_chat_pids(session, factory.pid_files[0], 2)

        await session.stop()

        await wait_gone(pids, within=10.0)
    finally:
        await session.stop()
        kill_quietly(pids)


def _ended_pids(session: OperatorSession) -> set[int]:
    """The pids in the groups *session*'s stop says it ended, given that it could look."""
    ending = session.tree_ending
    assert ending is not None, "the stop found no live child to look under"
    assert ending.read, "the stop could not read the process table"
    return {pid for group in ending.ended_groups for pid, _ in group.members}


def _fail_first_reads(monkeypatch, failures: int) -> list[bool]:
    """Make the first *failures* process-table reads fail; return the log of reads.

    Each entry says whether that read reached the real table.
    """
    real = process_table.snapshot
    reads: list[bool] = []

    def flaky():
        reads.append(len(reads) >= failures)
        if not reads[-1]:
            raise process_table.ProcessTableUnreadable(OSError("process table"))
        return real()

    monkeypatch.setattr(process_table, "snapshot", flaky)
    monkeypatch.setattr(process_table, "_READ_RETRY_S", 0.0)
    return reads


async def test_stop_ends_what_the_agent_started_when_the_first_process_table_read_fails(
    tmp_path, monkeypatch
):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py",))
    session = await start_chat(factory)
    pids: list[int] = []
    try:
        pids = wait_for_chat_pids(session, factory.pid_files[0], 2)
        grandchild = pids[0]
        reads = _fail_first_reads(monkeypatch, 1)

        await session.stop()

        assert reads[0] is False
        assert grandchild in _ended_pids(session)
        await wait_gone([grandchild])
    finally:
        monkeypatch.undo()
        await session.stop()
        kill_quietly(pids)


async def test_stop_warns_and_reports_an_unread_tree_when_the_table_stays_unreadable(
    tmp_path, monkeypatch, caplog
):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py",))
    session = await start_chat(factory)
    pids: list[int] = []
    try:
        pids = wait_for_chat_pids(session, factory.pid_files[0], 2)
        child = session.pid
        reads: list[None] = []

        def unreadable():
            reads.append(None)
            raise process_table.ProcessTableUnreadable(OSError("process table"))

        monkeypatch.setattr(process_table, "snapshot", unreadable)
        monkeypatch.setattr(process_tree, "READ_WAIT_S", 0.3)

        with caplog.at_level(logging.WARNING, logger=operator_session.__name__):
            await session.stop()

        assert session.process_exited is True
        # Retried while the child lived, and given up on once the wait had passed.
        assert len(reads) > 1
        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING and "process table" in record.getMessage()
        ]
        assert len(warnings) == 1, warnings
        assert f"chat child {child} " in warnings[0]
        assert session.tree_ending == operator_session.TreeEnding(read=False)
        # Nothing was looked for, so nothing was ended: the report does not claim otherwise.
        monkeypatch.undo()
        assert not group_is_gone(pids[0])
    finally:
        monkeypatch.undo()
        await session.stop()
        kill_quietly(pids)


async def test_stop_reports_started_processes_as_unknown_when_the_table_fails_after_the_first_read(
    tmp_path, monkeypatch, caplog
):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py",))
    session = await start_chat(factory)
    pids: list[int] = []
    try:
        pids = wait_for_chat_pids(session, factory.pid_files[0], 2)
        grandchild = pids[0]
        real = process_table.snapshot
        reads: list[None] = []

        def first_read_only():
            reads.append(None)
            if len(reads) > 1:
                raise process_table.ProcessTableUnreadable(PermissionError("process table"))
            return real()

        monkeypatch.setattr(process_table, "snapshot", first_read_only)

        with caplog.at_level(logging.WARNING, logger=operator_session.__name__):
            await session.stop()

        ending = session.tree_ending
        assert ending is not None
        assert ending.read is True
        assert grandchild in {pid for group in ending.unknown_groups for pid, _ in group.members}
        assert ending.surviving_groups == ()
        assert ending.ended_groups == ()
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        unknown = [
            w
            for w in warnings
            if "process table unreadable: started process(es) may still be running" in w
        ]
        assert len(unknown) == 1, warnings
        assert not any("survived SIGKILL" in w for w in warnings), warnings
    finally:
        monkeypatch.undo()
        await session.stop()
        kill_quietly(pids)


async def test_stop_of_a_child_that_already_exited_looks_for_nothing(monkeypatch):
    def factory(**_kwargs):
        return ChildClient([sys.executable, "-c", "pass"])

    session = await start_chat(factory)
    await async_wait_until(
        lambda: session.process_exited, timeout=CHILD_HANG_CEILING, what="the child to exit"
    )

    looks: list[None] = []

    def spy():
        looks.append(None)
        return {}

    monkeypatch.setattr(process_table, "snapshot", spy)

    await session.stop()

    assert looks == []


async def test_terminating_a_pooled_chat_ends_what_it_started(tmp_path):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py",))
    pool = _pool()
    pids: list[int] = []
    try:
        with sdk_seam(factory):
            session, _ = await pool.get_or_create(KEY, str(tmp_path))
        pids = wait_for_chat_pids(session, factory.pid_files[0], 2)

        await pool.terminate(KEY)

        await wait_gone([pids[0]])
    finally:
        await pool.drain_all()
        kill_quietly(pids)


async def test_evicting_a_chat_ends_what_it_started(tmp_path):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py",))
    pool = _pool(max_sessions=1)
    pids: list[int] = []
    try:
        with sdk_seam(factory):
            first, _ = await pool.get_or_create(KEY, str(tmp_path))
            pids = wait_for_chat_pids(first, factory.pid_files[0], 2)
            await pool.get_or_create(OTHER_KEY, str(tmp_path))

        await wait_gone([pids[0]])
    finally:
        await pool.terminate(KEY)
        await pool.terminate(OTHER_KEY)
        kill_quietly(pids)


async def test_reaping_an_idle_chat_ends_what_it_started(tmp_path):
    factory = child_factory(tmp_path)
    clock = _Clock()
    pool = _pool(idle_seconds=IDLE, now=clock)
    pids: list[int] = []
    try:
        with sdk_seam(factory):
            session, _ = await pool.get_or_create(KEY, str(tmp_path))
        pids = wait_for_chat_pids(session, factory.pid_files[0], 1)

        clock.now = session.last_activity
        assert await pool.reap_idle() == 0
        assert pool.get(KEY) is session

        clock.now = _an_idle_window_after(session, IDLE)
        assert await pool.reap_idle() == 1

        await wait_gone([pids[0]])
    finally:
        await pool.drain_all()
        kill_quietly(pids)


async def test_an_idle_chat_whose_agent_runs_a_started_command_is_not_reaped(tmp_path):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py",))
    clock = _Clock()
    pool = _pool(idle_seconds=IDLE, now=clock)
    pids: list[int] = []
    try:
        with sdk_seam(factory):
            session, _ = await pool.get_or_create(KEY, str(tmp_path))
        pids = wait_for_chat_pids(session, factory.pid_files[0], 2)
        clock.now = _an_idle_window_after(session, IDLE)

        assert await pool.reap_idle() == 0

        assert pool.get(KEY) is session
        assert session.is_active
        assert not group_is_gone(pids[0])
        assert session.last_activity == clock.now
    finally:
        await pool.drain_all()
        kill_quietly(pids)


async def test_an_idle_chat_is_reaped_once_its_started_command_has_exited(tmp_path):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py",))
    clock = _Clock()
    pool = _pool(idle_seconds=IDLE, now=clock)
    pids: list[int] = []
    try:
        with sdk_seam(factory):
            session, _ = await pool.get_or_create(KEY, str(tmp_path))
        pids = wait_for_chat_pids(session, factory.pid_files[0], 2)
        grandchild, helper = pids
        clock.now = seen = _an_idle_window_after(session, IDLE)
        assert await pool.reap_idle() == 0
        assert session.last_activity == seen

        kill_quietly([grandchild])
        await wait_gone([grandchild])
        # The clock counts from the sweep that last saw the command, not from the last turn.
        assert await pool.reap_idle() == 0
        clock.now = seen + IDLE - 1.0
        assert await pool.reap_idle() == 0
        assert pool.get(KEY) is session

        clock.now = seen + IDLE
        assert await pool.reap_idle() == 1
        assert pool.get(KEY) is None
        assert session.process_exited is True
        await wait_gone([helper])
    finally:
        await pool.drain_all()
        kill_quietly(pids)


async def test_a_launch_change_ends_what_the_old_child_started(tmp_path):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py",))
    pool = _pool()
    pids: list[int] = []
    try:
        with sdk_seam(factory):
            first, _ = await pool.get_or_create(KEY, str(tmp_path), {"X": "1"})
            pids = wait_for_chat_pids(first, factory.pid_files[0], 2)
            await pool.get_or_create(KEY, str(tmp_path), {"X": "2"})

        await wait_gone([pids[0]])
    finally:
        await pool.drain_all()
        kill_quietly(pids)


async def test_cleanup_all_ends_what_every_agent_started(tmp_path):
    """Restart, logout and server shutdown all drain the registry this way."""
    factory = child_factory(tmp_path, scripts=("magnet_scan.py",))
    operators = OperatorRegistry()
    pids: list[int] = []
    try:
        with sdk_seam(factory):
            first, _ = await operators.get_or_create_chat_session(KEY, str(tmp_path))
            second, _ = await operators.get_or_create_chat_session(OTHER_KEY, str(tmp_path))
            op = await operators.create_session("op", str(tmp_path))
        for session, pid_file in zip((first, second, op), factory.pid_files, strict=True):
            pids += wait_for_chat_pids(session, pid_file, 2)
        grandchildren = pids[0::2]

        await operators.cleanup_all()

        # What each stop says it ended comes first, so a command left running
        # is reported with its cause rather than as a pid that outlived a wait.
        for session, grandchild in zip((first, second, op), grandchildren, strict=True):
            assert grandchild in _ended_pids(session)
        await wait_gone(grandchildren)
    finally:
        await operators.cleanup_all()
        kill_quietly(pids)


async def test_a_replaced_operator_session_ends_what_it_started(tmp_path):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py",))
    operators = OperatorRegistry()
    pids: list[int] = []
    try:
        with sdk_seam(factory):
            first = await operators.create_session("op", str(tmp_path))
            pids = wait_for_chat_pids(first, factory.pid_files[0], 2)
            await operators.create_session("op", str(tmp_path))

        await wait_gone([pids[0]])
    finally:
        await operators.cleanup_all()
        kill_quietly(pids)


# ---------------------------------------------------------------------------
# The hand-off asks first
# ---------------------------------------------------------------------------


async def test_the_terminal_asks_before_ending_the_chats_started_processes(tmp_path):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py",))
    session = await start_chat(factory, session_key=KEY)
    app = make_app()
    chats(app).sessions[KEY] = session
    pids: list[int] = []
    try:
        pids = wait_for_chat_pids(session, factory.pid_files[0], 2)

        with pytest.raises(HandoffRefused) as refused:
            await acquire(app, KEY, "terminal", object(), spawn=pty_spawner(app))

        assert refused.value.error == "handoff_started_commands"
        assert chats(app).sessions[KEY] is session
        assert session.is_active
        assert not group_is_gone(pids[0])
        assert registry(app).get_session(KEY) is None
        assert_released(app)
    finally:
        await session.stop()
        kill_quietly(pids)


async def test_stop_both_ends_the_chat_and_what_it_started(tmp_path):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py",))
    session = await start_chat(factory, session_key=KEY)
    app = make_app()
    chats(app).sessions[KEY] = session
    pids: list[int] = []
    try:
        pids = wait_for_chat_pids(session, factory.pid_files[0], 2)

        await acquire(app, KEY, "terminal", object(), spawn=pty_spawner(app), end_started=True)

        assert session.process_exited is True
        await wait_gone(pids)
    finally:
        await session.stop()
        kill_quietly(pids)


async def test_started_commands_lists_only_groups_of_their_own(tmp_path):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py",))
    session = await start_chat(factory)
    pids: list[int] = []
    try:
        pids = wait_for_chat_pids(session, factory.pid_files[0], 2)
        grandchild, helper = pids

        commands = await asyncio.to_thread(session.started_commands)

        assert [c.label for c in commands] == ["magnet_scan.py"]
        assert commands[0].pgid == grandchild
        assert helper not in {pid for c in commands for pid, _ in c.members}
    finally:
        await session.stop()
        kill_quietly(pids)
