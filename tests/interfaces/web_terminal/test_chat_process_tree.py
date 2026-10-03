"""Ending the Simple view's agent, on every path that ends it, also ends what the agent started.

Real processes: the chat child is the ``_chat_child`` stand-in, held by a real
``OperatorSession`` through the agent runner, and the commands it starts are
real sleepers in sessions of their own, beside a helper in the server's group.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import signal
import sys
import time

import pytest

from osprey.interfaces.web_terminal import operator_session, process_tree
from osprey.interfaces.web_terminal.chat_session_pool import ChatSessionPool
from osprey.interfaces.web_terminal.operator_session import OperatorRegistry, OperatorSession
from osprey.interfaces.web_terminal.session_handoff import HandoffRefused
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
from tests.interfaces.web_terminal._pty_child import CHILD_HANG_CEILING, kill_quietly, pid_gone

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="no process groups on Windows"),
    pytest.mark.skipif(shutil.which("ps") is None, reason="needs ps to read the process tree"),
]


def _pool(idle_seconds: float = 900.0, max_sessions: int = 5) -> ChatSessionPool:
    """A real pool that builds ``OperatorSession``s, as ``OperatorRegistry`` does."""
    return ChatSessionPool(
        factory=lambda cwd, env, session_key: OperatorSession(
            cwd=cwd, env=env, session_key=session_key
        ),
        max_sessions=max_sessions,
        idle_seconds=idle_seconds,
    )


async def test_stop_ends_the_processes_the_chat_agent_started(tmp_path):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py", "orbit_poll.py"))
    session = await start_chat(factory)
    pids: list[int] = []
    try:
        pids = wait_for_chat_pids(session, factory.pid_files[0], 3)

        await session.stop()

        assert session.process_exited is True
        assert all(pid_gone(pid) for pid in pids)
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

        assert all(pid_gone(pid, within=10.0) for pid in pids)
    finally:
        await session.stop()
        kill_quietly(pids)


async def test_stop_logs_what_it_ended(tmp_path, caplog):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py",))
    session = await start_chat(factory)
    pids: list[int] = []
    try:
        pids = wait_for_chat_pids(session, factory.pid_files[0], 2)

        with caplog.at_level(logging.INFO, logger=operator_session.__name__):
            await session.stop()

        grandchild = pids[0]
        assert [
            record
            for record in caplog.records
            if "magnet_scan.py" in record.getMessage() and str(grandchild) in record.getMessage()
        ]
    finally:
        await session.stop()
        kill_quietly(pids)


async def test_stop_of_a_child_that_already_exited_looks_for_nothing(monkeypatch):
    def factory(**_kwargs):
        return ChildClient([sys.executable, "-c", "pass"])

    session = await start_chat(factory)
    deadline = time.monotonic() + CHILD_HANG_CEILING
    while not session.process_exited:
        assert time.monotonic() < deadline
        await asyncio.sleep(0.05)
    looks: list[None] = []

    def spy():
        looks.append(None)
        return {}

    monkeypatch.setattr(process_tree, "snapshot", spy)

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

        assert pid_gone(pids[0])
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

        assert pid_gone(pids[0])
    finally:
        await pool.terminate(KEY)
        await pool.terminate(OTHER_KEY)
        kill_quietly(pids)


async def test_reaping_an_idle_chat_ends_what_it_started(tmp_path):
    factory = child_factory(tmp_path)
    pool = _pool(idle_seconds=0.01)
    pids: list[int] = []
    try:
        with sdk_seam(factory):
            session, _ = await pool.get_or_create(KEY, str(tmp_path))
        pids = wait_for_chat_pids(session, factory.pid_files[0], 1)
        await asyncio.sleep(0.05)

        assert await pool.reap_idle() == 1

        assert pid_gone(pids[0])
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

        assert pid_gone(pids[0])
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

        assert all(pid_gone(pid) for pid in grandchildren)
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

        assert pid_gone(pids[0])
    finally:
        await operators.cleanup_all()
        kill_quietly(pids)


# ---------------------------------------------------------------------------
# The hand-off asks first
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="the hand-off does not look for started commands")
async def test_the_expert_view_asks_before_ending_the_chats_started_processes(tmp_path):
    factory = child_factory(tmp_path, scripts=("magnet_scan.py",))
    session = await start_chat(factory, session_key=KEY)
    app = make_app()
    chats(app).sessions[KEY] = session
    pids: list[int] = []
    try:
        pids = wait_for_chat_pids(session, factory.pid_files[0], 2)

        with pytest.raises(HandoffRefused) as refused:
            await acquire(app, KEY, "expert", object(), spawn=pty_spawner(app))

        assert refused.value.error == "handoff_started_commands"
        assert chats(app).sessions[KEY] is session
        assert session.is_active
        assert not pid_gone(pids[0], within=0.0)
        assert registry(app).get_session(KEY) is None
        assert_released(app)
    finally:
        await session.stop()
        kill_quietly(pids)
