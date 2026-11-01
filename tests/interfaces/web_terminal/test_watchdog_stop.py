"""The watchdog Stop: each mode against a turn that stops, ignores it, or left a command behind.

Simple mode runs a real ``OperatorSession`` over a fake SDK client; expert mode a
real ``PtyRegistry`` over a fake PTY and a fake clock. The hold record is the real
one on disk, so every assertion about a hold reads what a child process would read.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from claude_agent_sdk import TextBlock

from osprey.agent_runner.events import SystemEvent
from osprey.interfaces.web_terminal.operator_session import OperatorSession, _PendingAsk
from osprey.interfaces.web_terminal.process_tree import ProcessGroup, ProcessTableUnreadable
from osprey.interfaces.web_terminal.routes import watchdog as watchdog_routes
from osprey.interfaces.web_terminal.routes.watchdog import router as watchdog_router
from osprey.interfaces.web_terminal.session_handoff import interrupt_pty_turn
from osprey.services.watchdog.verdict import SURFACE_SAFETY_WATCHDOG, make_prompt_ref
from osprey_connectors.session_hold import is_held, open_flag, open_flags

from ._fakes import assistant_message, result_message, sdk_seam, start_turn
from ._handoff_harness import KEY, RecordingPty, make_app, pool_pty, set_store, until
from .conftest import bare_route_app

ESC = b"\x1b"
pytestmark = pytest.mark.usefixtures("ledger")

RECORD_SESSION = "0f6d7c2e-1a2b-4c3d-8e9f-0123456789ab"
_OS = "osprey.interfaces.web_terminal.operator_session."
SLOW_LOOP = ProcessGroup(
    pgid=4242, members=((4242, 1.0),), label="write_loop.py", command="python write_loop.py"
)


class Registry:
    """The operator registry as the Stop reads it: one chat under one key."""

    def __init__(self, chat=None):
        self.chat = chat

    def get_chat_session(self, key):
        return self.chat if key == KEY else None

    def get_session(self, _key):
        return None


class WritingClient:
    """A turn that loops writing until it is interrupted, unless told to ignore that."""

    def __init__(self, *, ignores_interrupt: bool = False) -> None:
        self.ignores_interrupt = ignores_interrupt
        self.interrupt_calls = 0
        self.writes = 0
        self.stopped_tasks: list[str] = []
        self.on_stop_task = lambda: None
        self.first_yielded = asyncio.Event()
        self._prompts: asyncio.Queue[str] = asyncio.Queue()
        self._interrupted = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def query(self, prompt):
        self._prompts.put_nowait(prompt)

    async def interrupt(self):
        self.interrupt_calls += 1
        if not self.ignores_interrupt:
            self._interrupted.set()

    async def stop_task(self, task_id):
        self.stopped_tasks.append(task_id)
        self.on_stop_task()

    async def receive_response(self):
        await self._prompts.get()
        yield assistant_message([TextBlock("partial")])
        self.first_yielded.set()
        while not self._interrupted.is_set():
            self.writes += 1
            await asyncio.sleep(0.001)
        yield result_message()


@contextlib.asynccontextmanager
async def running_chat(client: WritingClient):
    """A started session with its turn in flight; yields once the turn is writing."""
    session = OperatorSession(cwd="/tmp")
    with sdk_seam(client):
        await session.start()
        try:
            consumer, _frames = start_turn(session, "hi", timeout_s=30.0)
            await asyncio.wait_for(client.first_yielded.wait(), timeout=2.0)
            yield session
            consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer
        finally:
            await session.stop()


class CountingPty(RecordingPty):
    """A PTY whose turn loops writing until Escape arrives, unless it ignores that."""

    def __init__(self, on_escape=None) -> None:
        super().__init__()
        self.on_escape = on_escape
        self.turn_running = True

    def write_input(self, data: bytes) -> None:
        super().write_input(data)
        if data == ESC and self.on_escape is not None:
            self.turn_running = False
            self.on_escape()


@pytest.fixture
def root(tmp_path) -> Path:
    return tmp_path / "var" / "state" / "session_holds"


def _flag(root: Path, flag_id: str, *, source: str = "wd") -> None:
    assert open_flag(
        KEY,
        flag_id=flag_id,
        source=source,
        reason="risky",
        prompt_ref=make_prompt_ref(RECORD_SESSION, flag_id),
        prompt_excerpt="loosen the limits",
        root=root,
    )


def _chat_app(root: Path, session=None):
    return bare_route_app(
        watchdog_router, session_hold_root=root, operator_registry=Registry(session)
    )


def _pty_app(root: Path, *, hook: bool = True):
    """An app over the hand-off harness's PTY pool; ``clock`` is its fake clock."""
    harness = make_app(hook=hook)
    state = {**vars(harness.state), "operator_registry": Registry()}
    app = bare_route_app(watchdog_router, session_hold_root=root, **state)
    return app, harness.clock


async def _post(app, flag_ids, *, answer="stop"):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            "/api/watchdog/flags/answer",
            json={"session_key": KEY, "flag_ids": flag_ids, "answer": answer},
        )


def _entries(ledger, decision=None):
    entries = [e for e in ledger if e["surface"] == SURFACE_SAFETY_WATCHDOG]
    return [e for e in entries if decision is None or e["decision"] == decision]


def _frames(app, kind):
    calls = app.state.broadcaster.broadcast.call_args_list
    return [c.args[0] for c in calls if c.args[0]["type"] == kind]


# ---- cancel's outcome ----


async def test_cancel_reports_idle_with_nothing_running():
    with sdk_seam(WritingClient()):
        session = OperatorSession(cwd="/tmp")
        await session.start()
        try:
            assert await session.cancel() == "idle"
        finally:
            await session.stop()


async def test_cancel_reports_stopped_when_the_turn_ends():
    async with running_chat(WritingClient()) as session:
        assert await session.cancel() == "stopped"


async def test_cancel_reports_abandoned_and_a_second_cancel_does_not_call_it_stopped():
    client = WritingClient(ignores_interrupt=True)
    async with running_chat(client) as session:
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.02):
            assert await session.cancel() == "abandoned"
            # The turn is closed but the response is still running.
            assert await session.cancel() == "abandoned"
        client._interrupted.set()


# ---- Simple mode ----


async def test_simple_stop_ends_the_writing_turn_then_releases(root, ledger):
    _flag(root, "flag-a")
    client = WritingClient()
    async with running_chat(client) as session:
        app = _chat_app(root, session)
        resp = await _post(app, ["flag-a"])
        assert resp.json() == {"released": ["flag-a"], "stop_unconfirmed": []}
        assert not is_held(KEY, root=root)
        writes_at_release = client.writes
        await asyncio.sleep(0.02)
        assert client.writes == writes_at_release
    assert client.interrupt_calls == 1
    [entry] = _entries(ledger, "stop")
    assert entry["reason"] == "stop"
    assert _frames(app, "watchdog_resolved")[0]["flag_ids"] == ["flag-a"]


async def test_simple_turn_that_ignores_the_interrupt_keeps_the_flag_open(root, ledger):
    _flag(root, "flag-a")
    client = WritingClient(ignores_interrupt=True)
    async with running_chat(client) as session:
        app = _chat_app(root, session)
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.02):
            resp = await _post(app, ["flag-a"])
        assert resp.json() == {"released": [], "stop_unconfirmed": ["flag-a"]}
        client._interrupted.set()
    [flag] = open_flags(KEY, root=root)
    assert flag.stop_unconfirmed
    assert is_held(KEY, root=root)
    [entry] = _entries(ledger)
    assert entry["decision"] == "stop_unconfirmed"
    assert _frames(app, "watchdog_flag") == [
        {"type": "watchdog_flag", "session_key": KEY, "flag_id": "flag-a", "watchdog": "wd"}
    ]


async def test_simple_started_command_outliving_the_stop_keeps_the_flag_open_named(root, ledger):
    _flag(root, "flag-a")
    async with running_chat(WritingClient()) as session:
        session.started_commands = lambda: [SLOW_LOOP]
        resp = await _post(_chat_app(root, session), ["flag-a"])
    assert resp.json()["stop_unconfirmed"] == ["flag-a"]
    [flag] = open_flags(KEY, root=root)
    assert flag.started == ("write_loop.py",)
    assert "write_loop.py" in json.dumps(_entries(ledger, "stop_unconfirmed")[0]["detail"])


async def test_simple_unreadable_process_table_keeps_the_flag_open(root, ledger):
    _flag(root, "flag-a")

    def unreadable():
        raise ProcessTableUnreadable(OSError("process table"))

    async with running_chat(WritingClient()) as session:
        session.started_commands = unreadable
        resp = await _post(_chat_app(root, session), ["flag-a"])
    assert resp.json()["stop_unconfirmed"] == ["flag-a"]
    assert is_held(KEY, root=root)
    assert _entries(ledger, "stop_unconfirmed")


async def test_simple_continue_while_stop_waits_is_not_re_marked(root, ledger):
    _flag(root, "flag-a")
    _flag(root, "flag-b")
    client = WritingClient(ignores_interrupt=True)
    async with running_chat(client) as session:
        app = _chat_app(root, session)
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.3):
            stop = asyncio.create_task(_post(app, ["flag-a", "flag-b"]))
            await until(lambda: client.interrupt_calls == 1)
            await _post(app, ["flag-a"], answer="continue")
            resp = await stop
        client._interrupted.set()
    assert resp.json() == {"released": [], "stop_unconfirmed": ["flag-b"]}
    assert [flag.flag_id for flag in open_flags(KEY, root=root)] == ["flag-b"]
    [entry] = _entries(ledger, "stop_unconfirmed")
    assert "flag_id=flag-b" in entry["detail"]
    assert [f["flag_id"] for f in _frames(app, "watchdog_flag")] == ["flag-b"]


async def test_simple_idle_stop_records_and_releases_without_interrupting(root, ledger):
    _flag(root, "flag-a")
    client = WritingClient()
    with sdk_seam(client):
        session = OperatorSession(cwd="/tmp")
        await session.start()
        try:
            resp = await _post(_chat_app(root, session), ["flag-a"])
        finally:
            await session.stop()
    assert resp.json()["released"] == ["flag-a"]
    assert client.interrupt_calls == 0
    assert _entries(ledger, "stop")


async def test_simple_second_stop_stays_open_until_the_started_command_ends(root, ledger):
    _flag(root, "flag-a")
    async with running_chat(WritingClient()) as session:
        session.started_commands = lambda: [SLOW_LOOP]
        app = _chat_app(root, session)
        await _post(app, ["flag-a"])
        await until(lambda: not session._in_response)

        again = await _post(app, ["flag-a"])
        assert again.json() == {"released": [], "stop_unconfirmed": ["flag-a"]}
        assert is_held(KEY, root=root)
        assert len(_entries(ledger, "stop_unconfirmed")) == 2
        assert not _entries(ledger, "stop")

        session.started_commands = lambda: []
        last = await _post(app, ["flag-a"])
    assert last.json() == {"released": ["flag-a"], "stop_unconfirmed": []}
    assert not is_held(KEY, root=root)


async def test_simple_second_stop_with_an_unreadable_process_table_stays_open(root):
    _flag(root, "flag-a")

    def unreadable():
        raise ProcessTableUnreadable(OSError("process table"))

    async with running_chat(WritingClient()) as session:
        session.started_commands = unreadable
        app = _chat_app(root, session)
        await _post(app, ["flag-a"])
        await until(lambda: not session._in_response)
        again = await _post(app, ["flag-a"])
    assert again.json()["stop_unconfirmed"] == ["flag-a"]


async def test_simple_second_stop_after_an_abandoned_background_agent_stays_open(root):
    _flag(root, "flag-a")
    client = WritingClient(ignores_interrupt=True)
    async with running_chat(client) as session:
        await _with_approval_and_background(session)
        app = _chat_app(root, session)
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.02):
            await _post(app, ["flag-a"])
        client._interrupted.set()
        await until(lambda: not session._in_response)
        client.stopped_tasks.clear()

        again = await _post(app, ["flag-a"])
        assert again.json()["stop_unconfirmed"] == ["flag-a"]
        assert client.stopped_tasks == ["task-7"], "the Stop tries the agent again"

        session._track_background(
            SystemEvent(subtype="task_notification", data={"task_id": "task-7", "status": "killed"})
        )
        last = await _post(app, ["flag-a"])
    assert last.json()["released"] == ["flag-a"]


async def test_a_prompt_sent_while_a_response_ends_is_not_called_stopped():
    client = WritingClient(ignores_interrupt=True)
    async with running_chat(client) as session:
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.02):
            assert await session.cancel() == "abandoned"
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.2):
            second = asyncio.create_task(session.cancel())
            await asyncio.sleep(0.05)
            # What _end_response does with a prompt queued behind the response.
            session._pending_turn = True
            session._turn_closed.clear()
            session._in_response = False
            assert await second == "abandoned"
        client._interrupted.set()


async def test_simple_two_stops_at_once_on_a_turn_that_ignores_them_keep_the_flag_open(
    root, ledger
):
    _flag(root, "flag-a")
    client = WritingClient(ignores_interrupt=True)
    async with running_chat(client) as session:
        app = _chat_app(root, session)
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.1):
            first = asyncio.create_task(_post(app, ["flag-a"]))
            await until(lambda: client.interrupt_calls == 1)
            await asyncio.sleep(0.03)
            second = asyncio.create_task(_post(app, ["flag-a"]))
            answers = [(await first).json(), (await second).json()]
        assert answers == [{"released": [], "stop_unconfirmed": ["flag-a"]}] * 2
        assert is_held(KEY, root=root)
        client._interrupted.set()
    assert not _entries(ledger, "stop")


async def test_simple_chat_stop_then_watchdog_stop_on_a_turn_that_ignores_them_keeps_the_flag_open(
    root, ledger
):
    _flag(root, "flag-a")
    client = WritingClient(ignores_interrupt=True)
    async with running_chat(client) as session:
        app = _chat_app(root, session)
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.1):
            quiesce = session.spawn_quiesce()
            await until(lambda: client.interrupt_calls == 1)
            await asyncio.sleep(0.03)
            resp = await _post(app, ["flag-a"])
            assert await quiesce == "abandoned"
        assert resp.json() == {"released": [], "stop_unconfirmed": ["flag-a"]}
        assert is_held(KEY, root=root)
        client._interrupted.set()
    assert not _entries(ledger, "stop")


async def test_a_waiting_stop_sees_a_turn_that_opened_as_the_last_one_closed():
    client = WritingClient(ignores_interrupt=True)
    async with running_chat(client) as session:
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.2):
            stop = asyncio.create_task(session.cancel())
            await until(lambda: client.interrupt_calls == 1)
            session._in_response = False
            session._close_turn()
            session._turn_open = True
            assert await stop == "abandoned"
        session._turn_open = False
        client._interrupted.set()


async def test_a_prompted_turn_keeps_the_abandoned_agent_on_the_stop_list():
    client = WritingClient()
    async with running_chat(client) as session:
        session._abandoned_background["call-7"] = "task-7"
        await session._open_turn("next")
        assert session._abandoned_background == {"call-7": "task-7"}
        session._abandoned_background.clear()
        client._interrupted.set()


async def test_a_child_that_exited_took_its_abandoned_agents_with_it():
    class ExitingClient(WritingClient):
        async def receive_response(self):
            await self._prompts.get()
            yield assistant_message([TextBlock("partial")])
            self.first_yielded.set()
            await self._interrupted.wait()

    client = ExitingClient()
    async with running_chat(client) as session:
        session._abandoned_background["call-7"] = "task-7"
        client._interrupted.set()
        await until(lambda: session._stream_ended)
        assert await session.cancel() == "idle"


async def test_a_second_stop_on_a_gone_pump_does_not_spend_the_bound():
    client = WritingClient(ignores_interrupt=True)
    async with running_chat(client) as session:
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.02):
            assert await session.cancel() == "abandoned"
        session._pump_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await session._pump_task
        session._in_response = True
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 30.0):
            assert await asyncio.wait_for(session.cancel(), timeout=2.0) == "stopped"
        session._in_response = False


async def test_a_stop_with_no_flag_left_to_mark_still_audits_what_it_did(root, ledger):
    _flag(root, "flag-a")
    client = WritingClient(ignores_interrupt=True)
    async with running_chat(client) as session:
        await _with_approval_and_background(session)
        app = _chat_app(root, session)
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.3):
            stop = asyncio.create_task(_post(app, ["flag-a"]))
            await until(lambda: client.interrupt_calls == 1)
            await _post(app, ["flag-a"], answer="continue")
            resp = await stop
        client._interrupted.set()
    assert resp.json() == {"released": [], "stop_unconfirmed": []}
    [entry] = _entries(ledger, "stop_unconfirmed")
    assert "task-7" in entry["detail"]


async def test_a_confirmed_stop_with_no_flag_left_to_release_still_audits_it(root, ledger):
    _flag(root, "flag-a")
    client = WritingClient()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_interrupt():
        entered.set()
        await release.wait()
        client._interrupted.set()

    client.interrupt = slow_interrupt
    async with running_chat(client) as session:
        app = _chat_app(root, session)
        stop = asyncio.create_task(_post(app, ["flag-a"]))
        await asyncio.wait_for(entered.wait(), timeout=2.0)
        await _post(app, ["flag-a"], answer="continue")
        release.set()
        resp = await stop
    assert resp.json() == {"released": [], "stop_unconfirmed": []}
    [entry] = [e for e in _entries(ledger, "stop") if "flag_id=flag-a" in e["detail"]]
    assert entry["subject"] == "wd"


async def test_a_stop_audits_only_the_named_ids_that_were_flags(root, ledger):
    _flag(root, "flag-a")
    client = WritingClient(ignores_interrupt=True)
    async with running_chat(client) as session:
        app = _chat_app(root, session)
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.3):
            stop = asyncio.create_task(_post(app, ["flag-a", "never-a-flag"]))
            await until(lambda: client.interrupt_calls == 1)
            await _post(app, ["flag-a"], answer="continue")
            await stop
        client._interrupted.set()
    [entry] = _entries(ledger, "stop_unconfirmed")
    assert "flag_id=flag-a" in entry["detail"]
    assert entry["subject"] == "wd"


async def _with_approval_and_background(session: OperatorSession) -> None:
    future = asyncio.get_running_loop().create_future()
    session._permissions["req"] = _PendingAsk(future, {}, {"tool_name_raw": "channel_write"}, [])
    session._task_calls["task-7"] = "call-7"
    session._background.add("call-7")


async def test_released_stop_audits_what_it_denied_and_stopped(root, ledger):
    _flag(root, "flag-a")
    client = WritingClient()
    async with running_chat(client) as session:
        await _with_approval_and_background(session)
        # What the CLI's report of the stopped task does to the session.
        client.on_stop_task = session._background.clear
        await _post(_chat_app(root, session), ["flag-a"])
    assert client.stopped_tasks == ["task-7"]
    detail = _entries(ledger, "stop")[0]["detail"]
    assert "channel_write" in detail
    assert "task-7" in detail


async def test_unconfirmed_stop_audits_what_it_denied_and_stopped(root, ledger):
    _flag(root, "flag-a")
    client = WritingClient(ignores_interrupt=True)
    async with running_chat(client) as session:
        await _with_approval_and_background(session)
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.02):
            await _post(_chat_app(root, session), ["flag-a"])
        client._interrupted.set()
    detail = _entries(ledger, "stop_unconfirmed")[0]["detail"]
    assert "channel_write" in detail
    assert "task-7" in detail


async def test_a_stop_on_flags_already_closed_interrupts_nothing(root):
    _flag(root, "flag-a")
    client = WritingClient()
    async with running_chat(client) as session:
        app = _chat_app(root, session)
        await _post(app, ["flag-a"], answer="continue")
        resp = await _post(app, ["flag-a"])
        assert client.interrupt_calls == 0
        client._interrupted.set()
    assert resp.json() == {"released": [], "stop_unconfirmed": []}


async def test_a_failing_stop_is_not_a_stop(root):
    _flag(root, "flag-a")

    class Broken:
        is_active = False

        async def stop_turn(self):
            raise ValueError("boom")

    resp = await _post(_chat_app(root, Broken()), ["flag-a"])
    assert resp.json() == {"released": [], "stop_unconfirmed": ["flag-a"]}
    assert is_held(KEY, root=root)


async def test_a_key_no_session_runs_under_records_and_releases(root, ledger):
    _flag(root, "flag-a")
    resp = await _post(_chat_app(root), ["flag-a"])
    assert resp.json()["released"] == ["flag-a"]
    assert _entries(ledger, "stop")


class HoldsFirstEventClient(WritingClient):
    """A turn that took the prompt and has sent nothing yet, then writes, ignoring interrupts."""

    def __init__(self) -> None:
        super().__init__(ignores_interrupt=True)
        self.send_first_event = asyncio.Event()

    async def receive_response(self):
        await self._prompts.get()
        self.first_yielded.set()
        await self.send_first_event.wait()
        yield assistant_message([TextBlock("partial")])
        while not self._interrupted.is_set():
            self.writes += 1
            await asyncio.sleep(0.001)
        yield result_message()


async def test_a_sequential_stop_on_a_turn_with_no_event_yet_keeps_the_flag_open(root, ledger):
    _flag(root, "flag-a")
    client = HoldsFirstEventClient()
    async with running_chat(client) as session:
        app = _chat_app(root, session)
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.05):
            first = await _post(app, ["flag-a"])
            second = await _post(app, ["flag-a"])
        unconfirmed = {"released": [], "stop_unconfirmed": ["flag-a"]}
        assert first.json() == unconfirmed
        assert second.json() == unconfirmed
        assert client.interrupt_calls == 2, "the repeat interrupts the abandoned turn again"
        assert is_held(KEY, root=root)

        client.send_first_event.set()
        await until(lambda: client.writes > 0)
        assert is_held(KEY, root=root)
        client._interrupted.set()
        await until(lambda: not session._in_response)

        last = await _post(app, ["flag-a"])
    assert last.json() == {"released": ["flag-a"], "stop_unconfirmed": []}
    assert _entries(ledger, "stop")


async def test_two_stops_at_once_on_a_turn_with_no_event_yet_keep_the_flag_open(root, ledger):
    _flag(root, "flag-a")
    client = HoldsFirstEventClient()
    async with running_chat(client) as session:
        app = _chat_app(root, session)
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.1):
            first = asyncio.create_task(_post(app, ["flag-a"]))
            await until(lambda: client.interrupt_calls == 1)
            second = asyncio.create_task(_post(app, ["flag-a"]))
            answers = [(await first).json(), (await second).json()]
        assert answers == [{"released": [], "stop_unconfirmed": ["flag-a"]}] * 2
        assert is_held(KEY, root=root)
        client.send_first_event.set()
        client._interrupted.set()
    assert not _entries(ledger, "stop")


async def test_chat_stop_then_watchdog_stop_on_a_turn_with_no_event_yet_keeps_the_flag_open(
    root, ledger
):
    _flag(root, "flag-a")
    client = HoldsFirstEventClient()
    async with running_chat(client) as session:
        app = _chat_app(root, session)
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.05):
            assert await session.spawn_quiesce() == "abandoned"
            resp = await _post(app, ["flag-a"])
        assert resp.json() == {"released": [], "stop_unconfirmed": ["flag-a"]}
        assert is_held(KEY, root=root)
        client.send_first_event.set()
        client._interrupted.set()
    assert not _entries(ledger, "stop")


async def test_an_abandoned_turn_stays_abandoned_until_the_stream_shows_its_end():
    client = HoldsFirstEventClient()
    async with running_chat(client) as session:
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.02):
            assert await session.cancel() == "abandoned"
            assert await session.cancel() == "abandoned"
        client.send_first_event.set()
        client._interrupted.set()
        await until(lambda: not session._unanswered_prompts)
        assert await session.cancel() == "idle"


class SteppedClient(WritingClient):
    """Answers each prompt in order, and only as far as the test lets it; ignores interrupts."""

    def __init__(self) -> None:
        super().__init__(ignores_interrupt=True)
        self.begin = asyncio.Semaphore(0)
        self.finish = asyncio.Semaphore(0)

    async def receive_response(self):
        prompt = await self._prompts.get()
        self.first_yielded.set()
        await self.begin.acquire()
        yield assistant_message([TextBlock(f"answer to {prompt}")])
        await self.finish.acquire()
        yield result_message()

    def answer_one(self) -> None:
        self.begin.release()
        self.finish.release()


async def test_a_stop_as_the_abandoned_turns_answer_arrives_holds_for_the_prompt_behind_it(
    root, ledger
):
    client = SteppedClient()
    async with running_chat(client) as session:
        app = _chat_app(root, session)
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.02):
            assert await session.cancel() == "abandoned"
        await until(lambda: session._active_token is None)
        consumer, frames = start_turn(session, "b", timeout_s=30.0)
        await until(lambda: session._pending_turn)
        _flag(root, "flag-b")

        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.3):
            stop = asyncio.create_task(_post(app, ["flag-b"]))
            await until(lambda: client.interrupt_calls == 2)
            client.answer_one()
            resp = await stop
        assert resp.json() == {"released": [], "stop_unconfirmed": ["flag-b"]}
        assert is_held(KEY, root=root)
        assert not _entries(ledger, "stop")
        await asyncio.wait_for(consumer, timeout=2.0)
        assert frames[-1]["error_type"] == "TurnAbandoned"

        client.answer_one()
        await until(lambda: not session._unanswered_prompts)
        last = await _post(app, ["flag-b"])
    assert last.json() == {"released": ["flag-b"], "stop_unconfirmed": []}


async def test_a_prompt_behind_an_abandoned_turn_is_not_ended_by_its_answer():
    client = SteppedClient()
    async with running_chat(client) as session:
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.02):
            assert await session.cancel() == "abandoned"
        await until(lambda: session._active_token is None)
        consumer, frames = start_turn(session, "b", timeout_s=30.0)
        await until(lambda: session._pending_turn)
        client.answer_one()
        await until(lambda: session._turn_open)
        assert not consumer.done()
        assert not any(f.get("type") == "result" for f in frames)
        client.answer_one()
        await asyncio.wait_for(consumer, timeout=2.0)
        assert [f["content"] for f in frames if f.get("type") == "text"] == ["answer to b"]


async def test_abandoning_a_pending_turn_keeps_its_prompt_owed():
    client = SteppedClient()
    async with running_chat(client) as session:
        client.begin.release()
        await until(lambda: session._in_response)
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.02):
            assert await session.cancel() == "abandoned"
            await until(lambda: session._active_token is None)
            consumer, _frames = start_turn(session, "b", timeout_s=30.0)
            await until(lambda: session._pending_turn)
            assert await session.cancel() == "abandoned"
            assert not session._pending_turn

            client.finish.release()
            await until(lambda: not session._in_response)
            assert await session.cancel() == "abandoned", "the queued prompt is still to come"

            client.answer_one()
            await until(lambda: not session._unanswered_prompts)
            assert await session.cancel() == "idle"
        await asyncio.wait_for(consumer, timeout=2.0)


async def test_a_response_of_the_clis_own_answers_no_prompt():
    client = SteppedClient()
    async with running_chat(client) as session:
        client.answer_one()
        await until(lambda: not session._unanswered_prompts)
        client._prompts.put_nowait("its own")
        client.begin.release()
        await until(lambda: session._in_response)

        await until(lambda: session._active_token is None)
        consumer, _frames = start_turn(session, "b", timeout_s=30.0)
        await until(lambda: session._pending_turn)
        client.finish.release()
        await until(lambda: session._turn_open)
        assert session._unanswered_prompts == 1

        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.02):
            assert await session.cancel() == "abandoned"
            assert await session.cancel() == "abandoned"
        await asyncio.wait_for(consumer, timeout=2.0)


async def test_a_failed_submit_owes_no_answer():
    client = SteppedClient()
    async with running_chat(client) as session:
        client.answer_one()
        await until(lambda: not session._unanswered_prompts)

        async def refuses(_prompt):
            raise ConnectionError("gone")

        with patch.object(client, "query", refuses), pytest.raises(ConnectionError):
            await session.send_prompt("b")
        assert await session.cancel() == "idle"


async def test_an_abandon_after_the_last_agent_reported_still_owes_the_follow_up():
    client = HoldsFirstEventClient()
    async with running_chat(client) as session:
        session._idle_wait = True
        with patch(_OS + "_BACKGROUND_DRAIN_S", 0.02), patch(_OS + "_QUIESCE_TIMEOUT_S", 0.02):
            assert await session.cancel() == "abandoned"
            assert await session.cancel() == "abandoned", "the follow-up is still to come"
        client.send_first_event.set()
        client._interrupted.set()
        await until(lambda: not session._unanswered_prompts)
        assert await session.cancel() == "idle"


async def test_an_abandon_with_an_agent_open_owes_no_answer_until_it_reports():
    client = HoldsFirstEventClient()
    async with running_chat(client) as session:
        session._idle_wait = True
        session._task_calls["task-7"] = "call-7"
        session._background.add("call-7")
        with patch(_OS + "_BACKGROUND_DRAIN_S", 0.02):
            assert await session.cancel() == "abandoned"
        assert not session._unanswered_prompts
        assert session._abandoned_background == {"call-7": "task-7"}


async def test_a_stream_that_ends_clears_the_abandoned_turn():
    class ClosingClient(HoldsFirstEventClient):
        async def receive_response(self):
            await self._prompts.get()
            self.first_yielded.set()
            await self.send_first_event.wait()
            return
            yield  # makes this a generator, one that closes with no event

    client = ClosingClient()
    async with running_chat(client) as session:
        with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.02):
            assert await session.cancel() == "abandoned"
        client.send_first_event.set()
        await until(lambda: session._stream_ended)
        assert await session.cancel() == "idle"


async def test_a_stream_that_fails_keeps_the_abandoned_agents_on_the_list():
    class FailingClient(WritingClient):
        async def receive_response(self):
            await self._prompts.get()
            yield assistant_message([TextBlock("partial")])
            self.first_yielded.set()
            await self._interrupted.wait()
            raise RuntimeError("stream failed")

    client = FailingClient()
    async with running_chat(client) as session:
        session._abandoned_background["call-7"] = "task-7"
        client._interrupted.set()
        await until(lambda: session._stream_ended)
        assert session.process_exited is not True
        assert await session.cancel() == "abandoned"


async def test_simple_stop_on_a_fresh_flag_with_a_command_running_keeps_it_open(root, ledger):
    _flag(root, "flag-a")
    client = WritingClient()
    with sdk_seam(client):
        session = OperatorSession(cwd="/tmp")
        await session.start()
        try:
            session.started_commands = lambda: [SLOW_LOOP]
            resp = await _post(_chat_app(root, session), ["flag-a"])
        finally:
            await session.stop()
    assert resp.json() == {"released": [], "stop_unconfirmed": ["flag-a"]}
    assert client.interrupt_calls == 0
    [flag] = open_flags(KEY, root=root)
    assert flag.started == ("write_loop.py",)
    assert not _entries(ledger, "stop")


async def test_simple_stop_on_a_fresh_flag_with_an_unreadable_process_table_keeps_it_open(root):
    _flag(root, "flag-a")
    client = WritingClient()

    def unreadable():
        raise ProcessTableUnreadable("no table")

    with sdk_seam(client):
        session = OperatorSession(cwd="/tmp")
        await session.start()
        try:
            session.started_commands = unreadable
            resp = await _post(_chat_app(root, session), ["flag-a"])
        finally:
            await session.stop()
    assert resp.json()["stop_unconfirmed"] == ["flag-a"]
    assert is_held(KEY, root=root)


async def test_a_second_stop_during_the_first_ones_read_also_reads(root, ledger):
    _flag(root, "flag-a")
    client = WritingClient()
    both_reading = threading.Barrier(2, timeout=5.0)

    def started():
        both_reading.wait()
        return [SLOW_LOOP]

    with sdk_seam(client):
        session = OperatorSession(cwd="/tmp")
        await session.start()
        try:
            session.started_commands = started
            app = _chat_app(root, session)
            answers = await asyncio.gather(_post(app, ["flag-a"]), _post(app, ["flag-a"]))
        finally:
            await session.stop()
    assert [a.json() for a in answers] == [{"released": [], "stop_unconfirmed": ["flag-a"]}] * 2
    assert not _entries(ledger, "stop")


# ---- Expert mode ----


async def test_pty_stop_escapes_the_writing_turn_then_releases(root, ledger):
    _flag(root, "flag-a")
    app, clock = _pty_app(root)
    pty = CountingPty(on_escape=lambda: set_store(app, "idle", time.time()))
    pool_pty(app, pty)
    set_store(app, "busy", time.time() - 1)
    writes = {"n": 0}

    def write_while_running(_count):
        writes["n"] += pty.turn_running

    clock.on_sleep.append(write_while_running)
    resp = await _post(app, ["flag-a"])

    assert resp.json() == {"released": ["flag-a"], "stop_unconfirmed": []}
    assert not is_held(KEY, root=root)
    assert pty.writes == [ESC]
    assert pty.terminates == 0
    assert writes["n"] == 0
    assert _entries(ledger, "stop")


async def test_pty_turn_that_ignores_the_interrupt_stays_alive_and_flagged(root, ledger):
    _flag(root, "flag-a")
    app, _clock = _pty_app(root)
    pty = pool_pty(app)
    set_store(app, "busy", time.time())

    resp = await _post(app, ["flag-a"])

    assert resp.json() == {"released": [], "stop_unconfirmed": ["flag-a"]}
    assert pty.writes == [ESC]
    assert pty.terminates == 0
    assert pty.is_alive
    assert is_held(KEY, root=root)
    assert _entries(ledger, "stop_unconfirmed")
    assert [f["flag_id"] for f in _frames(app, "watchdog_flag")] == ["flag-a"]


async def test_pty_without_the_turn_hook_is_unconfirmed_and_sends_nothing(root, ledger):
    _flag(root, "flag-a")
    app, _clock = _pty_app(root, hook=False)
    pty = pool_pty(app)

    resp = await _post(app, ["flag-a"])

    assert resp.json() == {"released": [], "stop_unconfirmed": ["flag-a"]}
    assert pty.writes == []
    assert pty.terminates == 0
    assert _entries(ledger, "stop_unconfirmed")


async def test_pty_started_command_after_the_interrupt_keeps_the_flag_open_named(root, ledger):
    _flag(root, "flag-a")
    app, _clock = _pty_app(root)
    pty = CountingPty(on_escape=lambda: set_store(app, "idle", time.time()))
    pool_pty(app, pty)
    pty.started = [SLOW_LOOP]
    set_store(app, "busy", time.time() - 1)

    resp = await _post(app, ["flag-a"])

    assert resp.json() == {"released": [], "stop_unconfirmed": ["flag-a"]}
    assert pty.writes == [ESC], "the Stop has no started-commands refusal before Escape"
    [flag] = open_flags(KEY, root=root)
    assert flag.started == ("write_loop.py",)
    assert "write_loop.py" in _entries(ledger, "stop_unconfirmed")[0]["detail"]


async def test_pty_unreadable_process_table_keeps_the_flag_open(root):
    _flag(root, "flag-a")
    app, _clock = _pty_app(root)
    pty = CountingPty(on_escape=lambda: set_store(app, "idle", time.time()))
    pool_pty(app, pty)
    pty.started = None
    set_store(app, "busy", time.time() - 1)

    resp = await _post(app, ["flag-a"])

    assert resp.json()["stop_unconfirmed"] == ["flag-a"]
    assert is_held(KEY, root=root)


async def test_pty_continue_while_stop_waits_is_not_re_marked(root):
    _flag(root, "flag-a")
    _flag(root, "flag-b")
    app, clock = _pty_app(root)
    pty = pool_pty(app)
    set_store(app, "busy", time.time())

    with patch.object(watchdog_routes, "INTERRUPT_GRACE_S", 1e6):
        stop = asyncio.create_task(_post(app, ["flag-a", "flag-b"]))
        await until(lambda: pty.writes == [ESC])
        await _post(app, ["flag-a"], answer="continue")
        clock.now += 2e6
        resp = await stop

    assert resp.json() == {"released": [], "stop_unconfirmed": ["flag-b"]}
    [flag] = open_flags(KEY, root=root)
    assert flag.flag_id == "flag-b"
    assert [f["flag_id"] for f in _frames(app, "watchdog_flag")] == ["flag-b"]


async def test_pty_idle_stop_records_and_releases_sending_nothing(root, ledger):
    _flag(root, "flag-a")
    app, _clock = _pty_app(root)
    pty = pool_pty(app)
    set_store(app, "idle", time.time())

    resp = await _post(app, ["flag-a"])

    assert resp.json()["released"] == ["flag-a"]
    assert pty.writes == []
    assert _entries(ledger, "stop")


async def test_pty_stop_on_a_fresh_flag_with_a_command_running_keeps_it_open(root, ledger):
    _flag(root, "flag-a")
    app, _clock = _pty_app(root)
    pty = pool_pty(app)
    pty.started = [SLOW_LOOP]
    set_store(app, "idle", time.time())

    resp = await _post(app, ["flag-a"])

    assert resp.json() == {"released": [], "stop_unconfirmed": ["flag-a"]}
    assert pty.writes == []
    [flag] = open_flags(KEY, root=root)
    assert flag.started == ("write_loop.py",)
    assert not _entries(ledger, "stop")


async def test_pty_stop_on_a_fresh_flag_with_an_unreadable_process_table_keeps_it_open(root):
    _flag(root, "flag-a")
    app, _clock = _pty_app(root)
    pty = pool_pty(app)
    pty.started = None
    set_store(app, "idle", time.time())

    resp = await _post(app, ["flag-a"])

    assert resp.json()["stop_unconfirmed"] == ["flag-a"]
    assert is_held(KEY, root=root)


async def test_pty_second_stop_stays_open_until_the_started_command_ends(root, ledger):
    _flag(root, "flag-a")
    app, _clock = _pty_app(root)
    pty = CountingPty(on_escape=lambda: set_store(app, "idle", time.time()))
    pool_pty(app, pty)
    pty.started = [SLOW_LOOP]
    set_store(app, "busy", time.time() - 1)
    await _post(app, ["flag-a"])

    again = await _post(app, ["flag-a"])

    assert again.json() == {"released": [], "stop_unconfirmed": ["flag-a"]}
    assert is_held(KEY, root=root)
    assert pty.writes == [ESC], "no turn is running, so the repeat sends nothing"
    assert len(_entries(ledger, "stop_unconfirmed")) == 2
    assert not _entries(ledger, "stop")

    pty.started = []
    last = await _post(app, ["flag-a"])
    assert last.json() == {"released": ["flag-a"], "stop_unconfirmed": []}
    assert not is_held(KEY, root=root)
    assert "write_loop.py" not in _entries(ledger, "stop")[0]["detail"]


async def test_pty_second_stop_with_an_unreadable_process_table_stays_open(root):
    _flag(root, "flag-a")
    app, _clock = _pty_app(root)
    pty = CountingPty(on_escape=lambda: set_store(app, "idle", time.time()))
    pool_pty(app, pty)
    pty.started = None
    set_store(app, "busy", time.time() - 1)
    await _post(app, ["flag-a"])

    again = await _post(app, ["flag-a"])

    assert again.json()["stop_unconfirmed"] == ["flag-a"]


async def test_pty_two_stops_at_once_send_one_escape(root):
    _flag(root, "flag-a")
    app, clock = _pty_app(root)
    pty = pool_pty(app)
    set_store(app, "busy", time.time())

    with patch.object(watchdog_routes, "INTERRUPT_GRACE_S", 1e6):
        first = asyncio.create_task(_post(app, ["flag-a"]))
        await until(lambda: pty.writes == [ESC])
        second = asyncio.create_task(_post(app, ["flag-a"]))
        await asyncio.sleep(0.01)
        set_store(app, "idle", time.time())
        clock.now += 1
        await asyncio.gather(first, second)

    assert pty.writes == [ESC]
    assert not app.state.handoff.interrupts


async def test_pty_exiting_under_the_stop_releases(root):
    _flag(root, "flag-a")
    app, _clock = _pty_app(root)
    pty = pool_pty(app)
    pty._alive = False
    set_store(app, "busy", time.time())

    resp = await _post(app, ["flag-a"])

    assert resp.json()["released"] == ["flag-a"]
    assert pty.writes == []


# ---- interrupt_pty_turn ----


async def test_interrupt_pty_turn_words():
    idle_app = make_app()
    pool_pty(idle_app)
    set_store(idle_app, "idle", time.time())
    assert await interrupt_pty_turn(idle_app, KEY, grace=0.5) == "idle"

    hookless = make_app(hook=False)
    pty = pool_pty(hookless)
    assert await interrupt_pty_turn(hookless, KEY, grace=0.5) == "unconfirmed"
    assert pty.writes == []

    missing = make_app()
    assert await interrupt_pty_turn(missing, KEY, grace=0.5) == "exited"

    busy = make_app()
    pty = pool_pty(busy)
    set_store(busy, "busy", time.time())
    assert await interrupt_pty_turn(busy, KEY, grace=0.5) == "unconfirmed"
    assert pty.writes == [ESC]
    assert pty.terminates == 0

    stopping = make_app()
    pty = CountingPty(on_escape=lambda: set_store(stopping, "idle", time.time()))
    pool_pty(stopping, pty)
    set_store(stopping, "busy", time.time() - 1)
    assert await interrupt_pty_turn(stopping, KEY, grace=0.5) == "interrupted"
    assert pty.writes == [ESC]
