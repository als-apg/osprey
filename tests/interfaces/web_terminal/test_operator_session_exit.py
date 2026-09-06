"""The child's death, observable after the session has been torn down.

Tearing a chat session down closes its SDK client, and closing the client drops
the transport that holds the child's process handle. A surface handing a
conversation over has to wait for the outgoing child to be *gone*, not merely
asked to leave, so the handle is kept one step before the close and read back
through ``OperatorSession.process_exited``; ``terminate`` hands the torn-down
session out so a caller has something to read it on.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from unittest.mock import MagicMock, patch

import pytest

from osprey.interfaces.web_terminal.chat_session_pool import ChatSessionPool
from osprey.interfaces.web_terminal.operator_session import OperatorRegistry, OperatorSession

_SEAM = "osprey.interfaces.web_terminal.operator_session"


# ---------------------------------------------------------------------------
# Fakes — an SDK client whose transport exposes a process handle
# ---------------------------------------------------------------------------


class _FakeProcess:
    """The child handle the SDK transport holds (``returncode`` is the signal)."""

    def __init__(self, returncode: int | None = None):
        self.returncode = returncode


class _FakeTransport:
    def __init__(self, process: _FakeProcess):
        self._process = process


class _FakeSDKClient:
    """Stand-in for ``ClaudeSDKClient`` with a controllable transport.

    ``transport=None`` models a client that never exposed one — the shape the
    property must answer ``None`` for rather than ``False``.
    """

    def __init__(self, transport: _FakeTransport | None = None):
        if transport is not None:
            self._transport = transport
        self.exited = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.exited = True
        return False


@contextlib.contextmanager
def _sdk_seam(client):
    """Patch the SDK seam so ``start()`` connects *client* and nothing else."""
    with (
        patch(f"{_SEAM}.CLAUDE_SDK_AVAILABLE", True),
        patch(f"{_SEAM}.ClaudeAgentOptions", side_effect=lambda **kw: MagicMock()),
        patch(f"{_SEAM}.ClaudeSDKClient", return_value=client),
        patch(f"{_SEAM}.validate_project_directory", return_value=[]),
        patch(f"{_SEAM}.build_system_prompt", return_value={"type": "preset"}),
        patch(f"{_SEAM}.get_facility_timezone", return_value=None),
    ):
        yield


async def _started_session(client) -> OperatorSession:
    session = OperatorSession(cwd="/tmp")
    with _sdk_seam(client):
        await session.start()
    return session


# ---------------------------------------------------------------------------
# OperatorSession.process_exited
# ---------------------------------------------------------------------------


class TestProcessExited:
    """Three answers, and the difference between two of them matters."""

    @pytest.mark.asyncio
    async def test_a_child_still_running_after_the_close_reads_false(self):
        """Closing the client is a request, not a death certificate.

        The transport is gone but the process it held has no return code, so
        the child is still there — the case a handover must keep waiting on.
        """
        client = _FakeSDKClient(_FakeTransport(_FakeProcess(returncode=None)))
        session = await _started_session(client)

        await session.stop()

        assert client.exited is True
        assert session.process_exited is False

    @pytest.mark.asyncio
    async def test_a_child_that_exited_reads_true(self):
        client = _FakeSDKClient(_FakeTransport(_FakeProcess(returncode=0)))
        session = await _started_session(client)

        await session.stop()

        assert session.process_exited is True

    @pytest.mark.asyncio
    async def test_a_non_zero_exit_still_reads_true(self):
        """The question is whether the child is gone, not how it went."""
        client = _FakeSDKClient(_FakeTransport(_FakeProcess(returncode=137)))
        session = await _started_session(client)

        await session.stop()

        assert session.process_exited is True

    @pytest.mark.asyncio
    async def test_a_client_with_no_transport_reads_none(self):
        """Nothing to observe is not the same as *not exited*."""
        client = _FakeSDKClient(transport=None)
        session = await _started_session(client)

        await session.stop()

        assert session.process_exited is None

    def test_a_session_that_never_started_reads_none(self):
        assert OperatorSession(cwd="/tmp").process_exited is None

    @pytest.mark.asyncio
    async def test_the_answer_survives_a_second_stop(self):
        """``stop`` is called twice on ordinary paths (teardown, then a pool
        drain). The second one finds no client and must not erase what the
        first one captured."""
        client = _FakeSDKClient(_FakeTransport(_FakeProcess(returncode=0)))
        session = await _started_session(client)

        await session.stop()
        await session.stop()

        assert session.process_exited is True

    @pytest.mark.asyncio
    async def test_the_handle_is_read_before_the_client_is_closed(self):
        """Captured from the live transport, not from whatever survives close.

        The real client drops ``_transport`` on ``__aexit__``; a fake that does
        the same would answer ``None`` if the capture happened after.
        """
        process = _FakeProcess(returncode=None)
        client = _FakeSDKClient(_FakeTransport(process))

        async def _drop_transport(*exc):
            client._transport = None
            return False

        client.__aexit__ = _drop_transport  # type: ignore[method-assign]
        session = await _started_session(client)

        await session.stop()

        process.returncode = 0
        assert session.process_exited is True


# ---------------------------------------------------------------------------
# terminate hands the torn-down session back
# ---------------------------------------------------------------------------


class _FakeChatSession:
    """Lightweight OperatorSession double for the pool's terminate."""

    def __init__(self, cwd: str = "/tmp", env=None, session_key=None):
        self.cwd = cwd
        self.env = env
        self.session_key = session_key
        self.is_active = True
        self.is_busy = False
        self.last_activity = time.monotonic()
        self.teardown_calls = 0
        self.process_exited: bool | None = None

    async def start(self, *, resume_id=None):
        return None

    async def teardown(self):
        self.teardown_calls += 1
        self.is_active = False
        self.process_exited = True


def _pool() -> tuple[ChatSessionPool, list[_FakeChatSession]]:
    created: list[_FakeChatSession] = []

    def factory(cwd, env, session_key=None):
        session = _FakeChatSession(cwd=cwd, env=env, session_key=session_key)
        created.append(session)
        return session

    return ChatSessionPool(factory=factory), created


class TestTerminateReturnsTheSession:
    @pytest.mark.asyncio
    async def test_the_pool_returns_the_session_it_tore_down(self):
        pool, created = _pool()
        session, _ = await pool.get_or_create("a", cwd="/tmp")

        returned = await pool.terminate("a")

        assert returned is session
        assert created[0].teardown_calls == 1
        # And the caller can read the child's fate off what it got back.
        assert returned.process_exited is True

    @pytest.mark.asyncio
    async def test_an_unknown_key_returns_none(self):
        pool, _ = _pool()
        assert await pool.terminate("nobody") is None

    @pytest.mark.asyncio
    async def test_a_second_terminate_returns_none(self):
        """Idempotent, and the second call says there was nothing to pop."""
        pool, _ = _pool()
        await pool.get_or_create("a", cwd="/tmp")

        assert await pool.terminate("a") is not None
        assert await pool.terminate("a") is None

    @pytest.mark.asyncio
    async def test_a_creation_still_starting_is_superseded_and_returns_none(self):
        """A terminate racing a first prompt has nothing to pop, and says so."""
        started = asyncio.Event()

        class _SlowSession(_FakeChatSession):
            async def start(self, *, resume_id=None):
                started.set()
                await asyncio.sleep(0.05)

        def factory(cwd, env, session_key=None):
            return _SlowSession(cwd=cwd, env=env, session_key=session_key)

        pool = ChatSessionPool(factory=factory)
        creation = asyncio.create_task(pool.get_or_create("a", cwd="/tmp"))
        await asyncio.wait_for(started.wait(), timeout=1.0)

        assert await pool.terminate("a") is None

        with pytest.raises(Exception, match="terminated while it was starting"):
            await creation

    @pytest.mark.asyncio
    async def test_the_registry_facade_passes_the_session_through(self):
        registry = OperatorRegistry()
        with patch(f"{_SEAM}.OperatorSession", side_effect=_FakeChatSession):
            session, _ = await registry.get_or_create_chat_session("a", cwd="/tmp")

        returned = await registry.terminate_chat_session("a")

        assert returned is session
        assert returned.process_exited is True
        assert await registry.terminate_chat_session("a") is None
