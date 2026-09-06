"""The transcript a chat child is launched on, and how the pool compares it.

The chat pool is keyed on the session key, and that key never changes — the
same key answers for both views, before and after a flip. The *conversation*
under it can move: a ``/clear`` in the other view starts a new transcript, and
the child that comes back has to be launched on that one.

So the transcript joins the environment as part of a child's launch identity.
Both are fixed when the child is spawned and neither can be amended afterwards,
which makes them the same kind of fact and gives them the same rule: the same
launch identity reuses the live child, a different one tears it down and builds
a new one. The environment half is pinned in ``test_terminate_respawn.py``;
this file pins the transcript half, and the binding that carries the session
key from the pool's key into the child.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from osprey.interfaces.web_terminal.chat_session_pool import (
    ChatSessionPool,
    ChatSessionTerminatedError,
)
from osprey.interfaces.web_terminal.operator_session import OperatorRegistry

_SEAM = "osprey.interfaces.web_terminal.operator_session.OperatorSession"


class _FakeChatSession:
    """Lightweight OperatorSession double recording how it was launched.

    Mirrors the double in ``test_terminate_respawn.py`` — the surface the pool
    drives — and additionally keeps the two launch facts this file is about:
    the ``session_key`` it was built under and the ``resume_id`` it was started
    on.
    """

    def __init__(self, cwd="/tmp", env=None, session_key=None):
        self.cwd = cwd
        self.env = env
        self.session_key = session_key
        self.resume_id = None
        self.is_active = True
        self.is_busy = False
        self.last_activity = time.monotonic()
        self.start_calls = 0
        self.stop_calls = 0
        self.start_delay = 0.0
        self.started = asyncio.Event()

    async def start(self, *, resume_id=None):
        self.resume_id = resume_id
        self.started.set()
        if self.start_delay:
            await asyncio.sleep(self.start_delay)
        self.start_calls += 1

    async def teardown(self):
        self.stop_calls += 1
        self.is_active = False


def _pool(start_delay: float = 0.0, **kwargs) -> tuple[ChatSessionPool, list[_FakeChatSession]]:
    created: list[_FakeChatSession] = []

    def factory(cwd, env, session_key):
        session = _FakeChatSession(cwd=cwd, env=env, session_key=session_key)
        session.start_delay = start_delay
        created.append(session)
        return session

    return ChatSessionPool(factory=factory, **kwargs), created


async def _created_first(created, timeout: float = 1.0):
    """Wait until the factory has built its first session and entered ``start()``."""
    deadline = time.monotonic() + timeout
    while not created:
        assert time.monotonic() < deadline, "factory never ran"
        await asyncio.sleep(0)
    await asyncio.wait_for(created[0].started.wait(), timeout=timeout)


class TestTheResumeIdReachesTheChild:
    @pytest.mark.asyncio
    async def test_the_transcript_is_named_at_start_not_at_construction(self):
        """A child is built under the key and started on the transcript.

        The two are different identities and travel separately: the key is what
        the session is pooled, audited and given telemetry under, the
        transcript is only which conversation this launch continues.
        """
        pool, created = _pool()

        session, was_reused = await pool.get_or_create("K", "/tmp", resume_id="T1")

        assert was_reused is False
        assert session.session_key == "K"
        assert session.resume_id == "T1"
        assert created == [session]

    @pytest.mark.asyncio
    async def test_without_a_transcript_the_child_starts_its_own(self):
        """No resume id is the first-launch shape, not an error."""
        pool, _created = _pool()

        session, _ = await pool.get_or_create("K", "/tmp")

        assert session.session_key == "K"
        assert session.resume_id is None


class TestTheTranscriptJoinsTheReuseCheck:
    """The transcript half of the launch identity, mirroring the env half.

    ``test_terminate_respawn.TestChatPoolEnvFingerprint`` pins the
    environment; the transcript is compared in the same place, for the same
    reason, and these are the same two cases.
    """

    @pytest.mark.asyncio
    async def test_a_changed_resume_id_rebuilds_instead_of_reusing(self):
        """The core of it: a different transcript means a different child.

        This is what makes a ``/clear`` in the other view reach this one. The
        key is unchanged, so nothing else in the pool would notice.
        """
        pool, created = _pool()

        first, _ = await pool.get_or_create("K", "/tmp", resume_id="T1")
        second, was_reused = await pool.get_or_create("K", "/tmp", resume_id="T2")

        assert second is not first
        assert was_reused is False
        assert second.resume_id == "T2"
        assert first.stop_calls == 1
        assert len(created) == 2

    @pytest.mark.asyncio
    async def test_an_unchanged_resume_id_still_reuses_the_live_session(self):
        """The liveness half: a second prompt must not kill the conversation.

        The route hands the key's current transcript over on every turn, so a
        comparison that fired on an unchanged id would restart the agent under
        the operator mid-conversation — a worse failure than the one it
        prevents.
        """
        pool, created = _pool()

        first, _ = await pool.get_or_create("K", "/tmp", resume_id="T1")
        second, was_reused = await pool.get_or_create("K", "/tmp", resume_id="T1")

        assert second is first
        assert was_reused is True
        assert first.stop_calls == 0
        assert first.start_calls == 1
        assert len(created) == 1

    @pytest.mark.asyncio
    async def test_dropping_the_resume_id_rebuilds_as_well(self):
        """An absent transcript is a launch identity of its own, not a wildcard.

        A caller that has stopped naming a transcript is asking for a child
        that starts a conversation of its own, which the live one is not.
        """
        pool, created = _pool()

        first, _ = await pool.get_or_create("K", "/tmp", resume_id="T1")
        second, was_reused = await pool.get_or_create("K", "/tmp")

        assert second is not first
        assert was_reused is False
        assert second.resume_id is None
        assert first.stop_calls == 1
        assert len(created) == 2

    @pytest.mark.asyncio
    async def test_a_creation_on_a_stale_transcript_is_overtaken(self):
        """A caller must not be joined to a creation it would have replaced.

        The in-flight child is being launched on the transcript the second
        caller already knows is gone, so joining it would hand back exactly the
        child the change was meant to replace.
        """
        pool, created = _pool(start_delay=0.05)
        creation = asyncio.create_task(pool.get_or_create("K", "/tmp", resume_id="T1"))
        await _created_first(created)

        second, was_reused = await pool.get_or_create("K", "/tmp", resume_id="T2")

        with pytest.raises(ChatSessionTerminatedError):
            await creation
        assert was_reused is False
        assert second is created[1]
        assert second.resume_id == "T2"
        assert created[0].stop_calls == 1
        assert pool.get("K") is second


class TestTheRegistryBinding:
    @pytest.mark.asyncio
    async def test_the_facade_carries_the_key_and_the_transcript_through(self):
        """The pool's key IS the child's session key, and the resume id rides along.

        The factory bound in ``OperatorRegistry.__init__`` is the only place
        that turns a pool key into a child identity; a binding that minted an
        id of its own would split one conversation's audit and telemetry across
        two.
        """
        registry = OperatorRegistry()

        with patch(_SEAM, _FakeChatSession):
            session, was_reused = await registry.get_or_create_chat_session(
                "K", cwd="/tmp", resume_id="T1"
            )

        assert was_reused is False
        assert session.session_key == "K"
        assert session.resume_id == "T1"
        assert registry.get_chat_session("K") is session

    @pytest.mark.asyncio
    async def test_the_facade_defaults_to_no_transcript(self):
        """Omitting the transcript reaches the child as omitted, not as a blank."""
        registry = OperatorRegistry()

        with patch(_SEAM, _FakeChatSession):
            session, _ = await registry.get_or_create_chat_session("K", cwd="/tmp")

        assert session.resume_id is None
