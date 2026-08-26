"""Terminate-for-respawn goes through the pool that OWNS the live child.

A posture is only ever carried by a child process's environment, so recording
one is half the job: the child running under the *old* posture has to die and
come back. Which pool can do that depends on the topology the key names —

* a **PTY** key is owned by ``app.state.pty_registry``; and
* a **chat** key is owned by the Simple-mode ``ChatSessionPool`` behind
  ``app.state.operator_registry``.

Before this, ``POST /api/terminal/posture`` blind-called the PTY registry for
every key. For a chat key that pops nothing: the operator got a 200, the store
said ``sandbox``, and the SDK child kept running with writes — the badge lied.
These tests pin the routing, and pin that "terminated" survives the one race
that could undo it (a terminate arriving while the session is still starting).

Harness mirrors ``test_posture_routes.py``: each test builds its own app
through ``create_app`` under a patched ``_load_web_config``, entered as a
``TestClient`` context manager so the lifespan runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.chat_session_pool import (
    ChatSessionPool,
    ChatSessionTerminatedError,
)
from osprey.interfaces.web_terminal.operator_session import (
    POSTURE_SESSION_ENV,
    OperatorRegistry,
)
from osprey.interfaces.web_terminal.routes import chat as chat_routes
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey.interfaces.web_terminal.routes.websocket import PostureRequest

# A Claude session-file stem: the PTY topology's key.
SESSION_A = "aaaaaaaa-1111-2222-3333-444444444444"
# A chat-pool key, minted the way the shipped client mints one
# (``crypto.randomUUID()`` in static/js/chat.js): a bare lowercase UUID.
CHAT_A = "cccccccc-1111-2222-3333-444444444444"
# The operator websocket's per-connection key — outside the posture surface's
# closed grammar, and deliberately so (see TestOperatorSessionsGetNoRuntimeFlip).
OPERATOR_KEY = "operator-dddddddd"


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


@pytest.fixture
def shared_root(tmp_path):
    """Stand in for the deployment's shared agent-data root."""
    root = tmp_path / "shared_agent_data"
    root.mkdir()
    with patch(
        "osprey_connectors.workspace.resolve_shared_data_root",
        return_value=root,
    ):
        yield root


@pytest.fixture
def client(workspace_dir, shared_root):
    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(workspace_dir)},
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as test_client:
            yield test_client


@contextmanager
def known_sessions(*session_ids):
    """Make ``SessionDiscovery`` report *session_ids* as started on disk."""
    with patch(
        "osprey.interfaces.web_terminal.session_discovery.SessionDiscovery.snapshot_session_ids",
        return_value=set(session_ids),
    ):
        yield


def _write_config(tmp_path, *, writes_enabled: bool):
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump({"control_system": {"writes_enabled": writes_enabled}}),
        encoding="utf-8",
    )
    return path


class _ChatRegistryDouble:
    """Operator-registry stand-in whose chat pool is a plain dict.

    Only the two methods the posture route may call on it are real: the
    read-only membership probe and the terminate. Both record, so a test can
    tell "asked" from "acted", and the terminate records what the store held
    at the moment it ran — that is how the store-before-terminate ordering is
    pinned rather than assumed.
    """

    def __init__(self, app=None, live_chats=()):
        self._app = app
        self.chats: dict[str, object] = {chat_id: SimpleNamespace() for chat_id in live_chats}
        self.terminated: list[str] = []
        self.postures_seen: list[str | None] = []

    def get_chat_session(self, chat_id):
        return self.chats.get(chat_id)

    async def terminate_chat_session(self, chat_id):
        self.terminated.append(chat_id)
        store = getattr(self._app.state, "session_postures", {}) if self._app else {}
        self.postures_seen.append(store.get(chat_id))
        self.chats.pop(chat_id, None)

    async def terminate_session_if_owner(self, session_id, owner):
        return None

    async def cleanup_all(self):
        return None


class _DeafChatRegistry(_ChatRegistryDouble):
    """A registry that reports live chats but exposes no terminate at all.

    Stands in for every hand-rolled double in the suite (``test_posture_routes``
    has several). The route must not 500 on one — the *real* registry always
    has the method, and
    ``test_the_shipped_registry_exposes_the_facades_the_route_uses`` is what
    actually keeps the tolerance honest.
    """

    terminate_chat_session = None


def _record_pty_terminations(client) -> list[str]:
    """Record every key handed to ``pty_registry.terminate_session``."""
    registry = client.app.state.pty_registry
    real = registry.terminate_session
    seen: list[str] = []

    def _spy(session_id):
        seen.append(session_id)
        return real(session_id)

    registry.terminate_session = _spy
    return seen


class TestTerminateRoutesToTheOwningPool:
    def test_chat_key_terminates_the_chat_session(self, client):
        """The flip must reach the SDK child, not a PTY that was never there."""
        registry = _ChatRegistryDouble(app=client.app, live_chats=[CHAT_A])
        client.app.state.operator_registry = registry

        with known_sessions():
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": CHAT_A, "posture": "sandbox"},
            )

        assert resp.status_code == 200
        assert registry.terminated == [CHAT_A]

    def test_chat_key_does_not_blind_call_the_pty_registry(self, client):
        """The PTY pool owns nothing under a chat key, so it is not told to act.

        The old code called it for every key and relied on "unknown key pops to
        None". That no-op is what let the chat branch look handled while the
        SDK child kept running; routing by owner is what makes the log line and
        the behaviour agree.
        """
        client.app.state.operator_registry = _ChatRegistryDouble(
            app=client.app, live_chats=[CHAT_A]
        )
        pty_calls = _record_pty_terminations(client)

        with known_sessions():
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": CHAT_A, "posture": "sandbox"},
            )

        assert resp.status_code == 200
        assert pty_calls == []

    def test_pty_key_terminates_the_pty_session(self, client):
        """The PTY topology keeps its own answer, unchanged."""
        registry = _ChatRegistryDouble(app=client.app)
        client.app.state.operator_registry = registry
        pty_registry = client.app.state.pty_registry
        pty_registry.get_or_create_session(SESSION_A, "echo")
        assert pty_registry.get_session(SESSION_A) is not None

        with known_sessions(SESSION_A):
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "sandbox"},
            )

        assert resp.status_code == 200
        assert pty_registry.get_session(SESSION_A) is None
        # The chat pool holds nothing under this key, and is told to terminate
        # anyway — see TestTheChatTerminateIsUngated. For a key it does not
        # hold that is a no-op, and it is the only way to reach a chat that is
        # still inside ``start()``.
        assert registry.terminated == [SESSION_A]

    def test_one_key_live_in_both_pools_terminates_both(self, client):
        """``chat_id`` is caller-chosen, so it can name a live PTY session too.

        Nothing stops an embedder from picking a chat id equal to a Claude
        session UUID that already has a PTY. Terminating only one pool would
        leave a live child running the posture the operator just left, which is
        the exact lie this route exists to prevent — so every owner acts.
        """
        registry = _ChatRegistryDouble(app=client.app, live_chats=[CHAT_A])
        client.app.state.operator_registry = registry
        pty_registry = client.app.state.pty_registry
        pty_registry.get_or_create_session(CHAT_A, "echo")

        with known_sessions(CHAT_A):
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": CHAT_A, "posture": "sandbox"},
            )

        assert resp.status_code == 200
        assert registry.terminated == [CHAT_A]
        assert pty_registry.get_session(CHAT_A) is None

    def test_terminate_sees_the_new_posture_already_stored(self, client, shared_root):
        """Store, persist, *then* terminate — the respawn reads the store back.

        A terminate that ran first would race the reconnect: the fresh child
        could be built from the old store and the operator would have to toggle
        twice for one change.
        """
        registry = _ChatRegistryDouble(app=client.app, live_chats=[CHAT_A])
        client.app.state.operator_registry = registry

        with known_sessions():
            client.post(
                "/api/terminal/posture",
                json={"session_id": CHAT_A, "posture": "sandbox"},
            )

        assert registry.postures_seen == ["sandbox"]
        on_disk = json.loads((shared_root / "session-postures.json").read_text(encoding="utf-8"))
        assert on_disk == {CHAT_A: "sandbox"}

    def test_unaddressable_key_terminates_nothing(self, client):
        """A 409 must not tear down anything on either pool."""
        registry = _ChatRegistryDouble(app=client.app)
        client.app.state.operator_registry = registry
        pty_calls = _record_pty_terminations(client)

        with known_sessions():
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": CHAT_A, "posture": "sandbox"},
            )

        assert resp.status_code == 409
        assert registry.terminated == []
        assert pty_calls == []

    def test_writes_refusal_terminates_nothing(self, client, tmp_path):
        """A 403 leaves the live child alone — it was never granted the flip."""
        registry = _ChatRegistryDouble(app=client.app, live_chats=[CHAT_A])
        client.app.state.operator_registry = registry
        client.app.state.config_path = _write_config(tmp_path, writes_enabled=False)

        with known_sessions():
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": CHAT_A, "posture": "writes"},
            )

        assert resp.status_code == 403
        assert registry.terminated == []

    def test_registry_without_a_chat_terminate_does_not_break_the_route(self, client):
        """Unfamiliar registries answer the gate but cannot be told to act.

        Same tolerance ``_names_a_live_chat_session`` already grants: a registry
        that cannot be asked simply has no chat session to offer. The store is
        still written, so the next spawn under that key carries the posture.
        """
        client.app.state.operator_registry = _DeafChatRegistry(app=client.app, live_chats=[CHAT_A])

        with known_sessions():
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": CHAT_A, "posture": "sandbox"},
            )

        assert resp.status_code == 200
        assert client.app.state.session_postures[CHAT_A] == "sandbox"

    def test_the_shipped_registry_exposes_the_facades_the_route_uses(self):
        """What makes the tolerance above safe rather than a silent hole.

        The route reaches the chat pool through two facades and a rename of
        either is silent in a different way:

        * ``terminate_chat_session`` — every real deployment would take the
          tolerant branch and stop applying chat postures at all;
        * ``get_chat_session`` — :func:`_holds_a_chat_pool_entry` would answer
          ``False`` for every key, so the log line would stop naming the chat
          pool; and
        * ``has_chat_key`` — the addressability probe would fall back to the
          session-map read, which cannot see a creation still inside
          ``start()``, and a flip during a chat's first prompt would go back to
          being a 409 that writes nothing.

        The first two fail silently, so all three are pinned here.
        """
        for name in ("terminate_chat_session", "get_chat_session", "has_chat_key"):
            assert callable(getattr(OperatorRegistry, name, None)), name

    def test_the_addressability_probe_sees_a_creation_still_starting(self, client):
        """The pool answers to a key it has accepted but not yet registered.

        The gate's own unit: ``get_chat_session`` still says "nothing here"
        during ``start()`` — that is what it is for — while ``has_chat_key``
        says the pool will answer to this key. Only the second is a safe basis
        for refusing an operator's flip.
        """
        created: list[_FakeChatSession] = []
        registry = OperatorRegistry()
        client.app.state.operator_registry = registry

        async def _probe_mid_creation():
            with patch(
                "osprey.interfaces.web_terminal.operator_session.OperatorSession",
                _slow_session_class(created),
            ):
                creation = asyncio.create_task(
                    registry.get_or_create_chat_session(CHAT_A, "/tmp", None)
                )
                await created_first(created)
                seen = (
                    registry.get_chat_session(CHAT_A),
                    registry.has_chat_key(CHAT_A),
                    websocket_routes._posture_key_is_addressable(client.app, CHAT_A),
                )
                await creation
                return seen

        with known_sessions():
            pooled, has_key, addressable = asyncio.run(_probe_mid_creation())

        assert pooled is None
        assert has_key is True
        assert addressable is True


class TestOperatorSessionsGetNoRuntimeFlip:
    def test_operator_key_is_refused_by_the_posture_surface(self, client):
        """No runtime flip for ``/ws/operator`` in this phase, by construction.

        The key is minted per connection and nothing carries it across a
        respawn, so there is no reconnect protocol to respawn *into*. The closed
        key grammar refuses it before any store write, which is why every
        operator-session record can honestly say ``posture_source=spawn``.
        """
        registry = _ChatRegistryDouble(app=client.app, live_chats=[OPERATOR_KEY])
        client.app.state.operator_registry = registry
        pty_calls = _record_pty_terminations(client)

        with known_sessions(OPERATOR_KEY):
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": OPERATOR_KEY, "posture": "sandbox"},
            )

        assert resp.status_code == 400
        assert registry.terminated == []
        assert pty_calls == []
        assert OPERATOR_KEY not in getattr(client.app.state, "session_postures", {})


class _FakeChatSession:
    """Lightweight OperatorSession double for pool tests.

    Mirrors ``test_operator_session.FakeChatSession`` — the same surface the
    pool drives (``start``/``is_active``/``is_busy``/``last_activity``/
    ``teardown``), plus ``acquire_turn`` for the tests that go through the real
    ``_acquire_chat_turn`` rather than the pool directly.
    """

    def __init__(self, cwd="/tmp", env=None):
        self.cwd = cwd
        self.env = env
        self.is_active = True
        self.last_activity = time.monotonic()
        self.in_flight = False
        self.start_calls = 0
        self.stop_calls = 0
        self.start_delay = 0.0
        self.turns = 0
        self.started = asyncio.Event()

    async def start(self):
        self.started.set()
        if self.start_delay:
            await asyncio.sleep(self.start_delay)
        self.start_calls += 1

    def acquire_turn(self) -> int:
        self.turns += 1
        return self.turns

    @property
    def is_busy(self) -> bool:
        return self.in_flight

    async def teardown(self):
        self.stop_calls += 1
        self.is_active = False


def _pool(start_delay: float = 0.0, **kwargs) -> tuple[ChatSessionPool, list[_FakeChatSession]]:
    created: list[_FakeChatSession] = []

    def factory(cwd, env):
        session = _FakeChatSession(cwd=cwd, env=env)
        session.start_delay = start_delay
        created.append(session)
        return session

    return ChatSessionPool(factory=factory, **kwargs), created


class TestChatPoolEvictionOnTerminate:
    @pytest.mark.asyncio
    async def test_terminate_evicts_so_the_next_turn_respawns(self):
        """Eviction is the respawn half: the next turn builds a NEW child.

        The posture only reaches the agent through a fresh process env, so a
        terminate that left the entry in place — or left a torn-down session
        the next call could hand back — would apply nothing.
        """
        pool, created = _pool()
        first, _ = await pool.get_or_create("a", "/tmp", {"OSPREY_EXECUTION_MODE": "writes"})

        await pool.terminate("a")
        assert pool.get("a") is None
        assert first.stop_calls == 1

        second, was_reused = await pool.get_or_create(
            "a", "/tmp", {"OSPREY_EXECUTION_MODE": "readonly"}
        )
        assert second is not first
        assert was_reused is False
        assert second.env == {"OSPREY_EXECUTION_MODE": "readonly"}
        assert len(created) == 2

    @pytest.mark.asyncio
    async def test_terminate_during_creation_supersedes_it(self):
        """A terminate racing a first prompt is not undone by that creation.

        The session is not in the map yet, so ``terminate`` has nothing to pop.
        Without a superseded marker the creator would register a child built
        from the pre-flip environment *after* the operator was told 200 OK —
        a live session running the posture they just stepped out of.
        """
        pool, created = _pool(start_delay=0.05)
        creation = asyncio.create_task(pool.get_or_create("a", "/tmp", {"MODE": "old"}))
        await created_first(created)

        await pool.terminate("a")

        with pytest.raises(ChatSessionTerminatedError):
            await creation

        assert pool.get("a") is None
        assert created[0].stop_calls == 1

    @pytest.mark.asyncio
    async def test_a_superseded_creation_leaves_the_key_reusable(self):
        """The pending marker is cleared, so the next prompt starts cleanly."""
        pool, created = _pool(start_delay=0.05)
        creation = asyncio.create_task(pool.get_or_create("a", "/tmp", {"MODE": "old"}))
        await created_first(created)
        await pool.terminate("a")
        with pytest.raises(ChatSessionTerminatedError):
            await creation

        session, was_reused = await pool.get_or_create("a", "/tmp", {"MODE": "new"})

        assert was_reused is False
        assert session is created[1]
        assert session.env == {"MODE": "new"}
        assert pool.get("a") is session

    @pytest.mark.asyncio
    async def test_joiners_of_a_superseded_creation_see_the_refusal(self):
        """A double-submit must not hand one caller a torn-down session."""
        pool, created = _pool(start_delay=0.05)
        creator = asyncio.create_task(pool.get_or_create("a", "/tmp", {"MODE": "old"}))
        await created_first(created)
        joiner = asyncio.create_task(pool.get_or_create("a", "/tmp", {"MODE": "old"}))
        await asyncio.sleep(0)

        await pool.terminate("a")

        with pytest.raises(ChatSessionTerminatedError):
            await creator
        with pytest.raises(ChatSessionTerminatedError):
            await joiner
        assert len(created) == 1

    @pytest.mark.asyncio
    async def test_drain_all_supersedes_in_flight_creations(self):
        """Shutdown must not have a child land in the pool behind it."""
        pool, created = _pool(start_delay=0.05)
        creation = asyncio.create_task(pool.get_or_create("a", "/tmp", None))
        await created_first(created)

        await pool.drain_all()

        with pytest.raises(ChatSessionTerminatedError):
            await creation
        assert pool.get("a") is None
        assert created[0].stop_calls == 1

    @pytest.mark.asyncio
    async def test_an_untouched_creation_still_registers(self):
        """The marker is per-creation: an unterminated key is unaffected."""
        pool, created = _pool(start_delay=0.02)
        creation = asyncio.create_task(pool.get_or_create("a", "/tmp", None))
        other = asyncio.create_task(pool.get_or_create("b", "/tmp", None))
        await asyncio.sleep(0.01)

        await pool.terminate("b")
        with contextlib.suppress(ChatSessionTerminatedError):
            await other

        session, was_reused = await creation
        assert was_reused is False
        assert pool.get("a") is session
        assert session.stop_calls == 0

    @pytest.mark.asyncio
    async def test_terminate_is_idempotent(self):
        """Terminating a key twice tears the child down exactly once."""
        pool, created = _pool()
        await pool.get_or_create("a", "/tmp", None)
        await pool.terminate("a")
        await pool.terminate("a")
        assert pool.get("a") is None
        assert created[0].stop_calls == 1


async def created_first(created, timeout: float = 1.0):
    """Wait until the pool's factory has built its first session and started it.

    Waiting on the session's own ``started`` event (not a sleep) keeps the race
    deterministic: the terminate below has to arrive while ``start()`` is still
    running, which is the whole point of the test.
    """
    deadline = time.monotonic() + timeout
    while not created:
        if time.monotonic() > deadline:  # pragma: no cover - guards a hang
            raise AssertionError("factory never ran")
        await asyncio.sleep(0)
    await asyncio.wait_for(created[0].started.wait(), timeout=timeout)


class TestChatRouteMapsTheRefusal:
    @pytest.mark.asyncio
    async def test_terminated_mid_start_becomes_a_409(self):
        """The in-flight prompt gets an actionable answer, not a 500.

        Its child is gone by design; "send it again" respawns under the posture
        the operator just set, which is exactly what they asked for.
        """

        class _Registry:
            async def get_or_create_chat_session(self, chat_id, cwd, env=None):
                raise ChatSessionTerminatedError("terminated while starting")

        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(project_cwd="/tmp", operator_registry=_Registry())
            )
        )

        with pytest.raises(Exception) as excinfo:
            await chat_routes._acquire_chat_turn(request, CHAT_A)

        assert excinfo.value.status_code == 409
        assert excinfo.value.detail["error"] == "chat_terminated"


def _slow_session_class(created: list, delay: float = 0.05):
    """A chat-session class whose ``start()`` is still running on the next tick.

    The registry's factory builds ``OperatorSession(cwd=..., env=...)`` by
    resolving the name in ``operator_session`` at call time, so patching that
    name with this class is enough to put a REAL ``ChatSessionPool`` and a real
    ``OperatorRegistry`` under the route with a session that cannot start.
    """

    class _Slow(_FakeChatSession):
        def __init__(self, cwd="/tmp", env=None):
            super().__init__(cwd=cwd, env=env)
            self.start_delay = delay
            created.append(self)

    return _Slow


class TestAFlipDuringChatCreation:
    """The window the ownership probe used to leave open.

    ``get_chat_session`` reads the pool's session map; a creation still inside
    ``start()`` lives in ``_pending``. Gating the terminate on that probe meant
    a flip arriving mid-creation terminated nothing, returned 200, and let the
    child built from the PRE-flip environment register itself into the pool
    afterwards — with the creator going on to deliver the operator's prompt to
    it, because its ``get_or_create`` returned normally.

    The *addressability gate* had the same blind spot for longer, and it runs
    first: for a chat-only key it answered "no session here" and refused 409
    before the terminate was ever reached, so nothing was written and nothing
    was killed. Every test below therefore runs with ``known_sessions()``
    **empty** — the key is a chat and nothing else, which is the shipped
    topology. Only the SDK session is doubled; the registry, the pool, the
    route coroutine and the store are real.
    """

    @pytest.mark.asyncio
    async def test_the_pre_flip_child_never_lands_in_the_pool(self, client, shared_root):
        created: list[_FakeChatSession] = []
        registry = OperatorRegistry()
        client.app.state.operator_registry = registry

        with patch(
            "osprey.interfaces.web_terminal.operator_session.OperatorSession",
            _slow_session_class(created),
        ):
            creation = asyncio.create_task(
                registry.get_or_create_chat_session(
                    CHAT_A, "/tmp", {"OSPREY_EXECUTION_MODE": "writes"}
                )
            )
            await created_first(created)

            with known_sessions():
                result = await websocket_routes.set_terminal_posture(
                    PostureRequest(session_id=CHAT_A, posture="sandbox"),
                    SimpleNamespace(app=client.app),
                )

            with pytest.raises(ChatSessionTerminatedError):
                await creation

        assert result["posture"] == "sandbox"
        # The store was written — the 409 path writes nothing.
        assert client.app.state.session_postures[CHAT_A] == "sandbox"
        on_disk = json.loads((shared_root / "session-postures.json").read_text(encoding="utf-8"))
        assert on_disk == {CHAT_A: "sandbox"}
        # The pre-flip child is torn down and nothing is pooled under the key.
        assert registry.get_chat_session(CHAT_A) is None
        assert created[0].env == {"OSPREY_EXECUTION_MODE": "writes"}
        assert created[0].stop_calls == 1

    @pytest.mark.asyncio
    async def test_the_creator_is_refused_so_no_prompt_reaches_that_child(self, client):
        """The refusal is the guarantee, not just the teardown.

        A superseded creator raises out of ``_acquire_chat_turn`` before it can
        ``acquire_turn``, so the prompt that started the creation is never
        delivered to a child holding the old posture — it comes back as a 409
        the operator can act on.
        """
        created: list[_FakeChatSession] = []
        registry = OperatorRegistry()
        client.app.state.operator_registry = registry
        request = SimpleNamespace(app=client.app)

        with patch(
            "osprey.interfaces.web_terminal.operator_session.OperatorSession",
            _slow_session_class(created),
        ):
            turn = asyncio.create_task(chat_routes._acquire_chat_turn(request, CHAT_A))
            await created_first(created)

            with known_sessions():
                await websocket_routes.set_terminal_posture(
                    PostureRequest(session_id=CHAT_A, posture="sandbox"),
                    request,
                )

            with pytest.raises(Exception) as excinfo:
                await turn

        assert excinfo.value.status_code == 409
        assert excinfo.value.detail["error"] == "chat_terminated"
        assert registry.get_chat_session(CHAT_A) is None

    @pytest.mark.asyncio
    async def test_the_post_flip_child_carries_the_sandbox_env(self, client):
        """The retry the 409 asks for is what applies the posture.

        "Send the prompt again" is only an honest remedy if the child that
        comes back is the sandboxed one. This runs the real chat handler twice
        across the flip: the first creation is superseded, the second builds
        its environment from the store the flip wrote.
        """
        created: list[_FakeChatSession] = []
        client.app.state.operator_registry = OperatorRegistry()
        request = SimpleNamespace(app=client.app)

        with patch(
            "osprey.interfaces.web_terminal.operator_session.OperatorSession",
            _slow_session_class(created),
        ):
            turn = asyncio.create_task(chat_routes._acquire_chat_turn(request, CHAT_A))
            await created_first(created)

            with known_sessions():
                await websocket_routes.set_terminal_posture(
                    PostureRequest(session_id=CHAT_A, posture="sandbox"),
                    request,
                )

            with pytest.raises(Exception):
                await turn

            session, _token, was_reused = await chat_routes._acquire_chat_turn(request, CHAT_A)

        assert was_reused is False
        assert session is created[1]
        assert created[0].env.get("OSPREY_EXECUTION_MODE") != "readonly"
        assert created[1].env["OSPREY_EXECUTION_MODE"] == "readonly"

    def test_a_chat_key_the_pool_never_heard_of_is_still_a_409(self, client):
        """Widening the gate to ``_pending`` must not open it to any UUID.

        The pool answers to nothing under this key and no posture was ever set
        on it, so there is nothing to respawn and the refusal stands.
        """
        client.app.state.operator_registry = OperatorRegistry()

        with known_sessions():
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": CHAT_A, "posture": "sandbox"},
            )

        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "session_not_started"
        assert CHAT_A not in getattr(client.app.state, "session_postures", {})


class TestTheChatTerminateIsUngated:
    def test_the_pool_is_told_even_when_it_holds_no_entry(self, client):
        """The probe cannot see ``_pending``, so it must not gate the call.

        A key the pool holds nothing under is the same read the probe gets for
        a chat that is still starting, and only one of those is safe to skip.
        ``ChatSessionPool.terminate`` is idempotent and busy-safe, so the
        unconditional call costs a no-op on the keys that own no chat.
        """
        registry = _ChatRegistryDouble(app=client.app)
        client.app.state.operator_registry = registry

        with known_sessions(SESSION_A):
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "sandbox"},
            )

        assert resp.status_code == 200
        assert registry.terminated == [SESSION_A]


class TestChatPoolEnvFingerprint:
    """The chat pool's backstop: a reuse must not hand back a stale-env child.

    ``get_or_create`` used to return a live entry on an LRU bump alone,
    comparing nothing about the environment that child was built with. The
    posture route terminates the chat itself, so in the ordinary toggle flow
    there is no live entry left to compare against — but a missed terminate, a
    registry that cannot be told to act (``_DeafChatRegistry``), or a future
    caller that changes the launch env without knowing it must terminate first
    would all leave a warm SDK child serving the posture the operator just left,
    with the badge reporting the store. The PTY registry has carried an env
    fingerprint for exactly this since it was written; this is the same
    defence on the SDK topology, through the same helper.
    """

    @pytest.mark.asyncio
    async def test_an_env_change_rebuilds_instead_of_reusing(self):
        """The core of it: a different env means a different child."""
        pool, created = _pool()
        env = {"OSPREY_EXECUTION_MODE": "writes"}

        first, _ = await pool.get_or_create("a", "/tmp", lambda: dict(env))
        env["OSPREY_EXECUTION_MODE"] = "readonly"
        second, was_reused = await pool.get_or_create("a", "/tmp", lambda: dict(env))

        assert second is not first
        assert was_reused is False
        assert second.env == {"OSPREY_EXECUTION_MODE": "readonly"}
        assert first.stop_calls == 1
        assert len(created) == 2

    @pytest.mark.asyncio
    async def test_an_unchanged_env_still_reuses_the_live_session(self):
        """The liveness half: a second prompt must not kill the conversation.

        A fingerprint that fired on every turn would restart the agent under
        the operator mid-conversation, which is a worse failure than the one
        the comparison prevents.
        """
        pool, created = _pool()
        env = {"OSPREY_EXECUTION_MODE": "writes"}

        first, _ = await pool.get_or_create("a", "/tmp", lambda: dict(env))
        second, was_reused = await pool.get_or_create("a", "/tmp", lambda: dict(env))

        assert second is first
        assert was_reused is True
        assert first.stop_calls == 0
        assert len(created) == 1

    @pytest.mark.asyncio
    async def test_the_per_connection_names_do_not_force_a_respawn(self):
        """One deny list, shared with the PTY seam — not a second shape.

        ``OSPREY_POSTURE_SESSION`` is the pool key itself, so it carries no
        privilege the key does not; fingerprinting it would respawn a chat for
        a name that cannot differ in a way that matters.
        """
        pool, created = _pool()
        base = {"OSPREY_EXECUTION_MODE": "writes"}

        first, _ = await pool.get_or_create("a", "/tmp", lambda: {**base, POSTURE_SESSION_ENV: "a"})
        second, was_reused = await pool.get_or_create(
            "a", "/tmp", lambda: {**base, POSTURE_SESSION_ENV: "somewhere-else"}
        )

        assert second is first
        assert was_reused is True
        assert len(created) == 1

    @pytest.mark.asyncio
    async def test_a_creation_still_starting_under_the_old_env_is_overtaken(self):
        """The same rule for a child that is being built, not yet pooled.

        Joining an in-flight creation is what makes a concurrent double-submit
        share one SDK subprocess, but joining one that was built from a
        superseded environment would hand this caller the very child the change
        was meant to replace. The creation is overtaken instead — the same
        supersede a terminate uses — and this call builds the new one.
        """
        pool, created = _pool(start_delay=0.05)
        first = asyncio.create_task(
            pool.get_or_create("a", "/tmp", lambda: {"OSPREY_EXECUTION_MODE": "writes"})
        )
        await created_first(created)

        second, was_reused = await pool.get_or_create(
            "a", "/tmp", lambda: {"OSPREY_EXECUTION_MODE": "readonly"}
        )

        with pytest.raises(ChatSessionTerminatedError):
            await first
        assert was_reused is False
        assert second is created[1]
        assert second.env == {"OSPREY_EXECUTION_MODE": "readonly"}
        assert pool.get("a") is second
        assert created[0].stop_calls == 1

    @pytest.mark.asyncio
    async def test_a_concurrent_double_submit_still_shares_one_creation(self):
        """The liveness half of the same rule: identical env, one subprocess."""
        pool, created = _pool(start_delay=0.05)
        env = {"OSPREY_EXECUTION_MODE": "writes"}

        first = asyncio.create_task(pool.get_or_create("a", "/tmp", lambda: dict(env)))
        await created_first(created)
        second, was_reused = await pool.get_or_create("a", "/tmp", lambda: dict(env))

        session, _ = await first
        assert second is session
        assert was_reused is True
        assert len(created) == 1

    @pytest.mark.asyncio
    async def test_a_missed_terminate_still_cannot_serve_the_old_posture(self, client):
        """End to end on the real registry and the real chat handler.

        The store is flipped without the route's terminate — the state a
        deaf registry, a raced reconnect or a future caller would produce.
        The next prompt must still reach a child built from the store as it
        now stands.
        """
        created: list[_FakeChatSession] = []
        registry = OperatorRegistry()
        client.app.state.operator_registry = registry
        request = SimpleNamespace(app=client.app)

        with patch(
            "osprey.interfaces.web_terminal.operator_session.OperatorSession",
            _FakeChatSession,
        ):
            first, _token, _ = await chat_routes._acquire_chat_turn(request, CHAT_A)
            created.append(first)
            # The flip the route would have terminated for, minus the terminate.
            websocket_routes._session_postures(client.app)[CHAT_A] = "sandbox"
            second, _token2, was_reused = await chat_routes._acquire_chat_turn(request, CHAT_A)

        assert second is not first
        assert was_reused is False
        assert first.env.get("OSPREY_EXECUTION_MODE") != "readonly"
        assert second.env["OSPREY_EXECUTION_MODE"] == "readonly"


class TestTheEnvIsReadUnderThePoolLock:
    """Atomicity of "read the posture" and "register the creation".

    Before, the route read the store and handed the pool a finished mapping.
    Nothing kept those two steps together except the accident that no await on
    the way in actually suspends — one added ``await`` in the handler and a
    flip could land in the gap, with no test going red. The pool now resolves
    a builder inside the lock hold that registers the pending creation.
    """

    @pytest.mark.asyncio
    async def test_the_builder_runs_while_the_lock_is_held(self):
        pool, _created = _pool()
        held: list[bool] = []

        def build_env():
            held.append(pool._lock.locked())
            return {"OSPREY_EXECUTION_MODE": "readonly"}

        session, _ = await pool.get_or_create("a", "/tmp", build_env)

        assert held == [True]
        assert session.env == {"OSPREY_EXECUTION_MODE": "readonly"}

    @pytest.mark.asyncio
    async def test_a_flip_landing_after_the_call_still_governs_the_child(self):
        """The seam the finding is about, made real by holding the lock.

        With the lock held, the creation is parked *inside* ``get_or_create``:
        the caller has committed, and the environment has not been read yet.
        A posture written in that window is the one the child gets. Hand the
        pool a ready-made mapping instead and this asserts ``writes``.
        """
        pool, _created = _pool()
        store = {"posture": "writes"}

        def build_env():
            return {"OSPREY_EXECUTION_MODE": store["posture"]}

        async with pool._lock:
            creation = asyncio.create_task(pool.get_or_create("a", "/tmp", build_env))
            for _ in range(3):  # let the task reach the lock it cannot take
                await asyncio.sleep(0)
            store["posture"] = "readonly"

        session, _ = await creation

        assert session.env == {"OSPREY_EXECUTION_MODE": "readonly"}

    @pytest.mark.asyncio
    async def test_a_mapping_is_still_accepted(self):
        """Every other caller (and every test double) still passes a dict."""
        pool, _created = _pool()
        session, _ = await pool.get_or_create("a", "/tmp", {"MODE": "plain"})
        assert session.env == {"MODE": "plain"}

    def test_the_chat_route_hands_the_pool_a_builder(self, client):
        """The route side of the same invariant.

        A regression to a pre-built mapping is invisible in behaviour today,
        so this pins the shape: what reaches the pool must be callable, and
        calling it must produce the posture the store holds now.
        """
        captured: dict[str, object] = {}

        class _Registry:
            async def get_or_create_chat_session(self, chat_id, cwd, env=None):
                captured["env"] = env
                return SimpleNamespace(acquire_turn=lambda: 1), False

            async def cleanup_all(self):  # the lifespan's shutdown calls this
                return None

        client.app.state.operator_registry = _Registry()
        websocket_routes._session_postures(client.app)[CHAT_A] = "sandbox"

        asyncio.run(chat_routes._acquire_chat_turn(SimpleNamespace(app=client.app), CHAT_A))

        build_env = captured["env"]
        assert callable(build_env)
        assert build_env()["OSPREY_EXECUTION_MODE"] == "readonly"


class TestARaisingBuilderStrandsNothing:
    @pytest.mark.asyncio
    async def test_a_dead_entry_is_still_torn_down_when_the_builder_raises(self):
        """The dead entry is popped from the map before the builder runs; a
        builder that raises must not leave it popped-but-unreaped. The error
        still propagates, the key still answers afterwards."""
        pool, created = _pool()
        first, _ = await pool.get_or_create("a", "/tmp", {"MODE": "1"})
        first.is_active = False  # a dead entry — torn down on the way past

        def build_env():
            raise RuntimeError("the posture store is gone")

        with pytest.raises(RuntimeError, match="posture store"):
            await pool.get_or_create("a", "/tmp", build_env)

        assert first.stop_calls == 1
        assert pool.get("a") is None
        assert "a" not in pool._pending
        assert not pool._superseded
        second, was_reused = await pool.get_or_create("a", "/tmp", {"MODE": "2"})
        assert was_reused is False
        assert second is not first
        assert len(created) == 2

    @pytest.mark.asyncio
    async def test_a_raising_builder_at_capacity_evicts_nobody(self):
        """The builder runs before the capacity check, so its failure reserves
        no slot: a pool at capacity keeps every live session it had."""
        pool, _created = _pool(max_sessions=2)
        a, _ = await pool.get_or_create("a", "/tmp", {"MODE": "1"})
        b, _ = await pool.get_or_create("b", "/tmp", {"MODE": "1"})

        def build_env():
            raise RuntimeError("the posture store is gone")

        with pytest.raises(RuntimeError, match="posture store"):
            await pool.get_or_create("c", "/tmp", build_env)

        assert pool.get("a") is a and a.is_active and a.stop_calls == 0
        assert pool.get("b") is b and b.is_active and b.stop_calls == 0
        assert not pool._pending and not pool._superseded


class TestARaisingTeardownDoesNotWedgeTheKey:
    @pytest.mark.asyncio
    async def test_the_key_is_still_usable_after_a_failed_teardown(self):
        """The teardown of a dead entry runs after ``_pending`` is registered.

        Left uncaught it propagates out of ``get_or_create`` without clearing
        that entry, and the key wedges permanently: every later call joins a
        Future nobody will ever settle, and a terminate files that orphan in
        ``_superseded``, which nothing discards. A failed stop must cost a
        leaked child, not the key.
        """
        pool, created = _pool()
        first, _ = await pool.get_or_create("a", "/tmp", {"MODE": "1"})
        first.is_active = False  # a dead entry — torn down on the way past

        async def _boom():
            raise RuntimeError("teardown blew up")

        first.teardown = _boom

        second, was_reused = await pool.get_or_create("a", "/tmp", {"MODE": "2"})

        assert was_reused is False
        assert second is not first
        assert pool.get("a") is second
        # The pending entry was cleared, so the key still answers. Same env as
        # the call that built it, so this is a reuse: a *different* env here
        # would rebuild for its own reason (TestChatPoolEnvFingerprint) and
        # would say nothing about whether the key had wedged.
        third, reused = await pool.get_or_create("a", "/tmp", {"MODE": "2"})
        assert third is second
        assert reused is True
        assert len(created) == 2
