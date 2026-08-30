"""The chat pool's terminate, and the registry facades the chat surfaces use.

Terminating a chat child used to be half of applying a posture: the posture
travelled in the child's environment, so a flip had to kill the child and let
it come back. It does not any more — the per-target posture is recorded in the
store and read at write time, so a narrowing lands on a chat already
mid-conversation and ``POST /api/terminal/posture`` terminates nothing. (The
posture route asks no addressability question at all any more: any well-formed
key is accepted, because the store only narrows and both spawn paths read it
before the first write.)

What survives is everything that was never about the posture:

* **the pool's terminate itself**, which ``POST /api/chat/<id>/reset`` still
  drives — eviction, the supersede of a creation still inside ``start()``, and
  the 409 the chat route maps that supersede to;
* **the env fingerprint**, the SDK counterpart of the PTY registry's: a reuse
  must not hand back a child built with a different environment;
* **the registry facades** (``has_chat_key`` and friends) the GET roster and
  the reset route still call, pinned by name because their renames fail
  quietly.

Harness mirrors ``test_posture_routes.py``: each test builds its own app
through ``create_app`` under a patched ``_load_web_config``, entered as a
``TestClient`` context manager so the lifespan runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
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
from osprey_connectors import session_store

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
def shared_root(tmp_path, monkeypatch):
    """Stand in for the deployment's shared agent-data root.

    Stamped, not patched. ``session_store.agent_data_root()`` reads
    ``OSPREY_AGENT_DATA_ROOT`` FIRST and only falls back to
    ``resolve_shared_data_root``, so patching the resolver is inert for any
    developer whose shell carries the stamp — which is exactly the environment
    this feature creates — and the store these tests write would land in the
    repository's own ``var/agent_data``. The stamp is also the one anchor
    ``target_state.state_dir()`` prefers, so both halves land in one directory.
    Same idiom as ``test_posture_routes.py::agent_data_root``.
    """
    root = tmp_path / "shared_agent_data"
    root.mkdir()
    monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    monkeypatch.delenv("OSPREY_POSTURE_SESSION", raising=False)
    session_store.invalidate_cache()
    yield root
    session_store.invalidate_cache()


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


class TestAChatKeyIsAddressable:
    """The registry facades the chat surfaces rely on stay pinned by name."""

    def test_the_shipped_registry_exposes_the_facades_the_callers_use(self):
        """A rename of any of these is silent in its own way.

        * ``has_chat_key`` — :func:`_chat_pool_answers_to` would fall back to
          the session-map read, which cannot see a creation still inside
          ``start()``, and the GET roster would stop marking a starting chat's
          rows as ``chat_session``;
        * ``get_chat_session`` — that fallback itself would answer ``False``
          for every key; and
        * ``terminate_chat_session`` — the chat reset route would stop tearing
          anything down.

        All three fail quietly, so all three are pinned here.
        """
        for name in ("terminate_chat_session", "get_chat_session", "has_chat_key"):
            assert callable(getattr(OperatorRegistry, name, None)), name


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


class TestChatPoolEnvFingerprint:
    """The chat pool's backstop: a reuse must not hand back a stale-env child.

    ``get_or_create`` used to return a live entry on an LRU bump alone,
    comparing nothing about the environment that child was built with. The
    session's write posture no longer travels that way — it is read from the
    store at write time — but everything else a child is launched with still
    does: a rotated panel token, a deployment-wide readonly marker, a privilege
    name a later change adds. A caller that changes the launch env without
    knowing it must terminate first would otherwise keep a warm SDK child
    running under an environment nobody believes it has. The PTY registry has
    carried an env fingerprint for exactly this since it was written; this is
    the same defence on the SDK topology, through the same helper.
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
    async def test_a_narrowing_does_not_rebuild_the_chat_child(self, client):
        """End to end on the real registry and the real chat handler.

        The other side of the fingerprint, and the point of the whole feature:
        a posture flip is NOT an environment change, so the child that is
        already holding the operator\'s conversation is reused. The narrowing
        still governs that child — its next write reads the store — which is
        why nothing has to die for it to apply.
        """
        registry = OperatorRegistry()
        client.app.state.operator_registry = registry
        request = SimpleNamespace(app=client.app)

        with patch(
            "osprey.interfaces.web_terminal.operator_session.OperatorSession",
            _FakeChatSession,
        ):
            first, _token, _ = await chat_routes._acquire_chat_turn(request, CHAT_A)
            websocket_routes._session_postures(client.app)[CHAT_A] = {"standin": "sandbox"}
            second, _token2, was_reused = await chat_routes._acquire_chat_turn(request, CHAT_A)

        assert second is first
        assert was_reused is True
        assert first.stop_calls == 0
        # No mode was ever stamped; the child was handed the store key instead.
        assert "OSPREY_EXECUTION_MODE" not in first.env
        assert first.env[POSTURE_SESSION_ENV] == CHAT_A


class TestTheEnvIsReadUnderThePoolLock:
    """Atomicity of "build the environment" and "register the creation".

    Before, the route built the child\'s environment and handed the pool a
    finished mapping. Nothing kept those two steps together except the accident
    that no await on the way in actually suspends — one added ``await`` in the
    handler and a change could land in the gap, with no test going red. The
    pool now resolves a builder inside the lock hold that registers the pending
    creation, so whatever the environment is derived from is read at the moment
    the child is committed to.
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
    async def test_a_change_landing_after_the_call_still_governs_the_child(self):
        """The seam the finding is about, made real by holding the lock.

        With the lock held, the creation is parked *inside* ``get_or_create``:
        the caller has committed, and the environment has not been read yet.
        A change written in that window is the one the child gets. Hand the
        pool a ready-made mapping instead and this asserts the stale value.
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
        calling it must produce the environment this chat would be spawned
        with — the store key it reads its posture under, and no execution mode.
        """
        captured: dict[str, object] = {}

        class _Registry:
            async def get_or_create_chat_session(self, chat_id, cwd, env=None):
                captured["env"] = env
                return SimpleNamespace(acquire_turn=lambda: 1), False

            async def cleanup_all(self):  # the lifespan's shutdown calls this
                return None

        client.app.state.operator_registry = _Registry()

        asyncio.run(chat_routes._acquire_chat_turn(SimpleNamespace(app=client.app), CHAT_A))

        build_env = captured["env"]
        assert callable(build_env)
        built = build_env()
        assert built[POSTURE_SESSION_ENV] == CHAT_A
        assert "OSPREY_EXECUTION_MODE" not in built


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
