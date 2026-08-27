"""Tests for the rekey join: one stable key across a Claude-UUID discovery.

A PTY session is pooled under its telemetry id at spawn and rekeyed to the
discovered Claude UUID moments later. Two things have to survive that rename,
and before this task neither did:

* **The posture store entry.** ``_build_extra_env`` looks the posture up under
  the *pool* key, so an entry left behind under the old key is invisible to
  every later spawn and to the badge route — a sandboxed session would come
  back writable on its next respawn. :meth:`PtyRegistry.rekey_session` moves
  the entry with the session and persists the move.

* **The audit join key.** The live child's ``OSPREY_POSTURE_SESSION`` is fixed
  at ``execvp`` time and still carries the *spawn* key — deliberately, since
  the name is in :data:`POOL_FINGERPRINT_EXCLUDED_ENV` precisely so the rekey
  does not kill the child. Every tool record that child emits therefore joins
  on the telemetry id. A server-side toggle event, which only knows the
  current pool key, must resolve back to that same spawn key or the ledger
  splits one session into two.
  :meth:`PtyRegistry.audit_session_key` is that resolution.

The two keys deliberately differ after a rekey: the store follows the session
(the routes address it by its current id) while the audit join follows the
child (its exported marker cannot be rewritten). Both facts are pinned here.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from osprey.interfaces.web_terminal.pty_manager import PtyRegistry
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes

SPAWN_KEY = "11111111-1111-4111-8111-111111111111"
CLAUDE_KEY = "22222222-2222-4222-8222-222222222222"
OTHER_KEY = "33333333-3333-4333-8333-333333333333"


def _mock_session(alive: bool = True) -> MagicMock:
    """A stand-in PtySession — the pool never spawns a real PTY here."""
    s = MagicMock()
    s.is_alive = alive
    s.resize = MagicMock()
    s.terminate = MagicMock()
    return s


@pytest.fixture
def app(tmp_path):
    """An app object carrying a live, non-provisional posture store.

    Presetting ``session_postures`` (and clearing the provisional flag) makes
    :func:`_session_postures` return it without touching disk, so these tests
    exercise the move rather than the loader. The store *path* is redirected
    into ``tmp_path`` so the write-through :func:`_persist_postures` does is
    real and inspectable without escaping the test.
    """
    state = SimpleNamespace(
        session_postures={},
        session_postures_provisional=False,
        workspace_dir=tmp_path,
    )
    return SimpleNamespace(state=state)


@pytest.fixture
def store_path(tmp_path):
    """Pin the persisted store under tmp_path for the duration of a test."""
    path = tmp_path / "session-postures.json"
    with patch.object(websocket_routes, "_posture_store_path", return_value=path):
        yield path


@pytest.fixture
def registry():
    return PtyRegistry(max_background=3)


def _pooled(registry, key):
    """Put a live pooled session under *key* and return it."""
    session = _mock_session()
    registry._sessions[key] = session
    return session


class TestPostureStoreMovesWithTheSession:
    """The store entry follows the session's current key."""

    def test_entry_moves_from_old_key_to_new(self, registry, app, store_path):
        _pooled(registry, SPAWN_KEY)
        websocket_routes._session_postures(app)[SPAWN_KEY] = websocket_routes.POSTURE_SANDBOX

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY, app=app)

        store = websocket_routes._session_postures(app)
        assert SPAWN_KEY not in store
        assert store[CLAUDE_KEY] == websocket_routes.POSTURE_SANDBOX

    def test_moved_entry_is_what_the_next_spawn_reads(self, registry, app, store_path):
        """The move is only worth anything if the pool-key lookup now hits.

        ``_build_extra_env`` keys on ``claude_session_id or telemetry_id`` —
        the new key after a rekey. Reading through that expression is the
        actual regression this task fixes.
        """
        _pooled(registry, SPAWN_KEY)
        websocket_routes._session_postures(app)[SPAWN_KEY] = websocket_routes.POSTURE_SANDBOX

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY, app=app)

        posture_key = CLAUDE_KEY  # what the handler now keys the pool on
        assert websocket_routes._session_postures(app).get(posture_key) == (
            websocket_routes.POSTURE_SANDBOX
        )

    def test_move_is_persisted(self, registry, app, store_path):
        """A restart must not resurrect the entry under the stale key.

        The pool is empty after a restart and the client reconnects with the
        Claude UUID it was told about, so an on-disk entry still filed under
        the telemetry id would never be consulted again.
        """
        import json

        _pooled(registry, SPAWN_KEY)
        websocket_routes._session_postures(app)[SPAWN_KEY] = websocket_routes.POSTURE_SANDBOX

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY, app=app)

        written = json.loads(store_path.read_text(encoding="utf-8"))
        assert written == {CLAUDE_KEY: websocket_routes.POSTURE_SANDBOX}

    def test_rekey_with_no_store_entry_is_a_noop(self, registry, app, store_path):
        """An unposture'd session leaves the store untouched — no null entry.

        "No entry" is the store's spelling of ``writes``; writing one on every
        rekey would turn every discovery into a durable record of a deviation
        that never happened.
        """
        _pooled(registry, SPAWN_KEY)
        websocket_routes._session_postures(app)[OTHER_KEY] = websocket_routes.POSTURE_SANDBOX

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY, app=app)

        store = websocket_routes._session_postures(app)
        assert store == {OTHER_KEY: websocket_routes.POSTURE_SANDBOX}
        assert not store_path.exists()

    def test_other_sessions_are_untouched(self, registry, app, store_path):
        _pooled(registry, SPAWN_KEY)
        store = websocket_routes._session_postures(app)
        store[SPAWN_KEY] = websocket_routes.POSTURE_SANDBOX
        store[OTHER_KEY] = websocket_routes.POSTURE_WRITES

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY, app=app)

        assert websocket_routes._session_postures(app)[OTHER_KEY] == (
            websocket_routes.POSTURE_WRITES
        )

    def test_rekey_without_an_app_still_moves_the_pool_entry(self, registry):
        """``app`` is optional — the pool rename must not depend on it.

        Callers outside a request (and the pool's own unit tests) rekey with
        no app in hand; they get the rename and no store side effect.
        """
        session = _pooled(registry, SPAWN_KEY)

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)

        assert registry._sessions[CLAUDE_KEY] is session
        assert SPAWN_KEY not in registry._sessions

    def test_no_pool_entry_leaves_the_store_alone(self, registry, app, store_path):
        """Rekeying a key the pool does not hold renames nothing at all.

        There is no live child to keep joined, so moving a store entry would
        be renaming a session that does not exist.
        """
        websocket_routes._session_postures(app)[SPAWN_KEY] = websocket_routes.POSTURE_SANDBOX

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY, app=app)

        store = websocket_routes._session_postures(app)
        assert store == {SPAWN_KEY: websocket_routes.POSTURE_SANDBOX}
        assert CLAUDE_KEY not in registry._sessions

    def test_a_store_failure_does_not_undo_the_rename(self, registry, app):
        """The pool rename is the load-bearing half and must always land.

        Losing the store move degrades the posture join; losing the rename
        orphans a live child under a key nothing will ever attach to again.
        """
        session = _pooled(registry, SPAWN_KEY)
        websocket_routes._session_postures(app)[SPAWN_KEY] = websocket_routes.POSTURE_SANDBOX

        with patch.object(
            websocket_routes, "_session_postures", side_effect=RuntimeError("store is down")
        ):
            registry.rekey_session(SPAWN_KEY, CLAUDE_KEY, app=app)

        assert registry._sessions[CLAUDE_KEY] is session
        assert registry.audit_session_key(CLAUDE_KEY) == SPAWN_KEY


class TestAuditSessionKeyAlias:
    """The audit join resolves a current pool key back to its spawn key."""

    def test_alias_resolves_the_new_key_to_the_spawn_key(self, registry, app):
        _pooled(registry, SPAWN_KEY)

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY, app=app)

        assert registry.audit_session_key(CLAUDE_KEY) == SPAWN_KEY

    def test_a_toggle_after_rekey_would_emit_the_spawn_key(self, registry, app, store_path):
        """The emitter's whole contract, exercised through the two seams.

        A posture POST addresses the session by its *current* id: it writes the
        store under that id and terminates the PTY under it. The audit record
        it will emit (a later task wires the writer) must nonetheless carry the
        key the live child stamped into its own tool records — otherwise the
        toggle and the tool calls it governs land in the ledger as two
        unrelated sessions.
        """
        _pooled(registry, SPAWN_KEY)
        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY, app=app)

        # What the route sees and what the emitter must record.
        route_session_id = CLAUDE_KEY
        emitted_session = registry.audit_session_key(route_session_id)

        assert emitted_session == SPAWN_KEY
        # ...which is exactly the marker the live child was spawned with.
        assert emitted_session != route_session_id

    def test_unknown_key_resolves_to_itself(self, registry):
        """No rekey means the key already is the spawn key.

        The overwhelmingly common case — a resumed session, a chat key, a
        session whose UUID was known up front — must need no bookkeeping.
        """
        assert registry.audit_session_key(SPAWN_KEY) == SPAWN_KEY

    def test_double_rekey_still_names_the_original_spawn_key(self, registry, app):
        """Chained renames collapse to the first key, not the previous one."""
        _pooled(registry, SPAWN_KEY)

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY, app=app)
        registry.rekey_session(CLAUDE_KEY, OTHER_KEY, app=app)

        assert registry.audit_session_key(OTHER_KEY) == SPAWN_KEY
        assert registry.audit_session_key(CLAUDE_KEY) == CLAUDE_KEY

    def test_rekey_back_to_the_spawn_key_drops_the_alias(self, registry, app):
        """A round trip leaves no identity alias behind to reason about."""
        _pooled(registry, SPAWN_KEY)

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY, app=app)
        registry.rekey_session(CLAUDE_KEY, SPAWN_KEY, app=app)

        assert registry.audit_session_key(SPAWN_KEY) == SPAWN_KEY
        assert registry._audit_keys == {}

    def test_repeated_rekey_to_the_same_key_is_idempotent(self, registry, app):
        """The second call finds nothing to move and must not shift the alias."""
        _pooled(registry, SPAWN_KEY)

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY, app=app)
        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY, app=app)

        assert registry.audit_session_key(CLAUDE_KEY) == SPAWN_KEY


class TestAliasLifetime:
    """The alias describes a live child and dies with it."""

    def test_terminate_forgets_the_alias(self, registry, app):
        _pooled(registry, SPAWN_KEY)
        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY, app=app)

        registry.terminate_session(CLAUDE_KEY)

        assert registry.audit_session_key(CLAUDE_KEY) == CLAUDE_KEY

    def test_respawn_under_the_new_key_is_its_own_spawn_key(self, registry, app):
        """The toggle flow's real sequence, end to end.

        POST terminates the PTY; the client reconnects under the Claude UUID
        and a *fresh* child is spawned, exporting that UUID as its own
        ``OSPREY_POSTURE_SESSION``. Resolving it back to the dead child's
        telemetry id would misfile every record the new child produces.
        """
        _pooled(registry, SPAWN_KEY)
        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY, app=app)
        registry.terminate_session(CLAUDE_KEY)

        with patch.object(registry, "_spawn_session", return_value=_mock_session()):
            registry.get_or_create_session(CLAUDE_KEY, ["claude"], 24, 80)

        assert registry.audit_session_key(CLAUDE_KEY) == CLAUDE_KEY

    def test_eviction_forgets_the_alias(self, registry, app):
        _pooled(registry, SPAWN_KEY)
        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY, app=app)
        registry._sessions["b"] = _mock_session()
        registry._sessions["c"] = _mock_session()

        with patch.object(registry, "_spawn_session", return_value=_mock_session()):
            registry.get_or_create_session("d", ["claude"], 24, 80)

        assert CLAUDE_KEY not in registry._sessions
        assert registry.audit_session_key(CLAUDE_KEY) == CLAUDE_KEY

    def test_cleanup_all_clears_the_alias_map(self, registry, app):
        _pooled(registry, SPAWN_KEY)
        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY, app=app)

        registry.cleanup_all()

        assert registry._audit_keys == {}


class TestTheDiscoveryCallSiteCarriesTheApp:
    """The one call site that moves a posture entry across a rekey.

    ``_discover_and_notify`` is where a real session is rekeyed onto its
    discovered Claude UUID, and ``app=websocket.app`` is the whole of what
    carries the store entry along. Every test above drives ``rekey_session``
    directly, so dropping that keyword left them all green while a sandboxed
    session would come back writable on its next respawn — a
    privilege-widening regression with no test on it. This drives the shipped
    coroutine instead.
    """

    @pytest.mark.asyncio
    async def test_discovery_moves_the_posture_entry_to_the_discovered_id(
        self, registry, app, store_path
    ):
        _pooled(registry, SPAWN_KEY)
        websocket_routes._session_postures(app)[SPAWN_KEY] = websocket_routes.POSTURE_SANDBOX
        sent: list[str] = []

        async def _send_text(payload):
            sent.append(payload)

        websocket = SimpleNamespace(app=app, send_text=_send_text)
        discovery = SimpleNamespace(discover_new_session=lambda snapshot, timeout: CLAUDE_KEY)

        found = await websocket_routes._discover_and_notify(
            set(), discovery, registry, SPAWN_KEY, websocket
        )

        assert found == CLAUDE_KEY
        assert CLAUDE_KEY in sent[0]
        store = websocket_routes._session_postures(app)
        assert SPAWN_KEY not in store
        assert store[CLAUDE_KEY] == websocket_routes.POSTURE_SANDBOX
        # And the pool moved with it, so the two keys cannot drift apart.
        assert registry.get_session(CLAUDE_KEY) is not None

    @pytest.mark.asyncio
    async def test_a_discovery_that_finds_nothing_leaves_the_store_alone(
        self, registry, app, store_path
    ):
        """No rekey, no move: the session is still pooled under its spawn key."""
        _pooled(registry, SPAWN_KEY)
        websocket_routes._session_postures(app)[SPAWN_KEY] = websocket_routes.POSTURE_SANDBOX

        async def _send_text(payload):  # pragma: no cover - must not be reached
            raise AssertionError("nothing was discovered; the client is told nothing")

        websocket = SimpleNamespace(app=app, send_text=_send_text)
        discovery = SimpleNamespace(discover_new_session=lambda snapshot, timeout: None)

        found = await websocket_routes._discover_and_notify(
            set(), discovery, registry, SPAWN_KEY, websocket
        )

        assert found is None
        assert websocket_routes._session_postures(app) == {
            SPAWN_KEY: websocket_routes.POSTURE_SANDBOX
        }
