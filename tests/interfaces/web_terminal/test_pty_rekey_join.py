"""Tests for the rekey join: one stable narrowing across a Claude-UUID discovery.

A PTY session is pooled under its telemetry id at spawn and rekeyed to the
discovered Claude UUID moments later. The live child cannot follow: its
``OSPREY_POSTURE_SESSION`` was fixed at ``execvp`` time and still carries the
*spawn* key — deliberately, since the name is in
:data:`POOL_FINGERPRINT_EXCLUDED_ENV` precisely so the rekey does not kill the
child. So two readers of the posture store end up asking about the same
session under two different keys: the running child under the telemetry id,
and anything that comes after a server restart (a reattach, the badge, the
toggle) under the Claude UUID, because the alias map is memory-only.

The store answers both by **writing both**. Every entry the web server persists
lands under the current key *and* under
:meth:`PtyRegistry.audit_session_key`'s answer for it, and every read takes the
current key first and falls back to the spawn key.

**The rekey itself therefore copies nothing.** It fires moments after the
spawn, before any entry for this session can exist, so there has never been
anything to move; it renames the pooled session and records the audit alias,
and does not touch the store at all. That is what these tests pin — together
with the alias's own lifetime, which is the resolution the dual write depends
on.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from osprey.interfaces.web_terminal.pty_manager import PtyRegistry
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey_connectors import session_store

SPAWN_KEY = "11111111-1111-4111-8111-111111111111"
CLAUDE_KEY = "22222222-2222-4222-8222-222222222222"
OTHER_KEY = "33333333-3333-4333-8333-333333333333"

SANDBOX = websocket_routes.POSTURE_SANDBOX
NARROWED = {"live": SANDBOX}


def _mock_session(alive: bool = True) -> MagicMock:
    """A stand-in PtySession — the pool never spawns a real PTY here."""
    s = MagicMock()
    s.is_alive = alive
    s.resize = MagicMock()
    s.terminate = MagicMock()
    return s


@pytest.fixture
def registry():
    return PtyRegistry(max_background=3)


@pytest.fixture
def store_file(tmp_path, monkeypatch):
    """Pin the agent-data root, and return the store path under it.

    ``OSPREY_AGENT_DATA_ROOT`` is the first half of the store's one resolution
    rule, so a persist in these tests is real and inspectable without escaping
    the test.
    """
    root = tmp_path / "shared_agent_data"
    root.mkdir()
    monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    session_store.invalidate_cache()
    yield root / session_store.STATE_DIR_NAME / session_store.STORE_FILENAME
    session_store.invalidate_cache()


@pytest.fixture
def app(tmp_path, registry):
    """An app carrying a live, non-provisional store and the pool registry.

    Presetting ``session_postures`` (and clearing the provisional flag) makes
    :func:`_session_postures` return it without touching disk, so a test that
    seeds the store exercises the rekey rather than the loader.
    """
    state = SimpleNamespace(
        session_postures={},
        session_postures_provisional=False,
        workspace_dir=tmp_path,
        pty_registry=registry,
    )
    return SimpleNamespace(state=state)


def _pooled(registry, key):
    """Put a live pooled session under *key* and return it."""
    session = _mock_session()
    registry._sessions[key] = session
    return session


def _written(store_file) -> dict:
    return json.loads(store_file.read_text(encoding="utf-8"))


class TestRekeyCopiesNothing:
    """The rename is a pool operation; the store is not part of it."""

    def test_the_store_is_untouched_by_a_rekey(self, registry, app, store_file):
        """Nothing moves, nothing is minted, nothing is written.

        The store is addressed under both keys by construction, so moving an
        entry here would only have to choose which of the two readers to take
        it away from.
        """
        _pooled(registry, SPAWN_KEY)
        websocket_routes._session_postures(app)[SPAWN_KEY] = dict(NARROWED)

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)

        assert websocket_routes._session_postures(app) == {SPAWN_KEY: NARROWED}
        assert not store_file.exists()

    def test_an_unnarrowed_session_stays_unnarrowed(self, registry, app, store_file):
        """ "No entry" is how the store spells writes; a rekey must not mint one."""
        _pooled(registry, SPAWN_KEY)

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)

        assert websocket_routes._session_postures(app) == {}
        assert not store_file.exists()

    def test_a_spawn_key_entry_still_answers_under_the_new_key(self, registry, app, store_file):
        """The read fallback is what makes the missing move harmless.

        A store written before this rule — or by a child under its own key —
        holds only the spawn key. The current key still resolves to it.
        """
        _pooled(registry, SPAWN_KEY)
        websocket_routes._session_postures(app)[SPAWN_KEY] = dict(NARROWED)

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)

        assert websocket_routes._posture_entry(app, CLAUDE_KEY) == NARROWED

    def test_a_persist_after_a_rekey_lands_under_both_keys(self, registry, app, store_file):
        """The dual write is where the two readers are reunited.

        The route addresses the session by its *current* id; the live child
        reads the telemetry id it was spawned with. One narrowing, both keys.
        """
        _pooled(registry, SPAWN_KEY)
        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)

        websocket_routes.persist_or_raise(app, CLAUDE_KEY, NARROWED)

        assert _written(store_file) == {CLAUDE_KEY: NARROWED, SPAWN_KEY: NARROWED}
        assert websocket_routes._posture_entry(app, CLAUDE_KEY) == NARROWED
        assert websocket_routes._posture_entry(app, SPAWN_KEY) == NARROWED

    def test_clearing_after_a_rekey_clears_both_keys(self, registry, app, store_file):
        """A half-cleared session would keep narrowing whichever key survived."""
        _pooled(registry, SPAWN_KEY)
        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)
        websocket_routes.persist_or_raise(app, CLAUDE_KEY, NARROWED)

        websocket_routes.persist_or_raise(app, CLAUDE_KEY, {})

        assert _written(store_file) == {}
        assert websocket_routes._posture_entry(app, SPAWN_KEY) == {}

    def test_other_sessions_are_untouched(self, registry, app, store_file):
        _pooled(registry, SPAWN_KEY)
        websocket_routes._session_postures(app)[OTHER_KEY] = dict(NARROWED)

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)
        websocket_routes.persist_or_raise(app, CLAUDE_KEY, {"va": SANDBOX})

        assert websocket_routes._posture_entry(app, OTHER_KEY) == NARROWED

    def test_the_pool_entry_moves(self, registry):
        """The load-bearing half of the rename, with no app in sight."""
        session = _pooled(registry, SPAWN_KEY)

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)

        assert registry._sessions[CLAUDE_KEY] is session
        assert SPAWN_KEY not in registry._sessions

    def test_no_pool_entry_renames_nothing(self, registry, app, store_file):
        """Rekeying a key the pool does not hold is a no-op end to end."""
        websocket_routes._session_postures(app)[SPAWN_KEY] = dict(NARROWED)

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)

        assert websocket_routes._session_postures(app) == {SPAWN_KEY: NARROWED}
        assert CLAUDE_KEY not in registry._sessions


class TestAuditSessionKeyAlias:
    """The alias resolves a current pool key back to its spawn key."""

    def test_alias_resolves_the_new_key_to_the_spawn_key(self, registry):
        _pooled(registry, SPAWN_KEY)

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)

        assert registry.audit_session_key(CLAUDE_KEY) == SPAWN_KEY

    def test_a_write_after_rekey_reaches_the_key_the_child_exported(
        self, registry, app, store_file
    ):
        """The alias's whole contract, exercised through the two seams.

        A posture POST addresses the session by its *current* id. The entry it
        writes must nonetheless appear under the key the live child stamped
        into its own environment, or the child looks up a posture that is not
        there and writes on regardless.
        """
        _pooled(registry, SPAWN_KEY)
        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)

        route_session_id = CLAUDE_KEY
        websocket_routes.persist_or_raise(app, route_session_id, NARROWED)

        child_key = registry.audit_session_key(route_session_id)
        assert child_key != route_session_id
        assert _written(store_file)[child_key] == NARROWED

    def test_unknown_key_resolves_to_itself(self, registry):
        """No rekey means the key already is the spawn key.

        The overwhelmingly common case — a resumed session, a chat key, a
        session whose UUID was known up front — must need no bookkeeping.
        """
        assert registry.audit_session_key(SPAWN_KEY) == SPAWN_KEY

    def test_double_rekey_still_names_the_original_spawn_key(self, registry):
        """Chained renames collapse to the first key, not the previous one."""
        _pooled(registry, SPAWN_KEY)

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)
        registry.rekey_session(CLAUDE_KEY, OTHER_KEY)

        assert registry.audit_session_key(OTHER_KEY) == SPAWN_KEY
        assert registry.audit_session_key(CLAUDE_KEY) == CLAUDE_KEY

    def test_rekey_back_to_the_spawn_key_drops_the_alias(self, registry):
        """A round trip leaves no identity alias behind to reason about."""
        _pooled(registry, SPAWN_KEY)

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)
        registry.rekey_session(CLAUDE_KEY, SPAWN_KEY)

        assert registry.audit_session_key(SPAWN_KEY) == SPAWN_KEY
        assert registry._audit_keys == {}

    def test_repeated_rekey_to_the_same_key_is_idempotent(self, registry):
        """The second call finds nothing to move and must not shift the alias."""
        _pooled(registry, SPAWN_KEY)

        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)
        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)

        assert registry.audit_session_key(CLAUDE_KEY) == SPAWN_KEY


class TestAliasLifetime:
    """The alias describes a live child and dies with it."""

    def test_terminate_forgets_the_alias(self, registry):
        _pooled(registry, SPAWN_KEY)
        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)

        registry.terminate_session(CLAUDE_KEY)

        assert registry.audit_session_key(CLAUDE_KEY) == CLAUDE_KEY

    def test_respawn_under_the_new_key_is_its_own_spawn_key(self, registry):
        """The real sequence after a session ends and comes back.

        A *fresh* child exports the Claude UUID as its own
        ``OSPREY_POSTURE_SESSION``, so resolving that key back to the dead
        child's telemetry id would write its narrowing under a key nothing
        reads.
        """
        _pooled(registry, SPAWN_KEY)
        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)
        registry.terminate_session(CLAUDE_KEY)

        with patch.object(registry, "_spawn_session", return_value=_mock_session()):
            registry.get_or_create_session(CLAUDE_KEY, ["claude"], 24, 80)

        assert registry.audit_session_key(CLAUDE_KEY) == CLAUDE_KEY

    def test_a_write_after_the_alias_is_gone_names_one_key(self, registry, app, store_file):
        """No live child to keep joined, so no second entry to write."""
        _pooled(registry, SPAWN_KEY)
        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)
        registry.terminate_session(CLAUDE_KEY)

        websocket_routes.persist_or_raise(app, CLAUDE_KEY, NARROWED)

        assert _written(store_file) == {CLAUDE_KEY: NARROWED}

    def test_eviction_forgets_the_alias(self, registry):
        _pooled(registry, SPAWN_KEY)
        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)
        registry._sessions["b"] = _mock_session()
        registry._sessions["c"] = _mock_session()

        with patch.object(registry, "_spawn_session", return_value=_mock_session()):
            registry.get_or_create_session("d", ["claude"], 24, 80)

        assert CLAUDE_KEY not in registry._sessions
        assert registry.audit_session_key(CLAUDE_KEY) == CLAUDE_KEY

    def test_cleanup_all_clears_the_alias_map(self, registry):
        _pooled(registry, SPAWN_KEY)
        registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)

        registry.cleanup_all()

        assert registry._audit_keys == {}


class TestTheDiscoveryCallSite:
    """``_discover_and_notify`` is where a real session is rekeyed.

    It drives the shipped coroutine rather than :meth:`rekey_session` directly,
    so the call site's own contract — rename the pool, tell the client, leave
    the store alone — is pinned where it can actually regress.
    """

    @pytest.mark.asyncio
    async def test_discovery_renames_the_pool_and_leaves_the_store_alone(
        self, registry, app, store_file
    ):
        _pooled(registry, SPAWN_KEY)
        websocket_routes._session_postures(app)[SPAWN_KEY] = dict(NARROWED)
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
        assert registry.get_session(CLAUDE_KEY) is not None
        assert websocket_routes._session_postures(app) == {SPAWN_KEY: NARROWED}
        # ...and the entry still answers under the id the client now uses.
        assert websocket_routes._posture_entry(app, CLAUDE_KEY) == NARROWED

    @pytest.mark.asyncio
    async def test_a_discovery_that_finds_nothing_leaves_the_store_alone(
        self, registry, app, store_file
    ):
        """No rekey: the session is still pooled under its spawn key."""
        _pooled(registry, SPAWN_KEY)
        websocket_routes._session_postures(app)[SPAWN_KEY] = dict(NARROWED)

        async def _send_text(payload):  # pragma: no cover - must not be reached
            raise AssertionError("nothing was discovered; the client is told nothing")

        websocket = SimpleNamespace(app=app, send_text=_send_text)
        discovery = SimpleNamespace(discover_new_session=lambda snapshot, timeout: None)

        found = await websocket_routes._discover_and_notify(
            set(), discovery, registry, SPAWN_KEY, websocket
        )

        assert found is None
        assert websocket_routes._session_postures(app) == {SPAWN_KEY: NARROWED}
