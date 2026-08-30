"""Tests for the web server's half of the per-(session, target) posture store.

The store spans restarts on purpose — a container recreation must not silently
revert a narrowed session to writes. Four properties make that true, and they
are all pinned here:

* **The shared parser decides the shape.** Every reader of this file decodes it
  through ``osprey_connectors.session_store.parse_store``: the legacy bare
  ``"sandbox"`` the session-wide posture used to write narrows *every* target,
  a bare ``"writes"`` disappears (absence is how this store spells the writes
  posture), and anything unrecognised is dropped. The web server does not get
  its own filter.

* **``operator-`` keys do not survive a restart.** Three key shapes reach the
  store and they split: a **PTY session UUID** is a Claude session-file stem,
  still on disk after a restart, and durable. A **chat ``chat_id``** is minted
  by the browser and persisted per page load, so a reload respawns the pooled
  session under the same id — also durable. An **``operator-<hex8>``** is
  minted in ``/ws/operator`` when the websocket is accepted and held only in
  this process's registry, so a restored one can never name a live session: it
  is dead weight that grows without bound and a latent way for a future key
  collision to hand a fresh connection a stranger's narrowing. The drop is
  applied on the web server's startup load and nowhere else — an enforcement
  reader that dropped them would ignore a narrowing that is live for the rest
  of that process's life.

* **The persist is the commit point.** ``persist_or_raise`` writes the file
  first and updates memory only once it has landed. Enforcement reads the
  store, so memory and disk disagreeing is not a lost convenience: it is a
  badge that says sandboxed over a session whose next write is still permitted.

* **Every entry is written under both keys.** A rekeyed session's live child
  reads the telemetry id it was spawned with while a post-restart reattach
  reads the Claude UUID; one narrowing has to answer for both.

Route-level tests mirror ``test_posture_routes.py``: each builds its own app
through ``create_app`` under a patched ``_load_web_config``, entered as a
``TestClient`` context manager so the lifespan runs.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.pty_manager import PtyRegistry
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey.interfaces.web_terminal.routes.websocket import (
    PostureStoreUnavailable,
    _load_postures,
    _posture_entry,
    _session_postures,
    persist_or_raise,
)
from osprey_connectors import session_store
from osprey_connectors.types import CONTROL_TARGETS

# A PTY session key: a Claude session-file stem.
SESSION_A = "aaaaaaaa-1111-2222-3333-444444444444"
# A chat-pool key, minted the way the shipped client mints one
# (``crypto.randomUUID()`` in static/js/chat.js): a bare lowercase UUID.
CHAT_A = "cccccccc-1111-2222-3333-444444444444"
# An operator key, minted the way ``/ws/operator`` mints one:
# ``f"operator-{uuid.uuid4().hex[:8]}"``.
OPERATOR_A = "operator-0123abcd"
OPERATOR_B = "operator-89abcdef"

# A rekeyed PTY session: spawned under the telemetry id, renamed on discovery.
SPAWN_KEY = "11111111-1111-4111-8111-111111111111"
CLAUDE_KEY = "22222222-2222-4222-8222-222222222222"

SANDBOX = websocket_routes.POSTURE_SANDBOX
STORE_NAME = session_store.STORE_FILENAME
STATE_DIR = session_store.STATE_DIR_NAME

#: What the legacy bare ``"sandbox"`` means once parsed.
EVERY_TARGET = dict.fromkeys(CONTROL_TARGETS, SANDBOX)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def shared_root(tmp_path, monkeypatch):
    """Pin the agent-data root every reader resolves, the way a spawn does.

    ``OSPREY_AGENT_DATA_ROOT`` is the first half of the store's one resolution
    rule, so setting it exercises the real derivation rather than patching
    around it.
    """
    root = tmp_path / "shared_agent_data"
    root.mkdir()
    monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    session_store.invalidate_cache()
    yield root
    session_store.invalidate_cache()


@pytest.fixture
def store_file(shared_root):
    """The store's real location: ``<root>/control_target/session-postures.json``."""
    return shared_root / STATE_DIR / STORE_NAME


@pytest.fixture
def legacy_file(shared_root):
    """Where the session-wide posture kept its store, directly under the root."""
    return shared_root / STORE_NAME


@pytest.fixture
def app(tmp_path):
    """A bare app object — enough for the store helpers, no server."""
    return SimpleNamespace(state=SimpleNamespace(workspace_dir=tmp_path))


# ── The shared parser decides every shape ────────────────────────────────────


class TestTheSharedParserDecidesTheShape:
    def test_legacy_bare_sandbox_narrows_every_target(self, tmp_path):
        """The upgrade case: a session sandboxed by the session-wide toggle.

        It was written before targets existed and meant "this session writes
        nothing", so it has to keep meaning that — every target, not none.
        """
        store = _write_json(tmp_path / STORE_NAME, {SESSION_A: "sandbox"})

        assert _load_postures(store) == {SESSION_A: EVERY_TARGET}

    def test_bare_writes_is_dropped(self, tmp_path):
        """Absence is the only spelling of the writes posture.

        Nothing in this file may widen anything, so a stored ``"writes"`` is
        not an assertion to honour — it is an entry with nothing in it.
        """
        store = _write_json(tmp_path / STORE_NAME, {SESSION_A: "writes"})

        assert _load_postures(store) == {}

    def test_per_target_entries_keep_only_the_narrowings(self, tmp_path):
        store = _write_json(
            tmp_path / STORE_NAME,
            {SESSION_A: {"live": "sandbox", "va": "writes", "standin": "bogus"}},
        )

        assert _load_postures(store) == {SESSION_A: {"live": SANDBOX}}

    def test_an_entry_that_narrows_nothing_drops_its_key(self, tmp_path):
        store = _write_json(tmp_path / STORE_NAME, {SESSION_A: {}, CHAT_A: {"live": "writes"}})

        assert _load_postures(store) == {}


# ── The load-side filter: operator keys ──────────────────────────────────────


class TestLoadDropsOperatorKeys:
    def test_operator_key_is_not_restored(self, tmp_path):
        """The one property this filter exists for."""
        store = _write_json(
            tmp_path / STORE_NAME,
            {OPERATOR_A: {"live": "sandbox"}, SESSION_A: {"live": "sandbox"}},
        )

        loaded = _load_postures(store)

        assert OPERATOR_A not in loaded
        assert loaded == {SESSION_A: {"live": SANDBOX}}

    def test_durable_key_shapes_survive(self, tmp_path):
        """PTY and chat keys are durable and must not be caught by the filter.

        A chat id is client-persisted per page load, so its narrowing governs a
        respawn under the same id after a restart — dropping it would be the
        silent revert the store exists to prevent.
        """
        store = _write_json(
            tmp_path / STORE_NAME,
            {SESSION_A: {"live": "sandbox"}, CHAT_A: {"va": "sandbox"}},
        )

        assert _load_postures(store) == {
            SESSION_A: {"live": SANDBOX},
            CHAT_A: {"va": SANDBOX},
        }

    def test_a_store_of_only_operator_keys_loads_empty(self, tmp_path):
        """Not an error — just nothing worth restoring."""
        store = _write_json(
            tmp_path / STORE_NAME,
            {OPERATOR_A: "sandbox", OPERATOR_B: {"va": "sandbox"}},
        )

        assert _load_postures(store) == {}

    @pytest.mark.parametrize(
        "key",
        [
            "operator",  # no separator: not the minted shape
            "xoperator-0123abcd",  # prefix is anchored, not a substring match
            "OPERATOR-0123abcd",  # the minted key is lowercase
            "chat-operator-0123abcd",  # contains it, does not start with it
        ],
    )
    def test_filter_matches_only_the_anchored_prefix(self, tmp_path, key):
        """Only ``operator-``-*prefixed* keys are dropped.

        The rule is about the minted key shape, not about the word appearing
        somewhere in a key. Over-matching here would silently discard a durable
        narrowing, which is the failure this whole store guards against.
        """
        store = _write_json(tmp_path / STORE_NAME, {key: {"live": "sandbox"}})

        assert _load_postures(store) == {key: {"live": SANDBOX}}

    def test_value_filter_still_applies_to_operator_keys(self, tmp_path):
        """The two filters compose; neither shadows the other."""
        store = _write_json(
            tmp_path / STORE_NAME,
            {
                OPERATOR_A: "bogus-posture",
                SESSION_A: "bogus-posture",
                CHAT_A: {"live": "sandbox"},
            },
        )

        assert _load_postures(store) == {CHAT_A: {"live": SANDBOX}}

    def test_absent_and_corrupt_stores_are_empty(self, tmp_path):
        """A file nobody can repair from the browser must not wedge the server."""
        assert _load_postures(tmp_path / "nonexistent.json") == {}

        corrupt = tmp_path / STORE_NAME
        corrupt.write_text("{not json", encoding="utf-8")
        assert _load_postures(corrupt) == {}

        not_an_object = tmp_path / "list.json"
        not_an_object.write_text("[]", encoding="utf-8")
        assert _load_postures(not_an_object) == {}


# ── The persist is the commit point ──────────────────────────────────────────


class TestPersistIsTheCommitPoint:
    def test_a_narrowing_lands_on_disk_and_in_memory(self, app, store_file):
        stored = persist_or_raise(app, SESSION_A, {"live": "sandbox"})

        assert stored == {"live": SANDBOX}
        assert json.loads(store_file.read_text(encoding="utf-8")) == {SESSION_A: {"live": SANDBOX}}
        assert _posture_entry(app, SESSION_A) == {"live": SANDBOX}

    def test_the_store_is_co_sited_with_the_state_file(self, app, shared_root, store_file):
        """``<root>/control_target/`` — the directory the state file uses.

        A store the writer puts in one directory and a reader looks for in
        another is a narrowing that silently never applies.
        """
        persist_or_raise(app, SESSION_A, "sandbox")

        assert store_file.exists()
        assert store_file.parent == shared_root / STATE_DIR

    def test_a_bare_posture_is_normalised_through_the_shared_parser(self, app, store_file):
        """A route may hand over the legacy word and gets the shared reading."""
        stored = persist_or_raise(app, SESSION_A, "sandbox")

        assert stored == EVERY_TARGET
        assert json.loads(store_file.read_text(encoding="utf-8")) == {SESSION_A: EVERY_TARGET}

    def test_an_empty_entry_removes_the_key(self, app, store_file):
        """Absence is how this store spells writes; a stored ``{}`` would be a second spelling."""
        persist_or_raise(app, SESSION_A, "sandbox")

        assert persist_or_raise(app, SESSION_A, {}) == {}

        assert json.loads(store_file.read_text(encoding="utf-8")) == {}
        assert _posture_entry(app, SESSION_A) == {}

    def test_no_store_location_refuses_and_changes_nothing(self, app, shared_root):
        """``store_unavailable``: there is nowhere a reader would find this."""
        persist_or_raise(app, SESSION_A, {"live": "sandbox"})
        before = dict(_session_postures(app))

        with (
            patch.object(session_store, "store_path", return_value=None),
            pytest.raises(PostureStoreUnavailable) as excinfo,
        ):
            persist_or_raise(app, CHAT_A, {"live": "sandbox"})

        assert excinfo.value.error == "store_unavailable"
        assert _session_postures(app) == before

    def test_a_failed_write_refuses_and_leaves_memory_unchanged(self, app, store_file):
        """The write is the commit point, so a failed write changed nothing.

        Enforcement reads this file. A posture kept in memory that never
        reached disk is a badge saying sandboxed over a session whose next
        write is still permitted — worse than the 503 the operator can retry.
        """
        persist_or_raise(app, SESSION_A, {"live": "sandbox"})
        before = dict(_session_postures(app))
        on_disk = store_file.read_text(encoding="utf-8")

        with (
            patch.object(
                websocket_routes, "_atomic_write_json", side_effect=OSError("read-only fs")
            ),
            pytest.raises(PostureStoreUnavailable) as excinfo,
        ):
            persist_or_raise(app, SESSION_A, {"va": "sandbox"})

        assert excinfo.value.error == "store_write_failed"
        assert _session_postures(app) == before
        assert store_file.read_text(encoding="utf-8") == on_disk

    def test_a_read_only_directory_refuses_the_same_way(self, app, shared_root, store_file):
        """The real failure, not a patched one: nothing may be written here."""
        state_dir = shared_root / STATE_DIR
        state_dir.mkdir(parents=True)
        state_dir.chmod(0o500)
        try:
            with pytest.raises(PostureStoreUnavailable) as excinfo:
                persist_or_raise(app, SESSION_A, {"live": "sandbox"})
        finally:
            state_dir.chmod(0o700)

        assert excinfo.value.error == "store_write_failed"
        assert not store_file.exists()
        assert _session_postures(app) == {}

    def test_other_sessions_survive_a_write(self, app, store_file):
        persist_or_raise(app, SESSION_A, {"live": "sandbox"})
        persist_or_raise(app, CHAT_A, {"va": "sandbox"})

        assert json.loads(store_file.read_text(encoding="utf-8")) == {
            SESSION_A: {"live": SANDBOX},
            CHAT_A: {"va": SANDBOX},
        }


class TestPersistKeepsWritingOperatorKeys:
    def test_persisting_an_operator_key_does_not_crash(self, app, store_file):
        """The load side is the *single* enforcement point.

        Filtering on the way out too would put the rule in two places for no
        gain: the in-memory entry is live and load-bearing until the process
        ends, and only the restore can act on a dead key.
        """
        persist_or_raise(app, OPERATOR_A, {"live": "sandbox"})
        persist_or_raise(app, SESSION_A, {"live": "sandbox"})

        assert json.loads(store_file.read_text(encoding="utf-8")) == {
            OPERATOR_A: {"live": SANDBOX},
            SESSION_A: {"live": SANDBOX},
        }

    def test_round_trip_drops_the_operator_key(self, app, store_file):
        """Write-then-read is where the entry disappears, and only there."""
        persist_or_raise(app, OPERATOR_A, {"live": "sandbox"})
        persist_or_raise(app, SESSION_A, {"live": "sandbox"})

        assert _load_postures(store_file) == {SESSION_A: {"live": SANDBOX}}


# ── One narrowing, both keys ─────────────────────────────────────────────────


def _rekeyed_app(tmp_path) -> tuple[SimpleNamespace, PtyRegistry]:
    """An app whose PTY session was rekeyed from SPAWN_KEY to CLAUDE_KEY."""
    registry = PtyRegistry(max_background=3)
    session = MagicMock()
    session.is_alive = True
    registry._sessions[SPAWN_KEY] = session
    registry.rekey_session(SPAWN_KEY, CLAUDE_KEY)
    app = SimpleNamespace(state=SimpleNamespace(workspace_dir=tmp_path, pty_registry=registry))
    return app, registry


class TestDualKeyWrite:
    """A rekeyed session's live child and its next reattach read one narrowing."""

    def test_a_write_after_a_rekey_lands_under_both_keys(self, tmp_path, store_file):
        app, registry = _rekeyed_app(tmp_path)
        assert registry.audit_session_key(CLAUDE_KEY) == SPAWN_KEY

        persist_or_raise(app, CLAUDE_KEY, {"live": "sandbox"})

        assert json.loads(store_file.read_text(encoding="utf-8")) == {
            CLAUDE_KEY: {"live": SANDBOX},
            SPAWN_KEY: {"live": SANDBOX},
        }

    def test_both_readers_find_the_entry(self, tmp_path, store_file):
        """The live child reads its spawn key; a reattach reads the Claude UUID."""
        app, _ = _rekeyed_app(tmp_path)

        persist_or_raise(app, CLAUDE_KEY, {"live": "sandbox"})

        assert _posture_entry(app, CLAUDE_KEY) == {"live": SANDBOX}
        assert _posture_entry(app, SPAWN_KEY) == {"live": SANDBOX}

    def test_clearing_removes_both_keys(self, tmp_path, store_file):
        """A half-cleared session would keep narrowing whichever reader kept its key."""
        app, _ = _rekeyed_app(tmp_path)
        persist_or_raise(app, CLAUDE_KEY, {"live": "sandbox"})

        persist_or_raise(app, CLAUDE_KEY, {})

        assert json.loads(store_file.read_text(encoding="utf-8")) == {}
        assert _posture_entry(app, SPAWN_KEY) == {}

    def test_a_session_that_never_moved_writes_one_key(self, app, store_file):
        """The common case must not grow a duplicate entry."""
        persist_or_raise(app, SESSION_A, {"live": "sandbox"})

        assert list(json.loads(store_file.read_text(encoding="utf-8"))) == [SESSION_A]

    def test_the_read_prefers_the_current_key(self, tmp_path, store_file):
        """Current key first: the entry a route just wrote is the one it reads back.

        The two only differ while a store written before this rule is still
        around, but the order has to be pinned — a stale spawn-key entry
        winning would answer for a narrowing the operator has already changed.
        """
        app, _ = _rekeyed_app(tmp_path)
        store = _session_postures(app)
        store[SPAWN_KEY] = {"live": SANDBOX}
        store[CLAUDE_KEY] = {"va": SANDBOX}

        assert _posture_entry(app, CLAUDE_KEY) == {"va": SANDBOX}

    def test_an_unknown_key_answers_for_itself(self, app):
        assert _posture_entry(app, SESSION_A) == {}
        assert _posture_entry(app, None) == {}


# ── The one-shot migration off the old path ──────────────────────────────────


class TestMigrationFromTheOldPath:
    def test_an_old_store_is_migrated_and_written_to_the_new_path(
        self, app, legacy_file, store_file
    ):
        """A deployment upgrading with a sandboxed session live.

        The narrowing must not be lifted the moment the code lands, so the old
        path is read through once; and it must stop being read, so the migrated
        shape is written to the new path straight away.
        """
        _write_json(legacy_file, {SESSION_A: "sandbox", OPERATOR_A: "sandbox"})

        store = _session_postures(app)

        assert store == {SESSION_A: EVERY_TARGET}
        assert json.loads(store_file.read_text(encoding="utf-8")) == {SESSION_A: EVERY_TARGET}

    def test_the_old_file_is_left_alone(self, app, legacy_file, store_file):
        """A rollback has to find it. Nothing here writes to the old path."""
        payload = {SESSION_A: "sandbox"}
        _write_json(legacy_file, payload)

        _session_postures(app)

        assert json.loads(legacy_file.read_text(encoding="utf-8")) == payload

    def test_the_new_path_wins_once_it_exists(self, app, legacy_file, store_file):
        """The read-through is one-shot: the old file stops answering."""
        _write_json(legacy_file, {SESSION_A: "sandbox"})
        _write_json(store_file, {CHAT_A: {"va": "sandbox"}})

        assert _session_postures(app) == {CHAT_A: {"va": SANDBOX}}

    def test_an_old_store_that_narrows_nothing_writes_no_file(self, app, legacy_file, store_file):
        """There is no such thing as an empty entry worth migrating."""
        _write_json(legacy_file, {SESSION_A: "writes", OPERATOR_A: "sandbox"})

        assert _session_postures(app) == {}
        assert not store_file.exists()

    def test_no_old_store_at_all_is_an_empty_store(self, app, store_file):
        assert _session_postures(app) == {}
        assert not store_file.exists()

    def test_a_failed_migration_write_still_serves_the_migrated_store(
        self, app, legacy_file, store_file
    ):
        """The old file still answers until the write lands, so nothing is lost."""
        _write_json(legacy_file, {SESSION_A: "sandbox"})

        with patch.object(
            websocket_routes, "_atomic_write_json", side_effect=OSError("read-only fs")
        ):
            store = _session_postures(app)

        assert store == {SESSION_A: EVERY_TARGET}
        assert not store_file.exists()


class TestNoStoreLocation:
    def test_a_location_less_load_stays_provisional(self, app):
        """One transient config failure must not outlive itself.

        Caching an empty store because the root did not resolve once would
        report a narrowed session as unnarrowed for the life of the process.
        """
        with patch.object(session_store, "store_path", return_value=None):
            assert _session_postures(app) == {}
            assert app.state.session_postures_provisional is True

    def test_the_next_access_re_reads_once_the_root_is_back(self, app, store_file):
        with patch.object(session_store, "store_path", return_value=None):
            _session_postures(app)

        _write_json(store_file, {SESSION_A: {"live": "sandbox"}})

        assert _session_postures(app) == {SESSION_A: {"live": SANDBOX}}
        assert app.state.session_postures_provisional is False

    def test_a_narrowing_set_during_the_outage_wins_the_recovery_read(self, app, store_file):
        """Memory is the operator's newer intent; the file is what was there."""
        with patch.object(session_store, "store_path", return_value=None):
            _session_postures(app)[SESSION_A] = {"va": SANDBOX}

        _write_json(store_file, {SESSION_A: {"live": "sandbox"}, CHAT_A: {"live": "sandbox"}})

        assert _session_postures(app) == {
            SESSION_A: {"va": SANDBOX},
            CHAT_A: {"live": SANDBOX},
        }


# ── Route level: a seeded store, through a real app ──────────────────────────


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


@pytest.fixture
def make_client(workspace_dir, shared_root):
    """Build an app + TestClient over the same agent-data root."""

    @contextmanager
    def _make():
        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ):
            app = create_app(shell_command="echo")
            with TestClient(app) as client:
                yield client

    return _make


class TestRestartDropsOperatorPostures:
    def test_seeded_operator_key_never_reaches_the_live_store(self, make_client, store_file):
        """A store file left by a previous process, read by a fresh app.

        This is the real shape of the bug: the previous process persisted its
        operator narrowings, the container was recreated, and the new process
        must not carry keys its own operator registry can never mint again.
        """
        _write_json(
            store_file,
            {
                OPERATOR_A: {"live": "sandbox"},
                SESSION_A: {"live": "sandbox"},
                CHAT_A: "sandbox",
            },
        )

        with make_client() as client:
            store = _session_postures(client.app)

            assert OPERATOR_A not in store
            assert store == {SESSION_A: {"live": SANDBOX}, CHAT_A: EVERY_TARGET}
            assert _posture_entry(client.app, OPERATOR_A) == {}

    def test_in_memory_operator_posture_still_governs_this_process(self, make_client, store_file):
        """Non-durable is not the same as ignored.

        Only the restore path drops these. A narrowing set on an operator key
        while the process is alive must still answer for that connection — the
        key is addressable for exactly as long as the registry that minted it.
        """
        with make_client() as client:
            persist_or_raise(client.app, OPERATOR_A, {"live": "sandbox"})

            assert _posture_entry(client.app, OPERATOR_A) == {"live": SANDBOX}
            assert json.loads(store_file.read_text(encoding="utf-8")) == {
                OPERATOR_A: {"live": SANDBOX}
            }
