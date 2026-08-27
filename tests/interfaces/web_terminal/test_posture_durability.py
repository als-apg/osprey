"""Tests for what the session-posture store is allowed to restore from disk.

The store spans restarts on purpose — a container recreation must not silently
revert a sandboxed session to writes. But not every key it can hold is durable,
because not every key names something that can exist again after the process
that minted it is gone. Three key shapes reach the store and they split:

* **PTY session UUID** — a Claude session-file stem, written to disk and still
  there after a restart. Durable.
* **Chat ``chat_id``** — minted by the browser and persisted per page load, so
  a reload respawns the pooled session under the *same* id. Durable: the
  restored posture legitimately governs that later spawn.
* **``operator-<hex8>``** — minted in ``/ws/operator`` when the websocket is
  accepted, held only in this process's operator registry, addressable by
  nothing else. **Not durable.** A restored one can never name a live session,
  so it is dead weight that grows without bound and a latent way for a future
  key collision to hand a fresh connection a stranger's posture.

``_load_postures`` is the single enforcement point: it drops ``operator-``
keys on the way in. The write side deliberately keeps writing them — the
in-memory entries are live for the rest of the process's life — so these tests
pin both halves, and pin that the in-process behaviour of an operator posture
is untouched.

The narrowing is deliberate and documented: durability for the operator half
stays out of scope until an operator client exists to define its reconnect
protocol.

Harness mirrors ``test_posture_routes.py``: each test builds its own app
through ``create_app`` under a patched ``_load_web_config``, entered as a
``TestClient`` context manager so the lifespan runs.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.operator_session import build_operator_child_env
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey.interfaces.web_terminal.routes.websocket import (
    _load_postures,
    _persist_postures,
    _session_postures,
)

# A PTY session key: a Claude session-file stem.
SESSION_A = "aaaaaaaa-1111-2222-3333-444444444444"
# A chat-pool key, minted the way the shipped client mints one
# (``crypto.randomUUID()`` in static/js/chat.js): a bare lowercase UUID.
CHAT_A = "cccccccc-1111-2222-3333-444444444444"
# An operator key, minted the way ``/ws/operator`` mints one:
# ``f"operator-{uuid.uuid4().hex[:8]}"``.
OPERATOR_A = "operator-0123abcd"
OPERATOR_B = "operator-89abcdef"

STORE_NAME = websocket_routes._POSTURE_STORE_NAME


def _write_store(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ── Unit level: the load-side filter ─────────────────────────────────────────


class TestLoadDropsOperatorKeys:
    def test_operator_key_is_not_restored(self, tmp_path):
        """The one property the task exists for."""
        store = _write_store(
            tmp_path / STORE_NAME,
            {OPERATOR_A: "sandbox", SESSION_A: "sandbox"},
        )

        loaded = _load_postures(store)

        assert OPERATOR_A not in loaded
        assert loaded == {SESSION_A: "sandbox"}

    def test_durable_key_shapes_survive(self, tmp_path):
        """PTY and chat keys are durable and must not be caught by the filter.

        A chat id is client-persisted per page load, so its posture governs a
        respawn under the same id after a restart — dropping it would be the
        silent revert the store exists to prevent.
        """
        store = _write_store(
            tmp_path / STORE_NAME,
            {SESSION_A: "sandbox", CHAT_A: "writes"},
        )

        assert _load_postures(store) == {SESSION_A: "sandbox", CHAT_A: "writes"}

    def test_a_store_of_only_operator_keys_loads_empty(self, tmp_path):
        """Not an error — just nothing worth restoring."""
        store = _write_store(
            tmp_path / STORE_NAME,
            {OPERATOR_A: "sandbox", OPERATOR_B: "writes"},
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
        posture, which is the failure this whole store guards against.
        """
        store = _write_store(tmp_path / STORE_NAME, {key: "sandbox"})

        assert _load_postures(store) == {key: "sandbox"}

    def test_value_filter_still_applies_to_operator_keys(self, tmp_path):
        """The two filters compose; neither shadows the other."""
        store = _write_store(
            tmp_path / STORE_NAME,
            {OPERATOR_A: "bogus-posture", SESSION_A: "bogus-posture", CHAT_A: "sandbox"},
        )

        assert _load_postures(store) == {CHAT_A: "sandbox"}

    def test_absent_and_corrupt_stores_are_unchanged(self, tmp_path):
        """The added filter must not disturb the existing tolerance."""
        assert _load_postures(tmp_path / "nonexistent.json") == {}

        corrupt = tmp_path / STORE_NAME
        corrupt.write_text("{not json", encoding="utf-8")
        assert _load_postures(corrupt) == {}

        not_an_object = tmp_path / "list.json"
        not_an_object.write_text("[]", encoding="utf-8")
        assert _load_postures(not_an_object) == {}


# ── Unit level: the write side stays permissive ──────────────────────────────


class TestPersistKeepsWritingOperatorKeys:
    def test_persisting_an_operator_key_does_not_crash(self, tmp_path):
        """The load side is the *single* enforcement point.

        Filtering on the way out too would put the rule in two places for no
        gain: the in-memory entry is live and load-bearing until the process
        ends, and only the restore can act on a dead key.
        """
        app = SimpleNamespace(state=SimpleNamespace(workspace_dir=tmp_path))
        store_dir = tmp_path / "shared"
        store_dir.mkdir()

        with patch(
            "osprey_connectors.workspace.resolve_shared_data_root",
            return_value=store_dir,
        ):
            _persist_postures(app, {OPERATOR_A: "sandbox", SESSION_A: "sandbox"})

        written = json.loads((store_dir / STORE_NAME).read_text(encoding="utf-8"))
        assert written == {OPERATOR_A: "sandbox", SESSION_A: "sandbox"}

    def test_round_trip_drops_the_operator_key(self, tmp_path):
        """Write-then-read is where the entry disappears, and only there."""
        app = SimpleNamespace(state=SimpleNamespace(workspace_dir=tmp_path))
        store_dir = tmp_path / "shared"
        store_dir.mkdir()

        with patch(
            "osprey_connectors.workspace.resolve_shared_data_root",
            return_value=store_dir,
        ):
            _persist_postures(app, {OPERATOR_A: "sandbox", SESSION_A: "sandbox"})

        assert _load_postures(store_dir / STORE_NAME) == {SESSION_A: "sandbox"}


# ── Route level: a seeded store, through a real app ──────────────────────────


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
def make_client(workspace_dir, shared_root):
    """Build an app + TestClient over the same shared root."""

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
    def test_seeded_operator_key_never_reaches_the_live_store(
        self, make_client, shared_root, monkeypatch
    ):
        """A store file left by a previous process, read by a fresh app.

        This is the real shape of the bug: the previous process persisted its
        operator postures, the container was recreated, and the new process
        must not carry keys its own operator registry can never mint again.
        """
        monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
        _write_store(
            shared_root / STORE_NAME,
            {OPERATOR_A: "sandbox", SESSION_A: "sandbox", CHAT_A: "sandbox"},
        )

        with make_client() as client:
            store = _session_postures(client.app)

            assert OPERATOR_A not in store
            assert store == {SESSION_A: "sandbox", CHAT_A: "sandbox"}

            # And the spawn seam agrees: the restored operator key governs
            # nothing, while the durable keys still do.
            operator_env = build_operator_child_env(
                client.app.state.project_cwd, session_key=OPERATOR_A, app=client.app
            )
            chat_env = build_operator_child_env(
                client.app.state.project_cwd, session_key=CHAT_A, app=client.app
            )

            assert "OSPREY_EXECUTION_MODE" not in operator_env
            assert chat_env["OSPREY_EXECUTION_MODE"] == "readonly"

    def test_in_memory_operator_posture_still_governs_this_process(
        self, make_client, shared_root, monkeypatch
    ):
        """Non-durable is not the same as ignored.

        Only the restore path changed. A posture set on an operator key while
        the process is alive must still reach that connection's child — the
        key is addressable for exactly as long as the registry that minted it.
        """
        monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)

        with make_client() as client:
            _session_postures(client.app)[OPERATOR_A] = "sandbox"

            env = build_operator_child_env(
                client.app.state.project_cwd, session_key=OPERATOR_A, app=client.app
            )

            assert env["OSPREY_EXECUTION_MODE"] == "readonly"
