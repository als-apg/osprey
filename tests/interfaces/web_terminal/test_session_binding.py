"""Tests for the session binding a notebook kernel joins a PTY session by.

The binding is a single document with one rule: it names the PTY session most
recently attached. Everything here pins one half of that rule — the write that
moves it forward, the ownership check that stops a superseded session from
taking it away, and the tolerant read a kernel process relies on when the
document is absent or damaged.

The reader lives in :mod:`osprey.jupyter_kernel` and is standard library only,
because a kernel must not import the web terminal to find out which session it
belongs to. Tests exercise both ends against the same directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from osprey.interfaces.web_terminal import pty_manager
from osprey.interfaces.web_terminal.session_binding import clear_binding, write_binding
from osprey.jupyter_kernel import BINDING_RELPATH, binding_path, read_binding


def _document(root: Path) -> dict:
    return json.loads((root / BINDING_RELPATH).read_text())


# ── write_binding ──────────────────────────────────────────────────────────


def test_binding_lands_at_the_shared_path(tmp_path: Path) -> None:
    write_binding(tmp_path, "sess-1", 4321, str(tmp_path))

    assert (tmp_path / "jupyter" / "session-binding.json").is_file()
    assert binding_path(tmp_path) == tmp_path / BINDING_RELPATH


def test_binding_records_identity_only(tmp_path: Path) -> None:
    write_binding(tmp_path, "sess-1", 4321, "/srv/agent_data")

    assert _document(tmp_path) == {
        "session_id": "sess-1",
        "pty_pid": 4321,
        "agent_data_root": "/srv/agent_data",
    }


def test_binding_names_the_session_attached_last(tmp_path: Path) -> None:
    write_binding(tmp_path, "sess-1", 111, str(tmp_path))
    write_binding(tmp_path, "sess-2", 222, str(tmp_path))

    assert _document(tmp_path)["session_id"] == "sess-2"
    assert _document(tmp_path)["pty_pid"] == 222


def test_binding_write_survives_an_unusable_root(tmp_path: Path) -> None:
    # A file where the root should be: the directory creation cannot succeed.
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("")

    write_binding(blocked, "sess-1", 111, str(blocked))

    assert not binding_path(blocked).exists()


# ── clear_binding ──────────────────────────────────────────────────────────


def test_clearing_the_bound_session_removes_the_document(tmp_path: Path) -> None:
    write_binding(tmp_path, "sess-1", 111, str(tmp_path))

    clear_binding(tmp_path, "sess-1")

    assert not binding_path(tmp_path).exists()


def test_clearing_another_session_leaves_the_document(tmp_path: Path) -> None:
    write_binding(tmp_path, "sess-1", 111, str(tmp_path))
    write_binding(tmp_path, "sess-2", 222, str(tmp_path))

    # sess-1 was superseded before it died; the live binding is not its to drop.
    clear_binding(tmp_path, "sess-1")

    assert _document(tmp_path)["session_id"] == "sess-2"


def test_clearing_without_a_binding_is_silent(tmp_path: Path) -> None:
    clear_binding(tmp_path, "sess-1")

    assert not binding_path(tmp_path).exists()


def test_clearing_a_damaged_binding_leaves_it_alone(tmp_path: Path) -> None:
    path = binding_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{not json")

    clear_binding(tmp_path, "sess-1")

    # Nothing names sess-1, so nothing is claimed — and a damaged document is
    # not evidence that it belonged to the session being torn down.
    assert path.read_text() == "{not json"


# ── read_binding ───────────────────────────────────────────────────────────


def test_read_binding_round_trips_a_write(tmp_path: Path) -> None:
    write_binding(tmp_path, "sess-1", 4321, "/srv/agent_data")

    assert read_binding(tmp_path) == {
        "session_id": "sess-1",
        "pty_pid": 4321,
        "agent_data_root": "/srv/agent_data",
    }


def test_read_binding_answers_none_when_there_is_none(tmp_path: Path) -> None:
    assert read_binding(tmp_path) is None


@pytest.mark.parametrize("payload", ["{not json", '"a string"', "[1, 2]", ""])
def test_read_binding_answers_none_for_an_unusable_document(tmp_path: Path, payload: str) -> None:
    path = binding_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(payload)

    assert read_binding(tmp_path) is None


# ── PTY teardown ───────────────────────────────────────────────────────────


@pytest.fixture
def bound_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the registry's binding teardown at ``tmp_path``."""
    monkeypatch.setattr(
        "osprey.interfaces.web_terminal.operator_session.resolve_agent_data_root",
        lambda app=None: str(tmp_path),
    )
    return tmp_path


def test_terminating_the_bound_session_clears_the_binding(bound_root: Path) -> None:
    registry = pty_manager.PtyRegistry()
    registry._sessions["sess-1"] = MagicMock()
    write_binding(bound_root, "sess-1", 111, str(bound_root))

    registry.terminate_session("sess-1")

    assert not binding_path(bound_root).exists()


def test_terminating_another_session_leaves_the_binding(bound_root: Path) -> None:
    registry = pty_manager.PtyRegistry()
    registry._sessions["sess-1"] = MagicMock()
    registry._sessions["sess-2"] = MagicMock()
    write_binding(bound_root, "sess-2", 222, str(bound_root))

    registry.terminate_session("sess-1")

    assert _document(bound_root)["session_id"] == "sess-2"


def test_terminating_a_rekeyed_session_clears_its_spawn_key(bound_root: Path) -> None:
    registry = pty_manager.PtyRegistry()
    registry._sessions["claude-uuid"] = MagicMock()
    registry._audit_keys["claude-uuid"] = "telemetry-id"
    # The binding names the key the child stamps, not the current pool key.
    write_binding(bound_root, "telemetry-id", 333, str(bound_root))

    registry.terminate_session("claude-uuid")

    assert not binding_path(bound_root).exists()


def test_evicting_a_background_session_clears_its_binding(bound_root: Path) -> None:
    registry = pty_manager.PtyRegistry(max_background=1)
    registry._sessions["sess-1"] = MagicMock()
    write_binding(bound_root, "sess-1", 111, str(bound_root))

    registry._evict_lru()

    assert "sess-1" not in registry._sessions
    assert not binding_path(bound_root).exists()


# ── the attach-time gate ───────────────────────────────────────────────────


def _attach(tmp_path: Path, enabled: set[str]) -> Path:
    """Run one attach against an app serving *enabled*, and return its root.

    Args:
        tmp_path: The throwaway agent-data root the attach should resolve to.
        enabled: The panel ids the app has enabled.

    Returns:
        The root, so the caller can assert on what the attach left in it.
    """
    from osprey.interfaces.web_terminal.routes import websocket

    app = MagicMock()
    app.state.enabled_panels = enabled
    registry = MagicMock()
    registry.audit_session_key.return_value = "sess-1"
    session = MagicMock()
    session.pid = 4321

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(websocket, "resolve_agent_data_root", lambda app=None: str(tmp_path))
        websocket._bind_notebook_session(app, registry, session, "sess-1")
    return tmp_path


def test_attaching_writes_the_binding_where_the_panel_is_served(tmp_path: Path) -> None:
    _attach(tmp_path, {"artifacts", "jupyter"})

    assert _document(tmp_path)["session_id"] == "sess-1"


def test_attaching_writes_nothing_where_the_panel_is_absent(tmp_path: Path) -> None:
    """No notebook panel, no notebook state — not even the directory.

    Nothing would ever read the binding in such a deployment, and creating
    ``<root>/jupyter/`` under a root the server only guessed at is how an
    empty directory ends up in a tree that has nothing to do with notebooks.
    """
    _attach(tmp_path, {"artifacts"})

    assert not binding_path(tmp_path).exists()
    assert not (tmp_path / "jupyter").exists()
