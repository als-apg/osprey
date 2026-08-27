"""Tests for the per-session runtime posture store and its POST route.

The posture is the operator's per-session sandbox toggle: step a live session
into ``sandbox`` (the child respawns with ``OSPREY_EXECUTION_MODE=readonly``)
and back out to ``writes``. Three properties matter and each has tests here:

* **The store is the truth.** ``POST /api/terminal/posture`` records the
  intent, and ``_build_extra_env`` — the one seam that builds a PTY child's
  environment — reads it back on the next spawn. A posture that the store
  holds but the child does not carry would be a badge that lies.
* **It survives a restart.** A container recreation must never silently
  revert a sandboxed session to writes, so the store is write-through
  persisted beside the other agent-data stores and re-read by a fresh app.
* **It cannot grant what the render withholds.** Stepping *out* to ``writes``
  is refused when the render arms no control target for writes — neither a
  per-type ``control_system.connector.<type>.writes_enabled`` nor the
  deployment-wide ``control_system.writes_enabled`` they inherit from. The
  toggle narrows privilege, it never widens it.

Harness mirrors ``test_logout_route.py``: each test file builds its own app
through ``create_app`` under a patched ``_load_web_config``, entered as a
``TestClient`` context manager so the lifespan runs.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

from osprey.interfaces.web_auth import PANEL_TOKEN_ENV, get_web_credentials
from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.operator_session import (
    POSTURE_SESSION_ENV,
    POSTURE_SOURCE_ENV,
    build_operator_child_env,
)
from osprey.interfaces.web_terminal.routes import chat as chat_routes
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes

SESSION_A = "aaaaaaaa-1111-2222-3333-444444444444"
SESSION_B = "bbbbbbbb-1111-2222-3333-444444444444"
# A chat-pool key, minted the way the shipped client mints one
# (``crypto.randomUUID()`` in static/js/chat.js): a bare lowercase UUID.
CHAT_A = "cccccccc-1111-2222-3333-444444444444"


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


@pytest.fixture
def shared_root(tmp_path):
    """Stand in for the deployment's shared agent-data root.

    The posture store is sited on ``resolve_shared_data_root()`` — never a
    hard-coded ``var/agent_data`` and never ``resolve_agent_data_root()``,
    which appends ``sessions/<id>`` and would scope the store to a single
    session. Patching it here keeps the tests off the real repo tree and lets
    two app instances share one store directory.
    """
    root = tmp_path / "shared_agent_data"
    root.mkdir()
    with patch(
        "osprey_connectors.workspace.resolve_shared_data_root",
        return_value=root,
    ):
        yield root


@pytest.fixture
def make_client(workspace_dir, shared_root):
    """Build an app + TestClient, repeatably, over the same shared root."""

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


@pytest.fixture
def client(make_client):
    with make_client() as c:
        yield c


@contextmanager
def known_sessions(*session_ids):
    """Make ``SessionDiscovery`` report *session_ids* as started on disk.

    A posture can only be set on a session that already exists — the id the
    route is handed must name a real Claude session file, which is only
    written once the operator has sent a prompt.
    """
    with patch(
        "osprey.interfaces.web_terminal.session_discovery.SessionDiscovery.snapshot_session_ids",
        return_value=set(session_ids),
    ):
        yield


@contextmanager
def config_outage(shared_root):
    """Make ``resolve_shared_data_root()`` raise until ``outage.over()``.

    Stands in for a transient config-load failure: while it lasts, the posture
    store resolves to the workspace-dir fallback instead of the shared agent
    data root.
    """
    state = SimpleNamespace(active=True, over=lambda: None)
    state.over = lambda: setattr(state, "active", False)

    def _root():
        if state.active:
            raise RuntimeError("config unreadable")
        return shared_root

    with patch("osprey_connectors.workspace.resolve_shared_data_root", side_effect=_root):
        yield state


def _write_config(tmp_path, *, writes_enabled: bool):
    """Write a config.yml carrying the render's writes kill-switch."""
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump({"control_system": {"writes_enabled": writes_enabled}}),
        encoding="utf-8",
    )
    return path


def _write_shaped_config(tmp_path, section):
    """Write a config.yml carrying a whole ``control_system:`` section."""
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump({"control_system": section}), encoding="utf-8")
    return path


def _spawn_env(client, claude_session_id, telemetry_session_id=None):
    """Return the extra env the next PTY spawn for this session would carry."""
    return websocket_routes._build_extra_env(
        SimpleNamespace(app=client.app),
        claude_session_id,
        telemetry_session_id,
    )


class TestPostPosture:
    def test_post_sandbox_stores_persists_and_terminates(self, client, shared_root):
        """Happy path: the intent is recorded, durable, and applied at once.

        Applying it means terminating the session's PTY — the posture reaches
        the agent only through a fresh child process, so a route that stored
        without terminating would report a sandbox the running child is not in.
        """
        registry = client.app.state.pty_registry
        registry.get_or_create_session(SESSION_A, "echo")
        assert registry.get_session(SESSION_A) is not None

        with known_sessions(SESSION_A):
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "sandbox"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["session_id"] == SESSION_A
        assert body["posture"] == "sandbox"

        # In-memory store
        assert client.app.state.session_postures[SESSION_A] == "sandbox"

        # Write-through persistence
        store_file = shared_root / "session-postures.json"
        assert store_file.exists()
        assert json.loads(store_file.read_text(encoding="utf-8")) == {SESSION_A: "sandbox"}

        # The stale child is gone, so the next attach respawns under the posture.
        assert registry.get_session(SESSION_A) is None

    def test_post_unknown_session_id_conflicts(self, client):
        """409 for an id no session file backs, with the actionable remedy."""
        with known_sessions(SESSION_B):
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "sandbox"},
            )

        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert "send one prompt first" in json.dumps(detail)

    def test_post_unknown_session_id_stores_nothing(self, client, shared_root):
        """A refused toggle must leave neither memory nor disk changed."""
        with known_sessions():
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "sandbox"},
            )

        assert resp.status_code == 409
        assert SESSION_A not in getattr(client.app.state, "session_postures", {})
        assert not (shared_root / "session-postures.json").exists()

    def test_post_writes_forbidden_when_render_readonly(self, client, tmp_path):
        """403: stepping out to writes cannot exceed the rendered kill-switch.

        The posture toggle narrows privilege; it can never widen it past what
        ``control_system.writes_enabled`` already permits.
        """
        client.app.state.config_path = _write_config(tmp_path, writes_enabled=False)

        with known_sessions(SESSION_A):
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "writes"},
            )

        assert resp.status_code == 403
        assert SESSION_A not in getattr(client.app.state, "session_postures", {})

    def test_post_writes_forbidden_when_render_config_absent(self, client):
        """No readable config renders as writes-off, matching the renderer.

        ``cli/templates/claude_code.py`` and the ``osprey_writes_check`` hook
        both read ``writes_enabled`` with a ``False`` default, so an absent or
        unreadable config is a writes-off render everywhere. This gate agrees
        with them rather than inventing a permissive third answer.
        """
        client.app.state.config_path = None

        with known_sessions(SESSION_A):
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "writes"},
            )

        assert resp.status_code == 403

    def test_post_writes_allowed_when_render_enables_writes(self, client, tmp_path):
        """Backing out to writes succeeds on a writes-enabled render."""
        client.app.state.config_path = _write_config(tmp_path, writes_enabled=True)

        with known_sessions(SESSION_A):
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "writes"},
            )

        assert resp.status_code == 200
        assert client.app.state.session_postures[SESSION_A] == "writes"

    def test_post_sandbox_allowed_on_readonly_render(self, client, tmp_path):
        """A writes-off render still permits stepping *into* sandbox."""
        client.app.state.config_path = _write_config(tmp_path, writes_enabled=False)

        with known_sessions(SESSION_A):
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "sandbox"},
            )

        assert resp.status_code == 200

    @pytest.mark.parametrize("bad", ["readonly", "SANDBOX", "", "readwrite", "true"])
    def test_post_rejects_unknown_posture_value(self, client, bad):
        """Only the two named postures are accepted — no silent coercion."""
        with known_sessions(SESSION_A):
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": bad},
            )

        assert resp.status_code == 422

    def test_post_rejects_missing_fields(self, client):
        """A body without both fields is a malformed request, not a default."""
        resp = client.post("/api/terminal/posture", json={"posture": "sandbox"})
        assert resp.status_code == 422

    def test_post_rejects_malformed_session_id(self, client):
        """The id must look like a Claude session UUID before anything else.

        Reuses the module's ``_UUID_RE``, the same guard ``switch_session``
        applies, so a path traversal or an arbitrary string can never become a
        store key that is later written to disk.
        """
        with known_sessions(SESSION_A):
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": "../../etc/passwd", "posture": "sandbox"},
            )

        assert resp.status_code == 400

    def test_post_second_session_does_not_disturb_the_first(self, client, shared_root):
        """Postures are per session; the store holds both independently."""
        with known_sessions(SESSION_A, SESSION_B):
            assert (
                client.post(
                    "/api/terminal/posture",
                    json={"session_id": SESSION_A, "posture": "sandbox"},
                ).status_code
                == 200
            )
            assert (
                client.post(
                    "/api/terminal/posture",
                    json={"session_id": SESSION_B, "posture": "sandbox"},
                ).status_code
                == 200
            )

        on_disk = json.loads((shared_root / "session-postures.json").read_text(encoding="utf-8"))
        assert on_disk == {SESSION_A: "sandbox", SESSION_B: "sandbox"}


class TestPosturePersistence:
    def test_persisted_store_reloads_in_a_fresh_app(self, make_client, shared_root):
        """A container restart must not silently revert a sandboxed session.

        The second app is a different process's worth of state over the same
        agent-data directory — exactly what a recreated container sees.
        """
        with make_client() as first:
            with known_sessions(SESSION_A):
                assert (
                    first.post(
                        "/api/terminal/posture",
                        json={"session_id": SESSION_A, "posture": "sandbox"},
                    ).status_code
                    == 200
                )

        with make_client() as second:
            assert not hasattr(second.app.state, "session_postures")
            # First access — a spawn, before any route call — loads from disk.
            env = _spawn_env(second, None, SESSION_A)
            assert env["OSPREY_EXECUTION_MODE"] == "readonly"
            assert second.app.state.session_postures[SESSION_A] == "sandbox"

    def test_persist_tolerates_a_corrupt_store_file(self, make_client, shared_root):
        """Unreadable persisted state must not take the server down with it.

        Fail-open relative to operator intent: a corrupt file loses the
        recorded postures (the operator can set them again) rather than
        wedging every spawn and every toggle.
        """
        (shared_root / "session-postures.json").write_text("{not json", encoding="utf-8")

        with make_client() as client:
            with known_sessions(SESSION_A):
                resp = client.post(
                    "/api/terminal/posture",
                    json={"session_id": SESSION_A, "posture": "sandbox"},
                )
            assert resp.status_code == 200
            assert client.app.state.session_postures == {SESSION_A: "sandbox"}

    def test_persist_drops_unknown_postures_on_load(self, make_client, shared_root):
        """A hand-edited or future-version entry is ignored, not honored.

        Anything but the two known values would otherwise flow straight into
        ``_build_extra_env`` and decide a child's execution mode.
        """
        (shared_root / "session-postures.json").write_text(
            json.dumps({SESSION_A: "sandbox", SESSION_B: "wide-open"}),
            encoding="utf-8",
        )

        with make_client() as client:
            assert _spawn_env(client, SESSION_A).get("OSPREY_EXECUTION_MODE") == "readonly"
            assert "OSPREY_EXECUTION_MODE" not in _spawn_env(client, SESSION_B)
            assert client.app.state.session_postures == {SESSION_A: "sandbox"}

    def test_persistence_failure_does_not_fail_the_toggle(self, client, shared_root):
        """Disk trouble must not block the operator from sandboxing a session.

        The in-memory store still carries the intent and the session is still
        terminated, so the posture takes effect now; only its durability is
        lost, and that is logged rather than raised.
        """
        with (
            known_sessions(SESSION_A),
            patch.object(
                websocket_routes,
                "_atomic_write_json",
                side_effect=OSError("disk full"),
            ),
        ):
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "sandbox"},
            )

        assert resp.status_code == 200
        assert client.app.state.session_postures[SESSION_A] == "sandbox"


class TestSpawnEnvironment:
    def test_next_spawn_env_carries_sandbox_posture(self, client):
        """A sandboxed session's next child runs in readonly execution mode."""
        with known_sessions(SESSION_A):
            client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "sandbox"},
            )

        # Resume shape: both ids are the session id.
        assert _spawn_env(client, SESSION_A, SESSION_A)["OSPREY_EXECUTION_MODE"] == "readonly"
        # switch_session shape: telemetry id omitted.
        assert _spawn_env(client, SESSION_A)["OSPREY_EXECUTION_MODE"] == "readonly"

    def test_next_spawn_env_omits_execution_mode_for_writes(self, client, tmp_path):
        """Backing out to writes leaves the variable absent, not falsified.

        ``OSPREY_EXECUTION_MODE`` is read as a gate elsewhere; the writes
        posture is the *default* posture and must look exactly like a session
        that was never toggled.
        """
        client.app.state.config_path = _write_config(tmp_path, writes_enabled=True)

        with known_sessions(SESSION_A):
            client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "sandbox"},
            )
            client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "writes"},
            )

        assert "OSPREY_EXECUTION_MODE" not in _spawn_env(client, SESSION_A, SESSION_A)

    def test_untoggled_session_env_is_unchanged(self, client):
        """A session with no stored posture keeps the render's baseline env."""
        env = _spawn_env(client, None, SESSION_A)
        assert "OSPREY_EXECUTION_MODE" not in env
        assert env["OSPREY_WEB_UX"] == "expert"
        assert env["OSPREY_TELEMETRY_SESSION_ID"] == SESSION_A

    def test_new_session_env_is_keyed_on_the_telemetry_id(self, client):
        """New sessions are pooled under the forced id, so key on it too.

        ``terminal_ws`` sets ``current_key = claude_session_id or
        telemetry_session_id``; keying the store on anything else would leave
        a brand-new session unable to hold a posture at all.
        """
        with known_sessions(SESSION_A):
            client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "sandbox"},
            )

        # New-session shape: claude id is None, telemetry id is the pool key.
        assert _spawn_env(client, None, SESSION_A)["OSPREY_EXECUTION_MODE"] == "readonly"

    def test_posture_does_not_override_a_deployment_wide_readonly(self, client, tmp_path):
        """A hooks-supplied readonly survives a per-session writes posture.

        The toggle only ever *adds* the sandbox marker. Letting ``writes``
        strip a variable the deployment itself injected would turn a
        per-session convenience into a privilege escalation.
        """
        client.app.state.config_path = _write_config(tmp_path, writes_enabled=True)
        client.app.state.hooks_env = {"OSPREY_EXECUTION_MODE": "readonly"}

        with known_sessions(SESSION_A):
            client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "writes"},
            )

        assert _spawn_env(client, SESSION_A)["OSPREY_EXECUTION_MODE"] == "readonly"


class TestGetPosture:
    """``GET /api/terminal/posture?session_id=`` — the single truth the badge reads.

    Two fields, two different questions, and the badge needs both:

    * ``posture`` is what the *store* holds for this session, defaulting to
      ``writes`` when nothing is stored — the untoggled session spawns without
      the sandbox marker, so ``writes`` is the honest report of the posture.
    * ``rendered_writes_enabled`` is what the *render* permits: whether writes
      are armed for *some* control target. It is what makes the default reading
      honest on a writes-off deployment: the posture is ``writes`` and the
      effective write capability is still nil, and the badge is the surface
      that has to say so instead of implying the session can write.
    """

    def test_get_posture_returns_the_stored_entry(self, client, tmp_path):
        """A sandboxed session reads back as sandboxed."""
        client.app.state.config_path = _write_config(tmp_path, writes_enabled=True)

        with known_sessions(SESSION_A):
            client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "sandbox"},
            )

        resp = client.get("/api/terminal/posture", params={"session_id": SESSION_A})

        assert resp.status_code == 200
        assert resp.json() == {
            "session_id": SESSION_A,
            "posture": "sandbox",
            "rendered_writes_enabled": True,
        }

    def test_get_posture_defaults_to_writes_when_nothing_is_stored(self, client, tmp_path):
        """No entry means the session runs the render's baseline — ``writes``.

        The store only ever records a *deviation*: ``_build_extra_env`` adds
        ``OSPREY_EXECUTION_MODE`` for a sandboxed session and nothing at all
        otherwise. Reporting anything but ``writes`` here would describe an
        env the child does not carry.
        """
        client.app.state.config_path = _write_config(tmp_path, writes_enabled=True)

        body = client.get("/api/terminal/posture", params={"session_id": SESSION_B}).json()

        assert body == {
            "session_id": SESSION_B,
            "posture": "writes",
            "rendered_writes_enabled": True,
        }

    def test_get_posture_does_not_disturb_a_second_session(self, client, tmp_path):
        """Postures are per session on the way out as well as on the way in."""
        client.app.state.config_path = _write_config(tmp_path, writes_enabled=True)

        with known_sessions(SESSION_A):
            client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "sandbox"},
            )

        assert (
            client.get("/api/terminal/posture", params={"session_id": SESSION_A}).json()["posture"]
            == "sandbox"
        )
        assert (
            client.get("/api/terminal/posture", params={"session_id": SESSION_B}).json()["posture"]
            == "writes"
        )

    def test_get_posture_reports_a_writes_off_render(self, client, tmp_path):
        """The default posture on a writes-off render still reports the render.

        This is the pairing the badge exists for: ``posture: writes`` with
        ``rendered_writes_enabled: false`` means "not sandboxed, and cannot
        write anyway" — the kill-switch, not the toggle, is the binding
        constraint, and the operator must not read the badge as an offer.
        """
        client.app.state.config_path = _write_config(tmp_path, writes_enabled=False)

        body = client.get("/api/terminal/posture", params={"session_id": SESSION_A}).json()

        assert body["posture"] == "writes"
        assert body["rendered_writes_enabled"] is False

    def test_get_posture_reports_an_absent_config_as_writes_off(self, client):
        """No readable config is a writes-off render, as everywhere else.

        Same default as the POST gate, ``cli/templates/claude_code.py`` and the
        ``osprey_writes_check`` hook: an unreadable config never reads as
        permissive.
        """
        client.app.state.config_path = None

        body = client.get("/api/terminal/posture", params={"session_id": SESSION_A}).json()

        assert body["rendered_writes_enabled"] is False

    def test_get_posture_tolerates_a_session_that_has_not_started(self, client, tmp_path):
        """A syntactically valid id with no session file still answers 200.

        Unlike POST — which 409s because there is nothing to respawn — GET is
        deliberately tolerant. The badge renders with the terminal card, before
        the first prompt has written a session file, and a 409 there would
        blank the one surface that tells the operator what the render permits.
        Answering is safe: GET grants nothing and stores nothing, it reports
        the same default that an unstarted session will actually spawn under.
        """
        client.app.state.config_path = _write_config(tmp_path, writes_enabled=True)

        with known_sessions():
            resp = client.get("/api/terminal/posture", params={"session_id": SESSION_A})

        assert resp.status_code == 200
        assert resp.json()["posture"] == "writes"
        # Tolerating the read must not invent a store entry for it.
        assert SESSION_A not in getattr(client.app.state, "session_postures", {})

    @pytest.mark.parametrize("bad", ["../../etc/passwd", "not-a-uuid", "", "a" * 40])
    def test_get_posture_rejects_a_malformed_session_id(self, client, bad):
        """The id is shape-checked with the same ``_UUID_RE`` POST uses.

        A read is not a write, but the same guard keeps one error contract for
        the badge and keeps an arbitrary string from ever reaching a store
        lookup keyed on operator-supplied text.
        """
        resp = client.get("/api/terminal/posture", params={"session_id": bad})

        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "invalid_session_id"

    def test_get_posture_requires_a_session_id(self, client):
        """The query parameter is required — there is no "current session"."""
        assert client.get("/api/terminal/posture").status_code == 422

    def test_get_posture_retries_the_store_after_a_transient_config_failure(
        self, make_client, shared_root
    ):
        """A momentary config failure must not pin an empty store for the process.

        ``_posture_store_path`` falls back to the workspace dir when
        ``resolve_shared_data_root()`` raises. Caching *that* load would make
        one transient config error at first access outlive the error itself:
        every later read would serve the empty fallback store, reporting
        ``writes`` for a session the persisted store has sandboxed — the exact
        silent revert the persistence exists to prevent.
        """
        (shared_root / "session-postures.json").write_text(
            json.dumps({SESSION_A: "sandbox"}), encoding="utf-8"
        )

        with make_client() as client, config_outage(shared_root) as outage:
            during = client.get("/api/terminal/posture", params={"session_id": SESSION_A}).json()
            outage.over()
            after = client.get("/api/terminal/posture", params={"session_id": SESSION_A}).json()

        # The fallback root holds no store, so the outage answers the baseline…
        assert during["posture"] == "writes"
        # …and the very next read finds the primary store again.
        assert after["posture"] == "sandbox"

    def test_get_posture_keeps_a_posture_set_during_a_config_outage(
        self, make_client, shared_root, tmp_path
    ):
        """Re-reading the primary store must not drop the operator's intent.

        A posture set while the shared root was unresolvable lives only in
        memory and in the fallback file; the recovery read merges the primary
        store *under* what memory already holds rather than replacing it, or
        sandboxing a session during an outage would silently undo itself the
        moment the config came back.
        """
        with make_client() as client, config_outage(shared_root) as outage:
            client.app.state.config_path = _write_config(tmp_path, writes_enabled=True)
            with known_sessions(SESSION_B):
                assert (
                    client.post(
                        "/api/terminal/posture",
                        json={"session_id": SESSION_B, "posture": "sandbox"},
                    ).status_code
                    == 200
                )
            outage.over()
            body = client.get("/api/terminal/posture", params={"session_id": SESSION_B}).json()

        assert body["posture"] == "sandbox"


class TestPerTargetRenderPosture:
    """``rendered_writes_enabled`` means "some target may write", not one flag.

    Write posture is per connector type —
    ``control_system.connector.<type>.writes_enabled``, inheriting the
    deployment-wide ``control_system.writes_enabled`` where it is absent — so a
    deployment whose baseline is a live machine can arm its virtual accelerator
    alone. The render permits stepping out of the sandbox as soon as ONE target
    a session here can be pointed at is armed; *which* machine the session may
    then write to is the connector layer's refusal to make, not this gate's.

    Both the badge payload and the 403 gate read the same predicate, so the
    button an operator is offered and the answer they get on pressing it can
    never disagree. These tests pin that pair together.
    """

    def test_mock_render_arms_no_target(self, client, tmp_path):
        """A mock deployment that arms nothing is unchanged: writes off."""
        client.app.state.config_path = _write_shaped_config(tmp_path, {"type": "mock"})

        body = client.get("/api/terminal/posture", params={"session_id": SESSION_A}).json()

        assert body["rendered_writes_enabled"] is False

    def test_va_armed_alone_reports_and_permits_writes(self, client, tmp_path):
        """Global off, VA block on: the badge says writes and the gate agrees.

        The motivating shape of the whole feature — a deployment built for a
        live machine that arms writes on its simulator only. The session may
        step out of the sandbox because there is a target it can write to.
        """
        client.app.state.config_path = _write_shaped_config(
            tmp_path,
            {
                "type": "epics",
                "writes_enabled": False,
                "connector": {
                    "epics": {"timeout": 5.0},
                    "virtual_accelerator": {"writes_enabled": True},
                },
            },
        )

        body = client.get("/api/terminal/posture", params={"session_id": SESSION_A}).json()
        with known_sessions(SESSION_A):
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "writes"},
            )

        assert body["rendered_writes_enabled"] is True
        assert resp.status_code == 200

    def test_live_disarmed_leaves_the_inheriting_va_armed(self, client, tmp_path):
        """Global on with the live block off is still an armed deployment.

        The disarm-mixed shape: the facility has said "not this machine" under
        its live block, which does *not* fall back to the global key. The VA
        block says nothing and so keeps inheriting the global ``true``, and one
        armed target is all the render needs to permit ``writes``.
        """
        client.app.state.config_path = _write_shaped_config(
            tmp_path,
            {
                "type": "epics",
                "writes_enabled": True,
                "connector": {
                    "epics": {"writes_enabled": False},
                    "virtual_accelerator": {"simulation_file": "data/simulation/machine.json"},
                },
            },
        )

        body = client.get("/api/terminal/posture", params={"session_id": SESSION_A}).json()

        assert body["rendered_writes_enabled"] is True

    def test_va_baseline_with_its_only_target_disarmed_is_writes_off(self, client, tmp_path):
        """An unreachable ``live`` must not vote, or the global key over-permits.

        A virtual-accelerator deployment with no live block has exactly one
        target a session can be pointed at, and it is explicitly disarmed. The
        global ``true`` still answers for ``live`` — ``target_writes_enabled``
        falls back to it when the target does not resolve — so a union over both
        named targets would offer the operator a ``writes`` posture that arms
        nothing at all. The union runs over the *reachable* targets instead.
        """
        client.app.state.config_path = _write_shaped_config(
            tmp_path,
            {
                "type": "virtual_accelerator",
                "writes_enabled": True,
                "connector": {"virtual_accelerator": {"writes_enabled": False}},
            },
        )

        body = client.get("/api/terminal/posture", params={"session_id": SESSION_A}).json()
        with known_sessions(SESSION_A):
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "writes"},
            )

        assert body["rendered_writes_enabled"] is False
        assert resp.status_code == 403

    def test_malformed_config_arms_no_target(self, client, tmp_path):
        """A config that will not parse is a writes-off render, not a crash.

        The predicate answers ``False`` for everything it cannot read — an
        unparseable file, a section the resolver cannot make sense of — because
        the badge and the gate must fail towards "no target may write".
        """
        path = tmp_path / "config.yml"
        path.write_text("control_system: [unclosed\n", encoding="utf-8")
        client.app.state.config_path = path

        body = client.get("/api/terminal/posture", params={"session_id": SESSION_A}).json()

        assert body["rendered_writes_enabled"] is False

    def test_no_armed_target_refuses_naming_the_per_type_key(self, client, tmp_path):
        """The 403 names the keys an operator would actually have to edit.

        Saying only "``control_system.writes_enabled`` is off" would be a lie on
        a config that sets it ``true`` and disarms every type beneath it, and it
        would send the operator to the one key that cannot fix this.
        """
        client.app.state.config_path = _write_shaped_config(
            tmp_path,
            {
                "type": "epics",
                "writes_enabled": False,
                "connector": {
                    "epics": {"writes_enabled": False},
                    "virtual_accelerator": {"writes_enabled": False},
                },
            },
        )

        with known_sessions(SESSION_A):
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "writes"},
            )

        assert resp.status_code == 403
        detail = json.dumps(resp.json()["detail"])
        assert "no control target" in detail.lower()
        assert "control_system.connector.<type>.writes_enabled" in detail
        assert "control_system.writes_enabled" in detail


# ── SDK-topology parity ──────────────────────────────────────────────────────
#
# The posture landed on the PTY seam first, but the web terminal has a second
# topology: the SDK-backed surfaces (``/ws/operator`` and ``POST /api/chat``),
# whose child environment comes from ``build_operator_child_env`` rather than
# ``_build_extra_env``. Multi/single-user parity is a project rule, and a
# posture that only the PTY honours would be exactly the badge-that-lies the
# store exists to prevent: the operator sandboxes a session, the SDK child
# spawns with writes anyway.
#
# These tests pin the two seams to the SAME answer for the SAME key, in both
# postures, and pin the two call sites to the keys they actually hold.


class _RecordingOperatorRegistry:
    """Stand-in operator registry that records the env each spawn was handed.

    Both SDK call sites are exercised through their real handler code — the
    only thing replaced is the SDK session itself, which cannot start in a
    test. What the handler computed and passed is what gets asserted.
    """

    def __init__(self, live_chats=()):
        self.calls: list[dict] = []
        # Stands in for the real registry's chat pool. The posture gate probes
        # membership through ``get_chat_session``, so what lives here is what
        # the gate sees as a live chat session.
        self.chats: dict[str, object] = {
            chat_id: SimpleNamespace(acquire_turn=lambda: 1) for chat_id in live_chats
        }

    async def create_session(self, session_id, cwd, env=None):
        self.calls.append({"key": session_id, "cwd": cwd, "env": env})
        return SimpleNamespace(_queue=asyncio.Queue())

    async def get_or_create_chat_session(self, chat_id, cwd, env=None):
        # The route hands the pool a zero-arg env BUILDER (so the posture read
        # and the pool's registration of the creation cannot be separated —
        # see ChatSessionPool.get_or_create); the real pool calls it inside its
        # lock. Resolve it here too, so what is recorded is the mapping the
        # child would really be given.
        self.calls.append({"key": chat_id, "cwd": cwd, "env": env() if callable(env) else env})
        session = self.chats.setdefault(chat_id, SimpleNamespace(acquire_turn=lambda: 1))
        return session, False

    def get_chat_session(self, chat_id):
        return self.chats.get(chat_id)

    async def terminate_session_if_owner(self, session_id, owner):
        return None

    async def cleanup_all(self):
        # The app's own shutdown calls this; nothing here to clean up.
        return None


def _sdk_env(client, session_key=None):
    """Return the env the next SDK (operator/chat) child would carry."""
    return build_operator_child_env(
        client.app.state.project_cwd,
        session_key=session_key,
        app=client.app,
    )


def _seed_posture(client, key, posture):
    """Put *posture* in the live store under *key*.

    The SDK surfaces are keyed on identifiers the POST route cannot address
    (see ``test_sdk_parity_operator_ws_...``), so these tests write the store
    directly — through ``_session_postures``, the same accessor the routes
    use, so the lazy load and the app.state siting are the real ones.
    """
    websocket_routes._session_postures(client.app)[key] = posture


class TestSdkPostureParity:
    def test_sdk_parity_sandbox_marks_both_children(self, client):
        """One key, one posture, two topologies — both children go readonly."""
        with known_sessions(SESSION_A):
            assert (
                client.post(
                    "/api/terminal/posture",
                    json={"session_id": SESSION_A, "posture": "sandbox"},
                ).status_code
                == 200
            )

        pty = _spawn_env(client, SESSION_A, SESSION_A)
        sdk = _sdk_env(client, SESSION_A)

        assert pty["OSPREY_EXECUTION_MODE"] == "readonly"
        assert sdk["OSPREY_EXECUTION_MODE"] == "readonly"
        assert sdk["OSPREY_EXECUTION_MODE"] == pty["OSPREY_EXECUTION_MODE"]

    def test_sdk_parity_writes_marks_neither_child(self, client, tmp_path):
        """The default posture must look identical on both seams: absent.

        ``OSPREY_EXECUTION_MODE`` is read as a gate elsewhere, so ``writes``
        leaves it unset rather than setting a falsy value — on the SDK path
        just as on the PTY path.
        """
        client.app.state.config_path = _write_config(tmp_path, writes_enabled=True)

        with known_sessions(SESSION_A):
            client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "sandbox"},
            )
            assert (
                client.post(
                    "/api/terminal/posture",
                    json={"session_id": SESSION_A, "posture": "writes"},
                ).status_code
                == 200
            )

        assert "OSPREY_EXECUTION_MODE" not in _spawn_env(client, SESSION_A, SESSION_A)
        assert "OSPREY_EXECUTION_MODE" not in _sdk_env(client, SESSION_A)

    def test_sdk_parity_unstored_key_renders_baseline(self, client):
        """A key the store has never held is the render's baseline, both ways."""
        with known_sessions(SESSION_A):
            client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "sandbox"},
            )

        assert "OSPREY_EXECUTION_MODE" not in _spawn_env(client, SESSION_B, SESSION_B)
        assert "OSPREY_EXECUTION_MODE" not in _sdk_env(client, SESSION_B)

    def test_sdk_parity_missing_key_renders_baseline(self, client):
        """No key at all is the baseline env — never a lookup on ``None``.

        A caller that cannot name its session (or a caller not yet wired for
        the posture) gets exactly the environment it got before this seam
        existed, panel token included.
        """
        with known_sessions(SESSION_A):
            client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "sandbox"},
            )

        env = build_operator_child_env(client.app.state.project_cwd)
        with_app = _sdk_env(client, None)

        assert "OSPREY_EXECUTION_MODE" not in env
        assert "OSPREY_EXECUTION_MODE" not in with_app
        assert env[PANEL_TOKEN_ENV] == get_web_credentials(client.app).panel_token

    def test_sdk_parity_never_clears_an_inherited_readonly(self, client, tmp_path, monkeypatch):
        """A deployment-wide readonly survives a per-session ``writes``.

        The SDK child inherits ``OSPREY_EXECUTION_MODE`` through
        ``build_clean_env``'s copy of ``os.environ``. The posture may add the
        sandbox marker; clearing an inherited one would turn a per-session
        convenience into a privilege escalation — the same narrowing-only rule
        ``_build_extra_env`` follows for the PTY child's ``hooks_env``.
        """
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        client.app.state.config_path = _write_config(tmp_path, writes_enabled=True)
        client.app.state.hooks_env = {"OSPREY_EXECUTION_MODE": "readonly"}

        with known_sessions(SESSION_A):
            assert (
                client.post(
                    "/api/terminal/posture",
                    json={"session_id": SESSION_A, "posture": "writes"},
                ).status_code
                == 200
            )

        assert _spawn_env(client, SESSION_A)["OSPREY_EXECUTION_MODE"] == "readonly"
        assert _sdk_env(client, SESSION_A)["OSPREY_EXECUTION_MODE"] == "readonly"

    def test_sdk_parity_chat_child_env_is_keyed_on_the_chat_id(self, client):
        """``POST /api/chat`` spawns its session under its own ``chat_id``.

        The chat pool is keyed on the caller-supplied ``chat_id``, so that is
        the only identity the posture can attach to on this surface. The two
        keys here are bare UUIDs, the shape the shipped client mints and the
        only shape the posture route will write — seeding the store under
        anything else would prove the builder reads a state no route can
        produce.
        """
        registry = _RecordingOperatorRegistry()
        client.app.state.operator_registry = registry
        _seed_posture(client, CHAT_A, "sandbox")
        _seed_posture(client, SESSION_B, "writes")

        request = SimpleNamespace(app=client.app)
        asyncio.run(chat_routes._acquire_chat_turn(request, CHAT_A))
        asyncio.run(chat_routes._acquire_chat_turn(request, SESSION_B))

        sandboxed, writing = registry.calls
        assert sandboxed["key"] == CHAT_A
        assert sandboxed["env"]["OSPREY_EXECUTION_MODE"] == "readonly"
        assert writing["key"] == SESSION_B
        assert "OSPREY_EXECUTION_MODE" not in writing["env"]

    def test_sdk_parity_an_unaddressable_chat_id_is_not_labelled_live(self, client):
        """A chat id no posture route can name is spawned ``process``.

        ``chat_id`` is unconstrained on the chat surface, and an
        embedder-chosen key like ``user-42-chat-3`` is refused 400 by both
        posture verbs. ``live`` means "a store keeps answering for this key",
        so stamping it there would put a provenance in the ledger that says a
        runtime toggle governed a process no toggle can reach.
        """
        registry = _RecordingOperatorRegistry()
        client.app.state.operator_registry = registry
        request = SimpleNamespace(app=client.app)

        asyncio.run(chat_routes._acquire_chat_turn(request, CHAT_A))
        asyncio.run(chat_routes._acquire_chat_turn(request, "user-42-chat-3"))

        addressable, embedder_chosen = registry.calls
        assert addressable["env"][POSTURE_SOURCE_ENV] == "live"
        assert embedder_chosen["env"][POSTURE_SOURCE_ENV] == "process"
        # The key it was checked under is still recorded, either way.
        assert embedder_chosen["env"][POSTURE_SESSION_ENV] == "user-42-chat-3"
        # And that key really is unaddressable on both posture verbs.
        assert (
            client.post(
                "/api/terminal/posture",
                json={"session_id": "user-42-chat-3", "posture": "sandbox"},
            ).status_code
            == 400
        )

    def test_sdk_parity_operator_ws_child_env_is_keyed_on_its_session_key(self, client):
        """``/ws/operator`` spawns under the pool key it mints per connection.

        The operator websocket resumes nothing: it mints ``operator-<hex8>``
        at accept time and that key is the session's whole identity, so it is
        what the posture lookup is keyed on. (Such a key is outside the posture
        surface's closed grammar — ``_POSTURE_KEY_RE`` — by design, not for
        want of a UI: an operator connection is addressable by nobody and its
        posture is deliberately non-durable, so the seam honours whatever the
        store held when the child started and nothing can flip it afterwards.)
        """
        registry = _RecordingOperatorRegistry()
        client.app.state.operator_registry = registry
        forced = uuid.UUID("dddddddd-1111-2222-3333-444444444444")
        operator_key = f"operator-{forced.hex[:8]}"
        _seed_posture(client, operator_key, "sandbox")

        with patch("uuid.uuid4", return_value=forced):
            with client.websocket_connect("/ws/operator") as ws:
                assert ws.receive_json() == {"type": "system", "subtype": "init"}

        assert registry.calls[0]["key"] == operator_key
        assert registry.calls[0]["env"]["OSPREY_EXECUTION_MODE"] == "readonly"
        # Parity with the PTY seam on that very same key.
        assert _spawn_env(client, operator_key)["OSPREY_EXECUTION_MODE"] == "readonly"

    def test_sdk_parity_operator_ws_writes_key_stays_baseline(self, client):
        """The minted key with no sandbox posture spawns the baseline child."""
        registry = _RecordingOperatorRegistry()
        client.app.state.operator_registry = registry
        forced = uuid.UUID("eeeeeeee-1111-2222-3333-444444444444")
        operator_key = f"operator-{forced.hex[:8]}"
        _seed_posture(client, operator_key, "writes")

        with patch("uuid.uuid4", return_value=forced):
            with client.websocket_connect("/ws/operator") as ws:
                ws.receive_json()

        assert registry.calls[0]["key"] == operator_key
        assert "OSPREY_EXECUTION_MODE" not in registry.calls[0]["env"]


# ── The existence gate, across both topologies ───────────────────────────────
#
# ``POST /api/terminal/posture`` refuses a key that names no session, because a
# posture on such a key is a toggle nothing will ever read. "No session",
# though, is a question with two answers: the PTY topology's sessions are the
# JSONL stems ``SessionDiscovery`` walks, and the SDK topology's chat sessions
# live only in the operator registry's chat pool. Asking only the first is what
# made a chat session's posture unsettable — the chat spawn already reads the
# store back (``_acquire_chat_turn`` -> ``build_operator_child_env``), so the
# store was readable there and not writable.


class TestPostureGateAcrossTopologies:
    def test_post_accepts_a_live_chat_pool_key(self, client, shared_root):
        """A chat session live in the pool can have its posture set.

        Nothing is on disk for it — the chat topology writes no JSONL stem —
        so this is exactly the case the discovery-only gate used to 409.
        """
        client.app.state.operator_registry = _RecordingOperatorRegistry(live_chats=[CHAT_A])

        with known_sessions():
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": CHAT_A, "posture": "sandbox"},
            )

        assert resp.status_code == 200
        assert resp.json()["session_id"] == CHAT_A
        assert client.app.state.session_postures[CHAT_A] == "sandbox"
        on_disk = json.loads((shared_root / "session-postures.json").read_text(encoding="utf-8"))
        assert on_disk == {CHAT_A: "sandbox"}

    def test_stored_chat_posture_reaches_that_chat_child(self, client):
        """The point of writing the store: the next chat spawn carries it.

        The gate exists to make the store *writable* for a chat key; this pins
        that the key it writes is the same one the chat spawn looks up, so the
        toggle and the child cannot disagree.
        """
        registry = _RecordingOperatorRegistry(live_chats=[CHAT_A])
        client.app.state.operator_registry = registry

        with known_sessions():
            assert (
                client.post(
                    "/api/terminal/posture",
                    json={"session_id": CHAT_A, "posture": "sandbox"},
                ).status_code
                == 200
            )

        asyncio.run(chat_routes._acquire_chat_turn(SimpleNamespace(app=client.app), CHAT_A))

        assert registry.calls[-1]["key"] == CHAT_A
        assert registry.calls[-1]["env"]["OSPREY_EXECUTION_MODE"] == "readonly"

    def test_post_chat_key_without_a_live_session_conflicts(self, client):
        """A key naming neither topology is still a 409, with the same remedy."""
        client.app.state.operator_registry = _RecordingOperatorRegistry()

        with known_sessions():
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": CHAT_A, "posture": "sandbox"},
            )

        assert resp.status_code == 409
        assert "send one prompt first" in json.dumps(resp.json()["detail"])
        assert CHAT_A not in getattr(client.app.state, "session_postures", {})

    def test_a_sandboxed_chat_can_be_brought_back_out(self, client, tmp_path):
        """The store keeps a key addressable after the pool has dropped it.

        The pool is LRU-capped, idle-reaped — and evicted by the flip itself,
        which is the point: a successful sandbox is exactly what removes the
        entry the gate used to require. Refusing on pool membership alone would
        mean a chat could be sandboxed once and never brought back, with the
        badge offering a switch that always 409s. An entry in the store is the
        operator's own earlier decision about this key, and letting them revise
        it grants nothing: the store only ever narrows a spawn.
        """
        registry = _RecordingOperatorRegistry(live_chats=[CHAT_A])
        client.app.state.operator_registry = registry
        client.app.state.config_path = _write_config(tmp_path, writes_enabled=True)

        with known_sessions():
            assert (
                client.post(
                    "/api/terminal/posture",
                    json={"session_id": CHAT_A, "posture": "sandbox"},
                ).status_code
                == 200
            )
            registry.chats.pop(CHAT_A)
            assert (
                client.post(
                    "/api/terminal/posture",
                    json={"session_id": CHAT_A, "posture": "writes"},
                ).status_code
                == 200
            )

        assert client.app.state.session_postures[CHAT_A] == "writes"

    def test_post_pty_stem_is_accepted_with_an_empty_chat_pool(self, client):
        """Widening the gate must not cost the PTY topology its own answer."""
        client.app.state.operator_registry = _RecordingOperatorRegistry()

        with known_sessions(SESSION_A):
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": SESSION_A, "posture": "sandbox"},
            )

        assert resp.status_code == 200

    def test_gate_tolerates_a_registry_with_no_chat_pool(self, client):
        """A registry that cannot be asked has no chat session to offer.

        The probe answers ``False`` rather than raising, so a stand-in registry
        (or a future one without the pool) degrades to the PTY-only gate
        instead of turning every posture call into a 500.
        """

        # No ``get_chat_session`` at all; ``cleanup_all`` only because the
        # app's own shutdown calls it.
        async def _noop():
            return None

        client.app.state.operator_registry = SimpleNamespace(cleanup_all=_noop)

        with known_sessions(SESSION_A):
            assert (
                client.post(
                    "/api/terminal/posture",
                    json={"session_id": SESSION_A, "posture": "sandbox"},
                ).status_code
                == 200
            )
            assert (
                client.post(
                    "/api/terminal/posture",
                    json={"session_id": CHAT_A, "posture": "sandbox"},
                ).status_code
                == 409
            )

    def test_membership_probe_does_not_disturb_the_pool(self, client):
        """Asking whether a chat session is live must not touch the pool.

        The real accessor is a plain dict read: no idle-clock refresh, no
        eviction, no creation. A probe that created a session would let a
        posture call spawn an agent.
        """
        registry = _RecordingOperatorRegistry()
        client.app.state.operator_registry = registry

        with known_sessions():
            client.post(
                "/api/terminal/posture",
                json={"session_id": CHAT_A, "posture": "sandbox"},
            )

        assert registry.chats == {}
        assert registry.calls == []


class TestPostureKeyGrammar:
    """The posture surface's key grammar is closed: a bare canonical UUID.

    Both identities the route can legitimately name are minted that way — a
    Claude session-file stem and the shipped chat client's
    ``crypto.randomUUID()`` — so closing the grammar costs no reach, and it
    keeps a decorated or near-miss string from becoming a key in a store that
    decides a child process's execution mode.
    """

    @pytest.mark.parametrize(
        "bad",
        [
            "-" * 36,  # 36 legal characters, no UUID structure at all
            "aaaaaaaaa-111-2222-3333-444444444444",  # 9-3-4-4-12: wrong grouping
            "aaaaaaaa1111222233334444444444444444",  # 36 hex, no separators
            "aaaaaaaa-1111-2222-3333-44444444444",  # one digit short
        ],
    )
    def test_post_rejects_a_non_canonical_36_character_key(self, client, bad):
        """Length and alphabet are not the contract — the shape is."""
        with known_sessions(bad):
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": bad, "posture": "sandbox"},
            )

        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "invalid_session_id"
        assert bad not in getattr(client.app.state, "session_postures", {})

    def test_post_rejects_an_operator_pool_key(self, client):
        """``operator-<hex8>`` keys stay unreachable from this surface.

        ``/ws/operator`` mints its pool key per connection and hands it to
        nobody, so an operator cannot address one anyway; the spawn seam still
        honours a posture stored under such a key (see
        ``TestSdkPostureParity``), and offering a route to set one would mean
        accepting a decorated key into the store on the operator's say-so.
        """
        operator_key = "operator-dddddddd"

        with known_sessions(operator_key):
            resp = client.post(
                "/api/terminal/posture",
                json={"session_id": operator_key, "posture": "sandbox"},
            )

        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "invalid_session_id"

    def test_get_rejects_the_same_keys_as_post(self, client):
        """One grammar, one error contract, both routes."""
        for bad in ("-" * 36, "operator-dddddddd", "aaaaaaaa1111222233334444444444444444"):
            resp = client.get("/api/terminal/posture", params={"session_id": bad})
            assert resp.status_code == 400, bad
            assert resp.json()["detail"]["error"] == "invalid_session_id"

    def test_canonical_uuids_still_pass(self, client):
        """The closed grammar must not cost a real session its posture."""
        client.app.state.operator_registry = _RecordingOperatorRegistry(live_chats=[CHAT_A])

        with known_sessions(SESSION_A):
            assert (
                client.post(
                    "/api/terminal/posture",
                    json={"session_id": SESSION_A, "posture": "sandbox"},
                ).status_code
                == 200
            )
            assert (
                client.post(
                    "/api/terminal/posture",
                    json={"session_id": CHAT_A, "posture": "sandbox"},
                ).status_code
                == 200
            )


class TestGetPostureForChatKeys:
    """GET stays tolerant for chat keys, exactly as it is for PTY ones.

    The badge reads GET while the surface renders, which on the chat side can
    be before the pool holds anything at all. A 409 there would blank the one
    surface that tells the operator what the render permits, and answering
    grants nothing: GET stores nothing and reports the same default the next
    spawn under that key would actually carry.
    """

    def test_get_answers_for_a_chat_key_with_no_live_session(self, client, tmp_path):
        client.app.state.config_path = _write_config(tmp_path, writes_enabled=True)
        client.app.state.operator_registry = _RecordingOperatorRegistry()

        with known_sessions():
            resp = client.get("/api/terminal/posture", params={"session_id": CHAT_A})

        assert resp.status_code == 200
        assert resp.json() == {
            "session_id": CHAT_A,
            "posture": "writes",
            "rendered_writes_enabled": True,
        }
        assert CHAT_A not in getattr(client.app.state, "session_postures", {})

    def test_get_reads_back_a_chat_key_posture(self, client, tmp_path):
        """What POST wrote under a chat key is what the badge reads back."""
        client.app.state.config_path = _write_config(tmp_path, writes_enabled=True)
        client.app.state.operator_registry = _RecordingOperatorRegistry(live_chats=[CHAT_A])

        with known_sessions():
            client.post(
                "/api/terminal/posture",
                json={"session_id": CHAT_A, "posture": "sandbox"},
            )

        body = client.get("/api/terminal/posture", params={"session_id": CHAT_A}).json()

        assert body["posture"] == "sandbox"
