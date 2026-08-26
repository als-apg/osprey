"""Tests for the spawn-side ``(posture_source, session)`` env markers.

The audit envelope records *where* a posture decision came from — a closed set
of ``spawn | live | app | process`` — and *which* posture-store key governed
the record. Both are carried to the child as explicit environment markers:

* ``OSPREY_POSTURE_SOURCE`` — stamped by the spawning call site, never derived
  from the posture value. Inferring it from ``OSPREY_EXECUTION_MODE`` would
  collapse the distinction the envelope exists to record: a ``writes`` session
  and a session nobody ever gave a posture look identical in the posture
  value, and only the marker tells them apart.
* ``OSPREY_POSTURE_SESSION`` — the posture-store key the lookup was made
  under. It is exported whenever such a key exists, *regardless* of the
  posture that key holds, so a ``writes`` session is auditable as a session
  that was checked rather than one that was never asked about.

The posture **value** stays narrowing-only (a ``sandbox`` session gets
``OSPREY_EXECUTION_MODE=readonly``; ``writes`` sets nothing and clears
nothing). The markers are not privileges, so they carry no such rule.

Three spawn sites exist and each is pinned here:

===========================================  ==================  =============
site                                         posture_source      key
===========================================  ==================  =============
``routes/chat.py`` (Simple-mode chat)        ``live``/``process``  ``chat_id``
``routes/websocket.py`` ``/ws/operator``     ``spawn``           minted key
``routes/websocket.py`` ``_build_extra_env`` ``live``            pool key
===========================================  ==================  =============

The chat site is the one that chooses. Its ``chat_id`` is caller-supplied and
the posture surface's key grammar is closed, so a key outside that grammar can
never have a store entry: such a child is stamped ``process`` — "nothing
established this posture but the environment" — rather than claiming a live
store answered for it. The shipped client mints bare UUIDs and always takes the
``live`` arm.

Harness mirrors ``test_posture_routes.py``: each test builds its own app
through ``create_app`` under a patched ``_load_web_config``, entered as a
``TestClient`` context manager so the lifespan runs.
"""

from __future__ import annotations

import ast
import inspect
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from osprey.audit.envelope import POSTURE_SOURCE_PROCESS
from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.operator_session import (
    POSTURE_SESSION_ENV,
    POSTURE_SOURCE_ENV,
    POSTURE_SOURCE_LIVE,
    POSTURE_SOURCE_SPAWN,
    build_operator_child_env,
)
from osprey.interfaces.web_terminal.pty_manager import (
    POOL_FINGERPRINT_EXCLUDED_ENV,
    env_fingerprint,
)
from osprey.interfaces.web_terminal.routes import chat as chat_routes
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes

SESSION_A = "aaaaaaaa-1111-2222-3333-444444444444"
SESSION_B = "bbbbbbbb-1111-2222-3333-444444444444"
CHAT_ID = "chat-fixture-1"

EXECUTION_MODE_ENV = "OSPREY_EXECUTION_MODE"


# ---- Harness (mirrors test_posture_routes.py) ---- #


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


def _spawn_env(client, claude_session_id, telemetry_session_id=None):
    """The extra env the next PTY spawn for this session would carry."""
    return websocket_routes._build_extra_env(
        SimpleNamespace(app=client.app),
        claude_session_id,
        telemetry_session_id,
    )


def _sdk_env(client, session_key=None, *, posture_source=POSTURE_SOURCE_LIVE):
    """The env the next SDK (operator/chat) child would carry."""
    return build_operator_child_env(
        client.app.state.project_cwd,
        session_key=session_key,
        app=client.app,
        posture_source=posture_source,
    )


def _seed_posture(client, key, posture):
    """Put *posture* in the live store under *key*.

    The SDK surfaces are keyed on identifiers the POST route cannot address
    (a chat id, a minted ``operator-<hex8>``), so the store is seeded directly
    rather than through the route.
    """
    websocket_routes._session_postures(client.app)[key] = posture


# ---- Source-level pins on the three call sites ---- #


def _builder_calls(module) -> list[ast.Call]:
    """Every ``build_operator_child_env(...)`` call in *module*'s source."""
    tree = ast.parse(Path(inspect.getsourcefile(module)).read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_operator_child_env"
    ]


def _resolve_source(node: ast.expr, module) -> tuple:
    """Every literal source *node* can evaluate to.

    Resolves the spellings a call site may use: a bare string, the
    module-level constant (``POSTURE_SOURCE_LIVE``) looked up in *module*'s
    namespace, and a conditional between two of those. The conditional is what
    the chat site needs — its key is caller-supplied, so it says ``live`` only
    for one the posture surface can actually address — and both arms are
    returned so the pin covers each. Anything else (a lookup, a call, a value
    derived from the posture) is refused here rather than silently admitted.
    """
    if isinstance(node, ast.Constant):
        return (node.value,)
    if isinstance(node, ast.Name):
        return (getattr(module, node.id),)
    if isinstance(node, ast.IfExp):
        return _resolve_source(node.body, module) + _resolve_source(node.orelse, module)
    raise AssertionError("posture_source is passed as a computed expression, not a literal source")


def _keyword_values(call: ast.Call, name: str, module) -> tuple:
    """The sources keyword *name* can be called with, or ``()`` if not passed."""
    for kw in call.keywords:
        if kw.arg == name:
            return _resolve_source(kw.value, module)
    return ()


def _keyword_condition(call: ast.Call, name: str) -> ast.expr | None:
    """The test of keyword *name*'s conditional, or ``None`` if it is not one."""
    for kw in call.keywords:
        if kw.arg == name and isinstance(kw.value, ast.IfExp):
            return kw.value.test
    return None


class TestCallSitesPassThePairExplicitly:
    """Every builder call site names its own ``posture_source``.

    A default that a call site is allowed to fall through would put the
    envelope's provenance field one refactor away from being wrong in silence,
    so the pin is on the call sites and not only on the signature.
    """

    def test_chat_site_passes_live_or_process(self):
        calls = _builder_calls(chat_routes)
        assert len(calls) == 1, "chat.py should spawn its SDK child in exactly one place"
        assert set(_keyword_values(calls[0], "posture_source", chat_routes)) == {
            POSTURE_SOURCE_LIVE,
            POSTURE_SOURCE_PROCESS,
        }

    def test_the_chat_sites_choice_is_the_key_grammar(self):
        """What the chat site branches on, pinned as well as what it passes.

        ``live`` claims a store keeps answering for this key, so the only
        honest condition is whether the posture surface can address the key at
        all. Branching on anything else — the posture value above all — would
        put a provenance in the ledger that means nothing.
        """
        condition = _keyword_condition(_builder_calls(chat_routes)[0], "posture_source")
        assert condition is not None, "the chat site no longer chooses its source"
        names = {node.id for node in ast.walk(condition) if isinstance(node, ast.Name)}
        assert "is_posture_key" in names

    def test_operator_site_passes_spawn(self):
        calls = _builder_calls(websocket_routes)
        assert len(calls) == 1, "websocket.py should spawn its SDK child in exactly one place"
        assert _keyword_values(calls[0], "posture_source", websocket_routes) == (
            POSTURE_SOURCE_SPAWN,
        )

    def test_every_builder_call_site_is_explicit(self):
        """No in-tree caller relies on the parameter's default."""
        for module in (chat_routes, websocket_routes):
            for call in _builder_calls(module):
                assert _keyword_values(call, "posture_source", module), (
                    f"{module.__name__} calls the builder without an explicit posture_source"
                )

    def test_builder_signature_takes_an_explicit_keyword(self):
        """The source is a parameter, not something the builder works out."""
        params = inspect.signature(build_operator_child_env).parameters
        assert "posture_source" in params
        assert params["posture_source"].kind is inspect.Parameter.KEYWORD_ONLY


class TestBuilderMarkers:
    """The SDK seam: markers unconditional, posture value narrowing-only."""

    def test_writes_session_exports_both_markers_and_no_execution_mode(self, client):
        _seed_posture(client, CHAT_ID, websocket_routes.POSTURE_WRITES)
        env = _sdk_env(client, CHAT_ID)

        assert env[POSTURE_SOURCE_ENV] == POSTURE_SOURCE_LIVE
        assert env[POSTURE_SESSION_ENV] == CHAT_ID
        assert EXECUTION_MODE_ENV not in env

    def test_sandbox_session_exports_both_markers_and_readonly(self, client):
        _seed_posture(client, CHAT_ID, websocket_routes.POSTURE_SANDBOX)
        env = _sdk_env(client, CHAT_ID)

        assert env[POSTURE_SOURCE_ENV] == POSTURE_SOURCE_LIVE
        assert env[POSTURE_SESSION_ENV] == CHAT_ID
        assert env[EXECUTION_MODE_ENV] == "readonly"

    def test_unstored_key_still_exports_both_markers(self, client):
        """A key nobody has given a posture was still *checked*."""
        env = _sdk_env(client, SESSION_B)

        assert env[POSTURE_SOURCE_ENV] == POSTURE_SOURCE_LIVE
        assert env[POSTURE_SESSION_ENV] == SESSION_B
        assert EXECUTION_MODE_ENV not in env

    def test_no_session_key_exports_neither_marker(self, client):
        env = _sdk_env(client, None)

        assert POSTURE_SOURCE_ENV not in env
        assert POSTURE_SESSION_ENV not in env

    def test_operator_spawn_key_is_stamped_spawn(self, client):
        key = "operator-deadbeef"
        _seed_posture(client, key, websocket_routes.POSTURE_SANDBOX)
        env = _sdk_env(client, key, posture_source=POSTURE_SOURCE_SPAWN)

        assert env[POSTURE_SOURCE_ENV] == POSTURE_SOURCE_SPAWN
        assert env[POSTURE_SESSION_ENV] == key
        assert env[EXECUTION_MODE_ENV] == "readonly"

    def test_source_does_not_follow_the_posture_value(self, client):
        """Flipping the posture must not move the provenance marker."""
        _seed_posture(client, CHAT_ID, websocket_routes.POSTURE_SANDBOX)
        sandboxed = _sdk_env(client, CHAT_ID, posture_source=POSTURE_SOURCE_SPAWN)
        _seed_posture(client, CHAT_ID, websocket_routes.POSTURE_WRITES)
        writing = _sdk_env(client, CHAT_ID, posture_source=POSTURE_SOURCE_SPAWN)

        assert sandboxed[POSTURE_SOURCE_ENV] == writing[POSTURE_SOURCE_ENV] == POSTURE_SOURCE_SPAWN
        assert sandboxed[EXECUTION_MODE_ENV] == "readonly"
        assert EXECUTION_MODE_ENV not in writing

    def test_source_outside_the_closed_set_is_refused(self, client):
        with pytest.raises(ValueError):
            _sdk_env(client, CHAT_ID, posture_source="sandbox")


class TestPtyMarkers:
    """The PTY seam stamps ``live`` under the pool key."""

    def test_new_session_stamps_the_telemetry_key(self, client):
        env = _spawn_env(client, None, SESSION_A)

        assert env[POSTURE_SOURCE_ENV] == POSTURE_SOURCE_LIVE
        assert env[POSTURE_SESSION_ENV] == SESSION_A

    def test_resumed_session_stamps_the_claude_key(self, client):
        env = _spawn_env(client, SESSION_A)

        assert env[POSTURE_SOURCE_ENV] == POSTURE_SOURCE_LIVE
        assert env[POSTURE_SESSION_ENV] == SESSION_A

    def test_sandbox_keeps_the_narrowing_marker(self, client):
        _seed_posture(client, SESSION_A, websocket_routes.POSTURE_SANDBOX)
        env = _spawn_env(client, SESSION_A, SESSION_A)

        assert env[POSTURE_SOURCE_ENV] == POSTURE_SOURCE_LIVE
        assert env[POSTURE_SESSION_ENV] == SESSION_A
        assert env[EXECUTION_MODE_ENV] == "readonly"

    def test_writes_exports_markers_without_the_posture_value(self, client):
        _seed_posture(client, SESSION_A, websocket_routes.POSTURE_WRITES)
        env = _spawn_env(client, SESSION_A, SESSION_A)

        assert env[POSTURE_SOURCE_ENV] == POSTURE_SOURCE_LIVE
        assert env[POSTURE_SESSION_ENV] == SESSION_A
        assert EXECUTION_MODE_ENV not in env

    def test_no_posture_key_exports_neither_marker(self, client):
        env = _spawn_env(client, None, None)

        assert POSTURE_SOURCE_ENV not in env
        assert POSTURE_SESSION_ENV not in env


class TestPoolFingerprint:
    """The session marker is identity, the source marker is behaviour."""

    def test_posture_session_is_excluded(self):
        """It is absent-or-equal-to-the-pool-key, exactly like OSPREY_SESSION_ID.

        A pooled key's marker cannot name a different session than the key the
        pool already holds, so fingerprinting it would buy no safety and would
        cost a respawn every time a session is rekeyed onto its discovered
        Claude UUID.
        """
        assert POSTURE_SESSION_ENV in POOL_FINGERPRINT_EXCLUDED_ENV

        base = {"OSPREY_WEB_UX": "expert"}
        assert env_fingerprint({**base, POSTURE_SESSION_ENV: SESSION_A}) == env_fingerprint(
            {**base, POSTURE_SESSION_ENV: SESSION_B}
        )

    def test_posture_source_is_still_fingerprinted(self):
        """Deny-list discipline: a new name counts unless it is listed."""
        assert POSTURE_SOURCE_ENV not in POOL_FINGERPRINT_EXCLUDED_ENV

        base = {"OSPREY_WEB_UX": "expert"}
        assert env_fingerprint(
            {**base, POSTURE_SOURCE_ENV: POSTURE_SOURCE_LIVE}
        ) != env_fingerprint({**base, POSTURE_SOURCE_ENV: POSTURE_SOURCE_SPAWN})

    def test_pty_source_marker_never_churns_a_pooled_entry(self, client):
        """Being fingerprinted costs nothing: the PTY seam always says ``live``."""
        spawn = _spawn_env(client, None, SESSION_A)
        reattach = _spawn_env(client, SESSION_A, SESSION_A)
        switch = _spawn_env(client, SESSION_A)

        assert (
            spawn[POSTURE_SOURCE_ENV]
            == reattach[POSTURE_SOURCE_ENV]
            == switch[POSTURE_SOURCE_ENV]
            == POSTURE_SOURCE_LIVE
        )

    def test_rekey_does_not_respawn_on_the_session_marker_alone(self, client):
        """A spawn and its post-rekey reattach fingerprint identically.

        The pool key changes when the Claude UUID is discovered, so the value
        ``_build_extra_env`` computes for ``OSPREY_POSTURE_SESSION`` changes
        with it. Excluding the *name* from the fingerprint is what keeps that
        from killing the live child — and a child that is not killed keeps
        exporting the key it spawned under, because its environment was fixed
        at ``execvp`` time and no server-side rewrite can reach it.

        So the value below differs on purpose, and stabilising the export is
        the wrong fix: a genuine respawn under the new key *must* export the
        new key, or every record the fresh child emits is misfiled under a
        dead one. The join is carried instead by the registry's audit alias,
        which resolves the current pool key back to the key the running child
        actually stamped into its records.
        """
        before = _spawn_env(client, None, SESSION_A)
        after = _spawn_env(client, SESSION_B, SESSION_A)

        assert before[POSTURE_SESSION_ENV] != after[POSTURE_SESSION_ENV]
        assert env_fingerprint(before) == env_fingerprint(after)

        # The alias is what makes the differing export harmless: a toggle
        # event raised against the new key still names the spawn key the live
        # child exported above.
        registry = client.app.state.pty_registry
        registry._sessions[SESSION_A] = MagicMock(is_alive=True)
        try:
            registry.rekey_session(SESSION_A, SESSION_B, app=client.app)
            assert registry.audit_session_key(SESSION_B) == before[POSTURE_SESSION_ENV]
        finally:
            registry.terminate_session(SESSION_B)
