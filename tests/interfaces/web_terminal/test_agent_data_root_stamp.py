"""The agent-data root stamp: one anchor, stamped as a pair with the session key.

A session child, every MCP server below it and the stdlib-only hooks beside it
all have to agree on ONE directory — the one holding the control-context record
and the servers' reports beside it. Left to themselves they derive it three
different ways: the controls server through config, the store reader through
config again, the hooks through a repo-root guess plus the literal
``var/agent_data``. Those derivations agree only for a deployment that never
moves ``agent_data.base_dir``, and they disagree silently, which is the worst
shape a posture answer can take: "no record under this root" then means "you
looked in the wrong place" and not "no controls server for this session".

So the spawning surface resolves the root once and stamps it as
``OSPREY_AGENT_DATA_ROOT``, and everything below prefers it.

Three properties are pinned here, and the first is the one that makes the others
mean anything:

* **Co-stamp.** ``OSPREY_AGENT_DATA_ROOT`` travels with ``OSPREY_POSTURE_SESSION``
  and never without it, at BOTH spawn surfaces. The key names whose posture
  applies; the root names where that answer is read. A child holding one half
  is a child told whose posture to obey and left to guess where it lives — and
  the hook's fail-closed rules are written on the assumption that the halves
  arrive together.
* **Survival.** The stamp has to reach the processes that read it, which sit
  behind two deliberate scrubs. ``ConnectorHostManager.child_env()`` drops the
  EPICS family and is otherwise allow-by-default, so its test here is a
  regression pin: the day it grows a prefix rule, the connector-host child
  silently resolves a different directory from its parent. The execution
  sandbox's ``scrub_sandbox_child_env`` is an allowlist, pinned name by name in
  its own suite.
* **Provenance, and never a posture.** ``OSPREY_POSTURE_SOURCE`` rides with
  the pair and names where the posture decision came from — ``live`` for a
  key the posture route can address, ``spawn`` for a minted operator key,
  ``process`` for a chat key it cannot — stamped by the call site, never
  derived from the posture value. No seam stamps ``OSPREY_EXECUTION_MODE``:
  the narrowing is per target and read live from the record, so the spawn
  environment is the same whatever the record says, and a flip lands on a
  running child instead of respawning it.

The record's directory is pinned against ``target_state``'s own derivation in
``tests/mcp_server/test_target_state.py``.
"""

from __future__ import annotations

import ast
import inspect
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.audit.envelope import POSTURE_SOURCE_PROCESS
from osprey.audit.posture import OSPREY_AGENT_DATA_ROOT
from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.operator_session import (
    POSTURE_SESSION_ENV,
    POSTURE_SOURCE_ENV,
    POSTURE_SOURCE_LIVE,
    POSTURE_SOURCE_SPAWN,
    build_operator_child_env,
    resolve_agent_data_root,
)
from osprey.interfaces.web_terminal.pty_manager import env_fingerprint
from osprey.interfaces.web_terminal.routes import chat as chat_routes
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey.mcp_server.control_system.connector_host_manager import ConnectorHostManager
from osprey.mcp_server.control_system.server_context import MCPServerConfig
from osprey_connectors import posture_store

SESSION_A = "aaaaaaaa-1111-2222-3333-444444444444"
SESSION_B = "bbbbbbbb-1111-2222-3333-444444444444"
CHAT_ID = "cccccccc-1111-2222-3333-444444444444"
OPERATOR_KEY = "operator-deadbeef"

EXECUTION_MODE_ENV = "OSPREY_EXECUTION_MODE"

POSTURE_SANDBOX = websocket_routes.POSTURE_SANDBOX
POSTURE_WRITES = websocket_routes.POSTURE_WRITES


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


@pytest.fixture
def shared_root(tmp_path, monkeypatch):
    """Stand in for the deployment's shared agent-data root.

    Rebinding the resolver, not stamping the variable, for two reasons — the
    first of which makes the choice mandatory rather than merely preferable:

    * ``resolve_agent_data_root`` (``operator_session.py``) imports
      ``resolve_shared_data_root`` at call time and never reads
      ``OSPREY_AGENT_DATA_ROOT`` at all. A stamp would not reach it, so the
      root these spawn seams report would still be the real repo's.
    * the stamp is also what this file pins as *absent* from a keyless spawn.
      The SDK seam is the one that would break: ``build_operator_child_env`` →
      ``build_clean_env`` → ``build_base_child_env`` starts from
      ``dict(os.environ)``, so a stamp in this process would arrive in the
      child and ``test_sdk_spawn_with_no_key_stamps_neither`` (and
      ``test_neither_surface_ever_stamps_one_half``) would pass for the wrong
      reason. The PTY seam is immune — ``_build_extra_env`` starts from an
      empty dict and only adds — which is exactly why naming the PTY case here
      would misstate the risk.

    ``posture_store``'s own binding is rebound too, imported by name at module
    load, or the record these tests read would be the real repo's
    ``var/agent_data/control_target/control_context.json``.

    The delenv is what makes that rebinding do anything at all:
    ``posture_store.agent_data_root()`` reads the variable FIRST and falls back
    to the resolver only when it is unset. The suite-wide
    ``session_posture_leak_guard`` (``tests/conftest.py``) POINTS the variable
    at a throwaway root rather than clearing it, so here it must be cleared
    explicitly — without this line the fixture's patches would be inert and
    these spawn seams would report the guard's tmp root instead of the
    resolver's.
    """
    root = tmp_path / "shared_agent_data"
    root.mkdir()
    monkeypatch.delenv(OSPREY_AGENT_DATA_ROOT, raising=False)
    with (
        patch(
            "osprey_connectors.workspace.resolve_shared_data_root",
            return_value=root,
        ),
        patch.object(posture_store, "resolve_shared_data_root", return_value=root),
    ):
        posture_store.invalidate_cache()
        yield root
        posture_store.invalidate_cache()


# ``shared_root`` rebinds the agent-data resolver every app this factory builds reads.
@pytest.fixture
def make_client(workspace_dir, shared_root):  # noqa: ARG001
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


def _pty_env(client, claude_session_id, telemetry_session_id=None):
    """The extra env the next PTY spawn for this session would carry."""
    return websocket_routes._build_extra_env(
        SimpleNamespace(app=client.app),
        claude_session_id,
        telemetry_session_id,
    )


def _sdk_env(client, session_key=None, *, posture_source=POSTURE_SOURCE_LIVE, app=...):
    """The env the next SDK (operator/chat) child would carry."""
    return build_operator_child_env(
        client.app.state.project_cwd,
        session_key=session_key,
        app=client.app if app is ... else app,
        posture_source=posture_source,
    )


def _seed_posture(write_control_context, root, posture):
    """Record *posture* for every target on the deployment under *root*.

    The narrowing is deployment-wide now, so there is no key to seed it under:
    ``sandbox`` narrows every target and ``writes`` is spelled by the absence
    of an entry. Either way it is a fact about the record, and the point of
    the tests that call this is that neither shape moves the stamped pair.
    """
    narrowed = posture == POSTURE_SANDBOX
    write_control_context(
        root,
        posture=dict.fromkeys(posture_store.CONTROL_TARGETS, POSTURE_SANDBOX) if narrowed else {},
    )


# --------------------------------------------------------------------------- #
# The pair
# --------------------------------------------------------------------------- #


class TestTheStampTravelsWithTheSessionKey:
    """Both halves or neither — at both spawn surfaces, in every shape."""

    @pytest.mark.parametrize(
        ("claude_session_id", "telemetry_session_id"),
        [
            (None, SESSION_A),  # brand-new session: the telemetry id is the key
            (SESSION_A, SESSION_A),  # reattach
            (SESSION_A, None),  # switch_session
            (SESSION_A, SESSION_B),  # resumed under a second telemetry id
        ],
        ids=["new", "reattach", "switch", "resumed"],
    )
    def test_pty_spawn_stamps_both(
        self, client, shared_root, claude_session_id, telemetry_session_id
    ):
        env = _pty_env(client, claude_session_id, telemetry_session_id)
        assert env[POSTURE_SESSION_ENV] == (claude_session_id or telemetry_session_id)
        assert env[OSPREY_AGENT_DATA_ROOT] == str(shared_root)
        # A PTY pool key is always one the posture route can address, so every
        # spawn shape says ``live`` and the fingerprinted marker never churns.
        assert env[POSTURE_SOURCE_ENV] == POSTURE_SOURCE_LIVE

    def test_pty_spawn_with_no_key_stamps_neither(self, client):
        """No key to name means no posture to read: the root would say nothing."""
        env = _pty_env(client, None, None)
        assert POSTURE_SESSION_ENV not in env
        assert OSPREY_AGENT_DATA_ROOT not in env
        assert POSTURE_SOURCE_ENV not in env

    @pytest.mark.parametrize(
        ("session_key", "posture_source"),
        [
            (CHAT_ID, POSTURE_SOURCE_LIVE),  # POST /api/chat
            (OPERATOR_KEY, POSTURE_SOURCE_SPAWN),  # /ws/operator
        ],
        ids=["chat", "operator"],
    )
    def test_sdk_spawn_stamps_both(self, client, shared_root, session_key, posture_source):
        env = _sdk_env(client, session_key, posture_source=posture_source)
        assert env[POSTURE_SESSION_ENV] == session_key
        assert env[OSPREY_AGENT_DATA_ROOT] == str(shared_root)

    def test_sdk_spawn_with_no_key_stamps_neither(self, client):
        env = _sdk_env(client, None)
        assert POSTURE_SESSION_ENV not in env
        assert OSPREY_AGENT_DATA_ROOT not in env
        assert POSTURE_SOURCE_ENV not in env

    def test_sdk_spawn_stamps_both_even_without_an_app(self, client, shared_root):
        """*app* is the store's handle, not the root's: the root reads config.

        A caller with a key but no app gets no posture lookup — and still gets
        the pair, because the child can be told which key governs it and where
        to look even when this process never consulted the store.
        """
        env = _sdk_env(client, CHAT_ID, app=None)
        assert env[POSTURE_SESSION_ENV] == CHAT_ID
        assert env[OSPREY_AGENT_DATA_ROOT] == str(shared_root)

    @pytest.mark.parametrize("posture", [POSTURE_SANDBOX, POSTURE_WRITES, None])
    def test_the_pair_does_not_depend_on_the_posture(
        self, client, shared_root, write_control_context, posture
    ):
        """The markers are not a privilege; the narrowing-only rule is the
        posture VALUE's alone. A narrowed deployment and one nobody ever gave a
        posture are both auditable, and both know where their record lives.
        """
        if posture is not None:
            _seed_posture(write_control_context, shared_root, posture)

        pty = _pty_env(client, SESSION_A)
        sdk = _sdk_env(client, CHAT_ID)

        for env in (pty, sdk):
            assert env[OSPREY_AGENT_DATA_ROOT] == str(shared_root)
            assert env[POSTURE_SOURCE_ENV] == POSTURE_SOURCE_LIVE
        assert pty[POSTURE_SESSION_ENV] == SESSION_A
        assert sdk[POSTURE_SESSION_ENV] == CHAT_ID

    def test_neither_surface_ever_stamps_one_half(self, client):
        """The property itself, over every shape either surface can produce."""
        envs = [
            _pty_env(client, None, None),
            _pty_env(client, None, SESSION_A),
            _pty_env(client, SESSION_A, None),
            _pty_env(client, SESSION_A, SESSION_B),
            _sdk_env(client, None),
            _sdk_env(client, CHAT_ID),
            _sdk_env(client, OPERATOR_KEY, posture_source=POSTURE_SOURCE_SPAWN),
            _sdk_env(client, None, app=None),
            _sdk_env(client, CHAT_ID, app=None),
        ]
        for env in envs:
            assert (POSTURE_SESSION_ENV in env) == (OSPREY_AGENT_DATA_ROOT in env), (
                f"half a pair: session={env.get(POSTURE_SESSION_ENV)!r} "
                f"root={env.get(OSPREY_AGENT_DATA_ROOT)!r}"
            )

    def test_the_root_is_the_shared_one_not_the_session_scoped_one(self, client, shared_root):
        """``resolve_agent_data_root`` answers the SHARED root on purpose: the
        state file and the posture store span sessions, and a path carrying
        ``sessions/<OSPREY_SESSION_ID>`` could not be reproduced by a reader
        outside the session's own environment.
        """
        assert resolve_agent_data_root(client.app) == str(shared_root)
        assert "sessions" not in Path(_pty_env(client, SESSION_A)[OSPREY_AGENT_DATA_ROOT]).parts

    def test_an_unresolvable_root_still_stamps_the_pair(self, client, workspace_dir):
        """A config load can fail transiently, and half a pair is worse than a
        fallback: the store's own resolution falls back to the workspace dir,
        so the stamp does too and writer and readers stay on ONE directory.
        """
        with patch(
            "osprey_connectors.workspace.resolve_shared_data_root",
            side_effect=RuntimeError("no config"),
        ):
            env = _pty_env(client, SESSION_A)

        assert env[POSTURE_SESSION_ENV] == SESSION_A
        assert env[OSPREY_AGENT_DATA_ROOT] == str(workspace_dir)


# --------------------------------------------------------------------------- #
# Survival through the two scrubs
# --------------------------------------------------------------------------- #


@pytest.fixture
def stamped_env(monkeypatch):
    """A process environment carrying the pair, plus an EPICS scrub canary."""
    monkeypatch.setenv(OSPREY_AGENT_DATA_ROOT, "/deployments/als/var/agent_data")
    monkeypatch.setenv(POSTURE_SESSION_ENV, SESSION_A)
    monkeypatch.setenv(POSTURE_SOURCE_ENV, POSTURE_SOURCE_LIVE)
    monkeypatch.setenv("EPICS_CA_ADDR_LIST", "10.0.0.1")
    return os.environ


class TestTheStampSurvivesEveryScrub:
    """The connector-host child's EPICS scrub must let the pair past.

    The execution sandbox's allowlist is the other scrub; its own suite,
    ``tests/mcp_server/test_sandbox_child_env_allowlist.py``, pins both names
    on it.
    """

    def test_connector_host_child_keeps_the_pair(self, stamped_env):
        """The connector-host child reads the store per write. It also has the
        EPICS family taken away from it — this asserts the scrub stayed as
        narrow as its docstring says.
        """
        manager = ConnectorHostManager(
            MCPServerConfig(
                raw={"control_system": {"connector": {"type": "mock"}}}, config_path=None
            )
        )
        child = manager.child_env()

        assert child[OSPREY_AGENT_DATA_ROOT] == stamped_env[OSPREY_AGENT_DATA_ROOT]
        assert child[POSTURE_SESSION_ENV] == SESSION_A
        assert "EPICS_CA_ADDR_LIST" not in child, "the EPICS scrub is what this env proves"


# --------------------------------------------------------------------------- #
# Provenance: which source each spawn site stamps
# --------------------------------------------------------------------------- #


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


class TestPostureSource:
    """Every builder call site names its own ``posture_source``, and the builder stamps it.

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

    @pytest.mark.parametrize("posture", [POSTURE_SANDBOX, POSTURE_WRITES, None])
    def test_sdk_markers_and_no_execution_mode(
        self, client, shared_root, write_control_context, posture
    ):
        """A narrowed deployment is stamped, not sandboxed at spawn.

        The narrowing is in the record; the child is handed the key and the
        root it reads that record out of, and nothing else. Stamping
        ``OSPREY_EXECUTION_MODE`` here would sandbox EVERY target for the
        session, which is the one thing a per-target narrowing must not do —
        and a run under a record nobody narrowed was still *checked*.
        """
        if posture is not None:
            _seed_posture(write_control_context, shared_root, posture)
        env = _sdk_env(client, CHAT_ID)

        assert env[POSTURE_SOURCE_ENV] == POSTURE_SOURCE_LIVE
        assert env[POSTURE_SESSION_ENV] == CHAT_ID
        assert EXECUTION_MODE_ENV not in env

    @pytest.mark.parametrize("posture", [POSTURE_SANDBOX, POSTURE_WRITES])
    def test_operator_spawn_key_is_stamped_spawn(
        self, client, shared_root, write_control_context, posture
    ):
        """The operator's minted key says ``spawn``, whatever the record narrows."""
        _seed_posture(write_control_context, shared_root, posture)
        env = _sdk_env(client, OPERATOR_KEY, posture_source=POSTURE_SOURCE_SPAWN)

        assert env[POSTURE_SOURCE_ENV] == POSTURE_SOURCE_SPAWN
        assert env[POSTURE_SESSION_ENV] == OPERATOR_KEY
        assert EXECUTION_MODE_ENV not in env

    def test_source_outside_the_closed_set_is_refused(self, client):
        with pytest.raises(ValueError):
            _sdk_env(client, CHAT_ID, posture_source="sandbox")


class TestNoExecutionMode:
    """Neither seam reads the record to build the env: a flip churns no pool."""

    @pytest.mark.parametrize("posture", [POSTURE_SANDBOX, POSTURE_WRITES])
    def test_neither_seam_carries_an_execution_mode(
        self, client, shared_root, write_control_context, posture
    ):
        _seed_posture(write_control_context, shared_root, posture)

        assert EXECUTION_MODE_ENV not in _pty_env(client, SESSION_A, SESSION_A)
        assert EXECUTION_MODE_ENV not in _sdk_env(client, SESSION_A)

    def test_neither_seam_reads_the_record_to_build_the_env(
        self, client, shared_root, write_control_context
    ):
        """Spawning under a narrowing and without one produce the same overlay.

        That equality is what lets a flip land on a running child instead of
        killing it.
        """
        _seed_posture(write_control_context, shared_root, POSTURE_WRITES)
        unnarrowed = _pty_env(client, SESSION_A)
        _seed_posture(write_control_context, shared_root, POSTURE_SANDBOX)
        narrowed = _pty_env(client, SESSION_A)

        assert narrowed == unnarrowed


class TestPoolFingerprint:
    """The session marker is identity, the source marker is behaviour."""

    def test_posture_source_is_still_fingerprinted(self):
        """Deny-list discipline: a new name counts unless it is listed."""
        base = {"OSPREY_WEB_UX": "expert"}
        assert env_fingerprint(
            {**base, POSTURE_SOURCE_ENV: POSTURE_SOURCE_LIVE}
        ) != env_fingerprint({**base, POSTURE_SOURCE_ENV: POSTURE_SOURCE_SPAWN})

    def test_a_moved_session_marker_does_not_respawn_a_live_child(self, client):
        """Two spawn shapes whose marker differs fingerprint identically.

        ``_build_extra_env`` computes ``OSPREY_POSTURE_SESSION`` from the pool
        key, so two calls that resolve different keys export different values.
        Excluding the *name* from the fingerprint is what keeps that from
        killing a live child — and a child that is not killed keeps exporting
        the key it spawned under, because its environment was fixed at
        ``execvp`` time and no server-side rewrite can reach it.

        Stabilising the export would be the wrong fix: a genuine respawn under
        a new key *must* export the new key, or every record the fresh child
        emits is misfiled under a dead one.
        """
        before = _pty_env(client, None, SESSION_A)
        after = _pty_env(client, SESSION_B, SESSION_A)

        assert before[POSTURE_SESSION_ENV] != after[POSTURE_SESSION_ENV]
        assert env_fingerprint(before) == env_fingerprint(after)
