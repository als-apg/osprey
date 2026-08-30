"""The agent-data root stamp: one anchor, stamped as a pair with the session key.

A session child, every MCP server below it and the stdlib-only hooks beside it
all have to agree on ONE directory — the one holding the control-target state
file and the session-posture store. Left to themselves they derive it three
different ways: the controls server through config, the store reader through
config again, the hooks through a repo-root guess plus the literal
``var/agent_data``. Those derivations agree only for a deployment that never
moves ``agent_data.base_dir``, and they disagree silently, which is the worst
shape a posture answer can take: "no record under this root" then means "you
looked in the wrong place" and not "no controls server for this session".

So the spawning surface resolves the root once and stamps it as
``OSPREY_AGENT_DATA_ROOT``, and everything below prefers it.

Three properties are pinned here, and the first is the one that makes the other
two mean anything:

* **Co-stamp.** ``OSPREY_AGENT_DATA_ROOT`` travels with ``OSPREY_POSTURE_SESSION``
  and never without it, at BOTH spawn surfaces. The key names whose posture
  applies; the root names where that answer is read. A child holding one half
  is a child told whose posture to obey and left to guess where it lives — and
  the hook's fail-closed rules are written on the assumption that the halves
  arrive together.
* **Survival.** The stamp has to reach the processes that read it, which sit
  behind two deliberate scrubs: ``ConnectorHostManager.child_env()`` (drops the
  EPICS family) and ``scrub_sandbox_child_env`` (drops credentials and the
  web-terminal address book). Both are allow-by-default, so this is a
  regression pin rather than a new guarantee: the day one of them grows a
  prefix rule, the connector-host child or the execution sandbox silently
  starts resolving a different directory from its own parent.
* **Absence is the old behaviour, exactly.** With the variable unset —
  a CLI run, a dispatch worker, a controls server outside any web session, and
  every test that patches ``target_state.resolve_shared_data_root`` — the
  derivation is untouched.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

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
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.connector_host_manager import ConnectorHostManager
from osprey.mcp_server.control_system.server_context import MCPServerConfig
from osprey.mcp_server.sandbox_env import scrub_sandbox_child_env
from osprey_connectors import session_store

SESSION_A = "aaaaaaaa-1111-2222-3333-444444444444"
SESSION_B = "bbbbbbbb-1111-2222-3333-444444444444"
CHAT_ID = "cccccccc-1111-2222-3333-444444444444"
OPERATOR_KEY = "operator-deadbeef"

POSTURE_SANDBOX = websocket_routes.POSTURE_SANDBOX
POSTURE_WRITES = websocket_routes.POSTURE_WRITES


# --------------------------------------------------------------------------- #
# Harness (mirrors tests/interfaces/web_terminal/test_posture_source_pin.py)
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

    ``session_store``'s own binding is rebound too, imported by name at module
    load, or the posture store these tests read would be the real repo's
    ``var/agent_data/control_target/session-postures.json``.

    The delenv is what makes that rebinding do anything at all:
    ``session_store.agent_data_root()`` reads the variable FIRST and falls back
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
        patch.object(session_store, "resolve_shared_data_root", return_value=root),
    ):
        session_store.invalidate_cache()
        yield root
        session_store.invalidate_cache()


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


def _seed_posture(client, key, posture):
    """Put *posture* in the live store under *key*."""
    websocket_routes._session_postures(client.app)[key] = posture


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

    def test_pty_spawn_with_no_key_stamps_neither(self, client):
        """No key to name means no posture to read: the root would say nothing."""
        env = _pty_env(client, None, None)
        assert POSTURE_SESSION_ENV not in env
        assert OSPREY_AGENT_DATA_ROOT not in env

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
    def test_the_pair_does_not_depend_on_the_posture(self, client, shared_root, posture):
        """The markers are not a privilege; the narrowing-only rule is the
        posture VALUE's alone. A ``writes`` session and one nobody ever gave a
        posture are both auditable, and both know where their store lives.
        """
        if posture is not None:
            _seed_posture(client, SESSION_A, posture)
            _seed_posture(client, CHAT_ID, posture)

        pty = _pty_env(client, SESSION_A)
        sdk = _sdk_env(client, CHAT_ID)

        for env in (pty, sdk):
            assert env[OSPREY_AGENT_DATA_ROOT] == str(shared_root)
            assert POSTURE_SESSION_ENV in env
            assert POSTURE_SOURCE_ENV in env

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
    """A process environment carrying the pair, plus a scrub canary."""
    monkeypatch.setenv(OSPREY_AGENT_DATA_ROOT, "/deployments/als/var/agent_data")
    monkeypatch.setenv(POSTURE_SESSION_ENV, SESSION_A)
    monkeypatch.setenv(POSTURE_SOURCE_ENV, POSTURE_SOURCE_LIVE)
    monkeypatch.setenv("EPICS_CA_ADDR_LIST", "10.0.0.1")
    monkeypatch.setenv("OSPREY_TERMINAL_SECRET", "canary")
    return os.environ


class TestTheStampSurvivesEveryScrub:
    """The readers sit behind two deliberate narrowings; both must let it past."""

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

    def test_execution_sandbox_keeps_the_pair(self, stamped_env):
        """The executor's sandbox re-reads the state file before every write."""
        scrubbed = scrub_sandbox_child_env(stamped_env)

        assert scrubbed[OSPREY_AGENT_DATA_ROOT] == stamped_env[OSPREY_AGENT_DATA_ROOT]
        assert scrubbed[POSTURE_SESSION_ENV] == SESSION_A
        assert "OSPREY_TERMINAL_SECRET" not in scrubbed, "the credential scrub still runs"


# --------------------------------------------------------------------------- #
# The writer prefers the same anchor
# --------------------------------------------------------------------------- #


class TestStateDirPrefersTheStamp:
    """One resolution rule for the writer and every reader of its directory."""

    def test_the_stamp_wins_over_the_config_derivation(self, tmp_path, monkeypatch):
        stamped = tmp_path / "stamped"
        monkeypatch.setenv(OSPREY_AGENT_DATA_ROOT, str(stamped))
        monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: tmp_path / "config")

        assert target_state.state_dir() == stamped / target_state.STATE_DIR_NAME

    def test_the_state_file_lands_under_the_stamped_root(self, tmp_path, monkeypatch):
        """Not just the directory: what a reader globs for is under it too."""
        stamped = tmp_path / "stamped"
        monkeypatch.setenv(OSPREY_AGENT_DATA_ROOT, str(stamped))
        monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: tmp_path / "config")

        target_state.write_on_start(target_state.TARGET_LIVE, server_pid=4321)

        written = list((stamped / target_state.STATE_DIR_NAME).glob(target_state.STATE_FILE_GLOB))
        assert [p.name for p in written] == ["target_state_4321.json"]
        assert target_state.read(4321)["target"] == target_state.TARGET_LIVE
        assert not (tmp_path / "config").exists(), "the config derivation was consulted"

    def test_unset_is_the_old_derivation_exactly(self, tmp_path, monkeypatch):
        """The pin every existing test in this tree leans on: with no stamp,
        patching ``target_state.resolve_shared_data_root`` still decides the
        directory, unchanged and unconsulted by anything else.
        """
        monkeypatch.delenv(OSPREY_AGENT_DATA_ROOT, raising=False)
        monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: tmp_path)

        assert target_state.state_dir() == tmp_path / target_state.STATE_DIR_NAME
        assert target_state.state_file_path(99) == (
            tmp_path / target_state.STATE_DIR_NAME / "target_state_99.json"
        )

    def test_an_empty_stamp_is_no_stamp(self, tmp_path, monkeypatch):
        """``env=""`` is how a shell spells "unset" by accident; an empty root
        would resolve to the process CWD, which is nobody's agent-data root.
        """
        monkeypatch.setenv(OSPREY_AGENT_DATA_ROOT, "")
        monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: tmp_path)

        assert target_state.state_dir() == tmp_path / target_state.STATE_DIR_NAME
