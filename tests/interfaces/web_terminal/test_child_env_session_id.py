"""One execution root per conversation: ``OSPREY_SESSION_ID`` on every spawn.

Both views are windows onto one session key. The key is the PTY pool key, the
chat pool key, the posture-store key — and it has to be the execution root too,
or the two children a flip puts on either side of one conversation work out of
different directories: the python and sandbox executors put their run
directories under ``<agent-data root>/sessions/<OSPREY_SESSION_ID>``, so a
child spawned without the stamp writes to the shared root and a child spawned
with it writes to a room of its own.

Three spawn paths reach a child, and all three stamp the key:

* the PTY terminal's **new** session, whose key is the id forced onto the CLI
  with ``--session-id`` (this is the one that used to carry no stamp at all),
* the PTY terminal's **resume**, connect and ``switch_session`` alike,
* the SDK child behind ``POST /api/chat`` and ``/ws/operator``.

Pinned here: each path stamps its own pool key, a keyless SDK spawn stamps
nothing, the PTY and chat envs for ONE key resolve to ONE directory, and the
name stays outside the pool fingerprint so stamping it can never make a
reattach kill a live child.

The harness mirrors ``test_agent_data_root_stamp.py``; see its ``shared_root``
fixture for why the resolver is rebound rather than the variable stamped.
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
    POSTURE_SOURCE_LIVE,
    POSTURE_SOURCE_SPAWN,
    build_operator_child_env,
)
from osprey.interfaces.web_terminal.pty_manager import env_fingerprint
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey_connectors import session_store, workspace

SESSION_ID_ENV = "OSPREY_SESSION_ID"

SESSION_A = "aaaaaaaa-1111-2222-3333-444444444444"
SESSION_B = "bbbbbbbb-1111-2222-3333-444444444444"
CHAT_ID = "cccccccc-1111-2222-3333-444444444444"
OPERATOR_KEY = "operator-deadbeef"


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
    """Stand in for the deployment's shared agent-data root."""
    root = tmp_path / "shared_agent_data"
    root.mkdir()
    monkeypatch.delenv(OSPREY_AGENT_DATA_ROOT, raising=False)
    monkeypatch.delenv(SESSION_ID_ENV, raising=False)
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
def client(workspace_dir, shared_root):
    @contextmanager
    def _make():
        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ):
            app = create_app(shell_command="echo")
            with TestClient(app) as test_client:
                yield test_client

    with _make() as c:
        yield c


def _pty_env(client, claude_session_id, telemetry_session_id=None):
    """The extra env the next PTY spawn for this session would carry."""
    return websocket_routes._build_extra_env(
        SimpleNamespace(app=client.app),
        claude_session_id,
        telemetry_session_id,
    )


def _sdk_env(client, session_key=None, *, posture_source=POSTURE_SOURCE_LIVE):
    """The env the next SDK (chat or operator) child would carry."""
    return build_operator_child_env(
        client.app.state.project_cwd,
        session_key=session_key,
        app=client.app,
        posture_source=posture_source,
    )


# --------------------------------------------------------------------------- #
# Every spawn path stamps its pool key
# --------------------------------------------------------------------------- #


class TestEverySpawnPathStampsItsKey:
    @pytest.mark.parametrize(
        ("claude_session_id", "telemetry_session_id", "pool_key"),
        [
            (None, SESSION_A, SESSION_A),  # brand-new session
            (SESSION_A, SESSION_A, SESSION_A),  # reattach
            (SESSION_A, None, SESSION_A),  # switch_session
            (SESSION_A, SESSION_B, SESSION_A),  # resumed under a second telemetry id
        ],
        ids=["new", "reattach", "switch", "resumed"],
    )
    def test_pty_spawn_stamps_the_pool_key(
        self, client, claude_session_id, telemetry_session_id, pool_key
    ):
        """The stamp is the pool key, on the new-session path as on the rest.

        A new session's id is dictated on the command line, so the handler
        knows it before the child exists — there is nothing left to discover
        and no reason for that child to be the one without an execution root
        of its own.
        """
        env = _pty_env(client, claude_session_id, telemetry_session_id)
        assert env[SESSION_ID_ENV] == pool_key

    def test_chat_spawn_stamps_the_chat_key(self, client):
        assert _sdk_env(client, CHAT_ID)[SESSION_ID_ENV] == CHAT_ID

    def test_operator_spawn_stamps_its_minted_key(self, client):
        env = _sdk_env(client, OPERATOR_KEY, posture_source=POSTURE_SOURCE_SPAWN)
        assert env[SESSION_ID_ENV] == OPERATOR_KEY

    def test_keyless_sdk_spawn_stamps_nothing(self, client):
        """No key, no scope — the child gets the render's baseline root.

        The same rule the posture pair follows: nothing names a session, so
        nothing is said about one.
        """
        assert SESSION_ID_ENV not in _sdk_env(client, None)


# --------------------------------------------------------------------------- #
# One key, one directory
# --------------------------------------------------------------------------- #


class TestBothSurfacesResolveOneRoot:
    def test_pty_and_chat_envs_for_one_key_resolve_the_same_root(self, client, tmp_path):
        """The property the stamp exists for, read through the real resolver.

        A flip tears one child down and starts the other under the same key.
        Resolved from either child's environment,
        :func:`osprey_connectors.workspace.resolve_agent_data_root` has to name
        the same directory, or the conversation's execution artefacts land in
        two places.
        """
        pty = _pty_env(client, None, SESSION_A)
        chat = _sdk_env(client, SESSION_A)

        roots = [self._resolve_under(env, tmp_path) for env in (pty, chat)]

        assert roots[0] == roots[1]
        assert roots[0] == tmp_path / "var" / "agent_data" / "sessions" / SESSION_A

    def test_a_child_with_no_stamp_keeps_the_shared_root(self, client, tmp_path):
        """Absence is the old behaviour, unchanged: no ``sessions/`` segment."""
        root = self._resolve_under(_sdk_env(client, None), tmp_path)
        assert root == tmp_path / "var" / "agent_data"

    @staticmethod
    def _resolve_under(env: dict[str, str], project_root: Path) -> Path:
        """Resolve the execution root a child holding *env* would see."""
        with (
            patch.object(workspace, "load_osprey_config", return_value={}),
            patch.object(workspace, "resolve_project_root", return_value=project_root),
            patch.dict(
                os.environ,
                {SESSION_ID_ENV: env[SESSION_ID_ENV]} if SESSION_ID_ENV in env else {},
                clear=False,
            ),
        ):
            if SESSION_ID_ENV not in env:
                os.environ.pop(SESSION_ID_ENV, None)
            return workspace.resolve_agent_data_root()


# --------------------------------------------------------------------------- #
# The stamp cannot cost a live child
# --------------------------------------------------------------------------- #


def test_the_stamp_is_outside_the_pool_fingerprint(client):
    """Adding it to a spawn env respawns nothing.

    ``OSPREY_SESSION_ID`` names the key the pool is already keyed on, so it
    carries no privilege the key does not — and fingerprinting it would make
    the first reattach to a session spawned before this stamp kill the running
    agent. Asserted through :func:`env_fingerprint` rather than against the
    exclusion set, because the fingerprint is what the registry compares.
    """
    stamped = _pty_env(client, SESSION_A, SESSION_A)
    unstamped = {k: v for k, v in stamped.items() if k != SESSION_ID_ENV}

    assert SESSION_ID_ENV in stamped
    assert env_fingerprint(stamped) == env_fingerprint(unstamped)
