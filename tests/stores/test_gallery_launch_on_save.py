"""Whether a save starts the artifact gallery is decided when the store is built.

A save from a long-lived process (the MCP server, the web terminal) launches
the gallery so the artifact is viewable at once. A save from a process that
exits right after it must not: a server thread started there dies with the
process, serves nobody, and its teardown races the interpreter's exit.
"""

from __future__ import annotations

import os
import socket
from unittest.mock import patch

import pytest

from osprey.stores.artifact_store import ArtifactStore

PNG_BYTES = b"\x89PNG\r\n\x1a\n-rendered-frame"
NPY_BYTES = b"\x93NUMPY-raw-payload"

LAUNCHER = "osprey.infrastructure.server_launcher.ensure_artifact_server"


@pytest.fixture
def agent_data_root(tmp_path, monkeypatch):
    """A workspace root under a real repo root, so ``data_file`` anchors resolve."""
    repo_root = tmp_path / "repo"
    (repo_root / "build").mkdir(parents=True)
    config_path = repo_root / "build" / "config.yml"
    config_path.write_text(
        f"project_root: {repo_root}\nagent_data:\n  base_dir: state/agent\n",
    )
    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
    monkeypatch.chdir(repo_root)
    return repo_root / "state" / "agent"


def _save_file(store: ArtifactStore):
    return store.save_file(
        file_content=b"payload",
        filename="note.txt",
        artifact_type="text",
        title="A note",
        mime_type="text/plain",
        tool_source="test",
    )


def _save_data(store: ArtifactStore):
    return store.save_data(tool="test", data={"value": 1}, title="A reading")


def _save_channel_reading(store: ArtifactStore):
    return store.save_channel_reading(
        png_bytes=PNG_BYTES,
        npy_bytes=NPY_BYTES,
        title="Camera frame",
        metadata={"channel": "SIM:CAM1:ArrayData"},
    )


SAVES = pytest.mark.parametrize(
    "save",
    [_save_file, _save_data, _save_channel_reading],
    ids=["save_file", "save_data", "save_channel_reading"],
)


@SAVES
def test_a_save_launches_the_gallery_by_default(agent_data_root, save):
    store = ArtifactStore(workspace_root=agent_data_root)

    with patch(LAUNCHER) as launch:
        entry = save(store)

    assert store.get_entry(entry.id) is not None
    launch.assert_called_once_with()


@SAVES
def test_a_store_built_without_auto_launch_saves_without_starting_the_gallery(
    agent_data_root, save
):
    store = ArtifactStore(workspace_root=agent_data_root, auto_launch=False)

    with patch(LAUNCHER) as launch:
        entry = save(store)

    assert store.get_entry(entry.id) is not None
    launch.assert_not_called()


@pytest.mark.real_server_launch
def test_a_save_behind_the_proxy_does_not_consume_the_panel_token(agent_data_root, monkeypatch):
    """An agent-side save in the multi-user shape keeps the process's panel token.

    The process holds no operator secret, so the gallery app it would build
    refuses to be built, and the refusal takes the panel-token carrier with it.
    The real launch path runs here, against a fresh launcher so no earlier
    test's launched state decides the outcome.
    """
    from osprey.infrastructure import server_launcher
    from osprey.interfaces.web_auth import (
        BIND_HOST_ENV,
        OPERATOR_SECRET_ENV,
        PANEL_TOKEN_ENV,
        peek_web_credentials,
    )
    from osprey.registry.web import FRAMEWORK_WEB_SERVERS

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    defn = FRAMEWORK_WEB_SERVERS["artifact"]
    fresh = server_launcher.ServerLauncher(
        name=defn.name,
        config_reader=lambda: ("127.0.0.1", port),
        auto_launch_checker=lambda: True,
        app_factory=server_launcher._make_app_factory(defn),
        pass_workspace=defn.pass_workspace,
    )
    monkeypatch.setitem(server_launcher._launchers, "artifact", fresh)
    monkeypatch.setenv(BIND_HOST_ENV, "127.0.0.1")
    monkeypatch.delenv(OPERATOR_SECRET_ENV, raising=False)
    monkeypatch.setenv(PANEL_TOKEN_ENV, "pty-supplied-token")

    _save_file(ArtifactStore(workspace_root=agent_data_root))

    assert os.environ.get(PANEL_TOKEN_ENV) == "pty-supplied-token"
    assert peek_web_credentials() is None
