"""Regression tests: the ArtifactStore singleton lives on the SHARED data root.

The artifact gallery is a long-lived daemon serving the shared
``_agent_data/artifacts/`` store. If an MCP server roots its ArtifactStore at
the session-relocated path (``OSPREY_SESSION_ID`` appends
``sessions/<id>/`` in ``resolve_agent_data_root``), everything the session
saves lands in a store the gallery never reads — artifacts silently vanish
from the UI. Session attribution belongs in the index
(``ArtifactEntry.session_id``), never in the store path
(see ``resolve_shared_data_root``).
"""

import sys
import threading
import types

import pytest

from osprey.stores.artifact_store import get_artifact_store

SESSION_ID = "f5c059b4-sess"


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Minimal deployed project: config.yml with auto-launch disabled."""
    (tmp_path / "config.yml").write_text(
        "agent_data:\n  base_dir: ./_agent_data\nartifact_server:\n  auto_launch: false\n"
    )
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.unit
def test_singleton_ignores_session_relocation(project, monkeypatch):
    """With OSPREY_SESSION_ID set, the store must still root at the shared path."""
    monkeypatch.setenv("OSPREY_SESSION_ID", SESSION_ID)
    from osprey.mcp_server.startup import initialize_workspace_singletons

    initialize_workspace_singletons()
    store = get_artifact_store()

    assert store.artifact_dir == project / "_agent_data" / "artifacts"
    assert "sessions" not in store.artifact_dir.parts


@pytest.mark.unit
def test_saved_entry_carries_session_tag_on_shared_root(project, monkeypatch):
    """Session isolation happens at the index level: shared path, tagged entry."""
    monkeypatch.setenv("OSPREY_SESSION_ID", SESSION_ID)
    from osprey.mcp_server.startup import initialize_workspace_singletons

    initialize_workspace_singletons()
    store = get_artifact_store()
    entry = store.save_file(
        file_content=b"<html></html>",
        filename="plot.html",
        artifact_type="plot_html",
        title="Session plot",
    )

    assert entry.session_id == SESSION_ID
    # The gallery reads the shared root — the file must physically be there.
    assert (project / "_agent_data" / "artifacts" / entry.filename).exists()


@pytest.mark.unit
def test_singleton_root_without_session_env(project):
    """Sanity: without OSPREY_SESSION_ID the root is unchanged."""
    from osprey.mcp_server.startup import initialize_workspace_singletons

    initialize_workspace_singletons()
    assert get_artifact_store().artifact_dir == project / "_agent_data" / "artifacts"


@pytest.mark.unit
def test_daemon_web_server_receives_shared_root(project, monkeypatch):
    """ServerLauncher hands daemons the shared root even in session-scoped processes.

    The gallery may be auto-launched from inside an MCP server process (on
    first artifact save) where OSPREY_SESSION_ID is set; it must still serve
    the shared store.
    """
    monkeypatch.setenv("OSPREY_SESSION_ID", SESSION_ID)
    from osprey.infrastructure.server_launcher import ServerLauncher

    seen: dict = {}
    factory_called = threading.Event()

    def factory(workspace_root=None):
        seen["root"] = workspace_root
        factory_called.set()
        return object()

    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda *a, **k: None))

    launcher = ServerLauncher(
        name="test gallery",
        config_reader=lambda: ("127.0.0.1", 65500),
        auto_launch_checker=lambda: True,
        app_factory=factory,
        pass_workspace=True,
    )
    launcher._launch_in_thread("127.0.0.1", 65500)

    assert factory_called.wait(5), "app factory was never invoked"
    assert seen["root"] == project / "_agent_data"
    assert "sessions" not in seen["root"].parts
