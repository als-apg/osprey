"""The shipped example artifact: seeded once, deletable for good, invisible to the agent.

Covers:
  - the store marks ``origin: "demo"`` entries and writes the removed-sentinel
    when one is deleted by any path
  - the seeder writes the shipped HTML into an empty store exactly once and
    honours the sentinel
  - the gallery app seeds only when ``artifact_server.example_artifact`` is on
  - the human DELETE route makes the removal stick across app restarts
  - every agent-facing listing (artifact_list, session_summary, the run
    descriptor, the web terminal's session summary) skips the example
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.artifacts.app import create_app
from osprey.interfaces.artifacts.example_artifact import (
    EXAMPLE_TITLE,
    example_artifact_html,
    seed_example_artifact,
)
from osprey.stores.artifact_store import (
    EXAMPLE_ORIGIN,
    EXAMPLE_REMOVED_SENTINEL,
    ArtifactStore,
    initialize_artifact_store,
)
from tests.mcp_server.conftest import get_tool_fn


def _save_own(store: ArtifactStore, title: str = "Mine") -> object:
    return store.save_file(
        file_content=b"<html></html>",
        filename="mine.html",
        artifact_type="plot_html",
        title=title,
        mime_type="text/html",
        tool_source="test",
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TestStoreExampleOrigin:
    def test_shipped_html_is_a_plotly_page_without_bundled_plotlyjs(self):
        html = example_artifact_html()
        assert html.lstrip().startswith("<html>")
        assert "plotly-graph-div" in html
        # The gallery injects its vendored plotly.js; the file must not carry
        # a copy of the library (that would be megabytes) nor a CDN tag.
        assert "cdn.plot.ly" not in html
        assert len(html) < 200_000

    def test_save_file_accepts_an_explicit_session_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OSPREY_SESSION_ID", "sess-env")
        store = ArtifactStore(workspace_root=tmp_path)
        tagged = _save_own(store)
        untagged = store.save_file(
            file_content=b"x",
            filename="x.txt",
            artifact_type="text",
            title="untagged",
            session_id="",
        )
        assert tagged.session_id == "sess-env"
        assert untagged.session_id == ""

    def test_list_entries_can_exclude_examples(self, tmp_path):
        store = ArtifactStore(workspace_root=tmp_path)
        own = _save_own(store)
        demo = store.save_file(
            file_content=b"<html></html>",
            filename="demo.html",
            artifact_type="plot_html",
            title="Example",
            origin=EXAMPLE_ORIGIN,
        )
        assert {e.id for e in store.list_entries()} == {own.id, demo.id}
        assert [e.id for e in store.list_entries(exclude_examples=True)] == [own.id]

    def test_deleting_an_example_writes_the_removed_sentinel(self, tmp_path):
        store = ArtifactStore(workspace_root=tmp_path)
        sentinel = store.artifact_dir / EXAMPLE_REMOVED_SENTINEL
        own = _save_own(store)
        assert store.delete_entry(own.id)
        assert not sentinel.exists(), (
            "deleting the user's own artifact must not mark the example removed"
        )
        assert not store.example_removed

        demo = store.save_file(
            file_content=b"<html></html>",
            filename="demo.html",
            artifact_type="plot_html",
            title="Example",
            origin=EXAMPLE_ORIGIN,
        )
        assert store.delete_entry(demo.id)
        assert sentinel.exists()
        assert store.example_removed

    def test_bulk_delete_of_an_example_also_writes_the_sentinel(self, tmp_path):
        store = ArtifactStore(workspace_root=tmp_path)
        store.save_file(
            file_content=b"<html></html>",
            filename="demo.html",
            artifact_type="plot_html",
            title="Example",
            origin=EXAMPLE_ORIGIN,
        )
        store.delete_everything()
        assert store.example_removed


# ---------------------------------------------------------------------------
# Seeder
# ---------------------------------------------------------------------------


class TestSeeder:
    def test_seeds_the_example_into_an_empty_store_once(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OSPREY_SESSION_ID", "sess-env")
        store = ArtifactStore(workspace_root=tmp_path)

        first = seed_example_artifact(store)
        assert first is not None
        assert first.origin == EXAMPLE_ORIGIN
        assert first.title == EXAMPLE_TITLE
        assert first.artifact_type == "plot_html"
        assert first.mime_type == "text/html"
        assert first.tool_source == "osprey"
        assert first.session_id == "", "an example belongs to no session"
        assert first.run_id == ""
        assert (store.artifact_dir / first.filename).read_text(encoding="utf-8") == (
            example_artifact_html()
        )

        assert seed_example_artifact(store) is None
        assert seed_example_artifact(ArtifactStore(workspace_root=tmp_path)) is None
        assert len(store.list_entries()) == 1

    def test_seeds_beside_existing_work(self, tmp_path):
        store = ArtifactStore(workspace_root=tmp_path)
        _save_own(store)
        assert seed_example_artifact(store) is not None
        assert len(store.list_entries()) == 2

    def test_does_not_reseed_after_the_example_was_removed(self, tmp_path):
        store = ArtifactStore(workspace_root=tmp_path)
        entry = seed_example_artifact(store)
        assert entry is not None
        store.delete_entry(entry.id)
        assert seed_example_artifact(store) is None
        assert seed_example_artifact(ArtifactStore(workspace_root=tmp_path)) is None
        assert store.list_entries() == []

    def test_seeding_is_not_attributed_to_the_agent(self, tmp_path):
        from osprey.stores.artifact_store import (
            current_artifact_mutation_actor,
            register_artifact_listener,
            unregister_artifact_listener,
        )

        actors: list[str] = []

        def record(_entry):
            actors.append(current_artifact_mutation_actor())

        register_artifact_listener(record)
        try:
            seed_example_artifact(ArtifactStore(workspace_root=tmp_path))
        finally:
            unregister_artifact_listener(record)
        assert actors == ["system"]


# ---------------------------------------------------------------------------
# Gallery app
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Point the config chain at a throwaway ``config.yml`` and undo it after."""
    import osprey.utils.config as config_module

    monkeypatch.setattr(config_module, "_default_config", config_module._default_config)
    monkeypatch.setattr(config_module, "_default_configurable", config_module._default_configurable)

    def write(body: str) -> Path:
        path = tmp_path / "render" / "config.yml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        monkeypatch.setenv("OSPREY_CONFIG", str(path))
        return path

    return write


class TestGalleryAppSeeding:
    def test_no_config_key_means_no_example(self, tmp_path, isolated_config):
        isolated_config("project_name: t\n")
        client = TestClient(create_app(workspace_root=tmp_path / "ws"))
        assert client.get("/api/artifacts").json()["count"] == 0

    def test_key_off_means_no_example(self, tmp_path, isolated_config):
        isolated_config("artifact_server:\n  example_artifact: false\n")
        client = TestClient(create_app(workspace_root=tmp_path / "ws"))
        assert client.get("/api/artifacts").json()["count"] == 0

    def test_key_on_seeds_once_and_serves_the_page(self, tmp_path, isolated_config):
        isolated_config("artifact_server:\n  example_artifact: true\n")
        ws = tmp_path / "ws"

        client = TestClient(create_app(workspace_root=ws))
        listing = client.get("/api/artifacts").json()
        assert listing["count"] == 1
        (entry,) = listing["artifacts"]
        assert entry["origin"] == EXAMPLE_ORIGIN
        assert entry["title"] == EXAMPLE_TITLE

        page = client.get(f"/files/{entry['id']}/{entry['filename']}")
        assert page.status_code == 200
        assert "plotly-graph-div" in page.text

        # A second process start finds the example already there.
        again = TestClient(create_app(workspace_root=ws))
        assert again.get("/api/artifacts").json()["count"] == 1

    def test_human_delete_sticks_across_restarts(self, tmp_path, isolated_config):
        isolated_config("artifact_server:\n  example_artifact: true\n")
        ws = tmp_path / "ws"

        client = TestClient(create_app(workspace_root=ws))
        (entry,) = client.get("/api/artifacts").json()["artifacts"]
        assert client.delete(f"/api/artifacts/{entry['id']}").status_code == 200
        assert (ws / "artifacts" / EXAMPLE_REMOVED_SENTINEL).exists()

        again = TestClient(create_app(workspace_root=ws))
        assert again.get("/api/artifacts").json()["count"] == 0

    def test_session_scoped_listing_still_shows_the_example(self, tmp_path, isolated_config):
        """The gallery narrows to the current session; an example is untagged
        and therefore shown beside that session's own work."""
        isolated_config("artifact_server:\n  example_artifact: true\n")
        client = TestClient(create_app(workspace_root=tmp_path / "ws"))
        listing = client.get("/api/artifacts", params={"session_id": "some-session"}).json()
        assert listing["count"] == 1


# ---------------------------------------------------------------------------
# The agent never sees it
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_store(tmp_path):
    store = initialize_artifact_store(workspace_root=tmp_path)
    seed_example_artifact(store)
    _save_own(store, title="Real work")
    return store


class TestAgentFacingListingsSkipTheExample:
    @pytest.mark.asyncio
    async def test_artifact_list_tool(self, seeded_store):
        from osprey.mcp_server.workspace.tools.artifact_query import artifact_list

        result = json.loads(await get_tool_fn(artifact_list)())
        assert [e["title"] for e in result["entries"]] == ["Real work"]
        assert result["total_entries"] == 1

    @pytest.mark.asyncio
    async def test_session_summary_tool(self, seeded_store):
        from osprey.mcp_server.workspace.tools.session_summary import session_summary

        result = json.loads(await get_tool_fn(session_summary)())
        assert [e["title"] for e in result["entries"]] == ["Real work"]

    def test_run_descriptor(self, seeded_store, monkeypatch):
        from osprey.agent_runner import artifact_resolve

        monkeypatch.setattr(artifact_resolve, "_get_store", lambda: seeded_store)
        # An example is never tagged with a run, so the strict run filter
        # already drops it; pin that so a future relaxation cannot leak it.
        produced = seeded_store.save_file(
            file_content=b"x",
            filename="r.txt",
            artifact_type="text",
            title="run output",
            run_id="run-1",
        )
        described = artifact_resolve.describe_run_artifacts("run-1")
        assert [d["artifact_id"] for d in described] == [produced.id]

    def test_web_terminal_session_summary_route(self, seeded_store, tmp_path):
        from fastapi import FastAPI

        from osprey.interfaces.web_terminal.routes.session import router

        app = FastAPI()
        app.include_router(router)
        app.state.workspace_dir = tmp_path
        resp = TestClient(app).get("/api/session-summary")
        assert resp.status_code == 200
        assert [e["title"] for e in resp.json()["entries"]] == ["Real work"]
