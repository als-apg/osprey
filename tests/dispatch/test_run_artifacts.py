"""Dispatch runs expose the artifacts they produced.

A dispatched agent writes plot PNGs into the shared artifact store. The worker
must be able to say *which* artifacts a given run produced, and hand back their
bytes, so an external consumer (e.g. a chat bridge) can republish them.

Attribution rides on ``ArtifactEntry.run_id``, stamped from
``OSPREY_DISPATCH_RUN_ID``. It is deliberately NOT ``OSPREY_SESSION_ID``: that
variable also relocates the store into ``_agent_data/sessions/<id>/`` via
``resolve_agent_data_root``, which would move dispatch plots off the shared
root the gallery reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.stores.artifact_store import ArtifactStore

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"
TOKEN = "test-worker-token"


def _store_with(tmp_path: Path, entries: list[tuple[str, str]]) -> ArtifactStore:
    """Build a store under tmp_path containing one PNG per (run_id, title)."""
    store = ArtifactStore(workspace_root=tmp_path)
    for run_id, title in entries:
        import os

        prev = os.environ.get("OSPREY_DISPATCH_RUN_ID")
        if run_id:
            os.environ["OSPREY_DISPATCH_RUN_ID"] = run_id
        else:
            os.environ.pop("OSPREY_DISPATCH_RUN_ID", None)
        try:
            store.save_file(
                file_content=PNG_BYTES,
                filename=f"{title}.png",
                title=title,
                artifact_type="image",
                mime_type="image/png",
            )
        finally:
            if prev is None:
                os.environ.pop("OSPREY_DISPATCH_RUN_ID", None)
            else:
                os.environ["OSPREY_DISPATCH_RUN_ID"] = prev
    return store


class TestArtifactEntryRunId:
    def test_save_file_stamps_run_id_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OSPREY_DISPATCH_RUN_ID", "run-abc")
        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_file(
            file_content=PNG_BYTES,
            filename="p.png",
            title="p",
            artifact_type="image",
            mime_type="image/png",
        )
        assert entry.run_id == "run-abc"

    def test_run_id_defaults_empty_when_env_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OSPREY_DISPATCH_RUN_ID", raising=False)
        store = ArtifactStore(workspace_root=tmp_path)
        entry = store.save_file(
            file_content=PNG_BYTES,
            filename="p.png",
            title="p",
            artifact_type="image",
            mime_type="image/png",
        )
        assert entry.run_id == ""

    def test_old_index_without_run_id_still_loads(self, tmp_path):
        """A pre-existing artifacts.json has no run_id key; loading must not raise."""
        import json

        d = tmp_path / "artifacts"
        d.mkdir(parents=True)
        (d / "artifacts.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "updated": "2026-01-01T00:00:00Z",
                    "entry_count": 1,
                    "entries": [
                        {
                            "id": "old1",
                            "artifact_type": "image",
                            "title": "t",
                            "description": "",
                            "filename": "t.png",
                            "mime_type": "image/png",
                            "size_bytes": 3,
                            "timestamp": "2026-01-01T00:00:00Z",
                            "tool_source": "x",
                        }
                    ],
                }
            )
        )
        store = ArtifactStore(workspace_root=tmp_path)
        entries = store.list_entries()
        assert len(entries) == 1
        assert entries[0].run_id == ""


class TestArtifactStoreRooting:
    """The store must be rooted where the agent's tools actually write.

    Regression: rooting via ``resolve_shared_data_root()`` located the project
    through ``OSPREY_CONFIG`` and fell back to CWD. The worker is configured
    with ``CONFIG_FILE`` and runs from the image WORKDIR, so it resolved to
    ``<cwd>/_agent_data`` while the agent wrote to
    ``$OSPREY_PROJECT_DIR/_agent_data`` — every lookup returned nothing.
    """

    def test_root_follows_project_dir_not_cwd(self, tmp_path, monkeypatch):
        from osprey.mcp_server.dispatch_worker import dispatch_api

        project = tmp_path / "project"
        (project / "_agent_data" / "artifacts").mkdir(parents=True)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()

        monkeypatch.setenv("OSPREY_PROJECT_DIR", str(project))
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.chdir(elsewhere)

        store = dispatch_api._artifact_store()
        assert store._store_dir == project / "_agent_data" / "artifacts"

    def test_finds_artifacts_the_agent_wrote(self, tmp_path, monkeypatch):
        """End-to-end of the rooting: write via the agent's root, read via the worker."""
        from osprey.mcp_server.dispatch_worker import dispatch_api

        project = tmp_path / "project"
        (project / "_agent_data").mkdir(parents=True)
        monkeypatch.setenv("OSPREY_PROJECT_DIR", str(project))
        monkeypatch.chdir(tmp_path)

        monkeypatch.setenv("OSPREY_DISPATCH_RUN_ID", "run-x")
        ArtifactStore(workspace_root=project / "_agent_data").save_file(
            file_content=PNG_BYTES,
            filename="p.png",
            title="p",
            artifact_type="image",
            mime_type="image/png",
        )
        assert dispatch_api._artifact_ids_for_run("run-x")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPATCH_WORKER_TOKEN", TOKEN)
    from osprey.mcp_server.dispatch_worker import dispatch_api

    store = _store_with(
        tmp_path,
        [("run-1", "mine"), ("run-2", "theirs"), ("", "untagged-legacy")],
    )
    monkeypatch.setattr(dispatch_api, "_artifact_store", lambda: store)
    dispatch_api._runs.clear()
    dispatch_api._runs["run-1"] = {"status": "completed"}
    dispatch_api._runs["run-2"] = {"status": "completed"}
    with TestClient(dispatch_api.app) as c:
        yield c, store


def _ids(store, run_id):
    return [e.id for e in store.list_entries() if e.run_id == run_id]


class TestArtifactIdsForRun:
    def test_only_this_runs_artifacts(self, client):
        _, store = client
        from osprey.mcp_server.dispatch_worker import dispatch_api

        ids = dispatch_api._artifact_ids_for_run("run-1")
        assert ids == _ids(store, "run-1")
        assert len(ids) == 1

    def test_excludes_untagged_legacy_artifacts(self, client):
        _, store = client
        from osprey.mcp_server.dispatch_worker import dispatch_api

        ids = dispatch_api._artifact_ids_for_run("run-1")
        untagged = [e.id for e in store.list_entries() if not e.run_id]
        assert untagged, "fixture should contain an untagged artifact"
        assert not set(ids) & set(untagged)

    def test_unknown_run_yields_nothing(self):
        from osprey.mcp_server.dispatch_worker import dispatch_api

        assert dispatch_api._artifact_ids_for_run("nope") == []


class TestArtifactBytesRoute:
    def test_serves_png_bytes(self, client):
        c, store = client
        art_id = _ids(store, "run-1")[0]
        r = c.get(
            f"/dispatch/run-1/artifacts/{art_id}",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/png")
        assert r.content == PNG_BYTES

    def test_requires_auth(self, client):
        c, store = client
        art_id = _ids(store, "run-1")[0]
        r = c.get(f"/dispatch/run-1/artifacts/{art_id}")
        assert r.status_code in (401, 403)

    def test_cross_run_artifact_is_not_served(self, client):
        """An artifact belonging to run-2 must not be reachable under run-1."""
        c, store = client
        other = _ids(store, "run-2")[0]
        r = c.get(
            f"/dispatch/run-1/artifacts/{other}",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert r.status_code == 404

    def test_unknown_artifact_404(self, client):
        c, _ = client
        r = c.get(
            "/dispatch/run-1/artifacts/does-not-exist",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert r.status_code == 404

    def test_unknown_run_404(self, client):
        c, store = client
        art_id = _ids(store, "run-1")[0]
        r = c.get(
            f"/dispatch/no-such-run/artifacts/{art_id}",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert r.status_code == 404
