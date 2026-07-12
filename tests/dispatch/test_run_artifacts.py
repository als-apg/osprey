"""Dispatch runs expose the artifacts they produced — created-by isolation.

A dispatched agent writes plots into the shared artifact store. The worker must
say *which* artifacts a run produced and hand back their (renderable) bytes, so
a bridge can republish them — WITHOUT one run ever reading another run's output.

Association is **created-by**: ``ArtifactEntry.run_id`` is stamped from
``OSPREY_DISPATCH_RUN_ID`` at write time inside the trusted store, and every
artifact surface (the run-status ``artifacts`` list, the list route, and the
byte route) bottoms out in the single strict predicate ``entry.run_id ==
run_id``. Authorization never reads agent-controllable data, so a prompt-injected
run cannot widen its artifact set or reach another run's bytes. This is
deliberately NOT ``OSPREY_SESSION_ID`` (which relocates the whole store root).
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.artifacts import resolve as resolve_mod
from osprey.stores.artifact_store import ArtifactStore

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"
PDF_BYTES = b"%PDF-1.4\nfake-pdf-body\n%%EOF"
HTML_BYTES = b"<html><body><h1>plot</h1></body></html>"
TOKEN = "test-worker-token"


def _save(
    store: ArtifactStore,
    run_id: str,
    *,
    title: str,
    filename: str,
    mime: str,
    body: bytes,
) -> str:
    """Save one artifact tagged with ``run_id`` (via the env the store reads)."""
    prev = os.environ.get("OSPREY_DISPATCH_RUN_ID")
    if run_id:
        os.environ["OSPREY_DISPATCH_RUN_ID"] = run_id
    else:
        os.environ.pop("OSPREY_DISPATCH_RUN_ID", None)
    try:
        entry = store.save_file(
            file_content=body,
            filename=filename,
            title=title,
            artifact_type="image",
            mime_type=mime,
        )
    finally:
        if prev is None:
            os.environ.pop("OSPREY_DISPATCH_RUN_ID", None)
        else:
            os.environ["OSPREY_DISPATCH_RUN_ID"] = prev
    return entry.id


def _rooted_store(tmp_path: Path, monkeypatch) -> ArtifactStore:
    """Store rooted so ``resolve._get_store()`` reads the same place.

    ``_get_store()`` roots at ``$OSPREY_PROJECT_DIR/_agent_data``; the byte/list/
    describe surfaces all go through it, so tests must write there too.
    """
    project = tmp_path / "project"
    (project / "_agent_data").mkdir(parents=True)
    monkeypatch.setenv("OSPREY_PROJECT_DIR", str(project))
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    return ArtifactStore(workspace_root=project / "_agent_data")


# ---------------------------------------------------------------------------
# Store-level provenance (Slot 1)
# ---------------------------------------------------------------------------


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


class TestRunFilter:
    """``list_entries(run_filter=...)`` is the strict created-by boundary."""

    @pytest.fixture
    def store(self, tmp_path) -> ArtifactStore:
        s = ArtifactStore(workspace_root=tmp_path)
        _save(s, "run-1", title="mine", filename="a.png", mime="image/png", body=PNG_BYTES)
        _save(s, "run-1", title="mine2", filename="b.png", mime="image/png", body=PNG_BYTES)
        _save(s, "run-2", title="theirs", filename="c.png", mime="image/png", body=PNG_BYTES)
        _save(s, "", title="legacy", filename="d.png", mime="image/png", body=PNG_BYTES)
        return s

    def test_returns_only_this_runs_entries(self, store):
        got = {e.title for e in store.list_entries(run_filter="run-1")}
        assert got == {"mine", "mine2"}

    def test_excludes_other_runs(self, store):
        titles = {e.title for e in store.list_entries(run_filter="run-1")}
        assert "theirs" not in titles

    def test_excludes_untagged_legacy(self, store):
        titles = {e.title for e in store.list_entries(run_filter="run-1")}
        assert "legacy" not in titles

    def test_empty_run_filter_matches_nothing(self, store):
        # Symmetric foot-gun guard: an empty run tag must NOT match untagged
        # artifacts (unlike session_filter's OR-empty clause).
        assert store.list_entries(run_filter="") == []

    def test_none_run_filter_returns_all(self, store):
        assert len(store.list_entries(run_filter=None)) == 4


class TestGetRunEntry:
    """The single centralized cross-run gate."""

    @pytest.fixture
    def store(self, tmp_path):
        s = ArtifactStore(workspace_root=tmp_path)
        a1 = _save(s, "run-1", title="a", filename="a.png", mime="image/png", body=PNG_BYTES)
        a2 = _save(s, "run-2", title="b", filename="b.png", mime="image/png", body=PNG_BYTES)
        leg = _save(s, "", title="c", filename="c.png", mime="image/png", body=PNG_BYTES)
        return s, a1, a2, leg

    def test_exact_match_returns_entry(self, store):
        s, a1, _, _ = store
        assert s.get_run_entry("run-1", a1).id == a1

    def test_wrong_run_returns_none(self, store):
        s, _, a2, _ = store
        assert s.get_run_entry("run-1", a2) is None

    def test_untagged_entry_returns_none(self, store):
        s, _, _, leg = store
        assert s.get_run_entry("run-1", leg) is None

    def test_empty_run_id_returns_none(self, store):
        s, _, _, leg = store
        # Even against the untagged entry, an empty run_id must never match.
        assert s.get_run_entry("", leg) is None

    def test_unknown_id_returns_none(self, store):
        s, _, _, _ = store
        assert s.get_run_entry("run-1", "does-not-exist") is None


class TestStoreRooting:
    """resolve._get_store() must root where the agent's tools write."""

    def test_root_follows_project_dir_not_cwd(self, tmp_path, monkeypatch):
        project = tmp_path / "project"
        (project / "_agent_data" / "artifacts").mkdir(parents=True)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.setenv("OSPREY_PROJECT_DIR", str(project))
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.chdir(elsewhere)
        assert resolve_mod._get_store()._store_dir == project / "_agent_data" / "artifacts"


# ---------------------------------------------------------------------------
# Render-free descriptors (Slot 2)
# ---------------------------------------------------------------------------


class TestPredictDelivery:
    def test_image_passthrough(self):
        assert resolve_mod.predict_delivery("image/png") == ("image/png", True)

    def test_pdf_passthrough(self):
        assert resolve_mod.predict_delivery("application/pdf") == ("application/pdf", True)

    def test_html_renders_to_png(self):
        assert resolve_mod.predict_delivery("text/html") == ("image/png", True)

    def test_unknown_renders_to_png(self):
        assert resolve_mod.predict_delivery("application/whatever") == ("image/png", True)


class TestDescribeRunArtifacts:
    def test_unknown_run_is_empty(self, tmp_path, monkeypatch):
        _rooted_store(tmp_path, monkeypatch)
        assert resolve_mod.describe_run_artifacts("nope") == []

    def test_empty_run_id_is_empty(self, tmp_path, monkeypatch):
        _rooted_store(tmp_path, monkeypatch)
        assert resolve_mod.describe_run_artifacts("") == []

    def test_descriptor_shape_and_source_mime(self, tmp_path, monkeypatch):
        store = _rooted_store(tmp_path, monkeypatch)
        _save(store, "run-1", title="h", filename="h.html", mime="text/html", body=HTML_BYTES)
        [d] = resolve_mod.describe_run_artifacts("run-1")
        assert set(d) == {"artifact_id", "filename", "source_mime", "delivered_mime", "convertible"}
        assert d["source_mime"] == "text/html"
        assert d["delivered_mime"] == "image/png"  # rendered
        assert d["filename"].endswith(".png")  # predicted delivered name

    def test_render_free_never_invokes_converter(self, tmp_path, monkeypatch):
        store = _rooted_store(tmp_path, monkeypatch)
        _save(store, "run-1", title="h", filename="h.html", mime="text/html", body=HTML_BYTES)

        async def _boom(*a, **k):  # pragma: no cover - must never run
            raise AssertionError("describe must not render")

        import osprey.mcp_server.ariel.converters as conv

        monkeypatch.setitem(conv.CONVERTER_REGISTRY, "text/html", _boom)
        # describe only classifies (get_converter lookup), never calls the converter
        [d] = resolve_mod.describe_run_artifacts("run-1")
        assert d["delivered_mime"] == "image/png"


class TestResolveRunArtifacts:
    """The public bulk resolver: created-by scoped, lenient about missing files."""

    async def test_resolves_only_this_runs_artifacts(self, tmp_path, monkeypatch):
        store = _rooted_store(tmp_path, monkeypatch)
        a1 = _save(store, "run-1", title="a", filename="a.png", mime="image/png", body=PNG_BYTES)
        _save(store, "run-1", title="b", filename="b.png", mime="image/png", body=PNG_BYTES)
        _save(store, "run-2", title="c", filename="c.png", mime="image/png", body=PNG_BYTES)
        refs = await resolve_mod.resolve_run_artifacts("run-1")
        assert {r.artifact_id for r in refs} == {a1, refs[1].artifact_id}
        assert len(refs) == 2  # run-2's artifact excluded

    async def test_unknown_run_is_empty(self, tmp_path, monkeypatch):
        _rooted_store(tmp_path, monkeypatch)
        assert await resolve_mod.resolve_run_artifacts("nope") == []

    async def test_skips_artifact_whose_file_was_deleted(self, tmp_path, monkeypatch):
        store = _rooted_store(tmp_path, monkeypatch)
        keep = _save(store, "run-1", title="k", filename="k.png", mime="image/png", body=PNG_BYTES)
        gone = _save(store, "run-1", title="g", filename="g.png", mime="image/png", body=PNG_BYTES)
        (store.get_file_path(gone)).unlink()  # tag survives, file removed
        refs = await resolve_mod.resolve_run_artifacts("run-1")
        assert [r.artifact_id for r in refs] == [keep]


# ---------------------------------------------------------------------------
# HTTP surfaces (Slot 3): client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch) -> Iterator[tuple[TestClient, ArtifactStore, dict]]:
    monkeypatch.setenv("DISPATCH_WORKER_TOKEN", TOKEN)
    store = _rooted_store(tmp_path, monkeypatch)
    ids = {
        "a1": _save(
            store, "run-1", title="mine", filename="a.png", mime="image/png", body=PNG_BYTES
        ),
        "a2": _save(
            store, "run-1", title="mine2", filename="b.png", mime="image/png", body=PNG_BYTES
        ),
        "other": _save(
            store, "run-2", title="theirs", filename="c.png", mime="image/png", body=PNG_BYTES
        ),
        "legacy": _save(
            store, "", title="legacy", filename="d.png", mime="image/png", body=PNG_BYTES
        ),
    }
    from osprey.mcp_server.dispatch_worker import dispatch_api

    monkeypatch.setattr(
        dispatch_api, "_runs", {"run-1": {"status": "completed"}, "run-2": {"status": "completed"}}
    )
    monkeypatch.setattr(dispatch_api, "_queues", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})
    with TestClient(dispatch_api.app) as c:
        yield c, store, ids


def _auth():
    return {"Authorization": f"Bearer {TOKEN}"}


class TestByteRouteIsolation:
    """The byte route: identical, non-distinguishing 404 for every deny reason."""

    def test_serves_own_png_bytes(self, client):
        c, _, ids = client
        r = c.get(f"/dispatch/run-1/artifacts/{ids['a1']}", headers=_auth())
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/png")
        assert r.content == PNG_BYTES

    def test_cross_run_is_404(self, client):
        c, _, ids = client
        r = c.get(f"/dispatch/run-1/artifacts/{ids['other']}", headers=_auth())
        assert r.status_code == 404

    def test_untagged_legacy_is_404(self, client):
        c, _, ids = client
        r = c.get(f"/dispatch/run-1/artifacts/{ids['legacy']}", headers=_auth())
        assert r.status_code == 404

    def test_unknown_id_is_404(self, client):
        c, _, _ = client
        r = c.get("/dispatch/run-1/artifacts/does-not-exist", headers=_auth())
        assert r.status_code == 404

    def test_unknown_run_is_404(self, client):
        c, _, ids = client
        r = c.get(f"/dispatch/no-such-run/artifacts/{ids['a1']}", headers=_auth())
        assert r.status_code == 404

    def test_traversal_shaped_id_is_404(self, client):
        c, _, _ = client
        r = c.get("/dispatch/run-1/artifacts/..%2f..%2fetc%2fpasswd", headers=_auth())
        assert r.status_code == 404

    def test_deny_reasons_are_indistinguishable(self, client):
        """Cross-run, unknown-id, and unknown-run all return the SAME 404 body."""
        c, _, ids = client
        bodies = {
            c.get(f"/dispatch/run-1/artifacts/{ids['other']}", headers=_auth()).text,
            c.get("/dispatch/run-1/artifacts/nope", headers=_auth()).text,
            c.get(f"/dispatch/no-run/artifacts/{ids['a1']}", headers=_auth()).text,
        }
        assert len(bodies) == 1  # no existence oracle


class TestInjectionResistance:
    """HEADLINE: a run referencing another run's artifact-id cannot fetch it.

    Under the old referenced-by model, an artifact-id appearing in a run's
    tool_calls[].result would authorize the fetch. Created-by ignores the
    reference entirely — only the write-time tag counts.
    """

    def test_referenced_but_not_created_is_404(self, client, tmp_path):
        c, _, ids = client
        # Plant a persisted run record for run-1 whose results "reference" run-2's
        # artifact (as a prompt-injected agent might emit).
        log_dir = tmp_path / "project" / "_agent_data" / "dispatch"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "run-1.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "tool_calls": [{"result": json.dumps({"artifact_id": ids["other"]})}],
                }
            )
        )
        r = c.get(f"/dispatch/run-1/artifacts/{ids['other']}", headers=_auth())
        assert r.status_code == 404  # created-by wins; the reference is irrelevant


class TestListRoute:
    def test_lists_only_own_descriptors(self, client):
        c, _, ids = client
        r = c.get("/dispatch/run-1/artifacts", headers=_auth())
        assert r.status_code == 200
        got = {d["artifact_id"] for d in r.json()}
        assert got == {ids["a1"], ids["a2"]}

    def test_unknown_run_is_empty_not_404(self, client):
        c, _, _ = client
        r = c.get("/dispatch/no-such-run/artifacts", headers=_auth())
        assert r.status_code == 200
        assert r.json() == []

    def test_metadata_only_no_render(self, client, monkeypatch):
        """Listing must not invoke a converter even for renderable artifacts."""
        c, store, _ = client
        _save(store, "run-1", title="h", filename="h.html", mime="text/html", body=HTML_BYTES)
        import osprey.mcp_server.ariel.converters as conv

        async def _boom(*a, **k):  # pragma: no cover
            raise AssertionError("list must not render")

        monkeypatch.setitem(conv.CONVERTER_REGISTRY, "text/html", _boom)
        r = c.get("/dispatch/run-1/artifacts", headers=_auth())
        assert r.status_code == 200
        html_desc = [d for d in r.json() if d["source_mime"] == "text/html"]
        assert html_desc and html_desc[0]["delivered_mime"] == "image/png"

    def test_requires_auth(self, client):
        c, _, _ = client
        assert c.get("/dispatch/run-1/artifacts").status_code in (401, 403)


class TestCoherence:
    """KEYSTONE: status body, list route, and byte route agree on one set."""

    def test_all_surfaces_agree(self, client):
        c, _, ids = client
        # status body carries descriptors for run-1's artifacts
        status_ids = {d["artifact_id"] for d in _list_via_status(c, "run-1")}
        list_ids = {
            d["artifact_id"] for d in c.get("/dispatch/run-1/artifacts", headers=_auth()).json()
        }
        served = {
            aid
            for aid in (ids["a1"], ids["a2"], ids["other"])
            if c.get(f"/dispatch/run-1/artifacts/{aid}", headers=_auth()).status_code == 200
        }
        assert status_ids == list_ids == served == {ids["a1"], ids["a2"]}
        assert ids["other"] not in status_ids  # run-2's artifact appears nowhere under run-1


def _list_via_status(c, run_id):
    """Read the artifact descriptors from the run-status body."""
    from osprey.mcp_server.dispatch_worker import dispatch_api

    # Seed the status body the way completion does, then read it back over HTTP.
    dispatch_api._runs[run_id] = {
        "status": "completed",
        "artifacts": resolve_mod.describe_run_artifacts(run_id),
    }
    return c.get(f"/dispatch/{run_id}", headers=_auth()).json()["artifacts"]


class TestStatusBodyDiskFallback:
    def test_evicted_run_still_served_from_disk(self, client, tmp_path):
        c, _, _ = client
        from osprey.mcp_server.dispatch_worker import dispatch_api

        dispatch_api._runs.pop("run-1", None)  # evicted from memory
        log_dir = tmp_path / "project" / "_agent_data" / "dispatch"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "run-1.json").write_text(json.dumps({"status": "completed", "artifacts": []}))
        r = c.get("/dispatch/run-1", headers=_auth())
        assert r.status_code == 200
        assert r.json()["status"] == "completed"

    def test_truly_unknown_run_is_404(self, client):
        c, _, _ = client
        assert c.get("/dispatch/ghost", headers=_auth()).status_code == 404


class TestConversionFidelity:
    def test_pdf_passthrough(self, client):
        c, store, _ = client
        pid = _save(
            store, "run-1", title="doc", filename="d.pdf", mime="application/pdf", body=PDF_BYTES
        )
        r = c.get(f"/dispatch/run-1/artifacts/{pid}", headers=_auth())
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/pdf")
        assert r.content == PDF_BYTES

    def test_html_renders_to_png(self, client, monkeypatch):
        c, store, _ = client
        hid = _save(store, "run-1", title="h", filename="h.html", mime="text/html", body=HTML_BYTES)
        import osprey.mcp_server.ariel.converters as conv

        async def _fake_png(source: Path, output_dir: Path) -> Path:
            out = output_dir / f"{source.stem}.png"
            out.write_bytes(PNG_BYTES)
            return out

        monkeypatch.setitem(conv.CONVERTER_REGISTRY, "text/html", _fake_png)
        r = c.get(f"/dispatch/run-1/artifacts/{hid}", headers=_auth())
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/png")
        assert r.content == PNG_BYTES

    def test_converter_failure_falls_back_to_original(self, client, monkeypatch):
        c, store, _ = client
        hid = _save(store, "run-1", title="h", filename="h.html", mime="text/html", body=HTML_BYTES)
        import osprey.mcp_server.ariel.converters as conv

        async def _broken(*a, **k):
            raise RuntimeError("no playwright")

        monkeypatch.setitem(conv.CONVERTER_REGISTRY, "text/html", _broken)
        r = c.get(f"/dispatch/run-1/artifacts/{hid}", headers=_auth())
        assert r.status_code == 200  # graceful: original bytes, not a 500
        assert r.content == HTML_BYTES
        assert r.headers["content-type"].startswith("text/html")


class TestAuthMatrix:
    def test_all_artifact_routes_require_token(self, client):
        c, _, ids = client
        for path in (
            "/dispatch/run-1",
            "/dispatch/run-1/artifacts",
            f"/dispatch/run-1/artifacts/{ids['a1']}",
        ):
            assert c.get(path).status_code in (401, 403), path

    def test_unset_worker_token_fails_closed(self, client, monkeypatch):
        c, _, ids = client
        monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)
        r = c.get(f"/dispatch/run-1/artifacts/{ids['a1']}", headers={"Authorization": "Bearer x"})
        assert r.status_code == 500  # never serve when the gate is misconfigured

    def test_health_is_open(self, client):
        c, _, _ = client
        assert c.get("/health").status_code == 200


class TestDeniedToolsInvariant:
    """The created-by tag is only spoof-proof while the agent has no shell.

    Binds that load-bearing assumption to a test: if a future change re-enabled
    a shell/exec tool for dispatch, an agent could forge OSPREY_DISPATCH_RUN_ID
    and mis-tag its artifacts. Keep these denied.
    """

    def test_shell_tools_denied(self):
        from osprey.mcp_server.dispatch_worker.dispatch_api import DENIED_TOOLS

        assert {"Bash", "BashOutput", "KillShell"} <= set(DENIED_TOOLS)


# ---------------------------------------------------------------------------
# Terminal-state contract: every finished result carries artifacts (Slot 3)
# ---------------------------------------------------------------------------


async def _drive_failing_task(monkeypatch, tmp_path, *, kind: str) -> dict:
    monkeypatch.setenv("OSPREY_PROJECT_DIR", str(tmp_path))
    from osprey.mcp_server.dispatch_worker import dispatch_api

    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_queues", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})
    monkeypatch.setattr(dispatch_api, "_persist_run", lambda rid, r: None)

    if kind == "exception":

        async def _run(**kw):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(dispatch_api.sdk_runner, "run_dispatch", _run)
        await dispatch_api._run_dispatch_task(
            "run-x", dispatch_api.DispatchRequest(prompt="p", allowed_tools=[])
        )
    elif kind == "timeout":
        monkeypatch.setattr(dispatch_api, "DISPATCH_TIMEOUT_SEC", 0)

        async def _slow(**kw):
            await asyncio.sleep(1)

        monkeypatch.setattr(dispatch_api.sdk_runner, "run_dispatch", _slow)
        await dispatch_api._run_dispatch_task(
            "run-x", dispatch_api.DispatchRequest(prompt="p", allowed_tools=[])
        )
    elif kind == "cancel":

        async def _hang(**kw):
            await asyncio.sleep(10)

        monkeypatch.setattr(dispatch_api.sdk_runner, "run_dispatch", _hang)
        task = asyncio.create_task(
            dispatch_api._run_dispatch_task(
                "run-x", dispatch_api.DispatchRequest(prompt="p", allowed_tools=[])
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    return dispatch_api._runs["run-x"]


class TestTerminalStateContract:
    async def test_exception_result_has_empty_artifacts(self, monkeypatch, tmp_path):
        result = await _drive_failing_task(monkeypatch, tmp_path, kind="exception")
        assert result["status"] == "error"
        assert result["artifacts"] == []

    async def test_timeout_result_has_empty_artifacts(self, monkeypatch, tmp_path):
        result = await _drive_failing_task(monkeypatch, tmp_path, kind="timeout")
        assert result["status"] == "error"
        assert result["artifacts"] == []

    async def test_cancel_result_has_empty_artifacts(self, monkeypatch, tmp_path):
        result = await _drive_failing_task(monkeypatch, tmp_path, kind="cancel")
        assert result["status"] == "error"
        assert result["artifacts"] == []
