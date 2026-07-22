"""Worker-side ingestion of dispatch input files.

Covers the whole ingest path: ``ingest=True`` files are decoded into the
artifact store tagged with the run's ``run_id`` and ``origin="input"``;
``ingest=False`` files are never stored and pass through the seam; the run's
produced-artifact listing excludes inputs while ``input_artifacts`` surfaces
them; the created-by byte route still serves an input entry; and the run-status
body carries ``input_artifacts`` after a completed run.

Hermetic: a tmp-rooted store (so the resolver and worker share one root),
direct calls, and a FastAPI ``TestClient`` — no network, no real SDK.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.artifacts import resolve as resolve_mod
from osprey.mcp_server.dispatch_worker import dispatch_api
from osprey.mcp_server.dispatch_worker.dispatch_api import DispatchRequest, InputFile
from osprey.stores.artifact_store import ArtifactStore

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"
CSV_BYTES = b"a,b,c\n1,2,3\n"
TOKEN = "test-worker-token"


def _b64(body: bytes) -> str:
    return base64.b64encode(body).decode("ascii")


def _root(tmp_path: Path, monkeypatch) -> Path:
    """Point OSPREY_PROJECT_DIR at a tmp project so get_run_store() roots there."""
    project = tmp_path / "project"
    (project / "_agent_data" / "artifacts").mkdir(parents=True)
    monkeypatch.setenv("OSPREY_PROJECT_DIR", str(project))
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    return project


def _infile(name: str, mime: str, body: bytes, ingest: bool = True) -> InputFile:
    return InputFile(filename=name, mime=mime, content_b64=_b64(body), ingest=ingest)


# ---------------------------------------------------------------------------
# ingest_input_files — store writes + seam
# ---------------------------------------------------------------------------


def test_input_files_ingest_true_stored_with_origin_and_run_id(tmp_path, monkeypatch):
    _root(tmp_path, monkeypatch)
    seam = dispatch_api.ingest_input_files("run-1", [_infile("plot.png", "image/png", PNG_BYTES)])
    store = resolve_mod.get_run_store()
    [entry] = store.list_entries(run_filter="run-1")
    assert entry.origin == "input"
    assert entry.run_id == "run-1"
    assert entry.mime_type == "image/png"
    # bytes round-trip to disk
    assert store.get_file_path(entry.id).read_bytes() == PNG_BYTES
    # seam item for a stored file: entry_id set, no passthrough bytes
    assert seam == [
        {"filename": "plot.png", "mime": "image/png", "entry_id": entry.id, "content_b64": None}
    ]


def test_input_files_ingest_false_never_stored(tmp_path, monkeypatch):
    _root(tmp_path, monkeypatch)
    b64 = _b64(PNG_BYTES)
    seam = dispatch_api.ingest_input_files(
        "run-1", [InputFile(filename="keep.png", mime="image/png", content_b64=b64, ingest=False)]
    )
    store = resolve_mod.get_run_store()
    assert store.list_entries(run_filter="run-1") == []  # nothing written
    # seam carries the original bytes for pass-through, no entry_id
    assert seam == [
        {"filename": "keep.png", "mime": "image/png", "entry_id": None, "content_b64": b64}
    ]


def test_input_files_ingest_mixed_batch_preserves_order(tmp_path, monkeypatch):
    _root(tmp_path, monkeypatch)
    seam = dispatch_api.ingest_input_files(
        "run-1",
        [
            _infile("a.png", "image/png", PNG_BYTES),
            _infile("b.csv", "text/csv", CSV_BYTES, ingest=False),
            _infile("c.json", "application/json", b"{}"),
        ],
    )
    assert [s["filename"] for s in seam] == ["a.png", "b.csv", "c.json"]
    assert seam[0]["entry_id"] is not None and seam[0]["content_b64"] is None
    assert seam[1]["entry_id"] is None and seam[1]["content_b64"] == _b64(CSV_BYTES)
    assert seam[2]["entry_id"] is not None
    # only the two ingest=true files are stored
    assert len(resolve_mod.get_run_store().list_entries(run_filter="run-1")) == 2


def test_input_files_ingest_empty_batch_returns_empty(tmp_path, monkeypatch):
    _root(tmp_path, monkeypatch)
    assert dispatch_api.ingest_input_files("run-1", []) == []


# ---------------------------------------------------------------------------
# resolve descriptors: exclusion + input_artifacts shape
# ---------------------------------------------------------------------------


def test_input_files_ingest_excluded_from_produced_artifacts(tmp_path, monkeypatch):
    _root(tmp_path, monkeypatch)
    dispatch_api.ingest_input_files("run-1", [_infile("in.png", "image/png", PNG_BYTES)])
    # An input artifact is NOT a produced artifact.
    assert resolve_mod.describe_run_artifacts("run-1") == []


def test_input_files_input_artifacts_descriptor_shape(tmp_path, monkeypatch):
    _root(tmp_path, monkeypatch)
    seam = dispatch_api.ingest_input_files("run-1", [_infile("in.png", "image/png", PNG_BYTES)])
    entry_id = seam[0]["entry_id"]
    [d] = resolve_mod.describe_run_input_artifacts("run-1")
    assert d == {"entry_id": entry_id, "filename": "in.png", "mime": "image/png"}


def test_input_files_input_artifacts_filename_is_caller_name(tmp_path, monkeypatch):
    # The descriptor filename is the caller's name, not the id-prefixed store name.
    _root(tmp_path, monkeypatch)
    dispatch_api.ingest_input_files("run-1", [_infile("report.csv", "text/csv", CSV_BYTES)])
    [d] = resolve_mod.describe_run_input_artifacts("run-1")
    assert d["filename"] == "report.csv"


def test_input_files_input_artifacts_only_input_origin(tmp_path, monkeypatch):
    # A produced artifact (origin empty) must not appear among input_artifacts.
    project = _root(tmp_path, monkeypatch)
    store = ArtifactStore(workspace_root=project / "_agent_data")
    store.save_file(
        file_content=PNG_BYTES,
        filename="made.png",
        title="made",
        artifact_type="image",
        mime_type="image/png",
        run_id="run-1",
    )
    dispatch_api.ingest_input_files("run-1", [_infile("in.png", "image/png", PNG_BYTES)])
    input_names = {d["filename"] for d in resolve_mod.describe_run_input_artifacts("run-1")}
    produced = {d["filename"] for d in resolve_mod.describe_run_artifacts("run-1")}
    assert input_names == {"in.png"}
    assert "made.png" not in input_names
    assert "in.png" not in produced


def test_input_files_input_artifacts_unknown_run_empty(tmp_path, monkeypatch):
    _root(tmp_path, monkeypatch)
    assert resolve_mod.describe_run_input_artifacts("nope") == []
    assert resolve_mod.describe_run_input_artifacts("") == []


# ---------------------------------------------------------------------------
# take_input_seam pop-once
# ---------------------------------------------------------------------------


def test_input_files_ingest_take_seam_pop_once(monkeypatch):
    monkeypatch.setattr(dispatch_api, "_run_input_seam", {"run-1": [{"filename": "x"}]})
    assert dispatch_api.take_input_seam("run-1") == [{"filename": "x"}]
    assert dispatch_api.take_input_seam("run-1") == []  # popped
    assert dispatch_api.take_input_seam("absent") == []


# ---------------------------------------------------------------------------
# HTTP: byte route serves input entries; list route excludes them
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch) -> Iterator[tuple[TestClient, str]]:
    monkeypatch.setenv("DISPATCH_WORKER_TOKEN", TOKEN)
    _root(tmp_path, monkeypatch)
    seam = dispatch_api.ingest_input_files("run-1", [_infile("in.png", "image/png", PNG_BYTES)])
    entry_id = seam[0]["entry_id"]
    monkeypatch.setattr(dispatch_api, "_runs", {"run-1": {"status": "completed"}})
    monkeypatch.setattr(dispatch_api, "_queues", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})
    with TestClient(dispatch_api.app) as c:
        yield c, entry_id


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def test_input_files_ingest_byte_route_serves_input_entry(client):
    c, entry_id = client
    r = c.get(f"/dispatch/run-1/artifacts/{entry_id}", headers=_auth())
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/png")
    assert r.content == PNG_BYTES


def test_input_files_ingest_byte_route_cross_run_still_404(client):
    c, entry_id = client
    r = c.get(f"/dispatch/other-run/artifacts/{entry_id}", headers=_auth())
    assert r.status_code == 404


def test_input_files_ingest_excluded_from_list_route(client):
    c, _ = client
    r = c.get("/dispatch/run-1/artifacts", headers=_auth())
    assert r.status_code == 200
    assert r.json() == []  # input artifact is not a produced artifact


# ---------------------------------------------------------------------------
# Status body carries input_artifacts after a completed run
# ---------------------------------------------------------------------------


async def test_input_files_input_artifacts_on_status_body(tmp_path, monkeypatch):
    _root(tmp_path, monkeypatch)
    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_queues", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})
    monkeypatch.setattr(dispatch_api, "_run_input_seam", {})
    monkeypatch.setattr(dispatch_api, "_persist_run", lambda rid, r: None)

    async def _fake_run(**kw):
        return {
            "status": "completed",
            "text_output": "ok",
            "tool_calls": [],
            "error": None,
            "duration_sec": 0.01,
            "cost_usd": 0.0,
            "num_turns": 1,
        }

    monkeypatch.setattr(dispatch_api.sdk_runner, "run_dispatch", _fake_run)

    req = DispatchRequest(
        prompt="describe this",
        allowed_tools=[],
        input_files=[_infile("in.png", "image/png", PNG_BYTES)],
    )
    await dispatch_api._run_dispatch_task("run-x", req)

    result = dispatch_api._runs["run-x"]
    assert result["status"] == "completed"
    assert result["artifacts"] == []  # input is excluded from produced
    assert [d["filename"] for d in result["input_artifacts"]] == ["in.png"]
    assert result["input_artifacts"][0]["mime"] == "image/png"
    assert "entry_id" in result["input_artifacts"][0]
    # seam cleaned up by the task's finally
    assert dispatch_api._run_input_seam.get("run-x") is None
