"""Tests for the ARIEL drafts list and delete routes and the draft-id contract."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import osprey.interfaces.ariel.api.drafts as drafts_mod
from osprey.interfaces.ariel.api.drafts import DRAFT_TTL_SECONDS, draft_router

ID_A = "draft-0123456789ab"
ID_B = "draft-abcdef012345"
ID_C = "draft-fedcba987654"


@pytest.fixture
def drafts_dir(tmp_path, monkeypatch):
    path = tmp_path / "drafts"
    path.mkdir()
    monkeypatch.setattr(drafts_mod, "_drafts_dir", lambda: path)
    return path


@pytest.fixture
def client(drafts_dir):  # noqa: ARG001 - the store must be redirected first
    app = FastAPI()
    app.include_router(draft_router)
    return TestClient(app)


def _write(drafts_dir, draft_id: str, *, age: float = 0.0, **fields) -> None:
    data = {"draft_id": draft_id, "subject": "S", "details": "D", **fields}
    path = drafts_dir / f"{draft_id}.json"
    path.write_text(json.dumps(data))
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))


def _ids(client) -> list[str]:
    resp = client.get("/api/drafts")
    assert resp.status_code == 200
    return [d["draft_id"] for d in resp.json()["drafts"]]


# ---------------------------------------------------------------------------
# C-DRAFTS-LIST
# ---------------------------------------------------------------------------


def test_list_empty_store(client):
    assert client.get("/api/drafts").json() == {"drafts": []}


def test_list_without_store_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(drafts_mod, "_drafts_dir", lambda: tmp_path / "absent")
    app = FastAPI()
    app.include_router(draft_router)
    assert TestClient(app).get("/api/drafts").json() == {"drafts": []}


def test_list_row_shape(client, drafts_dir):
    _write(
        drafts_dir,
        ID_A,
        subject="Beam dump",
        author="Alice",
        logbook="Operations",
        attachment_paths=["/x/a.png", "/x/b.pdf"],
        metadata={"session_metadata": {"created_via": "ariel-mcp", "id": "s-1"}},
    )

    rows = client.get("/api/drafts").json()["drafts"]

    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {
        "draft_id",
        "subject",
        "author",
        "logbook",
        "created_at",
        "staged_by",
        "attachment_count",
    }
    assert row["draft_id"] == ID_A
    assert row["subject"] == "Beam dump"
    assert row["author"] == "Alice"
    assert row["logbook"] == "Operations"
    assert row["staged_by"] == "ariel-mcp"
    assert row["attachment_count"] == 2
    mtime = (drafts_dir / f"{ID_A}.json").stat().st_mtime
    assert datetime.fromisoformat(row["created_at"]).timestamp() == pytest.approx(mtime, abs=1)
    assert datetime.fromisoformat(row["created_at"]).tzinfo is not None


def test_list_row_without_provenance(client, drafts_dir):
    _write(drafts_dir, ID_A)

    row = client.get("/api/drafts").json()["drafts"][0]

    assert row["staged_by"] is None
    assert row["author"] is None
    assert row["logbook"] is None
    assert row["attachment_count"] == 0


@pytest.mark.parametrize(
    "metadata",
    [None, "text", {"session_metadata": None}, {"session_metadata": "s"}, {"other": 1}],
)
def test_list_staged_by_null_for_odd_metadata(client, drafts_dir, metadata):
    _write(drafts_dir, ID_A, metadata=metadata)
    assert client.get("/api/drafts").json()["drafts"][0]["staged_by"] is None


def test_list_newest_first(client, drafts_dir):
    _write(drafts_dir, ID_A, age=300)
    _write(drafts_dir, ID_B, age=10)
    _write(drafts_dir, ID_C, age=100)

    assert _ids(client) == [ID_B, ID_C, ID_A]


def test_list_excludes_expired_and_cleans_them_up(client, drafts_dir):
    _write(drafts_dir, ID_A, age=DRAFT_TTL_SECONDS + 60)
    _write(drafts_dir, ID_B)

    assert _ids(client) == [ID_B]
    assert not (drafts_dir / f"{ID_A}.json").exists()


def test_list_skips_sidecars(client, drafts_dir):
    _write(drafts_dir, ID_A)
    (drafts_dir / f"{ID_A}-metadata.json").write_text(json.dumps({"draft_id": ID_A}))

    assert _ids(client) == [ID_A]


@pytest.mark.parametrize(
    "content",
    ["{not json", '{"draft_id": "draft-0123456789ab", "sub', "[1, 2]", '"text"', "null", ""],
    ids=["corrupt", "truncated", "array", "string", "null", "empty"],
)
def test_list_skips_unparsable_or_non_object(client, drafts_dir, content):
    (drafts_dir / f"{ID_A}.json").write_text(content)
    _write(drafts_dir, ID_B)

    assert _ids(client) == [ID_B]


def test_list_skips_non_utf8_file(client, drafts_dir):
    (drafts_dir / f"{ID_A}.json").write_bytes(b"\xff\xfe\x00garbage")
    _write(drafts_dir, ID_B)

    assert _ids(client) == [ID_B]


def test_list_skips_mismatched_draft_id(client, drafts_dir):
    (drafts_dir / f"{ID_A}.json").write_text(json.dumps({"draft_id": ID_C, "subject": "S"}))
    (drafts_dir / f"{ID_B}.json").write_text(json.dumps({"subject": "no id"}))

    assert _ids(client) == []


def test_list_skips_stems_outside_the_id_contract(client, drafts_dir):
    # The uppercase stem must not collide with ID_A on a case-insensitive filesystem.
    for stem in ["draft-test001", "draft-ABCDEF012345", "notes", "draft-0123456789abc"]:
        (drafts_dir / f"{stem}.json").write_text(json.dumps({"draft_id": stem}))
    _write(drafts_dir, ID_A)

    assert _ids(client) == [ID_A]


# ---------------------------------------------------------------------------
# C-DRAFT-ID
# ---------------------------------------------------------------------------

BAD_IDS = [
    f"{ID_A}-metadata",  # sidecar
    "draft-test001",
    "draft-0123456789AB",
    "draft-0123456789abc",
    "draft-0123456789a",
    "%2E%2E",  # a bare ".." segment is normalised away by the client; encode it
    "draft-..",
    "..%5C..%5Cetc",
    "draft-0123456789ab%00",
    "draft-0123456789ab%0A",
    "draft-0123456789ab.json",
]


@pytest.fixture
def no_fs(client, monkeypatch):
    """Any drafts-store access fails the test."""

    def forbidden():
        raise AssertionError("drafts store touched for an invalid id")

    monkeypatch.setattr(drafts_mod, "_drafts_dir", forbidden)
    return client


@pytest.mark.parametrize("bad_id", BAD_IDS)
@pytest.mark.parametrize(
    ("method", "template"),
    [
        ("GET", "/api/drafts/{id}"),
        ("GET", "/api/drafts/{id}/attachments/file.png"),
        ("DELETE", "/api/drafts/{id}"),
    ],
    ids=["get", "attachment", "delete"],
)
def test_invalid_id_refused_before_filesystem(no_fs, method, template, bad_id):
    resp = no_fs.request(method, template.format(id=bad_id))
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "path_id", ["..", "../etc/passwd", "..%2F..%2Fetc", "draft-0123456789ab%2Fx"]
)
@pytest.mark.parametrize(
    ("method", "template"),
    [("GET", "/api/drafts/{id}"), ("DELETE", "/api/drafts/{id}")],
    ids=["get", "delete"],
)
def test_slash_ids_never_reach_the_route(no_fs, method, template, path_id):
    resp = no_fs.request(method, template.format(id=path_id))
    assert resp.status_code in (404, 405)


def test_sidecar_never_served_as_draft(client, drafts_dir):
    sidecar = drafts_dir / f"{ID_A}-metadata.json"
    sidecar.write_text(json.dumps({"draft_id": f"{ID_A}-metadata", "subject": "S", "details": "D"}))

    assert client.get(f"/api/drafts/{ID_A}-metadata").status_code == 422
    assert client.delete(f"/api/drafts/{ID_A}-metadata").status_code == 422
    assert sidecar.exists()


# ---------------------------------------------------------------------------
# C-DRAFT-DELETE
# ---------------------------------------------------------------------------


def test_delete_removes_draft_and_sidecar_only(client, drafts_dir, tmp_path):
    attachment = drafts_dir / "staging" / "plot.png"
    attachment.parent.mkdir()
    attachment.write_bytes(b"png")
    outside = tmp_path / "elsewhere.png"
    outside.write_bytes(b"png")
    _write(drafts_dir, ID_A, attachment_paths=[str(attachment), str(outside)])
    (drafts_dir / f"{ID_A}-metadata.json").write_text("{}")
    _write(drafts_dir, ID_B)
    (drafts_dir / f"{ID_B}-metadata.json").write_text("{}")

    resp = client.delete(f"/api/drafts/{ID_A}")

    assert resp.status_code == 204
    assert resp.content == b""
    assert not (drafts_dir / f"{ID_A}.json").exists()
    assert not (drafts_dir / f"{ID_A}-metadata.json").exists()
    assert attachment.exists()
    assert outside.exists()
    assert (drafts_dir / f"{ID_B}.json").exists()
    assert (drafts_dir / f"{ID_B}-metadata.json").exists()
    assert _ids(client) == [ID_B]


def test_delete_without_sidecar(client, drafts_dir):
    _write(drafts_dir, ID_A)

    resp = client.delete(f"/api/drafts/{ID_A}")

    assert resp.status_code == 204
    assert resp.content == b""
    assert not (drafts_dir / f"{ID_A}.json").exists()


def test_delete_absent_draft_is_404(client):
    assert client.delete(f"/api/drafts/{ID_A}").status_code == 404


def test_delete_absent_draft_with_orphan_sidecar_is_404(client, drafts_dir):
    (drafts_dir / f"{ID_A}-metadata.json").write_text("{}")

    assert client.delete(f"/api/drafts/{ID_A}").status_code == 404


def test_delete_twice(client, drafts_dir):
    _write(drafts_dir, ID_A)

    assert client.delete(f"/api/drafts/{ID_A}").status_code == 204
    assert client.delete(f"/api/drafts/{ID_A}").status_code == 404


def test_delete_corrupt_draft(client, drafts_dir):
    (drafts_dir / f"{ID_A}.json").write_text("{broken")

    assert client.delete(f"/api/drafts/{ID_A}").status_code == 204
    assert not (drafts_dir / f"{ID_A}.json").exists()


def test_deleted_draft_no_longer_served(client, drafts_dir):
    _write(drafts_dir, ID_A)
    client.delete(f"/api/drafts/{ID_A}")

    assert client.get(f"/api/drafts/{ID_A}").status_code == 404


# ---------------------------------------------------------------------------
# TTL cleanup pairs a draft with its sidecar
# ---------------------------------------------------------------------------


def test_cleanup_removes_expired_draft_with_its_sidecar(drafts_dir):
    _write(drafts_dir, ID_A, age=DRAFT_TTL_SECONDS + 60)
    sidecar = drafts_dir / f"{ID_A}-metadata.json"
    sidecar.write_text("{}")
    _write(drafts_dir, ID_B)
    fresh_sidecar = drafts_dir / f"{ID_B}-metadata.json"
    fresh_sidecar.write_text("{}")
    old = time.time() - DRAFT_TTL_SECONDS - 60
    os.utime(fresh_sidecar, (old, old))

    drafts_mod._cleanup_expired_drafts()

    assert not (drafts_dir / f"{ID_A}.json").exists()
    assert not sidecar.exists()
    assert (drafts_dir / f"{ID_B}.json").exists()
    assert fresh_sidecar.exists(), "a live draft's sidecar follows its draft, not its own mtime"


def test_cleanup_leaves_files_outside_the_id_contract(drafts_dir):
    stray = drafts_dir / "draft-test001.json"
    stray.write_text("{}")
    old = time.time() - DRAFT_TTL_SECONDS - 60
    os.utime(stray, (old, old))

    drafts_mod._cleanup_expired_drafts()

    assert stray.exists()


def test_cleanup_removes_expired_orphan_sidecar(drafts_dir):
    orphan = drafts_dir / f"{ID_A}-metadata.json"
    orphan.write_text("{}")
    old = time.time() - DRAFT_TTL_SECONDS - 60
    os.utime(orphan, (old, old))
    fresh_orphan = drafts_dir / f"{ID_B}-metadata.json"
    fresh_orphan.write_text("{}")

    drafts_mod._cleanup_expired_drafts()

    assert not orphan.exists()
    assert fresh_orphan.exists()
