"""Cap / validation tests for dispatch input files.

Two layers:

* the pure ``validate_input_files`` policy function (duck-typed on ``mime`` /
  ``content_b64`` / ``ingest``), covering every cap and the invalid-input codes;
* the HTTP surface — a valid batch is accepted (202) and each violation class
  rejects the whole request with the exact machine-readable 400 ``detail``.

Hermetic: FastAPI ``TestClient`` and direct calls, no network, no real SDK.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from osprey.mcp_server.dispatch_worker import dispatch_api
from osprey.mcp_server.dispatch_worker import input_files_policy as policy
from osprey.mcp_server.dispatch_worker.input_files_policy import (
    DETAIL_CAP_EXCEEDED,
    DETAIL_INVALID,
    MAX_IMAGE_BYTES,
    MAX_INGEST_FILES,
    MAX_TEXT_BYTES,
    InputFilesError,
    validate_input_files,
)

_TOKEN = "test-secret-token"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _b64(nbytes: int) -> str:
    """Base64 of ``nbytes`` zero bytes (decodes back to exactly ``nbytes``)."""
    return base64.b64encode(b"\x00" * nbytes).decode("ascii")


@dataclass
class _Spec:
    """Duck-typed stand-in for an InputFile at the policy layer."""

    mime: str
    content_b64: str
    ingest: bool = True


def _payload(specs: list[_Spec]) -> list[dict[str, Any]]:
    """Turn policy specs into request-body input_files dicts."""
    return [
        {"filename": f"f{i}", "mime": s.mime, "content_b64": s.content_b64, "ingest": s.ingest}
        for i, s in enumerate(specs)
    ]


# --------------------------------------------------------------------------
# pure validate_input_files — caps
# --------------------------------------------------------------------------


def test_input_files_caps_empty_and_none_pass():
    validate_input_files([])
    validate_input_files(None)  # falsy short-circuit


def test_input_files_caps_valid_batch_passes():
    validate_input_files(
        [
            _Spec("image/png", _b64(1024)),
            _Spec("text/csv", _b64(2048), ingest=False),
            _Spec("application/json", _b64(10)),
        ]
    )


def test_input_files_caps_ingest_count_boundary_ok():
    validate_input_files([_Spec("text/plain", _b64(10)) for _ in range(MAX_INGEST_FILES)])


def test_input_files_caps_ingest_count_exceeded():
    specs = [_Spec("text/plain", _b64(10)) for _ in range(MAX_INGEST_FILES + 1)]
    with pytest.raises(InputFilesError) as ei:
        validate_input_files(specs)
    assert ei.value.detail == DETAIL_CAP_EXCEEDED


def test_input_files_caps_ingest_false_not_counted():
    # 10 files but only ingest=True ones count toward MAX_INGEST_FILES.
    specs = [_Spec("text/plain", _b64(10), ingest=False) for _ in range(10)]
    validate_input_files(specs)


def test_input_files_caps_image_per_file_boundary_ok():
    validate_input_files([_Spec("image/png", _b64(MAX_IMAGE_BYTES))])


def test_input_files_caps_image_per_file_exceeded():
    with pytest.raises(InputFilesError) as ei:
        validate_input_files([_Spec("image/png", _b64(MAX_IMAGE_BYTES + 1))])
    assert ei.value.detail == DETAIL_CAP_EXCEEDED


def test_input_files_caps_text_per_file_boundary_ok():
    validate_input_files([_Spec("text/csv", _b64(MAX_TEXT_BYTES))])


def test_input_files_caps_text_per_file_exceeded():
    with pytest.raises(InputFilesError) as ei:
        validate_input_files([_Spec("application/json", _b64(MAX_TEXT_BYTES + 1))])
    assert ei.value.detail == DETAIL_CAP_EXCEEDED


def test_input_files_caps_total_exceeded_via_non_ingest():
    # Four 5 MB images (each within the per-file image cap) but ingest=False, so
    # the ingest-count cap is untouched — only the 18 MB total ceiling trips.
    specs = [_Spec("image/png", _b64(MAX_IMAGE_BYTES), ingest=False) for _ in range(4)]
    with pytest.raises(InputFilesError) as ei:
        validate_input_files(specs)
    assert ei.value.detail == DETAIL_CAP_EXCEEDED


# --------------------------------------------------------------------------
# pure validate_input_files — invalid
# --------------------------------------------------------------------------


def test_input_files_caps_disallowed_mime_invalid():
    with pytest.raises(InputFilesError) as ei:
        validate_input_files([_Spec("application/pdf", _b64(10))])
    assert ei.value.detail == DETAIL_INVALID


def test_input_files_caps_undecodable_b64_invalid():
    with pytest.raises(InputFilesError) as ei:
        validate_input_files([_Spec("text/plain", "not valid base64!!!")])
    assert ei.value.detail == DETAIL_INVALID


def test_input_files_caps_decoded_size_roundtrips():
    assert policy.decoded_size(_b64(1234)) == 1234


# --------------------------------------------------------------------------
# HTTP surface
# --------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DISPATCH_WORKER_TOKEN", _TOKEN)
    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_queues", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})

    async def _fake_run_dispatch(*, event_queue=None, **_kwargs):
        if event_queue is not None:
            await event_queue.put({"type": "done"})
        return {
            "status": "completed",
            "text_output": "ok",
            "tool_calls": [],
            "error": None,
            "duration_sec": 0.01,
            "cost_usd": 0.0,
            "num_turns": 1,
        }

    monkeypatch.setattr(dispatch_api.sdk_runner, "run_dispatch", _fake_run_dispatch)
    monkeypatch.setattr(dispatch_api, "_persist_run", lambda run_id, run: None)
    with TestClient(dispatch_api.app) as c:
        yield c


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {_TOKEN}"}


def test_input_files_caps_http_valid_accepted(client):
    resp = client.post(
        "/dispatch",
        json={
            "prompt": "look",
            "allowed_tools": ["Read"],
            "input_files": _payload([_Spec("image/png", _b64(1024))]),
        },
        headers=_auth(),
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "accepted"


def test_input_files_caps_http_omitted_accepted(client):
    resp = client.post(
        "/dispatch",
        json={"prompt": "look", "allowed_tools": ["Read"]},
        headers=_auth(),
    )
    assert resp.status_code == 202


def test_input_files_caps_http_count_exceeded_400(client):
    specs = [_Spec("text/plain", _b64(10)) for _ in range(MAX_INGEST_FILES + 1)]
    resp = client.post(
        "/dispatch",
        json={"prompt": "x", "allowed_tools": ["Read"], "input_files": _payload(specs)},
        headers=_auth(),
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == DETAIL_CAP_EXCEEDED
    # Rejected before any run was created.
    assert dispatch_api._runs == {}
    assert dispatch_api._tasks == {}


def test_input_files_caps_http_image_too_large_400(client):
    resp = client.post(
        "/dispatch",
        json={
            "prompt": "x",
            "allowed_tools": ["Read"],
            "input_files": _payload([_Spec("image/png", _b64(MAX_IMAGE_BYTES + 1))]),
        },
        headers=_auth(),
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == DETAIL_CAP_EXCEEDED


def test_input_files_caps_http_bad_mime_invalid_400(client):
    resp = client.post(
        "/dispatch",
        json={
            "prompt": "x",
            "allowed_tools": ["Read"],
            "input_files": _payload([_Spec("application/pdf", _b64(10))]),
        },
        headers=_auth(),
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == DETAIL_INVALID


def test_input_files_caps_http_bad_b64_invalid_400(client):
    resp = client.post(
        "/dispatch",
        json={
            "prompt": "x",
            "allowed_tools": ["Read"],
            "input_files": _payload([_Spec("text/plain", "@@@not-b64@@@")]),
        },
        headers=_auth(),
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == DETAIL_INVALID
