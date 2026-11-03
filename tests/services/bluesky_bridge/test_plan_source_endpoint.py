"""Coverage for `GET /plans/{name}/source`.

This route is the data source for the approval hook's plan-source
excerpt — the human backstop for the plan validator's documented, accepted
obfuscation residual (see `plan_validation.py`'s module docstring): an
approver who can actually SEE a plan's source has a chance to refuse an
obfuscated body even where the earlier automated stages could not catch it.

Mirrors `test_launch_validation_gate.py`'s isolation fixture and session-plan
helpers — every plan file here is pure pydantic/stdlib (no bluesky import).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import plan_loader
from osprey.services.bluesky_bridge.app import (
    _SOURCE_TRUNCATE_CHARS,
    _SOURCE_TRUNCATE_CHARS_MAX,
    app,
)
from osprey.services.bluesky_bridge.plan_validation import hash_plan_body
from osprey.services.bluesky_bridge.validation_record import validation_records
from tests.services.bluesky_bridge._session_plans import session_plan_source, write_session_plan


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_session_plan_with_no_passing_record_is_unvalidated(client: TestClient) -> None:
    source = session_plan_source("unvalidated_plan")
    write_session_plan("unvalidated_plan", source)

    resp = client.get("/plans/unvalidated_plan/source")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provenance"] == "session"
    assert body["validated"] is False
    assert body["truncated"] is False
    assert body["source"] == source


def test_session_plan_with_a_passing_record_is_validated(client: TestClient) -> None:
    source = session_plan_source("validated_plan")
    write_session_plan("validated_plan", source)
    validation_records.record(hash_plan_body(source))

    resp = client.get("/plans/validated_plan/source")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provenance"] == "session"
    assert body["validated"] is True


def test_session_plan_source_is_truncated_beyond_the_bound(client: TestClient) -> None:
    oversized_body = "\n# padding\n" * (_SOURCE_TRUNCATE_CHARS // 5)
    source = session_plan_source("huge_plan", body=oversized_body)
    assert len(source) > _SOURCE_TRUNCATE_CHARS
    write_session_plan("huge_plan", source)

    resp = client.get("/plans/huge_plan/source")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["truncated"] is True
    assert len(body["source"]) == _SOURCE_TRUNCATE_CHARS
    assert body["source"] == source[:_SOURCE_TRUNCATE_CHARS]


def test_max_chars_raises_the_bound_for_full_source_reads(client: TestClient) -> None:
    """The plan panel's Source tab asks for more than the skim default via
    `?max_chars=` — a source longer than the default but within the ask comes
    back whole, and the default-bound behavior is unchanged for callers (the
    approval hook) that don't pass the param."""
    oversized_body = "\n# padding\n" * (_SOURCE_TRUNCATE_CHARS // 5)
    source = session_plan_source("full_read_plan", body=oversized_body)
    assert _SOURCE_TRUNCATE_CHARS < len(source) < _SOURCE_TRUNCATE_CHARS_MAX
    write_session_plan("full_read_plan", source)

    resp = client.get(f"/plans/full_read_plan/source?max_chars={_SOURCE_TRUNCATE_CHARS_MAX}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["truncated"] is False
    assert body["source"] == source


def test_max_chars_beyond_the_ceiling_is_422(client: TestClient) -> None:
    """The explicit ask is itself capped — the response can never be made
    unbounded by a client-chosen number."""
    source = session_plan_source("capped_plan")
    write_session_plan("capped_plan", source)

    resp = client.get(f"/plans/capped_plan/source?max_chars={_SOURCE_TRUNCATE_CHARS_MAX + 1}")
    assert resp.status_code == 422

    resp = client.get("/plans/capped_plan/source?max_chars=0")
    assert resp.status_code == 422


def test_unknown_plan_name_is_404(client: TestClient) -> None:
    resp = client.get("/plans/does_not_exist_anywhere/source")
    assert resp.status_code == 404


def test_non_identifier_name_is_400(client: TestClient) -> None:
    """`name` is sanitized the SAME way `/plans/session` and `/plans/validate`
    already do (task 2.3's `_sanitize_plan_name`) before it ever touches a
    filesystem path — a file-reading route must not rely implicitly on
    FastAPI's no-slash-in-path-param behavior for traversal safety. A bare
    single-segment name that isn't a valid identifier reaches the handler
    (no embedded slash to trip routing) and must 400 from the sanitizer."""
    resp = client.get("/plans/..%2E%2E/source")
    assert resp.status_code == 400

    resp = client.get("/plans/not-a-valid-identifier/source")
    assert resp.status_code == 400


def test_embedded_slash_name_never_reaches_the_handler(client: TestClient) -> None:
    """An encoded-slash traversal attempt (`..%2F..%2Fetc%2Fpasswd`) doesn't
    even match the single-segment `{name}` path parameter — FastAPI's own
    routing 404s it before `_sanitize_plan_name` ever runs. Documented here
    so a routing change that started letting such paths through would be
    caught by a test, not just an implicit assumption."""
    resp = client.get("/plans/..%2F..%2Fetc%2Fpasswd/source")
    assert resp.status_code == 404

    resp = client.get("/plans/not a valid identifier/source")
    assert resp.status_code == 400


def test_shipped_plan_is_located_by_declared_name_not_file_stem(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shipped-tier file may declare a `PLAN_METADATA["name"]` different
    from its own filename (e.g. a `some_file_stem.py` declaring
    `"name": "declared_plan_name"`) — the source route must resolve by the
    declared name, not the stem."""
    shipped_dir = tmp_path / "shipped"
    shipped_dir.mkdir()
    (shipped_dir / "some_file_stem.py").write_text(session_plan_source("declared_plan_name"))
    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", shipped_dir)

    resp = client.get("/plans/declared_plan_name/source")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provenance"] == "shipped"
    # Shipped/preset/facility tiers carry no validation-record gate at all —
    # trusted by construction, not by a passing record.
    assert body["validated"] is True
    assert "declared_plan_name" in body["source"]


def test_re_authored_session_plan_re_reflects_unvalidated_after_edit(client: TestClient) -> None:
    """A previously-validated session file, edited afterward, is honestly
    reported as unvalidated again — mirrors the load/launch gates' own
    re-hash-on-every-check behavior (never a stale cached verdict)."""
    original = session_plan_source("edited_plan")
    path = write_session_plan("edited_plan", original)
    validation_records.record(hash_plan_body(original))
    assert client.get("/plans/edited_plan/source").json()["validated"] is True

    edited = session_plan_source("edited_plan", body="\n# an edit\n")
    path.write_text(edited)

    resp = client.get("/plans/edited_plan/source")
    body = resp.json()
    assert body["validated"] is False
    assert body["source"] == edited


class _ExternalBackendStub:
    """A queue backend whose worker is external and owns one plan."""

    external_worker = True

    async def external_plan_catalog(self):
        return [{"name": "scan_request_plan"}]


class _ExternalBackendUnreachable:
    external_worker = True

    async def external_plan_catalog(self):
        from osprey.services.bluesky_bridge.queue_backend import QueueUnavailableError

        raise QueueUnavailableError("manager down")


def test_external_plan_source_is_an_honest_answer_not_a_404(client: TestClient) -> None:
    """A manager-owned plan has no local file BY DESIGN; the panel must not
    treat that as a failed selection (it tears the parameter form down)."""
    from osprey.services.bluesky_bridge.app import set_queue_backend

    set_queue_backend(_ExternalBackendStub())
    try:
        response = client.get("/plans/scan_request_plan/source")
        assert response.status_code == 200
        payload = response.json()
        assert payload["external"] is True
        assert payload["provenance"] == "facility"
        assert payload["validated"] is True
        assert "facility worker" in payload["source"]
    finally:
        set_queue_backend(None)


def test_external_unknown_plan_still_404s(client: TestClient) -> None:
    from osprey.services.bluesky_bridge.app import set_queue_backend

    set_queue_backend(_ExternalBackendStub())
    try:
        assert client.get("/plans/not_a_plan/source").status_code == 404
    finally:
        set_queue_backend(None)


def test_external_source_with_unreachable_manager_404s_rather_than_guessing(
    client: TestClient,
) -> None:
    from osprey.services.bluesky_bridge.app import set_queue_backend

    set_queue_backend(_ExternalBackendUnreachable())
    try:
        assert client.get("/plans/scan_request_plan/source").status_code == 404
    finally:
        set_queue_backend(None)
