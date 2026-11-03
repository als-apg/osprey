"""ARIEL's multipart upload is the one form-body route, and it needs a header a form cannot send.

Every other mutating route behind the gate takes a JSON body only: the three
body types an HTML form can send (urlencoded, multipart, ``text/plain``) are
refused, because a form posted from a page on the console's own origin passes
the Origin check and needs no script to do so. ``POST /api/entries/upload``
takes ``multipart/form-data`` by design, so it is admitted only with
``X-Osprey-Upload`` — a non-simple header, which a ``<form>`` cannot set and a
cross-origin script cannot send without a preflight this system never grants.
ARIEL's own client sets it.

These tests drive ARIEL's routes behind the real gate rather than a bare router,
since the gate is what enforces the rule.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.ariel.api import routes
from osprey.interfaces.common_middleware import (
    FORM_UPLOAD_HEADER,
    OPERATOR_SECRET_HEADER,
    WebAuthMiddleware,
)
from osprey.interfaces.web_auth import WebCredentials

OPERATOR_SECRET = "ariel-upload-operator-secret"
REFUSAL = {"detail": "this route accepts a JSON body only"}
ARIEL_CLIENT_JS = (
    Path(__file__).resolve().parents[3] / "src/osprey/interfaces/ariel/static/js/api.js"
)


@pytest.fixture
def service() -> MagicMock:
    service = MagicMock()
    service.repository = MagicMock()
    service.repository.upsert_entry = AsyncMock()
    service.repository.store_attachment = AsyncMock()
    service.create_entry = AsyncMock(return_value="entry-001")
    return service


@pytest.fixture
def gated(service) -> TestClient:
    """ARIEL's routes behind :class:`WebAuthMiddleware`, authenticated by the operator header."""
    app = FastAPI()
    app.include_router(routes.router)
    app.state.ariel_service = service
    app.state.config_panel_enabled = True
    app.state.web_credentials = WebCredentials(
        operator_secret=OPERATOR_SECRET, panel_token="ariel-upload-panel-token"
    )
    app.add_middleware(WebAuthMiddleware)
    return TestClient(app)


def _upload(client: TestClient, headers: dict[str, str]):
    return client.post(
        "/api/entries/upload",
        headers={OPERATOR_SECRET_HEADER: OPERATOR_SECRET, **headers},
        data={"subject": "Beam", "details": "Stable."},
        files=[("files", ("shot.png", b"\x89PNG\r\n\x1a\nfakeimage", "image/png"))],
    )


def test_the_upload_without_the_header_is_refused_and_creates_nothing(gated, service):
    """What a plain ``<form enctype="multipart/form-data">`` would send."""
    response = _upload(gated, {})
    assert response.status_code == 415
    assert response.json() == REFUSAL
    service.create_entry.assert_not_called()
    service.repository.upsert_entry.assert_not_called()


def test_the_upload_with_the_header_passes_the_gate(gated):
    """Admitted: whatever the route then answers, it is not the gate's refusal.

    The route itself is exercised by ``test_routes.py``; here only the gate's
    decision is under test, so the route's own answer is not asserted on.
    """
    response = _upload(gated, {FORM_UPLOAD_HEADER: "1"})
    assert response.status_code != 415, response.text
    assert response.json().get("detail") != REFUSAL["detail"]


def test_a_urlencoded_entry_create_is_refused(gated, service):
    """The JSON create route takes no form body, header or not."""
    response = gated.post(
        "/api/entries",
        headers={OPERATOR_SECRET_HEADER: OPERATOR_SECRET, FORM_UPLOAD_HEADER: "1"},
        data={"subject": "Beam", "details": "Stable."},
    )
    assert response.status_code == 415
    service.create_entry.assert_not_called()


def test_the_ariel_client_sends_the_header_on_the_upload():
    """The client half: the one fetch that posts FormData names the header."""
    source = ARIEL_CLIENT_JS.read_text(encoding="utf-8")
    start = source.index("async createWithAttachments(")
    upload = source[start : source.index("return response.json();", start)]
    assert "'/entries/upload'" in upload
    assert FORM_UPLOAD_HEADER in upload
