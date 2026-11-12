"""The app factories behind the multi-user page recipes (``docs.screenshots.pages``).

Hermetic: the demo build is replaced by a canned :class:`~docs.screenshots.pages.WebTier`,
so no ``osprey`` subprocess runs.
"""

from __future__ import annotations

import pytest
from docs.screenshots import pages
from docs.screenshots.recipes import HISTORY_SESSION, REGISTRY
from fastapi.testclient import TestClient

from osprey.interfaces._serving import authorize_browser_context

_AUTH_ENV = {
    "OSPREY_AUTH_METHOD": "password",
    "OSPREY_AUTH_SESSION_LIFETIME": "43200",
    "OSPREY_AUTH_TLS_ENABLED": "false",
    "OSPREY_AUTH_EXTERNAL_ORIGIN": "http://127.0.0.1:10000",
    "OSPREY_AUTH_USERS": "alice,bob",
    "OSPREY_WEB_THEME": "main",
    "OSPREY_WEB_APP_NAME": "Example Research Facility",
}


@pytest.fixture
def web_tier(monkeypatch) -> pages.WebTier:
    tier = pages.WebTier(landing_html="<html><body>landing</body></html>", auth_env=_AUTH_ENV)
    monkeypatch.setattr(pages, "demo_web_tier", lambda: tier)
    return tier


class _Cookies:
    """The slice of a browser context ``authorize_browser_context`` uses."""

    def __init__(self, client: TestClient) -> None:
        self.client = client

    def add_cookies(self, cookies) -> None:
        for cookie in cookies:
            self.client.cookies.set(cookie["name"], cookie["value"])


def _signed_in(app) -> TestClient:
    """A client holding the operator session a capture's browser holds."""
    client = TestClient(app)
    authorize_browser_context(_Cookies(client))
    return client


def _recipe(name: str):
    return next(shot for shot in REGISTRY if shot.name == name)


def test_landing_app_serves_the_rendered_page(web_tier) -> None:
    response = TestClient(pages.landing_app()).get(_recipe("multi_user_landing").path)

    assert response.status_code == 200
    assert response.text == web_tier.landing_html


def test_sign_in_app_serves_the_password_form(web_tier) -> None:  # noqa: ARG001
    response = TestClient(pages.sign_in_app()).get(_recipe("sign_in").path)

    assert response.status_code == 200
    assert 'id="password"' in response.text
    assert "alice" in response.text


def test_history_app_serves_the_seeded_session() -> None:
    client = _signed_in(pages.history_app())

    session = client.get(_recipe("agent_record").path)
    listing = client.get("/history/")

    assert session.status_code == 200
    assert "SR:DIAG:BPM:05:POSITION:X" in session.text
    assert "Nothing was written." in session.text
    assert listing.status_code == 200
    assert HISTORY_SESSION in listing.text


def test_history_app_ignores_viewer_headers_from_the_browser() -> None:
    client = _signed_in(pages.history_app())

    response = client.get(
        f"/history/session/{HISTORY_SESSION}", headers={"X-Osprey-History-Person": "mallory"}
    )

    assert response.status_code == 200
    assert "mallory" not in response.text
