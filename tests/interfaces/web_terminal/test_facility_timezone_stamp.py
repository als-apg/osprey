"""Tests for the facility-timezone stamp on the web terminal's served pages.

Both Jinja-rendered documents (``index.html`` and ``static/session.html``)
carry ``data-facility-timezone="<IANA id>"`` on ``<html>``. The value is the
zone ``system.timezone`` names, resolved by the same call the operator chat's
system prompt uses, so the page and the agent read one clock.

Unlike the storage scope, the stamp is always present: the resolver never
raises and degrades to ``UTC`` on an unconfigured deployment, which is also
what the agent is told.
"""

from __future__ import annotations

import re
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app

#: The attribute under test.
ATTR = "data-facility-timezone"

# (page id, request path) -- both served HTML documents.
_PAGES = [
    ("index", "/"),
    ("session", "/static/session.html"),
]
_PAGE_IDS = [p[0] for p in _PAGES]


@pytest.fixture
def workspace_dir(tmp_path):
    """Temporary workspace directory for the app to watch."""
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


def _html_tag(body: str) -> str:
    """The document's opening ``<html …>`` tag."""
    match = re.search(r"<html\b[^>]*>", body)
    assert match, "served document has no <html> element"
    return match.group(0)


def _serve(workspace_dir, zone: str) -> dict[str, str]:
    """Render both pages with the facility resolver answering *zone*.

    Returns:
        ``{page_id: response_body}`` for every page in :data:`_PAGES`.
    """
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app.get_facility_timezone",
            return_value=ZoneInfo(zone),
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as client:
            bodies = {}
            for page_id, path in _PAGES:
                response = client.get(path)
                assert response.status_code == 200, f"{page_id} did not render"
                bodies[page_id] = response.text
            return bodies


@pytest.mark.parametrize("page_id", _PAGE_IDS)
def test_stamp_names_the_facility_zone(workspace_dir, page_id):
    bodies = _serve(workspace_dir, "Asia/Tokyo")
    assert f'{ATTR}="Asia/Tokyo"' in _html_tag(bodies[page_id])


@pytest.mark.parametrize("page_id", _PAGE_IDS)
def test_stamped_exactly_once(workspace_dir, page_id):
    bodies = _serve(workspace_dir, "Asia/Tokyo")
    assert bodies[page_id].count(ATTR) == 1


@pytest.mark.parametrize("page_id", _PAGE_IDS)
def test_unconfigured_deployment_stamps_utc(workspace_dir, page_id):
    """The resolver degrades to UTC, so the stamp is never absent."""
    bodies = _serve(workspace_dir, "UTC")
    assert f'{ATTR}="UTC"' in _html_tag(bodies[page_id])


def test_index_keeps_its_existing_stamps(workspace_dir):
    tag = _html_tag(_serve(workspace_dir, "Asia/Tokyo")["index"])
    for attr in ("data-theme=", "data-ui-mode=", "data-rail-position=", "data-bar-context="):
        assert attr in tag


def test_pages_and_operator_chat_share_one_resolver():
    """The page stamp and the operator chat's system prompt read one clock."""
    from osprey.interfaces.web_terminal import app as web_app
    from osprey.interfaces.web_terminal import operator_session

    assert web_app.get_facility_timezone is operator_session.get_facility_timezone
