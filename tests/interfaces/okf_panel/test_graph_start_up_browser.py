"""Browser tests: the KNOWLEDGE panel opened on the graph address, and the page-ready mark.

A graph that throws while it is drawn at start-up leaves the page unable to be
driven: the writer marks it ``failed``. A graph that could not be fetched draws
its own "did not load" state with a Retry, which can be driven: the page is
ready.

Run:
    uv run pytest tests/interfaces/okf_panel/test_graph_start_up_browser.py -m browser
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Route, expect

from osprey.facility.records import FacilityRecords
from osprey.facility.views.graph import graph_text
from osprey.interfaces._serving import run_app_server
from tests.interfaces._browser import wait_for_page_ready
from tests.interfaces.okf_panel.test_graph import BUNDLE, _doc

pytestmark = [pytest.mark.browser, pytest.mark.slow]

#: A graph ``indexGraph`` reads, whose one node carries no ``props``: drawing that
#: node's inspector throws.
_UNDRAWABLE = {"nodes": [{"id": "dev", "kind": "device", "label": "Dev"}], "edges": []}


@pytest.fixture
def okf_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """The panel, served on its own with a facility graph."""
    from osprey.interfaces.okf_panel.app import create_app

    ttl = tmp_path / "facility.ttl"
    ttl.write_text(graph_text(FacilityRecords(_doc())), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with run_app_server(create_app(str(BUNDLE), graph_path=str(ttl))) as url:
        yield url


def test_a_graph_that_throws_while_drawn_at_start_up_marks_the_page_failed(
    chromium_browser: Browser, okf_url: str
) -> None:
    def undrawable(route: Route) -> None:
        route.fulfill(body=json.dumps(_UNDRAWABLE), content_type="application/json")

    page = chromium_browser.new_page()
    try:
        page.route("**/api/graph", undrawable)
        page.goto(f"{okf_url}/#__graph/dev", wait_until="domcontentloaded")
        with pytest.raises(AssertionError, match="failed"):
            wait_for_page_ready(page)
        assert page.locator("html").get_attribute("data-page-ready") == "failed"
    finally:
        page.close()


def test_a_graph_that_did_not_load_draws_its_retry_and_the_page_is_ready(
    chromium_browser: Browser, okf_url: str
) -> None:
    def unavailable(route: Route) -> None:
        route.fulfill(status=500, body="{}", content_type="application/json")

    page = chromium_browser.new_page()
    try:
        page.route("**/api/graph", unavailable)
        page.goto(f"{okf_url}/#__graph", wait_until="domcontentloaded")
        wait_for_page_ready(page)
        assert page.locator("html").get_attribute("data-page-ready") == "true"
        expect(page.get_by_text("The graph did not load")).to_be_visible()
        expect(page.get_by_role("button", name="Retry")).to_be_visible()
    finally:
        page.close()
