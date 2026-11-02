"""Real-browser suite for ARIEL's one screen: rail | list | reader.

Covers what only a live page proves: the numbered Requirements of the
one-screen plan (standalone landing, selection in the hash and its reload,
deep links and their error states, search surfaces, create and draft review,
per-frame persistence, the stacked layout), each rail mailbox, the rail's
drag / snap / toggle, each key of the keyboard table, every legacy hash alias
standalone (``test_embedded_nav.py`` holds the embedded side), and the shared
design-system primitives the panes are built from.

The panel is the real ``create_app()`` on a real server (``launch_ariel``);
the database it lacks is faked at the network layer by ``FakeBackend``, which
answers every ``/api/...`` request the screen makes from fixed fixtures and
records each one, so a test can assert what the panel asked for.

Run:
    uv run pytest tests/interfaces/ariel/test_shell.py -m browser -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, quote, urlparse

import pytest
import yaml

from tests.interfaces._browser import assert_page_loads_clean

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser, Page, Route

# Guarded only so collection succeeds where playwright is not installed; the
# skip itself comes from the shared ``chromium_browser`` fixture.
try:
    from playwright.sync_api import expect
except ImportError:  # pragma: no cover
    expect = None  # type: ignore[assignment]

pytestmark = [pytest.mark.browser, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Fixtures: one panel per module, a faked backend per page
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _ariel_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Any, str]]:
    """One DB-less standalone ARIEL for the whole module, as ``launch_ariel`` boots it."""
    from osprey.interfaces._serving import run_app_server
    from osprey.interfaces.ariel.app import create_app

    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path_factory.mktemp("ariel-shell"))
        app = create_app()
        with run_app_server(app) as base_url:
            yield app, base_url.rstrip("/")


@pytest.fixture
def ariel_url(_ariel_server: tuple[Any, str]) -> str:
    """The shared panel's base URL, its auth gate re-pointed at this test's credentials.

    The root conftest replaces the process credential holder before every test,
    and the gate of an app built earlier keeps verifying against the old one;
    re-pointing it is what lets this test's pre-authorized browser in.
    """
    from osprey.interfaces.web_auth import get_web_credentials

    app, base_url = _ariel_server
    app.state.web_credentials = get_web_credentials()
    return base_url


HEALTHY_CAPABILITIES: dict[str, Any] = {
    "status": "healthy",
    "default_mode": "keyword",
    "categories": {
        "search": {
            "label": "Search",
            "modes": [
                {"name": "keyword", "label": "Keyword", "description": "", "parameters": []},
                {"name": "semantic", "label": "Semantic", "description": "", "parameters": []},
            ],
        }
    },
    "shared_parameters": [],
}

INVALID_CAPABILITIES: dict[str, Any] = {
    **HEALTHY_CAPABILITIES,
    "status": "configuration_invalid",
    "config_errors": ["ariel.vocabulary.path: file not found"],
    "remedy": "repoint ariel.vocabulary.path, then restart",
}

#: A 1x1 transparent PNG, served for every ``/attachments/...`` image.
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def make_entry(
    entry_id: str,
    subject: str,
    *,
    timestamp: str = "2026-08-20T12:00:00-07:00",
    author: str = "operator",
    attachments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """An ``EntryResponse``-shaped entry."""
    return {
        "entry_id": entry_id,
        "source_system": "olog",
        "timestamp": timestamp,
        "author": author,
        "raw_text": f"{subject}\nBody of {entry_id}.",
        "attachments": attachments or [],
        "metadata": {},
        "created_at": timestamp,
        "updated_at": None,
        "summary": None,
        "keywords": [],
        "score": None,
        "highlights": [],
    }


#: A viewable picture on E1, the target of every ``att=`` deep link here.
PICTURE = {
    "attachment_id": "A1",
    "filename": "beam.png",
    "mime_type": "image/png",
    "viewable": True,
    "display_url": "/attachments/A1",
}

DEFAULT_ENTRIES = [
    make_entry("E1", "Beam dump at injection", attachments=[PICTURE]),
    make_entry("E2", "Vacuum spike sector 3", timestamp="2026-08-20T09:00:00-07:00"),
    make_entry("E3", "Shift handover notes", timestamp="2026-08-19T17:00:00-07:00"),
]

#: Fetchable by id, never in the listing: the target of the deep-link test.
OFF_LIST = make_entry("E9", "Old entry far down the log", timestamp="2025-01-02T08:00:00-07:00")

DRAFT = {
    "draft_id": "D1",
    "subject": "Drafted by the agent",
    "details": "Agent-written details.",
    "author": "agent",
    "logbook": None,
    "shift": None,
    "tags": ["agent"],
    "attachment_paths": [],
    "metadata": {},
}


def search_response(
    entries: list[dict[str, Any]],
    *,
    diagnostics: list[dict[str, Any]] | None = None,
    answer: str | None = None,
    expanded_terms: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A ``SearchResponse``-shaped body."""
    return {
        "entries": entries,
        "answer": answer,
        "sources": [],
        "search_modes_used": ["keyword"],
        "reasoning": "",
        "total_results": len(entries),
        "execution_time_ms": 3,
        "diagnostics": diagnostics or [],
        "expanded_terms": expanded_terms or [],
    }


@dataclass
class Request:
    """One recorded ``/api`` request."""

    method: str
    path: str
    query: dict[str, list[str]]
    body: str | None

    def json(self) -> Any:
        """The request body parsed as JSON."""
        return json.loads(self.body or "null")


@dataclass
class FakeBackend:
    """Answers the screen's ``/api`` requests from fixtures and records them.

    Anything it does not know (status, config) falls through to the real
    DB-less backend.
    """

    capabilities: dict[str, Any] = field(default_factory=lambda: dict(HEALTHY_CAPABILITIES))
    entries: list[dict[str, Any]] = field(default_factory=lambda: list(DEFAULT_ENTRIES))
    publish_info: dict[str, Any] = field(
        default_factory=lambda: {
            "supports_write": True,
            "requires_auth": False,
            "source_system": "olog",
            "entry_fields": [],
        }
    )
    drafts: dict[str, dict[str, Any]] = field(default_factory=lambda: {"D1": dict(DRAFT)})
    search: dict[str, Any] = field(default_factory=lambda: search_response(DEFAULT_ENTRIES[:2]))
    requests: list[Request] = field(default_factory=list)
    created: int = 0

    def install(self, page: Page) -> FakeBackend:
        """Route the page's ``/api`` and ``/attachments`` traffic here."""
        page.route(re.compile(r".*/api/.*"), self._handle)
        page.route(
            re.compile(r".*/attachments/.*"),
            lambda route: route.fulfill(status=200, content_type="image/png", body=PNG_1X1),
        )
        return self

    def details(self) -> dict[str, dict[str, Any]]:
        """Every entry fetchable by id."""
        return {e["entry_id"]: e for e in [*self.entries, OFF_LIST]}

    def calls(self, method: str, path: str) -> list[Request]:
        """Recorded requests to *path* (exact) with *method*."""
        return [r for r in self.requests if r.method == method and r.path == path]

    def _handle(self, route: Route) -> None:
        req = route.request
        url = urlparse(req.url)
        path = url.path[url.path.index("/api/") :]
        raw = req.post_data_buffer
        body = raw.decode("utf-8", "replace") if raw is not None else None
        record = Request(req.method, path, parse_qs(url.query), body)
        self.requests.append(record)

        def ok(body: Any, status: int = 200) -> None:
            route.fulfill(status=status, content_type="application/json", body=json.dumps(body))

        if path == "/api/capabilities":
            ok(self.capabilities)
        elif path == "/api/publish-info":
            ok(self.publish_info)
        elif path == "/api/search" and req.method == "POST":
            ok(self.search)
        elif path == "/api/entries" and req.method == "GET":
            author = record.query.get("author", [None])[0]
            rows = [e for e in self.entries if author is None or e["author"] == author]
            ok({"entries": rows, "total": len(rows), "page": 1, "page_size": 20, "total_pages": 1})
        elif path in ("/api/entries", "/api/entries/upload") and req.method == "POST":
            self.created += 1
            new_id = f"NEW-{self.created}"
            self.entries.insert(0, make_entry(new_id, "Freshly created entry"))
            ok({"entry_id": new_id, "message": "created"}, status=201)
        elif path.startswith("/api/entries/"):
            entry = self.details().get(path.rsplit("/", 1)[1])
            ok(entry, 200) if entry else ok({"detail": "Entry not found"}, 404)
        elif path.startswith("/api/drafts/"):
            draft = self.drafts.get(path.rsplit("/", 1)[1])
            ok(draft, 200) if draft else ok({"detail": "Draft not found"}, 404)
        else:
            route.continue_()


def open_panel(
    browser: Browser,
    url: str,
    backend: FakeBackend | None = None,
    *,
    viewport: dict[str, int] | None = None,
) -> tuple[Page, FakeBackend]:
    """A new page on *url* behind a fake backend, with the list settled."""
    page = browser.new_page(viewport=viewport or {"width": 1280, "height": 800})
    backend = (backend or FakeBackend()).install(page)
    page.goto(url, wait_until="load")
    expect(page.locator("#ariel-list-body")).to_have_attribute("aria-busy", "false")
    return page, backend


def row(page: Page, entry_id: str):
    """The list row of *entry_id*."""
    return page.locator(f'#ariel-list [data-entry-id="{entry_id}"]')


def reader_title(page: Page):
    """The reader's entry title."""
    return page.locator("#ariel-reader-body .entry-detail-title")


def drag(page: Page, selector: str, dx: float) -> None:
    """Drag the element under *selector* horizontally by *dx* pixels."""
    box = page.locator(selector).bounding_box()
    assert box is not None, selector
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + dx / 2, y, steps=4)
    page.mouse.move(x + dx, y, steps=4)
    page.mouse.up()


def width(page: Page, selector: str) -> float:
    """Rendered width of *selector*."""
    box = page.locator(selector).bounding_box()
    assert box is not None, selector
    return box["width"]


def canonical(
    *,
    q: str | None = None,
    mode: str | None = None,
    box: str = "all",
    sel: str | None = None,
    att: str | None = None,
) -> str:
    """The router's canonical hash: keys in order, empty ones omitted (``box`` never is)."""
    pairs = [("q", q), ("mode", mode), ("box", box), ("sel", sel), ("att", att)]
    return "#" + "&".join(f"{k}={quote(v, safe='')}" for k, v in pairs if v)


def wait_for_hash(page: Page, expected: str) -> None:
    """Wait until the location hash is *expected* (the router rewrites it asynchronously)."""
    page.wait_for_function("(h) => location.hash === h", arg=expected)


def app(page: Page):
    """The panel root carrying the layout attributes."""
    return page.locator(".app-container")


# ---------------------------------------------------------------------------
# Requirement 1: the standalone landing
# ---------------------------------------------------------------------------


def test_req1_standalone_shows_rail_newest_entries_and_empty_reader(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Rail, newest-first list with day separators, empty reader; no page navigation."""
    page = chromium_browser.new_page()
    FakeBackend().install(page)
    assert_page_loads_clean(page, f"{ariel_url}/")

    expect(page.locator("#ariel-rail button[data-box]")).to_have_count(3)
    expect(page.locator("#ariel-rail")).to_be_visible()
    expect(app(page)).not_to_have_attribute("data-rail-collapsed", re.compile(".*"))
    ids = page.locator("#ariel-list [data-entry-id]")
    expect(ids).to_have_count(3)
    assert ids.evaluate_all("rows => rows.map(r => r.dataset.entryId)") == ["E1", "E2", "E3"]
    # Two calendar days in the fixtures -> two separators.
    expect(page.locator("#ariel-list .ariel-day-sep")).to_have_count(2)
    expect(page.locator("#ariel-reader")).to_be_visible()
    expect(page.locator("#ariel-reader-body")).to_be_empty()
    # The one screen: no view sections left to navigate between.
    expect(page.locator("[id^='view-']")).to_have_count(0)

    # Opening an entry stays in this document.
    page.evaluate("window.__sameDocument = true")
    row(page, "E2").click()
    expect(reader_title(page)).to_have_text("Vacuum spike sector 3")
    assert page.evaluate("window.__sameDocument") is True
    page.close()


# ---------------------------------------------------------------------------
# Requirement 2: selection in the hash, restored on reload
# ---------------------------------------------------------------------------


def test_req2_click_and_keyboard_selection_record_the_hash(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """A click and a ``j`` both show the entry and record ``sel=`` in the hash."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")

    row(page, "E1").click()
    expect(reader_title(page)).to_have_text("Beam dump at injection")
    expect(row(page, "E1")).to_have_attribute("aria-selected", "true")
    wait_for_hash(page, canonical(sel="E1"))

    page.locator("#ariel-reader-body").click(position={"x": 5, "y": 5})
    page.keyboard.press("j")
    expect(reader_title(page)).to_have_text("Vacuum spike sector 3")
    wait_for_hash(page, canonical(sel="E2"))

    page.reload(wait_until="load")
    expect(reader_title(page)).to_have_text("Vacuum spike sector 3")
    expect(row(page, "E2")).to_have_attribute("aria-selected", "true")
    page.close()


def test_req2_reload_restores_query_mode_selection_and_attachment(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """A canonical hash brings back the query, the mode, the entry and its open picture."""
    page, backend = open_panel(chromium_browser, f"{ariel_url}/#q=beam&mode=semantic&sel=E1&att=A1")

    expect(page.locator("#ariel-search-input")).to_have_value("beam")
    expect(page.locator("#ariel-mode-chip")).to_have_attribute("data-mode", "semantic")
    expect(reader_title(page)).to_have_text("Beam dump at injection")
    expect(page.locator("#image-lightbox")).to_be_visible()
    searches = backend.calls("POST", "/api/search")
    assert searches, backend.requests
    assert searches[-1].json()["query"] == "beam"
    assert searches[-1].json()["mode"] == "semantic"

    page.reload(wait_until="load")
    expect(page.locator("#ariel-search-input")).to_have_value("beam")
    expect(page.locator("#ariel-mode-chip")).to_have_attribute("data-mode", "semantic")
    expect(page.locator("#image-lightbox")).to_be_visible()
    wait_for_hash(page, canonical(q="beam", mode="semantic", sel="E1", att="A1"))
    page.close()


# ---------------------------------------------------------------------------
# Requirement 3: deep links and their error states
# ---------------------------------------------------------------------------


def test_req3_deep_link_to_an_entry_not_in_the_list(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """An entry outside the loaded rows still opens in the reader."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/#entry?id=E9")

    expect(reader_title(page)).to_have_text("Old entry far down the log")
    expect(row(page, "E9")).to_have_count(0)
    wait_for_hash(page, canonical(sel="E9"))
    page.close()


def test_req3_unknown_id_shows_the_reader_error_state(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """An id the backend does not know renders the reader's error state."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/#entry?id=no-such-entry")

    expect(page.locator("#ariel-reader-body")).to_contain_text("Failed to Load Entry")
    expect(page.locator("#image-lightbox")).to_have_count(0)
    page.close()


def test_req3_unfetchable_draft_leaves_the_form_empty_with_an_error(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """``#create?draft=`` of a missing draft opens an empty form and reports the failure."""
    page = chromium_browser.new_page()
    backend = FakeBackend().install(page)
    with page.expect_console_message(lambda m: "Failed to load draft" in m.text):
        page.goto(f"{ariel_url}/#create?draft=missing", wait_until="load")

    expect(page.locator("#ariel-compose")).to_be_visible()
    assert backend.calls("GET", "/api/drafts/missing")
    expect(page.locator("#entry-subject")).to_have_value("")
    expect(page.locator("#entry-details")).to_have_value("")
    expect(page.locator("#draft-banner")).to_have_count(0)
    page.close()


# ---------------------------------------------------------------------------
# Requirement 4: search surfaces (config + expansion banners: test_vocabulary_banner)
# ---------------------------------------------------------------------------


def test_req4_search_shows_diagnostics_and_no_answer_box(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Diagnostics, counts and rows render; a response's ``answer`` is never shown."""
    backend = FakeBackend(
        search=search_response(
            [DEFAULT_ENTRIES[1]],
            diagnostics=[{"level": "warning", "source": "semantic", "message": "model offline"}],
            answer="THE GENERATED ANSWER",
        )
    )
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", backend)

    page.fill("#ariel-search-input", "vacuum")
    page.press("#ariel-search-input", "Enter")

    expect(page.locator("#ariel-list .ariel-row")).to_have_count(1)
    expect(row(page, "E2")).to_be_visible()
    expect(page.locator("#ariel-list .diagnostics-bar")).to_contain_text("1 issue detected")
    expect(page.locator("#ariel-list .diagnostics-messages")).to_contain_text("model offline")
    expect(page.locator("#ariel-list .results-count")).to_contain_text("1")
    expect(page.locator("body")).not_to_contain_text("THE GENERATED ANSWER")
    wait_for_hash(page, canonical(q="vacuum"))
    page.close()


def test_req4_invalid_configuration_disables_search(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """``configuration_invalid`` shows the blocking banner and disables the box and chip."""
    page, _ = open_panel(
        chromium_browser, f"{ariel_url}/", FakeBackend(capabilities=dict(INVALID_CAPABILITIES))
    )

    expect(page.locator("#config-invalid-banner")).to_be_visible()
    expect(page.locator("#ariel-search-input")).to_be_disabled()
    expect(page.locator("#ariel-mode-chip")).to_be_disabled()
    # The newest-first listing still shows under the banner.
    expect(row(page, "E1")).to_be_visible()
    page.close()


# ---------------------------------------------------------------------------
# Requirement 5: create (plain, attachments, credentials) and draft review
# ---------------------------------------------------------------------------


def _fill_entry(page: Page, subject: str = "New entry subject") -> None:
    page.fill("#entry-subject", subject)
    page.fill("#entry-details", "Some details.")
    page.fill("#entry-author", "bob")


def test_req5_create_selects_the_new_entry_and_remembers_the_author(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Compose opens over the list; submitting shows the new entry and feeds Mine."""
    page, backend = open_panel(chromium_browser, f"{ariel_url}/")

    page.click("#ariel-compose-btn")
    expect(page.locator("#ariel-compose")).to_be_visible()
    expect(page.locator("#ariel-reader")).to_be_visible()  # never over the reader
    expect(page.locator("#publish-credentials")).to_be_hidden()
    _fill_entry(page)
    page.click("#create-entry-form button[type=submit]")

    expect(page.locator("#ariel-compose")).to_be_hidden()
    expect(reader_title(page)).to_have_text("Freshly created entry")
    wait_for_hash(page, canonical(sel="NEW-1"))
    body = backend.calls("POST", "/api/entries")[-1].json()
    assert body["subject"] == "New entry subject"
    assert body["author"] == "bob"

    # The name just used is Mine's: no prompt, the listing is scoped to it.
    page.click('#ariel-rail button[data-box="mine"]')
    expect(page.locator('[data-rail-field="mine"]')).to_be_hidden()
    page.wait_for_function("location.hash.includes('box=mine')")
    assert any(r.query.get("author") == ["bob"] for r in backend.calls("GET", "/api/entries"))
    page.close()


def test_req5_create_with_attachment_and_credentials(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """An adapter that needs credentials asks for them; files go multipart."""
    backend = FakeBackend()
    backend.publish_info["requires_auth"] = True
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", backend)

    page.click("#ariel-compose-btn")
    expect(page.locator("#publish-credentials")).to_be_visible()
    _fill_entry(page)
    page.fill("#entry-auth-user", "opsuser")
    page.fill("#entry-auth-password", "s3cret")
    page.set_input_files(
        "#entry-files", files=[{"name": "plot.png", "mimeType": "image/png", "buffer": PNG_1X1}]
    )
    page.click("#create-entry-form button[type=submit]")

    expect(reader_title(page)).to_have_text("Freshly created entry")
    uploads = backend.calls("POST", "/api/entries/upload")
    assert len(uploads) == 1, backend.requests
    assert "opsuser" in (uploads[0].body or "")
    assert "plot.png" in (uploads[0].body or "")
    page.close()


def test_req5_cancel_returns_to_the_list(ariel_url: str, chromium_browser: Browser) -> None:
    """Cancel hides the form and leaves the list in place."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")
    page.click("#ariel-compose-btn")
    expect(page.locator("#ariel-compose")).to_be_visible()
    page.click("#create-cancel-btn")
    expect(page.locator("#ariel-compose")).to_be_hidden()
    expect(row(page, "E1")).to_be_visible()
    page.close()


def test_req5_draft_review_prefills_the_form(ariel_url: str, chromium_browser: Browser) -> None:
    """``#create?draft=D1`` opens the form prefilled from the draft, with its banner."""
    page, backend = open_panel(chromium_browser, f"{ariel_url}/#create?draft=D1")

    expect(page.locator("#ariel-compose")).to_be_visible()
    expect(page.locator("#entry-subject")).to_have_value("Drafted by the agent")
    expect(page.locator("#entry-details")).to_have_value("Agent-written details.")
    expect(page.locator("#draft-banner")).to_have_attribute("data-draft-id", "D1")
    page.click("#create-entry-form button[type=submit]")
    expect(reader_title(page)).to_have_text("Freshly created entry")
    assert backend.calls("POST", "/api/entries")[-1].json()["subject"] == "Drafted by the agent"
    page.close()


# ---------------------------------------------------------------------------
# Requirement 6: per-frame persistence
# ---------------------------------------------------------------------------


def _storage_keys(page: Page, prefix: str) -> list[str]:
    return page.evaluate(
        "(p) => Object.keys(localStorage).filter((k) => k.startsWith(p)).sort()", prefix
    )


def test_req6_pane_sizes_survive_reload_per_frame_kind(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Standalone and embedded each restore their own sizes and never overwrite each other."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")

    drag(page, "#ariel-rail-splitter", 80)  # 208 -> ~288
    drag(page, "#ariel-list-splitter", 60)
    rail_w, list_w = width(page, "#ariel-rail"), width(page, "#ariel-list-col")
    assert rail_w > 260, rail_w
    assert all("embedded" not in k for k in _storage_keys(page, "ariel-"))

    page.reload(wait_until="load")
    expect(page.locator("#ariel-list-body")).to_have_attribute("aria-busy", "false")
    assert abs(width(page, "#ariel-rail") - rail_w) <= 2
    assert abs(width(page, "#ariel-list-col") - list_w) <= 2

    # Embedded: its own default (collapsed strip), then its own remembered state.
    page.goto(f"{ariel_url}/?embedded=true", wait_until="load")
    expect(app(page)).to_have_attribute("data-rail-collapsed", "")
    assert width(page, "#ariel-rail") < 60
    page.keyboard.press("[")
    expect(app(page)).not_to_have_attribute("data-rail-collapsed", re.compile(".*"))
    page.reload(wait_until="load")
    expect(app(page)).not_to_have_attribute("data-rail-collapsed", re.compile(".*"))
    assert any("ariel-rail-embedded" in k for k in _storage_keys(page, "ariel-rail-"))

    # Collapsing embedded leaves the standalone rail as it was.
    page.keyboard.press("[")
    expect(app(page)).to_have_attribute("data-rail-collapsed", "")
    page.goto(f"{ariel_url}/", wait_until="load")
    expect(app(page)).not_to_have_attribute("data-rail-collapsed", re.compile(".*"))
    assert abs(width(page, "#ariel-rail") - rail_w) <= 2
    page.close()


# ---------------------------------------------------------------------------
# Requirement 7: narrow stacking with Back
# ---------------------------------------------------------------------------


def test_req7_narrow_panel_stacks_and_back_returns_to_the_list(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Below ~640 px opening a row replaces the list; Back (and Esc) bring it back."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", viewport={"width": 560, "height": 800})

    expect(app(page)).to_have_attribute("data-stacked", "")
    expect(page.locator("#ariel-list")).to_be_visible()
    expect(page.locator("#ariel-reader")).to_be_hidden()

    row(page, "E1").click()
    expect(app(page)).to_have_attribute("data-narrow-view", "reader")
    expect(page.locator("#ariel-reader")).to_be_visible()
    expect(page.locator("#ariel-list")).to_be_hidden()
    expect(reader_title(page)).to_have_text("Beam dump at injection")

    page.click("#ariel-reader-back")
    expect(app(page)).to_have_attribute("data-narrow-view", "list")
    expect(page.locator("#ariel-list")).to_be_visible()
    expect(page.locator("#ariel-reader")).to_be_hidden()

    # Widening un-stacks: both panes show side by side.
    page.set_viewport_size({"width": 1280, "height": 800})
    expect(app(page)).not_to_have_attribute("data-stacked", re.compile(".*"))
    expect(page.locator("#ariel-reader")).to_be_visible()
    page.close()


# ---------------------------------------------------------------------------
# Mailboxes
# ---------------------------------------------------------------------------


def _box(page: Page, box: str):
    return page.locator(f'#ariel-rail button[data-box="{box}"]')


def test_mailbox_all_is_current_by_default(ariel_url: str, chromium_browser: Browser) -> None:
    """All entries is the selected mailbox on landing and lists everything."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")
    expect(_box(page, "all")).to_have_attribute("aria-current", "page")
    expect(page.locator("#ariel-rail [aria-current]")).to_have_count(1)
    expect(page.locator("#ariel-list [data-entry-id]")).to_have_count(3)
    page.close()


def test_mailbox_drafts_shows_its_empty_state(ariel_url: str, chromium_browser: Browser) -> None:
    """Drafts has no listing yet: it says agent drafts open from the agent's link."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")
    _box(page, "drafts").click()
    expect(_box(page, "drafts")).to_have_attribute("aria-current", "page")
    expect(page.locator("#ariel-list-body")).to_contain_text("No Draft Listing Yet")
    expect(page.locator("#ariel-list-body")).to_contain_text("open from the agent's link")
    expect(page.locator("#ariel-list [data-entry-id]")).to_have_count(0)
    wait_for_hash(page, canonical(box="drafts"))

    _box(page, "all").click()
    expect(page.locator("#ariel-list [data-entry-id]")).to_have_count(3)
    page.close()


def test_mailbox_mine_asks_for_the_name_then_scopes_the_list(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Without a known name Mine asks for one; then the listing is that author's."""
    backend = FakeBackend()
    backend.entries.append(make_entry("E4", "Alice's entry", author="alice"))
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", backend)

    _box(page, "mine").click()
    field_ = page.locator('[data-rail-field="mine"]')
    expect(field_).to_be_visible()
    expect(_box(page, "all")).to_have_attribute("aria-current", "page")
    field_.fill("alice")
    field_.press("Enter")

    expect(_box(page, "mine")).to_have_attribute("aria-current", "page")
    expect(page.locator("#ariel-list [data-entry-id]")).to_have_count(1)
    expect(row(page, "E4")).to_be_visible()
    wait_for_hash(page, canonical(box="mine"))
    assert backend.calls("GET", "/api/entries")[-1].query.get("author") == ["alice"]

    # The name is remembered: a reload on box=mine lists alice's entries directly.
    page.reload(wait_until="load")
    expect(row(page, "E4")).to_be_visible()
    expect(field_).to_be_hidden()
    page.close()


def test_mailbox_saved_search_saves_runs_and_removes(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """A saved search keeps query + mode under a name, reruns them, and can be removed."""
    page, backend = open_panel(chromium_browser, f"{ariel_url}/")

    page.fill("#ariel-search-input", "beam")
    page.press("#ariel-search-input", "Enter")
    expect(page.locator("#ariel-list .results-count")).to_be_visible()

    page.click('#ariel-rail [data-rail-action="save-search"]')
    name = page.locator('[data-rail-field="save"]')
    expect(name).to_be_visible()
    name.fill("Beam")
    name.press("Enter")
    saved = _box(page, "saved:Beam")
    expect(saved).to_be_visible()

    # Back to the plain listing, then rerun the saved search.
    page.fill("#ariel-search-input", "")
    page.press("#ariel-search-input", "Enter")
    expect(page.locator("#ariel-list .ariel-day-sep").first).to_be_visible()
    before = len(backend.calls("POST", "/api/search"))
    saved.click()
    expect(page.locator("#ariel-search-input")).to_have_value("beam")
    expect(saved).to_have_attribute("aria-current", "page")
    wait_for_hash(page, canonical(q="beam", box="saved:Beam"))
    expect(page.locator("#ariel-list .results-count")).to_be_visible()
    assert len(backend.calls("POST", "/api/search")) > before

    # Survives a reload (browser storage), then removal drops it and returns to All.
    page.reload(wait_until="load")
    expect(saved).to_be_visible()
    page.click('#ariel-rail [data-remove-saved="Beam"]')
    expect(saved).to_have_count(0)
    expect(_box(page, "all")).to_have_attribute("aria-current", "page")
    page.close()


# ---------------------------------------------------------------------------
# Rail: drag, snap, toggle, default per frame
# ---------------------------------------------------------------------------


def test_rail_default_state_per_frame(ariel_url: str, chromium_browser: Browser) -> None:
    """Expanded standalone, an icon strip embedded."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")
    expect(app(page)).not_to_have_attribute("data-rail-collapsed", re.compile(".*"))
    expect(page.locator("#ariel-rail .osprey-nav-label").first).to_be_visible()
    assert width(page, "#ariel-rail") > 150
    page.close()

    page, _ = open_panel(chromium_browser, f"{ariel_url}/?embedded=true")
    expect(app(page)).to_have_attribute("data-rail-collapsed", "")
    assert width(page, "#ariel-rail") < 60
    page.close()


def test_rail_drag_resizes_and_snaps_to_the_icon_strip(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Dragging the border resizes; releasing below ~110 px snaps to the strip."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")
    start = width(page, "#ariel-rail")

    drag(page, "#ariel-rail-splitter", 60)
    expect(app(page)).not_to_have_attribute("data-rail-collapsed", re.compile(".*"))
    grown = width(page, "#ariel-rail")
    assert abs(grown - (start + 60)) <= 3, (start, grown)

    drag(page, "#ariel-rail-splitter", -(grown - 90))  # release at ~90 px
    expect(app(page)).to_have_attribute("data-rail-collapsed", "")
    assert width(page, "#ariel-rail") < 60

    # Expanding returns a usable rail, the last width before the snap.
    page.dblclick("#ariel-rail-splitter")
    expect(app(page)).not_to_have_attribute("data-rail-collapsed", re.compile(".*"))
    assert abs(width(page, "#ariel-rail") - grown) <= 3
    page.close()


def test_rail_double_click_enter_and_bracket_toggle(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Double-click and Enter on the border, and ``[``, each toggle the rail."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")
    collapsed = app(page)

    page.dblclick("#ariel-rail-splitter")
    expect(collapsed).to_have_attribute("data-rail-collapsed", "")
    page.focus("#ariel-rail-splitter")
    page.keyboard.press("Enter")
    expect(collapsed).not_to_have_attribute("data-rail-collapsed", re.compile(".*"))

    page.locator("#ariel-reader-body").click(position={"x": 5, "y": 5})
    page.keyboard.press("[")
    expect(collapsed).to_have_attribute("data-rail-collapsed", "")
    page.keyboard.press("[")
    expect(collapsed).not_to_have_attribute("data-rail-collapsed", re.compile(".*"))
    page.close()


# ---------------------------------------------------------------------------
# Keyboard table
# ---------------------------------------------------------------------------


def test_key_j_and_k_move_the_selection_and_the_reader_follows(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """``j`` selects down from the top, ``k`` back up; the reader follows each move."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")

    page.keyboard.press("j")
    expect(row(page, "E1")).to_have_attribute("aria-selected", "true")
    expect(reader_title(page)).to_have_text("Beam dump at injection")
    page.keyboard.press("j")
    page.keyboard.press("j")
    expect(row(page, "E3")).to_have_attribute("aria-selected", "true")
    expect(reader_title(page)).to_have_text("Shift handover notes")
    page.keyboard.press("j")  # clamps at the bottom
    expect(row(page, "E3")).to_have_attribute("aria-selected", "true")
    page.keyboard.press("k")
    expect(row(page, "E2")).to_have_attribute("aria-selected", "true")
    expect(reader_title(page)).to_have_text("Vacuum spike sector 3")

    # Typing in the search box is not a shortcut.
    page.focus("#ariel-search-input")
    page.keyboard.press("j")
    expect(row(page, "E2")).to_have_attribute("aria-selected", "true")
    expect(page.locator("#ariel-search-input")).to_have_value("j")
    page.close()


def test_key_enter_opens_the_selection_in_the_reader(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Stacked, ``j`` selects without leaving the list; ``Enter`` opens the reader."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", viewport={"width": 560, "height": 800})

    page.keyboard.press("j")
    expect(row(page, "E1")).to_have_attribute("aria-selected", "true")
    expect(app(page)).to_have_attribute("data-narrow-view", "list")
    page.keyboard.press("Enter")
    expect(app(page)).to_have_attribute("data-narrow-view", "reader")
    expect(reader_title(page)).to_have_text("Beam dump at injection")
    page.close()


def test_key_slash_focuses_the_search_box(ariel_url: str, chromium_browser: Browser) -> None:
    """``/`` puts the cursor in the standalone search box (the key itself is not typed)."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")
    page.keyboard.press("/")
    expect(page.locator("#ariel-search-input")).to_be_focused()
    expect(page.locator("#ariel-search-input")).to_have_value("")
    page.close()


def test_key_bracket_toggles_the_rail(ariel_url: str, chromium_browser: Browser) -> None:
    """``[`` collapses and expands the rail."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")
    page.keyboard.press("[")
    expect(app(page)).to_have_attribute("data-rail-collapsed", "")
    page.keyboard.press("[")
    expect(app(page)).not_to_have_attribute("data-rail-collapsed", re.compile(".*"))
    page.close()


def test_key_escape_closes_one_layer_at_a_time(ariel_url: str, chromium_browser: Browser) -> None:
    """Esc closes the lightbox, then the popover, then compose, then goes Back when stacked."""
    page, _ = open_panel(
        chromium_browser,
        f"{ariel_url}/#entry?id=E1&attachment=A1",
        viewport={"width": 560, "height": 800},
    )
    expect(page.locator("#image-lightbox")).to_be_visible()
    expect(app(page)).to_have_attribute("data-narrow-view", "reader")

    page.keyboard.press("Escape")
    expect(page.locator("#image-lightbox")).to_have_count(0)
    wait_for_hash(page, canonical(sel="E1"))
    expect(app(page)).to_have_attribute("data-narrow-view", "reader")  # only one layer

    page.keyboard.press("Escape")
    expect(app(page)).to_have_attribute("data-narrow-view", "list")
    page.close()

    page, _ = open_panel(chromium_browser, f"{ariel_url}/")
    page.click("#ariel-mode-chip")
    expect(page.locator("#ariel-filters-popover")).to_be_visible()
    page.keyboard.press("Escape")
    expect(page.locator("#ariel-filters-popover")).to_be_hidden()

    page.click("#ariel-compose-btn")
    expect(page.locator("#ariel-compose")).to_be_visible()
    page.focus("#entry-subject")
    page.keyboard.press("Escape")  # Esc works while typing
    expect(page.locator("#ariel-compose")).to_be_hidden()
    page.close()


# ---------------------------------------------------------------------------
# Legacy hash aliases, standalone
# ---------------------------------------------------------------------------


def test_alias_entry_selects_and_opens_the_attachment(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """``#entry?id=X&attachment=Y`` selects X and opens Y; the hash goes canonical."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/#entry?id=E1&attachment=A1")
    expect(reader_title(page)).to_have_text("Beam dump at injection")
    expect(row(page, "E1")).to_have_attribute("aria-selected", "true")
    expect(page.locator("#image-lightbox")).to_be_visible()
    expect(page.locator("#image-lightbox [data-use-in-workspace]")).to_be_visible()
    wait_for_hash(page, canonical(sel="E1", att="A1"))
    page.close()


@pytest.mark.parametrize("alias", ["#create", "#create?draft=D1"])
def test_alias_create_opens_compose(ariel_url: str, chromium_browser: Browser, alias: str) -> None:
    """``#create`` opens the empty form; ``?draft=`` prefills it."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/{alias}")
    expect(page.locator("#ariel-compose")).to_be_visible()
    expected = "Drafted by the agent" if "draft" in alias else ""
    expect(page.locator("#entry-subject")).to_have_value(expected)
    wait_for_hash(page, canonical())
    page.close()


@pytest.mark.parametrize("alias", ["#search", "#browse"])
def test_alias_search_and_browse_show_the_list(
    ariel_url: str, chromium_browser: Browser, alias: str
) -> None:
    """``#search`` and ``#browse`` land on the plain list, whatever was open before."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/#q=beam&sel=E1")
    expect(reader_title(page)).to_be_visible()
    page.evaluate("(a) => { location.hash = a; }", alias)
    expect(page.locator("#ariel-search-input")).to_have_value("")
    expect(page.locator("#ariel-list [data-entry-id]")).to_have_count(3)
    expect(page.locator("#ariel-reader-body")).to_be_empty()
    wait_for_hash(page, canonical())
    page.close()


def test_alias_status_opens_the_status_drawer(ariel_url: str, chromium_browser: Browser) -> None:
    """Standalone ``#status`` opens the Status drawer over the list."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/#status")
    expect(page.locator("osprey-drawer#status-drawer")).to_have_attribute("open", re.compile(".*"))
    expect(page.locator("#ariel-list")).to_be_attached()
    page.close()


def test_repeated_alias_reopens_a_dismissed_lightbox(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Following the same ``#entry`` link again reopens the picture the user closed."""
    link = "#entry?id=E1&attachment=A1"
    page, _ = open_panel(chromium_browser, f"{ariel_url}/{link}")
    lightbox = page.locator("#image-lightbox")
    expect(lightbox).to_be_visible()

    lightbox.click(position={"x": 5, "y": 5})  # the scrim, not the picture
    expect(lightbox).to_have_count(0)
    wait_for_hash(page, canonical(sel="E1"))

    page.evaluate("(a) => { location.hash = a; }", link)
    expect(lightbox).to_be_visible()
    page.close()


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


def test_panes_use_the_shared_primitives(ariel_url: str, chromium_browser: Browser) -> None:
    """Rows, pane heads, the search box, rail rows and popovers are design-system classes."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")

    expect(page.locator("#ariel-list [data-entry-id]")).to_have_count(3)
    expect(page.locator("#ariel-list [data-entry-id].osprey-list-row")).to_have_count(3)
    expect(page.locator("#ariel-list .osprey-list-row .osprey-list-title").first).to_be_visible()
    expect(page.locator("#ariel-list .osprey-pane-head")).to_have_count(1)
    expect(page.locator("#ariel-reader > .osprey-pane-head")).to_have_count(1)
    expect(page.locator("#ariel-compose .osprey-pane-head")).to_have_count(1)
    expect(page.locator("label.osprey-search #ariel-search-input")).to_have_count(1)
    expect(page.locator("#ariel-rail button[data-box].osprey-nav-row")).to_have_count(3)
    expect(page.locator("#ariel-filters-popover.osprey-popover")).to_have_count(1)
    expect(page.locator("#ariel-gear-menu.osprey-popover")).to_have_count(1)

    page.click("#ariel-gear-btn")
    expect(page.locator("#ariel-gear-menu")).to_be_visible()
    expect(page.locator("#ariel-gear-menu .osprey-popover-item")).not_to_have_count(0)
    page.close()


# ---------------------------------------------------------------------------
# The list as a mail client: rows hold their content, on real-sized logbook text
# ---------------------------------------------------------------------------

#: A logbook paragraph far longer than the column, as every seeded entry body is.
LONG_BODY = " ".join(["Replaced the thermionic cathode after 12,000 hours of operation."] * 12)


def _long_entries() -> list[dict[str, Any]]:
    """Entries whose subject and body cannot fit one line of the list column."""
    return [
        {
            **make_entry(
                f"L{n}",
                f"Long subject line {n} that keeps going well past the width of the list column",
                author="An Operator With A Rather Long Name",
                timestamp=f"2026-08-20T{12 - n:02d}:00:00-07:00",
            ),
            "raw_text": f"Long subject line {n} that keeps going well past the list\n{LONG_BODY}",
        }
        for n in range(6)
    ]


_ROW_GEOMETRY = """() => {
  const col = document.querySelector('#ariel-list').getBoundingClientRect();
  return [...document.querySelectorAll('#ariel-list [data-entry-id]')].map((row) => {
    const box = (el) => {
      const r = el.getBoundingClientRect();
      return { left: r.left, right: r.right, top: r.top, bottom: r.bottom };
    };
    return {
      col: { left: col.left, right: col.right },
      row: box(row),
      subject: box(row.querySelector('.ariel-row-subject')),
      time: box(row.querySelector('.ariel-row-time')),
      author: box(row.querySelector('.ariel-row-author')),
      preview: box(row.querySelector('.ariel-row-preview')),
      previewLine: parseFloat(getComputedStyle(row.querySelector('.ariel-row-preview')).lineHeight),
      contents: [...row.querySelectorAll('*')].map(box),
    };
  });
}"""


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_list_rows_contain_their_content_inside_the_list_column(
    ariel_url: str, chromium_browser: Browser, theme: str
) -> None:
    """Subject + time, then author, then one clamped preview line, all inside the column."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", FakeBackend(entries=_long_entries()))
    page.evaluate("t => { document.documentElement.dataset.theme = t; }", f"desy-{theme}")
    expect(page.locator("#ariel-list [data-entry-id]")).to_have_count(6)

    rows = page.evaluate(_ROW_GEOMETRY)
    tolerance = 0.5
    for r in rows:
        assert r["row"]["left"] >= r["col"]["left"] - tolerance
        assert r["row"]["right"] <= r["col"]["right"] + tolerance
        for child in r["contents"]:
            assert child["right"] <= r["row"]["right"] + tolerance, "content leaves the row"
            assert child["bottom"] <= r["row"]["bottom"] + tolerance, "content leaves the row"
            assert child["top"] >= r["row"]["top"] - tolerance, "content leaves the row"
        assert r["subject"]["right"] <= r["time"]["left"] + tolerance, "subject runs under time"
        assert r["subject"]["top"] < r["time"]["bottom"], "time is not on the subject's line"
        assert r["time"]["top"] < r["subject"]["bottom"], "time is not on the subject's line"
        assert r["author"]["top"] >= r["subject"]["bottom"] - tolerance
        assert r["preview"]["top"] >= r["author"]["bottom"] - tolerance
        height = r["preview"]["bottom"] - r["preview"]["top"]
        assert height <= r["previewLine"] + 2, "preview is not clamped to one line"
    for upper, lower in zip(rows, rows[1:], strict=False):
        assert upper["row"]["bottom"] <= lower["row"]["top"] + tolerance, "rows overlap"
    page.close()


def test_list_head_controls_do_not_collide(ariel_url: str, chromium_browser: Browser) -> None:
    """Search box, mode control and compose sit side by side inside the pane head."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", FakeBackend(entries=_long_entries()))

    head = page.locator("#ariel-list .osprey-pane-head").bounding_box()
    boxes = [
        page.locator(sel).bounding_box()
        for sel in ("label.osprey-search", "#ariel-mode-chip", "#ariel-compose-btn")
    ]
    assert page.locator("#ariel-mode-chip.osprey-btn").count() == 1
    for box in boxes:
        assert box["y"] >= head["y"] and box["y"] + box["height"] <= head["y"] + head["height"]
    for left, right in zip(boxes, boxes[1:], strict=False):
        assert left["x"] + left["width"] <= right["x"], "head controls overlap"
    assert boxes[-1]["x"] + boxes[-1]["width"] <= head["x"] + head["width"]
    page.close()


def test_clicking_a_row_of_long_entries_opens_the_reader(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Every row takes its own click: no neighbour's text lies over it."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", FakeBackend(entries=_long_entries()))

    row(page, "L3").click(timeout=3000)
    expect(row(page, "L3")).to_have_attribute("aria-selected", "true")
    expect(page.locator("#ariel-reader-body .entry-detail")).to_be_visible()
    wait_for_hash(page, canonical(sel="L3"))
    page.close()


def _seeded_entries() -> list[dict[str, Any]]:
    """The tutorial logbook as the list receives it, newest first, each a day apart."""
    seed = (
        Path(__file__).resolve().parents[3]
        / "src/osprey/templates/facilities/example/scenarios/nominal.yaml"
    )
    logbook = yaml.safe_load(seed.read_text(encoding="utf-8"))["logbook"]
    logbook = sorted(logbook, key=lambda e: e["entry_id"], reverse=True)
    return [
        {
            **make_entry(
                e["entry_id"],
                e["title"],
                author=e["author"],
                timestamp=f"2026-08-{28 - n // 3:02d}T{20 - n % 3:02d}:00:00-07:00",
            ),
            "source_system": "Simulation",
            "raw_text": f"{e['title']}\n{e['text']}",
        }
        for n, e in enumerate(logbook)
    ]


def test_every_row_of_the_seeded_logbook_takes_its_own_click(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """On real rows the click lands on the row asked for, and the reader follows."""
    entries = _seeded_entries()
    page, _ = open_panel(
        chromium_browser,
        f"{ariel_url}/",
        FakeBackend(entries=entries),
        viewport={"width": 1280, "height": 900},
    )
    expect(page.locator("#ariel-list [data-entry-id]")).to_have_count(len(entries))

    for entry in entries[:8]:
        target = row(page, entry["entry_id"])
        target.click(timeout=3000)
        expect(target).to_have_attribute("aria-selected", "true")
        expect(reader_title(page)).to_contain_text(entry["raw_text"].split("\n")[0])
    page.close()


def test_gear_does_not_repeat_the_display_menus_sliders(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """The display menu trigger draws ``sliders``; the gear draws a gear."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")

    gear = page.locator("#ariel-gear-btn svg").first.inner_html()
    assert "M1 14h6" not in gear, "the gear draws the sliders icon"
    assert "circle" in gear
    page.close()


def test_brand_link_goes_to_the_canonical_empty_hash(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """The logo points at the bare screen, not at a legacy alias."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")

    assert page.locator("header a.logo").get_attribute("href") == "#"
    page.close()
