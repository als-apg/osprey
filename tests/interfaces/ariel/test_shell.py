"""Real-browser suite for ARIEL's one screen: rail | list | reader.

Covers what only a live page proves: the numbered Requirements of the
one-screen plan (the landing, selection in the hash and its reload, deep links
and their error states, search surfaces, create and draft review, persistence,
the stacked layout), each rail mailbox, the rail's drag / snap / toggle, each
key of the keyboard table, every legacy hash alias (``test_embedded_nav.py``
holds the tile-bar contract), and the shared design-system primitives the panes
are built from.

The panel is the real ``create_app()`` on a real server (``launch_ariel``),
framed with ``embedded=true`` by the host fixture page (``_panel_host.py``);
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
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, quote, urlparse
from zoneinfo import ZoneInfo

import pytest
import yaml

from tests.interfaces._browser import wait_for_page_ready
from tests.interfaces._panel_host import PanelHost, open_panel_host

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
    """One DB-less ARIEL for the whole module, as ``launch_ariel`` boots it."""
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
    # The zone the fixture timestamps are written in. Without it the day
    # separators follow the browser's zone, and these fixtures' two days are
    # one day in UTC.
    "facility_timezone": "America/Los_Angeles",
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
    "filter_dimensions": [
        {"name": "author", "label": "Author", "kind": "builtin"},
        {"name": "source_system", "label": "Source", "kind": "builtin"},
        {"name": "tags", "label": "Tags", "kind": "builtin"},
        {"name": "enhancement", "label": "Enhancements", "kind": "enhancement"},
    ],
    "enhancement_modules": [
        {"name": "text_embedding", "label": "Text Embedding"},
        {"name": "figure_caption", "label": "Figure Caption"},
    ],
}

#: The facility zone the fake buckets days in (the fixtures' ``-07:00``).
FACILITY_ZONE = ZoneInfo("America/Los_Angeles")

INVALID_CAPABILITIES: dict[str, Any] = {
    **HEALTHY_CAPABILITIES,
    "status": "configuration_invalid",
    "config_errors": ["ariel.vocabulary.path: file not found"],
    "remedy": "repoint ariel.vocabulary.path, then restart",
}

#: A ``StatusResponse`` with coverage for every enabled module, in execution order.
HEALTHY_STATUS: dict[str, Any] = {
    "healthy": True,
    "database_connected": True,
    "database_uri": "postgresql://localhost/ariel",
    "entry_count": 1234,
    "embedding_tables": [],
    "active_embedding_model": "all-minilm",
    "enabled_search_modules": ["keyword", "semantic"],
    "enabled_enhancement_modules": ["text_embedding", "figure_caption"],
    "last_ingestion": None,
    "errors": [],
    "enhancement_coverage": {
        "text_embedding": {
            "label": "Text Embedding",
            "complete": 1200,
            "not_run": 7,
            "pending": 30,
            "stale": 3,
            "failed": 1,
            "n/a": 0,
        },
        "figure_caption": {
            "label": "Figure Caption",
            "complete": 400,
            "not_run": 0,
            "pending": 0,
            "stale": 0,
            "failed": 2,
            "n/a": 832,
        },
    },
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
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """An ``EntryResponse``-shaped entry (``subject`` is filled, as the server fills it)."""
    return {
        "entry_id": entry_id,
        "source_system": "olog",
        "timestamp": timestamp,
        "author": author,
        "subject": subject,
        "raw_text": f"{subject}\nBody of {entry_id}.",
        "attachments": attachments or [],
        "metadata": dict(metadata or {}),
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
    # Recorded provenance: the form shows the draft banner only for this.
    "metadata": {"session_metadata": {"operator": "agent", "created_via": "osprey"}},
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


def facility_day(entry: dict[str, Any]) -> str:
    """The facility-zone day (``YYYY-MM-DD``) of *entry*'s timestamp."""
    stamp = datetime.fromisoformat(entry["timestamp"])
    return stamp.astimezone(FACILITY_ZONE).date().isoformat()


def dimension_values(entry: dict[str, Any], dim: str) -> list[str]:
    """*entry*'s values for filter dimension *dim* (C-FILTER)."""
    meta = entry.get("metadata") or {}
    if dim in ("author", "source_system"):
        value = entry.get(dim)
    elif dim == "enhancement":
        states = meta.get("enhancements") or {}
        return [f"{module}:{state}" for module, state in states.items()]
    else:
        value = meta.get(dim)
    if value is None:
        return []
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]


def matches_filter(entry: dict[str, Any], filt: dict[str, Any], skip: str | None = None) -> bool:
    """C-FILTER: OR within a dimension, AND across, exact and case-insensitive.

    ``time`` is an inclusive ``{start, end}`` window of facility days; *skip*
    leaves one dimension out (a facet's self-exclusion).
    """
    for dim, wanted in filt.items():
        if dim == skip:
            continue
        if dim == "time":
            day = facility_day(entry)
            start, end = (wanted or {}).get("start"), (wanted or {}).get("end")
            if (start and day < start[:10]) or (end and day > end[:10]):
                return False
            continue
        want = {str(v).lower() for v in wanted or []}
        if want and not any(v.lower() in want for v in dimension_values(entry, dim)):
            return False
    return True


def to_row(entry: dict[str, Any]) -> dict[str, Any]:
    """An ``EntryResponse``-shaped entry as its C-ROW list row (``view=row``)."""
    if "raw_text" not in entry:
        return entry
    subject, _, rest = entry["raw_text"].partition("\n")
    attachments = entry.get("attachments") or []
    meta = entry.get("metadata") or {}
    return {
        "entry_id": entry["entry_id"],
        "timestamp": entry["timestamp"],
        "author": entry["author"],
        "source_system": entry["source_system"],
        "subject": subject,
        "snippet": rest[:300],
        "facets": {"author": entry["author"], "source_system": entry["source_system"]},
        "attachment_count": len(attachments),
        "has_images": any(str(a.get("mime_type", "")).startswith("image/") for a in attachments),
        "enhancements": dict(meta.get("enhancements") or {}),
        "entry_url": None,
        "score": entry.get("score"),
        "highlights": entry.get("highlights") or [],
        "keywords": entry.get("keywords") or [],
        "matched_via": entry.get("matched_via") or [],
    }


def facets_response(
    entries: list[dict[str, Any]], filt: dict[str, Any], dimensions: list[dict[str, Any]]
) -> dict[str, Any]:
    """C-FACETS counts over *entries*: each dimension under the filter minus itself,
    values grouped case-insensitively, sorted by count and capped at 50; the day
    histogram under the filter minus ``time``."""
    out = []
    for dim in dimensions:
        name = dim["name"]
        counts: dict[str, list[Any]] = {}
        for entry in entries:
            if not matches_filter(entry, filt, skip=name):
                continue
            for value in dimension_values(entry, name):
                if name == "enhancement" and value.rsplit(":", 1)[-1] not in ("failed", "stale"):
                    continue
                counts.setdefault(value.lower(), [value, 0])[1] += 1
        values = sorted(counts.values(), key=lambda vc: (-vc[1], vc[0]))[:50]
        out.append(
            {
                "name": name,
                "label": dim.get("label", name),
                "kind": dim.get("kind", "builtin"),
                "values": [{"value": v, "count": c} for v, c in values],
            }
        )
    days: dict[str, int] = {}
    for entry in entries:
        if matches_filter(entry, filt, skip="time"):
            day = facility_day(entry)
            days[day] = days.get(day, 0) + 1
    return {
        "available": True,
        "dimensions": out,
        "histogram": [{"day": d, "count": c} for d, c in sorted(days.items())],
        "exact": True,
        "tier": None,
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

    def filters(self) -> dict[str, Any]:
        """The C-FILTER the request carries (GET ``filters`` JSON or POST body), else ``{}``."""
        if self.method == "GET":
            raw = self.query.get("filters", [None])[0]
            return json.loads(raw) if raw else {}
        return (self.json() or {}).get("filters") or {}


@dataclass
class FakeBackend:
    """Answers the screen's ``/api`` requests from fixtures and records them.

    Listings honour the request's ``filters`` (C-FILTER), keyset ``cursor`` /
    ``page_size`` (browse) and ``offset`` / ``max_results`` (search, over the
    ``search`` fixture's entries), and serve C-ROW rows for ``view=row``;
    ``/api/facets`` counts the same entries. ``facets_status`` other than 200
    makes the facets route fail. ``/api/status``, ``/enhancements`` and
    ``/related`` answer from the ``status``, ``enhancements`` and ``related``
    fixtures. Anything it does not know (config) falls through to the real
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
    facets_status: int = 200
    #: ``GET /api/entries/{id}/enhancements`` bodies by entry id; absent ids get no cards.
    enhancements: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: ``GET /api/entries/{id}/related`` candidates by entry id, nearest first.
    related: dict[str, list[str]] = field(default_factory=dict)
    #: The ``/related`` state; ``disabled`` is text embedding switched off.
    related_state: str = "ok"
    #: The ``GET /api/status`` body.
    status: dict[str, Any] = field(default_factory=lambda: dict(HEALTHY_STATUS))

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
            ok(self._search(record))
        elif path == "/api/entries" and req.method == "GET":
            ok(self._browse(record))
        elif path == "/api/facets" and req.method == "GET":
            if self.facets_status != 200:
                ok({"detail": "facets unavailable"}, self.facets_status)
                return
            query = record.query.get("q", [""])[0].strip()
            pool = self.search["entries"] if query else self.entries
            raw = record.query.get("filters", [None])[0]
            dims = [
                d for d in self.capabilities.get("filter_dimensions", []) if d["name"] != "time"
            ]
            ok(facets_response(pool, json.loads(raw) if raw else {}, dims))
        elif path == "/api/filter-options/authors":
            authors = sorted({e["author"] for e in self.entries})
            ok({"field": "authors", "options": [{"value": a, "label": a} for a in authors]})
        elif path in ("/api/entries", "/api/entries/upload") and req.method == "POST":
            self.created += 1
            new_id = f"NEW-{self.created}"
            self.entries.insert(0, make_entry(new_id, "Freshly created entry"))
            ok({"entry_id": new_id, "message": "created"}, status=201)
        elif path == "/api/status" and req.method == "GET":
            ok(self.status)
        elif path.startswith("/api/entries/") and path.endswith("/enhancements"):
            entry_id = path.split("/")[3]
            if entry_id not in self.details():
                ok({"detail": "Entry not found"}, 404)
                return
            ok(self.enhancements.get(entry_id, {"entry_id": entry_id, "modules": []}))
        elif path.startswith("/api/entries/") and path.endswith("/related"):
            limit = int(record.query.get("limit", ["5"])[0])
            ok(self._related(path.split("/")[3], limit))
        elif path.startswith("/api/entries/") and path.endswith("/thread"):
            thread = self._thread(path.split("/")[3])
            ok(thread, 200) if thread else ok({"detail": "Entry not found"}, 404)
        elif path.startswith("/api/entries/"):
            entry = self.details().get(path.rsplit("/", 1)[1])
            ok(entry, 200) if entry else ok({"detail": "Entry not found"}, 404)
        elif path == "/api/drafts" and req.method == "GET":
            ok({"drafts": self._draft_summaries()})
        elif path.startswith("/api/drafts/") and req.method == "DELETE":
            gone = self.drafts.pop(path.rsplit("/", 1)[1], None)
            route.fulfill(status=204) if gone else ok({"detail": "Draft not found"}, 404)
        elif path.startswith("/api/drafts/"):
            draft = self.drafts.get(path.rsplit("/", 1)[1])
            ok(draft, 200) if draft else ok({"detail": "Draft not found"}, 404)
        else:
            route.continue_()

    def _thread(self, entry_id: str) -> dict[str, Any] | None:
        """GET /api/entries/{id}/thread (C-THREAD): the parent row and the children, oldest first."""
        details = self.details()
        entry = details.get(entry_id)
        if entry is None:
            return None
        parent_id = (entry.get("metadata") or {}).get("follows_up")
        parent = details.get(parent_id) if parent_id else None
        children = sorted(
            (
                e
                for e in details.values()
                if (e.get("metadata") or {}).get("follows_up") == entry_id
            ),
            key=lambda e: datetime.fromisoformat(e["timestamp"]),
        )
        return {
            "parent_id": parent_id,
            "parent": to_row(parent) if parent else None,
            "children": [to_row(e) for e in children],
        }

    def _related(self, entry_id: str, limit: int) -> dict[str, Any]:
        """GET /api/entries/{id}/related (RelatedResponse): at most *limit* rows, never self."""
        if self.related_state != "ok":
            return {"state": self.related_state, "entries": []}
        details = self.details()
        ids = [i for i in self.related.get(entry_id, []) if i != entry_id][:limit]
        return {"state": "ok", "entries": [to_row(details[i]) for i in ids]}

    def _draft_summaries(self) -> list[dict[str, Any]]:
        """GET /api/drafts: one ``DraftSummary`` per held draft, newest first."""
        summaries = [
            {
                "draft_id": draft_id,
                "subject": draft.get("subject", ""),
                "author": draft.get("author"),
                "logbook": draft.get("logbook"),
                "created_at": draft.get("created_at", "2026-08-20T10:00:00-07:00"),
                "staged_by": ((draft.get("metadata") or {}).get("session_metadata") or {}).get(
                    "operator"
                ),
                "attachment_count": len(draft.get("attachment_paths") or []),
            }
            for draft_id, draft in self.drafts.items()
        ]
        return sorted(
            summaries, key=lambda d: datetime.fromisoformat(d["created_at"]), reverse=True
        )

    def _browse(self, record: Request) -> dict[str, Any]:
        """GET /api/entries: filtered, keyset-paged (the cursor is an offset), C-ROW on request."""
        query = record.query
        filt = record.filters()
        author = query.get("author", [None])[0]
        rows = [
            e
            for e in self.entries
            if matches_filter(e, filt) and (author is None or e["author"] == author)
        ]
        size = int(query.get("page_size", ["20"])[0])
        start = int(query.get("cursor", ["0"])[0] or 0)
        chunk = rows[start : start + size]
        if query.get("view", [None])[0] == "row":
            chunk = [to_row(e) for e in chunk]
        more = start + size < len(rows)
        return {
            "entries": chunk,
            "total": len(rows),
            "next_cursor": str(start + size) if more else None,
            "page": None,
            "page_size": size,
        }

    def _search(self, record: Request) -> dict[str, Any]:
        """POST /api/search: the ``search`` fixture, filtered and offset-paged."""
        body = record.json() or {}
        filt = body.get("filters") or {}
        pool = [e for e in self.search["entries"] if matches_filter(e, filt)]
        offset = int(body.get("offset") or 0)
        size = int(body.get("max_results") or 10)
        chunk = pool[offset : offset + size]
        if body.get("view") == "row":
            chunk = [to_row(e) for e in chunk]
        return {
            **self.search,
            "entries": chunk,
            "total": len(pool),
            "total_results": len(pool),
            "total_exact": True,
            "has_more": offset + size < len(pool),
        }


def wait_for_call(
    page: Page,
    backend: FakeBackend,
    method: str,
    path: str,
    predicate: Callable[[Request], bool] = lambda _r: True,
    *,
    timeout: float = 5.0,
) -> Request:
    """The latest recorded *method* *path* request matching *predicate*, polled until one lands.

    A request the page sends after a hash change is recorded asynchronously, so
    asserting on ``backend.calls`` right after the hash settles races it.
    """
    deadline = time.monotonic() + timeout
    while True:
        found = [r for r in backend.calls(method, path) if predicate(r)]
        if found:
            return found[-1]
        if time.monotonic() > deadline:
            raise AssertionError(f"no {method} {path} matching; saw {backend.calls(method, path)}")
        page.wait_for_timeout(50)


def open_panel(
    browser: Browser,
    url: str,
    backend: FakeBackend | None = None,
    *,
    viewport: dict[str, int] | None = None,
) -> tuple[PanelHost, FakeBackend]:
    """The panel at *url*, framed by the host page behind a fake backend, its list settled."""
    page = browser.new_page(viewport=viewport or {"width": 1280, "height": 800})
    backend = (backend or FakeBackend()).install(page)
    host = open_panel_host(page, url)
    expect(host.locator("#ariel-list-body")).to_have_attribute("aria-busy", "false")
    return host, backend


def search(page: PanelHost, text: str) -> None:
    """Type *text* into the host bar's search field, which reports it to the panel."""
    page.search_box().fill(text)


def compose(page: PanelHost) -> None:
    """Press the host bar's Compose button."""
    page.bar_item("ariel-compose").click()


def expand_rail(page: PanelHost) -> None:
    """Open the rail from its default icon strip, which is the only way to its facets."""
    page.keyboard.press("[")
    expect(page.locator(".app-container")).not_to_have_attribute(
        "data-rail-collapsed", re.compile(".*")
    )


def scratch_input(page: PanelHost) -> None:
    """Focus a text field inside the panel, for the keys that must not fire while typing.

    The panel's own fields live behind other screens (the search box is the host's),
    and which field it is does not matter to the guard.
    """
    page.evaluate(
        "() => { const i = document.createElement('input'); i.id = 'scratch-input';"
        " document.body.append(i); i.focus(); }"
    )


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
    f: dict[str, Any] | None = None,
    sel: str | None = None,
    att: str | None = None,
    compose: str | None = None,
) -> str:
    """The router's canonical hash: keys in order, empty ones omitted (``box`` never is).

    ``compose`` is the open draft's value (``draft:<id>``), written last.

    ``f`` is the filter as compact JSON with sorted top-level keys (the router's
    ``normalizeFilter`` order; a ``time`` window keeps ``start, end``). Under a
    ``saved:`` box ``q`` and ``f`` are always written, even empty (``q=&f={}``).
    """
    saved = box.startswith("saved:")
    f_json = json.dumps({k: f[k] for k in sorted(f)}, separators=(",", ":")) if f else None
    if saved:
        q = q or ""
        f_json = f_json or "{}"
    pairs = [
        ("q", q),
        ("mode", mode),
        ("box", box),
        ("f", f_json),
        ("sel", sel),
        ("att", att),
        ("compose", compose),
    ]
    always = {"q", "f"} if saved else set()
    return "#" + "&".join(
        f"{k}={quote(v or '', safe='')}" for k, v in pairs if v or (k in always and v is not None)
    )


def wait_for_hash(page: Page, expected: str) -> None:
    """Wait until the location hash is *expected* (the router rewrites it asynchronously)."""
    page.wait_for_function("(h) => location.hash === h", arg=expected)


def app(page: Page):
    """The panel root carrying the layout attributes."""
    return page.locator(".app-container")


# ---------------------------------------------------------------------------
# Requirement 1: the landing
# ---------------------------------------------------------------------------


def test_req1_shows_rail_newest_entries_and_empty_reader(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Rail, newest-first list with day separators, empty reader; no page navigation."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")

    expect(page.locator("#ariel-rail button[data-box]")).to_have_count(3)
    expect(page.locator("#ariel-rail")).to_be_visible()
    expect(app(page)).to_have_attribute("data-rail-collapsed", "")
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
    wait_for_page_ready(page)
    expect(reader_title(page)).to_have_text("Vacuum spike sector 3")
    expect(row(page, "E2")).to_have_attribute("aria-selected", "true")
    page.close()


def test_req2_reload_restores_query_mode_selection_and_attachment(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """A canonical hash brings back the query, the mode, the entry and its open picture."""
    page, backend = open_panel(chromium_browser, f"{ariel_url}/#q=beam&mode=semantic&sel=E1&att=A1")

    expect(page.search_box()).to_have_value("beam")
    expect(page.locator("#ariel-mode-chip")).to_have_attribute("data-mode", "semantic")
    expect(reader_title(page)).to_have_text("Beam dump at injection")
    expect(page.locator("#image-lightbox")).to_be_visible()
    searches = backend.calls("POST", "/api/search")
    assert searches, backend.requests
    assert searches[-1].json()["query"] == "beam"
    assert searches[-1].json()["mode"] == "semantic"

    page.reload(wait_until="load")
    wait_for_page_ready(page)
    expect(page.search_box()).to_have_value("beam")
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
    """``#create?draft=`` of a missing draft opens an empty form with an error toast.

    The draft is gone (404) and nothing was typed under it, so the sheet becomes
    a new entry (Decision 8).
    """
    outer = chromium_browser.new_page()
    backend = FakeBackend().install(outer)
    with outer.expect_console_message(lambda m: "Failed to load draft" in m.text):
        page = open_panel_host(outer, f"{ariel_url}/#create?draft=missing")

    expect(page.locator("#ariel-compose")).to_be_visible()
    assert backend.calls("GET", "/api/drafts/missing")
    expect(page.locator("#entry-subject")).to_have_value("")
    expect(page.locator("#entry-details")).to_have_value("")
    expect(page.locator("#ariel-compose-banner")).to_be_hidden()
    expect(page.locator(".osprey-toast-host [role=alert] .osprey-toast")).to_contain_text(
        "draft is gone"
    )
    page.close()


# ---------------------------------------------------------------------------
# Requirement 4: search surfaces (config + expansion banners: test_vocabulary_banner)
# ---------------------------------------------------------------------------


def test_req4_search_shows_diagnostics_and_no_answer_box(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Diagnostics ride on the mode chip, counts and rows render; a response's
    ``answer`` is never shown."""
    backend = FakeBackend(
        search=search_response(
            [DEFAULT_ENTRIES[1]],
            diagnostics=[{"level": "warning", "source": "semantic", "message": "model offline"}],
            answer="THE GENERATED ANSWER",
        )
    )
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", backend)

    search(page, "vacuum")

    expect(page.locator("#ariel-list .ariel-row")).to_have_count(1)
    expect(row(page, "E2")).to_be_visible()
    expect(page.locator("#ariel-list .diagnostics-bar")).to_have_count(0)
    chip = page.locator("#ariel-mode-chip")
    expect(chip.locator(".ariel-mode-chip-dot")).to_have_attribute("data-level", "warning")
    expect(chip).to_have_attribute("aria-description", re.compile("model offline"))
    chip.focus()
    page.keyboard.press("Enter")
    expect(page.locator("#ariel-filters-popover [data-search-issues]")).to_contain_text(
        "model offline"
    )
    page.keyboard.press("Escape")
    expect(page.locator("#ariel-list .results-count")).to_contain_text("1")
    expect(page.locator("body")).not_to_contain_text("THE GENERATED ANSWER")
    wait_for_hash(page, canonical(q="vacuum"))
    page.close()


def test_req4_invalid_configuration_disables_search(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """``configuration_invalid`` shows the blocking banner, omits the search item, disables the chip."""
    page, _ = open_panel(
        chromium_browser, f"{ariel_url}/", FakeBackend(capabilities=dict(INVALID_CAPABILITIES))
    )

    expect(page.locator("#config-invalid-banner")).to_be_visible()
    assert not [item for item in page.items() if item["kind"] == "search"]
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

    compose(page)
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

    # The list shows the new row, selected, with the new count, without a reload.
    expect(row(page, "NEW-1")).to_have_attribute("aria-selected", "true")
    expect(page.locator("#ariel-list [data-list-total]")).to_have_text(
        str(len(DEFAULT_ENTRIES) + 1)
    )

    # The name just used is Mine's: no prompt, the listing is scoped to it.
    page.click('#ariel-rail button[data-box="mine"]')
    expect(page.locator('[data-rail-field="mine"]')).to_be_hidden()
    page.wait_for_function("location.hash.includes('box=mine')")
    wait_for_call(
        page, backend, "GET", "/api/entries", lambda r: r.filters().get("author") == ["bob"]
    )
    page.close()


def test_req5_create_with_attachment_and_credentials(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """An adapter that needs credentials asks for them; files go multipart."""
    backend = FakeBackend()
    backend.publish_info["requires_auth"] = True
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", backend)

    compose(page)
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
    compose(page)
    expect(page.locator("#ariel-compose")).to_be_visible()
    page.click("#create-cancel-btn")
    expect(page.locator("#ariel-compose")).to_be_hidden()
    expect(row(page, "E1")).to_be_visible()
    page.close()


def test_req5_draft_review_prefills_the_form(ariel_url: str, chromium_browser: Browser) -> None:
    """``#create?draft=D1`` opens the form prefilled, with its provenance banner."""
    page, backend = open_panel(chromium_browser, f"{ariel_url}/#create?draft=D1")

    expect(page.locator("#ariel-compose")).to_be_visible()
    expect(page.locator("#entry-subject")).to_have_value("Drafted by the agent")
    expect(page.locator("#entry-details")).to_have_value("Agent-written details.")
    expect(page.locator("#ariel-compose-banner")).to_have_attribute("data-draft-id", "D1")
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


def test_req6_pane_sizes_survive_reload(ariel_url: str, chromium_browser: Browser) -> None:
    """The rail's state and both pane widths come back after a reload."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")

    page.keyboard.press("[")
    expect(app(page)).not_to_have_attribute("data-rail-collapsed", re.compile(".*"))
    drag(page, "#ariel-rail-splitter", 80)  # 208 -> ~288
    drag(page, "#ariel-list-splitter", 60)
    rail_w, list_w = width(page, "#ariel-rail"), width(page, "#ariel-list-col")
    assert rail_w > 260, rail_w
    assert any("ariel-rail-embedded" in k for k in _storage_keys(page, "ariel-rail-"))

    page.reload(wait_until="load")
    wait_for_page_ready(page)
    expect(page.locator("#ariel-list-body")).to_have_attribute("aria-busy", "false")
    expect(app(page)).not_to_have_attribute("data-rail-collapsed", re.compile(".*"))
    assert abs(width(page, "#ariel-rail") - rail_w) <= 2
    assert abs(width(page, "#ariel-list-col") - list_w) <= 2
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
    """Drafts with no live draft says so, and lists no entry rows."""
    page, backend = open_panel(chromium_browser, f"{ariel_url}/", FakeBackend(drafts={}))
    _box(page, "drafts").click()
    expect(_box(page, "drafts")).to_have_attribute("aria-current", "page")
    expect(page.locator("#ariel-list-body")).to_contain_text("No Drafts")
    expect(page.locator("#ariel-list-body")).to_contain_text(
        "Drafts an agent stages for you appear here."
    )
    wait_for_call(page, backend, "GET", "/api/drafts")
    expect(page.locator("#ariel-list [data-entry-id]")).to_have_count(0)
    expect(page.locator("#ariel-list [data-draft-id]")).to_have_count(0)
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
    wait_for_call(
        page, backend, "GET", "/api/entries", lambda r: r.filters().get("author") == ["alice"]
    )

    # The name is remembered: a reload on box=mine lists alice's entries directly.
    page.reload(wait_until="load")
    wait_for_page_ready(page)
    expect(row(page, "E4")).to_be_visible()
    expect(field_).to_be_hidden()
    page.close()


def test_mailbox_saved_search_saves_runs_and_removes(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """A saved search keeps query + mode under a name, reruns them, and can be removed."""
    page, backend = open_panel(chromium_browser, f"{ariel_url}/")

    search(page, "beam")
    expect(page.locator("#ariel-list .results-count")).to_be_visible()

    page.click('#ariel-rail [data-rail-action="save-search"]')
    name = page.locator('[data-rail-field="save"]')
    expect(name).to_be_visible()
    name.fill("Beam")
    name.press("Enter")
    saved = _box(page, "saved:Beam")
    expect(saved).to_be_visible()

    # Back to the plain listing, then rerun the saved search.
    search(page, "")
    expect(page.locator("#ariel-list .ariel-day-sep").first).to_be_visible()
    before = len(backend.calls("POST", "/api/search"))
    saved.click()
    expect(page.search_box()).to_have_value("beam")
    expect(saved).to_have_attribute("aria-current", "page")
    wait_for_hash(page, canonical(q="beam", box="saved:Beam"))
    expect(page.locator("#ariel-list .results-count")).to_be_visible()
    assert len(backend.calls("POST", "/api/search")) > before

    # Survives a reload (browser storage), then removal drops it and returns to All.
    page.reload(wait_until="load")
    wait_for_page_ready(page)
    expect(saved).to_be_visible()
    page.click('#ariel-rail [data-remove-saved="Beam"]')
    expect(saved).to_have_count(0)
    expect(_box(page, "all")).to_have_attribute("aria-current", "page")
    page.close()


# ---------------------------------------------------------------------------
# Rail: drag, snap, toggle, default
# ---------------------------------------------------------------------------


def test_rail_default_state(ariel_url: str, chromium_browser: Browser) -> None:
    """The rail starts as an icon strip."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")
    expect(app(page)).to_have_attribute("data-rail-collapsed", "")
    assert width(page, "#ariel-rail") < 60
    page.close()


def test_rail_drag_resizes_and_snaps_to_the_icon_strip(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Dragging the border resizes; releasing below ~110 px snaps to the strip."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")
    expand_rail(page)
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
    expand_rail(page)
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

    # Typing in a field is not a shortcut.
    scratch_input(page)
    page.keyboard.press("j")
    expect(row(page, "E2")).to_have_attribute("aria-selected", "true")
    expect(page.locator("#scratch-input")).to_have_value("j")
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


def test_key_bracket_toggles_the_rail(ariel_url: str, chromium_browser: Browser) -> None:
    """``[`` collapses and expands the rail."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")
    expand_rail(page)
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

    compose(page)
    expect(page.locator("#ariel-compose")).to_be_visible()
    page.focus("#entry-subject")
    page.keyboard.press("Escape")  # Esc works while typing
    expect(page.locator("#ariel-compose")).to_be_hidden()
    page.close()


# ---------------------------------------------------------------------------
# Legacy hash aliases
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
    """``#create`` opens the empty form; ``?draft=`` prefills it and lands on its ``compose``."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/{alias}")
    expect(page.locator("#ariel-compose")).to_be_visible()
    expected = "Drafted by the agent" if "draft" in alias else ""
    expect(page.locator("#entry-subject")).to_have_value(expected)
    wait_for_hash(page, canonical(compose="draft:D1" if "draft" in alias else None))
    page.close()


@pytest.mark.parametrize("alias", ["#search", "#browse"])
def test_alias_search_and_browse_show_the_list(
    ariel_url: str, chromium_browser: Browser, alias: str
) -> None:
    """``#search`` and ``#browse`` land on the plain list, whatever was open before."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/#q=beam&sel=E1")
    expect(reader_title(page)).to_be_visible()
    page.evaluate("(a) => { location.hash = a; }", alias)
    expect(page.search_box()).to_have_value("")
    expect(page.locator("#ariel-list [data-entry-id]")).to_have_count(3)
    expect(page.locator("#ariel-reader-body")).to_be_empty()
    wait_for_hash(page, canonical())
    page.close()


def test_alias_status_shows_the_list(ariel_url: str, chromium_browser: Browser) -> None:
    """``#status`` shows the list: no control opens the Status drawer from a link."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/#status")
    expect(page.locator("osprey-drawer#status-drawer")).not_to_have_attribute(
        "open", re.compile(".*")
    )
    expect(page.locator("#ariel-list [data-entry-id]")).to_have_count(3)
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
    expect(page.locator("#ariel-rail button[data-box].osprey-nav-row")).to_have_count(3)
    expect(page.locator("#ariel-filters-popover.osprey-popover")).to_have_count(1)
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
    # The first keyset page (20 rows); the rest wait behind Load more.
    expect(page.locator("#ariel-list [data-entry-id]")).to_have_count(min(len(entries), 20))

    for entry in entries[:8]:
        target = row(page, entry["entry_id"])
        target.click(timeout=3000)
        expect(target).to_have_attribute("aria-selected", "true")
        expect(reader_title(page)).to_contain_text(entry["raw_text"].split("\n")[0])
    page.close()


# ---------------------------------------------------------------------------
# Facets, query operators, paging, the badge strip
# ---------------------------------------------------------------------------


def _facet(page: Page, dim: str, value: str):
    """The rail facet row of *value* in dimension *dim*."""
    return page.locator(
        f'.ariel-rail-facets button[data-facet-dim="{dim}"][data-facet-value="{value}"]'
    )


def _with_alice() -> FakeBackend:
    """The default entries plus two of alice's (E4 on 08-20, E5 on 08-19)."""
    backend = FakeBackend()
    backend.entries[2:2] = [
        make_entry(
            "E4", "Alice's first entry", author="alice", timestamp="2026-08-20T08:00:00-07:00"
        ),
    ]
    backend.entries.append(
        make_entry(
            "E5", "Alice's second entry", author="Alice", timestamp="2026-08-19T08:00:00-07:00"
        )
    )
    return backend


def _paged_entries(count: int) -> list[dict[str, Any]]:
    """*count* entries, newest first, an hour apart."""
    return [
        make_entry(
            f"P{i:02d}",
            f"Paged entry {i}",
            timestamp=f"2026-08-{20 - i // 24:02d}T{23 - i % 24:02d}:00:00-07:00",
        )
        for i in range(count)
    ]


def _rows(page: Page):
    return page.locator("#ariel-list [data-entry-id]")


def test_facet_click_filters_the_list_and_writes_f(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Counts come from the entries; a value click narrows the list, presses the row, writes ``f``."""
    page, backend = open_panel(chromium_browser, f"{ariel_url}/", _with_alice())
    expand_rail(page)
    expect(page.locator(".ariel-rail-facets")).to_have_attribute("aria-busy", "false")

    # alice and Alice are one value: matching ignores case.
    expect(_facet(page, "author", "operator").locator(".osprey-nav-badge")).to_have_text("3")
    expect(_facet(page, "author", "alice").locator(".osprey-nav-badge")).to_have_text("2")

    _facet(page, "author", "alice").click()
    expect(_rows(page)).to_have_count(2)
    expect(row(page, "E4")).to_be_visible()
    expect(_facet(page, "author", "alice")).to_have_attribute("aria-pressed", "true")
    wait_for_hash(page, canonical(f={"author": ["alice"]}))
    wait_for_call(
        page, backend, "GET", "/api/entries", lambda r: r.filters() == {"author": ["alice"]}
    )
    # Self-exclusion: the other author keeps its count to OR with.
    expect(_facet(page, "author", "operator").locator(".osprey-nav-badge")).to_have_text("3")
    page.close()


def test_facet_filter_survives_reload(ariel_url: str, chromium_browser: Browser) -> None:
    """A reload restores the filter from ``f``: the list, the pressed row and the request."""
    page, backend = open_panel(chromium_browser, f"{ariel_url}/", _with_alice())
    expand_rail(page)
    _facet(page, "author", "alice").click()
    wait_for_hash(page, canonical(f={"author": ["alice"]}))

    backend.requests.clear()
    page.reload(wait_until="load")
    wait_for_page_ready(page)
    expect(_rows(page)).to_have_count(2)
    expect(_facet(page, "author", "alice")).to_have_attribute("aria-pressed", "true")
    wait_for_call(
        page, backend, "GET", "/api/entries", lambda r: r.filters() == {"author": ["alice"]}
    )
    page.close()


def test_facet_counts_equal_the_list_under_mine(ariel_url: str, chromium_browser: Browser) -> None:
    """Under Mine the author section shows only Mine's author, its count the list length."""
    page, backend = open_panel(chromium_browser, f"{ariel_url}/", _with_alice())
    _box(page, "mine").click()
    field_ = page.locator('[data-rail-field="mine"]')
    field_.fill("alice")
    field_.press("Enter")

    wait_for_hash(page, canonical(box="mine"))
    expect(_rows(page)).to_have_count(2)
    authors = page.locator('.ariel-rail-facets button[data-facet-dim="author"]')
    expect(authors).to_have_count(1)
    expect(authors.locator(".osprey-nav-badge")).to_have_text(str(_rows(page).count()))
    wait_for_call(
        page, backend, "GET", "/api/facets", lambda r: r.filters().get("author") == ["alice"]
    )
    page.close()


def test_histogram_day_click_lists_only_that_day(ariel_url: str, chromium_browser: Browser) -> None:
    """A histogram bar sets a one-day ``time`` window; the list shows only that day."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")
    expand_rail(page)
    day = page.locator('.ariel-rail-facets button.ariel-facet-day[data-day="2026-08-19"]')
    expect(day).to_have_attribute("aria-label", "2026-08-19: 1")
    # A year of bars in the rail is sub-pixel wide each, so a pointer click
    # cannot aim at one day; the bar's own click handler is what is tested.
    day.dispatch_event("click")

    expect(_rows(page)).to_have_count(1)
    expect(row(page, "E3")).to_be_visible()
    wait_for_hash(page, canonical(f={"time": {"start": "2026-08-19", "end": "2026-08-19"}}))
    expect(day).to_have_attribute("aria-pressed", "true")
    page.close()


def test_link_naming_an_undeclared_dimension_shows_the_notice(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """``f`` naming a dimension the logbook does not declare is dropped, with a notice."""
    f = quote(json.dumps({"bogus": ["x"]}, separators=(",", ":")), safe="")
    page, backend = open_panel(chromium_browser, f"{ariel_url}/#box=all&f={f}")

    notice = page.locator("[data-list-notice]")
    expect(notice).to_be_visible()
    expect(notice).to_have_text("Removed filter this logbook does not offer: bogus")
    expect(_rows(page)).to_have_count(3)
    assert all(r.filters() == {} for r in backend.calls("GET", "/api/entries"))
    page.close()


def test_facets_failure_falls_back_to_the_plain_list(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """A failing facets route shows no counts and no error; the list and mount stay usable."""
    backend = FakeBackend(facets_status=500)
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", backend)
    expand_rail(page)

    expect(page.locator(".ariel-rail-facets")).to_have_attribute("aria-busy", "false")
    wait_for_call(page, backend, "GET", "/api/facets")
    expect(page.locator(".ariel-rail-facets [data-facet-section]")).to_have_count(0)
    expect(_rows(page)).to_have_count(3)
    expect(page.locator("#ariel-rail button[data-box]")).to_have_count(3)
    page.close()


def test_search_lifts_operators_into_the_filter(ariel_url: str, chromium_browser: Browser) -> None:
    """``beam author:alice`` searches ``beam`` filtered to alice, and writes ``f``."""
    page, backend = open_panel(chromium_browser, f"{ariel_url}/", _with_alice())
    search(page, "beam author:alice")

    call = wait_for_call(
        page, backend, "POST", "/api/search", lambda r: r.filters() == {"author": ["alice"]}
    )
    assert call.json()["query"] == "beam"
    wait_for_hash(page, canonical(q="beam", f={"author": ["alice"]}))
    page.close()


def test_browse_loads_older_entries_to_the_end(ariel_url: str, chromium_browser: Browser) -> None:
    """Browse pages by cursor; at the end (no ``next_cursor``) the button goes away."""
    backend = FakeBackend(entries=_paged_entries(45))
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", backend)
    more = page.locator("#ariel-list [data-load-more]")

    expect(_rows(page)).to_have_count(20)
    more.click()
    expect(_rows(page)).to_have_count(40)
    more.click()
    expect(_rows(page)).to_have_count(45)
    expect(more).to_have_count(0)
    cursors = [r.query.get("cursor", [None])[0] for r in backend.calls("GET", "/api/entries")]
    assert cursors == [None, "20", "40"], cursors
    assert all(r.query.get("view") == ["row"] for r in backend.calls("GET", "/api/entries"))
    page.close()


def test_search_loads_more_results_to_the_end(ariel_url: str, chromium_browser: Browser) -> None:
    """Search pages by offset; ``has_more`` false hides the button."""
    entries = _paged_entries(45)
    backend = FakeBackend(entries=entries, search=search_response(entries))
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", backend)
    search(page, "paged")
    more = page.locator("#ariel-list [data-load-more]")

    expect(more).to_be_visible()
    expect(_rows(page)).to_have_count(20)
    more.click()
    expect(_rows(page)).to_have_count(40)
    more.click()
    expect(_rows(page)).to_have_count(45)
    expect(more).to_have_count(0)
    offsets = [r.json().get("offset", 0) for r in backend.calls("POST", "/api/search")]
    assert offsets[-2:] == [20, 40], offsets
    page.close()


def test_badge_strip_is_on_by_default_and_its_menu_entry_persists(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """A wide panel shows the strip; the ⋯ entry turns it off, across a reload."""
    backend = FakeBackend()
    backend.entries[0]["metadata"] = {"enhancements": {"text_embedding": "failed"}}
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", backend)
    toggle = page.bar_item("ariel-more", "toggle-strip")
    cells = page.locator("#ariel-list .ariel-strip-cell")

    expect(
        row(page, "E1").locator('.ariel-strip-cell[data-module="text_embedding"]')
    ).to_have_attribute("data-state", "failed")
    expect(cells).to_have_count(1)  # only E1 has a state; n/a cells render nothing
    expect(toggle).to_have_attribute("aria-checked", "true")

    toggle.click()
    expect(toggle).to_have_attribute("aria-checked", "false")
    expect(cells).to_have_count(0)

    page.reload(wait_until="load")
    wait_for_page_ready(page)
    expect(row(page, "E1")).to_be_visible()
    expect(cells).to_have_count(0)
    expect(toggle).to_have_attribute("aria-checked", "false")
    page.close()


def test_row_click_clears_a_stale_att(ariel_url: str, chromium_browser: Browser) -> None:
    """Opening another row drops the previous entry's ``att``."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/#sel=E1&att=A1")
    wait_for_hash(page, canonical(sel="E1", att="A1"))

    # The lightbox covers the list; the row's own click handler is what is tested.
    row(page, "E2").dispatch_event("click")
    wait_for_hash(page, canonical(sel="E2"))
    expect(reader_title(page)).to_have_text("Vacuum spike sector 3")
    page.close()


def test_filters_popover_author_and_facet_click_agree(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """A Filters & Options author pick and the author facet send the same filter and ``f``."""

    def backend() -> FakeBackend:
        fake = _with_alice()
        fake.capabilities = {
            **HEALTHY_CAPABILITIES,
            "shared_parameters": [{"name": "author", "type": "text", "label": "Author"}],
        }
        return fake

    # The facet click.
    page, fake = open_panel(chromium_browser, f"{ariel_url}/", backend())
    expand_rail(page)
    _facet(page, "author", "alice").click()
    by_facet = wait_for_call(
        page, fake, "GET", "/api/entries", lambda r: r.filters().get("author") == ["alice"]
    ).filters()
    expect(_rows(page)).to_have_count(2)
    hash_facet = page.evaluate("location.hash")
    page.close()

    # The popover's author pick.
    page, fake = open_panel(chromium_browser, f"{ariel_url}/", backend())
    page.click("#ariel-mode-chip")
    author = page.locator("#ariel-filters-popover #param-author")
    expect(author).to_be_visible()
    author.fill("alice")
    author.press("Enter")
    by_popover = wait_for_call(
        page, fake, "GET", "/api/entries", lambda r: r.filters().get("author") == ["alice"]
    ).filters()
    expect(_rows(page)).to_have_count(2)
    wait_for_hash(page, hash_facet)
    page.close()

    assert by_popover == by_facet == {"author": ["alice"]}
    assert hash_facet == canonical(f={"author": ["alice"]})


# ---------------------------------------------------------------------------
# Compose sheet: placement, kinds, drafts, follow-ups, autosave, keys
# ---------------------------------------------------------------------------


def _sheet(page: Page):
    """The compose sheet."""
    return page.locator("#ariel-compose")


def _sheet_title(page: Page):
    return page.locator("#ariel-compose-title")


def _submit(page: Page) -> None:
    page.click("#create-entry-form button[type=submit]")


def _toast(page: Page, role: str):
    """Toasts in the ``status`` (info/success/warning) or ``alert`` (error) region."""
    return page.locator(f".osprey-toast-host [role={role}] .osprey-toast")


def _go(page: Page, hash_: str) -> None:
    """Navigate the open page to *hash_* without a reload."""
    page.evaluate("(h) => { location.hash = h; }", hash_)


def _hold(page: Page, pattern: str) -> list[Route]:
    """Hold every request matching *pattern* unanswered; release one with ``.fallback()``.

    Registered after ``FakeBackend.install``, so it runs first; ``fallback()``
    hands the request on to the fake.
    """
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    page.route(re.compile(pattern), hold)
    return held


def _wait_until(page: Page, check: Callable[[], bool], what: str, timeout: float = 5.0) -> None:
    """Poll *check* while the page's routes keep being served."""
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        page.wait_for_timeout(50)


def _storage_dump(page: Page) -> str:
    """Everything this origin keeps in localStorage and sessionStorage, as one string."""
    return page.evaluate(
        "() => JSON.stringify([Object.entries(localStorage), Object.entries(sessionStorage)])"
    )


def _wait_autosaved(page: Page, text: str) -> None:
    """Wait until the compose autosave record holds *text* (the debounced save landed)."""
    page.wait_for_function(
        "(t) => Object.keys(localStorage).some((k) => k.startsWith('ariel-compose-autosave')"
        " && (localStorage.getItem(k) || '').includes(t))",
        arg=text,
    )


def test_compose_sheet_leaves_the_reader_showing_the_selection(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """With an entry selected, the sheet covers the list column and the reader keeps that entry."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/{canonical(sel='E2')}")
    expect(reader_title(page)).to_have_text("Vacuum spike sector 3")

    compose(page)
    expect(_sheet(page)).to_be_visible()
    expect(_sheet_title(page)).to_have_text("New entry")
    expect(reader_title(page)).to_be_visible()
    expect(reader_title(page)).to_have_text("Vacuum spike sector 3")
    sheet = _sheet(page).bounding_box()
    reader = page.locator("#ariel-reader").bounding_box()
    assert sheet is not None and reader is not None
    assert sheet["x"] + sheet["width"] <= reader["x"] + 1, (sheet, reader)
    # A new entry never reaches the address.
    wait_for_hash(page, canonical(sel="E2"))
    page.close()


def test_compose_sheet_maximises_and_restores(ariel_url: str, chromium_browser: Browser) -> None:
    """Maximise widens the sheet across the panel; restore returns it to the list column."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")
    compose(page)
    expect(_sheet(page)).to_be_visible()
    narrow = width(page, "#ariel-compose")
    sheet_width = "() => document.querySelector('#ariel-compose').getBoundingClientRect().width"

    page.click("#ariel-compose-max")
    expect(page.locator("#ariel-compose-max")).to_have_attribute("aria-pressed", "true")
    page.wait_for_function(f"(n) => ({sheet_width})() > n + 300", arg=narrow)
    expect(reader_title(page)).to_have_count(0)  # nothing selected: the reader is empty

    page.click("#ariel-compose-max")
    expect(page.locator("#ariel-compose-max")).to_have_attribute("aria-pressed", "false")
    page.wait_for_function(f"(n) => Math.abs(({sheet_width})() - n) <= 2", arg=narrow)
    page.close()


def test_compose_sheet_fills_the_tile_when_stacked(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Stacked, a follow-up opened from the reader pane fills the tile; closing shows the reader."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", viewport={"width": 560, "height": 800})
    row(page, "E1").click()
    expect(app(page)).to_have_attribute("data-narrow-view", "reader")

    page.click("#ariel-reader-body [data-follow-up]")
    expect(_sheet(page)).to_be_visible()
    expect(_sheet_title(page)).to_have_text("Follow up")
    expect(page.locator("#ariel-compose-max")).to_be_hidden()
    sheet = _sheet(page).bounding_box()
    assert sheet is not None
    assert sheet["x"] <= 1 and sheet["width"] >= 558, sheet
    assert sheet["y"] + sheet["height"] >= 798, sheet

    page.click("#ariel-compose-close")
    expect(_sheet(page)).to_be_hidden()
    expect(app(page)).to_have_attribute("data-narrow-view", "reader")
    expect(reader_title(page)).to_have_text("Beam dump at injection")
    page.close()


def test_compose_draft_address_opens_the_draft_and_never_submits_it(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """``#compose=draft:<id>`` opens that draft, survives a reload, and never submits by itself."""
    page, backend = open_panel(chromium_browser, f"{ariel_url}/{canonical(compose='draft:D1')}")

    expect(_sheet(page)).to_be_visible()
    expect(_sheet_title(page)).to_have_text("Draft")
    expect(page.locator("#entry-subject")).to_have_value("Drafted by the agent")
    expect(page.locator("#ariel-compose-banner")).to_have_attribute("data-draft-id", "D1")
    wait_for_hash(page, canonical(compose="draft:D1"))

    page.reload(wait_until="load")
    wait_for_page_ready(page)
    expect(page.locator("#entry-subject")).to_have_value("Drafted by the agent")
    wait_for_hash(page, canonical(compose="draft:D1"))
    page.wait_for_timeout(500)
    assert not backend.calls("POST", "/api/entries"), backend.requests
    assert not backend.calls("POST", "/api/entries/upload"), backend.requests
    assert not backend.calls("DELETE", "/api/drafts/D1"), backend.requests
    page.close()


def test_compose_draft_without_provenance_shows_no_banner(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """A draft that records no stager opens without the provenance banner."""
    plain = {**DRAFT, "draft_id": "D2", "subject": "Gallery draft", "metadata": {}}
    backend = FakeBackend(drafts={"D2": plain})
    page, _ = open_panel(chromium_browser, f"{ariel_url}/{canonical(compose='draft:D2')}", backend)

    expect(page.locator("#entry-subject")).to_have_value("Gallery draft")
    expect(page.locator("#ariel-compose-banner")).to_be_hidden()
    page.close()


def test_mailbox_drafts_lists_newest_first_discards_and_submits(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Drafts lists every draft newest first; Discard and Submit each remove one."""
    backend = FakeBackend(
        drafts={
            "D1": {**DRAFT, "created_at": "2026-08-20T08:00:00-07:00"},
            "D2": {
                **DRAFT,
                "draft_id": "D2",
                "subject": "Gallery draft",
                "metadata": {},
                "created_at": "2026-08-20T11:00:00-07:00",
            },
            "D3": {
                **DRAFT,
                "draft_id": "D3",
                "subject": "API draft",
                "metadata": {},
                "created_at": "2026-08-20T09:30:00-07:00",
            },
        }
    )
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", backend)
    _box(page, "drafts").click()
    rows = page.locator("#ariel-list [data-draft-id]")
    expect(rows).to_have_count(3)
    assert rows.evaluate_all("rs => rs.map(r => r.dataset.draftId)") == ["D2", "D3", "D1"]
    expect(page.locator("#ariel-list [data-entry-id]")).to_have_count(0)

    # Discard deletes the draft, says so, and the listing drops it.
    page.locator('#ariel-list [data-draft-id="D3"]').click()
    expect(_sheet_title(page)).to_have_text("Draft")
    expect(page.locator("#entry-subject")).to_have_value("API draft")
    wait_for_hash(page, canonical(box="drafts", compose="draft:D3"))
    page.click("#create-cancel-btn")
    wait_for_call(page, backend, "DELETE", "/api/drafts/D3")
    expect(_toast(page, "status")).to_contain_text("Draft discarded")
    expect(_sheet(page)).to_be_hidden()
    expect(rows).to_have_count(2)
    wait_for_hash(page, canonical(box="drafts"))

    # Submitting a draft removes it as well.
    page.locator('#ariel-list [data-draft-id="D2"]').click()
    expect(page.locator("#entry-subject")).to_have_value("Gallery draft")
    _submit(page)
    created = wait_for_call(page, backend, "POST", "/api/entries")
    assert created.json()["subject"] == "Gallery draft"
    wait_for_call(page, backend, "DELETE", "/api/drafts/D2")
    expect(_sheet(page)).to_be_hidden()
    expect(rows).to_have_count(1)
    expect(page.locator('#ariel-list [data-draft-id="D1"]')).to_be_visible()
    page.close()


def _thread_backend() -> FakeBackend:
    """T1 followed up by T2 and T3; T4 follows up an entry ARIEL no longer holds."""
    parent = make_entry("T1", "Magnet trip in sector 5", timestamp="2026-08-20T08:00:00-07:00")
    first = make_entry(
        "T2", "Magnet reset", timestamp="2026-08-20T09:00:00-07:00", metadata={"follows_up": "T1"}
    )
    second = make_entry(
        "T3",
        "Magnet back in service",
        timestamp="2026-08-20T10:00:00-07:00",
        metadata={"follows_up": "T1"},
    )
    orphan = make_entry(
        "T4",
        "Note on a pruned entry",
        timestamp="2026-08-20T11:00:00-07:00",
        metadata={"follows_up": "GONE-7"},
    )
    return FakeBackend(entries=[orphan, second, first, parent])


def test_reader_thread_links_parent_and_children(ariel_url: str, chromium_browser: Browser) -> None:
    """The body renders before the thread; each thread link opens its entry; a lost parent shows its id."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", _thread_backend())
    held = _hold(page, r".*/api/entries/T2/thread$")

    row(page, "T2").click()
    expect(reader_title(page)).to_have_text("Magnet reset")
    _wait_until(page, lambda: bool(held), "the T2 thread request")
    expect(page.locator("#ariel-reader-body [data-entry-thread]")).to_have_count(0)
    held.pop().fallback()

    thread = page.locator("#ariel-reader-body [data-entry-thread]")
    expect(thread).to_contain_text("Follows up")
    thread.locator('[data-thread-entry="T1"]').click()
    expect(reader_title(page)).to_have_text("Magnet trip in sector 5")
    wait_for_hash(page, canonical(sel="T1"))

    links = page.locator("#ariel-reader-body [data-entry-thread] [data-thread-entry]")
    expect(links).to_have_count(2)
    expect(thread).to_contain_text("Follow-ups (2)")
    assert links.evaluate_all("ls => ls.map(l => l.dataset.threadEntry)") == ["T2", "T3"]
    links.nth(1).click()
    expect(reader_title(page)).to_have_text("Magnet back in service")
    wait_for_hash(page, canonical(sel="T3"))

    row(page, "T4").click()
    expect(reader_title(page)).to_have_text("Note on a pruned entry")
    expect(page.locator("#ariel-reader-body .entry-thread-missing")).to_have_text("GONE-7")
    expect(page.locator("#ariel-reader-body [data-thread-entry]")).to_have_count(0)
    page.close()


def test_follow_up_prefills_names_the_parent_and_submits_follows_up(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Following up ``Follow-up: X`` keeps the subject, copies logbook and tags, sends follows_up."""
    parent = make_entry(
        "F1", "Follow-up: RF trip", metadata={"logbook": "Operations", "tags": ["rf", "trip"]}
    )
    backend = FakeBackend(entries=[parent, *DEFAULT_ENTRIES])
    page, _ = open_panel(chromium_browser, f"{ariel_url}/{canonical(sel='F1')}", backend)
    expect(reader_title(page)).to_have_text("Follow-up: RF trip")

    page.click("#ariel-reader-body [data-follow-up]")
    expect(_sheet_title(page)).to_have_text("Follow up")
    expect(page.locator("#entry-subject")).to_have_value("Follow-up: RF trip")
    expect(page.locator("#entry-logbook")).to_have_value("Operations")
    tags = page.locator("#entry-tags .tag")
    expect(tags).to_have_count(2)
    assert tags.evaluate_all("ts => ts.map(t => t.dataset.value)") == ["rf", "trip"]
    expect(page.locator("#ariel-compose-parent")).to_be_visible()
    expect(page.locator("#ariel-compose-parent-ref")).to_have_text("Follow-up: RF trip")
    expect(reader_title(page)).to_have_text("Follow-up: RF trip")
    wait_for_hash(page, canonical(sel="F1"))  # a follow-up never reaches the address

    page.fill("#entry-details", "Recovered after the reset.")
    page.fill("#entry-author", "bob")
    _submit(page)
    body = wait_for_call(page, backend, "POST", "/api/entries").json()
    assert body["metadata"]["follows_up"] == "F1", body
    assert body["subject"] == "Follow-up: RF trip"
    assert body["logbook"] == "Operations"
    assert body["tags"] == ["rf", "trip"]
    expect(_sheet(page)).to_be_hidden()
    page.close()


@pytest.mark.parametrize(
    ("publish", "destination"),
    [
        ({"supports_write": True, "requires_auth": False}, "→ olog logbook"),
        ({"supports_write": True, "requires_auth": True}, "→ olog logbook (credentials required)"),
        (
            {"supports_write": False, "requires_auth": False},
            "→ This Logbook only (facility logbook read-only)",
        ),
    ],
)
def test_compose_sheet_names_the_destination(
    ariel_url: str, chromium_browser: Browser, publish: dict[str, bool], destination: str
) -> None:
    """The sheet says where the entry goes: the facility logbook, or local ARIEL when read-only."""
    backend = FakeBackend()
    backend.publish_info.update(publish)
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", backend)
    compose(page)
    expect(page.locator("#ariel-compose-dest")).to_have_text(destination)
    page.close()


def test_compose_autosave_survives_reload_and_close_never_credentials(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Typed text survives a reload and a close/reopen; Discard and Submit clear it.

    Credentials and attachments are never written to browser storage nor restored.
    """
    backend = FakeBackend()
    backend.publish_info["requires_auth"] = True
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", backend)

    compose(page)
    expect(page.locator("#publish-credentials")).to_be_visible()
    page.fill("#entry-auth-user", "opsuser")
    page.fill("#entry-auth-password", "s3cret")
    page.set_input_files(
        "#entry-files", files=[{"name": "plot.png", "mimeType": "image/png", "buffer": PNG_1X1}]
    )
    _fill_entry(page, "Autosaved subject")
    _wait_autosaved(page, "Some details.")
    expect(page.locator("#ariel-compose-saved")).to_contain_text("saved locally")
    dump = _storage_dump(page)
    for secret in ("opsuser", "s3cret", "plot.png"):
        assert secret not in dump, (secret, dump)

    # Reload: a new entry is not in the address, so the sheet reopens by hand.
    page.reload(wait_until="load")
    wait_for_page_ready(page)
    expect(page.locator("#ariel-list-body")).to_have_attribute("aria-busy", "false")
    expect(_sheet(page)).to_be_hidden()
    compose(page)
    expect(page.locator("#entry-subject")).to_have_value("Autosaved subject")
    expect(page.locator("#entry-details")).to_have_value("Some details.")
    expect(page.locator("#entry-auth-user")).to_have_value("")
    expect(page.locator("#entry-auth-password")).to_have_value("")
    assert page.eval_on_selector("#entry-files", "el => el.files.length") == 0

    # Close keeps it.
    page.fill("#entry-subject", "Edited after reload")
    page.click("#ariel-compose-close")
    expect(_sheet(page)).to_be_hidden()
    compose(page)
    expect(page.locator("#entry-subject")).to_have_value("Edited after reload")

    # Discard clears it.
    page.click("#create-cancel-btn")
    expect(_sheet(page)).to_be_hidden()
    compose(page)
    expect(page.locator("#entry-subject")).to_have_value("")
    expect(page.locator("#entry-details")).to_have_value("")

    # A successful submit clears it.
    _fill_entry(page, "Submitted subject")
    page.fill("#entry-auth-user", "opsuser")
    page.fill("#entry-auth-password", "s3cret")
    _wait_autosaved(page, "Submitted subject")
    _submit(page)
    expect(_sheet(page)).to_be_hidden()
    expect(reader_title(page)).to_have_text("Freshly created entry")
    compose(page)
    expect(page.locator("#entry-subject")).to_have_value("")
    dump = _storage_dump(page)
    for secret in ("opsuser", "s3cret", "plot.png"):
        assert secret not in dump, (secret, dump)
    page.close()


def test_late_draft_response_never_fills_a_later_sheet(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Draft A answering after B opened (or after the sheet closed) never shows A's content."""
    backend = FakeBackend(
        drafts={
            "DA": {**DRAFT, "draft_id": "DA", "subject": "Late draft A", "details": "A body."},
            "DB": {**DRAFT, "draft_id": "DB", "subject": "Draft B", "details": "B body."},
        }
    )
    page, _ = open_panel(chromium_browser, f"{ariel_url}/", backend)
    held = _hold(page, r".*/api/drafts/DA$")

    def release_a() -> None:
        before = len(backend.calls("GET", "/api/drafts/DA"))
        held.pop().fallback()
        _wait_until(
            page, lambda: len(backend.calls("GET", "/api/drafts/DA")) > before, "draft A answered"
        )
        page.wait_for_timeout(300)  # let A's response reach the page

    # A, then B before A answers.
    _go(page, canonical(compose="draft:DA"))
    _wait_until(page, lambda: bool(held), "the draft A request")
    _go(page, canonical(compose="draft:DB"))
    expect(page.locator("#entry-subject")).to_have_value("Draft B")
    release_a()
    expect(page.locator("#entry-subject")).to_have_value("Draft B")
    expect(page.locator("#entry-details")).to_have_value("B body.")
    expect(page.locator("#ariel-compose-banner")).to_have_attribute("data-draft-id", "DB")

    # A, then closed before A answers; a new entry opened afterwards stays empty.
    _go(page, canonical(compose="draft:DA"))
    _wait_until(page, lambda: bool(held), "the second draft A request")
    page.click("#ariel-compose-close")
    expect(_sheet(page)).to_be_hidden()
    release_a()
    compose(page)
    expect(_sheet_title(page)).to_have_text("New entry")
    expect(page.locator("#entry-subject")).to_have_value("")
    expect(page.locator("#entry-details")).to_have_value("")
    expect(page.locator("#ariel-compose-banner")).to_be_hidden()
    page.close()


def test_tag_with_a_quote_adds_and_removes_without_an_error(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Tags holding ``"`` or ``\\`` are added once and removed without a page error."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    compose(page)

    for tag in ['a"b', "c\\d", 'a"b']:
        page.fill("#entry-tags-input", tag)
        page.press("#entry-tags-input", "Enter")
    tags = page.locator("#entry-tags .tag")
    expect(tags).to_have_count(2)
    assert tags.evaluate_all("ts => ts.map(t => t.dataset.value)") == ['a"b', "c\\d"]

    tags.first.locator("[data-tag-remove]").click()
    expect(tags).to_have_count(1)
    assert tags.evaluate_all("ts => ts.map(t => t.dataset.value)") == ["c\\d"]
    tags.first.locator("[data-tag-remove]").click()
    expect(tags).to_have_count(0)
    assert errors == []
    page.close()


def test_key_c_and_r_open_compose_but_not_while_typing(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """``c`` opens a new entry, ``r`` follows up the selection; typed in a field, neither does."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")

    page.keyboard.press("r")  # nothing selected: nothing to follow up
    expect(_sheet(page)).to_be_hidden()
    scratch_input(page)
    page.keyboard.type("cr")
    expect(page.locator("#scratch-input")).to_have_value("cr")
    expect(_sheet(page)).to_be_hidden()
    page.evaluate("document.activeElement.blur()")

    page.keyboard.press("c")
    expect(_sheet(page)).to_be_visible()
    expect(_sheet_title(page)).to_have_text("New entry")
    page.focus("#entry-subject")
    page.keyboard.type("rc")
    expect(page.locator("#entry-subject")).to_have_value("rc")
    expect(_sheet_title(page)).to_have_text("New entry")
    page.keyboard.press("Escape")
    expect(_sheet(page)).to_be_hidden()

    row(page, "E1").click()
    expect(reader_title(page)).to_have_text("Beam dump at injection")
    page.keyboard.press("r")
    expect(_sheet_title(page)).to_have_text("Follow up")
    expect(page.locator("#entry-subject")).to_have_value("Follow-up: Beam dump at injection")
    expect(page.locator("#ariel-compose-parent-ref")).to_have_text("Beam dump at injection")
    page.focus("#entry-details")
    page.keyboard.press("Escape")  # from another field of the sheet
    expect(_sheet(page)).to_be_hidden()
    wait_for_hash(page, canonical(sel="E1"))
    page.close()


def test_key_escape_closes_the_lightbox_before_the_sheet(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """With the sheet open, Esc closes a lightbox first, then the sheet from any of its fields."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/{canonical(sel='E1')}")
    expect(reader_title(page)).to_have_text("Beam dump at injection")
    compose(page)
    expect(_sheet(page)).to_be_visible()

    page.click("#ariel-reader-body [data-lightbox-url]")
    expect(page.locator("#image-lightbox")).to_be_visible()
    page.keyboard.press("Escape")
    expect(page.locator("#image-lightbox")).to_have_count(0)
    expect(_sheet(page)).to_be_visible()

    page.focus("#entry-tags-input")
    page.keyboard.press("Escape")
    expect(_sheet(page)).to_be_hidden()
    page.close()


# ---------------------------------------------------------------------------
# Inspect: the Enhancements tab, Related, the Status coverage, Ask agent
# ---------------------------------------------------------------------------

#: The re-run command a failed card carries (built by the server).
RERUN_COMMAND = "osprey ariel enhance --module text_embedding --entry E1"


def _card(module: str, order: int, state: str, **extra: Any) -> dict[str, Any]:
    """A ``ModuleCard``-shaped card."""
    labels = {m["name"]: m["label"] for m in HEALTHY_CAPABILITIES["enhancement_modules"]}
    return {
        "module": module,
        "label": labels.get(module, module),
        "order": order,
        "feeds": [],
        "state": state,
        "completed_at": None,
        "failed_at": None,
        "marker": None,
        "error": None,
        "attempts": None,
        "command": None,
        "detail": None,
        **extra,
    }


#: E1's cards, listed out of execution order so the tab has to sort them.
E1_ENHANCEMENTS = {
    "entry_id": "E1",
    "modules": [
        _card("figure_caption", 1, "complete", completed_at="2026-08-20T12:05:00-07:00"),
        _card(
            "text_embedding",
            0,
            "failed",
            failed_at="2026-08-20T12:01:00-07:00",
            error="model unavailable",
            attempts=3,
            command=RERUN_COMMAND,
        ),
    ],
}


def _tab(page: Page, tab_id: str):
    """The reader's tab button *tab_id* (``''`` is the entry itself)."""
    return page.locator(f'#ariel-reader-body [data-reader-tab="{tab_id}"]')


def test_enhancements_tab_lists_a_card_per_module_in_order_with_the_failed_command(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """The tab fetches on open only and shows one card per module in order; a failed one its command."""
    backend = FakeBackend(enhancements={"E1": E1_ENHANCEMENTS})
    page, _ = open_panel(chromium_browser, f"{ariel_url}/{canonical(sel='E1')}", backend)
    expect(reader_title(page)).to_have_text("Beam dump at injection")
    assert not backend.calls("GET", "/api/entries/E1/enhancements")  # not before the tab opens

    _tab(page, "enhancements").click()
    expect(_tab(page, "enhancements")).to_have_attribute("aria-selected", "true")
    cards = page.locator("#ariel-reader-body [data-inspector-card]")
    expect(cards).to_have_count(2)
    assert cards.evaluate_all("cs => cs.map(c => c.dataset.module)") == [
        "text_embedding",
        "figure_caption",
    ]
    assert cards.evaluate_all("cs => cs.map(c => c.dataset.state)") == ["failed", "complete"]
    expect(cards.nth(0).locator(".inspector-card-title")).to_have_text("Text Embedding")
    expect(cards.nth(1).locator(".inspector-card-title")).to_have_text("Figure Caption")

    failed = cards.nth(0)
    expect(failed.locator("[data-inspector-command] code")).to_have_text(RERUN_COMMAND)
    expect(failed.locator("[data-inspector-copy]")).to_be_visible()
    expect(failed).to_contain_text("model unavailable")
    expect(cards.nth(1).locator("[data-inspector-command]")).to_have_count(0)
    # The inspector only shows the command: no control re-runs a module.
    expect(page.locator("#ariel-reader-body [data-entry-inspector] button")).to_have_count(1)
    wait_for_call(page, backend, "GET", "/api/entries/E1/enhancements")
    page.close()


def test_enhancements_tab_says_when_no_module_is_enabled(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """An entry with no enabled module shows the tab's empty state, not a card."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/{canonical(sel='E2')}")
    expect(reader_title(page)).to_have_text("Vacuum spike sector 3")
    _tab(page, "enhancements").click()
    expect(page.locator("#ariel-reader-body .inspector-empty")).to_have_text(
        "No enhancement modules are enabled."
    )
    expect(page.locator("#ariel-reader-body [data-inspector-card]")).to_have_count(0)
    page.close()


def test_key_e_opens_the_enhancements_tab_and_the_entry_tab_returns(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """``e`` switches the reader to Enhancements, the Entry tab back, ``e`` again re-opens it."""
    backend = FakeBackend(enhancements={"E1": E1_ENHANCEMENTS})
    page, _ = open_panel(chromium_browser, f"{ariel_url}/{canonical(sel='E1')}", backend)
    expect(reader_title(page)).to_have_text("Beam dump at injection")
    grid = page.locator("#ariel-reader-body .entry-detail-grid")
    panel = page.locator("#ariel-reader-body [data-reader-tab-panel]")
    cards = page.locator("#ariel-reader-body [data-inspector-card]")

    page.keyboard.press("e")
    expect(_tab(page, "enhancements")).to_have_attribute("aria-selected", "true")
    expect(_tab(page, "")).to_have_attribute("aria-selected", "false")
    expect(panel).to_be_visible()
    expect(grid).to_be_hidden()
    expect(cards).to_have_count(2)

    _tab(page, "").click()
    expect(_tab(page, "")).to_have_attribute("aria-selected", "true")
    expect(grid).to_be_visible()
    expect(panel).to_be_hidden()

    page.evaluate("document.activeElement.blur()")
    page.keyboard.press("e")
    expect(_tab(page, "enhancements")).to_have_attribute("aria-selected", "true")
    expect(cards).to_have_count(2)

    # Typed in a field, ``e`` is text, not the tab.
    _tab(page, "").click()
    scratch_input(page)
    page.keyboard.type("e")
    expect(page.locator("#scratch-input")).to_have_value("e")
    expect(_tab(page, "")).to_have_attribute("aria-selected", "true")
    page.close()


#: Older entries the Related fixtures point at.
RELATED_ENTRIES = [
    make_entry(f"R{i}", f"Related entry {i}", timestamp=f"2026-08-1{i}T08:00:00-07:00")
    for i in range(1, 7)
]


def test_related_shows_at_most_five_rows_and_never_the_entry_itself(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """Related asks for five, lists at most five rows without the shown entry, and selects on click."""
    backend = FakeBackend(
        entries=[*DEFAULT_ENTRIES, *RELATED_ENTRIES],
        related={"E1": ["E1", "R1", "R2", "R3", "R4", "R5", "R6"]},
    )
    page, _ = open_panel(chromium_browser, f"{ariel_url}/{canonical(sel='E1')}", backend)
    expect(reader_title(page)).to_have_text("Beam dump at injection")

    section = page.locator("#ariel-reader-body [data-entry-related]")
    rows = section.locator("[data-thread-entry]")
    expect(section.locator("h4")).to_have_text("Related")
    expect(rows).to_have_count(5)
    shown = rows.evaluate_all("rs => rs.map(r => r.dataset.threadEntry)")
    assert shown == ["R1", "R2", "R3", "R4", "R5"], shown
    request = wait_for_call(page, backend, "GET", "/api/entries/E1/related")
    assert request.query.get("limit") == ["5"], request.query

    rows.nth(1).click()
    expect(reader_title(page)).to_have_text("Related entry 2")
    wait_for_hash(page, canonical(sel="R2"))
    page.close()


def test_related_is_hidden_when_text_embedding_is_off(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """With text embedding disabled the reader shows no Related section and raises no toast."""
    backend = FakeBackend(
        entries=[*DEFAULT_ENTRIES, *RELATED_ENTRIES],
        related={"E1": ["R1", "R2"]},
        related_state="disabled",
    )
    page, _ = open_panel(chromium_browser, f"{ariel_url}/{canonical(sel='E1')}", backend)
    expect(reader_title(page)).to_have_text("Beam dump at injection")
    wait_for_call(page, backend, "GET", "/api/entries/E1/related")
    page.wait_for_load_state("networkidle")
    expect(page.locator("#ariel-reader-body [data-entry-related]")).to_have_count(0)
    expect(page.locator("#ariel-reader-body")).not_to_contain_text("Related")
    expect(_toast(page, "alert")).to_have_count(0)
    page.close()


def test_status_drawer_lists_every_enabled_module_with_its_coverage(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """The Status drawer's coverage table has one row per enabled module, labelled, with counts."""
    page, backend = open_panel(chromium_browser, f"{ariel_url}/")
    page.evaluate("document.getElementById('status-drawer').open()")
    expect(page.locator("osprey-drawer#status-drawer")).to_have_attribute("open", re.compile(".*"))
    wait_for_call(page, backend, "GET", "/api/status")

    rows = page.locator("#status-content table.coverage-table tbody tr[data-module]")
    expect(rows).to_have_count(2)
    assert rows.evaluate_all("rs => rs.map(r => r.dataset.module)") == [
        "text_embedding",
        "figure_caption",
    ]
    expect(rows.nth(0).locator("td").first).to_have_text("Text Embedding")
    expect(rows.nth(1).locator("td").first).to_have_text("Figure Caption")
    expect(rows.nth(0).locator('td[data-state="complete"]')).to_have_text("1,200")
    expect(rows.nth(0).locator('td[data-state="failed"]')).to_have_text("1")
    expect(rows.nth(1).locator('td[data-state="n/a"]')).to_have_text("832")
    # Never ran and ran-but-unfinished are two columns, never one "pending".
    expect(rows.nth(0).locator('td[data-state="not_run"]')).to_have_text("7")
    expect(rows.nth(0).locator('td[data-state="pending"]')).to_have_text("30")
    expect(page.locator("#status-content table.coverage-table thead")).to_contain_text("Not run")
    page.close()


# ---------------------------------------------------------------------------
# Compose focus and the one banner
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("closer", ["close", "discard", "escape"])
def test_compose_returns_focus_to_its_opener(
    ariel_url: str, chromium_browser: Browser, closer: str
) -> None:
    """Close, Discard and Esc each hand focus back to where the sheet was opened from."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/")
    opener = page.locator("#ariel-mode-chip")
    opener.focus()
    page.keyboard.press("c")
    expect(_sheet(page)).to_be_visible()
    page.focus("#entry-subject")
    expect(opener).not_to_be_focused()

    if closer == "close":
        page.click("#ariel-compose-close")
    elif closer == "discard":
        page.click("#create-cancel-btn")
    else:
        page.keyboard.press("Escape")
    expect(_sheet(page)).to_be_hidden()
    expect(opener).to_be_focused()
    page.close()


def test_follow_up_returns_focus_to_the_reader_button_on_close(
    ariel_url: str, chromium_browser: Browser
) -> None:
    """A follow-up opened from the reader gives focus back to its Follow up button."""
    page, _ = open_panel(chromium_browser, f"{ariel_url}/{canonical(sel='E1')}")
    expect(reader_title(page)).to_have_text("Beam dump at injection")
    opener = page.locator("#ariel-reader-body [data-follow-up]")
    opener.click()
    expect(_sheet_title(page)).to_have_text("Follow up")
    page.focus("#entry-details")

    page.keyboard.press("Escape")
    expect(_sheet(page)).to_be_hidden()
    expect(opener).to_be_focused()
    page.close()


def test_compose_sheet_shows_exactly_one_banner(ariel_url: str, chromium_browser: Browser) -> None:
    """A staged draft shows one banner, still one after another draft opens, none for a new entry."""
    backend = FakeBackend(
        drafts={
            "D1": dict(DRAFT),
            "DB": {**DRAFT, "draft_id": "DB", "subject": "Draft B", "details": "B body."},
        }
    )
    page, _ = open_panel(chromium_browser, f"{ariel_url}/{canonical(compose='draft:D1')}", backend)
    shown = _sheet(page).locator("[class*='banner']:visible, [id*='banner']:visible")

    expect(page.locator("#entry-subject")).to_have_value("Drafted by the agent")
    expect(page.locator("#ariel-compose-banner")).to_be_visible()
    expect(page.locator("#ariel-compose-banner")).to_have_attribute("data-draft-id", "D1")
    expect(shown).to_have_count(1)
    expect(page.locator("#draft-banner")).to_have_count(0)

    _go(page, canonical(compose="draft:DB"))
    expect(page.locator("#entry-subject")).to_have_value("Draft B")
    expect(page.locator("#ariel-compose-banner")).to_have_attribute("data-draft-id", "DB")
    expect(shown).to_have_count(1)

    page.click("#ariel-compose-close")
    expect(_sheet(page)).to_be_hidden()
    compose(page)
    expect(_sheet_title(page)).to_have_text("New entry")
    expect(shown).to_have_count(0)
    page.close()
