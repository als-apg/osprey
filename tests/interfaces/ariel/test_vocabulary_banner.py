"""Real-browser suite for ARIEL's vocabulary surfaces: expansion strip + config banners.

Three things only a live page can prove, all of them wired in
``static/js/search.js`` from payloads the backend really produces:

- the query-expansion strip (``[data-testid="expansion-banner"]``) renders one
  pair per ``expanded_terms`` group in Expert view, and is absent in Simple
  view and whenever the response expanded nothing;
- a vocabulary file that could not be loaded, with no database behind the
  panel, still puts ``ariel.vocabulary.path`` and its remedy on screen inside
  ``#config-invalid-banner`` and leaves ``#search-btn`` disabled;
- the warning class (a pre-existing ``validate()`` error, valid vocabulary)
  renders ``#config-warning-banner`` instead and leaves the button usable.

The panel is booted the way ``test_load_smokes.py::_launch_ariel`` boots it —
a real ``create_app()`` on a real uvicorn server — with a real ``config.yml``
written into ``tmp_path`` and ``create_ariel_service`` patched, so the
configuration classification, the ``/api/capabilities`` payload and the
remedy text are all the real ones while no PostgreSQL is required. Only
``POST /api/search`` is faked, at the network layer (``page.route``), because
a search response carrying ``expanded_terms`` is exactly what a database-less
panel cannot produce.

Run:
    uv run pytest tests/interfaces/ariel/test_vocabulary_banner.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from tests.interfaces.conftest import _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from playwright.sync_api import Browser, Page, Route

# Guarded only so collection succeeds where playwright is not installed; the
# skip itself comes from the shared ``chromium_browser`` fixture, which never
# lets a test body run without a browser.
try:
    from playwright.sync_api import expect
except ImportError:  # pragma: no cover
    expect = None  # type: ignore[assignment]

pytestmark = [pytest.mark.browser, pytest.mark.slow]

# Unreachable on purpose: `database.uri` is a required key, and every test here
# runs with no database at all (`create_ariel_service` is patched out).
DSN = "postgresql://nobody@127.0.0.1:1/none"

VOCABULARY_YML = """
concepts:
  - canonical: troubleshoot
    kind: shorthand
    forms:
      - t/s
      - ts

  - canonical: beam position monitor
    kind: acronym
    forms:
      - bpm
      - bpms
"""

# The operator action the panel must show for a broken vocabulary. Pinned as a
# literal (not imported) because it is user-facing text a Playwright test is
# meant to hold still.
VOCABULARY_REMEDY = (
    "disable ariel.vocabulary.enabled or repoint ariel.vocabulary.path, then restart"
)


# ---------------------------------------------------------------------------
# Panel launcher
# ---------------------------------------------------------------------------


class _ServiceDouble:
    """A stand-in for a live ARIEL service carrying the real parsed config.

    ``GET /api/capabilities`` needs nothing from the service but ``.config``,
    so this is enough to get the genuine backend payload (mode tabs, shared
    parameters, the ``vocabulary`` block) without a database.

    Deliberately an explicit class rather than a bare ``AsyncMock``: only the
    attributes the lifespan and the capabilities route really touch exist, so a
    route that starts reading live service state fails loudly here instead of
    quietly being handed a fabricated Mock value.
    """

    def __init__(self, config: Any) -> None:
        """Store the ``ARIELConfig`` the lifespan parsed."""
        self.config = config

    async def health_check(self) -> tuple[bool, str]:
        """Report the double as healthy (the lifespan logs this)."""
        return True, "Service healthy"

    async def __aenter__(self) -> _ServiceDouble:
        """Enter the async context the service protocol defines."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Exit the async context (the lifespan calls this on shutdown)."""
        return None


@contextmanager
def _launch_ariel_with_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ariel_section: dict[str, Any],
    *,
    with_service: bool,
) -> Iterator[str]:
    """Boot standalone ARIEL against a real ``config.yml``, with no database.

    Mirrors ``test_load_smokes.py::_launch_ariel`` (cwd isolation + the real
    ``create_app()`` on a real server); the config file is written first and
    handed to the factory explicitly, so a relative ``ariel.vocabulary.path``
    resolves against ``tmp_path`` and no stray ``/app/config.yml`` or
    ``CONFIG_FILE`` from the environment can be picked up instead.

    Args:
        tmp_path: Per-test directory holding config.yml (and its data/).
        monkeypatch: Used to isolate cwd and the CONFIG_FILE override.
        ariel_section: The ``ariel:`` block of the config file.
        with_service: True to hand the lifespan a service double (the database
            is up), False to make service construction fail (database down).

    Yields:
        The live panel's base URL.
    """
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump({"ariel": ariel_section}))
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.chdir(tmp_path)

    if with_service:
        create_service = AsyncMock(side_effect=lambda config: _ServiceDouble(config))
    else:
        create_service = AsyncMock(side_effect=RuntimeError("connection refused"))

    from osprey.interfaces.ariel.app import create_app

    with patch("osprey.services.ariel_search.create_ariel_service", create_service):
        app = create_app(config_file)
        with _run_app_server(app) as base_url:
            yield base_url


def _write_vocabulary(tmp_path: Path) -> None:
    """Write a valid two-concept vocabulary at the configured relative path."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "vocabulary.yml").write_text(VOCABULARY_YML)


# ---------------------------------------------------------------------------
# Faked search responses
# ---------------------------------------------------------------------------

SIMPLE_TITLE = "Simple-view search hit"

TWO_GROUPS = [
    {"original": "ts", "alternatives": ["troubleshoot", "timing system"]},
    {"original": "bpm", "alternatives": ["beam position monitor"]},
]


def _entry(entry_id: str, subject: str) -> dict[str, Any]:
    """An ``EntryResponse``-shaped hit whose id and title are unique per response.

    Every faked response carries one of these so a test can wait for the render
    that response produced: ``search.js`` wipes ``#search-results`` into a
    loading overlay before the fetch, and ``index.html`` ships a static
    "Ready to Search" ``.empty-state`` inside the same container, so neither an
    empty container nor an empty state proves a response was ever drawn.

    Args:
        entry_id: Shown verbatim in the Expert card's ``.entry-id`` span.
        subject: First line of ``raw_text``; the Simple card's title.

    Returns:
        One entry of a ``SearchResponse`` body.
    """
    return {
        "entry_id": entry_id,
        "source_system": "olog",
        "timestamp": "2026-08-20T12:00:00-07:00",
        "author": "operator",
        "raw_text": f"{subject}\nBody of {entry_id}.",
        "attachments": [],
        "metadata": {},
        "created_at": "2026-08-20T12:00:00-07:00",
        "updated_at": None,
        "summary": None,
        "keywords": [],
        "score": None,
        "highlights": [],
    }


def _search_response(
    expanded_terms: list[dict[str, Any]], entries: list[dict[str, Any]]
) -> dict[str, Any]:
    """A ``SearchResponse``-shaped body carrying the given hits and expansions."""
    return {
        "entries": entries,
        "answer": None,
        "sources": [],
        "search_modes_used": ["keyword"],
        "reasoning": "",
        "total_results": len(entries),
        "execution_time_ms": 4,
        "diagnostics": [],
        "expanded_terms": expanded_terms,
    }


def _route_search(page: Page, state: dict[str, list[dict[str, Any]]]) -> None:
    """Answer ``POST /api/search`` from ``state['expanded_terms']`` and ``state['entries']``.

    The panel has no database here, so the one thing it cannot produce is a
    search response — it is faked at the network layer, leaving every other
    request (capabilities included) to the real backend. Both keys are read per
    request, so a test can change them and search again.

    Args:
        page: The page whose network is intercepted.
        state: Mutable holder of the ``expanded_terms`` and ``entries`` to
            answer with.
    """

    def handler(route: Route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(_search_response(state["expanded_terms"], state["entries"])),
        )

    page.route("**/api/search", handler)


def _run_search(page: Page, query: str) -> None:
    """Type *query* into the search box and submit it."""
    page.fill("#search-input", query)
    page.click("#search-btn")


# ---------------------------------------------------------------------------
# Config sections
# ---------------------------------------------------------------------------


def _invalid_section() -> dict[str, Any]:
    """Vocabulary enabled, pointing at a file that does not exist."""
    return {
        "database": {"uri": DSN},
        "search_modules": {"keyword": {"enabled": True}},
        "vocabulary": {"enabled": True, "path": "data/missing.yml"},
    }


def _warning_section() -> dict[str, Any]:
    """Valid vocabulary, but semantic search is enabled without a model."""
    return {
        "database": {"uri": DSN},
        "search_modules": {"keyword": {"enabled": True}, "semantic": {"enabled": True}},
        "vocabulary": {"enabled": True, "path": "data/vocabulary.yml"},
    }


def _healthy_section() -> dict[str, Any]:
    """Valid vocabulary, nothing to complain about."""
    return {
        "database": {"uri": DSN},
        "search_modules": {"keyword": {"enabled": True}},
        "vocabulary": {"enabled": True, "path": "data/vocabulary.yml"},
    }


# ---------------------------------------------------------------------------
# Configuration banners
# ---------------------------------------------------------------------------


def test_missing_vocabulary_blocks_the_search_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chromium_browser: Browser
) -> None:
    """No database + unloadable vocabulary: the page names the key and blocks search."""
    with _launch_ariel_with_config(
        tmp_path, monkeypatch, _invalid_section(), with_service=False
    ) as base_url:
        page = chromium_browser.new_page()
        page.goto(base_url, wait_until="load")

        banner = page.locator("#config-invalid-banner")
        expect(banner).to_be_visible(timeout=10_000)
        expect(banner).to_have_attribute("role", "alert")
        expect(banner).to_have_class("config-banner config-banner-invalid")
        expect(banner).to_have_attribute("data-testid", "config-banner")

        # The failing key and the operator action, both on screen.
        expect(banner).to_contain_text("ariel.vocabulary.path")
        expect(banner).to_contain_text(VOCABULARY_REMEDY)

        # The banner is the form's first child, not a replacement for it: the
        # controls stay in the DOM so the operator can see search is blocked.
        assert (
            page.evaluate("document.getElementById('search-form').firstElementChild.id")
            == "config-invalid-banner"
        )
        expect(page.locator("#search-btn")).to_be_disabled()
        expect(page.locator("#search-input")).to_be_disabled()

        expect(page.locator("#config-warning-banner")).to_have_count(0)

        page.close()


def test_configuration_warning_leaves_search_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chromium_browser: Browser
) -> None:
    """A pre-existing validate() error warns, but never disables the form."""
    _write_vocabulary(tmp_path)
    with _launch_ariel_with_config(
        tmp_path, monkeypatch, _warning_section(), with_service=True
    ) as base_url:
        page = chromium_browser.new_page()
        page.goto(base_url, wait_until="load")

        banner = page.locator("#config-warning-banner")
        expect(banner).to_be_visible(timeout=10_000)
        expect(banner).to_have_class("config-banner config-banner-warning")
        expect(banner).to_have_attribute("data-testid", "config-banner")
        expect(banner).to_contain_text("search_modules.semantic.model")
        # The vocabulary loaded, so its key must not be blamed here.
        expect(banner).not_to_contain_text("ariel.vocabulary.path")

        expect(page.locator("#config-invalid-banner")).to_have_count(0)
        expect(page.locator("#search-btn")).to_be_enabled()
        expect(page.locator("#search-input")).to_be_enabled()

        page.close()


# ---------------------------------------------------------------------------
# Expansion strip
# ---------------------------------------------------------------------------


def test_expansion_strip_renders_one_pair_per_group_in_expert_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chromium_browser: Browser
) -> None:
    """Expert view shows ``original -> canonical, canonical`` once per group."""
    _write_vocabulary(tmp_path)
    with _launch_ariel_with_config(
        tmp_path, monkeypatch, _healthy_section(), with_service=True
    ) as base_url:
        page = chromium_browser.new_page()
        state: dict[str, list[dict[str, Any]]] = {
            "expanded_terms": TWO_GROUPS,
            "entries": [_entry("EXPANDED-1", "First search hit")],
        }
        _route_search(page, state)
        page.goto(base_url, wait_until="load")

        # Expert is the default view; no pick needed to be in it.
        expect(page.locator("html")).to_have_attribute("data-ui-mode", "expert", timeout=10_000)
        expect(page.locator("#config-invalid-banner")).to_have_count(0)

        _run_search(page, "ts bpm")

        # The hit only THIS response carries, so everything below is asserted
        # against the rendered results, not the loading overlay.
        expect(page.locator(".entry-card .entry-id")).to_have_text("EXPANDED-1", timeout=10_000)

        strip = page.locator('[data-testid="expansion-banner"]')
        expect(strip).to_be_visible()
        pairs = strip.locator(".expansion-pair")
        expect(pairs).to_have_count(2)
        # One pair per GROUP: an original with two canonicals stays one pair.
        expect(pairs.nth(0)).to_have_text("ts → troubleshoot, timing system")
        expect(pairs.nth(1)).to_have_text("bpm → beam position monitor")

        # A response that expanded nothing renders no strip at all. The second
        # response carries its own entry id, and waiting for that id is what
        # makes the count of 0 meaningful: search.js empties #search-results
        # into a loading overlay the instant the button is clicked, so an
        # unanchored count would latch onto that transient state instead.
        state["expanded_terms"] = []
        state["entries"] = [_entry("NOT-EXPANDED-1", "Second search hit")]
        _run_search(page, "nothing to expand")
        expect(page.locator(".entry-card .entry-id")).to_have_text("NOT-EXPANDED-1", timeout=10_000)
        expect(strip).to_have_count(0)

        page.close()


def test_expansion_strip_absent_in_simple_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chromium_browser: Browser
) -> None:
    """Simple view omits the search-mechanics chrome, expansions included."""
    _write_vocabulary(tmp_path)
    with _launch_ariel_with_config(
        tmp_path, monkeypatch, _healthy_section(), with_service=True
    ) as base_url:
        state: dict[str, list[dict[str, Any]]] = {
            "expanded_terms": TWO_GROUPS,
            "entries": [_entry("SIMPLE-1", SIMPLE_TITLE)],
        }

        page = chromium_browser.new_page()
        _route_search(page, state)
        # ?mode= is the one-shot view deep-link the display menu also writes.
        page.goto(f"{base_url}/?mode=simple", wait_until="load")
        expect(page.locator("html")).to_have_attribute("data-ui-mode", "simple", timeout=10_000)

        _run_search(page, "ts bpm")

        # The response's own hit really rendered — not index.html's static
        # "Ready to Search" placeholder, which sits inside #search-results
        # before any search and would make a bare .empty-state check vacuous.
        # The strip is the only thing missing.
        expect(page.locator(".entry-card-simple-title")).to_have_text(SIMPLE_TITLE, timeout=10_000)
        expect(page.locator('[data-testid="expansion-banner"]')).to_have_count(0)

        page.close()

        # Control, same response: Expert view DOES draw the strip, so the
        # absence above is the view mode and not a response that expanded
        # nothing. A fresh page is a fresh context, hence the default view.
        expert_page = chromium_browser.new_page()
        _route_search(expert_page, state)
        expert_page.goto(base_url, wait_until="load")
        expect(expert_page.locator("html")).to_have_attribute(
            "data-ui-mode", "expert", timeout=10_000
        )

        _run_search(expert_page, "ts bpm")

        expect(expert_page.locator(".entry-card .entry-id")).to_have_text(
            "SIMPLE-1", timeout=10_000
        )
        expect(expert_page.locator('[data-testid="expansion-banner"]')).to_have_count(1)

        expert_page.close()
