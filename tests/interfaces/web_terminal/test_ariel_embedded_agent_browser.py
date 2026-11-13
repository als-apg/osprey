"""Browser tests: the agent drives an ARIEL panel embedded in the real hub.

The whole loop is real except the model: the web-terminal hub (with its chat
surface, the Claude SDK faked at the operator-session seam), a real ARIEL app
served behind the hub's ``/panel/ariel`` proxy, a Postgres started by
testcontainers, and the ARIEL MCP tools (``entry_open``, ``entry_create``,
``entry_publish``) called in-process on the generic JSON adapter. A tool call
reaches the browser the way it does in a deployment: ``entry_open`` through
the hub's ``open`` panel op and the in-place ``canopy-host-navigate``
handshake, ``entry_create`` and ``entry_publish`` through
``/api/agent-activity`` relayed into the iframe as ``osprey-agent-activity``.

ARIEL's ``/api/capabilities`` is delayed in the fixture. The router installs
its navigate listener only after capabilities answer, so the hub's load-time
probe lands before anyone listens and the in-place flag is earned by ARIEL's
own unsolicited ``canopy-host-navigate-ready`` — the boot order a deployment
has, not a fast path a test happens to win.

Coverage:

  (1) ``entry_open`` on a row of the current search keeps the query, the list
      and its scroll, selects the entry, flashes the row; a repeat re-flashes.
  (2) ``entry_open`` on an entry the search does not list opens it in the
      reader with the "not shown in the list" notice, query kept.
  (3) ``entry_create`` (direct write) while the list is scrolled adds a
      "1 new" pill and moves nothing; the pill's click brings the row in.
  (4) a real ``entry_publish`` whose facility entry lands in view flashes it,
      and the pill is unchanged (a publish never counts as new).
  (5) "Ask agent" lands the citation in the composer unsent; while a turn
      streams it waits, and lands after the turn, still unsent.
  (6) after the operator reloads the hub, ``entry_open`` reaches the new page,
      shows the entry, and its answer says ``shown: true``.

Flashes self-clean on animationend, so they are observed through a
document-start MutationObserver installed in every frame rather than by
polling.

Run:
    uv run pytest tests/interfaces/web_terminal/test_ariel_embedded_agent_browser.py -m browser -rs

A missing Docker daemon skips, or fails under ``ARIEL_TEST_REQUIRE_DB=1``;
a missing chromium skips. A skipped run of this file is not a pass.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import threading
import time
from collections.abc import Coroutine, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import requests
import yaml
from playwright.sync_api import Frame, FrameLocator, Page, expect

from osprey.interfaces._serving import run_app_server
from tests._container_support import (
    is_docker_available,
    require_container_runtime,
    start_or_fail,
    stop_quietly,
)
from tests._session_database import answers
from tests.interfaces._browser import wait_for_page_ready
from tests.interfaces.web_terminal._chat_server import (
    OBSERVED_PROMPTS,
    PLANS,
    live_chat_server,
)

# xdist_group("docker"): the module's Postgres container, on the worker that
# starts every other container, so the sweep does not load the runtime from two
# workers at once.
pytestmark = [pytest.mark.browser, pytest.mark.slow, pytest.mark.xdist_group("docker")]

#: Seconds ARIEL's ``/api/capabilities`` is held before it answers.
CAPABILITIES_DELAY_S = 1.5

#: Rows the search ``orbit`` matches (more than one page of 20).
ORBIT_ROWS = 26
#: Rows the search does not match; the newest seeded rows, so browse page 1
#: starts with them.
VACUUM_ROWS = 6

_ARIEL_IFRAME = 'iframe.panel-iframe[data-panel-id="ariel"]'
_OP = "#operator-container"
_INPUT = f"{_OP} .op-input-area textarea"
_LIST_BODY = "#ariel-list-body"
_ROWS = "#ariel-list [data-entry-id]"
_PILL = ".ariel-new-pill"
_READER_TITLE = "#ariel-reader-body .entry-detail-title"
_NOT_LISTED = "This entry is not shown in the list."

#: Installed in every frame before any page script. ``__flashed`` lists the
#: entry id of every row the moment it gains ``agent-flash``; ``__navReady``
#: counts the ``canopy-host-navigate-ready`` messages the hub window received
#: since a panel iframe last loaded, the rule the hub itself applies: a ready
#: from the document an iframe is leaving does not count for the next one.
_PROBES = """
(function () {
  window.__flashed = [];
  window.__navReady = 0;
  window.addEventListener('message', function (ev) {
    if (ev.data && ev.data.type === 'canopy-host-navigate-ready') { window.__navReady += 1; }
  });
  document.addEventListener('load', function (ev) {
    const t = ev.target;
    if (t && t.matches && t.matches('iframe.panel-iframe')) { window.__navReady = 0; }
  }, true);
  function attach(el) {
    new MutationObserver(function (muts) {
      // flashElement removes then re-adds the class in one task, so one flash
      // is two records in one batch: count each element once per batch.
      const seen = new Set();
      for (const m of muts) {
        const t = m.target;
        if (seen.has(t)) continue;
        if (t && t.classList && t.classList.contains('agent-flash') && t.dataset
            && t.dataset.entryId) {
          seen.add(t);
          window.__flashed.push(t.dataset.entryId);
        }
      }
    }).observe(el, { attributes: true, attributeFilter: ['class'], subtree: true });
  }
  if (document.documentElement) {
    attach(document.documentElement);
  } else {
    new MutationObserver(function (_m, obs) {
      if (document.documentElement) { obs.disconnect(); attach(document.documentElement); }
    }).observe(document, { childList: true, subtree: true });
  }
})();
"""

#: True once the iframe *sel* holds, fully loaded, the document its ``src``
#: names (path and query; the router rewrites only the hash): no load of
#: another document is pending behind the one on screen.
_IFRAME_AT_ITS_SRC = """
(sel) => {
  const f = document.querySelector(sel);
  if (!f || !f.getAttribute('src')) return false;
  let doc = null;
  try { doc = f.contentDocument; } catch (e) { return false; }
  if (!doc || doc.readyState !== 'complete') return false;
  const want = new URL(f.src, location.href);
  const have = new URL(doc.URL);
  return want.pathname === have.pathname && want.search === have.search;
}
"""


# ---------------------------------------------------------------------------
# Postgres (testcontainers), one container per module, one database per test
# ---------------------------------------------------------------------------


def _skip_or_fail(reason: str) -> None:
    """Skip for want of Postgres, or fail when the gate demands a database."""
    if os.environ.get("ARIEL_TEST_REQUIRE_DB") == "1":
        pytest.fail(f"{reason} (ARIEL_TEST_REQUIRE_DB=1)")
    pytest.skip(reason)


@pytest.fixture(scope="module")
def ariel_postgres_url() -> Iterator[str]:
    """The admin DSN of a pgvector Postgres owned by this module."""
    if not is_docker_available():
        _skip_or_fail("docker daemon is not reachable")
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        _skip_or_fail("testcontainers[postgres] not installed")
    require_container_runtime()
    container, _port = start_or_fail(
        lambda: PostgresContainer(
            image="ankane/pgvector:latest",
            username="ariel",
            password="ariel",
            dbname="ariel_embedded",
        ),
        "postgres (embedded ARIEL browser tests)",
        5432,
    )
    try:
        url = container.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
        require_container_runtime(lambda: answers(url), "postgres", container=container)
        yield url
    finally:
        stop_quietly(container)


def _fresh_database(admin_url: str, name: str) -> str:
    """Create database *name* beside the admin one and return its DSN."""
    import psycopg

    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}"')
        conn.execute(f'CREATE DATABASE "{name}"')
    return f"{admin_url.rsplit('/', 1)[0]}/{name}"


# ---------------------------------------------------------------------------
# One event loop for every coroutine the test runs: the MCP context caches its
# service (and its pool) on the loop that first built it.
# ---------------------------------------------------------------------------


class _Loop:
    """A private event loop running on a daemon thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro: Coroutine[Any, Any, Any], timeout: float = 60) -> Any:
        """Run *coro* on the loop and return its result."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def close(self) -> None:
        """Stop the loop and join its thread."""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=10)
        self.loop.close()


def _seed_entries(now: datetime) -> list[dict[str, Any]]:
    """Seeded rows, newest first: the vacuum rows, then the orbit rows.

    Every timestamp is in the past, so a row an MCP tool writes now sorts
    above all of them on browse page 1.
    """
    entries: list[dict[str, Any]] = []
    for i in range(VACUUM_ROWS):
        entries.append(
            {
                "entry_id": f"vac-{i:02d}",
                "subject": f"Vacuum gauge {i} reading logged",
                "details": "Sector pressure nominal after the bake.",
            }
        )
    for i in range(ORBIT_ROWS):
        entries.append(
            {
                "entry_id": f"orb-{i:02d}",
                "subject": f"Orbit correction {i} applied",
                "details": "Global orbit feedback converged after the kick.",
            }
        )
    for index, entry in enumerate(entries):
        entry["timestamp"] = now - timedelta(hours=index + 1)
    return entries


async def _migrate_and_seed(config_dict: dict[str, Any], entries: list[dict[str, Any]]) -> None:
    """Bring the database to today's schema and insert *entries*."""
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database import ARIELRepository
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import run_migrations

    config = ARIELConfig.from_dict(json.loads(json.dumps(config_dict)))
    async with await create_connection_pool(config.database) as pool:
        await run_migrations(pool, config)
        repository = ARIELRepository(pool, config)
        for entry in entries:
            await repository.upsert_entry(
                {
                    "entry_id": entry["entry_id"],
                    "source_system": "Generic JSON",
                    "timestamp": entry["timestamp"],
                    "author": "operator",
                    "raw_text": f"{entry['subject']}\n\n{entry['details']}",
                    "attachments": [],
                    "metadata": {},
                    "created_at": entry["timestamp"],
                    "updated_at": entry["timestamp"],
                }
            )


# ---------------------------------------------------------------------------
# The stack: hub + ARIEL + Postgres + MCP context
# ---------------------------------------------------------------------------


@dataclass
class Stack:
    """Everything one test drives."""

    base_url: str
    hub: Any
    ariel_app: Any
    page: Page
    loop: _Loop
    entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def frame_locator(self) -> FrameLocator:
        """The ARIEL iframe, for locators."""
        return self.page.frame_locator(_ARIEL_IFRAME)

    def frame(self) -> Frame:
        """The ARIEL iframe's frame, for evaluation."""
        handle = self.page.locator(_ARIEL_IFRAME).element_handle(timeout=15_000)
        assert handle is not None
        frame = handle.content_frame()
        assert frame is not None
        return frame

    def tool(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Call one ARIEL MCP tool in-process and return its parsed answer."""
        from osprey.mcp_server.ariel.tools import entry as entry_tools
        from osprey.mcp_server.ariel.tools import publish as publish_tools

        module = publish_tools if name == "entry_publish" else entry_tools
        tool = getattr(module, name)
        fn = getattr(tool, "fn", tool)
        raw = self.loop.run(fn(**kwargs))
        answer = json.loads(raw)
        assert "error" not in answer and answer.get("status") != "error", answer
        return answer

    def flashed(self) -> list[str]:
        """Entry ids of every row that gained ``agent-flash`` in the iframe."""
        return self.frame().evaluate("() => window.__flashed.slice()")

    def scroll_top(self) -> float:
        """The list body's scroll offset."""
        return self.frame().evaluate(f"() => document.querySelector('{_LIST_BODY}').scrollTop")

    def row_ids(self) -> list[str]:
        """The entry ids of the painted rows, in order."""
        return self.frame().evaluate(
            f"() => Array.from(document.querySelectorAll('{_ROWS}'), (r) => r.dataset.entryId)"
        )

    def hash(self) -> str:
        """The iframe's location hash."""
        return self.frame().evaluate("() => location.hash")

    def wait_settled(self) -> None:
        """Wait for the list to be painted and idle."""
        expect(self.frame_locator.locator(_LIST_BODY)).to_have_attribute(
            "aria-busy", "false", timeout=20_000
        )


def _open_panel(url: str) -> None:
    """The agent's ``open`` op for ARIEL at *url*, retried until a dock client exists."""
    from osprey.mcp_server.http import panel_op

    deadline = time.monotonic() + 20
    while True:
        result = panel_op("open", {"member": "ariel", "url": url})
        if result.get("ok"):
            return
        if result.get("reason") == "no_dock_client" and time.monotonic() < deadline:
            time.sleep(0.2)
            continue
        raise AssertionError(f"open ariel at {url} failed: {result}")


def _delay_capabilities(app: Any) -> None:
    """Hold ARIEL's ``/api/capabilities`` so the router boots late."""
    app.state.capabilities_delayed = 0

    @app.middleware("http")
    async def _slow_capabilities(request, call_next):
        if request.url.path.rstrip("/").endswith("/api/capabilities"):
            app.state.capabilities_delayed += 1
            await asyncio.sleep(CAPABILITIES_DELAY_S)
        return await call_next(request)


@contextlib.contextmanager
def _ariel_stack(
    tmp_path: Path,
    browser: Any,
    admin_url: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    initial_hash: str,
) -> Iterator[Stack]:
    """Serve hub + ARIEL over a fresh database and open ARIEL at *initial_hash*.

    Returns once ARIEL has painted its list and announced in-place
    navigation to the hub, so the next ``entry_open`` is an in-place one.
    """
    from osprey.interfaces.ariel.app import create_app as create_ariel_app
    from osprey.mcp_server.ariel.server_context import (
        get_ariel_context,
        initialize_ariel_context,
        reset_ariel_context,
    )
    from osprey_connectors.workspace import reset_config_cache

    db_name = f"ariel_{os.getpid()}_{int(time.time() * 1000) % 10_000_000}"
    db_url = _fresh_database(admin_url, db_name)

    config_dir = tmp_path / "deployment"
    config_dir.mkdir()
    logbook = config_dir / "logbook.json"
    logbook.write_text(json.dumps({"entries": []}))
    ariel_section = {
        "database": {"uri": db_url},
        "ingestion": {"adapter": "generic_json", "source_url": str(logbook)},
        "search_modules": {"keyword": {"enabled": True}},
    }
    config_path = config_dir / "config.yml"
    config_path.write_text(yaml.safe_dump({"ariel": ariel_section}))

    loop = _Loop()
    entries = _seed_entries(datetime.now(UTC))
    try:
        loop.run(_migrate_and_seed(ariel_section, entries))

        ariel_app = create_ariel_app(config_path)
        _delay_capabilities(ariel_app)
        with run_app_server(ariel_app) as ariel_url:
            with live_chat_server(
                tmp_path,
                enabled_panels=frozenset({"artifacts", "ariel"}),
                panel_urls={"ariel": ariel_url},
            ) as (base_url, hub):
                port = base_url.rsplit(":", 1)[1]
                # The MCP tools read config.yml from the cwd and reach the hub
                # through OSPREY_WEB_PORT, as they do in a deployment.
                monkeypatch.chdir(config_dir)
                monkeypatch.setenv("OSPREY_WEB_PORT", port)
                reset_config_cache()
                reset_ariel_context()
                initialize_ariel_context()
                try:
                    page = browser.new_page(viewport={"width": 1600, "height": 1000})
                    page.add_init_script(_PROBES)
                    page.goto(base_url, wait_until="domcontentloaded")
                    wait_for_page_ready(page)
                    expect(page.locator(_INPUT)).to_be_visible(timeout=15_000)

                    stack = Stack(
                        base_url=base_url,
                        hub=hub,
                        ariel_app=ariel_app,
                        page=page,
                        loop=loop,
                        entries=entries,
                    )
                    _open_panel(f"/panel/ariel/{initial_hash}")
                    # The hub may already show ARIEL at its default page when
                    # the open lands; the open then loads its url as a new
                    # document. Until that one has loaded, the old document's
                    # list and ready look like the opened ARIEL's.
                    page.wait_for_function(_IFRAME_AT_ITS_SRC, arg=_ARIEL_IFRAME, timeout=20_000)
                    # ARIEL's router announces itself only after the delayed
                    # capabilities answer; the hub navigates in place from then on.
                    page.wait_for_function("() => window.__navReady > 0", timeout=20_000)
                    stack.wait_settled()
                    assert ariel_app.state.capabilities_delayed >= 1
                    yield stack
                    page.close()
                finally:
                    with contextlib.suppress(Exception):
                        loop.run(get_ariel_context().shutdown(), timeout=20)
                    reset_ariel_context()
                    reset_config_cache()
    finally:
        loop.close()


@pytest.fixture
def stack_factory(tmp_path, chromium_browser, ariel_postgres_url, monkeypatch):
    """Build the stack opened at a given ARIEL hash; torn down after the test."""
    with contextlib.ExitStack() as exits:

        def make(initial_hash: str) -> Stack:
            return exits.enter_context(
                _ariel_stack(
                    tmp_path,
                    chromium_browser,
                    ariel_postgres_url,
                    monkeypatch,
                    initial_hash=initial_hash,
                )
            )

        yield make


def _scroll_list(stack: Stack, offset: int) -> float:
    """Scroll the ARIEL list body to *offset* and return where it settled (> 0)."""
    stack.frame().evaluate(
        f"(y) => {{ document.querySelector('{_LIST_BODY}').scrollTop = y; }}", offset
    )
    stack.page.wait_for_timeout(200)
    top = stack.scroll_top()
    assert top > 0, "the list is not scrollable; the fixture needs more rows or a smaller tile"
    return top


def _visible_row(stack: Stack) -> str:
    """The id of a row wholly inside the list body's viewport, away from its edges."""
    entry_id = stack.frame().evaluate(
        f"""() => {{
          const body = document.querySelector('{_LIST_BODY}').getBoundingClientRect();
          const rows = Array.from(document.querySelectorAll('{_ROWS}'));
          const inside = rows.filter((r) => {{
            const b = r.getBoundingClientRect();
            return b.top >= body.top + 4 && b.bottom <= body.bottom - 4;
          }});
          const pick = inside[Math.floor(inside.length / 2)];
          return pick ? pick.dataset.entryId : null;
        }}"""
    )
    assert entry_id, "no row lies wholly inside the list viewport"
    return entry_id


# ---------------------------------------------------------------------------
# (1) entry_open keeps the search and its scroll, selects, re-flashes
# ---------------------------------------------------------------------------


def test_ariel_entry_open_keeps_search_and_scroll_and_repeat_reflashes(stack_factory):
    """The agent shows a listed entry without the operator losing the search.

    The document is the same one before and after (a marker set on its window
    survives), which is what proves the open was a message and not a reload:
    a src load of ``#sel=<id>`` would drop ``q`` and the scroll with it.
    """
    stack = stack_factory("#q=orbit")
    frame_rows = stack.frame_locator.locator(_ROWS)
    expect(frame_rows.first).to_be_visible(timeout=15_000)
    assert "q=orbit" in stack.hash()
    rows_before = stack.row_ids()
    assert rows_before and all(r.startswith("orb-") for r in rows_before), rows_before

    top = _scroll_list(stack, 160)
    target = _visible_row(stack)
    stack.frame().evaluate("() => { window.__sameDocument = 'yes'; }")

    answer = stack.tool("entry_open", entry_id=target, attachment_id=None)
    assert answer["url"].endswith(f"#sel={target}"), answer

    expect(stack.frame_locator.locator(_READER_TITLE)).to_contain_text(
        next(e["subject"] for e in stack.entries if e["entry_id"] == target), timeout=10_000
    )
    frame = stack.frame()
    frame.wait_for_function(
        "(id) => location.hash.includes('sel=' + id) && location.hash.includes('q=orbit')",
        arg=target,
        timeout=10_000,
    )
    frame.wait_for_function(
        "(id) => window.__flashed.filter((x) => x === id).length === 1", arg=target, timeout=10_000
    )
    assert frame.evaluate("() => window.__sameDocument") == "yes", "the iframe reloaded"
    assert stack.row_ids() == rows_before, "the list was re-run"
    assert stack.scroll_top() == pytest.approx(top, abs=1), "the list scroll moved"

    # The same open again re-selects and re-flashes, still in place.
    stack.tool("entry_open", entry_id=target, attachment_id=None)
    frame.wait_for_function(
        "(id) => window.__flashed.filter((x) => x === id).length === 2", arg=target, timeout=10_000
    )
    assert frame.evaluate("() => window.__sameDocument") == "yes", "the iframe reloaded"
    assert "q=orbit" in stack.hash()
    assert stack.scroll_top() == pytest.approx(top, abs=1)


# ---------------------------------------------------------------------------
# (2) an entry the search does not list opens in the reader with the notice
# ---------------------------------------------------------------------------


def test_ariel_entry_open_out_of_list_opens_reader_with_notice(stack_factory):
    """An entry outside the current search still opens, and the operator is told why."""
    stack = stack_factory("#q=orbit")
    expect(stack.frame_locator.locator(_ROWS).first).to_be_visible(timeout=15_000)
    rows_before = stack.row_ids()
    outsider = "vac-02"
    assert outsider not in rows_before
    stack.frame().evaluate("() => { window.__sameDocument = 'yes'; }")

    stack.tool("entry_open", entry_id=outsider, attachment_id=None)

    expect(stack.frame_locator.locator(_READER_TITLE)).to_contain_text(
        "Vacuum gauge 2", timeout=10_000
    )
    expect(stack.frame_locator.locator(".osprey-toast-host")).to_contain_text(
        _NOT_LISTED, timeout=10_000
    )
    frame = stack.frame()
    assert frame.evaluate("() => window.__sameDocument") == "yes", "the iframe reloaded"
    assert "q=orbit" in stack.hash() and f"sel={outsider}" in stack.hash()
    assert stack.row_ids() == rows_before
    assert outsider not in stack.flashed()


# ---------------------------------------------------------------------------
# (3) entry_create while scrolled → "1 new"
# ---------------------------------------------------------------------------


def test_ariel_entry_create_while_scrolled_shows_one_new(stack_factory):
    """A direct write the operator cannot see (list scrolled) becomes a pill, not a jump."""
    stack = stack_factory("")
    expect(stack.frame_locator.locator(_ROWS).first).to_be_visible(timeout=15_000)
    pill = stack.frame_locator.locator(_PILL)
    expect(pill).to_be_hidden()
    rows_before = stack.row_ids()
    top = _scroll_list(stack, 200)

    created = stack.tool(
        "entry_create",
        subject="RF cavity 3 retuned",
        details="Detuned by 2 kHz after the trip.",
        draft=False,
    )
    entry_id = created["entry_id"]

    expect(pill).to_be_visible(timeout=15_000)
    expect(pill).to_have_text("1 new")
    assert stack.row_ids() == rows_before, "a create while scrolled must not re-run the list"
    assert stack.scroll_top() == pytest.approx(top, abs=1)
    assert entry_id not in stack.flashed()

    # The pill brings it in: top of the list, refreshed, pill gone.
    pill.click()
    expect(stack.frame_locator.locator(f'#ariel-list [data-entry-id="{entry_id}"]')).to_be_visible(
        timeout=10_000
    )
    expect(pill).to_be_hidden()
    assert stack.scroll_top() < 1


# ---------------------------------------------------------------------------
# (4) a real entry_publish in view flashes, pill unchanged
# ---------------------------------------------------------------------------


def test_ariel_entry_publish_in_view_flashes_and_pill_unchanged(stack_factory):
    """Publishing writes a facility entry that lands at the top of the list and flashes there.

    The publish goes through the generic adapter for real: the facility
    entry is appended to the JSON logbook and ingested back under the id the
    adapter assigned, which is the id the frame names and the row carries.
    """
    stack = stack_factory("")
    expect(stack.frame_locator.locator(_ROWS).first).to_be_visible(timeout=15_000)
    pill = stack.frame_locator.locator(_PILL)
    assert stack.scroll_top() < 1

    created = stack.tool(
        "entry_create",
        subject="Septum magnet cycled",
        details="Cycled twice before injection.",
        draft=False,
    )
    local_id = created["entry_id"]
    frame = stack.frame()
    # At the top of page 1 a create is a hit: refreshed in and flashed, no pill.
    frame.wait_for_function("(id) => window.__flashed.includes(id)", arg=local_id, timeout=15_000)
    expect(pill).to_be_hidden()
    pill_before = (pill.is_hidden(), pill.text_content())

    published = stack.tool("entry_publish", entry_id=local_id)
    facility_id = published["entry_id"]
    assert facility_id != local_id, published
    logbook = json.loads((Path.cwd() / "logbook.json").read_text())
    assert [e["id"] for e in logbook["entries"]] == [facility_id]

    frame.wait_for_function(
        "(id) => window.__flashed.includes(id)", arg=facility_id, timeout=15_000
    )
    expect(
        stack.frame_locator.locator(f'#ariel-list [data-entry-id="{facility_id}"]')
    ).to_be_visible()
    # Let any trailing side fetch settle before calling the pill unchanged.
    stack.page.wait_for_timeout(750)
    assert (pill.is_hidden(), pill.text_content()) == pill_before, "a publish changed the pill"
    assert pill_before[0], "the pill was showing before the publish"


# ---------------------------------------------------------------------------
# (5) Ask agent → composer, unsent; during a streaming turn too
# ---------------------------------------------------------------------------


def test_ariel_ask_agent_lands_in_composer_unsent_even_during_a_streaming_turn(
    stack_factory,
):
    """The reader's "Ask agent" cites the entry in the prompt and never sends it.

    First with the agent idle: the citation is in the composer at once. Then
    with a turn streaming (held on a gate): the composer stays open, so the
    citation lands at once there too, and is still unsent once the turn ends.
    The fake SDK's prompt log is the proof nothing was sent.
    """
    stack = stack_factory("#sel=orb-03")
    page = stack.page
    expect(stack.frame_locator.locator(_READER_TITLE)).to_contain_text(
        "Orbit correction 3", timeout=15_000
    )
    ask = stack.frame_locator.locator('[data-reader-action="ask-agent"]')
    textarea = page.locator(_INPUT)

    ask.click()
    expect(textarea).to_have_value(
        re.compile(r"Logbook entry orb-03 .*\n> Orbit correction 3 applied"),
        timeout=10_000,
    )
    page.wait_for_timeout(500)
    assert OBSERVED_PROMPTS == [], f"Ask agent sent a turn: {OBSERVED_PROMPTS}"

    # A turn that streams until released.
    prompt = "Summarise the orbit work"
    PLANS[prompt] = [("text", "Working on it"), ("gate",), ("text", "Done."), ("result",)]
    textarea.fill(prompt)
    textarea.press("Enter")
    expect(page.locator(f"{_OP} .op-entry.assistant")).to_contain_text(
        "Working on it", timeout=10_000
    )
    expect(textarea).to_be_enabled(timeout=10_000)

    stack.tool("entry_open", entry_id="orb-05", attachment_id=None)
    expect(stack.frame_locator.locator(_READER_TITLE)).to_contain_text(
        "Orbit correction 5", timeout=10_000
    )
    ask.click()
    expect(textarea).to_have_value(
        re.compile(r"Logbook entry orb-05 .*\n> Orbit correction 5 applied"),
        timeout=10_000,
    )

    requests.post(f"{stack.base_url}/__test__/release", timeout=10).raise_for_status()
    expect(page.locator(f"{_OP} .op-entry.assistant").last).to_contain_text("Done.", timeout=10_000)
    expect(textarea).to_have_value(re.compile(r"Logbook entry orb-05"), timeout=10_000)
    page.wait_for_timeout(500)
    assert OBSERVED_PROMPTS == [prompt], f"a pending citation was sent: {OBSERVED_PROMPTS}"


# ---------------------------------------------------------------------------
# (6) after the operator reloads the hub, entry_open still opens the entry
# ---------------------------------------------------------------------------


def test_ariel_entry_open_after_a_hub_reload_shows_the_entry(stack_factory):
    """A reload makes a new dock client; the open reaches it, shows the entry and says so."""
    stack = stack_factory("")
    expect(stack.frame_locator.locator(_ROWS).first).to_be_visible(timeout=15_000)

    stack.page.reload(wait_until="domcontentloaded")
    wait_for_page_ready(stack.page)
    expect(stack.page.locator(_INPUT)).to_be_visible(timeout=15_000)
    stack.wait_settled()

    target = stack.entries[3]
    answer = stack.tool("entry_open", entry_id=target["entry_id"], attachment_id=None)

    assert answer.get("shown") is True, answer
    expect(stack.frame_locator.locator(_READER_TITLE)).to_contain_text(
        target["subject"], timeout=10_000
    )
