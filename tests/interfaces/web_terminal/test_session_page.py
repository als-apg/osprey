"""The Activity page (session.html) in a real browser, over a faked record.

The record client is faked where the routes take it, so every case sends raw
record entries through the real ``/api/session-activity`` route and the real
page: the four views and their counts, a built-in call in the Tool calls
table, the operator's approval in Safety, the record chip of an incomplete
read, an open row and the scroll position across the 10 s poll of a live
session, the Tool calls table's layout at 1280 px, and the unavailable state.

The poll is driven with Playwright's clock (installed before navigation, so it
owns the page's timers) instead of a real 10 s wait.

Run:
    uv run pytest tests/interfaces/web_terminal/test_session_page.py -m browser -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from osprey.agent_record.client import ClientAnswer
from osprey.interfaces.web_terminal.routes import session as session_routes
from osprey.stores.artifact_store import ArtifactStore
from tests.agent_record.fakes import FakeRecordClient
from tests.interfaces._hub import hub_server
from tests.interfaces.web_terminal._fakes import FakePtySession

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]

KEY = "11111111-2222-4333-8444-555555555555"
TRANSCRIPT = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"

_T0 = datetime(2026, 11, 12, 16, 42, 0, tzinfo=UTC)

# What only that view draws (Safety has a table of its own).
_VIEW_CONTENT = {
    "timeline": "button.act-row",
    "tools": "th[data-sort]",
    "safety": ".act-cards",
    "artifacts": ".act-art",
}


def _at(seconds: float) -> str:
    return (_T0 + timedelta(seconds=seconds)).isoformat()


def _item(kind: str, ts: float, body: Any, **ids: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "key": f"k:{kind}:{ts}",
        "kind": kind,
        "ts": _at(ts),
        "body": body,
        "agent_id": "",
        "tool_use_id": "",
        "incomplete": False,
    }
    item.update(ids)
    return item


def _call(tid: str, name: str, ts: float, content: Any, source: str = "config", **args: Any):
    """A finished call: its decision 0.1 s after it, its result 0.5 s after."""
    node = _item("tool_call", ts, {"id": tid, "name": name, "input": args}, tool_use_id=tid)
    node["decision"] = _item(
        "decision", ts + 0.1, {"decision": "accept", "source": source}, tool_use_id=tid
    )
    node["result"] = _item(
        "tool_result", ts + 0.5, {"tool_use_id": tid, "content": content}, tool_use_id=tid
    )
    node["subagent"] = None
    return node


def _audit(ts: float, tid: str, **body: Any) -> dict[str, Any]:
    return _item("audit", ts, body, tool_use_id=tid)


def _nodes() -> list[dict[str, Any]]:
    """A prompt, a built-in call, an MCP read, an approved MCP write, an answer."""
    approval = {
        "outcome": "approved",
        "approver": "human",
        "permission_mode": "auto",
        "position": "asked",
    }
    return [
        _item("prompt", 0, {"text": "Start the queue"}),
        _call("t1", "Read", 2, "notes", file_path="/app/notes.md"),
        _call(
            "t2",
            "mcp__controls__channel_read",
            4,
            json.dumps({"result": {"value": 1.0}}),
            channels=["SR:DCCT"],
        ),
        _call("t3", "mcp__bluesky__queue_start", 10, "ok", source="user_temporary"),
        _audit(10, "t3", surface="hook_approval", decision="ask", reason="approval_required"),
        _audit(10.2, "t3", surface="hook_approval", decision="approved", reason="executed"),
        _audit(10.4, "t3", surface="tool_call", duration_ms=300, approval=approval, target="va"),
        _item("answer", 30, {"text": "The queue is running."}),
    ]


def _many_reads(count: int) -> list[dict[str, Any]]:
    return [
        _call(f"r{i}", "Read", 40 + i, "body", file_path=f"/app/file-{i}.md") for i in range(count)
    ]


@pytest.fixture
def record(monkeypatch):
    """The fake record the routes read; a case sets ``record.answers``."""
    fake = FakeRecordClient({"timeline": ClientAnswer(data=_nodes(), complete=True)})
    monkeypatch.setattr(session_routes, "record_client", lambda: fake)
    monkeypatch.delenv("OSPREY_DISPATCH_RUN", raising=False)
    return fake


@contextmanager
def _activity_page(tmp_path, monkeypatch, browser, *, live: bool = False, **page_kw: Any):
    """Yield ``(page, url)``: a page and the Activity URL of the test session.

    The session is named by its transcript in the hash; a *live* one is named
    by its session key, which the app maps to the transcript and holds a
    terminal for.
    """
    monkeypatch.chdir(tmp_path)
    with hub_server(tmp_path) as (base_url, app):
        app.state.transcript_map = {KEY: TRANSCRIPT}
        app.state.transcript_map_provisional = False
        if live:
            app.state.pty_registry._sessions[KEY] = FakePtySession()
        page = browser.new_page(**page_kw)
        page.set_default_timeout(15000)
        where = f"?session_id={KEY}" if live else f"#{TRANSCRIPT}"
        try:
            yield page, f"{base_url}/static/session.html{where}"
        finally:
            page.close()


def _view(page, name: str):
    return page.locator(f'#views button[data-view="{name}"]')


@pytest.mark.usefixtures("record")
def test_views_switch_and_show_their_counts(tmp_path, monkeypatch, chromium_browser):
    ArtifactStore(tmp_path).save_file(
        file_content=b"<html></html>",
        filename="p.html",
        artifact_type="plot_html",
        title="Beam current",
        mime_type="text/html",
        tool_source="create_interactive_plot",
        metadata={"transcript_id": TRANSCRIPT, "call_id": "t2"},
    )
    with _activity_page(tmp_path, monkeypatch, chromium_browser) as (page, url):
        page.goto(url, wait_until="load")

        expect(_view(page, "timeline")).to_have_attribute("aria-pressed", "true")
        expect(page.locator(_VIEW_CONTENT["timeline"]).first).to_be_visible()
        expect(_view(page, "timeline").locator(".osprey-count")).to_have_count(0)
        expect(_view(page, "tools").locator(".osprey-count")).to_have_text("3")
        expect(_view(page, "safety").locator(".osprey-count")).to_have_text("1")
        expect(_view(page, "artifacts").locator(".osprey-count")).to_have_text("1")

        for name in ("tools", "safety", "artifacts", "timeline"):
            _view(page, name).click()
            expect(_view(page, name)).to_have_attribute("aria-pressed", "true")
            expect(page.locator('#views button[aria-pressed="true"]')).to_have_count(1)
            expect(page.locator(_VIEW_CONTENT[name]).first).to_be_visible()
            others = [sel for view, sel in _VIEW_CONTENT.items() if view != name]
            for selector in others:
                expect(page.locator(f"#inner {selector}")).to_have_count(0)
            expect(page).to_have_url(re.compile(rf"#{TRANSCRIPT}/{name}$"))


@pytest.mark.usefixtures("record")
def test_builtin_call_is_listed_in_tool_calls(tmp_path, monkeypatch, chromium_browser):
    with _activity_page(tmp_path, monkeypatch, chromium_browser) as (page, url):
        page.goto(f"{url}/tools", wait_until="load")

        rows = page.locator("table.act-table tr.row")
        expect(rows).to_have_count(3)
        builtin = rows.filter(has_text="/app/notes.md")
        expect(builtin).to_have_count(1)
        expect(builtin.locator("td").nth(1)).to_have_text("Read")

        page.locator('[data-scope="builtin"]').click()
        expect(rows).to_have_count(1)
        expect(rows.first).to_contain_text("/app/notes.md")


@pytest.mark.usefixtures("record")
def test_safety_names_the_operator_approval(tmp_path, monkeypatch, chromium_browser):
    with _activity_page(tmp_path, monkeypatch, chromium_browser) as (page, url):
        page.goto(f"{url}/safety", wait_until="load")

        card = page.locator(".act-approval")
        expect(card).to_have_count(1)
        expect(card.locator("h3")).to_contain_text("queue_start")
        expect(card.locator("h3 .osprey-chip")).to_contain_text("Approved by operator")
        asked = page.locator(".act-card", has_text="Asked the operator")
        expect(asked.locator(".act-card-n")).to_have_text("1")


def test_incomplete_read_says_record_incomplete(tmp_path, monkeypatch, chromium_browser, record):
    record.answers["timeline"] = ClientAnswer(
        data=_nodes(), complete=False, reason="entries still arriving"
    )
    with _activity_page(tmp_path, monkeypatch, chromium_browser) as (page, url):
        page.goto(url, wait_until="load")

        expect(page.locator("#record-chip")).to_have_text("Record incomplete")
        expect(page.locator("button.act-row").first).to_be_visible()


@pytest.mark.usefixtures("record")
def test_complete_read_says_record_complete(tmp_path, monkeypatch, chromium_browser):
    with _activity_page(tmp_path, monkeypatch, chromium_browser) as (page, url):
        page.goto(url, wait_until="load")

        expect(page.locator("#record-chip")).to_have_text("Record complete")


def test_open_row_and_scroll_survive_the_poll(tmp_path, monkeypatch, chromium_browser, record):
    first = _nodes() + _many_reads(40)
    later = first + [_call("late", "Read", 200, "body", file_path="/app/arrived-later.md")]
    record.answers["timeline"] = [
        ClientAnswer(data=first, complete=True),
        ClientAnswer(data=later, complete=True),
    ]
    with _activity_page(
        tmp_path,
        monkeypatch,
        chromium_browser,
        live=True,
        viewport={"width": 1280, "height": 500},
    ) as (page, url):
        page.clock.install()
        page.goto(url, wait_until="load")

        row = page.locator('button.act-row[data-toggle="t1"]')
        row.click()
        expect(row).to_have_attribute("aria-expanded", "true")
        expect(page.locator(".act-detail")).to_have_count(1)
        page.evaluate("document.getElementById('main').scrollTop = 300")
        assert page.evaluate("document.getElementById('main').scrollTop") == 300
        late = page.locator("button.act-row", has_text="/app/arrived-later.md")
        expect(late).to_have_count(0)

        with page.expect_request(lambda r: "/api/session-activity?" in r.url):
            page.clock.fast_forward(10_000)

        expect(late).to_have_count(1)
        expect(row).to_have_attribute("aria-expanded", "true")
        expect(page.locator(".act-detail")).to_have_count(1)
        assert page.evaluate("document.getElementById('main').scrollTop") == 300


@pytest.mark.usefixtures("record")
def test_idle_session_is_not_read_again(tmp_path, monkeypatch, chromium_browser):
    with _activity_page(tmp_path, monkeypatch, chromium_browser) as (page, url):
        page.clock.install()
        page.goto(url, wait_until="load")
        expect(page.locator("button.act-row").first).to_be_visible()
        asked: list[str] = []
        page.on("request", lambda r: asked.append(r.url))

        # A poll would send its request from the timer this runs; the round
        # trip after it lets that request's event arrive.
        page.clock.fast_forward(30_000)
        page.evaluate("1")

        assert [url for url in asked if "/api/session-activity" in url] == []


def test_fresh_session_says_no_activity_yet(tmp_path, monkeypatch, chromium_browser, record):
    record.answers["timeline"] = ClientAnswer(data=[], complete=True)
    with _activity_page(tmp_path, monkeypatch, chromium_browser, live=True) as (page, url):
        page.goto(url, wait_until="load")

        expect(page.locator("#inner")).to_contain_text("No activity in this session yet")
        # The key's own transcript was read, and no other session's.
        assert {c["session_id"] for c in record.calls_to("timeline")} == {TRANSCRIPT}


_CELL_OVERLAP_JS = """
() => {
  const main = document.getElementById('main').getBoundingClientRect();
  const faults = [];
  const lines = [...document.querySelectorAll('table.act-table tr')].filter(
    (tr) => !tr.classList.contains('detail')
  );
  for (const [r, tr] of lines.entries()) {
    const cells = [...tr.children];
    for (const [c, cell] of cells.entries()) {
      const box = cell.getBoundingClientRect();
      if (box.right > main.right + 0.5) {
        faults.push(`row ${r} cell ${c} leaves the page by ${Math.round(box.right - main.right)}px`);
      }
      const next = cells[c + 1];
      if (next && box.right > next.getBoundingClientRect().left + 0.5) {
        faults.push(`row ${r} cell ${c} overlaps cell ${c + 1}`);
      }
      // What a cell holds stays inside it; the one wide column ellipsizes.
      const clips = getComputedStyle(cell).overflow === 'hidden';
      if (!clips && cell.scrollWidth > cell.clientWidth + 1) {
        faults.push(`row ${r} cell ${c} spills (${cell.scrollWidth} > ${cell.clientWidth})`);
      }
    }
  }
  return { faults, rows: lines.length };
}
"""


def test_tool_calls_table_cells_do_not_overlap_at_1280(
    tmp_path, monkeypatch, chromium_browser, record
):
    long_path = "/app/" + "a-very-long-directory-name/" * 12 + "notes.md"
    record.answers["timeline"] = ClientAnswer(
        data=_nodes()
        + [
            _call("w1", "Read", 50, "x" * 250_000, file_path=long_path),
            _call(
                "w2",
                "mcp__accelerator_physics_services__compute_closed_orbit_response_matrix",
                52,
                "ok",
                source="user_temporary",
                plane="horizontal",
                correctors=[f"SR{i:02d}C:HCM1" for i in range(1, 13)],
            ),
        ],
        complete=True,
    )
    with _activity_page(
        tmp_path, monkeypatch, chromium_browser, viewport={"width": 1280, "height": 800}
    ) as (page, url):
        page.goto(f"{url}/tools", wait_until="load")
        expect(page.locator("table.act-table tr.row")).to_have_count(5)

        found = page.evaluate(_CELL_OVERLAP_JS)

        assert found["rows"] == 6
        assert found["faults"] == []


def test_unavailable_record_offers_try_again(tmp_path, monkeypatch, chromium_browser, record):
    record.unavailable = "record service down"
    with _activity_page(tmp_path, monkeypatch, chromium_browser) as (page, url):
        page.goto(url, wait_until="load")

        expect(page.locator("#record-chip")).to_have_text("Record unavailable")
        again = page.locator("#inner").get_by_role("button", name="Try again")
        expect(again).to_be_visible()
        expect(page.locator("button.act-row")).to_have_count(0)

        record.unavailable = None
        again.click()

        expect(page.locator("#record-chip")).to_have_text("Record complete")
        expect(page.locator("button.act-row").first).to_be_visible()
