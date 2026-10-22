"""The time the panel takes to draw a full raw buffer, in a real browser.

A shot's raw buffer of 72 BPMs x 260 samples x 3 planes is drawn by the
panel's own wiring: the ``shot`` stream event fetches the buffer and the
render adapter reacts each of the Raw tab's three plots. One render is timed
from the moment the panel requests the buffer to the moment the last of the
three plots has finished drawing: a plot has finished when it has emitted as
many ``plotly_afterplot`` events as reacts were issued to it, because each
react ends in exactly one afterplot and an earlier event belongs to an
earlier react. The median of three renders must stay under one second.

The budget is an absolute wall-clock one, so the module carries the
``wallclock`` marker: every parallel run deselects it, and CI times it in a
serial step of its own.

Every page runs through :mod:`tests.interfaces.trajectory._stack`. Skips
cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from osprey.services.trajectory.constants import CLIENT_HEADER
from tests.interfaces.trajectory import _stack
from tests.interfaces.trajectory._stack import SETTLE_MS, Stack, connected, run_stack

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page

pytestmark = [pytest.mark.browser, pytest.mark.slow, pytest.mark.wallclock]

#: The payload: BPMs, turn-by-turn samples per record, and the planes drawn.
N_BPMS, N_SAMPLES, N_PLANES = 72, 260, 3
#: Turns recorded after the trigger sample: enough that the buffer keeps
#: every sample of the record.
TURNS = N_SAMPLES - _stack.TRIGGER_SAMPLE
#: The Raw tab's three plot hosts.
RAW_PLOTS = ("plot-raw-sum", "plot-raw-x", "plot-raw-y")
#: Renders timed; their median is held to the budget.
RENDERS = 3
BUDGET_S = 1.0

#: Counts, per raw plot, the reacts the panel issues and the afterplot events
#: they end in, and stamps the afterplot that completes the plot's last react.
_INSTRUMENT = """
(ids) => {
  const t = { ids, reacts: {}, afters: {}, base: {}, done: {}, bound: new WeakSet() };
  for (const id of ids) { t.reacts[id] = 0; t.afters[id] = 0; }
  const react = window.Plotly.react;
  window.Plotly.react = function (gd, ...args) {
    const out = react.call(this, gd, ...args);
    const id = gd && gd.id;
    if (id in t.reacts) {
      t.reacts[id] += 1;
      delete t.done[id];
      if (!t.bound.has(gd)) {
        t.bound.add(gd);
        gd.on('plotly_afterplot', () => {
          // An afterplot no react is waiting on (a resize redraw) is not counted.
          t.afters[id] = Math.min(t.afters[id] + 1, t.reacts[id]);
          if (t.afters[id] === t.reacts[id] && t.reacts[id] > t.base[id] && !(id in t.done)) {
            t.done[id] = performance.now();
          }
        });
      }
    }
    return out;
  };
  t.arm = () => {
    t.base = { ...t.reacts };
    t.done = {};
    performance.clearResourceTimings();
  };
  t.finished = () => t.ids.every((id) => id in t.done);
  t.elapsed = () => {
    const raw = performance.getEntriesByType('resource')
      .filter((entry) => new URL(entry.name).pathname.endsWith('/raw'));
    if (raw.length !== 1) return { error: `raw buffer requests: ${raw.length}` };
    const end = Math.max(...t.ids.map((id) => t.done[id]));
    return { seconds: (end - raw[0].startTime) / 1000 };
  };
  performance.setResourceTimingBufferSize(10000);
  window.__renderTiming = t;
}
"""

#: What each raw plot drew: trace count and points in the first trace.
_DRAWN = """
(ids) => ids.map((id) => {
  const gd = document.getElementById(id);
  return [gd.data.length, gd.data[0].x.length];
})
"""


@pytest.fixture
def stack(
    chromium_browser: Browser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Stack]:
    monkeypatch.setattr(_stack, "SAMPLES", N_SAMPLES)
    monkeypatch.setattr(_stack, "TURNS_TRACKED", TURNS)
    with run_stack(chromium_browser, tmp_path, monkeypatch, n_bpms=N_BPMS) as served:
        yield served


def _call(page: Page, method: str, path: str, client_id: str | None = None, body=None):
    """Send one panel-door call from *page*; answers ``[status, json]``."""
    return page.evaluate(
        """async ([method, path, header, id, body]) => {
             const headers = {'Content-Type': 'application/json'};
             if (id) headers[header] = id;
             const r = await fetch(path, {method, headers,
               body: body == null ? undefined : JSON.stringify(body)});
             return [r.status, await r.json()];
           }""",
        [method, path, CLIENT_HEADER, client_id, body],
    )


def _open_session(page: Page) -> str:
    """Open the session from *page*; answers the client id the grant handed it."""
    with page.expect_response(
        lambda r: r.url.endswith("/api/session") and r.request.method == "POST"
    ) as granted:
        page.locator("[data-view=session] [data-control=session-open]").click()
    expect(page.locator("[data-view=session] [data-control=session-end]")).to_be_enabled(
        timeout=SETTLE_MS
    )
    return granted.value.json()["client_id"]


def _render_once(page: Page, client_id: str) -> float:
    """Take one shot and answer the seconds its raw buffer took to draw."""
    page.evaluate("() => window.__renderTiming.arm()")
    status, shot = _call(
        page, "POST", "api/shot", client_id, {"steps": ["arm", "inject", "readout"]}
    )
    assert status == 200, shot
    page.wait_for_function("() => window.__renderTiming.finished()", timeout=SETTLE_MS)
    timing = page.evaluate("() => window.__renderTiming.elapsed()")
    assert "seconds" in timing, timing
    drawn = page.evaluate(_DRAWN, list(RAW_PLOTS))
    assert drawn == [[N_BPMS, N_SAMPLES]] * N_PLANES, (
        f"the raw plots drew {drawn}, not {N_BPMS} BPMs x {N_SAMPLES} samples"
    )
    return timing["seconds"]


def test_a_full_raw_buffer_draws_within_one_second(stack: Stack) -> None:
    page = stack.context("alice").new_page()
    page.goto(stack.panel_url("alice"), wait_until="load")
    connected(page)
    stack.assert_loaded_clean()
    page.evaluate(_INSTRUMENT, list(RAW_PLOTS))
    client_id = _open_session(page)
    status, answer = _call(page, "POST", "api/setup", client_id, {"settings": {}})
    assert status == 200, answer

    seconds = [_render_once(page, client_id) for _ in range(RENDERS)]

    median = statistics.median(seconds)
    assert median < BUDGET_S, (
        f"median raw render {median:.3f} s over {BUDGET_S} s budget; renders {seconds}"
    )
    stack.assert_loaded_clean()
