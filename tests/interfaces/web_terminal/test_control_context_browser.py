"""Browser tests: the control context is one deployment-wide fact, in every tab.

``test_posture_toggle_browser.py`` pins what ONE tab's chip does. Three things
survive only across tabs, across a surface hand-off and across the audit trail,
and none of them is observable from a single page or from a route test:

  (1) **A switch made in one tab reaches the other on the pushed frame.** The
      record is deployment-wide, so a second tab that made no gesture must
      still name the machine the deployment moved to — and it must learn it
      from the ``control_context`` frame the owning terminal pushes, not from
      its own 5 s fallback poll. "It changed eventually" would pass on the poll
      alone and prove nothing, so the proof here is a *phase* one: the change
      is required to land before the tab's next idle poll was due, measured
      against a poll tick this test watched go past.

  (4) **A gesture made from the chip carries no session at all.** The chip
      stopped being a session surface (3.5), so every POST it sends omits
      ``session_id`` — which is what the JupyterLab bar's gesture looks like
      too, since that surface has no session to name. The route files those on
      the ``jupyter_lab`` audit surface with a null session. The route tests
      pin that for a hand-built body; only a browser proves the shipped chip
      really sends one.

  (5) **A real surface hand-off does not move the deployment.** The Simple
      view takes a session off the terminal surface through
      ``POST /api/session/{key}/handoff``. Under the retired per-session model
      that flip reset the target to the baseline; under the record it cannot,
      because the target was never the session's to carry. So the hand-off is
      driven for real, the roster is exercised from the Simple view afterwards
      (the chip paints, and Switch still works there), and the flip back is
      asserted to leave the generation exactly where the switch left it.

The stack is :func:`~...test_posture_toggle_browser._chip_hub` — imported, not
rebuilt. One record written before the server starts, owned by this process, at
``va`` and generation 1; three configured targets; the agent-data root stamped
into ``OSPREY_AGENT_DATA_ROOT``. Everything below reads or moves that one file.

Run:
    uv run pytest tests/interfaces/web_terminal/test_control_context_browser.py -m browser -q

Skips cleanly when the chromium headless binary is not installed. Run it ALONE:
these lanes drive a real browser beside a real uvicorn, and a parallel pytest
has OOM-killed the runner here before.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import requests

from osprey.audit import writer as audit_writer
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey.interfaces.web_terminal.session_handoff import ERROR_SESSION_ATTACHED_ELSEWHERE
from osprey_connectors import control_context, posture_store

# The stack, the waits and the vocabulary all come from the sibling suite. The
# brief for this task is explicit that there must be ONE chip stack fixture in
# the tree: a second hub arranged slightly differently would be a second answer
# to "what does a writable deployment look like", and the two would drift.
from tests.interfaces.web_terminal.test_posture_toggle_browser import (
    ACTIVE_TARGET,
    CARD,
    CHIP,
    CHIP_SHORT,
    CHIP_STATE,
    MODAL_CONFIRM,
    MODAL_TITLE,
    NAMES,
    OPEN_MODAL,
    POPOVER,
    POSTURE_TARGET,
    RECORD_GENERATION,
    SWITCH_TARGET,
    TIMEOUT,
    _chip_hub,
    _open_popover,
    _publish_report,
    _read_record,
    _recorded_posture,
    _row,
    _settled_chip,
    _toggle,
)

if TYPE_CHECKING:
    from playwright.sync_api import Page

try:
    from playwright.sync_api import expect
except ImportError:  # pragma: no cover — the chromium_browser fixture skips the suite
    pass

pytestmark = [pytest.mark.browser, pytest.mark.slow]


#: The chip's fallback poll, in milliseconds, as ``control-target-chip.js``
#: exports it (``IDLE_POLL_MS``). Read out of the shipped module rather than
#: restated, because the whole of lane 1 is a claim about this number: if the
#: module's poll got faster, a bound written against a stale copy would start
#: passing on the poll it exists to rule out.
def _idle_poll_ms() -> int:
    """``IDLE_POLL_MS`` as the shipped chip module declares it."""
    from pathlib import Path

    import osprey.interfaces.web_terminal as web_terminal

    source = (
        Path(web_terminal.__file__).parent / "static" / "js" / "control-target-chip.js"
    ).read_text(encoding="utf-8")
    match = re.search(r"export const IDLE_POLL_MS = (\d+);", source)
    assert match, "control-target-chip.js no longer exports IDLE_POLL_MS"
    return int(match.group(1))


IDLE_POLL_MS = _idle_poll_ms()

#: How much of the gap to the next scheduled poll is left as slack for timer
#: drift and for the round trip that stamps the observation. The assertion is
#: "the change landed before the next idle poll was DUE", and a browser's
#: interval timer is allowed to be a little late, never early — so the margin
#: only has to cover the measurement, not the timer.
_POLL_MARGIN_MS = 500

# ---------------------------------------------------------------------------
# The probe a watching tab carries
# ---------------------------------------------------------------------------

#: Installed into a settled page, after the chip has mounted. It records two
#: things and interprets neither: when the chip read the posture route, and
#: when the chip's ``data-target-kind`` first changed away from what it said at
#: install time. Both are on the page's own ``performance.now()`` clock, so the
#: gap between them is not measured across the Playwright round trip.
_CHIP_PROBE = """
() => {
  const chip = document.querySelector('#control-target-chip');
  if (!chip) throw new Error('no chip to probe');
  const probe = {
    reads: [],
    changedAt: null,
    changedKind: null,
    startKind: chip.dataset.targetKind,
  };
  window.__ctcProbe = probe;
  const inner = window.fetch;
  window.fetch = function (input, init) {
    const url = typeof input === 'string' ? input : (input && input.url) || '';
    if (url.indexOf('/api/terminal/posture') !== -1) probe.reads.push(performance.now());
    return inner.apply(this, arguments);
  };
  new MutationObserver(() => {
    if (probe.changedAt !== null) return;
    if (chip.dataset.targetKind === probe.startKind) return;
    probe.changedAt = performance.now();
    probe.changedKind = chip.dataset.targetKind;
  }).observe(chip, { attributes: true, attributeFilter: ['data-target-kind'] });
}
"""


def _install_probe(page: Page) -> None:
    """Start watching this page's chip reads and its target attribute."""
    page.evaluate(_CHIP_PROBE)


def _await_poll_tick(page: Page) -> float:
    """Block until one more idle read goes past, and return when it happened.

    This is what makes lane 1 a phase measurement rather than a wall-clock one.
    The chip's fallback poll is a plain ``setInterval``, so once a read has been
    SEEN the next one is not due for another :data:`IDLE_POLL_MS`. Everything
    the test does after this returns therefore happens inside a window in which
    the poll is known to be silent.

    Returns:
        The page-clock timestamp of that read.
    """
    before = page.evaluate("() => window.__ctcProbe.reads.length")
    page.wait_for_function("n => window.__ctcProbe.reads.length > n", arg=before, timeout=TIMEOUT)
    return page.evaluate("() => window.__ctcProbe.reads[window.__ctcProbe.reads.length - 1]")


#: The one server-sent stream the hub's modules read. `panel-sse.js`, the
#: activity strip `session.js` wires, and the chip all name this URL, and
#: `createEventSource` (api.js) is what makes that one socket instead of
#: three.
FILES_EVENT_STREAM = "/api/files/events"

#: Seeded at document start, so the count covers the page's FIRST socket:
#: every `EventSource` the page constructs is kept, with the URL it was
#: opened on. Counting CONSTRUCTIONS rather than opens is the point — a
#: module that pays for its own connection has already paid by the time the
#: socket opens, and a duplicate that never opens is still a duplicate under
#: the browser's per-host cap.
_STREAM_PROBE = """
(function () {
  const Original = window.EventSource;
  if (!Original) return;
  const streams = [];
  window.__ctcStreams = streams;
  window.EventSource = new Proxy(Original, {
    construct(target, args) {
      const stream = new target(...args);
      streams.push(stream);
      return stream;
    },
  });
})();
"""


def _live_streams(page: Page, path: str) -> int:
    """How many event-stream sockets this page is holding open on *path*.

    LIVE rather than constructed: `readyState === 2` is CLOSED, and api.js
    replaces a socket the browser has given up on with a fresh one, so a
    reconnection is one socket held and not two. Counting corpses would
    redden this lane for the behaviour it is not about.
    """
    return page.evaluate(
        "(path) => (window.__ctcStreams || []).filter("
        "(stream) => stream.url.includes(path) && stream.readyState !== 2).length",
        path,
    )


def _await_stream_open(page: Page, path: str) -> None:
    """Block until one of this page's sockets on *path* is OPEN.

    The readiness barrier for everything below: until a stream is open the
    page is not subscribed to anything, so a count taken before it means
    only that the page has not got there yet.
    """
    page.wait_for_function(
        "(path) => (window.__ctcStreams || []).some("
        "(stream) => stream.url.includes(path) && stream.readyState === 1)",
        arg=path,
        timeout=TIMEOUT,
    )


#: Subscribe to the stream through the page's OWN copy of api.js — the module
#: instance app.js loaded, reached by rewriting the entrypoint's `src`, so the
#: URL is the one app.js's relative `./api.js` resolved to under whatever
#: prefix this deployment serves. A second copy fetched under any other URL
#: would carry its own `sharedStreams` map and prove the opposite of what
#: this asks.
_ADD_SUBSCRIBER = """
async (path) => {
  const entry = document.querySelector('script[type="module"][src$="/js/app.js"]');
  if (!entry) throw new Error('the hub page loads no app.js module entrypoint');
  const api = await import(entry.src.replace('/app.js', '/api.js'));
  window.__ctcExtra = api.createEventSource(path, {});
}
"""

#: Installed into a settled page: records every value `data-pending` ever
#: takes on the chip, including the one it already has. `switching…` is a
#: sticky state — nothing resolves a wait for a terminus that was never
#: written, so a point-in-time check would in fact be enough — but a log is
#: what makes the assertion read as "never" rather than "not just now".
_PENDING_PROBE = """
() => {
  const chip = document.querySelector('#control-target-chip');
  if (!chip) throw new Error('no chip to probe');
  const seen = [];
  window.__ctcPending = seen;
  if (chip.dataset.pending !== undefined) seen.push(chip.dataset.pending);
  new MutationObserver(() => {
    if (chip.dataset.pending !== undefined) seen.push(chip.dataset.pending);
  }).observe(chip, { attributes: true, attributeFilter: ['data-pending'] });
}
"""


# ---------------------------------------------------------------------------
# (1) a switch in one tab, on the frame, in the other
# ---------------------------------------------------------------------------


def test_a_switch_in_one_tab_reaches_the_other_before_its_next_poll(
    tmp_path, monkeypatch, chromium_browser
):
    """Two tabs, two sessions: the switch made in one lands in the other on the push.

    The second tab makes no gesture and holds no pending switch, so its only
    two ways of learning anything are the ``control_context`` frame the owning
    terminal pushes on the tick after the record moves, and its own idle poll
    :data:`IDLE_POLL_MS` later. This test rules the second one out by phase: it
    watches a poll read go past, and only then lets the switch be confirmed, so
    the next poll is not due for a full interval. A change that arrives inside
    that window came from the frame.

    The switch itself is made through the real popover in the first tab, and
    everything slow about it — opening the popover, reaching the confirm — is
    done BEFORE the window opens. Only the confirm click is inside it.

    A live controls server is planted at the pre-switch generation so the gate
    has a reachability sweep to read; it lags afterwards, which is why the first
    tab sits in ``switching…``. The second tab is not waiting on anything, so it
    simply names the machine the record now names.
    """
    with _chip_hub(tmp_path, monkeypatch) as (base_url, _app, root):
        _publish_report(root)
        switcher, switcher_session = _settled_chip(chromium_browser, base_url)
        watcher, watcher_session = _settled_chip(chromium_browser, base_url)
        try:
            # Two tabs, two sessions — the terminal route mints one per socket.
            assert switcher_session and watcher_session, (
                "both tabs must have minted a PTY session: "
                f"{switcher_session!r} / {watcher_session!r}"
            )
            assert switcher_session != watcher_session

            expect(watcher.locator(CHIP)).to_have_attribute(
                "data-target-kind", ACTIVE_TARGET, timeout=TIMEOUT
            )

            # Everything expensive about the gesture happens before the window.
            _open_popover(switcher)
            _row(switcher, SWITCH_TARGET).locator(".ctc-switch").click()
            expect(switcher.locator(MODAL_TITLE)).to_have_text(
                f"Switch to {NAMES[SWITCH_TARGET]}?", timeout=TIMEOUT
            )

            _install_probe(watcher)
            polled_at = _await_poll_tick(watcher)

            switcher.locator(MODAL_CONFIRM).click()
            expect(switcher.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)

            watcher.wait_for_function("() => window.__ctcProbe.changedAt !== null", timeout=TIMEOUT)
            probe = watcher.evaluate("() => window.__ctcProbe")
        finally:
            watcher.close()
            switcher.close()

    # The second tab names the machine the deployment moved to.
    assert probe["changedKind"] == SWITCH_TARGET, probe

    # …and it learned it off-cycle. The read that carried the change is the
    # FIRST one after the tick this test watched go past, and it came before
    # the next idle poll was due.
    after = [read for read in probe["reads"] if read > polled_at]
    assert after, f"the watching tab issued no read after the tick at {polled_at}: {probe}"
    carried_at = after[0]
    assert carried_at <= probe["changedAt"], probe
    gap = carried_at - polled_at
    assert gap < IDLE_POLL_MS - _POLL_MARGIN_MS, (
        f"the read that carried the switch came {gap:.0f} ms after the last idle poll, "
        f"which is within drift of the next one ({IDLE_POLL_MS} ms) — this proves nothing "
        f"about the pushed frame. Reads: {probe['reads']}"
    )


# ---------------------------------------------------------------------------
# (6) one stream for the whole page
# ---------------------------------------------------------------------------


def test_one_event_stream_serves_the_whole_page_across_a_switch(
    tmp_path, monkeypatch, chromium_browser
):
    """Three modules, one socket: the page never pays for the stream twice.

    ``panel-sse.js``, the activity strip ``session.js`` wires and the chip all
    read ``/api/files/events`` on one page, and ``createEventSource``
    (``api.js``) is what makes that one socket rather than three. So the number
    this test takes is a number about the whole PAGE, not about the chip: it is
    the count of live event-stream sockets the page is holding, and it is
    asserted at three moments, each ruling out a different way of regaining a
    second connection.

    At rest after boot, first — whichever modules mounted, they share. Then
    after one more subscriber is added through the page's OWN ``api.js``: the
    sharing is a module-registry fact (``sharedStreams`` is a Map inside one
    loaded copy of that module), so a second copy of it fetched under any other
    URL, or a subscriber that reached ``EventSource`` directly, shows up here
    and nowhere else. Then after the deployment moves under the page, because
    the chip follows a switch by re-READING and must not follow it by
    re-subscribing.

    The switch is made in a second tab so this page stays a pure watcher: the
    tab that made the gesture also runs a fast poll while its switch is out,
    and a lane about connections should not have to reason about which of a
    page's own timers is running.

    A live controls server is planted so the switch gate has a reachability
    sweep to read; without one no row offers Switch at all.
    """
    with _chip_hub(tmp_path, monkeypatch) as (base_url, _app, root):
        _publish_report(root)
        watcher, _watcher_session = _settled_chip(
            chromium_browser, base_url, init_script=_STREAM_PROBE
        )
        switcher, _switcher_session = _settled_chip(chromium_browser, base_url)
        try:
            _await_stream_open(watcher, FILES_EVENT_STREAM)
            assert _live_streams(watcher, FILES_EVENT_STREAM) == 1, (
                "a settled hub page holds more than one event-stream socket"
            )

            # A second subscriber, through the page's own api.js.
            watcher.evaluate(_ADD_SUBSCRIBER, FILES_EVENT_STREAM)
            assert _live_streams(watcher, FILES_EVENT_STREAM) == 1, (
                "subscribing again opened a second socket instead of sharing the one open"
            )

            # The deployment moves, from the other tab.
            _open_popover(switcher)
            _row(switcher, SWITCH_TARGET).locator(".ctc-switch").click()
            expect(switcher.locator(MODAL_TITLE)).to_have_text(
                f"Switch to {NAMES[SWITCH_TARGET]}?", timeout=TIMEOUT
            )
            switcher.locator(MODAL_CONFIRM).click()
            expect(switcher.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)

            # The watcher learned about it — so the frame really did arrive on a
            # socket this page is holding — and it learned without opening one.
            expect(watcher.locator(CHIP_SHORT)).to_have_text(NAMES[SWITCH_TARGET], timeout=TIMEOUT)
            assert _live_streams(watcher, FILES_EVENT_STREAM) == 1, (
                "following the switch cost the page a second socket"
            )

            # The page's own subscribers keep the socket after ours leaves.
            watcher.evaluate("() => window.__ctcExtra.stop()")
            assert _live_streams(watcher, FILES_EVENT_STREAM) == 1, (
                "the socket closed when the test's subscriber left, so the page's "
                "own modules were not sharing it"
            )
        finally:
            switcher.close()
            watcher.close()


# ---------------------------------------------------------------------------
# (4) a gesture with no session at all
# ---------------------------------------------------------------------------


@pytest.fixture
def ledger():
    """Capture every audit record the hub writes while it is up.

    ``osprey.audit.writer.record`` is the seam BOTH recorders resolve at call
    time — the routes' own and the HTTP middleware's — which is the same seam
    the route suites use. Reading a ledger file instead would put audit records
    under the repository's own ``var/`` and trip the agent-data guard.
    """
    records: list[dict] = []

    def _record(**fields):
        records.append(fields)
        from pathlib import Path

        return Path("/dev/null/ledger.jsonl")

    with patch.object(audit_writer, "record", side_effect=_record):
        yield records


def test_a_chip_gesture_sends_no_session_and_audits_on_the_lab_surface(
    tmp_path, monkeypatch, chromium_browser, ledger
):
    """The shipped chip's POST names no session, so the record is the Lab bar's.

    The chip lost its session id in 3.5 — it reads the deployment's record and
    sends ``session_id: null`` on every gesture. The route reads that as "a
    surface with no session to name", which is what the JupyterLab bar is, and
    files the gesture on ``jupyter_lab`` with a null session rather than on
    ``http_mutation`` with a hole in it.

    Both halves matter and only a browser has both: the route tests prove what
    the route does with a session-less body, and this proves the body the
    shipped module actually sends. A chip that quietly regained a session id
    would still pass every route test in the tree.

    The narrowing is used rather than the switch because it asks nothing —
    turning writes off only ever removes reach — so the audit record is the
    only thing between the click and the assertion.
    """
    with _chip_hub(tmp_path, monkeypatch) as (base_url, _app, _root):
        page, _session_id = _settled_chip(chromium_browser, base_url)
        try:
            _open_popover(page)
            _toggle(page, POSTURE_TARGET).click()
            expect(_row(page, POSTURE_TARGET)).to_have_attribute(
                "data-state", "sandbox", timeout=TIMEOUT
            )
        finally:
            page.close()

        # The gesture landed where the connector reads it, so the record below
        # is about a POST that really did something.
        assert _recorded_posture() == {POSTURE_TARGET: posture_store.POSTURE_SANDBOX}

    gestures = [
        entry
        for entry in ledger
        if entry.get("subject") == websocket_routes.AUDIT_SUBJECT_POSTURE_SET
    ]
    assert len(gestures) == 1, f"expected exactly one posture gesture, got {gestures}"
    gesture = gestures[0]
    assert gesture["session"] is None, gesture
    assert gesture["surface"] == websocket_routes.LAB_MUTATION_SURFACE, gesture
    assert gesture["decision"] == "allowed", gesture


# ---------------------------------------------------------------------------
# (5) a real hand-off, then the roster from the Simple view
# ---------------------------------------------------------------------------

#: Every answer ``POST /api/session/{key}/handoff`` is allowed to give for a
#: well-formed key on a live hub. 200 is the flip taken and 204 the caller
#: hanging up mid-wait; 409 is the "one process at a time" family (the outgoing
#: surface is still running, another request took the key, the chat was torn
#: down under it); 503 is a hub that cannot start the incoming surface at all —
#: no Agent SDK installed, or no chat capacity. Which one comes back depends on
#: the host, and NONE of them may move the deployment's target, which is the
#: whole assertion. A 400 would mean the key was not a session UUID and is
#: deliberately NOT in the set: that is a broken test, not a refusal.
_HANDOFF_ANSWERS = frozenset({200, 204, 409, 503})


def _open_display_menu(page: Page) -> None:
    """Open the header display-menu popover, where the view segmented control lives."""
    page.locator("#display-menu .display-menu-trigger").click()
    expect(page.locator("#display-menu .display-menu-card")).to_have_class(
        re.compile(r"\bopen\b"), timeout=TIMEOUT
    )


def _flip_view(page: Page, mode: str) -> None:
    """Drive the header's view control to *mode*, the way an operator does.

    A mode pick deliberately leaves the card open, so it is dismissed here to
    return the caller to the collapsed state — the chip and its popover live in
    the same header and an open card sits over them.
    """
    _open_display_menu(page)
    page.locator(
        f'#display-menu .display-menu-view .display-seg-option[data-mode="{mode}"]'
    ).click()
    page.locator("#display-menu .display-menu-trigger").click()
    expect(page.locator("#display-menu .display-menu-card")).not_to_have_class(
        re.compile(r"\bopen\b"), timeout=TIMEOUT
    )
    expect(page.locator("html")).to_have_attribute("data-ui-mode", mode, timeout=TIMEOUT)


def test_a_real_handoff_and_the_simple_view_leave_the_deployment_where_it_is(
    tmp_path, monkeypatch, chromium_browser
):
    """A surface hand-off moves a session; it does not move the deployment.

    The order is the point:

    1. the deployment is on ``va`` at generation 1, and a real hand-off is
       requested for the tab's own session through the shipped route. Whatever
       it answers — the flip taken, or refused because one process holds the
       session, or refused for capacity — the record is asserted UNMOVED. Under
       the per-session model this is where the target went back to baseline;
    2. the tab is flipped to the Simple view, where the terminal deliberately
       does not connect and the console holds the session instead. The chip
       still paints, still names ``va``, and is still writable — it is a
       deployment surface now, so a view with no PTY behind it changes nothing
       about it;
    3. Switch is exercised from that Simple view and moves the record one
       generation;
    4. the tab is flipped back to Expert, and the generation is asserted to be
       exactly where the switch left it — not re-derived, not reset.

    A live controls server is planted so the switch gate has a reachability
    sweep to read.
    """
    with _chip_hub(tmp_path, monkeypatch) as (base_url, _app, root):
        _publish_report(root)
        page, session_id = _settled_chip(chromium_browser, base_url)
        try:
            assert session_id, "the expert tab minted no PTY session to hand off"

            # --- 1. the hand-off, for real -------------------------------
            # One surface holds a session at a time, and the Expert tab is
            # holding this one on its terminal socket right now. So the honest
            # answer is the refusal that rule produces, and it is asserted by
            # its slug rather than by its sentence.
            answer = requests.post(
                f"{base_url}/api/session/{session_id}/handoff",
                json={"to": "simple"},
                timeout=60,
            )
            assert answer.status_code in _HANDOFF_ANSWERS, (
                f"the hand-off answered {answer.status_code}, which is outside the "
                f"route's contract: {answer.text}"
            )
            if answer.status_code == 409:
                assert answer.json()["detail"]["error"] == ERROR_SESSION_ATTACHED_ELSEWHERE, (
                    answer.text
                )

            settled = _read_record()
            assert settled is not None, "the hand-off left no record"
            assert settled.target == ACTIVE_TARGET, settled
            assert settled.generation == RECORD_GENERATION, settled

            # --- 2. the Simple view still speaks for the deployment ------
            # The flip is the client's own hand-off: chat.js POSTs the same
            # route as it takes the session onto the console. Watching for it
            # is what makes this a hand-off lane rather than a mode-flip lane —
            # the assertion below is about the request the SHIPPED page makes.
            handoffs: list[str] = []
            page.on(
                "request",
                lambda request: (
                    handoffs.append(request.url)
                    if request.method == "POST" and "/handoff" in request.url
                    else None
                ),
            )
            _flip_view(page, "simple")
            page.wait_for_function(
                "() => document.querySelector('#control-target-chip') !== null",
                timeout=TIMEOUT,
            )
            expect(page.locator(CHIP)).to_be_visible(timeout=TIMEOUT)
            expect(page.locator(CHIP)).to_have_attribute(
                "data-enforceable", "true", timeout=TIMEOUT
            )
            expect(page.locator(CHIP_SHORT)).to_have_text(NAMES[ACTIVE_TARGET], timeout=TIMEOUT)
            assert handoffs, (
                "flipping to Simple issued no hand-off — this lane would then be a "
                "mode flip, not a hand-off"
            )
            assert all(session_id in url for url in handoffs), (
                f"the client handed off some other key: {handoffs}"
            )

            # --- 3. and Switch works from it -----------------------------
            _open_popover(page)
            _row(page, SWITCH_TARGET).locator(".ctc-switch").click()
            expect(page.locator(MODAL_TITLE)).to_have_text(
                f"Switch to {NAMES[SWITCH_TARGET]}?", timeout=TIMEOUT
            )
            page.locator(MODAL_CONFIRM).click()
            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
            expect(page.locator(CARD)).to_have_attribute(
                "data-target", SWITCH_TARGET, timeout=TIMEOUT
            )

            switched = _read_record()
            assert switched is not None
            assert switched.target == SWITCH_TARGET, switched
            assert switched.generation == RECORD_GENERATION + 1, switched

            # --- 4. the flip back changes nothing ------------------------
            page.keyboard.press("Escape")
            expect(page.locator(POPOVER)).to_have_count(0, timeout=TIMEOUT)
            _flip_view(page, "expert")
            expect(page.locator(CHIP_SHORT)).to_have_text(NAMES[SWITCH_TARGET], timeout=TIMEOUT)

            final = _read_record()
            assert final is not None
            assert final.target == SWITCH_TARGET, final
            assert final.generation == switched.generation, (
                "flipping the view re-derived the deployment's generation",
                final,
            )
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (7) the machine already held is not a switch
# ---------------------------------------------------------------------------


def test_the_machine_already_held_is_offered_no_switch_and_pends_nothing(
    tmp_path, monkeypatch, chromium_browser
):
    """The machine the deployment stands on is offered no gesture, and pends none.

    Two halves of one fact, which is why they are one lane. The roster answers
    ``already_active`` for the machine of record, so the popover renders it as
    the card and not as a row — a card has no Switch slot, so there is no
    gesture here that could ask for the target the deployment is already on.

    A request for it can still arrive: the agent's own tool makes one, and so
    does a client whose roster went stale between the render it read and the
    confirm it sent. The route's answer is what keeps that from stranding every
    watching page — nothing is written, no generation is minted, and the 202
    carries the generation the fleet is already on. ``switching…`` is entered on
    a pending request and left when a terminus resolves it, so a generation
    minted here would leave a wait that nothing can ever end: no terminus is
    coming, and the page would sit there until its local TTL reported
    ``request_expired`` for a request that in fact succeeded.

    The negative is not taken at an arbitrary moment. The chip's own read is
    watched past the request, so the assertions below are made after the read
    that would have carried a pending state has happened.

    No reachability sweep is planted, deliberately: ``_switch_mutation`` answers
    the already-held target before it reaches the convergence check or the gate,
    so a sweep is not part of what this lane exercises and planting one would
    suggest it was.
    """
    with _chip_hub(tmp_path, monkeypatch) as (base_url, _app, _root):
        page, _session_id = _settled_chip(chromium_browser, base_url)
        try:
            # (i) the interface offers no way to ask for it.
            _open_popover(page)
            expect(page.locator(CARD)).to_have_attribute(
                "data-target", ACTIVE_TARGET, timeout=TIMEOUT
            )
            expect(page.locator(f"{CARD} .ctc-switch")).to_have_count(0)
            expect(_row(page, ACTIVE_TARGET)).to_have_count(0)
            page.keyboard.press("Escape")
            expect(page.locator(POPOVER)).to_have_count(0, timeout=TIMEOUT)

            before = _read_record()
            assert before is not None, "the hub started with no control-context record"
            settled_state = page.locator(CHIP_STATE).inner_text()

            page.evaluate(_PENDING_PROBE)
            _install_probe(page)

            # (ii) the request the shipped chip would send. No session id:
            #      the chip stopped being a session surface, so its body
            #      carries a target and nothing else.
            answer = requests.post(
                f"{base_url}/api/terminal/target",
                json={"target": ACTIVE_TARGET},
                timeout=60,
            )
            assert answer.status_code == 202, answer.text
            body = answer.json()
            assert body["target"] == ACTIVE_TARGET, body
            assert body["generation"] == RECORD_GENERATION, body
            assert body["request_id"], body
            assert body["detail"] == control_context.unchanged_detail(
                ACTIVE_TARGET, RECORD_GENERATION
            ), body

            # (iii) nothing moved, and no terminus was written for it.
            after = _read_record()
            assert after is not None
            assert after.target == ACTIVE_TARGET, after
            assert after.generation == RECORD_GENERATION, after
            assert after.last_switch == before.last_switch, (
                "a request that moved nothing wrote a terminus",
                after.last_switch,
            )

            # (iv) and no page waits for it. The poll tick is the barrier:
            #      the read that would have carried a pending state has
            #      happened by the time these are taken.
            _await_poll_tick(page)
            assert page.evaluate("() => window.__ctcPending") == [], (
                "the chip entered switching… for a request that moved nothing"
            )
            expect(page.locator(f"{CHIP}[data-pending]")).to_have_count(0)
            assert page.locator(CHIP_STATE).inner_text() == settled_state
        finally:
            page.close()
