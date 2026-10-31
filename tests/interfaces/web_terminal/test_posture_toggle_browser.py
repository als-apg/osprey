"""Browser tests: the control-target header chip, end to end.

The chip in the page header — ``● Simulator · writes asked ▾`` — and the
popover behind it are the operator's only route between a machine's write
positions, and
the only route onto another control target. Every interesting part of that is
client-side behavior a FastAPI TestClient cannot observe: the card, the rows
and both confirms are built in the DOM by ``control-target-popover.js``, and
every one of them repaints from a *re-read* of ``GET /api/terminal/posture``
rather than from what the module last POSTed. A real browser is what proves
the chip an operator looks at agrees with the record the connector will read.

**The chip speaks for the deployment, not for a session.** It reads the
control-context record — one file, one target, one posture map — so it needs
no session id to paint, sends none on its read, and shows the same answer in
every tab. That is why the arrangement below is a *file written before the
server starts* rather than a record addressed to some process's pid.

Every machine's write switch has three positions — ``off``, ``asked`` (each
write prompts first) and ``allowed`` (writes run with no prompt) — rendered as
one ``role="radiogroup"`` of three ``role="radio"`` options whose
``aria-checked`` is the STORED position. A move down (a narrowing) applies at
once; a move up (a widening) confirms.

Coverage (one test each):

  (a) the chip names the machine the deployment stands on by its display name,
      and the popover renders that machine as the card and every other
      configured target as a row, the server's own label demoted to tooltips.
  (a2) a row per position: off, asked and allowed each render as the checked
      option, the row's word and the group's accessible name, under the
      ``allowed`` ceiling where nothing is held.
  (a3) under an ``asked`` ceiling the ``allowed`` option is disabled with the
      tier's sentence, and a stored ``allowed`` is the capped row: checked,
      disabled, reading ``asked``; the route refuses ``allowed`` with
      ``tier_caps``.
  (b) narrowing to off asks nothing, lands in the login's switch where every
      write path reads it, and respawns nothing.
  (c) every widening confirms — off→asked and asked→allowed, each in its own
      words — Cancel is a true no-op on the row *and* in the record, and a
      narrowing allowed→asked applies with no dialog.
  (c2) turning off moves the endpoint the row's tooltip names off the write
      gateway and onto the read one, and the widening confirm names no
      endpoint at all.
  (c3) the keyboard: arrows, Home and End move focus among the options and
      never POST; Space and Enter commit — a narrowing at once, a widening
      into its confirm.
  (d) Switch confirms, is accepted as ``202``, moves the record's target and
      generation, and holds the chip and the row in ``switching…`` for as long
      as a live controls server still reports the generation before it.
  (e) a refusal raised from a confirm stays inside the dialog, which stays up.
  (f) both UI modes render the SAME popover DOM and show all of it.

The chip criteria, end to end — each one closed against the write verdict
itself (``posture_store.write_verdict``), and each one asserting that every
machine's ``data-state`` and the chip's own equal what that verdict and the
record say (:func:`assert_chip_matches_verdict`):

  (g) a rehearsal off→asked confirms once — the waiver is offered and taken —
      and applies directly after (Shift re-shows it); a move to ``allowed``
      offers no waiver even there. The verdict allows this login on that
      machine only.
  (h) the real machine confirms every widening, every time, with no waiver.
  (i) "Turn all writes off" works from a terminal that follows another, and
      afterwards every machine refuses with ``switch_off``.
  (j) a stop the store cannot record, on a readable record, says it was not
      taken and the machine stays armed.
  (j2) a widening a stop overtook answers ``409``: its confirm closes, the row
      says so, nothing is armed — and the next click, on a fresh read, lands.
  (k) an unreadable record shows the verdict's refusal on every machine, checks
      no option and never shows one off; off stays selectable.
  (l) a readonly login and an app render show nothing armed as selectable:
      both armed options are disabled with the verdict's sentence, and the
      route refuses them.

Session bootstrapping. The chip needs no session id, but the page still mints
one and case (b) reads it back to name the PTY it asserts was not respawned. It
arrives the way it does in production: a plain page load opens a NEW terminal
WebSocket, the route mints the session UUID itself (it dictates it on the CLI's
command line) and confirms it in a ``session_info`` frame — no Claude binary
and no session discovery involved. ``terminal.js`` writes it to
``localStorage['osprey-pty-session']`` on that same frame. The PTY command is a
long-lived ``sleep`` because the route appends ``--session-id``/``--resume``
arguments that ``echo`` would choke on. In the Simple view the terminal
deliberately does not connect — the console holds that view's session — and the
chip paints either way, which is the point of it no longer being a session
surface.

The record is seeded with this login's switch at ``asked`` for every machine
unless a test asks otherwise (``chip_hub(switch=...)``), so the first gesture a
test can make is the narrowing that asks nothing.

One fact has to be true before any toggle in the popover can move, and
:func:`chip_hub` arranges it: **a control-context record exists under the
stamped root, owned by this process, before ``create_app`` runs.** Without a
record the roster is read-only (``store_available: false``) and every toggle is
locked; without ownership this terminal follows another and the write routes
answer ``409``. Writing it before startup is also what keeps the lifespan's
claim a merge rather than a mint from ``load_osprey_config``.
``data-enforceable="true"`` on the chip is the observable proof the arrangement
landed — the attribute is fed by ``contextWritable(state)``, which is exactly
"this terminal's changes would be recorded".

``SessionDiscovery.snapshot_session_ids`` is patched to an empty set (the same
seam ``test_posture_routes.py`` uses) so no test depends on a ``.jsonl`` a real
model turn would have written. Nothing in the posture or target routes gates on
it any more, and pinning it empty keeps that true rather than accidental.

The agent-data root is stamped with ``OSPREY_AGENT_DATA_ROOT``, which is the
one seam the record reader and ``posture_store`` both prefer — patching
``resolve_shared_data_root`` would redirect one of them and leave the other
writing into the repository's own ``var/agent_data``.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_posture_toggle_browser.py -m browser -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import os
from contextlib import suppress
from typing import Any

import pytest
import requests

from osprey_connectors import control_context
from osprey_connectors.identity import acting_identity
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors_tests._harness import owner, write_payload
from tests.interfaces.web_terminal._control_context import read_record
from tests.interfaces.web_terminal._posture_chip import (
    ACTIVE_TARGET,
    ALL_ALLOWED,
    CARD,
    CHIP,
    CHIP_SHORT,
    CHIP_STATE,
    ENDPOINT_TARGET,
    FOOT_NOTE,
    HOST_PORT,
    KINDS,
    LABELS,
    MODAL_CANCEL,
    MODAL_CONFIRM,
    MODAL_ERROR,
    MODAL_TITLE,
    NAMES,
    OPEN_MODAL,
    POPOVER,
    POSITIONS,
    POSTURE_TARGET,
    READ_ENDPOINT,
    RECORD_GENERATION,
    SWITCH_TARGET,
    TIMEOUT,
    WRITE_ENDPOINT,
    assert_chip_matches_verdict,
    assert_options,
    chip_hub,
    open_popover,
    popover_machine,
    popover_row,
    publish_report,
    recorded_position,
    settled_chip,
    write_verdict_for,
    writes_option,
    writes_switch,
)

# without playwright, the chromium_browser fixture skips the suite
with suppress(ImportError):
    from playwright.sync_api import Page, expect

pytestmark = [pytest.mark.browser, pytest.mark.slow]


#: The sentence a widening carries — in its confirm, or once on the popover's
#: status line when the confirm was waived (control-target-facts.js).
TURN_ON_SENTENCE = (
    "Runs already going — a sandbox run or a notebook cell — keep the state they "
    "started with; the next one follows."
)

#: What a stop the store could not record says (C-POST 503 on off).
STOP_NOT_TAKEN = "The stop did not take effect — writes are still on."

#: What a widening a stop overtook says (C-POST 409 stopped_since).
STOPPED_SINCE = (
    "Writes were turned off after this turn-on began — nothing was turned on. "
    "Click again to turn them on."
)

#: The tier's ceiling, as the held ``allowed`` option and the capped row say it.
TIER_CAPS = "your tier caps writes at asked"

#: The route every switch gesture POSTs to.
POSTURE_ROUTE = "/api/terminal/posture"


def _pty_pid(app: Any, session_id: str) -> int:
    """The pid of the PTY this session runs in."""
    session = app.state.pty_registry.get_session(session_id)
    assert session is not None, f"no PTY session registered for {session_id}"
    pid = session.pid
    assert isinstance(pid, int) and pid > 0, f"the PTY reported no usable pid: {pid!r}"
    return pid


def _widen_title(target: str, position: str) -> str:
    return f"Set writes to {position} on {NAMES[target]}?"


def _move(page: Page, target: str, position: str, **click: Any) -> None:
    """Click *target*'s *position* option (re-resolved: the popover is rebuilt)."""
    writes_option(page, target, position).click(**click)


def _narrow(page: Page, target: str, position: str = "off") -> None:
    """Narrow *target* through the popover, and wait for the row to say so.

    Applies on click — a narrowing only ever removes reach, so no confirm is in
    the way, and the row's ``data-state`` flipping is the re-read landing.
    """
    _move(page, target, position)
    expect(popover_machine(page, target)).to_have_attribute("data-state", position, timeout=TIMEOUT)
    expect(page.locator(OPEN_MODAL)).to_have_count(0)


def _widen(page: Page, target: str, position: str) -> None:
    """Widen *target* through its confirm, and wait for the row to say so."""
    _move(page, target, position)
    expect(page.locator(MODAL_TITLE)).to_have_text(_widen_title(target, position), timeout=TIMEOUT)
    page.locator(MODAL_CONFIRM).click()
    expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
    expect(popover_machine(page, target)).to_have_attribute("data-state", position, timeout=TIMEOUT)


def _switch_posts(page: Page) -> list[str]:
    """Collect every POST this page makes to the switch route, as it happens."""
    posts: list[str] = []
    page.on(
        "request",
        lambda request: (
            posts.append(request.url)
            if request.method == "POST" and request.url.endswith(POSTURE_ROUTE)
            else None
        ),
    )
    return posts


def _focused_position(page: Page) -> str | None:
    return page.evaluate("() => document.activeElement?.dataset?.position ?? null")


# ---------------------------------------------------------------------------
# (a) the chip and its roster
# ---------------------------------------------------------------------------


def test_the_chip_names_the_recorded_target_and_lists_every_row(
    tmp_path, monkeypatch, chromium_browser
):
    """The chip speaks for the machine the deployment stands on; the popover lists all.

    The record puts the deployment on the simulator, so the chip reads
    ``Simulator · writes asked`` — the display name the render minted, not the
    target name — and the popover renders that machine as the card ("The agent
    is on") and each other configured target as a row named the same way, the
    server's own label demoted to the identity tooltip. The card offers no
    Switch; switching to the target you are on is a no-op.
    """
    with chip_hub(tmp_path, monkeypatch) as (base_url, _app, _root):
        page, _session_id = settled_chip(chromium_browser, base_url)
        try:
            # The chip's home is the sidebar's control-system block, and the
            # terminal card's own header carries no control-target affordance.
            # The badge module is deleted on this branch, so this is the guard
            # against a second control-target affordance reappearing in the
            # card rather than a live check.
            expect(page.locator(f"#sidebar-target {CHIP}")).to_have_count(1)
            expect(page.locator(".terminal-header .posture-badge")).to_have_count(0)
            expect(page.locator(f".terminal-header {CHIP}")).to_have_count(0)

            expect(page.locator(CHIP_SHORT)).to_have_text(NAMES[ACTIVE_TARGET], timeout=TIMEOUT)
            expect(page.locator(CHIP_STATE)).to_have_text("writes asked", timeout=TIMEOUT)
            expect(page.locator(CHIP)).to_have_attribute("data-state", "asked")
            expect(page.locator(CHIP)).to_have_attribute("data-target-kind", KINDS[ACTIVE_TARGET])
            expect(page.locator(CHIP)).to_have_attribute("aria-expanded", "false")

            open_popover(page)

            card = page.locator(CARD)
            expect(card).to_have_attribute("data-target", ACTIVE_TARGET)
            expect(page.locator(f"{POPOVER} > .osprey-popover-title").first).to_have_text(
                "The agent is on"
            )
            expect(card.locator(".ctc-name")).to_have_text(NAMES[ACTIVE_TARGET])
            expect(page.locator(f"{POPOVER} .ctc-row")).to_have_count(2, timeout=TIMEOUT)
            for target in (SWITCH_TARGET, POSTURE_TARGET):
                row = popover_row(page, target)
                expect(row.locator(".ctc-name")).to_have_text(NAMES[target])
                expect(row.locator(".ctc-switch-state")).to_have_text("asked")
                tip = row.locator(".ctc-tip").text_content()
                assert tip and LABELS[target] in tip, tip

            expect(card.locator(".ctc-switch")).to_have_count(0)
            expect(popover_row(page, SWITCH_TARGET).locator(".ctc-switch")).to_be_visible()
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (a2) one row per position, under the top ceiling
# ---------------------------------------------------------------------------


def test_each_position_renders_as_its_checked_option_and_its_word(
    tmp_path, monkeypatch, chromium_browser
):
    """off, asked and allowed each read the same on the option, the row and the group.

    One machine per position: the card (the simulator) off, the stand-in at
    ``asked``, the real machine at ``allowed``. The render's ceiling is
    ``allowed`` and no login tier is set, so nothing is held: every option of
    every machine is enabled, and the stored one is the checked radio. The
    tab stop is that checked option and only it, and the group's accessible
    name says the word and the machine — what a screen reader announces.
    """
    switch = {POSTURE_TARGET: "asked", SWITCH_TARGET: "allowed"}
    with chip_hub(tmp_path, monkeypatch, switch=switch) as (base_url, app, _root):
        page, _session_id = settled_chip(chromium_browser, base_url)
        try:
            expect(page.locator(CHIP_STATE)).to_have_text("writes off", timeout=TIMEOUT)
            open_popover(page)
            expected = {ACTIVE_TARGET: "off", **switch}
            for target, position in expected.items():
                node = popover_machine(page, target)
                expect(node).to_have_attribute("data-state", position, timeout=TIMEOUT)
                expect(node.locator(".ctc-switch-state")).to_have_text(position)
                group = writes_switch(page, target)
                expect(group).to_have_attribute(
                    "aria-label", f"writes {position} for {NAMES[target]}"
                )
                assert_options(page, target, checked=position)
                for option in POSITIONS:
                    expect(writes_option(page, target, option)).to_have_attribute(
                        "tabindex", "0" if option == position else "-1"
                    )
                    expect(writes_option(page, target, option)).to_have_text(option)
                assert recorded_position(target) == position
            assert_chip_matches_verdict(page, app)
            assert write_verdict_for(app, SWITCH_TARGET).position == "allowed"
            assert write_verdict_for(app, POSTURE_TARGET).position == "asked"
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (a3) an asked ceiling: allowed held, and the capped row
# ---------------------------------------------------------------------------


def test_an_asked_ceiling_holds_allowed_and_caps_a_stored_allowed(
    tmp_path, monkeypatch, chromium_browser
):
    """Under an ``asked`` tier, ``allowed`` is disabled with its sentence.

    The deployment's tier is ``asked`` (``writes: asked`` in the render — the
    ceiling the write verdict itself applies on a host), so every machine's
    ``allowed`` option is
    disabled and its hover carries the tier's sentence; ``off`` and ``asked``
    stay selectable — the tier caps a position, it does not lock the switch.

    The stand-in's switch was stored at ``allowed`` before the tier applied.
    That is the capped row: the stored option stays checked (and disabled),
    while the word beside it says what writes actually do — ``asked`` — with
    the sentence on its hover. The route behind the disabled option refuses the
    move too, with ``tier_caps`` and the ceiling: a disabled button is a
    courtesy, the 403 is the rule.
    """
    switch = {SWITCH_TARGET: "asked", POSTURE_TARGET: "allowed"}
    with chip_hub(tmp_path, monkeypatch, switch=switch, writes="asked") as (base_url, app, _root):
        page, _session_id = settled_chip(chromium_browser, base_url)
        try:
            open_popover(page)
            # A machine at asked: allowed held, with the tier's sentence.
            assert_options(page, SWITCH_TARGET, checked="asked", disabled=("allowed",))
            expect(writes_option(page, SWITCH_TARGET, "allowed")).to_have_attribute(
                "title", TIER_CAPS
            )
            # A machine at off: the same hold, and asked still selectable.
            assert_options(page, ACTIVE_TARGET, checked="off", disabled=("allowed",))

            # The capped row.
            capped = popover_row(page, POSTURE_TARGET)
            expect(capped).to_have_attribute("data-state", "asked", timeout=TIMEOUT)
            assert_options(page, POSTURE_TARGET, checked="allowed", disabled=("allowed",))
            expect(writes_option(page, POSTURE_TARGET, "asked")).to_have_attribute("tabindex", "0")
            word = capped.locator(".ctc-switch-state")
            expect(word).to_have_text("asked")
            expect(word).to_have_attribute("title", TIER_CAPS)
            assert recorded_position(POSTURE_TARGET) == "allowed"
            assert write_verdict_for(app, POSTURE_TARGET).position == "asked"
            assert_chip_matches_verdict(page, app)

            answer = requests.post(
                f"{base_url}{POSTURE_ROUTE}",
                json={"session_id": None, "target": SWITCH_TARGET, "state": "allowed"},
                timeout=60,
            )
            assert answer.status_code == 403, answer.text
            detail = answer.json()["detail"]
            assert detail["reason"] == "tier_caps", detail
            assert detail["ceiling"] == "asked", detail
            assert recorded_position(SWITCH_TARGET) == "asked"
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (b) narrowing to off
# ---------------------------------------------------------------------------


def test_turning_off_asks_nothing_lands_in_the_record_and_respawns_nothing(
    tmp_path, monkeypatch, chromium_browser
):
    """Off applies on click, is written where the connector reads, and is free.

    * no confirm — a narrowing only ever removes reach;
    * the record agrees, because the login's ``switch`` in that file is what
      the write verdict consults on every path, keyed by TARGET alone;
    * the PTY is the same process afterwards (FR17): the switch is read live
      on every write, so a respawn would cost the operator their session.
    """
    with chip_hub(tmp_path, monkeypatch) as (base_url, app, _root):
        page, session_id = settled_chip(chromium_browser, base_url)
        assert session_id, "the terminal card never settled on a session id"
        pty_pid = _pty_pid(app, session_id)
        try:
            open_popover(page)
            expect(popover_row(page, POSTURE_TARGET)).to_have_attribute("data-state", "asked")
            assert_options(page, POSTURE_TARGET, checked="asked")

            _narrow(page, POSTURE_TARGET)

            expect(popover_row(page, POSTURE_TARGET).locator(".ctc-switch-state")).to_have_text(
                "off"
            )
            assert_options(page, POSTURE_TARGET, checked="off")
            # The card is untouched: a position is per target.
            expect(page.locator(CARD)).to_have_attribute("data-state", "asked")

            assert recorded_position(POSTURE_TARGET) == "off"
            assert write_verdict_for(app, POSTURE_TARGET).reason == "switch_off"
            assert_chip_matches_verdict(page, app)

            assert _pty_pid(app, session_id) == pty_pid, "the session was respawned"
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (c) widening confirms; narrowing does not
# ---------------------------------------------------------------------------


def test_every_widening_confirms_and_cancel_changes_nothing(
    tmp_path, monkeypatch, chromium_browser
):
    """off→asked and asked→allowed each ask in their own words; allowed→asked does not.

    A widening is the gesture after which a write can land, or land with no
    prompt, so it is the direction that confirms. The dialog names the machine
    and the position; the popover stays open beneath it, the option that raised
    it is marked pending, and nothing is applied until it is confirmed — which
    is why the record is asserted on both sides of the cancellation.
    """
    with chip_hub(tmp_path, monkeypatch, switch={}) as (base_url, app, _root):
        page, _session_id = settled_chip(chromium_browser, base_url)
        try:
            open_popover(page)
            expect(popover_row(page, POSTURE_TARGET)).to_have_attribute(
                "data-state", "off", timeout=TIMEOUT
            )

            # --- off -> asked, cancelled ---
            _move(page, POSTURE_TARGET, "asked")
            expect(page.locator(MODAL_TITLE)).to_have_text(
                _widen_title(POSTURE_TARGET, "asked"), timeout=TIMEOUT
            )
            expect(page.locator(MODAL_CONFIRM)).to_have_text("Set writes to asked")
            expect(page.locator(f"{OPEN_MODAL} .posture-modal-body")).to_contain_text(
                "still prompt"
            )
            expect(page.locator(POPOVER)).to_be_visible()
            expect(writes_option(page, POSTURE_TARGET, "asked")).to_have_attribute(
                "data-pending", "true", timeout=TIMEOUT
            )
            assert recorded_position(POSTURE_TARGET) == "off"
            page.locator(MODAL_CANCEL).click()

            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
            expect(popover_row(page, POSTURE_TARGET)).to_have_attribute("data-state", "off")
            expect(writes_option(page, POSTURE_TARGET, "asked")).not_to_have_attribute(
                "data-pending", "true", timeout=TIMEOUT
            )
            assert recorded_position(POSTURE_TARGET) == "off"

            # --- off -> asked, confirmed ---
            _widen(page, POSTURE_TARGET, "asked")
            assert recorded_position(POSTURE_TARGET) == "asked"
            assert write_verdict_for(app, POSTURE_TARGET).position == "asked"

            # --- asked -> allowed: its own words, and no waiver anywhere ---
            _move(page, POSTURE_TARGET, "allowed")
            expect(page.locator(MODAL_TITLE)).to_have_text(
                _widen_title(POSTURE_TARGET, "allowed"), timeout=TIMEOUT
            )
            expect(page.locator(MODAL_CONFIRM)).to_have_text("Set writes to allowed")
            expect(page.locator(f"{OPEN_MODAL} .posture-modal-body")).to_contain_text("no prompt")
            expect(page.locator(f"{OPEN_MODAL} .posture-modal-skip")).to_have_count(0)
            assert recorded_position(POSTURE_TARGET) == "asked"
            page.locator(MODAL_CONFIRM).click()
            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
            expect(popover_row(page, POSTURE_TARGET)).to_have_attribute(
                "data-state", "allowed", timeout=TIMEOUT
            )
            assert recorded_position(POSTURE_TARGET) == "allowed"

            # --- allowed -> asked: a narrowing, so no dialog ---
            _narrow(page, POSTURE_TARGET, "asked")
            assert recorded_position(POSTURE_TARGET) == "asked"
            assert_chip_matches_verdict(page, app)
        finally:
            page.close()


def test_turning_off_moves_the_tooltip_onto_the_read_gateway(
    tmp_path, monkeypatch, chromium_browser
):
    """A row with writes off names the gateway it is actually talking to.

    Armed, the endpoint a row names is the write gateway; narrowed to off it is
    the read gateway, and the tooltip has to say so. The narrowing is made
    through the popover's own option and therefore the real POST. The widening
    confirm names no endpoint whatsoever: the one the roster carries describes
    reads under the posture the dialog is about to leave.
    """
    with chip_hub(tmp_path, monkeypatch) as (base_url, _app, _root):
        page, _session_id = settled_chip(chromium_browser, base_url)
        try:
            open_popover(page)

            ident = popover_row(page, ENDPOINT_TARGET).locator(".ctc-tip-ident")
            expect(ident).to_contain_text(WRITE_ENDPOINT, timeout=TIMEOUT)
            assert READ_ENDPOINT not in (ident.text_content() or "")

            _narrow(page, ENDPOINT_TARGET)

            ident = popover_row(page, ENDPOINT_TARGET).locator(".ctc-tip-ident")
            expect(ident).to_contain_text(READ_ENDPOINT, timeout=TIMEOUT)
            tip = ident.text_content() or ""
            assert WRITE_ENDPOINT not in tip
            assert LABELS[ENDPOINT_TARGET] in tip

            _move(page, ENDPOINT_TARGET, "asked")
            expect(page.locator(MODAL_TITLE)).to_have_text(
                _widen_title(ENDPOINT_TARGET, "asked"), timeout=TIMEOUT
            )
            body = page.locator(f"{OPEN_MODAL} .posture-modal-body").text_content() or ""
            assert body.strip(), "the confirm rendered an empty body"
            assert not HOST_PORT.search(body), body
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (c3) the keyboard
# ---------------------------------------------------------------------------


def test_arrows_move_focus_without_posting_and_space_or_enter_commits(
    tmp_path, monkeypatch, chromium_browser
):
    """Arrow keys, Home and End only move focus; Space and Enter commit.

    The radiogroup's keyboard contract: an operator walking the options must
    not move a machine's writes by walking. So every movement key is pressed
    and the page is shown to have POSTed nothing, then Space commits the
    focused narrowing at once and Enter raises the focused widening's confirm
    — and the focus survives the rebuild the narrowing's re-read causes.
    """
    with chip_hub(tmp_path, monkeypatch) as (base_url, _app, _root):
        page, _session_id = settled_chip(chromium_browser, base_url)
        posts = _switch_posts(page)
        try:
            open_popover(page)
            group = writes_switch(page, POSTURE_TARGET)
            expect(group).to_have_attribute(
                "aria-label", f"writes asked for {NAMES[POSTURE_TARGET]}"
            )
            expect(group.locator('[tabindex="0"]')).to_have_count(1)
            writes_option(page, POSTURE_TARGET, "asked").focus()
            assert _focused_position(page) == "asked"

            for key, lands in (
                ("ArrowRight", "allowed"),
                ("ArrowRight", "off"),
                ("ArrowLeft", "allowed"),
                ("ArrowDown", "off"),
                ("ArrowUp", "allowed"),
                ("Home", "off"),
                ("End", "allowed"),
            ):
                page.keyboard.press(key)
                page.wait_for_function(
                    "(p) => document.activeElement?.dataset?.position === p", arg=lands
                )
            page.wait_for_timeout(300)
            assert posts == [], posts
            expect(page.locator(OPEN_MODAL)).to_have_count(0)
            assert recorded_position(POSTURE_TARGET) == "asked"

            # Space on off: a narrowing, committed at once.
            page.keyboard.press("Home")
            page.keyboard.press(" ")
            expect(popover_row(page, POSTURE_TARGET)).to_have_attribute(
                "data-state", "off", timeout=TIMEOUT
            )
            expect(page.locator(OPEN_MODAL)).to_have_count(0)
            assert recorded_position(POSTURE_TARGET) == "off"
            assert len(posts) == 1, posts
            page.wait_for_function(
                "() => document.activeElement?.dataset?.position === 'off'", timeout=TIMEOUT
            )

            # Enter on allowed: a widening, so it asks.
            page.keyboard.press("End")
            page.keyboard.press("Enter")
            expect(page.locator(MODAL_TITLE)).to_have_text(
                _widen_title(POSTURE_TARGET, "allowed"), timeout=TIMEOUT
            )
            assert len(posts) == 1, posts
            page.locator(MODAL_CANCEL).click()
            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
            assert recorded_position(POSTURE_TARGET) == "off"
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (d) switching
# ---------------------------------------------------------------------------


def test_switch_confirms_moves_the_record_and_waits_for_the_fleet(
    tmp_path, monkeypatch, chromium_browser
):
    """Switch asks, moves the record, and the chip waits out loud for the fleet.

    An accepted gesture writes the new target, the next generation and an
    ``applied`` terminus into the record in one mutation, so the record is the
    first assertion. A live controls server planted still reporting the
    generation BEFORE the switch is a server that has not rebuilt its
    connector yet; while it lags, the chip and the card say ``switching…``.
    The dialog names the position held on the machine being switched TO,
    because a position is per machine and does not travel.
    """
    with chip_hub(tmp_path, monkeypatch) as (base_url, _app, root):
        publish_report(root)
        page, _session_id = settled_chip(chromium_browser, base_url)
        try:
            open_popover(page)
            popover_row(page, SWITCH_TARGET).locator(".ctc-switch").click()

            expect(page.locator(MODAL_TITLE)).to_have_text(
                f"Switch to {NAMES[SWITCH_TARGET]}?", timeout=TIMEOUT
            )
            expect(page.locator(f"{OPEN_MODAL} .posture-modal-body")).to_contain_text(
                "Writes are asked there"
            )
            expect(page.locator(f"{OPEN_MODAL} .posture-modal-live")).to_have_text(
                "Live — writes move hardware"
            )
            page.locator(MODAL_CONFIRM).click()

            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
            expect(page.locator(CHIP)).to_have_attribute("data-pending", "true", timeout=TIMEOUT)
            expect(page.locator(CHIP_STATE)).to_have_text("switching…", timeout=TIMEOUT)
            expect(page.locator(CARD)).to_have_attribute(
                "data-target", SWITCH_TARGET, timeout=TIMEOUT
            )
            expect(page.locator(f"{CARD} .ctc-outcome")).to_contain_text(
                "switching…", timeout=TIMEOUT
            )
            expect(page.locator(CHIP)).to_have_attribute("data-target-kind", KINDS[SWITCH_TARGET])

            record = read_record()
            assert record is not None, "the record disappeared"
            assert record.target == SWITCH_TARGET, record
            assert record.generation == RECORD_GENERATION + 1, record
            assert record.last_switch is not None, record
            assert record.last_switch["status"] == control_context.SWITCH_APPLIED, record
            assert record.last_switch["target"] == SWITCH_TARGET, record
            assert record.last_switch["request_id"], record
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (e) a refusal stays where the operator is looking
# ---------------------------------------------------------------------------


def test_a_refused_switch_keeps_its_sentence_inside_the_confirm(
    tmp_path, monkeypatch, chromium_browser
):
    """A refusal raised from a confirm stays in the confirm, which stays up.

    A live controls server reporting ``applying`` for the record's current
    generation is a server between two targets; the route answers 409 naming
    its pid, and the sentence goes into the dialog the operator is looking at.
    Nothing was written, so the chip must not fall into ``switching…``.
    """
    with chip_hub(tmp_path, monkeypatch) as (base_url, _app, root):
        page, _session_id = settled_chip(chromium_browser, base_url)
        try:
            publish_report(
                root,
                last_switch={
                    "status": "applying",
                    "target": SWITCH_TARGET,
                    "generation": RECORD_GENERATION,
                    "request_id": "11111111-2222-3333-4444-555555555555",
                },
            )

            open_popover(page)
            popover_row(page, SWITCH_TARGET).locator(".ctc-switch").click()
            expect(page.locator(MODAL_CONFIRM)).to_be_visible(timeout=TIMEOUT)
            page.locator(MODAL_CONFIRM).click()

            error = page.locator(MODAL_ERROR)
            expect(error).to_be_visible(timeout=TIMEOUT)
            expect(error).to_contain_text("already in flight on pid", timeout=TIMEOUT)
            expect(page.locator(OPEN_MODAL)).to_have_count(1)
            expect(page.locator(CHIP)).not_to_have_attribute("data-pending", "true")

            record = read_record()
            assert record is not None
            assert record.target == ACTIVE_TARGET, record
            assert record.generation == RECORD_GENERATION, record
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (f) one DOM, two densities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ui_mode", ["expert", "simple"])
def test_both_ui_modes_render_the_same_row_and_differ_only_in_density(
    tmp_path, monkeypatch, chromium_browser, ui_mode
):
    """The popover renders one DOM and shows the whole of it in either mode.

    ``html[data-ui-mode]`` stays a CSS concern, and this design leaves it
    nothing in the popover to gate: the name, its ⓘ, the three-position
    switch and Switch are shown in either density, and the confirms are
    identical whichever mode the deployment renders.
    """
    with chip_hub(tmp_path, monkeypatch, ui_mode=ui_mode) as (base_url, _app, _root):
        page, _session_id = settled_chip(chromium_browser, base_url)
        try:
            expect(page.locator("html")).to_have_attribute("data-ui-mode", ui_mode)
            open_popover(page)

            row = popover_row(page, SWITCH_TARGET)

            expect(row.locator(".ctc-name")).to_have_text(NAMES[SWITCH_TARGET])
            assert row.locator(".ctc-tip-what").text_content() == "Writes move hardware"
            expect(row.locator(".ctc-switch-state")).to_have_text("asked")
            expect(writes_switch(page, SWITCH_TARGET)).to_be_visible()
            for position in POSITIONS:
                expect(writes_option(page, SWITCH_TARGET, position)).to_be_visible()
            expect(row.locator(".ctc-switch")).to_be_visible()
            expect(page.locator(FOOT_NOTE)).to_have_text("Applies to every session of this login")

            tip = row.locator(".ctc-tip").text_content()
            assert tip and LABELS[SWITCH_TARGET] in tip, tip
            assert row.locator(".ctc-meta").count() == 0

            row.locator(".ctc-switch").click()
            expect(page.locator(MODAL_TITLE)).to_have_text(
                f"Switch to {NAMES[SWITCH_TARGET]}?", timeout=TIMEOUT
            )
            page.locator(MODAL_CANCEL).click()
            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)

            _move(page, SWITCH_TARGET, "allowed")
            expect(page.locator(MODAL_TITLE)).to_have_text(
                _widen_title(SWITCH_TARGET, "allowed"), timeout=TIMEOUT
            )
            page.locator(MODAL_CANCEL).click()
            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (g) the rehearsal: one confirm for asked, then the verdict
# ---------------------------------------------------------------------------

#: A login that is not the one signed in: the switch is one person's.
OTHER_LOGIN = "someone-else"


def test_a_rehearsal_widening_to_asked_confirms_once_and_arms_only_that_login(
    tmp_path, monkeypatch, chromium_browser
):
    """The rehearsal's off→asked asks once; ``allowed`` asks every time, even there.

    A live controls server publishes the stand-in as the rehearsal, and every
    switch starts off. The first off→asked confirms with the waiver offered,
    which the operator takes; from then on that move applies directly and the
    skipped confirm's sentence is said once on the status line, while Shift
    re-shows the dialog. A move to ``allowed`` removes the prompt, so its
    confirm offers no waiver on any machine.

    What the move unlocked is read from the verdict: this login on the
    rehearsal passes at ``asked``; the other machines meet ``switch_off``;
    another login is refused.
    """
    with chip_hub(tmp_path, monkeypatch, switch={}, rehearsal=True) as (base_url, app, _root):
        page, _session_id = settled_chip(chromium_browser, base_url)
        try:
            open_popover(page)
            row = popover_row(page, POSTURE_TARGET)
            expect(row).to_have_attribute("data-target-kind", "rehearsal", timeout=TIMEOUT)
            expect(row).to_have_attribute("data-state", "off")
            expect(row.locator(".ctc-name")).to_have_text(NAMES[POSTURE_TARGET])

            # --- the one confirm, with the waiver taken ---
            _move(page, POSTURE_TARGET, "asked")
            expect(page.locator(MODAL_TITLE)).to_have_text(
                _widen_title(POSTURE_TARGET, "asked"), timeout=TIMEOUT
            )
            expect(page.locator(f"{OPEN_MODAL} .posture-modal")).to_have_attribute(
                "data-target-kind", "rehearsal"
            )
            expect(page.locator(f"{OPEN_MODAL} .posture-modal-body")).to_contain_text(
                TURN_ON_SENTENCE
            )
            skip = page.locator(f"{OPEN_MODAL} .posture-modal-skip input")
            expect(skip).to_have_count(1)
            skip.check()
            page.locator(MODAL_CONFIRM).click()
            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
            expect(popover_row(page, POSTURE_TARGET)).to_have_attribute(
                "data-state", "asked", timeout=TIMEOUT
            )

            # --- the verdict, for this login and machine and no other ---
            verdict = write_verdict_for(app, POSTURE_TARGET)
            assert verdict.allowed and verdict.position == "asked", verdict
            for other in (SWITCH_TARGET, ACTIVE_TARGET):
                verdict = write_verdict_for(app, other)
                assert not verdict.allowed and verdict.reason == "switch_off", verdict
            foreign = write_verdict_for(app, POSTURE_TARGET, login=OTHER_LOGIN)
            assert not foreign.allowed, foreign
            assert_chip_matches_verdict(page, app)

            # --- asked -> allowed: no waiver, even on the rehearsal ---
            _move(page, POSTURE_TARGET, "allowed")
            expect(page.locator(MODAL_TITLE)).to_have_text(
                _widen_title(POSTURE_TARGET, "allowed"), timeout=TIMEOUT
            )
            expect(page.locator(f"{OPEN_MODAL} .posture-modal-skip")).to_have_count(0)
            page.locator(MODAL_CANCEL).click()
            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
            assert recorded_position(POSTURE_TARGET) == "asked"

            # --- off, then asked again: the waiver holds, so nothing asks ---
            _narrow(page, POSTURE_TARGET)
            _move(page, POSTURE_TARGET, "asked")
            expect(popover_row(page, POSTURE_TARGET)).to_have_attribute(
                "data-state", "asked", timeout=TIMEOUT
            )
            expect(page.locator(OPEN_MODAL)).to_have_count(0)
            expect(page.locator(f"{POPOVER} .ctc-outcome.ctc-status")).to_have_text(
                TURN_ON_SENTENCE, timeout=TIMEOUT
            )
            assert write_verdict_for(app, POSTURE_TARGET).allowed
            assert_chip_matches_verdict(page, app)

            # --- Shift re-shows the waived dialog ---
            _narrow(page, POSTURE_TARGET)
            _move(page, POSTURE_TARGET, "asked", modifiers=["Shift"])
            expect(page.locator(MODAL_TITLE)).to_have_text(
                _widen_title(POSTURE_TARGET, "asked"), timeout=TIMEOUT
            )
            page.locator(MODAL_CANCEL).click()
            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
            assert recorded_position(POSTURE_TARGET) == "off"
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (h) the real machine: a confirm every time, and no waiver
# ---------------------------------------------------------------------------


def test_a_real_machine_widening_confirms_every_time_and_offers_no_waiver(
    tmp_path, monkeypatch, chromium_browser
):
    """Every widening of the real machine's writes asks, every single time.

    A write there moves hardware, so neither ``asked`` nor ``allowed`` may be
    muted: the dialog offers no "don't ask again", carries the hardware banner,
    and is raised again on the second asked exactly as on the first. Every
    round ends in the record and the verdict.
    """
    with chip_hub(tmp_path, monkeypatch, switch={}) as (base_url, app, _root):
        page, _session_id = settled_chip(chromium_browser, base_url)
        try:
            open_popover(page)
            expect(popover_row(page, SWITCH_TARGET)).to_have_attribute(
                "data-target-kind", "real", timeout=TIMEOUT
            )
            for round_, position in enumerate(("asked", "asked", "allowed"), start=1):
                expect(popover_row(page, SWITCH_TARGET)).to_have_attribute(
                    "data-state", "off", timeout=TIMEOUT
                )
                _move(page, SWITCH_TARGET, position)
                expect(page.locator(MODAL_TITLE)).to_have_text(
                    _widen_title(SWITCH_TARGET, position), timeout=TIMEOUT
                )
                expect(page.locator(f"{OPEN_MODAL} .posture-modal-skip")).to_have_count(0)
                expect(page.locator(f"{OPEN_MODAL} .posture-modal-live")).to_contain_text(
                    "writes move hardware"
                )
                assert recorded_position(SWITCH_TARGET) == "off", round_
                page.locator(MODAL_CONFIRM).click()
                expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
                expect(popover_row(page, SWITCH_TARGET)).to_have_attribute(
                    "data-state", position, timeout=TIMEOUT
                )
                verdict = write_verdict_for(app, SWITCH_TARGET)
                assert verdict.allowed and verdict.position == position, (round_, verdict)
                assert_chip_matches_verdict(page, app)
                _narrow(page, SWITCH_TARGET)
                assert write_verdict_for(app, SWITCH_TARGET).reason == "switch_off", round_
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (i) all off, from a terminal that follows another
# ---------------------------------------------------------------------------

#: The port the holding terminal is named on in the follower's banner.
_HOLDER_PORT = 8999


def test_all_off_from_a_follower_turns_every_machine_off(tmp_path, monkeypatch, chromium_browser):
    """A follower cannot switch the target, but it can stop its login's writes.

    The record is owned by another live terminal, so this hub follows and no
    row offers Switch. The switch is the person's, not the terminal's, and no
    follower lock refuses a stop — so "Turn all writes off" works here from
    both armed positions at once. Afterwards every machine refuses with
    ``switch_off`` and the record still names the holder.
    """
    holder = owner(control_context.OWNER_WEB_TERMINAL, os.getppid(), _HOLDER_PORT)
    switch = {**ALL_ALLOWED, POSTURE_TARGET: "asked"}
    with chip_hub(tmp_path, monkeypatch, owned_by=holder, switch=switch) as (base_url, app, _root):
        page, _session_id = settled_chip(chromium_browser, base_url, enforceable=False)
        try:
            open_popover(page)
            expect(page.locator(f"{POPOVER} .ctc-switch")).to_have_count(0)
            for target, position in switch.items():
                expect(popover_machine(page, target)).to_have_attribute(
                    "data-state", position, timeout=TIMEOUT
                )
            all_off = page.locator(f"{POPOVER} .ctc-all-off")
            expect(all_off).to_be_enabled()
            all_off.click()

            for target in KINDS:
                expect(popover_machine(page, target)).to_have_attribute(
                    "data-state", "off", timeout=TIMEOUT
                )
                assert_options(page, target, checked="off")
            expect(page.locator(f"{POPOVER} .ctc-all-off")).to_be_disabled(timeout=TIMEOUT)

            record = read_record()
            assert record is not None
            assert dict(record.switch) == {}, record.switch
            assert record.owner is not None and record.owner.pid == os.getppid(), record.owner
            for target in KINDS:
                verdict = write_verdict_for(app, target)
                assert not verdict.allowed and verdict.reason == "switch_off", verdict
            assert_chip_matches_verdict(page, app)
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (j) a stop the store could not record
# ---------------------------------------------------------------------------


def test_a_stop_the_store_cannot_record_says_not_taken_and_stays_armed(
    tmp_path, monkeypatch, chromium_browser
):
    """A stop that did not land says so, and the machine is still shown armed.

    The record is readable, so the route gets as far as the write — and the
    write fails (``set_switch`` raises, as a full disk would). The answer is
    ``503 not_taken``, and the operator must read that writes are STILL on: a
    row that painted off here would be the most dangerous lie this surface
    could tell.
    """
    with chip_hub(tmp_path, monkeypatch) as (base_url, app, _root):
        page, _session_id = settled_chip(chromium_browser, base_url)
        try:
            open_popover(page)
            expect(popover_row(page, POSTURE_TARGET)).to_have_attribute(
                "data-state", "asked", timeout=TIMEOUT
            )

            def _cannot_record(*_args: Any, **_kwargs: Any) -> Any:
                raise OSError("no space left on device")

            monkeypatch.setattr(control_context, "set_switch", _cannot_record)

            _move(page, POSTURE_TARGET, "off")
            refused = popover_row(page, POSTURE_TARGET).locator(
                '.ctc-outcome[data-status="refused"]'
            )
            expect(refused).to_contain_text(STOP_NOT_TAKEN, timeout=TIMEOUT)
            expect(popover_row(page, POSTURE_TARGET)).to_have_attribute("data-state", "asked")
            assert_options(page, POSTURE_TARGET, checked="asked")

            _record, status = control_context.read_record_status()
            assert status == "ok", status
            assert recorded_position(POSTURE_TARGET) == "asked"
            assert write_verdict_for(app, POSTURE_TARGET).allowed
            assert_chip_matches_verdict(page, app)
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (j2) a widening a stop overtook
# ---------------------------------------------------------------------------


def test_a_widening_a_stop_overtook_is_refused_and_the_next_click_lands(
    tmp_path, monkeypatch, chromium_browser
):
    """A stop between the click and the confirm wins: ``409``, then a fresh retry.

    The confirm is raised on the switch epoch the page read. While it is up,
    a stop lands from elsewhere — every machine off, which moves the epoch — so
    the widening it confirms was begun before a stop it never saw. The route
    answers ``409 stopped_since``; the dialog's question no longer holds, so it
    closes, the row says nothing was turned on, and the record arms nothing.
    The next click is made on the re-read, carries the new epoch, and lands.
    """
    with chip_hub(tmp_path, monkeypatch, switch={SWITCH_TARGET: "asked"}) as (
        base_url,
        app,
        _root,
    ):
        page, _session_id = settled_chip(chromium_browser, base_url)
        try:
            open_popover(page)
            expect(popover_row(page, POSTURE_TARGET)).to_have_attribute(
                "data-state", "off", timeout=TIMEOUT
            )
            epoch_before = read_record().switch_epoch

            _move(page, POSTURE_TARGET, "asked")
            expect(page.locator(MODAL_TITLE)).to_have_text(
                _widen_title(POSTURE_TARGET, "asked"), timeout=TIMEOUT
            )
            # The stop, out of band: the same writer every surface uses.
            control_context.set_switch("all", "off")
            control_context.invalidate_cache()
            assert read_record().switch_epoch != epoch_before
            page.locator(MODAL_CONFIRM).click()

            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
            refused = popover_row(page, POSTURE_TARGET).locator(
                '.ctc-outcome[data-status="refused"]'
            )
            expect(refused).to_contain_text(STOPPED_SINCE, timeout=TIMEOUT)
            expect(popover_row(page, POSTURE_TARGET)).to_have_attribute("data-state", "off")
            assert recorded_position(POSTURE_TARGET) == "off"
            expect(popover_row(page, SWITCH_TARGET)).to_have_attribute(
                "data-state", "off", timeout=TIMEOUT
            )

            _widen(page, POSTURE_TARGET, "asked")
            assert recorded_position(POSTURE_TARGET) == "asked"
            assert_chip_matches_verdict(page, app)
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (k) a record nobody can read
# ---------------------------------------------------------------------------


def test_an_unreadable_record_shows_the_refusal_and_never_off(
    tmp_path, monkeypatch, chromium_browser
):
    """A record that cannot be read is a refusal on every machine, never "off".

    Nobody can say where a machine's switch stands, so the verdict refuses
    every write (``record_unavailable``) and the page says THAT: no option is
    checked, both armed options are held, and ``off`` stays selectable — a
    stop is never refused. The record's directory is made read-only once the
    garbage is in place so the holder cannot repair it mid-assertion.
    """
    with chip_hub(tmp_path, monkeypatch) as (base_url, app, root):
        page, _session_id = settled_chip(chromium_browser, base_url)
        record_path = control_context._record_path_under(root)
        try:
            write_payload(record_path, "not a control context")
            record_path.parent.chmod(0o500)
            control_context.invalidate_cache()
            assert control_context.read_record_status()[1] == "unreadable"

            open_popover(page)
            for target in KINDS:
                node = popover_machine(page, target)
                expect(node).to_have_attribute("data-state", "locked", timeout=TIMEOUT)
                verdict = write_verdict_for(app, target)
                assert verdict.reason == "record_unavailable", verdict
                phrase = refusal_phrase(verdict.reason, verdict.kind)
                expect(node.locator(".ctc-switch-state")).to_have_text("locked")
                expect(node.locator(".ctc-write-phrase")).to_have_text(phrase)
                assert_options(page, target, checked=None, disabled=("asked", "allowed"))
                for position in ("asked", "allowed"):
                    title = writes_option(page, target, position).get_attribute("title")
                    assert title in (phrase, "the write switch cannot be read"), title
            expect(page.locator(CHIP_STATE)).to_have_text("writes locked", timeout=TIMEOUT)
            expect(page.locator(f"{POPOVER} [data-state='off']")).to_have_count(0)
            assert_chip_matches_verdict(page, app)
            control_context.invalidate_cache()
            assert control_context.read_record_status()[1] == "unreadable"
        finally:
            record_path.parent.chmod(0o700)
            page.close()


# ---------------------------------------------------------------------------
# (l) a readonly login, and an app render: nothing armed is selectable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lock", ["readonly_login", "app_render"])
def test_a_readonly_login_or_an_app_render_offers_nothing_armed(
    tmp_path, monkeypatch, chromium_browser, lock
):
    """No chip gesture arms a machine where the tier never writes.

    Two ways to arrive at ``tier_never``: a login the render maps to ``never``,
    and an app render whose tier never writes for anyone. Every switch is off
    and must stay off, so on every machine both ``asked`` and ``allowed`` are
    disabled with the verdict's own sentence — which names who can lift the
    lock — on their hover and on the row. The route refuses both positions
    too, with the same phrase.
    """
    login = acting_identity()
    hub_args: dict[str, Any] = (
        {"login_tiers": {login: "never"}} if lock == "readonly_login" else {"writes": "never"}
    )
    with chip_hub(tmp_path, monkeypatch, switch={}, **hub_args) as (base_url, app, _root):
        page, _session_id = settled_chip(chromium_browser, base_url)
        try:
            open_popover(page)
            for target in KINDS:
                verdict = write_verdict_for(app, target)
                assert verdict.reason == "tier_never", (target, verdict)
                phrase = refusal_phrase(verdict.reason, verdict.kind)
                node = popover_machine(page, target)
                expect(node).to_have_attribute("data-state", "off", timeout=TIMEOUT)
                assert_options(page, target, checked="off", disabled=("asked", "allowed"))
                for position in ("asked", "allowed"):
                    expect(writes_option(page, target, position)).to_have_attribute("title", phrase)
                expect(node.locator(".ctc-write-phrase")).to_have_text(phrase)

                for position in ("asked", "allowed"):
                    answer = requests.post(
                        f"{base_url}{POSTURE_ROUTE}",
                        json={
                            "session_id": None,
                            "target": target,
                            "state": position,
                            "epoch": None,
                        },
                        timeout=60,
                    )
                    assert answer.status_code == 403, (position, answer.text)
                    detail = answer.json()["detail"]
                    assert detail["reason"] == "tier_never", (position, detail)
                    assert detail["phrase"] == phrase, (position, detail)

            assert read_record().switch == {}, read_record().switch
            assert_chip_matches_verdict(page, app)
        finally:
            page.close()
