"""Browser tests: the control-target header chip, end to end.

The chip in the page header — ``● Simulator · writes on ▾`` — and the popover
behind it are the operator's only route between writes on and writes off, and
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

Coverage (one test each):

  (a) the chip names the machine the deployment stands on by its display name,
      and the popover renders that machine as the card and every other
      configured target as a row, the server's own label demoted to tooltips.
  (b) turning a target's writes off asks nothing, lands in the login's switch
      where every write path reads it, and respawns nothing — the switch is
      read live, so the PTY the operator is talking to is the same process.
  (c) turning writes back on is the direction that confirms: the dialog names
      the machine, Cancel is a true no-op on the row *and* in the record, and
      only a confirmed dialog turns the switch on.
  (c2) turning off moves the endpoint the row's tooltip names off the write
      gateway and onto the read one, and the confirm that would turn writes
      on again names no endpoint at all.
  (d) Switch confirms, is accepted as ``202``, moves the record's target and
      generation — the terminal runs the gate itself now — and holds the chip
      and the row in ``switching…`` for exactly as long as a live controls
      server is still reporting the generation before it.
  (e) a refusal raised from a confirm stays inside the dialog, which stays up
      to carry it: a deployment already mid-swap answers ``409`` and the chip
      must not fall into ``switching…`` for a switch nobody made.
  (f) both UI modes render the SAME popover DOM and show all of it: the
      redesign leaves no popover node for the density stylesheet to gate —
      endpoints and the server's label are hover vocabulary in both modes.

The chip criteria, end to end — each one closed against the write verdict
itself (``posture_store.write_verdict``, the rule every write path applies), and
each one asserting that every machine's ``data-state`` and the chip's own equal
what that verdict and the record say (:func:`assert_chip_matches_verdict`):

  (g) a rehearsal turn-on confirms once — the waiver is offered and taken —
      and applies directly after; the verdict then allows this login on that
      machine and refuses the other machines and any other login.
  (h) a real-machine turn-on confirms every time and never offers a waiver.
  (i) "Turn all writes off" works from a terminal that follows another, and
      afterwards every machine refuses with ``switch_off``.
  (j) a stop the store cannot record, on a readable record, says it was not
      taken and the machine stays on.
  (k) an unreadable record shows the verdict's refusal on every machine and
      never shows one off.
  (l) a readonly login and an app render cannot turn writes on: every toggle
      is disabled and the verdict's sentence — who can lift it — is on the row.

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

The record is seeded with this login's switch ON for every machine unless a
test asks otherwise (``chip_hub(switch=...)``), so the first gesture a test
can make is the one that asks nothing.

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
    POSTURE_TARGET,
    READ_ENDPOINT,
    RECORD_GENERATION,
    SWITCH_TARGET,
    TIMEOUT,
    WRITE_ENDPOINT,
    assert_chip_matches_verdict,
    chip_hub,
    open_popover,
    popover_machine,
    popover_row,
    publish_report,
    recorded_switch,
    settled_chip,
    write_verdict_for,
    writes_toggle,
)

# without playwright, the chromium_browser fixture skips the suite
with suppress(ImportError):
    from playwright.sync_api import Page, expect

pytestmark = [pytest.mark.browser, pytest.mark.slow]


#: The sentence a turn-on carries — in its confirm, or once on the popover's
#: status line when the confirm was waived (control-target-facts.js).
TURN_ON_SENTENCE = (
    "Runs already going — a sandbox run or a notebook cell — keep the state they "
    "started with; the next one follows."
)

#: What a stop the store could not record says (C-POST 503 on off).
STOP_NOT_TAKEN = "The stop did not take effect — writes are still on."


def _pty_pid(app: Any, session_id: str) -> int:
    """The pid of the PTY this session runs in."""
    session = app.state.pty_registry.get_session(session_id)
    assert session is not None, f"no PTY session registered for {session_id}"
    pid = session.pid
    assert isinstance(pid, int) and pid > 0, f"the PTY reported no usable pid: {pid!r}"
    return pid


def _turn_off(page: Page, target: str) -> None:
    """Turn *target*'s writes off through the popover, and wait for the row.

    Applies on click — turning off only ever removes reach, so no confirm is
    in the way, and the row's ``data-state`` flipping is the re-read landing.
    """
    writes_toggle(page, target).click()
    expect(popover_row(page, target)).to_have_attribute("data-state", "off", timeout=TIMEOUT)


# ---------------------------------------------------------------------------
# (a) the chip and its roster
# ---------------------------------------------------------------------------


def test_the_chip_names_the_recorded_target_and_lists_every_row(
    tmp_path, monkeypatch, chromium_browser
):
    """The chip speaks for the machine the deployment stands on; the popover lists all.

    The record puts the deployment on the simulator, so the chip reads
    ``Simulator · writes on`` — the display name the render minted, not the
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
            expect(page.locator(CHIP_STATE)).to_have_text("writes on", timeout=TIMEOUT)
            expect(page.locator(CHIP)).to_have_attribute("data-target-kind", KINDS[ACTIVE_TARGET])
            expect(page.locator(CHIP)).to_have_attribute("aria-expanded", "false")

            open_popover(page)

            # The machine the agent is on is the card; every other machine a row.
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
                expect(row.locator(".ctc-switch-state")).to_have_text("on")
                # The server's own label lives inside the ⓘ tooltip, not at rest.
                tip = row.locator(".ctc-tip").text_content()
                assert tip and LABELS[target] in tip, tip

            expect(card.locator(".ctc-switch")).to_have_count(0)
            # The one target this render will switch onto.
            expect(popover_row(page, SWITCH_TARGET).locator(".ctc-switch")).to_be_visible()
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (b) turning writes off
# ---------------------------------------------------------------------------


def test_turning_off_asks_nothing_lands_in_the_record_and_respawns_nothing(
    tmp_path, monkeypatch, chromium_browser
):
    """Writes off applies on click, is written where the connector reads, and is free.

    Three separate claims, and each is load-bearing:

    * no confirm — turning off only ever removes reach, so asking would be
      ceremony over a gesture one click undoes;
    * the record agrees, because the login's ``switch`` in that file is what
      the write verdict consults on every path — a row that agreed with
      nothing would be a machine the operator believes is off and the agent
      can still write to. The switch is keyed by TARGET and by nothing else:
      one per login, not one per session;
    * the PTY is the same process afterwards (FR17). The posture is read live
      on every write, so a respawn would cost the operator their session for
      nothing.
    """
    with chip_hub(tmp_path, monkeypatch) as (base_url, app, _root):
        page, session_id = settled_chip(chromium_browser, base_url)
        assert session_id, "the terminal card never settled on a session id"
        pty_pid = _pty_pid(app, session_id)
        try:
            open_popover(page)
            row = popover_row(page, POSTURE_TARGET)
            expect(row).to_have_attribute("data-state", "on")
            expect(writes_toggle(page, POSTURE_TARGET)).to_have_attribute("aria-checked", "true")

            _turn_off(page, POSTURE_TARGET)

            # Nothing asked, and the switch is the readout as well as the
            # control: it flipped to off.
            expect(page.locator(OPEN_MODAL)).to_have_count(0)
            expect(popover_row(page, POSTURE_TARGET).locator(".ctc-switch-state")).to_have_text(
                "off"
            )
            expect(writes_toggle(page, POSTURE_TARGET)).to_have_attribute("aria-checked", "false")
            # The card is untouched: a posture is per target.
            expect(page.locator(CARD)).to_have_attribute("data-state", "on")

            assert POSTURE_TARGET not in recorded_switch(), recorded_switch()
            assert write_verdict_for(app, POSTURE_TARGET).reason == "switch_off"
            assert_chip_matches_verdict(page, app)

            assert _pty_pid(app, session_id) == pty_pid, "the session was respawned"
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (c) turning writes on
# ---------------------------------------------------------------------------


def test_turning_on_confirms_and_cancel_changes_nothing(tmp_path, monkeypatch, chromium_browser):
    """Only turning on asks — and Cancel leaves the row and the record as they were.

    Turning on is the gesture after which a write the agent makes can land, so it
    is the one direction that confirms. The dialog names the target it is
    about; the popover deliberately stays open beneath it, so the row the
    question is about is still on screen. A toggle that fired on the way to the
    dialog would be the worst possible failure of a confirm step, which is why
    the record is asserted on both sides of the cancellation.
    """
    with chip_hub(tmp_path, monkeypatch) as (base_url, _app, _root):
        page, _session_id = settled_chip(chromium_browser, base_url)
        try:
            open_popover(page)
            _turn_off(page, POSTURE_TARGET)

            # --- turning on, cancelled ---
            writes_toggle(page, POSTURE_TARGET).click()
            expect(page.locator(MODAL_TITLE)).to_have_text(
                f"Turn writes on for {NAMES[POSTURE_TARGET]}?", timeout=TIMEOUT
            )
            # The rows stay readable underneath: the confirm is a layer above
            # the popover, not a replacement for it.
            expect(page.locator(POPOVER)).to_be_visible()
            page.locator(MODAL_CANCEL).click()

            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
            expect(popover_row(page, POSTURE_TARGET)).to_have_attribute("data-state", "off")
            assert POSTURE_TARGET not in recorded_switch(), recorded_switch()

            # --- turning on, confirmed ---
            writes_toggle(page, POSTURE_TARGET).click()
            expect(page.locator(MODAL_CONFIRM)).to_have_text("Turn writes on", timeout=TIMEOUT)
            page.locator(MODAL_CONFIRM).click()

            expect(popover_row(page, POSTURE_TARGET)).to_have_attribute(
                "data-state", "on", timeout=TIMEOUT
            )
            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
            assert recorded_switch().get(POSTURE_TARGET) == "asked", recorded_switch()
        finally:
            page.close()


def test_turning_off_moves_the_tooltip_onto_the_read_gateway(
    tmp_path, monkeypatch, chromium_browser
):
    """A row with writes off names the gateway it is actually talking to.

    The endpoint on a row is where control *reads* go under the recorded
    posture. Armed, that is the write gateway; give the writes up and it is the
    read gateway — and the tooltip an operator hovers has to say so, because the
    render answering from the deployment ceiling would keep naming a gateway
    the deployment can no longer reach.

    The narrowing is made the way an operator makes it, through the popover's
    own toggle and therefore through the real POST: a record written from the
    test instead would prove the renderer reads a file, not that the round trip
    an operator drives ends on screen. The row is re-read from the page after
    the gesture — the popover subtree is replaced wholesale on every refetch.

    The confirm that would widen again is checked for the opposite: it names no
    endpoint whatsoever. The one the roster carries describes reads under the
    posture the dialog is about to leave, so quoting it in the question would
    name the wrong gateway at the one moment an operator is deciding about
    writes.
    """
    with chip_hub(tmp_path, monkeypatch) as (base_url, _app, _root):
        page, _session_id = settled_chip(chromium_browser, base_url)
        try:
            open_popover(page)

            # Armed: the write gateway, and only it.
            ident = popover_row(page, ENDPOINT_TARGET).locator(".ctc-tip-ident")
            expect(ident).to_contain_text(WRITE_ENDPOINT, timeout=TIMEOUT)
            assert READ_ENDPOINT not in (ident.text_content() or "")

            _turn_off(page, ENDPOINT_TARGET)

            # Narrowed: the read gateway. Re-resolved from the page, since the
            # refetch that carried the new posture replaced the node above.
            ident = popover_row(page, ENDPOINT_TARGET).locator(".ctc-tip-ident")
            expect(ident).to_contain_text(READ_ENDPOINT, timeout=TIMEOUT)
            tip = ident.text_content() or ""
            assert WRITE_ENDPOINT not in tip
            # Identity is derived from the ceiling and does not move with the
            # posture: the operator is standing in front of the same machine.
            assert LABELS[ENDPOINT_TARGET] in tip

            # The confirm that widens again names no gateway at all.
            writes_toggle(page, ENDPOINT_TARGET).click()
            expect(page.locator(MODAL_TITLE)).to_have_text(
                f"Turn writes on for {NAMES[ENDPOINT_TARGET]}?", timeout=TIMEOUT
            )
            body = page.locator(f"{OPEN_MODAL} .posture-modal-body").text_content() or ""
            assert body.strip(), "the confirm rendered an empty body"
            assert not HOST_PORT.search(body), body
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (d) switching
# ---------------------------------------------------------------------------


def test_switch_confirms_moves_the_record_and_waits_for_the_fleet(
    tmp_path, monkeypatch, chromium_browser
):
    """Switch asks, moves the record, and the chip waits out loud for the fleet.

    The terminal runs the switch gate itself now: an accepted gesture writes
    the new target, the next generation and an ``applied`` terminus into the
    record in one mutation — there is no request file and no reconciler in the
    middle. So the record is the first assertion.

    Landing in the record is not landing on the machine, though, and that gap
    is what ``switching…`` names. A live controls server is planted here still
    reporting the generation BEFORE the switch, which is a server that has not
    rebuilt its connector yet; while any live row lags, the chip and the card
    both say so. Without that planted report the fleet would be empty, the
    record's own terminus would answer for it, and the chip would settle
    immediately — a different (and also correct) story, told by the unit tests.

    The chip names the machine the RECORD is on from the moment it moves. That
    is deliberate: ``data-state`` still describes the posture there, and
    ``switching…`` is the whole of what has not finished.
    """
    with chip_hub(tmp_path, monkeypatch) as (base_url, _app, root):
        # A server that is up and reporting the pre-switch generation. It is
        # not blocking anything (no `applying` block), so the switch is
        # allowed; it simply has not caught up once the record moves.
        publish_report(root)
        page, _session_id = settled_chip(chromium_browser, base_url)
        try:
            open_popover(page)
            popover_row(page, SWITCH_TARGET).locator(".ctc-switch").click()

            expect(page.locator(MODAL_TITLE)).to_have_text(
                f"Switch to {NAMES[SWITCH_TARGET]}?", timeout=TIMEOUT
            )
            # The write state named is the one held on the machine being
            # switched TO, because writes on/off is per machine and does not
            # follow — and only the real machine carries the hardware sentence.
            expect(page.locator(f"{OPEN_MODAL} .posture-modal-body")).to_contain_text(
                "Writes are on there"
            )
            expect(page.locator(f"{OPEN_MODAL} .posture-modal-live")).to_have_text(
                "Real machine — writes move hardware"
            )
            page.locator(MODAL_CONFIRM).click()

            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
            expect(page.locator(CHIP)).to_have_attribute("data-pending", "true", timeout=TIMEOUT)
            expect(page.locator(CHIP_STATE)).to_have_text("switching…", timeout=TIMEOUT)
            # The card is the machine the record now names, and it carries the
            # wait — the target being switched to stops being a row the moment
            # the record moves.
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

    One swap crosses the deployment at a time, and one is planted here before
    the operator clicks: a live controls server reporting ``applying`` for the
    record's current generation is a server between two targets, and while it
    is there nobody switches anywhere. The route answers 409 naming its pid,
    and the dialog is where the operator is looking, so the sentence goes there
    and the dialog is kept up to carry it — dismissing it to put the reason on
    a row behind would hide the answer to the question they had just been
    asked. Nothing was written, so the chip must not fall into ``switching…``
    and the record must still name the target it started on.
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

    ``html[data-ui-mode]`` stays a CSS concern and only a CSS concern — and
    this design leaves it nothing in the popover to gate. The endpoint and the
    server's own label are hover vocabulary in BOTH modes, reachability only
    speaks when a machine is not answering, and what remains at rest — the
    name, its ⓘ, the writes switch, Switch — is what an
    operator acts on and is shown in either density. So the invariant pinned
    here is the stronger one: same DOM, same visibility, and the confirms
    identical, whichever mode the deployment renders.

    The mode is driven the way the deployment drives it — ``web.ui_mode``
    reaching the page as the server-rendered ``<html data-ui-mode>`` attribute
    — not by poking the attribute from the test.
    """
    with chip_hub(tmp_path, monkeypatch, ui_mode=ui_mode) as (base_url, _app, _root):
        page, _session_id = settled_chip(chromium_browser, base_url)
        try:
            expect(page.locator("html")).to_have_attribute("data-ui-mode", ui_mode)
            open_popover(page)

            row = popover_row(page, SWITCH_TARGET)

            # --- what an operator acts on, at rest, in either density ---
            expect(row.locator(".ctc-name")).to_have_text(NAMES[SWITCH_TARGET])
            assert row.locator(".ctc-tip-what").text_content() == "Writes move hardware"
            expect(row.locator(".ctc-switch-state")).to_have_text("on")
            expect(row.locator(".ctc-toggle")).to_be_visible()
            expect(row.locator(".ctc-switch")).to_be_visible()
            expect(page.locator(FOOT_NOTE)).to_have_text("Applies to every session of this login")

            # --- the machine vocabulary stays behind the ⓘ, in either density ---
            tip = row.locator(".ctc-tip").text_content()
            assert tip and LABELS[SWITCH_TARGET] in tip, tip
            # No endpoint/role line exists at rest for a stylesheet to gate.
            assert row.locator(".ctc-meta").count() == 0

            # Both confirms are identical in either mode: a safety gesture does
            # not get a density.
            row.locator(".ctc-switch").click()
            expect(page.locator(MODAL_TITLE)).to_have_text(
                f"Switch to {NAMES[SWITCH_TARGET]}?", timeout=TIMEOUT
            )
            page.locator(MODAL_CANCEL).click()
            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (g) the rehearsal: one confirm, then the verdict for this login and machine
# ---------------------------------------------------------------------------

#: A login that is not the one signed in. With no per-login tree bound, the
#: verdict refuses any owner but this host's own (``owner_foreign``): a switch is
#: one person's and is never lent.
OTHER_LOGIN = "someone-else"


def test_a_rehearsal_turn_on_confirms_once_and_arms_only_that_login_and_machine(
    tmp_path, monkeypatch, chromium_browser
):
    """The rehearsal asks once; its turn-on reaches this login on this machine only.

    A live controls server publishes the stand-in as the rehearsal, and every
    switch starts off. The first turn-on confirms — the dialog carries the
    rehearsal's kind and offers the waiver, which the operator takes. From then
    on a turn-on of that machine applies directly, and the sentence the skipped
    confirm would have carried is said once on the popover's status line.

    What it unlocked is read from the verdict itself, the one rule every write
    path applies: a write by this login to the rehearsal passes; the same login
    on the other two machines meets ``switch_off``; and another login on the
    rehearsal is refused, because the switch is this login's alone.
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
            writes_toggle(page, POSTURE_TARGET).click()
            expect(page.locator(MODAL_TITLE)).to_have_text(
                f"Turn writes on for {NAMES[POSTURE_TARGET]}?", timeout=TIMEOUT
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
                "data-state", "on", timeout=TIMEOUT
            )

            # --- the verdict, for this login and machine and no other ---
            assert write_verdict_for(app, POSTURE_TARGET).allowed, write_verdict_for(
                app, POSTURE_TARGET
            )
            for other in (SWITCH_TARGET, ACTIVE_TARGET):
                verdict = write_verdict_for(app, other)
                assert not verdict.allowed and verdict.reason == "switch_off", verdict
            foreign = write_verdict_for(app, POSTURE_TARGET, login=OTHER_LOGIN)
            assert not foreign.allowed, foreign
            assert_chip_matches_verdict(page, app)

            # --- off, then on again: the waiver holds, so nothing asks ---
            _turn_off(page, POSTURE_TARGET)
            writes_toggle(page, POSTURE_TARGET).click()
            expect(popover_row(page, POSTURE_TARGET)).to_have_attribute(
                "data-state", "on", timeout=TIMEOUT
            )
            expect(page.locator(OPEN_MODAL)).to_have_count(0)
            expect(page.locator(f"{POPOVER} .ctc-outcome.ctc-status")).to_have_text(
                TURN_ON_SENTENCE, timeout=TIMEOUT
            )
            assert write_verdict_for(app, POSTURE_TARGET).allowed
            assert_chip_matches_verdict(page, app)
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (h) the real machine: a confirm every time, and no waiver
# ---------------------------------------------------------------------------


def test_a_real_machine_turn_on_confirms_every_time_and_offers_no_waiver(
    tmp_path, monkeypatch, chromium_browser
):
    """Turning the real machine's writes on asks every single time.

    It is the one confirm that must not be muted: a write there moves hardware.
    So the dialog offers no "don't ask again", carries the hardware banner, and
    is raised again on the second turn-on exactly as on the first. Both rounds
    end in the record and the verdict, so a dialog that stopped asking because
    the gesture had quietly been applied would fail here too.
    """
    with chip_hub(tmp_path, monkeypatch, switch={}) as (base_url, app, _root):
        page, _session_id = settled_chip(chromium_browser, base_url)
        try:
            open_popover(page)
            expect(popover_row(page, SWITCH_TARGET)).to_have_attribute(
                "data-target-kind", "real", timeout=TIMEOUT
            )
            for round_ in (1, 2):
                expect(popover_row(page, SWITCH_TARGET)).to_have_attribute(
                    "data-state", "off", timeout=TIMEOUT
                )
                writes_toggle(page, SWITCH_TARGET).click()
                expect(page.locator(MODAL_TITLE)).to_have_text(
                    f"Turn writes on for {NAMES[SWITCH_TARGET]}?", timeout=TIMEOUT
                )
                expect(page.locator(f"{OPEN_MODAL} .posture-modal-skip")).to_have_count(0)
                expect(page.locator(f"{OPEN_MODAL} .posture-modal-live")).to_contain_text(
                    "writes move hardware"
                )
                # Nothing is applied while the question is up.
                assert SWITCH_TARGET not in recorded_switch(), (round_, recorded_switch())
                page.locator(MODAL_CONFIRM).click()
                expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
                expect(popover_row(page, SWITCH_TARGET)).to_have_attribute(
                    "data-state", "on", timeout=TIMEOUT
                )
                assert write_verdict_for(app, SWITCH_TARGET).allowed, round_
                assert_chip_matches_verdict(page, app)
                _turn_off(page, SWITCH_TARGET)
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

    The record is owned by another live terminal (the parent process stands in
    for it), so this hub follows: ``data-enforceable`` is false and no row
    offers Switch. The switch, though, is the person's and not the terminal's,
    and no follower lock refuses a stop — so "Turn all writes off" works here
    exactly as it does on the holder. Afterwards every machine refuses with
    ``switch_off``, the record still names the holder, and the chip agrees.
    """
    holder = owner(control_context.OWNER_WEB_TERMINAL, os.getppid(), _HOLDER_PORT)
    with chip_hub(tmp_path, monkeypatch, owned_by=holder) as (base_url, app, _root):
        page, _session_id = settled_chip(chromium_browser, base_url, enforceable=False)
        try:
            open_popover(page)
            expect(page.locator(f"{POPOVER} .ctc-switch")).to_have_count(0)
            for target in KINDS:
                expect(popover_machine(page, target)).to_have_attribute(
                    "data-state", "on", timeout=TIMEOUT
                )
            all_off = page.locator(f"{POPOVER} .ctc-all-off")
            expect(all_off).to_be_enabled()
            all_off.click()

            for target in KINDS:
                expect(popover_machine(page, target)).to_have_attribute(
                    "data-state", "off", timeout=TIMEOUT
                )
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


def test_a_stop_the_store_cannot_record_says_not_taken_and_stays_on(
    tmp_path, monkeypatch, chromium_browser
):
    """A stop that did not land says so, and the machine is still shown on.

    The record is readable, so the route gets as far as the write — and the
    write fails (``set_switch`` raises, as a full disk or a lost lock would).
    The answer is ``503 not_taken``, and the operator must read that writes are
    STILL on: a row that painted off here would be the most dangerous lie this
    surface could tell. So the row carries the stop sentence, its switch stays
    on, and the record and the verdict both still allow the write.
    """
    with chip_hub(tmp_path, monkeypatch) as (base_url, app, _root):
        page, _session_id = settled_chip(chromium_browser, base_url)
        try:
            open_popover(page)
            expect(popover_row(page, POSTURE_TARGET)).to_have_attribute(
                "data-state", "on", timeout=TIMEOUT
            )

            def _cannot_record(*_args: Any, **_kwargs: Any) -> Any:
                raise OSError("no space left on device")

            monkeypatch.setattr(control_context, "set_switch", _cannot_record)

            writes_toggle(page, POSTURE_TARGET).click()
            refused = popover_row(page, POSTURE_TARGET).locator(
                '.ctc-outcome[data-status="refused"]'
            )
            expect(refused).to_contain_text(STOP_NOT_TAKEN, timeout=TIMEOUT)
            expect(popover_row(page, POSTURE_TARGET)).to_have_attribute("data-state", "on")
            expect(writes_toggle(page, POSTURE_TARGET)).to_have_attribute("aria-checked", "true")

            _record, status = control_context.read_record_status()
            assert status == "ok", status
            assert recorded_switch().get(POSTURE_TARGET) == "asked", recorded_switch()
            assert write_verdict_for(app, POSTURE_TARGET).allowed
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

    Every switch was on. Then the file stops parsing. Nobody can say whether a
    machine is on, so the verdict refuses every write (``record_unavailable``),
    and the page has to say THAT — the verdict's own sentence, with the switch
    shown as one that may be on, so its stop control stays live. Painting the
    machines off would tell the operator writes are stopped when nothing has
    recorded a stop.

    The record's directory is made read-only once the garbage is in place. A
    holder that CAN write replaces an unparseable record with a fresh one on
    its next owner tick — every switch off, a record that is then readable and
    true — so without this the unreadable state would last under a second and
    the page would be asserted against whichever of the two it caught. A store
    that can neither be read nor repaired is the state this lane is about.
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
                expect(node.locator(".ctc-switch-state")).to_have_text("locked")
                expect(node.locator(".ctc-write-phrase")).to_have_text(
                    refusal_phrase(verdict.reason, verdict.kind)
                )
                expect(node.locator(".ctc-toggle")).to_be_enabled()
            expect(page.locator(CHIP_STATE)).to_have_text("writes locked", timeout=TIMEOUT)
            expect(page.locator(f"{POPOVER} [data-state='off']")).to_have_count(0)
            assert_chip_matches_verdict(page, app)
            # Still unreadable at the end, so every assertion above was taken
            # against the state this lane names and not a repaired record.
            control_context.invalidate_cache()
            assert control_context.read_record_status()[1] == "unreadable"
        finally:
            record_path.parent.chmod(0o700)
            page.close()


# ---------------------------------------------------------------------------
# (l) a readonly login, and an app render: no turn-on, and who can lift it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lock", ["readonly_login", "app_render"])
def test_a_readonly_login_or_an_app_render_cannot_turn_on(
    tmp_path, monkeypatch, chromium_browser, lock
):
    """No chip gesture turns writes on where the tier never writes.

    Two ways to arrive at ``tier_never``: a login the render maps to ``never``
    on an otherwise writable deployment, and an app render whose tier never
    writes for anyone. Either way every switch is off and must stay off, so
    each machine's toggle is disabled and the verdict's own sentence — which
    names who can lift the lock — stands on the row and on the toggle's hover.
    The route behind it refuses the turn-on too, with the same phrase: a
    disabled button is a courtesy, the 403 is the rule.
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
                toggle = node.locator(".ctc-toggle")
                expect(toggle).to_be_disabled()
                expect(toggle).to_have_attribute("aria-checked", "false")
                expect(toggle).to_have_attribute("title", phrase)
                expect(node.locator(".ctc-write-phrase")).to_have_text(phrase)

                answer = requests.post(
                    f"{base_url}/api/terminal/posture",
                    json={"session_id": None, "target": target, "state": "on", "epoch": None},
                    timeout=60,
                )
                assert answer.status_code == 403, answer.text
                detail = answer.json()["detail"]
                assert detail["reason"] == "tier_never", detail
                assert detail["phrase"] == phrase, detail

            assert recorded_switch() == {}, recorded_switch()
            assert_chip_matches_verdict(page, app)
        finally:
            page.close()
