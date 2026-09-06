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
  (b) turning a target's writes off asks nothing, lands in the record where
      the connector reads it, and respawns nothing — the posture is read live,
      so the PTY the operator is talking to is the same process afterwards.
  (c) turning writes back on is the direction that confirms: the dialog names
      the machine, Cancel is a true no-op on the row *and* in the record, and
      only a confirmed dialog widens.
  (c2) a narrowing moves the endpoint the row's tooltip names off the write
      gateway and onto the read one, and the confirm that would widen again
      names no endpoint at all.
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

One fact has to be true before any toggle in the popover can move, and
:func:`_chip_hub` arranges it: **a control-context record exists under the
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
import re
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import yaml

from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey_connectors import control_context, posture_store
from tests._control_context_fixtures import write_control_context, write_server_report
from tests.interfaces._panel_launch import publish_artifact_url
from tests.interfaces.conftest import _apply_all, _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

try:
    from playwright.sync_api import Browser, Page, expect
except ImportError:  # pragma: no cover — the chromium_browser fixture skips the suite
    pass

pytestmark = [pytest.mark.browser, pytest.mark.slow]


# Every wait in this file uses one generous bound rather than a tuned-per-step
# one. These suites run under parallel load where a step that normally takes
# milliseconds can take seconds, and a tight bound buys nothing: a state that
# is never going to arrive fails the assertion either way, only later. It also
# has to cover one full idle poll of the chip (5 s), which is how the record
# published mid-test reaches the page.
TIMEOUT = 15_000

CHIP = "#control-target-chip"
CHIP_SHORT = f"{CHIP} .ctc-short"
CHIP_STATE = f"{CHIP} .ctc-state"
POPOVER = ".ctc-popover.open"
CARD = f"{POPOVER} .ctc-card"
FOOT_NOTE = f"{POPOVER} .ctc-foot-note"

# `:not([data-closing])` is the contract for "the dialog is OPEN". Dismissal is
# marked by the attribute and the node is only detached after the fade (~300ms),
# so asserting on detachment would be asserting on an animation.
OPEN_MODAL = ".posture-modal-overlay:not([data-closing])"
MODAL_TITLE = f"{OPEN_MODAL} .posture-modal-title"
MODAL_CONFIRM = f"{OPEN_MODAL} .posture-modal-confirm"
MODAL_CANCEL = f"{OPEN_MODAL} .posture-modal-cancel"
MODAL_ERROR = f"{OPEN_MODAL} .posture-modal-error"

# A PTY command that outlives the test AND tolerates the arguments the
# websocket route appends (``--session-id <uuid>`` on a new session,
# ``--resume <uuid>`` on a reconnect). ``echo``/``sleep`` would exit or error on
# those, and an exit inside the resume-failover window makes terminal.js
# discard the session id the chip is pointed at.
_LONG_LIVED_SHELL = [sys.executable, "-c", "import time; time.sleep(3600)"]

#: The Channel Access port the co-deployed stand-in serves on.
STANDIN_PORT = 5074

#: The facility machine's two gateway ports, deliberately distinct. Which of
#: them a row names is the whole observable of the posture-aware render: under
#: writes the row is talking to :data:`WRITE_PORT`, and a session that has given
#: up writes is talking to :data:`READ_PORT`.
READ_PORT = 5064
WRITE_PORT = 5065

#: The endpoints those two ports spell on the row, on the host both roles share.
READ_ENDPOINT = f"gw:{READ_PORT}"
WRITE_ENDPOINT = f"gw:{WRITE_PORT}"

#: Anything shaped like a gateway address. Used to assert an ABSENCE: the
#: turn-writes-on confirm names no endpoint at all, because the one the roster
#: carries is where reads go under the current posture and not where the write
#: this dialog allows would land.
_HOST_PORT = re.compile(r"\b\S+:\d{2,5}\b")

#: The target the record puts the deployment on. Chosen so the roster carries
#: all three interesting rows at once: ``va`` is active (no Switch), ``live``
#: is switchable, and ``standin`` is a real machine that is not.
ACTIVE_TARGET = "va"

#: The generation the record is seeded at. Deliberately not 0, so a test that
#: asserts a switch moved it can tell the new value from "never set".
RECORD_GENERATION = 1

#: The row every posture gesture below is made on. Deliberately not the active
#: one: narrowing the target the deployment stands on also raises the "applies
#: after the running execution finishes" line when a realign is pending, which
#: is a different contract from the one these tests pin.
POSTURE_TARGET = "standin"

#: The row Switch is exercised on — the only one this render offers it for.
SWITCH_TARGET = "live"

#: The row the endpoint case is made on. The facility's own machine is the only
#: target here whose two gateway roles sit on different ports, so it is the only
#: row whose named endpoint can be seen to move when a session gives up writes.
ENDPOINT_TARGET = "live"

#: The server's own labels. The controls server mints these once, and this
#: render keeps them where the machine vocabulary now lives: on tooltips.
LABELS = {
    "live": "LIVE MACHINE",
    "va": "virtual accelerator (simulation)",
    "standin": "LIVE MACHINE (stand-in)",
}

#: What the popover and the chip NAME each target: the kind's own word, since
#: this render configures no ``control_system.target_display_names``. The
#: confirm titles quote these — never the labels above.
NAMES = {
    "live": "Real machine",
    "va": "Simulator",
    "standin": "Rehearsal",
}


# ---------------------------------------------------------------------------
# The deployment under test
# ---------------------------------------------------------------------------


def _write_config(path: Path, *, writes_enabled: bool = True) -> Path:
    """A render carrying all three control targets, with writes armed.

    Mirrors the render ``test_posture_get_contract.py`` pins the route's own
    answers against: ``epics`` is the facility's own machine, ``live_standin``
    the co-deployed stand-in, ``virtual_accelerator`` the simulator — three
    connector blocks, therefore three targets and three rows.

    The keys beyond the gateways are what make a switch judgeable at all: a
    channel to probe, strict limits (required toward the live family) and the
    operator's acknowledgement of the live gateway. Without them every row
    would carry an eligibility refusal and no Switch would be offered anywhere.

    The ``epics`` block is the one target here whose two roles sit on DIFFERENT
    ports, which is what makes a narrowing observable at all: a block whose
    ``read_only`` and ``write_access`` name one gateway renders the same
    endpoint under either posture, so a test asserting the endpoint moved would
    pass on a render that never consulted the posture. The host stays ``gw`` in
    both roles — a loopback address is the stand-in predicate's own signal, and
    moving the write role off it would rename the machine mid-test.
    """
    gateway = {"address": "gw", "port": READ_PORT, "use_name_server": True}
    write_gateway = {"address": "gw", "port": WRITE_PORT, "use_name_server": True}
    standin_gateway = {"address": "localhost", "port": STANDIN_PORT, "use_name_server": True}
    path.write_text(
        yaml.safe_dump(
            {
                "control_system": {
                    "type": "live_standin",
                    "writes_enabled": writes_enabled,
                    "limits_checking": {"enabled": True, "allow_unlisted_channels": False},
                    "target_switch": {"live_gateway_acknowledged": "operator@example"},
                    "connector": {
                        "epics": {
                            "probe_channel": "SR:PROBE",
                            "gateways": {
                                "read_only": dict(gateway),
                                "write_access": dict(write_gateway),
                            },
                        },
                        "live_standin": {
                            "probe_channel": "SR:PROBE",
                            "gateways": {
                                "read_only": dict(standin_gateway),
                                "write_access": dict(standin_gateway),
                            },
                        },
                        "virtual_accelerator": {
                            "simulation_file": "data/sim.json",
                            "probe_channel": "SIM:PROBE",
                            "gateways": {"read_only": dict(gateway)},
                        },
                    },
                },
                "services": {"live_standin": {"port": STANDIN_PORT}},
                "deployed_services": ["virtual_accelerator", "live_standin"],
            }
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Live-server helpers
# ---------------------------------------------------------------------------


@contextmanager
def _chip_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    ui_mode: str = "expert",
) -> Iterator[tuple[str, Any, Path]]:
    """Launch a real web-terminal hub wired for the control-target chip.

    The companion-backend patches are the ones every hub browser suite uses.
    What is specific to this feature is the environment rather than a patch:

    * ``posture_store.AGENT_DATA_ROOT_ENV_VAR`` — the ONE stamp both the
      control-context record reader and ``posture_store`` resolve through, and
      the stamp this feature puts in every session child's environment. Pinning
      it to *tmp_path* keeps every write off the real agent-data tree. Patching
      ``resolve_shared_data_root`` instead is a no-op on the store —
      ``posture_store`` reads this stamp first and binds the resolver at import
      — so it would redirect one half and leave the other writing into the
      repository's own ``var/agent_data``.
    * ``OSPREY_EXECUTION_MODE`` is cleared: a read-only *run* is a
      deployment-wide fact this process must not inherit from whatever ran
      before it, and it would zero every row's ``effective``.
    * ``OSPREY_POSTURE_SESSION`` is cleared for the same reason: it names
      whichever session happened to spawn this test process, and no read here
      may be answered for that stranger's key.
    * ``snapshot_session_ids`` answers an empty set. No session file is ever
      written here and nothing on the posture or target routes gates on one;
      pinning it empty keeps that a property rather than an accident.

    The **record is written before ``create_app``**, at :data:`ACTIVE_TARGET`
    and owned by this process. That is what makes the lifespan's claim a merge
    instead of a mint from ``load_osprey_config``, and what makes this terminal
    the owner — the two conditions behind ``data-enforceable="true"``.

    ``web.ui_mode`` reaches the page through ``app.state.web_ui_mode``, which
    the ``GET /`` handler reads per request; it is overridden post-startup, the
    same seam ``test_ui_mode_browser.py`` uses.

    Yields:
        (base_url, app, root) — the hub's address, its app (how a test reaches
        the PTY registry) and the stamped agent-data root (how a test plants a
        controls server's report or reads the record back).
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir(exist_ok=True)
    root = tmp_path / "agent_data"
    (root / posture_store.STATE_DIR_NAME).mkdir(parents=True, exist_ok=True)
    config = _write_config(tmp_path / "config.yml")

    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv("OSPREY_POSTURE_SESSION", raising=False)
    _reset_process_memos()
    write_control_context(root, target=ACTIVE_TARGET, generation=RECORD_GENERATION)

    patches = [
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=({"artifacts"}, [], None),
        ),
        patch(
            "osprey.interfaces.web_terminal.app._launch_panel_server",
            side_effect=publish_artifact_url(),
        ),
        patch(
            "osprey.interfaces.web_terminal.session_discovery.SessionDiscovery"
            ".snapshot_session_ids",
            side_effect=lambda *_args, **_kwargs: set(),
        ),
    ]
    try:
        with _apply_all(patches):
            from osprey.interfaces.web_terminal.app import create_app

            app = create_app(shell_command=list(_LONG_LIVED_SHELL), config_path=config)
            with _run_app_server(app) as base_url:
                app.state.web_ui_mode = ui_mode
                yield base_url, app, root
    finally:
        _reset_process_memos()


def _reset_process_memos() -> None:
    """Drop every cross-request memo this route family keeps.

    All three are keyed on a file signature or a path, and a tmp directory
    reused across tests could otherwise serve one test's record, render or
    narrowing to the next.
    """
    posture_store.invalidate_cache()
    control_context.invalidate_cache()
    websocket_routes._reset_rendered_config_memo()


#: A sweep in which every configured target answered. The switch gate refuses
#: ``reachability_unknown`` when a controls server is live but has published no
#: probe for the target being asked for — "nobody has looked yet" — so a report
#: planted without this would refuse every switch for a reason no test here is
#: about. The shape is the prober's own: a ``targets`` table under the block,
#: then one row per gateway role.
def _reachability_sweep() -> dict[str, Any]:
    probed_at = datetime.now(UTC).isoformat()
    return {
        "published_at": probed_at,
        "targets": {
            target: {"write_access": {"state": "reached", "probed_at": probed_at}}
            for target in (SWITCH_TARGET, ACTIVE_TARGET, POSTURE_TARGET)
        },
    }


def _publish_report(
    root: Path,
    *,
    applied_target: str = ACTIVE_TARGET,
    applied_generation: int | None = RECORD_GENERATION,
    last_switch: dict | None = None,
    server_pid: int | None = None,
) -> Path:
    """Publish one controls server's report under the stamped root.

    *server_pid* defaults to this test process, which is unambiguously alive —
    ``live_reports`` filters out a report whose writer is gone, so a report
    addressed to a corpse would be invisible and every assertion resting on it
    would pass for the wrong reason.

    ``applied_generation`` is what makes a report interesting: a row still on
    the generation before the record's is a server that has not caught up, and
    that lag is the whole of what keeps the chip in ``switching…``.
    """
    return write_server_report(
        root,
        os.getpid() if server_pid is None else server_pid,
        applied_target=applied_target,
        applied_generation=applied_generation,
        last_switch=last_switch,
        reachability=_reachability_sweep(),
    )


def _recorded_posture() -> dict[str, str]:
    """The narrowings on disk, as the connector and the hook will read them."""
    posture_store.invalidate_cache()
    return dict(posture_store.recorded_posture())


def _read_record():
    """The deployment's control-context record, re-read from disk."""
    control_context.invalidate_cache()
    return control_context.read_record()


def _pty_pid(app: Any, session_id: str) -> int:
    """The pid of the PTY this session runs in."""
    session = app.state.pty_registry.get_session(session_id)
    assert session is not None, f"no PTY session registered for {session_id}"
    pid = session.pid
    assert isinstance(pid, int) and pid > 0, f"the PTY reported no usable pid: {pid!r}"
    return pid


#: Seeded into every page before load: marks the onboarding tour as already
#: dismissed. Under the default `once` policy the invite card (scrim + modal)
#: would otherwise overlay the shell on the fresh profile these tests run
#: under and swallow every chip and popover click this suite drives. The tour
#: has its own dedicated coverage (tour.test.mjs).
_DISMISS_TOUR = "try { localStorage.setItem('osprey-tour-dismissed-v1', '1') } catch (e) {}"


def _row(page: Page, target: str) -> Any:
    """The popover row for *target*.

    Always re-resolved from the page: the popover subtree is REPLACED on every
    render (a 5 s idle poll, 500 ms while a switch is out), so an element
    handle held across a wait would be detached by the time it is used.
    """
    return page.locator(f'{POPOVER} .ctc-row[data-target="{target}"]')


def _toggle(page: Page, target: str) -> Any:
    """The row's writes switch — ``aria-checked`` is the readout."""
    return _row(page, target).locator(".ctc-toggle")


def _settled_chip(browser: Browser, base_url: str) -> tuple[Page, str | None]:
    """Open the hub and wait until the chip speaks for a writable deployment.

    A visible chip already means the whole chain ran: the module mounted and
    ``GET /api/terminal/posture`` answered — the chip stays hidden until a read
    succeeds, and it no longer waits for a session to exist first.

    ``data-enforceable="true"`` is the second half, and it is a fact about the
    *deployment*: it is fed by ``contextWritable(state)``, which is false when
    the record has no location and false when another terminal owns it. Both
    conditions are arranged in :func:`_chip_hub` before the server starts, so
    reaching this line proves the record is there and this terminal may write
    it — while it is false the popover locks every toggle.

    Returns:
        (page, session_id) — the id ``terminal.js`` settled on, which is
        ``None`` in a view whose terminal never connects. Nothing the chip does
        needs it; case (b) uses it to name the PTY it asserts was not
        respawned.
    """
    page = browser.new_page()
    page.add_init_script(_DISMISS_TOUR)
    page.goto(base_url, wait_until="domcontentloaded")

    expect(page.locator(CHIP)).to_be_visible(timeout=TIMEOUT)
    expect(page.locator(CHIP)).to_have_attribute("data-enforceable", "true", timeout=TIMEOUT)

    session_id = page.evaluate("() => localStorage.getItem('osprey-pty-session')")
    return page, session_id


def _open_popover(page: Page) -> None:
    """Click the chip open and wait for the card to be on screen."""
    page.locator(CHIP).click()
    expect(page.locator(POPOVER)).to_be_visible(timeout=TIMEOUT)
    expect(page.locator(CHIP)).to_have_attribute("aria-expanded", "true", timeout=TIMEOUT)
    expect(page.locator(CARD)).to_be_visible(timeout=TIMEOUT)


def _narrow(page: Page, target: str) -> None:
    """Turn *target*'s writes off through the popover, and wait for the row.

    Applies on click — turning off only ever removes reach, so no confirm is
    in the way, and the row's ``data-state`` flipping is the re-read landing.
    """
    _toggle(page, target).click()
    expect(_row(page, target)).to_have_attribute("data-state", "sandbox", timeout=TIMEOUT)


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
    with _chip_hub(tmp_path, monkeypatch) as (base_url, _app, _root):
        page, _session_id = _settled_chip(chromium_browser, base_url)
        try:
            # The chip's home is the global header, and the terminal card's own
            # header is back to what it was before the badge: LED, label,
            # selector, + New. The badge module is deleted on this branch, so
            # this is the guard against a second control-target affordance
            # reappearing in the card rather than a live check.
            expect(page.locator(f".header-actions {CHIP}")).to_have_count(1)
            expect(page.locator(".terminal-header .posture-badge")).to_have_count(0)
            expect(page.locator(f".terminal-header {CHIP}")).to_have_count(0)

            expect(page.locator(CHIP_SHORT)).to_have_text(NAMES[ACTIVE_TARGET], timeout=TIMEOUT)
            expect(page.locator(CHIP_STATE)).to_have_text("writes on", timeout=TIMEOUT)
            expect(page.locator(CHIP)).to_have_attribute("data-target-kind", "va")
            expect(page.locator(CHIP)).to_have_attribute("aria-expanded", "false")

            _open_popover(page)

            # The machine the agent is on is the card; every other machine a row.
            card = page.locator(CARD)
            expect(card).to_have_attribute("data-target", ACTIVE_TARGET)
            expect(card.locator(".ctc-card-eyebrow")).to_have_text("The agent is on")
            expect(card.locator(".ctc-name")).to_have_text(NAMES[ACTIVE_TARGET])
            expect(page.locator(f"{POPOVER} .ctc-row")).to_have_count(2, timeout=TIMEOUT)
            for target in (SWITCH_TARGET, POSTURE_TARGET):
                row = _row(page, target)
                expect(row.locator(".ctc-name")).to_have_text(NAMES[target])
                expect(row.locator(".ctc-switch-state")).to_have_text("on")
                # The server's own label lives inside the ⓘ tooltip, not at rest.
                tip = row.locator(".ctc-tip").text_content()
                assert tip and LABELS[target] in tip, tip

            expect(card.locator(".ctc-switch")).to_have_count(0)
            # The one target this render will switch onto.
            expect(_row(page, SWITCH_TARGET).locator(".ctc-switch")).to_be_visible()
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (b) narrowing
# ---------------------------------------------------------------------------


def test_narrowing_asks_nothing_lands_in_the_record_and_respawns_nothing(
    tmp_path, monkeypatch, chromium_browser
):
    """Read-only applies on click, is written where the connector reads, and is free.

    Three separate claims, and each is load-bearing:

    * no confirm — narrowing only ever removes reach, so asking would be
      ceremony over a gesture one click undoes;
    * the record agrees, because ``posture_store.recorded_posture`` reading
      that field is what the connector and the hook consult — a row that
      agreed with nothing would be a target the operator believes is sandboxed
      and the agent is not. The map is keyed by TARGET and by nothing else:
      one narrowing for the deployment, not one per session;
    * the PTY is the same process afterwards (FR17). The posture is read live
      on every write, so a respawn would cost the operator their session for
      nothing.
    """
    with _chip_hub(tmp_path, monkeypatch) as (base_url, app, _root):
        page, session_id = _settled_chip(chromium_browser, base_url)
        assert session_id, "the terminal card never settled on a session id"
        pty_pid = _pty_pid(app, session_id)
        try:
            _open_popover(page)
            row = _row(page, POSTURE_TARGET)
            expect(row).to_have_attribute("data-state", "writes")
            expect(_toggle(page, POSTURE_TARGET)).to_have_attribute("aria-checked", "true")

            _narrow(page, POSTURE_TARGET)

            # Nothing asked, and the switch is the readout as well as the
            # control: it flipped to off.
            expect(page.locator(OPEN_MODAL)).to_have_count(0)
            expect(_row(page, POSTURE_TARGET).locator(".ctc-switch-state")).to_have_text("off")
            expect(_toggle(page, POSTURE_TARGET)).to_have_attribute("aria-checked", "false")
            # The card is untouched: a posture is per target.
            expect(page.locator(CARD)).to_have_attribute("data-state", "writes")

            assert _recorded_posture() == {POSTURE_TARGET: "sandbox"}, _recorded_posture()

            assert _pty_pid(app, session_id) == pty_pid, "the session was respawned"
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (c) widening
# ---------------------------------------------------------------------------


def test_arming_confirms_and_cancel_changes_nothing(tmp_path, monkeypatch, chromium_browser):
    """Only widening asks — and Cancel leaves the row and the record as they were.

    Arming is the gesture after which a write the agent makes can land, so it
    is the one direction that confirms. The dialog names the target it is
    about; the popover deliberately stays open beneath it, so the row the
    question is about is still on screen. A toggle that fired on the way to the
    dialog would be the worst possible failure of a confirm step, which is why
    the record is asserted on both sides of the cancellation.
    """
    with _chip_hub(tmp_path, monkeypatch) as (base_url, _app, _root):
        page, _session_id = _settled_chip(chromium_browser, base_url)
        try:
            _open_popover(page)
            _narrow(page, POSTURE_TARGET)

            # --- turning on, cancelled ---
            _toggle(page, POSTURE_TARGET).click()
            expect(page.locator(MODAL_TITLE)).to_have_text(
                f"Turn writes on for {NAMES[POSTURE_TARGET]}?", timeout=TIMEOUT
            )
            # The rows stay readable underneath: the confirm is a layer above
            # the popover, not a replacement for it.
            expect(page.locator(POPOVER)).to_be_visible()
            page.locator(MODAL_CANCEL).click()

            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
            expect(_row(page, POSTURE_TARGET)).to_have_attribute("data-state", "sandbox")
            assert _recorded_posture() == {POSTURE_TARGET: "sandbox"}

            # --- turning on, confirmed ---
            _toggle(page, POSTURE_TARGET).click()
            expect(page.locator(MODAL_CONFIRM)).to_have_text("Turn writes on", timeout=TIMEOUT)
            page.locator(MODAL_CONFIRM).click()

            expect(_row(page, POSTURE_TARGET)).to_have_attribute(
                "data-state", "writes", timeout=TIMEOUT
            )
            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
            # Widening is the ABSENCE of a narrowing, so the row's key is gone
            # from the record's posture map rather than set to "writes" — the
            # field only ever records what was taken away.
            assert POSTURE_TARGET not in _recorded_posture(), _recorded_posture()
        finally:
            page.close()


def test_narrowing_moves_the_tooltip_onto_the_read_gateway(tmp_path, monkeypatch, chromium_browser):
    """A narrowed row names the gateway it is actually talking to.

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
    with _chip_hub(tmp_path, monkeypatch) as (base_url, _app, _root):
        page, _session_id = _settled_chip(chromium_browser, base_url)
        try:
            _open_popover(page)

            # Armed: the write gateway, and only it.
            ident = _row(page, ENDPOINT_TARGET).locator(".ctc-tip-ident")
            expect(ident).to_contain_text(WRITE_ENDPOINT, timeout=TIMEOUT)
            assert READ_ENDPOINT not in (ident.text_content() or "")

            _narrow(page, ENDPOINT_TARGET)

            # Narrowed: the read gateway. Re-resolved from the page, since the
            # refetch that carried the new posture replaced the node above.
            ident = _row(page, ENDPOINT_TARGET).locator(".ctc-tip-ident")
            expect(ident).to_contain_text(READ_ENDPOINT, timeout=TIMEOUT)
            tip = ident.text_content() or ""
            assert WRITE_ENDPOINT not in tip
            # Identity is derived from the ceiling and does not move with the
            # posture: the operator is standing in front of the same machine.
            assert LABELS[ENDPOINT_TARGET] in tip

            # The confirm that widens again names no gateway at all.
            _toggle(page, ENDPOINT_TARGET).click()
            expect(page.locator(MODAL_TITLE)).to_have_text(
                f"Turn writes on for {NAMES[ENDPOINT_TARGET]}?", timeout=TIMEOUT
            )
            body = page.locator(f"{OPEN_MODAL} .posture-modal-body").text_content() or ""
            assert body.strip(), "the confirm rendered an empty body"
            assert not _HOST_PORT.search(body), body
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
    with _chip_hub(tmp_path, monkeypatch) as (base_url, _app, root):
        # A server that is up and reporting the pre-switch generation. It is
        # not blocking anything (no `applying` block), so the switch is
        # allowed; it simply has not caught up once the record moves.
        _publish_report(root)
        page, _session_id = _settled_chip(chromium_browser, base_url)
        try:
            _open_popover(page)
            _row(page, SWITCH_TARGET).locator(".ctc-switch").click()

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
                "Real machine — writes move hardware."
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
            expect(page.locator(CHIP)).to_have_attribute("data-target-kind", SWITCH_TARGET)

            record = _read_record()
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
    with _chip_hub(tmp_path, monkeypatch) as (base_url, _app, root):
        page, _session_id = _settled_chip(chromium_browser, base_url)
        try:
            _publish_report(
                root,
                last_switch={
                    "status": "applying",
                    "target": SWITCH_TARGET,
                    "generation": RECORD_GENERATION,
                    "request_id": "11111111-2222-3333-4444-555555555555",
                },
            )

            _open_popover(page)
            _row(page, SWITCH_TARGET).locator(".ctc-switch").click()
            expect(page.locator(MODAL_CONFIRM)).to_be_visible(timeout=TIMEOUT)
            page.locator(MODAL_CONFIRM).click()

            error = page.locator(MODAL_ERROR)
            expect(error).to_be_visible(timeout=TIMEOUT)
            expect(error).to_contain_text("already in flight on pid", timeout=TIMEOUT)
            expect(page.locator(OPEN_MODAL)).to_have_count(1)
            expect(page.locator(CHIP)).not_to_have_attribute("data-pending", "true")

            record = _read_record()
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
    with _chip_hub(tmp_path, monkeypatch, ui_mode=ui_mode) as (base_url, _app, _root):
        page, _session_id = _settled_chip(chromium_browser, base_url)
        try:
            expect(page.locator("html")).to_have_attribute("data-ui-mode", ui_mode)
            _open_popover(page)

            row = _row(page, SWITCH_TARGET)

            # --- what an operator acts on, at rest, in either density ---
            expect(row.locator(".ctc-name")).to_have_text(NAMES[SWITCH_TARGET])
            assert row.locator(".ctc-tip-what").text_content() == "Writes move hardware"
            expect(row.locator(".ctc-switch-state")).to_have_text("on")
            expect(row.locator(".ctc-toggle")).to_be_visible()
            expect(row.locator(".ctc-switch")).to_be_visible()
            expect(page.locator(FOOT_NOTE)).to_have_text("Applies deployment-wide")

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
