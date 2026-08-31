"""Browser tests: the control-target header chip, end to end.

The chip in the page header — ``● Simulator · writes on ▾`` — and the popover
behind it are the operator's only route between writes on and writes off on a
live session, and now the only route onto another control target. Every
interesting part of that is client-side behavior a FastAPI TestClient cannot
observe: the chip only exists once ``terminal.js`` has reported a session id,
the card, the rows and both confirms are built in the DOM by
``control-target-popover.js``, and every
one of them repaints from a *re-read* of ``GET /api/terminal/posture`` rather
than from what the module last POSTed. A real browser is what proves the chip
an operator looks at agrees with the store the connector will read.

Coverage (one test each):

  (a) the chip names the machine the session stands on by its display name,
      and the popover renders that machine as the card and every other
      configured target as a row, the server's own label demoted to tooltips.
  (b) turning a target's writes off asks nothing, lands in the server's own
      store under the key that session answers to, and respawns nothing — the
      posture is read live, so the PTY the operator is talking to is the same
      process afterwards.
  (c) turning writes back on is the direction that confirms: the dialog names
      the machine, Cancel is a true no-op on the row *and* in the store, and
      only a confirmed dialog widens.
  (d) Switch confirms, is accepted as ``202``, and puts the chip and the row
      into ``switching…`` with a request file addressed to the controls server
      — this route only *asks*; the reconciler that would answer is not running
      in these tests.
  (e) a refusal keeps the server's own sentence on screen, in the place the
      operator is looking: on the row for a gesture that asked nothing (a
      session the server has never seen, refused with "send one prompt first"),
      and inside the dialog for one raised from a confirm — which stays up to
      carry it.
  (f) both UI modes render the SAME popover DOM and show all of it: the
      redesign leaves no popover node for the density stylesheet to gate —
      endpoints and the server's label are hover vocabulary in both modes.

Session bootstrapping. The chip needs a session id, and one arrives the way it
does in production: a plain page load opens a NEW terminal WebSocket, and the
route mints the session UUID itself (it dictates it on the CLI's command line)
and confirms it immediately in a ``session_info`` frame — no Claude binary and
no session discovery involved. The id the card settled on is then read back
from ``localStorage['osprey-pty-session']``, which ``terminal.js`` writes on
that same frame. The PTY command is a long-lived ``sleep`` because the route
appends ``--session-id``/``--resume`` arguments that ``echo`` would choke on.

Two more facts have to be true before any toggle in the popover can move, and
both are arranged in :func:`_settled_chip`:

* ``SessionDiscovery.snapshot_session_ids`` is patched to a set the test owns
  (the same seam ``test_posture_routes.py`` uses): "this session exists on
  disk" is otherwise only true after a real model turn has written a
  ``.jsonl``, and it is exactly the distinction case (e) turns on.
* a controls-server state record is published for the PTY's own pid. Without
  one the route answers ``enforceable: false`` — a PTY that resolves no record
  is a session whose toggles would govern nothing — and every toggle in the
  popover is locked. The record is written *after* the page has settled,
  because it is addressed to a pid only the running PTY can supply; the chip
  picks it up on its 5 s idle poll.

The agent-data root is stamped with ``OSPREY_AGENT_DATA_ROOT``, which is the
one seam ``target_state`` and ``session_store`` both prefer — patching
``resolve_shared_data_root`` would redirect one of them and leave the other
writing into the repository's own ``var/agent_data``.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_posture_toggle_browser.py -m browser -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import yaml

from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey.mcp_server.control_system import target_state
from osprey_connectors import session_store
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

#: The target the published record puts the session on. Chosen so the roster
#: carries all three interesting rows at once: ``va`` is active (no Switch),
#: ``live`` is switchable, and ``standin`` is a real machine that is not.
ACTIVE_TARGET = "va"

#: The row every posture gesture below is made on. Deliberately not the active
#: one: narrowing the target a session stands on also raises the "applies after
#: the running execution finishes" line when a realign is pending, which is a
#: different contract from the one these tests pin.
POSTURE_TARGET = "standin"

#: The row Switch is exercised on — the only one this render offers it for.
SWITCH_TARGET = "live"

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
    """
    gateway = {"address": "gw", "port": 5064, "use_name_server": True}
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
                                "write_access": dict(gateway),
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
    known_ids: set[str],
    ui_mode: str = "expert",
) -> Iterator[tuple[str, Any]]:
    """Launch a real web-terminal hub wired for the control-target chip.

    The companion-backend patches are the ones every hub browser suite uses.
    What is specific to this feature is the environment rather than a patch:

    * ``session_store.AGENT_DATA_ROOT_ENV_VAR`` — the ONE stamp both
      ``target_state`` (where the controls-server record and the switch request
      live) and ``session_store`` (where a narrowing is recorded) resolve
      through, and the stamp this feature puts in every session child's
      environment. Pinning it to *tmp_path* keeps every write off the real
      agent-data tree. Patching ``resolve_shared_data_root`` instead is a
      no-op on the store — ``session_store`` reads this stamp first and binds
      the resolver at import — so it would redirect one half and leave the
      other writing into the repository's own ``var/agent_data``.
    * ``OSPREY_EXECUTION_MODE`` is cleared: a read-only *run* is a
      deployment-wide fact this process must not inherit from whatever ran
      before it, and it would zero every row's ``effective``.
    * ``OSPREY_POSTURE_SESSION`` is cleared for the same reason: it names
      whichever session happened to spawn this test process, and no store read
      here may be answered for that stranger's key.
    * ``snapshot_session_ids`` — POST refuses (409) an id that names no session
      file, and no session file is ever written here. *known_ids* is the test's
      own set and is read on every call, so a test can add the id the server
      minted once the page has told it what that id is.

    ``web.ui_mode`` reaches the page through ``app.state.web_ui_mode``, which
    the ``GET /`` handler reads per request; it is overridden post-startup, the
    same seam ``test_ui_mode_browser.py`` uses.

    Yields:
        (base_url, app) — the hub's address and its app, which is how a test
        reaches the PTY registry for the pid a state record is addressed to.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir(exist_ok=True)
    root = tmp_path / "agent_data"
    root.mkdir(exist_ok=True)
    config = _write_config(tmp_path / "config.yml")

    monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv("OSPREY_POSTURE_SESSION", raising=False)
    _reset_process_memos()

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
            side_effect=lambda *_args, **_kwargs: set(known_ids),
        ),
    ]
    try:
        with _apply_all(patches):
            from osprey.interfaces.web_terminal.app import create_app

            app = create_app(shell_command=list(_LONG_LIVED_SHELL), config_path=config)
            with _run_app_server(app) as base_url:
                app.state.web_ui_mode = ui_mode
                yield base_url, app
    finally:
        _reset_process_memos()


def _reset_process_memos() -> None:
    """Drop every cross-request memo this route family keeps.

    All three are keyed on a pid, a file signature or a path, and a tmp
    directory reused across tests could otherwise serve one test's record,
    render or narrowing to the next.
    """
    session_store.invalidate_cache()
    websocket_routes._reset_session_record_memo()
    websocket_routes._reset_rendered_config_memo()


def _publish_record(
    *,
    target: str = ACTIVE_TARGET,
    owner_ppid: int,
    server_pid: int | None = None,
    last_switch: dict | None = None,
) -> Path:
    """Publish one controls-server state record under the stamped root.

    *owner_ppid* is the PTY's own pid: the resolver walks the ancestors of each
    record's ``owner_ppid`` and asks whether the PTY pid is on that chain, so
    the PTY itself is the shortest honest chain there is. ``server_pid``
    defaults to this test process, which is unambiguously alive — a record
    whose writer is dead is filtered out before it is ever matched.
    """
    server_pid = os.getpid() if server_pid is None else server_pid
    # The writer's own name composer, rather than a second spelling of it here:
    # a name the resolver's `_pid_from_name` rejects would fail as a 15 s wait
    # on `data-enforceable="true"` in every case below, pointing at the chip
    # rather than at this fixture.
    path = target_state.state_file_path(server_pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "target": target,
                "generation": 1,
                "server_pid": server_pid,
                "owner_ppid": owner_ppid,
                # Empty: the render names the targets through the same
                # `target_display_metadata` a controls server would, so a record
                # that has published no metadata yet still yields real labels.
                "targets": {},
                "children": [],
                "reachability": None,
                "last_switch": last_switch,
                "last_posture_realign": None,
            }
        ),
        encoding="utf-8",
    )
    return path


def _stored_postures() -> dict[str, dict[str, str]]:
    """The narrowings on disk, as the connector and the hook will read them."""
    path = session_store.store_path()
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _pty_pid(app: Any, session_id: str) -> int:
    """The pid of the PTY this session runs in."""
    session = app.state.pty_registry.get_session(session_id)
    assert session is not None, f"no PTY session registered for {session_id}"
    pid = session.pid
    assert isinstance(pid, int) and pid > 0, f"the PTY reported no usable pid: {pid!r}"
    return pid


def _row(page: Page, target: str) -> Any:
    """The popover row for *target*.

    Always re-resolved from the page: the popover subtree is REPLACED on every
    render (a 5 s idle poll, 500 ms while a switch is out), so an element
    handle held across a wait would be detached by the time it is used.
    """
    return page.locator(f'{POPOVER} .ctc-row[data-target="{target}"]')


def _verb(page: Page, target: str) -> Any:
    """The row's one write verb — reads ``Turn writes off`` or ``Turn writes on``."""
    return _row(page, target).locator(".ctc-verb")


def _settled_chip(
    browser: Browser,
    base_url: str,
    app: Any,
    known_ids: set[str],
) -> tuple[Page, str, int]:
    """Open the hub and wait until the chip speaks for an enforceable session.

    A visible chip already means the whole chain ran: the terminal connected,
    the route confirmed a session id, and ``GET /api/terminal/posture``
    answered for it (the chip stays hidden until a read succeeds). Chip
    visibility is what badge visibility used to be — the "this session has
    settled" signal every test here starts from.

    Two things are then arranged that only a settled session can supply: the id
    is added to the discovery set (POST refuses an id that names no session
    file), and a controls-server record is published against the PTY's pid.
    ``data-enforceable="true"`` is the observable proof both landed — until the
    record resolves, the route says the toggles would govern nothing and the
    popover locks every one of them.

    Returns:
        (page, session_id, pty_pid).
    """
    page = browser.new_page()
    page.goto(base_url, wait_until="domcontentloaded")

    expect(page.locator(CHIP)).to_be_visible(timeout=TIMEOUT)

    session_id = page.evaluate("() => localStorage.getItem('osprey-pty-session')")
    assert session_id, "the terminal card never settled on a session id"

    known_ids.add(session_id)
    pty_pid = _pty_pid(app, session_id)
    _publish_record(owner_ppid=pty_pid)

    expect(page.locator(CHIP)).to_have_attribute("data-enforceable", "true", timeout=TIMEOUT)
    return page, session_id, pty_pid


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
    _verb(page, target).click()
    expect(_row(page, target)).to_have_attribute("data-state", "sandbox", timeout=TIMEOUT)


# ---------------------------------------------------------------------------
# (a) the chip and its roster
# ---------------------------------------------------------------------------


def test_the_chip_names_the_session_target_and_lists_every_row(
    tmp_path, monkeypatch, chromium_browser
):
    """The chip speaks for the machine the session stands on; the popover lists all.

    The record puts the session on the simulator, so the chip reads
    ``Simulator · writes on`` — the display name the render minted, not the
    target name — and the popover renders that machine as the card ("The agent
    is on") and each other configured target as a row named the same way, the
    server's own label demoted to the identity tooltip. The card offers no
    Switch; switching to the target you are on is a no-op.
    """
    known_ids: set[str] = set()

    with _chip_hub(tmp_path, monkeypatch, known_ids=known_ids) as (base_url, app):
        page, _session_id, _pid = _settled_chip(chromium_browser, base_url, app, known_ids)
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
                expect(row.locator(".ctc-pill")).to_have_text("writes on")
                # The server's own label is hover vocabulary now, not rest copy.
                title = row.locator(".ctc-name-line").get_attribute("title")
                assert title and LABELS[target] in title, title

            expect(card.locator(".ctc-switch")).to_have_count(0)
            # The one target this render will switch onto.
            expect(_row(page, SWITCH_TARGET).locator(".ctc-switch")).to_be_visible()
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (b) narrowing
# ---------------------------------------------------------------------------


def test_narrowing_asks_nothing_lands_in_the_store_and_respawns_nothing(
    tmp_path, monkeypatch, chromium_browser
):
    """Read-only applies on click, is written where the connector reads, and is free.

    Three separate claims, and each is load-bearing:

    * no confirm — narrowing only ever removes reach, so asking would be
      ceremony over a gesture one click undoes;
    * the server's own store agrees, under the key this session answers to,
      because the store is what ``session_store`` hands the connector and the
      hook — a row that agreed with nothing would be a target the operator
      believes is sandboxed and the agent is not;
    * the PTY is the same process afterwards (FR17). The posture is read live
      on every write, so a respawn would cost the operator their session for
      nothing.
    """
    known_ids: set[str] = set()

    with _chip_hub(tmp_path, monkeypatch, known_ids=known_ids) as (base_url, app):
        page, session_id, pty_pid = _settled_chip(chromium_browser, base_url, app, known_ids)
        try:
            _open_popover(page)
            row = _row(page, POSTURE_TARGET)
            expect(row).to_have_attribute("data-state", "writes")
            expect(_verb(page, POSTURE_TARGET)).to_have_text("Turn writes off")

            _narrow(page, POSTURE_TARGET)

            # Nothing asked, and the row is the readout as well as the
            # control: the pill flipped and the verb reversed.
            expect(page.locator(OPEN_MODAL)).to_have_count(0)
            expect(_row(page, POSTURE_TARGET).locator(".ctc-pill")).to_have_text("writes off")
            expect(_verb(page, POSTURE_TARGET)).to_have_text("Turn writes on")
            # The card is untouched: a posture is per target.
            expect(page.locator(CARD)).to_have_attribute("data-state", "writes")

            # A session that was never rekeyed answers to ONE key:
            # PtyRegistry.audit_session_key returns its argument unchanged.
            # Asserted so the single-key store below reads as the expected
            # shape rather than as a dual write that happened to coincide. The
            # real dual-key write (a live child outliving a rekey) is pinned in
            # test_posture_durability.py::TestDualKeyWrite, which a browser
            # cannot reach.
            spawn_key = websocket_routes._spawn_posture_key(app, session_id)
            assert spawn_key == session_id, spawn_key
            stored = _stored_postures()
            assert stored == {session_id: {POSTURE_TARGET: "sandbox"}}, stored

            assert _pty_pid(app, session_id) == pty_pid, "the session was respawned"
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (c) widening
# ---------------------------------------------------------------------------


def test_arming_confirms_and_cancel_changes_nothing(tmp_path, monkeypatch, chromium_browser):
    """Only widening asks — and Cancel leaves the row and the store as they were.

    Arming is the gesture after which a write the agent makes can land, so it
    is the one direction that confirms. The dialog names the target it is
    about; the popover deliberately stays open beneath it, so the row the
    question is about is still on screen. A toggle that fired on the way to the
    dialog would be the worst possible failure of a confirm step, which is why
    the store is asserted on both sides of the cancellation.
    """
    known_ids: set[str] = set()

    with _chip_hub(tmp_path, monkeypatch, known_ids=known_ids) as (base_url, app):
        page, session_id, _pid = _settled_chip(chromium_browser, base_url, app, known_ids)
        try:
            _open_popover(page)
            _narrow(page, POSTURE_TARGET)

            # --- turning on, cancelled ---
            _verb(page, POSTURE_TARGET).click()
            expect(page.locator(MODAL_TITLE)).to_have_text(
                f"Turn writes on for {NAMES[POSTURE_TARGET]}?", timeout=TIMEOUT
            )
            # The rows stay readable underneath: the confirm is a layer above
            # the popover, not a replacement for it.
            expect(page.locator(POPOVER)).to_be_visible()
            page.locator(MODAL_CANCEL).click()

            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
            expect(_row(page, POSTURE_TARGET)).to_have_attribute("data-state", "sandbox")
            assert _stored_postures()[session_id] == {POSTURE_TARGET: "sandbox"}

            # --- turning on, confirmed ---
            _verb(page, POSTURE_TARGET).click()
            expect(page.locator(MODAL_CONFIRM)).to_have_text("Turn writes on", timeout=TIMEOUT)
            page.locator(MODAL_CONFIRM).click()

            expect(_row(page, POSTURE_TARGET)).to_have_attribute(
                "data-state", "writes", timeout=TIMEOUT
            )
            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
            # Widening is the ABSENCE of a narrowing, so the row's key is gone
            # from this session's entry rather than set to "writes" — the store
            # only ever records what was taken away.
            stored = _stored_postures()
            assert POSTURE_TARGET not in stored.get(session_id, {}), stored
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (d) switching
# ---------------------------------------------------------------------------


def test_switch_confirms_is_accepted_and_reads_switching(tmp_path, monkeypatch, chromium_browser):
    """Switch asks, is accepted 202, and the chip waits out loud.

    The route does not switch anything: it writes one request file addressed to
    the controls server's pid, and the reconciler inside that server answers by
    publishing ``last_switch``. No reconciler runs here, so what is pinned is
    the whole of the browser's half — the confirm naming the target and the
    posture the session will have THERE, the request landing on disk with that
    target, and both the chip and the row reading ``switching…`` while it is
    outstanding.
    """
    known_ids: set[str] = set()

    with _chip_hub(tmp_path, monkeypatch, known_ids=known_ids) as (base_url, app):
        page, _session_id, _pid = _settled_chip(chromium_browser, base_url, app, known_ids)
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
                "Writes are on there for your session"
            )
            expect(page.locator(f"{OPEN_MODAL} .posture-modal-live")).to_have_text(
                "Real machine — writes move hardware."
            )
            page.locator(MODAL_CONFIRM).click()

            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
            expect(page.locator(CHIP)).to_have_attribute("data-pending", "true", timeout=TIMEOUT)
            expect(page.locator(CHIP_STATE)).to_have_text("switching…", timeout=TIMEOUT)
            expect(_row(page, SWITCH_TARGET).locator(".ctc-outcome")).to_have_text(
                "switching…", timeout=TIMEOUT
            )
            # `data-state` keeps describing the machine the session is still
            # on: nothing has switched yet.
            expect(page.locator(CHIP)).to_have_attribute("data-target-kind", "va")

            request = target_state.read_request(os.getpid())
            assert request is not None, "no switch request was addressed to the controls server"
            assert request["target"] == SWITCH_TARGET, request
            assert request["request_id"], request
        finally:
            page.close()


# ---------------------------------------------------------------------------
# (e) an unstarted session is already addressable
# ---------------------------------------------------------------------------


def test_an_unstarted_session_accepts_a_narrowing_the_moment_it_opens(
    tmp_path, monkeypatch, chromium_browser
):
    """A session with no file on disk narrows all the same.

    ``known_ids`` stays empty, so the id the chip is on names no session file —
    the state a terminal is in before its first prompt. The store only ever
    narrows and both spawn paths read it before the first write, so there is
    nothing an unstarted session could evade: the gesture lands in the store
    under this session's key, and the row settles into the narrowed state
    instead of surfacing a remedy sentence.
    """
    known_ids: set[str] = set()

    with _chip_hub(tmp_path, monkeypatch, known_ids=known_ids) as (base_url, app):
        # Deliberately NOT _settled_chip: that helper makes the session
        # addressable, and working WITHOUT that fact is the case.
        page = chromium_browser.new_page()
        page.goto(base_url, wait_until="domcontentloaded")
        try:
            expect(page.locator(CHIP)).to_be_visible(timeout=TIMEOUT)
            session_id = page.evaluate("() => localStorage.getItem('osprey-pty-session')")
            assert session_id, "the terminal card never settled on a session id"
            _publish_record(owner_ppid=_pty_pid(app, session_id))
            expect(page.locator(CHIP)).to_have_attribute(
                "data-enforceable", "true", timeout=TIMEOUT
            )

            _open_popover(page)
            _verb(page, POSTURE_TARGET).click()

            expect(_row(page, POSTURE_TARGET)).to_have_attribute(
                "data-state", "sandbox", timeout=TIMEOUT
            )
            stored = _stored_postures()
            assert stored == {session_id: {POSTURE_TARGET: "sandbox"}}, stored
        finally:
            page.close()


def test_a_refused_switch_keeps_its_sentence_inside_the_confirm(
    tmp_path, monkeypatch, chromium_browser
):
    """A refusal raised from a confirm stays in the confirm, which stays up.

    One switch request is outstanding at a time, and one is planted here before
    the operator clicks — the shape a second open tab, or a colleague, would
    produce. The route answers 409 and the dialog is where the operator is
    looking, so the sentence goes there and the dialog is kept up to carry it:
    dismissing it to put the reason on a row behind would hide the answer to
    the question they had just been asked. Nothing was requested, so the chip
    must not fall into ``switching…`` either.
    """
    known_ids: set[str] = set()

    with _chip_hub(tmp_path, monkeypatch, known_ids=known_ids) as (base_url, app):
        page, _session_id, _pid = _settled_chip(chromium_browser, base_url, app, known_ids)
        try:
            target_state.write_request(
                {
                    "request_id": "11111111-2222-3333-4444-555555555555",
                    "target": ACTIVE_TARGET,
                    "server_pid": os.getpid(),
                    "requested_by": "someone@example",
                }
            )

            _open_popover(page)
            _row(page, SWITCH_TARGET).locator(".ctc-switch").click()
            expect(page.locator(MODAL_CONFIRM)).to_be_visible(timeout=TIMEOUT)
            page.locator(MODAL_CONFIRM).click()

            error = page.locator(MODAL_ERROR)
            expect(error).to_be_visible(timeout=TIMEOUT)
            expect(error).to_contain_text("has not been answered yet", timeout=TIMEOUT)
            expect(page.locator(OPEN_MODAL)).to_have_count(1)
            expect(page.locator(CHIP)).not_to_have_attribute("data-pending", "true")
            expect(page.locator(CHIP_STATE)).to_have_text("writes on")
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
    name, the consequence line, the pill, the verb, Switch — is what an
    operator acts on and is shown in either density. So the invariant pinned
    here is the stronger one: same DOM, same visibility, and the confirms
    identical, whichever mode the deployment renders.

    The mode is driven the way the deployment drives it — ``web.ui_mode``
    reaching the page as the server-rendered ``<html data-ui-mode>`` attribute
    — not by poking the attribute from the test.
    """
    known_ids: set[str] = set()

    with _chip_hub(tmp_path, monkeypatch, known_ids=known_ids, ui_mode=ui_mode) as (
        base_url,
        app,
    ):
        page, _session_id, _pid = _settled_chip(chromium_browser, base_url, app, known_ids)
        try:
            expect(page.locator("html")).to_have_attribute("data-ui-mode", ui_mode)
            _open_popover(page)

            row = _row(page, SWITCH_TARGET)

            # --- what an operator acts on, at rest, in either density ---
            expect(row.locator(".ctc-name")).to_have_text(NAMES[SWITCH_TARGET])
            expect(row.locator(".ctc-desc")).to_have_text("Writes move hardware")
            expect(row.locator(".ctc-pill")).to_have_text("writes on")
            expect(row.locator(".ctc-verb")).to_be_visible()
            expect(row.locator(".ctc-switch")).to_be_visible()
            expect(page.locator(FOOT_NOTE)).to_have_text("Your session only")

            # --- the machine vocabulary stays on hover, in either density ---
            title = row.locator(".ctc-name-line").get_attribute("title")
            assert title and LABELS[SWITCH_TARGET] in title, title
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
