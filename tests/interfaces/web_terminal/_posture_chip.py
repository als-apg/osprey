"""The control-target chip's deployment, selectors and waits, for its browser suites.

:func:`chip_hub` serves a real web-terminal hub whose control-context record
exists, is owned by this process, and carries all three control targets before
``create_app`` runs, which is what unlocks every toggle in the popover.
:func:`settled_chip` opens a page on it and waits until the chip speaks for a
writable deployment. The selectors and target vocabulary below are the ones
every chip suite asserts against: two suites disagreeing about what a writable
deployment looks like would drift apart.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import yaml

from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey_connectors import control_context, posture_store
from osprey_connectors_tests._harness import write_control_context, write_server_report
from tests.interfaces._hub import LONG_LIVED_SHELL, hub_server

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import pytest

with suppress(ImportError):
    from playwright.sync_api import Browser, Page, expect

# Every chip wait uses one generous bound rather than a tuned-per-step
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
HOST_PORT = re.compile(r"\b\S+:\d{2,5}\b")

#: The target the record puts the deployment on. Chosen so the roster carries
#: all three interesting rows at once: ``va`` is active (no Switch), ``live``
#: is switchable, and ``standin`` is a real machine that is not.
ACTIVE_TARGET = "va"

#: The generation the record is seeded at. Deliberately not 0, so a test that
#: asserts a switch moved it can tell the new value from "never set".
RECORD_GENERATION = 1

#: The row every posture gesture is made on. Deliberately not the active
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


def write_config(path: Path, *, writes_enabled: bool = True) -> Path:
    """A render carrying all three control targets, with writes armed.

    Mirrors the render ``test_posture_get_contract.py`` pins the route's own
    answers against: ``epics`` is the facility's own machine, ``live_standin``
    the co-deployed stand-in, ``virtual_accelerator`` the simulator — three
    connector blocks, therefore three targets and three rows.

    The keys beyond the gateways are what make a switch judgeable at all: a
    channel to probe and the
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
                    "limits_checking": {"enabled": True, "mode": "exclusive"},
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
def chip_hub(
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
    config = write_config(tmp_path / "config.yml")

    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv("OSPREY_POSTURE_SESSION", raising=False)
    reset_process_memos()
    write_control_context(root, target=ACTIVE_TARGET, generation=RECORD_GENERATION)

    try:
        with hub_server(
            workspace,
            patch(
                "osprey.interfaces.web_terminal.session_discovery.SessionDiscovery"
                ".snapshot_session_ids",
                side_effect=lambda *_args, **_kwargs: set(),
            ),
            shell_command=LONG_LIVED_SHELL,
            config_path=config,
        ) as (base_url, app):
            app.state.web_ui_mode = ui_mode
            yield base_url, app, root
    finally:
        reset_process_memos()


def reset_process_memos() -> None:
    """Drop every cross-request memo this route family keeps.

    All three are keyed on a file signature or a path, and a tmp directory
    reused across tests could otherwise serve one test's record, render or
    narrowing to the next.
    """
    posture_store.invalidate_cache()
    control_context.invalidate_cache()
    websocket_routes.reset_rendered_config_memo()


#: A sweep in which every configured target answered. The switch gate refuses
#: ``reachability_unknown`` when a controls server is live but has published no
#: probe for the target being asked for — "nobody has looked yet" — so a report
#: planted without this would refuse every switch for a reason no test here is
#: about. The shape is the prober's own: a ``targets`` table under the block,
#: then one row per gateway role.
def reachability_sweep() -> dict[str, Any]:
    probed_at = datetime.now(UTC).isoformat()
    return {
        "published_at": probed_at,
        "targets": {
            target: {"write_access": {"state": "reached", "probed_at": probed_at}}
            for target in (SWITCH_TARGET, ACTIVE_TARGET, POSTURE_TARGET)
        },
    }


def publish_report(
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
    that lag is the whole of what keeps the chip in ``switching…``. The row
    publishes a connector-host child for the same reason — the chip's wait is
    scoped to servers holding a connector, and a report bound at a generation
    with nothing serving would be passed over instead of waited on.
    """
    return write_server_report(
        root,
        os.getpid() if server_pid is None else server_pid,
        applied_target=applied_target,
        applied_generation=applied_generation,
        children=[4242],
        last_switch=last_switch,
        reachability=reachability_sweep(),
    )


def recorded_posture() -> dict[str, str]:
    """The narrowings on disk, as the connector and the hook will read them."""
    posture_store.invalidate_cache()
    return dict(posture_store.recorded_posture())


def popover_row(page: Page, target: str) -> Any:
    """The popover row for *target*.

    Always re-resolved from the page: the popover subtree is REPLACED on every
    render (a 5 s idle poll, 500 ms while a switch is out), so an element
    handle held across a wait would be detached by the time it is used.
    """
    return page.locator(f'{POPOVER} .ctc-row[data-target="{target}"]')


def writes_toggle(page: Page, target: str) -> Any:
    """The row's writes switch — ``aria-checked`` is the readout."""
    return popover_row(page, target).locator(".ctc-toggle")


def settled_chip(
    browser: Browser, base_url: str, *, init_script: str | None = None
) -> tuple[Page, str]:
    """Open the hub and wait until the chip speaks for a writable deployment.

    A visible chip already means the whole chain ran: the module mounted and
    ``GET /api/terminal/posture`` answered — the chip stays hidden until a read
    succeeds, and it no longer waits for a session to exist first.

    ``data-enforceable="true"`` is the second half, and it is a fact about the
    *deployment*: it is fed by ``contextWritable(state)``, which is false when
    the record has no location and false when another terminal owns it. Both
    conditions are arranged in :func:`chip_hub` before the server starts, so
    reaching this line proves the record is there and this terminal may write
    it — while it is false the popover locks every toggle.

    Args:
        browser: The chromium instance the suite runs against; one page is
            opened on it per call.
        base_url: The hub this page loads.
        init_script: An optional probe, seeded at document start — before
            ``goto`` — rather than evaluated once the page is settled. A probe
            that has to see the page's FIRST socket, fetch or frame cannot be
            installed after ``goto``: by then the module graph has already run
            and the thing it meant to count has already happened. Default
            ``None``, which is every call site that only wants a settled chip.

    Returns:
        (page, session_id) — the id the owning surface settled on: the
        terminal's confirmation in the Expert view, the console's key in the
        Simple one. Both views write it and this helper waits for it, so a
        missing id is a timeout here, never an empty return. Nothing the chip
        does needs it; test_posture_toggle_browser's case (b) uses it to name
        the PTY it asserts was not respawned.
    """
    page = browser.new_page()
    if init_script:
        page.add_init_script(init_script)
    page.goto(base_url, wait_until="domcontentloaded")

    expect(page.locator(CHIP)).to_be_visible(timeout=TIMEOUT)
    expect(page.locator(CHIP)).to_have_attribute("data-enforceable", "true", timeout=TIMEOUT)

    # The pointer is written by the surface that owns the session — the
    # terminal's own confirmation in the Expert view, the console's key in the
    # Simple one — and the chip settles on two local reads, ahead of either. A
    # single read here would be a read of whatever had happened by then.
    page.wait_for_function(
        "() => Boolean(localStorage.getItem('osprey-pty-session'))", timeout=TIMEOUT
    )

    session_id = page.evaluate("() => localStorage.getItem('osprey-pty-session')")
    return page, session_id


def open_popover(page: Page) -> None:
    """Click the chip open and wait for the card to be on screen."""
    page.locator(CHIP).click()
    expect(page.locator(POPOVER)).to_be_visible(timeout=TIMEOUT)
    expect(page.locator(CHIP)).to_have_attribute("aria-expanded", "true", timeout=TIMEOUT)
    expect(page.locator(CARD)).to_be_visible(timeout=TIMEOUT)
