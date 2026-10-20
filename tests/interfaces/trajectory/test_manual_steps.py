"""Per-step browser tests for the trajectory panel, keyed by the manual's steps.

The operator's manual walks the panel in eighteen counted steps, and the
acceptance inventory gives every control a step touches one stable
``data-control`` name. :data:`MANUAL_STEPS` is that map, one entry per counted
step; it is the one tuple the frozen per-step tests and the deployed replay in
``test_stack_regression.py`` are keyed by.

What only a real browser through the stack proves here: every inventory
control is on the page under its name and keeps it across a re-render; the
panel embedded as a hub tile in Expert mode is driven from the tile bar (the
session buttons and the centre tabs) and still hands a reference file to the
browser as a download; in Simple mode the session opens from the in-body
strip; Run Selected draws the three raw plots; a drawn plot follows a theme
change; and Apply CM step is enabled exactly when the step the panel shows is
the one the service would apply.

Every page runs through :mod:`tests.interfaces.trajectory._stack`: the fake
machine answers for the panel's behaviour, never for physics. Skips cleanly
when the chromium headless binary is not installed.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.interfaces.trajectory._stack import (
    SETTLE_MS,
    Stack,
    run_stack,
    tile,
)

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

if TYPE_CHECKING:
    from playwright.sync_api import Browser, FrameLocator, Locator, Page

pytestmark = [pytest.mark.browser, pytest.mark.slow]


@dataclass(frozen=True)
class ManualStep:
    """One counted step of the manual and the controls it touches.

    Attributes:
        number: The manual's step number, 1 to 18.
        acceptance: The acceptance item the step is checked as.
        title: The step in the manual's words.
        controls: The ``data-control`` names the step touches, in the order the
            manual uses them.
    """

    number: int
    acceptance: str
    title: str
    controls: tuple[str, ...]

    @property
    def id(self) -> str:
        """The step's stable test id."""
        return f"manual-{self.number:02d}"


#: The session controls: used before step 1, and counted by no step.
SESSION_CONTROLS = ("session-open", "session-takeover", "session-end")

#: The plot ranges, checked inside steps 1 and 5 (they have no step of their own).
RANGES = ("range-turns", "range-sum", "range-x", "range-y")

#: The three raw plots, the trajectories, and the difference to the reference.
RAW_PLOTS = ("plot-raw-sum", "plot-raw-x", "plot-raw-y")
TRAJECTORY_PLOTS = ("plot-traj-x", "plot-traj-y")
DIFFERENCE_PLOTS = ("plot-diff-x", "plot-diff-y")

#: The eighteen counted steps of the manual, each with its acceptance item.
MANUAL_STEPS: tuple[ManualStep, ...] = (
    ManualStep(
        1,
        "A1",
        "Set up BPMs",
        (
            "bpm-family",
            "bpm-exclude",
            "setting-turns-tracked",
            "setting-attenuation",
            "setting-bpm-mode",
            "setting-pilot-tone",
            "setup-bpms",
            "tab-raw",
            *RANGES,
        ),
    ),
    ManualStep(2, "A2", "Open the BPM control panel", ("open-bpm-control",)),
    ManualStep(
        3,
        "A3",
        "BPM control panel settings",
        (
            "setting-pilot-tone",
            "setting-bpm-mode",
            "stored-current",
            "stored-current-readback",
            "plot-bpm-atten",
            "plot-bpm-pt-state",
        ),
    ),
    ManualStep(
        4,
        "A4",
        "Select the injection chain's steps",
        (
            "chain-readout-mode",
            "chain-n-shots",
            "chain-n-turns",
            "step-arm",
            "step-arm-run",
            "step-inject",
            "step-inject-run",
            "step-inject-options",
            "step-clean",
            "step-clean-run",
            "step-clean-options",
            "step-readout",
            "step-readout-run",
            "step-readout-options",
            "step-plot",
            "step-plot-run",
            "step-plot-options",
        ),
    ),
    ManualStep(
        5, "A5", "Run Selected and observe", ("run-selected", "tab-raw", *RAW_PLOTS, *RANGES)
    ),
    ManualStep(
        6, "A7", "Record injection shots", ("record-toggle", "records-save", "records-flush")
    ),
    ManualStep(7, "A8", "Continuous mode", ("chain-continuous", *RAW_PLOTS)),
    ManualStep(
        8,
        "A9",
        "One-turn trajectory",
        ("tab-correction", "solve-loss-cutoff", *TRAJECTORY_PLOTS, *DIFFERENCE_PLOTS),
    ),
    ManualStep(
        9,
        "A10",
        "Set the reference",
        ("ref-select-current", "ref-save-file", "ref-load-file", *DIFFERENCE_PLOTS),
    ),
    ManualStep(10, "A11", "Shot-to-shot variation", ("run-selected", *DIFFERENCE_PLOTS)),
    ManualStep(11, "A12", "Correction demo: a kicked corrector", DIFFERENCE_PLOTS),
    ManualStep(12, "A13", "Set up CMs", ("cm-family", "cm-exclude", "setup-cms")),
    ManualStep(13, "A14", "Calculate the response matrix", ("rm-calculate", "rm-status")),
    ManualStep(
        14,
        "A15",
        "Pseudo-inverse",
        ("solve-alpha", "solve-n-discard", "solve-gain", "solve-calculate", "plot-sv"),
    ),
    ManualStep(15, "A16", "Calculate the CM step", ("step-calculate", "step-plot-bars")),
    ManualStep(16, "A16", "Plot the CM step", ("step-plot-bars", "plot-cm-step")),
    ManualStep(17, "A17", "Apply the CM step", ("step-apply", "apply-result", "put-back")),
    ManualStep(18, "A18", "Correction result", ("run-selected", *DIFFERENCE_PLOTS)),
)

#: Every control of the acceptance inventory, in its order, then the names the
#: plan-stage ruling adds (the centre plots). Written out on its own so the
#: steps above are checked against it, not derived from them.
INVENTORY: tuple[str, ...] = (
    *SESSION_CONTROLS,
    "bpm-family",
    "bpm-exclude",
    "setting-turns-tracked",
    "setting-attenuation",
    "setting-bpm-mode",
    "setting-pilot-tone",
    "setup-bpms",
    "open-bpm-control",
    "stored-current",
    "stored-current-readback",
    "plot-bpm-atten",
    "plot-bpm-pt-state",
    "chain-readout-mode",
    "chain-n-shots",
    "chain-n-turns",
    "step-arm",
    "step-arm-run",
    "step-inject",
    "step-inject-run",
    "step-inject-options",
    "step-clean",
    "step-clean-run",
    "step-clean-options",
    "step-readout",
    "step-readout-run",
    "step-readout-options",
    "step-plot",
    "step-plot-run",
    "step-plot-options",
    "run-selected",
    "tab-raw",
    *RANGES,
    "record-toggle",
    "records-save",
    "records-flush",
    "chain-continuous",
    "tab-correction",
    "solve-loss-cutoff",
    "ref-select-current",
    "ref-save-file",
    "ref-load-file",
    "cm-family",
    "cm-exclude",
    "setup-cms",
    "rm-calculate",
    "rm-status",
    "solve-alpha",
    "solve-n-discard",
    "solve-gain",
    "solve-calculate",
    "plot-sv",
    "step-calculate",
    "step-plot-bars",
    "plot-cm-step",
    "step-apply",
    "apply-result",
    "put-back",
    *RAW_PLOTS,
    *TRAJECTORY_PLOTS,
    *DIFFERENCE_PLOTS,
)

#: Controls that exist only once their action ran: the apply-result table is
#: the outcome of an Apply, and the Apply test asserts it.
AFTER_AN_ACTION = frozenset({"apply-result"})

#: The reasons Apply CM step gives while disabled, in the operator's words.
NO_STEP = "Calculate CM step first"
STEP_CHANGED = "step changed: Calculate CM step again"

#: The hub tile bar's contributions, by the key the hub gives each one.
CENTRE_TABS_NAV = '[data-contrib-key="nav:centre-tabs"]'


@pytest.fixture
def stack(
    chromium_browser: Browser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Stack]:
    with run_stack(chromium_browser, tmp_path, monkeypatch) as served:
        yield served


# -- helpers --------------------------------------------------------------------


def _named(page: Page | FrameLocator, name: str) -> Locator:
    """Every element carrying the ``data-control`` *name*."""
    return page.locator(f'[data-control="{name}"]')


def _one(page: Page | FrameLocator, name: str) -> Locator:
    """The control *name* where the operator uses it: a setting and the
    session buttons are in the body, never the drawer's or the tile bar's
    copy."""
    if name.startswith("session-"):
        return page.locator(f'[data-view=session] [data-control="{name}"]')
    if name.startswith("setting-"):
        return page.locator(f'[data-area=setup] [data-control="{name}"]')
    return _named(page, name)


def _hub_button(page: Page, name: str) -> Locator:
    """The tile bar's copy of the panel's *name* button, in the hub page."""
    return page.locator(f'[data-contrib-key="button:{name}"]')


def _open_session(page: Page | FrameLocator) -> None:
    _one(page, "session-open").click()
    expect(_one(page, "session-end")).to_be_enabled(timeout=SETTLE_MS)


def _set_up_bpms(page: Page | FrameLocator) -> None:
    """Commit the BPM setup, which every shot needs first."""
    expect(_one(page, "setup-bpms")).to_be_enabled(timeout=SETTLE_MS)
    _one(page, "setup-bpms").click()
    expect(_one(page, "run-selected")).to_be_enabled(timeout=SETTLE_MS)


def _drawn(page: Page | FrameLocator, plot_id: str) -> Locator:
    """The plot *plot_id*'s drawn SVG once Plotly drew it."""
    return page.locator(f"#{plot_id} .main-svg").first


def _traces(page: Page | FrameLocator, plot_id: str) -> int:
    return page.locator(f"#{plot_id}").evaluate("e => (e.data || []).length")


def _paper(page: Page | FrameLocator, plot_id: str) -> str:
    return page.locator(f"#{plot_id}").evaluate(
        "e => (e._fullLayout && e._fullLayout.paper_bgcolor) || ''"
    )


def _run_selected(page: Page, panel: Page | FrameLocator, n_bpms: int = 3) -> str:
    """Run the ticked chain from *panel* and wait until its shot's three raw
    plots drew one trace per BPM; answers the shot's id. *page* is the page
    the panel's requests are seen on (the hub page for a tile)."""
    with page.expect_response(
        lambda r: r.url.endswith("/api/shot") and r.request.method == "POST",
        timeout=SETTLE_MS,
    ) as shot:
        _one(panel, "run-selected").click()
    assert shot.value.ok, shot.value.text()
    shot_id = shot.value.json()["shot"]["shot_id"]
    for plot_id in RAW_PLOTS:
        expect(_drawn(panel, plot_id)).to_be_attached(timeout=SETTLE_MS)
        _wait_traces(panel, plot_id, n_bpms)
    return shot_id


def _wait_traces(panel: Page | FrameLocator, plot_id: str, n: int) -> None:
    """Wait until the plot *plot_id* holds *n* traces."""
    host = panel.locator(f"#{plot_id}")
    host.evaluate(
        """(e, [n, ms]) => new Promise((resolve, reject) => {
             const until = Date.now() + ms;
             const poll = () => {
               if ((e.data || []).length === n) return resolve(true);
               if (Date.now() > until) return reject(new Error(`${e.id}: ${(e.data || []).length} traces`));
               setTimeout(poll, 50);
             };
             poll();
           })""",
        [n, SETTLE_MS],
    )


# -- the scaffold: one test per counted manual step ------------------------------


def test_the_steps_are_the_manuals_eighteen_over_the_whole_inventory() -> None:
    """The tuple is the gate's count, steps 1 to 18 each once; every name a
    step touches is an inventory name, and every inventory name but the
    session's is touched by some step."""
    assert [step.number for step in MANUAL_STEPS] == list(range(1, 19))
    assert len({step.id for step in MANUAL_STEPS}) == 18
    assert len(set(INVENTORY)) == len(INVENTORY), "an inventory name is listed twice"
    touched = {name for step in MANUAL_STEPS for name in step.controls}
    assert sorted(touched - set(INVENTORY)) == []
    assert sorted(set(INVENTORY) - touched) == sorted(SESSION_CONTROLS)


@pytest.mark.parametrize("step", MANUAL_STEPS, ids=lambda step: step.id)
def test_each_step_finds_its_controls_by_name(stack: Stack, step: ManualStep) -> None:
    page = stack.open("alice")
    for name in step.controls:
        if name in AFTER_AN_ACTION:
            continue
        expect(_named(page, name).first).to_be_attached(timeout=SETTLE_MS)
    stack.assert_loaded_clean()


# -- every inventory control, by its stable name -----------------------------------


def _counts(page: Page | FrameLocator, names: tuple[str, ...]) -> dict[str, int]:
    return {name: _named(page, name).count() for name in names}


def test_every_inventory_control_exists_with_its_stable_name(stack: Stack) -> None:
    """Each inventory name is on the page, and a session, a setup and a shot
    re-render the panel without renaming, dropping or doubling any of them."""
    page = stack.open("alice")
    names = tuple(name for name in INVENTORY if name not in AFTER_AN_ACTION)
    for name in names:
        expect(_named(page, name).first).to_be_attached(timeout=SETTLE_MS)
    before = _counts(page, names)
    assert [name for name, count in before.items() if count == 0] == []

    _open_session(page)
    _set_up_bpms(page)
    _run_selected(page, page)

    assert _counts(page, names) == before
    stack.assert_loaded_clean()


# -- embedded: Expert drives the tile bar, Simple keeps the in-body strip ----------


def test_an_expert_tile_is_driven_from_the_tile_bar_and_saves_a_reference_file(
    stack: Stack,
) -> None:
    page = stack.open_tile("alice", mode="expert")
    panel = tile(page)
    # The in-body session strip and centre tabs are handed to the tile bar.
    expect(panel.locator("[data-view=session]")).to_be_hidden(timeout=SETTLE_MS)
    expect(panel.locator("nav.centre-tabs")).to_be_hidden()
    tabs = page.locator(CENTRE_TABS_NAV)
    expect(tabs.locator(".contrib-nav-link")).to_have_count(2, timeout=SETTLE_MS)

    expect(_hub_button(page, "session-open")).to_be_enabled(timeout=SETTLE_MS)
    _hub_button(page, "session-open").click()
    expect(_hub_button(page, "session-end")).to_be_enabled(timeout=SETTLE_MS)
    expect(_hub_button(page, "session-open")).to_be_disabled()

    # Manual steps 8 to 10: the Trajectory Correction tab, then a shot on it.
    tabs.locator(".contrib-nav-link", has_text="Trajectory Correction").click()
    expect(panel.locator('[data-tab-panel="correction"]').first).to_be_visible(timeout=SETTLE_MS)
    expect(panel.locator('[data-tab-panel="raw"]').first).to_be_hidden()
    _set_up_bpms(panel)
    _run_selected(page, panel)
    for plot_id in (*TRAJECTORY_PLOTS, *DIFFERENCE_PLOTS):
        expect(_drawn(panel, plot_id)).to_be_visible(timeout=SETTLE_MS)

    _named(panel, "reference-name").fill("golden")
    _one(panel, "ref-select-current").click()
    expect(panel.locator("[data-role=reference-current]")).to_contain_text(
        "golden", timeout=SETTLE_MS
    )
    expect(_one(panel, "ref-save-file")).to_be_enabled(timeout=SETTLE_MS)
    with page.expect_download(timeout=SETTLE_MS) as saved:
        _one(panel, "ref-save-file").click()
    download = saved.value
    assert download.failure() is None, download.failure()
    assert download.suggested_filename.endswith(".json"), download.suggested_filename
    assert "golden" in Path(download.path()).read_text(encoding="utf-8")

    tabs.locator(".contrib-nav-link", has_text="Raw BPM Signal").click()
    expect(panel.locator('[data-tab-panel="raw"]').first).to_be_visible(timeout=SETTLE_MS)
    stack.assert_loaded_clean()


def test_a_simple_tile_opens_the_session_from_the_in_body_strip(stack: Stack) -> None:
    page = stack.open_tile("alice", mode="simple")
    panel = tile(page)
    # Simple mode keeps the session strip and the centre tabs in the body.
    expect(panel.locator("[data-view=session]")).to_be_visible(timeout=SETTLE_MS)
    expect(panel.locator("nav.centre-tabs")).to_be_visible()

    _one(panel, "session-open").click()

    expect(_one(panel, "session-end")).to_be_enabled(timeout=SETTLE_MS)
    expect(_one(panel, "session-open")).to_be_disabled()
    expect(panel.locator("[data-fact=holder] .fact-value")).to_have_text(
        "panel of alice (this tab)", timeout=SETTLE_MS
    )
    stack.assert_loaded_clean()


# -- Run Selected draws the raw plots, and a theme change redraws them -------------


def test_run_selected_draws_the_three_raw_plots(stack: Stack) -> None:
    page = stack.open("alice")
    _open_session(page)
    _set_up_bpms(page)
    expect(_one(page, "step-plot")).to_be_checked()

    _run_selected(page, page)

    for plot_id in RAW_PLOTS:
        size = _drawn(page, plot_id).evaluate(
            "e => { const r = e.getBoundingClientRect(); return [r.width, r.height]; }"
        )
        assert size[0] > 0 and size[1] > 0, f"{plot_id} drew at size {size}"
        assert _traces(page, plot_id) == len(stack.machine.devices), plot_id
    stack.assert_loaded_clean()


def test_a_drawn_plot_follows_the_hub_theme(stack: Stack) -> None:
    page = stack.open_tile("alice", theme="dark", mode="expert")
    panel = tile(page)
    _hub_button(page, "session-open").click()
    expect(_hub_button(page, "session-end")).to_be_enabled(timeout=SETTLE_MS)
    _set_up_bpms(panel)
    _run_selected(page, panel)
    expect(panel.locator("html")).to_have_attribute("data-theme", "dark")
    dark = {plot_id: _paper(panel, plot_id) for plot_id in RAW_PLOTS}
    assert all(dark.values()), dark

    page.locator("osprey-display-menu .display-menu-trigger").click()
    page.locator('osprey-display-menu .display-seg-option[data-appearance="light"]').click()

    expect(page.locator("html")).to_have_attribute("data-theme", "light", timeout=SETTLE_MS)
    expect(panel.locator("html")).to_have_attribute("data-theme", "light", timeout=SETTLE_MS)
    for plot_id in RAW_PLOTS:
        panel.locator(f"#{plot_id}").evaluate(
            """(e, [dark, ms]) => new Promise((resolve, reject) => {
                 const until = Date.now() + ms;
                 const poll = () => {
                   if (e._fullLayout && e._fullLayout.paper_bgcolor !== dark) return resolve(true);
                   if (Date.now() > until) return reject(new Error(`${e.id} kept ${dark}`));
                   setTimeout(poll, 50);
                 };
                 poll();
               })""",
            [dark[plot_id], SETTLE_MS],
        )
        assert _traces(panel, plot_id) == len(stack.machine.devices), plot_id
    stack.assert_loaded_clean()


# -- Apply CM step: enabled only for the step the service would apply --------------


def _apply_reason(page: Page) -> str | None:
    return _one(page, "step-apply").get_attribute("data-reason")


def test_apply_is_enabled_only_for_the_step_the_panel_shows(stack: Stack) -> None:
    page = stack.open("alice")
    _open_session(page)
    apply = _one(page, "step-apply")
    expect(apply).to_be_disabled()
    expect(apply).to_have_attribute("title", NO_STEP, timeout=SETTLE_MS)

    _set_up_bpms(page)
    _run_selected(page, page)
    _one(page, "tab-correction").click()
    _one(page, "step-calculate").click()
    expect(apply).to_be_enabled(timeout=SETTLE_MS)
    assert _apply_reason(page) is None

    # A fresh shot moves the service's pending step on: the one shown is stale.
    _run_selected(page, page)
    expect(apply).to_be_disabled(timeout=SETTLE_MS)
    expect(apply).to_have_attribute("title", STEP_CHANGED, timeout=SETTLE_MS)

    _one(page, "step-calculate").click()
    expect(apply).to_be_enabled(timeout=SETTLE_MS)
    writes = len(stack.machine.writes)
    with page.expect_response(
        lambda r: r.url.endswith("/api/apply") and r.request.method == "POST"
    ) as applied:
        apply.click()
    assert applied.value.ok, applied.value.text()
    result = _named(page, "apply-result")
    expect(result).to_be_visible(timeout=SETTLE_MS)
    expect(result.locator("tr[data-moved]").first).to_be_attached(timeout=SETTLE_MS)
    assert len(stack.machine.writes) > writes, "the apply wrote nothing"
    stack.assert_loaded_clean()
