"""Browser tests: the Channels panel beside a PV directory, end to end.

One real build of the demo preset with the simulator's section only: the
``pv_directory`` block names no ``type`` and replays the packaged alarm feed, so
on the ``va`` target the directory is the build's simulator view (its channels,
status and IOC health PVs) and the roster is the facility file, which lists the
channels and none of the simulator's own PVs. The simulator is served in
process, inside the value child the panel supervises: every current value on
these pages is a real read through that child.

Coverage, each in Chromium:

  (a) the view switch reads Curated · <directory label> · Alarms · Feedback; All PVs
      holds the toolbar search and result count and draws its filters in the
      shared sidebar, and no ``nav`` header contribution is sent.
  (b) a word search lists curated (◆) and control-system-only (◇) rows, the
      IOC health PVs among the ◇ ones, with current values read.
  (c) a filter narrows the rows, and the result count follows it; the address
      bar carries it, so Back from a PV page restores it.
  (d) a PV page shows its sections in order, the live alarm state (severity,
      latched, acknowledged) and the current value; no alarm or put history,
      since the section configures neither; the logbook is not available on
      this target.
  (e) ``#/pv/<name>`` and ``#/ioc/<name>`` open their pages; the IOC page says
      alive once its heartbeat advances, and links to the IOC's PVs.
  (f) without the section the switch is Explore · Feedback and ``#/pv/<name>``
      shows Explore.

Run:
    .venv/bin/pytest tests/interfaces/channel_finder/test_all_pvs_browser.py -m browser -v
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import pytest

pytest.importorskip("osprey_connectors.service_answers")

import yaml
from click.testing import CliRunner

from osprey.interfaces._serving import run_app_server
from osprey.runtime.session_env import FORBIDDEN_ENV
from osprey.services.pv_directory import _connectors
from osprey_connectors import posture_store
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from tests._builds import init_project
from tests._control_system import IN_PROCESS
from tests.interfaces._panel_host import PanelHost, open_panel_host
from tests.interfaces.conftest import use_process_web_credentials

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from playwright.sync_api import Browser

pytestmark = [pytest.mark.browser, pytest.mark.slow]

VIEWPORT = {"width": 1280, "height": 800}

#: The demo's simulator-only section: no ``type``, the packaged alarm replay.
SECTION = {"alarm_state": {"replay_file": "demo:example-alarms"}}

#: A curated channel the replay holds latched and unacknowledged.
LATCHED_PV = "SR:RF:CAVITY:01:TEMPERATURE:RB"
#: The storage ring's simulation IOC, and the IOC health PVs it serves.
IOC = "sim-SR"
HEALTH_PVS = ("ca:SIM:SR:ENGINEER", "ca:SIM:SR:HEARTBEAT", "ca:SIM:SR:UPTIME")

ROWS = ".allpvs-table tbody tr[data-name]"
#: A word in the descriptions of a few curated channels and of every UPTIME
#: IOC health PV, so one page holds both origins.
WORD = "since"
#: The IOC page's second HEARTBEAT read comes 2 s after the first.
ALIVE_TIMEOUT_MS = 30_000


def _build(parent: Path) -> Path:
    """The demo repo built with :data:`SECTION` and the simulator served in process."""
    from osprey.cli.build_cmd import build

    repo = init_project(parent, "demo", "allpvs")
    profile_file = repo / "profile.yml"
    profile = yaml.safe_load(profile_file.read_text(encoding="utf-8"))
    profile["config"]["pv_directory"] = SECTION
    profile["config"]["control_system.connector.virtual_accelerator.serving"] = IN_PROCESS
    profile_file.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    result = CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert result.exit_code == 0, result.output
    return repo / "build"


@pytest.fixture(scope="module")
def render(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The render root of the built repo; its ``config.yml`` carries the section."""
    render = _build(tmp_path_factory.mktemp("all_pvs"))
    config = yaml.safe_load((render / "config.yml").read_text(encoding="utf-8"))
    assert config["pv_directory"] == SECTION
    assert _connectors.has_simulator(config)
    off = {key: value for key, value in config.items() if key != "pv_directory"}
    (render / "config-off.yml").write_text(yaml.safe_dump(off, sort_keys=False), encoding="utf-8")
    return render


@contextmanager
def _serve(render: Path, config_name: str, state_dir: Path) -> Iterator[tuple[str, Any]]:
    """The Channel Finder over ``render/config_name``, with the value child's inherited env."""
    from osprey.interfaces.channel_finder import app as app_module
    from tests.interfaces.channel_finder.graph_fixture import demo_context

    with pytest.MonkeyPatch.context() as patch:
        config_file = render / config_name
        patch.chdir(render)
        patch.setenv("OSPREY_CONFIG", str(config_file))
        patch.setenv("CONFIG_FILE", str(config_file))
        state_dir.mkdir(parents=True, exist_ok=True)
        patch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(state_dir))
        patch.setenv(AUDIT_IDENTITY_ENV, "test-operator")
        patch.delenv(TERMINAL_USER_ENV, raising=False)
        for name in list(os.environ):
            if name in FORBIDDEN_ENV or name.startswith("OSPREY_CONTROL_TARGET"):
                patch.delenv(name)
        # The graph store is not dialled: the curated tab reads the demo corpus.
        patch.setattr(app_module, "_make_graph_context", demo_context)
        app = app_module.create_app(project_cwd=str(render))
        with run_app_server(app) as base_url:
            yield base_url, app


@pytest.fixture(scope="module")
def panel_server(
    render: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[str, Any]]:
    """The panel beside the section, its value child reading the in-process simulator.

    Module-scoped: one value child, and one simulator, for every test.
    """
    with _serve(render, "config.yml", tmp_path_factory.mktemp("control_target")) as served:
        yield served


@pytest.fixture()
def panel_url(panel_server: tuple[str, Any]) -> str:
    """The shared panel's URL, its gate on this test's credentials."""
    url, app = panel_server
    use_process_web_credentials(app)
    return url


@pytest.fixture()
def bare_panel_url(render: Path, tmp_path: Path) -> Iterator[str]:
    """The same panel with no ``pv_directory`` section."""
    with _serve(render, "config-off.yml", tmp_path / "control_target") as (url, _app):
        yield url


def _open(browser: Browser, url: str, route: str) -> PanelHost:
    page = browser.new_page(viewport=VIEWPORT)
    host = open_panel_host(page, f"{url}/{route}")
    host.wait_for_selector("#cf-views button[data-view]")
    return host


def _tabs(host: PanelHost) -> list[str]:
    """The view switch's labels, without the Alarms tab's count pill."""
    return host.eval_on_selector_all(
        "#cf-views button[data-view]",
        """buttons => buttons.map(b => [...b.childNodes]
            .filter(n => n.nodeType === Node.TEXT_NODE)
            .map(n => n.textContent).join('').trim())""",
    )


def _pressed(host: PanelHost) -> str | None:
    return host.locator('#cf-views button[aria-pressed="true"]').get_attribute("data-view")


def _rows(host: PanelHost) -> list[dict[str, Any]]:
    return host.eval_on_selector_all(
        ROWS,
        """rows => rows.map(r => ({
            name: r.dataset.name,
            mark: r.querySelector('.mark').textContent,
            value: r.querySelector('.allpvs-value').textContent,
        }))""",
    )


def _count(host: PanelHost) -> int:
    text = host.locator("#cf-result-count strong").inner_text()
    return int(text.replace(",", ""))


def _search(host: PanelHost, words: str) -> None:
    host.locator("#cf-search").fill(words)
    host.wait_for_function(
        "words => document.querySelector('.allpvs-searched')?.textContent.includes(words)"
        " && document.querySelector('.allpvs-table tbody tr[data-name]') !== null",
        arg=words,
    )


def _search_no_match(host: PanelHost, words: str) -> None:
    host.locator("#cf-search").fill(words)
    state = host.locator(".allpvs-body .osprey-state[role=status] .osprey-state-message")
    state.filter(has_text="No match").wait_for()
    assert host.locator(ROWS).count() == 0
    assert host.locator("#cf-result-count").is_hidden()


def _values_read(host: PanelHost) -> None:
    host.wait_for_function(
        "() => [...document.querySelectorAll('.allpvs-table .allpvs-value')]"
        ".every(cell => cell.textContent !== '…')"
    )


def _sections(host: PanelHost) -> list[str]:
    return host.eval_on_selector_all(
        ".pv-page .pvp-section", "els => els.map(e => e.dataset.section)"
    )


def _facts(host: PanelHost, section: str) -> dict[str, str]:
    return host.eval_on_selector(
        f'.pvp-section[data-section="{section}"] .pvp-facts',
        """list => Object.fromEntries([...list.querySelectorAll('dt')].map(
            dt => [dt.textContent, dt.nextElementSibling.textContent]))""",
    )


def test_all_pvs_holds_the_toolbar_and_the_shared_sidebar(
    chromium_browser: Browser, panel_url: str
):
    host = _open(chromium_browser, panel_url, "#/allpvs")
    host.wait_for_selector(ROWS)
    assert _tabs(host) == ["Curated", "All PVs", "Alarms", "Feedback"]
    assert _pressed(host) == "allpvs"

    assert host.locator("#cf-search").is_enabled()
    assert host.locator("#cf-result-count").is_visible()
    assert _count(host) > 0
    sidebar = host.locator(".allpvs .osprey-panel-sidebar.allpvs-facets")
    sidebar.locator('[data-facet="origin"]').first.wait_for()
    assert host.locator("#cf-sidebar-toggle").is_enabled()

    assert not [
        item
        for contribution in host.contributions()
        for item in contribution.get("items", [])
        if item.get("kind") == "nav"
    ]


def test_a_word_search_lists_curated_and_control_system_only_rows(
    chromium_browser: Browser, panel_url: str
):
    host = _open(chromium_browser, panel_url, "#/allpvs")
    _search(host, WORD)
    _values_read(host)
    rows = {row["name"]: row for row in _rows(host)}
    assert _count(host) == len(rows)
    health = {name for name in rows if name.startswith("ca:SIM:")}
    assert {f"ca:SIM:{ioc}:UPTIME" for ioc in ("BTS", "LINE", "LTB", "SR", "texture")} == health
    for name, row in rows.items():
        assert row["mark"] == ("◇" if name in health else "◆"), row
        assert row["value"] not in ("…", "—", "no answer"), row

    _search_no_match(host, "zzzz-no-such-word")


def test_a_filter_narrows_the_rows_and_the_count_follows(chromium_browser: Browser, panel_url: str):
    host = _open(chromium_browser, panel_url, "#/allpvs")
    _search(host, WORD)
    total = _count(host)

    facet = host.locator('.allpvs-facet[data-facet="origin"][data-value="directory"]')
    expected = int(facet.locator(".allpvs-facet-n").inner_text())
    assert 0 < expected < total
    facet.click()
    host.wait_for_function(
        "n => document.querySelector('#cf-result-count strong')?.textContent === String(n)"
        " && document.querySelectorAll('.allpvs-table tbody tr[data-name]').length === n",
        arg=expected,
    )
    assert facet.get_attribute("aria-pressed") == "true"
    rows = _rows(host)
    assert len(rows) == expected == _count(host)
    assert {row["mark"] for row in rows} == {"◇"}


def test_back_from_a_pv_page_restores_the_filter(chromium_browser: Browser, panel_url: str):
    host = _open(chromium_browser, panel_url, "#/allpvs")
    _search(host, WORD)
    facet = host.locator('.allpvs-facet[data-facet="origin"][data-value="directory"]')
    expected = int(facet.locator(".allpvs-facet-n").inner_text())
    facet.click()
    host.wait_for_function("() => location.hash.includes('origin=directory')")
    assert f"q={WORD}" in host.evaluate("location.hash")
    host.wait_for_function(
        "n => document.querySelectorAll('.allpvs-table tbody tr[data-name]').length === n",
        arg=expected,
    )

    host.locator(f"{ROWS} td.name a").first.click()
    host.wait_for_selector(".pv-page .pvp-name")
    # Back inside the frame: the host page sees no navigation of its own to wait for.
    host.evaluate("history.back()")
    host.wait_for_selector(
        '.allpvs-facet[data-facet="origin"][data-value="directory"][aria-pressed="true"]'
    )
    host.wait_for_function(
        "n => document.querySelector('#cf-result-count strong')?.textContent === String(n)",
        arg=expected,
    )


def test_a_pv_page_shows_its_sections_alarm_state_and_value(
    chromium_browser: Browser, panel_url: str
):
    host = _open(chromium_browser, panel_url, f"#/pv/{quote(LATCHED_PV, safe='')}")
    host.wait_for_selector(".pv-page .pvp-name")
    assert host.locator(".pvp-name").inner_text() == LATCHED_PV
    assert host.locator(".pvp-mark").inner_text() == "◆"
    assert _pressed(host) == "allpvs"

    sections = _sections(host)
    assert sections == ["directory", "alarm_state", "alarm_config", "logbook", "value"]
    assert "alarm_history" not in sections and "put_history" not in sections

    alarm = _facts(host, "alarm_state")
    assert alarm["Alarm server"].startswith("MAJOR"), alarm
    assert alarm["Latched"] == "yes", alarm
    assert alarm["Acknowledged"] == "no", alarm

    host.wait_for_function(
        """() => document.querySelector('.pvp-section[data-section="value"] dt')
                  ?.textContent === 'Value'"""
    )
    value = _facts(host, "value")
    assert value["Value"] != "", value
    float(value["Value"])

    logbook = host.locator('.pvp-section[data-section="logbook"] [data-slot="logbook"]')
    host.wait_for_function(
        """() => document.querySelector('.pvp-section[data-section="logbook"] [data-slot="logbook"]')
                  ?.textContent !== 'Reading…'"""
    )
    assert logbook.inner_text().lower() == "not available for this target"


def test_deep_links_open_the_pv_and_ioc_pages(chromium_browser: Browser, panel_url: str):
    host = _open(chromium_browser, panel_url, f"#/ioc/{quote(IOC, safe='')}")
    host.wait_for_selector(".cf-ioc-page .cf-ioc-name")
    assert host.locator(".cf-ioc-name").inner_text() == IOC
    pvs = host.eval_on_selector_all(
        ".cf-ioc-health .cf-ioc-pv", "els => els.map(e => e.textContent)"
    )
    assert set(HEALTH_PVS) <= set(pvs), pvs
    alive = host.locator('.cf-ioc-head .osprey-chip[data-role="alive"]')
    host.wait_for_selector('.cf-ioc-head .osprey-chip[data-alive="yes"]', timeout=ALIVE_TIMEOUT_MS)
    assert alive.inner_text() == "alive: yes"
    link = host.locator(".cf-ioc-link")
    assert link.get_attribute("href") == f"#/allpvs?ioc={quote(IOC, safe='')}"

    link.click()
    host.wait_for_selector(ROWS)
    assert (
        host.locator(f'.allpvs-facet[data-facet="ioc"][data-value="{IOC}"]').get_attribute(
            "aria-pressed"
        )
        == "true"
    )

    host.goto(f"{panel_url}/#/pv/{quote('ca:SIM:SR:HEARTBEAT', safe='')}")
    host.wait_for_selector(".pv-page .pvp-name")
    assert host.locator(".pvp-name").inner_text() == "ca:SIM:SR:HEARTBEAT"
    assert host.locator(".pvp-mark").inner_text() == "◇"
    assert host.locator(".pvp-ioc").inner_text() == IOC


def test_without_the_section_the_tabs_are_todays_and_a_pv_link_shows_explore(
    chromium_browser: Browser, bare_panel_url: str
):
    host = _open(chromium_browser, bare_panel_url, f"#/pv/{quote(LATCHED_PV, safe='')}")
    host.wait_for_function(
        "() => document.querySelector('#cf-views button[aria-pressed=\"true\"]')"
        "?.dataset.view === 'explore'"
    )
    assert _tabs(host) == ["Explore", "Feedback"]
    host.wait_for_selector("#graph-finder-table")
    assert host.locator(".pv-page").count() == 0
