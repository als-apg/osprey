"""Live: the server fleet end to end over the served demo.

The demo is built, its simulator view served by the model runner on Channel
Access (the seam's harness, ``tests/va/test_lume_pva_seam.py``, under its
``fleet`` port), and the deployment's own processes run against one rendered
config pointed at it: the server fleet service (``python -m
osprey.services.channel_directory.fleet``, which spawns the refresher of ``va``),
several Channels panel backends (the IOCs view's route and the IOC page) and
the channel_directory MCP server calling ``server_fleet``. Every reader reaches the
service through the fleet connection's resolver, from ``services.server_fleet.port``.

What it proves is the proposal's demo criteria:

* two refresh periods after the service starts, every simulated IOC is
  ``alive`` with a host and a state age;
* the demo's stop scenario makes its IOC ``silent`` and the first row in the
  view and the tool within two periods, and ``alive`` again once removed;
* the tool's rows and counts are the view's, and the IOC page's fleet fact is
  the view's row, one snapshot apart from none;
* the refresher reads as many channels and queries the directory as often per
  period with several panel backends and many view and tool reads as with no
  reader at all.

The refresher's side is read from what it writes -- its snapshot and memory
files and its one audit record per cycle -- so watching it adds no reader.
Run by the live-CA gate (``scripts/va/live_ca/gate.py``), which fails it on
any skip and gives it a longer wall-clock limit than other suites.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors_tests._harness._ports import reserved_ports
from tests._builds import BuiltProject, copy_build
from tests.va import _fleet_e2e as e2e
from tests.va._collection import assert_module_collected
from tests.va.test_lume_pva_seam import (  # the seam's harness and its held CA port
    CA_PORTS,
    _served,
    _Server,
)

#: Floor for this module's own test count.
MIN_COLLECTED_TESTS = 5

#: The demo's stop-an-IOC scenario and the IOC it stops.
STOP_SCENARIO = "bts-server-stopped"
STOPPED_SERVER = "sim-BTS"

#: How many panel backends and tool readers read during the loaded window.
PANELS = 3
TOOL_READERS = 2
#: Threads reading the IOCs view of each panel backend during the loaded window.
VIEW_THREADS_PER_PANEL = 2
#: Consecutive cycles in each of the quiet and the loaded window.
WINDOW_CYCLES = 3
#: How far one cycle's start may sit from one period after the previous one's:
#: the refresher sleeps to the tick, so only scheduling jitter separates them.
TICK_JITTER_S = 1.0
#: The fewest view reads and tool calls the loaded window must have made: many,
#: so a reader that touched the control system could not hide in the counts.
MIN_VIEW_READS = 100
MIN_TOOL_CALLS = 20

#: How long a held CA client may take to find a restarted IOC's channel again:
#: it searches for a lost channel with a growing backoff
#: (``test_stopped_servers_live.RESTART_TIMEOUT_S``).
RESTART_TIMEOUT_S = 120.0


@dataclass
class _Fleet:
    """The served demo, the fleet service and the readers started against them."""

    project: e2e.FleetProject
    served: _Server
    va_state: Path
    root: Path
    spool: Path
    watcher: e2e.SnapshotWatcher
    started: float
    period_s: float
    expected: frozenset[str]
    ca_env: dict[str, str]
    panel_ports: list[int]
    panels: list[e2e.Panel] = field(default_factory=list)
    tools: list[e2e.ToolReader] = field(default_factory=list)
    first_reader_at: float | None = None

    def readers(self) -> tuple[list[e2e.Panel], list[e2e.ToolReader]]:
        """The panel backends and tool readers, started on first use."""
        if not self.panels:
            if self.first_reader_at is None:
                self.first_reader_at = time.time()
            for index, port in enumerate(self.panel_ports):
                self.panels.append(
                    e2e.start_panel(
                        self.project,
                        port,
                        self.root / f"spool-panel-{index}",
                        self.root / f"panel-{index}.log",
                        self.ca_env,
                    )
                )
            for index in range(TOOL_READERS):
                self.tools.append(
                    e2e.ToolReader(
                        self.project,
                        self.root / f"spool-tool-{index}",
                        self.root / f"tool-{index}.log",
                        self.ca_env,
                    )
                )
        return self.panels, self.tools

    def stop(self) -> None:
        for tool in self.tools:
            tool.stop()
        for panel in self.panels:
            panel.stop()


def _activate(state_dir: Path, *names: str) -> None:
    """Write the active set, moving the file's modification time forward."""
    path = state_dir / "active_scenarios"
    before = path.stat().st_mtime_ns if path.exists() else 0
    path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
    stamp = max(before + 1_000_000, path.stat().st_mtime_ns)
    os.utime(path, ns=(stamp, stamp))


def _render_dirs(src: str, names: list[str]) -> list[str]:
    """The per-render directories of ``build/``: the fleet reads the deployment's config only."""
    return [name for name in names if Path(src).name == "build" and name.startswith("demo-")]


@pytest.fixture(scope="class")
def fleet(built_demo: BuiltProject, tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Fleet]:
    from osprey.services.channel_directory.fleet.settings import resolve_fleet
    from osprey_connectors.simulation import DEFAULT_TICK_S
    from osprey_connectors.simulation.servers import served_servers

    root = tmp_path_factory.mktemp("fleet")
    repo = copy_build(built_demo.repo, root, ignore=_render_dirs)
    config_path = repo / "build" / "config.yml"
    view_dir = repo / "build" / "data" / "simulator"
    models = json.loads((view_dir / "served_models.json").read_text(encoding="utf-8"))["models"]
    va_state = root / "va-state"
    va_state.mkdir()
    fleet_state = root / "fleet-state"
    spool = root / "spool-fleet"
    ca_port = int(CA_PORTS["fleet"])
    ca_env = {"EPICS_CA_ADDR_LIST": f"127.0.0.1:{ca_port}", "EPICS_CA_AUTO_ADDR_LIST": "NO"}
    with reserved_ports(1 + PANELS) as (fleet_port, *panel_ports):
        e2e.point_config(config_path, ca_port=ca_port, fleet_port=fleet_port)
        project = e2e.FleetProject(repo, config_path, view_dir, fleet_port)
        config = project.config()
        period_s = resolve_fleet(config, config["channel_directory"]["fleet"]).period_s
        with _served(
            view_dir, "fleet", state_dir=va_state, tick_interval_s=DEFAULT_TICK_S, physics=True
        ) as served:
            service = e2e.start_fleet_service(
                project, fleet_state, spool, root / "fleet-service.log", ca_env
            )
            run = _Fleet(
                project=project,
                served=served,
                va_state=va_state,
                root=root,
                spool=spool,
                watcher=e2e.SnapshotWatcher(fleet_state),
                started=time.time(),
                period_s=period_s,
                expected=served_servers(models),
                ca_env=ca_env,
                panel_ports=list(panel_ports),
            )
            try:
                yield run
            finally:
                _activate(va_state)
                run.stop()
                run.watcher.stop()
                e2e.stop_tree(service)


def _same_snapshot(fleet: _Fleet, attempts: int = 5) -> tuple[dict[str, Any], dict[str, Any]]:
    """A tool answer and a view answer of one snapshot (a cycle may land between two reads)."""
    panels, tools = fleet.readers()
    for _ in range(attempts):
        tool = tools[0].call()
        view = panels[0].view()
        if tool["generated_at"] == view["generated_at"]:
            return tool, view
    raise AssertionError(f"the tool and the view never answered one snapshot in {attempts} tries")


def _first_row(fleet: _Fleet) -> tuple[dict[str, Any], dict[str, Any]]:
    """The view's and the tool's first row, from one snapshot."""
    tool, view = _same_snapshot(fleet)
    assert view["outcome"] == "ok", view
    assert tool["outcome"] == "ok", tool
    return view["snapshot"]["servers"][0], tool["rows"][0]


class TestServerFleetOverTheServedDemo:
    """The fleet service, its readers and the served demo, in the order the demo runs them."""

    def test_the_refresher_reads_the_same_per_period_with_no_reader_and_with_many(
        self, fleet: _Fleet
    ) -> None:
        from osprey.services.channel_directory.fleet.refresher import LISTING_EVERY

        period = fleet.period_s
        # The quiet window: cycles 2.. (cycle 1 is the process's first listing),
        # every one published before any reader existed.
        assert fleet.first_reader_at is None, "a reader started before the quiet window"
        last_quiet = 1 + WINDOW_CYCLES
        assert fleet.watcher.wait_for(
            lambda c: c.number >= last_quiet, timeout=(last_quiet + 2) * period + 60.0
        ), f"the refresher never reached cycle {last_quiet}: {fleet.root / 'fleet-service.log'}"
        quiet = [c for c in fleet.watcher.cycles() if 2 <= c.number <= last_quiet]

        panels, tools = fleet.readers()
        stop = threading.Event()
        view_reads: list[int] = []
        failures: list[str] = []

        def read_view(panel: e2e.Panel) -> None:
            reads = 0
            while not stop.is_set():
                try:
                    answer = panel.view()
                except Exception as exc:
                    failures.append(f"{panel.url}: {type(exc).__name__}: {exc}")
                    return
                if answer["outcome"] != "ok":
                    failures.append(f"{panel.url}: {answer['outcome']} {answer['reason']}")
                    return
                reads += 1
            view_reads.append(reads)

        load_s = (WINDOW_CYCLES + 2) * period
        load_started = time.time()
        for tool in tools:
            tool.start_loop(load_s)
        threads = [
            threading.Thread(target=read_view, args=(panel,), daemon=True)
            for panel in panels
            for _ in range(VIEW_THREADS_PER_PANEL)
        ]
        for thread in threads:
            thread.start()
        try:
            # The loaded window: the first cycles to start after the load did,
            # every one published while it still ran.
            started_after = [c for c in fleet.watcher.cycles() if c.generated_at > load_started]
            first_loaded = (
                started_after[0].number
                if started_after
                else max(c.number for c in fleet.watcher.cycles()) + 1
            )
            last_loaded = first_loaded + WINDOW_CYCLES - 1
            assert fleet.watcher.wait_for(
                lambda c: c.number >= last_loaded, timeout=(WINDOW_CYCLES + 1) * period + 30.0
            ), f"the refresher never reached cycle {last_loaded} under load"
            loaded = [c for c in fleet.watcher.cycles() if first_loaded <= c.number <= last_loaded]
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=30)
        tool_calls = [tool.loop_count(load_s) for tool in tools]

        assert not failures, failures
        # The whole loaded window ran under the load: started after it began,
        # published before the tool readers' share of it ended.
        assert loaded[0].generated_at > load_started
        assert loaded[-1].seen_at <= load_started + load_s, (
            "the loaded window outlasted the tool readers' load"
        )
        assert sum(view_reads) >= MIN_VIEW_READS, view_reads
        assert sum(tool_calls) >= MIN_TOOL_CALLS, tool_calls

        reads = e2e.cycle_reads(fleet.spool)
        for window in (quiet, loaded):
            assert [c.number for c in window] == list(
                range(window[0].number, window[0].number + WINDOW_CYCLES)
            ), [c.number for c in window]
        quiet_reads = [reads[c.number] for c in quiet]
        loaded_reads = [reads[c.number] for c in loaded]
        # Channel reads: the same count every cycle, with readers or without.
        assert quiet_reads[0] > 0
        assert set(quiet_reads) == set(loaded_reads) == {quiet_reads[0]}, (
            quiet_reads,
            loaded_reads,
        )
        # One cycle per period in both windows: a reader never triggers a cycle.
        for window in (quiet, loaded):
            for before, after in zip(window, window[1:], strict=False):
                gap = after.generated_at - before.generated_at
                assert abs(gap - period) <= TICK_JITTER_S, (
                    f"cycles {before.number} and {after.number} started {gap:.2f} s apart"
                )
        # Directory queries: a listing on the refresher's own schedule, never on a read.
        for cycle in fleet.watcher.cycles():
            assert cycle.listed == (cycle.number % LISTING_EVERY == 1), (
                f"cycle {cycle.number} listed={cycle.listed}"
            )

    def test_every_simulated_server_is_alive_with_a_host_and_age_two_periods_after_start(
        self, fleet: _Fleet
    ) -> None:
        deadline = fleet.started + 2 * fleet.period_s
        assert fleet.watcher.wait_for(lambda c: c.seen_at > deadline, timeout=3 * fleet.period_s)
        current = [c for c in fleet.watcher.cycles() if c.seen_at <= deadline]
        assert current, "no cycle was published within two periods of the start"
        rows = {row["name"]: row for row in current[-1].document["servers"]}
        assert set(rows) == fleet.expected, sorted(rows)
        for name, row in rows.items():
            assert row["state"] == "alive", f"{name} at two periods: {row['state']} {row['reason']}"
            assert row["host"], f"{name} has no host"
            assert row["since"] is not None, f"{name} has no state age"
        # The view and the tool say so too.
        tool, view = _same_snapshot(fleet)
        assert e2e.states(view["snapshot"]) == dict.fromkeys(fleet.expected, "alive")
        assert {row["name"]: row["state"] for row in tool["rows"]} == dict.fromkeys(
            fleet.expected, "alive"
        )

    def test_the_tool_the_view_and_the_server_page_answer_one_snapshot_alike(
        self, fleet: _Fleet
    ) -> None:
        from osprey.server_fleet_connection import fleet_url

        # Every reader resolves the service from the config, not an override.
        assert fleet_url(fleet.project.config()) == fleet.project.fleet_url
        assert "SERVER_FLEET_URL" not in e2e.environment(fleet.project, fleet.spool)
        panels, _tools = fleet.readers()
        for _ in range(5):
            tool, view = _same_snapshot(fleet)
            page = panels[0].server_page(STOPPED_SERVER)
            if page["fleet"]["generated_at"] == view["generated_at"]:
                break
        else:
            raise AssertionError("the IOC page and the view never answered one snapshot")
        assert view["outcome"] == tool["outcome"] == "ok"
        snapshot = view["snapshot"]
        assert e2e.without_links(tool["rows"]) == snapshot["servers"]
        assert tool["counts"] == snapshot["counts"]
        assert tool["total"] == len(snapshot["servers"]) == tool["matched"]
        row = next(r for r in snapshot["servers"] if r["name"] == STOPPED_SERVER)
        assert page["fleet"]["outcome"] == "ok"
        assert page["fleet"]["row"] == row
        # The other panel backends answer the same snapshot alike.
        for panel in panels[1:]:
            other = panel.view()
            if other["generated_at"] == view["generated_at"]:
                assert other == view

    def test_the_stop_scenario_makes_its_server_silent_and_first_then_alive_again(
        self, fleet: _Fleet
    ) -> None:
        period = fleet.period_s
        applied_at = time.time()
        try:
            _activate(fleet.va_state, STOP_SCENARIO)
            silent = fleet.watcher.wait_for(
                lambda c: (
                    c.generated_at > applied_at
                    and e2e.states(c.document).get(STOPPED_SERVER) == "silent"
                ),
                timeout=2 * period + 30.0,
            )
            assert silent is not None, f"{STOPPED_SERVER} never went silent"
            assert silent.seen_at - applied_at <= 2 * period, (
                f"{STOPPED_SERVER} went silent {silent.seen_at - applied_at:.1f} s after the stop"
            )
            assert silent.document["servers"][0]["name"] == STOPPED_SERVER
            others = {
                name: state
                for name, state in e2e.states(silent.document).items()
                if name != STOPPED_SERVER
            }
            assert set(others.values()) == {"alive"}, others
            # The view and the tool, read now, show it first and silent.
            view_row, tool_row = _first_row(fleet)
            assert (view_row["name"], view_row["state"]) == (STOPPED_SERVER, "silent"), view_row
            assert (tool_row["name"], tool_row["state"]) == (STOPPED_SERVER, "silent"), tool_row
        finally:
            _activate(fleet.va_state)
        removed_at = time.time()
        alive = fleet.watcher.wait_for(
            lambda c: (
                c.generated_at > removed_at
                and e2e.states(c.document) == dict.fromkeys(fleet.expected, "alive")
            ),
            timeout=RESTART_TIMEOUT_S + 2 * period,
        )
        assert alive is not None, f"{STOPPED_SERVER} never came back alive"
        tool, view = _same_snapshot(fleet)
        assert e2e.states(view["snapshot"]) == dict.fromkeys(fleet.expected, "alive")
        assert {row["name"]: row["state"] for row in tool["rows"]} == dict.fromkeys(
            fleet.expected, "alive"
        )


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    assert_module_collected(request, __file__, MIN_COLLECTED_TESTS)
