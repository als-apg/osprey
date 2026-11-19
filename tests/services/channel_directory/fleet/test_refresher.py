"""The refresher of one target's fleet (D2, D3, D5, D7, D8, D12), driven with fakes."""

from __future__ import annotations

import asyncio
import fnmatch
from types import SimpleNamespace

import pytest

pytest.importorskip("osprey_connectors.safety_file")
pytest.importorskip("osprey_connectors.service_answers")

from osprey.runtime import raw_put_block
from osprey.services.channel_directory.fleet import refresher as refresher_module
from osprey.services.channel_directory.fleet.memory import FleetMemory
from osprey.services.channel_directory.fleet.refresher import (
    DIRECTORY_SOURCE,
    EXIT_CANNOT_BUILD,
    EXIT_CONNECTOR_LOST,
    EXIT_DONE,
    Refresher,
    prepare_read_only_process,
    run_target,
)
from osprey.services.channel_directory.fleet.settings import (
    ChannelSelector,
    FleetClass,
    FleetSettings,
    FleetTarget,
    LivenessChannel,
)
from osprey.services.channel_directory.fleet.snapshot import Outcome, read_snapshot
from osprey.services.channel_directory.fleet.states import ReadPath, State
from osprey.services.channel_directory.roster import RosterRow
from osprey_connectors.control_system._base import WriteOutcome
from osprey_connectors.directory import ServerRow
from osprey_connectors.errors import ChannelUnreachableError, ConnectorUnavailableError
from osprey_connectors.posture_env import POSTURE_ENV_VAR
from osprey_connectors.service_answers import Unavailable
from osprey_connectors_tests._write_fakes import (
    RecordingConnector,
    arm_switch,
    writes_enabled_config,
)

PERIOD = 10.0
START = 1_000_000.0
PROBE = "SR:PROBE"
ROUTE = "epics"


# --- Fakes ----------------------------------------------------------------------------


class Clock:
    """Wall time that moves only when the refresher sleeps (or a test moves it)."""

    def __init__(self, now: float = START) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeConnector:
    """Reads from a table; a callable value is called on each read. Never writes."""

    def __init__(self, values: dict, *, unreachable=(), failing=()) -> None:
        self.values = dict(values)
        self.unreachable = set(unreachable)
        self.failing = set(failing)
        self.reads: list[str] = []
        self.stale = asyncio.Event()
        self.lose_on: str | None = None
        self.on_read = None

    async def read_channel(self, address: str, timeout: float | None = None):  # noqa: ARG002
        self.reads.append(address)
        if self.on_read is not None:
            self.on_read(address)
        if address == self.lose_on:
            self.stale.set()
            await asyncio.Event().wait()  # the transport is gone; this read never ends
        if address in self.unreachable:
            raise ChannelUnreachableError(address)
        if address in self.failing:
            raise ValueError(f"bad value from {address}")
        value = self.values[address]
        return SimpleNamespace(value=value() if callable(value) else value)

    async def wait_stale(self) -> None:
        await self.stale.wait()
        raise ConnectorUnavailableError("the transport went stale")

    async def disconnect(self) -> None:
        pass


class FakeDirectory:
    """A directory listing the given IOCs, each with ``<ioc>:HEARTBEAT``; counts its calls."""

    def __init__(self, iocs, *, host: str = "host-1") -> None:
        self.servers = list(iocs)
        self.host = host
        self.calls: list[tuple] = []
        self.fail = False

    def rows(self):
        return [{"name": f"{ioc}:HEARTBEAT", "server": ioc} for ioc in self.servers]

    async def list_servers(self, deadline: float):  # noqa: ARG002
        self.calls.append(("list_servers",))
        if self.fail:
            return Unavailable("the directory service answered 503")
        return [
            ServerRow(name=i, host=self.host, channel_count=3, inactive_count=0)
            for i in self.servers
        ]

    async def search(self, query, deadline: float):  # noqa: ARG002
        self.calls.append(("search", query.name, query.page))
        if self.fail:
            return SimpleNamespace(
                outcome="unavailable", reason="503", rows=[], total=None, partial=[]
            )
        matched = [
            SimpleNamespace(name=r["name"], server=r["server"])
            for r in self.rows()
            if query.name is None or fnmatch.fnmatchcase(r["name"], query.name)
        ]
        start = query.page * query.page_size
        page = matched[start : start + query.page_size]
        return SimpleNamespace(
            outcome="found" if matched else "no_match",
            reason=None,
            rows=page,
            total=len(matched),
            partial=[],
        )


class Roster:
    name = "ops"

    def __init__(self, rows) -> None:
        self.rows = rows

    def read(self, deadline: float):  # noqa: ARG002
        return list(self.rows)


def counter(start: float = 0.0):
    """A heartbeat that moves on every read."""
    state = {"value": start}

    def read():
        state["value"] += 1
        return state["value"]

    return read


def settings(**overrides) -> FleetSettings:
    fields = {
        "targets": (FleetTarget(name="live", read_bound_s=1.0),),
        "period_s": PERIOD,
        "read_bound_s": 1.0,
    }
    fields.update(overrides)
    return FleetSettings(**fields)


def make(tmp_path, connector, directory, *, clock=None, audits=None, **overrides) -> Refresher:
    clock = clock or Clock()
    fields = {
        "target": "live",
        "settings": settings(),
        "connector": connector,
        "state_dir": tmp_path,
        "route": ROUTE,
        "probe_channel": PROBE,
        "directory": directory,
        "directory_kind": "channelfinder",
        "audit": (lambda *a: audits.append(a)) if audits is not None else (lambda *a: None),
        "clock": clock,
        "sleep": clock.sleep,
    }
    fields.update(overrides)
    return Refresher(**fields)


def states(snapshot) -> dict[str, State]:
    return {row.name: row.state for row in snapshot.servers}


def fleet_values(names, *, still=()):
    values = {PROBE: 1}
    for name in names:
        values[f"{name}:HEARTBEAT"] = 7 if name in still else counter()
    return values


# --- Connector loss and the successor (D3) ------------------------------------------


async def test_connector_loss_publishes_exits_and_the_successor_resumes_the_states(tmp_path):
    names = ["ioc-a", "ioc-b", "ioc-c"]
    values = fleet_values(names, still={"ioc-c"})
    connector = FakeConnector(values)
    directory = FakeDirectory(names)
    clock = Clock()
    first = make(tmp_path, connector, directory, clock=clock)

    published = []
    original = first.cycle

    async def cycle():
        snapshot = await original()
        published.append(snapshot)
        if len(published) == 2:
            connector.lose_on = "ioc-b:HEARTBEAT"  # the next cycle loses the connector mid-read
        return snapshot

    first.cycle = cycle
    status = await first.run()

    assert status == EXIT_CONNECTOR_LOST
    assert status != EXIT_CANNOT_BUILD
    before_loss = states(published[1])
    assert before_loss == {"ioc-a": State.ALIVE, "ioc-b": State.ALIVE, "ioc-c": State.STALLED}
    # The cycle the connector was lost in is published from what was read.
    on_disk = read_snapshot(tmp_path, "live")
    assert on_disk.cycle == 3
    assert states(on_disk)["ioc-a"] is State.ALIVE
    assert states(on_disk)["ioc-c"] is State.STALLED
    lost = on_disk.row("ioc-b")
    assert lost.state is State.UNKNOWN
    assert "connector lost" in lost.reason
    last_cycle = FleetMemory.load(tmp_path, retention=1e9).last_cycle_time("live", ROUTE)
    assert last_cycle == clock.now

    # The successor: a fresh process a moment later, the counters still moving.
    clock.now += 1.0
    successor = make(tmp_path, FakeConnector(values), FakeDirectory(names), clock=clock)
    clock.sleeps.clear()

    assert await successor.run(max_cycles=1) == EXIT_DONE

    assert clock.sleeps == [pytest.approx(PERIOD - 1.0)]  # one period after the stored cycle
    resumed = read_snapshot(tmp_path, "live")
    assert resumed.cycle == 4
    assert states(resumed) == before_loss
    moving = {"ioc-a", "ioc-b"}
    assert not [r.name for r in resumed.servers if r.name in moving and r.state is State.STALLED]


async def test_a_stale_transport_while_waiting_exits_without_another_read(tmp_path):
    names = ["ioc-a"]
    connector = FakeConnector(fleet_values(names))
    clock = Clock()

    async def sleep(_seconds):
        connector.stale.set()
        await asyncio.Event().wait()

    refresher = make(tmp_path, connector, FakeDirectory(names), clock=clock, sleep=sleep)
    refresher.memory.record_cycle("live", ROUTE, START)  # the first cycle is a period away

    assert await refresher.run() == EXIT_CONNECTOR_LOST
    assert connector.reads == []


# --- Scheduling ----------------------------------------------------------------------


async def test_nothing_but_the_clock_triggers_a_read(tmp_path):
    names = ["ioc-a", "ioc-b"]
    connector = FakeConnector(fleet_values(names))
    clock = Clock()
    tick = asyncio.Event()
    asked: list[float] = []

    async def sleep(seconds):
        asked.append(seconds)
        await tick.wait()
        tick.clear()
        clock.now += seconds

    refresher = make(tmp_path, connector, FakeDirectory(names), clock=clock, sleep=sleep)
    task = asyncio.create_task(refresher.run(max_cycles=2))
    for _ in range(50):
        await asyncio.sleep(0)
    after_first = len(connector.reads)
    assert after_first == 3  # the probe and two heartbeats
    assert asked == [PERIOD]

    # Files changing, the directory changing, time passing on the loop: no read.
    (tmp_path / "unrelated").write_text("x")
    FakeDirectory(names + ["ioc-z"])
    for _ in range(50):
        await asyncio.sleep(0)
    assert len(connector.reads) == after_first

    tick.set()
    assert await task == EXIT_DONE
    assert len(connector.reads) == 2 * after_first


async def test_an_overrunning_cycle_skips_the_next_tick(tmp_path):
    names = ["ioc-a"]
    connector = FakeConnector(fleet_values(names))
    clock = Clock()

    def slow(address):
        if address == "ioc-a:HEARTBEAT" and len(connector.reads) <= 2:
            clock.now += 1.5 * PERIOD  # the first cycle runs one and a half periods

    connector.on_read = slow
    refresher = make(tmp_path, connector, FakeDirectory(names), clock=clock)

    await refresher.run(max_cycles=2)

    # Scheduled at START; it ended at START + 15; the tick at START + 10 is skipped.
    assert clock.sleeps == [pytest.approx(5.0)]


# --- The read path (D5) --------------------------------------------------------------


async def test_a_silent_probe_publishes_every_server_unknown_and_reads_nothing_else(tmp_path):
    names = ["ioc-a", "ioc-b"]
    connector = FakeConnector(fleet_values(names))
    directory = FakeDirectory(names)
    refresher = make(tmp_path, connector, directory)
    await refresher.cycle()
    reads, calls = len(connector.reads), len(directory.calls)

    connector.unreachable.add(PROBE)
    snapshot = await refresher.cycle()

    assert snapshot.read_path.status is ReadPath.DOWN
    assert set(states(snapshot).values()) == {State.UNKNOWN}
    assert connector.reads[reads:] == [PROBE]
    assert len(directory.calls) == calls


async def test_a_target_without_a_probe_is_unverified(tmp_path):
    names = ["ioc-a"]
    connector = FakeConnector(fleet_values(names))
    snapshot = await make(tmp_path, connector, FakeDirectory(names), probe_channel=None).cycle()

    assert snapshot.read_path.status is ReadPath.UNVERIFIED
    assert PROBE not in connector.reads


# --- Reads (C7) ------------------------------------------------------------------------


async def test_no_answer_and_a_failed_read_feed_different_states(tmp_path):
    names = ["ioc-a", "ioc-b", "ioc-c"]
    connector = FakeConnector(fleet_values(names))
    clock = Clock()
    refresher = make(tmp_path, connector, FakeDirectory(names), clock=clock)
    await refresher.cycle()
    clock.now += PERIOD

    connector.unreachable.add("ioc-a:HEARTBEAT")
    connector.failing.add("ioc-b:HEARTBEAT")
    snapshot = await refresher.cycle()

    assert snapshot.row("ioc-a").state is State.SILENT
    assert snapshot.row("ioc-b").state is State.UNKNOWN
    assert snapshot.row("ioc-c").state is State.ALIVE


# --- Listing (D7, D8) -----------------------------------------------------------------


async def test_a_failed_listing_keeps_the_rows_and_reports_the_directory_unavailable(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(refresher_module, "LISTING_EVERY", 1)
    names = ["ioc-a", "ioc-b"]
    directory = FakeDirectory(names)
    clock = Clock()
    refresher = make(tmp_path, FakeConnector(fleet_values(names)), directory, clock=clock)
    first = await refresher.cycle()
    clock.now += PERIOD

    directory.fail = True
    snapshot = await refresher.cycle()

    assert snapshot.outcome is Outcome.OK
    assert [r.name for r in snapshot.servers] == [r.name for r in first.servers]
    [source] = [s for s in snapshot.sources if s.name == DIRECTORY_SOURCE]
    assert source.outcome == "unavailable"
    assert "10 s old" in source.reason
    assert all(r.state is State.ALIVE for r in snapshot.servers)

    # It retries on the next cycle and recovers.
    clock.now += PERIOD
    directory.fail = False
    recovered = await refresher.cycle()
    [source] = [s for s in recovered.sources if s.name == DIRECTORY_SOURCE]
    assert source.outcome == "ok"


async def test_cannot_list_only_when_no_listing_ever_succeeded(tmp_path):
    directory = FakeDirectory(["ioc-a"])
    directory.fail = True
    snapshot = await make(tmp_path, FakeConnector(fleet_values(["ioc-a"])), directory).cycle()

    assert snapshot.outcome is Outcome.CANNOT_LIST
    assert "503" in snapshot.reason


async def test_a_target_without_a_listable_directory_publishes_cannot_list(tmp_path):
    connector = FakeConnector({PROBE: 1})
    snapshot = await make(
        tmp_path, connector, None, no_directory_reason="the target 'live' has no channel directory"
    ).cycle()

    assert snapshot.outcome is Outcome.CANNOT_LIST
    assert connector.reads == []


@pytest.mark.parametrize("count", [3, 40])
async def test_directory_calls_per_cycle_are_constant_in_the_number_of_servers(tmp_path, count):
    names = [f"ioc-{i:03d}" for i in range(count)]
    directory = FakeDirectory(names)
    clock = Clock()
    refresher = make(tmp_path, FakeConnector(fleet_values(names)), directory, clock=clock)

    await refresher.cycle()
    listing_cycle = list(directory.calls)
    clock.now += PERIOD
    await refresher.cycle()

    # One listing and one paged search per selector, whatever the fleet's size.
    assert listing_cycle == [("list_servers",), ("search", "*HEARTBEAT", 0)]
    assert directory.calls == listing_cycle  # no directory call between listings


async def test_the_listing_repeats_every_tenth_cycle(tmp_path):
    names = ["ioc-a"]
    directory = FakeDirectory(names)
    clock = Clock()
    refresher = make(tmp_path, FakeConnector(fleet_values(names)), directory, clock=clock)
    for _ in range(11):
        await refresher.cycle()
        clock.now += PERIOD

    assert directory.calls.count(("list_servers",)) == 2


async def test_a_remembered_server_no_longer_listed_is_still_read(tmp_path, monkeypatch):
    monkeypatch.setattr(refresher_module, "LISTING_EVERY", 1)
    names = ["ioc-a", "ioc-b"]
    directory = FakeDirectory(names)
    connector = FakeConnector(fleet_values(names))
    clock = Clock()
    refresher = make(tmp_path, connector, directory, clock=clock)
    await refresher.cycle()
    clock.now += PERIOD

    directory.servers = ["ioc-a"]
    snapshot = await refresher.cycle()

    gone = snapshot.row("ioc-b")
    assert gone is not None and gone.listed is False
    assert gone.state is State.ALIVE
    assert connector.reads.count("ioc-b:HEARTBEAT") == 2


# --- Roster and classes (live only) -------------------------------------------------


async def test_a_roster_only_server_is_read_through_its_class_pattern(tmp_path):
    motion = FleetClass(
        name="motion",
        liveness=LivenessChannel(selector=ChannelSelector(pattern="{server}:ALIVE"), kind="status"),
    )
    roster = Roster([RosterRow("ioc-x", "host-9", "motion", "running", group="motion")])
    values = {**fleet_values(["ioc-a"]), "ioc-x:ALIVE": 1}
    connector = FakeConnector(values)
    refresher = make(
        tmp_path,
        connector,
        FakeDirectory(["ioc-a"]),
        settings=settings(classes={"motion": motion}),
        roster_sources=[roster],
    )

    snapshot = await refresher.cycle()

    row = snapshot.row("ioc-x")
    assert "ioc-x:ALIVE" in connector.reads
    assert row.liveness.channel == "ioc-x:ALIVE"
    assert row.state is State.ALIVE
    assert row.listed is False
    assert row.class_name == "motion"
    assert row.sources == ("ops",)
    assert row.host == "host-9"
    assert [s.name for s in snapshot.sources] == [DIRECTORY_SOURCE, "ops"]


async def test_roster_sources_are_not_read_off_live(tmp_path):
    roster = Roster([RosterRow("ioc-x", None, None, "running")])
    refresher = make(
        tmp_path,
        FakeConnector(fleet_values(["ioc-a"])),
        FakeDirectory(["ioc-a"]),
        target="va",
        settings=settings(targets=(FleetTarget(name="va", read_bound_s=1.0),)),
        roster_sources=[roster],
    )

    snapshot = await refresher.cycle()

    assert snapshot.row("ioc-x") is None
    assert [s.name for s in snapshot.sources] == [DIRECTORY_SOURCE]


# --- Publishing --------------------------------------------------------------------------


async def test_each_cycle_writes_one_audit_record_and_its_duration(tmp_path):
    names = ["ioc-a", "ioc-b"]
    audits: list[tuple] = []
    connector = FakeConnector(fleet_values(names))
    clock = Clock()
    connector.on_read = lambda address: setattr(clock, "now", clock.now + 0.25)
    refresher = make(tmp_path, connector, FakeDirectory(names), clock=clock, audits=audits)

    snapshot = await refresher.cycle()

    assert audits == [("live", 3, 1)]
    assert snapshot.cycle_duration_s == pytest.approx(0.75)
    assert read_snapshot(tmp_path, "live") == snapshot


# --- The process -------------------------------------------------------------------------

CONFIG = {
    "control_system": {"type": "epics", "connector": {"epics": {"timeout_s": 2.0}}},
    "channel_directory": {"type": "channelfinder", "fleet": {"targets": ["live"]}},
}


async def test_a_connector_that_cannot_be_built_writes_not_refreshed(tmp_path):
    async def cannot_build(_section, _target):
        raise ConnectionError("no gateway answers")

    status = await run_target(
        CONFIG, "live", tmp_path, prepare=lambda: None, build_connector=cannot_build
    )

    assert status == EXIT_CANNOT_BUILD
    assert status != EXIT_CONNECTOR_LOST
    snapshot = read_snapshot(tmp_path, "live")
    assert snapshot.outcome is Outcome.NOT_REFRESHED
    assert "no gateway answers" in snapshot.reason


async def test_the_posture_is_entered_before_the_connector_is_built(tmp_path):
    order: list[str] = []

    async def build(_section, _target):
        order.append("build")
        raise ConnectionError("stop here")

    await run_target(
        CONFIG, "live", tmp_path, prepare=lambda: order.append("posture"), build_connector=build
    )

    assert order == ["posture", "build"]


@pytest.fixture
def writes_enabled_deployment(monkeypatch, tmp_path):
    """A write tier that allows and the switch on: only the refresher's posture refuses."""
    monkeypatch.setattr("osprey_connectors.config.get_config_value", writes_enabled_config)
    arm_switch(monkeypatch, tmp_path / "armed-data")
    monkeypatch.setenv(POSTURE_ENV_VAR, "readwrite")
    installed: list[str] = []
    monkeypatch.setattr(raw_put_block, "install", lambda mode, **contract: installed.append(mode))
    return installed


@pytest.mark.usefixtures("writes_enabled_deployment")
async def test_the_refreshers_connector_cannot_issue_a_put(writes_enabled_deployment):
    prepare_read_only_process()
    connector = RecordingConnector()

    result = await connector.write_channel("SR:MAG:QF:01:CURRENT:SP", 1.0)

    assert result.outcome is WriteOutcome.REFUSED
    assert connector.writes == []
    assert writes_enabled_deployment == ["armed"]  # raw client puts are blocked too
