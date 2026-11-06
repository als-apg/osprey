"""The demo-history story is told once: archive, plot and logbook agree.

With the example profile's default set plus ``demo-history`` active at a fixed
anchor, the values the archive rewrite (``seed_archiver``) stores for the stored
current and the sector 7 gauge around 14:32:08 yesterday are the values the
seeded session's plot draws -- the same drop, at the same instant -- and the
logbook entry that narrates the burst is stamped at that instant too. A demo
whose plot disagrees with the archive an operator can query next to it would
teach the opposite of what it is for.

No MongoDB: ``seed_archiver`` reads every value it writes from the archive
composite through ``_recomputed_values`` (stored documents) and
``_dense_documents`` (dense inserts at the event instants), over the windows
``event_window`` computes at the persisted anchor. This test calls those same
functions on the same composite.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.records import FacilityRecords
from osprey.simulation.apply import (
    _dense_documents,
    _recomputed_values,
    active_archiver_events,
    active_logbook_entries,
    apply_scenarios,
    event_window,
    persisted_scenario_anchor,
)
from osprey.simulation.session_artifacts import PLOT_DIV_ID, render_session_artifacts
from osprey.simulation.sessions import parse_sessions, resolve
from osprey_connectors.archiver import field_name
from osprey_connectors.config import get_facility_timezone
from osprey_connectors.simulation.series import anchored_instant

TEMPLATE_FACILITY = Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example"
SCENARIO_FILE = TEMPLATE_FACILITY / "scenarios" / "demo-history.yaml"
VIEW_CONFIG: dict[str, Any] = {"control_system": {"type": "virtual_accelerator"}}

DCCT = "SR:DIAG:DCCT:01:CURRENT:RB"
SR07 = "SR:VAC:GAUGE:SR07:PRESSURE:RB"
ANCHOR = datetime(2026, 11, 5, 12, 0, 0, tzinfo=UTC)
NAMES = ["rf-thermal", "demo-history"]
PLOT_KEY = ("sector-7-beam-loss", "loss-plot")
# The magnitude of each demo-history spike, as the scenario states it.
AMPLITUDE = {DCCT: 18.0, SR07: 1.5e-07}
# The loss plot's own margin either side of the burst (``PLOT_MARGIN_S``).
NEAR_S = 600.0
# Older than any window the demo touches; the real store reads it from its oldest sample.
HORIZON_START = ANCHOR.timestamp() - 30 * 86400


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The example facility's simulator view, with the set applied at ``ANCHOR``."""
    from osprey.facility.build import build_facility
    from osprey.facility.served import resolve_served
    from osprey.facility.views import ViewInputs
    from osprey.facility.views.simulator import write_simulator_view

    root = tmp_path_factory.mktemp("project")
    doc = build_facility(TEMPLATE_FACILITY, project_name="example")
    write_simulator_view(
        root / "data" / "simulator",
        ViewInputs(
            records=FacilityRecords(doc),
            rendered_config=VIEW_CONFIG,
            facility_dir=TEMPLATE_FACILITY,
            served=resolve_served(VIEW_CONFIG, FacilityRecords(doc)),
            reported=None,
        ),
    )
    (root / "config.yml").write_text(yaml.safe_dump({}), encoding="utf-8")
    apply_scenarios(root, NAMES, seed_logbook=False, seed_archive=False, now=ANCHOR)
    return root


@pytest.fixture(scope="module")
def config(project: Path) -> dict:
    return yaml.safe_load((project / "config.yml").read_text(encoding="utf-8")) or {}


@pytest.fixture(scope="module")
def plot(project: Path) -> dict[str, dict[str, Any]]:
    """The seeded loss plot's traces, by channel."""
    raw = yaml.safe_load(SCENARIO_FILE.read_text(encoding="utf-8"))["sessions"]
    payloads = render_session_artifacts(
        project, NAMES, resolve(parse_sessions(raw), ANCHOR), ANCHOR
    )
    html = payloads[PLOT_KEY].content.decode("utf-8")
    marker = f'Plotly.newPlot(                        "{PLOT_DIV_ID}",'
    start = html.index(marker) + len(marker)
    traces, _end = json.JSONDecoder().raw_decode(html[start:].lstrip())
    return {trace["name"]: trace for trace in traces}


def _event_instant() -> float:
    event = {"at_when": {"days_ago": 1, "time": "14:32:08"}}
    return anchored_instant(event, ANCHOR.timestamp(), get_facility_timezone())


def _axis_seconds(axis: list[str]) -> list[float]:
    tz = get_facility_timezone()
    return [datetime.fromisoformat(stamp).replace(tzinfo=tz).timestamp() for stamp in axis]


def _seeded(
    project: Path, config: dict, epochs: list[float]
) -> tuple[dict[str, dict[int, Any]], dict[str, tuple[float, float]]]:
    """What ``seed_archiver`` rewrites at ``epochs``, and the window it rewrites, by channel.

    Its own composite (at the persisted anchor), its own windows and its own reader.
    """
    from osprey_connectors.simulation.archive import build
    from osprey_connectors.simulation.view import SimulatorView

    persisted = persisted_scenario_anchor(config, project)
    assert persisted is not None
    events = {pv: script for pv, script in active_archiver_events(project, NAMES).items() if script}
    tz = get_facility_timezone()
    with build(SimulatorView.of_project(project), NAMES, anchor_s=persisted.timestamp()) as archive:
        spans = {
            pv: event_window(script, ANCHOR.timestamp(), HORIZON_START, tz=tz)
            for pv, script in events.items()
            if pv in set(archive.addresses)
        }
        assert {DCCT, SR07} <= set(spans)
        return _recomputed_values(archive, spans, [DCCT, SR07], epochs), spans


def _base(project: Path, epochs: list[float]) -> dict[str, list[Any]]:
    """The two channels with ``demo-history`` inactive: what the base seed stores.

    Outside the rewrite's window the archive keeps the values the base seed wrote,
    which no active event moves; ``rf-thermal`` touches neither channel.
    """
    from osprey_connectors.simulation.archive import build
    from osprey_connectors.simulation.view import SimulatorView

    with build(
        SimulatorView.of_project(project), ["rf-thermal"], anchor_s=ANCHOR.timestamp()
    ) as archive:
        return archive.samples([DCCT, SR07], epochs)


# ---------------------------------------------------------------------------
# One clock
# ---------------------------------------------------------------------------


def test_the_seed_reads_the_anchor_the_set_was_applied_at(project: Path, config: dict) -> None:
    assert persisted_scenario_anchor(config, project) == ANCHOR


def test_the_burst_is_at_1432_yesterday_in_the_facility_zone() -> None:
    local = datetime.fromtimestamp(_event_instant(), get_facility_timezone())
    assert local.strftime("%H:%M:%S") == "14:32:08"
    assert local.date() < ANCHOR.astimezone(get_facility_timezone()).date()


# ---------------------------------------------------------------------------
# The archive and the plot
# ---------------------------------------------------------------------------


def test_the_archived_values_are_the_plotted_values(project: Path, config: dict, plot) -> None:
    seconds = _axis_seconds(plot[DCCT]["x"])
    assert _axis_seconds(plot[SR07]["x"]) == seconds
    event = _event_instant()

    seeded, spans = _seeded(project, config, seconds)
    base = _base(project, seconds)

    for pv in (DCCT, SR07):
        low, high = spans[pv]
        assert low < event < high, pv
        inside = [i for i, t in enumerate(seconds) if low <= t <= high]
        assert sorted(seeded[pv]) == inside, pv
        # Inside the rewritten window the archive holds exactly what the plot draws.
        assert [float(seeded[pv][i]) for i in inside] == [plot[pv]["y"][i] for i in inside], pv
        # Outside it the archive keeps its base. The plot draws the composite there,
        # which still carries the spike's tail beyond the rewrite's 4-sigma window;
        # the two may differ by that tail and no more -- a thousandth of the spike.
        outside = [i for i in range(len(seconds)) if i not in seeded[pv]]
        assert outside, "the plot's margin reaches past the rewritten window"
        tail = 1e-3 * AMPLITUDE[pv]
        for i in outside:
            assert abs(plot[pv]["y"][i] - float(base[pv][i])) < tail, (pv, i)


def test_the_archive_drops_and_spikes_at_the_plotted_instant(
    project: Path, config: dict, plot
) -> None:
    event = _event_instant()
    seconds = _axis_seconds(plot[DCCT]["x"])
    at = seconds.index(event)
    seeded, _spans = _seeded(project, config, seconds)
    base = _base(project, seconds)
    # Only the stretch around the burst, where the two scenarios' events cannot mix.
    near = [i for i, t in enumerate(seconds) if abs(t - event) <= NEAR_S]
    at = near.index(at)

    def archived(pv: str) -> list[float]:
        return [float(seeded[pv].get(i, base[pv][i])) for i in near]

    def plotted(pv: str) -> list[float]:
        return [plot[pv]["y"][i] for i in near]

    current, pressure = archived(DCCT), archived(SR07)

    # The same instant is the extreme of both, in the archive as in the plot.
    assert min(current) == pytest.approx(current[at], abs=2.0)
    assert max(pressure) == pressure[at]
    assert min(plotted(DCCT)) == pytest.approx(plotted(DCCT)[at], abs=2.0)
    assert plotted(SR07).index(max(plotted(SR07))) == at
    # The same drop: about 18 mA off a ~500 mA beam, the gauge near 1.5e-07 Torr.
    assert 490 < current[0] < 510
    assert current[0] - current[at] == pytest.approx(18.0, abs=3.0)
    assert pressure[at] == pytest.approx(1.5e-07, rel=0.2)
    assert pressure[0] < 1e-8


def test_the_dense_sample_at_the_burst_is_the_plotted_peak(project: Path, plot) -> None:
    """The rewrite inserts a dense sample at the event instant; it carries the plot's value."""
    from osprey_connectors.simulation.archive import build
    from osprey_connectors.simulation.view import SimulatorView

    event = _event_instant()
    seconds = _axis_seconds(plot[DCCT]["x"])
    at = seconds.index(event)
    with build(SimulatorView.of_project(project), NAMES, anchor_s=ANCHOR.timestamp()) as archive:
        (document,) = _dense_documents(archive, {}, {event: (DCCT, SR07)})

    assert document["date"] == datetime.fromtimestamp(event, UTC)
    assert document[field_name(DCCT)] == plot[DCCT]["y"][at]
    assert document[field_name(SR07)] == plot[SR07]["y"][at]


def test_the_loss_plot_stays_around_the_burst_with_the_default_set(plot) -> None:
    """rf-thermal's beam dump days earlier on the same DCCT stays out: the plot shows
    the window the turn's archiver_read stated, not every event in the active set."""
    seconds = _axis_seconds(plot[DCCT]["x"])
    event = _event_instant()

    assert seconds[0] >= event - 2 * NEAR_S
    assert seconds[-1] <= event + 2 * NEAR_S
    assert min(plot[DCCT]["y"]) > 400


# ---------------------------------------------------------------------------
# Logbook
# ---------------------------------------------------------------------------


def test_the_logbook_entry_is_stamped_at_the_burst(project: Path, config: dict) -> None:
    entries = {entry["entry_id"]: entry for entry in active_logbook_entries(config, project)}

    assert "DEMO-036" in entries
    stamp = entries["DEMO-036"]["timestamp"]
    assert stamp.timestamp() == _event_instant()
