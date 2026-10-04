"""The demo scenario pictures say what the scenarios' own definitions say.

``scripts/scenario_plots.py`` draws the pictures the control-assistant demo
scenarios attach to their logbook entries. Each carries a fact only the picture
shows, so these tests pin that fact to the scenario definition it is derived
from, and pin that a re-run draws the same bytes.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from datetime import datetime
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pytest

from osprey_connectors.relative_time import RelativeTimestamp, resolve_relative_timestamp

REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "scenario_plots", REPO_ROOT / "scripts" / "scenario_plots.py"
)
assert _SPEC is not None and _SPEC.loader is not None
plots = importlib.util.module_from_spec(_SPEC)
# Registered before it runs: its dataclasses resolve their module by name.
sys.modules[_SPEC.name] = plots
_SPEC.loader.exec_module(plots)

SIM = plots.DEFAULT_SIMULATION_DIR


def _scenario(name: str) -> dict:
    return json.loads((SIM / "scenarios" / name / "scenario.json").read_text())


def test_bump_test_flips_exactly_the_scenarios_reversed_bpm():
    test = plots.bump_test(SIM)
    reversed_in_scenario = {
        name
        for name, err in _scenario("bpm-polarity")["physics"]["bpm_errors"].items()
        if err.get("polarity") == -1
    }
    assert set(test.reversed_bpms) == reversed_in_scenario
    assert set(test.reversed_bpms) <= set(test.bpms)
    kicks = plots.KICKS_URAD
    for bpm, model, readings in zip(test.bpms, test.model_slopes, test.readings_um, strict=True):
        measured = float(np.polyfit(kicks, readings, 1)[0])
        expected = -model if bpm in test.reversed_bpms else model
        assert measured == pytest.approx(expected, abs=0.1), bpm
        assert abs(model) > 0.5, f"{bpm} barely responds; the picture would show nothing"


def _entry(scenario: str, plot: str) -> dict:
    """The one entry in ``scenario``'s logbook that attaches ``plot``."""
    entries = json.loads((SIM / "scenarios" / scenario / "logbook.json").read_text())
    (entry,) = [e for e in entries if any(a["path"] == plot for a in e.get("attachments", []))]
    return entry


def _resolved(entry: dict) -> datetime:
    when = entry["when"]
    spec = RelativeTimestamp(days_ago=when["days_ago"], time=dtime.fromisoformat(when["time"]))
    return resolve_relative_timestamp(spec, plots.ANCHOR)


@pytest.mark.parametrize(
    ("scenario", "plot", "trend_of"),
    [
        ("rf-thermal", plots.RF_THERMAL_PLOT, plots.cavity_temperatures),
        ("nominal", plots.NOMINAL_PLOT, plots.orbit_rms_week),
    ],
)
def test_each_trend_ends_at_its_entrys_time(scenario, plot, trend_of):
    """A picture shows what its author could have seen: its window ends at the entry."""
    end = plots.entry_time(SIM, scenario, plot)
    assert end == _resolved(_entry(scenario, plot))

    trend = trend_of(SIM, end)
    assert trend.end == end
    assert trend.hours[-1] == 0.0
    assert trend.hours[0] == -plots.WINDOW.total_seconds() / 3600.0


def test_cavity_peaks_are_the_scenarios_spikes_inside_the_window():
    channel = plots.CAVITY01_TEMPERATURE
    events = next(a for a in _scenario("rf-thermal")["archiver"] if a["channel"] == channel)
    baseline = json.loads((SIM / "machine.json").read_text())["channels"][channel]["value"]
    end = plots.entry_time(SIM, "rf-thermal", plots.RF_THERMAL_PLOT)
    expected = []
    for event in events["events"]:
        hours = (plots.event_instant(event) - end).total_seconds() / 3600.0
        if -plots.WINDOW.total_seconds() / 3600.0 <= hours <= 0.0:
            expected.append((hours, baseline + event["amplitude"]))
    peaks = plots.excursion_peaks(SIM, end)
    assert peaks == expected
    # The investigation entry narrates three excursions in the week before it.
    assert len(peaks) == 3

    trend = plots.cavity_temperatures(SIM, end)
    hours = np.asarray(trend.hours)
    values = np.asarray(trend.series[channel])
    for at, peak in peaks:
        near = values[np.abs(hours - at) <= 0.5]
        assert near.max() == pytest.approx(peak, abs=0.8), at


def test_orbit_rms_stays_inside_the_bpm_texture_envelope():
    channels = json.loads((SIM / "machine.json").read_text())["channels"]
    envelope_um = channels["SR:DIAG:BPM:01:POSITION:X"]["texture"]["amplitude"] * 1e6
    end = plots.entry_time(SIM, "nominal", plots.NOMINAL_PLOT)
    trend = plots.orbit_rms_week(SIM, end)
    for plane in ("X", "Y"):
        values = np.asarray(trend.series[plane])
        assert len(values) == 7 * 24 + 1
        assert 0.0 < values.min() and values.max() < envelope_um


def test_a_rerun_draws_the_same_bytes(tmp_path):
    sim = tmp_path / "simulation"
    shutil.copytree(SIM, sim)
    plots.main(["scenario_plots.py", str(sim)])
    first = {p.name: p.read_bytes() for p in sim.glob("scenarios/*/plots/*.png")}
    plots.main(["scenario_plots.py", str(sim)])
    second = {p.name: p.read_bytes() for p in sim.glob("scenarios/*/plots/*.png")}
    assert first == second
    assert set(first) == {
        Path(plots.BPM_POLARITY_PLOT).name,
        Path(plots.RF_THERMAL_PLOT).name,
        Path(plots.NOMINAL_PLOT).name,
    }
