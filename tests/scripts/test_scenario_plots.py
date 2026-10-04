"""The demo scenario pictures say what the scenarios' own definitions say.

``scripts/scenario_plots.py`` writes the pictures the control-assistant demo
scenarios attach to their logbook entries: a shipped bump-test PNG and two plot
specs the seeder draws at each entry's real dates. These tests pin the data
each carries to the scenario definition it is derived from, pin that nothing on
them names a finding, and pin that a re-run writes the same bytes.
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

from osprey.simulation.machine import parse_plot_spec
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
WINDOW_HOURS = plots.WINDOW.total_seconds() / 3600.0


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


def test_the_bump_test_draws_every_bpm_alike(monkeypatch, tmp_path):
    """Only the data tells the reversed BPM apart: same colour, title = BPM name."""
    from matplotlib.figure import Figure

    drawn: list[Figure] = []
    monkeypatch.setattr("osprey.simulation.plots.figure_png", lambda fig: drawn.append(fig) or b"")
    test = plots.bump_test(SIM)
    plots.draw_bump_test(test, tmp_path / "bump.png")

    (fig,) = drawn
    assert fig._suptitle.get_text() == f"{test.corrector} bump test, horizontal"
    axes = fig.axes
    assert [ax.get_title() for ax in axes] == list(test.bpms)
    styles = {
        tuple((line.get_color(), line.get_linestyle(), line.get_marker()) for line in ax.lines)
        for ax in axes
    }
    assert len(styles) == 1, "one BPM is drawn differently from the others"
    assert all(len(ax.texts) == 0 for ax in axes)


def _entry(scenario: str, spec: str) -> dict:
    """The one entry in ``scenario``'s logbook that attaches plot spec ``spec``."""
    entries = json.loads((SIM / "scenarios" / scenario / "logbook.json").read_text())
    (entry,) = [e for e in entries if any(a.get("plot") == spec for a in e.get("attachments", []))]
    return entry


def _resolved(entry: dict) -> datetime:
    when = entry["when"]
    spec = RelativeTimestamp(days_ago=when["days_ago"], time=dtime.fromisoformat(when["time"]))
    return resolve_relative_timestamp(spec, plots.ANCHOR)


def _shipped_spec(scenario: str, spec: str) -> dict:
    return json.loads((SIM / "scenarios" / scenario / spec).read_text())


@pytest.mark.parametrize(
    ("scenario", "spec", "spec_of", "step_hours"),
    [
        ("rf-thermal", plots.RF_THERMAL_SPEC, plots.cavity_temperatures_spec, 1 / 6),
        ("nominal", plots.NOMINAL_SPEC, plots.orbit_rms_spec, 1.0),
    ],
)
def test_each_spec_is_a_week_ending_at_its_entrys_time(scenario, spec, spec_of, step_hours):
    """A picture shows what its author could have seen: its window ends at the entry."""
    end = plots.entry_time(SIM, scenario, spec)
    assert end == _resolved(_entry(scenario, spec))

    written = spec_of(SIM, end)
    hours = written["hours_before"]
    assert hours[0] == WINDOW_HOURS
    assert hours[-1] == 0.0
    assert len(hours) == round(WINDOW_HOURS / step_hours) + 1
    assert written == _shipped_spec(scenario, spec), "the bundle's spec is stale; re-run"
    parse_plot_spec(written)


@pytest.mark.parametrize(
    ("scenario", "spec", "title", "ylabel", "labels"),
    [
        (
            "rf-thermal",
            plots.RF_THERMAL_SPEC,
            "CAVITY01 / CAVITY02 body temperature",
            "°C",
            ["CAVITY01", "CAVITY02"],
        ),
        ("nominal", plots.NOMINAL_SPEC, "SR orbit RMS", "µm", ["X", "Y"]),
    ],
)
def test_each_spec_says_only_what_a_strip_chart_export_says(scenario, spec, title, ylabel, labels):
    written = _shipped_spec(scenario, spec)
    assert written["title"] == title
    assert written["ylabel"] == ylabel
    assert [s["label"] for s in written["series"]] == labels


def test_cavity_maxima_are_the_scenarios_spikes_inside_the_window():
    channel = plots.CAVITY_TEMPERATURES[0]
    events = next(a for a in _scenario("rf-thermal")["archiver"] if a["channel"] == channel)
    baseline = json.loads((SIM / "machine.json").read_text())["channels"][channel]["value"]
    end = plots.entry_time(SIM, "rf-thermal", plots.RF_THERMAL_SPEC)
    expected = []
    for event in events["events"]:
        if event["shape"] != "spike":
            continue
        before = (end - plots.event_instant(event)).total_seconds() / 3600.0
        if 0.0 <= before <= WINDOW_HOURS:
            expected.append((before, baseline + event["amplitude"]))
    # The investigation entry narrates three excursions in the week before it.
    assert len(expected) == 3

    spec = _shipped_spec("rf-thermal", plots.RF_THERMAL_SPEC)
    hours = np.asarray(spec["hours_before"])
    values = np.asarray(spec["series"][0]["values"])
    for at, peak in expected:
        near = values[np.abs(hours - at) <= 0.5]
        assert near.max() == pytest.approx(peak, abs=0.8), at
    # Away from the spikes the cavity sits at its baseline.
    quiet = np.all([np.abs(hours - at) > 3.0 for at, _ in expected], axis=0)
    assert values[quiet].max() < baseline + 1.5


def test_orbit_rms_stays_inside_the_bpm_texture_envelope():
    channels = json.loads((SIM / "machine.json").read_text())["channels"]
    envelope_um = channels["SR:DIAG:BPM:01:POSITION:X"]["texture"]["amplitude"] * 1e6
    spec = _shipped_spec("nominal", plots.NOMINAL_SPEC)
    for series in spec["series"]:
        values = np.asarray(series["values"])
        assert len(values) == 7 * 24 + 1
        assert 0.0 < values.min() and values.max() < envelope_um


def test_a_rerun_writes_the_same_bytes(tmp_path):
    sim = tmp_path / "simulation"
    shutil.copytree(SIM, sim)
    outputs = [
        sim / "scenarios" / "bpm-polarity" / plots.BPM_POLARITY_PLOT,
        sim / "scenarios" / "rf-thermal" / plots.RF_THERMAL_SPEC,
        sim / "scenarios" / "nominal" / plots.NOMINAL_SPEC,
    ]
    plots.main(["scenario_plots.py", str(sim)])
    first = [p.read_bytes() for p in outputs]
    plots.main(["scenario_plots.py", str(sim)])
    assert [p.read_bytes() for p in outputs] == first
    # The bundle holds exactly what the script writes, byte for byte.
    assert first == [(SIM / p.relative_to(sim)).read_bytes() for p in outputs]
    assert sorted(p.name for p in sim.glob("scenarios/*/plots/*")) == sorted(
        p.name for p in outputs
    )
