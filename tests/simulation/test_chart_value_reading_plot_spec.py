"""The beam-loss comparison fixture: the picture's numbers come from one module.

The tier-2 benchmark in ``tests/e2e/tier2/test_chart_value_reading.py`` grades an
agent's reading of a plot against ``DIP_DEPTH_MA``. These tests pin the
picture to that constant without a deploy: the spec parses, the drop at the
loss instant is the constant, the dip ends on a refill so the sawtooth never
jumps inside it, and nothing on the picture or in the entry texts states a
magnitude.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.e2e.fixtures.chart_value_reading import plot_spec as fixture

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _values() -> list[float]:
    return fixture.build_plot_spec()["series"][0]["values"]


def test_axis_is_non_increasing_and_ends_at_zero() -> None:
    hours = fixture.hours_axis()
    assert hours[-1] == 0.0
    assert hours[0] == fixture.WINDOW_H
    assert all(later <= earlier for earlier, later in zip(hours, hours[1:], strict=False))
    assert len(hours) == len(_values())


def test_spec_parses_with_the_bundle_loader() -> None:
    from osprey.simulation.machine import parse_plot_spec

    spec = parse_plot_spec(fixture.build_plot_spec(), "beam-loss fixture")
    assert spec.filename == "dcct_decay_run.png"
    assert spec.ylim == (0.0, 600.0)


def test_drop_at_the_loss_instant_is_the_planted_depth() -> None:
    hours = fixture.hours_axis()
    values = _values()
    # The first sample inside the dip and the last one before it.
    inside = next(i for i, h in enumerate(hours) if h <= fixture.DIP_HOURS_BEFORE)
    drop = values[inside - 1] - values[inside]
    # One minute of sawtooth decay separates the two samples; noise is 0.3 mA.
    assert abs(drop - fixture.DIP_DEPTH_MA) < 2.0, drop


def test_dip_ends_on_a_refill_so_the_sawtooth_never_jumps_inside_it() -> None:
    elapsed_end = fixture.WINDOW_H - (fixture.DIP_HOURS_BEFORE - fixture.DIP_DURATION_H)
    assert elapsed_end % fixture.REFILL_PERIOD_H == pytest.approx(0.0, abs=1e-9)
    elapsed_start = fixture.WINDOW_H - fixture.DIP_HOURS_BEFORE
    refills_inside = [
        k * fixture.REFILL_PERIOD_H
        for k in range(1, 20)
        if elapsed_start < k * fixture.REFILL_PERIOD_H < elapsed_end
    ]
    assert refills_inside == []


def test_every_honest_reading_of_the_loss_is_near_the_planted_depth() -> None:
    # "How big was the loss" has three honest readings off the picture: the step
    # at the loss instant, the dip's trough against a normal trough, and the
    # nominal current down to the dip's bottom. The sawtooth makes the last one
    # the largest; it must stay close enough to DIP_DEPTH_MA that an agent who
    # reads it correctly is not graded as a misread.
    hours = fixture.hours_axis()
    values = _values()
    low = min(
        v
        for h, v in zip(hours, values, strict=True)
        if fixture.DIP_HOURS_BEFORE - fixture.DIP_DURATION_H < h <= fixture.DIP_HOURS_BEFORE
    )
    nominal_to_bottom = fixture.BASELINE_MA - low
    assert nominal_to_bottom <= 1.15 * fixture.DIP_DEPTH_MA, (nominal_to_bottom, low)


def test_the_larger_loss_follows_from_the_constants_by_a_clear_margin() -> None:
    hours = fixture.hours_axis()
    values = _values()
    low = min(
        v
        for h, v in zip(hours, values, strict=True)
        if fixture.DIP_HOURS_BEFORE - fixture.DIP_DURATION_H < h <= fixture.DIP_HOURS_BEFORE
    )
    readings = (fixture.DIP_DEPTH_MA, fixture.BASELINE_MA - low)
    expected = "recent" if fixture.RECENT_DIP_MA > fixture.DIP_DEPTH_MA else "logged"
    assert fixture.LARGER_LOSS == expected
    # Every honest reading of the plot sits on the same side of the recent dip,
    # at least a tenth away, so the comparison never hinges on reading precision.
    for reading in readings:
        assert abs(fixture.RECENT_DIP_MA - reading) >= 0.10 * fixture.RECENT_DIP_MA, readings
        assert (fixture.RECENT_DIP_MA > reading) == (expected == "recent"), readings


def test_build_is_deterministic() -> None:
    assert fixture.build_plot_spec() == fixture.build_plot_spec()


def test_picture_text_states_no_number() -> None:
    spec = fixture.build_plot_spec()
    texts = [spec["title"], spec["ylabel"], *(s["label"] for s in spec["series"])]
    assert not any(re.search(r"\d", t) for t in texts if t != fixture.DCCT_CHANNEL), texts


def test_entry_texts_state_no_current_in_milliamps() -> None:
    entries = json.loads(
        (fixture.FIXTURE_DIR / fixture.LOGBOOK_BUNDLE / "logbook.json").read_text("utf-8")
    )
    for entry in entries:
        text = f"{entry['title']}\n{entry['text']}"
        assert not re.search(r"\d+(\.\d+)?\s?mA", text), entry["entry_id"]


def test_recent_bundle_steps_match_the_constant() -> None:
    scenario = json.loads(
        (fixture.FIXTURE_DIR / fixture.RECENT_BUNDLE / "scenario.json").read_text("utf-8")
    )
    (channel,) = scenario["archiver"]
    assert channel["channel"] == fixture.DCCT_CHANNEL
    down, up = channel["events"]
    assert down["shape"] == up["shape"] == "step"
    assert down["at_offset"] < up["at_offset"] < 0
    assert fixture.BASELINE_MA - down["to"] == pytest.approx(fixture.RECENT_DIP_MA)
    assert up["to"] == pytest.approx(fixture.BASELINE_MA)


def test_installed_bundles_load_as_narratives(tmp_path: Path) -> None:
    from osprey.simulation.machine import PlotSpec, load_narratives

    scenarios = tmp_path / "scenarios"
    fixture.install_bundles(scenarios)
    assert (scenarios / fixture.LOGBOOK_BUNDLE / "plots" / "dcct_decay_run.json").is_file()
    narratives = load_narratives(scenarios)
    entries = {e.entry_id: e for e in narratives[fixture.LOGBOOK_BUNDLE]}
    target = entries[fixture.TARGET_ENTRY_ID]
    assert len(target.attachments) == 1 and isinstance(target.attachments[0], PlotSpec)
    assert all(not e.attachments for k, e in entries.items() if k != fixture.TARGET_ENTRY_ID)
    # The recent bundle is telemetry only: it contributes no narrative at all.
    assert fixture.RECENT_BUNDLE not in narratives


def test_spec_renders_to_a_png(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from osprey.simulation.machine import parse_plot_spec
    from osprey.simulation.plots import render_plot_spec

    spec = parse_plot_spec(fixture.build_plot_spec(), "beam-loss fixture")
    png = render_plot_spec(spec, datetime(2025, 10, 3, 2, 40, tzinfo=UTC))
    assert png.startswith(PNG_SIGNATURE)
    (tmp_path / "dcct_decay_run.png").write_bytes(png)
