"""The rf-thermal beam dump rides on the top-off baseline.

The scenario states only the drop: one bounded step of the DCCT to zero over
the dump window. Inside the window the stored current reads zero within its
noise; after it the current is back on the baseline, so the stored history
carries the top-off sawtooth again, and inside one tooth the current decays at
the rate the lifetime readback names.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest
import yaml

from osprey.facility.sources import read_yaml
from osprey_connectors.archiver._field_names import field_name
from osprey_connectors.config import get_facility_timezone
from osprey_connectors.simulation import topoff
from osprey_connectors.simulation.archive import ArchiveComposite, _synthesize_documents, build
from osprey_connectors.simulation.series import anchored_instant, step_end

DCCT = "SR:DIAG:DCCT:01:CURRENT:RB"
LIFETIME = "SR:DIAG:DCCT:01:LIFETIME:RB"
#: The instant rf-thermal is applied at: the middle of a working day.
RF_ANCHOR_S = datetime(2026, 6, 13, 9, 30, tzinfo=UTC).timestamp()
#: The stored archive's hot cadence, in seconds.
CADENCE_S = 10.0
#: How many noise standard deviations a reading may stray before it is not noise.
NOISE_SIGMAS = 6.0


@pytest.fixture(scope="module")
def facility(built_control_assistant) -> dict[str, Any]:
    root = built_control_assistant.facility_dir
    return {
        "seeds": read_yaml((root / "seeds.yaml").read_text(encoding="utf-8")),
        "scenario": yaml.safe_load(
            (root / "scenarios" / "rf-thermal.yaml").read_text(encoding="utf-8")
        ),
    }


@pytest.fixture(scope="module")
def archive(built_control_assistant) -> ArchiveComposite:
    view = built_control_assistant.build_dir / "data" / "simulator"
    return build(view, ["rf-thermal"], anchor_s=RF_ANCHOR_S)


@pytest.fixture(scope="module")
def dump(facility) -> dict[str, Any]:
    """The scenario's one DCCT event and the window it holds the current over."""
    (event,) = next(
        item["events"] for item in facility["scenario"]["archiver"] if item["channel"] == DCCT
    )
    tz = get_facility_timezone()
    start = anchored_instant(event, RF_ANCHOR_S, tz)
    end = step_end(event, RF_ANCHOR_S, tz)
    assert end is not None, "the dump is a bounded step"
    return {"event": event, "start": start, "end": end}


def _noise(seed: dict[str, Any]) -> float:
    return float(seed["noise"]["absolute"])


def test_the_dump_is_one_bounded_step_to_zero(dump) -> None:
    event = dump["event"]
    assert event["shape"] == "step"
    assert event["to"] == 0.0
    assert dump["end"] - dump["start"] == pytest.approx(75 * 60.0)


def test_inside_the_window_the_dcct_reads_zero_within_its_noise(facility, archive, dump) -> None:
    times = np.arange(dump["start"] + CADENCE_S, dump["end"] - CADENCE_S, CADENCE_S)
    current = np.asarray(archive.series(DCCT, times))
    band = NOISE_SIGMAS * _noise(facility["seeds"][DCCT])
    assert np.all(np.abs(current) <= band), float(np.abs(current).max())


def test_after_the_recovery_the_stored_current_carries_the_sawtooth(
    facility, archive, dump
) -> None:
    seed = facility["seeds"][DCCT]
    record = seed["topoff"]
    held = topoff.held_value(seed)
    assert held is not None
    # One hour of stored documents, starting a minute after the recovery.
    epoch = np.arange(dump["end"] + 60.0, dump["end"] + 3660.0, CADENCE_S)
    field = field_name(DCCT)
    stored = np.array([document[field] for document in _synthesize_documents(archive, epoch)])
    # A full tooth: from a refill to just before the next.
    full_tooth = record["ceiling"] * (1.0 - topoff.shape(record, record["period_s"] - 1e-6))
    assert np.ptp(stored) >= full_tooth, (float(np.ptp(stored)), full_tooth)
    assert np.ptp(stored) <= full_tooth + 2 * NOISE_SIGMAS * _noise(seed)
    assert float(np.mean(stored)) == pytest.approx(held, abs=3 * _noise(seed))


def test_inside_one_tooth_ln_current_decays_at_one_over_the_lifetime(
    facility, archive, dump
) -> None:
    record = facility["seeds"][DCCT]["topoff"]
    period = float(record["period_s"])
    # The first whole tooth an hour after the recovery: refill to next refill.
    refill = (np.floor((dump["end"] + 3600.0) / period) + 1.0) * period
    times = refill + np.linspace(0.5, period - 0.5, 601)
    current = np.asarray(archive.series(DCCT, times))
    lifetime_h = np.asarray(archive.series(LIFETIME, times))
    assert np.ptp(lifetime_h) == 0.0, "the lifetime readback holds over a tooth"
    rate = -np.polyfit(times - refill, np.log(current), 1)[0]
    expected = 1.0 / (float(lifetime_h[0]) * topoff.SECONDS_PER_HOUR)
    assert rate == pytest.approx(expected, rel=0.05), (rate, expected)
