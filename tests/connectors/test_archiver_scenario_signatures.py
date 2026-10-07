"""The control-assistant scenarios' archived signatures, read through the mock archiver.

The agentic scenario suites grade an agent's diagnosis with an LLM judge; a
diagnosis is only reachable if the archived data carries the signature the
scenario documents. This module pins those signatures with no LLM: each case
copies the control-assistant build's simulator view, pins one scenario in its
``active_scenarios`` state file and reads the history through
``MockArchiverConnector.get_data``, the path the agent's archiver tools take.

Signatures pinned:

- ``vacuum-burst``: in a 10-min window straddling 14:32:08, SR07 pressure and
  the DCCT current anti-correlate (Pearson r <= -0.75); SR07 is the most
  anti-correlated of the twelve sectors by more than 0.3; SR07 peaks above
  1.5e-7; the DCCT dips by 4-6.5 mA.
- ``rf-thermal``: the story its logbook tells -- three CAVITY01 excursions over
  32 degC in the week before the investigation entry, the last the trip four
  days back (34.2 degC, forward power off 03:15-04:30, reflected power over
  80 kW), nothing after the repair (26.5 degC).
- ``rf-thermal-live``: over 5-min windows at 1 s, temperature against reflected
  power wanders inside its band and against the tuner stays tight.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

from osprey.connectors.archiver.mock_archiver_connector import MockArchiverConnector
from osprey.connectors.control_system.mock_connector import simulation_state_dir
from osprey_connectors.config import get_facility_timezone
from osprey_connectors.relative_time import RelativeTimestamp, resolve_relative_timestamp
from tests.facility.served_tree import mock_config

GAUGE = "SR:VAC:GAUGE:SR{:02d}:PRESSURE:RB"
DCCT = "SR:DIAG:DCCT:01:CURRENT:RB"
CAV = "SR:RF:CAVITY:{:02d}:{}"
CAV_T = CAV.format(1, "TEMPERATURE:RB")
CAV_REV = CAV.format(1, "POWER:REV")
CAV_FWD = CAV.format(1, "POWER:FWD")
CAV_TUNER = CAV.format(1, "TUNER:RB")

#: The instant rf-thermal is applied at: the middle of a working day.
RF_ANCHOR = datetime(2026, 6, 13, 9, 30, tzinfo=UTC)

Reader = Callable[[list[str], datetime, datetime, timedelta], dict[str, np.ndarray]]


def _pinned_view(built_control_assistant, root: Path, state: str) -> Path:
    """A copy of the build's simulator view whose state file reads ``state``."""
    view = root / "build" / "data" / "simulator"
    shutil.copytree(built_control_assistant.build_dir / "data" / "simulator", view)
    state_dir = simulation_state_dir(view)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "active_scenarios").write_text(state)
    return view


def _reader(view: Path) -> Reader:
    """Read channels' archived values over ``[start, end]`` at one sample per ``step``."""

    async def _read(channels, start, end, step):
        connector = MockArchiverConnector()
        await connector.connect(mock_config(view, sample_rate_hz=1.0))
        try:
            frame = await connector.get_data(
                channels, start, end, precision_ms=int(step.total_seconds() * 1000)
            )
        finally:
            await connector.disconnect()
        return {
            channel: frame.loc[frame["channel"] == channel, "value"].to_numpy(dtype=float)
            for channel in channels
        }

    def read(channels, start, end, step):
        return asyncio.run(_read(channels, start, end, step))

    return read


def _r(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


class TestVacuumBurst:
    """SR07 pressure spike anti-correlates with the DCCT beam-current dip."""

    @pytest.fixture(scope="class")
    def window(self, built_control_assistant, tmp_path_factory) -> dict[str, np.ndarray]:
        view = _pinned_view(
            built_control_assistant, tmp_path_factory.mktemp("vacuum-burst"), "vacuum-burst\n"
        )
        center = (datetime.now(UTC) - timedelta(days=1)).replace(
            hour=14, minute=32, second=8, microsecond=0
        )
        channels = [GAUGE.format(sector) for sector in range(1, 13)] + [DCCT]
        return _reader(view)(
            channels,
            center - timedelta(minutes=5),
            center + timedelta(minutes=5),
            timedelta(seconds=1),
        )

    def test_sector7_anticorrelates_with_the_beam_current(self, window):
        r = _r(window[GAUGE.format(7)], window[DCCT])
        assert r <= -0.75, f"SR07/DCCT Pearson r = {r:.3f}, contract <= -0.75"

    def test_sector7_is_unambiguously_the_anomaly(self, window):
        """SR07 leads the runner-up sector by a wide margin.

        A separation is asserted rather than an absolute bound on the quiet
        sectors' |r|, which is a tail statistic on pure noise.
        """
        r = {sector: _r(window[GAUGE.format(sector)], window[DCCT]) for sector in range(1, 13)}
        ranked = sorted(r, key=r.__getitem__)
        assert ranked[0] == 7, f"most anti-correlated sector is SR{ranked[0]:02d}, expected SR07"
        separation = abs(r[7]) - abs(r[ranked[1]])
        assert separation > 0.3, f"SR07 leads the next sector by {separation:.3f}, contract > 0.3"

    def test_the_spike_and_the_dip_have_their_size(self, window):
        sr07 = window[GAUGE.format(7)]
        dcct = window[DCCT]
        assert sr07.max() > 1.5e-7, f"SR07 peak {sr07.max():.3e}, contract > 1.5e-7"
        dip = float(np.median(dcct) - dcct.min())
        assert 4.0 < dip < 6.5, f"DCCT dip = {dip:.3f} mA, contract 4.0 < dip < 6.5"


class TestRfThermal:
    """The rf-thermal history tells the story its logbook entries tell."""

    @pytest.fixture(scope="class")
    def scenario(self, built_control_assistant) -> dict:
        path = built_control_assistant.facility_dir / "scenarios" / "rf-thermal.yaml"
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    @pytest.fixture(scope="class")
    def read(self, built_control_assistant, tmp_path_factory) -> Reader:
        state = f"anchor={RF_ANCHOR.isoformat()}\nrf-thermal\n"
        return _reader(
            _pinned_view(built_control_assistant, tmp_path_factory.mktemp("rf-thermal"), state)
        )

    @staticmethod
    def _narrated(when: dict) -> datetime:
        """The instant a ``{days_ago, time}`` stamp resolves to against the anchor."""
        clock = when["time"]
        spec = RelativeTimestamp(
            days_ago=when["days_ago"],
            time=clock if isinstance(clock, time) else time.fromisoformat(str(clock)),
        )
        return resolve_relative_timestamp(spec, RF_ANCHOR.astimezone(get_facility_timezone()))

    def _entry(self, scenario: dict, entry_id: str) -> datetime:
        when = next(entry["when"] for entry in scenario["logbook"] if entry["entry_id"] == entry_id)
        return self._narrated(when)

    def _spikes(self, scenario: dict, channel: str) -> list[datetime]:
        events = next(item["events"] for item in scenario["archiver"] if item["channel"] == channel)
        return sorted(self._narrated(e["at_when"]) for e in events if e["shape"] == "spike")

    def _trip_day(self, scenario: dict, hh: int, mm: int) -> datetime:
        """A clock time on the day the trip entry narrates."""
        return self._entry(scenario, "DEMO-026").replace(hour=hh, minute=mm, second=0)

    def test_three_excursions_over_32_in_the_week_before_the_investigation(self, scenario, read):
        spikes = self._spikes(scenario, CAV_T)
        end = self._entry(scenario, "DEMO-027")
        assert len(spikes) == 3
        assert all(end - timedelta(days=7) <= at <= end for at in spikes)
        for at in spikes:
            near = read(
                [CAV_T],
                at - timedelta(minutes=20),
                at + timedelta(minutes=20),
                timedelta(minutes=1),
            )[CAV_T]
            assert near.max() > 32.0, f"no CAVITY01 excursion over 32 degC at {at}"

    def test_the_last_excursion_is_the_trip_at_34_2_degc(self, scenario, read):
        last = self._spikes(scenario, CAV_T)[-1]
        assert self._trip_day(scenario, 1, 0) < last < self._trip_day(scenario, 3, 15)
        peak = read(
            [CAV_T], last - timedelta(minutes=5), last + timedelta(minutes=5), timedelta(seconds=10)
        )[CAV_T].max()
        assert peak == pytest.approx(34.2, abs=0.3)
        assert peak < 35.0, "the narrated trip stayed below the thermal interlock"
        before = read(
            [CAV_T],
            self._trip_day(scenario, 0, 40),
            self._trip_day(scenario, 0, 45),
            timedelta(seconds=30),
        )[CAV_T]
        after = read(
            [CAV_T],
            self._trip_day(scenario, 4, 30),
            self._trip_day(scenario, 4, 35),
            timedelta(seconds=30),
        )[CAV_T]
        assert before.max() < 28.5, "the climb starts around 01:00, not before"
        assert after.max() < 30.0, "below 30 degC again by the 04:30 recovery"

    def test_forward_power_is_off_from_0315_to_0430_on_the_trip_night(self, scenario, read):
        off = read(
            [CAV_FWD],
            self._trip_day(scenario, 3, 17),
            self._trip_day(scenario, 4, 28),
            timedelta(minutes=1),
        )[CAV_FWD]
        assert off.max() < 5.0, f"forward power {off.max():.1f} kW during the trip"
        for hh, mm in ((1, 0), (5, 30)):
            on = read(
                [CAV_FWD],
                self._trip_day(scenario, hh, mm),
                self._trip_day(scenario, hh, mm) + timedelta(minutes=1),
                timedelta(seconds=6),
            )[CAV_FWD]
            assert on.min() > 400.0, f"forward power {on.min():.1f} kW at {hh:02d}:{mm:02d}"

    def test_reflected_power_passes_80_kw_before_the_interlock(self, scenario, read):
        reflected = read(
            [CAV_REV],
            self._trip_day(scenario, 2, 45),
            self._trip_day(scenario, 3, 15),
            timedelta(minutes=1),
        )[CAV_REV]
        assert reflected.max() > 80.0

    def test_nothing_happens_after_the_repair(self, scenario, read):
        after = read(
            [CAV_T, CAV_FWD],
            self._entry(scenario, "DEMO-028") + timedelta(hours=1),
            RF_ANCHOR,
            timedelta(minutes=5),
        )
        assert after[CAV_T].mean() == pytest.approx(26.5, abs=0.1)
        assert after[CAV_T].max() < 27.5
        assert after[CAV_FWD].min() > 400.0


class TestRfThermalLive:
    """The live thermal wander's correlation bands, on 1 s samples over 5-min windows."""

    WINDOW = 300
    #: The end of the three hours the bands are read over.
    END = datetime.fromtimestamp(1_790_000_000, UTC)

    @pytest.fixture(scope="class")
    def series(self, built_control_assistant, tmp_path_factory) -> dict[str, np.ndarray]:
        read = _reader(
            _pinned_view(
                built_control_assistant,
                tmp_path_factory.mktemp("rf-thermal-live"),
                "rf-thermal-live\n",
            )
        )
        channels = [CAV_T, CAV_REV, CAV_FWD, CAV_TUNER]
        # The mock serves at most 10,000 points a call: read the hours in halves.
        halves = [
            read(channels, start, start + timedelta(seconds=5399), timedelta(seconds=1))
            for start in (self.END - timedelta(hours=3), self.END - timedelta(hours=1.5))
        ]
        return {channel: np.concatenate([half[channel] for half in halves]) for channel in channels}

    def _windowed(self, x: np.ndarray, y: np.ndarray) -> list[float]:
        n, w = len(x), self.WINDOW
        return [_r(x[s : s + w], y[s : s + w]) for s in range(0, n - w + 1, 30)]

    def test_temperature_against_reflected_power_wanders_in_band(self, series):
        rs = self._windowed(series[CAV_T], series[CAV_REV])
        assert 0.35 < min(rs) and max(rs) < 0.95
        assert 0.6 < float(np.median(rs)) < 0.85
        assert max(rs) - min(rs) > 0.2

    def test_temperature_against_the_tuner_is_tight(self, series):
        rs = self._windowed(series[CAV_T], series[CAV_TUNER])
        assert min(rs) > 0.9 and max(rs) < 0.995

    def test_levels_stay_physical(self, series):
        assert series[CAV_REV].min() > 0.0
        assert 26.0 < series[CAV_T].min() and series[CAV_T].max() < 28.0
