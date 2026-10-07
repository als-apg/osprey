"""Statistical contract for the control_assistant simulation scenarios.

The e2e scenario tests (``test_vacuum_burst_scenario``,
``test_rf_cavity_correlation_scenario``) grade agent *diagnoses* with an LLM
judge; those diagnoses are only reachable if the synthesized data carries the
documented signatures. This file pins the signatures deterministically (no
LLM, fast) so the expensive e2e judge layers run against a known-good data
substrate. If a contract here fails, the fix is to tune ``machine.json``
amplitudes/widths — never to weaken the e2e prompts.

Signatures pinned (target value -> asserted threshold, with margin):

- ``vacuum-burst``: SR07 vs DCCT Pearson r ~= -0.89 (assert <= -0.75) in a
  10-min window straddling 14:32:08; SR07 is the single most anti-correlated
  sector, leading the runner-up by ~0.7 (assert separation > 0.3 — robust to
  the noise tail, unlike an absolute |r| bound); SR07 spike ~4x baseline;
  DCCT dips ~5 mA.
- ``rf-thermal``: the story its logbook tells -- three CAVITY01 excursions
  over 32 degC in the week before the investigation entry, the last the trip
  four days back (34.2 degC, forward power off 03:15-04:30, reflected power over
  80 kW), nothing after the repair (26.5 degC); CAVITY02 inside 25-28.5 degC;
  reflected power tracks temperature (assert r > 0.8); POWER:NET is FWD-REV;
  FREQUENCY:RB detunes downward during the excursions.
"""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from osprey.cli.build_profile_archiver import VAArchiverConfig
from tests.simulation.conftest import TEMPLATE_SIM

# The shipped control_assistant engine is built by the shared ``engine_factory``
# fixture in conftest.py (copies machine.json + the scenarios/ bundle tree).
GAUGE = "SR:VAC:GAUGE:SR{:02d}:PRESSURE:RB"
DCCT = "SR:DIAG:DCCT:01:CURRENT:RB"


def _window(center: datetime, minutes: int = 10, step_s: int = 1) -> list[datetime]:
    """Return per-second timestamps for a window centered on ``center``."""
    start = center - timedelta(minutes=minutes / 2)
    return [start + timedelta(seconds=i) for i in range(minutes * 60 // step_s)]


def _yesterday_event() -> datetime:
    """Yesterday 14:32:08 in the facility zone (UTC in tests) — the daily ``at_time``
    anchor fires on any past date. Building the window tz-aware in the facility zone
    keeps the contract independent of the deploy host's ``$TZ``."""
    day = datetime.now(UTC) - timedelta(days=1)
    return day.replace(hour=14, minute=32, second=8, microsecond=0)


class TestVacuumBurstContract:
    """SR07 pressure spike anti-correlates with the DCCT beam-current dip."""

    def test_sector7_dcct_anticorrelation(self, engine_factory):
        """SR07 vs DCCT Pearson r <= -0.75 (target ~ -0.89)."""
        engine = engine_factory("vacuum-burst")
        ts = _window(_yesterday_event())
        sr07 = np.array(engine.synthesize_series(GAUGE.format(7), ts))
        dcct = np.array(engine.synthesize_series(DCCT, ts))
        r = np.corrcoef(sr07, dcct)[0, 1]
        assert r <= -0.75, f"SR07/DCCT Pearson r = {r:.3f}, contract <= -0.75"

    def test_sector7_is_unambiguously_the_anomaly(self, engine_factory):
        """SR07 is the single most anti-correlated sector, by a wide margin.

        This is the contract the e2e agent actually has to satisfy: pick SR07
        out of all 12 sectors as *the* one correlated with the beam loss. A
        separation contract (SR07 leads the field) is asserted instead of an
        absolute ``max |r| < threshold`` on the quiet sectors, because the
        latter is a tail-sensitive statistic on pure noise — over 3000 trials
        a quiet sector's |r| occasionally reaches ~0.16, while SR07's lead
        over the runner-up never drops below ~0.7. The wide gap is the robust,
        non-flaky invariant.
        """
        engine = engine_factory("vacuum-burst")
        ts = _window(_yesterday_event())
        dcct = np.array(engine.synthesize_series(DCCT, ts))
        r = {
            s: np.corrcoef(np.array(engine.synthesize_series(GAUGE.format(s), ts)), dcct)[0, 1]
            for s in range(1, 13)
        }
        ranked = sorted(range(1, 13), key=lambda s: r[s])  # most anti-correlated first
        assert ranked[0] == 7, f"most anti-correlated sector is SR{ranked[0]:02d}, expected SR07"
        runner_up = abs(r[ranked[1]])
        separation = abs(r[7]) - runner_up
        assert separation > 0.3, (
            f"SR07 |r|={abs(r[7]):.3f} vs next sector |r|={runner_up:.3f} "
            f"(separation {separation:.3f}), contract > 0.3"
        )

    def test_spike_and_dip_magnitudes(self, engine_factory):
        """SR07 spikes ~4x baseline and the DCCT dips ~5 mA."""
        engine = engine_factory("vacuum-burst")
        ts = _window(_yesterday_event())
        sr07 = np.array(engine.synthesize_series(GAUGE.format(7), ts))
        dcct = np.array(engine.synthesize_series(DCCT, ts))
        assert sr07.max() > 1.5e-7, f"SR07 peak {sr07.max():.3e}, contract > 1.5e-7"
        dip = 500.0 - dcct.min()
        assert 4.0 < dip < 6.5, f"DCCT dip = {dip:.3f} mA, contract 4.0 < dip < 6.5"

    def test_quiet_window_is_flat(self, engine_factory):
        """A morning window (09:32) shows no SR07 event (max < baseline+noise)."""
        engine = engine_factory("vacuum-burst")
        ts = _window(_yesterday_event().replace(hour=9))
        sr07 = np.array(engine.synthesize_series(GAUGE.format(7), ts))
        assert sr07.max() < 1.0e-7, f"quiet-window SR07 peak {sr07.max():.3e}, contract < 1e-7"

    def test_nominal_scenario_has_no_event(self, engine_factory):
        """The nominal scenario shows no SR07 spike even at 14:32."""
        engine = engine_factory("nominal")
        ts = _window(_yesterday_event())
        sr07 = np.array(engine.synthesize_series(GAUGE.format(7), ts))
        assert sr07.max() < 1.0e-7, f"nominal SR07 peak {sr07.max():.3e}, contract < 1e-7"


class TestRfThermalContract:
    """The rf-thermal telemetry tells the story its logbook entries tell.

    The bundle places its events with ``at_when`` -- the logbook's own
    ``{days_ago, time}`` -- so every check here resolves both the events and the
    entries against one explicit anchor and asserts at the instants the entries
    narrate: three CAVITY01 excursions over 32 degC in the week before the
    investigation (DEMO-027), the last one the trip of DEMO-026 (a climb from
    about 01:00 to 34.2 degC, below the 35 degC interlock, reflected power over
    80 kW, forward power at zero from 03:15 to 04:30, back under 30 degC by
    04:30), and nothing after the repair of DEMO-028.
    """

    CAV = "SR:RF:CAVITY:{:02d}:{}"
    #: An apply-time anchor in the middle of a working day.
    T0 = datetime(2026, 6, 13, 9, 30, tzinfo=UTC)

    @pytest.fixture
    def engine(self, engine_factory):
        engine = engine_factory("nominal")
        engine.set_active_scenarios(["rf-thermal"], anchor=self.T0)
        return engine

    def _zone(self):
        from osprey.utils.config import get_facility_timezone

        return get_facility_timezone()

    def _entry_time(self, entry_id: str) -> datetime:
        from osprey.utils.relative_time import RelativeTimestamp, resolve_relative_timestamp

        entries = json.loads((TEMPLATE_SIM / "scenarios/rf-thermal/logbook.json").read_text())
        when = next(e["when"] for e in entries if e["entry_id"] == entry_id)
        spec = RelativeTimestamp(
            days_ago=when["days_ago"], time=datetime.strptime(when["time"], "%H:%M:%S").time()
        )
        return resolve_relative_timestamp(spec, self.T0.astimezone(self._zone()))

    def _trip_day(self, hh: int, mm: int) -> datetime:
        """A clock time on the day DEMO-026 narrates."""
        return self._entry_time("DEMO-026").replace(hour=hh, minute=mm, second=0)

    def _spikes(self, channel: str) -> list[tuple[datetime, float]]:
        """``(instant, amplitude)`` of the bundle's spikes on ``channel``."""
        from osprey.simulation.series import anchored_instant

        bundle = json.loads((TEMPLATE_SIM / "scenarios/rf-thermal/scenario.json").read_text())
        events = next(a["events"] for a in bundle["archiver"] if a["channel"] == channel)
        zone = self._zone()
        return [
            (
                datetime.fromtimestamp(anchored_instant(e, self.T0.timestamp(), zone), UTC),
                e["amplitude"],
            )
            for e in events
            if e["shape"] == "spike"
        ]

    def _read(self, engine, dev, suffix, times):
        return np.array(engine.synthesize_series(self.CAV.format(dev, suffix), times))

    def _span(self, start: datetime, end: datetime, step: timedelta) -> list[datetime]:
        count = int((end - start) / step)
        return [start + step * i for i in range(count + 1)]

    def _week_before_investigation(self) -> list[datetime]:
        end = self._entry_time("DEMO-027")
        return self._span(end - timedelta(days=7), end, timedelta(minutes=5))

    def test_three_excursions_over_32_in_the_week_before_the_investigation(self, engine):
        spikes = self._spikes(self.CAV.format(1, "TEMPERATURE:RB"))
        end = self._entry_time("DEMO-027")
        inside = [at for at, _ in spikes if end - timedelta(days=7) <= at <= end]
        assert len(inside) == len(spikes) == 3
        for at, _amplitude in spikes:
            near = self._read(
                engine,
                1,
                "TEMPERATURE:RB",
                self._span(
                    at - timedelta(minutes=20), at + timedelta(minutes=20), timedelta(minutes=1)
                ),
            )
            assert near.max() > 32.0, f"no CAVITY01 excursion over 32 degC at {at}"

    def test_the_last_excursion_is_the_trip_on_the_reported_night(self, engine):
        spikes = self._spikes(self.CAV.format(1, "TEMPERATURE:RB"))
        last, amplitude = max(spikes)
        assert self._trip_day(1, 0) < last < self._trip_day(3, 15)
        peak = 27.0 + amplitude
        assert peak == pytest.approx(34.2, abs=0.05)
        assert peak < 35.0, "the narrated trip stayed below the thermal interlock"

        temperature = self._read(
            engine, 1, "TEMPERATURE:RB", [self._trip_day(0, 45), last, self._trip_day(4, 30)]
        )
        assert temperature[0] < 28.5, "the climb starts around 01:00, not before"
        assert temperature[1] > 33.0
        assert temperature[2] < 30.0, "below 30 degC again by the 04:30 recovery"

    def test_forward_power_is_off_from_0315_to_0430_on_the_trip_night(self, engine):
        off = self._read(
            engine,
            1,
            "POWER:FWD",
            self._span(self._trip_day(3, 17), self._trip_day(4, 28), timedelta(minutes=1)),
        )
        assert off.max() < 5.0, f"forward power {off.max():.1f} kW during the trip"
        on = self._read(engine, 1, "POWER:FWD", [self._trip_day(1, 0), self._trip_day(5, 30)])
        assert on.min() > 400.0

    def test_reflected_power_passes_80_kw_before_the_interlock(self, engine):
        reflected = self._read(
            engine,
            1,
            "POWER:REV",
            self._span(self._trip_day(2, 45), self._trip_day(3, 15), timedelta(minutes=1)),
        )
        assert reflected.max() > 80.0

    def test_nothing_happens_after_the_repair(self, engine):
        after = self._span(
            self._entry_time("DEMO-028") + timedelta(hours=1), self.T0, timedelta(minutes=5)
        )
        temperature = self._read(engine, 1, "TEMPERATURE:RB", after)
        assert temperature.mean() == pytest.approx(26.5, abs=0.1)
        assert temperature.max() < 27.5
        assert self._read(engine, 1, "POWER:FWD", after).min() > 400.0
        live = engine.read(self.CAV.format(1, "TEMPERATURE:RB")).value
        assert live == pytest.approx(26.5, abs=1.0)

    def test_cavity01_is_flat_before_the_excursion_week(self, engine):
        end = self._entry_time("DEMO-027") - timedelta(days=7)
        quiet = self._read(
            engine,
            1,
            "TEMPERATURE:RB",
            self._span(end - timedelta(days=7), end, timedelta(minutes=5)),
        )
        assert quiet.max() < 29.0

    def test_cavity02_stays_inside_its_normal_band(self, engine):
        """DEMO-026/027: CAVITY02 stays at 25-28 degC; its one bump is minor."""
        temperature = self._read(engine, 2, "TEMPERATURE:RB", self._week_before_investigation())
        assert temperature.max() < 28.5, f"CAVITY02 peak {temperature.max():.2f}"
        assert temperature.min() > 25.0

    def test_reflected_power_spikes_with_temperature(self, engine):
        times = self._week_before_investigation()
        temperature = self._read(engine, 1, "TEMPERATURE:RB", times)
        reflected = self._read(engine, 1, "POWER:REV", times)
        r = np.corrcoef(temperature, reflected)[0, 1]
        assert r > 0.8, f"TEMP/REV correlation r = {r:.3f}, contract > 0.8"

    def test_net_power_is_fwd_minus_rev(self, engine):
        """POWER:NET is the live expression FWD - REV, proven exactly.

        A correlation test on separately-synthesized series cannot falsify a
        wrong ``FWD + REV`` formula — the excursion trips dominate the noise,
        so both signs correlate. Instead this writes FWD and REV and reads NET
        back: a noise-free derived channel must yield *exactly* FWD - REV,
        which ``FWD + REV`` could not. The archived NET history is then checked
        to collapse alongside the forward-power trips.
        """
        fwd_pv = self.CAV.format(1, "POWER:FWD")
        rev_pv = self.CAV.format(1, "POWER:REV")
        net_pv = self.CAV.format(1, "POWER:NET")
        engine.write(fwd_pv, 300.0)
        engine.write(rev_pv, 50.0)
        net = engine.read(net_pv).value
        assert net == pytest.approx(250.0), f"NET={net}, expected FWD-REV=250.0 (not FWD+REV=350)"

        times = self._week_before_investigation()
        fwd = self._read(engine, 1, "POWER:FWD", times)
        net_series = self._read(engine, 1, "POWER:NET", times)
        trip = int(np.argmin(fwd))
        assert net_series[trip] < net_series.mean() - 100.0, (
            f"NET at the forward-power trip ({net_series[trip]:.1f}) does not collapse "
            f"below its mean ({net_series.mean():.1f})"
        )

    def test_frequency_detunes_during_excursions(self, engine):
        freq = self._read(engine, 1, "FREQUENCY:RB", self._week_before_investigation())
        assert freq.min() < 499.654 - 0.0005, (
            f"CAVITY01 freq min {freq.min():.6f}, contract < 499.6535"
        )


class TestEventsFitTheSeedWindow:
    """Every shipped scenario event is placeable inside the archiver's seed window.

    The archiver is seeded once at deploy time over ``[T0 - horizon, T0]`` and
    scenario activation later rewrites only the windows its events touch. An
    event outside that span has nowhere to be written, so it would silently go
    missing from stored history while still showing up in the mock's on-the-fly
    synthesis — exactly the divergence this whole feature exists to remove.
    These are data contracts on the shipped bundles, so they hold for any deploy
    rather than for one engine instance.

    **Anchoring assumption.** Two separate clocks are involved: the deploy-time
    seed anchor and the ``osprey sim apply`` anchor. This property holds only
    because they are one and the same — the seeder records its T0 in the
    seed-manifest document and scenario activation resolves ``at_offset`` and
    ``at_when`` against that persisted anchor, not against its own wall clock. If those two
    ever drift apart, an event within ``horizon`` of the apply anchor can still
    land outside the seeded span, and this test no longer proves what it claims.
    """

    # Seed span: bound to the shipped ``va_archiver.retention_days`` default
    # rather than restated, so lowering that default in the profile block makes
    # this test fail instead of quietly passing against a window the seeder no
    # longer covers.
    SEED_HORIZON_S = VAArchiverConfig().retention_days * 24 * 3600.0
    # Fraction of a Gaussian sigma an anchored spike may extend past T0. Spikes
    # are symmetric, so an event centred too close to T0 has half its shape in
    # the unseedable future; 3 sigma covers essentially all of it.
    SIGMA_MARGIN = 3.0

    def _bundles(self):
        """(name, parsed scenario.json) for every shipped control_assistant bundle."""
        roots = sorted(p for p in (TEMPLATE_SIM / "scenarios").iterdir() if p.is_dir())
        assert roots, "no scenario bundles found — the template layout moved"
        return [(p.name, json.loads((p / "scenario.json").read_text())) for p in roots]

    def _events(self):
        """(bundle, channel, event) for every archiver event in every bundle."""
        return [
            (name, entry["channel"], event)
            for name, spec in self._bundles()
            for entry in spec.get("archiver", [])
            for event in entry["events"]
        ]

    def test_shipped_bundles_carry_no_window_fraction_events(self):
        """No shipped event uses ``at`` — fraction positions are unseedable.

        A fraction-positioned event lands at a fixed *proportion* of whichever
        window is queried, which has no absolute time and so cannot be written
        into a store. Only ``at_offset`` and ``at_when`` (anchored) and
        ``at_time`` (daily recurrence) resolve to real timestamps.
        """
        fraction_events = [
            f"{name}/{pv}: at={event['at']}" for name, pv, event in self._events() if "at" in event
        ]
        assert not fraction_events, (
            "shipped scenario events must be anchored ('at_offset', 'at_when') or "
            f"daily ('at_time'), not window fractions: {fraction_events}"
        )

    def test_anchored_events_land_inside_the_seed_window(self):
        """Every ``at_offset`` event sits inside ``[T0 - horizon, T0]``."""
        for name, pv, event in self._events():
            if "at_offset" not in event:
                continue
            at = float(event["at_offset"])
            assert -self.SEED_HORIZON_S <= at <= 0.0, (
                f"{name}/{pv}: at_offset={at:.0f}s falls outside the seed window "
                f"[T0-{self.SEED_HORIZON_S:.0f}s, T0]"
            )
            # A ramp's far end and a spike's tail must fit too, not just its start.
            end = at + self.SIGMA_MARGIN * float(event["width"]) if "width" in event else at
            if "until_offset" in event:
                end = float(event["until_offset"])
            assert end <= 0.0, (
                f"{name}/{pv}: event extends to T0{end:+.0f}s, past the end of the "
                f"seed window (nothing after T0 is seeded)"
            )

    def test_calendar_events_land_inside_the_seed_window_at_any_apply_time(self):
        """Every ``at_when`` event sits inside ``[T0 - horizon, T0]`` whatever T0's clock time.

        An ``at_when`` instant is a calendar day before T0's date at a clock time,
        so how far back it lands moves with the time of day T0 falls at. The two
        extremes bound it: T0 just after midnight puts the event closest to T0,
        T0 just before the next midnight puts it furthest back.
        """
        day = 24 * 3600.0
        calendar = [e for e in self._events() if "at_when" in e[2]]
        assert calendar, "expected rf-thermal's calendar-placed events; coverage went vacuous"
        for name, pv, event in calendar:
            when = event["at_when"]
            clock = datetime.strptime(when["time"], "%H:%M:%S")
            clock_s = clock.hour * 3600.0 + clock.minute * 60.0 + clock.second
            tail = self.SIGMA_MARGIN * float(event.get("width", 0.0))
            nearest = when["days_ago"] * day - clock_s  # seconds before a T0 at 00:00
            furthest = nearest + day  # seconds before a T0 at 24:00
            assert nearest - tail >= 0.0, (
                f"{name}/{pv}: at_when={when} can reach past T0 (nothing after T0 is seeded)"
            )
            assert furthest + tail <= self.SEED_HORIZON_S, (
                f"{name}/{pv}: at_when={when} can fall before the seed window "
                f"[T0-{self.SEED_HORIZON_S:.0f}s, T0]"
            )

    def test_daily_events_recur_inside_the_seed_window(self):
        """``at_time`` events recur daily, so a horizon of >= 1 day always contains one."""
        daily = [e for e in self._events() if "at_time" in e[2]]
        assert daily, "expected at least one daily event (vacuum-burst); coverage went vacuous"
        assert self.SEED_HORIZON_S >= 24 * 3600.0, (
            "seed horizon is under a day — daily events are no longer guaranteed a "
            "seeded occurrence"
        )
