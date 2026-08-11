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
- ``rf-thermal``: CAVITY01 temperature excursions at the anchored offsets
  T0-38h/-21h/-7h (peaks 32-36 degC, assert > 31 over base 27), with the
  stretch older than T0-48h flat; CAVITY02 stays quiet (~28.5 degC, assert
  < 29.5, far below the CAVITY01 excursions); CAVITY01 reflected power tracks
  temperature (r ~= 0.97, assert > 0.8); derived
  POWER:NET tracks FWD-REV (r ~= 0.998, assert > 0.95); FREQUENCY:RB detunes
  downward during the excursions.
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
RF_SERIES_N = 2016  # 7-day window at 5-minute resolution

# rf-thermal's three CAVITY01 excursions, in hours before the scenario-activation
# anchor T0, and their Gaussian sigma. Mirrors ``rf-thermal/scenario.json``; the
# tests below read the bundle back rather than trusting these, so a bundle edit
# that forgets this file fails loudly instead of silently drifting.
RF_EXCURSION_OFFSETS_H = (-38.0, -21.0, -7.0)
RF_EXCURSION_SIGMA_S = 7200.0


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
    """CAVITY01 thermal excursions drive reflected power, forward trips, detuning.

    The bundle positions its events with ``at_offset`` — seconds before the
    scenario-activation anchor T0 — so an excursion sits at a fixed wall-clock
    time rather than at a fixed fraction of whatever window is queried. The
    ``engine_factory`` fixture writes ``active_scenarios`` as it builds the
    engine and the bundle carries no explicit ``anchor=`` line, so T0 is that
    file's mtime: within a second of ``datetime.now()`` here. The
    excursion-position windows below are hours wide, which absorbs that.
    """

    CAV = "SR:RF:CAVITY:{:02d}:{}"

    def _series(self, engine, dev, suffix, n=RF_SERIES_N):
        """A 7-day window ending now, at 5-minute resolution."""
        end = datetime.now()
        ts = [end - timedelta(days=7) + timedelta(minutes=5 * i) for i in range(n)]
        return np.array(engine.synthesize_series(self.CAV.format(dev, suffix), ts))

    def _hours_before_now(self, n=RF_SERIES_N):
        """Signed hours-from-now for each sample of ``_series``'s window."""
        return np.linspace(-7 * 24.0, 0.0, n, endpoint=False)

    def test_cavity01_excursions_at_anchored_offsets(self, engine_factory):
        """CAVITY01 temperature peaks > 31 degC (over base 27) at T0-38h/-21h/-7h."""
        engine = engine_factory("rf-thermal")
        temp = self._series(engine, 1, "TEMPERATURE:RB")
        hours = self._hours_before_now(len(temp))
        for offset_h in RF_EXCURSION_OFFSETS_H:
            # +/- 3h brackets the 2h-sigma spike without reaching its neighbours,
            # which sit 15-17h away.
            window = temp[(hours > offset_h - 3.0) & (hours < offset_h + 3.0)]
            assert window.max() > 31.0, (
                f"no CAVITY01 excursion at T0{offset_h:+.0f}h (max {window.max():.2f})"
            )

    def test_cavity01_is_flat_before_the_excursion_band(self, engine_factory):
        """Older than T0-48h, CAVITY01 temperature is quiet baseline + noise.

        Under the previous window-fraction positioning no part of any window was
        event-free — the excursions rescaled to whatever span was queried. Anchored
        offsets confine them to the last 40 hours, so this stretch is genuinely
        undisturbed, and asserting that is what stops an accidental slide back to
        fraction semantics from passing silently.
        """
        engine = engine_factory("rf-thermal")
        temp = self._series(engine, 1, "TEMPERATURE:RB")
        hours = self._hours_before_now(len(temp))
        quiet = temp[hours < -48.0]
        assert quiet.max() < 29.0, (
            f"CAVITY01 temp peak {quiet.max():.2f} older than T0-48h, contract < 29.0 "
            f"(base 27 + noise); an excursion has leaked outside the anchored band"
        )

    def test_bundle_offsets_match_the_pinned_contract(self, engine_factory):
        """The shipped bundle carries exactly the offsets/width this file asserts."""
        bundle = json.loads((TEMPLATE_SIM / "scenarios/rf-thermal/scenario.json").read_text())
        cav01_temp = next(
            entry
            for entry in bundle["archiver"]
            if entry["channel"] == self.CAV.format(1, "TEMPERATURE:RB")
        )
        offsets_h = tuple(ev["at_offset"] / 3600.0 for ev in cav01_temp["events"])
        assert offsets_h == RF_EXCURSION_OFFSETS_H
        assert {ev["width"] for ev in cav01_temp["events"]} == {RF_EXCURSION_SIGMA_S}

    def test_cavity02_stays_quiet(self, engine_factory):
        """CAVITY02 temperature stays < 29.5 degC — far below CAVITY01's 32-36 degC excursions."""
        engine = engine_factory("rf-thermal")
        temp = self._series(engine, 2, "TEMPERATURE:RB")
        # Base 26.5 + one minor 1.75-degC event => peak ~28.5; noise can nudge
        # to ~29.0. 29.5 keeps margin while staying well below CAVITY01
        # (32-36 degC), so the assertion still fails on genuinely-anomalous
        # CAVITY02 data.
        assert temp.max() < 29.5, f"CAVITY02 temp peak {temp.max():.2f}, contract < 29.5"

    def test_reflected_power_spikes_with_temperature(self, engine_factory):
        """CAVITY01 reflected power (POWER:REV) correlates with temperature: r > 0.8 (target ~ 0.96)."""
        engine = engine_factory("rf-thermal")
        temp = self._series(engine, 1, "TEMPERATURE:RB")
        rev = self._series(engine, 1, "POWER:REV")
        r = np.corrcoef(temp, rev)[0, 1]
        assert r > 0.8, f"TEMP/REV correlation r = {r:.3f}, contract > 0.8"

    def test_net_power_is_fwd_minus_rev(self, engine_factory):
        """POWER:NET is the live expression FWD - REV, proven exactly.

        A correlation test on separately-synthesized series cannot falsify a
        wrong ``FWD + REV`` formula — the excursion trips dominate the noise,
        so both signs correlate. Instead this writes FWD and REV and reads NET
        back: a noise-free derived channel must yield *exactly* FWD - REV,
        which ``FWD + REV`` could not. The archived NET history is then checked
        to collapse alongside the forward-power trips.
        """
        engine = engine_factory("rf-thermal")
        fwd_pv = self.CAV.format(1, "POWER:FWD")
        rev_pv = self.CAV.format(1, "POWER:REV")
        net_pv = self.CAV.format(1, "POWER:NET")
        engine.write(fwd_pv, 300.0)
        engine.write(rev_pv, 50.0)
        net = engine.read(net_pv).value
        assert net == pytest.approx(250.0), f"NET={net}, expected FWD-REV=250.0 (not FWD+REV=350)"

        # Archived NET history collapses where forward power trips toward zero.
        fwd = self._series(engine, 1, "POWER:FWD")
        net_series = self._series(engine, 1, "POWER:NET")
        trip = int(np.argmin(fwd))
        assert net_series[trip] < net_series.mean() - 100.0, (
            f"NET at the forward-power trip ({net_series[trip]:.1f}) does not collapse "
            f"below its mean ({net_series.mean():.1f})"
        )

    def test_frequency_detunes_during_excursions(self, engine_factory):
        """CAVITY01 resonant frequency detunes downward during the thermal excursions."""
        engine = engine_factory("rf-thermal")
        freq = self._series(engine, 1, "FREQUENCY:RB")
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
    seed-manifest document and scenario activation resolves ``at_offset``
    against that persisted anchor, not against its own wall clock. If those two
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
        into a store. Only ``at_offset`` (anchored) and ``at_time`` (daily
        recurrence) resolve to real timestamps.
        """
        fraction_events = [
            f"{name}/{pv}: at={event['at']}" for name, pv, event in self._events() if "at" in event
        ]
        assert not fraction_events, (
            "shipped scenario events must be anchored ('at_offset') or daily "
            f"('at_time'), not window fractions: {fraction_events}"
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

    def test_daily_events_recur_inside_the_seed_window(self):
        """``at_time`` events recur daily, so a horizon of >= 1 day always contains one."""
        daily = [e for e in self._events() if "at_time" in e[2]]
        assert daily, "expected at least one daily event (vacuum-burst); coverage went vacuous"
        assert self.SEED_HORIZON_S >= 24 * 3600.0, (
            "seed horizon is under a day — daily events are no longer guaranteed a "
            "seeded occurrence"
        )
