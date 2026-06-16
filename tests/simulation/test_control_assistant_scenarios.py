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
- ``rf-thermal``: C1 temperature excursions near window fractions
  0.20/0.55/0.85 (peaks 32-36 degC, assert > 31 over base 27); C2 stays
  quiet (~28.5 degC, assert < 29.5, far below the C1 excursions); C1
  reflected power tracks temperature (r ~= 0.96, assert > 0.8); derived
  POWER:NET tracks FWD-REV (r ~= 0.998, assert > 0.95); FREQUENCY:RB detunes
  downward during the excursions.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

# The shipped control_assistant engine is built by the shared ``engine_factory``
# fixture in conftest.py (copies machine.json + the scenarios/ bundle tree).
GAUGE = "SR:VAC:GAUGE:SR{:02d}:PRESSURE:RB"
DCCT = "SR:DIAG:DCCT:01:CURRENT:RB"
RF_SERIES_N = 2016  # 7-day window at 5-minute resolution


def _window(center: datetime, minutes: int = 10, step_s: int = 1) -> list[datetime]:
    """Return per-second timestamps for a window centered on ``center``."""
    start = center - timedelta(minutes=minutes / 2)
    return [start + timedelta(seconds=i) for i in range(minutes * 60 // step_s)]


def _yesterday_event() -> datetime:
    """Yesterday at 14:32:08 — the daily ``at_time`` anchor fires on any past date."""
    day = datetime.now() - timedelta(days=1)
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
    """Cavity-1 thermal excursions drive reflected power, forward trips, detuning."""

    CAV = "SR:RF:CAVITY:{:02d}:{}"

    def _series(self, engine, dev, suffix, n=RF_SERIES_N):
        end = datetime.now()
        ts = [end - timedelta(days=7) + timedelta(minutes=5 * i) for i in range(n)]
        return np.array(engine.synthesize_series(self.CAV.format(dev, suffix), ts))

    def test_c1_excursions_at_documented_positions(self, engine_factory):
        """C1 temperature peaks > 31 degC (over base 27) near t=0.20/0.55/0.85."""
        engine = engine_factory("rf-thermal")
        temp = self._series(engine, 1, "TEMPERATURE:RB")
        t = np.linspace(0, 1, len(temp))
        for pos in (0.20, 0.55, 0.85):
            window = temp[(t > pos - 0.03) & (t < pos + 0.03)]
            assert window.max() > 31.0, f"no C1 excursion near t={pos} (max {window.max():.2f})"

    def test_c2_stays_quiet(self, engine_factory):
        """C2 temperature stays < 29.5 degC — far below C1's 32-36 degC excursions."""
        engine = engine_factory("rf-thermal")
        temp = self._series(engine, 2, "TEMPERATURE:RB")
        # Base 26.5 + one minor 1.75-degC event => peak ~28.5; noise can nudge
        # to ~29.0. 29.5 keeps margin while staying well below C1 (32-36 degC),
        # so the assertion still fails on genuinely-anomalous C2 data.
        assert temp.max() < 29.5, f"C2 temp peak {temp.max():.2f}, contract < 29.5"

    def test_reflected_power_spikes_with_temperature(self, engine_factory):
        """C1 reflected power (POWER:REV) correlates with temperature: r > 0.8 (target ~ 0.96)."""
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
        """C1 resonant frequency detunes downward during the thermal excursions."""
        engine = engine_factory("rf-thermal")
        freq = self._series(engine, 1, "FREQUENCY:RB")
        assert freq.min() < 499.654 - 0.0005, f"C1 freq min {freq.min():.6f}, contract < 499.6535"
