"""Shared latent drivers: ``drivers`` / ``couple`` / ``noise`` scenario blocks.

A driver is a named ``wander`` stack evaluated on absolute epoch time and keyed
by the driver's name, so every channel coupled to it sees the identical value
at a given instant — that shared term is what correlates otherwise independent
channels. These tests pin the schema (and each refusal), composition, the
correlation behaviour (|r| ~ 1 without noise, an elongated cloud with noise,
drifting strength under ``gain_wander``), live/history agreement, and the
shipped ``rf-thermal-live`` bundle's target correlation bands.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from osprey.simulation import SimulationEngine
from osprey_connectors.simulation.machine import DriverCoupling, NoiseOverride, TextureSpec

T0 = 1_790_000_000.0
DRIVER = {"kind": "wander", "amplitude": 1.0, "period_s": 300.0}

CAV_T = "SR:RF:CAVITY:01:TEMPERATURE:RB"
CAV_REV = "SR:RF:CAVITY:01:POWER:REV"
CAV_FWD = "SR:RF:CAVITY:01:POWER:FWD"
CAV_TUNER = "SR:RF:CAVITY:01:TUNER:RB"


def _r(x, y) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def _freeze_now(monkeypatch, epoch: float) -> None:
    """Freeze the engine module's wall clock without touching stdlib time."""
    monkeypatch.setattr("osprey.simulation.engine.time", SimpleNamespace(time=lambda: epoch))


@pytest.fixture
def coupled_machine(machine_dict):
    """Inline machine with two noise-free numeric channels and a coupling scenario."""
    machine_dict["channels"]["T:A"] = {"value": 10.0, "noise": 0.0, "description": "A"}
    machine_dict["channels"]["T:B"] = {"value": -3.0, "noise": 0.0, "description": "B"}
    machine_dict["scenarios"]["shared"] = {
        "description": "A and B share one driver.",
        "drivers": {"d": dict(DRIVER)},
        "couple": {
            "T:A": [{"driver": "d", "gain": 2.0}],
            "T:B": [{"driver": "d", "gain": -0.5}],
        },
    }
    return machine_dict


def _engine(machine, make_machine_file, *active: str) -> SimulationEngine:
    engine = SimulationEngine.from_file(make_machine_file(machine))
    if active:
        engine.set_active_scenarios(list(active))
    return engine


def _window(n: int, start: float = T0, step: float = 1.0) -> list[float]:
    return [start + i * step for i in range(n)]


class TestParsing:
    def test_blocks_parse_into_the_typed_model(self, coupled_machine, make_machine_file):
        coupled_machine["scenarios"]["shared"]["couple"]["T:A"][0]["gain_wander"] = {
            "amplitude": 0.4,
            "period_s": 180,
        }
        coupled_machine["scenarios"]["shared"]["noise"] = {"T:A": {"noise_abs": 0.1}}
        engine = _engine(coupled_machine, make_machine_file)
        scenario = engine._scenarios["shared"]
        spec = TextureSpec("wander", 1.0, 300.0)
        assert scenario.drivers == {"d": spec}
        assert scenario.couple["T:A"] == (
            DriverCoupling("d", spec, 2.0, TextureSpec("wander", 0.4, 180.0)),
        )
        assert scenario.couple["T:B"] == (DriverCoupling("d", spec, -0.5, None),)
        assert scenario.noise == {"T:A": NoiseOverride(noise=None, noise_abs=0.1)}

    def test_scenarios_without_the_blocks_parse_unchanged(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        scenario = engine._scenarios["quad-drift"]
        assert scenario.drivers == {} and scenario.couple == {} and scenario.noise == {}

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda s: s.update(drivers=[]), "'drivers' must be a mapping"),
            (lambda s: s["drivers"].update(d=5), "driver 'd' must be a mapping"),
            (lambda s: s["drivers"]["d"].pop("period_s"), "driver 'd' missing keys"),
            (lambda s: s["drivers"]["d"].update(seed=1), "driver 'd' has unknown keys"),
            (lambda s: s["drivers"]["d"].update(kind="pink"), "driver 'd' kind must be one of"),
            (lambda s: s["drivers"]["d"].update(amplitude=0), "driver 'd' amplitude must be"),
            (lambda s: s["drivers"]["d"].update(period_s=-1), "driver 'd' period_s must be"),
            (lambda s: s["drivers"]["d"].update(period_s=True), "driver 'd' period_s must be"),
            (lambda s: s.update(couple=[]), "'couple' must be a mapping"),
            (lambda s: s["couple"].update({"T:NOPE": []}), "couple for unknown channel"),
            (
                lambda s: s["couple"].update({"T:MODE": [{"driver": "d", "gain": 1}]}),
                "couple for string-valued channel",
            ),
            (lambda s: s["couple"].update({"T:A": []}), "must be a non-empty list"),
            (lambda s: s["couple"].update({"T:A": [3]}), "couple['T:A'] entry must be a mapping"),
            (
                lambda s: s["couple"].update({"T:A": [{"driver": "zz", "gain": 1}]}),
                "references unknown driver 'zz'",
            ),
            (
                lambda s: s["couple"].update({"T:A": [{"gain": 1}]}),
                "missing key 'driver'",
            ),
            (
                lambda s: s["couple"].update({"T:A": [{"driver": "d"}]}),
                "missing key 'gain'",
            ),
            (
                lambda s: s["couple"].update({"T:A": [{"driver": "d", "gain": "2"}]}),
                "gain must be a number",
            ),
            (
                lambda s: s["couple"].update({"T:A": [{"driver": "d", "gain": 1, "lag": 2}]}),
                "has unknown keys ['lag']",
            ),
            (
                lambda s: s["couple"].update(
                    {"T:A": [{"driver": "d", "gain": 1}, {"driver": "d", "gain": 2}]}
                ),
                "couples driver 'd' twice",
            ),
            (
                lambda s: s["couple"].update(
                    {"T:A": [{"driver": "d", "gain": 1, "gain_wander": {"amplitude": 0.3}}]}
                ),
                "gain_wander missing keys ['period_s']",
            ),
            (
                lambda s: s["couple"].update(
                    {
                        "T:A": [
                            {
                                "driver": "d",
                                "gain": 1,
                                "gain_wander": {"amplitude": 0.3, "period_s": 0},
                            }
                        ]
                    }
                ),
                "gain_wander period_s must be a number > 0",
            ),
            (lambda s: s.update(noise=[]), "'noise' must be a mapping"),
            (
                lambda s: s.update(noise={"T:NOPE": {"noise_abs": 1}}),
                "noise override for unknown channel",
            ),
            (
                lambda s: s.update(noise={"T:MODE": {"noise_abs": 1}}),
                "noise override for string-valued channel",
            ),
            (lambda s: s.update(noise={"T:A": {}}), "must set at least one of"),
            (lambda s: s.update(noise={"T:A": {"sigma": 1}}), "has unknown keys ['sigma']"),
            (
                lambda s: s.update(noise={"T:A": {"noise_abs": -0.1}}),
                "noise_abs must be a non-negative number",
            ),
        ],
    )
    def test_malformed_blocks_are_refused(
        self, coupled_machine, make_machine_file, mutate, message
    ):
        mutate(coupled_machine["scenarios"]["shared"])
        with pytest.raises(ValueError, match="Scenario 'shared'") as excinfo:
            SimulationEngine.from_file(make_machine_file(coupled_machine))
        assert message in str(excinfo.value)


class TestComposition:
    @pytest.mark.parametrize(
        "other",
        [
            {"overrides": {"T:A": 1.0}},
            {"noise": {"T:A": {"noise_abs": 0.5}}},
            {"drivers": {"e": dict(DRIVER)}, "couple": {"T:A": [{"driver": "e", "gain": 1}]}},
        ],
        ids=["override", "noise", "couple"],
    )
    def test_coupled_channel_counts_as_touched(self, coupled_machine, make_machine_file, other):
        coupled_machine["scenarios"]["other"] = {"description": "", **other}
        engine = _engine(coupled_machine, make_machine_file)
        problems = engine.validate_composition(["nominal", "shared", "other"])
        assert any("'T:A' is touched by both 'shared' and 'other'" in p for p in problems)
        with pytest.raises(ValueError, match="Cannot activate scenarios"):
            engine.set_active_scenarios(["shared", "other"])

    def test_noise_only_channel_counts_as_touched(self, coupled_machine, make_machine_file):
        coupled_machine["scenarios"]["shared"]["noise"] = {"T:NOISY": {"noise": 0.0}}
        coupled_machine["scenarios"]["other"] = {"overrides": {"T:NOISY": 1.0}}
        engine = _engine(coupled_machine, make_machine_file)
        assert engine.validate_composition(["nominal", "shared", "other"])

    def test_disjoint_scenarios_compose(self, coupled_machine, make_machine_file):
        engine = _engine(coupled_machine, make_machine_file)
        assert engine.validate_composition(["nominal", "shared", "quad-drift"]) == []


class TestCoupling:
    def test_inactive_scenario_contributes_nothing(self, coupled_machine, make_machine_file):
        engine = _engine(coupled_machine, make_machine_file)
        assert engine.synthesize_series("T:A", _window(50)) == [10.0] * 50

    def test_coupled_channels_see_the_identical_driver(self, coupled_machine, make_machine_file):
        engine = _engine(coupled_machine, make_machine_file, "shared")
        stamps = _window(600)
        a = (np.array(engine.synthesize_series("T:A", stamps)) - 10.0) / 2.0
        b = (np.array(engine.synthesize_series("T:B", stamps)) + 3.0) / -0.5
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
        assert np.ptp(a) > 0.5  # anti-degenerate: the driver actually moves

    def test_zero_noise_correlation_is_perfect(self, coupled_machine, make_machine_file):
        engine = _engine(coupled_machine, make_machine_file, "shared")
        stamps = _window(600)
        a = engine.synthesize_series("T:A", stamps)
        b = engine.synthesize_series("T:B", stamps)
        assert _r(a, b) < -0.9999  # negative gain anti-correlates

    def test_live_reads_share_the_driver(self, coupled_machine, make_machine_file, monkeypatch):
        engine = _engine(coupled_machine, make_machine_file, "shared")
        a, b = [], []
        for t in _window(300):
            _freeze_now(monkeypatch, t)
            a.append(engine.read("T:A").value)
            b.append(engine.read("T:B").value)
        assert _r(a, b) < -0.9999

    def test_noise_gives_an_elongated_cloud(self, coupled_machine, make_machine_file):
        coupled_machine["scenarios"]["shared"]["noise"] = {
            "T:A": {"noise_abs": 0.4},
            "T:B": {"noise_abs": 0.1},
        }
        engine = _engine(coupled_machine, make_machine_file, "shared")
        stamps = _window(3600)
        r = _r(engine.synthesize_series("T:A", stamps), engine.synthesize_series("T:B", stamps))
        assert -0.95 < r < -0.4

    def test_gain_wander_makes_the_gain_drift(self, coupled_machine, make_machine_file):
        coupled_machine["scenarios"]["shared"]["couple"]["T:B"][0]["gain_wander"] = {
            "amplitude": 0.8,
            "period_s": 600,
        }
        engine = _engine(coupled_machine, make_machine_file, "shared")
        stamps = _window(3600)
        driver = (np.array(engine.synthesize_series("T:A", stamps)) - 10.0) / 2.0
        b = np.array(engine.synthesize_series("T:B", stamps)) + 3.0
        mask = np.abs(driver) > 0.1
        effective_gain = b[mask] / driver[mask]
        assert effective_gain.min() < -0.5 * 1.4  # envelope above 1 ...
        assert effective_gain.max() > -0.5 * 0.6  # ... and below 1 within the hour
        assert np.all(effective_gain <= 0)  # amplitude < 1 never flips the sign

    def test_gain_wander_makes_windowed_correlation_vary(self, coupled_machine, make_machine_file):
        couple = coupled_machine["scenarios"]["shared"]["couple"]
        couple["T:B"][0]["gain_wander"] = {"amplitude": 0.8, "period_s": 900}
        coupled_machine["scenarios"]["shared"]["noise"] = {
            "T:A": {"noise_abs": 0.1},
            "T:B": {"noise_abs": 0.2},
        }
        engine = _engine(coupled_machine, make_machine_file, "shared")
        stamps = _window(3600)
        a = np.array(engine.synthesize_series("T:A", stamps))
        b = np.array(engine.synthesize_series("T:B", stamps))
        windowed = [_r(a[s : s + 300], b[s : s + 300]) for s in range(0, 3300, 60)]
        assert max(windowed) - min(windowed) > 0.2

    def test_noise_override_replaces_machine_noise(self, coupled_machine, make_machine_file):
        coupled_machine["scenarios"]["shared"]["noise"] = {
            "T:NOISY": {"noise": 0.0, "noise_abs": 2.0}
        }
        nominal = _engine(coupled_machine, make_machine_file)
        stamps = _window(2000)
        assert np.std(nominal.synthesize_series("T:NOISY", stamps)) == pytest.approx(5.0, rel=0.1)
        nominal.set_active_scenarios(["shared"])
        assert np.std(nominal.synthesize_series("T:NOISY", stamps)) == pytest.approx(2.0, rel=0.1)
        live = np.array([nominal.read("T:NOISY").value for _ in range(2000)])
        assert live.std() == pytest.approx(2.0, rel=0.1)

    def test_partial_noise_override_keeps_the_other_term(self, coupled_machine, make_machine_file):
        coupled_machine["scenarios"]["shared"]["noise"] = {"T:NOISY": {"noise_abs": 3.0}}
        engine = _engine(coupled_machine, make_machine_file, "shared")
        # relative 0.05 on 100 (sigma 5) kept, absolute 3 added: quadrature ~5.83
        std = np.std(engine.synthesize_series("T:NOISY", _window(4000)))
        assert std == pytest.approx(np.hypot(5.0, 3.0), rel=0.1)


class TestLiveHistoryConsistency:
    INSTANTS = (T0, T0 + 47.0, T0 + 133.0, T0 + 1201.0)

    def test_live_read_matches_synthesis_at_frozen_now(
        self, coupled_machine, make_machine_file, monkeypatch
    ):
        coupled_machine["channels"]["T:A"]["texture"] = {
            "kind": "wander",
            "amplitude": 0.3,
            "period_s": 3600.0,
        }
        coupled_machine["scenarios"]["shared"]["couple"]["T:A"][0]["gain_wander"] = {
            "amplitude": 0.5,
            "period_s": 200,
        }
        engine = _engine(coupled_machine, make_machine_file, "shared")
        contributions = []
        for frozen in self.INSTANTS:
            _freeze_now(monkeypatch, frozen)
            for pv, base in (("T:A", 10.0), ("T:B", -3.0)):
                live = engine.read(pv).value
                (synthesized,) = engine.synthesize_series(pv, [frozen])
                assert live == synthesized  # bit-exact, not approx
                contributions.append(abs(live - base))
        assert max(contributions) > 0.1  # anti-degenerate

    def test_windowed_history_matches_pointwise_live(
        self, coupled_machine, make_machine_file, monkeypatch
    ):
        engine = _engine(coupled_machine, make_machine_file, "shared")
        stamps = _window(120, step=0.5)
        history = engine.synthesize_series("T:A", stamps)
        for t, h in zip(stamps[::17], history[::17], strict=True):
            _freeze_now(monkeypatch, t)
            assert engine.read("T:A").value == h


class TestShippedRfThermalLive:
    """The shipped bundle's correlation targets, on 1 s sampling over 5-min windows."""

    WINDOW = 300

    @pytest.fixture
    def series(self, engine_factory):
        engine = engine_factory("rf-thermal-live")
        stamps = _window(3 * 3600)
        return {
            pv: np.array(engine.synthesize_series(pv, stamps))
            for pv in (CAV_T, CAV_REV, CAV_FWD, CAV_TUNER)
        }

    def _windowed(self, x, y) -> list[float]:
        n, w = len(x), self.WINDOW
        return [_r(x[s : s + w], y[s : s + w]) for s in range(0, n - w + 1, 30)]

    def test_temperature_vs_reflected_power_wanders_in_band(self, series):
        rs = self._windowed(series[CAV_T], series[CAV_REV])
        assert 0.35 < min(rs) and max(rs) < 0.95
        assert 0.6 < float(np.median(rs)) < 0.85
        assert max(rs) - min(rs) > 0.2  # correlation strength drifts

    def test_temperature_vs_tuner_is_tight(self, series):
        rs = self._windowed(series[CAV_T], series[CAV_TUNER])
        assert min(rs) > 0.9 and max(rs) < 0.995  # tight, but a cloud, not a line

    def test_fastest_motion_is_tens_of_seconds(self, series):
        # Noise-free driver view: temperature minus its mean, smoothed over 5 s.
        t = np.convolve(series[CAV_T] - series[CAV_T].mean(), np.ones(5) / 5, mode="valid")
        crossings = np.count_nonzero(np.diff(np.sign(t)) != 0)
        # a 27 s fastest component would give ~800 crossings in 3 h; noise-only ~thousands
        assert 50 < crossings < 2000

    def test_levels_stay_physical(self, series):
        assert series[CAV_REV].min() > 0.0
        assert 26.0 < series[CAV_T].min() and series[CAV_T].max() < 28.0

    def test_composes_with_vacuum_burst_but_not_rf_thermal(self, engine_factory):
        engine = engine_factory("nominal")
        assert engine.validate_composition(["nominal", "rf-thermal-live", "vacuum-burst"]) == []
        assert engine.validate_composition(["nominal", "rf-thermal-live", "rf-thermal"])
