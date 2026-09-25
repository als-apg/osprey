"""The engine's signal model applied to a level it did not compute.

A physics backend owns the level of a monitor it models, and the machine file
owns how that monitor moves around its level. ``SimulationEngine.measure``
puts the second on top of the first with the same arithmetic synthesis uses,
so a live sample at time ``t`` lands where the archived history puts it.
"""

import json

import numpy as np
import pytest

from osprey.simulation import SimulationEngine

T0 = 1_764_000_000.0


def _engine(machine_dict, make_machine_file):
    return SimulationEngine.from_file(make_machine_file(machine_dict))


class TestMeasure:
    def test_a_level_equal_to_the_baseline_reproduces_the_synthesized_sample(
        self, machine_dict, make_machine_file
    ):
        engine = _engine(machine_dict, make_machine_file)
        instants = [T0 + 700.0 * k for k in range(6)]
        synthesized = engine.synthesize_series("T:ZERO:TEXTURED", instants)
        measured = [engine.measure("T:ZERO:TEXTURED", 0.0, t) for t in instants]
        assert measured == synthesized

    def test_the_motion_rides_on_whatever_level_it_is_given(self, machine_dict, make_machine_file):
        engine = _engine(machine_dict, make_machine_file)
        instants = [T0 + 700.0 * k for k in range(6)]
        synthesized = np.asarray(engine.synthesize_series("T:ZERO:TEXTURED", instants))
        measured = np.asarray([engine.measure("T:ZERO:TEXTURED", 0.3, t) for t in instants])
        np.testing.assert_allclose(measured - 0.3, synthesized, rtol=0, atol=1e-12)
        assert len(set(measured.tolist())) == len(instants)

    def test_a_channel_declaring_no_motion_returns_its_level_unchanged(
        self, machine_dict, make_machine_file
    ):
        engine = _engine(machine_dict, make_machine_file)
        assert not engine.has_motion("T:VAC")
        assert engine.measure("T:VAC", 1.25e-9, T0) == 1.25e-9

    def test_an_unknown_channel_is_left_alone(self, machine_dict, make_machine_file):
        engine = _engine(machine_dict, make_machine_file)
        assert not engine.has_motion("NOT:A:CHANNEL")
        assert engine.measure("NOT:A:CHANNEL", 7.0, T0) == 7.0

    @pytest.mark.parametrize("pv", ["T:ZERO:NOISY", "T:ZERO:TEXTURED", "T:NOISY"])
    def test_texture_or_either_noise_is_motion(self, machine_dict, make_machine_file, pv):
        assert _engine(machine_dict, make_machine_file).has_motion(pv)

    def test_a_string_channel_never_moves(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:MODE"]["noise"] = 0.5
        assert not _engine(machine_dict, make_machine_file).has_motion("T:MODE")

    def test_absolute_noise_has_its_declared_width(self, machine_dict, make_machine_file):
        engine = _engine(machine_dict, make_machine_file)
        values = np.asarray(
            [engine.measure("T:ZERO:NOISY", 0.0, T0 + 10.0 * k) for k in range(500)]
        )
        assert values.std() == pytest.approx(0.02, rel=0.15)


class TestBaselines:
    """An engine built with ``baselines`` synthesizes around those levels instead."""

    def test_a_value_channel_is_rebased(self, machine_dict, make_machine_file):
        path = make_machine_file(machine_dict)
        plain = SimulationEngine.from_file(path)
        rebased = SimulationEngine(
            json.loads(path.read_text()),
            path,
            state_dir=path.parent,
            baselines={"T:ZERO:TEXTURED": -3.5e-6},
        )
        instants = [T0 + 700.0 * k for k in range(4)]
        assert rebased.synthesize_series("T:ZERO:TEXTURED", instants) == [
            rebased.measure("T:ZERO:TEXTURED", -3.5e-6, t) for t in instants
        ]
        assert rebased.synthesize_series("T:ZERO:TEXTURED", instants) != (
            plain.synthesize_series("T:ZERO:TEXTURED", instants)
        )
        assert rebased.read("T:VAC").value == plain.read("T:VAC").value

    def test_expression_string_and_unknown_channels_keep_their_definition(
        self, machine_dict, make_machine_file
    ):
        path = make_machine_file(machine_dict)
        plain = SimulationEngine.from_file(path)
        rebased = SimulationEngine(
            json.loads(path.read_text()),
            path,
            state_dir=path.parent,
            baselines={"T:Q1:CUR:RB": 1.0, "T:MODE": 2.0, "NOT:A:CHANNEL": 3.0},
        )
        assert rebased.read("T:Q1:CUR:RB").value == plain.read("T:Q1:CUR:RB").value
        assert rebased.read("T:MODE").value == "CW"
        assert not rebased.has_channel("NOT:A:CHANNEL")
