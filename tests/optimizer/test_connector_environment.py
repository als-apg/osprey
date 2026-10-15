"""ConnectorEnvironment against a recording fake interface: no control system needed."""

import math

import pytest

from osprey_optimizer.environments.connector_env import ConnectorEnvironment, reduce_values
from osprey_optimizer.environments.interface import pool_section
from osprey_optimizer.environments.spec import EnvironmentFile
from osprey_optimizer.models.config import OptimizationConfig, VariableOverride
from osprey_optimizer.models.exceptions import (
    CancellationError,
    EvaluationError,
    InterfaceError,
    ObservableNotFoundError,
    VariableOutOfBoundsError,
)
from osprey_optimizer.models.protocols import PVReading
from osprey_optimizer.services.draft_resolution import resolve_draft_config


class FakeInterface:
    """Channels in a dict; a write sets the setpoint and (unless stuck) its readback."""

    def __init__(self, values: dict[str, float], readback_of: dict[str, str] | None = None):
        self.values = dict(values)
        self.readback_of = readback_of or {}
        self.stuck: set[str] = set()
        self.writes: list[dict[str, float]] = []

    def read_one(self, channel):
        return PVReading(value=self.values[channel])

    def read_many(self, channel_names):
        return {
            c: PVReading(value=self.values[c])
            if c in self.values
            else PVReading(value=None, connected=False)
            for c in channel_names
        }

    def set_values(self, channel_inputs):
        self.writes.append(dict(channel_inputs))
        for channel, value in channel_inputs.items():
            self.values[channel] = value
            rb = self.readback_of.get(channel)
            if rb is not None and rb not in self.stuck:
                self.values[rb] = value
        return dict(channel_inputs)

    def check_connection(self, pv_name=None):
        return None if pv_name is None else pv_name in self.values


SPEC = EnvironmentFile.model_validate(
    {
        "name": "orbit",
        "variables": {
            "VCM1": {
                "setpoint": "VCM1:SP",
                "readback": "VCM1:RB",
                "min": -12,
                "max": 12,
                "step_size": 0.01,
            },
            "VCM2": {
                "setpoint": "VCM2:SP",
                "readback": "VCM2:RB",
                "min": -12,
                "max": 12,
                "step_size": 0.01,
            },
        },
        "observables": {
            "rms_y": {
                "reduce": "rms",
                "channels": ["BPM{1..3}:Y"],
                "direction": "MINIMIZE",
                "units": "m",
            },
            "max_y": {"reduce": "max", "channels": ["BPM{1..3}:Y"]},
            "bpm1": {"channel": "BPM1:Y", "selectable_as_objective": False},
        },
        "default_objective": "rms_y",
        "readback_timeout_seconds": 0.3,
    }
)


@pytest.fixture
def fake():
    return FakeInterface(
        {
            "VCM1:SP": 0.0,
            "VCM1:RB": 0.0,
            "VCM2:SP": 0.0,
            "VCM2:RB": 0.0,
            "BPM1:Y": 3.0,
            "BPM2:Y": -4.0,
            "BPM3:Y": 0.0,
        },
        readback_of={"VCM1:SP": "VCM1:RB", "VCM2:SP": "VCM2:RB"},
    )


def _env(fake, **config):
    seen = {}

    def factory(target, mode):
        seen["key"] = (target, mode)
        return fake

    env = ConnectorEnvironment(
        {"spec": SPEC, "control_target": "va", **config}, interface_factory=factory
    )
    return env, seen


def test_reductions():
    assert reduce_values("rms", [3.0, -4.0, 0.0]) == pytest.approx(math.sqrt(25 / 3))
    assert reduce_values("mean", [1.0, 2.0, 6.0]) == 3.0
    assert reduce_values("max", [1.0, 7.0, 2.0]) == 7.0


def test_metadata_needs_no_target():
    env = ConnectorEnvironment({"spec": SPEC})
    assert set(env.variables) == {"VCM1", "VCM2"}
    assert env.available_objectives == ["rms_y", "max_y"]
    assert env.default_objective == "rms_y"


def test_channel_access_without_a_target_is_refused():
    env = ConnectorEnvironment({"spec": SPEC})
    with pytest.raises(InterfaceError, match="control_target"):
        env.get_variables(["VCM1"])


def test_the_interface_is_built_for_the_named_target_and_mode(fake):
    env, seen = _env(fake, control_target="standin", execution_mode="readonly")
    env.get_variables(["VCM1"])
    assert seen["key"] == ("standin", "readonly")


def test_active_variables_subset(fake):
    env, _ = _env(fake, variables=["VCM2"])
    assert list(env.variables) == ["VCM2"]
    with pytest.raises(ValueError, match="Unknown variable"):
        _env(fake, variables=["NOPE"])


def test_evaluate_writes_setpoints_and_reduces_observables(fake):
    env, _ = _env(fake)
    out = env.evaluate({"VCM1": 1.5}, ["rms_y", "max_y", "bpm1"])
    assert fake.writes == [{"VCM1:SP": 1.5}]
    assert out["rms_y"].value == pytest.approx(math.sqrt(25 / 3))
    assert out["rms_y"].units == "m"
    assert out["max_y"].value == 3.0
    assert out["bpm1"].value == 3.0


def test_get_variables_reads_the_readback(fake):
    env, _ = _env(fake)
    fake.values["VCM1:RB"] = 0.42
    assert env.get_variables(["VCM1"])["VCM1"].value == 0.42


def test_get_setpoints_reads_the_setpoint_not_the_readback(fake):
    env, _ = _env(fake)
    fake.values["VCM1:SP"] = 1.0
    fake.values["VCM1:RB"] = 1.5  # a calibration offset
    assert env.get_setpoints(["VCM1"])["VCM1"].value == 1.0
    assert fake.writes == []


@pytest.mark.parametrize("value", [12.01, -12.5, float("nan"), float("inf")])
def test_out_of_bounds_or_non_finite_writes_are_refused_before_any_write(fake, value):
    env, _ = _env(fake)
    with pytest.raises(VariableOutOfBoundsError):
        env.set_variables({"VCM1": 1.0, "VCM2": value})
    assert fake.writes == []


def test_unknown_variable_write_is_refused(fake):
    env, _ = _env(fake)
    with pytest.raises(VariableOutOfBoundsError, match="Unknown"):
        env.set_variables({"NOPE": 1.0})
    assert fake.writes == []


def test_a_readback_that_never_agrees_fails_the_evaluation(fake):
    env, _ = _env(fake)
    fake.stuck.add("VCM1:RB")
    with pytest.raises(EvaluationError, match="VCM1"):
        env.set_variables({"VCM1": 2.0})


def test_readback_within_tolerance_is_accepted(fake):
    env, _ = _env(fake)
    fake.stuck.add("VCM1:RB")
    fake.values["VCM1:RB"] = 2.005  # within step_size 0.01
    env.set_variables({"VCM1": 2.0})


def test_cancellation_is_checked_while_waiting_for_readbacks(fake):
    env, _ = _env(fake)
    fake.stuck.add("VCM1:RB")

    def cancel():
        raise CancellationError("cancelled")

    with pytest.raises(CancellationError):
        env.set_variables({"VCM1": 2.0}, check_cancellation=cancel)


def test_one_unreadable_channel_makes_the_reduction_unconnected(fake):
    env, _ = _env(fake)
    del fake.values["BPM2:Y"]
    reading = env.get_observables(["rms_y"])["rms_y"]
    assert reading.connected is False
    assert reading.value is None


def test_unknown_observable_is_refused(fake):
    env, _ = _env(fake)
    with pytest.raises(ObservableNotFoundError):
        env.get_observables(["nope"])


def test_pool_section_turns_fresh_reads_on_in_every_connector_block():
    section = {"type": "epics", "connector": {"epics": {"timeout": 5}, "virtual_accelerator": {}}}
    out = pool_section(section)
    assert out["connector"]["epics"] == {"timeout": 5, "fresh_reads": True}
    assert out["connector"]["virtual_accelerator"] == {"fresh_reads": True}
    assert "fresh_reads" not in section["connector"]["epics"]  # the caller's section is untouched


# -- channel limits --------------------------------------------------------------


class _Limit:
    def __init__(self, min_value=None, max_value=None, writable=True, max_step=None):
        self.min_value, self.max_value, self.writable = min_value, max_value, writable
        self.max_step = max_step


def _limited(limits, target="va"):
    return ConnectorEnvironment(
        {"spec": SPEC, "control_target": target}, limits_lookup=lambda t: limits
    )


def test_variable_ranges_narrow_to_the_channel_limits():
    env = _limited({"VCM1:SP": _Limit(-5.0, 20.0)})
    assert (env.variables["VCM1"].min_value, env.variables["VCM1"].max_value) == (-5.0, 12.0)
    assert (env.variables["VCM2"].min_value, env.variables["VCM2"].max_value) == (-12.0, 12.0)


def test_a_write_outside_the_narrowed_range_is_refused_before_any_write(fake):
    env = ConnectorEnvironment(
        {"spec": SPEC, "control_target": "va"},
        interface_factory=lambda t, m: fake,
        limits_lookup=lambda t: {"VCM1:SP": _Limit(-5.0, 5.0)},
    )
    with pytest.raises(VariableOutOfBoundsError):
        env.set_variables({"VCM1": 8.0})
    assert fake.writes == []


def test_a_read_only_setpoint_is_refused():
    with pytest.raises(ValueError, match="read-only"):
        _limited({"VCM1:SP": _Limit(writable=False)})


def test_a_range_that_misses_the_channel_limits_is_refused():
    with pytest.raises(ValueError, match="does not meet"):
        _limited({"VCM1:SP": _Limit(20.0, 30.0)})


def test_unlisted_channels_and_no_limits_leave_the_ranges_alone():
    assert _limited({})._all_variables == _limited(None)._all_variables == SPEC.variable_metadata()


def test_the_limits_asked_for_are_the_targets():
    asked = []

    def lookup(target):
        asked.append(target)

    ConnectorEnvironment({"spec": SPEC, "control_target": "live"}, limits_lookup=lookup)
    ConnectorEnvironment({"spec": SPEC}, limits_lookup=lookup)
    assert asked == ["live", None]  # None: the most restrictive across targets


def test_the_channel_max_step_is_carried_on_the_variable():
    env = _limited({"VCM1:SP": _Limit(max_step=0.5)})
    assert env.variables["VCM1"].max_step == 0.5
    assert env.variables["VCM2"].max_step is None


@pytest.mark.parametrize(
    ("window", "refused"),
    [pytest.param(0.5, False, id="window_equal_to_max_step"), pytest.param(0.6, True, id="wider")],
)
def test_a_search_window_wider_than_max_step_is_refused_at_resolution(fake, window, refused):
    """The engine moves a variable in one write, so a window the step limit cuts
    would have moves inside it refused mid-run: refuse it when the run is resolved."""
    env = ConnectorEnvironment(
        {"spec": SPEC, "control_target": "va"},
        interface_factory=lambda t, m: fake,
        limits_lookup=lambda t: {"VCM1:SP": _Limit(max_step=0.5)},
    )
    config = OptimizationConfig(
        environment_name="orbit",
        control_target="va",
        objective_name="rms_y",
        variables=["VCM1"],
        variable_overrides={"VCM1": VariableOverride(min=0.0, max=window)},
    )
    if refused:
        with pytest.raises(ValueError, match=r"wider than the max_step 0.5 .*VCM1:SP"):
            resolve_draft_config(config, env)
    else:
        resolve_draft_config(config, env)
        assert config.variable_overrides is not None
        assert config.variable_overrides["VCM1"].max == pytest.approx(0.5)
    assert fake.writes == []
