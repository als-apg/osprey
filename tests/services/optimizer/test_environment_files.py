"""Environment files: parsing, channel-range expansion, and what they refuse."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from osprey.services.optimizer.environments.spec import (
    EnvironmentFile,
    expand_channels,
    load_environment_file,
)
from osprey.services.optimizer.models.protocols import PVObservable, ReducedObservable

VA_ORBIT = """
name: va_vertical_orbit
variables:
  SR:MAG:VCM:01:CURRENT:SP:
    readback: SR:MAG:VCM:01:CURRENT:RB
    min: -12
    max: 12
    step_size: 0.01
    units: A
observables:
  orbit_rms_y:
    reduce: rms
    channels: ["SR:DIAG:BPM:{01..72}:POSITION:Y"]
    direction: MINIMIZE
  current:
    channel: SR:DIAG:DCCT:01:CURRENT:RB
    selectable_as_objective: false
default_objective: orbit_rms_y
"""


def _minimal(**overrides):
    data = {
        "name": "env",
        "variables": {"A:SP": {"min": 0, "max": 1}},
        "observables": {"obj": {"channel": "OBJ"}},
    }
    data.update(overrides)
    return data


def test_range_expands_zero_padded():
    assert expand_channels("BPM:{01..03}:Y") == ["BPM:01:Y", "BPM:02:Y", "BPM:03:Y"]
    assert expand_channels("NO:RANGE") == ["NO:RANGE"]


def test_two_ranges_expand_as_a_product():
    assert expand_channels("C{1..2}:B{1..2}") == ["C1:B1", "C1:B2", "C2:B1", "C2:B2"]


def test_descending_range_is_refused():
    with pytest.raises(ValueError, match="Descending"):
        expand_channels("BPM:{03..01}")


def test_va_orbit_file_loads(tmp_path: Path):
    path = tmp_path / "va.yml"
    path.write_text(VA_ORBIT)
    env = load_environment_file(path)

    var = env.variable_metadata()["SR:MAG:VCM:01:CURRENT:SP"]
    assert var.setpoint_pv == "SR:MAG:VCM:01:CURRENT:SP"
    assert var.readback_pv == "SR:MAG:VCM:01:CURRENT:RB"
    assert (var.min_value, var.max_value, var.step_size) == (-12, 12, 0.01)

    obs = env.observable_metadata()
    rms = obs["orbit_rms_y"]
    assert isinstance(rms, ReducedObservable)
    assert len(rms.channels) == 72
    assert rms.channels[0] == "SR:DIAG:BPM:01:POSITION:Y"
    assert rms.channels[-1] == "SR:DIAG:BPM:72:POSITION:Y"
    assert isinstance(obs["current"], PVObservable)


def test_readback_defaults_to_setpoint():
    var = EnvironmentFile.model_validate(_minimal()).variable_metadata()["A:SP"]
    assert var.readback_pv == var.setpoint_pv == "A:SP"


def test_readback_tolerance_defaults_to_step_size():
    env = EnvironmentFile.model_validate(
        _minimal(
            variables={
                "A": {"min": 0, "max": 1, "step_size": 0.1},
                "B": {"min": 0, "max": 1, "readback_tolerance": 0.5},
            }
        )
    )
    assert env.readback_tolerance("A") == 0.1
    assert env.readback_tolerance("B") == 0.5


@pytest.mark.parametrize(
    "observable",
    [
        {},  # neither channel nor reduce
        {"channel": "X", "reduce": "rms", "channels": ["Y"]},  # both
        {"reduce": "rms"},  # reduce without channels
        {"channel": "X", "channels": ["Y"]},  # channels without reduce
        {"reduce": "median", "channels": ["Y"]},  # unknown reduction
    ],
)
def test_malformed_observables_are_refused(observable):
    with pytest.raises(ValidationError):
        EnvironmentFile.model_validate(_minimal(observables={"o": observable}))


def test_unknown_keys_are_refused():
    with pytest.raises(ValidationError):
        EnvironmentFile.model_validate(_minimal(variables={"A": {"min": 0, "max": 1, "maxx": 2}}))


def test_min_above_max_is_refused():
    with pytest.raises(ValidationError):
        EnvironmentFile.model_validate(_minimal(variables={"A": {"min": 2, "max": 1}}))


def test_default_objective_must_be_a_selectable_observable():
    with pytest.raises(ValidationError, match="not an observable"):
        EnvironmentFile.model_validate(_minimal(default_objective="nope"))
    with pytest.raises(ValidationError, match="not selectable"):
        EnvironmentFile.model_validate(
            _minimal(
                observables={"o": {"channel": "X", "selectable_as_objective": False}},
                default_objective="o",
            )
        )


def test_a_file_with_no_selectable_objective_is_refused():
    with pytest.raises(ValidationError, match="no observable is selectable"):
        EnvironmentFile.model_validate(
            _minimal(observables={"o": {"channel": "X", "selectable_as_objective": False}})
        )


def test_bad_range_fails_at_load(tmp_path: Path):
    path = tmp_path / "bad.yml"
    path.write_text(VA_ORBIT.replace("{01..72}", "{72..01}"))
    with pytest.raises(ValueError, match="Descending"):
        load_environment_file(path)
