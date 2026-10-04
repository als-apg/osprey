"""The texture model over a synthetic simulator view."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from lume.exceptions import ReadOnlyError
from lume.variables import (
    EnumVariable,
    IntVariable,
    NDVariable,
    ScalarVariable,
    StrVariable,
)

from osprey_connectors.simulation import series
from osprey_connectors.simulation.texture import TextureModel

T0 = 1_760_000_000.0


def _channel(address: str, **fields: Any) -> dict[str, Any]:
    role = fields.pop("role", "readback")
    return {
        "address": address,
        "role": role,
        "pair": fields.pop("pair", address if role == "setpoint" else None),
        "value_type": fields.pop("value_type", "float"),
        "unit": fields.pop("unit", None),
        "description": None,
        "writable": fields.pop("writable", False),
        "value_range": fields.pop("value_range", None),
        "owner": fields.pop("owner", "texture"),
        **fields,
    }


def _view() -> tuple[dict[str, Any], dict[str, Any]]:
    channels = [
        _channel(
            "T:HEAT:SP", role="setpoint", pair="T:HEAT:RB", writable=True, value_range=[0, 50]
        ),
        _channel("T:HEAT:RB", unit="degC"),
        _channel("T:NOISY", unit="A"),
        _channel("T:CLAMPED"),
        _channel("T:SUM"),
        _channel("T:LONE:SP", role="setpoint", writable=True, value_range=[-5, 5]),
        _channel("T:FREE:SP", role="setpoint", writable=True, value_range=None),
        _channel("T:LOCKED:SP", role="setpoint", writable=False, value_range=[0, 1]),
        _channel("T:VALVE", role="setpoint", value_type="bool", writable=True),
        _channel("T:MODE", value_type="enum", options=["OFF", "STANDBY", "ON"]),
        _channel("T:COUNT", value_type="int"),
        _channel("T:NAME", value_type="string"),
        _channel("T:TRACE", value_type="waveform", shape=[2, 3]),
        _channel("T:ZERO"),
        _channel("P:SERVED:RB", owner="served"),
        _channel("P:IDLE:SP", role="setpoint", writable=True, owner="idle"),
    ]
    variables = {
        "schema": "simulator-variables",
        "code": "T",
        "models": [
            {
                "name": "idle",
                "engine": "x",
                "served": False,
                "settings": {},
                "deck": None,
                "wiring": [{"id": "w1", "address": "P:IDLE:SP", "default": 2.5}],
            },
            {
                "name": "served",
                "engine": "x",
                "served": True,
                "settings": {},
                "deck": None,
                "wiring": [{"id": "w2", "address": "P:SERVED:RB", "default": 9.0}],
            },
        ],
        "channels": channels,
    }
    seeds = {
        "schema": "simulator-seeds",
        "seeds": {
            "T:HEAT:SP": {"nominal": 20.0},
            "T:NOISY": {
                "nominal": 10.0,
                "noise": 0.5,
                "drift": {"amplitude": 1.0, "period_s": 600},
            },
            "T:CLAMPED": {"nominal": 3.0, "noise": 5.0, "clamp": [0.0, None]},
            "T:SUM": {"linear": {"T:HEAT:SP": 2.0, "T:NOISY": {"coefficient": -1.0}}},
            "T:VALVE": {"nominal": "TRUE"},
            "T:MODE": {"nominal": "STANDBY"},
            "T:COUNT": {"nominal": 7},
            "T:NAME": {"nominal": "alpha"},
            "T:TRACE": {"nominal": [[1, 2, 3], [4, 5, 6]]},
            "P:SERVED:RB": {"noise": 0.25},
        },
    }
    return variables, seeds


def _model(t_s: float = T0) -> TextureModel:
    variables, seeds = _view()
    return TextureModel(variables, seeds, clock=lambda: t_s)


def test_noise_is_keyed_on_address_and_epoch_ms():
    expected = (
        0.5
        * series.keyed_normals(series.channel_key_bytes("T:NOISY"), np.array([round(T0 * 1000)]))[0]
        + series.wander(series.channel_key_bytes("T:NOISY"), np.array([T0]), 1.0, 600)[0]
    )

    first = _model().get("T:NOISY")
    again = _model().get("T:NOISY")

    assert first == again
    assert first == pytest.approx(10.0 + expected)
    assert _model(T0 + 0.001).get("T:NOISY") != first


def test_motion_at_an_instant_matches_a_live_read_at_that_instant():
    model = _model()
    window = np.array([T0 - 1.0, T0, T0 + 1.0])

    archived = 10.0 + model.motion("T:NOISY", window)

    assert archived[1] == pytest.approx(model.get("T:NOISY"))


def test_a_channel_with_no_seed_is_a_static_zero():
    assert _model().get("T:ZERO") == 0.0
    assert _model(T0 + 5).get("T:ZERO") == 0.0


def test_clamp_holds_the_stated_side_and_leaves_the_null_side_open():
    model = _model()
    reads = [_model(T0 + k * 0.137).get("T:CLAMPED") for k in range(200)]

    assert min(reads) == 0.0
    assert max(reads) > 3.0
    assert model.clamp("T:CLAMPED", 1e9) == 1e9


def test_writing_a_setpoint_echoes_into_its_readback():
    model = _model()

    model.set({"T:HEAT:SP": 31.5})

    assert model.get(["T:HEAT:SP", "T:HEAT:RB"]) == {"T:HEAT:SP": 31.5, "T:HEAT:RB": 31.5}


def test_a_paired_readback_starts_at_its_setpoints_nominal():
    model = _model()

    assert model.get("T:HEAT:RB") == 20.0
    assert model.supported_variables["T:HEAT:RB"].default_value == 20.0


def test_a_refused_echo_leaves_the_setpoint_unchanged():
    variables, seeds = _view()
    variables["channels"] += [
        _channel("T:STEP:SP", role="setpoint", pair="T:STEP:RB", writable=True),
        _channel("T:STEP:RB", value_type="int"),
    ]
    seeds["seeds"]["T:STEP:SP"] = {"nominal": 3.0}
    model = TextureModel(variables, seeds, clock=lambda: T0)

    with pytest.raises(ValueError):
        model.set({"T:STEP:SP": 3.5})

    assert model.get(["T:STEP:SP", "T:STEP:RB"]) == {"T:STEP:SP": 3.0, "T:STEP:RB": 3}


def test_a_setpoint_paired_with_itself_changes_no_other_channel():
    model = _model()
    before = model.held([name for name in model.supported_variables if name != "T:LONE:SP"])

    model.set({"T:LONE:SP": 4.0})

    assert model.get("T:LONE:SP") == 4.0
    assert model.held(list(before)) == before


def test_a_linear_channel_follows_its_inputs_held_values():
    model = _model()

    assert model.get("T:SUM") == pytest.approx(2.0 * 20.0 - 10.0)
    model.set({"T:HEAT:SP": 25.0})
    assert model.get("T:SUM") == pytest.approx(2.0 * 25.0 - 10.0)


def test_a_linear_channel_reads_a_linear_input_as_its_weighted_sum():
    variables, seeds = _view()
    variables["channels"].append(_channel("T:CHAIN"))
    seeds["seeds"]["T:CHAIN"] = {"linear": {"T:SUM": 0.5}}
    model = TextureModel(variables, seeds, clock=lambda: T0)

    assert model.get("T:CHAIN") == pytest.approx(0.5 * (2.0 * 20.0 - 10.0))
    assert model.supported_variables["T:CHAIN"].default_value == pytest.approx(15.0)


def test_a_linear_channels_default_is_its_start_value():
    assert _model().supported_variables["T:SUM"].default_value == pytest.approx(30.0)


def test_bool_and_enum_hold_labels():
    model = _model()
    valve = model.supported_variables["T:VALVE"]
    mode = model.supported_variables["T:MODE"]

    assert isinstance(valve, EnumVariable)
    assert valve.options == ["FALSE", "TRUE"]
    assert valve.default_value == "TRUE"
    assert isinstance(mode, EnumVariable)
    assert mode.default_value == "STANDBY"
    model.set({"T:VALVE": "FALSE"})
    assert model.get("T:VALVE") == "FALSE"


def test_value_type_picks_the_variable_class():
    variables = _model().supported_variables

    assert type(variables["T:NOISY"]) is ScalarVariable
    assert type(variables["T:COUNT"]) is IntVariable
    assert type(variables["T:NAME"]) is StrVariable
    assert type(variables["T:TRACE"]) is NDVariable
    assert variables["T:TRACE"].shape == (2, 3)
    assert variables["T:HEAT:RB"].unit == "degC"
    assert _model().get("T:TRACE").tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_only_a_setpoint_the_view_marks_writable_is_settable():
    variables = _model().supported_variables

    assert variables["T:HEAT:SP"].read_only is False
    assert variables["T:HEAT:SP"].value_range == (0.0, 50.0)
    assert variables["T:LOCKED:SP"].read_only is True
    assert variables["T:HEAT:RB"].read_only is True
    with pytest.raises(ReadOnlyError):
        _model().set({"T:LOCKED:SP": 0.5})


def test_a_writable_setpoint_without_limits_has_no_range():
    variable = _model().supported_variables["T:FREE:SP"]

    assert variable.read_only is False
    assert variable.value_range is None


def test_a_one_sided_range_leaves_the_other_side_unbounded():
    variables, seeds = _view()
    variables["channels"].append(
        _channel("T:HALF:SP", role="setpoint", writable=True, value_range=[None, 3.0])
    )

    variable = TextureModel(variables, seeds).supported_variables["T:HALF:SP"]

    assert variable.value_range == (-math.inf, 3.0)


def test_a_served_model_keeps_its_channels_and_texture_still_moves_them():
    model = _model()

    assert "P:SERVED:RB" not in model.supported_variables
    assert model.motion("P:SERVED:RB", np.array([T0]))[0] != 0.0


def test_an_unserved_model_falls_to_texture_with_its_wiring_default():
    model = _model()

    assert model.get("P:IDLE:SP") == 2.5
    assert model.supported_variables["P:IDLE:SP"].default_value == 2.5


def test_reset_returns_every_channel_to_its_nominal():
    model = _model()
    model.set({"T:HEAT:SP": 40.0})

    model.reset()

    assert model.get(["T:HEAT:SP", "T:HEAT:RB"]) == {"T:HEAT:SP": 20.0, "T:HEAT:RB": 20.0}
