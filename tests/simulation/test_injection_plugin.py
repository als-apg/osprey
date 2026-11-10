"""The injection engine plug-in: descriptions, refusals, start values, build and discovery."""

from __future__ import annotations

import ast
import copy
import importlib
import re
import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from osprey.facility.errors import FacilityBuildError
from osprey.simulation.engines import ENTRY_POINT_GROUP, TIME_CONTRACT, EnginePlugin
from osprey.simulation.engines import injection as engine

MODEL = "CHAIN"
T0 = 1024.0
PERIOD = 2.0
WIDTH = 1.0
PER_CHARGE = 3.0

SETTINGS = {
    "injection": {
        "cycle": {"period_s": PERIOD, "width_s": WIDTH},
        "stages": ["transfer", "injection"],
        "current_per_charge": PER_CHARGE,
    }
}


def _record(address: str, direction: str = "read", **block: Any) -> dict[str, Any]:
    return {
        "id": f"{MODEL}/{address}",
        "address": address,
        "direction": direction,
        "engine": block,
    }


def _wiring() -> list[dict[str, Any]]:
    return [
        _record("req", "write", signal="request"),
        _record("src", "write", signal="parameter", quantity="charge", start=4.0),
        _record("eff0", "write", signal="parameter", quantity="efficiency", stage=0, start=0.5),
        _record("eff1", "write", signal="parameter", quantity="efficiency", stage=1, start=0.25),
        _record("stored", "write", signal="parameter", quantity="current", start=10.0),
        _record("count", signal="counter", offset_s=0.125),
        _record("charge0", signal="charge", offset_s=0.25, stage=0),
        _record("eff0_read", signal="efficiency", offset_s=0.375, stage=0),
        _record("charge1", signal="charge", offset_s=0.5, stage=1),
        _record("stored_read", signal="current", offset_s=0.75),
        _record("stored_fast", signal="current_fast"),
        _record("busy", signal="busy"),
    ]


def _without(*addresses: str) -> list[dict[str, Any]]:
    return [r for r in _wiring() if r["address"] not in addresses]


def _settings(**block: Any) -> dict[str, Any]:
    settings = copy.deepcopy(SETTINGS)
    settings["injection"].update(block)
    return settings


def _settings_without(key: str) -> dict[str, Any]:
    settings = copy.deepcopy(SETTINGS)
    del settings["injection"][key]
    return settings


# -- discovery ---------------------------------------------------------------------


class TestEntryPoint:
    def test_group_loads_the_module(self):
        loaded = entry_points(group=ENTRY_POINT_GROUP)["injection"].load()
        assert loaded is importlib.import_module("osprey.simulation.engines.injection")

    def test_satisfies_the_contract(self):
        assert isinstance(engine, EnginePlugin)

    def test_states_the_time_contract(self):
        for name in TIME_CONTRACT:
            assert callable(getattr(engine, name)), name

    def test_starts_values_without_a_deck(self):
        assert engine.deckless_start_values is True

    def test_import_leaves_lume_unloaded(self):
        code = (
            "import sys\n"
            "import osprey.simulation.engines.injection\n"
            "loaded = [m for m in ('lume', 'lume.model', 'numpy') if m in sys.modules]\n"
            "assert not loaded, loaded\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr


# -- describe ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("address", "role", "refresh"),
    [
        ("req", "setpoint", "pass"),
        ("src", "setpoint", "pass"),
        ("eff0", "setpoint", "pass"),
        ("stored", "setpoint", "pass"),
        ("count", "output", "event"),
        ("charge0", "output", "event"),
        ("eff0_read", "output", "event"),
        ("stored_read", "output", "pass"),
        ("stored_fast", "output", "fast"),
        ("busy", "output", "event"),
    ],
)
def test_describe(address: str, role: str, refresh: str):
    record = next(r for r in _wiring() if r["address"] == address)
    description = engine.describe(record)
    assert description["role"] == role
    assert description["refresh"] == refresh
    assert description["plane"] is None


def test_describe_reads_attributes_too():
    class Record:
        address = "count"
        engine = {"signal": "counter", "offset_s": 0.5}

    assert engine.describe(Record())["refresh"] == "event"


@pytest.mark.parametrize(
    ("block", "words"),
    [
        (None, "no engine block"),
        ({}, "signal"),
        ({"signal": "beam"}, "beam"),
        ({"signal": "counter"}, "offset_s"),
        ({"signal": "current"}, "offset_s"),
        ({"signal": "counter", "offset_s": -0.5}, "negative"),
        ({"signal": "counter", "offset_s": True}, "offset_s"),
        ({"signal": "charge", "offset_s": 0.5}, "stage"),
        ({"signal": "efficiency", "offset_s": 0.5, "stage": -1}, "stage"),
        ({"signal": "charge", "offset_s": 0.5, "stage": "0"}, "stage"),
        ({"signal": "counter", "offset_s": 0.5, "stage": 0}, "stage"),
        ({"signal": "parameter"}, "quantity"),
        ({"signal": "parameter", "quantity": "voltage"}, "voltage"),
        ({"signal": "parameter", "quantity": "efficiency"}, "stage"),
        ({"signal": "parameter", "quantity": "charge", "stage": 0}, "stage"),
        ({"signal": "parameter", "quantity": "charge", "start": "high"}, "start"),
        ({"signal": "request", "offset_s": 0.5}, "offset_s"),
        ({"signal": "current_fast", "scale": "milli"}, "scale"),
        ({"signal": "busy", "scale": 2.0}, "scale"),
        ({"signal": "busy", "colour": "red"}, "colour"),
    ],
)
def test_describe_refuses_a_record(block: Any, words: str):
    record = {"id": f"{MODEL}/bad", "address": "bad", "engine": block}
    with pytest.raises(ValueError, match=re.escape(words)) as caught:
        engine.describe(record)
    assert "bad" in str(caught.value)


# -- start values ------------------------------------------------------------------


def test_start_values_are_each_setpoints_declared_start():
    values = engine.start_values(None, _wiring(), SETTINGS)
    assert values == {"req": 0.0, "src": 4.0, "eff0": 0.5, "eff1": 0.25, "stored": 10.0}


def test_start_values_accept_the_build_keywords():
    values = engine.start_values(
        None, _wiring(), MappingProxyType(SETTINGS), readbacks={}, value_types={}, shapes={}
    )
    assert values["src"] == 4.0


def test_a_parameter_without_a_start_starts_at_zero():
    wiring = _wiring()
    del wiring[1]["engine"]["start"]
    assert engine.start_values(None, wiring, SETTINGS)["src"] == 0.0


def test_a_chain_without_a_request_has_start_values():
    assert "req" not in engine.start_values(None, _without("req"), SETTINGS)


def _stop(wiring: Any, settings: Any, deck: Any = None) -> FacilityBuildError:
    with pytest.raises(FacilityBuildError) as caught:
        engine.start_values(deck, wiring, settings)
    return caught.value


@pytest.mark.parametrize(
    ("wiring", "settings", "words"),
    [
        (
            [*_wiring(), _record("req2", "write", signal="request")],
            SETTINGS,
            "more than one request",
        ),
        (_wiring(), _settings_without("cycle"), "cycle"),
        (_wiring(), _settings_without("stages"), "stages"),
        (_wiring(), _settings_without("current_per_charge"), "current_per_charge"),
        (_wiring(), _settings(stages=[]), "stages"),
        (_wiring(), _settings(stages="two"), "stages"),
        (_wiring(), _settings(cycle={"period_s": PERIOD}), "width_s"),
        (_wiring(), _settings(cycle={"period_s": 0.0, "width_s": WIDTH}), "period_s"),
        (_wiring(), _settings(cycle={"period_s": PERIOD, "width_s": PERIOD}), "width_s"),
        (_wiring(), _settings(cycle="fast"), "cycle"),
        (_wiring(), _settings(current_per_charge="lots"), "current_per_charge"),
        (_wiring(), _settings(colour="red"), "colour"),
        (_wiring(), None, "settings"),
        (_wiring(), {"injection": "on"}, "not a map"),
        (_wiring(), _settings(stages=["transfer"]), "stage 1"),
        (
            [
                *_without("eff1"),
                _record("eff1", "write", signal="parameter", quantity="efficiency", stage=0),
            ],
            SETTINGS,
            "stage 0",
        ),
        (
            [*_wiring(), _record("charge9", signal="charge", offset_s=0.5, stage=2)],
            SETTINGS,
            "stage 2",
        ),
        (_without("src"), SETTINGS, "charge"),
        (_without("stored"), SETTINGS, "current"),
        (
            [*_wiring(), _record("src2", "write", signal="parameter", quantity="charge")],
            SETTINGS,
            "charge",
        ),
        (
            [*_wiring(), _record("stored_read2", signal="current", offset_s=0.875)],
            SETTINGS,
            "more than one current",
        ),
    ],
)
def test_start_values_refuse_the_model(wiring: Any, settings: Any, words: str):
    stop = _stop(wiring, settings)
    assert stop.kind == "engine-invalid"
    assert stop.record_id == MODEL
    assert stop.record_kind == "model"
    assert words in stop.detail


def test_start_values_refuse_a_record_by_its_id():
    wiring = [*_wiring(), _record("bad", signal="beam")]
    stop = _stop(wiring, SETTINGS)
    assert stop.kind == "engine-invalid"
    assert stop.record_id == f"{MODEL}/bad"
    assert stop.record_kind == "wiring"
    assert "beam" in stop.detail


def test_start_values_refuse_a_deck():
    stop = _stop(_wiring(), SETTINGS, deck=Path("ring.m"))
    assert stop.record_id == MODEL
    assert "deck" in stop.detail


# -- locate and prepare ------------------------------------------------------------


def test_locate_says_there_is_no_deck():
    with pytest.raises(FacilityBuildError) as caught:
        engine.locate(None, "Q1", model=MODEL)
    assert caught.value.kind == "engine-invalid"
    assert caught.value.record_id == MODEL
    assert "no deck" in caught.value.detail
    assert "Q1" in caught.value.detail


def test_prepare_says_there_is_no_deck():
    with pytest.raises(FacilityBuildError) as caught:
        engine.prepare(None, SETTINGS, model=MODEL)
    assert caught.value.kind == "engine-invalid"
    assert caught.value.record_id == MODEL
    assert "no deck" in caught.value.detail


# -- build -------------------------------------------------------------------------


def _build(wiring: Any = None, settings: Any = None, **active: Any):
    return engine.build(
        MODEL,
        _wiring() if wiring is None else wiring,
        None,
        MappingProxyType(SETTINGS if settings is None else settings),
        active=active,
    )


def test_a_scaled_current_reads_the_stored_current_times_its_scale():
    wiring = [*_wiring(), _record("stored_scaled", signal="current_fast", scale=0.5)]
    model = _build(wiring)
    assert model.get(["stored_fast", "stored_scaled"]) == {
        "stored_fast": 10.0,
        "stored_scaled": 5.0,
    }
    model.set({"stored": 30.0})
    assert model.get(["stored_scaled"]) == {"stored_scaled": 15.0}


def test_build_serves_every_record():
    model = _build()
    assert set(model.supported_variables) == {r["address"] for r in _wiring()}
    writable = {n for n, v in model.supported_variables.items() if not v.read_only}
    assert writable == {"req", "src", "eff0", "eff1", "stored"}


def test_build_starts_setpoints_at_their_engine_start():
    model = _build()
    assert model.get(["src", "eff0", "stored", "stored_fast"]) == {
        "src": 4.0,
        "eff0": 0.5,
        "stored": 10.0,
        "stored_fast": 10.0,
    }


def test_build_prefers_the_default_then_the_active_value():
    wiring = _wiring()
    for record in wiring:
        if record["address"] in ("src", "stored", "count"):
            record["default"] = 8.0
    model = _build(wiring, stored=20.0)
    assert model.get(["src", "stored", "count"]) == {"src": 8.0, "stored": 20.0, "count": 8.0}


def test_build_refuses_an_active_key_that_names_no_setpoint():
    with pytest.raises(ValueError, match="count"):
        _build(count=3.0)


def test_build_rechecks_the_model():
    with pytest.raises(FacilityBuildError) as caught:
        _build(settings=_settings_without("cycle"))
    assert caught.value.record_id == MODEL
    with pytest.raises(FacilityBuildError) as caught:
        _build([*_wiring(), _record("req2", "write", signal="request")])
    assert "more than one request" in caught.value.detail


def test_build_refuses_a_deck():
    with pytest.raises(FacilityBuildError):
        engine.build(MODEL, _wiring(), Path("ring.m"), SETTINGS)


def test_a_built_chain_runs_one_shot_through_the_time_contract():
    model = _build()
    engine.advance(model, T0)
    assert engine.emitted(model) == ["busy"]
    assert engine.next_deadline(model) == T0 + WIDTH
    model.set({"req": 1.0})
    engine.client_write(model, {"req": 1.0}, T0 + 0.0625)
    engine.advance(model, T0 + 0.0625 + 0.75)
    assert engine.emitted(model) == ["count", "charge0", "eff0_read", "charge1"]
    final = 4.0 * 0.5 * 0.25
    assert model.get(["count", "charge0", "charge1", "stored_read", "stored_fast"]) == {
        "count": 1.0,
        "charge0": 2.0,
        "charge1": final,
        "stored_read": 10.0 + final * PER_CHARGE,
        "stored_fast": 10.0 + final * PER_CHARGE,
    }


def test_a_chain_without_a_request_builds_and_starts_no_shot():
    model = _build(_without("req"))
    engine.advance(model, T0)
    engine.client_write(model, {"src": 1.0}, T0 + 0.0625)
    engine.advance(model, T0 + 0.875)
    assert model.get(["count"]) == {"count": 0.0}


# -- facility neutrality -----------------------------------------------------------


def test_the_module_names_no_facility_number():
    tree = ast.parse(Path(engine.__file__).read_text())
    numbers = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    }
    assert numbers <= {0, 1}
