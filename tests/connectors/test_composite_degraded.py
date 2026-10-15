"""The composite's third model state: a built model whose engine reports a condition."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from lume.model import LUMEModel
from lume.variables import ScalarVariable, Variable

from osprey_connectors.simulation import composite as composite_module
from osprey_connectors.simulation.composite import STATUS_MAX_BYTES, Composite, WriteRefusedError
from osprey_connectors.simulation.status import model_built, model_status
from osprey_connectors.simulation.view import SCHEMAS

T0 = 1_760_000_000.0
STATUS_M = "T:SIM:M:STATUS"
STATUS_N = "T:SIM:N:STATUS"
NO_BEAM = "no beam"
LOST_ABOVE = 50.0
REFUSED_ABOVE = 100.0
UDF = {"condition": "udf"}
M_CHANNELS = ["M:SP", "M:RB", "M:OTHER", STATUS_M]


# -- a stub engine that can report a condition ---------------------------------


class StubModel(LUMEModel):
    """Setpoints and readbacks of one wiring; the beam is lost above a threshold."""

    def __init__(
        self, name: str, wiring: list[Mapping[str, Any]], settings: Mapping[str, Any]
    ) -> None:
        self.text = str(settings.get("text", NO_BEAM))
        self.broken = bool(settings.get("broken"))
        self.listed = frozenset({f"{name}:RB", "optics"})
        self.truth: dict[str, float] = {}
        self._defaults: dict[str, float] = {"knob": 1.0}
        self._variables: dict[str, Variable] = {}
        for entry in wiring:
            address = str(entry["address"])
            write = entry.get("direction") == "write"
            self._variables[address] = ScalarVariable(name=address, read_only=not write)
            (self._defaults if write else self.truth)[address] = float(entry.get("default", 0.0))
        self._variables["knob"] = ScalarVariable(name="knob", read_only=False)
        self.inputs: dict[str, float] = {}
        self.reset()

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    @property
    def lost(self) -> bool:
        return any(value > LOST_ABOVE for value in self.inputs.values())

    def reset(self) -> None:
        self.inputs = dict(self._defaults)

    def _set(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            if float(value) > REFUSED_ABOVE:
                raise ValueError(f"{name} cannot reach {value}")
        self.inputs.update({name: float(value) for name, value in values.items()})

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self.inputs.get(name, self.truth.get(name)) for name in names}


def _build(model, wiring, deck, settings, active=None):
    del deck, active
    if settings.get("fail"):
        raise RuntimeError(settings["fail"])
    return StubModel(model, list(wiring), settings)


def _condition(model: StubModel) -> tuple[str, frozenset[str]] | None:
    if not model.lost:
        return None
    if model.broken:
        raise RuntimeError("the condition lookup broke")
    return model.text, model.listed


ENGINES = {
    "stating": SimpleNamespace(build=_build, condition=_condition),
    "plain": SimpleNamespace(build=_build),
}


@pytest.fixture(autouse=True)
def _stub_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Composite, "_engine", staticmethod(lambda name: ENGINES[name]))
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)


# -- a synthetic view of two physics models ------------------------------------


def _channel(address: str, owner: str, role: str = "readback") -> dict[str, Any]:
    return {
        "address": address,
        "role": role,
        "pair": address if role == "setpoint" else None,
        "value_type": "float",
        "unit": None,
        "description": None,
        "writable": role == "setpoint",
        "value_range": None,
        "owner": owner,
        "on": None,
    }


def _model(name: str, engine: str, settings: Mapping[str, Any], sp: float) -> dict[str, Any]:
    described = {"plane": None, "refresh": "pass"}
    return {
        "name": name,
        "engine": engine,
        "served": True,
        "settings": dict(settings),
        "deck": None,
        "wiring": [
            {"id": "1", "address": f"{name}:SP", "direction": "write", "default": sp}
            | {"role": "setpoint"}
            | described,
            {"id": "2", "address": f"{name}:RB", "direction": "read", "default": 4.0}
            | {"role": "readback"}
            | described,
            {"id": "3", "address": f"{name}:OTHER", "direction": "read", "default": 6.0}
            | {"role": "readback"}
            | described,
        ],
    }


def _view(
    path: Path, *, engine: str = "stating", settings: Mapping[str, Any] | None = None, sp: float
) -> Path:
    channels = [
        channel
        for name in ("M", "N")
        for channel in (
            _channel(f"{name}:SP", name, "setpoint"),
            _channel(f"{name}:RB", name),
            _channel(f"{name}:OTHER", name),
        )
    ] + [_channel("T:SP", "texture", "setpoint")]
    documents = {
        "served_models.json": {"models": ["M", "N", "texture"]},
        "addresses.json": {
            "channels": sorted(channel["address"] for channel in channels),
            "status": [STATUS_M, STATUS_N],
        },
        "variables.json": {
            "code": "T",
            "models": [
                _model("M", engine, settings or {}, sp),
                _model("N", engine, {}, 2.0),
                {
                    "name": "texture",
                    "engine": "texture",
                    "served": True,
                    "settings": {},
                    "deck": None,
                    "wiring": [],
                },
            ],
            "channels": sorted(channels, key=lambda channel: channel["address"]),
        },
        "seeds.json": {"seeds": {"T:SP": {"nominal": 5.0}}},
        "scenarios.json": {"scenarios": [{"name": "nominal"}]},
    }
    view = path / "simulator"
    view.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (view / name).write_text(
            json.dumps({"schema": SCHEMAS[name], **document}), encoding="utf-8"
        )
    return view


def _composite(tmp_path: Path, *, sp: float = 2.0, **view: Any) -> Composite:
    return Composite(
        _view(tmp_path, sp=sp, **view),
        state_dir=tmp_path / "state",
        log_dir=tmp_path / "log",
        clock=lambda: T0,
    )


def _log(tmp_path: Path, model: str = "M") -> list[dict[str, Any]]:
    lines = (tmp_path / "log" / f"{model}.log").read_text(encoding="utf-8").splitlines()
    return [
        {key: value for key, value in json.loads(line).items() if key not in ("instance", "pid")}
        for line in lines
    ]


def _state(composite: Composite) -> tuple[str, dict[str, dict[str, str]]]:
    """Model ``M`` as a reader sees it: its status and its channels' conditions."""
    return composite.status("M"), composite.output_severity(M_CHANNELS)


OK = ("ok", {})
DEGRADED = (NO_BEAM, {"M:RB": UDF})


# -- each write path flips status and severities -------------------------------


def test_a_model_whose_engine_states_no_condition_is_ok(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    assert _state(composite) == OK
    assert composite.get(STATUS_M) == "ok"
    assert composite.built("M") is True


def test_a_channel_write_flips_status_and_severities(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    composite.set({"M:SP": 60.0})
    assert _state(composite) == DEGRADED
    assert composite.get(STATUS_M) == NO_BEAM
    assert composite.held([STATUS_M]) == {STATUS_M: NO_BEAM}

    composite.set({"M:SP": 3.0})
    assert _state(composite) == OK
    assert composite.get(STATUS_M) == "ok"


def test_a_model_variable_write_flips_status_and_severities(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    composite.model_set({"M/knob": 60.0})
    assert _state(composite) == DEGRADED

    composite.model_set({"M/knob": 1.0})
    assert _state(composite) == OK


def test_a_rolled_back_batch_leaves_an_ok_model_ok(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    with pytest.raises(WriteRefusedError):
        composite.set({"M:SP": 60.0, "N:SP": 500.0})

    assert composite.held(["M:SP"]) == {"M:SP": 2.0}
    assert _state(composite) == OK


def test_a_rolled_back_batch_leaves_a_degraded_model_degraded(tmp_path: Path) -> None:
    composite = _composite(tmp_path)
    composite.set({"M:SP": 60.0})

    with pytest.raises(WriteRefusedError):
        composite.set({"M:SP": 3.0, "N:SP": 500.0})

    assert composite.held(["M:SP"]) == {"M:SP": 60.0}
    assert _state(composite) == DEGRADED


def test_a_write_straight_to_the_model_shows_at_the_next_read(tmp_path: Path) -> None:
    composite = _composite(tmp_path)
    model = composite._children["M"].model
    assert model is not None

    model.set({"knob": 60.0})

    assert _state(composite) == DEGRADED


def test_reset_clears_a_condition_the_start_state_does_not_have(tmp_path: Path) -> None:
    composite = _composite(tmp_path)
    composite.set({"M:SP": 60.0})

    composite.reset()

    assert _state(composite) == OK


def test_a_model_built_under_its_condition_is_degraded_from_the_start(tmp_path: Path) -> None:
    composite = _composite(tmp_path, sp=60.0)

    assert _state(composite) == DEGRADED
    assert composite.built("M") is True


# -- what a degraded model serves ----------------------------------------------


def test_a_listed_channel_keeps_serving_its_value(tmp_path: Path) -> None:
    composite = _composite(tmp_path)
    composite.set({"M:SP": 60.0})

    assert composite.get(["M:SP", "M:RB", "M:OTHER"]) == {
        "M:SP": 60.0,
        "M:RB": 4.0,
        "M:OTHER": 6.0,
    }
    assert composite.held(["M:RB"]) == {"M:RB": 4.0}


def test_only_the_listed_channels_of_the_degraded_model_report_udf(tmp_path: Path) -> None:
    composite = _composite(tmp_path)
    composite.set({"M:SP": 60.0})

    everything = sorted(composite.supported_variables)

    assert composite.output_severity(everything) == {"M:RB": UDF}
    assert composite.status("N") == "ok"


def test_a_degraded_model_keeps_its_variables_reachable(tmp_path: Path) -> None:
    composite = _composite(tmp_path)
    composite.model_set({"M/knob": 60.0})

    assert composite.model_get(["M/knob"]) == {"M/knob": 60.0}
    assert "M/knob" in composite.model_variables()


def test_a_long_condition_text_is_capped_like_a_failure_text(tmp_path: Path) -> None:
    composite = _composite(tmp_path, sp=60.0, settings={"text": "Ä" * 1024})

    status = composite.status("M")

    assert len(status.encode("utf-8")) <= STATUS_MAX_BYTES
    assert status.endswith("…")
    assert composite.get(STATUS_M) == status


# -- failed stays failed -------------------------------------------------------


def test_a_failed_model_is_not_built_and_reads_as_before(tmp_path: Path) -> None:
    composite = _composite(tmp_path, settings={"fail": "the deck does not load"})

    assert composite.built("M") is False
    assert composite.built("N") is True
    assert composite.status("M") == "the deck does not load"
    assert math.isnan(composite.get("M:RB"))
    assert composite.output_severity(M_CHANNELS) == {"M:SP": UDF, "M:RB": UDF, "M:OTHER": UDF}


def test_a_condition_lookup_that_raises_fails_the_model(tmp_path: Path) -> None:
    composite = _composite(tmp_path, settings={"broken": True})

    composite.set({"M:SP": 60.0})

    assert composite.status("M") == "the condition lookup broke"
    assert composite.built("M") is False
    assert composite.output_severity(["M:OTHER"]) == {"M:OTHER": UDF}


def test_built_names_the_served_models_for_an_unknown_one(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    with pytest.raises(ValueError, match=r"model 'X' is not served; served: \['M', 'N'\]"):
        composite.built("X")


def test_the_connector_lookups_answer_from_the_composite(tmp_path: Path) -> None:
    connector = SimpleNamespace(_composite=_composite(tmp_path, sp=60.0))

    assert model_status(connector, "M") == NO_BEAM
    assert model_built(connector, "M") is True
    with pytest.raises(ValueError, match="serves no simulator in process"):
        model_built(SimpleNamespace(), "M")


# -- an engine without the name ------------------------------------------------


def test_an_engine_without_condition_is_never_degraded(tmp_path: Path) -> None:
    composite = _composite(tmp_path, engine="plain")

    composite.set({"M:SP": 60.0})
    composite.model_set({"M/knob": 60.0})

    assert _state(composite) == OK
    assert composite.get(STATUS_M) == "ok"
    assert composite.built("M") is True
    assert _log(tmp_path) == [{"model": "M", "event": "built", "status": "ok"}]


# -- the model log -------------------------------------------------------------


def test_a_build_logs_the_status_the_model_starts_with(tmp_path: Path) -> None:
    _composite(tmp_path, sp=60.0)

    assert _log(tmp_path, "M") == [{"model": "M", "event": "built", "status": NO_BEAM}]
    assert _log(tmp_path, "N") == [{"model": "N", "event": "built", "status": "ok"}]


def test_a_write_logs_the_condition_only_when_it_changed(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    composite.set({"M:SP": 60.0})
    composite.set({"M:SP": 70.0})
    composite.model_set({"M/knob": 2.0})
    composite.set({"M:SP": 3.0})
    composite.model_set({"M/knob": 60.0})
    composite.reset()

    assert _log(tmp_path, "M") == [
        {"model": "M", "event": "built", "status": "ok"},
        {"model": "M", "event": "condition", "status": NO_BEAM},
        {"model": "M", "event": "condition", "status": "ok"},
        {"model": "M", "event": "condition", "status": NO_BEAM},
        {"model": "M", "event": "condition", "status": "ok"},
    ]
    assert _log(tmp_path, "N") == [{"model": "N", "event": "built", "status": "ok"}]


def test_a_rolled_back_batch_logs_no_condition_it_did_not_keep(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    with pytest.raises(WriteRefusedError):
        composite.set({"M:SP": 60.0, "N:SP": 500.0})

    assert _log(tmp_path, "M") == [{"model": "M", "event": "built", "status": "ok"}]


def test_reading_logs_nothing(tmp_path: Path) -> None:
    composite = _composite(tmp_path)
    model = composite._children["M"].model
    assert model is not None
    model.set({"knob": 60.0})

    assert _state(composite) == DEGRADED
    assert composite.get(STATUS_M) == NO_BEAM

    assert _log(tmp_path, "M") == [{"model": "M", "event": "built", "status": "ok"}]


def test_no_reply_or_record_names_the_state_by_a_word(tmp_path: Path) -> None:
    composite = _composite(tmp_path)
    composite.set({"M:SP": 60.0})

    everything = sorted(composite.supported_variables)
    replies = {
        "status": composite.status("M"),
        "get": composite.get(everything),
        "held": composite.held(everything),
        "severity": composite.output_severity(everything),
        "log": _log(tmp_path, "M"),
    }

    said = json.dumps(replies)
    assert "degraded" not in said
    assert '"state"' not in said
