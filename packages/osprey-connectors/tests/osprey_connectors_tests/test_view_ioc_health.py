"""The simulator view reads the optional ``ioc_health`` field of ``addresses.json``, per model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors.simulation.view import (
    ADDRESSES_FILE,
    ADDRESSES_SCHEMA,
    ENGINEER,
    HEARTBEAT,
    IOC_HEALTH_SIGNALS,
    PERIODIC_IOC_HEALTH,
    SERVED_MODELS_FILE,
    SERVED_MODELS_SCHEMA,
    STATUS_SIGNAL,
    UPTIME,
    VARIABLES_FILE,
    VARIABLES_SCHEMA,
    SimulatorView,
    ViewSchemaError,
)


def _health(model: str) -> list[str]:
    return [f"T:SIM:{model}:{signal}" for signal in (HEARTBEAT, UPTIME, ENGINEER)]


def _view(tmp_path: Path, addresses: dict[str, Any], schema: str = ADDRESSES_SCHEMA) -> Path:
    """Write a view serving the physics model ``SR`` and the texture model."""
    documents = {
        ADDRESSES_FILE: {"schema": schema, "channels": [], **addresses},
        SERVED_MODELS_FILE: {"schema": SERVED_MODELS_SCHEMA, "models": ["SR", "texture"]},
        VARIABLES_FILE: {
            "schema": VARIABLES_SCHEMA,
            "code": "T",
            "models": [
                {"name": "SR", "engine": "pyat", "served": True},
                {"name": "texture", "engine": "texture", "served": True},
            ],
            "channels": [],
        },
    }
    for name, document in documents.items():
        (tmp_path / name).write_text(json.dumps(document), encoding="utf-8")
    return tmp_path


def test_a_view_without_the_field_loads_and_yields_nothing(tmp_path: Path) -> None:
    view = SimulatorView.open(_view(tmp_path, {"status": ["T:SIM:SR:STATUS"]}))

    assert view.ioc_health_by_model() == {}


def test_a_view_without_the_field_needs_no_other_file(tmp_path: Path) -> None:
    (tmp_path / ADDRESSES_FILE).write_text(json.dumps({"schema": ADDRESSES_SCHEMA}), "utf-8")

    assert SimulatorView.open(tmp_path).ioc_health_by_model() == {}


def test_an_empty_field_yields_nothing(tmp_path: Path) -> None:
    view = SimulatorView.open(_view(tmp_path, {"status": [], "ioc_health": []}))

    assert view.ioc_health_by_model() == {}


def test_every_listed_model_yields_its_addresses_including_texture(tmp_path: Path) -> None:
    listed = _health("SR") + _health("texture")
    view = SimulatorView.open(_view(tmp_path, {"status": [], "ioc_health": listed}))

    by_model = view.ioc_health_by_model()

    assert by_model == {
        "SR": {
            "HEARTBEAT": "T:SIM:SR:HEARTBEAT",
            "UPTIME": "T:SIM:SR:UPTIME",
            "ENGINEER": "T:SIM:SR:ENGINEER",
        },
        "texture": {
            "HEARTBEAT": "T:SIM:texture:HEARTBEAT",
            "UPTIME": "T:SIM:texture:UPTIME",
            "ENGINEER": "T:SIM:texture:ENGINEER",
        },
    }
    assert list(by_model["SR"]) == ["HEARTBEAT", "UPTIME", "ENGINEER"]


def test_the_signals_are_the_views_vocabulary() -> None:
    assert IOC_HEALTH_SIGNALS == (HEARTBEAT, UPTIME, ENGINEER)
    assert set(PERIODIC_IOC_HEALTH) < set(IOC_HEALTH_SIGNALS)
    assert ENGINEER not in PERIODIC_IOC_HEALTH
    assert STATUS_SIGNAL not in IOC_HEALTH_SIGNALS


def test_the_mapping_is_read_only(tmp_path: Path) -> None:
    view = SimulatorView.open(_view(tmp_path, {"status": [], "ioc_health": _health("SR")}))

    with pytest.raises(TypeError):
        view.ioc_health_by_model()["texture"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        view.ioc_health_by_model()["SR"][HEARTBEAT] = "T:SIM:SR:OTHER"  # type: ignore[index]


@pytest.mark.parametrize(
    "addresses",
    [{"ioc_health": []}, {"ioc_health": _health("SR") + _health("texture")}],
    ids=["empty", "listed"],
)
def test_status_addresses_ignore_the_field(tmp_path: Path, addresses: dict[str, Any]) -> None:
    view = SimulatorView.open(_view(tmp_path, {"status": ["T:SIM:SR:STATUS"], **addresses}))

    assert view.status_addresses() == {"SR": "T:SIM:SR:STATUS"}


@pytest.mark.parametrize(
    "listed",
    ["T:SIM:SR:HEARTBEAT", ["SR:RF:TEMP"], ["T:SIM:HEARTBEAT"], [1.0], ["T:SIM:SR:TEMPERATURE"]],
    ids=["word", "other-address", "no-model", "number", "unknown-signal"],
)
def test_a_field_other_than_health_addresses_is_refused(tmp_path: Path, listed: Any) -> None:
    view = SimulatorView.open(_view(tmp_path, {"status": [], "ioc_health": listed}))

    with pytest.raises(ViewSchemaError, match="ioc_health"):
        view.ioc_health_by_model()


def test_another_addresses_schema_is_refused_with_the_field_present(tmp_path: Path) -> None:
    path = _view(
        tmp_path,
        {"status": [], "ioc_health": _health("SR")},
        schema="osprey.facility.addresses/2",
    )

    with pytest.raises(ViewSchemaError, match="rebuild"):
        SimulatorView.open(path)
