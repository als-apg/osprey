"""Model inputs in the facility file: the ``inputs`` key and the build's refusals."""

from __future__ import annotations

import sys
import types
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest
import yaml

at = pytest.importorskip("at")

from osprey.facility.build import LATER_STAGES  # noqa: E402
from osprey.facility.compute import check_model_inputs, check_served_inputs  # noqa: E402
from osprey.facility.validate import StageReport, run_stages  # noqa: E402
from osprey.simulation.engines import ENTRY_POINT_GROUP  # noqa: E402
from osprey.simulation.engines import pyat as pyat_engine  # noqa: E402

CONSUMER = "fake_input_consumer_engine"
PLAIN = "fake_input_plain_engine"
ENGINE = {"attribute": "PolynomB", "index": 1}
REFUSED = "a refused input block"

#: Every input record describe_input saw, in call order.
DESCRIBED: list[dict[str, Any]] = []


def _like_pyat(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in vars(pyat_engine).items():
        if not key.startswith("__"):
            setattr(module, key, value)
    return module


def _describe_input(input_record: Any) -> None:
    DESCRIBED.append(dict(input_record))
    engine = input_record.get("engine")
    if isinstance(engine, dict) and engine.get("refuse"):
        raise ValueError(REFUSED)


def _take_inputs(*args: Any, **kwargs: Any) -> None:
    return None


@pytest.fixture(autouse=True)
def engines(monkeypatch: pytest.MonkeyPatch) -> None:
    """``pyat``; ``consumer``, pyat plus both input names; ``plain``, pyat without them."""
    consumer = _like_pyat(CONSUMER)
    consumer.describe_input = _describe_input
    consumer.take_inputs = _take_inputs
    plain = _like_pyat(PLAIN)
    for name in ("describe_input", "take_inputs"):
        if hasattr(plain, name):
            delattr(plain, name)
    monkeypatch.setitem(sys.modules, CONSUMER, consumer)
    monkeypatch.setitem(sys.modules, PLAIN, plain)
    found = metadata.EntryPoints(
        [
            metadata.EntryPoint(name="pyat", value=pyat_engine.__name__, group=ENTRY_POINT_GROUP),
            metadata.EntryPoint(name="consumer", value=CONSUMER, group=ENTRY_POINT_GROUP),
            metadata.EntryPoint(name="plain", value=PLAIN, group=ENTRY_POINT_GROUP),
        ]
    )

    def entry_points(**kwargs: Any) -> Any:
        return found

    monkeypatch.setattr(metadata, "entry_points", entry_points)
    DESCRIBED.clear()


def _save_deck(path: Path) -> None:
    lattice = at.Lattice(
        [
            at.Drift("D0", 1.0),
            at.Quadrupole("A", 0.2, 1.2),
            at.Drift("D1", 1.8),
            at.Quadrupole("B", 0.2, -0.4),
            at.Drift("D2", 0.5),
            at.Monitor("BPM1"),
        ],
        energy=3e9,
        particle="electron",
        periodicity=1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    at.save_lattice(lattice, str(path))


def _model(
    name: str, engine: str, wiring: list[dict[str, Any]], inputs: list[Any] | None
) -> dict[str, Any]:
    model: dict[str, Any] = {"name": name, "engine": engine, "deck": f"decks/{name}.json"}
    model["wiring"] = wiring
    if inputs is not None:
        model["inputs"] = inputs
    return model


def _files(
    *,
    up_inputs: list[Any] | None = None,
    down_inputs: list[Any] | None = None,
    up_engine: str = "pyat",
    down_engine: str = "consumer",
) -> dict[str, Any]:
    """Two models: ``up`` wires UP:X and UP:Y, ``down`` wires DOWN:X; T:X is texture's."""
    channels = [{"id": address, "unit": "mm"} for address in ("UP:X", "UP:Y", "DOWN:X", "T:X")]
    reading = {"axis": "x"}
    up = _model(
        "up",
        up_engine,
        [
            {"address": "UP:X", "element": "BPM1", "engine": reading},
            {"address": "UP:Y", "element": "BPM1", "engine": {"axis": "y"}},
        ],
        up_inputs,
    )
    down = _model(
        "down",
        down_engine,
        [{"address": "DOWN:X", "element": "BPM1", "engine": reading}],
        down_inputs,
    )
    return {"records/channels.yaml": channels, "models.yaml": [up, down]}


def _run(tmp_path: Path, files: dict[str, Any]) -> StageReport:
    root = tmp_path / "facility"
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    for model in files["models.yaml"]:
        _save_deck(root / model["deck"])
    return run_stages(root, project_name="p", later=LATER_STAGES)


def _lines(result: StageReport) -> list[str]:
    return [error.format_message() for error in result.errors]


def _only(result: StageReport):
    assert not result.ok
    (error,) = result.errors
    return error


# --- accepted --------------------------------------------------------------------------


def test_a_two_model_facility_with_inputs_builds_and_carries_them(tmp_path: Path) -> None:
    inputs = [
        {"address": "UP:Y", "engine": {"role": "charge"}},
        {"address": "UP:X", "engine": {"role": "shot"}},
    ]
    result = _run(tmp_path, _files(down_inputs=inputs))
    assert result.ok, _lines(result)
    document = result.validated.document
    assert document is not None
    models = {model["name"]: model for model in document["models"]}
    assert models["down"]["inputs"] == inputs
    assert "inputs" not in models["up"]
    assert DESCRIBED == inputs


def test_an_input_is_not_wiring_so_no_wiring_conflict(tmp_path: Path) -> None:
    result = _run(tmp_path, _files(down_inputs=[{"address": "UP:X"}]))
    assert result.ok, _lines(result)
    assert not any(error.kind == "wiring-conflict" for error in result.errors)


def test_a_facility_without_inputs_calls_no_input_hook(tmp_path: Path) -> None:
    result = _run(tmp_path, _files())
    assert result.ok, _lines(result)
    assert DESCRIBED == []
    assert all("inputs" not in model for model in result.validated.document["models"])


# --- refused ---------------------------------------------------------------------------


def test_a_cycle_is_refused_naming_its_inputs(tmp_path: Path) -> None:
    files = _files(
        up_engine="consumer",
        up_inputs=[{"address": "DOWN:X"}],
        down_inputs=[{"address": "UP:X"}],
    )
    error = _only(_run(tmp_path, files))
    assert error.kind == "value-invalid"
    assert (error.record_kind, error.record_id) == ("model", "down")
    assert "cycle" in error.detail
    assert "`UP:X`" in error.detail and "`DOWN:X`" in error.detail


def test_an_address_no_model_wires_is_refused(tmp_path: Path) -> None:
    error = _only(_run(tmp_path, _files(down_inputs=[{"address": "T:X"}])))
    assert error.kind == "reference-missing"
    assert (error.record_kind, error.record_id) == ("model", "down")
    assert "`T:X`" in error.detail
    assert DESCRIBED == []


def test_an_input_on_the_models_own_record_is_refused(tmp_path: Path) -> None:
    error = _only(_run(tmp_path, _files(down_inputs=[{"address": "DOWN:X"}])))
    assert error.kind == "value-invalid"
    assert (error.record_kind, error.record_id) == ("model", "down")
    assert "`DOWN:X`" in error.detail
    assert "own" in error.detail


def test_a_duplicate_address_is_refused(tmp_path: Path) -> None:
    inputs = [{"address": "UP:X", "engine": {"a": 1}}, {"address": "UP:X", "engine": {"a": 2}}]
    error = _only(_run(tmp_path, _files(down_inputs=inputs)))
    assert error.kind == "value-invalid"
    assert (error.record_kind, error.record_id) == ("model", "down")
    assert "`UP:X`" in error.detail
    assert "twice" in error.detail


def test_an_engine_without_take_inputs_is_refused(tmp_path: Path) -> None:
    files = _files(down_engine="plain", down_inputs=[{"address": "UP:X"}])
    error = _only(_run(tmp_path, files))
    assert error.kind == "engine-invalid"
    assert (error.record_kind, error.record_id) == ("model", "down")
    assert "`UP:X`" in error.detail
    assert "take_inputs" in error.detail


def test_an_engine_block_the_engine_refuses_is_refused_with_its_text(tmp_path: Path) -> None:
    inputs = [{"address": "UP:X", "engine": {"refuse": True}}]
    error = _only(_run(tmp_path, _files(down_inputs=inputs)))
    assert error.kind == "engine-invalid"
    assert (error.record_kind, error.record_id) == ("model", "down")
    assert "`UP:X`" in error.detail
    assert REFUSED in error.detail
    assert DESCRIBED == inputs


def test_an_input_from_a_model_that_is_not_served_is_refused(tmp_path: Path) -> None:
    result = _run(tmp_path, _files(down_inputs=[{"address": "UP:X"}]))
    assert result.ok, _lines(result)
    document = result.validated.document
    (error,) = check_served_inputs(document, ["down", "texture"])
    assert error.kind == "profile-invalid"
    assert (error.record_kind, error.record_id) == ("model", "down")
    assert "`UP:X`" in error.detail
    assert "up" in error.detail
    assert check_served_inputs(document, ["down", "up", "texture"]) == []
    assert check_served_inputs(document, ["up", "texture"]) == []


def test_the_check_runs_on_a_document_alone() -> None:
    document = {
        "models": [
            {"name": "up", "engine": "pyat", "wiring": [{"id": "up/A", "address": "A"}]},
            {"name": "down", "engine": "consumer", "inputs": [{"address": "B"}], "wiring": []},
        ]
    }
    (error,) = check_model_inputs(document)
    assert error.kind == "reference-missing"
    assert "`B`" in error.detail
