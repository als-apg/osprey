"""One process reads a simulator view once, and again only when a file of it changed."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors.directory._simulator import SimulatorDirectory
from osprey_connectors.service_answers import BackendSpec
from osprey_connectors.simulation import view as view_module
from osprey_connectors.simulation.view import (
    ADDRESSES_FILE,
    ADDRESSES_SCHEMA,
    VARIABLES_FILE,
    VARIABLES_SCHEMA,
    VIEW_RELPATH,
    SimulatorView,
)

#: Older than the time a written file's stat takes to be trusted.
_SETTLED_S = 60.0


@pytest.fixture(autouse=True)
def _no_view_held() -> Iterator[None]:
    """Each test starts and ends with no view held by the process."""
    view_module._OPENED.clear()
    yield
    view_module._OPENED.clear()


@pytest.fixture
def parses(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The name of each view file parsed, in order."""
    seen: list[str] = []
    real = json.loads

    def counting(text: str, *args: Any, **kwargs: Any) -> Any:
        parsed = real(text, *args, **kwargs)
        if isinstance(parsed, dict) and str(parsed.get("schema", "")).startswith("osprey.facility"):
            seen.append(str(parsed["schema"]))
        return parsed

    monkeypatch.setattr(view_module.json, "loads", counting)
    return seen


def _settle(path: Path) -> None:
    """Date ``path``'s last write well before now."""
    then = time.time() - _SETTLED_S
    os.utime(path, (then, then))


def _write(view_dir: Path, *, code: str = "FAC", settled: bool = True) -> Path:
    view_dir.mkdir(parents=True, exist_ok=True)
    wiring = {
        "id": "w1",
        "address": "FAC:MAG:01:SP",
        "direction": "write",
        "engine_role": "setpoint",
        "plane": "x",
        "refresh": "pass",
        "engine": {"fields": [1, 2, {"deep": [3]}]},
    }
    channel = {
        "address": "FAC:MAG:01:SP",
        "role": "setpoint",
        "pair": "FAC:MAG:01:SP",
        "value_type": "float",
        "unit": "A",
        "description": "A magnet. Its current.",
        "writable": True,
        "value_range": None,
        "owner": "RING",
        "on": None,
    }
    model = {
        "name": "RING",
        "engine": "pyat",
        "served": True,
        "settings": {"harmonics": [1, 2]},
        "wiring": [wiring],
    }
    documents = {
        ADDRESSES_FILE: {
            "schema": ADDRESSES_SCHEMA,
            "channels": ["FAC:MAG:01:SP"],
            "status": [f"{code}:SIM:RING:STATUS"],
        },
        VARIABLES_FILE: {
            "schema": VARIABLES_SCHEMA,
            "code": code,
            "models": [model],
            "channels": [channel],
        },
    }
    for name, document in documents.items():
        (view_dir / name).write_text(json.dumps(document), encoding="utf-8")
        if settled:
            _settle(view_dir / name)
    return view_dir


def test_a_second_open_is_given_the_view_the_first_read(tmp_path: Path, parses: list[str]) -> None:
    _write(tmp_path)

    first = SimulatorView.open(tmp_path)
    first.models()
    second = SimulatorView.open(tmp_path)
    second.models()
    second.status_addresses()

    assert second is first
    assert parses == [ADDRESSES_SCHEMA, VARIABLES_SCHEMA]


def test_find_of_a_project_reads_its_view_once(tmp_path: Path, parses: list[str]) -> None:
    _write(tmp_path / VIEW_RELPATH)

    for _ in range(3):
        found = SimulatorView.find(tmp_path)
        assert found is not None
        assert [model.name for model in found.physics_models()] == ["RING"]

    assert parses == [ADDRESSES_SCHEMA, VARIABLES_SCHEMA]


def test_a_changed_file_is_read_again(tmp_path: Path, parses: list[str]) -> None:
    _write(tmp_path)
    first = SimulatorView.open(tmp_path)
    assert first.code == "FAC"

    _write(tmp_path, code="OTHER")
    second = SimulatorView.open(tmp_path)

    assert second is not first
    assert second.code == "OTHER"
    assert parses.count(VARIABLES_SCHEMA) == 2


def test_a_touched_file_is_read_again(tmp_path: Path) -> None:
    _write(tmp_path)
    first = SimulatorView.open(tmp_path)

    later = time.time() - _SETTLED_S / 2
    os.utime(tmp_path / VARIABLES_FILE, (later, later))

    assert SimulatorView.open(tmp_path) is not first


def test_a_file_that_appears_is_read(tmp_path: Path) -> None:
    _write(tmp_path)
    variables = (tmp_path / VARIABLES_FILE).read_text(encoding="utf-8")
    (tmp_path / VARIABLES_FILE).unlink()
    first = SimulatorView.open(tmp_path)

    (tmp_path / VARIABLES_FILE).write_text(variables, encoding="utf-8")
    _settle(tmp_path / VARIABLES_FILE)
    second = SimulatorView.open(tmp_path)

    assert second is not first
    assert second.code == "FAC"


def test_a_file_just_written_is_not_held(tmp_path: Path) -> None:
    _write(tmp_path, settled=False)

    assert SimulatorView.open(tmp_path) is not SimulatorView.open(tmp_path)


def test_the_constructor_reads_afresh(tmp_path: Path) -> None:
    _write(tmp_path)

    assert SimulatorView(tmp_path) is not SimulatorView.open(tmp_path)


def test_threads_asking_first_parse_once(tmp_path: Path, parses: list[str]) -> None:
    _write(tmp_path)
    start = threading.Barrier(8)
    views: list[SimulatorView] = []
    models: list[tuple[Any, ...]] = []

    def ask() -> None:
        start.wait()
        view = SimulatorView.open(tmp_path)
        views.append(view)
        models.append(view.models())

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(views) == 8
    assert all(view is views[0] for view in views)
    assert all(read is models[0] for read in models)
    assert parses == [ADDRESSES_SCHEMA, VARIABLES_SCHEMA]


def test_a_wiring_record_shares_nothing_with_the_file_as_parsed(tmp_path: Path) -> None:
    view = SimulatorView.open(_write(tmp_path))

    binding = view.bindings()[0]
    binding.record["engine"]["fields"][2]["deep"].append(4)  # type: ignore[index]
    view.model("RING").settings["harmonics"].append(3)  # type: ignore[attr-defined]

    assert view.document(VARIABLES_FILE)["models"][0]["wiring"][0]["engine"]["fields"] == (
        1,
        2,
        {"deep": (3,)},
    )
    assert view.document(VARIABLES_FILE)["models"][0]["settings"]["harmonics"] == (1, 2)
    assert isinstance(binding.record["engine"]["fields"], list)  # type: ignore[index]
    with pytest.raises(TypeError):
        binding.record["engine"] = {}  # type: ignore[index]


async def test_two_directories_build_the_rows_of_one_view_once(
    tmp_path: Path, parses: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from osprey_connectors.directory import _simulator

    _write(tmp_path / VIEW_RELPATH)
    built: list[SimulatorView] = []
    real = _simulator._read_records

    def counting(view: SimulatorView) -> Any:
        built.append(view)
        return real(view)

    monkeypatch.setattr(_simulator, "_read_records", counting)

    answers = []
    for _ in range(2):
        directory = SimulatorDirectory(
            BackendSpec(
                kind="directory", type="simulator_view", target="va", settings={}, rehearsal=False
            ),
            project_dir=tmp_path,
        )
        answers.append(await directory.probe(1.0))

    assert answers == ["simulator view read: 2 PVs"] * 2
    assert len(built) == 1
    assert parses == [ADDRESSES_SCHEMA, VARIABLES_SCHEMA]
