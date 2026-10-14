"""The archived channel set of a simulator view, and what the archive does with it.

A view's turn series are records of a shot: they have no value at an instant
between shots, so they have no archived history. Every other channel has, a
command channel among them. The archive composite holds exactly that set, is
built whenever every model is built (under a condition or not), and the mock
archiver tells a channel the archive leaves out from an address the view does
not hold.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from lume.model import LUMEModel
from lume.variables import ScalarVariable, Variable

from osprey_connectors.archiver.mock_archiver_connector import MockArchiverConnector
from osprey_connectors.control_system.va_in_process_connector import not_in_facility
from osprey_connectors.simulation import composite as composite_module
from osprey_connectors.simulation.archive import archived_channels, build
from osprey_connectors.simulation.composite import Composite
from osprey_connectors.simulation.view import SimulatorView
from tests._simulator_view import write_texture_view
from tests.facility.served_tree import in_process_config

T0 = 1_760_000_000.0
TIMES = np.arange(T0, T0 + 60.0, 10.0)
NO_BEAM = "no closed orbit"

SETPOINT = "M:SP"
READBACK = "M:RB"
SERIES_X = "M:TBT:X"
ARMED = "M:ARMED"
ARM = "M:ARM"
TEXTURE = "T:TEMP"

#: Every channel of the view: one physics model's and one texture channel.
CHANNELS: dict[str, dict[str, Any]] = {
    SETPOINT: {"owner": "M", "role": "setpoint"},
    READBACK: {"owner": "M"},
    SERIES_X: {"owner": "M"},
    ARMED: {"owner": "M"},
    ARM: {"owner": "M", "role": "setpoint"},
    TEXTURE: {"nominal": 25.0, "noise": {"absolute": 0.05}},
}
SERIES = {SERIES_X, ARMED}
ARCHIVED = sorted(set(CHANNELS) - SERIES)


def _described(role: str, plane: str | None = None) -> dict[str, Any]:
    return {"role": role, "plane": plane, "refresh": "pass"}


def _model(served: bool = True, **settings: Any) -> dict[str, Any]:
    """Model ``M`` as a built view records it: a setpoint, a readback, two series, a command."""
    return {
        "name": "M",
        "engine": "stub",
        "served": served,
        "settings": settings,
        "deck": None,
        "wiring": [
            {"id": "1", "address": SETPOINT, "direction": "write", "default": 2.0}
            | _described("setpoint"),
            {"id": "2", "address": READBACK, "direction": "read", "default": 4.0}
            | _described("readback"),
            {"id": "3", "address": SERIES_X, "direction": "read", "default": 0.0}
            | _described("turn_series", "x"),
            {"id": "4", "address": ARMED, "direction": "read", "default": 0.0}
            | _described("turn_series"),
            {"id": "5", "address": ARM, "direction": "write", "default": 0.0, "command": True}
            | _described("setpoint"),
        ],
    }


def _view(render: Path, **settings: Any) -> SimulatorView:
    return SimulatorView.open(write_texture_view(render, CHANNELS, models=[_model(**settings)]))


# -- a stub engine: builds, fails to build, or builds under a condition --------


class StubModel(LUMEModel):
    """Writes held as inputs, readings at their wiring default; NaN where the beam is lost."""

    def __init__(self, wiring: list[Mapping[str, Any]], settings: Mapping[str, Any]) -> None:
        self.lost = bool(settings.get("lost"))
        self.listed: frozenset[str] = frozenset()
        self._variables: dict[str, Variable] = {}
        self._defaults: dict[str, float] = {}
        readings = set()
        for entry in wiring:
            address = str(entry["address"])
            write = entry.get("direction") == "write"
            self._variables[address] = ScalarVariable(name=address, read_only=not write)
            self._defaults[address] = float(entry.get("default", 0.0))
            if not write and entry.get("role") == "readback":
                readings.add(address)
        self.listed = frozenset(readings)
        self.inputs: dict[str, float] = {}
        self.reset()

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    def reset(self) -> None:
        self.inputs = dict(self._defaults)

    def _set(self, values: dict[str, Any]) -> None:
        self.inputs.update({name: float(value) for name, value in values.items()})

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {
            name: math.nan if self.lost and name in self.listed else self.inputs[name]
            for name in names
        }


def _build(model, wiring, deck, settings, active=None):
    del model, deck, active
    if settings.get("fail"):
        raise RuntimeError(settings["fail"])
    return StubModel(list(wiring), settings)


def _condition(model: StubModel) -> tuple[str, frozenset[str]] | None:
    return (NO_BEAM, model.listed) if model.lost else None


@pytest.fixture(autouse=True)
def _stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = SimpleNamespace(build=_build, condition=_condition)
    real = Composite._engine
    monkeypatch.setattr(
        Composite, "_engine", staticmethod(lambda name: stub if name == "stub" else real(name))
    )
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)


# -- the set -------------------------------------------------------------------


def test_the_archived_set_is_the_view_minus_its_turn_series(tmp_path: Path) -> None:
    view = _view(tmp_path)

    assert list(archived_channels(view)) == ARCHIVED
    assert set(view.channels()) - set(archived_channels(view)) == SERIES


def test_a_command_channel_is_archived(tmp_path: Path) -> None:
    view = _view(tmp_path)

    assert view.binding(ARM).command
    assert ARM in archived_channels(view)


def test_a_view_without_series_records_archives_every_channel(tmp_path: Path) -> None:
    channels = {address: CHANNELS[address] for address in (SETPOINT, READBACK, TEXTURE)}
    model = _model()
    model["wiring"] = model["wiring"][:2]
    view = SimulatorView.open(write_texture_view(tmp_path, channels, models=[model]))

    assert archived_channels(view) == view.channels()


def test_an_unserved_models_series_stay_unless_every_model_is_asked(tmp_path: Path) -> None:
    """A model the render does not serve has its channels served as texture."""
    path = write_texture_view(tmp_path, CHANNELS, models=[_model(served=False)])
    served = path / "served_models.json"
    document = json.loads(served.read_text(encoding="utf-8"))
    served.write_text(json.dumps({**document, "models": ["texture"]}), encoding="utf-8")
    view = SimulatorView.open(path)

    assert archived_channels(view) == view.channels()
    assert list(archived_channels(view, served_only=False)) == ARCHIVED


# -- the archive composite holds the set ---------------------------------------


def test_the_archive_holds_the_archived_set(tmp_path: Path) -> None:
    view = _view(tmp_path)

    archive = build(view, [], anchor_s=T0)

    assert archive.addresses == list(archived_channels(view))
    assert archive.series(ARM, TIMES) == [0.0] * len(TIMES)
    with pytest.raises(KeyError, match=SERIES_X):
        archive.series(SERIES_X, TIMES)


def test_leaving_the_series_out_leaves_every_other_channel_as_it_was(tmp_path: Path) -> None:
    """The channels a view shares with one that has no series read the same in both."""
    kept = {address: spec for address, spec in CHANNELS.items() if address not in SERIES}
    plain = _model()
    plain["wiring"] = [entry for entry in plain["wiring"] if entry["address"] not in SERIES]
    without = build(write_texture_view(tmp_path / "plain", kept, models=[plain]), [], anchor_s=T0)

    with_series = build(_view(tmp_path / "series"), [], anchor_s=T0)

    assert with_series.addresses == without.addresses
    assert with_series.samples(ARCHIVED, TIMES) == without.samples(ARCHIVED, TIMES)


# -- the gate: built, not ok ---------------------------------------------------


def test_a_model_built_under_a_condition_is_archived_with_its_listed_readings_nan(
    tmp_path: Path,
) -> None:
    archive = build(_view(tmp_path, lost=True), [], anchor_s=T0)

    assert archive.addresses == ARCHIVED
    assert all(math.isnan(value) for value in archive.series(READBACK, TIMES))
    assert archive.series(SETPOINT, TIMES) == [2.0] * len(TIMES)
    assert not any(math.isnan(value) for value in archive.series(TEXTURE, TIMES))


def test_a_model_that_fails_to_build_stops_the_archive(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="^M: the deck does not load$"):
        build(_view(tmp_path, fail="the deck does not load"), [], anchor_s=T0)


# -- the mock archiver's two refusals ------------------------------------------


@pytest.fixture
async def archiver(tmp_path: Path):
    view = _view(tmp_path)
    connector = MockArchiverConnector()
    await connector.connect(in_process_config(view.path, sample_rate_hz=1.0))
    yield connector
    await connector.disconnect()


def _window() -> tuple[datetime, datetime]:
    start = datetime.fromtimestamp(T0, UTC)
    return start, start + timedelta(minutes=1)


@pytest.mark.asyncio
async def test_a_channel_the_archive_leaves_out_is_refused_as_having_no_history(
    archiver: MockArchiverConnector,
) -> None:
    with pytest.raises(ValueError) as refused:
        await archiver.get_data([TEXTURE, SERIES_X], *_window())

    assert str(refused.value) == f"{SERIES_X} has no archived history"
    with pytest.raises(ValueError, match=f"^{ARMED} has no archived history$"):
        await archiver.get_metadata(ARMED)
    assert await archiver.check_availability([SERIES_X, ARM]) == {SERIES_X: False, ARM: True}


@pytest.mark.asyncio
async def test_an_address_the_view_does_not_hold_is_refused_as_not_in_the_facility(
    archiver: MockArchiverConnector,
) -> None:
    with pytest.raises(ValueError) as refused:
        await archiver.get_data(["NOT:A:CHANNEL"], *_window())

    assert str(refused.value) == not_in_facility("NOT:A:CHANNEL")
    assert str(refused.value) == "NOT:A:CHANNEL is not in build/facility.json"
    with pytest.raises(ValueError, match="is not in build/facility.json"):
        await archiver.get_metadata("NOT:A:CHANNEL")


@pytest.mark.asyncio
async def test_an_archived_channel_is_served(archiver: MockArchiverConnector) -> None:
    frame = await archiver.get_data([TEXTURE, ARM], *_window())

    assert not frame.empty
    assert (await archiver.get_metadata(ARM)).is_archived
