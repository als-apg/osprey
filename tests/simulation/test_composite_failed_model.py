"""A failed physics model in the composite and in the mock connector.

A model whose engine raises while it is built or read is failed: its float
outputs read NaN, its other outputs their last good value or their type's zero,
every one of its channels reports the ``udf`` condition, the texture serves on
unchanged, and :func:`~osprey_connectors.simulation.model_status` answers with
the engine's error text, capped at 1023 UTF-8 bytes. A write the model cannot
solve is refused and leaves the model serving.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from click.testing import CliRunner
from lume.model import LUMEModel
from lume.variables import ScalarVariable, StrVariable, Variable
from lume_pyat.exceptions import OrbitSolveError

from osprey.cli.sim import sim_group
from osprey.connectors.control_system.base import WriteOutcome
from osprey.connectors.control_system.va_in_process_connector import (
    VAInProcessConnector,
    simulation_state_dir,
)
from osprey_connectors.simulation import composite as composite_module
from osprey_connectors.simulation import model_status
from osprey_connectors.simulation.composite import ENGINE_GROUP, STATUS_MAX_BYTES, Composite
from osprey_connectors.simulation.view import SCHEMAS
from tests.cli._lifecycle_build import stub_build
from tests.fixtures.lifecycle_repo import build_exemplar_repo

if TYPE_CHECKING:
    from types import ModuleType

    from tests._builds import BuiltProject

T0 = 1_760_000_000.0

SR_STATUS = "ca:SIM:SR:STATUS"
BPM_X = "SR:DIAG:BPM:03:POSITION:X"
QUADRUPOLE = "SR:MAG:QF:01:CURRENT:SP"

#: Twice the demo's QF01 current: the one-turn map has no stable orbit there.
UNSTABLE_CURRENT = 712.2


def _writes_enabled(key: str, default: Any = None) -> Any:
    if key == "control_system.writes_enabled":
        return True
    return default


@pytest.fixture(autouse=True)
def _no_model_log_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)
    monkeypatch.setattr("osprey.utils.config.get_config_value", _writes_enabled)


async def _connected(view: Path) -> VAInProcessConnector:
    connector = VAInProcessConnector()
    await connector.connect({"simulator_view": str(view), "response_delay_ms": 0})
    return connector


def _activate(view: Path, *names: str) -> None:
    state = simulation_state_dir(view)
    state.mkdir(parents=True, exist_ok=True)
    (state / "active_scenarios").write_text("\n".join(names) + "\n", encoding="utf-8")


def _composite(view: Path) -> Composite:
    return Composite(view, state_dir=simulation_state_dir(view), clock=lambda: T0)


# -- the demo's SR -------------------------------------------------------------


@pytest.fixture
def demo_view(built_control_assistant: BuiltProject, tmp_path: Path) -> Path:
    prefix = "data/simulator/"
    view = tmp_path / "demo" / "data" / "simulator"
    for name, data in built_control_assistant.outputs[0].files.items():
        if name.startswith(prefix):
            target = view / name[len(prefix) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return view


def _channels(view: Path) -> list[dict[str, Any]]:
    channels: list[dict[str, Any]] = json.loads(
        (view / "variables.json").read_text(encoding="utf-8")
    )["channels"]
    return channels


def _sr_float_readbacks(view: Path) -> list[str]:
    return [
        channel["address"]
        for channel in _channels(view)
        if channel["owner"] == "SR"
        and channel["role"] != "setpoint"
        and channel["value_type"] == "float"
    ]


def _sr_setpoints(view: Path) -> list[str]:
    return [
        channel["address"]
        for channel in _channels(view)
        if channel["owner"] == "SR" and channel["role"] == "setpoint"
    ]


def _texture_channels(view: Path) -> list[str]:
    return [channel["address"] for channel in _channels(view) if channel["owner"] == "texture"]


def _with_scenario(view: Path, scenario: Mapping[str, Any]) -> None:
    path = view / "scenarios.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["scenarios"].append(dict(scenario))
    path.write_text(json.dumps(document), encoding="utf-8")


def _record_solve_errors(monkeypatch: pytest.MonkeyPatch) -> list[OrbitSolveError]:
    """Keep every ``OrbitSolveError`` the closed-orbit solve raises."""
    import lume_pyat.simulator

    real = lume_pyat.simulator.solve_orbit
    raised: list[OrbitSolveError] = []

    def recording(ring: Any) -> Any:
        try:
            return real(ring)
        except OrbitSolveError as exc:
            raised.append(exc)
            raise

    monkeypatch.setattr("lume_pyat.simulator.solve_orbit", recording)
    return raised


@pytest.mark.slow
async def test_a_solve_that_raises_fails_sr_and_leaves_the_texture(
    demo_view: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serving = _composite(demo_view)
    error = OrbitSolveError("closed orbit is not finite at 3 monitors")

    def failing(_ring: Any) -> Any:
        raise error

    monkeypatch.setattr("lume_pyat.simulator.solve_orbit", failing)
    failed = _composite(demo_view)
    readbacks = _sr_float_readbacks(demo_view)
    texture = _texture_channels(demo_view)
    connector = await _connected(demo_view)

    read = failed.get(readbacks)
    reading = await connector.read_channel(BPM_X)

    assert readbacks and texture
    assert [address for address in readbacks if not math.isnan(read[address])] == []
    assert failed.output_severity(readbacks) == dict.fromkeys(readbacks, {"condition": "udf"})
    assert failed.get(texture) == serving.get(texture)
    assert math.isnan(reading.value)
    assert reading.metadata.alarm_severity == 3
    assert model_status(connector, "SR") == str(error)
    assert failed.get(SR_STATUS) == str(error)
    await connector.disconnect()


@pytest.mark.slow
def test_an_unstable_active_quadrupole_fails_sr_with_the_solve_error(
    demo_view: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_scenario(
        demo_view, {"name": "unstable-quad", "overrides": {QUADRUPOLE: UNSTABLE_CURRENT}}
    )
    _activate(demo_view, "unstable-quad")
    raised = _record_solve_errors(monkeypatch)

    composite = _composite(demo_view)

    assert raised
    assert composite.status("SR") == str(raised[-1])
    assert composite.output_severity([BPM_X, QUADRUPOLE]) == {
        BPM_X: {"condition": "udf"},
        QUADRUPOLE: {"condition": "udf"},
    }
    assert math.isnan(composite.get(BPM_X))
    assert composite.get(QUADRUPOLE) == UNSTABLE_CURRENT


@pytest.mark.slow
async def test_a_live_quadrupole_write_that_destabilises_the_orbit_is_refused(
    demo_view: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raised = _record_solve_errors(monkeypatch)
    connector = await _connected(demo_view)
    composite = connector._composite
    assert composite is not None
    setpoints = _sr_setpoints(demo_view)
    before = composite.held(setpoints)
    journal = simulation_state_dir(demo_view) / "inprocess" / "writes.json"

    result = await connector.write_channel(QUADRUPOLE, UNSTABLE_CURRENT)

    assert result.outcome is WriteOutcome.REFUSED
    assert raised
    assert result.error_message == str(raised[-1])
    assert composite.held(setpoints) == before
    assert model_status(connector, "SR") == "ok"
    assert not journal.exists()
    await connector.disconnect()


# -- the demo's LINE -----------------------------------------------------------

LINE_SOURCE = Path(__file__).resolve().parents[2] / "scripts" / "facility_demo" / "_line.py"

#: A single-pass loss: the element index, its ``FamName`` and the turn.
LOST = re.compile(r"^lost at element \d+ \(\w+\) turn 0: ")


def _line_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_line", LINE_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LINE = _line_module()


def _line_corrector() -> str:
    """The first horizontal corrector's current setpoint on the line."""
    return next(
        row["id"]
        for row in LINE.channels()
        if row.get("role") == "setpoint" and ":HCM:" in row["id"]
    )


@pytest.mark.slow
async def test_a_live_line_corrector_write_that_loses_the_beam_is_refused(
    demo_view: Path,
) -> None:
    corrector = _line_corrector()
    connector = await _connected(demo_view)
    composite = connector._composite
    assert composite is not None
    before = composite.held([corrector])

    result = await connector.write_channel(corrector, LINE.LOSING_KICK)

    assert result.outcome is WriteOutcome.REFUSED
    assert result.error_message is not None
    assert LOST.match(result.error_message), result.error_message
    assert composite.held([corrector]) == before
    assert model_status(connector, "LINE") == "ok"
    await connector.disconnect()


@pytest.fixture
def _contain_env_written_by_the_cli() -> Any:
    """Keep what ``sim apply`` loads into the environment inside the test that ran it."""
    before = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(before)


def _stage(built: BuiltProject, tmp_path: Path) -> tuple[Path, Path]:
    """A deployment repo whose render carries the demo's simulator view.

    Returns:
        The repo and its simulator view.
    """
    repo = build_exemplar_repo(tmp_path / "repo")
    config = {"control_system": {"connector": {"virtual_accelerator": {"serving": "in_process"}}}}
    build = stub_build(repo, config=yaml.safe_dump(config))
    prefix = "data/simulator/"
    for name, data in built.outputs[0].files.items():
        if name.startswith(prefix):
            target = build / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return repo, build / "data" / "simulator"


@pytest.mark.slow
@pytest.mark.usefixtures("_contain_env_written_by_the_cli")
def test_an_applied_line_fault_that_loses_the_beam_fails_line(
    built_control_assistant: BuiltProject, tmp_path: Path
) -> None:
    repo, view = _stage(built_control_assistant, tmp_path)
    _with_scenario(
        view,
        {
            "name": "line-loss",
            "faults": {"LINE": {"writes": {_line_corrector(): LINE.LOSING_KICK}}},
        },
    )
    assert _composite(view).status("LINE") == "ok"

    result = CliRunner().invoke(
        sim_group, ["apply", "--repo", str(repo), "line-loss", "--no-seed", "--yes"]
    )
    failed = _composite(view)

    assert result.exit_code == 0, result.output
    assert LOST.match(failed.status("LINE")), failed.status("LINE")
    assert failed.status("SR") == "ok"


# -- a stub engine ---------------------------------------------------------------

STUB_STATUS = "T:SIM:M:STATUS"
FLAG = "M:FLAG"
MODE = "M:MODE"
MODE_OPTIONS = ["IDLE", "RUN", "FAULT"]


class StubModel(LUMEModel):
    """A float readback, a bool and an enum output; ``knob`` -1 fails every read."""

    def __init__(self) -> None:
        self._variables: dict[str, Variable] = {
            "M:RB": ScalarVariable(name="M:RB", read_only=True),
            FLAG: StrVariable(name=FLAG, read_only=True),
            MODE: StrVariable(name=MODE, read_only=True),
            "knob": ScalarVariable(name="knob", read_only=False),
        }
        self.reset()

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    def reset(self) -> None:
        self.knob = 1.0

    def _set(self, values: dict[str, Any]) -> None:
        self.knob = float(values.get("knob", self.knob))

    def _get(self, names: list[str]) -> dict[str, Any]:
        if self.knob == -1.0:
            raise RuntimeError("the stub lost its state")
        served = {"M:RB": 4.0, FLAG: "TRUE", MODE: "RUN", "knob": self.knob}
        return {name: served[name] for name in names}


def _stub_build(model: str, wiring: Any, deck: Any, settings: Any, active: Any = None) -> Any:
    del model, wiring, deck, active
    if settings and settings.get("fail"):
        raise RuntimeError(settings["fail"])
    return StubModel()


STUB = SimpleNamespace(build=_stub_build, error_text=lambda exc: " ".join(str(exc).split()))


@pytest.fixture
def stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register :data:`STUB` as engine ``stub`` in the entry-point lookup."""
    real = importlib.metadata.entry_points

    def entry_points(**selection: Any) -> Any:
        if selection == {"group": ENGINE_GROUP, "name": "stub"}:
            return [SimpleNamespace(load=lambda: STUB)]
        return real(**selection)

    monkeypatch.setattr(composite_module, "metadata", SimpleNamespace(entry_points=entry_points))


def _channel(address: str, owner: str, **fields: Any) -> dict[str, Any]:
    role = fields.pop("role", "readback")
    return {
        "address": address,
        "role": role,
        "pair": address if role == "setpoint" else None,
        "value_type": fields.pop("value_type", "float"),
        "unit": None,
        "description": None,
        "writable": role == "setpoint",
        "value_range": None,
        "owner": owner,
        "on": None,
        **fields,
    }


def _stub_view(root: Path, settings: Mapping[str, Any] | None = None) -> Path:
    channels = [
        _channel("M:RB", "M"),
        _channel(FLAG, "M", value_type="bool"),
        _channel(MODE, "M", value_type="enum", options=MODE_OPTIONS),
        _channel("T:SP", "texture", role="setpoint"),
    ]
    wiring = [
        {
            "id": str(index),
            "address": channel["address"],
            "direction": "read",
            "role": "readback",
            "plane": None,
            "refresh": "pass",
        }
        for index, channel in enumerate(channels[:3])
    ]
    documents = {
        "served_models.json": {"models": ["M", "texture"]},
        "addresses.json": {
            "channels": sorted(channel["address"] for channel in channels),
            "status": [STUB_STATUS],
        },
        "variables.json": {
            "code": "T",
            "models": [
                {
                    "name": "M",
                    "engine": "stub",
                    "served": True,
                    "settings": dict(settings or {}),
                    "deck": None,
                    "wiring": wiring,
                },
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
    view = root / "stub" / "data" / "simulator"
    view.mkdir(parents=True)
    for name, document in documents.items():
        (view / name).write_text(
            json.dumps({"schema": SCHEMAS[name], **document}), encoding="utf-8"
        )
    return view


@pytest.mark.usefixtures("stub_engine")
def test_a_failed_stubs_bool_and_enum_hold_their_last_good_values(tmp_path: Path) -> None:
    composite = _composite(_stub_view(tmp_path))
    assert composite.get([FLAG, MODE]) == {FLAG: "TRUE", MODE: "RUN"}

    composite.model_set({"M/knob": -1.0})

    read = composite.get([FLAG, MODE, "M:RB"])

    assert (read[FLAG], read[MODE]) == ("TRUE", "RUN")
    assert math.isnan(read["M:RB"])
    assert composite.status("M") == "the stub lost its state"
    assert composite.output_severity([FLAG, MODE, "T:SP"]) == {
        FLAG: {"condition": "udf"},
        MODE: {"condition": "udf"},
    }


@pytest.mark.usefixtures("stub_engine")
def test_a_stub_that_never_built_reads_its_bool_and_enum_as_zero(tmp_path: Path) -> None:
    composite = _composite(_stub_view(tmp_path, {"fail": "the stub has no deck"}))

    assert composite.get([FLAG, MODE]) == {FLAG: "FALSE", MODE: "IDLE"}
    assert composite.output_severity([FLAG, MODE]) == {
        FLAG: {"condition": "udf"},
        MODE: {"condition": "udf"},
    }
    assert composite.get("T:SP") == 5.0


@pytest.mark.usefixtures("stub_engine")
async def test_a_2kb_non_ascii_engine_text_is_capped_alike_in_the_composite_and_in_process(
    tmp_path: Path,
) -> None:
    message = "Ä" * 1024
    view = _stub_view(tmp_path, {"fail": message})
    composite = _composite(view)
    connector = await _connected(view)

    status = composite.status("M")

    assert len(message.encode("utf-8")) == 2048
    assert len(status.encode("utf-8")) == STATUS_MAX_BYTES == 1023
    assert status.endswith("…")
    assert message.startswith(status[:-1])
    assert model_status(connector, "M") == status
    assert (await connector.read_channel(STUB_STATUS)).value == status
    await connector.disconnect()


@pytest.mark.usefixtures("stub_engine")
async def test_an_unknown_model_is_refused_naming_the_served_ones(tmp_path: Path) -> None:
    connector = await _connected(_stub_view(tmp_path))

    with pytest.raises(ValueError, match=r"'NOPE' is not served; served: \['M'\]"):
        model_status(connector, "NOPE")
    await connector.disconnect()


def test_a_connector_without_an_in_process_simulator_is_refused() -> None:
    with pytest.raises(ValueError, match="VAInProcessConnector serves no simulator in process"):
        model_status(VAInProcessConnector(), "SR")
