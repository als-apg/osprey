"""A shot on the built example is a command write: it fires on an increment only.

The storage ring of the shipped example arms its turn-by-turn acquisition on a
monitor's arm command and fires one shot on each increase of the injection
request. A command acts and holds no setting, so nothing that only re-applies
settings fires one: an equal, lower or default request write, a reset, a
rebuild of the composite, and a second connector replaying the journal all
leave the shot count where it was. A shot fills the series only when the
acquisition was armed; an increment without arm leaves every buffer as it was.

The connector writes one channel per call, so the case of a command written
beside another owner's channel is pinned at the composite: the whole batch is
refused and no shot fires.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import yaml

pytest.importorskip("at")

from osprey_connectors.control_system._base import WriteOutcome
from osprey_connectors.control_system._va_in_process_connector import (
    VAInProcessConnector,
    simulation_state_dir,
)
from osprey_connectors.simulation import composite as composite_module
from osprey_connectors.simulation.composite import Composite, WriteRefusedError

if TYPE_CHECKING:
    from tests._builds import BuiltProject

pytestmark = pytest.mark.slow

T0 = 1_760_000_000.0

MODEL = "SR"
ARM = "SR:DIAG:BPM:01:TBT:ARM"
ARMED = "SR:DIAG:BPM:01:TBT:ARMED"
REQUEST = "SR:DIAG:INJECTION:REQUEST"
CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"
LINE_CORRECTOR = "LINE:MAG:HCM:01:CURRENT:SP"
#: The example's machine, where each sum record declares its calibration gain.
MODELS_FILE = (
    Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example/models.yaml"
)


def _writes_enabled(key: str, default: Any = None) -> Any:
    if key == "control_system.writes_enabled":
        return True
    return default


@pytest.fixture(autouse=True)
def _no_model_log_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)
    monkeypatch.setattr("osprey_connectors.config.get_config_value", _writes_enabled)


@pytest.fixture
def view(built_control_assistant: BuiltProject, tmp_path: Path) -> Path:
    """The built example's simulator view, staged fresh for each test."""
    prefix = "data/simulator/"
    view = tmp_path / "example" / "data" / "simulator"
    for name, data in built_control_assistant.outputs[0].files.items():
        if name.startswith(prefix):
            target = view / name[len(prefix) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return view


Connect = Callable[[], Any]


@pytest.fixture
async def connect(view: Path) -> AsyncIterator[Connect]:
    """Connect one more connector to the view; every one is disconnected at the end."""
    connectors: list[VAInProcessConnector] = []

    async def one() -> VAInProcessConnector:
        connector = VAInProcessConnector()
        await connector.connect({"simulator_view": str(view), "response_delay_ms": 0})
        connectors.append(connector)
        return connector

    yield one
    for connector in connectors:
        await connector.disconnect()


def _channels(view: Path) -> list[dict[str, Any]]:
    channels: list[dict[str, Any]] = json.loads(
        (view / "variables.json").read_text(encoding="utf-8")
    )["channels"]
    return channels


def _series_addresses(view: Path) -> list[str]:
    """Every turn-by-turn series of the ring: X, Y and SUM of each monitor."""
    return sorted(
        channel["address"]
        for channel in _channels(view)
        if channel["owner"] == MODEL
        and channel["address"].endswith((":TBT:X", ":TBT:Y", ":TBT:SUM"))
    )


def _sr(connector: VAInProcessConnector) -> Any:
    composite = connector._composite
    assert composite is not None
    return composite._children[MODEL].model


def _shots(connector: VAInProcessConnector) -> int:
    return int(_sr(connector).shot_count)


async def _write(connector: VAInProcessConnector, address: str, value: Any) -> None:
    """One write through the connector; no limits file here, so every write is confirmed."""
    result = await connector.write_channel(address, value)
    assert result.outcome is WriteOutcome.CONFIRMED, result.error_message


async def _shoot(connector: VAInProcessConnector, request: int) -> None:
    await _write(connector, ARM, 1)
    await _write(connector, REQUEST, request)


async def _armed(connector: VAInProcessConnector) -> int:
    return int((await connector.read_channel(ARMED)).value)


async def _buffers(connector: VAInProcessConnector, view: Path) -> dict[str, np.ndarray]:
    return {
        address: np.asarray((await connector.read_channel(address)).value, dtype=np.float64)
        for address in _series_addresses(view)
    }


def _declared_sum_gains() -> dict[str, float]:
    """Each sum record's declared calibration gain, ``1/k_b`` (hardware to physics), by address.

    The example declares each monitor's charge-to-sum factor ``k_b`` in the
    machine: a served sum is its physics sum divided by this gain.
    """
    models = yaml.safe_load(MODELS_FILE.read_text(encoding="utf-8"))
    return {
        record["address"]: float(record["calibration"]["curve"]["linear"]["gain"])
        for model in models
        for record in model.get("wiring") or ()
        if record["address"].endswith(":TBT:SUM")
    }


def _assert_shot(buffers: dict[str, np.ndarray]) -> None:
    """A shot filled the buffers: every monitor counted the beam on some turn."""
    sums = {address: values for address, values in buffers.items() if address.endswith(":SUM")}
    assert len(sums) == 72
    gains = _declared_sum_gains()
    for address, values in sums.items():
        assert np.any(values == 1.0 / gains[address]), address


def _assert_pre_shot(buffers: dict[str, np.ndarray]) -> None:
    assert buffers
    for address, values in buffers.items():
        assert not np.any(values), address


def _assert_unchanged(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> None:
    assert before.keys() == after.keys()
    for address in before:
        np.testing.assert_array_equal(after[address], before[address], err_msg=address)


# -- the fixture is the shipped example -----------------------------------------


def test_the_example_view_marks_the_arm_and_request_commands(view: Path) -> None:
    from osprey_connectors.simulation.view import SimulatorView

    marked = {binding.address for binding in SimulatorView.open(view).bindings() if binding.command}

    assert {ARM, REQUEST} <= marked
    assert len(_series_addresses(view)) == 3 * 72


async def test_an_armed_increment_fires_one_shot_and_fills_every_buffer(
    view: Path, connect: Connect
) -> None:
    connector = await connect()
    _assert_pre_shot(await _buffers(connector, view))

    await _shoot(connector, 1)

    assert _shots(connector) == 1
    assert await _armed(connector) == 0
    _assert_shot(await _buffers(connector, view))


# -- a write that is no increment fires none -------------------------------------


async def test_a_default_request_write_fires_none(view: Path, connect: Connect) -> None:
    connector = await connect()
    default = (await connector.read_channel(REQUEST)).value
    await _write(connector, ARM, 1)

    await _write(connector, REQUEST, default)

    assert _shots(connector) == 0
    assert await _armed(connector) == 1
    _assert_pre_shot(await _buffers(connector, view))


@pytest.mark.parametrize("request_value", [2, 1], ids=["equal", "lower"])
async def test_an_equal_or_lower_request_write_fires_none(
    view: Path, connect: Connect, request_value: int
) -> None:
    connector = await connect()
    await _shoot(connector, 2)
    shot = await _buffers(connector, view)
    await _write(connector, ARM, 1)

    await _write(connector, REQUEST, request_value)

    assert _shots(connector) == 1
    assert await _armed(connector) == 1
    _assert_unchanged(shot, await _buffers(connector, view))


async def test_a_lower_write_lowers_what_counts_as_an_increment(connect: Connect) -> None:
    connector = await connect()
    await _shoot(connector, 2)
    await _write(connector, REQUEST, 1)

    await _shoot(connector, 2)

    assert _shots(connector) == 2
    assert await _armed(connector) == 0


# -- an increment without arm leaves the buffers --------------------------------


async def test_an_increment_without_arm_leaves_every_buffer_at_rest(
    view: Path, connect: Connect
) -> None:
    connector = await connect()
    before = await _buffers(connector, view)

    await _write(connector, REQUEST, 1)

    assert _shots(connector) == 1
    assert await _armed(connector) == 0
    after = await _buffers(connector, view)
    _assert_pre_shot(after)
    _assert_unchanged(before, after)


async def test_an_increment_without_arm_leaves_the_last_shot_in_every_buffer(
    view: Path, connect: Connect
) -> None:
    connector = await connect()
    await _shoot(connector, 1)
    shot = await _buffers(connector, view)

    await _write(connector, REQUEST, 2)

    assert _shots(connector) == 2
    _assert_unchanged(shot, await _buffers(connector, view))


# -- nothing that only re-applies settings fires ----------------------------------


async def test_a_reset_returns_to_rest_and_fires_none(view: Path, connect: Connect) -> None:
    connector = await connect()
    await _write(connector, CORRECTOR, 1.0)
    await _shoot(connector, 1)

    await connector.reset()

    assert _shots(connector) == 0
    assert await _armed(connector) == 0
    _assert_pre_shot(await _buffers(connector, view))


@pytest.mark.parametrize("corrector_first", [False, True], ids=["shot-only", "corrector-then-shot"])
async def test_a_rebuild_reads_pre_shot_and_fires_none(
    view: Path, connect: Connect, corrector_first: bool
) -> None:
    from osprey_connectors.simulation.state import ACTIVE_SCENARIOS_FILENAME

    connector = await connect()
    if corrector_first:
        await _write(connector, CORRECTOR, 1.0)
    await _shoot(connector, 1)
    built = _sr(connector)

    # The same active set, written again: the composite rebuilds from the view.
    state_file = simulation_state_dir(view) / ACTIVE_SCENARIOS_FILENAME
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text("", encoding="utf-8")

    assert await _armed(connector) == 0
    assert _sr(connector) is not built
    assert _shots(connector) == 0
    _assert_pre_shot(await _buffers(connector, view))
    if corrector_first:
        assert (await connector.read_channel(CORRECTOR)).value == pytest.approx(1.0)


@pytest.mark.parametrize("corrector_first", [False, True], ids=["shot-only", "corrector-then-shot"])
async def test_a_second_connector_replaying_the_journal_fires_none(
    view: Path, connect: Connect, corrector_first: bool
) -> None:
    first = await connect()
    if corrector_first:
        await _write(first, CORRECTOR, 1.0)
    await _shoot(first, 1)

    late = await connect()

    assert _shots(late) == 0
    assert await _armed(late) == 0
    _assert_pre_shot(await _buffers(late, view))
    if corrector_first:
        assert (await late.read_channel(CORRECTOR)).value == pytest.approx(1.0)
    # The late connector wrote nothing, so the first one's shot stands.
    assert _shots(first) == 1
    _assert_shot(await _buffers(first, view))


# -- a foreign setpoint write between a shot and its read -------------------------


@pytest.mark.parametrize("corrector_first", [False, True], ids=["shot-only", "corrector-then-shot"])
async def test_a_foreign_setpoint_write_leaves_pre_shot_buffers(
    view: Path, connect: Connect, corrector_first: bool
) -> None:
    first = await connect()
    second = await connect()
    if corrector_first:
        await _write(first, CORRECTOR, 1.0)
    await _shoot(first, 1)

    await _write(second, LINE_CORRECTOR, 0.5)

    assert await _armed(first) == 0
    _assert_pre_shot(await _buffers(first, view))
    assert _shots(first) == 0
    assert _shots(second) == 0
    _assert_pre_shot(await _buffers(second, view))
    assert (await first.read_channel(LINE_CORRECTOR)).value == pytest.approx(0.5)
    if corrector_first:
        assert (await second.read_channel(CORRECTOR)).value == pytest.approx(1.0)


# -- a command beside another owner's channel is refused whole --------------------


def _composite(view: Path) -> Composite:
    return Composite(view, state_dir=simulation_state_dir(view), clock=lambda: T0)


@pytest.mark.parametrize(
    "batch",
    [
        {REQUEST: 1, LINE_CORRECTOR: 0.5},
        {ARM: 1, REQUEST: 1, LINE_CORRECTOR: 0.5},
        {REQUEST: 1, CORRECTOR: 1.0, LINE_CORRECTOR: 0.5},
    ],
    ids=["request-and-line", "arm-request-and-line", "request-ring-and-line"],
)
def test_a_command_beside_another_owners_channel_is_refused_whole(
    view: Path, batch: dict[str, Any]
) -> None:
    composite = _composite(view)
    series = _series_addresses(view)
    composite.set({ARM: 1})
    sr = composite._children[MODEL].model
    held = composite.held([CORRECTOR, LINE_CORRECTOR])

    with pytest.raises(WriteRefusedError, match="a command is written with channels of its own"):
        composite.set(batch)

    assert sr.shot_count == 0
    assert sr.armed
    assert composite.held([CORRECTOR, LINE_CORRECTOR]) == held
    _assert_pre_shot(
        {address: np.asarray(value) for address, value in composite.get(series).items()}
    )


def test_the_same_batch_without_the_other_owner_fires(view: Path) -> None:
    composite = _composite(view)
    series = _series_addresses(view)
    composite.set({ARM: 1})

    composite.set({REQUEST: 1, CORRECTOR: 1.0})

    assert composite._children[MODEL].model.shot_count == 1
    _assert_shot({address: np.asarray(value) for address, value in composite.get(series).items()})
