"""Command writes stay out of the in-process simulator's journal and its replay.

A command is a setpoint whose write acts and holds no setting: arming a
monitor, requesting a shot. The journal carries settings from one connector to
the next, so a command never enters it: a second connector, or one whose
composite was rebuilt, sees the settings another wrote and never fires a shot
of its own. What a shot left behind lives in the one connector that fired it,
and is held state there: a journal another connector moved is replayed over a
reset composite, so the turn series read zero and the armed flag reads 0 again.

Every write here goes through the connector as a deployment runs it. The
rendered config beside the view arms writes and names the limits file the build
wrote, the control-context record puts the deployment on its baseline machine,
and nothing in the write path is patched: the write gate, the limits validator
and the composite are the product's own. Each scenario shows the gate and the
validator refusing, so a pass is a pass through them.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

pytest.importorskip("at")

from osprey_connectors import posture_store
from osprey_connectors.control_system._base import WriteOutcome
from osprey_connectors.control_system._va_in_process_connector import (
    VAInProcessConnector,
    simulation_state_dir,
)
from osprey_connectors.errors import ChannelLimitsViolationError
from osprey_connectors.simulation.composite import shot_time, shot_times
from osprey_connectors.simulation.view import SimulatorView
from osprey_connectors.workspace import reset_config_cache
from osprey_connectors_tests._harness._control_context_fixtures import write_control_context
from tests.facility._pyaml_trees import measured_tree, with_correctors
from tests.facility._synthetic_trees import with_shot_wiring, with_trajectory, write_tree
from tests.facility.served_tree import in_process_config

MODEL = "SR"
ARM = "SR:BPM:ARM"
ARMED = "SR:BPM:ARMED"
REQUEST = "SR:INJ:REQ"
SERIES = ("SR:BPM:TBT:X", "SR:BPM:TBT:Y", "SR:BPM:TBT:SUM")
CORRECTOR = "SCOR:H:SP"
OTHER_CORRECTOR = "SCOR:V:SP"
COMMANDS = frozenset({ARM, REQUEST})

#: The corrector band the limits file states, and a value outside it.
CORRECTOR_BAND = 5.0
OUTSIDE_BAND = 6.0
#: The most requests a session may make.
MAX_REQUESTS = 1000

JOURNAL = ("inprocess", "writes.json")


def _tree() -> dict[str, Any]:
    """The periodic model measured by trajectory, every shot record wired, limits stated."""
    tree = with_trajectory(with_correctors(measured_tree()), MODEL, "SR/BPM1", marker="M1")
    with_shot_wiring(tree, MODEL, "BPM1")
    # A launch off axis, so a shot leaves a series that is not the zeros of no shot.
    # The launch is a model-wide fault; the always-active baseline scenario states it.
    tree["scenarios/nominal.yaml"] = {
        "description": "The baseline, launched off axis.",
        "faults": {MODEL: {"launch_x": 1.0e-3}},
    }
    tree["limits.yaml"] = {
        "records": [
            {"address": ARM, "min_value": 0, "max_value": 1, "confirm": False},
            {"address": REQUEST, "min_value": 0, "max_value": MAX_REQUESTS, "confirm": False},
            {"address": CORRECTOR, "min_value": -CORRECTOR_BAND, "max_value": CORRECTOR_BAND},
            {
                "address": OTHER_CORRECTOR,
                "min_value": -CORRECTOR_BAND,
                "max_value": CORRECTOR_BAND,
            },
        ]
    }
    return tree


def _render_config(view: Path, *, writes_enabled: bool) -> dict[str, Any]:
    """The deployment's config: the simulator in process, every unlisted channel refused."""
    return {
        "control_system": {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": in_process_config(view, response_delay_ms=0)},
            "writes_enabled": writes_enabled,
            "limits_checking": {
                "enabled": True,
                "database_path": "data/channel_limits.json",
                "mode": "exclusive",
                "on_violation": "error",
            },
        }
    }


def _render(root: Path) -> Path:
    """Build the tree under ``root`` and render it; returns the render directory."""
    from osprey.facility.build import build_facility
    from osprey.facility.render import render_facility_outputs

    facility = root / "data" / "facility"
    if not facility.exists():
        write_tree(facility, _tree())
    render = root / "build"
    render.mkdir(exist_ok=True)
    view = render / "data" / "simulator"
    config = _render_config(view, writes_enabled=True)
    document = build_facility(facility, project_name="served")
    render_facility_outputs(render, document, config, facility, omitted_reported=set())
    return render


def _deploy(render: Path, monkeypatch: pytest.MonkeyPatch, *, writes_enabled: bool = True) -> None:
    """Write the rendered config beside the view and make it the one this process reads."""
    view = render / "data" / "simulator"
    config_path = render / "config.yml"
    config_path.write_text(
        yaml.safe_dump(_render_config(view, writes_enabled=writes_enabled)), encoding="utf-8"
    )
    monkeypatch.setenv("CONFIG_FILE", str(config_path))
    reset_config_cache()


@pytest.fixture
def view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The rendered simulator view of a deployment whose writes are armed."""
    render = _render(tmp_path)
    agent_data = tmp_path / "_agent_data"
    agent_data.mkdir()
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(agent_data))
    write_control_context(agent_data, target="live", generation=1)
    _deploy(render, monkeypatch)
    yield render / "data" / "simulator"
    reset_config_cache()


Connect = Callable[[], Any]


@pytest.fixture
async def connect(view: Path) -> AsyncIterator[Connect]:
    """Connect one more connector to the view; every one is disconnected at the end."""
    connectors: list[VAInProcessConnector] = []

    async def one() -> VAInProcessConnector:
        connector = VAInProcessConnector()
        await connector.connect(in_process_config(view, response_delay_ms=0))
        connectors.append(connector)
        return connector

    yield one
    for connector in connectors:
        await connector.disconnect()


def _journal(view: Path) -> Path:
    return simulation_state_dir(view).joinpath(*JOURNAL)


def _journalled(view: Path) -> list[str]:
    """The addresses the journal holds, in order; none when there is no journal."""
    path = _journal(view)
    if not path.exists():
        return []
    return [address for _seq, address, _value in json.loads(path.read_text())["writes"]]


async def _write(connector: VAInProcessConnector, address: str, value: Any) -> None:
    """One write through the connector's own gate, validator and confirmation policy."""
    result = await connector.write_channel(address, value)
    expected = WriteOutcome.UNREQUESTED if address in COMMANDS else WriteOutcome.CONFIRMED
    assert result.outcome is expected, result.error_message


async def _shoot(connector: VAInProcessConnector, request: int = 1) -> None:
    await _write(connector, ARM, 1)
    await _write(connector, REQUEST, request)


async def _armed(connector: VAInProcessConnector) -> int:
    return int((await connector.read_channel(ARMED)).value)


async def _series(connector: VAInProcessConnector) -> dict[str, np.ndarray]:
    return {
        address: np.asarray((await connector.read_channel(address)).value) for address in SERIES
    }


def _assert_shot(series: dict[str, np.ndarray]) -> None:
    """A shot filled the buffers: beam counted, and the launch seen off axis."""
    assert np.any(series[SERIES[2]] == 1.0)
    assert np.any(np.nan_to_num(series[SERIES[0]]) != 0.0)


def _assert_pre_shot(series: dict[str, np.ndarray]) -> None:
    for address, values in series.items():
        assert not np.any(values), address


# -- the fixture is the product's own chain --------------------------------------


def test_the_view_marks_exactly_the_two_commands(view: Path) -> None:
    marked = {binding.address for binding in SimulatorView.open(view).bindings() if binding.command}

    assert marked == COMMANDS


async def test_the_limits_validator_stands_in_front_of_every_write(
    view: Path, connect: Connect
) -> None:
    connector = await connect()

    with pytest.raises(ChannelLimitsViolationError):
        await connector.write_channel(CORRECTOR, OUTSIDE_BAND)
    with pytest.raises(ChannelLimitsViolationError):
        await connector.write_channel(ARM, 2)
    with pytest.raises(ChannelLimitsViolationError):
        await connector.write_channel(REQUEST, MAX_REQUESTS + 1)

    assert _journalled(view) == []
    assert await _armed(connector) == 0
    _assert_pre_shot(await _series(connector))


async def test_the_write_gate_stands_in_front_of_a_command(
    tmp_path: Path, view: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _deploy(tmp_path / "build", monkeypatch, writes_enabled=False)
    connector = VAInProcessConnector()
    await connector.connect({})
    try:
        for address in (ARM, REQUEST, CORRECTOR):
            result = await connector.write_channel(address, 1)
            assert result.outcome is WriteOutcome.REFUSED, address
            assert "writes are disabled" in (result.error_message or "")
        assert await _armed(connector) == 0
        assert _journalled(view) == []
    finally:
        await connector.disconnect()


# -- a command write is never appended -----------------------------------------


async def test_a_shot_fills_the_buffers_and_journals_nothing(view: Path, connect: Connect) -> None:
    connector = await connect()

    await _write(connector, ARM, 1)
    assert await _armed(connector) == 1
    await _write(connector, REQUEST, 1)

    assert await _armed(connector) == 0
    _assert_shot(await _series(connector))
    assert _journalled(view) == []


async def test_a_corrector_write_then_a_shot_journals_the_corrector_alone(
    view: Path, connect: Connect
) -> None:
    connector = await connect()

    await _write(connector, CORRECTOR, 1.0)
    await _shoot(connector)

    _assert_shot(await _series(connector))
    assert _journalled(view) == [CORRECTOR]
    document = json.loads(_journal(view).read_text())
    assert document["seq"] == 1


async def test_a_shot_between_two_corrector_writes_leaves_both_in_the_journal(
    view: Path, connect: Connect
) -> None:
    connector = await connect()

    await _write(connector, CORRECTOR, 1.0)
    await _shoot(connector)
    await _write(connector, OTHER_CORRECTOR, 2.0)

    assert _journalled(view) == [CORRECTOR, OTHER_CORRECTOR]
    # The connector's own setpoint write replays nothing: its shot stands.
    _assert_shot(await _series(connector))


async def test_the_connectors_own_reads_keep_the_shot(connect: Connect) -> None:
    connector = await connect()
    await _write(connector, CORRECTOR, 1.0)
    await _shoot(connector)

    first = await _series(connector)
    second = await _series(connector)

    _assert_shot(first)
    for address in SERIES:
        np.testing.assert_array_equal(first[address], second[address])


# -- a second connector's write between a shot and its read ---------------------


@pytest.mark.parametrize("corrector_first", [False, True], ids=["shot-only", "corrector-then-shot"])
async def test_a_foreign_corrector_write_resets_the_shot_and_fires_none(
    view: Path, connect: Connect, corrector_first: bool
) -> None:
    first = await connect()
    second = await connect()
    if corrector_first:
        await _write(first, CORRECTOR, 1.0)
    await _shoot(first)

    await _write(second, OTHER_CORRECTOR, 2.0)

    assert await _armed(first) == 0
    _assert_pre_shot(await _series(first))
    # The settings carried across; the shot did not, in either direction.
    assert (await first.read_channel(OTHER_CORRECTOR)).value == pytest.approx(2.0)
    assert await _armed(second) == 0
    _assert_pre_shot(await _series(second))
    expected = [CORRECTOR, OTHER_CORRECTOR] if corrector_first else [OTHER_CORRECTOR]
    assert _journalled(view) == expected
    if corrector_first:
        assert (await first.read_channel(CORRECTOR)).value == pytest.approx(1.0)
        assert (await second.read_channel(CORRECTOR)).value == pytest.approx(1.0)


@pytest.mark.parametrize("corrector_first", [False, True], ids=["arm-only", "corrector-then-arm"])
async def test_a_foreign_corrector_write_disarms_and_the_request_fires_none(
    connect: Connect, corrector_first: bool
) -> None:
    first = await connect()
    second = await connect()
    if corrector_first:
        await _write(first, CORRECTOR, 1.0)
    await _write(first, ARM, 1)
    assert await _armed(first) == 1

    await _write(second, OTHER_CORRECTOR, 2.0)

    assert await _armed(first) == 0
    await _write(first, REQUEST, 1)
    assert await _armed(first) == 0
    _assert_pre_shot(await _series(first))


async def test_a_shot_after_the_reset_fills_the_buffers_again(view: Path, connect: Connect) -> None:
    first = await connect()
    second = await connect()
    await _shoot(first)
    await _write(second, OTHER_CORRECTOR, 2.0)
    _assert_pre_shot(await _series(first))

    await _shoot(first, request=1)

    assert await _armed(first) == 0
    _assert_shot(await _series(first))
    assert _journalled(view) == [OTHER_CORRECTOR]


# -- a second or a rebuilt connector never fires ----------------------------------


@pytest.mark.parametrize("corrector_first", [False, True], ids=["shot-only", "corrector-then-shot"])
async def test_a_connector_connecting_after_a_shot_fires_none(
    connect: Connect, corrector_first: bool
) -> None:
    first = await connect()
    if corrector_first:
        await _write(first, CORRECTOR, 1.0)
    await _shoot(first)

    late = await connect()

    assert await _armed(late) == 0
    _assert_pre_shot(await _series(late))
    if corrector_first:
        assert (await late.read_channel(CORRECTOR)).value == pytest.approx(1.0)
    # The late connector wrote nothing, so the first one's shot stands.
    _assert_shot(await _series(first))


async def test_a_connector_connecting_while_another_is_armed_is_not_armed(
    connect: Connect,
) -> None:
    first = await connect()
    await _write(first, CORRECTOR, 1.0)
    await _write(first, ARM, 1)

    late = await connect()
    await _write(late, REQUEST, 1)

    assert await _armed(late) == 0
    _assert_pre_shot(await _series(late))


@pytest.mark.parametrize("corrector_first", [False, True], ids=["shot-only", "corrector-then-shot"])
async def test_a_rebuilt_connector_reads_pre_shot_and_fires_none(
    view: Path, connect: Connect, corrector_first: bool
) -> None:
    from osprey_connectors.simulation.state import ACTIVE_SCENARIOS_FILENAME

    connector = await connect()
    if corrector_first:
        await _write(connector, CORRECTOR, 1.0)
    await _shoot(connector)
    _assert_shot(await _series(connector))

    # The same active set, written again: the composite rebuilds from the view.
    state_file = simulation_state_dir(view) / ACTIVE_SCENARIOS_FILENAME
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text("", encoding="utf-8")

    assert await _armed(connector) == 0
    _assert_pre_shot(await _series(connector))
    if corrector_first:
        assert (await connector.read_channel(CORRECTOR)).value == pytest.approx(1.0)
    assert _journalled(view) == ([CORRECTOR] if corrector_first else [])


async def test_a_reset_returns_the_shot_state_to_pre_shot(view: Path, connect: Connect) -> None:
    connector = await connect()
    await _shoot(connector)

    await connector.reset()

    assert await _armed(connector) == 0
    _assert_pre_shot(await _series(connector))
    assert _journalled(view) == []


# -- a journal naming a command is rejected ---------------------------------------


async def _plant(view: Path, connect: Connect, writes: list[list[Any]]) -> None:
    """Write a journal under the current active set, as another process would."""
    from osprey_connectors.control_system._va_in_process_connector import active_set_sha256

    witness = await connect()
    sha = active_set_sha256(witness._composite.active)
    await witness.disconnect()
    path = _journal(view)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"active_set_sha256": sha, "seq": len(writes), "writes": writes}),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("writes", "address"),
    [
        ([[1, ARM, 1], [2, REQUEST, 1]], ARM),
        ([[1, REQUEST, 1]], REQUEST),
        ([[1, CORRECTOR, 1.0], [2, ARM, 1], [3, REQUEST, 1]], ARM),
        ([[1, CORRECTOR, 1.0], [2, REQUEST, 0]], REQUEST),
    ],
    ids=["arm-and-request", "request", "corrector-arm-request", "request-of-zero"],
)
async def test_a_journal_naming_a_command_is_rejected_whole(
    view: Path,
    connect: Connect,
    caplog: pytest.LogCaptureFixture,
    writes: list[list[Any]],
    address: str,
) -> None:
    await _plant(view, connect, writes)

    with caplog.at_level(logging.WARNING):
        connector = await connect()
        armed = await _armed(connector)
        series = await _series(connector)

    assert f"writes journal rejected: {address}: a command is never replayed" in caplog.text
    assert caplog.text.count("writes journal rejected") == 1
    assert armed == 0
    _assert_pre_shot(series)
    # Nothing from it is applied, the setpoint beside the command included.
    assert (await connector.read_channel(CORRECTOR)).value == pytest.approx(0.0)
    assert json.loads(_journal(view).read_text())["writes"] == []


async def test_a_journal_naming_a_command_planted_under_a_live_connector_resets_its_shot(
    view: Path, connect: Connect, caplog: pytest.LogCaptureFixture
) -> None:
    connector = await connect()
    await _write(connector, CORRECTOR, 1.0)
    await _shoot(connector)
    seq = json.loads(_journal(view).read_text())["seq"]
    path = _journal(view)
    document = json.loads(path.read_text())
    document["seq"] = seq + 2
    document["writes"] += [[seq + 1, ARM, 1], [seq + 2, REQUEST, 2]]
    path.write_text(json.dumps(document), encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        armed = await _armed(connector)
        series = await _series(connector)

    assert f"writes journal rejected: {ARM}: a command is never replayed" in caplog.text
    assert armed == 0
    _assert_pre_shot(series)
    assert json.loads(path.read_text())["writes"] == []


# -- a turn series carries the time of its shot ------------------------------------


async def _stamps(connector: VAInProcessConnector) -> dict[str, datetime]:
    return {address: (await connector.read_channel(address)).timestamp for address in SERIES}


async def test_two_shots_read_with_strictly_increasing_series_timestamps(
    connect: Connect,
) -> None:
    connector = await connect()

    await _shoot(connector, 1)
    first = await _stamps(connector)
    await _shoot(connector, 2)
    second = await _stamps(connector)

    for address in SERIES:
        assert second[address] > first[address], address
        assert second[address].tzinfo is not None


async def test_a_series_read_with_no_shot_between_keeps_the_shot_timestamp(
    connect: Connect,
) -> None:
    connector = await connect()
    await _shoot(connector, 1)
    first = await _stamps(connector)

    await _write(connector, CORRECTOR, 1.0)
    again = await _stamps(connector)
    many = await connector.read_multiple_channels(list(SERIES))
    await _write(connector, REQUEST, 2)

    assert again == first
    assert {address: reading.timestamp for address, reading in many.items()} == first
    assert await _stamps(connector) == first


async def test_every_other_channel_keeps_the_read_timestamp(connect: Connect) -> None:
    connector = await connect()
    await _shoot(connector, 1)
    shot = await _stamps(connector)

    before = datetime.now().astimezone()
    readings = {address: await connector.read_channel(address) for address in (CORRECTOR, ARMED)}
    after = datetime.now().astimezone()

    for address, reading in readings.items():
        assert before <= reading.timestamp <= after, address
        assert reading.timestamp > shot[SERIES[0]], address


async def test_a_series_no_shot_has_filled_carries_the_read_timestamp(
    connect: Connect,
) -> None:
    connector = await connect()

    before = datetime.now().astimezone()
    stamps = await _stamps(connector)
    after = datetime.now().astimezone()

    for address, stamp in stamps.items():
        assert before <= stamp <= after, address


async def test_the_shot_timestamp_is_the_models_shot_time(connect: Connect) -> None:
    connector = await connect()
    await _shoot(connector, 1)
    composite = connector._composite
    assert composite is not None

    times = shot_times(composite)
    stamps = await _stamps(connector)

    assert set(SERIES) <= set(times)
    for address in SERIES:
        assert shot_time(composite, address) == times[address]
        assert stamps[address].timestamp() == pytest.approx(times[address], abs=1e-6)
    assert shot_time(composite, CORRECTOR) is None


async def test_a_reset_returns_the_series_to_the_read_timestamp_until_the_next_shot(
    connect: Connect,
) -> None:
    connector = await connect()
    await _shoot(connector, 1)
    shot = await _stamps(connector)

    await connector.reset()
    composite = connector._composite
    assert composite is not None

    assert not set(SERIES) & set(shot_times(composite))
    await _shoot(connector, 1)
    assert all(stamp > shot[address] for address, stamp in (await _stamps(connector)).items())


def test_shot_times_of_a_model_with_no_physics_children_names_none() -> None:
    assert shot_times(object()) == {}
    assert shot_time(object(), SERIES[0]) is None
