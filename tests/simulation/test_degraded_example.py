"""The built example's SR in the degraded state: built, serving, without a closed orbit.

With ``first-turn-errors`` active the example's SR has no closed orbit. It is
built all the same and serves on: its status reads ``no closed orbit``, a
corrector write commits, and only what the closed orbit sets -- the orbit at the
monitors, the tunes, the beta functions and the chromaticity -- reads NaN under
the ``udf`` condition. The turn-by-turn records and every setpoint carry no
condition, because a shot is tracked whether the ring closes or not.

No write takes a scenario's misalignments away, so recovery is pinned on the
clean example: a corrector write that removes the closed orbit degrades SR, and
writing the corrector back brings ``ok`` and clears every condition.

The archive is seeded from the same degraded model: the seeding builds, the
orbit history reads NaN, and neither the seeded documents nor the recorder's
address list hold a turn series. Adding the turn series to the view changes no
document count and no other channel's content.
"""

from __future__ import annotations

import asyncio
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("at")

from osprey.connectors.control_system.base import WriteOutcome
from osprey.connectors.control_system.va_in_process_connector import (
    VAInProcessConnector,
    simulation_state_dir,
)
from osprey.services.archiver_recorder.config import resolve_channel_addresses
from osprey.simulation.engines.pyat import NO_CLOSED_ORBIT, TURN_SERIES
from osprey_connectors.archiver.field_names import field_name
from osprey_connectors.simulation import composite as composite_module
from osprey_connectors.simulation import model_status
from osprey_connectors.simulation.archive import (
    SeedKnobs,
    archived_channels,
    build,
    seed_base,
    seed_grid,
    synthesize_documents,
)
from osprey_connectors.simulation.composite import STATUS_OK, Composite
from osprey_connectors.simulation.status import model_built, read_model_status
from osprey_connectors.simulation.view import SimulatorView
from tests.simulation.conftest import StubCollection

if TYPE_CHECKING:
    from tests._builds import BuiltProject

pytestmark = pytest.mark.slow

SCENARIO = "first-turn-errors"
MODEL = "SR"
T0 = 1_760_000_000.0
SEED_T0 = datetime(2026, 3, 14, 9, 26, 53, tzinfo=UTC)

#: Two hours of history at a coarse cadence: enough grid to compare, quick to seed.
SMALL = SeedKnobs(retention_days=1, hot_span_hours=1, hot_cadence_sec=10, tail_cadence_sec=60)
#: The grid instants the document comparisons synthesize.
SAMPLES = 6

SR_STATUS = "ca:SIM:SR:STATUS"
CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"
#: Inside the corrector's limits band, where the clean ring keeps its closed orbit.
SMALL_KICK = 1.0
#: Far outside the limits band: the clean ring has no closed orbit there.
LOSING_KICK = 100.0

ORBIT = ("SR:DIAG:BPM:03:POSITION:X", "SR:DIAG:BPM:03:POSITION:Y")
TUNES = ("SR:DIAG:TUNE:X", "SR:DIAG:TUNE:Y")
CHROMATICITY = ("SR:DIAG:CHROM:X", "SR:DIAG:CHROM:Y")
#: What the closed orbit sets and the example serves as channels.
VOIDED = (*ORBIT, *TUNES, *CHROMATICITY)
#: The model's own optics arrays: the beta functions have no channel in the example.
VOIDED_MODEL = ("SR/orbit_at_monitors", "SR/beta_at_monitors", "SR/tunes", "SR/chromaticity")
TURN_RECORDS = ("SR:DIAG:BPM:03:TBT:X", "SR:DIAG:BPM:03:TBT:Y", "SR:DIAG:BPM:03:TBT:SUM")

#: The connector's alarm severity of a channel under the ``udf`` condition.
UDF_SEVERITY = 3
#: The longest wait for one tick of the in-process connector.
TICK_WAIT_S = 10.0


def _writes_enabled(key: str, default: Any = None) -> Any:
    if key == "control_system.writes_enabled":
        return True
    return default


@pytest.fixture(autouse=True)
def _no_model_log_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)
    monkeypatch.setattr("osprey.utils.config.get_config_value", _writes_enabled)


@pytest.fixture
def example_view(built_control_assistant: BuiltProject, tmp_path: Path) -> Path:
    prefix = "data/simulator/"
    view = tmp_path / "example" / "data" / "simulator"
    for name, data in built_control_assistant.outputs[0].files.items():
        if name.startswith(prefix):
            target = view / name[len(prefix) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return view


def _activate(view: Path, *names: str) -> None:
    state = simulation_state_dir(view)
    state.mkdir(parents=True, exist_ok=True)
    (state / "active_scenarios").write_text("\n".join(names) + "\n", encoding="utf-8")


def _composite(view: Path) -> Composite:
    return Composite(view, state_dir=simulation_state_dir(view), clock=lambda: T0, model_log=False)


async def _connected(view: Path) -> VAInProcessConnector:
    connector = VAInProcessConnector()
    await connector.connect({"simulator_view": str(view), "response_delay_ms": 0})
    return connector


def _channels(view: Path) -> list[dict[str, Any]]:
    channels: list[dict[str, Any]] = json.loads(
        (view / "variables.json").read_text(encoding="utf-8")
    )["channels"]
    return channels


def _sr_float_readbacks(view: Path) -> list[str]:
    return [
        channel["address"]
        for channel in _channels(view)
        if channel["owner"] == MODEL
        and channel["role"] != "setpoint"
        and channel["value_type"] == "float"
    ]


def _sr_setpoints(view: Path) -> list[str]:
    return [
        channel["address"]
        for channel in _channels(view)
        if channel["owner"] == MODEL and channel["role"] == "setpoint"
    ]


def _turn_series(view: Path) -> set[str]:
    return {binding.address for binding in SimulatorView.open(view).bindings(role=TURN_SERIES)}


def _all_nan(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return all(_all_nan(item) for item in value)
    if hasattr(value, "tolist"):
        return _all_nan(value.tolist())
    return isinstance(value, float) and math.isnan(value)


def _comparable(value: Any) -> Any:
    """``value`` with every NaN a marker, so two NaN readings compare equal."""
    if isinstance(value, dict):
        return {key: _comparable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_comparable(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return value


def _without_turn_series(view: Path, root: Path) -> Path:
    """A copy of ``view`` whose SR states no turn series: the view before they were added."""
    copy = root / "without" / "data" / "simulator"
    shutil.copytree(view, copy)
    series = _turn_series(view)
    variables = copy / "variables.json"
    document = json.loads(variables.read_text(encoding="utf-8"))
    (sr,) = [model for model in document["models"] if model["name"] == MODEL]
    sr["wiring"] = [entry for entry in sr["wiring"] if entry["address"] not in series]
    document["channels"] = [
        channel for channel in document["channels"] if channel["address"] not in series
    ]
    variables.write_text(json.dumps(document), encoding="utf-8")
    addresses = copy / "addresses.json"
    listed = json.loads(addresses.read_text(encoding="utf-8"))
    listed["channels"] = [address for address in listed["channels"] if address not in series]
    addresses.write_text(json.dumps(listed), encoding="utf-8")
    return copy


# -- the scenario active ---------------------------------------------------------


async def test_first_turn_errors_serves_sr_degraded_and_a_corrector_write_commits(
    example_view: Path,
) -> None:
    _activate(example_view, SCENARIO)
    connector = await _connected(example_view)
    composite = connector._composite
    assert composite is not None

    result = await connector.write_channel(CORRECTOR, SMALL_KICK)

    assert result.outcome is WriteOutcome.CONFIRMED, result.error_message
    assert composite.held([CORRECTOR]) == {CORRECTOR: SMALL_KICK}
    assert model_status(connector, MODEL) == NO_CLOSED_ORBIT
    assert model_built(connector, MODEL)
    assert await read_model_status(connector, SR_STATUS) == NO_CLOSED_ORBIT
    for address in VOIDED:
        reading = await connector.read_channel(address)
        assert math.isnan(reading.value), f"{address} reads {reading.value} under {SCENARIO}"
        assert reading.metadata.alarm_severity == UDF_SEVERITY, address
    assert composite.output_severity(list(VOIDED)) == {
        address: {"condition": "udf"} for address in VOIDED
    }
    optics = composite.model_get(list(VOIDED_MODEL))
    assert [name for name, value in optics.items() if not _all_nan(value)] == []
    assert composite.output_severity(list(TURN_RECORDS)) == {}
    assert composite.output_severity(_sr_setpoints(example_view)) == {}
    for address in TURN_RECORDS:
        reading = await connector.read_channel(address)
        assert reading.metadata.alarm_severity != UDF_SEVERITY, address
    await connector.disconnect()


async def test_the_degraded_chromaticity_holds_its_condition_after_a_tick(
    example_view: Path,
) -> None:
    _activate(example_view, SCENARIO)
    connector = await _connected(example_view)
    ticks: list[None] = []
    sync = connector._sync

    def counting_sync() -> None:
        ticks.append(None)
        sync()

    connector._sync = counting_sync  # type: ignore[method-assign]
    loop = asyncio.get_running_loop()
    deadline = loop.time() + TICK_WAIT_S
    while not ticks and loop.time() < deadline:
        await asyncio.sleep(0.05)

    assert ticks, f"the connector did not tick within {TICK_WAIT_S} s"
    for address in CHROMATICITY:
        reading = await connector.read_channel(address)
        assert math.isnan(reading.value), f"{address} reads {reading.value} after a tick"
        assert reading.metadata.alarm_severity == UDF_SEVERITY, address
    await connector.disconnect()


# -- recovery on the clean example -----------------------------------------------


def test_a_corrector_write_removes_the_closed_orbit_and_writing_it_back_recovers(
    example_view: Path,
) -> None:
    composite = _composite(example_view)
    readbacks = _sr_float_readbacks(example_view)
    start = composite.get(CORRECTOR)
    assert composite.status(MODEL) == STATUS_OK

    composite.set({CORRECTOR: LOSING_KICK})

    assert composite.built(MODEL)
    assert composite.status(MODEL) == NO_CLOSED_ORBIT
    assert composite.get(SR_STATUS) == NO_CLOSED_ORBIT
    listed = composite.output_severity(readbacks)
    assert set(VOIDED) <= set(listed)
    assert all(severity == {"condition": "udf"} for severity in listed.values())

    composite.set({CORRECTOR: start})

    assert composite.status(MODEL) == STATUS_OK
    assert composite.get(SR_STATUS) == STATUS_OK
    assert composite.output_severity(readbacks) == {}
    read = composite.get(list(VOIDED))
    assert [address for address, value in read.items() if not math.isfinite(value)] == []


# -- the archive -------------------------------------------------------------------


def test_the_archive_seeds_with_the_scenario_active_and_the_orbit_history_nan(
    example_view: Path,
) -> None:
    series = _turn_series(example_view)
    times, _ = seed_grid(SMALL, SEED_T0)

    archive = build(example_view, [SCENARIO], anchor_s=SEED_T0.timestamp())
    documents = synthesize_documents(archive, times[:SAMPLES])

    assert {field_name(address) for address in TURN_RECORDS} <= {
        field_name(address) for address in series
    }
    assert documents
    for address in ORBIT:
        assert all(math.isnan(document[field_name(address)]) for document in documents), address
    stored = {field_name(address) for address in series}
    assert [key for document in documents for key in document if key in stored] == []


def test_the_seeded_store_and_the_recorder_hold_no_turn_series(example_view: Path) -> None:
    series = _turn_series(example_view)
    archived = archived_channels(SimulatorView.open(example_view))
    collection = StubCollection()

    report = seed_base(
        collection,
        build(example_view, [SCENARIO], anchor_s=SEED_T0.timestamp()),
        SMALL,
        t0=SEED_T0,
    )
    recorded = resolve_channel_addresses(example_view)

    assert series >= set(TURN_RECORDS)
    assert set(recorded).isdisjoint(series)
    assert set(archived).isdisjoint(series)
    assert sorted(recorded) == sorted(archived)
    assert report.channels == len(archived)
    assert report.documents == len(seed_grid(SMALL, SEED_T0)[0])
    stored = {field_name(address) for address in series}
    assert [key for document in collection.documents for key in document if key in stored] == []


def test_the_turn_series_change_no_document_count_and_no_other_channels_content(
    example_view: Path, tmp_path: Path
) -> None:
    without = _without_turn_series(example_view, tmp_path)
    times, _ = seed_grid(SMALL, SEED_T0)
    anchor = SEED_T0.timestamp()

    documents = synthesize_documents(build(example_view, [SCENARIO], anchor_s=anchor), times)
    before = synthesize_documents(build(without, [SCENARIO], anchor_s=anchor), times)

    assert len(documents) == len(before)
    assert set(archived_channels(SimulatorView.open(example_view))) == set(
        archived_channels(SimulatorView.open(without))
    )
    assert _comparable(documents) == _comparable(before)
