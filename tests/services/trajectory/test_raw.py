"""Raw buffers: captured at readout, kept beside their shot, pruned with it, served to the panel.

A chain keeps every included injection's samples — the trigger sample, then
the turns tracked — as one float32 buffer ``[n_included, 3, n_bpm,
raw_samples]`` with a JSON header ``{shape, trigger_sample, units}``; the store
keeps the raw retention's root shots with their buffers and derived shots, and
``GET /panel/api/shots/{shot_id}/raw`` serves the file, 410 ``raw_evicted``
once its shot was pruned. Chain turns run on the acquisition tests' stand-in
machine; the route runs on the door tests' app.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.services.trajectory import test_acquisition as acquisition_tests
from tests.services.trajectory.test_acquisition import ROWS, TRIGGER_SAMPLE
from tests.services.trajectory.test_chain import (
    CURRENT,
    TURNS,
    TURNS_RB,
    _machine_layer,
    _run,
    _slots,
)
from tests.services.trajectory.test_doors import (
    ALICE_SECRET,
    BOB_SECRET,
    COMMANDS,
    OWN_SECRET,
    TARGET,
    TOOL,
    _as,
    _code,
    _Machine,
    _shot,
)

from osprey.interfaces.web_auth import WebCredentials
from osprey.services.trajectory.app import LAUNCH_TOKEN_ENV, RouteRefused, create_app
from osprey.services.trajectory.raw import (
    RAW_EVICTED,
    RawBuffer,
    RawEvicted,
    RawUnits,
    decode_raw,
    encode_raw,
    units_of,
)
from osprey.services.trajectory.startup import TrajectorySettings, TrajectoryStartup
from osprey.services.trajectory.state import ChainBody, ShotResult
from osprey.services.trajectory.store import STATE_DIR_ENV_VAR, TrajectoryStore

# The acquisition tests' deployment, validator and store fixtures, shared here.
_deployment = acquisition_tests._deployment
optional_validator = acquisition_tests.optional_validator
store = acquisition_tests.store

UNITS = RawUnits(x="mm", y="mm", sum="counts")


def _slots_with_units() -> dict[str, Any]:
    slots = _slots()
    slots["devices"] = {
        "SR/BPM1": {"units": {"x": "mm", "y": "mm", "sum": None}},
        "SR/BPM2": {"units": {"x": "um", "y": None, "sum": "counts"}},
    }
    return slots


def _buffer(n_included: int = 1, n_bpm: int = 3, samples: int = 5) -> RawBuffer:
    rng = np.random.default_rng(7)
    injections = [
        tuple(rng.normal(size=(n_bpm, samples)) for _ in range(3)) for _ in range(n_included)
    ]
    return RawBuffer.of(injections, trigger_sample=2, units=UNITS)


def _root(shot_id: str) -> ShotResult:
    return _shot(shot_id)


def _derived(shot_id: str, parent: str) -> ShotResult:
    return _shot(shot_id).model_copy(update={"parent_id": parent})


# -- the file ----------------------------------------------------------------------


def test_the_file_is_a_length_a_json_header_and_aligned_float32_samples() -> None:
    buffer = _buffer(n_included=2)
    payload = encode_raw(buffer)

    (length,) = struct.unpack_from("<I", payload)
    assert length % 4 == 0
    header = json.loads(payload[4 : 4 + length])
    assert header == {
        "shape": [2, 3, 3, 5],
        "trigger_sample": 2,
        "units": {"x": "mm", "y": "mm", "sum": "counts"},
    }
    samples = np.frombuffer(payload, dtype="<f4", offset=4 + length).reshape(header["shape"])
    np.testing.assert_array_equal(samples, buffer.samples)
    assert samples.dtype == np.float32


def test_a_file_reads_back_as_the_buffer_it_was_written_from() -> None:
    buffer = _buffer(n_included=3, n_bpm=4, samples=7)
    read = decode_raw(encode_raw(buffer))
    assert read.header == buffer.header
    np.testing.assert_array_equal(read.samples, buffer.samples)


def test_a_truncated_file_is_refused() -> None:
    payload = encode_raw(_buffer())
    with pytest.raises(ValueError, match="samples"):
        decode_raw(payload[:-4])
    with pytest.raises(ValueError):
        decode_raw(b"\x01")


def test_injections_of_different_shapes_do_not_make_a_buffer() -> None:
    planes_a = tuple(np.zeros((3, 5)) for _ in range(3))
    planes_b = tuple(np.zeros((3, 6)) for _ in range(3))
    with pytest.raises(ValueError):
        RawBuffer.of([planes_a, planes_b], trigger_sample=0, units=UNITS)
    with pytest.raises(ValueError):
        RawBuffer.of([], trigger_sample=0, units=UNITS)


def test_each_planes_unit_is_the_first_a_device_states() -> None:
    assert units_of(_slots_with_units()) == RawUnits(x="mm", y="mm", sum="counts")
    assert units_of(None) == RawUnits()
    assert units_of({"devices": {"a": {"units": {"x": None, "y": None, "sum": None}}}}) == (
        RawUnits()
    )


# -- capture at readout ------------------------------------------------------------


@pytest.mark.usefixtures("optional_validator")
async def test_a_chain_keeps_every_included_injection_trimmed_to_the_turns_tracked(
    store: TrajectoryStore,
) -> None:
    layer, machine = _machine_layer()
    machine.lose_from = [None, None, None]

    result = await _run(
        layer,
        store,
        ChainBody(steps=["arm", "inject", "readout"], n_shots=3),
        slots=_slots_with_units(),
    )

    shot = result.shot
    assert shot is not None and shot.shots_included == 3
    raw_samples = TRIGGER_SAMPLE + TURNS
    assert shot.raw_samples == raw_samples
    root, buffer = await store.load_raw("SR", shot.shot_id)
    assert root == shot.shot_id
    assert buffer.header.shape == (3, 3, len(ROWS.bpms), raw_samples)
    assert buffer.header.trigger_sample == TRIGGER_SAMPLE
    assert buffer.header.units == RawUnits(x="mm", y="mm", sum="counts")
    assert buffer.samples.dtype == np.float32
    # Plane order x, y, sum; one row per BPM; the readout's own samples.
    for index in range(len(ROWS.bpms)):
        np.testing.assert_array_equal(
            buffer.samples[0, 0, index], np.asarray(machine.values[ROWS.x[index]][:raw_samples])
        )
        np.testing.assert_array_equal(
            buffer.samples[0, 2, index], np.asarray(machine.values[ROWS.sum[index]][:raw_samples])
        )


@pytest.mark.usefixtures("optional_validator")
async def test_a_failed_injection_does_not_enter_the_buffer(store: TrajectoryStore) -> None:
    layer, machine = _machine_layer()
    # The second injection's arm write is lost: it fails and is excluded.
    machine.drop_arm_on = {1}

    result = await _run(layer, store, ChainBody(steps=["arm", "inject", "readout"], n_shots=3))

    shot = result.shot
    assert shot is not None and len(shot.shots_failed) >= 1
    _root_id, buffer = await store.load_raw("SR", shot.shot_id)
    assert buffer.header.shape[0] == shot.shots_included


@pytest.mark.usefixtures("optional_validator")
async def test_a_chain_without_an_included_injection_keeps_no_buffer(
    store: TrajectoryStore,
) -> None:
    layer, machine = _machine_layer()
    machine.drop_arm = True

    result = await _run(layer, store, ChainBody(steps=["arm", "inject", "readout"]))

    shot = result.shot
    assert shot is not None and shot.shots_included == 0 and shot.raw_samples is None
    with pytest.raises(LookupError, match="no raw buffer"):
        await store.load_raw("SR", shot.shot_id)
    assert not list((store.root / "raw" / "SR").glob("*.raw"))


def _long_series(level: float, *, stored: float, arrival: int, samples: int) -> list[float]:
    """Stored charge from the first sample; the injected shot adds ``level`` from ``arrival``."""
    series = [stored] * samples
    for index in range(arrival, samples):
        series[index] = stored + level
    return series


@pytest.mark.usefixtures("optional_validator")
async def test_a_hundred_turns_tracked_with_stored_charge_keep_110_samples_and_no_loss(
    store: TrajectoryStore,
) -> None:
    trigger, turns, samples = 10, 100, 400
    layer, machine = _machine_layer()
    layer.startup.commands = dataclasses.replace(layer.startup.commands, trigger_sample=trigger)
    machine._post(TURNS_RB, turns, acquisition_tests.EPOCH)
    machine._post(CURRENT, 5.0, acquisition_tests.EPOCH)
    plain_inject = machine._inject

    def inject() -> None:
        plain_inject()
        stamp = machine.stamps[ROWS.sum[0]]
        for index in range(len(ROWS.bpms)):
            machine._post(
                ROWS.x[index],
                _long_series(0.1 * (index + 1), stored=0.0, arrival=trigger, samples=samples),
                stamp,
            )
            machine._post(
                ROWS.y[index],
                _long_series(-0.1 * (index + 1), stored=0.0, arrival=trigger, samples=samples),
                stamp,
            )
            # Stored charge on every BPM; after the tracked turns a much larger
            # step that the whole record would take for the arrival.
            total = _long_series(2.0, stored=1.0, arrival=trigger, samples=samples)
            for later in range(trigger + turns + 40, samples):
                total[later] += 50.0
            machine._post(ROWS.sum[index], total, stamp)

    machine._inject = inject

    result = await _run(layer, store, ChainBody(steps=["arm", "inject", "readout"]))

    shot = result.shot
    assert shot is not None and shot.shots_included == 1, shot
    assert shot.raw_samples == trigger + turns == 110
    # No loss point: every BPM kept beam over samples 0..109, and the arrival
    # is the injected shot's, not the later step outside the tracked turns.
    assert shot.lost == []
    assert shot.i0 == trigger
    assert shot.window is not None and shot.window[1] <= 110
    _root_id, buffer = await store.load_raw("SR", shot.shot_id)
    assert buffer.header.shape == (1, 3, len(ROWS.bpms), 110)
    assert float(buffer.samples[..., 2, :, :].max()) == pytest.approx(3.0)


# -- the store ---------------------------------------------------------------------


async def test_shots_and_buffers_prune_together_by_root_shot_keeping_the_latest(
    tmp_path: Path,
) -> None:
    store = TrajectoryStore(tmp_path, shots_per_model=2)
    await store.save_shot(_root("r1"), _buffer())
    await store.save_shot(_root("r2"), _buffer())
    await store.save_shot(_derived("r1-d1", "r1"), None)
    # r1's derivation keeps r1's group newest of the two roots.
    await store.save_shot(_root("r3"), _buffer())

    stored = {shot.shot_id for shot in await store.list_shots("SR")}
    assert stored == {"r3", "r1-d1", "r1"}
    raw = {path.stem for path in (tmp_path / "raw" / "SR").glob("*.raw")}
    assert raw == {"r1", "r3"}
    with pytest.raises(RawEvicted):
        await store.load_raw("SR", "r2")

    await store.save_shot(_root("r4"), _buffer())
    stored = {shot.shot_id for shot in await store.list_shots("SR")}
    assert stored == {"r4", "r3"}
    for pruned in ("r1", "r1-d1"):
        with pytest.raises(RawEvicted):
            await store.load_raw("SR", pruned)
    assert (await store.load_raw("SR", "r4"))[0] == "r4"


async def test_the_latest_shot_is_kept_at_a_retention_of_one(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path, shots_per_model=1)
    await store.save_shot(_root("r1"), _buffer())
    await store.save_shot(_derived("r1-d1", "r1"), None)
    assert {s.shot_id for s in await store.list_shots("SR")} == {"r1", "r1-d1"}
    await store.save_shot(_root("r2"), _buffer())
    assert [s.shot_id for s in await store.list_shots("SR")] == ["r2"]


async def test_a_derived_shot_reads_its_roots_buffer(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path)
    buffer = _buffer()
    await store.save_shot(_root("r1"), buffer)
    await store.save_shot(_derived("r1-d1", "r1"), None)
    await store.save_shot(_derived("r1-d2", "r1-d1"), None)

    root, read = await store.load_raw("SR", "r1-d2")
    assert root == "r1"
    np.testing.assert_array_equal(read.samples, buffer.samples)


async def test_an_unknown_shot_is_not_found_not_evicted(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path)
    with pytest.raises(LookupError) as raised:
        await store.load_raw("SR", "never")
    assert not isinstance(raised.value, RawEvicted)


async def test_a_shot_saved_again_after_its_eviction_is_no_longer_evicted(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path, shots_per_model=1)
    await store.save_shot(_root("r1"), _buffer())
    await store.save_shot(_root("r2"), _buffer())
    with pytest.raises(RawEvicted):
        await store.load_raw("SR", "r1")
    await store.save_shot(_root("r1"), _buffer())
    assert (await store.load_raw("SR", "r1"))[0] == "r1"


def test_the_store_default_cap_is_the_raw_retention_default(tmp_path: Path) -> None:
    from osprey.services.trajectory import store as store_module
    from osprey.services.trajectory.constants import DEFAULT_RAW_RETENTION

    assert TrajectoryStore(tmp_path).shots_per_model == DEFAULT_RAW_RETENTION
    assert not hasattr(store_module, "SHOTS_PER_MODEL")


# -- the route ---------------------------------------------------------------------

RETENTION = 2


def _app(machine: Any) -> FastAPI:
    async def start() -> TrajectoryStartup:
        return TrajectoryStartup(
            target=TARGET,
            connector=machine,
            commands=COMMANDS,
            guarded_run_dir=Path("/unused"),
            tool=TOOL,
            settings=TrajectorySettings(raw_retention=RETENTION),
        )

    app = create_app(start=start)
    app.state.web_credentials = WebCredentials(
        operator_secret=OWN_SECRET,
        panel_token="panel-token",
        roster_secrets=(ALICE_SECRET, BOB_SECRET),
        roster_owners=("alice", "bob"),
    )
    return app


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "state"
    monkeypatch.setenv(STATE_DIR_ENV_VAR, str(directory))
    return directory


@pytest.fixture
def raw_client(state_dir: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ARG001
    monkeypatch.setenv(LAUNCH_TOKEN_ENV, "launch-token")
    with TestClient(_app(_Machine())) as test_client:
        yield test_client


def _panel() -> dict[str, str]:
    return _as(ALICE_SECRET, "/panel/api")


def test_the_route_serves_the_raw_file_as_octet_stream(raw_client: TestClient) -> None:
    store: TrajectoryStore = raw_client.app.state.store  # type: ignore[attr-defined]
    buffer = _buffer(n_included=2)
    shot = _shot()
    asyncio.run(store.save_shot(shot, buffer))

    response = raw_client.get(f"/panel/api/shots/{shot.shot_id}/raw", headers=_panel())

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.content == encode_raw(buffer)


def test_a_derived_shot_serves_its_roots_file(raw_client: TestClient) -> None:
    store: TrajectoryStore = raw_client.app.state.store  # type: ignore[attr-defined]
    buffer = _buffer()
    asyncio.run(store.save_shot(_root("r1"), buffer))
    asyncio.run(store.save_shot(_derived("r1-d1", "r1"), None))

    response = raw_client.get("/panel/api/shots/r1-d1/raw", headers=_panel())

    assert response.status_code == 200, response.text
    assert response.content == encode_raw(buffer)


def test_a_pruned_shot_answers_410_raw_evicted(raw_client: TestClient) -> None:
    store: TrajectoryStore = raw_client.app.state.store  # type: ignore[attr-defined]
    for shot_id in ("r1", "r2", "r3"):
        asyncio.run(store.save_shot(_root(shot_id), _buffer()))

    response = raw_client.get("/panel/api/shots/r1/raw", headers=_panel())

    assert response.status_code == 410
    assert _code(response) == RAW_EVICTED
    assert RouteRefused.statuses[RAW_EVICTED] == 410


def test_an_unknown_shot_answers_404_and_a_bad_id_400(raw_client: TestClient) -> None:
    response = raw_client.get("/panel/api/shots/never/raw", headers=_panel())
    assert response.status_code == 404
    assert _code(response) == "shot_not_found"
    response = raw_client.get("/panel/api/shots/bad%20id/raw", headers=_panel())
    assert response.status_code == 400
    assert _code(response) == "invalid_name"


def test_the_agent_door_has_no_raw_route(raw_client: TestClient) -> None:
    response = raw_client.get(
        "/api/shots/r1/raw", headers=_as(ALICE_SECRET, "/api", **{"X-Launch-Token": "x"})
    )
    assert response.status_code == 404


def test_the_published_retention_is_the_stores_cap(raw_client: TestClient) -> None:
    store: TrajectoryStore = raw_client.app.state.store  # type: ignore[attr-defined]
    assert store.shots_per_model == RETENTION
    state = raw_client.get("/panel/api/state", headers=_panel()).json()
    assert state["retention"] == RETENTION
