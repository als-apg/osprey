"""One world: the live septum trace and the archived one are the same function.

The demo's injection septum (SEK) serves a 500-point current trace. Two paths
reach it, and they must never disagree about what the machine did at an instant:

* **live** — the texture model behind the virtual accelerator computes the trace
  at its clock from the held driver (the septum setpoint) and enable (its on
  state);
* **seeded** — the archive composite computes the trace at a grid instant from
  the same driver and enable levels, and the seeder stores it as a waveform
  document that ``MongoDBArchiverConnector`` and ``archiver_read`` read back.

Both are :func:`osprey_connectors.simulation.pulse.trace`. This file pins that
from the outside, on the built example facility rather than a hand-written view:
the Mongo-free comparison always runs, and the store-backed ones run against the
session's shared MongoDB container. A skipped store test is a failed gate here,
not a pass: the gate runs ``-rs`` and refuses any skip.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from osprey_connectors.archiver.field_names import field_name
from osprey_connectors.simulation import pulse
from osprey_connectors.simulation.archive import (
    DATE_FIELD,
    SeedKnobs,
    build,
    seed_base,
    waveform_collection,
    waveform_document,
)
from osprey_connectors.simulation.texture import TextureModel
from osprey_connectors.simulation.view import SEEDS_FILE, VARIABLES_FILE

# The session-scoped MongoDB container, shared with the connector and recorder
# tests rather than re-declared.
from tests.connectors.conftest import mongodb_container  # noqa: F401

if TYPE_CHECKING:
    from tests._builds import BuiltProject

#: The injection septum's trace, and the two channels it is a function of.
SEK = "SR:MAG:SEPTUM:01:CURRENT:WAVEFORM"
DRIVER = "SR:MAG:SEPTUM:01:CURRENT:SP"
ENABLE = "SR:MAG:SEPTUM:01:STATUS:ON"
#: The septum's seeded nominal setpoint, in A.
NOMINAL_DRIVE = 3000.0

#: The fixed instant everything is read at: on the minute, so it is a tail-grid
#: instant the seeder stores a trace at.
T = datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC)

#: The smallest archive the knobs allow: a day of minute traces, an hour dense.
SMALL = SeedKnobs(retention_days=1, hot_span_hours=1, hot_cadence_sec=10, tail_cadence_sec=60)

#: How many seeded minutes the ``archiver_read`` comparison samples.
SAMPLED_MINUTES = 8


# ---------------------------------------------------------------------------
# The built facility's simulator view
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_view(built_control_assistant: BuiltProject, tmp_path_factory) -> Path:
    """The example facility's simulator view, as the build renders it."""
    root = tmp_path_factory.mktemp("equivalence") / "simulator"
    prefix = "data/simulator/"
    for name, data in built_control_assistant.outputs[0].files.items():
        if name.startswith(prefix):
            target = root / name[len(prefix) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return root


def live_texture(view: Path, clock) -> TextureModel:
    """The texture model the virtual accelerator serves SEK from, on ``clock``."""
    variables = json.loads((view / VARIABLES_FILE).read_text())
    seeds = json.loads((view / SEEDS_FILE).read_text())
    return TextureModel(variables, seeds, clock=clock)


def live_trace(texture: TextureModel) -> np.ndarray:
    """SEK as the live path serves it right now."""
    return np.asarray(texture.get([SEK])[SEK], dtype=np.float64)


@pytest.fixture(scope="module")
def seeded(demo_view: Path):
    """The archive composite of the example facility, nominal active, anchored at T."""
    return build(demo_view, [], anchor_s=T.timestamp())


# ---------------------------------------------------------------------------
# Live against seeded, no store
# ---------------------------------------------------------------------------


def test_the_built_facility_serves_sek_as_a_pulse_of_its_setpoint_and_on_state(demo_view):
    """The premise: the build carries SEK as a pulsed waveform of driver and enable."""
    seeds = json.loads((demo_view / SEEDS_FILE).read_text())["seeds"]
    variables = json.loads((demo_view / VARIABLES_FILE).read_text())
    (channel,) = [c for c in variables["channels"] if c["address"] == SEK]

    assert seeds[SEK]["pulse"] == {"driver": DRIVER, "enable": ENABLE}
    assert channel["value_type"] == "waveform"
    assert channel["shape"] == [pulse.TRACE_POINTS]


def test_live_at_t_equals_seeded_at_t(demo_view, seeded):
    """The texture at clock T and the archive composite at T: elementwise equal."""
    texture = live_texture(demo_view, clock=lambda: T.timestamp())

    live = live_trace(texture)
    (archived,) = seeded.traces(SEK, [T.timestamp()])

    assert live.shape == (pulse.TRACE_POINTS,)
    assert np.array_equal(live, archived)
    assert np.array_equal(live, pulse.trace(SEK, T.timestamp(), NOMINAL_DRIVE, True))
    assert live.any()


def test_the_stored_document_holds_the_live_trace(demo_view, seeded):
    """The document the seeder writes at T carries exactly the live samples."""
    texture = live_texture(demo_view, clock=lambda: T.timestamp())
    (archived,) = seeded.traces(SEK, [T.timestamp()])
    expires = T + timedelta(days=1)

    document = waveform_document(SEK, T, archived, expire_at=expires)

    assert document[field_name(SEK)] == live_trace(texture).tolist()


@pytest.mark.parametrize("offset_s", [-3540.0, -60.0, 0.0, 60.0, 1234.5])
def test_live_and_seeded_agree_at_every_instant(demo_view, seeded, offset_s):
    """Not a coincidence at one instant: the two agree wherever both are asked."""
    instant = T.timestamp() + offset_s
    texture = live_texture(demo_view, clock=lambda: instant)

    (archived,) = seeded.traces(SEK, [instant])

    assert np.array_equal(live_trace(texture), archived)


def test_a_moved_driver_moves_live_as_the_pulse_says(demo_view):
    """The live trace follows the held driver and enable, as the seeded one does."""
    texture = live_texture(demo_view, clock=lambda: T.timestamp())

    texture.set({DRIVER: 1500.0})
    lowered = live_trace(texture)
    # The on state is read-only to an operator; a scenario's write holds it.
    texture.set_active({DRIVER: 1500.0, ENABLE: "FALSE"})
    off = live_trace(texture)

    assert np.array_equal(lowered, pulse.trace(SEK, T.timestamp(), 1500.0, True))
    assert not off.any()


# ---------------------------------------------------------------------------
# Through the store
# ---------------------------------------------------------------------------


def _client(container):
    from pymongo import MongoClient

    return MongoClient(
        host=container["host"],
        port=container["port"],
        username=container["username"],
        password=container["password"],
        authSource=container["auth_db"],
    )


@pytest.fixture
def store(mongodb_container):  # noqa: F811
    """A fresh sample collection and its waveform sibling, dropped afterwards."""
    client = _client(mongodb_container)
    database = client[mongodb_container["db_name"]]
    name = "waveform_equivalence"
    collection = database[name]
    waveforms = database[waveform_collection(name)]
    collection.drop()
    waveforms.drop()
    try:
        yield collection, waveforms
    finally:
        collection.drop()
        waveforms.drop()
        client.close()


@pytest.fixture
def connector_config(mongodb_container, store, monkeypatch):  # noqa: F811
    """The connector config of the store, password by environment name."""
    from osprey_connectors.archiver.mongodb_archiver_connector import (
        HOST_OVERRIDE_ENV,
        PORT_OVERRIDE_ENV,
    )

    monkeypatch.delenv(HOST_OVERRIDE_ENV, raising=False)
    monkeypatch.delenv(PORT_OVERRIDE_ENV, raising=False)
    password_env = "WAVEFORM_EQUIVALENCE_MONGO_PASSWORD"
    monkeypatch.setenv(password_env, mongodb_container["password"])
    collection, _ = store
    return {
        "host": mongodb_container["host"],
        "port": mongodb_container["port"],
        "name": mongodb_container["db_name"],
        "collection": collection.name,
        "auth": {
            "source": mongodb_container["auth_db"],
            "username": mongodb_container["username"],
            "password_env": password_env,
        },
        "timeout_s": 10,
    }


@pytest.fixture
async def connector(connector_config):
    """A connected ``MongoDBArchiverConnector`` on the store."""
    from osprey_connectors.archiver.mongodb_archiver_connector import MongoDBArchiverConnector

    archiver = MongoDBArchiverConnector()
    await archiver.connect(connector_config)
    try:
        yield archiver
    finally:
        await archiver.disconnect()


@pytest.fixture
def read_tool(connector, tmp_path, monkeypatch):
    """``archiver_read`` serving from the real connector on the store.

    The project is the test's working directory, so the saved artifact's path is
    relative to ``tmp_path``. The artifact store and the server context are
    process-wide singletons: both are reset on the way in, so a store an earlier
    test in this worker rooted elsewhere is never reused, and on the way out.
    """
    from osprey.mcp_server.control_system.server_context import (
        initialize_server_context,
        reset_server_context,
    )
    from osprey.mcp_server.control_system.tools.archiver_read import archiver_read
    from osprey.stores.artifact_store import reset_artifact_store
    from tests.mcp_server.conftest import get_tool_fn

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("archiver:\n  type: mongodb_archiver\n")
    reset_artifact_store()
    initialize_server_context()
    try:
        with patch(
            "osprey_connectors.factory.ConnectorFactory.create_archiver_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            yield get_tool_fn(archiver_read)
    finally:
        reset_server_context()
        reset_artifact_store()


async def _read_traces(read_tool, tmp_path: Path, start: datetime, end: datetime):
    """``archiver_read(bin_size=0)`` of SEK over ``[start, end]``: (timestamps, traces)."""
    from tests.mcp_server.conftest import extract_response_dict

    result = await read_tool(
        channels=[SEK],
        start_time=start.isoformat(),
        end_time=end.isoformat(),
        processing="raw",
        bin_size=0,
    )
    response = extract_response_dict(result)
    assert response["status"] == "success", response
    saved = json.loads((tmp_path / response["data_file"]).read_text())["series"][SEK]
    stamps = [datetime.fromisoformat(stamp) for stamp in saved["timestamps"]]
    return stamps, [np.asarray(values, dtype=np.float64) for values in saved["values"]]


async def test_archiver_read_returns_one_seeded_trace_per_minute_equal_to_the_pulse(
    seeded, store, read_tool, tmp_path
):
    """Seed a window holding T; each sampled minute reads back as exactly ``trace(t)``."""
    collection, waveforms = store
    seed_base(collection, seeded, SMALL, t0=T, chunk_size=256, waveforms=waveforms)

    minutes = [T - timedelta(minutes=k * 97) for k in range(SAMPLED_MINUTES)]
    for minute in minutes:
        stamps, traces = await _read_traces(
            read_tool, tmp_path, minute - timedelta(seconds=30), minute + timedelta(seconds=29)
        )

        assert stamps == [minute]
        (trace,) = traces
        assert np.array_equal(trace, pulse.trace(SEK, minute.timestamp(), NOMINAL_DRIVE, True))


async def test_the_seeded_trace_at_t_reads_back_as_the_live_trace(
    demo_view, seeded, store, connector
):
    """The connector's trace at T is the texture's trace at clock T."""
    collection, waveforms = store
    seed_base(collection, seeded, SMALL, t0=T, chunk_size=256, waveforms=waveforms)
    texture = live_texture(demo_view, clock=lambda: T.timestamp())

    frame = await connector.get_data(
        channels=[SEK],
        start_date=T - timedelta(seconds=30),
        end_date=T + timedelta(seconds=30),
        precision_ms=0,
    )

    (value,) = frame.loc[frame["channel"] == SEK, "value"].tolist()
    assert np.array_equal(np.asarray(value, dtype=np.float64), live_trace(texture))


@pytest.fixture
def mongo_login(mongodb_container):  # noqa: F811
    """The shared container's connection parameters."""
    return mongodb_container


async def test_a_live_hour_records_sixty_traces_each_the_pulse_of_its_minute(
    demo_view, mongo_login, store, connector, tmp_path
):
    """A recorder driven by a stepped clock over one hour, reading the live texture."""
    import yaml

    from osprey.services.archiver_recorder import ArchiveWriter, Recorder, RecorderSettings
    from osprey.services.archiver_recorder.config import ChannelSet, resolve_channel_addresses

    collection, waveforms = store
    resolved = resolve_channel_addresses(demo_view)
    assert SEK in resolved.waveforms

    clock = {"now": T}
    texture = live_texture(demo_view, clock=lambda: clock["now"].timestamp())

    async def reader(addresses):
        return texture.get(list(addresses))

    settings = RecorderSettings(
        host=mongo_login["host"],
        port=mongo_login["port"],
        database=mongo_login["db_name"],
        collection=collection.name,
        auth_source=mongo_login["auth_db"],
        username=mongo_login["username"],
        password_env="UNUSED",
        timeout_s=10,
        cadence_sec=10,
        tail_cadence_sec=60,
        poll_sec=30,
        hot_span_hours=1,
        retention_days=1,
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump({"control_system": {"type": "virtual_accelerator"}}))
    writer = ArchiveWriter(settings, mongo_login["password"])
    writer.connect()
    recorder = Recorder(
        settings=settings,
        addresses=ChannelSet([DRIVER, ENABLE, SEK], waveforms=[SEK]),
        writer=writer,
        config_path=config_path,
        reader=reader,
        recording=True,
    )
    try:
        for step in range(0, 3600, 10):
            clock["now"] = T + timedelta(seconds=step)
            await recorder.tick(clock["now"])
    finally:
        writer.close()

    hour_end = T + timedelta(hours=1)
    recorded = waveforms.count_documents({DATE_FIELD: {"$gte": T, "$lt": hour_end}})
    assert abs(recorded - 60) <= 1

    frame = await connector.get_data(
        channels=[SEK], start_date=T, end_date=hour_end, precision_ms=0
    )
    rows = frame.loc[frame["channel"] == SEK]
    assert abs(len(rows) - 60) <= 1
    for stamp, value in zip(rows["timestamp"], rows["value"], strict=True):
        instant = stamp.to_pydatetime().timestamp()
        expected = pulse.trace(SEK, instant, NOMINAL_DRIVE, True)
        assert np.array_equal(np.asarray(value, dtype=np.float64), expected)
