"""Contracts for the archiver recorder.

The recorder writes into a collection whose other half is written by the base
seeder and read by ``MongoDBArchiverConnector``. Almost everything worth
pinning here is about staying compatible with those two neighbours rather than
about the recorder in isolation:

* the sample grid is absolute, so recorded and seeded documents share instants;
* the document shape is the one the connector reads, and a channel that did not
  answer is absent rather than filled in;
* the expiry stamp is per-document and two-tiered, and a document the recorder
  did not create never gets its expiry rewritten (that is how the scenario
  seeder protects an event window from retention);
* enablement is polled and a torn config read changes nothing.

The Channel Access side is a plain async callable, so every test here drives it
directly. The write shape is additionally checked against a real MongoDB via
the existing session-scoped testcontainers fixture, because "the document the
connector can read" is not a claim a fake can settle.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from osprey.services.archiver_recorder import (
    ArchiveWriter,
    Recorder,
    RecorderConfigError,
    RecorderSettings,
    expiry_for,
    load_settings,
    read_control_system_type,
    resolve_channel_addresses,
)

# Not re-exported by the package: the enablement facts are this service's own
# internal contract, and the package surface stays the narrow reader it was.
from osprey.services.archiver_recorder.config import RecordingFacts, read_recording_facts

# The session-scoped MongoDB container, reused rather than re-declared: one
# container for the whole test session, skipping cleanly without Docker.
from tests.connectors.conftest import mongodb_container  # noqa: F401


def _settings(**overrides) -> RecorderSettings:
    base = {
        "host": "archiver-mongodb",
        "port": 27017,
        "database": "osprey_archiver",
        "collection": "pv_history",
        "auth_database": "admin",
        "username": "osprey",
        "password_env": "MONGO_ROOT_PASSWORD",
        "timeout_sec": 5,
        "cadence_sec": 10,
        "tail_cadence_sec": 60,
        "poll_sec": 30,
        "hot_span_hours": 48,
        "retention_days": 30,
    }
    base.update(overrides)
    return RecorderSettings(**base)


def _write_config(
    path: Path,
    control_system_type: str = "virtual_accelerator",
    *,
    live_gateway: tuple[str, int | str] | None = None,
    standin_gateway: tuple[str, int | str] | None = None,
    standin_port: int | None = None,
    recorder_deployed: bool = True,
    **extra,
) -> Path:
    """A rendered config.yml, optionally shaped like a stand-in deployment.

    ``live_gateway`` writes the ``epics`` connector block — the facility's
    authored real machine, which is what ``live`` always means and what an SSH
    tunnel also lands on. ``standin_gateway`` writes the
    ``live_standin`` connector block the ``standin`` target resolves to, the
    one the build fills with the stand-in's loopback address and port; it is the
    only block the recorded endpoint is ever derived from.

    ``standin_port`` writes ``services.live_standin.port``, the deployment
    saying it stood a second virtual accelerator up, and ``recorder_deployed``
    decides whether ``archiver_recorder`` is in ``deployed_services`` — the two
    halves of "whose past is in this store". All four are independent arguments
    because the interesting cases are the ones where they disagree.
    """
    config: dict = {
        "control_system": {"type": control_system_type},
        # This service is the recorder, so the deploy that mounts this config
        # ordinarily runs one. `recorder_deployed=False` is the render that
        # merely stood a stand-in up without ever sampling it.
        "deployed_services": ["mongodb", *(["archiver_recorder"] if recorder_deployed else [])],
        "archiver": {
            "mongodb_archiver": {
                "host": "localhost",
                "port": 27017,
                "name": "osprey_archiver",
                "collection": "pv_history",
                "auth": "admin",
                "username": "osprey",
                "password_env": "MONGO_ROOT_PASSWORD",
                "timeout": 5,
            }
        },
        "va_archiver": {
            "retention_days": 30,
            "hot_span_hours": 48,
            "hot_cadence_sec": 10,
            "tail_cadence_sec": 60,
            "recorder_cadence_sec": 10,
            "recorder_tail_cadence_sec": 60,
            "recorder_poll_sec": 30,
        },
    }
    for block, gateway in (("epics", live_gateway), ("live_standin", standin_gateway)):
        if gateway is None:
            continue
        address, port = gateway
        connector = config["control_system"].setdefault("connector", {})
        connector[block] = {"gateways": {"read_only": {"address": address, "port": port}}}
    if standin_port is not None:
        config["services"] = {"live_standin": {"port": standin_port}}
        config["deployed_services"].append("live_standin")
    config.update(extra)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class _FakeWriter:
    """Records what would have been stored, without a database."""

    def __init__(self) -> None:
        self.samples: list[tuple[datetime, dict[str, float]]] = []
        self.fail_with: Exception | None = None

    def write_sample(self, timestamp: datetime, values: dict[str, float]) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        self.samples.append((timestamp, dict(values)))


def _reader(values):
    async def read(addresses):
        return {a: v for a, v in values.items() if a in set(addresses)}

    return read


# ---------------------------------------------------------------------------
# Settings: no defaults, and the in-network override wins
# ---------------------------------------------------------------------------


def test_settings_come_from_the_rendered_config(tmp_path: Path) -> None:
    """Every knob is read from config.yml — the same block the profile emits
    and the seeder reads, so the two halves of the collection cannot end up
    with different cadences or retention spans."""
    settings = load_settings(_write_config(tmp_path / "config.yml"))

    assert (settings.cadence_sec, settings.tail_cadence_sec, settings.poll_sec) == (10, 60, 30)
    assert (settings.hot_span_hours, settings.retention_days) == (48, 30)
    assert (settings.database, settings.collection) == ("osprey_archiver", "pv_history")
    assert settings.password_env == "MONGO_ROOT_PASSWORD"


def test_in_network_host_and_port_override_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rendered connection block is the HOST view (loopback + the published
    port). Inside the compose network that names this container's own loopback,
    so the compose template's alias/container-port pair has to win."""
    monkeypatch.setenv("OSPREY_ARCHIVER_MONGODB_HOST", "archiver-mongodb")
    monkeypatch.setenv("OSPREY_ARCHIVER_MONGODB_PORT", "27017")

    settings = load_settings(_write_config(tmp_path / "config.yml"))

    assert (settings.host, settings.port) == ("archiver-mongodb", 27017)


@pytest.mark.parametrize(
    ("block", "key"),
    [("va_archiver", "recorder_cadence_sec"), ("archiver", "mongodb_archiver")],
)
def test_a_missing_knob_is_an_error_naming_it(tmp_path: Path, block: str, key: str) -> None:
    """No silent defaults. A recorder that invented a cadence would write an
    archive whose density disagreed with the seeded half of the same
    collection, and nothing downstream would notice."""
    path = _write_config(tmp_path / "config.yml")
    config = yaml.safe_load(path.read_text())
    del config[block][key]
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(RecorderConfigError, match=key):
        load_settings(path)


# ---------------------------------------------------------------------------
# Channel source: the manifest, never the limits DB
# ---------------------------------------------------------------------------


def test_channel_source_is_the_manifest_named_by_va_channels_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolved exactly as the IOC resolves it — a relative name against the
    shared data mount — so the recorder covers the channels actually served."""
    manifest = {
        "channels": [
            {
                "address": "SR:BPM01:X",
                "ring": "SR",
                "system": "diagnostics",
                "family": "BPM",
                "device": "BPM01",
                "field": "X",
                "subfield": "",
                "record_type": "ai",
                "noise": True,
                "partition": "static_noisy",
            }
        ]
    }
    (tmp_path / "channel_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("VA_CHANNELS_FILE", "channel_manifest.json")

    assert resolve_channel_addresses(tmp_path) == ["SR:BPM01:X"]


def test_an_unloadable_manifest_is_fatal_not_a_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Falling back to the built-in channel set would record another facility's
    namespace into this facility's archive — worse than not recording."""
    monkeypatch.setenv("VA_CHANNELS_FILE", "missing.json")

    with pytest.raises(RecorderConfigError, match="VA_CHANNELS_FILE"):
        resolve_channel_addresses(tmp_path)


# ---------------------------------------------------------------------------
# Retention: two tiers, decided by the timestamp alone
# ---------------------------------------------------------------------------


def test_tail_grid_samples_are_kept_for_the_full_retention_span() -> None:
    """A sample on the tail grid is the one that survives the hot span, so
    recorded history thins to the seeded tail's density instead of ending."""
    settings = _settings()
    on_grid = datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)
    off_grid = datetime(2026, 8, 10, 12, 0, 10, tzinfo=UTC)

    assert expiry_for(on_grid, settings) == on_grid + timedelta(days=30)
    assert expiry_for(off_grid, settings) == off_grid + timedelta(hours=48)


def test_expiry_tiers_follow_the_configured_spans() -> None:
    """Both spans are knobs: changing retention_days in the profile changes
    what the recorder stamps, with no code edit."""
    settings = _settings(retention_days=2, hot_span_hours=2)
    on_grid = datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)

    assert expiry_for(on_grid, settings) == on_grid + timedelta(days=2)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


async def test_samples_land_on_the_absolute_grid(tmp_path: Path) -> None:
    """Timestamps are floored to a multiple of the cadence since the epoch —
    not spaced from process start — so recorded documents share instants with
    seeded ones and a restart lands on the same grid."""
    writer = _FakeWriter()
    recorder = Recorder(
        settings=_settings(),
        addresses=["SR:BPM01:X"],
        writer=writer,
        config_path=_write_config(tmp_path / "config.yml"),
        reader=_reader({"SR:BPM01:X": 1.5}),
    )

    await recorder.tick(datetime(2026, 8, 10, 12, 0, 7, 400000, tzinfo=UTC))

    timestamp, values = writer.samples[0]
    assert timestamp == datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)
    assert values == {"SR:BPM01:X": 1.5}


async def test_a_slot_is_recorded_once(tmp_path: Path) -> None:
    """Two ticks inside one cadence interval write one document: a restart
    mid-interval must not double-write the instant just covered."""
    writer = _FakeWriter()
    recorder = Recorder(
        settings=_settings(),
        addresses=["SR:BPM01:X"],
        writer=writer,
        config_path=_write_config(tmp_path / "config.yml"),
        reader=_reader({"SR:BPM01:X": 1.5}),
    )

    assert await recorder.tick(datetime(2026, 8, 10, 12, 0, 1, tzinfo=UTC)) is True
    assert await recorder.tick(datetime(2026, 8, 10, 12, 0, 9, tzinfo=UTC)) is False
    assert await recorder.tick(datetime(2026, 8, 10, 12, 0, 10, tzinfo=UTC)) is True
    assert len(writer.samples) == 2


async def test_a_channel_that_did_not_answer_is_absent_not_filled_in(tmp_path: Path) -> None:
    """No carried-forward value and no placeholder. The connector treats an
    absent field as no sample, which is the only record that lets a reader tell
    a quiet channel from an unreachable one."""
    writer = _FakeWriter()
    recorder = Recorder(
        settings=_settings(),
        addresses=["SR:BPM01:X", "SR:BPM02:X"],
        writer=writer,
        config_path=_write_config(tmp_path / "config.yml"),
        reader=_reader({"SR:BPM01:X": 1.5}),
    )

    await recorder.tick(datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC))

    assert writer.samples[0][1] == {"SR:BPM01:X": 1.5}


async def test_nothing_is_written_when_no_channel_answers(tmp_path: Path) -> None:
    """An unreachable control system leaves a gap, and the gap is the honest
    record of it."""
    writer = _FakeWriter()
    recorder = Recorder(
        settings=_settings(),
        addresses=["SR:BPM01:X"],
        writer=writer,
        config_path=_write_config(tmp_path / "config.yml"),
        reader=_reader({}),
    )

    assert await recorder.tick(datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)) is False
    assert writer.samples == []


async def test_a_failed_write_does_not_end_the_loop(tmp_path: Path) -> None:
    """The store may be restarting and the next tick is seconds away. Crashing
    would take the Channel Access channel cache with it and reconnect every
    channel from scratch."""
    writer = _FakeWriter()
    writer.fail_with = RuntimeError("store is restarting")
    recorder = Recorder(
        settings=_settings(),
        addresses=["SR:BPM01:X"],
        writer=writer,
        config_path=_write_config(tmp_path / "config.yml"),
        reader=_reader({"SR:BPM01:X": 1.5}),
    )

    assert await recorder.tick(datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)) is False

    writer.fail_with = None
    assert await recorder.tick(datetime(2026, 8, 10, 12, 0, 10, tzinfo=UTC)) is True


async def test_the_channel_access_reader_drops_channels_that_did_not_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """aioca reports a failed read as a falsy ``CANothing`` in the results list
    rather than raising, which is what makes ``throw=False`` safe: one
    unreachable channel must not cost the whole sample. Dropping those here is
    what turns "did not answer" into an absent field downstream."""
    import aioca

    from osprey.services.archiver_recorder.recorder import channel_access_reader

    async def fake_caget(pvs, **kwargs):
        return [1.5, aioca.CANothing(pvs[1], -1)]

    monkeypatch.setattr(aioca, "caget", fake_caget)

    values = await channel_access_reader(_settings())(["SR:BPM01:X", "SR:BPM02:X"])

    assert values == {"SR:BPM01:X": 1.5}


async def test_non_numeric_channels_are_skipped_and_warned_about_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The schema is one number per channel per instant. Saying so every
    cadence would bury everything else in the log."""
    writer = _FakeWriter()
    recorder = Recorder(
        settings=_settings(),
        addresses=["SR:BPM01:X", "SR:STATE:MODE"],
        writer=writer,
        config_path=_write_config(tmp_path / "config.yml"),
        reader=_reader({"SR:BPM01:X": 1.5, "SR:STATE:MODE": "USER"}),
    )

    with caplog.at_level(logging.WARNING):
        await recorder.tick(datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC))
        await recorder.tick(datetime(2026, 8, 10, 12, 0, 10, tzinfo=UTC))

    assert writer.samples[0][1] == {"SR:BPM01:X": 1.5}
    assert caplog.text.count("SR:STATE:MODE") == 1


# ---------------------------------------------------------------------------
# Enablement
# ---------------------------------------------------------------------------


async def test_a_mock_control_system_is_not_recorded(tmp_path: Path) -> None:
    """A real machine's readings spliced onto a synthesized past is exactly the
    two-world archive this service exists to prevent, and a mock has nothing
    worth recording in the first place."""
    writer = _FakeWriter()
    recorder = Recorder(
        settings=_settings(),
        addresses=["SR:BPM01:X"],
        writer=writer,
        config_path=_write_config(tmp_path / "config.yml", control_system_type="mock"),
        reader=_reader({"SR:BPM01:X": 1.5}),
    )

    assert await recorder.tick(datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)) is False
    assert writer.samples == []


async def test_the_control_system_flip_takes_effect_without_a_restart(tmp_path: Path) -> None:
    """The documented post-build flip is a config edit, and the poll is what
    makes it enough."""
    config_path = _write_config(tmp_path / "config.yml", control_system_type="mock")
    writer = _FakeWriter()
    recorder = Recorder(
        settings=_settings(poll_sec=30),
        addresses=["SR:BPM01:X"],
        writer=writer,
        config_path=config_path,
        reader=_reader({"SR:BPM01:X": 1.5}),
    )

    assert await recorder.tick(datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)) is False
    _write_config(config_path, control_system_type="virtual_accelerator")

    # Still inside the poll interval: the flip is not visible yet.
    assert await recorder.tick(datetime(2026, 8, 10, 12, 0, 10, tzinfo=UTC)) is False
    # Past it: recording resumes with no restart.
    assert await recorder.tick(datetime(2026, 8, 10, 12, 0, 40, tzinfo=UTC)) is True


async def test_a_stand_in_is_recorded_though_the_type_no_longer_says_so(tmp_path: Path) -> None:
    """`osprey set connector=epics` rewrites the rendered control_system.type
    without touching the stand-in this deployment stood up and records, and the
    machine on the other end of the Channel Access connection has not changed at
    all. Recording follows the machine, so the gate has to see the stand-in the
    type stopped naming."""
    writer = _FakeWriter()
    recorder = Recorder(
        settings=_settings(),
        addresses=["SR:BPM01:X"],
        writer=writer,
        config_path=_write_config(
            tmp_path / "config.yml",
            control_system_type="epics",
            standin_gateway=("127.0.0.1", 5074),
            standin_port=5074,
        ),
        reader=_reader({"SR:BPM01:X": 1.5}),
    )

    assert await recorder.tick(datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)) is True
    assert writer.samples[0][1] == {"SR:BPM01:X": 1.5}


async def test_a_stand_in_block_does_not_record_a_gateway_pointed_off_the_host(
    tmp_path: Path,
) -> None:
    """The recorded endpoint follows the `standin` target's own gateways, never
    a leftover `services:` block. Once those gateways name a real facility's
    host the target dials a real machine, whatever the deployment still says it
    once stood up — and recording a real machine into a synthesized archive is
    the failure this gate exists for."""
    writer = _FakeWriter()
    recorder = Recorder(
        settings=_settings(),
        addresses=["SR:BPM01:X"],
        writer=writer,
        config_path=_write_config(
            tmp_path / "config.yml",
            control_system_type="epics",
            standin_gateway=("epics-gateway.example.org", 5074),
            standin_port=5074,
        ),
        reader=_reader({"SR:BPM01:X": 1.5}),
    )

    assert await recorder.tick(datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)) is False
    assert writer.samples == []


async def test_a_stand_in_nothing_records_is_not_this_archives_machine(
    tmp_path: Path,
) -> None:
    """The store's history belongs to the machine the store records.

    A render that stood a stand-in up but runs no recorder samples nothing, so
    the stand-in is not what its archive holds — and this service must not start
    filing readings into it on the strength of the port alone. Both halves of
    the predicate the compose entry and the archive seed bind to are needed
    here too."""
    writer = _FakeWriter()
    recorder = Recorder(
        settings=_settings(),
        addresses=["SR:BPM01:X"],
        writer=writer,
        config_path=_write_config(
            tmp_path / "config.yml",
            control_system_type="epics",
            standin_gateway=("127.0.0.1", 5074),
            standin_port=5074,
            recorder_deployed=False,
        ),
        reader=_reader({"SR:BPM01:X": 1.5}),
    )

    assert await recorder.tick(datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)) is False
    assert writer.samples == []


async def test_an_ssh_tunnel_to_a_real_gateway_is_not_a_stand_in(tmp_path: Path) -> None:
    """Loopback alone proves nothing: forwarding a real facility's gateway to a
    local port is one hop from hardware. Without a stand-in the deployment
    built, there is no synthesized past for those readings to join."""
    writer = _FakeWriter()
    recorder = Recorder(
        settings=_settings(),
        addresses=["SR:BPM01:X"],
        writer=writer,
        config_path=_write_config(
            tmp_path / "config.yml",
            control_system_type="epics",
            live_gateway=("localhost", 5064),
        ),
        reader=_reader({"SR:BPM01:X": 1.5}),
    )

    assert await recorder.tick(datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)) is False
    assert writer.samples == []


async def test_a_torn_config_read_keeps_the_last_known_state(tmp_path: Path) -> None:
    """Config writes are truncate-in-place, not atomic. Reading a half-written
    file as "the control system changed" would stop and restart recording on a
    write that changed nothing."""
    config_path = _write_config(tmp_path / "config.yml")
    writer = _FakeWriter()
    recorder = Recorder(
        settings=_settings(poll_sec=30),
        addresses=["SR:BPM01:X"],
        writer=writer,
        config_path=config_path,
        reader=_reader({"SR:BPM01:X": 1.5}),
    )
    assert await recorder.tick(datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)) is True

    config_path.write_text("control_system:\n  type: virtual_ac", encoding="utf-8")
    config_path.write_text("{unterminated", encoding="utf-8")

    assert recorder.refresh_enablement(datetime(2026, 8, 10, 12, 0, 40, tzinfo=UTC)) is True


async def test_a_torn_read_keeps_stand_in_recording_alive_too(tmp_path: Path) -> None:
    """The torn-read tolerance covers the whole answer, not the type half of it.
    A recorder enabled by the stand-in must not fall out of the machine's own
    archive because a poll landed on a half-written file."""
    config_path = _write_config(
        tmp_path / "config.yml",
        control_system_type="epics",
        standin_gateway=("127.0.0.1", 5074),
        standin_port=5074,
    )
    recorder = Recorder(
        settings=_settings(poll_sec=30),
        addresses=["SR:BPM01:X"],
        writer=_FakeWriter(),
        config_path=config_path,
        reader=_reader({"SR:BPM01:X": 1.5}),
    )
    assert await recorder.tick(datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)) is True

    config_path.write_text("{unterminated", encoding="utf-8")

    assert recorder.refresh_enablement(datetime(2026, 8, 10, 12, 0, 40, tzinfo=UTC)) is True


async def test_the_idle_line_names_both_ways_a_machine_could_be_recordable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The idle line is the whole explanation an operator gets from `docker
    logs`. Naming only the type would send someone to edit the one condition
    that is not the reason they are being told nothing is written."""
    recorder = Recorder(
        settings=_settings(poll_sec=1),
        addresses=["SR:BPM01:X"],
        writer=_FakeWriter(),
        config_path=_write_config(
            tmp_path / "config.yml",
            control_system_type="epics",
            live_gateway=("localhost", 5064),
        ),
        reader=_reader({"SR:BPM01:X": 1.5}),
    )

    with caplog.at_level(logging.INFO):
        await recorder.tick(datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC))

    assert "control_system.type=epics" in caplog.text
    assert "virtual_accelerator" in caplog.text
    assert "services.live_standin.port" in caplog.text


async def test_idling_logs_once_not_every_poll(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A recorder idling for a week must not fill the logs saying so."""
    recorder = Recorder(
        settings=_settings(poll_sec=1),
        addresses=["SR:BPM01:X"],
        writer=_FakeWriter(),
        config_path=_write_config(tmp_path / "config.yml", control_system_type="mock"),
        reader=_reader({"SR:BPM01:X": 1.5}),
    )

    with caplog.at_level(logging.INFO):
        for second in range(0, 40, 10):
            await recorder.tick(datetime(2026, 8, 10, 12, 0, second, tzinfo=UTC))

    assert caplog.text.count("Idle, writing nothing") == 1


# ---------------------------------------------------------------------------
# The write shape, against a real MongoDB
# ---------------------------------------------------------------------------


@pytest.fixture
def archive_collection(mongodb_container):  # noqa: F811
    """A clean collection plus the settings pointing at it."""
    from pymongo import MongoClient

    client = MongoClient(
        host=mongodb_container["host"],
        port=mongodb_container["port"],
        username=mongodb_container["username"],
        password=mongodb_container["password"],
        authSource=mongodb_container["auth_db"],
    )
    collection = client[mongodb_container["db_name"]]["recorder_shape"]
    collection.delete_many({})
    settings = _settings(
        host=mongodb_container["host"],
        port=mongodb_container["port"],
        database=mongodb_container["db_name"],
        collection="recorder_shape",
        username=mongodb_container["username"],
        auth_database=mongodb_container["auth_db"],
    )
    try:
        yield settings, collection, mongodb_container["password"]
    finally:
        collection.delete_many({})
        client.close()


def test_stored_documents_carry_the_shape_the_connector_reads(archive_collection) -> None:
    """One document per timestamp, ``date`` plus a field per PV — the shape
    ``MongoDBArchiverConnector`` queries and projects."""
    settings, collection, password = archive_collection
    writer = ArchiveWriter(settings, password)
    writer.connect()
    timestamp = datetime(2026, 8, 10, 12, 0, 10, tzinfo=UTC)
    try:
        writer.write_sample(timestamp, {"SR:BPM01:X": 1.5, "SR:BPM02:X": -0.25})
    finally:
        writer.close()

    doc = collection.find_one({"date": timestamp})
    assert doc is not None
    assert doc["SR:BPM01:X"] == 1.5
    assert doc["SR:BPM02:X"] == -0.25
    assert doc["expireAt"].replace(tzinfo=UTC) == timestamp + timedelta(hours=48)


def test_writing_an_instant_twice_merges_and_never_restamps_expiry(archive_collection) -> None:
    """The two guards that let the recorder coexist with the seeded half of the
    collection: a second write leaves values it did not read alone, and it does
    not put an expiry back on a document whose expiry the scenario seeder
    deliberately removed to protect an event window from retention."""
    settings, collection, password = archive_collection
    timestamp = datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)
    collection.insert_one({"date": timestamp, "SR:SEEDED:VALUE": 42.0})

    writer = ArchiveWriter(settings, password)
    writer.connect()
    try:
        writer.write_sample(timestamp, {"SR:BPM01:X": 1.5})
    finally:
        writer.close()

    doc = collection.find_one({"date": timestamp})
    assert doc["SR:SEEDED:VALUE"] == 42.0, "a value the recorder does not read must survive"
    assert doc["SR:BPM01:X"] == 1.5
    assert "expireAt" not in doc, (
        "re-stamping expiry on a document the recorder did not create would make a "
        "scenario-protected event window expirable"
    )


def test_a_missing_credential_is_a_connection_error(archive_collection) -> None:
    """A recorder that cannot write is not degraded, it is pointless — so the
    cause lands in the log at startup rather than as a silently empty archive."""
    settings, _collection, _password = archive_collection

    with pytest.raises(ConnectionError, match="archive store"):
        ArchiveWriter(settings, "not-the-password").connect()


def test_control_system_type_reads_the_file_each_time(tmp_path: Path) -> None:
    """The enablement question is answered from disk, not from a cached parse:
    the whole point is that an edit is visible without a restart."""
    config_path = _write_config(tmp_path / "config.yml", control_system_type="mock")
    assert read_control_system_type(config_path) == "mock"

    _write_config(config_path, control_system_type="virtual_accelerator")
    assert read_control_system_type(config_path) == "virtual_accelerator"


#: The stand-in this deployment stood up and records, as `_write_config` kwargs:
#: the port it serves, the `standin` target's own gateways selecting it, and the
#: recorder that samples it into the store (deployed by default).
_RECORDS_ITS_STANDIN = {"standin_gateway": ("127.0.0.1", 5074), "standin_port": 5074}


@pytest.mark.parametrize(
    ("config_kwargs", "expected"),
    [
        pytest.param(
            {"control_system_type": "virtual_accelerator"},
            RecordingFacts("virtual_accelerator", False),
            id="the-baseline-virtual-accelerator-stood-nothing-up",
        ),
        # A stand-in deployment before and after `osprey set connector=epics`:
        # the rendered type moves, the machine being recorded does not.
        pytest.param(
            {"control_system_type": "virtual_accelerator", **_RECORDS_ITS_STANDIN},
            RecordingFacts("virtual_accelerator", True),
            id="records-its-standin",
        ),
        pytest.param(
            {"control_system_type": "epics", **_RECORDS_ITS_STANDIN},
            RecordingFacts("epics", True),
            id="records-its-standin-after-set-connector-epics",
        ),
        # localhost and ::1 are loopback the same way an address is.
        pytest.param(
            {**_RECORDS_ITS_STANDIN, "standin_gateway": ("localhost", 5074)},
            RecordingFacts("virtual_accelerator", True),
            id="loopback-spelled-localhost",
        ),
        pytest.param(
            {**_RECORDS_ITS_STANDIN, "standin_gateway": ("::1", 5074)},
            RecordingFacts("virtual_accelerator", True),
            id="loopback-spelled-v6",
        ),
        # The `standin` block's gateways moved to a real machine: the leftover
        # `services:` block loses, and the endpoint is what answers.
        pytest.param(
            {**_RECORDS_ITS_STANDIN, "standin_gateway": ("epics-gateway.example.org", 5074)},
            RecordingFacts("virtual_accelerator", False),
            id="standin-gateways-moved-off-the-host",
        ),
        pytest.param(
            {**_RECORDS_ITS_STANDIN, "standin_gateway": ("127.0.0.1", 5064)},
            RecordingFacts("virtual_accelerator", False),
            id="standin-built-elsewhere-than-the-gateways-dial",
        ),
        # A quoted port in the gateway block, coerced the same way the roster's
        # own label derivation coerces it: one endpoint, never two answers.
        pytest.param(
            {**_RECORDS_ITS_STANDIN, "standin_gateway": ("127.0.0.1", "5074")},
            RecordingFacts("virtual_accelerator", True),
            id="a-quoted-gateway-port-is-still-that-port",
        ),
        pytest.param(
            {**_RECORDS_ITS_STANDIN, "standin_gateway": ("127.0.0.1", "not-a-port")},
            RecordingFacts("virtual_accelerator", False),
            id="a-gateway-port-that-names-no-port",
        ),
        # The two halves of "whose past is in this store", each without the
        # other. Neither alone makes the archive the stand-in's.
        pytest.param(
            {**_RECORDS_ITS_STANDIN, "recorder_deployed": False},
            RecordingFacts("virtual_accelerator", False),
            id="a-standin-this-deployment-never-records",
        ),
        pytest.param(
            {"standin_gateway": ("127.0.0.1", 5074)},
            RecordingFacts("virtual_accelerator", False),
            id="a-recorder-with-no-standin-stood-up",
        ),
        # `live` always means the facility's authored `epics` block, and the
        # recorded endpoint is never read from it — in either direction.
        pytest.param(
            {
                "control_system_type": "epics",
                **_RECORDS_ITS_STANDIN,
                "live_gateway": ("epics-gateway.example.org", 5064),
            },
            RecordingFacts("epics", True),
            id="a-real-live-machine-beside-the-recorded-standin",
        ),
        pytest.param(
            {
                "control_system_type": "epics",
                "live_gateway": ("127.0.0.1", 5074),
                "standin_port": 5074,
            },
            RecordingFacts("epics", False),
            id="an-epics-block-on-the-standins-port-is-not-the-standin",
        ),
        # An SSH tunnel: loopback, and nothing else.
        pytest.param(
            {"control_system_type": "epics", "live_gateway": ("localhost", 5064)},
            RecordingFacts("epics", False),
            id="an-ssh-tunnel-to-a-real-gateway",
        ),
        pytest.param(
            {"control_system_type": "mock"},
            RecordingFacts("mock", False),
            id="the-ordinary-mock-project",
        ),
    ],
)
def test_recording_facts_answer_for_the_machine_this_archive_records(
    tmp_path: Path,
    config_kwargs: dict,
    expected: RecordingFacts,
) -> None:
    """Both facts out of one parse, and the stand-in half decided by the shared
    predicate — whose past is in this store — and then by the `standin` target's
    own derived endpoint, never by the presence of a `services:` block and never
    from the `epics` block. A guard that read the block directly would keep
    claiming a stand-in after the gateways had been moved to hardware; one that
    read `epics` would answer for the facility's real machine instead."""
    config_path = _write_config(tmp_path / "config.yml", **config_kwargs)

    assert read_recording_facts(config_path) == expected


def test_recording_facts_need_a_control_system_block(tmp_path: Path) -> None:
    """No silent defaults on the poll path either: a config with nothing to say
    about its control system cannot say what is being recorded, and the caller
    keeps its last answer rather than acting on the silence."""
    config_path = _write_config(tmp_path / "config.yml")
    config = yaml.safe_load(config_path.read_text())
    del config["control_system"]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(RecorderConfigError, match="control_system"):
        read_recording_facts(config_path)


def _recorder_template_text() -> str:
    """The recorder compose template as written, jinja expressions intact."""
    import osprey

    path = (
        Path(osprey.__file__).parent
        / "templates"
        / "services"
        / "archiver_recorder"
        / "docker-compose.yml.j2"
    )
    return path.read_text(encoding="utf-8")


def test_the_template_mounts_the_live_project_not_a_staged_config_copy() -> None:
    """The enablement poll is only honest if it re-reads the operator's own file.

    Every other service mounts the flattened config.yml the compose generator
    stages into its build dir, and returning this one to that convention is the
    obvious tidy-up for someone who has not read why it deviates. It would also
    be silent: the poll would keep running against a copy rewritten once per
    `osprey build`, so a flip would need a rebuild while the service went on
    logging that it does not — the exact dishonesty this feature removes.

    The mount source is `.` — bind sources resolve against the pinned compose
    project directory, which is the deployment REPO ROOT — and CONFIG_FILE
    names the as-built render inside it, reached through the directory mount so
    a host edit is seen on the next poll.
    """
    template = _recorder_template_text()

    assert "- .:/app/project:ro" in template, (
        "the recorder must mount the deployment repo; a staged copy is "
        "rewritten only by `osprey build`, so the enablement poll would re-read "
        "a stale copy and the documented no-rebuild flip would never arrive"
    )
    assert ":/app/project/config.yml" not in template, (
        "a single-file config.yml mount is the staged-copy pattern returning; "
        "see the mount comment for why this service deviates from the "
        "convention the bluesky bridge follows"
    )
    assert "CONFIG_FILE: /app/project/build/config.yml" in template, (
        "CONFIG_FILE must keep naming the path the repo mount now makes live, "
        "or the service reads nothing"
    )


def test_the_template_mounts_a_directory_so_editor_saves_are_seen() -> None:
    """A single-file bind would pass a truncate-in-place test and fail a real edit.

    Docker resolves a single-file bind to an inode at container start. vim and
    VS Code save by writing a temp file and renaming it over the original,
    which leaves a new inode, and the container then reads the old one forever
    without saying so. The directory mount re-resolves on every read, so the
    reason it is a directory has to survive in the file.
    """
    template = _recorder_template_text()

    assert "inode" in template, (
        "the mount comment must keep the inode rationale: without it the "
        "directory mount reads as an oversight and gets narrowed to the one "
        "file, which silently breaks the flip for editors that save by rename"
    )
