"""The archive's facts: its configuration as deployed, the targets it serves, its extent."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import pytest

from osprey.services.pv_directory import archive
from osprey_connectors.archiver import ArchiverConnector

#: The deployment's own store as a build renders it: an address, a login, the knobs.
OWN_STORE = {
    "control_system": {"type": "virtual_accelerator"},
    "services": {"virtual_accelerator": {"port": 5064}},
    "archiver": {
        "type": "mongodb_archiver",
        "mongodb_archiver": {
            "host": "localhost",
            "port": 27801,
            "name": "osprey_archiver",
            "collection": "pv_history",
            "auth": {
                "source": "admin",
                "username": "osprey",
                "password_env": "MONGO_ROOT_PASSWORD",
            },
            "timeout_s": 5,
        },
    },
    "va_archiver": {
        "retention_days": 30,
        "hot_span_hours": 48,
        "hot_cadence_sec": 10,
        "tail_cadence_sec": 60,
        "recorder_cadence_sec": 10,
        "recorder_tail_cadence_sec": 60,
    },
}

APPLIANCE = {
    "control_system": {"type": "epics"},
    "archiver": {"type": "epics_archiver", "settings": {"url": "https://archiver.example"}},
}


def test_the_own_stores_configuration_is_its_type_and_shape_never_a_login() -> None:
    shown = archive.deployed_configuration(OWN_STORE, "va")

    assert shown == {
        "archiver": {"type": "mongodb_archiver"},
        "va_archiver": {
            "retention_days": 30,
            "hot_span_hours": 48,
            "hot_cadence_sec": 10,
            "tail_cadence_sec": 60,
            "recorder_cadence_sec": 10,
        },
    }
    text = json.dumps(shown)
    for secret in ("MONGO_ROOT_PASSWORD", "username", "localhost", "27801", "auth", "password"):
        assert secret not in text


def test_a_facility_archive_shows_its_type_only() -> None:
    shown = archive.deployed_configuration(APPLIANCE, "live")

    assert shown == {"archiver": {"type": "epics_archiver"}}
    assert "archiver.example" not in json.dumps(shown)


def test_the_own_store_on_another_target_shows_no_knobs() -> None:
    assert archive.deployed_configuration(OWN_STORE, "live") == {
        "archiver": {"type": "mongodb_archiver"}
    }
    assert archive.own_store(OWN_STORE, "va") is True
    assert archive.own_store(APPLIANCE, "live") is False


def test_the_newest_sample_bound_follows_the_recorders_cadence() -> None:
    slow = {**OWN_STORE, "va_archiver": {**OWN_STORE["va_archiver"], "recorder_cadence_sec": 30}}

    assert archive.freshness_bound_s(OWN_STORE) == 60
    assert archive.freshness_bound_s(slow) == 90


def test_each_configured_target_says_whether_it_has_an_archive() -> None:
    assert archive.served_targets(OWN_STORE) == [{"target": "va", "archive": True}]
    assert archive.served_targets(APPLIANCE) == [{"target": "live", "archive": True}]
    no_simulator = {**OWN_STORE, "services": {}}
    assert archive.served_targets({**no_simulator, "control_system": {"type": "epics"}}) == [
        {"target": "live", "archive": False}
    ]


class _Answers:
    def __init__(self, answer: Any = None, failure: BaseException | None = None) -> None:
        self._answer = answer
        self._failure = failure

    async def store_extent(self) -> Any:
        if self._failure is not None:
            raise self._failure
        return self._answer


class _Extent:
    channels = ["SR:A", "SR:B"]
    oldest = datetime(2026, 10, 13)
    newest = datetime(2026, 11, 12)
    reason = None


async def test_an_extent_is_read_as_the_connector_answers_it() -> None:
    extent = await archive.read_extent(_Answers(_Extent()), 1.0)

    assert extent == archive.Extent(
        channels=frozenset({"SR:A", "SR:B"}),
        oldest=datetime(2026, 10, 13),
        newest=datetime(2026, 11, 12),
    )


async def test_a_connector_without_store_extent_answers_the_upgrade_reason() -> None:
    extent = await archive.read_extent(object(), 1.0)

    assert extent == archive.Extent(reason=archive.NO_EXTENT_REASON)


async def test_a_failing_or_slow_extent_read_answers_its_reason() -> None:
    class Slow:
        async def store_extent(self) -> Any:
            await asyncio.sleep(5)

    failed = await archive.read_extent(_Answers(failure=RuntimeError("down")), 1.0)
    late = await archive.read_extent(Slow(), 0.05)

    assert (failed.channels, failed.reason) == (None, "the archive failed: RuntimeError")
    assert (late.channels, late.reason) == (None, "no answer within 0.05 s")


async def test_a_failed_extent_read_never_serves_the_exception_text() -> None:
    failure = ConnectionError("cannot reach mongodb://reader:secret@archive.example:27017")

    extent = await archive.read_extent(_Answers(failure=failure), 1.0)

    assert extent.reason == "the archive failed: ConnectionError"
    assert "mongodb://" not in extent.reason and "secret" not in extent.reason


@pytest.mark.skipif(
    not hasattr(ArchiverConnector, "store_extent"),
    reason="installed osprey-connectors predates ArchiverConnector.store_extent",
)
async def test_the_appliance_cannot_say_what_it_holds() -> None:
    from osprey_connectors.archiver._epics_archiver_connector import EPICSArchiverConnector

    connector = EPICSArchiverConnector()
    await connector.connect({"url": "https://archiver.example"})

    extent = await archive.read_extent(connector, 1.0)

    assert extent.channels is None
    assert extent.reason == "this archiver does not report what it holds"
