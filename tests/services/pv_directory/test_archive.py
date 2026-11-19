"""The archive's facts: its configuration as deployed, the targets it serves, its extent."""

from __future__ import annotations

import json

from osprey.services.pv_directory import archive
from osprey_connectors.service_answers import BackendSpec, Unavailable

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
        "hot_span_h": 48,
        "hot_cadence_s": 10,
        "tail_cadence_s": 60,
        "recorder_cadence_s": 10,
        "recorder_tail_cadence_s": 60,
    },
}

APPLIANCE = {
    "control_system": {"type": "epics"},
    "archiver": {"type": "epics_archiver", "settings": {"url": "https://archiver.example"}},
}


def test_the_own_stores_configuration_is_its_type_and_shape_never_a_login() -> None:
    """The served knob names are the directory API's own, kept as served."""
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
    slow = {**OWN_STORE, "va_archiver": {**OWN_STORE["va_archiver"], "recorder_cadence_s": 30}}

    assert archive.freshness_bound_s(OWN_STORE) == 60
    assert archive.freshness_bound_s(slow) == 90


def test_each_configured_target_says_whether_it_has_an_archive() -> None:
    assert archive.served_targets(OWN_STORE) == [{"target": "va", "archive": True}]
    assert archive.served_targets(APPLIANCE) == [{"target": "live", "archive": True}]
    no_simulator = {**OWN_STORE, "services": {}}
    assert archive.served_targets({**no_simulator, "control_system": {"type": "epics"}}) == [
        {"target": "live", "archive": False}
    ]


async def test_the_appliance_cannot_say_what_it_holds() -> None:
    """The default extent: unsupported, in the words the archive view serves."""
    from osprey_connectors.archiver._epics_archiver_connector import EPICSArchiverConnector

    spec = BackendSpec(
        kind="archive",
        type="epics_archiver",
        target="live",
        settings={"url": "https://archiver.example"},
        rehearsal=False,
    )
    extent = await EPICSArchiverConnector(spec).extent(timeout_s=1.0)

    assert extent == Unavailable(
        cause="unsupported", reason="this archiver does not report what it holds"
    )
