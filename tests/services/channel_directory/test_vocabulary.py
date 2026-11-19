"""The control system's words for a channel and its server, resolved per control target."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from osprey.services.channel_directory.vocabulary import (
    EPICS_WORDS,
    NEUTRAL_WORDS,
    VOCABULARY_ATTRIBUTE,
    Vocabulary,
    connector_vocabulary,
    source_label,
    target_vocabulary,
    target_words,
)

_MODULE = "facility_words_connector"


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    connector_vocabulary.cache_clear()


@pytest.fixture()
def facility_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """A facility's connector module, importable by its dotted path."""
    module = ModuleType(_MODULE)
    monkeypatch.setitem(sys.modules, _MODULE, module)
    return module


def _connector(module: ModuleType, name: str, words: Any = None) -> str:
    attributes = {} if words is None else {VOCABULARY_ATTRIBUTE: words}
    setattr(module, name, type(name, (), attributes))
    return f"{_MODULE}.{name}"


def test_epics_names_pvs_and_iocs() -> None:
    words = connector_vocabulary("epics")

    assert words == EPICS_WORDS
    assert (words.channel, words.channels, words.server, words.servers) == (
        "PV",
        "PVs",
        "IOC",
        "IOCs",
    )
    assert (words.directory, words.servers_view) == ("All PVs", "IOCs")


@pytest.mark.parametrize("connector_type", ["virtual_accelerator", "live_standin"])
def test_the_simulator_and_the_stand_in_are_epics_shaped(connector_type: str) -> None:
    assert connector_vocabulary(connector_type) == EPICS_WORDS


def test_tango_names_attributes_and_device_servers() -> None:
    assert connector_vocabulary("tango") == Vocabulary(
        channel="attribute",
        channels="attributes",
        server="device server",
        servers="device servers",
        directory="Attributes",
        servers_view="Device servers",
    )


def test_doocs_names_properties_and_servers() -> None:
    assert connector_vocabulary("doocs") == Vocabulary(
        channel="property",
        channels="properties",
        server="server",
        servers="servers",
        directory="Properties",
        servers_view="Servers",
    )


def test_an_unknown_type_gets_the_neutral_words() -> None:
    assert connector_vocabulary("nosuch") == NEUTRAL_WORDS
    assert (NEUTRAL_WORDS.directory, NEUTRAL_WORDS.servers_view) == ("Channels", "Servers")
    assert (NEUTRAL_WORDS.channel, NEUTRAL_WORDS.server) == ("channel", "server")


def test_a_facility_connector_states_its_own_words(facility_module: ModuleType) -> None:
    stated = {
        "channel": "signal",
        "channels": "signals",
        "server": "front end",
        "servers": "front ends",
        "directory": "All signals",
    }
    connector_type = _connector(facility_module, "SignalConnector", stated)

    assert connector_vocabulary(connector_type) == Vocabulary(
        channel="signal",
        channels="signals",
        server="front end",
        servers="front ends",
        directory="All signals",
        servers_view="Front ends",
    )


def test_a_facility_connector_stating_nothing_gets_the_neutral_words(
    facility_module: ModuleType,
) -> None:
    assert connector_vocabulary(_connector(facility_module, "Plain")) == NEUTRAL_WORDS


@pytest.mark.parametrize(
    "stated",
    [
        "PVs",
        {"channel": "signal", "channels": "signals", "server": "front end"},
        {"channel": "", "channels": "signals", "server": "a", "servers": "as"},
        {"channel": "x", "channels": "xs", "server": "a", "servers": "as", "directory": 3},
    ],
)
def test_a_malformed_statement_gets_the_neutral_words(
    facility_module: ModuleType, stated: Any
) -> None:
    assert connector_vocabulary(_connector(facility_module, "Malformed", stated)) == NEUTRAL_WORDS


def test_a_connector_that_does_not_import_gets_the_neutral_words() -> None:
    assert connector_vocabulary("no_such_facility_package.Connector") == NEUTRAL_WORDS


def test_each_target_gets_the_words_of_the_type_it_selects() -> None:
    control_system = {"type": "tango"}

    assert target_vocabulary(control_system, "live") == connector_vocabulary("tango")
    assert target_vocabulary(control_system, "va") == EPICS_WORDS
    assert target_vocabulary(control_system, "standin") == EPICS_WORDS


def test_an_unresolvable_live_target_gets_the_neutral_words() -> None:
    assert target_vocabulary({"type": "virtual_accelerator"}, "live") == NEUTRAL_WORDS


def test_target_words_names_every_target() -> None:
    words = target_words({"control_system": {"type": "epics"}})

    assert words == {"live": EPICS_WORDS, "va": EPICS_WORDS, "standin": EPICS_WORDS}


def test_a_directory_source_is_labelled_with_the_directory_title() -> None:
    assert source_label("directory", "channelfinder", EPICS_WORDS) == "All PVs"
    assert source_label("directory", "channelfinder") == "Channels"
    assert source_label("put_log", "caputlog", EPICS_WORDS) == "caputlog"


def test_the_served_form_names_every_field() -> None:
    assert EPICS_WORDS.as_json() == {
        "channel": "PV",
        "channels": "PVs",
        "server": "IOC",
        "servers": "IOCs",
        "directory": "All PVs",
        "servers_view": "IOCs",
    }
