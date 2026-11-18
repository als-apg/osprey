"""The one rule for an IOC's liveness and health PVs: selectors, matcher, tie-break."""

from __future__ import annotations

import pytest

pytest.importorskip("osprey_connectors.simulation.view")

from osprey.services.channel_directory.health_selectors import (
    SIMULATOR_DIRECTORY,
    HealthMatch,
    HealthSelector,
    choose,
    health_selectors,
    match_rows,
)
from osprey_connectors.simulation.view import (
    ENGINEER,
    HEARTBEAT,
    SERVED_SIGNAL_PROPERTY,
    UPTIME,
)

DEFAULT_SUFFIXES = {"heartbeat": HEARTBEAT, "uptime": UPTIME}


def test_default_suffixes_select_by_name_off_a_simulator_directory() -> None:
    selectors = health_selectors("channelfinder", DEFAULT_SUFFIXES)

    assert selectors == (
        HealthSelector(role="heartbeat", suffix=HEARTBEAT),
        HealthSelector(role="uptime", suffix=UPTIME),
    )
    assert [selector.query() for selector in selectors] == [
        {"name": f"*{HEARTBEAT}"},
        {"name": f"*{UPTIME}"},
    ]


def test_default_suffixes_select_by_served_signal_on_a_simulator_directory() -> None:
    selectors = health_selectors(SIMULATOR_DIRECTORY, DEFAULT_SUFFIXES)

    assert [selector.query() for selector in selectors] == [
        {"properties": {SERVED_SIGNAL_PROPERTY: HEARTBEAT}},
        {"properties": {SERVED_SIGNAL_PROPERTY: UPTIME}},
    ]


def test_a_custom_suffix_on_a_simulator_directory_still_selects_by_signal() -> None:
    selectors = health_selectors(SIMULATOR_DIRECTORY, {"heartbeat": ":ALIVE", "engineer": ":WHO"})

    assert selectors == (
        HealthSelector(role="heartbeat", signal=HEARTBEAT),
        HealthSelector(role="engineer", signal=ENGINEER),
    )
    rows = [{"name": "ca:SIM:SR:HEARTBEAT", "server": "sim-SR"}]
    assert match_rows(selectors[0], rows) == [
        HealthMatch(server="sim-SR", role="heartbeat", name="ca:SIM:SR:HEARTBEAT")
    ]


def test_a_custom_suffix_off_a_simulator_directory_selects_by_that_suffix() -> None:
    (selector,) = health_selectors("channelfinder", {"heartbeat": ":ALIVE"})

    assert selector.query() == {"name": "*:ALIVE"}


def test_a_role_without_a_served_signal_selects_by_suffix_even_on_a_simulator() -> None:
    (selector,) = health_selectors(SIMULATOR_DIRECTORY, {"temperature": ":TEMP"})

    assert selector == HealthSelector(role="temperature", suffix=":TEMP")


def test_the_matcher_assigns_role_and_server_and_drops_a_name_the_suffix_does_not_end() -> None:
    selector = HealthSelector(role="heartbeat", suffix="HEARTBEAT")
    rows = [
        {"name": "SR:IOC-A:HEARTBEAT", "server": "ioc-a"},
        # A case-insensitive name glob may return this; the suffix itself is exact.
        {"name": "SR:IOC-B:heartbeat", "server": "ioc-b"},
        {"name": "SR:IOC-C:HEARTBEAT", "server": None},
        {"name": "SR:IOC-D:HEARTBEAT"},
    ]

    assert match_rows(selector, rows) == [
        HealthMatch(server="ioc-a", role="heartbeat", name="SR:IOC-A:HEARTBEAT"),
        HealthMatch(server=None, role="heartbeat", name="SR:IOC-C:HEARTBEAT"),
        HealthMatch(server=None, role="heartbeat", name="SR:IOC-D:HEARTBEAT"),
    ]


def test_a_signal_selector_admits_every_row_its_search_returned() -> None:
    selector = HealthSelector(role="uptime", signal=UPTIME)

    assert match_rows(selector, [{"name": "anything", "server": "sim-X"}]) == [
        HealthMatch(server="sim-X", role="uptime", name="anything")
    ]


def test_a_selector_names_exactly_one_of_signal_and_suffix() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        HealthSelector(role="heartbeat")
    with pytest.raises(ValueError, match="exactly one"):
        HealthSelector(role="heartbeat", signal=HEARTBEAT, suffix=HEARTBEAT)


def test_one_match_per_server_and_role_is_not_ambiguous() -> None:
    chosen = choose(
        [
            HealthMatch(server="ioc-a", role="heartbeat", name="A:HEARTBEAT"),
            HealthMatch(server="ioc-a", role="uptime", name="A:UPTIME"),
            HealthMatch(server="ioc-b", role="heartbeat", name="B:HEARTBEAT"),
        ]
    )

    assert {key: (pick.name, pick.ambiguous) for key, pick in chosen.items()} == {
        ("ioc-a", "heartbeat"): ("A:HEARTBEAT", False),
        ("ioc-a", "uptime"): ("A:UPTIME", False),
        ("ioc-b", "heartbeat"): ("B:HEARTBEAT", False),
    }


def test_an_ambiguous_match_takes_the_shortest_name_then_the_first_by_name() -> None:
    chosen = choose(
        [
            HealthMatch(server="ioc-a", role="heartbeat", name="SR:IOC-A:EXTRA:HEARTBEAT"),
            HealthMatch(server="ioc-a", role="heartbeat", name="SR:IOC-B:HEARTBEAT"),
            HealthMatch(server="ioc-a", role="heartbeat", name="SR:IOC-A:HEARTBEAT"),
            HealthMatch(server="ioc-a", role="heartbeat", name="SR:IOC-A:HEARTBEAT"),
        ]
    )

    pick = chosen["ioc-a", "heartbeat"]
    assert pick.name == "SR:IOC-A:HEARTBEAT"
    assert pick.ambiguous is True
    assert pick.candidates == (
        "SR:IOC-A:HEARTBEAT",
        "SR:IOC-B:HEARTBEAT",
        "SR:IOC-A:EXTRA:HEARTBEAT",
    )


def test_the_same_name_twice_is_not_ambiguous() -> None:
    match = HealthMatch(server="ioc-a", role="heartbeat", name="A:HEARTBEAT")

    pick = choose([match, match])["ioc-a", "heartbeat"]

    assert (pick.name, pick.ambiguous, pick.candidates) == ("A:HEARTBEAT", False, ("A:HEARTBEAT",))


def test_a_match_without_a_server_is_not_chosen() -> None:
    assert choose([HealthMatch(server=None, role="heartbeat", name="A:HEARTBEAT")]) == {}
