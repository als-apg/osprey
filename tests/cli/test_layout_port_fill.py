"""Tests for :func:`~osprey.cli.build_profile_ports.layout_port_fill`.

The helper answers one question per layout row: does this profile deploy the
service, and did it leave the port unspelled? Only then is a number emitted,
and the number is the row's port at the base the deployment resolved.

Every expected port is read back out of :func:`~osprey.port_layout.layout_ports`
rather than written down, so a layout that moved a slot fails here loudly
instead of pinning the old number in a second place.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.cli.build_profile_ports import layout_port_fill
from osprey.port_layout import DEFAULT_PORT_BASE, layout_ports

#: A moved deployment: the second base a host would hand a second stack.
_MOVED_BASE = DEFAULT_PORT_BASE + 20000

#: The ``services:`` blocks the control-assistant render carries, written the
#: way an explicit profile writes them — dotted keys, and no ports at all.
_CONTROL_ASSISTANT: dict[str, Any] = {
    "services.postgresql.path": "./services/postgresql",
    "services.postgresql.database_name": "ariel",
    "services.postgresql.username": "ariel",
    "services.openobserve.path": "./services/openobserve",
    "services.openobserve.retention_days": 14,
    "services.qmd.path": "./services/qmd",
    "services.qmd.interval": 30,
    "services.graphdb.path": "./services/graphdb",
    "services.graphdb.image": "neo4j:5.26-community",
    "services.graphdb.ttl_path": "./data/demo_machine.ttl",
}

#: Every port the blocks above leave for the layout to fill, as
#: ``config key -> layout slot``.
_CONTROL_ASSISTANT_FILLS = {
    "services.postgresql.port_host": "postgres",
    "services.openobserve.port": "openobserve",
    "services.qmd.port": "qmd",
    "services.graphdb.port_host": "graphdb_bolt",
    "services.graphdb.http_port_host": "graphdb_http",
}


def _expected(base: int, fills: dict[str, str]) -> dict[str, int]:
    """The ports *fills* names, read off the layout at *base*."""
    ports = layout_ports(base)
    return {key: ports[slot] for key, slot in fills.items()}


@pytest.mark.parametrize("base", [DEFAULT_PORT_BASE, _MOVED_BASE])
def test_a_control_assistant_shaped_config_gets_exactly_its_own_ports(base: int) -> None:
    """Each deployed block is filled; nothing else in the layout is."""
    filled = layout_port_fill(_CONTROL_ASSISTANT, base)

    assert filled == _expected(base, _CONTROL_ASSISTANT_FILLS)


def test_a_moved_base_moves_every_filled_port() -> None:
    """The layout's one rule: the base is the caller's, never the module default."""
    at_default = layout_port_fill(_CONTROL_ASSISTANT, DEFAULT_PORT_BASE)
    at_moved = layout_port_fill(_CONTROL_ASSISTANT, _MOVED_BASE)

    assert set(at_default) == set(at_moved)
    assert all(
        at_moved[key] - at_default[key] == _MOVED_BASE - DEFAULT_PORT_BASE for key in at_moved
    )


def test_an_external_store_still_gets_both_of_its_ports() -> None:
    """A ``graphdb`` block that names only a ``uri`` is a block all the same.

    Whether the store is this deployment's or someone else's is decided by
    ``deployed_services``, not by this helper: the block is present, so the
    two ports it can carry are derived.
    """
    filled = layout_port_fill(
        {"services.graphdb.uri": "bolt://graph.example.org:7687"}, DEFAULT_PORT_BASE
    )

    assert filled == _expected(
        DEFAULT_PORT_BASE,
        {
            "services.graphdb.port_host": "graphdb_bolt",
            "services.graphdb.http_port_host": "graphdb_http",
        },
    )


def test_a_spelled_port_is_left_alone() -> None:
    """Skip if spelled — the fill never competes with a number the author wrote."""
    moved = layout_ports(DEFAULT_PORT_BASE)["qmd"] + 4321
    filled = layout_port_fill({**_CONTROL_ASSISTANT, "services.qmd.port": moved}, DEFAULT_PORT_BASE)

    assert "services.qmd.port" not in filled
    assert filled == _expected(
        DEFAULT_PORT_BASE,
        {key: slot for key, slot in _CONTROL_ASSISTANT_FILLS.items() if key != "services.qmd.port"},
    )


def test_a_port_spelled_as_a_nested_mapping_is_left_alone() -> None:
    """Both legal spellings reach the same leaf, so both count as spelled."""
    filled = layout_port_fill(
        {**_CONTROL_ASSISTANT, "services": {"qmd": {"port": 12345}}}, DEFAULT_PORT_BASE
    )

    assert "services.qmd.port" not in filled


def test_a_service_whose_keys_were_deleted_is_skipped() -> None:
    """Deleting a store's keys takes the service away, and its ports with it.

    This is how a deployment says it runs no graph store: the
    ``services.graphdb.*`` keys are gone from the profile, and so is the
    ``graphdb`` entry in ``deployed_services``. There is no null-valued block to
    read — a whole-block ``services.graphdb:`` override is refused at profile
    validation — so absence is the whole spelling, and every other service's
    port must still be filled around the gap.
    """
    kept = {key: value for key, value in _CONTROL_ASSISTANT.items() if "graphdb" not in key}
    filled = layout_port_fill(kept, DEFAULT_PORT_BASE)

    assert "services.graphdb.port_host" not in filled
    assert "services.graphdb.http_port_host" not in filled
    assert filled == _expected(
        DEFAULT_PORT_BASE,
        {key: slot for key, slot in _CONTROL_ASSISTANT_FILLS.items() if "graphdb" not in key},
    )


def test_a_service_the_profile_never_mentions_is_not_invented() -> None:
    """Absent is not the same as unspelled: no block, no port."""
    filled = layout_port_fill(_CONTROL_ASSISTANT, DEFAULT_PORT_BASE)

    assert not [key for key in filled if key.startswith("services.bluesky")]
    assert "services.event_dispatcher.port" not in filled


def test_a_config_with_no_services_block_fills_nothing() -> None:
    """Nothing to deploy, nothing to derive."""
    assert layout_port_fill({"deployment.port_base": DEFAULT_PORT_BASE}, DEFAULT_PORT_BASE) == {}


def test_bluesky_gets_lane_ones_port_not_lane_twos() -> None:
    """Two rows name ``services.bluesky.port``; the first one is the one that moves it.

    Lane 2's bridge port is derived from lane 1's rather than authored, so its
    row points at lane 1's key. Filling that key from lane 2's offset would
    hand the profile the wrong number.
    """
    ports = layout_ports(DEFAULT_PORT_BASE)
    filled = layout_port_fill({"services.bluesky.path": "./services/bluesky"}, DEFAULT_PORT_BASE)

    assert filled["services.bluesky.port"] == ports["bluesky"]
    assert filled["services.bluesky.port"] != ports["bluesky_second_lane"]
