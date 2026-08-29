"""Tests for per-user host port allocation (osprey.deployment.web_terminals.ports)."""

from __future__ import annotations

import pytest

from osprey.deployment.web_terminals.ports import (
    FAMILY_BASE_FIELDS,
    PANEL_ENV_VARS,
    allocate_ports,
    base_ports_from_config,
    default_base_ports,
)
from osprey.port_layout import INDEX_MAX, LAYOUT, SLOTS_BY_NAME, default_port
from osprey.registry.web import FRAMEWORK_WEB_SERVERS

_FAMILIES = tuple(FAMILY_BASE_FIELDS.values())

# One distinct synthetic base per family — spacing 100 mirrors the layout's
# family stride and keeps per-family ranges disjoint for the tests below.
_BASE_PORTS = {family: 8000 + i * 100 for i, family in enumerate(_FAMILIES)}

# A base that is nobody's default, so a port derived from it can only have come
# from the base that was handed in.
_TEST_BASE = 20000


def test_allocate_ports_returns_every_family() -> None:
    """Every derived family (web + one per registry companion) must be present."""
    # Arrange
    base_ports = dict(_BASE_PORTS)

    # Act
    result = allocate_ports(base_ports, index=1)

    # Assert
    assert set(result.keys()) == set(_FAMILIES)


def test_allocate_ports_adds_index_to_base_port() -> None:
    """Each family's allocated port equals its configured base port plus the index."""
    # Arrange
    base_ports = dict(_BASE_PORTS)
    index = 3

    # Act
    result = allocate_ports(base_ports, index)

    # Assert
    assert result == {family: base + index for family, base in _BASE_PORTS.items()}


def test_allocate_ports_index_zero_returns_base_ports_unchanged() -> None:
    """Index 0 (the first user) must map exactly onto the configured base ports."""
    # Arrange
    base_ports = dict(_BASE_PORTS)

    # Act
    result = allocate_ports(base_ports, index=0)

    # Assert
    assert result == _BASE_PORTS


def test_allocate_ports_distinct_indices_do_not_collide() -> None:
    """Two different user indices must never allocate the same port for a family."""
    # Arrange
    base_ports = dict(_BASE_PORTS)

    # Act
    user_a = allocate_ports(base_ports, index=0)
    user_b = allocate_ports(base_ports, index=1)

    # Assert
    for family in _BASE_PORTS:
        assert user_a[family] != user_b[family]


def test_allocate_ports_missing_family_raises_value_error() -> None:
    """A base_ports dict missing a required family must raise a clear ValueError."""
    # Arrange
    incomplete_base_ports = {f: p for f, p in _BASE_PORTS.items() if f != "lattice"}

    # Act / Assert
    with pytest.raises(ValueError, match="lattice"):
        allocate_ports(incomplete_base_ports, index=0)


def test_allocate_ports_accepts_the_last_index_of_the_band() -> None:
    """The band is inclusive: user INDEX_MAX is the hundredth user and allocates."""
    # Arrange
    base_ports = dict(_BASE_PORTS)

    # Act
    result = allocate_ports(base_ports, index=INDEX_MAX)

    # Assert
    assert result == {family: base + INDEX_MAX for family, base in _BASE_PORTS.items()}


def test_allocate_ports_index_past_the_band_is_refused() -> None:
    """A roster larger than a family band would take the next family's ports. The
    refusal names the band and the `<family>_base_port` escape rather than
    silently handing out a port that belongs to another panel."""
    # Arrange
    base_ports = dict(_BASE_PORTS)

    # Act / Assert
    with pytest.raises(ValueError) as excinfo:
        allocate_ports(base_ports, index=INDEX_MAX + 1)
    message = str(excinfo.value)
    assert str(INDEX_MAX) in message
    assert "artifact" in message
    assert "modules.web_terminals.artifact_base_port" in message


def test_allocate_ports_negative_index_is_refused() -> None:
    """A negative index would allocate below the family's band, into whatever the
    layout put there."""
    # Act / Assert
    with pytest.raises(ValueError, match="out of range"):
        allocate_ports(dict(_BASE_PORTS), index=-1)


# ---------------------------------------------------------------------------
# Registry-derived family parity — the "a companion server cannot miss
# multi-user wiring" invariants. These are the guards that turn a forgotten
# port family (the channel-finder crash-loop class) from a runtime collision
# into a red unit test at development time.
# ---------------------------------------------------------------------------


def test_every_registry_server_has_a_port_family() -> None:
    """FAMILY_BASE_FIELDS must contain web plus exactly one family per
    FRAMEWORK_WEB_SERVERS entry — registering a companion server IS the wiring."""
    expected_families = {"web"} | {
        defn.port_family or key for key, defn in FRAMEWORK_WEB_SERVERS.items()
    }
    assert set(FAMILY_BASE_FIELDS.values()) == expected_families
    # One config field per family, named `<family>_base_port`.
    assert FAMILY_BASE_FIELDS == {f"{family}_base_port": family for family in _FAMILIES}
    # Family names must be unique across servers (a duplicate would silently
    # merge two servers onto one port family).
    assert len(expected_families) == len(FRAMEWORK_WEB_SERVERS) + 1


def test_every_family_is_a_per_index_slot_of_the_layout() -> None:
    """No port is authored here or in the registry: every family — web included —
    is a per-index slot of the port layout, so its default comes from the block
    the deployment resolved. A family without a slot has no default at all."""
    for family in _FAMILIES:
        slot = SLOTS_BY_NAME.get(family)
        assert slot is not None, (
            f"port family {family!r} has no slot in osprey.port_layout.LAYOUT, so it "
            "has no default port and a config that omits its base port cannot deploy"
        )
        assert slot.per_index, (
            f"port layout slot {family!r} is not per_index, but the web terminals "
            "allocate one port per user from it"
        )
        assert slot.config_key == f"modules.web_terminals.{family}_base_port"


def test_default_base_ports_derive_from_the_base_handed_in() -> None:
    """Every family's default is its layout offset from the base the CALLER
    resolved — the whole point of the block. A different base moves every
    family by exactly that difference and nothing else."""
    # Act
    at_test_base = default_base_ports(_TEST_BASE)
    at_shifted_base = default_base_ports(_TEST_BASE + 1000)

    # Assert
    assert at_test_base == {
        family: default_port(family, 0, base=_TEST_BASE) for family in _FAMILIES
    }
    assert at_shifted_base == {family: port + 1000 for family, port in at_test_base.items()}


def test_default_base_port_ranges_are_disjoint() -> None:
    """Family bands must not overlap each other for any roster the block admits
    (the layout reserves INDEX_MAX + 1 ports per family)."""
    # Arrange
    stride = INDEX_MAX + 1

    # Act
    ranges = sorted(default_base_ports(_TEST_BASE).values())

    # Assert
    for lower, upper in zip(ranges, ranges[1:], strict=False):
        assert upper - lower >= stride, (
            f"default base ports {lower} and {upper} are closer than the {stride}-port "
            "family stride — a full roster would collide across families"
        )
    # And the bands sit above every singleton slot, so a full roster cannot run
    # back down into the gateway/services tiers either.
    lowest_family_offset = min(SLOTS_BY_NAME[family].offset for family in _FAMILIES)
    singletons = [entry.offset for entry in LAYOUT if not entry.per_index]
    assert all(
        offset < lowest_family_offset or offset >= lowest_family_offset + stride * len(_FAMILIES)
        for offset in singletons
    )


def test_panel_env_vars_match_launcher_derivation() -> None:
    """The env var the compose render exports per family must be exactly the one
    the in-container launcher reads (WebServerDefinition.port_env_var)."""
    for key, defn in FRAMEWORK_WEB_SERVERS.items():
        family = defn.port_family or key
        assert PANEL_ENV_VARS[family] == defn.port_env_var
        assert defn.port_env_var == f"OSPREY_{defn.config_key.upper()}_PORT"


def test_base_ports_from_config_fills_layout_defaults() -> None:
    """A config that predates a companion server (sets only the classic four
    fields) must still resolve every family — new ones from the layout block."""
    # Arrange
    config = {
        "web_base_port": 9091,
        "artifact_base_port": 9291,
        "ariel_base_port": 9391,
        "lattice_base_port": 9491,
    }

    # Act
    base_ports = base_ports_from_config(config, base=_TEST_BASE)

    # Assert
    assert set(base_ports) == set(_FAMILIES)
    assert base_ports["web"] == 9091
    assert base_ports["channel_finder"] == default_port("channel_finder", 0, base=_TEST_BASE)
    # An explicit config field always beats the layout default.
    overridden = base_ports_from_config(
        {**config, "channel_finder_base_port": 21200}, base=_TEST_BASE
    )
    assert overridden["channel_finder"] == 21200


def test_base_ports_from_config_web_takes_the_layout_default() -> None:
    """`web` is the terminal itself, and it now has a layout slot like every other
    family — an empty stanza deploys on the block rather than failing to
    allocate."""
    # Act
    base_ports = base_ports_from_config({}, base=_TEST_BASE)

    # Assert
    assert base_ports == default_base_ports(_TEST_BASE)
    assert base_ports["web"] == default_port("web", 0, base=_TEST_BASE)
    # And it allocates: nothing about an unconfigured stanza is fatal any more.
    assert allocate_ports(base_ports, index=0) == base_ports


def test_base_ports_from_config_ignores_malformed_values() -> None:
    """A non-integer (or boolean) base port is not a port. It falls back to the
    layout default rather than being templated into compose verbatim."""
    # Arrange
    config = {"web_base_port": "9091", "artifact_base_port": True, "ariel_base_port": None}

    # Act
    base_ports = base_ports_from_config(config, base=_TEST_BASE)

    # Assert
    assert base_ports == default_base_ports(_TEST_BASE)


def test_base_ports_from_config_requires_the_base_keyword() -> None:
    """The base is keyword-only and has no default: a caller that cannot name the
    base the deployment resolved must not be able to allocate at all."""
    # Act / Assert
    with pytest.raises(TypeError):
        base_ports_from_config({})  # type: ignore[call-arg]
