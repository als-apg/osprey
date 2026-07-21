"""Tests for per-user host port allocation (osprey.deployment.web_terminals.ports)."""

from __future__ import annotations

import pytest

from osprey.deployment.web_terminals.ports import (
    DEFAULT_BASE_PORTS,
    FAMILY_BASE_FIELDS,
    PANEL_ENV_VARS,
    allocate_ports,
    base_ports_from_config,
)
from osprey.registry.web import FRAMEWORK_WEB_SERVERS

_FAMILIES = tuple(FAMILY_BASE_FIELDS.values())

# One distinct synthetic base per family — spacing 100 mirrors the documented
# convention and keeps per-family ranges disjoint for the tests below.
_BASE_PORTS = {family: 8000 + i * 100 for i, family in enumerate(_FAMILIES)}


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


def test_every_registry_server_declares_a_default_base_port() -> None:
    """Every companion server must declare multi_user_base_port — that default is
    what keeps a facility config written before the server existed deploying
    unchanged (and what guarantees the family allocates at all)."""
    for key, defn in FRAMEWORK_WEB_SERVERS.items():
        assert isinstance(defn.multi_user_base_port, int), (
            f"FRAMEWORK_WEB_SERVERS[{key!r}] must declare multi_user_base_port — "
            "without it the server has no per-user port family default and a "
            "multi-user deploy collides across users (shared host netns)"
        )
        family = defn.port_family or key
        assert DEFAULT_BASE_PORTS[family] == defn.multi_user_base_port


def test_registry_default_base_port_ranges_are_disjoint() -> None:
    """Default families must not overlap each other for any plausible roster
    (up to the ×100 spacing the convention reserves per family)."""
    ranges = sorted(DEFAULT_BASE_PORTS.values())
    for lower, upper in zip(ranges, ranges[1:], strict=False):
        assert upper - lower >= 100, (
            f"default base ports {lower} and {upper} are closer than the 100-port "
            "family spacing — a large roster would collide across families"
        )


def test_panel_env_vars_match_launcher_derivation() -> None:
    """The env var the compose render exports per family must be exactly the one
    the in-container launcher reads (WebServerDefinition.port_env_var)."""
    for key, defn in FRAMEWORK_WEB_SERVERS.items():
        family = defn.port_family or key
        assert PANEL_ENV_VARS[family] == defn.port_env_var
        assert defn.port_env_var == f"OSPREY_{defn.config_key.upper()}_PORT"


def test_base_ports_from_config_fills_registry_defaults() -> None:
    """A config that predates a companion server (sets only the classic four
    fields) must still resolve every family — new ones from registry defaults."""
    config = {
        "web_base_port": 9091,
        "artifact_base_port": 9291,
        "ariel_base_port": 9391,
        "lattice_base_port": 9491,
    }
    base_ports = base_ports_from_config(config)
    assert set(base_ports) == set(_FAMILIES)
    assert base_ports["web"] == 9091
    assert base_ports["channel_finder"] == DEFAULT_BASE_PORTS["channel_finder"]
    # An explicit config field always beats the registry default.
    overridden = base_ports_from_config({**config, "channel_finder_base_port": 21200})
    assert overridden["channel_finder"] == 21200


def test_base_ports_from_config_web_has_no_default() -> None:
    """`web` is the terminal itself, not a registry companion — a config without
    web_base_port must NOT invent one (allocate_ports then fails loudly)."""
    base_ports = base_ports_from_config({})
    assert "web" not in base_ports
    with pytest.raises(ValueError, match="web"):
        allocate_ports(base_ports, index=0)
