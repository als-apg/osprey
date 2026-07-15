"""Tests for per-user host port allocation (osprey.deployment.web_terminals.ports)."""

from __future__ import annotations

import pytest

from osprey.deployment.web_terminals.ports import allocate_ports

_BASE_PORTS = {"web": 8087, "artifact": 8086, "ariel": 8085, "lattice": 8097}


def test_allocate_ports_returns_all_four_families() -> None:
    """Every one of the four canonical families must be present in the result."""
    # Arrange
    base_ports = dict(_BASE_PORTS)

    # Act
    result = allocate_ports(base_ports, index=1)

    # Assert
    assert set(result.keys()) == {"web", "artifact", "ariel", "lattice"}


def test_allocate_ports_adds_index_to_base_port() -> None:
    """Each family's allocated port equals its configured base port plus the index."""
    # Arrange
    base_ports = dict(_BASE_PORTS)
    index = 3

    # Act
    result = allocate_ports(base_ports, index)

    # Assert
    assert result == {
        "web": 8090,
        "artifact": 8089,
        "ariel": 8088,
        "lattice": 8100,
    }


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
    incomplete_base_ports = {"web": 8087, "artifact": 8086, "ariel": 8085}

    # Act / Assert
    with pytest.raises(ValueError, match="lattice"):
        allocate_ports(incomplete_base_ports, index=0)
