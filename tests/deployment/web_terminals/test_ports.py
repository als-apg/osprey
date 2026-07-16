"""Tests for per-user host port allocation (osprey.deployment.web_terminals.ports)."""

from __future__ import annotations

import pytest

from osprey.deployment.web_terminals.ports import allocate_ports, normalize_users

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


def test_normalize_users_bare_strings_indexed_by_position() -> None:
    """A legacy bare-string roster gets its raw list position as its index."""
    # Arrange
    users_raw = ["alice", "bob"]

    # Act
    result = normalize_users(users_raw)

    # Assert
    assert result == [
        {"name": "alice", "index": 0},
        {"name": "bob", "index": 1},
    ]


def test_normalize_users_explicit_entries_pass_through_unchanged() -> None:
    """Already-explicit object entries keep their own index, not their position."""
    # Arrange
    users_raw = [{"name": "alice", "index": 5}]

    # Act
    result = normalize_users(users_raw)

    # Assert
    assert result == [{"name": "alice", "index": 5}]


def test_normalize_users_mixed_bare_and_explicit_entries() -> None:
    """Bare entries keep raw position; object entries keep their explicit index."""
    # Arrange
    users_raw = ["alice", {"name": "bob", "index": 7}, "carol"]

    # Act
    result = normalize_users(users_raw)

    # Assert
    assert result == [
        {"name": "alice", "index": 0},
        {"name": "bob", "index": 7},
        {"name": "carol", "index": 2},
    ]


def test_normalize_users_is_idempotent() -> None:
    """Normalizing an already-normalized list must be a no-op."""
    # Arrange
    users_raw = ["alice", {"name": "bob", "index": 7}, "carol"]

    # Act
    once = normalize_users(users_raw)
    twice = normalize_users(once)

    # Assert
    assert once == twice


def test_normalize_users_drops_malformed_entries() -> None:
    """Non-string entries and dicts missing a str name or int index are dropped."""
    # Arrange
    users_raw = [
        "alice",
        123,
        {"name": "bob"},  # missing index
        {"index": 2},  # missing name
        {"name": 4, "index": 1},  # name not a str
        {"name": "carol", "index": "1"},  # index not an int
        {"name": "dave", "index": True},  # bool is not a valid index (config typo)
        None,
        [],
    ]

    # Act
    result = normalize_users(users_raw)

    # Assert
    assert result == [{"name": "alice", "index": 0}]


def test_normalize_users_empty_list_returns_empty_list() -> None:
    """An empty users list normalizes to an empty list."""
    # Act / Assert
    assert normalize_users([]) == []


def test_normalize_users_non_list_input_returns_empty_list() -> None:
    """Anything that isn't a list (including None) normalizes to an empty list."""
    # Act / Assert
    assert normalize_users(None) == []
    assert normalize_users({"name": "alice", "index": 0}) == []
    assert normalize_users("alice") == []


def test_normalize_users_does_not_mutate_input_entries() -> None:
    """Normalization must return new dicts, never the original entry by reference."""
    # Arrange
    original_entry = {"name": "alice", "index": 5}
    users_raw = [original_entry]

    # Act
    result = normalize_users(users_raw)
    result[0]["index"] = 99

    # Assert
    assert original_entry == {"name": "alice", "index": 5}
