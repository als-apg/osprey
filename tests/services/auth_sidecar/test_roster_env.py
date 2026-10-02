"""The roster username to per-user env-var suffix mapping and its collision detector."""

from pathlib import Path

from osprey.services.auth_sidecar.roster_env import (
    PW_HASH_VAR_PREFIX,
    env_var_suffix,
    env_var_suffix_collisions,
)


def test_env_var_suffix_uppercases_and_maps_dashes_to_underscores() -> None:
    """The one definition of how a username keys its per-user env vars."""
    # Act / Assert
    assert env_var_suffix("alice") == "ALICE"
    assert env_var_suffix("alice-b") == "ALICE_B"
    assert env_var_suffix("Alice-B-C") == "ALICE_B_C"


def test_env_var_suffix_leaves_already_conforming_names_untouched() -> None:
    """Idempotent on its own output — an already-uppercase, underscored name is
    returned unchanged, so re-keying an existing entry can't drift."""
    # Arrange
    once = env_var_suffix("alice-b")

    # Act / Assert
    assert env_var_suffix(once) == once


def test_env_var_suffix_is_total_and_does_not_validate_charset() -> None:
    """Charset enforcement belongs to the preflight raise and lint, not here —
    this helper maps whatever it is given rather than raising."""
    # Act / Assert
    assert env_var_suffix("") == ""
    assert env_var_suffix("alice.b") == "ALICE.B"


def test_env_var_suffix_collisions_reports_names_sharing_one_suffix() -> None:
    """`alice-b` and `alice_b` both key OSPREY_AUTH_PW_HASH_ALICE_B — without
    this check one user's password would open the other's terminal."""
    # Act
    result = env_var_suffix_collisions(["alice-b", "alice_b", "carol"])

    # Assert
    assert result == {"ALICE_B": ["alice-b", "alice_b"]}


def test_env_var_suffix_collisions_empty_for_an_unambiguous_roster() -> None:
    """A roster whose usernames map one-to-one reports nothing."""
    # Act / Assert
    assert env_var_suffix_collisions(["alice", "bob", "carol"]) == {}
    assert env_var_suffix_collisions([]) == {}


def test_env_var_suffix_collisions_ignores_case_only_and_repeated_names() -> None:
    """A verbatim-repeated name is one user listed twice (a duplicate-name error
    reported separately), not two users sharing a credential — while names
    differing only in case really do collide onto one suffix."""
    # Act / Assert
    assert env_var_suffix_collisions(["alice", "alice"]) == {}
    assert env_var_suffix_collisions(["alice", "Alice"]) == {"ALICE": ["Alice", "alice"]}


def test_env_var_suffix_collisions_output_is_sorted_for_stable_messages() -> None:
    """Suffix keys and the names under each are sorted, so a lint or preflight
    message built from this reads the same across runs."""
    # Act
    result = env_var_suffix_collisions(["zed_x", "b-1", "zed-x", "b_1"])

    # Assert
    assert list(result) == ["B_1", "ZED_X"]
    assert result == {"B_1": ["b-1", "b_1"], "ZED_X": ["zed-x", "zed_x"]}


def test_env_var_suffix_collisions_ignores_non_string_entries() -> None:
    """Drop-don't-raise, like the roster readers: a malformed roster entry
    that slipped through can't crash the collision check."""
    # Act
    result = env_var_suffix_collisions(["alice-b", None, 7, "alice_b"])  # type: ignore[list-item]

    # Assert
    assert result == {"ALICE_B": ["alice-b", "alice_b"]}


def test_pw_hash_prefix_is_spelled_once_in_the_source_tree() -> None:
    """The variable name is the contract between the credential writer, the
    sidecar and lint, so the stored-hash stem is defined in one place."""
    # Arrange
    src_root = Path(__file__).resolve().parents[3] / "src" / "osprey"
    quoted = ('"OSPREY_AUTH_PW_HASH_"', "'OSPREY_AUTH_PW_HASH_'")

    # Act
    spelled_in = sorted(
        path.relative_to(src_root).as_posix()
        for path in src_root.rglob("*.py")
        if "__pycache__" not in path.parts
        and any(literal in path.read_text(encoding="utf-8") for literal in quoted)
    )

    # Assert
    assert spelled_in == ["services/auth_sidecar/roster_env.py"]
    assert PW_HASH_VAR_PREFIX == "OSPREY_AUTH_PW_HASH_"
    assert f"{PW_HASH_VAR_PREFIX}{env_var_suffix('alice-b')}" == "OSPREY_AUTH_PW_HASH_ALICE_B"
