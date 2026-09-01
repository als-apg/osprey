"""Tests for the identity-headers module's constant-time comparator.

:func:`~osprey.services.auth_sidecar.identity_headers.same_value` is the one
equality check every identity decision in the sidecar goes through, so what is
pinned here is small and load-bearing: equal values match, unequal values do
not, and a non-ASCII value on either side is *compared* rather than raising —
the exact failure mode that motivated encoding to UTF-8 bytes before handing
the pair to :func:`secrets.compare_digest`.
"""

from __future__ import annotations

import pytest

from osprey.services.auth_sidecar.identity_headers import same_value


def test_equal_values_match() -> None:
    assert same_value("alice", "alice") is True


def test_unequal_values_do_not_match() -> None:
    assert same_value("alice", "alicia") is False


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("jörg@example.org", "jörg@example.org", True),
        ("jörg@example.org", "jorg@example.org", False),
        ("ascii", "jörg", False),
    ],
)
def test_non_ascii_values_compare_instead_of_raising(left: str, right: str, expected: bool) -> None:
    assert same_value(left, right) is expected


def test_empty_against_nonempty_does_not_match() -> None:
    assert same_value("", "alice") is False
    assert same_value("", "") is True


def test_public_name_is_importable() -> None:
    from osprey.services.auth_sidecar.identity_headers import same_value as imported

    assert imported is same_value
