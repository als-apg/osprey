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

from osprey.services.auth_sidecar.identity_headers import (
    CASE_INSENSITIVE_CLAIMS,
    same_identity,
    same_value,
)


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


# --- same_identity: the claim decides whether case matters -------------------


def test_only_email_is_matched_without_regard_to_case() -> None:
    """The list is exactly ``email``: the one claim that names a mailbox."""
    assert CASE_INSENSITIVE_CLAIMS == frozenset({"email"})


@pytest.mark.parametrize(
    ("asserted", "configured"),
    [
        ("THellert@lbl.gov", "thellert@lbl.gov"),
        ("thellert@lbl.gov", "THellert@lbl.gov"),
        ("Tom_Scarvie@LBL.GOV", "tom_scarvie@lbl.gov"),
        ("thellert@lbl.gov", "thellert@lbl.gov"),
    ],
)
def test_an_email_claim_matches_the_same_mailbox_in_any_case(
    asserted: str, configured: str
) -> None:
    assert same_identity(asserted, configured, claim="email") is True


def test_an_email_claim_still_refuses_a_different_mailbox() -> None:
    assert same_identity("thellert@lbl.gov", "thellert@lbl.org", claim="email") is False
    assert same_identity("t.hellert@lbl.gov", "thellert@lbl.gov", claim="email") is False


@pytest.mark.parametrize("claim", ["sub", "preferred_username", "uid"])
def test_every_other_claim_stays_byte_exact(claim: str) -> None:
    """``sub`` is a case-sensitive opaque identifier by specification, and no
    other claim is on the list: a case difference is a different identity."""
    assert same_identity("idp|Alice", "idp|alice", claim=claim) is False
    assert same_identity("idp|alice", "idp|alice", claim=claim) is True


def test_same_identity_compares_non_ascii_instead_of_raising() -> None:
    assert same_identity("JÖRG@example.org", "jörg@example.org", claim="email") is True
    assert same_identity("JÖRG@example.org", "jörg@example.org", claim="sub") is False
