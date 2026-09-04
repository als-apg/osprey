"""Tests for the identity-headers module's constant-time comparator.

:func:`~osprey.services.auth_sidecar.identity_headers.same_value` is the one
equality check every identity decision in the sidecar goes through, so what is
pinned here is small and load-bearing: equal values match, unequal values do
not, and a non-ASCII value on either side is *compared* rather than raising —
the exact failure mode that motivated encoding to UTF-8 bytes before handing
the pair to :func:`secrets.compare_digest`.

:func:`~osprey.services.auth_sidecar.identity_headers.same_identity` and
:func:`~osprey.services.auth_sidecar.identity_headers.same_domain` are pinned
here for the same reason: each is the *one* rule its kind of admission decision
goes through, so what counts as a match must not be re-decided at a call site.
"""

from __future__ import annotations

import inspect

import pytest

from osprey.services.auth_sidecar import identity_headers
from osprey.services.auth_sidecar.identity_headers import (
    CASE_INSENSITIVE_CLAIMS,
    same_domain,
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


# --- same_domain: the domain part decides, exactly ---------------------------


def test_an_address_in_the_domain_matches() -> None:
    assert same_domain("alice@lbl.gov", "lbl.gov") is True


@pytest.mark.parametrize(
    ("identity", "domain"),
    [
        ("alice@LBL.GOV", "lbl.gov"),
        ("alice@Lbl.Gov", "lbl.gov"),
        ("alice@lbl.gov", "LBL.GOV"),
        ("alice@LBL.gov", "lbl.GOV"),
    ],
)
def test_the_domain_is_matched_without_regard_to_case(identity: str, domain: str) -> None:
    """DNS is case-insensitive, so the spelling a provider happens to assert
    cannot decide whether an operator gets in."""
    assert same_domain(identity, domain) is True


@pytest.mark.parametrize(
    ("identity", "domain"),
    [
        ("alice@als.lbl.gov", "lbl.gov"),
        ("alice@lbl.gov", "als.lbl.gov"),
        ("alice@notlbl.gov", "lbl.gov"),
        ("alice@lbl.gov.example.org", "lbl.gov"),
    ],
)
def test_only_the_exact_domain_matches(identity: str, domain: str) -> None:
    """Never a suffix match: a rule written for one domain must not hand the
    grant to every host delegated beneath it, nor to a domain that merely ends
    in the same letters."""
    assert same_domain(identity, domain) is False


@pytest.mark.parametrize("identity", ["Alice@lbl.gov", "ALICE@lbl.gov", "a.LICE@lbl.gov"])
def test_the_local_part_case_changes_nothing(identity: str) -> None:
    """RFC 5321 leaves the local part to the destination host, so this
    comparator neither folds it nor lets it affect the answer."""
    assert same_domain(identity, "lbl.gov") is True


def test_the_last_at_sign_separates() -> None:
    """A quoted local part may contain ``@``; mail reads the last one as the
    separator, and so does this."""
    assert same_domain('"a@b"@lbl.gov', "lbl.gov") is True
    assert same_domain('"a@b"@lbl.gov', "b") is False


@pytest.mark.parametrize(
    ("identity", "domain"),
    [
        ("alice", "lbl.gov"),
        ("lbl.gov", "lbl.gov"),
        ("alice@", "lbl.gov"),
        ("@lbl.gov", "lbl.gov"),
        ("", "lbl.gov"),
        ("alice@lbl.gov", ""),
        ("", ""),
    ],
)
def test_anything_that_names_no_mailbox_in_a_domain_is_refused(identity: str, domain: str) -> None:
    """Refused, not raised: the input is untrusted and an admission check needs
    an answer it can act on, which for a value it cannot read is ``False``."""
    assert same_domain(identity, domain) is False


@pytest.mark.parametrize(
    ("identity", "domain"),
    [
        ("alice@lbl.gov.", "lbl.gov"),
        ("alice@lbl.gov", "lbl.gov."),
        ("alice@.lbl.gov", "lbl.gov"),
    ],
)
def test_a_trailing_or_leading_dot_is_a_different_domain(identity: str, domain: str) -> None:
    assert same_domain(identity, domain) is False


def test_a_non_ascii_domain_is_folded_only_across_a_to_z() -> None:
    """The fold matches the one the authored value already went through, so the
    two agree byte for byte. ``str.lower()`` here would fold the umlaut on this
    side and not on the config side, and the domain would silently never
    match."""
    assert same_domain("alice@BÜRO.example", "bÜro.example") is True
    assert same_domain("alice@büro.example", "büro.example") is True
    assert same_domain("alice@BÜRO.example", "büro.example") is False


def test_a_unicode_domain_never_matches_its_punycode_spelling() -> None:
    """By design: nothing here invents the translation between the two."""
    assert same_domain("alice@xn--bro-vka.example", "büro.example") is False


# --- a lone surrogate is compared, never raised ------------------------------

_SURROGATE = "\ud800"
"""A lone surrogate, the one code point plain UTF-8 refuses to encode.

An identity provider can put it in a claim as the well-formed JSON escape
``"\\ud800"``, so it reaches the comparators as untrusted input. Strict UTF-8
would make that an unhandled ``UnicodeEncodeError`` — a 500 at the callback
rather than a refusal — so all three comparators have to answer it.
"""


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (_SURROGATE, "alice", False),
        ("alice", _SURROGATE, False),
        (_SURROGATE, _SURROGATE, True),
    ],
)
def test_same_value_compares_a_lone_surrogate(left: str, right: str, expected: bool) -> None:
    assert same_value(left, right) is expected


@pytest.mark.parametrize("claim", ["email", "sub"])
def test_same_identity_compares_a_lone_surrogate(claim: str) -> None:
    assert same_identity(_SURROGATE, "alice@lbl.gov", claim=claim) is False
    assert same_identity("alice@lbl.gov", _SURROGATE, claim=claim) is False


def test_same_domain_compares_a_lone_surrogate() -> None:
    assert same_domain(f"alice@{_SURROGATE}", "lbl.gov") is False
    assert same_domain("alice@lbl.gov", _SURROGATE) is False


# --- the module's declared surface -------------------------------------------


def _defined_here() -> set[str]:
    """The public names ``identity_headers`` itself defines.

    An import is not part of the surface — pinning ``secrets`` would make the
    assertion a diff of the import block. What is left is what the module
    writes: its header-name and charset constants (which carry no
    ``__module__``) and its own functions.
    """
    surface: set[str] = set()
    for name in dir(identity_headers):
        if name.startswith("_"):
            continue
        value = getattr(identity_headers, name)
        if inspect.ismodule(value):
            continue
        origin = getattr(value, "__module__", None)
        if origin is not None and origin != identity_headers.__name__:
            continue
        surface.add(name)
    return surface


def test_the_declared_surface_is_the_defined_one() -> None:
    """``__all__`` is checked against reality in both directions, so a new
    comparator cannot reach a call site without being declared here (or be
    declared and never written)."""
    assert set(identity_headers.__all__) == _defined_here()


def test_the_public_surface_is_exactly_these_nine_names() -> None:
    """Structural, not a naming canary: any new public name added beside these
    fails this test whatever it is called, so a second, subtly different
    comparator cannot slip in unreviewed."""
    assert _defined_here() == {
        "ACCOUNT_HEADER",
        "SUBJECT_HEADER",
        "ROLE_HEADER",
        "ROLE_SOURCE_HEADER",
        "CASE_INSENSITIVE_CLAIMS",
        "is_header_safe",
        "same_domain",
        "same_identity",
        "same_value",
    }
