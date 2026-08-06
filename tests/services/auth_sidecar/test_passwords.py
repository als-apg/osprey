"""Tests for auth-sidecar password hashing and credential-generation tags."""

from __future__ import annotations

import hashlib

import pytest

from osprey.services.auth_sidecar.passwords import (
    GENERATION_TAG_CHARS,
    SCHEME,
    SCRYPT_N,
    SCRYPT_P,
    SCRYPT_R,
    generation_tag,
    hash_password,
    verify_generation_tag,
    verify_password,
)

PASSWORD = "correct horse battery staple"


@pytest.fixture(scope="module")
def stored() -> str:
    """A stored-hash string for :data:`PASSWORD`, minted once for the module."""
    return hash_password(PASSWORD)


class TestHashFormat:
    """The stored string carries its own scheme and cost parameters."""

    def test_fields_are_scheme_then_pinned_parameters(self, stored: str) -> None:
        fields = stored.split("$")
        assert len(fields) == 6
        scheme, n, r, p = fields[:4]
        assert scheme == SCHEME
        assert (int(n), int(r), int(p)) == (SCRYPT_N, SCRYPT_R, SCRYPT_P)

    def test_pinned_cost_is_the_documented_one(self) -> None:
        assert (SCRYPT_N, SCRYPT_R, SCRYPT_P) == (2**14, 8, 1)

    def test_salt_and_hash_are_non_empty(self, stored: str) -> None:
        salt, digest = stored.split("$")[4:]
        assert salt and digest

    def test_plaintext_never_appears_in_the_stored_string(self, stored: str) -> None:
        assert PASSWORD not in stored

    def test_each_hash_draws_a_fresh_salt(self) -> None:
        first, second = hash_password("same"), hash_password("same")
        assert first != second
        assert first.split("$")[4] != second.split("$")[4]

    def test_empty_password_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            hash_password("")


class TestVerify:
    """Verification round-trips and fails closed."""

    def test_correct_password_verifies(self, stored: str) -> None:
        assert verify_password(PASSWORD, stored) is True

    def test_wrong_password_is_rejected(self, stored: str) -> None:
        assert verify_password("wrong horse battery staple", stored) is False

    def test_empty_password_is_rejected(self, stored: str) -> None:
        assert verify_password("", stored) is False

    def test_non_ascii_password_round_trips(self) -> None:
        secret = "pässwörd-Ω-日本語"
        assert verify_password(secret, hash_password(secret)) is True

    def test_parameters_are_read_from_the_stored_string(self) -> None:
        """A hash minted at a different cost still verifies."""
        legacy = hash_password(PASSWORD, n=2**4, r=1, p=1)
        assert legacy.split("$")[1:4] == ["16", "1", "1"]
        assert verify_password(PASSWORD, legacy) is True
        assert verify_password("nope", legacy) is False

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "not-a-hash",
            "scrypt$16384$8$1$c2FsdA",  # too few fields
            "scrypt$16384$8$1$c2FsdA$aGFzaA$extra",  # too many fields
            "bcrypt$16384$8$1$c2FsdA$aGFzaA",  # unknown scheme
            "scrypt$many$8$1$c2FsdA$aGFzaA",  # non-integer cost
            "scrypt$0$8$1$c2FsdA$aGFzaA",  # out-of-range cost
            "scrypt$16384$8$1$!!!$aGFzaA",  # undecodable salt
            "scrypt$16384$8$1$$aGFzaA",  # empty salt
            "scrypt$16384$8$1$c2FsdA$",  # empty hash
        ],
    )
    def test_malformed_stored_hash_fails_closed(self, bad: str) -> None:
        assert verify_password(PASSWORD, bad) is False


class TestGenerationTag:
    """The tag is a truncated one-way digest of the stored-hash string."""

    def test_tag_is_truncated_sha256_of_the_stored_string(self, stored: str) -> None:
        expected = hashlib.sha256(stored.encode("utf-8")).hexdigest()[:GENERATION_TAG_CHARS]
        assert generation_tag(stored) == expected

    def test_tag_shape_is_short_lowercase_hex(self, stored: str) -> None:
        tag = generation_tag(stored)
        assert len(tag) == GENERATION_TAG_CHARS == 16
        assert all(char in "0123456789abcdef" for char in tag)

    def test_tag_is_deterministic(self, stored: str) -> None:
        assert generation_tag(stored) == generation_tag(stored)

    def test_tag_discloses_neither_the_hash_nor_the_password(self, stored: str) -> None:
        tag = generation_tag(stored)
        salt, digest = stored.split("$")[4:]
        assert tag not in stored
        assert salt not in tag
        assert digest not in tag
        assert PASSWORD not in tag

    def test_empty_stored_hash_has_no_tag(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            generation_tag("")

    def test_tag_verifies_against_its_own_stored_hash(self, stored: str) -> None:
        assert verify_generation_tag(generation_tag(stored), stored) is True

    def test_rotated_password_invalidates_the_tag(self, stored: str) -> None:
        rotated = hash_password("a brand new password")
        assert verify_generation_tag(generation_tag(stored), rotated) is False

    def test_rehashing_the_same_password_invalidates_the_tag(self, stored: str) -> None:
        """A fresh salt means a fresh generation, even for an unchanged password."""
        rehashed = hash_password(PASSWORD)
        assert rehashed != stored
        assert verify_generation_tag(generation_tag(stored), rehashed) is False

    @pytest.mark.parametrize("tag", ["", "0" * GENERATION_TAG_CHARS, "short"])
    def test_missing_or_foreign_tag_is_rejected(self, tag: str, stored: str) -> None:
        assert verify_generation_tag(tag, stored) is False

    def test_tag_check_fails_closed_without_a_stored_hash(self, stored: str) -> None:
        assert verify_generation_tag(generation_tag(stored), "") is False
