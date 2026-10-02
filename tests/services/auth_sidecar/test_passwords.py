"""Tests for auth-sidecar password hashing and credential-generation tags."""

from __future__ import annotations

import hashlib

import pytest

from osprey.services.auth_sidecar.passwords import (
    FIELD_SEP,
    GENERATION_TAG_CHARS,
    SCHEME,
    SCRYPT_MAXMEM,
    SCRYPT_N,
    SCRYPT_P,
    SCRYPT_R,
    PasswordCheck,
    check_password,
    generation_tag,
    hash_password,
    stored_hash_problem,
    verify_generation_tag,
    verify_password,
)

PASSWORD = "correct horse battery staple"

MALFORMED = [
    "",
    "not-a-hash",
    "scrypt.16384.8.1.c2FsdA",  # too few fields
    "scrypt.16384.8.1.c2FsdA.aGFzaA.extra",  # too many fields
    "bcrypt.16384.8.1.c2FsdA.aGFzaA",  # unknown scheme
    "scrypt.many.8.1.c2FsdA.aGFzaA",  # non-integer cost
    "scrypt.0.8.1.c2FsdA.aGFzaA",  # out-of-range cost
    "scrypt.16384.8.1.!!!.aGFzaA",  # undecodable salt
    "scrypt.16384.8.1..aGFzaA",  # empty salt
    "scrypt.16384.8.1.c2FsdA.",  # empty hash
    "scrypt.3.8.1.c2FsdA.aGFzaA",  # cost is not a power of two
    "scrypt.65536.1.1.c2FsdA.aGFzaA",  # cost at or above 2**(16*r)
    "scrypt.65536.8.1.c2FsdA.aGFzaA",  # over the memory ceiling
    "scrypt.16384.8.1.c2Fsd.aGFzaA",  # base64 of an impossible length
    "scrypt$16384$8$1",  # a $-separated hash after compose interpolation
]
"""Stored strings the service cannot evaluate, one per refusal the parse makes."""


@pytest.fixture(scope="module")
def stored() -> str:
    """A stored-hash string for :data:`PASSWORD`, minted once for the module."""
    return hash_password(PASSWORD)


class TestHashFormat:
    """The stored string carries its own scheme and cost parameters."""

    def test_fields_are_scheme_then_pinned_parameters(self, stored: str) -> None:
        fields = stored.split(FIELD_SEP)
        assert len(fields) == 6
        scheme, n, r, p = fields[:4]
        assert scheme == SCHEME
        assert (int(n), int(r), int(p)) == (SCRYPT_N, SCRYPT_R, SCRYPT_P)

    def test_pinned_cost_is_the_documented_one(self) -> None:
        assert (SCRYPT_N, SCRYPT_R, SCRYPT_P) == (2**14, 8, 1)

    def test_salt_and_hash_are_non_empty(self, stored: str) -> None:
        salt, digest = stored.split(FIELD_SEP)[4:]
        assert salt and digest

    def test_plaintext_never_appears_in_the_stored_string(self, stored: str) -> None:
        assert PASSWORD not in stored

    def test_each_hash_draws_a_fresh_salt(self) -> None:
        first, second = hash_password("same"), hash_password("same")
        assert first != second
        assert first.split(FIELD_SEP)[4] != second.split(FIELD_SEP)[4]

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
        assert legacy.split(FIELD_SEP)[1:4] == ["16", "1", "1"]
        assert verify_password(PASSWORD, legacy) is True
        assert verify_password("nope", legacy) is False

    @pytest.mark.parametrize("bad", MALFORMED)
    def test_malformed_stored_hash_fails_closed(self, bad: str) -> None:
        assert verify_password(PASSWORD, bad) is False


class TestCheck:
    """A check has three outcomes: match, mismatch, or a stored hash it cannot evaluate."""

    def test_a_matching_password_is_a_match(self, stored: str) -> None:
        assert check_password(PASSWORD, stored) is PasswordCheck.MATCH

    def test_a_wrong_password_is_a_mismatch(self, stored: str) -> None:
        assert check_password("wrong horse battery staple", stored) is PasswordCheck.MISMATCH

    def test_an_empty_password_is_a_mismatch(self, stored: str) -> None:
        assert check_password("", stored) is PasswordCheck.MISMATCH

    @pytest.mark.parametrize("bad", MALFORMED)
    def test_a_malformed_stored_hash_is_unevaluable(self, bad: str) -> None:
        assert check_password(PASSWORD, bad) is PasswordCheck.UNEVALUABLE

    def test_an_empty_password_against_a_malformed_hash_is_unevaluable(self) -> None:
        assert check_password("", "scrypt.16384.8.1.c2FsdA") is PasswordCheck.UNEVALUABLE

    def test_a_kdf_refusal_is_unevaluable(
        self, stored: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from osprey.services.auth_sidecar import passwords

        def refuse(*_args: object, **_kwargs: object) -> bytes:
            raise MemoryError

        monkeypatch.setattr(passwords.hashlib, "scrypt", refuse)
        assert check_password(PASSWORD, stored) is PasswordCheck.UNEVALUABLE

    def test_verify_password_is_the_match_projection(self, stored: str) -> None:
        cases = [(PASSWORD, stored), ("wrong", stored), ("", stored), (PASSWORD, MALFORMED[2])]
        for password, value in cases:
            expected = check_password(password, value) is PasswordCheck.MATCH
            assert verify_password(password, value) is expected


class TestStoredHashProblem:
    """One shape test, shared by every surface that judges a stored hash before login."""

    def test_a_minted_hash_has_no_problem(self, stored: str) -> None:
        assert stored_hash_problem(stored) is None
        assert stored_hash_problem(hash_password(PASSWORD, n=2**4, r=1, p=1)) is None

    @pytest.mark.parametrize("bad", MALFORMED)
    def test_every_malformed_hash_names_a_problem(self, bad: str) -> None:
        problem = stored_hash_problem(bad)
        assert isinstance(problem, str) and problem

    @pytest.mark.parametrize("bad", MALFORMED)
    def test_the_problem_never_quotes_the_stored_value(self, bad: str) -> None:
        problem = stored_hash_problem(bad)
        assert problem is not None
        fields = [f for part in bad.split(FIELD_SEP) for f in part.split("$")]
        # The scheme's own name is the module's constant, not part of the operator's value.
        for field in fields:
            if len(field) >= 3 and field != SCHEME:
                assert field not in problem

    @pytest.mark.parametrize(
        ("n", "r", "p"),
        [
            (2, 1, 1),
            (3, 1, 1),
            (2**15, 1, 1),
            (2**16, 1, 1),
            (2**4, 2, 1),
            (2**15, 8, 1),
            (2**16, 8, 1),
            (2**16, 7, 1),
            (2**14, 64, 1),
            (2**4, 1, 200),
        ],
    )
    def test_the_shape_check_agrees_with_the_kdf(self, n: int, r: int, p: int) -> None:
        value = FIELD_SEP.join((SCHEME, str(n), str(r), str(p), "c2FsdA", "aGFzaA"))
        try:
            hashlib.scrypt(b"x", salt=b"salt", n=n, r=r, p=p, maxmem=SCRYPT_MAXMEM, dklen=32)
        except ValueError:
            kdf_accepts = False
        else:
            kdf_accepts = True
        assert (stored_hash_problem(value) is None) is kdf_accepts


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
        salt, digest = stored.split(FIELD_SEP)[4:]
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
