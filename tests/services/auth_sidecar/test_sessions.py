"""Tests for the auth sidecar's signed session cookies."""

from __future__ import annotations

import pytest
from itsdangerous import URLSafeSerializer

from osprey.services.auth_sidecar.exceptions import InvalidSessionError
from osprey.services.auth_sidecar.passwords import (
    generation_tag,
    hash_password,
    verify_generation_tag,
)
from osprey.services.auth_sidecar.sessions import (
    PAYLOAD_VERSION,
    SIGNATURE_SALT,
    SessionCodec,
    SessionState,
    UnlockedUser,
)

SECRET = "test-session-secret"
HOUR = 3600.0


class FakeClock:
    """A hand-advanced stand-in for ``time.time`` (absolute epoch seconds)."""

    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def codec(clock: FakeClock) -> SessionCodec:
    return SessionCodec(SECRET, max_age=12 * HOUR, clock=clock)


def sign_payload(payload: object, *, secret: str = SECRET) -> str:
    """Sign an arbitrary payload the way the codec does.

    Lets a test present a well-signed but wrong-shaped cookie — the case a
    signature check alone cannot catch.
    """
    return URLSafeSerializer(secret, salt=SIGNATURE_SALT).dumps(payload)


# --- construction -----------------------------------------------------------


@pytest.mark.parametrize("secret", ["", "   ", "\t\n"])
def test_missing_secret_refuses_to_build_a_codec(secret: str) -> None:
    """Fail closed: no secret means no sidecar, never unsigned cookies."""
    with pytest.raises(ValueError, match="secret is unset"):
        SessionCodec(secret)


def test_non_positive_max_age_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_age must be positive"):
        SessionCodec(SECRET, max_age=0)


def test_new_state_stamps_the_injected_clock(codec: SessionCodec, clock: FakeClock) -> None:
    state = codec.new_state()

    assert state.issued_at == clock.now
    assert state.users == ()
    assert state.session_id


def test_session_ids_are_unique(codec: SessionCodec) -> None:
    ids = {codec.new_state().session_id for _ in range(50)}

    assert len(ids) == 50


# --- round trips ------------------------------------------------------------


def test_round_trip_preserves_session_id_and_issue_time(codec: SessionCodec) -> None:
    state = codec.new_state()

    decoded = codec.decode(codec.encode(state))

    assert decoded.session_id == state.session_id
    assert decoded.issued_at == state.issued_at


def test_round_trip_preserves_a_user_entry(codec: SessionCodec, clock: FakeClock) -> None:
    state = codec.new_state().with_user(
        "alice", expires_at=clock.now + HOUR, generation_tag="0123456789abcdef"
    )

    decoded = codec.decode(codec.encode(state))

    assert decoded.users == (
        UnlockedUser(
            username="alice", expires_at=clock.now + HOUR, generation_tag="0123456789abcdef"
        ),
    )


def test_oidc_entry_round_trips_without_a_tag(codec: SessionCodec, clock: FakeClock) -> None:
    """OIDC has no stored hash behind it, so its entry carries no tag."""
    state = codec.new_state().with_user("alice", expires_at=clock.now + HOUR)

    decoded = codec.decode(codec.encode(state))

    entry = decoded.entry("alice")
    assert entry is not None
    assert entry.generation_tag == ""


def test_an_empty_session_round_trips(codec: SessionCodec) -> None:
    """What a session looks like after its last user logs out — not an error."""
    state = codec.new_state()

    decoded = codec.decode(codec.encode(state))

    assert decoded.users == ()
    assert decoded.unlocked_usernames(codec.now()) == ()


def test_re_issuing_does_not_extend_the_overall_session_age(
    codec: SessionCodec, clock: FakeClock
) -> None:
    """A login later in the session keeps the original issued-at stamp."""
    state = codec.new_state()
    clock.advance(HOUR)

    reissued = codec.decode(codec.encode(state.with_user("alice", expires_at=clock.now + HOUR)))

    assert reissued.issued_at == state.issued_at


# --- multi-user add and remove ----------------------------------------------


def test_several_users_are_unlocked_at_once(codec: SessionCodec, clock: FakeClock) -> None:
    state = (
        codec.new_state()
        .with_user("alice", expires_at=clock.now + HOUR)
        .with_user("bob", expires_at=clock.now + HOUR)
    )

    decoded = codec.decode(codec.encode(state))

    assert decoded.unlocked_usernames(clock.now) == ("alice", "bob")


def test_logging_out_one_user_keeps_the_others(codec: SessionCodec, clock: FakeClock) -> None:
    state = (
        codec.new_state()
        .with_user("alice", expires_at=clock.now + HOUR)
        .with_user("bob", expires_at=clock.now + HOUR)
    )

    decoded = codec.decode(codec.encode(state.without_user("alice")))

    assert decoded.is_unlocked("alice", clock.now) is False
    assert decoded.is_unlocked("bob", clock.now) is True


def test_logout_keeps_the_session_id(codec: SessionCodec, clock: FakeClock) -> None:
    """The revocation store retires the old cookie by id, so it must not change."""
    state = codec.new_state().with_user("alice", expires_at=clock.now + HOUR)

    assert state.without_user("alice").session_id == state.session_id


def test_removing_an_absent_user_is_inert(codec: SessionCodec, clock: FakeClock) -> None:
    state = codec.new_state().with_user("alice", expires_at=clock.now + HOUR)

    assert state.without_user("bob") == state


def test_re_adding_a_user_refreshes_that_entry_in_place(
    codec: SessionCodec, clock: FakeClock
) -> None:
    state = (
        codec.new_state()
        .with_user("alice", expires_at=clock.now + 60, generation_tag="aaaaaaaaaaaaaaaa")
        .with_user("bob", expires_at=clock.now + HOUR)
        .with_user("alice", expires_at=clock.now + HOUR, generation_tag="bbbbbbbbbbbbbbbb")
    )

    assert state.unlocked_usernames(clock.now) == ("alice", "bob")
    alice = state.entry("alice")
    assert alice is not None
    assert alice.expires_at == clock.now + HOUR
    assert alice.generation_tag == "bbbbbbbbbbbbbbbb"


def test_states_are_immutable(codec: SessionCodec, clock: FakeClock) -> None:
    """Routes build new states; an error path can never leave a half-update."""
    state = codec.new_state()

    state.with_user("alice", expires_at=clock.now + HOUR)

    assert state.users == ()


def test_empty_username_is_rejected(codec: SessionCodec, clock: FakeClock) -> None:
    with pytest.raises(ValueError, match="username must not be empty"):
        codec.new_state().with_user("", expires_at=clock.now + HOUR)


def test_unknown_user_has_no_entry(codec: SessionCodec, clock: FakeClock) -> None:
    state = codec.new_state().with_user("alice", expires_at=clock.now + HOUR)

    assert state.entry("bob") is None
    assert state.is_unlocked("bob", clock.now) is False


# --- per-user expiry --------------------------------------------------------


def test_entry_authorises_right_up_to_its_expiry(codec: SessionCodec, clock: FakeClock) -> None:
    state = codec.new_state().with_user("alice", expires_at=clock.now + HOUR)

    clock.advance(HOUR - 1)

    assert codec.decode(codec.encode(state)).is_unlocked("alice", clock.now) is True


def test_entry_stops_authorising_at_its_expiry(codec: SessionCodec, clock: FakeClock) -> None:
    state = codec.new_state().with_user("alice", expires_at=clock.now + HOUR)

    clock.advance(HOUR)

    assert codec.decode(codec.encode(state)).is_unlocked("alice", clock.now) is False


def test_decode_drops_expired_entries(codec: SessionCodec, clock: FakeClock) -> None:
    """Fail closed: an expired entry never comes back, clock check or not."""
    state = codec.new_state().with_user("alice", expires_at=clock.now + 60)
    encoded = codec.encode(state)

    clock.advance(120)

    assert codec.decode(encoded).entry("alice") is None


def test_one_user_expiring_leaves_the_others_unlocked(
    codec: SessionCodec, clock: FakeClock
) -> None:
    state = (
        codec.new_state()
        .with_user("alice", expires_at=clock.now + 60)
        .with_user("bob", expires_at=clock.now + HOUR)
    )
    encoded = codec.encode(state)

    clock.advance(120)

    assert codec.decode(encoded).unlocked_usernames(clock.now) == ("bob",)


def test_prune_expired_keeps_live_entries(codec: SessionCodec, clock: FakeClock) -> None:
    state = (
        codec.new_state()
        .with_user("alice", expires_at=clock.now + 60)
        .with_user("bob", expires_at=clock.now + HOUR)
    )

    pruned = state.prune_expired(clock.now + 120)

    assert pruned.unlocked_usernames(clock.now + 120) == ("bob",)
    assert pruned.session_id == state.session_id


# --- maximum session age ----------------------------------------------------


def test_cookie_within_the_maximum_age_decodes(codec: SessionCodec, clock: FakeClock) -> None:
    encoded = codec.encode(codec.new_state())

    clock.advance(12 * HOUR)

    assert codec.decode(encoded).users == ()


def test_cookie_past_the_maximum_age_is_rejected(codec: SessionCodec, clock: FakeClock) -> None:
    encoded = codec.encode(codec.new_state())

    clock.advance(12 * HOUR + 1)

    with pytest.raises(InvalidSessionError, match="older than the maximum session age"):
        codec.decode(encoded)


def test_maximum_age_can_be_overridden_per_call(codec: SessionCodec, clock: FakeClock) -> None:
    encoded = codec.encode(codec.new_state())

    clock.advance(120)

    with pytest.raises(InvalidSessionError, match="older than the maximum session age"):
        codec.decode(encoded, max_age=60)


def test_without_a_maximum_age_only_per_user_expiry_bounds_the_session(
    clock: FakeClock,
) -> None:
    codec = SessionCodec(SECRET, clock=clock)
    encoded = codec.encode(codec.new_state().with_user("alice", expires_at=clock.now + HOUR))

    clock.advance(365 * 24 * HOUR)

    decoded = codec.decode(encoded)

    assert decoded.users == ()


# --- tampering and malformed payloads ---------------------------------------


def test_a_swapped_payload_is_rejected(codec: SessionCodec, clock: FakeClock) -> None:
    """A signature only covers the body it was made for."""
    encoded = codec.encode(codec.new_state().with_user("alice", expires_at=clock.now + HOUR))
    signature = encoded.rpartition(".")[2]
    forged_body = sign_payload({"nope": True}).rpartition(".")[0]

    with pytest.raises(InvalidSessionError, match="signature verification"):
        codec.decode(f"{forged_body}.{signature}")


def test_a_tampered_signature_is_rejected(codec: SessionCodec, clock: FakeClock) -> None:
    """A changed signature must not verify.

    The FIRST character is changed, never the last. A base64url character
    carries six bits, but the final one of a 32-byte digest carries only four
    that mean anything — so for roughly one signature in fourteen, changing it
    decodes to the very same bytes, the codec correctly accepts a signature that
    was never actually altered, and this test fails for being wrong rather than
    for the codec being wrong. Every earlier character is fully significant, so
    changing one always changes the digest.
    """
    encoded = codec.encode(codec.new_state().with_user("alice", expires_at=clock.now + HOUR))
    body, _, signature = encoded.rpartition(".")
    flipped = ("a" if signature[0] != "a" else "b") + signature[1:]

    with pytest.raises(InvalidSessionError, match="signature verification"):
        codec.decode(f"{body}.{flipped}")


def test_a_cookie_signed_with_another_secret_is_rejected(codec: SessionCodec) -> None:
    """A cookie from a differently-keyed sidecar must not authorise here."""
    foreign = sign_payload(
        {"v": PAYLOAD_VERSION, "sid": "s", "iat": 0.0, "users": {}}, secret="other-secret"
    )

    with pytest.raises(InvalidSessionError, match="signature verification"):
        codec.decode(foreign)


def test_a_cookie_signed_with_another_salt_is_rejected(clock: FakeClock) -> None:
    other = SessionCodec(SECRET, clock=clock, salt="some-other-purpose")
    encoded = other.encode(other.new_state())

    with pytest.raises(InvalidSessionError, match="signature verification"):
        SessionCodec(SECRET, clock=clock).decode(encoded)


def test_an_empty_cookie_is_rejected(codec: SessionCodec) -> None:
    with pytest.raises(InvalidSessionError, match="empty"):
        codec.decode("")


def test_garbage_is_rejected(codec: SessionCodec) -> None:
    with pytest.raises(InvalidSessionError, match="signature verification"):
        codec.decode("not-a-cookie")


@pytest.mark.parametrize("version", [0, 2, "1", None])
def test_an_unknown_payload_version_is_rejected(codec: SessionCodec, version: object) -> None:
    """A retained secret across an upgrade must not resurrect old-shaped cookies."""
    encoded = sign_payload({"v": version, "sid": "s", "iat": 0.0, "users": {}})

    with pytest.raises(InvalidSessionError, match="unsupported session payload version"):
        codec.decode(encoded)


def test_the_current_version_is_stamped_into_the_payload(codec: SessionCodec) -> None:
    encoded = codec.encode(codec.new_state())

    payload = URLSafeSerializer(SECRET, salt=SIGNATURE_SALT).loads(encoded)

    assert payload["v"] == PAYLOAD_VERSION


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (["not", "an", "object"], "not an object"),
        ({"v": PAYLOAD_VERSION, "iat": 0.0, "users": {}}, "no session id"),
        ({"v": PAYLOAD_VERSION, "sid": "", "iat": 0.0, "users": {}}, "no session id"),
        ({"v": PAYLOAD_VERSION, "sid": 7, "iat": 0.0, "users": {}}, "no session id"),
        ({"v": PAYLOAD_VERSION, "sid": "s", "users": {}}, "'iat' is not a timestamp"),
        ({"v": PAYLOAD_VERSION, "sid": "s", "iat": "soon", "users": {}}, "not a timestamp"),
        ({"v": PAYLOAD_VERSION, "sid": "s", "iat": 0.0, "users": []}, "'users' is not an object"),
        (
            {"v": PAYLOAD_VERSION, "sid": "s", "iat": 0.0, "users": {"alice": "yes"}},
            "not an object",
        ),
        (
            {"v": PAYLOAD_VERSION, "sid": "s", "iat": 0.0, "users": {"alice": {"tag": "a"}}},
            "exp.* is not a timestamp",
        ),
        (
            {
                "v": PAYLOAD_VERSION,
                "sid": "s",
                "iat": 0.0,
                "users": {"alice": {"exp": 1, "tag": 9}},
            },
            "non-string tag",
        ),
        (
            {"v": PAYLOAD_VERSION, "sid": "s", "iat": 0.0, "users": {"": {"exp": 1}}},
            "empty username",
        ),
    ],
)
def test_a_malformed_payload_is_rejected(payload: object, match: str) -> None:
    """Signed but wrong-shaped: what a version skew or a bug upstream looks like.

    Age-unbounded on purpose — these payloads are about shape, and the
    fixed ``iat`` in them would otherwise trip the maximum-age check first.
    """
    with pytest.raises(InvalidSessionError, match=match):
        SessionCodec(SECRET).decode(sign_payload(payload))


def test_a_payload_without_users_decodes_as_empty(codec: SessionCodec) -> None:
    encoded = sign_payload({"v": PAYLOAD_VERSION, "sid": "s", "iat": 0.0})

    assert SessionCodec(SECRET).decode(encoded).users == ()


# --- what the cookie may carry ----------------------------------------------


def test_the_stored_hash_never_enters_the_cookie(codec: SessionCodec, clock: FakeClock) -> None:
    """Cookies are signed, not encrypted: only the one-way tag may travel."""
    stored = hash_password("correct horse battery staple")
    encoded = codec.encode(
        codec.new_state().with_user(
            "alice", expires_at=clock.now + HOUR, generation_tag=generation_tag(stored)
        )
    )

    assert stored not in encoded
    assert stored.split("$")[-1] not in encoded


def test_the_carried_tag_verifies_against_the_stored_hash(
    codec: SessionCodec, clock: FakeClock
) -> None:
    stored = hash_password("correct horse battery staple")
    state = codec.new_state().with_user(
        "alice", expires_at=clock.now + HOUR, generation_tag=generation_tag(stored)
    )

    entry = codec.decode(codec.encode(state)).entry("alice")

    assert entry is not None
    assert verify_generation_tag(entry.generation_tag, stored) is True


def test_a_rotated_password_invalidates_the_carried_tag(
    codec: SessionCodec, clock: FakeClock
) -> None:
    """Rotation kills sessions statelessly — this survives a sidecar restart."""
    old_stored = hash_password("old-password")
    state = codec.new_state().with_user(
        "alice", expires_at=clock.now + HOUR, generation_tag=generation_tag(old_stored)
    )
    rotated = hash_password("new-password")

    entry = codec.decode(codec.encode(state)).entry("alice")

    assert entry is not None
    assert verify_generation_tag(entry.generation_tag, rotated) is False


def test_an_oidc_entry_never_satisfies_the_password_mode_tag_check(
    codec: SessionCodec, clock: FakeClock
) -> None:
    """An untagged entry must not authorise where a stored hash exists."""
    state = codec.new_state().with_user("alice", expires_at=clock.now + HOUR)
    stored = hash_password("alice-password")

    entry = codec.decode(codec.encode(state)).entry("alice")

    assert entry is not None
    assert verify_generation_tag(entry.generation_tag, stored) is False


def test_state_defaults_to_wall_clock_time() -> None:
    """Expiries are absolute epoch seconds, matching the revocation store."""
    import time

    codec = SessionCodec(SECRET, max_age=12 * HOUR)
    state = codec.new_state().with_user("alice", expires_at=time.time() + HOUR)

    decoded = codec.decode(codec.encode(state))

    assert decoded.is_unlocked("alice", time.time()) is True


def test_state_is_hashable() -> None:
    """Frozen all the way down, so a state can key a cache or land in a set."""
    state = SessionState(session_id="s", issued_at=0.0).with_user("alice", expires_at=1.0)

    assert len({state, state}) == 1
