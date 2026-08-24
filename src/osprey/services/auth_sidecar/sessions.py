"""Signed session cookies for the auth sidecar.

One browser may hold several unlocked users at once, so a session is not "who am
I" but "which roster users has this browser unlocked, and until when". That set
lives in a single :mod:`itsdangerous`-signed cookie::

    {"v": 1, "sid": "<session id>", "iat": <epoch>, "users": {
        "alice": {"exp": <epoch>, "tag": "0123456789abcdef", "sub": ""}
    }}

**Signed, not encrypted.** Anyone holding the cookie can read the payload; they
just cannot change it. Nothing secret may go in — in particular the stored
password hash never does. Each password-mode entry carries only the
credential-generation tag from :mod:`~osprey.services.auth_sidecar.passwords`, a
one-way digest the verify endpoint recomputes from the current stored hash so a
rotated password invalidates that user's sessions without server-side state.
OIDC entries carry no tag (there is no stored hash), which is why
:func:`~osprey.services.auth_sidecar.passwords.verify_generation_tag` refuses an
empty tag: a password-mode session must never authorise without one. They carry
the provider's subject instead — an opaque account identifier, not a
credential — so a later request can tell which provider account is behind an
unlocked user without re-contacting the provider.

The ``"sub"`` key is read with a default rather than being made mandatory, and
was added without a :data:`PAYLOAD_VERSION` bump: a cookie minted before it
existed still decodes, with an empty subject.

**The codec only signs and verifies.** Cookie attributes — ``SameSite=Lax``,
``HttpOnly`` always, ``Secure`` under TLS — are the caller's to set on the
response; this module never touches a request or response object.

**Fail closed.** Constructing a codec without a signing secret raises rather than
falling back to an unsigned or default-keyed cookie, so a misconfigured sidecar
cannot start and hand out forgeable sessions. Decoding raises
:class:`~osprey.services.auth_sidecar.exceptions.InvalidSessionError` for a
bad signature, a malformed payload, an
unknown payload version, or a cookie older than the permitted age, and silently
drops entries that have passed their expiry — an expired user is never returned
as unlocked even if a caller forgets to re-check the clock.

Expiries are absolute epoch seconds, matching
:class:`~osprey.services.auth_sidecar.revocation.RevocationStore`, so a session's
expiry can be passed straight from one to the other.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from itsdangerous import BadSignature, URLSafeSerializer

from .exceptions import InvalidSessionError

__all__ = [
    "PAYLOAD_VERSION",
    "SESSION_COOKIE_NAME",
    "SIGNATURE_SALT",
    "SessionCodec",
    "SessionState",
    "UnlockedUser",
]

PAYLOAD_VERSION = 1
"""Version stamped into every payload; decoding rejects any other value.

A deployment that keeps its signing secret across an upgrade would otherwise
have old cookies deserialise into a new payload shape. Bumping this retires
every outstanding session instead.
"""

SESSION_COOKIE_NAME = "osprey_auth_session"
"""Cookie name the sidecar's routes use for the session.

Defined here so the login, logout, verify and OIDC routes agree on one name.
The codec itself never reads or writes a cookie.
"""

SIGNATURE_SALT = "osprey-auth-session"
"""Namespacing salt for the signature.

Keeps session signatures distinct from any other value signed with the same
secret, so a token minted for one purpose cannot be replayed as another.
"""

SESSION_ID_BYTES = 16
"""Entropy drawn for a new session id."""


@dataclass(frozen=True, slots=True)
class UnlockedUser:
    """One roster user this browser has unlocked.

    Attributes:
        username: The roster username, exactly as it appears in ``/u/<user>/``.
        expires_at: Absolute epoch seconds after which this entry no longer
            authorises, whatever the rest of the session's state.
        generation_tag: The credential-generation tag from
            :func:`~osprey.services.auth_sidecar.passwords.generation_tag`, or
            ``""`` for an OIDC entry, which has no stored hash behind it.
        oidc_subject: The ``sub`` claim of the OIDC provider that unlocked this
            entry, or ``""`` in password mode and for any session minted before
            the subject was carried. An opaque account identifier, never a
            credential — the cookie is signed, not encrypted.
    """

    username: str
    expires_at: float
    generation_tag: str = ""
    oidc_subject: str = ""

    def is_expired(self, now: float) -> bool:
        """Whether this entry has reached its expiry at ``now``."""
        return now >= self.expires_at


@dataclass(frozen=True, slots=True)
class SessionState:
    """The decoded contents of one session cookie.

    Immutable: :meth:`with_user` and :meth:`without_user` return a new state
    rather than mutating this one, so a route that re-issues a cookie cannot
    accidentally leave the old state half-updated on an error path.

    Attributes:
        session_id: Stable across logins and logouts within one browser session;
            this is the id the revocation store records at logout.
        issued_at: Absolute epoch seconds when the session id was minted. Bounds
            the whole session's life independently of any single user's expiry.
        users: The unlocked-user entries, in the order they were added.
    """

    session_id: str
    issued_at: float
    users: tuple[UnlockedUser, ...] = field(default=())

    @classmethod
    def new(cls, now: float) -> SessionState:
        """Start an empty session with a freshly minted id.

        Args:
            now: Current time as absolute epoch seconds.
        """
        return cls(session_id=secrets.token_urlsafe(SESSION_ID_BYTES), issued_at=now, users=())

    def entry(self, username: str) -> UnlockedUser | None:
        """The entry for ``username``, or ``None`` if this session has none.

        Returns the entry whether or not it has expired; the caller that needs
        the generation tag also needs to know an expired entry exists. Use
        :meth:`is_unlocked` for the authorisation question.
        """
        for user in self.users:
            if user.username == username:
                return user
        return None

    def is_unlocked(self, username: str, now: float) -> bool:
        """Whether ``username`` is unlocked and unexpired at ``now``.

        This is the session's half of the verify decision. The generation tag
        and the revocation list are checked separately by the verify route,
        which has the stored hash and the store.

        Args:
            username: The roster user the request is for.
            now: Current time as absolute epoch seconds.
        """
        entry = self.entry(username)
        return entry is not None and not entry.is_expired(now)

    def unlocked_usernames(self, now: float) -> tuple[str, ...]:
        """Every username unlocked and unexpired at ``now``, in insertion order."""
        return tuple(user.username for user in self.users if not user.is_expired(now))

    def with_user(
        self,
        username: str,
        *,
        expires_at: float,
        generation_tag: str = "",
        oidc_subject: str = "",
    ) -> SessionState:
        """Return this state with ``username`` unlocked until ``expires_at``.

        Re-adding a user replaces that entry in place — a fresh login restates
        the expiry, the tag and the subject without disturbing the order of the
        others. Replacement is wholesale, not a merge: a login that omits the
        subject clears the one the previous login carried, exactly as it already
        does for the tag.

        Args:
            username: The roster user being unlocked.
            expires_at: Absolute epoch seconds at which the entry lapses.
            generation_tag: The credential-generation tag in password mode;
                omitted for OIDC.
            oidc_subject: The provider's ``sub`` claim in OIDC mode; omitted in
                password mode, which has no provider account behind it.

        Raises:
            ValueError: If ``username`` is empty.
        """
        if not username:
            raise ValueError("username must not be empty")

        entry = UnlockedUser(
            username=username,
            expires_at=float(expires_at),
            generation_tag=generation_tag,
            oidc_subject=oidc_subject,
        )
        if self.entry(username) is None:
            return replace(self, users=(*self.users, entry))
        return replace(
            self,
            users=tuple(entry if user.username == username else user for user in self.users),
        )

    def without_user(self, username: str) -> SessionState:
        """Return this state with ``username`` locked again.

        This is logout: the other unlocked users survive, and the session id is
        unchanged so the revocation store can retire the old cookie by id.
        """
        return replace(self, users=tuple(user for user in self.users if user.username != username))

    def prune_expired(self, now: float) -> SessionState:
        """Return this state with every entry that has passed its expiry dropped."""
        return replace(self, users=tuple(user for user in self.users if not user.is_expired(now)))


def _as_epoch(value: Any, field_name: str) -> float:
    """Read a payload field as absolute epoch seconds.

    Raises:
        InvalidSessionError: If the value is not a real number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidSessionError(f"session payload field {field_name!r} is not a timestamp")
    return float(value)


class SessionCodec:
    """Signs and verifies :class:`SessionState` payloads.

    The codec holds the only clock in the session path: the issued-at stamp is
    written from it and the maximum-age check reads it back, so tests advance
    time instead of sleeping. ``itsdangerous``' own timed serializer is
    deliberately not used — its timestamp comes from the real wall clock and
    could not be injected.
    """

    def __init__(
        self,
        secret: str,
        *,
        max_age: float | None = None,
        clock: Callable[[], float] = time.time,
        salt: str = SIGNATURE_SALT,
    ) -> None:
        """Create a codec.

        Args:
            secret: The signing secret, typically ``OSPREY_AUTH_SESSION_SECRET``.
            max_age: Maximum age in seconds for a cookie, measured from its
                issued-at stamp. ``None`` bounds the session only by its
                per-user expiries; deployments pass ``auth.session_lifetime``.
            clock: Returns the current time as absolute epoch seconds.
            salt: Signature namespacing salt.

        Raises:
            ValueError: If ``secret`` is unset, empty or only whitespace. The
                sidecar fails to start rather than issuing forgeable cookies.
        """
        if not secret or not secret.strip():
            raise ValueError("session signing secret is unset; refusing to issue session cookies")
        if max_age is not None and max_age <= 0:
            raise ValueError("max_age must be positive when set")

        self._clock = clock
        self._max_age = max_age
        self._serializer = URLSafeSerializer(secret, salt=salt)

    def now(self) -> float:
        """Current time as absolute epoch seconds, from the injected clock.

        Exposed so routes stamp expiries from the same clock the codec checks
        them against.
        """
        return self._clock()

    def new_state(self) -> SessionState:
        """Start an empty session, stamped with this codec's clock."""
        return SessionState.new(self._clock())

    def encode(self, state: SessionState) -> str:
        """Sign ``state`` into the cookie value.

        Args:
            state: The session to serialise. Its ``issued_at`` is preserved, so
                re-issuing a cookie after a login or logout does not extend the
                session's overall age.

        Returns:
            A URL-safe signed string for the caller to set as a cookie value.
        """
        payload = {
            "v": PAYLOAD_VERSION,
            "sid": state.session_id,
            "iat": state.issued_at,
            "users": {
                user.username: {
                    "exp": user.expires_at,
                    "tag": user.generation_tag,
                    "sub": user.oidc_subject,
                }
                for user in state.users
            },
        }
        return self._serializer.dumps(payload)

    def decode(self, raw: str, *, max_age: float | None = None) -> SessionState:
        """Verify a cookie value and return the session it carries.

        Entries that have passed their expiry are dropped from the result, so an
        expired user cannot be authorised even by a caller that skips the clock
        check. An empty result is normal and not an error: it is what a session
        looks like after its last user logs out.

        Args:
            raw: The cookie value as received.
            max_age: Overrides the codec's maximum age for this call.

        Returns:
            The decoded state, with expired entries removed.

        Raises:
            InvalidSessionError: If the signature does not verify, the payload
                is malformed or of an unknown version, or the cookie is older
                than the permitted age.
        """
        if not raw:
            raise InvalidSessionError("session cookie is empty")

        try:
            payload = self._serializer.loads(raw)
        except BadSignature as exc:
            raise InvalidSessionError("session cookie failed signature verification") from exc

        if not isinstance(payload, dict):
            raise InvalidSessionError("session payload is not an object")

        version = payload.get("v")
        if version != PAYLOAD_VERSION:
            raise InvalidSessionError(f"unsupported session payload version: {version!r}")

        session_id = payload.get("sid")
        if not isinstance(session_id, str) or not session_id:
            raise InvalidSessionError("session payload carries no session id")

        issued_at = _as_epoch(payload.get("iat"), "iat")
        effective_max_age = self._max_age if max_age is None else max_age
        now = self._clock()
        if effective_max_age is not None and now - issued_at > effective_max_age:
            raise InvalidSessionError("session cookie is older than the maximum session age")

        state = SessionState(
            session_id=session_id,
            issued_at=issued_at,
            users=self._decode_users(payload.get("users")),
        )
        return state.prune_expired(now)

    @staticmethod
    def _decode_users(raw_users: Any) -> tuple[UnlockedUser, ...]:
        """Read the unlocked-user map out of a verified payload.

        Both ``tag`` and ``sub`` default to ``""`` when absent, so a cookie
        minted before either key existed decodes at the current payload version
        instead of signing that browser out.

        Raises:
            InvalidSessionError: If the map or any entry is malformed.
        """
        if raw_users is None:
            return ()
        if not isinstance(raw_users, dict):
            raise InvalidSessionError("session payload 'users' is not an object")

        users = []
        for username, entry in raw_users.items():
            if not isinstance(username, str) or not username:
                raise InvalidSessionError("session payload carries an empty username")
            if not isinstance(entry, dict):
                raise InvalidSessionError(f"session entry for {username!r} is not an object")
            tag = entry.get("tag", "")
            if not isinstance(tag, str):
                raise InvalidSessionError(f"session entry for {username!r} has a non-string tag")
            subject = entry.get("sub", "")
            if not isinstance(subject, str):
                raise InvalidSessionError(
                    f"session entry for {username!r} has a non-string subject"
                )
            users.append(
                UnlockedUser(
                    username=username,
                    expires_at=_as_epoch(entry.get("exp"), f"users[{username}].exp"),
                    generation_tag=tag,
                    oidc_subject=subject,
                )
            )
        return tuple(users)
