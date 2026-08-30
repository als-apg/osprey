"""The four identity headers an authorized subrequest may carry, and their charset.

nginx turns the sidecar's ``auth_request`` answer into four forwarded headers —
:data:`ACCOUNT_HEADER` naming the roster card the request is on,
:data:`SUBJECT_HEADER` naming who proved the login,
:data:`ROLE_HEADER` naming the privilege it holds and
:data:`ROLE_SOURCE_HEADER` naming where that role came from — and every terminal
behind that boundary reads its authorization from them. All four names and the
one rule about what may travel in them live here rather than in the route,
because three layers have to agree on it: the session model (which refuses to
*store* a value that could not be carried), the verify route (which emits
them), and the OIDC callback (which refuses a *login* whose identity could not
be carried).

**ASCII only, and that is not a deferral.** An HTTP header value is latin-1 on
the wire, so ``jörg@example.org`` survives one hop and arrives at the terminal
as mojibake — or, one proxy later, as nothing. Osprey therefore requires the
stricter ASCII of every value, which costs nothing real: an OIDC ``sub`` is
ASCII by specification, and a roster username and a role name are already
constrained to ``USERNAME_CHARSET_RE`` by the render-time lint *for this
reason*. Control characters are refused on top of that, so a value can never
split a header or inject a second one, and a leading or trailing space is
refused because an HTTP parser strips it — the value that arrives would not be
the value that was authorized.

**An unsafe value is never carried, and never quietly dropped either.** Each of
the three layers refuses in the way that is closed for it: the model raises
rather than signing such a session, the decode path rejects the cookie, and the
verify route denies the subrequest. A header silently omitted from an otherwise
successful authorization would leave the terminal running with less identity
than the deployment thinks it forwarded, which is exactly the failure this
module exists to prevent.
"""

from __future__ import annotations

__all__ = [
    "ACCOUNT_HEADER",
    "ROLE_HEADER",
    "ROLE_SOURCE_HEADER",
    "SUBJECT_HEADER",
    "is_header_safe",
]

ACCOUNT_HEADER = "X-Osprey-Auth-Account"
"""Names the roster card an authorized request is on.

The account is the roster entry whose card was clicked — the name ``/verify``
checked the session against, and the one a consumer can compare with the
account it believes it is serving. :data:`SUBJECT_HEADER` answers a different
question: who proved the login. The two coincide in a password session, where
the roster username *is* the proof, and diverge in an OIDC session, where the
provider asserts an opaque id or an email that names a person and not a card.

On a shared card that difference is the whole point: the account names the
card, the subject names whoever opened it. Emitted on every authorized answer —
an authorized request always has a card — so a consumer seeing no account
header is talking to a sidecar older than this release, not to a request
without an account.
"""

SUBJECT_HEADER = "X-Osprey-Auth-Subject"
"""Names who proved the login behind an authorized request.

An OIDC session carries the provider's subject; a password session carries the
roster username, which *is* the account in that method. Emitted only when the
session holds one, so its presence always means a known identity and no
consumer has to tell an empty value from an absent one.
"""

ROLE_HEADER = "X-Osprey-Auth-Role"
"""Names the role the authorized user holds, when the session carries one.

Absent means no role, which every consumer must read as "no privileges" — never
as a default role. That is what makes the field's empty default deny-safe end to
end: an unresolved role and a refused one look the same downstream.
"""

ROLE_SOURCE_HEADER = "X-Osprey-Auth-Role-Source"
"""Names where the role in :data:`ROLE_HEADER` came from, when one is carried.

``roster`` means the roster's ``role:`` entry resolved it; ``claim`` means the
OIDC ID token's role claim did. Emitted only beside a role and absent whenever
the role is absent, so provenance can never name the origin of a privilege the
session does not hold. Consumers read it for display, never as authorization:
the role is what grants, and where it came from changes nothing about that.
"""

_LOWEST_PRINTABLE = 0x20
"""Space. Everything below it is a control character, newlines included."""

_HIGHEST_PRINTABLE = 0x7E
"""``~``. ``0x7F`` is DEL and everything above it is outside ASCII."""


def is_header_safe(value: str) -> bool:
    """Whether ``value`` can cross the nginx boundary unchanged.

    Args:
        value: The candidate subject or role.

    Returns:
        ``True`` only for a non-empty ASCII value made of printable characters,
        with no leading or trailing space. Empty is ``False``: the callers all
        treat "no value" as its own case (omit the header, name no role) and
        must not reach this asking whether nothing is carryable.
    """
    if not value:
        return False
    if value[0] == " " or value[-1] == " ":
        return False
    return all(_LOWEST_PRINTABLE <= ord(char) <= _HIGHEST_PRINTABLE for char in value)
