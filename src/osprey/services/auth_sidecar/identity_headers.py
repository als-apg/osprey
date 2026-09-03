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

The module also owns :func:`same_value`, the constant-time comparator every
check of an identity value uses, and :func:`same_identity`, the one rule for
when a *mapped identity* may match under a different spelling — an ``email``
claim is a mailbox, and a mailbox does not change with its case.
"""

from __future__ import annotations

import secrets

__all__ = [
    "ACCOUNT_HEADER",
    "ROLE_HEADER",
    "ROLE_SOURCE_HEADER",
    "SUBJECT_HEADER",
    "CASE_INSENSITIVE_CLAIMS",
    "is_header_safe",
    "same_identity",
    "same_value",
]

ACCOUNT_HEADER = "X-Osprey-Auth-Account"
"""Names the roster card an authorized request is on.

The account is the roster entry whose card was clicked — the name ``/verify``
checked the session against, and the one a consumer can compare with the
account it believes it is serving. :data:`SUBJECT_HEADER` answers a different
question: who proved the login. The two coincide on an own card under either
login method and diverge on a shared card under both.

On a shared card that difference is the whole point: the account names the
card, the subject names whoever opened it. Emitted on every authorized answer —
an authorized request always has a card — so a consumer seeing no account
header is talking to a sidecar older than this release, not to a request
without an account.
"""

SUBJECT_HEADER = "X-Osprey-Auth-Subject"
"""Names who proved the login behind an authorized request.

An OIDC session carries the provider's subject; a password session carries the
roster username of whoever proved the login. Under either method the value
matches the account on an own card and differs from it on a shared card.
Emitted only when the session holds one, so its presence always means a known
identity and no consumer has to tell an empty value from an absent one.
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


def same_value(left: str, right: str) -> bool:
    """Whether two text values are equal, compared in constant time.

    As UTF-8 bytes, never as ``str``: :func:`secrets.compare_digest` raises
    ``TypeError`` the moment either side carries a non-ASCII character, and both
    things compared here can. A ``state`` parameter is unauthenticated input, so
    one accented character in it would be an unhandled 500 rather than a
    refusal; a mapped identity like ``jörg@example.org`` would raise on the
    comparison that was about to *succeed*, locking that operator out for good.
    """
    return secrets.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


CASE_INSENSITIVE_CLAIMS: frozenset[str] = frozenset({"email"})
"""The ID-token claims whose values name a mailbox, matched without regard to case.

``email`` is the one claim OpenID Connect defines as an RFC 5322 address, and
an address is the same mailbox in any case: the domain by RFC 5321, the local
part by every mail provider in practice — the same person is
``THellert@lbl.gov`` in a directory and ``thellert@lbl.gov`` in daily use, and
a provider releases whichever spelling it stores. A byte-exact match would
couple a roster to that cosmetic choice, refusing every login with a 403 that
names nothing an operator can see in the config.

Nothing else is on the list, and that is deliberate. ``sub`` is a
case-sensitive opaque identifier by specification, so two subjects differing
only in case are two accounts, and ``preferred_username`` is a display value
the specification tells a relying party not to lean on at all — a deployment
that maps on it gets the exact compare and can move to ``email`` when its
provider spells that one unpredictably.
"""


def same_identity(asserted: str, configured: str, *, claim: str) -> bool:
    """Whether an asserted identity matches a roster mapping, under ``claim``'s rule.

    The one comparator for a mapped identity: the callback's own-card and
    shared-card branches and ``/verify``'s opener re-validation all go through
    it, so the three cannot disagree about what "the same identity" means.

    Args:
        asserted: The value the identity provider asserted (or, on
            re-validation, the value a session recorded).
        configured: The roster entry's mapped value.
        claim: The ID-token claim the deployment maps on.

    Returns:
        For a claim in :data:`CASE_INSENSITIVE_CLAIMS`, whether the two are the
        same mailbox — both sides casefolded, then compared in constant time.
        For every other claim, exactly :func:`same_value`.
    """
    if claim in CASE_INSENSITIVE_CLAIMS:
        return same_value(asserted.casefold(), configured.casefold())
    return same_value(asserted, configured)
