"""The two identity headers an authorized subrequest may carry, and their charset.

nginx turns the sidecar's ``auth_request`` answer into two forwarded headers —
:data:`SUBJECT_HEADER` naming the account behind the request and
:data:`ROLE_HEADER` naming the privilege it holds — and every terminal behind
that boundary reads its authorization from them. Both names and the one rule
about what may travel in them live here rather than in the route, because three
layers have to agree on it: the session model (which refuses to *store* a value
that could not be carried), the verify route (which emits them), and the OIDC
callback (which refuses a *login* whose identity could not be carried).

**ASCII only, and that is not a deferral.** An HTTP header value is latin-1 on
the wire, so ``jörg@example.org`` survives one hop and arrives at the terminal
as mojibake — or, one proxy later, as nothing. Osprey therefore requires the
stricter ASCII of both values, which costs nothing real: an OIDC ``sub`` is
ASCII by specification, and a role name is already constrained to
``USERNAME_CHARSET_RE`` by the render-time lint *for this reason*. Control
characters are refused on top of that, so a value can never split a header or
inject a second one, and a leading or trailing space is refused because an HTTP
parser strips it — the value that arrives would not be the value that was
authorized.

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
    "ROLE_HEADER",
    "SUBJECT_HEADER",
    "is_header_safe",
]

SUBJECT_HEADER = "X-Osprey-Auth-Subject"
"""Names the account behind an authorized request.

An OIDC session carries the provider's subject; a password session carries the
roster username, which *is* the account in that method. Emitted only when the
session holds one, so its presence always means a known account and no consumer
has to tell an empty value from an absent one.
"""

ROLE_HEADER = "X-Osprey-Auth-Role"
"""Names the role the authorized user holds, when the session carries one.

Absent means no role, which every consumer must read as "no privileges" — never
as a default role. That is what makes the field's empty default deny-safe end to
end: an unresolved role and a refused one look the same downstream.
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
