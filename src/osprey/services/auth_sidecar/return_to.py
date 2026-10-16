"""The one rule for where a login may send the browser afterwards.

Both login flows accept a return-to from the client — the password route reads
it off the link nginx emits and again out of a hidden form field, the OIDC route
off the same link and then back out of a decoded state cookie — and both must
hold it to the same rule. That is why the rule lives here rather than in either
route: a return-to that one flow rejects and the other honours is an open
redirect with a second front door.
"""

from __future__ import annotations

import logging
import re

from osprey.interfaces.common_middleware import url_mount_prefix

logger = logging.getLogger(__name__)

MAX_RETURN_TO_LENGTH = 2048
"""Longest return-to a login will honour, in characters.

nginx's allowlist bounds what a return-to may *contain*, not how much of it there
may be: a path is limited only by the request line, which is kilobytes. A
return-to that long is nobody's terminal — it is a ``Location`` header wide
enough to exceed the proxy's own buffer, which reaches the operator as a 502
instead of a login, and the OIDC hand-off percent-encodes it into something
several times larger again. Generous for the real thing (``/u/<user>/panel/…``)
and far below where any of that starts.
"""

_ENCODED_SEPARATOR_RE = re.compile(r"%(?:2f|5c)", re.IGNORECASE)
"""Percent-encoded ``/`` and ``\\`` in a return-to.

A path carrying them means different things to this service, to nginx's location
matching, and to the browser that finally resolves it. Nothing legitimate needs
an encoded separator in a terminal's path, so a return-to carrying one is
discarded rather than decoded.
"""


HISTORY_PATH = "/history/"
"""The one path an absolute return-to may name, under the history origin."""


def _path_rejected(candidate: str) -> bool:
    """Whether a same-origin path breaks any rule a return-to must keep."""
    return (
        # Relative and absolute-URL forms both let the browser leave this origin.
        not candidate.startswith("/")
        # "//host" and "/\host" are read by browsers as another origin.
        or candidate.startswith(("//", "/\\"))
        # A backslash anywhere is normalised to "/" by browsers, so it can
        # reconstruct one of the forms above after this check.
        or "\\" in candidate
        # Control characters (CR/LF above all) have no place in a path and are
        # how a Location header gets split.
        or any(character < " " or character == "\x7f" for character in candidate)
        or bool(_ENCODED_SEPARATOR_RE.search(candidate))
    )


def acceptable_return_to(raw: object, *, history_origin: str = "") -> str | None:
    """The return-to ``raw`` names when every rule admits it, else ``None``.

    Two shapes are admitted. A same-origin absolute path, as every login has
    always taken. And, when this deployment serves the history page, an
    absolute URL that starts with exactly ``history_origin + "/history/"`` and
    whose path passes the same rules — the readers' origin is a second port on
    this host, so the page's sign-in has to come back across it. Nothing else
    off this origin is ever admitted: the prefix is compared verbatim, so a
    look-alike host, another scheme, a userinfo part or another port fails it.

    Args:
        raw: The requested return-to; ``object`` because a decoded cookie
            payload carries whatever was stored.
        history_origin: The readers' origin, or ``""`` when there is none.

    Returns:
        The stripped return-to, or ``None`` when it is absent or refused.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = raw.strip()
    # Longer than any terminal path, and long enough to make a Location header
    # the proxy in front of this service will not carry.
    if len(candidate) > MAX_RETURN_TO_LENGTH:
        return None
    if history_origin and candidate.startswith(f"{history_origin}{HISTORY_PATH}"):
        return None if _path_rejected(candidate[len(history_origin) :]) else candidate
    return None if _path_rejected(candidate) else candidate


def safe_return_to(raw: object, user: str, *, flow: str = "login", history_origin: str = "") -> str:
    """Validate a return-to target, falling back to ``user``'s terminal.

    The value is client input on every leg it arrives on, and is validated on
    each of them. Anything that could send the browser to another origin — an
    absolute URL, a protocol-relative ``//host``, a backslash form browsers
    normalise into one — is discarded rather than repaired, as is a path
    carrying a percent-encoded separator, which this service, nginx and the
    browser would each read differently. The one absolute URL admitted is the
    history page's, under ``history_origin`` (see :func:`acceptable_return_to`).
    A rejected target is not an error to the person logging in; they came here
    to reach their terminal, so that is where they go.

    An empty value is normal and frequent, not an error: nginx collapses
    ``next`` to nothing for any path carrying a space, ``+``, ``@``, ``:``,
    ``,``, ``%`` or a non-ASCII character, so a deep link to an artifact whose
    filename has a space in it simply lands the operator on their terminal root.

    Args:
        raw: The requested return-to. Typed as ``object`` because the OIDC
            callback reads it back out of a decoded cookie payload, where its
            shape is whatever was stored rather than whatever was declared.
        user: The roster user being logged in, for the default target.
        flow: Which login the request came in on, for the rejection log line.
        history_origin: The readers' origin, or ``""`` when this deployment
            serves no history page.

    Returns:
        A same-origin absolute path or a history-page URL, safe to put in a
        ``Location`` header and in a hidden field.
    """
    default = f"{url_mount_prefix(user)}/"
    if not isinstance(raw, str) or not raw:
        return default
    accepted = acceptable_return_to(raw, history_origin=history_origin)
    if accepted is None:
        logger.warning(
            "%s for %r requested an off-origin return-to; sending them to their own "
            "terminal instead",
            flow,
            user,
        )
        return default
    return accepted
