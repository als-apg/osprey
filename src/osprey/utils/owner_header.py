"""The one reader of the ``X-Osprey-Owner`` request header.

Two services learn who queued a piece of work from the same header: the
Bluesky bridge, on the add path that stamps an owner onto a queue item, and
the event dispatcher, in ``manual_fire`` and the dashboard's retry route. The
header is minted in-container — the terminal proxy and the auth sidecar's own
gate both strip any inbound value and set their own — so by the time it
reaches either reader it is expected to name a roster account. "Expected" is
not "guaranteed": a direct call to an open service port carries whatever a
caller typed, and the CLI leaves an unexpanded ``${OSPREY_TERMINAL_USER}``
literal behind when that variable is unset.

So both readers need the same answer to the same question, and this module is
where that question is answered once. A second copy of the rule would drift,
and the drift would be invisible: the two readers would accept different sets
of names and attribute the same forged header differently.

The accepted shape is the roster username charset, ``[A-Za-z0-9._-]``, bounded
at :data:`MAX_OWNER_LENGTH` characters. That is deliberately narrower than the
audit ladder's rule in ``osprey.utils.identity``, which rejects only what
breaks path semantics: an identity there is a local account that already
exists, while this value arrives over the wire, is written into queue metadata
and run records, and is read back by a browser. An allowlist is the right
shape for a value with that reach.

A refused value yields ``None`` — owner-less, never a raised exception. A
malformed header must not fail a queue add or a trigger fire; losing the
attribution is the whole cost. Each refusal logs exactly one warning naming
the *shape* that was refused and never the value itself, because the value is
attacker-chosen text and an operator's log is not a place to render it. An
absent header (``None``) is the ordinary owner-less case — cron fires jobs
that way — and logs nothing at all.

This module imports only the standard library, so the bridge, the dispatcher
and the leaf utilities can all depend on it without an import cycle.
"""

import logging
import re

logger = logging.getLogger(__name__)

#: The header's wire spelling. Callers import it rather than restating the
#: string, so the minting sites and the reading sites cannot disagree about
#: capitalisation-insensitive lookups or about the name itself.
OWNER_HEADER: str = "X-Osprey-Owner"

#: The longest accepted name. Matches the roster username bound: a deployment
#: names its accounts, and a value longer than this is not one of them.
MAX_OWNER_LENGTH: int = 64

#: The accepted charset, anchored and bounded. Letters, digits, dot, underscore
#: and hyphen — the characters a roster username is built from.
_OWNER_RE = re.compile(rf"\A[A-Za-z0-9._-]{{1,{MAX_OWNER_LENGTH}}}\Z")

# Characters that would let a name escape its own component or split into
# several. Checked before the charset so the warning can say "path" rather than
# the less useful "charset".
_PATH_SEPARATORS: tuple[str, ...] = ("/", "\\", "\0")

# Names that are a single path component syntactically but resolve elsewhere.
# Both match the charset above, so they need naming explicitly.
_RESERVED_NAMES: tuple[str, ...] = (".", "..")

# The marker of a shell placeholder the CLI never expanded. Its own characters
# would fail the charset anyway; matching it first buys a warning that tells an
# operator which mistake they actually made.
_PLACEHOLDER_MARKER: str = "${"


def _refuse(shape: str) -> None:
    """Log the one warning a refusal is allowed, naming *shape* only."""
    logger.warning(
        "Ignoring %s: the value is %s, so this request is recorded owner-less. "
        "The value itself is not logged.",
        OWNER_HEADER,
        shape,
    )


def owner_from_header(value: str | None) -> str | None:
    """Return the owner *value* names, or ``None`` when it names nobody.

    *value* is the raw header as received. ``None`` means the header was
    absent, which is a legitimate owner-less call and passes silently. Any
    other value that does not match the accepted shape is refused with one
    warning; see the module docstring for why refusal is never an exception.

    The returned string is the header verbatim. Nothing is stripped or
    case-folded: a name that needed trimming to be accepted is not the name
    the deployment issued, and quietly repairing it would attribute work to an
    account whose spelling nobody chose.
    """
    if value is None:
        return None

    if not isinstance(value, str):
        _refuse("not a string")
        return None

    if not value.strip():
        _refuse("empty or whitespace-only")
        return None

    if _PLACEHOLDER_MARKER in value:
        _refuse("an unexpanded ${...} placeholder rather than a name")
        return None

    if len(value) > MAX_OWNER_LENGTH:
        _refuse(f"longer than the {MAX_OWNER_LENGTH}-character limit")
        return None

    if value in _RESERVED_NAMES or any(sep in value for sep in _PATH_SEPARATORS):
        _refuse("a path fragment rather than a single name")
        return None

    if not _OWNER_RE.match(value):
        _refuse("outside the accepted charset [A-Za-z0-9._-]")
        return None

    return value
