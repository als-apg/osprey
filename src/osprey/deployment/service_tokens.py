"""Secret-generation and format-validation recipes for deploy-time service tokens.

This module owns the per-variable *policy* half of service-token provisioning:
how to mint a secret for a given env var (the alphabet/entropy recipe) and how
to validate an effective value against the downstream consumer's parsing rules.
The deploy-time *orchestration* — which services require which vars, the
write-arming safety gate, and appending minted values to the project ``.env`` —
lives in :mod:`osprey.deployment.container_lifecycle`, which imports the recipes
below. Splitting the recipes out keeps the deterministic, side-effect-free
generation/validation logic testable in isolation from the provisioning flow.
"""

import os
import secrets
import string
from collections.abc import Callable
from urllib.parse import unquote, urlsplit


def _default_token() -> str:
    """The default secret recipe, also the one ``.env.template`` documents."""
    return secrets.token_urlsafe(32)


def _generate_openobserve_password() -> str:
    """Mint a ``ZO_ROOT_USER_PASSWORD`` that satisfies OpenObserve's policy.

    OpenObserve refuses to start unless the root password is 8–128 characters
    with at least one lowercase letter, one uppercase letter, one digit, and
    one special (non-alphanumeric) character — otherwise the container
    crash-loops at startup. ``_default_token``'s ``token_urlsafe`` draws from
    ``[A-Za-z0-9_-]``, which carries no character a strict policy counts as
    "special", so that recipe crash-loops the container non-deterministically:
    the same class of failure as ``BLUESKY_TILED_API_KEY``'s Tiled-alphabet
    constraint (see ``_VAR_GENERATORS`` below).

    Build a value that guarantees all four required classes instead, drawing
    every character from ``secrets`` (never ``random``): a 44-char alphanumeric
    core (>=256 bits of entropy on its own, meeting the module's CSPRNG bar)
    plus one guaranteed member of each class, then shuffled so the class
    positions are not fixed. The special is drawn from ``@%*^`` — punctuation
    every reasonable policy counts as "special", and each of which is safe both
    in a ``.env`` value (unlike ``#``, ``$``, quotes, backslash, ``=`` or a
    space, which break dotenv parsing) and in the base64 Basic-auth header the
    resolver computes from it.
    """
    alphabet = string.ascii_letters + string.digits
    chars = [secrets.choice(alphabet) for _ in range(44)]
    chars += [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits),
        secrets.choice("@%*^"),
    ]
    secrets.SystemRandom().shuffle(chars)
    return "".join(chars)


# Per-variable overrides of the default recipe. A var absent here gets
# ``_default_token``.
#
# CSPRNG bar: any recipe registered here MUST draw from ``secrets`` (never
# ``random``, a hashed timestamp, or any other non-cryptographic source) and
# MUST yield at least 256 bits of entropy — the same bar ``_default_token``'s
# ``token_urlsafe(32)`` meets. A registered recipe exists to change the
# *alphabet* for a downstream consumer's parsing rules (see
# BLUESKY_TILED_API_KEY below), never to weaken the randomness.
#
# BLUESKY_TILED_API_KEY: Tiled validates its ``--api-key`` during server startup
# and raises ``ValueError("The API key must only contain alphanumeric
# characters")`` for anything else, so a rejected key makes the container exit
# before it ever listens. ``token_urlsafe``'s alphabet includes ``-`` and ``_``,
# which land in roughly 7 of 10 values, so that recipe crash-loops Tiled on most
# deploys — non-deterministically. ``token_hex(32)`` draws from ``[0-9a-f]``:
# alphanumeric by construction, and the same 256 bits of entropy.
#
# Generate from an alphanumeric alphabet rather than stripping ``-``/``_`` out of
# a urlsafe value, which would shorten the secret by a variable amount and drop
# entropy silently.
_VAR_GENERATORS: dict[str, Callable[[], str]] = {
    "BLUESKY_TILED_API_KEY": lambda: secrets.token_hex(32),
    # OpenObserve rejects a root password that misses any of its four required
    # character classes and crash-loops — see _generate_openobserve_password.
    "ZO_ROOT_USER_PASSWORD": _generate_openobserve_password,
}


def _generate_token(var: str) -> str:
    """Mint one secret for ``var`` using its registered recipe."""
    return _VAR_GENERATORS.get(var, _default_token)()


def _validate_ariel_dsn(value: str) -> bool:
    """True if ``value`` parses as a URI whose password is cleanly encoded.

    Ports pySC's discipline of never trusting a hand-assembled connection
    string (F3): a DSN's password segment sits between ``:`` and ``@`` in the
    URI's authority component, so an *unescaped* reserved character (``@ : /
    ? #``) inside the password either steals characters from the wrong field
    (an unescaped ``/`` truncates the authority early, eating the host into
    the path — caught below by the missing/wrong ``hostname``) or silently
    changes what the URI means without raising a parse error. Requiring every
    reserved character to appear only in its ``%XX`` form, and requiring that
    form to percent-decode without error, is what "parses cleanly" means here.
    """
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.hostname:
        return False
    try:
        _ = parsed.port
    except ValueError:
        # An unescaped reserved character in the password (/ ? #) truncates
        # the authority component early, leaving a non-numeric fragment where
        # the port belongs — the tell that the real host was swallowed into
        # the path/query/fragment instead of being parsed as part of netloc.
        # (``.hostname`` alone does not catch this: the truncated netloc
        # still yields a plausible-looking, but wrong, hostname.)
        return False
    password = parsed.password
    if password is None:
        return True
    if any(reserved in password for reserved in "@:/?#"):
        return False
    try:
        unquote(password, errors="strict")
    except UnicodeDecodeError:
        return False
    return True


def _validate_openobserve_password(value: str) -> bool:
    """True if ``value`` satisfies OpenObserve's root-password policy.

    OpenObserve refuses to start unless ``ZO_ROOT_USER_PASSWORD`` is 8–128
    characters with at least one lowercase letter, one uppercase letter, one
    digit, and one special (non-alphanumeric) character — a non-conforming
    value crash-loops the container at startup with an OpenObserve-internal
    error. Rejecting it here turns that opaque crash-loop into a clear
    deploy-time failure for an *operator-supplied* password (a minted one
    already conforms — see ``_generate_openobserve_password``), mirroring the
    ``BLUESKY_TILED_API_KEY``/Tiled-alphabet check.
    """
    if not 8 <= len(value) <= 128:
        return False
    return (
        any(c.islower() for c in value)
        and any(c.isupper() for c in value)
        and any(c.isdigit() for c in value)
        and any(not c.isalnum() for c in value)
    )


# Per-variable validators applied to the *effective* value of a required var
# at the deploy boundary (see ``_ensure_service_tokens``), regardless of
# whether that value was freshly minted, carried over from an existing
# ``.env``, supplied by the operator, or overridden in the process
# environment. A var absent from this map has no registered constraint and
# ``_validate_var`` returns True for it — this fails OPEN, the deliberate
# inverse of the ``_LOCAL_EXEC_SAFE_VARS`` allowlist in container_lifecycle.
#
# ``_LOCAL_EXEC_SAFE_VARS`` fails CLOSED on an unenumerated var because
# minting there is a privilege grant — the bar for opting a var *out* of that
# safety restriction must be high. Validating a var nobody has triaged yet is
# the opposite kind of decision: opting a var *into* a format constraint is
# additive hardening, not a prerequisite the deploy must clear, so withholding
# it by default must not block an otherwise-working deploy. Adding an entry
# here is opt-in per var, exactly like ``_VAR_GENERATORS``.
_VAR_VALIDATORS: dict[str, Callable[[str], bool]] = {
    # Tiled rejects a non-alphanumeric --api-key at startup (see
    # _VAR_GENERATORS above); reject it here too so an *operator-supplied*
    # key (never minted, so _VAR_GENERATORS never runs on it) fails at deploy
    # time instead of crash-looping the Tiled container.
    "BLUESKY_TILED_API_KEY": str.isalnum,
    "ARIEL_DSN": _validate_ariel_dsn,
    # OpenObserve crash-loops on a root password that misses any required
    # character class; validate an operator-supplied value at deploy time.
    "ZO_ROOT_USER_PASSWORD": _validate_openobserve_password,
}

# Human-readable constraint text shown in the RuntimeError _ensure_service_tokens
# raises on a _VAR_VALIDATORS failure — never the offending value itself. A
# var validated with no entry here falls back to a generic description.
_VAR_VALIDATOR_DESCRIPTIONS: dict[str, str] = {
    "BLUESKY_TILED_API_KEY": (
        "must be alphanumeric — Tiled rejects any other character in --api-key at startup"
    ),
    "ARIEL_DSN": (
        "must parse as a URI whose password contains no unescaped reserved "
        "character (@ : / ? #); percent-encode the password"
    ),
    "ZO_ROOT_USER_PASSWORD": (
        "must be 8–128 characters with at least one lowercase letter, one "
        "uppercase letter, one digit, and one special character — OpenObserve "
        "rejects any weaker root password at startup"
    ),
}


def _effective_value(var: str, dotenv: dict[str, str]) -> str:
    """The value ``_ensure_service_tokens`` treats as authoritative for ``var``.

    Process env wins over ``.env`` (matches ``docker compose --env-file``);
    absent from both yields ``""``.
    """
    return os.environ.get(var, dotenv.get(var, ""))


def _validate_var(var: str, value: str) -> bool:
    """Check ``value`` against ``var``'s registered constraint, if any.

    Returns True (pass) for any var with no registered validator in
    ``_VAR_VALIDATORS`` — see that dict's docstring for the fail-open
    rationale.
    """
    validator = _VAR_VALIDATORS.get(var)
    if validator is None:
        return True
    return validator(value)


def _raise_invalid_var(var: str) -> None:
    """Raise the standard "invalid var" RuntimeError, never the value."""
    constraint = _VAR_VALIDATOR_DESCRIPTIONS.get(
        var, "does not satisfy its registered format constraint"
    )
    raise RuntimeError(f"{var} is invalid: {constraint}. Refusing to deploy. (Value not shown.)")
