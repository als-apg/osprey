"""Secret-generation and format-validation recipes for deploy-time service tokens.

This module owns the per-variable *policy* half of service-token provisioning:
how to mint a secret for a given env var (the alphabet/entropy recipe) and how
to validate an effective value against the downstream consumer's parsing rules.
The deploy-time *orchestration* — which services require which vars, and
appending minted values to the project ``.env`` — lives in
:mod:`osprey.deployment.container_lifecycle`, which imports the recipes below.
Splitting the recipes out keeps the deterministic, side-effect-free
generation/validation logic testable in isolation from the provisioning flow.
"""

import os
import secrets
import string
from collections.abc import Callable
from urllib.parse import unquote, urlsplit


def _default_token() -> str:
    """The default secret recipe, also the one ``.env.example`` documents."""
    return secrets.token_urlsafe(32)


# Characters excluded from the OpenObserve password alphabet because a human
# reads this one off a screen: ``l``/``I``/``1`` and ``O``/``0`` are the pairs
# people transcribe wrongly, and a rejected OpenObserve login says nothing about
# which character was misread.
_AMBIGUOUS_CHARS = "lI1O0"

#: Length of a minted ``ZO_ROOT_USER_PASSWORD``. See
#: :func:`_generate_openobserve_password` for why this one credential is short.
_OPENOBSERVE_PASSWORD_LENGTH = 12


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
    every character from ``secrets`` (never ``random``), then shuffling so the
    class positions are not fixed. The special is drawn from ``@%*^`` —
    punctuation every reasonable policy counts as "special", and each of which
    is safe both in a ``.env`` value (unlike ``#``, ``$``, quotes, backslash,
    ``=`` or a space, which break dotenv parsing) and in the base64 Basic-auth
    header the resolver computes from it.

    **Why this recipe is deliberately short.** It is the one minted credential
    with a human in the loop: every other secret in ``_VAR_GENERATORS`` travels
    machine-to-machine, but this one is read off a terminal and typed into a
    browser login form, often on a projector during a demo. At 12 characters
    from an alphabet with the easily-misread characters removed it carries ~65
    bits — under the module's 256-bit default bar, and that is the point, not
    an oversight. That bar is sized for secrets an attacker can attack offline;
    the only attack this one faces is online guessing against an HTTP login
    form, where 65 bits of CSPRNG entropy is many orders of magnitude out of
    reach. A character count is the wrong instrument here in any case — it
    would reject this value while admitting a longer, guessable one an operator
    typed. What actually disqualifies a password is being absent or being
    public, and both are refused on their own terms: the empty-value check in
    ``_ensure_service_tokens`` under exposure, and ``_VAR_FORBIDDEN_VALUES``
    for the template's published default.
    """
    lower = "".join(c for c in string.ascii_lowercase if c not in _AMBIGUOUS_CHARS)
    upper = "".join(c for c in string.ascii_uppercase if c not in _AMBIGUOUS_CHARS)
    digits = "".join(c for c in string.digits if c not in _AMBIGUOUS_CHARS)
    specials = "@%*^"

    # One guaranteed member of each required class, then fill to length from the
    # union so the class counts are not themselves a fixed pattern.
    chars = [
        secrets.choice(lower),
        secrets.choice(upper),
        secrets.choice(digits),
        secrets.choice(specials),
    ]
    pool = lower + upper + digits + specials
    chars += [secrets.choice(pool) for _ in range(_OPENOBSERVE_PASSWORD_LENGTH - len(chars))]
    secrets.SystemRandom().shuffle(chars)
    return "".join(chars)


# Per-variable overrides of the default recipe. A var absent here gets
# ``_default_token``.
#
# CSPRNG bar: every recipe registered here MUST draw from ``secrets`` (never
# ``random``, a hashed timestamp, or any other non-cryptographic source). That
# half is absolute and has no exceptions.
#
# Entropy bar: a recipe SHOULD yield at least 256 bits — the same bar
# ``_default_token``'s ``token_urlsafe(32)`` meets. Ordinarily a registered
# recipe exists to change the *alphabet* for a downstream consumer's parsing
# rules (see BLUESKY_TILED_API_KEY below), never to weaken the randomness.
#
# ONE exception is allowed, and only on this ground: a credential a HUMAN types
# into a login form may trade entropy for legibility, because a password nobody
# can transcribe is a password that gets pasted into a chat window or replaced
# with something worse. A recipe claiming the exception must say so in its
# docstring and must still draw every character from ``secrets``: legibility
# buys a shorter value, never a weaker source. ZO_ROOT_USER_PASSWORD is the only
# var that qualifies today: the Tiled key, both dispatch tokens, and the
# Postgres/Mongo passwords are read by processes, never by people.
#
# A DELIBERATE ABSENCE, recorded because the omission looks like an oversight:
# ``ZO_INGEST_SA_TOKEN`` — the OpenObserve telemetry ingest identity's secret —
# has no recipe here and must never be given one. That value is issued by the
# STORE (a 16-character alphanumeric token, read back from the live container
# once it is up), not chosen by osprey. A recipe registered here would fabricate
# a token the store never issued: the ``.env`` would look healthy and every
# telemetry request would 401. See ``_STORE_ISSUED_VARS`` in
# ``container_lifecycle`` for where such a var IS registered, and why.
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
    # The ARIEL Postgres password is substituted into the DSN URI that
    # ``resolve_ariel_dsn`` derives from ``services.postgresql``
    # (``postgresql://ariel:${ARIEL_DB_PASSWORD:-ariel}@…``) as well as the
    # container's POSTGRES_PASSWORD, so it must stay free of URI-reserved
    # characters — ``token_urlsafe``'s ``-``/``_`` would be fine, but hex is
    # alphanumeric by construction and matches the Tiled recipe: same 256
    # bits of entropy, zero escaping concerns in .env, YAML, or the URI.
    "ARIEL_DB_PASSWORD": lambda: secrets.token_hex(32),
    # The archiver Mongo root password follows the ARIEL_DB_PASSWORD rationale.
    # The connector passes it to ``MongoClient`` as a keyword argument, where
    # escaping would not matter — but it is not the only consumer: the recorder
    # and the scenario seeder connect to the same store, the compose
    # healthcheck reaches it through ``mongosh``, and any of those may spell the
    # connection as a ``mongodb://user:pass@host`` URI. Hex is alphanumeric by
    # construction, so every spelling is safe unescaped, with the same 256 bits
    # of entropy as the default recipe.
    "MONGO_ROOT_PASSWORD": lambda: secrets.token_hex(32),
    # The graph store's password follows the same hex rationale one step
    # further. Neo4j takes its initial credentials as the COMPOSITE
    # ``NEO4J_AUTH: "neo4j/<password>"``, which the container splits on the
    # first ``/``: a value carrying one hands the container a different user
    # and a truncated password rather than failing loudly. Neo4j also refuses
    # an initial password under 8 characters. Hex satisfies both by
    # construction — no ``/`` in the alphabet, 64 characters long — with the
    # same 256 bits of entropy, and stays safe unescaped in the bolt URI a
    # health check or an operator may spell by hand.
    "GRAPHDB_PASSWORD": lambda: secrets.token_hex(32),
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


def _validate_uri_safe_password(value: str) -> bool:
    """True if ``value`` can sit unescaped in a URI's password slot.

    The password half of :func:`_validate_ariel_dsn`, applied to a bare secret
    rather than to an assembled URI: a value carrying a URI-reserved character
    (``@ : / ? #``) or whitespace reshapes any connection string it is
    substituted into, silently and without a parse error. Shared by every
    minted password that some consumer may spell as a URI, so the character
    class is stated once — two copies of a rule this subtle drift apart.
    """
    if not value:
        return False
    return not any(c in value for c in "@:/?#") and not any(c.isspace() for c in value)


#: Neo4j's own floor on an initial password: the container refuses anything
#: shorter and crash-loops at start. Named rather than inlined because the
#: operator-facing description below has to quote the same number.
_GRAPHDB_PASSWORD_MIN_LENGTH = 8


def _validate_graphdb_password(value: str) -> bool:
    """True if ``value`` is safe as the password half of ``NEO4J_AUTH``.

    Two constraints, from two different places, and an operator can trip either
    one alone:

    * **The composite.** The graphdb compose template passes credentials as
      ``NEO4J_AUTH: "neo4j/${GRAPHDB_PASSWORD:-…}"`` — one string the container
      splits on ``/``. A password containing ``/`` therefore moves the split:
      the container is initialized as a *different user* with a *truncated*
      password, and nothing in the value is malformed enough for anything to
      say so. That is a strictly worse failure than a refusal, because the
      store comes up and the operator's own credentials then do not work.
    * **The length floor.** Neo4j rejects an initial password under
      ``_GRAPHDB_PASSWORD_MIN_LENGTH`` characters and crash-loops at container
      start, in the ``BLUESKY_TILED_API_KEY``/``ZO_ROOT_USER_PASSWORD`` shape:
      an opaque restart loop that says nothing about ``.env``.

    The ``/`` check is stated here even though
    :func:`_validate_uri_safe_password` already rejects it as a URI-reserved
    character. That is deliberate and not redundancy for its own sake: the two
    rules rest on different facts — one on how a URI's authority component
    parses, the other on how this one container splits one variable — and the
    shared helper is free to change its character class for URI reasons that
    have nothing to do with the composite. Restating it keeps the composite's
    safety from depending on an unrelated helper's future.
    """
    if len(value) < _GRAPHDB_PASSWORD_MIN_LENGTH:
        return False
    if "/" in value:
        return False
    return _validate_uri_safe_password(value)


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

    **Scoped to the ROOT password, and to nothing else OpenObserve holds.** This
    is a policy the store enforces on a value osprey mints, so it grades an
    osprey-generated secret. It must not be pointed at ``ZO_INGEST_SA_TOKEN``:
    that token is issued by the store as 16 alphanumeric characters, which
    carries no special character and so fails this check outright — the deploy
    would refuse the very credential the store just handed it. Validate what
    osprey generates; record what the store issues.
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
# environment. A var absent from this map has no registered *format* constraint
# and ``_validate_var`` returns True for it — deliberately fail-open. Opting a
# var *into* a format constraint is additive hardening (it turns a downstream
# crash-loop into a clear deploy-time error), not a prerequisite the deploy
# must clear, so withholding it by default must not block an otherwise-working
# deploy. Adding an entry here is opt-in per var, exactly like
# ``_VAR_GENERATORS``.
#
# The one rule this fail-open posture does NOT cover is ``$``, which
# ``_validate_var`` applies to every var ahead of this map: a ``$`` is
# corrupted by how compose loads the file, not by what any one service
# accepts, so it cannot be a per-var opt-in. See ``_validate_var``.
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
    # The value lands unescaped inside the ariel DSN's password slot, so an
    # operator-supplied value containing a URI-reserved character would
    # silently reshape the DSN (see _validate_ariel_dsn) — reject it at the
    # deploy boundary instead.
    "ARIEL_DB_PASSWORD": _validate_uri_safe_password,
    # Same character rule, different consumer: the archiver Mongo password is
    # read by the recorder, the seeder, and the agent's connector, at least one
    # of which may assemble a mongodb:// URI around it.
    "MONGO_ROOT_PASSWORD": _validate_uri_safe_password,
    # The URI-safe rule plus the two constraints Neo4j adds on top of it: the
    # value is the password half of the composite NEO4J_AUTH, and Neo4j has a
    # minimum length. See _validate_graphdb_password.
    "GRAPHDB_PASSWORD": _validate_graphdb_password,
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
    "ARIEL_DB_PASSWORD": (
        "must be non-empty with no whitespace and no URI-reserved character "
        "(@ : / ? #) — the value is substituted into the ariel DSN's password slot"
    ),
    "MONGO_ROOT_PASSWORD": (
        "must be non-empty with no whitespace and no URI-reserved character "
        "(@ : / ? #) — the archiver store's clients may spell the connection "
        "as a mongodb:// URI carrying this value unescaped"
    ),
    "GRAPHDB_PASSWORD": (
        f"must be at least {_GRAPHDB_PASSWORD_MIN_LENGTH} characters with no "
        "whitespace and no URI-reserved character (@ : / ? #) — the graph store "
        "receives it as the password half of the composite NEO4J_AUTH "
        "('neo4j/<password>'), which the container splits on '/', and Neo4j "
        f"refuses an initial password shorter than {_GRAPHDB_PASSWORD_MIN_LENGTH} "
        "characters"
    ),
}


# Values a var must never hold, whatever its format says. Keyed and opted into
# exactly like ``_VAR_VALIDATORS``: a var absent here has no forbidden value and
# ``_is_forbidden_value`` returns False for it.
#
# A separate map rather than another validator, because it answers a different
# question. The openobserve compose template's fallback satisfies
# ``_validate_openobserve_password`` completely — it carries all four required
# character classes — so no *format* rule can ever catch it. What disqualifies
# it is that it is published: it ships in the template every rendered project
# carries, which makes it a shared password rather than a secret, guarding a
# store that holds full agent conversation transcripts.
#
# Deliberately narrow, and there is no build-time twin of this check. On a fresh
# repo the template default is what the ``${VAR:-default}`` form resolves to
# transiently, and the very next step mints a real value — refusing there would
# fire on the ordinary first deploy. By the time this map is consulted (the
# post-mint validation loop in ``_ensure_service_tokens``) the effective value is
# either something just minted or something an operator supplied, so a match
# means the operator pinned the published default as their actual password.
_VAR_FORBIDDEN_VALUES: dict[str, frozenset[str]] = {
    # The ``${ZO_ROOT_USER_PASSWORD:-…}`` fallback in
    # ``osprey/templates/services/openobserve/docker-compose.yml.j2``.
    "ZO_ROOT_USER_PASSWORD": frozenset({"Complexpass#123"}),
    # The password half of the ``NEO4J_AUTH: "neo4j/${GRAPHDB_PASSWORD:-…}"``
    # fallback in ``osprey/templates/services/graphdb/docker-compose.yml.j2``.
    # Registered as the BARE password, not the composite, because that is what
    # the variable holds: an operator writes ``GRAPHDB_PASSWORD=ospreygraph``
    # into ``.env``, never the ``neo4j/`` prefix, so a composite entry here
    # would match nothing an operator can actually type and the refusal would
    # silently never fire.
    "GRAPHDB_PASSWORD": frozenset({"ospreygraph"}),
}

# The ``_VAR_VALIDATOR_DESCRIPTIONS`` twin for _VAR_FORBIDDEN_VALUES: the
# operator-facing constraint text shown in the RuntimeError
# ``_ensure_service_tokens`` raises on a forbidden value — never the value
# itself. Two maps rather than one because a var can fail either way and the two
# failures have different fixes; sharing one entry would send an operator
# reading about character classes when the problem is that their password is
# public. A var with a forbidden value but no entry here falls back to a generic
# description.
_VAR_FORBIDDEN_DESCRIPTIONS: dict[str, str] = {
    "ZO_ROOT_USER_PASSWORD": (
        "must not be the default published in the openobserve compose template — "
        "that value ships in every rendered project, so it is a shared password "
        "rather than a secret, and this store holds full agent conversation "
        "transcripts; remove the line from .env (and unset any shell export of the "
        "same name) so the next run mints a per-deploy value"
    ),
    "GRAPHDB_PASSWORD": (
        "must not be the default published in the graphdb compose template — "
        "that value ships in every rendered project, so it is a shared password "
        "rather than a secret, and it guards the facility knowledge graph the "
        "agent answers from; remove the line from .env (and unset any shell "
        "export of the same name) so the next run mints a per-deploy value"
    ),
}


def _effective_value(var: str, dotenv: dict[str, str]) -> str:
    """The value ``_ensure_service_tokens`` treats as authoritative for ``var``.

    Reads process env first, then ``dotenv``, then ``""``. On the CLI deploy
    path the project ``.env`` has already been loaded into the process env with
    override (``ConfigBuilder`` treats the file as the source of truth), so a
    shell export is only visible here when the ``.env`` does not carry the key
    — the derived file, not the ambient shell, decides a present key's value.
    """
    return os.environ.get(var, dotenv.get(var, ""))


def _validate_var(var: str, value: str) -> bool:
    """Check ``value`` against the universal ``$`` rule, then ``var``'s own.

    Two layers, and the order matters. Every var here ends up in the project
    ``.env``, which the dispatch worker receives *in its entirety* via
    ``env_file: ../../.env`` — so a ``$`` in any of them is truncated on the way
    into the container regardless of which var it is. That check is therefore
    universal and runs first, ahead of the per-var lookup.

    Per-*format* constraints stay opt-in and fail-open, as ``_VAR_VALIDATORS``
    documents: a var absent from that dict still passes. Only the ``$`` rule is
    unconditional, because it is a property of how compose loads the file rather
    than of what any one service accepts. The registered validators that
    reason about character safety — ``_validate_ariel_dsn`` and
    ``_validate_uri_safe_password``, which reject the URI-reserved ``@:/?#``, and
    ``_validate_openobserve_password``, whose "at least one special character"
    requirement a ``$`` actively *satisfied* — all admitted ``$`` on their own.
    Layering it here fixes all of them and every var added later in one place.
    """
    if "$" in value:
        return False
    validator = _VAR_VALIDATORS.get(var)
    if validator is None:
        return True
    return validator(value)


def _is_forbidden_value(var: str, value: str) -> bool:
    """True if ``value`` is one of ``var``'s registered forbidden values.

    Kept out of :func:`_validate_var` on purpose. That function answers "is this
    value *well-formed* for its consumer?", and every value registered in
    ``_VAR_FORBIDDEN_VALUES`` is — the published openobserve default would start
    the container quite happily. This asks a second question, "is this value
    *anyone's* secret?", which only the deploy boundary is in a position to ask.
    """
    return value in _VAR_FORBIDDEN_VALUES.get(var, frozenset())


def _raise_forbidden_var(var: str) -> None:
    """Raise the standard "forbidden value" RuntimeError, never the value.

    The twin of :func:`_raise_invalid_var` for the other refusal, in the same
    shape and with the same rule: the message names the VARIABLE and how to fix
    it, and never echoes the offending value back — a published default is still
    the password some deployment is currently running on, and error text travels
    into terminals, transcripts and CI logs.
    """
    constraint = _VAR_FORBIDDEN_DESCRIPTIONS.get(var, "must not be its published template default")
    raise RuntimeError(f"{var} is invalid: {constraint}. Refusing to deploy. (Value not shown.)")


def _raise_invalid_var(var: str, value: str = "") -> None:
    """Raise the standard "invalid var" RuntimeError, never the value.

    ``value`` is inspected only to pick the right explanation — a ``$`` failure
    and a format failure are different problems with different fixes, and the
    generic "does not satisfy its registered format constraint" text sends an
    operator hunting through a validator that is not the one that rejected them.
    It is never rendered into the message.
    """
    if "$" in value:
        constraint = (
            "must not contain '$' — compose interpolates env_file values, so the "
            "container would receive a truncated secret while .env still reads "
            "correctly ('$' cannot be escaped portably here)"
        )
    else:
        constraint = _VAR_VALIDATOR_DESCRIPTIONS.get(
            var, "does not satisfy its registered format constraint"
        )
    raise RuntimeError(f"{var} is invalid: {constraint}. Refusing to deploy. (Value not shown.)")
