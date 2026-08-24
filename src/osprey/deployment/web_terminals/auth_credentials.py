"""Deploy-time provisioning of per-user web-terminal auth credentials.

Owns ``.env.auth``: the 0600 project-root file holding one
``OSPREY_AUTH_PW_HASH_<USER>`` entry per roster user, referenced ONLY by the
auth sidecar's compose ``env_file`` so an auth secret never reaches a per-user
agent container. :func:`ensure_auth_credentials` establishes a hash for every
roster user; :func:`purge_auth_credentials` removes a departed user's entries so
a re-added same-name user gets a fresh credential rather than inheriting the
previous holder's password.

Provisioning follows the "existing value wins, append what's missing"
convention of the service-token writer
(``osprey.deployment.container_lifecycle._ensure_service_tokens``): re-running a
deploy never rewrites a hash that is already there, so the operation is
idempotent and pre-existing sessions survive a routine deploy.

Two namespaces live under the ``OSPREY_AUTH_PW_`` stem and are deliberately
kept in different files: plaintext ``OSPREY_AUTH_PW_<USER>`` is read from the
project ``.env`` as a convenience input and is hashed on the way in, while
``OSPREY_AUTH_PW_HASH_<USER>`` is written to (and read from) ``.env.auth``.
Only the latter is ever handed to a container, so a plaintext password set for
convenience never ships.

One further per-user secret lives here without living in ``.env.auth``: the
web-terminal handshake secret (:func:`ensure_terminal_secrets`), keyed
``OSPREY_TERMINAL_SECRET_<USER>`` and minted into the deploy ``.env``. It is
provisioned by the same "existing value wins" rule and by the same suffix
mapping, which is why it is written beside its siblings rather than in a module
of its own — but it is a different *file*, because two services (nginx and that
user's own terminal) must both read it, and ``.env.auth`` exists precisely to be
mounted by neither.

Username validity is enforced *here* as a hard raise rather than left to lint:
``osprey up`` never runs lint, and two usernames that normalize onto one
env-var suffix (``alice-b`` and ``alice_b``) would silently share a single hash
— one operator's password opening the other's terminal, the exact isolation
failure this feature exists to prevent.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# The service-token recipes are imported rather than restated so the signing
# secrets below are minted and validated exactly like every other deploy-time
# secret, and pick up any constraint registered for them later.
from osprey.cli.output import report_fact
from osprey.deployment.errors import ComposeInterpolationError
from osprey.deployment.service_tokens import (
    _generate_token,
    _raise_invalid_var,
    _validate_var,
)

# `USERNAME_CHARSET_RE` is the single definition of the roster-username charset,
# shared with the lint rule that reports the same violation at scaffold time and
# with render's fail-closed gate. Restating the pattern here would let the
# deploy-time gate and the other two drift apart.
from osprey.deployment.web_terminals.personas import (
    USERNAME_CHARSET_RE,
    env_var_suffix,
    env_var_suffix_collisions,
)
from osprey.interfaces.web_auth import ROSTER_SECRET_ENV_PREFIX
from osprey.services.auth_sidecar.passwords import hash_password
from osprey.utils.dotenv import (
    DEPLOY_MINTED_BANNER,
    ENV_AUTH_BANNER,
    ENV_LOCAL_FILENAME,
    append_profile_env,
    atomic_write,
    compose_unsafe_vars,
    dotenv_line_var,
    env_file_lock,
    parse_dotenv_file,
)
from osprey.utils.logger import get_logger

logger = get_logger("deployment.lifecycle")

#: Project-root file holding the per-user password hashes. Separate from the
#: project ``.env`` so the sidecar can be the only service that mounts it.
AUTH_ENV_FILENAME = ".env.auth"

#: Env-var stem for a stored hash, completed by :func:`env_var_suffix`.
PW_HASH_VAR_PREFIX = "OSPREY_AUTH_PW_HASH_"

#: Env-var stem for an operator-supplied plaintext password in the project
#: ``.env``. Consumed and hashed at preflight; never forwarded to a container.
PW_PLAINTEXT_VAR_PREFIX = "OSPREY_AUTH_PW_"

#: The sidecar's cookie-signing secret, required in EVERY auth method — an
#: empty one takes the whole sidecar to 503.
SESSION_SECRET_VAR = "OSPREY_AUTH_SESSION_SECRET"

#: Secret for the short-lived OIDC state cookie. Required by the sidecar only
#: in ``oidc`` mode, but minted alongside the session secret regardless.
STATE_SECRET_VAR = "OSPREY_AUTH_STATE_SECRET"

#: Both signing secrets, in a fixed order so an appended block is byte-stable.
SESSION_SECRET_VARS = (SESSION_SECRET_VAR, STATE_SECRET_VAR)

#: Env-var stem for one user's web-terminal handshake secret, completed by
#: :func:`env_var_suffix`. Deliberately NOT an ``.env.auth`` variable: nginx
#: reads it (to stamp the header on every proxied request) and so does that
#: user's own terminal container (to verify it), and ``.env.auth`` is the file
#: only the auth sidecar ever mounts. The deploy ``.env`` is the one store both
#: of those services already interpolate from. Spelled by the web gate that
#: verifies these secrets (:data:`osprey.interfaces.web_auth.ROSTER_SECRET_ENV_PREFIX`),
#: so the mint and the gate cannot disagree about the prefix.
TERMINAL_SECRET_VAR_PREFIX = ROSTER_SECRET_ENV_PREFIX

#: What a terminal-secret refusal calls the thing it could not provision. The
#: mint runs in EVERY auth method, so this message reaches operators who
#: configured no authentication at all; naming the secrets rather than the auth
#: credentials is what keeps such a refusal about something they can find.
TERMINAL_SECRET_SUBJECT = "secrets"

_HASH_HEADER = "# Auto-generated web-terminal auth password hashes (osprey deploy)"
_SECRET_HEADER = "# Auto-generated web-terminal auth signing secrets (osprey deploy)"

# Minted-password alphabet: alphanumerics minus the transcription lookalikes
# (0/O, 1/l/I). A minted password is read off a terminal and retyped by a
# person, so ambiguity costs support calls; staying alphanumeric also keeps the
# value free of any character a dotenv parser treats specially.
_MINT_ALPHABET = "abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"

# Length of a minted password in characters. 16 characters from this 57-symbol
# alphabet is ~93 bits of entropy — far above what the sidecar's per-user
# attempt throttle leaves reachable, and short enough to retype. This is a
# human-facing password, not a machine token, so it deliberately does not carry
# the 256-bit bar the service-token recipes hold themselves to.
_MINT_LENGTH = 16


@dataclass(frozen=True)
class AuthCredentialsResult:
    """Outcome of one :func:`ensure_auth_credentials` run.

    ``changed`` is the field the deploy path keys off: whenever ``.env.auth``
    gains content, the sidecar must be force-recreated rather than restarted,
    because compose bakes ``env_file`` content into the container at *creation*
    time — a restarted sidecar would keep serving the previous file's hashes.

    ``missing`` names users for whom no hash could be established (the file
    could not be written). It is the post-mint invariant the fail-closed deploy
    gate checks: a ``method: password`` deploy with a non-empty ``missing`` must
    abort before any compose invocation rather than come up with a user who can
    never log in.
    """

    env_auth_path: Path
    changed: bool
    minted: tuple[str, ...]
    hashed_from_plaintext: tuple[str, ...]
    preexisting: tuple[str, ...]
    missing: tuple[str, ...]


#: What :func:`_validate_usernames` calls the thing it was about to provision,
#: by default. Names the auth credentials because that is what the module's
#: original caller mints; the terminal-secret caller passes its own, because it
#: runs on EVERY deployment — ``auth.method: none`` included — and a refusal
#: naming "auth credentials" on an auth-off deployment describes a feature the
#: operator did not turn on.
_AUTH_CREDENTIALS_SUBJECT = "auth credentials"


def _validate_usernames(
    usernames: list[str],
    *,
    var_prefix: str = PW_HASH_VAR_PREFIX,
    subject: str = _AUTH_CREDENTIALS_SUBJECT,
) -> None:
    """Reject a roster this module cannot key credentials for, with a hard raise.

    Args:
        usernames: The roster names about to be keyed.
        var_prefix: The env-var stem the collision message names, so a refusal
            points at the variable the *caller* was about to write rather than
            at whichever namespace happened to be checked first. Every stem here
            shares one suffix mapping, so a collision under any of them is a
            collision under all of them.
        subject: What the refusal says it could not provision. Parametrized for
            the same reason ``var_prefix`` is: this gate now runs on every
            deployment through :func:`ensure_terminal_secrets`, so a roster name
            outside the charset is refused on an auth-off deployment too — and
            the operator reading that refusal has no auth to go and look at.

    Raises:
        RuntimeError: If any username falls outside the roster charset, or if
            two distinct usernames normalize onto the same env-var suffix.
    """
    invalid = [name for name in usernames if not USERNAME_CHARSET_RE.fullmatch(name)]
    if invalid:
        raise RuntimeError(
            f"Cannot provision web-terminal {subject}: "
            f"{', '.join(repr(name) for name in sorted(invalid))} does not match "
            f"{USERNAME_CHARSET_RE.pattern!r} (usernames become nginx location keys "
            "and URL path segments). Refusing to deploy."
        )

    collisions = env_var_suffix_collisions(usernames)
    if collisions:
        detail = "; ".join(
            f"{', '.join(repr(n) for n in names)} -> {var_prefix}{suffix}"
            for suffix, names in collisions.items()
        )
        raise RuntimeError(
            f"Cannot provision web-terminal {subject}: distinct usernames map "
            f"onto one credential variable ({detail}). They would share a single "
            "secret, so one user's credentials would open another's terminal. "
            "Rename one of them. Refusing to deploy."
        )


def _mint_password() -> str:
    """Generate one human-typeable password from a CSPRNG."""
    return "".join(secrets.choice(_MINT_ALPHABET) for _ in range(_MINT_LENGTH))


def _append_entries(env_auth_path: Path, entries: dict[str, str], header: str) -> None:
    """Append ``entries`` to ``.env.auth`` under ``header``, creating it 0600.

    A file created here is stamped with
    :data:`~osprey.utils.dotenv.ENV_AUTH_BANNER` at the top, ahead of the first
    per-block ``header`` — creation is the only moment the file-top readme can
    be added, since every later write appends below it and the rewrite paths
    preserve existing lines.

    The file is opened with an explicit 0600 creation mode rather than being
    created and then chmod'ed, so a hash is never briefly world-readable.

    The append is not atomic: a crash mid-write can leave the final line
    truncated. That degrades fail-closed in every case — a truncated hash value
    no longer verifies any password, and a truncated variable name leaves its
    user with no hash at all (the next deploy re-mints one). Neither outcome
    grants access, and an operator recovers either with ``osprey users passwd
    <user>``. Entries written before the torn line are complete and unaffected,
    since each occupies its own line.

    Raises:
        OSError: If the file or its directory cannot be written.
    """
    prefix = ""
    file_banner = ""
    if env_auth_path.is_file():
        text = env_auth_path.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            prefix = "\n"
    else:
        file_banner = ENV_AUTH_BANNER
    block = "".join(f"{key}={value}\n" for key, value in entries.items())
    fd = os.open(env_auth_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(f"{prefix}{file_banner}{header}\n{block}")


def _normalize_mode(env_auth_path: Path) -> None:
    """Tighten an existing ``.env.auth`` to 0600, never raising.

    Guarded on its own because the callers' write paths deliberately *report*
    an OSError rather than raise it: on a read-only filesystem this chmod
    raises the same error, and letting it escape would break that contract from
    the last statement of the caller. A mode change is not a content change, so
    this never affects a caller's ``changed`` report.
    """
    if not env_auth_path.is_file():
        return
    try:
        os.chmod(env_auth_path, 0o600)
    except OSError as exc:
        logger.warning(
            "Could not set 0600 permissions on %s (%s). Check them by hand, since "
            "the file holds web-terminal auth secrets.",
            env_auth_path,
            exc,
        )


def raise_if_env_auth_would_be_interpolated(project_root: str | Path) -> None:
    """Refuse to act on a ``.env.auth`` holding a ``$``-bearing value.

    Scans the file *as it exists on disk*, which is the whole point. Everything
    this module writes is ``$``-free by construction — base64url hashes joined
    by :data:`~osprey.services.auth_sidecar.passwords.FIELD_SEP`,
    ``token_urlsafe`` signing secrets — and a unit test already pins that. The
    value this catches is the one OSPREY never writes: the OIDC client secret,
    which the compose template references by *name* only and the operator
    appends by hand, minted by an IdP whose alphabet routinely includes
    punctuation. There is no write boundary to hook, and the generator-side test
    cannot see it because that test calls the generators. Reading the finished
    file is the only place both the minted and the hand-added lines are visible
    at once.

    Note the sidecar's own startup check tests the client secret for
    *emptiness*. A fully-eaten secret trips it; a partially truncated one is
    non-empty, clears startup, and fails later against the IdP with an opaque
    token-endpoint rejection. This scan is what closes that gap.

    CALL THIS BEFORE MUTATING ``.env.auth``, never between a mutation and the
    sidecar recreate that puts it into force. ``decommission``/``prune`` remove
    a departed user's hash and then recreate; a refusal wedged in between would
    leave the file saying the user is gone while the running sidecar still
    accepts their password — the exact divergence those verbs exist to prevent,
    traded for an unrelated secret. Refusing *before* any mutation costs
    nothing, and a corrupt value left in place is still caught by the next
    ``osprey up``.
    """
    env_auth = Path(project_root) / AUTH_ENV_FILENAME
    if not env_auth.is_file():
        return
    offenders = compose_unsafe_vars(parse_dotenv_file(env_auth))
    if offenders:
        raise ComposeInterpolationError(offenders, env_auth)


def ensure_auth_credentials(
    usernames: Iterable[str],
    project_root: str | Path,
    *,
    echo: Callable[[str], None],
) -> AuthCredentialsResult:
    """Establish a password hash in ``.env.auth`` for every roster user.

    Per user, in precedence order:

    1. An ``OSPREY_AUTH_PW_HASH_<USER>`` entry already in ``.env.auth`` is kept
       untouched — re-deploying is idempotent and does not invalidate sessions.
       An entry present but *empty* (``OSPREY_AUTH_PW_HASH_ALICE=``) does not
       count as established: that user falls through to the steps below and the
       freshly written entry, appended after the empty one, is the one the
       parser returns (last assignment wins). An empty value would otherwise
       leave a roster user permanently unable to log in.
    2. Otherwise a plaintext ``OSPREY_AUTH_PW_<USER>`` in the project ``.env``
       is hashed in. Leading and trailing whitespace is trimmed before hashing,
       so a value padded by an editor or a copy-paste hashes to what the
       operator will actually type at the login prompt; a password whose real
       content is only whitespace is treated as absent. The plaintext stays in
       ``.env`` (which no container receives on the auth path) and is never
       echoed: the operator already knows it.
    3. Otherwise a random password is minted, hashed, and printed **once**.
       That print is the only moment the password exists in cleartext output;
       it is never logged, and only its hash is persisted, so it cannot be
       recovered afterwards.

    Neither the process environment nor ``.env`` is consulted for a *hash*: the
    sidecar reads its credentials solely from ``.env.auth`` via compose
    ``env_file``, so a hash visible only to this process would suppress the mint
    and leave the user unable to log in. Likewise a plaintext password is read
    only from the project ``.env`` file, never from the process environment — a
    stray exported variable must not silently become a deployed credential.

    A write failure (read-only ``.env.auth``, unwritable project root) is
    reported through ``missing`` rather than raised, so the deploy path's
    fail-closed gate produces the abort with its own context. Nothing is printed
    for a password that could not be persisted.

    Args:
        usernames: Roster usernames, typically ``entry["name"]`` for each
            ``normalize_users`` entry. Order is preserved; a name repeated
            verbatim is one user listed twice and is processed once.
        project_root: Directory holding ``.env`` and ``.env.auth``.
        echo: Sink for the one-time minted-password notice. Required rather
            than defaulted, because a password is the one line that must not
            reach a stream nobody chose: every caller says where it goes.
            Deliberately not the logger, which ships elsewhere.

    Returns:
        An :class:`AuthCredentialsResult` describing what happened, including
        whether ``.env.auth`` changed.

    Raises:
        RuntimeError: If a username is outside the roster charset, or two
            distinct usernames collide onto one credential variable.
    """
    ordered: list[str] = []
    for name in usernames:
        if name not in ordered:
            ordered.append(name)
    _validate_usernames(ordered)

    root = Path(project_root)
    env_auth_path = root / AUTH_ENV_FILENAME
    dotenv_path = root / ".env"

    stored = parse_dotenv_file(env_auth_path) if env_auth_path.is_file() else {}
    project_env = parse_dotenv_file(dotenv_path) if dotenv_path.is_file() else {}

    preexisting: list[str] = []
    hashed_from_plaintext: list[str] = []
    minted: list[str] = []
    pending: dict[str, str] = {}
    minted_passwords: dict[str, str] = {}

    for name in ordered:
        suffix = env_var_suffix(name)
        hash_var = f"{PW_HASH_VAR_PREFIX}{suffix}"
        if stored.get(hash_var, "").strip():
            preexisting.append(name)
            continue
        plaintext = project_env.get(f"{PW_PLAINTEXT_VAR_PREFIX}{suffix}", "").strip()
        if plaintext:
            pending[hash_var] = hash_password(plaintext)
            hashed_from_plaintext.append(name)
            continue
        password = _mint_password()
        pending[hash_var] = hash_password(password)
        minted_passwords[name] = password
        minted.append(name)

    changed = False
    missing: list[str] = []
    if pending:
        try:
            _append_entries(env_auth_path, pending, _HASH_HEADER)
        except OSError as exc:
            # Fail-closed, but not from here: the deploy gate owns the abort so
            # it can name the deploy context. Say so loudly meanwhile — never
            # naming a password, minted or supplied.
            missing = hashed_from_plaintext + minted
            logger.error(
                "Could not write web-terminal auth credentials to %s (%s). "
                "No password hash could be established for: %s",
                env_auth_path,
                exc,
                ", ".join(sorted(missing)),
            )
            hashed_from_plaintext = []
            minted = []
        else:
            changed = True
            # Log the variable names only — a hash identifies a credential
            # generation and must not travel to a log sink.
            report_fact(
                logger,
                f"Provisioned web-terminal auth credential(s) {', '.join(pending)} "
                f"in {env_auth_path} (gitignored, 0600)",
            )
            for name in minted:
                echo(
                    f"Minted a login password for web-terminal user '{name}': "
                    f"{minted_passwords[name]}"
                )
            if minted:
                echo(
                    "Record these now. They are shown once and cannot be "
                    "recovered; only their hashes are stored."
                )

    _normalize_mode(env_auth_path)

    return AuthCredentialsResult(
        env_auth_path=env_auth_path,
        changed=changed,
        minted=tuple(minted),
        hashed_from_plaintext=tuple(hashed_from_plaintext),
        preexisting=tuple(preexisting),
        missing=tuple(missing),
    )


def seeded_logins(project_root: str | Path, usernames: Iterable[str]) -> list[tuple[str, str]]:
    """The roster logins whose password is still the one the PROFILE shipped.

    :func:`ensure_auth_credentials` documents why a plaintext password is never
    echoed: the operator typed it into ``.env``, so they already know it. That
    reasoning holds for a password an operator chose and fails for one a profile
    seeded — ``control-assistant.yml`` declares ``OSPREY_AUTH_PW_ALICE: alice``
    under ``env.defaults`` and ``osprey init`` writes it into ``.env``, so the
    login exists and nobody was ever told what it is. This function names that
    narrow case, and only it.

    A password qualifies when the value in the project ``.env`` is still
    IDENTICAL to the one ``profile.yml`` declares for that variable. Such a
    value is published in the profile — in this repo and, for a preset, in the
    OSPREY source tree — so printing it discloses nothing that reading two
    tracked files would not. The moment an operator edits either file the values
    stop matching and the login drops out of the result, which is what keeps a
    real facility's credential out of a terminal and out of any transcript.

    Advisory throughout: an unreadable, absent or malformed ``profile.yml`` or
    ``.env`` yields no logins rather than an error. Nothing here is load-bearing
    for a deployment that has already started.

    :param project_root: The deployment repo — the directory holding
        ``profile.yml`` and ``.env``.
    :param usernames: Roster usernames that sit behind the login wall, in the
        order they should be shown.
    :return: ``(username, password)`` pairs, in ``usernames`` order, holding
        only the users whose password is still the profile's declared default.
    """
    root = Path(project_root)
    try:
        declared = _profile_env_defaults(root)
        if not declared:
            return []
        env_path = root / ".env"
        project_env = parse_dotenv_file(env_path) if env_path.is_file() else {}
    except Exception as exc:  # pragma: no cover - advisory read
        logger.debug(f"Seeded logins skipped: {exc}")
        return []

    logins: list[tuple[str, str]] = []
    for name in usernames:
        variable = f"{PW_PLAINTEXT_VAR_PREFIX}{env_var_suffix(name)}"
        # Trimmed on both sides, because that is what `ensure_auth_credentials`
        # hashed: comparing the raw values would drop a login whose `.env` entry
        # an editor padded, even though the padded value produced exactly the
        # profile default's hash.
        current = project_env.get(variable, "").strip()
        if current and current == str(declared.get(variable, "")).strip():
            logins.append((name, current))
    return logins


def _profile_env_defaults(root: Path) -> dict[str, Any]:
    """``env.defaults`` as ``profile.yml`` declares it, or an empty mapping.

    Read from the repo's own ``profile.yml`` rather than from the preset it was
    materialized from: the profile is what an operator edits, so it is the only
    file whose values can honestly be called "what this deployment declared".
    """
    from osprey.cli.repo_resolver import PROFILE_FILENAME

    profile_path = root / PROFILE_FILENAME
    if not profile_path.is_file():
        return {}
    with profile_path.open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)
    if not isinstance(profile, dict):
        return {}
    env_section = profile.get("env")
    if not isinstance(env_section, dict):
        return {}
    defaults = env_section.get("defaults")
    return defaults if isinstance(defaults, dict) else {}


@dataclass(frozen=True)
class AuthSecretsResult:
    """Outcome of one :func:`ensure_auth_session_secrets` run.

    Fields carry env-var NAMES, never values. ``changed`` and ``missing`` mean
    what they do on :class:`AuthCredentialsResult`: a changed ``.env.auth``
    obliges the deploy to force-recreate the sidecar, and a non-empty
    ``missing`` is the fail-closed gate's signal that a secret could not be
    established.
    """

    env_auth_path: Path
    changed: bool
    minted: tuple[str, ...]
    preexisting: tuple[str, ...]
    missing: tuple[str, ...]


def ensure_auth_session_secrets(project_root: str | Path) -> AuthSecretsResult:
    """Mint the sidecar's cookie-signing secrets into ``.env.auth``.

    Call on the deploy preflight path whenever ``auth.method != "none"``; the
    method itself is the caller's to read, exactly as it decides whether to
    call :func:`ensure_auth_credentials`.

    Both secrets are minted through the shared service-token recipe
    (``token_urlsafe(32)``, 256 bits) rather than a private generator here, and
    the effective value is checked against the same registered-constraint
    validator, so a constraint added for these vars later takes effect on this
    path without further work. They are deliberately NOT registered in
    ``_SERVICE_TOKEN_VARS``: that map only mints for members of
    ``deployed_services``, which no web-terminal-stack service ever is, so an
    entry there would silently mint nothing.

    ``OSPREY_AUTH_STATE_SECRET`` is minted unconditionally even though the
    sidecar only requires it in ``oidc`` mode, so that switching a deployment's
    method to ``oidc`` cannot 503 on a secret nobody remembered to add.

    Existing values always win, making this idempotent — re-minting would
    invalidate every live session. The one exception is a present-but-empty (or
    whitespace-only) value, which is re-minted: the sidecar strips and rejects
    such a value as missing configuration and answers every request with 503,
    so preserving it would keep the whole deployment locked out. That is the
    same judgement ``_ensure_service_tokens`` makes about the tokens it mints —
    a machine-generated secret has no meaningful empty state, so a blank is a
    blank rather than a decision to honour.

    A write failure is reported through ``missing`` rather than raised, matching
    :func:`ensure_auth_credentials` so the deploy gate owns the abort.

    Args:
        project_root: Directory holding ``.env.auth``.

    Returns:
        An :class:`AuthSecretsResult` naming which variables were minted, which
        were already present, and which could not be established.

    Raises:
        RuntimeError: If an effective value fails a constraint registered for
            it in ``service_tokens``. The message names the variable and its
            constraint, never the value.
    """
    env_auth_path = Path(project_root) / AUTH_ENV_FILENAME
    stored = parse_dotenv_file(env_auth_path) if env_auth_path.is_file() else {}

    minted: list[str] = []
    preexisting: list[str] = []
    pending: dict[str, str] = {}
    for var in SESSION_SECRET_VARS:
        if stored.get(var, "").strip():
            preexisting.append(var)
            continue
        pending[var] = _generate_token(var)
        minted.append(var)

    changed = False
    missing: list[str] = []
    if pending:
        try:
            _append_entries(env_auth_path, pending, _SECRET_HEADER)
        except OSError as exc:
            missing = minted
            minted = []
            logger.error(
                "Could not write web-terminal auth secrets to %s (%s). "
                "No value could be established for: %s",
                env_auth_path,
                exc,
                ", ".join(sorted(missing)),
            )
        else:
            changed = True
            # Names only — a signing secret must never reach a log sink.
            report_fact(
                logger,
                f"Provisioned web-terminal auth secret(s) {', '.join(pending)} "
                f"in {env_auth_path} (gitignored, 0600)",
            )

    _normalize_mode(env_auth_path)

    # Validate whatever the sidecar will actually read, whether just minted or
    # carried over from a previous deploy. Resolved from the file alone: the
    # sidecar receives .env.auth through compose env_file, so a value visible
    # only to this process is not the one that will be in force.
    post = parse_dotenv_file(env_auth_path) if env_auth_path.is_file() else {}
    for var in SESSION_SECRET_VARS:
        if var in missing:
            continue
        effective = post.get(var, "")
        if effective and not _validate_var(var, effective):
            _raise_invalid_var(var, effective)

    return AuthSecretsResult(
        env_auth_path=env_auth_path,
        changed=changed,
        minted=tuple(minted),
        preexisting=tuple(preexisting),
        missing=tuple(missing),
    )


@dataclass(frozen=True)
class TerminalSecretsResult:
    """Outcome of one :func:`ensure_terminal_secrets` run.

    Every field carries env-var NAMES, never values — a terminal secret is the
    whole of a user's proof to their own terminal, so it must not reach a log,
    a return value, or an exception message.

    ``missing`` is the fail-closed gate's signal, and it is computed from the
    file as it stands AFTER the write rather than from what the write attempted:
    the question a deploy has to answer is "does every roster user have a usable
    secret now?", which a report of what this call happened to mint cannot
    answer for a value some earlier run established.
    """

    env_path: Path
    changed: bool
    minted: tuple[str, ...]
    preexisting: tuple[str, ...]
    missing: tuple[str, ...]


def _drop_env_assignments(env_path: Path, is_target: Callable[[str], bool]) -> tuple[str, ...]:
    """Remove every assignment line in ``env_path`` whose variable is a target.

    The deploy ``.env`` is append-only to
    :func:`~osprey.utils.dotenv.append_profile_env` — a key already in the file
    is never rewritten, which is what protects an operator's secrets from a
    re-deploy. This is the one narrow exception to that, and it works by
    *removal* rather than by rewriting a value: the callers either have nothing
    to preserve (a departed user's secret) or are about to append a replacement
    (a blank value that must be re-minted), and a line that is gone cannot be
    the one a later parse returns.

    ALL assignments of a target variable go, not just the last one. A file where
    the same variable is assigned twice would otherwise keep a superseded line
    that a differently-ordered reader could still pick up.

    Comments, blank lines, and every other variable's lines are preserved
    verbatim. The rewrite goes through
    :func:`~osprey.utils.dotenv.atomic_write`, so a reader sees either the whole
    old file or the whole new one, and the result carries the 0600 mode that
    function enforces on a file holding facility secrets.

    The read, the filter and the replace happen under
    :func:`~osprey.utils.dotenv.env_file_lock` — the SAME lock
    :func:`~osprey.utils.dotenv.append_profile_env` takes, which is the whole
    point. The deploy ``.env`` is appended to by several unrelated writers
    (``osprey up`` persisting a minted service token, ``osprey build`` writing
    its derived keys, profile seeding), and a locked append landing between an
    unlocked read here and its replace would be discarded without a trace. The
    lock is re-entrant, so a caller that already holds it — to make a drop and
    the append that replaces it one critical section — is not blocked here.

    The emptiness check runs BEFORE the lock: a file with nothing to remove is
    not rewritten at all, so it needs no lock, and taking one would create a
    ``.lock`` file (and fail on a read-only directory) for a call that does
    nothing.

    Args:
        env_path: The ``.env`` to rewrite. A file that does not exist is a
            no-op, not an error.
        is_target: Predicate over a variable name; ``True`` drops the line.

    Returns:
        The variable names actually removed, de-duplicated, in file order.
        Empty when nothing matched — in which case the file is not rewritten at
        all, so an unwritable ``.env`` with nothing to remove still succeeds.

    Raises:
        OSError: If the file cannot be read or the rewrite cannot be written.
    """
    if not env_path.is_file():
        return ()
    with env_file_lock(env_path):
        lines = env_path.read_text(encoding="utf-8").splitlines()
        kept: list[str] = []
        removed: list[str] = []
        for line in lines:
            var = dotenv_line_var(line)
            if var is not None and is_target(var):
                removed.append(var)
                continue
            kept.append(line)
        if not removed:
            return ()
        atomic_write(env_path, "".join(f"{line}\n" for line in kept))
    return tuple(dict.fromkeys(removed))


def terminal_secret_var(username: str) -> str:
    """The env var holding ``username``'s web-terminal handshake secret.

    One definition of the name, shared by the mint, the purge and the render,
    so a secret can never be written under one spelling and looked up under
    another.
    """
    return f"{TERMINAL_SECRET_VAR_PREFIX}{env_var_suffix(username)}"


def ensure_terminal_secrets(
    project_root: str | Path, usernames: Iterable[str]
) -> TerminalSecretsResult:
    """Establish an ``OSPREY_TERMINAL_SECRET_<USER>`` for every roster user.

    The per-user half of the terminal's own front door: nginx stamps this value
    onto every request it proxies to that user's terminal, and the terminal
    refuses anything arriving without it. One secret per user rather than one
    per deployment, so a value leaked from one container cannot be replayed
    against a neighbour's terminal.

    Written to the deploy ``.env``, NOT to ``.env.auth``. Both nginx and the
    per-user terminal must read it, and ``.env.auth`` is by construction the
    file only the auth sidecar mounts; the deploy ``.env`` is what compose
    interpolates ``${OSPREY_TERMINAL_SECRET_<USER>}`` from for both services.

    Called on the deploy preflight path for EVERY deployment, including one with
    ``auth.method: none``. Authentication decides who may reach a terminal
    through the front door; this secret decides that the front door is the only
    way in at all, which an auth-off multi-user deployment needs just as much —
    arguably more, since nothing else stands between one user's browser and
    another user's terminal port.

    Per user, in precedence order:

    1. A non-empty value already in the deploy ``.env`` wins and is left exactly
       as it is. Re-deploying is therefore idempotent, and an operator who
       pinned a secret by hand keeps it.
    2. A present-but-empty (or whitespace-only) value is RE-MINTED, and its
       existing line(s) are removed first — ``append_profile_env`` would
       otherwise refuse to touch a key already in the file, leaving the user
       with a blank secret their terminal reads as "no secret configured". A
       machine-generated secret has no meaningful empty state, the same
       judgement :func:`ensure_auth_session_secrets` makes about the signing
       secrets.
    3. Otherwise a fresh secret is minted through the shared service-token
       recipe (``token_urlsafe(32)``, 256 bits), so it is generated and checked
       exactly like every other deploy-time secret and picks up any constraint
       registered for it later.

    Neither the process environment nor any other file is consulted: compose
    interpolates the compose file's ``${...}`` references from the deploy
    ``.env``, so a value visible only to this process is not the one that will
    be in force, and suppressing the mint on account of it would deploy a
    terminal whose secret nobody holds.

    A write failure is reported through ``missing`` rather than raised, matching
    :func:`ensure_auth_credentials` and :func:`ensure_auth_session_secrets` so
    the deploy gate owns the abort and can name the deployment's context.

    Args:
        project_root: Directory holding the deploy ``.env``.
        usernames: Roster usernames — ``entry["name"]`` for each
            :func:`~osprey.deployment.web_terminals.personas.normalize_users`
            entry. The WHOLE roster, including ``login: false`` entries: opting
            out of the login wall does not opt a terminal out of needing a front
            door. Order is preserved; a name repeated verbatim is processed once.

    Returns:
        A :class:`TerminalSecretsResult` naming which variables were minted,
        which were already established, and which are still unusable.

    Raises:
        RuntimeError: If a username is outside the roster charset, if two
            distinct usernames collide onto one secret variable, or if the
            effective value fails a constraint registered for it in
            ``service_tokens`` (including the universal "no ``$``" rule, which a
            compose-interpolated value cannot survive). The message names the
            variable, never the value.
    """
    ordered: list[str] = []
    for name in usernames:
        if name not in ordered:
            ordered.append(name)
    _validate_usernames(
        ordered,
        var_prefix=TERMINAL_SECRET_VAR_PREFIX,
        subject=TERMINAL_SECRET_SUBJECT,
    )

    env_path = Path(project_root) / ENV_LOCAL_FILENAME
    stored = parse_dotenv_file(env_path) if env_path.is_file() else {}

    preexisting: list[str] = []
    pending: dict[str, str] = {}
    blank: set[str] = set()
    for name in ordered:
        var = terminal_secret_var(name)
        value = stored.get(var)
        if value is not None and value.strip():
            preexisting.append(var)
            continue
        if value is not None:
            blank.add(var)
        pending[var] = _generate_token(var)

    changed = False
    minted: list[str] = []
    if pending:
        try:
            # ONE critical section over the drop and the append: a re-mint that
            # released the lock in between would let another writer's append
            # land on the file this call is about to rewrite, and would leave a
            # window in which the user has no assignment at all. The lock is
            # re-entrant, so the calls below take it and find it held.
            with env_file_lock(env_path):
                if blank:
                    _drop_env_assignments(env_path, blank.__contains__)
                appended = append_profile_env(env_path, pending, DEPLOY_MINTED_BANNER)
        except OSError as exc:
            # Fail-closed, but not from here: the deploy gate owns the abort so
            # it can name the deploy context. Say so loudly meanwhile — variable
            # names only.
            logger.error(
                "Could not write web-terminal secrets to %s (%s). No value could be "
                "established for: %s",
                env_path,
                exc,
                ", ".join(sorted(pending)),
            )
        else:
            minted = list(appended.added)
            changed = bool(minted)
            # A key that reappeared between the read and the write — a
            # concurrent `osprey up`, an operator saving the file — keeps
            # whatever it now holds, because the file always wins. It is
            # reported as pre-existing rather than as a conflict to warn about:
            # nobody chose the value this call would have written.
            preexisting.extend(appended.unchanged)
            preexisting.extend(conflict.key for conflict in appended.conflicts)
            if minted:
                report_fact(
                    logger,
                    f"Provisioned web-terminal secret(s) {', '.join(minted)} in {env_path}",
                )

    # Read back what is actually on file, which is what compose will
    # interpolate — for the just-minted values and the carried-over ones alike.
    post = parse_dotenv_file(env_path) if env_path.is_file() else {}
    missing: list[str] = []
    for name in ordered:
        var = terminal_secret_var(name)
        effective = post.get(var, "").strip()
        if not effective:
            missing.append(var)
            continue
        if not _validate_var(var, effective):
            _raise_invalid_var(var, effective)

    return TerminalSecretsResult(
        env_path=env_path,
        changed=changed,
        minted=tuple(minted),
        preexisting=tuple(preexisting),
        missing=tuple(missing),
    )


def purge_terminal_secret(username: str, project_root: str | Path) -> bool:
    """Remove ``username``'s terminal secret from the deploy ``.env``.

    The deploy-``.env`` counterpart of :func:`purge_auth_credentials`, which
    only ever touches ``.env.auth``: without this a decommissioned user's secret
    would sit in the deployment's secret store indefinitely, and a re-added
    same-name user would silently inherit it — a value the previous holder's
    browser, container and any copy they took away still knows.

    Every other line — comments, banners, other users' secrets, every unrelated
    variable — is preserved verbatim.

    Like :func:`purge_auth_credentials` this does not validate ``username``:
    removing an entry for a name the roster should never have accepted is always
    safe, and a lifecycle teardown must not be blocked by the departing user's
    spelling.

    Args:
        username: The roster username whose secret should be removed.
        project_root: Directory holding the deploy ``.env``.

    Returns:
        ``True`` if the file changed, ``False`` if it was absent or held no
        secret for that user.

    Raises:
        OSError: If the file exists, holds the entry, and cannot be rewritten.
    """
    env_path = Path(project_root) / ENV_LOCAL_FILENAME
    target = terminal_secret_var(username)
    removed = _drop_env_assignments(env_path, lambda var: var == target)
    if not removed:
        return False
    report_fact(logger, f"Removed web-terminal secret {target} from {env_path}")
    return True


def purge_orphan_terminal_secrets(
    project_root: str | Path, usernames: Iterable[str]
) -> tuple[str, ...]:
    """Remove terminal secrets belonging to nobody on ``usernames``.

    What ``osprey users prune`` needs and :func:`purge_terminal_secret` cannot
    give it: prune targets orphans discovered in the runtime rather than a named
    user, so the departed names are not known to the caller. This asks the
    complementary question — which ``OSPREY_TERMINAL_SECRET_*`` variables in the
    deploy ``.env`` belong to no current roster user — and removes exactly those.

    Defined against the roster rather than against the runtime on purpose: a
    secret for a user who has no roster entry is stale whether or not their
    container was still around to be pruned.

    Args:
        project_root: Directory holding the deploy ``.env``.
        usernames: The CURRENT roster. An empty roster means every terminal
            secret in the file is an orphan, which is the honest reading — a
            deployment with no web-terminal users has no terminal to hold one.

    Returns:
        The variable names removed, empty when there was nothing to remove.

    Raises:
        OSError: If entries had to be removed and the file could not be
            rewritten.
    """
    keep = {terminal_secret_var(name) for name in usernames}

    def _is_orphan(var: str) -> bool:
        return var.startswith(TERMINAL_SECRET_VAR_PREFIX) and var not in keep

    env_path = Path(project_root) / ENV_LOCAL_FILENAME
    removed = _drop_env_assignments(env_path, _is_orphan)
    if removed:
        report_fact(
            logger,
            f"Removed orphaned web-terminal secret(s) {', '.join(removed)} from {env_path}",
        )
    return removed


def purge_auth_credentials(username: str, project_root: str | Path) -> bool:
    """Remove ``username``'s ``OSPREY_AUTH_PW_*`` entries from ``.env.auth``.

    Called when a user is decommissioned or pruned: a re-added same-name user
    must be minted a fresh credential instead of silently inheriting the
    departed user's password. Both the hash entry and any plaintext entry that
    was placed in ``.env.auth`` by hand are removed; every other line —
    comments, blank lines, other users' entries — is preserved line-for-line.
    Line *content* is untouched, but the file is rewritten with LF endings and
    a trailing newline, so CRLF input is normalized on the way out.

    The rewrite is atomic (temporary file plus rename) so an interrupted purge
    cannot leave a truncated credential file, and the replacement is 0600 like
    the original.

    Unlike :func:`ensure_auth_credentials` this does not validate ``username``:
    removing entries for a name the roster should never have accepted is always
    safe, and a lifecycle teardown must not be blocked by the departing user's
    spelling.

    Args:
        username: The roster username whose entries should be removed.
        project_root: Directory holding ``.env.auth``.

    Returns:
        ``True`` if the file changed, ``False`` if it was absent or held no
        entry for that user. The deploy path force-recreates the sidecar on
        ``True`` for the same reason :attr:`AuthCredentialsResult.changed`
        exists.
    """
    env_auth_path = Path(project_root) / AUTH_ENV_FILENAME
    if not env_auth_path.is_file():
        return False

    suffix = env_var_suffix(username)
    targets = {
        f"{PW_HASH_VAR_PREFIX}{suffix}",
        f"{PW_PLAINTEXT_VAR_PREFIX}{suffix}",
    }
    lines = env_auth_path.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if dotenv_line_var(line) not in targets]
    if len(kept) == len(lines):
        return False

    _atomic_rewrite(env_auth_path, kept)
    report_fact(
        logger, f"Removed web-terminal auth credential(s) for {username!r} from {env_auth_path}"
    )
    return True


def _atomic_rewrite(env_auth_path: Path, lines: list[str]) -> None:
    """Replace ``.env.auth``'s whole content with ``lines``, atomically and 0600.

    Write-to-temporary-then-rename, so an interrupted write cannot leave a
    truncated credential file: a reader sees either the previous content or the
    new content, never a half-written line. The temporary is created 0600 rather
    than chmod'ed afterwards, so a hash is never briefly world-readable, and
    ``os.replace`` carries that mode onto the destination — which is why callers
    need no separate mode fix-up.

    Shared by :func:`purge_auth_credentials` and :func:`set_auth_password` so
    there is one definition of how this file is rewritten.

    Raises:
        OSError: If the temporary or the destination cannot be written.
    """
    tmp_path = env_auth_path.with_name(f"{env_auth_path.name}.tmp")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("".join(f"{line}\n" for line in lines))
    os.replace(tmp_path, env_auth_path)


def set_auth_password(username: str, password: str, project_root: str | Path) -> Path:
    """Replace ``username``'s stored hash with one derived from ``password``.

    The rotation entry point behind ``osprey users passwd <user>``, and the only
    operation in this module that deliberately overwrites an established
    credential — :func:`ensure_auth_credentials` exists precisely never to do
    that.

    **Replaces in place; never appends.** The user's existing entry is
    substituted where it already sits — every other line, including other users'
    entries, the signing secrets and the headers, is preserved byte for byte —
    and the file is rewritten exactly once. Appending a second entry would also
    "work", since the parser takes the last assignment, but every rotation would
    leave the superseded credential sitting in the file: correctness would rest
    on parse order rather than on content, a hash an operator believed retired
    would still be readable, and a facility rotating on a schedule would grow
    this file without bound.

    Whitespace is stripped before hashing, matching the plaintext path in
    :func:`ensure_auth_credentials`, so a password set through ``.env`` and one
    set here behave identically at the login prompt.

    Every live session of that user dies as a consequence, with no revocation
    bookkeeping: a session cookie carries the credential-generation tag of the
    hash it was issued against (see
    :func:`osprey.services.auth_sidecar.passwords.generation_tag`), and a new
    hash necessarily produces a different tag.

    **This function RAISES on a write failure, and that is a deliberate
    divergence from the rest of this module — do not "harmonize" it.**
    :func:`ensure_auth_credentials` and :func:`ensure_auth_session_secrets`
    report through a ``missing`` tuple because they run on the deploy path,
    where a fail-closed gate downstream owns the abort and can name the whole
    deployment's context. Nothing downstream owns this one: it is reached from
    an interactive command, and an operator who was just prompted for a
    password must never be told it took effect when the file could not be
    written. Because the replacement is a single atomic rewrite, a failure
    leaves the file exactly as it was — the old password keeps working, which
    is the state the raised error describes.

    The password is never logged, and never appears in a return value or an
    exception message.

    Args:
        username: Roster username whose password is being replaced.
        password: The new cleartext password; hashed here and never stored.
        project_root: Directory holding ``.env.auth``.

    Returns:
        The path of the ``.env.auth`` that now holds the new hash. Returning at
        all means the credential is in force once the sidecar is recreated,
        which is the caller's next obligation.

    Raises:
        RuntimeError: If ``username`` is outside the roster charset, or if
            ``.env.auth`` could not be written — the message says plainly that
            the password did not change.
        ValueError: If ``password`` is empty or only whitespace.
    """
    _validate_usernames([username])

    secret = password.strip()
    if not secret:
        raise ValueError(f"Refusing to set an empty password for web-terminal user {username!r}.")

    root = Path(project_root)
    env_auth_path = root / AUTH_ENV_FILENAME
    suffix = env_var_suffix(username)
    hash_var = f"{PW_HASH_VAR_PREFIX}{suffix}"
    plaintext_var = f"{PW_PLAINTEXT_VAR_PREFIX}{suffix}"
    entry = f"{hash_var}={hash_password(secret)}"

    existing = (
        env_auth_path.read_text(encoding="utf-8").splitlines() if env_auth_path.is_file() else []
    )
    lines = _lines_with_entry_replaced(existing, entry, hash_var, plaintext_var)

    try:
        _atomic_rewrite(env_auth_path, lines)
    except OSError as exc:
        raise RuntimeError(
            f"Could not write {env_auth_path} ({exc}); the password for {username!r} "
            "was NOT changed. Check that file's permissions and retry."
        ) from exc

    # The variable name only — a hash identifies a credential generation and
    # must not travel to a log sink, and the password never does at all.
    report_fact(logger, f"Replaced web-terminal auth credential {hash_var} in {env_auth_path}")
    return env_auth_path


def _lines_with_entry_replaced(
    existing: list[str], entry: str, hash_var: str, plaintext_var: str
) -> list[str]:
    """``existing`` with ``entry`` substituted in place for ``hash_var``.

    The user's current assignment is overwritten where it already sits, so the
    file's shape does not change: no line moves, no header is duplicated, and a
    repeated rotation is a no-growth operation. Any *further* assignments of the
    same variable are dropped, which cleans up a file that an earlier
    append-style write had already doubled up. A plaintext entry for the same
    user is dropped too, since it is superseded by the hash being written and
    would otherwise sit in the credential file as cleartext.

    A user with no entry yet is appended under the existing
    :data:`_HASH_HEADER` when the file already has one — placed after the last
    hash line so it joins that block — and under a newly added header when it
    does not.
    """
    lines: list[str] = []
    replaced = False
    for line in existing:
        var = dotenv_line_var(line)
        if var == hash_var:
            if not replaced:
                lines.append(entry)
                replaced = True
            continue
        if var == plaintext_var:
            continue
        lines.append(line)
    if replaced:
        return lines

    if _HASH_HEADER in lines:
        last_hash = max(
            (
                index
                for index, line in enumerate(lines)
                if (dotenv_line_var(line) or "").startswith(PW_HASH_VAR_PREFIX)
            ),
            default=lines.index(_HASH_HEADER),
        )
        lines.insert(last_hash + 1, entry)
        return lines

    if lines and lines[-1].strip():
        lines.append("")
    lines.extend((_HASH_HEADER, entry))
    return lines
