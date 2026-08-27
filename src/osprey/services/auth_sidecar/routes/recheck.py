"""The login re-check: one identity matrix, enforced in one place.

Every route in this service that mints a session first asks the same question —
*given how this browser proved who it is, what is that proof worth?* — and the
answer is a table, not a judgement call. This module is that table. It is the
only place that maps an auth method onto the ``(subject, role)`` a session may
carry, so a row cannot be admitted by the password path and refused by the OIDC
one, and a method nobody wrote a row for cannot fall through into a plausible
reading of itself.

The matrix, one row per posture:

===================  =====================================  ===================
method               subject                                role
===================  =====================================  ===================
``none``             — (no session is minted)               — (no login events)
``password``         the roster username                    the roster ``role:``
``oidc``, unbound    the asserted IdP subject               the roster ``role:``
``oidc``, bound      the asserted IdP subject               the claim's role,
                                                            cross-checked
                                                            against the roster's
anything else        refused at mint; verify names the      — (refused)
                     locally verified identity
===================  =====================================  ===================

**The last row's two halves dispose of it differently, which is why it is the
one row the table qualifies.** :func:`recheck_login` refuses an unsupported
method outright, so no session can ever be minted under one — that half is
absolute. :func:`session_subject` does not re-check the closed set: it reads a
session that already exists, and the only way to reach it under an unsupported
method is a *method change* made under an outstanding cookie. There the honest
answer is the identity this service verified locally, not a second refusal of a
login that already happened. Nothing widens — the row still mints nothing.

**The ``none`` row is somebody else's.** A deployment with no login method runs
no sidecar: the identity a single-user install acts under comes from
:func:`osprey.utils.identity.acting_identity`, its role is whatever its render
pinned, and there is no login event to record because there is no login. The row
is in the table so that the method reaching this service *anyway* — a
half-rendered compose file, a typo, a value a future release adds — is refused
by a rule rather than by whichever branch happened to be written last.

**The role a login carries must be the role its terminal was built as.** A
roster entry's ``role:`` is not a second opinion about privilege — it is what
the *render* resolved this user's persona from, so it already decided which
container the login lands in. The matrix therefore reads it under every method
that mints a session:

* Under ``password`` it is the whole answer. The entry names a role, the render
  built that user's terminal from the persona it resolves to, and no third party
  is involved.
* Under ``oidc`` with **no** claim binding it is still the answer. The provider
  proves *who* this is; nothing in the deployment asked it about privilege, so
  the role is the one the deployment itself bound at render time.
* Under ``oidc`` **with** a claim binding it becomes the cross-check target. The
  validated ID token decides the role, and it must be the role the roster named
  for the user whose card was clicked — otherwise the session would say one role
  while sitting in another's container. A disagreement is refused
  (:data:`~osprey.services.auth_sidecar.audit.REASON_ROLE_MISMATCH`), and
  refusing grants nothing: it turns away a login whose asserted privilege does
  not match the terminal behind the door.

**The one gap, stated rather than papered over.** A roster entry that names no
role — a ``persona:`` pin, or an entry riding the default persona — gives the
cross-check nothing to compare against. Its container is not role-bound at all,
so there is no rendered role for the claim to disagree with, and the claim's
role is what the session carries. That is the honest reading: the deployment
declined to bind this entry to a role, and the only authority left is the one
that did.

**Anti-lookup.** Nothing here searches. Every resolution is keyed by the
username whose card was clicked — :meth:`RosterRoles.role_for` is a mapping
lookup, and the OIDC subject comparison is against *that* user's mapped subject
and no other. There is deliberately no helper answering "which user holds this
role" or "which user does this identity belong to": a mis-click must end in a
refusal, never in somebody else's terminal. Pinned structurally rather than by
spelling: :data:`__all__` declares the whole public surface and a test asserts
it equals what the module defines, so a reverse lookup cannot arrive under a
name nobody predicted.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from fastapi import Request

from osprey.deployment.web_terminals.personas import env_var_suffix

from .. import audit
from ..identity_headers import is_header_safe
from ..sessions import UnlockedUser

logger = logging.getLogger(__name__)

__all__ = [
    "ENV_ROSTER_ROLE_PREFIX",
    "METHOD_OIDC",
    "METHOD_PASSWORD",
    "REASON_METHOD_MISMATCH",
    "REASON_ROLE_MISMATCH",
    "REASON_UNSUPPORTED_METHOD",
    "SUPPORTED_METHODS",
    "LoginGrant",
    "RecheckRefused",
    "RosterRoles",
    "recheck_login",
    "roster_roles",
    "session_role",
    "session_subject",
]
"""This module's whole public surface, declared rather than inferred.

Pinned by a test against what the module actually defines, both ways: a new
public helper has to be *looked at* rather than merely spelled unlike the two
names an anti-lookup guard thought of. See the **Anti-lookup** note below."""

METHOD_PASSWORD = "password"
"""The roster-credential posture: this service verifies the proof itself."""

METHOD_OIDC = "oidc"
"""The federated posture: an IdP proves the identity and may decide the role."""

SUPPORTED_METHODS: frozenset[str] = frozenset({METHOD_PASSWORD, METHOD_OIDC})
"""The methods a session may be minted for.

Matched exactly, never case-folded: the value arrives from the environment
through :class:`~osprey.services.auth_sidecar.app.AuthSettings`, which already
lowercases it once. Folding again here would mean two components disagreeing
about what counts as a match, and the one that is stricter should be the one
handing out sessions."""

REASON_UNSUPPORTED_METHOD = audit.REASON_UNSUPPORTED_METHOD
"""Re-exported so a route reads its category from the module that decided it."""

REASON_METHOD_MISMATCH = audit.REASON_METHOD_MISMATCH
"""Likewise. The categories themselves live in :mod:`..audit`, which owns the
closed set an operator reads the ledger against."""

REASON_ROLE_MISMATCH = audit.REASON_ROLE_MISMATCH
"""Likewise — the federated posture's cross-check: the claim resolved to a role
other than the one this user's terminal was rendered from."""

ENV_ROSTER_ROLE_PREFIX = "OSPREY_AUTH_ROSTER_ROLE_"
"""Per-user static role: ``OSPREY_AUTH_ROSTER_ROLE_<SUFFIX>``.

The suffix is :func:`~osprey.deployment.web_terminals.personas.env_var_suffix`'s,
the same derivation that keys this user's password hash and mapped IdP subject,
so one username cannot key three variables three ways.

**Deliberately not ``OSPREY_AUTH_ROLE_``**, which would have matched the shape
of the other two per-user prefixes: ``OSPREY_AUTH_ROLE_CLAIM`` and
``OSPREY_AUTH_ROLE_MAP`` already exist and mean something else entirely (the
OIDC group binding, :mod:`~osprey.services.auth_sidecar.routes.oidc`), so a
roster user named ``claim`` or ``map`` would have read the deployment's group
binding as their own role. A collision that grants a privilege is not one to
leave to the unlikeliness of the username."""


@dataclass(frozen=True)
class RosterRoles:
    """The static role each roster user holds, as the environment carries it.

    Read from the environment directly rather than from
    :class:`~osprey.services.auth_sidecar.app.AuthSettings`, on the pattern
    :class:`~osprey.services.auth_sidecar.routes.oidc.RoleBinding` set and for
    the same reason: these belong on that class the moment it grows them, and
    until then one module reads them, caches the result on ``app.state``, and a
    test (or the factory) can install a table directly to bypass the
    environment entirely.

    Attributes:
        roles: ``{username: role}`` for the roster users that have one. A user
            with no role is *absent*, never empty-string keyed — an empty role
            is "no privileges", and storing it as a value would invite a reader
            to treat it as a role named ``""``.
    """

    roles: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls, users: tuple[str, ...] | list[str], env: Mapping[str, str] | None = None
    ) -> RosterRoles:
        """Read one role per roster user out of ``env``.

        Never raises, like every other setting this service parses: a value it
        cannot use degrades to "no role", which is the deny-safe reading, and a
        sidecar that still answers its healthcheck is what lets an operator find
        out why nobody has privileges.

        Iterating ``users`` is not a search — it is how the *keys* are built,
        from the roster this deployment declared. Nothing in this class ever
        goes the other way, from a role or an identity back to a user.

        Args:
            users: The roster, in declaration order.
            env: Environment mapping to read. Defaults to the process
                environment, which is what the container actually runs on.

        Returns:
            The parsed table.
        """
        source: Mapping[str, str] = os.environ if env is None else env
        roles: dict[str, str] = {}
        for user in users:
            role = (source.get(f"{ENV_ROSTER_ROLE_PREFIX}{env_var_suffix(user)}") or "").strip()
            if role:
                roles[user] = role
        return cls(roles=roles)

    def role_for(self, user: str) -> str:
        """The role ``user`` holds, or ``""`` when the roster gives them none.

        One keyed read. Written with ``get`` rather than a membership test
        followed by an index so that the anti-lookup property is visible in the
        code as well as pinned by a test: there is one access, by the name that
        was clicked.
        """
        return self.roles.get(user, "")


def roster_roles(request: Request) -> RosterRoles:
    """The app's one roster-role table, parsed on first use and cached.

    Cached on ``app.state`` rather than at module scope for the reason
    :func:`~osprey.services.auth_sidecar.routes.oidc._role_binding` is: two apps
    in one process must not share the first one's authorization table.
    """
    table = getattr(request.app.state, "roster_roles", None)
    if table is None:
        from ..app import get_settings

        table = RosterRoles.from_env(get_settings(request).users)
        request.app.state.roster_roles = table
    return table


@dataclass(frozen=True)
class LoginGrant:
    """What one authenticated login is allowed to carry.

    Attributes:
        subject: The account this session names — the roster username under
            ``password``, the provider's subject under ``oidc``.
        role: The role it holds, or ``""`` for none. Empty is the deny-safe
            value: verify omits the role header for it and every consumer reads
            an absent header as "no privileges".
    """

    subject: str
    role: str


class RecheckRefused(Exception):
    """The identity matrix does not admit this login.

    Carries the audited category and the sentence the browser may be told, and
    nothing else — the value that failed the matrix stays with the caller, which
    already has it and knows what it may say about it.

    Raised rather than returned as a sentinel because the callers have three
    different refusal shapes (an HTML page, a 403, nginx's bare 401) and none of
    them should be able to reach the minting code by forgetting to check a
    return value.

    Attributes:
        reason: One of :data:`~osprey.services.auth_sidecar.audit.LOGIN_REASONS`.
        message: A category, never a value, safe to put in a response.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


def recheck_login(
    *,
    method: str,
    user: str,
    roster_roles: RosterRoles,
    asserted_subject: str | None = None,
    claim_role: str | None = None,
) -> LoginGrant:
    """Re-check one authenticated login against the identity matrix.

    Called from the branch that has *already* decided the credential or the
    token was good, and before the session is minted: a matrix failure must not
    be able to leave a browser holding a session the matrix refused.

    The OIDC arguments are the seam that makes the matrix checkable. A password
    login has no IdP behind it, so supplying either of them is two flows having
    been confused — refused rather than ignored. An OIDC login must supply both:
    ``claim_role=""`` says "this deployment binds no roles", which is an answer,
    while ``None`` says "nobody asked", which is not one.

    Args:
        method: The deployment's auth method, as
            :class:`~osprey.services.auth_sidecar.app.AuthSettings` parsed it.
        user: The roster user whose card was clicked.
        roster_roles: The static per-user role table — the roles the *render*
            bound, which is what makes it the role under ``password`` and the
            cross-check target under ``oidc``. See the module docstring.
        asserted_subject: The IdP subject this login proved, under ``oidc``.
        claim_role: The role the validated ID token resolved to, under ``oidc``.

    Returns:
        What the session may carry.

    Raises:
        RecheckRefused: On any combination the matrix does not describe, on a
            claim role that disagrees with the one this user's terminal was
            rendered from, and on a role this deployment could not carry across
            the identity-header boundary.
    """
    if not user:
        # Not a defensive check: `record_login_refusal` cannot build an envelope
        # without a subject either, so a login for nobody would refuse *and*
        # vanish from the ledger. Caught here, where the caller still has its
        # own log line.
        raise RecheckRefused(REASON_METHOD_MISMATCH, "this login names no user")

    if method not in SUPPORTED_METHODS:
        raise RecheckRefused(REASON_UNSUPPORTED_METHOD, "this deployment mints no sessions")

    if method == METHOD_PASSWORD:
        if asserted_subject is not None or claim_role is not None:
            raise RecheckRefused(
                REASON_METHOD_MISMATCH, "this login was decided by the wrong method"
            )
        return _grant(subject=user, role=roster_roles.role_for(user))

    # METHOD_OIDC.
    if not asserted_subject or claim_role is None:
        raise RecheckRefused(REASON_METHOD_MISMATCH, "this login was decided by the wrong method")

    rendered_role = roster_roles.role_for(user)
    if not claim_role:
        # This deployment binds no claims, so nothing asked the provider about
        # privilege. The role is the one the RENDER bound — the same roster
        # entry the persona behind this user's door was resolved from.
        return _grant(subject=asserted_subject, role=rendered_role)

    if rendered_role and claim_role != rendered_role:
        # The cross-check. Refusing grants nothing: it turns away a login whose
        # asserted privilege names a different role than the container this user
        # would land in. Neither value goes in the message — the categories are
        # what the browser is told, and the operator has both in their own
        # config and their IdP.
        raise RecheckRefused(
            REASON_ROLE_MISMATCH, "this login's role is not the one this terminal was built as"
        )

    # Either the two agree, or the roster bound this entry no role at all — a
    # `persona:` pin or the default persona, whose container is not role-bound,
    # so there is nothing for the claim to disagree with. See the module
    # docstring's "one gap".
    return _grant(subject=asserted_subject, role=claim_role)


def _grant(*, subject: str, role: str) -> LoginGrant:
    """Build the grant, refusing a role the boundary cannot carry.

    Checked here rather than left to
    :meth:`~osprey.services.auth_sidecar.sessions.SessionState.with_user`, whose
    ``ValueError`` would surface as a 500 on what is a denial. Render-time lint
    holds role names to the username charset for exactly this reason, so
    reaching this branch means the table the sidecar was handed did not come
    through it.

    The *subject* is deliberately not checked the same way: its carriage is
    verify's gate, which denies an uncarryable one on every subrequest — a
    single decision on the one hot path, rather than a second copy of it here
    that could drift.
    """
    if role and not is_header_safe(role):
        raise RecheckRefused(
            audit.REASON_UNSAFE_ROLE, "the role this login resolved to cannot be carried"
        )
    return LoginGrant(subject=subject, role=role)


def session_subject(*, method: str, username: str, entry: UnlockedUser | None) -> str:
    """Which account an already-minted session names, by the same matrix.

    The verify half of the table. It reads the *session* rather than a fresh
    login, so it is tolerant where :func:`recheck_login` is strict: an OIDC
    entry minted before the subject was carried has an empty one, and the honest
    answer for it is "this session names no provider account" — the header is
    then omitted rather than filled with the roster name, which would report an
    account this login never asserted.

    Written as "not OIDC" so the fallback branch is the one naming an identity
    this service verified itself, never a stored claim.

    Args:
        method: The deployment's auth method.
        username: The roster user this subrequest authorized, already checked
            against the roster and byte-identical to the entry's own username.
        entry: The unlocked entry, or ``None`` on the defensive path.

    Returns:
        The subject to report, or ``""`` when this session names no account.
    """
    if method == METHOD_OIDC:
        return entry.oidc_subject if entry is not None else ""
    return username


def session_role(*, method: str, entry: UnlockedUser | None) -> str:
    """Which role an already-minted session still holds, by the same matrix.

    The role half of :func:`session_subject`, and it exists for the case that
    half already handles: a **method change made under an outstanding cookie**.
    The signing secret survives a re-render, so switching a deployment from
    ``password`` to ``oidc`` leaves every live password cookie readable — and
    the role inside it was granted by a proof this deployment no longer accepts.
    Read straight off the entry, it would keep being forwarded for the rest of
    the session lifetime, naming a privilege under a method that never asserted
    it. The password rotation an operator usually performs alongside the switch
    does not stop it either: the generation-tag check is skipped under ``oidc``.

    A password-minted entry is recognised the way :func:`session_subject`
    recognises one — it carries no ``oidc_subject`` — and its role lapses to
    ``""``, which verify turns into an omitted header rather than a default
    privilege. The reverse flip needs nothing: an OIDC entry read under
    ``password`` has an empty generation tag and fails verify's tag check
    outright, so it never reaches here.

    This is *not* the retirement of a role by config edit, which deliberately
    lapses with the session (see :func:`~..verify._authorized`). A changed
    METHOD is a different question — not "is this role still granted?" but "is
    this proof still one this deployment accepts?" — and the answer is no.

    Args:
        method: The deployment's auth method.
        entry: The unlocked entry, or ``None`` on the defensive path.

    Returns:
        The role to report, or ``""`` when this session holds none this
        deployment's current method could have granted.
    """
    if entry is None:
        return ""
    if method == METHOD_OIDC and not entry.oidc_subject:
        return ""
    return entry.role
