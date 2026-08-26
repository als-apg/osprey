"""OIDC login and callback for the auth sidecar.

Two routes under the public ``/auth/`` prefix nginx proxies here:
:data:`LOGIN_PATH` starts the handshake for one roster user and
:data:`CALLBACK_PATH` finishes it. Both exist only when the deployment runs
``auth.method: oidc``; in password mode they answer 404, because the factory
includes every route module regardless of method and "this endpoint is not part
of this deployment" is the honest answer.

**The clicked card is the username.** The landing page offers one card per
roster user, so the login route is addressed per user and the callback's job is
not "who is this person" but "is this person the user whose card was clicked".
The identity the IdP asserts is compared against
:meth:`~osprey.services.auth_sidecar.app.AuthSettings.oidc_subject` for *that*
user and nothing else: an identity that maps to no roster user, or to a
different one, is refused with 403. Searching the roster for whichever user the
subject happens to match would turn a mis-click into someone else's terminal,
which is exactly the isolation this sidecar exists to establish.

**Which user was clicked is server-side state.** It travels in the pinned
Starlette session cookie (:data:`PENDING_FLOW_SESSION_KEY`) alongside Authlib's
own ``state``/``nonce``, keyed by the same ``state`` value — never as a query
parameter on the ``redirect_uri``, which the browser could edit between the two
legs of the handshake and thereby pick which roster user a successful IdP login
unlocks. The session cookie is signed with the state secret, so its contents are
the sidecar's own word.

**The identity comes from the verified ID token.** Authlib validates the token's
signature, issuer, audience and nonce before
:meth:`authorize_access_token` returns it as ``token["userinfo"]``; this module
reads the configured claim from there and never calls the userinfo endpoint. The
audience is on that list only because :func:`_claims_options` asks for it —
Authlib checks ``aud`` when the caller names it and not otherwise, so removing
that argument would silently reduce the audience check to the OIDC ``azp`` rule.
A deployment whose IdP carries the mapped claim only at that endpoint therefore
locks out with a log line naming the claim, rather than authorising on a second,
weaker trust path.

**The role comes from the same token, and only from it.** When the deployment
binds roles to an IdP group claim (:class:`RoleBinding`), the claim is read out
of those already-validated claims — the same ones the identity came from — and
resolved by *intersecting* its values with the configured map. Two rules make
that resolution safe to hand a privilege to: the intersection must name exactly
one distinct role, and anything else fails the login closed under its own
audited category. An empty intersection is "this deployment maps nothing to
what you are in"; more than one distinct role is ambiguity, and picking the
first would make the granted privilege depend on the order the provider
happened to list groups in. A claim that never arrived — Entra's group overage
strips ``groups`` from the ID token and leaves a pointer to Microsoft Graph
behind — is the missing-claim refusal, not a fallback: the userinfo endpoint
this module refuses to call could not have answered it either.

**Nothing about a failure reaches the browser but its category.** Tokens,
client secrets, and the claim values examined while validating a login never
enter a response or a log line; refusals name the user and the reason class
only. The one identity value that is retained is the accepted subject: a
successful login stores it in the (signed, not encrypted) session cookie and
records it on the success log line, because it is an opaque account identifier
rather than a credential and a later verify subrequest reports it back.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse

from .. import audit
from ..app import (
    AuthSettings,
    get_audit_throttle,
    get_revocation_store,
    get_session_codec,
    get_settings,
)
from ..exceptions import InvalidSessionError
from ..identity_headers import is_header_safe
from ..return_to import safe_return_to
from ..sessions import SESSION_COOKIE_NAME, SessionState
from ..throttle import AttemptThrottle
from .recheck import RecheckRefused, recheck_login, roster_roles

logger = logging.getLogger(__name__)

router = APIRouter()

LOGIN_PATH = "/auth/oidc/login"
"""Starts the handshake for one roster user: ``?user=<name>&next=<return-to>``."""

CALLBACK_PATH = "/auth/oidc/callback"
"""Where the IdP sends the browser back. Registered with the IdP as
:attr:`~osprey.services.auth_sidecar.app.AuthSettings.external_origin` + this
path — both legs must agree on it byte for byte, and the token exchange repeats
it, so it is derived from the deployment's own origin rather than from the
inbound request (which arrives from nginx over loopback and names the wrong
host)."""

CLIENT_NAME = "osprey_oidc"
"""Authlib client name. It namespaces Authlib's own session entries
(``_state_osprey_oidc_<state>``), so it is part of the state cookie's shape."""

PENDING_FLOW_SESSION_KEY = "osprey_oidc_pending"
"""State-cookie key holding the in-flight handshake: ``{"state", "user", "next"}``.

One entry, not a map: Authlib's Starlette integration drops every previous
``_state_*`` entry each time it stores a new one, so a browser can only ever
have one handshake it could complete. A second entry here would outlive the
Authlib data it is paired with and could only ever fail the state check."""

DEFAULT_SCOPE = "openid profile email"
"""Requested scopes.

``openid`` is what makes this OIDC rather than bare OAuth2 (it is also what
makes Authlib generate and check a nonce). ``profile`` and ``email`` are asked
for because the claim a facility maps onto a roster user is commonly
``preferred_username`` or ``email``, and a claim that was never requested is
simply absent from the token — which this module reads as "deny"."""

ENV_ROLE_CLAIM = "OSPREY_AUTH_ROLE_CLAIM"
"""Names the ID-token claim carrying group membership, e.g. ``groups``.

Set from the rendered ``modules.web_terminals.authorization.claims.claim``.
Empty or absent means this deployment binds no roles, and every login resolves
the deny-safe empty role — the shape every ``oidc`` deployment that predates the
``authorization:`` block keeps running in."""

ENV_ROLE_MAP = "OSPREY_AUTH_ROLE_MAP"
"""The claim-value → role table, as one compact JSON object.

JSON rather than a delimited list because the values are the provider's, not
ours: an Entra group id is a GUID, but a Keycloak or LDAP-backed group name can
carry commas, colons and equals signs (``CN=Operators,OU=Groups``), and every
delimiter this could have picked is a character some directory puts inside a
name. Compact (no space after ``:``) so the value stays a plain YAML scalar
where the compose file writes it.

Unreadable content — hand-edited JSON, or JSON of the wrong shape — leaves the
table empty rather than disabling the binding: the claim name is what says
"this deployment binds roles", so an unreadable map refuses every login instead
of silently downgrading them all to the roleless one."""

# --- refusal categories ------------------------------------------------------
#
# Defined in :mod:`~osprey.services.auth_sidecar.audit` beside the rest of the
# closed set — one place enumerates what an operator may find in a record's
# `reason`, rather than each route owning the half it happens to raise. Kept
# addressable under this module's names because these are the categories the
# OIDC flow is read and tested through, and a call site naming its own module
# is what makes a refusal readable at the point of decision.

REASON_UNMAPPED_USER = audit.REASON_UNMAPPED_USER
"""The clicked roster user has no mapped IdP identity at all."""

REASON_NO_ASSERTED_IDENTITY = audit.REASON_NO_ASSERTED_IDENTITY
"""The validated ID token carried no usable value in the identity claim."""

REASON_IDENTITY_MISMATCH = audit.REASON_IDENTITY_MISMATCH
"""The asserted identity belongs to a different roster user, or to none."""

REASON_UNVALIDATED_TOKEN = audit.REASON_UNVALIDATED_TOKEN
"""The token response carried no ID token, so no claim from it is trustworthy."""

REASON_MISSING_ROLE_CLAIM = audit.REASON_MISSING_ROLE_CLAIM
"""The group claim did not arrive, or arrived in a shape carrying no value."""

REASON_UNMAPPED_ROLE_CLAIM = audit.REASON_UNMAPPED_ROLE_CLAIM
"""No value in the group claim is mapped to a role by this deployment."""

REASON_AMBIGUOUS_ROLE_CLAIM = audit.REASON_AMBIGUOUS_ROLE_CLAIM
"""The group claim maps to more than one distinct role."""

REASON_UNSAFE_ROLE = audit.REASON_UNSAFE_ROLE
"""The resolved role cannot be carried in an identity header."""


@dataclass(frozen=True)
class RoleBinding:
    """How this deployment turns an IdP group claim into one role.

    Parsed from the two environment variables the rendered compose file exports
    (:data:`ENV_ROLE_CLAIM`, :data:`ENV_ROLE_MAP`) and cached on ``app.state``,
    on the same pattern as the Authlib client — the sidecar's other settings
    live on :class:`~osprey.services.auth_sidecar.app.AuthSettings`, and these
    two belong there the moment that class grows them; until then this is the
    one place that reads them, and the factory or a test can install a binding
    directly to bypass the environment entirely.

    Attributes:
        claim: The ID-token claim carrying group membership. Empty when the
            deployment binds no roles.
        claim_map: ``{claim value: role}``. Every role in it is a role the
            render declared; the sidecar does not re-check that, because it
            never sees the persona table.
    """

    claim: str = ""
    claim_map: Mapping[str, str] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        """Whether this deployment binds roles at all.

        True if *either* half is set. Half a binding cannot resolve anything, so
        it denies: a deployment that named a claim but lost its table, or shipped
        a table with no claim to read it from, is misconfigured in a way that
        must not read as "this deployment has no roles" — that reading would turn
        a broken authorization block into an unauthorized login.
        """
        return bool(self.claim or self.claim_map)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> RoleBinding:
        """Parse the binding out of ``env`` (default :data:`os.environ`).

        Never raises: like every other setting this service reads, a malformed
        value degrades to the fail-closed reading rather than taking the sidecar
        down — an unreadable table refuses logins, and a healthcheck that still
        answers is what lets an operator find out why.

        Args:
            env: Environment mapping to read. Defaults to the process
                environment, which is what the container actually runs on.

        Returns:
            The parsed binding.
        """
        source: Mapping[str, str] = os.environ if env is None else env
        claim = (source.get(ENV_ROLE_CLAIM) or "").strip()
        raw = (source.get(ENV_ROLE_MAP) or "").strip()

        claim_map: dict[str, str] = {}
        if raw:
            try:
                parsed = json.loads(raw)
            except ValueError:
                logger.warning(
                    "%s is not readable as JSON: no group maps to a role, so every oidc login "
                    "that needs one will be refused",
                    ENV_ROLE_MAP,
                )
            else:
                if isinstance(parsed, dict):
                    claim_map = {
                        key: value
                        for key, value in parsed.items()
                        # Entries that are not string→string cannot have come
                        # from the render, and a non-string key could never
                        # match a claim value anyway. Dropped rather than
                        # refused: what remains is a strictly smaller table, so
                        # the failure direction is still "no role", never "the
                        # wrong role".
                        if isinstance(key, str) and isinstance(value, str) and key and value
                    }
                else:
                    logger.warning(
                        "%s is not a JSON object: no group maps to a role, so every oidc login "
                        "that needs one will be refused",
                        ENV_ROLE_MAP,
                    )

        return cls(claim=claim, claim_map=claim_map)


def _role_binding(request: Request) -> RoleBinding:
    """The app's one role binding, parsed on first use and cached.

    Cached on ``app.state`` rather than at module scope for the reason
    :func:`_oauth_client` is: two apps in one process must not share the first
    one's authorization table. Tests (and, later, the factory) assign
    ``app.state.role_binding`` before the first request.
    """
    binding = getattr(request.app.state, "role_binding", None)
    if binding is None:
        binding = RoleBinding.from_env()
        request.app.state.role_binding = binding
    return binding


def _claim_values(raw: Any) -> tuple[str, ...]:
    """The usable values in a group claim, whatever shape it arrived in.

    Providers disagree about the shape and both spellings are legitimate: Entra
    and Keycloak send an array (of GUIDs and of names respectively), while a
    deployment mapping a single app role commonly sends a bare string.

    Anything else — an object, a number, ``null`` — yields nothing, which the
    caller reads as "the claim did not arrive". Non-string entries inside an
    array are dropped rather than poisoning the strings beside them: a value
    that is not a string could never match a key of the table, so dropping it
    cannot widen what the login resolves to.

    Args:
        raw: The claim's value as it came out of the validated ID token.

    Returns:
        The non-empty string values, in arrival order.
    """
    if isinstance(raw, str):
        return (raw,) if raw else ()
    if isinstance(raw, (list, tuple)):
        return tuple(value for value in raw if isinstance(value, str) and value)
    return ()


def _claim_names(claims: Mapping[str, Any]) -> str:
    """The names of the claims that arrived, sorted, for a diagnostic line.

    Names only, never values: a claim value can be an email address, a display
    name, or a whole group membership, and none of that belongs in a service
    log. The names alone are what makes the missing-claim refusal actionable —
    seeing ``_claim_names`` where ``groups`` was expected *is* the Entra overage
    diagnosis.
    """
    return ", ".join(sorted(str(name) for name in claims))


FOLDED_DETAIL_KEY = "folded_refusals"
"""Ledger detail naming how many refusals a filed record stands in for.

``folded_refusals=417`` on a record means 417 further refusals for that same
user arrived while its window was held open and were not filed. Zero folds
carry no note at all, so an ordinary record reads exactly as it did before —
the key appears only where something *was* suppressed. See
:class:`_LedgerBound`.
"""


@dataclass(frozen=True)
class _LedgerBound:
    """The window that bounds how often the ledger repeats one refusal.

    Not the login throttle. This one is grown by refusals that were never
    evaluated, which is precisely what
    :mod:`~osprey.services.auth_sidecar.throttle` forbids doing to the window
    that decides logins — doing it there would let anyone spamming
    ``/auth/oidc/login?user=<roster user>`` delay that operator's real login.
    The app builds a second instance for this use alone
    (:func:`~osprey.services.auth_sidecar.app.get_audit_throttle`), so the two
    key spaces can never meet however the routes are later rewired.

    Attributes:
        throttle: The app's audit window, keyed on the roster user.
        folded: How many refusals have been suppressed for each user since the
            last record was filed for them. Bounded by the roster: every caller
            has already checked the name against it.
    """

    throttle: AttemptThrottle
    folded: dict[str, int]

    def file_or_fold(self, user: str) -> int | None:
        """Whether to file this refusal, and what it stands in for if so.

        Returns:
            ``None`` when the window is open and this refusal is folded into
            the record already filed; otherwise how many refusals were folded
            since that record — ``0`` when this is simply the next one.
        """
        if self.throttle.retry_after(user) > 0:
            self.folded[user] = self.folded.get(user, 0) + 1
            return None
        self.throttle.record_failure(user)
        return self.folded.pop(user, 0)


def _ledger_bound(request: Request) -> _LedgerBound:
    """The app's ledger window, paired with its per-user fold counts.

    The counts are cached on ``app.state`` for the reason
    :func:`_role_binding` caches there: two apps in one process must not share
    the first one's state. The window itself is built by the factory.
    """
    folded = getattr(request.app.state, "folded_refusals", None)
    if folded is None:
        folded = {}
        request.app.state.folded_refusals = folded
    return _LedgerBound(throttle=get_audit_throttle(request), folded=folded)


def _folded_detail(detail: str | None, folded: int) -> str | None:
    """``detail`` with the fold count appended, or unchanged when nothing was folded."""
    if folded <= 0:
        return detail
    note = f"{FOLDED_DETAIL_KEY}={folded}"
    return note if detail is None else f"{detail}; {note}"


def _refuse_login(
    user: str,
    *,
    reason: str,
    message: str,
    detail: str | None = None,
    bound: _LedgerBound | None = None,
    status_code: int = 403,
) -> HTTPException:
    """Audit a refused login and return the response to raise for it.

    Returned rather than raised so the call site reads as the refusal it is
    (``raise _refuse_login(...)``) and keeps its own log line, which says the
    same thing in the operator's language.

    **``bound`` bounds the ledger, not the answer.** The refusals reachable
    *before* the token exchange cost an unauthenticated caller one GET apiece —
    ``/auth/oidc/login?user=<a roster user with no mapped subject>`` is a free
    request that would otherwise append a record every time. The file is
    root-owned in a directory nothing else binds, so nobody inside the
    deployment can rotate or truncate it; unbounded append at request rate is
    disk pressure against the audit zone and it buries the genuine refusals an
    operator greps for. Passing the bound records the *first* refusal in each
    window and folds the rest into it. The refusal itself is unchanged: the
    caller gets the same status and the same message every time, because the
    decision has not changed either — only the number of times the ledger says
    so.

    **A folded refusal is still countable.** What is suppressed is the record,
    not the fact: the next record filed for that user carries
    ``folded_refusals=<n>`` naming how many arrived in between. That matters
    because an attacker holding a user's window open with one category masks a
    genuine refusal of another for the same name — the ledger cannot show that
    category, but it can show that the silence was not quiet, so a reader can
    tell one refusal from four hundred.

    Post-exchange refusals pass no bound. Reaching one costs a full IdP round
    trip, so they are not free to generate, and they are the records that
    describe a login that actually authenticated.

    Args:
        user: The roster user whose login is being refused.
        reason: One of this module's ``REASON_*`` categories.
        message: What the browser is told. A category, never a value.
        detail: Supplementary context for the ledger — config keys and role
            names only, never a claim value.
        bound: The app's ledger window, on the pre-exchange refusals that an
            unauthenticated caller can reach at request rate.
        status_code: The status to raise. 403 for a login this deployment
            refuses; 502 where the *provider's* answer is what failed.

    Returns:
        The exception to raise.
    """
    if bound is None:
        audit.record_login_refusal(user=user, reason=reason, detail=detail)
    else:
        folded = bound.file_or_fold(user)
        if folded is not None:
            audit.record_login_refusal(
                user=user, reason=reason, detail=_folded_detail(detail, folded)
            )
    return HTTPException(status_code=status_code, detail=message)


def _resolved_role(request: Request, *, user: str, claims: Mapping[str, Any]) -> str:
    """The one role ``claims`` resolves to, or ``""`` when none is bound.

    Args:
        request: The callback request, carrying the app's role binding.
        user: The roster user being logged in.
        claims: The validated ID token's claims.

    Returns:
        The resolved role, or ``""`` for a deployment that binds no roles.

    Raises:
        HTTPException: 403, audited, when the claim is missing, maps to nothing,
            maps to more than one distinct role, or names a role that could not
            be carried in an identity header.
    """
    binding = _role_binding(request)
    if not binding.configured:
        return ""

    logger.debug(
        "oidc callback for %r: the ID token carried these claims: %s", user, _claim_names(claims)
    )

    values = _claim_values(claims.get(binding.claim)) if binding.claim else ()
    if not values:
        logger.warning(
            "oidc callback refused for %r: the ID token carries no usable %r claim; it carried: %s",
            user,
            binding.claim,
            _claim_names(claims),
        )
        raise _refuse_login(
            user,
            reason=REASON_MISSING_ROLE_CLAIM,
            message="the identity provider asserted no group membership",
            # A binding with a map but no claim NAME reaches here too, and it is
            # a different fault from a genuine overage: naming the variable that
            # is missing is what tells the operator to look at their own render
            # rather than at their IdP. Both are config keys, never values.
            detail=binding.claim or ENV_ROLE_CLAIM,
        )

    # Intersection, not first match: the granted privilege must not depend on
    # the order the provider listed groups in.
    roles = {binding.claim_map[value] for value in values if value in binding.claim_map}

    if not roles:
        logger.warning(
            "oidc callback refused for %r: no role is mapped to any value of the %r claim",
            user,
            binding.claim,
        )
        raise _refuse_login(
            user,
            reason=REASON_UNMAPPED_ROLE_CLAIM,
            message="no role is mapped to this account's group membership",
            detail=binding.claim or None,
        )

    # Header safety is settled for EVERY candidate before anything else reads a
    # role name, not just for the one that survives. Two reasons, and the second
    # is why it moved up here: a map that names an uncarryable role is a
    # poisoned table whatever the intersection turns out to be, and the
    # ambiguity refusal below puts the role names it found into the audit
    # record — so an unchecked candidate is a deployer-supplied string with CR
    # and LF in it landing in the ledger's `detail`, which bounds length but
    # validates no charset.
    if any(not is_header_safe(candidate) for candidate in roles):
        logger.warning(
            "oidc callback refused for %r: a role mapped to the %r claim cannot be carried in "
            "an identity header",
            user,
            binding.claim,
        )
        raise _refuse_login(
            user,
            reason=REASON_UNSAFE_ROLE,
            message="the role this login resolved to cannot be carried",
            # The claim name, never the offending role name: the value is the
            # deployment's own config, and it is the thing that must not reach
            # a record unvalidated.
            detail=binding.claim or None,
        )

    if len(roles) > 1:
        logger.warning(
            "oidc callback refused for %r: the %r claim maps to more than one role (%s)",
            user,
            binding.claim,
            ", ".join(sorted(roles)),
        )
        raise _refuse_login(
            user,
            reason=REASON_AMBIGUOUS_ROLE_CLAIM,
            # Role names are this deployment's own identifiers — the actionable
            # half — while the group values that produced them are the IdP's and
            # stay out of the ledger. Every one of them passed the header-safety
            # gate above, so the join cannot carry a control character.
            detail=", ".join(sorted(roles)),
            message="this account's group membership maps to more than one role",
        )

    return roles.pop()


def _discovery_url(issuer: str) -> str:
    """The OIDC discovery document for ``issuer``."""
    return f"{issuer.rstrip('/')}/.well-known/openid-configuration"


def _require_oidc_mode(settings: AuthSettings) -> None:
    """Refuse with 404 unless this deployment authenticates with OIDC.

    Raises:
        HTTPException: 404 in password mode. Not 403: the endpoint is not part
            of this deployment's surface at all, and saying "forbidden" would
            imply an identity was evaluated.
    """
    if settings.method != "oidc":
        raise HTTPException(status_code=404, detail="this deployment does not use OIDC login")


def _oauth_client(request: Request) -> Any:
    """The app's one Authlib client, built on first use and cached.

    Cached on ``app.state`` rather than at module scope: two apps in one process
    (every test that builds a second one) would otherwise share a client
    configured for the first app's issuer and credentials. Building it lazily
    also keeps the discovery fetch off the factory's path — the sidecar starts
    and answers its healthcheck whether or not the IdP is reachable — and the
    cached client holds the fetched metadata, so discovery happens once rather
    than per login.

    Tests replace it by assigning ``app.state.oidc_client`` before the first
    request.
    """
    client = getattr(request.app.state, "oidc_client", None)
    if client is not None:
        return client

    settings = get_settings(request)
    registry = OAuth()
    registry.register(
        name=CLIENT_NAME,
        server_metadata_url=_discovery_url(settings.oidc_issuer or ""),
        client_id=settings.oidc_client_id,
        client_secret=settings.oidc_client_secret,
        client_kwargs={"scope": DEFAULT_SCOPE},
    )
    client = registry.create_client(CLIENT_NAME)
    request.app.state.oidc_client = client
    return client


def _same_value(left: str, right: str) -> bool:
    """Whether two text values are equal, compared in constant time.

    As UTF-8 bytes, never as ``str``: :func:`secrets.compare_digest` raises
    ``TypeError`` the moment either side carries a non-ASCII character, and both
    things compared here can. A ``state`` parameter is unauthenticated input, so
    one accented character in it would be an unhandled 500 rather than a
    refusal; a mapped identity like ``jörg@example.org`` would raise on the
    comparison that was about to *succeed*, locking that operator out for good.
    """
    return secrets.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def _claims_options(settings: AuthSettings) -> dict[str, dict[str, list[str]]]:
    """Which ID-token claims Authlib must check, and against what.

    **``aud`` is only checked when it is named here.** Authlib's own default
    supplies ``iss`` and nothing else, so without this an ID token minted for a
    *different* relying party validates as long as it carries an ``azp`` naming
    this client — the OIDC ``azp`` rule is then the only thing standing in the
    way, and an IdP emits ``azp`` at its discretion. Naming ``aud`` makes the
    audience check this deployment's own rather than a side effect of a claim
    that may never arrive.

    **``iss`` is repeated here because this mapping replaces Authlib's default
    rather than extending it.** Omitting it would trade the audience hole for an
    issuer one.

    **The group claim is deliberately absent.** This is a validation contract —
    "these claims must be present and must equal these values" — and a group
    claim has no expected value to check against: naming it here would demand
    every roster user carry the same membership. Its own absence is a decision
    :func:`_resolved_role` makes, with a category and an audit record; Authlib's
    would be an unvalidatable-token 502 that says nothing about roles.

    Both spellings of the issuer are accepted. OIDC Discovery requires the
    published ``issuer`` to equal the prefix its document was fetched from, and
    :func:`_discovery_url` strips one trailing slash off that prefix — so a
    deployment whose configured issuer differs from the IdP's published one by
    exactly that character is correctly configured, and locking it out would be
    this function inventing a requirement neither party has.

    Args:
        settings: The deployment's settings. Both values it reads are
            deployment-wide requirements in OIDC mode, so the configuration
            guard has already refused every request that could reach here
            without them.

    Returns:
        A ``claims_options`` mapping for
        :meth:`authorize_access_token`.
    """
    issuer = (settings.oidc_issuer or "").strip()
    trimmed = issuer.rstrip("/")
    return {
        "iss": {"values": [issuer] if issuer == trimmed else [issuer, trimmed]},
        "aud": {"values": [settings.oidc_client_id]},
    }


def _current_session(request: Request) -> SessionState:
    """The browser's auth session, or a fresh one.

    An unreadable cookie — tampered, stale-keyed, or from a retired payload
    version — is not an error here: it carries no authorisation, so the login
    it is about to be replaced by starts from an empty session.

    **A revoked session id is not inherited.** Logout revokes by id and verify
    refuses every cookie carrying a revoked one, so a callback that kept the id
    would re-sign a cookie verify then rejects — and the next handshake would
    inherit it again, locking the browser out of the terminal it just
    authenticated for until the entry expired. Retiring the id here is what makes
    a logout followed by a fresh login mean what it says.
    """
    codec = get_session_codec(request)
    raw = request.cookies.get(SESSION_COOKIE_NAME)
    if not raw:
        return codec.new_state()
    try:
        state = codec.decode(raw)
    except InvalidSessionError:
        logger.info("discarding an unreadable auth session cookie during oidc login")
        return codec.new_state()

    if get_revocation_store(request).is_revoked(state.session_id):
        logger.info("starting a new session: the cookie presented at oidc login had been revoked")
        return codec.new_state()
    return state


@router.get(LOGIN_PATH)
async def oidc_login(
    request: Request,
    user: str = Query(..., description="Roster user whose card was clicked"),
    return_to: str | None = Query(None, alias="next", description="Same-origin path to return to"),
) -> Response:
    """Redirect to the IdP to authenticate as ``user``.

    Args:
        request: The inbound request.
        user: The roster user being logged in.
        return_to: Where to send the browser after a successful callback.

    Returns:
        A redirect to the IdP's authorization endpoint.

    Raises:
        HTTPException: 404 when this deployment is not in OIDC mode or ``user``
            is not on the roster; 403 when ``user`` has no mapped IdP identity
            (the handshake could only end in a refusal, so it does not start);
            502 when the issuer's discovery document cannot be fetched or read.
    """
    settings = get_settings(request)
    _require_oidc_mode(settings)

    if not settings.knows_user(user):
        # Not a login decision but a routing error: no login was attempted for
        # this name, because this deployment has no card for it. The roster is
        # public (the landing page lists it), so saying so leaks nothing — and
        # the asymmetry with the password path, where a name that was never on
        # the roster IS recorded under `bad_credential`, is deliberate: there a
        # credential was evaluated, and refusing to say which of three reasons
        # applied is what keeps the ledger from enumerating accounts.
        raise HTTPException(status_code=404, detail="no such user")
    if settings.oidc_subject(user) is None:
        logger.warning("oidc login refused for %r: no IdP identity is mapped to that user", user)
        raise _refuse_login(
            user,
            reason=REASON_UNMAPPED_USER,
            message="this user has no mapped identity provider",
            # Reachable by an unauthenticated GET, one request per record.
            bound=_ledger_bound(request),
        )

    target = safe_return_to(return_to, user, flow="oidc login")
    client = _oauth_client(request)
    redirect_uri = f"{settings.external_origin}{CALLBACK_PATH}"

    # Deliberately not `authorize_redirect`, which is these three steps with the
    # state value kept inside it. The state is what binds this handshake to the
    # user whose card was clicked, so this route needs it in hand.
    try:
        authorization = await client.create_authorization_url(redirect_uri)
    except httpx.HTTPError:
        logger.warning("oidc login failed: the issuer's discovery document is unreachable")
        raise HTTPException(
            status_code=502, detail="the identity provider could not be reached"
        ) from None
    except Exception as exc:
        # Reached, not defensive: a discovery document that is served but wrong
        # — HTML from a captive portal, JSON with no authorization_endpoint —
        # raises out of Authlib rather than out of httpx, and a mistyped issuer
        # is where that shows up first. Same reasoning as the callback's broad
        # arm, and the same discipline: class name only, never the response.
        logger.warning(
            "oidc login failed: the issuer's discovery document is unusable (%s)",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=502, detail="the identity provider's configuration could not be read"
        ) from None

    await client.save_authorize_data(request, redirect_uri=redirect_uri, **authorization)
    request.session[PENDING_FLOW_SESSION_KEY] = {
        "state": authorization["state"],
        "user": user,
        "next": target,
    }
    return RedirectResponse(authorization["url"], status_code=302)


@router.get(CALLBACK_PATH)
async def oidc_callback(request: Request) -> Response:
    """Finish the handshake and unlock the user whose card was clicked.

    Args:
        request: The inbound request, carrying the IdP's ``code`` and ``state``.

    Returns:
        A redirect to the validated return-to, carrying the re-issued auth
        session cookie.

    Raises:
        HTTPException: 404 outside OIDC mode; 400 when no handshake is in flight
            for this browser, when the returned ``state`` does not match the one
            that started it, or when the IdP reports an error; 403 when the
            clicked user has no mapped identity, when that identity cannot be
            carried in an identity header, when the asserted identity is not the
            one mapped to the clicked user, or when a bound role cannot be
            resolved to exactly one carryable role; 502 when the IdP is
            unreachable or its response does not validate. Every 403 is recorded
            under its own audited category.
    """
    settings = get_settings(request)
    _require_oidc_mode(settings)

    # Popped before anything is validated, so a callback that fails any check
    # below still consumes the pending flow. The trade is deliberate: it costs a
    # cancelled login (any page that can reach this URL can make the operator
    # click their card again), and it buys a handshake that cannot be replayed,
    # probed repeatedly, or left half-open in the cookie. A login is cheap to
    # restart; a reusable one is not cheap to hold.
    pending = request.session.pop(PENDING_FLOW_SESSION_KEY, None)
    if not isinstance(pending, dict) or not pending.get("state") or not pending.get("user"):
        logger.warning("oidc callback rejected: no login is in flight for this browser")
        raise HTTPException(status_code=400, detail="no login is in progress")

    # Checked here as well as inside Authlib, and against our own record of the
    # state: Authlib proves the callback belongs to a handshake this browser
    # started, while this proves it belongs to the handshake that carried this
    # user. Without it a browser holding two half-finished flows could complete
    # one of them under the other's username.
    returned_state = request.query_params.get("state") or ""
    if not _same_value(str(pending["state"]), returned_state):
        logger.warning("oidc callback rejected: returned state does not match the login in flight")
        raise HTTPException(status_code=400, detail="login state mismatch")

    user = str(pending["user"])
    expected_subject = settings.oidc_subject(user)
    if expected_subject is None:
        # Unreachable through the login route, which refuses first; re-checked
        # because this is the decision that must never fall through to "some
        # other user's subject matched".
        logger.warning("oidc callback refused for %r: no IdP identity is mapped to that user", user)
        raise _refuse_login(
            user,
            reason=REASON_UNMAPPED_USER,
            message="this user has no mapped identity provider",
            # Pre-exchange, and the state cookie that gets here is minted by a
            # free GET — two requests per record without the bound.
            bound=_ledger_bound(request),
        )

    if not is_header_safe(expected_subject):
        # The mapped identity has to reach the terminal as an
        # `X-Osprey-Auth-Subject` header, which is latin-1 on the wire and
        # ASCII by this deployment's stricter rule. A subject that cannot make
        # that trip is refused *here*, before the token exchange: the login
        # could only have ended in a mangled identity or a dropped header, and
        # a login that silently grants less identity than it says is worse than
        # one that does not happen. An OIDC `sub` is ASCII by specification, so
        # this is a configuration fault — the deployment mapped a claim
        # spelling the boundary cannot carry — and the audited category says
        # exactly that. The value itself stays out of both the record and the
        # log line; the operator has it in their own config.
        logger.warning(
            "oidc callback refused for %r: the mapped identity cannot be carried in an "
            "identity header",
            user,
        )
        raise _refuse_login(
            user,
            reason=audit.REASON_NON_ASCII_SUBJECT,
            message="this user's mapped identity cannot be carried",
            # Pre-exchange, same bound as the refusal above.
            bound=_ledger_bound(request),
        )

    client = _oauth_client(request)
    try:
        token = await client.authorize_access_token(
            request, claims_options=_claims_options(settings)
        )
    except OAuthError as exc:
        # The IdP reported an error, or Authlib's own state check failed.
        logger.warning("oidc callback rejected by the authorization step: %s", type(exc).__name__)
        raise HTTPException(
            status_code=400, detail="the identity provider rejected the login"
        ) from None
    except httpx.HTTPError:
        logger.warning(
            "oidc callback failed: the identity provider's token endpoint is unreachable"
        )
        raise HTTPException(
            status_code=502, detail="the identity provider could not be reached"
        ) from None
    except Exception as exc:  # ID-token validation: signature, issuer, audience, nonce, claims.
        # Deliberately broad. The concrete types come from whichever JWT library
        # the installed Authlib delegates to, and pinning them here would turn a
        # dependency bump into a 500 on a token that should simply be refused.
        # Only the class name is logged: an exception from claim validation can
        # carry claim material, and none of it belongs in the service log.
        logger.warning(
            "oidc callback rejected: the ID token did not validate (%s)", type(exc).__name__
        )
        raise HTTPException(
            status_code=502, detail="the identity provider's response could not be validated"
        ) from None

    if "id_token" not in token:
        # Authlib fills `token["userinfo"]` with the PARSED, validated ID token
        # only when the token response carried an `id_token` (and the state a
        # nonce). Without one, that key is never written — and whatever the
        # token endpoint's own JSON body happened to carry under the name
        # `userinfo` would flow straight through as claims: unsigned, never
        # seen by `_claims_options`, and now load-bearing for a ROLE. An OAuth2
        # provider where an OIDC one was configured is a deployment fault of
        # the same class as an ID token that fails validation, so it is refused
        # the same way rather than continuing on the access token alone.
        logger.warning(
            "oidc callback rejected for %r: the token response carried no ID token", user
        )
        # Filed, unlike the validation arms above it: this is the one refusal in
        # the module driven by a hostile or substituted token endpoint rather
        # than by configuration, which makes it the closest thing this service
        # has to an attack signal — and the one denial an operator would want to
        # investigate from the ledger rather than from a log line. The status
        # stays 502: the record does not change the answer, and what failed is
        # the provider's response, not this user's login.
        raise _refuse_login(
            user,
            reason=REASON_UNVALIDATED_TOKEN,
            message="the identity provider's response could not be validated",
            status_code=502,
        )

    claims = token.get("userinfo") or {}
    asserted = claims.get(settings.oidc_claim)
    if not isinstance(asserted, str) or not asserted:
        logger.warning(
            "oidc callback refused for %r: the ID token carries no usable %r claim",
            user,
            settings.oidc_claim,
        )
        raise _refuse_login(
            user,
            reason=REASON_NO_ASSERTED_IDENTITY,
            message="the identity provider asserted no identity",
            # The claim *name* is configuration and is what an operator has to
            # change; the value that was (not) in it never enters the record.
            detail=settings.oidc_claim,
        )

    if not _same_value(asserted, expected_subject):
        # No search of the roster for a user this identity *would* match: the
        # card that was clicked is the only user this login can unlock.
        logger.warning(
            "oidc callback refused for %r: the asserted identity is mapped to a different user "
            "or to none",
            user,
        )
        raise _refuse_login(
            user,
            reason=REASON_IDENTITY_MISMATCH,
            message="this identity is not permitted for this user",
        )

    # Identity first, privilege second, and both from the same validated token:
    # the role question is only worth asking about a login that already proved
    # it is the user whose card was clicked.
    claim_role = _resolved_role(request, user=user, claims=claims)

    # The same matrix the password path is held to, asked the same way and
    # before anything is minted. `expected_subject` and `claim_role` are what
    # this method is *allowed* to supply; a deployment that binds no roles
    # supplies `""`, which is an answer, and the re-check refuses a caller that
    # supplies neither.
    try:
        grant = recheck_login(
            method=settings.method,
            user=user,
            roster_roles=roster_roles(request),
            asserted_subject=expected_subject,
            claim_role=claim_role,
        )
    except RecheckRefused as refused:
        logger.warning("oidc callback refused for %r: %s", user, refused.reason)
        raise _refuse_login(user, reason=refused.reason, message=refused.message) from None
    role = grant.role

    codec = get_session_codec(request)
    now = codec.now()
    # No generation tag: that is the password mode's rotation signal, computed
    # from a stored hash which OIDC has none of. An OIDC entry is bounded by its
    # expiry and by logout alone. It carries the asserted subject instead — an
    # opaque account identifier, not a credential — so a later verify subrequest
    # can name which provider account is behind this unlocked user without
    # re-contacting the IdP. `expected_subject` and `asserted` are byte-equal
    # here (the constant-time check above just proved it); the configured value
    # is stored so the cookie carries the deployment's own canonical spelling.
    session = _current_session(request).with_user(
        user,
        expires_at=now + settings.session_lifetime,
        generation_tag="",
        oidc_subject=grant.subject,
        # The matrix's answer, not this route's: the claim's role where this
        # deployment binds claims (cross-checked there against the role the
        # render bound for this user), and the roster's own where it binds
        # none. Empty means "no privileges" — the deny-safe value, which verify
        # turns into an omitted role header rather than a default privilege.
        # Every other outcome refused the login above, so `with_user` can only
        # be reached with a role it can carry.
        role=role,
    )

    response = RedirectResponse(
        safe_return_to(pending.get("next"), user, flow="oidc login"), status_code=303
    )
    response.set_cookie(
        SESSION_COOKIE_NAME,
        codec.encode(session),
        httponly=True,
        samesite="lax",
        secure=settings.tls_enabled,
        # Not the state cookie's "/auth": this one has to reach the terminals it
        # authorises, whose verify subrequest nginx issues from "/u/<user>/".
        path="/",
    )
    # The subject is a claim value, but an opaque account identifier rather than
    # a credential (the cookie already carries it, signed not encrypted), so it
    # is recorded on this success line. `%r` quotes it, so a value carrying a
    # newline cannot forge a second log line.
    logger.info("oidc login succeeded for %r (subject %r)", user, expected_subject)
    # The counterpart of `_refuse_login`: the same seam, one record, the roster
    # user as the subject. The asserted subject stays out of it — it is a claim
    # value, and the record names the roster user the login unlocked.
    audit.record_login_success(user=user, method=settings.method, role=role)
    return response
