"""The sidecar's audit seam: one function per login decision it must record.

A login is a safety decision like any other Osprey records, so it belongs in the
unified ledger (:mod:`osprey.audit.envelope`) rather than in a service log line
an operator has to grep for. This module is the only place in the sidecar that
builds an envelope, so every category the login path admits or refuses under is
shaped the same way and joins the rest of the ledger on the same columns.

**The seam is deliberately one call wide.** Every route reaches the ledger
through :func:`record_login_success` or :func:`record_login_refusal`, which are
the only two builders of an envelope in this service, and both file through
:func:`write_envelope`. Nothing else in the sidecar knows where a record lands.

**Where a record lands is env-only.** :data:`AUDIT_DIR_ENV` names the one
directory this service may write in — the subdirectory compose binds for it —
and it is read per call, never resolved from a project root. The sidecar image
is one uvicorn app under ``/app``: there is no ``osprey.yml`` to anchor on, and
:func:`osprey.audit.writer.audit_dir`'s resolver would either raise here or
point at a directory the container does not bind. An unset or blank variable
degrades to the log line below — a sidecar rendered before the audit mount
existed still says what it decided, it simply cannot store it.

**The record is the user's, the directory is the service's.** The sidecar is
the ``actor`` (it decided, under its own container identity) and the roster user
is the ``subject`` (they were acted on). That asymmetry is the whole reason
these records file under ``var/audit/sidecar/`` rather than under the user's
own subdirectory: a user's container binds their own directory read-write, so a
login history filed there would be theirs to rewrite. The sidecar's directory is
bound by nobody else — and, the service running as root with no entrypoint and
no ``user:``, its records are root-owned on the host by design.

**What bounds the file's growth.** Every refusal an unauthenticated caller can
reach cheaply is recorded through a window: the first refusal in each window is
filed and the rest are folded into it, so the append rate is bounded rather than
set by the caller. On the password path that window is the login throttle,
which the credential attempt had to clear anyway. On the OIDC path — where the
cheap refusals are the ones reachable *before* the token exchange, and were
never evaluated at all — it is a second, separate instance of the same class
(:func:`~osprey.services.auth_sidecar.app.get_audit_throttle`), because growing
the login window on an unevaluated attempt is how an unauthenticated caller
would delay a named operator's real login. A folded refusal is still countable:
the next record filed for that user names how many were folded into it (see
:data:`~osprey.services.auth_sidecar.routes.oidc.FOLDED_DETAIL_KEY`). The
post-exchange OIDC categories are unbounded on purpose — reaching one costs a
full IdP round trip, so they are not free to generate, and they are the records
that describe a login that actually authenticated. The distinction matters
because this file is root-owned in a directory nothing else in the deployment
binds: nobody inside it can rotate or truncate what an unbounded append filled.

**An audit failure never costs the decision.** Both record functions swallow
everything the write can raise: a refusal that was audited and a refusal that
could not be audited are both refusals, and a sidecar that 500s because its
ledger is unwritable has turned an audit gap into an outage. The reverse —
recording a decision that did not happen — is what the callers prevent by
calling these only from the branch that actually decided.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from osprey.audit import writer
from osprey.audit.envelope import (
    DECISION_ALLOWED,
    DECISION_REFUSED,
    POSTURE_SOURCE_APP,
    AuditEnvelope,
)
from osprey.utils.identity import acting_identity

logger = logging.getLogger(__name__)

__all__ = [
    "AUDIT_DIR_ENV",
    "LOGIN_REASONS",
    "REASON_AMBIGUOUS_ROLE_CLAIM",
    "REASON_BAD_CREDENTIAL",
    "REASON_IDENTITY_MISMATCH",
    "REASON_METHOD_MISMATCH",
    "REASON_MISSING_ROLE_CLAIM",
    "REASON_NON_ASCII_SUBJECT",
    "REASON_NO_ASSERTED_IDENTITY",
    "REASON_OIDC_LOGIN",
    "REASON_PASSWORD_LOGIN",
    "REASON_ROLE_MISMATCH",
    "REASON_UNMAPPED_ROLE_CLAIM",
    "REASON_UNMAPPED_USER",
    "REASON_UNSAFE_ROLE",
    "REASON_UNSUPPORTED_METHOD",
    "REASON_UNVALIDATED_TOKEN",
    "SIDECAR_POSTURE",
    "SURFACE",
    "ledger_path",
    "record_login_refusal",
    "record_login_success",
    "write_envelope",
]

SURFACE = "auth_sidecar"
"""The surface every record from this service names.

Fixed rather than derived: the sidecar decides exactly one kind of thing, and a
record that named the route instead would split one question ("who tried to log
in, and what happened") across as many surfaces as the login flow has legs.
"""

AUDIT_DIR_ENV = "OSPREY_AUTH_AUDIT_DIR"
"""Names the directory this service files its records in.

Auth-specific rather than the generic ``OSPREY_AUDIT_DIR``: that one is the
variable the project image's entrypoint iterates to fix up group ownership, and
the sidecar image has no entrypoint and no root phase to read it. Rendered by
``deployment/web_terminals/render.py`` from the same derivation that names the
bind's host side, so the path this module writes to and the path the host
provisions cannot diverge.
"""

SIDECAR_POSTURE = "sandbox"
"""The posture this service stamps on its own records.

The sidecar governs *entry*, not writes: it holds no posture store, spawns no
session, and touches no facility state. ``sandbox`` is the honest reading of
"this surface writes nothing", and :data:`~osprey.audit.envelope.POSTURE_SOURCE_APP`
alongside it says the value is the app's own stamp rather than a session's true
posture — so nobody reads it as evidence about what the operator was allowed to
do once inside their terminal.
"""

REASON_NON_ASCII_SUBJECT = "non_ascii_subject"
"""The mapped identity cannot be carried in an identity header.

Its own category, distinct from "no identity is mapped" and from "the asserted
identity is somebody else's": nothing about the login was wrong except that this
deployment mapped a claim spelling the boundary cannot carry, and the operator
reading the ledger needs to be pointed at their config rather than at their IdP.
"""

REASON_BAD_CREDENTIAL = "bad_credential"
"""A password attempt was evaluated and did not unlock the user.

**One category on purpose.** A wrong password, a roster user with no provisioned
credential, and a name that was never on the roster all arrive here — the same
anti-lookup discipline the login page keeps, extended to the ledger, so a reader
of the file cannot use it to enumerate accounts either. What varies between
those cases is nothing the record could say without saying who exists.
"""

REASON_UNMAPPED_USER = "unmapped_user"
"""The clicked roster user has no mapped IdP identity at all."""

REASON_NO_ASSERTED_IDENTITY = "no_asserted_identity"
"""The validated ID token carried no usable value in the identity claim."""

REASON_IDENTITY_MISMATCH = "identity_mismatch"
"""The asserted identity belongs to a different roster user, or to none.

Its own category rather than a shade of :data:`REASON_UNMAPPED_USER`: this is a
login that authenticated successfully and was still refused, which is the one
denial an operator should be able to count separately — a run of them is either
a misconfigured subject mapping or somebody clicking a card that is not theirs.
"""

REASON_UNVALIDATED_TOKEN = "unvalidated_token"
"""The token response could not be validated as an OIDC one.

Reached when the provider's answer carried no ID token where one was required:
the claims that would have decided both the identity and the role would then be
unsigned. The one refusal in this service driven by a hostile or substituted
token endpoint rather than by configuration, which is exactly why it is filed
rather than merely logged."""

REASON_MISSING_ROLE_CLAIM = "missing_role_claim"
"""The group claim did not arrive, or arrived in a shape carrying no value.

Entra's group overage lands here: past its limit the provider drops ``groups``
from the ID token and leaves ``_claim_names``/``_claim_sources`` pointing at
Microsoft Graph. The remedies are the IdP's (emit only assigned or security
groups, or map app roles instead), which is why the diagnostic names the claims
that *did* arrive."""

REASON_UNMAPPED_ROLE_CLAIM = "unmapped_role_claim"
"""No value in the group claim is mapped to a role by this deployment."""

REASON_AMBIGUOUS_ROLE_CLAIM = "ambiguous_role_claim"
"""The group claim maps to more than one distinct role.

Refused rather than resolved. The alternative — take the first — would make the
privilege granted depend on the order the provider listed the groups in, or on
the order the roles were declared in YAML, which is exactly how an operator ends
up with a privilege nobody decided to give them."""

REASON_ROLE_MISMATCH = "role_mismatch"
"""The claim's role is not the role this user's terminal was rendered into.

The cross-check the roster half of the federated posture exists for. A roster
entry's ``role:`` decides, at build time, which persona this user's container
runs — so a login whose validated ID token maps to a *different* role is a login
that would land the person in a terminal their asserted privilege does not
describe. Refused rather than reconciled: taking the claim would hand them a
session that names one role while sitting in another's container, and taking the
roster's would grant a privilege the login never proved.

Distinct from :data:`REASON_UNMAPPED_ROLE_CLAIM` (the claim maps to nothing at
all) and from :data:`REASON_AMBIGUOUS_ROLE_CLAIM` (it maps to several): here the
provider asserted exactly one role, and the disagreement is with the deployment
rather than inside the token. A run of these is a roster and an IdP that have
drifted apart — someone moved between groups without their roster entry
following — which is the one refusal in this set an operator fixes in *both*
places.
"""

REASON_UNSAFE_ROLE = "unsafe_role"
"""The resolved role cannot be carried in an identity header.

A configuration fault of the same family as :data:`REASON_NON_ASCII_SUBJECT`:
render-time lint holds role names to ``USERNAME_CHARSET_RE`` for this reason, so
reaching here means the table the sidecar was handed did not come through it.
Refused at the re-check rather than left to
:meth:`~osprey.services.auth_sidecar.sessions.SessionState.with_user`, whose
``ValueError`` would surface as a 500 on what is a denial."""

REASON_UNSUPPORTED_METHOD = "unsupported_method"
"""The deployment's auth method is not one this service can mint a session for.

The identity matrix is a closed table, and a method outside it — ``none``, a
typo, a method a future release adds — resolves to no row. Refusing is the only
reading that cannot grant something nobody decided on."""

REASON_METHOD_MISMATCH = "method_mismatch"
"""The facts offered for a login do not belong to its method.

A password login carrying a provider subject, or an OIDC login carrying none:
either way two flows have been confused, and the matrix has no row that says
what such a login is worth."""

REASON_PASSWORD_LOGIN = "password_login"
"""A password login succeeded — how the subject proved who they were."""

REASON_OIDC_LOGIN = "oidc_login"
"""An OIDC login succeeded — how the subject proved who they were."""


_SUCCESS_REASONS: dict[str, str] = {
    "password": REASON_PASSWORD_LOGIN,
    "oidc": REASON_OIDC_LOGIN,
}
"""The reason each supported method's success is recorded under.

Keyed by ``AuthSettings.method`` spelling so the two stay one concept. A method
outside the closed set falls back to :data:`_GENERIC_LOGIN` rather than
inventing a category from a caller-supplied string.
"""

_GENERIC_LOGIN = "login"

LOGIN_REASONS: frozenset[str] = frozenset(
    {
        REASON_AMBIGUOUS_ROLE_CLAIM,
        REASON_BAD_CREDENTIAL,
        REASON_IDENTITY_MISMATCH,
        REASON_METHOD_MISMATCH,
        REASON_MISSING_ROLE_CLAIM,
        REASON_NON_ASCII_SUBJECT,
        REASON_NO_ASSERTED_IDENTITY,
        REASON_OIDC_LOGIN,
        REASON_PASSWORD_LOGIN,
        REASON_ROLE_MISMATCH,
        REASON_UNMAPPED_ROLE_CLAIM,
        REASON_UNMAPPED_USER,
        REASON_UNSAFE_ROLE,
        REASON_UNSUPPORTED_METHOD,
        REASON_UNVALIDATED_TOKEN,
        _GENERIC_LOGIN,
    }
)
"""Every category this service may file a login record under.

A closed set, and the reason the categories live *here* rather than beside the
route that raises each one: an operator reading the ledger needs one place that
enumerates what they might find in ``reason``, and a new category added at a
call site without joining this set is a category nothing documents. Not
enforced at write time — :func:`record_login_refusal` takes the caller's word,
because a record that names an unlisted category is still better than a lost
one — but pinned by a test, which is the layer that can fail loudly without
costing a decision."""


def ledger_path() -> Path | None:
    """The file this service's records are appended to, or ``None``.

    ``None`` means :data:`AUDIT_DIR_ENV` names nothing usable, which is the
    documented degrade rather than an error. The file is named for the surface,
    inside the directory the variable names — the same
    ``<identity>/<surface>.jsonl`` shape every other emitter files under, so the
    sidecar's ledger reads next to the rest of them without a special case.

    Read per call, never cached: the variable is a property of the container's
    environment, and a value captured at import would be whatever the first
    importer happened to see.

    **A relative value is refused like a blank one.** Resolved against the
    process's working directory it would name ``/app/<something>`` inside the
    image — a path the host binds nothing at — and the records would accumulate
    in the container's writable layer and vanish with it, while this function
    kept returning a path that says they were durably stored. The documented
    degrade (the log line in :func:`write_envelope`) is the honest answer to a
    value this service cannot write anything durable under.

    **The stem is the surface literal, not a routed name.**
    :func:`osprey.audit.writer.ledger_name` consults
    :func:`~osprey.audit.writer.writer_context` first, so an
    ``OSPREY_AUDIT_WRITER`` value that happened to be present in this
    container's environment would silently rename this ledger. The sidecar has
    no root maintenance phase for that marker to describe — one uvicorn app, no
    entrypoint — so borrowing a helper that routes on it can only misfire here.
    :data:`~osprey.audit.writer.LEDGER_SUFFIX` is still the writer's, because
    that one *is* shared vocabulary.
    """
    raw = os.environ.get(AUDIT_DIR_ENV)
    if not isinstance(raw, str) or not raw.strip():
        return None
    directory = Path(raw.strip())
    if not directory.is_absolute():
        logger.warning(
            "%s is not an absolute path; this service files no records until it names the "
            "directory compose binds for it",
            AUDIT_DIR_ENV,
        )
        return None
    return directory / f"{SURFACE}{writer.LEDGER_SUFFIX}"


def record_login_success(*, user: str, method: str, role: str = "") -> None:
    """Record that ``user`` logged in, and how.

    Recorded for the same reason refusals are: "who is in this deployment, and
    since when" is the first question asked of an audit trail, and a ledger that
    holds only failures cannot answer it.

    Args:
        user: The roster user who logged in — the record's *subject*.
        method: The deployment's auth method (``password`` or ``oidc``), which
            names the category the success is recorded under.
        role: The role the session was minted with, where the deployment binds
            one. Empty is the deny-safe value and is recorded as no role at all,
            never as a role named ``""``.
    """
    _record(
        user=user,
        decision=DECISION_ALLOWED,
        reason=_SUCCESS_REASONS.get(method, _GENERIC_LOGIN),
        role=role,
    )


def record_login_refusal(
    *,
    user: str,
    reason: str,
    detail: str | None = None,
    role: str = "",
) -> None:
    """Record that a login for ``user`` was refused, and why.

    Args:
        user: The roster user the login was for. The record's *subject* — what
            was acted on — while the *actor* is the service identity that
            decided, which is what the ledger files the record under.
        reason: A fixed category string such as
            :data:`REASON_NON_ASCII_SUBJECT`. Never an interpolated claim value,
            token, or credential: the ledger carries identifiers, and every
            caller already has a config it can read the offending value from.
        detail: Optional supplementary context — identifiers and config keys
            only, on the same terms as ``reason``.
        role: The role the refused login would have held, where one is known.
    """
    _record(user=user, decision=DECISION_REFUSED, reason=reason, detail=detail, role=role)


def _record(
    *,
    user: str,
    decision: str,
    reason: str,
    detail: str | None = None,
    role: str = "",
) -> None:
    """Build one envelope and file it, swallowing everything either can raise."""
    try:
        envelope = AuditEnvelope(
            surface=SURFACE,
            actor=acting_identity(),
            posture=SIDECAR_POSTURE,
            posture_source=POSTURE_SOURCE_APP,
            # No posture-store key exists: the browser being decided about has
            # no spawned session, and inventing one would join this record to a
            # session that never existed.
            session=None,
            subject=user,
            decision=decision,
            reason=reason,
            detail=detail,
            role=role or None,
        )
    except ValueError:
        # A malformed envelope is a programming error in the caller, not a
        # reason to fail the decision it describes.
        logger.warning("could not build an audit record for a login (%s)", reason)
        return

    try:
        write_envelope(envelope)
    except Exception:  # noqa: BLE001 - the audit trail degrades; the decision does not.
        logger.warning("could not file an audit record for a login (%s)", reason)
        # The rung that keeps the ladder monotone: a zone that raised must not
        # be a worse degrade than a zone that was never configured. See
        # :func:`_log_unfiled`.
        _log_unfiled(envelope)


def _log_unfiled(envelope: AuditEnvelope) -> None:
    """Put a record that could not be stored on the log instead.

    The bottom rung of this module's degrade ladder, and the reason it is one
    function rather than a line repeated at each failure: *file, else log* has
    to hold for every way the write can fail, not only for the one that is
    easiest to reach. The deployment failure that is actually likely — a bind
    that mounted read-only, a full disk, a root-owned file this process cannot
    open — is the one that would otherwise erase the record entirely, while the
    merely cosmetic failure (an unset variable on a laptop) kept it.

    The form is readable on purpose: every field of an envelope is an
    identifier or a category by construction, so the whole record can go on the
    line without any of it being a credential or a claim value.
    """
    try:
        logger.info("audit (unfiled): %s", json.dumps(envelope.to_dict(), sort_keys=True))
    except Exception:  # noqa: BLE001 - the last rung cannot itself cost the decision.
        logger.warning("could not log an unfiled audit record for a login")


def write_envelope(envelope: AuditEnvelope) -> Path | None:
    """File one record in the audit ledger; returns where it landed, or ``None``.

    The single seam every record in this service passes through. The line
    shaping and the append are the shared writer's, reached through
    :func:`osprey.audit.writer.append_envelope`: the byte budget that makes
    concurrent appends non-interleaving and the one ``O_APPEND`` write that
    makes the ledger append-only are not re-implemented one service over. Only
    the directory is this module's own — the writer's other entry points route
    through :func:`~osprey.audit.writer.audit_dir`, which resolves a project
    root the sidecar image does not have.

    Args:
        envelope: The already-validated, already-bounded record.

    Returns:
        The ledger path when the record was durably stored, otherwise ``None`` —
        an unset variable, an unwritable zone, a torn write. The caller does not
        branch on it (the decision is enforced by the caller, not by this record
        landing), but a test can tell "wrote" from "degraded quietly".
        *Every* ``None`` has put the record on the log first, so the degrade is
        the same one whichever rung gave way.
    """
    path = ledger_path()
    if path is None:
        _log_unfiled(envelope)
        return None
    if writer.append_envelope(path, envelope):
        return path
    # A short or refused append: the writer has warned about the *path*, which
    # does not say what was lost. This says what was lost.
    _log_unfiled(envelope)
    return None
