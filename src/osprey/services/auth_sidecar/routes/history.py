"""The history page's ``auth_request`` target: which person, as which view user.

nginx asks ``/verify-history`` once per request under ``/history/`` on the
readers' origin. Where ``/verify`` answers "may this browser reach this card",
this route answers "whose sessions may this browser see, right now": it resolves
**one person** out of the session cookie and maps them to exactly one store view
user and that user's tables. The page runs every query as that view user, so the
answer here is the scope; the database enforces it.

**Every entry is re-validated against the current settings.** A card entry is
asked :func:`~osprey.services.auth_sidecar.routes.verify.check_entry`, the very
question ``/verify`` asks of it; a ``reader:<name>`` entry is asked
:func:`check_reader_entry`. An entry that fails is not a sign-in at all. The
mapping reads ``history: all`` from the person's *current* roster entry and the
*current* roster role carrier, never from the role stored in the entry, so a
person moved to another role is mapped by the new one on the next request after
``osprey up``. The one exception is a role an OIDC claim granted: only the token
can assert it, so it is read from the session and lapses with it.

**The answer.** 401 when no entry is a valid sign-in, or the request selects one
it may not (an unknown or a repeated ``as``); 403 when valid sign-ins exist but
none maps to a view user — a stranger admitted to a card by a ``domain:`` rule,
a sign-in as a shared card without ``history: all``; otherwise 200 with the
``X-Osprey-History-*`` headers below. Like ``/verify`` it never redirects and
never says why; the reason goes to the log at debug level by category.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated
from urllib.parse import parse_qs, urlsplit

from fastapi import APIRouter, Cookie, Depends, Header, Request, Response

from ..app import AuthSettings, get_revocation_store, get_session_codec, get_settings
from ..exceptions import InvalidSessionError
from ..identity_headers import is_header_safe, same_identity
from ..revocation import RevocationStore
from ..roster_env import READER_ENTRY_PREFIX
from ..sessions import SESSION_COOKIE_NAME, SessionCodec, SessionState, UnlockedUser
from .recheck import ROLE_SOURCE_CLAIM, RosterRoles, roster_roles
from .verify import check_entry

logger = logging.getLogger(__name__)

router = APIRouter()

VERIFY_HISTORY_PATH = "/verify-history"
"""The ``auth_request`` target of the readers' ``location /history/``.

Not under the public ``/auth/`` prefix, for the reason
:data:`~osprey.services.auth_sidecar.routes.verify.VERIFY_PATH` is not."""

SELECTOR_PARAM = "as"
"""The page's selection, read from ``X-Original-URI``: ``?as=<entry key>``."""

PERSON_HEADER = "X-Osprey-History-Person"
SCOPE_HEADER = "X-Osprey-History-Scope"
VIEW_HEADER = "X-Osprey-History-View"
ROLE_HEADER = "X-Osprey-History-Role"
TABLES_HEADER = "X-Osprey-History-Tables"
ENTRY_HEADER = "X-Osprey-History-Entry"
CHOICES_HEADER = "X-Osprey-History-Choices"

SCOPE_OWN = "own"
SCOPE_ALL = "all"
SCOPE_READER = "reader"

VIEW_ALL = "view__all"
"""The view user that sees every person's sessions."""

VIEW_USER_PREFIX = "view_"
"""A roster person's own view user is ``view_<person>``."""

VIEW_READER_PREFIX = "view__reader_"
"""A reader's view user is ``view__reader_<name>``."""

VIEW_TABLES = ("otel_logs", "otel_traces", "audit_ledger")
"""The tables a ``view_<person>`` and ``view__all`` read: every store table."""

# Default-selection precedence among entries sharing the greatest expiry. One
# card-less login mints all its admissions with one expiry, so ties are the
# normal case; `with_user` replaces a re-login's entries in place, so insertion
# order says nothing about which sign-in is newest.
TIER_READER = 0
TIER_OWN_CARD = 1
TIER_SHARED_CARD = 2
TIER_OPENER = 3
TIER_RULE_ADMITTED = 4


@dataclass(frozen=True)
class HistoryView:
    """One valid sign-in, mapped to the view user it reads as.

    Attributes:
        key: The session entry's key — a roster card or ``reader:<name>``.
        person: Who the page is viewing as: a roster person, a reader name, or a
            shared card's name under ``history: all``.
        view_user: The store account the page impersonates.
        scope: ``own``, ``all`` or ``reader``.
        tables: The tables that view user may read.
        role: The roster or claim role behind the scope, or ``""``.
        tier: Default-selection precedence; lower wins a tie.
        expires_at: The entry's expiry; the newest valid sign-in is the default.
    """

    key: str
    person: str
    view_user: str
    scope: str
    tables: tuple[str, ...]
    role: str
    tier: int
    expires_at: float


@dataclass(frozen=True)
class _Row:
    """A mapping-table row before it is attached to an entry."""

    person: str
    view_user: str
    scope: str
    tables: tuple[str, ...]
    role: str = ""


def check_reader_entry(entry: UnlockedUser, settings: AuthSettings) -> str | None:
    """Whether a ``reader:<name>`` entry still signs a reader in.

    A reader entry is minted only from an IdP identity matching a configured
    reader's subject, so it is re-asked exactly that: the name is a configured
    reader now, **a subject is configured for it now**, and the entry's stored
    identity is still that subject. A reader whose subject was removed is
    therefore refused, not skipped. Expiry and revocation are the session's,
    checked by the caller.

    Args:
        entry: The entry, keyed ``reader:<name>``.
        settings: The deployment's frozen settings.

    Returns:
        ``None`` when the entry is valid; otherwise a fixed category string for
        the log.
    """
    name = entry.username[len(READER_ENTRY_PREFIX) :]
    if not name:
        return "reader entry names no reader"
    if entry.opener or entry.admitted_identity:
        return "reader entry carries a card's admission story"
    if settings.method != "oidc":
        return "reader entries are minted under oidc only"
    if not settings.knows_reader(name):
        return "the reader is no longer configured"
    configured = settings.reader_subject(name)
    if not configured:
        return "the reader has no configured identity"
    if not entry.oidc_subject or not same_identity(
        configured, entry.oidc_subject, claim=settings.oidc_claim
    ):
        return "the reader's configured identity has changed"
    return None


def _shared_card_row(settings: AuthSettings, roles: RosterRoles, card: str) -> _Row | None:
    """A sign-in AS a shared card: ``view__all`` under ``history: all``, else nothing.

    Read from the card's own roster entry and its roster-bound role. Never a
    ``view_<card>``: a shared card is not a person.
    """
    role = roles.role_for(card)
    if settings.user_history_all(card):
        return _Row(card, VIEW_ALL, SCOPE_ALL, VIEW_TABLES, role)
    if settings.role_history_all(role):
        return _Row(card, VIEW_ALL, SCOPE_ALL, VIEW_TABLES, role)
    return None


def _person_row(
    settings: AuthSettings, roles: RosterRoles, person: str, *, claim_role: str = ""
) -> _Row | None:
    """A roster person's own row, from their current roster entry and role.

    A reader of that name takes precedence; then ``history: all`` from the
    person's roster entry, their current roster role, or a claim-granted role;
    else their own view. A shared card in the person's place answers as a
    sign-in as that card.
    """
    if not settings.knows_user(person):
        return None
    if settings.shared(person):
        return _shared_card_row(settings, roles, person)
    role = roles.role_for(person)
    if settings.knows_reader(person):
        return _Row(
            person,
            f"{VIEW_READER_PREFIX}{person}",
            SCOPE_READER,
            settings.tables_for_reader(person),
            role,
        )
    if settings.user_history_all(person):
        return _Row(person, VIEW_ALL, SCOPE_ALL, VIEW_TABLES, role)
    if settings.role_history_all(role):
        return _Row(person, VIEW_ALL, SCOPE_ALL, VIEW_TABLES, role)
    if settings.role_history_all(claim_role):
        return _Row(person, VIEW_ALL, SCOPE_ALL, VIEW_TABLES, claim_role)
    return _Row(person, f"{VIEW_USER_PREFIX}{person}", SCOPE_OWN, VIEW_TABLES, role or claim_role)


def _rule_admitted_person(settings: AuthSettings, identity: str) -> str | None:
    """The one roster person a rule-admitted identity belongs to, or ``None``.

    Every roster mapping is compared (:meth:`AuthSettings.subject_matches`); a
    shared card is not a person, so its mapping does not count. Zero or several
    persons is ``None`` — a stranger stays a stranger.
    """
    persons = [name for name, _ in settings.subject_matches(identity) if not settings.shared(name)]
    return persons[0] if len(persons) == 1 else None


def _map_card_entry(
    settings: AuthSettings, roles: RosterRoles, entry: UnlockedUser
) -> tuple[_Row | None, int]:
    """Map one valid card entry through the mapping table."""
    card = entry.username
    if entry.opener:
        # A shared card opened by roster person U: U's own row, from U's
        # current entry and role — never the entry's role, which is the card's.
        return _person_row(settings, roles, entry.opener), TIER_OPENER
    if entry.admitted_identity:
        person = _rule_admitted_person(settings, entry.admitted_identity)
        if person is None:
            return None, TIER_RULE_ADMITTED
        return _person_row(settings, roles, person), TIER_RULE_ADMITTED
    if settings.shared(card):
        return _shared_card_row(settings, roles, card), TIER_SHARED_CARD
    claim_role = (
        entry.role if settings.method == "oidc" and entry.role_source == ROLE_SOURCE_CLAIM else ""
    )
    return _person_row(settings, roles, card, claim_role=claim_role), TIER_OWN_CARD


@dataclass(frozen=True)
class Mappable:
    """What a session cookie offers the history page.

    Attributes:
        views: The valid sign-ins that map to a view user, best default first.
        valid: How many entries were valid sign-ins at all, mapped or not.
    """

    views: tuple[HistoryView, ...]
    valid: int


def mappable_entries(state: SessionState, settings: AuthSettings, roles: RosterRoles) -> Mappable:
    """Every valid sign-in in ``state`` that maps to a view user.

    Card entries are re-validated with ``/verify``'s own
    :func:`~osprey.services.auth_sidecar.routes.verify.check_entry`, reader
    entries with :func:`check_reader_entry`. An entry whose key or answer
    cannot be carried in a header is left out.

    Args:
        state: The decoded session; expired entries are already dropped.
        settings: The deployment's frozen settings.
        roles: The current roster role table.

    Returns:
        The mapped views ordered as the default is chosen — greatest expiry
        first, then precedence, then key — and how many entries were valid.
    """
    views: list[HistoryView] = []
    valid = 0
    for entry in state.users:
        key = entry.username
        if key.startswith(READER_ENTRY_PREFIX):
            reason = check_reader_entry(entry, settings)
            if reason is not None:
                logger.debug("history: entry %r refused: %s", key, reason)
                continue
            valid += 1
            name = key[len(READER_ENTRY_PREFIX) :]
            row: _Row | None = _Row(
                name,
                f"{VIEW_READER_PREFIX}{name}",
                SCOPE_READER,
                settings.tables_for_reader(name),
            )
            tier = TIER_READER
        else:
            if not settings.knows_user(key):
                logger.debug("history: entry %r refused: not on the roster", key)
                continue
            reason = check_entry(key, entry, settings)
            if reason is not None:
                logger.debug("history: entry %r refused: %s", key, reason)
                continue
            valid += 1
            row, tier = _map_card_entry(settings, roles, entry)
        if row is None:
            logger.debug("history: entry %r maps to no view user", key)
            continue
        carried = (key, row.person, row.view_user, row.role, *row.tables)
        if not all(is_header_safe(value) for value in carried if value):
            logger.debug("history: entry %r cannot be carried in a header", key)
            continue
        views.append(
            HistoryView(
                key=key,
                person=row.person,
                view_user=row.view_user,
                scope=row.scope,
                tables=row.tables,
                role=row.role,
                tier=tier,
                expires_at=entry.expires_at,
            )
        )
    views.sort(key=lambda view: (-view.expires_at, view.tier, view.key))
    return Mappable(views=tuple(views), valid=valid)


def _selector(original_uri: str | None) -> tuple[bool, str | None]:
    """The ``as`` selection in the original request URI.

    Returns:
        ``(ok, key)``: ``key`` is ``None`` when nothing is selected; ``ok`` is
        ``False`` when ``as`` is repeated, which is refused rather than resolved.
    """
    if not original_uri:
        return True, None
    query = urlsplit(original_uri).query
    values = parse_qs(query, keep_blank_values=True).get(SELECTOR_PARAM, [])
    if len(values) > 1:
        return False, None
    return True, values[0] if values else None


def _deny(status: int, reason: str) -> Response:
    logger.debug("verify-history answered %d: %s", status, reason)
    return Response(status_code=status)


@router.get(VERIFY_HISTORY_PATH)
async def verify_history(
    request: Request,
    settings: Annotated[AuthSettings, Depends(get_settings)],
    codec: Annotated[SessionCodec, Depends(get_session_codec)],
    revocations: Annotated[RevocationStore, Depends(get_revocation_store)],
    original_uri: Annotated[str | None, Header(alias="X-Original-URI")] = None,
    session_cookie: Annotated[str | None, Cookie(alias=SESSION_COOKIE_NAME)] = None,
) -> Response:
    """Resolve one person and their view user for a history-page request.

    Args:
        request: The subrequest, for the app's roster role table.
        settings: The deployment's frozen settings.
        codec: The app's session codec and clock.
        revocations: The app's revocation store.
        original_uri: The page request's URI, which carries ``?as=<key>``.
        session_cookie: The signed session cookie, if the browser sent one.

    Returns:
        200 with the ``X-Osprey-History-*`` headers; 401 or 403 bare.
    """
    ok, selected = _selector(original_uri)
    if not ok:
        return _deny(401, "the selector is repeated")
    if not session_cookie:
        return _deny(401, "no session cookie")
    try:
        state = codec.decode(session_cookie)
    except InvalidSessionError:
        return _deny(401, "session cookie rejected")
    if revocations.is_revoked(state.session_id):
        return _deny(401, "session was revoked by a logout")

    mappable = mappable_entries(state, settings, roster_roles(request))
    if not mappable.valid:
        return _deny(401, "no valid sign-in in this session")
    if not mappable.views:
        return _deny(403, "no sign-in in this session maps to a view user")

    if selected is None:
        chosen = mappable.views[0]
    else:
        matches = [view for view in mappable.views if view.key == selected]
        if not matches:
            return _deny(401, "the selector names no mappable sign-in")
        chosen = matches[0]

    headers = {
        PERSON_HEADER: chosen.person,
        SCOPE_HEADER: chosen.scope,
        VIEW_HEADER: chosen.view_user,
        TABLES_HEADER: ",".join(chosen.tables),
        ENTRY_HEADER: chosen.key,
        CHOICES_HEADER: ",".join(f"{view.key}={view.person}" for view in mappable.views),
    }
    if chosen.role:
        headers[ROLE_HEADER] = chosen.role
    return Response(status_code=200, headers=headers)
