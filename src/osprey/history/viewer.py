"""Who is looking: the person, view user and tables nginx hands the page.

nginx asks the auth sidecar's ``/verify-history`` once per request and forwards
its answer as the ``X-Osprey-History-*`` headers below; the page trusts them
because only nginx reaches it (the gate secret, published on loopback only).
Every link the page renders carries ``?as=<entry>``, the sidecar's selector, so
the person a viewer picked stays picked as they navigate.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlencode

#: The path every page URL is under: nginx strips it before forwarding, and the
#: page writes it back onto every link and asset URL it renders.
ROOT_PATH = "/history"

#: The sidecar's answer, one header each (the same names
#: :mod:`osprey.services.auth_sidecar.routes.history` sets).
PERSON_HEADER = "X-Osprey-History-Person"
SCOPE_HEADER = "X-Osprey-History-Scope"
VIEW_HEADER = "X-Osprey-History-View"
ROLE_HEADER = "X-Osprey-History-Role"
TABLES_HEADER = "X-Osprey-History-Tables"
ENTRY_HEADER = "X-Osprey-History-Entry"
CHOICES_HEADER = "X-Osprey-History-Choices"

#: The query parameter that selects which sign-in the page is viewed as.
SELECTOR_PARAM = "as"

SCOPE_OWN = "own"
SCOPE_ALL = "all"
SCOPE_READER = "reader"

#: A store user name the page may put after ``EXECUTE AS``: the roster's
#: charset plus the underscores the view users are spelled with.
_VIEW_USER = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


@dataclass(frozen=True)
class Viewer:
    """One request's viewer, as the sidecar resolved them."""

    person: str
    scope: str
    view_user: str
    tables: tuple[str, ...]
    entry: str
    choices: tuple[tuple[str, str], ...]
    role: str = ""

    @property
    def own(self) -> bool:
        """Whether this view shows the person's own sessions only."""
        return self.scope == SCOPE_OWN

    def reads(self, table: str) -> bool:
        """Whether the view user may read *table*."""
        return table in self.tables

    def href(self, path: str = "/", **params: object) -> str:
        """A page URL under :data:`ROOT_PATH` that keeps this viewer's selection.

        ``None`` values are left out; ``as`` is always last and always present.
        """
        query = [(key, str(value)) for key, value in params.items() if value is not None]
        query.append((SELECTOR_PARAM, self.entry))
        return f"{ROOT_PATH}{path}?{urlencode(query)}"

    def switch_href(self, entry: str) -> str:
        """The session list viewed as another of this browser's sign-ins."""
        return f"{ROOT_PATH}/?{urlencode([(SELECTOR_PARAM, entry)])}"


class MissingViewer(ValueError):
    """The request carries no complete answer from the sidecar."""


def _choices(raw: str) -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    for item in raw.split(","):
        key, sep, person = item.strip().partition("=")
        if sep and key and person:
            found.append((key, person))
    return tuple(found)


def viewer_from_headers(headers: Mapping[str, str]) -> Viewer:
    """The viewer the sidecar's headers name.

    Raises:
        MissingViewer: A required header is absent or empty, or the view user
            is not a name the page may run a query as.
    """
    lowered = {key.lower(): value for key, value in headers.items()}

    def header(name: str) -> str:
        return (lowered.get(name.lower()) or "").strip()

    person, scope, view_user, entry = (
        header(PERSON_HEADER),
        header(SCOPE_HEADER),
        header(VIEW_HEADER),
        header(ENTRY_HEADER),
    )
    missing = [
        name
        for name, value in (
            (PERSON_HEADER, person),
            (SCOPE_HEADER, scope),
            (VIEW_HEADER, view_user),
            (ENTRY_HEADER, entry),
        )
        if not value
    ]
    if missing:
        raise MissingViewer(f"missing {', '.join(missing)}")
    if scope not in (SCOPE_OWN, SCOPE_ALL, SCOPE_READER):
        raise MissingViewer(f"unknown scope {scope!r}")
    if not _VIEW_USER.fullmatch(view_user):
        raise MissingViewer(f"view user {view_user!r} is not a store user name")
    tables = tuple(t.strip() for t in header(TABLES_HEADER).split(",") if t.strip())
    choices = _choices(header(CHOICES_HEADER)) or ((entry, person),)
    return Viewer(
        person=person,
        scope=scope,
        view_user=view_user,
        tables=tables,
        entry=entry,
        choices=choices,
        role=header(ROLE_HEADER),
    )
