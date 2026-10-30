"""Who is looking: the person and the record viewer nginx hands the page.

nginx asks the auth sidecar's ``/verify-history`` once per request and forwards
its answer as the ``X-Osprey-History-*`` headers below; the page trusts them
because only nginx reaches it (the gate secret, published on loopback only).
The view header names the agent-record viewer (``all``, ``own:<person>`` or
``reader:<name>``) the page reads every answer for. Every link the page renders
carries ``?as=<entry>``, the sidecar's selector, so the person a viewer picked
stays picked as they navigate.
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
ENTRY_HEADER = "X-Osprey-History-Entry"
CHOICES_HEADER = "X-Osprey-History-Choices"

#: The query parameter that selects which sign-in the page is viewed as.
SELECTOR_PARAM = "as"

SCOPE_OWN = "own"
SCOPE_ALL = "all"
SCOPE_READER = "reader"

#: The record viewers, spelled as the agent-record service reads them.
VIEWER_ALL = "all"
VIEWER_OWN_PREFIX = "own:"
VIEWER_READER_PREFIX = "reader:"

#: A person or reader name a viewer may carry: the roster's charset.
_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


@dataclass(frozen=True)
class Viewer:
    """One request's viewer, as the sidecar resolved them."""

    person: str
    scope: str
    record_viewer: str
    entry: str
    choices: tuple[tuple[str, str], ...]
    role: str = ""

    @property
    def own(self) -> bool:
        """Whether this view shows the person's own sessions only."""
        return self.scope == SCOPE_OWN

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


def record_viewer_scope(value: str) -> str:
    """The scope the record viewer *value* reads.

    Raises:
        MissingViewer: *value* is not ``all``, ``own:<person>`` or
            ``reader:<name>`` with a name the roster could hold.
    """
    if value == VIEWER_ALL:
        return SCOPE_ALL
    for prefix, scope in ((VIEWER_OWN_PREFIX, SCOPE_OWN), (VIEWER_READER_PREFIX, SCOPE_READER)):
        if value.startswith(prefix) and _NAME.fullmatch(value[len(prefix) :]):
            return scope
    raise MissingViewer(f"viewer {value!r} is not all, own:<person> or reader:<name>")


def viewer_from_headers(headers: Mapping[str, str]) -> Viewer:
    """The viewer the sidecar's headers name.

    Raises:
        MissingViewer: A required header is absent or empty, the record viewer
            is not one the record reads, or it disagrees with the scope.
    """
    lowered = {key.lower(): value for key, value in headers.items()}

    def header(name: str) -> str:
        return (lowered.get(name.lower()) or "").strip()

    person, scope, record_viewer, entry = (
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
            (VIEW_HEADER, record_viewer),
            (ENTRY_HEADER, entry),
        )
        if not value
    ]
    if missing:
        raise MissingViewer(f"missing {', '.join(missing)}")
    if record_viewer_scope(record_viewer) != scope:
        raise MissingViewer(f"viewer {record_viewer!r} does not read scope {scope!r}")
    choices = _choices(header(CHOICES_HEADER)) or ((entry, person),)
    return Viewer(
        person=person,
        scope=scope,
        record_viewer=record_viewer,
        entry=entry,
        choices=choices,
        role=header(ROLE_HEADER),
    )
