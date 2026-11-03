"""The history page: the across-users view of the agent record.

Served on the history page's origin under ``/history/``. nginx asks the auth sidecar
who the browser is, strips the ``/history`` prefix and forwards the request
here with the sidecar's ``X-Osprey-History-*`` answer and this service's gate
secret. The gate is :class:`~osprey.interfaces.common_middleware.WebAuthMiddleware`:
the gate secret arrives through the roster-secret seam, so a request that did
not come through nginx is refused before any route runs.

Every page is one or two reads of the agent record through
:class:`~osprey.agent_record.client.RecordClient`, with the record's
``history`` login and the viewer the sidecar resolved (``all``,
``own:<person>`` or ``reader:<name>``): the record decides what a person
sees, and the page never widens a read. An answer the record could not give is
rendered as "record unavailable", and an answer that may be missing entries
says so; neither is ever shown as an empty, complete page.

The app receives bare paths (nginx strips ``/history``) and writes
:data:`~osprey.history.viewer.ROOT_PATH` back onto every link and asset URL it
renders. It sets no ASGI ``root_path``: Starlette's static mounts would then
expect the prefix on the bare paths nginx forwards and answer every asset 404.
"""

from __future__ import annotations

import datetime
import json
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

from osprey.agent_record.client import ClientAnswer, RecordClient
from osprey.history.viewer import ROOT_PATH, MissingViewer, Viewer, viewer_from_headers

#: How far back the session list's "older" control may step.
MAX_DAYS_ENV = "OSPREY_HISTORY_MAX_DAYS"

#: The days the session list shows before "older" is pressed.
SESSION_WINDOW_DAYS = 7
#: The most days the "older" control steps back when nothing configures it.
DEFAULT_MAX_DAYS = 30
#: The characters of a body shown inline; a longer one links to its full entry.
PREVIEW_CHARS = 2000

#: Builds the record client one request reads with, from its record viewer.
ClientFor = Callable[[str], RecordClient]

_HERE = Path(__file__).parent
_SESSION_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_TEXT_FIELDS = ("text", "name", "decision", "status", "summary", "message")


def _templates() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_HERE / "templates")),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals["root"] = ROOT_PATH
    env.globals["preview_chars"] = PREVIEW_CHARS
    env.filters["clock"] = clock
    env.filters["full_text"] = full_text
    return env


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def clock(value: Any) -> str:
    """An ISO-8601 time as ``YYYY-MM-DD HH:MM:SS`` UTC; anything else as given."""
    if not isinstance(value, str) or not value:
        return ""
    try:
        when = datetime.datetime.fromisoformat(value)
    except ValueError:
        return value
    if when.tzinfo is not None:
        when = when.astimezone(datetime.UTC)
    return when.strftime("%Y-%m-%d %H:%M:%S")


def full_text(body: Any) -> str:
    """An entry body as the page shows it: its text field, else its JSON."""
    if body is None:
        return ""
    if isinstance(body, Mapping):
        for name in _TEXT_FIELDS:
            value = body.get(name)
            if isinstance(value, str) and value:
                return value
    return json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def _default_client_for(env: Mapping[str, str]) -> ClientFor:
    def client_for(record_viewer: str) -> RecordClient:
        return RecordClient.from_env(env, viewer=record_viewer)

    return client_for


def create_app(
    *,
    env: Mapping[str, str] | None = None,
    max_days: int | None = None,
    client_for: ClientFor | None = None,
) -> FastAPI:
    """Build the history page.

    Args:
        env: Where the record client reads ``OSPREY_AGENT_RECORD_URL`` and the
            ``history`` login; the process environment when ``None``.
        max_days: How far back "older" may step; :data:`MAX_DAYS_ENV`, else
            :data:`DEFAULT_MAX_DAYS`, when ``None``.
        client_for: Builds the client a request reads with from its record
            viewer (tests pass a fake); :meth:`RecordClient.from_env` when
            ``None``.
    """
    from osprey.interfaces._app_setup import configure_interface_app

    source = os.environ if env is None else env
    app = FastAPI(title="OSPREY history")
    if max_days is None:
        raw_days = source.get(MAX_DAYS_ENV, "")
        max_days = int(raw_days) if raw_days.isdigit() else DEFAULT_MAX_DAYS
    app.state.max_days = max(1, max_days)
    app.state.client_for = client_for or _default_client_for(source)
    templates = _templates()

    def page(name: str, status: int = 200, **context: Any) -> HTMLResponse:
        return HTMLResponse(templates.get_template(name).render(**context), status_code=status)

    def unavailable(viewer: Viewer, answer: ClientAnswer) -> HTMLResponse:
        return page("refused.html", status=503, viewer=viewer, reason=answer.unavailable)

    def viewer_of(request: Request) -> Viewer | HTMLResponse:
        try:
            return viewer_from_headers(request.headers)
        except MissingViewer:
            return page("no_scope.html", status=403, viewer=None)

    def client(viewer: Viewer) -> RecordClient:
        record_client: RecordClient = app.state.client_for(viewer.record_viewer)
        return record_client

    @app.get("/health", include_in_schema=False)
    def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/", response_class=HTMLResponse)
    def session_list(request: Request, older: int = 0, person: str | None = None) -> HTMLResponse:
        viewer = viewer_of(request)
        if isinstance(viewer, HTMLResponse):
            return viewer
        days = min(SESSION_WINDOW_DAYS + max(0, older), app.state.max_days)
        older = days - SESSION_WINDOW_DAYS if days > SESSION_WINDOW_DAYS else 0
        since = _now() - datetime.timedelta(days=days)
        record = client(viewer)
        sessions = record.list_sessions(since=since)
        if not sessions.available:
            return unavailable(viewer, sessions)
        users = record.session_users(since=since)
        if not users.available:
            return unavailable(viewer, users)
        rows = [row for row in sessions.data or [] if isinstance(row, Mapping)]
        if person:
            rows = [row for row in rows if person in (row.get("persons") or [])]
        return page(
            "sessions.html",
            viewer=viewer,
            rows=rows,
            users=[u for u in users.data or [] if isinstance(u, Mapping)],
            person=person,
            days=days,
            complete=sessions.complete and users.complete,
            more=sessions.more,
            older_href=(
                viewer.href("/", older=older + 1, person=person)
                if days < app.state.max_days
                else None
            ),
            newer_href=viewer.href("/", older=older - 1 or None, person=person) if older else None,
        )

    @app.get("/session/{session_id}", response_class=HTMLResponse)
    def session_page(request: Request, session_id: str, after: str | None = None) -> HTMLResponse:
        viewer = viewer_of(request)
        if isinstance(viewer, HTMLResponse):
            return viewer
        if not _SESSION_ID.fullmatch(session_id):
            return page("not_found.html", status=404, viewer=viewer, session=session_id)
        answer = client(viewer).timeline(session_id, after=after)
        if not answer.available:
            return unavailable(viewer, answer)
        nodes = [node for node in answer.data or [] if isinstance(node, Mapping)]
        if not nodes and after is None and answer.complete:
            return page("not_found.html", status=404, viewer=viewer, session=session_id)
        return page(
            "session.html",
            viewer=viewer,
            session=session_id,
            nodes=nodes,
            complete=answer.complete,
            next_href=(
                viewer.href(f"/session/{session_id}", after=answer.next) if answer.next else None
            ),
            newer_href=viewer.href(f"/session/{session_id}") if answer.newer else None,
        )

    @app.get("/full/entry", response_class=HTMLResponse)
    def full_entry(request: Request, key: str, session: str = "") -> HTMLResponse:
        viewer = viewer_of(request)
        if isinstance(viewer, HTMLResponse):
            return viewer
        answer = client(viewer).full_entry(key)
        if not answer.available:
            return unavailable(viewer, answer)
        entry = answer.data if isinstance(answer.data, Mapping) else None
        back = (
            viewer.href(f"/session/{session}")
            if session and _SESSION_ID.fullmatch(session)
            else viewer.href("/")
        )
        return page(
            "full.html",
            status=200 if entry else 404,
            viewer=viewer,
            entry=entry,
            back=back,
        )

    configure_interface_app(app, static_dir=_HERE / "static")
    return app
