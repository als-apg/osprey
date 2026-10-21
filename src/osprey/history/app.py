"""The history page: the across-users view of the agent record.

Served on the readers' origin under ``/history/``. nginx asks the auth sidecar
who the browser is, strips the ``/history`` prefix and forwards the request
here with the sidecar's ``X-Osprey-History-*`` answer and this service's gate
secret. The gate is :class:`~osprey.interfaces.common_middleware.WebAuthMiddleware`:
the gate secret arrives through the roster-secret seam, so a request that did
not come through nginx is refused before any route runs.

Every page is a set of view-contract queries (:mod:`osprey.history.queries`)
run as the viewer's store view user (:mod:`osprey.history.store`): the store's
grants and row policies decide what a person sees, and the page never widens a
read. A view whose table the view user may not read is not queried at all and
says so ("not in your tables").

The app receives bare paths (nginx strips ``/history``) and writes
:data:`~osprey.history.viewer.ROOT_PATH` back onto every link and asset URL it
renders. It sets no ASGI ``root_path``: Starlette's static mounts would then
expect the prefix on the bare paths nginx forwards and answer every asset 404.
"""

from __future__ import annotations

import datetime
import os
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

from osprey.deployment.telemetry_credentials import HISTORY_PASSWORD_ENV
from osprey.history import queries
from osprey.history.store import HistoryStore, StoreRefusal
from osprey.history.transcript import (
    assign_approvals,
    group_sessions,
    match_preflight,
    parse_ts,
    timeline,
    visible_preflight,
)
from osprey.history.viewer import ROOT_PATH, MissingViewer, Viewer, viewer_from_headers

#: Where the page reads its store, its queueservers and its "older" limit.
STORE_URL_ENV = "OSPREY_HISTORY_STORE_URL"
QUEUESERVERS_ENV = "OSPREY_HISTORY_QUEUESERVERS"
MAX_DAYS_ENV = "OSPREY_HISTORY_MAX_DAYS"

LOGS_TABLE = "otel_logs"
TRACES_TABLE = "otel_traces"
LEDGER_TABLE = "audit_ledger"

#: How many users one deep link shows a transcript for.
DEEP_LINK_USERS = 20
#: The most days the "older" control steps back when nothing configures it.
DEFAULT_MAX_DAYS = 30

_HERE = Path(__file__).parent
_SESSION_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the store connection the page runs on, unless the factory built one.

    A store :func:`create_app` already holds (a test's ``transport``) is kept:
    replacing it here would point a served test app at a real store. Only a
    connection this lifespan opened is closed by it.
    """
    if getattr(app.state, "store", None) is not None:
        yield
        return
    client = httpx.AsyncClient()
    app.state.store = HistoryStore(app.state.store_url, app.state.store_password, client=client)
    try:
        yield
    finally:
        await client.aclose()


def _templates() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_HERE / "templates")),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals["root"] = ROOT_PATH
    env.globals["preview_chars"] = queries.PREVIEW_CHARS
    env.globals["redacted"] = queries.REDACTED
    return env


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


async def _transcript(
    store: HistoryStore,
    queueservers: tuple[str, ...],
    viewer: Viewer,
    user: str,
    session_id: str,
    bounds: dict[str, Any],
    cursors: dict[str, Any],
) -> dict[str, Any]:
    """One user's transcript of one session: rows, refusals and the next page."""
    result: dict[str, Any] = {"user": user, "session": session_id, "missing": []}
    conversation: list[dict[str, Any]] = []
    tools: list[dict[str, Any]] = []
    preflight_rows: list[dict[str, Any]] = []
    approvals: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    refusals: list[StoreRefusal] = []

    async def read(name: str, values: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            return await store.rows(viewer, name, values)
        except StoreRefusal as refusal:
            refusals.append(refusal)
            return []

    if viewer.reads(LOGS_TABLE):
        conversation = await read(
            "conversation",
            {
                "user": user,
                "session": session_id,
                "after_ts": cursors["after_ts"] or queries.FIRST_CURSOR,
                "after_hash": cursors["after_hash"] or 0,
            },
        )
    else:
        result["missing"].append(LOGS_TABLE)
    start = parse_ts(bounds.get("first_ts")) - datetime.timedelta(seconds=60)
    end = parse_ts(bounds.get("last_ts")) + datetime.timedelta(seconds=60)
    if viewer.reads(LEDGER_TABLE):
        tools = await read(
            "tool_log",
            {
                "user": user,
                "session": session_id,
                "after_ts": cursors["tool_after_ts"] or queries.FIRST_CURSOR,
                "after_hash": cursors["tool_after_hash"] or 0,
            },
        )
        if bounds.get("first_ts"):
            approvals = await read("approvals", {"user": user, "from": start, "to": end})
            if queueservers:
                preflight_rows = await read(
                    "preflight",
                    {
                        "queueservers": list(queueservers),
                        "user": user,
                        "from": start,
                        "to": end,
                    },
                )
    else:
        result["missing"].append(LEDGER_TABLE)
    if viewer.reads(TRACES_TABLE):
        agents = await read("agents", {"user": user, "session": session_id})

    items = visible_preflight(match_preflight(preflight_rows, tools), own=viewer.own)
    by_tool, loose = assign_approvals(
        approvals,
        tools,
        terminal=str(bounds.get("terminal_session") or ""),
        first_ts=bounds.get("first_ts"),
        last_ts=bounds.get("last_ts"),
    )
    result["entries"] = timeline(conversation, tools, items, by_tool, loose)
    result["agents"] = agents
    result["refusals"] = refusals
    result["terminal"] = bounds.get("terminal_session") or ""
    # The next page starts after the last row shown of each source, so a
    # source that ended on this page contributes nothing twice.
    more: dict[str, Any] = {}
    for prefix, rows in (("", conversation), ("tool_", tools)):
        if rows:
            more[f"{prefix}after_ts"] = rows[-1].get("Timestamp")
            more[f"{prefix}after_hash"] = rows[-1].get("body_hash")
        else:
            more[f"{prefix}after_ts"] = cursors[f"{prefix}after_ts"]
            more[f"{prefix}after_hash"] = cursors[f"{prefix}after_hash"]
    full = len(conversation) >= queries.PAGE_LIMIT or len(tools) >= queries.PAGE_LIMIT
    result["next_href"] = viewer.href(f"/session/{session_id}", user=user, **more) if full else None
    return result


def create_app(
    *,
    store_url: str | None = None,
    store_password: str | None = None,
    queueservers: tuple[str, ...] | None = None,
    max_days: int | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the history page.

    Every argument defaults to the environment the compose template renders;
    *transport* replaces the store connection (tests pass an
    :class:`httpx.MockTransport`).
    """
    from osprey.interfaces._app_setup import configure_interface_app

    app = FastAPI(title="OSPREY history", lifespan=_lifespan)
    app.state.store_url = store_url or os.environ.get(STORE_URL_ENV, "http://telemetry_store:8123")
    app.state.store_password = store_password or os.environ.get(HISTORY_PASSWORD_ENV, "")
    if queueservers is None:
        raw = os.environ.get(QUEUESERVERS_ENV, "")
        queueservers = tuple(name.strip() for name in raw.split(",") if name.strip())
    app.state.queueservers = queueservers
    if max_days is None:
        raw_days = os.environ.get(MAX_DAYS_ENV, "")
        max_days = int(raw_days) if raw_days.isdigit() else DEFAULT_MAX_DAYS
    app.state.max_days = max_days
    templates = _templates()

    if transport is not None:
        # A test's store: built here so a TestClient without lifespan still works.
        app.state.store = HistoryStore(
            app.state.store_url,
            app.state.store_password,
            client=httpx.AsyncClient(transport=transport),
        )

    def page(name: str, status: int = 200, **context: Any) -> HTMLResponse:
        return HTMLResponse(templates.get_template(name).render(**context), status_code=status)

    def refused(viewer: Viewer, refusal: StoreRefusal) -> HTMLResponse:
        return page("refused.html", status=502, viewer=viewer, refusal=refusal)

    def viewer_of(request: Request) -> Viewer | HTMLResponse:
        try:
            return viewer_from_headers(request.headers)
        except MissingViewer:
            return page("no_scope.html", status=403, viewer=None)

    @app.get("/health", include_in_schema=False)
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/", response_class=HTMLResponse)
    async def session_list(
        request: Request,
        older: int = 0,
        cursor_ts: str | None = None,
        cursor_user: str = "",
        cursor_session: str = "",
    ) -> HTMLResponse:
        viewer = viewer_of(request)
        if isinstance(viewer, HTMLResponse):
            return viewer
        older = max(0, min(older, app.state.max_days))
        if not viewer.reads(LOGS_TABLE):
            return page("sessions.html", viewer=viewer, not_in_tables=LOGS_TABLE, groups=[])
        now = _now()
        days = datetime.timedelta(days=1)
        window = queries.SESSION_LIST_WINDOW_DAYS * days
        if older:
            to = now - window - (older - 1) * days
            start = to - days
        else:
            to, start = now, now - window
        values = {
            "from": start,
            "to": to,
            "cursor_ts": cursor_ts or to,
            "cursor_user": cursor_user,
            "cursor_session": cursor_session,
        }
        try:
            rows = await app.state.store.rows(viewer, "session_list", values)
        except StoreRefusal as refusal:
            return refused(viewer, refusal)
        next_href = None
        if len(rows) >= queries.SESSION_LIST_LIMIT:
            last = rows[-1]
            next_href = viewer.href(
                "/",
                older=older or None,
                cursor_ts=last.get("last_ts"),
                cursor_user=last.get("User"),
                cursor_session=last.get("SessionId"),
            )
        older_href = None
        if queries.SESSION_LIST_WINDOW_DAYS + older < app.state.max_days:
            older_href = viewer.href("/", older=older + 1)
        newer_href = viewer.href("/", older=older - 1 or None) if older else None
        return page(
            "sessions.html",
            viewer=viewer,
            groups=group_sessions(rows),
            window_from=start,
            window_to=to,
            next_href=next_href,
            older_href=older_href,
            newer_href=newer_href,
            not_in_tables=None,
        )

    @app.get("/session/{session_id}", response_class=HTMLResponse)
    async def session_page(
        request: Request,
        session_id: str,
        user: str | None = None,
        after_ts: str | None = None,
        after_hash: str | None = None,
        tool_after_ts: str | None = None,
        tool_after_hash: str | None = None,
    ) -> HTMLResponse:
        viewer = viewer_of(request)
        if isinstance(viewer, HTMLResponse):
            return viewer
        if not _SESSION_ID.fullmatch(session_id):
            return page("not_found.html", status=404, viewer=viewer, session=session_id)
        if not viewer.reads(LOGS_TABLE):
            return page(
                "session.html",
                viewer=viewer,
                session=session_id,
                transcripts=[],
                not_in_tables=LOGS_TABLE,
            )
        try:
            users = await app.state.store.rows(viewer, "session_users", {"session": session_id})
        except StoreRefusal as refusal:
            return refused(viewer, refusal)
        if user is not None:
            users = [row for row in users if row.get("User") == user]
        if not users:
            return page("not_found.html", status=404, viewer=viewer, session=session_id)
        cursors = {
            "after_ts": after_ts,
            "after_hash": after_hash,
            "tool_after_ts": tool_after_ts,
            "tool_after_hash": tool_after_hash,
        }
        transcripts = [
            await _transcript(
                app.state.store,
                app.state.queueservers,
                viewer,
                str(row.get("User")),
                session_id,
                row,
                cursors,
            )
            for row in users[:DEEP_LINK_USERS]
        ]
        return page(
            "session.html",
            viewer=viewer,
            session=session_id,
            transcripts=transcripts,
            not_in_tables=None,
        )

    async def full_record(
        request: Request,
        table: str,
        query: str,
        values: dict[str, Any],
        back: Callable[[Viewer], str],
    ) -> HTMLResponse:
        viewer = viewer_of(request)
        if isinstance(viewer, HTMLResponse):
            return viewer
        if not viewer.reads(table):
            return page("full.html", viewer=viewer, rows=[], not_in_tables=table)
        try:
            rows = await app.state.store.rows(viewer, query, values)
        except StoreRefusal as refusal:
            return refused(viewer, refusal)
        return page(
            "full.html",
            status=200 if rows else 404,
            viewer=viewer,
            rows=rows,
            not_in_tables=None,
            back=back(viewer),
        )

    @app.get("/full/log", response_class=HTMLResponse)
    async def full_log(
        request: Request, user: str, session: str, ts: str, body_hash: str
    ) -> HTMLResponse:
        return await full_record(
            request,
            LOGS_TABLE,
            "load_full_log",
            {"user": user, "session": session, "ts": ts, "body_hash": body_hash},
            lambda viewer: viewer.href(f"/session/{session}", user=user),
        )

    @app.get("/full/ledger", response_class=HTMLResponse)
    async def full_ledger(
        request: Request, user: str, tool_use_id: str, surface: str, session: str = ""
    ) -> HTMLResponse:
        return await full_record(
            request,
            LEDGER_TABLE,
            "load_full_ledger",
            {"user": user, "tool_use_id": tool_use_id, "surface": surface},
            lambda viewer: (
                viewer.href(f"/session/{session}", user=user) if session else viewer.href("/")
            ),
        )

    configure_interface_app(app, static_dir=_HERE / "static")
    return app
