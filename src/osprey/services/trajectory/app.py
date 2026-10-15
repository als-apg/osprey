"""The trajectory service's one gated app: a panel bundle and two doors.

One FastAPI app behind the shared web gate (:func:`configure_interface_app`:
``WebAuthMiddleware`` admits on the terminal secret, a roster secret or the
panel token, and stamps the owner the credential names; ``HttpAuditMiddleware``
sits innermost):

- ``/panel/`` — the panel bundle;
- ``/panel/api/*`` — the **panel door**, kind ``panel``;
- ``/api/*`` — the **agent door**, kind ``agent``, which also refuses a caller
  without this service's launch token (``TRAJECTORY_LAUNCH_TOKEN``);
- ``GET /health`` — exempt from the gate; asks the target live.

Each door serves ``POST session``, ``POST takeover``, ``DELETE session`` and
``GET state``. The door is the route, never the request; the owner is the
account the gate minted from the credential that admitted the request, read
through :func:`owner_from_header` on both doors — a name it refuses is no
owner, and an action with no owner is refused by ``admit``. The panel door
never checks the launch token.

A refusal is ``HTTPException(detail={code, detail})``. Each writing action is
recorded once, by the route, on :data:`SURFACE_TRAJECTORY_SERVICE` with the
owner as actor: a control-path action (take-over, release) as it is decided
at admission, a worker action (open) after the worker answered. A release is
never admitted, so its record carries only the owner, the target and the
``client_id``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles

from osprey.interfaces._app_setup import configure_interface_app
from osprey.interfaces.common_middleware import OWNER_STATE_KEY
from osprey.services.trajectory import reach
from osprey.services.trajectory.actions import ActionLayer
from osprey.services.trajectory.admit import AdmitGate
from osprey.services.trajectory.reach import REACHABLE
from osprey.services.trajectory.refusal import Refused
from osprey.services.trajectory.sessions import (
    BAD_REQUEST,
    SessionRefused,
    Sessions,
)
from osprey.services.trajectory.startup import TrajectoryStartup
from osprey.utils.launch_token import LAUNCH_TOKEN_HEADER, verify_launch_token
from osprey.utils.owner_header import owner_from_header

logger = logging.getLogger(__name__)

__all__ = [
    "AGENT_DOOR",
    "CLIENT_HEADER",
    "LAUNCH_TOKEN_ENV",
    "PANEL_DOOR",
    "create_app",
]

#: The door ``/api/*`` admits through: the agent's, behind the launch token.
AGENT_DOOR = "agent"
#: The door ``/panel/api/*`` admits through: the panel's.
PANEL_DOOR = "panel"

#: The environment variable that holds the agent door's launch token.
LAUNCH_TOKEN_ENV = "TRAJECTORY_LAUNCH_TOKEN"

#: The header a holder's later calls carry its ``client_id`` in.
CLIENT_HEADER = "X-Trajectory-Client"

#: The panel bundle, served under ``/panel/``.
_PANEL_DIR = Path(__file__).parent / "panel"

#: The posture an admitted action acts under; a refusal is filed under the
#: posture that held it back.
_POSTURE_WRITES = "writes"
_POSTURE_SANDBOX = "sandbox"
_SANDBOX_CODES = frozenset({"narrowing", "writes_disabled"})

StartFn = Callable[[], Awaitable[TrajectoryStartup]]


# -- dependencies --------------------------------------------------------------


async def door_owner(request: Request) -> str | None:
    """The owner the gate minted for this request, or ``None``.

    Read from the name ``WebAuthMiddleware`` stamped from the credential that
    admitted the request; a name :func:`owner_from_header` refuses is no owner.
    """
    return owner_from_header(getattr(request.state, OWNER_STATE_KEY, None))


async def _launch_token(request: Request) -> None:
    """Refuse an agent-door request that lacks this service's launch token."""
    verify_launch_token(request.headers.get(LAUNCH_TOKEN_HEADER), LAUNCH_TOKEN_ENV)


def _sessions(request: Request) -> Sessions:
    sessions: Sessions = request.app.state.sessions
    return sessions


async def _body(request: Request) -> Mapping[str, Any] | None:
    raw = await request.body()
    if not raw.strip():
        return None
    try:
        body: Mapping[str, Any] | None = json.loads(raw)
    except ValueError as exc:
        raise SessionRefused(BAD_REQUEST, "a session request carries no body") from exc
    return body


# -- audit ---------------------------------------------------------------------


def _record(
    *,
    owner: str | None,
    subject: str,
    refusal: Refused | None,
    detail: Mapping[str, Any],
) -> None:
    """File the one record of a writing action; the action never fails on it."""
    from osprey.audit.dedup import record_and_mark
    from osprey.audit.envelope import (
        DECISION_ALLOWED,
        DECISION_REFUSED,
        POSTURE_SOURCE_APP,
        SURFACE_TRAJECTORY_SERVICE,
    )

    code = refusal.code if refusal is not None else None
    fields: dict[str, Any] = {
        "surface": SURFACE_TRAJECTORY_SERVICE,
        "posture": _POSTURE_SANDBOX if code in _SANDBOX_CODES else _POSTURE_WRITES,
        "posture_source": POSTURE_SOURCE_APP,
        "session": None,
        "subject": subject,
        "detail": " ".join(f"{key}={value}" for key, value in detail.items()),
    }
    if owner:
        fields["actor"] = owner
    try:
        record_and_mark(
            decision=DECISION_ALLOWED if refusal is None else DECISION_REFUSED,
            reason=subject if code is None else code,
            **fields,
        )
    except Exception:  # the audit trail degrades; the action does not
        logger.warning("Could not record the trajectory %s for audit", subject, exc_info=True)


@contextmanager
def _refusals_recorded(
    owner: str | None, subject: str, presented: Mapping[str, Any]
) -> Iterator[None]:
    """Record a refusal raised in the body under *presented*, and answer it as the door does."""
    try:
        yield
    except Refused as exc:
        _record(owner=owner, subject=subject, refusal=exc, detail=presented)
        raise HTTPException(status_code=exc.status, detail=exc.as_detail()) from exc


# -- the doors -----------------------------------------------------------------


def _door(kind: str, *, prefix: str, dependencies: list[Any]) -> APIRouter:
    """The four routes of door *kind*; every writing route is ``async def``,
    so its record reaches ``HttpAuditMiddleware`` on the same task."""
    router = APIRouter(prefix=prefix, dependencies=dependencies)

    @router.post("/session")
    async def open_session(
        request: Request, owner: str | None = Depends(door_owner)
    ) -> dict[str, str]:
        sessions = _sessions(request)
        detail: dict[str, Any] = {"owner": owner, "kind": kind, "target": sessions.target}
        with _refusals_recorded(owner, "session_open", detail):
            grant = await sessions.open(owner, kind, await _body(request))
        _record(owner=owner, subject="session_open", refusal=None, detail=detail)
        return grant.as_body()

    @router.post("/takeover")
    async def take_over(
        request: Request, owner: str | None = Depends(door_owner)
    ) -> dict[str, str]:
        sessions = _sessions(request)
        detail: dict[str, Any] = {"owner": owner, "kind": kind, "target": sessions.target}
        with _refusals_recorded(owner, "session_takeover", detail):
            grant = sessions.take_over(owner, kind, await _body(request))
        _record(owner=owner, subject="session_takeover", refusal=None, detail=detail)
        return grant.as_body()

    @router.delete("/session")
    async def release(request: Request, owner: str | None = Depends(door_owner)) -> dict[str, str]:
        sessions = _sessions(request)
        client_id = request.headers.get(CLIENT_HEADER, "").strip() or None
        presented = {"owner": owner, "target": sessions.target, "client_id": client_id}
        with _refusals_recorded(owner, "session_release", presented):
            if client_id is None:
                raise SessionRefused(
                    BAD_REQUEST, f"the call names no session; send {CLIENT_HEADER}"
                )
            done = sessions.release(owner, kind, client_id, await _body(request))
        _record(owner=owner, subject="session_release", refusal=None, detail=done.as_record())
        return done.as_record()

    @router.get("/state")
    async def state(request: Request, owner: str | None = Depends(door_owner)) -> dict[str, Any]:
        sessions = _sessions(request)
        return {
            **sessions.state(),
            "caller": owner,
            "writing_available": sessions.layer.startup.writing_available,
        }

    return router


async def _default_start() -> TrajectoryStartup:
    from osprey.services.trajectory.startup import start

    return await start()


def _load_config() -> Mapping[str, Any]:
    from osprey_connectors.workspace import load_osprey_config

    return load_osprey_config()


def create_app(*, start: StartFn | None = None) -> FastAPI:
    """Build the service's app.

    Args:
        start: Builds the started service; the deployment checks of
            :func:`osprey.services.trajectory.startup.start` by default.

    Returns:
        The gated app. Its lifespan starts the service, the admission gate,
        the action worker and the sessions, and closes the worker on the way
        out.
    """
    start_fn = start or _default_start

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        startup = await start_fn()
        gate = AdmitGate.from_config(_load_config(), startup.target)
        layer = ActionLayer(startup, gate)
        layer.start()
        app.state.startup = startup
        app.state.sessions = Sessions(layer)
        try:
            yield
        finally:
            await layer.close()
            disconnect = getattr(startup.connector, "disconnect", None)
            if disconnect is not None:
                try:
                    await disconnect()
                except Exception:
                    logger.warning("trajectory service: connector did not close", exc_info=True)

    app = FastAPI(title="OSPREY Trajectory", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        startup: TrajectoryStartup = request.app.state.startup
        code = await reach.probe(startup)
        return {"status": "ok", "target": code or REACHABLE, "control_target": startup.target}

    app.include_router(_door(PANEL_DOOR, prefix="/panel/api", dependencies=[]))
    app.include_router(_door(AGENT_DOOR, prefix="/api", dependencies=[Depends(_launch_token)]))
    # After the door routes: Starlette matches in declaration order, so
    # /panel/api/* reaches its routes and every other /panel/ path the bundle.
    app.mount("/panel", StaticFiles(directory=_PANEL_DIR, html=True), name="panel")
    configure_interface_app(app, static_dir=_PANEL_DIR / "static")
    return app
