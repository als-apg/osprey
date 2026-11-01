"""The application's backend: the panel, and the state and write routes behind the panel door.

The application has no machine of its own. A state read answers from the machine the
request's owner is on, which the owner's record says; nobody tells the backend. A write goes
to the target it was approved for, and whether it goes ahead is the connector's verdict for
the owner: the login's tier, then that owner's switch for the target.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from knob_example import APP_NAME, STATE_ROUTE, WRITE_ROUTE
from osprey.apps.sdk import (
    LAUNCH_TOKEN_HEADER,
    OWNER_STATE_KEY,
    bind_owner,
    configure_app,
    owner_from_header,
    owner_target,
    verify_launch_token,
)
from osprey_connectors.config import get_config_value
from osprey_connectors.errors import (
    ChannelError,
    ChannelReadFailedError,
    ChannelWriteBlockedError,
    ConnectorUnavailableError,
)
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.posture_store import RESERVED_OWNER_KWARG
from osprey_connectors.types import resolve_target

#: The one channel this application shows and sets.
CHANNEL = "SR:MAG:HCM:01:CURRENT:SP"
#: Where the deployment hands this application its launch token.
LAUNCH_TOKEN_ENV = APP_NAME.upper().replace("-", "_") + "_LAUNCH_TOKEN"

ConnectorFor = Callable[[str], Awaitable[Any]]


def panel_dir() -> Path:
    """The panel's files: inside the installed package, else beside it in the source tree."""
    package = Path(__file__).resolve().parent
    installed = package / "panel"
    return installed if installed.is_dir() else package.parent / "panel"


def _section() -> dict[str, Any]:
    section = get_config_value("control_system", {})
    return section if isinstance(section, dict) else {}


async def build_connector(target: str) -> Any:
    """A connector for the control target *target*, connected."""
    section = _section()
    config = {**section, "type": resolve_target(section, target)}
    return await ConnectorFactory.create_control_system_connector(config, control_target=target)


class WriteRequest(BaseModel):
    """One approved write: the value, the target it was approved for, what it was approved over."""

    model_config = ConfigDict(extra="forbid")

    value: float
    target: str
    readback: float


def create_app(connector_for: ConnectorFor = build_connector) -> FastAPI:
    """Build the backend; *connector_for* builds the connector of one control target."""
    connectors: dict[str, Any] = {}
    building = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        for connector in connectors.values():
            await connector.disconnect()

    async def connector(target: str) -> Any:
        async with building:
            if target not in connectors:
                try:
                    connectors[target] = await connector_for(target)
                except ValueError as exc:
                    raise HTTPException(422, str(exc)) from exc
            return connectors[target]

    async def read(target: str) -> float:
        try:
            value = await (await connector(target)).read_channel(CHANNEL)
        except (ChannelError, ChannelReadFailedError, ConnectorUnavailableError, OSError) as exc:
            raise HTTPException(503, f"{CHANNEL} could not be read: {exc}") from exc
        return float(value.value)

    app = FastAPI(title=APP_NAME, lifespan=lifespan)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(panel_dir() / "index.html")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(STATE_ROUTE)
    async def state(request: Request) -> dict[str, Any]:
        owner = owner_from_header(getattr(request.state, OWNER_STATE_KEY, None))
        target = owner_target(_section(), owner)
        if target is None:
            raise HTTPException(503, f"the control target of {owner} could not be read")
        return {"channel": CHANNEL, "target": target, "readback": await read(target)}

    @app.post(WRITE_ROUTE)
    async def write(body: WriteRequest, request: Request) -> dict[str, Any]:
        verify_launch_token(request.headers.get(LAUNCH_TOKEN_HEADER), LAUNCH_TOKEN_ENV)
        current = await read(body.target)
        if current != body.readback:
            raise HTTPException(
                409, f"{CHANNEL} is at {current}; the write was approved over {body.readback}"
            )
        owner = owner_from_header(getattr(request.state, OWNER_STATE_KEY, None))
        with bind_owner({RESERVED_OWNER_KWARG: owner}):
            try:
                result = await (await connector(body.target)).write_channel_checked(
                    CHANNEL, body.value
                )
            except ChannelWriteBlockedError as exc:
                raise HTTPException(403, str(exc)) from exc
            except (ChannelError, ConnectorUnavailableError, OSError) as exc:
                raise HTTPException(503, f"{CHANNEL} could not be written: {exc}") from exc
        return {"channel": CHANNEL, "target": body.target, "outcome": str(result.outcome)}

    configure_app(app, static_dir=panel_dir())
    return app


def main() -> None:
    """Serve the backend where the deployment says: ``OSPREY_APP_BIND``, ``OSPREY_APP_PORT``."""
    uvicorn.run(
        create_app(),
        host=os.environ.get("OSPREY_APP_BIND", "127.0.0.1"),
        port=int(os.environ["OSPREY_APP_PORT"]),
    )


if __name__ == "__main__":
    main()
