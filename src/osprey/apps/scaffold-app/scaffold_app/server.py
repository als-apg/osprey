"""The application's backend: the panel, and the state and write routes behind the panel door.

The channel is the deployment's choice, read from the application's config
(``apps.scaffold-app.config.channel``); the application names no channel of its own.
A state read answers from the machine the request's owner is on, which the owner's
record says. A write goes to the target it was approved for, and whether it goes ahead
is the connector's verdict for the owner.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

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
from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors
from osprey_connectors.posture_store import RESERVED_OWNER_KWARG
from osprey_connectors.types import resolve_target
from scaffold_app import APP_NAME, CHANNEL_KEY, CHANNEL_SETTING, STATE_ROUTE, WRITE_ROUTE

#: Where the deployment hands this application its rendered ``apps.<name>.config``.
APP_CONFIG_ENV = "OSPREY_APP_CONFIG"
#: Where the deployment hands this application its launch token.
LAUNCH_TOKEN_ENV = APP_NAME.upper().replace("-", "_") + "_LAUNCH_TOKEN"
#: What the state and the write say when the deployment names no channel.
NO_CHANNEL = f"no channel is configured: set {CHANNEL_SETTING} in the deployment's profile"

ConnectorFor = Callable[[str], Awaitable[Any]]


def panel_dir() -> Path:
    """The panel's files: inside the installed package, else beside it in the source tree."""
    package = Path(__file__).resolve().parent
    installed = package / "panel"
    return installed if installed.is_dir() else package.parent / "panel"


def read_app_config() -> dict[str, Any]:
    """The application's config the deployment rendered; empty when it rendered none."""
    path = os.environ.get(APP_CONFIG_ENV)
    if not path or not Path(path).is_file():
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def configured_channel(config: Mapping[str, Any]) -> str | None:
    """The channel *config* names, or ``None`` when it names none."""
    channel = config.get(CHANNEL_KEY)
    return channel if isinstance(channel, str) and channel.strip() else None


def _section() -> dict[str, Any]:
    section = get_config_value("control_system", {})
    return section if isinstance(section, dict) else {}


async def build_connector(target: str) -> Any:
    """A connector for the control target *target*, connected."""
    section = _section()
    kind = resolve_target(section, target)
    config = {**section, "type": kind}
    if os.environ.get("EPICS_CA_NAME_SERVERS"):
        # In its container: compose names the name server it reaches. The config's
        # gateways are the host's view, and connecting through them would replace it.
        connectors = dict(section.get("connector") or {})
        block = {k: v for k, v in (connectors.get(kind) or {}).items() if k != "gateways"}
        config["connector"] = {**connectors, kind: block}
    register_builtin_connectors()  # idempotent; a fresh process has none registered
    return await ConnectorFactory.create_control_system_connector(config, control_target=target)


class WriteRequest(BaseModel):
    """One approved write: the value, the target it was approved for, what it was approved over."""

    model_config = ConfigDict(extra="forbid")

    value: float
    target: str
    readback: float


def create_app(
    connector_for: ConnectorFor = build_connector, config: Mapping[str, Any] | None = None
) -> FastAPI:
    """Build the backend.

    Args:
        connector_for: Builds the connector of one control target.
        config: The application's config; ``None`` reads the one the deployment rendered.
    """
    channel = configured_channel(read_app_config() if config is None else config)
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

    async def read(name: str, target: str) -> float:
        try:
            value = await (await connector(target)).read_channel(name)
        except (ChannelError, ChannelReadFailedError, ConnectorUnavailableError, OSError) as exc:
            raise HTTPException(503, f"{name} could not be read: {exc}") from exc
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
        if channel is None:
            return {"channel": None, "target": None, "readback": None, "error": NO_CHANNEL}
        owner = owner_from_header(getattr(request.state, OWNER_STATE_KEY, None))
        target = owner_target(_section(), owner)
        if target is None:
            raise HTTPException(503, f"the control target of {owner} could not be read")
        return {
            "channel": channel,
            "target": target,
            "readback": await read(channel, target),
            "error": None,
        }

    @app.post(WRITE_ROUTE)
    async def write(body: WriteRequest, request: Request) -> dict[str, Any]:
        verify_launch_token(request.headers.get(LAUNCH_TOKEN_HEADER), LAUNCH_TOKEN_ENV)
        if channel is None:
            raise HTTPException(409, NO_CHANNEL)
        current = await read(channel, body.target)
        if current != body.readback:
            raise HTTPException(
                409, f"{channel} is at {current}; the write was approved over {body.readback}"
            )
        owner = owner_from_header(getattr(request.state, OWNER_STATE_KEY, None))
        with bind_owner({RESERVED_OWNER_KWARG: owner}):
            try:
                result = await (await connector(body.target)).write_channel_checked(
                    channel, body.value
                )
            except ChannelWriteBlockedError as exc:
                raise HTTPException(403, str(exc)) from exc
            except (ChannelError, ConnectorUnavailableError, OSError) as exc:
                raise HTTPException(503, f"{channel} could not be written: {exc}") from exc
        return {"channel": channel, "target": body.target, "outcome": str(result.outcome)}

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
