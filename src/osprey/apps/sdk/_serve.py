"""Serve an application's tool server: :func:`serve_tools`.

An application's tool module hands its ``register(server)`` function here; the
SDK builds the server, named for the application, and the function registers the
application's tools on it. The server runs the way every framework MCP server runs (project ``.env``,
logging on stderr, config primed, the audit middleware), with two additions:

* at start-up, every tool the server registers must be declared in the
  rendered config of the application ``OSPREY_APP_NAME`` names, in
  ``apps.<app>.tools.read`` or as an ``apps.<app>.approval`` entry; one that
  is in neither refuses the start, naming the tool and the config key;
* every call answers an application error type: an :class:`ApprovalRefused`
  the tool does not catch answers ``<app>.refused``, and
  ``make_error("unreachable", ...)`` answers ``<app>.unreachable``.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware

from osprey.apps.errors import AppManifestError
from osprey.apps.sdk._approval import APP_NAME_ENV_VAR, ApprovalRefused
from osprey.mcp_server.errors import make_error

if TYPE_CHECKING:
    from fastmcp import FastMCP

#: The application error word for a call refused before the application acted.
REFUSED = "refused"
#: The application error word for a backend the tool could not reach.
UNREACHABLE = "unreachable"


def _app_name() -> str:
    """The application ``OSPREY_APP_NAME`` names; a server without one refuses to start."""
    app = (os.environ.get(APP_NAME_ENV_VAR) or "").strip()
    if not app:
        raise AppManifestError(
            "",
            APP_NAME_ENV_VAR,
            "unnamed_app",
            f"the tool server is started without {APP_NAME_ENV_VAR}, so no application "
            "declares its tools; serve it as the application's projected MCP server",
        )
    return app


def _declared(config: Mapping[str, Any], app: str) -> set[str]:
    """Every tool the rendered *config* declares for *app*: ``tools.read`` and ``approval``."""
    node = config.get("apps")
    node = node.get(app) if isinstance(node, Mapping) else None
    if not isinstance(node, Mapping):
        return set()
    tools = node.get("tools")
    read = tools.get("read") if isinstance(tools, Mapping) else None
    approval = node.get("approval")
    return {
        *(name for name in read or () if isinstance(name, str)),
        *(approval if isinstance(approval, Mapping) else ()),
    }


def _refuse_undeclared(server: FastMCP, app: str, config: Mapping[str, Any]) -> None:
    """Refuse a server that registers a tool the rendered config does not declare."""
    declared = _declared(config, app)
    registered = sorted(tool.name for tool in asyncio.run(server.list_tools()))
    undeclared = [name for name in registered if name not in declared]
    if undeclared:
        names = ", ".join(undeclared)
        raise AppManifestError(
            app,
            f"apps.{app}.tools.read",
            "undeclared_tool",
            f"the tool server registers {names}, which is in neither apps.{app}.tools.read "
            f"nor apps.{app}.approval of the rendered config; declare it in the "
            "application's tools.read or tools.approval, or remove it from the server",
        )


def _app_error(app: str, exc: BaseException) -> ToolError | None:
    """The ``<app>.<type>`` error *exc* answers, ``None`` when it is no application error."""
    refused = exc if isinstance(exc, ApprovalRefused) else exc.__cause__
    if isinstance(refused, ApprovalRefused):
        details = {"reason": refused.reason, **refused.details}
        try:
            make_error(f"{app}.{REFUSED}", refused.message, details=details)
        except ToolError as error:
            return error
    if isinstance(exc, ToolError):
        try:
            envelope = json.loads(str(exc))
        except ValueError:
            return None
        if isinstance(envelope, dict) and envelope.get("error_type") == UNREACHABLE:
            envelope["error_type"] = f"{app}.{UNREACHABLE}"
            return ToolError(json.dumps(envelope))
    return None


class _AppErrors(Middleware):
    """Answer every application error of a call under its ``<app>.<type>`` name."""

    def __init__(self, app: str) -> None:
        self._app = app

    async def on_call_tool(self, context: Any, call_next: Any) -> Any:
        try:
            return await call_next(context)
        except Exception as exc:
            error = _app_error(self._app, exc)
            if error is None:
                raise
            raise error from exc


_PENDING: list[Callable[[FastMCP], None]] = []


def create_server() -> FastMCP:
    """Build the application server and have the function :func:`serve_tools` was handed fill it.

    :func:`~osprey.mcp_server.startup.run_mcp_server` calls this, so an
    application server is built and served in the one sequence that installs
    the audit middleware.
    """
    from osprey.mcp_server import startup
    from osprey_connectors.workspace import load_osprey_config

    register = _PENDING.pop()
    app = _app_name()
    startup.prime_config_builder()
    server = startup.application_server(app)
    register(server)
    _refuse_undeclared(server, app, load_osprey_config())
    server.add_middleware(_AppErrors(app))
    return server


def serve_tools(register: Callable[[FastMCP], None]) -> None:
    """Build the application's tool server, check it against the rendered config and serve it.

    Stable: part of the application SDK's public surface. The server runs as
    every framework MCP server runs, through
    :func:`~osprey.mcp_server.startup.run_mcp_server`. The application is the
    one ``OSPREY_APP_NAME`` names; its declared tools are the rendered
    config's ``apps.<app>.tools.read`` and ``apps.<app>.approval`` entries.

    Args:
        register: Registers the application's tools on the server the SDK
            built for it, named for the application; called once the project
            ``.env`` and the config are loaded, so tool code may read config.

    Raises:
        AppManifestError: ``OSPREY_APP_NAME`` is unset, or the server registers
            a tool the rendered config declares in neither ``tools.read`` nor
            ``approval``.
    """
    from osprey.mcp_server.startup import run_mcp_server

    _PENDING[:] = [register]
    try:
        run_mcp_server(__name__)
    finally:
        _PENDING.clear()
