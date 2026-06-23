"""Webhook trigger source for the event dispatcher.

Registers a single dynamic Starlette route ``/webhook/{trigger_name}`` on the
dispatcher's FastMCP app. The HTTP plumbing is kept thin and delegates all logic
to :meth:`WebhookSource._handle`, which is a pure coroutine that is easy to unit
test without an ASGI stack.
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar

from starlette.requests import Request
from starlette.responses import JSONResponse

from osprey.dispatch.pool import QueueFullError

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from osprey.dispatch.sources.base import FireCallback
    from osprey.dispatch.trigger_config import TriggerConfig

logger = logging.getLogger("osprey.dispatch.sources.webhook")


class WebhookSource:
    """Event source that fires triggers from authenticated webhook POSTs."""

    source_type: ClassVar[str] = "webhook"

    def __init__(self) -> None:
        self._triggers: dict[str, TriggerConfig] = {}
        self._fire_callback: FireCallback | None = None

    def register_routes(self, mcp_app: FastMCP) -> None:
        """Register the dynamic webhook route (factory time, before app build).

        The route adapter is thin: it extracts the trigger name, Authorization
        header, and JSON body, then delegates to :meth:`_handle`. The closure
        captures ``self`` and reads ``_triggers``/``_fire_callback`` at REQUEST
        time, so it is safe to register before :meth:`start` runs.
        """

        @mcp_app.custom_route("/webhook/{trigger_name}", methods=["POST"])
        async def handle_webhook(request: Request) -> JSONResponse:
            trigger_name = request.path_params["trigger_name"]
            auth_header = request.headers.get("Authorization", "")
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            body, status = await self._handle(trigger_name, auth_header, payload)
            return JSONResponse(body, status_code=status)

    async def start(self, triggers: list[TriggerConfig], fire_callback: FireCallback) -> None:
        """Record this source's triggers and fire callback (lifespan startup)."""
        self._triggers = {t.name: t for t in triggers}
        self._fire_callback = fire_callback
        logger.info("Webhook source started with %d trigger(s)", len(self._triggers))

    async def stop(self) -> None:
        """Clear state so the handler rejects everything after stop().

        Routes registered on the FastMCP app cannot be unregistered; clearing
        state makes the handler reject everything after stop().
        """
        self._triggers = {}
        self._fire_callback = None

    async def _handle(
        self, trigger_name: str, auth_header: str, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        """Core webhook logic. Returns (json_body, http_status).

        503 if ``EVENT_DISPATCHER_TOKEN`` is not configured (fail closed rather
        than accept an empty token); 401 if the bearer token is wrong; 503 if the
        source is not started; 404 if the trigger is unknown to this source; 409
        if fire_callback reports the trigger is disabled (returns None); 429 if
        the pool queue is full; 202 with the dispatch_id on success.

        Non-JSON and non-dict request bodies are coerced to an empty dict by the
        route adapter before reaching this method.
        """
        expected_token = os.environ.get("EVENT_DISPATCHER_TOKEN", "")
        if not expected_token:
            logger.error("EVENT_DISPATCHER_TOKEN is not configured; rejecting webhook")
            return {"detail": "Server misconfigured"}, 503

        # Constant-time comparison to avoid leaking the token via timing.
        if not hmac.compare_digest(auth_header, f"Bearer {expected_token}"):
            return {"detail": "Unauthorized"}, 401

        if self._fire_callback is None:
            return {"detail": "Source not started"}, 503

        if trigger_name not in self._triggers:
            return {"detail": f"Trigger '{trigger_name}' not found"}, 404

        trigger = self._triggers[trigger_name]
        try:
            dispatch_id = await self._fire_callback(trigger, payload)
        except QueueFullError as exc:
            return {"detail": str(exc)}, 429

        if dispatch_id is None:
            return {"detail": f"Trigger '{trigger_name}' is disabled"}, 409

        return {"dispatched": True, "dispatch_id": dispatch_id}, 202
