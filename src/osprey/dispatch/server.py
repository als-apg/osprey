"""Event dispatcher FastMCP server.

Loads triggers from triggers.yml, wires trigger sources → dispatch pool →
dispatch worker client, and exposes MCP tools plus a dashboard for inspection.

The server uses a two-phase startup that respects FastMCP's route-timing rule:
``create_server()`` (factory time) registers ALL routes — dashboard routes plus
the webhook routes owned by :class:`~osprey.dispatch.source_registry.SourceRegistry`
— before the ASGI app is built, while ``_dispatch_lifespan`` (lifespan time, with a
running event loop) registers triggers into the registry and starts the sources.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from osprey.dispatch.dashboard import render_dashboard_html
from osprey.dispatch.mcp_tools import register_tools
from osprey.dispatch.pool import DispatchPool, QueueFullError
from osprey.dispatch.registry import TriggerRegistry
from osprey.dispatch.source_registry import SourceRegistry
from osprey.dispatch.trigger_config import (
    DispatcherConfig,
    TriggerConfig,
    load_triggers,
)
from osprey.dispatch.worker_client import (
    AuthError,
    DispatchError,
    cancel_worker_run,
    dispatch_to_worker,
    fetch_worker_runs,
    proxy_worker_stream,
)

logger = logging.getLogger("osprey.dispatch.server")


# ---------------------------------------------------------------------------
# Dispatch helper with on_error policy handling
# ---------------------------------------------------------------------------


async def _dispatch_with_policy(
    trigger: TriggerConfig,
    payload: dict[str, Any],
    registry: TriggerRegistry,
    dispatch_target: str,
    dispatch_token: str,
    attempt: int = 1,
) -> dict | None:
    """Execute the trigger action and handle on_error policy.

    Returns the worker response dict on success (contains run_id, status),
    or None if the dispatch was dropped/failed after retries.
    """
    action = trigger.action
    prompt = action.get("prompt", "")
    allowed_tools = action.get("allowed_tools", [])

    # Per-surface prompt fragment and tool scope, resolved as
    # source default -> trigger override. No trigger source currently supplies
    # a default (see SourceRegistry) — the two locals below are the seam a
    # future source-level default would extend without moving this read site.
    # ``trigger.surface_prompt`` is already the parsed ``action.surface_prompt``
    # (TriggerConfig); ``surface_tools`` has no typed field yet, so it is read
    # straight off the free-form ``action`` mapping like ``allowed_tools`` above.
    source_default_surface_prompt: str | None = None
    surface_prompt = trigger.surface_prompt or source_default_surface_prompt

    source_default_surface_tools: list[str] | None = None
    surface_tools = action.get("surface_tools") or source_default_surface_tools

    # Fold the event payload into the prompt so payload-driven triggers can act
    # on it. The payload is UNTRUSTED input (a webhook body); the per-trigger
    # tool allowlist and the worker's denylist — not the prompt — are the
    # security controls. Empty payloads (e.g. a bare ping) add nothing.
    if payload:
        prompt = f"{prompt}\n\nEvent payload (JSON):\n{json.dumps(payload, indent=2, default=str)}"

    try:
        result = await dispatch_to_worker(
            url=dispatch_target,
            prompt=prompt,
            allowed_tools=allowed_tools,
            token=dispatch_token,
            surface_prompt=surface_prompt,
            surface_tools=surface_tools,
        )
        await registry.record_event(trigger.name, payload, "dispatched")
        return result

    except (DispatchError, AuthError) as exc:
        # Only dispatch/auth failures flow through the on_error policy. A genuine
        # bug (any other exception) must NOT be misclassified as a retryable
        # dispatch error — let it propagate to the pool, which records it.
        error_str = str(exc)
        on_error = trigger.on_error
        policy = on_error.get("action", "drop")
        max_retries = on_error.get("max_retries", 0)
        backoff_sec = float(on_error.get("backoff_sec", 0.0))

        await registry.record_event(trigger.name, payload, f"error: {error_str}")

        if policy == "alert":
            logger.error(
                "Dispatch error for trigger '%s' (attempt %d): %s",
                trigger.name,
                attempt,
                error_str,
            )
        elif policy == "retry" and attempt <= max_retries:
            logger.warning(
                "Dispatch error for trigger '%s' (attempt %d/%d), retrying in %.1fs: %s",
                trigger.name,
                attempt,
                max_retries,
                backoff_sec,
                error_str,
            )
            if backoff_sec > 0:
                await asyncio.sleep(backoff_sec)
            return await _dispatch_with_policy(
                trigger, payload, registry, dispatch_target, dispatch_token, attempt + 1
            )
        else:
            # drop (or retry exhausted)
            logger.warning(
                "Dropping dispatch for trigger '%s' after error: %s",
                trigger.name,
                error_str,
            )
        return None


_SECRET_CONFIG_KEYS = {"secret", "token", "password", "key", "api_key", "signing_secret"}


def _sanitize_source_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Redact secret-like keys from a trigger's source_config.

    The dashboard read endpoints are intentionally unauthenticated (see the
    dashboard-routes note in ``create_server``), so source_config — which may
    carry a webhook signing secret or similar in some facility configs — is
    redacted before it is exposed to the dashboard.
    """
    return {k: ("***" if k.lower() in _SECRET_CONFIG_KEYS else v) for k, v in cfg.items()}


def _check_auth(request: Request) -> JSONResponse | None:
    """Bearer-token guard for dispatcher routes.

    Fail-closed and constant-time, matching ``WebhookSource._handle``: returns
    503 when ``EVENT_DISPATCHER_TOKEN`` is unset (never accept an empty token),
    401 when the credential is wrong, else ``None`` (authorized).

    The token is only ever read from the ``Authorization`` header — never a query
    parameter — so it cannot leak into access logs or browser history.
    """
    expected_token = os.environ.get("EVENT_DISPATCHER_TOKEN", "")
    if not expected_token:
        logger.error("EVENT_DISPATCHER_TOKEN is not configured; rejecting request")
        return JSONResponse({"detail": "Server misconfigured"}, status_code=503)
    auth_header = request.headers.get("Authorization", "")
    if hmac.compare_digest(auth_header, f"Bearer {expected_token}"):
        return None
    return JSONResponse({"detail": "Unauthorized"}, status_code=401)


# ---------------------------------------------------------------------------
# Lifespan: async setup once a running event loop exists
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _dispatch_lifespan(_):  # noqa: ANN001, ANN202 - FastMCP passes the app instance
    """Register triggers and start sources once the serving loop is up.

    All route registration happens at factory time in ``create_server()``; this
    lifespan only does the async work that needs a running event loop: registering
    triggers into the :class:`TriggerRegistry` and starting the trigger sources.
    """
    registry = getattr(mcp, "_dispatcher_registry", None)
    trigger_list = getattr(mcp, "_trigger_list", [])
    source_reg = getattr(mcp, "_source_registry", None)
    fire_callback = getattr(mcp, "_fire_callback", None)
    # Register triggers into the registry now that a loop exists.
    if registry is not None:
        for trigger in trigger_list:
            await registry.register(trigger)
    # Start sources (cron tasks, etc.). Webhook routes were already registered at factory time.
    if source_reg is not None and fire_callback is not None:
        await source_reg.start_all(fire_callback)
    try:
        yield
    finally:
        if source_reg is not None:
            await source_reg.stop_all()


mcp = FastMCP(
    "event_dispatcher",
    instructions=(
        "Inspect and manage event-driven dispatch triggers. "
        "List triggers, view dispatch history, check status, and manually fire triggers for testing."
    ),
    lifespan=_dispatch_lifespan,
)


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Liveness probe — returns pool status and trigger count."""
    # Registry and pool are available after create_server() initializes them
    _registry = getattr(mcp, "_dispatcher_registry", None)
    _pool = getattr(mcp, "_dispatcher_pool", None)

    pool_status = _pool.get_pool_status() if _pool else {}
    trigger_count = len(_registry._triggers) if _registry else 0

    return JSONResponse(
        {
            "status": "ok",
            "trigger_count": trigger_count,
            "pool": pool_status,
        }
    )


@mcp.custom_route("/dispatch/{dispatch_id}", methods=["GET"])
async def get_dispatch_result(request: Request) -> JSONResponse:
    """Poll a dispatch by its dispatch_id. Returns worker run_id once available."""
    unauth = _check_auth(request)
    if unauth is not None:
        return unauth

    dispatch_id = request.path_params["dispatch_id"]
    _pool = getattr(mcp, "_dispatcher_pool", None)

    if not _pool:
        return JSONResponse({"detail": "Dispatcher not initialized"}, status_code=503)

    result = _pool.get_result(dispatch_id)
    if result is None:
        return JSONResponse({"detail": f"dispatch_id {dispatch_id!r} not found"}, status_code=404)

    # Do NOT echo the internal worker URL (_dispatch_target) back to callers — it
    # is an internal topology detail and the poll response should carry only the
    # dispatch's own status/run_id.
    return JSONResponse({**result})


# FastMCP has no static-file mount (unlike the FastAPI interfaces, which mount
# StaticFiles at "/design-system"), so the dashboard's tokens.css/base.css/
# theme-boot.js/theme-manager.js are served through this hand-rolled route
# instead. Resolved once at import time — package-relative from this file, up
# to src/osprey/, then across into interfaces/design_system/static.
_DESIGN_SYSTEM_STATIC_DIR = (
    Path(__file__).parent.parent / "interfaces" / "design_system" / "static"
).resolve()


@mcp.custom_route("/design-system/{path:path}", methods=["GET"])
async def design_system_asset(request: Request) -> FileResponse | JSONResponse:
    """Serve design-system static assets (tokens.css, base.css, theme JS).

    Ungated on purpose, like the ``/dashboard`` shell (see the security-model
    note above ``create_server``'s dashboard routes): these are non-secret,
    static assets that the dashboard's ``<head>`` must load before any auth
    handshake happens.

    Path-traversal guard: the requested path is joined onto the static root
    and resolved (following ``..`` segments and symlinks), then checked with
    ``is_relative_to`` against the resolved root. Anything that escapes the
    root — or simply doesn't exist — gets a 404, never a 403, so a traversal
    probe learns nothing about what does or doesn't exist outside the root.
    """
    requested = request.path_params["path"]
    candidate = (_DESIGN_SYSTEM_STATIC_DIR / requested).resolve()
    if not candidate.is_relative_to(_DESIGN_SYSTEM_STATIC_DIR) or not candidate.is_file():
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return FileResponse(candidate)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def create_server() -> FastMCP:
    """Initialize the event dispatcher server (factory time, fully synchronous).

    Loads triggers.yml, wires the dispatch pool and fire callback, registers
    trigger-source routes (via :class:`SourceRegistry`), dashboard routes, and
    MCP tools, then returns the configured FastMCP instance. The actual trigger
    registration and source startup happen later in ``_dispatch_lifespan``.
    """
    triggers_path = os.environ.get("TRIGGERS_YML", "/app/triggers.yml")
    # Outbound bearer for dispatcher->worker calls. This is the WORKER's token
    # (DISPATCH_WORKER_TOKEN), distinct from EVENT_DISPATCHER_TOKEN which guards
    # inbound webhook/dashboard auth. Captured once at factory time, so a runtime
    # token rotation is only picked up for outbound calls on restart (inbound auth
    # via _check_auth re-reads the env per request).
    dispatch_token = os.environ.get("DISPATCH_WORKER_TOKEN", "")

    # Load triggers config
    try:
        dispatcher_cfg, trigger_list = load_triggers(triggers_path)
    except FileNotFoundError:
        logger.warning("triggers.yml not found at %s — starting with no triggers", triggers_path)
        dispatcher_cfg = DispatcherConfig(
            dispatch_target=os.environ.get("DISPATCH_TARGET", "http://localhost:9088")
        )
        trigger_list = []

    dispatch_target = dispatcher_cfg.dispatch_target

    # Build registry and pool
    registry = TriggerRegistry()
    pool = DispatchPool(
        max_concurrent=dispatcher_cfg.max_concurrent_runs,
        max_queue_depth=dispatcher_cfg.max_queue_depth,
    )

    async def fire_callback(trigger: TriggerConfig, payload: dict[str, Any]) -> str | None:
        """Fire a trigger through the dispatch pool.

        Returns the dispatch_id of the queued run, or None if the trigger is
        disabled (the webhook source maps None to HTTP 409). Raises
        :class:`QueueFullError` when the pool is saturated (mapped to HTTP 429).
        """
        if registry._status.get(trigger.name) == "disabled":
            await registry.record_event(trigger.name, payload, "ignored: disabled")
            return None

        async def fn() -> dict | None:
            return await _dispatch_with_policy(
                trigger, payload, registry, dispatch_target, dispatch_token
            )

        # _dispatch_with_policy owns outcome recording ("dispatched" on success,
        # "error: ..." on failure); do not also record here or the dashboard
        # timeline double-counts each fire. Queued-but-pending state is observable
        # via the pool's _results.
        return await pool.submit(trigger.name, fn)

    # Trigger sources: discover from the ``osprey.trigger_sources`` entry-point
    # group (built-ins: webhook, cron — see pyproject.toml), then register their
    # routes at factory time. A source type with no registered class is skipped
    # gracefully (logged in SourceRegistry.setup).
    source_reg = SourceRegistry()
    source_reg.discover()
    source_reg.setup(trigger_list, mcp)  # FACTORY: registers webhook routes

    # Stash on mcp instance for use in routes and the lifespan.
    mcp._dispatcher_registry = registry  # type: ignore[attr-defined]
    mcp._dispatcher_pool = pool  # type: ignore[attr-defined]
    mcp._dispatch_target = dispatch_target  # type: ignore[attr-defined]
    mcp._trigger_list = trigger_list  # type: ignore[attr-defined]
    mcp._source_registry = source_reg  # type: ignore[attr-defined]
    mcp._fire_callback = fire_callback  # type: ignore[attr-defined]

    # -----------------------------------------------------------------------
    # Dashboard routes (closures over registry, pool, dispatch_target)
    #
    # SECURITY MODEL: the dashboard DATA endpoints (/dashboard/triggers,
    # /dashboard/runs, /dashboard/state, /dashboard/stream/{id}) AND /dispatch/{id}
    # are bearer-gated via _check_auth — they surface agent output, trigger
    # prompts, and run history. The dashboard reaches them two ways:
    #   • In-terminal EVENTS tab: the web-terminal proxy injects the bearer token
    #     server-side (the browser never holds it).
    #   • Standalone (direct to :8020): the dashboard JS supplies the token from a
    #     one-time ``#token=`` fragment handoff (kept out of server logs) as an
    #     Authorization header on its fetch/poll calls. The live SSE stream is
    #     header-gated too, so it is unavailable on the standalone path (EventSource
    #     cannot set headers, and we deliberately do NOT accept a ``?token=`` query
    #     that would leak the bearer into access logs); polled state still renders.
    # WRITE endpoints (/retry, /dashboard/cancel, /trigger/.../status) are gated the
    # same way. The /dashboard HTML SHELL itself stays ungated on purpose: it carries
    # no agent data, and the standalone token handoff requires the page to load so
    # its JS can read the fragment. Secret-like source_config keys are still redacted
    # in /dashboard/state. Production deployments should still network-isolate the port.
    # -----------------------------------------------------------------------

    @mcp.custom_route("/dashboard", methods=["GET"])
    async def dashboard(request: Request) -> HTMLResponse:
        """Serve the unified dashboard HTML with runtime config injected.

        Ungated on purpose (see the security-model note above): the shell carries
        no agent data and must load so its JS can complete the standalone token
        handoff; every DATA endpoint it then calls is bearer-gated.
        """
        return HTMLResponse(
            render_dashboard_html(
                facility_name=os.environ.get("OSPREY_FACILITY_NAME", ""),
                pv_strip_prefix=os.environ.get("PV_STRIP_PREFIX", ""),
            )
        )

    @mcp.custom_route("/dashboard/triggers", methods=["GET"])
    async def dashboard_triggers(request: Request) -> JSONResponse:
        """Return trigger config JSON for the dashboard sidebar."""
        unauth = _check_auth(request)
        if unauth is not None:
            return unauth
        triggers = await registry.list_triggers()
        for t in triggers:
            cfg = registry._triggers.get(t["name"])
            if cfg:
                t["on_error"] = cfg.on_error.get("action", "drop")
                t["allowed_tools"] = cfg.action.get("allowed_tools", [])
                t["prompt"] = cfg.action.get("prompt", "")
        return JSONResponse(triggers)

    @mcp.custom_route("/dashboard/runs", methods=["GET"])
    async def dashboard_runs(request: Request) -> JSONResponse:
        """Proxy runs from the worker, enriched with trigger_name from pool."""
        unauth = _check_auth(request)
        if unauth is not None:
            return unauth
        try:
            runs = await fetch_worker_runs(dispatch_target, dispatch_token)
        except DispatchError as exc:
            # Surface the degraded state instead of masking it as an empty success.
            logger.error("Failed to fetch runs from worker: %s", exc)
            return JSONResponse(
                {"detail": "worker unreachable", "worker_error": str(exc)}, status_code=502
            )

        # Build reverse index: run_id → (trigger_name, dispatch_id)
        reverse: dict[str, dict] = {}
        for did, res in pool._results.items():
            if res.get("status") == "completed":
                rid = (res.get("result") or {}).get("run_id")
                if rid:
                    reverse[rid] = {
                        "trigger_name": res.get("trigger_name"),
                        "dispatch_id": did,
                    }

        for run in runs:
            enrichment = reverse.get(run.get("run_id", ""), {})
            run["trigger_name"] = enrichment.get("trigger_name")
            run["dispatch_id"] = enrichment.get("dispatch_id")

        return JSONResponse(runs)

    @mcp.custom_route("/dashboard/stream/{run_id}", methods=["GET"])
    async def dashboard_stream(request: Request) -> StreamingResponse | JSONResponse:
        """Proxy SSE stream from the worker for a specific run.

        Header-gated like the other read routes. The in-terminal EVENTS tab
        reaches this through the web-terminal proxy, which injects the bearer
        header server-side. A direct standalone browser cannot set a header on
        EventSource, so its live stream is unavailable (by design — we do NOT
        accept the token as a query parameter, which would leak it into access
        logs); the standalone dashboard still shows runs via its polled state.
        """
        unauth = _check_auth(request)
        if unauth is not None:
            return unauth
        run_id = request.path_params["run_id"]
        return StreamingResponse(
            proxy_worker_stream(dispatch_target, dispatch_token, run_id),
            media_type="text/event-stream",
        )

    @mcp.custom_route("/dashboard/state", methods=["GET"])
    async def dashboard_state(request: Request) -> JSONResponse:
        """Unified aggregation endpoint.

        Returns {pool, triggers, runs, timeline, server_time_iso} in one shot so
        the dashboard can render a consistent snapshot with a single poll.

        Query params:
            timeline_hours (float, default 24): how far back to include timeline events.
        """
        unauth = _check_auth(request)
        if unauth is not None:
            return unauth
        try:
            timeline_hours = float(request.query_params.get("timeline_hours", "24"))
        except ValueError:
            timeline_hours = 24.0
        since_seconds = max(0.0, timeline_hours) * 3600.0

        # Triggers (enrich with config detail)
        triggers = await registry.list_triggers()
        for t in triggers:
            cfg = registry._triggers.get(t["name"])
            if cfg:
                t["on_error"] = cfg.on_error.get("action", "drop")
                t["allowed_tools"] = cfg.action.get("allowed_tools", [])
                t["prompt"] = cfg.action.get("prompt", "")
                t["source_config"] = _sanitize_source_config(cfg.source_config or {})

        # Runs (proxy + enrich with trigger_name)
        runs: list[dict[str, Any]] = []
        worker_error: str | None = None
        try:
            runs = await fetch_worker_runs(dispatch_target, dispatch_token)
        except DispatchError as exc:
            # Don't mask a down worker as an empty-but-healthy run list; carry a
            # marker so the dashboard can show a degraded state.
            logger.error("dashboard_state: failed to fetch worker runs: %s", exc)
            worker_error = str(exc)

        reverse: dict[str, dict] = {}
        for did, res in pool._results.items():
            rid = None
            if res.get("status") == "completed" and "result" in res:
                rid = (res.get("result") or {}).get("run_id")
            if rid:
                reverse[rid] = {
                    "trigger_name": res.get("trigger_name"),
                    "dispatch_id": did,
                }
        for run in runs:
            enrichment = reverse.get(run.get("run_id", ""), {})
            run["trigger_name"] = enrichment.get("trigger_name")
            run["dispatch_id"] = enrichment.get("dispatch_id")

        # Timeline per trigger — from registry history, filtered by time window
        timeline: dict[str, list[dict]] = {}
        for t in triggers:
            try:
                events = await registry.get_history(
                    t["name"], limit=1000, since_seconds=since_seconds
                )
            except KeyError:
                events = []
            compact = []
            for ev in events:
                result = ev.get("result", "")
                if result == "dispatched":
                    status = "dispatched"
                elif result.startswith("error"):
                    status = "error"
                elif result.startswith("queue_full"):
                    status = "queue_full"
                else:
                    status = result or "unknown"
                compact.append(
                    {
                        "timestamp": ev.get("timestamp"),
                        "status": status,
                    }
                )
            timeline[t["name"]] = compact

        pool_status = pool.get_pool_status()

        return JSONResponse(
            {
                "pool": pool_status,
                "triggers": triggers,
                "runs": runs,
                "timeline": timeline,
                "worker_error": worker_error,
                "server_time_iso": datetime.now(tz=UTC).isoformat(),
            }
        )

    @mcp.custom_route("/retry/{trigger_name}", methods=["POST"])
    async def retry_dispatch(request: Request) -> JSONResponse:
        """Re-fire a trigger with a payload.

        Body can contain either:
            {"payload": {...}}       — fire with the given payload
            {"dispatch_id": "..."}   — recorded as retry_of in the response
        With no body, fires with an empty payload.
        """
        unauth = _check_auth(request)
        if unauth is not None:
            return unauth

        trigger_name = request.path_params["trigger_name"]
        if trigger_name not in registry._triggers:
            return JSONResponse({"detail": f"Trigger '{trigger_name}' not found"}, status_code=404)

        trigger = registry._triggers[trigger_name]

        try:
            body = await request.json()
        except Exception:
            body = {}
        payload = body.get("payload")
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            return JSONResponse({"detail": "payload must be an object"}, status_code=400)

        async def fn() -> dict | None:
            return await _dispatch_with_policy(
                trigger, payload, registry, dispatch_target, dispatch_token
            )

        try:
            dispatch_id = await pool.submit(trigger.name, fn)
        except QueueFullError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=429)

        # _dispatch_with_policy records the outcome; don't double-record here.
        return JSONResponse(
            {"dispatched": True, "dispatch_id": dispatch_id, "retry_of": body.get("dispatch_id")},
            status_code=202,
        )

    @mcp.custom_route("/dashboard/cancel/{run_id}", methods=["POST"])
    async def dashboard_cancel(request: Request) -> JSONResponse:
        """Proxy a cancel request to the worker.

        Bearer-auth against the dispatcher's own token; the dispatcher holds
        the worker token itself so the browser never sees it.
        """
        unauth = _check_auth(request)
        if unauth is not None:
            return unauth
        run_id = request.path_params["run_id"]
        try:
            result = await cancel_worker_run(dispatch_target, dispatch_token, run_id)
        except AuthError:
            return JSONResponse({"detail": "worker auth failed"}, status_code=502)
        except DispatchError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=502)
        return JSONResponse(result)

    @mcp.custom_route("/trigger/{trigger_name}/status", methods=["PUT"])
    async def set_trigger_status(request: Request) -> JSONResponse:
        """Enable or disable a trigger at runtime.

        Body: {"status": "active" | "disabled"}
        """
        unauth = _check_auth(request)
        if unauth is not None:
            return unauth

        trigger_name = request.path_params["trigger_name"]
        if trigger_name not in registry._triggers:
            return JSONResponse({"detail": f"Trigger '{trigger_name}' not found"}, status_code=404)

        try:
            body = await request.json()
        except Exception:
            body = {}
        new_status = body.get("status")
        if new_status not in ("active", "disabled"):
            return JSONResponse(
                {"detail": "status must be 'active' or 'disabled'"},
                status_code=400,
            )

        try:
            await registry.set_status(trigger_name, new_status)
        except (KeyError, ValueError) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)

        return JSONResponse({"name": trigger_name, "status": new_status})

    # Register MCP tools. NOTE: do NOT register /webhook here — WebhookSource
    # owns it via register_routes (invoked by source_reg.setup above).
    # Pass fire_callback so manual_fire routes through the real dispatch path
    # (disabled short-circuit + per-trigger allowlist), not a record-only stub.
    register_tools(mcp, registry, pool, fire_callback)

    logger.info(
        "Event dispatcher initialized: %d trigger(s), pool max=%d",
        len(trigger_list),
        dispatcher_cfg.max_concurrent_runs,
    )

    return mcp
