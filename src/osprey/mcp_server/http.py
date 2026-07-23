"""HTTP/IPC utilities for MCP tool code."""

import logging

logger = logging.getLogger("osprey.mcp_server.http")


def gallery_url() -> str:
    """Build the gallery base URL from config."""
    from osprey.utils.workspace import load_osprey_config

    config = load_osprey_config()
    art_config = config.get("artifact_server", {})
    host = art_config.get("host", "127.0.0.1")
    port = art_config.get("port", 8086)
    return f"http://{host}:{port}"


def web_terminal_url() -> str:
    """Build the web terminal base URL from config.

    In containerized deployments the actual port is set via OSPREY_WEB_PORT
    (docker-compose env), which may differ from the default in config.yml.
    The env var takes precedence when present.
    """
    import os

    from osprey.utils.workspace import load_osprey_config

    config = load_osprey_config()
    wt = config.get("web_terminal", {})
    host = wt.get("host", "127.0.0.1")
    port = int(os.environ.get("OSPREY_WEB_PORT", wt.get("port", 8087)))
    return f"http://{host}:{port}"


def phoebus_bridge_url() -> str:
    """Build the Phoebus agent-bridge base URL from env or config.

    The bridge is the in-JVM HTTP server embedded in a running Phoebus product
    (default ``http://127.0.0.1:7979``). Resolution order:

    1. ``PHOEBUS_BRIDGE_URL`` env var (full URL) — set by the framework server
       definition; wins outright.
    2. ``PHOEBUS_BRIDGE_PORT`` env var overrides only the port.
    3. ``phoebus.host`` / ``phoebus.port`` in config.yml.
    4. ``127.0.0.1:7979`` default (matches ``bridge_preferences.properties``).
    """
    import os

    from osprey.utils.workspace import load_osprey_config

    full = os.environ.get("PHOEBUS_BRIDGE_URL")
    if full:
        return full.rstrip("/")

    config = load_osprey_config()
    ph = config.get("phoebus", {})
    host = ph.get("host", "127.0.0.1")
    port = int(os.environ.get("PHOEBUS_BRIDGE_PORT", ph.get("port", 7979)))
    return f"http://{host}:{port}"


def post_json(url: str, payload: dict, *, timeout: int = 3) -> None:
    """Fire-and-forget JSON POST to a local HTTP endpoint.

    Non-fatal: logs a warning if the target is unreachable.
    Used by focus tools and panel-focus notifications.
    """
    import json as _json
    import urllib.request

    try:
        data = _json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=timeout)
    except Exception as exc:
        logger.warning("POST %s failed (non-fatal): %s", url, exc)


def _post_json_with_response(url: str, payload: dict, *, timeout: int = 3) -> tuple[int, dict]:
    """POST JSON and return ``(status_code, parsed_body)``.

    Unlike :func:`post_json` this propagates connection-level exceptions so the
    caller can distinguish "server rejected" from "server unreachable".

    Args:
        url: Full URL to POST to.
        payload: Dict that will be JSON-serialised as the request body.
        timeout: Socket timeout in seconds.

    Returns:
        A ``(status_code, body_dict)`` tuple.  On an ``HTTPError`` the body is
        parsed from the error response (best-effort; falls back to ``{}``).

    Raises:
        urllib.error.URLError: When the target is unreachable.
        OSError: On other socket-level failures.
    """
    import json as _json
    import urllib.error
    import urllib.request

    data = _json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, _json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body: dict = {}
        try:
            body = _json.loads(exc.read())
        except Exception:
            pass
        return exc.code, body


def notify_panel_visibility(panel: str, visible: bool) -> None:
    """Fire-and-forget POST to show or hide a panel in the Web Terminal.

    Mirrors :func:`notify_panel_focus`.  Non-fatal if the web terminal is not
    running (CLI-only mode).  Show/hide is always permitted server-side so
    there is no status to surface.

    Args:
        panel: Panel identifier string.
        visible: ``True`` to show the panel, ``False`` to hide it.
    """
    base = web_terminal_url()
    post_json(
        f"{base}/api/panel-visibility",
        {"panel": panel, "visible": visible, "source": "agent"},
        timeout=2,
    )


def notify_panel_register(
    panel_id: str,
    label: str,
    url: str,
    path: str = "/",
    health_endpoint: str | None = None,
) -> dict:
    """Register a new panel with the Web Terminal server and return the outcome.

    Unlike the fire-and-forget helpers, registration is gated and validated
    server-side (SSRF allowlist, ``web.allow_runtime_panels`` flag), so the
    real HTTP response must reach the caller so the MCP tool can surface it to
    the agent.

    Args:
        panel_id: Unique panel identifier.
        label: Human-readable panel name shown in the UI.
        url: Upstream URL the Web Terminal will proxy/embed.
        path: Sub-path to open inside *url* (default ``"/"``).
        health_endpoint: Optional health-check URL; ``None`` omits the field.

    Returns:
        A dict with the following keys:

        * ``ok`` (bool) — ``True`` on HTTP 200, ``False`` otherwise.
        * ``status`` (int | None) — HTTP status code, or ``None`` when the
          web terminal was unreachable.
        * ``data`` (dict) — Parsed response body on success (``ok=True`` only).
        * ``detail`` (str) — Server ``"detail"`` string on rejection, or
          ``"Web Terminal is not running."`` when unreachable (``ok=False`` only).
    """
    base = web_terminal_url()
    payload: dict = {
        "id": panel_id,
        "label": label,
        "url": url,
        "path": path,
        "health_endpoint": health_endpoint,
        "source": "agent",
    }
    try:
        status, body = _post_json_with_response(f"{base}/api/panels/register", payload, timeout=5)
    except Exception as exc:
        logger.warning("panel register POST failed (web terminal unreachable): %s", exc)
        return {"ok": False, "status": None, "detail": "Web Terminal is not running."}

    if status == 200:
        return {"ok": True, "status": 200, "data": body}
    detail = body.get("detail", "") if isinstance(body, dict) else str(body)
    return {"ok": False, "status": status, "detail": detail}


def notify_panel_focus(panel_id: str, url: str | None = None) -> None:
    """Fire-and-forget POST to switch the Web Terminal's active panel.

    Non-fatal if the web terminal is not running (CLI-only mode).
    """
    base = web_terminal_url()
    payload: dict = {"panel": panel_id, "source": "agent"}
    if url is not None:
        payload["url"] = url
    post_json(f"{base}/api/panel-focus", payload, timeout=2)


def notify_agent_activity(
    tool: str,
    kind: str,
    panel: str | None = None,
    detail: str | None = None,
) -> None:
    """Fire-and-forget POST reporting agent tool activity to the Web Terminal.

    Posts ``{"tool": tool, "target": {"kind": kind, "panel"?: ..., "detail"?: ...}}``
    to ``/api/agent-activity`` so the UI can highlight what the agent is
    touching.  ``panel``/``detail`` are omitted from the body when ``None``.
    Non-fatal if the web terminal is not running (CLI-only mode): all
    exceptions are swallowed and this function never raises.

    .. warning::
        This function performs a **blocking** HTTP call (bounded at 1s).
        Async tools MUST call it via ``await anyio.to_thread.run_sync(...)``
        — never inline in a coroutine.  Calling a sync ``post_json``-style
        helper directly from async code stalls the event loop (a known
        foot-gun with the existing sync-post-in-async pattern; do not copy it).

    Args:
        tool: Name of the tool the agent invoked.
        kind: Activity target kind (validated server-side; unknown kinds 422).
        panel: Optional panel identifier the activity targets.
        detail: Optional free-form detail (e.g. channel name, file path).
            Truncated to the route's 1024-char bound so an unbounded caller
            (e.g. a bulk channel write) cannot turn the emit into a silent 422.
    """
    try:
        target: dict = {"kind": kind}
        if panel is not None:
            target["panel"] = panel
        if detail is not None:
            if len(detail) > 1024:
                detail = detail[:1023] + "…"
            target["detail"] = detail
        base = web_terminal_url()
        post_json(f"{base}/api/agent-activity", {"tool": tool, "target": target}, timeout=1)
    except Exception as exc:
        logger.warning("agent-activity notify failed (non-fatal): %s", exc)
