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
    post_json(f"{base}/api/panel-visibility", {"panel": panel, "visible": visible}, timeout=2)


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
    payload: dict = {"panel": panel_id}
    if url is not None:
        payload["url"] = url
    post_json(f"{base}/api/panel-focus", payload, timeout=2)
