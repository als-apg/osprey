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


def notify_panel_focus(panel_id: str, url: str | None = None) -> None:
    """Fire-and-forget POST to switch the Web Terminal's active panel.

    Non-fatal if the web terminal is not running (CLI-only mode).
    """
    base = web_terminal_url()
    payload: dict = {"panel": panel_id}
    if url is not None:
        payload["url"] = url
    post_json(f"{base}/api/panel-focus", payload, timeout=2)
