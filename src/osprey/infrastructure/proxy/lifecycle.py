"""Proxy lifecycle management: start, stop, port allocation, auto-detection."""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Any

#: The two wire protocols a provider entry may declare, imported from the
#: catalog contract that owns them rather than kept as a second copy.
from osprey.profiles.providers import VALID_API_PROTOCOLS

logger = logging.getLogger("osprey.infrastructure.proxy")

# Providers known to speak Anthropic Messages API natively.
# Everything else is assumed to be OpenAI-compatible and needs the proxy.
_ANTHROPIC_NATIVE_PROVIDERS = frozenset({"anthropic", "cborg", "als-apg"})

_state: dict[str, Any] = {
    "server": None,
    "thread": None,
    "port": None,
}
_lock = threading.Lock()


def is_proxy_needed(
    provider_name: str,
    api_providers: dict | None = None,
) -> bool:
    """Determine if a provider needs the translation proxy.

    Returns True when the provider speaks OpenAI protocol but not Anthropic.

    Logic:
    1. Built-in Anthropic-native providers → False
    2. Explicit ``api_protocol: anthropic`` in config → False
    3. Everything else → True

    An absent ``api_protocol`` means OpenAI, which is right for nine of the
    twelve proxied built-ins. A PRESENT one is checked against
    :data:`~osprey.profiles.providers.VALID_API_PROTOCOLS`: the old exact
    comparison meant a typo took the step-3 branch, so a provider written
    ``api_protocol: Anthropic`` was routed through the translation proxy the
    config template explicitly warns against — with nothing said about it.
    The catalog loader refuses such a value at load; this check stays because
    an ``api.providers`` block can reach a build without passing through the
    catalog — a hand-edited ``build/config.yml``, for one.

    Args:
        provider_name: The provider being resolved.
        api_providers: The ``api.providers`` block, if the caller has one.

    Returns:
        Whether the translation proxy has to sit in front of this provider.

    Raises:
        ValueError: If this provider declares an ``api_protocol`` that is
            neither ``anthropic`` nor ``openai``.
    """
    declared = None
    if api_providers:
        provider_conf = api_providers.get(provider_name) or {}
        if isinstance(provider_conf, dict):
            declared = provider_conf.get("api_protocol")

    if declared is not None and declared not in VALID_API_PROTOCOLS:
        raise ValueError(
            f"api.providers.{provider_name}.api_protocol is {declared!r}; "
            f"expected one of {', '.join(sorted(VALID_API_PROTOCOLS))}."
        )

    if provider_name in _ANTHROPIC_NATIVE_PROVIDERS or declared == "anthropic":
        logger.info("Provider %r speaks Anthropic natively; no proxy", provider_name)
        return False

    logger.info("Provider %r speaks OpenAI; routing through the translation proxy", provider_name)
    return True


def find_free_port() -> int:
    """Find a free port on localhost using OS allocation."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_proxy(
    upstream_base_url: str,
    upstream_api_key: str | None = None,
) -> int:
    """Start the translation proxy in a daemon thread.

    Returns the port number. Thread-safe; repeated calls are no-ops.
    """
    with _lock:
        if _state["server"] is not None:
            return _state["port"]

        from osprey.infrastructure.proxy.app import create_proxy_app

        app = create_proxy_app(upstream_base_url, upstream_api_key)
        port = find_free_port()

        import uvicorn

        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)

        thread = threading.Thread(target=server.run, daemon=True, name="osprey-proxy")
        thread.start()

        # Wait for server to be ready (up to 5 seconds)
        for _ in range(50):
            if server.started:
                break
            time.sleep(0.1)

        _state["server"] = server
        _state["thread"] = thread
        _state["port"] = port

        logger.info("Translation proxy started on port %d → %s", port, upstream_base_url)
        return port


def stop_proxy() -> None:
    """Shutdown the proxy server if running."""
    with _lock:
        server = _state.get("server")
        if server is not None:
            server.should_exit = True
            thread = _state.get("thread")
            if thread:
                thread.join(timeout=5)
            _state["server"] = None
            _state["thread"] = None
            _state["port"] = None
            logger.info("Translation proxy stopped")


def get_proxy_url() -> str | None:
    """Return http://127.0.0.1:<port> if proxy is running, else None."""
    port = _state.get("port")
    return f"http://127.0.0.1:{port}" if port else None
