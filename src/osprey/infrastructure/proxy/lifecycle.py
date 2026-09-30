"""Proxy lifecycle management: start, stop, port allocation, auto-detection."""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, TypedDict

#: The two wire protocols a provider entry may declare, imported from the
#: catalog contract that owns them rather than kept as a second copy.
from osprey.profiles.providers import VALID_API_PROTOCOLS

if TYPE_CHECKING:
    import uvicorn

logger = logging.getLogger("osprey.infrastructure.proxy")


class _ProxyState(TypedDict):
    """The running proxy's server, its thread and its port, set and cleared together."""

    server: uvicorn.Server | None
    thread: threading.Thread | None
    port: int | None


_state: _ProxyState = {
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
    1. The provider's ``api_protocol`` in config, when present, decides in
       either direction.
    2. Otherwise the ``api_protocol`` its adapter class declares, read from the
       registry without importing a built-in's class.
    3. Otherwise OpenAI.

    An absent ``api_protocol`` defers to the adapter's declaration. A PRESENT
    one is checked against
    :data:`~osprey.profiles.providers.VALID_API_PROTOCOLS`, because an
    unrecognised spelling (``api_protocol: Anthropic``) must be refused by name
    rather than fall through to the OpenAI default and route the provider
    through the translation proxy in silence.
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

    if declared is None:
        from osprey.models.provider_registry import get_provider_registry

        declared = get_provider_registry().api_protocol(provider_name)

    if declared == "anthropic":
        logger.info("Provider %r speaks Anthropic natively; no proxy", provider_name)
        return False

    logger.info("Provider %r speaks OpenAI; routing through the translation proxy", provider_name)
    return True


def find_free_port() -> int:
    """Find a free port on localhost using OS allocation."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def _request_shape(provider: str | None, supports_images: bool | None = None) -> dict[str, Any]:
    """The request parameters *provider*'s adapter class declares for its endpoint.

    An unregistered or absent provider gets ``max_tokens`` and no temperature
    rule (every model sent the caller's temperature), the OpenAI Chat
    Completions defaults. Whether the route carries images is the provider
    entry's own declaration (*supports_images*) when it makes one, else the
    adapter's; an unregistered or absent provider with no declaration takes
    none.
    """
    provider_class = None
    if provider:
        from osprey.models.provider_registry import get_provider_registry

        provider_class = get_provider_registry().get_provider(provider)
    return {
        "max_tokens_param": getattr(provider_class, "max_tokens_param", "max_tokens"),
        "accepts_temperature": (
            provider_class.accepts_temperature if provider_class is not None else None
        ),
        "supports_images": (
            supports_images
            if supports_images is not None
            else bool(getattr(provider_class, "supports_images", False))
        ),
    }


def start_proxy(
    upstream_base_url: str,
    upstream_api_key: str | None = None,
    *,
    provider: str | None = None,
    forward_headers: Iterable[str],
    supports_images: bool | None,
) -> int:
    """Start the translation proxy in a daemon thread.

    Args:
        upstream_base_url: OpenAI-compatible endpoint the proxy forwards to.
        upstream_api_key: API key for the upstream provider.
        provider: The provider behind the upstream; its adapter class decides
            the token-cap parameter and, for each request's model, whether a
            temperature is sent.
        forward_headers: The request headers the launch declared (see
            :func:`osprey.models.spend_attribution.declared_header_names`),
            forwarded to the upstream. A repeat call returns the running proxy
            unchanged, as it does for the upstream.
        supports_images: The provider entry's own ``supports_images``
            (``ClaudeCodeModelSpec.supports_images``), or ``None`` to follow the
            adapter. Required, so a launch path cannot drop a site's opt-in by
            omission. A repeat call returns the running proxy unchanged, as for
            the upstream.

    Returns the port number. Thread-safe; repeated calls are no-ops.
    """
    with _lock:
        running_port = _state["port"]
        if _state["server"] is not None and running_port is not None:
            return running_port

        from osprey.infrastructure.proxy.app import create_proxy_app

        app = create_proxy_app(
            upstream_base_url,
            upstream_api_key,
            provider=provider,
            forward_headers=frozenset(forward_headers),
            **_request_shape(provider, supports_images),
        )
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
