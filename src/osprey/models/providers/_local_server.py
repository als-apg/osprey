"""Find a reachable local model server (Ollama, llama.cpp) across host and containers.

A server configured as ``localhost`` is reached as ``host.docker.internal`` from a
Docker container and as ``host.containers.internal`` from a Podman one, and the
other way round. This module is the one place that knows those host names:

- :func:`container_fallback_urls` lists the other spellings of a configured URL
  and never probes;
- :func:`resolve_local_server` walks the environment override, the configured
  URL and its fallbacks on every call, raising :class:`LocalServerUnreachable`
  when nothing answers;
- :func:`resolve_cached` does the same walk once per (url, probe path, env var)
  and remembers the answer, so every caller shares one cache with one staleness
  rule (``refresh=True`` re-checks it).
"""

from __future__ import annotations

import os
import threading
import time
from urllib.parse import urlsplit, urlunsplit

from osprey.utils.logger import get_logger

logger = get_logger("local_server")

_LOCAL = "local"
_DOCKER = "docker"
_CONTAINERS = "containers"
_OTHER = "other"

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1"})
_DOCKER_HOST = "host.docker.internal"
_CONTAINERS_HOST = "host.containers.internal"

# The hosts each class falls back to, in order.
_FALLBACK_ORDER: dict[str, tuple[str, ...]] = {
    _LOCAL: (_DOCKER_HOST, _CONTAINERS_HOST),
    _CONTAINERS: (_DOCKER_HOST, "localhost"),
    _DOCKER: (_CONTAINERS_HOST, "localhost"),
    _OTHER: ("localhost", _DOCKER_HOST, _CONTAINERS_HOST),
}


class LocalServerUnreachable(ConnectionError, RuntimeError):
    """No candidate URL of a local model server answered its probe.

    A ``ConnectionError`` so health checks classify it as ``unreachable``, and a
    ``RuntimeError`` so callers that catch runtime failures still catch it.
    """


def _host_class(host: str) -> str:
    if host in _LOCAL_HOSTS:
        return _LOCAL
    if host == _DOCKER_HOST:
        return _DOCKER
    if host == _CONTAINERS_HOST:
        return _CONTAINERS
    return _OTHER


def container_fallback_urls(base_url: str, default_port: int) -> list[str]:
    """The other host spellings of *base_url* to try when it does not answer.

    For a local or container host, the URL is rewritten onto each other host
    class, keeping its scheme, path and port (*default_port* when it names
    none). For any other (remote) host the fallbacks are plain
    ``http://<host>:<default_port>`` for localhost and both container hosts.

    Never probes.

    Args:
        base_url: The configured server URL.
        default_port: The server's well-known port.

    Returns:
        Fallback URLs in the order to try them, without duplicates.
    """
    parts = urlsplit(base_url)
    host = (parts.hostname or "").lower()
    host_class = _host_class(host)
    targets = _FALLBACK_ORDER[host_class]

    if host_class == _OTHER:
        candidates = [f"http://{target}:{default_port}" for target in targets]
    else:
        port = parts.port or default_port
        candidates = [
            urlunsplit((parts.scheme or "http", f"{target}:{port}", parts.path, "", ""))
            for target in targets
        ]

    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def probe(url: str, path: str, timeout: float) -> bool:
    """Whether ``GET <url><path>`` answers 200 within *timeout* seconds."""
    try:
        import requests

        response = requests.get(url.rstrip("/") + path, timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def _walk(
    url: str, *, probe_path: str, env_var: str | None, default_port: int, timeout: float
) -> tuple[str | None, list[str]]:
    """The first answering candidate (env override, url, fallbacks) and the fallbacks tried."""
    env_url = os.environ.get(env_var) if env_var else None
    if env_url:
        if probe(env_url, probe_path, timeout):
            logger.debug(f"Local server answered via {env_var} at {env_url}")
            return env_url, []
        logger.debug(f"{env_var}={env_url} not accessible, trying other options")

    if probe(url, probe_path, timeout):
        logger.debug(f"Local server answered at {url}")
        return url, []

    fallbacks = container_fallback_urls(url, default_port)
    for fallback in fallbacks:
        logger.debug(f"Attempting fallback connection at {fallback}")
        if probe(fallback, probe_path, timeout):
            logger.warning(
                f"Local server fallback: configured URL '{url}' failed, using fallback "
                f"'{fallback}'. Consider updating your configuration."
            )
            return fallback, fallbacks
    return None, fallbacks


def resolve_local_server(
    url: str,
    *,
    probe_path: str,
    env_var: str | None,
    default_port: int,
    timeout: float = 2.0,
    label: str = "local server",
) -> str:
    """Probe the env override, *url*, then its container fallbacks; return the first that answers.

    Probes on every call.

    Args:
        url: The configured server URL.
        probe_path: Path a live server answers 200 on (``/api/tags`` for Ollama).
        env_var: Environment variable whose value, when set, is tried first.
        default_port: The server's well-known port, for the fallbacks.
        timeout: Seconds per probe.
        label: Server name for the error message.

    Returns:
        The first answering URL.

    Raises:
        LocalServerUnreachable: When no candidate answers.
    """
    found, fallbacks = _walk(
        url, probe_path=probe_path, env_var=env_var, default_port=default_port, timeout=timeout
    )
    if found is not None:
        return found
    raise LocalServerUnreachable(
        f"Failed to connect to {label} at configured URL '{url}' "
        f"and all fallback URLs {fallbacks}. Please ensure {label} is running "
        f"and accessible, or update your configuration."
    )


_cache: dict[tuple[str, str, str | None], str] = {}
_cache_lock = threading.Lock()


def reset_cache() -> None:
    """Forget every cached resolution."""
    with _cache_lock:
        _cache.clear()


def resolve_cached(
    url: str,
    *,
    probe_path: str,
    env_var: str | None,
    default_port: int,
    refresh: bool = False,
    deadline_s: float | None = None,
    timeout: float = 2.0,
) -> str:
    """The reachable URL for *url*, walked once and then remembered.

    The walk is the one :func:`resolve_local_server` does (env override, url,
    container fallbacks). The first answering URL is cached under
    (url, probe_path, env_var); when nothing answers, *url* is returned and
    nothing is cached.

    Args:
        url: The configured server URL.
        probe_path: Path a live server answers 200 on.
        env_var: Environment variable whose value, when set, is tried first.
        default_port: The server's well-known port, for the fallbacks.
        refresh: Give a cached URL one probe and walk again when it fails, so a
            server that moved is found without a restart.
        deadline_s: Stop waiting for the walk after this many seconds and
            return *url*. A walk that finishes later still writes the cache.
        timeout: Seconds per probe.

    Returns:
        The cached or newly found URL, else *url*.
    """
    key = (url, probe_path, env_var)
    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None:
        if not refresh or probe(cached, probe_path, timeout):
            return cached
        with _cache_lock:
            if _cache.get(key) == cached:
                del _cache[key]

    def walk() -> str | None:
        found, _ = _walk(
            url, probe_path=probe_path, env_var=env_var, default_port=default_port, timeout=timeout
        )
        if found is not None:
            with _cache_lock:
                _cache[key] = found
        return found

    if deadline_s is None:
        found = walk()
        return found if found is not None else url

    result: list[str | None] = []
    worker = threading.Thread(target=lambda: result.append(walk()), daemon=True)
    started = time.monotonic()
    worker.start()
    worker.join(max(0.0, deadline_s - (time.monotonic() - started)))
    if result and result[0] is not None:
        return result[0]
    return url
