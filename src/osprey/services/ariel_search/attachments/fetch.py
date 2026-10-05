"""Where attachment bytes may be fetched from, and how fetch talks about URLs.

* :func:`origin_of` turns an absolute http(s) URL into the (scheme, host,
  effective port) origin fetch compares; anything else has no origin.
* :func:`origins_for` is the set of origins an adapter's attachments may be
  fetched from: the adapter's own answer plus ``ariel.attachments.allowed_origins``.
  Origins come from the adapter, never from a table keyed by adapter name, so a
  registered adapter gets its own origin without an OSPREY edit.
* :func:`is_file_source` is the one answer to "is this a file source".
* :func:`redact_url` hides userinfo before a URL reaches a log line.
* :func:`fetch_attachment_bytes` is the only function that dereferences a URL
  or path taken from upstream data; it answers with one :class:`FetchOutcome`.

The origin rule applies to absolute http(s) URLs only; a relative path on a
file source is confined to the adapter's file base instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlsplit

import aiohttp

from osprey.services.ariel_search.config import _DEFAULT_PORTS, Origin

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.ingestion.base import FacilityAdapter


def origin_of(url: str | None) -> Origin | None:
    """Return the origin of an absolute http(s) URL.

    Args:
        url: Any string.

    Returns:
        ``(scheme, host, effective port)`` with scheme and host lower-cased and
        the default port filled in, so ``https://h`` and ``https://h:443`` are
        the same origin and ``https://h:8443`` is another. ``None`` when the
        value is not an absolute http(s) URL with a host and a valid port.
    """
    if not isinstance(url, str) or not url:
        return None
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not parts.hostname:
        return None
    try:
        port = parts.port
    except ValueError:
        return None
    return (scheme, parts.hostname, port if port is not None else _DEFAULT_PORTS[scheme])


def origins_for(adapter: FacilityAdapter, config: ARIELConfig) -> frozenset[Origin]:
    """Return every origin the adapter's attachments may be fetched from.

    Args:
        adapter: The ingestion adapter whose entries carry the attachments.
        config: ARIEL configuration; its ``attachments.allowed_origins`` widen
            the adapter's own answer.

    Returns:
        ``adapter.attachment_origins()`` united with the configured extra origins.
    """
    return frozenset(adapter.attachment_origins()) | frozenset(config.attachments.allowed_origins)


def is_file_source(adapter: FacilityAdapter) -> bool:
    """Return whether the adapter reads its entries from local files.

    Args:
        adapter: The ingestion adapter.

    Returns:
        True when the adapter names a file base its relative attachment paths
        resolve against.
    """
    return adapter.attachment_file_base() is not None


def redact_url(u: str) -> str:
    """Return ``u`` with any userinfo replaced by ``***``.

    Args:
        u: A URL, typically a proxy URL about to be logged.

    Returns:
        ``scheme://***@host:port`` when the URL carries a user name or
        password -- path, query and fragment are dropped with it; the port is
        the explicit one, else the scheme's default, else omitted. A URL
        without userinfo comes back unchanged.
    """
    parts = urlsplit(u)
    if "@" not in parts.netloc:
        return u
    scheme = parts.scheme.lower()
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    if port is None:
        port = _DEFAULT_PORTS.get(scheme)
    suffix = f":{port}" if port is not None else ""
    return f"{scheme}://***@{host}{suffix}"


# -- fetching ------------------------------------------------------------------

COPY_ENTRY_DEADLINE: float = 60.0
"""Seconds one entry's pictures may take to fetch and render."""

_CONNECT_TIMEOUT: float = 5.0
_SOCK_READ_TIMEOUT: float = 30.0
_MAX_REDIRECTS: int = 3
_REDIRECT_STATUSES: frozenset[int] = frozenset({301, 302, 303, 307, 308})
_HOST_UP_4XX: frozenset[int] = frozenset({408, 429})
_GONE_STATUSES: frozenset[int] = frozenset({404, 410})
_CHUNK: int = 64 * 1024


@dataclass(frozen=True)
class FetchOutcome:
    """The one answer :func:`fetch_attachment_bytes` gives for every classified result.

    Attributes:
        data: The bytes on success (empty for a ``HEAD``); ``None`` otherwise.
        code: The skip code (``source_gone``, ``source_refused``,
            ``origin_not_allowed``, ``size_cap`` or a file-source confinement
            code); ``None`` on success and for a transient failure.
        transient: True when the failure may clear on its own and the row
            stays pending.
        host_up: For a transient, True when the host answered or was reached
            (5xx, 408, 429, a read or total timeout after the request was sent);
            False for a connect-class failure.
        observed_size: What ``size_cap`` records: ``min(Content-Length, cap+1)``,
            or ``cap+1`` when the stream outgrew the cap; ``None`` otherwise.
    """

    data: bytes | None = None
    code: str | None = None
    transient: bool = False
    host_up: bool = False
    observed_size: int | None = None

    @property
    def ok(self) -> bool:
        """True when the fetch delivered bytes."""
        return self.data is not None


def _skip(code: str, *, observed_size: int | None = None) -> FetchOutcome:
    return FetchOutcome(code=code, observed_size=observed_size)


def _transient(*, host_up: bool) -> FetchOutcome:
    return FetchOutcome(transient=True, host_up=host_up)


class _SentState:
    """Tracks whether the request reached the wire and fires ``on_sent`` once."""

    def __init__(self, on_sent: Callable[[], None] | None) -> None:
        self._on_sent = on_sent
        self.sent = False

    def mark(self) -> None:
        if self.sent:
            return
        self.sent = True
        if self._on_sent is not None:
            self._on_sent()


async def _on_request_headers_sent(_session: aiohttp.ClientSession, ctx: Any, _params: Any) -> None:
    state = getattr(ctx, "trace_request_ctx", None)
    if isinstance(state, _SentState):
        state.mark()


def fetch_trace_config() -> aiohttp.TraceConfig:
    """Return the trace config that tells :func:`fetch_attachment_bytes` a request was sent.

    A session built with it reports the moment the request headers reach the
    wire; a session without it is told only when the response headers arrive.

    Returns:
        A fresh :class:`aiohttp.TraceConfig`.
    """
    trace = aiohttp.TraceConfig()
    trace.on_request_headers_sent.append(_on_request_headers_sent)
    return trace


def make_fetch_session(adapter: FacilityAdapter) -> aiohttp.ClientSession:
    """Return a session for attachment fetches through the adapter's connector.

    Args:
        adapter: The ingestion adapter; its connector carries the SOCKS proxy.

    Returns:
        A :class:`aiohttp.ClientSession` the caller closes.
    """
    return aiohttp.ClientSession(
        connector=adapter._create_connector(), trace_configs=[fetch_trace_config()]
    )


def _redirect_allowed(current: Origin, target: Origin | None) -> bool:
    """Return whether a redirect from ``current`` may be followed to ``target``.

    The target must be the exact same origin, or the same host upgraded from
    ``http`` on port 80 to ``https`` on port 443.
    """
    if target is None:
        return False
    if target == current:
        return True
    return current == ("http", current[1], 80) and target == ("https", current[1], 443)


def _is_relative_path(url: str) -> bool:
    return isinstance(url, str) and bool(url) and not urlsplit(url).scheme


_ERRNO_CODES: dict[int, str] = {
    errno.ENOENT: "source_gone",
    errno.ENOTDIR: "source_gone",
    errno.EACCES: "source_refused",
    errno.EPERM: "source_refused",
    errno.ELOOP: "not_a_regular_file",
}
_DIR_FLAGS: int = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_LEAF_FLAGS: int = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)


class _Refused(Exception):
    """A confinement failure that already knows its skip code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _confined_parts(url: str) -> list[str] | None:
    """Split a relative path into components, or ``None`` when it may escape the base."""
    if url.startswith("/") or "\x00" in url:
        return None
    parts = url.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None
    return parts


def _open_error(exc: OSError, name: str, dir_fd: int) -> _Refused:
    """Classify an ``os.open`` failure on ``name`` under ``dir_fd``.

    A symlinked component is refused as ``not_a_regular_file`` whichever errno
    the platform reports for ``O_NOFOLLOW`` meeting a link.
    """
    if exc.errno in (errno.ELOOP, errno.ENOTDIR, errno.EMLINK):
        with contextlib.suppress(OSError):
            if stat.S_ISLNK(os.lstat(name, dir_fd=dir_fd).st_mode):
                return _Refused("not_a_regular_file")
    return _Refused(_ERRNO_CODES.get(exc.errno or 0, "source_refused"))


def _read_confined(
    base: os.PathLike[str] | str, parts: list[str], cap: int, head: bool
) -> FetchOutcome:
    """Open ``parts`` one component at a time under ``base`` and read at most ``cap + 1`` bytes.

    Every intermediate is opened ``O_DIRECTORY|O_NOFOLLOW`` relative to its
    parent's descriptor and the leaf ``O_NOFOLLOW|O_NONBLOCK``, so no symlink
    is followed and a FIFO never blocks; the leaf must be a regular file by
    ``fstat`` of the open descriptor. Every descriptor is closed before return.
    """
    fds: list[int] = []
    try:
        try:
            fds.append(os.open(base, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)))
        except OSError as exc:
            return _skip(_ERRNO_CODES.get(exc.errno or 0, "source_refused"))
        for index, part in enumerate(parts):
            leaf = index == len(parts) - 1
            try:
                fds.append(os.open(part, _LEAF_FLAGS if leaf else _DIR_FLAGS, dir_fd=fds[-1]))
            except OSError as exc:
                raise _open_error(exc, part, fds[-1]) from None
        fd = fds[-1]
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return _skip("not_a_regular_file")
        if head:
            if info.st_size > cap:
                return _skip("size_cap", observed_size=cap + 1)
            return FetchOutcome(data=b"")
        chunks: list[bytes] = []
        size = 0
        while size <= cap:
            chunk = os.read(fd, min(_CHUNK, cap + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        if size > cap:
            return _skip("size_cap", observed_size=cap + 1)
        return FetchOutcome(data=b"".join(chunks))
    except _Refused as refused:
        return _skip(refused.code)
    except OSError as exc:
        return _skip(_ERRNO_CODES.get(exc.errno or 0, "source_refused"))
    finally:
        for opened in reversed(fds):
            with contextlib.suppress(OSError):
                os.close(opened)


async def _fetch_file_source(
    url: str, cap: int, adapter: FacilityAdapter, method: str
) -> FetchOutcome:
    """Read a relative attachment path confined under the adapter's file base.

    The base is ``adapter.attachment_file_base()``, resolved at fetch time;
    ``None`` refuses the path. Absolute paths and empty, ``.`` or ``..``
    components are refused unread. Open errors map to outcomes: ``ENOENT`` and
    ``ENOTDIR`` are ``source_gone``, ``EACCES`` and ``EPERM`` are
    ``source_refused``, a symlink or ``ELOOP`` is ``not_a_regular_file``; any
    other error is ``source_refused``.

    Args:
        url: The relative path from upstream data.
        cap: Byte cap.
        adapter: A file-source adapter.
        method: ``GET`` or ``HEAD``; a ``HEAD`` reads nothing.

    Returns:
        The :class:`FetchOutcome`.
    """
    base = adapter.attachment_file_base()
    parts = _confined_parts(url) if isinstance(url, str) else None
    if base is None or parts is None:
        return _skip("source_refused")
    return await asyncio.to_thread(_read_confined, base, parts, cap, method == "HEAD")


async def _read_capped(resp: aiohttp.ClientResponse, cap: int) -> FetchOutcome:
    chunks: list[bytes] = []
    size = 0
    async for chunk in resp.content.iter_chunked(_CHUNK):
        size += len(chunk)
        if size > cap:
            return _skip("size_cap", observed_size=cap + 1)
        chunks.append(chunk)
    return FetchOutcome(data=b"".join(chunks))


async def _fetch_http(
    url: str,
    cap: int,
    method: str,
    session: aiohttp.ClientSession,
    ssl: Any,
    timeout: aiohttp.ClientTimeout,
    state: _SentState,
) -> FetchOutcome:
    current = url
    current_origin = origin_of(url)
    for hop in range(_MAX_REDIRECTS + 1):
        async with session.request(
            method,
            current,
            allow_redirects=False,
            ssl=ssl,
            timeout=timeout,
            trace_request_ctx=state,
        ) as resp:
            state.mark()
            status = resp.status
            if status in _REDIRECT_STATUSES:
                location = resp.headers.get("Location")
                if not location or hop == _MAX_REDIRECTS:
                    return _skip("source_refused")
                target = urljoin(current, location)
                target_origin = origin_of(target)
                if current_origin is None or not _redirect_allowed(current_origin, target_origin):
                    return _skip("source_refused")
                current, current_origin = target, target_origin
                continue
            if 200 <= status < 300:
                length = resp.content_length
                if length is not None and length > cap:
                    return _skip("size_cap", observed_size=min(length, cap + 1))
                if method == "HEAD":
                    return FetchOutcome(data=b"")
                return await _read_capped(resp, cap)
            if status in _GONE_STATUSES:
                return _skip("source_gone")
            if status in _HOST_UP_4XX or status >= 500:
                return _transient(host_up=True)
            return _skip("source_refused")
    return _skip("source_refused")  # pragma: no cover - the loop always returns


async def fetch_attachment_bytes(
    url: str,
    cap: int,
    origins: frozenset[Origin],
    adapter: FacilityAdapter,
    method: str = "GET",
    *,
    semaphore: asyncio.Semaphore | None = None,
    session: aiohttp.ClientSession | None = None,
    total: float = COPY_ENTRY_DEADLINE,
    on_sent: Callable[[], None] | None = None,
) -> FetchOutcome:
    """Fetch one attachment named by upstream data, within the cap and the origin set.

    Absolute http(s) URLs must have an origin in ``origins``; redirects are
    followed by hand, at most three, each to the same origin (or the same host
    upgraded from http:80 to https:443). A relative path on a file source is
    read from the adapter's file base. Anything else is refused unread.

    Args:
        url: The attachment URL or relative path from upstream data.
        cap: The largest body accepted, in bytes.
        origins: Origins the URL may be fetched from; empty allows none.
        adapter: The ingestion adapter; its connector and SSL context carry the
            request.
        method: ``GET`` or ``HEAD``; a ``HEAD`` reads no body.
        semaphore: The run's concurrency limit, or ``None`` for no limit.
        session: The run's session; ``None`` builds a one-shot session from
            the adapter's connector and closes it.
        total: Total timeout in seconds for the whole fetch.
        on_sent: Called once, after the semaphore slot is held and the request
            went out on an open connection.

    Returns:
        The :class:`FetchOutcome`; this function raises only on cancellation.
    """
    method = method.upper()
    if method not in ("GET", "HEAD"):
        raise ValueError(f"method must be GET or HEAD, not {method!r}")
    origin = origin_of(url)
    if origin is None:
        if is_file_source(adapter) and _is_relative_path(url):
            return await _fetch_file_source(url, cap, adapter, method)
        return _skip("source_refused")
    if origin not in origins:
        return _skip("origin_not_allowed")

    state = _SentState(on_sent)
    timeout = aiohttp.ClientTimeout(
        total=total, connect=_CONNECT_TIMEOUT, sock_read=_SOCK_READ_TIMEOUT
    )
    slot = semaphore if semaphore is not None else contextlib.nullcontext()
    async with slot:
        try:
            ssl = adapter._ssl_context()
            if session is not None:
                return await _fetch_http(url, cap, method, session, ssl, timeout, state)
            async with make_fetch_session(adapter) as own:
                return await _fetch_http(url, cap, method, own, ssl, timeout, state)
        except asyncio.CancelledError:
            raise
        except aiohttp.ConnectionTimeoutError:
            return _transient(host_up=False)
        except (aiohttp.SocketTimeoutError, TimeoutError):
            return _transient(host_up=state.sent)
        except Exception:
            return _transient(host_up=False)
