"""The single HTTP path of every read-only service backend, and why it can only read.

A service sits on the same address as its write routes, so what a
backend may send is closed here rather than left to its query code: each backend
names its (method, path) pairs in ``ALLOWED_REQUESTS``, and anything else is
refused before a socket opens. ChannelFinder's own read is a ``GET``, but
caPutLog's search is a ``POST``, so the method alone would not say which requests
are reads.

An allowed path is exact, so ``/`` admits only the root and ``/search`` nothing
beneath it, or ends in ``/*`` to admit plain segments under that prefix. Plain
means letters, digits, ``_`` and ``-``: no dot segment, no percent escape and no
query can walk a prefix somewhere else.

Every failure is a value, because backends never raise. That covers a connection
block that cannot be used: ``read_connection_settings`` and ``urllib_opener`` raise
for it, and the reason they give names the key to fix. A malformed block is
refused earlier, by the build's validation, for the author; this is what a
deployment reads if one reaches a running process anyway.

A login or url the opener cannot send raises an exception whose text may hold the
secret, so that reason is fixed text and never the exception's. A reply is bounded
in size as well as in time, because a deadline does not bound memory on a fast link.

An access proxy in front of a service answers with a login page, often with a
success status. Reading that as a service failure would send a reader looking at
the wrong machine, so a redirect or HTML answered with 200, 401 or 403 names the
proxy instead, and a redirect is never followed: the login stays out of any request
to another address. HTML with any other status is a front-end's error page for a
service that is down or a wrong path, so it names the status. Every other content
type is parsed as JSON, because the alarm logger serves its JSON as ``text/plain``.
No ``Accept`` header is sent, for the same reason: such a route may refuse to honour
one.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from osprey_connectors.connection import (
    ConnectionSettings,
    read_connection_settings,
    urllib_opener,
)
from osprey_connectors.service_answers import Unavailable

#: The reason a redirect and an HTML answer with a login status carry.
LOGIN_PAGE_REASON = "answered with a login page (an access proxy is in front of the service)"

_MALFORMED_REASON = "malformed response"
_UNENCODABLE_QUERY_REASON = "the query cannot be sent"
_CUT_SHORT_REASON = "could not reach the service: the connection closed before the answer ended"

#: The statuses an access proxy serves its login page with.
_LOGIN_PAGE_STATUSES = frozenset({200, 401, 403})

#: The largest reply read, well above a page or a whole set at the default filter cap.
_MAX_BODY_BYTES = 64 * 1024 * 1024

#: Marks an allowed path as a prefix rather than an exact path.
_PREFIX_MARK = "/*"
_PLAIN_SEGMENTS = re.compile(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*")

#: How far into a body the ``<html`` of a page with a comment or doctype first is looked for.
_HTML_SNIFF_BYTES = 1024
_READ_CHUNK_BYTES = 65536


class _RefuseRedirects(urllib.request.BaseHandler):
    """Turn a redirect into the error it is here, ahead of urllib's following handler."""

    handler_order = urllib.request.HTTPRedirectHandler.handler_order - 1

    def _refuse(self, req, fp, code, msg, headers):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)

    http_error_301 = http_error_302 = http_error_303 = _refuse
    http_error_307 = http_error_308 = _refuse


class ReadOnlyTransport:
    """Send a backend's allowed requests; answer parsed JSON or :class:`Unavailable`.

    This is what a backend's ``transport`` keyword fills: a test passes any object
    with the same :meth:`request`. Building one opens nothing and raises nothing;
    the CA file and the login's variable are read on the first request.

    Args:
        settings: The connection block; only ``url``, ``timeout_s``, ``auth`` and
            ``tls`` are read, so a backend passes ``spec.settings`` whole.
        allowed: The backend's ``ALLOWED_REQUESTS``, as (method, path) pairs.
        where: The block's dotted key, named in every reason about the block.
    """

    def __init__(
        self, settings: Mapping[str, Any], allowed: Iterable[tuple[str, str]], *, where: str
    ) -> None:
        self._exact: set[tuple[str, str]] = set()
        self._prefixes: list[tuple[str, str]] = []
        for method, path in allowed:
            if path.endswith(_PREFIX_MARK):
                self._prefixes.append((method, path[: -len(_PREFIX_MARK)] + "/"))
            else:
                self._exact.add((method, path))
        self._where = where
        self._opener: urllib.request.OpenerDirector | None = None
        self._connection: ConnectionSettings | None = None
        self._block_reason = ""
        try:
            self._connection = read_connection_settings(settings, where=where)
        except ValueError as e:
            self._block_reason = str(e)

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
        json_body: object = None,
        deadline: float,
    ) -> Any:
        """One request, finished within *deadline* seconds or the block's ``timeout_s``.

        Args:
            method: The HTTP method, as spelled in ``ALLOWED_REQUESTS``.
            path: The path under the configured url, with no query.
            params: The query; a sequence of pairs may repeat a key.
            json_body: Sent as a JSON body when not None.
            deadline: The seconds this call may spend.

        Returns:
            The parsed JSON, or :class:`Unavailable` with the reason in plain words.
        """
        if not self._allows(method, path):
            return Unavailable(f"refused: {method} {path!r} is not a request this backend may send")
        connection = self._connection
        if connection is None:
            return Unavailable(self._block_reason)
        bound = min(deadline, connection.timeout_or(deadline))
        no_answer = Unavailable(f"no answer within {bound:g} s")
        if bound <= 0:
            return no_answer
        request = self._build_request(connection, method, path, params, json_body)
        if isinstance(request, Unavailable):
            return request
        try:
            return await asyncio.wait_for(asyncio.to_thread(self._exchange, request, bound), bound)
        except TimeoutError:
            return no_answer

    def _allows(self, method: str, path: str) -> bool:
        if (method, path) in self._exact:
            return True
        return any(
            allowed_method == method
            and path.startswith(prefix)
            and _PLAIN_SEGMENTS.fullmatch(path[len(prefix) :])
            for allowed_method, prefix in self._prefixes
        )

    def _build_request(
        self,
        connection: ConnectionSettings,
        method: str,
        path: str,
        params: Any,
        json_body: Any,
    ) -> urllib.request.Request | Unavailable:
        base = connection.url or ""
        if "?" in base or "#" in base:
            # The allowed path would land in the query or fragment, not in the path sent.
            return Unavailable(f"`{self._where}.url` must not carry a query or fragment")
        url = base.rstrip("/") + path
        if params:
            try:
                query = urllib.parse.urlencode(params, doseq=True, quote_via=urllib.parse.quote)
            except UnicodeError:
                # A lone surrogate survives JSON decoding, so a search word can carry one here.
                return Unavailable(_UNENCODABLE_QUERY_REASON)
            url = f"{url}?{query}"
        headers = {}
        data = None
        if json_body is not None:
            data = json.dumps(json_body).encode()
            headers["Content-Type"] = "application/json"
        try:
            return urllib.request.Request(url, data=data, headers=headers, method=method)
        except ValueError:
            return Unavailable(f"`{self._where}.url` must be an http or https url")

    def _open(self) -> urllib.request.OpenerDirector:
        if self._opener is None:
            if self._connection is None:  # request() answers before a request is built
                raise ConnectionError(self._block_reason)
            opener = urllib_opener(self._connection)
            opener.add_handler(_RefuseRedirects())
            self._opener = opener
        return self._opener

    def _exchange(self, request: urllib.request.Request, bound: float) -> Any:
        """Runs in a worker thread, abandoned by the caller once *bound* has passed.

        The read loop ends the thread at *bound* while a body arrives. Dripped response
        headers or a slow name lookup can keep it longer, in asyncio's shared executor.
        """
        try:
            opener = self._open()
        except UnicodeError:
            return Unavailable(self._unsendable_reason())
        except (ValueError, ConnectionError) as e:
            return Unavailable(str(e))
        ends_at = time.monotonic() + bound
        try:
            try:
                response = opener.open(request, timeout=bound)
            except urllib.error.HTTPError as e:
                response = e
            with response:
                body = _read_until(response, ends_at)
                if isinstance(body, Unavailable):
                    return body
                content_type = response.headers.get("Content-Type", "")
                return _read_answer(response.status, content_type, body)
        except http.client.IncompleteRead:
            return Unavailable(_CUT_SHORT_REASON)
        except (ValueError, http.client.InvalidURL):
            # What these say can quote the login header or the url, so none of it is repeated.
            return Unavailable(self._unsendable_reason())
        except (OSError, http.client.HTTPException) as e:
            if isinstance(e, TimeoutError) or isinstance(getattr(e, "reason", None), TimeoutError):
                return Unavailable(f"no answer within {bound:g} s")
            return Unavailable(f"could not reach the service: {getattr(e, 'reason', e)}")

    def _unsendable_reason(self) -> str:
        return f"`{self._where}`: the url or login cannot be sent"


def _read_until(response: Any, ends_at: float) -> bytes | Unavailable:
    # A socket timeout restarts with every byte, so a slow drip would outlast it.
    too_large = Unavailable(f"answered more than {_MAX_BODY_BYTES // (1024 * 1024)} MiB")
    if response.length is not None and response.length > _MAX_BODY_BYTES:
        return too_large
    chunks = []
    received = 0
    while chunk := response.read1(_READ_CHUNK_BYTES):
        received += len(chunk)
        if received > _MAX_BODY_BYTES:
            return too_large
        chunks.append(chunk)
        if time.monotonic() > ends_at:
            raise TimeoutError
    # read1 reports an early close as an empty read, leaving the declared length unmet.
    if response.length:
        return Unavailable(_CUT_SHORT_REASON)
    return b"".join(chunks)


def _read_answer(status: int, content_type: str, body: bytes) -> Any:
    html = "html" in content_type.lower() or _is_html(body)
    if 300 <= status < 400 or (html and status in _LOGIN_PAGE_STATUSES):
        return Unavailable(LOGIN_PAGE_REASON)
    if html or not 200 <= status < 300:
        return Unavailable(f"answered HTTP {status}")
    try:
        return json.loads(body)
    except (ValueError, RecursionError):
        return Unavailable(_MALFORMED_REASON)


def _is_html(body: bytes) -> bool:
    start = body[:_HTML_SNIFF_BYTES].lstrip().lower()
    return start.startswith(b"<") and (b"<html" in start or b"<!doctype html" in start)
