"""Python client for the qmd search sidecar.

qmd exposes **no REST query endpoint**. Its HTTP transport speaks MCP Streamable
HTTP on ``POST /mcp`` plus a plain ``GET /health``, and — despite the transport
being documented as stateless — a session handshake is mandatory: the
``initialize`` response carries an ``Mcp-Session-Id`` header that every later
request must echo, and a request without one is refused with ``HTTP 400 Missing
session ID``. This module owns that protocol so its callers never see it.

Three properties of the client are deliberate rather than incidental:

* **Unconfigured is a state, not a degradation.** A deployment with no qmd
  sidecar is a supported configuration. A client built with ``None`` reports
  itself unavailable without ever opening a socket, and says nothing about it.
* **A configured-but-down sidecar is worth exactly one log line.** Availability
  is tracked as a state; a warning is emitted when the state *changes*, not on
  every call, so a panel polling a stopped sidecar cannot flood the log.
* **No subprocess management anywhere.** The daemon's lifecycle belongs to the
  container that runs it. This client only ever talks to one.

Typical use, with the caller gating on availability and taking its own fallback
path when the sidecar is absent::

    client = QMDClient(resolve_qmd_service_config(config))
    if client.is_available():
        results = client.query("okf", "quadrupole calibration", limit=5)
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from osprey.deployment.qmd_service import QMDServiceConfig

logger = logging.getLogger("osprey.services.qmd")

#: MCP protocol revision the sidecar's qmd version speaks.
PROTOCOL_VERSION = "2025-06-18"

#: Seconds a ``GET /health`` verdict is reused before the endpoint is probed
#: again. Short enough that a restarted sidecar is picked up promptly, long
#: enough that a render loop asking "is search on?" per keystroke costs one
#: request rather than hundreds.
HEALTH_TTL_SECONDS = 5.0

#: Per-request timeout. Generous because a reranked query over a large corpus
#: is measured in seconds, not milliseconds.
DEFAULT_TIMEOUT_SECONDS = 30.0

#: Sub-query kinds qmd accepts: BM25 keywords, semantic vector, and a
#: hypothetical-answer passage.
SearchType = Literal["lex", "vec", "hyde"]

#: Default hybrid: keyword first (qmd weights the first sub-query 2x), semantic
#: second. Callers that want one or the other pass ``search_types`` explicitly.
DEFAULT_SEARCH_TYPES: tuple[SearchType, ...] = ("lex", "vec")


class QMDClientError(RuntimeError):
    """The sidecar was reached but could not answer the request."""


class QMDUnavailableError(QMDClientError):
    """The sidecar is unconfigured or unreachable, so no answer is possible."""


class _SessionExpiredError(QMDClientError):
    """The daemon rejected our session id; a fresh handshake is needed."""


@dataclass(frozen=True)
class QMDSearchResult:
    """One ranked hit from a qmd query.

    Attributes:
        docid: qmd's opaque document identifier, e.g. ``#de06c5``.
        file: Path of the matching document **relative to its collection root**
            — qmd reports it prefixed with the collection name, and that prefix
            is stripped here so callers can percent-decode the remainder back
            into an ``entry_id`` / ``concept_id`` without knowing it was there.
        collection: The collection name that prefixed ``file``, or ``""`` when
            the daemon reported an unprefixed path.
        title: The document's first heading, as indexed.
        score: Relevance in 0-1. This is an **ordering signal, not a calibrated
            relevance probability**: with ``rerank=False`` it is derived from
            rank alone (1, 0.5, 0.33 …) and carries no information beyond the
            order it already expresses; with ``rerank=True`` it comes from the
            reranker model and is comparable only within one result set. Do not
            threshold it against a fixed number and do not compare it across
            queries.
        line: 1-indexed best-matching line, suitable for expanding context.
        snippet: Line-numbered, match-centered excerpt. Prefer it over
            re-deriving an excerpt from the document body.
        context: Surrounding-section text when the daemon supplies it, else
            ``None``.
    """

    docid: str
    file: str
    collection: str
    title: str
    score: float
    line: int
    snippet: str
    context: str | None = None


@dataclass(frozen=True)
class QMDResponse:
    """A decoded HTTP reply, including non-2xx ones.

    Attributes:
        status: HTTP status code.
        body: Response body decoded as UTF-8.
        headers: Response headers, keys lowercased.
    """

    status: int
    body: str
    headers: Mapping[str, str] = field(default_factory=dict)


class QMDTransport(Protocol):
    """The HTTP seam the client talks through.

    Exists so tests can drive the full protocol — handshake, session reuse,
    session expiry, tool errors — against an in-process stub, without a daemon
    and without patching the standard library.
    """

    def post(
        self, url: str, body: bytes, headers: Mapping[str, str], timeout: float
    ) -> QMDResponse:
        """POST ``body`` and return the reply, including error statuses.

        Raises:
            QMDUnavailableError: If the host could not be reached at all.
        """
        ...

    def get(self, url: str, timeout: float) -> QMDResponse:
        """GET ``url`` and return the reply, including error statuses.

        Raises:
            QMDUnavailableError: If the host could not be reached at all.
        """
        ...


class _UrllibTransport:
    """The shipped transport: the standard library, and nothing else.

    urllib is enough for two request shapes against a loopback endpoint, and
    keeping it means the search client adds no dependency to any deployment
    that never turns the sidecar on.
    """

    def post(
        self, url: str, body: bytes, headers: Mapping[str, str], timeout: float
    ) -> QMDResponse:
        """POST ``body`` to ``url``.

        Args:
            url: Absolute request URL.
            body: Encoded request body.
            headers: Request headers.
            timeout: Per-request timeout in seconds.

        Returns:
            The decoded reply. Non-2xx statuses are returned, not raised: the
            client reads them to tell a dead session apart from a dead daemon.

        Raises:
            QMDUnavailableError: If the endpoint could not be reached.
        """
        request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
        return self._send(request, url, timeout)

    def get(self, url: str, timeout: float) -> QMDResponse:
        """GET ``url``.

        Args:
            url: Absolute request URL.
            timeout: Per-request timeout in seconds.

        Returns:
            The decoded reply.

        Raises:
            QMDUnavailableError: If the endpoint could not be reached.
        """
        return self._send(urllib.request.Request(url, method="GET"), url, timeout)

    @staticmethod
    def _send(request: urllib.request.Request, url: str, timeout: float) -> QMDResponse:
        """Perform one request, mapping unreachability to an availability fault.

        Args:
            request: The prepared request.
            url: The URL, for the error message.
            timeout: Per-request timeout in seconds.

        Returns:
            The decoded reply.

        Raises:
            QMDUnavailableError: If the endpoint could not be reached.
        """
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return QMDResponse(
                    status=int(response.status),
                    body=response.read().decode("utf-8", errors="replace"),
                    headers={k.lower(): v for k, v in response.headers.items()},
                )
        except urllib.error.HTTPError as exc:
            return QMDResponse(
                status=int(exc.code),
                body=exc.read().decode("utf-8", errors="replace"),
                headers={k.lower(): v for k, v in (exc.headers or {}).items()},
            )
        except OSError as exc:
            raise QMDUnavailableError(f"qmd sidecar unreachable at {url}: {exc}") from exc


class QMDClient:
    """Queries one qmd sidecar, or none at all.

    Args:
        config: Resolved ``services.qmd`` settings, or ``None`` when the
            deployment has no sidecar. ``None`` is a first-class argument: the
            resulting client is permanently unavailable, makes no requests, and
            logs nothing.
        timeout: Per-request timeout in seconds.
        transport: HTTP seam. Defaults to a standard-library implementation;
            tests pass a stub.
        client_name: Name reported to the daemon during the MCP handshake.
    """

    def __init__(
        self,
        config: QMDServiceConfig | None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: QMDTransport | None = None,
        client_name: str = "osprey",
    ) -> None:
        self._config = config
        self._timeout = timeout
        self._transport: QMDTransport = transport if transport is not None else _UrllibTransport()
        self._client_name = client_name

        self._lock = threading.RLock()
        self._session_id: str | None = None
        self._next_request_id = 0

        # None means "not yet known", which is distinct from "known down" — the
        # first failure of a client that has never succeeded is still a state
        # change worth one warning.
        self._available: bool | None = None
        self._health_checked_at: float | None = None

    # -- configuration -----------------------------------------------------

    @property
    def is_configured(self) -> bool:
        """Whether this deployment has a qmd sidecar at all."""
        return self._config is not None

    @property
    def base_url(self) -> str | None:
        """The sidecar's base URL, or ``None`` when unconfigured."""
        return self._config.base_url if self._config is not None else None

    # -- availability ------------------------------------------------------

    def is_available(self) -> bool:
        """Report whether the sidecar can currently answer queries.

        Backed by ``GET /health`` and cached for :data:`HEALTH_TTL_SECONDS`, so
        a caller may ask per query without turning the health endpoint into the
        hot path. An unconfigured client answers ``False`` immediately and
        performs no HTTP request.

        Returns:
            ``True`` when the sidecar reported healthy within the TTL window.
        """
        if self._config is None:
            return False

        now = time.monotonic()
        with self._lock:
            cached = self._available
            checked_at = self._health_checked_at
        if (
            cached is not None
            and checked_at is not None
            and (now - checked_at) < HEALTH_TTL_SECONDS
        ):
            return cached

        try:
            response = self._transport.get(f"{self._config.base_url}/health", self._timeout)
        except QMDUnavailableError as exc:
            self._record_availability(False, str(exc))
            return False

        healthy = response.status == 200 and _health_is_ok(response.body)
        detail = "" if healthy else f"HTTP {response.status}: {response.body[:200]}"
        self._record_availability(healthy, detail)
        return healthy

    def _record_availability(self, available: bool, detail: str) -> None:
        """Store the availability verdict, logging only on a state change.

        Args:
            available: The verdict just observed.
            detail: Human-readable reason, logged when the state turns bad.
        """
        with self._lock:
            previous = self._available
            self._available = available
            self._health_checked_at = time.monotonic()
            if not available:
                # A dead session is worthless once the daemon is in doubt.
                self._session_id = None
        if previous == available:
            return
        if available:
            logger.info("qmd sidecar at %s is available", self.base_url)
        else:
            logger.warning("qmd sidecar at %s is unavailable: %s", self.base_url, detail)

    def _invalidate_health(self) -> None:
        """Force the next :meth:`is_available` call to re-probe the endpoint."""
        with self._lock:
            self._health_checked_at = None

    # -- queries -----------------------------------------------------------

    def query(
        self,
        collection: str | None,
        text: str,
        *,
        limit: int = 10,
        search_types: Sequence[SearchType] = DEFAULT_SEARCH_TYPES,
        min_score: float = 0.0,
        candidate_limit: int | None = None,
        rerank: bool | None = None,
        intent: str | None = None,
    ) -> list[QMDSearchResult]:
        """Run one search and return its ranked hits.

        Args:
            collection: Collection to restrict the search to — this is how a
                single sidecar serves several corpora side by side. ``None``
                searches every collection the daemon has indexed.
            text: The query text, applied to each kind in ``search_types``.
            limit: Maximum number of hits to return.
            search_types: Sub-query kinds to run, in order. qmd weights the
                first entry twice as heavily as the rest, so order matters: the
                default runs keywords first, semantics second.
            min_score: Relevance floor in 0-1. Read
                :attr:`QMDSearchResult.score` before setting this above 0 — the
                scale is not calibrated.
            candidate_limit: Maximum candidates the reranker considers (qmd's
                ``candidateLimit``, default 40). Lowering it trades recall for
                latency. ``None`` leaves qmd's default in place.
            rerank: Whether to rerank with the LLM reranker. qmd defaults this
                to ``True``, and reranking dominates query latency — roughly 4x
                — so an interactive caller will usually pass ``False`` and an
                agent-facing caller ``True``. ``None`` leaves qmd's default in
                place; the choice belongs to the caller's config, not here.
            intent: Background context disambiguating the query. It does not
                search on its own.

        Returns:
            Hits in descending relevance order, at most ``limit`` of them. An
            empty list means the search matched nothing — including when
            ``collection`` names a collection the daemon does not have.

        Raises:
            QMDUnavailableError: If the client is unconfigured or the sidecar
                cannot be reached.
            QMDClientError: If the daemon answered with an error or a response
                this client cannot parse.
        """
        if self._config is None:
            raise QMDUnavailableError("no qmd sidecar is configured for this deployment")
        if not search_types:
            raise ValueError("search_types must name at least one sub-query kind")

        arguments: dict[str, Any] = {
            "searches": [{"type": kind, "query": text} for kind in search_types],
            "limit": limit,
            "minScore": min_score,
        }
        if collection is not None:
            arguments["collections"] = [collection]
        if candidate_limit is not None:
            arguments["candidateLimit"] = candidate_limit
        if rerank is not None:
            arguments["rerank"] = rerank
        if intent:
            arguments["intent"] = intent

        result = self._call_tool("query", arguments)
        return _parse_results(result)

    # -- MCP protocol ------------------------------------------------------

    def _call_tool(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """Invoke one MCP tool, re-handshaking once if the session expired.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            The ``tools/call`` result payload.

        Raises:
            QMDUnavailableError: If the sidecar could not be reached.
            QMDClientError: If the daemon reported a tool or protocol error.
        """
        params = {"name": name, "arguments": dict(arguments)}
        self._ensure_session()
        try:
            result = self._rpc("tools/call", params)
        except _SessionExpiredError:
            # The daemon restarted, or evicted us. One retry, from a clean
            # handshake; a second failure is a real fault.
            with self._lock:
                self._session_id = None
            self._ensure_session()
            try:
                result = self._rpc("tools/call", params)
            except _SessionExpiredError as exc:
                raise QMDClientError(f"qmd rejected the session twice for {name!r}") from exc
        if result.get("isError"):
            raise QMDClientError(f"qmd tool {name!r} failed: {_error_text(result)}")
        return result

    def _ensure_session(self) -> None:
        """Establish an MCP session if this client does not hold one.

        Raises:
            QMDUnavailableError: If the sidecar could not be reached.
            QMDClientError: If the handshake was refused.
        """
        with self._lock:
            if self._session_id is not None:
                return
            self._rpc(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": self._client_name, "version": "1"},
                },
            )
            if self._session_id is None:
                raise QMDClientError("qmd initialize returned no Mcp-Session-Id header")
            self._notify("notifications/initialized")

    def _rpc(self, method: str, params: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        """Send one JSON-RPC request and return its result payload.

        Args:
            method: JSON-RPC method name.
            params: Method parameters, if any.

        Returns:
            The ``result`` member of the reply.

        Raises:
            QMDUnavailableError: If the sidecar could not be reached.
            _SessionExpiredError: If the daemon rejected the session id.
            QMDClientError: On any other protocol or transport-level error.
        """
        with self._lock:
            self._next_request_id += 1
            request_id = self._next_request_id
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = dict(params)

        response = self._post(payload)
        message = _decode_jsonrpc(response.body)
        if message is None:
            raise QMDClientError(f"qmd returned an empty response to {method!r}")
        if "error" in message:
            raise QMDClientError(f"qmd error on {method!r}: {message['error']}")
        result = message.get("result")
        if not isinstance(result, Mapping):
            raise QMDClientError(f"qmd returned no result for {method!r}")
        return result

    def _notify(self, method: str) -> None:
        """Send one JSON-RPC notification, which carries no reply.

        Args:
            method: Notification method name.
        """
        self._post({"jsonrpc": "2.0", "method": method})

    def _post(self, payload: Mapping[str, Any]) -> QMDResponse:
        """POST one JSON-RPC message to ``/mcp`` and classify the reply status.

        Args:
            payload: The JSON-RPC message.

        Returns:
            The successful reply.

        Raises:
            QMDUnavailableError: If the sidecar could not be reached.
            _SessionExpiredError: On the statuses qmd uses for a missing or
                unknown session.
            QMDClientError: On any other non-2xx status.
        """
        assert self._config is not None  # guarded by every public entry point
        headers = {
            "Content-Type": "application/json",
            # The daemon negotiates between plain JSON and SSE framing.
            "Accept": "application/json, text/event-stream",
        }
        with self._lock:
            session_id = self._session_id
        if session_id:
            headers["Mcp-Session-Id"] = session_id

        url = f"{self._config.base_url}/mcp"
        try:
            response = self._transport.post(
                url, json.dumps(dict(payload)).encode("utf-8"), headers, self._timeout
            )
        except QMDUnavailableError as exc:
            self._record_availability(False, str(exc))
            raise exc

        if response.status in (400, 404) and "session" in response.body.lower():
            raise _SessionExpiredError(response.body[:200])
        if response.status >= 400:
            self._invalidate_health()
            raise QMDClientError(f"qmd returned HTTP {response.status}: {response.body[:200]}")

        with self._lock:
            if self._session_id is None:
                new_session = response.headers.get("mcp-session-id")
                if new_session:
                    self._session_id = new_session
        return response


def _health_is_ok(body: str) -> bool:
    """Decide whether a ``/health`` body reports a healthy daemon.

    Args:
        body: The raw response body.

    Returns:
        ``True`` when the payload is JSON carrying ``status: "ok"``.
    """
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, Mapping) and payload.get("status") == "ok"


def _decode_jsonrpc(raw: str) -> Mapping[str, Any] | None:
    """Decode a JSON-RPC reply sent as plain JSON or in SSE framing.

    Args:
        raw: The response body.

    Returns:
        The decoded message, or ``None`` when the body carried no data — which
        is the normal reply to a notification.

    Raises:
        QMDClientError: If the body carried data that is not valid JSON.
    """
    text = raw.strip()
    if not text:
        return None
    candidates = [text] if text.startswith("{") else _sse_data_lines(text)
    for candidate in candidates:
        try:
            message = json.loads(candidate)
        except ValueError as exc:
            raise QMDClientError(f"qmd returned undecodable JSON: {candidate[:200]}") from exc
        if isinstance(message, Mapping):
            return message
    return None


def _sse_data_lines(text: str) -> list[str]:
    """Extract the payloads of ``data:`` lines from an SSE-framed body.

    Args:
        text: The response body.

    Returns:
        Each non-empty ``data:`` payload, in order.
    """
    payloads = []
    for line in text.splitlines():
        if line.startswith("data:"):
            chunk = line[len("data:") :].strip()
            if chunk:
                payloads.append(chunk)
    return payloads


def _error_text(result: Mapping[str, Any]) -> str:
    """Render an MCP tool error result as one line of text.

    Args:
        result: The ``tools/call`` result payload.

    Returns:
        The concatenated text blocks, or the raw payload when it has none.
    """
    content = result.get("content")
    if isinstance(content, Sequence) and not isinstance(content, str | bytes):
        texts = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, Mapping) and block.get("type") == "text"
        ]
        joined = " ".join(part for part in texts if part)
        if joined:
            return joined
    return str(result)[:200]


def _parse_results(result: Mapping[str, Any]) -> list[QMDSearchResult]:
    """Convert a ``query`` tool result into typed hits.

    Reads ``structuredContent.results`` and ignores ``content``, which is the
    same data rendered for a human reader.

    Args:
        result: The ``tools/call`` result payload.

    Returns:
        The parsed hits, in the order the daemon returned them.

    Raises:
        QMDClientError: If the payload carries no structured results.
    """
    structured = result.get("structuredContent")
    if not isinstance(structured, Mapping):
        raise QMDClientError("qmd query response carried no structuredContent")
    rows = structured.get("results")
    if rows is None:
        raise QMDClientError("qmd query response carried no results field")
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
        raise QMDClientError(f"qmd query results were not a list: {type(rows).__name__}")

    hits = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise QMDClientError(f"qmd query returned a non-object result: {row!r}")
        collection, _, relative_path = str(row.get("file", "")).partition("/")
        hits.append(
            QMDSearchResult(
                docid=str(row.get("docid", "")),
                file=relative_path or collection,
                collection=collection if relative_path else "",
                title=str(row.get("title", "")),
                score=_as_float(row.get("score")),
                line=_as_int(row.get("line")),
                snippet=str(row.get("snippet") or ""),
                context=None if row.get("context") is None else str(row["context"]),
            )
        )
    return hits


def _as_float(value: Any) -> float:
    """Coerce a JSON number to ``float``, defaulting to 0.0.

    Args:
        value: The raw field value.

    Returns:
        The value as a float, or 0.0 when it is absent or non-numeric.
    """
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _as_int(value: Any) -> int:
    """Coerce a JSON number to ``int``, defaulting to 1.

    Args:
        value: The raw field value.

    Returns:
        The value as an int, or 1 — the first line — when it is absent or
        non-numeric.
    """
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else 1
