"""Connection lifecycle and the one read path for the graph MCP server.

Every graph tool reads the store through :meth:`GraphContext.run_read`, and that
is deliberate. The store is one bolt endpoint behind one credential, and a tool
that dialed it itself would have to re-answer the same four questions each time:
is a store configured at all, how long may a query run, how many rows may come
back, and what does an operator do about each way the dial can fail. Answering
them once here is what makes every tool's error identical for the same cause,
and it is what keeps the driver — imported lazily, cached, closed once — from
being constructed per call.

Three states, not two
---------------------
"Configured" is decided at ``initialize()`` from the ``services.graphdb`` block
and never re-derived per call, so a project with no graph store answers every
tool immediately rather than dialing an address it does not have. A block that
is present but malformed (``port_host: "abc"``) counts as *not* configured — the
same reading :func:`~osprey.deployment.graphdb_service.resolve_graphdb_service_config`
refuses at deploy time — but it is logged at warning level rather than swallowed,
because an operator who typed the key meant to have a store and would otherwise
see only "not configured" for what is really a typo. The third state, configured
but *empty*, is not a failure at all: :meth:`GraphContext.is_empty` answers it so
the caller can name ``osprey knowledge seed-graph`` instead of returning zero
rows that read as "the data is wrong".

Failing fast
------------
An unreachable store must surface as one error in seconds, not as a tool call an
agent waits on. ``connection_timeout=3.0`` caps the TCP dial and
``max_transaction_retry_time=0.0`` turns off the driver's retry loop, so the
whole path from ``run_read`` to :class:`GraphUnreachable` is bounded well inside
the eight seconds the proposal allows. Retrying a store that is down would only
multiply the wait: the remedy is ``osprey up``, not another attempt.

Errors carry their own remedy
-----------------------------
Each :class:`GraphStoreError` subclass fixes an ``error_type`` and the operator
suggestions that go with it, so the tool layer converts one to an error envelope
without deciding what to advise. Those ``error_type`` values are drawn from the
PostToolUse error-guidance hook's ``ERROR_CLASS_MAP`` rather than invented here —
see :class:`GraphStoreError` for what goes wrong when they are not. The auth
remedy names ``GRAPHDB_PASSWORD`` and
the project ``.env`` and never the value; nothing here logs or formats the
password, and :class:`~osprey.deployment.graphdb_service.GraphdbConnection`
keeps it out of ``repr`` for the case where one lands in a traceback anyway.

**neo4j is imported lazily**, inside the driver factory and inside
:meth:`GraphContext.run_read`, never at module top — importing this module must
not pull the driver into a runtime that has no graph store, which
``tests/services/facility_knowledge/test_import_isolation.py`` guards.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from osprey.deployment.graphdb_service import (
    GRAPHDB_PASSWORD_ENV,
    GRAPHDB_PORT_CONFIG_KEY,
    GRAPHDB_SEED_COMMAND,
    GRAPHDB_SERVICE_NAME,
    GraphdbConnection,
    resolve_graphdb_connection,
    resolve_graphdb_service_config,
)
from osprey.port_layout import resolve_port_base
from osprey.utils.config import get_config_value
from osprey.utils.workspace import load_osprey_config

logger = logging.getLogger("osprey.mcp_server.graph.server_context")

# ---------------------------------------------------------------------------
# Config keys and their built-in defaults
# ---------------------------------------------------------------------------

#: Seconds a single read query may run on the server before Neo4j terminates it.
#: Enforced server-side through ``neo4j.unit_of_work`` rather than by a client
#: -side clock, so a runaway traversal stops consuming the store's resources
#: instead of merely being abandoned by the caller.
QUERY_TIMEOUT_CONFIG_KEY = "services.graphdb.query_timeout_s"

#: Rows a single read may return. A cap rather than an agent-supplied limit
#: because the caller is a language model writing Cypher: an unbounded ``MATCH``
#: over a facility graph answers with tens of thousands of rows, and the cost of
#: that lands in the agent's context rather than in the store.
QUERY_MAX_ROWS_CONFIG_KEY = "services.graphdb.query_max_rows"

#: Fifteen seconds: long enough for a multi-hop traversal over a facility-sized
#: corpus, short enough that a query an agent should have narrowed comes back as
#: a timeout it can act on within one turn.
DEFAULT_QUERY_TIMEOUT_S = 15

#: Two hundred rows. Above roughly this the answer stops being something an agent
#: reasons over and becomes something it summarises, which is a sign the query
#: wanted aggregation — so the cap is set where truncation is useful feedback.
DEFAULT_QUERY_MAX_ROWS = 200

#: Seconds the driver may spend opening a TCP connection. Not configurable: it
#: exists to bound the unreachable case, and an operator who raised it would be
#: trading the one property that makes a down store legible for nothing.
CONNECTION_TIMEOUT_S = 3.0

#: Database named explicitly on every session so the driver skips the
#: home-database lookup round-trip. Matches the seeder's ``DEFAULT_DATABASE``.
DATABASE = "neo4j"

#: Counts the corpus rather than the store's own bookkeeping: ``:Resource`` is
#: the label neosemantics gives every imported RDF subject, and a bootstrapped
#: store carries ``_GraphConfig`` and namespace nodes whether or not a corpus was
#: ever imported.
RESOURCE_COUNT_CYPHER = "MATCH (r:Resource) RETURN count(r) AS n"

#: Message of every :class:`GraphNotConfigured` this module raises. Two code
#: paths can raise it — the read itself and the driver factory behind it — and
#: the agent should see the same sentence from both.
_NOT_CONFIGURED_MESSAGE = (
    "This deployment has no graph store configured, so there is nothing to query."
)

#: Server-side refusal for a write attempted inside a read transaction. The
#: gate refuses writes before dialing, so seeing this means a mutating construct
#: got past it — worth naming distinctly rather than folding into query errors.
_READ_ONLY_CODE = "Neo.ClientError.Statement.AccessMode"

#: The two codes Neo4j uses for a transaction it terminated on the timeout.
_TIMEOUT_CODES = frozenset(
    {
        "Neo.ClientError.Transaction.TransactionTimedOut",
        "Neo.TransientError.Transaction.TransactionTimedOutClientConfiguration",
    }
)

#: Substrings that identify a terminated transaction on a server whose code for
#: it we do not recognise. Checked only after the codes above, and only for a
#: client/transient error, so a query whose *data* mentions "timed out" cannot be
#: mistaken for one.
_TIMEOUT_MESSAGE_MARKERS = ("timed out", "terminated")

#: Sequence types the JSON-safety pass recurses into. Matched on the *exact*
#: type rather than with ``isinstance``, because two of the driver values this
#: pass exists to convert — ``neo4j.time.Duration`` and ``neo4j.spatial.Point``
#: — are themselves tuple subclasses. Recursing into those would render a point
#: as a bare pair of numbers with its coordinate system silently dropped.
_CONTAINER_TYPES = (list, tuple, set, frozenset)


# ---------------------------------------------------------------------------
# Error family
# ---------------------------------------------------------------------------


class GraphStoreError(Exception):
    """A graph read failed, with the operator remedy for that cause attached.

    The tool layer turns one of these into an error envelope verbatim:
    :attr:`error_type` becomes the envelope's type, :attr:`suggestions` its
    remedies, :attr:`details` its ``details``, and ``str(exc)`` its message.
    Subclasses exist so that decision is made once, next to the mapping that
    knows *why* the read failed, rather than at each of the tools that catch it.

    **:attr:`error_type` values are not free-form.** The PostToolUse
    error-guidance hook classes each envelope by looking its ``error_type`` up in
    ``ERROR_CLASS_MAP``
    (``src/osprey/templates/claude_code/claude/hooks/osprey_error_guidance.py``),
    and a value that is not in that table falls through to "Internal" — which
    tells the agent to stop and fetch an operator, exactly the wrong advice for a
    query it could have narrowed and retried. So a new subclass takes an
    ``error_type`` the hook already knows and distinguishes itself in
    :attr:`details` instead, which is why two subclasses below share
    ``validation_error``.
    """

    #: Envelope ``error_type`` for this cause; must be a key of the hook's
    #: ``ERROR_CLASS_MAP``. Overridden by every subclass.
    error_type: str = "internal_error"

    #: ``details["kind"]`` this subclass stamps on every instance, or ``None``
    #: when its ``error_type`` already identifies it uniquely. This is what keeps
    #: two causes that must share one ``error_type`` distinguishable downstream
    #: without the caller having to remember to spell the kind.
    details_kind: str | None = None

    def __init__(
        self,
        message: str,
        suggestions: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Build the error.

        Args:
            message: Operator-facing description of what failed.
            suggestions: Remedies to attach, or ``None`` to use the subclass's
                own — which is the normal path; an explicit list is for a caller
                that knows something more specific about the context.
            details: Structured context for the envelope's ``details`` field —
                where the store's own status code travels. The subclass's
                :attr:`details_kind` is merged in, so a caller never has to
                remember it and cannot spell it differently.
        """
        super().__init__(message)
        self.suggestions: list[str] = (
            list(suggestions) if suggestions is not None else self.default_suggestions()
        )
        merged = dict(details) if details else {}
        if self.details_kind is not None:
            merged.setdefault("kind", self.details_kind)
        self.details: dict[str, Any] | None = merged or None

    @staticmethod
    def default_suggestions() -> list[str]:
        """Return the remedies for this cause."""
        return []


class GraphNotConfigured(GraphStoreError):
    """No graph store is described by this deployment's config."""

    #: Classes as "Internal" in the hook's table — the right reading here: no
    #: wording of the query fixes a missing config block, and an operator has
    #: to edit config.yml.
    error_type = "not_configured"

    @staticmethod
    def default_suggestions() -> list[str]:
        """Point at the block that declares a store, since none was found."""
        return [
            f"Add a 'services.{GRAPHDB_SERVICE_NAME}' block to config.yml to describe the "
            "graph store this deployment connects to.",
            f"If the store should be deployed here, also list '{GRAPHDB_SERVICE_NAME}' in "
            "deployed_services and run `osprey up`.",
        ]


class GraphUnreachable(GraphStoreError):
    """The store did not answer — down, not started, or at another address."""

    error_type = "service_unavailable"

    @staticmethod
    def default_suggestions() -> list[str]:
        """Name the service to start and the two keys that fix a wrong address."""
        return [
            f"The graph store runs as the '{GRAPHDB_SERVICE_NAME}' compose service — check "
            "that it is up, and start it with `osprey up` if it is not.",
            f"If it is running, verify {GRAPHDB_PORT_CONFIG_KEY} (or services.graphdb.uri "
            "for an external store) names the address it is actually published on.",
        ]


class GraphAuthFailed(GraphStoreError):
    """The store rejected the credential."""

    #: "Connection" class rather than a security-flavoured one: the store was
    #: reached and the fix is a credential the deployment holds, not a refusal
    #: the agent should escalate.
    error_type = "connection_error"

    @staticmethod
    def default_suggestions() -> list[str]:
        """Name the variable and the file, never the value."""
        return [
            f"Both ends read {GRAPHDB_PASSWORD_ENV}: the store keeps the value the project "
            ".env carried when its data volume was first created.",
            f"Restore that value in the project .env, or reset the store's data volume so "
            f"it is created again with the current {GRAPHDB_PASSWORD_ENV}.",
        ]


class GraphQueryTimeout(GraphStoreError):
    """The server terminated the query for exceeding the configured timeout."""

    error_type = "timeout_error"

    @staticmethod
    def default_suggestions() -> list[str]:
        """Steer toward a narrower query before steering toward a longer budget."""
        return [
            "Narrow the query — add a label or property filter, bound the traversal depth, "
            "or aggregate instead of returning every path.",
            f"If the query is genuinely long-running, raise {QUERY_TIMEOUT_CONFIG_KEY} in "
            "config.yml.",
        ]


class GraphReadOnlyViolation(GraphStoreError):
    """The query tried to write, and the read transaction refused it."""

    error_type = "validation_error"
    details_kind = "read_only"

    @staticmethod
    def default_suggestions() -> list[str]:
        """State the posture; there is no setting that relaxes it."""
        return [
            "This server reads the graph and never writes to it — rewrite the query using "
            "MATCH / RETURN only.",
            f"The graph is rebuilt from its TTL corpus, so changes belong in that file "
            f"followed by `{GRAPHDB_SEED_COMMAND}`.",
        ]


class GraphQueryError(GraphStoreError):
    """The store rejected the query — a syntax error, an unknown name, and so on."""

    error_type = "validation_error"
    details_kind = "query_error"

    @staticmethod
    def default_suggestions() -> list[str]:
        """Send the caller to the schema rather than to a retry."""
        return [
            "Check the Cypher syntax and that every label, relationship type and property "
            "name exists — get_schema lists the ones this graph has.",
            "example_queries shows working queries for the shapes this corpus contains.",
        ]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryResult:
    """Rows from one read, and whether the row cap cut them short.

    Attributes:
        rows: Records as plain JSON-safe dicts, at most the effective cap.
        truncated: True when the store had more rows than the cap allowed. A
            flag rather than a silent slice: an agent that cannot tell a
            complete answer from a clipped one draws conclusions from the
            first two hundred rows of an unbounded match.
    """

    rows: list[dict[str, Any]]
    truncated: bool


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


class GraphContext:
    """Resolved graph-store settings and the single read path over them.

    Attributes:
        query_timeout_s: Server-side transaction timeout for every read.
        query_max_rows: Default row cap, overridable per call.
        configured: Whether this deployment describes a graph store at all. A
            malformed block reads as ``False`` here — see the module docstring.
    """

    def __init__(self) -> None:
        self.query_timeout_s: int = DEFAULT_QUERY_TIMEOUT_S
        self.query_max_rows: int = DEFAULT_QUERY_MAX_ROWS
        self.configured: bool = False
        self._connection: GraphdbConnection | None = None
        self._driver: Any = None
        self._initialized = False

    # -- lifecycle -----------------------------------------------------------

    def initialize(self) -> None:
        """Read config once and resolve the connection. Repeat calls are no-ops.

        Nothing is dialed here: the driver is created on the first read, so a
        server whose store is down still starts and still answers ``capabilities``
        and ``example_queries``.
        """
        if self._initialized:
            return

        config = self._load_config()
        self.query_timeout_s = _positive_int(
            self._setting(QUERY_TIMEOUT_CONFIG_KEY, DEFAULT_QUERY_TIMEOUT_S),
            DEFAULT_QUERY_TIMEOUT_S,
            QUERY_TIMEOUT_CONFIG_KEY,
        )
        self.query_max_rows = _positive_int(
            self._setting(QUERY_MAX_ROWS_CONFIG_KEY, DEFAULT_QUERY_MAX_ROWS),
            DEFAULT_QUERY_MAX_ROWS,
            QUERY_MAX_ROWS_CONFIG_KEY,
        )
        self._resolve_connection(config)

        self._initialized = True
        logger.info(
            "GraphContext: initialized (configured=%s, uri=%s, timeout=%ss, max_rows=%s)",
            self.configured,
            self.uri,
            self.query_timeout_s,
            self.query_max_rows,
        )

    def shutdown(self) -> None:
        """Close the cached driver, if one was ever created."""
        if self._driver is None:
            return
        try:
            self._driver.close()
        except Exception:  # noqa: BLE001 - closing is best-effort teardown
            logger.debug("Error closing the graph driver (ignored)", exc_info=True)
        self._driver = None

    @property
    def uri(self) -> str | None:
        """Address reads are sent to, or ``None`` when no store is configured."""
        return self._connection.uri if self._connection is not None else None

    # -- reads ---------------------------------------------------------------

    def run_read(
        self,
        cypher: str,
        params: Mapping[str, Any] | None = None,
        *,
        max_rows: int | None = None,
    ) -> QueryResult:
        """Run one read query and return its rows.

        Args:
            cypher: The query to run. Already vetted by the caller — this method
                enforces read-*transaction* semantics, not query policy.
            params: Query parameters, or ``None``.
            max_rows: Row cap for this call, or ``None`` for
                :attr:`query_max_rows`. A value below one is raised to one: a
                zero-row answer would be indistinguishable from an empty graph.

        Returns:
            The rows, and whether the cap cut them short.

        Raises:
            GraphNotConfigured: No ``services.graphdb`` block describes a store.
            GraphUnreachable: The store did not answer.
            GraphAuthFailed: The store rejected the credential.
            GraphQueryTimeout: The server terminated the query on the timeout.
            GraphReadOnlyViolation: The query tried to write.
            GraphQueryError: The store rejected the query for any other reason.
        """
        if not self.configured:
            raise GraphNotConfigured(_NOT_CONFIGURED_MESSAGE)

        cap = self.query_max_rows if max_rows is None else max(1, int(max_rows))
        driver = self._driver_or_raise()

        from neo4j import unit_of_work

        @unit_of_work(timeout=self.query_timeout_s)
        def _read(tx: Any) -> list[dict[str, Any]]:
            """Stream at most ``cap + 1`` rows so truncation is detectable."""
            result = tx.run(cypher, dict(params or {}))
            rows: list[dict[str, Any]] = []
            for record in result:
                rows.append(_json_safe(record.data()))
                if len(rows) > cap:
                    break
            return rows

        try:
            with driver.session(database=DATABASE) as session:
                fetched = session.execute_read(_read)
        except Exception as exc:
            raise _map_driver_error(exc) from exc

        return QueryResult(rows=list(fetched[:cap]), truncated=len(fetched) > cap)

    def is_empty(self) -> bool:
        """Whether the graph holds no corpus.

        Returns:
            True when the store carries no ``:Resource`` nodes — bootstrapped
            but never seeded. Callers name :data:`GRAPHDB_SEED_COMMAND` when it
            is, which is why this is a separate question from "did the query
            match anything".

        Raises:
            GraphStoreError: Whatever :meth:`run_read` raises; an unreachable
                store is not an empty one.
        """
        result = self.run_read(RESOURCE_COUNT_CYPHER)
        if not result.rows:
            return True
        return int(result.rows[0].get("n") or 0) <= 0

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _load_config() -> Mapping[str, Any]:
        """Load the project config, degrading to an empty mapping.

        Returns:
            The parsed config, or ``{}`` when it could not be read — which
            leaves the context unconfigured rather than failing server startup
            over a file the graph tools are not the only reader of.
        """
        try:
            return load_osprey_config() or {}
        except Exception as exc:  # noqa: BLE001 - a bad config must not kill startup
            logger.warning("GraphContext: could not load config.yml (%s)", exc)
            return {}

    @staticmethod
    def _setting(key: str, default: int) -> Any:
        """Read one tuning key, degrading to *default*.

        Args:
            key: Dotted config key.
            default: Value to use when the key cannot be read at all.

        Returns:
            The configured value, or ``default``. ``get_config_value`` builds
            the project's ConfigBuilder on first use, so where
            :meth:`_load_config` degrades a *missing* config file this degrades
            the same file being missing one call later — without it the server
            would still die at startup over a config the graph tools are not
            the only reader of, and the agent would lose ``capabilities`` and
            ``example_queries`` along with the store it never had.
        """
        try:
            return get_config_value(key, default)
        except Exception as exc:  # noqa: BLE001 - a bad config must not kill startup
            logger.warning("GraphContext: could not read %s (%s)", key, exc)
            return default

    def _resolve_connection(self, config: Mapping[str, Any]) -> None:
        """Decide :attr:`configured` and resolve what to dial.

        Args:
            config: The parsed project config.
        """
        try:
            self.configured = resolve_graphdb_service_config(config) is not None
        except ValueError as exc:
            logger.warning(
                "GraphContext: the services.%s block is malformed, so graph tools will "
                "report no store is configured — %s",
                GRAPHDB_SERVICE_NAME,
                exc,
            )
            self.configured = False
            return

        if not self.configured:
            return

        block = (config.get("services") or {}).get(GRAPHDB_SERVICE_NAME)
        try:
            self._connection = resolve_graphdb_connection(block, base=resolve_port_base(config))
        except ValueError as exc:
            logger.warning(
                "GraphContext: the services.%s block does not resolve to an address to "
                "dial, so graph tools will report no store is configured — %s",
                GRAPHDB_SERVICE_NAME,
                exc,
            )
            self.configured = False

    def _driver_or_raise(self) -> Any:
        """Return the cached driver, creating it on first use.

        Returns:
            An open neo4j driver. Construction does not dial the store, so a
            store that is down surfaces on the query rather than here.

        Raises:
            GraphUnreachable: The driver could not be constructed — which for
                this path means the resolved address is not one it can use.
        """
        if self._driver is not None:
            return self._driver

        connection = self._connection
        if connection is None:
            raise GraphNotConfigured(_NOT_CONFIGURED_MESSAGE)

        from neo4j import GraphDatabase

        try:
            self._driver = GraphDatabase.driver(
                connection.uri,
                auth=(connection.username, connection.password),
                connection_timeout=CONNECTION_TIMEOUT_S,
                max_transaction_retry_time=0.0,
            )
        except Exception as exc:
            raise GraphUnreachable(
                f"Cannot open a connection to the graph store at {connection.uri}: {_describe(exc)}"
            ) from exc
        return self._driver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _positive_int(value: Any, default: int, key: str) -> int:
    """Coerce a config value to a positive integer, or fall back to ``default``.

    Args:
        value: The raw value read from config.
        default: Value to use when it is absent or unusable.
        key: Dotted config key, named in the warning.

    Returns:
        The value when it is a positive integer, otherwise ``default``. A bad
        value warns rather than raises: this runs at MCP-server startup, and a
        typo in a tuning key should not leave the agent with no graph tools at
        all. ``bool`` is rejected explicitly — it is an ``int`` subclass, so
        ``query_max_rows: true`` would otherwise resolve to a cap of one.
    """
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        logger.warning(
            "%s must be a positive integer, got %r — falling back to %s", key, value, default
        )
        return default
    return int(value)


def _describe(exc: Exception) -> str:
    """Render a driver exception as one operator-readable line.

    Args:
        exc: The exception to describe.

    Returns:
        The server's own message where there is one. ``str()`` on a modern
        ``Neo4jError`` renders the whole GQL diagnostic record, which buries the
        one sentence an operator needs, so ``.message`` is preferred — but only
        when ``str()`` actually contains it. A ``Neo4jError`` raised by the
        driver itself rather than hydrated from a server response carries a
        placeholder ``.message`` ("an unknown error occurred") while ``str()``
        holds the real text, and that case has to come out the other way round.
    """
    message = getattr(exc, "message", None)
    text = str(exc).strip()
    if isinstance(message, str) and message.strip() and message.strip() in text:
        return message.strip()
    return text or exc.__class__.__name__


def _json_safe(value: Any) -> Any:
    """Convert driver-native values into something JSON can carry.

    ``Record.data()`` already flattens nodes, relationships and paths into
    dicts and lists, but it leaves temporal and spatial values as driver
    objects: ``neo4j.time`` Date/Time/DateTime/Duration and
    ``neo4j.spatial.Point`` all reach here intact. Rendering them as strings
    keeps their meaning legible to an agent, which is what the value is for,
    and avoids a per-type table that would go stale the next time the driver
    grows a type.

    Args:
        value: Any value from a record.

    Returns:
        The value if JSON already carries it, a converted container if it holds
        one, otherwise its string form.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if type(value) in _CONTAINER_TYPES:
        return [_json_safe(item) for item in value]
    return str(value)


def _map_driver_error(exc: Exception) -> GraphStoreError:
    """Translate a driver exception into the error class carrying its remedy.

    Args:
        exc: The exception raised by the driver or the server.

    Returns:
        The mapped error. Anything unrecognised becomes
        :class:`GraphQueryError`, which is the honest reading: the store was
        reached and refused, and the caller is told so rather than being sent
        to check whether the service is running.
    """
    from neo4j.exceptions import (
        AuthError,
        ClientError,
        Neo4jError,
        ServiceUnavailable,
        SessionExpired,
        TransientError,
    )

    detail = _describe(exc)
    code = str(getattr(exc, "code", "") or "")

    # AuthError subclasses ClientError, so it is matched first or never.
    if isinstance(exc, AuthError):
        return GraphAuthFailed(f"The graph store rejected the credentials: {detail}")
    if isinstance(exc, (ServiceUnavailable, SessionExpired, OSError)):
        return GraphUnreachable(f"The graph store did not answer: {detail}")
    if isinstance(exc, (ClientError, TransientError)):
        if code == _READ_ONLY_CODE:
            return GraphReadOnlyViolation(
                f"The query attempted to modify the graph, which this server does not "
                f"permit: {detail}",
                details={"code": code},
            )
        if code in _TIMEOUT_CODES or _looks_like_timeout(detail):
            return GraphQueryTimeout(f"The graph store terminated the query: {detail}")
        return GraphQueryError(
            f"The graph store rejected the query: {detail}", details={"code": code}
        )
    if isinstance(exc, Neo4jError):
        return GraphQueryError(
            f"The graph store rejected the query: {detail}", details={"code": code}
        )
    return GraphQueryError(f"The graph read failed: {detail}", details={"code": code})


def _looks_like_timeout(detail: str) -> bool:
    """Whether a client/transient error message reads as a terminated transaction.

    Args:
        detail: The server's message.

    Returns:
        True when it names a timeout or a termination. Consulted only after the
        known codes, and only for errors the server already classified as
        client or transient, so it cannot catch an unrelated failure.
    """
    lowered = detail.lower()
    return any(marker in lowered for marker in _TIMEOUT_MESSAGE_MARKERS)


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors osprey.mcp_server.bluesky.server_context)
# ---------------------------------------------------------------------------

_context: GraphContext | None = None


def get_server_context() -> GraphContext:
    """Return the GraphContext singleton.

    Raises:
        RuntimeError: If :func:`initialize_server_context` has not been called.
    """
    if _context is None:
        raise RuntimeError(
            "Graph server context not initialized. Call initialize_server_context() first."
        )
    return _context


def initialize_server_context() -> GraphContext:
    """Create and initialize the GraphContext singleton."""
    global _context
    _context = GraphContext()
    _context.initialize()
    return _context


def reset_server_context() -> None:
    """Drop the singleton and close its driver (for testing)."""
    global _context
    if _context is not None:
        _context.shutdown()
    _context = None
