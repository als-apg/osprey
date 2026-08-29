"""Core ``graphdb`` health category.

Probes the deployment's graph store, but **only when it is configured**.
Presence is keyed on the ``services.graphdb`` config block — the same block the
compose fragment, the port sweep and the seeder read — so the category stays a
valid ``--category`` name at all times while contributing no rows to a project
that runs no graph store. A minimal build therefore shows no graphdb tile at
all.

When configured the category opens one driver against the resolved connection
(:func:`osprey.deployment.graphdb_service.resolve_graphdb_connection`, which
also decides whether the address is the locally published bolt port or an
explicit external ``uri``) and derives both rows from it:

* ``graphdb_connection`` — the store answered a trivial query over bolt (``ok``
  with ``latency_ms``); ``warning`` — and, as the sole row, since nothing
  further can be read — when the ``neo4j`` driver is not installed, the store is
  unreachable, the credential was rejected, or the config block is malformed;
* ``graphdb_resources`` — the number of ``(:Resource)`` nodes in the graph
  (``value``); ``warning`` when that is zero or the count could not be read.

The row counts ``(:Resource)`` nodes specifically rather than all nodes, and
says so in its message: bootstrapping the store creates neosemantics
bookkeeping nodes (``_GraphConfig``, namespace prefixes) whether or not a corpus
was ever imported, so a bare node count reports a freshly bootstrapped, empty
graph as populated.

Every row is advisory (``ok``/``warning``). A store that is down is a service
that is not running, not a broken build, and the whole point of the category is
to say so in one line rather than to fail the suite — so no path here produces
an ``error``, and no driver failure escapes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from time import perf_counter
from typing import TYPE_CHECKING, Any

from osprey.deployment.graphdb_service import (
    GRAPHDB_PASSWORD_ENV,
    GRAPHDB_SEED_COMMAND,
    GRAPHDB_SERVICE_NAME,
    GraphdbConnection,
    resolve_graphdb_connection,
)
from osprey.health.models import CheckResult, Status
from osprey.port_layout import resolve_port_base

if TYPE_CHECKING:
    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "graphdb"

_CONNECTION_ROW = "graphdb_connection"
_RESOURCES_ROW = "graphdb_resources"

#: Seconds the driver may spend acquiring a connection before it gives up. Kept
#: well inside the category timeout: an unreachable store should render as one
#: warning row quickly, not hold the suite open on a TCP timeout.
_CONNECTION_TIMEOUT_S = 5.0

#: Trivial round-trip that proves the store is up, authenticated and answering
#: Cypher — cheap enough that its timing is a fair reading of bolt latency
#: rather than of the graph's size.
_PING_QUERY = "RETURN 1 AS ok"

#: The corpus's node count. ``(:Resource)`` is the label neosemantics gives every
#: imported RDF subject, so this counts data and not the store's own bookkeeping.
_RESOURCE_COUNT_QUERY = "MATCH (n:Resource) RETURN count(n) AS count"

#: Named by both the empty-graph row and the deploy that auto-seeds, so an
#: operator reading either is pointed at the same verb — imported rather than
#: re-spelled, which is what makes that true rather than merely intended.
_SEED_COMMAND = GRAPHDB_SEED_COMMAND


def graphdb(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,
) -> CategoryCallable:
    """Build the ``graphdb`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable).
            Read for the ``services.graphdb`` block — the presence gate, and the
            ``uri``/``username``/``port_host`` keys the connection resolves from.
        context: Health runtime. Unused — the graph store is dialed over bolt, so
            no control-system connector is needed.

    Returns:
        A no-argument async callable returning the category's check results.
    """
    cfg: Mapping[str, Any] = config or {}

    async def _run() -> list[CheckResult]:
        block = _graphdb_block(cfg)
        if block is None:
            return []

        try:
            connection = resolve_graphdb_connection(block, base=resolve_port_base(cfg))
        except ValueError as exc:
            # A malformed block is reported as the connection row rather than
            # raised: the same typo already refuses loudly at deploy time, and
            # a health suite that crashes tells an operator less than one that
            # names the key.
            return [
                CheckResult(
                    _CONNECTION_ROW,
                    CATEGORY,
                    Status.WARNING,
                    f"Cannot resolve the graph store's address: {exc}",
                    details="Fix the key named above in config.yml.",
                )
            ]

        return await asyncio.to_thread(_probe, connection)

    return _run


def _graphdb_block(cfg: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return a non-empty ``services.graphdb`` block, or ``None`` (the gate).

    Args:
        cfg: The parsed config mapping.

    Returns:
        The block when the project describes a graph store, otherwise ``None``
        — which includes the bare ``graphdb:`` key YAML parses as ``None``, and
        an empty block, neither of which names a store to dial.
    """
    services = cfg.get("services")
    if not isinstance(services, Mapping):
        return None
    block = services.get(GRAPHDB_SERVICE_NAME)
    if not isinstance(block, Mapping) or not block:
        return None
    return block


def _probe(connection: GraphdbConnection) -> list[CheckResult]:
    """Dial the store on a worker thread and build both rows.

    The driver is imported lazily here, inside the thread, for two reasons: the
    package must not be paid for by every ``osprey`` import when most
    deployments run no graph store, and a missing one has to degrade to a
    ``warning`` row rather than break the category's import.

    Args:
        connection: Address and credentials to open the driver with.

    Returns:
        Both rows, or the connection row alone when nothing further could be
        read from the store.
    """
    try:
        from neo4j import GraphDatabase
        from neo4j.exceptions import AuthError
    except ImportError:
        return [
            CheckResult(
                _CONNECTION_ROW,
                CATEGORY,
                Status.WARNING,
                "Cannot reach the graph store: the 'neo4j' driver is not installed",
                details="Install the 'neo4j' dependency to enable graph-store checks.",
            )
        ]

    start = perf_counter()
    try:
        driver = GraphDatabase.driver(
            connection.uri,
            auth=(connection.username, connection.password),
            connection_acquisition_timeout=_CONNECTION_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 - a bad address degrades, never crashes
        return [_unreachable_row(connection, exc, perf_counter() - start)]

    try:
        try:
            driver.execute_query(_PING_QUERY)
        except AuthError as exc:
            return [_auth_row(connection, exc, perf_counter() - start)]
        except Exception as exc:  # noqa: BLE001 - any dial failure is a warning
            return [_unreachable_row(connection, exc, perf_counter() - start)]

        latency_ms = (perf_counter() - start) * 1000.0
        connection_row = CheckResult(
            _CONNECTION_ROW,
            CATEGORY,
            Status.OK,
            f"Graph store reachable over bolt ({connection.uri})",
            latency_ms=latency_ms,
        )
        return [connection_row, _resources_row(driver)]
    finally:
        try:
            driver.close()
        except Exception:  # noqa: BLE001 - closing is best-effort teardown
            pass


def _resources_row(driver: Any) -> CheckResult:
    """Count ``(:Resource)`` nodes; ``warning`` on zero or on a failed read.

    Args:
        driver: An open neo4j driver, already known to answer Cypher.

    Returns:
        The ``graphdb_resources`` row. Zero is a warning rather than an error
        because a bootstrapped-but-unseeded store is a recoverable state with a
        one-command remedy, not a broken deployment.
    """
    try:
        result = driver.execute_query(_RESOURCE_COUNT_QUERY)
        records = list(result.records)
        count = int(records[0]["count"]) if records else 0
    except Exception as exc:  # noqa: BLE001 - an unreadable count is a warning
        return CheckResult(
            _RESOURCES_ROW,
            CATEGORY,
            Status.WARNING,
            f"Could not count the graph's Resource nodes: {exc}",
        )

    if count <= 0:
        return CheckResult(
            _RESOURCES_ROW,
            CATEGORY,
            Status.WARNING,
            "Graph holds no Resource nodes — it has been bootstrapped but not seeded",
            details=f"Import the TTL corpus with `{_SEED_COMMAND}`.",
        )
    return CheckResult(
        _RESOURCES_ROW,
        CATEGORY,
        Status.OK,
        "Resource nodes imported from the TTL corpus",
        value=f"{count:,} Resource nodes",
    )


def _unreachable_row(
    connection: GraphdbConnection, exc: Exception, elapsed_s: float
) -> CheckResult:
    """Build the sole row for a store that could not be dialed."""
    return CheckResult(
        _CONNECTION_ROW,
        CATEGORY,
        Status.WARNING,
        f"Graph store unreachable at {connection.uri}: {exc}",
        latency_ms=elapsed_s * 1000.0,
        details=(
            f"The graph store runs as the '{GRAPHDB_SERVICE_NAME}' compose service — check "
            "that it is up, or verify services.graphdb.port_host / services.graphdb.uri."
        ),
    )


def _auth_row(connection: GraphdbConnection, exc: Exception, elapsed_s: float) -> CheckResult:
    """Build the sole row for a store that rejected the credential."""
    return CheckResult(
        _CONNECTION_ROW,
        CATEGORY,
        Status.WARNING,
        f"Graph store rejected the credentials at {connection.uri}",
        latency_ms=elapsed_s * 1000.0,
        details=(
            f"Both ends read {GRAPHDB_PASSWORD_ENV}: the store was created with the value "
            "the project .env carried at the time, and keeps it. If that value has since "
            "changed, restore it — or reset the store's data volume so it is created "
            f"again with the current one. Underlying error: {exc}"
        ),
    )
