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
  (``value``); ``warning`` when that is zero or the count could not be read;
* ``graphdb_seed`` — the ``(:_OspreySeed)`` marker the seeder writes after a
  successful import: ``ok`` with the corpus digest's prefix (and the direction
  source, when the corpus declared one) as ``value``; ``warning`` when the store
  holds a corpus the marker cannot identify, when nothing has been imported at
  all, or when the marker could not be read.

The seed row's ``details`` carries the corpus's **full** sha256 behind
:data:`SEED_DIGEST_DETAIL_PREFIX`, so a caller comparing the store's corpus
against another artifact derived from the same one reads it off the row with
:func:`seed_digest` instead of opening a second driver.

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
_SEED_ROW = "graphdb_seed"

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

#: The seeder's singleton provenance marker, matched on the same ``kind`` the
#: seeder MERGEs it under. It carries the sha256 of the Turtle text that was
#: imported and, when that corpus declared one, the source its read/write
#: directions came from. The property names are the seeder's own spelling
#: (``sha256``, ``directionSource``) and are aliased here to the row builder's.
_SEED_MARKER_QUERY = (
    "MATCH (m:_OspreySeed {kind: 'ttl'}) "
    "RETURN m.sha256 AS sha256, m.directionSource AS direction_source "
    "LIMIT 1"
)

#: How much of the digest the row shows. Long enough to tell two corpora apart
#: at a glance in a tile, short enough to leave the line readable — the full
#: value stays available in ``details`` for callers that must compare exactly.
_DIGEST_PREFIX_LEN = 12

#: ``details`` prefix under which the ``graphdb_seed`` row carries the corpus's
#: full sha256. Public, with :func:`seed_digest` as its reader: the row is the
#: only place the digest is read from the store, so anything that needs to
#: compare against it parses it back out here rather than re-querying.
SEED_DIGEST_DETAIL_PREFIX = "Seeded corpus sha256: "

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
        All three rows, or the connection row alone when nothing further could
        be read from the store.
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
        count, count_error = _read_resource_count(driver)
        return [
            connection_row,
            _resources_row(count, count_error),
            _seed_row(driver, count),
        ]
    finally:
        try:
            driver.close()
        except Exception:  # noqa: BLE001 - closing is best-effort teardown
            pass


def _read_resource_count(driver: Any) -> tuple[int | None, str]:
    """Count ``(:Resource)`` nodes once, for both rows that read that number.

    The count is taken here rather than inside :func:`_resources_row` because
    the seed row needs the same number: whether a store with no marker is an
    unfinished import or simply an unseeded one is decided by whether it holds
    a corpus at all, and asking twice could answer those two rows from two
    different readings of the store.

    Args:
        driver: An open neo4j driver, already known to answer Cypher.

    Returns:
        ``(count, "")`` when the count was read, ``(None, error_text)`` when it
        could not be.
    """
    try:
        result = driver.execute_query(_RESOURCE_COUNT_QUERY)
        records = list(result.records)
        return (int(records[0]["count"]) if records else 0), ""
    except Exception as exc:  # noqa: BLE001 - an unreadable count is a warning
        return None, str(exc)


def _resources_row(count: int | None, error: str) -> CheckResult:
    """Report the ``(:Resource)`` count; ``warning`` on zero or a failed read.

    Args:
        count: What :func:`_read_resource_count` read, or ``None``.
        error: The failure text when *count* is ``None``.

    Returns:
        The ``graphdb_resources`` row. Zero is a warning rather than an error
        because a bootstrapped-but-unseeded store is a recoverable state with a
        one-command remedy, not a broken deployment.
    """
    if count is None:
        return CheckResult(
            _RESOURCES_ROW,
            CATEGORY,
            Status.WARNING,
            f"Could not count the graph's Resource nodes: {error}",
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


def _seed_row(driver: Any, count: int | None) -> CheckResult:
    """Read the seed marker and say which corpus the store was imported from.

    Args:
        driver: An open neo4j driver, already known to answer Cypher.
        count: The ``(:Resource)`` count already read for this same store, or
            ``None`` when it could not be read. Only consulted when there is no
            marker, to tell an unfinished import from an unseeded store.

    Returns:
        The ``graphdb_seed`` row — ``ok`` with the digest prefix as ``value``
        and the full digest in ``details``, otherwise a ``warning``. Like every
        row here it is advisory: a store nobody has seeded yet is a step not
        taken, not a broken build.
    """
    try:
        result = driver.execute_query(_SEED_MARKER_QUERY)
        records = list(result.records)
        record = records[0] if records else None
        sha256 = record["sha256"] if record is not None else None
        direction_source = record["direction_source"] if record is not None else None
    except Exception as exc:  # noqa: BLE001 - an unreadable marker is a warning
        return CheckResult(
            _SEED_ROW,
            CATEGORY,
            Status.WARNING,
            f"Could not read the graph's seed marker: {exc}",
        )

    if sha256 is None:
        # A marker node whose digest is missing reads the same as no marker at
        # all: the seeder writes the digest in the very MERGE that creates the
        # node, so a marker without one identifies nothing.
        return _unmarked_row(count)

    digest = str(sha256)
    value = digest[:_DIGEST_PREFIX_LEN]
    if direction_source:
        value = f"{value} (directions from {direction_source})"
    return CheckResult(
        _SEED_ROW,
        CATEGORY,
        Status.OK,
        "Store carries the seed marker of the TTL corpus it was imported from",
        value=value,
        details=f"{SEED_DIGEST_DETAIL_PREFIX}{digest}.",
    )


def _unmarked_row(count: int | None) -> CheckResult:
    """Build the ``graphdb_seed`` row for a store carrying no marker.

    Args:
        count: The ``(:Resource)`` count, or ``None`` when it could not be read.

    Returns:
        The ``warning`` row. A store with data but no marker is the more serious
        of the two — an import that died partway, or a store seeded outside
        osprey — and says so, because "unseeded" would understate a graph that
        already answers queries from a corpus nobody can name.
    """
    if count is None:
        return CheckResult(
            _SEED_ROW,
            CATEGORY,
            Status.WARNING,
            "Graph store carries no seed marker",
            details=(
                "Its Resource count could not be read either, so whether it holds a corpus "
                f"is unknown. Import the TTL corpus with `{_SEED_COMMAND}`."
            ),
        )
    if count > 0:
        return CheckResult(
            _SEED_ROW,
            CATEGORY,
            Status.WARNING,
            "Graph store holds a corpus with no seed marker",
            details=(
                "The marker is written last, so an import that failed partway leaves this "
                "state — as does a store seeded outside osprey. Either way the corpus it "
                f"holds cannot be identified: re-import it with `{_SEED_COMMAND}`."
            ),
        )
    return CheckResult(
        _SEED_ROW,
        CATEGORY,
        Status.WARNING,
        "Graph store is unseeded: no TTL corpus has been imported",
        details=f"Import the TTL corpus with `{_SEED_COMMAND}`.",
    )


def seed_digest(row: CheckResult) -> str:
    """Return the full corpus digest a ``graphdb_seed`` row carries.

    Args:
        row: A ``graphdb_seed`` row — under this name or a caller's relabelling
            of it, since only ``details`` is read.

    Returns:
        The sha256 hex digest, or ``""`` for any row that carries none (a
        warning row, or a row from some other check).
    """
    details = row.details
    if not details.startswith(SEED_DIGEST_DETAIL_PREFIX):
        return ""
    return details[len(SEED_DIGEST_DETAIL_PREFIX) :].split()[0].rstrip(".")


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
