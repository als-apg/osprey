"""Neo4j graph store shared by the facility knowledge graph and Channel finder.

* :mod:`.seeder` holds the seeding primitives — bootstrap the neosemantics
  config, import a NARAD TTL, read and write the seed marker — over an
  already-open driver session.
* :mod:`.prompt_snapshot` captures the seeded store's schema into the rendered
  graph agent prompts at deploy time.

The connection itself is resolved by :mod:`osprey.deployment.graphdb_service`.
The ``neo4j`` driver is imported lazily inside :func:`open_session`, so importing
this package keeps it out of ``sys.modules``.

Example::

    from osprey.services.graph_store import bootstrap, open_session

    with open_session(
        conn.uri, conn.username, conn.password, database=conn.database
    ) as session:
        bootstrap(session)
"""

from .seeder import (
    COMMIT_SIZE,
    CONFIG_DRIFT_MESSAGE,
    N10S_GRAPH_CONFIG,
    NARAD_PREFIXES,
    SEED_MARKER_LABEL,
    TERMINATION_OK,
    BootstrapResult,
    BootstrapStatus,
    ImportResult,
    bootstrap,
    import_ttl,
    local_name,
    open_session,
    read_marker,
    resource_count,
    ttl_sha256,
    wipe,
    write_marker,
)

__all__ = [
    "COMMIT_SIZE",
    "CONFIG_DRIFT_MESSAGE",
    "N10S_GRAPH_CONFIG",
    "NARAD_PREFIXES",
    "SEED_MARKER_LABEL",
    "TERMINATION_OK",
    "BootstrapResult",
    "BootstrapStatus",
    "ImportResult",
    "bootstrap",
    "import_ttl",
    "local_name",
    "open_session",
    "read_marker",
    "resource_count",
    "ttl_sha256",
    "wipe",
    "write_marker",
]
