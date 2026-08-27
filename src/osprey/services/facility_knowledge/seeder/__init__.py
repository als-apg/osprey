"""Facility knowledge seeder subpackage.

Two seeders, both offline and both driven from ``osprey knowledge``:

* :func:`seed_from_ttl` (:mod:`.ttl_seeder`) turns a NARAD/als-ontology RDF/TTL
  file into OKF stub *documents* — a text bundle on disk.
* :func:`bootstrap` / :func:`import_ttl` and friends (:mod:`.graph_seeder`) load
  that same TTL into a deployed Neo4j *graph store* through neosemantics.

Import this subpackage directly — it is intentionally excluded from the parent
``facility_knowledge`` namespace so that the runtime read path pulls in neither
``rdflib`` nor the ``neo4j`` driver.  Both are core dependencies, so this is an
import-graph boundary rather than an optional-install one; both are imported
lazily inside the functions that need them.

Example::

    from osprey.services.facility_knowledge.seeder import open_session, seed_from_ttl

    stubs = seed_from_ttl(Path("/path/to/corpus.ttl"))

    with open_session(conn.uri, conn.username, conn.password) as session:
        bootstrap(session)
"""

from .graph_seeder import (
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
    open_session,
    parse_direction_source,
    read_direction_source,
    read_marker,
    resource_count,
    ttl_sha256,
    wipe,
    write_marker,
)
from .ttl_seeder import DeviceStub, local_name, seed_from_ttl

__all__ = [
    "COMMIT_SIZE",
    "CONFIG_DRIFT_MESSAGE",
    "N10S_GRAPH_CONFIG",
    "NARAD_PREFIXES",
    "SEED_MARKER_LABEL",
    "TERMINATION_OK",
    "BootstrapResult",
    "BootstrapStatus",
    "DeviceStub",
    "ImportResult",
    "bootstrap",
    "import_ttl",
    "local_name",
    "open_session",
    "parse_direction_source",
    "read_direction_source",
    "read_marker",
    "resource_count",
    "seed_from_ttl",
    "ttl_sha256",
    "wipe",
    "write_marker",
]
