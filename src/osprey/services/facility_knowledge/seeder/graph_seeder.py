"""Neo4j/neosemantics graph-store seeding primitives.

Pure functions over an already-open neo4j **driver session**.  Nothing here
reads project config, resolves a TTL path, or decides policy: the connection is
resolved by :mod:`osprey.deployment.graphdb_service` and the five-state seeding
vocabulary (seeded / unchanged / differs / unmanaged-partial / overwritten) is
assembled by the callers — ``osprey knowledge seed-graph`` and the deploy-time
``_stage_graphdb_store`` staging step — from the primitives below.  Keeping the
policy out of this module is what lets both consumers behave identically.

**neo4j is imported lazily** (inside :func:`open_session`), never at module top,
mirroring the ``rdflib`` treatment in :mod:`.ttl_seeder`: importing this module
must not pull the driver into the runtime read path's ``sys.modules``, which is
guarded by ``tests/services/facility_knowledge/test_import_isolation.py``.

Driver-API constraint: the repo's ``neo4j>=5.20`` floor resolves to the 6.x
driver line while the server it dials is pinned to ``neo4j:5.26-community``.
Only APIs present in *both* lines are used — ``session.run()`` and
``result.consume()``/``result.single()``, never ``Session.read_transaction`` /
``Session.write_transaction`` (removed in 6.0).

The canonical n10s graph config and the NARAD prefix table are spelled exactly
once, here, and are taken verbatim from the als-ontology prototype's
``neo4j/init/01-init-constraints.cypher``.  Bootstrap diffs a store's live
config against :data:`N10S_GRAPH_CONFIG` rather than silently skipping
``graphconfig.init``, so a store initialized under older settings is reported
instead of quietly serving a graph shaped differently from what the queries
assume.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    # Only imported at type-checking time so the runtime import stays lazy.
    from neo4j import Session  # noqa: F401

# ---------------------------------------------------------------------------
# Canonical n10s configuration (single source of truth)
# ---------------------------------------------------------------------------

#: The n10s graph configuration osprey initializes every graph store with.
#:
#: Verbatim from the als-ontology prototype's init cypher, whose reasoning is:
#:
#: * ``handleVocabUris='MAP'`` — property names are stored as-is with the
#:   namespaces kept on a separate ``_NsPrefDef`` node, so Cypher sees
#:   ``narad_p:sPositionM`` rather than a mangled URI.
#: * ``handleMultival='ARRAY'`` restricted by ``multivalPropList`` to
#:   ``skos:altLabel`` — the one predicate in the NARAD vocabulary that
#:   genuinely repeats, since an ontology class carries a handful of synonyms
#:   ("bend", "bending magnet", "dipole").  Those land as a Cypher list the
#:   synonym lookups can search element by element.  The prop list is what keeps
#:   the rest of the graph scalar (``.sectionCode = 'GTL'``, not ``['GTL']``):
#:   ARRAY on its own would wrap *every* property in a list and rewrite every
#:   query.
#: * ``keepLangTag=False`` — the ALS data carries no language tags.
#: * ``handleRDFTypes='LABELS_AND_NODES'`` — every ``rdf:type`` becomes both a
#:   ``:Resource`` label and a typed node, which is what the class-hierarchy
#:   rollup queries traverse.
#: * ``applyNeo4jNaming=True`` — neo4j naming conventions for the mapped names.
N10S_GRAPH_CONFIG: dict[str, Any] = {
    "handleVocabUris": "MAP",
    "handleMultival": "ARRAY",
    "multivalPropList": ["http://www.w3.org/2004/02/skos/core#altLabel"],
    "keepLangTag": False,
    "handleRDFTypes": "LABELS_AND_NODES",
    "applyNeo4jNaming": True,
}

#: Namespace prefixes registered before import so n10s preserves the NARAD
#: semantic structure instead of autogenerating ``ns0``/``ns1``/... names.
NARAD_PREFIXES: dict[str, str] = {
    "narad": "https://narad.example.org/",
    "narad_sem": "https://narad.example.org/schema/shared_semantics/",
    "narad_p": "https://narad.example.org/property/",
    "narad_sig": "https://narad.example.org/signal/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
}

#: Triples per commit for ``n10s.rdf.import.inline``.
#:
#: Spelled once, and passed explicitly rather than left to the n10s default so
#: the transaction-heap envelope is a property of osprey rather than of whatever
#: plugin build happens to be installed.  The shipped ALS TTL (~8k triples) lands
#: in a single batch; a larger facility TTL streams in bounded batches that stay
#: well inside the default 512m heap.  Note the consequence, which is why the
#: seed marker exists: batches commit as they go, so an import that dies partway
#: leaves committed data behind (the "unmanaged-partial" state).
COMMIT_SIZE = 10_000

#: ``terminationStatus`` value n10s reports for a fully successful import.
TERMINATION_OK = "OK"

#: Default Neo4j database.  Named explicitly so the driver skips the home-database
#: lookup round-trip on every session.
DEFAULT_DATABASE = "neo4j"

#: Label of the singleton node carrying the seeded TTL's sha256.
SEED_MARKER_LABEL = "_OspreySeed"

#: Operator-facing text for a store whose graph config is not osprey's.
CONFIG_DRIFT_MESSAGE = (
    "graph config differs from osprey's canonical settings — re-seed with --force"
)


# ---------------------------------------------------------------------------
# Cypher statements
# ---------------------------------------------------------------------------

_CONSTRAINT_CYPHER = (
    "CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS FOR (r:Resource) REQUIRE r.uri IS UNIQUE"
)

_GRAPHCONFIG_SHOW_CYPHER = "CALL n10s.graphconfig.show() YIELD param, value RETURN param, value"

_GRAPHCONFIG_INIT_CYPHER = "CALL n10s.graphconfig.init($params)"

_NSPREFIX_ADD_CYPHER = "CALL n10s.nsprefixes.add($prefix, $namespace)"

_RESOURCE_COUNT_CYPHER = "MATCH (r:Resource) RETURN count(r) AS resource_count"

_READ_MARKER_CYPHER = f"MATCH (m:{SEED_MARKER_LABEL}) RETURN m.sha256 AS sha256 LIMIT 1"

_WRITE_MARKER_CYPHER = (
    f"MERGE (m:{SEED_MARKER_LABEL} {{kind: 'ttl'}}) SET m.sha256 = $sha256, m.seededAt = datetime()"
)

_IMPORT_CYPHER = (
    "CALL n10s.rdf.import.inline($payload, 'Turtle', {commitSize: $commitSize}) "
    "YIELD terminationStatus, triplesLoaded, triplesParsed, extraInfo "
    "RETURN terminationStatus, triplesLoaded, triplesParsed, extraInfo"
)

_WIPE_CYPHER = "MATCH (n) DETACH DELETE n"

#: Server identity, read only to name the version in the missing-plugin error.
_COMPONENTS_CYPHER = "CALL dbms.components() YIELD name, versions, edition"

#: Driver error code for a Cypher ``CALL`` naming a procedure the server does
#: not have. Matched by attribute rather than by exception type so this module
#: keeps its no-driver-import property (see :func:`open_session`).
_PROCEDURE_NOT_FOUND_CODE = "Neo.ClientError.Procedure.ProcedureNotFound"


class MissingN10sPluginError(RuntimeError):
    """The store answered, but the neosemantics (n10s) plugin is not loaded.

    The default image installs n10s at container start from the plugin
    manifest; on a Neo4j version the manifest has no entry for — or on a host
    whose egress mangles the manifest fetch — the jar is silently absent, the
    server comes up healthy, and the first ``n10s.*`` call dies with
    ``ProcedureNotFound``. Surfaced raw, that reads like a corrupt database;
    this error says what actually happened and which server version has no
    plugin, so the remedy (a Neo4j version n10s publishes for, or a pre-baked
    jar) is findable.
    """


def _server_identity(session: Session) -> str:
    """Best-effort ``<version> <edition>`` of the server, for error text only.

    Never raises: an error message that dies producing itself reports the
    wrong fault, so any failure here degrades to a placeholder.
    """
    try:
        for record in session.run(_COMPONENTS_CYPHER):
            if record["name"] == "Neo4j Kernel":
                return f"{record['versions'][0]} {record['edition']}"
    except Exception:  # noqa: BLE001 — see docstring
        pass
    return "<unknown version>"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class BootstrapStatus(StrEnum):
    """Outcome of the graph-config half of :func:`bootstrap`.

    Attributes:
        INITIALIZED: No config existed; osprey's canonical one was installed.
        ALREADY_CANONICAL: A config existed and matches osprey's canonical one.
        DIFFERS: A config existed and does *not* match — ``graphconfig.init``
            was deliberately **not** run, because n10s rejects re-initialization
            of a configured store and silently continuing would query a graph
            shaped differently from what the callers assume.
    """

    INITIALIZED = "initialized"
    ALREADY_CANONICAL = "already_canonical"
    DIFFERS = "differs"


@dataclass(frozen=True)
class BootstrapResult:
    """What :func:`bootstrap` did, in a form callers can branch on.

    Attributes:
        status: See :class:`BootstrapStatus`.
        differing_keys: Canonical keys whose live value disagrees (or is
            missing), sorted.  Empty unless ``status`` is
            :attr:`BootstrapStatus.DIFFERS`.
    """

    status: BootstrapStatus
    differing_keys: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when the store's config is osprey's — freshly set or already so."""
        return self.status is not BootstrapStatus.DIFFERS

    @property
    def message(self) -> str:
        """A one-line operator-facing summary of this outcome."""
        if self.status is BootstrapStatus.INITIALIZED:
            return "graph config initialized with osprey's canonical settings"
        if self.status is BootstrapStatus.ALREADY_CANONICAL:
            return "graph config already matches osprey's canonical settings"
        keys = ", ".join(self.differing_keys)
        return f"{CONFIG_DRIFT_MESSAGE} (differing keys: {keys})"


@dataclass(frozen=True)
class ImportResult:
    """What ``n10s.rdf.import.inline`` reported.

    Attributes:
        termination_status: n10s's own verdict — :data:`TERMINATION_OK` or a
            failure code.  This, not a non-zero ``triples_loaded``, is the thing
            callers must gate on: a partially applied import still loads triples.
        triples_loaded: Triples committed to the store.
        triples_parsed: Triples read out of the payload.
        extra_info: n10s's failure detail; empty string on success.
    """

    termination_status: str
    triples_loaded: int
    triples_parsed: int
    extra_info: str = ""

    @property
    def ok(self) -> bool:
        """True when n10s reported ``terminationStatus='OK'``."""
        return self.termination_status == TERMINATION_OK


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


@contextmanager
def open_session(
    uri: str,
    username: str,
    password: str,
    *,
    database: str = DEFAULT_DATABASE,
) -> Iterator[Session]:
    """Open a driver session against the graph store, closing both on exit.

    Takes the three fields of
    :class:`~osprey.deployment.graphdb_service.GraphdbConnection` explicitly
    rather than the dataclass itself, so this module stays free of any
    deployment/config import and can be driven from a test or a script with
    three literals.

    Args:
        uri: Bolt address to dial, e.g. ``bolt://localhost:7687``.
        username: Account to authenticate as.
        password: That account's password.
        database: Database to open the session against.

    Yields:
        An open :class:`neo4j.Session`.

    Raises:
        ImportError: If the ``neo4j`` driver is not installed.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:  # pragma: no cover - exercised via sys.modules patch
        raise ImportError(
            "neo4j is required to reach the graph store; it is a core osprey "
            "dependency, so a missing driver means a broken install — repair it "
            "with: pip install 'osprey-framework'"
        ) from exc

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=database) as session:
            yield session
    finally:
        driver.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize(value: Any) -> str:
    """Render a config value as the string used for canonical-vs-live comparison.

    n10s hands booleans back as booleans and enums back as strings, while a few
    builds stringify everything; normalizing both sides through the same funnel
    keeps ``keepLangTag=False`` from reading as drift against a returned
    ``"false"``.

    The same split runs on multi-valued parameters: ``multivalPropList`` comes
    back as a list from some builds and as a comma-joined string from others,
    in no guaranteed order.  Both shapes are funnelled to one sorted,
    comma-joined string, so the diff only reports a genuinely different set of
    properties.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set, frozenset)):
        parts = [_normalize(item) for item in value]
    else:
        parts = str(value).split(",")
    return ",".join(sorted(part.strip() for part in parts))


def _live_graph_config(session: Session) -> dict[str, Any]:
    """Return the store's current n10s config as ``{param: value}``.

    An unconfigured store makes ``n10s.graphconfig.show`` yield zero rows (it
    swallows its own "graph config not found"), so an empty mapping means "no
    config", not "empty config".
    """
    return {record["param"]: record["value"] for record in session.run(_GRAPHCONFIG_SHOW_CYPHER)}


def _config_drift(live: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the canonical keys whose live value is missing or different."""
    return tuple(
        sorted(
            key
            for key, expected in N10S_GRAPH_CONFIG.items()
            if key not in live or _normalize(live[key]) != _normalize(expected)
        )
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bootstrap(session: Session) -> BootstrapResult:
    """Make a graph store ready for RDF import.  Safe to re-run.

    Three steps, in the prototype's order:

    1. The ``Resource(uri)`` uniqueness constraint n10s requires before any
       import — ``IF NOT EXISTS``, so re-running is a no-op.
    2. ``n10s.graphconfig.init`` with :data:`N10S_GRAPH_CONFIG`, **gated on
       ``n10s.graphconfig.show`` reporting no config**.  That gate is what makes
       the bootstrap idempotent; n10s refuses to re-initialize a configured
       store.  When a config does exist, its load-bearing keys are diffed
       against the canonical dict and any drift is *reported*
       (:attr:`BootstrapStatus.DIFFERS`) rather than silently skipped.
    3. Every :data:`NARAD_PREFIXES` mapping, always — re-adding an identical
       prefix mapping is a no-op, and running it on the drift path too keeps
       "prefixes are registered after a successful bootstrap call" true without
       a caveat.

    Args:
        session: An open driver session.

    Returns:
        A :class:`BootstrapResult` distinguishing the three config outcomes.
        Callers should treat :attr:`BootstrapStatus.DIFFERS` as "do not import;
        tell the operator to re-seed with ``--force``" — see
        :attr:`BootstrapResult.message`.

    Raises:
        MissingN10sPluginError: If the store has no ``n10s.*`` procedures —
            the plugin never installed. Detected on the first n10s call, which
            is what any later one would die on too, and named as the plugin
            gap it is rather than surfacing as a missing stored procedure.
    """
    session.run(_CONSTRAINT_CYPHER).consume()

    try:
        live = _live_graph_config(session)
    except Exception as exc:
        if getattr(exc, "code", None) != _PROCEDURE_NOT_FOUND_CODE:
            raise
        raise MissingN10sPluginError(
            f"no compatible n10s plugin installed for Neo4j {_server_identity(session)} — "
            f"the server is up and answering, but the neosemantics plugin that imports "
            f"RDF never loaded (the container log carries the matching 'No compatible "
            f"n10s plugin found' line from startup). Point services.graphdb.image at a "
            f"Neo4j version neosemantics publishes a build for, or bake the n10s jar "
            f"into the image, then recreate the store."
        ) from exc
    if not live:
        session.run(_GRAPHCONFIG_INIT_CYPHER, params=dict(N10S_GRAPH_CONFIG)).consume()
        result = BootstrapResult(status=BootstrapStatus.INITIALIZED)
    else:
        differing = _config_drift(live)
        result = (
            BootstrapResult(status=BootstrapStatus.DIFFERS, differing_keys=differing)
            if differing
            else BootstrapResult(status=BootstrapStatus.ALREADY_CANONICAL)
        )

    for prefix, namespace in NARAD_PREFIXES.items():
        session.run(_NSPREFIX_ADD_CYPHER, prefix=prefix, namespace=namespace).consume()

    return result


def resource_count(session: Session) -> int:
    """Return the number of ``(:Resource)`` nodes in the store.

    This is osprey's definition of "empty": zero ``(:Resource)`` nodes.  n10s's
    own bookkeeping nodes (``_GraphConfig``, ``_NsPrefDef``) and the
    ``(:_OspreySeed)`` marker carry no ``:Resource`` label and so do not count —
    which is what lets a bootstrapped-but-unseeded store still read as empty.

    Args:
        session: An open driver session.

    Returns:
        The ``(:Resource)`` node count.
    """
    record = session.run(_RESOURCE_COUNT_CYPHER).single()
    return int(record["resource_count"]) if record is not None else 0


def ttl_sha256(text: str) -> str:
    """Return the seed marker's identity for a TTL payload: sha256 over its text.

    The marker's whole job is to answer "is the store already holding *this*
    corpus?", and it can only do that if every consumer computes the digest the
    same way.  Two consumers write markers — ``osprey knowledge seed-graph`` and
    the deploy-time ``_stage_graphdb_store`` — and each one reads the other's:
    a deploy skips seeding a store the verb filled, and the verb reports
    ``unchanged`` for a store the deploy seeded.  Spelled twice, the two would
    agree only by coincidence, and a drifting encoding or a stripped newline
    would make each side see the other's marker as a different corpus and
    re-seed on sight.

    Args:
        text: The Turtle payload, exactly as it will be passed to
            :func:`import_ttl` — the digest covers what was imported, not what
            is on disk under some other normalization.

    Returns:
        The hex digest of the UTF-8 encoding of *text*.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_marker(session: Session) -> str | None:
    """Return the sha256 recorded on the ``(:_OspreySeed)`` marker, if any.

    Args:
        session: An open driver session.

    Returns:
        The stored sha256 hex digest, or ``None`` when no marker exists.  A
        ``None`` marker on a store with a non-zero :func:`resource_count` is the
        unmanaged-partial state — a crashed import or an adopted pre-osprey
        store — which callers refuse without ``--force``.
    """
    record = session.run(_READ_MARKER_CYPHER).single()
    if record is None:
        return None
    sha256 = record["sha256"]
    return str(sha256) if sha256 is not None else None


def write_marker(session: Session, sha256: str) -> None:
    """Record *sha256* on the singleton ``(:_OspreySeed)`` marker node.

    **Sequencing is the caller's job, and it is load-bearing:** call this only
    after :func:`import_ttl` has returned a result whose
    :attr:`ImportResult.ok` is true.  :data:`COMMIT_SIZE` batches commit as they
    go, so an import that fails partway still leaves triples in the store; a
    marker written regardless would label that half-graph as a good seed and
    every later run would report "unchanged" and skip it forever.  Writing the
    marker last is what makes the unmanaged-partial state detectable.

    Args:
        session: An open driver session.
        sha256: Hex digest of the TTL text that was imported.
    """
    session.run(_WRITE_MARKER_CYPHER, sha256=sha256).consume()


def import_ttl(session: Session, text: str) -> ImportResult:
    """Import Turtle *text* through ``n10s.rdf.import.inline``.

    Inline rather than ``import.fetch`` so the TTL needs no bind mount into the
    container's import directory, and so the same code path works against an
    external store.  :data:`COMMIT_SIZE` is passed explicitly.

    Args:
        session: An open driver session, against a store that has been
            :func:`bootstrap`-ed.
        text: The Turtle payload.

    Returns:
        The :class:`ImportResult` n10s reported.  This function does **not**
        raise on a failed import — n10s reports failure in
        ``terminationStatus``, and the caller owns the message.  Check
        :attr:`ImportResult.ok` before :func:`write_marker`.
    """
    record = session.run(_IMPORT_CYPHER, payload=text, commitSize=COMMIT_SIZE).single()
    if record is None:
        return ImportResult(
            termination_status="",
            triples_loaded=0,
            triples_parsed=0,
            extra_info="n10s.rdf.import.inline returned no result row",
        )
    return ImportResult(
        termination_status=str(record["terminationStatus"]),
        triples_loaded=int(record["triplesLoaded"] or 0),
        triples_parsed=int(record["triplesParsed"] or 0),
        extra_info=str(record["extraInfo"] or ""),
    )


def wipe(session: Session) -> None:
    """Delete every node and relationship in the store.

    This is the first step of a ``--force`` re-seed.  It removes the n10s
    ``_GraphConfig``/``_NsPrefDef`` bookkeeping and the ``(:_OspreySeed)`` marker
    along with the data, which is exactly why ``--force`` re-runs
    :func:`bootstrap` afterwards rather than going straight to the import.

    Args:
        session: An open driver session.
    """
    session.run(_WIPE_CYPHER).consume()
