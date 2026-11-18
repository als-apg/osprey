"""Neo4j/neosemantics graph-store seeding primitives.

Pure functions over an already-open neo4j **driver session**.  Nothing here
reads project config, resolves a TTL path, or decides policy: the connection is
resolved by :mod:`osprey.deployment.graphdb_service` and the seeding policy
(keep a store whose marker matches, replace any other) is assembled by the
deploy-time ``_stage_graphdb_store`` staging step from the primitives below.

**neo4j is imported lazily** (inside :func:`open_session`), never at module top,
mirroring the ``rdflib`` treatment in
:mod:`osprey.services.facility_knowledge.seeder.ttl_seeder`: importing this module
must not pull the driver into the runtime read path's ``sys.modules``, which is
guarded by ``tests/services/facility_knowledge/test_import_isolation.py``.

Driver-API constraint: the repo's ``neo4j>=5.20`` floor resolves to the 6.x
driver line while the server it dials is pinned to ``neo4j:5.26-community``.
Only APIs present in *both* lines are used — ``session.run()`` and
``result.consume()``/``result.single()``, never ``Session.read_transaction`` /
``Session.write_transaction`` (removed in 6.0).

The canonical n10s graph config and the NARAD prefix table are spelled exactly
once, here, and are taken verbatim from the prototype ontology's
``neo4j/init/01-init-constraints.cypher``.  Bootstrap diffs a store's live
config against :data:`N10S_GRAPH_CONFIG` rather than silently skipping
``graphconfig.init``, so a store initialized under older settings is reported
instead of quietly serving a graph shaped differently from what the queries
assume.
"""

from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    # Only imported at type-checking time so the runtime import stays lazy.
    from neo4j import Session

# ---------------------------------------------------------------------------
# Canonical n10s configuration (single source of truth)
# ---------------------------------------------------------------------------

#: The n10s graph configuration osprey initializes every graph store with.
#:
#: Verbatim from the prototype ontology's init cypher, whose reasoning is:
#:
#: * ``handleVocabUris='MAP'`` — property names are stored as-is with the
#:   namespaces kept on a separate ``_NsPrefDef`` node, so Cypher sees
#:   ``narad_p:sPositionM`` rather than a mangled URI.
#: * ``handleMultival='ARRAY'`` restricted by ``multivalPropList`` to
#:   ``skos:altLabel`` — the one predicate in the NARAD vocabulary that
#:   genuinely repeats, since an ontology class carries a handful of synonyms
#:   ("bend", "bending magnet", "dipole") and a place, device or channel any
#:   number of names.  Those land as a Cypher list the synonym and name
#:   lookups can search element by element.  The prop list is what keeps
#:   the rest of the graph scalar (``.sectionCode = 'GTL'``, not ``['GTL']``):
#:   ARRAY on its own would wrap *every* property in a list and rewrite every
#:   query.
#: * ``keepLangTag=False`` — the corpora this generator emits carry no
#:   language tags.
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
#: plugin build happens to be installed.  The shipped demo TTL (~8k triples)
#: lands in a single batch; a larger facility TTL commits in bounded batches.
#: This bounds what one commit holds, not what one request holds: see
#: :data:`IMPORT_CHUNK_CHARS`.  Note the consequence, which is why the
#: seed marker exists: batches commit as they go, so an import that dies partway
#: leaves committed data behind (the "unmanaged-partial" state).
COMMIT_SIZE = 10_000

#: Most characters of Turtle one ``n10s.rdf.import.inline`` request carries.
#:
#: The payload of a request is a query parameter, and the server holds all of
#: it while the request runs.  A corpus sent whole therefore needs a heap sized
#: to the corpus, whatever :data:`COMMIT_SIZE` says: about 110 MB of Turtle in
#: one request fails at a 1G heap and at 2G.  :func:`split_ttl` cuts a larger
#: corpus into requests of at most this size (one statement that is larger on
#: its own still travels whole), so the heap a store needs does not grow with
#: the corpus it is seeded from.  At this size the same corpus is 28 requests
#: and seeds at 1G.
IMPORT_CHUNK_CHARS = 4 * 1024 * 1024

#: ``terminationStatus`` value n10s reports for a fully successful import.
TERMINATION_OK = "OK"

#: Label of the singleton node carrying the seeded TTL's sha256.
SEED_MARKER_LABEL = "_OspreySeed"

#: Operator-facing text for a store whose graph config is not osprey's.
CONFIG_DRIFT_MESSAGE = "graph config differs from osprey's canonical settings"


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

#: Deletes in transactions of bounded size, so wiping a large store stays inside
#: the server's transaction heap. Each needs an auto-commit transaction, which is
#: what ``session.run`` opens.
#:
#: Two properties keep a transaction bounded, and both are load-bearing. The
#: rows to batch are matched *outside* the ``CALL``: a subquery that does its
#: own ``MATCH`` runs once, as one transaction over the whole store, whatever
#: the batch size says. And the relationships go first, on their own: under
#: ``handleRDFTypes='LABELS_AND_NODES'`` a class is a node every one of its
#: instances points at, and a ``DETACH DELETE`` that reaches such a node deletes
#: all of its relationships in the one transaction that row is in.
_WIPE_CYPHER: tuple[str, ...] = (
    "MATCH ()-[r]->() CALL { WITH r DELETE r } IN TRANSACTIONS OF 10000 ROWS",
    "MATCH (n) CALL { WITH n DELETE n } IN TRANSACTIONS OF 10000 ROWS",
)

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
    except Exception:  # see docstring
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
    database: str,
) -> Iterator[Session]:
    """Open a driver session against the graph store, closing both on exit.

    Takes the four fields of
    :class:`~osprey.deployment.graphdb_service.GraphdbConnection` explicitly
    rather than the dataclass itself, so this module stays free of any
    deployment/config import and can be driven from a test or a script with
    four literals.

    Args:
        uri: Bolt address to dial, e.g. ``bolt://localhost:7687``.
        username: Account to authenticate as.
        password: That account's password.
        database: Database to open the session against, named explicitly so the
            driver skips the home-database lookup round-trip. Required rather
            than defaulted: the one place that name is decided is the resolved
            ``GraphdbConnection``, never a literal restated here.

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
        Callers should treat :attr:`BootstrapStatus.DIFFERS` as "this store
        does not hold the corpus": wipe it, which removes the config with the
        data, and bootstrap again before importing.

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
    same way.  The deploy-time ``_stage_graphdb_store`` writes the marker, and
    the build's channel search index records the same digest so the health row
    can compare the two: spelled twice, they would agree only by coincidence,
    and a drifting encoding or a stripped newline would make one side see the
    other's digest as a different corpus.

    Args:
        text: The Turtle payload, exactly as it will be passed to
            :func:`import_ttl` — the digest covers what was imported, not what
            is on disk under some other normalization.

    Returns:
        The hex digest of the UTF-8 encoding of *text*.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ttl_file_sha256(path: Path) -> str:
    """:func:`ttl_sha256` of the file at ``path`` as ``read_text(encoding="utf-8")`` reads it.

    Read a block at a time: a corpus that is valid UTF-8 and holds no carriage
    return reads back as its own bytes, so their digest is the text's. A file
    with a carriage return, whose newlines ``read_text`` translates, is read
    whole and hashed as text, as is one that is not UTF-8, so the digest, and
    the error, are always those of ``ttl_sha256(path.read_text(encoding="utf-8"))``.

    Args:
        path: The Turtle file.

    Returns:
        The hex digest.

    Raises:
        OSError: The file cannot be read.
        UnicodeDecodeError: The file is not UTF-8, as ``read_text`` raises it.
    """
    import codecs

    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                if b"\r" in block:
                    break
                decoder.decode(block)
                digest.update(block)
            else:
                decoder.decode(b"", final=True)
                return digest.hexdigest()
    except UnicodeDecodeError:
        pass
    # A carriage return, or bytes that are not UTF-8: read as the text, which
    # translates the one and raises the other exactly as ``read_text`` does.
    return ttl_sha256(path.read_text(encoding="utf-8"))


def read_marker(session: Session) -> str | None:
    """Return the sha256 recorded on the ``(:_OspreySeed)`` marker, if any.

    Args:
        session: An open driver session.

    Returns:
        The stored sha256 hex digest, or ``None`` when no marker exists.  A
        ``None`` marker on a store with a non-zero :func:`resource_count` is the
        unmanaged-partial state — a crashed import or an adopted pre-osprey
        store — which the deploy replaces like any other mismatch.
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


# What a pass over Turtle has to recognize to find where a statement ends: the
# tokens a ``.`` can hide in (a string, an IRI, a comment), the brackets a
# statement cannot end inside, the directives, and blank node labels.  Nothing
# else is tokenized; a prefixed name or a number is skipped over, which is safe
# because neither can end in a ``.`` that whitespace follows.
_NOT_IN_NAME = r"(?<![\w\-.:%\\·̀-ͯ‿⁀])"
_TTL_TOKEN = re.compile(
    r"""
      (?P<string>
          (?: \"\"\"(?:\"{0,2}(?:[^"\\]|\\[\s\S]))*\"\"\"
            | '''(?:'{0,2}(?:[^'\\]|\\[\s\S]))*'''
            | \"(?:[^"\\\n\r]|\\.)*\"
            | '(?:[^'\\\n\r]|\\.)*'
          )
          (?:@[A-Za-z]+(?:-[A-Za-z0-9]+)*)?     # its language tag, never a directive
      )
    | (?P<iri>     <(?:[^\x00-\x20<>"{}|^`\\]|\\[uU][0-9A-Fa-f]{4,8})*> )
    | (?P<comment> \#[^\n\r]* )
    | (?P<open>    [\[(] )
    | (?P<close>   [\])] )
    | (?P<escape>  \\. )                         # an escaped character in a local name
    | (?P<at>      @(?:prefix|base)\b )
    | (?P<sparql>  """
    + _NOT_IN_NAME
    + r"""(?i:prefix|base)(?=\s) )
    | (?P<label>   """
    + _NOT_IN_NAME
    + r"""_: )
    | (?P<dot>     \.(?=[\s\#]|\Z) )
    """,
    re.VERBOSE,
)

#: Kinds of the pieces :func:`_ttl_pieces` cuts a document into.
_PLAIN, _DIRECTIVE, _LABELLED, _BLANK = "plain", "directive", "labelled", "blank"


def _ttl_pieces(text: str) -> Iterator[tuple[int, int, str]]:
    """Cut Turtle *text* into consecutive ``(start, end, kind)`` pieces.

    The pieces cover the text end to end, and each holds one statement together
    with the whitespace and comments in front of it: a directive (``@prefix``,
    ``@base`` and the SPARQL spellings ``PREFIX``, ``BASE``), a statement that
    names a labelled blank node (``_:b0``), or any other statement.  What
    follows the last statement is one closing piece: blank when it is only
    whitespace and comments, plain when it is anything else.
    """
    start = 0  # where the open piece begins
    blank_to = 0  # only whitespace and comments lie between ``start`` and here
    begun = False  # the open piece holds more than whitespace and comments
    depth = 0
    kind = _PLAIN
    sparql_open = False  # inside a SPARQL-style directive, which its IRI closes
    for match in _TTL_TOKEN.finditer(text):
        group = match.lastgroup
        # A name or a number is no token here, so the text skipped over on the
        # way to this one says whether the statement has begun.
        begun = begun or bool(text[blank_to : match.start()].strip())
        if group == "comment":
            if not begun:
                blank_to = match.end()
            continue
        if group == "sparql":
            if not begun and depth == 0:
                kind, sparql_open = _DIRECTIVE, True
            begun = True
            continue
        begun = True
        if group == "at":
            kind = _DIRECTIVE
        elif group == "label":
            if kind == _PLAIN:
                kind = _LABELLED
        elif group == "open":
            depth += 1
        elif group == "close":
            depth = max(0, depth - 1)
        if (group == "iri" and sparql_open) or (group == "dot" and depth == 0 and not sparql_open):
            yield start, match.end(), kind
            start = blank_to = match.end()
            begun, kind, sparql_open = False, _PLAIN, False
    if start < len(text):
        if kind == _PLAIN and not begun and not text[blank_to:].strip():
            kind = _BLANK
        yield start, len(text), kind


def split_ttl(text: str, max_chars: int = IMPORT_CHUNK_CHARS) -> Iterator[str]:
    """Cut Turtle *text* into documents that together state exactly its triples.

    A text of at most *max_chars* is yielded as it is.  A larger one is cut
    between statements, never inside one, into documents of at most *max_chars*
    each (a single statement that is larger travels alone).  Three properties
    make the parts equal to the whole:

    * **Directives travel with every part.**  Each document opens with every
      ``@prefix``/``@base`` (and ``PREFIX``/``BASE``) the text stated before
      the document's first statement, in the text's order, so a name resolves
      as it does at that place in the text, a redefined prefix included.
    * **A statement is never cut.**  Its anonymous blank nodes (``[ ... ]``),
      collections and multi-line strings stay in one document.
    * **Labelled blank nodes stay in one document.**  A label (``_:b0``) names
      one node for the whole text, and a parser gives each document its own, so
      every statement that uses a label is held back and yielded together,
      with the directives at their places between them, as the last document.
      That document is as large as those statements are.

    Args:
        text: The Turtle document.
        max_chars: Most characters per yielded document.

    Yields:
        The documents, in the text's order.
    """
    if len(text) <= max_chars:
        yield text
        return

    directives: list[str] = []
    labelled: list[str] = []
    has_labelled = False
    body: list[str] = []
    size = 0
    # Whether the open document states anything beyond the directives it
    # carries. One that does not is never yielded: it would be a request that
    # imports nothing.
    stated = False
    yielded = False
    for start, end, kind in _ttl_pieces(text):
        piece = text[start:end]
        if kind == _LABELLED:
            labelled.append(piece)
            has_labelled = True
            continue
        if stated and size + len(piece) > max_chars:
            yield "".join(body)
            yielded = True
            body = [directive.lstrip() + "\n" for directive in directives]
            size = sum(len(part) for part in body)
            stated = False
        body.append(piece)
        size += len(piece)
        if kind == _DIRECTIVE:
            directives.append(piece)
            labelled.append(piece)
        elif kind == _PLAIN:
            stated = True
    if stated:
        yield "".join(body)
        yielded = True
    if has_labelled:
        yield "\n".join(labelled)
    elif not yielded:
        # Nothing but directives and comments: the text itself is the request.
        yield text


def import_ttl(session: Session, text: str, *, max_chars: int = IMPORT_CHUNK_CHARS) -> ImportResult:
    """Import Turtle *text* through ``n10s.rdf.import.inline``, in bounded requests.

    Inline rather than ``import.fetch`` so the TTL needs no bind mount into the
    container's import directory, and so the same code path works against an
    external store.  :data:`COMMIT_SIZE` is passed explicitly.

    A text larger than *max_chars* is sent as the documents :func:`split_ttl`
    cuts it into, one request each, in order: the server holds one request's
    payload at a time, so the heap it needs does not grow with the corpus.  The
    import stops at the first request that fails.  Requests commit as they go,
    exactly as :data:`COMMIT_SIZE` batches already did within one request, so a
    failed import leaves the earlier requests' triples behind.

    Args:
        session: An open driver session, against a store that has been
            :func:`bootstrap`-ed.
        text: The Turtle payload.
        max_chars: Most characters of Turtle per request.

    Returns:
        One :class:`ImportResult` for the whole text: the triple counts summed
        over the requests sent, and the status and detail of the request that
        failed, if one did.  This function does **not** raise on a failed
        import — n10s reports failure in ``terminationStatus``, and the caller
        owns the message.  Check :attr:`ImportResult.ok` before
        :func:`write_marker`.
    """
    loaded = parsed = 0
    split = len(text) > max_chars
    for number, payload in enumerate(split_ttl(text, max_chars), start=1):
        result = _import_request(session, payload)
        loaded += result.triples_loaded
        parsed += result.triples_parsed
        if not result.ok:
            detail = result.extra_info
            if split:
                detail = (
                    f"import request {number} failed after {loaded} triples were loaded "
                    f"(a line number in what follows counts from that request's start): "
                    f"{detail}"
                )
            return ImportResult(
                termination_status=result.termination_status,
                triples_loaded=loaded,
                triples_parsed=parsed,
                extra_info=detail,
            )
    return ImportResult(
        termination_status=TERMINATION_OK, triples_loaded=loaded, triples_parsed=parsed
    )


def _import_request(session: Session, payload: str) -> ImportResult:
    """Run one ``n10s.rdf.import.inline`` request and return what it reported."""
    record = session.run(_IMPORT_CYPHER, payload=payload, commitSize=COMMIT_SIZE).single()
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

    This is the first step of every re-seed, on a marker mismatch and on a
    graph config that is not osprey's alike.  It removes the n10s
    ``_GraphConfig``/``_NsPrefDef`` bookkeeping and the ``(:_OspreySeed)``
    marker along with the data, which is exactly why the caller re-runs
    :func:`bootstrap` afterwards rather than going straight to the import.

    Args:
        session: An open driver session.
    """
    for statement in _WIPE_CYPHER:
        session.run(statement).consume()


# ---------------------------------------------------------------------------
# IRI names
# ---------------------------------------------------------------------------


def local_name(iri: str) -> str:
    """Return the local name fragment of an IRI (after the last ``/`` or ``#``).

    This is the single source of truth for IRI → local-name derivation.  It is
    exported from the ``seeder`` subpackage and reused by the knowledge CLI so
    that stub *placement* (the bundle filename) and stub *identity* (the OKF §2
    concept ID) are computed by exactly one rule, including ``#``-fragment IRIs.
    """
    for sep in ("#", "/"):
        idx = iri.rfind(sep)
        if idx != -1:
            return iri[idx + 1 :]
    return iri
