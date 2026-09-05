"""DuckDB tables that mirror the facility knowledge graph's n10s-imported shape.

The graph paradigm's store is a Turtle corpus imported into Neo4j by n10s, and
the explorer reads that store with Cypher. The tables here hold the same
population, flattened once per ``osprey build`` so that a search is a scan over
rows instead of a traversal.

Store shape the rows are derived from (n10s-imported, Neo4j 5.26 community, no
APOC): ``(:Resource)-[:HASBINDING]->(:ChannelBinding)``, a device typed by
``(:Resource)-[:TYPE]->(:Class)`` with ``:Class`` nodes chained by
``[:SUBCLASSOF]``, and a binding reaching its meaning through
``(:ChannelBinding)-[:READSSIGNAL|WRITESSIGNAL]->(:SemanticSignal)``. Devices
carry ``uri``, ``sourceName``, ``sectionCode`` and ``system``; bindings carry
``fullPv`` and ``description``; signals carry ``uri`` and ``label``; classes
carry ``uri`` and a list-valued ``altLabel``. Every node also wears n10s's
``Resource`` label, which is why the Cypher these tables replace matches on no
node label at all.

The mapping onto the tables:

* ``bindings`` — one row per ``(device, binding)`` pair, denormalised with the
  device's columns, the URIs and names of the signals the binding reaches, and
  the URIs of every class the device rolls up to under ``SUBCLASSOF``.
  ``haystack`` is the lowercase text the token filter matches, so a search
  never joins.
* ``classes`` — one row per class of the pruned device taxonomy, with the
  ancestors it hangs under and the device counts the ontology rail shows.
* ``channels`` — the channel roster: one record per address, so a ``full_pv``
  bound under two devices collapses to a single row.
* ``meta`` — a single row: the schema version, the corpus digest that both this
  index and the store's seed marker carry, and the badge counts.

``classes.uri`` and ``channels.address`` are each unique by construction.
``binding_uri`` is not: ``HASBINDING`` is matched per device, so a binding node
hung under two devices is two rows, one carrying each device's columns, exactly
as the store answers it. No key or index is declared on any column: the builder
writes a file once, from rows it has already collapsed, and an index would only
slow the bulk insert of a large corpus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only; duckdb stays a lazy import
    import duckdb

#: Version of the table layout below. The reader refuses an index whose
#: ``meta.schema_version`` differs, so bump this whenever a column is added,
#: removed, renamed or retyped.
SCHEMA_VERSION = 1

#: One row per ``(device, binding)`` pair, denormalised over the device, the
#: binding's signals and the classes the device rolls up to.
BINDINGS_DDL = """
CREATE TABLE bindings (
    binding_uri  VARCHAR NOT NULL,
    full_pv      VARCHAR NOT NULL,
    description  VARCHAR,
    device_uri   VARCHAR,
    device_name  VARCHAR,
    section      VARCHAR,
    system       VARCHAR,
    edges        VARCHAR[],
    signal_uris  VARCHAR[],
    signal_names VARCHAR[],
    class_uris   VARCHAR[],
    haystack     VARCHAR NOT NULL
)
"""

#: One row per class of the pruned device taxonomy.
CLASSES_DDL = """
CREATE TABLE classes (
    uri            VARCHAR NOT NULL,
    name           VARCHAR NOT NULL,
    alt_labels     VARCHAR[],
    parents        VARCHAR[],
    direct_devices BIGINT NOT NULL,
    rollup_devices BIGINT NOT NULL
)
"""

#: The channel roster: one record per address. ``readback`` is NULL when the
#: corpus names no readback for an address, never the empty string.
CHANNELS_DDL = """
CREATE TABLE channels (
    address   VARCHAR NOT NULL,
    direction VARCHAR,
    readback  VARCHAR
)
"""

#: A single row describing the index itself.
META_DDL = """
CREATE TABLE meta (
    schema_version  INTEGER NOT NULL,
    corpus_sha256   VARCHAR NOT NULL,
    corpus_filename VARCHAR NOT NULL,
    binding_count   BIGINT NOT NULL,
    device_count    BIGINT NOT NULL,
    class_count     BIGINT NOT NULL,
    signal_count    BIGINT NOT NULL,
    section_count   BIGINT NOT NULL
)
"""

#: The four statements :func:`create_tables` runs, in order.
CREATE_TABLE_STATEMENTS = (BINDINGS_DDL, CLASSES_DDL, CHANNELS_DDL, META_DDL)

#: Column names of the ``meta`` row, in table order. The builder writes them in
#: this order and the reader reads them back by name.
META_KEYS = (
    "schema_version",
    "corpus_sha256",
    "corpus_filename",
    "binding_count",
    "device_count",
    "class_count",
    "signal_count",
    "section_count",
)


def create_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create the four index tables on an open connection.

    The connection must be writable and the tables must not already exist: the
    builder writes every index into a fresh file, so there is nothing to
    migrate and no reason to tolerate a half-built one.
    """
    for statement in CREATE_TABLE_STATEMENTS:
        con.execute(statement)
