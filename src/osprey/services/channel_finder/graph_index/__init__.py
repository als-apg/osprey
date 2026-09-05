"""The graph paradigm's flat search index.

The facility knowledge graph is a Turtle corpus, seeded into Neo4j for the
agent's semantic reads and flattened here into a DuckDB file for everything the
explorer asks on a click: search, ontology, statistics and the channel roster.
One ``osprey build`` derives the index; every reader afterwards opens a file.

Importing this package must stay cheap. It pulls in neither ``duckdb``,
``rdflib``, ``neo4j`` nor ``osprey.services.qmd``: the roster and the health
check import it on paths where dragging a graph stack in would be a regression.
Every public name is therefore resolved on first attribute access, and the
modules behind them import their own heavy dependencies inside the functions
that need them.
"""

from __future__ import annotations

from typing import Any

#: Public name -> the submodule of this package (or ``..core.exceptions``) that
#: defines it. Entries are resolved on first attribute access, never at import.
_LAZY_EXPORTS: dict[str, str] = {
    "build_graph_index": ".builder",
    "build_from_rows": ".builder",
    "IndexBuildReport": ".builder",
    "parse_corpus": ".builder",
    "ParsedCorpus": ".builder",
    "BindingRow": ".builder",
    "ClassRow": ".builder",
    "ChannelRow": ".builder",
    "channels_from_corpus": ".builder",
    "channels_from_rows": ".builder",
    "open_graph_index": ".reader",
    "GraphIndex": ".reader",
    "GraphIndexAbsence": ".reader",
    "GraphIndexMeta": ".reader",
    "SCHEMA_VERSION": ".schema",
    "META_KEYS": ".schema",
    "create_tables": ".schema",
    "GraphIndexBuildError": "..core.exceptions",
    "class_name": ".taxonomy",
    "prune_device_taxonomy": ".taxonomy",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public name from its defining module on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
