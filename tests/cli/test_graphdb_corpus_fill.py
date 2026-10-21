"""Tests for :func:`~osprey.cli.build_injectors.graphdb_corpus_fill`.

A render that carries a ``services.graphdb`` block seeds its store from the
graph view the build writes, so the corpus key is filled whenever the block is
there and the profile left the key unspelled. A profile that spells it keeps
its own value; a profile with no block gets nothing.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.cli.build_injectors import graphdb_corpus_fill
from osprey.deployment.graphdb_service import DEFAULT_TTL_PATH, GRAPHDB_TTL_PATH_CONFIG_KEY


def test_the_corpus_is_the_graph_view_the_build_writes() -> None:
    assert DEFAULT_TTL_PATH == "./data/graph/facility.ttl"


@pytest.mark.parametrize(
    "config",
    [
        {"services.graphdb.path": "./services/graphdb"},
        {"services.graphdb.uri": "bolt://graph.example.org:7687"},
        {"services": {"graphdb": {"path": "./services/graphdb"}}},
        {"services": {"graphdb.image": "neo4j:5.26-community"}},
    ],
    ids=["dotted", "external-store", "nested", "mixed"],
)
def test_a_graphdb_block_gets_the_derived_corpus(config: dict[str, Any]) -> None:
    assert graphdb_corpus_fill(config) == {GRAPHDB_TTL_PATH_CONFIG_KEY: DEFAULT_TTL_PATH}


@pytest.mark.parametrize(
    "config",
    [
        {"services.graphdb.path": "./services/graphdb", "services.graphdb.ttl_path": "./x.ttl"},
        {"services": {"graphdb": {"path": "./services/graphdb", "ttl_path": "./x.ttl"}}},
    ],
    ids=["dotted", "nested"],
)
def test_a_spelled_corpus_is_left_alone(config: dict[str, Any]) -> None:
    assert graphdb_corpus_fill(config) == {}


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"services.qmd.path": "./services/qmd"},
        {"services.graphdb": None},
    ],
    ids=["empty", "other-service", "removed-block"],
)
def test_no_graphdb_block_gets_nothing(config: dict[str, Any]) -> None:
    assert graphdb_corpus_fill(config) == {}
