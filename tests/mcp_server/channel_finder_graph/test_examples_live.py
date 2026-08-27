"""The channel-finder catalogue, run against real seeded stores.

``tests/mcp_server/channel_finder_graph/test_examples_data.py`` reads the
catalogue against the Turtle files: it proves every predicate a query names is
in the corpus and every parameter value occurs there verbatim.  That is a strong
static contract and it still cannot see the three ways a shipped example fails
an operator, because all three are properties of the *store* rather than of the
text:

* **The projection.**  Node labels, relationship-type casing and property
  spellings are produced by neosemantics from the Turtle, not written in it.  A
  query asking for ``:Device``, or for ``hasBinding`` instead of ``HASBINDING``,
  names tokens the corpus genuinely contains and matches nothing in the graph.
* **Direction.**  ``(d)-[:HASBINDING]->(b)`` and ``(b)-[:HASBINDING]->(d)`` are
  the same predicate to a file-level test and opposite walks to the store.  One
  of them returns rows; the other returns none, silently.
* **Composition.**  Two predicates that both exist can still never co-occur on
  one subject.  ``by_field_meaning`` requires a field description *and* a
  subfield description on the same binding; ``by_synonym`` requires a class with
  the synonym, a subclass chain down from it, a device typed into that chain and
  a binding hanging off the device — a four-way join whose emptiness no corpus
  census predicts.

Each of those returns zero rows and reads to a caller as "the machine has no
such channel", which is the single worst answer a channel finder can give.  So
this lane runs **every** example, with its shipped parameter set, through the
real ``read_cypher`` callable, and asserts each one comes back with at least one
row, no null column, and its own ``LIMIT`` still doing the bounding.  Going
through the tool rather than the driver is deliberate: the query gate vets the
text before the store is dialled, so a catalogue entry the gate would refuse
fails here rather than in an operator's session.

The corpus is seeded into a wiped store — the sequence ``seed-graph --force``
performs — and every example runs with its shipped parameter set.  What this
file does pin is that the lane ran a non-empty set: a catalogue that declared
nothing would otherwise pass by running nothing.

The container recipe (pinned image, n10s jar from the pinned release or
``OSPREY_TEST_N10S_JAR``, APOC copied out of the image) comes from
:mod:`tests._graphdb_container` rather than being copied; see that module's
docstring for why the container starts without ``NEO4J_PLUGINS``.  The store
fixture is module-scoped, not session-scoped: this module wipes the store
before seeding, and a container shared with the integration suite's
session-scoped fixture would hand that suite an emptied graph mid-run.  The jars
it mounts are resolved once per session by the shared ``graphdb_plugin_dir``
fixture, which is also the skip gate — without Docker, without the image, or
without the jar this module skips and says which.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

from osprey.mcp_server.channel_finder_graph.tools.examples_data import EXAMPLE_QUERIES
from tests._graphdb_container import (
    GRAPHDB_TEST_PASSWORD,
    GRAPHDB_TEST_USERNAME,
    graphdb_store,
)

logger = logging.getLogger(__name__)

# xdist_group("docker") keeps every container-starting module on one xdist
# worker: two testcontainers sessions starting at once race their own reapers
# over the port publisher.  ``integration`` because this needs a real service.
pytestmark = [pytest.mark.integration, pytest.mark.xdist_group("docker")]

DEMO = "demo"

#: The row cap this lane installs.  It sits above every example's own ``LIMIT``
#: (all of which are at most 200) and below the corpus' 2908 channel bindings,
#: which is what gives ``truncated`` its meaning here: an example that
#: reports truncation filled this cap, and the only way to do that is to have
#: lost the ``LIMIT`` the catalogue says every query carries.  Under the shipped
#: default cap the two would be indistinguishable — a query bounded at 200 and a
#: query bounded at nothing both come back as 200 rows.
#:
#: Deliberately not the shipped default of 200 (pinned in
#: ``tests/mcp_server/test_graph_server_context.py``), because the cap and the
#: catalogue are different claims: the cap bounds an answer, and this lane is
#: asking whether the query bounds itself.
LANE_ROW_CAP = 600


# ---------------------------------------------------------------------------
# Plugins + container
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def examples_store_uri(graphdb_plugin_dir: Path) -> Iterator[str]:
    """Start a throwaway graph store for this module and yield its bolt URI.

    Module-scoped rather than session-scoped: the test below wipes the store
    before seeding, and a store shared with another module's session-scoped
    fixture would hand that module an emptied graph halfway through its run.
    The plugins it mounts *are* session-scoped — see ``graphdb_plugin_dir``,
    which is also this module's skip gate.
    """
    with graphdb_store(graphdb_plugin_dir) as uri:
        yield uri


@pytest.fixture(scope="module")
def demo_ttl() -> str:
    """The generated demo corpus, read the way installed code reads it."""
    resource = (
        files("osprey.templates")
        .joinpath("apps")
        .joinpath("control_assistant")
        .joinpath("data")
        .joinpath("demo_machine.ttl")
    )
    return resource.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Seeding and the server context
# ---------------------------------------------------------------------------


def _seed(uri: str, ttl: str, label: str) -> None:
    """Wipe the store and load *ttl* into it the way ``seed-graph --force`` does.

    ``wipe`` takes the n10s graph config and the namespace prefixes with the
    data, so the corpus is bootstrapped into a store in the state a fresh
    deploy starts from.
    """
    from osprey.services.facility_knowledge.seeder import graph_seeder

    with graph_seeder.open_session(uri, GRAPHDB_TEST_USERNAME, GRAPHDB_TEST_PASSWORD) as session:
        graph_seeder.wipe(session)

        result = graph_seeder.bootstrap(session)
        assert result.status is graph_seeder.BootstrapStatus.INITIALIZED, result.message

        imported = graph_seeder.import_ttl(session, ttl)
        assert imported.termination_status == graph_seeder.TERMINATION_OK, (
            f"n10s refused the {label} corpus: {imported.extra_info}"
        )
        assert imported.triples_loaded == imported.triples_parsed, (
            f"n10s parsed {imported.triples_parsed} triples out of the {label} corpus "
            f"but committed {imported.triples_loaded}"
        )
        graph_seeder.write_marker(
            session,
            graph_seeder.ttl_sha256(ttl),
            graph_seeder.parse_direction_source(ttl),
        )
        logger.info(
            f"{label}: seeded {imported.triples_loaded} triples, "
            f"{graph_seeder.resource_count(session)} Resource nodes"
        )


@contextmanager
def _installed_context(uri: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Install a ``GraphContext`` singleton pointed at *uri*.

    The two config seams ``initialize()`` reads through are replaced rather than
    a config file written: what this module is about is the store, and a
    fabricated ``config.yml`` would only re-test ``resolve_graphdb_connection``.
    The password travels the way a deployment delivers it — through
    ``GRAPHDB_PASSWORD`` in the environment — and is never logged or
    interpolated into the URI.
    """
    from osprey.deployment.graphdb_service import GRAPHDB_PASSWORD_ENV
    from osprey.mcp_server.graph import server_context as graph_ctx

    monkeypatch.setenv(GRAPHDB_PASSWORD_ENV, GRAPHDB_TEST_PASSWORD)
    monkeypatch.setattr(
        graph_ctx, "load_osprey_config", lambda: {"services": {"graphdb": {"uri": uri}}}
    )
    monkeypatch.setattr(graph_ctx, "get_config_value", lambda key, default=None: default)

    context = graph_ctx.initialize_server_context()
    try:
        # Inside the try: initialize() has already installed the singleton, so a
        # failed assertion here must still tear it down or every later test
        # inherits a context pointed at a stopped container.
        assert context.configured, (
            "the graph context did not read the container as a configured store"
        )
        context.query_max_rows = LANE_ROW_CAP
        yield context
    finally:
        graph_ctx.reset_server_context()


def _read_cypher(query: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """Run one query through the channel finder's own ``read_cypher`` callable.

    Imported from this package rather than from the main-agent graph server so
    the re-registration under the ``channel-finder`` server name is part of what
    runs: the tool an operator reaches is the one being exercised.  A gate
    refusal raises ``ToolError`` out of here, which is the loud failure a
    catalogue entry the gate would not pass deserves.
    """
    from osprey.mcp_server.channel_finder_graph.tools.read_cypher import read_cypher

    fn = getattr(read_cypher, "fn", read_cypher)
    return json.loads(fn(query, params))


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------


def _applicable() -> list[Any]:
    """Every catalogue example, checked to carry a runnable parameter set.

    Asserted rather than filtered: an example without values would otherwise
    drop out of this lane without anything saying so.
    """
    for example in EXAMPLE_QUERIES:
        assert example.parameters, (
            f"{example.key} ships no parameter values; an example this lane "
            "cannot run is one an operator cannot run either"
        )
    return list(EXAMPLE_QUERIES)


def _usability_failures(example: Any, corpus: str, payload: dict[str, Any]) -> list[str]:
    """Every way *payload* falls short of being an answer, as prose lines.

    Returned rather than asserted so one run reports every broken example
    instead of the first.  A lane that stopped at the first failure would be
    walked one example per run, and each of those runs pays for a container and
    a re-import of the corpus.
    """
    failures: list[str] = []
    if payload.get("row_count", 0) < 1:
        failures.append(
            f"{example.key} on {corpus}: returned no rows with its shipped parameter set "
            f"{example.parameters!r}. To a caller that reads as 'the machine has no "
            f"such channel'."
        )
        return failures

    if payload.get("truncated"):
        failures.append(
            f"{example.key} on {corpus}: filled the lane's {LANE_ROW_CAP}-row cap, so its own "
            f"LIMIT did not bound the answer. Every catalogue query is supposed to end in one, "
            f"and an unbounded example is how a caller gets a truncated result it did not ask for."
        )

    for row in payload["rows"]:
        for column, value in row.items():
            if value is None:
                failures.append(f"{example.key} on {corpus}: null {column!r} in row {row!r}")
    return failures


def _run_catalogue(corpus: str) -> None:
    """Run every example against the seeded store and report all the broken ones."""
    examples = _applicable()
    assert examples, "an empty catalogue would let this lane pass having run nothing"

    failures: list[str] = []
    for example in examples:
        payload = _read_cypher(example.cypher, dict(example.parameters))
        failures.extend(_usability_failures(example, corpus, payload))

    assert not failures, (
        f"{len(failures)} problem(s) across {len(examples)} example(s) on the "
        f"{corpus} corpus:\n" + "\n".join(f"  - {line}" for line in failures)
    )


def test_every_demo_example_answers_on_the_demo_corpus(
    examples_store_uri: str, demo_ttl: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seed the generated demo corpus and run every example declared for it.

    One test over one seeded store rather than one per example: every example
    below is a question about the *same* graph, so a parametrized lane would
    re-import 2.7 MB of Turtle per question and prove nothing extra.
    ``_usability_failures`` keeps the per-example detail that parametrizing
    would otherwise have bought.

    Both halves of the catalogue run here: the prose searches — description,
    field/subfield and system — which the demo corpus carries because it is
    generated from a channel database with prose, and the structural ones —
    synonym lookup, section listing, a device's addresses, an address back to
    its device.
    """
    _seed(examples_store_uri, demo_ttl, DEMO)
    with _installed_context(examples_store_uri, monkeypatch):
        _run_catalogue(DEMO)
