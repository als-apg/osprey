"""E2E benchmark tests for the channel-finder MCP server.

Thin pytest wrapper over ``BenchmarkRunner`` that exercises each of the four
channel-finder paradigms (hierarchical, middle_layer, in_context, graph)
against the preset's channel corpus. Each build materializes the tier-resolved
unified query set into the render (``build/data/benchmarks/queries.json``) —
one source of truth, paradigm-resolved at build time — so this test reads that
file and slices the first ``SLICE_SIZE`` queries.

The runner, queries, and evaluator all live in
``osprey.services.channel_finder.benchmarks``; this file only wires them into
pytest, parameterizes per-query result inspection, and asserts aggregate
quality thresholds.

Three of the four paradigms read a database file the build renders beside the
config, so a lane is one ``init_project`` call. The fourth, ``graph``, searches
a Neo4j store, so its lane stands one up: a throwaway container on a random
port, the render pointed at it, and the shipped corpus seeded through ``osprey
knowledge seed-graph``. The recipe is ``tests/e2e/test_graph_mcp_smoke.py``'s,
imported rather than copied. That lane therefore skips — loudly, with its
reason — on a host with no Docker daemon, while the other three run.

Skip-gating mirrors the safety/SDK suite in this directory: ``pytestmark`` is
module-local in pytest, so conftest's gates do not cascade — each test file
must declare its own.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from osprey.agent_runner import expected_mcp_servers
from osprey.services.channel_finder.benchmarks.models import BenchmarkRun, QueryResult
from osprey.services.channel_finder.benchmarks.runner import BenchmarkRunner
from tests._container_support import (
    is_docker_available,
    is_image_present,
    start_or_fail,
    stop_quietly,
)

# The container recipe (image, jar version, throwaway password, plugin
# helpers) is the shared one in tests/_graphdb_container.py; the two patches
# that point a build at the store stay the graph MCP acceptance test's —
# imported, not re-spelled, so there is exactly one spelling of each.
from tests._graphdb_container import (
    GRAPHDB_TEST_PASSWORD,
    NEO4J_IMAGE,
    _copy_bundled_apoc,
    _fetch_n10s_jar,
)
from tests.e2e.sdk_helpers import (
    HAS_SDK,
    init_project,
    render_dir,
)
from tests.e2e.test_graph_mcp_smoke import (
    BOLT_PORT,
    _point_project_at_the_store,
    _seed_demo_corpus,
)

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.channel_finder_benchmark,
    pytest.mark.slow,
    pytest.mark.requires_als_apg,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
]

PARADIGMS = ("hierarchical", "middle_layer", "in_context", "graph")
SLICE_SIZE = 10
F1_THRESHOLD = 0.75
PERFECT_THRESHOLD = 0.80

#: Channel bindings in the shipped demo corpus — what ``osprey knowledge
#: seed-graph`` puts in the store and what the graph lane's census must find.
#: The same 2,908 addresses the tier-3 database files hold, which is what makes
#: the four lanes comparable: one corpus, four ways of searching it.
#: ``tests/templates/test_control_assistant_demo_ttl.py`` owns the corpus-side
#: pin; this is the benchmark-side one.
GRAPH_CHANNEL_COUNT = 2908


def perfect_match_rate(run: BenchmarkRun) -> float:
    if not run.query_results:
        return 0.0
    return sum(1 for r in run.query_results if r.f1 == 1.0) / len(run.query_results)


def _assert_honest(run: BenchmarkRun) -> None:
    """Fail the lane unless every query in the slice produced a result.

    ``BenchmarkRunner`` catches per-query exceptions, logs them, drops the query
    and records the count in ``num_failed``; the aggregates are then a mean over
    the survivors only. So a lane whose hardest queries all errored out can post
    a high ``aggregate_f1`` and a high perfect-match rate over the handful that
    finished. Every aggregate lane calls this *first*, so a benchmark result is
    only ever read as a score once it is known to cover the whole slice.
    """
    assert run.num_failed == 0, f"{run.num_failed} queries failed"


def _resolve_dataset_path(render: Path) -> Path:
    """The materialized query set, resolved against the RENDER.

    ``dataset_path`` is configured relative (``data/benchmarks/queries.json``)
    and the file is a build output, so it resolves against the directory holding
    the rendered ``config.yml`` — not the repo's source ``data/`` tree.
    """
    config = yaml.safe_load((render / "config.yml").read_text(encoding="utf-8"))
    rel = config["channel_finder"]["benchmark"]["dataset_path"]
    path = Path(rel)
    return path if path.is_absolute() else render / path


NEAR_MISS_CATEGORY = "near-miss"


def _slice_indices(dataset_path: Path) -> list[int]:
    """Deterministic, category-stratified slice with a reserved near-miss slot.

    Slot 1 is always a discrimination (``near-miss``) query when the dataset
    ships one, guaranteeing a distractor-family probe (QFA/SHF/SHD/SD) executes
    in the gate. The remaining slots round-robin over the *other* categories in
    sorted order — one query per category per round, in file order within a
    category — until ``SLICE_SIZE`` is reached. With more categories than slots
    (tier 3: 11 categories, 10 slots) exactly one non-near-miss category goes
    unsampled per run; with fewer (tier 1: 6 categories) every category is
    covered. A final file-order fallback fills any slots a degenerate dataset
    (e.g. near-miss only) leaves open, so the slice always reaches
    ``min(SLICE_SIZE, len(queries))`` indices with no duplicates.
    """
    queries = json.loads(dataset_path.read_text(encoding="utf-8"))
    limit = min(SLICE_SIZE, len(queries))

    by_category: dict[str, list[int]] = {}
    for idx, query in enumerate(queries):
        by_category.setdefault(query.get("category", ""), []).append(idx)

    selected: list[int] = []
    seen: set[int] = set()

    def take(idx: int) -> None:
        if idx not in seen:
            seen.add(idx)
            selected.append(idx)

    # Reserve slot 1 for a discrimination query when the dataset ships one.
    near_miss = by_category.get(NEAR_MISS_CATEGORY, [])
    if near_miss:
        take(near_miss[0])

    # Round-robin the remaining categories (near-miss excluded so it cannot
    # crowd out breadth) in sorted order to fill the slice.
    pools = [by_category[cat] for cat in sorted(by_category) if cat != NEAR_MISS_CATEGORY]
    depth = max((len(pool) for pool in pools), default=0)
    for round_idx in range(depth):
        if len(selected) >= limit:
            break
        for pool in pools:
            if round_idx < len(pool):
                take(pool[round_idx])
                if len(selected) >= limit:
                    break

    # Fallback: fill any remaining slots from unused indices in file order.
    if len(selected) < limit:
        for idx in range(len(queries)):
            take(idx)
            if len(selected) >= limit:
                break

    return selected


def _run_slice(render: Path, output_root: Path) -> BenchmarkRun:
    """Run the stratified slice against a built project; return the run.

    Every lane uses the same runner settings, so a difference between two lanes
    is a difference between two paradigms and nothing else.
    """
    indices = _slice_indices(_resolve_dataset_path(render))
    runner = BenchmarkRunner(
        render,
        model="als-apg/claude-haiku-4-5-20251001",
        max_concurrent=3,
        max_budget_per_query=0.20,
        use_llm_judge=True,
    )
    return asyncio.run(
        runner.run_queries(
            query_indices=indices,
            output_dir=output_root / "benchmark_results",
        )
    )


def _run_paradigm_benchmark(tmp_path_factory, paradigm: str) -> BenchmarkRun:
    tmp = tmp_path_factory.mktemp(f"cf-bench-{paradigm}")
    # Build with provider=als-apg so the gate (ALS_APG_API_KEY) lines up with
    # the routing the test will actually exercise. The runner picks up the
    # provider/wire-id mapping from the rendered config.yml — no model knob
    # to plumb through this fixture.
    repo = init_project(
        tmp,
        f"cf-bench-{paradigm}",
        provider="als-apg",
        channel_finder_mode=paradigm,
    )
    # The runner reads ``<dir>/config.yml`` and runs with its cwd there, so it
    # takes the render.
    return _run_slice(render_dir(repo), tmp)


@pytest.fixture
def hierarchical_run(tmp_path_factory) -> BenchmarkRun:
    return _run_paradigm_benchmark(tmp_path_factory, "hierarchical")


@pytest.fixture
def middle_layer_run(tmp_path_factory) -> BenchmarkRun:
    return _run_paradigm_benchmark(tmp_path_factory, "middle_layer")


@pytest.fixture
def in_context_run(tmp_path_factory) -> BenchmarkRun:
    return _run_paradigm_benchmark(tmp_path_factory, "in_context")


# ---------------------------------------------------------------------------
# The fourth lane: graph. Same query set, same thresholds, same runner — but a
# store instead of a database file, so the fixture chain stands one up.
# ---------------------------------------------------------------------------


TIER_BOUNDARY_CATEGORY = "tier-boundary"


def _assert_graph_pipeline_is_rendered(repo: Path) -> None:
    """Fail loudly if the build did not put a *graph* channel finder in the render.

    Two claims, and the lane is vacuous without either:

    * ``.mcp.json`` declares ``channel-finder``. The benchmark's SDK harness
      selects its server by that name (a substring match on the key) and runs
      with ``allowed_tools=['mcp__channel-finder__*']``, so a render that named
      the graph surface anything else would leave the agent with no tools at
      all — and this lane would grade an agent answering from its own priors.
    * The tools granted include ``read_cypher``. That is the graph vocabulary
      and no file paradigm has it, so it is what distinguishes "the graph
      channel finder was rendered" from "a channel finder was rendered".
    """
    render = render_dir(repo)
    servers = expected_mcp_servers(render)
    assert "channel-finder" in servers, (
        f"the graph-mode build rendered no 'channel-finder' MCP server — .mcp.json "
        f"declares {sorted(servers)}. The benchmark harness selects its server by "
        "that name, so the agent would run this lane with no channel-finder tools."
    )
    settings = json.loads((render / ".claude" / "settings.json").read_text(encoding="utf-8"))
    granted = [
        tool
        for tool in settings["permissions"]["allow"]
        if tool.startswith("mcp__channel-finder__")
    ]
    assert "mcp__channel-finder__read_cypher" in granted, (
        f"the graph-mode build granted {granted}, which is not the graph tool "
        "vocabulary — this lane would be scoring a different paradigm."
    )


@pytest.fixture(scope="module")
def graph_bench_plugin_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """n10s + APOC for the lane's store, resolved once.

    The lane's Docker gate lives here — first thing the store depends on — so
    an unreachable daemon is reported once with its reason instead of as an
    unexplained skip on the benchmark itself. The other three lanes need no
    container and are unaffected, which is why this gate is per-fixture rather
    than module-level ``pytestmark``.
    """
    if not is_docker_available():
        pytest.skip(
            "docker daemon is not reachable — the graph benchmark lane can only "
            "be scored against a real Neo4j + neosemantics store"
        )

    if not is_image_present(NEO4J_IMAGE):
        # Pull explicitly: the APOC copy reads the image directly and, unlike
        # ``containers.run``, ``containers.create`` does not pull for us.
        import docker

        logger.info(f"pulling {NEO4J_IMAGE} (not present locally)")
        try:
            docker.from_env().images.pull(NEO4J_IMAGE)
        except Exception as exc:
            pytest.skip(f"could not pull {NEO4J_IMAGE}: {exc}")

    plugin_dir = tmp_path_factory.mktemp("cf-bench-graph-plugins")
    _fetch_n10s_jar(plugin_dir)
    _copy_bundled_apoc(plugin_dir)
    # The server runs as a non-root user and only ever reads these.
    for jar in plugin_dir.glob("*.jar"):
        jar.chmod(0o644)
    plugin_dir.chmod(0o755)
    return plugin_dir


@pytest.fixture(scope="module")
def graph_bench_store_port(graph_bench_plugin_dir: Path) -> Iterator[int]:
    """A throwaway graph store for this lane; yields its published host port.

    Random port, generated container name: this cannot collide with a
    ``graphdb`` service deployed on the same host, nor with the graph MCP
    acceptance test's own store if both modules run in one session.
    """
    try:
        from testcontainers.community.neo4j import Neo4jContainer
    except ImportError:  # pragma: no cover - depends on the installed extras
        pytest.skip("testcontainers' neo4j module is not installed")

    def _build() -> Neo4jContainer:
        container = Neo4jContainer(image=NEO4J_IMAGE, password=GRAPHDB_TEST_PASSWORD)
        container.with_volume_mapping(str(graph_bench_plugin_dir), "/plugins", "rw")
        # n10s calls into APOC, so both need the unrestricted grant; without
        # the allowlist every n10s.* call fails with "not on the allowlist".
        container.with_env("NEO4J_dbms_security_procedures_unrestricted", "apoc.*,n10s.*")
        container.with_env("NEO4J_dbms_security_procedures_allowlist", "apoc.*,n10s.*")
        return container

    container, port = start_or_fail(_build, "graphdb (neo4j + n10s)", BOLT_PORT)
    logger.info(f"graph benchmark store published bolt on host port {port}")
    try:
        yield port
    finally:
        stop_quietly(container)


@pytest.fixture(scope="module")
def graph_bench_project(
    graph_bench_store_port: int, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Path]:
    """A built graph-mode project wired to the seeded store; yields the repo root.

    Module-scoped: the build renders container contexts and three persona
    projects, and the seed loads the whole corpus, so a ``@flaky`` rerun should
    re-ask the ten questions rather than rebuild and re-seed. Nothing in this
    lane writes to the store, so every attempt sees the same corpus. The run
    itself is function-scoped, so a rerun is still a fresh draw.

    ``GRAPHDB_PASSWORD`` is pinned in **this** process's environment as well as
    in the project's ``.env``, and here that is not merely belt-and-braces: the
    benchmark runner counts the store's channels itself, in-process, through
    ``resolve_graphdb_connection``, which reads the ambient environment. Without
    the pin a developer's exported password for some other deployment would be
    what the census authenticates with. A module-scoped
    :class:`pytest.MonkeyPatch` is used because the ``monkeypatch`` fixture is
    function-scoped and cannot be requested here.
    """
    tmp = tmp_path_factory.mktemp("cf-bench-graph")
    monkeypatch = pytest.MonkeyPatch()
    repo = init_project(
        tmp,
        "cf-bench-graph",
        provider="als-apg",
        channel_finder_mode="graph",
    )
    try:
        monkeypatch.setenv("GRAPHDB_PASSWORD", GRAPHDB_TEST_PASSWORD)
        _assert_graph_pipeline_is_rendered(repo)
        _point_project_at_the_store(repo, graph_bench_store_port)
        logger.info("seed-graph: %s", _seed_demo_corpus(repo).strip().replace("\n", " | "))
        yield repo
    finally:
        monkeypatch.undo()
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def graph_run(graph_bench_project: Path, tmp_path: Path) -> BenchmarkRun:
    return _run_slice(render_dir(graph_bench_project), tmp_path)


def annotate_categories(run: BenchmarkRun, dataset_path: Path) -> dict[int, str]:
    """Label each result with the query's category and log the composition.

    The tier-3 query set carries a ``tier-boundary`` category: questions whose
    answers straddle what a tier-1 corpus holds and what a tier-3 one does.
    They are scored exactly like every other query — this annotation changes no
    number and drops nothing from the aggregate. What it does is make a miss
    *legible*: a graph lane that loses a tier-boundary query has said something
    different about the paradigm than one that loses a near-miss query, and
    without the label both arrive as an unexplained low F1.

    The category is written into each result's ``eval_meta`` so it travels with
    the run object a caller inspects after the lane, and the per-category
    composition is logged.

    Args:
        run: The completed run.
        dataset_path: The query set the run's ``query_id`` values index into.

    Returns:
        ``query_id`` -> category, for the queries the run actually scored.
    """
    queries = json.loads(dataset_path.read_text(encoding="utf-8"))
    labelled: dict[int, str] = {}
    for result in run.query_results:
        if not 0 <= result.query_id < len(queries):  # pragma: no cover - defensive
            continue
        category = str(queries[result.query_id].get("category", ""))
        result.eval_meta["category"] = category
        labelled[result.query_id] = category

    by_category: dict[str, list[float]] = {}
    for result in run.query_results:
        category = labelled.get(result.query_id)
        if category is not None:
            by_category.setdefault(category, []).append(result.f1)
    logger.info(
        "%s slice categories: %s",
        run.paradigm,
        ", ".join(
            f"{category}={sum(scores) / len(scores):.2f}(n={len(scores)})"
            for category, scores in sorted(by_category.items())
        ),
    )
    boundary = sorted(qid for qid, cat in labelled.items() if cat == TIER_BOUNDARY_CATEGORY)
    logger.info(
        "%s tier-boundary queries in this slice: %s",
        run.paradigm,
        boundary or "none (the stratified slice reserves its slots for other categories)",
    )
    return labelled


# ---------------------------------------------------------------------------
# Aggregate tests — load-bearing assertions per project_channel_finder_
# benchmark_thresholds memory: target ~90%, assert at 80%; on a miss,
# investigate dataset/DB drift, do not lower the threshold.
#
# Per-query F1 is surfaced via review.html / BenchmarkRun output files; it is
# intentionally NOT asserted in pytest. Earlier `f1 >= 0.0` per-query asserts
# were tautological and created false-positive CI signal.
#
# reruns=2: the run is a real LLM agent, so the perfect-match rate has draw
# variance around the 0.80 bar. The `*_run` fixtures are function-scoped so each
# rerun re-executes the benchmark (a fresh draw); CI only turns red if all three
# attempts miss. The threshold is unchanged — a genuine (reproducible) regression
# still fails every attempt; only single-draw variance is absorbed.
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=2)
def test_hierarchical_aggregate(hierarchical_run: BenchmarkRun) -> None:
    _assert_honest(hierarchical_run)
    assert hierarchical_run.aggregate_f1 >= F1_THRESHOLD, (
        f"aggregate_f1={hierarchical_run.aggregate_f1:.3f} below {F1_THRESHOLD}"
    )
    pmr = perfect_match_rate(hierarchical_run)
    assert pmr >= PERFECT_THRESHOLD, f"perfect_match_rate={pmr:.3f} below {PERFECT_THRESHOLD}"


@pytest.mark.flaky(reruns=2)
def test_middle_layer_aggregate(middle_layer_run: BenchmarkRun) -> None:
    _assert_honest(middle_layer_run)
    assert middle_layer_run.aggregate_f1 >= F1_THRESHOLD, (
        f"aggregate_f1={middle_layer_run.aggregate_f1:.3f} below {F1_THRESHOLD}"
    )
    pmr = perfect_match_rate(middle_layer_run)
    assert pmr >= PERFECT_THRESHOLD, f"perfect_match_rate={pmr:.3f} below {PERFECT_THRESHOLD}"


@pytest.mark.flaky(reruns=2)
def test_in_context_aggregate(in_context_run: BenchmarkRun) -> None:
    _assert_honest(in_context_run)
    assert in_context_run.aggregate_f1 >= F1_THRESHOLD, (
        f"aggregate_f1={in_context_run.aggregate_f1:.3f} below {F1_THRESHOLD}"
    )
    pmr = perfect_match_rate(in_context_run)
    assert pmr >= PERFECT_THRESHOLD, f"perfect_match_rate={pmr:.3f} below {PERFECT_THRESHOLD}"


@pytest.mark.flaky(reruns=2)
def test_graph_aggregate(graph_run: BenchmarkRun, graph_bench_project: Path) -> None:
    """The graph paradigm, scored against the same ground truth at the same bar.

    Parity is the claim: same tier-3 query set, same stratified slice, same
    judge, same two thresholds as the three file paradigms. A graph lane that
    needed its own thresholds would not be evidence that the paradigm works,
    only that it was graded gently.

    ``channel_count`` is asserted rather than merely recorded because it is the
    one number that says the agent searched the corpus this lane thinks it did.
    The graph census reaches the store and lets its errors propagate, so an
    unreachable store fails here with the store's own error — it can never
    arrive as a quiet ``channel_count`` of 0 beside real-looking scores.
    """
    _assert_honest(graph_run)
    render = render_dir(graph_bench_project)
    annotate_categories(graph_run, _resolve_dataset_path(render))

    assert graph_run.channel_count == GRAPH_CHANNEL_COUNT, (
        f"the graph store holds {graph_run.channel_count} channel bindings, "
        f"not the {GRAPH_CHANNEL_COUNT} the shipped corpus seeds — the lane "
        "scored a different corpus than the other three paradigms search."
    )
    assert graph_run.aggregate_f1 >= F1_THRESHOLD, (
        f"aggregate_f1={graph_run.aggregate_f1:.3f} below {F1_THRESHOLD}"
    )
    pmr = perfect_match_rate(graph_run)
    assert pmr >= PERFECT_THRESHOLD, f"perfect_match_rate={pmr:.3f} below {PERFECT_THRESHOLD}"


# ---------------------------------------------------------------------------
# Hermetic unit tests for the slice selector. Pure function, no live run — they
# pin the reserved near-miss slot and the deterministic stratification so a
# discrimination query is guaranteed in the gate slice by construction. Live
# scoring happens in the aggregate lanes above.
# ---------------------------------------------------------------------------


def _write_dataset(tmp_path: Path, categories: list[tuple[str, int]]) -> tuple[Path, list[dict]]:
    """Write a synthetic queries file; return (path, queries) in file order."""
    queries: list[dict] = []
    for category, count in categories:
        for k in range(count):
            queries.append(
                {
                    "user_query": f"{category}-{k}",
                    "targeted_pv": [f"PV:{category}:{k:02d}"],
                    "category": category,
                }
            )
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(queries), encoding="utf-8")
    return path, queries


# An 11-category shape mirroring the shipped tier-3 query set (near-miss is the
# largest category since it holds the discrimination queries).
_TIER3_SHAPE = [
    ("aggregate", 4),
    ("ambiguous", 3),
    ("cross-ring", 5),
    ("device-specific", 7),
    ("multi-target", 7),
    ("near-miss", 11),
    ("range", 4),
    ("sector-based", 3),
    ("semantic", 3),
    ("single-target", 9),
    ("tier-boundary", 4),
]

# A 6-category shape mirroring the leaner tier-1 query set.
_TIER1_SHAPE = [
    ("aggregate", 3),
    ("device-specific", 4),
    ("multi-target", 3),
    ("near-miss", 3),
    ("range", 3),
    ("single-target", 4),
]


def test_slice_reserves_near_miss_slot(tmp_path: Path) -> None:
    path, queries = _write_dataset(tmp_path, _TIER3_SHAPE)
    indices = _slice_indices(path)

    assert queries[indices[0]]["category"] == NEAR_MISS_CATEGORY
    # Exactly one near-miss query is reserved — it must not crowd out breadth.
    near_miss_hits = [i for i in indices if queries[i]["category"] == NEAR_MISS_CATEGORY]
    assert len(near_miss_hits) == 1


def test_slice_is_full_and_unique(tmp_path: Path) -> None:
    path, _ = _write_dataset(tmp_path, _TIER3_SHAPE)
    indices = _slice_indices(path)

    assert len(indices) == SLICE_SIZE
    assert len(set(indices)) == len(indices)


def test_slice_tier3_leaves_one_category_unsampled(tmp_path: Path) -> None:
    path, queries = _write_dataset(tmp_path, _TIER3_SHAPE)
    indices = _slice_indices(path)

    sampled = {queries[i]["category"] for i in indices}
    all_categories = {cat for cat, _ in _TIER3_SHAPE}
    # 11 categories, 10 slots, near-miss reserved -> exactly one non-near-miss
    # category goes unsampled.
    assert NEAR_MISS_CATEGORY in sampled
    unsampled = all_categories - sampled
    assert unsampled == {"tier-boundary"}, unsampled


def test_slice_tier1_covers_every_category(tmp_path: Path) -> None:
    path, queries = _write_dataset(tmp_path, _TIER1_SHAPE)
    indices = _slice_indices(path)

    sampled = {queries[i]["category"] for i in indices}
    assert sampled == {cat for cat, _ in _TIER1_SHAPE}
    assert len(indices) == SLICE_SIZE


def test_slice_is_deterministic(tmp_path: Path) -> None:
    path, _ = _write_dataset(tmp_path, _TIER3_SHAPE)
    assert _slice_indices(path) == _slice_indices(path)


def test_slice_without_near_miss_category(tmp_path: Path) -> None:
    shape = [(cat, cnt) for cat, cnt in _TIER3_SHAPE if cat != NEAR_MISS_CATEGORY]
    path, queries = _write_dataset(tmp_path, shape)
    indices = _slice_indices(path)

    assert len(indices) == SLICE_SIZE
    assert len(set(indices)) == len(indices)
    assert all(queries[i]["category"] != NEAR_MISS_CATEGORY for i in indices)


# ---------------------------------------------------------------------------
# Hermetic unit tests for the honesty gate. They pin ``_assert_honest`` on
# synthetic runs so the guard the live lanes depend on is itself proven without
# spending a real benchmark.
# ---------------------------------------------------------------------------


def _synthetic_run(*, num_results: int, num_failed: int) -> BenchmarkRun:
    """A BenchmarkRun with perfect scores over ``num_results`` surviving queries."""
    results = [
        QueryResult(
            query_id=i,
            user_query=f"q-{i}",
            expected=[f"PV:{i:02d}"],
            predicted=[f"PV:{i:02d}"],
            precision=1.0,
            recall=1.0,
            f1=1.0,
        )
        for i in range(num_results)
    ]
    return BenchmarkRun.from_query_results(
        model="synthetic",
        results=results,
        num_failed=num_failed,
    )


def test_assert_honest_accepts_a_complete_run() -> None:
    run = _synthetic_run(num_results=SLICE_SIZE, num_failed=0)

    _assert_honest(run)


def test_assert_honest_rejects_a_run_with_dropped_queries() -> None:
    # A single dropped query still leaves a perfect mean over the survivors —
    # exactly the inflated score the gate exists to catch.
    run = _synthetic_run(num_results=SLICE_SIZE - 1, num_failed=1)
    assert run.aggregate_f1 == 1.0
    assert perfect_match_rate(run) == 1.0

    with pytest.raises(AssertionError, match="1 queries failed"):
        _assert_honest(run)


def test_slice_smaller_than_slice_size(tmp_path: Path) -> None:
    path, queries = _write_dataset(tmp_path, [("near-miss", 2), ("range", 3)])
    indices = _slice_indices(path)

    assert len(indices) == len(queries)
    assert len(set(indices)) == len(indices)
    assert queries[indices[0]]["category"] == NEAR_MISS_CATEGORY


# ---------------------------------------------------------------------------
# Hermetic unit tests for the graph lane — the two claims that would otherwise
# only be provable by spending a container and a real benchmark: that the
# category annotation labels without excluding, and that a lane whose store is
# gone fails with the store's own error instead of recording an empty corpus.
# ---------------------------------------------------------------------------


def _run_with_categories(*, f1_by_index: dict[int, float]) -> BenchmarkRun:
    """A BenchmarkRun whose results index into a dataset written by the caller."""
    results = [
        QueryResult(
            query_id=index,
            user_query=f"q-{index}",
            expected=[f"PV:{index:02d}"],
            predicted=[f"PV:{index:02d}"] if f1 == 1.0 else [],
            precision=f1,
            recall=f1,
            f1=f1,
        )
        for index, f1 in sorted(f1_by_index.items())
    ]
    return BenchmarkRun.from_query_results(model="synthetic", results=results, paradigm="graph")


def test_annotate_categories_labels_the_slice(tmp_path: Path) -> None:
    path, queries = _write_dataset(tmp_path, _TIER3_SHAPE)
    indices = _slice_indices(path)
    run = _run_with_categories(f1_by_index=dict.fromkeys(indices, 1.0))

    labelled = annotate_categories(run, path)

    assert labelled == {i: queries[i]["category"] for i in indices}
    # The label travels with the run, not just with the log line.
    assert all(r.eval_meta["category"] == labelled[r.query_id] for r in run.query_results)


def test_annotate_categories_keeps_tier_boundary_in_the_slice(tmp_path: Path) -> None:
    """A tier-boundary query is annotated and still scored, never set aside."""
    path, queries = _write_dataset(tmp_path, _TIER3_SHAPE)
    boundary = next(i for i, q in enumerate(queries) if q["category"] == TIER_BOUNDARY_CATEGORY)
    other = next(i for i, q in enumerate(queries) if q["category"] == NEAR_MISS_CATEGORY)
    # The tier-boundary query is the one this run got wrong.
    run = _run_with_categories(f1_by_index={boundary: 0.0, other: 1.0})

    labelled = annotate_categories(run, path)

    assert labelled[boundary] == TIER_BOUNDARY_CATEGORY
    # Scored, not excused: it is still in the denominator, so the aggregate is
    # the mean over both queries rather than the survivor's perfect score.
    assert run.aggregate_f1 == 0.5
    assert perfect_match_rate(run) == 0.5


def _closed_local_port() -> int:
    """A loopback port with nothing listening on it."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_graph_lane_fails_loud_when_the_store_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the store stopped the lane raises; it never records ``channel_count`` 0.

    This is the counterfactual behind ``test_graph_aggregate``'s
    ``channel_count`` assertion. The file paradigms treat an unreadable
    database as a count of zero on purpose — the number labels a run, it does
    not score one, and a missing file must not take a benchmark down. The graph
    census cannot afford that: a zero saved beside real scores would claim the
    agent searched an empty corpus and found things in it.

    Hermetic by construction: a config naming a loopback port with nothing
    listening is the same thing the container fixture's store being stopped
    would produce, and it costs no Docker.
    """
    port = _closed_local_port()
    monkeypatch.setenv("GRAPHDB_PASSWORD", GRAPHDB_TEST_PASSWORD)
    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {
                "channel_finder": {
                    "pipeline_mode": "graph",
                    "benchmark": {"dataset_path": "queries.json"},
                },
                "services": {"graphdb": {"port_host": port}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "queries.json").write_text("[]", encoding="utf-8")

    runner = BenchmarkRunner(tmp_path, model="als-apg/claude-haiku-4-5-20251001")

    with pytest.raises(Exception) as raised:
        asyncio.run(runner.run_queries(query_indices=[]))

    # The store's own error, naming the address nobody answered on.
    assert str(port) in str(raised.value), raised.value
