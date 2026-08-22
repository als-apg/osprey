"""Agentic e2e acceptance gate for the **graph channel-finder paradigm**.

Sibling to :mod:`tests.e2e.test_graph_mcp_smoke`, and deliberately the other
half of the same story. That module proves the *main agent's* ``graph`` server
answers a structural question about the machine. This one proves the **fourth
channel-finder paradigm**: a deployment built with
``channel_finder_mode: graph`` answers an ordinary "which channel do I write
to?" question through the **channel-finder subagent**, whose four tools are the
read-only Cypher vocabulary shipped under the ``channel-finder`` server name.

Three claims, in the order the chain makes them:

1. **The render selects the paradigm's server.** ``osprey init --set
   channel_finder_mode=graph`` followed by ``osprey build`` puts
   ``-m osprey.mcp_server.channel_finder_graph`` in the render's ``.mcp.json``
   under the ``channel-finder`` key, and arms the rendered subagent with
   exactly ``capabilities``, ``example_queries``, ``get_schema`` and
   ``read_cypher`` (plus ``submit_response``) — and with no ``mcp__graph__``
   entry, because the subagent reaches the store through its own server name.
   ``tests/cli/test_channel_finder_graph_tools.py`` pins the same shape off
   ``create_project``; this module pins it off the render ``osprey build``
   actually produces, which is a second code path (``regenerate_claude_code``).
2. **That server talks to a real store.** A throwaway Neo4j 5.20 +
   neosemantics container, seeded with the shipped demo-machine corpus through
   ``osprey knowledge seed-graph`` — the container, plugin and seeding recipe
   are imported wholesale from the sibling module rather than restated, so the
   two lanes cannot drift on how a graph store is stood up.
3. **The agent uses it, unprompted, through the subagent.** One operator-style
   question naming no tool, no server and no query language. The floor is
   deterministic: a ``mcp__channel-finder__read_cypher`` call that originated
   in a sub-agent context, at least one real ``fullPv`` from the corpus in what
   that call returned *and* in the agent's prose, and no fabricated address —
   graded by :func:`tests.e2e.test_graph_mcp_smoke.fabricated_addresses`, the
   same helper, so the two lanes share one definition of "invented".

There is no LLM judge here. The sibling needs one because "where does this sit
along the ring" is prose that only a reader can grade; a channel-address answer
is checkable against the corpus by string equality, and a cheaper gate that
proves the same thing is the better gate.

**Why the main-agent graph server is switched off.** ``control_assistant``
declares ``services.graphdb``, so a graph-paradigm render carries *both* the
standalone ``graph`` server (main agent) and the ``channel-finder`` one
(subagent) — two doors onto the same store. Left open, which door the model
picks is a coin flip, and a run that answered perfectly through
``mcp__graph__read_cypher`` would prove nothing about the paradigm. So the run
passes ``disallowed_tools=["mcp__graph"]``, the CLI's server-level deny form:
the main agent cannot see that server at all, and the only remaining route to
the store is the one under test. The subagent is unaffected — the deny is
session-scoped, but ``mcp__graph__*`` is not in its vocabulary to begin with.

Budget: ``max_turns=14``, ``max_budget_usd=1.5`` — the same envelope as the
sibling, because it is the same shape of work: the subagent will usually read
``example_queries`` or ``get_schema`` before it writes Cypher.

The render half needs neither Docker nor a credential and is the task's
validation gate::

    pytest tests/e2e/test_graph_paradigm_smoke.py -k render

which is why this module, like its sibling, carries **no** module-level
``pytestmark``: a blanket ``requires_als_apg`` would skip a half that has
nothing to do with a model. The live test carries its own gating decorators.

One trap worth naming for anyone running this outside an installed checkout:
``osprey knowledge seed-graph`` is spawned as a subprocess **with its cwd at
the render**, so a relative ``PYTHONPATH=src`` does not resolve there and the
subprocess silently falls back to whatever ``osprey`` the interpreter has
installed. Point ``PYTHONPATH`` at an absolute ``src`` and the seeding step
runs the code under test rather than another checkout's.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.agent_runner import expected_mcp_servers
from osprey.registry.mcp import CHANNEL_FINDER_TOOLS_BY_PIPELINE
from tests.e2e.sdk_helpers import (
    HAS_SDK,
    e2e_budget_scale,
    init_project,
    is_claude_code_available,
    render_dir,
    run_sdk_query,
)

# The container, its plugins, the seeding verb and the fabrication floor all
# come from the sibling module. Importing the two fixtures by name is what
# registers them for this module (pytest resolves fixtures through the test
# module's namespace); ``noqa: F401`` because nothing here references them
# directly.
from tests.e2e.test_graph_mcp_smoke import (  # noqa: F401 — fixture registration
    _MARKDOWN_EMPHASIS,
    PROBE_DEVICE_BINDINGS,
    PROBE_PV,
    _point_project_at_the_store,
    _seed_demo_corpus,
    fabricated_addresses,
    graph_smoke_plugin_dir,
    graph_store_port,
)

logger = logging.getLogger(__name__)

#: The paradigm under test, spelled once.
GRAPH_MODE = "graph"

#: What the subagent's ``tools:`` frontmatter must carry, qualified. Derived
#: from the registry so this module cannot drift from the vocabulary the render
#: reads; ``tests/cli/test_channel_finder_graph_tools.py`` holds the literal
#: four-name spelling, so a registry edit still has to be made deliberately in
#: two places.
_SUBMIT_RESPONSE = "mcp__osprey_workspace__submit_response"
_EXPECTED_AGENT_TOOLS = [
    *(f"mcp__channel-finder__{tool}" for tool in CHANNEL_FINDER_TOOLS_BY_PIPELINE[GRAPH_MODE]),
    _SUBMIT_RESPONSE,
]

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


# ---------------------------------------------------------------------------
# The deployment
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graph_paradigm_repo(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """A built ``control_assistant`` repo in graph channel-finder mode.

    No container and no credential: this is the render alone, which is all the
    ``-k render`` half needs. The live test layers the store on top through
    :func:`seeded_graph_paradigm_project` rather than building a second time —
    ``osprey init`` + ``osprey build`` is the expensive step in this module and
    doing it once keeps the live lane's wall time honest.

    ``tier`` is deliberately not passed: ``init_project`` asks
    ``tier_mode_conflict`` and drops the derived tier for ``graph``, which ships
    no tiered database files. Passing one would be rejected by the build.
    """
    tmp = tmp_path_factory.mktemp("graph-paradigm")
    try:
        yield init_project(
            tmp,
            "graph_paradigm",
            template="control_assistant",
            provider="als-apg",
            channel_finder_mode=GRAPH_MODE,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _agent_frontmatter_tools(render: Path) -> list[str]:
    """The rendered channel-finder subagent's ``tools:`` CSV, split."""
    md = render / ".claude" / "agents" / "channel-finder.md"
    assert md.is_file(), f"the build rendered no channel-finder subagent at {md}"
    match = _FRONTMATTER_RE.match(md.read_text(encoding="utf-8"))
    assert match, f"no YAML frontmatter in {md}"
    frontmatter = yaml.safe_load(match.group(1))
    assert isinstance(frontmatter, dict), f"frontmatter is not a mapping in {md}"
    return [entry.strip() for entry in str(frontmatter["tools"]).split(",") if entry.strip()]


@pytest.fixture(scope="module")
def seeded_graph_paradigm_project(
    graph_paradigm_repo: Path,
    # Requesting an imported fixture by name reads to ruff as a redefinition of
    # the import above; pytest resolves it as the fixture, which is the point.
    graph_store_port: int,  # noqa: F811
) -> Iterator[Path]:
    """The graph-mode deployment, repointed at the throwaway store and seeded.

    Module-scoped so a ``@flaky`` rerun re-asks the question rather than
    rebuilding and re-seeding; nothing in this module writes to the graph, so
    the second attempt sees exactly the corpus the first did.

    ``GRAPHDB_PASSWORD`` is pinned in this process's environment as well as in
    the project's ``.env`` for the reason the sibling module spells out at
    length: ``load_dotenv_from_project`` loads with ``override=False``, so a
    developer's already-exported password for some *other* store would outrank
    the ``.env`` and the lane would fail as "every graph call errored".
    """
    from tests.e2e.test_graph_mcp_smoke import GRAPHDB_TEST_PASSWORD

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv("GRAPHDB_PASSWORD", GRAPHDB_TEST_PASSWORD)
        _point_project_at_the_store(graph_paradigm_repo, graph_store_port)
        logger.info(
            "seed-graph: %s", _seed_demo_corpus(graph_paradigm_repo).strip().replace("\n", " | ")
        )
        yield graph_paradigm_repo
    finally:
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# The render half — no Docker, no credential, no agent
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.harness_benchmark
def test_render_targets_the_graph_channel_finder_server(graph_paradigm_repo: Path) -> None:
    """``.mcp.json`` runs the graph paradigm under the ``channel-finder`` name.

    The server *name* is the load-bearing half. Every consumer that selects a
    paradigm by config key — the registry's permission fill, the subagent's
    tool allowlist, the benchmark harness's ``mcp__channel-finder__*`` filter —
    reaches this paradigm unchanged precisely because it does not get a name of
    its own. A graph paradigm shipped as a server called ``graph`` would be
    filtered out of all three.
    """
    render = render_dir(graph_paradigm_repo)
    servers = json.loads((render / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]

    assert "channel-finder" in servers, (
        f"the graph-mode build rendered no 'channel-finder' MCP server — "
        f".mcp.json declares {sorted(servers)}"
    )
    assert servers["channel-finder"]["args"] == [
        "-m",
        "osprey.mcp_server.channel_finder_graph",
    ], (
        "the channel-finder server does not run the graph paradigm's module; "
        f"args are {servers['channel-finder']['args']}"
    )
    for var in ("OSPREY_CONFIG", "CONFIG_FILE"):
        assert servers["channel-finder"]["env"][var].endswith("/config.yml"), (
            f"the channel-finder server has no usable {var}; it resolves the graph "
            "connection from the config file at runtime and would not find the store"
        )

    # The readiness barrier the live test relies on reads the same file.
    assert "channel-finder" in expected_mcp_servers(render)


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.harness_benchmark
def test_render_arms_the_subagent_with_the_paradigm_vocabulary(
    graph_paradigm_repo: Path,
) -> None:
    """Exactly the four graph tools, and nothing from the main agent's server.

    ``mcp__graph__*`` belongs to the main agent. Its appearance here would mean
    the subagent had a second, unmanaged route to the store, and the delegation
    assertion in the live test below would stop being a claim about the
    paradigm.
    """
    render = render_dir(graph_paradigm_repo)

    assert _agent_frontmatter_tools(render) == _EXPECTED_AGENT_TOOLS

    granted = json.loads((render / ".claude" / "settings.json").read_text(encoding="utf-8"))[
        "permissions"
    ]["allow"]
    for tool in CHANNEL_FINDER_TOOLS_BY_PIPELINE[GRAPH_MODE]:
        assert f"mcp__channel-finder__{tool}" in granted, (
            f"the render granted no permission for mcp__channel-finder__{tool}; the "
            "subagent would be denied a tool its own frontmatter names"
        )


# ---------------------------------------------------------------------------
# The live half
# ---------------------------------------------------------------------------

#: Operator-style and naming nothing the model could pattern-match onto a tool:
#: no server, no ``mcp__`` prefix, no Cypher keyword, no mention of a graph, and
#: no channel address. It asks the one thing a channel finder is *for* — the
#: address behind a piece of hardware described in words — about the device
#: whose complete binding set this module knows
#: (:data:`PROBE_DEVICE_BINDINGS`), so both the "named a real address" floor
#: and the fabrication floor can grade it.
OPERATOR_PROMPT = (
    "I need to trim the current on the first dipole magnet in the storage ring. "
    "Which channel do I write the new setpoint to, and what can I read back on "
    "that same magnet afterwards to confirm it took?"
)


def _real_addresses_named(text: str) -> list[str]:
    """The probe device's genuine channel addresses that *text* names.

    Markdown emphasis is stripped with the sibling module's own rule before the
    match, so a bolded ``SR:MAG:DIPOLE:01:CURRENT:**SP**`` counts as the address
    it renders — the same normalisation
    :func:`~tests.e2e.test_graph_mcp_smoke.fabricated_addresses` applies before
    it looks for inventions.
    """
    plain = _MARKDOWN_EMPHASIS.sub("", text)
    return sorted(pv for pv in PROBE_DEVICE_BINDINGS if pv in plain)


def _all_assistant_text(result: Any) -> str:
    """Every assistant text block of the run, joined — not just the last one.

    Same reasoning as the sibling module: the agent narrates as it delegates, so
    grading the whole of its prose makes the fabrication floor stricter than a
    final-message-only read.
    """
    return "\n".join(result.text_blocks).strip()


def _print_run(result: Any, prose: str) -> None:
    print("\n--- Graph channel-finder paradigm smoke ---")
    print(f"  prompt: {OPERATOR_PROMPT}")
    print(f"  mcp servers: {[(s.get('name'), s.get('status')) for s in result.mcp_servers]}")
    print(f"  tools called: {result.tool_names}")
    print(f"  num_turns: {result.num_turns}")
    print(f"  tokens: {result.input_tokens} in / {result.output_tokens} out")
    print(f"  cost: ${result.cost_usd:.4f}" if result.cost_usd else "  cost: N/A")
    print("  --- agent prose (all assistant text blocks) ---")
    print("\n".join("    " + line for line in prose.splitlines()))
    print("  --- end prose ---\n")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.agentic_benchmark
@pytest.mark.requires_als_apg
@pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed")
@pytest.mark.skipif(not is_claude_code_available(), reason="claude CLI not available")
@pytest.mark.flaky(reruns=2)  # multi-step agentic delegation; absorb stochastic misses
@pytest.mark.asyncio
async def test_channel_finder_graph_paradigm_answers_an_address_question(
    seeded_graph_paradigm_project: Path,
) -> None:
    """An operator asks for a channel; the graph paradigm answers with a real one.

    The whole deterministic floor, in order: the readiness barrier saw
    ``channel-finder`` connect, the agent delegated rather than answering from
    training data, the subagent wrote Cypher against the seeded store, the store
    returned a real address, the prose passed one on, and nothing anywhere in
    the run invented a channel this device does not have.
    """
    render = render_dir(seeded_graph_paradigm_project)
    assert "channel-finder" in expected_mcp_servers(render), (
        "the readiness barrier would not wait for the channel-finder server — "
        f".mcp.json declares {sorted(expected_mcp_servers(render))}"
    )

    result = await run_sdk_query(
        seeded_graph_paradigm_project,
        OPERATOR_PROMPT,
        max_turns=14,
        max_budget_usd=1.5,
        # The main agent's own door onto the same store, shut — see the module
        # docstring. ``mcp__graph`` (no trailing ``__*``) is the CLI's
        # server-level deny form.
        disallowed_tools=["mcp__graph"],
    )
    prose = _all_assistant_text(result)
    _print_run(result, prose)

    assert result.result is not None, "no ResultMessage received"
    assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

    # --- The barrier's verdict -------------------------------------------
    # ``run_sdk_query`` polls ``get_mcp_status()`` before the first turn, so an
    # unconnected server is reported as the infrastructure failure it is rather
    # than as a model miss.
    cf_status = next(
        (s.get("status") for s in result.mcp_servers if s.get("name") == "channel-finder"), None
    )
    assert cf_status == "connected", (
        f"the channel-finder MCP server never reached 'connected' (status={cf_status!r}); "
        f"full snapshot: {[(s.get('name'), s.get('status')) for s in result.mcp_servers]}. "
        "This is an infrastructure failure, not a model miss."
    )

    # --- Floor 1: the question went through the subagent -------------------
    cf_calls = result.tools_matching("mcp__channel-finder__")
    assert cf_calls, (
        "the agent answered a channel-address question without ever reaching the "
        f"channel finder. Tools called: {result.tool_names}. Answering from "
        "training data is the canonical silent failure this gate exists to catch."
    )
    direct = [t for t in cf_calls if t.parent_tool_use_id is None]
    assert not direct, (
        f"the main agent called {len(direct)} channel-finder tool(s) itself instead of "
        f"delegating: {[t.name for t in direct]}. Those tools belong to the subagent."
    )

    # --- Floor 2: it wrote Cypher, and the store answered ------------------
    cypher_calls = [t for t in cf_calls if t.name == "mcp__channel-finder__read_cypher"]
    assert cypher_calls, (
        "the subagent never called mcp__channel-finder__read_cypher — the graph "
        "paradigm's only retrieval tool. It called "
        f"{sorted({t.name for t in cf_calls})}, which can describe the store but "
        "cannot answer a question about it."
    )
    answered = [t for t in cypher_calls if not t.is_error and t.result]
    assert answered, (
        "every read_cypher call errored or came back empty: "
        f"{[(t.is_error, (t.result or '')[:300]) for t in cypher_calls]}"
    )

    from_store = sorted({pv for t in answered for pv in _real_addresses_named(t.result or "")})
    assert from_store, (
        "no read_cypher call returned any of this device's channel addresses. The "
        "store may be seeded but unreachable, or the query missed. Corpus binds "
        f"{list(PROBE_DEVICE_BINDINGS)}; results were "
        f"{[(t.result or '')[:400] for t in answered]}"
    )

    # --- Floor 3: the answer carries a real address, and invents none -------
    assert prose, (
        "the agent produced no assistant text at all — there is nothing to grade. "
        f"Tools called: {result.tool_names}"
    )
    named = _real_addresses_named(prose)
    assert named, (
        "the agent reached the graph but its answer names none of the device's real "
        f"channel addresses. The corpus binds {list(PROBE_DEVICE_BINDINGS)} and the "
        f"setpoint the question asks for is {PROBE_PV}."
    )

    fabricated = fabricated_addresses(prose)
    assert not fabricated, (
        f"the agent's prose names channel addresses this device does not have: "
        f"{fabricated}. The corpus binds exactly {list(PROBE_DEVICE_BINDINGS)} to it."
    )

    print(f"  addresses returned by the store: {from_store}")
    print(f"  addresses named in the answer:   {named}")

    # --- Cost, as a hard ceiling ------------------------------------------
    # ``max_budget_usd`` already hard-errors mid-query at the cap; this catches
    # the run that finished just under it, which is the drift worth seeing.
    if result.cost_usd is not None:
        budget = 1.5 * e2e_budget_scale()
        assert result.cost_usd < budget, (
            f"the run cost ${result.cost_usd:.4f}, at or over the ${budget:.2f} cap"
        )
