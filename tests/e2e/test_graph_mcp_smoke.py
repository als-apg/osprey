"""Agentic e2e acceptance gate for the graph MCP server.

This is the feature's acceptance test: a deployment built from the
``control_assistant`` preset, pointed at a *real* Neo4j 5.20 + neosemantics
store seeded with the shipped demo-machine corpus, driven by a real agent
session with an operator-style question about the machine's structure. The
claim under test is the whole chain at once — the preset renders ``graph`` into
``.mcp.json`` and the ``facility-knowledge-graph`` subagent beside it, the
server comes up, the main agent delegates to the subagent *on its own*, the
read-only Cypher path answers, and the answer is right about the corpus.

What the fixture stands up, and why each piece is shaped the way it is:

* **A store of its own.** ``neo4j:5.20-community`` with the pinned neosemantics
  5.20.0 jar bind-mounted at ``/plugins`` and APOC copied out of the image —
  the recipe ``tests/integration/test_graphdb_store.py`` established; see that
  module's docstring for why ``NEO4J_PLUGINS`` is deliberately unset. The
  container takes a **random published port and testcontainers' own generated
  name**, so it can never collide with a ``graphdb`` service a developer has
  deployed on the same host, and two of these can run side by side.
* **A project patched onto that port.** ``services.graphdb.port_host`` in the
  render's ``config.yml`` and ``GRAPHDB_PASSWORD`` in the repo's ``.env`` are
  the two values that decide *which* store the MCP server dials and with what
  credential. Both are read at server **runtime** (the connection is resolved
  on first use from ``CONFIG_FILE``; the ``.env`` chain is loaded by
  ``run_mcp_server`` at launch), neither participates in rendering — which is
  why they are patched onto the finished build rather than through a profile
  override, exactly as ``tests/e2e/claude_code/conftest.py`` repoints the
  limits database. That the render carries ``graph`` at all is not assumed:
  :func:`_assert_graph_is_rendered` pins it before anything is patched.
* **The corpus, seeded through the shipped verb.** ``osprey knowledge
  seed-graph`` against the rendered project, so the graph the agent queries got
  there by the same path an operator's would — including the ``_OspreySeed``
  marker.

Grading is the two layers ``tests/e2e/README.md`` asks for:

* a **deterministic floor** over the tool trace — a ``mcp__graph__*`` tool was
  reached, through the subagent and never by the main agent directly, that
  call succeeded, and the prose (all of it, not just the closing message)
  invents no channel address that is not in the corpus; and
* an **LLM judge** over the prose, against a rubric written from facts read
  back out of this very corpus (see :data:`_VERIFIED` — every number in the
  rubric was queried from the seeded store, not assumed from the generator).

The prompt is operator-style and names **no** tool, server, MCP prefix, Cypher
keyword or JSON key. It names one PV, because that is the operator's own
handle on the problem, and asks three things a flat channel list cannot answer
on its own: which *device* that address belongs to, what else that device
exposes, and where it sits along the ring. If a natural question like that
fails to reach the graph, that is a wiring or tool-description finding to fix
in the framework — never by steering the prompt toward the tool.

Budget: ``max_turns=14``, ``max_budget_usd=1.5`` — larger than the
single-tool health smoke because this is a multi-hop question the agent may
approach through ``example_queries``/``get_schema`` before writing Cypher.

The floor half is dry-verifiable offline, the way
``test_plan_stack_agentic.py`` makes its own floor dry-verifiable — no Docker,
no API key, no agent session, no container::

    uv run pytest tests/e2e/test_graph_mcp_smoke.py -k floor

Those tests grade :func:`fabricated_addresses`, the same helper the live test
calls, against hand-written strings. That is why this module carries **no**
module-level ``pytestmark``: a blanket ``requires_als_apg`` plus the SDK and
CLI skips would skip the offline half too, and a floor nobody can run cheaply
is a floor nobody re-runs when they change the regex. The live test carries
its own gating decorators instead.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.agent_runner import expected_mcp_servers
from tests._container_support import start_or_fail, stop_quietly
from tests._graphdb_container import GRAPHDB_TEST_PASSWORD, NEO4J_IMAGE
from tests.e2e.judge import LLMJudge
from tests.e2e.sdk_helpers import (
    HAS_SDK,
    init_project,
    is_claude_code_available,
    render_dir,
    run_sdk_query,
)
from tests.e2e.test_preset_agentic import _to_workflow_result

logger = logging.getLogger(__name__)

# No module-level ``pytestmark`` — see the module docstring. The two halves of
# this file have different prerequisites and therefore different markers:
#
#   * the live acceptance test carries ``e2e``/``slow``/``agentic_benchmark``
#     (the lane: the agent must choose and sequence tools and reason about what
#     came back, and a weak model plausibly fails it), ``requires_als_apg``, and
#     the SDK/CLI skips, and
#   * the offline floor tests carry ``harness_benchmark`` (an assertion about
#     OSPREY's own grading logic, model-independent by construction) and nothing
#     else, so they run on any host.
#
# ALS_APG_API_KEY is enforced via ``requires_als_apg`` — the root
# ``tests/conftest.py`` hook auto-skips when the key is missing, and the CI lane
# fails outright if the secret is unset, so a missing credential is never a
# silent pass. Both halves are collected by the shared ``e2e-tests`` job
# (``pytest tests/e2e/``) — no dedicated lane and no ``--ignore`` entry: the
# container takes a random port and a generated name, so it leaves no
# host-global residue.


# ---------------------------------------------------------------------------
# Pins (the shared recipe in tests/_graphdb_container.py — see its docstring)
# ---------------------------------------------------------------------------

#: This module publishes bolt on a fixed host port (the render's config.yml
#: names it), unlike the ephemeral-port stores the shared recipe starts.
BOLT_PORT = 7687


# ---------------------------------------------------------------------------
# Ground truth, read back out of the seeded demo corpus
# ---------------------------------------------------------------------------

#: The address the operator prompt names. A real ``fullPv`` in the generated
#: demo corpus, and the writable one of its device's six bindings.
PROBE_PV = "SR:MAG:DIPOLE:01:CURRENT:SP"

#: Every channel address the demo corpus binds to the device that owns
#: :data:`PROBE_PV`. Read from the store, not derived from the grammar: the
#: anti-hallucination floor below is only meaningful if this set is what the
#: graph actually holds.
PROBE_DEVICE_BINDINGS = (
    "SR:MAG:DIPOLE:01:CURRENT:GOLDEN",
    "SR:MAG:DIPOLE:01:CURRENT:RB",
    "SR:MAG:DIPOLE:01:CURRENT:SP",
    "SR:MAG:DIPOLE:01:STATUS:FAULT",
    "SR:MAG:DIPOLE:01:STATUS:ON",
    "SR:MAG:DIPOLE:01:STATUS:READY",
)

#: Facts the judge rubric is built from. Every one was queried from a store
#: seeded with the same ``demo_machine.ttl`` this test seeds (25,845 triples,
#: 512 devices, 2,908 bindings). Restated here rather than re-queried at run
#: time on purpose: a rubric that asked the graph what the answer is could be
#: satisfied by a graph that is wrong.
#:
#: ``dipole_count_sr`` and ``dipole_count_machine`` are BOTH here, and the
#: distinction is not pedantry: the ``Dipole`` class holds 44 devices across the
#: whole corpus, but only 36 of them are in the storage ring — the other 8 are
#: the booster's, at s = 10-17 m. A rubric carrying only the machine-wide figure
#: marks a correct "36 dipoles in the ring" answer as fabricated, which is
#: exactly the wrong-answer-key failure a hand-written rubric invites. Both
#: numbers are named so the judge can recognise either as right.
_VERIFIED = {
    "device_name": "DIPOLE01",
    "device_uri": "https://narad.example.org/device/demo_SR_DIPOLE01",
    "section": "SR",
    "s_position_m": 82.0,
    "ordinal_in_section": 82,
    "dipole_count_sr": 36,
    "dipole_count_machine": 44,
    "sr_dipole_span_m": (82.0, 117.0),
    "sr_dipole_spacing_m": 1.0,
    "next_device": ("DIPOLE02", 83.0),
    "previous_device": ("NEUTRON04", 81.0),
}

#: Any ``SR:MAG:DIPOLE:01:…`` address the corpus does *not* hold is a fabricated
#: one. Matching on that device's own prefix keeps the check narrow enough that
#: it can only fire on an invention, never on a legitimate mention of some other
#: device's channel.
#:
#: The **whole** six-token grammar is required — ``RING:SYSTEM:FAMILY:DEVICE``
#: plus ``FIELD:SUBFIELD`` — rather than "the prefix and one or more segments".
#: A greedy prefix match reads a *truncated* mention as an address in its own
#: right, and then reports it as fabricated because the corpus holds no such
#: channel. That is not hypothetical: an otherwise perfect answer that rendered
#: the flagged channel as ``SR:MAG:DIPOLE:01:CURRENT:**SP**`` (markdown emphasis
#: on the last token) failed this floor on ``SR:MAG:DIPOLE:01:CURRENT``. Hence
#: also :data:`_MARKDOWN_EMPHASIS`: agents format addresses, and a fabrication
#: detector that trips on bold text detects formatting, not fabrication.
#:
#: The trailing negative lookahead is the other half of that rule, in the other
#: direction: without it a *longer* invention such as
#: ``SR:MAG:DIPOLE:01:CURRENT:SP:LIMIT`` matches on its first six tokens, which
#: ARE a real address, and the floor reports the whole thing as legitimate. With
#: it, that string matches nothing at all — so this floor never launders a
#: deeper-token invention as real, and the seven-token case is left to the judge
#: (criterion 4), which reads the answer rather than a prefix of it. Both edges
#: are deliberately biased toward the false negative: a hard gate that fails a
#: correct answer costs more than one that lets an exotic invention through to a
#: second grader.
_PROBE_DEVICE_PV_RE = re.compile(r"\bSR:MAG:DIPOLE:01:[A-Za-z0-9_]+:[A-Za-z0-9_]+(?![:A-Za-z0-9_])")

#: Emphasis and code markers stripped before the scan above, so that bold,
#: italic and inline-code renderings of an address all normalise to the address.
#: Stripping ``_`` is safe *here* precisely because the scan is scoped to ONE
#: device: none of that device's six real addresses
#: (:data:`PROBE_DEVICE_BINDINGS`) contains an underscore, so no string this
#: regex can legitimately match is corrupted by removing them — while
#: ``…CURRENT:_SP_`` (underscore italics) is caught rather than mistaken for a
#: fabricated ``…CURRENT:_SP_`` token.
_MARKDOWN_EMPHASIS = re.compile(r"[*`_]")


def fabricated_addresses(text: str) -> list[str]:
    """Channel addresses *text* attributes to the probe device that do not exist.

    The anti-hallucination floor, factored out so the live test and the offline
    ``-k floor`` tests grade the *same* code — a dry-verification that exercised
    a re-spelled copy of the rule would prove nothing about the rule the gate
    uses.

    Markdown emphasis is stripped first, then every complete six-token address
    on this one device's prefix is collected and the six real ones subtracted.
    Anything left is a channel the answer claims the device has and the corpus
    says it does not.

    Args:
        text: The agent's prose, as written (markdown and all).

    Returns:
        The fabricated addresses, sorted and deduplicated; empty when the answer
        invents nothing.
    """
    plain = _MARKDOWN_EMPHASIS.sub("", text)
    return sorted(set(_PROBE_DEVICE_PV_RE.findall(plain)) - set(PROBE_DEVICE_BINDINGS))


# ---------------------------------------------------------------------------
# Plugins + container
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graph_smoke_plugin_dir(graphdb_plugin_dir: Path) -> Path:
    """n10s + APOC, from the session-wide resolver (tests/_graphdb_container).

    The module's loud-skip gate lives in that resolver, because this is the
    first thing the store fixture depends on: an unreachable daemon is reported
    once, with its reason, rather than as an unexplained skip on the
    acceptance test.
    """
    return graphdb_plugin_dir


@pytest.fixture(scope="module")
def graph_store_port(graph_smoke_plugin_dir: Path) -> Iterator[int]:
    """Start a throwaway graph store and yield its **published host port**.

    Neither the port nor the container name is fixed: testcontainers publishes
    bolt on an ephemeral host port and generates the container's name, so this
    fixture cannot collide with a ``graphdb`` service already deployed on the
    host (which publishes 7687 under a project-derived name) and two runs of
    this module can overlap.

    ``start_or_fail`` rather than ``start_or_skip``: Docker was established
    reachable by :func:`graph_smoke_plugin_dir`, so past that point a container
    that will not start is a real failure, and it retries the testcontainers
    port-publish race that ``start_or_skip`` would turn into a vacuous green.
    """
    try:
        from testcontainers.community.neo4j import Neo4jContainer
    except ImportError:  # pragma: no cover - depends on the installed extras
        pytest.skip("testcontainers' neo4j module is not installed")

    def _build() -> Neo4jContainer:
        container = Neo4jContainer(image=NEO4J_IMAGE, password=GRAPHDB_TEST_PASSWORD)
        container.with_volume_mapping(str(graph_smoke_plugin_dir), "/plugins", "rw")
        # The allowlist is the half of NEO4J_PLUGINS that is not a download:
        # without it every n10s.* call fails with "not on the allowlist", and
        # n10s needs the unrestricted grant because it calls into APOC.
        container.with_env("NEO4J_dbms_security_procedures_unrestricted", "apoc.*,n10s.*")
        container.with_env("NEO4J_dbms_security_procedures_allowlist", "apoc.*,n10s.*")
        return container

    container, port = start_or_fail(_build, "graphdb (neo4j + n10s)", BOLT_PORT)
    logger.info(f"graph store published bolt on host port {port}")
    try:
        yield port
    finally:
        stop_quietly(container)


# ---------------------------------------------------------------------------
# The deployment, patched onto that store and seeded through the shipped verb
# ---------------------------------------------------------------------------


def _assert_graph_is_rendered(repo: Path) -> None:
    """Fail loudly if the build did not put the graph surface in the render.

    The whole test rests on ``control_assistant`` shipping a ``services.graphdb``
    block that makes ``graphdb_configured`` true; if that ever stops being so,
    every assertion below would still "pass" against an agent that simply had no
    graph tools to reach for — which is precisely the vacuous green this pins.
    """
    render = render_dir(repo)
    servers = expected_mcp_servers(render)
    assert "graph" in servers, (
        f"the control_assistant build rendered no 'graph' MCP server — .mcp.json "
        f"declares {sorted(servers)}. Without it the readiness barrier never waits "
        "for the graph server and the agent has nothing to reach for."
    )
    settings = json.loads((render / ".claude" / "settings.json").read_text(encoding="utf-8"))
    granted = [tool for tool in settings["permissions"]["allow"] if tool.startswith("mcp__graph__")]
    assert granted, (
        "the render granted no mcp__graph__* permissions; the facility-knowledge-graph "
        "subagent would be denied the tools this test asserts it reaches"
    )
    agent_md = render / ".claude" / "agents" / "facility-knowledge-graph.md"
    assert agent_md.is_file(), (
        "the build rendered no facility-knowledge-graph subagent — the graph tools "
        "belong to it, so without it the delegation floor below would grade a "
        "project that cannot delegate"
    )


def _point_project_at_the_store(repo: Path, port: int) -> None:
    """Repoint the built project at the throwaway store and give it the password.

    Two runtime-read values, patched onto the finished build:

    * ``services.graphdb.port_host`` in the **render's** ``config.yml`` — what
      ``resolve_graphdb_connection`` derives ``bolt://localhost:{port}`` from
      when the MCP server first dials. It is read live through ``CONFIG_FILE``,
      never baked into ``.mcp.json`` or ``settings.json``, so no re-render is
      needed and none is done.
    * ``GRAPHDB_PASSWORD`` in the **repo's** ``.env`` — the deployment's own
      credential channel: ``run_mcp_server`` loads that chain at launch, and the
      repo root (not the render) is where it lives, because every build
      re-creates the render from scratch.
    """
    config_path = render_dir(repo) / "config.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    graphdb = config["services"]["graphdb"]
    assert graphdb.get("ttl_path"), (
        "control_assistant's services.graphdb block declares no ttl_path, so "
        "`osprey knowledge seed-graph` has no corpus to load"
    )
    graphdb["port_host"] = port
    config_path.write_text(
        yaml.dump(config, default_flow_style=False, sort_keys=False), encoding="utf-8"
    )

    env_path = repo / ".env"
    existing = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    env_path.write_text(f"{existing}GRAPHDB_PASSWORD={GRAPHDB_TEST_PASSWORD}\n", encoding="utf-8")


def _seed_demo_corpus(repo: Path) -> str:
    """Seed the store through ``osprey knowledge seed-graph``; return its report.

    The shipped verb, run as a subprocess against the rendered project, rather
    than the seeder primitives directly: the point is that the corpus reached
    the store by the path an operator's would, resolving ``ttl_path`` against
    the config file's own directory and dialing the address the patched config
    names. A non-zero exit fails here rather than as a mysterious empty graph
    three minutes and one LLM call later.
    """
    config_path = render_dir(repo) / "config.yml"
    env = dict(os.environ)
    env["CONFIG_FILE"] = str(config_path)
    env["OSPREY_CONFIG"] = str(config_path)
    env["GRAPHDB_PASSWORD"] = GRAPHDB_TEST_PASSWORD

    result = subprocess.run(
        [sys.executable, "-m", "osprey.cli.main", "knowledge", "seed-graph"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(render_dir(repo)),
        timeout=600,
    )
    assert result.returncode == 0, (
        f"osprey knowledge seed-graph failed (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr[-4000:]}"
    )
    assert "triples loaded" in result.stdout, (
        f"seed-graph reported no import; the store may be empty:\n{result.stdout}"
    )
    return result.stdout


@pytest.fixture(scope="module")
def graph_project(
    graph_store_port: int, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Path]:
    """A built ``control_assistant`` deployment wired to the seeded store.

    Module-scoped so a ``@flaky`` rerun re-asks the question instead of
    rebuilding and re-seeding: nothing in this test writes to the graph, so the
    second attempt sees exactly the corpus the first did.

    The project tree is removed on teardown on every path. ``osprey build``
    renders four container build contexts and three persona projects beside the
    corpus, which is a lot of bytes to leave behind on a CI runner.

    ``GRAPHDB_PASSWORD`` is pinned in this process's own environment as well as
    in the project's ``.env``, and the belt matters as much as the braces. The
    SDK merges its ``env`` over the inherited ``os.environ``, and
    ``mcp_env.load_dotenv_from_project`` loads the chain with ``override=False``
    — so a variable the shell already exports outranks the whole ``.env``. On a
    developer's host with some *other* deployment's ``GRAPHDB_PASSWORD``
    exported, the graph MCP server would authenticate against this throwaway
    store with the wrong credential and the test would fail as "every graph call
    errored", which reads as a broken feature rather than a polluted
    environment. A module-scoped :class:`pytest.MonkeyPatch` is used because the
    ``monkeypatch`` fixture is function-scoped and cannot be requested here.
    """
    tmp = tmp_path_factory.mktemp("graph-smoke")
    monkeypatch = pytest.MonkeyPatch()
    repo = init_project(tmp, "graph_smoke", template="control_assistant", provider="als-apg")
    try:
        monkeypatch.setenv("GRAPHDB_PASSWORD", GRAPHDB_TEST_PASSWORD)
        _assert_graph_is_rendered(repo)
        _point_project_at_the_store(repo, graph_store_port)
        logger.info("seed-graph: %s", _seed_demo_corpus(repo).strip().replace("\n", " | "))
        yield repo
    finally:
        monkeypatch.undo()
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# The prompt and the rubric
# ---------------------------------------------------------------------------

#: Operator-style, and deliberately naming nothing the agent could pattern-match
#: onto a tool: no server, no ``mcp__`` prefix, no Cypher keyword, no JSON key,
#: no mention of a graph. The one identifier it carries is a PV, because that is
#: what an operator would have in front of them. The three asks are structural —
#: which *device* owns an address, what else that device exposes, and where it
#: sits along the ring — and the last of those is not in any channel list.
OPERATOR_PROMPT = (
    "A handover note flags SR:MAG:DIPOLE:01:CURRENT:SP for follow-up, but it "
    "doesn't say why. Before I chase it: which piece of hardware does that "
    "address actually belong to, what else can I read or set on that same piece "
    "of hardware, and whereabouts does it sit in the storage ring relative to "
    "the other dipoles?"
)

_RUBRIC = f"""\
Evaluation of an accelerator operator's question about one device in the
facility. The agent had read-only access to a knowledge graph describing the
machine, and the deterministic half of this test has ALREADY verified, from the
tool trace, that the agent queried that graph successfully and that it invented
no channel addresses. Do NOT re-grade either of those; grade only the answer's
substance against the ground truth below.

GROUND TRUTH (read directly out of the corpus the agent was querying):
  * The address {PROBE_PV} belongs to a single device, a storage-ring dipole
    magnet named {_VERIFIED["device_name"]} (URI {_VERIFIED["device_uri"]}).
    Equivalent renderings of that identity — "DIPOLE 01", "dipole 1", "the
    first storage-ring dipole", the bare URI — are all correct.
  * That device carries exactly six channel bindings:
    {", ".join(PROBE_DEVICE_BINDINGS)}.
    Only {PROBE_PV} is writable; the other five are read-only.
  * The device sits in section {_VERIFIED["section"]} at longitudinal position
    s = {_VERIFIED["s_position_m"]} m, ordinal {_VERIFIED["ordinal_in_section"]}
    within its section. It is the FIRST dipole along the storage ring. The
    device immediately downstream is {_VERIFIED["next_device"][0]} at
    s = {_VERIFIED["next_device"][1]} m; the one immediately upstream is
    {_VERIFIED["previous_device"][0]} (a beam-loss monitor) at
    s = {_VERIFIED["previous_device"][1]} m.
  * Dipole census, and BOTH figures are correct depending on what is being
    counted: the storage ring holds {_VERIFIED["dipole_count_sr"]} dipoles
    (DIPOLE01-DIPOLE{_VERIFIED["dipole_count_sr"]}), evenly spaced
    {_VERIFIED["sr_dipole_spacing_m"]} m apart from
    s = {_VERIFIED["sr_dipole_span_m"][0]} m to
    s = {_VERIFIED["sr_dipole_span_m"][1]} m, while the machine as a whole holds
    {_VERIFIED["dipole_count_machine"]} — the other
    {_VERIFIED["dipole_count_machine"] - _VERIFIED["dipole_count_sr"]} are the
    booster's. An answer that attributes the machine-wide figure
    ({_VERIFIED["dipole_count_machine"]}) to the storage ring, or the ring
    figure ({_VERIFIED["dipole_count_sr"]}) to the machine as a whole, IS wrong
    under (4). Do not penalise an answer merely for choosing the scope the
    operator did not ask about, as long as it names that scope.

PASS only if ALL of the following hold:

  (1) The answer identifies the owning device correctly — a storage-ring
      dipole, the first one / number 01. Naming it by any of the equivalent
      renderings above is fine. Getting the device wrong (a different dipole, a
      quadrupole, "a magnet power supply" with no device identity) is a FAIL.

  (2) The answer says what else that device exposes, and what it says is
      consistent with the six bindings above. It need NOT list all six, and it
      may describe them in words ("a current setpoint, a readback, a golden
      reference and three status flags") rather than as literal addresses.
      Naming channels that are not among the six, or claiming the device
      exposes something the corpus does not carry, is a FAIL.

  (3) The answer places the device along the ring in a way consistent with the
      ground truth — its s-position, its ordinal, or its being the first dipole
      with the others following. A vague "it's in the storage ring" with no
      ordering or position at all does NOT satisfy this; an explicit,
      well-grounded statement that the position information could not be
      retrieved is also a FAIL, because it IS in the corpus the agent queried.

  (4) The answer does not fabricate. Every concrete claim it makes about THIS
      device, the storage-ring dipole line, or this device's immediate
      neighbours must be consistent with the ground truth above. Hedged or
      partial answers are acceptable; confidently wrong ones are not.

Scope limits on (4), so it grades fabrication rather than incompleteness:

  * The ground truth above is the ONLY answer key. It describes DIPOLE01, its
    bindings, its neighbours and the dipole census. The corpus also holds ~512
    devices this rubric says nothing about (correctors, quadrupoles,
    sextupoles, BPMs, vacuum and RF equipment). A claim about one of those is
    NOT a fabrication merely because this rubric does not list it — mark it
    wrong only if it contradicts something stated above.
  * Omitting anything the operator did not ask for is not a failure.

Do NOT grade formatting, verbosity, tone, or whether the agent went on to read
live values. Numerical position may be rounded or given in different units as
long as the ordering claim is right. FAIL if any of (1)-(4) is violated."""


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def _all_assistant_text(result: Any) -> str:
    """Every assistant text block of the run, joined — not just the last one.

    The agent narrates as it goes ("I'll delegate to the channel-finder…"), so
    this is the whole of its prose rather than its closing message. That makes
    the fabrication floor and the judge STRICTER than a final-message-only read:
    an invented address in an interim remark still counts against the run. The
    name says so, because "final message" would quietly overstate what a passing
    run proved.
    """
    return "\n".join(result.text_blocks).strip()


def _print_run(result: Any, prose: str) -> None:
    print("\n--- Graph MCP smoke ---")
    print(f"  prompt: {OPERATOR_PROMPT}")
    print(f"  mcp servers: {[(s.get('name'), s.get('status')) for s in result.mcp_servers]}")
    print(f"  tools called: {result.tool_names}")
    print(f"  num_turns: {result.num_turns}")
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
@pytest.mark.flaky(reruns=2)  # multi-step agentic + LLM judge; absorb stochastic misses
@pytest.mark.asyncio
async def test_graph_answers_a_facility_structure_question(graph_project: Path) -> None:
    """Operator asks what a PV belongs to; the agent reaches the graph and answers.

    The deterministic floor: the readiness barrier saw ``graph`` come up, the
    question reached a ``mcp__graph__*`` tool unprompted — and through the
    ``facility-knowledge-graph`` subagent, never the main agent directly — at
    least one such call succeeded, and the prose names no
    ``SR:MAG:DIPOLE:01:…`` address the corpus does not hold. The judge then
    grades the substance against :data:`_RUBRIC`, which is built from figures
    read out of this corpus.
    """
    render = render_dir(graph_project)
    assert "graph" in expected_mcp_servers(render), (
        "the readiness barrier would not wait for the graph server — .mcp.json "
        f"declares {sorted(expected_mcp_servers(render))}"
    )

    result = await run_sdk_query(graph_project, OPERATOR_PROMPT, max_turns=14, max_budget_usd=1.5)
    prose = _all_assistant_text(result)
    _print_run(result, prose)

    # --- The barrier's verdict -------------------------------------------
    # ``run_sdk_query`` polls ``get_mcp_status()`` against
    # ``expected_mcp_servers(render)`` before the first turn; its snapshot is
    # the authoritative infra-vs-model record, so an unconnected graph server is
    # reported as the infrastructure failure it is rather than as a model miss.
    graph_status = next(
        (s.get("status") for s in result.mcp_servers if s.get("name") == "graph"), None
    )
    assert graph_status == "connected", (
        f"the graph MCP server never reached 'connected' (status={graph_status!r}); "
        f"full snapshot: {[(s.get('name'), s.get('status')) for s in result.mcp_servers]}. "
        "This is an infrastructure failure, not a model miss."
    )

    # --- Floor 1: the agent reached the graph, unprompted -----------------
    graph_calls = [t for t in result.tool_traces if t.name.startswith("mcp__graph__")]
    assert graph_calls, (
        "agent answered a facility-structure question without ever reaching the "
        f"graph. Tools called: {result.tool_names}. If this is a wiring or "
        "tool-description gap, fix it in the framework — do not steer the prompt "
        "toward the tool."
    )
    direct = [t for t in graph_calls if t.parent_tool_use_id is None]
    assert not direct, (
        f"the main agent called {len(direct)} graph tool(s) itself instead of "
        f"delegating: {[t.name for t in direct]}. Those tools belong to the "
        "facility-knowledge-graph subagent."
    )

    successful = [t for t in graph_calls if not t.is_error and t.result]
    assert successful, (
        "the graph was reached but every call errored or returned nothing: "
        f"{[(t.name, t.is_error, (t.result or '')[:300]) for t in graph_calls]}"
    )

    assert prose, (
        "agent produced no assistant text at all — there is nothing to grade. "
        f"Tools called: {result.tool_names}"
    )

    # --- Floor 2: no invented channel addresses ---------------------------
    # Cheap, unspoofable, and it cannot fire on a correct terse answer: only a
    # complete address on THIS device's prefix that the corpus does not hold
    # trips it. Scoped over the agent's WHOLE prose, so an invented address in
    # an interim remark counts too. Dry-verified by the ``-k floor`` tests.
    fabricated = fabricated_addresses(prose)
    assert not fabricated, (
        f"the agent's prose names channel addresses this device does not have: "
        f"{fabricated}. The corpus binds exactly {list(PROBE_DEVICE_BINDINGS)} to it."
    )

    # --- Judge: the substance --------------------------------------------
    judge = LLMJudge(provider="als-apg")
    evaluation = await judge.evaluate(
        _to_workflow_result(OPERATOR_PROMPT, result), expectations=_RUBRIC
    )
    print("  --- judge ---")
    print(f"    passed: {evaluation.passed}  confidence: {evaluation.confidence:.2f}")
    print("\n".join("      " + line for line in evaluation.reasoning.splitlines()))
    assert evaluation.passed, evaluation.reasoning


# ---------------------------------------------------------------------------
# Offline dry-verification of the fabrication floor.
#
# No Docker, no API key, no agent session, no container — these grade
# :func:`fabricated_addresses`, the SAME helper the live test calls, against
# hand-written prose. Run them with
# ``pytest tests/e2e/test_graph_mcp_smoke.py -k floor``.
#
# They exist because both of this floor's edges have already been wrong once in
# development, and both failures were expensive to see: a live run, a container,
# a build and an LLM call each. ``harness_benchmark`` is the honest lane — the
# assertion is about OSPREY's own grading code and no model runs, so any model
# "passes" it.
# ---------------------------------------------------------------------------

#: ``(id, prose, expected fabrications)``. The PASS cases are the renderings a
#: real agent produced or plausibly will; the FAIL case is an invention.
_FLOOR_CASES: list[tuple[str, str, list[str]]] = [
    (
        "plain_address",
        f"The setpoint is {PROBE_PV} and the readback is SR:MAG:DIPOLE:01:CURRENT:RB.",
        [],
    ),
    (
        # The exact rendering that failed an earlier version of this floor.
        "markdown_bold_last_token",
        "| SR:MAG:DIPOLE:01:CURRENT:**SP** | Write |\n| SR:MAG:DIPOLE:01:CURRENT:**RB** | Read |",
        [],
    ),
    (
        "backticked",
        "`SR:MAG:DIPOLE:01:STATUS:READY` and `SR:MAG:DIPOLE:01:STATUS:FAULT` are read-only.",
        [],
    ),
    (
        "underscore_italics",
        "the flagged channel is SR:MAG:DIPOLE:01:CURRENT:_SP_, which is writable",
        [],
    ),
    (
        # Not a complete address, so out of this floor's grammar; the judge
        # grades prose like this, not the regex.
        "brace_shorthand",
        "the three flags SR:MAG:DIPOLE:01:STATUS:{ON,READY,FAULT}",
        [],
    ),
    (
        "words_not_addresses",
        "a current setpoint, a readback, a golden reference and three status flags",
        [],
    ),
    (
        "genuine_fabrication",
        "It also exposes SR:MAG:DIPOLE:01:VOLTAGE:RB for the supply voltage.",
        ["SR:MAG:DIPOLE:01:VOLTAGE:RB"],
    ),
    (
        # Another device's channels are none of this floor's business: it is
        # scoped to the one device whose full binding set is known.
        "other_device",
        "DIPOLE02 next door exposes SR:MAG:DIPOLE:02:CURRENT:SP and SR:MAG:DIPOLE:02:VOLTAGE:RB.",
        [],
    ),
    (
        # Seven tokens: NOT a real address, but the floor must not launder it as
        # one by matching its six-token prefix. Reporting nothing here is the
        # designed outcome — see _PROBE_DEVICE_PV_RE on why this edge is biased
        # toward the false negative, with the judge as the second grader.
        "deeper_token_invention",
        "and the limit lives at SR:MAG:DIPOLE:01:CURRENT:SP:LIMIT",
        [],
    ),
]


@pytest.mark.harness_benchmark
@pytest.mark.parametrize(
    ("prose", "expected"),
    [pytest.param(prose, expected, id=case_id) for case_id, prose, expected in _FLOOR_CASES],
)
def test_floor_reports_only_addresses_the_corpus_lacks(prose: str, expected: list[str]) -> None:
    """The fabrication floor flags inventions and nothing else."""
    assert fabricated_addresses(prose) == expected


@pytest.mark.harness_benchmark
def test_floor_is_not_vacuous_on_the_real_binding_set() -> None:
    """Every real address passes, and mutating one makes it fail.

    Non-vacuity for the whole set at once: a floor that silently matched nothing
    — a broken regex, a renamed constant — would pass every case above that
    expects ``[]``, which is most of them. Feeding it the six real addresses and
    then a corrupted copy of each proves it is looking.
    """
    assert fabricated_addresses(" ".join(PROBE_DEVICE_BINDINGS)) == []

    for real in PROBE_DEVICE_BINDINGS:
        mutated = real.replace(":CURRENT:", ":CURENT:").replace(":STATUS:", ":STATE:")
        assert mutated != real, f"mutation produced no change for {real}"
        assert fabricated_addresses(f"the channel {mutated} reports it") == [mutated]
