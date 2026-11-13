"""Tier 2 model benchmark: finding PVs and IOCs, curated list first, the full list after.

The scenarios run against the demo's virtual accelerator, where the control
system's own list of PVs (All PVs) holds every channel the curated list declares
plus the simulator's IOC health PVs, which only the full list has. Each scenario
records the PV and IOC it expects:

* curated hits: the curated channel finder knows the PV;
* curated misses: only the full list has the PV, found by its description
  (the IOC health PVs, ``<code>:SIM:<model>:{HEARTBEAT,UPTIME,ENGINEER}``);
* IOC questions: host, health PVs, engineer, aliveness;
* alarm state now, answered from the demo's replayed alarm feed: one latched,
  one acknowledged and one cleared alarm.

Alarm and put history have no source on the simulator and are not in this set.

A scenario passes on the facts of its answer. How the agent got there is
reported as scores, never asserted: whether it asked the curated channel finder
before the full list, whether it labels a full-list PV as a control-system name,
and, when the question asks to see a PV or IOC, whether it opened the panel (an
``open_panel`` call with a tool result's ``links.panel`` URL). A link merely
returned in the answer is a separate number. Each test records its own scores
as properties of its result; the module prints the totals at teardown.

A headless run has no browser attached to the web terminal, so every
``open_panel`` call answers an error envelope; the call itself is what is
scored.

The world is a real deployment (``osprey up --dev``) trimmed to the virtual
accelerator. It runs on the benchmark board (the model matrix), not in a CI
lane. The board runs its cells in parallel on one host while this stack has one
fixed project name and port block, so the cells take turns on a host-wide lock
and share the images, which are rebuilt once per checked-out revision.

Run locally:
    OSPREY_E2E_PROVIDER=als-apg uv run pytest tests/e2e/tier2/test_pv_directory_agentic.py -v -s
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml
from pydantic import BaseModel, Field

from osprey.services.pv_directory.facts import panel_link
from tests.e2e import _orm_stack
from tests.e2e._deploy_diagnostics import dead_container_logs
from tests.e2e._deploy_support import (
    checkout_revision,
    host_stack_lock,
    images_built_for,
    record_images_built,
    run_command,
)
from tests.e2e._volumes import remove_project_volumes
from tests.e2e.judge import LLMJudge
from tests.e2e.profile_edits import e2e_scenario_pin, set_pairs
from tests.e2e.provider import build_model, e2e_provider
from tests.e2e.sdk_helpers import (
    HAS_SDK,
    SCENARIO_INTEGRITY_DISALLOWED_TOOLS,
    render_dir,
    run_sdk_query,
)

pytestmark = [
    pytest.mark.requires_e2e_provider,
    pytest.mark.slow,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
    # The board's per-test timeout counts fixture setup: waiting for the host's
    # stack behind the other cells, plus one cold image build per revision.
    pytest.mark.timeout(7200),
]

PROJECT_NAME = "pv-directory-agentic"
JUDGE_PROVIDER = "als-apg"

# This module's own ports, away from the default 10000 block, the board's cell
# blocks and the other board stacks (plan stack 21800, chart value reading 22800
# with its VA on 5264/5275), so it can run beside them on one host.
PORT_BASE = 20300
VA_CA_PORT = 5364
VA_PVA_PORT = 5375

#: One stack of this lane per host at a time, and the revision its images were
#: built at. Both live beside each other in the host's temp directory.
STACK_LOCK = Path(tempfile.gettempdir()) / f"osprey-{PROJECT_NAME}.lock"
IMAGES_STAMP = Path(tempfile.gettempdir()) / f"osprey-{PROJECT_NAME}.images-revision"

BUILD_TIMEOUT_SEC = 600
DEPLOY_UP_TIMEOUT_SEC = 2400
DOWN_TIMEOUT_SEC = 300

#: What the trimmed profile must deploy: the virtual accelerator alone.
DEPLOYED_SERVICES = ["virtual_accelerator"]

CHANNEL_FINDER_PREFIX = "mcp__channel-finder__"
PV_DIRECTORY_PREFIX = "mcp__pv-directory__"
OPEN_PANEL = "mcp__osprey_workspace__open_panel"
CHANNEL_READ = "mcp__controls__channel_read"

#: The value the simulator serves on every IOC's ENGINEER PV.
DEMO_ENGINEER = "demo engineer"
#: The host the simulator's directory gives every IOC.
SIMULATOR_HOST = "virtual-accelerator"


@dataclass(frozen=True)
class Expected:
    """What a scenario expects: the PV and IOC it is about, and what it asks of the agent.

    ``curated`` is True when the curated channel list declares the PV, False when
    only the full list has it, None for a question about an IOC. ``show`` names
    the panel page the question asks to see.
    """

    pv: str
    ioc: str
    curated: bool | None
    show: Literal["pv", "ioc"] | None = None


@dataclass
class Scoreboard:
    """Each score's results over the scenarios it applies to."""

    scores: dict[str, list[bool]] = field(default_factory=dict)

    def add(self, name: str, value: bool | None) -> None:
        if value is not None:
            self.scores.setdefault(name, []).append(value)

    def summary(self) -> dict[str, str]:
        return {name: f"{sum(values)}/{len(values)}" for name, values in self.scores.items()}


def _profile_edits() -> dict[str, Any]:
    """The profile this lane deploys: the virtual accelerator and nothing else.

    The knowledge services go as every substrate lane drops them (the channel
    finder answers from the hierarchical database the bundle ships). The stored
    archive, the live stand-in, the bluesky stack, dispatch, the record archive
    and the web terminals are trimmed as the chart-value-reading lane trims
    them. The preset's ``pv_directory`` section, with its replayed alarm feed,
    stays as the preset ships it.
    """
    return _orm_stack.knowledge_services_off(
        {
            "config": {
                "control_system.type": "virtual_accelerator",
                "modules.web_terminals.enabled": False,
                "claude_code.servers.bluesky.enabled": False,
            },
            "va_archiver": None,
            "virtual_accelerator": {"live_standin": None},
            "bluesky": None,
            "bluesky_web": None,
            "dispatch": None,
            "services": {},
        }
    )


def _init_args(repo: Path) -> list[str]:
    """``osprey init`` arguments (after the verb) for the deployment at *repo*.

    The model is named at build, not only per query: the build writes it into
    every subagent's definition. Left unnamed, the provider catalog's default
    would answer inside a subagent whatever model the run asked for.
    """
    return [
        str(repo),
        "--preset",
        "demo",
        "--no-git",
        *e2e_scenario_pin(),
        *set_pairs(_profile_edits()),
        "--set",
        f"virtual_accelerator.port={VA_CA_PORT}",
        "--set",
        f"virtual_accelerator.pva_port={VA_PVA_PORT}",
        "--set",
        f"port_base={PORT_BASE}",
        "--set",
        f"provider={e2e_provider()}",
        "--set",
        f"model={build_model(None)}",
    ]


def _build_project(output_dir: Path) -> Path:
    """``osprey init`` + ``osprey build`` out of process."""
    osprey_bin = _orm_stack.find_osprey_console_script()
    repo = output_dir / PROJECT_NAME
    result = run_command(
        [str(osprey_bin), "init", *_init_args(repo)], output_dir, BUILD_TIMEOUT_SEC
    )
    assert result.returncode == 0, f"osprey init failed:\n{result.stdout}\n{result.stderr}"
    build = [
        str(osprey_bin),
        "build",
        "--repo",
        str(repo),
        "--skip-deps",
        "--skip-lifecycle",
        "--dev",
    ]
    result = run_command(build, output_dir, BUILD_TIMEOUT_SEC)
    assert result.returncode == 0, f"osprey build failed:\n{result.stdout}\n{result.stderr}"

    _orm_stack.assert_off_default_block(repo, PROJECT_NAME)
    # An extra service lands on a host port a developer box may already hold,
    # and a missing one fails twenty minutes later as a model miss.
    config = yaml.safe_load((render_dir(repo) / "config.yml").read_text(encoding="utf-8"))
    built = sorted(config.get("deployed_services") or [])
    assert built == DEPLOYED_SERVICES, f"the built project deploys {built}"
    assert config.get("pv_directory"), "the built project has no pv_directory section"
    return repo


def _dead_containers() -> str:
    return dead_container_logs(_orm_stack.project_prefix(PROJECT_NAME))


@pytest.fixture(scope="module")
def deployed_repo(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Build and deploy once; tear everything down after.

    Everything from the build to the teardown runs under the host-wide stack
    lock, so a parallel cell waits rather than colliding on the project name or
    the port block.
    """
    base = tmp_path_factory.mktemp("pv_directory_agentic_build")
    with host_stack_lock(STACK_LOCK):
        repo = _build_project(base)
        _orm_stack.seed_repo_env(repo)
        revision = checkout_revision()
        if not images_built_for(IMAGES_STAMP, revision):
            _orm_stack.force_image_rebuild(_orm_stack.va_image(PROJECT_NAME))
        project = _orm_stack.project_prefix(PROJECT_NAME)
        remove_project_volumes(project)

        osprey_bin = _orm_stack.find_osprey_console_script()
        try:
            try:
                up = run_command(
                    [str(osprey_bin), "up", "-d", "--dev"], repo, DEPLOY_UP_TIMEOUT_SEC
                )
            except subprocess.TimeoutExpired as exc:
                pytest.fail(
                    f"osprey up -d --dev timed out after {DEPLOY_UP_TIMEOUT_SEC}s:\n"
                    f"{exc.stdout}\n{exc.stderr}\n"
                    f"--- containers not running ---\n{_dead_containers()}"
                )
            if up.returncode != 0:
                pytest.fail(
                    f"osprey up -d --dev failed (rc={up.returncode}):\n{up.stdout}\n{up.stderr}\n"
                    f"--- containers not running ---\n{_dead_containers()}"
                )
            record_images_built(IMAGES_STAMP, revision)
            yield repo
        finally:
            try:
                down = run_command([str(osprey_bin), "down"], repo, DOWN_TIMEOUT_SEC)
                if down.returncode != 0:
                    print(f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}")
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"osprey down could not complete ({type(exc).__name__}: {exc})")
            remove_project_volumes(project)


@pytest.fixture(scope="module")
def scoreboard(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Scoreboard]:
    """The set's scores; printed and written beside the run's temp files at teardown."""
    board = Scoreboard()
    yield board
    summary = board.summary()
    path = tmp_path_factory.getbasetemp() / "pv_directory_scores.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"PV directory benchmark scores: {summary} ({path})")


# --- Reading the trace ---------------------------------------------------------


def _lookups(result: Any) -> list[str]:
    """The lookup tools the run called, in order: curated finder and full list."""
    return [
        t.name
        for t in result.tool_traces
        if t.name.startswith((CHANNEL_FINDER_PREFIX, PV_DIRECTORY_PREFIX))
    ]


def _curated_first(result: Any) -> bool:
    """Whether the first lookup went to the curated channel finder."""
    lookups = _lookups(result)
    return bool(lookups) and lookups[0].startswith(CHANNEL_FINDER_PREFIX)


def _returned_links(result: Any) -> set[str]:
    """Every ``links.panel`` URL a directory tool returned."""
    links: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            panel = node.get("links")
            if isinstance(panel, dict) and isinstance(panel.get("panel"), str):
                links.add(panel["panel"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for t in result.tool_traces:
        if not t.name.startswith(PV_DIRECTORY_PREFIX) or not t.result:
            continue
        try:
            walk(json.loads(t.result))
        except (TypeError, ValueError):
            continue
    return links


def _opened_panel(result: Any, link: str) -> bool:
    """An ``open_panel`` call with *link*, as a directory tool returned it.

    Scored on the call, whatever envelope it answered: a headless run has no
    browser for the panel to open in.
    """
    if link not in _returned_links(result):
        return False
    return any(
        t.name == OPEN_PANEL and (t.input or {}).get("url") == link for t in result.tool_traces
    )


def _read_channel(result: Any, address: str) -> bool:
    return any(
        t.name == CHANNEL_READ and address in json.dumps(t.input or {}) for t in result.tool_traces
    )


class Labelling(BaseModel):
    """Whether the answer says where the PV name comes from."""

    directory_label: bool | None = Field(
        default=None,
        description="True if the answer marks the PV as a control-system name from the full list",
    )


LABELLING_INSTRUCTIONS = (
    "directory_label: true if the answer tells the reader that the PV it names comes from the "
    "control system's own full list of PV names (the directory, All PVs, a name the control "
    "system announces) rather than from the curated channel list, or otherwise marks it as "
    "not curated; false if it presents the PV without saying where it comes from or calls it "
    "curated."
)


async def _ask(
    request: pytest.FixtureRequest,
    repo: Path,
    tmp_path: Path,
    expected: Expected,
    scoreboard: Scoreboard,
    record_property: Any,
) -> tuple[Any, str]:
    """Run the scenario's question (its marker's ``asks``), score the run, return it and the answer."""
    prompt = request.node.get_closest_marker("agentic").kwargs["asks"]
    result = await run_sdk_query(
        repo,
        prompt,
        max_turns=30,
        max_budget_usd=5.0,
        disallowed_tools=SCENARIO_INTEGRITY_DISALLOWED_TOOLS,
    )
    final = (result.result.result if result.result is not None else None) or ""
    excerpt = {
        "expected": {"pv": expected.pv, "ioc": expected.ioc},
        "tools": [
            {
                "name": t.name,
                "input": t.input,
                "is_error": t.is_error,
                "result": (t.result or "")[:400],
            }
            for t in result.tool_traces
        ],
        "answer": final,
    }
    (tmp_path / "trace_excerpt.json").write_text(
        json.dumps(excerpt, indent=2, default=str), encoding="utf-8"
    )

    scores: dict[str, bool | None] = {
        "curated_first": _curated_first(result) if expected.curated is not None else None,
        "labelled": None,
        "opened_panel": None,
        "link_returned": None,
    }
    if expected.curated is False:
        judge = LLMJudge(provider=JUDGE_PROVIDER)
        labelling = await judge.extract(final, Labelling, LABELLING_INSTRUCTIONS)
        scores["labelled"] = bool(labelling.directory_label)
    if expected.show is not None:
        link = panel_link(expected.show, expected.pv if expected.show == "pv" else expected.ioc)[
            "panel"
        ]
        scores["opened_panel"] = _opened_panel(result, link)
        scores["link_returned"] = link in final
    for name, value in scores.items():
        scoreboard.add(name, value)
        if value is not None:
            record_property(name, value)
    record_property("expected_pv", expected.pv)
    record_property("expected_ioc", expected.ioc)
    print(json.dumps({"answer": final, "scores": scores, "tools": result.tool_names}, default=str))

    assert result.result is not None, "no ResultMessage received"
    assert not result.result.is_error, f"the agent turn ended in error: {final}"
    return result, final


def _assert_names(final: str, *names: str) -> None:
    missing = [name for name in names if name not in final]
    assert not missing, f"the answer does not name {missing}: {final!r}"


# --- Curated hits ----------------------------------------------------------------


@pytest.mark.agentic(
    tier=2,
    title="Find a curated PV",
    asks="Which PV reads the storage ring beam lifetime?",
    passes_if="names SR:DIAG:DCCT:01:LIFETIME:RB",
)
@pytest.mark.asyncio
async def test_curated_hit_beam_lifetime(
    request, deployed_repo, tmp_path, scoreboard, record_property
) -> None:
    expected = Expected(pv="SR:DIAG:DCCT:01:LIFETIME:RB", ioc="sim-texture", curated=True)
    _, final = await _ask(request, deployed_repo, tmp_path, expected, scoreboard, record_property)
    _assert_names(final, expected.pv)


@pytest.mark.agentic(
    tier=2,
    title="Show a curated PV",
    asks="Show me the forward power PV of storage ring RF cavity 1.",
    passes_if="names SR:RF:CAVITY:01:POWER:FWD",
)
@pytest.mark.asyncio
async def test_curated_hit_shown(
    request, deployed_repo, tmp_path, scoreboard, record_property
) -> None:
    expected = Expected(pv="SR:RF:CAVITY:01:POWER:FWD", ioc="sim-texture", curated=True, show="pv")
    _, final = await _ask(request, deployed_repo, tmp_path, expected, scoreboard, record_property)
    _assert_names(final, expected.pv)


# --- Curated misses the full list has by description ---------------------------


@pytest.mark.agentic(
    tier=2,
    title="Find a PV only the full list has",
    asks="Which PV tells how long the storage-ring IOC, sim-SR, has been running?",
    passes_if="names ca:SIM:SR:UPTIME",
)
@pytest.mark.asyncio
async def test_curated_miss_ioc_uptime(
    request, deployed_repo, tmp_path, scoreboard, record_property
) -> None:
    expected = Expected(pv="ca:SIM:SR:UPTIME", ioc="sim-SR", curated=False)
    _, final = await _ask(request, deployed_repo, tmp_path, expected, scoreboard, record_property)
    _assert_names(final, expected.pv)


@pytest.mark.agentic(
    tier=2,
    title="Show a PV only the full list has",
    asks="Show me the heartbeat PV of IOC sim-BTS.",
    passes_if="names ca:SIM:BTS:HEARTBEAT",
)
@pytest.mark.asyncio
async def test_curated_miss_shown(
    request, deployed_repo, tmp_path, scoreboard, record_property
) -> None:
    expected = Expected(pv="ca:SIM:BTS:HEARTBEAT", ioc="sim-BTS", curated=False, show="pv")
    _, final = await _ask(request, deployed_repo, tmp_path, expected, scoreboard, record_property)
    _assert_names(final, expected.pv)


# --- IOC questions -----------------------------------------------------------------


class Aliveness(BaseModel):
    """Whether the answer says the IOC is alive."""

    alive: bool | None = Field(default=None, description="The answer says the IOC is alive")


@pytest.mark.agentic(
    tier=2,
    title="Is an IOC alive",
    asks="Is the storage-ring IOC, sim-SR, alive?",
    passes_if="reads ca:SIM:SR:HEARTBEAT and answers alive",
)
@pytest.mark.asyncio
async def test_ioc_alive(request, deployed_repo, tmp_path, scoreboard, record_property) -> None:
    expected = Expected(pv="ca:SIM:SR:HEARTBEAT", ioc="sim-SR", curated=None)
    result, final = await _ask(
        request, deployed_repo, tmp_path, expected, scoreboard, record_property
    )
    assert _read_channel(result, expected.pv), (
        f"{CHANNEL_READ} never read {expected.pv}. Tools: {result.tool_names}"
    )
    judge = LLMJudge(provider=JUDGE_PROVIDER)
    reported = await judge.extract(
        final,
        Aliveness,
        "alive: true if the answer says the IOC is alive or running, false if it says it is "
        "not, null if it does not commit.",
    )
    assert reported.alive is True, f"the answer does not say sim-SR is alive: {final!r}"


@pytest.mark.agentic(
    tier=2,
    title="IOC host and engineer",
    asks="Which host does IOC sim-LTB run on, and who is its engineer?",
    passes_if="answers host virtual-accelerator and engineer 'demo engineer'",
)
@pytest.mark.asyncio
async def test_ioc_host_and_engineer(
    request, deployed_repo, tmp_path, scoreboard, record_property
) -> None:
    expected = Expected(pv="ca:SIM:LTB:ENGINEER", ioc="sim-LTB", curated=None)
    _, final = await _ask(request, deployed_repo, tmp_path, expected, scoreboard, record_property)
    lowered = final.casefold()
    assert SIMULATOR_HOST in lowered, f"the answer does not name host {SIMULATOR_HOST}: {final!r}"
    assert DEMO_ENGINEER in lowered, f"the answer does not name {DEMO_ENGINEER!r}: {final!r}"


@pytest.mark.agentic(
    tier=2,
    title="Show an IOC's health PVs",
    asks="Show me IOC sim-SR and list its health PVs.",
    passes_if="names ca:SIM:SR:HEARTBEAT, ca:SIM:SR:UPTIME and ca:SIM:SR:ENGINEER",
)
@pytest.mark.asyncio
async def test_ioc_health_pvs_shown(
    request, deployed_repo, tmp_path, scoreboard, record_property
) -> None:
    expected = Expected(pv="ca:SIM:SR:HEARTBEAT", ioc="sim-SR", curated=None, show="ioc")
    _, final = await _ask(request, deployed_repo, tmp_path, expected, scoreboard, record_property)
    _assert_names(final, "ca:SIM:SR:HEARTBEAT", "ca:SIM:SR:UPTIME", "ca:SIM:SR:ENGINEER")


# --- Alarm state now, from the demo's replayed feed ------------------------------


class AlarmAnswer(BaseModel):
    """The alarm state the answer reports."""

    severity: Literal["MAJOR", "MINOR", "INVALID", "OK"] | None = Field(
        default=None, description="The alarm severity the answer reports; OK for no alarm"
    )
    latched: bool | None = Field(
        default=None, description="The answer says the alarm is latched (condition cleared)"
    )
    acknowledged: bool | None = Field(
        default=None, description="The answer says the alarm is acknowledged"
    )


ALARM_INSTRUCTIONS = (
    "severity: the alarm severity the answer reports for the channel (MAJOR, MINOR, INVALID), "
    "or OK when it says the channel is not in alarm. latched: true if the answer says the alarm "
    "is latched, that is it stays raised although the condition has cleared; false if it says "
    "the alarm is not latched or is still active; null if it does not say. acknowledged: true "
    "if the answer says an operator acknowledged the alarm, false if it says it is "
    "unacknowledged, null if it does not say."
)


async def _alarm_answer(final: str) -> AlarmAnswer:
    return await LLMJudge(provider=JUDGE_PROVIDER).extract(final, AlarmAnswer, ALARM_INSTRUCTIONS)


@pytest.mark.agentic(
    tier=2,
    title="A latched alarm",
    asks="Is the storage ring RF cavity 1 temperature in alarm right now?",
    passes_if="answers a latched, unacknowledged MAJOR alarm on SR:RF:CAVITY:01:TEMPERATURE:RB",
)
@pytest.mark.asyncio
async def test_alarm_latched(request, deployed_repo, tmp_path, scoreboard, record_property) -> None:
    expected = Expected(pv="SR:RF:CAVITY:01:TEMPERATURE:RB", ioc="sim-texture", curated=True)
    _, final = await _ask(request, deployed_repo, tmp_path, expected, scoreboard, record_property)
    _assert_names(final, expected.pv)
    reported = await _alarm_answer(final)
    assert (reported.severity, reported.latched, reported.acknowledged) == ("MAJOR", True, False), (
        f"extracted {reported}: {final!r}"
    )


@pytest.mark.agentic(
    tier=2,
    title="An acknowledged alarm",
    asks="What is the alarm state of the storage ring sector 3 vacuum gauge pressure right now?",
    passes_if="answers an acknowledged MINOR alarm on SR:VAC:GAUGE:SR03:PRESSURE:RB",
)
@pytest.mark.asyncio
async def test_alarm_acknowledged(
    request, deployed_repo, tmp_path, scoreboard, record_property
) -> None:
    expected = Expected(pv="SR:VAC:GAUGE:SR03:PRESSURE:RB", ioc="sim-texture", curated=True)
    _, final = await _ask(request, deployed_repo, tmp_path, expected, scoreboard, record_property)
    _assert_names(final, expected.pv)
    reported = await _alarm_answer(final)
    assert (reported.severity, reported.acknowledged) == ("MINOR", True), (
        f"extracted {reported}: {final!r}"
    )


@pytest.mark.agentic(
    tier=2,
    title="A cleared alarm",
    asks="Is the heartbeat PV of IOC sim-SR in alarm right now?",
    passes_if="answers no alarm on ca:SIM:SR:HEARTBEAT",
)
@pytest.mark.asyncio
async def test_alarm_cleared(request, deployed_repo, tmp_path, scoreboard, record_property) -> None:
    expected = Expected(pv="ca:SIM:SR:HEARTBEAT", ioc="sim-SR", curated=False)
    _, final = await _ask(request, deployed_repo, tmp_path, expected, scoreboard, record_property)
    _assert_names(final, expected.pv)
    reported = await _alarm_answer(final)
    assert reported.severity == "OK", f"extracted {reported}: {final!r}"
