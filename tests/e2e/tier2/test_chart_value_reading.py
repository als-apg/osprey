"""Tier 2 model benchmark: reading a value off a chart that no text states.

The scenario: a recent beam-current dip in the archive against an older partial
loss that only a logbook plot records.

Tier 2 tests measure what a model can do in OSPREY rather than gate a change:
the task is fair, but a capable model may still miss it. They run on demand
(the ``tier2-model-benchmark`` CI job) and in the model matrix, never on a pull
request. No rerun marker: a benchmark scores every attempt.

The lower tiers already prove the road: find a channel, read the archive,
search the logbook. This test takes that road and measures what is at the end
of it, reading a quantity off a picture. The operator says the beam current
dipped in the last hour and asks how it compares with a similar partial loss in
the logbook. The recent dip is in the archive, planted as two step events so its
depth is exact. The older loss is a year old, long outside the archive's
retention, and exists only as a strip-chart export attached to one entry: a
decay-mode sawtooth on a 0-600 mA axis with one refill cycle sitting lower than
the rest. Nothing states its size; the agent has to read it against the axes.
The recent dip is the larger of the two, so an agent that assumes the logged
event must be the bigger one, or over-reads the plot, reverses the comparison.

The world is a real deployment (``osprey up --dev``): the virtual accelerator,
the MongoDB archiver with its recorder, ARIEL's Postgres and the qmd sidecar. A
mock archive would synthesize last year on request; a store with retention
answers honestly that it holds nothing there, which is what makes the picture the
only record. Both scenarios are facility scenario files, installed into the
deployment's ``data/facility/scenarios/`` before the build. The logbook scenario
is concealed after seeding (its file, its render copy, its simulator-view entry
and attached files, its active-state line), so the plot's numbers are not one
``Read`` away in the agent's cwd; the recent scenario stays, being the live
overlay, with the filesystem-search tools closed as in every scenario benchmark.

Grading has three layers. A deterministic floor over the trace: a channel-finder
tool ran, ``archiver_read`` ran on the DCCT, ``attachment_view`` ran on the
seeded picture inside the ``logbook-search`` subagent, no ARIEL tool erred. A
number check: the judge extracts the two magnitudes and the stated direction
without seeing the references, and code compares them with tolerances. A prose
check: the judge rules whether the answer compares the two events coherently.

Every planted number comes from :mod:`tests.e2e.fixtures.chart_value_reading.plot_spec`.

It runs on the benchmark board (the model matrix), not in a CI lane. The board
runs its cells in parallel on one host while this stack has one fixed project
name and port block, so the cells take turns on a host-wide lock and share the
images, which are rebuilt once per checked-out revision.

Run locally (~40 min cold, ~5 warm):
    OSPREY_E2E_PROVIDER=als-apg uv run pytest tests/e2e/tier2/test_chart_value_reading.py -v -s
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml
from pydantic import BaseModel, Field

from tests.e2e import _orm_stack
from tests.e2e._deploy_diagnostics import dead_container_logs
from tests.e2e._volumes import remove_project_volumes
from tests.e2e.ariel_picture import assert_no_ariel_errors, assert_viewed_in_logbook_turn
from tests.e2e.fixtures.chart_value_reading import plot_spec as fixture
from tests.e2e.judge import LLMJudge, Reference, WorkflowResult, check_reported
from tests.e2e.profile_edits import set_pairs
from tests.e2e.provider import build_model, e2e_provider
from tests.e2e.sdk_helpers import (
    HAS_SDK,
    SCENARIO_INTEGRITY_DISALLOWED_TOOLS,
    activate_scenarios,
    conceal_scenario_ground_truth,
    render_dir,
    run_sdk_query,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.agentic_benchmark,
    pytest.mark.requires_e2e_provider,
    pytest.mark.slow,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
    # The board's per-test timeout counts fixture setup: waiting for the host's
    # stack behind the other cells, plus one cold image build per revision.
    pytest.mark.timeout(7200),
]

PROJECT_NAME = "chart-value-reading"
JUDGE_PROVIDER = "als-apg"

# This module's own ports, away from the default 10000 block and from the other
# stack lanes (plan stack 21800, archiver world 27117/5164), so it can run beside
# a demo stack or another lane on one host.
PORT_BASE = 22800
VA_CA_PORT = 5264
VA_PVA_PORT = 5275
MONGO_PORT_HOST = 27217
POSTGRES_PORT_HOST = 15532

#: One stack of this lane per host at a time, and the revision its images were
#: built at. Both live beside each other in the host's temp directory.
STACK_LOCK = Path(tempfile.gettempdir()) / f"osprey-{PROJECT_NAME}.lock"
IMAGES_STAMP = Path(tempfile.gettempdir()) / f"osprey-{PROJECT_NAME}.images-revision"
REPO_ROOT = Path(__file__).resolve().parents[3]

BUILD_TIMEOUT_SEC = 600
DEPLOY_UP_TIMEOUT_SEC = 2400
DOWN_TIMEOUT_SEC = 300

#: What the trimmed profile must deploy: the two services it keeps plus the
#: three the virtual-accelerator and archiver injectors add at build.
DEPLOYED_SERVICES = ["archiver_recorder", "mongodb", "postgresql", "qmd", "virtual_accelerator"]

#: The operator's question: names the logbook and the quantity, hides the
#: channel, the window, the entry and both numbers.
PROMPT = (
    "The beam current dipped in the last hour. Is there a similar partial beam loss in the "
    "logbook, and how do the two compare in size?"
)

CHANNEL_FINDER_PREFIX = "mcp__channel-finder__"
ARCHIVER_TOOL = "mcp__controls__archiver_read"

#: The archive holds the planted depth exactly, under sample noise of about
#: 0.05 mA (one sigma): plateau means give the depth itself, and max minus min
#: over the samples overshoots it by a few tenths at most. So the archive value
#: gets an absolute allowance for that noise and nothing for reading.
RECENT_TOLERANCE_MA = 0.5

#: The plot-read value is a 24-pixel feature on a 350-pixel axis, so a quarter
#: either way separates a reading from a guess.
LOGGED_TOLERANCE = 0.25


class Reported(BaseModel):
    """What the extractor reads out of the answer; a field the text does not state stays None."""

    recent_drop_ma: float | None = Field(
        default=None, description="Size of the beam-current dip in the last hour, in mA"
    )
    logged_drop_ma: float | None = Field(
        default=None,
        description="Size of the partial beam loss the logbook entry's plot shows, in mA",
    )
    larger: Literal["recent", "logged", "same"] | None = Field(
        default=None, description="Which of the two losses the answer says was larger"
    )


EXTRACT_INSTRUCTIONS = (
    "recent_drop_ma: the size of the dip in the stored beam current during the last hour, "
    "as a positive number of mA. "
    "logged_drop_ma: the size of the partial beam loss shown on the logbook entry's plot, "
    "as a positive number of mA. "
    "larger: 'logged' if the answer says the logbook's loss was the bigger one, 'recent' if "
    "the recent dip was, 'same' if it calls them alike; null if it does not say. "
    "A size counts as stated when the answer gives it directly, gives the two levels it "
    "ran between (a fall from 300 to 290 mA is 10), or gives a range for it (use the "
    "midpoint). Do no other arithmetic, and leave a field null when none of these holds."
)

_LARGER_PHRASE = {
    "recent": "the recent dip in the archive",
    "logged": "the older loss on the logbook plot",
}[fixture.LARGER_LOSS]

JUDGE_EXPECTATIONS = (
    "Method and magnitudes were already checked in code; do not re-judge them. Judge the prose: "
    "the answer must compare the two events coherently, as one recent dip read from the "
    "archive and one older partial loss read from a plot attached to a logbook entry, and "
    f"state plainly that {_LARGER_PHRASE} was the larger of the two. It must not invent a "
    "cause for the recent dip, must not claim the archive holds the older event, and must "
    "not report 'no similar entry' or refuse to commit. A hedge that still names "
    f"{_LARGER_PHRASE} as larger passes; an answer that reverses the comparison fails."
)


@dataclass(frozen=True)
class DeployedWorld:
    """The one deployment and the ids the assertions need."""

    repo: Path
    attachment_id: str


def _profile_edits() -> dict[str, Any]:
    """The profile this lane deploys: VA, stored archive, ARIEL; nothing else.

    ``control_system.type`` is the virtual accelerator because the recorder
    records a VA and idles on a mock. ``archiver.type`` selects the store that
    ``va_archiver:`` declares (declaring alone would leave the agent reading the
    mock beside a full store). ARIEL stays on, so ``deployed_services`` keeps
    its Postgres and the qmd sidecar hybrid search answers from; the telemetry
    store and gateway go with telemetry, graphdb with the graph channel-finder
    paradigm (the hierarchical database the preset also ships answers instead).
    The bluesky stack, dispatch, the record archive and the web terminals are
    trimmed as the archiver-world lane trims them. The archive is CI-sized: two
    days of retention are a lifetime away from a 380-day-old entry.
    """
    return {
        "config": {
            "control_system.type": "virtual_accelerator",
            "archiver.type": "mongodb_archiver",
            "modules.web_terminals.enabled": False,
            "deployed_services": ["postgresql", "qmd"],
            "services.postgresql.port_host": POSTGRES_PORT_HOST,
            "claude_code.telemetry.enabled": False,
            "claude_code.servers.bluesky.enabled": False,
        },
        "channel_finder_mode": "hierarchical",
        "va_archiver": {
            **_orm_stack.VA_ARCHIVER_CI_KNOBS["va_archiver"],
            "port_host": MONGO_PORT_HOST,
        },
        "virtual_accelerator": {"live_standin": None},
        "bluesky": None,
        "bluesky_web": None,
        "dispatch": None,
        "services": {},
    }


def _run(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "CLAUDECODE": ""},
    )


def _init_args(repo: Path) -> list[str]:
    """``osprey init`` arguments (after the verb) for the deployment at *repo*.

    The model is named at build, not only per query: the build writes it into
    every subagent's definition, and the plot is read inside ``logbook-search``.
    Left unnamed, the provider catalog's default would read it whatever model
    the run asked for.
    """
    return [
        str(repo),
        "--preset",
        "control-assistant",
        "--no-git",
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
    """``osprey init`` + ``osprey build`` out of process; the scenarios go in between.

    Its own two verbs rather than the plan-stack builder, which pins a bluesky
    bridge this lane does not deploy.
    """
    osprey_bin = _orm_stack.find_osprey_console_script()
    repo = output_dir / PROJECT_NAME
    result = _run([str(osprey_bin), "init", *_init_args(repo)], output_dir, BUILD_TIMEOUT_SEC)
    assert result.returncode == 0, f"osprey init failed:\n{result.stdout}\n{result.stderr}"

    # Between the two verbs: the build reads <repo>/data/facility into the render
    # and its simulator view, so a scenario written now exists in every tree
    # concealment scrubs.
    fixture.install_scenarios(repo / "data" / "facility" / "scenarios")

    build = [
        str(osprey_bin),
        "build",
        "--repo",
        str(repo),
        "--skip-deps",
        "--skip-lifecycle",
        "--dev",
    ]
    result = _run(build, output_dir, BUILD_TIMEOUT_SEC)
    assert result.returncode == 0, f"osprey build failed:\n{result.stdout}\n{result.stderr}"

    _orm_stack.assert_off_default_block(repo, PROJECT_NAME)
    # An extra service lands on a host port a developer box may already hold,
    # and a missing one fails twenty minutes later as a model miss.
    config = yaml.safe_load((render_dir(repo) / "config.yml").read_text(encoding="utf-8"))
    built = sorted(config.get("deployed_services") or [])
    assert built == DEPLOYED_SERVICES, f"the built project deploys {built}"
    return repo


@contextmanager
def host_stack_lock(path: Path) -> Iterator[None]:
    """Hold this lane's host-wide stack lock for the duration of the block.

    An advisory ``flock``, so the operating system releases it when the holder
    dies: a crashed cell never leaves the host locked.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _checkout_revision() -> str | None:
    """The checked-out commit, or ``None`` when it cannot be read (always rebuild then)."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if head.returncode != 0:
        return None
    return head.stdout.strip() or None


def images_built_for(stamp: Path, revision: str | None) -> bool:
    """Whether this lane's images were last built at *revision*."""
    if revision is None or not stamp.is_file():
        return False
    return stamp.read_text(encoding="utf-8").strip() == revision


def record_images_built(stamp: Path, revision: str | None) -> None:
    """Record that this lane's images now match *revision*; an unknown one records nothing."""
    if revision is None:
        stamp.unlink(missing_ok=True)
        return
    stamp.write_text(revision, encoding="utf-8")


def _dead_containers() -> str:
    return dead_container_logs(_orm_stack.project_prefix(PROJECT_NAME))


async def _seeded_attachment_id(config: dict[str, Any]) -> str:
    """The target entry's picture id, checked viewable, read from the deployed ARIEL."""
    from osprey.port_layout import resolve_port_base
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database import create_connection_pool
    from osprey.services.ariel_search.database.repository import ARIELRepository

    ariel = ARIELConfig.from_dict(
        config["ariel"],
        (config.get("services") or {}).get("postgresql"),
        base=resolve_port_base(config),
    )
    pool = await create_connection_pool(ariel.database)
    try:
        repository = ARIELRepository(pool, ariel)
        rows = await repository.get_copy_rows(fixture.TARGET_ENTRY_ID)
        assert len(rows) == 1, f"expected one picture on {fixture.TARGET_ENTRY_ID}, got {rows}"
        row = rows[0]
        assert row["copy_status"] == "copied", (
            f"the seeded picture is {row['copy_status']!r}, not 'copied' "
            f"(skip_reason={row['skip_reason']!r}); attachment_view would answer not-ready"
        )
        attachment_id = str(row["attachment_id"])
        assert await repository.get_rendition(attachment_id) is not None, (
            f"the seeded picture {attachment_id} has no viewable rendition"
        )
        return attachment_id
    finally:
        await pool.close()


def _seed_world(repo: Path) -> str:
    """Apply both scenarios, conceal the logbook scenario, return the picture id.

    The seeding path reads ``services.postgresql`` from the process config and
    the ARIEL password from the environment, exactly as ``osprey sim apply``
    does when run from the repo; both are arranged here for the duration of the
    seed. The Mongo password is read from the repo's ``.env`` by the rewrite
    itself.
    """
    from osprey.utils.config import config_anchored_at, load_config
    from osprey.utils.dotenv import parse_dotenv_file

    env = parse_dotenv_file(repo / ".env")
    patch = pytest.MonkeyPatch()
    patch.setenv("ARIEL_DB_PASSWORD", env["ARIEL_DB_PASSWORD"])
    try:
        config_path = render_dir(repo) / "config.yml"
        with config_anchored_at(config_path):
            applied = activate_scenarios(repo, fixture.RECENT_SCENARIO, fixture.LOGBOOK_SCENARIO)
            assert applied.logbook_seeded >= 3, (
                f"logbook seeding wrote {applied.logbook_seeded} entries"
            )
            archive = applied.archiver
            assert archive is not None, "the stored archive was not rewritten"
            assert fixture.DCCT_CHANNEL in archive.channels, (
                f"the archive rewrite did not touch {fixture.DCCT_CHANNEL}: {archive}"
            )
            conceal_scenario_ground_truth(repo, fixture.LOGBOOK_SCENARIO)
            config = load_config(str(config_path))
            return asyncio.run(_seeded_attachment_id(config))
    finally:
        patch.undo()


@pytest.fixture(scope="module")
def deployed_world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DeployedWorld]:
    """Build, deploy, seed and conceal once; tear everything down after.

    Everything from the build to the teardown runs under the host-wide stack
    lock, so a parallel cell waits rather than colliding on the project name or
    the port block.
    """
    base = tmp_path_factory.mktemp("chart_value_reading_build")
    with host_stack_lock(STACK_LOCK):
        repo = _build_project(base)
        _orm_stack.seed_repo_env(repo)
        revision = _checkout_revision()
        if not images_built_for(IMAGES_STAMP, revision):
            _orm_stack.force_image_rebuild(_orm_stack.va_image(PROJECT_NAME))
        project = _orm_stack.project_prefix(PROJECT_NAME)
        # A volume left by a run that died before its teardown keeps that run's
        # minted credentials, and the next deploy is refused by its own store.
        remove_project_volumes(project)

        osprey_bin = _orm_stack.find_osprey_console_script()
        try:
            try:
                up = _run([str(osprey_bin), "up", "-d", "--dev"], repo, DEPLOY_UP_TIMEOUT_SEC)
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
            attachment_id = _seed_world(repo)
            yield DeployedWorld(repo=repo, attachment_id=attachment_id)
        finally:
            try:
                down = _run([str(osprey_bin), "down"], repo, DOWN_TIMEOUT_SEC)
                if down.returncode != 0:
                    print(f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}")
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"osprey down could not complete ({type(exc).__name__}: {exc})")
            remove_project_volumes(project)


def _excerpt(result: Any, final: str) -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": t.name,
                "input": t.input,
                "parent_tool_use_id": t.parent_tool_use_id,
                "is_error": t.is_error,
                "result": (t.result or "")[:400],
            }
            for t in result.tool_traces
        ],
        "answer": final,
        "num_turns": result.num_turns,
        "cost_usd": result.cost_usd,
    }


def _workflow_result(result: Any, final: str) -> WorkflowResult:
    lines = []
    for t in result.tool_traces:
        lines.append(f"TOOL: {t.name}  input={t.input}")
        if t.result:
            lines.append(f"  result: {t.result[:300]}")
    return WorkflowResult(
        query=PROMPT, response=final, execution_trace="\n".join(lines), artifacts=[]
    )


@pytest.mark.asyncio
async def test_agent_reads_a_loss_off_a_logbook_chart(
    deployed_world: DeployedWorld, tmp_path: Path
) -> None:
    """Asked to compare, the agent reads the archive for today and the plot for last year."""
    result = await run_sdk_query(
        deployed_world.repo,
        PROMPT,
        max_turns=30,
        max_budget_usd=10.0,
        disallowed_tools=SCENARIO_INTEGRITY_DISALLOWED_TOOLS,
    )
    final = (result.result.result if result.result is not None else None) or ""
    excerpt = json.dumps(_excerpt(result, final), indent=2, default=str)
    (tmp_path / "trace_excerpt.json").write_text(excerpt, encoding="utf-8")
    print(excerpt)

    assert result.result is not None, "no ResultMessage received"
    assert not result.result.is_error, f"the agent turn ended in error: {final}"
    traces = result.tool_traces

    # --- Floor: the road was travelled -----------------------------------------
    assert any(t.name.startswith(CHANNEL_FINDER_PREFIX) for t in traces), (
        f"no channel-finder tool ran; the DCCT was not discovered. Tools: {result.tool_names}"
    )
    archiver_reads = [t for t in traces if t.name == ARCHIVER_TOOL]
    assert archiver_reads, f"archiver_read never ran. Tools: {result.tool_names}"
    assert any(fixture.DCCT_CHANNEL in json.dumps(t.input or {}) for t in archiver_reads), (
        f"archiver_read never asked for {fixture.DCCT_CHANNEL}: {[t.input for t in archiver_reads]}"
    )
    assert_viewed_in_logbook_turn(traces, deployed_world.attachment_id, result.tool_names)
    assert_no_ariel_errors(traces)

    # --- Numbers: the judge extracts, code compares ------------------------------
    judge = LLMJudge(provider=JUDGE_PROVIDER)
    reported = await judge.extract(final, Reported, EXTRACT_INSTRUCTIONS)
    misses = check_reported(
        {"recent_drop_ma": reported.recent_drop_ma, "logged_drop_ma": reported.logged_drop_ma},
        {
            "recent_drop_ma": Reference(fixture.RECENT_DIP_MA, RECENT_TOLERANCE_MA),
            "logged_drop_ma": Reference(fixture.DIP_DEPTH_MA, LOGGED_TOLERANCE, relative=True),
        },
    )
    assert not misses, "; ".join(misses) + f"\nanswer: {final!r}"
    assert reported.larger == fixture.LARGER_LOSS, (
        f"the answer must say the {fixture.LARGER_LOSS} loss was larger; "
        f"extracted {reported.larger!r}: {final!r}"
    )

    # --- Prose: the comparison holds together ------------------------------------
    verdict = await judge.evaluate(_workflow_result(result, final), expectations=JUDGE_EXPECTATIONS)
    assert verdict.passed, verdict.reasoning
