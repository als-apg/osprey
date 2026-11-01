"""The agent engine end to end, with the model replaced by a script.

A judge's run is real from :func:`~osprey.agent_runner.stream_query` down: the
pinned Claude Code, the ``agent-profile`` fixture built as a deployment, the
control-system MCP server and the in-process simulator. Only the model answers
from a script, so a forbidden call is made on every run, and its verdict is
computed from what the channel read returned, so a flag says the seed reached
the judge's own read.

Writes are armed in this deployment. What stops a judge's write here is the
judge's read-only run, not the deployment's posture, and a seed is an ordinary
connector write.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.watchdog_cmd import _MachineSeed
from osprey.services.watchdog.config import WatchdogDefinition
from osprey.services.watchdog.definitions import load_profile_watchdogs
from osprey.services.watchdog.engines.agent import AgentEngine
from osprey_connectors.control_system._va_in_process_connector import (
    SIMULATOR_VIEW_SETTING,
    VAInProcessConnector,
)
from tests._preset_data import copy_bundle_data
from tests.e2e.scripted_model import ScriptedCall, ScriptedModel
from tests.e2e.sdk_helpers import enable_writes_in_project, render_dir

pytestmark = [pytest.mark.model_free, pytest.mark.harness_benchmark]

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "watchdog"
PAIR_DIR = FIXTURES / "agent"

READER = "agent-reader"
READ = "mcp__controls__channel_read"
WRITE = "mcp__controls__channel_write"

#: The channel the fixture pair's verdict turns on, and the value its reference
#: names as the limit.
PRESSURE = "SR03:VAC:PRESSURE:RB"
PRESSURE_LIMIT = 1e-7
#: A setpoint of the hello_world tree, for a judge to try to move.
SETPOINT = "SR:MAG:QF:01:CURRENT:SP"
#: Far outside anything the setpoint serves, so a write that landed shows.
WRITTEN = 123.0


def read_pressure() -> ScriptedCall:
    return ScriptedCall(READ, {"channels": [PRESSURE], "include_metadata": False})


def build_deployment(
    root: Path, *, provider: str | None = None, model: str | None = None
) -> tuple[Path, Path]:
    """Build the ``agent-profile`` fixture as a deployment; return its repo and render.

    The fixture lays the hello_world data beside its profile, and this adds the
    pair's pressure channel to that tree. It is a writable setpoint there because
    a seed is a write; a judge's own tools still decide what a judge may do with it.

    Args:
        root: An empty directory for the repo.
        provider: The provider the deployment builds with, else the fixture's.
        model: The model it builds with, else the fixture's.
    """
    repo = root / "repo"
    shutil.copytree(FIXTURES / "agent-profile", repo)
    copy_bundle_data(repo / "data", "hello_world")
    records_path = repo / "data" / "facility" / "records" / "channels.yaml"
    records = yaml.safe_load(records_path.read_text(encoding="utf-8"))
    records.append(
        {
            "id": PRESSURE,
            "role": "setpoint",
            "unit": "torr",
            "label": "sector 3 vacuum pressure",
            "description": "Vacuum pressure in sector 3",
        }
    )
    records_path.write_text(yaml.safe_dump(records, sort_keys=False), encoding="utf-8")
    limits_path = repo / "data" / "facility" / "limits.yaml"
    limits = yaml.safe_load(limits_path.read_text(encoding="utf-8"))
    limits["records"].append(
        {"address": PRESSURE, "min_value": 0.0, "max_value": 1.0, "writable": True}
    )
    limits_path.write_text(yaml.safe_dump(limits, sort_keys=False), encoding="utf-8")
    profile_path = repo / "profile.yml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["provider"] = provider or profile["provider"]
    profile["model"] = model or profile["model"]
    # A deployment that allows writes must install every gate on its write tools.
    profile["hooks"] = [*profile["hooks"], "approval", "limits"]
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    enable_writes_in_project(repo)
    return repo, render_dir(repo)


def reader_definition(repo: Path) -> WatchdogDefinition:
    """The fixture's ``agent-reader``, switched on as a replay switches it."""
    definitions = {defn.name: defn for defn in load_profile_watchdogs(repo)}
    return replace(definitions[READER], enabled=True)


def read_pair_fixture(label: str) -> tuple[str, list[dict[str, Any]], Mapping[str, Any]]:
    """The prompt, session nodes and machine values of one fixture of the pair."""
    lines = (PAIR_DIR / f"gate-valve-{label}.jsonl").read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    return header["prompt"], [json.loads(line) for line in lines[1:]], header["machine"]


@asynccontextmanager
async def seeded(build: Path, machine: Mapping[str, Any]) -> AsyncIterator[_MachineSeed]:
    """The simulator holding *machine*; leaving hands it back in its start state.

    Leaving resets every channel, so a check that a channel kept its value belongs
    inside this block: past it a write that landed no longer shows.
    """
    seed = _MachineSeed(rendered_config(build), build)
    try:
        why_not = await seed.apply(machine)
        assert why_not is None, why_not
        yield seed
    finally:
        await seed.close()


async def read_channel(build: Path, channel: str) -> float:
    """What a second connector reads, the way a judge's own server would."""
    connector = VAInProcessConnector()
    await connector.connect({SIMULATOR_VIEW_SETTING: str(build / "data" / "simulator")})
    try:
        return float((await connector.read_channel(channel)).value)
    finally:
        await connector.disconnect()


def rendered_config(build: Path) -> dict[str, Any]:
    return yaml.safe_load((build / "config.yml").read_text(encoding="utf-8"))


def _result_text(block: Mapping[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


def _read_pressure(tool_results: list[dict[str, Any]]) -> float | None:
    """The pressure a channel read in this conversation returned, if one did."""
    pattern = re.escape(PRESSURE) + r".*?value\W+(-?[\d.]+(?:[eE][-+]?\d+)?)"
    for block in tool_results:
        found = re.search(pattern, _result_text(block), re.DOTALL)
        if found is not None and not block.get("is_error"):
            return float(found.group(1))
    return None


def answer_from_the_read(model: ScriptedModel) -> None:
    """Make *model* end a run with the verdict the channel read calls for.

    Its turns stay as scripted. A conversation with no reading in it answers
    with no verdict document at all, so a judge that never read cannot pass.
    """
    scripted = model.answer

    def answer(body: dict[str, Any], credential: str | None):
        blocks, stop = scripted(body, credential)
        if stop == "end_turn" and body.get("tools"):
            messages = [m for m in body.get("messages") or [] if isinstance(m, dict)]
            results = [
                block
                for message in messages
                if message.get("role") == "user" and isinstance(message.get("content"), list)
                for block in message["content"]
                if isinstance(block, dict) and block.get("type") == "tool_result"
            ]
            value = _read_pressure(results)
            if value is not None:
                verdict = {
                    "flagged": value >= PRESSURE_LIMIT,
                    "explanation": f"{PRESSURE} reads {value:g} torr.",
                    "sources": [PRESSURE],
                }
                blocks = [{"type": "text", "text": json.dumps(verdict)}]
        return blocks, stop

    model.answer = answer  # type: ignore[method-assign]


@dataclass(frozen=True)
class JudgeRun:
    """One judge's run, with :data:`SETPOINT` as the seeded simulator held it around the run."""

    verdict: Any
    model: ScriptedModel
    setpoint_before: float
    setpoint_after: float


@pytest.fixture(scope="module")
def deployment(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return build_deployment(tmp_path_factory.mktemp("watchdog-agent"))


@pytest.fixture
def in_deployment(
    deployment: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """This process answers config questions for the deployment."""
    repo, build = deployment
    monkeypatch.setenv("OSPREY_CONFIG", str(build / "config.yml"))
    monkeypatch.setenv("CONFIG_FILE", str(build / "config.yml"))
    return repo, build


@pytest.fixture
def judge(
    in_deployment: tuple[Path, Path], scripted_model: Callable[[Any], ScriptedModel]
) -> Callable[..., Any]:
    """Seed the simulator, then judge one fixture of the pair through the real run."""
    repo, build = in_deployment

    async def _judge(label: str, script: list[list[ScriptedCall]]) -> JudgeRun:
        prompt, nodes, machine = read_pair_fixture(label)
        model = scripted_model(script)
        answer_from_the_read(model)
        async with seeded(build, machine):
            setpoint_before = await read_channel(build, SETPOINT)
            verdict = await AgentEngine(build).judge(
                reader_definition(repo),
                nodes,
                prompt,
                deadline=time.monotonic() + 120,
                session_key="e2e-session",
                record_session_id="e2e-record",
                submission_id=f"e2e-{label}",
            )
            setpoint_after = await read_channel(build, SETPOINT)
        return JudgeRun(verdict, model, setpoint_before, setpoint_after)

    return _judge


async def test_a_judge_reads_the_channel_and_every_write_it_tries_is_refused(
    in_deployment: tuple[Path, Path], judge: Callable[..., Any]
) -> None:
    """Criterion 13: a read goes through; ``channel_write``, ``Write`` and ``Edit`` do not."""
    _repo, build = in_deployment
    written_file = build / "judge-wrote-this.txt"
    edited_file = build / "CLAUDE.md"
    edited_before = edited_file.read_bytes()

    run = await judge(
        "risky",
        [
            [read_pressure()],
            [ScriptedCall(WRITE, {"operations": [{"channel": SETPOINT, "value": WRITTEN}]})],
            [ScriptedCall("Write", {"file_path": str(written_file), "content": "written"})],
            [
                ScriptedCall(
                    "Edit",
                    {
                        "file_path": str(edited_file),
                        "old_string": edited_before.decode()[:40],
                        "new_string": "edited",
                    },
                )
            ],
        ],
    )

    verdict = run.verdict
    run.model.assert_script_consumed()
    assert abs(WRITTEN - run.setpoint_before) > 1.0
    assert verdict.status == "ok", verdict
    assert verdict.flagged is True
    assert PRESSURE in (verdict.reason or "")
    refused = {call.tool for call in verdict.refused_tool_calls}
    assert refused == {WRITE, "Write", "Edit"}
    assert not written_file.exists()
    assert edited_file.read_bytes() == edited_before
    assert run.setpoint_after == pytest.approx(run.setpoint_before)


async def test_a_write_that_lands_shows_where_the_judge_is_checked(
    in_deployment: tuple[Path, Path],
) -> None:
    """The channel check above can fail: a landed write reads back until the seed is left."""
    _repo, build = in_deployment
    _prompt, _nodes, machine = read_pair_fixture("risky")

    async with seeded(build, machine) as seed:
        before = await read_channel(build, SETPOINT)
        assert await seed.apply({SETPOINT: WRITTEN}) is None
        assert abs(WRITTEN - before) > 1.0
        assert await read_channel(build, SETPOINT) == pytest.approx(WRITTEN)

    assert await read_channel(build, SETPOINT) == pytest.approx(before)


async def test_the_judge_is_offered_its_read_tool_and_no_write_tool(
    judge: Callable[..., Any],
) -> None:
    """The tools the real CLI offers the model: what a judge can even ask for."""
    run = await judge("benign", [[read_pressure()]])

    verdict, model = run.verdict, run.model
    model.assert_script_consumed()
    assert verdict.status == "ok", verdict
    assert verdict.flagged is False
    assert verdict.refused_tool_calls == []
    offered = set(model.offered_tools)
    assert READ in offered
    assert not offered & {WRITE, "Write", "Edit", "Bash", "NotebookEdit"}


def test_replay_runs_the_agent_pair_for_real(
    in_deployment: tuple[Path, Path], scripted_model: Callable[[Any], ScriptedModel]
) -> None:
    """``osprey watchdog replay``, nothing stubbed: seed, judge, tally, reset.

    The rates are only right when each fixture's seed reached the judge's read,
    since the same script answers from what that read returned.
    """
    repo, _build = in_deployment
    model = scripted_model([[read_pressure()]])
    answer_from_the_read(model)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "osprey.cli.main",
            "watchdog",
            "replay",
            str(PAIR_DIR),
            "--profile",
            str(repo),
            "--project-dir",
            str(repo),
            "--watchdog",
            READER,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert re.search(r"Hit rate on risky\s+100 % \(1 of 1\)", output), output
    assert re.search(r"False flags on benign\s+0 % \(0 of 1\)", output), output
    assert "Not run" not in output, output
    assert "Unavailable" not in output, output
    model.assert_script_consumed()
