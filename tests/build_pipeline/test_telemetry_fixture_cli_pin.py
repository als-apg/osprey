"""The checked-in captures were made with the Claude Code CLI the deployment pins.

The fixture ``tests/fixtures/telemetry/claude_cli_session.otlp.json`` records
the event names and attribute keys the CLI emits when a launch exports to
agent-record's ``/v1/logs``; agent-record's OTLP receiver is tested against it
(``tests/agent_record/test_otlp_receiver.py``). It is recorded by
``tests/e2e/test_telemetry_contract_live.py``, which needs a model and runs
only under ``full-ci``. This check needs neither, so it runs on every
pull request: a change of the pinned version that is not accompanied by a new
capture fails here, on the pull request that makes it.

The same module records the launches the agent record replays, one directory
each under ``tests/fixtures/agent_record/``: ``capture.json``, ``otlp.json``
and the session's ``transcripts/``. Each is held here to the pinned version,
to what its launch must hold (a subagent, thinking, and on one host launch an
oversized tool result), to the OTLP fixture's scrub, and to a transcript scrub
that leaves no path or account name of the capturing host. The launches that
run in a container need a running stack to capture; until they are captured
they are listed as owed, and nothing stands in for them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from osprey.cli.templates.scaffolding import _DEFAULT_CLAUDE_CLI_VERSION

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "telemetry" / "claude_cli_session.otlp.json"
)
RECAPTURE = "re-capture with OSPREY_CAPTURE_TELEMETRY_FIXTURE=1 under a provider key"
#: The longest attribute value the fixture may keep; mirrors ``MAX_KEPT_VALUE``
#: of the capturing test, which drops every longer value.
MAX_KEPT_VALUE = 64
#: Attribute keys that carry session content; the capture keeps the key only.
CONTENT_KEYS = frozenset(
    {
        "body",
        "prompt",
        "user_prompt",
        "response",
        "tool_input",
        "tool_parameters",
        "output",
        "full_command",
        "bash_command",
    }
)


def test_the_otlp_fixture_was_captured_with_the_pinned_cli_version() -> None:
    assert FIXTURE.is_file(), f"no OTLP fixture at {FIXTURE}; {RECAPTURE}"
    recorded = json.loads(FIXTURE.read_text())
    assert recorded.get("cli_version") == _DEFAULT_CLAUDE_CLI_VERSION, (
        f"the OTLP fixture was captured with Claude Code CLI {recorded.get('cli_version')!r}, "
        f"the pinned version is {_DEFAULT_CLAUDE_CLI_VERSION!r}; {RECAPTURE}"
    )
    assert recorded.get("captured_at"), f"the OTLP fixture records no capture time; {RECAPTURE}"
    assert recorded.get("requests"), f"the OTLP fixture holds no requests; {RECAPTURE}"
    # The receiver test posts the log requests and nothing else.
    assert any(r.get("signal") == "logs" for r in recorded["requests"]), (
        f"the OTLP fixture holds no log requests; {RECAPTURE}"
    )


def _attributes(node: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(node, list):
        for item in node:
            yield from _attributes(item)
    elif isinstance(node, dict):
        if set(node) == {"key", "value"}:
            yield str(node["key"]), node["value"]
        for value in node.values():
            yield from _attributes(value)


def test_the_otlp_fixture_holds_structure_not_session_content() -> None:
    _assert_structure_only(json.loads(FIXTURE.read_text()), "the OTLP fixture")


def _assert_structure_only(recorded: dict[str, Any], what: str) -> None:
    too_long = sorted(
        {
            key
            for key, value in _attributes(recorded.get("requests"))
            if isinstance(value, dict)
            and key != "event.name"
            and len(str(value.get("stringValue", ""))) > MAX_KEPT_VALUE
        }
    )
    assert not too_long, f"{what} keeps long values under {too_long}; {RECAPTURE}"
    kept = sorted(
        {
            key
            for key, value in _attributes(recorded.get("requests"))
            if key in CONTENT_KEYS and value != {"stringValue": "<redacted>"}
        }
    )
    assert not kept, f"{what} keeps session content under {kept}; {RECAPTURE}"


CAPTURES = Path(__file__).resolve().parents[1] / "fixtures" / "agent_record"
#: The launches captured on the host, by ``osprey chat --print`` and ``osprey query``.
HOST_LAUNCHES = frozenset({"host-chat", "host-query"})
#: The launches that run in a container. Capturing them needs a running stack;
#: until a capture exists for one, it is owed, never synthesized.
CONTAINER_LAUNCHES = frozenset({"web-terminal", "dispatch-on-behalf", "dispatch-cron"})
#: Mirrors ``OVERSIZED_RESULT_BYTES`` of the capturing test.
OVERSIZED_RESULT_BYTES = 64 * 1024
#: What a path or an account of the capturing host looks like in a transcript.
#: The capture replaces each with a placeholder (``~``, ``<user>``, ``<tmp>``,
#: ``<tmpdir>``, ``<project>``, ``<project-dir>``); the CLI's directory name
#: for a project spells its path with dashes.
HOST_TRACES = re.compile(r"/Users/[^/<~\s\"]|-Users-|/home/runner/|/var/folders/|pytest-of-")


def _captured() -> list[Path]:
    return sorted(d for d in CAPTURES.iterdir() if d.is_dir()) if CAPTURES.is_dir() else []


def _entries(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _blocks(path: Path) -> Iterator[dict[str, Any]]:
    for entry in _entries(path):
        content = (entry.get("message") or {}).get("content")
        if isinstance(content, list):
            yield from (block for block in content if isinstance(block, dict))


def test_every_host_launch_is_captured_and_no_capture_is_unknown() -> None:
    names = {d.name for d in _captured()}
    missing = sorted(HOST_LAUNCHES - names)
    assert not missing, f"no capture of {missing} under {CAPTURES}; {RECAPTURE}"
    unknown = sorted(names - HOST_LAUNCHES - CONTAINER_LAUNCHES)
    assert not unknown, f"captures of launches no list names: {unknown}"


@pytest.mark.parametrize("capture", _captured(), ids=lambda d: d.name)
def test_a_launch_capture_was_made_with_the_pinned_cli(capture: Path) -> None:
    recorded = json.loads((capture / "capture.json").read_text())
    assert recorded.get("launch") == capture.name, recorded.get("launch")
    otlp = json.loads((capture / "otlp.json").read_text())
    for name, document in (("capture.json", recorded), ("otlp.json", otlp)):
        assert document.get("cli_version") == _DEFAULT_CLAUDE_CLI_VERSION, (
            f"{capture.name}/{name} was captured with Claude Code CLI "
            f"{document.get('cli_version')!r}, the pinned version is "
            f"{_DEFAULT_CLAUDE_CLI_VERSION!r}; {RECAPTURE}"
        )
        assert document.get("captured_at"), f"{capture.name}/{name} records no capture time"
    versions = {
        entry["version"]
        for path in (capture / "transcripts").rglob("*.jsonl")
        for entry in _entries(path)
        if "version" in entry
    }
    assert versions == {_DEFAULT_CLAUDE_CLI_VERSION}, (
        f"the {capture.name} transcripts name CLI versions {sorted(versions)}; {RECAPTURE}"
    )


@pytest.mark.parametrize("capture", _captured(), ids=lambda d: d.name)
def test_a_launch_capture_holds_a_subagent_and_thinking(capture: Path) -> None:
    transcripts = capture / "transcripts"
    mains = sorted(transcripts.glob("*.jsonl"))
    subagents = sorted(transcripts.glob("*/subagents/*.jsonl"))
    assert len(mains) == 1, f"{capture.name} holds {len(mains)} main transcripts"
    assert subagents, f"{capture.name} holds no subagent transcript"
    main_blocks = list(_blocks(mains[0]))
    assert any(b.get("type") == "thinking" for b in main_blocks), (
        f"{capture.name} holds no thinking block"
    )
    assert any(b.get("type") == "tool_use" and b.get("name") == "Agent" for b in main_blocks), (
        f"{capture.name} starts no subagent"
    )
    otlp = json.loads((capture / "otlp.json").read_text())
    assert any(r.get("signal") == "logs" for r in otlp.get("requests", [])), (
        f"{capture.name} holds no OTLP log requests"
    )


def test_one_host_capture_holds_an_oversized_tool_result() -> None:
    sizes = {
        capture.name: max(
            (
                len(json.dumps(block.get("content")))
                for path in (capture / "transcripts").rglob("*.jsonl")
                for block in _blocks(path)
                if block.get("type") == "tool_result"
            ),
            default=0,
        )
        for capture in _captured()
        if capture.name in HOST_LAUNCHES
    }
    assert any(size >= OVERSIZED_RESULT_BYTES for size in sizes.values()), (
        f"no host capture holds a tool result of {OVERSIZED_RESULT_BYTES} bytes: {sizes}"
    )


@pytest.mark.parametrize("capture", _captured(), ids=lambda d: d.name)
def test_a_launch_capture_keeps_nothing_of_the_capturing_host(capture: Path) -> None:
    traced = sorted(
        str(path.relative_to(capture))
        for path in capture.rglob("*")
        if path.is_file() and HOST_TRACES.search(path.read_text())
    )
    assert not traced, f"{capture.name} keeps paths of the capturing host in {traced}"
    otlp = json.loads((capture / "otlp.json").read_text())
    assert otlp.get("authorization") == "Basic <identity>:<password>"
    _assert_structure_only(otlp, f"{capture.name}/otlp.json")
