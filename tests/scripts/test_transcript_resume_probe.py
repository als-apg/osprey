"""Tests for the transcript resume probe.

None of these starts a real CLI or reaches a model: the verdict is fed a request
body recorded from the pinned build (``fixtures/transcript_resume_probe``) and
literal CLI outputs, and the stub and run are driven by stand-in scripts.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from tests._by_path import load_module

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "transcript_resume_probe.py"
probe = load_module("transcript_resume_probe", _MODULE_PATH)

#: The first agent-turn request 2.1.300 sent on resuming the probe's transcript,
#: messages only, with the build's own context blocks trimmed.
_RECORDED = (
    Path(__file__).parent
    / "fixtures"
    / "transcript_resume_probe"
    / ("resumed_request_2.1.300.json")
)
_NOT_FOUND = f"No conversation found with session ID: {probe.SESSION_ID}\n"


@pytest.fixture(scope="module")
def authored(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[str], list[str]]:
    root = tmp_path_factory.mktemp("authored")
    (root / "project").mkdir()
    lines = probe.author(root / "home" / ".claude", root / "project", "2.1.300")
    prompts, ids = probe.authored_expectations(lines)
    return [*prompts, probe.PROMPT], ids


def _recorded() -> dict:
    return json.loads(_RECORDED.read_text())


def _body(messages: list[dict]) -> bytes:
    return json.dumps({"messages": messages, "tools": [{"name": "Bash"}]}).encode()


# ---------------------------------------------------------------------------
# Authoring
# ---------------------------------------------------------------------------


def test_resume_probe_authors_where_the_build_looks(tmp_path: Path) -> None:
    from osprey.agent_runner.project_paths import encode_claude_project_path

    project = tmp_path / "project"
    project.mkdir()
    lines = probe.author(tmp_path / "cfg", project, "2.1.300")
    path = tmp_path / "cfg" / "projects" / encode_claude_project_path(project)
    written = (path / f"{probe.SESSION_ID}.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in written] == lines
    assert {line["cwd"] for line in lines} == {str(project.resolve())}


def test_resume_probe_expects_every_prompt_and_call_of_the_session(authored) -> None:
    prompts, ids = authored
    assert len(prompts) == 4 and prompts[-1] == probe.PROMPT
    assert prompts[1] == "Go ahead, nudge HCM13 to 0.162 A."
    assert len(ids) == 3 and all(i.startswith("toolu_") for i in ids)


def test_resume_probe_authored_transcript_carries_an_approval_attachment(tmp_path) -> None:
    (tmp_path / "p").mkdir()
    lines = probe.author(tmp_path / "cfg", tmp_path / "p", "2.1.300")
    assert [line["type"] for line in lines].count("attachment") == 1


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def test_resume_probe_passes_the_recorded_request(authored) -> None:
    body = json.dumps(_recorded()).encode()
    assert probe.verdict(body, "", *authored) == []


def test_resume_probe_fails_when_no_request_arrives(authored) -> None:
    [failure] = probe.verdict(None, "", *authored)
    assert "no /v1/messages request" in failure


def test_resume_probe_fails_when_the_build_finds_no_conversation(authored) -> None:
    [failure] = probe.verdict(None, _NOT_FOUND, *authored)
    assert "no conversation" in failure and probe.SESSION_ID in failure


def test_resume_probe_no_conversation_wins_over_a_captured_body(authored) -> None:
    body = json.dumps(_recorded()).encode()
    assert probe.verdict(body, _NOT_FOUND, *authored)


def test_resume_probe_fails_on_a_body_that_does_not_parse(authored) -> None:
    [failure] = probe.verdict(b'{"messages": [', "", *authored)
    assert "not JSON" in failure


def test_resume_probe_fails_on_a_body_without_messages(authored) -> None:
    [failure] = probe.verdict(b'{"model": "x"}', "", *authored)
    assert "no messages list" in failure


def _without_prompt(messages: list[dict], prompt: str) -> list[dict]:
    return [m for m in messages if m.get("content") != prompt]


def test_resume_probe_fails_on_a_dropped_prompt(authored) -> None:
    prompts, _ = authored
    messages = _without_prompt(_recorded()["messages"], prompts[1])
    failures = probe.verdict(_body(messages), "", *authored)
    assert any("prompts sent" in f for f in failures)


def test_resume_probe_fails_on_a_repeated_prompt(authored) -> None:
    messages = _recorded()["messages"]
    messages.insert(0, copy.deepcopy(messages[0]))
    failures = probe.verdict(_body(messages), "", *authored)
    assert any("prompts sent" in f for f in failures)


def test_resume_probe_fails_on_prompts_out_of_order(authored) -> None:
    prompts, _ = authored
    messages = _recorded()["messages"]
    first, second = (next(m for m in messages if m.get("content") == p) for p in prompts[:2])
    first["content"], second["content"] = second["content"], first["content"]
    failures = probe.verdict(_body(messages), "", *authored)
    assert any("prompts sent" in f for f in failures)


def test_resume_probe_reads_prompts_from_merged_user_messages(authored) -> None:
    # The recording already merges a context block with the new prompt; dropping
    # the build's system entries and merging the user runs that leaves, as a build
    # may, keeps the verdict.
    merged: list[dict] = []
    for m in _recorded()["messages"]:
        if m["role"] == "system":
            continue
        blocks = (
            m["content"]
            if isinstance(m["content"], list)
            else [{"type": "text", "text": m["content"]}]
        )
        if merged and m["role"] == "user" and merged[-1]["role"] == "user":
            merged[-1]["content"] += blocks
        else:
            merged.append({"role": m["role"], "content": list(blocks)})
    assert len(merged) < len(_recorded()["messages"])
    assert probe.verdict(_body(merged), "", *authored) == []


def _drop_blocks(messages: list[dict], kind: str, index: int) -> list[dict]:
    seen = 0
    for m in messages:
        if not isinstance(m["content"], list):
            continue
        kept = []
        for b in m["content"]:
            if b.get("type") == kind:
                seen += 1
                if seen - 1 == index:
                    continue
            kept.append(b)
        m["content"] = kept
    return [m for m in messages if m["content"]]


def test_resume_probe_fails_on_a_tool_use_without_its_result(authored) -> None:
    _, ids = authored
    messages = _drop_blocks(_recorded()["messages"], "tool_result", 1)
    failures = probe.verdict(_body(messages), "", *authored)
    assert failures == [f"tool_use ids without exactly one tool_result: {[ids[1]]!r}"]


def test_resume_probe_fails_on_a_dropped_tool_use(authored) -> None:
    messages = _drop_blocks(_recorded()["messages"], "tool_use", 2)
    failures = probe.verdict(_body(messages), "", *authored)
    assert any("tool_use ids sent" in f for f in failures)


def test_resume_probe_fails_on_a_renamed_tool_result(authored) -> None:
    messages = _recorded()["messages"]
    result = next(
        b
        for m in messages
        if isinstance(m["content"], list)
        for b in m["content"]
        if b.get("type") == "tool_result"
    )
    result["tool_use_id"] = "toolu_elsewhere"
    failures = probe.verdict(_body(messages), "", *authored)
    assert any("without exactly one tool_result" in f for f in failures)
    assert any("toolu_elsewhere" in f for f in failures)


# ---------------------------------------------------------------------------
# Stub and run, against stand-in builds
# ---------------------------------------------------------------------------


def _stand_in(tmp_path: Path, body: str) -> Path:
    """A stand-in build: a Python script with ``body`` after a small preamble."""
    script = tmp_path / "claude"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, urllib.request, urllib.error\n"
        "BASE = os.environ['ANTHROPIC_BASE_URL']\n"
        "def post(path, payload):\n"
        "    req = urllib.request.Request(BASE + path, data=payload, method='POST',\n"
        "        headers={'content-type': 'application/json'})\n"
        "    try:\n"
        "        urllib.request.urlopen(req, timeout=10)\n"
        "    except urllib.error.HTTPError as exc:\n"
        "        print('status', exc.code, flush=True)\n" + body
    )
    script.chmod(0o755)
    return script


def test_resume_probe_stub_records_the_first_agent_turn_request(tmp_path: Path) -> None:
    cli = _stand_in(
        tmp_path,
        "post('/v1/messages/count_tokens', b'{\"tools\": [1]}')\n"
        "post('/v1/messages', b'{\"messages\": [], \"max_tokens\": 1}')\n"  # a side request
        "post('/v1/messages?beta=true', b'{\"messages\": [1], \"tools\": [1]}')\n"
        "post('/v1/messages', b'{\"messages\": [2], \"tools\": [1]}')\n"
        "import time; time.sleep(30)\n",
    )
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    raw, output = probe.run(cli, tmp_path / "home", tmp_path / "project")
    assert json.loads(raw) == {"messages": [1], "tools": [1]}
    assert "status 500" in output


def test_resume_probe_run_returns_the_refusal_when_nothing_is_sent(tmp_path: Path) -> None:
    cli = _stand_in(tmp_path, f"print({_NOT_FOUND.strip()!r}, file=sys.stderr)\nsys.exit(1)\n")
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    raw, output = probe.run(cli, tmp_path / "home", tmp_path / "project")
    assert raw is None
    assert "No conversation found" in output


def test_resume_probe_run_bounds_a_silent_build(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(probe, "RUN_TIMEOUT_S", 1)
    cli = _stand_in(tmp_path, "import time; time.sleep(60)\n")
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    assert probe.run(cli, tmp_path / "home", tmp_path / "project") == (None, "")


def test_resume_probe_passes_a_build_that_sends_the_conversation(tmp_path, capsys) -> None:
    recorded = json.dumps({**_recorded(), "tools": [{"name": "Bash"}]})
    cli = _stand_in(tmp_path, f"post('/v1/messages', {recorded.encode()!r})\n")
    assert probe.main(["--cli", str(cli)]) == 0
    assert capsys.readouterr().out.startswith("ok: ")


def test_resume_probe_fails_a_build_that_finds_no_conversation(tmp_path, capsys) -> None:
    cli = _stand_in(tmp_path, f"print({_NOT_FOUND.strip()!r})\nsys.exit(1)\n")
    assert probe.main(["--cli", str(cli)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("FAIL ") and "remedy:" in out
