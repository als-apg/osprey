"""The agent engine: what a run is launched with, and how its ending becomes a verdict.

The real ``build_agent_options`` is applied to the keywords the engine hands
``stream_query``, so the option assertions hold against what the SDK would get.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from osprey.agent_runner import (
    AgentRunError,
    McpNotReadyError,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolUseEvent,
    build_agent_options,
    read_only_disallowed_tools,
)
from osprey.agent_runner import launcher as agent_launcher
from osprey.agent_runner.tool_names import READ_ONLY_DENIED_BUILTINS
from osprey.agent_runner.tool_policy import PermissionResultAllow, PermissionResultDeny
from osprey.services.watchdog.config import DEFAULT_MAX_COST_USD, WatchdogDefinition
from osprey.services.watchdog.engines import agent as agent_module
from osprey.services.watchdog.engines.agent import AgentEngine, _parse_answer
from osprey.services.watchdog.history import SessionNodes

READ_TOOL = "mcp__controls__channel_read"
ANSWER = json.dumps({"flagged": True, "explanation": "beam is off", "sources": ["SR:CURRENT"]})


@pytest.fixture
def build_dir(tmp_path: Path, monkeypatch) -> Path:
    build = tmp_path / "build"
    build.mkdir()
    (build / "config.yml").write_text("control_system:\n  writes: allowed\n")
    (build / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "controls": {
                        "command": "controls-mcp",
                        "env": {
                            "OSPREY_CONFIG": "config.yml",
                            "OSPREY_CONTROL_TARGET": "va",
                            "OSPREY_CONTROL_TARGET_GENERATION": "3",
                            "OSPREY_SESSION_ID": "live-session",
                        },
                    },
                    "osprey_facility_knowledge": {"command": "knowledge-mcp"},
                    "python": {"command": "python-mcp"},
                    "extra": {"command": "extra-mcp"},
                }
            }
        )
    )
    monkeypatch.setattr(
        agent_launcher, "project_agent_cli_path", lambda _dir: "/opt/bin/claude", raising=True
    )
    routed = {"ANTHROPIC_API_KEY": "k", "OSPREY_CONTROL_TARGET": "va", "OSPREY_SESSION_ID": "s"}
    monkeypatch.setattr("osprey.agent_runner.primitives.sdk_env", lambda *a, **k: dict(routed))
    monkeypatch.setattr(
        "osprey.agent_runner.primitives.resolve_default_model", lambda *a, **k: "provider-default"
    )
    monkeypatch.setattr(
        "osprey.agent_runner.primitives._resolve_project_spec", lambda *a, **k: None
    )
    return build


def _definition(tmp_path: Path, **overrides) -> WatchdogDefinition:
    references = tmp_path / "watchdogs" / "w" / "references"
    references.mkdir(parents=True, exist_ok=True)
    limits = references / "limits.md"
    limits.write_text("limits")
    values = {
        "name": "w",
        "engine": "agent",
        "instructions": "INSTRUCTIONS-TEXT",
        "references": (limits,),
        "tools": (READ_TOOL,),
    }
    return WatchdogDefinition(**{**values, **overrides})


def _result(*, subtype="success", is_error=False, result=ANSWER, cost=0.01) -> ResultEvent:
    return ResultEvent(
        subtype=subtype,
        is_error=is_error,
        num_turns=1,
        duration_ms=1,
        session_id="judge-run",
        total_cost_usd=cost,
        usage=None,
        result=result,
        api_error_status=None,
    )


class _Scripted:
    """A ``stream_query`` stand-in: records its keywords, then plays *events*."""

    def __init__(self, *events, raises=None, hang=False):
        self.events, self.raises, self.hang = events, raises, hang
        self.kwargs = {}
        self.prompt = None

    def __call__(self, project_dir, prompt, **kwargs):
        self.project_dir, self.prompt, self.kwargs = project_dir, prompt, kwargs

        async def stream():
            for event in self.events:
                yield event
            if self.raises is not None:
                raise self.raises
            if self.hang:
                await asyncio.sleep(3600)

        return stream()


def _judge(build, defn, stream, monkeypatch, *, nodes=None, prompt="raise the current", **kw):
    monkeypatch.setattr(agent_module, "stream_query", stream)
    kw.setdefault("deadline", time.monotonic() + 3600)
    return asyncio.run(
        AgentEngine(build).judge(
            defn,
            nodes if nodes is not None else SessionNodes(),
            prompt,
            record_session_id="rec",
            submission_id="sub",
            **kw,
        )
    )


def _ran(build, tmp_path, monkeypatch, **definition) -> _Scripted:
    stream = _Scripted(_result())
    _judge(build, _definition(tmp_path, **definition), stream, monkeypatch)
    return stream


class TestOptions:
    def test_every_write_tool_is_in_the_disallowed_set(self, build_dir, tmp_path, monkeypatch):
        kwargs = _ran(build_dir, tmp_path, monkeypatch).kwargs
        floor = read_only_disallowed_tools(build_dir)
        assert kwargs["disallowed_tools"] == floor
        for tool in (*READ_ONLY_DENIED_BUILTINS, "mcp__controls__channel_write"):
            assert tool in floor

    def test_the_real_options_load_no_project_layer(self, build_dir, tmp_path, monkeypatch):
        kwargs = _ran(build_dir, tmp_path, monkeypatch).kwargs
        kept = {k: v for k, v in kwargs.items() if "mcp_servers" not in k or k == "mcp_servers"}
        options = build_agent_options(build_dir, **kept)
        assert options.setting_sources == []
        assert options.permission_mode == "bypassPermissions"
        assert options.strict_mcp_config is True
        assert options.can_use_tool is not None
        assert options.hooks and options.hooks["PreToolUse"]
        assert set(options.mcp_servers) == {"controls", "osprey_facility_knowledge"}

    def test_allowed_tools_are_the_read_tools_and_the_file_tools(
        self, build_dir, tmp_path, monkeypatch
    ):
        kwargs = _ran(build_dir, tmp_path, monkeypatch).kwargs
        assert kwargs["allowed_tools"] == [READ_TOOL, "Read", "Glob", "Grep"]

    def test_mcp_env_is_readonly_with_no_target_or_session(self, build_dir, tmp_path, monkeypatch):
        servers = _ran(build_dir, tmp_path, monkeypatch).kwargs["mcp_servers"]
        for config in servers.values():
            env = config["env"]
            assert env["OSPREY_EXECUTION_MODE"] == "readonly"
            assert not [key for key in env if key.startswith("OSPREY_CONTROL_TARGET")]
            for key in (
                "OSPREY_SESSION_ID",
                "OSPREY_POSTURE_SESSION",
                "OSPREY_SESSION_HOLD_DIR",
                "OSPREY_PANEL_TOKEN",
            ):
                assert env[key] == ""
        assert servers["controls"]["env"]["OSPREY_CONFIG"] == "config.yml"

    def test_run_env_carries_the_overlay_and_the_skip_marker(
        self, build_dir, tmp_path, monkeypatch
    ):
        env = _ran(build_dir, tmp_path, monkeypatch).kwargs["env"]
        assert env["ANTHROPIC_API_KEY"] == "k"
        assert env["OSPREY_WATCHDOG_SKIP"] == "1"
        assert env["OSPREY_EXECUTION_MODE"] == "readonly"
        assert env["OSPREY_SESSION_ID"] == ""
        assert "OSPREY_CONTROL_TARGET" not in env

    def test_a_named_server_is_required_and_a_base_one_is_not(
        self, build_dir, tmp_path, monkeypatch
    ):
        kwargs = _ran(
            build_dir, tmp_path, monkeypatch, tools=("mcp__extra__look", READ_TOOL)
        ).kwargs
        assert set(kwargs["mcp_servers"]) == {"controls", "extra", "osprey_facility_knowledge"}
        assert set(kwargs["require_mcp_servers"]) == {"controls", "extra"}
        assert kwargs["await_mcp_servers"] == kwargs["require_mcp_servers"]

    def test_a_server_the_build_lacks_is_a_run_error(self, build_dir, tmp_path, monkeypatch):
        stream = _Scripted(_result())
        defn = _definition(tmp_path, tools=("mcp__gone__read",))
        verdict = _judge(build_dir, defn, stream, monkeypatch)
        assert (verdict.status, verdict.error_type) == ("unavailable", "run_error")
        assert stream.prompt is None

    def test_budget_and_timeout_defaults(self, build_dir, monkeypatch):
        defn = WatchdogDefinition(name="w", engine="agent", instructions="i")
        assert defn.timeout_s == 90.0
        assert defn.max_cost_usd == DEFAULT_MAX_COST_USD == 0.25
        waits: list[float | None] = []
        real_wait = asyncio.wait

        async def spy(tasks, timeout=None):
            waits.append(timeout)
            return await real_wait(tasks, timeout=timeout)

        monkeypatch.setattr(agent_module.asyncio, "wait", spy)
        stream = _Scripted(_result())
        _judge(build_dir, defn, stream, monkeypatch)
        assert stream.kwargs["max_budget_usd"] == 0.25
        assert waits == [90.0]

    def test_the_deadline_shortens_the_wall_clock(self, build_dir, tmp_path, monkeypatch):
        waits: list[float | None] = []
        real_wait = asyncio.wait

        async def spy(tasks, timeout=None):
            waits.append(timeout)
            return await real_wait(tasks, timeout=timeout)

        monkeypatch.setattr(agent_module.asyncio, "wait", spy)
        defn = _definition(tmp_path)
        _judge(build_dir, defn, _Scripted(_result()), monkeypatch, deadline=time.monotonic() + 10)
        assert waits[0] <= 10

    def test_the_model_is_the_watchdogs_else_the_providers(self, build_dir, tmp_path, monkeypatch):
        assert _ran(build_dir, tmp_path, monkeypatch).kwargs["model"] == "provider-default"
        chosen = _ran(build_dir, tmp_path, monkeypatch, model="judge-model")
        assert chosen.kwargs["model"] == "judge-model"

    def test_the_prompt_carries_instructions_view_and_references(
        self, build_dir, tmp_path, monkeypatch
    ):
        nodes = SessionNodes(nodes=[{"kind": "prompt", "body": {"text": "HISTORY-LINE"}}])
        stream = _Scripted(_result())
        defn = _definition(tmp_path)
        _judge(build_dir, defn, stream, monkeypatch, nodes=nodes)
        assert "INSTRUCTIONS-TEXT" in stream.prompt
        assert "HISTORY-LINE" in stream.prompt
        assert "USER: raise the current" in stream.prompt
        assert str(defn.references[0]) in stream.prompt


def _gate_of(build, tmp_path, monkeypatch, **definition):
    """The hook, the backstop and the verdict-side state of one finished run."""
    stream = _Scripted(_result())
    defn = _definition(tmp_path, **definition)
    monkeypatch.setattr(agent_module, "stream_query", stream)
    engine = AgentEngine(build)
    verdict = asyncio.run(
        engine.judge(
            defn,
            SessionNodes(),
            "p",
            deadline=time.monotonic() + 60,
            record_session_id="rec",
            submission_id="sub",
        )
    )
    assert verdict.status == "ok"
    return (
        stream.kwargs["pre_tool_use_hooks"][0],
        stream.kwargs["can_use_tool"],
        defn,
    )


def _denied(decision: dict) -> bool:
    return decision.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"


def _hook(hook, tool, tool_input=None, tool_use_id="t1") -> dict:
    return asyncio.run(hook({"tool_name": tool, "tool_input": tool_input or {}}, tool_use_id, None))


class TestDenyCallbacks:
    def test_the_read_tool_is_left_undecided(self, build_dir, tmp_path, monkeypatch):
        hook, _, _ = _gate_of(build_dir, tmp_path, monkeypatch)
        assert _hook(hook, READ_TOOL) == {}

    @pytest.mark.parametrize(
        "tool", ["mcp__controls__channel_write", "Write", "Edit", "Bash", "mcp__other__thing"]
    )
    def test_writes_and_strangers_are_denied(self, build_dir, tmp_path, monkeypatch, tool):
        hook, _, _ = _gate_of(build_dir, tmp_path, monkeypatch)
        assert _denied(_hook(hook, tool))

    def test_a_delegation_is_denied(self, build_dir, tmp_path, monkeypatch):
        hook, _, _ = _gate_of(build_dir, tmp_path, monkeypatch)
        assert _denied(_hook(hook, "Agent", {"subagent_type": "general-purpose"}))

    def test_a_wildcard_entry_covers_its_tools_and_only_those(
        self, build_dir, tmp_path, monkeypatch
    ):
        hook, _, _ = _gate_of(
            build_dir, tmp_path, monkeypatch, tools=("mcp__controls__channel_r*",)
        )
        assert _hook(hook, READ_TOOL) == {}
        assert _denied(_hook(hook, "mcp__controls__channel_write"))
        assert _denied(_hook(hook, "mcp__controls__other"))

    def test_read_is_scoped_to_the_reference_files(self, build_dir, tmp_path, monkeypatch):
        hook, _, defn = _gate_of(build_dir, tmp_path, monkeypatch)
        reference = defn.references[0]
        assert _hook(hook, "Read", {"file_path": str(reference)}) == {}
        for path in (
            str(build_dir / "config.yml"),
            "config.yml",
            f"{reference.parent}/../instructions.md",
            "/etc/passwd",
            "",
        ):
            assert _denied(_hook(hook, "Read", {"file_path": path})), path
        assert _denied(_hook(hook, "Read", {}))

    def test_read_refuses_a_link_out_of_the_references(self, build_dir, tmp_path, monkeypatch):
        hook, _, defn = _gate_of(build_dir, tmp_path, monkeypatch)
        link = defn.references[0].parent / "link.md"
        link.symlink_to(build_dir / "config.yml")
        assert _denied(_hook(hook, "Read", {"file_path": str(link)}))

    def test_glob_and_grep_need_a_path_inside_the_references(
        self, build_dir, tmp_path, monkeypatch
    ):
        hook, _, defn = _gate_of(build_dir, tmp_path, monkeypatch)
        root = str(defn.references[0].parent)
        assert _hook(hook, "Glob", {"pattern": "*.md", "path": root}) == {}
        assert _hook(hook, "Grep", {"pattern": "x", "path": root, "glob": "*.md"}) == {}
        assert _denied(_hook(hook, "Glob", {"pattern": "*.yml"}))
        assert _denied(_hook(hook, "Grep", {"pattern": "secret"}))
        assert _denied(_hook(hook, "Glob", {"pattern": "*", "path": str(build_dir)}))
        assert _denied(_hook(hook, "Glob", {"pattern": "../*", "path": root}))
        assert _denied(_hook(hook, "Glob", {"pattern": "/etc/*", "path": root}))
        assert _denied(_hook(hook, "Grep", {"pattern": "x", "path": root, "glob": "../**"}))

    def test_with_no_references_the_file_tools_read_nothing(self, build_dir, tmp_path, monkeypatch):
        hook, _, _ = _gate_of(build_dir, tmp_path, monkeypatch, references=())
        assert _denied(_hook(hook, "Read", {"file_path": str(build_dir / "config.yml")}))
        assert _denied(_hook(hook, "Glob", {"pattern": "*", "path": str(build_dir)}))

    def test_the_backstop_holds_the_same_line(self, build_dir, tmp_path, monkeypatch):
        _, backstop, defn = _gate_of(build_dir, tmp_path, monkeypatch)
        context = SimpleNamespace(agent_id=None, tool_use_id="t")

        def ask(tool, tool_input):
            return asyncio.run(backstop(tool, tool_input, context))

        assert isinstance(ask(READ_TOOL, {}), PermissionResultAllow)
        assert isinstance(
            ask("Read", {"file_path": str(defn.references[0])}), PermissionResultAllow
        )
        assert isinstance(ask("Read", {"file_path": "/etc/passwd"}), PermissionResultDeny)
        assert isinstance(ask("Write", {}), PermissionResultDeny)
        assert isinstance(ask("mcp__other__thing", {}), PermissionResultDeny)


class TestEndings:
    def test_a_flag_carries_the_explanation_and_its_sources(self, build_dir, tmp_path, monkeypatch):
        verdict = _judge(build_dir, _definition(tmp_path), _Scripted(_result()), monkeypatch)
        assert verdict.status == "ok"
        assert verdict.flagged is True
        assert verdict.reason == "beam is off (Sources: SR:CURRENT)"
        assert verdict.cost_usd == 0.01
        assert verdict.engine == "agent"
        assert verdict.prompt_ref == "rec:sub"

    def test_a_clear_answer_is_ok_and_unflagged(self, build_dir, tmp_path, monkeypatch):
        answer = json.dumps({"flagged": False, "explanation": "nothing here"})
        stream = _Scripted(_result(result=f"I looked.\n```json\n{answer}\n```"))
        verdict = _judge(build_dir, _definition(tmp_path), stream, monkeypatch)
        assert (verdict.status, verdict.flagged) == ("ok", False)

    def test_the_answer_may_come_from_the_text_blocks(self, build_dir, tmp_path, monkeypatch):
        stream = _Scripted(TextEvent(text=ANSWER, parent_tool_use_id=None), _result(result=None))
        verdict = _judge(build_dir, _definition(tmp_path), stream, monkeypatch)
        assert verdict.flagged is True

    def test_the_digest_describes_the_view(self, build_dir, tmp_path, monkeypatch):
        stream = _Scripted(_result())
        nodes = SessionNodes(drain_timed_out=True, record_available=False)
        verdict = _judge(build_dir, _definition(tmp_path), stream, monkeypatch, nodes=nodes)
        digest = verdict.input_digest
        view_text = "USER: raise the current"
        assert digest.sha256 == hashlib.sha256(view_text.encode()).hexdigest()
        assert (digest.entries_kept, digest.entries_cut) == (1, 0)
        assert digest.drain_timed_out is True
        assert digest.record_available is False

    def test_bare_nodes_are_a_whole_read(self, build_dir, tmp_path, monkeypatch):
        stream = _Scripted(_result())
        nodes = [{"kind": "prompt", "body": {"text": "HISTORY-LINE"}}]
        verdict = _judge(build_dir, _definition(tmp_path), stream, monkeypatch, nodes=nodes)
        assert verdict.status == "ok"
        assert verdict.input_digest.record_available is True
        assert "HISTORY-LINE" in stream.prompt

    def test_the_view_holds_its_token_budget(self, build_dir, tmp_path, monkeypatch):
        nodes = SessionNodes(nodes=[{"kind": "prompt", "body": {"text": "x" * 3000}}] * 200)
        monkeypatch.setattr(agent_module, "stream_query", _Scripted(_result()))
        verdict = asyncio.run(
            AgentEngine(build_dir, view_tokens=2000).judge(
                _definition(tmp_path),
                nodes,
                "p",
                deadline=time.monotonic() + 60,
                record_session_id="r",
                submission_id="s",
            )
        )
        assert verdict.input_digest.tokens_estimate <= 2000
        assert verdict.input_digest.entries_cut > 0

    def test_timeout_is_unavailable_and_keeps_what_it_refused(
        self, build_dir, tmp_path, monkeypatch
    ):
        stream = _Scripted(
            ToolUseEvent(tool_use_id="w", name="Write", input={}, parent_tool_use_id=None),
            ToolResultEvent(tool_use_id="w", content="no", is_error=True, parent_tool_use_id=None),
            hang=True,
        )
        started = time.monotonic()
        verdict = _judge(build_dir, _definition(tmp_path, timeout_s=0.2), stream, monkeypatch)
        assert time.monotonic() - started < 3
        assert (verdict.status, verdict.error_type) == ("unavailable", "timeout")
        assert verdict.flagged is False
        assert [c.tool for c in verdict.refused_tool_calls] == ["Write"]

    def test_a_run_that_will_not_unwind_still_answers(self, build_dir, tmp_path, monkeypatch):
        monkeypatch.setattr(agent_module, "_TEARDOWN_GRACE_S", 0.1)

        class Stubborn:
            def __aiter__(self):
                return self

            async def __anext__(self):
                while True:
                    try:
                        await asyncio.sleep(0.5)
                        raise StopAsyncIteration
                    except asyncio.CancelledError:
                        continue

            async def aclose(self):
                return None

        started = time.monotonic()
        verdict = _judge(
            build_dir,
            _definition(tmp_path, timeout_s=0.1),
            lambda *a, **k: Stubborn(),
            monkeypatch,
        )
        assert time.monotonic() - started < 2
        assert verdict.error_type == "timeout"

    def test_a_past_deadline_starts_nothing(self, build_dir, tmp_path, monkeypatch):
        stream = _Scripted(_result())
        verdict = _judge(
            build_dir,
            _definition(tmp_path),
            stream,
            monkeypatch,
            deadline=time.monotonic() - 1,
        )
        assert verdict.error_type == "timeout"
        assert stream.prompt is None

    def test_budget_is_unavailable_with_its_cost(self, build_dir, tmp_path, monkeypatch):
        stream = _Scripted(
            _result(subtype="error_max_budget_usd", is_error=True, result=ANSWER, cost=0.26)
        )
        verdict = _judge(build_dir, _definition(tmp_path), stream, monkeypatch)
        assert (verdict.status, verdict.error_type) == ("unavailable", "budget")
        assert verdict.flagged is False
        assert verdict.cost_usd == 0.26

    def test_budget_wins_over_the_process_failing_after_it(self, build_dir, tmp_path, monkeypatch):
        stream = _Scripted(
            _result(subtype="error_max_budget_usd", is_error=True),
            raises=AgentRunError("exit 1", error_type="unknown"),
        )
        verdict = _judge(build_dir, _definition(tmp_path), stream, monkeypatch)
        assert verdict.error_type == "budget"

    @pytest.mark.parametrize(
        "stream_args",
        [
            {"raises": AgentRunError("boom", error_type="unknown")},
            {"raises": McpNotReadyError("controls", servers=[], missing=["controls"])},
            {"raises": ValueError("anything")},
        ],
    )
    def test_a_failed_run_is_run_error(self, build_dir, tmp_path, monkeypatch, stream_args):
        stream = _Scripted(**stream_args)
        verdict = _judge(build_dir, _definition(tmp_path), stream, monkeypatch)
        assert (verdict.status, verdict.error_type) == ("unavailable", "run_error")

    def test_an_error_result_and_no_result_are_run_errors(self, build_dir, tmp_path, monkeypatch):
        errored = _Scripted(_result(subtype="error_max_turns", is_error=True))
        assert (
            _judge(build_dir, _definition(tmp_path), errored, monkeypatch).error_type == "run_error"
        )
        assert (
            _judge(build_dir, _definition(tmp_path), _Scripted(), monkeypatch).error_type
            == "run_error"
        )

    @pytest.mark.parametrize(
        "text",
        [
            "no document here",
            '{"flagged": "yes", "explanation": "x"}',
            '{"flagged": true}',
            '{"flagged": true, "explanation": ""}',
            "",
        ],
    )
    def test_an_unreadable_answer_is_a_parse_error(self, build_dir, tmp_path, monkeypatch, text):
        stream = _Scripted(_result(result=text))
        verdict = _judge(build_dir, _definition(tmp_path), stream, monkeypatch)
        assert (verdict.status, verdict.error_type) == ("unavailable", "parse_error")

    def test_options_that_cannot_be_built_are_a_run_error(self, build_dir, tmp_path, monkeypatch):
        (build_dir / ".mcp.json").unlink()
        verdict = _judge(build_dir, _definition(tmp_path), _Scripted(_result()), monkeypatch)
        assert verdict.error_type == "run_error"

    def test_judge_cancellation_stops_the_run(self, build_dir, tmp_path, monkeypatch):
        stream = _Scripted(hang=True)
        monkeypatch.setattr(agent_module, "stream_query", stream)

        async def scenario():
            before = asyncio.all_tasks()
            task = asyncio.create_task(
                AgentEngine(build_dir).judge(
                    _definition(tmp_path),
                    SessionNodes(),
                    "p",
                    deadline=time.monotonic() + 60,
                    record_session_id="r",
                    submission_id="s",
                )
            )
            await asyncio.sleep(0.2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.05)
            return asyncio.all_tasks() - before - {asyncio.current_task()}

        assert asyncio.run(scenario()) == set()


class TestRefusedToolCalls:
    def _error_result(self, tool, tool_use_id="u1"):
        return (
            ToolUseEvent(tool_use_id=tool_use_id, name=tool, input={}, parent_tool_use_id=None),
            ToolResultEvent(
                tool_use_id=tool_use_id,
                content="refused",
                is_error=True,
                parent_tool_use_id=None,
            ),
        )

    def test_an_error_result_of_a_disallowed_tool_is_recorded(
        self, build_dir, tmp_path, monkeypatch
    ):
        stream = _Scripted(*self._error_result("mcp__controls__channel_write"), _result())
        verdict = _judge(build_dir, _definition(tmp_path), stream, monkeypatch)
        assert [(c.tool, c.reason) for c in verdict.refused_tool_calls] == [
            ("mcp__controls__channel_write", "disallowed")
        ]

    def test_an_error_result_of_an_unlisted_tool_is_recorded(
        self, build_dir, tmp_path, monkeypatch
    ):
        stream = _Scripted(*self._error_result("mcp__other__thing"), _result())
        verdict = _judge(build_dir, _definition(tmp_path), stream, monkeypatch)
        assert [(c.tool, c.reason) for c in verdict.refused_tool_calls] == [
            ("mcp__other__thing", "not_allowed")
        ]

    def test_a_read_tool_failing_on_its_own_is_not_a_refusal(
        self, build_dir, tmp_path, monkeypatch
    ):
        stream = _Scripted(*self._error_result(READ_TOOL), _result())
        verdict = _judge(build_dir, _definition(tmp_path), stream, monkeypatch)
        assert verdict.refused_tool_calls == []

    def test_a_denied_call_and_its_error_result_count_once(self, build_dir, tmp_path, monkeypatch):
        defn = _definition(tmp_path)
        reference_escape = {"file_path": "/etc/passwd"}

        class Replay(_Scripted):
            def __call__(self, project_dir, prompt, **kwargs):
                hook = kwargs["pre_tool_use_hooks"][0]
                inner = super().__call__(project_dir, prompt, **kwargs)

                async def stream():
                    decision = await hook(
                        {"tool_name": "Read", "tool_input": reference_escape}, "u1", None
                    )
                    assert _denied(decision)
                    async for event in inner:
                        yield event

                return stream()

        stream = Replay(*self._error_result("Read"), _result())
        verdict = _judge(build_dir, defn, stream, monkeypatch)
        assert [(c.tool, c.reason) for c in verdict.refused_tool_calls] == [
            ("Read", "outside_references")
        ]


class TestParseAnswer:
    def test_the_last_document_wins(self):
        first = json.dumps({"flagged": True, "explanation": "draft"})
        last = json.dumps({"flagged": False, "explanation": "final"})
        parsed = _parse_answer(f"{first}\nthen\n{last}")
        assert parsed is not None and parsed.explanation == "final"

    def test_an_inner_object_is_not_taken_for_the_answer(self):
        assert _parse_answer('{"data": {"flagged": true, "explanation": "x"}}') is None


class TestReadonlyServerRefusal:
    async def test_the_stamped_controls_server_refuses_a_direct_write(
        self, build_dir, tmp_path, monkeypatch
    ):
        """A write that slipped every option layer still stops at the server."""
        from tests.mcp_server.conftest import get_tool_fn

        from osprey.mcp_server.control_system import server_context
        from osprey.mcp_server.control_system.server_context import initialize_server_context
        from osprey.mcp_server.control_system.tools.channel_write import channel_write
        from osprey_connectors.refusals import refusal_text
        from osprey_connectors_tests._harness import (
            isolate_verdict_environment,
            write_control_context,
        )

        stream = _Scripted(_result())
        monkeypatch.setattr(agent_module, "stream_query", stream)
        await AgentEngine(build_dir).judge(
            _definition(tmp_path),
            SessionNodes(),
            "p",
            deadline=time.monotonic() + 60,
            record_session_id="r",
            submission_id="s",
        )

        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.chdir(project)
        (project / "config.yml").write_text(
            "control_system:\n  type: virtual_accelerator\n  writes: allowed\n"
            "  connector:\n    virtual_accelerator:\n      serving: in_process\n"
        )
        root = project / "agent_data"
        isolate_verdict_environment(monkeypatch, root)
        write_control_context(root, target="va", owned_by=None, switch={"va": "on"})
        env = stream.kwargs["mcp_servers"]["controls"]["env"]
        for key in ("OSPREY_EXECUTION_MODE", "OSPREY_SESSION_ID", "OSPREY_SESSION_HOLD_DIR"):
            monkeypatch.setenv(key, env[key])
        # A process singleton: restored at teardown so later tests on this worker
        # do not inherit this test's controls server.
        monkeypatch.setattr(server_context, "_registry", None)
        initialize_server_context()

        connector = AsyncMock()
        with patch(
            "osprey_connectors.factory.ConnectorFactory.create_control_system_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            fn = get_tool_fn(channel_write)
            from fastmcp.exceptions import ToolError

            with pytest.raises(ToolError) as refused:
                await fn(operations=[{"channel": "TEST:PV", "value": 1.0}])
        assert refusal_text("readonly_run").split(";")[0] in str(refused.value)
        connector.write_channel.assert_not_called()
        connector.write_multiple_channels.assert_not_called()
