"""Unit tests for tests/e2e/sdk_helpers.py pure-logic helpers.

No LLM, no build, no DB — fast logic checks. Excluded from the model-capability
matrix (scripts/benchmark/matrix_e2e_config.json) since it measures nothing
about the model under test.
"""

from __future__ import annotations

import inspect
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from osprey.agent_runner.primitives import SDKWorkflowResult, ToolTrace
from osprey.agent_runner.project_paths import claude_project_dir
from osprey.utils.owner_header import OWNER_HEADER
from osprey_connectors import control_context
from osprey_connectors.identity import acting_identity
from osprey_connectors.types import baseline_target
from tests.e2e import _queue_drive, sdk_helpers
from tests.e2e.sdk_helpers import (
    HOOK_ATTACHMENT_PREFIX,
    JUDGE_RESULT_CHARS,
    ONE_SHOT_DISALLOWED_TOOLS,
    TRANSCRIPT_SUBDIR,
    HookEvent,
    _bind_approval_policy,
    _log_query_timing,
    assert_agent_ran,
    dump_agent_transcript,
    e2e_budget_cap,
    e2e_budget_scale,
    hook_attachments,
)

pytestmark = pytest.mark.model_free

# ---------------------------------------------------------------------------
# The agent-transcript dump. Its whole reason to exist is that the judge sees
# tool results previewed at JUDGE_RESULT_CHARS, so the property worth pinning is
# that the artifact is NOT truncated — a dump that inherited the same limit
# would look like coverage while answering nothing.
# ---------------------------------------------------------------------------


# Twice the judge's preview, so a dump that inherited the preview's cut fails.
_LONG_RESULT_CHARS = 2 * JUDGE_RESULT_CHARS


def _result_with_a_long_tool_result(size: int = _LONG_RESULT_CHARS):
    """One trace whose result is far longer than the judge's preview."""
    return SDKWorkflowResult(
        text_blocks=["I measured the ring.", "The response is linear."],
        tool_traces=[
            ToolTrace(
                name="mcp__bluesky__get_run_data",
                input={"run_uid": "abc123"},
                result="x" * size,
                tool_use_id="tu_1",
            )
        ],
    )


def test_transcript_dump_is_inert_when_unarmed(tmp_path, monkeypatch):
    """No OSPREY_CI_DIAG_DIR — every local run — writes nothing at all."""
    monkeypatch.delenv("OSPREY_CI_DIAG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    # ``render`` is passed to prove the transcript read is downstream of the
    # arming check: an unarmed run must not go near the session directory.
    assert (
        dump_agent_transcript("orbit_response", _result_with_a_long_tool_result(), render=tmp_path)
        is None
    )
    assert list(tmp_path.iterdir()) == []


def test_transcript_dump_keeps_tool_results_whole(tmp_path, monkeypatch):
    """The payload must carry the full tool result, not a preview.

    This is the defect the dump exists to close: `_to_workflow_result` cuts
    every result to `JUDGE_RESULT_CHARS` before the judge sees it, so a long
    `get_run_data` body is never seen whole.
    """
    monkeypatch.setenv("OSPREY_CI_DIAG_DIR", str(tmp_path))

    target = dump_agent_transcript("orbit_response", _result_with_a_long_tool_result())

    assert target is not None
    assert target == tmp_path / TRANSCRIPT_SUBDIR / "orbit_response.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert len(payload["tool_traces"][0]["result"]) == _LONG_RESULT_CHARS
    assert payload["response"] == "I measured the ring.\nThe response is linear."
    # No render, no hook stdout: the six plan-stack dump sites that call this
    # without one must keep exactly the payload they had.
    assert "hook_attachments" not in payload


def test_transcript_dump_sanitises_the_name(tmp_path, monkeypatch):
    """Parametrised ids carry `[]`, and paths carry `/`; neither may escape
    the diagnostics directory or produce an unopenable filename."""
    monkeypatch.setenv("OSPREY_CI_DIAG_DIR", str(tmp_path))

    target = dump_agent_transcript("../grid[two_axis]", SDKWorkflowResult(), render=tmp_path)

    assert target is not None
    assert target.parent == tmp_path / TRANSCRIPT_SUBDIR
    assert target.name == ".._grid_two_axis_.json"
    # A result with no session id resolves to no transcript, and the key is
    # still written — an empty list says "looked, found none", which a missing
    # key cannot.
    assert json.loads(target.read_text(encoding="utf-8"))["hook_attachments"] == []


def test_transcript_dump_never_raises(tmp_path, monkeypatch):
    """A diagnostic that fails the test it observes would mask the failure it
    exists to explain. Here the target directory is occupied by a file, so the
    mkdir cannot succeed."""
    blocker = tmp_path / "diag"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("OSPREY_CI_DIAG_DIR", str(blocker))

    assert dump_agent_transcript("orbit_response", SDKWorkflowResult(), render=blocker) is None


# ---------------------------------------------------------------------------
# Hook stdout read back out of the session transcript. Synthetic JSONL: the
# layout is Claude Code's, not ours, so these pin what the reader tolerates
# rather than what the CLI happens to write.
# ---------------------------------------------------------------------------


def _session_result(session_id: str) -> SDKWorkflowResult:
    """A workflow result carrying nothing but the session id the reader needs."""
    return SDKWorkflowResult(result=SimpleNamespace(session_id=session_id))  # type: ignore[arg-type]


def _write_transcript(tmp_path, monkeypatch, session_id: str, records: list) -> Path:
    """Plant a session transcript where `claude_project_dir` will look for it.

    Returns the render directory to hand the reader. `CLAUDE_CONFIG_DIR` is
    redirected so the test never reads — or creates — anything under the
    developer's real ~/.claude.
    """
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    render = (tmp_path / "render").resolve()
    render.mkdir(parents=True, exist_ok=True)
    transcript = claude_project_dir(render) / f"{session_id}.jsonl"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    return render


def test_hook_attachments_reads_every_hook_prefixed_attachment(tmp_path, monkeypatch):
    """Prefix match, not an allow-list of one.

    `hook_success` is the only spelling verified against CLI 2.1.241, so a hook
    that failed or timed out is recorded under a type this test cannot name.
    Matching the prefix is what keeps that hook's stdout in the CI artifact —
    and a hook that died mid-approval is exactly the run someone needs to read.
    """
    render = _write_transcript(
        tmp_path,
        monkeypatch,
        "sess-1",
        [
            {"type": "user", "message": {"role": "user", "content": "hi"}},
            {"attachment": {"type": "selected_lines_in_ide", "stdout": "not a hook"}},
            {
                "attachment": {
                    "type": "hook_success",
                    "hookName": "PreToolUse:mcp__controls__channel_write",
                    "hookEvent": "PreToolUse",
                    "toolUseID": "tu_ok",
                    "stdout": '{"hookSpecificOutput": {"permissionDecision": "ask"}}',
                }
            },
            {
                "attachment": {
                    "type": "hook_error",
                    "hookName": "PreToolUse:mcp__controls__channel_write",
                    "hookEvent": "PreToolUse",
                    "toolUseID": "tu_broken",
                    "stdout": "Traceback (most recent call last):",
                }
            },
        ],
    )

    found = hook_attachments(_session_result("sess-1"), render)

    assert [a["toolUseID"] for a in found] == ["tu_ok", "tu_broken"]
    assert all(a["type"].startswith(HOOK_ATTACHMENT_PREFIX) for a in found)
    # The dicts come back raw — nothing is projected away, because a reader
    # chasing an unexplained refusal does not know in advance which key holds
    # the answer.
    assert found[1]["stdout"] == "Traceback (most recent call last):"
    assert found[1]["hookName"] == "PreToolUse:mcp__controls__channel_write"


def test_hook_attachments_survives_a_malformed_transcript(tmp_path, monkeypatch):
    """A truncated write or a bare JSON scalar must not raise out of a
    diagnostic path — the run being diagnosed is the one that already failed."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    render = (tmp_path / "render").resolve()
    render.mkdir(parents=True, exist_ok=True)
    transcript = claude_project_dir(render) / "sess-2.jsonl"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(
        "\n".join(
            [
                "{not json at all",
                '"a bare string"',
                "[1, 2, 3]",
                json.dumps({"attachment": "not a dict"}),
                json.dumps({"attachment": {"type": "hook_success", "toolUseID": "tu_1"}}),
            ]
        ),
        encoding="utf-8",
    )

    assert [a["toolUseID"] for a in hook_attachments(_session_result("sess-2"), render)] == ["tu_1"]


def test_hook_attachments_empty_without_a_session_or_transcript(tmp_path, monkeypatch):
    """No session id and no transcript both mean "nothing to read", not an error.

    Callers assert on what they expected to find; a bare empty list is a
    clearer failure than an exception raised from an observer.
    """
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    render = (tmp_path / "render").resolve()
    render.mkdir(parents=True, exist_ok=True)

    assert hook_attachments(SDKWorkflowResult(), render) == []
    assert hook_attachments(_session_result("never-written"), render) == []


def test_transcript_dump_carries_the_hook_stdout_when_given_a_render(tmp_path, monkeypatch):
    """The artifact reason this exists: tool traces alone cannot say why a call
    was asked about, because the approval wording only ever reaches the
    transcript."""
    render = _write_transcript(
        tmp_path,
        monkeypatch,
        "sess-3",
        [
            {
                "attachment": {
                    "type": "hook_success",
                    "hookName": "PreToolUse:mcp__controls__channel_write",
                    "hookEvent": "PreToolUse",
                    "toolUseID": "tu_1",
                    "stdout": '{"hookSpecificOutput": {"permissionDecisionReason": "live machine"}}',
                }
            }
        ],
    )
    monkeypatch.setenv("OSPREY_CI_DIAG_DIR", str(tmp_path / "diag"))

    target = dump_agent_transcript("switch_run", _session_result("sess-3"), render=render)

    assert target is not None
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert [a["toolUseID"] for a in payload["hook_attachments"]] == ["tu_1"]
    assert "live machine" in payload["hook_attachments"][0]["stdout"]


# ---------------------------------------------------------------------------
# Approval-policy arity. A policy written before the permission context existed
# takes two arguments and must keep working untouched; a policy that wants to
# see why the hook asked takes a third. The binder decides which from the
# signature, so a `TypeError` raised inside a three-argument policy surfaces as
# itself instead of being mistaken for the older shape.
# ---------------------------------------------------------------------------


class _FakeContext:
    """Stand-in for the SDK's permission context — no SDK install required."""

    def __init__(self, decision_reason: str | None = None):
        self.decision_reason = decision_reason


def test_two_arg_policy_is_called_with_two_arguments():
    """The pre-existing form is dispatched unchanged — the context is dropped."""
    seen: list[tuple] = []

    def policy(tool_name, tool_input):
        seen.append((tool_name, tool_input))
        return True

    bound = _bind_approval_policy(policy)

    assert bound("Bash", {"command": "ls"}, _FakeContext("hook said ask")) is True
    assert seen == [("Bash", {"command": "ls"})]


def test_three_arg_policy_receives_the_context():
    """The widened form gets the context object itself, not a copy of a field."""
    seen: list[tuple] = []
    context = _FakeContext("write outside the render")

    def policy(tool_name, tool_input, ctx):
        seen.append((tool_name, tool_input, ctx))
        return False

    bound = _bind_approval_policy(policy)

    assert bound("Write", {"file_path": "/etc/hosts"}, context) is False
    assert seen == [("Write", {"file_path": "/etc/hosts"}, context)]
    assert seen[0][2].decision_reason == "write outside the render"


def test_var_positional_policy_receives_the_context():
    """A ``*args`` policy can take the context, so it is given the context."""
    seen: list[tuple] = []

    def policy(*args):
        seen.append(args)
        return True

    _bind_approval_policy(policy)("Read", {}, _FakeContext("why"))

    assert len(seen[0]) == 3


def test_binder_does_not_swallow_type_errors_from_the_policy():
    """A TypeError raised *inside* a policy is a bug, not an arity signal."""

    def policy(_tool_name, _tool_input, _ctx):
        raise TypeError("boom inside the policy")

    with pytest.raises(TypeError, match="boom inside the policy"):
        _bind_approval_policy(policy)("Bash", {}, _FakeContext())


def test_hook_event_decision_reason_defaults_and_accepts():
    """The new field is optional (existing constructions keep working) and
    carries the hook's own wording when the callback records it."""
    without = HookEvent(tool_name="Bash", tool_input={}, decision="allow")
    assert without.decision_reason is None
    assert without.reason is None

    with_reason = HookEvent(
        tool_name="Write",
        tool_input={"file_path": "notes.md"},
        decision="deny",
        reason="custom_policy",
        decision_reason="target is outside the render",
    )
    assert with_reason.reason == "custom_policy"
    assert with_reason.decision_reason == "target is outside the render"


def test_run_claude_holds_the_first_turn_for_every_mcp_server(tmp_path, monkeypatch):
    """``run_claude`` spawns the CLI with the readiness env, so a subagent
    spawned on the first turn gets every declared server's tools."""
    import subprocess

    from osprey.agent_runner.primitives import MCP_READY_TIMEOUT_S
    from tests.e2e import test_claude_code_build_integration as integration

    (tmp_path / "build").mkdir()
    for key in ("MCP_CONNECTION_NONBLOCKING", "MCP_CONNECT_TIMEOUT_MS", "MCP_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(integration, "provider_env_for_project", lambda render: {})
    monkeypatch.setattr(integration, "_resolve_claude_binary", lambda: "claude")
    seen: dict[str, str] = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs["env"])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(integration.subprocess, "run", fake_run)
    integration.run_claude(tmp_path, "hello")

    ready_ms = str(int(MCP_READY_TIMEOUT_S * 1000))
    assert seen["MCP_CONNECTION_NONBLOCKING"] == "0"
    assert seen["MCP_CONNECT_TIMEOUT_MS"] == ready_ms
    assert seen["MCP_TIMEOUT"] == ready_ms


# ---------------------------------------------------------------------------
# Per-query timing log for the benchmark matrix. The dashboard reads it to show
# how long each model takes and how much of that is spent waiting on the model.
# ---------------------------------------------------------------------------


def _timed_result(**overrides):
    fields = {
        "duration_ms": 120_000,
        "duration_api_ms": 96_000,
        "num_turns": 7,
        "is_error": False,
        "usage": {"input_tokens": 41_000, "output_tokens": 3_200},
    }
    fields.update(overrides)
    return SDKWorkflowResult(result=SimpleNamespace(**fields))


def test_query_timing_is_inert_when_unarmed(tmp_path, monkeypatch):
    """No OSPREY_E2E_QUERY_LOG — every non-benchmark run — writes nothing."""
    monkeypatch.delenv("OSPREY_E2E_QUERY_LOG", raising=False)
    monkeypatch.chdir(tmp_path)
    _log_query_timing(_timed_result())
    assert list(tmp_path.iterdir()) == []


def test_query_timing_appends_one_record_per_query(tmp_path, monkeypatch):
    log = tmp_path / "m__seed1.queries.jsonl"
    monkeypatch.setenv("OSPREY_E2E_QUERY_LOG", str(log))
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/e2e/test_x.py::TestX::test_y[case0] (call)")
    _log_query_timing(_timed_result())
    _log_query_timing(_timed_result(duration_ms=60_000, usage=None))

    first, second = (json.loads(line) for line in log.read_text().splitlines())
    assert first == {
        "test": "tests/e2e/test_x.py::TestX::test_y[case0]",
        "duration_ms": 120_000,
        "duration_api_ms": 96_000,
        "num_turns": 7,
        "input_tokens": 41_000,
        "output_tokens": 3_200,
        "is_error": False,
    }
    assert second["duration_ms"] == 60_000
    assert second["output_tokens"] is None


def test_query_timing_skips_a_query_without_a_result(tmp_path, monkeypatch):
    """A query that never produced a ResultMessage has no timing to record."""
    log = tmp_path / "q.jsonl"
    monkeypatch.setenv("OSPREY_E2E_QUERY_LOG", str(log))
    _log_query_timing(SDKWorkflowResult())
    assert not log.exists()


def test_query_timing_never_raises(tmp_path, monkeypatch):
    """Instrumentation must never fail the test it observes."""
    monkeypatch.setenv("OSPREY_E2E_QUERY_LOG", str(tmp_path / "missing-dir" / "q.jsonl"))
    _log_query_timing(_timed_result())


# --- the per-query dollar cap handed to the Agent SDK -----------------------


@pytest.mark.parametrize(
    ("raw", "base", "cap"),
    [(None, 2.0, 2.0), ("3", 2.0, 6.0), ("0", 2.0, 2.0), ("junk", 2.0, 2.0)],
)
def test_budget_cap_scales_the_call_site_budget(monkeypatch, raw, base, cap):
    if raw is None:
        monkeypatch.delenv("OSPREY_E2E_BUDGET_SCALE", raising=False)
    else:
        monkeypatch.setenv("OSPREY_E2E_BUDGET_SCALE", raw)
    assert e2e_budget_cap(base) == cap


def test_an_infinite_budget_scale_sends_no_cap_and_passes_every_cost_ceiling(monkeypatch):
    monkeypatch.setenv("OSPREY_E2E_BUDGET_SCALE", "inf")
    assert e2e_budget_cap(2.0) is None
    assert math.isinf(e2e_budget_scale())
    assert 1e6 < 0.5 * e2e_budget_scale()


# ---------------------------------------------------------------------------
# assert_agent_ran: a deny test may only pass on a session that actually ran.
# ---------------------------------------------------------------------------


def _ended(*, is_error: bool, tool_calls: int, text: str = "") -> SDKWorkflowResult:
    return SDKWorkflowResult(
        tool_traces=[
            ToolTrace(name="mcp__controls__channel_read", input={}) for _ in range(tool_calls)
        ],
        result=SimpleNamespace(is_error=is_error, result=text),
    )


def test_a_session_that_errored_before_any_tool_call_never_ran():
    with pytest.raises(AssertionError, match="never ran.*Not logged in"):
        assert_agent_ran(_ended(is_error=True, tool_calls=0, text="Not logged in"))


def test_a_session_without_a_result_message_fails():
    with pytest.raises(AssertionError, match="No ResultMessage"):
        assert_agent_ran(SDKWorkflowResult())


@pytest.mark.parametrize(
    ("is_error", "tool_calls"),
    [(False, 0), (True, 3), (False, 2)],
    ids=["text-refusal", "turn-cap", "ordinary-run"],
)
def test_a_session_that_ran_passes(is_error, tool_calls):
    assert_agent_ran(_ended(is_error=is_error, tool_calls=tool_calls))


# ---------------------------------------------------------------------------
# A one-shot SDK session is never re-invoked, so a tool that ends the turn
# expecting a later wake-up strands the run with its work half done.
# ---------------------------------------------------------------------------


def test_the_one_shot_floor_is_the_scheduling_subset_of_the_headless_denies():
    """Every name is one the pinned CLI lists (the conformance test holds that
    for READ_ONLY_DENIED_BUILTINS), and the floor names scheduling only."""
    from osprey.agent_runner.tool_names import READ_ONLY_DENIED_BUILTINS

    assert "ScheduleWakeup" in ONE_SHOT_DISALLOWED_TOOLS
    assert set(ONE_SHOT_DISALLOWED_TOOLS) <= set(READ_ONLY_DENIED_BUILTINS)
    assert "Bash" not in ONE_SHOT_DISALLOWED_TOOLS


def _runners_stopped_at_their_options(tmp_path, monkeypatch) -> dict:
    """Stub what a runner resolves before it builds its options, and stop it there.

    Returns the dict the options' keyword arguments land in once a run reaches them;
    the run then raises ``RuntimeError("stop here ...")``.
    """
    from tests.e2e import sdk_helpers

    (tmp_path / "build").mkdir()
    seen: dict = {}

    class _Options:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop here: the options are what the test reads")

    monkeypatch.setattr(sdk_helpers, "ClaudeAgentOptions", _Options)
    monkeypatch.setattr(sdk_helpers, "resolve_default_model", lambda render: "m")
    monkeypatch.setattr(sdk_helpers, "project_agent_cli_path", lambda render: "claude")
    monkeypatch.setattr(sdk_helpers, "sdk_env", lambda render: {})
    return seen


@pytest.mark.parametrize("runner", ["run_sdk_query", "run_sdk_query_with_hooks"])
async def test_every_sdk_run_disallows_the_one_shot_floor(tmp_path, monkeypatch, runner):
    """Both runners add the floor under whatever the caller disallowed."""
    from tests.e2e import sdk_helpers

    seen = _runners_stopped_at_their_options(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="stop here"):
        await getattr(sdk_helpers, runner)(tmp_path, "hello", disallowed_tools=["Bash"])

    assert "Bash" in seen["disallowed_tools"]
    for name in ONE_SHOT_DISALLOWED_TOOLS:
        assert name in seen["disallowed_tools"], name


# ---------------------------------------------------------------------------
# Claude Code state: the trust seed lands only in a directory the harness owns
# (``harness_config_dir``, which the e2e conftest sets for a model-free test),
# never in the developer's own. That the conftest sets it is proved where it
# is relied on, by the scripted proofs that seed trust.
# ---------------------------------------------------------------------------


def test_trust_render_refuses_the_developers_own_state(tmp_path, monkeypatch):
    """With no harness-owned directory, the seed is refused rather than written
    into whatever ``~/.claude`` the process would otherwise resolve."""
    from tests.e2e import sdk_helpers

    written: list = []
    monkeypatch.setattr(sdk_helpers, "seed_claude_state", lambda render: written.append(render))
    monkeypatch.setattr(sdk_helpers, "harness_config_dir", None)

    with pytest.raises(RuntimeError, match="model_free"):
        sdk_helpers.trust_render(tmp_path)
    assert written == []


@pytest.mark.parametrize("harness_owned", [True, False], ids=["harness-state", "developer-state"])
async def test_a_hooked_run_seeds_trust_only_under_the_harness_own_state(
    tmp_path, monkeypatch, harness_owned
):
    """``run_sdk_query_with_hooks`` trusts the render when the state is the
    harness's, and leaves the developer's state alone otherwise."""
    from tests.e2e import sdk_helpers

    _runners_stopped_at_their_options(tmp_path, monkeypatch)
    seeded: list = []
    monkeypatch.setattr(sdk_helpers, "seed_claude_state", lambda render: seeded.append(render))
    monkeypatch.setattr(
        sdk_helpers, "harness_config_dir", tmp_path / "claude-config" if harness_owned else None
    )

    with pytest.raises(RuntimeError, match="stop here"):
        await sdk_helpers.run_sdk_query_with_hooks(tmp_path, "hello")
    assert seeded == ([tmp_path / "build"] if harness_owned else [])


# ---------------------------------------------------------------------------
# A rerun must not erase the attempt it is retrying.
# ---------------------------------------------------------------------------


def test_transcript_dump_keeps_every_attempt(tmp_path, monkeypatch):
    """Two dumps under one name leave two files, the first untouched: the
    attempt a rerun retries is the one worth reading."""
    monkeypatch.setenv("OSPREY_CI_DIAG_DIR", str(tmp_path))

    first = dump_agent_transcript("orbit_response", _result_with_a_long_tool_result(10))
    second = dump_agent_transcript("orbit_response", _result_with_a_long_tool_result(20))

    assert first is not None and second is not None
    assert first != second
    assert first.parent == second.parent
    assert len(json.loads(first.read_text())["tool_traces"][0]["result"]) == 10
    assert len(json.loads(second.read_text())["tool_traces"][0]["result"]) == 20


# ---------------------------------------------------------------------------
# The arming helper (D9). A write is allowed when the login's tier says
# ``allowed`` AND that login's switch is on for the machine, so the helper is
# pinned on both halves and on whose switch it moves: arming the wrong login's
# record is a lane that silently runs with writes off.
# ---------------------------------------------------------------------------


def _fake_repo(tmp_path: Path, profile_config: dict | None = None) -> Path:
    """A repo with a source profile and a render, without running a build."""
    repo = tmp_path / "deployment"
    (repo / "build").mkdir(parents=True)
    (repo / "profile.yml").write_text(
        yaml.safe_dump({"name": "deployment", "config": profile_config or {}}, sort_keys=False)
    )
    (repo / "build" / "config.yml").write_text(
        yaml.safe_dump(
            {
                "project_root": str(repo),
                "agent_data": {"base_dir": "var/agent_data"},
                "control_system": {"type": "virtual_accelerator"},
            }
        )
    )
    return repo


def _builds(monkeypatch) -> list[list[str]]:
    """Stand in for ``osprey <verb>``; record each call instead of building."""
    calls: list[list[str]] = []
    monkeypatch.setattr(
        sdk_helpers, "_run_osprey", lambda verb, args, *, timeout: calls.append([verb, *args])
    )
    return calls


def _baseline() -> str:
    return baseline_target({"type": "virtual_accelerator"})


def test_set_writes_tier_spells_the_tier_key_once_and_rebuilds(tmp_path, monkeypatch):
    builds = _builds(monkeypatch)
    repo = _fake_repo(
        tmp_path,
        {
            "archiver.type": "mock_archiver",
            "control_system.writes": "never",
            "control_system": {"writes": "never", "type": "virtual_accelerator"},
        },
    )

    sdk_helpers.set_writes_tier(repo)

    config = yaml.safe_load((repo / "profile.yml").read_text())["config"]
    assert config["control_system.writes"] == "allowed"
    assert config["control_system"] == {"type": "virtual_accelerator"}
    assert config["archiver.type"] == "mock_archiver"
    assert builds == [["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"]]


def test_set_writes_tier_writes_a_config_block_a_profile_lacks(tmp_path, monkeypatch):
    _builds(monkeypatch)
    repo = _fake_repo(tmp_path)
    (repo / "profile.yml").write_text("name: deployment\n")

    sdk_helpers.set_writes_tier(repo)

    profile = yaml.safe_load((repo / "profile.yml").read_text())
    assert profile == {"name": "deployment", "config": {"control_system.writes": "allowed"}}


def test_the_record_arm_turns_on_the_named_owners_switch_only(tmp_path):
    repo = _fake_repo(tmp_path)

    stored = sdk_helpers.switch_on_in_record(repo, owner="alice")

    path = repo / "var" / "agent_data" / "control_target" / "alice" / "control_context.json"
    assert sdk_helpers.switch_record_path(repo, "alice") == path
    control_context.invalidate_cache()
    on_disk = control_context.read_record(path=path)
    assert on_disk is not None
    assert dict(on_disk.switch) == {_baseline(): "asked"} == dict(stored.switch)
    assert on_disk.target == _baseline()
    assert isinstance(on_disk.switch_epoch, int) and on_disk.switch_epoch > 0
    assert not (path.parent.parent / "bob").exists()


def test_the_record_arm_keeps_an_existing_record_and_is_idempotent(tmp_path):
    repo = _fake_repo(tmp_path)
    path = sdk_helpers.switch_record_path(repo, "alice")
    control_context.write_record(
        control_context.ControlContext(target=_baseline(), generation=7), path=path
    )

    sdk_helpers.switch_on_in_record(repo, owner="alice")
    again = sdk_helpers.switch_on_in_record(repo, owner="alice")

    assert again.generation == 7
    assert dict(again.switch) == {_baseline(): "asked"}


def test_the_record_arm_retries_an_epochless_record_once(tmp_path):
    repo = _fake_repo(tmp_path)
    path = sdk_helpers.switch_record_path(repo, "alice")
    path.parent.mkdir(parents=True)
    payload = control_context.ControlContext(target=_baseline(), generation=1).to_payload()
    payload["switch_epoch"] = None
    path.write_text(json.dumps(payload))
    control_context.invalidate_cache()

    stored = sdk_helpers.switch_on_in_record(repo, owner="alice")

    assert dict(stored.switch) == {_baseline(): "asked"}
    assert isinstance(stored.switch_epoch, int)


def test_the_record_arm_refuses_an_owner_that_is_not_a_login(tmp_path):
    repo = _fake_repo(tmp_path)
    with pytest.raises(ValueError):
        sdk_helpers.switch_on_in_record(repo, owner="../escape")
    assert not (repo / "var" / "agent_data" / "escape").exists()


class _FakeTerminal:
    """A local HTTP server answering the posture and queue routes from a script."""

    def __init__(self, answers):
        self.answers = answers
        self.seen: list[tuple[str, str, dict, dict | None]] = []
        terminal = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self, method):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                body = json.loads(raw) if raw else None
                terminal.seen.append((method, self.path, dict(self.headers), body))
                status, reply = terminal.answers(method, self.path, body)
                data = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._answer("GET")

            def do_POST(self):
                self._answer("POST")

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def test_the_chip_arm_sends_the_epoch_it_read_and_retries_a_stamp_once():
    def answers(method, _path, body):
        if method == "GET":
            return 200, {"switch_epoch": None}
        if body["epoch"] is None:
            return 409, {"detail": {"reason": "stopped_since", "epoch": "42", "stamped": True}}
        return 200, {"target": body["target"], "state": "on", "switch_epoch": body["epoch"]}

    with _FakeTerminal(answers) as terminal:
        chip = sdk_helpers.ChipRoute(
            terminal.base_url, session_id="sess-1", headers={"X-Osprey-Terminal-Secret": "s"}
        )
        reply = sdk_helpers.switch_on_through_chip(chip, "va")

    assert reply == {"target": "va", "state": "on", "switch_epoch": "42"}
    posts = [entry for entry in terminal.seen if entry[0] == "POST"]
    assert [body for *_rest, body in posts] == [
        {"session_id": "sess-1", "target": "va", "state": "on", "epoch": None},
        {"session_id": "sess-1", "target": "va", "state": "on", "epoch": "42"},
    ]
    assert all(entry[1] == sdk_helpers.CHIP_POSTURE_PATH for entry in posts)
    assert terminal.seen[0][1] == f"{sdk_helpers.CHIP_POSTURE_PATH}?session_id=sess-1"
    assert all(entry[2].get("X-Osprey-Terminal-Secret") == "s" for entry in terminal.seen)


def test_the_chip_arm_fails_with_the_refusal_it_was_given():
    def answers(method, _path, _body):
        if method == "GET":
            return 200, {"switch_epoch": "5"}
        return 403, {"detail": {"reason": "tier_never", "phrase": "WRITES OFF"}}

    with _FakeTerminal(answers) as terminal:
        with pytest.raises(AssertionError, match="tier_never"):
            sdk_helpers.switch_on_through_chip(sdk_helpers.ChipRoute(terminal.base_url), "va")
    assert len([entry for entry in terminal.seen if entry[0] == "POST"]) == 1


def _arming_steps(monkeypatch) -> list[tuple]:
    steps: list[tuple] = []
    monkeypatch.setattr(sdk_helpers, "set_writes_tier", lambda _repo: steps.append(("tier",)))

    def record(_repo, *, owner, targets):
        steps.append(("record", owner, targets))
        return control_context.ControlContext(
            target=targets[0], generation=1, switch=dict.fromkeys(targets, "on"), switch_epoch=9
        )

    def chip(route, target):
        steps.append(("chip", route.base_url, target))
        return {"target": target, "state": "on", "switch_epoch": "11"}

    monkeypatch.setattr(sdk_helpers, "switch_on_in_record", record)
    monkeypatch.setattr(sdk_helpers, "switch_on_through_chip", chip)
    return steps


def test_arm_writes_sets_the_tier_before_the_record_switch(tmp_path, monkeypatch):
    steps = _arming_steps(monkeypatch)
    repo = _fake_repo(tmp_path)

    armed = sdk_helpers.arm_writes(repo, owner="alice")

    assert steps == [("tier",), ("record", "alice", [_baseline()])]
    assert armed == sdk_helpers.ArmedWrites(
        owner="alice", targets=[_baseline()], via="record", switch_epoch=9
    )


def test_arm_writes_through_the_chip_posts_each_target(tmp_path, monkeypatch):
    steps = _arming_steps(monkeypatch)
    repo = _fake_repo(tmp_path)

    armed = sdk_helpers.arm_writes(
        repo, owner="bob", targets=["va", "live"], chip=sdk_helpers.ChipRoute("http://t")
    )

    assert steps == [("tier",), ("chip", "http://t", "live"), ("chip", "http://t", "va")]
    assert armed.via == "chip" and armed.owner == "bob" and armed.switch_epoch == 11


def test_enable_writes_in_project_arms_the_identity_this_process_acts_as(tmp_path, monkeypatch):
    seen: list[dict] = []
    monkeypatch.setattr(sdk_helpers, "arm_writes", lambda repo, **kwargs: seen.append(kwargs))

    sdk_helpers.enable_writes_in_project(tmp_path)
    sdk_helpers.enable_writes_in_project(tmp_path, owner="carol")

    assert seen == [{"owner": acting_identity()}, {"owner": "carol"}]


@pytest.mark.parametrize("owner", ["alice", None])
def test_a_queue_add_names_its_owner_in_the_owner_header(owner):
    def answers(_method, _path, body):
        return 200, {"run_id": "run-1", "revision": body["draft_revision"]}

    with _FakeTerminal(answers) as bridge:
        run_id = _queue_drive.enqueue(bridge.base_url, 3, token="tok", owner=owner)

    assert run_id == "run-1"
    ((_method, path, headers, _body),) = bridge.seen
    assert path == "/queue/items"
    assert headers.get(OWNER_HEADER) == owner
    assert headers.get(_queue_drive.LAUNCH_TOKEN_HEADER) == "tok"


def test_the_arming_helpers_spell_no_retired_posture_key():
    retired = "writes" + "_enabled"
    for module in (sdk_helpers, _queue_drive):
        assert retired not in inspect.getsource(module), module.__name__
