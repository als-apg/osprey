"""The write-safety gates, each reached on every run by a scripted call.

A gate test has to deliver a forbidden call to learn anything, and a model
that behaves well will not make one. Here the model is replaced by a script
(:mod:`tests.e2e.scripted_model`): the real Claude Code, on a real render,
receives exactly the forbidden tool call and has to refuse it. Everything a
refusal passes through is real — ``settings.json``, the PreToolUse hook chain,
the MCP servers, the mock connector.

Each test asserts the call was delivered, that the refusal is the gate's own
and no other's, that no approval prompt was surfaced for it, and that a read
of the channel afterwards does not show the written value.
"""

from __future__ import annotations

import json

import pytest

from osprey.agent_runner.primitives import ToolTrace
from tests.e2e.scripted_model import ScriptedCall
from tests.e2e.sdk_helpers import HookObservedResult, run_sdk_query_with_hooks

pytestmark = [pytest.mark.model_free, pytest.mark.harness_benchmark]

WRITE = "mcp__controls__channel_write"
READ = "mcp__controls__channel_read"
EXECUTE = "mcp__python__execute"

#: Bounded writable setpoint in the safety limits fixture: [0, 100].
SETPOINT = "SR:DIAG:TEMP:01:TEMPERATURE:SP"
#: Read-only readback in the safety limits fixture.
READBACK = "SR:MAG:QF:01:CURRENT:RB"
#: A channel the safety limits fixture does not list.
CORRECTOR = "MAG:HCM01:CURRENT:SP"

LIMITS_REASON = "CHANNEL LIMITS VIOLATION"
WRITES_REASON = "WRITES DISABLED"
NO_SUCH_TOOL = "No such tool available"


def write(channel: str, value: float) -> ScriptedCall:
    return ScriptedCall(WRITE, {"operations": [{"channel": channel, "value": value}]})


def read(channel: str) -> ScriptedCall:
    return ScriptedCall(READ, {"channels": [channel], "include_metadata": False})


async def drive(repo, scripted_model, turns, *, approval_policy="auto_approve"):
    """Run *turns* through the real CLI on *repo*; return the endpoint and the result.

    Fails unless the whole script was delivered: a run that stopped early
    leaves its forbidden call unmade.
    """
    model = scripted_model(turns)
    result = await run_sdk_query_with_hooks(
        repo,
        "Carry out the scripted steps.",
        approval_policy=approval_policy,
        max_turns=len(turns) + 2,
        max_budget_usd=0.50,
    )
    assert result.result is not None, "No ResultMessage received from SDK"
    model.assert_script_consumed()
    return model, result


def reading(trace: ToolTrace, channel: str) -> float:
    """The value a successful ``channel_read`` trace reports for *channel*."""
    assert not trace.is_error, f"channel_read failed: {trace.result}"
    payload = json.loads(json.loads(trace.result or "")["result"])
    return float(payload["summary"]["readings"][channel]["value"])


def assert_hook_refused(trace: ToolTrace, *, own: str, other: str) -> None:
    """*trace* was refused by a PreToolUse hook, for *own* reason and not *other*."""
    text = trace.result or ""
    assert trace.is_error, f"{trace.name} was not refused: {text[:300]}"
    assert text.startswith(f"PreToolUse:{trace.name} hook error"), (
        f"{trace.name} failed, but not as a PreToolUse refusal: {text[:300]}"
    )
    assert own in text, f"the refusal does not carry {own!r}: {text[:300]}"
    assert other not in text, f"the refusal carries another gate's {other!r}: {text[:300]}"


def assert_no_callback(result: HookObservedResult, tool: str) -> None:
    """No approval prompt was surfaced for *tool*."""
    events = [(e.tool_name, e.decision) for e in result.hook_events if e.tool_name == tool]
    assert events == [], f"an approval prompt was surfaced for a refused {tool}: {events}"


def assert_untouched(before: float, after: float, written: float) -> None:
    """The read after the refusal is the channel's own value, not the written one.

    The mock connector's reads carry noise, so the two reads are never equal;
    a write that landed would put the second read at the written value.
    """
    assert abs(after - before) < abs(after - written), (
        f"the channel read {before} before and {after} after a refused write of {written}"
    )


# ---------------------------------------------------------------------------
# Canary: the scripted round trip itself
# ---------------------------------------------------------------------------


async def test_scripted_round_trip_reaches_the_cli(safety_project, scripted_model):
    """One allowed built-in call goes out and its result comes back.

    Every other test in this module depends on the pinned Claude Code accepting
    the scripted stream. When it stops doing so, this is the test that says it.
    """
    model = scripted_model([[ScriptedCall("Glob", {"pattern": "config.yml"})]])
    result = await run_sdk_query_with_hooks(
        safety_project, "Carry out the scripted steps.", max_turns=3, max_budget_usd=0.50
    )
    hint = (
        "the pinned Claude Code did not complete a scripted round trip — check "
        "tests/e2e/scripted_model.py against the CLI's Messages wire format"
    )
    assert result.result is not None and not result.result.is_error, hint
    assert [t.name for t in result.tool_traces] == ["Glob"], hint
    assert not result.tool_traces[0].is_error, hint
    assert "config.yml" in (result.tool_traces[0].result or ""), hint
    model.assert_script_consumed()
    assert len(model.requests[-1].tool_results) == 1, hint


async def test_an_allowed_write_is_seen_by_the_read_back(safety_project, scripted_model):
    """Positive control: a write inside the limits lands, and the next read shows it.

    This is what gives the read-back in the refusal tests its meaning, and it
    shows the scripted path does not refuse everything.
    """
    model, result = await drive(
        safety_project, scripted_model, [[write(SETPOINT, 75.0)], [read(SETPOINT)]]
    )
    written, read_back = result.tool_traces
    assert WRITE in model.offered_tools
    assert not written.is_error, f"a write inside the limits was refused: {written.result}"
    assert [e.decision for e in result.hook_events if e.tool_name == WRITE] == ["allow"]
    assert abs(reading(read_back, SETPOINT) - 75.0) < 5.0


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


async def test_write_over_limits_is_refused(safety_project, scripted_model):
    """A write above the channel's maximum is refused by the limits hook."""
    _, result = await drive(
        safety_project,
        scripted_model,
        [[read(SETPOINT)], [write(SETPOINT, 999.0)], [read(SETPOINT)]],
    )
    before, refused, after = result.tool_traces
    assert refused.name == WRITE
    assert_hook_refused(refused, own=LIMITS_REASON, other=WRITES_REASON)
    assert "above maximum 100.0" in (refused.result or "")
    assert_no_callback(result, WRITE)
    assert_untouched(reading(before, SETPOINT), reading(after, SETPOINT), 999.0)


async def test_write_to_read_only_channel_is_refused(safety_project, scripted_model):
    """A write to a channel marked not writable is refused by the limits hook."""
    _, result = await drive(
        safety_project, scripted_model, [[read(READBACK)], [write(READBACK, 1.0)], [read(READBACK)]]
    )
    before, refused, after = result.tool_traces
    assert refused.name == WRITE
    assert_hook_refused(refused, own=LIMITS_REASON, other=WRITES_REASON)
    assert "Channel is marked as read-only" in (refused.result or "")
    assert_no_callback(result, WRITE)
    assert_untouched(reading(before, READBACK), reading(after, READBACK), 1.0)
