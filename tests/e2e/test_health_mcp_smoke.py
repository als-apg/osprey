"""Agentic e2e acceptance gate for the system-health MCP server.

The health MCP server is opt-in (``claude_code.servers.health.enabled: true``);
the ``control_assistant`` preset turns it on and defaults to a mock control
system. This test is the end-to-end proof that the *poll tier* is wired all the
way through: a project built from that preset, driven by a real agent session
with an operator-style prompt, reaches for ``mcp__health__health_check`` on its
own and gets back a result that parses as the locked wire contract.

Design notes:
  * Operator-style prompt only — it asks whether the machine is healthy right
    now and does *not* name the tool, the server, any category name, or any
    JSON key. If a natural operator prompt fails to reach the tool, that is a
    real wiring/tool-description finding, to be fixed in the framework, never by
    steering the prompt toward the tool.
  * Poll tier ONLY. The test never invokes or depends on ``health_check_full``
    (that tool is approval-gated; e2e must not depend on approval flows).
  * ``@flaky(reruns=2)`` mirrors the other multi-step agentic e2e tests
    (e.g. ``test_rf_cavity_correlation_scenario``) to absorb rare stochastic
    LLM misses without weakening the contract.

Budget: max_turns=4, max_budget_usd=0.25 (per the preset-agentic convention).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e.sdk_helpers import (
    HAS_SDK,
    init_project,
    is_claude_code_available,
    run_sdk_query,
)

# ALS_APG_API_KEY is enforced via ``requires_als_apg`` — the root
# ``tests/conftest.py`` hook auto-skips when the key is missing. The test passes
# ``provider="als-apg"`` to ``init_project`` (the GitHub-Actions default).
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.e2e_smoke,
    pytest.mark.requires_als_apg,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
    pytest.mark.skipif(not is_claude_code_available(), reason="claude CLI not available"),
]

# The poll-tier wire contract (see health_check tool docstring / CheckReport).
_REPORT_FIELDS = ("summary", "ok", "warnings", "errors", "skips", "total")
_ENVELOPE_FIELDS = ("cached", "age_s", "refresh_suppressed")


def _unwrap_health_payload(raw: str) -> dict | None:
    """Parse a captured health_check tool result into the health report dict.

    ``health_check`` returns a JSON *string*; FastMCP surfaces a ``str`` return
    value as structured content nested under a single ``result`` key (and the
    inner value is itself a JSON string). Depending on which representation the
    SDK captured, the raw text is either the health payload directly or that
    ``{"result": "<json>"}`` transport envelope. Peel off the transport layer
    (recursively, tolerating the double-encoded inner string) and return the
    first dict that looks like the health report, else ``None``.
    """
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError):
        return None
    for _ in range(3):  # bounded unwrap of the transport nesting
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except (TypeError, ValueError):
                return None
            continue
        if not isinstance(obj, dict):
            return None
        if "summary" in obj:
            return obj
        if set(obj) == {"result"}:  # FastMCP str-return transport envelope
            obj = obj["result"]
            continue
        return None  # a dict that is neither the report nor the transport envelope
    return obj if isinstance(obj, dict) and "summary" in obj else None


@pytest.mark.flaky(reruns=2)  # multi-step agentic; absorb rare stochastic LLM misses
@pytest.mark.asyncio
async def test_health_poll_tier_from_operator_prompt(tmp_path: Path) -> None:
    """Operator asks 'is the machine healthy?'; agent runs the poll-tier check.

    Builds a ``control_assistant`` project (health server enabled, mock control
    system) and asserts:
      1. the transcript shows a *successful* ``mcp__health__health_check`` call
         (poll tier, allow-listed — no approval flow), and
      2. its result parses as the wire contract — the ``summary`` and count
         fields plus the ``cached``/``age_s``/``refresh_suppressed`` envelope.
    """
    project = init_project(
        tmp_path, "health_smoke", template="control_assistant", provider="als-apg"
    )

    # Operator-style prompt: no tool/server/category/JSON-key hand-feeding.
    query = "Is the machine healthy right now? Give me a quick status rundown."
    result = await run_sdk_query(project, query, max_turns=4, max_budget_usd=0.25)

    health_calls = [t for t in result.tool_traces if t.name == "mcp__health__health_check"]
    assert health_calls, (
        "agent did not call mcp__health__health_check from a natural operator "
        f"health prompt. Tools called: {result.tool_names}. If this is a wiring "
        "or tool-description gap, fix it in the framework — do not tune the "
        "prompt toward the tool."
    )

    # A successful poll-tier call: at least one health_check trace that is not an
    # error and whose result parses as the locked wire contract.
    successful = [t for t in health_calls if not t.is_error and t.result]
    assert successful, (
        "health_check was called but every call errored or returned no result: "
        f"{[(t.is_error, (t.result or '')[:200]) for t in health_calls]}"
    )

    envelope = None
    seen: list[str] = []
    for trace in successful:
        candidate = _unwrap_health_payload(trace.result)
        if candidate is not None and "summary" in candidate:
            envelope = candidate
            break
        seen.append((trace.result or "")[:200])
    assert envelope is not None, (
        "no successful health_check result parsed as the health report contract. "
        f"Raw results (truncated): {seen}"
    )

    missing_report = [k for k in _REPORT_FIELDS if k not in envelope]
    assert not missing_report, (
        f"health_check result missing report fields {missing_report}. Got keys: {sorted(envelope)}"
    )

    missing_envelope = [k for k in _ENVELOPE_FIELDS if k not in envelope]
    assert not missing_envelope, (
        f"health_check result missing envelope fields {missing_envelope}. "
        f"Got keys: {sorted(envelope)}"
    )

    # Poll tier must never route through the approval-gated full tier.
    assert not any(t.name == "mcp__health__health_check_full" for t in result.tool_traces), (
        "poll-tier smoke test must not depend on the approval-gated "
        "health_check_full tool; the agent should not have called it."
    )
