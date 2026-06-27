"""Tests for osprey.agent_runner.verdict.evaluate_verdict.

Covers:
  - All expected servers connected + no error + within budget → EXIT_PASS (0).
  - An expected server not connected → EXIT_VERDICT_FAIL (1).
  - Result indicates error → EXIT_VERDICT_FAIL (1).
  - Budget exceeded → EXIT_VERDICT_FAIL (1).
  - Turn-limit exceeded → EXIT_VERDICT_FAIL (1).
  - Empty expected_servers set + ok result → EXIT_PASS (0).
"""

from __future__ import annotations

from claude_agent_sdk import ResultMessage

from osprey.agent_runner import SDKWorkflowResult
from osprey.agent_runner.verdict import EXIT_PASS, EXIT_VERDICT_FAIL, evaluate_verdict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_result(subtype: str = "success") -> ResultMessage:
    """Return a ResultMessage with is_error=False and the given subtype."""
    return ResultMessage(
        subtype=subtype,
        duration_ms=100,
        duration_api_ms=100,
        is_error=False,
        num_turns=3,
        session_id="test-session",
    )


def _error_result(subtype: str = "error_during_execution") -> ResultMessage:
    """Return a ResultMessage with is_error=True and the given subtype."""
    return ResultMessage(
        subtype=subtype,
        duration_ms=100,
        duration_api_ms=100,
        is_error=True,
        num_turns=1,
        session_id="test-session",
    )


def _workflow(
    result: ResultMessage | None,
    server_names: list[str],
    status: str = "connected",
) -> SDKWorkflowResult:
    """Construct an SDKWorkflowResult with a synthetic MCP server snapshot."""
    mcp_servers = [{"name": name, "status": status} for name in server_names]
    return SDKWorkflowResult(
        result=result,
        mcp_servers=mcp_servers,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_all_connected_no_error_returns_pass() -> None:
    """All expected servers connected and a clean result → EXIT_PASS."""
    wf = _workflow(_ok_result(), ["controls", "python"])
    assert evaluate_verdict(wf, {"controls", "python"}) == EXIT_PASS


def test_empty_expected_servers_ok_result_returns_pass() -> None:
    """Empty expected_servers set with a clean result → EXIT_PASS."""
    wf = _workflow(_ok_result(), [])
    assert evaluate_verdict(wf, set()) == EXIT_PASS


# ---------------------------------------------------------------------------
# MCP server not connected
# ---------------------------------------------------------------------------


def test_missing_expected_server_returns_fail() -> None:
    """A required server absent from the snapshot → EXIT_VERDICT_FAIL."""
    # Only 'python' is in the snapshot; 'controls' is missing entirely.
    wf = _workflow(_ok_result(), ["python"])
    assert evaluate_verdict(wf, {"controls", "python"}) == EXIT_VERDICT_FAIL


def test_expected_server_not_connected_returns_fail() -> None:
    """Required server present but status != 'connected' → EXIT_VERDICT_FAIL."""
    mcp_servers = [
        {"name": "controls", "status": "disconnected"},
        {"name": "python", "status": "connected"},
    ]
    wf = SDKWorkflowResult(result=_ok_result(), mcp_servers=mcp_servers)
    assert evaluate_verdict(wf, {"controls", "python"}) == EXIT_VERDICT_FAIL


# ---------------------------------------------------------------------------
# Result error conditions
# ---------------------------------------------------------------------------


def test_result_is_error_returns_fail() -> None:
    """ResultMessage with is_error=True → EXIT_VERDICT_FAIL."""
    wf = _workflow(_error_result("error_during_execution"), ["controls"])
    assert evaluate_verdict(wf, {"controls"}) == EXIT_VERDICT_FAIL


def test_result_none_returns_fail() -> None:
    """Missing ResultMessage (result=None) → EXIT_VERDICT_FAIL."""
    wf = _workflow(None, ["controls"])
    assert evaluate_verdict(wf, {"controls"}) == EXIT_VERDICT_FAIL


# ---------------------------------------------------------------------------
# Budget / turn-limit exceeded
# ---------------------------------------------------------------------------


def test_budget_exceeded_returns_fail() -> None:
    """ResultMessage with subtype='error_max_budget_usd' → EXIT_VERDICT_FAIL."""
    wf = _workflow(_error_result("error_max_budget_usd"), ["controls"])
    assert evaluate_verdict(wf, {"controls"}) == EXIT_VERDICT_FAIL


def test_turn_limit_exceeded_returns_fail() -> None:
    """ResultMessage with subtype='error_max_turns' → EXIT_VERDICT_FAIL."""
    wf = _workflow(_error_result("error_max_turns"), ["controls"])
    assert evaluate_verdict(wf, {"controls"}) == EXIT_VERDICT_FAIL


def test_budget_subtype_without_error_flag_returns_fail() -> None:
    """The _BUDGET_SUBTYPES branch (defense-in-depth) must fail even when
    is_error is NOT set — the only input shape that actually reaches it, since
    the is_error gate short-circuits first. Pins the documented contract that
    is robust to future SDK changes."""
    wf = _workflow(_ok_result("error_max_budget_usd"), ["controls"])
    assert evaluate_verdict(wf, {"controls"}) == EXIT_VERDICT_FAIL


def test_turn_subtype_without_error_flag_returns_fail() -> None:
    """error_max_turns with is_error=False also fails via the subtype branch."""
    wf = _workflow(_ok_result("error_max_turns"), ["controls"])
    assert evaluate_verdict(wf, {"controls"}) == EXIT_VERDICT_FAIL


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_constants_values() -> None:
    """EXIT_PASS=0, EXIT_VERDICT_FAIL=1 (EXIT_USAGE=2 reserved for CLI layer)."""
    from osprey.agent_runner.verdict import EXIT_USAGE

    assert EXIT_PASS == 0
    assert EXIT_VERDICT_FAIL == 1
    assert EXIT_USAGE == 2
