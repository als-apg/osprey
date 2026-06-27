"""Exit-code verdict for a completed SDK query run.

Translates the structured ``SDKWorkflowResult`` into one of two exit codes:

* ``EXIT_PASS`` (0) — all expected MCP servers are connected, the run
  completed without error, and no budget or turn-limit was exceeded.
* ``EXIT_VERDICT_FAIL`` (1) — any of those conditions is not met.

The third constant, ``EXIT_USAGE`` (2), is reserved for the CLI layer
(Task 2.1) to report infrastructure or usage errors that prevented the run
from starting at all.  It lives here so callers can import all three
constants from a single module.
"""

from __future__ import annotations

from osprey.agent_runner.primitives import SDKWorkflowResult

# ---------------------------------------------------------------------------
# Exit-code constants
# ---------------------------------------------------------------------------

EXIT_PASS: int = 0
"""Verdict: the query completed successfully."""

EXIT_VERDICT_FAIL: int = 1
"""Verdict: the query failed (server missing, result error, or budget hit)."""

EXIT_USAGE: int = 2
"""Reserved for the CLI layer: infrastructure / usage error before run start."""

# ResultMessage subtypes that signal a resource cap was hit.  Both carry
# ``is_error=True`` in practice, but the explicit subtype check makes the
# budget/turn branch self-documenting and robust to future SDK changes.
_BUDGET_SUBTYPES: frozenset[str] = frozenset({"error_max_budget_usd", "error_max_turns"})


def evaluate_verdict(result: SDKWorkflowResult, expected_servers: set[str]) -> int:
    """Evaluate whether an SDK query run should be considered passing.

    Args:
        result: Aggregated result from a completed SDK query run.
        expected_servers: MCP server names (e.g. ``{"controls", "python"}``)
            that must report ``"connected"`` in the pre-prompt snapshot for
            the run to be considered valid.  An empty set skips the server
            check.

    Returns:
        ``EXIT_PASS`` (0) when every server in *expected_servers* reports
        ``"connected"``, the ``ResultMessage`` is present, ``is_error`` is
        ``False``, and the subtype is not a budget or turn-limit sentinel.
        ``EXIT_VERDICT_FAIL`` (1) otherwise.
    """
    # --- MCP server connectivity -----------------------------------------
    if expected_servers:
        status = result.mcp_server_status  # dict[str, str]
        for server in expected_servers:
            if status.get(server) != "connected":
                return EXIT_VERDICT_FAIL

    # --- Result presence and error flag ------------------------------------
    msg = result.result
    if msg is None:
        return EXIT_VERDICT_FAIL

    if msg.is_error:
        return EXIT_VERDICT_FAIL

    # --- Budget / turn-limit (defence-in-depth) ----------------------------
    if msg.subtype in _BUDGET_SUBTYPES:
        return EXIT_VERDICT_FAIL

    return EXIT_PASS
