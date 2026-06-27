"""Single-source guard for agent_runner primitives.

Prevents the "third diverging copy" regression: SDKWorkflowResult, ToolTrace,
sdk_env, and combined_text must live ONLY in osprey.agent_runner.primitives.
Both tests/e2e/sdk_helpers.py and src/osprey/services/channel_finder/benchmarks/sdk.py
re-export (import) these symbols rather than re-defining them. This test fails fast
if anyone reintroduces a local copy in either consumer module.
"""

import inspect

import osprey.services.channel_finder.benchmarks.sdk as bench
import tests.e2e.sdk_helpers as helpers
from osprey.agent_runner import (
    SDKWorkflowResult as canonical_workflow_result,
)
from osprey.agent_runner import (
    ToolTrace as canonical_tool_trace,
)
from osprey.agent_runner import (
    sdk_env as canonical_sdk_env,
)

# ---------------------------------------------------------------------------
# 1. Identity checks — the object must be the *same* object, not a copy
# ---------------------------------------------------------------------------


def test_sdk_workflow_result_identity_helpers():
    assert helpers.SDKWorkflowResult is canonical_workflow_result, (
        "tests/e2e/sdk_helpers.SDKWorkflowResult is not the canonical class; "
        "it was re-defined instead of imported from osprey.agent_runner"
    )


def test_sdk_workflow_result_identity_bench():
    assert bench.SDKWorkflowResult is canonical_workflow_result, (
        "osprey.services.channel_finder.benchmarks.sdk.SDKWorkflowResult is not "
        "the canonical class; it was re-defined instead of imported from osprey.agent_runner"
    )


def test_tool_trace_identity_helpers():
    assert helpers.ToolTrace is canonical_tool_trace, (
        "tests/e2e/sdk_helpers.ToolTrace is not the canonical class"
    )


def test_tool_trace_identity_bench():
    assert bench.ToolTrace is canonical_tool_trace, (
        "osprey.services.channel_finder.benchmarks.sdk.ToolTrace is not the canonical class"
    )


def test_sdk_env_identity_helpers():
    assert helpers.sdk_env is canonical_sdk_env, (
        "tests/e2e/sdk_helpers.sdk_env is not the canonical function"
    )


def test_sdk_env_identity_bench():
    assert bench.sdk_env is canonical_sdk_env, (
        "osprey.services.channel_finder.benchmarks.sdk.sdk_env is not the canonical function"
    )


# ---------------------------------------------------------------------------
# 2. Source-grep guard — the consumer files must not re-define the primitives
#    File paths are resolved via __file__ so the test is portable.
# ---------------------------------------------------------------------------

_FORBIDDEN_PATTERNS = [
    "class SDKWorkflowResult",
    "class ToolTrace",
    "def sdk_env",
    "def combined_text",
    "def _ingest_tool_result",
]


def _assert_no_local_definition(module, label: str) -> None:
    path = inspect.getfile(module)
    source = open(path).read()
    for pattern in _FORBIDDEN_PATTERNS:
        assert pattern not in source, (
            f"{label} ({path}) re-defines '{pattern}'. "
            "Move the definition back to osprey.agent_runner.primitives and import from there."
        )


def test_sdk_helpers_no_local_definitions():
    _assert_no_local_definition(helpers, "tests/e2e/sdk_helpers")


def test_bench_sdk_no_local_definitions():
    _assert_no_local_definition(bench, "osprey.services.channel_finder.benchmarks.sdk")
