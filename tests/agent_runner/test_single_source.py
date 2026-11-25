"""Single-source guard for agent_runner primitives.

Prevents the "third diverging copy" regression: SDKWorkflowResult, ToolTrace,
sdk_env, and combined_text must live ONLY in osprey.agent_runner.primitives.
Both tests/e2e/sdk_helpers.py and src/osprey/services/channel_finder/benchmarks/sdk.py
re-export (import) these symbols rather than re-defining them. This test fails fast
if anyone reintroduces a local copy in either consumer module.
"""

import inspect

import pytest

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


@pytest.mark.parametrize(
    ("module", "attr", "canonical"),
    [
        pytest.param(
            helpers, "SDKWorkflowResult", canonical_workflow_result, id="helpers-SDKWorkflowResult"
        ),
        pytest.param(
            bench, "SDKWorkflowResult", canonical_workflow_result, id="bench-SDKWorkflowResult"
        ),
        pytest.param(helpers, "ToolTrace", canonical_tool_trace, id="helpers-ToolTrace"),
        pytest.param(bench, "ToolTrace", canonical_tool_trace, id="bench-ToolTrace"),
        pytest.param(helpers, "sdk_env", canonical_sdk_env, id="helpers-sdk_env"),
        pytest.param(bench, "sdk_env", canonical_sdk_env, id="bench-sdk_env"),
    ],
)
def test_consumer_reexports_the_canonical_object(module, attr, canonical):
    assert getattr(module, attr) is canonical, (
        f"{module.__name__}.{attr} is not the canonical object; "
        "it was re-defined instead of imported from osprey.agent_runner"
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


# ---------------------------------------------------------------------------
# 3. Fence guard — the bench module reaches the agent only through the runner
# ---------------------------------------------------------------------------


def test_bench_sdk_reaches_the_agent_only_through_the_runner():
    path = inspect.getfile(bench)
    source = open(path).read()
    for pattern in ("claude_agent_sdk", "agent_runner.primitives"):
        assert pattern not in source, (
            f"osprey.services.channel_finder.benchmarks.sdk ({path}) names '{pattern}'. "
            "Run the agent through osprey.agent_runner.run_query instead."
        )
