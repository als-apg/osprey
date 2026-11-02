"""A code review note reaches the agent in every reply the execute tools give.

At ``flag`` the review lets the run proceed and leaves its concern on
``ExecutionResult.code_review``. ``build_execution_response`` is what puts it in
front of the agent, so these tests pin the two reply shapes it builds, on success
and on failure, and that a run with no note replies exactly as it always did.
"""

from __future__ import annotations

import pytest

from osprey.mcp_server.python_executor.executor import (
    FAILURE_KIND_APPROVAL_MISSING,
    FAILURE_KIND_SETUP,
    FAILURE_KIND_TIMEOUT,
    ExecutionResult,
)
from osprey.mcp_server.python_executor.tools._response_builder import build_execution_response
from tests.mcp_server.conftest import assert_error, extract_response_dict

NOTE = "The code review flagged this code: it sets a magnet without reading it back."


@pytest.fixture(autouse=True)
def _store(tmp_path):
    """Initialize the ArtifactStore singleton in a throwaway workspace."""
    from osprey.stores.artifact_store import initialize_artifact_store

    initialize_artifact_store(workspace_root=tmp_path)
    yield


def _result(*, success: bool, code_review: str | None = None, **kw) -> ExecutionResult:
    return ExecutionResult(
        success=success,
        stdout="hello\n",
        stderr="" if success else "Traceback: boom",
        execution_method_used="subprocess",
        code_review=code_review,
        **kw,
    )


async def _build(exec_result: ExecutionResult, *, save_output: bool) -> str:
    return await build_execution_response(
        code="print('hi')",
        description="demo run",
        execution_mode="readwrite",
        exec_result=exec_result,
        patterns={},
        save_output=save_output,
    )


async def _failure_envelope(exec_result: ExecutionResult, *, save_output: bool) -> dict:
    with pytest.raises(Exception) as exc_info:
        await _build(exec_result, save_output=save_output)
    return assert_error(str(exc_info.value))


async def test_a_successful_inline_run_carries_the_note():
    data = extract_response_dict(
        await _build(_result(success=True, code_review=NOTE), save_output=False)
    )
    assert data["code_review"] == NOTE
    assert data["has_errors"] is False


async def test_a_successful_saved_run_carries_the_note_in_its_summary():
    data = extract_response_dict(
        await _build(_result(success=True, code_review=NOTE), save_output=True)
    )
    assert data["summary"]["code_review"] == NOTE
    assert data["summary"]["status"] == "Success"


async def test_a_long_note_is_not_cut_at_the_output_preview_limit():
    """The summary trims stdout and stderr; a truncated concern would misstate the review."""
    note = "x" * 2000
    data = extract_response_dict(
        await _build(_result(success=True, code_review=note), save_output=True)
    )
    assert data["summary"]["code_review"] == note


@pytest.mark.parametrize("save_output", [False, True], ids=["inline", "saved"])
@pytest.mark.parametrize(
    ("failure_kind", "error_type", "kind"),
    [
        (None, "execution_error", "script_error"),
        (FAILURE_KIND_TIMEOUT, "execution_error", "timeout"),
        (FAILURE_KIND_APPROVAL_MISSING, "execution_error", "approval_missing"),
        (FAILURE_KIND_SETUP, "service_unavailable", "setup_failed"),
    ],
)
async def test_a_failing_run_carries_the_note_in_details(
    failure_kind, error_type, kind, save_output
):
    """Every failure class, both shapes: the concern outlives the run's failure."""
    envelope = await _failure_envelope(
        _result(success=False, code_review=NOTE, failure_kind=failure_kind),
        save_output=save_output,
    )
    assert envelope["error_type"] == error_type
    assert envelope["details"]["kind"] == kind
    assert envelope["details"]["code_review"] == NOTE


async def test_a_failing_saved_run_also_carries_the_note_in_its_summary():
    envelope = await _failure_envelope(_result(success=False, code_review=NOTE), save_output=True)
    assert envelope["details"]["summary"]["code_review"] == NOTE


async def test_without_a_note_the_inline_reply_has_exactly_its_old_keys():
    data = extract_response_dict(await _build(_result(success=True), save_output=False))
    assert set(data) == {
        "description",
        "execution_mode",
        "execution_method",
        "stdout",
        "stderr",
        "has_errors",
        "detected_patterns",
        "artifact_ids",
    }


async def test_without_a_note_the_saved_summary_has_exactly_its_old_keys():
    data = extract_response_dict(await _build(_result(success=True), save_output=True))
    assert "code_review" not in data
    assert set(data["summary"]) == {
        "description",
        "status",
        "output",
        "error",
        "has_errors",
        "detected_patterns",
        "artifact_ids",
        "artifact_count",
    }


@pytest.mark.parametrize("save_output", [False, True], ids=["inline", "saved"])
async def test_without_a_note_a_failure_has_no_code_review_key(save_output):
    envelope = await _failure_envelope(_result(success=False), save_output=save_output)
    assert "code_review" not in envelope["details"]
    assert "code_review" not in envelope["details"].get("summary", {})
