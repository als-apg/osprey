"""``suggested_revision`` end to end, on both surfaces, with only the reviewer's run stubbed.

A revision is advice for the agent. It reaches the agent verbatim, it never replaces the code
that was submitted, and it never reaches an audit record. The reviewer is stubbed at
``review.run_query``, the single seam, so the real execute adapter and the real hook ``main``
do everything else.
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from osprey.agent_runner.primitives import SDKWorkflowResult
from osprey.audit import posture
from osprey.mcp_server.python_executor import executor
from osprey.mcp_server.python_executor.executor import ExecutionResult
from osprey.mcp_server.python_executor.tools import _execution_gates
from osprey.services.code_review import review as review_module
from osprey.services.code_review.review import AUDIT_REASON
from osprey.services.code_review.settings import DEFAULT_PROMPT
from osprey_connectors.workspace import DEFAULT_AGENT_DATA_BASE_DIR
from osprey_connectors_tests._write_fakes import ARMED_SECTION, ARMED_TARGET, arm_switch
from tests.hooks.conftest import _reset_hook_log_caches, hook_audit_records, import_hook
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

SUBMITTED = "print('SUBMITTED_BY_THE_AGENT')\n"
SEED_SOURCE = "seed = 1"
REASON = "the stub objects to this code"

#: Multi-line, indented on its first line, mixed case and ending in a newline, so that any
#: stripping, normalising or re-flowing on the way to the agent breaks the comparison.
REVISION = (
    "    def Revised_Write(Value):\n"
    "        # Keep THE Limit Check\n"
    "        return Clamp_To_Limit(Value)\n"
    "\n"
    "\tRevised_Result = Revised_Write(1)\n"
)
REVISION_LINES = [line.strip() for line in REVISION.splitlines() if line.strip()]

#: verdict, level, whether the review refuses.
CASES = pytest.mark.parametrize(
    ("verdict", "level", "refused"),
    [
        pytest.param("block", "block", True, id="block-at-block"),
        pytest.param("block", "flag", False, id="block-at-flag"),
        pytest.param("concern", "block", False, id="concern-at-block"),
        pytest.param("concern", "flag", False, id="concern-at-flag"),
    ],
)

UNDER_THE_LABEL = f"Suggested revision:\n{REVISION}"


class ReviewerStub:
    """Stands in for ``run_query``: records each call, then answers with the revision."""

    total_calls = 0

    def __init__(self, verdict: str) -> None:
        self.reply = json.dumps(
            {"verdict": verdict, "reasons": [REASON], "suggested_revision": REVISION}
        )
        self.messages: list[str] = []

    async def __call__(self, _render_dir, message, **_options) -> SDKWorkflowResult:
        ReviewerStub.total_calls += 1
        self.messages.append(message)
        return SDKWorkflowResult(
            text_blocks=[self.reply],
            result=SimpleNamespace(subtype="success", is_error=False),  # type: ignore[arg-type]
        )

    def assert_reviewed_once(self) -> None:
        (message,) = self.messages
        assert SUBMITTED.strip() in message
        assert not any(line in message for line in REVISION_LINES)


@pytest.fixture(autouse=True)
def _a_recorded_call(monkeypatch):
    """The record a prompt, or the switch at ``allowed``, leaves: a governed call needs one."""
    monkeypatch.setattr(executor, "_check_approved_call", lambda *args, **kwargs: None)


@pytest.fixture(autouse=True, scope="module")
def _a_silent_decline_fails_loudly():
    """An admission refusal skips the review silently, and would leave nothing to assert on."""
    yield
    assert ReviewerStub.total_calls, "the reviewer stub was never called"


def _write_deployment(root: Path, level: str) -> None:
    """A config that reviews at *level* and asks nobody, and the prompt the review needs."""
    prompt = root / DEFAULT_PROMPT
    prompt.parent.mkdir(parents=True, exist_ok=True)
    prompt.write_text("Review this code.", encoding="utf-8")
    config = {
        "code_review": {"enabled": True, "level": level},
    }
    (root / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _assert_free_of_revision(record) -> None:
    text = str(record)
    assert REVISION not in text
    for line in REVISION_LINES:
        assert line not in text


# ---------------------------------------------------------------------------
# execute: the real adapter, with the subprocess backend replaced
# ---------------------------------------------------------------------------


@pytest.fixture
def execute_root(tmp_path, control_context_root, monkeypatch):
    """A scratch render on the simulator with the switch on, read through ``OSPREY_CONFIG``."""
    from osprey_connectors import posture_store

    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(executor, "_resolve_project_root", lambda: root)
    monkeypatch.chdir(root)
    monkeypatch.setenv("OSPREY_CONFIG", str(root / "config.yml"))
    monkeypatch.delenv("OSPREY_CONTROL_TARGET", raising=False)
    monkeypatch.delenv(_execution_gates.TOOL_PREFIX_ENV_VAR, raising=False)
    monkeypatch.setattr(posture, "control_system_section", lambda: dict(ARMED_SECTION))
    monkeypatch.setattr(_execution_gates, "notify_agent_activity_async", AsyncMock())
    arm_switch(monkeypatch, control_context_root, ARMED_TARGET, stamp=False)
    yield root
    posture_store.invalidate_cache()


@pytest.fixture
def launch():
    """The subprocess backend; what it is handed is the code that would have run."""
    result = ExecutionResult(
        success=True,
        stdout="",
        stderr="",
        figures=[],
        execution_method_used="subprocess",
        execution_time_seconds=0.1,
    )
    mock = AsyncMock(return_value=result)
    with patch("osprey.mcp_server.python_executor.executor.execute_code", mock):
        yield mock


@pytest.fixture
def records(monkeypatch):
    """The fields of every record the gate files, whichever writer it uses."""
    filed: list[dict] = []

    def file_record(**fields):
        filed.append(fields)
        return "entry"

    monkeypatch.setattr("osprey.audit.dedup.record_and_mark", file_record)
    monkeypatch.setattr("osprey.audit.writer.record", file_record)
    return filed


async def _execute() -> str:
    from osprey.mcp_server.python_executor.tools.python_execute import execute

    return await get_tool_fn(execute)(
        code=SUBMITTED, description="revision probe", execution_mode="readwrite", save_output=False
    )


@CASES
async def test_execute_carries_the_revision_and_runs_the_submitted_code(
    execute_root, launch, records, monkeypatch, verdict, level, refused
):
    _write_deployment(execute_root, level)
    reviewer = ReviewerStub(verdict)
    monkeypatch.setattr(review_module, "run_query", reviewer)

    if refused:
        with assert_raises_error(error_type="review_blocked") as ctx:
            await _execute()
        envelope = ctx["envelope"]
        assert UNDER_THE_LABEL in envelope["error_message"]
        assert envelope["details"]["suggested_revision"] == REVISION
        launch.assert_not_awaited()
    else:
        reply = extract_response_dict(await _execute())
        assert UNDER_THE_LABEL in reply["code_review"]
        launch.assert_awaited_once()
        assert launch.await_args.kwargs["code"] == SUBMITTED

    reviewer.assert_reviewed_once()
    assert [record["reason"] for record in records] == [AUDIT_REASON]
    for record in records:
        _assert_free_of_revision(record)


# ---------------------------------------------------------------------------
# NotebookEdit: the real hook ``main``, which must run from a sync test (it calls asyncio.run)
# ---------------------------------------------------------------------------


@pytest.fixture
def hook_root(tmp_path, monkeypatch):
    """A scratch deployment the hook finds through ``CLAUDE_PROJECT_DIR``, with its own spool."""
    root = tmp_path / "deployment"
    root.mkdir()
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(root))
    monkeypatch.setenv("OSPREY_SPOOL_DIR", str(root / "spool"))
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.delenv("OSPREY_HOOK_DEBUG", raising=False)
    monkeypatch.chdir(root)
    _reset_hook_log_caches()
    yield root
    _reset_hook_log_caches()


def _notebook(root: Path) -> Path:
    path = root / DEFAULT_AGENT_DATA_BASE_DIR / "notebooks" / "x.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    cell = {"id": "c1", "cell_type": "code", "source": SEED_SOURCE, "outputs": []}
    path.write_text(json.dumps({"cells": [cell]}), encoding="utf-8")
    return path


def _run_hook(notebook: Path, monkeypatch, capsys) -> tuple[dict, dict]:
    """Run ``main`` on an edit submitting ``SUBMITTED``; return its output and the edit."""
    hook = import_hook("osprey_code_review")
    monkeypatch.setattr(hook, "notify_activity", lambda *_args: None)
    tool_input = {
        "notebook_path": str(notebook),
        "new_source": SUBMITTED,
        "cell_id": "c1",
        "edit_mode": "replace",
        "cell_type": "code",
    }
    payload = {"tool_name": "NotebookEdit", "tool_use_id": "toolu_1", "tool_input": tool_input}
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps(payload)))
    monkeypatch.setattr(sys, "argv", ["osprey_code_review.py"])
    with pytest.raises(SystemExit) as exit_info:
        hook.main()
    assert exit_info.value.code == 0
    return json.loads(capsys.readouterr().out)["hookSpecificOutput"], tool_input


def _apply_edit(tool_input: dict) -> None:
    """Do what Claude Code does on a non-deny, since the hook only ever decides."""
    path = Path(tool_input["notebook_path"])
    notebook = json.loads(path.read_text(encoding="utf-8"))
    (cell,) = [cell for cell in notebook["cells"] if cell["id"] == tool_input["cell_id"]]
    cell["source"] = tool_input["new_source"]
    path.write_text(json.dumps(notebook), encoding="utf-8")


@CASES
def test_a_notebook_edit_carries_the_revision_and_lands_the_submitted_cell(
    hook_root, monkeypatch, capsys, verdict, level, refused
):
    _write_deployment(hook_root, level)
    reviewer = ReviewerStub(verdict)
    monkeypatch.setattr(review_module, "run_query", reviewer)
    notebook = _notebook(hook_root)
    before = notebook.read_bytes()

    output, tool_input = _run_hook(notebook, monkeypatch, capsys)

    assert notebook.read_bytes() == before
    if refused:
        assert output["permissionDecision"] == "deny"
        assert UNDER_THE_LABEL in output["permissionDecisionReason"]
    else:
        assert "permissionDecision" not in output
        assert UNDER_THE_LABEL in output["additionalContext"]
        _apply_edit(tool_input)
        (cell,) = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        assert cell["source"] == SUBMITTED
        _assert_free_of_revision(notebook.read_text(encoding="utf-8"))

    reviewer.assert_reviewed_once()
    (record,) = hook_audit_records(hook_root, "hook_code_review")
    assert record["reason"] == AUDIT_REASON
    _assert_free_of_revision(record)
    spool = [path.read_text(encoding="utf-8") for path in hook_root.rglob("*.sealed")]
    assert spool
    _assert_free_of_revision(spool)
