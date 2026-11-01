"""The code review step of the executor tools, for ``execute`` and ``execute_file`` at once.

The reviewer is a stub that applies the real level table, so these cases pin what the gate does
with an outcome: the one value-free record, the notice, the error, and whether the code launches.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml

from osprey.audit import posture
from osprey.audit.call import call_scope
from osprey.audit.envelope import AuditEnvelope
from osprey.mcp_server.python_executor import executor
from osprey.mcp_server.python_executor.executor import ExecutionResult
from osprey.mcp_server.python_executor.tools import _execution_gates, pyaml_measure
from osprey.mcp_server.python_executor.tools._execution_gates import run_gated_execution
from osprey.services.code_review.contract import SURFACE_EXECUTE, ReviewFailure, ReviewVerdict
from osprey.services.code_review.review import AUDIT_REASON, agent_message, decide
from osprey_connectors_tests._write_fakes import ARMED_SECTION, ARMED_TARGET, arm_switch
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

TOOLS = pytest.mark.parametrize("tool_name", ["execute", "execute_file"])

#: Code no gate objects to: no write pattern, so the write-activity report stays out of it too.
CODE = "print('reviewed')\n"
DESCRIPTION = "review probe"

#: The policy that asks nobody, so the reviewed calls clear the admission pre-check.
NO_ASK = {"tools": {"execute": "skip", "execute_file": "skip", "pyaml_measure": "skip"}}

#: Marks a config with no ``code_review`` section at all.
NO_SECTION = object()

BLOCK = ReviewVerdict("block", ("the stub objects",), "print('revised')\n")
CONCERN = ReviewVerdict("concern", ("the stub worries",))
PASS = ReviewVerdict("pass", ())
FAILURE = ReviewFailure("timeout", "no reply within 60 seconds")

#: result, level, refused error type (``None`` launches), record detail, decision, notice.
LEVEL_TABLE = [
    pytest.param(
        BLOCK,
        "block",
        "review_blocked",
        "review=block level=block mode=readwrite",
        "refused",
        "block, refused",
        id="block-block",
    ),
    pytest.param(
        BLOCK,
        "flag",
        None,
        "review=block level=flag mode=readwrite",
        "allowed",
        "block, proceeded",
        id="block-flag",
    ),
    pytest.param(
        CONCERN,
        "block",
        None,
        "review=concern level=block mode=readwrite",
        "allowed",
        "concern, proceeded",
        id="concern-block",
    ),
    pytest.param(
        CONCERN,
        "flag",
        None,
        "review=concern level=flag mode=readwrite",
        "allowed",
        "concern, proceeded",
        id="concern-flag",
    ),
    pytest.param(
        PASS,
        "block",
        None,
        "review=pass level=block mode=readwrite",
        "allowed",
        None,
        id="pass-block",
    ),
    pytest.param(
        PASS, "flag", None, "review=pass level=flag mode=readwrite", "allowed", None, id="pass-flag"
    ),
    pytest.param(
        FAILURE,
        "block",
        "review_unavailable",
        "review=failure level=block kind=timeout mode=readwrite",
        "refused",
        "failure (timeout), refused",
        id="failure-block",
    ),
    pytest.param(
        FAILURE,
        "flag",
        None,
        "review=failure level=flag kind=timeout mode=readwrite",
        "allowed",
        "failure (timeout), proceeded",
        id="failure-flag",
    ),
]


class _Tally:
    """How many reviews this module expected and got, across its tests."""

    expected = 0
    called = 0


@pytest.fixture(autouse=True, scope="module")
def _a_silent_decline_fails_loudly():
    """A broken fixture would decline every call and pass the never-reviewed cases silently."""
    yield
    assert _Tally.called or not _Tally.expected, "the review stub was never called"


class ReviewStub:
    """Stands in for ``review_code``: records its inputs, then applies the real level table."""

    def __init__(self) -> None:
        self.result: ReviewVerdict | ReviewFailure = BLOCK
        self.calls: list[dict[str, Any]] = []

    def stage(self, result: ReviewVerdict | ReviewFailure) -> None:
        self.result = result
        _Tally.expected += 1

    async def __call__(self, request, settings, *, repo_root, render_dir):
        _Tally.called += 1
        self.calls.append(
            {
                "request": request,
                "settings": settings,
                "repo_root": repo_root,
                "render_dir": render_dir,
            }
        )
        return decide(self.result, settings.level)


@pytest.fixture(autouse=True)
def stub():
    review = ReviewStub()
    with patch("osprey.services.code_review.review.review_code", review):
        yield review


@pytest.fixture(autouse=True)
def _no_ambient_deployment(control_context_root, monkeypatch):
    """A scratch deployment whose record is on the simulator with the switch on."""
    from osprey_connectors import posture_store

    monkeypatch.delenv("OSPREY_CONTROL_TARGET", raising=False)
    monkeypatch.delenv(_execution_gates.TOOL_PREFIX_ENV_VAR, raising=False)
    monkeypatch.setattr(posture, "control_system_section", lambda: dict(ARMED_SECTION))
    arm_switch(monkeypatch, control_context_root, ARMED_TARGET, stamp=False)
    yield control_context_root
    posture_store.invalidate_cache()


@pytest.fixture(autouse=True)
def _a_recorded_call(request, monkeypatch):
    """Each call carries the record a prompt, or the switch at ``allowed``, leaves for it.

    ``execute`` and ``execute_file`` are governed, so no policy spares them the record, and a
    call without one is declined before the review. These cases are about the review;
    ``TestAdmissionRefusal`` keeps the real check.
    """
    if request.cls is not TestAdmissionRefusal:
        monkeypatch.setattr(executor, "_check_approved_call", lambda *args, **kwargs: None)


@pytest.fixture
def project_root(tmp_path, monkeypatch):
    """A throwaway render: the project root, holding the config the gate reads."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(executor, "_resolve_project_root", lambda: root)
    monkeypatch.chdir(root)
    monkeypatch.setenv("OSPREY_CONFIG", str(root / "config.yml"))
    return root


@pytest.fixture
def launch():
    """The subprocess backend, replaced; a refusal must leave it unawaited."""
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
def notices(monkeypatch):
    notify = AsyncMock()
    monkeypatch.setattr(_execution_gates, "notify_agent_activity_async", notify)
    return notify


@pytest.fixture
def records(monkeypatch):
    """Every record the gate files, as ``(writer used, fields)``."""
    filed: list[tuple[str, dict[str, Any]]] = []

    def mark(**fields):
        filed.append(("record_and_mark", fields))
        return "entry"

    def write(**fields):
        filed.append(("record", fields))
        return "entry"

    monkeypatch.setattr("osprey.audit.dedup.record_and_mark", mark)
    monkeypatch.setattr("osprey.audit.writer.record", write)
    return filed


def _configure(root: Path, section: Any = NO_SECTION, *, approval: Any = NO_ASK) -> None:
    config: dict[str, Any] = {} if approval is None else {"approval": approval}
    if section is not NO_SECTION:
        config["code_review"] = section
    (root / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")


async def _call(tool_name, root, execution_mode="readwrite", *, script_args=None):
    if tool_name == "execute":
        from osprey.mcp_server.python_executor.tools.python_execute import execute

        return await get_tool_fn(execute)(
            code=CODE, description=DESCRIPTION, execution_mode=execution_mode, save_output=False
        )

    from osprey.mcp_server.python_executor.tools.python_execute_file import execute_file

    script = root / "probe.py"
    script.write_text(CODE, encoding="utf-8")
    return await get_tool_fn(execute_file)(
        file_path=str(script),
        description=DESCRIPTION,
        execution_mode=execution_mode,
        save_output=False,
        script_args=script_args,
    )


def _review_records(filed):
    return [entry for entry in filed if entry[1].get("reason") == AUDIT_REASON]


def _review_notices(notify):
    return [call for call in notify.await_args_list if call.args[1] == "review"]


def _result(launched: AsyncMock) -> ExecutionResult:
    return launched.return_value


# ---------------------------------------------------------------------------
# The level table
# ---------------------------------------------------------------------------


@TOOLS
@pytest.mark.parametrize(
    ("result", "level", "error_type", "detail", "decision", "taken"), LEVEL_TABLE
)
async def test_each_outcome_at_each_level(
    tool_name,
    result,
    level,
    error_type,
    detail,
    decision,
    taken,
    project_root,
    launch,
    notices,
    records,
    stub,
):
    _configure(project_root, {"enabled": True, "level": level})
    stub.stage(result)

    if error_type is None:
        await _call(tool_name, project_root)
        launch.assert_awaited_once()
        message = agent_message(decide(result, level))
        assert _result(launch).code_review == message
    else:
        with assert_raises_error(error_type=error_type):
            await _call(tool_name, project_root)
        launch.assert_not_awaited()

    assert len(stub.calls) == 1

    assert len(_review_records(records)) == 1
    writer, fields = _review_records(records)[0]
    assert fields["detail"] == detail
    assert fields["decision"] == decision
    assert fields["surface"] == "executor"
    assert fields["subject"] == tool_name
    assert writer == ("record_and_mark" if decision == "refused" else "record")
    # Value-free, and a record the envelope accepts: the writer swallows one it does not.
    assert "source" not in fields
    assert "stub" not in str(fields)
    assert "reviewed" not in str(fields)
    AuditEnvelope(actor="someone", **fields)

    sent = _review_notices(notices)
    if taken is None:
        assert sent == []
    else:
        assert len(sent) == 1
        assert sent[0].args == (tool_name, "review")
        assert sent[0].kwargs == {
            "detail": f"Code review on {SURFACE_EXECUTE}: {taken}: {tool_name}"
        }


@TOOLS
@pytest.mark.usefixtures("launch")
async def test_a_block_carries_the_verdict_to_the_agent(tool_name, project_root, stub):
    _configure(project_root, {"enabled": True, "level": "block"})
    stub.stage(BLOCK)

    with assert_raises_error(error_type="review_blocked") as ctx:
        await _call(tool_name, project_root)

    envelope = ctx["envelope"]
    assert envelope["error_message"] == agent_message(decide(BLOCK, "block"))
    assert envelope["details"] == {
        "verdict": "block",
        "reasons": ["the stub objects"],
        "suggested_revision": "print('revised')\n",
    }


@TOOLS
@pytest.mark.usefixtures("launch")
async def test_a_failure_names_its_kind_to_the_agent(tool_name, project_root, stub):
    _configure(project_root, {"enabled": True, "level": "block"})
    stub.stage(FAILURE)

    with assert_raises_error(error_type="review_unavailable") as ctx:
        await _call(tool_name, project_root)

    assert ctx["envelope"]["error_message"] == agent_message(decide(FAILURE, "block"))
    assert ctx["envelope"]["details"] == {"kind": "timeout"}


async def test_a_notice_that_fails_changes_nothing(project_root, launch, notices, records, stub):
    _configure(project_root, {"enabled": True, "level": "flag"})
    stub.stage(CONCERN)
    notices.side_effect = RuntimeError("no web terminal")

    await _call("execute", project_root)

    launch.assert_awaited_once()
    assert len(_review_records(records)) == 1


@pytest.mark.usefixtures("launch")
async def test_the_record_is_filed_before_the_notice(project_root, notices, records, stub):
    _configure(project_root, {"enabled": True, "level": "block"})
    stub.stage(BLOCK)
    seen: list[int] = []
    notices.side_effect = lambda *_a, **_k: seen.append(len(_review_records(records)))

    with assert_raises_error(error_type="review_blocked"):
        await _call("execute", project_root)

    assert seen == [1]


# ---------------------------------------------------------------------------
# What the reviewer is handed
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("launch")
async def test_the_stub_receives_the_execute_call(project_root, stub):
    _configure(project_root, {"enabled": True})
    stub.stage(PASS)

    await _call("execute", project_root)

    call = stub.calls[0]
    request = call["request"]
    assert request.surface == SURFACE_EXECUTE
    assert request.code == CODE
    assert request.subject == "execute"
    assert request.description == DESCRIPTION
    assert request.mode == "readwrite"
    assert request.script_args == ()
    assert call["repo_root"] == project_root
    assert call["render_dir"] == project_root
    assert call["settings"].level == "block"


async def test_the_stub_reviews_the_file_and_its_arguments_not_the_preamble(
    project_root, launch, stub
):
    _configure(project_root, {"enabled": True})
    stub.stage(PASS)

    await _call("execute_file", project_root, script_args=["--turns", "3"])

    request = stub.calls[0]["request"]
    assert request.code == CODE
    assert request.subject == "execute_file"
    assert request.script_args == ("--turns", "3")
    # The launch keeps the preamble the review never saw.
    assert "sys.argv" in launch.await_args.kwargs["code"]


async def test_the_project_root_a_tool_resolved_anchors_the_review(tmp_path, project_root, stub):
    _configure(project_root, {"enabled": True})
    stub.stage(PASS)
    other = tmp_path / "elsewhere"

    with patch("osprey.mcp_server.python_executor.executor.execute_code", AsyncMock()) as launched:
        launched.return_value = ExecutionResult(success=True, stdout="", stderr="")
        await run_gated_execution(
            tool="execute",
            code=CODE,
            description=DESCRIPTION,
            execution_mode="readwrite",
            project_root=other,
        )

    assert stub.calls[0]["repo_root"] == other


# ---------------------------------------------------------------------------
# Off, and which modes are reviewed
# ---------------------------------------------------------------------------


@TOOLS
@pytest.mark.parametrize(
    "section",
    [
        pytest.param({"enabled": False}, id="disabled"),
        pytest.param(NO_SECTION, id="no-section"),
        # Off at runtime with nothing further validated; the build is what refuses this one.
        pytest.param({"enabled": False, "level": "warn"}, id="disabled-malformed"),
    ],
)
async def test_a_feature_that_is_off_never_reviews(
    tool_name, section, project_root, launch, notices, records, stub
):
    _configure(project_root, section)

    await _call(tool_name, project_root)

    assert stub.calls == []
    assert _review_records(records) == []
    assert _review_notices(notices) == []
    launch.assert_awaited_once()
    assert _result(launch).code_review is None


@TOOLS
async def test_no_config_file_never_reviews(tool_name, project_root, launch, records, stub):
    assert not (project_root / "config.yml").exists()

    await _call(tool_name, project_root)

    assert stub.calls == []
    assert _review_records(records) == []
    launch.assert_awaited_once()


@TOOLS
async def test_by_default_a_readonly_call_is_not_reviewed(
    tool_name, project_root, launch, records, stub
):
    _configure(project_root, {"enabled": True})

    await _call(tool_name, project_root, "readonly")

    assert stub.calls == []
    assert _review_records(records) == []
    launch.assert_awaited_once()


@TOOLS
async def test_a_readonly_call_is_reviewed_when_modes_name_it(
    tool_name, project_root, launch, records, stub
):
    _configure(project_root, {"enabled": True, "modes": ["readonly", "readwrite"]})
    stub.stage(PASS)

    await _call(tool_name, project_root, "readonly")

    assert stub.calls[0]["request"].mode == "readonly"
    assert _review_records(records)[0][1]["detail"] == "review=pass level=block mode=readonly"
    launch.assert_awaited_once()


# ---------------------------------------------------------------------------
# pyaml_measure opts out
# ---------------------------------------------------------------------------


async def _run_pyaml_gate(launch) -> None:
    await run_gated_execution(
        tool="pyaml_measure",
        code=CODE,
        description=DESCRIPTION,
        execution_mode="readwrite",
        code_review=False,
    )
    launch.assert_awaited_once()


async def test_pyaml_measure_runs_under_a_blocking_reviewer(project_root, launch, records, stub):
    _configure(project_root, {"enabled": True, "level": "block"})

    await _run_pyaml_gate(launch)

    assert stub.calls == []
    assert _review_records(records) == []


@pytest.mark.parametrize(
    "written",
    [
        pytest.param("code_review: {enabled: true, level: warn}\n", id="malformed-section"),
        pytest.param("code_review: [unclosed\n", id="garbage-yaml"),
    ],
)
async def test_pyaml_measure_never_reads_the_section(written, project_root, launch, records, stub):
    (project_root / "config.yml").write_text(written, encoding="utf-8")

    with patch("osprey.services.code_review.settings.load_review_settings", Mock()) as load:
        await _run_pyaml_gate(launch)

    load.assert_not_called()
    assert stub.calls == []
    assert _review_records(records) == []


def test_the_pyaml_measure_tool_opts_out():
    tree = ast.parse(Path(pyaml_measure.__file__).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "run_gated_execution"
    ]
    assert len(calls) == 1
    opted = {kw.arg: kw.value for kw in calls[0].keywords}["code_review"]
    assert isinstance(opted, ast.Constant) and opted.value is False


# ---------------------------------------------------------------------------
# A section that cannot be read refuses, whatever the mode
# ---------------------------------------------------------------------------


@TOOLS
@pytest.mark.parametrize("execution_mode", ["readonly", "readwrite"])
@pytest.mark.parametrize(
    "written",
    [
        pytest.param("code_review: {enabled: true, level: warn}\n", id="bad-level"),
        pytest.param("code_review: {enabled: 'yes'}\n", id="non-bool-enabled"),
        pytest.param("code_review: {enabled: 1}\n", id="numeric-enabled"),
        pytest.param("code_review: [not, a, mapping]\n", id="non-mapping-section"),
        pytest.param("code_review: [unclosed\n", id="garbage-yaml"),
        pytest.param("- not\n- a mapping\n", id="non-mapping-file"),
    ],
)
async def test_a_config_failure_refuses_without_a_level(
    tool_name, execution_mode, written, project_root, launch, notices, records, stub
):
    (project_root / "config.yml").write_text(written, encoding="utf-8")

    with assert_raises_error(error_type="review_unavailable") as ctx:
        await _call(tool_name, project_root, execution_mode)

    assert ctx["envelope"]["details"] == {"kind": "config"}
    launch.assert_not_awaited()
    assert stub.calls == []

    assert len(_review_records(records)) == 1
    writer, fields = _review_records(records)[0]
    assert writer == "record_and_mark"
    assert fields["detail"] == "review=failure kind=config"
    assert fields["decision"] == "refused"
    assert fields["subject"] == tool_name
    AuditEnvelope(actor="someone", **fields)
    assert len(_review_notices(notices)) == 1


# ---------------------------------------------------------------------------
# A run the executor declines is never reviewed
# ---------------------------------------------------------------------------


async def _gate(tool_name: str) -> ExecutionResult:
    result, _patterns = await run_gated_execution(
        tool=tool_name, code=CODE, description=DESCRIPTION, execution_mode="readwrite"
    )
    return result


async def _declined_directly(tool_name: str) -> ExecutionResult:
    return await executor.execute_code(
        code=CODE, execution_mode="readwrite", description=DESCRIPTION, tool=tool_name
    )


@TOOLS
async def test_a_run_missing_its_approval_is_declined_unreviewed(
    tool_name, project_root, notices, records, stub
):
    _configure(project_root, {"enabled": True, "level": "block"}, approval=None)

    result = await _gate(tool_name)

    assert result.failure_kind == "approval_missing"
    assert result == await _declined_directly(tool_name)
    assert stub.calls == []
    assert _review_records(records) == []
    assert _review_notices(notices) == []


@TOOLS
async def test_a_run_mid_switch_is_declined_unreviewed(
    tool_name, project_root, monkeypatch, notices, records, stub
):
    _configure(project_root, {"enabled": True, "level": "block"})
    monkeypatch.setattr(executor, "_blocking_pids", lambda _record: (4242,))

    result = await _gate(tool_name)

    assert result.failure_kind == "switch_in_progress"
    assert result == await _declined_directly(tool_name)
    assert stub.calls == []
    assert _review_records(records) == []
    assert _review_notices(notices) == []


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {str(p): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _admit(tool: str = "execute", code: str = CODE, mode: str = "readwrite", **approved) -> bool:
    return executor.admission_refusal(
        code,
        mode,
        tool,
        {"approved_target": None, "approved_view_sha256": None, **approved},
    )


class TestAdmissionRefusal:
    """Agrees with the decline ``execute_code`` would produce, and touches nothing."""

    def test_a_skip_policy_does_not_spare_a_governed_call_its_record(self, project_root):
        _configure(project_root)
        assert _admit() is True

    def test_a_readonly_call_is_admitted_with_no_stamp(self, project_root):
        _configure(project_root, approval=None)
        assert _admit(mode="readonly") is False

    def test_a_missing_stamp_declines(self, project_root):
        _configure(project_root, approval=None)
        assert _admit() is True

    def test_a_stamp_is_read_and_left_in_place(
        self, project_root, tmp_path, monkeypatch, control_context_root
    ):
        from osprey.mcp_server.control_system import target_state

        state = tmp_path / "state"
        state.mkdir()
        monkeypatch.setattr(target_state, "state_dir", lambda: state)
        _configure(project_root, approval=None)
        payload = {
            "tool_use_id": "toolu_admit",
            "tool": "mcp__python__execute",
            "approver": "human",
            "approved_target": ARMED_TARGET,
        }
        (state / "approval_ask_toolu_admit.json").write_text(json.dumps(payload), encoding="utf-8")
        before = _tree_bytes(state), _tree_bytes(control_context_root)

        with call_scope("toolu_admit", None):
            assert _admit(approved_target=ARMED_TARGET) is False

        assert (_tree_bytes(state), _tree_bytes(control_context_root)) == before

    def test_a_decline_leaves_the_control_context_alone(self, project_root, control_context_root):
        _configure(project_root, approval=None)
        before = _tree_bytes(control_context_root)

        assert _admit() is True

        assert _tree_bytes(control_context_root) == before

    def test_a_switch_in_progress_declines(self, project_root, monkeypatch):
        _configure(project_root)
        monkeypatch.setattr(executor, "_blocking_pids", lambda _record: (4242,))
        assert _admit() is True

    def test_store_naming_code_declines_in_a_readonly_call(self, project_root):
        from osprey_connectors.refusals import STORE_PATH_MARKERS

        _configure(project_root)
        code = f"open({next(iter(STORE_PATH_MARKERS))!r})\n"
        assert _admit(code=code, mode="readonly") is True

    def test_an_unexpected_error_admits_so_the_review_still_runs(self, project_root, monkeypatch):
        """Skipping on an unknown would run code unreviewed if ``execute_code`` then admitted it."""

        def boom(_env):
            raise RuntimeError("unreadable")

        _configure(project_root)
        monkeypatch.setattr(executor, "_apply_target_stamp", boom)
        assert _admit() is False

    def test_the_process_environment_is_untouched(self, project_root, monkeypatch):
        import os

        _configure(project_root)
        monkeypatch.setenv(executor.ENV_CONTROL_TARGET, "inherited")
        before = dict(os.environ)

        _admit()

        assert dict(os.environ) == before


# ---------------------------------------------------------------------------
# Edits reach the next call
# ---------------------------------------------------------------------------


async def test_a_level_switch_back_takes_effect_on_the_next_call(project_root, launch, stub):
    stub.stage(BLOCK)
    _configure(project_root, {"enabled": True, "level": "block"})
    with assert_raises_error(error_type="review_blocked"):
        await _call("execute", project_root)
    launch.assert_not_awaited()

    _configure(project_root, {"enabled": True, "level": "flag"})
    await _call("execute", project_root)
    launch.assert_awaited_once()


async def test_disabling_takes_effect_on_the_next_call(project_root, launch, stub):
    stub.stage(BLOCK)
    _configure(project_root, {"enabled": True, "level": "block"})
    with assert_raises_error(error_type="review_blocked"):
        await _call("execute", project_root)
    assert len(stub.calls) == 1

    _configure(project_root, {"enabled": False})
    await _call("execute", project_root)
    assert len(stub.calls) == 1
    launch.assert_awaited_once()


# ---------------------------------------------------------------------------
# A review the approval prompt already made
# ---------------------------------------------------------------------------
#
# The approval hook reviews the code before it asks and leaves the digest of what
# it read, and what the agent is told, in the ask stamp. The gate reads that stamp
# for the call in scope and does not read the same code twice; any other stamp
# leaves this gate as the backstop it is.


@pytest.fixture
def ask_stamps(tmp_path, monkeypatch):
    """The state directory the executor reads ask stamps from; returns a stamp writer."""
    from osprey.mcp_server.control_system import target_state

    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(target_state, "state_dir", lambda: state)

    def write(tool_use_id: str, **fields) -> None:
        payload = {"tool_use_id": tool_use_id, "approver": "human", **fields}
        (state / f"approval_ask_{tool_use_id}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    return write


def _stamped(code: str, result: ReviewVerdict, level: str = "block") -> dict[str, Any]:
    from osprey.services.code_review.stamp import reviewed_fields

    return reviewed_fields(code, decide(result, level))


@TOOLS
async def test_a_review_the_prompt_stamped_for_this_code_is_not_run_again(
    tool_name, project_root, launch, notices, records, stub, ask_stamps
):
    _configure(project_root, {"enabled": True, "level": "block"})
    ask_stamps("toolu_stamped", tool=f"mcp__python__{tool_name}", **_stamped(CODE, CONCERN))

    with call_scope("toolu_stamped", None):
        await _call(tool_name, project_root)

    assert stub.calls == [], "the prompt's review answers for this code"
    launch.assert_awaited_once()
    assert _result(launch).code_review == agent_message(decide(CONCERN, "block"))
    assert _review_records(records) == [], "the hook filed the one record"
    assert _review_notices(notices) == [], "the hook sent the one notice"


async def test_a_silent_pass_the_prompt_stamped_has_nothing_to_add(
    project_root, launch, stub, ask_stamps
):
    _configure(project_root, {"enabled": True, "level": "block"})
    ask_stamps("toolu_pass", tool="mcp__python__execute", **_stamped(CODE, PASS))

    with call_scope("toolu_pass", None):
        await _call("execute", project_root)

    assert stub.calls == []
    launch.assert_awaited_once()
    assert _result(launch).code_review is None


async def test_a_stamp_for_other_code_leaves_the_review_in_place(
    project_root, launch, stub, ask_stamps
):
    stub.stage(PASS)
    _configure(project_root, {"enabled": True, "level": "block"})
    ask_stamps("toolu_other", tool="mcp__python__execute", **_stamped(CODE + "\n", CONCERN))

    with call_scope("toolu_other", None):
        await _call("execute", project_root)

    assert len(stub.calls) == 1
    launch.assert_awaited_once()
    assert _result(launch).code_review is None


async def test_a_stamp_for_another_tool_leaves_the_review_in_place(
    project_root, launch, stub, ask_stamps
):
    stub.stage(PASS)
    _configure(project_root, {"enabled": True, "level": "block"})
    ask_stamps("toolu_file", tool="mcp__python__execute_file", **_stamped(CODE, CONCERN))

    with call_scope("toolu_file", None):
        await _call("execute", project_root)

    assert len(stub.calls) == 1
    launch.assert_awaited_once()


async def test_a_stamp_the_prompt_left_unreviewed_leaves_the_review_in_place(
    project_root, launch, stub, ask_stamps
):
    """A stamp with no review fields is an ask the prompt could not review."""
    stub.stage(BLOCK)
    _configure(project_root, {"enabled": True, "level": "block"})
    ask_stamps("toolu_bare", tool="mcp__python__execute")

    with call_scope("toolu_bare", None), assert_raises_error(error_type="review_blocked"):
        await _call("execute", project_root)

    assert len(stub.calls) == 1
    launch.assert_not_awaited()
