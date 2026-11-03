"""The refusal order of the executor tools, pinned for both of them at once.

``execute`` and ``execute_file`` run one gate sequence — the posture clamp
(the write verdict for the run's target), ``quick_safety_check``, path policy,
the readonly import denylist, pattern detection, the readonly pattern refusal,
then the launch, the write-activity report and the runtime-refusal report. When one call
trips several gates, the agent is told about the first one only, so the order
is part of what each tool says. These cases stage exactly two gates at a time
and assert which one answers, for both tools, so the two can never drift apart
and the shared :func:`run_gated_execution` cannot reorder them.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from osprey.audit import posture
from osprey.mcp_server.python_executor.executor import ExecutionResult
from osprey_connectors.refusals import refusal_phrase, refusal_text
from osprey_connectors_tests._write_fakes import ARMED_SECTION, ARMED_TARGET, arm_switch
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

TOOLS = pytest.mark.parametrize("tool_name", ["execute", "execute_file"])

#: A literal write into the render zone: the path policy's canonical refusal.
PROTECTED_WRITE = "open('build/config.yml', 'w')\n"

#: Code ``quick_safety_check`` refuses.
UNSAFE = "eval('1 + 1')\n"

#: A control-system client import the readonly denylist refuses.
CLIENT_IMPORT = "import epics\n"

#: A write the pattern detector recognises.
CLIENT_WRITE = "epics.caput('SR:QF1:SP', 1.0)\n"

#: What each refusal says, matched on its own words.
SAYS_CLAMP = refusal_phrase("readonly_run")
SAYS_OFF = refusal_text("switch_off", "simulator")
SAYS_PATH = "location the deployment protects"
SAYS_IMPORT = "cannot be imported in readonly mode"
SAYS_PATTERN = "write patterns detected in readonly mode"

_SAFETY_MESSAGE = {
    "execute": "Code failed pre-execution safety checks.",
    "execute_file": "File failed pre-execution safety checks.",
}
_TOOL_MODULE = {
    "execute": "osprey.mcp_server.python_executor.tools.python_execute",
    "execute_file": "osprey.mcp_server.python_executor.tools.python_execute_file",
}


@pytest.fixture(autouse=True)
def _no_ambient_deployment(control_context_root, monkeypatch):
    """Answer the posture gates from a scratch deployment, never the machine's.

    The deployment's tier allows writes and its record is on the simulator with
    the switch on, so a readwrite run passes the clamp unless a test turns the
    switch off (:func:`_switched_off`) or makes the run read-only.
    """
    from osprey_connectors import posture_store

    monkeypatch.delenv("OSPREY_CONTROL_TARGET", raising=False)
    monkeypatch.setattr(posture, "control_system_section", lambda: dict(ARMED_SECTION))
    arm_switch(monkeypatch, control_context_root, ARMED_TARGET, stamp=False)
    yield control_context_root
    posture_store.invalidate_cache()


@pytest.fixture
def project_root(tmp_path, monkeypatch):
    """A throwaway project root, so ``execute_file``'s containment check passes."""
    import osprey.mcp_server.python_executor.executor as executor

    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(executor, "_resolve_project_root", lambda: root)
    monkeypatch.chdir(root)
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


def _switched_off() -> None:
    """Turn the record's switch off for the run's target: the clamp then refuses."""
    from osprey_connectors_tests._write_fakes import turn_switch

    turn_switch(ARMED_TARGET, "off")


async def _call(tool_name, root, code, execution_mode, *, missing_file=False):
    """Run *code* through one tool. ``execute_file`` reads it from a script."""
    if tool_name == "execute":
        from osprey.mcp_server.python_executor.tools.python_execute import execute

        return await get_tool_fn(execute)(
            code=code,
            description="order probe",
            execution_mode=execution_mode,
            save_output=False,
        )

    from osprey.mcp_server.python_executor.tools.python_execute_file import execute_file

    script = root / "probe.py"
    if not missing_file:
        script.write_text(code, encoding="utf-8")
    return await get_tool_fn(execute_file)(
        file_path=str(script),
        description="order probe",
        execution_mode=execution_mode,
        save_output=False,
    )


def _said(ctx) -> str:
    envelope = ctx["envelope"]
    return " ".join([envelope["error_message"], *envelope.get("suggestions", [])])


# ---------------------------------------------------------------------------
# Two gates staged at once: the earlier one answers
# ---------------------------------------------------------------------------


@TOOLS
async def test_posture_clamp_answers_before_the_safety_check(
    tool_name, project_root, launch, monkeypatch
):
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    with assert_raises_error(error_type="safety_error") as ctx:
        await _call(tool_name, project_root, UNSAFE + PROTECTED_WRITE, "readwrite")

    assert ctx["envelope"]["error_message"].startswith(SAYS_CLAMP)
    launch.assert_not_awaited()


async def test_execute_file_clamp_answers_before_the_file_is_read(
    project_root, launch, monkeypatch
):
    """The clamp runs ahead of path resolution: a sandboxed caller is told so."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    with assert_raises_error(error_type="safety_error") as ctx:
        await _call("execute_file", project_root, "", "readwrite", missing_file=True)

    assert ctx["envelope"]["error_message"].startswith(SAYS_CLAMP)
    launch.assert_not_awaited()


@TOOLS
@pytest.mark.parametrize("execution_mode", ["readonly", "readwrite"])
async def test_safety_check_answers_before_the_path_policy(
    tool_name, execution_mode, project_root, launch
):
    with assert_raises_error(error_type="safety_error") as ctx:
        await _call(tool_name, project_root, UNSAFE + PROTECTED_WRITE, execution_mode)

    assert ctx["envelope"]["error_message"] == _SAFETY_MESSAGE[tool_name]
    launch.assert_not_awaited()


@TOOLS
async def test_path_policy_answers_before_the_import_denylist(tool_name, project_root, launch):
    with assert_raises_error(error_type="safety_error") as ctx:
        await _call(tool_name, project_root, CLIENT_IMPORT + PROTECTED_WRITE, "readonly")

    assert SAYS_PATH in ctx["envelope"]["error_message"]
    launch.assert_not_awaited()


@TOOLS
async def test_a_switched_off_target_answers_before_the_path_policy(
    tool_name, project_root, launch
):
    _switched_off()
    with assert_raises_error(error_type="safety_error") as ctx:
        await _call(tool_name, project_root, PROTECTED_WRITE, "readwrite")

    assert ctx["envelope"]["error_message"] == SAYS_OFF
    launch.assert_not_awaited()


@TOOLS
async def test_import_denylist_answers_before_the_pattern_refusal(tool_name, project_root, launch):
    with assert_raises_error(error_type="safety_error") as ctx:
        await _call(tool_name, project_root, CLIENT_IMPORT + CLIENT_WRITE, "readonly")

    assert SAYS_IMPORT in ctx["envelope"]["error_message"]
    launch.assert_not_awaited()


@TOOLS
async def test_pattern_refusal_answers_before_the_launch(tool_name, project_root, launch):
    code = "from osprey.runtime import write_channel\nwrite_channel('SR:QF1:SP', 1.0)\n"
    with assert_raises_error(error_type="safety_error") as ctx:
        await _call(tool_name, project_root, code, "readonly")

    assert SAYS_PATTERN in ctx["envelope"]["error_message"]
    launch.assert_not_awaited()


@TOOLS
async def test_a_switched_off_target_answers_before_the_launch(tool_name, project_root, launch):
    _switched_off()
    with assert_raises_error(error_type="safety_error") as ctx:
        await _call(tool_name, project_root, "x = 1\n", "readwrite")

    assert ctx["envelope"]["error_message"] == SAYS_OFF
    assert "writes_enabled" not in _said(ctx)
    launch.assert_not_awaited()


# ---------------------------------------------------------------------------
# A clean run: every static gate, in order, then launch, report, runtime report
# ---------------------------------------------------------------------------


@TOOLS
@pytest.mark.parametrize("execution_mode", ["readonly", "readwrite"])
async def test_a_clean_run_passes_every_gate_in_order(
    tool_name, execution_mode, project_root, monkeypatch
):
    """The static gates are consulted in the order the refusals above imply."""
    from osprey.mcp_server.python_executor.tools import _execution_gates as gates
    from osprey.services.python_executor.analysis import pattern_detection, safety_checks
    from osprey.services.python_executor.execution import path_policy
    from osprey.services.python_executor.execution.wrapper import READONLY_REFUSAL_MARKER

    order: list[str] = []

    def spy(name, fn):
        def wrapped(*args, **kwargs):
            order.append(name)
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(
        safety_checks, "quick_safety_check", spy("safety", safety_checks.quick_safety_check)
    )
    monkeypatch.setattr(
        path_policy, "path_policy_issues", spy("path_policy", path_policy.path_policy_issues)
    )
    monkeypatch.setattr(
        safety_checks,
        "check_readonly_imports",
        spy("import_denylist", safety_checks.check_readonly_imports),
    )
    monkeypatch.setattr(
        pattern_detection,
        "detect_control_system_operations",
        spy("patterns", pattern_detection.detect_control_system_operations),
    )

    stderr = f"PermissionError: {READONLY_REFUSAL_MARKER} write refused\n"

    async def fake_launch(**kwargs):
        order.append("launch")
        # A run that finished, carrying the guard's refusal on stderr.
        return ExecutionResult(
            success=True,
            stdout="",
            stderr=stderr,
            figures=[],
            execution_method_used="subprocess",
            execution_time_seconds=0.1,
        )

    async def notify(*args, **kwargs):
        order.append("notify")

    async def record(**kwargs):
        order.append("runtime_refusal")

    with (
        patch("osprey.mcp_server.python_executor.executor.execute_code", fake_launch),
        patch(f"{_TOOL_MODULE[tool_name]}.notify_agent_activity_async", notify),
        patch.object(gates, "record_and_alert_refusal", record),
    ):
        await _call(tool_name, project_root, "x = 1\n", execution_mode)

    static = ["safety", "path_policy"]
    if execution_mode == "readonly":
        static.append("import_denylist")
    static.append("patterns")
    tail = ["launch", "notify", "runtime_refusal"]
    if execution_mode == "readonly":
        # A readonly run with no detected write reports no write activity.
        tail.remove("notify")
    assert order == static + tail


# ---------------------------------------------------------------------------
# The helper itself
# ---------------------------------------------------------------------------


async def test_helper_gates_the_recorded_code_and_launches_the_given_one(project_root):
    """``record_code`` is what the gates see; ``code`` is what the backend runs."""
    from osprey.mcp_server.python_executor.tools._execution_gates import run_gated_execution

    launched = ExecutionResult(
        success=True,
        stdout="",
        stderr="",
        figures=[],
        execution_method_used="subprocess",
        execution_time_seconds=0.1,
    )
    mock = AsyncMock(return_value=launched)
    preamble = "import sys\nsys.argv = ['probe.py']\n"
    with patch("osprey.mcp_server.python_executor.executor.execute_code", mock):
        exec_result, patterns = await run_gated_execution(
            tool="execute_file",
            code=preamble + "x = 1\n",
            record_code="x = 1\n",
            description="helper probe",
            execution_mode="readonly",
            project_root=project_root,
        )

    assert exec_result is launched
    assert patterns["has_writes"] is False
    assert mock.await_args.kwargs["code"] == preamble + "x = 1\n"
    assert mock.await_args.kwargs["execution_mode"] == "readonly"
    assert "timeout" not in mock.await_args.kwargs


@pytest.mark.usefixtures("project_root")
async def test_helper_refuses_an_unknown_mode_before_any_gate(launch):
    from osprey.mcp_server.python_executor.tools._execution_gates import run_gated_execution

    with assert_raises_error(error_type="validation_error"):
        await run_gated_execution(
            tool="execute",
            code=UNSAFE,
            description="helper probe",
            execution_mode="rw",
        )
    launch.assert_not_awaited()


@pytest.mark.usefixtures("project_root", "launch")
async def test_helper_reports_write_activity_through_its_default_notify():
    """A caller that passes no ``notify`` reports through the gates module's own."""
    from osprey.mcp_server.python_executor.tools import _execution_gates as gates

    notify = AsyncMock()
    with patch.object(gates, "notify_agent_activity_async", notify):
        await gates.run_gated_execution(
            tool="pyaml_measure",
            code="x = 1\n",
            description="helper probe",
            execution_mode="readwrite",
        )

    notify.assert_awaited_once()
    assert notify.await_args.args[:2] == ("pyaml_measure", "channel")
