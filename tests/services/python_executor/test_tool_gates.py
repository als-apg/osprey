"""Tool-layer pins for the static path policy on ``execute`` / ``execute_file``.

The unit suite (``test_path_policy.py``) pins what the walker *sees*. This one
pins that both MCP tools actually consult it, that they do so in **every**
execution mode rather than under the readonly branch, and that a refusal leaves
the same audit record the other pre-execution layers leave — only with
``reason: "path_policy"``, on the unified ledger's ``executor`` surface.

The refusals here all fire before the subprocess is launched, so nothing is
mocked for them: reaching an assertion at all proves the gate ran ahead of
execution.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.unit

#: A literal write into the render zone — the canonical thing the policy exists
#: to refuse. Relative, so it matches whichever repo root the tool resolves.
RENDER_ZONE_WRITE = "open('build/config.yml', 'w')"

#: Code that names an internal of the emitted sandbox guard.
GUARD_TAMPER = "_restore_patched_targets()\n"

BOTH_MODES = pytest.mark.parametrize("execution_mode", ["readonly", "readwrite"])


def _execute():
    from osprey.mcp_server.python_executor.tools.python_execute import execute

    return get_tool_fn(execute)


def _execute_file():
    from osprey.mcp_server.python_executor.tools.python_execute_file import execute_file

    return get_tool_fn(execute_file)


@pytest.fixture(autouse=True)
def audit_zone(tmp_path, monkeypatch):
    """Redirect the audit zone; ``writer.audit_dir`` is the one seam it documents.

    Autouse, so a test that records without asking for the zone cannot append
    to the deployment's real ledger: the posture-clamp cases below refuse
    *before* they read any code, and a refusal an operator finds in
    ``var/audit/`` that no operator caused is worse than no record at all.
    """
    from osprey.audit import writer

    zone = tmp_path / "audit"
    monkeypatch.setattr(writer, "audit_dir", lambda: zone)
    return zone


@pytest.fixture
def script_root(tmp_path, monkeypatch):
    """Anchor ``execute_file`` on a throwaway repo root.

    The tool resolves the root once and hands it to both root resolvers, so
    patching it here keeps the protected set — and the containment check the
    file tool runs — inside ``tmp_path``.
    """
    import osprey.mcp_server.python_executor.executor as executor

    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(executor, "_resolve_project_root", lambda: root)
    return root


def _script(root, code):
    """Write *code* to a .py file inside the resolved project root."""
    target = root / "probe.py"
    target.write_text(code, encoding="utf-8")
    return target


def _refusal_records(audit_zone):
    """Every executor record this identity filed — the ledger is per-identity."""
    from osprey.audit.envelope import SURFACE_EXECUTOR
    from osprey.utils.identity import acting_identity

    log = audit_zone / acting_identity() / f"{SURFACE_EXECUTOR}.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# The gate is mode-independent
# ---------------------------------------------------------------------------


@BOTH_MODES
async def test_execute_refuses_render_zone_write_in_every_mode(execution_mode, audit_zone):
    """readwrite is not a way around the policy — both modes refuse."""
    with assert_raises_error(error_type="safety_error") as ctx:
        await _execute()(
            code=RENDER_ZONE_WRITE,
            description="write the rendered config",
            execution_mode=execution_mode,
        )

    envelope = ctx["envelope"]
    assert "every execution mode" in envelope["error_message"]
    assert any("build/config.yml" in s for s in envelope["suggestions"])


@BOTH_MODES
async def test_readwrite_wording_does_not_blame_readonly_mode(execution_mode, audit_zone):
    """The refusal must not send a readwrite caller back to try readwrite."""
    with assert_raises_error(error_type="safety_error") as ctx:
        await _execute()(
            code=RENDER_ZONE_WRITE,
            description="write the rendered config",
            execution_mode=execution_mode,
        )

    text = " ".join([ctx["envelope"]["error_message"], *ctx["envelope"]["suggestions"]])
    if execution_mode == "readwrite":
        assert "readonly" not in text.lower()
    else:
        assert "readwrite will not lift this" in text


@BOTH_MODES
async def test_execute_file_matches_execute(execution_mode, script_root, audit_zone):
    """The same code through the file tool gives the same refusal."""
    script = _script(script_root, RENDER_ZONE_WRITE + "\n")
    with assert_raises_error(error_type="safety_error") as file_ctx:
        await _execute_file()(
            file_path=str(script),
            description="write the rendered config",
            execution_mode=execution_mode,
        )
    with assert_raises_error(error_type="safety_error") as code_ctx:
        await _execute()(
            code=RENDER_ZONE_WRITE + "\n",
            description="write the rendered config",
            execution_mode=execution_mode,
        )

    assert file_ctx["envelope"]["error_message"] == code_ctx["envelope"]["error_message"]
    assert file_ctx["envelope"]["suggestions"] == code_ctx["envelope"]["suggestions"]


# ---------------------------------------------------------------------------
# The refusal is auditable
# ---------------------------------------------------------------------------


@BOTH_MODES
async def test_refusal_is_recorded_with_the_path_policy_layer(execution_mode, audit_zone):
    from osprey.mcp_server.python_executor.tools._execution_gates import LAYER_PATH_POLICY

    with assert_raises_error(error_type="safety_error"):
        await _execute()(
            code=RENDER_ZONE_WRITE,
            description="write the rendered config",
            execution_mode=execution_mode,
        )

    records = _refusal_records(audit_zone)
    assert len(records) == 1
    assert records[0]["reason"] == LAYER_PATH_POLICY == "path_policy"
    assert records[0]["subject"] == "execute"
    assert records[0]["source"] == RENDER_ZONE_WRITE
    # The layer, the mode and what matched all ride in the record's detail.
    assert "tool=execute" in records[0]["detail"]
    assert f"mode={execution_mode}" in records[0]["detail"]
    assert "build/config.yml" in records[0]["detail"]


async def test_execute_file_refusal_is_recorded(script_root, audit_zone):
    script = _script(script_root, RENDER_ZONE_WRITE + "\n")
    with assert_raises_error(error_type="safety_error"):
        await _execute_file()(
            file_path=str(script),
            description="write the rendered config",
            execution_mode="readonly",
        )

    records = _refusal_records(audit_zone)
    assert len(records) == 1
    assert records[0]["reason"] == "path_policy"
    assert records[0]["subject"] == "execute_file"


# ---------------------------------------------------------------------------
# Guard tampering, at the tool layer
# ---------------------------------------------------------------------------


@BOTH_MODES
async def test_guard_tamper_is_refused_at_the_tool_layer(execution_mode, audit_zone):
    with assert_raises_error(error_type="safety_error") as ctx:
        await _execute()(
            code=GUARD_TAMPER,
            description="unpatch the guard",
            execution_mode=execution_mode,
        )

    assert any("_restore_patched_targets" in s for s in ctx["envelope"]["suggestions"])
    assert _refusal_records(audit_zone)[0]["reason"] == "path_policy"


# ---------------------------------------------------------------------------
# The gate stays out of the way of ordinary work
# ---------------------------------------------------------------------------


async def test_clean_read_still_executes(tmp_path, monkeypatch, audit_zone):
    """A read is not a write — the gate must not intercept ordinary analysis."""
    monkeypatch.chdir(tmp_path)

    from osprey.mcp_server.python_executor.executor import ExecutionResult

    mock_exec = AsyncMock(
        return_value=ExecutionResult(
            success=True,
            stdout="read ok\n",
            stderr="",
            execution_method_used="subprocess",
            execution_time_seconds=0.01,
        )
    )
    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection"
            ".detect_control_system_operations",
            return_value={"has_writes": False, "has_reads": False, "detected_patterns": {}},
        ),
        patch("osprey.mcp_server.python_executor.executor.execute_code", mock_exec),
    ):
        result = await _execute()(
            code="data = open('build/config.yml').read()\nprint('read ok')",
            description="read the rendered config",
            execution_mode="readonly",
            save_output=False,
        )

    data = extract_response_dict(result)
    assert data.get("error") is not True
    assert data["has_errors"] is False
    assert "read ok" in data["stdout"]
    assert _refusal_records(audit_zone) == []


# ---------------------------------------------------------------------------
# Session posture clamp
#
# The gate itself is unit-tested in ``test_posture_clamp.py``. What is pinned
# here is that both tools actually *call* it — the clamp existed and was
# correct for a while without being wired into either tool, which left the
# executor gate that FR6 names as enforced doing nothing at all.
# ---------------------------------------------------------------------------

CLEAN_READWRITE_CODE = "print('would write')\n"


@pytest.fixture
def sandbox_posture(monkeypatch):
    """Put the session in the sandbox posture, as the Web Terminal does."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")


def _assert_posture_envelope(envelope):
    """The refusal must name the posture, and must not blame the deployment.

    Sending an operator to ``control_system.writes_enabled`` when the real
    gate is the session's posture costs them a config edit that changes
    nothing.
    """
    message = envelope["error_message"]
    assert "posture" in message.lower()
    assert "writes_enabled" not in message
    assert not any("writes_enabled" in s for s in envelope["suggestions"])


async def test_execute_readwrite_is_refused_under_the_sandbox_posture(sandbox_posture):
    with assert_raises_error(error_type="safety_error") as ctx:
        await _execute()(
            code=CLEAN_READWRITE_CODE,
            description="write a channel",
            execution_mode="readwrite",
        )

    _assert_posture_envelope(ctx["envelope"])


async def test_execute_file_readwrite_is_refused_under_the_sandbox_posture(
    sandbox_posture, script_root
):
    script = _script(script_root, CLEAN_READWRITE_CODE)
    with assert_raises_error(error_type="safety_error") as file_ctx:
        await _execute_file()(
            file_path=str(script),
            description="write a channel",
            execution_mode="readwrite",
        )
    with assert_raises_error(error_type="safety_error") as code_ctx:
        await _execute()(
            code=CLEAN_READWRITE_CODE,
            description="write a channel",
            execution_mode="readwrite",
        )

    _assert_posture_envelope(file_ctx["envelope"])
    assert file_ctx["envelope"]["error_message"] == code_ctx["envelope"]["error_message"]
    assert file_ctx["envelope"]["suggestions"] == code_ctx["envelope"]["suggestions"]


async def _run_clean_readwrite(tmp_path, monkeypatch):
    """Run a harmless readwrite script with the subprocess mocked out.

    The deployment kill switch is neutralised (``None`` = the deployment has
    nothing to say about writes), because a tmp_path with no ``config.yml``
    falls back to safe defaults and would refuse the run at that *later* gate.
    These tests are about the posture clamp, so the gate behind it must not be
    what answers.
    """
    monkeypatch.chdir(tmp_path)

    from osprey.mcp_server.python_executor.executor import ExecutionResult

    mock_exec = AsyncMock(
        return_value=ExecutionResult(
            success=True,
            stdout="ran\n",
            stderr="",
            execution_method_used="subprocess",
            execution_time_seconds=0.01,
        )
    )
    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection"
            ".detect_control_system_operations",
            return_value={"has_writes": False, "has_reads": False, "detected_patterns": {}},
        ),
        patch(
            "osprey.services.python_executor.execution.control.get_execution_control_config",
            return_value=None,
        ),
        patch("osprey.mcp_server.python_executor.executor.execute_code", mock_exec),
    ):
        return await _execute()(
            code=CLEAN_READWRITE_CODE,
            description="harmless script",
            execution_mode="readwrite",
            save_output=False,
        )


async def test_readonly_still_runs_under_the_sandbox_posture(
    sandbox_posture, tmp_path, monkeypatch
):
    """The posture clamps writes, not reads — ordinary work must be unaffected."""
    monkeypatch.chdir(tmp_path)

    from osprey.mcp_server.python_executor.executor import ExecutionResult

    mock_exec = AsyncMock(
        return_value=ExecutionResult(
            success=True,
            stdout="ran\n",
            stderr="",
            execution_method_used="subprocess",
            execution_time_seconds=0.01,
        )
    )
    with (
        patch(
            "osprey.services.python_executor.analysis.pattern_detection"
            ".detect_control_system_operations",
            return_value={"has_writes": False, "has_reads": False, "detected_patterns": {}},
        ),
        patch("osprey.mcp_server.python_executor.executor.execute_code", mock_exec),
    ):
        result = await _execute()(
            code="print('ran')",
            description="a read",
            execution_mode="readonly",
            save_output=False,
        )

    data = extract_response_dict(result)
    assert data.get("error") is not True
    assert "ran" in data["stdout"]


async def test_readwrite_is_unchanged_with_no_posture_in_the_environment(tmp_path, monkeypatch):
    """No posture variable means no clamp — the gate must not fire on absence."""
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)

    data = extract_response_dict(await _run_clean_readwrite(tmp_path, monkeypatch))
    assert data.get("error") is not True
    assert data["execution_mode"] == "readwrite"


async def test_the_writes_posture_does_not_clamp(tmp_path, monkeypatch):
    """Value comparison, not presence: the writes posture sets the same variable."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readwrite")

    data = extract_response_dict(await _run_clean_readwrite(tmp_path, monkeypatch))
    assert data.get("error") is not True
    assert data["execution_mode"] == "readwrite"
