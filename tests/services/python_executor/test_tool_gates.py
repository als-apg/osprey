"""Tool-layer pins for the static path policy on ``execute`` / ``execute_file``.

The unit suite (``test_path_policy.py``) pins what the walker *sees*. This one
pins that both MCP tools actually consult it, that they do so in **every**
execution mode rather than under the readonly branch, and that a refusal leaves
the same audit record the other pre-execution layers leave — only with
``reason: "path_policy"``, as an ``executor`` audit entry in the agent record.

The refusals here all fire before the subprocess is launched, so nothing is
mocked for them: reaching an assertion at all proves the gate ran ahead of
execution.
"""

import re
from unittest.mock import AsyncMock, patch

import pytest

from osprey.audit import posture
from osprey_connectors.refusals import refusal_phrase, refusal_text
from osprey_connectors_tests._harness import REFUSAL_BAN_LIST, write_control_context
from osprey_connectors_tests._write_fakes import ARMED_SECTION, ARMED_TARGET, arm_switch
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

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
def _no_ambient_deployment(control_context_root, monkeypatch):
    """Answer every gate here from a scratch deployment, never from the machine's.

    ``enforce_posture_clamp`` runs ahead of every readwrite tool call and reads
    the control-context record. ``read_record()`` with no
    ``OSPREY_AGENT_DATA_ROOT`` falls back to ``<repo>/var/agent_data``, which on
    a developer box holds whatever their own running stack put there — so a
    narrowing an operator made by hand would refuse the tool calls below and
    read as a regression in the path policy. Autouse so a case added here later
    does not have to rediscover it.

    ``OSPREY_CONTROL_TARGET`` is cleared beside it because an inherited stamp
    names a target no test here wrote.

    The scratch deployment allows writes: its tier allows them and its record
    is on the simulator with the switch on, so a readwrite call reaches the
    gates behind the clamp unless a test says otherwise.
    """
    from osprey_connectors import posture_store

    monkeypatch.delenv("OSPREY_CONTROL_TARGET", raising=False)
    monkeypatch.setattr(posture, "control_system_section", lambda: dict(ARMED_SECTION))
    arm_switch(monkeypatch, control_context_root, ARMED_TARGET, stamp=False)
    yield control_context_root
    posture_store.invalidate_cache()


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


#: A whitespace-delimited token is a filesystem path when it opens -- after any
#: quoting -- on a path separator or a drive letter. The path-policy refusal
#: quotes the protected root it matched, and that root is an absolute path under
#: whatever directory the checkout happens to live in. On a checkout named
#: ``readonly-guard-hotfix`` the refusal text therefore contains "readonly"
#: while blaming nothing at all. Prose words carry no separator, so dropping
#: these tokens leaves every word the gate itself composed.
_ABSOLUTE_PATH_TOKEN = re.compile(r"^[\"'(\[]*(?:[A-Za-z]:)?[\\/]")


def _refusal_prose(envelope) -> str:
    """The refusal's own words, lowercased, with absolute paths dropped."""
    text = " ".join([envelope["error_message"], *envelope["suggestions"]])
    return " ".join(word for word in text.split() if not _ABSOLUTE_PATH_TOKEN.match(word)).lower()


def _refusal_records(recorded_entries):
    """Every ``executor`` audit entry in the record, oldest first."""
    from osprey.audit.envelope import SURFACE_EXECUTOR

    entries = sorted(recorded_entries("audit"), key=lambda e: e.ts)
    return [dict(e.body) for e in entries if e.body["surface"] == SURFACE_EXECUTOR]


# ---------------------------------------------------------------------------
# The gate is mode-independent
# ---------------------------------------------------------------------------


@BOTH_MODES
async def test_execute_refuses_render_zone_write_in_every_mode(execution_mode):
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
async def test_readwrite_wording_does_not_blame_readonly_mode(execution_mode):
    """The refusal must not send a readwrite caller back to try readwrite."""
    with assert_raises_error(error_type="safety_error") as ctx:
        await _execute()(
            code=RENDER_ZONE_WRITE,
            description="write the rendered config",
            execution_mode=execution_mode,
        )

    prose = _refusal_prose(ctx["envelope"])
    if execution_mode == "readwrite":
        assert "readonly" not in prose
    else:
        assert "readwrite will not lift this" in prose


class TestTheProseCheckIsBlindToTheCheckoutName:
    """``_refusal_prose`` must drop the quoted paths and nothing else.

    Both directions are pinned here because a helper that silently dropped too
    much would turn the test above into one that can no longer fail: the point
    of that test is that a readwrite caller is never told to try readwrite, and
    the only reason to strip anything is that the protected root is spelled
    with the checkout's own directory name in it.
    """

    def test_a_protected_root_under_a_readonly_named_checkout_is_not_blame(self):
        envelope = {
            "error_message": (
                "This code writes into a location the deployment protects. "
                "The path policy applies in every execution mode."
            ),
            "suggestions": [
                "Write to 'build/config.yml' via open() is not allowed: the path is "
                "inside the protected location "
                "'/home/dev/checkouts/readonly-guard-hotfix/build'",
                "Write analysis output under the agent data zone instead.",
                "The write posture permits control-system writes; it does not "
                "permit edits to OSPREY's own configuration.",
            ],
        }

        assert "readonly" not in _refusal_prose(envelope)

    def test_prose_that_blames_readonly_mode_is_still_caught(self):
        envelope = {
            "error_message": (
                "This code writes into a location the deployment protects. "
                "The path policy applies in every execution mode."
            ),
            "suggestions": ["Re-run this in readonly mode instead."],
        }

        assert "readonly" in _refusal_prose(envelope)


@BOTH_MODES
async def test_execute_file_matches_execute(execution_mode, script_root):
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
async def test_refusal_is_recorded_with_the_path_policy_layer(execution_mode, recorded_entries):
    from osprey.mcp_server.python_executor.tools._execution_gates import LAYER_PATH_POLICY

    with assert_raises_error(error_type="safety_error"):
        await _execute()(
            code=RENDER_ZONE_WRITE,
            description="write the rendered config",
            execution_mode=execution_mode,
        )

    records = _refusal_records(recorded_entries)
    assert len(records) == 1
    assert records[0]["reason"] == LAYER_PATH_POLICY == "path_policy"
    assert records[0]["subject"] == "execute"
    assert records[0]["source"] == RENDER_ZONE_WRITE
    # The layer, the mode and what matched all ride in the record's detail.
    assert "tool=execute" in records[0]["detail"]
    assert f"mode={execution_mode}" in records[0]["detail"]
    assert "build/config.yml" in records[0]["detail"]


async def test_execute_file_refusal_is_recorded(script_root, recorded_entries):
    script = _script(script_root, RENDER_ZONE_WRITE + "\n")
    with assert_raises_error(error_type="safety_error"):
        await _execute_file()(
            file_path=str(script),
            description="write the rendered config",
            execution_mode="readonly",
        )

    records = _refusal_records(recorded_entries)
    assert len(records) == 1
    assert records[0]["reason"] == "path_policy"
    assert records[0]["subject"] == "execute_file"


# ---------------------------------------------------------------------------
# Guard tampering, at the tool layer
# ---------------------------------------------------------------------------


@BOTH_MODES
async def test_guard_tamper_is_refused_at_the_tool_layer(execution_mode, recorded_entries):
    with assert_raises_error(error_type="safety_error") as ctx:
        await _execute()(
            code=GUARD_TAMPER,
            description="unpatch the guard",
            execution_mode=execution_mode,
        )

    assert any("_restore_patched_targets" in s for s in ctx["envelope"]["suggestions"])
    assert _refusal_records(recorded_entries)[0]["reason"] == "path_policy"


# ---------------------------------------------------------------------------
# The gate stays out of the way of ordinary work
# ---------------------------------------------------------------------------


async def test_clean_read_still_executes(tmp_path, monkeypatch, recorded_entries):
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
    assert _refusal_records(recorded_entries) == []


# ---------------------------------------------------------------------------
# Session posture clamp
#
# The gate itself is unit-tested in ``test_posture_clamp.py``. What is pinned
# here is that both tools actually *call* it — the clamp existed and was
# correct for a while without being wired into either tool, which left the
# executor gate that FR6 names as enforced doing nothing at all.
#
# The verdict refuses for several reasons and each phrase says who lifts it, so
# the assertions below fork on the reason: a read-only run, which the chip
# cannot lift, and a switch that is off, which the chip turns back on.
# ---------------------------------------------------------------------------

CLEAN_READWRITE_CODE = "print('would write')\n"

SESSION_KEY = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


@pytest.fixture
def readonly_run(monkeypatch):
    """A deployment-wide read-only run: the variable is on this process."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")


@pytest.fixture
def switched_off(control_context_root, monkeypatch):
    """The other source: the operator's switch is off for the run's machine.

    No ``OSPREY_EXECUTION_MODE`` anywhere — turning one machine off must not
    make the process read-only. The session key is still stamped because audit
    entries join on it.
    """
    from osprey_connectors import control_context, posture_store

    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)

    write_control_context(control_context_root, target=ARMED_TARGET, owned_by=None, switch={})
    posture_store.invalidate_cache()
    control_context.invalidate_cache()
    yield control_context_root
    posture_store.invalidate_cache()


#: The phrase each source refuses with, and the one it must NOT borrow.
POSTURE_REMEDIES = {
    "deployment": (refusal_phrase("readonly_run"), "turn it on from the chip"),
    "store": (refusal_phrase("switch_off"), "the chip cannot lift this"),
}


def _assert_posture_envelope(envelope, *, source="deployment"):
    """The refusal is its own reason's phrase, and names no key, file or variable.

    Sending an operator to a config key when the real gate is the switch costs
    them an edit that changes nothing — and sending them to the chip when the
    real gate is a read-only run costs them a click that changes nothing.
    """
    expected, forbidden = POSTURE_REMEDIES[source]
    text = " ".join([envelope["error_message"], *envelope["suggestions"]])
    assert envelope["error_message"].startswith(expected), text
    assert forbidden not in text.lower(), text
    assert REFUSAL_BAN_LIST.search(text) is None, text


@pytest.mark.usefixtures("readonly_run")
async def test_execute_readwrite_is_refused_under_a_read_only_run():
    with assert_raises_error(error_type="safety_error") as ctx:
        await _execute()(
            code=CLEAN_READWRITE_CODE,
            description="write a channel",
            execution_mode="readwrite",
        )

    _assert_posture_envelope(ctx["envelope"])


@pytest.mark.usefixtures("readonly_run")
async def test_execute_file_readwrite_is_refused_under_a_read_only_run(script_root):
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


@pytest.mark.usefixtures("readonly_run")
async def test_a_read_only_run_does_not_send_the_operator_to_the_chip():
    """The deployment-wide source names the chip only as what cannot lift it.

    The run itself is what refuses, so a message pointing at the chip as a fix
    is a dead end. It is still named, because it is the first place an
    operator would look.
    """
    with assert_raises_error(error_type="safety_error") as ctx:
        await _execute()(
            code=CLEAN_READWRITE_CODE,
            description="write a channel",
            execution_mode="readwrite",
        )

    envelope = ctx["envelope"]
    _assert_posture_envelope(envelope, source="deployment")
    assert "the chip cannot lift this" in envelope["error_message"]


@pytest.mark.usefixtures("switched_off")
async def test_a_switched_off_target_sends_the_operator_to_the_chip():
    """The switch is the operator's own, so the refusal points at the chip that
    turns it back on — the sentence the read-only branch must not borrow — and
    names the kind of machine it is about."""
    with assert_raises_error(error_type="safety_error") as ctx:
        await _execute()(
            code=CLEAN_READWRITE_CODE,
            description="write a channel",
            execution_mode="readwrite",
        )

    envelope = ctx["envelope"]
    _assert_posture_envelope(envelope, source="store")
    assert envelope["error_message"] == refusal_text("switch_off", "simulator")
    assert envelope["details"] == {"reason": "switch_off", "target": ARMED_TARGET}


async def _run_clean_readwrite(tmp_path, monkeypatch):
    """Run a harmless readwrite script with the subprocess mocked out.

    The scratch deployment's switch is on (``_no_ambient_deployment``), so only
    what the test changes can refuse.
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
        patch("osprey.mcp_server.python_executor.executor.execute_code", mock_exec),
    ):
        return await _execute()(
            code=CLEAN_READWRITE_CODE,
            description="harmless script",
            execution_mode="readwrite",
            save_output=False,
        )


@pytest.mark.usefixtures("readonly_run")
async def test_readonly_still_runs_under_a_read_only_run(tmp_path, monkeypatch):
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
    """No read-only variable and the switch on: the gate must not fire on absence."""
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


# ---------------------------------------------------------------------------
# The verdict, per control target
#
# Writes are switched per machine, so the gate has to ask about the machine the
# run will launch on. The deployment below is the shape that makes the
# difference visible: one tier that allows writes, the simulator switched on
# and the live machine switched off.
# ---------------------------------------------------------------------------

MIXED_POSTURE_SECTION = {
    "type": "epics",
    "writes": "allowed",
    "connector": {"epics": {}, "virtual_accelerator": {}},
}


@pytest.fixture
def mixed_posture(monkeypatch):
    """A deployment whose simulator is switched on and whose real machine is not."""
    monkeypatch.setattr(posture, "control_system_section", lambda: dict(MIXED_POSTURE_SECTION))
    monkeypatch.setattr(
        "osprey_connectors.config.get_config_value",
        lambda path, default=None, config_path=None: (
            MIXED_POSTURE_SECTION if path == "control_system" else default
        ),
    )


@pytest.fixture
def control_target(control_context_root):
    """Put the deployment on a control target, as its control-context record does.

    ``None`` removes the record: no record is every switch off, and the run
    would launch on the baseline.
    """
    from osprey_connectors import control_context, posture_store

    def _select(target, generation=1):
        if target is None:
            path = control_context.record_path()
            if path is not None and path.exists():
                path.unlink()
        else:
            write_control_context(
                control_context_root,
                target=target,
                generation=generation,
                owned_by=None,
                switch={"va": "on"},
            )
        posture_store.invalidate_cache()
        control_context.invalidate_cache()

    return _select


async def _run_readwrite(tmp_path, monkeypatch):
    """A harmless readwrite run with only the subprocess mocked out.

    Unlike :func:`_run_clean_readwrite` the deployment gate is left real — it is
    what these tests are about.
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
            return_value={"has_writes": True, "has_reads": False, "detected_patterns": {}},
        ),
        patch("osprey.mcp_server.python_executor.executor.execute_code", mock_exec),
    ):
        return await _execute()(
            code=CLEAN_READWRITE_CODE,
            description="a write",
            execution_mode="readwrite",
            save_output=False,
        )


def _assert_refused_for_the_live_machine(envelope, expected_target):
    """The refusal is the switch-off phrase for the real machine, and names no key."""
    text = " ".join([envelope["error_message"], *envelope["suggestions"]])
    assert envelope["error_message"] == refusal_text("switch_off", "real")
    assert REFUSAL_BAN_LIST.search(text) is None, text
    assert envelope["details"] == {"reason": "switch_off", "target": expected_target}


@pytest.mark.usefixtures("mixed_posture")
async def test_readwrite_runs_on_a_target_whose_switch_is_on(tmp_path, monkeypatch, control_target):
    """The live machine's switch being off does not hold the simulator off."""
    control_target("va")

    data = extract_response_dict(await _run_readwrite(tmp_path, monkeypatch))

    assert data.get("error") is not True
    assert data["execution_mode"] == "readwrite"


@pytest.mark.usefixtures("mixed_posture")
async def test_readwrite_is_refused_on_the_live_target(tmp_path, monkeypatch, control_target):
    """The same deployment, the other target: the machine's own switch refuses."""
    control_target("live")

    with assert_raises_error(error_type="safety_error") as ctx:
        await _run_readwrite(tmp_path, monkeypatch)

    _assert_refused_for_the_live_machine(ctx["envelope"], "live")


@pytest.mark.usefixtures("mixed_posture")
async def test_an_unstamped_run_is_answered_for_the_baseline_target(
    tmp_path, monkeypatch, control_target
):
    """No record means the baseline — here the live machine — with every switch off."""
    control_target(None)

    with assert_raises_error(error_type="safety_error") as ctx:
        await _run_readwrite(tmp_path, monkeypatch)

    _assert_refused_for_the_live_machine(ctx["envelope"], "live")


@pytest.mark.usefixtures("mixed_posture")
async def test_a_failing_target_read_answers_the_baseline_rather_than_skipping(
    tmp_path, monkeypatch
):
    """Not knowing the target narrows the question to the baseline; it never drops the gate."""
    from osprey_connectors import control_context

    def unreadable(**kwargs):
        raise RuntimeError("state directory unreadable")

    monkeypatch.setattr(control_context, "read_record", unreadable)

    with assert_raises_error(error_type="safety_error") as ctx:
        await _run_readwrite(tmp_path, monkeypatch)

    _assert_refused_for_the_live_machine(ctx["envelope"], "live")
