"""Every write a held session starts is refused where it executes.

The hold is read by the write tools, the executor's launch gate, the Bluesky
queue, the Phoebus drive, the audit middleware's clamp and the connector inside
an executed script, so an approval granted before a flag opened buys nothing.
The ways out of what a flag was raised about stay open: a plain ``queue_stop``,
``stop_run`` and ``control_target_set`` are never refused ``session_held``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastmcp.exceptions import ToolError

from osprey.mcp_server.bluesky.server_context import initialize_server_context, reset_server_context
from osprey.mcp_server.bluesky.tools import queue, stop
from osprey.mcp_server.control_system.tools import channel_write
from osprey.mcp_server.phoebus.tools import bridge_tools
from osprey.mcp_server.python_executor import executor
from osprey.mcp_server.python_executor.tools._execution_gates import enforce_posture_clamp
from osprey.mcp_server.sandbox_env import scrub_sandbox_child_env
from osprey.registry.mcp import framework_mixed_read_write_tools, framework_write_tools
from osprey.services.python_executor.execution.wrapper import ExecutionWrapper
from osprey.utils.workspace import STATE_DIR_RELPATH
from osprey_connectors import posture_store, session_hold
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors.types import TARGET_VA
from osprey_connectors_tests._harness import (
    isolate_verdict_environment,
    reset_verdict_state,
    write_control_context,
)
from osprey_connectors_tests._write_fakes import ARMED_SECTION
from tests._approval_stamp_fixture import ApprovalStamps, use_stamp_dir
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn
from tests.mcp_server.test_audit_middleware import _call, _context, _hook_config, am

KEY = "3f2b8c1e-6a0d-4c55-9a1e-0b7f2d4e9c11"  # gitleaks:allow - a test session id
SESSION_ID = "OSPREY_SESSION_ID"
SECTION = {"type": "epics", "writes": "allowed"}
ASKING = {"enabled": True, "default_policy": "always"}
HELD = "[osprey-refusal: session_held]"
SOURCE = "watchdog:scope"
TOKEN = "genuinely-valid-token"
OPS = [{"channel": "SR:TEST:SP", "value": 1.0}]

QUEUE = "osprey.mcp_server.bluesky.tools.queue"
STOP = "osprey.mcp_server.bluesky.tools.stop"
PHOEBUS = "osprey.mcp_server.phoebus.tools.bridge_tools"


@pytest.fixture(autouse=True)
def deployment(tmp_path, monkeypatch) -> Iterator[Path]:
    """A writable deployment: tier allows, every switch this file names on, one config."""
    data = tmp_path / "agent_data"
    isolate_verdict_environment(monkeypatch, data)
    monkeypatch.delenv(SESSION_ID, raising=False)
    monkeypatch.delenv(session_hold.HOLD_DIR_ENV_VAR, raising=False)
    monkeypatch.setenv("PHOEBUS_REQUIRE_HANDLE", "0")
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yml"
    config.write_text(
        yaml.dump(
            {
                "control_system": SECTION,
                "phoebus": {"agent_access": "read_write"},
                "approval": ASKING,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setenv("CONFIG_FILE", str(config))
    write_control_context(data, target="live", switch={"live": "on"})
    reset_verdict_state()
    yield config
    reset_verdict_state()
    reset_server_context()


@pytest.fixture
def hold_root(tmp_path, monkeypatch) -> Path:
    """The session's record under a stamped store, nothing open."""
    root = tmp_path / "var" / "state" / "session_holds"
    monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(root))
    monkeypatch.setenv(SESSION_ID, KEY)
    session_hold.ensure_session(KEY)
    return root


@pytest.fixture
def stamps(tmp_path, monkeypatch) -> ApprovalStamps:
    return use_stamp_dir(monkeypatch, tmp_path / "state")


def _flag() -> None:
    assert session_hold.open_flag(
        KEY,
        flag_id="f1",
        source=SOURCE,
        reason="off the plan",
        prompt_ref="record:f1",
        prompt_excerpt="move the corrector",
    )


def _release() -> None:
    session_hold.answer_flags(KEY, ["f1"], answer="continue", operator="op")


def _assert_held(envelope: dict) -> None:
    assert envelope["details"]["reason"] == "session_held"
    assert refusal_phrase("session_held") in envelope["error_message"]
    assert SOURCE in envelope["error_message"]
    assert envelope["error_message"].endswith(HELD)


# -- channel_write -----------------------------------------------------------


@pytest.mark.usefixtures("hold_root")
async def test_channel_write_approved_before_the_flag_is_refused_at_execution(stamps):
    tool = get_tool_fn(channel_write.channel_write)
    with stamps.approved("channel_write", operations=OPS):
        _flag()
        with assert_raises_error(error_type=channel_write.WRITES_DISABLED_ERROR) as ctx:
            await tool(operations=OPS)
    _assert_held(ctx["envelope"])


@pytest.mark.usefixtures("hold_root")
def test_channel_write_passes_its_gate_after_the_release(deployment):
    config = yaml.safe_load(deployment.read_text(encoding="utf-8"))
    _flag()
    with pytest.raises(ToolError):
        channel_write._check_write_verdict(None, config)
    _release()
    assert channel_write._check_write_verdict(None, config) is None


# -- the Phoebus drive -------------------------------------------------------


@pytest.mark.usefixtures("hold_root")
async def test_phoebus_drive_is_refused(stamps):
    _flag()
    with patch(f"{PHOEBUS}._http_post_drive") as post:
        with stamps.approved("phoebus_drive"):
            with assert_raises_error(error_type="writes_disabled") as ctx:
                await get_tool_fn(bridge_tools.phoebus_drive)(widget="SetButton", verb="click")
    post.assert_not_called()
    _assert_held(ctx["envelope"])


# -- the Bluesky queue -------------------------------------------------------


@pytest.fixture
def lane(deployment) -> Iterator[None]:
    """A queue server over the deployment, approval off: no prompt runs in front of these calls."""
    config = yaml.safe_load(deployment.read_text(encoding="utf-8"))
    config["approval"] = {"enabled": False}
    deployment.write_text(yaml.dump(config), encoding="utf-8")
    initialize_server_context()
    with (
        patch(f"{QUEUE}.notify_agent_activity_async"),
        patch(f"{STOP}.notify_agent_activity_async"),
    ):
        yield


@pytest.mark.usefixtures("hold_root", "lane")
async def test_queue_start_is_refused():
    _flag()
    with patch(f"{QUEUE}._http_post_json") as post:
        with assert_raises_error(error_type="writes_disabled") as ctx:
            await get_tool_fn(queue.queue_start)()
    post.assert_not_called()
    _assert_held(ctx["envelope"])


@pytest.mark.usefixtures("hold_root", "lane")
async def test_withdrawing_a_halt_is_refused():
    _flag()
    with patch(f"{QUEUE}._http_post_json") as post:
        with assert_raises_error(error_type="writes_disabled") as ctx:
            await get_tool_fn(queue.queue_stop)(cancel=True)
    post.assert_not_called()
    _assert_held(ctx["envelope"])


@pytest.mark.usefixtures("hold_root", "lane")
async def test_with_no_clamp_in_front_queue_add_goes_out_unarmed_and_says_why():
    """The tool's own answer, for a server the audit middleware does not wrap."""
    _flag()
    body = {"run_id": "r1", "revision": 3, "item": {"item_uid": "u1"}}
    with patch(f"{QUEUE}._http_post_json", return_value=(200, body)) as post:
        result = await get_tool_fn(queue.queue_add)(draft_revision=3)
    assert "X-Launch-Token" not in post.call_args.kwargs["headers"]
    refused = extract_response_dict(result)["writes_refused"]
    assert refused["reason"] == "session_held"
    assert SOURCE in refused["message"]


@pytest.mark.usefixtures("hold_root", "lane")
async def test_queue_add_arms_after_the_release():
    _flag()
    _release()
    body = {"run_id": "r1", "revision": 3, "item": {"item_uid": "u1"}}
    with patch(f"{QUEUE}._http_post_json", return_value=(200, body)) as post:
        result = await get_tool_fn(queue.queue_add)(draft_revision=3)
    assert post.call_args.kwargs["headers"]["X-Launch-Token"] == TOKEN
    assert "writes_refused" not in extract_response_dict(result)


@pytest.mark.usefixtures("hold_root", "lane")
async def test_a_plain_queue_stop_still_halts():
    _flag()
    with patch(f"{QUEUE}._http_post_json", return_value=(200, {"stop_pending": True})) as post:
        await get_tool_fn(queue.queue_stop)()
    post.assert_called_once()


@pytest.mark.usefixtures("hold_root", "lane")
async def test_stop_run_still_aborts():
    _flag()
    answer = (200, {"success": True, "state": "aborting"})
    with patch(f"{STOP}._http_post_json", return_value=answer) as post:
        await get_tool_fn(stop.stop_run)()
    post.assert_called_once()


# -- the audit middleware's clamp ---------------------------------------------


@pytest.fixture
def clamp(tmp_path, monkeypatch) -> Iterator[Path]:
    """A render whose hook config clamps a write tool and declares every halting tool."""
    build = tmp_path / "build"
    (build / ".claude" / "hooks").mkdir(parents=True)
    (build / "config.yml").write_text("control_system: {}\n", encoding="utf-8")
    hook_config = build / ".claude" / "hooks" / "hook_config.json"
    hook_config.write_text(
        json.dumps(
            _hook_config(
                write_tools=[
                    "mcp__controls__channel_write",
                    "mcp__controls__queue_stop",
                    "mcp__controls__stop_run",
                    "mcp__controls__control_target_set",
                ]
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(am.CONFIG_ENV, str(build / "config.yml"))
    monkeypatch.setenv(am.TOOL_PREFIX_ENV, "controls")
    from osprey.audit import posture as posture_module

    monkeypatch.setattr(posture_module, "control_system_section", lambda: dict(SECTION))
    am.reset_audit_state()
    yield hook_config
    am.reset_audit_state()


def _degrade(monkeypatch, hook_config: Path) -> None:
    """Take the verified path away: no hook config, an unverified server name."""
    hook_config.unlink()
    monkeypatch.setenv(am.TOOL_PREFIX_ENV, "controls_ring")
    am.reset_audit_state()


async def _refused_by_the_clamp(tool: str) -> dict:
    seen: list[str] = []

    async def call_next(context):  # pragma: no cover - must never run
        seen.append(context.message.name)
        return "ran"

    with pytest.raises(ToolError) as excinfo:
        await am.AuditMiddleware().on_call_tool(_context(tool), call_next)
    assert seen == []
    return json.loads(str(excinfo.value))


@pytest.mark.usefixtures("hold_root")
@pytest.mark.parametrize("degraded", [False, True], ids=["verified", "degraded"])
async def test_a_clamped_tool_is_refused_and_a_release_lifts_it(clamp, monkeypatch, degraded):
    if degraded:
        _degrade(monkeypatch, clamp)
    _flag()
    envelope = await _refused_by_the_clamp("channel_write")
    assert envelope["error_type"] == "safety_error"
    assert refusal_phrase("session_held") in envelope["error_message"]
    assert SOURCE in envelope["error_message"]
    assert envelope["error_message"].endswith(HELD)

    _release()
    _returned, seen = await _call(am.AuditMiddleware(), "channel_write")
    assert seen == ["channel_write"]


@pytest.mark.usefixtures("hold_root", "clamp")
async def test_a_clamp_refusal_is_recorded_under_the_posture_reason(recorded_entries):
    _flag()
    await _refused_by_the_clamp("channel_write")
    bodies = [dict(entry.body) for entry in recorded_entries("audit")]
    record = next(body for body in bodies if body.get("surface") == "controls")
    assert record["reason"] == am.REASON_POSTURE
    assert "reason=session_held" in record["detail"]


@pytest.mark.usefixtures("hold_root")
@pytest.mark.parametrize("degraded", [False, True], ids=["verified", "degraded"])
@pytest.mark.parametrize("tool", ["queue_stop", "stop_run", "control_target_set"])
async def test_a_halting_tool_is_never_held(clamp, monkeypatch, tool, degraded):
    """Declared in the render on purpose: the exemption must hold when membership would not."""
    if degraded:
        _degrade(monkeypatch, clamp)
        # The degraded path matches bare names against the floor.
        monkeypatch.setattr(am, "_FLOOR_CLAMP", am._FLOOR_CLAMP | {f"mcp__controls__{tool}"})
    _flag()
    _returned, seen = await _call(am.AuditMiddleware(), tool)
    assert seen == [tool]


@pytest.mark.usefixtures("hold_root", "clamp")
async def test_a_halting_tool_still_answers_to_the_backstop(monkeypatch):
    monkeypatch.setenv(am.POSTURE_ENV, "readonly")
    _flag()
    envelope = await _refused_by_the_clamp("queue_stop")
    assert envelope["error_message"].endswith("[osprey-refusal: readonly_run]")


@pytest.mark.usefixtures("hold_root", "clamp")
async def test_a_tool_outside_the_clamp_reads_no_hold():
    _flag()
    with patch.object(am, "session_hold_refusal") as read:
        _returned, seen = await _call(am.AuditMiddleware(), "channel_read")
    assert seen == ["channel_read"]
    read.assert_not_called()


@pytest.mark.usefixtures("hold_root")
async def test_queue_add_is_refused_by_the_clamp_on_the_real_render(clamp, monkeypatch):
    """On a wrapped server the clamp answers first: the tool never runs, nothing is staged."""
    write_tools = framework_write_tools()
    assert "mcp__bluesky__queue_add" in write_tools
    clamp.write_text(
        json.dumps(
            _hook_config(write_tools=write_tools, mixed=framework_mixed_read_write_tools()),
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(am.TOOL_PREFIX_ENV, "bluesky")
    am.reset_audit_state()
    _flag()
    envelope = await _refused_by_the_clamp("queue_add")
    assert envelope["error_message"].endswith(HELD)

    _release()
    _returned, seen = await _call(am.AuditMiddleware(), "queue_add")
    assert seen == ["queue_add"]


# -- the executor -------------------------------------------------------------


@pytest.fixture
def launchable(monkeypatch) -> None:
    from osprey.audit import posture as posture_module

    monkeypatch.setattr(posture_module, "control_system_section", lambda: dict(SECTION))


@pytest.mark.usefixtures("hold_root", "launchable")
def test_a_script_launched_while_held_is_refused_and_its_pin_stays_writes():
    _flag()
    with pytest.raises(ToolError) as excinfo:
        enforce_posture_clamp("readwrite", tool="execute")
    _assert_held(json.loads(str(excinfo.value)))

    expected = posture_store.launch_posture_stamp("live", posture_store.POSTURE_WRITES)
    assert executor._launch_posture("live") == expected


@pytest.mark.usefixtures("hold_root", "launchable")
def test_a_release_lets_the_script_launch_and_pins_it_for_writes():
    _flag()
    _release()
    assert enforce_posture_clamp("readwrite", tool="execute") is None
    stamp = executor._launch_posture("live")
    assert stamp == posture_store.launch_posture_stamp("live", posture_store.POSTURE_WRITES)
    assert posture_store.POSTURE_SANDBOX not in stamp


@pytest.mark.usefixtures("hold_root", "launchable")
def test_a_readonly_script_is_not_held_at_launch():
    _flag()
    assert enforce_posture_clamp("readonly", tool="execute") is None


def test_the_sandbox_child_keeps_the_session_and_the_store():
    child = scrub_sandbox_child_env(
        {SESSION_ID: KEY, session_hold.HOLD_DIR_ENV_VAR: "/holds", "UNRELATED": "x"}
    )
    assert child[SESSION_ID] == KEY
    assert child[session_hold.HOLD_DIR_ENV_VAR] == "/holds"
    assert "UNRELATED" not in child


_WRITE_SCRIPT = """
import osprey.runtime as runtime
from osprey_connectors_tests._write_fakes import RecordingConnector

connector = RecordingConnector()
connector._control_target = runtime._stamped_target()


async def _get_connector():
    return connector


runtime._get_connector = _get_connector
try:
    runtime.write_channel("SR:TEST:SP", 1.0)
except Exception as exc:
    print("REFUSED", exc)
else:
    print("WRITTEN")
print("LANDED", connector.writes)
"""


def _run_script(tmp_path: Path, script: str) -> str:
    """Run *script* in a child with the environment an executed script is given."""
    child = scrub_sandbox_child_env(os.environ)
    child["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    child[executor.ENV_CONTROL_TARGET] = TARGET_VA
    child[executor.ENV_CONTROL_TARGET_GENERATION] = "1"
    path = tmp_path / "executed.py"
    path.write_text(script, encoding="utf-8")
    proc = subprocess.run(  # fixed argv, test-authored script
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=child,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.mark.usefixtures("hold_root")
def test_runtime_write_channel_in_an_executed_script_is_refused(tmp_path, deployment):
    """The child reads the session and the store from the environment it was given."""
    config = yaml.safe_load(deployment.read_text(encoding="utf-8"))
    config["approval"] = {"enabled": False}
    config["control_system"] = ARMED_SECTION
    deployment.write_text(yaml.dump(config), encoding="utf-8")
    write_control_context(tmp_path / "agent_data", target=TARGET_VA, switch={TARGET_VA: "on"})
    written = _run_script(tmp_path, _WRITE_SCRIPT)
    assert written.startswith("WRITTEN"), written

    _flag()
    output = _run_script(tmp_path, _WRITE_SCRIPT)
    assert output.startswith("REFUSED"), output
    assert "session_held" in output
    assert "LANDED []" in output


def _guard_probe(repo: Path, statement: str) -> str:
    """Run the executor's guard plus *statement* in a child; return stdout."""
    agent_data = repo / "var" / "agent_data"
    folder = agent_data / "run_0001"
    folder.mkdir(parents=True, exist_ok=True)
    wrapper = ExecutionWrapper(
        execution_mode="readwrite",
        protected_roots=executor.resolve_protected_roots(repo),
        permitted_roots=(agent_data,),
    )
    body = textwrap.dedent(
        """
        import os
        from pathlib import Path

        try:
            {statement}
        except PermissionError as exc:
            print("REFUSED:", exc)
        else:
            print("ALLOWED")
        """
    ).format(statement=statement)
    script = repo / "probe_script.py"
    script.write_text(wrapper._get_filesystem_guard(folder) + "\n" + body + "\n", encoding="utf-8")
    proc = subprocess.run(  # fixed argv, test-authored script
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=str(repo),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.mark.parametrize("form", ["literal", "computed"])
@pytest.mark.parametrize(
    "statement",
    [
        "open({path}, 'w').write('{{}}')",
        "os.remove({path})",
        "Path({path}).write_text('{{}}')",
    ],
    ids=["open", "remove", "pathlib"],
)
def test_an_executed_script_cannot_release_the_hold(tmp_path, monkeypatch, form, statement):
    """The honest path: the record sits under a protected root and stays what it was."""
    repo = tmp_path.resolve() / "repo"
    (repo / "var" / "agent_data").mkdir(parents=True)
    (repo / "profile.yml").write_text("project: demo\n", encoding="utf-8")
    root = repo / STATE_DIR_RELPATH / "session_holds"
    monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(root))
    monkeypatch.setenv(SESSION_ID, KEY)
    session_hold.ensure_session(KEY)
    _flag()
    record = root / KEY / session_hold.RECORD_NAME
    before = record.read_text(encoding="utf-8")

    text = str(record)
    middle = len(text) // 2
    path = repr(text) if form == "literal" else f"{text[:middle]!r} + {text[middle:]!r}"
    output = _guard_probe(repo, statement.format(path=path))

    assert output.strip().splitlines()[-1].startswith("REFUSED"), output
    assert record.read_text(encoding="utf-8") == before
    assert session_hold.is_held(KEY) is True


# -- a process with no session -----------------------------------------------


@pytest.mark.usefixtures("hold_root", "launchable")
def test_a_kernel_with_no_session_is_not_held(deployment, monkeypatch):
    """A Jupyter kernel is the operator's own: it carries no session id."""
    config = yaml.safe_load(deployment.read_text(encoding="utf-8"))
    _flag()
    monkeypatch.delenv(SESSION_ID)
    assert channel_write._check_write_verdict(None, config) is None
    assert enforce_posture_clamp("readwrite", tool="execute") is None


# -- an older connectors install ---------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "osprey.utils.session_hold_api",
        "osprey.mcp_server.audit_middleware",
        "osprey.mcp_server.bluesky.tools.queue",
        "osprey.mcp_server.control_system.tools.channel_write",
        "osprey.mcp_server.phoebus.tools.bridge_tools",
        "osprey.mcp_server.python_executor.tools._execution_gates",
    ],
)
def test_an_older_connectors_install_fails_at_import_with_the_version_message(module):
    """Every site stops the server, in words that say what to install."""
    probe = (
        "import importlib\n"
        "from osprey_connectors import posture_store\n"
        "del posture_store.execution_verdict, posture_store.session_hold_refusal\n"
        "try:\n"
        f"    importlib.import_module({module!r})\n"
        "except RuntimeError as exc:\n"
        "    print(exc)\n"
        "else:\n"
        "    print('IMPORTED')\n"
    )
    proc = subprocess.run(  # fixed argv, test-authored script
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "needs osprey-connectors with session_hold" in proc.stdout, proc.stdout


# -- the hook ------------------------------------------------------------------


@pytest.mark.usefixtures("hold_root")
def test_the_writes_check_hook_refuses_before_a_prompt(monkeypatch):
    from tests.integration._write_verdict_surfaces_registry import _import_hook

    hook = _import_hook("osprey_writes_check")
    monkeypatch.setattr(hook._target_state, "hook_write_target", lambda *_a, **_k: "live")
    loader = hook._stage_two.__globals__["load_osprey_config"]
    monkeypatch.setitem(loader.__globals__, "_osprey_config_cache", None)
    _flag()
    answer = hook._stage_two({"tool_input": {}}, "mcp__controls__channel_write", "channel_write")
    assert answer[:2] == (False, "session_held")


def _run_hook(monkeypatch, capsys, tool: str) -> dict | None:
    """``main()`` of the writes-check hook on *tool*: the deny it emits, or ``None``."""
    from tests.integration._write_verdict_surfaces_registry import _import_hook

    hook = _import_hook("osprey_writes_check")
    loader = hook._stage_two.__globals__["load_osprey_config"]
    monkeypatch.setitem(loader.__globals__, "_osprey_config_cache", None)
    monkeypatch.setattr(hook, "get_hook_input", lambda: {"tool_name": tool, "tool_input": {}})
    monkeypatch.setattr(hook, "write_tools", lambda: [tool])
    rendered = {
        "server_prefixes": ["bluesky"],
        hook._LANE_ADDRESSED_KEY: ["queue_add", "queue_start"],
    }
    monkeypatch.setattr(hook, "load_hook_config", lambda: rendered)
    capsys.readouterr()
    with pytest.raises(SystemExit) as stopped:
        hook.main()
    assert stopped.value.code == 0
    out = capsys.readouterr().out
    return json.loads(out)["hookSpecificOutput"] if out else None


@pytest.mark.usefixtures("hold_root")
@pytest.mark.parametrize("tool", ["queue_add", "queue_start"])
def test_a_held_lane_tool_is_denied_ahead_of_its_prompt(monkeypatch, capsys, tool):
    _flag()
    deny = _run_hook(monkeypatch, capsys, f"mcp__bluesky__{tool}")
    assert deny["permissionDecision"] == "deny"
    assert deny["permissionDecisionReason"].endswith(HELD)

    _release()
    assert _run_hook(monkeypatch, capsys, f"mcp__bluesky__{tool}") is None


@pytest.mark.usefixtures("hold_root")
def test_a_lane_tool_in_a_session_with_no_flag_keeps_the_carve_out(monkeypatch, capsys):
    assert _run_hook(monkeypatch, capsys, "mcp__bluesky__queue_add") is None


def test_a_lane_tool_with_no_session_is_not_held(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(tmp_path / "absent"))
    assert _run_hook(monkeypatch, capsys, "mcp__bluesky__queue_add") is None


@pytest.mark.usefixtures("hold_root")
def test_the_tier_answers_ahead_of_the_hold_for_a_lane_tool(monkeypatch, capsys, deployment):
    """A tier that refuses keeps the lane exit: the tool's own gate words that refusal."""
    config = yaml.safe_load(deployment.read_text(encoding="utf-8"))
    config["control_system"]["writes"] = "never"
    deployment.write_text(yaml.dump(config), encoding="utf-8")
    _flag()
    assert _run_hook(monkeypatch, capsys, "mcp__bluesky__queue_add") is None
