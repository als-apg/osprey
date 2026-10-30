"""Every case-table row through every surface that answers "may this write?" (D6).

The case table (``osprey_connectors_tests/_harness/fixtures/write_verdict_cases.json``)
states one write verdict per row. The registry
(:mod:`tests.integration._write_verdict_surfaces_registry`) names every src
surface that asks that verdict, with the projection of a row onto what the
surface returns. This module holds the assertions:

* the registry covers every src module that calls ``write_verdict`` or
  ``backstop_verdict`` (an AST scan), so a new caller left out fails here;
* every row in a surface's domain reads on that surface as the row says —
  write rows on write surfaces, backstop rows on backstop surfaces;
* the switch off with no stamp reads ``switch_off`` on every stage-1 surface;
* the headless invariant: a headless read-only run blocks every framework and
  declared write tool, whatever the switch says.

A disagreement is fixed in src, never by loosening a projection.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey_connectors.workspace import reset_config_cache
from osprey_connectors_tests._harness import isolate_verdict_environment, reset_verdict_state
from osprey_connectors_tests._harness._write_verdict_cases import CASES, Case, host_root, stage
from tests.integration._write_verdict_surfaces_registry import (
    BY_NAME,
    OUT_OF_DOMAIN,
    SURFACES,
    Staged,
    Surface,
    write_config,
)

REPO = Path(__file__).resolve().parents[2]
SRC_ROOTS = (REPO / "src", REPO / "packages" / "osprey-connectors" / "src")
VERDICT_NAMES = frozenset({"write_verdict", "backstop_verdict"})

#: The modules that define a verdict rather than ask one.
DEFINING_MODULES = frozenset(
    {
        "osprey_connectors.posture_store",
        "osprey.templates.claude_code.claude.hooks.osprey_target_state",
    }
)

#: The modules a surface reads the stored position from rather than asking the
#: verdict: ``osprey status`` prints each login's switch positions at its tier.
POSITION_READERS = frozenset({"osprey.deployment.status_display"})

#: The row with the switch off, no stamped owner and no launch pin.
SWITCH_OFF_NO_STAMP = "switch-off-no-stamp"


# -- staging -------------------------------------------------------------------


def _project(surface: Surface, case: Case, tmp_path: Path) -> Any:
    """Stage *case* in its own directory and ask *surface*; restore everything after."""
    row_dir = tmp_path / surface.name / case.id
    row_dir.mkdir(parents=True)
    with pytest.MonkeyPatch.context() as monkeypatch:
        isolate_verdict_environment(monkeypatch, host_root(row_dir))
        config_path = write_config(row_dir / "config.yml", case.section)
        monkeypatch.setenv("CONFIG_FILE", str(config_path))
        monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
        monkeypatch.chdir(row_dir)
        reset_config_cache()
        try:
            stage(case, monkeypatch, row_dir)
            return surface.project(Staged(case, row_dir, monkeypatch, config_path))
        finally:
            reset_verdict_state()
            reset_config_cache()


def _rows_for(surface: Surface) -> list[Case]:
    return [case for case in CASES if case.verdict == surface.verdict]


# -- the registry covers every caller ----------------------------------------


def _module_name(path: Path) -> str:
    for root in SRC_ROOTS:
        if path.is_relative_to(root):
            return ".".join(path.relative_to(root).with_suffix("").parts).removesuffix(".__init__")
    raise AssertionError(path)


def _verdict_callers() -> set[str]:
    callers = set()
    for root in SRC_ROOTS:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else None
                )
                if name in VERDICT_NAMES:
                    callers.add(_module_name(path))
                    break
    return callers


def test_the_registry_covers_every_src_caller():
    callers = _verdict_callers()
    registered = {surface.module for surface in SURFACES}

    assert callers, "the scan found no caller: it is looking in the wrong place"
    assert callers - DEFINING_MODULES <= registered, sorted(callers - registered)
    assert registered <= callers | DEFINING_MODULES | POSITION_READERS, sorted(
        registered - callers - DEFINING_MODULES - POSITION_READERS
    )


def test_the_registry_names_are_unique():
    assert len(BY_NAME) == len(SURFACES)


def test_every_verdict_kind_has_a_surface_and_a_row():
    for surface in SURFACES:
        assert surface.verdict in {"write", "backstop"}, surface.name
        assert _rows_for(surface), surface.name


# -- every row through every surface -------------------------------------------


@pytest.mark.parametrize("surface", SURFACES, ids=[surface.name for surface in SURFACES])
def test_the_surface_reads_every_row_in_its_domain(surface: Surface, tmp_path):
    mismatches = []
    asked = 0
    for case in _rows_for(surface):
        answer = _project(surface, case, tmp_path)
        if answer is OUT_OF_DOMAIN:
            continue
        asked += 1
        observed, expected = answer
        if observed != expected:
            mismatches.append(f"{case.id}: {surface.name} read {observed!r}, row says {expected!r}")

    assert asked, f"{surface.name} was asked no row"
    assert not mismatches, "\n".join(mismatches)


def test_the_table_has_the_switch_off_no_stamp_row():
    (case,) = [case for case in CASES if case.id == SWITCH_OFF_NO_STAMP]
    assert case.verdict == "write"
    assert not case.env, "no stamped owner, no launch pin, no run mode"
    assert not case.has_owner
    assert case.record is not None and not case.record.get("switch")
    assert case.expect == {"allowed": False, "reason": "switch_off"}


STAGE1 = [surface for surface in SURFACES if surface.stage1]


@pytest.mark.parametrize("surface", STAGE1, ids=[surface.name for surface in STAGE1])
def test_switch_off_with_no_stamp_reads_switch_off_at_stage_1(surface: Surface, tmp_path):
    (case,) = [case for case in CASES if case.id == SWITCH_OFF_NO_STAMP]

    answer = _project(surface, case, tmp_path)

    assert answer is not OUT_OF_DOMAIN, f"{surface.name} cannot be asked the switch-off row"
    observed, expected = answer
    assert observed == expected
    if surface.reports_reason:
        assert observed == (False, "switch_off")
    else:
        assert observed is False


# -- the headless invariant ----------------------------------------------------

_DECLARING_SERVER = (
    "claude_code:\n  servers:\n    plc:\n      command: plc-server\n      writes: [set_value]\n"
)


@pytest.mark.parametrize("switch", [None, {"live": "on"}], ids=["switch-off", "switch-on"])
def test_a_headless_run_blocks_every_write_tool_whatever_the_switch(switch, monkeypatch, tmp_path):
    """Framework and declared write tools are blocked headless, switch on or off."""
    from osprey.agent_runner.write_tools import _FALLBACK_WRITE_TOOLS, read_only_disallowed_tools
    from osprey_connectors_tests._harness import write_control_context

    isolate_verdict_environment(monkeypatch, host_root(tmp_path))
    write_control_context(host_root(tmp_path), switch=switch, owned_by=None)
    (tmp_path / "config.yml").write_text(_DECLARING_SERVER, encoding="utf-8")

    blocked = set(read_only_disallowed_tools(tmp_path))

    assert set(_FALLBACK_WRITE_TOOLS) <= blocked
    assert "mcp__plc__set_value" in blocked
    assert "Bash" in blocked


# -- the switch's positions ----------------------------------------------------

#: The entries that read the position a write runs at, not only whether it runs.
POSITION_SURFACES = (
    "approval-position",
    "control-context-word",
    "chip-row-position",
    "status-position",
)


def _sides(staged: Staged) -> Any:
    """The package's and the twin's ``(allowed, reason, position)`` for one row."""
    from osprey_connectors.posture_store import write_verdict
    from osprey_connectors_tests._harness._write_verdict_cases import owner_argument
    from tests.integration._write_verdict_surfaces_registry import _import_hook

    case = staged.case
    twin = _import_hook("osprey_target_state")
    package = write_verdict(
        case.section, case.target, ignore_launch_pin=case.ignore_launch_pin, **owner_argument(case)
    )
    hook = twin.write_verdict(
        case.section,
        case.target,
        ignore_launch_pin=case.ignore_launch_pin,
        **owner_argument(case, no_owner=twin.NO_OWNER),
    )
    return {
        side: (verdict.allowed, verdict.reason, verdict.position)
        for side, verdict in (("package", package), ("twin", hook))
    }


_SIDES = Surface("sides", "", "write", stage1=False, reports_reason=True, project=_sides)


def test_the_package_and_the_twin_read_every_rows_position(tmp_path):
    from osprey_connectors_tests._harness._write_verdict_cases import expected_position

    mismatches = []
    for case in _rows_for(_SIDES):
        expected = (case.expect["allowed"], case.expect["reason"], expected_position(case))
        for side, observed in _project(_SIDES, case, tmp_path).items():
            if side in case.applies_to and observed != expected:
                mismatches.append(f"{case.id}: {side} read {observed!r}, row says {expected!r}")

    assert not mismatches, "\n".join(mismatches)


@pytest.mark.parametrize("name", POSITION_SURFACES)
def test_a_position_surface_is_asked_every_position(name, tmp_path):
    """A position entry that never meets ``asked`` and ``allowed`` and a refusal pins nothing."""
    from osprey_connectors_tests._harness._write_verdict_cases import expected_position

    surface = BY_NAME[name]
    met = set()
    for case in _rows_for(surface):
        if _project(surface, case, tmp_path) is not OUT_OF_DOMAIN:
            met.add(expected_position(case) if case.expect["allowed"] else "refused")

    assert met == {"asked", "allowed", "refused"}, f"{name} met only {sorted(met)}"


# -- stage 1 against the hook: a record exactly where the hook leaves one ---------

_HOOK = REPO / "src" / "osprey" / "templates" / "claude_code" / "claude" / "hooks"

#: The framework's governed tools as a render lists them; ``phoebus_drive`` keeps its policy.
_GOVERNED = (
    "mcp__controls__channel_write",
    "mcp__python__execute",
    "mcp__python__execute_file",
    "mcp__python__pyaml_measure",
)
_PREFIXES = ["mcp__controls__", "mcp__python__", "mcp__phoebus__"]
_HOOK_CONFIG = {
    "server_prefixes": _PREFIXES,
    "approval_prefixes": _PREFIXES,
    "write_tools": [*_GOVERNED, "mcp__phoebus__phoebus_drive"],
    "governed_tools": list(_GOVERNED),
    "lane_addressed_tools": [],
}

_SECTION = {
    "type": "epics",
    "writes": "allowed",
    "connector": {"epics": {}, "virtual_accelerator": {}},
}
_OPERATIONS = [{"channel": "TEST:PV", "value": 1.0}]

#: Approval settings that would each let a non-governed call through unasked, and the default.
_APPROVALS = {
    "disabled": {"enabled": False},
    "skip": {"enabled": True, "default_policy": "skip"},
    "tool-skip": {"enabled": True, "tools": {"channel_write": "skip", "execute": "skip"}},
    "always": {"enabled": True, "default_policy": "always"},
}


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """A simulator deployment whose agent data, config and hook config live in *tmp_path*."""
    from osprey_connectors_tests._harness import pin_identity, state_dir_under

    root = tmp_path / "agent_data"
    isolate_verdict_environment(monkeypatch, root)
    pin_identity(monkeypatch)
    for name in ("OSPREY_DISPATCH_RUN", "OSPREY_POSTURE_SESSION"):
        monkeypatch.delenv(name, raising=False)
    state = state_dir_under(root)
    state.mkdir(parents=True)
    hook_config = tmp_path / "hook_config.json"
    hook_config.write_text(json.dumps(_HOOK_CONFIG), encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("OSPREY_HOOK_CONFIG", str(hook_config))
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
    monkeypatch.chdir(tmp_path)
    yield _Deployment(tmp_path, root, state, repo, monkeypatch)
    reset_verdict_state()
    reset_config_cache()


class _Deployment:
    def __init__(self, tmp_path, root, state, repo, monkeypatch):
        self.tmp_path = tmp_path
        self.root = root
        self.state = state
        self.repo = repo
        self.monkeypatch = monkeypatch

    def arm(self, position: str, approval: dict[str, Any]) -> None:
        """Point the session at ``va`` at *position* under *approval*."""
        from osprey_connectors_tests._harness import write_control_context

        write_control_context(
            self.root, target="va", switch=None if position == "off" else {"va": position}
        )
        config = self.tmp_path / "config.yml"
        config.write_text(
            yaml.safe_dump({"approval": approval, "control_system": _SECTION}), encoding="utf-8"
        )
        self.monkeypatch.setenv("OSPREY_CONFIG", str(config))
        self.monkeypatch.setenv("CONFIG_FILE", str(config))
        reset_verdict_state()
        reset_config_cache()

    def hook(self, tool: str, call: dict[str, Any], tool_use_id: str) -> dict[str, Any] | None:
        """Run the approval hook on one call as Claude Code does; its JSON answer or ``None``."""
        payload = {
            "tool_name": tool,
            "tool_input": call,
            "hook_event_name": "PreToolUse",
            "tool_use_id": tool_use_id,
            "session_id": "conv-pair",
            "permission_mode": "default",
            "cwd": str(self.repo),
        }
        result = subprocess.run(
            [sys.executable, str(_HOOK / "osprey_approval.py")],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            env=dict(os.environ),
            cwd=str(self.repo),
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
        return json.loads(lines[-1]) if lines else None

    def stamp(self, tool_use_id: str) -> dict[str, Any] | None:
        path = self.state / f"approval_ask_{tool_use_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _decision(answer: dict[str, Any] | None) -> str | None:
    if answer is None:
        return None
    return answer.get("hookSpecificOutput", {}).get("permissionDecision")


def _server_channel_write(tool_use_id: str) -> None:
    """``channel_write``'s stage 1, in its own order, for the call *tool_use_id* names."""
    from osprey.audit.call import call_scope
    from osprey.mcp_server.control_system.tools import channel_write

    config = channel_write.load_osprey_config()
    with call_scope(tool_use_id, None):
        record = channel_write._read_record()
        channel_write._check_write_verdict(record, config)
        required = channel_write._check_approval_record(config)
        channel_write._check_approval_window(
            _OPERATIONS, None, channel_write._binding_of(record), required=required
        )


@pytest.mark.parametrize("approval", sorted(_APPROVALS))
@pytest.mark.parametrize("position", ["off", "asked", "allowed"])
def test_channel_write_needs_a_record_exactly_where_the_hook_leaves_one(
    deployment, position, approval
):
    from osprey.cli.build_posture_check import ask_capable
    from osprey.mcp_server.approval_record import require_approval_record

    deployment.arm(position, _APPROVALS[approval])
    tool_use_id = f"toolu_{position}_{approval.replace('-', '_')}"

    answer = deployment.hook(
        "mcp__controls__channel_write", {"operations": _OPERATIONS}, tool_use_id
    )
    left = deployment.stamp(tool_use_id) is not None

    config = {"approval": _APPROVALS[approval], "control_system": _SECTION}
    assert ask_capable(config, "channel_write"), "a governed write always needs its record"
    if position == "off":
        # Refused: the hook defers to the writes check and files nothing,
        # and the server refuses on the verdict before it looks for a record.
        assert answer is None and not left
        with pytest.raises(Exception, match="switch_off"):
            _server_channel_write(tool_use_id)
        return
    assert left, f"the hook at {position} left no record under {approval}"
    assert _decision(answer) == ("ask" if position == "asked" else "allow")
    _server_channel_write(tool_use_id)  # the hook's records satisfy every stage-1 check
    from osprey.audit.call import call_scope

    with call_scope("toolu_never_stamped", None):
        refusal = require_approval_record("channel_write", config)
    assert refusal is not None and refusal.error_type == "approval_missing"


@pytest.mark.parametrize("approval", sorted(_APPROVALS))
@pytest.mark.parametrize("tool", [name.rsplit("__", 1)[1] for name in _GOVERNED])
def test_every_governed_stage_1_reader_requires_a_record_whatever_approval_says(tool, approval):
    """The hook stamps every governed write it lets run, so every server must ask for it."""
    from osprey.cli.build_posture_check import ask_capable

    assert ask_capable({"approval": _APPROVALS[approval], "control_system": _SECTION}, tool)


@pytest.mark.parametrize(
    ("approval", "asks"), [("disabled", False), ("skip", False), ("always", True)]
)
def test_a_tool_that_keeps_its_policy_requires_a_record_only_where_it_asks(approval, asks):
    from osprey.cli.build_posture_check import ask_capable

    config = {"approval": _APPROVALS[approval], "control_system": _SECTION}
    assert ask_capable(config, "phoebus_drive") is asks


def test_an_allowed_channel_write_pairs_the_hooks_stamps_with_the_servers_checks(deployment):
    """Hook → server: at ``allowed`` the switch files the records a person's answer would."""
    deployment.arm("allowed", _APPROVALS["disabled"])

    answer = deployment.hook(
        "mcp__controls__channel_write", {"operations": _OPERATIONS}, "toolu_pair"
    )

    assert _decision(answer) == "allow"
    stamp = deployment.stamp("toolu_pair")
    assert stamp is not None
    assert (stamp["approver"], stamp["position"]) == ("write_switch", "allowed")
    assert sorted(deployment.state.glob("write_approval_*.json")), "no window stamp"
    _server_channel_write("toolu_pair")
