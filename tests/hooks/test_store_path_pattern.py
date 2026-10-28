"""The approval hook's store-path check, driven by the shared case table.

Executor code that names the posture store runs only on a person's approval.
The hook applies the rule ahead of the ``enabled``/``skip`` allows and the
selective-readonly allow: where the policy asks, the call is put to a person
and stamped; where it asks nobody, or in a dispatch run, it is denied. The
python executor applies the same rule again before it spawns; the parity test
(``tests/test_store_path_parity.py``) drives both from the same table.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey_connectors.refusals import STORE_PATH_MARKERS
from osprey_connectors.types import baseline_target
from osprey_connectors_tests._harness import write_control_context

#: The shared case table.
CASES_PATH = Path(__file__).parents[1] / "fixtures" / "store_path_cases.json"
CASES: list[dict[str, Any]] = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]

#: The rows the hook can judge: it sees ``execute``'s code, and only a path for
#: ``execute_file``.
HOOK_CASES = [case for case in CASES if case["tool"] == "execute"]

#: A deployment whose tier allows writes on both targets.
WRITES_ALLOWED_SECTION = {
    "type": "epics",
    "writes": "allowed",
    "connector": {"epics": {}, "virtual_accelerator": {}},
}

#: Every machine switched on.
SWITCHED_ON = {"live": "on", "va": "on", "standin": "on"}

#: The hook config: the executor's tools are write tools and asked about.
HOOK_CONFIG = {
    "server_prefixes": ["mcp__controls__", "mcp__python__"],
    "approval_prefixes": ["mcp__controls__", "mcp__python__"],
    "write_tools": ["mcp__controls__channel_write", "mcp__python__execute"],
}

#: Code with no marker and no write, the baseline an ``unchanged`` row is held to.
NEUTRAL_CODE = "x = 1\n"

#: The tool-use id every hook run asks under.
TOOL_USE_ID = "toolu_storepath1"

#: The executor's refusal, which the hook's deny restates word for word.
STORE_PATH_REASON = (
    "not run: this code names the posture store, which only a call a person "
    "approves may touch, and no person is asked here (approval_missing). Remove "
    "the store's name from the code, or run it where approval asks."
)


def case_params(cases: list[dict[str, Any]]) -> list[Any]:
    """*cases* as pytest params, each named by its id."""
    return [pytest.param(case, id=case["id"]) for case in cases]


def write_case_config(tmp_path: Path, case: dict[str, Any]) -> Path:
    """Write the row's ``config.yml`` and control-context record; return the config."""
    config = {
        "control_system": WRITES_ALLOWED_SECTION,
        "approval": {
            "enabled": case["enabled"],
            "default_policy": "always",
            "tools": {case["tool"]: case["policy"]},
        },
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    stamped = os.environ.get("OSPREY_AGENT_DATA_ROOT")
    root = Path(stamped) if stamped else tmp_path / "var" / "agent_data"
    write_control_context(
        root,
        target=baseline_target(WRITES_ALLOWED_SECTION),
        switch=SWITCHED_ON if case["switch"] == "on" else None,
    )
    return path


def run_case_hook(
    hook_runner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: dict[str, Any],
    *,
    code: str | None = None,
) -> dict | None:
    """Run the approval hook on the row's ``execute`` call; return its output."""
    if case["dispatch"]:
        monkeypatch.setenv("OSPREY_DISPATCH_RUN", "1")
    else:
        monkeypatch.delenv("OSPREY_DISPATCH_RUN", raising=False)
    config = write_case_config(tmp_path, case)
    return hook_runner(
        "osprey_approval.py",
        f"mcp__python__{case['tool']}",
        {
            "code": case["code"] if code is None else code,
            "execution_mode": case["execution_mode"],
            "description": "store-path case",
        },
        config_path=config,
        cwd=tmp_path,
        hook_config=HOOK_CONFIG,
        hook_input_extra={"tool_use_id": TOOL_USE_ID, "hook_event_name": "PreToolUse"},
    )


def decision(result: dict | None) -> str | None:
    """The hook's permission decision, ``None`` when it made none."""
    if not result:
        return None
    return result.get("hookSpecificOutput", {}).get("permissionDecision")


def ask_stamp(tmp_path: Path) -> Path | None:
    """The ask stamp the hook filed for :data:`TOOL_USE_ID` under *tmp_path*, if any."""
    found = sorted(tmp_path.rglob(f"approval_ask_{TOOL_USE_ID}.json"))
    return found[0] if found else None


def assert_hook_outcome(
    hook_runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: dict[str, Any]
) -> dict | None:
    """Run the row through the hook and assert the table's outcome; return the output."""
    result = run_case_hook(hook_runner, tmp_path, monkeypatch, case)
    expected = case["expected"]
    if expected == "ask":
        assert decision(result) == "ask", result
        assert ask_stamp(tmp_path) is not None
    elif expected == "deny":
        assert decision(result) == "deny", result
        reason = result["hookSpecificOutput"]["permissionDecisionReason"]
        if case["deny_reason"] == "store_path":
            assert reason == STORE_PATH_REASON
        else:
            assert reason != STORE_PATH_REASON
        assert ask_stamp(tmp_path) is None
    elif expected == "defer":
        assert result is None
        assert ask_stamp(tmp_path) is None
    else:
        assert expected == "unchanged"
        baseline_dir = tmp_path / "neutral"
        baseline_dir.mkdir()
        baseline = run_case_hook(hook_runner, baseline_dir, monkeypatch, case, code=NEUTRAL_CODE)
        assert decision(result) == decision(baseline), (result, baseline)
    return result


@pytest.mark.parametrize("case", case_params(HOOK_CASES))
def test_each_row_gives_the_tables_outcome(hook_runner, tmp_path, monkeypatch, case):
    assert_hook_outcome(hook_runner, tmp_path, monkeypatch, case)


@pytest.mark.parametrize(
    "approval",
    [
        pytest.param({"enabled": False}, id="approval-disabled"),
        pytest.param({"enabled": True, "tools": {"execute": "skip"}}, id="skip"),
        pytest.param({"enabled": True, "default_policy": "skip"}, id="default-skip"),
    ],
)
def test_no_early_allow_lets_store_naming_code_through(
    hook_runner, tmp_path, monkeypatch, approval
):
    monkeypatch.delenv("OSPREY_DISPATCH_RUN", raising=False)
    case = {"tool": "execute", "enabled": True, "policy": "always", "switch": "on"}
    config = write_case_config(tmp_path, case)
    config.write_text(
        yaml.safe_dump({"control_system": WRITES_ALLOWED_SECTION, "approval": approval}),
        encoding="utf-8",
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__python__execute",
        {"code": "print('control_context.json')\n", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
        hook_config=HOOK_CONFIG,
    )

    assert decision(result) == "deny"
    assert result["hookSpecificOutput"]["permissionDecisionReason"] == STORE_PATH_REASON


def test_the_ask_names_the_store(hook_runner, tmp_path, monkeypatch):
    case = next(c for c in HOOK_CASES if c["id"] == "marker-in-comment-selective-readonly")

    result = run_case_hook(hook_runner, tmp_path, monkeypatch, case)

    assert "Code names the posture store." in result["hookSpecificOutput"].get(
        "permissionDecisionReason", ""
    )


def test_the_deny_names_no_key_file_or_command():
    for marker in STORE_PATH_MARKERS:
        assert marker not in STORE_PATH_REASON
    for word in ("approval.", "enabled", "skip", "osprey ", "config", ".yml", ".json"):
        assert word not in STORE_PATH_REASON


def test_the_hook_reads_the_markers_from_the_twin(hook_module):
    module = hook_module("osprey_approval")

    assert module._store_path_markers() == STORE_PATH_MARKERS
    assert module._FALLBACK_STORE_PATH_MARKERS == STORE_PATH_MARKERS
    assert module._STORE_PATH_REASON == STORE_PATH_REASON


def test_the_hook_falls_back_to_the_restated_markers(hook_module, monkeypatch):
    module = hook_module("osprey_approval")
    monkeypatch.setattr(module, "_target_state", None)

    assert module._store_path_markers() == STORE_PATH_MARKERS
    assert module._names_store("open('control_context.json')")


@pytest.mark.parametrize(
    ("code", "named"),
    [
        ("p = 'control_target'", True),
        ("# control_context.json", True),
        ("/abs/path/control_target/x", True),
        ("CONTROL_TARGET", False),
        ("'control' + '_target'", False),
        ("", False),
        (None, False),
    ],
)
def test_the_match_is_a_case_sensitive_substring(hook_module, code, named):
    assert hook_module("osprey_approval")._names_store(code) is named
