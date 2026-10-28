"""The approval hook and the executor apply one store-path rule.

Every row of ``tests/fixtures/store_path_cases.json`` runs through the real
approval hook (a subprocess, as Claude Code runs it), then through the
executor's pre-spawn check with whatever ask stamp the hook left. The hook's
outcome and the executor's must agree: a call the hook asks about runs on the
hook's own stamp; a call it denies or defers leaves no stamp and is refused;
code naming no store passes the executor's check. ``execute_file`` rows give the
hook a path only, so there the executor's check alone holds the rule, run on
whatever stamp the hook's ordinary policy left.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.mcp_server.python_executor import executor
from tests.hooks import conftest as _hooks
from tests.hooks.test_store_path_pattern import (
    CASES,
    HOOK_CASES,
    STORE_PATH_REASON,
    TOOL_USE_ID,
    ask_stamp,
    case_params,
    decision,
    run_case_hook,
)

# The hook suite's runner and its isolation: a curated subprocess environment,
# a throwaway home and repo, and blank config memos.
hook_runner = _hooks.hook_runner
hook_home = _hooks.hook_home
_throwaway_repo = _hooks._throwaway_repo
_isolated_home = _hooks._isolated_home
_hook_root_never_the_checkout = _hooks._hook_root_never_the_checkout
_no_session_posture = _hooks._no_session_posture
_reset_hook_log_config_caches = _hooks._reset_hook_log_config_caches

FILE_CASES = [case for case in CASES if case["tool"] == "execute_file"]


def executor_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: dict[str, Any], code: str
) -> str | None:
    """The executor check's refusal for the row on the stamp the hook left, ``None`` if it passes."""
    from osprey.audit.call import call_scope
    from osprey.mcp_server.control_system import target_state

    stamp = ask_stamp(tmp_path)
    state = stamp.parent if stamp is not None else tmp_path / "no-stamps"
    state.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(target_state, "state_dir", lambda: state)
    config = yaml.safe_load((tmp_path / "config.yml").read_text(encoding="utf-8"))
    with call_scope(TOOL_USE_ID, None):
        refusal = executor._check_store_path(code, config, case["tool"])
    return None if refusal is None else refusal.message


@pytest.mark.parametrize("case", case_params(HOOK_CASES))
def test_hook_and_executor_agree_on_every_execute_row(hook_runner, tmp_path, monkeypatch, case):
    result = run_case_hook(hook_runner, tmp_path, monkeypatch, case)
    hook = decision(result)
    refusal = executor_refusal(tmp_path, monkeypatch, case, case["code"])

    expected = case["expected"]
    if expected == "ask":
        assert hook == "ask", result
        assert refusal is None
    elif expected in ("deny", "defer"):
        assert hook == ("deny" if expected == "deny" else None), result
        assert refusal == STORE_PATH_REASON
        if case.get("deny_reason") == "store_path":
            assert result["hookSpecificOutput"]["permissionDecisionReason"] == refusal
    else:
        assert expected == "unchanged"
        assert hook != "deny", result
        assert refusal is None


@pytest.mark.parametrize("case", case_params(FILE_CASES))
def test_the_executor_holds_the_rule_for_execute_file(hook_runner, tmp_path, monkeypatch, case):
    result = run_case_hook(hook_runner, tmp_path, monkeypatch, case)
    refusal = executor_refusal(tmp_path, monkeypatch, case, case["code"])

    assert decision(result) != "deny", result
    if case["expected"] in ("ask", "unchanged"):
        assert refusal is None
    else:
        assert refusal == STORE_PATH_REASON
