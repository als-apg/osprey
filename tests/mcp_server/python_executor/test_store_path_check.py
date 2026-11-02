"""The executor's store-path check: store-naming code runs only on a person's approval.

Checked before spawn, in every execution mode, whether or not the approval hook
ran. Code that names the posture store runs only when the hook's ask stamp for
the call names the tool and was answered by a person; a ``write_switch`` stamp
(filed at ``allowed`` with nobody asked) never lets it run, and a dispatch run
is refused whatever the stamp. ``execute`` and ``execute_file`` are governed, so
``approval.enabled: false`` and a ``skip`` policy have no say: the call is asked
in every switch position, and without a person's stamp it is refused. The rows of
``tests/fixtures/store_path_cases.json`` hold the outcomes; the parity test
(``tests/test_store_path_parity.py``) drives the real hook against the same rows.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from osprey.mcp_server.approval_record import APPROVAL_MISSING, Refusal
from osprey.mcp_server.python_executor import executor
from osprey.mcp_server.python_executor.executor import (
    FAILURE_KIND_APPROVAL_CONFIG_INVALID,
    FAILURE_KIND_APPROVAL_MISSING,
    STORE_PATH_REFUSAL,
    ExecutionResult,
)

CASES_PATH = Path(__file__).parents[3] / "tests" / "fixtures" / "store_path_cases.json"
CASES: list[dict[str, Any]] = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]

#: The tool-use id every call runs under.
CALL_ID = "toolu_store_exec1"


def case_config(case: dict[str, Any]) -> dict[str, Any]:
    """The rendered config a row runs under."""
    return {
        "control_system": {"type": "epics", "connector": {"epics": {}}},
        "approval": {
            "enabled": case["enabled"],
            "default_policy": "always",
            "tools": {case["tool"]: case["policy"]},
        },
    }


@pytest.fixture
def stamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The state directory ask stamps are read from; no dispatch run unless a test says so."""
    from osprey.mcp_server.control_system import target_state

    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(target_state, "state_dir", lambda: state)
    monkeypatch.delenv("OSPREY_DISPATCH_RUN", raising=False)
    return state


def stamp(state: Path, tool: str, approver: str = "human") -> None:
    """File the ask stamp for :data:`CALL_ID`, recording *tool*."""
    payload = {"tool_use_id": CALL_ID, "tool": f"mcp__python__{tool}", "approver": approver}
    (state / f"approval_ask_{CALL_ID}.json").write_text(json.dumps(payload), encoding="utf-8")


def check(code: str, config: dict[str, Any], tool: str = "execute") -> Refusal | None:
    """Run the store-path check for one call under :data:`CALL_ID`."""
    from osprey.audit.call import call_scope

    with call_scope(CALL_ID, None):
        return executor._check_store_path(code, config, tool)


def refused(code: str, config: dict[str, Any], tool: str = "execute") -> str:
    """The message of the ``approval_missing`` refusal :func:`check` answers."""
    refusal = check(code, config, tool)
    assert refusal is not None and refusal.error_type == APPROVAL_MISSING
    return refusal.message


def launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    code: str,
    *,
    mode: str = "readonly",
    tool: str = "execute",
) -> tuple[ExecutionResult, bool]:
    """Launch one call through the real pre-spawn path; return its result and whether it spawned."""
    from osprey.audit.call import call_scope

    spawned: list[bool] = []

    class _Proc:
        returncode = 0
        pid = 4244

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def spawn(*_args: Any, **_kwargs: Any) -> _Proc:
        spawned.append(True)
        return _Proc()

    def stamp_target(sandbox_env: dict[str, str]) -> str:
        sandbox_env["OSPREY_CONTROL_TARGET"] = "live"
        return "live"

    folder = tmp_path / "exec"
    (folder / "figures").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(executor, "_resolve_project_root", lambda: tmp_path)
    monkeypatch.setattr("osprey_connectors.workspace.load_osprey_config", lambda: config)
    monkeypatch.setattr(executor, "_apply_target_stamp", stamp_target)
    monkeypatch.setattr(executor, "_load_limits_validator", lambda target: None)
    monkeypatch.setattr(executor, "resolve_agent_interpreter", lambda root=None: "/bin/true")
    monkeypatch.setattr(executor.asyncio, "create_subprocess_exec", spawn)

    async def run() -> ExecutionResult:
        with call_scope(CALL_ID, None):
            return await executor._execute_via_local(
                code, mode, {"timeout": 30}, folder, tool=tool, approved_target="live"
            )

    result = asyncio.run(run())
    return result, bool(spawned)


# -- the table ----------------------------------------------------------------


@pytest.mark.parametrize("case", [pytest.param(c, id=c["id"]) for c in CASES])
@pytest.mark.usefixtures("stamps")
def test_each_row_without_a_stamp(monkeypatch, case: dict[str, Any]) -> None:
    """Without a stamp only an ``unchanged`` row passes the check."""
    if case["dispatch"]:
        monkeypatch.setenv("OSPREY_DISPATCH_RUN", "1")
    if case["expected"] == "unchanged":
        assert check(case["code"], case_config(case), case["tool"]) is None
        return
    assert refused(case["code"], case_config(case), case["tool"]) == STORE_PATH_REFUSAL


@pytest.mark.parametrize("case", [pytest.param(c, id=c["id"]) for c in CASES])
def test_each_row_with_a_stamp(stamps: Path, monkeypatch, case: dict[str, Any]) -> None:
    """With a stamp, only a row the store-path rule itself denies stays refused.

    A ``defer`` or raw-client ``deny`` row is refused by another gate, which
    files no stamp; a stamp is forged here only to show the store-path rule
    does not refuse it again.
    """
    if case["dispatch"]:
        monkeypatch.setenv("OSPREY_DISPATCH_RUN", "1")
    stamp(stamps, case["tool"])
    if case.get("deny_reason") != "store_path" or not case["dispatch"]:
        # The row's hook outcome (a ``skip`` or disabled approval read as
        # nobody-asked) is the ungoverned spelling; the executor reads
        # ``execute``/``execute_file`` as governed, so a person's stamp is the
        # approval the call was asked for.
        assert check(case["code"], case_config(case), case["tool"]) is None
        return
    assert refused(case["code"], case_config(case), case["tool"]) == STORE_PATH_REFUSAL


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c["id"]) for c in CASES if c.get("deny_reason") == "store_path"],
)
def test_each_store_path_row_refuses_a_switch_stamp(
    stamps: Path, monkeypatch, case: dict[str, Any]
) -> None:
    """A stamp nobody answered (``write_switch``) never lets store-naming code run."""
    if case["dispatch"]:
        monkeypatch.setenv("OSPREY_DISPATCH_RUN", "1")
    stamp(stamps, case["tool"], approver="write_switch")
    assert refused(case["code"], case_config(case), case["tool"]) == STORE_PATH_REFUSAL


# -- named guarantees -----------------------------------------------------------

STORE_CODE = "print(open('_agent_data/control_target/default/control_context.json').read())\n"


@pytest.mark.parametrize(
    "approval",
    [
        pytest.param({"enabled": False}, id="approval-disabled"),
        pytest.param({"enabled": True, "tools": {"execute": "skip"}}, id="skip"),
    ],
)
@pytest.mark.parametrize("approver", [None, "write_switch"], ids=["no-stamp", "switch-stamp"])
def test_store_naming_code_is_refused_without_a_person_whatever_the_policy(
    stamps: Path, approval: dict[str, Any], approver: str | None
) -> None:
    """``execute`` is governed: ``approval.*`` cannot waive the person's stamp."""
    if approver is not None:
        stamp(stamps, "execute", approver=approver)
    assert refused(STORE_CODE, {"approval": approval}) == STORE_PATH_REFUSAL


@pytest.mark.parametrize(
    "approval",
    [
        pytest.param({"enabled": False}, id="approval-disabled"),
        pytest.param({"enabled": True, "tools": {"execute": "skip"}}, id="skip"),
    ],
)
def test_store_naming_code_runs_on_a_persons_stamp_whatever_the_policy(
    stamps: Path, approval: dict[str, Any]
) -> None:
    """The governed hook asks in every position, so a person's stamp is a real approval."""
    stamp(stamps, "execute")
    assert check(STORE_CODE, {"approval": approval}) is None


@pytest.mark.usefixtures("stamps")
def test_a_readonly_run_without_a_stamp_is_refused_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {"approval": {"enabled": True, "default_policy": "always"}}

    result, spawned = launch(tmp_path, monkeypatch, config, STORE_CODE)

    assert not spawned
    assert result.success is False
    assert result.failure_kind == FAILURE_KIND_APPROVAL_MISSING
    assert result.error_message == STORE_PATH_REFUSAL


def test_a_readonly_run_with_a_stamp_proceeds(
    stamps: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp(stamps, "execute")
    config = {"approval": {"enabled": True, "default_policy": "always"}}

    result, spawned = launch(tmp_path, monkeypatch, config, STORE_CODE)

    assert spawned
    assert result.failure_kind != FAILURE_KIND_APPROVAL_MISSING


@pytest.mark.usefixtures("stamps")
def test_execute_file_is_checked_on_the_files_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {"approval": {"enabled": True, "tools": {"execute_file": "skip"}}}

    result, spawned = launch(tmp_path, monkeypatch, config, STORE_CODE, tool="execute_file")

    assert not spawned
    assert result.failure_kind == FAILURE_KIND_APPROVAL_MISSING


def test_a_stamp_for_another_tool_does_not_satisfy_it(stamps: Path) -> None:
    stamp(stamps, "execute_file")
    assert refused(STORE_CODE, {"approval": {"enabled": True}}) == STORE_PATH_REFUSAL


@pytest.mark.parametrize("approver", ["dispatch_policy", "human"])
def test_a_dispatch_run_is_refused_whatever_the_stamp(
    stamps: Path, monkeypatch: pytest.MonkeyPatch, approver: str
) -> None:
    monkeypatch.setenv("OSPREY_DISPATCH_RUN", "1")
    stamp(stamps, "execute", approver=approver)
    assert refused(STORE_CODE, {"approval": {"enabled": True}}) == STORE_PATH_REFUSAL


@pytest.mark.parametrize("value", [None, "false", 0])
@pytest.mark.usefixtures("stamps")
def test_a_non_bool_switch_refuses_store_naming_code_as_a_config_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: Any
) -> None:
    config = {"approval": {"enabled": value}}

    result, spawned = launch(tmp_path, monkeypatch, config, STORE_CODE)

    assert not spawned
    assert result.failure_kind == FAILURE_KIND_APPROVAL_CONFIG_INVALID


@pytest.mark.parametrize(
    "approval",
    [
        pytest.param({"enabled": False}, id="approval-disabled"),
        pytest.param({"enabled": True, "tools": {"execute": "skip"}}, id="skip"),
        pytest.param({"enabled": True, "default_policy": "always"}, id="always"),
    ],
)
@pytest.mark.usefixtures("stamps")
def test_code_with_no_marker_runs_exactly_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, approval: dict[str, Any]
) -> None:
    """A readonly run naming no store is spawned with no stamp under every policy."""
    result, spawned = launch(tmp_path, monkeypatch, {"approval": approval}, "print(1)\n")

    assert spawned
    assert result.failure_kind is None or result.failure_kind != FAILURE_KIND_APPROVAL_MISSING


@pytest.mark.usefixtures("stamps")
def test_other_tools_are_not_checked() -> None:
    assert check(STORE_CODE, {"approval": {"enabled": False}}, tool="pyaml_measure") is None


def test_the_executor_reads_the_shared_markers() -> None:
    from osprey_connectors.refusals import STORE_PATH_MARKERS

    for marker in STORE_PATH_MARKERS:
        assert executor.names_store(f"x = '{marker}'")
    assert not executor.names_store("x = 'CONTROL_TARGET'")
