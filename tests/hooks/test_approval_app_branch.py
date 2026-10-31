"""The approval hook's branch for an application's declared write tools.

An application declares, per write tool, the prompt a person approves, the state
route the prompt reads, and the approved fields the call is bound to. The render
writes that declaration into ``hook_config.json`` under ``app_approvals``, keyed
by the tool's full name. These tests run the real hook script as a subprocess
against a loopback stand-in for the web terminal's panel route.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from osprey.interfaces._serving import run_app_server
from osprey_connectors_tests._harness import pin_identity, state_dir_under, write_control_context
from tests.apps.test_manifest_schema import FULL
from tests.hooks.conftest import ask_reason, hook_audit_records, import_hook

HOOK = "osprey_approval.py"
TOOL = "mcp__knob__set_knob"
TOKEN = "panel-token-for-tests"
TOOL_USE_ID = "toolu_app_1"

#: The staging manifest's ``set_knob`` declaration, as the render writes it.
_DECLARED = FULL["tools"]["approval"][0]
ENTRY: dict[str, Any] = {
    "app": "knob",
    "panel_id": "knob",
    "state_path": _DECLARED["state_path"],
    "prompt": _DECLARED["prompt"],
    "display_lines_key": _DECLARED["display_lines_key"],
    "bind": copy.deepcopy(_DECLARED["bind"]),
    "require": copy.deepcopy(_DECLARED["require"]),
    "errors": dict(FULL["tools"]["errors"]),
}
UNREACHABLE = ENTRY["errors"]["unreachable"]
REFUSED = ENTRY["errors"]["refused"]

#: A step tool with no state route: one input shape, its digest in ``all_of``.
STEP_ENTRY: dict[str, Any] = {
    **ENTRY,
    "state_path": None,
    "display_lines_key": None,
    "prompt": "Step the knob",
    "bind": {
        "state_fields": [],
        "input_digests": [{"field": "approved_step_digest", "over": ["step"]}],
    },
    "require": {"all_of": ["approved_target", "approved_step_digest"], "one_of": []},
}

#: The value shape is a second tool, with its own digest and prompt.
VALUE_TOOL = "mcp__knob__apply_value"
VALUE_ENTRY: dict[str, Any] = {
    **STEP_ENTRY,
    "prompt": "Apply {{ input.value }}",
    "bind": {
        "state_fields": [],
        "input_digests": [{"field": "approved_value_digest", "over": ["value"]}],
    },
    "require": {"all_of": ["approved_target", "approved_value_digest"], "one_of": []},
}


def _hook_config(entry=ENTRY, *, gated=True, tool=TOOL, **extra) -> dict[str, Any]:
    return {
        "server_prefixes": ["mcp__knob__", "mcp__controls__"],
        "approval_prefixes": ["mcp__knob__", "mcp__controls__"],
        "write_tools": [tool, "mcp__controls__channel_write"],
        "facility_write_tools": [tool] if gated else [],
        "app_approvals": {tool: entry},
        **extra,
    }


APPROVAL = {"enabled": True, "default_policy": "always"}


class _Panel:
    """What the stand-in panel route answers, and what it was asked."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.state: Any = {"readback": 1.0, "lines": ["knob A: 1.0 -> 3.0"]}
        self.status = 200
        self.raw: bytes | None = None
        self.delay = 0.0
        self.authorizations: list[str | None] = []


@pytest.fixture(scope="module")
def _served() -> Iterator[tuple[int, _Panel]]:
    panel = _Panel()
    app = FastAPI()

    @app.get("/panel/knob/api/state")
    def _state(request: Request) -> Response:
        panel.authorizations.append(request.headers.get("authorization"))
        if panel.delay:
            time.sleep(panel.delay)
        if panel.raw is not None:
            return Response(panel.raw, media_type="application/json")
        return JSONResponse(panel.state, status_code=panel.status)

    with run_app_server(app) as url:
        yield int(url.rsplit(":", 1)[1]), panel


@pytest.fixture
def panel(_served, monkeypatch) -> _Panel:
    port, served = _served
    served.reset()
    monkeypatch.setenv("OSPREY_WEB_PORT", str(port))
    monkeypatch.setenv("OSPREY_PANEL_TOKEN", TOKEN)
    return served


@pytest.fixture
def deployment(tmp_path, monkeypatch, make_config, panel):  # noqa: ARG001 - panel serves the state
    """A session on ``va`` with its switch on, one pinned identity for both halves."""
    identity = pin_identity(monkeypatch)
    root = tmp_path / "agent_data"
    state = state_dir_under(root)
    state.mkdir(parents=True)
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(root))
    write_control_context(root, target="va", switch={"va": "on"})

    class _Deployment:
        who = identity
        state_dir = state
        record_root = root

        @staticmethod
        def config(approval=APPROVAL, **extra) -> Path:
            return make_config(
                {"approval": approval, "control_system": {"writes": "allowed"}, **extra}
            )

        @staticmethod
        def stamps() -> list[dict]:
            return [json.loads(p.read_text()) for p in sorted(state.glob("approval_ask_*.json"))]

    return _Deployment


def _run(
    hook_runner, deployment, tool_input, *, config=None, hook_config=None, args=None, tool=TOOL
):
    return hook_runner(
        HOOK,
        tool,
        tool_input,
        config_path=config or deployment.config(),
        cwd=deployment.record_root.parent,
        hook_config=hook_config or _hook_config(),
        hook_input_extra={
            "hook_event_name": "PreToolUse",
            "tool_use_id": TOOL_USE_ID,
            "session_id": "conv-app",
        },
        args=args or ("--budget", "15"),
    )


def _deny_reason(result) -> str:
    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    return output["permissionDecisionReason"]


def _digest(tool_input, over) -> str:
    return import_hook("osprey_prompt_render").input_digest(tool_input, over)


# --- the ask ------------------------------------------------------------------


def test_the_ask_shows_the_header_the_template_and_the_display_lines(
    hook_runner, deployment, panel
):
    result = _run(hook_runner, deployment, {"value": 3})

    reason = ask_reason(result)
    words = import_hook("osprey_approval")._KIND_WORDS.values()
    assert "Application: knob" in reason
    assert f"Tool: {TOOL}" in reason
    assert "Target: " in reason
    assert any(phrase in reason for phrase in words)
    assert "This call writes." in reason
    assert "Set 3" in reason
    assert "knob A: 1.0 -&gt; 3.0" in reason
    assert panel.authorizations == [f"Bearer {TOKEN}"]


def test_the_ask_binds_the_target_and_the_declared_fields(hook_runner, deployment):
    result = _run(hook_runner, deployment, {"value": 3})

    bound = {
        "approved_target": "va",
        "approved_readback": 1.0,
        "approved_digest": _digest({"value": 3}, ["value"]),
    }
    assert result["hookSpecificOutput"]["updatedInput"] == {"value": 3, **bound}
    (stamp,) = deployment.stamps()
    assert stamp["tool"] == TOOL
    assert stamp["approver"] == "human"
    assert {k: v for k, v in stamp.items() if k.startswith("approved_")} == bound


def test_agent_sent_approved_fields_are_stripped(hook_runner, deployment):
    sent = {"value": 3, "approved_target": "live", "approved_digest": "x", "approved_other": 1}

    result = _run(hook_runner, deployment, sent)

    updated = result["hookSpecificOutput"]["updatedInput"]
    assert updated == {
        "value": 3,
        "approved_target": "va",
        "approved_readback": 1.0,
        "approved_digest": _digest({"value": 3}, ["value"]),
    }
    (stamp,) = deployment.stamps()
    assert "approved_other" not in stamp


def test_a_state_field_binds_from_the_fetched_state(hook_runner, deployment, panel):
    panel.state = {"readback": 2.5, "lines": []}

    result = _run(hook_runner, deployment, {"value": 3})

    assert result["hookSpecificOutput"]["updatedInput"]["approved_readback"] == 2.5


# --- binding and require ------------------------------------------------------


def test_the_ask_binds_every_all_of_field(hook_runner, deployment):
    result = _run(hook_runner, deployment, {"step": 1}, hook_config=_hook_config(STEP_ENTRY))

    updated = result["hookSpecificOutput"]["updatedInput"]
    assert updated == {
        "step": 1,
        "approved_target": "va",
        "approved_step_digest": _digest({"step": 1}, ["step"]),
    }


def test_a_second_input_shape_is_a_second_tool(hook_runner, deployment):
    hook_config = _hook_config(VALUE_ENTRY, tool=VALUE_TOOL)

    result = _run(hook_runner, deployment, {"value": 2}, hook_config=hook_config, tool=VALUE_TOOL)

    assert "Apply 2" in ask_reason(result)
    updated = result["hookSpecificOutput"]["updatedInput"]
    assert updated["approved_value_digest"] == _digest({"value": 2}, ["value"])
    assert "approved_step_digest" not in updated


def test_a_digest_over_a_key_the_call_lacks_is_refused_before_any_prompt(hook_runner, deployment):
    result = _run(hook_runner, deployment, {"value": 2}, hook_config=_hook_config(STEP_ENTRY))

    reason = _deny_reason(result)
    assert "`approved_step_digest` is bound over `step`, which the call does not carry" in reason
    assert REFUSED in reason
    assert deployment.stamps() == []


def test_an_absent_state_key_is_unbound_and_the_unmet_require_refuses(
    hook_runner, deployment, panel
):
    """No ``readback`` in the state: ``approved_readback`` is unbound, its one_of group empty."""
    panel.state = {"lines": []}

    result = _run(hook_runner, deployment, {"value": 3})

    reason = _deny_reason(result)
    assert "approved_readback" in reason
    assert REFUSED in reason
    assert deployment.stamps() == []


# --- the binding rule, against the application SDK's -------------------------------

#: ``(id, tool input, over)``: one digest bound over *over* of the call's input.
_DIGEST_CASES = [
    ("one-key", {"step": 1}, ["step"]),
    ("two-keys", {"a": 1, "b": [2, 3], "c": "x"}, ["a", "b"]),
    ("nested", {"plan": {"knobs": {"A": 1.5}}}, ["plan"]),
    ("missing-key", {"value": 2}, ["step"]),
    ("missing-one-of-two", {"a": 1}, ["a", "b"]),
    ("missing-both", {}, ["a", "b"]),
    ("not-json", {"step": float("nan")}, ["step"]),
]


@pytest.mark.parametrize(
    ("tool_input", "over"),
    [case[1:] for case in _DIGEST_CASES],
    ids=[case[0] for case in _DIGEST_CASES],
)
def test_the_hook_binds_a_digest_exactly_when_the_sdk_recomputes_it(approval, tool_input, over):
    """A digest the hook binds is the one the SDK recomputes; one it refuses, the SDK refuses."""
    from osprey.apps.sdk._approval import _recomputed
    from osprey.mcp_server.approval_record import Refusal

    field = "approved_step_digest"
    entry = {"bind": {"input_digests": [{"field": field, "over": over}]}}

    sdk = _recomputed(tool_input, field, over)
    try:
        hook = approval._app_binding(entry, tool_input, {}, "va")[field]
    except approval._AppRefused as refused:
        assert refused.kind == "refused"
        assert isinstance(sdk, Refusal)
        if any(key not in tool_input for key in over):
            assert sdk.message.startswith(f"not run: {refused.detail} (")
    else:
        assert hook == sdk


# --- fail closed ----------------------------------------------------------------


def test_a_failed_fetch_is_refused_as_unreachable(hook_runner, deployment, panel):
    panel.status = 503

    assert UNREACHABLE in _deny_reason(_run(hook_runner, deployment, {"value": 3}))
    assert deployment.stamps() == []


def test_a_fetch_past_the_budget_is_refused_as_unreachable(hook_runner, deployment, panel):
    panel.delay = 3.0

    result = _run(hook_runner, deployment, {"value": 3}, args=("--budget", "4"))

    assert UNREACHABLE in _deny_reason(result)
    assert deployment.stamps() == []


def test_an_oversized_state_is_refused_as_unreachable(hook_runner, deployment, panel):
    cap = import_hook("osprey_approval")._APP_STATE_MAX_BYTES
    panel.raw = json.dumps({"lines": [], "pad": "a" * cap}).encode()

    assert UNREACHABLE in _deny_reason(_run(hook_runner, deployment, {"value": 3}))


def test_no_panel_token_is_refused_as_unreachable(hook_runner, deployment, panel, monkeypatch):
    monkeypatch.delenv("OSPREY_PANEL_TOKEN")

    assert UNREACHABLE in _deny_reason(_run(hook_runner, deployment, {"value": 3}))
    assert panel.authorizations == []


def test_a_tool_outside_the_facility_write_set_is_refused_unasked(
    hook_runner, deployment, panel, tmp_path
):
    """No write verdict ran for it, so no person may be asked."""
    result = _run(hook_runner, deployment, {"value": 3}, hook_config=_hook_config(gated=False))

    assert REFUSED in _deny_reason(result)
    assert panel.authorizations == []
    assert deployment.stamps() == []
    (record,) = hook_audit_records(tmp_path, "hook_approval")
    assert (record["decision"], record["reason"]) == ("refused", "app_not_gated")


def test_missing_prompt_helpers_are_refused_as_unreachable(approval, monkeypatch, capsys):
    monkeypatch.setattr(approval, "_prompt_render", None)
    monkeypatch.setattr(approval, "load_hook_config", _hook_config)

    with pytest.raises(SystemExit):
        approval._app_approval({"tool_name": TOOL}, TOOL, {"value": 3}, {}, {}, ENTRY)

    assert UNREACHABLE in _deny_reason(json.loads(capsys.readouterr().out))


# --- policy -------------------------------------------------------------------


def test_approval_off_allows_before_the_branch(hook_runner, deployment, panel):
    result = _run(
        hook_runner, deployment, {"value": 3}, config=deployment.config({"enabled": False})
    )

    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert panel.authorizations == []
    assert deployment.stamps() == []


def test_a_skip_policy_keyed_by_full_name_allows_with_no_stamp(hook_runner, deployment, panel):
    config = deployment.config({**APPROVAL, "tools": {TOOL: "skip"}})

    result = _run(hook_runner, deployment, {"value": 3}, config=config)

    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert panel.authorizations == []
    assert deployment.stamps() == []


def test_an_invalid_approval_section_denies(hook_runner, deployment):
    hook = import_hook("osprey_approval")

    result = _run(hook_runner, deployment, {"value": 3}, config=deployment.config("nope"))

    assert _deny_reason(result) == hook._APPROVAL_SECTION_INVALID_REASON.format(value="nope")


# --- posture is decided before the branch ------------------------------------------


@pytest.mark.parametrize("refusal", ["switch_off", "tier_never", "owner_unscheduled"])
def test_a_refused_write_defers_before_the_branch(
    hook_runner, deployment, panel, make_config, monkeypatch, refusal
):
    config = deployment.config()
    if refusal == "switch_off":
        write_control_context(deployment.record_root, target="va", switch={"va": "off"})
    elif refusal == "tier_never":
        config = make_config({"approval": APPROVAL, "control_system": {"writes": "never"}})
    else:
        monkeypatch.setenv("OSPREY_DISPATCH_RUN", "1")

    assert _run(hook_runner, deployment, {"value": 3}, config=config) is None
    assert panel.authorizations == []
    assert deployment.stamps() == []


# --- dispatch jobs ------------------------------------------------------------


@pytest.fixture
def job(deployment, monkeypatch):
    monkeypatch.setenv("OSPREY_DISPATCH_RUN", "1")
    monkeypatch.setenv("OSPREY_CONTROL_OWNER", deployment.who)
    return deployment


def test_a_job_refuses_a_tool_that_binds_fetched_state(hook_runner, job, panel):
    config = job.config(dispatch={"denied_app_tools": []})

    reason = _deny_reason(_run(hook_runner, job, {"value": 3}, config=config))

    assert "dispatch job" in reason
    assert panel.authorizations == []


@pytest.mark.parametrize("denied", [[TOOL], None], ids=["listed", "list-missing"])
def test_a_job_refuses_a_tool_the_render_denies_in_jobs(hook_runner, job, denied):
    extra = {} if denied is None else {"dispatch": {"denied_app_tools": denied}}
    hook_config = _hook_config(STEP_ENTRY)

    result = _run(
        hook_runner, job, {"step": 1}, config=job.config(**extra), hook_config=hook_config
    )

    assert "dispatch job" in _deny_reason(result)
    assert job.stamps() == []


def test_a_job_asks_with_the_dispatch_policy_as_approver(hook_runner, job):
    config = job.config(dispatch={"denied_app_tools": []})

    result = _run(
        hook_runner, job, {"step": 1}, config=config, hook_config=_hook_config(STEP_ENTRY)
    )

    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "ask"
    assert output["updatedInput"]["approved_target"] == "va"
    (stamp,) = job.stamps()
    assert stamp["approver"] == "dispatch_policy"
    assert stamp["approved_step_digest"] == _digest({"step": 1}, ["step"])


def test_a_job_never_emits_an_allow(hook_runner, job):
    config = job.config({**APPROVAL, "tools": {TOOL: "skip"}}, dispatch={"denied_app_tools": []})

    result = _run(
        hook_runner, job, {"step": 1}, config=config, hook_config=_hook_config(STEP_ENTRY)
    )

    assert not result or "hookSpecificOutput" not in result


# --- other tools --------------------------------------------------------------


def test_other_tools_answer_exactly_as_without_the_key(hook_runner, deployment):
    """A framework tool's prompt is byte-identical whether or not applications are declared."""
    with_apps = _hook_config()
    without = {k: v for k, v in with_apps.items() if k != "app_approvals"}

    def ask(hook_config):
        return hook_runner(
            HOOK,
            "mcp__controls__channel_write",
            {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
            config_path=deployment.config(),
            cwd=deployment.record_root.parent,
            hook_config=hook_config,
        )

    assert ask(with_apps) == ask(without)
