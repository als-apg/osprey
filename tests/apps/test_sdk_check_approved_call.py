"""``osprey.apps.sdk.check_approved_call``: the stage an application tool server runs first.

The check runs main's write chain for one application tool call, in order:
the render must offer the tool, the write verdict must allow the session's
target, the ask stamp for the call must record the values the call carries
(with every input digest recomputed over the arguments), and the stamp's
target must be the target the session is on now. Each test lays down a real
deployment under ``tmp_path``: a rendered ``config.yml``, a control-context
record and, where the policy asks, the stamp the approval hook files.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.apps.sdk import ApprovalRefused, BoundState, check_approved_call
from osprey.apps.sdk._approval import input_digest
from osprey.audit.posture import POSTURE_SESSION_ENV_VAR
from osprey_connectors import control_context, posture_store
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors.workspace import reset_config_cache
from osprey_connectors_tests._harness import (
    isolate_verdict_environment,
    state_dir_under,
    write_control_context,
)
from tests._approval_stamp_fixture import ApprovalStamps, use_stamp_dir
from tests.apps.test_manifest_schema import FULL

APP = "knob"
TOOL = "set_knob"
FULL_NAME = f"mcp__{APP}__{TOOL}"
REFUSED_GUIDANCE = FULL["tools"]["errors"]["refused"]


def _entry() -> dict[str, Any]:
    """The staging manifest fixture's ``set_knob`` approval, as the render writes it.

    Its input digest is required, as the manifest rules require of every bound digest.
    """
    approval = copy.deepcopy(FULL["tools"]["approval"][0])
    approval.pop("tool")
    approval["require"] = {"all_of": ["approved_target", "approved_digest"]}
    return {
        "app": APP,
        "panel_id": "knob",
        **approval,
        "errors": copy.deepcopy(FULL["tools"]["errors"]),
    }


def _config(policy: str = "always", *, switch_enabled: bool = True) -> dict[str, Any]:
    return {
        "control_system": {
            "type": "virtual_accelerator",
            "writes": "allowed",
            "connector": {
                "virtual_accelerator": {"serving": "in_process"},
                "epics": {"prefix": "RING:"},
            },
        },
        "approval": {"enabled": switch_enabled, "tools": {FULL_NAME: policy}},
        "apps": {APP: {"approval": {TOOL: _entry()}}},
    }


@pytest.fixture
def deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A rendered deployment with a record on ``va``, every switch on, and a stamp dir."""
    root = tmp_path / "var" / "agent_data"
    isolate_verdict_environment(monkeypatch, root)
    state_dir_under(root).mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv(POSTURE_SESSION_ENV_VAR, raising=False)
    monkeypatch.setenv("OSPREY_APP_NAME", APP)
    monkeypatch.chdir(tmp_path)

    def write_config(config: dict[str, Any]) -> None:
        (tmp_path / "config.yml").write_text(yaml.safe_dump(config))
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
        reset_config_cache()

    def write_record(target: str = "va", switch: dict[str, str] | None = None) -> None:
        write_control_context(
            posture_store.agent_data_root(),
            target,
            1,
            switch={"va": "on", "live": "on"} if switch is None else switch,
        )
        control_context.invalidate_cache()

    write_config(_config())
    write_record()
    stamps = use_stamp_dir(monkeypatch, tmp_path / "stamps")
    yield _Deployment(stamps, write_config, write_record)
    reset_config_cache()


class _Deployment:
    def __init__(self, stamps: ApprovalStamps, write_config, write_record) -> None:
        self.stamps = stamps
        self.write_config = write_config
        self.write_record = write_record


VALUE = {"value": 4.2}
DIGEST = input_digest(VALUE, ["value"])


def _call(arguments: dict[str, Any]) -> BoundState:
    return check_approved_call(TOOL, arguments)


def _refused(arguments: dict[str, Any]) -> ApprovalRefused:
    with pytest.raises(ApprovalRefused) as info:
        _call(arguments)
    return info.value


def _stamped(deployment: _Deployment, *, tool: str = FULL_NAME, approver: str = "human", **fields):
    fields.setdefault("approved_target", "va")
    return deployment.stamps.approved(tool, approver=approver, **fields)


# --- Approved calls -------------------------------------------------------------


def test_an_approved_digest_call_returns_the_bound_state(deployment):
    with _stamped(deployment, approved_digest=DIGEST):
        state = _call({**VALUE, "approved_target": "va", "approved_digest": DIGEST})
    assert state == BoundState(
        bound=True,
        approved_target="va",
        state_fields={},
        input_digests={"approved_digest": DIGEST},
        approver="human",
    )


def test_an_approved_readback_call_returns_the_stamped_state_fields(deployment):
    with _stamped(deployment, approved_digest=DIGEST, approved_readback="4.0"):
        state = _call(
            {
                **VALUE,
                "approved_target": "va",
                "approved_digest": DIGEST,
                "approved_readback": "4.0",
            }
        )
    assert state.bound is True
    assert state.state_fields == {"approved_readback": "4.0"}
    assert state.input_digests == {"approved_digest": DIGEST}


def test_dispatch_policy_stamp_accepted(deployment):
    with _stamped(deployment, approver="dispatch_policy", approved_digest=DIGEST):
        state = _call({**VALUE, "approved_target": "va", "approved_digest": DIGEST})
    assert state.approver == "dispatch_policy"
    assert state.approved_target == "va"
    assert state.input_digests == {"approved_digest": DIGEST}


def test_policy_skip_no_stamp_unbound(deployment):
    deployment.write_config(_config("skip"))
    state = _call({**VALUE})
    assert state == BoundState(bound=False)


def test_approval_switched_off_no_stamp_unbound(deployment):
    deployment.write_config(_config(switch_enabled=False))
    assert _call({**VALUE}).bound is False


# --- Refusals, in the order the chain runs them -------------------------------


def test_not_offered_refused(deployment):
    config = _config()
    del config["apps"][APP]["approval"][TOOL]
    deployment.write_config(config)
    with _stamped(deployment, approved_digest=DIGEST):
        refusal = _refused({**VALUE, "approved_target": "va", "approved_digest": DIGEST})
    assert refusal.reason == "not_offered"
    assert FULL_NAME in refusal.message


@pytest.mark.usefixtures("deployment")
def test_not_offered_without_an_app_name(monkeypatch):
    monkeypatch.delenv("OSPREY_APP_NAME")
    assert _refused({**VALUE}).reason == "not_offered"


def test_verdict_refuses_before_the_record_switch_off(deployment, monkeypatch):
    deployment.write_record(switch={"va": "off", "live": "off"})
    from osprey.mcp_server import approval_record

    def no_stamp_read(_tool_use_id):
        raise AssertionError("the stamp was read after a refused verdict")

    monkeypatch.setattr(approval_record, "approval_ask_stamp", no_stamp_read)
    refusal = _refused({**VALUE, "approved_target": "va", "approved_digest": DIGEST})
    assert refusal.reason == "switch_off"
    assert refusal_phrase("switch_off") in refusal.message
    assert f"{TOOL} was refused before reaching the application." in refusal.message


@pytest.mark.usefixtures("deployment")
def test_unreachable_verdict_refuses_as_record_unavailable(monkeypatch):
    from osprey.apps.sdk import _approval

    def broken(*_args, **_kwargs):
        raise OSError("record store gone")

    monkeypatch.setattr(_approval, "write_verdict", broken)
    assert _refused({**VALUE}).reason == "record_unavailable"


@pytest.mark.usefixtures("deployment")
def test_policy_always_no_stamp_refused():
    refusal = _refused({**VALUE, "approved_target": "va", "approved_digest": DIGEST})
    assert refusal.reason == "approval_missing"
    assert refusal.message.endswith(REFUSED_GUIDANCE)


def test_policy_selective_no_stamp_refused(deployment):
    deployment.write_config(_config("selective"))
    refusal = _refused({**VALUE, "approved_target": "va", "approved_digest": DIGEST})
    assert refusal.reason == "approval_missing"


def test_stale_stamp_refused(deployment):
    deployment.stamps.ask(FULL_NAME, tool_use_id="toolu_other", approved_target="va")
    from osprey.audit.call import call_scope

    with call_scope("toolu_this_call", None):
        refusal = _refused({**VALUE, "approved_target": "va", "approved_digest": DIGEST})
    assert refusal.reason == "approval_missing"


def test_stamp_for_another_apps_same_named_tool_refused(deployment):
    with _stamped(deployment, tool=f"mcp__other__{TOOL}", approved_digest=DIGEST):
        refusal = _refused({**VALUE, "approved_target": "va", "approved_digest": DIGEST})
    assert refusal.reason == "approval_missing"


def test_recomputed_digest_that_differs_refuses(deployment):
    with _stamped(deployment, approved_digest=DIGEST):
        refusal = _refused({"value": 9.9, "approved_target": "va", "approved_digest": DIGEST})
    assert refusal.reason == "approval_missing"
    assert "approved_digest" in refusal.message


def test_bound_state_differs_refuses(deployment):
    with _stamped(deployment, approved_digest=DIGEST, approved_readback="4.0"):
        refusal = _refused(
            {
                **VALUE,
                "approved_target": "va",
                "approved_digest": DIGEST,
                "approved_readback": "5.0",
            }
        )
    assert refusal.reason == "approval_missing"
    assert "approved_readback" in refusal.message


def test_missing_digest_key_refused(deployment):
    with _stamped(deployment, approved_digest=DIGEST):
        refusal = _refused({"approved_target": "va", "approved_digest": DIGEST})
    assert refusal.reason == "approval_missing"
    assert "approved_digest" in refusal.message and "`value`" in refusal.message


def test_missing_digest_key_refused_even_when_the_entry_does_not_require_it(deployment):
    config = _config()
    config["apps"][APP]["approval"][TOOL]["require"] = {"all_of": ["approved_target"]}
    deployment.write_config(config)
    with _stamped(deployment):
        refusal = _refused({"approved_target": "va"})
    assert refusal.reason == "approval_missing"
    assert "approved_digest" in refusal.message


def test_a_digest_the_stamp_lacks_refuses_even_when_the_entry_does_not_require_it(deployment):
    config = _config()
    config["apps"][APP]["approval"][TOOL]["require"] = {"all_of": ["approved_target"]}
    deployment.write_config(config)
    with _stamped(deployment):
        refusal = _refused({**VALUE, "approved_target": "va"})
    assert refusal.reason == "approval_missing"
    assert "approved_digest" in refusal.message


def test_unserialisable_argument_refused(deployment):
    with _stamped(deployment, approved_digest=DIGEST):
        refusal = _refused({"value": float("nan"), "approved_target": "va"})
    assert refusal.reason == "approval_config_invalid"
    assert "approved_digest" in refusal.message


def test_target_switch_refuses(deployment):
    deployment.write_record("live")
    with _stamped(deployment, approved_digest=DIGEST):
        refusal = _refused({**VALUE, "approved_target": "va", "approved_digest": DIGEST})
    assert refusal.reason == "approval_missing"
    assert "approved for target `va`; the run is on `live`" in refusal.message
    assert refusal.message.endswith(REFUSED_GUIDANCE)


def test_malformed_bind_refuses_as_config_invalid(deployment):
    config = _config()
    config["apps"][APP]["approval"][TOOL]["bind"] = {"input_digests": "approved_digest"}
    deployment.write_config(config)
    with _stamped(deployment, approved_digest=DIGEST):
        refusal = _refused({**VALUE, "approved_target": "va", "approved_digest": DIGEST})
    assert refusal.reason == "approval_config_invalid"


# --- The exception ---------------------------------------------------------------


def test_approval_refused_from_refusal_appends_guidance():
    from osprey.mcp_server.approval_record import Refusal

    refused = ApprovalRefused.from_refusal(Refusal("approval_missing", "not run."), "Re-read.")
    assert refused.reason == "approval_missing"
    assert refused.message == "not run.\nRe-read."
    assert str(refused) == refused.message


def test_approval_refused_from_verdict_names_the_tool():
    from osprey_connectors.posture_store import WriteVerdict

    verdict = WriteVerdict(allowed=False, reason="tier_never", target="va", kind="simulator")
    refused = ApprovalRefused.from_verdict(verdict, TOOL)
    assert refused.reason == "tier_never"
    assert refused.details == {"reason": "tier_never", "target": "va"}
    assert TOOL in refused.message
