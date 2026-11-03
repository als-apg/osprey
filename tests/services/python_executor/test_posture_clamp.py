"""Tests for the executor's entry gate: the write verdict for the run's target.

``enforce_posture_clamp`` asks :func:`osprey_connectors.posture_store.write_verdict`
about the control target a run launched now would write to — the deployment's
record's target, or its baseline when there is no record — and refuses a
readwrite run with that verdict's phrase. These tests stage each term the
verdict can refuse on (a read-only run, a tier that allows nothing, an unknown
target or owner, an unreadable record, a switch that is off) and pin that the
refusal speaks the verdict's own words, names no config key, and lands in the
agent record as an ``executor`` audit entry with ``reason=posture`` and
``reason=<r> target=<t>`` detail.
"""

from __future__ import annotations

import json

import pytest
from fastmcp.exceptions import ToolError

from osprey.audit import posture as posture_module
from osprey.mcp_server.python_executor.tools._execution_gates import enforce_posture_clamp
from osprey_connectors import posture_store
from osprey_connectors.refusals import refusal_text
from osprey_connectors_tests._harness import (
    FIXTURE_IDENTITY,
    REFUSAL_BAN_LIST,
    isolate_verdict_environment,
    pin_identity,
    reset_verdict_state,
    write_control_context,
)

#: A tier that allows writes, on a deployment whose live machine is real.
ARMED = {"type": "epics", "writes": "allowed"}

SESSION_KEY = "11111111-2222-3333-4444-555555555555"


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """A clean verdict environment over a scratch agent-data root.

    Returns a handle: ``.section`` is the ``control_system`` section the
    verdict reads (a tier that allows writes unless a test says otherwise) and
    ``.on(target, switch=...)`` writes the deployment's record.
    """
    root = tmp_path / "agent_data"
    root.mkdir()
    isolate_verdict_environment(monkeypatch, root)
    pin_identity(monkeypatch, FIXTURE_IDENTITY)
    monkeypatch.setenv(posture_module.POSTURE_SESSION_ENV_VAR, SESSION_KEY)

    class _Deployment:
        section: dict = dict(ARMED)

        @staticmethod
        def on(target: str, *, switch: dict | None = None):
            path = write_control_context(root, target=target, switch=switch or {})
            reset_verdict_state()
            return path

    monkeypatch.setattr(posture_module, "control_system_section", lambda: _Deployment.section)
    reset_verdict_state()
    yield _Deployment
    reset_verdict_state()


def _refusal(tool: str = "execute") -> dict:
    """The structured envelope of the clamp's refusal of a readwrite run."""
    with pytest.raises(ToolError) as exc_info:
        enforce_posture_clamp("readwrite", tool=tool)
    envelope = json.loads(str(exc_info.value))
    assert envelope["error_type"] == "safety_error"
    return envelope


def _executor_records(recorded_entries) -> list[dict]:
    """Every ``executor`` audit entry in the record, oldest first."""
    from osprey.audit.envelope import SURFACE_EXECUTOR

    entries = sorted(recorded_entries("audit"), key=lambda e: e.ts)
    return [dict(e.body) for e in entries if e.body["surface"] == SURFACE_EXECUTOR]


# -- allowed ----------------------------------------------------------------


def test_a_switched_on_target_runs_and_records_nothing(deployment, recorded_entries):
    deployment.on("live", switch={"live": "on"})

    assert enforce_posture_clamp("readwrite", tool="execute") is None
    assert _executor_records(recorded_entries) == []


def test_a_readonly_run_is_never_asked_about(deployment, recorded_entries, monkeypatch):
    """Reads are unaffected whatever the verdict would say."""
    monkeypatch.setenv(posture_module.POSTURE_ENV_VAR, "readonly")
    deployment.section = {"type": "epics"}

    assert enforce_posture_clamp("readonly", tool="execute") is None
    assert _executor_records(recorded_entries) == []


@pytest.mark.parametrize("value", ["readwrite", "READONLY", "", "sandbox", "true"])
def test_only_the_exact_readonly_value_is_a_read_only_run(deployment, monkeypatch, value):
    """Value comparison, not presence: no other spelling of the variable refuses."""
    monkeypatch.setenv(posture_module.POSTURE_ENV_VAR, value)
    deployment.on("live", switch={"live": "on"})

    assert enforce_posture_clamp("readwrite", tool="execute") is None


# -- each term refuses in its own words ------------------------------------


def test_a_read_only_run_refuses_with_its_phrase(deployment, monkeypatch):
    deployment.on("live", switch={"live": "on"})
    monkeypatch.setenv(posture_module.POSTURE_ENV_VAR, "readonly")

    envelope = _refusal()

    assert envelope["error_message"] == refusal_text("readonly_run", "real")
    assert envelope["details"] == {"reason": "readonly_run", "target": "live"}


def test_a_tier_that_allows_nothing_refuses(deployment):
    deployment.section = {"type": "epics"}
    deployment.on("live", switch={"live": "on"})

    assert _refusal()["error_message"] == refusal_text("tier_never", "real")


def test_a_switch_that_is_off_refuses(deployment):
    deployment.on("live", switch={"va": "on"})

    envelope = _refusal()

    assert envelope["error_message"] == refusal_text("switch_off", "real")
    assert envelope["details"] == {"reason": "switch_off", "target": "live"}


@pytest.mark.usefixtures("deployment")
def test_no_record_refuses_the_baseline_as_switched_off():
    """No record: the run would build the baseline connector, and no switch is on."""
    envelope = _refusal()

    assert envelope["details"] == {"reason": "switch_off", "target": "live"}
    assert envelope["error_message"] == refusal_text("switch_off", "real")


def test_a_record_target_this_deployment_cannot_build_refuses(deployment):
    """``live`` does not resolve on a section that names no real control system."""
    deployment.section = {"writes": "allowed"}
    deployment.on("live", switch={"live": "on"})

    envelope = _refusal()

    assert envelope["details"]["reason"] == "target_unknown"
    assert envelope["error_message"] == refusal_text("target_unknown")


def test_an_unknown_owner_refuses(deployment, monkeypatch):
    deployment.on("live", switch={"live": "on"})
    monkeypatch.setattr(posture_store, "current_owner", lambda owner=None: posture_store.NO_OWNER)

    envelope = _refusal()

    assert envelope["details"]["reason"] == "owner_unknown"
    assert envelope["error_message"] == refusal_text("owner_unknown", "real")


def test_an_unreadable_record_refuses(deployment):
    path = deployment.on("live", switch={"live": "on"})
    path.write_text("{ not json", encoding="utf-8")
    reset_verdict_state()

    envelope = _refusal()

    assert envelope["details"]["reason"] == "record_unavailable"
    assert envelope["error_message"] == refusal_text("record_unavailable", "real")


def test_an_inherited_target_stamp_is_not_what_the_gate_asks_about(deployment, monkeypatch):
    """The launch drops an inherited stamp, so the gate must not honour one.

    The stamp names a machine whose switch is on; the record — what the run
    will actually be stamped from — names one whose switch is off.
    """
    deployment.on("live", switch={"va": "on"})
    monkeypatch.setenv(posture_module.CONTROL_TARGET_ENV_VAR, "va")

    assert _refusal()["details"] == {"reason": "switch_off", "target": "live"}


@pytest.mark.parametrize(
    "stage",
    ["readonly_run", "tier_never", "target_unknown", "owner_unknown", "record_unavailable", "off"],
)
def test_no_refusal_names_a_key_a_file_or_a_variable(deployment, monkeypatch, stage):
    """Every refusal reads as who can lift it — never as a config edit."""
    path = deployment.on("live", switch={"live": "on" if stage != "off" else "off"})
    if stage == "readonly_run":
        monkeypatch.setenv(posture_module.POSTURE_ENV_VAR, "readonly")
    elif stage == "tier_never":
        deployment.section = {"type": "epics"}
    elif stage == "target_unknown":
        deployment.section = {"writes": "allowed"}
    elif stage == "owner_unknown":
        monkeypatch.setattr(
            posture_store, "current_owner", lambda owner=None: posture_store.NO_OWNER
        )
    elif stage == "record_unavailable":
        path.write_text("{ not json", encoding="utf-8")
        reset_verdict_state()

    envelope = _refusal()
    text = " ".join([envelope["error_message"], *envelope["suggestions"]])

    assert REFUSAL_BAN_LIST.search(text) is None, text


def test_a_verdict_that_raises_still_refuses(deployment, monkeypatch):
    """The verdict never raises by contract; if it does, the gate refuses."""
    deployment.on("live", switch={"live": "on"})

    def _explode(*_args, **_kwargs):
        raise RuntimeError("the verdict is on fire")

    monkeypatch.setattr(posture_store, "write_verdict", _explode)

    assert _refusal()["details"]["reason"] == "record_unavailable"


# -- the record --------------------------------------------------------------


def test_the_refusal_files_reason_posture_with_the_verdicts_reason_and_target(
    deployment, recorded_entries
):
    deployment.on("live", switch={})

    _refusal()

    record = _executor_records(recorded_entries)[-1]
    assert record["reason"] == "posture"
    assert record["posture"] == "sandbox"
    assert record["decision"] == "refused"
    assert record["session"] == SESSION_KEY
    assert record["detail"] == "reason=switch_off target=live"


@pytest.mark.usefixtures("deployment")
def test_a_refusal_with_no_target_files_no_target(recorded_entries, monkeypatch):
    from osprey.mcp_server.python_executor.tools import _execution_gates as gates

    monkeypatch.setattr(gates, "_launch_target", lambda _section: None)

    assert _refusal()["details"] == {"reason": "target_unknown", "target": None}
    assert _executor_records(recorded_entries)[-1]["detail"] == "reason=target_unknown"
