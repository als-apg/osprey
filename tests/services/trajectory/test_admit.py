"""The trajectory service's one admission gate.

Every refusal :func:`AdmitGate.admit` can give has a test here, in the order
the gate checks them — the owner's name, the deployment ceiling, then the
owner's record out of the read-only tree — together with the facts an admitted
action's audit record carries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.runtime.guarded_run import ENV_CONTROL_IDENTITY
from osprey.services.trajectory.admit import (
    Admission,
    AdmitGate,
    AdmitRefused,
)
from osprey_connectors import control_context, posture_store
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, UNKNOWN_IDENTITY

OWNER = "alice"
KIND = "panel"
#: Writes armed; a section stating no type is the simulator in process, on ``va``.
ARMED = {"writes_enabled": True}
DISARMED = {"writes_enabled": False}


def _provision_tree(root: Path) -> Path:
    tree = root / posture_store.STATE_DIR_NAME
    tree.mkdir(parents=True, exist_ok=True)
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    return tree


def _write_record(tree: Path, owner: str, *, target: str, generation: int, posture=None) -> None:
    control_context.write_record(
        control_context.ControlContext(
            target=target, generation=generation, posture=dict(posture or {})
        ),
        path=tree / owner / control_context.RECORD_FILENAME,
    )
    posture_store.invalidate_cache()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_CONTEXT_DIR_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        ENV_CONTROL_IDENTITY,
        AUDIT_IDENTITY_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    posture_store.invalidate_cache()
    yield
    posture_store.invalidate_cache()


@pytest.fixture
def tree(tmp_path, monkeypatch):
    tree = _provision_tree(tmp_path)
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
    posture_store.invalidate_cache()
    return tree


def _refusal(gate: AdmitGate, owner, kind: str = KIND) -> AdmitRefused:
    with pytest.raises(AdmitRefused) as caught:
        gate.admit(owner, kind)
    return caught.value


# -- admitted ----------------------------------------------------------------


@pytest.mark.usefixtures("tree")
def test_no_record_follows_the_baseline_and_admits():
    gate = AdmitGate(target="va", section=ARMED)
    admission = gate.admit(OWNER, KIND)
    assert admission == Admission(owner=OWNER, kind=KIND, target="va", generation=0, posture={})


def test_record_on_the_service_target_returns_its_audit_facts(tree):
    _write_record(tree, OWNER, target="live", generation=3, posture={"va": "sandbox"})
    gate = AdmitGate(target="live", section=ARMED)
    admission = gate.admit(OWNER, "agent")
    assert admission.owner == OWNER
    assert admission.kind == "agent"
    assert admission.target == "live"
    assert admission.generation == 3
    # A narrowing of another target narrows nothing here, and is still recorded.
    assert admission.posture == {"va": "sandbox"}


def test_admission_posture_is_a_copy(tree):
    _write_record(tree, OWNER, target="live", generation=1, posture={"va": "sandbox"})
    gate = AdmitGate(target="live", section=ARMED)
    gate.admit(OWNER, KIND).posture["live"] = "sandbox"
    assert gate.admit(OWNER, KIND).posture == {"va": "sandbox"}


def test_baseline_comes_from_the_section():
    assert AdmitGate(target="va", section=ARMED).baseline == "va"
    assert AdmitGate(target="live", section={"type": "mock", "writes_enabled": True}).baseline == (
        "live"
    )


@pytest.mark.usefixtures("tree")
def test_from_config_reads_the_control_system_section():
    gate = AdmitGate.from_config({"control_system": ARMED}, "va")
    assert gate.target == "va"
    assert gate.admit(OWNER, KIND).target == "va"


# -- no_operator_identity ----------------------------------------------------


@pytest.mark.parametrize("owner", [None, "", "   ", 7, UNKNOWN_IDENTITY])
@pytest.mark.usefixtures("tree")
def test_missing_or_unknown_owner_is_refused(owner):
    refused = _refusal(AdmitGate(target="va", section=ARMED), owner)
    assert refused.code == "no_operator_identity"
    assert refused.status == 403


@pytest.mark.parametrize("env", [ENV_CONTROL_IDENTITY, AUDIT_IDENTITY_ENV])
@pytest.mark.usefixtures("tree")
def test_this_process_identity_is_a_floor_name(monkeypatch, env):
    monkeypatch.setenv(env, "trajectory")
    refused = _refusal(AdmitGate(target="va", section=ARMED), "trajectory")
    assert refused.code == "no_operator_identity"
    assert refused.status == 403
    # Another person is still admitted while the variable names the service.
    assert AdmitGate(target="va", section=ARMED).admit(OWNER, KIND).owner == OWNER


@pytest.mark.usefixtures("tree")
def test_empty_identity_variable_names_nobody(monkeypatch):
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "")
    assert AdmitGate(target="va", section=ARMED).admit(OWNER, KIND).owner == OWNER


@pytest.mark.usefixtures("tree")
def test_identity_refusal_comes_before_the_ceiling():
    refused = _refusal(AdmitGate(target="va", section=DISARMED), None)
    assert refused.code == "no_operator_identity"


# -- writes_disabled ---------------------------------------------------------


@pytest.mark.usefixtures("tree")
def test_ceiling_off_is_writes_disabled():
    refused = _refusal(AdmitGate(target="va", section=DISARMED), OWNER)
    assert refused.code == "writes_disabled"
    assert refused.status == 403
    assert "va" in refused.detail


def test_ceiling_is_read_before_the_tree():
    # No tree bound: the ceiling still answers first.
    refused = _refusal(AdmitGate(target="va", section={}), OWNER)
    assert refused.code == "writes_disabled"


@pytest.mark.usefixtures("tree")
def test_ceiling_is_read_through_target_writes_enabled(monkeypatch):
    from osprey.services.trajectory import admit as admit_mod

    seen = []

    def fake(section, target):
        seen.append((section, target))
        return False

    monkeypatch.setattr(admit_mod, "_target_writes_enabled", fake)
    refused = _refusal(AdmitGate(target="live", section=ARMED), OWNER)
    assert refused.code == "writes_disabled"
    assert seen == [(ARMED, "live")]


# -- control_context_unavailable --------------------------------------------


def test_unbound_tree_is_unavailable():
    refused = _refusal(AdmitGate(target="va", section=ARMED), OWNER)
    assert refused.code == "control_context_unavailable"
    assert refused.status == 403
    assert posture_store.CONTROL_CONTEXT_TREE_ENV_VAR in refused.detail


def test_unusable_tree_bind_is_unavailable(monkeypatch):
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, "relative/tree")
    posture_store.invalidate_cache()
    refused = _refusal(AdmitGate(target="va", section=ARMED), OWNER)
    assert refused.code == "control_context_unavailable"


def test_unreadable_record_is_unavailable(tree):
    record = tree / OWNER / control_context.RECORD_FILENAME
    record.parent.mkdir(parents=True)
    record.write_text("{not json", encoding="utf-8")
    posture_store.invalidate_cache()
    refused = _refusal(AdmitGate(target="va", section=ARMED), OWNER)
    assert refused.code == "control_context_unavailable"
    assert refused.status == 403


@pytest.mark.usefixtures("tree")
def test_owner_context_is_read_once(monkeypatch):
    from osprey.services.trajectory import admit as admit_mod

    calls = []
    real = admit_mod._owner_context

    def counting(owner, target, *, baseline):
        calls.append((owner, target, baseline))
        return real(owner, target, baseline=baseline)

    monkeypatch.setattr(admit_mod, "_owner_context", counting)
    AdmitGate(target="va", section=ARMED).admit(OWNER, KIND)
    assert calls == [(OWNER, "va", "va")]


# -- control_target_mismatch -------------------------------------------------


def test_record_on_another_target_is_a_mismatch(tree):
    _write_record(tree, OWNER, target="live", generation=2)
    refused = _refusal(AdmitGate(target="va", section=ARMED), OWNER)
    assert refused.code == "control_target_mismatch"
    assert refused.status == 409
    assert "live" in refused.detail and "va" in refused.detail


@pytest.mark.usefixtures("tree")
def test_no_record_off_the_baseline_is_a_mismatch():
    # Baseline va (no type stated); the service drives live; no record → va.
    refused = _refusal(AdmitGate(target="live", section=ARMED), OWNER)
    assert refused.code == "control_target_mismatch"
    assert refused.status == 409


def test_unswitched_record_follows_the_baseline(tree):
    # Generation 0 is claimed onto the baseline whatever target it was written with.
    _write_record(tree, OWNER, target="live", generation=0)
    admission = AdmitGate(target="va", section=ARMED).admit(OWNER, KIND)
    assert admission.target == "va"
    assert admission.generation == 0


# -- narrowing ---------------------------------------------------------------


def test_narrowed_record_is_refused(tree):
    _write_record(tree, OWNER, target="va", generation=1, posture={"va": "sandbox"})
    refused = _refusal(AdmitGate(target="va", section=ARMED), OWNER)
    assert refused.code == "narrowing"
    assert refused.status == 403


def test_mismatch_is_reported_before_narrowing(tree):
    _write_record(tree, OWNER, target="live", generation=1, posture={"va": "sandbox"})
    refused = _refusal(AdmitGate(target="va", section=ARMED), OWNER)
    assert refused.code == "control_target_mismatch"


@pytest.mark.usefixtures("tree")
def test_refusal_carries_the_bridge_detail_shape():
    refused = _refusal(AdmitGate(target="va", section=DISARMED), OWNER)
    assert refused.as_detail() == {"code": "writes_disabled", "detail": refused.detail}
