"""The backstop verdict: the target-blind, record-blind floor under every write.

:func:`~osprey_connectors.posture_store.backstop_verdict` answers for callers that
know no target and hold no owner — a write that cannot be traced to a machine
still has to be refused under a read-only run or a ``never`` tier. It refuses
``readonly_run`` first, then ``tier_never``, and otherwise allows. These tests pin
both refusals, their order, the allow, and that nothing past the tier — the
target, the owner, the owner's tier, the record — is ever consulted.
"""

from __future__ import annotations

import pytest

from osprey_connectors import control_context, posture_store, types
from osprey_connectors.posture_store import NO_OWNER, WriteVerdict, backstop_verdict
from osprey_connectors.refusals import REASONS
from osprey_connectors_tests._harness import write_control_context
from osprey_connectors_tests._harness._write_verdict_cases import case_environment

__all__ = ["case_environment"]

LIVE = {"type": "epics", "writes": "allowed"}
ASKED = {"type": "epics", "writes": "asked"}
NEVER = {"type": "epics", "writes": "never"}

EXECUTION_MODE = "OSPREY_EXECUTION_MODE"


@pytest.fixture(autouse=True)
def clean_environment(case_environment):
    """No stamped owner, tree, pin or run mode; the host root at a scratch dir."""


def _fail(name: str):
    def reader(*_args, **_kwargs):
        raise AssertionError(f"the backstop verdict must not consult {name}")

    return reader


@pytest.fixture
def blind(monkeypatch):
    """Every reader past the tier fails on call: the record, target and owner."""
    for module, name in (
        (control_context, "read_record_status"),
        (control_context, "read_record"),
        (posture_store, "_read_tree_context"),
        (posture_store, "_tree_is_bound"),
        (posture_store, "_control_context_tree"),
        (posture_store, "resolve_target"),
        (posture_store, "machine_kind"),
        (posture_store, "current_owner"),
        (posture_store, "owner_tier_term"),
        (posture_store, "launch_pin_term"),
        (types, "resolve_target"),
        (types, "machine_kind"),
    ):
        monkeypatch.setattr(module, name, _fail(name))


def _refused(verdict: WriteVerdict, reason: str) -> None:
    assert verdict.allowed is False
    assert verdict.reason == reason
    assert verdict.reason in REASONS
    assert verdict.target is None
    assert verdict.kind is None
    assert verdict.position is None


# --- the allow ------------------------------------------------------------


@pytest.mark.usefixtures("blind")
def test_allowed_tier_allows_with_no_target_and_no_kind():
    # Arrange / Act
    verdict = backstop_verdict(LIVE)

    # Assert
    assert verdict == WriteVerdict(allowed=True, reason=None, target=None, kind=None)


@pytest.mark.usefixtures("blind")
def test_asked_tier_allows():
    verdict = backstop_verdict(ASKED)

    assert verdict == WriteVerdict(allowed=True, reason=None, target=None, kind=None)


@pytest.mark.parametrize("stored", ["asked", "allowed"])
def test_an_allow_reports_no_position_whatever_is_stored(tmp_path, stored):
    """The backstop is switch-blind: it never answers for a position."""
    write_control_context(tmp_path / "data", switch={"live": stored}, owned_by=None)

    assert backstop_verdict(LIVE).position is None


@pytest.mark.usefixtures("blind")
def test_a_retired_leaf_does_not_arm_the_tier():
    """The tier reader decides, and a retired ``writes_enabled: true`` arms nothing.

    Rows ``tier-absent-old-armed-flat`` / ``tier-absent-old-armed-connector``
    (ruling D2: no runtime retired-leaf fallback).
    """
    verdict = backstop_verdict({"type": "epics", "writes_enabled": True})

    assert verdict.allowed is False
    assert verdict.reason == "tier_never"


@pytest.mark.usefixtures("blind")
def test_allow_needs_no_record_no_switch_and_no_owner():
    """No record exists, no owner is bound: the backstop still allows."""
    verdict = backstop_verdict(LIVE)

    assert verdict.allowed is True
    assert posture_store._owner_var.get() is NO_OWNER


# --- the refusals ---------------------------------------------------------


@pytest.mark.parametrize(
    "section",
    [NEVER, {"type": "epics"}, {}, None, True, "allowed", ["writes"]],
    ids=["never", "silent", "empty", "none", "bool", "string", "list"],
)
@pytest.mark.usefixtures("blind")
def test_tier_that_does_not_allow_refuses_tier_never(section):
    _refused(backstop_verdict(section), "tier_never")


@pytest.mark.usefixtures("blind")
def test_readonly_run_refuses_with_tier_allowed(monkeypatch):
    monkeypatch.setenv(EXECUTION_MODE, "readonly")

    _refused(backstop_verdict(LIVE), "readonly_run")


@pytest.mark.usefixtures("blind")
def test_readonly_run_comes_before_tier_never(monkeypatch):
    monkeypatch.setenv(EXECUTION_MODE, "readonly")

    _refused(backstop_verdict(NEVER), "readonly_run")


@pytest.mark.usefixtures("blind")
def test_other_execution_modes_are_not_readonly(monkeypatch):
    monkeypatch.setenv(EXECUTION_MODE, "readwrite")

    assert backstop_verdict(LIVE).allowed is True


def test_readonly_run_refuses_even_with_the_switch_on(monkeypatch, tmp_path):
    """A switched-on record does not lift a read-only run in the backstop either."""
    write_control_context(tmp_path / "data", switch={"live": "asked"}, owned_by=None)
    monkeypatch.setenv(EXECUTION_MODE, "readonly")

    _refused(backstop_verdict(LIVE), "readonly_run")


def test_a_switched_off_record_does_not_refuse_the_backstop(tmp_path):
    """The switch is not a backstop term: an off switch on disk still allows."""
    write_control_context(tmp_path / "data", switch={"live": "off"}, owned_by=None)

    assert backstop_verdict(LIVE).allowed is True


# --- never raises ---------------------------------------------------------


@pytest.mark.usefixtures("blind")
def test_a_tier_reader_that_raises_refuses_tier_never(monkeypatch):
    monkeypatch.setattr(posture_store, "tier_allows", _fail("tier_allows (raising)"))

    _refused(backstop_verdict(LIVE), "tier_never")


@pytest.mark.usefixtures("blind")
def test_readonly_run_does_not_read_the_tier(monkeypatch):
    monkeypatch.setenv(EXECUTION_MODE, "readonly")
    monkeypatch.setattr(posture_store, "tier_allows", _fail("tier_allows"))

    _refused(backstop_verdict(LIVE), "readonly_run")
