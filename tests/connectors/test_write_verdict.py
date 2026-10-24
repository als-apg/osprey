"""The write verdict: one total order over every reason a write is refused.

:func:`~osprey_connectors.posture_store.write_verdict` combines the run mode, the
login tier, the target, the owner, the owner's mapped tier, the record (the host's
own, or the owner's out of a bound tree), the person's switch and the launch pin,
in that order; the first that refuses names the reason. These tests pin each
reason alone, the order when several conditions hold at once, and the record
arms that must fail closed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey_connectors import control_context, posture_store
from osprey_connectors.posture_store import NO_OWNER, WriteVerdict, write_verdict
from osprey_connectors.refusals import REASONS
from tests._control_context_fixtures import write_control_context, write_payload
from tests._write_verdict_cases import case_environment

__all__ = ["case_environment"]

LIVE = {"type": "epics", "writes": "allowed"}
NEVER = {"type": "epics", "writes": "never"}

EXECUTION_MODE = "OSPREY_EXECUTION_MODE"


@pytest.fixture(autouse=True)
def clean_environment(case_environment):
    """No stamped owner, tree, pin or run mode; the host root at a scratch dir."""


@pytest.fixture
def root(tmp_path) -> Path:
    return tmp_path / "data"


def _host(root: Path, *, switch=None, posture=None) -> Path:
    return write_control_context(root, switch=switch, posture=posture, owned_by=None)


def _provision_tree(base: Path) -> Path:
    tree = base / posture_store.STATE_DIR_NAME
    tree.mkdir(parents=True, exist_ok=True)
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    return tree


def _tree_record(tree: Path, owner: str, *, switch=None, posture=None) -> Path:
    record = control_context.ControlContext(
        target="live",
        generation=3,
        posture=dict(posture or {}),
        switch=dict(switch or {}),
        switch_epoch=1,
    )
    return write_payload(tree / owner / control_context.RECORD_FILENAME, record.to_payload())


@pytest.fixture
def tree(tmp_path, monkeypatch) -> Path:
    tree = _provision_tree(tmp_path / "shared")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
    return tree


def _refused(verdict: WriteVerdict, reason: str) -> None:
    assert verdict.allowed is False
    assert verdict.reason == reason
    assert verdict.reason in REASONS


# --- the allow ------------------------------------------------------------


def test_everything_in_place_allows(root):
    _host(root, switch={"live": "on"})
    verdict = write_verdict(LIVE, "live")
    assert verdict == WriteVerdict(allowed=True, reason=None, target="live", kind="real")


def test_kind_names_the_machine(root):
    _host(root, switch={"va": "on", "standin": "on"})
    section = {**LIVE, "connector": {"live_standin": {"prefix": "S:"}}}
    assert write_verdict(section, "va").kind == "simulator"
    assert write_verdict(section, "standin").kind == "rehearsal"


# --- each reason alone ---------------------------------------------------


def test_readonly_run(root, monkeypatch):
    _host(root, switch={"live": "on"})
    monkeypatch.setenv(EXECUTION_MODE, "readonly")
    verdict = write_verdict(LIVE, "live")
    _refused(verdict, "readonly_run")
    assert verdict.target == "live"
    assert verdict.kind == "real"


def test_tier_never(root):
    _host(root, switch={"live": "on"})
    _refused(write_verdict(NEVER, "live"), "tier_never")


def test_silent_tier_is_never(root):
    _host(root, switch={"live": "on"})
    _refused(write_verdict({"type": "epics"}, "live"), "tier_never")


@pytest.mark.parametrize("target", ["bogus", None, "", 7])
def test_target_unknown(root, target):
    _host(root, switch={"live": "on"})
    verdict = write_verdict(LIVE, target)
    _refused(verdict, "target_unknown")
    assert verdict.kind is None


def test_live_without_a_real_machine_is_target_unknown(root):
    _host(root, switch={"live": "on"})
    _refused(write_verdict({"writes": "allowed"}, "live"), "target_unknown")


@pytest.mark.parametrize("owner", [NO_OWNER, "", "   ", 7, ["alice"]])
def test_owner_unknown(root, owner):
    _host(root, switch={"live": "on"})
    _refused(write_verdict(LIVE, "live", owner=owner), "owner_unknown")


def test_tree_with_nobody_bound_is_owner_unknown(tree):
    _tree_record(tree, "alice", switch={"live": "on"})
    section = {**LIVE, "login_tiers": {"alice": "allowed"}}
    _refused(write_verdict(section, "live"), "owner_unknown")


def test_owner_missing_from_the_map_is_owner_unknown(root):
    _host(root, switch={"live": "on"})
    section = {**LIVE, "login_tiers": {"alice": "allowed"}}
    _refused(write_verdict(section, "live", owner="mallory"), "owner_unknown")


def test_tree_with_no_map_is_owner_unknown(tree):
    _tree_record(tree, "alice", switch={"live": "on"})
    _refused(write_verdict(LIVE, "live", owner="alice"), "owner_unknown")


def test_mapped_never_is_tier_never(root):
    _host(root, switch={"live": "on"})
    section = {**LIVE, "login_tiers": {"alice": "never"}}
    _refused(write_verdict(section, "live", owner="alice"), "tier_never")


def test_corrupt_host_record_is_record_unavailable(root):
    path = control_context.record_path_under(root)
    write_payload(path, "{not json")
    path.write_text("{not json", encoding="utf-8")
    control_context.invalidate_cache()
    _refused(write_verdict(LIVE, "live"), "record_unavailable")


def test_foreign_schema_host_record_is_record_unavailable(root):
    path = _host(root, switch={"live": "on"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] = 2
    write_payload(path, payload)
    _refused(write_verdict(LIVE, "live"), "record_unavailable")


def test_unresolvable_host_root_is_record_unavailable(monkeypatch):
    monkeypatch.setattr(posture_store, "agent_data_root", lambda: None)
    monkeypatch.setattr(control_context, "record_path", lambda: None)
    _refused(write_verdict(LIVE, "live"), "record_unavailable")


def test_missing_host_record_is_switch_off(root):
    assert not control_context.record_path_under(root).exists()
    _refused(write_verdict(LIVE, "live"), "switch_off")


def test_switch_off(root):
    _host(root)
    _refused(write_verdict(LIVE, "live"), "switch_off")


def test_another_targets_switch_does_not_arm_this_one(root):
    _host(root, switch={"va": "on"})
    _refused(write_verdict(LIVE, "live"), "switch_off")
    assert write_verdict(LIVE, "va").allowed is True


def test_recorded_sandbox_for_the_target_is_switch_off(root):
    """The interim narrowing: a recorded sandbox refuses even with the switch on."""
    _host(root, switch={"live": "on"}, posture={"live": "sandbox"})
    _refused(write_verdict(LIVE, "live"), "switch_off")


def test_recorded_sandbox_for_another_target_does_not_refuse(root):
    _host(root, switch={"live": "on"}, posture={"va": "sandbox"})
    assert write_verdict(LIVE, "live").allowed is True


@pytest.mark.parametrize("stamp", ["live=sandbox", "*=sandbox", "garbage", "live=maybe"])
def test_launch_pinned(root, monkeypatch, stamp):
    _host(root, switch={"live": "on"})
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, stamp)
    _refused(write_verdict(LIVE, "live"), "launch_pinned")


@pytest.mark.parametrize("stamp", ["live=sandbox", "garbage"])
def test_ignore_launch_pin_skips_only_the_pin(root, monkeypatch, stamp):
    _host(root, switch={"live": "on"})
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, stamp)
    assert write_verdict(LIVE, "live", ignore_launch_pin=True).allowed is True


def test_ignore_launch_pin_still_reports_the_switch(root, monkeypatch):
    _host(root)
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "live=sandbox")
    _refused(write_verdict(LIVE, "live", ignore_launch_pin=True), "switch_off")


def test_pin_on_another_target_does_not_refuse(root, monkeypatch):
    _host(root, switch={"live": "on"})
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "va=sandbox")
    assert write_verdict(LIVE, "live").allowed is True


# --- the order on multi-condition inputs ----------------------------------


def test_order_readonly_beats_everything(monkeypatch):
    monkeypatch.setenv(EXECUTION_MODE, "readonly")
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "garbage")
    _refused(write_verdict(NEVER, "bogus", owner=NO_OWNER), "readonly_run")


def test_order_tier_beats_target_and_owner(monkeypatch):
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "garbage")
    _refused(write_verdict(NEVER, "bogus", owner=NO_OWNER), "tier_never")


def test_order_target_beats_owner():
    _refused(write_verdict(LIVE, "bogus", owner=NO_OWNER), "target_unknown")


def test_order_owner_beats_the_rung_and_the_record(root):
    write_payload(control_context.record_path_under(root), "{not json")
    section = {**LIVE, "login_tiers": {"alice": "never"}}
    _refused(write_verdict(section, "live", owner=NO_OWNER), "owner_unknown")


def test_order_mapped_never_beats_the_record(root):
    write_payload(control_context.record_path_under(root), "{not json")
    section = {**LIVE, "login_tiers": {"alice": "never"}}
    _refused(write_verdict(section, "live", owner="alice"), "tier_never")


def test_order_record_beats_the_pin(root, monkeypatch):
    write_payload(control_context.record_path_under(root), "{not json")
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "live=sandbox")
    _refused(write_verdict(LIVE, "live"), "record_unavailable")


def test_order_switch_beats_the_pin(root, monkeypatch):
    _host(root)
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "live=sandbox")
    _refused(write_verdict(LIVE, "live"), "switch_off")


# --- two owners on one host -----------------------------------------------


def test_two_owners_on_one_host_read_their_own_tiers(root):
    _host(root, switch={"live": "on"})
    section = {**LIVE, "login_tiers": {"alice": "allowed", "bob": "never"}}
    assert write_verdict(section, "live", owner="alice").allowed is True
    _refused(write_verdict(section, "live", owner="bob"), "tier_never")


def test_two_owners_in_one_tree_read_their_own_records(tree):
    _tree_record(tree, "alice", switch={"live": "on"})
    _tree_record(tree, "bob")
    section = {**LIVE, "login_tiers": {"alice": "allowed", "bob": "allowed"}}
    assert write_verdict(section, "live", owner="alice").allowed is True
    _refused(write_verdict(section, "live", owner="bob"), "switch_off")


def test_the_bound_owner_is_the_one_read(tree):
    _tree_record(tree, "alice", switch={"live": "on"})
    section = {**LIVE, "login_tiers": {"alice": "allowed"}}
    with posture_store.bind_owner({posture_store.RESERVED_OWNER_KWARG: "alice"}):
        assert write_verdict(section, "live").allowed is True


# --- the tree record ------------------------------------------------------


TREE_SECTION = {**LIVE, "login_tiers": {"alice": "allowed"}}


@pytest.mark.usefixtures("tree")
def test_tree_enoent_is_switch_off():
    _refused(write_verdict(TREE_SECTION, "live", owner="alice"), "switch_off")


def test_tree_record_without_the_owner_file_is_switch_off(tree):
    (tree / "alice").mkdir()
    _refused(write_verdict(TREE_SECTION, "live", owner="alice"), "switch_off")


def test_tree_recorded_sandbox_is_switch_off(tree):
    _tree_record(tree, "alice", switch={"live": "on"}, posture={"live": "sandbox"})
    _refused(write_verdict(TREE_SECTION, "live", owner="alice"), "switch_off")


def test_tree_degraded_switch_value_does_not_arm(tree):
    path = _tree_record(tree, "alice")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["switch"] = {"live": "ON", "va": True}
    write_payload(path, payload)
    _refused(write_verdict(TREE_SECTION, "live", owner="alice"), "switch_off")
    _refused(write_verdict(TREE_SECTION, "va", owner="alice"), "switch_off")


def test_corrupt_tree_record_is_record_unavailable(tree):
    path = tree / "alice" / control_context.RECORD_FILENAME
    path.parent.mkdir()
    path.write_text("{not json", encoding="utf-8")
    _refused(write_verdict(TREE_SECTION, "live", owner="alice"), "record_unavailable")


def test_foreign_schema_tree_record_is_record_unavailable(tree):
    path = _tree_record(tree, "alice", switch={"live": "on"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] = 2
    write_payload(path, payload)
    _refused(write_verdict(TREE_SECTION, "live", owner="alice"), "record_unavailable")


def test_unprovisioned_tree_is_record_unavailable(tmp_path, monkeypatch):
    bare = tmp_path / "bare"
    bare.mkdir()
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(bare))
    _refused(write_verdict(TREE_SECTION, "live", owner="alice"), "record_unavailable")


def test_unusable_tree_bind_is_record_unavailable(monkeypatch):
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, "relative/tree")
    _refused(write_verdict(TREE_SECTION, "live", owner="alice"), "record_unavailable")


def test_planted_link_in_the_tree_is_record_unavailable(tree, tmp_path):
    elsewhere = _tree_record(_provision_tree(tmp_path / "other"), "x", switch={"live": "on"})
    (tree / "alice").mkdir()
    (tree / "alice" / control_context.RECORD_FILENAME).symlink_to(elsewhere)
    _refused(write_verdict(TREE_SECTION, "live", owner="alice"), "record_unavailable")


@pytest.mark.usefixtures("tree")
def test_the_tree_is_read_not_the_host(root):
    """A container under a tree never falls back to its own host record."""
    _host(root, switch={"live": "on"})
    _refused(write_verdict(TREE_SECTION, "live", owner="alice"), "switch_off")


# --- a demoted login's "on" ------------------------------------------------


def test_demoted_login_with_on_is_tier_never_and_the_switch_stays(root):
    path = _host(root, switch={"live": "on"})
    before = path.read_bytes()
    _refused(write_verdict(NEVER, "live"), "tier_never")
    assert path.read_bytes() == before
    record, status = control_context.read_record_status(path=path)
    assert status == "ok"
    assert record.switch == {"live": "on"}
    # Restored, the same "on" arms again.
    assert write_verdict(LIVE, "live").allowed is True


def test_demoted_login_in_the_tree_is_tier_never_and_the_switch_stays(tree):
    path = _tree_record(tree, "alice", switch={"live": "on"})
    before = path.read_bytes()
    section = {**LIVE, "login_tiers": {"alice": "never"}}
    _refused(write_verdict(section, "live", owner="alice"), "tier_never")
    assert path.read_bytes() == before
    assert write_verdict(TREE_SECTION, "live", owner="alice").allowed is True


# --- never raises ---------------------------------------------------------


@pytest.mark.parametrize("section", [None, "garbage", 7, [], {"writes": "allowed"}])
def test_garbage_sections_refuse(section, root):
    _host(root, switch={"live": "on"})
    verdict = write_verdict(section, "live")
    assert verdict.allowed is False
    assert verdict.reason in REASONS


def test_a_reader_that_raises_refuses_rather_than_raises(root, monkeypatch):
    _host(root, switch={"live": "on"})

    def boom(**_kwargs):
        raise RuntimeError("reader exploded")

    monkeypatch.setattr(control_context, "read_record_status", boom)
    verdict = write_verdict(LIVE, "live")
    assert verdict.allowed is False
    assert verdict.reason in REASONS


def test_never_emits_the_lock_reasons(root):
    _host(root, switch={"live": "on"})
    seen = {write_verdict(section, "live").reason for section in (LIVE, NEVER, None)}
    assert not seen & {"switching", "stopped_since"}


def test_verdict_is_frozen():
    verdict = WriteVerdict(allowed=True, reason=None, target="live", kind="real")
    with pytest.raises(AttributeError):
        verdict.allowed = False  # type: ignore[misc]
