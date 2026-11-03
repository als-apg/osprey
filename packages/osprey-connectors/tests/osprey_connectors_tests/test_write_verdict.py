"""The write verdict: one total order over every reason a write is refused.

:func:`~osprey_connectors.posture_store.write_verdict` combines the run mode, the
login tier, the target, the owner, the owner's mapped tier, the record (the host's
own, or the owner's out of a bound tree), the person's switch and the launch pin,
in that order; the first that refuses names the reason. These tests pin each
reason alone, the order when several conditions hold at once, the record arms
that must fail closed, and the effective position an allowed write carries.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from osprey_connectors import control_context, posture_store
from osprey_connectors.posture_store import NO_OWNER, WriteVerdict, write_verdict
from osprey_connectors.refusals import REASONS, refusal_text
from osprey_connectors_tests._harness import (
    write_control_context,
    write_payload,
    write_server_report,
)
from osprey_connectors_tests._harness._control_context_fixtures import FIXTURE_IDENTITY
from osprey_connectors_tests._harness._write_verdict_cases import case_environment

__all__ = ["case_environment"]

LIVE = {"type": "epics", "writes": "allowed"}
ASKED = {"type": "epics", "writes": "asked"}
NEVER = {"type": "epics", "writes": "never"}

EXECUTION_MODE = "OSPREY_EXECUTION_MODE"

#: The host's own login: :func:`case_environment` pins the acting identity to it.
ME = FIXTURE_IDENTITY


@pytest.fixture(autouse=True)
def clean_environment(case_environment):
    """No stamped owner, tree, pin or run mode; the host root at a scratch dir."""


@pytest.fixture
def root(tmp_path) -> Path:
    return tmp_path / "data"


def _host(root: Path, *, switch=None, posture=None) -> Path:
    """This host's record; *posture* is an old narrowing laid over the mirror verbatim."""
    path = write_control_context(root, switch=switch, owned_by=None)
    if posture is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["posture"] = dict(posture)
        write_payload(path, payload)
        control_context.invalidate_cache()
    return path


def _legacy_host(root: Path, switch: dict[str, str]) -> Path:
    """This host's record still spelling *switch* as an older release stored it."""
    path = write_control_context(root, owned_by=None)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["switch"] = dict(switch)
    write_payload(path, payload)
    control_context.invalidate_cache()
    return path


def _provision_tree(base: Path) -> Path:
    tree = base / posture_store.STATE_DIR_NAME
    tree.mkdir(parents=True, exist_ok=True)
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    return tree


def _tree_record(tree: Path, owner: str, *, switch=None, posture=None) -> Path:
    record = control_context.ControlContext(
        target="live",
        generation=3,
        switch=dict(switch or {}),
        switch_epoch=1,
    )
    payload = record.to_payload()
    if posture is not None:
        payload["posture"] = dict(posture)  # an old narrowing, verbatim
    return write_payload(tree / owner / control_context.RECORD_FILENAME, payload)


@pytest.fixture
def tree(tmp_path, monkeypatch) -> Path:
    tree = _provision_tree(tmp_path / "shared")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
    return tree


def _refused(verdict: WriteVerdict, reason: str) -> None:
    assert verdict.allowed is False
    assert verdict.reason == reason
    assert verdict.reason in REASONS
    assert verdict.position is None


# --- the allow ------------------------------------------------------------


def test_everything_in_place_allows(root):
    _host(root, switch={"live": "asked"})
    verdict = write_verdict(LIVE, "live")
    assert verdict == WriteVerdict(
        allowed=True, reason=None, target="live", kind="real", position="asked"
    )


STANDIN_SECTION = {**LIVE, "connector": {"live_standin": {"prefix": "S:"}}}

#: The words only the rehearsal machine may be described with (FR 7).
REHEARSAL_WORDS = ("rehearsal", "nothing moves")


def _report(root: Path, kind: str | None, *, pid: int | None = None) -> None:
    """A controls server's report publishing *kind* for ``standin`` (C-KIND)."""
    meta = {} if kind is None else {"machine_kind": kind}
    write_server_report(root, os.getpid() if pid is None else pid, targets={"standin": meta})


def test_kind_names_the_machine(root):
    _host(root, switch={"va": "asked", "standin": "asked"})
    _report(root, "rehearsal")
    assert write_verdict(STANDIN_SECTION, "va").kind == "simulator"
    assert write_verdict(STANDIN_SECTION, "live").kind == "real"
    assert write_verdict(STANDIN_SECTION, "standin").kind == "rehearsal"


def test_a_live_standin_reads_rehearsal(root):
    _host(root, switch={"standin": "asked"})
    _report(root, "rehearsal")
    assert write_verdict(STANDIN_SECTION, "standin") == WriteVerdict(
        allowed=True, reason=None, target="standin", kind="rehearsal", position="asked"
    )


def test_a_standin_pointing_elsewhere_reads_real_and_its_phrase_has_no_rehearsal_words(root):
    _host(root, switch={})
    _report(root, "real")
    verdict = write_verdict(STANDIN_SECTION, "standin")
    assert (verdict.reason, verdict.kind) == ("switch_off", "real")
    text = refusal_text(verdict.reason, verdict.kind).lower()
    assert not any(word in text for word in REHEARSAL_WORDS), text


def test_no_live_report_reads_real(root):
    _host(root, switch={"standin": "asked"})
    assert write_verdict(STANDIN_SECTION, "standin").kind == "real"


def test_a_dead_servers_report_is_ignored(root):
    _host(root, switch={"standin": "asked"})
    _report(root, "rehearsal", pid=2_147_483_646)
    assert write_verdict(STANDIN_SECTION, "standin").kind == "real"


def test_an_unresolved_target_has_no_kind_whatever_is_published(root):
    _report(root, "rehearsal")
    assert write_verdict(STANDIN_SECTION, "bogus").kind is None


# --- each reason alone ---------------------------------------------------


def test_readonly_run(root, monkeypatch):
    _host(root, switch={"live": "asked"})
    monkeypatch.setenv(EXECUTION_MODE, "readonly")
    verdict = write_verdict(LIVE, "live")
    _refused(verdict, "readonly_run")
    assert verdict.target == "live"
    assert verdict.kind == "real"


def test_tier_never(root):
    _host(root, switch={"live": "asked"})
    _refused(write_verdict(NEVER, "live"), "tier_never")


def test_silent_tier_is_never(root):
    _host(root, switch={"live": "asked"})
    _refused(write_verdict({"type": "epics"}, "live"), "tier_never")


@pytest.mark.parametrize("target", ["bogus", None, "", 7])
def test_target_unknown(root, target):
    _host(root, switch={"live": "asked"})
    verdict = write_verdict(LIVE, target)
    _refused(verdict, "target_unknown")
    assert verdict.kind is None


def test_live_without_a_real_machine_is_target_unknown(root):
    _host(root, switch={"live": "asked"})
    _refused(write_verdict({"writes": "allowed"}, "live"), "target_unknown")


@pytest.mark.parametrize("owner", [NO_OWNER, "", "   ", 7, ["alice"]])
def test_owner_unknown(root, owner):
    _host(root, switch={"live": "asked"})
    _refused(write_verdict(LIVE, "live", owner=owner), "owner_unknown")


def test_tree_with_nobody_bound_is_owner_unknown(tree):
    _tree_record(tree, "alice", switch={"live": "asked"})
    section = {**LIVE, "login_tiers": {"alice": "allowed"}}
    _refused(write_verdict(section, "live"), "owner_unknown")


def test_owner_missing_from_the_map_is_owner_unknown(root):
    _host(root, switch={"live": "asked"})
    section = {**LIVE, "login_tiers": {"alice": "allowed"}}
    _refused(write_verdict(section, "live", owner=ME), "owner_unknown")


def test_tree_with_no_map_is_owner_unknown(tree):
    _tree_record(tree, "alice", switch={"live": "asked"})
    _refused(write_verdict(LIVE, "live", owner="alice"), "owner_unknown")


def test_mapped_never_is_tier_never(root):
    _host(root, switch={"live": "asked"})
    section = {**LIVE, "login_tiers": {ME: "never"}}
    _refused(write_verdict(section, "live", owner=ME), "tier_never")


def test_corrupt_host_record_is_record_unavailable(root):
    path = control_context._record_path_under(root)
    write_payload(path, "{not json")
    path.write_text("{not json", encoding="utf-8")
    control_context.invalidate_cache()
    _refused(write_verdict(LIVE, "live"), "record_unavailable")


def test_foreign_schema_host_record_is_record_unavailable(root):
    path = _host(root, switch={"live": "asked"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] = 2
    write_payload(path, payload)
    _refused(write_verdict(LIVE, "live"), "record_unavailable")


def test_unresolvable_host_root_is_record_unavailable(monkeypatch):
    monkeypatch.setattr(posture_store, "agent_data_root", lambda: None)
    monkeypatch.setattr(control_context, "record_path", lambda: None)
    _refused(write_verdict(LIVE, "live"), "record_unavailable")


def test_missing_host_record_is_switch_off(root):
    assert not control_context._record_path_under(root).exists()
    _refused(write_verdict(LIVE, "live"), "switch_off")


def test_switch_off(root):
    _host(root)
    _refused(write_verdict(LIVE, "live"), "switch_off")


def test_another_targets_switch_does_not_arm_this_one(root):
    _host(root, switch={"va": "asked"})
    _refused(write_verdict(LIVE, "live"), "switch_off")
    assert write_verdict(LIVE, "va").allowed is True


def test_an_old_sandbox_narrowing_does_not_refuse_with_the_switch_on(root):
    """The interim reading is retired: an old narrowing left in the file is inert."""
    _host(root, switch={"live": "asked"}, posture={"live": "sandbox"})
    verdict = write_verdict(LIVE, "live")
    assert verdict.allowed is True
    assert verdict.reason is None


def test_an_old_narrowing_does_not_arm_a_switch_that_is_off(root):
    _host(root, switch={"va": "asked"}, posture={})
    _refused(write_verdict(LIVE, "live"), "switch_off")


def test_recorded_sandbox_for_another_target_does_not_refuse(root):
    _host(root, switch={"live": "asked"}, posture={"va": "sandbox"})
    assert write_verdict(LIVE, "live").allowed is True


@pytest.mark.parametrize("stamp", ["live=sandbox", "*=sandbox", "garbage", "live=maybe"])
def test_launch_pinned(root, monkeypatch, stamp):
    _host(root, switch={"live": "asked"})
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, stamp)
    _refused(write_verdict(LIVE, "live"), "launch_pinned")


@pytest.mark.parametrize("stamp", ["live=sandbox", "garbage"])
def test_ignore_launch_pin_skips_only_the_pin(root, monkeypatch, stamp):
    _host(root, switch={"live": "asked"})
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, stamp)
    assert write_verdict(LIVE, "live", ignore_launch_pin=True).allowed is True


def test_ignore_launch_pin_still_reports_the_switch(root, monkeypatch):
    _host(root)
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "live=sandbox")
    _refused(write_verdict(LIVE, "live", ignore_launch_pin=True), "switch_off")


def test_pin_on_another_target_does_not_refuse(root, monkeypatch):
    _host(root, switch={"live": "asked"})
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
    write_payload(control_context._record_path_under(root), "{not json")
    section = {**LIVE, "login_tiers": {"alice": "never"}}
    _refused(write_verdict(section, "live", owner=NO_OWNER), "owner_unknown")


def test_order_mapped_never_beats_the_record(root):
    write_payload(control_context._record_path_under(root), "{not json")
    section = {**LIVE, "login_tiers": {ME: "never"}}
    _refused(write_verdict(section, "live", owner=ME), "tier_never")


def test_order_owner_unknown_beats_the_foreign_owner(root):
    _host(root, switch={"live": "asked"})
    _refused(write_verdict(LIVE, "live", owner=NO_OWNER), "owner_unknown")


def test_order_foreign_owner_beats_the_rung_and_the_record(root):
    write_payload(control_context._record_path_under(root), "{not json")
    section = {**LIVE, "login_tiers": {"alice": "never"}}
    _refused(write_verdict(section, "live", owner="alice"), "owner_foreign")


def test_order_record_beats_the_pin(root, monkeypatch):
    write_payload(control_context._record_path_under(root), "{not json")
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "live=sandbox")
    _refused(write_verdict(LIVE, "live"), "record_unavailable")


def test_order_switch_beats_the_pin(root, monkeypatch):
    _host(root)
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "live=sandbox")
    _refused(write_verdict(LIVE, "live"), "switch_off")


# --- one login per host ---------------------------------------------------


def test_two_owners_on_one_host_read_their_own_tiers(root):
    """A host answers for its own login only; another owner is foreign, whatever its tier."""
    _host(root, switch={"live": "asked"})
    section = {**LIVE, "login_tiers": {ME: "allowed", "bob": "allowed"}}
    assert write_verdict(section, "live", owner=ME).allowed is True
    _refused(write_verdict(section, "live", owner="bob"), "owner_foreign")
    demoted = {**LIVE, "login_tiers": {ME: "never", "bob": "allowed"}}
    _refused(write_verdict(demoted, "live", owner=ME), "tier_never")
    _refused(write_verdict(demoted, "live", owner="bob"), "owner_foreign")


def test_a_foreign_owner_on_the_host_never_reads_the_record(root, monkeypatch):
    _host(root, switch={"live": "asked"})

    def _unread(*args, **kwargs):
        raise AssertionError("the host record was read for a foreign owner")

    monkeypatch.setattr(control_context, "read_record_status", _unread)
    _refused(write_verdict(LIVE, "live", owner="alice"), "owner_foreign")


def test_a_stamped_foreign_owner_on_the_host_is_refused(root, monkeypatch):
    _host(root, switch={"live": "asked"})
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "alice")
    _refused(write_verdict(LIVE, "live"), "owner_foreign")


def test_a_bound_foreign_owner_on_the_host_is_refused(root):
    _host(root, switch={"live": "asked"})
    with posture_store.bind_owner({posture_store.RESERVED_OWNER_KWARG: "alice"}):
        _refused(write_verdict(LIVE, "live"), "owner_foreign")


@pytest.mark.parametrize("owner", [None, ME, f"  {ME} "])
def test_the_hosts_own_login_is_unchanged(root, owner):
    _host(root, switch={"live": "asked"})
    kwargs = {} if owner is None else {"owner": owner}
    assert write_verdict(LIVE, "live", **kwargs).allowed is True


def test_the_foreign_owner_follows_the_acting_identity(root, monkeypatch):
    monkeypatch.setenv("OSPREY_AUDIT_IDENTITY", "alice")
    _host(root, switch={"live": "asked"})  # under alice's own directory
    assert write_verdict(LIVE, "live", owner="alice").allowed is True
    _refused(write_verdict(LIVE, "live", owner=ME), "owner_foreign")


def test_under_a_tree_the_owner_need_not_be_the_process_login(tree):
    _tree_record(tree, "alice", switch={"live": "asked"})
    section = {**LIVE, "login_tiers": {"alice": "allowed"}}
    assert write_verdict(section, "live", owner="alice").allowed is True


def test_two_owners_in_one_tree_read_their_own_records(tree):
    _tree_record(tree, "alice", switch={"live": "asked"})
    _tree_record(tree, "bob")
    section = {**LIVE, "login_tiers": {"alice": "allowed", "bob": "allowed"}}
    assert write_verdict(section, "live", owner="alice").allowed is True
    _refused(write_verdict(section, "live", owner="bob"), "switch_off")


def test_the_bound_owner_is_the_one_read(tree):
    _tree_record(tree, "alice", switch={"live": "asked"})
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


def test_tree_old_sandbox_narrowing_does_not_refuse_with_the_switch_on(tree):
    _tree_record(tree, "alice", switch={"live": "asked"}, posture={"live": "sandbox"})
    assert write_verdict(TREE_SECTION, "live", owner="alice").allowed is True


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
    path = _tree_record(tree, "alice", switch={"live": "asked"})
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
    elsewhere = _tree_record(_provision_tree(tmp_path / "other"), "x", switch={"live": "asked"})
    (tree / "alice").mkdir()
    (tree / "alice" / control_context.RECORD_FILENAME).symlink_to(elsewhere)
    _refused(write_verdict(TREE_SECTION, "live", owner="alice"), "record_unavailable")


@pytest.mark.usefixtures("tree")
def test_the_tree_is_read_not_the_host(root):
    """A container under a tree never falls back to its own host record."""
    _host(root, switch={"live": "asked"})
    _refused(write_verdict(TREE_SECTION, "live", owner="alice"), "switch_off")


# --- the effective position -----------------------------------------------


@pytest.mark.parametrize(
    ("tier", "stored", "position"),
    [
        (LIVE, "asked", "asked"),
        (LIVE, "allowed", "allowed"),
        (ASKED, "asked", "asked"),
        (ASKED, "allowed", "asked"),
    ],
    ids=["allowed-asked", "allowed-allowed", "asked-asked", "asked-allowed"],
)
def test_the_position_is_the_stored_one_capped_by_the_tier(root, tier, stored, position):
    _host(root, switch={"live": stored})
    verdict = write_verdict(tier, "live")
    assert verdict.allowed is True
    assert verdict.reason is None
    assert verdict.position == position


def test_a_stored_on_reads_asked(root):
    _legacy_host(root, {"live": "on"})
    verdict = write_verdict(LIVE, "live")
    assert verdict.allowed is True
    assert verdict.position == "asked"


def test_a_position_above_the_ceiling_stays_on_disk_and_is_live_again(root):
    path = _host(root, switch={"live": "allowed"})
    before = path.read_bytes()
    assert write_verdict(ASKED, "live").position == "asked"
    _refused(write_verdict(NEVER, "live"), "tier_never")
    assert path.read_bytes() == before
    assert write_verdict(LIVE, "live").position == "allowed"


@pytest.mark.parametrize("switch", [{}, {"va": "allowed"}, {"live": "off"}, {"live": "ON"}])
def test_no_stored_position_for_the_target_is_switch_off(root, switch):
    _legacy_host(root, switch)
    _refused(write_verdict(LIVE, "live"), "switch_off")


def test_the_hosts_ceiling_is_its_own_tier_not_the_map(root):
    """On a host the section is the login's own render; a map entry does not cap it."""
    _host(root, switch={"live": "allowed"})
    capped = {**ASKED, "login_tiers": {ME: "allowed"}}
    assert write_verdict(capped, "live").position == "asked"
    uncapped = {**LIVE, "login_tiers": {ME: "asked"}}
    assert write_verdict(uncapped, "live").position == "allowed"


@pytest.mark.parametrize(
    ("rung", "stored", "position"),
    [
        ("allowed", "asked", "asked"),
        ("allowed", "allowed", "allowed"),
        ("asked", "asked", "asked"),
        ("asked", "allowed", "asked"),
        ("allowed", "on", "asked"),
    ],
    ids=["allowed-asked", "allowed-allowed", "asked-asked", "asked-allowed", "allowed-on"],
)
def test_under_a_tree_the_rung_word_caps(tree, rung, stored, position):
    _tree_record(tree, "alice", switch={"live": stored})
    section = {**LIVE, "login_tiers": {"alice": rung}}
    verdict = write_verdict(section, "live", owner="alice")
    assert verdict.allowed is True
    assert verdict.position == position


def test_under_a_tree_the_containers_own_tier_does_not_cap(tree):
    """The record is the owner's, so the owner's rung is the ceiling, not the section's word."""
    _tree_record(tree, "alice", switch={"live": "allowed"})
    section = {**ASKED, "login_tiers": {"alice": "allowed"}}
    assert write_verdict(section, "live", owner="alice").position == "allowed"


def test_ignoring_the_pin_still_reports_the_position(root, monkeypatch):
    _host(root, switch={"live": "allowed"})
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "live=sandbox")
    _refused(write_verdict(LIVE, "live"), "launch_pinned")
    assert write_verdict(LIVE, "live", ignore_launch_pin=True).position == "allowed"


def test_the_position_is_optional_for_a_refusal_built_by_hand():
    """Callers that refuse before reaching the verdict build one without the field."""
    verdict = WriteVerdict(allowed=False, reason="record_unavailable", target=None, kind=None)
    assert verdict.position is None


# --- a demoted login's stored position -------------------------------------


def test_demoted_login_is_tier_never_and_the_switch_stays(root):
    path = _host(root, switch={"live": "asked"})
    before = path.read_bytes()
    _refused(write_verdict(NEVER, "live"), "tier_never")
    assert path.read_bytes() == before
    record, status = control_context.read_record_status(path=path)
    assert status == "ok"
    assert record.switch == {"live": "asked"}
    # Restored, the same position arms again.
    assert write_verdict(LIVE, "live").allowed is True


def test_demoted_login_in_the_tree_is_tier_never_and_the_switch_stays(tree):
    path = _tree_record(tree, "alice", switch={"live": "asked"})
    before = path.read_bytes()
    section = {**LIVE, "login_tiers": {"alice": "never"}}
    _refused(write_verdict(section, "live", owner="alice"), "tier_never")
    assert path.read_bytes() == before
    assert write_verdict(TREE_SECTION, "live", owner="alice").allowed is True


# --- never raises ---------------------------------------------------------


@pytest.mark.parametrize("section", [None, "garbage", 7, [], {"writes": "allowed"}])
def test_garbage_sections_refuse(section, root):
    _host(root, switch={"live": "asked"})
    verdict = write_verdict(section, "live")
    assert verdict.allowed is False
    assert verdict.reason in REASONS
    assert verdict.position is None


def test_a_reader_that_raises_refuses_rather_than_raises(root, monkeypatch):
    _host(root, switch={"live": "asked"})

    def boom(**_kwargs):
        raise RuntimeError("reader exploded")

    monkeypatch.setattr(control_context, "read_record_status", boom)
    verdict = write_verdict(LIVE, "live")
    assert verdict.allowed is False
    assert verdict.reason in REASONS
    assert verdict.position is None


def test_never_emits_the_lock_reasons(root):
    _host(root, switch={"live": "asked"})
    seen = {write_verdict(section, "live").reason for section in (LIVE, NEVER, None)}
    assert not seen & {"switching", "stopped_since"}


def test_verdict_is_frozen():
    verdict = WriteVerdict(allowed=True, reason=None, target="live", kind="real")
    with pytest.raises(AttributeError):
        verdict.allowed = False  # type: ignore[misc]
