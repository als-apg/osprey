"""The shared ask-stamp module: its spellings, its per-tool rule, its values digest."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from osprey.mcp_server import approval_stamp as stamp_mod
from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.python_executor import executor

_TOOL = "mcp__trajectory__trajectory_apply"


@pytest.fixture
def hook():
    from tests.hooks.conftest import import_hook

    return import_hook("osprey_approval")


def test_the_shared_module_spells_the_ask_stamp_like_the_hook(hook) -> None:
    assert stamp_mod.APPROVAL_ASK_PREFIX == hook.APPROVAL_ASK_PREFIX
    assert stamp_mod.APPROVAL_ASK_SUFFIX == hook.APPROVAL_ASK_SUFFIX
    assert stamp_mod.ask_stamp_filename("toolu_x") == hook.approval_ask_filename("toolu_x")
    assert all(name.startswith(hook.APPROVED_FIELD_PREFIX) for name in stamp_mod.APPROVED_FIELDS)


def test_the_executor_reads_the_stamp_through_the_shared_module() -> None:
    assert executor.APPROVAL_ASK_PREFIX is stamp_mod.APPROVAL_ASK_PREFIX
    assert executor.APPROVAL_ASK_SUFFIX is stamp_mod.APPROVAL_ASK_SUFFIX
    assert executor.APPROVED_FIELDS is stamp_mod.APPROVED_FIELDS
    assert not hasattr(executor, "_approved_call_stamp")
    assert not hasattr(executor, "_TOOL_REQUIRED_APPROVED_FIELDS")


def test_the_trajectory_fields_are_approved_fields() -> None:
    assert set(stamp_mod.APPROVED_FIELDS) == {
        "approved_target",
        "approved_view_sha256",
        "approved_step_digest",
        "approved_values_sha256",
    }
    assert len(set(stamp_mod.APPROVED_FIELDS)) == len(stamp_mod.APPROVED_FIELDS)


def test_every_rule_names_only_approved_fields() -> None:
    known = set(stamp_mod.APPROVED_FIELDS)
    rules = [stamp_mod.DEFAULT_REQUIRED_FIELDS, *stamp_mod.TOOL_REQUIRED_FIELDS.values()]
    for rule in rules:
        assert "approved_target" in rule.all_of
        assert set(rule.all_of) <= known
        for group in rule.exactly_one_of:
            assert len(group) >= 2
            assert set(group) <= known
            assert not set(group) & set(rule.all_of)


def test_the_per_tool_rules() -> None:
    assert stamp_mod.required_fields("execute") == stamp_mod.RequiredFields(("approved_target",))
    assert stamp_mod.required_fields("pyaml_measure").all_of == (
        "approved_target",
        "approved_view_sha256",
    )
    shot = stamp_mod.required_fields("trajectory_shot")
    assert shot.all_of == ("approved_target",) and shot.exactly_one_of == ()
    apply = stamp_mod.required_fields("trajectory_apply")
    assert apply.all_of == ("approved_target",)
    assert apply.exactly_one_of == (("approved_step_digest", "approved_values_sha256"),)


# --- stamp_refusal ----------------------------------------------------------


def _stamp(**fields: str) -> dict[str, str]:
    return {"tool": _TOOL, "tool_use_id": "toolu_1", **fields}


def test_no_stamp_or_another_tool_is_refused() -> None:
    approved = {"approved_target": "live", "approved_step_digest": "d"}
    assert "left no record" in stamp_mod.stamp_refusal(None, "trajectory_apply", approved)
    other = {**_stamp(approved_target="live", approved_step_digest="d"), "tool": "x__shot"}
    assert "left no record" in stamp_mod.stamp_refusal(other, "trajectory_apply", approved)


@pytest.mark.parametrize("digest", ["approved_step_digest", "approved_values_sha256"])
def test_an_apply_with_the_target_and_one_digest_is_admitted(digest: str) -> None:
    approved = {"approved_target": "live", digest: "abc"}
    stamp = _stamp(approved_target="live", **{digest: "abc"})
    assert stamp_mod.stamp_refusal(stamp, "trajectory_apply", approved) is None


@pytest.mark.parametrize(
    "approved",
    [
        {"approved_target": "live"},
        {
            "approved_target": "live",
            "approved_step_digest": "abc",
            "approved_values_sha256": "abc",
        },
        {"approved_target": "live", "approved_step_digest": None, "approved_values_sha256": None},
    ],
    ids=["neither", "both", "both-absent"],
)
def test_an_apply_without_exactly_one_digest_is_refused(approved) -> None:
    stamp = _stamp(approved_target="live", approved_step_digest="abc", approved_values_sha256="abc")
    refusal = stamp_mod.stamp_refusal(stamp, "trajectory_apply", approved)
    assert refusal is not None and "exactly one of" in refusal
    assert "(approval_missing)" in refusal


def test_an_apply_whose_digest_differs_is_refused() -> None:
    stamp = _stamp(approved_target="live", approved_step_digest="abc")
    approved = {"approved_target": "live", "approved_step_digest": "xyz"}
    refusal = stamp_mod.stamp_refusal(stamp, "trajectory_apply", approved)
    assert refusal is not None and "`approved_step_digest` differs" in refusal


def test_an_apply_carrying_the_other_digest_than_the_stamp_is_refused() -> None:
    stamp = _stamp(approved_target="live", approved_step_digest="abc")
    approved = {"approved_target": "live", "approved_values_sha256": "abc"}
    refusal = stamp_mod.stamp_refusal(stamp, "trajectory_apply", approved)
    assert refusal is not None and "`approved_values_sha256` differs" in refusal


@pytest.mark.parametrize("tool", ["trajectory_apply", "trajectory_shot"])
def test_a_trajectory_call_without_or_with_another_target_is_refused(tool: str) -> None:
    stamp = {**_stamp(approved_target="live", approved_step_digest="abc"), "tool": f"x__{tool}"}
    for target in (None, "other"):
        approved = {"approved_target": target, "approved_step_digest": "abc"}
        refusal = stamp_mod.stamp_refusal(stamp, tool, approved)
        assert refusal is not None and "`approved_target` differs" in refusal


def test_a_shot_needs_only_the_target() -> None:
    stamp = {"tool": "mcp__trajectory__trajectory_shot", "approved_target": "live"}
    assert stamp_mod.stamp_refusal(stamp, "trajectory_shot", {"approved_target": "live"}) is None


# --- read_ask_stamp ---------------------------------------------------------


def test_read_ask_stamp_reads_the_file_the_hook_names(tmp_path: Path, monkeypatch, hook) -> None:
    monkeypatch.setattr(target_state, "state_dir", lambda: tmp_path)
    record = _stamp(approved_target="live")
    (tmp_path / hook.approval_ask_filename("toolu_1")).write_text(json.dumps(record))
    assert stamp_mod.read_ask_stamp("toolu_1") == record


def test_read_ask_stamp_is_none_without_an_id_a_file_or_a_mapping(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(target_state, "state_dir", lambda: tmp_path)
    assert stamp_mod.read_ask_stamp(None) is None
    assert stamp_mod.read_ask_stamp("") is None
    assert stamp_mod.read_ask_stamp("toolu_missing") is None
    (tmp_path / stamp_mod.ask_stamp_filename("toolu_list")).write_text("[1, 2]")
    assert stamp_mod.read_ask_stamp("toolu_list") is None


def test_read_ask_stamp_is_none_when_the_state_dir_cannot_resolve(monkeypatch) -> None:
    def _broken() -> Path:
        raise RuntimeError("no state dir")

    monkeypatch.setattr(target_state, "state_dir", _broken)
    assert stamp_mod.read_ask_stamp("toolu_1") is None


# --- approved_values_sha256 -------------------------------------------------

_VALUES = [{"name": "HCM2", "before": -0.25}, {"name": "HCM1", "before": 1.5}]


def test_the_values_digest_is_sha256_of_the_canonical_json() -> None:
    canonical = '{"model":"ring","pairs":[["HCM1",1.5],["HCM2",-0.25]],"target":"live"}'
    expected = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    assert stamp_mod.approved_values_sha256("live", "ring", _VALUES) == expected


def test_the_values_digest_ignores_value_order_and_extra_keys() -> None:
    reordered = [{"before": 1.5, "name": "HCM1", "extra": 9}, {"name": "HCM2", "before": -0.25}]
    assert stamp_mod.approved_values_sha256(
        "live", "ring", reordered
    ) == stamp_mod.approved_values_sha256("live", "ring", _VALUES)


@pytest.mark.parametrize(
    ("target", "model", "values"),
    [
        ("sim", "ring", _VALUES),
        ("live", "booster", _VALUES),
        ("live", "ring", [{"name": "HCM2", "before": -0.25}, {"name": "HCM1", "before": 1.25}]),
        ("live", "ring", _VALUES[:1]),
    ],
    ids=["target", "model", "value", "dropped-pair"],
)
def test_the_values_digest_changes_with_what_the_prompt_showed(target, model, values) -> None:
    base = stamp_mod.approved_values_sha256("live", "ring", _VALUES)
    assert stamp_mod.approved_values_sha256(target, model, values) != base


def test_the_values_digest_is_a_hex_sha256() -> None:
    digest = stamp_mod.approved_values_sha256("live", "ring", [])
    assert len(digest) == 64 and int(digest, 16) >= 0


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_the_values_digest_refuses_a_non_finite_value(bad: float) -> None:
    with pytest.raises(ValueError):
        stamp_mod.approved_values_sha256("live", "ring", [{"name": "HCM1", "before": bad}])


def test_the_values_digest_refuses_an_entry_without_name_or_before() -> None:
    with pytest.raises(KeyError):
        stamp_mod.approved_values_sha256("live", "ring", [{"name": "HCM1"}])
