"""The shared ask-stamp module: its spellings and its per-tool rule."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.mcp_server import approval_stamp as stamp_mod
from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.python_executor import executor

_TOOL = "mcp__python__pyaml_measure"


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


def test_the_approved_fields() -> None:
    assert stamp_mod.APPROVED_FIELDS == ("approved_target", "approved_view_sha256")


def test_every_rule_names_only_approved_fields() -> None:
    known = set(stamp_mod.APPROVED_FIELDS)
    rules = [stamp_mod.DEFAULT_REQUIRED_FIELDS, *stamp_mod.TOOL_REQUIRED_FIELDS.values()]
    for rule in rules:
        assert "approved_target" in rule.all_of
        assert set(rule.all_of) <= known


def test_the_per_tool_rules() -> None:
    assert stamp_mod.required_fields("execute") == stamp_mod.RequiredFields(("approved_target",))
    assert stamp_mod.required_fields("pyaml_measure").all_of == (
        "approved_target",
        "approved_view_sha256",
    )


# --- stamp_refusal ----------------------------------------------------------


def _stamp(**fields: str) -> dict[str, str]:
    return {"tool": _TOOL, "tool_use_id": "toolu_1", **fields}


def test_no_stamp_or_another_tool_is_refused() -> None:
    approved = {"approved_target": "live", "approved_view_sha256": "v"}
    assert "left no record" in stamp_mod.stamp_refusal(None, "pyaml_measure", approved)
    other = {**_stamp(approved_target="live", approved_view_sha256="v"), "tool": "x__execute"}
    assert "left no record" in stamp_mod.stamp_refusal(other, "pyaml_measure", approved)


def test_a_call_carrying_every_required_field_as_stamped_is_admitted() -> None:
    approved = {"approved_target": "live", "approved_view_sha256": "v"}
    stamp = _stamp(approved_target="live", approved_view_sha256="v")
    assert stamp_mod.stamp_refusal(stamp, "pyaml_measure", approved) is None


@pytest.mark.parametrize("field", ["approved_target", "approved_view_sha256"])
@pytest.mark.parametrize("value", [None, "other"])
def test_a_required_field_absent_or_differing_is_refused(field: str, value) -> None:
    stamp = _stamp(approved_target="live", approved_view_sha256="v")
    approved = {"approved_target": "live", "approved_view_sha256": "v", field: value}
    refusal = stamp_mod.stamp_refusal(stamp, "pyaml_measure", approved)
    assert refusal is not None and f"`{field}` differs" in refusal
    assert "(approval_missing)" in refusal


def test_an_optional_field_the_call_carries_must_match() -> None:
    stamp = {"tool": "mcp__python__execute", "approved_target": "live"}
    approved = {"approved_target": "live", "approved_view_sha256": "v"}
    refusal = stamp_mod.stamp_refusal(stamp, "execute", approved)
    assert refusal is not None and "`approved_view_sha256` differs" in refusal
    assert stamp_mod.stamp_refusal(stamp, "execute", {"approved_target": "live"}) is None


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
