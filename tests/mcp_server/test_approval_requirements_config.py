"""An application tool's approval requirements, read from the rendered config."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from osprey.audit.call import call_scope
from osprey.mcp_server.approval_record import (
    APPROVAL_CONFIG_INVALID,
    APPROVAL_MISSING,
    Refusal,
    Requirements,
    app_approval_entry,
    require_approval_record,
    requirements,
)
from tests._approval_stamp_fixture import ApprovalStamps, use_stamp_dir

TOOL = "mcp__knob__set_knob"

_ENTRY: dict[str, Any] = {
    "prompt": "Set {{ input.value }}",
    "panel_id": "knob",
    "require": {
        "all_of": ["approved_target"],
        "one_of": [["approved_digest", "approved_readback"]],
    },
}

CONFIG: dict[str, Any] = {
    "approval": {"enabled": True, "default_policy": "always"},
    "apps": {"knob": {"approval": {"set_knob": _ENTRY}}},
}

_EXACTLY_ONE = (
    "not run: exactly one of `approved_digest`, `approved_readback` must be carried, "
    "as the approval prompt recorded it (approval_missing). Call the tool again."
)


@pytest.fixture
def stamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ApprovalStamps:
    return use_stamp_dir(monkeypatch, tmp_path / "state")


def _require(config: dict[str, Any], approved: dict[str, Any] | None) -> Refusal | None:
    with call_scope("toolu_stamped", None):
        return require_approval_record(TOOL, config, approved=approved, guarded_run=True)


def _with(require: Any = None, **approval: Any) -> dict[str, Any]:
    config = copy.deepcopy(CONFIG)
    if require is not None:
        config["apps"]["knob"]["approval"]["set_knob"]["require"] = require
    config["approval"].update(approval)
    return config


class TestTheReader:
    def test_the_entry_is_read_by_the_full_name(self) -> None:
        assert app_approval_entry(CONFIG, TOOL) == _ENTRY

    @pytest.mark.parametrize(
        "tool", ["set_knob", "mcp__other__set_knob", "mcp__knob__absent", "mcp__knob", "mcp____x"]
    )
    def test_any_other_spelling_or_an_absent_leaf_is_none(self, tool: str) -> None:
        assert app_approval_entry(CONFIG, tool) is None

    def test_a_leaf_that_is_not_a_mapping_is_none(self) -> None:
        config = {"apps": {"knob": {"approval": {"set_knob": ["x"]}}}}
        assert app_approval_entry(config, TOOL) is None
        assert app_approval_entry({"apps": None}, TOOL) is None

    def test_the_app_tools_requirements(self) -> None:
        assert requirements(TOOL, CONFIG, guarded_run=True) == Requirements(
            ("approved_target",), (("approved_digest", "approved_readback"),)
        )

    def test_a_built_in_keeps_its_static_requirements(self) -> None:
        assert requirements("pyaml_measure", CONFIG, guarded_run=True) == Requirements(
            ("approved_target", "approved_view_sha256"), ()
        )

    def test_approved_target_is_required_under_a_guarded_run_when_the_manifest_omits_it(
        self,
    ) -> None:
        config = _with({"one_of": [["approved_digest", "approved_readback"]]})

        assert requirements(TOOL, config, guarded_run=True).all_of == ("approved_target",)
        assert requirements(TOOL, config).all_of == ()


class TestTheGrammar:
    def test_a_step_shaped_call_passes(self, stamps: ApprovalStamps) -> None:
        stamps.ask(TOOL, approved_target="live", approved_digest="d")

        assert _require(CONFIG, {"approved_target": "live", "approved_digest": "d"}) is None

    def test_a_both_shaped_call_refuses(self, stamps: ApprovalStamps) -> None:
        stamps.ask(TOOL, approved_target="live", approved_digest="d")

        refusal = _require(
            CONFIG,
            {"approved_target": "live", "approved_digest": "d", "approved_readback": "r"},
        )

        assert refusal == Refusal(APPROVAL_MISSING, _EXACTLY_ONE)

    def test_a_neither_shaped_call_refuses(self, stamps: ApprovalStamps) -> None:
        stamps.ask(TOOL, approved_target="live", approved_digest="d")

        refusal = _require(CONFIG, {"approved_target": "live", "approved_readback": None})

        assert refusal == Refusal(APPROVAL_MISSING, _EXACTLY_ONE)

    def test_a_stamp_holding_both_of_a_group_refuses(self, stamps: ApprovalStamps) -> None:
        stamps.ask(TOOL, approved_target="live", approved_digest="d", approved_readback="r")

        refusal = _require(CONFIG, {"approved_target": "live", "approved_digest": "d"})

        assert refusal == Refusal(APPROVAL_MISSING, _EXACTLY_ONE)

    def test_a_one_of_value_differing_from_the_stamp_refuses(self, stamps: ApprovalStamps) -> None:
        stamps.ask(TOOL, approved_target="live", approved_digest="d")

        refusal = _require(CONFIG, {"approved_target": "live", "approved_digest": "other"})

        assert refusal is not None and refusal.error_type == APPROVAL_MISSING
        assert "`approved_digest` differs" in refusal.message

    def test_a_stamp_for_another_apps_tool_refuses(self, stamps: ApprovalStamps) -> None:
        stamps.ask("mcp__other__set_knob", approved_target="live", approved_digest="d")

        refusal = _require(CONFIG, {"approved_target": "live", "approved_digest": "d"})

        assert refusal is not None and refusal.error_type == APPROVAL_MISSING
        assert "left no record" in refusal.message

    def test_approved_target_is_required_even_when_the_manifest_omits_it(
        self, stamps: ApprovalStamps
    ) -> None:
        config = _with({"one_of": [["approved_digest", "approved_readback"]]})
        stamps.ask(TOOL, approved_target="live", approved_digest="d")

        refusal = _require(config, {"approved_digest": "d"})

        assert refusal is not None and "`approved_target` differs" in refusal.message

    def test_a_dispatch_policy_stamp_passes(self, stamps: ApprovalStamps) -> None:
        stamps.ask(TOOL, approver="dispatch_policy", approved_target="live", approved_digest="d")

        assert _require(CONFIG, {"approved_target": "live", "approved_digest": "d"}) is None


@pytest.mark.usefixtures("stamps")
class TestWhenTheToolAsks:
    def test_skip_passes_with_no_stamp_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from osprey.mcp_server import approval_record

        def _unread(_: str | None) -> None:
            raise AssertionError("the stamp was read")

        monkeypatch.setattr(approval_record, "approval_ask_stamp", _unread)

        assert _require(_with(tools={TOOL: "skip"}), None) is None

    def test_selective_asks(self) -> None:
        refusal = _require(
            _with(tools={TOOL: "selective"}), {"approved_target": "live", "approved_digest": "d"}
        )

        assert refusal is not None and refusal.error_type == APPROVAL_MISSING


class TestAMalformedRequire:
    @pytest.mark.parametrize(
        "require",
        [
            pytest.param({"all_of": "approved_target"}, id="all-of-not-a-list"),
            pytest.param({"all_of": [""]}, id="all-of-empty-name"),
            pytest.param({"one_of": [[]]}, id="empty-group"),
            pytest.param({"one_of": ["approved_digest"]}, id="group-not-a-list"),
            pytest.param({"one_of": [["approved_digest", 3]]}, id="group-name-not-a-string"),
            pytest.param(["approved_target"], id="require-not-a-mapping"),
        ],
    )
    def test_it_refuses_closed(self, stamps: ApprovalStamps, require: Any) -> None:
        stamps.ask(TOOL, approved_target="live", approved_digest="d")

        refusal = _require(_with(require), {"approved_target": "live", "approved_digest": "d"})

        assert refusal == Refusal(
            APPROVAL_CONFIG_INVALID,
            f"not run: apps.knob.approval.set_knob.require is {require!r} "
            "(approval_config_invalid).",
        )
