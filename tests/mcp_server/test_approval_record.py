"""Stage 1 of approval: ``require_approval_record`` over the ask stamp."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey.audit.call import call_scope
from osprey.cli.build_posture_check import (
    GUARDED_TOOLS,
    STAGE1_TOOLS,
    _governed_matchers,
    _short_name,
    ask_capable,
)
from osprey.mcp_server import approval_record
from osprey.mcp_server.approval_record import (
    ANSWERED_AT_KEY,
    APPROVAL_CONFIG_INVALID,
    APPROVAL_MISSING,
    APPROVED_FIELDS,
    APPROVER_DISPATCH_POLICY,
    APPROVER_HUMAN,
    APPROVER_WRITE_SWITCH,
    POSITION_ALLOWED,
    POSITION_ASKED,
    PROMPTED_APPROVERS,
    STAMP_APPROVERS,
    STAMP_POSITIONS,
    TOOL_REQUIRED_FIELDS,
    Refusal,
    Requirements,
    require_approval_record,
    required_fields,
    requirements,
)
from tests._approval_stamp_fixture import ApprovalStamps, use_stamp_dir

#: Every tool stage 1 guards.
_TOOLS: tuple[str, ...] = (*GUARDED_TOOLS, *STAGE1_TOOLS)

#: The tools the write switch decides in a default render, by short name.
_GOVERNED: tuple[str, ...] = tuple(map(_short_name, _governed_matchers({}) or ()))

#: The tools stage 1 guards that the write switch does not decide.
_UNGOVERNED: tuple[str, ...] = tuple(tool for tool in _TOOLS if tool not in _GOVERNED)

#: Configs under which the policy flow asks nobody.
_ASKS_NOBODY: dict[str, dict[str, Any]] = {
    "disabled": {"approval": {"enabled": False}},
    "default-skip": {"approval": {"default_policy": "skip"}},
    "tool-skip": {"approval": {"default_policy": "always", "tools": dict.fromkeys(_TOOLS, "skip")}},
    "disabled-and-skip": {"approval": {"enabled": False, "default_policy": "skip"}},
}

#: A rendered config whose approval policy asks about every tool.
_ASKING: dict[str, Any] = {"approval": {"enabled": True, "default_policy": "always"}}

_NO_RECORD = (
    "not run: this call's approval prompt left no record (approval_missing). "
    "Call the tool again so the approval prompt can ask."
)


@pytest.fixture
def stamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ApprovalStamps:
    return use_stamp_dir(monkeypatch, tmp_path / "state")


def _require(tool: str, config: dict[str, Any], **kwargs: Any) -> Refusal | None:
    with call_scope("toolu_stamped", None):
        return require_approval_record(tool, config, **kwargs)


def test_the_module_spells_the_ask_stamp_like_the_hook() -> None:
    from tests.hooks.conftest import import_hook

    hook = import_hook("osprey_approval")
    assert approval_record.APPROVAL_ASK_PREFIX == hook.APPROVAL_ASK_PREFIX
    assert approval_record.APPROVAL_ASK_SUFFIX == hook.APPROVAL_ASK_SUFFIX


class TestTheStampSpellings:
    """The ``approver``, ``position`` and ``answered_at`` words are the hook's."""

    def test_the_approver_and_position_words_are_the_hooks(self) -> None:
        from tests.hooks.conftest import import_hook

        hook = import_hook("osprey_approval")
        assert hook.APPROVER_WRITE_SWITCH == APPROVER_WRITE_SWITCH
        assert (hook._POSITION_ASKED, hook._POSITION_ALLOWED) == (POSITION_ASKED, POSITION_ALLOWED)
        assert STAMP_POSITIONS == (POSITION_ASKED, POSITION_ALLOWED)
        assert STAMP_APPROVERS == ("human", "dispatch_policy", "write_switch")
        assert PROMPTED_APPROVERS == {APPROVER_HUMAN, APPROVER_DISPATCH_POLICY}
        assert APPROVER_WRITE_SWITCH not in PROMPTED_APPROVERS

    @pytest.mark.parametrize(
        ("dispatch", "position", "approver"),
        [
            (False, None, APPROVER_HUMAN),
            (False, POSITION_ASKED, APPROVER_HUMAN),
            (False, POSITION_ALLOWED, APPROVER_WRITE_SWITCH),
            (True, None, APPROVER_DISPATCH_POLICY),
            (True, POSITION_ASKED, APPROVER_DISPATCH_POLICY),
            (True, POSITION_ALLOWED, APPROVER_DISPATCH_POLICY),
        ],
    )
    def test_the_hooks_stamp_reads_back_in_this_modules_words(
        self,
        stamps: ApprovalStamps,
        monkeypatch: pytest.MonkeyPatch,
        dispatch: bool,
        position: str | None,
        approver: str,
    ) -> None:
        """The stamp the hook files is the one this module reads, and admits the call."""
        from tests.hooks.conftest import import_hook

        hook = import_hook("osprey_approval")
        monkeypatch.setattr(hook, "_approval_state_dir", lambda _input: str(stamps.state))
        if dispatch:
            monkeypatch.setenv("OSPREY_DISPATCH_RUN", "1")
        else:
            monkeypatch.delenv("OSPREY_DISPATCH_RUN", raising=False)
        hook_input = {
            "tool_use_id": "toolu_stamped",
            "tool_name": "mcp__controls__channel_write",
            "session_id": "s1",
            "permission_mode": "default",
        }

        hook._stamp_ask(hook_input, approved={"approved_target": "live"}, position=position)

        stamp = approval_record.approval_ask_stamp("toolu_stamped")
        assert stamp is not None
        assert stamp["approver"] == approver and approver in STAMP_APPROVERS
        if position is None:
            assert "position" not in stamp
        else:
            assert stamp["position"] == position and position in STAMP_POSITIONS
        assert (ANSWERED_AT_KEY in stamp) is (position == POSITION_ALLOWED)
        if position == POSITION_ALLOWED:
            assert stamp[ANSWERED_AT_KEY] == stamp["asked_at"]
        assert _require("channel_write", _ASKING, approved={"approved_target": "live"}) is None


class TestTheGovernedRecord:
    """A governed tool needs the record whatever ``approval.*`` says."""

    def test_the_default_render_governs_the_write_tools(self) -> None:
        assert set(_GOVERNED) == {"channel_write", "execute", "execute_file", "pyaml_measure"}

    @pytest.mark.parametrize("tool", _GOVERNED)
    @pytest.mark.parametrize("config", list(_ASKS_NOBODY.values()), ids=list(_ASKS_NOBODY))
    def test_a_governed_tool_is_ask_capable_where_the_policy_asks_nobody(
        self, tool: str, config: dict[str, Any]
    ) -> None:
        assert ask_capable(config, tool) is True

    @pytest.mark.usefixtures("stamps")
    @pytest.mark.parametrize("tool", _UNGOVERNED)
    @pytest.mark.parametrize("config", list(_ASKS_NOBODY.values()), ids=list(_ASKS_NOBODY))
    def test_an_ungoverned_tool_keeps_the_policy_reading(
        self, tool: str, config: dict[str, Any]
    ) -> None:
        assert ask_capable(config, tool) is False
        assert _require(tool, config) is None

    @pytest.mark.usefixtures("stamps")
    @pytest.mark.parametrize("tool", _GOVERNED)
    @pytest.mark.parametrize("config", list(_ASKS_NOBODY.values()), ids=list(_ASKS_NOBODY))
    def test_a_governed_call_without_a_stamp_refuses_where_the_policy_asks_nobody(
        self, tool: str, config: dict[str, Any]
    ) -> None:
        assert _require(tool, config) == Refusal(APPROVAL_MISSING, _NO_RECORD)

    @pytest.mark.parametrize("tool", _GOVERNED)
    @pytest.mark.parametrize("config", [_ASKING, *_ASKS_NOBODY.values()])
    @pytest.mark.parametrize(
        ("approver", "position"),
        [
            pytest.param(APPROVER_WRITE_SWITCH, POSITION_ALLOWED, id="switch-allowed"),
            pytest.param(APPROVER_HUMAN, POSITION_ASKED, id="human-asked"),
            pytest.param(APPROVER_DISPATCH_POLICY, POSITION_ALLOWED, id="dispatch-allowed"),
        ],
    )
    def test_a_write_switch_stamp_is_a_record_like_a_humans(
        self,
        stamps: ApprovalStamps,
        tool: str,
        config: dict[str, Any],
        approver: str,
        position: str,
    ) -> None:
        fields = dict.fromkeys(required_fields(tool, guarded_run=True), "x")
        extra = {ANSWERED_AT_KEY: 1.0} if position == POSITION_ALLOWED else {}
        stamps.ask(tool, approver=approver, position=position, **extra, **fields)

        assert _require(tool, config, approved=fields, guarded_run=True) is None

    @pytest.mark.parametrize("tool", _GOVERNED)
    def test_a_write_switch_stamp_still_binds_the_approved_fields(
        self, stamps: ApprovalStamps, tool: str
    ) -> None:
        fields = dict.fromkeys(required_fields(tool, guarded_run=True), "x")
        stamps.ask(tool, approver=APPROVER_WRITE_SWITCH, position=POSITION_ALLOWED, **fields)

        refusal = _require(
            tool, _ASKS_NOBODY["disabled"], approved={"approved_target": "y"}, guarded_run=True
        )

        assert refusal is not None and refusal.error_type == APPROVAL_MISSING
        assert "`approved_target` differs" in refusal.message

    @pytest.mark.parametrize("tool", _GOVERNED)
    def test_a_write_switch_stamp_for_another_tool_refuses(
        self, stamps: ApprovalStamps, tool: str
    ) -> None:
        other = "queue_add"
        stamps.ask(other, approver=APPROVER_WRITE_SWITCH, position=POSITION_ALLOWED)

        assert _require(tool, _ASKS_NOBODY["disabled"]) == Refusal(APPROVAL_MISSING, _NO_RECORD)


class TestTheConfigFault:
    """A switch that is not a bool refuses every call, asking or not, stamped or not."""

    @pytest.mark.parametrize("tool", _TOOLS)
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({"approval": {"enabled": "yes"}}, id="enabled-not-bool"),
            pytest.param({"approval": None}, id="section-not-mapping"),
            pytest.param(
                {"approval": {"enabled": 1, "tools": {"x": "skip"}, "default_policy": "skip"}},
                id="skip-policy",
            ),
        ],
    )
    def test_a_fault_refuses_everywhere(
        self, stamps: ApprovalStamps, tool: str, config: dict[str, Any]
    ) -> None:
        stamps.ask(tool, approved_target="live")

        refusal = _require(tool, config, approved={"approved_target": "live"})

        assert refusal is not None
        assert refusal.error_type == APPROVAL_CONFIG_INVALID
        assert refusal.message.startswith("not run: approval")
        assert refusal.message.endswith("(approval_config_invalid).")


class TestWhereItApplies:
    """The record is required exactly where the tool's call is put to a human."""

    @pytest.mark.usefixtures("stamps")
    @pytest.mark.parametrize("tool", _TOOLS)
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({}, id="unstated"),
            pytest.param(_ASKING, id="always"),
            pytest.param({"approval": {"enabled": False}}, id="disabled"),
            pytest.param({"approval": {"default_policy": "skip"}}, id="default-skip"),
            pytest.param({"approval": {"default_policy": "selective"}}, id="selective"),
            pytest.param(
                {"approval": {"default_policy": "always", "tools": dict.fromkeys(_TOOLS, "skip")}},
                id="tool-skip",
            ),
        ],
    )
    def test_the_check_applies_exactly_where_ask_capable(
        self, tool: str, config: dict[str, Any]
    ) -> None:
        refusal = _require(tool, config)

        assert (refusal is not None) is ask_capable(config, tool)
        if refusal is not None:
            assert refusal == Refusal(APPROVAL_MISSING, _NO_RECORD)


class TestTheStamp:
    """A missing, wrong-tool or unreadable stamp refuses ``approval_missing``."""

    @pytest.mark.parametrize("tool", _TOOLS)
    def test_a_stamp_for_the_tool_satisfies_it(self, stamps: ApprovalStamps, tool: str) -> None:
        """A stamp naming the tool, carrying what the tool's rule demands, admits the call."""
        fields = dict.fromkeys(required_fields(tool), "x")
        stamps.ask(tool, **fields)

        assert _require(tool, _ASKING, approved=fields or None) is None

    @pytest.mark.parametrize("tool", STAGE1_TOOLS)
    def test_a_dispatch_policy_stamp_satisfies_it(self, stamps: ApprovalStamps, tool: str) -> None:
        stamps.ask(tool, approver="dispatch_policy")

        assert _require(tool, _ASKING) is None

    @pytest.mark.usefixtures("stamps")
    @pytest.mark.parametrize("tool", STAGE1_TOOLS)
    def test_a_missing_stamp_refuses(self, tool: str) -> None:
        assert _require(tool, _ASKING) == Refusal(APPROVAL_MISSING, _NO_RECORD)

    def test_no_call_in_scope_refuses(self, stamps: ApprovalStamps) -> None:
        stamps.ask("channel_write")

        assert require_approval_record("channel_write", _ASKING) == Refusal(
            APPROVAL_MISSING, _NO_RECORD
        )

    def test_another_calls_stamp_refuses(self, stamps: ApprovalStamps) -> None:
        stamps.ask("channel_write", tool_use_id="toolu_other")

        assert _require("channel_write", _ASKING) == Refusal(APPROVAL_MISSING, _NO_RECORD)

    @pytest.mark.parametrize(
        ("stamped", "tool"),
        [
            ("queue_add", "queue_start"),
            ("queue_start", "queue_add"),
            ("channel_write", "phoebus_drive"),
            ("execute", "channel_write"),
        ],
    )
    def test_a_wrong_tool_stamp_refuses(
        self, stamps: ApprovalStamps, stamped: str, tool: str
    ) -> None:
        stamps.ask(stamped)

        assert _require(tool, _ASKING) == Refusal(APPROVAL_MISSING, _NO_RECORD)

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("{not json", id="not-json"),
            pytest.param("[1, 2]", id="not-a-mapping"),
            pytest.param("", id="empty"),
            pytest.param(json.dumps({"tool_use_id": "toolu_stamped"}), id="no-tool"),
        ],
    )
    def test_an_unreadable_stamp_refuses(self, stamps: ApprovalStamps, content: str) -> None:
        stamps.ask("channel_write").write_text(content, encoding="utf-8")

        assert _require("channel_write", _ASKING) == Refusal(APPROVAL_MISSING, _NO_RECORD)

    def test_an_unresolvable_state_directory_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from osprey.mcp_server.control_system import target_state

        def _unresolvable() -> Path:
            raise RuntimeError("no state directory")

        monkeypatch.setattr(target_state, "state_dir", _unresolvable)

        assert _require("queue_start", _ASKING) == Refusal(APPROVAL_MISSING, _NO_RECORD)


class TestTheApprovedFields:
    """Every field the call carries, and every required one, equals the stamp's."""

    def test_no_fields_and_none_required_is_valid(self, stamps: ApprovalStamps) -> None:
        stamps.ask("execute", approved_target="live")

        assert _require("execute", _ASKING, approved=None) is None

    def test_an_equal_field_passes(self, stamps: ApprovalStamps) -> None:
        stamps.ask("queue_start", approved_target="sim")

        assert _require("queue_start", _ASKING, approved={"approved_target": "sim"}) is None

    def test_a_differing_field_refuses(self, stamps: ApprovalStamps) -> None:
        stamps.ask("queue_start", approved_target="sim")

        refusal = _require("queue_start", _ASKING, approved={"approved_target": "live"})

        assert refusal == Refusal(
            APPROVAL_MISSING,
            "not run: `approved_target` differs from what the approval prompt recorded "
            "(approval_missing). Call the tool again.",
        )

    def test_a_required_field_the_call_lacks_refuses(self, stamps: ApprovalStamps) -> None:
        stamps.ask("execute", approved_target="live")

        refusal = _require("execute", _ASKING, approved={"approved_target": None}, guarded_run=True)

        assert refusal is not None and refusal.error_type == APPROVAL_MISSING
        assert "`approved_target` differs" in refusal.message

    def test_an_absent_optional_field_is_not_compared(self, stamps: ApprovalStamps) -> None:
        stamps.ask("execute", approved_target="live", approved_view_sha256="abc")

        assert (
            _require(
                "execute",
                _ASKING,
                approved={"approved_target": "live", "approved_view_sha256": None},
            )
            is None
        )

    def test_a_field_beyond_the_executors_is_compared_too(self, stamps: ApprovalStamps) -> None:
        stamps.ask("phoebus_drive", approved_screen="a.bob")

        assert _require("phoebus_drive", _ASKING, approved={"approved_screen": "a.bob"}) is None
        refusal = _require("phoebus_drive", _ASKING, approved={"approved_screen": "b.bob"})
        assert refusal is not None and "`approved_screen` differs" in refusal.message

    def test_a_refusal_reads_as_its_message(self) -> None:
        assert str(Refusal(APPROVAL_MISSING, _NO_RECORD)) == _NO_RECORD


class TestTheRequiredFields:
    """One rule of which approved fields an asking call must carry, held here."""

    _MEASURE = {"approved_target": "live", "approved_view_sha256": "v"}

    def test_every_rule_names_only_approved_fields(self) -> None:
        known = set(APPROVED_FIELDS)
        for tool in (*_TOOLS, "execute_file", "pyaml_measure", "unknown"):
            for guarded_run in (False, True):
                assert set(required_fields(tool, guarded_run=guarded_run)) <= known
        for fields in TOOL_REQUIRED_FIELDS.values():
            assert set(fields) <= known

    def test_the_per_tool_rules_over_guarded_run(self) -> None:
        assert required_fields("execute") == ()
        assert required_fields("execute", guarded_run=True) == ("approved_target",)
        assert required_fields("execute_file", guarded_run=True) == ("approved_target",)
        assert required_fields("pyaml_measure") == ("approved_view_sha256",)
        assert required_fields("pyaml_measure", guarded_run=True) == (
            "approved_target",
            "approved_view_sha256",
        )
        for tool in ("channel_write", "phoebus_drive", "queue_add", "queue_start"):
            assert required_fields(tool) == ()

    def test_the_guarded_run_rule_is_the_hooks_binding(self) -> None:
        """A guarded run carries the target the hook binds; a measurement its view too."""
        from tests.hooks.conftest import import_hook

        hook = import_hook("osprey_approval")
        assert required_fields("execute", guarded_run=True) == (hook._APPROVED_TARGET_KEY,)
        assert required_fields("pyaml_measure", guarded_run=True) == (
            hook._APPROVED_TARGET_KEY,
            hook._APPROVED_VIEW_KEY,
        )
        assert all(name.startswith(hook.APPROVED_FIELD_PREFIX) for name in APPROVED_FIELDS)

    def test_a_guarded_call_carrying_every_required_field_as_stamped_is_admitted(
        self, stamps: ApprovalStamps
    ) -> None:
        stamps.ask("pyaml_measure", **self._MEASURE)

        assert _require("pyaml_measure", _ASKING, approved=self._MEASURE, guarded_run=True) is None

    @pytest.mark.parametrize("field", ["approved_target", "approved_view_sha256"])
    @pytest.mark.parametrize("value", [None, "other"])
    def test_a_required_field_absent_or_differing_refuses(
        self, stamps: ApprovalStamps, field: str, value: str | None
    ) -> None:
        stamps.ask("pyaml_measure", **self._MEASURE)

        refusal = _require(
            "pyaml_measure", _ASKING, approved={**self._MEASURE, field: value}, guarded_run=True
        )

        assert refusal is not None and refusal.error_type == APPROVAL_MISSING
        assert f"`{field}` differs" in refusal.message
        assert "(approval_missing)" in refusal.message

    def test_an_optional_field_the_call_carries_must_match(self, stamps: ApprovalStamps) -> None:
        stamps.ask("execute", approved_target="live")

        refusal = _require("execute", _ASKING, approved=self._MEASURE, guarded_run=True)

        assert refusal is not None and "`approved_view_sha256` differs" in refusal.message
        assert (
            _require("execute", _ASKING, approved={"approved_target": "live"}, guarded_run=True)
            is None
        )

    def test_the_static_table_names_no_application_tool(self) -> None:
        assert not any(tool.startswith("mcp__") for tool in TOOL_REQUIRED_FIELDS)

    @pytest.mark.parametrize("tool", _TOOLS)
    @pytest.mark.parametrize("guarded_run", [False, True])
    def test_a_built_ins_requirements_are_its_required_fields(
        self, tool: str, guarded_run: bool
    ) -> None:
        assert requirements(tool, {}, guarded_run=guarded_run) == Requirements(
            required_fields(tool, guarded_run=guarded_run), ()
        )

    def test_the_executor_holds_no_spelling_or_rule_of_its_own(self) -> None:
        import importlib.util

        from osprey.mcp_server.python_executor import executor

        assert importlib.util.find_spec("osprey.mcp_server.approval_stamp") is None
        for name in (
            "_approved_call_stamp",
            "_REQUIRED_APPROVED_FIELDS",
            "_TOOL_REQUIRED_APPROVED_FIELDS",
            "_ApprovalMissing",
            "_ApprovalConfigInvalid",
            "APPROVAL_ASK_PREFIX",
        ):
            assert not hasattr(executor, name)
        assert executor.FAILURE_KIND_APPROVAL_MISSING is APPROVAL_MISSING
        assert executor.FAILURE_KIND_APPROVAL_CONFIG_INVALID is APPROVAL_CONFIG_INVALID


class TestTheFixture:
    """The opt-in fixture files stamps the readers accept."""

    def test_approved_binds_the_call_and_satisfies_the_record(self, stamps: ApprovalStamps) -> None:
        from osprey.audit.call import current_tool_use_id

        with stamps.approved("queue_add", tool_use_id="toolu_fixture"):
            assert current_tool_use_id() == "toolu_fixture"
            assert require_approval_record("queue_add", _ASKING) is None
        assert current_tool_use_id() is None

    def test_channel_write_reads_the_write_approval_stamp(self, stamps: ApprovalStamps) -> None:
        from osprey.mcp_server.control_system.tools import channel_write as cw

        operations = [{"channel": "SR:X", "value": 1.0}]
        with stamps.approved("channel_write", operations=operations, target="va", generation=3):
            assert require_approval_record("channel_write", _ASKING) is None
            assert cw._read_approval_stamp(operations, None) == (True, ("va", 3))
