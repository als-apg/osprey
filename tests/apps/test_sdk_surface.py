"""``osprey.apps.sdk``'s public surface: exactly the names an application imports."""

from __future__ import annotations

import osprey.apps.sdk as sdk

#: The application SDK's export list. A change here is a change of the
#: application contract.
SDK_NAMES = {
    "serve_tools",
    "make_error",
    "web_terminal_url",
    "agent_panel_headers",
    "check_approved_call",
    "ApprovalRefused",
    "BoundState",
    "OWNER_STATE_KEY",
    "raw_put_block",
    "configure_app",
    "port_layout",
    "lock",
    "ActionContext",
    "command_write",
    "write_verdict",
    "WriteVerdict",
    "refusal_phrase",
    "bind_owner",
    "owner_from_header",
    "owner_target",
    "load_by_path",
    "verify_launch_token",
    "LAUNCH_TOKEN_HEADER",
    "facility_file",
    "read_facts",
    "queue_is_active",
    "non_negative_seconds",
    "positive_seconds",
    "current_tool_use_id",
    "OspreyRunBusy",
    "OspreyWriteRefused",
    "GuardedRunDirError",
    "guarded_run_dir",
    "ENV_CONTROL_IDENTITY",
    "armed_contract",
    "record_and_mark",
    "DECISION_ALLOWED",
    "DECISION_REFUSED",
    "POSTURE_SOURCE_APP",
    "startup_timer",
    "tool_keys",
}


def test_all_is_exactly_the_sdk_list():
    assert len(sdk.__all__) == len(set(sdk.__all__))
    assert set(sdk.__all__) == SDK_NAMES


def test_every_name_is_a_bare_identifier_the_module_carries():
    for name in sdk.__all__:
        assert name.isidentifier(), name
        assert hasattr(sdk, name), name
