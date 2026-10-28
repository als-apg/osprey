"""The framework modules the application SDK wraps, at the paths it imports.

The SDK re-exports a small set of framework internals: the approval record
check, the module loader and the launch-token verifier, plus the run-lock
action context. These tests pin that each one exists at its exact import
path with the signature the SDK calls, so a move or a rename fails here
rather than inside an application.

Setting ``OSPREY_ATTEST_BASE=1`` additionally runs the absence checks: they
attest that the framework base carries none of the mechanisms the
application SDK adds on top of it. They are skipped otherwise, since those
mechanisms land later and would turn them red.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import inspect
import os
from pathlib import Path

import pytest

attest_base = pytest.mark.skipif(
    os.environ.get("OSPREY_ATTEST_BASE") != "1",
    reason=(
        "set OSPREY_ATTEST_BASE=1 when attesting the framework base the "
        "application SDK wraps; absence checks are skipped otherwise"
    ),
)

#: A tool name no per-tool rule names, so the default rule answers for it.
SYNTHETIC_TOOL = "synthetic_sdk_source_tool"


def _parameters(func) -> list[str]:
    return list(inspect.signature(func).parameters)


# --- Presence and shape --------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "osprey.utils.launch_token",
        "osprey.utils.module_loading",
        "osprey.mcp_server.approval_record",
        "osprey.runtime._run_state",
    ],
)
def test_wrapped_module_imports_at_its_path(module):
    assert importlib.import_module(module).__name__ == module


def test_approval_record_names_no_application_field():
    from osprey.mcp_server import approval_record

    for name in approval_record.APPROVED_FIELDS:
        assert "trajectory" not in name
    for tool, fields in approval_record.TOOL_REQUIRED_FIELDS.items():
        assert "trajectory" not in tool
        for name in fields:
            assert "trajectory" not in name


def test_unnamed_tool_requires_the_approved_target_under_a_guarded_run():
    from osprey.mcp_server.approval_record import TOOL_REQUIRED_FIELDS, required_fields

    assert SYNTHETIC_TOOL not in TOOL_REQUIRED_FIELDS
    assert required_fields(SYNTHETIC_TOOL, guarded_run=True) == ("approved_target",)
    assert required_fields(SYNTHETIC_TOOL) == ()


def test_record_check_signature():
    from osprey.mcp_server.approval_record import require_approval_record

    assert _parameters(require_approval_record) == ["tool", "config", "approved", "guarded_run"]


def test_module_loader_signature():
    from osprey.utils.module_loading import load_by_path

    assert _parameters(load_by_path) == ["path", "prefix"]


def test_launch_token_verifier_and_header():
    from osprey.utils.launch_token import LAUNCH_TOKEN_HEADER, verify_launch_token

    assert _parameters(verify_launch_token) == ["presented", "env_name"]
    assert LAUNCH_TOKEN_HEADER == "X-Launch-Token"


def test_action_context_fields():
    from osprey.runtime._run_state import ActionContext

    names = [field.name for field in dataclasses.fields(ActionContext)]
    assert names[:3] == ["target", "owner", "approved_target"]


def test_panel_validator_skips_vendor_builds(tmp_path: Path):
    from osprey.interfaces.design_system.panels.validator import _hex_scan_files

    (tmp_path / "panel.js").write_text("const c = '#ffffff';\n")
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "lib.js").write_text("const c = '#000000';\n")

    assert _hex_scan_files(tmp_path, None) == [tmp_path / "panel.js"]


# --- Absence (attest only) -----------------------------------------------


@attest_base
def test_required_fields_is_a_flat_all_of_tuple():
    from osprey.mcp_server.approval_record import required_fields

    assert required_fields(SYNTHETIC_TOOL, guarded_run=True) == ("approved_target",)


@attest_base
def test_action_context_has_no_limits_switch():
    from osprey.runtime._run_state import ActionContext

    names = [field.name for field in dataclasses.fields(ActionContext)]
    assert names == ["target", "owner", "approved_target"]


@attest_base
def test_guarded_run_has_no_command_write():
    from osprey.runtime import guarded_run

    assert not hasattr(guarded_run, "command_write")


@attest_base
def test_posture_store_has_no_owner_context():
    from osprey_connectors import posture_store

    assert not hasattr(posture_store, "owner_context")


@attest_base
def test_application_sdk_package_is_absent():
    assert importlib.util.find_spec("osprey.apps") is None


@attest_base
def test_vendor_fetch_has_no_package_targets():
    from osprey.cli.vendor_cmd import fetch
    from osprey.interfaces import vendor

    assert "targets" not in inspect.signature(vendor.fetch_all).parameters
    assert not hasattr(vendor, "_target_spec")
    assert all("--target" not in param.opts for param in fetch.params)
