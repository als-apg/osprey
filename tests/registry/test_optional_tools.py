"""The ``python`` server's optional tools: what follows the pyAML view and what never does.

``model_response_matrix`` and ``pyaml_measure`` are registered by the server
only when the render carries a pyAML view, and ``model_response_matrix``'s
allow row follows the view with them. ``pyaml_measure``'s ask row and its
write and approval hooks render whatever the view says, so the write floor,
the deny list and the approval set never depend on it.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.registry.mcp import (
    _APPROVAL,
    _WRITES_CHECK,
    FRAMEWORK_SERVERS,
    framework_mixed_read_write_tools,
    is_mixed_matcher,
    resolve_servers,
)

MEASURE = "mcp__python__pyaml_measure"


def _python(view: bool, servers: dict[str, Any] | None = None, name: str = "python") -> dict:
    resolved = resolve_servers({"servers": servers or {}}, {"pyaml_view_present": view})
    (server,) = [entry for entry in resolved if entry["name"] == name]
    return server


def _hooks(server: dict, matcher: str) -> list[str]:
    (rule,) = [rule for rule in server["hooks_pre"] if rule["matcher"] == matcher]
    return [hook["command"] for hook in rule["hooks"]]


def test_both_tools_are_optional_on_the_view():
    assert FRAMEWORK_SERVERS["python"].optional_tools == {
        "pyaml_measure": "pyaml_view_present",
        "model_response_matrix": "pyaml_view_present",
    }


@pytest.mark.parametrize("view", [True, False])
def test_the_measurement_gate_renders_with_and_without_a_view(view: bool):
    server = _python(view)

    assert "pyaml_measure" in server["permissions_ask"]
    assert "pyaml_measure" not in server["permissions_allow"]
    assert _hooks(server, MEASURE) == [_WRITES_CHECK.command, _APPROVAL.command]


def test_the_model_tool_is_allowed_only_with_a_view():
    assert "model_response_matrix" in _python(True)["permissions_allow"]
    assert "model_response_matrix" not in _python(False)["permissions_allow"]
    assert "model_response_matrix" not in _python(True)["permissions_ask"]
    assert not [
        rule for rule in _python(True)["hooks_pre"] if "model_response_matrix" in rule["matcher"]
    ]


@pytest.mark.parametrize("view", [True, False])
def test_an_extends_clone_carries_the_optional_tools_under_its_own_name(view: bool):
    clone = _python(view, {"python2": {"extends": "python"}}, name="python2")

    assert "pyaml_measure" in clone["permissions_ask"]
    assert _hooks(clone, "mcp__python2__pyaml_measure") == [
        _WRITES_CHECK.command,
        _APPROVAL.command,
    ]
    assert ("model_response_matrix" in clone["permissions_allow"]) is view


def test_the_measurement_is_pure_write():
    assert not is_mixed_matcher("python", MEASURE)
    assert not is_mixed_matcher("python", "mcp__python2__pyaml_measure")
    assert is_mixed_matcher("python", "mcp__python2__execute")
    assert MEASURE not in framework_mixed_read_write_tools()


@pytest.mark.parametrize("view", [True, False])
def test_with_writes_off_the_measurement_is_denied(tmp_path, monkeypatch, view: bool):
    from osprey.cli.templates import claude_code
    from tests.cli.test_killswitch_bluesky_driftguard import (
        _build_project,
        _rendered_permissions,
    )

    derived = claude_code.config_derived_context

    def with_view(config: dict, project_dir: Any) -> dict:
        ctx = derived(config, project_dir)
        ctx["pyaml_view_present"] = view
        return ctx

    monkeypatch.setattr(claude_code, "config_derived_context", with_view)

    permissions = _rendered_permissions(_build_project(tmp_path, writes_enabled=False))

    assert MEASURE in permissions["deny"]
    assert MEASURE not in permissions["allow"]
    assert ("mcp__python__model_response_matrix" in permissions["allow"]) is view
