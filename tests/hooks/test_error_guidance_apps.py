"""Application error types in the osprey_error_guidance PostToolUse hook.

An application's MCP server emits dotted ``<app>.<type>`` error types. The
rendered ``hook_config.json`` key ``app_error_types`` maps each one to an error
class and a guidance text; the hook classifies with it before the framework's
own ``ERROR_CLASS_MAP`` and falls back to today's behaviour whenever the map is
missing or malformed.
"""

import json

import pytest

from tests.hooks.conftest import error_response

APP_ERROR_TYPES = {
    "beam_viewer.unreachable": {
        "class": "Connection",
        "text": "The beam viewer panel is not running; ask the operator to open it.",
    },
    "beam_viewer.refused": {
        "class": "Safety",
        "text": "The operator declined the request in the panel.",
    },
}


# -- classification ---------------------------------------------------------


def test_app_error_type_takes_its_class_and_text_from_the_app_map(error_guidance):
    response = error_response("beam_viewer.unreachable", "panel down")
    assert error_guidance._detect_error(response, APP_ERROR_TYPES) == (
        "Connection",
        "panel down",
        "The beam viewer panel is not running; ask the operator to open it.",
    )


def test_app_error_type_carries_its_guidance_text(error_guidance):
    response = error_response("beam_viewer.refused", "declined")
    assert error_guidance._detect_error(response, APP_ERROR_TYPES) == (
        "Safety",
        "declined",
        "The operator declined the request in the panel.",
    )


def test_framework_error_type_keeps_its_class_and_has_no_text(error_guidance):
    response = error_response("connection_error", "refused")
    assert error_guidance._detect_error(response, APP_ERROR_TYPES) == (
        "Connection",
        "refused",
        None,
    )


def test_type_in_neither_map_falls_back_to_internal(error_guidance):
    response = error_response("beam_viewer.exploded", "boom")
    assert error_guidance._detect_error(response, APP_ERROR_TYPES) == ("Internal", "boom", None)


def test_missing_app_map_takes_the_framework_path(error_guidance):
    response = error_response("beam_viewer.unreachable", "panel down")
    assert error_guidance._detect_error(response, None) == ("Internal", "panel down", None)
    assert error_guidance._detect_error(response) == ("Internal", "panel down", None)


@pytest.mark.parametrize(
    "app_error_types",
    [
        ["beam_viewer.unreachable"],
        "beam_viewer.unreachable",
        {"beam_viewer.unreachable": "Connection"},
        {"beam_viewer.unreachable": ["Connection"]},
        {"beam_viewer.unreachable": {"class": 7, "text": "x"}},
        {"beam_viewer.unreachable": {"text": "x"}},
    ],
    ids=["list-map", "str-map", "str-entry", "list-entry", "int-class", "no-class"],
)
def test_malformed_app_map_takes_the_framework_path(error_guidance, app_error_types):
    response = error_response("beam_viewer.unreachable", "panel down")
    assert error_guidance._detect_error(response, app_error_types) == (
        "Internal",
        "panel down",
        None,
    )


def test_non_str_text_is_dropped_but_the_class_stands(error_guidance):
    response = error_response("beam_viewer.unreachable", "panel down")
    app_error_types = {"beam_viewer.unreachable": {"class": "Connection", "text": 42}}
    assert error_guidance._detect_error(response, app_error_types) == (
        "Connection",
        "panel down",
        None,
    )


# -- the guidance the hook injects -------------------------------------------


def _run_main(error_guidance, monkeypatch, tmp_path, capsys, config, tool_name, tool_response):
    path = tmp_path / "hook_config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("OSPREY_HOOK_CONFIG", str(path))
    monkeypatch.setattr(
        error_guidance,
        "get_hook_input",
        lambda: {"tool_name": tool_name, "tool_input": {}, "tool_response": tool_response},
    )
    monkeypatch.setattr(error_guidance, "log_hook", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        error_guidance.main()
    return json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]


def test_app_text_is_injected_on_its_own_line(error_guidance, monkeypatch, tmp_path, capsys):
    context = _run_main(
        error_guidance,
        monkeypatch,
        tmp_path,
        capsys,
        {"server_prefixes": ["mcp__beam_viewer__"], "app_error_types": APP_ERROR_TYPES},
        "mcp__beam_viewer__set_screen",
        error_response("beam_viewer.refused", "declined"),
    )
    lines = context.split("\n")
    assert lines[0] == "ERROR DETECTED [Safety]: declined"
    assert lines[1] == "Application guidance: The operator declined the request in the panel."
    assert context.index("Application guidance") < context.index("Follow the error-handling")


def test_framework_guidance_is_unchanged_by_an_app_map(
    error_guidance, monkeypatch, tmp_path, capsys
):
    context = _run_main(
        error_guidance,
        monkeypatch,
        tmp_path,
        capsys,
        {"server_prefixes": ["mcp__controls__"], "app_error_types": APP_ERROR_TYPES},
        "mcp__controls__channel_read",
        error_response("connection_error", "refused"),
    )
    assert context == (
        "ERROR DETECTED [Connection]: refused\n\n"
        "Follow the error-handling protocol (.claude/rules/error-handling.md):\n"
        "- Report the error clearly to the user with actionable next steps.\n"
        "- Do NOT debug infrastructure, write mock data, or work around the failure.\n"
        "- Do NOT retry unless the error explicitly indicates a transient condition."
    )


def test_hook_subprocess_classifies_an_app_error(hook_runner, make_config):
    result = hook_runner(
        "osprey_error_guidance.py",
        "mcp__beam_viewer__set_screen",
        {"screen": "in"},
        config_path=make_config({}),
        tool_response=error_response("beam_viewer.unreachable", "panel down"),
        hook_config={
            "server_prefixes": ["mcp__controls__", "mcp__beam_viewer__"],
            "approval_prefixes": [],
            "app_error_types": APP_ERROR_TYPES,
        },
    )
    context = result["hookSpecificOutput"]["additionalContext"]
    assert "ERROR DETECTED [Connection]: panel down" in context
    assert "Application guidance: The beam viewer panel is not running" in context
