"""Explicit ``--set`` override recording in the project manifest.

``extract_build_args`` has always recorded the *resolved* provider/model/
channel_finder_mode (preset defaults included). These tests lock in the
``explicit_overrides`` field: the subset of those keys the user actually
passed via ``--set``, which persona auto-render re-applies to derived
builds so one ``--set provider=`` retints the whole multi-user stack.
"""

from osprey.cli.build_profile import explicit_model_override_keys
from osprey.cli.templates.manifest import build_reproducible_command, extract_build_args


def _args(context):
    return extract_build_args(
        project_name="proj",
        preset_name="control-assistant",
        profile_path=None,
        data_bundle="control_assistant",
        context=context,
    )


def test_explicit_set_keys_recorded_as_explicit_overrides():
    args = _args(
        {
            "default_provider": "als-apg",
            "default_model": "anthropic/claude-opus",
            "channel_finder_mode": "hierarchical",
            "explicit_set_keys": ["provider", "model"],
        }
    )
    assert args["explicit_overrides"] == ["provider", "model"]
    # The resolved values themselves are recorded exactly as before.
    assert args["provider"] == "als-apg"
    assert args["channel_finder_mode"] == "hierarchical"


def test_no_explicit_set_keys_omits_field():
    args = _args({"default_provider": "anthropic", "default_model": "claude-haiku-4-5"})
    assert "explicit_overrides" not in args


def test_explicit_key_without_recorded_value_is_dropped():
    # A key claimed explicit but with no recorded value (e.g. --set model=
    # parsed to an empty string) must not be marked forwardable.
    args = _args({"default_provider": "anthropic", "explicit_set_keys": ["model"]})
    assert "explicit_overrides" not in args


def test_reproducible_command_ignores_explicit_marker():
    """The rebuild command renders --set for every recorded value, explicit or
    not — the marker only governs persona forwarding, never the command."""
    args = _args(
        {
            "default_provider": "als-apg",
            "default_model": "anthropic/claude-opus",
            "explicit_set_keys": ["provider"],
        }
    )
    cmd = build_reproducible_command(args)
    assert "--set provider=als-apg" in cmd
    assert "--set model=anthropic/claude-opus" in cmd
    assert "explicit" not in cmd


def test_explicit_model_override_keys_top_level_only():
    """Only bare top-level model-selection keys count; dotted paths into
    config (or unrelated keys) never do."""
    keys = explicit_model_override_keys(
        (
            "provider=als-apg",
            "config.claude_code.provider=cborg",
            "deploy_services=false",
            "channel_finder_mode=in_context",
        )
    )
    assert keys == ["provider", "channel_finder_mode"]


def test_explicit_model_override_keys_empty_for_no_pairs():
    assert explicit_model_override_keys(()) == []
