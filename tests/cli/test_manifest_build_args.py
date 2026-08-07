"""What the project manifest records about the build that produced it.

Under profile-always builds the profile is the source of truth, so the manifest
records the *profile* a project came from and stops trying to be a second copy
of the settings inside it: no ``explicit_overrides`` marker, and no ``--set``
replayed onto the rebuild command.
"""

from osprey.cli.templates.manifest import build_reproducible_command, extract_build_args


def _args(context, *, preset_name="control-assistant", profile_path=None):
    return extract_build_args(
        project_name="proj",
        preset_name=preset_name,
        profile_path=profile_path,
        data_bundle="control_assistant",
        context=context,
    )


def test_resolved_selection_values_are_recorded():
    args = _args(
        {
            "default_provider": "als-apg",
            "default_model": "anthropic/claude-opus",
            "channel_finder_mode": "hierarchical",
        }
    )
    assert args["provider"] == "als-apg"
    assert args["model"] == "anthropic/claude-opus"
    assert args["channel_finder_mode"] == "hierarchical"


def test_explicit_overrides_field_is_retired():
    """The marker is gone even when a caller still stamps the old context key.

    Nothing produces ``explicit_set_keys`` any more — an override is written
    into the profile now, not carried beside the project — so a stray one must
    not resurrect a field readers would act on.
    """
    args = _args(
        {
            "default_provider": "als-apg",
            "default_model": "anthropic/claude-opus",
            "explicit_set_keys": ["provider", "model"],
        }
    )
    assert "explicit_overrides" not in args


def test_profile_path_abs_recorded_for_a_preset_build():
    """A --preset build materializes a profile, so it names one like any other.

    The deploy side follows this path to write minted service secrets back to
    the profile that owns them; a preset-built project without it would keep
    them only in its own ``.env``.
    """
    args = _args(
        {"default_provider": "anthropic", "profile_path_abs": "/tmp/proj-profile/profile.yml"}
    )
    assert args["source"] == "preset"
    assert args["preset"] == "control-assistant"
    assert args["profile_path_abs"] == "/tmp/proj-profile/profile.yml"


def test_profile_path_abs_recorded_for_a_positional_build():
    args = _args(
        {"default_provider": "anthropic", "profile_path_abs": "/abs/prof/profile.yml"},
        preset_name=None,
        profile_path="prof/profile.yml",
    )
    assert args["source"] == "profile"
    assert args["profile_path"] == "prof/profile.yml"
    assert args["profile_path_abs"] == "/abs/prof/profile.yml"


def test_reproducible_command_replays_no_set_pairs():
    """The command names the source and stops there.

    Provider and model live in the profile the command re-reads, so replaying
    them would only pin a stale copy on the command line.
    """
    args = _args({"default_provider": "als-apg", "default_model": "anthropic/claude-opus"})
    assert build_reproducible_command(args) == "osprey build proj --preset control-assistant"


def test_reproducible_command_positional_form():
    args = _args(
        {"default_provider": "anthropic"}, preset_name=None, profile_path="prof/profile.yml"
    )
    assert build_reproducible_command(args) == "osprey build proj prof/profile.yml"
