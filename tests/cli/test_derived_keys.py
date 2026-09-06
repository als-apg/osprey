"""The refusal of a ``config:`` spelling of a key the build renders.

``profile.yml`` is the whole declarative input, and a handful of keys are the
framework template's rather than the operator's — the project layout, the port
layout, and the sections derived from a profile field. This covers
:mod:`osprey.cli.derived_keys`: every listed key in both spellings, the two
prefix branches, the siblings that must stay authorable, the one derived key
that deliberately lives elsewhere (``web.panels.<id>.enabled``), and the
refusal reaching an operator through ``BuildProfile.validate``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli.build_profile import BuildProfile
from osprey.cli.derived_keys import DERIVED_KEYS, derived_key_errors, is_derived_key
from osprey.errors import BuildProfileError


@pytest.fixture(autouse=True)
def _facility_data_tree(tmp_path: Path) -> None:
    """The tree every profile's ``data:`` key names, beside the profile.

    ``data:`` is required of a repo profile and must resolve to a real
    directory, so without this each profile below would report one extra
    failure about a key none of these tests is about.
    """
    (tmp_path / "data").mkdir(exist_ok=True)


def _nested(dotted_key: str, value: object) -> dict:
    """The fully nested spelling of one dotted key."""
    node: object = value
    for segment in reversed(dotted_key.split(".")):
        node = {segment: node}
    assert isinstance(node, dict)
    return node


@pytest.mark.parametrize("dotted_key", sorted(DERIVED_KEYS))
def test_every_derived_key_is_refused_dotted(dotted_key: str) -> None:
    """The flat spelling — how a preset and `osprey set` write a key."""
    errors = derived_key_errors({dotted_key: "x"})

    assert len(errors) == 1
    assert dotted_key in errors[0]
    assert "profile.yml" in errors[0]


@pytest.mark.parametrize("dotted_key", sorted(DERIVED_KEYS))
def test_every_derived_key_is_refused_nested(dotted_key: str) -> None:
    """The fully nested spelling reaches the same rendered leaf."""
    errors = derived_key_errors(_nested(dotted_key, "x"))

    assert len(errors) == 1
    assert dotted_key in errors[0]


def test_a_mixed_spelling_is_refused_and_named_as_written() -> None:
    """A dotted prefix over a mapping — the split the flat walk would miss."""
    errors = derived_key_errors({"claude_code": {"default_model": "opus"}})

    assert len(errors) == 1
    assert "claude_code: default_model" in errors[0]
    assert "claude_code.default_model" in errors[0]


def test_the_message_names_the_field_that_supplies_the_key() -> None:
    """The refusal is actionable: it says what to set instead."""
    provider = derived_key_errors({"claude_code.provider": "anthropic"})[0]
    mode = derived_key_errors({"channel_finder.pipeline_mode": "in_context"})[0]
    presets = derived_key_errors({"web.presets": {"ops": ["chat"]}})[0]

    assert "`provider:`" in provider
    assert "`channel_finder_mode:`" in mode
    assert "`panel_presets:`" in presets


@pytest.mark.parametrize(
    "dotted_key",
    [
        "file_paths.docs_dir",
        "channel_finder.pipelines.in_context.enable_summaries",
        "web.presets.ops",
        "execution.environment.packages.0",
    ],
)
def test_a_key_beneath_a_derived_branch_is_refused(dotted_key: str) -> None:
    """A member of DERIVED_KEYS claims every key under it, not only itself."""
    assert is_derived_key(dotted_key)
    errors = derived_key_errors({dotted_key: "x"})

    assert len(errors) == 1
    assert dotted_key in errors[0]


def test_a_nested_derived_branch_is_one_refusal_naming_the_branch() -> None:
    """Four leaves under `file_paths:` are one line to delete, so one error."""
    errors = derived_key_errors(
        {"file_paths": {"docs_dir": "a", "data_dir": "b", "cache_dir": "c", "logs_dir": "d"}}
    )

    assert len(errors) == 1
    assert "file_paths" in errors[0]


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"web.theme": "dark"}, id="web_theme"),
        pytest.param({"web": {"theme": "dark"}}, id="web_theme_nested"),
        pytest.param({"claude_code.servers.ariel.enabled": True}, id="ariel_server"),
        pytest.param({"claude_code": {"servers": {"ariel": {"enabled": True}}}}, id="ariel_nested"),
        pytest.param({"claude_code.permissions.deny": ["Bash"]}, id="permissions"),
        pytest.param({"channel_finder.tier": 3}, id="channel_finder_sibling"),
        pytest.param({"artifact_server.host": "0.0.0.0"}, id="artifact_server_host"),
        pytest.param({"agent_data.retention_days": 30}, id="agent_data_sibling"),
        pytest.param({"execution.timeout": 60}, id="execution_sibling"),
        pytest.param({"logbook.composition.style": "terse"}, id="logbook_sibling"),
        pytest.param({"project_name_suffix": "x"}, id="not_a_prefix_without_a_dot"),
    ],
)
def test_an_unrelated_sibling_is_allowed(config: dict) -> None:
    """Only the listed branches are the build's; everything beside them is the
    operator's to state, and a prefix match needs a segment boundary."""
    assert derived_key_errors(config) == []


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"web.panels.chat.enabled": True}, id="dotted"),
        pytest.param({"web": {"panels": {"chat": {"enabled": True}}}}, id="nested"),
        pytest.param({"web.panels": {"chat": {"enabled": False}}}, id="mixed"),
    ],
)
def test_panel_enabled_is_not_reported_here(config: dict) -> None:
    """`web.panels.<id>.enabled` is derived from `web_panels:` too, but a
    spelling that AGREES with the selection is accepted, so it stays with
    panel_selection_errors."""
    assert derived_key_errors(config) == []


def test_a_non_mapping_config_yields_nothing() -> None:
    """A malformed `config:` is refused by name elsewhere in validate."""
    assert derived_key_errors(["claude_code.provider"]) == []
    assert derived_key_errors(None) == []


def test_validate_refuses_a_derived_key(tmp_path: Path) -> None:
    """End to end: the operator meets the refusal through the build."""
    profile = BuildProfile(name="x", data="data", config={"claude_code.provider": "anthropic"})

    with pytest.raises(BuildProfileError) as excinfo:
        profile.validate(tmp_path)

    message = str(excinfo.value)
    assert "claude_code.provider" in message
    assert "`provider:`" in message


def test_validate_accepts_a_profile_that_states_nothing_derived(tmp_path: Path) -> None:
    """The guard adds no refusal to an ordinary profile."""
    BuildProfile(name="x", data="data", config={"web.theme": "dark"}).validate(tmp_path)
