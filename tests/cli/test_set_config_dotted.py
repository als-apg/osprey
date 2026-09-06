"""``--set config.*`` as a literal dotted key, and the prefix-aware merge.

A profile's ``config:`` block is a flat bag of LITERAL DOTTED KEYS: each one
addresses a single leaf of the rendered ``config.yml``, and ``config_update_fields``
applies them in iteration order, setting each addressed path verbatim. So a
``--set`` pair aiming at that block has to be spelled the same way. Nesting it
instead — ``{"approval": {"tools": {…}}}`` beside the preset's
``approval.tools.channel_read`` — puts two statements about one path in the
block at once, where the deep merge keeps both and which of them survives to
``config.yml`` depends on where the keys happen to sit.

The other half of the same rule is what a stated key OUTRANKS. A command line
is the last word, so a ``config`` key it states wins the whole subtree it names:
every inherited key beneath it (segment-prefix match) is dropped before the
render sees it, whether the operator stated it with ``--set``, in a ``-O``
overlay, or through ``osprey set``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile_load import CONNECTOR_CONFIG_KEY
from osprey.cli.build_profile_presets import list_presets
from osprey.cli.build_profile_resolve import (
    _drop_shadowed_config_keys,
    _parse_set_pairs,
    _stated_config_paths,
    merge_cli_overrides,
    resolve_build_profile,
)
from osprey.cli.init_cmd import init
from osprey.cli.set_cmd import set as set_cmd
from osprey.errors import BuildProfileError


def _flat(text: str) -> str:
    """Collapse whitespace so assertions survive terminal line wrapping."""
    return " ".join(text.split())


def _init(runner: CliRunner, repo: Path, preset: str, *extra: str) -> None:
    result = runner.invoke(init, [str(repo), "--preset", preset, "--no-git", *extra])
    assert result.exit_code == 0, result.output


def _build(runner: CliRunner, repo: Path) -> dict:
    """Render *repo* and return its ``build/config.yml``."""
    result = runner.invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert result.exit_code == 0, result.output
    return yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))


# ── the spelling ─────────────────────────────────────────────────────────────


def test_a_config_pair_parses_as_one_literal_dotted_key() -> None:
    """Everything after ``config.`` is one key, dots and all."""
    assert _parse_set_pairs(("config.approval.tools.channel_read=always",)) == {
        "config": {"approval.tools.channel_read": "always"}
    }


def test_a_config_pair_keeps_its_value_shape() -> None:
    """The RHS is still YAML: a mapping value stays a mapping under one key."""
    assert _parse_set_pairs(("config.approval.tools={channel_read: always}",)) == {
        "config": {"approval.tools": {"channel_read": "always"}}
    }


def test_a_bare_config_pair_replaces_the_whole_block() -> None:
    """``--set config=...`` addresses no path inside the block."""
    assert _parse_set_pairs(("config={}",)) == {"config": {}}


def test_a_non_config_pair_still_nests() -> None:
    """Only ``config:`` is a flat bag; the profile's own keys are a document."""
    assert _parse_set_pairs(("deploy.registry=ghcr.io",)) == {"deploy": {"registry": "ghcr.io"}}


def test_the_set_spelling_is_the_write_back_spelling(tmp_path: Path) -> None:
    """``osprey init --set`` and ``osprey set`` write the same key into a file.

    They are one rule with two entry points: the profile is the source of
    truth, so a pair written back has to land at the same key the merge would
    have produced, or the two commands would disagree about the same profile.
    """
    from osprey.cli.build_profile_resolve import write_back_cli_overrides

    profile = tmp_path / "profile.yml"
    profile.write_text("name: X\nconfig:\n  approval.enabled: true\n", encoding="utf-8")

    written = write_back_cli_overrides(profile, set_pairs=("config.approval.tools.execute=skip",))

    assert written == ["config.approval.tools.execute"]
    config = yaml.safe_load(profile.read_text(encoding="utf-8"))["config"]
    assert config["approval.tools.execute"] == "skip"
    assert config["approval.enabled"] is True


# ── what a stated key outranks ───────────────────────────────────────────────


def test_stated_paths_are_the_layers_own_config_keys() -> None:
    """A ``config`` block's keys ARE paths, in whichever spelling it uses."""
    assert _stated_config_paths({"config": {"a.b": 1, "c": {"d": 2}}}) == [("a", "b"), ("c",)]


def test_a_layer_that_states_no_config_states_no_path() -> None:
    assert _stated_config_paths({"provider": "anthropic"}) == []


def test_an_inherited_leaf_beneath_a_stated_key_is_dropped() -> None:
    raw = {"config": {"approval.tools.channel_read": "ask", "approval.enabled": True}}

    pruned = _drop_shadowed_config_keys(raw, [("approval", "tools")])

    assert pruned["config"] == {"approval.enabled": True}


def test_the_match_is_by_segment_not_by_string() -> None:
    """``approval.tools`` claims ``approval.tools.x`` but not ``approval.tools_extra``."""
    raw = {"config": {"approval.tools_extra": 1, "approval.tools.x": 2}}

    pruned = _drop_shadowed_config_keys(raw, [("approval", "tools")])

    assert pruned["config"] == {"approval.tools_extra": 1}


def test_a_layers_own_deeper_key_is_not_dropped() -> None:
    """One layer refining its own key is the emitter's collapse, not a shadow."""
    raw = {"config": {"a": {"b": 1}, "a.c": 2}}

    pruned = _drop_shadowed_config_keys(raw, [("a",), ("a", "c")])

    assert pruned["config"] == {"a": {"b": 1}, "a.c": 2}


def test_nothing_stated_prunes_nothing() -> None:
    """The rule is about what the command line said, not about the block."""
    raw = {"config": {"a": {"b": 1}, "a.c": 2}}

    assert _drop_shadowed_config_keys(raw, []) is raw


# ── resolution: preset + --set ───────────────────────────────────────────────


def test_a_set_pair_replaces_the_presets_key_of_the_same_name() -> None:
    """The straight case: one key, one value, no second spelling to lose to."""
    profile, _dir = resolve_build_profile(
        None, "hello-world", set_pairs=("config.approval.tools.channel_read=always",)
    )

    assert profile.config["approval.tools.channel_read"] == "always"
    assert "approval" not in profile.config


def test_a_stated_subtree_replaces_the_presets_leaves_beneath_it() -> None:
    """``--set config.approval.tools={…}`` is the whole of ``approval.tools``.

    That is what ``config_update_fields`` does with the key anyway — applied
    last, a mapping value sets the addressed path verbatim — so the resolved
    profile now says what the render was always going to do.
    """
    profile, _dir = resolve_build_profile(
        None, "hello-world", set_pairs=("config.approval.tools={channel_read: always}",)
    )

    assert profile.config["approval.tools"] == {"channel_read": "always"}
    assert not [key for key in profile.config if key.startswith("approval.tools.")]
    # Only what sits BENEATH the stated key goes; its siblings are untouched.
    assert profile.config["approval.enabled"] is True


def test_a_nested_override_file_mapping_replaces_the_leaves_beneath_it(tmp_path: Path) -> None:
    """An ``-O`` overlay states its paths by nesting, and outranks the same way."""
    override = tmp_path / "over.yml"
    override.write_text(
        "config:\n  approval:\n    tools:\n      channel_read: always\n", encoding="utf-8"
    )

    profile, _dir = resolve_build_profile(None, "hello-world", (override,))

    assert profile.config["approval"] == {"tools": {"channel_read": "always"}}
    assert not [key for key in profile.config if key.split(".")[0] == "approval" and "." in key]


def test_a_literal_key_inherited_from_a_parent_is_shadowed_too() -> None:
    """The prune runs post-``extends``: a parent's key is inherited content."""
    profile, _dir = resolve_build_profile(
        None,
        "control-assistant-readonly",
        set_pairs=("config.control_system={type: mock}",),
    )

    assert profile.config["control_system"] == {"type": "mock"}
    assert CONNECTOR_CONFIG_KEY not in profile.config


def test_an_unrelated_set_pair_leaves_the_preset_alone() -> None:
    """Pruning is scoped to what the command line named."""
    profile, _dir = resolve_build_profile(
        None, "hello-world", set_pairs=("config.facility.name=Ring",)
    )

    assert profile.config["facility.name"] == "Ring"
    assert profile.config["approval.tools.channel_read"] == "skip"


@pytest.mark.parametrize("preset", list_presets())
def test_every_shipped_preset_still_resolves(preset: str) -> None:
    """No preset shadows itself — the prune is about CLI layers only."""
    profile, _dir = resolve_build_profile(None, preset)

    assert profile.name


# ── connector shorthand vs the literal key it stands for ─────────────────────


def test_connector_with_a_literal_type_set_is_refused() -> None:
    """One command line stating the connector twice is a usage error.

    The shorthand is the short spelling of that one config key, so nothing in
    the profile says which of the two wins. This is the one conflict the
    layering step still refuses: unlike a prefix pair, neither spelling is
    beneath the other.
    """
    with pytest.raises(BuildProfileError) as excinfo:
        resolve_build_profile(
            None,
            "hello-world",
            set_pairs=("connector=epics", "config.control_system.type=doocs"),
        )

    message = _flat(str(excinfo.value))
    assert "Conflicting connector overrides" in message
    assert "connector='epics'" in message
    assert f"config.{CONNECTOR_CONFIG_KEY}='doocs'" in message


def test_connector_with_a_literal_key_in_an_override_file_is_refused(tmp_path: Path) -> None:
    """The ``-O`` file spelling is the same statement, so it conflicts the same way."""
    override = tmp_path / "over.yml"
    override.write_text(f"config:\n  {CONNECTOR_CONFIG_KEY}: doocs\n", encoding="utf-8")

    with pytest.raises(BuildProfileError, match="Conflicting connector overrides"):
        resolve_build_profile(None, "hello-world", (override,), ("connector=epics",))


def test_the_conflict_is_caught_at_merge_time() -> None:
    """Before extends resolution — the two spellings are both CLI-layer facts."""
    with pytest.raises(BuildProfileError, match="Conflicting connector overrides"):
        merge_cli_overrides({}, (), ("connector=epics", "config.control_system.type=doocs"))


def test_the_shorthand_alone_still_overrides_the_preset_literal_key() -> None:
    """The preset's own literal key is what the shorthand exists to override."""
    profile, _dir = resolve_build_profile(None, "control-assistant", set_pairs=("connector=mock",))

    assert profile.config[CONNECTOR_CONFIG_KEY] == "mock"


def test_a_literal_type_set_now_reaches_the_render() -> None:
    """No shorthand needed: the dotted key replaces the preset's own entry."""
    profile, _dir = resolve_build_profile(
        None, "control-assistant", set_pairs=("config.control_system.type=mock",)
    )

    assert profile.config[CONNECTOR_CONFIG_KEY] == "mock"


# ── the CLI surface: init, set, and the render they feed ─────────────────────


def test_init_set_reaches_the_rendered_config(runner_repo: tuple[CliRunner, Path]) -> None:
    """Requirement 4: `osprey init --set config.approval.tools.channel_read=always`."""
    runner, repo = runner_repo
    _init(runner, repo, "hello-world", "--set", "config.approval.tools.channel_read=always")

    assert _build(runner, repo)["approval"]["tools"]["channel_read"] == "always"


def test_osprey_set_reaches_the_rendered_config(runner_repo: tuple[CliRunner, Path]) -> None:
    """Requirement 4, the other entry point: the same key, written after init."""
    runner, repo = runner_repo
    _init(runner, repo, "hello-world")

    written = runner.invoke(
        set_cmd, ["--repo", str(repo), "config.approval.tools.channel_read=always"]
    )
    assert written.exit_code == 0, written.output

    assert _build(runner, repo)["approval"]["tools"]["channel_read"] == "always"


def test_osprey_set_moves_a_service_port_and_the_ledger_with_it(
    runner_repo: tuple[CliRunner, Path],
) -> None:
    """Requirement 4: a store's host port reaches the render AND the port ledger.

    The ledger is what a second deployment's port collision is reported
    against, so a port the operator moved has to be visible there under the
    key they edited — not only in the rendered config.
    """
    runner, repo = runner_repo
    _init(runner, repo, "control-assistant")

    written = runner.invoke(
        set_cmd, ["--repo", str(repo), "config.services.graphdb.port_host=9999"]
    )
    assert written.exit_code == 0, written.output

    profile, _dir = resolve_build_profile(repo / "profile.yml", None)
    assert profile._claimed_ports()["services.graphdb.port_host"] == 9999
    assert _build(runner, repo)["services"]["graphdb"]["port_host"] == 9999


@pytest.fixture
def runner_repo(tmp_path: Path) -> tuple[CliRunner, Path]:
    return CliRunner(), tmp_path / "deployment"
