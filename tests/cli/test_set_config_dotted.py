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
render sees it, whether the operator stated it with ``--set`` at ``osprey
init`` or with ``osprey set`` afterwards.
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
    apply_cli_edits,
    cli_edit_layer,
    resolve_build_profile,
)
from osprey.cli.init_cmd import init
from osprey.cli.set_cmd import set as set_cmd
from osprey.errors import BuildProfileError


def _flat(text: str) -> str:
    """Collapse whitespace so assertions survive terminal line wrapping."""
    return " ".join(text.split())


def _set_args(pairs: tuple[str, ...]) -> list[str]:
    """The ``--set`` command-line form of *pairs*."""
    return [arg for pair in pairs for arg in ("--set", pair)]


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


def test_a_config_mapping_value_replaces_the_subtree_it_names() -> None:
    """``--set config.approval.tools={…}`` IS ``approval.tools``, as one key.

    A mapping given as a VALUE is written whole at the key it names, and every
    deeper key the document spells beneath it goes: two statements about one
    rendered path would otherwise both survive, and which of them reached
    ``config.yml`` would depend on key order nobody wrote down.
    """
    document = {
        "config": {
            "approval.tools.channel_read": "skip",
            "approval.tools.execute": "ask",
            "approval.enabled": True,
        }
    }

    edited = apply_cli_edits(document, ("config.approval.tools={channel_read: always}",))

    assert edited["config"] == {
        "approval.enabled": True,
        "approval.tools": {"channel_read": "always"},
    }


def test_a_config_leaf_pair_edits_one_key_and_keeps_its_siblings() -> None:
    """The other spelling: ``config.approval.tools.channel_read=`` is one leaf.

    Naming the leaf states the leaf, so the document's other ``approval.tools.*``
    entries are not beneath what was said and stay. The pair with the test above
    is the whole rule — the value's SHAPE decides how much the edit claims.
    """
    document = {
        "config": {
            "approval.tools.channel_read": "skip",
            "approval.tools.execute": "ask",
            "approval.enabled": True,
        }
    }

    edited = apply_cli_edits(document, ("config.approval.tools.channel_read=always",))

    assert edited["config"] == {
        "approval.tools.channel_read": "always",
        "approval.tools.execute": "ask",
        "approval.enabled": True,
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


def test_a_nested_mapping_value_replaces_the_leaves_beneath_it() -> None:
    """A pair whose value nests states the path it names, and outranks the same way.

    ``--set config.approval={...}`` names one path — ``approval`` — however
    deeply its value nests, so every one of the preset's ``approval.*`` leaves
    is beneath it and goes.
    """
    profile, _dir = resolve_build_profile(
        None, "hello-world", set_pairs=("config.approval={tools: {channel_read: always}}",)
    )

    assert profile.config["approval"] == {"tools": {"channel_read": "always"}}
    assert not [key for key in profile.config if key.split(".")[0] == "approval" and "." in key]


def test_a_top_level_mapping_value_states_the_whole_block() -> None:
    """``--set bluesky={port: 2}`` is the whole of ``bluesky``, at top level too.

    The rule is the same above ``config:`` as inside it: a mapping given as
    the VALUE states the block at the key it names, so whatever the document
    held there is gone, siblings included.
    """
    raw = apply_cli_edits(
        {"name": "x", "bluesky": {"port": 1, "host": "h"}}, ("bluesky={port: 2}",)
    )

    assert raw["bluesky"] == {"port": 2}


def test_a_top_level_dotted_key_edits_one_leaf() -> None:
    """``--set bluesky.port=2`` names a leaf: it is replaced, its siblings stay."""
    raw = apply_cli_edits({"name": "x", "bluesky": {"port": 1, "host": "h"}}, ("bluesky.port=2",))

    assert raw["bluesky"] == {"port": 2, "host": "h"}


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


def test_the_conflict_is_caught_when_the_edit_is_parsed() -> None:
    """Before the edit reaches any document — both spellings are the edit's own."""
    with pytest.raises(BuildProfileError, match="Conflicting connector overrides"):
        cli_edit_layer(("connector=epics", "config.control_system.type=doocs"))


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


def test_osprey_set_with_a_mapping_value_prunes_the_file_beneath_it(
    runner_repo: tuple[CliRunner, Path],
) -> None:
    """The file edit prunes what it outranks, the same as the in-memory one.

    ``osprey set`` writes the profile a facility hand-edits, so the document
    has to say what the resolver would: one ``approval.tools`` key holding the
    mapping, none of the preset's ``approval.tools.*`` leaves left beside it to
    say something else, and the siblings above it untouched.
    """
    runner, repo = runner_repo
    _init(runner, repo, "hello-world")

    written = runner.invoke(
        set_cmd, ["--repo", str(repo), 'config.approval.tools={"channel_read": "always"}']
    )
    assert written.exit_code == 0, written.output

    config = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))["config"]
    assert config["approval.tools"] == {"channel_read": "always"}
    assert not [key for key in config if key.startswith("approval.tools.")]
    assert config["approval.enabled"] is True
    assert _build(runner, repo)["approval"]["tools"] == {"channel_read": "always"}


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


def test_osprey_set_with_a_leaf_key_keeps_the_siblings_in_the_file(
    runner_repo: tuple[CliRunner, Path],
) -> None:
    """And the leaf spelling edits one key of the file, as it does in memory."""
    runner, repo = runner_repo
    _init(runner, repo, "hello-world")
    before = _profile_config(repo)
    sibling = "approval.tools.execute"
    assert sibling in before

    written = runner.invoke(
        set_cmd, ["--repo", str(repo), "config.approval.tools.channel_read=always"]
    )
    assert written.exit_code == 0, written.output

    config = _profile_config(repo)
    assert config["approval.tools.channel_read"] == "always"
    assert config[sibling] == before[sibling]
    assert "approval.tools" not in config


#: One edit of each shape a ``--set`` value can take: a scalar, a list, and a
#: null. A list and a null are the two an inheritance merge could not express —
#: a union cannot narrow, and a null read as "absent" would keep the old value —
#: so they are where the two entry points would drift apart if either still
#: layered instead of edited.
_EQUIVALENT_EDITS = (
    "config.approval.tools.channel_read=always",
    "config.claude_code.permissions.deny=[]",
    "config.hooks.debug=null",
)


def _profile_config(repo: Path) -> dict:
    """The ``config:`` block the profile at *repo* HOLDS, as written."""
    return yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))["config"]


def _resolved_config(repo: Path) -> dict:
    """The config block the profile at *repo* resolves to."""
    profile, _dir = resolve_build_profile(repo / "profile.yml", None)
    return profile.config


def test_init_set_and_osprey_set_resolve_to_the_same_profile(tmp_path: Path) -> None:
    """``osprey init --set k=v`` == ``osprey init`` then ``osprey set k=v``.

    Both are the same edit of the same profile — one made to the document
    before it is written, one made to the file afterwards — so they have to
    land the same value at the same key. Pinned over a list and a null as well
    as a scalar, because those are the shapes a layered merge would have got
    wrong while the scalar looked fine.
    """
    runner = CliRunner()
    # Same LEAF name under different parents: the deployment name is derived
    # from the directory, and it reaches the profile's own config.
    at_init = tmp_path / "at-init" / "deployment"
    afterwards = tmp_path / "afterwards" / "deployment"

    _init(runner, at_init, "control-assistant", *_set_args(_EQUIVALENT_EDITS))
    _init(runner, afterwards, "control-assistant")
    written = runner.invoke(set_cmd, ["--repo", str(afterwards), *_EQUIVALENT_EDITS])
    assert written.exit_code == 0, written.output

    assert _resolved_config(at_init) == _resolved_config(afterwards)
    for key, value in (
        ("approval.tools.channel_read", "always"),
        ("claude_code.permissions.deny", []),
        ("hooks.debug", None),
    ):
        assert _resolved_config(at_init)[key] == value


def test_init_set_and_osprey_set_write_the_same_profile_text(tmp_path: Path) -> None:
    """The two entry points also agree on the FILE, key for key.

    Resolution could paper over a difference in spelling — a nested block and a
    dotted key resolve alike — but the profile is what a facility reads and
    hand-edits, so the same edit has to leave the same document behind.
    """
    runner = CliRunner()
    # Same LEAF name under different parents: the deployment name is derived
    # from the directory, and it reaches the profile's own config.
    at_init = tmp_path / "at-init" / "deployment"
    afterwards = tmp_path / "afterwards" / "deployment"

    _init(runner, at_init, "control-assistant", *_set_args(_EQUIVALENT_EDITS))
    _init(runner, afterwards, "control-assistant")
    assert runner.invoke(set_cmd, ["--repo", str(afterwards), *_EQUIVALENT_EDITS]).exit_code == 0

    def _config(repo: Path) -> dict:
        return yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))["config"]

    assert _config(at_init) == _config(afterwards)


@pytest.fixture
def runner_repo(tmp_path: Path) -> tuple[CliRunner, Path]:
    return CliRunner(), tmp_path / "deployment"
