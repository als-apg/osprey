"""`osprey set` and the write ceiling: it may narrow, never raise.

The profile states the write ceiling as ``control_system.writes`` (``never``,
``asked`` or ``allowed``). No CLI path turns writes on, so ``osprey set`` judges
each move against the ceiling the root renders today: ``=never`` is accepted from
any ceiling, ``=asked`` only from ``allowed``. It refuses everything else that
would touch the posture: ``=allowed`` and ``=asked`` from ``never`` or ``asked``
(edit the profile or tier file instead), any other value (naming the three
words), and every retired ``writes_enabled`` leaf (pointing at
``control_system.writes``). A refusal rejects the whole command line before
anything is written.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey.cli.set_cmd import set as set_command
from osprey.utils.config_writer import config_update_fields
from osprey_connectors.types import tier_writes

_ROOT_LINE = re.compile(r"^  control_system\.writes: .*\n", re.MULTILINE)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _invoke(runner: CliRunner, repo: Path, *args: str):
    return runner.invoke(set_command, ["--repo", str(repo), *args])


def _profile_bytes(repo: Path) -> bytes:
    return (repo / "profile.yml").read_bytes()


def _state_ceiling(repo: Path, word: str | None) -> None:
    """Rewrite the exemplar root's ceiling line to *word*; ``None`` removes it."""
    path = repo / "profile.yml"
    line = "" if word is None else f"  control_system.writes: {word}\n"
    text, count = _ROOT_LINE.subn(line, path.read_text(encoding="utf-8"))
    assert count == 1
    path.write_text(text, encoding="utf-8")


def _rendered_ceiling(repo: Path) -> str:
    """The ceiling of the config the build renders: each key applied in order."""
    profile, _ = resolve_build_profile(repo / "profile.yml", None)
    target = repo / "rendered-config.yml"
    target.write_text("{}\n", encoding="utf-8")
    config_update_fields(target, profile.config)
    return tier_writes(yaml.safe_load(target.read_text(encoding="utf-8")).get("control_system"))


def _silent_root(repo: Path, preset: str, *config_lines: str) -> Path:
    """A root over *preset*; the ceiling is the preset's unless *config_lines* state one."""
    repo.mkdir()
    (repo / "data").mkdir()
    body = f"extends: {preset}\nname: Demo\nproject_name: demo\ndata: data\n"
    if config_lines:
        body += "config:\n" + "".join(f"  {line}\n" for line in config_lines)
    (repo / "profile.yml").write_text(body, encoding="utf-8")
    return repo


def _preset_word(preset: str) -> str:
    """What the bundled preset file itself spells, read without the code under test."""
    text = (resources.files("osprey.profiles.presets") / f"{preset}.yml").read_text("utf-8")
    return yaml.safe_load(text)["config"]["control_system.writes"]


def test_never_is_accepted_and_written(runner, lifecycle_repo):
    result = _invoke(runner, lifecycle_repo, "config.control_system.writes=never")

    assert result.exit_code == 0, result.output
    assert "control_system.writes: never" in (lifecycle_repo / "profile.yml").read_text()


@pytest.mark.parametrize("ceiling", ["never", "asked", "allowed", None])
def test_never_is_accepted_from_every_ceiling(runner, lifecycle_repo, ceiling):
    _state_ceiling(lifecycle_repo, ceiling)

    result = _invoke(runner, lifecycle_repo, "config.control_system.writes=never")

    assert result.exit_code == 0, result.output
    assert _rendered_ceiling(lifecycle_repo) == "never"


def test_asked_is_accepted_from_allowed(runner, lifecycle_repo):
    _state_ceiling(lifecycle_repo, "allowed")

    result = _invoke(runner, lifecycle_repo, "config.control_system.writes=asked")

    assert result.exit_code == 0, result.output
    assert "control_system.writes: asked" in (lifecycle_repo / "profile.yml").read_text()
    assert _rendered_ceiling(lifecycle_repo) == "asked"


@pytest.mark.parametrize("ceiling", ["never", "asked", None])
def test_asked_is_refused_unless_the_ceiling_is_allowed(runner, lifecycle_repo, ceiling):
    _state_ceiling(lifecycle_repo, ceiling)
    before = _profile_bytes(lifecycle_repo)

    result = _invoke(runner, lifecycle_repo, "config.control_system.writes=asked")

    assert result.exit_code != 0
    assert "profile" in result.output and "tier file" in result.output, result.output
    assert _profile_bytes(lifecycle_repo) == before


@pytest.mark.parametrize("ceiling", ["never", "asked", "allowed", None])
def test_allowed_is_refused_and_points_at_the_profile(runner, lifecycle_repo, ceiling):
    _state_ceiling(lifecycle_repo, ceiling)
    before = _profile_bytes(lifecycle_repo)

    result = _invoke(runner, lifecycle_repo, "config.control_system.writes=allowed")

    assert result.exit_code != 0
    assert "profile" in result.output and "tier file" in result.output, result.output
    assert _profile_bytes(lifecycle_repo) == before


def test_a_narrowing_cannot_be_walked_back(runner, lifecycle_repo):
    _state_ceiling(lifecycle_repo, "allowed")
    asked = "config.control_system.writes=asked"
    never = "config.control_system.writes=never"

    assert _invoke(runner, lifecycle_repo, asked).exit_code == 0
    assert _invoke(runner, lifecycle_repo, asked).exit_code != 0
    assert _invoke(runner, lifecycle_repo, never).exit_code == 0
    assert _invoke(runner, lifecycle_repo, asked).exit_code != 0
    assert _rendered_ceiling(lifecycle_repo) == "never"


@pytest.mark.parametrize("preset", ["control-assistant", "hello-world"])
def test_a_silent_root_is_judged_by_the_preset_it_extends(runner, tmp_path, preset):
    repo = _silent_root(tmp_path / "repo", preset)
    before = _profile_bytes(repo)
    word = _preset_word(preset)
    assert _rendered_ceiling(repo) == word

    result = _invoke(runner, repo, "config.control_system.writes=asked")

    if word == "allowed":
        assert result.exit_code == 0, result.output
        assert _rendered_ceiling(repo) == "asked"
    else:
        assert result.exit_code != 0
        assert "tier file" in result.output, result.output
        assert _profile_bytes(repo) == before


@pytest.mark.parametrize("preset", ["control-assistant", "hello-world"])
def test_never_is_accepted_on_a_silent_root_over_any_preset(runner, tmp_path, preset):
    repo = _silent_root(tmp_path / "repo", preset)

    result = _invoke(runner, repo, "config.control_system.writes=never")

    assert result.exit_code == 0, result.output
    assert _rendered_ceiling(repo) == "never"


@pytest.mark.parametrize(
    ("line", "accepted"),
    [
        ("control_system.writes: allowed", True),
        ("control_system: {writes: allowed}", True),
        ("control_system.writes: never", False),
        ("control_system: {writes: never}", False),
    ],
    ids=["dotted-allowed", "nested-allowed", "dotted-never", "nested-never"],
)
@pytest.mark.parametrize("preset", ["control-assistant", "hello-world"])
def test_the_ceiling_is_read_through_the_render(runner, tmp_path, preset, line, accepted):
    """The root's own word beats the preset's, in either way a config: block spells it."""
    repo = _silent_root(tmp_path / "repo", preset, line)

    assert _rendered_ceiling(repo) == ("allowed" if accepted else "never")

    result = _invoke(runner, repo, "config.control_system.writes=asked")

    assert (result.exit_code == 0) is accepted, result.output
    assert _rendered_ceiling(repo) == ("asked" if accepted else "never")


_SHADOWING_ROOTS = {
    "block": ("control_system: {writes: allowed}",),
    "block-with-other-keys": ("control_system: {type: virtual_accelerator, writes: allowed}",),
    "dotted-ahead-of-block": ("control_system.writes: asked", "control_system: {writes: allowed}"),
}


@pytest.mark.parametrize("word", ["never", "asked"])
@pytest.mark.parametrize("spelling", _SHADOWING_ROOTS, ids=str)
@pytest.mark.parametrize("preset", ["control-assistant", "hello-world"])
def test_an_accepted_move_lands_past_the_roots_own_block(runner, tmp_path, preset, spelling, word):
    """The merge keeps a preset key's position, so a dotted write can sit behind a block."""
    repo = _silent_root(tmp_path / "repo", preset, *_SHADOWING_ROOTS[spelling])
    assert _rendered_ceiling(repo) == "allowed"

    result = _invoke(runner, repo, f"config.control_system.writes={word}")

    assert result.exit_code == 0, result.output
    assert _rendered_ceiling(repo) == word


def test_landing_inside_a_block_keeps_the_rest_of_it(runner, tmp_path):
    repo = _silent_root(
        tmp_path / "repo",
        "control-assistant",
        "# the machine this deployment talks to",
        "control_system: {type: virtual_accelerator, writes: allowed}",
    )

    assert _invoke(runner, repo, "config.control_system.writes=never").exit_code == 0

    text = (repo / "profile.yml").read_text()
    assert "# the machine this deployment talks to" in text
    block = yaml.safe_load(text)["config"]["control_system"]
    assert block == {"type": "virtual_accelerator", "writes": "never"}


@pytest.mark.parametrize("word", ["never", "asked"])
def test_an_accepted_move_lands_past_a_block_the_parent_states(runner, tmp_path, word):
    repo = tmp_path / "repo"
    _silent_root(repo, "control-assistant")
    (repo / "parent.yml").write_text(
        "extends: control-assistant\nconfig:\n  control_system: {writes: allowed}\n",
        encoding="utf-8",
    )
    path = repo / "profile.yml"
    path.write_text(
        path.read_text().replace("extends: control-assistant", "extends: parent.yml"),
        encoding="utf-8",
    )
    assert _rendered_ceiling(repo) == "allowed"

    result = _invoke(runner, repo, f"config.control_system.writes={word}")

    assert result.exit_code == 0, result.output
    assert _rendered_ceiling(repo) == word


def test_a_move_that_cannot_land_restores_the_file_and_refuses(runner, tmp_path, monkeypatch):
    repo = _silent_root(tmp_path / "repo", "hello-world", "control_system: {writes: allowed}")
    before = _profile_bytes(repo)

    def lands_nowhere(path, *_, **__):
        path.write_text(path.read_text() + "# touched\n", encoding="utf-8")

    monkeypatch.setattr("osprey.cli.build_profile_resolve._write_profile_values", lands_nowhere)

    result = _invoke(runner, repo, "config.control_system.writes=never")

    assert result.exit_code != 0
    assert "`allowed`" in result.output, result.output
    assert "profile or tier file" in result.output, result.output
    assert _profile_bytes(repo) == before


def test_a_root_that_does_not_resolve_refuses_asked_and_writes_nothing(runner, tmp_path):
    repo = _silent_root(tmp_path / "repo", "control-assistant")
    path = repo / "profile.yml"
    path.write_text(path.read_text().replace("data: data\n", ""), encoding="utf-8")
    before = path.read_bytes()

    result = _invoke(runner, repo, "config.control_system.writes=asked")

    assert result.exit_code != 0
    assert "data is required" in result.output, result.output
    assert path.read_bytes() == before


def test_narrowing_to_never_needs_no_resolvable_root(runner, tmp_path):
    repo = _silent_root(tmp_path / "repo", "control-assistant")
    path = repo / "profile.yml"
    path.write_text(path.read_text().replace("data: data\n", ""), encoding="utf-8")

    result = _invoke(runner, repo, "config.control_system.writes=never")

    assert result.exit_code == 0, result.output
    assert "not checked" in result.output, result.output


@pytest.mark.parametrize(
    "value",
    ["yes", "true", "on", "Allowed", "Asked", "NEVER", "", "[allowed]", "{x: 1}"],
    ids=["yes", "bool", "on", "capitalized", "capitalized-asked", "upper", "empty", "list", "map"],
)
def test_any_other_value_is_refused_naming_the_three_words(runner, lifecycle_repo, value):
    before = _profile_bytes(lifecycle_repo)

    result = _invoke(runner, lifecycle_repo, f"config.control_system.writes={value}")

    assert result.exit_code != 0
    for word in ("never", "asked", "allowed"):
        assert word in result.output, result.output
    assert _profile_bytes(lifecycle_repo) == before


@pytest.mark.parametrize(
    "pair",
    [
        "config.control_system.writes_enabled=false",
        "config.control_system.writes_enabled=true",
        "config.control_system.connector.virtual_accelerator.writes_enabled=true",
        "config.control_system.connector.epics.writes_enabled=false",
        "config.control_system.writes_enabled.x=true",
        "config.control_system.writes_enabled={x: true}",
        "config.control_system.connector.epics.writes_enabled={x: true}",
        "config={control_system: {connector: {va: {writes_enabled: {x: true}}}}}",
    ],
    ids=[
        "global-false",
        "global-true",
        "per-type-true",
        "per-type-false",
        "beneath-the-leaf",
        "mapping-at-the-leaf",
        "mapping-at-a-per-type-leaf",
        "inside-a-config-block",
    ],
)
def test_retired_key_is_refused_and_points_at_the_new_key(runner, lifecycle_repo, pair):
    before = _profile_bytes(lifecycle_repo)

    result = _invoke(runner, lifecycle_repo, pair)

    assert result.exit_code != 0
    assert "control_system.writes" in result.output, result.output
    assert "retired" in result.output, result.output
    assert "never | asked | allowed" in result.output, result.output
    assert _profile_bytes(lifecycle_repo) == before


@pytest.mark.parametrize(
    "pair",
    [
        "config.control_system={writes: allowed}",
        "config.control_system={writes_enabled: true}",
        "config.control_system.connector={epics: {writes_enabled: true}}",
        "config={control_system: {writes: allowed}}",
    ],
    ids=["block-new-allowed", "block-retired", "connector-block-retired", "config-block"],
)
def test_a_mapping_value_cannot_smuggle_the_posture(runner, lifecycle_repo, pair):
    before = _profile_bytes(lifecycle_repo)

    result = _invoke(runner, lifecycle_repo, pair)

    assert result.exit_code != 0
    assert "control_system.writes" in result.output, result.output
    assert _profile_bytes(lifecycle_repo) == before


def test_a_mapping_that_only_narrows_is_accepted(runner, lifecycle_repo):
    result = _invoke(runner, lifecycle_repo, "config.control_system={writes: never}")

    assert result.exit_code == 0, result.output
    assert _rendered_ceiling(lifecycle_repo) == "never"


@pytest.mark.parametrize(
    "pair",
    ["config.control_system={writes: asked}", "config={control_system: {writes: asked}}"],
    ids=["block", "config-block"],
)
def test_a_mapping_is_judged_against_the_ceiling_too(runner, lifecycle_repo, pair):
    _state_ceiling(lifecycle_repo, "never")
    before = _profile_bytes(lifecycle_repo)

    refused = _invoke(runner, lifecycle_repo, pair)

    assert refused.exit_code != 0
    assert _profile_bytes(lifecycle_repo) == before

    _state_ceiling(lifecycle_repo, "allowed")
    assert _invoke(runner, lifecycle_repo, pair).exit_code == 0
    assert _rendered_ceiling(lifecycle_repo) == "asked"


@pytest.mark.parametrize(
    "pair",
    [
        "config.control_system={writes: never}",
        "config={control_system: {writes: never}}",
    ],
    ids=["block", "config-block"],
)
@pytest.mark.parametrize("preset", ["control-assistant", "hello-world"])
def test_a_mapping_lands_over_the_roots_own_block(runner, tmp_path, preset, pair):
    repo = _silent_root(tmp_path / "repo", preset, "control_system: {writes: allowed}")

    assert _invoke(runner, repo, pair).exit_code == 0

    assert _rendered_ceiling(repo) == "never"


def _root_over_parent(repo: Path, parent_block: str, *config_lines: str) -> Path:
    """A root extending a local parent whose ``config:`` holds *parent_block*."""
    _silent_root(repo, "control-assistant", *config_lines)
    (repo / "parent.yml").write_text(
        f"extends: control-assistant\nconfig:\n  {parent_block}\n", encoding="utf-8"
    )
    path = repo / "profile.yml"
    path.write_text(
        path.read_text().replace("extends: control-assistant", "extends: parent.yml"),
        encoding="utf-8",
    )
    return repo


@pytest.mark.parametrize(
    "pair",
    ["config={}", "config={control_system: {type: virtual_accelerator}}"],
    ids=["empty", "block-without-the-key"],
)
def test_a_config_block_cannot_delete_a_roots_never(runner, tmp_path, pair):
    """The dotted line is what this command writes; the preset beneath it may allow writes."""
    repo = _silent_root(tmp_path / "repo", "control-assistant", "control_system.writes: never")
    before = _profile_bytes(repo)
    assert _rendered_ceiling(repo) == "never"

    result = _invoke(runner, repo, pair)

    assert result.exit_code != 0
    assert "tier file" in result.output, result.output
    assert "`never`" in result.output, result.output
    assert _profile_bytes(repo) == before


@pytest.mark.parametrize("word", ["never", "asked"])
@pytest.mark.parametrize(
    "pair",
    [
        "config.control_system={}",
        "config.control_system={type: virtual_accelerator}",
        "config={}",
    ],
    ids=["empty-block", "block-without-the-key", "empty-config"],
)
def test_a_block_cannot_uncover_the_ceiling_a_parent_states(runner, tmp_path, pair, word):
    repo = _root_over_parent(
        tmp_path / "repo",
        "control_system: {writes: allowed}",
        f"control_system: {{writes: {word}}}",
    )
    before = _profile_bytes(repo)
    assert _rendered_ceiling(repo) == word

    result = _invoke(runner, repo, pair)

    assert result.exit_code != 0
    assert "tier file" in result.output, result.output
    assert _profile_bytes(repo) == before
    assert _rendered_ceiling(repo) == word


@pytest.mark.parametrize(
    ("config_lines", "pair", "after"),
    [
        (
            ("control_system.writes: never",),
            "config.control_system={type: virtual_accelerator}",
            "never",
        ),
        (
            ("control_system.writes: allowed",),
            "config.control_system={type: virtual_accelerator}",
            "never",
        ),
        (
            ("control_system: {type: virtual_accelerator, writes: asked}",),
            "config={control_system: {type: virtual_accelerator, writes: never}}",
            "never",
        ),
    ],
    ids=["holds-never", "lowers-allowed", "states-never"],
)
def test_a_block_that_does_not_raise_the_ceiling_is_written(
    runner, tmp_path, config_lines, pair, after
):
    repo = _silent_root(tmp_path / "repo", "control-assistant", *config_lines)

    result = _invoke(runner, repo, pair)

    assert result.exit_code == 0, result.output
    assert _rendered_ceiling(repo) == after


def test_an_unrelated_key_does_not_touch_the_ceiling(runner, tmp_path):
    repo = _silent_root(tmp_path / "repo", "control-assistant", "control_system.writes: never")

    result = _invoke(runner, repo, "config.system.timezone=UTC")

    assert result.exit_code == 0, result.output
    assert _rendered_ceiling(repo) == "never"


@pytest.mark.parametrize("value", ["5", "null", "[x]"], ids=["number", "null", "list"])
def test_a_block_of_another_shape_cannot_raise_the_ceiling(runner, tmp_path, value):
    repo = _root_over_parent(
        tmp_path / "repo", "control_system: {writes: allowed}", "control_system: {writes: never}"
    )
    before = _profile_bytes(repo)

    result = _invoke(runner, repo, f"config.control_system={value}")

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit), result.output
    assert _profile_bytes(repo) == before
    assert _rendered_ceiling(repo) == "never"


@pytest.mark.parametrize("value", ["5", "true", "[x]", "", "null"])
def test_config_takes_a_mapping(runner, tmp_path, value):
    repo = _silent_root(tmp_path / "repo", "control-assistant", "control_system.writes: never")
    before = _profile_bytes(repo)

    result = _invoke(runner, repo, f"config={value}")

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit), result.output
    assert "takes a mapping" in result.output, result.output
    assert _profile_bytes(repo) == before


@pytest.mark.parametrize(
    "pair",
    [
        "config.={}",
        "config..control_system={}",
        "config.control_system.={}",
        "config. control_system={}",
        "config.control_system..writes=allowed",
    ],
)
def test_an_odd_spelling_is_judged_by_the_render(runner, tmp_path, pair):
    repo = _root_over_parent(
        tmp_path / "repo", "control_system: {writes: allowed}", "control_system: {writes: never}"
    )

    result = _invoke(runner, repo, pair)

    assert result.exit_code == 0, result.output
    assert _rendered_ceiling(repo) == "never"


def test_a_line_that_breaks_a_root_with_a_block_is_taken_back(runner, tmp_path):
    repo = _silent_root(tmp_path / "repo", "control-assistant", "control_system.writes: never")
    before = _profile_bytes(repo)

    result = _invoke(runner, repo, "config.control_system={writes: never, type: mock}")

    assert result.exit_code != 0
    assert _profile_bytes(repo) == before


def _without_data(repo: Path) -> Path:
    """Drop the root's ``data:`` line, so it no longer resolves until it is put back."""
    path = repo / "profile.yml"
    path.write_text(path.read_text().replace("data: data\n", ""), encoding="utf-8")
    return repo


@pytest.mark.parametrize(
    "pairs",
    [
        ("config.control_system={type: virtual_accelerator}",),
        ("config={}",),
        ("data=data", "config={}"),
        ("data=data", "config.control_system={}"),
        ("data=data",),
    ],
    ids=["block", "empty-config", "repair-and-empty-config", "repair-and-empty-block", "repair"],
)
@pytest.mark.parametrize("parent", [False, True], ids=["over-preset", "over-allowed-parent"])
def test_a_line_on_a_root_that_does_not_resolve_cannot_be_judged(runner, tmp_path, pairs, parent):
    """Without a render before the write there is nothing to compare; the line is refused."""
    if parent:
        repo = _root_over_parent(
            tmp_path / "repo",
            "control_system: {writes: allowed}",
            "control_system: {writes: never}",
        )
    else:
        repo = _silent_root(tmp_path / "repo", "control-assistant", "control_system.writes: never")
    _without_data(repo)
    before = _profile_bytes(repo)

    result = _invoke(runner, repo, *pairs)

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit), result.output
    assert "cannot be judged" in result.output, result.output
    assert "data is required" in result.output, result.output
    assert _profile_bytes(repo) == before
    (repo / "profile.yml").write_text(
        (repo / "profile.yml").read_text().replace("name: Demo\n", "name: Demo\ndata: data\n"),
        encoding="utf-8",
    )
    assert _rendered_ceiling(repo) == "never"


def test_a_line_that_cannot_reach_the_ceiling_is_written_to_a_root_that_does_not_resolve(
    runner, tmp_path
):
    repo = _without_data(
        _silent_root(tmp_path / "repo", "control-assistant", "control_system.writes: never")
    )

    result = _invoke(runner, repo, "config.system.timezone=UTC")

    assert result.exit_code == 0, result.output
    assert "system.timezone: UTC" in (repo / "profile.yml").read_text()


def test_never_beside_another_ceiling_input_is_refused_on_a_root_that_does_not_resolve(
    runner, tmp_path
):
    repo = _without_data(
        _silent_root(tmp_path / "repo", "control-assistant", "control_system.writes: never")
    )
    before = _profile_bytes(repo)

    result = _invoke(runner, repo, "config.control_system.writes=never", "config={}")

    assert result.exit_code != 0
    assert "cannot be judged" in result.output, result.output
    assert _profile_bytes(repo) == before


@pytest.mark.parametrize("pair", ["config={}", "config.control_system={type: mock}"])
def test_a_render_that_fails_in_any_way_leaves_the_ceiling_unjudged(
    runner, tmp_path, monkeypatch, pair
):
    """Not only the resolve's own errors: whatever the render raises, the line is refused."""
    repo = _silent_root(tmp_path / "repo", "control-assistant", "control_system.writes: never")
    before = _profile_bytes(repo)

    def fails(_repo_root):
        raise TypeError("render broke")

    monkeypatch.setattr("osprey.cli.set_cmd._rendered_ceiling", fails)

    result = _invoke(runner, repo, pair)

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit), result.output
    assert "render broke" in result.output, result.output
    assert _profile_bytes(repo) == before


def test_a_render_that_fails_after_the_write_takes_the_line_back(runner, tmp_path, monkeypatch):
    repo = _silent_root(tmp_path / "repo", "control-assistant", "control_system.writes: never")
    before = _profile_bytes(repo)
    calls = []

    def fails_after_the_write(repo_root):
        calls.append(repo_root)
        if len(calls) > 1:
            raise TypeError("render broke")
        return "never"

    monkeypatch.setattr("osprey.cli.set_cmd._rendered_ceiling", fails_after_the_write)

    result = _invoke(runner, repo, "config={}")

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit), result.output
    assert _profile_bytes(repo) == before


@pytest.mark.parametrize(
    "text",
    ["config:\n  control_system.writes: [never\n", "name: a\nname: b\n"],
    ids=["unclosed-flow", "duplicate-key"],
)
@pytest.mark.parametrize(
    "pair", ["config.control_system.writes=never", "config.system.timezone=UTC"]
)
def test_a_profile_that_is_not_yaml_is_refused_cleanly(runner, tmp_path, text, pair):
    repo = _silent_root(tmp_path / "repo", "control-assistant")
    (repo / "profile.yml").write_text(text, encoding="utf-8")
    before = _profile_bytes(repo)

    result = _invoke(runner, repo, pair)

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit), result.output
    assert "not valid YAML" in result.output, result.output
    assert _profile_bytes(repo) == before


@pytest.mark.parametrize("word", ["never", "asked"])
@pytest.mark.parametrize(
    "line",
    ["control_system.writes_enabled: true", "control_system: {writes_enabled: true}"],
    ids=["dotted", "block"],
)
def test_a_move_that_breaks_a_retired_root_is_taken_back(runner, tmp_path, line, word):
    """The retired leaf resolves alone; beside the new key the next resolve refuses it."""
    repo = _silent_root(tmp_path / "repo", "control-assistant", line)
    before = _profile_bytes(repo)

    result = _invoke(runner, repo, f"config.control_system.writes={word}")

    assert result.exit_code != 0
    assert "osprey init --revalidate" in result.output, result.output
    assert _profile_bytes(repo) == before


def test_one_refused_pair_rejects_the_whole_command_line(runner, lifecycle_repo):
    before = _profile_bytes(lifecycle_repo)

    result = _invoke(
        runner,
        lifecycle_repo,
        "config.system.timezone=UTC",
        "config.control_system.writes=allowed",
    )

    assert result.exit_code != 0
    assert _profile_bytes(lifecycle_repo) == before


def test_a_refused_narrowing_rejects_the_whole_command_line(runner, lifecycle_repo):
    _state_ceiling(lifecycle_repo, "never")
    before = _profile_bytes(lifecycle_repo)

    result = _invoke(
        runner,
        lifecycle_repo,
        "config.system.timezone=UTC",
        "config.control_system.writes=asked",
    )

    assert result.exit_code != 0
    assert _profile_bytes(lifecycle_repo) == before


def test_help_example_names_the_new_key(runner):
    result = runner.invoke(set_command, ["--help"])

    assert result.exit_code == 0
    assert "config.control_system.writes=never" in result.output
    assert "=asked" in result.output
    assert "writes_enabled" not in result.output


@pytest.mark.parametrize("config", ["5", "true", "abc", "[a]", ""])
def test_never_is_written_over_a_config_that_is_not_a_mapping(runner, tmp_path, config):
    repo = _silent_root(tmp_path / "repo", "control-assistant")
    path = repo / "profile.yml"
    path.write_text(path.read_text() + f"config: {config}\n", encoding="utf-8")

    result = _invoke(runner, repo, "config.control_system.writes=never")

    assert result.exit_code == 0, result.output
    assert _rendered_ceiling(repo) == "never"


@pytest.mark.parametrize("config", ["5", "true", "[a]", ""])
def test_a_block_over_a_config_that_is_not_a_mapping_cannot_be_judged(runner, tmp_path, config):
    repo = _silent_root(tmp_path / "repo", "control-assistant")
    path = repo / "profile.yml"
    path.write_text(path.read_text() + f"config: {config}\n", encoding="utf-8")
    before = _profile_bytes(repo)

    result = _invoke(runner, repo, "config.control_system={type: virtual_accelerator}")

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit), result.output
    assert "cannot be judged" in result.output, result.output
    assert _profile_bytes(repo) == before


@pytest.mark.parametrize("text", ["- a\n", "hello\n"], ids=["list", "scalar"])
def test_a_profile_that_is_not_a_mapping_is_refused_cleanly(runner, tmp_path, text):
    repo = _silent_root(tmp_path / "repo", "control-assistant")
    (repo / "profile.yml").write_text(text, encoding="utf-8")
    before = _profile_bytes(repo)

    result = _invoke(runner, repo, "config.control_system.writes=never")

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit), result.output
    assert "not a mapping" in result.output, result.output
    assert _profile_bytes(repo) == before


def test_never_is_written_into_an_empty_profile(runner, tmp_path):
    repo = _silent_root(tmp_path / "repo", "control-assistant")
    (repo / "profile.yml").write_text("", encoding="utf-8")

    result = _invoke(runner, repo, "config.control_system.writes=never")

    assert result.exit_code == 0, result.output
    assert "not checked" in result.output, result.output
    assert "control_system.writes: never" in (repo / "profile.yml").read_text()


@pytest.mark.parametrize("config", ["5", "true", "[a]", ""])
def test_replacing_a_config_that_is_not_a_mapping_cannot_be_judged(runner, tmp_path, config):
    """The line names no ceiling, but turning `config:` back into a mapping repairs the root."""
    repo = _silent_root(tmp_path / "repo", "control-assistant")
    path = repo / "profile.yml"
    path.write_text(path.read_text() + f"config: {config}\n", encoding="utf-8")
    before = _profile_bytes(repo)

    result = _invoke(runner, repo, "config.system.timezone=UTC")

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit), result.output
    assert "cannot be judged" in result.output, result.output
    assert _profile_bytes(repo) == before
