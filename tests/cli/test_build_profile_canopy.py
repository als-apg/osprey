"""`osprey build` and `osprey validate` judge `web.panels.canopy.home_screen` against the library.

A build that names no CANOPY payload leaves the panel out of the render.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile_canopy import (
    HOME_SCREEN_KEY,
    canopy_home_screen_errors,
    without_unstaged_panel,
)
from osprey.cli.init_cmd import init
from osprey.cli.validate_cmd import validate
from osprey.integrations.canopy.library import find_home_screen

PANELS = ["canopy"]

# --------------------------------------------------------------------------- #
# Unit half
# --------------------------------------------------------------------------- #


def _screen(library: Path, screen_id: str, kind: str = "screen") -> Path:
    path = library / "screens" / f"{screen_id}.screen.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": kind, "widgets": []}))
    return path


@pytest.fixture
def library(tmp_path: Path) -> Path:
    root = tmp_path / "data" / "canopy" / "library"
    _screen(root, "facility/overview")
    _screen(root, "sr/orbit-feedback")
    return root


def _errors(value, library: Path, panels=PANELS) -> list[str]:
    return canopy_home_screen_errors({HOME_SCREEN_KEY: value}, panels, library)


def test_a_screen_the_library_holds_draws_no_error(library: Path):
    assert _errors("facility/overview", library) == []
    assert find_home_screen(library, "facility/overview").found


def test_the_helper_names_the_store_path_and_the_file(library: Path):
    home = find_home_screen(library, "sr/orbit-feedback")
    assert home.store_path == "screens/sr/orbit-feedback.screen.json"
    assert home.path == library / "screens" / "sr" / "orbit-feedback.screen.json"
    assert home.screens == ("facility/overview", "sr/orbit-feedback")


def test_a_missing_screen_names_key_id_path_fix_and_holdings(library: Path):
    (error,) = _errors("facility/nope", library)
    assert "`web.panels.canopy.home_screen`" in error
    assert "'facility/nope'" in error
    assert str(library / "screens" / "facility" / "nope.screen.json") in error
    assert "remove the key" in error
    assert "'facility/overview', 'sr/orbit-feedback'" in error


def test_the_library_prefix_gets_a_fix_naming_the_plain_id(library: Path):
    (error,) = _errors("library/facility/overview", library)
    assert "`library/` prefix" in error
    assert "Write it as 'facility/overview'." in error


def test_the_screen_suffix_gets_a_fix_naming_the_plain_id(library: Path):
    (error,) = _errors("facility/overview.screen.json", library)
    assert "`.screen.json` suffix" in error
    assert "Write it as 'facility/overview'." in error


def test_prefix_and_suffix_together_get_one_plain_id(library: Path):
    (error,) = _errors("library/facility/overview.screen.json", library)
    assert "prefix" in error and "suffix" in error
    assert "Write it as 'facility/overview'." in error


@pytest.mark.parametrize(
    "value",
    ["/facility/overview", "facility/../facility/overview", "../library/facility/overview"],
    ids=["leading-slash", "dotdot-inside", "dotdot-leading"],
)
def test_an_id_that_is_not_a_relative_path_is_refused(library: Path, value: str):
    (error,) = _errors(value, library)
    assert repr(value) in error
    assert "relative path under screens/" in error


def test_a_dotfile_screen_is_refused_as_the_listing_refuses_it(library: Path):
    _screen(library, ".hidden/overview")
    (error,) = _errors(".hidden/overview", library)
    assert "listing refuses it" in error
    assert "dotfile" in error


def test_a_symlinked_screen_is_refused_as_the_listing_refuses_it(library: Path):
    link = library / "screens" / "facility" / "linked.screen.json"
    link.symlink_to(library / "screens" / "facility" / "overview.screen.json")
    (error,) = _errors("facility/linked", library)
    assert "listing refuses it" in error
    assert "symlink" in error
    assert not find_home_screen(library, "facility/linked").found


def test_no_library_is_refused(tmp_path: Path):
    (error,) = _errors("facility/overview", tmp_path / "data" / "canopy" / "library")
    assert "no screen library" in error
    assert "The library holds no screens." in error


def test_a_file_whose_kind_is_not_screen_is_refused(library: Path):
    _screen(library, "facility/members", kind="collection")
    (error,) = _errors("facility/members", library)
    assert "not a screen" in error
    assert "'collection'" in error


def test_a_file_that_is_not_json_is_refused(library: Path):
    (library / "screens" / "facility" / "broken.screen.json").write_text("{not json")
    (error,) = _errors("facility/broken", library)
    assert "not a screen" in error


def test_the_holdings_list_stops_at_ten(library: Path):
    for index in range(10):
        _screen(library, f"extra/screen-{index:02d}")
    (error,) = _errors("facility/nope", library)
    # In list order the ten extras come first; the library's own two are counted.
    assert "'extra/screen-09'" in error
    assert "'sr/orbit-feedback'" not in error
    assert "and 2 more" in error


def test_canopy_not_selected_leaves_the_key_inert(library: Path):
    assert _errors("facility/nope", library, panels=["ariel"]) == []
    assert _errors("facility/nope", library, panels=None) == []


@pytest.mark.parametrize("value", ["", None], ids=["empty", "null"])
def test_an_unset_value_means_the_first_screen(library: Path, value):
    assert _errors(value, library) == []


def test_an_environment_reference_is_not_judged(library: Path):
    assert _errors("${HOME_SCREEN}", library) == []


def test_a_non_string_value_is_refused(library: Path):
    (error,) = _errors(5, library)
    assert "not a screen id" in error


def test_a_nested_spelling_is_named_as_written(library: Path):
    config = {"web": {"panels": {"canopy": {"home_screen": "facility/nope"}}}}
    (error,) = canopy_home_screen_errors(config, PANELS, library)
    assert "`web: panels: canopy: home_screen`" in error


# --------------------------------------------------------------------------- #
# No payload named: the panel leaves the render
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class _Profile:
    """The three fields the payload rule reads, on a dataclass as the real profile is."""

    web_panels: list[str]
    default_panel: str | None
    skills: list[str]


def _canopy_profile() -> _Profile:
    profile = _Profile(
        web_panels=["artifacts", "canopy"],
        default_panel="canopy",
        skills=["canopy-operator", "canopy-screens", "canopy-panel-builder", "logbook"],
    )
    profile.inherited_preset = "control-assistant"  # type: ignore[attr-defined]
    return profile


@pytest.fixture
def no_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OSPREY_CANOPY_PAYLOAD", raising=False)
    monkeypatch.delenv("OSPREY_CANOPY_PAYLOAD_CHECK", raising=False)


@pytest.mark.usefixtures("no_payload")
def test_naming_no_payload_takes_the_panel_its_default_and_its_skills_out(tmp_path: Path):
    profile = _canopy_profile()

    kept = without_unstaged_panel(profile, tmp_path)

    assert kept.web_panels == ["artifacts"]
    assert kept.default_panel is None
    assert kept.skills == ["logbook"]
    assert kept.inherited_preset == "control-assistant"
    assert profile.web_panels == ["artifacts", "canopy"], "the input is not edited"


def test_a_named_payload_keeps_the_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OSPREY_CANOPY_PAYLOAD", str(tmp_path / "payload"))
    profile = _canopy_profile()

    assert without_unstaged_panel(profile, tmp_path) is profile


@pytest.mark.usefixtures("no_payload")
def test_a_payload_named_in_the_repo_env_keeps_the_panel(tmp_path: Path):
    (tmp_path / ".env").write_text(f"OSPREY_CANOPY_PAYLOAD={tmp_path / 'payload'}\n")
    profile = _canopy_profile()

    assert without_unstaged_panel(profile, tmp_path) is profile


@pytest.mark.usefixtures("no_payload")
def test_skipped_staging_keeps_the_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OSPREY_CANOPY_PAYLOAD_CHECK", "0")
    profile = _canopy_profile()

    assert without_unstaged_panel(profile, tmp_path) is profile


@pytest.mark.usefixtures("no_payload")
def test_a_profile_without_the_panel_is_returned_as_is(tmp_path: Path):
    profile = _Profile(web_panels=["artifacts"], default_panel="artifacts", skills=["logbook"])

    assert without_unstaged_panel(profile, tmp_path) is profile


# --------------------------------------------------------------------------- #
# CLI half — the shipped control-assistant preset and its tiers
# --------------------------------------------------------------------------- #

ANCHOR = "  web.panels.canopy.vole_bytes_max: 268435456\n"
TIER_ANCHOR = "\nconfig:\n"


@pytest.fixture(scope="module")
def materialized(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One real control-assistant materialization to edit copies of."""
    root = tmp_path_factory.mktemp("home-screen") / "demo"
    result = CliRunner().invoke(init, [str(root), "--preset", "control-assistant", "--no-git"])
    assert result.exit_code == 0, result.output
    return root


@pytest.fixture
def repo(materialized: Path, tmp_path: Path) -> Path:
    target = tmp_path / "demo"
    shutil.copytree(materialized, target, symlinks=True)
    return target


def _insert(path: Path, anchor: str, line: str) -> None:
    body = path.read_text()
    assert body.count(anchor) == 1
    path.write_text(body.replace(anchor, anchor + line + "\n"))


def _set_home(repo: Path, value: str) -> None:
    _insert(repo / "profile.yml", ANCHOR, f"  web.panels.canopy.home_screen: {value}")


def _set_tier_home(repo: Path, tier: str, value: str) -> None:
    _insert(
        repo / "tiers" / f"{tier}.yml",
        TIER_ANCHOR,
        f"  web.panels.canopy.home_screen: {value}",
    )


def _build(repo: Path):
    return CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])


def _validate(*args: str):
    return CliRunner().invoke(validate, [*args, "--drift", "warn"])


def test_validate_passes_a_home_screen_the_library_holds(repo: Path):
    _set_home(repo, "facility/overview")
    result = _validate("--repo", str(repo))
    assert result.exit_code == 0, result.output
    assert "Profile is valid" in result.output


def test_validate_refuses_a_tier_overriding_to_a_missing_id(repo: Path):
    _set_tier_home(repo, "readonly", "ro/missing")
    result = _validate("--repo", str(repo))
    assert result.exit_code == 2, result.output
    assert "tiers/readonly.yml:" in result.output
    assert "'ro/missing'" in result.output


def test_validate_of_the_tier_itself_refuses_it(repo: Path):
    _set_tier_home(repo, "readonly", "ro/missing")
    result = _validate(str(repo / "tiers" / "readonly.yml"))
    assert result.exit_code == 2, result.output
    assert "'ro/missing'" in result.output
    assert "ro/missing.screen.json" in result.output


def test_build_refuses_a_tier_overriding_to_a_missing_id(repo: Path):
    _set_tier_home(repo, "readwrite", "rw/missing")
    result = _build(repo)
    assert result.exit_code == 2, result.output
    assert "tiers/readwrite.yml:" in result.output
    assert "'rw/missing'" in result.output
    assert not (repo / "build" / "config.yml").exists()


def _rendered_configs(repo: Path) -> list[Path]:
    return [repo / "build" / "config.yml", *sorted((repo / "build").glob("demo-*/config.yml"))]


def _canopy_enabled(config: Path) -> bool:
    import yaml

    panels = yaml.safe_load(config.read_text())["web"]["panels"]
    return bool(panels.get("canopy", {}).get("enabled"))


def test_a_build_naming_no_payload_leaves_the_panel_out_and_says_so_once(
    repo: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("OSPREY_CANOPY_PAYLOAD_CHECK", raising=False)
    monkeypatch.delenv("OSPREY_CANOPY_PAYLOAD", raising=False)
    result = _build(repo)
    assert result.exit_code == 0, result.output
    assert result.output.count("CANOPY panel off") == 1
    configs = _rendered_configs(repo)
    assert len(configs) > 1
    assert [c for c in configs if _canopy_enabled(c)] == []
    assert list((repo / "build").glob("**/.claude/skills/canopy-*")) == []


def test_a_build_naming_a_payload_keeps_the_panel(repo: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OSPREY_CANOPY_PAYLOAD_CHECK", raising=False)
    monkeypatch.setenv("OSPREY_CANOPY_PAYLOAD", str(repo.parent / "payload"))
    result = _build(repo)
    assert result.exit_code == 0, result.output
    assert "CANOPY panel off" not in result.output
    assert _canopy_enabled(repo / "build" / "config.yml")
    assert (repo / "build" / ".claude" / "skills" / "canopy-operator").is_dir()
