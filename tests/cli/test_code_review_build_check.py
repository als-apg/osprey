"""The build's refusals of a ``code_review`` section.

A present section is validated whether or not it is enabled, so the build refuses a
superset of what a running session refuses. Hook selection and the prompt are checked
only when review is on: a section that is off asks nothing of either.

The real-build half goes through ``osprey build``, since a check nothing calls would
pass every unit test here.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_posture_check import PROFILE_FILENAME, check_code_review
from osprey.cli.init_cmd import init
from osprey.facility.errors import FacilityBuildError
from osprey.services.code_review.settings import DEFAULT_PROMPT
from tests.cli._lifecycle_build import invoke_build

BOTH_HOOKS = ("code-review", "memory-guard")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo root holding a reviewer prompt at the default path."""
    prompt = tmp_path / DEFAULT_PROMPT
    prompt.parent.mkdir(parents=True)
    prompt.write_text("Review this.\n", "utf-8")
    return tmp_path


def _write(repo: Path, relative: str) -> str:
    """Write a prompt, so a refusal of it comes from where it lies and not from its absence."""
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Review this.\n", "utf-8")
    return relative


# ---------------------------------------------------------------------------
# What passes
# ---------------------------------------------------------------------------


def test_an_absent_section_passes_without_any_check(tmp_path: Path) -> None:
    check_code_review({"hooks": {"debug": False}}, [], tmp_path)


def test_a_valid_enabled_section_passes(repo: Path) -> None:
    check_code_review({"code_review": {"enabled": True}}, BOTH_HOOKS, repo)


def test_a_full_valid_enabled_section_passes(repo: Path) -> None:
    section = {
        "enabled": True,
        "level": "flag",
        "prompt": _write(repo, "data/review/mine.md"),
        "modes": ["readonly", "readwrite"],
        "time_limit": 12.5,
    }

    check_code_review({"code_review": section}, BOTH_HOOKS, repo)


@pytest.mark.parametrize(
    "section",
    [
        pytest.param({}, id="empty"),
        pytest.param({"enabled": False}, id="off"),
        pytest.param({"enabled": False, "level": "flag", "time_limit": 5}, id="off-with-keys"),
        pytest.param({"enabled": False, "prompt": "data/absent.md"}, id="off-absent-prompt"),
        pytest.param({"enabled": False, "prompt": "build/reviewer.md"}, id="off-outside-zone"),
    ],
)
def test_a_valid_disabled_section_skips_the_prompt_and_hook_checks(
    tmp_path: Path, section: dict[str, Any]
) -> None:
    check_code_review({"code_review": section}, [], tmp_path)


# ---------------------------------------------------------------------------
# A present section is validated, enabled or not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
@pytest.mark.parametrize(
    ("overrides", "key"),
    [
        pytest.param({"level": "warn"}, "code_review.level", id="unknown-level"),
        pytest.param({"prompt": 7}, "code_review.prompt", id="non-string-prompt"),
        pytest.param({"prompt": "  "}, "code_review.prompt", id="blank-prompt"),
        pytest.param({"modes": "readwrite"}, "code_review.modes", id="modes-not-a-list"),
        pytest.param({"modes": ["sideways"]}, "code_review.modes", id="unknown-mode"),
        pytest.param({"time_limit": 0}, "code_review.time_limit", id="zero-limit"),
        pytest.param({"time_limit": "60"}, "code_review.time_limit", id="string-limit"),
        pytest.param({"time_limit": True}, "code_review.time_limit", id="bool-limit"),
        pytest.param({"time_limit": float("nan")}, "code_review.time_limit", id="nan-limit"),
    ],
)
def test_a_malformed_key_is_refused_whatever_enabled_says(
    repo: Path, enabled: bool, overrides: dict[str, Any], key: str
) -> None:
    section = {"enabled": enabled, **overrides}

    with pytest.raises(FacilityBuildError) as caught:
        check_code_review({"code_review": section}, BOTH_HOOKS, repo)

    error = caught.value
    assert error.kind == "profile-invalid"
    assert key in str(error)
    assert PROFILE_FILENAME in error.remedy
    assert list(error.sources) == [PROFILE_FILENAME]


@pytest.mark.parametrize(
    ("section", "key"),
    [
        pytest.param(None, "code_review", id="null-section"),
        pytest.param("yes", "code_review", id="string-section"),
        pytest.param(["enabled"], "code_review", id="list-section"),
        pytest.param({"enabled": "false"}, "code_review.enabled", id="quoted-false"),
        pytest.param({"enabled": 1}, "code_review.enabled", id="number-enabled"),
        pytest.param({"enabled": None}, "code_review.enabled", id="empty-enabled"),
    ],
)
def test_a_section_that_is_neither_on_nor_off_is_refused(
    repo: Path, section: object, key: str
) -> None:
    with pytest.raises(FacilityBuildError) as caught:
        check_code_review({"code_review": section}, BOTH_HOOKS, repo)

    assert caught.value.kind == "profile-invalid"
    assert key in str(caught.value)
    assert PROFILE_FILENAME in caught.value.remedy


def test_a_disabled_section_with_an_unknown_level_is_a_build_error(tmp_path: Path) -> None:
    """Runtime reads this section as off; the build still refuses it."""
    with pytest.raises(FacilityBuildError, match=r"code_review\.level"):
        check_code_review({"code_review": {"enabled": False, "level": "warn"}}, [], tmp_path)


# ---------------------------------------------------------------------------
# Enabled: both hooks are selected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("selected", "missing"),
    [
        pytest.param(["memory-guard"], ["code-review"], id="no-review-hook"),
        pytest.param(["code-review"], ["memory-guard"], id="no-memory-guard"),
        pytest.param([], ["code-review", "memory-guard"], id="neither"),
    ],
)
def test_an_enabled_section_needs_both_hooks(
    repo: Path, selected: list[str], missing: list[str]
) -> None:
    with pytest.raises(FacilityBuildError) as caught:
        check_code_review({"code_review": {"enabled": True}}, selected, repo)

    error = caught.value
    assert error.kind == "profile-invalid"
    assert error.record_id == "code_review.enabled"
    for hook in missing:
        assert f"`{hook}`" in error.detail
        assert f"`{hook}`" in error.remedy
    assert "`hooks:`" in error.remedy
    assert PROFILE_FILENAME in error.remedy


# ---------------------------------------------------------------------------
# Enabled: the prompt resolves
# ---------------------------------------------------------------------------


def test_an_enabled_section_whose_prompt_is_missing_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FacilityBuildError) as caught:
        check_code_review({"code_review": {"enabled": True}}, BOTH_HOOKS, tmp_path)

    error = caught.value
    assert error.kind == "profile-invalid"
    assert error.record_id == "code_review.prompt"
    assert "cannot be read" in error.detail
    assert PROFILE_FILENAME in error.remedy


@pytest.mark.parametrize(
    "prompt",
    ["build/reviewer.md", "var/state/reviewer.md", "reviewer.md", "../outside/reviewer.md"],
)
def test_a_prompt_outside_the_source_zone_is_refused(repo: Path, prompt: str) -> None:
    _write(repo, prompt)

    with pytest.raises(FacilityBuildError) as caught:
        check_code_review({"code_review": {"enabled": True, "prompt": prompt}}, BOTH_HOOKS, repo)

    assert caught.value.record_id == "code_review.prompt"
    assert "source zone" in caught.value.detail


def test_a_prompt_linked_out_of_the_source_zone_is_refused(repo: Path) -> None:
    target = repo / _write(repo, "build/reviewer.md")
    link = repo / "data" / "code_review" / "linked.md"
    link.symlink_to(target)

    with pytest.raises(FacilityBuildError, match="source zone"):
        check_code_review(
            {"code_review": {"enabled": True, "prompt": "data/code_review/linked.md"}},
            BOTH_HOOKS,
            repo,
        )


def test_a_prompt_under_a_relocated_agent_data_root_is_refused(repo: Path) -> None:
    """The carve-out is read from the render, so it overlaps the zone only as stated there."""
    prompt = _write(repo, "data/agent/reviewer.md")
    section = {"enabled": True, "prompt": prompt}

    # Not under the agent's root by default, so the render's own statement is what refuses.
    check_code_review({"code_review": section}, BOTH_HOOKS, repo)
    with pytest.raises(FacilityBuildError) as caught:
        check_code_review(
            {"agent_data": {"base_dir": "data/agent"}, "code_review": section},
            BOTH_HOOKS,
            repo,
        )

    assert caught.value.record_id == "code_review.prompt"
    assert "agent's own data root" in caught.value.detail


# ---------------------------------------------------------------------------
# A real build
#
# A real build that drops memory-guard stops earlier, at the write-tool lint of its
# first render, so that refusal is pinned by the unit tests above.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scaffold(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One hello-world scaffold; each test builds its own copy of it."""
    repo = tmp_path_factory.mktemp("code-review-check") / "scaffold"
    result = CliRunner().invoke(init, [str(repo), "--preset", "hello-world", "--no-git"])
    assert result.exit_code == 0, result.output
    return repo


@pytest.fixture
def project(scaffold: Path, tmp_path: Path) -> Callable[..., Path]:
    """Copy the scaffold and state a review section.

    Returns:
        A callable taking the section and whether to select the review hook; it
        returns the repo unbuilt.
    """

    def _project(section: dict[str, Any], select_hook: bool = True) -> Path:
        repo = tmp_path / "repo"
        shutil.copytree(scaffold, repo)
        profile_path = repo / "profile.yml"
        profile = yaml.safe_load(profile_path.read_text("utf-8"))
        if select_hook:
            profile["hooks"] = [*profile["hooks"], "code-review"]
        profile["config"]["code_review"] = section
        profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), "utf-8")
        return repo

    return _project


def test_a_valid_enabled_profile_builds(project: Callable[..., Path]) -> None:
    repo = project({"enabled": True})

    result = invoke_build(repo)

    assert result.exit_code == 0, result.output


def test_a_valid_disabled_profile_with_an_absent_prompt_builds(
    project: Callable[..., Path],
) -> None:
    repo = project({"enabled": False, "prompt": "data/code_review/absent.md"}, select_hook=False)
    assert not (repo / "data/code_review/absent.md").exists()

    result = invoke_build(repo)

    assert result.exit_code == 0, result.output


def test_a_disabled_profile_with_an_unknown_level_does_not_build(
    project: Callable[..., Path],
) -> None:
    repo = project({"enabled": False, "level": "warn"}, select_hook=False)

    result = invoke_build(repo)

    assert result.exit_code != 0
    assert "code_review.level" in result.output
    assert PROFILE_FILENAME in result.output


@pytest.mark.parametrize("prompt", ["build/reviewer.md", "var/state/reviewer.md"])
def test_an_enabled_profile_whose_prompt_lies_outside_the_zone_does_not_build(
    project: Callable[..., Path], prompt: str
) -> None:
    repo = project({"enabled": True, "prompt": prompt})
    _write(repo, prompt)

    result = invoke_build(repo)

    assert result.exit_code != 0
    assert "code_review.prompt" in result.output
    assert "source zone" in result.output


def test_an_enabled_profile_without_the_review_hook_does_not_build(
    project: Callable[..., Path],
) -> None:
    repo = project({"enabled": True}, select_hook=False)

    result = invoke_build(repo)

    assert result.exit_code != 0
    assert "code_review.enabled" in result.output
    assert "`code-review`" in result.output
