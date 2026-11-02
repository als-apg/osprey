"""The ``code_review`` config section: reading it, validating it, finding its prompt."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from osprey.mcp_server.python_executor.tools._execution_gates import VALID_EXECUTION_MODES
from osprey.services.code_review.settings import (
    DEFAULT_PROMPT,
    DEFAULT_TIME_LIMIT_S,
    HOOK_OVERHEAD_S,
    LEVELS,
    REVIEW_STATE_DIR,
    ReviewConfigError,
    ReviewSettings,
    hook_deadline_s,
    load_review_settings,
    parse_review_section,
    read_review_config,
    resolve_prompt,
)
from osprey.utils.workspace import STATE_DIR_NAME, STATE_DIR_RELPATH
from osprey_connectors.call_timeout import refuse_renamed_timeout_keys

#: Every malformed section ``parse_review_section`` refuses, ``enabled: true`` so
#: the same table also runs through ``load_review_settings``.
MALFORMED_ENABLED_SECTIONS = [
    pytest.param({"enabled": True, "level": "warn"}, "code_review.level", id="unknown-level"),
    pytest.param({"enabled": True, "level": None}, "code_review.level", id="null-level"),
    pytest.param({"enabled": True, "level": ["block"]}, "code_review.level", id="list-level"),
    pytest.param({"enabled": True, "prompt": 3}, "code_review.prompt", id="int-prompt"),
    pytest.param({"enabled": True, "prompt": None}, "code_review.prompt", id="null-prompt"),
    pytest.param({"enabled": True, "prompt": " "}, "code_review.prompt", id="blank-prompt"),
    pytest.param(
        {"enabled": True, "prompt": "${REVIEW_PROMPT}", "time_limit": "${LIMIT}"},
        "code_review.time_limit",
        id="placeholder-is-a-literal",
    ),
    pytest.param({"enabled": True, "modes": ["write"]}, "code_review.modes", id="unknown-mode"),
    pytest.param(
        {"enabled": True, "modes": "readwrite"}, "code_review.modes", id="string-not-a-list"
    ),
    pytest.param({"enabled": True, "modes": [1]}, "code_review.modes", id="int-mode"),
    pytest.param({"enabled": True, "modes": [["readonly"]]}, "code_review.modes", id="nested-mode"),
    pytest.param({"enabled": True, "modes": {"readonly": 1}}, "code_review.modes", id="map-modes"),
    pytest.param({"enabled": True, "modes": None}, "code_review.modes", id="null-modes"),
    pytest.param({"enabled": True, "time_limit": 0}, "code_review.time_limit", id="zero-limit"),
    pytest.param({"enabled": True, "time_limit": -1.5}, "code_review.time_limit", id="negative"),
    pytest.param({"enabled": True, "time_limit": True}, "code_review.time_limit", id="bool-limit"),
    pytest.param({"enabled": True, "time_limit": "60"}, "code_review.time_limit", id="str-limit"),
    pytest.param({"enabled": True, "time_limit": None}, "code_review.time_limit", id="null-limit"),
    pytest.param(
        {"enabled": True, "time_limit": float("inf")}, "code_review.time_limit", id="inf-limit"
    ),
    pytest.param(
        {"enabled": True, "time_limit": float("nan")}, "code_review.time_limit", id="nan-limit"
    ),
    pytest.param({"enabled": True, "time_limit": 10**400}, "code_review.time_limit", id="huge"),
]


@pytest.fixture
def config_file(tmp_path, monkeypatch) -> Path:
    """A config path ``resolve_config_path`` answers with; the file is not written yet."""
    path = tmp_path / "config.yml"
    monkeypatch.setenv("OSPREY_CONFIG", str(path))
    return path


@pytest.fixture
def repo(tmp_path) -> Path:
    """A repo root holding the default prompt."""
    root = tmp_path / "repo"
    prompt = root / DEFAULT_PROMPT
    prompt.parent.mkdir(parents=True)
    prompt.write_text("You review code.\n", encoding="utf-8")
    return root


def _write(root: Path, relpath: str, text: str = "prompt") -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Constants and defaults
# ---------------------------------------------------------------------------


def test_defaults_are_off_and_block():
    settings = ReviewSettings()
    assert settings.enabled is False
    assert settings.level == "block"
    assert settings.prompt == DEFAULT_PROMPT == "data/code_review/reviewer.md"
    assert settings.modes == frozenset({"readwrite"})
    assert settings.time_limit == DEFAULT_TIME_LIMIT_S == 60.0
    assert LEVELS == ("flag", "block")


def test_settings_are_frozen():
    with pytest.raises(AttributeError):
        ReviewSettings().enabled = True  # type: ignore[misc]


def test_review_state_dir_is_a_sibling_of_the_state_dir():
    assert REVIEW_STATE_DIR == f"{STATE_DIR_NAME}/code_review"
    assert not Path(REVIEW_STATE_DIR).is_relative_to(STATE_DIR_RELPATH)


def test_hook_deadline_of_the_defaults_is_the_hook_frontmatter_timeout():
    assert HOOK_OVERHEAD_S == 30
    assert hook_deadline_s(ReviewSettings()) == 90


@pytest.mark.parametrize(
    ("limit", "deadline"),
    [(0.5, 31), (60.0, 90), (60.1, 91), (120, 150)],
    ids=["sub-second", "whole", "rounds-up", "int"],
)
def test_hook_deadline_outlasts_the_time_limit(limit, deadline):
    settings = ReviewSettings(time_limit=limit)
    assert hook_deadline_s(settings) == deadline
    assert isinstance(hook_deadline_s(settings), int)
    assert hook_deadline_s(settings) > settings.time_limit


# ---------------------------------------------------------------------------
# parse_review_section
# ---------------------------------------------------------------------------


def test_parse_explicit_mapping():
    settings = parse_review_section(
        {
            "enabled": True,
            "level": "flag",
            "prompt": "data/review/mine.md",
            "modes": ["readonly", "readwrite"],
            "time_limit": 45,
        }
    )
    assert settings == ReviewSettings(
        enabled=True,
        level="flag",
        prompt="data/review/mine.md",
        modes=frozenset({"readonly", "readwrite"}),
        time_limit=45.0,
    )
    assert isinstance(settings.time_limit, float)


def test_parse_empty_section_is_the_defaults():
    assert parse_review_section({}) == ReviewSettings()


def test_parse_accepts_every_mode_the_executor_knows():
    modes = sorted(VALID_EXECUTION_MODES)
    assert parse_review_section({"modes": modes}).modes == VALID_EXECUTION_MODES


def test_parse_accepts_an_empty_mode_list():
    assert parse_review_section({"enabled": True, "modes": []}).modes == frozenset()


@pytest.mark.parametrize(("section", "names"), MALFORMED_ENABLED_SECTIONS)
def test_parse_refuses_malformed_section(section, names):
    with pytest.raises(ReviewConfigError, match=names):
        parse_review_section(section)


@pytest.mark.parametrize("section", [None, "on", 1, ["enabled"]], ids=repr)
def test_parse_refuses_non_mapping_section(section):
    with pytest.raises(ReviewConfigError, match="code_review"):
        parse_review_section(section)


@pytest.mark.parametrize("enabled", ["yes", 1, None], ids=repr)
def test_parse_refuses_non_bool_enabled(enabled):
    with pytest.raises(ReviewConfigError, match="code_review.enabled"):
        parse_review_section({"enabled": enabled})


def test_parse_validates_a_disabled_section_too():
    with pytest.raises(ReviewConfigError, match="code_review.level"):
        parse_review_section({"enabled": False, "level": "warn"})


# ---------------------------------------------------------------------------
# load_review_settings, with a config in hand
# ---------------------------------------------------------------------------


def test_load_without_a_section_is_off():
    assert load_review_settings({"other": 1}) == ReviewSettings()


def test_load_explicit_mapping():
    settings = load_review_settings(
        {"code_review": {"enabled": True, "level": "flag", "time_limit": 10}}
    )
    assert settings == ReviewSettings(enabled=True, level="flag", time_limit=10.0)


@pytest.mark.parametrize(
    "section",
    [
        {"enabled": False, "level": "warn"},
        {"level": "warn"},
        {"enabled": False, "time_limit": True, "modes": "x", "prompt": 3},
    ],
    ids=["explicit-false", "absent", "everything-wrong"],
)
def test_load_of_an_off_section_does_not_validate_it(section):
    assert load_review_settings({"code_review": section}) == ReviewSettings()


def test_runtime_and_build_differ_on_a_malformed_off_section():
    section = {"enabled": False, "level": "warn"}
    assert load_review_settings({"code_review": section}) == ReviewSettings()
    with pytest.raises(ReviewConfigError):
        parse_review_section(section)


@pytest.mark.parametrize(("section", "names"), MALFORMED_ENABLED_SECTIONS)
def test_load_refuses_malformed_enabled_section(section, names):
    with pytest.raises(ReviewConfigError, match=names):
        load_review_settings({"code_review": section})


@pytest.mark.parametrize("enabled", ["yes", 1, 0, None], ids=repr)
def test_load_refuses_non_bool_enabled(enabled):
    with pytest.raises(ReviewConfigError, match="code_review.enabled"):
        load_review_settings({"code_review": {"enabled": enabled}})


@pytest.mark.parametrize("section", [None, "off", False, []], ids=repr)
def test_load_refuses_non_mapping_section(section):
    with pytest.raises(ReviewConfigError, match="code_review"):
        load_review_settings({"code_review": section})


# ---------------------------------------------------------------------------
# load_review_settings(None) and read_review_config: the file itself
# ---------------------------------------------------------------------------


def test_no_config_file_is_the_defaults(config_file):
    assert not config_file.exists()
    assert read_review_config(config_file) == {}
    assert load_review_settings() == ReviewSettings()


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("code_review: {enabled: [unclosed\n", id="parse-error"),
        pytest.param("- just\n- a list\n", id="non-mapping"),
        pytest.param("", id="empty-document"),
        pytest.param("when: 2026-13-45\n", id="impossible-date"),
        pytest.param("[" * 100_000, id="too-deep"),
    ],
)
def test_broken_config_file_is_refused_and_never_reads_as_off(config_file, text):
    config_file.write_text(text, encoding="utf-8")
    with pytest.raises(ReviewConfigError, match=str(config_file)):
        read_review_config(config_file)
    with pytest.raises(ReviewConfigError):
        load_review_settings()


def test_non_utf8_config_file_is_refused(config_file):
    config_file.write_bytes(b"code_review: \xff\xfe\n")
    with pytest.raises(ReviewConfigError, match=str(config_file)):
        load_review_settings()


def test_config_path_that_is_a_directory_is_refused(tmp_path):
    with pytest.raises(ReviewConfigError, match=str(tmp_path)):
        read_review_config(tmp_path)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 file")
def test_unreadable_config_file_is_refused(config_file):
    config_file.write_text("code_review: {enabled: true}\n", encoding="utf-8")
    config_file.chmod(0)
    try:
        with pytest.raises(ReviewConfigError, match=str(config_file)):
            load_review_settings()
    finally:
        config_file.chmod(0o600)


def test_placeholders_are_literals_and_fail_validation(config_file, monkeypatch):
    monkeypatch.setenv("REVIEW_LEVEL", "flag")
    config_file.write_text(
        'code_review: {enabled: true, level: "${REVIEW_LEVEL}"}\n', encoding="utf-8"
    )
    assert read_review_config(config_file)["code_review"]["level"] == "${REVIEW_LEVEL}"
    with pytest.raises(ReviewConfigError, match="code_review.level"):
        load_review_settings()


def test_defect_in_another_section_is_not_this_features_concern(config_file):
    defect = "control_system:\n  connector:\n    epics:\n      timeout: 5\n"
    # The section the other readers refuse really is a defect: that is what is being ignored.
    config_file.write_text(defect, encoding="utf-8")
    with pytest.raises(ValueError, match="timeout"):
        refuse_renamed_timeout_keys(read_review_config(config_file))

    config_file.write_text(
        defect + "code_review:\n  enabled: true\n  level: flag\n", encoding="utf-8"
    )
    assert load_review_settings() == ReviewSettings(enabled=True, level="flag")


def test_an_edit_is_seen_by_the_next_load(config_file):
    config_file.write_text("code_review: {enabled: true, level: flag}\n", encoding="utf-8")
    assert load_review_settings().level == "flag"

    config_file.write_text("code_review: {enabled: true, level: block}\n", encoding="utf-8")
    assert load_review_settings().level == "block"

    config_file.write_text("code_review: {enabled: false}\n", encoding="utf-8")
    assert load_review_settings().enabled is False


# ---------------------------------------------------------------------------
# resolve_prompt
# ---------------------------------------------------------------------------


def test_resolve_default_prompt_anchors_at_the_repo_root(repo, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_prompt(ReviewSettings(), repo, {}) == "You review code.\n"


def test_resolve_absolute_prompt_inside_the_source_zone(repo):
    settings = ReviewSettings(prompt=str(repo / DEFAULT_PROMPT))
    assert resolve_prompt(settings, repo, {}) == "You review code.\n"


def test_resolve_prompt_in_another_source_entry(repo):
    _write(repo, "rules/review.md", "from rules")
    assert resolve_prompt(ReviewSettings(prompt="rules/review.md"), repo, {}) == "from rules"


def test_resolve_refuses_missing_prompt(repo):
    with pytest.raises(ReviewConfigError, match="data/code_review/absent.md"):
        resolve_prompt(ReviewSettings(prompt="data/code_review/absent.md"), repo, {})


def test_resolve_refuses_directory_prompt(repo):
    with pytest.raises(ReviewConfigError, match="code_review.prompt"):
        resolve_prompt(ReviewSettings(prompt="data/code_review"), repo, {})


def test_resolve_refuses_non_utf8_prompt(repo):
    (repo / "data/latin1.md").write_bytes(b"caf\xe9")
    with pytest.raises(ReviewConfigError, match="data/latin1.md"):
        resolve_prompt(ReviewSettings(prompt="data/latin1.md"), repo, {})


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 file")
def test_resolve_refuses_unreadable_prompt(repo):
    prompt = repo / DEFAULT_PROMPT
    prompt.chmod(0)
    try:
        with pytest.raises(ReviewConfigError, match="code_review.prompt"):
            resolve_prompt(ReviewSettings(), repo, {})
    finally:
        prompt.chmod(0o600)


def test_resolve_refuses_nul_in_prompt_path(repo):
    with pytest.raises(ReviewConfigError, match="code_review.prompt"):
        resolve_prompt(ReviewSettings(prompt="data/a\0b.md"), repo, {})


@pytest.mark.parametrize(
    "relpath",
    [
        "build/reviewer.md",
        "var/state/reviewer.md",
        "reviewer.md",
        "data_evil/reviewer.md",
        "src/reviewer.md",
    ],
)
def test_resolve_refuses_prompt_outside_the_source_zone(repo, relpath):
    _write(repo, relpath)
    with pytest.raises(ReviewConfigError, match="source"):
        resolve_prompt(ReviewSettings(prompt=relpath), repo, {})


def test_resolve_refuses_dotdot_out_of_the_source_zone(repo):
    _write(repo, "build/reviewer.md")
    with pytest.raises(ReviewConfigError, match="source"):
        resolve_prompt(ReviewSettings(prompt="data/../build/reviewer.md"), repo, {})


def test_resolve_refuses_absolute_prompt_outside_the_repo(repo, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("nope", encoding="utf-8")
    with pytest.raises(ReviewConfigError, match="source"):
        resolve_prompt(ReviewSettings(prompt=str(outside)), repo, {})


def test_resolve_refuses_symlink_out_of_the_source_zone(repo):
    _write(repo, "var/state/generated.md")
    (repo / "data/link.md").symlink_to(repo / "var/state/generated.md")
    with pytest.raises(ReviewConfigError, match="source"):
        resolve_prompt(ReviewSettings(prompt="data/link.md"), repo, {})


def test_resolve_refuses_prompt_under_a_configured_agent_data_root(repo):
    _write(repo, "data/agent/reviewer.md", "the agent wrote this")
    settings = ReviewSettings(prompt="data/agent/reviewer.md")
    # Inside the source zone, and acceptable until agent data is relocated over it.
    assert resolve_prompt(settings, repo, {}) == "the agent wrote this"

    config = {"agent_data": {"base_dir": "data/agent"}}
    with pytest.raises(ReviewConfigError, match="agent"):
        resolve_prompt(settings, repo, config)


def test_resolve_refuses_prompt_under_an_absolute_agent_data_root(repo):
    _write(repo, "data/agent/reviewer.md")
    config = {"agent_data": {"base_dir": str(repo / "data")}}
    with pytest.raises(ReviewConfigError, match="agent"):
        resolve_prompt(ReviewSettings(prompt="data/agent/reviewer.md"), repo, config)
