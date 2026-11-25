"""The retired-root detector and the refusal that carries its script.

No live row moves yet, so every test plants ``moved_from`` declarations on a
substituted layout: the agent-facing rows under ``agent/``, the operator rows
under ``ops/`` and ``data/code_review`` to ``agent/code_review``. The runtime
child's move from ``data/.runtime`` to ``var/runtime`` is the live table's own.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from pathlib import Path

import click
import pytest
import yaml
from click.testing import CliRunner

from osprey import repo_layout
from osprey.cli.main import facility_errors_as_click
from osprey.repo_layout import RootChild, RootEntry, check_layout, with_paths
from osprey.repo_retired import (
    PATH_KEYS,
    RetiredRootRefused,
    find_retired_root,
    refuse_retired_layout,
)
from osprey_facility.errors import RetiredLayoutRefused
from osprey_facility.retired import refuse_retired
from osprey_facility.roots import FacilityRoots

#: Where each planted row lives now; each row's old path is its key's old path.
NEW_PATHS: dict[str, str] = {
    "rules": "agent/rules",
    "skills": "agent/skills",
    "hooks": "agent/hooks",
    "web-terminal-context": "agent/context",
    "scripts": "ops/scripts",
    "services": "ops/services",
    "ci-extra.yml": "ops/ci-extra.yml",
    "osprey.service": "ops/osprey.service",
}

#: The row each moved key had before.
OLD_PATHS: dict[str, str] = {
    "rules": "rules",
    "skills": "skills",
    "hooks": "hooks",
    "web-terminal-context": "web-terminal-context",
    "scripts": "scripts",
    "services": "services",
    "ci-extra.yml": "ci-extra.yml",
    "osprey.service": "osprey.service",
}


def _planted_layout() -> tuple[tuple[RootEntry, ...], tuple[RootChild, ...]]:
    rows = tuple(
        replace(row, moved_from=(OLD_PATHS[row.key],)) if row.key in OLD_PATHS else row
        for row in with_paths(NEW_PATHS)
    )
    # The live children: the runtime child already moved from data/.runtime.
    children = repo_layout.ROOT_CHILDREN
    check_layout(rows, children)
    return rows, children


@pytest.fixture(autouse=True)
def _hermetic_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.resolve()))
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "test@example.invalid")


@pytest.fixture
def planted(monkeypatch: pytest.MonkeyPatch) -> None:
    rows, children = _planted_layout()
    monkeypatch.setattr(repo_layout, "ROOT_ENTRIES", rows)
    monkeypatch.setattr(repo_layout, "ROOT_CHILDREN", children)


def _write(root: Path, relative: str, text: str = "x\n") -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout


def _commit(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "seed")


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _write(root, "profile.yml", "name: demo\n")
    return root


def _refusal(root: Path, roots: FacilityRoots | None = None) -> RetiredRootRefused:
    with pytest.raises(RetiredRootRefused) as caught:
        refuse_retired_layout(root, roots)
    return caught.value


def _run(message: str, elsewhere: Path) -> None:
    elsewhere.mkdir(exist_ok=True)
    done = subprocess.run(
        ["sh", "-e"], input=message, cwd=elsewhere, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


# --- nothing to move ---------------------------------------------------------------


def test_the_live_layout_finds_an_agent_convention_at_its_old_root_path(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "rules/a.md")
    assert not find_retired_root(root).empty


def test_the_live_layout_leaves_a_migrated_tree_alone(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "agent/rules/a.md")
    assert find_retired_root(root).empty
    refuse_retired_layout(root, None)


@pytest.mark.usefixtures("planted")
def test_a_current_tree_is_not_refused(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "agent/rules/a.md")
    _write(root, "canopy-1.0-py3-none-any.whl")
    layout = find_retired_root(root)
    assert layout.empty
    assert layout.deletes == ()
    refuse_retired_layout(root, None)


@pytest.mark.usefixtures("planted")
def test_an_empty_leftover_directory_is_no_retired_layout(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "rules").mkdir()
    assert find_retired_root(root).empty


# --- moves and clashes ---------------------------------------------------------------


@pytest.mark.usefixtures("planted")
def test_a_whole_directory_moves_when_its_successor_is_absent(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "rules/a.md")
    [move] = find_retired_root(root).moves
    assert (move.old, move.new, move.is_dir, move.merge) == ("rules", "agent/rules", True, False)


@pytest.mark.usefixtures("planted")
def test_a_merge_moves_each_file_dot_files_included(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "agent/rules/kept.md")
    _write(root, "rules/a.md")
    _write(root, "rules/.gitkeep")
    _write(root, "rules/sub/.hidden/b.md")
    moves = find_retired_root(root).moves
    assert [(move.old, move.new, move.merge) for move in moves] == [
        ("rules/.gitkeep", "agent/rules/.gitkeep", True),
        ("rules/a.md", "agent/rules/a.md", True),
        ("rules/sub/.hidden/b.md", "agent/rules/sub/.hidden/b.md", True),
    ]


@pytest.mark.usefixtures("planted")
def test_a_file_at_both_sides_is_a_clash(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "agent/rules/a.md")
    _write(root, "rules/a.md")
    layout = find_retired_root(root)
    assert layout.moves == ()
    [clash] = layout.clashes
    assert (clash.old, clash.new) == (
        root.resolve() / "rules/a.md",
        root.resolve() / "agent/rules/a.md",
    )
    message = _refusal(root).format_message()
    assert (
        f"# {root.resolve() / 'rules/a.md'} and {root.resolve() / 'agent/rules/a.md'} both exist:"
        " keep one, delete the other."
    ) in message


@pytest.mark.usefixtures("planted")
def test_a_file_where_the_move_needs_a_directory_is_a_clash(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "agent", "not a directory\n")
    _write(root, "rules/a.md")
    layout = find_retired_root(root)
    assert [(clash.old.name, clash.new.name) for clash in layout.clashes] == [("rules", "agent")]
    assert layout.moves == ()


@pytest.mark.usefixtures("planted")
def test_a_dot_named_old_path_counts(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "data/.runtime/cert.pem")
    [move] = find_retired_root(root).moves
    assert (move.old, move.new) == ("data/.runtime", "var/runtime")


@pytest.mark.usefixtures("planted")
def test_old_paths_are_found_at_the_repo_root_whatever_the_data_key_says(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "profile.yml", "name: demo\ndata: elsewhere\n")
    (root / "elsewhere").mkdir()
    _write(root, "data/.runtime/cert.pem")
    _write(root, "data/code_review/reviewer.md")
    refusal = _refusal(root)
    assert {(move.old, move.new) for move in refusal.layout.moves} == {
        ("data/.runtime", "var/runtime"),
        ("data/code_review", "agent/code_review"),
    }


def test_the_live_layout_refuses_certificates_under_the_repo_data_tree_for_any_data_key(
    tmp_path: Path,
) -> None:
    """Live table, no plant: ``<repo>/data/.runtime`` is refused wherever ``data:`` points."""
    root = _repo(tmp_path)
    _write(root, "profile.yml", "name: demo\ndata: ./elsewhere\n")
    (root / "elsewhere").mkdir()
    _write(root, "data/.runtime/bluesky_curve/bridge/proxy.key_secret", "secret\n")
    refusal = _refusal(root)
    assert [(move.old, move.new) for move in refusal.layout.moves] == [
        ("data/.runtime", "var/runtime")
    ]
    assert not (root / "var").exists()


# --- line edits ------------------------------------------------------------------------


@pytest.mark.usefixtures("planted")
def test_a_dot_slash_value_keeps_its_spelling(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(
        root,
        "profile.yml",
        "name: demo\nconfig:\n  code_review.prompt: ./data/code_review/reviewer.md\n",
    )
    _write(root, "data/code_review/reviewer.md")
    [edit] = find_retired_root(root).line_edits
    assert (edit.line, edit.old, edit.new) == (
        3,
        "  code_review.prompt: ./data/code_review/reviewer.md",
        "  code_review.prompt: ./agent/code_review/reviewer.md",
    )
    assert edit.file == root.resolve() / "profile.yml"


@pytest.mark.usefixtures("planted")
def test_a_facility_service_template_follows_the_services_move(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "services/fmcp/Dockerfile", "FROM x\n")
    _write(root, "services/other/Dockerfile", "FROM x\n")
    _write(
        root,
        "profile.yml",
        "name: demo\n"
        "services:\n"
        "  fmcp:\n    template: services/fmcp\n"
        "  other:\n    template: ./services/other\n"
        "  bundled:\n    template: osprey.mcp\n",
    )
    edits = find_retired_root(root).line_edits
    assert [(edit.line, edit.new) for edit in edits] == [
        (4, "    template: ops/services/fmcp"),
        (6, "    template: ./ops/services/other"),
    ]


@pytest.mark.usefixtures("planted")
def test_layered_documents_are_edited_too(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "services/x/Dockerfile", "FROM x\n")
    _write(
        root,
        "tiers/prod.yml",
        "config:\n  services:\n    x:\n      path: services/x\n    graphdb:\n      ttl_path: scripts/x.ttl\n",
    )
    _write(
        root,
        "profiles/lab.yml",
        'lifecycle:\n  pre_build:\n    - name: s\n      run: echo hi\n      cwd: "scripts"\n',
    )
    edits = find_retired_root(root).line_edits
    assert [(edit.file.name, edit.line, edit.new) for edit in edits] == [
        ("prod.yml", 6, "      ttl_path: ops/scripts/x.ttl"),
        ("lab.yml", 5, '      cwd: "ops/scripts"'),
    ]


@pytest.mark.usefixtures("planted")
def test_an_absolute_or_anchored_value_goes_to_check_by_hand(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    absolute = f"{root.resolve()}/data/code_review/reviewer.md"
    _write(
        root,
        "profile.yml",
        "name: demo\n"
        f"config:\n  code_review.prompt: {absolute}\n"
        "dispatch:\n  triggers: '{project_root}/scripts/triggers.yml'\n"
        "lifecycle:\n  validate:\n    - name: v\n      run: sh scripts/check.sh\n",
    )
    _write(root, "data/code_review/reviewer.md")
    layout = find_retired_root(root)
    assert layout.line_edits == ()
    assert layout.check_by_hand[:3] == (
        f"profile.yml:3: config.code_review.prompt: {absolute}",
        "profile.yml:5: dispatch.triggers: {project_root}/scripts/triggers.yml",
        "profile.yml:9: lifecycle.validate.*.run: sh scripts/check.sh",
    )


@pytest.mark.usefixtures("planted")
def test_convention_exclude_and_other_values_are_left_alone(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(
        root,
        "profile.yml",
        "name: rules/x\n"
        "skills:\n  - skills/beam-check\n"
        "exclude:\n  - hooks/x.py\n"
        "scaffold:\n  user_owned:\n    - rules/mine.md\n"
        "config:\n  project_name: services/x\n",
    )
    _write(root, "rules/a.md")
    layout = find_retired_root(root)
    assert layout.line_edits == ()
    assert not any("profile.yml" in item for item in layout.check_by_hand)


@pytest.mark.usefixtures("planted")
def test_pipeline_and_dockerignore_lines_are_edited(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "services/x/Dockerfile", "FROM x\n")
    _write(root, "ci-extra.yml", "job: {}\n")
    _write(
        root,
        ".gitlab-ci.yml",
        "# Facility jobs go in ci-extra.yml\n"
        "include:\n  - local: ci-extra.yml\n    rules:\n      - exists:\n          - ci-extra.yml\n"
        "build:\n  script:\n"
        '    - docker build -t "$REGISTRY/$SERVICE" "services/$SERVICE"\n'
        "    - echo services are built\n",
    )
    _write(root, ".dockerignore", "services/\n!./scripts/keep.sh\n**/rules\nrulesets\n")
    edits = find_retired_root(root).line_edits
    assert [(edit.file.name, edit.line, edit.new) for edit in edits] == [
        (".gitlab-ci.yml", 3, "  - local: ops/ci-extra.yml"),
        (".gitlab-ci.yml", 6, "          - ops/ci-extra.yml"),
        (".gitlab-ci.yml", 9, '    - docker build -t "$REGISTRY/$SERVICE" "ops/services/$SERVICE"'),
        (".dockerignore", 1, "ops/services/"),
        (".dockerignore", 2, "!./ops/scripts/keep.sh"),
    ]


def test_every_config_path_key_is_a_manifest_key() -> None:
    manifest = Path(__file__).parents[1] / "src/osprey/profiles/config_key_manifest.yml"
    keys = yaml.safe_load(manifest.read_text(encoding="utf-8"))["keys"]
    for pattern in PATH_KEYS:
        if not pattern.startswith("config."):
            continue
        want = pattern.removeprefix("config.").split(".")
        assert any(
            len(key.split(".")) == len(want)
            and all(w in ("*", k) for w, k in zip(want, key.split("."), strict=True))
            for key in keys
        ), pattern


def test_every_manifest_path_default_under_a_moved_row_is_a_path_key() -> None:
    """A manifest key whose default is a repo-relative path is rewritten when it moves."""
    manifest = Path(__file__).parents[1] / "src/osprey/profiles/config_key_manifest.yml"
    keys = yaml.safe_load(manifest.read_text(encoding="utf-8"))["keys"]
    from osprey.repo_retired import _key_matches

    for key, spec in keys.items():
        if not isinstance(spec, dict):
            continue
        default = str(spec.get("default"))
        if default.startswith("data/"):
            assert _key_matches(("config", *key.split(".")), PATH_KEYS), key


# --- deletes, host steps, check by hand -------------------------------------------------


@pytest.mark.usefixtures("planted")
def test_canopy_staging_at_the_root_is_deleted_with_the_moves(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "rules/a.md")
    _write(root, "canopy_payload/STAGED.json", "{}\n")
    _write(root, "canopy-1.0-py3-none-any.whl")
    _write(root, "pydmconverter-2.0-py3-none-any.whl")
    _write(root, "osprey_framework-1.0-py3-none-any.whl")
    _write(root, "canopy_extras-1.0-py3-none-any.whl")
    layout = find_retired_root(root)
    assert layout.deletes == (
        "canopy_payload",
        "canopy-1.0-py3-none-any.whl",
        "pydmconverter-2.0-py3-none-any.whl",
    )


@pytest.mark.usefixtures("planted")
def test_host_steps_name_only_osprey_build_without_a_unit_or_hook(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "rules/a.md")
    _write(root, "osprey.service", "[Unit]\nDescription=hand-written\n")
    assert find_retired_root(root).host_steps == ("osprey build",)


@pytest.mark.usefixtures("planted")
def test_host_steps_reinstall_a_generated_unit_and_boot_hook(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "osprey.service", "# header\n# osprey-scaffold: deploy/systemd\n[Unit]\n")
    _write(root, "scripts/osprey-boot-hook.sh", "#!/bin/sh\n# osprey-scaffold: deploy/boot-hook\n")
    steps = find_retired_root(root).host_steps
    resolved = root.resolve()
    assert steps[:5] == (
        "osprey build",
        f"cp {resolved}/ops/osprey.service ~/.config/systemd/user/",
        "systemctl --user daemon-reload",
        "systemctl --user enable --now osprey.service",
        "crontab -e, replacing the osprey-boot-hook lines with:",
    )
    assert steps[5:7] == ("SHELL=/bin/sh", "HOME=/")
    assert f"{resolved}/ops/scripts/osprey-boot-hook.sh" in steps[7]


@pytest.mark.usefixtures("planted")
def test_moved_files_with_inner_paths_are_listed_to_check_by_hand(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "ci-extra.yml", "job: {}\n")
    _write(root, "services/x/Dockerfile", "FROM x\n")
    _write(root, "scripts/run.sh", "echo\n")
    assert find_retired_root(root).check_by_hand == (
        "ops/ci-extra.yml",
        "ops/services/x/Dockerfile",
        "ops/scripts/ (operator scripts)",
    )


# --- the script ----------------------------------------------------------------------


@pytest.mark.usefixtures("planted")
def test_untracked_sources_move_with_mv(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "rules/a.md")
    script = _refusal(root).script
    assert script == (f"cd {root.resolve()}", "mkdir -p agent", "mv rules agent/rules")


@pytest.mark.usefixtures("planted")
def test_tracked_sources_move_with_git_mv(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "rules/a.md")
    _write(root, "skills/b/SKILL.md")
    _commit(root)
    _write(root, "hooks/new.py")
    script = _refusal(root).script
    assert "git mv rules agent/rules" in script
    assert "git mv skills agent/skills" in script
    assert "mv hooks agent/hooks" in script


@pytest.mark.usefixtures("planted")
def test_the_whole_refusal_runs_under_sh_and_leaves_nothing_retired(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(
        root,
        "profile.yml",
        "name: demo\nconfig:\n  code_review.prompt: ./data/code_review/reviewer.md\n"
        "  services.x.path: ./services/x\n",
    )
    _write(root, "data/code_review/reviewer.md", "review\n")
    _write(root, "data/.runtime/cert.pem")
    _write(root, "services/x/Dockerfile", "FROM x\n")
    _write(root, "agent/rules/kept.md")
    _write(root, "rules/a.md")
    _write(root, "rules/.gitkeep", "")
    _write(root, ".dockerignore", "services/\n")
    _write(root, "canopy_payload/STAGED.json", "{}\n")
    _write(root, "canopy-1.0-py3-none-any.whl")
    _commit(root)
    _write(root, "skills/new/SKILL.md")
    message = _refusal(root).format_message()
    _run(message, tmp_path / "elsewhere")
    assert find_retired_root(root).empty
    assert (root / "agent/rules/.gitkeep").is_file()
    assert (root / "agent/rules/a.md").is_file()
    assert (root / "agent/skills/new/SKILL.md").is_file()
    assert (root / "var/runtime/cert.pem").is_file()
    assert not (root / "rules").exists()
    assert not (root / "data").exists()
    assert not (root / "canopy_payload").exists()
    assert not (root / "canopy-1.0-py3-none-any.whl").exists()
    profile = (root / "profile.yml").read_text(encoding="utf-8")
    assert "./agent/code_review/reviewer.md" in profile
    assert "services.x.path: ./services/x" in profile
    assert (root / ".dockerignore").read_text(encoding="utf-8") == "ops/services/\n"


@pytest.mark.usefixtures("planted")
def test_the_refusal_reads_as_the_facility_refusal_does(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "osprey.service", "# osprey-scaffold: deploy/systemd\n[Unit]\n")
    _write(root, "ci-extra.yml", "job: {}\n")
    lines = _refusal(root).format_message().splitlines()
    resolved = root.resolve()
    assert lines[0] == (
        f"# The deployment repo at {resolved} still holds sources at retired paths;"
        " nothing was read."
    )
    assert lines[1] == "# Move them to their current paths, then run the command again:"
    assert lines[2] == f"cd {resolved}"
    assert "# Host steps, once the files moved:" in lines
    assert "#   osprey build" in lines
    assert "# Check by hand; the paths inside these are not rewritten:" in lines
    assert "#   ops/ci-extra.yml" in lines
    assert lines[-1] == "# or run: osprey init --revalidate"
    assert lines.index("# Host steps, once the files moved:") < lines.index(
        "# Check by hand; the paths inside these are not rewritten:"
    )


# --- the facility half -----------------------------------------------------------------


@pytest.mark.usefixtures("planted")
def test_the_combined_block_moves_the_root_then_the_facility(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "profile.yml", "name: demo\ndata: data\n")
    _write(root, "data/facility/identity.yaml", "code: demo\n")
    _write(root, "data/facility/classes.yaml", "- class: Widget\n")
    _write(root, "rules/a.md")
    _commit(root)
    refusal = _refusal(root)
    lines = refusal.format_message().splitlines()
    resolved = root.resolve()
    assert lines[1] == f"# So does the facility tree at {resolved / 'data/facility'}."
    script = list(refusal.script)
    assert script[0] == f"cd {resolved}"
    assert sum(line.startswith("cd ") for line in script) == 1
    root_move = script.index("git mv rules agent/rules")
    facility_move = next(i for i, line in enumerate(script) if "classes.yaml" in line)
    assert root_move < facility_move
    _run(refusal.format_message(), tmp_path / "elsewhere")
    assert (root / "agent/rules/a.md").is_file()
    assert not (root / "data/facility/classes.yaml").exists()


@pytest.mark.usefixtures("planted")
def test_given_facility_roots_win_over_the_profile(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "rules/a.md")
    facility = tmp_path / "machine"
    _write(facility, "identity.yaml", "code: demo\n")
    _write(facility, "classes.yaml", "- class: Widget\n")
    refusal = _refusal(root, FacilityRoots.single(facility))
    assert any("classes.yaml" in line for line in refusal.script)
    _run(refusal.format_message(), tmp_path / "elsewhere")
    assert not (facility / "classes.yaml").exists()


@pytest.mark.usefixtures("planted")
def test_an_unreadable_profile_drops_only_the_facility_half(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "profile.yml", "name: [unclosed\n")
    _write(root, "rules/a.md")
    refusal = _refusal(root)
    assert refusal.facility is None
    assert "mv rules agent/rules" in refusal.script


@pytest.mark.usefixtures("planted")
def test_a_facility_only_tree_raises_the_facility_refusal_unchanged(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    facility = root / "data/facility"
    _write(facility, "identity.yaml", "code: demo\n")
    _write(facility, "classes.yaml", "- class: Widget\n")
    with pytest.raises(RetiredLayoutRefused) as expected:
        refuse_retired(facility, facility)
    with pytest.raises(RetiredLayoutRefused) as caught:
        refuse_retired_layout(root, FacilityRoots.single(facility))
    assert type(caught.value) is RetiredLayoutRefused
    assert caught.value.format_message() == expected.value.format_message()


def test_without_roots_a_clean_root_does_not_read_the_facility(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "profile.yml", "name: demo\ndata: data\n")
    _write(root, "data/facility/classes.yaml", "- class: Widget\n")
    refuse_retired_layout(root, None)


# --- how the command line prints it -------------------------------------------------------


@pytest.mark.usefixtures("planted")
def test_the_refusal_prints_alone_and_exits_one(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "rules/a.md")
    expected = _refusal(root).format_message()

    @click.command()
    def cmd() -> None:
        with facility_errors_as_click():
            refuse_retired_layout(root, None)

    result = CliRunner().invoke(cmd, [])
    assert result.exit_code == 1
    assert result.stderr == expected + "\n"
    assert "Error: " not in result.stderr
    assert result.stdout == ""
