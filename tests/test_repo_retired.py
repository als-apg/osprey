"""The retired-root detector, the one migration plan, and the refusal that carries its block.

Every test reads the live layout table: the agent-facing rows moved under
``agent/``, the operator rows under ``ops/``, the facility tree out of
``data/facility`` to ``facility/``, the web content to ``web/``, the runtime
directory and each lane's certificate directory to ``var/runtime``. ``data/``
itself is dissolved. A test that needs a move the table does not declare
substitutes the table. Each printed block runs under ``sh -e`` with a
stand-in ``osprey`` that records how the block's closing command called it.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import click
import pytest
import yaml
from click.testing import CliRunner

from osprey import repo_layout, repo_retired
from osprey.cli.main import facility_errors_as_click
from osprey.repo_layout import EntryKind, RootMove
from osprey.repo_retired import (
    _PASTE_THEN_PLACE_HEADER,
    PATH_KEYS,
    RetiredRootRefused,
    after_moves_view,
    find_retired_root,
    plan_retired,
    refuse_retired_layout,
    source_of,
    successor_of,
)
from osprey_facility.errors import (
    _PLACE_THESE,
    MOVE_HEADER,
    PASTE_HEADER,
    RetiredLayoutRefused,
    _revalidate_command,
)
from osprey_facility.retired import refuse_retired
from osprey_facility.roots import FacilityRoots

#: The bundled example deck, a real one for a facility tree that must build.
DECK = (
    Path(__file__).parents[1] / "src/osprey/templates/facilities/example/simulation/decks/LTB.json"
)


@pytest.fixture(autouse=True)
def _hermetic_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.resolve()))
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "test@example.invalid")


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


def _run(message: str, elsewhere: Path) -> list[str]:
    """Run *message* under ``sh -e`` from *elsewhere*; each ``osprey`` call it made."""
    elsewhere.mkdir(exist_ok=True)
    bin_dir = elsewhere / "bin"
    bin_dir.mkdir(exist_ok=True)
    calls = elsewhere / "osprey-calls"
    stand_in = bin_dir / "osprey"
    stand_in.write_text(f'#!/bin/sh\necho "$*" >> {calls}\n', encoding="utf-8")
    stand_in.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    done = subprocess.run(
        ["sh", "-e"],
        input=message,
        cwd=elsewhere,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert done.returncode == 0, done.stderr
    return calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []


def _old_facility(root: Path) -> Path:
    """A facility tree at its retired place, with its slots at their retired paths."""
    facility = root / "data/facility"
    _write(facility, "identity.yaml", "code: demo\n")
    _write(facility, "classes.yaml", "- class: Widget\n  parent: Magnet\n")
    _write(facility, "models.yaml", "- name: ring\n  engine: pyat\n  deck: decks/ring.json\n")
    (facility / "decks").mkdir(parents=True)
    shutil.copy(DECK, facility / "decks/ring.json")
    return facility


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


def test_a_current_tree_is_not_refused(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "agent/rules/a.md")
    _write(root, "canopy-1.0-py3-none-any.whl")
    layout = find_retired_root(root)
    assert layout.empty
    assert layout.deletes == ()
    refuse_retired_layout(root, None)


def test_an_empty_leftover_directory_is_no_retired_layout(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "rules").mkdir()
    assert find_retired_root(root).empty


# --- moves and clashes ---------------------------------------------------------------


def test_a_whole_directory_moves_when_its_successor_is_absent(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "rules/a.md")
    [move] = find_retired_root(root).moves
    assert (move.old, move.new, move.is_dir, move.merge) == ("rules", "agent/rules", True, False)


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


def test_a_file_where_the_move_needs_a_directory_is_a_clash(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "agent", "not a directory\n")
    _write(root, "rules/a.md")
    layout = find_retired_root(root)
    assert [(clash.old.name, clash.new.name) for clash in layout.clashes] == [("rules", "agent")]
    assert layout.moves == ()


def test_a_dot_named_old_path_counts(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "data/.runtime/cert.pem")
    [move] = find_retired_root(root).moves
    assert (move.old, move.new) == ("data/.runtime", "var/runtime")


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


def test_layered_documents_are_edited_too(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "services/x/Dockerfile", "FROM x\n")
    _write(
        root,
        "tiers/prod.yml",
        "config:\n  services:\n    x:\n      path: services/x\n    bluesky:\n      plan_dir: scripts/plans\n",
    )
    _write(
        root,
        "profiles/lab.yml",
        'lifecycle:\n  pre_build:\n    - name: s\n      run: echo hi\n      cwd: "scripts"\n',
    )
    edits = find_retired_root(root).line_edits
    assert [(edit.file.name, edit.line, edit.new) for edit in edits] == [
        ("prod.yml", 6, "      plan_dir: ops/scripts/plans"),
        ("lab.yml", 5, '      cwd: "ops/scripts"'),
    ]


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


def _pipeline_lines(tmp_path: Path, script: list[str]) -> tuple[Path, dict[int, str]]:
    """A repo whose pipeline runs *script*; each rewritten line by its number."""
    root = _repo(tmp_path)
    _write(root, "rules/a.md")
    body = "".join(f"    - {command}\n" for command in script)
    _write(root, ".gitlab-ci.yml", f"job:\n  script:\n{body}")
    edits = find_retired_root(root).line_edits
    assert {edit.file.name for edit in edits} <= {".gitlab-ci.yml"}
    return root, {edit.line: edit.new.strip() for edit in edits}


@pytest.mark.parametrize("project_dir", ["$CI_PROJECT_DIR", "${CI_PROJECT_DIR}"])
def test_a_pipeline_path_under_the_project_directory_variable_is_rewritten(
    tmp_path: Path, project_dir: str
) -> None:
    _, rewritten = _pipeline_lines(
        tmp_path,
        [
            f"sh {project_dir}/scripts/verify.sh",
            f'ls "{project_dir}/data/facility" {project_dir}/services',
            f"cp {project_dir}/osprey.service /etc/systemd/user/",
            f"cat '{project_dir}/ci-extra.yml'; ls {project_dir}/build/scripts",
            f"ls {project_dir}/scriptsets {project_dir}/data/other {project_dir}",
            f"ls /srv{project_dir}/scripts/ {project_dir}X/scripts/",
        ],
    )
    assert rewritten == {
        3: f"- sh {project_dir}/ops/scripts/verify.sh",
        4: f'- ls "{project_dir}/facility" {project_dir}/ops/services',
        5: f"- cp {project_dir}/ops/osprey.service /etc/systemd/user/",
        6: f"- cat '{project_dir}/ops/ci-extra.yml'; ls {project_dir}/build/scripts",
    }


def test_the_project_directory_variable_is_a_root_only_in_the_pipeline(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "rules/a.md")
    _write(root, ".dockerignore", "$CI_PROJECT_DIR/scripts/\n")
    assert find_retired_root(root).line_edits == ()


def test_every_config_path_key_is_a_manifest_key() -> None:
    manifest = Path(__file__).parents[1] / "src/osprey/profiles/config_key_manifest.yml"
    keys = yaml.safe_load(manifest.read_text(encoding="utf-8"))["keys"]
    for pattern in PATH_KEYS:
        if not pattern.startswith("config."):
            continue
        # A trailing ``*`` is an item of a list-valued key.
        want = pattern.removeprefix("config.").removesuffix(".*").split(".")
        assert any(
            len(key.split(".")) == len(want)
            and all(w in ("*", k) for w, k in zip(want, key.split("."), strict=True))
            for key in keys
        ), pattern


def _unrewritten(keys: dict[str, object]) -> list[str]:
    """The manifest keys whose default lies under a rendered source but is no path key."""
    from osprey.repo_retired import _key_matches

    sources = [source.source_rel for source in repo_layout.rendered_sources()]
    found = []
    for key, spec in keys.items():
        if not isinstance(spec, dict):
            continue
        defaults = spec.get("default")
        for default in defaults if isinstance(defaults, list) else [defaults]:
            value = str(default)
            if not any(value == source or value.startswith(f"{source}/") for source in sources):
                continue
            path = ("config", *key.split("."))
            if isinstance(defaults, list):
                path = (*path, "*")
            if not _key_matches(path, PATH_KEYS):
                found.append(key)
    return found


def test_every_manifest_path_default_under_a_rendered_source_is_a_path_key() -> None:
    """A manifest key whose default lies in a moved source is rewritten when it moves."""
    manifest = Path(__file__).parents[1] / "src/osprey/profiles/config_key_manifest.yml"
    keys = yaml.safe_load(manifest.read_text(encoding="utf-8"))["keys"]
    assert _unrewritten(keys) == []
    source = repo_layout.rendered_sources()[0].source_rel
    assert _unrewritten({"planted.key": {"default": f"{source}/x"}}) == ["planted.key"]


# --- deletes, host steps, check by hand -------------------------------------------------


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


def test_host_steps_name_only_osprey_build_without_a_unit_or_hook(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "rules/a.md")
    _write(root, "osprey.service", "[Unit]\nDescription=hand-written\n")
    assert find_retired_root(root).host_steps == ("osprey build",)


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


def test_untracked_sources_move_with_mv(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "rules/a.md")
    script = _refusal(root).script
    assert script == (f"cd {root.resolve()}", "mkdir -p agent", "mv rules agent/rules")


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
    assert lines[1] == PASTE_HEADER
    assert lines[2] == f"cd {resolved}"
    assert "# Host steps, once the files moved:" in lines
    assert "#   osprey build" in lines
    assert "# Check by hand; the paths inside these are not rewritten:" in lines
    assert "#   ops/ci-extra.yml" in lines
    assert lines[-1] == _revalidate_command(resolved)
    assert lines[-1] == f"osprey init --revalidate {resolved}"
    assert lines.index("# Host steps, once the files moved:") < lines.index(
        "# Check by hand; the paths inside these are not rewritten:"
    )


# --- the facility half -----------------------------------------------------------------


def test_a_pre_epic_tree_gives_one_block_that_runs_and_leaves_no_refusal(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    _write(root, "profile.yml", "name: demo\ndata: data\n")
    _old_facility(root)
    _write(root, "rules/a.md")
    _commit(root)
    refusal = _refusal(root)
    lines = refusal.format_message().splitlines()
    resolved = root.resolve()
    assert lines[1] == f"# So does the facility tree at {resolved / 'facility'}."
    assert lines[2] == PASTE_HEADER
    script = list(refusal.script)
    assert script[0] == f"cd {resolved}"
    assert sum(line.startswith("cd ") for line in script) == 1
    root_move = script.index("git mv data/facility facility")
    facility_move = script.index("git mv facility/classes.yaml facility/vocabulary/classes.yaml")
    assert root_move < facility_move
    assert script[facility_move - 1] == "mkdir -p facility/vocabulary"
    assert "perl -ni -e 'print unless $. == 2 && m{^data: data\\r?$}' profile.yml" in script
    assert lines[-1] == f"osprey init --revalidate {resolved}"
    assert lines.count(lines[-1]) == 1

    calls = _run(refusal.format_message(), tmp_path / "elsewhere")

    assert calls == [f"init --revalidate {resolved}"]
    assert (root / "agent/rules/a.md").is_file()
    assert (root / "facility/vocabulary/classes.yaml").is_file()
    assert "simulation/decks/ring.json" in (root / "facility/simulation/models.yaml").read_text(
        encoding="utf-8"
    )
    assert not (root / "data").exists()
    assert (root / "profile.yml").read_text(encoding="utf-8") == "name: demo\n"
    refuse_retired_layout(root, FacilityRoots.single(root / "facility"))
    assert plan_retired(root).empty
    status = _git(root, "status", "--porcelain")
    assert "R  data/facility/classes.yaml -> facility/vocabulary/classes.yaml" in status


def test_the_plan_reads_the_facility_tree_as_the_root_moves_leave_it(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _old_facility(root)
    _write(root, "facility/simulation/decks/other.json", "{}\n")
    plan = plan_retired(root)
    resolved = root.resolve()
    assert [(move.old, move.new) for move in plan.layout.moves][:1] == [
        ("data/facility/classes.yaml", "facility/classes.yaml")
    ]
    assert plan.facility_roots == FacilityRoots.single(resolved / "facility")
    assert plan.facility is not None
    assert {move.root for move in plan.facility.moves} == {resolved / "facility"}
    assert ("decks/ring.json", "simulation/decks/ring.json") in [
        (move.old, move.new) for move in plan.facility.moves
    ]
    [edit] = plan.facility.deck_edits
    assert edit.file == resolved / "facility/simulation/models.yaml"


def test_a_facility_clash_after_the_root_moves_names_where_each_side_lies_now(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    _old_facility(root)
    _write(root, "facility/vocabulary/classes.yaml")
    refusal = _refusal(root)
    resolved = root.resolve()
    assert refusal.facility is not None
    [clash] = refusal.facility.clashes
    assert (clash.old, clash.new) == (
        resolved / "data/facility/classes.yaml",
        resolved / "facility/vocabulary/classes.yaml",
    )
    lines = refusal.format_message().splitlines()
    assert MOVE_HEADER in lines
    assert lines[-1] == _PLACE_THESE


def test_given_facility_roots_win_over_the_profile(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "rules/a.md")
    facility = tmp_path / "machine"
    _write(facility, "identity.yaml", "code: demo\n")
    _write(facility, "classes.yaml", "- class: Widget\n")
    refusal = _refusal(root, FacilityRoots.single(facility))
    assert any("classes.yaml" in line for line in refusal.script)
    lines = refusal.format_message().splitlines()
    assert lines[-1] == _revalidate_command(root.resolve())
    assert lines.count(lines[-1]) == 1
    _run(refusal.format_message(), tmp_path / "elsewhere")
    assert not (facility / "classes.yaml").exists()


def test_an_unreadable_profile_drops_only_the_facility_half(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "profile.yml", "name: [unclosed\n")
    _write(root, "rules/a.md")
    refusal = _refusal(root)
    assert refusal.facility is None
    assert "mv rules agent/rules" in refusal.script


def test_a_facility_only_tree_raises_the_facility_refusal_naming_the_repo(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    facility = root / "facility"
    _write(facility, "identity.yaml", "code: demo\n")
    _write(facility, "classes.yaml", "- class: Widget\n")
    with pytest.raises(RetiredLayoutRefused) as expected:
        refuse_retired(facility, facility, repo=root)
    with pytest.raises(RetiredLayoutRefused) as caught:
        refuse_retired_layout(root, FacilityRoots.single(facility))
    assert type(caught.value) is RetiredLayoutRefused
    assert caught.value.format_message() == expected.value.format_message()
    assert caught.value.message.splitlines()[-1] == _revalidate_command(root.resolve())


def test_without_roots_a_clean_root_does_not_read_the_facility(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "facility/classes.yaml", "- class: Widget\n")
    refuse_retired_layout(root, None)


# --- the path map -------------------------------------------------------------------------


def test_a_live_path_maps_back_through_the_longest_successor() -> None:
    moves = repo_layout.retired_moves()
    assert source_of("var/runtime/bluesky_curve/x", moves) == "data/bluesky_curve/x"
    assert source_of("var/runtime/certs/key.pem", moves) == "data/.runtime/certs/key.pem"
    assert source_of("facility/vocabulary", moves) == "data/facility/vocabulary"
    assert source_of("profile.yml", moves) == "profile.yml"
    assert successor_of("data/bluesky_curve/x", moves) == "var/runtime/bluesky_curve/x"
    assert successor_of("data/.runtime/bluesky_curve/x", moves) == "var/runtime/bluesky_curve/x"
    assert successor_of("profile.yml", moves) is None


def test_the_after_moves_view_shows_the_tree_as_the_moves_leave_it(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "data/facility/classes.yaml", "moved\n")
    _write(root, "facility/kept.yaml", "kept\n")
    moves = [RootMove("data/facility", "facility", EntryKind.DIRECTORY)]
    with after_moves_view(root, moves, ["facility"]) as view:
        assert (view / "facility/classes.yaml").read_text(encoding="utf-8") == "moved\n"
        assert (view / "facility/kept.yaml").read_text(encoding="utf-8") == "kept\n"
    assert not view.exists()
    assert (root / "data/facility/classes.yaml").is_file()


# --- inter-move clashes and the lane moves --------------------------------------------------


def test_two_moves_to_one_successor_are_a_clash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = (
        RootMove("rules", "agent/rules", EntryKind.DIRECTORY),
        RootMove("old_rules", "agent/rules", EntryKind.DIRECTORY),
    )
    monkeypatch.setattr(repo_layout, "retired_moves", lambda: table)
    root = _repo(tmp_path)
    _write(root, "rules/a.md")
    _write(root, "old_rules/b.md")
    layout = find_retired_root(root)
    assert [(move.old, move.new) for move in layout.moves] == [("rules", "agent/rules")]
    resolved = root.resolve()
    assert [(clash.old, clash.new) for clash in layout.clashes] == [
        (resolved / "old_rules", resolved / "rules")
    ]
    assert _refusal(root).format_message().splitlines()[-1] == _PLACE_THESE


def test_a_lanes_certificates_in_both_old_places_merge_flat_when_disjoint(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    _write(root, "data/.runtime/bluesky_curve/a.pem", "a\n")
    _write(root, "data/.runtime/other.txt")
    _write(root, "data/bluesky_curve/b.pem", "b\n")
    _commit(root)
    layout = find_retired_root(root)
    assert layout.clashes == ()
    assert [(move.old, move.new) for move in layout.moves] == [
        ("data/.runtime", "var/runtime"),
        ("data/bluesky_curve/b.pem", "var/runtime/bluesky_curve/b.pem"),
    ]
    _run(_refusal(root).format_message(), tmp_path / "elsewhere")
    assert (root / "var/runtime/bluesky_curve/a.pem").read_text(encoding="utf-8") == "a\n"
    assert (root / "var/runtime/bluesky_curve/b.pem").read_text(encoding="utf-8") == "b\n"
    assert not (root / "data").exists()
    assert find_retired_root(root).empty


def test_a_lanes_certificate_in_both_old_places_is_a_clash(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "data/.runtime/bluesky_curve/a.pem", "a\n")
    _write(root, "data/bluesky_curve/a.pem", "other\n")
    layout = find_retired_root(root)
    resolved = root.resolve()
    assert [(clash.old, clash.new) for clash in layout.clashes] == [
        (resolved / "data/bluesky_curve/a.pem", resolved / "data/.runtime/bluesky_curve/a.pem")
    ]
    assert _refusal(root).format_message().splitlines()[-1] == _PLACE_THESE


# --- leftovers ---------------------------------------------------------------------------------


def test_a_finder_file_alone_is_not_refused(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "data/.DS_Store", "")
    assert find_retired_root(root).empty
    refuse_retired_layout(root)


def test_a_finder_file_is_pruned_with_the_vacated_directory(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "data/.DS_Store", "")
    _write(root, "data/empty/.DS_Store", "")
    _write(root, "data/.runtime/cert.pem")
    layout = find_retired_root(root)
    assert layout.leftovers == ()
    assert set(layout.deletes) >= {"data/.DS_Store", "data/empty"}
    _run(_refusal(root).format_message(), tmp_path / "elsewhere")
    assert not (root / "data").exists()


def test_an_untracked_note_in_data_is_refused_naming_it(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "data/notes.md", "mine\n")
    _write(root, "data/raw/scan.csv")
    _write(root, "data/bluesky_unknown_curve/k.pem")
    layout = find_retired_root(root)
    assert not layout.empty
    assert layout.moves == ()
    assert layout.leftovers == (
        "data/bluesky_unknown_curve/",
        "data/notes.md",
        "data/raw/",
    )
    refusal = _refusal(root)
    assert refusal.script == ()
    assert refusal.format_message().splitlines()[1:] == [
        "# Still in data/, place by hand:",
        "#   data/bluesky_unknown_curve/",
        "#   data/notes.md",
        "#   data/raw/",
        "# Content the render must see goes under facility/.",
        _PLACE_THESE,
    ]


def test_a_leftover_beside_moves_is_listed_and_the_block_says_it_stays(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "data/notes.md", "mine\n")
    _write(root, "data/canopy/library/panel.json", "{}\n")
    _write(root, "data/canopy/stray.txt")
    refusal = _refusal(root)
    assert refusal.layout.leftovers == ("data/canopy/stray.txt", "data/notes.md")
    lines = refusal.format_message().splitlines()
    assert lines[1] == _PASTE_THEN_PLACE_HEADER
    assert PASTE_HEADER not in lines
    assert "#   data/notes.md" in lines
    assert lines[-1] == _revalidate_command(root.resolve())
    _run(refusal.format_message(), tmp_path / "elsewhere")
    assert (root / "web/canopy/library/panel.json").is_file()
    assert (root / "data/notes.md").is_file()
    after = _refusal(root)
    assert after.script == ()
    assert after.format_message().splitlines()[1:] == [
        "# Still in data/, place by hand:",
        "#   data/canopy/stray.txt",
        "#   data/notes.md",
        "# Content the render must see goes under facility/.",
        _PLACE_THESE,
    ]


def test_the_shipped_data_readme_is_deleted_and_an_edited_one_is_a_leftover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped = "# What data/ holds\n"
    digest = hashlib.sha256(shipped.encode()).hexdigest()
    monkeypatch.setattr(
        repo_retired,
        "_SHIPPED_DATA_README_DIGESTS",
        repo_retired._SHIPPED_DATA_README_DIGESTS | {digest},
    )
    root = _repo(tmp_path)
    _write(root, "data/README.md", shipped)
    layout = find_retired_root(root)
    assert layout.shipped == ("data/README.md",)
    assert layout.leftovers == ()
    _run(_refusal(root).format_message(), tmp_path / "elsewhere")
    assert not (root / "data").exists()
    _write(root, "data/README.md", shipped + "my notes\n")
    layout = find_retired_root(root)
    assert layout.shipped == ()
    assert layout.leftovers == ("data/README.md",)


# --- the data: key -----------------------------------------------------------------------------


def test_the_default_data_key_alone_is_refused_and_deleted(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "profile.yml", "name: demo\ndata: ./data/\nmachine: ''\n")
    layout = find_retired_root(root)
    assert layout.data_keys == ("profile.yml:2: data: ./data/",)
    [delete] = layout.line_deletes
    assert (delete.line, delete.text) == (2, "data: ./data/")
    _run(_refusal(root).format_message(), tmp_path / "elsewhere")
    assert (root / "profile.yml").read_text(encoding="utf-8") == "name: demo\nmachine: ''\n"
    assert find_retired_root(root).empty


def test_a_tag_no_loader_knows_does_not_hide_the_data_key(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "profile.yml", "name: demo\ndata: ./data/\nextra: !custom value\n")
    _write(root, "profiles/lab.yml", "data: !custom data\n")
    layout = find_retired_root(root)
    assert layout.data_keys == (
        "profile.yml:2: data: ./data/",
        "profiles/lab.yml:1: data: data",
    )
    assert layout.line_deletes == ()
    assert any(item.startswith("profiles/lab.yml:1: data: ") for item in layout.check_by_hand)
    with pytest.raises(RetiredRootRefused):
        refuse_retired_layout(root)


def test_an_out_of_repo_data_key_is_not_deleted_and_still_refuses(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    outside = tmp_path / "outside"
    _write(outside, "facility/classes.yaml")
    _write(root, "profile.yml", f"name: demo\ndata: {outside}\n")
    _write(root, "data/facility/identity.yaml", "code: demo\n")
    layout = find_retired_root(root)
    assert layout.line_deletes == ()
    assert layout.data_root is None
    assert layout.data_keys == (f"profile.yml:2: data: {outside}",)
    assert any(item.startswith(f"profile.yml:2: data: {outside}") for item in layout.check_by_hand)
    assert not any(move.new == "facility" for move in layout.moves)
    refusal = _refusal(root)
    assert refusal.format_message().splitlines()[-1] == _PLACE_THESE
    assert (outside / "facility/classes.yaml").is_file()


@pytest.mark.parametrize("value", ["facility/content", "data/content", ".", "[a, b]"])
def test_a_data_key_naming_no_valid_directory_goes_to_check_by_hand(
    tmp_path: Path, value: str
) -> None:
    root = _repo(tmp_path)
    _write(root, "profile.yml", f"name: demo\ndata: {value}\n")
    _write(root, f"{value.strip('[]')}/facility/classes.yaml")
    layout = find_retired_root(root)
    assert layout.data_root is None
    assert layout.line_deletes == ()
    assert len(layout.data_keys) == 1
    assert any(item.startswith("profile.yml:2: data: ") for item in layout.check_by_hand)
    assert not layout.empty


def test_two_distinct_data_values_go_to_check_by_hand(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "profile.yml", "name: demo\ndata: content\n")
    _write(root, "profiles/lab.yml", "data: data\n")
    layout = find_retired_root(root)
    assert layout.line_deletes == ()
    assert layout.data_keys == ("profile.yml:2: data: content", "profiles/lab.yml:1: data: data")
    assert layout.data_root is None


def test_an_in_repo_data_root_moves_the_rendered_sources_and_its_line_goes(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    _write(
        root,
        "profile.yml",
        "name: demo\ndata: content\nconfig:\n"
        "  facility_knowledge:\n    bundle_path: content/facility/knowledge\n",
    )
    _write(root, "content/facility/identity.yaml", "code: demo\n")
    _write(root, "content/facility/knowledge/a.md")
    _write(root, "content/landing/index.md")
    _write(root, "data/.runtime/cert.pem")
    layout = find_retired_root(root)
    assert layout.data_root == "content"
    assert [(move.old, move.new) for move in layout.moves] == [
        ("content/facility", "facility"),
        ("content/landing", "web/landing"),
        ("data/.runtime", "var/runtime"),
    ]
    assert [edit.new for edit in layout.line_edits] == ["    bundle_path: facility/knowledge"]
    _run(_refusal(root).format_message(), tmp_path / "elsewhere")
    assert not (root / "content").exists()
    assert not (root / "data").exists()
    assert (root / "web/landing/index.md").is_file()
    assert "data:" not in (root / "profile.yml").read_text(encoding="utf-8")
    assert find_retired_root(root).empty


def test_an_in_repo_data_root_with_a_stray_file_keeps_its_line(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "profile.yml", "name: demo\ndata: content\n")
    _write(root, "content/facility/identity.yaml", "code: demo\n")
    _write(root, "content/stray.txt")
    refusal = _refusal(root)
    layout = refusal.layout
    assert layout.leftovers == ("content/stray.txt",)
    assert layout.line_deletes == ()
    lines = refusal.format_message().splitlines()
    assert "# Still in content/, place by hand:" in lines
    assert lines[-1] == _PLACE_THESE
    _run(refusal.format_message(), tmp_path / "elsewhere")
    assert "data: content" in (root / "profile.yml").read_text(encoding="utf-8")
    assert (root / "facility/identity.yaml").is_file()
    assert not find_retired_root(root).empty


# --- what the rewrite keeps true ------------------------------------------------------------------


def test_a_rewritten_key_resolves_after_the_moves_iff_it_resolved_before(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    values = {
        "code_review.prompt": "data/code_review/reviewer.md",
        "facility_knowledge.bundle_path": "./data/facility/knowledge",
        "services.bluesky.plan_dir": "scripts/plans",
        "services.qmd.models_dir": "data/.runtime/models",
        "web.panels.canopy.theme_file": "data/canopy/theme/missing.css",
    }
    notices = ["data/landing/notice.md", "data/landing/gone.md"]
    _write(
        root,
        "profile.yml",
        "name: demo\nconfig:\n"
        + "".join(f"  {key}: {value}\n" for key, value in values.items())
        + "  modules.web_terminals.landing.notices:\n"
        + "".join(f"    - {notice}\n" for notice in notices),
    )
    _write(root, "data/code_review/reviewer.md")
    _write(root, "data/facility/knowledge/a.md")
    _write(root, "data/.runtime/models/m.bin")
    _write(root, "data/landing/notice.md")
    _write(root, "scripts/run.sh")

    def resolved(document: dict) -> list[bool]:
        config = document["config"]
        keys = [*values, *(f"notice{i}" for i in range(len(notices)))]
        found = [config[key] for key in values] + config["modules.web_terminals.landing.notices"]
        assert len(found) == len(keys)
        return [(root / value).exists() for value in found]

    before = resolved(yaml.safe_load((root / "profile.yml").read_text(encoding="utf-8")))
    assert before == [True, True, False, True, False, True, False]
    _run(_refusal(root).format_message(), tmp_path / "elsewhere")
    after = yaml.safe_load((root / "profile.yml").read_text(encoding="utf-8"))
    assert resolved(after) == before
    assert after["config"]["services.bluesky.plan_dir"] == "ops/scripts/plans"


def _status(root: Path) -> list[str]:
    return sorted(_git(root, "status", "--porcelain", "--untracked-files=all").splitlines())


def test_after_the_block_git_status_matches_revalidate(tmp_path: Path) -> None:
    """The block, closed by ``osprey init --revalidate``, leaves what that command alone leaves."""
    from osprey.cli.init_cmd import init
    from tests._builds import init_project

    template = init_project(tmp_path / "template", "hello-world", "plant")
    trees = {}
    for name in ("block", "revalidate"):
        root = Path(shutil.copytree(template, tmp_path / name / "plant", symlinks=True))
        for move in repo_layout.retired_moves():
            new, old = root / move.new, root / move.old
            if new.exists() and not old.exists():
                old.parent.mkdir(parents=True, exist_ok=True)
                new.rename(old)
                if not any(new.parent.iterdir()):
                    new.parent.rmdir()
        _write(root, "rules/plant.md", "# A rule.\n")
        _write(root, "data/.runtime/certs/key.pem", "key\n")
        _write(root, "data/bluesky_curve/proxy.key_secret", "secret\n")
        _commit(root)
        trees[name] = root
    _run(_refusal(trees["block"]).format_message(), tmp_path / "elsewhere")
    for root in trees.values():
        result = CliRunner().invoke(init, [str(root), "--revalidate"], catch_exceptions=False)
        assert result.exit_code == 0, result.output
    assert _status(trees["block"]) == _status(trees["revalidate"])
    assert any(line.endswith("agent/rules/plant.md") for line in _status(trees["block"]))
    assert (trees["block"] / "var/runtime/bluesky_curve/proxy.key_secret").is_file()


# --- how the command line prints it -------------------------------------------------------


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
