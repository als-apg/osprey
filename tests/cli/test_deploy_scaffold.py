"""The deploy-scaffolding emission engine and the verb that drives it.

The templates are held to the goldens elsewhere
(``tests/deployment/test_deploy_scaffold_goldens.py``); what is tested here is
everything that happens *around* a render — where the two files land, and what
the engine does when one of them is already there.

That second half is the reason the engine exists. An emitted file lands in a
git repository, where it is read, diffed, and sometimes edited, so re-emission
has to be boring: unchanged inputs must produce no diff at all, and a file
somebody wrote by hand must survive a scaffold that did not ask about it.

The fixture repo is built from the same exemplar profile the goldens were
emitted for, so a byte comparison against them doubles as proof that the engine
renders what the golden tests pin, through the code path the CLI actually uses.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import stat
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.deploy_cmd import deploy
from osprey.cli.deploy_scaffold import (
    CI_OUTPUT_NAMES,
    VERIFY_OUTPUT_PATH,
    find_facility_repo_root,
    scaffold_deploy_files,
)
from osprey.cli.deploy_scaffold_templates import CI_MARKER, CI_TEMPLATES, VERIFY_MARKER
from osprey.errors import ConfigurationError

GOLDENS = Path(__file__).parents[1] / "deployment" / "goldens"
EXEMPLAR_DIR = GOLDENS / "exemplar-profile"

#: The goldens freeze the version stamp as this literal, and the repo they were
#: emitted for is named after the project every pipeline path is keyed off.
FROZEN_VERSION = "OSPREY_VERSION"
REPO_NAME = "demo-facility"

CI_PATH = CI_OUTPUT_NAMES["gitlab"]
VERIFY_PATH = "/".join(VERIFY_OUTPUT_PATH)


@pytest.fixture
def facility_repo(tmp_path: Path) -> Path:
    """A facility repo holding the exemplar profile, and nothing else yet."""
    repo = tmp_path / REPO_NAME
    shutil.copytree(EXEMPLAR_DIR, repo / "profile")
    return repo


def scaffold(repo: Path, **kwargs) -> list:
    """Emit into *repo* with the goldens' frozen version stamp."""
    kwargs.setdefault("osprey_version", FROZEN_VERSION)
    return scaffold_deploy_files(repo, **kwargs)


def edit_profile(repo: Path, old: str, new: str) -> None:
    """Change one value in the fixture repo's profile."""
    profile = repo / "profile" / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    assert old in text, f"fixture profile does not contain {old!r}"
    profile.write_text(text.replace(old, new), encoding="utf-8")


# ── Where the files land ─────────────────────────────────────────────────────


def test_emission_lands_both_files_where_the_repo_expects_them(facility_repo: Path) -> None:
    """The pipeline at the repo root, the health check in the profile mirror."""
    emitted = scaffold(facility_repo)

    assert [result.path for result in emitted] == [
        facility_repo / CI_PATH,
        facility_repo / VERIFY_PATH,
    ]
    assert [result.action for result in emitted] == ["created", "created"]
    assert all(result.path.is_file() for result in emitted)


def test_the_mirror_path_is_the_one_the_post_up_hook_reads(facility_repo: Path) -> None:
    """verify.sh goes where a build carries it to ``<project>/scripts/``.

    The ``project/`` mirror copies its tree verbatim onto the project root, so
    emitting to ``profile/project/scripts/verify.sh`` is what puts the script at
    ``<project>/scripts/verify.sh`` — the exact path
    ``osprey.deployment.web_terminals.postup_hooks.run_verify_script`` looks for
    after ``compose up`` succeeds. Emitting anywhere else leaves the auto-run
    silently doing nothing.
    """
    from osprey.deployment.web_terminals import postup_hooks

    scaffold(facility_repo)
    mirror_relative = Path(*VERIFY_OUTPUT_PATH).relative_to(Path("profile") / "project")

    assert mirror_relative == Path("scripts") / "verify.sh"
    source = Path(postup_hooks.__file__).read_text(encoding="utf-8")
    assert '"scripts" / "verify.sh"' in source


def test_the_health_check_is_emitted_executable(facility_repo: Path) -> None:
    """Its own header tells operators to run it as ``./scripts/verify.sh``."""
    scaffold(facility_repo)

    mode = stat.S_IMODE((facility_repo / VERIFY_PATH).stat().st_mode)
    assert mode & stat.S_IXUSR
    assert not stat.S_IMODE((facility_repo / CI_PATH).stat().st_mode) & stat.S_IXUSR


def test_the_project_name_is_the_repo_directory_name(tmp_path: Path) -> None:
    """No profile key names the project — the repo directory does."""
    repo = tmp_path / "other-facility"
    shutil.copytree(EXEMPLAR_DIR, repo / "profile")
    scaffold(repo)

    pipeline = (repo / CI_PATH).read_text(encoding="utf-8")
    assert "OSPREY_PROJECT_NAME: other-facility\n" in pipeline
    assert "OSPREY_BUILD_DIR: build/other-facility\n" in pipeline


# ── What is emitted ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["gitlab-ci.yml", "verify.sh"])
def test_the_engine_emits_the_goldens(facility_repo: Path, name: str) -> None:
    """Emission through the engine reproduces the hand-built reference.

    The golden tests pin the templates against a direct render; this pins the
    engine's own path — profile resolution, deploy-block parse, context build,
    destination — against the same two files.
    """
    scaffold(facility_repo)
    emitted = CI_PATH if name == "gitlab-ci.yml" else VERIFY_PATH

    assert (facility_repo / emitted).read_text(encoding="utf-8") == (GOLDENS / name).read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(("path", "marker"), [(CI_PATH, CI_MARKER), (VERIFY_PATH, VERIFY_MARKER)])
def test_the_provenance_header_is_two_lines_and_names_the_version_once(
    facility_repo: Path, path: str, marker: str
) -> None:
    """Marker plus stamp, each appearing exactly once.

    The marker is the overwrite-refusal key and the stamp is what re-emission
    masks before diffing, so a second copy of either would make one of those
    two mechanisms read the wrong line.
    """
    scaffold(facility_repo)
    text = (facility_repo / path).read_text(encoding="utf-8")

    assert text.count(f"# osprey-scaffold: {marker}\n") == 1
    assert len(re.findall(r"^# osprey-version: ", text, re.MULTILINE)) == 1
    lines = text.splitlines()
    assert lines[lines.index(f"# osprey-scaffold: {marker}") + 1].startswith("# osprey-version: ")


def test_the_stamp_carries_the_installed_version_by_default(facility_repo: Path) -> None:
    """Nothing but the tests freezes it."""
    import osprey

    scaffold_deploy_files(facility_repo)

    for path in (CI_PATH, VERIFY_PATH):
        text = (facility_repo / path).read_text(encoding="utf-8")
        assert f"# osprey-version: {osprey.__version__}\n" in text


# ── Re-emission ──────────────────────────────────────────────────────────────


def test_re_emission_with_unchanged_inputs_writes_nothing(facility_repo: Path) -> None:
    """A second run is a no-op, down to the file's mtime."""
    scaffold(facility_repo)
    before = {
        path: ((facility_repo / path).read_bytes(), (facility_repo / path).stat().st_mtime_ns)
        for path in (CI_PATH, VERIFY_PATH)
    }

    again = scaffold(facility_repo)

    assert [result.action for result in again] == ["unchanged", "unchanged"]
    assert not any(result.written for result in again)
    for path, (content, mtime) in before.items():
        assert (facility_repo / path).read_bytes() == content
        assert (facility_repo / path).stat().st_mtime_ns == mtime


def test_a_version_bump_alone_is_not_a_change(facility_repo: Path) -> None:
    """An OSPREY upgrade that changes no template produces no diff.

    The stamp moves with every release, so comparing bytes would rewrite both
    files on every upgrade and fill the repo's history with diffs that change
    nothing. The stamp left in place still names the release that produced the
    content actually on disk.
    """
    scaffold(facility_repo, osprey_version="2026.1.0")
    original = (facility_repo / CI_PATH).read_text(encoding="utf-8")

    again = scaffold(facility_repo, osprey_version="2099.12.31")

    assert [result.action for result in again] == ["unchanged", "unchanged"]
    assert (facility_repo / CI_PATH).read_text(encoding="utf-8") == original
    assert "# osprey-version: 2026.1.0\n" in original


def test_a_changed_deploy_block_is_re_emitted(facility_repo: Path) -> None:
    """The point of re-running: the profile moved, so the pipeline follows."""
    scaffold(facility_repo)
    edit_profile(facility_repo, "fqdn: demo-deploy.example.org", "fqdn: new-host.example.org")

    again = scaffold(facility_repo)

    assert [result.action for result in again] == ["updated", "unchanged"]
    pipeline = (facility_repo / CI_PATH).read_text(encoding="utf-8")
    assert "new-host.example.org" in pipeline
    assert "demo-deploy.example.org" not in pipeline


def test_an_edit_that_only_the_health_check_reads_re_emits_only_it(facility_repo: Path) -> None:
    """Each file is compared on its own content, not on "the profile changed"."""
    scaffold(facility_repo)
    edit_profile(facility_repo, "nginx_port: 9080", "nginx_port: 9081")

    again = scaffold(facility_repo)

    assert [result.action for result in again] == ["unchanged", "updated"]
    assert "9081" in (facility_repo / VERIFY_PATH).read_text(encoding="utf-8")


# ── Refusing to overwrite what we did not write ──────────────────────────────


@pytest.mark.parametrize(("path", "marker"), [(CI_PATH, CI_MARKER), (VERIFY_PATH, VERIFY_MARKER)])
def test_a_marker_less_file_is_refused_and_left_alone(
    facility_repo: Path, path: str, marker: str
) -> None:
    """A hand-written file at either destination survives a scaffold."""
    target = facility_repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# hand-rolled, do not touch\n", encoding="utf-8")

    emitted = scaffold(facility_repo)
    refused = [result for result in emitted if result.path == target]

    assert [result.action for result in refused] == ["refused"]
    assert marker in refused[0].reason
    assert "--force" in refused[0].reason
    assert target.read_text(encoding="utf-8") == "# hand-rolled, do not touch\n"


def test_stripping_the_marker_from_an_emitted_file_makes_it_hand_written(
    facility_repo: Path,
) -> None:
    """The marker line is the key, not the file's resemblance to a render.

    An operator who deleted the marker was telling the scaffolder the file is
    theirs now — the rest of the header saying otherwise does not override that.
    """
    scaffold(facility_repo)
    target = facility_repo / CI_PATH
    edited = "\n".join(
        line for line in target.read_text(encoding="utf-8").splitlines() if CI_MARKER not in line
    )
    target.write_text(edited + "\n", encoding="utf-8")

    assert scaffold(facility_repo)[0].action == "refused"
    assert target.read_text(encoding="utf-8") == edited + "\n"


def test_the_other_marker_is_not_our_marker(facility_repo: Path) -> None:
    """A file carrying some other generator's marker is foreign too."""
    target = facility_repo / CI_PATH
    target.write_text(
        f"# osprey-scaffold: {VERIFY_MARKER}\n# osprey-version: x\n", encoding="utf-8"
    )

    assert scaffold(facility_repo)[0].action == "refused"


def test_a_marker_quoted_deep_in_a_hand_written_file_is_not_provenance(
    facility_repo: Path,
) -> None:
    """Provenance is a header line, so prose mentioning it does not qualify."""
    target = facility_repo / CI_PATH
    body = "\n".join(f"# line {index}" for index in range(40))
    target.write_text(f"{body}\n# osprey-scaffold: {CI_MARKER}\n", encoding="utf-8")

    assert scaffold(facility_repo)[0].action == "refused"


def test_force_replaces_a_hand_written_file(facility_repo: Path) -> None:
    """--force is how an operator says they meant it."""
    target = facility_repo / CI_PATH
    target.write_text("# hand-rolled\n", encoding="utf-8")

    emitted = scaffold(facility_repo, force=True)

    assert emitted[0].action == "updated"
    assert target.read_text(encoding="utf-8") == (GOLDENS / "gitlab-ci.yml").read_text(
        encoding="utf-8"
    )


def test_a_refusal_does_not_hold_back_the_other_file(facility_repo: Path) -> None:
    """The two files have independent histories.

    A hand-edited pipeline is no reason to leave the health check stale, so the
    refusal is reported per file rather than aborting the emission.
    """
    (facility_repo / CI_PATH).write_text("# hand-rolled\n", encoding="utf-8")

    emitted = scaffold(facility_repo)

    assert [result.action for result in emitted] == ["refused", "created"]
    assert (facility_repo / VERIFY_PATH).is_file()


# ── Inputs the engine refuses outright ───────────────────────────────────────


def test_an_unsupported_ci_platform_names_the_supported_ones(facility_repo: Path) -> None:
    """A platform with no pipeline template has nothing to render."""
    from osprey.cli.build_profile_deploy import SUPPORTED_CI_PLATFORMS

    edit_profile(facility_repo, "ci: gitlab", "ci: jenkins")

    with pytest.raises(ConfigurationError) as excinfo:
        scaffold(facility_repo)

    message = str(excinfo.value)
    assert "jenkins" in message
    for platform in SUPPORTED_CI_PLATFORMS:
        assert repr(platform) in message


def test_a_profile_without_a_deploy_block_is_refused(facility_repo: Path) -> None:
    """Deployment coordinates are opt-in, and there is no default to guess."""
    profile = facility_repo / "profile" / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    profile.write_text(text[: text.index("\ndeploy:\n") + 1] + "data: data\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="no 'deploy:' block"):
        scaffold(facility_repo)

    assert not (facility_repo / CI_PATH).exists()


def test_a_missing_profile_points_at_how_to_get_one(tmp_path: Path) -> None:
    """Nothing is emitted into a directory that is not a facility repo."""
    with pytest.raises(ConfigurationError, match="profile/profile.yml"):
        scaffold(tmp_path)


def test_every_supported_platform_has_an_output_name() -> None:
    """The three tables that make a platform renderable stay in step."""
    from osprey.cli.build_profile_deploy import SUPPORTED_CI_PLATFORMS

    assert set(CI_OUTPUT_NAMES) == set(SUPPORTED_CI_PLATFORMS) == set(CI_TEMPLATES)
    assert CI_OUTPUT_NAMES["gitlab"] == ".gitlab-ci.yml"


# ── Finding the repo ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "start",
    ["", "profile", "profile/personas", "profile/services/facility-mcp"],
    ids=["root", "profile", "personas", "service"],
)
def test_the_repo_is_found_from_anywhere_inside_it(facility_repo: Path, start: str) -> None:
    """Position-independence: the same command anywhere acts on the same repo."""
    assert find_facility_repo_root(facility_repo / start) == facility_repo


def test_a_directory_outside_any_repo_finds_nothing(tmp_path: Path) -> None:
    """No profile above it means no repo to scaffold."""
    outside = tmp_path / "somewhere" / "else"
    outside.mkdir(parents=True)

    assert find_facility_repo_root(outside) is None


# ── The verb ─────────────────────────────────────────────────────────────────


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_the_verb_emits_from_the_working_directory(
    runner: CliRunner, facility_repo: Path, monkeypatch, caplog
) -> None:
    """``osprey deploy scaffold`` run inside a repo needs no arguments."""
    monkeypatch.chdir(facility_repo / "profile")

    with caplog.at_level(logging.INFO, logger="deploy"):
        result = runner.invoke(deploy, ["scaffold"])

    assert result.exit_code == 0, result.output
    assert (facility_repo / CI_PATH).is_file()
    assert (facility_repo / VERIFY_PATH).is_file()
    assert CI_PATH in caplog.text


def test_the_verb_takes_the_repo_explicitly(
    runner: CliRunner, facility_repo: Path, tmp_path: Path, monkeypatch
) -> None:
    """--repo, not --project: this verb acts on a repo, not a built project."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = runner.invoke(deploy, ["scaffold", "--repo", str(facility_repo)])

    assert result.exit_code == 0, result.output
    assert (facility_repo / CI_PATH).is_file()


def test_the_verb_reports_a_no_op_re_run(
    runner: CliRunner, facility_repo: Path, monkeypatch, caplog
) -> None:
    """Re-running says so rather than pretending it wrote something."""
    monkeypatch.chdir(facility_repo)
    runner.invoke(deploy, ["scaffold"])
    caplog.clear()

    with caplog.at_level(logging.INFO, logger="deploy"):
        result = runner.invoke(deploy, ["scaffold"])

    assert result.exit_code == 0, result.output
    assert "Unchanged" in caplog.text


def test_the_verb_exits_nonzero_on_a_refusal(
    runner: CliRunner, facility_repo: Path, monkeypatch, caplog
) -> None:
    """A file left alone is a failure the operator has to act on."""
    monkeypatch.chdir(facility_repo)
    (facility_repo / CI_PATH).write_text("# hand-rolled\n", encoding="utf-8")

    result = runner.invoke(deploy, ["scaffold"])

    assert result.exit_code == 1
    assert "--force" in caplog.text
    assert (facility_repo / CI_PATH).read_text(encoding="utf-8") == "# hand-rolled\n"


def test_the_verb_honors_force(runner: CliRunner, facility_repo: Path, monkeypatch) -> None:
    """The refusal is advice, and --force is how it is overruled."""
    monkeypatch.chdir(facility_repo)
    (facility_repo / CI_PATH).write_text("# hand-rolled\n", encoding="utf-8")

    result = runner.invoke(deploy, ["scaffold", "--force"])

    assert result.exit_code == 0, result.output
    assert "hand-rolled" not in (facility_repo / CI_PATH).read_text(encoding="utf-8")


def test_the_verb_outside_a_repo_explains_itself(
    runner: CliRunner, tmp_path: Path, monkeypatch, caplog
) -> None:
    """Run in the wrong place, it says where it expected to be."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(deploy, ["scaffold"])

    assert result.exit_code == 1
    assert "profile/profile.yml" in caplog.text
    assert not list(tmp_path.glob("*.yml"))


def test_the_verb_reports_a_profile_it_cannot_scaffold_from(
    runner: CliRunner, facility_repo: Path, monkeypatch, caplog
) -> None:
    """A configuration failure is a message, not a traceback."""
    monkeypatch.chdir(facility_repo)
    edit_profile(facility_repo, "ci: gitlab", "ci: jenkins")

    result = runner.invoke(deploy, ["scaffold"])

    assert result.exit_code == 1
    assert "jenkins" in caplog.text
    assert not (facility_repo / CI_PATH).exists()


def test_the_verb_does_not_read_a_config_yml(
    runner: CliRunner, facility_repo: Path, monkeypatch
) -> None:
    """No config.yml exists at a repo root, and none is needed.

    Pinned because every other deploy verb requires one: routing this one
    through the shared project session would make a facility repo unscaffoldable
    until someone put a config.yml at its root.
    """
    monkeypatch.chdir(facility_repo)
    assert not (facility_repo / "config.yml").exists()

    result = runner.invoke(deploy, ["scaffold"])

    assert result.exit_code == 0, result.output
    assert not (facility_repo / "config.yml").exists()


def test_the_verb_leaves_the_working_directory_alone(
    runner: CliRunner, facility_repo: Path, monkeypatch
) -> None:
    """Unlike the project verbs, scaffolding does not chdir anywhere."""
    monkeypatch.chdir(facility_repo)
    before = os.getcwd()

    runner.invoke(deploy, ["scaffold"])

    assert os.getcwd() == before
