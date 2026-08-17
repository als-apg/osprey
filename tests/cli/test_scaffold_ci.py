"""``osprey scaffold`` against a three-zone deployment repo.

The rendered bytes are somebody else's subject: ``test_emitted_artifacts_clean``
holds the templates to the hand-authored exemplar, and
``test_template_cli_weld`` drives the pipeline's argv through the real parser.
What is tested here is the verb and everything around a render — where the two
files land in a repo, how the repo is found, what happens when one of them is
already there, and that the ownership verbs beside it read the build zone of
the same repo.

Three things are worth naming as the reasons this file exists.

*The acceptance.* ``osprey scaffold ci``, run in the exemplar repo, reproduces
both CI files byte for byte. Not the render — the command, through Click, with
the installed version stamped in and the file written to disk.

*The allowlist.* Every ``osprey`` command named in the emitted files, in a
script line or in the header prose, must exist in the CLI this OSPREY ships.
The denylist elsewhere (SC-8's retired-verb sweep) catches a name that used to
exist; this catches one that never did. Both directions matter, and the case
that motivated it — the emitted files advertising ``osprey scaffold ci`` before
the verb existed — was a header comment, which a script-only scan would miss.

*The single destination.* The engine's output paths are not parameters. Three
modules spell the health check's location, and a test that pins them against
each other is the only thing standing between "emitted" and "emitted where the
pipeline and the post-up hook look".
"""

from __future__ import annotations

import inspect
import json
import re
import stat
import subprocess
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from osprey.cli.deploy_scaffold import scaffold_deploy_files
from osprey.cli.deploy_scaffold_templates import CI_MARKER, VERIFY_MARKER
from osprey.cli.main import cli
from osprey.cli.scaffold_cmd import scaffold
from osprey.cli.templates.manifest import MANIFEST_FILENAME
from osprey.errors import ConfigurationError
from tests.fixtures.lifecycle_repo import (
    CI_PIPELINE_FILES,
    build_exemplar_repo,
    exemplar_source_files,
)

#: The two files the verb emits, named by the hand-authored exemplar rather
#: than by the engine's own constants. The exemplar is the specification, so a
#: destination the engine moved on its own has to read here as a file that was
#: not written — not as a test that quietly followed it.
EMITTED = tuple(CI_PIPELINE_FILES)
CI_PATH, VERIFY_PATH = EMITTED


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """The exemplar repo with deployment coordinates and no CI files yet.

    Materialized ``with_ci=True`` and then stripped of the pair, rather than
    materialized without it: ``with_ci=False`` also comments out the ``deploy:``
    block, which is a repo with nothing to render *from*. What is wanted here is
    a repo with coordinates whose files have not been emitted yet — the state an
    operator is in the moment they fill the block in.
    """
    root = build_exemplar_repo(tmp_path / "als-exemplar", with_ci=True)
    for relative in EMITTED:
        (root / relative).unlink()
    return root


@pytest.fixture
def specification() -> dict[str, str]:
    """The hand-authored exemplar's CI pair, sentinels expanded."""
    files = exemplar_source_files(with_ci=True)
    return {relative: files[relative] for relative in EMITTED}


def emit(runner: CliRunner, repo: Path, *args: str):
    """Run ``osprey scaffold ci`` against *repo* from outside it."""
    return runner.invoke(scaffold, ["ci", "--repo", str(repo), *args])


def subprocess_result() -> subprocess.CompletedProcess[bytes]:
    """A successful run, for a hook whose subprocess is intercepted."""
    return subprocess.CompletedProcess(args=[], returncode=0)


def edit_profile(repo: Path, old: str, new: str) -> None:
    """Change one value in the repo's profile."""
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    assert old in text, f"the exemplar profile does not contain {old!r}"
    profile.write_text(text.replace(old, new), encoding="utf-8")


# ── The acceptance ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("relative", EMITTED)
def test_the_verb_reproduces_the_exemplar_byte_for_byte(
    runner: CliRunner, repo: Path, specification: dict[str, str], relative: str
) -> None:
    """The command an operator runs produces the specification exactly.

    The render is pinned to the same specification in
    ``test_emitted_artifacts_clean``; what this adds is the whole path from
    ``osprey scaffold ci`` to bytes on disk — repo discovery, profile
    resolution, the ``extends:`` merge, the context build, the version stamp and
    the atomic write. A drift anywhere along it shows up here and nowhere else.
    """
    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    assert (repo / relative).read_text(encoding="utf-8") == specification[relative]


def test_both_files_are_reported_as_created(runner: CliRunner, repo: Path) -> None:
    """The summary names each file and what happened to it."""
    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    for relative in EMITTED:
        assert relative in result.output
    assert result.output.count("created") == 2


def test_nothing_else_in_the_repo_is_touched(runner: CliRunner, repo: Path) -> None:
    """Emission writes two files and reads the rest.

    ``ci-extra.yml`` is the one to watch: the pipeline includes it, an operator
    owns it, and it sits at the repo root beside the file that *is* regenerated.
    Comparing the whole source zone rather than that one file keeps the property
    general — no emission may quietly rewrite the profile it rendered from
    either.
    """
    before = {
        path.relative_to(repo).as_posix(): path.read_bytes()
        for path in sorted(repo.rglob("*"))
        if path.is_file()
    }

    assert emit(runner, repo).exit_code == 0

    after = {
        path.relative_to(repo).as_posix(): path.read_bytes()
        for path in sorted(repo.rglob("*"))
        if path.is_file()
    }
    assert set(after) - set(before) == set(EMITTED)
    assert {name: text for name, text in after.items() if name in before} == before


def test_the_health_check_is_emitted_executable(runner: CliRunner, repo: Path) -> None:
    """Its own header tells an operator to run it as ``./scripts/verify.sh``."""
    assert emit(runner, repo).exit_code == 0

    assert stat.S_IMODE((repo / VERIFY_PATH).stat().st_mode) & stat.S_IXUSR
    assert not stat.S_IMODE((repo / CI_PATH).stat().st_mode) & stat.S_IXUSR


# ── Finding the repo ─────────────────────────────────────────────────────────


def test_the_verb_needs_no_arguments_inside_the_repo(
    runner: CliRunner, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run from a nested subdirectory, it emits into the repo above it.

    Position-independence is the resolver's contract, and it is what lets the
    emitted pipeline run the same commands a laptop does.
    """
    monkeypatch.chdir(repo / "data" / "facility_knowledge")

    result = runner.invoke(scaffold, ["ci"])

    assert result.exit_code == 0, result.output
    assert all((repo / relative).is_file() for relative in EMITTED)


def test_no_emitted_path_is_keyed_off_the_repo_directory_name(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Two checkouts of one deployment emit the same two files.

    The retired layout keyed paths off the repo's directory name — the profile
    under ``profile/``, the build under ``build/<name>/`` — so a second checkout
    at another path produced a different pipeline. Every path is repo-relative
    now, and the directory name reaches the render only as a title fallback the
    exemplar's own ``name:`` supplies instead.
    """
    here = build_exemplar_repo(tmp_path / "one-checkout", with_ci=True)
    there = build_exemplar_repo(tmp_path / "another-checkout", with_ci=True)
    for root in (here, there):
        for relative in EMITTED:
            (root / relative).unlink()

    assert emit(runner, here).exit_code == 0
    assert emit(runner, there).exit_code == 0

    for relative in EMITTED:
        emitted = (here / relative).read_text(encoding="utf-8")
        assert emitted == (there / relative).read_text(encoding="utf-8")
        assert "one-checkout" not in emitted


def test_the_verb_needs_no_config_yml_and_leaves_the_cwd_alone(
    runner: CliRunner, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two properties that separate this verb from the deploy verbs.

    It renders from the profile, so a repo that has never been built — no
    ``build/``, no ``config.yml`` anywhere — scaffolds fine; routing it through
    a project session instead would make a fresh repo unscaffoldable until
    somebody built it. And it does not chdir: the verbs that run compose do,
    and a scaffold that inherited the habit would leave an operator's shell
    somewhere they did not ask to be.
    """
    import os

    monkeypatch.chdir(repo)
    before = os.getcwd()
    assert not (repo / "build").exists()

    result = runner.invoke(scaffold, ["ci"])

    assert result.exit_code == 0, result.output
    assert os.getcwd() == before
    assert not (repo / "build").exists()


def test_outside_any_repo_the_failure_is_about_the_repo(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is emitted, and the message is the shared one."""
    elsewhere = tmp_path / "not-a-repo"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = runner.invoke(scaffold, ["ci"])

    assert result.exit_code == 1
    assert "No OSPREY deployment repo found" in result.output
    assert not any(elsewhere.iterdir())


def test_a_profile_without_coordinates_is_refused_cleanly(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A commented-out ``deploy:`` block is the fresh-repo state, not a crash.

    ``osprey init`` says so and moves on; this verb's entire job is the two
    files, so it reports the missing block as the failure it is — with an exit
    code and a message, not a traceback.
    """
    bare = build_exemplar_repo(tmp_path / "no-coordinates", with_ci=False)

    result = emit(runner, bare)

    assert result.exit_code == 1
    assert "no 'deploy:' block" in result.output
    assert not (bare / CI_PATH).exists()


def test_the_refusal_points_at_the_commented_block_the_profile_already_carries(
    tmp_path: Path,
) -> None:
    """ "Add one naming the CI platform" reads as authoring a block from scratch.

    Every emitted profile already ends with a commented ``deploy:`` example, so
    the operator's work is uncommenting and filling in — a smaller, different
    task than the one the old wording described, and one they can only find if
    the refusal says where to look.
    """
    bare = build_exemplar_repo(tmp_path / "no-coordinates", with_ci=False)

    with pytest.raises(ConfigurationError) as caught:
        scaffold_deploy_files(bare)

    message = " ".join(str(caught.value).split())
    assert "commented 'deploy:' example" in message
    assert "at the end of the file" in message
    assert "uncomment it" in message
    assert "the CI platform, the registry and the deploy host" in message


def test_the_engines_missing_profile_message_names_this_verb(tmp_path: Path) -> None:
    """The one refusal the resolver cannot pre-empt still points somewhere.

    Reached through the CLI only in a race — the resolver found a ``profile.yml``
    and it was gone a moment later — so it is exercised directly. It is still
    the message an operator would read, and it has to name a verb that exists.
    """
    with pytest.raises(ConfigurationError) as caught:
        scaffold_deploy_files(tmp_path)

    message = str(caught.value)
    assert "osprey scaffold ci" in message
    assert "osprey init" in message


# ── Re-emission ──────────────────────────────────────────────────────────────


def test_re_emission_with_unchanged_inputs_writes_nothing(runner: CliRunner, repo: Path) -> None:
    """A second run is a no-op, down to the file's mtime."""
    assert emit(runner, repo).exit_code == 0
    before = {
        relative: ((repo / relative).read_bytes(), (repo / relative).stat().st_mtime_ns)
        for relative in EMITTED
    }

    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    assert result.output.count("unchanged") == 2
    for relative, (content, mtime) in before.items():
        assert (repo / relative).read_bytes() == content
        assert (repo / relative).stat().st_mtime_ns == mtime


def test_a_version_bump_alone_is_not_a_change(repo: Path) -> None:
    """An OSPREY upgrade that changes no template produces no diff.

    Driven through the engine rather than the verb, because the verb has no way
    to pass a version — which is the point: the stamp moves with every release,
    so a byte comparison would rewrite both files on every upgrade and fill the
    repo's history with diffs that change nothing.
    """
    scaffold_deploy_files(repo, osprey_version="2026.1.0")
    original = (repo / CI_PATH).read_text(encoding="utf-8")

    again = scaffold_deploy_files(repo, osprey_version="2099.12.31")

    assert [result.action for result in again] == ["unchanged", "unchanged"]
    assert (repo / CI_PATH).read_text(encoding="utf-8") == original
    assert "# osprey-version: 2026.1.0\n" in original


def test_a_changed_deploy_block_re_emits_the_pipeline(runner: CliRunner, repo: Path) -> None:
    """The point of re-running: the profile moved, so the pipeline follows."""
    assert emit(runner, repo).exit_code == 0
    edit_profile(repo, "fqdn: appsdev2.example.org", "fqdn: new-host.example.org")

    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    pipeline = (repo / CI_PATH).read_text(encoding="utf-8")
    assert "new-host.example.org" in pipeline
    assert "appsdev2.example.org" not in pipeline


def test_an_edit_only_the_health_check_reads_re_emits_only_it(
    runner: CliRunner, repo: Path
) -> None:
    """Each file is compared on its own content, not on "the profile changed"."""
    assert emit(runner, repo).exit_code == 0
    pipeline_before = (repo / CI_PATH).read_bytes()
    edit_profile(repo, "nginx_port: 9080", "nginx_port: 9081")

    assert emit(runner, repo).exit_code == 0

    assert "9081" in (repo / VERIFY_PATH).read_text(encoding="utf-8")
    assert (repo / CI_PATH).read_bytes() == pipeline_before


# ── Refusing to overwrite what we did not write ──────────────────────────────


@pytest.mark.parametrize(
    ("relative", "marker"), [(CI_PATH, CI_MARKER), (VERIFY_PATH, VERIFY_MARKER)]
)
def test_a_marker_less_file_is_refused_and_names_this_verbs_force(
    runner: CliRunner, repo: Path, relative: str, marker: str
) -> None:
    """A hand-written file survives, and the remedy names the flag that works.

    ``osprey init --force`` deliberately cannot regenerate a CI file, so a
    refusal that said only "--force" would send an operator who just ran it
    round a loop with no exit. The reason names the whole command.
    """
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# hand-rolled, do not touch\n", encoding="utf-8")

    result = emit(runner, repo)

    assert result.exit_code == 1
    assert marker in result.output
    assert "osprey scaffold ci --force" in result.output
    assert target.read_text(encoding="utf-8") == "# hand-rolled, do not touch\n"


def test_a_refusal_does_not_hold_back_the_other_file(runner: CliRunner, repo: Path) -> None:
    """The two files have independent histories.

    A hand-edited pipeline is no reason to leave the health check stale, so the
    refusal is per file — reported, and non-zero, but not an abort.
    """
    (repo / CI_PATH).write_text("# hand-rolled\n", encoding="utf-8")

    result = emit(runner, repo)

    assert result.exit_code == 1
    assert (repo / VERIFY_PATH).is_file()


def test_force_replaces_a_hand_written_file(
    runner: CliRunner, repo: Path, specification: dict[str, str]
) -> None:
    """``--force`` is how an operator says they meant it.

    The replacement is the full emission, not a patch over what was there.
    """
    (repo / CI_PATH).write_text("# hand-rolled\n", encoding="utf-8")

    result = emit(runner, repo, "--force")

    assert result.exit_code == 0, result.output
    assert (repo / CI_PATH).read_text(encoding="utf-8") == specification[CI_PATH]


def test_stripping_the_marker_makes_an_emitted_file_hand_written(
    runner: CliRunner, repo: Path
) -> None:
    """The marker line is the key, not the file's resemblance to a render.

    An operator who deleted the marker was telling the scaffolder the file is
    theirs now — the rest of the header saying otherwise does not override that.
    """
    assert emit(runner, repo).exit_code == 0
    target = repo / CI_PATH
    edited = "\n".join(
        line for line in target.read_text(encoding="utf-8").splitlines() if CI_MARKER not in line
    )
    target.write_text(edited + "\n", encoding="utf-8")

    assert emit(runner, repo).exit_code == 1
    assert target.read_text(encoding="utf-8") == edited + "\n"


@pytest.mark.parametrize(
    ("header", "case"),
    [
        (f"# osprey-scaffold: {VERIFY_MARKER}\n# osprey-version: x\n", "another generator"),
        (
            "\n".join(f"# line {index}" for index in range(40))
            + f"\n# osprey-scaffold: {CI_MARKER}\n",
            "quoted in prose",
        ),
    ],
    ids=["another-marker", "marker-quoted-deep"],
)
def test_what_does_not_count_as_our_provenance(
    runner: CliRunner, repo: Path, header: str, case: str
) -> None:
    """Provenance is OUR marker, in the header.

    Some other generator's marker is not ours, and our own marker mentioned
    forty lines down — a README pasted into a comment — is prose, not a claim
    about who wrote the file. Reading either as provenance would overwrite
    somebody's file.
    """
    target = repo / CI_PATH
    target.write_text(header, encoding="utf-8")

    assert emit(runner, repo).exit_code == 1, case
    assert target.read_text(encoding="utf-8") == header


# ── Inputs the engine refuses outright ───────────────────────────────────────


def test_an_unsupported_ci_platform_names_the_supported_ones(runner: CliRunner, repo: Path) -> None:
    """A platform with no pipeline template has nothing to render."""
    from osprey.cli.build_profile_deploy import SUPPORTED_CI_PLATFORMS

    edit_profile(repo, "ci: gitlab", "ci: jenkins")

    result = emit(runner, repo)

    assert result.exit_code == 1
    assert "jenkins" in result.output
    for platform in SUPPORTED_CI_PLATFORMS:
        assert repr(platform) in result.output


def test_every_supported_platform_has_an_output_name() -> None:
    """The three tables that make a platform renderable stay in step."""
    from osprey.cli.build_profile_deploy import SUPPORTED_CI_PLATFORMS
    from osprey.cli.deploy_scaffold import CI_OUTPUT_NAMES
    from osprey.cli.deploy_scaffold_templates import CI_TEMPLATES

    assert set(CI_OUTPUT_NAMES) == set(SUPPORTED_CI_PLATFORMS) == set(CI_TEMPLATES)
    assert CI_OUTPUT_NAMES["gitlab"] == CI_PATH


def test_the_stamp_carries_the_installed_version(runner: CliRunner, repo: Path) -> None:
    """Nothing but a test freezes it: the verb takes no version at all."""
    import osprey

    assert emit(runner, repo).exit_code == 0

    for relative in EMITTED:
        text = (repo / relative).read_text(encoding="utf-8")
        assert f"# osprey-version: {osprey.__version__}\n" in text


# ── One destination, spelled once ────────────────────────────────────────────


def test_the_health_check_lands_where_every_reader_looks(
    runner: CliRunner, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The emitted file is at the path its three readers name.

    Three modules besides the engine name the health check's location:
    ``deploy_scaffold_templates`` prints it into the rendered pipeline's prose,
    ``init_cmd`` lists it among the files a re-materialization preserves, and
    ``postup_hooks`` runs it at the end of a deploy that provisions the web
    tier. Each would keep passing its own tests while the file moved out from
    under it, and the deployment would simply stop being verified — silently,
    because the hook skips a missing script without a log line. So each is
    asked about the file that was actually written, rather than about a string.
    """
    from osprey.cli.deploy_scaffold import VERIFY_OUTPUT_PATH
    from osprey.cli.deploy_scaffold_templates import VERIFY_PATH as RENDERED_VERIFY_PATH
    from osprey.cli.init_cmd import REPO_VERIFY_PATH
    from osprey.deployment.web_terminals import postup_hooks

    assert emit(runner, repo).exit_code == 0
    emitted = repo / VERIFY_PATH
    assert repo.joinpath(*VERIFY_OUTPUT_PATH) == emitted

    # What the pipeline tells an operator to run, and where init promises the
    # file is safe from a re-materialization.
    assert (repo / RENDERED_VERIFY_PATH) == emitted
    assert RENDERED_VERIFY_PATH in (repo / CI_PATH).read_text(encoding="utf-8")
    assert repo.joinpath(*REPO_VERIFY_PATH) == emitted

    # And what `osprey up` runs afterwards. Intercepted rather than executed:
    # the emitted script probes live services, and what is under test is which
    # path the hook reached for.
    ran: list[list[str]] = []
    monkeypatch.setattr(
        postup_hooks.subprocess,
        "run",
        lambda argv, **kwargs: ran.append(argv) or subprocess_result(),
    )
    postup_hooks.run_verify_script(str(repo), {})

    assert ran == [["bash", str(emitted)]]


def test_the_destinations_are_not_caller_choosable() -> None:
    """The engine takes a repo, not a pair of output paths.

    Both destinations were parameters while two layouts existed, and the
    three-zone one was reached by passing them in. Keeping the parameters after
    the second layout went away would leave every caller free to emit a
    pipeline that names ``scripts/verify.sh`` beside a health check written
    somewhere else — a mismatch nothing downstream can detect.
    """
    parameters = set(inspect.signature(scaffold_deploy_files).parameters)

    assert parameters == {"repo_root", "force", "osprey_version"}


# ── The allowlist: every command the emitted files name exists ───────────────

#: An ``osprey`` invocation or prose mention, with the verb chain that follows
#: it. Bounded to spaces and tabs so a mention does not run across a line, and
#: to bare words so it stops at the first flag, quote or backtick — the chain is
#: what has to resolve, and its arguments are ``test_template_cli_weld``'s
#: subject.
OSPREY_MENTION_RE = re.compile(r"\bosprey((?:[ \t]+[a-z][a-z0-9-]*)+)")

#: Every command chain the two emitted files name, script lines and header prose
#: alike. Pinned as a set rather than derived: this is the surface a deployment
#: repo's generated files depend on, and a template that starts naming a new
#: verb should have to say so here.
NAMED_COMMANDS = {
    ("build",),
    ("scaffold", "ci"),
    ("up",),
    ("users", "env"),
    ("validate",),
}


def named_commands(text: str) -> set[tuple[str, ...]]:
    """Every ``osprey`` command chain *text* names."""
    return {tuple(match.group(1).split()) for match in OSPREY_MENTION_RE.finditer(text)}


def unresolvable(chain: tuple[str, ...]) -> str | None:
    """Why *chain* does not resolve against the live CLI, or ``None``.

    Walks the real command tree the way an operator's shell would: each name is
    looked up on the group above it, and the walk stops at the first leaf —
    anything after a leaf command is an argument or, in prose, an ordinary
    English word.
    """
    context = click.Context(cli)
    command: click.Command = cli
    for index, name in enumerate(chain):
        if not isinstance(command, click.Group):
            return None
        resolved = command.get_command(context, name)
        if resolved is None:
            return f"osprey {' '.join(chain[: index + 1])} does not exist"
        command = resolved
    return None


@pytest.fixture
def emitted_text(repo: Path) -> dict[str, str]:
    """Both emitted files as text, keyed by repo-relative path.

    Emitted through the engine rather than the verb, deliberately: what the
    files name and what the CLI offers are two independent facts, and a check
    that reached the bytes *through* one of the commands under test would turn
    a missing verb into a fixture error instead of the assertion it is.
    """
    scaffold_deploy_files(repo)
    return {relative: (repo / relative).read_text(encoding="utf-8") for relative in EMITTED}


def test_the_emitted_files_name_exactly_these_commands(emitted_text: dict[str, str]) -> None:
    """The extraction finds the expected set — and finds something at all.

    Without this the allowlist below could pass while reading nothing, which is
    the failure mode a pattern-driven check is most prone to. ``scaffold ci`` is
    the case worth naming: both files advertise it in their header, so a
    script-only extraction would drop it — and it is precisely the command that
    was advertised before it existed.
    """
    found = set().union(*(named_commands(text) for text in emitted_text.values()))

    assert found == NAMED_COMMANDS
    for text in emitted_text.values():
        assert ("scaffold", "ci") in named_commands(text)


def test_every_command_the_emitted_files_name_exists(emitted_text: dict[str, str]) -> None:
    """The forward direction of SC-8.

    The retired-verb sweep catches a name that used to exist. Nothing catches a
    name that never did — a pipeline can be emitted, committed, and reviewed
    against a hand-written specification that named a verb someone intended to
    add. It fails on the deploy host, in front of an operator, at the one moment
    that matters.
    """
    for path, text in emitted_text.items():
        for chain in sorted(named_commands(text)):
            problem = unresolvable(chain)
            assert problem is None, f"{path}: {problem}"


def test_the_allowlist_can_fail() -> None:
    """A fabricated verb is reported, at the right depth.

    Both cases matter: a top-level name nothing defines, and a real group with a
    subcommand it does not have — the second is the shape the emitted files
    actually risk, and a checker that stopped at the group would call it fine.
    """
    assert unresolvable(("nonesuch",)) == "osprey nonesuch does not exist"
    assert unresolvable(("scaffold", "nonesuch")) == "osprey scaffold nonesuch does not exist"
    assert unresolvable(("scaffold", "ci")) is None


# ── The ownership verbs, on the same repo ────────────────────────────────────


@pytest.fixture
def built_repo(repo: Path) -> Path:
    """The exemplar repo with a build zone, as a build would leave it.

    Stands in for a real ``osprey build``, which would install a venv to answer
    a question about ownership. What the ownership verbs read from the build
    zone is its ``config.yml`` and its manifest, so those are what this writes —
    with the manifest naming the repo's own profile, which is what a build in
    this repo records.
    """
    build = repo / "build"
    build.mkdir()
    (build / "config.yml").write_text("project_name: als-exemplar\n", encoding="utf-8")
    (build / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "build_args": {"profile_path_abs": str(repo / "profile.yml")},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return repo


def test_list_reads_the_build_zone_of_the_repo_it_is_standing_in(
    runner: CliRunner, built_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``--project``: the repo is found, and its build zone is where to look."""
    monkeypatch.chdir(built_repo / "personas")

    result = runner.invoke(scaffold, ["list"])

    assert result.exit_code == 0, result.output
    assert "Build Artifacts" in result.output
    assert str(built_repo) in result.output.replace("\n", "")


def test_a_claim_moves_the_artifact_from_the_build_zone_into_the_source_zone(
    runner: CliRunner, built_repo: Path
) -> None:
    """The move this verb exists for, in the layout it runs in.

    The build zone is disposable — the next build replaces it wholesale — so an
    artifact an operator wants to keep has to leave it. It lands beside the
    profile, in the tracked source zone, which is where the next build reads its
    convention directories from.
    """
    rendered = built_repo / "build" / ".claude" / "rules" / "safety.md"
    rendered.parent.mkdir(parents=True)
    rendered.write_text("# my safety rules\n", encoding="utf-8")

    result = runner.invoke(scaffold, ["claim", "rules/safety", "--repo", str(built_repo)])

    assert result.exit_code == 0, result.output
    claimed = built_repo / "rules" / "safety.md"
    assert claimed.read_text(encoding="utf-8") == "# my safety rules\n"
    assert not rendered.exists()
    assert "osprey build" in result.output


def test_a_claim_needs_no_arguments_inside_the_repo(
    runner: CliRunner, built_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same claim, run from a subdirectory, acts on the same repo."""
    rendered = built_repo / "build" / ".claude" / "rules" / "safety.md"
    rendered.parent.mkdir(parents=True)
    rendered.write_text("# my safety rules\n", encoding="utf-8")
    monkeypatch.chdir(built_repo / "data")

    result = runner.invoke(scaffold, ["claim", "rules/safety"])

    assert result.exit_code == 0, result.output
    assert (built_repo / "rules" / "safety.md").is_file()


@pytest.mark.parametrize(
    "argv",
    [
        ["diff", "rules/safety"],
        ["unclaim", "rules/safety"],
        ["web-terminals", "lint"],
    ],
    ids=["diff", "unclaim", "web-terminals-lint"],
)
def test_a_repo_with_no_build_zone_says_so(runner: CliRunner, repo: Path, argv: list[str]) -> None:
    """Every verb that needs the build says the same thing about not having it.

    The build zone is git-ignored and disposable, so a fresh clone has none.
    That is an ordinary state with one remedy, and naming the remedy is the
    whole job — the old message asked whether the operator was in a project
    directory, which in a one-repo layout is a question with no useful answer.
    """
    result = runner.invoke(scaffold, [*argv, "--repo", str(repo)])

    assert result.exit_code != 0
    assert "No build at" in result.output
    assert "osprey build" in result.output


def test_list_still_works_before_the_first_build(runner: CliRunner, repo: Path) -> None:
    """Listing what the framework manages needs no build to have happened."""
    result = runner.invoke(scaffold, ["list", "--repo", str(repo)])

    assert result.exit_code == 0, result.output
    assert "Framework-managed" in result.output
