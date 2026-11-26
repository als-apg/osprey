"""A deployment on a retired layout reaches the fresh-init layout with one ``--revalidate``.

The main tree is the previous release's repo (:func:`_previous_release_repo`:
its root entries at their retired paths and its facility tree at
``data/facility`` with the old facility slots), extended here with what a
deployment in service also holds and the shared fixture does not: a generated
CI pair, a generated systemd unit and boot hook, an operator script and
service, a ``.dockerignore`` line and a profile value naming retired paths,
certificates under ``data/.runtime`` and canopy staging left at the root. It is
committed to its own git repository.

Three trees each reach the layout a fresh ``osprey init`` of their preset
writes, with ``data/`` dissolved, in one ``osprey init --revalidate``; each then
builds, and a second ``--revalidate`` changes nothing. They are the previous
release; the previous release with the printed block pasted only in part
(stopped midway, or without its closing command); and the previous release
with its rendered sources under a non-default in-repo ``data:`` root and its
certificates under both ``data/.runtime`` and ``data/<lane>_curve``.
Certificates arrive byte-identical under ``var/runtime/<lane>_curve``. Every
rewritten path-valued profile key resolves after the migration exactly when it
resolved before; a ``./data/facility/...`` value becomes ``./facility/...``,
while absolute and ``{project_root}/`` values are listed for the operator and
left as they are.

Every command that reads the source zone refuses it before writing anything.
What the migration cannot finish stays refused and is listed: a ``data:``
outside the repo (nothing of it moves), a clash (the tree is left unchanged)
and a file left in ``data/``. A merge into a successor that already holds other
files keeps them all. ``osprey init --force`` over the migrated tree replaces
only the entries it ships and keeps every other file under ``agent/`` and
``ops/``. A deployment mounting a machine repository that is a submodule on the
previous layout, its facility at ``data/facility``, builds.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey import repo_layout
from osprey.bluesky_bridge_connection import LANE_KEYS
from osprey.cli.deploy_scaffold import scaffold_systemd_unit
from osprey.cli.init_cmd import init
from osprey.cli.main import cli
from osprey.cli.profile_cmd import MATERIALIZED_SOURCE_ENTRIES
from osprey.cli.repo_resolver import HELD_SOURCE_ZONE_DIRNAME
from osprey.repo_retired import RetiredRootRefused, find_retired_root, refuse_retired_layout
from osprey.utils.workspace import IMAGE_DIR_NAME
from osprey_facility.errors import _PLACE_THESE, PASTE_HEADER, _revalidate_command
from osprey_facility.retired import find_retired
from tests._builds import run_build
from tests.cli.test_init_revalidate import _MIGRATED, _previous_release_repo
from tests.cli.test_root_retired_refusal import REFUSING
from tests.fixtures.lifecycle_repo import exemplar_source_files
from tests.fixtures.machine_repository import commit_all, git, git_init, make_machine_repository

pytestmark = [pytest.mark.slow]

#: The repo's directory name, as the previous release's fixture names it.
PROJECT = "als-exemplar"

#: The facility tree, relative to the deployment, and where the previous release kept it.
FACILITY = repo_layout.entry("facility").path
RETIRED_FACILITY = "data/facility"  # retired-spelling

#: The directory the migration dissolves.
DISSOLVED = repo_layout.dissolved_names()[0]

#: Where the runtime state lives, and where the previous release kept it.
RUNTIME = "var/runtime"
RETIRED_RUNTIME = "data/.runtime"  # retired-spelling

#: The generated CI pipeline and its health check, at their retired paths.
CI_FILES = {".gitlab-ci.yml": ".gitlab-ci.yml", "ops/scripts/verify.sh": "scripts/verify.sh"}

#: The generated pipeline names the current paths; the previous release's
#: pipeline named the retired ones.
PIPELINE_SPELLINGS = (
    ("ops/ci-extra.yml", "ci-extra.yml"),
    ("ops/scripts/verify.sh", "scripts/verify.sh"),
)

#: The pipeline's include, before and after the migration.
RETIRED_INCLUDE = "- local: ci-extra.yml"
MIGRATED_INCLUDE = "- local: ops/ci-extra.yml"

#: The generated unit and boot hook: where the scaffolder writes them now, and
#: where the previous release kept them.
SCAFFOLDED = {
    "ops/osprey.service": "osprey.service",
    "ops/scripts/osprey-boot-hook.sh": "scripts/osprey-boot-hook.sh",
}

#: Operator files at retired paths, with no counterpart in the fixture.
OPERATOR_FILES = {
    "scripts/private/token.txt": "not for the image\n",
    "services/worker/Dockerfile": "FROM python:3.12\n",
    "data/code_review/reviewer.md": "# The reviewer's prompt.\n",
}

#: A certificate set as a lane minted it under the data tree.
CERTIFICATES = (
    "data/.runtime/bluesky_curve/bridge/proxy.key_secret",
    "data/.runtime/bluesky_curve/queueserver/publisher.key",
)

#: A second lane, whose certificates sat directly under ``data/<lane>_curve``.
SECOND_LANE = LANE_KEYS[1]
LANE_CERTIFICATES = (
    f"data/{SECOND_LANE}_curve/bridge/proxy.key_secret",
    f"data/{SECOND_LANE}_curve/queueserver/publisher.key",
)

#: Canopy staging a previous build left at the root.
CANOPY = (
    "canopy_payload/index.json",
    "canopy-1.0-py3-none-any.whl",
    "pydmconverter-1.0-py3-none-any.whl",
)

#: A ``.dockerignore`` naming a path under a moved entry, before and after.
DOCKERIGNORE = "scripts/private/\n"
MIGRATED_DOCKERIGNORE = "ops/scripts/private/\n"

#: A profile value naming a retired path, added under ``config:``, and the
#: value the migration writes.
PROFILE_VALUE = "  code_review:\n    prompt: ./data/code_review/reviewer.md\n"
MIGRATED_PROFILE_VALUE = "    prompt: ./agent/code_review/reviewer.md\n"

#: The previous release's profile names a CANOPY home screen, and its repo
#: holds no application and so no screen library: no build of it passes with
#: the key, migrated or not.
HOME_SCREEN = "  web.panels.canopy.home_screen: facility/overview\n"

#: The first line of the refusal: the block a user pastes starts here.
REFUSAL_OPENS = "# The deployment repo at "

#: The non-default in-repo data root, and the rendered sources it holds.
DATA_ROOT = "content"
RENDERED_SOURCES = ("facility", "canopy", "landing")

#: Profile values naming a facility path: one the migration rewrites, and the
#: absolute and anchored ones it lists and leaves alone (``{repo}`` is the
#: repo's absolute path).
DOT_SLASH_KEY = "facility_knowledge.bundle_path"
DOT_SLASH_VALUE = "./data/facility/knowledge"  # retired-spelling
MIGRATED_DOT_SLASH_VALUE = f"./{FACILITY}/knowledge"
ABSOLUTE_KEY = "code_review.prompt"
ABSOLUTE_VALUE = "{repo}/data/facility/knowledge/index.md"  # retired-spelling
ANCHORED_KEY = "services.qmd.models_dir"
ANCHORED_VALUE = "{project_root}/data/.runtime/models"  # retired-spelling

#: What a fresh init of the preset writes at the top, beside which the
#: deployment's sources hold only the ``.dockerignore`` its operator wrote.
FRESH_TOP = frozenset(rel.split("/")[0] for rel in exemplar_source_files(with_ci=True))
OPERATOR_TOP = frozenset({".dockerignore"})

#: The files a fresh init of the preset writes that the previous release also
#: held; the rest it wrote differently, and a migration keeps its own.
FRESH_FILES = tuple(sorted(rel for rel in exemplar_source_files() if not rel.startswith(_MIGRATED)))

#: Where the machine submodule is mounted in its deployment.
SUBMODULE_MOUNT = "vendor/example-machine"

#: What a build writes beside its own render that a migration never touches,
#: and git's own directory.
NOT_SOURCES = (".git", "build", "var")

#: Tracked moves the migration stages as renames: retired path, current path.
RENAMES = (
    ("ci-extra.yml", "ops/ci-extra.yml"),
    ("osprey.service", "ops/osprey.service"),
    ("scripts/osprey-boot-hook.sh", "ops/scripts/osprey-boot-hook.sh"),
    ("services/worker/Dockerfile", "ops/services/worker/Dockerfile"),
    ("data/code_review/reviewer.md", "agent/code_review/reviewer.md"),
)

#: What both routes leave for the operator: the host steps and the files whose
#: inner paths nothing rewrites.
HOST_STEPS = (
    "# Host steps, once the files moved:",
    "osprey build",
    "systemctl --user daemon-reload",
    "systemctl --user enable --now osprey.service",
    "crontab -e, replacing the osprey-boot-hook lines with:",
    "# Check by hand; the paths inside these are not rewritten:",
    "ops/ci-extra.yml",
    "ops/services/worker/Dockerfile",
    "ops/scripts/ (operator scripts)",
)

#: The entries ``init --force`` replaces under the two groups, by design.
REPLACED_GROUP_ENTRIES = ("agent/context", "agent/watchdogs", "agent/code_review")

#: A git identity and no developer config (signing, templates, hooks).
GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_AUTHOR_NAME": "OSPREY",
    "GIT_COMMITTER_NAME": "OSPREY",
    "GIT_AUTHOR_EMAIL": "osprey@example.org",
    "GIT_COMMITTER_EMAIL": "osprey@example.org",
}


@pytest.fixture(scope="module")
def _git_env() -> Iterator[None]:
    """Hold :data:`GIT_ENV` for every git a module fixture or test runs."""
    with pytest.MonkeyPatch.context() as patch:
        for name, value in GIT_ENV.items():
            patch.setenv(name, value)
        yield


def _git(root: Path, *args: str) -> str:
    # No detached auto-gc: it repacks the objects a test is copying.
    return subprocess.run(
        [
            "git",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "gc.auto=0",
            "-c",
            "maintenance.auto=false",
            *args,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout


def _commit_all(root: Path, message: str) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _extend(repo: Path) -> None:
    """Add to the previous release's repo what a deployment in service also holds."""
    ci = exemplar_source_files(with_ci=True)
    pipeline = ci[".gitlab-ci.yml"]
    for current, retired in PIPELINE_SPELLINGS:
        assert current in pipeline, current
        pipeline = pipeline.replace(current, retired)
    assert RETIRED_INCLUDE in pipeline
    ci[".gitlab-ci.yml"] = pipeline
    _write(repo, {old: ci[new] for new, old in CI_FILES.items()})
    # The scaffolder writes the current paths; the previous release kept the
    # same files at the retired ones.
    scaffold_systemd_unit(repo)
    for new, old in SCAFFOLDED.items():
        (repo / new).rename(repo / old)
    shutil.rmtree(repo / "ops")
    _write(repo, OPERATOR_FILES)
    _write(repo, {rel: f"minted {rel}\n" for rel in CERTIFICATES})
    _write(repo, dict.fromkeys(CANOPY, "staged\n"))
    _write(repo, {".dockerignore": DOCKERIGNORE})
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    assert text.count("\nconfig:\n") == 1
    assert text.count(HOME_SCREEN) == 1
    text = text.replace(HOME_SCREEN, "").replace("\nconfig:\n", f"\nconfig:\n{PROFILE_VALUE}")
    profile.write_text(text, encoding="utf-8")


def _retired_tree(parent: Path) -> Path:
    """The extended previous-release repo, committed to its own repository."""
    repo = _previous_release_repo(parent / PROJECT)
    _extend(repo)
    _git(repo, "init", "-q")
    _commit_all(repo, "previous release, in service")
    return repo


def _copy(repo: Path, parent: Path) -> Path:
    return Path(shutil.copytree(repo, parent / PROJECT, symlinks=True))


def _sources(repo: Path) -> dict[str, bytes]:
    """Every file of the deployment a migration may touch, with its bytes."""
    files: dict[str, bytes] = {}
    for path in sorted(repo.rglob("*")):
        relative = path.relative_to(repo)
        if relative.parts[0] in NOT_SOURCES or not path.is_file():
            continue
        files[relative.as_posix()] = path.read_bytes()
    return files


def _snapshot(repo: Path) -> dict[str, bytes | None]:
    """Every path in the deployment, git's directory and the state zone included."""
    return {
        path.relative_to(repo).as_posix(): (
            path.read_bytes() if path.is_file() and not path.is_symlink() else None
        )
        for path in sorted(repo.rglob("*"))
    }


def _refusal_message(repo: Path) -> str:
    with pytest.raises(RetiredRootRefused) as refusal:
        refuse_retired_layout(repo)
    return refusal.value.format_message()


def _refusal_block(output: str) -> list[str]:
    """The refusal's lines, from its first comment through its closing line.

    The retired-key warnings a build prints before it are not part of it.
    """
    lines = output.rstrip("\n").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(REFUSAL_OPENS))
    return lines[start:]


def _assert_refused(repo: Path, output: str, exit_code: int) -> None:
    assert exit_code == 1, output
    assert REFUSAL_OPENS + str(repo.resolve()) in output, output
    assert f"# So does the facility tree at {(repo / FACILITY).resolve()}." in output, output
    assert _refusal_block(output)[-1] == _revalidate_command(repo.resolve()), output
    assert "Traceback" not in output, output
    assert not (repo / "build").exists()


def _revalidate(repo: Path) -> str:
    result = CliRunner().invoke(init, [str(repo), "--revalidate"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result.output


def _left_by_hand(output: str) -> list[str]:
    """What ``--revalidate`` listed for the operator, one stripped line each."""
    _, found, tail = output.partition("Left for you to do by hand:\n")
    assert found, output
    return [line.strip() for line in tail.splitlines() if line.strip()]


def _assert_fresh_init_layout(migrated: Migrated) -> None:
    """The migrated sources sit where a fresh init of their preset puts them."""
    repo, sources = migrated.repo, migrated.sources
    assert not (repo / DISSOLVED).exists(), repo
    assert find_retired_root(repo).empty, repo
    assert find_retired(repo / FACILITY).empty, repo
    top = {rel.split("/")[0] for rel in sources}
    assert top <= FRESH_TOP | OPERATOR_TOP, sorted(top - FRESH_TOP - OPERATOR_TOP)
    missing = [rel for rel in FRESH_FILES if rel not in sources]
    assert not missing, missing


def _assert_second_run_changes_nothing(repo: Path, tmp_path: Path) -> None:
    copy = _copy(repo, tmp_path)
    before = _snapshot(copy)
    output = _revalidate(copy)
    assert "nothing to rewrite" in output, output
    assert "Host steps" not in output, output
    assert _snapshot(copy) == before


def _path_values(node: object, key: tuple[str, ...] = ()) -> dict[tuple[str, ...], str]:
    """Every string in a profile document that reads as a path relative to the repo."""
    if isinstance(node, dict):
        return {
            found: value
            for name, child in node.items()
            for found, value in _path_values(child, (*key, str(name))).items()
        }
    if isinstance(node, list):
        return {
            found: value
            for index, child in enumerate(node)
            for found, value in _path_values(child, (*key, str(index))).items()
        }
    if (
        isinstance(node, str)
        and "/" in node
        and not node.startswith("/")
        and "{" not in node
        and "://" not in node
        and not any(character.isspace() for character in node)
    ):
        return {key: node}
    return {}


def _assert_keys_resolve_as_before(before: Path, after: Path) -> None:
    """Each rewritten path value of *after*'s profile resolves exactly when it did in *before*.

    Args:
        before: The tree before the migration.
        after: The same tree migrated, at another path or the same.
    """
    old = _path_values(yaml.safe_load((before / "profile.yml").read_text(encoding="utf-8")))
    new = _path_values(yaml.safe_load((after / "profile.yml").read_text(encoding="utf-8")))
    rewritten = sorted(key for key in old.keys() & new.keys() if old[key] != new[key])
    assert rewritten, (old, new)
    changed = [
        (".".join(key), old[key], new[key])
        for key in rewritten
        if (before / old[key]).exists() != (after / new[key]).exists()
    ]
    assert not changed, changed


def _normalised(output: str) -> list[str]:
    return [" ".join(line.split()) for line in output.splitlines()]


def _assert_names_host_steps(repo: Path, output: str) -> None:
    lines = _normalised(output)
    for step in HOST_STEPS:
        assert any(step in line for line in lines), (step, output)
    unit = f"cp {(repo / 'ops' / 'osprey.service').resolve()} ~/.config/systemd/user/"
    assert any(unit in line for line in lines), output
    hook = str((repo / "ops" / "scripts" / "osprey-boot-hook.sh").resolve())
    assert any(line.startswith("# @reboot ") and hook in line for line in lines), output


@pytest.fixture(scope="module")
def retired(tmp_path_factory: pytest.TempPathFactory, _git_env: None) -> Path:
    """The extended previous-release repo every test copies; never changed itself."""
    return _retired_tree(tmp_path_factory.mktemp("retired"))


@dataclass(frozen=True)
class Migrated:
    """A retired-layout deployment taken through refusal, ``--revalidate`` and a build.

    Attributes:
        repo: The deployment.
        refusal: What the build before the migration printed.
        revalidate: What ``--revalidate`` printed.
        sources: The deployment's files just after the migration.
        tracked: ``git status`` just after the migration.
        built: What the build after the migration printed.
        built_code: That build's exit code.
    """

    repo: Path
    refusal: str
    revalidate: str
    sources: dict[str, bytes]
    tracked: str
    built: str
    built_code: int


@pytest.fixture(scope="module")
def revalidated(
    retired: Path, tmp_path_factory: pytest.TempPathFactory, _git_env: None
) -> Migrated:
    """A copy migrated by ``osprey init --revalidate``, then built."""
    repo = _copy(retired, tmp_path_factory.mktemp("revalidated"))
    refused = run_build(repo)
    _assert_refused(repo, refused.output, refused.exit_code)
    return _migrated(repo, refused.output, _revalidate(repo))


def _migrated(repo: Path, refusal: str, revalidate: str) -> Migrated:
    """*repo* just after its migration, then built."""
    sources = _sources(repo)
    tracked = _git(repo, "status", "--porcelain=v1", "-uall")
    built = run_build(repo)
    return Migrated(
        repo=repo,
        refusal=refusal,
        revalidate=revalidate,
        sources=sources,
        tracked=tracked,
        built=built.output,
        built_code=built.exit_code,
    )


# ── (a) every command refuses before any write ──────────────────────────────


_CASES = [
    pytest.param(args, id=" ".join(args[:4]))
    for invocations in REFUSING.values()
    for args in invocations
]


@pytest.mark.parametrize("args", _CASES)
def test_every_command_refuses_before_any_write(
    retired: Path, tmp_path: Path, args: list[str]
) -> None:
    repo = _copy(retired, tmp_path)
    export = tmp_path / "export.csv"
    export.write_text("address\nSR:NEW:CHANNEL\n", encoding="utf-8")
    values = {
        "repo": str(repo),
        "tier": str(sorted((repo / "tiers").glob("*.yml"))[0]),
        "export": str(export),
        "out": str(tmp_path / "render"),
    }
    expected = _refusal_message(repo)
    before = _snapshot(repo)

    result = CliRunner().invoke(cli, [arg.format(**values) for arg in args])

    assert result.exit_code == 1, result.output
    assert result.output.count(expected) == 1, result.output
    assert "Traceback" not in result.output
    assert _snapshot(repo) == before
    assert not (tmp_path / "render").exists()


def test_the_refusal_covers_root_and_facility_together(retired: Path) -> None:
    block = _refusal_message(retired).splitlines()

    for old, new in (
        ("rules", "agent/rules"),
        ("web-terminal-context", "agent/context"),
        ("services", "ops/services"),
        ("scripts", "ops/scripts"),
        ("osprey.service", "ops/osprey.service"),
        ("ci-extra.yml", "ops/ci-extra.yml"),
        ("data/code_review", "agent/code_review"),
        ("data/.runtime", "var/runtime"),
        (RETIRED_FACILITY, FACILITY),
        (f"{FACILITY}/scenarios", f"{FACILITY}/simulation/scenarios"),
    ):
        assert any(line.endswith(f"mv {old} {new}") for line in block), (old, block)
    assert f"# So does the facility tree at {(retired / FACILITY).resolve()}." in block, block
    assert any(line.endswith(" .dockerignore") for line in block), block
    assert any(line.endswith(" .gitlab-ci.yml") for line in block), block
    assert any(line.endswith(" profile.yml") for line in block), block
    for rel in ("canopy_payload", *CANOPY[1:]):
        assert any(line.endswith(f"-- {rel}") for line in block), (rel, block)


# ── (b) both routes, one tree, which builds ─────────────────────────────────


def test_the_revalidated_tree_has_the_fresh_init_layout(revalidated: Migrated) -> None:
    repo = revalidated.repo

    _assert_fresh_init_layout(revalidated)
    for gone in ("rules", "web-terminal-context", "services", "scripts", "osprey.service"):
        assert not (repo / gone).exists(), gone
    for gone in ("ci-extra.yml", "data/code_review", "data/.runtime", *CANOPY):
        assert not (repo / gone).exists(), gone
    assert not (repo / "canopy_payload").exists()
    profile = (repo / "profile.yml").read_text(encoding="utf-8")
    assert MIGRATED_PROFILE_VALUE in profile
    assert "data/code_review" not in profile
    pipeline = (repo / ".gitlab-ci.yml").read_text(encoding="utf-8")
    assert MIGRATED_INCLUDE in pipeline, pipeline
    assert RETIRED_INCLUDE not in pipeline, pipeline


def test_the_migrated_tree_builds(revalidated: Migrated) -> None:
    assert revalidated.built_code == 0, revalidated.built
    assert "Traceback" not in revalidated.built
    assert find_retired_root(revalidated.repo).empty


def test_the_moves_keep_their_history(revalidated: Migrated, tmp_path: Path) -> None:
    staged = _git(revalidated.repo, "diff", "--cached", "--name-status", "-M")
    for old, new in RENAMES:
        assert f"\t{old}\t{new}\n" in staged, (old, staged)
    rule = next(rel for rel in sorted(revalidated.sources) if rel.startswith("agent/rules/"))

    repo = _copy(revalidated.repo, tmp_path)
    _git(repo, "commit", "-q", "-m", "current root layout")
    log = _git(repo, "log", "--follow", "--oneline", "--", rule)
    assert len(log.splitlines()) == 2, (rule, log)


def test_the_certificates_arrive_byte_identical(revalidated: Migrated, retired: Path) -> None:
    for rel in CERTIFICATES:
        moved = revalidated.repo / RUNTIME / rel.removeprefix(f"{RETIRED_RUNTIME}/")
        assert moved.read_bytes() == (retired / rel).read_bytes(), rel


def test_the_rewritten_keys_resolve_as_before(revalidated: Migrated, retired: Path) -> None:
    _assert_keys_resolve_as_before(retired, revalidated.repo)


def test_a_second_run_is_a_no_op(revalidated: Migrated, tmp_path: Path, _git_env: None) -> None:
    _assert_second_run_changes_nothing(revalidated.repo, tmp_path)


# ── (b2) the printed block pasted only in part, then one --revalidate ───────


def _stopped_midway(block: list[str]) -> list[str]:
    """The block up to the middle of its commands: the shell stopped there."""
    commands = [i for i, line in enumerate(block) if line and not line.startswith("#")]
    return block[: commands[len(commands) // 2]]


def _without_its_last_line(block: list[str]) -> list[str]:
    """Every line of the block but its closing ``osprey init --revalidate``."""
    return block[:-1]


#: How much of the printed block was pasted, by case.
PARTIAL_PASTES = {
    "stopped-midway": _stopped_midway,
    "without-its-last-line": _without_its_last_line,
}


def _sh(script: str, cwd: Path) -> None:
    """Run *script* under ``sh -e`` from *cwd*, as a user pastes it."""
    cwd.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(
        ["sh", "-e"],
        input=script,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **GIT_ENV},
        timeout=300,
    )
    assert done.returncode == 0, (script, done.stdout, done.stderr)


@pytest.fixture(scope="module", params=sorted(PARTIAL_PASTES))
def pasted(
    request: pytest.FixtureRequest,
    retired: Path,
    tmp_path_factory: pytest.TempPathFactory,
    _git_env: None,
) -> Migrated:
    """A copy with part of the printed block pasted, then migrated by ``--revalidate``."""
    parent = tmp_path_factory.mktemp(request.param)
    repo = _copy(retired, parent)
    refused = run_build(repo)
    _assert_refused(repo, refused.output, refused.exit_code)
    block = _refusal_block(refused.output)
    assert block[2] == PASTE_HEADER, refused.output
    part = PARTIAL_PASTES[request.param](block)
    assert len(part) < len(block), part
    _sh("\n".join(part) + "\n", parent / "elsewhere")
    return _migrated(repo, refused.output, _revalidate(repo))


def test_a_partial_paste_then_revalidate_leaves_the_revalidated_tree(
    pasted: Migrated, revalidated: Migrated
) -> None:
    assert pasted.sources == revalidated.sources
    assert pasted.tracked == revalidated.tracked
    _assert_fresh_init_layout(pasted)


def test_a_partial_paste_then_revalidate_builds(pasted: Migrated) -> None:
    assert pasted.built_code == 0, pasted.built
    assert "Traceback" not in pasted.built


def test_a_partial_paste_then_revalidate_runs_twice_as_once(
    pasted: Migrated, tmp_path: Path, _git_env: None
) -> None:
    _assert_second_run_changes_nothing(pasted.repo, tmp_path)


# ── (b3) rendered sources under a non-default in-repo data: root ────────────


def _in_repo_data_tree(parent: Path) -> Path:
    """The previous release, its rendered sources under ``content/`` named by ``data:``.

    Its ``data/`` keeps ``.runtime/`` and a second lane's certificates at
    ``data/<lane>_curve``.
    """
    repo = _previous_release_repo(parent / PROJECT)
    _extend(repo)
    (repo / DATA_ROOT).mkdir()
    for name in RENDERED_SOURCES:
        old = repo / DISSOLVED / name
        if old.exists():
            old.rename(repo / DATA_ROOT / name)
    assert (repo / DATA_ROOT / "facility").is_dir()
    for rel in LANE_CERTIFICATES:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(range(256)) + rel.encode("utf-8"))
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    text = text.replace(f"./{RETIRED_FACILITY}", f"./{DATA_ROOT}/facility")
    text = text.replace(f" {RETIRED_FACILITY}", f" {DATA_ROOT}/facility")
    profile.write_text(f"{text.rstrip()}\ndata: {DATA_ROOT}\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _commit_all(repo, "previous release, rendered sources under content/")
    return repo


@pytest.fixture(scope="module")
def in_repo_data(tmp_path_factory: pytest.TempPathFactory, _git_env: None) -> Path:
    """The in-repo ``data:`` tree every test of it copies; never changed itself."""
    return _in_repo_data_tree(tmp_path_factory.mktemp("in-repo-data"))


@pytest.fixture(scope="module")
def in_repo_data_migrated(
    in_repo_data: Path, tmp_path_factory: pytest.TempPathFactory, _git_env: None
) -> Migrated:
    """A copy of the in-repo ``data:`` tree migrated by ``--revalidate``, then built."""
    repo = _copy(in_repo_data, tmp_path_factory.mktemp("in-repo-data-migrated"))
    refused = run_build(repo)
    _assert_refused(repo, refused.output, refused.exit_code)
    return _migrated(repo, refused.output, _revalidate(repo))


def test_an_in_repo_data_root_reaches_the_fresh_init_layout(
    in_repo_data_migrated: Migrated,
) -> None:
    repo = in_repo_data_migrated.repo

    _assert_fresh_init_layout(in_repo_data_migrated)
    assert not (repo / DATA_ROOT).exists()
    assert (repo / FACILITY).is_dir()
    profile = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
    assert "data" not in profile, profile.get("data")


def test_an_in_repo_data_root_keeps_every_certificate_byte_identical(
    in_repo_data_migrated: Migrated, in_repo_data: Path
) -> None:
    repo = in_repo_data_migrated.repo
    moved = {rel: rel.removeprefix(f"{RETIRED_RUNTIME}/") for rel in CERTIFICATES}
    moved |= {rel: rel.removeprefix(f"{DISSOLVED}/") for rel in LANE_CERTIFICATES}
    for old, new in moved.items():
        assert (repo / RUNTIME / new).read_bytes() == (in_repo_data / old).read_bytes(), old
    lanes = {new.split("/")[0] for new in moved.values()}
    assert lanes == {f"{lane}_curve" for lane in LANE_KEYS[:2]}, lanes


def test_an_in_repo_data_root_builds(in_repo_data_migrated: Migrated) -> None:
    assert in_repo_data_migrated.built_code == 0, in_repo_data_migrated.built
    assert "Traceback" not in in_repo_data_migrated.built


def test_an_in_repo_data_root_runs_twice_as_once(
    in_repo_data_migrated: Migrated, tmp_path: Path, _git_env: None
) -> None:
    _assert_second_run_changes_nothing(in_repo_data_migrated.repo, tmp_path)


def test_an_in_repo_data_root_keeps_every_key_resolving(
    in_repo_data_migrated: Migrated, in_repo_data: Path
) -> None:
    _assert_keys_resolve_as_before(in_repo_data, in_repo_data_migrated.repo)


# ── (b4) path values: ./ rewritten, absolute and anchored listed ────────────


@dataclass(frozen=True)
class Anchored:
    """The previous release with facility paths spelled three ways, migrated in place.

    Attributes:
        repo: The deployment, migrated.
        before: An untouched copy of it from before the migration.
        absolute: The absolute value as written.
        revalidate: What ``--revalidate`` printed.
    """

    repo: Path
    before: Path
    absolute: str
    revalidate: str


@pytest.fixture(scope="module")
def anchored(tmp_path_factory: pytest.TempPathFactory, _git_env: None) -> Anchored:
    """The previous release whose profile spells facility paths ``./``, absolute and anchored."""
    parent = tmp_path_factory.mktemp("anchored")
    repo = _previous_release_repo(parent / PROJECT)
    _extend(repo)
    absolute = ABSOLUTE_VALUE.format(repo=repo.resolve())
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    values = (
        f"  {DOT_SLASH_KEY}: {DOT_SLASH_VALUE}\n"
        f"  {ABSOLUTE_KEY}: {absolute}\n"
        f"  {ANCHORED_KEY}: '{ANCHORED_VALUE}'\n"
    )
    stated = f"  {DOT_SLASH_KEY}: "
    assert sum(line.startswith(stated) for line in text.splitlines()) == 1, text
    text = "".join(line for line in text.splitlines(keepends=True) if not line.startswith(stated))
    text = text.replace(PROFILE_VALUE, "").replace("\nconfig:\n", f"\nconfig:\n{values}")
    profile.write_text(text, encoding="utf-8")
    _git(repo, "init", "-q")
    _commit_all(repo, "previous release, facility paths spelled three ways")
    before = Path(shutil.copytree(repo, parent / "before" / PROJECT, symlinks=True))
    return Anchored(repo, before, absolute, _revalidate(repo))


def _config(repo: Path) -> dict[str, object]:
    return yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))["config"]


def test_a_dot_slash_facility_value_follows_the_move(anchored: Anchored) -> None:
    config = _config(anchored.repo)

    assert config[DOT_SLASH_KEY] == MIGRATED_DOT_SLASH_VALUE
    assert (anchored.repo / MIGRATED_DOT_SLASH_VALUE).is_dir()


def test_absolute_and_anchored_values_are_listed_and_left_alone(anchored: Anchored) -> None:
    config = _config(anchored.repo)
    left = _left_by_hand(anchored.revalidate)

    assert config[ABSOLUTE_KEY] == anchored.absolute
    assert config[ANCHORED_KEY] == ANCHORED_VALUE
    for key, value in ((ABSOLUTE_KEY, anchored.absolute), (ANCHORED_KEY, ANCHORED_VALUE)):
        assert any(line.endswith(f": config.{key}: {value}") for line in left), (key, left)


def test_the_anchored_tree_keeps_every_key_resolving(anchored: Anchored) -> None:
    _assert_keys_resolve_as_before(anchored.before, anchored.repo)


# ── (b5) what the migration cannot finish stays refused and is listed ──────


def test_an_out_of_repo_data_root_stays_refused_and_nothing_of_it_moves(
    retired: Path, tmp_path: Path, _git_env: None
) -> None:
    repo = _copy(retired, tmp_path)
    outside = (tmp_path / "outside").resolve()
    outside.mkdir()
    for name in RENDERED_SOURCES:
        old = repo / DISSOLVED / name
        if old.exists():
            shutil.move(str(old), outside / name)
    assert (outside / "facility").is_dir()
    profile = repo / "profile.yml"
    data_line = f"data: {outside}"
    profile.write_text(f"{profile.read_text(encoding='utf-8').rstrip()}\n{data_line}\n")
    _commit_all(repo, "rendered sources outside the repo")
    held = _snapshot(outside)
    listed = f"{data_line}: names no single directory inside the repo"

    refused = run_build(repo)
    revalidate = CliRunner().invoke(init, [str(repo), "--revalidate"], catch_exceptions=False)
    again = run_build(repo)

    for output in (refused.output, again.output):
        block = _refusal_block(output)
        assert block[-1] == _PLACE_THESE, output
        assert not any(line.startswith("osprey init") for line in block), output
        assert any(listed in line for line in block), output
    assert refused.exit_code == again.exit_code == 1
    assert revalidate.exit_code == 0, revalidate.output
    assert any(listed in line for line in _left_by_hand(revalidate.output)), revalidate.output
    assert data_line in profile.read_text(encoding="utf-8").splitlines()
    assert _snapshot(outside) == held
    assert not (repo / FACILITY).exists()
    assert not (repo / "build").exists()


def test_a_note_left_in_data_is_refused_and_listed(
    revalidated: Migrated, tmp_path: Path, _git_env: None
) -> None:
    repo = _copy(revalidated.repo, tmp_path)
    note = repo / DISSOLVED / "notes.md"
    note.parent.mkdir()
    note.write_text("the operator's notes\n", encoding="utf-8")
    listing = [
        f"# Still in {DISSOLVED}/, place by hand:",
        f"#   {DISSOLVED}/notes.md",
        f"# Content the render must see goes under {FACILITY}/.",
    ]

    refused = run_build(repo)
    revalidate = CliRunner().invoke(init, [str(repo), "--revalidate"], catch_exceptions=False)

    assert refused.exit_code == 1, refused.output
    block = _refusal_block(refused.output)
    assert block[-len(listing) - 1 :] == [*listing, _PLACE_THESE], refused.output
    assert revalidate.exit_code == 0, revalidate.output
    assert _left_by_hand(revalidate.output) == listing, revalidate.output
    assert note.read_text(encoding="utf-8") == "the operator's notes\n"
    assert find_retired_root(repo).leftovers == (f"{DISSOLVED}/notes.md",)


# ── (c) a clash refuses by both routes; a merge keeps every file ────────────


def test_a_clash_refuses_by_both_routes_naming_every_file(
    retired: Path, tmp_path: Path, _git_env: None
) -> None:
    repo = _copy(retired, tmp_path)
    rule = sorted(p.name for p in (repo / "rules").iterdir() if p.is_file())[0]
    held = next(p for p in sorted((repo / RETIRED_FACILITY).rglob("*")) if p.is_file())
    inner = held.relative_to(repo / RETIRED_FACILITY).as_posix()
    clashes = (
        (f"rules/{rule}", f"agent/rules/{rule}"),
        ("ci-extra.yml", "ops/ci-extra.yml"),
        ("services/worker/Dockerfile", "ops/services/worker/Dockerfile"),
        (f"{RETIRED_FACILITY}/{inner}", f"{FACILITY}/{inner}"),
    )
    _write(repo, {new: "# Kept here too.\n" for _old, new in clashes})
    _commit_all(repo, "copies at the current paths")
    before = _snapshot(repo)

    built = run_build(repo)
    revalidate = CliRunner().invoke(init, [str(repo), "--revalidate"])

    assert built.exit_code == 1, built.output
    assert revalidate.exit_code != 0, revalidate.output
    assert isinstance(revalidate.exception, SystemExit), revalidate.exception
    assert "Traceback" not in built.output + revalidate.output
    root = repo.resolve()
    refusal = _refusal_block(built.output)
    assert refusal[-1] == _PLACE_THESE, built.output
    named = " ".join(revalidate.output.split())
    for old, new in clashes:
        line = f"# {root / old} and {root / new} both exist: keep one, delete the other."
        assert line in refusal, (old, built.output)
        assert f"{old} and {new}" in named, (old, revalidate.output)
    assert _snapshot(repo) == before


def test_a_merge_into_a_successor_keeps_every_file(
    retired: Path, tmp_path: Path, _git_env: None
) -> None:
    repo = _copy(retired, tmp_path)
    old_rules = sorted(
        p.relative_to(repo / "rules").as_posix() for p in (repo / "rules").rglob("*") if p.is_file()
    )
    _write(
        repo, {"agent/rules/current.md": "# Already moved.\n", "ops/scripts/current.sh": "true\n"}
    )
    _commit_all(repo, "files already at the current paths")

    result = CliRunner().invoke(init, [str(repo), "--revalidate"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    rules = sorted(
        p.relative_to(repo / "agent" / "rules").as_posix()
        for p in (repo / "agent" / "rules").rglob("*")
        if p.is_file()
    )
    assert rules == sorted([*old_rules, "current.md"])
    for rel in ("current.sh", "verify.sh", "osprey-boot-hook.sh", "private/token.txt"):
        assert (repo / "ops" / "scripts" / rel).is_file(), rel
    assert not (repo / "rules").exists()
    assert not (repo / "scripts").exists()
    assert find_retired_root(repo).empty


# ── (d) the .dockerignore follows the move; both outputs name the hand steps ─


def test_the_migrated_dockerignore_keeps_the_moved_entry_out_of_the_image(
    revalidated: Migrated,
) -> None:
    repo = revalidated.repo
    assert revalidated.built_code == 0, revalidated.built
    assert (repo / ".dockerignore").read_text(encoding="utf-8") == MIGRATED_DOCKERIGNORE
    assert (repo / "ops" / "scripts" / "private" / "token.txt").is_file()

    contexts = sorted(path for path in (repo / "build" / IMAGE_DIR_NAME).iterdir() if path.is_dir())
    assert contexts, revalidated.built
    for context in contexts:
        assert (context / "ops" / "scripts" / "verify.sh").is_file(), context
        assert not (context / "ops" / "scripts" / "private").exists(), context
        assert not (context / "scripts").exists(), context
        assert not (context / "var").exists(), context
        assert not (context / "data" / ".runtime").exists(), context
        assert not (context / "canopy_payload").exists(), context
        assert not list(context.glob("*.whl")), context


def test_both_outputs_name_the_host_steps_and_the_files_to_check(
    revalidated: Migrated,
) -> None:
    _assert_names_host_steps(revalidated.repo, "\n".join(_refusal_block(revalidated.refusal)))
    steps = revalidated.revalidate[revalidated.revalidate.index("Left for you to do by hand") :]
    _assert_names_host_steps(revalidated.repo, steps)


# ── (e) init --force over the migrated tree ─────────────────────────────────


def test_force_keeps_operator_content_and_replaces_its_own_entries(
    revalidated: Migrated, tmp_path: Path, _git_env: None
) -> None:
    repo = _copy(revalidated.repo, tmp_path)
    for entry in REPLACED_GROUP_ENTRIES:
        assert entry in MATERIALIZED_SOURCE_ENTRIES, entry
    _write(
        repo,
        {
            "agent/rules/operator.md": "# The operator's own rule.\n",
            "agent/skills/operator-skill/SKILL.md": "# The operator's own skill.\n",
            "ops/project/notes.md": "operator notes\n",
            **{
                f"{entry}/operator-note.md": "replaced by design\n"
                for entry in REPLACED_GROUP_ENTRIES
            },
        },
    )
    replaced = tuple(entry for entry in MATERIALIZED_SOURCE_ENTRIES if "/" in entry)

    def kept(rel: str) -> bool:
        return rel.startswith(("agent/", "ops/")) and not any(
            rel == entry or rel.startswith(f"{entry}/") for entry in replaced
        )

    before = {rel: data for rel, data in _sources(repo).items() if kept(rel)}
    assert "agent/rules/operator.md" in before
    assert "ops/osprey.service" in before
    assert "ops/services/worker/Dockerfile" in before

    result = CliRunner().invoke(
        cli, ["init", str(repo), "--preset", "demo", "--force"], catch_exceptions=False
    )

    assert result.exit_code == 0, result.output
    after = _sources(repo)
    assert {rel: after.get(rel) for rel in before} == before
    for entry in REPLACED_GROUP_ENTRIES:
        assert not (repo / entry / "operator-note.md").exists(), entry
    assert (repo / "agent" / "context").is_dir()
    assert (repo / "agent" / "watchdogs").is_dir()
    assert (repo / "agent" / "code_review").is_dir()
    assert not (repo / HELD_SOURCE_ZONE_DIRNAME).exists()
    assert not any(path.name == HELD_SOURCE_ZONE_DIRNAME for path in repo.rglob("*"))
    assert find_retired_root(repo).empty


# ── (f) a machine submodule on the previous layout ──────────────────────────


@pytest.fixture(scope="module")
def previous_layout_submodule(tmp_path_factory: pytest.TempPathFactory, _git_env: None) -> Path:
    """A deployment mounting, as a submodule, a machine repository on the previous layout.

    The machine repository keeps its facility at ``data/facility``, as the
    previous release laid it; the deployment's own ``facility/`` holds no
    content directory.
    """
    parent = tmp_path_factory.mktemp("submodule")
    source = make_machine_repository(parent / "example-machine")
    (source / DISSOLVED).mkdir()
    git("mv", FACILITY, RETIRED_FACILITY, cwd=source)
    commit_all(source, "the machine, on the previous layout")
    repo = parent / "deployment"
    (repo / FACILITY).mkdir(parents=True)
    (repo / FACILITY / ".gitkeep").write_text("", encoding="utf-8")
    (repo / "profile.yml").write_text(
        "extends: hello-world\n"
        "name: Mounted Deployment\n"
        "project_name: deployment\n"
        f"machine: {SUBMODULE_MOUNT}/{RETIRED_FACILITY}\n",
        encoding="utf-8",
    )
    git_init(repo)
    commit_all(repo, "the deployment")
    git("submodule", "add", "-q", str(source), SUBMODULE_MOUNT, cwd=repo)
    commit_all(repo, "mount the machine")
    return repo


def test_a_machine_submodule_on_the_previous_layout_builds(
    previous_layout_submodule: Path, _git_env: None
) -> None:
    repo = previous_layout_submodule

    built = run_build(repo)

    assert built.exit_code == 0, built.output
    assert "Traceback" not in built.output, built.output
    assert REFUSAL_OPENS not in built.output, built.output
    assert "# So does the facility tree" not in built.output, built.output
    assert find_retired_root(repo).empty
    machine = repo / SUBMODULE_MOUNT / RETIRED_FACILITY
    assert (machine / "identity.yaml").is_file()
