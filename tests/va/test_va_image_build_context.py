"""Every VA image's build context carries the uv workspace members.

The root ``pyproject.toml`` depends on its workspace members by source
(``[tool.uv.sources] ... = { workspace = true }``), so an image that installs
the project, or only its dependency closure, needs each member directory on
disk before its install step runs. Two things must hold for that, in two
different files:

* the script that builds the image stages the member into the scratch
  directory it hands to the build as its context, and
* the Containerfile COPYs it to where the install step looks for it, before
  that step runs.

Neither file is ever exercised by the unit lanes -- building an image is a
container run -- so a context that lacks a member fails only when somebody
next builds the image by hand. This module reads the real staging code and the
real Containerfile instructions, and the member set from ``pyproject.toml``
cross-checked against ``uv.lock``, so adding a member or changing how a script
stages its context is checked here without a build.
"""

from __future__ import annotations

import re
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ImageBuild:
    """One image: the script that stages its context and the Containerfile."""

    script: str
    containerfile: str


IMAGE_BUILDS = (
    ImageBuild("scripts/va/live_ca/run_live_ca.sh", "scripts/va/live_ca/Containerfile"),
    ImageBuild("scripts/va/run_va.sh", "docker/virtual-accelerator/Containerfile"),
)

_ROOT_REF = re.compile(r'"?\$\{WORKTREE_ROOT\}/([^"\s]+)"?')


def _logical_lines(text: str) -> list[str]:
    """Join backslash continuations; drop comments and blank lines."""
    lines: list[str] = []
    pending = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not pending and (not stripped or stripped.startswith("#")):
            continue
        if stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
            continue
        lines.append(pending + stripped)
        pending = ""
    if pending:
        lines.append(pending)
    return lines


def workspace_members() -> set[str]:
    """Member directories, declared in pyproject.toml and locked in uv.lock."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    declared = set(pyproject["tool"]["uv"]["workspace"]["members"])
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    locked = {
        pkg["source"]["editable"]
        for pkg in lock["package"]
        if "editable" in pkg.get("source", {}) and pkg["source"]["editable"] != "."
    }
    assert declared == locked, (
        f"pyproject.toml declares workspace members {sorted(declared)} but uv.lock "
        f"locks {sorted(locked)}"
    )
    return declared


def staged_paths(script: Path) -> set[str]:
    """Repo paths a script copies into its build context with ``cp``."""
    staged: set[str] = set()
    for line in _logical_lines(script.read_text()):
        if not line.startswith("cp "):
            continue
        for match in _ROOT_REF.finditer(line):
            path = PurePosixPath(match.group(1))
            staged.add(str(path))
    return staged


def build_id_inputs(script: Path) -> set[str]:
    """Repo paths the script's ``BUILD_ID=$(...)`` digest reads."""
    for line in _logical_lines(script.read_text()):
        if line.startswith("BUILD_ID="):
            return {str(PurePosixPath(m.group(1))) for m in _ROOT_REF.finditer(line)}
    return set()


def _covers(path: str, member: str) -> bool:
    return member == path or member.startswith(path.rstrip("/") + "/")


@dataclass(frozen=True)
class Instruction:
    keyword: str
    args: str


def _instructions(containerfile: Path) -> list[Instruction]:
    parsed = []
    for line in _logical_lines(containerfile.read_text()):
        keyword, _, args = line.partition(" ")
        parsed.append(Instruction(keyword.upper(), args.strip()))
    return parsed


def _installs_project(run_args: str) -> bool:
    """A RUN that installs the project or its dependency closure."""
    if "uv sync" in run_args:
        return True
    return any(
        "pip install" in step and re.search(r'(^|\s)"?\.(/|\[|\s|$)', step)
        for step in run_args.split("&&")
    )


@dataclass(frozen=True)
class InstallPlan:
    """What a Containerfile has put on disk when its install step runs."""

    install_index: int
    install_args: str
    workdir: str
    copied: frozenset[str]  # in-image paths COPY has landed on


def install_plan(containerfile: Path) -> InstallPlan:
    workdir = "/"
    copied: set[str] = set()
    for index, instr in enumerate(_instructions(containerfile)):
        if instr.keyword == "WORKDIR":
            workdir = str(PurePosixPath(workdir) / instr.args)
        elif instr.keyword == "COPY":
            tokens = [t for t in shlex.split(instr.args) if not t.startswith("--")]
            *sources, dest = tokens
            dest_path = PurePosixPath(workdir) / dest
            for source in sources:
                if len(sources) == 1 and source.endswith("/"):
                    landing = dest_path
                else:
                    landing = dest_path / PurePosixPath(source).name
                copied.add(str(landing))
        elif instr.keyword == "RUN" and _installs_project(instr.args):
            return InstallPlan(index, instr.args, workdir, frozenset(copied))
    raise AssertionError(f"{containerfile}: no RUN installs the project")


@pytest.fixture(scope="module")
def members() -> set[str]:
    found = workspace_members()
    assert found, "pyproject.toml declares no uv workspace members"
    return found


@pytest.mark.parametrize("build", IMAGE_BUILDS, ids=lambda b: b.script)
class TestImageBuildContext:
    def test_script_stages_every_member(self, build: ImageBuild, members: set[str]) -> None:
        staged = staged_paths(REPO_ROOT / build.script)
        assert staged, f"{build.script}: found no cp into the build context"
        missing = sorted(m for m in members if not any(_covers(p, m) for p in staged))
        assert not missing, (
            f"{build.script} stages {sorted(staged)} into the build context, "
            f"which leaves out the workspace members {missing}"
        )

    def test_containerfile_copies_every_member_before_install(
        self, build: ImageBuild, members: set[str]
    ) -> None:
        plan = install_plan(REPO_ROOT / build.containerfile)
        for member in sorted(members):
            expected = str(PurePosixPath(plan.workdir) / member)
            landed = [path for path in plan.copied if _covers(path, expected)]
            assert landed, (
                f"{build.containerfile}: nothing is COPYed to {expected} before the "
                f"install step, so the workspace member {member} is missing when it runs"
            )

    def test_install_step_versions_the_members(self, build: ImageBuild) -> None:
        """The context has no .git, so the VCS-versioned members need a version.

        Every workspace member takes its version from git through hatch-vcs;
        without one the build backend refuses to produce metadata, and the
        install step fails on the staged member.
        """
        plan = install_plan(REPO_ROOT / build.containerfile)
        assert "SETUPTOOLS_SCM_PRETEND_VERSION=" in plan.install_args, (
            f"{build.containerfile}: the install step sets no "
            "SETUPTOOLS_SCM_PRETEND_VERSION, so the workspace members cannot build"
        )


def test_live_ca_image_tag_digests_every_member(members: set[str]) -> None:
    """A change to a member changes the live-CA image tag, so it rebuilds."""
    inputs = build_id_inputs(REPO_ROOT / "scripts/va/live_ca/run_live_ca.sh")
    assert inputs, "run_live_ca.sh: found no BUILD_ID digest"
    missing = sorted(m for m in members if not any(_covers(p, m) for p in inputs))
    assert not missing, (
        f"run_live_ca.sh digests {sorted(inputs)} into the image tag, which leaves "
        f"out the workspace members {missing}"
    )


class TestTheParsersSeeWhatTheyMustSee:
    """The helpers must fail on a context that lacks a member, not only pass."""

    def test_a_script_without_the_member_is_caught(self, tmp_path: Path) -> None:
        script = tmp_path / "run.sh"
        script.write_text(
            'cp "${WORKTREE_ROOT}/pyproject.toml" \\\n'
            '   "${WORKTREE_ROOT}/uv.lock" \\\n'
            '   "${CONTEXT}/"\n'
            'cp -R "${WORKTREE_ROOT}/src/." "${CONTEXT}/src/"\n'
        )
        assert staged_paths(script) == {"pyproject.toml", "uv.lock", "src"}

    def test_a_copy_after_the_install_is_not_counted(self, tmp_path: Path) -> None:
        containerfile = tmp_path / "Containerfile"
        containerfile.write_text(
            "FROM scratch\n"
            "WORKDIR /opt/osprey\n"
            "COPY pyproject.toml uv.lock /opt/osprey/\n"
            "RUN uv sync --frozen \\\n"
            "    --no-install-project\n"
            "COPY packages/ /opt/osprey/packages/\n"
        )
        plan = install_plan(containerfile)
        assert "/opt/osprey/packages" not in plan.copied
        assert plan.workdir == "/opt/osprey"

    def test_a_pip_install_of_the_project_is_the_install_step(self, tmp_path: Path) -> None:
        containerfile = tmp_path / "Containerfile"
        containerfile.write_text(
            "FROM scratch\n"
            'RUN pip install "some-dist==1.0"\n'
            "WORKDIR /opt/osprey\n"
            "COPY packages/ /opt/osprey/packages/\n"
            'RUN pip install "a==1" && pip install ".[extra]"\n'
        )
        plan = install_plan(containerfile)
        assert plan.install_index == 4
        assert "/opt/osprey/packages" in plan.copied
