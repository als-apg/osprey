"""The live Channel Access venue image must keep up with the uv workspace.

``scripts/va/live_ca/Containerfile`` installs only the dependency closure from
``uv.lock`` and leaves every project's source on the bind mount. That broke
silently once already: when ``osprey-connectors`` became a uv workspace member,
``uv sync --no-install-project`` went on trying to install it, and the image
stopped building ("Distribution not found at: .../packages/osprey-connectors").
No CI job builds the image -- CI runs ``gate.py`` natively -- so the breakage
only showed up the next time someone ran ``run_live_ca.sh`` by hand.

Building the image here would cost an amd64 pcaspy install per run. This module
instead pins the three assumptions the image makes about the workspace, all of
which can be read from the repo:

* every workspace member is a directory directly under ``packages/`` with a
  ``pyproject.toml`` and a ``src/`` layout, and every directory there is a
  member -- the staging, the image-tag digest and the ``.pth`` file all glob
  ``packages/*`` and rely on exactly that;
* the sync skips the whole workspace, not only the root project;
* the staging, digest and path wiring are globbed rather than naming members,
  so adding one needs no edit to either file.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VENUE_DIR = REPO_ROOT / "scripts" / "va" / "live_ca"
CONTAINERFILE = VENUE_DIR / "Containerfile"
RUN_SCRIPT = VENUE_DIR / "run_live_ca.sh"
PACKAGES_DIR = REPO_ROOT / "packages"


def _workspace_members() -> list[str]:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    return list(pyproject["tool"]["uv"]["workspace"]["members"])


def _sync_lines() -> list[str]:
    return [
        line
        for line in CONTAINERFILE.read_text().splitlines()
        if line.startswith("RUN ") and "uv sync" in line
    ]


class TestWorkspaceLayoutMatchesTheGlob:
    """``packages/*`` is the venue's definition of "the workspace members"."""

    def test_there_is_at_least_one_member(self) -> None:
        """An empty member list would make every check below vacuous."""
        assert _workspace_members()

    @pytest.mark.parametrize("member", _workspace_members())
    def test_member_lives_directly_under_packages(self, member: str) -> None:
        path = Path(member)
        assert path.parent == Path("packages") and "*" not in member, (
            f"workspace member {member!r} is not a single directory directly under packages/; "
            f"scripts/va/live_ca/ stages and wires members by globbing packages/*, so it would "
            f"be missing from the live-CA image"
        )

    @pytest.mark.parametrize("member", _workspace_members())
    def test_member_has_the_files_the_image_relies_on(self, member: str) -> None:
        assert (REPO_ROOT / member / "pyproject.toml").is_file()
        assert (REPO_ROOT / member / "src").is_dir(), (
            f"{member} has no src/ layout; the image puts /work/{member}/src on the path"
        )

    def test_every_directory_under_packages_is_a_member(self) -> None:
        """The glob would stage a non-member too, and fail on one without a
        ``pyproject.toml``."""
        on_disk = {f"packages/{p.name}" for p in PACKAGES_DIR.iterdir() if p.is_dir()}
        assert on_disk == set(_workspace_members())


class TestTheImageSkipsTheWholeWorkspace:
    def test_there_is_exactly_one_sync(self) -> None:
        assert len(_sync_lines()) == 1

    def test_the_sync_skips_every_workspace_project(self) -> None:
        """``--no-install-project`` skips only the root and still installs the
        members as editables, which cannot build without ``.git/``."""
        (sync,) = _sync_lines()
        assert "--no-install-workspace" in sync
        assert "--no-install-project" not in sync

    def test_the_sync_uses_the_lock_as_committed(self) -> None:
        (sync,) = _sync_lines()
        assert "--frozen" in sync

    def test_the_uv_image_is_pinned(self) -> None:
        """Which workspace projects a sync installs is uv's behaviour, not the
        lockfile's, so a floating uv could break the image with no repo change."""
        (base,) = re.findall(r"^FROM\s+\S+\s+(\S+)", CONTAINERFILE.read_text(), re.MULTILINE)
        assert re.search(r":\d+\.\d+\.\d+-", base), f"uv base image is not version-pinned: {base}"


class TestMembersAreGlobbedNotNamed:
    """A member named in either file is one the next member will not match."""

    @pytest.mark.parametrize("member", _workspace_members())
    @pytest.mark.parametrize("path", [CONTAINERFILE, RUN_SCRIPT], ids=lambda p: p.name)
    def test_no_member_is_named(self, path: Path, member: str) -> None:
        code = "\n".join(
            line for line in path.read_text().splitlines() if not line.lstrip().startswith("#")
        )
        assert member not in code

    def test_the_whole_packages_tree_is_copied_into_the_image(self) -> None:
        assert re.search(r"^COPY packages/ ", CONTAINERFILE.read_text(), re.MULTILINE)

    def test_every_staged_member_gets_a_path_entry(self) -> None:
        text = CONTAINERFILE.read_text()
        assert "/opt/osprey/packages/*/" in text
        assert '"/work/packages/$(basename "${member}")/src"' in text

    def test_the_run_script_stages_every_member(self) -> None:
        assert '"${WORKTREE_ROOT}"/packages/*/; do' in RUN_SCRIPT.read_text()

    def test_every_member_pyproject_is_in_the_image_tag(self) -> None:
        """A member dependency change must force a rebuild, not reuse a stale
        image under a tag that no longer describes it."""
        build_id = RUN_SCRIPT.read_text().split('BUILD_ID="$(', 1)[1].split(")", 1)[0]
        assert '"${WORKTREE_ROOT}"/packages/*/pyproject.toml' in build_id
