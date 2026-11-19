"""The example bundle's figure: in the concept, in a built project, and in the wheel.

The example facility ships one picture, a schematic its BPM concept embeds. A
picture only helps if it travels with the bundle, and three things decide that:
the preset copies it into a project's ``data/facility/``, git tracks it, and
hatch packages it. Hatch builds the wheel from the source tree minus the
``exclude`` patterns in ``pyproject.toml``, so those patterns, not a manifest,
are what keep a file in or out.
"""

from __future__ import annotations

import fnmatch
import re
import shutil
import subprocess
import tomllib
from pathlib import Path, PurePosixPath

import pytest

from tests._preset_data import bundle_data_root, packaged_facility_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURE = PurePosixPath("knowledge/devices/figures/bpm-button-pickups.png")
CONCEPT = PurePosixPath("knowledge/devices/bpm.md")
MAX_BYTES = 200 * 1024
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<target>[^)\s]+)\)")


@pytest.fixture(scope="module")
def facility() -> Path:
    """The packaged example facility the demo preset lands at ``data/facility/``."""
    return packaged_facility_dir("demo")


def _figure(facility: Path) -> Path:
    return facility.joinpath(*FIGURE.parts)


def test_the_figure_is_a_small_png(facility: Path) -> None:
    """A PNG (the repository ignores JPEG and GIF), under 200 KB."""
    figure = _figure(facility)

    assert figure.is_file()
    data = figure.read_bytes()
    assert data.startswith(_PNG_SIGNATURE)
    assert len(data) < MAX_BYTES


def test_the_bpm_concept_embeds_the_figure_with_alt_text(facility: Path) -> None:
    """The concept shows the figure by a link that resolves to it, and describes it."""
    concept = facility.joinpath(*CONCEPT.parts)
    images = _IMAGE_RE.findall(concept.read_text(encoding="utf-8"))

    assert len(images) == 1
    alt, target = images[0]
    assert len(alt.split()) >= 5, "alt text must describe the figure, not name it"
    assert (concept.parent / target).resolve() == _figure(facility).resolve()


def test_the_figure_reaches_a_project_built_from_the_example_facility(facility: Path) -> None:
    """The demo preset's composed ``data/`` tree carries the figure byte for byte."""
    built = bundle_data_root("demo").joinpath("facility", *FIGURE.parts)

    assert built.is_file()
    assert built.read_bytes() == _figure(facility).read_bytes()


def test_the_figure_is_not_ignored_by_git(facility: Path) -> None:
    """An ignored file never reaches a commit, so it never reaches a release."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    top = subprocess.run(
        ["git", "-C", str(facility), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if top.returncode != 0:
        pytest.skip("the template is not in a git checkout")
    result = subprocess.run(
        ["git", "-C", top.stdout.strip(), "check-ignore", "--no-index", "-v", "--stdin"],
        input=f"{_figure(facility)}\n",
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.stdout == "", result.stdout


def _hatch_excludes() -> list[str]:
    """The build-wide and the wheel target's exclude patterns, in that order."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    build = pyproject["tool"]["hatch"]["build"]
    return [*build.get("exclude", []), *build["targets"]["wheel"].get("exclude", [])]


def _excluded_by(relative: PurePosixPath, pattern: str) -> bool:
    """Whether a gitignore-style hatch *pattern* takes out the file at *relative*.

    Deliberately generous: ``fnmatch``'s ``*`` also crosses ``/``, a pattern
    without a slash matches any single path part, and a pattern that names a
    directory takes out everything beneath it. A false match fails the test
    loudly; a missed one would ship a wheel without the figure.
    """
    pattern = pattern.strip("/")
    variants = {pattern, pattern.removeprefix("**/")}
    prefixes = [
        PurePosixPath(*relative.parts[:end]).as_posix() for end in range(1, len(relative.parts) + 1)
    ]
    for variant in variants:
        if "/" not in variant and any(fnmatch.fnmatch(part, variant) for part in relative.parts):
            return True
        if any(fnmatch.fnmatch(prefix, variant) for prefix in prefixes):
            return True
        if any(fnmatch.fnmatch(prefix, f"*/{variant}") for prefix in prefixes):
            return True
    return False


def test_the_exclude_matcher_catches_what_pyproject_excludes() -> None:
    """The matcher below is not vacuous: each shipped pattern takes out its own target."""
    assert _excluded_by(
        PurePosixPath("src/osprey/integrations/canopy/repo_checks.py"),
        "src/osprey/integrations/canopy/repo_checks.py",
    )
    assert _excluded_by(
        PurePosixPath("src/osprey/x/prefetched-models/model.gguf"), "**/prefetched-models"
    )
    assert not _excluded_by(PurePosixPath("src/osprey/x/models.py"), "**/prefetched-models")


def test_no_hatch_exclude_pattern_drops_the_figure(facility: Path) -> None:
    """The wheel packages the source tree minus these patterns; none may name the figure."""
    relative = PurePosixPath(_figure(facility).resolve().relative_to(REPO_ROOT).as_posix())
    patterns = _hatch_excludes()

    assert patterns, "pyproject.toml declares no hatch excludes; the reading is wrong"
    assert [pattern for pattern in patterns if _excluded_by(relative, pattern)] == []
