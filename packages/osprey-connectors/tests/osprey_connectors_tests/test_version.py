"""The package versions from its own ``osprey-connectors-v*`` calendar tags.

``[tool.hatch.version].raw-options`` in this package's ``pyproject.toml`` hands
the build backend a describe glob and a tag regex. The glob decides which tags
the version can come from: only this package's calendar tags, never the
framework's ``v*`` tags (a second stream would rebase onto the framework's
numbers) and never the retired ``osprey-connectors-v0.1.0`` (every dev build
would become ``0.1.0.postN``, which sorts below the framework's floor). The
regex strips the prefix so the backend can parse what the glob found. The
framework's own glob, read from the root ``pyproject.toml``, must in turn never
reach a connectors tag.

The checks on the glob and the regex are pure string checks; the two that read
git skip where the checkout has no connectors calendar tag reachable.
"""

from __future__ import annotations

import fnmatch
import re
import shlex
import subprocess
import tomllib
import warnings
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dist_version

import pytest
from packaging.version import Version

from osprey_connectors_tests._paths import PACKAGE_ROOT, REPO_ROOT

#: The first calendar release, below which no version from this stream sorts.
CALENDAR_FLOOR = Version("2026")


def _raw_options(pyproject) -> dict:
    with pyproject.open("rb") as handle:
        return tomllib.load(handle)["tool"]["hatch"]["version"]["raw-options"]


def _match_glob(describe_command: str) -> str:
    """Return the ``--match`` pattern of a ``git describe`` command line."""
    argv = shlex.split(describe_command)
    assert argv[:2] == ["git", "describe"], describe_command
    return argv[argv.index("--match") + 1]


def _glob_matches(glob: str, tag: str) -> bool:
    # git describe's --match is an fnmatch-style glob over the whole tag name.
    return fnmatch.fnmatchcase(tag, glob)


@pytest.fixture(scope="module")
def raw_options() -> dict:
    return _raw_options(PACKAGE_ROOT / "pyproject.toml")


@pytest.fixture(scope="module")
def connectors_glob(raw_options) -> str:
    return _match_glob(raw_options["git_describe_command"])


@pytest.fixture(scope="module")
def tag_regex(raw_options) -> re.Pattern[str]:
    return re.compile(raw_options["tag_regex"])


def _describe(glob: str) -> str | None:
    """``git describe`` this checkout over *glob*, or None where git cannot answer."""
    try:
        described = subprocess.run(
            ["git", "describe", "--tags", "--long", "--abbrev=40", "--match", glob],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if described.returncode != 0:
        return None
    return described.stdout.strip()


class TestTheDescribeGlob:
    def test_the_describe_command_is_the_pinned_one(self, raw_options):
        assert raw_options["git_describe_command"] == (
            "git describe --dirty --tags --long --abbrev=40 "
            "--match osprey-connectors-v2[0-9][0-9][0-9].*"
        )

    def test_the_backend_keeps_the_base_on_the_last_tag(self, raw_options):
        # A calendar cannot predict its own next number, so a dev build names
        # the release it descends from (`.postN`) and never guesses the next.
        assert raw_options["version_scheme"] == "post-release"

    def test_the_tags_live_at_the_repository_root(self, raw_options):
        assert (PACKAGE_ROOT / raw_options["root"]).resolve() == REPO_ROOT.resolve()

    @pytest.mark.parametrize(
        "tag",
        [
            "osprey-connectors-v2026.10.0",
            "osprey-connectors-v2026.10.0b1",
            "osprey-connectors-v2027.1.3rc2",
        ],
    )
    def test_reads_the_calendar_tags(self, connectors_glob, tag):
        assert _glob_matches(connectors_glob, tag)

    @pytest.mark.parametrize(
        "tag",
        [
            "osprey-connectors-v0.1.0",  # retired; PyPI already carries 0.1.0
            "v2026.10.0",  # the framework's stream
            "v2026.9.0b3",
            "checkpoint-2026-10-01",
            "pyaml-cs-osprey-v2026.10.0",
        ],
    )
    def test_skips_every_other_tag(self, connectors_glob, tag):
        assert not _glob_matches(connectors_glob, tag)


class TestTheTagRegex:
    def test_the_tag_regex_is_the_pinned_one(self, raw_options):
        assert raw_options["tag_regex"] == r"^osprey-connectors-v(?P<version>2[0-9]{3}\..+)$"

    @pytest.mark.parametrize(
        ("tag", "version"),
        [
            ("osprey-connectors-v2026.10.0", "2026.10.0"),
            ("osprey-connectors-v2026.10.0b1", "2026.10.0b1"),
            ("osprey-connectors-v2026.9.0rc1", "2026.9.0rc1"),
        ],
    )
    def test_extracts_the_calendar_version(self, tag_regex, tag, version):
        found = tag_regex.match(tag)
        assert found is not None
        assert found.group("version") == version
        assert Version(found.group("version")) >= CALENDAR_FLOOR

    @pytest.mark.parametrize(
        "tag", ["osprey-connectors-v0.1.0", "v2026.10.0", "osprey-connectors-v2026"]
    )
    def test_refuses_what_the_glob_skips(self, tag_regex, tag):
        assert tag_regex.match(tag) is None


class TestTheFrameworkStream:
    def test_the_framework_glob_never_reads_a_connectors_tag(self):
        root_pyproject = REPO_ROOT / "pyproject.toml"
        if not root_pyproject.is_file():
            pytest.skip(
                f"no framework pyproject.toml at {root_pyproject}: not a repository checkout"
            )
        framework_glob = _match_glob(_raw_options(root_pyproject)["git_describe_command"])

        assert framework_glob == "v[0-9]*"
        assert _glob_matches(framework_glob, "v2026.10.0")
        assert not _glob_matches(framework_glob, "osprey-connectors-v2026.10.0")
        assert not _glob_matches(framework_glob, "osprey-connectors-v0.1.0")


class TestLiveCheckout:
    """What this checkout's own tags make of the package."""

    def test_the_installed_version_is_on_the_calendar(self, connectors_glob):
        if _describe(connectors_glob) is None:
            pytest.skip(f"no tag matching {connectors_glob} reachable from this checkout")
        try:
            installed = dist_version("osprey-connectors")
        except PackageNotFoundError:
            pytest.skip("osprey-connectors is not installed, only importable from source")

        # 0.0.0.postN (no calendar tag reachable at build time) or 0.1.0.postN
        # (the retired tag read) both sort below the framework's floor.
        assert Version(installed) >= CALENDAR_FLOOR, installed

    def test_the_build_backend_derives_the_described_version(
        self, raw_options, connectors_glob, tag_regex
    ):
        """The backend's version is the describe's tag plus its distance, nothing else."""
        setuptools_scm = pytest.importorskip("setuptools_scm")
        described = _describe(connectors_glob)
        if described is None:
            pytest.skip(f"no tag matching {connectors_glob} reachable from this checkout")
        tag, distance, _node = described.rsplit("-", 2)
        found = tag_regex.match(tag)
        assert found is not None, tag

        options = dict(raw_options)
        root = PACKAGE_ROOT / options.pop("root")
        # setuptools-scm 10 deprecates the tag_regex keyword in favour of its own
        # configuration table; hatch-vcs passes it the same way, so this call is
        # the build's own.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            stamped = Version(setuptools_scm.get_version(root=str(root), **options))

        base = Version(found.group("version"))
        assert stamped.release == base.release
        assert stamped.pre == base.pre
        if int(distance):  # a dev build: post-release keeps the base and counts the distance
            assert stamped.post == int(distance)
        else:  # on the tag: exact, or .post0 for a dirty tree
            assert stamped.post in (None, 0)
