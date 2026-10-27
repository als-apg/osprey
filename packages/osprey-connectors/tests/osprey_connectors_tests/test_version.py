"""The package versions from its own ``osprey-connectors-v*`` calendar tags.

``[tool.hatch.version].raw-options`` in this package's ``pyproject.toml`` hands
the build backend a describe command, in the ``scm.git`` table setuptools-scm 10
reads it from. Its glob decides which tags the version can come from: only this
package's calendar tags, never the framework's ``v*`` tags (a second stream
would rebase onto the framework's numbers) and never the retired
``osprey-connectors-v0.1.0`` (every dev build would become ``0.1.0.postN``,
which sorts below the framework's floor). No tag regex is configured: the
backend's default one strips the ``osprey-connectors-`` prefix and the ``v``
itself, and the flat ``tag_regex`` key that would override it is deprecated. The
framework's own glob, read from the root ``pyproject.toml``, must in turn never
reach a connectors tag.

The checks on the glob are pure string checks; the ones that need the backend
skip where setuptools-scm is not installed (the framework-free lane). The ones
that read the repository skip outside this monorepo's checkout (an unpacked
sdist) and in a shallow clone, whose history may not reach a tag; a full
checkout that still describes no connectors calendar tag fails, because that is
the checkout that builds the package as ``0.0.postN+g…``.
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


def _pyproject(path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _raw_options(pyproject) -> dict:
    return _pyproject(pyproject)["tool"]["hatch"]["version"]["raw-options"]


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
def describe_command(raw_options) -> str:
    return raw_options["scm"]["git"]["describe_command"]


@pytest.fixture(scope="module")
def connectors_glob(describe_command) -> str:
    return _match_glob(describe_command)


@pytest.fixture(scope="module")
def tag_regex() -> re.Pattern[str]:
    """The backend's default tag regex, which is the one the build applies."""
    config = pytest.importorskip("vcs_versioning._config")
    return config.DEFAULT_TAG_REGEX


def _in_the_monorepo() -> bool:
    """Whether :data:`REPO_ROOT` is the monorepo this package sits in, not an sdist's parent."""
    return (REPO_ROOT / "packages" / "osprey-connectors").resolve() == PACKAGE_ROOT.resolve()


def _git(*args: str) -> subprocess.CompletedProcess[str] | None:
    """Run git in :data:`REPO_ROOT`, or None where git cannot run at all."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _described_tag(glob: str) -> str:
    """``git describe`` this checkout over *glob*.

    Skips outside the monorepo's git checkout and in a shallow clone; fails a
    full checkout that describes no matching tag, since that checkout builds
    the package off the calendar.
    """
    if not _in_the_monorepo():
        pytest.skip(f"{PACKAGE_ROOT} does not sit in the monorepo checkout")
    shallow = _git("rev-parse", "--is-shallow-repository")
    if shallow is None or shallow.returncode != 0:
        pytest.skip(f"{REPO_ROOT} is not a git checkout")
    if shallow.stdout.strip() != "false":
        pytest.skip("a shallow clone's history may not reach a connectors tag")
    described = _git("describe", "--tags", "--long", "--abbrev=40", "--match", glob)
    if described is None or described.returncode != 0:
        pytest.fail(
            f"no tag matching {glob} is reachable from this full checkout, so "
            "osprey-connectors builds as 0.0.postN+g… and the framework's floor "
            "refuses it. Run `git fetch --tags` (a mirror or fork must sync the "
            "osprey-connectors-v* tags); if the remote has none yet, push the "
            "bootstrap tag osprey-connectors-v2026.9.0b3."
        )
    return described.stdout.strip()


class TestTheDescribeGlob:
    def test_the_describe_command_is_the_pinned_one(self, describe_command):
        assert describe_command == (
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


class TestTheBackendConfiguration:
    """Nothing the build reads is a key setuptools-scm 10 deprecates."""

    def test_no_deprecated_flat_key_is_configured(self, raw_options):
        # Both are deprecated in vcs-versioning 2 and warn on every build; the
        # describe command lives in scm.git, and the default regex needs no help.
        assert "tag_regex" not in raw_options
        assert "git_describe_command" not in raw_options

    def test_the_build_requires_a_backend_that_reads_the_scm_table(self):
        requires = _pyproject(PACKAGE_ROOT / "pyproject.toml")["build-system"]["requires"]
        assert "setuptools-scm>=9.2" in requires


class TestTheTagRegex:
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
        # The regex keeps the `v`; the version parser drops it.
        assert Version(found.group("version")) == Version(version)
        assert Version(found.group("version")) >= CALENDAR_FLOOR

    def test_the_framework_tag_would_parse_too_so_the_glob_is_the_gate(self, tag_regex):
        # The default regex reads any dashed prefix, which is why the describe
        # glob, not the regex, keeps the framework's tags out of this stream.
        found = tag_regex.match("v2026.10.0")
        assert found is not None and found.group("version") == "v2026.10.0"


class TestTheFrameworkStream:
    def test_the_framework_glob_never_reads_a_connectors_tag(self):
        if not _in_the_monorepo():
            pytest.skip(f"{PACKAGE_ROOT} does not sit in the monorepo checkout")
        root_options = _raw_options(REPO_ROOT / "pyproject.toml")
        framework_glob = _match_glob(
            root_options.get("git_describe_command")
            or root_options["scm"]["git"]["describe_command"]
        )

        assert framework_glob == "v[0-9]*"
        assert _glob_matches(framework_glob, "v2026.10.0")
        assert not _glob_matches(framework_glob, "osprey-connectors-v2026.10.0")
        assert not _glob_matches(framework_glob, "osprey-connectors-v0.1.0")


class TestLiveCheckout:
    """What this checkout's own tags make of the package."""

    def test_the_installed_version_is_on_the_calendar(self, connectors_glob):
        """The installed version descends from the tag this checkout describes.

        Only the release segment and the pre-release are compared, never the
        distance: an editable install's metadata is stamped when it was last
        synced and may predate the newest commit. The same staleness limits
        what this proves — metadata stamped under the framework's old shared
        stream also reads ``2026.9.0b3.postN`` while that release is the newest
        tag of both streams, so it passes until the connectors stream moves on.
        """
        tag = _described_tag(connectors_glob).rsplit("-", 2)[0]
        try:
            installed = Version(dist_version("osprey-connectors"))
        except PackageNotFoundError:
            pytest.skip("osprey-connectors is not installed, only importable from source")
        # The glob admitted only osprey-connectors-v<calendar>; the version parser
        # drops the `v`, so this needs no backend (the framework-free lane has none).
        base = Version(tag.removeprefix("osprey-connectors-"))

        # 0.0.postN (no calendar tag reachable at build time) or 0.1.0.postN
        # (the retired tag read) both differ from the described tag.
        assert installed >= CALENDAR_FLOOR, installed
        assert (installed.release, installed.pre) == (base.release, base.pre), (
            f"installed {installed}, described {tag}: re-sync the environment"
        )

    def test_the_build_backend_derives_the_described_version(
        self, raw_options, connectors_glob, tag_regex
    ):
        """The backend's version is the describe's tag plus its distance, nothing else."""
        setuptools_scm = pytest.importorskip("setuptools_scm")
        tag, distance, _node = _described_tag(connectors_glob).rsplit("-", 2)
        found = tag_regex.match(tag)
        assert found is not None, tag

        options = dict(raw_options)
        root = PACKAGE_ROOT / options.pop("root")
        # hatch-vcs hands raw-options to get_version() as keywords, so this call
        # is the build's own — and it must not warn: a deprecated key would.
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            stamped = Version(setuptools_scm.get_version(root=str(root), **options))

        base = Version(found.group("version"))
        assert stamped.release == base.release
        assert stamped.pre == base.pre
        if int(distance):  # a dev build: post-release keeps the base and counts the distance
            assert stamped.post == int(distance)
        else:  # on the tag: exact, or .post0 for a dirty tree
            assert stamped.post in (None, 0)
