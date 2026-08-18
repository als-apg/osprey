"""Resolve the version of the running OSPREY framework.

This module is the single source every consumer reads from. It distinguishes two
values that are easy to conflate but have different jobs:

- :func:`get_running_version` — what this process actually *is*, including how far
  past the last release it sits (``2026.6.2.post783+g83fda5e60``). Everything shown
  to a human or stamped as informational metadata wants this.
- :func:`get_release_version` — the last shipped release (``2026.6.2``). PEP 440
  comparisons and PyPI pins want this, because a version with a local segment has no
  distribution behind it.

The version is derived from the git tag at build time (``hatch-vcs``), so a release
is cut by tagging rather than by editing a literal. At runtime it is resolved by the
first of these that answers:

1. ``git describe``, anchored to OSPREY's *own* source root. An editable install's
   stamp is written once when the wheel is rebuilt and is not refreshed by
   ``uv sync``, so a source checkout must ask git directly or it reports a stale
   version commit-to-commit.
2. ``osprey/_version.py``, stamped into the artifact at build time. This is what
   makes a wheel — and any container built from one — honest, where no git exists.
3. Installed distribution metadata.
4. A sentinel carrying a local segment, so a broken environment can never be
   mistaken for a release.

Step 1 is anchored rather than cwd-relative on purpose. ``osprey build`` runs
``git init`` inside the project it generates, and the Claude Code status line
imports osprey from a process whose working directory is the operator's own repo —
a cwd-relative probe would report someone else's tag as OSPREY's version.
"""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

__all__ = [
    "get_release_version",
    "get_running_version",
    "is_release",
    "unreleased_pin_reason",
]

#: Distribution name on PyPI. Note this is *not* the import name ``osprey``.
_DIST_NAME = "osprey-framework"

#: OSPREY's own source root: ``<root>/src/osprey/version.py`` -> ``<root>``. For an
#: installed copy this points into ``site-packages``' parent, which has no ``.git``,
#: so the git probe short-circuits without forking a subprocess.
_SOURCE_ROOT = Path(__file__).resolve().parents[2]

#: Emitted when nothing else answers. The local segment is deliberate: it keeps
#: :func:`is_release` false, so a broken environment fails closed instead of
#: claiming to be a release and pinning against a version that does not exist.
_UNKNOWN_VERSION = "0.0.0+unknown"

_GIT_TIMEOUT_SECONDS = 2


def _anchored_at_osprey_source() -> bool:
    """Report whether :data:`_SOURCE_ROOT` really is OSPREY's own git checkout.

    Two independent checks, both cheap. The ``.git`` test is a plain filesystem
    probe — it is what keeps installed copies from paying for a subprocess — and
    uses ``exists()`` rather than ``is_dir()`` because a git worktree's ``.git`` is
    a file pointing at the real git directory.

    Returns:
        True when the source root carries a git checkout that declares itself to be
        the ``osprey-framework`` distribution.
    """
    if not (_SOURCE_ROOT / ".git").exists():
        return False

    pyproject = _SOURCE_ROOT / "pyproject.toml"
    try:
        import tomllib

        with pyproject.open("rb") as handle:
            declared = tomllib.load(handle).get("project", {}).get("name")
    except (OSError, ValueError, ImportError):
        return False
    return bool(declared == _DIST_NAME)


def _pep440_from_describe(described: str) -> str | None:
    """Convert ``git describe`` output into the same string the build stamp holds.

    Mirrors ``setuptools-scm``'s ``post-release`` version scheme with the default
    ``node-and-date`` local scheme, so a given commit reports one version whether it
    is run from the source tree or from a wheel built out of it.

    Args:
        described: Output of ``git describe --tags --long --dirty``, e.g.
            ``v2026.6.2-783-g83fda5e60`` or ``v2026.6.2-0-g1234567-dirty``.

    Returns:
        A PEP 440 version string, or None when the output cannot be parsed.
    """
    dirty = described.endswith("-dirty")
    if dirty:
        described = described[: -len("-dirty")]

    tag, _, remainder = described.rpartition("-g")
    tag, _, distance = tag.rpartition("-")
    if not tag or not distance.isdigit() or not remainder:
        return None

    base = tag.removeprefix("v")
    if distance == "0" and not dirty:
        return base

    # setuptools-scm slices every node to a fixed width ("g" + 9 hex); mirror
    # it, or the build stamp and this string disagree on hash length alone.
    local = f"g{remainder[:9]}"
    if dirty:
        from datetime import date

        local = f"{local}.d{date.today():%Y%m%d}"
    return f"{base}.post{distance}+{local}"


def _version_from_git() -> str | None:
    """Derive the version from OSPREY's own checkout, or None if unavailable."""
    if not _anchored_at_osprey_source():
        return None

    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(_SOURCE_ROOT),
                "describe",
                "--tags",
                "--long",
                "--dirty",
                # Full node, sliced below. git's default abbreviation width
                # scales with the size of the clone, so left to it the same
                # commit reports g1234567a here and g1234567ab elsewhere.
                "--abbrev=40",
                "--match",
                "v[0-9]*",
            ],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        # No git binary on this host, or the call timed out. Fall through.
        return None

    if completed.returncode != 0:
        # A shallow clone or a repo with no matching tag reaches here.
        return None
    return _pep440_from_describe(completed.stdout.strip())


def _version_from_stamp() -> str | None:
    """Read the version stamped into the artifact by the build hook."""
    try:
        from osprey._version import __version__

        return str(__version__)
    except (ImportError, AttributeError):
        return None


def _version_from_metadata() -> str | None:
    """Read the version from installed distribution metadata."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as metadata_version

    try:
        return metadata_version(_DIST_NAME)
    except PackageNotFoundError:
        return None


@functools.cache
def get_running_version() -> str:
    """Return the version of the OSPREY currently executing.

    Includes distance past the last release and the commit, so a build from a
    development checkout is distinguishable from the release it descends from. This
    is the value to show a human or stamp as informational metadata.

    Returns:
        A PEP 440 version string, e.g. ``"2026.6.2"`` at a tag or
        ``"2026.6.2.post783+g83fda5e60"`` 783 commits past one. Never raises.
    """
    return (
        _version_from_git() or _version_from_stamp() or _version_from_metadata() or _UNKNOWN_VERSION
    )


@functools.cache
def get_release_version() -> str:
    """Return the last shipped release this build descends from.

    Strips the distance and commit that :func:`get_running_version` carries, leaving
    a version that exists on PyPI and compares cleanly under PEP 440. Use this for
    specifier checks and install pins — but pair it with :func:`is_release` before
    pinning, since from a development checkout it names a release whose code is
    *not* what is running.

    Returns:
        A release version string, e.g. ``"2026.6.2"``. Never raises.
    """
    from packaging.version import InvalidVersion, Version

    running = get_running_version()
    try:
        return Version(running).base_version
    except InvalidVersion:
        return running


def unreleased_pin_reason() -> str:
    """Explain why the running build cannot be pinned to a published release.

    Shared by every producer of an ``osprey-framework==`` requirement so they all
    refuse for the same stated reason. Callers supply their own remedy, since the
    way out differs by command.

    Returns:
        A one-sentence reason naming the running version and its distance from the
        last release.
    """
    from packaging.version import InvalidVersion, Version

    running = get_running_version()
    release = get_release_version()
    try:
        distance = Version(running).post
    except InvalidVersion:
        distance = None

    past = f", {distance} commits past v{release}" if distance else ""
    return (
        f"running {running} is not a released version{past}. A container built from "
        "PyPI would run different code than this checkout."
    )


def is_release() -> bool:
    """Report whether this build is exactly a tagged, clean release.

    False for anything carrying distance, a pre/post/dev segment, a local segment,
    or uncommitted changes — and for an environment where the version could not be
    resolved at all.

    Returns:
        True only when the running version is a published release.
    """
    from packaging.version import InvalidVersion, Version

    try:
        parsed = Version(get_running_version())
    except InvalidVersion:
        return False
    return not (
        parsed.is_devrelease or parsed.is_postrelease or parsed.is_prerelease or parsed.local
    )
