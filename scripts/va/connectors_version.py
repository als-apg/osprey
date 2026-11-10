#!/usr/bin/env python3
"""Print the version osprey-connectors builds as from this checkout.

The virtual accelerator image installs every workspace member from a staged
build context that has no ``.git``, so the host resolves each version and passes
it in as a build arg (see docker/virtual-accelerator/Containerfile). The
framework's comes from ``osprey.__version__``; this is the connectors package's,
which has its own calendar stream on its own ``osprey-connectors-v*`` tags.

The describe command is read from packages/osprey-connectors/pyproject.toml
itself (``[tool.hatch.version].raw-options.scm.git.describe_command``) and run
as written, so the glob here cannot drift from the one the build backend uses.
Its output is converted the way the backend's ``post-release`` version scheme
and default ``node-and-date`` local scheme convert it: the tag's version on the
tag, ``<tag>.post<distance>+g<node9>`` past it, ``.d<YYYYMMDD>`` (UTC) for a
dirty tree.

With no connectors calendar tag reachable (a shallow clone, or a repository
before the stream's first tag) it prints ``--fallback`` instead, with a warning
on stderr — the framework's number, which the framework's connectors floor
accepts, as the image carried before connectors had its own stream. Without a
fallback that case exits 1.

    python scripts/va/connectors_version.py [--fallback <version>]

Stdlib-only.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "packages" / "osprey-connectors" / "pyproject.toml"
TAG_PREFIX = "osprey-connectors-v"

#: PEP 440's pre-release spellings, for the normalizer used where ``packaging``
#: is not installed: ``2026.10.0-rc1`` / ``2026.10.0.RC.1`` -> ``2026.10.0rc1``.
_PRE_RE = re.compile(
    r"^(?P<release>\d+(?:\.\d+)*)[-_.]?(?P<letter>alpha|beta|preview|pre|rc|a|b|c)[-_.]?(?P<n>\d*)$"
)
_PRE_LETTER = {"alpha": "a", "beta": "b", "c": "rc", "pre": "rc", "preview": "rc"}


def normalize(version: str) -> str:
    """*version* in PEP 440 normal form, as the backend stamps it.

    ``packaging`` where it is installed (the project venv every caller runs
    this from); a minimal normalizer of release and pre-release segments
    otherwise, which is every form a calendar tag takes.
    """
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:
        match = _PRE_RE.match(version.lower())
        if match is None:
            return version.lower()
        letter = _PRE_LETTER.get(match["letter"], match["letter"])
        return f"{match['release']}{letter}{match['n'] or 0}"
    try:
        return str(Version(version))
    except InvalidVersion:
        return version


def describe_command(pyproject: Path = PYPROJECT) -> list[str]:
    """The describe argv the build backend runs for the package at *pyproject*.

    The connectors package's by default; a package's raw-options spell it under
    ``scm.git`` or the older flat key. A missing one is a ``KeyError``.
    """
    with pyproject.open("rb") as handle:
        options = tomllib.load(handle)["tool"]["hatch"]["version"]["raw-options"]
    nested = options.get("scm", {}).get("git", {}).get("describe_command")
    return shlex.split(nested or options["git_describe_command"])


def version_from_describe(
    described: str, *, prefix: str = TAG_PREFIX, today: datetime | None = None
) -> str | None:
    """``osprey-connectors-v2026.10.0-3-g<sha>[-dirty]`` -> ``2026.10.0.post3+g<sha9>``.

    None for anything that is not a calendar describe of the *prefix* stream.
    """
    dirty = described.endswith("-dirty")
    if dirty:
        described = described.removesuffix("-dirty")
    tag, _, node = described.rpartition("-g")
    tag, _, distance = tag.rpartition("-")
    if not tag.startswith(prefix) or not distance.isdigit() or not node:
        return None
    base = tag.removeprefix(prefix)
    if not base[:1].isdigit():
        return None
    base = normalize(base)
    if distance == "0" and not dirty:
        return base
    local = f"g{node[:9]}"
    if dirty:
        local += f".d{(today or datetime.now(UTC)):%Y%m%d}"
    return f"{base}.post{distance}+{local}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="connectors_version.py", description=__doc__)
    parser.add_argument("--fallback", help="printed when no connectors calendar tag is reachable")
    args = parser.parse_args(argv)

    command = describe_command()
    if command[:2] != ["git", "describe"]:
        print(f"not a git describe command: {shlex.join(command)}", file=sys.stderr)
        return 1
    done = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *command[1:]], capture_output=True, text=True
    )
    version = version_from_describe(done.stdout.strip()) if done.returncode == 0 else None
    if version is None:
        reason = done.stderr.strip() or f"unexpected describe output {done.stdout.strip()!r}"
        if args.fallback is None:
            print(f"no osprey-connectors calendar tag reachable: {reason}", file=sys.stderr)
            return 1
        print(
            f"warning: no osprey-connectors calendar tag reachable ({reason}); "
            f"stamping osprey-connectors {args.fallback} (git fetch --tags fixes this)",
            file=sys.stderr,
        )
        version = args.fallback
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
