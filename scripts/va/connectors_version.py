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
import shlex
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "packages" / "osprey-connectors" / "pyproject.toml"
TAG_PREFIX = "osprey-connectors-v"


def describe_command(pyproject: Path = PYPROJECT) -> list[str]:
    """The describe argv the build backend runs for the connectors package."""
    with pyproject.open("rb") as handle:
        options = tomllib.load(handle)["tool"]["hatch"]["version"]["raw-options"]
    return shlex.split(options["scm"]["git"]["describe_command"])


def version_from_describe(described: str, *, today: datetime | None = None) -> str | None:
    """``osprey-connectors-v2026.10.0-3-g<sha>[-dirty]`` -> ``2026.10.0.post3+g<sha9>``.

    None for anything that is not a connectors calendar describe.
    """
    dirty = described.endswith("-dirty")
    if dirty:
        described = described.removesuffix("-dirty")
    tag, _, node = described.rpartition("-g")
    tag, _, distance = tag.rpartition("-")
    if not tag.startswith(TAG_PREFIX) or not distance.isdigit() or not node:
        return None
    base = tag.removeprefix(TAG_PREFIX)
    if not base[:1].isdigit():
        return None
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
