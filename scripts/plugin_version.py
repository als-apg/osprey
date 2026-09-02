#!/usr/bin/env python3
"""Read and bump the CalVer version that OSPREY's two plugin manifests share.

The plugin ships to two marketplaces, so ``plugins/osprey/.claude-plugin/plugin.json``
and ``plugins/osprey/.codex-plugin/plugin.json`` each carry the same ``version`` string,
and a CI gate fails any pull request that edits the plugin tree without advancing it.
This script is the only writer of that string: ``show`` prints it and refuses to guess
when the two files disagree, ``bump`` advances ``YYYY.M.N`` in both at once.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Both manifests, relative to the repository root, in the order messages name them.
MANIFESTS = (
    Path("plugins/osprey/.claude-plugin/plugin.json"),
    Path("plugins/osprey/.codex-plugin/plugin.json"),
)

#: CalVer, ``YYYY.M.N``. The month is never zero-padded on the way out; ``--set``
#: tolerates a padded one because the next bump reads it as an int and normalizes it.
VERSION = re.compile(r"^\d{4}\.\d{1,2}\.\d+$")


def read_versions(root: Path) -> dict[Path, str]:
    """Map each manifest's relative path to the version string it carries."""
    return {
        path: json.loads((root / path).read_text(encoding="utf-8"))["version"] for path in MANIFESTS
    }


def next_version(current: str, today: date) -> str:
    """Advance *current*: the same calendar month bumps N, a new month resets N to 0."""
    year, month, serial = (int(part) for part in current.split("."))
    if (year, month) == (today.year, today.month):
        return f"{year}.{month}.{serial + 1}"
    return f"{today.year}.{today.month}.0"


def write_version(root: Path, version: str) -> None:
    """Rewrite ``version`` in both manifests, keeping key order, indent and final newline."""
    for path in MANIFESTS:
        target = root / path
        manifest = json.loads(target.read_text(encoding="utf-8"))
        manifest["version"] = version
        target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Parse *argv*, run the subcommand, and return the process exit code."""
    parser = argparse.ArgumentParser(
        prog="plugin_version.py",
        description="Show or bump the CalVer version shared by OSPREY's two plugin manifests.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository root holding plugins/osprey (default: this checkout)",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("show", help="print the shared version")
    bump = subcommands.add_parser("bump", help="advance the version in both manifests")
    bump.add_argument(
        "--set", dest="explicit", metavar="YYYY.M.N", help="write this version instead"
    )
    bump.add_argument("--today", type=date.fromisoformat, help=argparse.SUPPRESS)

    args = parser.parse_args(argv)
    explicit = getattr(args, "explicit", None)
    if explicit is not None and not VERSION.match(explicit):
        print(f"--set {explicit}: not CalVer YYYY.M.N", file=sys.stderr)
        return 2
    try:
        versions = read_versions(args.root)
    except (OSError, KeyError, json.JSONDecodeError) as error:
        print(f"cannot read the plugin manifests under {args.root}: {error}", file=sys.stderr)
        return 1

    distinct = sorted(set(versions.values()))
    if len(distinct) > 1 and explicit is None:
        skew = ", ".join(f"{path} says {version}" for path, version in versions.items())
        print(f"version skew: {skew} — repair with: bump --set YYYY.M.N", file=sys.stderr)
        return 1
    if args.command == "show":
        print(distinct[0])
        return 0
    if explicit is None and not VERSION.match(distinct[0]):
        print(f"{distinct[0]} is not CalVer; write one with: bump --set", file=sys.stderr)
        return 1
    new = explicit or next_version(distinct[0], args.today or date.today())
    write_version(args.root, new)
    print(f"{' and '.join(distinct)} -> {new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
