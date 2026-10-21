#!/usr/bin/env python3
"""Read, bump and check the CalVer version that OSPREY's two plugin manifests share.

The plugin ships to two marketplaces, so ``plugins/osprey/.claude-plugin/plugin.json``
and ``plugins/osprey/.codex-plugin/plugin.json`` each carry the same ``version`` string.
This script is the only writer of that string and the one statement of the bump rule:
``show`` prints it and refuses to guess when the two files disagree, ``bump`` advances
``YYYY.M.N`` in both at once, and ``check --base <ref>`` refuses a branch that changes
the plugin tree while still carrying a version the branch it merges into already has.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable
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

#: The paths, relative to the repository root, whose change owes a version bump.
PLUGIN_TREE = ("plugins", ".claude-plugin", ".agents/plugins")

#: ``check``'s answer when the branch leaves every path of ``PLUGIN_TREE`` alone.
UNTOUCHED = "plugin tree untouched — no version bump required"

#: ``check``'s answer when the base carries no manifest: the branch adds the plugin.
NEW_MANIFEST = "new plugin manifest — no bump required"

#: ``check``'s refusal when the head version is one the base already carries.
UNCHANGED = "plugin tree changed without a version bump — run python scripts/plugin_version.py bump"

#: A plain ref — letters, digits, ``.``, ``_``, ``-`` and nothing else — is read as a
#: branch name, the same reading the changelog gate gives its ``--base``.
PLAIN_REF = re.compile(r"^[A-Za-z0-9._-]+$")

#: One git invocation: argv in, a finished process out.
Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def read_versions(root: Path) -> dict[Path, str]:
    """Map each manifest's relative path to the version string it carries."""
    return {
        path: json.loads((root / path).read_text(encoding="utf-8"))["version"] for path in MANIFESTS
    }


def unreadable(root: Path, error: Exception) -> str:
    """The sentence for head manifests that cannot be read."""
    return f"cannot read the plugin manifests under {root}: {error}"


def skewed(versions: dict[Path, str]) -> str:
    """The sentence for head manifests that carry two different versions."""
    skew = ", ".join(f"{path} says {version}" for path, version in versions.items())
    return f"version skew: {skew} — repair with: bump --set YYYY.M.N"


def resolve_base(ref: str, run: Runner) -> str | None:
    """Resolve *ref* to a commit, preferring the remote for a plain branch name.

    The same resolution as the changelog gate, so one sweep reads one base: a plain
    name other than ``HEAD`` tries ``origin/<ref>`` first and the local name second,
    because in a fresh CI checkout the only copy of a branch is the remote one;
    anything else (``HEAD~1``, a sha, a name that already says ``origin/``) is taken
    at its word first and only then looked for under ``origin/``.

    Returns the commit sha, or ``None`` when neither spelling resolves.
    """
    plain = PLAIN_REF.match(ref) is not None and ref != "HEAD"
    candidates = [f"origin/{ref}", ref] if plain else [ref, f"origin/{ref}"]
    for candidate in candidates:
        result = run(["git", "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}"])
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


def git_runner(root: Path) -> Runner:
    """The runner ``check`` uses unless given one: every git call runs in *root*."""

    def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", cwd=root)

    return run


def check(root: Path, base: str, run: Runner | None = None) -> int:
    """Refuse a changed plugin tree whose version the base tip already carries.

    Exit 0 when the tree is untouched, new, or carries a version the base tip does
    not; exit 1 when the check ran and refuses; exit 2 when git could not answer
    and the check never ran.
    """
    if run is None:
        run = git_runner(root)
    # An unresolved base is never read as an untouched tree.
    base_sha = resolve_base(base, run)
    if base_sha is None:
        print(
            f"base ref {base} not found — git fetch origin <branch>; "
            f"in CI, actions/checkout needs fetch-depth: 0",
            file=sys.stderr,
        )
        return 2
    merge_base = run(["git", "merge-base", base_sha, "HEAD"])
    if merge_base.returncode != 0 or not merge_base.stdout.strip():
        print(
            f"git merge-base {base_sha} HEAD found no common ancestor — shallow clone? "
            f"git fetch --unshallow\n{merge_base.stderr.strip()}",
            file=sys.stderr,
        )
        return 2
    diff_argv = ["git", "diff", "--name-only", merge_base.stdout.strip(), "HEAD", "--"]
    diff_argv += PLUGIN_TREE
    diff = run(diff_argv)
    if diff.returncode != 0:
        print(f"{' '.join(diff_argv)} failed\n{diff.stderr.strip()}", file=sys.stderr)
        return 2
    # The diff is answered before any manifest is read: an untouched tree owes nothing,
    # whatever state the working tree's manifests are in.
    if not diff.stdout.strip():
        print(UNTOUCHED)
        return 0

    try:
        head_versions = read_versions(root)
    except (OSError, KeyError, json.JSONDecodeError) as error:
        print(unreadable(root, error), file=sys.stderr)
        return 1
    if len(set(head_versions.values())) > 1:
        print(skewed(head_versions), file=sys.stderr)
        return 1
    head = next(iter(head_versions.values()))

    # The diff runs against the merge base but the versions are compared against the
    # base tip, so a base that minted the same version after the branch point is
    # refused. A manifest absent at the tip is a new plugin; one present that does not
    # parse is an error, not a free pass.
    base_versions: set[str] = set()
    for path in MANIFESTS:
        shown = run(["git", "show", f"{base_sha}:{path.as_posix()}"])
        if shown.returncode != 0:
            continue
        try:
            base_versions.add(json.loads(shown.stdout)["version"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            print(f"cannot read {path} at {base}: {error}", file=sys.stderr)
            return 1
    if not base_versions:
        print(NEW_MANIFEST)
        return 0
    # Inequality, not ordering: any version the base tip does not carry passes.
    if head in base_versions:
        print(UNCHANGED)
        return 1
    print(f"plugin version {' and '.join(sorted(base_versions))} -> {head}")
    return 0


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
        description="Show, bump or check the CalVer version shared by OSPREY's two plugin manifests.",
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
    check_parser = subcommands.add_parser(
        "check", help="refuse a changed plugin tree that carries the base's version"
    )
    check_parser.add_argument(
        "--base", required=True, metavar="REF", help="the ref this branch merges into"
    )

    args = parser.parse_args(argv)
    explicit = getattr(args, "explicit", None)
    if explicit is not None and not VERSION.match(explicit):
        print(f"--set {explicit}: not CalVer YYYY.M.N", file=sys.stderr)
        return 2
    if args.command == "check":
        return check(args.root, args.base)
    try:
        versions = read_versions(args.root)
    except (OSError, KeyError, json.JSONDecodeError) as error:
        print(unreadable(args.root, error), file=sys.stderr)
        return 1

    distinct = sorted(set(versions.values()))
    if len(distinct) > 1 and explicit is None:
        print(skewed(versions), file=sys.stderr)
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
