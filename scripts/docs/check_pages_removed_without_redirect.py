#!/usr/bin/env python3
"""Guard: no docs page disappears without a redirect for its old URL.

The published site keeps every URL it has ever served. When a page is renamed,
moved into a hub, or folded into another page, ``sphinx_reredirects`` writes a
stub at the old URL from the ``redirects`` map in ``conf.py`` — but only if
someone remembered to add the entry. Nothing in a Sphinx build notices the
omission: the build is green and the old URL simply 404s.

This script compares the page set of a base ref (``origin/main`` by default)
against the working tree and fails when a page that vanished has no redirect
key. ``redirects`` is read out of ``conf.py`` with :mod:`ast` rather than
executed, so the check needs no Sphinx and no ``osprey`` import.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def redirect_keys(conf_path: Path) -> set[str]:
    """The keys of the ``redirects`` dict literal in *conf_path*."""
    for node in ast.walk(ast.parse(conf_path.read_text(encoding="utf-8"))):
        target = getattr(node, "target", None) if isinstance(node, ast.AnnAssign) else None
        if target is None and isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "redirects":
            if not isinstance(node.value, ast.Dict):
                raise SystemExit(f"{conf_path}: `redirects` is not a dict literal")
            return {ast.literal_eval(key) for key in node.value.keys}
    raise SystemExit(f"{conf_path}: no `redirects` assignment found")


def docnames_at(ref: str, repo_root: Path, prefix: str) -> set[str]:
    """Docnames of every ``.rst`` under *prefix* at *ref*."""
    listed = subprocess.run(
        ["git", "-C", str(repo_root), "ls-tree", "-r", "--name-only", ref, "--", prefix],
        capture_output=True,
        text=True,
    )
    if listed.returncode != 0:
        raise SystemExit(f"cannot list {ref}:{prefix}: {listed.stderr.strip()}")
    return {
        path[len(prefix) + 1 : -4]
        for path in listed.stdout.split()
        if path.endswith(".rst") and path.startswith(f"{prefix}/")
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check every page removed since the base ref has a redirect key."
    )
    parser.add_argument("--base", default="origin/main", help="ref to compare the tree against")
    parser.add_argument("--docs-root", type=Path, default=REPO_ROOT / "docs" / "source")
    parser.add_argument("--conf", type=Path, default=REPO_ROOT / "docs" / "source" / "conf.py")
    args = parser.parse_args(argv)

    docs_root = args.docs_root.resolve()
    prefix = docs_root.relative_to(REPO_ROOT).as_posix()
    before = docnames_at(args.base, REPO_ROOT, prefix)
    now = {
        page.relative_to(docs_root).with_suffix("").as_posix() for page in docs_root.rglob("*.rst")
    }
    removed = before - now
    offenders = sorted(removed - redirect_keys(args.conf))

    if offenders:
        conf_shown = args.conf.resolve()
        conf_shown = (
            conf_shown.relative_to(REPO_ROOT)
            if conf_shown.is_relative_to(REPO_ROOT)
            else conf_shown
        )
        print(
            "A page removed since the base ref must keep its old URL alive: add a key to\n"
            f"`redirects` in {conf_shown} for each of these docnames.\n"
        )
        for docname in offenders:
            print(f"  {prefix}/{docname}.rst")
        print(f"\n{len(offenders)} removed page(s) without a redirect (base {args.base})")
        return 1
    print(
        f"redirect coverage: OK ({len(removed)} page(s) removed since {args.base}, "
        f"all redirected; {len(before)} page(s) at the base)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
