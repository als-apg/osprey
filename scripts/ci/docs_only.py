#!/usr/bin/env python3
"""Decide whether a pull request changes documentation and nothing else.

ci.yml's ``changes`` job pipes the pull request's changed paths into this
script and publishes the verdict as ``docs_only``. Every pull-request-scoped
lane (the e2e and deploy set) reads it and stands down when it is ``true``:
a page under ``docs/`` or a fragment under ``changelog.d/`` reaches no
container, no deployed stack and no model, so those lanes would spend about
ten runner-hours proving a tree the change did not touch.

The always-on lanes (unit tests, lint, the docs build, the browser suites,
...) carry no condition and are not affected. That is where the 70-odd tests
that read ``docs/`` run, so a docs-only change is still checked by them.

What counts as docs-only is deliberately narrow: every path, old and new name
alike for a rename, must sit under one of ``DOCS_PREFIXES``. Root files such as
``README.md`` are excluded, because the package build and the Dockerfile e2e
lane copy it. An empty list is not docs-only: a diff nobody could read must
never turn lanes off.

Usage::

    gh api --paginate "repos/$REPO/pulls/$PR/files" \\
        --jq '.[] | .filename, (.previous_filename // empty)' \\
        | python3 scripts/ci/docs_only.py >> "$GITHUB_OUTPUT"

Prints ``docs_only=true`` or ``docs_only=false`` on stdout, and the reason on
stderr. Stdlib-only, run with the runner's bare ``python3``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable

#: The only places a docs-only pull request may touch.
DOCS_PREFIXES = ("docs/", "changelog.d/")

#: GitHub's pull-request files endpoint stops listing at this many entries, so a
#: list this long may be truncated and cannot vouch for the rest of the diff.
API_FILE_LIMIT = 3000


def verdict(paths: Iterable[str]) -> tuple[bool, str]:
    """Return ``(docs_only, reason)`` for the changed *paths*."""
    changed = [path.strip() for path in paths if path.strip()]
    if not changed:
        return False, "no changed paths were read"
    if len(changed) >= API_FILE_LIMIT:
        return False, f"{len(changed)} paths listed; the API may have truncated the list"
    others = [path for path in changed if not path.startswith(DOCS_PREFIXES)]
    if others:
        return False, f"{len(others)} path(s) outside {', '.join(DOCS_PREFIXES)}, e.g. {others[0]}"
    return True, f"all {len(changed)} path(s) under {', '.join(DOCS_PREFIXES)}"


def main() -> int:
    docs_only, reason = verdict(sys.stdin)
    print(f"docs_only={'true' if docs_only else 'false'}")
    print(f"docs_only={docs_only}: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
