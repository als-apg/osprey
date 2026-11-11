#!/usr/bin/env python3
"""What the shared ``e2e-tests`` lane runs, derived from the lanes check.py collects.

Every ``tests/e2e`` test module runs in exactly one place: the dedicated lane
whose ``Lane`` claims it, nowhere (``E2E_OFF_CI``), or the shared lane, which
takes every module nothing else claims. ``repo_checks.e2e_hosts`` computes that
map once; the shared lane's ``--ignore`` list and the invariant test both read it.

Subcommands:

    ignores   prints one ``--ignore=<module>`` per line on stdout, the
              argument list the shared pytest takes, and the module -> host
              table on stderr, so a CI log shows what the lane left out
    stacks    prints the compose project (``PROJECT_NAME``) of each module the
              run selected: the ``E2E_FILTER`` paths on a dispatch that sets
              one, the shared sweep otherwise. The cleanup step removes those.

Standard library and the repository's own ``scripts/`` modules only, so it runs
under the runner's ``python3``.

Usage:
    python3 scripts/ci/e2e_shared_lane.py ignores > "$RUNNER_TEMP/e2e-ignores"
    E2E_FILTER='tests/e2e/test_dispatch_deploy.py' python3 scripts/ci/e2e_shared_lane.py stacks
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from collections.abc import Sequence
from pathlib import Path

# scripts/ is not a package; check.py and the checks modules import by name from it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check

FW = check.FRAMEWORK


def ignores(files: Sequence[str]) -> list[str]:
    """The shared lane's ``--ignore`` arguments, one per module another host runs."""
    return [f"--ignore={path}" for path in FW.e2e_shared_ignores(check.LANES, files)]


def selected(files: Sequence[str], e2e_filter: str) -> list[str]:
    """The test modules a run selects: the filter's paths, or the shared sweep.

    The filter is split on whitespace, as the run step's unquoted
    ``$E2E_FILTER`` is; a token that is no tracked path (``-k``, a keyword) is
    passed over, a node id is read as its file and a directory as its modules.
    """
    if not e2e_filter.strip():
        hosts = FW.e2e_hosts(check.LANES, files)
        return [path for path, host in hosts.items() if host == (FW.SHARED_E2E_LANE,)]
    chosen: list[str] = []
    for token in e2e_filter.split():
        path = token.partition("::")[0].rstrip("/")
        chosen += [
            f
            for f in files
            if (f == path or f.startswith(f"{path}/"))
            and f.endswith(".py")
            and Path(f).name.startswith("test_")
        ]
    return list(dict.fromkeys(chosen))


def project_name(path: Path) -> str | None:
    """The module's top-level ``PROJECT_NAME`` string, if it assigns one."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "PROJECT_NAME" for t in node.targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    return None


def stacks(files: Sequence[str], e2e_filter: str, root: Path = check.ROOT) -> list[str]:
    """The compose projects of the modules the run selects, in selection order."""
    names = (project_name(root / path) for path in selected(files, e2e_filter))
    return list(dict.fromkeys(name for name in names if name))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("command", choices=("ignores", "stacks"))
    args = parser.parse_args(argv)
    files = check.repository_files()
    if args.command == "ignores":
        hosts = FW.e2e_hosts(check.LANES, files)
        left = ignores(files)
        print(f"{FW.SHARED_E2E_LANE} leaves {len(left)} modules to their hosts:", file=sys.stderr)
        for arg in left:
            path = arg.removeprefix("--ignore=")
            print(f"  {path}  ({', '.join(hosts[path])})", file=sys.stderr)
        lines = left
    else:
        lines = stacks(files, os.environ.get("E2E_FILTER", ""))
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
