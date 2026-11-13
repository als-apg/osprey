#!/usr/bin/env python3
"""What the e2e lanes run, derived from the lanes check.py collects.

Every ``tests/e2e`` test module runs in exactly one place: the dedicated lane
whose ``Lane`` claims it, nowhere (``E2E_OFF_CI``), or the shared lane, which
takes every module nothing else claims. ``repo_checks.e2e_hosts`` computes that
map once; the shared lane's ``--ignore`` list and the invariant test both read it.
A dedicated lane that names its modules can take its pytest paths from here too.

Subcommands:

    ignores   prints one ``--ignore=<module>`` per line on stdout, the
              argument list the shared pytest takes, and the module -> host
              table on stderr, so a CI log shows what the lane left out.
              An empty list is refused: the shared lane would run every
              other host's modules too.
    stacks    prints the compose project (``PROJECT_NAME``) of each module the
              run selected: the ``E2E_FILTER`` paths on a dispatch that sets
              one, the shared sweep otherwise. The cleanup step removes those.
    modules LANE
              prints the test modules lane LANE names, one per line, as
              declared: a module that is gone reaches pytest and exits it 4.
              A lane that names none is refused: pytest handed no path
              collects the whole tree.

``ignores`` and ``modules`` exit 2 rather than print nothing, so under ``bash -e``
the step stops at the redirecting command, before ``mapfile`` reads an empty file.
``stacks`` may print nothing: a run that selected no stack removes none.

Standard library and the repository's own ``scripts/`` modules only, so it runs
under the runner's ``python3``.

Usage:
    python3 scripts/ci/e2e_shared_lane.py ignores > "$RUNNER_TEMP/e2e-ignores"
    E2E_FILTER='tests/e2e/test_dispatch_deploy.py' python3 scripts/ci/e2e_shared_lane.py stacks
    python3 scripts/ci/e2e_shared_lane.py modules e2e-no-model > "$RUNNER_TEMP/e2e-modules"
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
from check_kit import Lane

FW = check.FRAMEWORK


def ignores(files: Sequence[str]) -> list[str]:
    """The shared lane's ``--ignore`` arguments, one per module another host runs."""
    return [f"--ignore={path}" for path in FW.e2e_shared_ignores(check.LANES, files)]


def modules(lane_id: str, lanes: Sequence[Lane] = check.LANES) -> list[str]:
    """The test modules lane *lane_id* names, in declaration order, as pytest paths.

    Read as declared, not resolved against the tree, so a module that is gone
    still reaches pytest. A lane that names no modules, or names them by glob,
    has no such list and is refused.
    """
    named = [lane for lane in lanes if lane.id == lane_id]
    if len(named) != 1:
        raise ValueError(f"{len(named)} lanes have the id {lane_id!r}")
    paths = named[0].modules
    if not paths or any(c in path for path in paths for c in "*?["):
        raise ValueError(f"lane {lane_id!r} names no module list of plain paths")
    return list(paths)


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
    parser.add_argument("command", choices=("ignores", "stacks", "modules"))
    parser.add_argument("lane", nargs="?", help="the lane id, for modules")
    args = parser.parse_args(argv)
    if (args.command == "modules") != (args.lane is not None):
        parser.error("modules takes a lane id, and only modules does")
    if args.command == "modules":
        try:
            lines = modules(args.lane, check.LANES)
        except ValueError as refusal:
            parser.error(str(refusal))
    elif args.command == "ignores":
        files = check.repository_files()
        hosts = FW.e2e_hosts(check.LANES, files)
        left = ignores(files)
        if not left:
            parser.error("derived no --ignore list: the shared lane would run every module")
        print(f"{FW.SHARED_E2E_LANE} leaves {len(left)} modules to their hosts:", file=sys.stderr)
        for arg in left:
            path = arg.removeprefix("--ignore=")
            print(f"  {path}  ({', '.join(hosts[path])})", file=sys.stderr)
        lines = left
    else:
        lines = stacks(check.repository_files(), os.environ.get("E2E_FILTER", ""))
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
