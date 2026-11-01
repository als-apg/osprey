#!/usr/bin/env python3
"""Answer "is this change clean" with one command, from one declaration.

A change is clean when every check below that applies to it ran here and passed.
Each check is declared once, by the code that owns it, and every caller runs it
from there::

    check.py quick [--changed BASE [--worktree] | --base BASE] [--fix] [-- PATH...]
    check.py full  [--changed BASE [--worktree] | --base BASE]         [-- PATH...]
    check.py run ID... [--base BASE] [--shard cli|rest-a|rest-b] [-- EXEC-FLAGS...]
    check.py list [--tier quick|full] [--changed BASE] [--json]

An owner declares its checks in a ``repo_checks.py`` module beside its code:
``scripts/repo_checks.py`` for the framework and the repository as a whole, and
one in each package or integration that brings checks of its own. This command
collects every such module the tree holds, the framework's first, and owns no
check itself. A module names its checks in ``CHECKS``, built from the records in
``scripts/check_kit.py``; each record carries everything this command does with
it -- its command, its scope or how a range selects its tests, its tiers, what
it needs, its bound, its report, its hooks. A module may also name ``LANES`` (CI
jobs this command never runs), ``CALLER_JOBS`` (CI jobs that call it),
``NEEDS`` and ``VENUES`` (the probes its checks' needs and venues name), and
``LISTED`` (what ``list --json`` prints beside the checks).

Four kinds of check:

* ``invariant`` -- holds for the repository as a whole and always runs: the
  lockfile, ruff, the config-key guard, the changelog and plugin-version gates,
  the rule that no tracked file is gitignored, the secret scan, and the tests
  that pin this declaration.
* ``static`` -- one per language or rule: mypy, the import contracts, the
  private-name imports, tsc, eslint, prettier, actionlint.
* ``tests`` -- one per runner: the parallel pytest sweep, the serial
  real-terminal and wall-clock suites, the live Channel Access gate, Vitest,
  and the dock-engine's own Vitest suite.
* ``build`` -- the documentation and its links, the dock-engine bundle, and the
  packages.

Two tiers. ``quick`` is what a commit needs, ``full`` is what a push needs and
adds the serial suites and the builds. ``run`` runs named checks as declared and
is what a CI step calls; flags after ``--`` are execution flags only (workers,
verbosity, coverage, timeouts), because a selection flag would make the caller a
second declaration.

``--changed BASE`` scopes a tier to the range ``BASE...HEAD``, or with
``--worktree`` to the merge base against the working tree, and then also hands
every check that judges a range the working tree (staged, unstaged and
untracked), so one verdict judges one range. A check whose scope
the range does not touch is reported as not touched; a path the range deleted or
moved away touches the scope it left. The test runners get only the tests the
changed paths map to. A path the mapping cannot place widens that runner to its
whole selection; scoping never narrows. A script a check runs through, and a
checks module, selects ``tests/scripts/``; a file under ``tests/<dir>/`` selects
that directory when it holds a test module and widens otherwise. A base also arms the
checks that judge a range. Paths after ``--`` are test files, pytest node ids
or test directories added to what the range selects.

``list --changed BASE`` adds which checks and CI lanes the range ``BASE...HEAD``
touches, read as a tier reads it; it runs nothing.

``--base BASE`` arms every check that judges a range against ``BASE`` and leaves
the runners on their whole selection, so ``full --base BASE`` judges the whole
tree and the range in one verdict. ``--changed`` and ``--base`` may not name
different bases.

Exit status: 0 clean, 1 a check failed, 2 refused, 3 this machine cannot judge.
A refusal means the command could not judge: a missing toolchain, a base that
does not resolve, an unknown check, a flag that is not an execution flag. Every
check of the run is attempted even after an earlier one failed, and the last
line is the verdict. A failed test check names its failing tests, read from the
report its runner writes for the command. A refusal names every check it cannot
judge and what to install.
An environment fault stops the run before any check starts and prints one last
line, ``check: environment: <probe>: <fact>``: too little free space, an install
a re-sync could not make current, Docker storage that does not work.

Before any check runs, the needs put the tree's pins in place: the pinned
Claude Code first on PATH, node_modules reinstalled when it differs from
package-lock.json, and a stale editable install re-synced.

What cannot run here is named, never counted as passed: a check bound to a venue
this machine is not, a check whose install is absent, a check that needs a base
when none was given, and the CI lanes this command never runs. A check that
proves something only against one install refuses any other install, and under
``run`` refuses its absence too; when it runs, the verdict names the install it
ran against.

Standard library only: this runs before the project's dependencies are installed
and must still report when installing them is what broke. So does every checks
module.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from xml.etree import ElementTree


def _load(path: Path, name: str) -> ModuleType:
    """The module at *path*, registered as *name* before it runs, so its
    dataclasses resolve their module and a later import of *name* finds it."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Loaded by path, not imported: scripts/ is not a package, and an isolated
# interpreter does not put it on the path. A checks module imports it by name.
kit = sys.modules.get("check_kit") or _load(
    Path(__file__).resolve().with_name("check_kit.py"), "check_kit"
)

ROOT = kit.ROOT
TIERS, KINDS = kit.TIERS, kit.KINDS
CLEAN, FAILED_EXIT, REFUSED_EXIT = kit.CLEAN, kit.FAILED_EXIT, kit.REFUSED_EXIT
ENVIRONMENT_EXIT, EnvironmentFault = kit.ENVIRONMENT_EXIT, kit.EnvironmentFault
BOTH, FULL = kit.BOTH, kit.FULL
WHOLE, PATHS, NOT_TOUCHED = kit.WHOLE, kit.PATHS, kit.NOT_TOUCHED
ABSENT_EXIT = kit.ABSENT_EXIT
Check, Lane, Install, Venue = kit.Check, kit.Lane, kit.Install, kit.Venue
Selection, Context, Refused = kit.Selection, kit.Context, kit.Refused
matches, touches = kit.matches, kit.touches

#: The name every checks module carries.
OWNER_MODULE = "repo_checks.py"

#: The framework's checks module, collected first: its checks hold for the
#: repository as a whole, and the lockfile is checked before any ``uv run``.
FRAMEWORK_MODULE = f"scripts/{OWNER_MODULE}"


@dataclass(frozen=True)
class Declaration:
    """What every checks module declares, collected in module order."""

    checks: tuple[Check, ...]
    lanes: tuple[Lane, ...]
    caller_jobs: tuple[str, ...]
    needs: dict[str, Callable[[], str | None]]
    venues: dict[str, Venue]
    listed: dict[str, object]
    #: Each check id to the module that declares it, repo-relative.
    owners: dict[str, str]
    #: Each module, by its repo-relative path.
    modules: dict[str, ModuleType]


def owner_modules(root: Path) -> list[str]:
    """Every checks module the tree at *root* holds, tracked or not, the
    framework's first and the rest in path order."""
    done = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--"]
        + [f":(glob)**/{OWNER_MODULE}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise Refused(f"cannot list the checks modules: {done.stderr.strip()}")
    found = {path for path in done.stdout.split("\0") if path}
    if FRAMEWORK_MODULE not in found:
        raise Refused(f"{FRAMEWORK_MODULE} is missing: the framework declares no check")
    return [FRAMEWORK_MODULE, *sorted(found - {FRAMEWORK_MODULE})]


def _merged(kind: str, owned: Iterable[tuple[str, Mapping[str, object]]]) -> dict[str, object]:
    """One mapping from every module's, refusing a key two modules declare."""
    merged: dict[str, object] = {}
    owner: dict[str, str] = {}
    for path, mapping in owned:
        for key, value in mapping.items():
            if key in merged:
                raise Refused(f"{kind} {key} is declared by both {owner[key]} and {path}")
            merged[key], owner[key] = value, path
    return merged


def collect(paths: Sequence[str], root: Path) -> Declaration:
    """Load each checks module at *paths* under *root* and merge what they declare.

    A check id, a lane, a need or a venue declared by two modules is refused:
    each has one owner.
    """
    modules = {
        path: _load(root / path, "_repo_checks_" + re.sub(r"\W", "_", path)) for path in paths
    }
    for path, module in modules.items():
        if not hasattr(module, "CHECKS"):
            raise Refused(f"{path} declares no CHECKS")

    def owned(name: str) -> list[tuple[str, Mapping[str, object]]]:
        return [(path, getattr(module, name, {})) for path, module in modules.items()]

    def each(name: str) -> list[tuple[str, object]]:
        return [(path, item) for path, m in modules.items() for item in getattr(m, name, ())]

    checks = each("CHECKS")
    owners = _merged("check", [(path, {c.id: path}) for path, c in checks])
    lanes = each("LANES")
    _merged("lane", [(path, {lane.job: path}) for path, lane in lanes])
    return Declaration(
        checks=tuple(c for _path, c in checks),
        lanes=tuple(lane for _path, lane in lanes),
        caller_jobs=tuple(dict.fromkeys(job for _path, job in each("CALLER_JOBS"))),
        needs=_merged("need", owned("NEEDS")),
        venues=_merged("venue", owned("VENUES")),
        listed=_merged("listed key", owned("LISTED")),
        owners={check_id: str(path) for check_id, path in owners.items()},
        modules=modules,
    )


try:
    DECLARED = collect(owner_modules(ROOT), ROOT)
except Refused as refusal:
    sys.exit(f"check: refused: {refusal}")

CHECKS: tuple[Check, ...] = DECLARED.checks
LANES: tuple[Lane, ...] = DECLARED.lanes
#: The CI jobs that call this command; every other required job is a lane.
CALLER_JOBS: tuple[str, ...] = DECLARED.caller_jobs
#: Each need to its probe: ``None`` when met, else the refusal. ``base`` is not a
#: probe, it is an argument.
NEED_PROBES = DECLARED.needs
VENUES = DECLARED.venues
#: The framework's checks module.
FRAMEWORK = DECLARED.modules[FRAMEWORK_MODULE]

#: Every command ``--fix`` can run, in declaration order; the checks never write.
FIX_COMMANDS: tuple[tuple[str, ...], ...] = tuple(argv for c in CHECKS for argv in c.fix)

#: What a failed check that declares a fix tells its reader to run.
FIX_HINT = "uv run python scripts/check.py quick --fix"


#: Each report to the flags that make its runner write it; appended last on the line.
REPORT_FLAGS: dict[str, tuple[str, ...]] = {
    # xunit1 is the family whose file attribute rebuilds the node id exactly.
    "junit-pytest": ("--junitxml={reports}/{id}/junit.xml", "-o", "junit_family=xunit1"),
    "junit-vitest": (
        "--reporter=default",
        "--reporter=junit",
        "--outputFile.junit={reports}/{id}/junit.xml",
    ),
}


def _by_id() -> dict[str, Check]:
    return {check.id: check for check in CHECKS}


@dataclass(frozen=True)
class Plan:
    """A tier resolved against a range: one selection per check, and the lanes to name."""

    tier: str
    selections: dict[str, Selection]
    lanes: tuple[str, ...]
    scoped: bool


def probe_install(install: Install) -> tuple[int, str]:
    """The exit status of *install*'s probe and the line it printed last."""
    status, out = kit.capture(install.probe)
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    said = lines[-1] if lines else f"{shlex.join(install.probe)} exited {status}"
    return status, said


def require_base(base: str) -> None:
    """Refuse a base that names no commit."""
    done = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}"],
        cwd=kit.ROOT,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise Refused(f"{base} does not resolve to a commit: fetch it or name another base")


# ---------------------------------------------------------------------------
# Selection


def _range(base: str, worktree: bool) -> str:
    """The git revision argument for the range: committed state, or the working tree."""
    if worktree:
        return kit.git("merge-base", base, "HEAD").strip()
    return f"{base}...HEAD"


def touched(base: str, worktree: bool = False, *, scope: bool = False) -> list[str]:
    """The paths the range changed and still holds, repo-relative.

    Committed state is ``BASE...HEAD``. With *worktree* it is the merge base
    against the working tree, so staged, unstaged and untracked files count.

    With *scope* it is every path the range changed, the deleted ones too, and
    a move names both the path it left and the one it reached: a change that
    takes a file out of a scope touches that scope. A runner can be handed
    only a file that is there, so its selection reads the default form.
    """
    listing = ("--name-only", "--no-renames") if scope else ("--name-only", "--diff-filter=d")
    paths = set(kit.git_paths("diff", *listing, _range(base, worktree)))
    if worktree:
        paths.update(kit.git_paths("ls-files", "--others", "--exclude-standard"))
    return sorted(paths)


def repository_files(worktree: bool = False) -> list[str]:
    """Every tracked path, plus the untracked ones when the working tree is judged."""
    files = kit.git_paths("ls-files")
    if worktree:
        files += kit.git_paths("ls-files", "--others", "--exclude-standard")
    return files


def _paths(chosen: Iterable[str]) -> Selection:
    unique = tuple(dict.fromkeys(chosen))
    return Selection(PATHS, unique) if unique else Selection(NOT_TOUCHED)


def _route_named(named: Sequence[str]) -> dict[str, list[str]]:
    """Send each named path to the check that can take it.

    A file goes to the check whose test suffixes it carries. A node id
    (``file::node``) goes there only when that check takes node ids, and its
    file must exist. A directory goes, as the directory, to a check that takes
    directories and has a test file in it.
    """
    takers = [check for check in CHECKS if check.named_suffixes or check.named_dirs]
    routed: dict[str, list[str]] = {check.id: [] for check in takers}
    for path in named:
        file, sep, _ = path.partition("::")
        if sep:
            routed[_node_id_taker(path, file, takers).id].append(path)
            continue
        taker = next((c for c in takers if path.endswith(c.named_suffixes)), None)
        if taker is None and any(c.named_dirs for c in takers) and (ROOT / path).is_dir():
            taker = _directory_taker(path, takers)
            path = path.rstrip("/")
        if taker is None:
            raise Refused(f"{path}: not a test file; name {_spelled(_named_forms(takers))}")
        routed[taker.id].append(path)
    return routed


def _spelled(items: Sequence[str]) -> str:
    return ", ".join(items[:-1]) + f" or {items[-1]}" if len(items) > 1 else items[0]


def _named_forms(takers: Sequence[Check]) -> list[str]:
    """Every form of path *takers* accept after ``--``, spelled for a refusal."""
    forms = [f"a {suffix} file" for c in takers for suffix in c.named_suffixes]
    forms += [
        f"a {suffix}::node id" for c in takers if c.named_node_ids for suffix in c.named_suffixes
    ]
    if any(c.named_dirs for c in takers):
        forms.append("a test directory")
    return forms


def _node_id_taker(path: str, file: str, takers: Sequence[Check]) -> Check:
    taker = next((c for c in takers if c.named_node_ids and file.endswith(c.named_suffixes)), None)
    if taker is None:
        suffixes = [s for c in takers if c.named_node_ids for s in c.named_suffixes]
        takes = f"only a {_spelled(suffixes)} file takes" if suffixes else "no check takes"
        raise Refused(f"{path}: {takes} a node id; name the file")
    if not (ROOT / file).is_file():
        raise Refused(f"{path}: {file} is not a file")
    return taker


def _directory_taker(path: str, takers: Sequence[Check]) -> Check:
    takes = [(c, c.named_dirs) for c in takers if c.named_dirs]
    names = [found.name for found in (ROOT / path).rglob("*") if found.is_file()]
    taker = next((c for c, is_test in takes if any(map(is_test, names))), None)
    if taker is None:
        raise Refused(f"{path}: holds no {_spelled([c.id for c, _ in takes])} file")
    return taker


def tier_checks(tier: str) -> list[Check]:
    """The checks of *tier*, in declaration order."""
    return [check for check in CHECKS if tier in check.tiers]


def select(
    tier: str,
    paths: Sequence[str] | None,
    named: Sequence[str] = (),
    files: Sequence[str] | None = None,
    reach: Sequence[str] | None = None,
) -> Plan:
    """Resolve *tier* against the changed *paths*, or run it whole when there are none.

    *paths* are the changed files the range still holds, which the runners'
    selections read. *reach* is every path the range changed, deleted and
    moved-away ones included, which every scope reads; it is *paths* when not
    given. *named* are test files added to what the range selects. *files* is
    the repository's file list, read from git when not given.
    """
    routed = _route_named(named)
    if paths is None:
        whole = {check.id: Selection(WHOLE) for check in tier_checks(tier)}
        lanes = tuple(lane.job for lane in LANES) if tier == "full" else ()
        return Plan(tier, whole, lanes, scoped=False)
    if files is None:
        files = repository_files()
    if reach is None:
        reach = paths
    selections: dict[str, Selection] = {}
    for check in tier_checks(tier):
        if check.selector:
            chosen = check.selector(paths, files, SHAPING_SCRIPTS)
        elif check.derive_scope:
            derived = check.derive_scope(files)
            chosen = Selection(WHOLE) if touches(reach, derived) else Selection(NOT_TOUCHED)
        elif touches(reach, check.scope):
            chosen = Selection(WHOLE)
        else:
            chosen = Selection(NOT_TOUCHED)
        extra = routed.get(check.id, [])
        if extra and chosen.state != WHOLE:
            chosen = _paths([*chosen.paths, *extra])
        selections[check.id] = chosen
    lanes = tuple(lane.job for lane in LANES if touches(reach, lane.scope))
    return Plan(tier, selections, lanes, scoped=True)


# ---------------------------------------------------------------------------
# Commands


def _is_exec_flag(flag: str) -> bool:
    return (
        flag in ("-v", "-q")
        or re.fullmatch(r"-n(\d+|auto|logical)", flag) is not None
        or flag.startswith(
            ("--tb=", "--durations=", "--timeout=", "--timeout-method=", "--maxfail=", "--cov")
        )
    )


def validate_exec_flags(flags: Sequence[str]) -> None:
    """Refuse anything that is not an execution flag."""
    index = 0
    while index < len(flags):
        flag = flags[index]
        following = flags[index + 1] if index + 1 < len(flags) else ""
        if flag == "-n" and re.fullmatch(r"\d+|auto|logical", following):
            index += 2
        elif _is_exec_flag(flag):
            index += 1
        else:
            raise Refused(
                f"{flag}: not an execution flag; what a check selects is declared by "
                "its checks module"
            )


def _shaping_scripts(checks: Iterable[Check], modules: Iterable[str] = ()) -> frozenset[str]:
    """The scripts a check runs through and the modules that declare the checks:
    a change to one can change any check's verdict."""
    argvs = [*(c.argv for c in checks), kit.bounded(0, [])]
    named = {
        arg for argv in argvs for arg in argv if arg.startswith("scripts/") and arg.endswith(".py")
    }
    own = (Path(__file__).resolve(), Path(kit.__file__).resolve())
    return frozenset({*named, *modules, *(path.relative_to(ROOT).as_posix() for path in own)})


#: Derived from the declaration, so a check that names a new script makes it shaping.
SHAPING_SCRIPTS = _shaping_scripts(CHECKS, DECLARED.modules)


def report_flags(check: Check, context: Context) -> list[str]:
    """The flags that make *check*'s runner write its report into the run's directory."""
    if not check.report:
        return []
    return [
        flag.replace("{reports}", context.reports).replace("{id}", check.id)
        for flag in REPORT_FLAGS[check.report]
    ]


def _worktree_lines(lines: Iterable[Sequence[str]], base: str) -> list[list[str]]:
    """*lines* with ``{base}`` filled and each ``{untracked}`` line once per untracked path."""
    untracked: list[str] | None = None
    filled: list[list[str]] = []
    for line in lines:
        line = [part.replace("{base}", base) for part in line]
        if "{untracked}" not in line:
            filled.append(line)
            continue
        if untracked is None:
            untracked = kit.git_paths("ls-files", "--others", "--exclude-standard")
        filled += [[path if part == "{untracked}" else part for part in line] for path in untracked]
    return filled


def commands(check: Check, selection: Selection, context: Context) -> list[list[str]]:
    """The effective command lines of *check* under *selection*, without running them.

    A check spelled by its own ``command`` hook is built there. Otherwise the
    line is the check's ``argv`` with the base filled, the selected paths, the
    execution flags when it takes them and the report flags, stopped at its
    bound when it has one.
    """
    flags = list(context.exec_flags.get(check.id, context.exec_flags.get("*", ())))
    base = context.base or "{base}"
    written = report_flags(check, context)
    if check.command:
        return check.command(check, selection, context, flags, written)
    if check.annotate and os.environ.get("GITHUB_ACTIONS"):
        return [[*check.argv, *check.annotate]]
    if context.worktree and check.worktree:
        return [[*line, *written] for line in _worktree_lines(check.worktree, base)]
    if not check.argv:
        return []
    argv = [
        *(part.replace("{base}", base) for part in check.argv),
        *selection.paths,
        *(flags if check.exec_flags else ()),
        *written,
    ]
    return [kit.bounded(check.bound_s, argv)] if check.bound_s else [argv]


# ---------------------------------------------------------------------------
# Running

#: What can happen to a check; nothing else is ever reported.
RAN, FAILED, NOT_RUN, REFUSED = "ran", "failed", "not run", "refused"
STATES = (RAN, FAILED, NOT_RUN, REFUSED)

#: Why a check did not run, each with the words the report uses.
OFF_VENUE, NOT_INSTALLED, NO_BASE, DESELECTED = (
    "off venue",
    "not installed",
    "no base",
    "deselected",
)
REASONS = {
    NOT_TOUCHED: "not touched",
    OFF_VENUE: "not run here",
    NOT_INSTALLED: "not run here",
    NO_BASE: "not run, no base given",
    DESELECTED: "not run, the marker expression deselects every test in these paths",
}
#: The reasons a line names the check after its words, as ``not run here: ID``.
_NAMED_AFTER = (OFF_VENUE, NOT_INSTALLED)

#: The most failing tests a line names; the rest are counted.
FAILURES_SHOWN = 5
#: run_bounded.py's status for a command stopped at its bound.
TIMED_OUT = 124


@dataclass(frozen=True)
class Outcome:
    """What happened to one check: one state, and only the facts that state carries."""

    id: str
    state: str
    #: ``NOT_RUN`` only: a key of ``REASONS``.
    reason: str | None = None
    #: ``DESELECTED`` only: the checks and CI lanes that run the deselected tests.
    hosts: tuple[str, ...] = ()
    #: ``FAILED`` only: the exit status, never 0.
    status: int = 0
    #: ``FAILED`` only: the failing test ids, from the runner's report.
    failures: tuple[str, ...] = ()
    #: ``REFUSED`` only: what to install, as a ``NEED_PROBES`` answer.
    need: str = ""
    #: What the check's install probe said: on ``RAN`` and ``FAILED``, the line
    #: naming the install the check ran against; on ``NOT_INSTALLED``, the line
    #: saying it is absent.
    attestation: str = ""

    def __post_init__(self) -> None:
        problems = []
        if self.state not in STATES:
            problems.append(f"unknown state {self.state!r}")
        if (self.state == NOT_RUN) != (self.reason in REASONS):
            problems.append(f"reason {self.reason!r} on state {self.state!r}")
        if self.hosts and self.reason != DESELECTED:
            problems.append("hosts on a check that was not deselected")
        if (self.reason == NOT_INSTALLED) and not self.attestation:
            problems.append("no probe line on a check whose install is absent")
        if self.attestation and not (self.state in (RAN, FAILED) or self.reason == NOT_INSTALLED):
            problems.append(f"attestation on state {self.state!r}")
        if (self.state == FAILED) != (self.status != 0):
            problems.append(f"status {self.status} on state {self.state!r}")
        if self.failures and self.state != FAILED:
            problems.append(f"failures on state {self.state!r}")
        if (self.state == REFUSED) != bool(self.need):
            problems.append(f"need {self.need!r} on state {self.state!r}")
        if problems:
            raise ValueError(f"{self.id}: {'; '.join(problems)}")


def unmet_needs(
    running: Iterable[tuple[Check, Selection]], context: Context
) -> list[tuple[str, str]]:
    """Each check of *running* with an unmet need and its refusal, probing each need
    once. A probe that raises ``EnvironmentFault`` ends the run there."""
    answers: dict[str, str | None] = {}
    unmet: list[tuple[str, str]] = []
    for check, selection in running:
        selected = check.selected_needs(selection, context) if check.selected_needs else ()
        for need in (*check.needs, *selected):
            if need == "base":
                continue
            if need not in answers:
                answers[need] = NEED_PROBES[need]()
            refusal = answers[need]
            if refusal:
                unmet.append((check.id, refusal))
                break
    return unmet


def _hosts_over(paths: Sequence[str]) -> list[str]:
    """The checks with a derived scope and the CI lanes whose scope holds one of
    *paths*, a file or a directory, or lies beneath one of the directories: what
    runs the tests the sweep deselects. A node id is read as its file."""
    paths = [path.partition("::")[0] for path in paths]
    inside = [candidate for path in paths for candidate in (path, path.rstrip("/") + "/")]
    above = tuple(path.rstrip("/") + "/" for path in paths)

    def hosts(scope: Sequence[str]) -> bool:
        return touches(inside, scope) or any(pattern.startswith(above) for pattern in scope)

    derived = [check for check in CHECKS if check.derive_scope]
    files = repository_files() if derived else []
    checks = [c.id for c in derived if c.derive_scope and hosts(c.derive_scope(files))]
    lanes = [lane.job for lane in LANES if hosts(lane.scope)]
    return list(dict.fromkeys([*checks, *lanes]))


def _pytest_id(case: ElementTree.Element) -> str:
    """The node id of an xunit1 ``testcase``: its file, its class path and its name."""
    path, classname, name = (case.get(key, "") for key in ("file", "classname", "name"))
    if not path:
        return "::".join(part for part in (classname, name) if part)
    if not classname:
        return path
    module = path.removesuffix(".py").replace("/", ".")
    if classname == module:
        inner = ""
    elif classname.startswith(module + "."):
        inner = classname[len(module) + 1 :]
    else:
        inner = classname
    return "::".join(part for part in (path, inner.replace(".", "::"), name) if part)


def _vitest_id(case: ElementTree.Element) -> str:
    """A Vitest ``testcase``'s id: its file, then its name unless that is the file."""
    classname, name = case.get("classname", ""), case.get("name", "")
    return classname if classname == name else f"{classname} > {name}"


def failing_tests(check: Check, reports_dir: str) -> tuple[str, ...]:
    """The failing test ids in the reports *check*'s runner wrote under *reports_dir*.

    A missing, empty or unreadable report names nothing; the exit status stands.
    """
    named = _pytest_id if check.report == "junit-pytest" else _vitest_id
    found: list[str] = []
    for path in sorted(Path(reports_dir, check.id).glob("*.xml")):
        try:
            cases = list(ElementTree.parse(path).iter("testcase"))
        except (ElementTree.ParseError, OSError):
            continue
        found += [
            named(case)
            for case in cases
            if case.find("failure") is not None or case.find("error") is not None
        ]
    return tuple(dict.fromkeys(found))


def _run_check(check: Check, selection: Selection, context: Context) -> Outcome:
    status = check.prepare(context) if check.prepare else CLEAN
    if not check.argv:
        kit.say(f"{check.id}: running")
    elif status == 0:
        cwd = kit.ROOT / check.workdir if check.workdir else None
        for argv in commands(check, selection, context):
            kit.say(f"{check.id}: {shlex.join(argv)}")
            status = kit.run(argv, cwd)
            if status != 0:
                break
    if status == 0 and check.builtin:
        status = check.builtin(context)
    if status == 0:
        return Outcome(check.id, RAN)
    if status == check.empty_exit and selection.state == PATHS:
        # The runner collected the paths and the marker expression left none
        # of their tests: nothing ran, so the check is named and not counted.
        return Outcome(check.id, NOT_RUN, DESELECTED, hosts=tuple(_hosts_over(selection.paths)))
    named = check.report and status != TIMED_OUT
    failures = failing_tests(check, context.reports) if named else ()
    return Outcome(check.id, FAILED, status=status, failures=failures)


def _classify(check: Check, selection: Selection, context: Context) -> Outcome | None:
    """Why *check* will not run here, or ``None`` when it will."""
    if selection.state == NOT_TOUCHED:
        return Outcome(check.id, NOT_RUN, NOT_TOUCHED)
    if "base" in check.needs and not context.base:
        return Outcome(check.id, NOT_RUN, NO_BASE)
    if check.venue and not VENUES[check.venue].probe():
        return Outcome(check.id, NOT_RUN, OFF_VENUE)
    return None


def execute(
    planned: Sequence[tuple[Check, Selection]], context: Context, fix: bool = False
) -> list[Outcome]:
    """Run every planned check, each one whatever the earlier ones did.

    A check that will not run here is classified first, and only the needs of
    the rest are probed. When one is unmet, nothing runs and every check with an
    unmet need is refused.
    """
    try:
        directory = tempfile.TemporaryDirectory(prefix="check-reports-")
    except OSError as error:
        raise EnvironmentFault("free space", str(error)) from error
    with directory as reports:
        return _execute(planned, dataclasses.replace(context, reports=reports), fix)


def _execute(
    planned: Sequence[tuple[Check, Selection]], context: Context, fix: bool
) -> list[Outcome]:
    outcomes: dict[str, Outcome] = {}
    running: list[tuple[Check, Selection]] = []
    for check, selection in planned:
        skipped = _classify(check, selection, context)
        if skipped:
            outcomes[check.id] = skipped
        else:
            running.append((check, selection))
    refused = dict(unmet_needs(running, context))
    installed: dict[str, Install] = {}
    for check, _selection in running:
        install = check.install
        if install is None or check.id in refused:
            continue
        status, said = probe_install(install)
        if status == 0:
            installed[check.id] = install
        elif status == ABSENT_EXIT and not context.named:
            outcomes[check.id] = Outcome(check.id, NOT_RUN, NOT_INSTALLED, attestation=said)
        else:
            refused[check.id] = f"{said}: {install.instruction()}"
    if refused:
        return [
            Outcome(check.id, REFUSED, need=refused[check.id])
            for check, _selection in running
            if check.id in refused
        ]
    if fix:
        for check, _selection in running:
            for argv in check.fix:
                kit.say(f"fix: {shlex.join(argv)}")
                kit.run(argv)
    for check, selection in running:
        if check.id in outcomes:
            continue
        outcome = _run_check(check, selection, context)
        if check.id in installed and outcome.state in (RAN, FAILED):
            line = installed[check.id].attestation(outcome.state == RAN)
            outcome = dataclasses.replace(outcome, attestation=line)
        outcomes[check.id] = outcome
    return [outcomes[check.id] for check, _selection in planned]


def _names(failures: Sequence[str]) -> str:
    """The first ``FAILURES_SHOWN`` failing tests, then a count of the rest."""
    hidden = len(failures) - FAILURES_SHOWN
    return ", ".join([*failures[:FAILURES_SHOWN], *([f"+{hidden}"] if hidden > 0 else [])])


def _fact(outcome: Outcome) -> str:
    """The parenthesised fact of *outcome*'s line, or ``""`` when it has none."""
    by_id = _by_id()
    declared = by_id.get(outcome.id)
    if outcome.state == FAILED:
        if outcome.failures:
            return _names(outcome.failures)
        if outcome.status == TIMED_OUT and declared and declared.bound_s:
            return "still running at its bound"
        return f"exit {outcome.status}"
    if outcome.state == REFUSED:
        return outcome.need
    if outcome.reason == OFF_VENUE and declared and declared.venue:
        return VENUES[declared.venue].words
    if outcome.reason == NOT_INSTALLED:
        return outcome.attestation
    if outcome.reason == DESELECTED and outcome.hosts:
        checks = [host for host in outcome.hosts if host in by_id]
        lanes = [host for host in outcome.hosts if host not in by_id]
        said = [f"run by {', '.join(checks)}"] if checks else []
        said += [f"run on CI: {', '.join(lanes)}"] if lanes else []
        return ", ".join(said)
    return ""


def _repair(outcome: Outcome) -> str:
    """The command that repairs *outcome*, or ``""`` when its check declares no fix."""
    declared = _by_id().get(outcome.id)
    if outcome.state == FAILED and declared and declared.fix:
        return f"fix: {FIX_HINT}"
    return ""


def _line(outcome: Outcome) -> str:
    """The per-check line of *outcome*."""
    detail = "; ".join(part for part in (_fact(outcome), _repair(outcome)) if part)
    suffix = f" ({detail})" if detail else ""
    if outcome.reason in _NAMED_AFTER:
        return f"{REASONS[outcome.reason]}: {outcome.id}{suffix}"
    words = REASONS[outcome.reason] if outcome.reason else outcome.state
    return f"{outcome.id}: {words}{suffix}"


def _refused_parts(outcomes: Sequence[Outcome]) -> list[str]:
    """One ``refused`` part per need, naming every check it refused."""
    by_need: dict[str, list[str]] = {}
    for outcome in outcomes:
        if outcome.state == REFUSED:
            by_need.setdefault(outcome.need, []).append(outcome.id)
    return [f"refused: {', '.join(ids)} ({need})" for need, ids in by_need.items()]


def verdict(outcomes: Sequence[Outcome], lanes: Sequence[str], scoped: bool) -> str:
    """The last line: clean or not, and everything that was not run here."""
    refused = _refused_parts(outcomes)
    if refused:
        return "; ".join(refused)
    failed = []
    for o in outcomes:
        if o.state == FAILED:
            detail = "; ".join(part for part in (_names(o.failures), _repair(o)) if part)
            failed.append(f"{o.id} ({detail})" if detail else o.id)
    parts = [f"not clean: failed: {', '.join(failed)}" if failed else "clean"]
    parts += [o.attestation for o in outcomes if o.state in (RAN, FAILED) and o.attestation]
    named: dict[str, list[str]] = {}
    for reason in (OFF_VENUE, NOT_INSTALLED, NO_BASE, DESELECTED):
        for outcome in outcomes:
            if outcome.reason == reason:
                fact = _fact(outcome)
                said = f"{outcome.id} ({fact})" if fact else outcome.id
                named.setdefault(REASONS[reason], []).append(said)
    parts += [f"{words}: {', '.join(ids)}" for words, ids in named.items()]
    if lanes:
        reason = "this range touches" if scoped else "never run here"
        parts.append(f"run on CI, {reason}: {', '.join(lanes)}")
    return "; ".join(parts)


def report(outcomes: Sequence[Outcome], lanes: Sequence[str], scoped: bool) -> int:
    """Print one line per check and the verdict, and return the exit status.

    A refused run prints to stderr, as every refusal does; any other to stdout.
    """
    refused = any(o.state == REFUSED for o in outcomes)
    stream = sys.stderr if refused else sys.stdout
    for outcome in outcomes:
        print(f"check: {_line(outcome)}", file=stream, flush=True)
    print(f"check: {verdict(outcomes, lanes, scoped)}", file=stream, flush=True)
    if refused:
        return REFUSED_EXIT
    return FAILED_EXIT if any(o.state == FAILED for o in outcomes) else CLEAN


# ---------------------------------------------------------------------------
# Command line


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check.py",
        description="Run the repository's checks from their one declaration.",
    )
    verbs = parser.add_subparsers(dest="verb", required=True)
    for tier in TIERS:
        verb = verbs.add_parser(tier, help=f"run the {tier} tier; tests may follow --")
        verb.add_argument(
            "--changed", metavar="BASE", help="scope the tier to BASE...HEAD; implies --base BASE"
        )
        verb.add_argument(
            "--base",
            metavar="BASE",
            help="arm the checks that judge a range against BASE; the runners stay whole",
        )
        verb.add_argument(
            "--worktree", action="store_true", help="judge the working tree, not HEAD"
        )
        if tier == "quick":
            verb.add_argument(
                "--fix",
                action="store_true",
                help="let the checks that declare a fix rewrite files first",
            )
    run = verbs.add_parser("run", help="run the named checks; execution flags may follow --")
    run.add_argument("ids", nargs="+", metavar="ID")
    run.add_argument("--base", metavar="BASE", help="the base of the checks that judge a range")
    run.add_argument("--shard", choices=sorted({s for c in CHECKS for s in c.shards}))
    listing = verbs.add_parser("list", help="print the declaration")
    listing.add_argument("--tier", choices=TIERS)
    listing.add_argument(
        "--changed", metavar="BASE", help="also name the checks and lanes BASE...HEAD touches"
    )
    listing.add_argument("--json", action="store_true")
    return parser


def _split(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """The arguments before the first ``--`` and those after it."""
    argv = list(argv)
    if "--" in argv:
        cut = argv.index("--")
        return argv[:cut], argv[cut + 1 :]
    return argv, []


@dataclass(frozen=True)
class _Request:
    planned: list[tuple[Check, Selection]]
    context: Context
    lanes: tuple[str, ...] = ()
    scoped: bool = False
    fix: bool = False


def _tier_flags(tier: str) -> dict[str, tuple[str, ...]]:
    """The execution flags *tier* passes, per check."""
    return {check.id: check.tier_flags[tier] for check in CHECKS if tier in check.tier_flags}


def _request(argv: Sequence[str]) -> _Request:
    """Parse a ``quick``, ``full`` or ``run`` command line into what it will run."""
    head, tail = _split(argv)
    args = _parser().parse_args(head)
    by_id = _by_id()
    if args.verb == "run":
        unknown = [name for name in args.ids if name not in by_id]
        if unknown:
            raise Refused(f"unknown check: {', '.join(unknown)}; see check.py list")
        validate_exec_flags(tail)
        chosen = [by_id[name] for name in args.ids]
        if tail and not any(check.exec_flags for check in chosen):
            takers = [check.id for check in CHECKS if check.exec_flags]
            raise Refused(f"execution flags apply to {', '.join(takers)} only")
        unbased = [c.id for c in chosen if "base" in c.needs and not args.base]
        if unbased:
            raise Refused(f"{', '.join(unbased)}: needs --base BASE")
        context = Context("full", args.base, args.shard, {"*": tuple(tail)}, named=True)
        return _Request([(check, Selection(WHOLE)) for check in chosen], context)
    if args.changed and args.base and args.changed != args.base:
        raise Refused("--changed and --base name different bases")
    base = args.changed or args.base
    if args.worktree and not args.changed:
        raise Refused("--worktree needs --changed BASE")
    paths = files = reach = None
    if args.changed:
        paths = touched(args.changed, args.worktree)
        reach = touched(args.changed, args.worktree, scope=True)
        files = repository_files(args.worktree)
    plan = select(args.verb, paths, tail, files, reach)
    context = Context(args.verb, base, None, _tier_flags(args.verb), worktree=args.worktree)
    planned = [(by_id[name], selection) for name, selection in plan.selections.items()]
    return _Request(planned, context, plan.lanes, plan.scoped, getattr(args, "fix", False))


def resolve(argv: Sequence[str]) -> list[list[str]]:
    """The effective commands of a ``quick``, ``full`` or ``run`` line, without running them.

    A check the range does not touch contributes no line, and ``list`` runs nothing.
    """
    if _parser().parse_args(_split(argv)[0]).verb == "list":
        return []
    request = _request(argv)
    return [
        argv
        for check, selection in request.planned
        if selection.state != NOT_TOUCHED
        for argv in commands(check, selection, request.context)
    ]


def declaration(tier: str | None = None) -> dict[str, object]:
    """The declaration as data; a derived scope is the one computed for this tree."""
    files = repository_files() if any(check.derive_scope for check in CHECKS) else []

    def plain(item: Check | Lane) -> dict[str, object]:
        # A hook is behaviour, not declaration, and is not data either.
        data = {
            f.name: getattr(item, f.name)
            for f in dataclasses.fields(item)
            if not f.metadata.get("hook")
        }
        if "tiers" in data:
            data["tiers"] = sorted(data["tiers"])
        if isinstance(item, Check) and item.derive_scope:
            data["scope"] = list(item.derive_scope(files))
        return data

    return {
        "tiers": list(TIERS),
        "checks": [plain(check) for check in CHECKS if tier is None or tier in check.tiers],
        **DECLARED.listed,
        "lanes": [plain(lane) for lane in LANES],
    }


def touched_by(base: str, tier: str = "full") -> list[str]:
    """The ids of the checks of *tier* and of the CI lanes the range ``BASE...HEAD`` touches."""
    files = repository_files()
    plan = select(tier, touched(base), (), files, touched(base, scope=True))
    checks = [name for name, chosen in plan.selections.items() if chosen.state != NOT_TOUCHED]
    return [*checks, *plan.lanes]


def _list(tier: str | None, as_json: bool, changed: str | None = None) -> int:
    hit = touched_by(changed, tier or "full") if changed else None
    if as_json:
        data = declaration(tier)
        if hit is not None:
            data["touched"] = hit
        print(json.dumps(data, indent=2))
        return CLEAN
    for check in CHECKS:
        if tier is None or tier in check.tiers:
            owner = DECLARED.owners.get(check.id, FRAMEWORK_MODULE)
            spelling = shlex.join(check.argv) if check.argv else f"(in {owner})"
            tiers = ",".join(t for t in TIERS if t in check.tiers)
            print(f"{check.id:<20}{check.kind:<11}{tiers:<12}{spelling}")
    if hit is not None:
        print(f"touched: {', '.join(hit)}")
    return CLEAN


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the exit status."""
    argv = sys.argv[1:] if argv is None else list(argv)
    head, _tail = _split(argv)
    args = _parser().parse_args(head)
    try:
        if args.verb == "list":
            if args.changed:
                require_base(args.changed)
            return _list(args.tier, args.json, args.changed)
        base = args.base if args.verb == "run" else (args.changed or args.base)
        if base:
            require_base(base)
        request = _request(argv)
        outcomes = execute(request.planned, request.context, request.fix)
    except Refused as refusal:
        print(f"check: refused: {refusal}", file=sys.stderr, flush=True)
        return REFUSED_EXIT
    except EnvironmentFault as fault:
        print(f"check: environment: {fault}", file=sys.stderr, flush=True)
        return ENVIRONMENT_EXIT
    return report(outcomes, request.lanes, request.scoped)


if __name__ == "__main__":
    sys.exit(main())
